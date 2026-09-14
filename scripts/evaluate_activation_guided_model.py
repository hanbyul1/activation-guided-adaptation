#!/usr/bin/env python3
"""
evaluate_activation_guided_model.py

Re-evaluate the permanently merged activation-guided models for the three
independent SWE-bench target properties:

    1. complexity
    2. coupling
    3. modularity

The script corresponds to the paper's "Re-evaluation" stage.  It compares the
original pretrained model with each standalone merged model on the held-out
SWE-bench test partition and produces three complementary forms of evidence:

1. Output-level evidence
   Generate a unified-diff repair for every test instance and apply the same
   deterministic structural labeling procedure used by
   prepare_dataset_and_labels.py.  The property-positive rate of the merged
   model is compared with the pretrained baseline.

2. Activation-level evidence
   Extract hidden representations at the property-specific layers selected by
   discover_activation_centroids.py.  Compare pooled activations with the
   fixed target-property centroids using Euclidean distance and cosine
   similarity.  No runtime activation steering is applied.

3. Original-task preservation evidence
   Compute conditional negative log-likelihood and perplexity of the reference
   SWE-bench patch given the benchmark prompt.  This is a lightweight language-
   modeling proxy, not the official repository-execution SWE-bench resolved
   rate.  Official SWE-bench evaluation can be performed separately from the
   generated prediction files saved by this script.

Permanent adaptation is supported when, after adapter merging and without PEFT
or runtime activation intervention, the merged property-specific model:

* increases the corresponding target-property-positive output rate;
* produces activations closer to the fixed target centroid; and
* preserves the reference-patch language-modeling metric within a user-defined
  tolerance.

Expected inputs
---------------
outputs/prepared_swebench/prepared_dataset.pt
outputs/activation_centroids/property_centroids.pt
outputs/lora_complexity/merged_model/
outputs/lora_coupling/merged_model/
outputs/lora_modularity/merged_model/
prepare_dataset_and_labels.py

Example
-------
python evaluate_activation_guided_model.py \\
    --prepared-cache outputs/prepared_swebench/prepared_dataset.pt \\
    --centroid-file outputs/activation_centroids/property_centroids.pt \\
    --labeling-script prepare_dataset_and_labels.py \\
    --complexity-model outputs/lora_complexity/merged_model \\
    --coupling-model outputs/lora_coupling/merged_model \\
    --modularity-model outputs/lora_modularity/merged_model \\
    --output-dir outputs/activation_guided_evaluation \\
    --device cuda --dtype float16

Outputs
-------
<output-dir>/
    evaluation_summary.json
    permanent_adaptation_summary.csv
    activation_level_results.csv
    output_level_results.csv
    capability_preservation.csv
    per_instance_results.jsonl
    predictions/
        pretrained.jsonl
        complexity_merged.jsonl
        coupling_merged.jsonl
        modularity_merged.jsonl
    evaluation_config.json

Notes
-----
* Each target property is evaluated against its own independently adapted and
  merged model.
* The baseline model is evaluated once and reused for all three comparisons.
* Generated patches may be unparsable.  Parse coverage is therefore reported
  and must be interpreted together with the positive-output rate.
* The script intentionally does not load PEFT and does not modify activations
  during inference.  A merged model that loads through AutoModelForCausalLM is
  evaluated as a standalone causal language model.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.util
import json
import logging
import math
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

LOGGER = logging.getLogger("evaluate_activation_guided_model")
PROPERTIES: Tuple[str, ...] = ("complexity", "coupling", "modularity")


@dataclass
class ActivationAggregate:
    count: int = 0
    distance_sum: float = 0.0
    cosine_sum: float = 0.0
    norm_sum: float = 0.0

    def update(
        self,
        activations: torch.Tensor,
        centroid: torch.Tensor,
    ) -> None:
        activations = activations.detach().to(torch.float32)
        centroid = centroid.detach().to(
            device=activations.device, dtype=torch.float32
        )
        distances = torch.linalg.vector_norm(activations - centroid, dim=-1)
        cosine = F.cosine_similarity(
            activations,
            centroid.unsqueeze(0).expand_as(activations),
            dim=-1,
            eps=1e-8,
        )
        norms = torch.linalg.vector_norm(activations, dim=-1)
        self.count += int(activations.size(0))
        self.distance_sum += float(distances.sum().item())
        self.cosine_sum += float(cosine.sum().item())
        self.norm_sum += float(norms.sum().item())

    def means(self) -> Dict[str, Optional[float]]:
        if self.count == 0:
            return {
                "count": 0,
                "centroid_distance": None,
                "cosine_similarity": None,
                "activation_norm": None,
            }
        return {
            "count": self.count,
            "centroid_distance": self.distance_sum / self.count,
            "cosine_similarity": self.cosine_sum / self.count,
            "activation_norm": self.norm_sum / self.count,
        }


@dataclass
class CapabilityAggregate:
    token_count: int = 0
    nll_sum: float = 0.0

    def update(self, nll_sum: float, token_count: int) -> None:
        self.nll_sum += float(nll_sum)
        self.token_count += int(token_count)

    def metrics(self) -> Dict[str, Optional[float]]:
        if self.token_count == 0:
            return {
                "token_count": 0,
                "mean_reference_patch_nll": None,
                "reference_patch_perplexity": None,
            }
        mean_nll = self.nll_sum / self.token_count
        return {
            "token_count": self.token_count,
            "mean_reference_patch_nll": mean_nll,
            "reference_patch_perplexity": math.exp(min(mean_nll, 50.0)),
        }


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(name: str) -> torch.device:
    name = name.lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "mps" and (
        not getattr(torch.backends, "mps", None)
        or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable.")
    return device


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    table = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = table[name]
    if device.type == "cpu" and dtype == torch.float16:
        LOGGER.warning("float16 on CPU is unsuitable; using float32.")
        return torch.float32
    if device.type == "mps" and dtype == torch.bfloat16:
        LOGGER.warning("bfloat16 on MPS is not consistently supported; using float32.")
        return torch.float32
    return dtype


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled or device.type not in {"cuda", "cpu"}:
        return nullcontext()
    if device.type == "cpu" and dtype != torch.bfloat16:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def load_prepared_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    cache = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "instance_ids", "input_ids", "attention_masks", "target_ids",
        "target_attention_masks", "labels", "splits",
    }
    missing = required.difference(cache)
    if missing:
        raise KeyError("Prepared cache is missing: " + ", ".join(sorted(missing)))
    count = len(cache["instance_ids"])
    for key in ("input_ids", "attention_masks", "target_ids", "target_attention_masks"):
        if len(cache[key]) != count:
            raise ValueError(f"Prepared cache length mismatch for {key}.")
    for property_name in PROPERTIES:
        if property_name not in cache["labels"]:
            raise KeyError(f"Missing labels for {property_name}.")
    return cache


def load_centroids(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"centroids", "selected_layers"}
    missing = required.difference(payload)
    if missing:
        raise KeyError("Centroid payload is missing: " + ", ".join(sorted(missing)))
    for property_name in PROPERTIES:
        if property_name not in payload["centroids"]:
            raise KeyError(f"Missing centroids for {property_name}.")
        if property_name not in payload["selected_layers"]:
            raise KeyError(f"Missing selected layers for {property_name}.")
    return payload


def split_indices(cache: Mapping[str, Any], split_name: str) -> List[int]:
    if split_name not in cache["splits"]:
        raise KeyError(f"Prepared cache has no split '{split_name}'.")
    values = cache["splits"][split_name]
    if isinstance(values, torch.Tensor):
        return [int(item) for item in values.tolist()]
    return [int(item) for item in values]


import sys

def import_label_patch(script_path: Path):

    spec = importlib.util.spec_from_file_location(
        "activation_guided_labeling",
        script_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import labeling script: {script_path}")

    module = importlib.util.module_from_spec(spec)

    # ★ 반드시 추가
    sys.modules[spec.name] = module

    spec.loader.exec_module(module)

    if not hasattr(module, "label_patch"):
        raise AttributeError(f"{script_path} does not define label_patch().")

    return module.label_patch


def load_tokenizer(model_path: str, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_model(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    low_cpu_memory_usage: bool,
):
    LOGGER.info("Loading model: %s", model_path)
    kwargs: Dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": low_cpu_memory_usage,
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.to(device)
    model.eval()
    model.config.use_cache = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def infer_layer_count(model: torch.nn.Module) -> int:
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(model.config, key, None)
        if value is not None:
            return int(value)
    raise RuntimeError("Cannot infer transformer layer count.")


def pool_hidden_state(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    if hidden_state.ndim != 3:
        raise ValueError(f"Expected [batch, sequence, hidden], got {hidden_state.shape}.")
    if pooling == "last-token":
        positions = attention_mask.sum(dim=1).clamp(min=1) - 1
        rows = torch.arange(hidden_state.size(0), device=hidden_state.device)
        return hidden_state[rows, positions]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
        return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    if pooling == "first-token":
        return hidden_state[:, 0, :]
    raise ValueError(f"Unsupported pooling method: {pooling}")


def make_prompt_batch(
    cache: Mapping[str, Any],
    indices: Sequence[int],
    pad_token_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ids = [cache["input_ids"][index].to(torch.long) for index in indices]
    masks = [cache["attention_masks"][index].to(torch.long) for index in indices]
    input_ids = pad_sequence(ids, batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence(masks, batch_first=True, padding_value=0)
    return input_ids.to(device), attention_mask.to(device)


def conditional_patch_nll(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    target_ids: torch.Tensor,
    max_sequence_length: int,
    device: torch.device,
    mixed_precision: bool,
    dtype: torch.dtype,
) -> Tuple[float, int]:
    """Return summed target-token NLL and evaluated target-token count."""
    prompt = prompt_ids.to(torch.long)
    target = target_ids.to(torch.long)
    if target.numel() == 0:
        return 0.0, 0

    # Reserve at least one prompt token so target likelihood is conditional.
    if prompt.numel() + target.numel() > max_sequence_length:
        target = target[: max(1, max_sequence_length - 1)]
        allowed_prompt = max_sequence_length - target.numel()
        prompt = prompt[-allowed_prompt:]

    sequence = torch.cat((prompt, target), dim=0).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(sequence, device=device)
    labels = sequence.clone()
    labels[:, : prompt.numel()] = -100

    with torch.inference_mode(), autocast_context(device, mixed_precision, dtype):
        output = model(
            input_ids=sequence,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
    token_count = int(target.numel())
    return float(output.loss.detach().to(torch.float32).item()) * token_count, token_count


def labels_from_generated_patch(label_patch, patch: str) -> Dict[str, Any]:
    labels, fragment_count, parseable_pairs = label_patch(patch)
    return {
        "complexity": int(labels.complexity.label),
        "coupling": int(labels.coupling.label),
        "modularity": int(labels.modularity.label),
        "complexity_criteria": list(labels.complexity.criteria_met),
        "coupling_criteria": list(labels.coupling.criteria_met),
        "modularity_criteria": list(labels.modularity.criteria_met),
        "fragment_count": int(fragment_count),
        "parseable_fragment_pairs": int(parseable_pairs),
    }


def evaluate_one_model(
    *,
    model_name_for_results: str,
    model: torch.nn.Module,
    tokenizer: Any,
    cache: Mapping[str, Any],
    test_indices: Sequence[int],
    centroids_payload: Mapping[str, Any],
    properties_to_measure: Sequence[str],
    label_patch: Any,
    output_dir: Path,
    batch_size: int,
    max_new_tokens: int,
    max_sequence_length: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    pooling: str,
    device: torch.device,
    dtype: torch.dtype,
    mixed_precision: bool,
    seed: int,
    log_every: int,
) -> Dict[str, Any]:
    layer_count = infer_layer_count(model)
    union_layers = sorted({
        int(layer)
        for property_name in properties_to_measure
        for layer in centroids_payload["selected_layers"][property_name]
    })
    invalid = [layer for layer in union_layers if layer < 0 or layer >= layer_count]
    if invalid:
        raise IndexError(f"Selected layers outside model range: {invalid}")

    activation_all: Dict[str, Dict[int, ActivationAggregate]] = {}
    activation_positive: Dict[str, Dict[int, ActivationAggregate]] = {}
    for property_name in properties_to_measure:
        layers = [int(x) for x in centroids_payload["selected_layers"][property_name]]
        activation_all[property_name] = {
            layer: ActivationAggregate() for layer in layers
        }
        activation_positive[property_name] = {
            layer: ActivationAggregate() for layer in layers
        }

    output_positive_counts = {property_name: 0 for property_name in PROPERTIES}
    output_parseable = 0
    output_fragment_count = 0
    capability = CapabilityAggregate()
    rows: List[Dict[str, Any]] = []
    prediction_rows: List[Dict[str, Any]] = []

    generator = None
    if device.type == "cuda":
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator()
    generator.manual_seed(seed)

    for start in range(0, len(test_indices), batch_size):
        batch_indices = list(test_indices[start : start + batch_size])
        input_ids, attention_mask = make_prompt_batch(
            cache, batch_indices, tokenizer.pad_token_id, device
        )

        # Activation extraction is performed on the prompt, matching discovery.
        with torch.inference_mode(), autocast_context(device, mixed_precision, dtype):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states.")
        # hidden_states[0] is the embedding output; block L is hidden_states[L+1].
        for property_name in properties_to_measure:
            property_labels = cache["labels"][property_name][
                torch.tensor(batch_indices, dtype=torch.long)
            ].to(torch.bool)
            for layer in centroids_payload["selected_layers"][property_name]:
                layer = int(layer)
                pooled = pool_hidden_state(
                    hidden_states[layer + 1], attention_mask, pooling
                )
                centroid = centroids_payload["centroids"][property_name][layer]
                activation_all[property_name][layer].update(pooled, centroid)
                if bool(property_labels.any()):
                    selected = pooled[property_labels.to(device)]
                    activation_positive[property_name][layer].update(
                        selected, centroid
                    )
        del outputs, hidden_states

        generation_kwargs: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "do_sample": do_sample,
            "use_cache": True,
        }
        if do_sample:
            generation_kwargs.update({
                "temperature": temperature,
                "top_p": top_p,
                "generator": generator,
            })
        with torch.inference_mode(), autocast_context(device, mixed_precision, dtype):
            generated = model.generate(**generation_kwargs)

        prompt_width = input_ids.size(1)
        continuations = generated[:, prompt_width:]
        decoded = tokenizer.batch_decode(
            continuations, skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for local_index, source_index in enumerate(batch_indices):
            instance_id = str(cache["instance_ids"][source_index])
            patch = decoded[local_index].strip()
            generated_labels = labels_from_generated_patch(label_patch, patch)
            if generated_labels["parseable_fragment_pairs"] > 0:
                output_parseable += 1
            output_fragment_count += generated_labels["fragment_count"]
            for property_name in PROPERTIES:
                output_positive_counts[property_name] += generated_labels[property_name]

            nll_sum, token_count = conditional_patch_nll(
                model=model,
                prompt_ids=cache["input_ids"][source_index],
                target_ids=cache["target_ids"][source_index],
                max_sequence_length=max_sequence_length,
                device=device,
                mixed_precision=mixed_precision,
                dtype=dtype,
            )
            capability.update(nll_sum, token_count)

            row = {
                "model": model_name_for_results,
                "source_index": int(source_index),
                "instance_id": instance_id,
                "generated_complexity_label": generated_labels["complexity"],
                "generated_coupling_label": generated_labels["coupling"],
                "generated_modularity_label": generated_labels["modularity"],
                "generated_fragment_count": generated_labels["fragment_count"],
                "generated_parseable_fragment_pairs": generated_labels[
                    "parseable_fragment_pairs"
                ],
                "reference_patch_nll_sum": nll_sum,
                "reference_patch_token_count": token_count,
            }
            rows.append(row)
            prediction_rows.append({
                **row,
                "generated_patch": patch,
                "complexity_criteria": generated_labels["complexity_criteria"],
                "coupling_criteria": generated_labels["coupling_criteria"],
                "modularity_criteria": generated_labels["modularity_criteria"],
            })

        processed = min(start + batch_size, len(test_indices))
        if processed % log_every == 0 or processed == len(test_indices):
            LOGGER.info(
                "%s: evaluated %d/%d instances",
                model_name_for_results, processed, len(test_indices),
            )

    prediction_path = output_dir / "predictions" / f"{model_name_for_results}.jsonl"
    write_jsonl(prediction_path, prediction_rows)

    activation_result: Dict[str, Any] = {}
    for property_name in properties_to_measure:
        activation_result[property_name] = {"all": {}, "positive_reference": {}}
        for layer in centroids_payload["selected_layers"][property_name]:
            layer = int(layer)
            activation_result[property_name]["all"][str(layer)] = (
                activation_all[property_name][layer].means()
            )
            activation_result[property_name]["positive_reference"][str(layer)] = (
                activation_positive[property_name][layer].means()
            )

    instance_count = len(test_indices)
    output_metrics = {
        "instance_count": instance_count,
        "parseable_output_count": output_parseable,
        "parseable_output_rate": output_parseable / instance_count,
        "mean_generated_fragment_count": output_fragment_count / instance_count,
        "positive_counts": output_positive_counts,
        "positive_rates": {
            property_name: output_positive_counts[property_name] / instance_count
            for property_name in PROPERTIES
        },
    }
    return {
        "model": model_name_for_results,
        "output_metrics": output_metrics,
        "activation_metrics": activation_result,
        "capability_metrics": capability.metrics(),
        "per_instance_rows": rows,
        "prediction_file": str(prediction_path),
    }


def average_layer_metric(
    model_result: Mapping[str, Any],
    property_name: str,
    subset: str,
    metric: str,
) -> Optional[float]:
    values = []
    layers = model_result["activation_metrics"][property_name][subset]
    for layer_metrics in layers.values():
        value = layer_metrics[metric]
        if value is not None:
            values.append(float(value))
    return sum(values) / len(values) if values else None


def relative_change(new: Optional[float], old: Optional[float]) -> Optional[float]:
    if new is None or old is None or old == 0:
        return None
    return (new - old) / abs(old)


def build_summaries(
    baseline: Mapping[str, Any],
    adapted: Mapping[str, Mapping[str, Any]],
    capability_tolerance: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    output_rows: List[Dict[str, Any]] = []
    activation_rows: List[Dict[str, Any]] = []
    capability_rows: List[Dict[str, Any]] = []
    permanent_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}

    baseline_nll = baseline["capability_metrics"]["mean_reference_patch_nll"]

    for property_name in PROPERTIES:
        adapted_result = adapted[property_name]
        baseline_rate = baseline["output_metrics"]["positive_rates"][property_name]
        adapted_rate = adapted_result["output_metrics"]["positive_rates"][property_name]
        output_delta = adapted_rate - baseline_rate
        output_improved = output_delta > 0

        output_rows.extend([
            {
                "property": property_name,
                "model": "pretrained",
                "positive_count": baseline["output_metrics"]["positive_counts"][property_name],
                "instance_count": baseline["output_metrics"]["instance_count"],
                "positive_rate": baseline_rate,
                "parseable_output_rate": baseline["output_metrics"]["parseable_output_rate"],
                "change_from_pretrained": 0.0,
            },
            {
                "property": property_name,
                "model": f"{property_name}_merged",
                "positive_count": adapted_result["output_metrics"]["positive_counts"][property_name],
                "instance_count": adapted_result["output_metrics"]["instance_count"],
                "positive_rate": adapted_rate,
                "parseable_output_rate": adapted_result["output_metrics"]["parseable_output_rate"],
                "change_from_pretrained": output_delta,
            },
        ])

        for subset in ("all", "positive_reference"):
            for layer in adapted_result["activation_metrics"][property_name][subset]:
                base_layer = baseline["activation_metrics"][property_name][subset][layer]
                adapted_layer = adapted_result["activation_metrics"][property_name][subset][layer]
                activation_rows.extend([
                    {
                        "property": property_name,
                        "subset": subset,
                        "layer": int(layer),
                        "model": "pretrained",
                        "count": base_layer["count"],
                        "centroid_distance": base_layer["centroid_distance"],
                        "cosine_similarity": base_layer["cosine_similarity"],
                        "activation_norm": base_layer["activation_norm"],
                    },
                    {
                        "property": property_name,
                        "subset": subset,
                        "layer": int(layer),
                        "model": f"{property_name}_merged",
                        "count": adapted_layer["count"],
                        "centroid_distance": adapted_layer["centroid_distance"],
                        "cosine_similarity": adapted_layer["cosine_similarity"],
                        "activation_norm": adapted_layer["activation_norm"],
                    },
                ])

        base_distance = average_layer_metric(
            baseline, property_name, "all", "centroid_distance"
        )
        adapted_distance = average_layer_metric(
            adapted_result, property_name, "all", "centroid_distance"
        )
        base_cosine = average_layer_metric(
            baseline, property_name, "all", "cosine_similarity"
        )
        adapted_cosine = average_layer_metric(
            adapted_result, property_name, "all", "cosine_similarity"
        )
        activation_improved = bool(
            base_distance is not None
            and adapted_distance is not None
            and base_cosine is not None
            and adapted_cosine is not None
            and adapted_distance < base_distance
            and adapted_cosine > base_cosine
        )

        adapted_nll = adapted_result["capability_metrics"]["mean_reference_patch_nll"]
        nll_change = relative_change(adapted_nll, baseline_nll)
        capability_preserved = bool(
            nll_change is not None and nll_change <= capability_tolerance
        )
        capability_rows.extend([
            {
                "property": property_name,
                "model": "pretrained",
                **baseline["capability_metrics"],
                "relative_nll_change": 0.0,
            },
            {
                "property": property_name,
                "model": f"{property_name}_merged",
                **adapted_result["capability_metrics"],
                "relative_nll_change": nll_change,
            },
        ])

        permanent_supported = bool(
            output_improved and activation_improved and capability_preserved
        )
        permanent_row = {
            "property": property_name,
            "output_positive_rate_pretrained": baseline_rate,
            "output_positive_rate_merged": adapted_rate,
            "output_rate_change": output_delta,
            "output_improved": output_improved,
            "centroid_distance_pretrained": base_distance,
            "centroid_distance_merged": adapted_distance,
            "cosine_similarity_pretrained": base_cosine,
            "cosine_similarity_merged": adapted_cosine,
            "activation_improved": activation_improved,
            "reference_patch_nll_pretrained": baseline_nll,
            "reference_patch_nll_merged": adapted_nll,
            "relative_nll_change": nll_change,
            "capability_tolerance": capability_tolerance,
            "capability_preserved": capability_preserved,
            "permanent_adaptation_supported": permanent_supported,
        }
        permanent_rows.append(permanent_row)
        summary[property_name] = permanent_row

    summary["interpretation"] = {
        "structural_condition": (
            "Each evaluated adapted model is loaded directly from its merged-model "
            "directory through AutoModelForCausalLM; PEFT and runtime activation "
            "intervention are not used."
        ),
        "decision_rule": (
            "Permanent adaptation is supported for a property only when its merged "
            "model increases the corresponding generated-output positive rate, "
            "decreases mean centroid distance while increasing mean cosine "
            "similarity, and keeps reference-patch NLL degradation within the "
            "configured tolerance."
        ),
        "capability_metric_limitation": (
            "Reference-patch NLL/perplexity is a lightweight preservation proxy. "
            "It does not replace official container-based SWE-bench resolved-rate "
            "evaluation."
        ),
    }
    return output_rows, activation_rows, capability_rows, permanent_rows, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-evaluate merged activation-guided models for permanent adaptation."
    )
    parser.add_argument(
        "--prepared-cache", type=Path,
        default=Path("outputs/prepared_swebench/prepared_dataset.pt"),
    )
    parser.add_argument(
        "--centroid-file", type=Path,
        default=Path("outputs/activation_centroids/property_centroids.pt"),
    )
    parser.add_argument(
        "--labeling-script", type=Path,
        default=Path("prepare_dataset_and_labels.py"),
        help="Script defining the original label_patch() function.",
    )
    parser.add_argument("--base-model", default="bigcode/starcoderbase-1b")
    parser.add_argument(
        "--complexity-model", type=Path,
        default=Path("outputs/lora_complexity_v4_1536/merged_model"),
    )
    parser.add_argument(
        "--coupling-model", type=Path,
        default=Path("outputs/lora_coupling_v4_1536/merged_model"),
    )
    parser.add_argument(
        "--modularity-model", type=Path,
        default=Path("outputs/lora_modularity_v4_1536/merged_model"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/activation_guided_evaluation"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-sequence-length", type=int, default=1536)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--capability-tolerance", type=float, default=0.05,
        help="Maximum allowed relative increase in reference-patch NLL.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-low-cpu-memory", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configure_logging(args.verbose)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be at least 1.")
    if not 0.0 <= args.capability_tolerance:
        raise ValueError("--capability-tolerance must be nonnegative.")
    if args.do_sample and args.temperature <= 0:
        raise ValueError("--temperature must be positive when sampling.")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {args.output_dir}. Use --overwrite."
            )
        import shutil
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    cache = load_prepared_cache(args.prepared_cache)
    centroids = load_centroids(args.centroid_file)
    label_patch = import_label_patch(args.labeling_script)
    pooling = str(centroids.get("pooling", "last-token"))

    test_indices = split_indices(cache, args.split)
    if args.limit is not None:
        test_indices = test_indices[: args.limit]
    if not test_indices:
        raise RuntimeError(f"Split '{args.split}' is empty.")

    model_paths = {
        "complexity": args.complexity_model,
        "coupling": args.coupling_model,
        "modularity": args.modularity_model,
    }
    for property_name, path in model_paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Merged model for {property_name} does not exist: {path}"
            )

    config = {
        "prepared_cache": str(args.prepared_cache),
        "prepared_cache_sha256": sha256_file(args.prepared_cache),
        "centroid_file": str(args.centroid_file),
        "centroid_file_sha256": sha256_file(args.centroid_file),
        "labeling_script": str(args.labeling_script),
        "labeling_script_sha256": sha256_file(args.labeling_script),
        "base_model": args.base_model,
        "merged_models": {key: str(value) for key, value in model_paths.items()},
        "split": args.split,
        "instance_count": len(test_indices),
        "pooling": pooling,
        "selected_layers": centroids["selected_layers"],
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_sequence_length": args.max_sequence_length,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "capability_tolerance": args.capability_tolerance,
        "runtime_activation_intervention": False,
        "peft_loaded_during_evaluation": False,
        "seed": args.seed,
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(args.output_dir / "evaluation_config.json", config)

    # Use the base-model tokenizer for identical tokenization across models.
    tokenizer = load_tokenizer(args.base_model, args.trust_remote_code)

    baseline_model = load_model(
        args.base_model, device, dtype, args.trust_remote_code,
        not args.disable_low_cpu_memory,
    )
    baseline_result = evaluate_one_model(
        model_name_for_results="pretrained",
        model=baseline_model,
        tokenizer=tokenizer,
        cache=cache,
        test_indices=test_indices,
        centroids_payload=centroids,
        properties_to_measure=PROPERTIES,
        label_patch=label_patch,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_sequence_length=args.max_sequence_length,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        pooling=pooling,
        device=device,
        dtype=dtype,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        log_every=args.log_every,
    )
    del baseline_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    adapted_results: Dict[str, Dict[str, Any]] = {}
    all_instance_rows = list(baseline_result.pop("per_instance_rows"))
    for property_name in PROPERTIES:
        merged_model = load_model(
            str(model_paths[property_name]), device, dtype,
            args.trust_remote_code, not args.disable_low_cpu_memory,
        )
        result = evaluate_one_model(
            model_name_for_results=f"{property_name}_merged",
            model=merged_model,
            tokenizer=tokenizer,
            cache=cache,
            test_indices=test_indices,
            centroids_payload=centroids,
            properties_to_measure=(property_name,),
            label_patch=label_patch,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            max_sequence_length=args.max_sequence_length,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            pooling=pooling,
            device=device,
            dtype=dtype,
            mixed_precision=args.mixed_precision,
            seed=args.seed,
            log_every=args.log_every,
        )
        all_instance_rows.extend(result.pop("per_instance_rows"))
        adapted_results[property_name] = result
        del merged_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "mps":
            torch.mps.empty_cache()

    (
        output_rows,
        activation_rows,
        capability_rows,
        permanent_rows,
        summary,
    ) = build_summaries(
        baseline_result, adapted_results, args.capability_tolerance
    )

    write_csv(args.output_dir / "output_level_results.csv", output_rows)
    write_csv(args.output_dir / "activation_level_results.csv", activation_rows)
    write_csv(args.output_dir / "capability_preservation.csv", capability_rows)
    write_csv(
        args.output_dir / "permanent_adaptation_summary.csv", permanent_rows
    )
    write_jsonl(args.output_dir / "per_instance_results.jsonl", all_instance_rows)
    write_json(
        args.output_dir / "evaluation_summary.json",
        {
            "summary": summary,
            "baseline": baseline_result,
            "adapted_models": adapted_results,
            "config": config,
        },
    )

    LOGGER.info("Evaluation completed: %s", args.output_dir)
    for row in permanent_rows:
        LOGGER.info(
            "%s | output=%s activation=%s capability=%s permanent=%s",
            row["property"], row["output_improved"],
            row["activation_improved"], row["capability_preserved"],
            row["permanent_adaptation_supported"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
