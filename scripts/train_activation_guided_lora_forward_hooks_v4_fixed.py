#!/usr/bin/env python3
"""
train_activation_guided_lora.py

Train a property-specific LoRA adapter for StarCoderBase using a joint objective:

    L_total = L_language + lambda_activation * L_activation

L_language is the causal language-modeling loss. L_activation is the mean
squared Euclidean distance between forward-hook-captured representations and the
selected target-property centroids created by discover_activation_centroids.py.
Activation guidance is applied only to positively labeled instances. Only the selected transformer blocks are captured through forward hooks.

The pretrained backbone remains frozen. After optimization, the best LoRA
adapter is saved and, unless --skip-merge is supplied, merged into the base
model to produce a standalone permanently adapted model.

Example
-------
python train_activation_guided_lora.py \
    --prepared-cache outputs/prepared_swebench/prepared_dataset.pt \
    --centroid-file outputs/activation_centroids/property_centroids.pt \
    --property complexity \
    --model-name bigcode/starcoderbase-1b \
    --output-dir outputs/lora_complexity \
    --device mps \
    --dtype float32 \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --epochs 3 \
    --learning-rate 2e-4 \
    --activation-weight 1.0
"""

from __future__ import annotations

FORWARD_HOOK_BUILD = "v3-no-hidden-state-output"


import argparse
import csv
import gc
import hashlib
import json
import logging
import math
import random
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

LOGGER = logging.getLogger("train_activation_guided_lora")
PROPERTIES = ("coupling", "complexity", "modularity")


@dataclass
class EpochMetrics:
    epoch: int
    split: str
    instance_count: int
    positive_instance_count: int
    optimization_steps: int
    mean_total_loss: float
    mean_language_loss: float
    mean_activation_loss: float
    mean_selected_layer_distance: float
    learning_rate: float
    elapsed_seconds: float


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {path}. Use --overwrite."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir(parents=True, exist_ok=True)


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


def load_prepared_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    cache = torch.load(path, map_location="cpu", weights_only=False)
    required = {
        "instance_ids",
        "input_ids",
        "attention_masks",
        "target_ids",
        "target_attention_masks",
        "labels",
        "splits",
    }
    missing = required.difference(cache)
    if missing:
        raise KeyError("Prepared cache is missing: " + ", ".join(sorted(missing)))

    count = len(cache["instance_ids"])
    sequence_fields = (
        "input_ids",
        "attention_masks",
        "target_ids",
        "target_attention_masks",
    )
    for field in sequence_fields:
        if len(cache[field]) != count:
            raise ValueError(
                f"Prepared cache field '{field}' does not match instance_ids."
            )
    for property_name in PROPERTIES:
        if property_name not in cache["labels"]:
            raise KeyError(f"Missing labels for {property_name}.")
        if len(cache["labels"][property_name]) != count:
            raise ValueError(f"Label length mismatch for {property_name}.")
    return cache


def load_centroid_payload(
    path: Path,
    property_name: str,
) -> Tuple[List[int], Dict[int, torch.Tensor], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "centroids" not in payload or property_name not in payload["centroids"]:
        raise KeyError(f"No centroids found for property '{property_name}'.")

    centroids = {
        int(layer): tensor.detach().cpu().to(torch.float32)
        for layer, tensor in payload["centroids"][property_name].items()
    }
    selected_layers = [
        int(layer)
        for layer in payload.get("selected_layers", {}).get(
            property_name, sorted(centroids)
        )
    ]
    if not selected_layers:
        raise RuntimeError(f"No selected layers for {property_name}.")
    missing = [layer for layer in selected_layers if layer not in centroids]
    if missing:
        raise KeyError(f"Centroid tensors are missing selected layers: {missing}")

    for layer, centroid in centroids.items():
        if centroid.ndim != 1:
            raise ValueError(f"Centroid at layer {layer} must be one-dimensional.")
        if not torch.isfinite(centroid).all():
            raise FloatingPointError(f"Centroid at layer {layer} contains NaN/Inf.")
    return selected_layers, centroids, payload


def split_indices(cache: Mapping[str, Any], split_name: str) -> List[int]:
    if split_name not in cache["splits"]:
        raise KeyError(f"Prepared cache has no split '{split_name}'.")
    value = cache["splits"][split_name]
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.tolist()]
    return [int(item) for item in value]


def infer_layer_count(model: torch.nn.Module) -> int:
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        value = getattr(model.config, key, None)
        if value is not None:
            return int(value)
    raise RuntimeError("Cannot infer transformer layer count.")


def determine_pad_token_id(
    cache: Mapping[str, Any], model: torch.nn.Module, tokenizer: Optional[Any]
) -> int:
    value = cache.get("tokenizer", {}).get("pad_token_id")
    if value is None and tokenizer is not None:
        value = tokenizer.pad_token_id
    if value is None:
        value = getattr(model.config, "pad_token_id", None)
    if value is None:
        value = getattr(model.config, "eos_token_id", None)
    if value is None:
        raise RuntimeError("Could not determine pad_token_id.")
    return int(value)


def make_batch(
    cache: Mapping[str, Any],
    indices: Sequence[int],
    property_name: str,
    pad_token_id: int,
    device: torch.device,
    max_sequence_length: int,
) -> Dict[str, Any]:
    """Build causal-LM sequences that supervise only the gold patch tokens.

    Each training sequence is ``prompt + target_patch``. The prompt positions are
    masked with ``-100`` in ``language_labels`` so that the language-modeling loss
    teaches prompt-to-patch generation rather than prompt reconstruction.

    ``activation_attention_mask`` marks only prompt tokens. The activation
    centroids were constructed from prompt-only forward passes, so activation
    pooling must remain restricted to the prompt portion even though the LM
    forward pass also contains the target patch.
    """
    combined_ids: List[torch.Tensor] = []
    combined_masks: List[torch.Tensor] = []
    language_labels: List[torch.Tensor] = []
    activation_masks: List[torch.Tensor] = []

    for index in indices:
        prompt_ids = cache["input_ids"][index].to(torch.long)
        prompt_mask = cache["attention_masks"][index].to(torch.long)
        target_ids = cache["target_ids"][index].to(torch.long)
        target_mask = cache["target_attention_masks"][index].to(torch.long)

        # Remove any cached padding before concatenation.
        prompt_ids = prompt_ids[prompt_mask.bool()]
        target_ids = target_ids[target_mask.bool()]

        if prompt_ids.numel() == 0:
            raise ValueError(
                f"Instance {cache['instance_ids'][index]} has an empty prompt."
            )
        if target_ids.numel() == 0:
            raise ValueError(
                f"Instance {cache['instance_ids'][index]} has an empty target patch."
            )

        # Preserve the gold patch whenever truncation is necessary. The prepared
        # cache normally uses prompt_length + target_length <= this limit.
        if target_ids.numel() >= max_sequence_length:
            # Keep at least one prompt token so prompt-only activation pooling
            # remains well-defined and comparable with the stored centroids.
            target_ids = target_ids[: max_sequence_length - 1]
            prompt_ids = prompt_ids[:1]
        else:
            prompt_budget = max_sequence_length - target_ids.numel()
            prompt_ids = prompt_ids[:prompt_budget]

        sequence = torch.cat((prompt_ids, target_ids), dim=0)
        attention = torch.ones_like(sequence, dtype=torch.long)
        labels = torch.cat(
            (
                torch.full_like(prompt_ids, -100, dtype=torch.long),
                target_ids.clone(),
            ),
            dim=0,
        )
        activation_mask = torch.cat(
            (
                torch.ones_like(prompt_ids, dtype=torch.long),
                torch.zeros_like(target_ids, dtype=torch.long),
            ),
            dim=0,
        )

        if not bool((labels != -100).any().item()):
            raise ValueError(
                f"Instance {cache['instance_ids'][index]} has no supervised target tokens."
            )

        combined_ids.append(sequence)
        combined_masks.append(attention)
        language_labels.append(labels)
        activation_masks.append(activation_mask)

    padded_ids = pad_sequence(
        combined_ids, batch_first=True, padding_value=pad_token_id
    )
    padded_masks = pad_sequence(
        combined_masks, batch_first=True, padding_value=0
    )
    padded_labels = pad_sequence(
        language_labels, batch_first=True, padding_value=-100
    )
    padded_activation_masks = pad_sequence(
        activation_masks, batch_first=True, padding_value=0
    )

    property_labels = cache["labels"][property_name][
        torch.tensor(indices, dtype=torch.long)
    ].to(torch.long)

    return {
        "input_ids": padded_ids.to(device),
        "attention_mask": padded_masks.to(device),
        "language_labels": padded_labels.to(device),
        "activation_attention_mask": padded_activation_masks.to(device),
        "property_labels": property_labels.to(device),
        "instance_ids": [cache["instance_ids"][index] for index in indices],
    }


def pool_hidden_state(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str,
) -> torch.Tensor:
    """Pool one transformer-block output without detaching its graph."""
    if hidden_state.ndim != 3:
        raise ValueError(
            f"Expected [batch, sequence, hidden] activation, got "
            f"shape={tuple(hidden_state.shape)}"
        )
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


def _first_tensor(value: Any) -> torch.Tensor:
    """Extract the block hidden-state tensor from common HF block outputs."""
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    if hasattr(value, "last_hidden_state") and torch.is_tensor(value.last_hidden_state):
        return value.last_hidden_state
    raise TypeError(
        "The hooked transformer block returned an unsupported output type: "
        f"{type(value)!r}"
    )


def resolve_transformer_blocks(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    """
    Resolve the ordered transformer-block container from a base or PEFT model.

    StarCoderBase/GPTBigCode resolves to ``transformer.h``. Additional paths are
    included so the hook implementation remains usable with closely related
    Hugging Face causal language models.
    """
    roots: List[torch.nn.Module] = [model]
    if hasattr(model, "get_base_model"):
        try:
            roots.insert(0, model.get_base_model())
        except Exception:
            pass
    for attribute in ("base_model", "model"):
        candidate = getattr(model, attribute, None)
        if isinstance(candidate, torch.nn.Module):
            roots.append(candidate)

    paths = (
        ("transformer", "h"),       # GPTBigCode / GPT-2 style
        ("model", "layers"),        # LLaMA-like
        ("gpt_neox", "layers"),     # GPT-NeoX-like
        ("transformer", "blocks"),  # some remote-code models
        ("transformer", "layers"),
    )
    checked: List[str] = []
    seen = set()
    for root in roots:
        if id(root) in seen:
            continue
        seen.add(id(root))
        for path in paths:
            value: Any = root
            valid = True
            for part in path:
                checked.append(f"{root.__class__.__name__}." + ".".join(path))
                if not hasattr(value, part):
                    valid = False
                    break
                value = getattr(value, part)
            if valid and isinstance(value, (torch.nn.ModuleList, list, tuple)):
                if len(value) > 0 and all(isinstance(x, torch.nn.Module) for x in value):
                    return value
    raise RuntimeError(
        "Could not locate the model transformer blocks for forward hooks. "
        "Checked standard paths including transformer.h and model.layers."
    )


class SelectedLayerActivationHooks:
    """Capture only selected transformer-block outputs with forward hooks."""

    def __init__(
        self,
        model: torch.nn.Module,
        selected_layers: Sequence[int],
    ) -> None:
        self.blocks = resolve_transformer_blocks(model)
        self.selected_layers = tuple(dict.fromkeys(int(x) for x in selected_layers))
        invalid = [x for x in self.selected_layers if x < 0 or x >= len(self.blocks)]
        if invalid:
            raise IndexError(
                f"Hook layers outside transformer range 0..{len(self.blocks)-1}: "
                f"{invalid}"
            )
        self.activations: Dict[int, torch.Tensor] = {}
        self.handles: List[Any] = []
        self.capture_enabled = True
        self._register()

    def _register(self) -> None:
        for layer in self.selected_layers:
            def hook(
                module: torch.nn.Module,
                inputs: Tuple[Any, ...],
                output: Any,
                layer_index: int = layer,
            ) -> None:
                if self.capture_enabled:
                    # Do not detach: activation guidance must backpropagate through
                    # the selected representation into the LoRA parameters.
                    self.activations[layer_index] = _first_tensor(output)

            self.handles.append(self.blocks[layer].register_forward_hook(hook))

    def begin_forward(self, capture: bool) -> None:
        self.activations.clear()
        self.capture_enabled = bool(capture)

    def require_complete_capture(self) -> Mapping[int, torch.Tensor]:
        missing = [x for x in self.selected_layers if x not in self.activations]
        if missing:
            raise RuntimeError(
                "Forward hooks did not capture all selected layers. Missing: "
                f"{missing}"
            )
        return self.activations

    def clear(self) -> None:
        self.activations.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.activations.clear()

    def __enter__(self) -> "SelectedLayerActivationHooks":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def activation_guidance_loss(
    activations: Mapping[int, torch.Tensor],
    attention_mask: torch.Tensor,
    property_labels: torch.Tensor,
    selected_layers: Sequence[int],
    centroids: Mapping[int, torch.Tensor],
    pooling: str,
    normalize_activations: bool,
    zero_reference: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Compute centroid distance using only hook-captured selected layers."""
    positive_mask = property_labels == 1
    positive_count = int(positive_mask.sum().item())
    if positive_count == 0:
        zero = zero_reference * 0.0
        return zero, zero.detach(), 0

    losses: List[torch.Tensor] = []
    distances: List[torch.Tensor] = []
    for layer in selected_layers:
        if layer not in activations:
            raise KeyError(f"No hook-captured activation for selected layer {layer}.")
        pooled = pool_hidden_state(
            activations[layer], attention_mask, pooling
        )[positive_mask]
        pooled32 = pooled.to(torch.float32)
        centroid = centroids[layer].to(
            device=pooled32.device,
            dtype=torch.float32,
        )

        if centroid.numel() != pooled32.size(-1):
            raise ValueError(
                f"Centroid/activation mismatch at layer {layer}: "
                f"{centroid.numel()} vs {pooled32.size(-1)}"
            )
        if not torch.isfinite(pooled32).all():
            raise FloatingPointError(
                f"Non-finite activation encountered at layer {layer}."
            )

        if normalize_activations:
            pooled32 = F.normalize(pooled32, p=2, dim=-1)
            target = F.normalize(centroid.unsqueeze(0), p=2, dim=-1)
        else:
            target = centroid.unsqueeze(0)

        squared = (pooled32 - target).pow(2).sum(dim=-1)
        losses.append(squared.mean())
        distances.append(torch.sqrt(squared + 1e-12).mean())

    return (
        torch.stack(losses).mean(),
        torch.stack(distances).mean().detach(),
        positive_count,
    )


def discover_lora_target_modules(
    model: torch.nn.Module, user_targets: Optional[str]
) -> List[str]:
    if user_targets:
        targets = [item.strip() for item in user_targets.split(",") if item.strip()]
        if not targets:
            raise ValueError("--target-modules contained no module names.")
        return targets

    preferred = (
        "c_attn", "c_proj", "c_fc", "q_attn", "q_proj", "k_proj",
        "v_proj", "o_proj", "out_proj", "fc_in", "fc_out",
    )
    found = set()
    for name, module in model.named_modules():
        suffix = name.rsplit(".", 1)[-1]
        if suffix not in preferred:
            continue
        class_name = module.__class__.__name__.lower()
        if isinstance(module, torch.nn.Linear) or "conv1d" in class_name:
            found.add(suffix)
    targets = [name for name in preferred if name in found]
    if not targets:
        raise RuntimeError(
            "Could not discover LoRA target modules; use --target-modules."
        )
    LOGGER.info("LoRA target modules: %s", targets)
    return targets


def load_base_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool,
    low_cpu_mem_usage: bool,
) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    model.config.use_cache = False
    model.to(device)
    return model


def load_tokenizer(model_name: str, trust_remote_code: bool) -> Optional[Any]:
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as exc:
        LOGGER.warning("Tokenizer could not be loaded: %s", exc)
        return None


def trainable_parameter_report(model: torch.nn.Module) -> Dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percentage": 100.0 * trainable / max(total, 1),
        "trainable_parameter_names": names,
    }


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=dtype)
    if device.type == "cpu" and dtype == torch.bfloat16:
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return nullcontext()


def minibatches(indices: Sequence[int], batch_size: int) -> Iterable[List[int]]:
    for start in range(0, len(indices), batch_size):
        yield list(indices[start : start + batch_size])


def run_epoch(
    model: torch.nn.Module,
    cache: Mapping[str, Any],
    indices: Sequence[int],
    property_name: str,
    selected_layers: Sequence[int],
    centroids: Mapping[int, torch.Tensor],
    pad_token_id: int,
    device: torch.device,
    batch_size: int,
    max_sequence_length: int,
    pooling: str,
    activation_weight: float,
    normalize_activations: bool,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    gradient_accumulation_steps: int,
    max_grad_norm: float,
    mixed_precision: bool,
    autocast_dtype: torch.dtype,
    shuffle: bool,
    seed: int,
    epoch_number: int,
    log_every: int,
) -> EpochMetrics:
    training = optimizer is not None
    model.train(training)
    ordered = list(indices)
    if shuffle:
        random.Random(seed + epoch_number).shuffle(ordered)

    batches = list(minibatches(ordered, batch_size))
    if training:
        optimizer.zero_grad(set_to_none=True)

    total_sum = language_sum = activation_sum = distance_sum = 0.0
    instance_count = positive_count_total = optimization_steps = 0
    started = time.time()
    grad_context = torch.enable_grad if training else torch.inference_mode

    # Hooks are registered once per epoch, not once per batch. They capture only
    # the selected transformer blocks and are disabled for all-negative batches.
    with SelectedLayerActivationHooks(model, selected_layers) as hooks:
        with grad_context():
            for batch_number, batch_indices in enumerate(batches, start=1):
                batch = make_batch(
                    cache,
                    batch_indices,
                    property_name,
                    pad_token_id,
                    device,
                    max_sequence_length,
                )
                
                has_positive = bool((batch["property_labels"] == 1).any().item())
                hooks.begin_forward(capture=has_positive)

                with autocast_context(device, mixed_precision, autocast_dtype):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["language_labels"],
                        use_cache=False,
                        return_dict=True,
                    )
                    language_loss = outputs.loss
                    if language_loss is None:
                        raise RuntimeError("Model did not return a language-modeling loss.")

                    captured = (
                        hooks.require_complete_capture() if has_positive else {}
                    )
                    activation_loss, mean_distance, positive_count = (
                        activation_guidance_loss(
                            captured,
                            batch["activation_attention_mask"],
                            batch["property_labels"],
                            selected_layers,
                            centroids,
                            pooling,
                            normalize_activations,
                            zero_reference=language_loss,
                        )
                    )
                    total_loss = language_loss + activation_weight * activation_loss

                if not torch.isfinite(total_loss):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch={epoch_number}, "
                        f"batch={batch_number}."
                    )

                if training:
                    (total_loss / gradient_accumulation_steps).backward()
                    should_step = (
                        batch_number % gradient_accumulation_steps == 0
                        or batch_number == len(batches)
                    )
                    if should_step:
                        clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            max_grad_norm,
                        )
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()
                        optimizer.zero_grad(set_to_none=True)
                        optimization_steps += 1

                size = len(batch_indices)
                total_sum += float(total_loss.detach()) * size
                language_sum += float(language_loss.detach()) * size
                activation_sum += float(activation_loss.detach()) * size
                distance_sum += float(mean_distance) * size
                instance_count += size
                positive_count_total += positive_count

                if batch_number % log_every == 0 or batch_number == len(batches):
                    LOGGER.info(
                        "%s epoch=%d batch=%d/%d total=%.6f language=%.6f "
                        "activation=%.6f distance=%.6f",
                        "train" if training else "validation",
                        epoch_number,
                        batch_number,
                        len(batches),
                        total_sum / instance_count,
                        language_sum / instance_count,
                        activation_sum / instance_count,
                        distance_sum / instance_count,
                    )

                # Drop references to the graph before the next forward pass.
                hooks.clear()
                del outputs, total_loss, language_loss, activation_loss, batch
                if device.type == "cuda" and batch_number % 25 == 0:
                    torch.cuda.empty_cache()
                elif device.type == "mps" and batch_number % 25 == 0:
                    torch.mps.empty_cache()

    learning_rate = 0.0
    if optimizer is not None and optimizer.param_groups:
        learning_rate = float(optimizer.param_groups[0]["lr"])

    return EpochMetrics(
        epoch=epoch_number,
        split="train" if training else "validation",
        instance_count=instance_count,
        positive_instance_count=positive_count_total,
        optimization_steps=optimization_steps,
        mean_total_loss=total_sum / max(instance_count, 1),
        mean_language_loss=language_sum / max(instance_count, 1),
        mean_activation_loss=activation_sum / max(instance_count, 1),
        mean_selected_layer_distance=distance_sum / max(instance_count, 1),
        learning_rate=learning_rate,
        elapsed_seconds=time.time() - started,
    )


def save_history(output_dir: Path, history: Sequence[EpochMetrics]) -> None:
    rows = [asdict(item) for item in history]
    write_json(output_dir / "training_history.json", rows)
    if rows:
        with (output_dir / "training_history.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def save_checkpoint(
    model: torch.nn.Module,
    output_dir: Path,
    epoch: int,
    metrics: EpochMetrics,
) -> Path:
    path = output_dir / "checkpoints" / f"epoch_{epoch:03d}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    write_json(path / "validation_metrics.json", asdict(metrics))
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and merge an activation-guided LoRA adapter."
    )
    parser.add_argument(
        "--prepared-cache", type=Path,
        default=Path("outputs/prepared_swebench/prepared_dataset.pt")
    )
    parser.add_argument(
        "--centroid-file", type=Path,
        default=Path("outputs/activation_centroids/property_centroids.pt")
    )
    parser.add_argument("--property", required=True, choices=PROPERTIES)
    parser.add_argument("--model-name", default="bigcode/starcoderbase-1b")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float32"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=2048,
        help=(
            "Maximum length of each prompt-plus-target training sequence. "
            "Gold patch tokens are preserved preferentially during truncation."
        ),
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument(
        "--scheduler", choices=("linear", "cosine", "none"), default="linear"
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--activation-weight", type=float, default=1.0)
    parser.add_argument(
        "--pooling", choices=("last-token", "mean", "first-token"), default=None
    )
    parser.add_argument("--normalize-activations", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default=None)
    parser.add_argument("--modules-to-save", default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-validation", type=int, default=None)
    parser.add_argument("--mixed-precision", action="store_true")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_true",
        help="Enable non-reentrant gradient checkpointing.",
    )
    checkpoint_group.add_argument(
        "--no-gradient-checkpointing",
        dest="gradient_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing, including the MPS default.",
    )
    parser.set_defaults(gradient_checkpointing=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--disable-low-cpu-memory", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Batch and accumulation sizes must be at least one.")
    if args.max_sequence_length < 2:
        raise ValueError("--max-sequence-length must be at least two.")
    if args.epochs < 1 or args.learning_rate <= 0:
        raise ValueError("Epochs and learning rate must be positive.")
    if args.activation_weight < 0:
        raise ValueError("--activation-weight cannot be negative.")
    if args.lora_rank < 1 or args.lora_alpha < 1:
        raise ValueError("LoRA rank and alpha must be positive.")
    if not 0 <= args.lora_dropout < 1:
        raise ValueError("--lora-dropout must be in [0, 1).")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("--warmup-ratio must be in [0, 1).")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    validate_args(args)

    if args.output_dir is None:
        args.output_dir = Path("outputs") / f"lora_{args.property}"
    ensure_output_directory(args.output_dir, args.overwrite)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    gradient_checkpointing = (
        device.type == "mps"
        if args.gradient_checkpointing is None
        else bool(args.gradient_checkpointing)
    )
    if gradient_checkpointing:
        LOGGER.info(
            "Gradient checkpointing enabled%s.",
            " automatically for MPS"
            if args.gradient_checkpointing is None and device.type == "mps"
            else "",
        )
    cache = load_prepared_cache(args.prepared_cache)
    selected_layers, centroids, centroid_payload = load_centroid_payload(
        args.centroid_file, args.property
    )

    centroid_model = centroid_payload.get("model_name")
    if centroid_model and centroid_model != args.model_name:
        raise ValueError(
            f"Centroid model mismatch: {centroid_model} != {args.model_name}"
        )
    pooling = args.pooling or centroid_payload.get("pooling", "last-token")

    train_indices = split_indices(cache, "adaptation")
    validation_indices = split_indices(cache, "validation")
    if args.limit_train is not None:
        train_indices = train_indices[: args.limit_train]
    if args.limit_validation is not None:
        validation_indices = validation_indices[: args.limit_validation]
    if not train_indices or not validation_indices:
        raise RuntimeError("Training and validation splits must be nonempty.")

    train_labels = cache["labels"][args.property][
        torch.tensor(train_indices, dtype=torch.long)
    ]
    validation_labels = cache["labels"][args.property][
        torch.tensor(validation_indices, dtype=torch.long)
    ]
    if int(train_labels.sum()) == 0:
        raise RuntimeError(f"No positive {args.property} training instances.")

    LOGGER.info(
        "property=%s layers=%s pooling=%s train=%d positives=%d "
        "validation=%d positives=%d",
        args.property,
        selected_layers,
        pooling,
        len(train_indices),
        int(train_labels.sum()),
        len(validation_indices),
        int(validation_labels.sum()),
    )

    tokenizer = load_tokenizer(args.model_name, args.trust_remote_code)
    base_model = load_base_model(
        args.model_name,
        device,
        dtype,
        args.trust_remote_code,
        not args.disable_low_cpu_memory,
    )
    layer_count = infer_layer_count(base_model)
    invalid = [layer for layer in selected_layers if not 0 <= layer < layer_count]
    if invalid:
        raise IndexError(f"Centroid layers outside model range: {invalid}")

    pad_token_id = determine_pad_token_id(cache, base_model, tokenizer)
    base_model.config.pad_token_id = pad_token_id
    if gradient_checkpointing:
        try:
            base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            LOGGER.warning(
                "Installed Transformers does not accept non-reentrant "
                "checkpointing arguments; using its default implementation."
            )
            base_model.gradient_checkpointing_enable()
        if hasattr(base_model, "enable_input_require_grads"):
            base_model.enable_input_require_grads()

    target_modules = discover_lora_target_modules(base_model, args.target_modules)
    modules_to_save = None
    if args.modules_to_save:
        modules_to_save = [
            item.strip() for item in args.modules_to_save.split(",") if item.strip()
        ]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.print_trainable_parameters()
    write_json(
        args.output_dir / "trainable_parameters.json",
        trainable_parameter_report(model),
    )

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    batches_per_epoch = math.ceil(len(train_indices) / args.batch_size)
    steps_per_epoch = math.ceil(
        batches_per_epoch / args.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    if args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
    elif args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
    else:
        scheduler = None

    config = {
        "prepared_cache": str(args.prepared_cache),
        "prepared_cache_sha256": sha256_file(args.prepared_cache),
        "centroid_file": str(args.centroid_file),
        "centroid_file_sha256": sha256_file(args.centroid_file),
        "property": args.property,
        "model_name": args.model_name,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "selected_layers": selected_layers,
        "activation_extraction": "selected-layer-forward-hooks",
        "language_modeling_objective": "prompt-to-gold-patch",
        "prompt_tokens_masked_from_lm_loss": True,
        "max_sequence_length": args.max_sequence_length,
        "gradient_checkpointing": gradient_checkpointing,
        "pooling": pooling,
        "activation_weight": args.activation_weight,
        "normalize_activations": args.normalize_activations,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": target_modules,
        "modules_to_save": modules_to_save,
        "train_instance_count": len(train_indices),
        "train_positive_count": int(train_labels.sum()),
        "validation_instance_count": len(validation_indices),
        "validation_positive_count": int(validation_labels.sum()),
        "seed": args.seed,
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(args.output_dir / "training_config.json", config)

    history: List[EpochMetrics] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    best_checkpoint: Optional[Path] = None
    patience = 0

    for epoch in range(1, args.epochs + 1):
        LOGGER.info("Starting epoch %d/%d with selected-layer forward hooks", epoch, args.epochs)
        train_metrics = run_epoch(
            model, cache, train_indices, args.property, selected_layers,
            centroids, pad_token_id, device, args.batch_size,
            args.max_sequence_length, pooling,
            args.activation_weight, args.normalize_activations, optimizer,
            scheduler, args.gradient_accumulation_steps, args.max_grad_norm,
            args.mixed_precision, dtype, True, args.seed, epoch, args.log_every,
        )
        validation_metrics = run_epoch(
            model, cache, validation_indices, args.property, selected_layers,
            centroids, pad_token_id, device, args.batch_size,
            args.max_sequence_length, pooling,
            args.activation_weight, args.normalize_activations, None, None, 1,
            args.max_grad_norm, args.mixed_precision, dtype, False, args.seed,
            epoch, args.log_every,
        )
        history.extend((train_metrics, validation_metrics))
        save_history(args.output_dir, history)

        LOGGER.info(
            "Epoch %d complete: train_total=%.6f val_total=%.6f "
            "val_language=%.6f val_activation=%.6f",
            epoch,
            train_metrics.mean_total_loss,
            validation_metrics.mean_total_loss,
            validation_metrics.mean_language_loss,
            validation_metrics.mean_activation_loss,
        )

        improved = validation_metrics.mean_total_loss < (
            best_validation_loss - args.minimum_delta
        )
        if improved:
            best_validation_loss = validation_metrics.mean_total_loss
            best_epoch = epoch
            patience = 0
            best_checkpoint = save_checkpoint(
                model, args.output_dir, epoch, validation_metrics
            )
        else:
            patience += 1
            if (
                args.early_stopping_patience > 0
                and patience >= args.early_stopping_patience
            ):
                LOGGER.info("Early stopping activated.")
                break

    if best_checkpoint is None:
        raise RuntimeError("No best checkpoint was saved.")

    del model, base_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    best_base = load_base_model(
        args.model_name,
        device,
        dtype,
        args.trust_remote_code,
        not args.disable_low_cpu_memory,
    )
    best_base.config.pad_token_id = pad_token_id
    best_model = PeftModel.from_pretrained(
        best_base, best_checkpoint, is_trainable=False
    )
    best_model.eval()

    adapter_dir = args.output_dir / "adapter"
    best_model.save_pretrained(adapter_dir)
    if tokenizer is not None:
        tokenizer.save_pretrained(adapter_dir)

    merged_dir = None
    if not args.skip_merge:
        LOGGER.info("Merging LoRA parameters into the pretrained model.")
        merged = best_model.merge_and_unload()
        merged.eval()
        merged_dir = args.output_dir / "merged_model"
        merged.save_pretrained(merged_dir, safe_serialization=True)
        if tokenizer is not None:
            tokenizer.save_pretrained(merged_dir)
        del merged

    best_metrics = next(
        item for item in history
        if item.split == "validation" and item.epoch == best_epoch
    )
    write_json(
        args.output_dir / "validation_summary.json",
        {
            "property": args.property,
            "best_epoch": best_epoch,
            "best_validation_total_loss": best_validation_loss,
            "best_validation_metrics": asdict(best_metrics),
            "selected_layers": selected_layers,
        "activation_extraction": "selected-layer-forward-hooks",
        "language_modeling_objective": "prompt-to-gold-patch",
        "prompt_tokens_masked_from_lm_loss": True,
        "max_sequence_length": args.max_sequence_length,
            "adapter_directory": str(adapter_dir),
            "merged_model_directory": str(merged_dir) if merged_dir else None,
        },
    )

    LOGGER.info("Activation-guided LoRA adaptation completed.")
    LOGGER.info("Best epoch: %d", best_epoch)
    LOGGER.info("Adapter saved to: %s", adapter_dir)
    if merged_dir is not None:
        LOGGER.info("Merged model saved to: %s", merged_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
