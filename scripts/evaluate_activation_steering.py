#!/usr/bin/env python3
"""
evaluate_activation_steering.py

Evaluate the learned inference-time activation-steering comparative baseline on
exactly the same held-out test protocol used by evaluate_activation_guided_model.py.

The script dynamically imports the finalized permanent-model evaluator and
reuses its dataset loader, deterministic output labeling, generation settings,
activation metrics, capability-preservation metric, and per-instance reporting.
The only added mechanism is a runtime forward-hook intervention on the frozen
pretrained model using property-specific learned alpha coefficients.

Expected coefficient files
--------------------------
outputs/steering_complexity/steering_coefficients.json
outputs/steering_coupling/steering_coefficients.json
outputs/steering_modularity/steering_coefficients.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

LOGGER = logging.getLogger("evaluate_activation_steering")
PROPERTIES = ("coupling", "complexity", "modularity")


def import_source(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("ag_eval_source", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import source evaluation script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def replace_first_tensor(output: Any, tensor: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        result = list(output)
        result[0] = tensor
        return result
    raise TypeError(f"Unsupported transformer block output type: {type(output)!r}")


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported transformer block output type: {type(output)!r}")


def resolve_transformer_blocks(model: torch.nn.Module):
    paths = (
        ("transformer", "h"),
        ("model", "layers"),
        ("gpt_neox", "layers"),
        ("transformer", "blocks"),
        ("transformer", "layers"),
    )
    roots = [model]
    for root in roots:
        for path in paths:
            value: Any = root
            ok = True
            for part in path:
                if not hasattr(value, part):
                    ok = False
                    break
                value = getattr(value, part)
            if ok and isinstance(value, (torch.nn.ModuleList, list, tuple)):
                return value
    raise RuntimeError("Could not locate transformer blocks for steering hooks.")


class FixedCentroidSteering:
    def __init__(
        self,
        model: torch.nn.Module,
        layers: Sequence[int],
        centroids: Mapping[int, torch.Tensor],
        coefficients: Mapping[str, float],
    ) -> None:
        self.blocks = resolve_transformer_blocks(model)
        self.layers = tuple(int(x) for x in layers)
        self.centroids = {int(k): v.detach().cpu().to(torch.float32) for k, v in centroids.items()}
        self.coefficients = {str(k): float(v) for k, v in coefficients.items()}
        self.handles: List[Any] = []
        missing = [layer for layer in self.layers if str(layer) not in self.coefficients]
        if missing:
            raise KeyError(f"Missing steering coefficients for layers: {missing}")
        self._register()

    def _register(self) -> None:
        for layer in self.layers:
            def hook(module, inputs, output, layer_index: int = layer):
                hidden = first_tensor(output)
                centroid = self.centroids[layer_index].to(
                    device=hidden.device, dtype=hidden.dtype
                ).view(1, 1, -1)
                if centroid.size(-1) != hidden.size(-1):
                    raise ValueError(
                        f"Centroid/hidden mismatch at layer {layer_index}: "
                        f"{centroid.size(-1)} vs {hidden.size(-1)}"
                    )
                alpha = torch.as_tensor(
                    self.coefficients[str(layer_index)],
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                steered = hidden + alpha * (centroid - hidden)
                return replace_first_tensor(output, steered)
            self.handles.append(self.blocks[layer].register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def load_coefficients(path: Path, property_name: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("property") != property_name:
        raise ValueError(
            f"Coefficient file property mismatch: expected {property_name}, "
            f"found {payload.get('property')}"
        )
    if "coefficients" not in payload:
        raise KeyError(f"No coefficients in {path}")
    return payload


def build_comparison_rows(source, baseline, steered, capability_tolerance: float):
    output_rows: List[Dict[str, Any]] = []
    activation_rows: List[Dict[str, Any]] = []
    capability_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    summary: Dict[str, Any] = {}
    baseline_nll = baseline["capability_metrics"]["mean_reference_patch_nll"]

    for property_name in PROPERTIES:
        result = steered[property_name]
        base_rate = baseline["output_metrics"]["positive_rates"][property_name]
        steering_rate = result["output_metrics"]["positive_rates"][property_name]
        rate_delta = steering_rate - base_rate
        output_improved = rate_delta > 0
        output_rows.extend([
            {
                "property": property_name, "model": "pretrained",
                "positive_count": baseline["output_metrics"]["positive_counts"][property_name],
                "instance_count": baseline["output_metrics"]["instance_count"],
                "positive_rate": base_rate,
                "parseable_output_rate": baseline["output_metrics"]["parseable_output_rate"],
                "change_from_pretrained": 0.0,
            },
            {
                "property": property_name, "model": f"{property_name}_activation_steering",
                "positive_count": result["output_metrics"]["positive_counts"][property_name],
                "instance_count": result["output_metrics"]["instance_count"],
                "positive_rate": steering_rate,
                "parseable_output_rate": result["output_metrics"]["parseable_output_rate"],
                "change_from_pretrained": rate_delta,
            },
        ])

        for subset in ("all", "positive_reference"):
            for layer in result["activation_metrics"][property_name][subset]:
                b = baseline["activation_metrics"][property_name][subset][layer]
                s = result["activation_metrics"][property_name][subset][layer]
                activation_rows.extend([
                    {
                        "property": property_name, "subset": subset, "layer": int(layer),
                        "model": "pretrained", "count": b["count"],
                        "centroid_distance": b["centroid_distance"],
                        "cosine_similarity": b["cosine_similarity"],
                        "activation_norm": b["activation_norm"],
                    },
                    {
                        "property": property_name, "subset": subset, "layer": int(layer),
                        "model": f"{property_name}_activation_steering", "count": s["count"],
                        "centroid_distance": s["centroid_distance"],
                        "cosine_similarity": s["cosine_similarity"],
                        "activation_norm": s["activation_norm"],
                    },
                ])

        base_distance = source.average_layer_metric(baseline, property_name, "all", "centroid_distance")
        steer_distance = source.average_layer_metric(result, property_name, "all", "centroid_distance")
        base_cosine = source.average_layer_metric(baseline, property_name, "all", "cosine_similarity")
        steer_cosine = source.average_layer_metric(result, property_name, "all", "cosine_similarity")
        activation_improved = bool(
            base_distance is not None and steer_distance is not None
            and base_cosine is not None and steer_cosine is not None
            and steer_distance < base_distance and steer_cosine > base_cosine
        )

        steering_nll = result["capability_metrics"]["mean_reference_patch_nll"]
        nll_change = source.relative_change(steering_nll, baseline_nll)
        capability_preserved = bool(
            nll_change is not None and nll_change <= capability_tolerance
        )
        capability_rows.extend([
            {
                "property": property_name, "model": "pretrained",
                **baseline["capability_metrics"], "relative_nll_change": 0.0,
            },
            {
                "property": property_name, "model": f"{property_name}_activation_steering",
                **result["capability_metrics"], "relative_nll_change": nll_change,
            },
        ])

        supported = bool(output_improved and activation_improved and capability_preserved)
        row = {
            "property": property_name,
            "output_positive_rate_pretrained": base_rate,
            "output_positive_rate_steering": steering_rate,
            "output_rate_change": rate_delta,
            "output_improved": output_improved,
            "centroid_distance_pretrained": base_distance,
            "centroid_distance_steering": steer_distance,
            "cosine_similarity_pretrained": base_cosine,
            "cosine_similarity_steering": steer_cosine,
            "activation_improved": activation_improved,
            "reference_patch_nll_pretrained": baseline_nll,
            "reference_patch_nll_steering": steering_nll,
            "relative_nll_change": nll_change,
            "capability_tolerance": capability_tolerance,
            "capability_preserved": capability_preserved,
            "activation_steering_supported": supported,
        }
        summary_rows.append(row)
        summary[property_name] = row

    summary["interpretation"] = {
        "structural_condition": (
            "Each steering condition uses the original frozen pretrained model and "
            "applies property-specific learned activation intervention at runtime."
        ),
        "decision_rule": (
            "Steering is supported when output positive rate increases, mean centroid "
            "distance decreases while cosine similarity increases, and reference-patch "
            "NLL degradation remains within the configured tolerance."
        ),
        "comparison_scope": (
            "This is a controlled baseline for runtime activation intervention, not a "
            "claim that the learned coefficient formulation represents all activation-"
            "engineering methods."
        ),
    }
    return output_rows, activation_rows, capability_rows, summary_rows, summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate learned activation-steering baseline.")
    p.add_argument(
        "--source-evaluation-script", type=Path,
        default=Path("evaluate_activation_guided_model.py"),
    )
    p.add_argument("--prepared-cache", type=Path, default=Path("outputs/prepared_swebench/prepared_dataset.pt"))
    p.add_argument("--centroid-file", type=Path, default=Path("outputs/activation_centroids/property_centroids.pt"))
    p.add_argument("--labeling-script", type=Path, default=Path("prepare_dataset_and_labels.py"))
    p.add_argument("--base-model", default="bigcode/starcoderbase-1b")
    p.add_argument("--complexity-coefficients", type=Path, default=Path("outputs/steering_complexity/steering_coefficients.json"))
    p.add_argument("--coupling-coefficients", type=Path, default=Path("outputs/steering_coupling/steering_coefficients.json"))
    p.add_argument("--modularity-coefficients", type=Path, default=Path("outputs/steering_modularity/steering_coefficients.json"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/activation_steering_evaluation"))
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--max-sequence-length", type=int, default=1536)
    p.add_argument("--do-sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--capability-tolerance", type=float, default=0.05)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--disable-low-cpu-memory", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory not empty: {args.output_dir}; use --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "predictions").mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    source = import_source(args.source_evaluation_script)
    device = source.resolve_device(args.device)
    dtype = source.resolve_dtype(args.dtype, device)
    cache = source.load_prepared_cache(args.prepared_cache)
    centroids = source.load_centroids(args.centroid_file)
    label_patch = source.import_label_patch(args.labeling_script)
    pooling = str(centroids.get("pooling", "last-token"))

    test_indices = source.split_indices(cache, args.split)
    if args.limit is not None:
        test_indices = test_indices[:args.limit]
    if not test_indices:
        raise RuntimeError(f"Split '{args.split}' is empty.")

    coefficient_paths = {
        "complexity": args.complexity_coefficients,
        "coupling": args.coupling_coefficients,
        "modularity": args.modularity_coefficients,
    }
    coefficient_payloads = {
        prop: load_coefficients(path, prop) for prop, path in coefficient_paths.items()
    }

    config = {
        "baseline": "learned_inference_time_activation_steering",
        "source_evaluation_script": str(args.source_evaluation_script),
        "source_evaluation_script_sha256": sha256_file(args.source_evaluation_script),
        "prepared_cache": str(args.prepared_cache),
        "prepared_cache_sha256": sha256_file(args.prepared_cache),
        "centroid_file": str(args.centroid_file),
        "centroid_file_sha256": sha256_file(args.centroid_file),
        "labeling_script": str(args.labeling_script),
        "labeling_script_sha256": sha256_file(args.labeling_script),
        "base_model": args.base_model,
        "coefficient_files": {k: str(v) for k, v in coefficient_paths.items()},
        "coefficients": {k: v["coefficients"] for k, v in coefficient_payloads.items()},
        "split": args.split,
        "instance_count": len(test_indices),
        "pooling": pooling,
        "selected_layers": centroids["selected_layers"],
        "runtime_activation_intervention": True,
        "model_parameters_modified": False,
        "steering_equation": "h_prime = h + alpha_l * (centroid_l - h)",
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_sequence_length": args.max_sequence_length,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "capability_tolerance": args.capability_tolerance,
        "seed": args.seed,
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(args.output_dir / "evaluation_config.json", config)

    tokenizer = source.load_tokenizer(args.base_model, args.trust_remote_code)

    # Pretrained reference: exactly the original evaluator's no-intervention path.
    baseline_model = source.load_model(
        args.base_model, device, dtype, args.trust_remote_code,
        not args.disable_low_cpu_memory,
    )
    baseline_result = source.evaluate_one_model(
        model_name_for_results="pretrained",
        model=baseline_model, tokenizer=tokenizer, cache=cache,
        test_indices=test_indices, centroids_payload=centroids,
        properties_to_measure=PROPERTIES, label_patch=label_patch,
        output_dir=args.output_dir, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_sequence_length=args.max_sequence_length,
        do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p,
        pooling=pooling, device=device, dtype=dtype,
        mixed_precision=args.mixed_precision, seed=args.seed, log_every=args.log_every,
    )
    all_instance_rows = list(baseline_result.pop("per_instance_rows"))
    del baseline_model
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    elif device.type == "mps": torch.mps.empty_cache()

    steering_results: Dict[str, Dict[str, Any]] = {}
    for property_name in PROPERTIES:
        model = source.load_model(
            args.base_model, device, dtype, args.trust_remote_code,
            not args.disable_low_cpu_memory,
        )
        layers = [int(x) for x in centroids["selected_layers"][property_name]]
        prop_centroids = {
            int(layer): centroids["centroids"][property_name][int(layer)]
            for layer in layers
        }
        coefficients = coefficient_payloads[property_name]["coefficients"]
        with FixedCentroidSteering(model, layers, prop_centroids, coefficients):
            result = source.evaluate_one_model(
                model_name_for_results=f"{property_name}_activation_steering",
                model=model, tokenizer=tokenizer, cache=cache,
                test_indices=test_indices, centroids_payload=centroids,
                properties_to_measure=(property_name,), label_patch=label_patch,
                output_dir=args.output_dir, batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                max_sequence_length=args.max_sequence_length,
                do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p,
                pooling=pooling, device=device, dtype=dtype,
                mixed_precision=args.mixed_precision, seed=args.seed, log_every=args.log_every,
            )
        all_instance_rows.extend(result.pop("per_instance_rows"))
        steering_results[property_name] = result
        del model
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
        elif device.type == "mps": torch.mps.empty_cache()

    output_rows, activation_rows, capability_rows, summary_rows, summary = build_comparison_rows(
        source, baseline_result, steering_results, args.capability_tolerance
    )
    source.write_csv(args.output_dir / "output_level_results.csv", output_rows)
    source.write_csv(args.output_dir / "activation_level_results.csv", activation_rows)
    source.write_csv(args.output_dir / "capability_preservation.csv", capability_rows)
    source.write_csv(args.output_dir / "activation_steering_summary.csv", summary_rows)
    source.write_jsonl(args.output_dir / "per_instance_results.jsonl", all_instance_rows)
    write_json(
        args.output_dir / "evaluation_summary.json",
        {
            "summary": summary,
            "baseline": baseline_result,
            "steering_models": steering_results,
            "steering_coefficients": {
                prop: payload["coefficients"] for prop, payload in coefficient_payloads.items()
            },
            "config": config,
        },
    )

    LOGGER.info("Activation-steering evaluation completed: %s", args.output_dir)
    for row in summary_rows:
        LOGGER.info(
            "%s | output=%s activation=%s capability=%s steering_supported=%s",
            row["property"], row["output_improved"], row["activation_improved"],
            row["capability_preserved"], row["activation_steering_supported"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
