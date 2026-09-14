#!/usr/bin/env python3
"""
train_activation_steering.py

Learned inference-time activation-steering comparative baseline.

This script intentionally reuses the data loading, batching, pooling, model
loading, activation-loss, scheduler, and utility functions from the finalized
activation-guided LoRA training script.  The only methodological change is the
location of the learned adaptation:

    Proposed approach: optimize LoRA parameters; no runtime intervention.
    This baseline:     freeze all model weights; optimize one scalar alpha_l
                       per selected layer and apply runtime activation steering.

At selected transformer block l, every block-output vector h is replaced by

    h' = h + alpha_l * (c_l - h)

where c_l is the same fixed target-property centroid produced by
``discover_activation_centroids.py``.  alpha_l is initialized to 0, so training
starts exactly from the pretrained model.  The coefficients are unconstrained
real scalars and are learned with the same joint objective used by the LoRA
experiment:

    L_total = L_language + lambda_activation * L_activation

The base model remains frozen throughout.  The saved artifact is a small JSON
file containing the learned steering coefficients; no model weights are saved.

Example
-------
python train_activation_steering.py \
    --source-training-script train_activation_guided_lora_forward_hooks_v4_fixed.py \
    --prepared-cache outputs/prepared_swebench/prepared_dataset.pt \
    --centroid-file outputs/activation_centroids/property_centroids.pt \
    --property coupling \
    --model-name bigcode/starcoderbase-1b \
    --output-dir outputs/steering_coupling \
    --device mps --dtype float32 \
    --batch-size 1 --gradient-accumulation-steps 8 \
    --epochs 3 --learning-rate 2e-4 --activation-weight 1.0
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
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

LOGGER = logging.getLogger("train_activation_steering")
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


def import_source(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("ag_lora_source", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import source training script: {path}")
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


def write_history(path: Path, rows: Sequence[EpochMetrics]) -> None:
    fieldnames = list(EpochMetrics.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def replace_first_tensor(output: Any, tensor: torch.Tensor) -> Any:
    """Replace a transformer block's primary hidden-state tensor."""
    if torch.is_tensor(output):
        return tensor
    if isinstance(output, tuple):
        return (tensor, *output[1:])
    if isinstance(output, list):
        result = list(output)
        result[0] = tensor
        return result
    if hasattr(output, "last_hidden_state"):
        # Transformer blocks used here normally return Tensor/tuple.  Avoid
        # silently mutating unfamiliar structured outputs.
        raise TypeError(
            "Structured block output with last_hidden_state is not supported for "
            "steering replacement; expected Tensor or tuple/list."
        )
    raise TypeError(f"Unsupported transformer block output type: {type(output)!r}")


class LearnedCentroidSteering(torch.nn.Module):
    """Runtime forward hooks with one learned scalar per selected layer."""

    def __init__(
        self,
        source: Any,
        model: torch.nn.Module,
        selected_layers: Sequence[int],
        centroids: Mapping[int, torch.Tensor],
        initial_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.source = source
        self.blocks = source.resolve_transformer_blocks(model)
        self.selected_layers = tuple(dict.fromkeys(int(x) for x in selected_layers))
        invalid = [x for x in self.selected_layers if x < 0 or x >= len(self.blocks)]
        if invalid:
            raise IndexError(f"Steering layers outside transformer range: {invalid}")

        self.alphas = torch.nn.ParameterDict({
            str(layer): torch.nn.Parameter(torch.tensor(float(initial_alpha), dtype=torch.float32))
            for layer in self.selected_layers
        })
        self.centroids: Dict[int, torch.Tensor] = {
            int(layer): centroids[int(layer)].detach().cpu().to(torch.float32)
            for layer in self.selected_layers
        }
        self.activations: Dict[int, torch.Tensor] = {}
        self.handles: List[Any] = []
        self.enabled = True
        self.capture_enabled = True
        self._register()

    def _register(self) -> None:
        for layer in self.selected_layers:
            def hook(module, inputs, output, layer_index: int = layer):
                hidden = self.source._first_tensor(output)
                if not self.enabled:
                    if self.capture_enabled:
                        self.activations[layer_index] = hidden
                    return output

                centroid = self.centroids[layer_index].to(
                    device=hidden.device, dtype=hidden.dtype
                ).view(1, 1, -1)
                if centroid.size(-1) != hidden.size(-1):
                    raise ValueError(
                        f"Centroid/hidden mismatch at layer {layer_index}: "
                        f"{centroid.size(-1)} vs {hidden.size(-1)}"
                    )
                alpha = self.alphas[str(layer_index)].to(
                    device=hidden.device, dtype=hidden.dtype
                )
                steered = hidden + alpha * (centroid - hidden)
                if self.capture_enabled:
                    self.activations[layer_index] = steered
                return replace_first_tensor(output, steered)

            self.handles.append(self.blocks[layer].register_forward_hook(hook))

    def begin_forward(self, capture: bool = True) -> None:
        self.activations.clear()
        self.capture_enabled = bool(capture)

    def require_complete_capture(self) -> Mapping[int, torch.Tensor]:
        missing = [x for x in self.selected_layers if x not in self.activations]
        if missing:
            raise RuntimeError(f"Steering hooks missed selected layers: {missing}")
        return self.activations

    def coefficient_dict(self) -> Dict[str, float]:
        return {
            str(layer): float(self.alphas[str(layer)].detach().cpu().item())
            for layer in self.selected_layers
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.activations.clear()


def minibatches(indices: Sequence[int], batch_size: int) -> Iterable[List[int]]:
    for start in range(0, len(indices), batch_size):
        yield list(indices[start:start + batch_size])


def run_epoch(
    *,
    source: Any,
    model: torch.nn.Module,
    steering: LearnedCentroidSteering,
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
    model.eval()  # Keep dropout etc. fixed; only alpha scalars are optimized.
    steering.train(training)

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

    with grad_context():
        for batch_number, batch_indices in enumerate(batches, start=1):
            batch = source.make_batch(
                cache, batch_indices, property_name, pad_token_id, device,
                max_sequence_length,
            )
            steering.begin_forward(capture=True)

            with source.autocast_context(device, mixed_precision, autocast_dtype):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["language_labels"],
                    use_cache=False,
                    return_dict=True,
                )
                language_loss = outputs.loss
                if language_loss is None:
                    raise RuntimeError("Model did not return language-modeling loss.")
                captured = steering.require_complete_capture()
                activation_loss, mean_distance, positive_count = source.activation_guidance_loss(
                    captured,
                    batch["activation_attention_mask"],
                    batch["property_labels"],
                    selected_layers,
                    centroids,
                    pooling,
                    normalize_activations,
                    zero_reference=language_loss,
                )
                total_loss = language_loss + activation_weight * activation_loss

            if not torch.isfinite(total_loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch_number}, batch={batch_number}."
                )

            if training:
                (total_loss / gradient_accumulation_steps).backward()
                should_step = (
                    batch_number % gradient_accumulation_steps == 0
                    or batch_number == len(batches)
                )
                if should_step:
                    clip_grad_norm_(list(steering.parameters()), max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimization_steps += 1

            size = len(batch_indices)
            total_sum += float(total_loss.detach().cpu()) * size
            language_sum += float(language_loss.detach().cpu()) * size
            activation_sum += float(activation_loss.detach().cpu()) * size
            distance_sum += float(mean_distance.detach().cpu()) * size
            instance_count += size
            positive_count_total += positive_count

            if batch_number % log_every == 0 or batch_number == len(batches):
                LOGGER.info(
                    "epoch=%d %s batch=%d/%d total=%.6f lm=%.6f act=%.6f alpha=%s",
                    epoch_number,
                    "train" if training else "validation",
                    batch_number,
                    len(batches),
                    float(total_loss.detach().cpu()),
                    float(language_loss.detach().cpu()),
                    float(activation_loss.detach().cpu()),
                    steering.coefficient_dict(),
                )

            del outputs, total_loss, language_loss, activation_loss, batch

    lr = 0.0
    if optimizer is not None and optimizer.param_groups:
        lr = float(optimizer.param_groups[0]["lr"])
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
        learning_rate=lr,
        elapsed_seconds=time.time() - started,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train learned activation-steering baseline.")
    p.add_argument(
        "--source-training-script", type=Path,
        default=Path("train_activation_guided_lora_forward_hooks_v4_fixed.py"),
        help="Finalized LoRA training script whose utilities/objective are reused.",
    )
    p.add_argument("--prepared-cache", type=Path, default=Path("outputs/prepared_swebench/prepared_dataset.pt"))
    p.add_argument("--centroid-file", type=Path, default=Path("outputs/activation_centroids/property_centroids.pt"))
    p.add_argument("--property", choices=PROPERTIES, required=True)
    p.add_argument("--model-name", default="bigcode/starcoderbase-1b")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation-steps", type=int, default=8)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--activation-weight", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--scheduler", choices=("none", "linear", "cosine"), default="none")
    p.add_argument("--warmup-ratio", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--max-sequence-length", type=int, default=1536)
    p.add_argument("--normalize-activations", action="store_true")
    p.add_argument("--mixed-precision", action="store_true")
    p.add_argument("--initial-alpha", type=float, default=0.0)
    p.add_argument("--early-stopping-patience", type=int, default=0)
    p.add_argument("--minimum-delta", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--disable-low-cpu-memory", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1 or args.epochs < 1:
        raise ValueError("batch size, accumulation steps, and epochs must be >= 1")
    if args.max_sequence_length < 2:
        raise ValueError("--max-sequence-length must be >= 2")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory not empty: {args.output_dir}; use --overwrite")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    source = import_source(args.source_training_script)
    device = source.resolve_device(args.device)
    dtype = source.resolve_dtype(args.dtype, device)
    cache = source.load_prepared_cache(args.prepared_cache)
    selected_layers, centroids, centroid_payload = source.load_centroid_payload(
        args.centroid_file, args.property
    )
    pooling = str(centroid_payload.get("pooling", "last-token"))
    train_indices = source.split_indices(cache, "adaptation")
    validation_indices = source.split_indices(cache, "validation")
    if not train_indices or not validation_indices:
        raise RuntimeError("Adaptation and validation splits must be nonempty.")

    tokenizer = source.load_tokenizer(args.model_name, args.trust_remote_code)
    model = source.load_base_model(
        args.model_name, device, dtype, args.trust_remote_code,
        not args.disable_low_cpu_memory,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    pad_token_id = source.determine_pad_token_id(cache, model, tokenizer)
    model.config.pad_token_id = pad_token_id

    steering = LearnedCentroidSteering(
        source, model, selected_layers, centroids, initial_alpha=args.initial_alpha
    ).to(device)
    optimizer = AdamW(
        steering.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    batches_per_epoch = math.ceil(len(train_indices) / args.batch_size)
    steps_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    if args.scheduler == "linear":
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    elif args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = None

    train_labels = cache["labels"][args.property][torch.tensor(train_indices, dtype=torch.long)]
    validation_labels = cache["labels"][args.property][torch.tensor(validation_indices, dtype=torch.long)]
    config = {
        "baseline": "learned_inference_time_activation_steering",
        "source_training_script": str(args.source_training_script),
        "source_training_script_sha256": sha256_file(args.source_training_script),
        "prepared_cache": str(args.prepared_cache),
        "prepared_cache_sha256": sha256_file(args.prepared_cache),
        "centroid_file": str(args.centroid_file),
        "centroid_file_sha256": sha256_file(args.centroid_file),
        "property": args.property,
        "model_name": args.model_name,
        "selected_layers": selected_layers,
        "pooling": pooling,
        "steering_equation": "h_prime = h + alpha_l * (centroid_l - h)",
        "alpha_parameterization": "unconstrained_real_scalar_per_selected_layer",
        "initial_alpha": args.initial_alpha,
        "model_weights_frozen": True,
        "runtime_activation_intervention": True,
        "joint_objective_reused_from_proposed_training": True,
        "language_modeling_objective": "prompt-to-gold-patch",
        "activation_guidance_positive_instances_only": True,
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
        "max_sequence_length": args.max_sequence_length,
        "train_instance_count": len(train_indices),
        "train_positive_count": int(train_labels.sum()),
        "validation_instance_count": len(validation_indices),
        "validation_positive_count": int(validation_labels.sum()),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "seed": args.seed,
        "script_sha256": sha256_file(Path(__file__)),
    }
    write_json(args.output_dir / "training_config.json", config)

    history: List[EpochMetrics] = []
    best_loss = float("inf")
    best_epoch = 0
    best_coefficients: Optional[Dict[str, float]] = None
    patience = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            source=source, model=model, steering=steering, cache=cache,
            indices=train_indices, property_name=args.property,
            selected_layers=selected_layers, centroids=centroids,
            pad_token_id=pad_token_id, device=device, batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length, pooling=pooling,
            activation_weight=args.activation_weight,
            normalize_activations=args.normalize_activations,
            optimizer=optimizer, scheduler=scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm, mixed_precision=args.mixed_precision,
            autocast_dtype=dtype, shuffle=True, seed=args.seed,
            epoch_number=epoch, log_every=args.log_every,
        )
        validation_metrics = run_epoch(
            source=source, model=model, steering=steering, cache=cache,
            indices=validation_indices, property_name=args.property,
            selected_layers=selected_layers, centroids=centroids,
            pad_token_id=pad_token_id, device=device, batch_size=args.batch_size,
            max_sequence_length=args.max_sequence_length, pooling=pooling,
            activation_weight=args.activation_weight,
            normalize_activations=args.normalize_activations,
            optimizer=None, scheduler=None, gradient_accumulation_steps=1,
            max_grad_norm=args.max_grad_norm, mixed_precision=args.mixed_precision,
            autocast_dtype=dtype, shuffle=False, seed=args.seed,
            epoch_number=epoch, log_every=args.log_every,
        )
        history.extend((train_metrics, validation_metrics))
        write_history(args.output_dir / "training_history.csv", history)

        LOGGER.info(
            "epoch=%d train_total=%.6f val_total=%.6f coefficients=%s",
            epoch, train_metrics.mean_total_loss, validation_metrics.mean_total_loss,
            steering.coefficient_dict(),
        )
        if validation_metrics.mean_total_loss < best_loss - args.minimum_delta:
            best_loss = validation_metrics.mean_total_loss
            best_epoch = epoch
            best_coefficients = steering.coefficient_dict()
            patience = 0
        else:
            patience += 1
            if args.early_stopping_patience > 0 and patience >= args.early_stopping_patience:
                LOGGER.info("Early stopping activated.")
                break

    if best_coefficients is None:
        raise RuntimeError("No best steering coefficients were selected.")

    coefficient_payload = {
        "baseline": "learned_inference_time_activation_steering",
        "property": args.property,
        "selected_layers": selected_layers,
        "pooling": pooling,
        "equation": "h_prime = h + alpha_l * (centroid_l - h)",
        "coefficients": best_coefficients,
        "best_epoch": best_epoch,
        "best_validation_total_loss": best_loss,
        "centroid_file": str(args.centroid_file),
        "centroid_file_sha256": sha256_file(args.centroid_file),
        "base_model": args.model_name,
    }
    write_json(args.output_dir / "steering_coefficients.json", coefficient_payload)
    write_json(
        args.output_dir / "validation_summary.json",
        {
            **coefficient_payload,
            "best_validation_metrics": asdict(next(
                row for row in history if row.split == "validation" and row.epoch == best_epoch
            )),
            "model_weights_updated": False,
            "runtime_intervention_required": True,
        },
    )

    LOGGER.info("Learned activation-steering baseline training completed.")
    LOGGER.info("Best epoch: %d", best_epoch)
    LOGGER.info("Best coefficients: %s", best_coefficients)
    LOGGER.info("Saved: %s", args.output_dir / "steering_coefficients.json")

    steering.close()
    del steering, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
