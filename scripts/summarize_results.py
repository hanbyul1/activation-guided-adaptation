#!/usr/bin/env python3
"""
summarize_results.py

Assemble the artifacts produced by the four earlier pipeline scripts into
paper-ready CSV tables, LaTeX tables, figures, and a machine-readable summary.
No model inference, probing, or optimization is performed here.

Pipeline correspondence
-----------------------
prepare_dataset_and_labels.py       -> benchmark labeling statistics
discover_activation_centroids.py    -> layer selection and representation discovery
train_activation_guided_lora.py     -> adaptation histories and merged-model metadata
evaluate_activation_guided_model.py -> post-merge output, activation, and capability evidence
evaluate_activation_steering.py      -> inference-time activation-steering baseline evidence
summarize_results.py                -> tables, figures, and final evidence summary

Default expected layout
-----------------------
outputs/prepared_swebench/
    labeling_statistics.json
outputs/activation_centroids/
    selected_layers.json
    layer_analysis.csv
    centroid_statistics.json
outputs/lora_complexity/
    training_history.csv
    validation_summary.json
    merged_model/config.json
outputs/lora_coupling/
    ...
outputs/lora_modularity/
    ...
outputs/activation_guided_evaluation/
    evaluation_summary.json
    permanent_adaptation_summary.csv
    output_level_results.csv
    activation_level_results.csv
    capability_preservation.csv
outputs/steering_complexity/
    training_history.csv
    validation_summary.json
    steering_coefficients.json
    training_config.json
outputs/steering_coupling/
    ...
outputs/steering_modularity/
    ...
outputs/activation_steering_evaluation/
    evaluation_summary.json
    activation_steering_summary.csv

Outputs
-------
<output-dir>/
    summary.json
    paper_statistics.json
    evidence_report.txt
    tables/csv/*.csv
    tables/latex/*.tex
    figures/*.pdf
    figures/*.png

The script is deliberately conservative: missing optional inputs are reported
and omitted rather than silently replaced with assumed values.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LOGGER = logging.getLogger("summarize_results")
PROPERTIES: Tuple[str, ...] = ("complexity", "coupling", "modularity")
PROPERTY_LABELS = {
    "complexity": "Complexity",
    "coupling": "Coupling",
    "modularity": "Modularity",
}
SIGNATURE_LABELS = {
    "000": "None",
    "010": "Complexity only",
    "100": "Coupling only",
    "001": "Modularity only",
    "110": "Complexity + Coupling",
    "011": "Complexity + Modularity",
    "101": "Coupling + Modularity",
    "111": "Complexity + Coupling + Modularity",
}


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def read_json(path: Path, required: bool = False) -> Any:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        LOGGER.warning("Missing optional input: %s", path)
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv(path: Path, required: bool = False) -> List[Dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        LOGGER.warning("Missing optional input: %s", path)
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def integer(value: Any) -> Optional[int]:
    numeric = number(value)
    return int(numeric) if numeric is not None else None


def boolean(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if value is None or value == "":
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def percent(value: Any, digits: int = 1) -> str:
    numeric = number(value)
    return "--" if numeric is None else f"{100.0 * numeric:.{digits}f}\\%"


def fmt(value: Any, digits: int = 4) -> str:
    numeric = number(value)
    return "--" if numeric is None else f"{numeric:.{digits}f}"


def fmt_int(value: Any) -> str:
    numeric = integer(value)
    return "--" if numeric is None else f"{numeric:,}"


def yes_no(value: Any) -> str:
    flag = boolean(value)
    return "--" if flag is None else ("Yes" if flag else "No")


def latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_table(
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    alignment: str,
    table_star: bool = False,
    font_size: str = r"\scriptsize",
    tabcolsep: str = "3.5pt",
    arraystretch: str = "0.90",
    group_breaks: Optional[Iterable[int]] = None,
) -> str:
    environment = "table*" if table_star else "table"
    breaks = set(group_breaks or [])
    lines = [
        rf"\begin{{{environment}}}[!htb]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        font_size,
        rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}",
        rf"\renewcommand{{\arraystretch}}{{{arraystretch}}}",
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        " & ".join(rf"\textbf{{{header}}}" for header in headers) + r" \\",
        r"\hline",
    ]
    for index, row in enumerate(rows, start=1):
        lines.append(" & ".join(str(cell) for cell in row) + r" \\")
        if index in breaks:
            lines.append(r"\hline")
    if not rows or len(rows) not in breaks:
        lines.append(r"\hline")
    lines.extend([rf"\end{{tabular}}", rf"\end{{{environment}}}", ""])
    return "\n".join(lines)


def save_latex(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve_labeling_statistics(prepared_dir: Path, explicit: Optional[Path]) -> Optional[Path]:
    candidates = [
        explicit,
        prepared_dir / "labeling_statistics.json",
        prepared_dir.parent / "labeling_statistics.json",
        Path("labeling_statistics.json"),
    ]
    return next((path for path in candidates if path and path.exists()), None)


def dataset_tables(stats: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    retained = integer(stats.get("retained_instances"))
    rejected = integer(stats.get("rejected_instances"))
    original = retained + rejected if retained is not None and rejected is not None else None
    splits = stats.get("splits", {}) or {}
    positives = stats.get("positive_labels", {}) or {}
    rates = stats.get("positive_rates", {}) or {}
    fragments = stats.get("parseable_fragment_pairs", {}) or {}
    signatures = stats.get("label_signatures", {}) or {}
    none_count = integer(signatures.get("000"))
    at_least_one = retained - none_count if retained is not None and none_count is not None else None

    rows = [
        {"statistic": "Original SWE-bench instances", "value": original},
        {"statistic": "Retained instances", "value": retained},
        {"statistic": "Rejected instances", "value": rejected},
        {"statistic": "Average parseable fragment pairs per instance", "value": fragments.get("mean_per_instance")},
        {"statistic": "Total parseable fragment pairs", "value": fragments.get("total")},
        {"statistic": "Adaptation partition", "value": splits.get("adaptation")},
        {"statistic": "Validation partition", "value": splits.get("validation")},
        {"statistic": "Test partition", "value": splits.get("test")},
        {"statistic": "Positive complexity labels", "value": positives.get("complexity"), "rate": rates.get("complexity")},
        {"statistic": "Positive coupling labels", "value": positives.get("coupling"), "rate": rates.get("coupling")},
        {"statistic": "Positive modularity labels", "value": positives.get("modularity"), "rate": rates.get("modularity")},
        {"statistic": "No positive property labels (000)", "value": none_count, "rate": none_count / retained if none_count is not None and retained else None},
        {"statistic": "At least one positive property label", "value": at_least_one, "rate": at_least_one / retained if at_least_one is not None and retained else None},
    ]
    distribution = []
    for signature in ("000", "010", "100", "001", "110", "011", "101", "111"):
        count = integer(signatures.get(signature))
        distribution.append({
            "signature": signature,
            "combination": SIGNATURE_LABELS[signature],
            "instances": count,
            "rate": count / retained if count is not None and retained else None,
        })
    return rows, distribution


def select_layer_rows(layer_rows: Sequence[Mapping[str, Any]], selected_payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    selected = selected_payload.get("selected_layers", {}) if selected_payload else {}
    lookup: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    for row in layer_rows:
        prop = str(row.get("property_name", row.get("property", ""))).lower()
        layer = integer(row.get("layer"))
        if prop and layer is not None:
            lookup[(prop, layer)] = row

    output: List[Dict[str, Any]] = []
    for prop in PROPERTIES:
        for rank, layer_value in enumerate(selected.get(prop, []), start=1):
            layer = int(layer_value)
            row = lookup.get((prop, layer), {})
            output.append({
                "property": prop,
                "layer": layer,
                "balanced_accuracy": number(row.get("balanced_accuracy")),
                "roc_auc": number(row.get("roc_auc")),
                "bounded_fisher_ratio": number(row.get("bounded_fisher_ratio", row.get("bounded_fisher"))),
                "score": number(row.get("score", row.get("composite_score"))),
                "rank": integer(row.get("rank")) or rank,
            })
    return output


def discovery_rows(selected_payload: Mapping[str, Any], centroid_stats: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    selected = selected_payload.get("selected_layers", {}) if selected_payload else {}
    discovery = selected_payload.get("discovery", {}) if selected_payload else {}
    positive_counts: Dict[str, int] = {}
    for row in centroid_stats:
        prop = str(row.get("property_name", row.get("property", ""))).lower()
        count = integer(row.get("positive_count"))
        if prop and count is not None:
            positive_counts[prop] = max(positive_counts.get(prop, 0), count)
    return [{
        "property": prop,
        "positive_instances": positive_counts.get(prop),
        "selected_layers": selected.get(prop, []),
        "representation_found": discovery.get(prop, {}).get("distinguishable_representation_found"),
    } for prop in PROPERTIES]


def load_training_histories(training_root: Path, overrides: Mapping[str, Optional[Path]]) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    histories: Dict[str, List[Dict[str, Any]]] = {}
    validation: Dict[str, Any] = {}
    for prop in PROPERTIES:
        directory = overrides.get(prop) or training_root / f"lora_{prop}_v4_1536"
        rows = read_csv(directory / "training_history.csv")
        if not rows:
            raw = read_json(directory / "training_history.json") or []
            rows = [dict(item) for item in raw]
        normalized: List[Dict[str, Any]] = []
        for row in rows:
            normalized.append({
                "property": prop,
                "epoch": integer(row.get("epoch")),
                "split": str(row.get("split", "")),
                "instance_count": integer(row.get("instance_count")),
                "positive_instance_count": integer(row.get("positive_instance_count")),
                "total_loss": number(row.get("mean_total_loss", row.get("total_loss"))),
                "language_loss": number(row.get("mean_language_loss", row.get("language_loss"))),
                "activation_loss": number(row.get("mean_activation_loss", row.get("activation_loss"))),
                "activation_distance": number(row.get("mean_selected_layer_distance", row.get("activation_distance"))),
                "learning_rate": number(row.get("learning_rate")),
                "elapsed_seconds": number(row.get("elapsed_seconds")),
            })
        histories[prop] = normalized
        validation[prop] = read_json(directory / "validation_summary.json")
    return histories, validation


def load_steering_training_artifacts(
    training_root: Path,
    overrides: Mapping[str, Optional[Path]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Load learned activation-steering optimization histories and metadata."""
    histories: Dict[str, List[Dict[str, Any]]] = {}
    validation: Dict[str, Any] = {}
    coefficients: Dict[str, Any] = {}
    configs: Dict[str, Any] = {}
    for prop in PROPERTIES:
        directory = overrides.get(prop) or training_root / f"steering_{prop}"
        rows = read_csv(directory / "training_history.csv")
        normalized: List[Dict[str, Any]] = []
        for row in rows:
            normalized.append({
                "property": prop,
                "epoch": integer(row.get("epoch")),
                "split": str(row.get("split", "")),
                "instance_count": integer(row.get("instance_count")),
                "positive_instance_count": integer(row.get("positive_instance_count")),
                "optimization_steps": integer(row.get("optimization_steps")),
                "total_loss": number(row.get("mean_total_loss", row.get("total_loss"))),
                "language_loss": number(row.get("mean_language_loss", row.get("language_loss"))),
                "activation_loss": number(row.get("mean_activation_loss", row.get("activation_loss"))),
                "activation_distance": number(row.get("mean_selected_layer_distance", row.get("activation_distance"))),
                "learning_rate": number(row.get("learning_rate")),
                "elapsed_seconds": number(row.get("elapsed_seconds")),
            })
        histories[prop] = normalized
        validation[prop] = read_json(directory / "validation_summary.json")
        coefficients[prop] = read_json(directory / "steering_coefficients.json")
        configs[prop] = read_json(directory / "training_config.json")
    return histories, validation, coefficients, configs


def flatten_steering_coefficients(payloads: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Normalize property-specific learned steering coefficients for paper-ready output."""
    rows: List[Dict[str, Any]] = []
    for prop in PROPERTIES:
        payload = payloads.get(prop) or {}
        selected_layers = payload.get("selected_layers") or []
        coeffs = payload.get("learned_coefficients", payload.get("steering_coefficients", payload.get("coefficients", {})))
        if isinstance(coeffs, Mapping):
            for layer in selected_layers or coeffs.keys():
                value = coeffs.get(str(layer), coeffs.get(layer))
                if number(value) is not None:
                    rows.append({"property": prop, "layer": integer(layer), "learned_alpha": number(value)})
        elif isinstance(coeffs, Sequence) and not isinstance(coeffs, (str, bytes)):
            for layer, value in zip(selected_layers, coeffs):
                if number(value) is not None:
                    rows.append({"property": prop, "layer": integer(layer), "learned_alpha": number(value)})
    return rows


def build_final_summary(
    dataset_stats: Optional[Mapping[str, Any]],
    selected_payload: Optional[Mapping[str, Any]],
    training_histories: Mapping[str, Sequence[Mapping[str, Any]]],
    steering_training_histories: Mapping[str, Sequence[Mapping[str, Any]]],
    steering_validation_summaries: Mapping[str, Any],
    steering_coefficients: Mapping[str, Any],
    steering_training_configs: Mapping[str, Any],
    permanent_evaluation_summary: Optional[Mapping[str, Any]],
    steering_evaluation_summary: Optional[Mapping[str, Any]],
    permanent_rows: Sequence[Mapping[str, Any]],
    steering_rows: Sequence[Mapping[str, Any]],
    missing_inputs: Sequence[str],
) -> Dict[str, Any]:
    permanent_lookup = {str(row.get("property", "")).lower(): row for row in permanent_rows}
    steering_lookup = {str(row.get("property", "")).lower(): row for row in steering_rows}
    selected = (selected_payload or {}).get("selected_layers", {})
    discovery = (selected_payload or {}).get("discovery", {})
    result: Dict[str, Any] = {
        "dataset": dataset_stats,
        "properties": {},
        "interpretation": {
            "permanent": ((permanent_evaluation_summary or {}).get("summary") or {}).get("interpretation"),
            "steering": ((steering_evaluation_summary or {}).get("summary") or {}).get("interpretation"),
        },
        "missing_inputs": list(missing_inputs),
    }
    for prop in PROPERTIES:
        history = list(training_histories.get(prop, []))
        train_rows = [row for row in history if str(row.get("split", "")).lower() == "train"]
        validation_rows = [row for row in history if str(row.get("split", "")).lower() == "validation"]
        decision = permanent_lookup.get(prop, {})
        steering_decision = steering_lookup.get(prop, {})
        result["properties"][prop] = {
            "representation_found": discovery.get(prop, {}).get("distinguishable_representation_found"),
            "selected_layers": selected.get(prop),
            "adaptation_epochs": max((integer(row.get("epoch")) or 0 for row in history), default=0),
            "training_start": train_rows[0] if train_rows else None,
            "training_final": train_rows[-1] if train_rows else None,
            "validation_start": validation_rows[0] if validation_rows else None,
            "validation_final": validation_rows[-1] if validation_rows else None,
            "steering_training": {
                "epochs": max((integer(row.get("epoch")) or 0 for row in steering_training_histories.get(prop, [])), default=0),
                "history": list(steering_training_histories.get(prop, [])),
                "validation_summary": steering_validation_summaries.get(prop),
                "coefficients": steering_coefficients.get(prop),
                "training_config": steering_training_configs.get(prop),
            },
            "comparative_evaluation": {
                "pretrained": {
                    "output_positive_rate": number(decision.get("output_positive_rate_pretrained", steering_decision.get("output_positive_rate_pretrained"))),
                    "centroid_distance": number(decision.get("centroid_distance_pretrained", steering_decision.get("centroid_distance_pretrained"))),
                    "cosine_similarity": number(decision.get("cosine_similarity_pretrained", steering_decision.get("cosine_similarity_pretrained"))),
                    "reference_patch_nll": number(decision.get("reference_patch_nll_pretrained", steering_decision.get("reference_patch_nll_pretrained"))),
                },
                "activation_steering": dict(steering_decision) if steering_decision else None,
                "permanent_adaptation": dict(decision) if decision else None,
            },
            "post_merge_evaluation": dict(decision) if decision else None,
            "activation_steering_supported": boolean(steering_decision.get("activation_steering_supported")) if steering_decision else None,
            "permanent_adaptation_supported": boolean(decision.get("permanent_adaptation_supported")) if decision else None,
        }
    result["all_permanent_properties_confirmed"] = all(
        result["properties"][prop]["permanent_adaptation_supported"] is True
        for prop in PROPERTIES
    ) if all(result["properties"][prop]["permanent_adaptation_supported"] is not None for prop in PROPERTIES) else None
    result["all_steering_properties_confirmed"] = all(
        result["properties"][prop]["activation_steering_supported"] is True
        for prop in PROPERTIES
    ) if all(result["properties"][prop]["activation_steering_supported"] is not None for prop in PROPERTIES) else None
    return result


def steering_rows_from_summary(payload: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summary = (payload or {}).get("summary", {}) or {}
    rows: List[Dict[str, Any]] = []
    for prop in PROPERTIES:
        item = summary.get(prop)
        if isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def comparative_rows(permanent_rows: Sequence[Mapping[str, Any]], steering_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    pmap = {str(r.get("property", "")).lower(): r for r in permanent_rows}
    smap = {str(r.get("property", "")).lower(): r for r in steering_rows}
    rows: List[Dict[str, Any]] = []
    for prop in PROPERTIES:
        p, st = pmap.get(prop, {}), smap.get(prop, {})
        rows.append({
            "property": prop,
            "output_positive_rate_pretrained": number(p.get("output_positive_rate_pretrained", st.get("output_positive_rate_pretrained"))),
            "output_positive_rate_steering": number(st.get("output_positive_rate_steering")),
            "output_positive_rate_permanent": number(p.get("output_positive_rate_merged")),
            "centroid_distance_pretrained": number(p.get("centroid_distance_pretrained", st.get("centroid_distance_pretrained"))),
            "centroid_distance_steering": number(st.get("centroid_distance_steering")),
            "centroid_distance_permanent": number(p.get("centroid_distance_merged")),
            "cosine_similarity_pretrained": number(p.get("cosine_similarity_pretrained", st.get("cosine_similarity_pretrained"))),
            "cosine_similarity_steering": number(st.get("cosine_similarity_steering")),
            "cosine_similarity_permanent": number(p.get("cosine_similarity_merged")),
            "relative_nll_change_steering": number(st.get("relative_nll_change")),
            "relative_nll_change_permanent": number(p.get("relative_nll_change")),
            "steering_supported": boolean(st.get("activation_steering_supported")),
            "permanent_supported": boolean(p.get("permanent_adaptation_supported")),
        })
    return rows


def parseability_rows(permanent_summary: Optional[Mapping[str, Any]], steering_summary: Optional[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    pb = ((permanent_summary or {}).get("baseline") or {}).get("output_metrics", {}) or {}
    sb = ((steering_summary or {}).get("baseline") or {}).get("output_metrics", {}) or {}
    base_rate = number(pb.get("parseable_output_rate", sb.get("parseable_output_rate")))
    adapted = (permanent_summary or {}).get("adapted_models", {}) or {}
    steering = (steering_summary or {}).get("steering_models", {}) or {}
    return [{
        "property": prop,
        "pretrained_parseable_output_rate": base_rate,
        "steering_parseable_output_rate": number(((steering.get(prop) or {}).get("output_metrics") or {}).get("parseable_output_rate")),
        "permanent_parseable_output_rate": number(((adapted.get(prop) or {}).get("output_metrics") or {}).get("parseable_output_rate")),
    } for prop in PROPERTIES]


def generate_figures(
    figure_dir: Path,
    histories: Mapping[str, Sequence[Mapping[str, Any]]],
    steering_histories: Mapping[str, Sequence[Mapping[str, Any]]],
    permanent_rows: Sequence[Mapping[str, Any]],
    steering_rows: Sequence[Mapping[str, Any]],
    dpi: int,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib is unavailable; figures were not generated.")
        return []

    figure_dir.mkdir(parents=True, exist_ok=True)
    created: List[str] = []

    # One figure per metric, no subplots.
    metric_specs = [
        ("total_loss", "Total loss", "training_total_loss"),
        ("language_loss", "Language-modeling loss", "training_language_loss"),
        ("activation_loss", "Activation-guidance loss", "training_activation_loss"),
        ("activation_distance", "Activation distance", "training_activation_distance"),
    ]
    for metric, ylabel, stem in metric_specs:
        has_data = False
        fig = plt.figure()
        for prop in PROPERTIES:
            rows = [row for row in histories.get(prop, []) if str(row.get("split", "")).lower() == "validation"]
            xs = [integer(row.get("epoch")) for row in rows]
            ys = [number(row.get(metric)) for row in rows]
            pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
            if pairs:
                has_data = True
                plt.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=PROPERTY_LABELS[prop])
        if has_data:
            plt.xlabel("Adaptation epoch")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            for suffix in ("pdf", "png"):
                path = figure_dir / f"{stem}.{suffix}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                created.append(str(path))
        plt.close(fig)

    # Learned activation-steering optimization histories.
    steering_metric_specs = [
        ("total_loss", "Total loss", "steering_training_total_loss"),
        ("language_loss", "Language-modeling loss", "steering_training_language_loss"),
        ("activation_loss", "Activation-guidance loss", "steering_training_activation_loss"),
        ("activation_distance", "Activation distance", "steering_training_activation_distance"),
    ]
    for metric, ylabel, stem in steering_metric_specs:
        has_data = False
        fig = plt.figure()
        for prop in PROPERTIES:
            rows = [row for row in steering_histories.get(prop, []) if str(row.get("split", "")).lower() == "validation"]
            pairs = [(integer(row.get("epoch")), number(row.get(metric))) for row in rows]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if pairs:
                has_data = True
                plt.plot([p[0] for p in pairs], [p[1] for p in pairs], marker="o", label=PROPERTY_LABELS[prop])
        if has_data:
            plt.xlabel("Steering optimization epoch")
            plt.ylabel(ylabel)
            plt.legend()
            plt.tight_layout()
            for suffix in ("pdf", "png"):
                path = figure_dir / f"{stem}.{suffix}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                created.append(str(path))
        plt.close(fig)

    if permanent_rows and steering_rows:
        steering_map = {str(item.get("property", "")).lower(): item for item in steering_rows}
        labels, baseline, steering_vals, merged = [], [], [], []
        for prop in PROPERTIES:
            row = next((item for item in permanent_rows if str(item.get("property", "")).lower() == prop), None)
            if row:
                b = number(row.get("output_positive_rate_pretrained"))
                st = number((steering_map.get(prop) or {}).get("output_positive_rate_steering"))
                m = number(row.get("output_positive_rate_merged"))
                if b is not None and st is not None and m is not None:
                    labels.append(PROPERTY_LABELS[prop]); baseline.append(b * 100); steering_vals.append(st * 100); merged.append(m * 100)
        if labels:
            import numpy as np
            x = np.arange(len(labels)); width = 0.25
            fig = plt.figure()
            plt.bar(x - width, baseline, width, label="Pretrained")
            plt.bar(x, steering_vals, width, label="Activation Steering")
            plt.bar(x + width, merged, width, label="Permanent Adaptation")
            plt.xticks(x, labels)
            plt.ylabel("Property-positive outputs (%)")
            plt.legend(); plt.tight_layout()
            for suffix in ("pdf", "png"):
                path = figure_dir / f"output_positive_rates.{suffix}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight"); created.append(str(path))
            plt.close(fig)

        labels, base_dist, steering_dist, merged_dist = [], [], [], []
        for prop in PROPERTIES:
            row = next((item for item in permanent_rows if str(item.get("property", "")).lower() == prop), None)
            if row:
                b = number(row.get("centroid_distance_pretrained")); st = number((steering_map.get(prop) or {}).get("centroid_distance_steering")); m = number(row.get("centroid_distance_merged"))
                if b is not None and st is not None and m is not None:
                    labels.append(PROPERTY_LABELS[prop]); base_dist.append(b); steering_dist.append(st); merged_dist.append(m)
        if labels:
            import numpy as np
            x = np.arange(len(labels)); width = 0.25
            fig = plt.figure()
            plt.bar(x - width, base_dist, width, label="Pretrained")
            plt.bar(x, steering_dist, width, label="Activation Steering")
            plt.bar(x + width, merged_dist, width, label="Permanent Adaptation")
            plt.xticks(x, labels); plt.ylabel("Mean centroid distance")
            plt.legend(); plt.tight_layout()
            for suffix in ("pdf", "png"):
                path = figure_dir / f"activation_centroid_distance.{suffix}"
                fig.savefig(path, dpi=dpi, bbox_inches="tight"); created.append(str(path))
            plt.close(fig)

    return created


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate paper-ready summaries from the activation-guided adaptation pipeline.")
    parser.add_argument("--prepared-dir", type=Path, default=Path("outputs/prepared_swebench"))
    parser.add_argument("--labeling-statistics", type=Path, default=None)
    parser.add_argument("--discovery-dir", type=Path, default=Path("outputs/activation_centroids"))
    parser.add_argument("--training-root", type=Path, default=Path("outputs"))
    parser.add_argument("--complexity-dir", type=Path, default=None)
    parser.add_argument("--coupling-dir", type=Path, default=None)
    parser.add_argument("--modularity-dir", type=Path, default=None)
    parser.add_argument("--steering-complexity-dir", type=Path, default=None)
    parser.add_argument("--steering-coupling-dir", type=Path, default=None)
    parser.add_argument("--steering-modularity-dir", type=Path, default=None)
    parser.add_argument("--permanent-evaluation-dir", "--evaluation-dir", dest="permanent_evaluation_dir", type=Path, default=Path("outputs/activation_guided_evaluation"))
    parser.add_argument("--steering-evaluation-dir", type=Path, default=Path("outputs/activation_steering_evaluation"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_results"))
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Fail when a core input group is missing.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    configure_logging(args.verbose)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output directory is not empty: {args.output_dir}. Use --overwrite.")
        shutil.rmtree(args.output_dir)
    csv_dir = args.output_dir / "tables" / "csv"
    tex_dir = args.output_dir / "tables" / "latex"
    fig_dir = args.output_dir / "figures"
    for directory in (csv_dir, tex_dir, fig_dir):
        directory.mkdir(parents=True, exist_ok=True)

    missing: List[str] = []
    stats_path = resolve_labeling_statistics(args.prepared_dir, args.labeling_statistics)
    dataset_stats = read_json(stats_path) if stats_path else None
    if dataset_stats is None:
        missing.append("labeling_statistics.json")

    selected_payload = read_json(args.discovery_dir / "selected_layers.json")
    layer_analysis = read_csv(args.discovery_dir / "layer_analysis.csv")
    centroid_stats = read_json(args.discovery_dir / "centroid_statistics.json") or []
    if selected_payload is None:
        missing.append("selected_layers.json")
    if not layer_analysis:
        missing.append("layer_analysis.csv")

    overrides = {"complexity": args.complexity_dir, "coupling": args.coupling_dir, "modularity": args.modularity_dir}
    histories, validation_summaries = load_training_histories(args.training_root, overrides)
    for prop in PROPERTIES:
        if not histories[prop]:
            missing.append(f"{prop} training_history")

    steering_overrides = {
        "complexity": args.steering_complexity_dir,
        "coupling": args.steering_coupling_dir,
        "modularity": args.steering_modularity_dir,
    }
    steering_histories, steering_validation_summaries, steering_coefficients, steering_training_configs = load_steering_training_artifacts(
        args.training_root, steering_overrides
    )
    for prop in PROPERTIES:
        if not steering_histories[prop]:
            missing.append(f"{prop} steering training_history")
        if not steering_coefficients.get(prop):
            missing.append(f"{prop} steering_coefficients.json")

    permanent_evaluation_summary = read_json(args.permanent_evaluation_dir / "evaluation_summary.json")
    permanent_rows = read_csv(args.permanent_evaluation_dir / "permanent_adaptation_summary.csv")
    output_rows = read_csv(args.permanent_evaluation_dir / "output_level_results.csv")
    activation_rows = read_csv(args.permanent_evaluation_dir / "activation_level_results.csv")
    capability_rows = read_csv(args.permanent_evaluation_dir / "capability_preservation.csv")

    steering_evaluation_summary = read_json(args.steering_evaluation_dir / "evaluation_summary.json")
    steering_rows = read_csv(args.steering_evaluation_dir / "activation_steering_summary.csv")
    if not steering_rows:
        steering_rows = steering_rows_from_summary(steering_evaluation_summary)
    if not steering_rows:
        missing.append("activation_steering_summary.csv/evaluation_summary.json")
    if not permanent_rows:
        missing.append("permanent_adaptation_summary.csv")

    if args.strict and missing:
        raise FileNotFoundError("Missing required result artifacts: " + ", ".join(missing))

    # Table 1 and 2: prepared dataset.
    if dataset_stats:
        dataset_rows, distribution_rows = dataset_tables(dataset_stats)
        write_csv(csv_dir / "table1_dataset_summary.csv", dataset_rows)
        write_csv(csv_dir / "table2_property_distribution.csv", distribution_rows)
        table1_rows = []
        for row in dataset_rows:
            value = row.get("value")
            if row.get("rate") is not None and integer(value) is not None:
                rendered = f"{fmt_int(value)} ({percent(row['rate'])})"
            elif isinstance(value, float) and not float(value).is_integer():
                rendered = fmt(value, 2)
            else:
                rendered = fmt_int(value)
            table1_rows.append([latex_escape(row["statistic"]), rendered])
        save_latex(tex_dir / "table1_dataset_summary.tex", latex_table(
            "Summary of the prepared SWE-bench adaptation dataset after deterministic structural property labeling.",
            "tab:prepared_dataset_summary", ["Statistic", "Value"], table1_rows, "p{0.68\\linewidth}p{0.22\\linewidth}",
            font_size=r"\footnotesize", tabcolsep="5pt", arraystretch="0.92"))
        table2_rows = [[latex_escape(row["combination"]), f"{fmt_int(row['instances'])} ({percent(row['rate'])})"] for row in distribution_rows]
        save_latex(tex_dir / "table2_property_distribution.tex", latex_table(
            "Distribution of structural property combinations in the prepared SWE-bench dataset.",
            "tab:property_distribution", ["Structural Property Combination", "Instances (\\%)"], table2_rows,
            "p{0.68\\linewidth}p{0.22\\linewidth}", font_size=r"\footnotesize", tabcolsep="5pt", arraystretch="0.92"))

    # Table 3 and 4: discovery.
    selected_rows = select_layer_rows(layer_analysis, selected_payload or {})
    if selected_rows:
        write_csv(csv_dir / "table3_selected_layer_performance.csv", selected_rows)
        tex_rows, breaks, previous_prop = [], [], None
        for row in selected_rows:
            prop = row["property"]
            if previous_prop is not None and prop != previous_prop:
                breaks.append(len(tex_rows))
            tex_rows.append([
                PROPERTY_LABELS[prop] if prop != previous_prop else "",
                row["layer"], fmt(row["balanced_accuracy"]), fmt(row["roc_auc"]),
                fmt(row["bounded_fisher_ratio"]), fmt(row["score"]), row["rank"],
            ])
            previous_prop = prop
        breaks.append(len(tex_rows))
        save_latex(tex_dir / "table3_selected_layer_performance.tex", latex_table(
            "Performance of the selected transformer layers during activation representation discovery. The reported score is the combined layer-selection criterion used for ranking candidate layers.",
            "tab:selected_layer_performance", ["Property", "Layer", "Bal. Acc.", "ROC AUC", "Bound. Fisher", "Score", "Rank"],
            tex_rows, "llccccc", table_star=True, group_breaks=breaks))

    discovery_summary_rows = discovery_rows(selected_payload or {}, centroid_stats)
    if selected_payload:
        write_csv(csv_dir / "table4_activation_discovery_summary.csv", discovery_summary_rows)
        tex_rows = [[PROPERTY_LABELS[row["property"]], fmt_int(row["positive_instances"]), ", ".join(str(x) for x in row["selected_layers"]), yes_no(row["representation_found"])] for row in discovery_summary_rows]
        save_latex(tex_dir / "table4_activation_discovery_summary.tex", latex_table(
            "Summary of activation representation discovery on the SWE-bench adaptation set.",
            "tab:activation_discovery_summary", ["Property", "Positive Instances", "Selected Layers", "Representation Found"],
            tex_rows, "lccc", table_star=True))

    # Training tables: one per property, only when data are present.
    all_training_rows: List[Dict[str, Any]] = []
    for prop in PROPERTIES:
        rows = histories[prop]
        if not rows:
            continue
        all_training_rows.extend(rows)
        tex_rows = [[row["epoch"], "Validation" if row["split"].lower() == "validation" else "Train", fmt(row["total_loss"], 3), fmt(row["language_loss"], 3), fmt(row["activation_loss"], 3), fmt(row["activation_distance"], 3)] for row in rows]
        epoch_counts = Counter(row["epoch"] for row in rows)
        running = 0; breaks = []
        for epoch in sorted(epoch_counts):
            running += epoch_counts[epoch]; breaks.append(running)
        save_latex(tex_dir / f"table5_{prop}_training.tex", latex_table(
            f"Training and validation performance during activation-guided LoRA adaptation for the \\textit{{{prop}}} property.",
            f"tab:{prop}_lora_training", ["Epoch", "Split", "Total", "LM Loss", "Act. Loss", "Act. Dist."],
            tex_rows, "cccccc", group_breaks=breaks))
    write_csv(csv_dir / "training_histories_all_properties.csv", all_training_rows)

    # Learned activation-steering training evidence: histories and final learned alpha values.
    all_steering_training_rows: List[Dict[str, Any]] = []
    for prop in PROPERTIES:
        rows = steering_histories[prop]
        if not rows:
            continue
        all_steering_training_rows.extend(rows)
        write_csv(csv_dir / f"table10a_{prop}_steering_training.csv", rows)
        tex_rows = [[row["epoch"], "Validation" if row["split"].lower() == "validation" else "Train", fmt(row["total_loss"], 3), fmt(row["language_loss"], 3), fmt(row["activation_loss"], 3), fmt(row["activation_distance"], 3)] for row in rows]
        epoch_counts = Counter(row["epoch"] for row in rows)
        running = 0; breaks = []
        for epoch in sorted(epoch_counts):
            running += epoch_counts[epoch]; breaks.append(running)
        save_latex(tex_dir / f"table10a_{prop}_steering_training.tex", latex_table(
            f"Training and validation performance while learning inference-time activation-steering coefficients for the \\textit{{{prop}}} property.",
            f"tab:{prop}_steering_training", ["Epoch", "Split", "Total", "LM Loss", "Act. Loss", "Act. Dist."],
            tex_rows, "cccccc", group_breaks=breaks))
    write_csv(csv_dir / "steering_training_histories_all_properties.csv", all_steering_training_rows)

    coefficient_rows = flatten_steering_coefficients(steering_coefficients)
    if coefficient_rows:
        write_csv(csv_dir / "table10b_steering_coefficients.csv", coefficient_rows)
        tex_rows = [[PROPERTY_LABELS[row["property"]], row["layer"], fmt(row["learned_alpha"], 5)] for row in coefficient_rows]
        save_latex(tex_dir / "table10b_steering_coefficients.tex", latex_table(
            "Learned layer-specific coefficients for the inference-time activation-steering baseline.",
            "tab:steering_coefficients", ["Property", "Layer", "Learned $\\alpha$"], tex_rows, "lcc", table_star=False))

    # Post-merge evaluation tables.
    if output_rows:
        write_csv(csv_dir / "table6_output_level_evaluation.csv", output_rows)
    if activation_rows:
        write_csv(csv_dir / "table7_activation_level_evaluation.csv", activation_rows)
    if capability_rows:
        write_csv(csv_dir / "table8_capability_preservation.csv", capability_rows)
    if permanent_rows:
        write_csv(csv_dir / "table9_permanent_adaptation_summary.csv", permanent_rows)
        tex_rows = []
        for prop in PROPERTIES:
            row = next((item for item in permanent_rows if str(item.get("property", "")).lower() == prop), None)
            if not row:
                continue
            tex_rows.append([
                PROPERTY_LABELS[prop], percent(row.get("output_positive_rate_pretrained")), percent(row.get("output_positive_rate_merged")),
                fmt(row.get("centroid_distance_pretrained"), 3), fmt(row.get("centroid_distance_merged"), 3),
                yes_no(row.get("capability_preserved")), yes_no(row.get("permanent_adaptation_supported")),
            ])
        save_latex(tex_dir / "table9_permanent_adaptation_summary.tex", latex_table(
            "Post-merge evidence for permanent activation-guided adaptation.",
            "tab:permanent_adaptation_summary", ["Property", "Base Pos. (\\%)", "Merged Pos. (\\%)", "Base Dist.", "Merged Dist.", "Capability Preserved", "Permanent Adaptation"],
            tex_rows, "lcccccc", table_star=True))

    # Comparative baseline tables.
    if steering_rows:
        write_csv(csv_dir / "table10_activation_steering_summary.csv", steering_rows)
        tex_rows = []
        for prop in PROPERTIES:
            row = next((item for item in steering_rows if str(item.get("property", "")).lower() == prop), None)
            if row:
                tex_rows.append([
                    PROPERTY_LABELS[prop], percent(row.get("output_positive_rate_pretrained")), percent(row.get("output_positive_rate_steering")),
                    fmt(row.get("centroid_distance_pretrained"), 3), fmt(row.get("centroid_distance_steering"), 3),
                    yes_no(row.get("capability_preserved")), yes_no(row.get("activation_steering_supported")),
                ])
        save_latex(tex_dir / "table10_activation_steering_summary.tex", latex_table(
            "Evidence for the learned inference-time activation-steering comparative baseline.",
            "tab:activation_steering_summary", ["Property", "Base Pos. (\\%)", "Steering Pos. (\\%)", "Base Dist.", "Steering Dist.", "Capability Preserved", "Steering Supported"],
            tex_rows, "lcccccc", table_star=True))

    comparison = comparative_rows(permanent_rows, steering_rows) if permanent_rows and steering_rows else []
    if comparison:
        write_csv(csv_dir / "table11_comparative_evaluation_summary.csv", comparison)
        tex_rows = []
        for row in comparison:
            tex_rows.extend([
                [PROPERTY_LABELS[row["property"]], "Output positive rate", percent(row["output_positive_rate_pretrained"]), percent(row["output_positive_rate_steering"]), percent(row["output_positive_rate_permanent"])],
                ["", "Centroid distance", fmt(row["centroid_distance_pretrained"], 3), fmt(row["centroid_distance_steering"], 3), fmt(row["centroid_distance_permanent"], 3)],
                ["", "Cosine similarity", fmt(row["cosine_similarity_pretrained"], 3), fmt(row["cosine_similarity_steering"], 3), fmt(row["cosine_similarity_permanent"], 3)],
                ["", "Relative NLL change", "--", percent(row["relative_nll_change_steering"], 2), percent(row["relative_nll_change_permanent"], 2)],
                ["", "Overall supported", "--", yes_no(row["steering_supported"]), yes_no(row["permanent_supported"])],
            ])
        save_latex(tex_dir / "table11_comparative_evaluation_summary.tex", latex_table(
            "Controlled comparison of pretrained behavior, learned inference-time activation steering, and permanent activation-guided parameter adaptation.",
            "tab:comparative_evaluation_summary", ["Property", "Metric", "Pretrained", "Activation Steering", "Permanent Adaptation"],
            tex_rows, "llccc", table_star=True, group_breaks=[5, 10, 15]))

    parse_rows = parseability_rows(permanent_evaluation_summary, steering_evaluation_summary)
    if any(row.get("steering_parseable_output_rate") is not None or row.get("permanent_parseable_output_rate") is not None for row in parse_rows):
        write_csv(csv_dir / "table12_parseable_output_comparison.csv", parse_rows)
        tex_rows = [[PROPERTY_LABELS[r["property"]], percent(r["pretrained_parseable_output_rate"]), percent(r["steering_parseable_output_rate"]), percent(r["permanent_parseable_output_rate"])] for r in parse_rows]
        save_latex(tex_dir / "table12_parseable_output_comparison.tex", latex_table(
            "Parseable generated-output rates under the three evaluation conditions.",
            "tab:parseable_output_comparison", ["Property", "Pretrained", "Activation Steering", "Permanent Adaptation"],
            tex_rows, "lccc", table_star=True))

    final_summary = build_final_summary(dataset_stats, selected_payload, histories, steering_histories, steering_validation_summaries, steering_coefficients, steering_training_configs, permanent_evaluation_summary, steering_evaluation_summary, permanent_rows, steering_rows, missing)
    final_summary["validation_summaries"] = validation_summaries
    write_json(args.output_dir / "summary.json", final_summary)

    paper_statistics = {
        "dataset": dataset_stats,
        "activation_discovery": discovery_summary_rows if selected_payload else None,
        "selected_layer_performance": selected_rows,
        "training": {prop: histories[prop] for prop in PROPERTIES},
        "steering_training": {prop: steering_histories[prop] for prop in PROPERTIES},
        "steering_validation_summaries": steering_validation_summaries,
        "steering_coefficients": steering_coefficients,
        "steering_training_configs": steering_training_configs,
        "steering_coefficient_table": coefficient_rows,
        "post_merge_evaluation": permanent_rows,
        "activation_steering_evaluation": steering_rows,
        "comparative_evaluation": comparison,
        "parseable_output_comparison": parse_rows,
    }
    write_json(args.output_dir / "paper_statistics.json", paper_statistics)

    figure_paths = generate_figures(fig_dir, histories, steering_histories, permanent_rows, steering_rows, args.dpi)
    final_summary["generated_figures"] = figure_paths
    write_json(args.output_dir / "summary.json", final_summary)

    report_lines = ["ACTIVATION-GUIDED ADAPTATION — COMPARATIVE EVIDENCE SUMMARY", "=" * 64, ""]
    for prop in PROPERTIES:
        item = final_summary["properties"][prop]
        report_lines.extend([
            PROPERTY_LABELS[prop], "-" * len(PROPERTY_LABELS[prop]),
            f"Representation discovered : {yes_no(item.get('representation_found')).upper()}",
            f"Selected layers           : {', '.join(map(str, item.get('selected_layers') or [])) or '--'}",
            f"Adaptation epochs         : {item.get('adaptation_epochs', 0)}",
            f"Steering epochs           : {(item.get('steering_training') or {}).get('epochs', 0)}",
            f"Steering supported        : {yes_no(item.get('activation_steering_supported')).upper()}",
            f"Permanent adaptation      : {yes_no(item.get('permanent_adaptation_supported')).upper()}", "",
        ])
    report_lines.append(f"All steering properties supported  : {yes_no(final_summary.get('all_steering_properties_confirmed')).upper()}")
    report_lines.append(f"All permanent properties supported : {yes_no(final_summary.get('all_permanent_properties_confirmed')).upper()}")
    if missing:
        report_lines.extend(["", "Missing inputs:"] + [f"- {item}" for item in missing])
    (args.output_dir / "evidence_report.txt").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    LOGGER.info("Summary completed: %s", args.output_dir)
    LOGGER.info("Missing optional artifacts: %d", len(missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
