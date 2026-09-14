#!/usr/bin/env python3
"""
prepare_dataset_and_labels.py

Prepare SWE-bench for the permanent activation-guided adaptation experiment.

This script combines two stages:
  1. Dataset loading, prompt construction, tokenization, and partitioning.
  2. Deterministic binary labeling for coupling, complexity, and modularity.

SWE-bench supplies a gold unified diff rather than separate complete buggy and
patched functions. The script therefore reconstructs before/after Python code
fragments from diff hunks and applies conservative structural heuristics.
These labels are operational experimental labels, not semantic ground truth.

Example:
  python prepare_dataset_and_labels.py \
      --dataset-name princeton-nlp/SWE-bench \
      --split test \
      --model-name bigcode/starcoderbase-1b \
      --output-dir outputs/prepared_swebench

Outputs:
  prepared_dataset.pt
  instances.jsonl
  property_labels.csv
  labeling_statistics.json
  split_manifest.json
  preprocessing_config.json
  rejected_instances.jsonl
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
import random
import re
import textwrap
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

LOGGER = logging.getLogger("prepare_dataset_and_labels")


@dataclass(frozen=True)
class StructuralMetrics:
    parse_succeeded: bool
    imported_modules: int
    external_calls: int
    dependency_references: int
    fan_out: int
    cyclomatic_complexity: int
    conditional_branches: int
    maximum_nesting_depth: int
    control_flow_constructs: int
    function_count: int
    helper_function_count: int
    top_level_statement_count: int
    responsibility_tokens: int


@dataclass(frozen=True)
class PropertyEvidence:
    label: int
    criteria_met: Tuple[str, ...]


@dataclass(frozen=True)
class InstanceLabels:
    coupling: PropertyEvidence
    complexity: PropertyEvidence
    modularity: PropertyEvidence


@dataclass(frozen=True)
class DiffFragment:
    file_path: str
    hunk_header: str
    before_code: str
    after_code: str


@dataclass(frozen=True)
class PreparedRecord:
    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    gold_patch: str
    test_patch: str
    hints_text: str
    created_at: str
    version: str
    prompt: str
    coupling_label: int
    complexity_label: int
    modularity_label: int
    coupling_criteria: Tuple[str, ...]
    complexity_criteria: Tuple[str, ...]
    modularity_criteria: Tuple[str, ...]
    fragment_count: int
    parseable_fragment_pairs: int


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item) for item in value)
    return str(value)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


DIFF_FILE_RE = re.compile(r"^\+\+\+\s+(?:b/)?(.+)$")
HUNK_RE = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@")


def iter_diff_fragments(patch: str) -> Iterator[DiffFragment]:
    """Yield reconstructed before/after fragments for Python diff hunks."""
    current_file = ""
    hunk_header = ""
    before_lines: List[str] = []
    after_lines: List[str] = []
    in_hunk = False

    def flush() -> Optional[DiffFragment]:
        nonlocal before_lines, after_lines, hunk_header, in_hunk
        if not in_hunk:
            return None
        result = DiffFragment(
            file_path=current_file,
            hunk_header=hunk_header,
            before_code="\n".join(before_lines).strip("\n"),
            after_code="\n".join(after_lines).strip("\n"),
        )
        before_lines, after_lines = [], []
        hunk_header = ""
        in_hunk = False
        return result

    for line in patch.splitlines():
        file_match = DIFF_FILE_RE.match(line)
        if file_match:
            pending = flush()
            if pending and pending.file_path.endswith(".py"):
                yield pending
            current_file = file_match.group(1)
            continue

        if HUNK_RE.match(line):
            pending = flush()
            if pending and pending.file_path.endswith(".py"):
                yield pending
            in_hunk = True
            hunk_header = line
            continue

        if not in_hunk or line.startswith("\\ No newline"):
            continue
        if line.startswith("+") and not line.startswith("+++"):
            after_lines.append(line[1:])
        elif line.startswith("-") and not line.startswith("---"):
            before_lines.append(line[1:])
        elif line.startswith(" "):
            before_lines.append(line[1:])
            after_lines.append(line[1:])
        elif line.startswith("diff --git ") or line.startswith("--- "):
            pending = flush()
            if pending and pending.file_path.endswith(".py"):
                yield pending

    pending = flush()
    if pending and pending.file_path.endswith(".py"):
        yield pending


BUILTIN_CALLS = set(dir(__builtins__))
BRANCH_TYPES = (
    ast.If, ast.IfExp, ast.For, ast.AsyncFor, ast.While,
    ast.ExceptHandler, ast.comprehension, ast.Match,
)
CONTROL_FLOW_TYPES = (
    ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try,
    ast.With, ast.AsyncWith, ast.Match,
)
NESTING_TYPES = CONTROL_FLOW_TYPES + (
    ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
)
RESPONSIBILITY_RE = re.compile(
    r"(load|save|read|write|parse|format|validate|check|build|create|update|"
    r"delete|remove|fetch|send|render|compute|calculate|convert|transform|"
    r"handle|process|resolve|generate|test|cache|log|serialize|deserialize)",
    re.IGNORECASE,
)


def parse_fragment(code: str) -> Optional[ast.AST]:
    """Parse a possibly incomplete diff fragment with conservative fallbacks."""
    normalized = textwrap.dedent(code).strip()
    if not normalized:
        return ast.parse("")

    candidates = [
        normalized,
        "def __fragment_wrapper__():\n" + textwrap.indent(normalized, "    "),
    ]
    lines = normalized.splitlines()
    for start in range(1, min(len(lines), 20)):
        suffix = "\n".join(lines[start:]).strip()
        if suffix:
            candidates.append(suffix)
            candidates.append(
                "def __fragment_wrapper__():\n" + textwrap.indent(suffix, "    ")
            )

    for candidate in candidates:
        try:
            return ast.parse(candidate)
        except (SyntaxError, ValueError, TypeError):
            pass
    return None


def dotted_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def collect_defined_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


def maximum_nesting_depth(tree: ast.AST) -> int:
    maximum = 0

    def visit(node: ast.AST, depth: int) -> None:
        nonlocal maximum
        current = depth + 1 if isinstance(node, NESTING_TYPES) else depth
        maximum = max(maximum, current)
        for child in ast.iter_child_nodes(node):
            visit(child, current)

    visit(tree, 0)
    return maximum


def cyclomatic_complexity(tree: ast.AST) -> int:
    score = 1
    for node in ast.walk(tree):
        if isinstance(node, BRANCH_TYPES):
            score += 1
        elif isinstance(node, ast.BoolOp):
            score += max(0, len(node.values) - 1)
        elif isinstance(node, ast.Match):
            score += max(0, len(node.cases) - 1)
    return score


def responsibility_token_count(tree: ast.AST) -> int:
    tokens: set[str] = set()
    for node in ast.walk(tree):
        name: Optional[str] = None
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
        elif isinstance(node, ast.Call):
            name = dotted_name(node.func)
        if name:
            tokens.update(m.group(0).lower() for m in RESPONSIBILITY_RE.finditer(name))
    return len(tokens)


def compute_metrics(code: str) -> StructuralMetrics:
    tree = parse_fragment(code)
    if tree is None:
        return StructuralMetrics(False, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    defined_names = collect_defined_names(tree)
    imported_roots: set[str] = set()
    dependencies: set[str] = set()
    external_calls: set[str] = set()
    function_names: List[str] = []
    imported_modules = 0

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules += len(node.names)
            for alias in node.names:
                imported_roots.add(alias.asname or alias.name.split(".")[0])
                dependencies.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            imported_modules += len(node.names)
            if node.module:
                imported_roots.add(node.module.split(".")[0])
                dependencies.add(node.module)
            for alias in node.names:
                imported_roots.add(alias.asname or alias.name)
        elif isinstance(node, ast.Call):
            name = dotted_name(node.func)
            if name:
                root = name.split(".")[0]
                if root not in BUILTIN_CALLS and not (
                    root in defined_names and root not in imported_roots
                ):
                    external_calls.add(name)
                    dependencies.add(root)
        elif isinstance(node, ast.Attribute):
            name = dotted_name(node)
            if name and name.split(".")[0] in imported_roots:
                dependencies.add(name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.append(node.name)

    helper_count = sum(
        1 for name in function_names
        if name.startswith("_") or any(
            token in name.lower() for token in (
                "helper", "util", "parse", "validate", "normalize",
                "convert", "build", "extract", "format", "compute",
            )
        )
    )

    return StructuralMetrics(
        parse_succeeded=True,
        imported_modules=imported_modules,
        external_calls=len(external_calls),
        dependency_references=len(dependencies),
        fan_out=len(dependencies | external_calls),
        cyclomatic_complexity=cyclomatic_complexity(tree),
        conditional_branches=sum(isinstance(n, BRANCH_TYPES) for n in ast.walk(tree)),
        maximum_nesting_depth=maximum_nesting_depth(tree),
        control_flow_constructs=sum(
            isinstance(n, CONTROL_FLOW_TYPES) for n in ast.walk(tree)
        ),
        function_count=len(function_names),
        helper_function_count=helper_count,
        top_level_statement_count=len(getattr(tree, "body", [])),
        responsibility_tokens=responsibility_token_count(tree),
    )


def label_coupling(before: StructuralMetrics, after: StructuralMetrics) -> PropertyEvidence:
    criteria = []
    if after.imported_modules < before.imported_modules:
        criteria.append("reduced_imported_modules")
    if after.external_calls < before.external_calls:
        criteria.append("reduced_external_calls")
    if after.dependency_references < before.dependency_references:
        criteria.append("reduced_dependency_references")
    if after.fan_out < before.fan_out:
        criteria.append("reduced_fan_out")
    return PropertyEvidence(int(bool(criteria)), tuple(criteria))


def label_complexity(before: StructuralMetrics, after: StructuralMetrics) -> PropertyEvidence:
    criteria = []
    if after.cyclomatic_complexity < before.cyclomatic_complexity:
        criteria.append("reduced_cyclomatic_complexity")
    if after.conditional_branches < before.conditional_branches:
        criteria.append("reduced_conditional_branches")
    if after.maximum_nesting_depth < before.maximum_nesting_depth:
        criteria.append("reduced_nesting_depth")
    if after.control_flow_constructs < before.control_flow_constructs:
        criteria.append("reduced_control_flow_constructs")
    return PropertyEvidence(int(bool(criteria)), tuple(criteria))


def label_modularity(before: StructuralMetrics, after: StructuralMetrics) -> PropertyEvidence:
    criteria = []
    if after.function_count > before.function_count:
        criteria.append("increased_function_decomposition")
    if after.helper_function_count > before.helper_function_count:
        criteria.append("introduced_helper_function")
    if after.responsibility_tokens < before.responsibility_tokens:
        criteria.append("reduced_apparent_responsibilities")
    if (
        after.top_level_statement_count < before.top_level_statement_count
        and after.function_count >= before.function_count
    ):
        criteria.append("moved_logic_behind_function_boundary")
    if (
        after.helper_function_count > before.helper_function_count
        and after.cyclomatic_complexity <= before.cyclomatic_complexity
    ):
        criteria.append("helper_extraction_with_nonincreasing_complexity")
    return PropertyEvidence(int(bool(criteria)), tuple(criteria))


def combine_evidence(items: Sequence[PropertyEvidence]) -> PropertyEvidence:
    criteria = sorted({criterion for item in items for criterion in item.criteria_met})
    return PropertyEvidence(int(any(item.label for item in items)), tuple(criteria))


def label_patch(patch: str) -> Tuple[InstanceLabels, int, int]:
    fragments = list(iter_diff_fragments(patch))
    coupling: List[PropertyEvidence] = []
    complexity: List[PropertyEvidence] = []
    modularity: List[PropertyEvidence] = []
    parseable_pairs = 0

    for fragment in fragments:
        before = compute_metrics(fragment.before_code)
        after = compute_metrics(fragment.after_code)
        if not before.parse_succeeded or not after.parse_succeeded:
            continue
        parseable_pairs += 1
        coupling.append(label_coupling(before, after))
        complexity.append(label_complexity(before, after))
        modularity.append(label_modularity(before, after))

    return (
        InstanceLabels(
            coupling=combine_evidence(coupling),
            complexity=combine_evidence(complexity),
            modularity=combine_evidence(modularity),
        ),
        len(fragments),
        parseable_pairs,
    )


def build_prompt(example: Mapping[str, Any]) -> str:
    sections = [
        "You are given a software issue from a Git repository.",
        f"Repository: {safe_text(example.get('repo'))}",
        f"Base commit: {safe_text(example.get('base_commit'))}",
        "",
        "Issue:",
        safe_text(example.get("problem_statement")).strip(),
    ]
    hints = safe_text(example.get("hints_text")).strip()
    if hints:
        sections.extend(["", "Hints:", hints])
    sections.extend(["", "Generate a unified diff patch that resolves the issue."])
    return "\n".join(sections).strip()


def tokenize(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].squeeze(0).long(),
        "attention_mask": encoded["attention_mask"].squeeze(0).long(),
    }


def load_swebench(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    revision: Optional[str],
) -> Dataset:
    kwargs: Dict[str, Any] = {"split": split}
    if revision:
        kwargs["revision"] = revision
    dataset = (
        load_dataset(dataset_name, dataset_config, **kwargs)
        if dataset_config
        else load_dataset(dataset_name, **kwargs)
    )
    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected Dataset, received {type(dataset).__name__}")
    required = {"instance_id", "repo", "base_commit", "problem_statement", "patch"}
    missing = required - set(dataset.column_names)
    if missing:
        raise KeyError(f"Dataset missing required fields: {sorted(missing)}")
    return dataset


def multilabel_signature(record: PreparedRecord) -> str:
    return f"{record.coupling_label}{record.complexity_label}{record.modularity_label}"


def stratified_split_indices(
    records: Sequence[PreparedRecord],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, List[int]]:
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("Split fractions must be nonnegative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be < 1")

    groups: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[multilabel_signature(record)].append(index)

    rng = random.Random(seed)
    result = {"adaptation": [], "validation": [], "test": []}
    for indices in groups.values():
        rng.shuffle(indices)
        n_test = int(round(len(indices) * test_fraction))
        n_val = int(round(len(indices) * validation_fraction))
        while n_test + n_val >= len(indices) and n_test > 0:
            n_test -= 1
        while n_test + n_val >= len(indices) and n_val > 0:
            n_val -= 1
        result["test"].extend(indices[:n_test])
        result["validation"].extend(indices[n_test:n_test + n_val])
        result["adaptation"].extend(indices[n_test + n_val:])

    for values in result.values():
        values.sort()
    return result


def prepare_records(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_input_length: int,
    max_target_length: int,
    limit: Optional[int],
    include_unparseable: bool,
):
    records: List[PreparedRecord] = []
    input_ids: List[torch.Tensor] = []
    attention_masks: List[torch.Tensor] = []
    target_ids: List[torch.Tensor] = []
    target_attention_masks: List[torch.Tensor] = []
    rejected: List[Dict[str, Any]] = []

    total = min(len(dataset), limit) if limit is not None else len(dataset)
    for index in range(total):
        example = dataset[index]
        instance_id = safe_text(example.get("instance_id"))
        try:
            gold_patch = safe_text(example.get("patch"))
            labels, fragment_count, parseable_pairs = label_patch(gold_patch)
            if parseable_pairs == 0 and not include_unparseable:
                rejected.append({
                    "instance_id": instance_id,
                    "reason": "no_parseable_python_fragment_pairs",
                    "python_fragment_count": fragment_count,
                })
                continue

            prompt = build_prompt(example)
            encoded_input = tokenize(tokenizer, prompt, max_input_length)
            encoded_target = tokenize(tokenizer, gold_patch, max_target_length)
            records.append(PreparedRecord(
                instance_id=instance_id,
                repo=safe_text(example.get("repo")),
                base_commit=safe_text(example.get("base_commit")),
                problem_statement=safe_text(example.get("problem_statement")),
                gold_patch=gold_patch,
                test_patch=safe_text(example.get("test_patch")),
                hints_text=safe_text(example.get("hints_text")),
                created_at=safe_text(example.get("created_at")),
                version=safe_text(example.get("version")),
                prompt=prompt,
                coupling_label=labels.coupling.label,
                complexity_label=labels.complexity.label,
                modularity_label=labels.modularity.label,
                coupling_criteria=labels.coupling.criteria_met,
                complexity_criteria=labels.complexity.criteria_met,
                modularity_criteria=labels.modularity.criteria_met,
                fragment_count=fragment_count,
                parseable_fragment_pairs=parseable_pairs,
            ))
            input_ids.append(encoded_input["input_ids"])
            attention_masks.append(encoded_input["attention_mask"])
            target_ids.append(encoded_target["input_ids"])
            target_attention_masks.append(encoded_target["attention_mask"])
        except Exception as exc:
            LOGGER.exception("Failed on %s", instance_id)
            rejected.append({
                "instance_id": instance_id,
                "reason": f"{type(exc).__name__}: {exc}",
            })

        if (index + 1) % 100 == 0 or index + 1 == total:
            LOGGER.info(
                "Processed %d/%d; retained=%d rejected=%d",
                index + 1, total, len(records), len(rejected),
            )

    if not records:
        raise RuntimeError("No records retained; inspect rejected_instances.jsonl")
    return (
        records, input_ids, attention_masks, target_ids,
        target_attention_masks, rejected,
    )


def save_outputs(
    output_dir: Path,
    records: Sequence[PreparedRecord],
    input_ids: Sequence[torch.Tensor],
    attention_masks: Sequence[torch.Tensor],
    target_ids: Sequence[torch.Tensor],
    target_attention_masks: Sequence[torch.Tensor],
    splits: Mapping[str, Sequence[int]],
    rejected: Sequence[Mapping[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    config: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "instances.jsonl", [asdict(r) for r in records])
    write_jsonl(output_dir / "rejected_instances.jsonl", rejected)

    with (output_dir / "property_labels.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "instance_id", "repo", "coupling_label", "complexity_label",
            "modularity_label", "coupling_criteria", "complexity_criteria",
            "modularity_criteria", "fragment_count", "parseable_fragment_pairs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "instance_id": r.instance_id,
                "repo": r.repo,
                "coupling_label": r.coupling_label,
                "complexity_label": r.complexity_label,
                "modularity_label": r.modularity_label,
                "coupling_criteria": "|".join(r.coupling_criteria),
                "complexity_criteria": "|".join(r.complexity_criteria),
                "modularity_criteria": "|".join(r.modularity_criteria),
                "fragment_count": r.fragment_count,
                "parseable_fragment_pairs": r.parseable_fragment_pairs,
            })

    cache = {
        "format_version": 1,
        "instance_ids": [r.instance_id for r in records],
        "repos": [r.repo for r in records],
        "prompts": [r.prompt for r in records],
        "gold_patches": [r.gold_patch for r in records],
        "input_ids": list(input_ids),
        "attention_masks": list(attention_masks),
        "target_ids": list(target_ids),
        "target_attention_masks": list(target_attention_masks),
        "labels": {
            "coupling": torch.tensor([r.coupling_label for r in records], dtype=torch.long),
            "complexity": torch.tensor([r.complexity_label for r in records], dtype=torch.long),
            "modularity": torch.tensor([r.modularity_label for r in records], dtype=torch.long),
        },
        "criteria": {
            "coupling": [r.coupling_criteria for r in records],
            "complexity": [r.complexity_criteria for r in records],
            "modularity": [r.modularity_criteria for r in records],
        },
        "splits": {
            name: torch.tensor(list(indices), dtype=torch.long)
            for name, indices in splits.items()
        },
        "tokenizer": {
            "name_or_path": tokenizer.name_or_path,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id,
        },
        "config": dict(config),
    }
    torch.save(cache, output_dir / "prepared_dataset.pt")

    write_json(output_dir / "split_manifest.json", {
        name: [records[i].instance_id for i in indices]
        for name, indices in splits.items()
    })

    positives = {
        "coupling": sum(r.coupling_label for r in records),
        "complexity": sum(r.complexity_label for r in records),
        "modularity": sum(r.modularity_label for r in records),
    }
    statistics = {
        "retained_instances": len(records),
        "rejected_instances": len(rejected),
        "positive_labels": positives,
        "positive_rates": {k: v / len(records) for k, v in positives.items()},
        "label_signatures": dict(sorted(Counter(multilabel_signature(r) for r in records).items())),
        "splits": {name: len(indices) for name, indices in splits.items()},
        "parseable_fragment_pairs": {
            "total": sum(r.parseable_fragment_pairs for r in records),
            "mean_per_instance": sum(r.parseable_fragment_pairs for r in records) / len(records),
        },
    }
    write_json(output_dir / "labeling_statistics.json", statistics)
    write_json(output_dir / "preprocessing_config.json", dict(config))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="princeton-nlp/SWE-bench")
    parser.add_argument("--dataset-config", default=None)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-name", default="bigcode/starcoderbase-1b")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prepared_swebench"))
    parser.add_argument("--max-input-length", type=int, default=4096)
    parser.add_argument("--max-target-length", type=int, default=2048)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-unparseable", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}; use --overwrite"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_swebench(
        args.dataset_name, args.dataset_config, args.split, args.dataset_revision
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    prepared = prepare_records(
        dataset=dataset,
        tokenizer=tokenizer,
        max_input_length=args.max_input_length,
        max_target_length=args.max_target_length,
        limit=args.limit,
        include_unparseable=args.include_unparseable,
    )
    records, input_ids, masks, targets, target_masks, rejected = prepared
    splits = stratified_split_indices(
        records, args.validation_fraction, args.test_fraction, args.seed
    )

    config = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dataset_revision": args.dataset_revision,
        "source_split": args.split,
        "model_name": args.model_name,
        "max_input_length": args.max_input_length,
        "max_target_length": args.max_target_length,
        "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "include_unparseable": args.include_unparseable,
        "script_sha256": stable_hash(Path(__file__).read_text(encoding="utf-8")),
    }
    save_outputs(
        args.output_dir, records, input_ids, masks, targets, target_masks,
        splits, rejected, tokenizer, config,
    )
    LOGGER.info("Prepared dataset saved to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
