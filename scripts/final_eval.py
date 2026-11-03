"""Evaluate every completed run on the test split, once model selection is done on val.

This is the only code path that loads the test split. Every invocation is appended to
results/final/test_access_log.json before any test data is read.

With --quant-root, the ONNX variants that scripts/quantize.py wrote for each signer-independent
run are evaluated instead of the checkpoints.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

from edgevision.data.io import write_json
from edgevision.evaluation.evaluators import Evaluator, OnnxEvaluator, TorchCheckpointEvaluator
from edgevision.evaluation.metrics import top_k_accuracy
from edgevision.evaluation.tables import (
    PRIMARY_SPLIT_MODE,
    append_access_log,
    per_participant_table,
    score_test_split,
    select_best_model,
    summarize_quant_test,
    summarize_test,
    top_confusions,
)
from edgevision.runs import (
    CHECKPOINT_FILE,
    QUANT_VARIANTS,
    CompletedRun,
    find_completed_runs,
    variant_file,
)
from edgevision.train.data import SplitArrays, load_split_arrays
from edgevision.train.loop import resolve_device

ACCESS_LOG = "test_access_log.json"
QUANT_DIR = "quant"
# Largest allowed gap between the fp32 ONNX and torch test top-1: 0.1 percentage points.
SANITY_TOLERANCE = 0.001


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("results/runs"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--final-dir", type=Path, default=Path("results/final"))
    parser.add_argument(
        "--processed-dir", type=Path, help="override the processed_dir each run recorded"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--quant-root",
        type=Path,
        help="evaluate the ONNX variants under this directory (scripts/quantize.py output)",
    )
    return parser.parse_args(argv)


def load_test_split(
    args: argparse.Namespace, run: CompletedRun, cache: dict[tuple[str, str], SplitArrays]
) -> SplitArrays:
    """The run's test split, loaded once per (processed dir, split file contents)."""
    processed_dir = args.processed_dir or Path(run.summary["processed_dir"])
    splits_path = run.directory / run.summary["splits_file"]
    key = (str(processed_dir), hashlib.sha256(splits_path.read_bytes()).hexdigest())
    if key not in cache:
        cache[key] = load_split_arrays(processed_dir, splits_path, ["test"], allow_test=True)[
            "test"
        ]
    return cache[key]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs = find_completed_runs(args.runs_root)
    if not runs:
        print(f"error: no completed runs under {args.runs_root}", file=sys.stderr)
        return 1
    val_summary_path = args.tables_dir / "val_summary.csv"
    if not val_summary_path.exists():
        print(f"error: {val_summary_path} is missing; run aggregate_val.py first", file=sys.stderr)
        return 1
    best = select_best_model(pd.read_csv(val_summary_path))
    if not any((run.summary["model"], run.summary["split_mode"]) == best for run in runs):
        print(
            f"error: {val_summary_path} picks {best}, which has no completed run", file=sys.stderr
        )
        return 1
    if args.quant_root is not None:
        return evaluate_variants(args, runs)

    append_access_log(args.final_dir / ACCESS_LOG, [run.key for run in runs])
    device = resolve_device(args.device)
    loaded: dict[tuple[str, str], SplitArrays] = {}
    records: list[dict[str, Any]] = []
    best_predictions: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]] = []
    best_label_map: dict[int, str] = {}
    for run in runs:
        summary = run.summary
        test = load_test_split(args, run, loaded)

        evaluator: Evaluator = TorchCheckpointEvaluator(
            run.directory / CHECKPOINT_FILE, device, args.batch_size
        )
        logits = evaluator.predict(test.X)
        record = {
            "model": summary["model"],
            "split_mode": summary["split_mode"],
            "seed": summary["seed"],
            "evaluator": evaluator.name,
            **score_test_split(logits, test.y, test.participant_id),
        }
        target = args.final_dir / run.key / "test_metrics.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        write_json(target, record)
        records.append(record)
        if (summary["model"], summary["split_mode"]) == best:
            best_predictions.append((test.y, logits.argmax(axis=1)))
            best_label_map = evaluator.label_map
        print(
            f"{run.key}: test top-1 {record['test_top1']:.4f} top-5 {record['test_top5']:.4f} "
            f"macro-F1 {record['test_macro_f1']:.4f} (n={record['n_sequences']})",
            flush=True,
        )

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    summary_table = summarize_test(records)
    summary_table.to_csv(args.tables_dir / "test_fp32_summary.csv", index=False)
    per_participant_table(records).to_csv(args.tables_dir / "test_per_participant.csv", index=False)
    confusions = top_confusions(best_predictions, best_label_map)
    confusions.insert(0, "split_mode", best[1])
    confusions.insert(0, "model", best[0])
    confusions.to_csv(args.tables_dir / "top_confusions.csv", index=False)
    print(summary_table.to_string(index=False), flush=True)
    print(f"top confusions from {best[0]} ({best[1]}), chosen by mean val top-1", flush=True)
    return 0


def torch_test_top1(
    args: argparse.Namespace, run: CompletedRun, test: SplitArrays, device: torch.device
) -> tuple[float, str]:
    """The checkpoint's test top-1 from its test_metrics.json if present, else computed."""
    path = args.final_dir / run.key / "test_metrics.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("evaluator") != "torch_fp32" or record.get("n_sequences") != len(test.y):
            raise ValueError(f"{path} does not hold torch_fp32 metrics for this test split")
        return float(record["test_top1"]), str(path)
    evaluator = TorchCheckpointEvaluator(run.directory / CHECKPOINT_FILE, device, args.batch_size)
    return top_k_accuracy(evaluator.predict(test.X), test.y, 1), f"computed from {CHECKPOINT_FILE}"


def evaluate_variants(args: argparse.Namespace, runs: Sequence[CompletedRun]) -> int:
    """Test metrics of each ONNX variant of each signer-independent run.

    Each run's fp32 variant must match the torch checkpoint's test top-1 within
    SANITY_TOLERANCE, or the evaluation stops with an error.
    """
    plan: list[tuple[CompletedRun, dict[str, Path]]] = []
    for run in runs:
        if run.summary["split_mode"] != PRIMARY_SPLIT_MODE:
            continue
        directory = args.quant_root / run.key
        present = {
            variant: directory / variant_file(variant)
            for variant in QUANT_VARIANTS
            if (directory / variant_file(variant)).is_file()
        }
        if "fp32" not in present:
            print(
                f"error: {directory / variant_file('fp32')} is missing; run quantize.py first",
                file=sys.stderr,
            )
            return 1
        for variant in QUANT_VARIANTS:
            if variant not in present:
                print(f"warning: {run.key} has no {variant} variant; skipped", file=sys.stderr)
        plan.append((run, present))
    if not plan:
        print(f"error: no completed {PRIMARY_SPLIT_MODE} runs to evaluate", file=sys.stderr)
        return 1

    evaluated: list[str] = []
    for run, present in plan:
        if not (args.final_dir / run.key / "test_metrics.json").exists():
            evaluated.append(run.key)  # The checkpoint itself, for the fp32 sanity check.
        evaluated += [f"{QUANT_DIR}/{run.key}/{variant}" for variant in present]
    append_access_log(args.final_dir / ACCESS_LOG, evaluated)

    device = resolve_device(args.device)
    loaded: dict[tuple[str, str], SplitArrays] = {}
    records: list[dict[str, Any]] = []
    for run, present in plan:
        summary = run.summary
        test = load_test_split(args, run, loaded)
        evaluators = {variant: OnnxEvaluator(path) for variant, path in present.items()}
        logits = {variant: evaluator.predict(test.X) for variant, evaluator in evaluators.items()}
        try:
            torch_top1, source = torch_test_top1(args, run, test, device)
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 1
        onnx_top1 = top_k_accuracy(logits["fp32"], test.y, 1)
        if abs(onnx_top1 - torch_top1) > SANITY_TOLERANCE:
            print(
                f"error: {run.key}: fp32 ONNX test top-1 {onnx_top1:.4f} differs from the torch "
                f"checkpoint's {torch_top1:.4f} ({source}) by more than 0.1 percentage points",
                file=sys.stderr,
            )
            return 1

        reference = logits["fp32"].argmax(axis=1)
        for variant, path in present.items():
            record: dict[str, Any] = {
                "model": summary["model"],
                "split_mode": summary["split_mode"],
                "seed": summary["seed"],
                "variant": variant,
                "evaluator": evaluators[variant].name,
                "file": str(path),
                "size_bytes": path.stat().st_size,
                **score_test_split(logits[variant], test.y, test.participant_id),
                "agreement_with_fp32": float((logits[variant].argmax(axis=1) == reference).mean()),
            }
            if variant == "fp32":
                record.update(torch_test_top1=torch_top1, torch_reference=source)
            target = (
                args.final_dir
                / QUANT_DIR
                / summary["model"]
                / f"seed_{summary['seed']}"
                / f"{variant}_test_metrics.json"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            write_json(target, record)
            records.append(record)
            print(
                f"{run.key} {variant}: test top-1 {record['test_top1']:.4f} "
                f"top-5 {record['test_top5']:.4f} macro-F1 {record['test_macro_f1']:.4f} "
                f"agreement {record['agreement_with_fp32']:.4f} (n={record['n_sequences']})",
                flush=True,
            )

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    table = summarize_quant_test(records)
    table.to_csv(args.tables_dir / "test_quant_summary.csv", index=False)
    print(table.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
