# Verbatim copy of scripts/final_eval.py at commit 2365285, before --quant-root existed. The test
# suite runs it beside the current script to show that runs without the flag are unchanged.
"""Evaluate every completed run on the test split, once model selection is done on val.

This is the only code path that loads the test split. Every invocation is appended to
results/final/test_access_log.json before any test data is read.
"""

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from edgevision.data.io import write_json
from edgevision.evaluation.evaluators import Evaluator, TorchCheckpointEvaluator
from edgevision.evaluation.tables import (
    append_access_log,
    per_participant_table,
    score_test_split,
    select_best_model,
    summarize_test,
    top_confusions,
)
from edgevision.runs import CHECKPOINT_FILE, find_completed_runs
from edgevision.train.data import SplitArrays, load_split_arrays
from edgevision.train.loop import resolve_device

ACCESS_LOG = "test_access_log.json"


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
    return parser.parse_args(argv)


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

    append_access_log(args.final_dir / ACCESS_LOG, [run.key for run in runs])
    device = resolve_device(args.device)
    loaded: dict[tuple[str, str], SplitArrays] = {}
    records: list[dict[str, Any]] = []
    best_predictions: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]] = []
    best_label_map: dict[int, str] = {}
    for run in runs:
        summary = run.summary
        processed_dir = args.processed_dir or Path(summary["processed_dir"])
        splits_path = run.directory / summary["splits_file"]
        key = (str(processed_dir), hashlib.sha256(splits_path.read_bytes()).hexdigest())
        if key not in loaded:
            loaded[key] = load_split_arrays(processed_dir, splits_path, ["test"], allow_test=True)[
                "test"
            ]
        test = loaded[key]

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


if __name__ == "__main__":
    raise SystemExit(main())
