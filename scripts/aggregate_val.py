"""Aggregate validation metrics over seeds from every completed run's summary.json, and score
the seed ensemble of every experiment with at least two completed seeds on val.

val_summary.csv comes from summary files only. The ensemble (mean of the members' logits) needs
the checkpoints and the val split; it is skipped with a warning when a checkpoint or the
processed dir is not available. --processed-dir takes one or more directories; each experiment
reads the first whose landmark layout matches its checkpoints. The test split is never touched.
"""

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from edgevision.evaluation.evaluators import TorchCheckpointEvaluator
from edgevision.evaluation.tables import (
    aggregate_val,
    ensemble_row,
    ensemble_scores,
    ensemble_table,
)
from edgevision.models.registry import checkpoint_layout
from edgevision.runs import CHECKPOINT_FILE, CompletedRun, find_completed_runs, runs_by_experiment
from edgevision.train.data import load_split_arrays, matching_processed_dir
from edgevision.train.loop import resolve_device

MIN_ENSEMBLE_SEEDS = 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("results/runs"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/val_summary.csv"))
    parser.add_argument(
        "--ensemble-output", type=Path, help="default: val_ensemble.csv beside --output"
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        nargs="+",
        help="override the processed_dir each run recorded; with several, each experiment uses "
        "the first whose landmark layout matches its checkpoints",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args(argv)


def member_val_logits(
    args: argparse.Namespace, members: Sequence[CompletedRun]
) -> tuple[list[npt.NDArray[np.float32]], npt.NDArray[np.int64]] | None:
    """Each member's val logits and the labels, or None (with a warning) when the members'
    split files differ, a checkpoint is missing or no processed dir fits."""
    name = members[0].experiment
    missing = [run.key for run in members if not (run.directory / CHECKPOINT_FILE).is_file()]
    if missing:
        print(
            f"warning: {name}: no {CHECKPOINT_FILE} in {missing}; no val ensemble", file=sys.stderr
        )
        return None
    digests = {
        hashlib.sha256((run.directory / run.summary["splits_file"]).read_bytes()).hexdigest()
        for run in members
    }
    if len(digests) != 1:
        print(f"warning: {name}: seeds used different split files; no ensemble", file=sys.stderr)
        return None
    first = members[0]
    candidates = args.processed_dir or [Path(first.summary["processed_dir"])]
    processed_dir = matching_processed_dir(
        candidates, checkpoint_layout(first.directory / CHECKPOINT_FILE)
    )
    if processed_dir is None:
        shown = ", ".join(str(path) for path in candidates)
        print(
            f"warning: {name}: no features with its landmark layout in {shown}; no val ensemble "
            "(pass --processed-dir)",
            file=sys.stderr,
        )
        return None
    splits_path = first.directory / first.summary["splits_file"]
    val = load_split_arrays(processed_dir, splits_path, ["val"])["val"]
    device = resolve_device(args.device)
    logits = [
        TorchCheckpointEvaluator(run.directory / CHECKPOINT_FILE, device, args.batch_size).predict(
            val.X
        )
        for run in members
    ]
    return logits, val.y


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs = find_completed_runs(args.runs_root)
    if not runs:
        print(f"error: no completed runs under {args.runs_root}", file=sys.stderr)
        return 1
    table = aggregate_val(runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(table.to_string(index=False), flush=True)
    print(f"{len(runs)} runs -> {args.output}", flush=True)

    groups = {
        name: members
        for name, members in runs_by_experiment(runs).items()
        if len(members) >= MIN_ENSEMBLE_SEEDS
    }
    rows = []
    for members in groups.values():
        scored = member_val_logits(args, members)
        if scored is not None:
            rows.append(ensemble_row(members, ensemble_scores(*scored), "val"))
    if rows:
        ensemble_path = args.ensemble_output or args.output.with_name("val_ensemble.csv")
        ensemble = ensemble_table(rows, "val")
        ensemble.to_csv(ensemble_path, index=False)
        print(ensemble.to_string(index=False), flush=True)
        print(f"{len(rows)} seed ensembles -> {ensemble_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
