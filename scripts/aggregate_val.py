"""Aggregate validation metrics over seeds from every completed run's summary.json.

Reads summary files only; the test split is never touched.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from edgevision.evaluation.tables import aggregate_val
from edgevision.runs import find_completed_runs


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("results/runs"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/val_summary.csv"))
    return parser.parse_args(argv)


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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
