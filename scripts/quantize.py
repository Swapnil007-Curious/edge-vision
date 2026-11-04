"""Build fp32, fp16, int8 dynamic and int8 static ONNX variants of every completed
signer-independent run, plus int8_static_full (int8 static with nothing excluded, an ablation),
check them on val, and write results/tables/quant_ops_summary.csv.

A variant that fails is recorded in its run's quant_report.json and the batch continues; the exit
code is 1 if anything failed. A file built with an older schema version of its variant is
rebuilt even without --overwrite.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from edgevision.evaluation.tables import PRIMARY_SPLIT_MODE
from edgevision.models.config import MODEL_NAMES
from edgevision.quant.pipeline import find_quant_reports, quant_ops_summary, quantize_run
from edgevision.runs import find_completed_runs, run_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("results/runs"))
    parser.add_argument(
        "--processed-dir", type=Path, required=True, help="preprocessing output (val, calibration)"
    )
    parser.add_argument("--out-root", type=Path, default=Path("results/quant"))
    parser.add_argument("--tables-dir", type=Path, default=Path("results/tables"))
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument("--seeds", nargs="+", type=int, help="default: every seed found")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rebuild variant files that already exist (older-schema files are always rebuilt)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs = [
        run
        for run in find_completed_runs(args.runs_root)
        if run.summary["split_mode"] == PRIMARY_SPLIT_MODE
        and run.summary["model"] in args.models
        and (args.seeds is None or run.summary["seed"] in args.seeds)
    ]
    if not runs:
        print(
            f"error: no completed {PRIMARY_SPLIT_MODE} runs under {args.runs_root} match "
            f"models {args.models} and seeds {args.seeds or 'any'}",
            file=sys.stderr,
        )
        return 1

    failed = []
    for index, run in enumerate(runs, start=1):
        print(f"[{index}/{len(runs)}] {run.key}", flush=True)
        summary = run.summary
        out_dir = run_dir(args.out_root, summary["model"], summary["split_mode"], summary["seed"])
        report = quantize_run(
            run,
            args.processed_dir,
            out_dir,
            overwrite=args.overwrite,
            log=lambda line: print(f"  {line}", flush=True),
        )
        failed += [f"{run.key} {variant}" for variant in report["failures"]]

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    table = quant_ops_summary(find_quant_reports(args.out_root))
    table.to_csv(args.tables_dir / "quant_ops_summary.csv", index=False)
    print(table.to_string(index=False), flush=True)
    if failed:
        print(f"error: {len(failed)} variant(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
