"""Run the training grid in sequence; completed runs are skipped, so a restarted session resumes.

--grid v1 (default): signer-independent gru, cnn, transformer at every seed, then each model
signer-dependent at the first seed. --grid v2: the ablation experiments b, c and d (see
edgevision.experiments) at every seed, seed-major so all three finish seed 0 first.

--max-hours H: before starting each run, if more than H hours have passed since the grid
started, that run and every later incomplete one are skipped and listed.
"""

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from edgevision.experiments import GRIDS, v2_grid_plan
from edgevision.runs import SUMMARY_FILE, experiment_run_dir, is_complete
from edgevision.train.config import load_train_config
from edgevision.train.loop import SplitMode, grid_plan, run_training

CONFIG_DIR = Path(__file__).resolve().parents[1] / "experiments"
SECONDS_PER_HOUR = 3600.0

# Monotonic wall clock in seconds; a module attribute so tests can replace it.
clock = time.monotonic


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--grid", choices=GRIDS, default="v1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-root", type=Path, help="override output_root")
    parser.add_argument("--epochs", type=int, help="override epochs (for smoke runs)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--max-hours",
        type=float,
        help="skip runs not yet started once this much wall time has passed",
    )
    args = parser.parse_args(argv)
    if args.max_hours is not None and args.max_hours < 0:
        parser.error("--max-hours must not be negative")
    return args


def plan_runs(grid: str, seeds: Sequence[int]) -> list[tuple[Path, SplitMode | None, int]]:
    """(config file, split mode override, seed) for each run of the grid, in order."""
    if grid == "v2":
        return [
            (CONFIG_DIR / experiment.config, None, seed) for experiment, seed in v2_grid_plan(seeds)
        ]
    return [
        (CONFIG_DIR / f"train_{model}.yaml", split_mode, seed)
        for model, split_mode, seed in grid_plan(seeds)
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = plan_runs(args.grid, args.seeds)
    started = clock()
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for position, (config_path, split_mode, seed) in enumerate(plan, start=1):
        cfg = load_train_config(
            config_path,
            processed_dir=args.processed_dir,
            output_root=args.output_root,
            epochs=args.epochs,
            split_mode=split_mode,
        )
        name = cfg.experiment_name
        directory = experiment_run_dir(cfg.output_root, name, seed)
        label = f"[{position}/{len(plan)}] {name} seed {seed}"
        if is_complete(directory):
            print(f"{label}: complete, skipped", flush=True)
        else:
            elapsed_hours = (clock() - started) / SECONDS_PER_HOUR
            if skipped or (args.max_hours is not None and elapsed_hours > args.max_hours):
                skipped.append(f"{name} seed {seed}")
                continue
            print(f"{label}: training", flush=True)
            run_training(cfg, seed=seed, device=args.device)
        with (directory / SUMMARY_FILE).open(encoding="utf-8") as handle:
            summary = json.load(handle)
        rows.append(
            {
                "experiment": name,
                "model": cfg.model.name,
                "split_mode": cfg.split.mode,
                "seed": seed,
                "best_val_top1": summary["best_val_top1"],
                "best_epoch": summary["best_epoch"],
            }
        )
    if rows:
        print(pd.DataFrame(rows).to_string(index=False), flush=True)
    if skipped:
        hours = (clock() - started) / SECONDS_PER_HOUR
        print(
            f"time budget: {hours:.2f} h elapsed, over --max-hours {args.max_hours}; "
            f"skipped {len(skipped)} run(s) not yet started: {', '.join(skipped)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
