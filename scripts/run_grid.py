"""Run the training grid in sequence; completed runs are skipped, so a restarted session resumes.

Signer-independent: gru, cnn, transformer at every seed. Signer-dependent: each model at the
first seed.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from edgevision.runs import SUMMARY_FILE, is_complete, run_dir, run_name
from edgevision.train.config import load_train_config
from edgevision.train.loop import grid_plan, run_training

CONFIG_DIR = Path(__file__).resolve().parents[1] / "experiments"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--output-root", type=Path, help="override output_root")
    parser.add_argument("--epochs", type=int, help="override epochs (for smoke runs)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    plan = grid_plan(args.seeds)
    rows = []
    for position, (model, split_mode, seed) in enumerate(plan, start=1):
        cfg = load_train_config(
            CONFIG_DIR / f"train_{model}.yaml",
            processed_dir=args.processed_dir,
            output_root=args.output_root,
            epochs=args.epochs,
            split_mode=split_mode,
        )
        directory = run_dir(cfg.output_root, model, split_mode, seed)
        label = f"[{position}/{len(plan)}] {run_name(model, split_mode)} seed {seed}"
        if is_complete(directory):
            print(f"{label}: complete, skipped", flush=True)
        else:
            print(f"{label}: training", flush=True)
            run_training(cfg, seed=seed, device=args.device)
        with (directory / SUMMARY_FILE).open(encoding="utf-8") as handle:
            summary = json.load(handle)
        rows.append(
            {
                "model": model,
                "split_mode": split_mode,
                "seed": seed,
                "best_val_top1": summary["best_val_top1"],
                "best_epoch": summary["best_epoch"],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
