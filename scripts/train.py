"""Train one model on preprocessed asl-signs features and write its run directory."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from edgevision.runs import run_dir
from edgevision.train.config import load_train_config
from edgevision.train.loop import RunExistsError, run_training


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiments/train_<model>.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--processed-dir", type=Path, help="override processed_dir")
    parser.add_argument("--output-root", type=Path, help="override output_root")
    parser.add_argument("--epochs", type=int, help="override epochs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing completed run"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_train_config(
        args.config,
        processed_dir=args.processed_dir,
        output_root=args.output_root,
        epochs=args.epochs,
    )
    try:
        summary = run_training(cfg, seed=args.seed, device=args.device, overwrite=args.overwrite)
    except RunExistsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    directory = run_dir(cfg.output_root, cfg.model.name, cfg.split.mode, args.seed)
    print(
        f"best val top-1 {summary['best_val_top1']:.4f} (top-5 {summary['best_val_top5']:.4f}, "
        f"macro-F1 {summary['best_val_macro_f1']:.4f}) at epoch {summary['best_epoch']}; "
        f"{summary['total_train_seconds']:.1f} s; wrote {directory}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
