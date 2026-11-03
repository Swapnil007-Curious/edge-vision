"""Preprocess the asl-signs dataset into features, a manifest, a report and splits."""

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from edgevision.config import ExperimentConfig, load_config
from edgevision.data.io import (
    build_report,
    format_summary,
    process_dataset,
    write_features,
    write_json,
)
from edgevision.data.splits import (
    assert_signer_disjoint,
    make_splits,
    splits_to_json,
    summarize_splits,
)
from edgevision.utils.seed import set_seed

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "experiments" / "preprocess_default.yaml"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="YAML config (default: %(default)s)"
    )
    parser.add_argument("--data-root", type=Path, help="override preprocess.data_root")
    parser.add_argument("--output-dir", type=Path, help="override preprocess.output_dir")
    parser.add_argument("--limit", type=int, help="override preprocess.limit (first N rows)")
    parser.add_argument(
        "--no-cache", action="store_true", help="neither read nor write the per-sequence cache"
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    return load_config(
        args.config, data_root=args.data_root, output_dir=args.output_dir, limit=args.limit
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_config(args)
    cfg = config.preprocess
    set_seed(cfg.seed)
    use_cache = not args.no_cache

    started = time.perf_counter()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    batch = process_dataset(cfg, use_cache=use_cache)
    write_features(cfg.output_dir / "features.npz", batch)
    batch.manifest.to_csv(cfg.output_dir / "manifest.csv", index=False)
    report = build_report(
        batch, config, wall_time_s=time.perf_counter() - started, use_cache=use_cache
    )
    write_json(cfg.output_dir / "preprocess_report.json", report)
    print(format_summary(report), flush=True)

    # Splits come last: a limited smoke run with too few signers still leaves its outputs.
    splits = make_splits(batch.manifest, config.split)
    if config.split.mode == "signer_independent":
        assert_signer_disjoint(splits, batch.manifest)
    summary = summarize_splits(splits, batch.manifest)
    payload = splits_to_json(splits, summary, config.split)
    write_json(cfg.output_dir / f"splits_{config.split.mode}.json", payload)
    print(f"splits ({config.split.mode}): {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
