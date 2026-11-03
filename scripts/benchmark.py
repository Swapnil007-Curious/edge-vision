"""CPU latency of the trained models and their ONNX variants, one sequence per call by default.

Inputs are val sequences cycled in order; test is never read. Each configuration (model, seed,
variant, intra-op threads) gets a fresh session, warm-up calls, then individually timed calls.
The whole configuration list runs for several rounds, shuffled within each round by a seeded
generator, with a pause between configurations. Writes <out>/bench/bench_raw.csv,
<out>/bench/bench_env.json and <out>/tables/bench_summary.csv (with a _batch<N> suffix when
--batch is above 1).
"""

import argparse
import hashlib
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from edgevision.bench.env import BATTERY_WARNING, collect_bench_env, cpu_frequency, power_source
from edgevision.bench.latency import (
    BENCH_VARIANTS,
    FRAME_MS,
    ONNX_VARIANT,
    RAW_COLUMNS,
    SHUFFLE_SEED,
    BenchConfig,
    measure,
    round_orders,
    summarize_rounds,
)
from edgevision.data.io import write_json
from edgevision.evaluation.tables import PRIMARY_SPLIT_MODE
from edgevision.models.config import MODEL_NAMES
from edgevision.runs import CHECKPOINT_FILE, CompletedRun, find_completed_runs, variant_file
from edgevision.runtime import session_settings
from edgevision.train.data import load_split_arrays


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quant-root", type=Path, default=Path("results/quant"))
    parser.add_argument("--runs-root", type=Path, default=Path("results/runs"))
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="preprocessing output to read val inputs from",
    )
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument(
        "--variants", nargs="+", choices=BENCH_VARIANTS, default=list(BENCH_VARIANTS)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--threads", nargs="+", type=positive, default=[1, 2, 4])
    parser.add_argument("--rounds", type=positive, default=3)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=positive, default=2000)
    parser.add_argument(
        "--batch", type=positive, default=1, help="sequences per call; above 1 is throughput mode"
    )
    parser.add_argument(
        "--sleep", type=float, default=3.0, help="seconds of pause between configurations"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results"), help="results root for bench/ and tables/"
    )
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.sleep < 0:
        parser.error("--warmup and --sleep must not be negative")
    return args


def val_inputs(
    runs: dict[tuple[str, int], CompletedRun], processed_dir: Path
) -> dict[tuple[str, int], npt.NDArray[np.float32]]:
    """Val inputs per (model, seed) from each run's split file, loaded once per file contents."""
    loaded: dict[str, npt.NDArray[np.float32]] = {}
    inputs = {}
    for key, run in runs.items():
        splits_path = run.directory / run.summary["splits_file"]
        digest = hashlib.sha256(splits_path.read_bytes()).hexdigest()
        if digest not in loaded:
            loaded[digest] = load_split_arrays(processed_dir, splits_path, ["val"])["val"].X
        inputs[key] = loaded[digest]
    return inputs


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runs = {
        (run.summary["model"], run.summary["seed"]): run
        for run in find_completed_runs(args.runs_root)
        if run.summary["split_mode"] == PRIMARY_SPLIT_MODE
    }
    wanted = [(model, seed) for model in args.models for seed in args.seeds]
    absent = [f"{model} seed {seed}" for model, seed in wanted if (model, seed) not in runs]
    if absent:
        print(f"error: no completed {PRIMARY_SPLIT_MODE} run for {absent}", file=sys.stderr)
        return 1

    configs = []
    for model, seed in wanted:
        run = runs[(model, seed)]
        for variant in args.variants:
            path = (
                run.directory / CHECKPOINT_FILE
                if variant == "torch_fp32"
                else args.quant_root / run.key / variant_file(ONNX_VARIANT[variant])
            )
            configs += [
                BenchConfig(model, seed, variant, threads, args.batch, path)
                for threads in args.threads
            ]
    missing = sorted({str(config.path) for config in configs if not config.path.is_file()})
    if missing:
        print(f"error: missing model files (run quantize.py first): {missing}", file=sys.stderr)
        return 1
    inputs = val_inputs({key: runs[key] for key in wanted}, args.processed_dir)

    suffix = "" if args.batch == 1 else f"_batch{args.batch}"
    raw_path = args.out / "bench" / f"bench_raw{suffix}.csv"
    env_path = args.out / "bench" / f"bench_env{suffix}.json"
    summary_path = args.out / "tables" / f"bench_summary{suffix}.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    protocol: dict[str, Any] = {
        "inputs": "val sequences cycled in order",
        "timer": "time.perf_counter_ns around each call; garbage collector paused while timing",
        "models": args.models,
        "variants": args.variants,
        "seeds": args.seeds,
        "threads": args.threads,
        "batch": args.batch,
        "rounds": args.rounds,
        "warmup": args.warmup,
        "iters": args.iters,
        "sleep_s": args.sleep,
        "shuffle_seed": SHUFFLE_SEED,
        "onnxruntime_session": session_settings(intra_op_threads=None),
        "torch": "eager, torch.inference_mode, torch.set_num_threads(threads)",
        "rss_increase": "approximate: process RSS after minus before session creation",
        "frame_ms": FRAME_MS,
    }
    env = collect_bench_env(protocol)
    write_json(env_path, env)
    for warning in env["warnings"]:
        print(f"warning: {warning}", file=sys.stderr)
    print(
        f"{len(configs)} configurations x {args.rounds} rounds, {args.iters} timed calls each; "
        f"pauses alone take {len(configs) * args.rounds * args.sleep / 60:.1f} min",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    total = len(configs) * args.rounds
    for round_number, order in enumerate(round_orders(len(configs), args.rounds), start=1):
        for position, index in enumerate(order):
            if rows and args.sleep > 0:
                time.sleep(args.sleep)
            config = configs[index]
            row = {
                "round": round_number,
                "position": position,
                **measure(
                    config,
                    inputs[(config.model, config.seed)],
                    warmup=args.warmup,
                    iters=args.iters,
                ),
            }
            rows.append(row)
            pd.DataFrame(rows, columns=list(RAW_COLUMNS)).to_csv(raw_path, index=False)
            print(
                f"[{len(rows)}/{total}] round {round_number} {config.model} seed {config.seed} "
                f"{config.variant} threads {config.threads}: p50 {row['p50_ms']:.3f} ms "
                f"p95 {row['p95_ms']:.3f} ms",
                flush=True,
            )

    env["cpu_freq_end"] = cpu_frequency()
    env["power_end"] = power_source()
    if env["power_end"]["on_ac_power"] is False and BATTERY_WARNING not in env["warnings"]:
        env["warnings"].append(BATTERY_WARNING)
        print(f"warning: {BATTERY_WARNING}", file=sys.stderr)
    write_json(env_path, env)

    summary = summarize_rounds(pd.DataFrame(rows))
    summary.to_csv(summary_path, index=False)
    compact = summary.rename(
        columns={
            "p50_spread_ms": "spread_ms",
            "throughput": "seq_per_s",
            "p95_share_of_33ms_frame": "p95_frame",
        }
    )
    print(compact.to_string(index=False, float_format=lambda value: f"{value:.3f}"), flush=True)
    print(f"wrote {raw_path}, {env_path} and {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
