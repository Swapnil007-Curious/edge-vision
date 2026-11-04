"""Latency of one benchmark configuration, and summaries across rounds.

Every call is timed on its own with time.perf_counter_ns. Inputs are val sequences taken in order
and cycled, and Python's garbage collector is paused while a configuration is timed.
"""

import gc
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

# psutil ships no type stubs, and types-psutil is not one of this project's dev dependencies.
import psutil  # type: ignore[import-untyped]
import torch

from edgevision.models.registry import load_checkpoint
from edgevision.runtime import OrtModel

FloatArray = npt.NDArray[np.float32]
Runner = Callable[[FloatArray], object]

BENCH_VARIANTS = (
    "torch_fp32",
    "onnx_fp32",
    "fp16",
    "int8_dynamic",
    "int8_static",
    "int8_static_full",
)
# The scripts/quantize.py variant each ONNX benchmark variant runs.
ONNX_VARIANT = {
    "onnx_fp32": "fp32",
    "fp16": "fp16",
    "int8_dynamic": "int8_dynamic",
    "int8_static": "int8_static",
    "int8_static_full": "int8_static_full",
}
SHUFFLE_SEED = 0
FRAME_MS = 1000.0 / 30.0  # One camera frame at 30 fps.
RAW_COLUMNS = (
    "round",
    "position",
    "model",
    "seed",
    "variant",
    "threads",
    "batch",
    "session_create_ms",
    "rss_increase_mb_approx",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "min_ms",
    "max_ms",
    "throughput",
    "warmup",
    "iters",
    "size_mb",
    "file",
)
SUMMARY_COLUMNS = (
    "model",
    "variant",
    "threads",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "p50_spread_ms",
    "throughput",
    "size_mb",
    "p95_share_of_33ms_frame",
)


@dataclass(frozen=True)
class BenchConfig:
    model: str
    seed: int
    variant: str
    threads: int
    batch: int
    path: Path  # best.pt for torch_fp32, otherwise the ONNX variant file.


def load_runner(config: BenchConfig) -> Runner:
    """A callable that runs one batch: a fresh onnxruntime session (see OrtModel), or the
    checkpoint in eager torch under inference_mode after torch.set_num_threads(threads)."""
    if config.variant != "torch_fp32":
        return OrtModel(config.path, intra_op_threads=config.threads).run
    torch.set_num_threads(config.threads)
    model = load_checkpoint(config.path).model

    def run_torch(x: FloatArray) -> object:
        with torch.inference_mode():
            return model(torch.from_numpy(x))

    return run_torch


def time_calls(
    runner: Runner, inputs: FloatArray, *, batch: int, warmup: int, iters: int
) -> npt.NDArray[np.int64]:
    """Nanoseconds taken by each of ``iters`` calls that follow ``warmup`` untimed calls.

    Call k gets rows k*batch .. k*batch+batch-1 of ``inputs``, wrapping around at the end; the
    batch is prepared before its timer starts.
    """
    n = len(inputs)

    def batch_at(call: int) -> FloatArray:
        start = (call * batch) % n
        if start + batch <= n:
            return inputs[start : start + batch]
        return np.take(inputs, np.arange(start, start + batch), axis=0, mode="wrap")

    durations = np.empty(iters, dtype=np.int64)
    gc.collect()
    collecting = gc.isenabled()
    gc.disable()
    try:
        for call in range(warmup):
            runner(batch_at(call))
        for index in range(iters):
            x = batch_at(warmup + index)
            started = time.perf_counter_ns()
            runner(x)
            durations[index] = time.perf_counter_ns() - started
    finally:
        if collecting:
            gc.enable()
    return durations


def latency_stats(durations_ns: npt.NDArray[np.int64], batch: int) -> dict[str, float]:
    """Milliseconds per call, and throughput in sequences per second."""
    ms = durations_ns / 1e6
    mean = float(ms.mean())
    return {
        "mean_ms": mean,
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "min_ms": float(ms.min()),
        "max_ms": float(ms.max()),
        "throughput": batch * 1000.0 / mean,
    }


def measure(config: BenchConfig, inputs: FloatArray, *, warmup: int, iters: int) -> dict[str, Any]:
    """One configuration: session creation time, approximate RSS growth, then call latency."""
    process = psutil.Process()
    rss_before = process.memory_info().rss
    started = time.perf_counter_ns()
    runner = load_runner(config)
    created_ns = time.perf_counter_ns() - started
    rss_after = process.memory_info().rss
    durations = time_calls(runner, inputs, batch=config.batch, warmup=warmup, iters=iters)
    return {
        "model": config.model,
        "seed": config.seed,
        "variant": config.variant,
        "threads": config.threads,
        "batch": config.batch,
        "session_create_ms": created_ns / 1e6,
        "rss_increase_mb_approx": (rss_after - rss_before) / 1e6,
        **latency_stats(durations, config.batch),
        "warmup": warmup,
        "iters": iters,
        "size_mb": config.path.stat().st_size / 1e6,
        "file": str(config.path),
    }


def round_orders(n_configs: int, rounds: int, seed: int = SHUFFLE_SEED) -> list[list[int]]:
    """A fresh shuffle of the configuration indices for each round, from one seeded generator."""
    rng = np.random.default_rng(seed)
    return [[int(index) for index in rng.permutation(n_configs)] for _ in range(rounds)]


def summarize_rounds(raw: pd.DataFrame) -> pd.DataFrame:
    """Per (model, variant, threads): medians of p50, p95, p99 and throughput across rounds (and
    seeds); p50_spread_ms is the largest minus the smallest p50."""
    frame = raw.assign(
        variant=pd.Categorical(raw["variant"], categories=BENCH_VARIANTS, ordered=True)
    )
    table = (
        frame.groupby(["model", "variant", "threads"], sort=True, observed=True)
        .agg(
            p50_ms=("p50_ms", "median"),
            p95_ms=("p95_ms", "median"),
            p99_ms=("p99_ms", "median"),
            p50_min=("p50_ms", "min"),
            p50_max=("p50_ms", "max"),
            throughput=("throughput", "median"),
            size_mb=("size_mb", "mean"),
        )
        .reset_index()
    )
    table["p50_spread_ms"] = table["p50_max"] - table["p50_min"]
    table["p95_share_of_33ms_frame"] = table["p95_ms"] / FRAME_MS
    table["variant"] = table["variant"].astype(str)
    return table.loc[:, list(SUMMARY_COLUMNS)]
