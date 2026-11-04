"""Unit tests for graph statistics, latency statistics, summaries and environment probes."""

import gc
import sys
from collections import namedtuple
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from onnx import helper

from edgevision.bench import env as bench_env
from edgevision.bench.latency import (
    BENCH_VARIANTS,
    FRAME_MS,
    SUMMARY_COLUMNS,
    latency_stats,
    round_orders,
    summarize_rounds,
    time_calls,
)
from edgevision.evaluation.tables import QUANT_TEST_SUMMARY_COLUMNS, summarize_quant_test
from edgevision.quant.graph_stats import graph_stats
from edgevision.quant.pipeline import QUANT_OPS_COLUMNS, quant_ops_summary
from edgevision.quant.variants import drop_identical_nodes


def model_of(*nodes):
    graph = helper.make_graph(list(nodes), "g", inputs=[], outputs=[])
    return helper.make_model(graph)


def test_graph_stats_counts_float_and_quantised_compute_nodes():
    stats = graph_stats(
        model_of(
            helper.make_node("MatMul", ["x", "w"], ["a"], name="float_matmul"),
            helper.make_node("Gemm", ["a", "w2", "b"], ["c"], name="float_gemm"),
            helper.make_node("GRU", ["c", "W", "R"], ["h"], name="gru"),
        )
    )
    assert stats["float_compute_nodes"] == {"MatMul": 1, "Gemm": 1, "Conv": 0, "GRU": 1}
    assert stats["fraction_compute_ops_quantised"] == 0.0
    assert (stats["gru_float_remaining"], stats["any_gru_float"]) == (1, True)
    assert stats["node_count"] == 3
    assert stats["op_counts"] == {"GRU": 1, "Gemm": 1, "MatMul": 1}


def test_graph_stats_counts_qdq_wrapped_and_integer_nodes_as_quantised():
    stats = graph_stats(
        model_of(
            helper.make_node("DequantizeLinear", ["xq", "s", "z"], ["x"]),
            helper.make_node("DequantizeLinear", ["wq", "s", "z"], ["w"]),
            helper.make_node("MatMul", ["x", "w"], ["y"], name="qdq_matmul"),
            helper.make_node("MatMul", ["y", "w_float"], ["y2"], name="half_qdq_matmul"),
            helper.make_node("MatMulInteger", ["aq", "bq"], ["i"]),
            helper.make_node("ConvInteger", ["aq", "cq"], ["j"]),
            helper.make_node("QuantizeLinear", ["y2", "s", "z"], ["out"]),
        )
    )
    assert stats["qdq_compute_nodes"]["MatMul"] == 1
    assert stats["float_compute_nodes"]["MatMul"] == 1  # Only one operand was dequantized.
    assert stats["integer_and_qdq_nodes"]["MatMulInteger"] == 1
    assert stats["integer_and_qdq_nodes"]["ConvInteger"] == 1
    assert stats["integer_and_qdq_nodes"]["DequantizeLinear"] == 2
    assert stats["integer_and_qdq_nodes"]["QuantizeLinear"] == 1
    assert stats["fraction_compute_ops_quantised"] == pytest.approx(3 / 4)
    assert not stats["any_gru_float"]


def test_drop_identical_nodes_keeps_one_copy_and_only_drops_exact_copies():
    cast = helper.make_node("Cast", ["x"], ["x32"], name="cast", to=1)
    renamed_twin = helper.make_node("Cast", ["x"], ["x32"], name="cast_b", to=1)
    graph = model_of(
        cast, helper.make_node("Cast", ["x"], ["x32"], name="cast", to=1), renamed_twin
    ).graph
    drop_identical_nodes(graph)
    assert [node.name for node in graph.node] == ["cast", "cast_b"]


def test_latency_stats_in_milliseconds_and_sequences_per_second():
    stats = latency_stats(np.array([1, 2, 3, 4], dtype=np.int64) * 1_000_000, batch=2)
    assert stats["mean_ms"] == pytest.approx(2.5)
    assert stats["p50_ms"] == pytest.approx(2.5)
    assert (stats["min_ms"], stats["max_ms"]) == (1.0, 4.0)
    assert stats["p95_ms"] == pytest.approx(np.percentile([1, 2, 3, 4], 95))
    assert stats["throughput"] == pytest.approx(2 * 1000 / 2.5)


def test_time_calls_cycles_inputs_in_order_and_restores_the_collector():
    inputs = np.arange(5, dtype=np.float32)[:, None]
    seen = []
    durations = time_calls(
        lambda x: seen.append(x[:, 0].tolist()), inputs, batch=3, warmup=1, iters=2
    )
    assert seen == [[0, 1, 2], [3, 4, 0], [1, 2, 3]]
    assert durations.shape == (2,)
    assert (durations >= 0).all()
    assert gc.isenabled()


def test_round_orders_are_seeded_permutations_that_differ_between_rounds():
    orders = round_orders(10, 3)
    assert orders == round_orders(10, 3)
    assert all(sorted(order) == list(range(10)) for order in orders)
    assert len({tuple(order) for order in orders}) == 3


def test_summarize_rounds_takes_medians_and_the_p50_spread():
    rows = []
    for round_number, p50 in ((1, 1.0), (2, 3.0), (3, 2.5)):
        for variant in ("int8_static", "torch_fp32"):
            rows.append(
                {
                    "round": round_number,
                    "model": "cnn",
                    "variant": variant,
                    "threads": 1,
                    "p50_ms": p50,
                    "p95_ms": 2 * p50,
                    "p99_ms": 3 * p50,
                    "throughput": 1000 / p50,
                    "size_mb": 1.5,
                }
            )
    table = summarize_rounds(pd.DataFrame(rows))
    assert table.columns.tolist() == list(SUMMARY_COLUMNS)
    assert table["variant"].tolist() == ["torch_fp32", "int8_static"]  # BENCH_VARIANTS order.
    row = table.iloc[0]
    assert (row["p50_ms"], row["p95_ms"], row["p99_ms"]) == (2.5, 5.0, 7.5)
    assert row["p50_spread_ms"] == 2.0
    assert row["throughput"] == pytest.approx(400.0)
    assert row["p95_share_of_33ms_frame"] == pytest.approx(5.0 / FRAME_MS)
    assert set(BENCH_VARIANTS) >= set(table["variant"])


def test_cpu_flags_come_from_py_cpuinfo_and_are_none_without_it(monkeypatch):
    fake = SimpleNamespace(get_cpu_info=lambda: {"flags": ["sse4_2", "avx2", "avx512vnni"]})
    monkeypatch.setitem(sys.modules, "cpuinfo", fake)
    assert bench_env.cpu_flags() == {"avx2": True, "avx512f": False, "avx512_vnni": True}

    def broken():
        raise RuntimeError("no CPUID")

    monkeypatch.setitem(sys.modules, "cpuinfo", SimpleNamespace(get_cpu_info=broken))
    assert bench_env.cpu_flags() == {"avx2": None, "avx512f": None, "avx512_vnni": None}


def test_battery_power_is_recorded_with_a_warning(monkeypatch):
    battery = namedtuple("battery", "percent secsleft power_plugged")
    monkeypatch.setattr(bench_env.psutil, "sensors_battery", lambda: battery(55.0, 3600, False))
    monkeypatch.setattr(bench_env, "cpu_flags", lambda: dict.fromkeys(bench_env.CPU_FLAGS))
    env = bench_env.collect_bench_env({"rounds": 1})
    assert env["power"] == {"on_ac_power": False, "battery_percent": 55.0}
    assert env["warnings"] == [bench_env.BATTERY_WARNING]
    assert env["protocol"] == {"rounds": 1}

    monkeypatch.setattr(bench_env.psutil, "sensors_battery", lambda: None)
    assert bench_env.power_source() == {"on_ac_power": None, "battery_percent": None}
    assert bench_env.collect_bench_env({})["warnings"] == []


def test_summarize_quant_test_pairs_deltas_by_seed():
    records = []
    for seed, (fp32, int8) in enumerate(((0.80, 0.78), (0.70, 0.69))):
        for variant, top1 in (("int8_static", int8), ("fp32", fp32)):
            records.append(
                {
                    "model": "gru",
                    "seed": seed,
                    "variant": variant,
                    "test_top1": top1,
                    "test_top5": 0.9,
                    "test_macro_f1": 0.5,
                    "agreement_with_fp32": 1.0 if variant == "fp32" else 0.95,
                    "size_bytes": 2_000_000 if variant == "fp32" else 1_000_000,
                }
            )
    table = summarize_quant_test(records)
    assert table.columns.tolist() == list(QUANT_TEST_SUMMARY_COLUMNS)
    assert table["variant"].tolist() == ["fp32", "int8_static"]
    int8 = table.iloc[1]
    assert int8["n_seeds"] == 2
    assert int8["top1_mean"] == pytest.approx(0.735)
    assert int8["delta_vs_fp32_mean"] == pytest.approx(-0.015)
    assert int8["delta_vs_fp32_std"] == pytest.approx(np.std([-0.02, -0.01], ddof=1))
    assert int8["agreement_mean"] == pytest.approx(0.95)
    assert int8["size_mb"] == pytest.approx(1.0)
    assert table.iloc[0]["delta_vs_fp32_mean"] == 0.0


def test_quant_ops_summary_counts_failures_and_leaves_their_sizes_empty():
    def entry(size, fraction=0.0, gru=0):
        graph = {"fraction_compute_ops_quantised": fraction, "gru_float_remaining": gru}
        return {"error": None, "size_bytes": size, "graph": graph}

    reports = [
        {
            "model": "cnn",
            "variants": {
                "fp32": entry(4_000_000),
                "fp16": entry(2_000_000),
                "int8_dynamic": {"error": "RuntimeError: unsupported", "status": "failed"},
                "int8_static": entry(1_000_000, 1.0),
            },
        },
        {"model": "cnn", "error": "FileNotFoundError: best.pt", "variants": {}},
    ]
    table = quant_ops_summary(reports).set_index("variant")
    assert table.columns.tolist() == list(QUANT_OPS_COLUMNS[0:2] + QUANT_OPS_COLUMNS[3:])
    assert (table["experiment"] == "cnn_signer_independent").all()
    assert table.loc["fp32", "failures"] == 1  # The second run failed as a whole.
    assert table.loc["int8_dynamic", "failures"] == 2
    assert np.isnan(table.loc["int8_dynamic", "size_mb"])
    assert table.loc["int8_static", "size_ratio_vs_fp32"] == pytest.approx(0.25)
    assert table.loc["fp16", "size_mb"] == pytest.approx(2.0)
    assert table.loc["int8_static", "fraction_compute_ops_quantised"] == 1.0
