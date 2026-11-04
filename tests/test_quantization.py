"""Quantisation, test evaluation of the variants, and the benchmark smoke run, on one set of
briefly trained fixture models (see quant_support.py)."""

import ast
import importlib.util
import inspect
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
import pytest
import torch

from conftest import SIGNS
from edgevision.bench import env as bench_env
from edgevision.bench.latency import BENCH_VARIANTS, FRAME_MS, RAW_COLUMNS, SUMMARY_COLUMNS
from edgevision.data.landmarks import RIGHT_HAND
from edgevision.evaluation.tables import QUANT_TEST_SUMMARY_COLUMNS
from edgevision.layout import FRAME_DIM, INPUT_NAME, NUM_FRAMES
from edgevision.models.config import MODEL_NAMES, default_model_config
from edgevision.models.features import STD_FLOOR
from edgevision.models.registry import load_checkpoint
from edgevision.quant import calibration, pipeline, variants
from edgevision.quant.calibration import TrainCalibrationReader
from edgevision.quant.pipeline import QUANT_OPS_COLUMNS, quantize_run, sha256_file
from edgevision.quant.variants import (
    FEATURE_SCOPE,
    feature_block_tensors,
    read_metadata,
    save_with_metadata,
)
from edgevision.runs import QUANT_VARIANTS, variant_file
from edgevision.runtime import (
    META_LABEL_MAP,
    META_SCHEMA_VERSION,
    META_SOURCE_SHA256,
    META_VARIANT,
    OrtModel,
)
from edgevision.train.data import load_split_arrays, read_splits
from quant_support import (
    TEST_SEQUENCES,
    TRAIN_SEQUENCES,
    VAL_SEQUENCES,
    FixtureRuns,
    train_fixture_runs,
)
from training_support import REPO_ROOT, load_script, snapshot

INDEPENDENT = "signer_independent"
RUN_KEYS = [f"{model}_{INDEPENDENT}/seed_0" for model in sorted(MODEL_NAMES)]
BASELINE_FINAL_EVAL = REPO_ROOT / "tests" / "baselines" / "final_eval_before_quant.py"
ACCESS_LOG = Path("final") / "test_access_log.json"


@dataclass(frozen=True)
class Workspace:
    runs: FixtureRuns
    results: Path
    quantize_code: int

    @property
    def quant_root(self) -> Path:
        return self.results / "quant"

    def quant_dir(self, model: str) -> Path:
        return self.quant_root / f"{model}_{INDEPENDENT}" / "seed_0"

    def report(self, model: str) -> dict:
        return json.loads((self.quant_dir(model) / "quant_report.json").read_text("utf-8"))


def quantize_args(runs, results, *extra):
    return [
        "--runs-root",
        str(runs.runs_root),
        "--processed-dir",
        str(runs.processed.root),
        "--out-root",
        str(results / "quant"),
        "--tables-dir",
        str(results / "tables"),
        *extra,
    ]


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    """Three trained runs and one scripts/quantize.py pass over them."""
    root = tmp_path_factory.mktemp("quantization")
    runs = train_fixture_runs(root)
    results = root / "results"
    code = load_script("quantize").main(quantize_args(runs, results))
    return Workspace(runs, results, code)


def val_arrays(workspace, model):
    run = workspace.runs.run(model)
    splits_path = run.directory / run.summary["splits_file"]
    return load_split_arrays(workspace.runs.processed.root, splits_path, ["val"])["val"]


def test_quantize_writes_every_variant_and_its_report(workspace):
    assert workspace.quantize_code == 0
    for model in MODEL_NAMES:
        directory = workspace.quant_dir(model)
        expected = sorted(
            [*(variant_file(variant) for variant in QUANT_VARIANTS), "quant_report.json"]
        )
        assert sorted(path.name for path in directory.iterdir()) == expected  # No scratch files.

        report = workspace.report(model)
        assert (report["error"], report["failures"]) == (None, [])
        assert report["val_sequences"] == VAL_SEQUENCES
        assert report["checkpoint_sha256"] == sha256_file(
            workspace.runs.run(model).directory / "best.pt"
        )
        # The fixture has fewer than 1024 train sequences, so all of them calibrate.
        assert report["calibration"] == {
            "split": "train",
            "requested_samples": 1024,
            "samples": TRAIN_SEQUENCES,
            "train_sequences": TRAIN_SEQUENCES,
            "seed": 0,
            "batch_size": 64,
        }
        assert set(report["tool_versions"]) == {"python", "numpy", "torch", "onnx", "onnxruntime"}
        assert {"git_commit", "cpu_model", "packages"} <= set(report["run_info"])
        for variant in QUANT_VARIANTS:
            entry = report["variants"][variant]
            assert (entry["status"], entry["error"]) == ("built", None)
            assert entry["onnx_checker"] == {"passed": True, "error": None}
            assert entry["ort_load"] == {"passed": True, "error": None}
            assert entry["size_bytes"] == (directory / variant_file(variant)).stat().st_size
            graph = entry["graph"]
            assert sum(graph["op_counts"].values()) == graph["node_count"]
            assert entry["val"]["all_finite"]
            assert entry["val"]["diff_samples"] == VAL_SEQUENCES  # min(512, val size)
        assert report["variants"]["fp16"]["feature_block_fp32"] is False
        assert report["variants"]["int8_dynamic"]["settings"] == {
            "weight_type": "QInt8",
            "per_channel": True,
        }
        assert report["variants"]["int8_static"]["settings"] == {
            "quant_format": "QDQ",
            "activation_type": "QUInt8",
            "weight_type": "QInt8",
            "per_channel": True,
            "reduce_range": False,
            "calibrate_method": "MinMax",
        }


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_every_variant_passes_the_checker_and_runs_in_onnxruntime(workspace, model):
    val = val_arrays(workspace, model)
    source = sha256_file(workspace.runs.run(model).directory / "best.pt")
    for variant in QUANT_VARIANTS:
        path = workspace.quant_dir(model) / variant_file(variant)
        onnx.checker.check_model(str(path))
        session = OrtModel(path)
        for batch in (1, 7):
            logits = session.run(val.X[:batch])
            assert logits.shape == (batch, len(SIGNS))
            assert logits.dtype == np.float32
            assert np.isfinite(logits).all()
        metadata = read_metadata(path)
        assert metadata[META_VARIANT] == variant
        assert metadata[META_SOURCE_SHA256] == source
        assert json.loads(metadata[META_LABEL_MAP]) == {
            str(i): sign for i, sign in enumerate(SIGNS)
        }


def test_int8_variants_are_smaller_than_fp32(workspace):
    for model in MODEL_NAMES:
        variants = workspace.report(model)["variants"]
        ratio = {
            variant: variants[variant]["size_bytes"] / variants["fp32"]["size_bytes"]
            for variant in QUANT_VARIANTS
        }
        print(f"{model} size vs fp32: " + ", ".join(f"{v} {r:.3f}" for v, r in ratio.items()))
        if model == "gru":
            continue  # GRU nodes stay float under both int8 modes, so its sizes are only recorded.
        assert ratio["int8_static"] < 0.5
        assert ratio["int8_dynamic"] < 1.0


def test_variants_agree_with_fp32_on_val(workspace):
    for model in MODEL_NAMES:
        report = workspace.report(model)
        val = {variant: report["variants"][variant]["val"] for variant in QUANT_VARIANTS}
        print(
            f"{model} val agreement: "
            + ", ".join(f"{v} {m['agreement_with_fp32']:.4f}" for v, m in val.items())
        )
        assert val["fp32"]["agreement_with_fp32"] == 1.0
        assert val["fp16"]["agreement_with_fp32"] >= 0.99
        assert val["int8_dynamic"]["agreement_with_fp32"] >= 0.90
        assert val["int8_static"]["agreement_with_fp32"] >= 0.90
        # The exported graph reproduces the checkpoint's own val top-1 (one near-tie allowed).
        assert val["fp32"]["top1"] == pytest.approx(
            report["checkpoint_val_top1"], abs=1 / VAL_SEQUENCES
        )


def test_quant_ops_summary_counts_what_stayed_float(workspace):
    table = pd.read_csv(workspace.results / "tables" / "quant_ops_summary.csv")
    assert table.columns.tolist() == list(QUANT_OPS_COLUMNS)
    assert len(table) == len(MODEL_NAMES) * len(QUANT_VARIANTS)
    assert (table["failures"] == 0).all()
    rows = table.set_index(["model", "variant"])
    assert (rows.xs("fp32", level="variant")["size_ratio_vs_fp32"] == 1.0).all()
    assert (rows.xs("fp16", level="variant")["fraction_compute_ops_quantised"] == 0.0).all()
    assert rows.loc[("cnn", "int8_static"), "fraction_compute_ops_quantised"] == 1.0
    assert rows.loc[("transformer", "int8_static"), "fraction_compute_ops_quantised"] == 1.0
    gru_layers = default_model_config("gru").num_layers
    for model in MODEL_NAMES:
        assert (
            rows.loc[model, "gru_float_remaining"] == (gru_layers if model == "gru" else 0)
        ).all()

    gru = workspace.report("gru")["variants"]
    assert gru["int8_static"]["graph"]["any_gru_float"]
    assert gru["int8_static"]["graph"]["integer_and_qdq_nodes"]["QuantizeLinear"] > 0
    assert gru["int8_dynamic"]["graph"]["integer_and_qdq_nodes"]["MatMulInteger"] > 0
    assert (
        workspace.report("cnn")["variants"]["int8_dynamic"]["graph"]["integer_and_qdq_nodes"][
            "ConvInteger"
        ]
        > 0
    )


def test_existing_variants_are_reused_without_overwrite(workspace, tmp_path):
    results = tmp_path / "results"
    shutil.copytree(workspace.quant_root, results / "quant")
    before = {key: value for key, value in snapshot(results).items() if key.endswith(".onnx")}

    assert load_script("quantize").main(quantize_args(workspace.runs, results)) == 0

    after = snapshot(results)
    assert {key: after[key] for key in before} == before
    for model in MODEL_NAMES:
        report = json.loads(
            (
                results / "quant" / f"{model}_{INDEPENDENT}" / "seed_0" / "quant_report.json"
            ).read_text("utf-8")
        )
        assert {entry["status"] for entry in report["variants"].values()} == {"existing"}
        assert report["variants"]["fp16"]["feature_block_fp32"] is False
        assert report["calibration"] == workspace.report(model)["calibration"]


def test_overwrite_rebuilds_and_files_from_another_checkpoint_are_refused(workspace, tmp_path):
    results = tmp_path / "results"
    shutil.copytree(workspace.quant_root, results / "quant")
    quantize = load_script("quantize")

    cnn = results / "quant" / f"cnn_{INDEPENDENT}" / "seed_0"
    stamps = {path.name: path.stat().st_mtime_ns for path in cnn.glob("*.onnx")}
    assert (
        quantize.main(quantize_args(workspace.runs, results, "--models", "cnn", "--overwrite")) == 0
    )
    report = json.loads((cnn / "quant_report.json").read_text("utf-8"))
    assert {entry["status"] for entry in report["variants"].values()} == {"built"}
    assert all(path.stat().st_mtime_ns != stamps[path.name] for path in cnn.glob("*.onnx"))

    gru = results / "quant" / f"gru_{INDEPENDENT}" / "seed_0"
    save_with_metadata(
        onnx.load(str(gru / "fp16.onnx")), gru / "fp16.onnx", {META_SOURCE_SHA256: "0" * 64}
    )
    assert quantize.main(quantize_args(workspace.runs, results, "--models", "gru")) == 1
    report = json.loads((gru / "quant_report.json").read_text("utf-8"))
    assert report["failures"] == ["fp16"]
    assert "--overwrite" in report["variants"]["fp16"]["error"]
    assert report["variants"]["int8_static"]["status"] == "existing"
    table = pd.read_csv(results / "tables" / "quant_ops_summary.csv").set_index(
        ["model", "variant"]
    )
    assert table.loc[("gru", "fp16"), "failures"] == 1
    assert table["failures"].sum() == 1


def test_a_failing_variant_is_recorded_and_the_others_still_build(
    workspace, tmp_path, monkeypatch, capsys
):
    def unsupported(*args, **kwargs):
        raise RuntimeError("unsupported op in this test")

    monkeypatch.setattr(pipeline, "quantize_int8_dynamic", unsupported)
    results = tmp_path / "results"
    assert (
        load_script("quantize").main(quantize_args(workspace.runs, results, "--models", "cnn")) == 1
    )

    directory = results / "quant" / f"cnn_{INDEPENDENT}" / "seed_0"
    report = json.loads((directory / "quant_report.json").read_text("utf-8"))
    assert report["failures"] == ["int8_dynamic"]
    entry = report["variants"]["int8_dynamic"]
    assert entry["status"] == "failed"
    assert entry["error"] == "RuntimeError: unsupported op in this test"
    assert not (directory / "int8_dynamic.onnx").exists()
    assert report["variants"]["int8_static"]["status"] == "built"
    assert "1 variant(s) failed" in capsys.readouterr().err


def float_types_in_scope(model, scope):
    """Float element types of the initializers and constants that nodes under ``scope`` read."""
    initializers = {tensor.name: tensor.data_type for tensor in model.graph.initializer}
    types = set()
    for node in model.graph.node:
        if node.name.startswith(scope):
            types |= {initializers[name] for name in node.input if name in initializers}
            types |= {a.t.data_type for a in node.attribute if a.type == onnx.AttributeProto.TENSOR}
    return types & {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}


def test_fp16_falls_back_to_a_float32_feature_block(workspace, tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "FP16_MIN_AGREEMENT", 1.01)  # Unreachable: forces the fallback.
    run = workspace.runs.run("gru")
    processed = workspace.runs.processed.root
    report = quantize_run(run, processed, tmp_path, variants=("fp32", "fp16"), log=lambda _: None)

    entry = report["variants"]["fp16"]
    assert entry["feature_block_fp32"] is True
    assert "needs >= 1.01" in entry["fallback_reason"]
    assert [attempt["feature_block_fp32"] for attempt in entry["attempts"]] == [False, True]
    assert entry["onnx_checker"]["passed"] and entry["ort_load"]["passed"]
    blocked = onnx.load(str(tmp_path / "fp16.onnx"))
    assert float_types_in_scope(blocked, FEATURE_SCOPE) == {onnx.TensorProto.FLOAT}
    assert onnx.TensorProto.FLOAT16 in {tensor.data_type for tensor in blocked.graph.initializer}
    plain = onnx.load(str(workspace.quant_dir("gru") / "fp16.onnx"))
    assert onnx.TensorProto.FLOAT16 in float_types_in_scope(plain, FEATURE_SCOPE)

    # A rerun reuses the file and reads the decision back from its metadata.
    again = quantize_run(run, processed, tmp_path, variants=("fp32", "fp16"), log=lambda _: None)
    assert again["variants"]["fp16"]["status"] == "existing"
    assert again["variants"]["fp16"]["feature_block_fp32"] is True
    assert again["variants"]["fp16"]["fallback_reason"] == entry["fallback_reason"]


def quantized_tensors(path):
    """Tensors read by the QuantizeLinear nodes of the ONNX file at ``path``."""
    nodes = onnx.load(str(path)).graph.node
    return {node.input[0] for node in nodes if node.op_type == "QuantizeLinear"}


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_int8_static_keeps_the_input_and_feature_block_in_float32(workspace, model):
    directory = workspace.quant_dir(model)
    fp32 = onnx.load(str(directory / "fp32.onnx"))
    output, internal = feature_block_tensors(fp32)
    (projection,) = [node for node in fp32.graph.node if output in node.input]
    assert projection.op_type == "MatMul"

    read = quantized_tensors(directory / "int8_static.onnx")
    assert INPUT_NAME not in read
    assert not read & internal
    assert output in read  # Quantisation starts at the projection input.

    entry = workspace.report(model)["variants"]["int8_static"]
    assert entry["input_quantised"] is False
    assert entry["excluded_node_count"] == len(entry["excluded_node_names"]) > 0
    assert all(name.startswith(FEATURE_SCOPE) for name in entry["excluded_node_names"])


def test_int8_static_full_quantises_the_input_with_the_same_calibration(workspace):
    for model in MODEL_NAMES:
        entries = workspace.report(model)["variants"]
        full, static = entries["int8_static_full"], entries["int8_static"]
        assert full["input_quantised"] is True
        assert INPUT_NAME in quantized_tensors(workspace.quant_dir(model) / "int8_static_full.onnx")
        assert (full["excluded_node_count"], full["excluded_node_names"]) == (0, [])
        assert full["settings"] == static["settings"]
        assert full["calibration"] == static["calibration"]
        for variant in ("fp32", "fp16", "int8_dynamic"):
            entry = entries[variant]
            assert (entry["excluded_node_count"], entry["input_quantised"]) == (0, False)


def exposing(path, tensor, target):
    """Write to ``target`` a copy of the ONNX file at ``path`` that also outputs ``tensor``."""
    model = onnx.load(str(path))
    model.graph.output.append(
        onnx.helper.make_tensor_value_info(tensor, onnx.TensorProto.FLOAT, None)
    )
    onnx.save(model, str(target))
    return target


@pytest.mark.parametrize("model", MODEL_NAMES)
def test_near_zero_coordinates_reach_the_projection_as_in_fp32(workspace, model, tmp_path):
    """One landmark sits near the origin, where shoulder normalisation puts the torso centre,
    and moves slightly. int8_static must hand the projection the same features as fp32."""
    directory = workspace.quant_dir(model)
    output, _ = feature_block_tensors(onnx.load(str(directory / "fp32.onnx")))
    x = val_arrays(workspace, model).X[:1].copy()
    phase = 2 * np.pi * np.arange(NUM_FRAMES) / NUM_FRAMES
    column = 2 * RIGHT_HAND.start  # The right wrist's x; its y follows.
    x[0, :, column] = 0.004 + 0.0002 * np.sin(phase)
    x[0, :, column + 1] = -0.003 + 0.0002 * np.cos(phase)

    block = load_checkpoint(workspace.runs.run(model).directory / "best.pt").model.features
    with torch.no_grad():
        expected = block(torch.from_numpy(x)).numpy()
        velocity = block.raw_features(torch.from_numpy(x)).numpy()[0, 1:, FRAME_DIM + column]
    assert (velocity != 0.0).all()  # Present and moving in every frame, as fp32 sees it.

    features = {}
    for variant in ("fp32", "int8_static", "int8_static_full"):
        copy = exposing(directory / variant_file(variant), output, tmp_path / variant_file(variant))
        (features[variant],) = OrtModel(copy).session.run([output], {INPUT_NAME: x})
    np.testing.assert_allclose(features["fp32"], expected, rtol=0, atol=1e-5)
    np.testing.assert_allclose(features["int8_static"], features["fp32"], rtol=0, atol=1e-5)

    # Recorded, not asserted: what quantising the raw input does to the same features. Undoing the
    # standardisation shows the landmark in input units, and whether its velocity was zeroed.
    std, mean = block.std.clamp_min(STD_FLOOR).numpy(), block.mean.numpy()
    raw = {variant: value * std + mean for variant, value in features.items()}
    coordinates, velocities = [column, column + 1], [FRAME_DIM + column, FRAME_DIM + column + 1]
    difference = np.abs(features["int8_static_full"] - features["fp32"])
    print(
        f"{model} int8_static_full vs fp32 FeatureBlock output: max abs difference "
        f"{difference.max():.4g} (standardised); near-zero landmark in input units: coordinate "
        f"difference {np.abs(raw['int8_static_full'] - raw['fp32'])[..., coordinates].max():.4g}, "
        f"max |velocity| fp32 {np.abs(raw['fp32'][..., velocities]).max():.3g} vs "
        f"int8_static_full {np.abs(raw['int8_static_full'][..., velocities]).max():.3g}"
    )


def rewrite_metadata(path, *, drop, values=None):
    """Rewrite the metadata of an ONNX file without the keys in ``drop`` and with ``values``."""
    model = onnx.load(str(path))
    properties = {entry.key: entry.value for entry in model.metadata_props if entry.key not in drop}
    onnx.helper.set_model_props(model, {**properties, **(values or {})})
    onnx.save(model, str(path))


def test_an_old_schema_int8_static_is_rebuilt_and_other_variants_are_kept(workspace, tmp_path):
    results = tmp_path / "results"
    directory = results / "quant" / f"cnn_{INDEPENDENT}" / "seed_0"
    shutil.copytree(workspace.quant_dir("cnn"), directory)
    # A run quantised before schema versions: no int8_static_full, and an int8_static built
    # without exclusions.
    (directory / "int8_static_full.onnx").replace(directory / "int8_static.onnx")
    rewrite_metadata(
        directory / "int8_static.onnx",
        drop=(META_SCHEMA_VERSION, pipeline.META_EXCLUDED_NODES),
        values={META_VARIANT: "int8_static"},
    )
    for variant in ("fp32", "fp16", "int8_dynamic"):
        rewrite_metadata(directory / variant_file(variant), drop=(META_SCHEMA_VERSION,))
    stamps = {path.name: path.stat().st_mtime_ns for path in directory.glob("*.onnx")}

    quantize = load_script("quantize")
    assert quantize.main(quantize_args(workspace.runs, results, "--models", "cnn")) == 0

    report = json.loads((directory / "quant_report.json").read_text("utf-8"))
    statuses = {variant: entry["status"] for variant, entry in report["variants"].items()}
    assert statuses == {
        "fp32": "existing",
        "fp16": "existing",
        "int8_dynamic": "existing",
        "int8_static": "built",
        "int8_static_full": "built",
    }
    for variant in ("fp32", "fp16", "int8_dynamic"):
        assert (directory / variant_file(variant)).stat().st_mtime_ns == stamps[
            variant_file(variant)
        ]
    assert (directory / "int8_static.onnx").stat().st_mtime_ns != stamps["int8_static.onnx"]
    static = report["variants"]["int8_static"]
    assert static["rebuild_reason"] == "built with variant schema 1; current is 2"
    assert static["input_quantised"] is False
    assert read_metadata(directory / "int8_static.onnx")[META_SCHEMA_VERSION] == "2"

    # Once rebuilt, the file is current and reused.
    assert quantize.main(quantize_args(workspace.runs, results, "--models", "cnn")) == 0
    report = json.loads((directory / "quant_report.json").read_text("utf-8"))
    assert {entry["status"] for entry in report["variants"].values()} == {"existing"}


def test_int8_static_fails_if_its_exclusions_find_nothing_or_are_ignored(
    workspace, tmp_path, monkeypatch
):
    def build():
        return quantize_run(
            workspace.runs.run("cnn"),
            workspace.runs.processed.root,
            tmp_path,
            variants=("fp32", "int8_static"),
            log=lambda _: None,
        )

    real = variants.quantize_static

    def ignoring_exclusions(*args, nodes_to_exclude=None, **kwargs):
        return real(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(variants, "quantize_static", ignoring_exclusions)
        report = build()
    entry = report["variants"]["int8_static"]
    assert (report["failures"], entry["status"]) == (["int8_static"], "failed")
    assert "landmarks_QuantizeLinear" in entry["error"]
    assert not (tmp_path / "int8_static.onnx").exists()

    with monkeypatch.context() as patch:
        patch.setattr(variants, "FEATURE_SCOPE", "/no_such_scope/")
        report = build()
    assert report["failures"] == ["int8_static"]
    assert "no nodes under '/no_such_scope/'" in report["variants"]["int8_static"]["error"]
    assert not (tmp_path / "int8_static.onnx").exists()


def test_calibration_reader_reads_the_train_split_only(workspace, monkeypatch):
    run = workspace.runs.run("cnn")
    splits_path = run.directory / run.summary["splits_file"]
    processed = workspace.runs.processed.root
    requested = []
    real = calibration.load_split_arrays

    def spy(processed_dir, path, split_names, allow_test=False):
        requested.append(list(split_names))
        return real(processed_dir, path, split_names, allow_test)

    monkeypatch.setattr(calibration, "load_split_arrays", spy)
    reader = TrainCalibrationReader(processed, splits_path, n_samples=100)
    again = TrainCalibrationReader(processed, splits_path, n_samples=100)
    everything = TrainCalibrationReader(processed, splits_path)
    assert requested == [["train"]] * 3

    splits = read_splits(splits_path)
    chosen = set(reader.sequence_ids.tolist())
    assert chosen <= set(splits["train"])
    assert not chosen & set(splits["val"] + splits["test"])
    np.testing.assert_array_equal(again.sequence_ids, reader.sequence_ids)  # Seeded.
    assert len(reader) == 100
    assert len(everything) == TRAIN_SEQUENCES  # Fewer than 1024 exist.
    assert [len(batch["landmarks"]) for batch in iter(reader.get_next, None)] == [64, 36]
    reader.rewind()
    assert reader.get_next() is not None


def test_calibration_reader_cannot_be_pointed_at_val_or_test(workspace):
    run = workspace.runs.run("cnn")
    splits_path = run.directory / run.summary["splits_file"]
    parameters = list(inspect.signature(TrainCalibrationReader).parameters)
    assert parameters == ["processed_dir", "splits_path", "n_samples", "seed", "batch_size"]
    for name in ("split", "split_names", "allow_test"):
        with pytest.raises(TypeError):
            TrainCalibrationReader(workspace.runs.processed.root, splits_path, **{name: "val"})

    # The module asks load_split_arrays for one literal list: the train split.
    calls = [
        node
        for node in ast.walk(ast.parse(inspect.getsource(calibration)))
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "load_split_arrays"
    ]
    assert len(calls) == 1
    assert ast.literal_eval(calls[0].args[2]) == ["train"]
    assert not calls[0].keywords


def load_baseline_final_eval():
    spec = importlib.util.spec_from_file_location("baseline_final_eval", BASELINE_FINAL_EVAL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def aggregate_val(workspace, results):
    output = results / "tables" / "val_summary.csv"
    args = ["--runs-root", str(workspace.runs.runs_root), "--output", str(output)]
    assert load_script("aggregate_val").main(args) == 0


def final_eval_args(workspace, results, *extra):
    return [
        "--runs-root",
        str(workspace.runs.runs_root),
        "--tables-dir",
        str(results / "tables"),
        "--final-dir",
        str(results / "final"),
        "--device",
        "cpu",
        *extra,
    ]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_final_eval_quant_root_adds_tables_and_flagless_results_are_unchanged(workspace, tmp_path):
    before, after = tmp_path / "before", tmp_path / "after"
    for results in (before, after):
        aggregate_val(workspace, results)
    assert load_baseline_final_eval().main(final_eval_args(workspace, before)) == 0
    final_eval = load_script("final_eval")
    assert final_eval.main(final_eval_args(workspace, after)) == 0

    files = sorted(
        path.relative_to(before).as_posix() for path in before.rglob("*") if path.is_file()
    )
    assert files == sorted(
        path.relative_to(after).as_posix() for path in after.rglob("*") if path.is_file()
    )
    for name in files:
        if name == "tables/top_confusions.csv":
            # The script labels this table itself, so only it gains the experiment column the
            # library adds to every other table; the rest of the file is unchanged.
            confusions = pd.read_csv(after / name, keep_default_na=False)
            assert (confusions["experiment"] == confusions["model"] + "_" + INDEPENDENT).all()
            stripped = confusions.drop(columns="experiment").to_csv(index=False).encode()
            expected = (before / name).read_bytes()
            assert stripped.replace(b"\r\n", b"\n") == expected.replace(b"\r\n", b"\n"), name
        elif name != ACCESS_LOG.as_posix():  # Timestamps differ; compared below.
            assert (after / name).read_bytes() == (before / name).read_bytes(), name
    old_log, new_log = read_json(before / ACCESS_LOG), read_json(after / ACCESS_LOG)
    assert [entry["runs_evaluated"] for entry in new_log] == [
        entry["runs_evaluated"] for entry in old_log
    ]
    assert [set(entry) for entry in new_log] == [set(entry) for entry in old_log]

    stamps = snapshot(after)
    assert (
        final_eval.main(
            final_eval_args(workspace, after, "--quant-root", str(workspace.quant_root))
        )
        == 0
    )
    current = snapshot(after)
    unchanged = set(stamps) - {ACCESS_LOG.as_posix()}
    assert {key: current[key] for key in unchanged} == {key: stamps[key] for key in unchanged}

    table = pd.read_csv(after / "tables" / "test_quant_summary.csv")
    assert table.columns.tolist() == list(QUANT_TEST_SUMMARY_COLUMNS)
    assert len(table) == len(MODEL_NAMES) * len(QUANT_VARIANTS)
    assert (table["n_seeds"] == 1).all()
    fp32_rows = table.loc[table["variant"] == "fp32"]
    assert (fp32_rows["delta_vs_fp32_mean"] == 0.0).all()
    assert (fp32_rows["agreement_mean"] == 1.0).all()
    for model in MODEL_NAMES:
        torch_metrics = read_json(
            after / "final" / f"{model}_{INDEPENDENT}" / "seed_0" / "test_metrics.json"
        )
        for variant in QUANT_VARIANTS:
            record = read_json(
                after / "final" / "quant" / model / "seed_0" / f"{variant}_test_metrics.json"
            )
            assert (record["model"], record["variant"]) == (model, variant)
            assert record["evaluator"] == f"onnx_{variant}"
            assert record["n_sequences"] == TEST_SEQUENCES
            assert len(record["per_participant"]) == 2  # The two test signers.
            assert 0.0 <= record["agreement_with_fp32"] <= 1.0
        fp32 = read_json(after / "final" / "quant" / model / "seed_0" / "fp32_test_metrics.json")
        assert fp32["torch_test_top1"] == torch_metrics["test_top1"]
        assert fp32["torch_reference"].endswith("test_metrics.json")
        assert abs(fp32["test_top1"] - torch_metrics["test_top1"]) <= 0.001

    log = read_json(after / ACCESS_LOG)
    assert len(log) == 2
    assert log[1]["runs_evaluated"] == [
        f"quant/{key}/{variant}" for key in RUN_KEYS for variant in QUANT_VARIANTS
    ]


def test_final_eval_quant_root_checks_fp32_against_the_checkpoint(workspace, tmp_path, capsys):
    results = tmp_path / "results"
    aggregate_val(workspace, results)
    final_eval = load_script("final_eval")
    args = final_eval_args(workspace, results, "--quant-root", str(workspace.quant_root))
    assert final_eval.main(args) == 0
    fp32 = read_json(results / "final" / "quant" / "gru" / "seed_0" / "fp32_test_metrics.json")
    assert fp32["torch_reference"] == "computed from best.pt"
    (entry,) = read_json(results / ACCESS_LOG)
    assert set(RUN_KEYS) <= set(entry["runs_evaluated"])  # The checkpoints were read too.

    # A torch result more than 0.1 percentage points away stops the evaluation.
    target = results / "final" / f"gru_{INDEPENDENT}" / "seed_0" / "test_metrics.json"
    target.parent.mkdir(parents=True)
    wrong = (fp32["test_top1"] + 0.5) % 1.0
    target.write_text(
        json.dumps({"evaluator": "torch_fp32", "n_sequences": TEST_SEQUENCES, "test_top1": wrong}),
        encoding="utf-8",
    )
    (results / "tables" / "test_quant_summary.csv").unlink()
    capsys.readouterr()
    assert final_eval.main(args) == 1
    assert "differs from the torch checkpoint" in capsys.readouterr().err
    assert not (results / "tables" / "test_quant_summary.csv").exists()
    assert len(read_json(results / ACCESS_LOG)) == 2  # The attempt is still logged.


def test_final_eval_quant_root_needs_fp32_variants_and_then_logs_nothing(
    workspace, tmp_path, capsys
):
    results = tmp_path / "results"
    aggregate_val(workspace, results)
    args = final_eval_args(workspace, results, "--quant-root", str(tmp_path / "empty"))
    assert load_script("final_eval").main(args) == 1
    assert "quantize.py" in capsys.readouterr().err
    assert not (results / "final").exists()


def benchmark_args(workspace, out, *extra):
    return [
        "--quant-root",
        str(workspace.quant_root),
        "--runs-root",
        str(workspace.runs.runs_root),
        "--processed-dir",
        str(workspace.runs.processed.root),
        "--out",
        str(out),
        *extra,
    ]


def test_benchmark_smoke_writes_raw_summary_and_environment(workspace, tmp_path, monkeypatch):
    benchmark = load_script("benchmark")
    requested = []
    real = benchmark.load_split_arrays

    def spy(processed_dir, path, split_names, allow_test=False):
        requested.append(list(split_names))
        return real(processed_dir, path, split_names, allow_test)

    monkeypatch.setattr(benchmark, "load_split_arrays", spy)
    out = tmp_path / "results"
    argv = benchmark_args(
        workspace,
        out,
        "--iters",
        "20",
        "--warmup",
        "5",
        "--rounds",
        "1",
        "--threads",
        "1",
        "--sleep",
        "0",
    )
    assert benchmark.main(argv) == 0
    assert requested == [["val"]]  # One val load serves every model; test is never read.

    raw = pd.read_csv(out / "bench" / "bench_raw.csv")
    assert raw.columns.tolist() == list(RAW_COLUMNS)
    assert len(raw) == len(MODEL_NAMES) * len(BENCH_VARIANTS)
    assert set(raw["variant"]) == set(BENCH_VARIANTS)
    assert (raw[["round", "threads", "batch", "iters", "warmup"]] == [1, 1, 1, 20, 5]).all().all()
    ordered = raw[["min_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms"]].to_numpy()
    assert (ordered[:, 0] > 0).all()
    assert (np.diff(ordered, axis=1) >= 0).all()

    summary = pd.read_csv(out / "tables" / "bench_summary.csv")
    assert summary.columns.tolist() == list(SUMMARY_COLUMNS)
    assert len(summary) == len(raw)
    assert (summary["p50_spread_ms"] == 0.0).all()  # A single round.
    np.testing.assert_allclose(summary["p95_share_of_33ms_frame"], summary["p95_ms"] / FRAME_MS)

    env = read_json(out / "bench" / "bench_env.json")
    assert {
        "run_info",
        "cpu_flags",
        "torch_cpu_capability",
        "versions",
        "onnxruntime_providers",
        "cpu_freq_start",
        "cpu_freq_end",
        "power",
        "power_end",
        "windows_power_scheme",
        "windows_power_mode",
        "protocol",
        "warnings",
    } <= set(env)
    assert set(env["cpu_flags"]) == {"avx2", "avx512f", "avx512_vnni"}
    assert all(value is None or isinstance(value, bool) for value in env["cpu_flags"].values())
    assert "CPUExecutionProvider" in env["onnxruntime_providers"]
    assert set(env["power"]) == {"on_ac_power", "battery_percent"}
    assert (env["protocol"]["iters"], env["protocol"]["rounds"]) == (20, 1)
    assert env["protocol"]["inputs"] == "val sequences cycled in order"


def test_benchmark_throughput_mode_writes_suffixed_files(workspace, tmp_path, monkeypatch):
    monkeypatch.setattr(bench_env, "cpu_flags", lambda: dict.fromkeys(bench_env.CPU_FLAGS))
    out = tmp_path / "results"
    argv = benchmark_args(
        workspace,
        out,
        *("--models", "cnn", "--variants", "torch_fp32", "onnx_fp32", "--threads", "1", "2"),
        *("--batch", "4", "--iters", "5", "--warmup", "1", "--rounds", "2", "--sleep", "0"),
    )
    assert load_script("benchmark").main(argv) == 0
    raw = pd.read_csv(out / "bench" / "bench_raw_batch4.csv")
    assert len(raw) == 2 * 2 * 2
    assert sorted(raw["round"].unique()) == [1, 2]
    assert (raw["batch"] == 4).all()
    np.testing.assert_allclose(raw["throughput"], 4 * 1000.0 / raw["mean_ms"])
    assert len(pd.read_csv(out / "tables" / "bench_summary_batch4.csv")) == 4
    assert (out / "bench" / "bench_env_batch4.json").exists()
    assert not (out / "tables" / "bench_summary.csv").exists()


def test_benchmark_refuses_to_start_without_the_variant_files(workspace, tmp_path, capsys):
    out = tmp_path / "results"
    argv = benchmark_args(workspace, out, "--sleep", "0")
    argv[1] = str(tmp_path / "empty")
    assert load_script("benchmark").main(argv) == 1
    assert "quantize.py" in capsys.readouterr().err
    assert not out.exists()
