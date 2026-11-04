"""Browser export: threshold selection, the variant rule, meta.json and golden-file schemas, the
val-only data access, and consistency of the committed web/ model files."""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydantic
import pytest

import conftest
from edgevision import web_export
from edgevision.data.preprocess import preprocess_sequence, snap_x
from edgevision.layout import FRAME_DIM, NUM_FRAMES, NUM_LANDMARKS, layout_dict
from edgevision.quant.pipeline import quantize_run
from edgevision.runs import CHECKPOINT_FILE, CompletedRun, find_completed_runs, run_dir
from edgevision.runtime import OrtModel
from edgevision.train.config import AugmentConfig, load_train_config
from edgevision.train.data import read_splits
from edgevision.train.loop import run_training
from edgevision.web_export import (
    FALLBACK_MIN_COVERAGE,
    MODEL_CASES,
    ModelGolden,
    PreprocessGolden,
    WebMeta,
    export_web,
    load_eval_split,
    pack_features,
    rows_from_json,
    select_threshold,
    select_variant,
    sha256_file,
    softmax,
    threshold_from_logits,
)
from quant_support import FIXTURE_BATCH, FIXTURE_EPOCHS, FIXTURE_LR, FIXTURE_SPLIT
from training_support import EXPERIMENTS, REPO_ROOT, build_processed_dir, load_script

WEB_MODEL_DIR = REPO_ROOT / "web" / "public" / "model"
WEB_GOLDEN_DIR = REPO_ROOT / "web" / "tests" / "golden"


# ---------------------------------------------------------------------------------------------
# Threshold selection


def test_threshold_is_smallest_reaching_target():
    p = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([True, True, False, True, True, False])
    block = select_threshold(p, correct, target_precision=0.75)
    # Accuracy above t: 0.5 -> 4/6, 0.6 -> 4/5 (first to reach 0.75).
    assert block.threshold == 0.6
    assert block.coverage == pytest.approx(5 / 6)
    assert block.accuracy_above_threshold == pytest.approx(0.8)
    assert block.overall_val_top1 == pytest.approx(4 / 6)
    assert (block.method, block.warning, block.computed_on) == ("target_precision", None, "val")
    assert block.val_sequences == 6


def test_threshold_ties_accept_every_equal_probability():
    block = select_threshold(np.array([0.7, 0.7, 0.9]), np.array([True, False, True]), 0.9)
    assert block.threshold == 0.9
    assert block.coverage == pytest.approx(1 / 3)
    assert block.accuracy_above_threshold == 1.0


def test_threshold_accepts_everything_when_overall_accuracy_suffices():
    p = np.array([0.4, 0.8, 0.6])
    block = select_threshold(p, np.array([True, True, True]), 0.9)
    assert (block.threshold, block.coverage) == (0.4, 1.0)


def test_threshold_fallback_maximises_accuracy_above_min_coverage():
    # 20 samples, most confident first: wrong, right, right, then 17 wrong. No t reaches 0.9.
    p = np.linspace(0.99, 0.30, 20)
    correct = np.zeros(20, dtype=bool)
    correct[[1, 2]] = True
    block = select_threshold(p, correct, target_precision=0.9)
    # Accuracy 2/3 at coverage 0.15 is ignored (below 0.2); 2/4 at coverage exactly 0.2 wins.
    assert block.method == "fallback_max_accuracy"
    assert block.threshold == p[3]
    assert block.coverage == pytest.approx(FALLBACK_MIN_COVERAGE)
    assert block.accuracy_above_threshold == pytest.approx(0.5)
    assert block.warning is not None and "0.90" in block.warning
    assert block.fallback_min_coverage == FALLBACK_MIN_COVERAGE


def test_threshold_fallback_prefers_smaller_threshold_on_ties():
    p = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([False, True, False, True])  # accuracy 0, 1/2, 1/3, 2/4
    block = select_threshold(p, correct, target_precision=0.9)
    assert block.method == "fallback_max_accuracy"
    assert (block.threshold, block.coverage) == (0.6, 1.0)


@pytest.mark.parametrize(
    ("p", "correct", "target"),
    [
        (np.array([]), np.array([], dtype=bool), 0.9),
        (np.array([0.5, 0.6]), np.array([True]), 0.9),
        (np.array([0.5]), np.array([True]), 0.0),
        (np.array([0.5]), np.array([True]), 1.5),
    ],
)
def test_threshold_rejects_bad_input(p, correct, target):
    with pytest.raises(ValueError):
        select_threshold(p, correct, target)


def test_threshold_from_logits_uses_softmax_max_probability():
    logits = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.1]], dtype=np.float32)
    labels = np.array([0, 1, 0])
    probabilities = softmax(logits)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    block = threshold_from_logits(logits, labels, target_precision=0.99)
    assert block.threshold == pytest.approx(probabilities[1].max())
    assert block.coverage == pytest.approx(2 / 3)
    assert block.overall_val_top1 == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------------------------
# Variant rule


def quant_report(fp32, int8, int8_error=None):
    return {
        "variants": {
            "fp32": {"error": None, "val": {"top1": fp32}},
            "int8_static": {"error": int8_error, "val": None if int8 is None else {"top1": int8}},
        }
    }


@pytest.mark.parametrize(
    ("fp32", "int8", "error", "recommended"),
    [
        (0.80, 0.792, None, "int8_static"),  # 0.8 pp drop
        (0.85, 0.84, None, "int8_static"),  # exactly 1.0 pp, despite float noise
        (0.80, 0.785, None, "fp32"),  # 1.5 pp drop
        (0.80, 0.81, None, "int8_static"),  # int8 better
        (0.80, None, "RuntimeError: failed", "fp32"),  # int8 unavailable
    ],
)
def test_variant_rule(fp32, int8, error, recommended):
    selection = select_variant(quant_report(fp32, int8, error), "int8_static")
    assert selection.recommended == recommended
    assert selection.follows_rule == (recommended == "int8_static")
    assert selection.max_drop_pp == 1.0
    if int8 is not None:
        assert selection.drop_pp == pytest.approx((fp32 - int8) * 100)


def test_variant_rule_requires_fp32():
    with pytest.raises(ValueError, match="fp32"):
        select_variant({"variants": {"fp32": {"error": "boom"}}}, "fp32")


# ---------------------------------------------------------------------------------------------
# Export of a trained fixture run


@dataclass(frozen=True)
class Exported:
    run: CompletedRun
    processed: Path
    quant_root: Path
    out: Path
    golden: Path
    result: web_export.ExportResult

    def run_val_ids(self) -> list[int]:
        splits = read_splits(self.run.directory / self.run.summary["splits_file"])
        return list(splits["val"])


def train_cnn_run(root):
    processed = build_processed_dir(root / "processed", write_splits=False)
    runs_root = root / "runs"
    cfg = load_train_config(
        EXPERIMENTS / "train_cnn.yaml",
        processed_dir=processed.root,
        output_root=runs_root,
        epochs=FIXTURE_EPOCHS,
    ).model_copy(
        update={
            "batch_size": FIXTURE_BATCH,
            "lr": FIXTURE_LR,
            "split": FIXTURE_SPLIT,
            "augment": AugmentConfig(affine_p=0.0, temporal_p=0.0, jitter_p=0.0),
        }
    )
    run_training(cfg, seed=0, device="cpu", log=lambda _: None)
    (run,) = find_completed_runs(runs_root)
    return processed.root, run


def run_export(exported_root, run, processed, quant_root, **overrides):
    kwargs = {
        "out_dir": exported_root / "model",
        "golden_dir": exported_root / "golden",
        "landmark_blocks": conftest.landmark_blocks,
        "long_frame": conftest.long_frame,
        "data_source": "synthetic",
    }
    kwargs.update(overrides)
    return export_web(run.directory, "int8_static", quant_root, processed, **kwargs)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("web_export")
    processed, run = train_cnn_run(root)
    quant_root = root / "quant"
    summary = run.summary
    out_dir = run_dir(quant_root, summary["model"], summary["split_mode"], summary["seed"])
    report = quantize_run(
        run, processed, out_dir, variants=("fp32", "int8_static"), log=lambda _: None
    )
    assert report["failures"] == []
    export_root = root / "export"
    result = run_export(export_root, run, processed, quant_root)
    return Exported(
        run, processed, quant_root, export_root / "model", export_root / "golden", result
    )


def test_meta_json_schema(exported):
    meta = WebMeta.model_validate_json((exported.out / "meta.json").read_text("utf-8"))
    model_path = exported.out / "model.onnx"
    assert meta == exported.result.meta
    assert (meta.model_sha256, meta.model_bytes) == (
        sha256_file(model_path),
        model_path.stat().st_size,
    )
    assert meta.source_checkpoint_sha256 == sha256_file(exported.run.directory / CHECKPOINT_FILE)
    assert meta.labels == list(conftest.SIGNS)
    assert (meta.model_name, meta.variant, meta.data_source) == ("cnn", "int8_static", "synthetic")
    assert meta.input_layout.model_dump() == layout_dict()
    assert meta.num_frames == NUM_FRAMES and len(meta.landmark_order) == NUM_LANDMARKS
    assert meta.landmark_order[0] == "left_hand_0" and meta.landmark_order[-1] == "pose_16"
    assert meta.landmark_order[21] == "right_hand_0"
    assert meta.preprocessing.min_hand_frames == 3 and meta.preprocessing.use_z is False
    assert "PreprocessConfig defaults" in meta.preprocessing.source
    assert meta.opset == 17
    assert meta.threshold.computed_on == "val"
    assert meta.threshold.val_sequences == len(exported.run_val_ids())
    assert meta.variant_selection.chosen == "int8_static"


def test_meta_schema_rejects_inconsistent_files(exported):
    payload = json.loads((exported.out / "meta.json").read_text("utf-8"))
    for broken in (
        {**payload, "labels": payload["labels"][:-1]},
        {**payload, "unexpected": 1},
        {**payload, "model_sha256": "not-a-hash"},
        {**payload, "num_frames": 16},
        {**payload, "variant": "fp32"},
    ):
        with pytest.raises(pydantic.ValidationError):
            WebMeta.model_validate(broken)


def test_golden_files_schema(exported):
    preprocess = PreprocessGolden.model_validate_json(
        (exported.golden / "preprocess_cases.json").read_text("utf-8")
    )
    infos = [case.info for case in preprocess.cases]
    assert len(preprocess.cases) >= 12
    assert any(info.mirrored for info in infos)
    assert any(info.norm_method == "hand_fallback" and not info.excluded for info in infos)
    assert {info.reason for info in infos} >= {"insufficient_hand_frames", "degenerate_scale"}
    kept = [info for info in infos if not info.excluded]
    assert min(info.t_kept for info in kept) < NUM_FRAMES < max(info.t_kept for info in kept)
    assert any(info.t_kept < info.t_raw for info in kept)
    names = {case.name for case in preprocess.cases}
    assert {"right_only_longer", "both_hands_tie_shorter", "left_only_mirrored"} <= names

    model = ModelGolden.model_validate_json(
        (exported.golden / "model_cases.json").read_text("utf-8")
    )
    assert len(model.cases) == MODEL_CASES
    assert (model.source, model.split, model.variant) == ("synthetic", "val", "int8_static")
    assert model.model_sha256 == exported.result.meta.model_sha256
    assert {case.sequence_id for case in model.cases} <= set(exported.run_val_ids())
    inputs = np.array([case.input for case in model.cases], dtype=np.float32)
    logits = OrtModel(exported.out / "model.onnx", intra_op_threads=1).run(inputs)
    stored = np.array([case.logits for case in model.cases], dtype=np.float32)
    np.testing.assert_array_equal(stored, logits)


def test_golden_schema_rejects_bad_shapes(exported):
    payload = json.loads((exported.golden / "model_cases.json").read_text("utf-8"))
    short = {**payload, "cases": payload["cases"][:-1]}
    with pytest.raises(pydantic.ValidationError):
        ModelGolden.model_validate(short)
    case = payload["cases"][0]
    with pytest.raises(pydantic.ValidationError):
        ModelGolden.model_validate(
            {**payload, "cases": [{**case, "input": case["input"][:-1]}, *payload["cases"][1:]]}
        )


def check_preprocess_golden(golden: PreprocessGolden) -> None:
    """Re-run Python preprocessing on every stored input and compare with the stored outputs."""
    config = golden.config
    for case in golden.cases:
        result = preprocess_sequence(
            rows_from_json(case.rows),
            num_frames=config.num_frames,
            use_z=False,
            pose_indices=config.pose_indices,
            min_hand_frames=config.min_hand_frames,
        )
        assert dict(result.info) == case.info.model_dump(), case.name
        if result.features is None or result.hand_presence is None:
            assert case.features is None, case.name
            continue
        packed = pack_features(result.features, result.hand_presence)
        assert packed.shape == (NUM_FRAMES, FRAME_DIM)
        np.testing.assert_array_equal(
            np.array(case.features, dtype=np.float32), packed, err_msg=case.name
        )
        np.testing.assert_array_equal(
            np.array(case.presence, dtype=np.float32), result.hand_presence, err_msg=case.name
        )
    xs = np.array([item.x for item in golden.snap_x_cases])
    np.testing.assert_array_equal(snap_x(xs), [item.snapped for item in golden.snap_x_cases])


def test_golden_preprocessing_is_reproducible(exported):
    check_preprocess_golden(
        PreprocessGolden.model_validate_json(
            (exported.golden / "preprocess_cases.json").read_text("utf-8")
        )
    )


def test_export_reads_only_val(exported, tmp_path, monkeypatch):
    calls = []
    original = web_export.load_split_arrays

    def spy(processed_dir, splits_path, split_names, allow_test=False):
        calls.append((list(split_names), allow_test))
        return original(processed_dir, splits_path, split_names, allow_test)

    monkeypatch.setattr(web_export, "load_split_arrays", spy)
    run_export(tmp_path, exported.run, exported.processed, exported.quant_root)
    assert calls == [(["val"], False)]


@pytest.mark.parametrize("split", ["test", "train"])
def test_export_refuses_other_splits(exported, split):
    splits_path = exported.run.directory / exported.run.summary["splits_file"]
    with pytest.raises(ValueError, match="only the val split"):
        load_eval_split(exported.processed, splits_path, split)


def test_export_refuses_variant_from_another_checkpoint(exported, tmp_path):
    run_copy = tmp_path / "runs" / exported.run.directory.parent.name / exported.run.directory.name
    shutil.copytree(exported.run.directory, run_copy)
    with (run_copy / CHECKPOINT_FILE).open("ab") as handle:
        handle.write(b"\0")
    copy = CompletedRun(run_copy, exported.run.summary)
    with pytest.raises(ValueError, match="was not built from"):
        run_export(tmp_path / "out", copy, exported.processed, exported.quant_root)


def test_cli_prints_variant_comparison(exported, tmp_path, capsys):
    code = load_script("export_web").main(
        [
            "--run",
            str(exported.run.directory),
            "--variant",
            "fp32",
            "--quant-root",
            str(exported.quant_root),
            "--processed-dir",
            str(exported.processed),
            "--out",
            str(tmp_path / "model"),
            "--golden-dir",
            str(tmp_path / "golden"),
            "--data-source",
            "synthetic",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "variant rule: fp32 val top-1" in output and "-> recommended" in output
    assert "threshold" in output and "coverage" in output
    meta = WebMeta.model_validate_json((tmp_path / "model" / "meta.json").read_text("utf-8"))
    assert meta.variant == "fp32"


# ---------------------------------------------------------------------------------------------
# Committed files in web/


def test_committed_golden_preprocessing_matches_python():
    path = WEB_GOLDEN_DIR / "preprocess_cases.json"
    if not path.exists():
        pytest.skip("web/tests/golden/preprocess_cases.json not generated")
    check_preprocess_golden(PreprocessGolden.model_validate_json(path.read_text("utf-8")))


def test_committed_model_files_are_consistent():
    meta_path = WEB_MODEL_DIR / "meta.json"
    if not meta_path.exists():
        pytest.skip("web/public/model/meta.json not exported")
    meta = WebMeta.model_validate_json(meta_path.read_text("utf-8"))
    assert meta.input_layout.model_dump() == layout_dict()
    golden = ModelGolden.model_validate_json(
        (WEB_GOLDEN_DIR / "model_cases.json").read_text("utf-8")
    )
    assert (golden.model_sha256, golden.variant) == (meta.model_sha256, meta.variant)
    assert golden.num_classes == meta.num_classes
    model_path = WEB_MODEL_DIR / meta.model_file
    if model_path.exists():
        assert sha256_file(model_path) == meta.model_sha256
        assert model_path.stat().st_size == meta.model_bytes
