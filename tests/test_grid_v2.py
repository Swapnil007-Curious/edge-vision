"""The v2 ablation grid: configs, run order, resume, --max-hours, discovery over mixed v1 and v2
runs, seed-ensemble tables and the ensemble ONNX export."""

import contextlib
import io
import json
import shutil

import numpy as np
import onnxruntime as ort
import pandas as pd
import pytest
import torch

from conftest import landmark_blocks, long_frame
from edgevision.config import SplitConfig
from edgevision.evaluation.evaluators import TorchCheckpointEvaluator
from edgevision.evaluation.tables import (
    TEST_ENSEMBLE_COLUMNS,
    VAL_ENSEMBLE_COLUMNS,
    VAL_SUMMARY_COLUMNS,
    ensemble_scores,
    select_best_experiment,
)
from edgevision.experiments import V2_EXPERIMENTS, v2_grid_plan
from edgevision.export.ensemble import export_ensemble_onnx, load_ensemble
from edgevision.export.onnx_export import export_onnx
from edgevision.layout import INPUT_NAME, OUTPUT_NAME
from edgevision.models.registry import load_checkpoint
from edgevision.runs import find_completed_runs
from edgevision.train.config import load_train_config
from edgevision.train.data import load_split_arrays
from edgevision.train.loop import run_training
from edgevision.web_export import export_web
from training_support import EXPERIMENTS, build_processed_dir, load_script
from v2_support import V2_LAYOUT, build_v2_processed_dir

V2_NAMES = [experiment.name for experiment in V2_EXPERIMENTS]
V1_RUN = "cnn_signer_independent/seed_0"
SEEDS = ("0", "1")


def test_v2_configs_match_the_registry_and_the_v1_split():
    v1 = load_train_config(EXPERIMENTS / "train_cnn.yaml")
    for experiment in V2_EXPERIMENTS:
        cfg = load_train_config(EXPERIMENTS / experiment.config)
        assert cfg.experiment_name == experiment.name
        assert (cfg.model.name, cfg.feature_set, cfg.recipe) == (
            experiment.model,
            experiment.feature_set,
            experiment.recipe,
        )
        assert cfg.split == v1.split == SplitConfig()
        assert (cfg.batch_size, cfg.weight_decay, cfg.warmup_fraction) == (256, 0.05, 0.05)
        assert (cfg.label_smoothing, cfg.grad_clip) == (0.1, 1.0)
        assert str(cfg.processed_dir).replace("\\", "/") == "/kaggle/working/processed_v2"


def test_b_differs_from_the_v1_cnn_only_in_features():
    v1 = load_train_config(EXPERIMENTS / "train_cnn.yaml")
    b = load_train_config(EXPERIMENTS / "train_b_cnn_v2feat_v1recipe.yaml")
    ignored = {"experiment", "feature_set", "processed_dir"}
    assert b.model_dump(exclude=ignored) == v1.model_dump(exclude=ignored)


@pytest.mark.parametrize(
    ("name", "lr"), [("c_cnn_v2feat_v2recipe", 1e-3), ("d_hybrid_v2feat_v2recipe", 5e-4)]
)
def test_v2_recipe_values(name, lr):
    cfg = load_train_config(EXPERIMENTS / f"train_{name}.yaml")
    assert (cfg.epochs, cfg.patience, cfg.lr, cfg.ema_decay) == (150, 30, lr, 0.999)
    augment = cfg.augment
    assert (augment.affine_p, augment.temporal_p, augment.jitter_p) == (0.8, 0.5, 0.5)
    assert augment.v2_probabilities == (0.5, 0.5, 0.3, 0.5)


def test_v2_plan_finishes_seed_0_for_every_experiment_first():
    plan = [(experiment.name, seed) for experiment, seed in v2_grid_plan([0, 1, 2])]
    assert plan[:3] == [(name, 0) for name in V2_NAMES]
    assert plan == [(name, seed) for seed in (0, 1, 2) for name in V2_NAMES]


def grid_args(processed, runs_root, *extra):
    return [
        "--grid",
        "v2",
        "--processed-dir",
        str(processed),
        "--seeds",
        *SEEDS,
        "--output-root",
        str(runs_root),
        "--epochs",
        "1",
        "--device",
        "cpu",
        *extra,
    ]


def run_grid(argv):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = load_script("run_grid").main(argv)
    return code, output.getvalue()


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    """b, c and d at seeds 0 and 1 (one epoch each), beside a v1 CNN run whose summary.json
    is stripped to the fields it had before experiments were named."""
    work = tmp_path_factory.mktemp("grid_v2")
    processed = build_v2_processed_dir(work / "processed_v2")
    runs_root = work / "results" / "runs"
    code, output = run_grid(grid_args(processed.root, runs_root))
    assert code == 0

    v1_processed = build_processed_dir(work / "processed_v1")
    cfg = load_train_config(
        EXPERIMENTS / "train_cnn.yaml",
        processed_dir=v1_processed.root,
        output_root=runs_root,
        epochs=1,
    )
    run_training(cfg, seed=0, device="cpu", log=lambda _: None)
    summary_path = runs_root / V1_RUN / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("experiment", "feature_set", "recipe"):
        del summary[key]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return work, processed, runs_root, output


def test_grid_trains_every_experiment_seed_major(grid):
    _, _, runs_root, output = grid
    trained = [line.split("] ")[1] for line in output.splitlines() if line.endswith(": training")]
    assert trained == [f"{name} seed {seed}: training" for seed in SEEDS for name in V2_NAMES]
    for name in V2_NAMES:
        for seed in SEEDS:
            summary = json.loads((runs_root / name / f"seed_{seed}" / "summary.json").read_text())
            assert summary["experiment"] == name
            assert summary["feature_set"] == "v2"
            assert summary["epochs_run"] == 1
            expected = next(item for item in V2_EXPERIMENTS if item.name == name)
            assert (summary["model"], summary["recipe"]) == (expected.model, expected.recipe)


def copy_runs(grid, tmp_path):
    _, processed, runs_root, _ = grid
    copied = tmp_path / "runs"
    shutil.copytree(runs_root, copied)
    return processed, copied


def test_grid_resume_retrains_only_the_incomplete_run(grid, tmp_path):
    processed, runs_root = copy_runs(grid, tmp_path)
    victim = runs_root / "d_hybrid_v2feat_v2recipe" / "seed_1" / "summary.json"
    victim.unlink()
    others = {path: path.stat().st_mtime_ns for path in runs_root.glob("*/seed_*/summary.json")}
    code, output = run_grid(grid_args(processed.root, runs_root))
    assert code == 0
    assert output.count("complete, skipped") == 5
    assert output.count(": training") == 1
    assert "d_hybrid_v2feat_v2recipe seed 1: training" in output
    assert victim.exists()
    assert {path: path.stat().st_mtime_ns for path in others} == others


def test_max_hours_skips_the_runs_not_yet_started(grid, tmp_path, monkeypatch):
    processed, runs_root = copy_runs(grid, tmp_path)
    for name in ("c_cnn_v2feat_v2recipe", "d_hybrid_v2feat_v2recipe"):
        (runs_root / name / "seed_1" / "summary.json").unlink()
    module = load_script("run_grid")
    ticks = iter([0.0, *[2.5 * 3600.0] * 20])  # The grid starts, then 2.5 h have passed.
    monkeypatch.setattr(module, "clock", lambda: next(ticks))
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert module.main(grid_args(processed.root, runs_root, "--max-hours", "2")) == 0
    text = output.getvalue()
    assert ": training" not in text
    assert text.count("complete, skipped") == 4
    assert "skipped 2 run(s) not yet started: c_cnn_v2feat_v2recipe seed 1, " in text
    assert "d_hybrid_v2feat_v2recipe seed 1" in text
    assert not (runs_root / "c_cnn_v2feat_v2recipe" / "seed_1" / "summary.json").exists()


def test_max_hours_lets_runs_start_within_the_budget(grid, tmp_path, monkeypatch):
    processed, runs_root = copy_runs(grid, tmp_path)
    module = load_script("run_grid")
    monkeypatch.setattr(module, "clock", lambda: 0.0)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert module.main(grid_args(processed.root, runs_root, "--max-hours", "0.5")) == 0
    assert "time budget" not in output.getvalue()


def test_discovery_reads_summary_fields_for_mixed_runs(grid):
    _, _, runs_root, _ = grid
    runs = {run.key: run for run in find_completed_runs(runs_root)}
    assert len(runs) == 7
    legacy = runs[V1_RUN]
    assert (legacy.experiment, legacy.feature_set, legacy.recipe) == (
        "cnn_signer_independent",
        "v1",
        "v1",
    )
    b = runs["b_cnn_v2feat_v1recipe/seed_0"]
    assert (b.experiment, b.model, b.feature_set, b.recipe) == (
        "b_cnn_v2feat_v1recipe",
        "cnn",
        "v2",
        "v1",
    )


@pytest.fixture(scope="module")
def tables(grid):
    work, _, runs_root, _ = grid
    tables_dir = work / "results" / "tables"
    aggregate = load_script("aggregate_val")
    args = ["--runs-root", str(runs_root), "--output", str(tables_dir / "val_summary.csv")]
    assert aggregate.main([*args, "--device", "cpu"]) == 0
    final_eval = load_script("final_eval")
    final_args = [
        "--runs-root",
        str(runs_root),
        "--tables-dir",
        str(tables_dir),
        "--final-dir",
        str(work / "results" / "final"),
        "--device",
        "cpu",
    ]
    assert final_eval.main(final_args) == 0
    return tables_dir, work / "results" / "final"


def test_val_summary_has_one_row_per_experiment(grid, tables):
    tables_dir, _ = tables
    table = pd.read_csv(tables_dir / "val_summary.csv")
    assert table.columns.tolist() == list(VAL_SUMMARY_COLUMNS)
    assert table["experiment"].tolist() == [
        "b_cnn_v2feat_v1recipe",
        "c_cnn_v2feat_v2recipe",
        "cnn_signer_independent",
        "d_hybrid_v2feat_v2recipe",
    ]
    assert table.set_index("experiment")["n_seeds"].to_dict() == {
        "b_cnn_v2feat_v1recipe": 2,
        "c_cnn_v2feat_v2recipe": 2,
        "cnn_signer_independent": 1,
        "d_hybrid_v2feat_v2recipe": 2,
    }


def member_logits(runs_root, processed_root, name):
    paths = [runs_root / name / f"seed_{seed}" for seed in SEEDS]
    split = load_split_arrays(processed_root, paths[0] / "splits_signer_independent.json", ["val"])
    logits = [TorchCheckpointEvaluator(path / "best.pt").predict(split["val"].X) for path in paths]
    return logits, split["val"].y


def test_val_ensemble_averages_the_seeds_logits(grid, tables):
    _, processed, runs_root, _ = grid
    tables_dir, _ = tables
    table = pd.read_csv(tables_dir / "val_ensemble.csv")
    assert table.columns.tolist() == list(VAL_ENSEMBLE_COLUMNS)
    assert table["experiment"].tolist() == [
        "b_cnn_v2feat_v1recipe",
        "c_cnn_v2feat_v2recipe",
        "d_hybrid_v2feat_v2recipe",
    ]
    assert (table["n_seeds"] == 2).all() and (table["seeds"] == "0 1").all()
    assert (table["feature_set"] == "v2").all()
    assert table["recipe"].tolist() == ["v1", "v2", "v2"]
    for row in table.itertuples():
        expected = ensemble_scores(*member_logits(runs_root, processed.root, row.experiment))
        assert row.val_top1 == pytest.approx(expected["top1"], abs=1e-12)
        assert row.val_top5 == pytest.approx(expected["top5"], abs=1e-12)
        assert row.val_macro_f1 == pytest.approx(expected["macro_f1"], abs=1e-12)
        assert row.val_top1_seed_mean == pytest.approx(expected["top1_seed_mean"], abs=1e-12)


def test_final_eval_adds_test_ensembles_and_logs_the_access(grid, tables):
    _, _, runs_root, _ = grid
    tables_dir, final_dir = tables
    ensemble = pd.read_csv(tables_dir / "test_ensemble.csv")
    assert ensemble.columns.tolist() == list(TEST_ENSEMBLE_COLUMNS)
    assert ensemble["experiment"].tolist() == V2_NAMES
    assert ensemble[["test_top1", "test_top5", "test_macro_f1"]].stack().between(0.0, 1.0).all()
    summary = pd.read_csv(tables_dir / "test_fp32_summary.csv")
    assert (
        summary["experiment"].tolist()
        == pd.read_csv(tables_dir / "val_summary.csv")["experiment"].tolist()
    )
    best = select_best_experiment(pd.read_csv(tables_dir / "val_summary.csv"))
    confusions = pd.read_csv(tables_dir / "top_confusions.csv", keep_default_na=False)
    assert set(confusions["experiment"]) <= {best}
    log = json.loads((final_dir / "test_access_log.json").read_text(encoding="utf-8"))
    assert len(log) == 1
    assert sorted(log[0]["runs_evaluated"]) == sorted(
        run.key for run in find_completed_runs(runs_root)
    )
    assert (final_dir / "d_hybrid_v2feat_v2recipe" / "seed_1" / "test_metrics.json").exists()


def test_quantize_lists_v2_runs_and_skips_them(grid, tmp_path, capsys):
    _, processed, runs_root, _ = grid
    args = [
        "--runs-root",
        str(runs_root),
        "--processed-dir",
        str(processed.root),
        "--seeds",
        "1",
        "--out-root",
        str(tmp_path / "quant"),
        "--tables-dir",
        str(tmp_path / "tables"),
    ]
    assert load_script("quantize").main(args) == 1  # Only v2 runs at seed 1: nothing to build.
    captured = capsys.readouterr()
    assert "b_cnn_v2feat_v1recipe/seed_1: feature set v2 is not quantised; skipped" in captured.out
    assert "c_cnn_v2feat_v2recipe/seed_1: feature set v2 is not quantised; skipped" in captured.out
    assert "no completed signer_independent runs" in captured.err
    assert not (tmp_path / "quant").exists()


def test_web_export_refuses_v2_features(grid, tmp_path):
    _, processed, runs_root, _ = grid
    with pytest.raises(ValueError, match="v1 preprocessing only"):
        export_web(
            runs_root / "b_cnn_v2feat_v1recipe" / "seed_0",
            "fp32",
            tmp_path,
            processed.root,
            out_dir=tmp_path / "out",
            golden_dir=tmp_path / "golden",
            landmark_blocks=landmark_blocks,
            long_frame=long_frame,
        )


@pytest.mark.parametrize("name", ["b_cnn_v2feat_v1recipe", "d_hybrid_v2feat_v2recipe"])
def test_ensemble_onnx_matches_the_mean_of_the_members(grid, tmp_path, name):
    _, _, runs_root, _ = grid
    checkpoints = [runs_root / name / f"seed_{seed}" / "best.pt" for seed in SEEDS]
    ensemble_path = export_ensemble_onnx(checkpoints, tmp_path / "ensemble.onnx")
    member_paths = [
        export_onnx(load_checkpoint(path).model, tmp_path / f"member_{index}.onnx")
        for index, path in enumerate(checkpoints)
    ]
    ensemble = ort.InferenceSession(str(ensemble_path), providers=["CPUExecutionProvider"])
    members = [
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]) for path in member_paths
    ]
    torch_ensemble, label_map = load_ensemble(checkpoints)
    assert len(label_map) == 5
    for batch in (1, 7):
        generator = torch.Generator().manual_seed(batch)
        x = torch.randn(batch, 32, V2_LAYOUT.frame_dim, generator=generator)
        x[..., :118][torch.rand(batch, 32, 118, generator=generator) < 0.2] = 0.0
        (combined,) = ensemble.run([OUTPUT_NAME], {INPUT_NAME: x.numpy()})
        averaged = np.mean(
            [session.run([OUTPUT_NAME], {INPUT_NAME: x.numpy()})[0] for session in members], axis=0
        )
        difference = float(np.abs(combined - averaged).max())
        print(f"{name} ensemble batch {batch}: max abs diff {difference:.2e}")
        assert combined.shape == (batch, 5)
        assert difference < 1e-4
        with torch.no_grad():
            assert float(np.abs(torch_ensemble(x).numpy() - combined).max()) < 1e-4


def test_ensemble_refuses_members_of_different_experiments(grid):
    _, _, runs_root, _ = grid
    mixed = [
        runs_root / "b_cnn_v2feat_v1recipe" / "seed_0" / "best.pt",
        runs_root / "d_hybrid_v2feat_v2recipe" / "seed_0" / "best.pt",
    ]
    with pytest.raises(ValueError, match="same experiment"):
        load_ensemble(mixed)


def final_eval_args(work, runs_root, tables_dir, final_dir, *extra):
    return [
        "--runs-root",
        str(runs_root),
        "--tables-dir",
        str(tables_dir),
        "--final-dir",
        str(final_dir),
        "--device",
        "cpu",
        *extra,
    ]


def test_several_processed_dirs_are_matched_to_each_run_by_layout(grid, tables, tmp_path):
    work, processed, runs_root, _ = grid
    tables_dir, _ = tables
    shutil.copy(tables_dir / "val_summary.csv", tmp_path / "val_summary.csv")
    overrides = ["--processed-dir", str(work / "processed_v1"), str(processed.root)]
    args = final_eval_args(work, runs_root, tmp_path, tmp_path / "final", *overrides)
    assert load_script("final_eval").main(args) == 0
    for name in ("test_fp32_summary.csv", "test_ensemble.csv"):
        assert (tmp_path / name).read_bytes() == (tables_dir / name).read_bytes(), name


def test_a_summary_without_its_checkpoint_is_left_out(grid, tmp_path, capsys):
    """As in a fresh clone: committed summary.json files arrive without best.pt."""
    work, _, runs_root, _ = grid
    copied = tmp_path / "runs"
    shutil.copytree(runs_root, copied)
    (copied / V1_RUN / "best.pt").unlink()
    (copied / "c_cnn_v2feat_v2recipe" / "seed_1" / "best.pt").unlink()
    aggregate = load_script("aggregate_val")
    output = tmp_path / "tables" / "val_summary.csv"
    assert aggregate.main(["--runs-root", str(copied), "--output", str(output)]) == 0
    assert "c_cnn_v2feat_v2recipe: no best.pt" in capsys.readouterr().err
    ensembles = pd.read_csv(tmp_path / "tables" / "val_ensemble.csv")
    assert ensembles["experiment"].tolist() == ["b_cnn_v2feat_v1recipe", "d_hybrid_v2feat_v2recipe"]

    args = final_eval_args(work, copied, tmp_path / "tables", tmp_path / "final")
    assert load_script("final_eval").main(args) == 0
    out = capsys.readouterr().out
    assert f"{V1_RUN}: no best.pt; left out of the test evaluation" in out
    summary = pd.read_csv(tmp_path / "tables" / "test_fp32_summary.csv")
    assert "cnn_signer_independent" not in set(summary["experiment"])
    log = json.loads((tmp_path / "final" / "test_access_log.json").read_text(encoding="utf-8"))
    assert V1_RUN not in log[0]["runs_evaluated"]
    test_ensembles = pd.read_csv(tmp_path / "tables" / "test_ensemble.csv")
    assert test_ensembles["experiment"].tolist() == [
        "b_cnn_v2feat_v1recipe",
        "d_hybrid_v2feat_v2recipe",
    ]
