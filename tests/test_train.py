import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from pydantic import ValidationError

from conftest import SIGNS
from edgevision.layout import layout_dict
from edgevision.models.features import FeatureBlock, compute_feature_stats
from edgevision.models.registry import load_checkpoint
from edgevision.train.config import TrainConfig, dump_train_config, load_train_config
from edgevision.train.data import load_split_arrays
from edgevision.train.loop import METRICS_COLUMNS, grid_plan, warmup_cosine
from training_support import EXPERIMENTS, build_processed_dir, load_script, snapshot

SUMMARY_KEYS = {
    "model",
    "params",
    "seed",
    "split_mode",
    "experiment",
    "feature_set",
    "recipe",
    "num_classes",
    "train_sequences",
    "val_sequences",
    "best_epoch",
    "best_val_top1",
    "best_val_top5",
    "best_val_macro_f1",
    "epochs_run",
    "total_train_seconds",
    "device",
    "device_name",
    "amp",
    "torch_version",
    "determinism",
    "processed_dir",
    "splits_file",
    "run_info",
}


def train_args(config, processed_dir, output_root, *extra):
    return [
        "--config",
        str(EXPERIMENTS / f"train_{config}.yaml"),
        "--seed",
        "0",
        "--processed-dir",
        str(processed_dir),
        "--output-root",
        str(output_root),
        "--device",
        "cpu",
        *extra,
    ]


@pytest.mark.parametrize(("model", "lr"), [("gru", 1e-3), ("cnn", 1e-3), ("transformer", 5e-4)])
def test_experiment_configs_load_with_the_specified_values(model, lr):
    cfg = load_train_config(EXPERIMENTS / f"train_{model}.yaml")
    assert cfg.model.name == model
    assert cfg.lr == lr
    assert (cfg.epochs, cfg.batch_size, cfg.weight_decay) == (60, 256, 0.05)
    assert (cfg.warmup_fraction, cfg.label_smoothing, cfg.grad_clip) == (0.05, 0.1, 1.0)
    assert (cfg.patience, cfg.num_threads) == (15, 4)
    augment = cfg.augment
    assert (augment.affine_p, augment.temporal_p, augment.jitter_p) == (0.8, 0.5, 0.5)
    assert cfg.split.mode == "signer_independent"
    assert cfg.output_root == Path("results/runs")


def test_train_config_rejects_unknown_keys(tmp_path):
    raw = yaml.safe_load((EXPERIMENTS / "train_gru.yaml").read_text(encoding="utf-8"))
    raw["model"]["typo"] = 1
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValidationError, match="typo"):
        load_train_config(path)


def test_train_config_overrides_and_round_trip(tmp_path):
    cfg = load_train_config(
        EXPERIMENTS / "train_cnn.yaml",
        processed_dir=tmp_path,
        output_root=tmp_path / "runs",
        epochs=2,
        split_mode="signer_dependent",
    )
    assert (cfg.processed_dir, cfg.output_root, cfg.epochs) == (tmp_path, tmp_path / "runs", 2)
    assert cfg.split.mode == "signer_dependent"
    path = tmp_path / "resolved.yaml"
    path.write_text(dump_train_config(cfg), encoding="utf-8")
    assert load_train_config(path) == cfg


def test_warmup_then_cosine_schedule():
    factor = warmup_cosine(100, 0.05)
    assert [factor(step) for step in range(5)] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert factor(5) == pytest.approx(1.0)
    assert factor(52) == pytest.approx(0.5 * (1 + math.cos(math.pi * 47 / 95)))
    decay = [factor(step) for step in range(5, 100)]
    assert all(later <= earlier for earlier, later in itertools.pairwise(decay))
    assert 0.0 < factor(99) < 0.01


def test_grid_plan_matches_the_protocol():
    plan = grid_plan([0, 1, 2])
    independent = [run for run in plan if run[1] == "signer_independent"]
    dependent = [run for run in plan if run[1] == "signer_dependent"]
    assert len(independent) == 9
    assert {(model, seed) for model, _, seed in independent} == {
        (model, seed) for model in ("gru", "cnn", "transformer") for seed in (0, 1, 2)
    }
    assert dependent == [(model, "signer_dependent", 0) for model in ("gru", "cnn", "transformer")]
    assert [seed for _, _, seed in independent[:3]] == [0, 0, 0]  # Every model at seed 0 first.


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    return build_processed_dir(tmp_path_factory.mktemp("processed"))


@pytest.fixture(scope="module")
def smoke_run(processed, tmp_path_factory):
    output_root = tmp_path_factory.mktemp("runs")
    before = snapshot(processed.root)
    code = load_script("train").main(
        train_args("gru", processed.root, output_root, "--epochs", "3")
    )
    return code, output_root / "gru_signer_independent" / "seed_0", before


def test_train_smoke_writes_every_output(smoke_run, processed):
    code, run_dir, before = smoke_run
    assert code == 0
    assert sorted(path.name for path in run_dir.iterdir()) == [
        "best.pt",
        "config.yaml",
        "label_map.json",
        "metrics.csv",
        "splits_signer_independent.json",
        "summary.json",
    ]
    assert snapshot(processed.root) == before  # processed_dir is read-only on Kaggle.

    metrics = pd.read_csv(run_dir / "metrics.csv")
    assert metrics.columns.tolist() == list(METRICS_COLUMNS)
    assert metrics["epoch"].tolist() == [1, 2, 3]
    # One step per epoch: a one-step warmup, then cosine over the remaining two steps.
    assert metrics["lr"].tolist() == pytest.approx([1e-3, 1e-3, 5e-4])
    assert metrics[["val_top1", "val_top5", "train_top1"]].stack().between(0.0, 1.0).all()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert set(summary) == SUMMARY_KEYS
    assert (summary["model"], summary["seed"], summary["split_mode"]) == (
        "gru",
        0,
        "signer_independent",
    )
    assert (summary["experiment"], summary["feature_set"], summary["recipe"]) == (
        "gru_signer_independent",
        "v1",
        "v1",
    )
    assert summary["epochs_run"] == 3
    assert summary["best_val_top1"] == metrics["val_top1"].max()
    best_row = metrics.loc[metrics["epoch"] == summary["best_epoch"]].iloc[0]
    assert summary["best_val_top5"] == best_row["val_top5"]
    assert 0.0 <= summary["best_val_macro_f1"] <= 1.0
    assert summary["device"] == "cpu"
    assert summary["torch_version"] == torch.__version__
    assert "non-bitwise-deterministic" in summary["determinism"]
    assert {"git_commit", "cpu_model", "packages"} <= set(summary["run_info"])

    resolved = load_train_config(run_dir / "config.yaml")
    assert resolved.epochs == 3
    assert resolved.processed_dir == processed.root
    labels = json.loads((run_dir / "label_map.json").read_text(encoding="utf-8"))
    assert labels == {str(index): sign for index, sign in enumerate(SIGNS)}


def test_checkpoint_holds_everything_needed_to_rebuild(smoke_run):
    _, run_dir, _ = smoke_run
    payload = torch.load(run_dir / "best.pt", weights_only=True)
    assert {"model_name", "model_config", "num_classes", "input_layout", "label_map"} <= set(
        payload
    )
    assert payload["input_layout"] == layout_dict()
    assert {"features.mean", "features.std"} <= set(payload["state_dict"])

    loaded = load_checkpoint(run_dir / "best.pt")
    assert loaded.num_classes == 5
    assert loaded.label_map == dict(enumerate(SIGNS))
    assert not loaded.model.training
    assert not torch.equal(loaded.model.features.std, torch.ones(198))


def test_checkpoint_with_another_layout_is_refused(smoke_run, tmp_path):
    _, run_dir, _ = smoke_run
    payload = torch.load(run_dir / "best.pt", weights_only=True)
    payload["input_layout"] = {**payload["input_layout"], "frame_dim": 101}
    torch.save(payload, tmp_path / "other.pt")
    with pytest.raises(ValueError, match="input layout"):
        load_checkpoint(tmp_path / "other.pt")


def test_completed_run_is_refused_then_replaced_with_overwrite(processed, tmp_path, capsys):
    train = load_script("train")
    args = train_args("cnn", processed.root, tmp_path, "--epochs", "1")
    assert train.main(args) == 0
    summary = tmp_path / "cnn_signer_independent" / "seed_0" / "summary.json"
    stamp = summary.stat().st_mtime_ns

    assert train.main(args) == 1
    assert "pass --overwrite" in capsys.readouterr().err
    assert summary.stat().st_mtime_ns == stamp

    assert train.main([*args, "--overwrite"]) == 0
    assert summary.stat().st_mtime_ns != stamp


def test_feature_stats_come_from_the_train_split_only(tmp_path):
    data = build_processed_dir(tmp_path / "processed", extreme_eval=True)
    args = train_args("cnn", data.root, tmp_path / "runs", "--epochs", "1")
    assert load_script("train").main(args) == 0
    run_dir = tmp_path / "runs" / "cnn_signer_independent" / "seed_0"

    stored = load_checkpoint(run_dir / "best.pt").model.features
    arrays = load_split_arrays(
        data.root, run_dir / "splits_signer_independent.json", ["train", "val"]
    )
    expected = FeatureBlock()
    compute_feature_stats(expected, arrays["train"].X)
    torch.testing.assert_close(stored.mean, expected.mean)
    torch.testing.assert_close(stored.std, expected.std)
    assert stored.mean.abs().max() < 5.0

    # The check is sensitive: val values in the statistics would move them far.
    leaked = FeatureBlock()
    compute_feature_stats(leaked, np.concatenate([arrays["train"].X, arrays["val"].X]))
    assert leaked.mean.abs().max() > 100.0


def test_config_validation_errors():
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"model": {"name": "lstm"}, "processed_dir": "x", "lr": 1e-3})
    with pytest.raises(ValidationError):
        TrainConfig.model_validate({"model": {"name": "gru"}, "processed_dir": "x", "lr": 0.0})
