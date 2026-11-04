"""v1 outputs are unchanged by the v2 additions; see tests/v1_baseline.py for the references.

The committed real-data result tables are also rebuilt from the committed per-run records: apart
from the added experiment column (holding the old run names), every byte is the same.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from edgevision.evaluation.tables import (
    aggregate_val,
    per_participant_table,
    summarize_quant_test,
    summarize_test,
)
from edgevision.quant.pipeline import find_quant_reports, quant_ops_summary
from edgevision.runs import find_completed_runs
from edgevision.train.config import dump_train_config, load_train_config
from training_support import EXPERIMENTS, REPO_ROOT
from v1_baseline import (
    METRIC_VALUES,
    load_reference,
    metric_columns,
    platform_key,
    preprocess_digest,
    run_v1_preprocessing,
    run_v1_training,
)

V1_CONFIGS = ("train_gru.yaml", "train_cnn.yaml", "train_transformer.yaml")
V2_ONLY_KEYS = {"experiment", "feature_set", "recipe", "ema_decay"}
V1_AUGMENT_KEYS = {
    "affine_p",
    "temporal_p",
    "jitter_p",
    "rotation_deg",
    "scale_min",
    "scale_max",
    "translate",
    "window_min",
    "jitter_std",
}
RESULTS = REPO_ROOT / "results"


@pytest.fixture(scope="module")
def v1_preprocessed(tmp_path_factory):
    return run_v1_preprocessing(tmp_path_factory.mktemp("v1_preprocess"))


def test_v1_preprocessing_output_matches_the_stored_digest(v1_preprocessed):
    assert preprocess_digest(v1_preprocessed) == load_reference()["preprocess_digest"]


def test_v1_report_keeps_every_old_field_and_adds_the_landmark_selection(v1_preprocessed):
    report = json.loads((v1_preprocessed / "preprocess_report.json").read_text("utf-8"))
    old_fields = {"counts", "per_sign_kept", "mirrored_fraction", "norm_method_fractions"}
    assert old_fields | {"cache", "wall_time_s", "config", "run_info"} <= set(report)
    assert report["landmarks"]["pose_indices"] == [0, 11, 12, 13, 14, 15, 16]
    assert report["landmarks"]["pose_swap_pairs"] == [[11, 12], [13, 14], [15, 16]]
    assert report["landmarks"]["num_landmarks"] == 49
    stats = report["aspect_stats"]
    assert 0.0 < stats["p10"] <= stats["median"] <= stats["p90"]
    assert 0.0 < stats["shoulders_finite_fraction"] <= 1.0


def test_v1_training_metrics_match_the_stored_reference(tmp_path):
    """A seeded CPU run repeats exactly on the platform that recorded the reference; elsewhere
    float kernels differ in the last bits, so values are compared with a tolerance."""
    reference = load_reference()
    expected = reference["metrics"]
    metrics = run_v1_training(tmp_path)
    assert metrics.columns.tolist() == [*METRIC_VALUES, "epoch_seconds"]
    actual = metric_columns(metrics)
    if reference["metrics_platform"] == platform_key():
        assert actual == expected
        return
    assert actual["epoch"] == expected["epoch"]
    assert actual["lr"] == expected["lr"]  # The schedule is plain Python arithmetic.
    for column in ("train_loss", "val_loss"):
        np.testing.assert_allclose(actual[column], expected[column], rtol=1e-3)
    for column in ("train_top1", "val_top1", "val_top5"):
        np.testing.assert_allclose(actual[column], expected[column], atol=2 / 120)


@pytest.mark.parametrize("name", V1_CONFIGS)
def test_v1_configs_resolve_to_v1_defaults_and_dump_without_v2_keys(name, tmp_path):
    cfg = load_train_config(EXPERIMENTS / name, processed_dir=tmp_path)
    assert (cfg.feature_set, cfg.recipe, cfg.ema_decay, cfg.experiment) == ("v1", "v1", None, None)
    assert cfg.experiment_name == f"{cfg.model.name}_signer_independent"
    assert cfg.augment.v2_probabilities == (0.0, 0.0, 0.0, 0.0)
    dumped = yaml.safe_load(dump_train_config(cfg))
    assert not V2_ONLY_KEYS & set(dumped)
    assert set(dumped["augment"]) == V1_AUGMENT_KEYS
    assert load_train_config_text(dump_train_config(cfg), tmp_path) == cfg


def load_train_config_text(text, tmp_path):
    path = tmp_path / "dumped.yaml"
    path.write_text(text, encoding="utf-8")
    return load_train_config(path)


def committed_csv(name):
    """A committed table as LF text, whatever line endings the checkout uses."""
    return (RESULTS / "tables" / name).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")


def without_experiment(table, old_names):
    assert table.columns[0] == "experiment"
    assert table["experiment"].tolist() == old_names
    return table.drop(columns="experiment").to_csv(index=False, lineterminator="\n")


def read_records(pattern):
    return [json.loads(path.read_text("utf-8")) for path in sorted(RESULTS.glob(pattern))]


def test_committed_val_summary_is_rebuilt_from_the_run_summaries():
    table = aggregate_val(find_completed_runs(RESULTS / "runs"))
    names = (table["model"] + "_" + table["split_mode"]).tolist()
    assert without_experiment(table, names) == committed_csv("val_summary.csv")


def test_committed_test_tables_are_rebuilt_from_the_test_records():
    records = read_records("final/*/seed_*/test_metrics.json")
    assert len(records) == 12
    summary = summarize_test(records)
    names = (summary["model"] + "_" + summary["split_mode"]).tolist()
    assert without_experiment(summary, names) == committed_csv("test_fp32_summary.csv")
    participants = per_participant_table(records)
    names = (participants["model"] + "_" + participants["split_mode"]).tolist()
    assert without_experiment(participants, names) == committed_csv("test_per_participant.csv")


def test_committed_quant_tables_are_rebuilt_from_the_reports_and_records():
    ops = quant_ops_summary(find_quant_reports(RESULTS / "quant"))
    names = (ops["model"] + "_signer_independent").tolist()
    assert without_experiment(ops, names) == committed_csv("quant_ops_summary.csv")
    records = read_records("final/quant/*/seed_*/*_test_metrics.json")
    assert len(records) == 45
    quant = summarize_quant_test(records)
    names = (quant["model"] + "_signer_independent").tolist()
    assert without_experiment(quant, names) == committed_csv("test_quant_summary.csv")


def test_legacy_summaries_are_v1_runs():
    """Committed summary.json files predate the experiment, feature_set and recipe fields."""
    for run in find_completed_runs(RESULTS / "runs"):
        assert not {"experiment", "feature_set", "recipe"} & set(run.summary)
        assert run.experiment == run.directory.parent.name
        assert (run.feature_set, run.recipe) == ("v1", "v1")
        assert Path(run.key).parts[0] == run.experiment
