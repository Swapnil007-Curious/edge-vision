import json
import zipfile

import pandas as pd
import pytest

from edgevision.evaluation.tables import (
    CONFUSION_COLUMNS,
    PER_PARTICIPANT_COLUMNS,
    TEST_SUMMARY_COLUMNS,
    VAL_SUMMARY_COLUMNS,
)
from training_support import build_processed_dir, load_script

RUN_KEYS = sorted(
    f"{model}_{mode}/seed_0"
    for model in ("gru", "cnn", "transformer")
    for mode in ("signer_independent", "signer_dependent")
)


@pytest.fixture(scope="module")
def grid(tmp_path_factory):
    """Seed 0 only: three signer-independent and three signer-dependent one-epoch runs."""
    processed = build_processed_dir(tmp_path_factory.mktemp("processed"))
    results = tmp_path_factory.mktemp("work") / "results"
    argv = [
        "--processed-dir",
        str(processed.root),
        "--seeds",
        "0",
        "--output-root",
        str(results / "runs"),
        "--epochs",
        "1",
        "--device",
        "cpu",
    ]
    assert load_script("run_grid").main(argv) == 0
    return results, argv


def summaries(results):
    return {
        path.parent.relative_to(results / "runs").as_posix(): path
        for path in (results / "runs").glob("*/seed_*/summary.json")
    }


def aggregate(results):
    args = [
        "--runs-root",
        str(results / "runs"),
        "--output",
        str(results / "tables/val_summary.csv"),
    ]
    assert load_script("aggregate_val").main(args) == 0
    return pd.read_csv(results / "tables" / "val_summary.csv")


def final_eval_args(results):
    return [
        "--runs-root",
        str(results / "runs"),
        "--tables-dir",
        str(results / "tables"),
        "--final-dir",
        str(results / "final"),
        "--device",
        "cpu",
    ]


def test_grid_completes_every_planned_run(grid):
    results, _ = grid
    assert sorted(summaries(results)) == RUN_KEYS
    for path in summaries(results).values():
        assert json.loads(path.read_text(encoding="utf-8"))["epochs_run"] == 1


def test_grid_resume_skips_completed_runs(grid, capsys):
    results, argv = grid
    paths = summaries(results)
    victim = paths.pop("cnn_signer_dependent/seed_0")
    victim.unlink()  # As if the session died during this run.
    stamps = {key: path.stat().st_mtime_ns for key, path in paths.items()}
    capsys.readouterr()

    assert load_script("run_grid").main(argv) == 0

    output = capsys.readouterr().out
    assert output.count("complete, skipped") == 5
    assert output.count(": training") == 1
    assert "cnn_signer_dependent seed 0: training" in output
    assert victim.exists()
    assert {key: path.stat().st_mtime_ns for key, path in paths.items()} == stamps
    assert "best_val_top1" in output  # The final table.


def test_aggregate_val_columns(grid):
    results, _ = grid
    table = aggregate(results)
    assert table.columns.tolist() == list(VAL_SUMMARY_COLUMNS)
    assert len(table) == 6
    assert (table["n_seeds"] == 1).all()
    assert table["val_top1_mean"].between(0.0, 1.0).all()
    assert table["val_top1_std"].isna().all()  # One seed: no sample standard deviation.
    assert set(table["model"]) == {"gru", "cnn", "transformer"}


def test_final_eval_writes_every_table_and_logs_each_access(grid):
    results, _ = grid
    val_summary = aggregate(results)
    final_eval = load_script("final_eval")

    assert final_eval.main(final_eval_args(results)) == 0

    for key in RUN_KEYS:
        metrics = json.loads((results / "final" / key / "test_metrics.json").read_text("utf-8"))
        assert {"test_top1", "test_top5", "test_macro_f1", "per_participant"} <= set(metrics)
        assert metrics["evaluator"] == "torch_fp32"
        expected_signers = 4 if "independent" in key else 8
        assert len(metrics["per_participant"]) == expected_signers

    tables = results / "tables"
    summary = pd.read_csv(tables / "test_fp32_summary.csv")
    assert summary.columns.tolist() == list(TEST_SUMMARY_COLUMNS)
    assert len(summary) == 6
    per_participant = pd.read_csv(tables / "test_per_participant.csv")
    assert per_participant.columns.tolist() == list(PER_PARTICIPANT_COLUMNS)
    counts = per_participant.groupby("split_mode").size().to_dict()
    assert counts == {"signer_independent": 3 * 4, "signer_dependent": 3 * 8}

    confusions = pd.read_csv(tables / "top_confusions.csv", keep_default_na=False)
    assert confusions.columns.tolist() == ["experiment", "model", "split_mode", *CONFUSION_COLUMNS]
    assert (confusions["experiment"] == confusions["model"] + "_signer_independent").all()
    assert len(confusions) <= 30
    independent = val_summary.loc[val_summary["split_mode"] == "signer_independent"]
    best = independent.sort_values(["val_top1_mean", "model"], ascending=[False, True]).iloc[0]
    assert set(confusions["model"]) <= {best["model"]}
    assert (confusions["split_mode"] == "signer_independent").all()
    assert confusions["count"].is_monotonic_decreasing
    assert (confusions["true_label"] != confusions["predicted_label"]).all()

    log_path = results / "final" / "test_access_log.json"
    log = json.loads(log_path.read_text(encoding="utf-8"))
    assert len(log) == 1
    assert set(log[0]) == {"timestamp_utc", "git_commit", "git_dirty", "runs_evaluated"}
    assert log[0]["runs_evaluated"] == RUN_KEYS

    assert final_eval.main(final_eval_args(results)) == 0
    assert len(json.loads(log_path.read_text(encoding="utf-8"))) == 2


def test_final_eval_needs_val_summary_and_then_logs_nothing(grid, tmp_path, capsys):
    results, _ = grid
    args = final_eval_args(results)
    args[3], args[5] = str(tmp_path / "tables"), str(tmp_path / "final")
    assert load_script("final_eval").main(args) == 1
    assert "aggregate_val.py" in capsys.readouterr().err
    assert not (tmp_path / "final").exists()


def test_pack_results_zips_the_whole_results_directory(grid, tmp_path):
    results, _ = grid
    pack_results = load_script("pack_results")
    output = tmp_path / "results_bundle.zip"
    assert pack_results.main(["--results-dir", str(results), "--output", str(output)]) == 0
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    expected = {
        path.relative_to(results.parent).as_posix() for path in results.rglob("*") if path.is_file()
    }
    assert names == expected
    assert "results/runs/gru_signer_independent/seed_0/best.pt" in names
    with pytest.raises(ValueError, match="outside"):
        pack_results.pack(results, results / "bundle.zip")
