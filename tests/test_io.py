import json
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import (
    PARTICIPANTS,
    SCRIPT,
    build_fake_dataset,
    landmark_blocks,
    load_cli_module,
    long_frame,
)
from edgevision.config import PreprocessConfig
from edgevision.data.io import (
    SEQUENCE_COLUMNS,
    WorkerParams,
    _load_cached,
    attach_labels,
    cache_fingerprint,
    load_table,
    process_dataset,
    process_row,
    read_sequence,
    read_sign_map,
    read_train_csv,
    to_storage,
)
from edgevision.data.preprocess import preprocess_sequence
from edgevision.utils.runinfo import collect_run_info
from edgevision.utils.seed import set_seed

N_SEQUENCES = 120
EXPECTED_EXCLUSIONS = {"insufficient_hand_frames": 3, "degenerate_scale": 1, "read_error": 1}


def worker_params(dataset, cache_dir=None, **overrides):
    settings = {
        "data_root": dataset.root,
        "cache_dir": cache_dir,
        "num_frames": 32,
        "use_z": False,
        "pose_indices": (0, 11, 12, 13, 14, 15, 16),
        "min_hand_frames": 3,
    }
    return WorkerParams(**{**settings, **overrides})


def row_for_case(dataset, case):
    row = dataset.table.loc[dataset.table["case"] == case].iloc[0]
    return int(row["sequence_id"]), str(row["path"])


def write_train_csv(path, **replacements):
    frame = pd.DataFrame(
        {
            "path": ["a.parquet", "b.parquet"],
            "participant_id": [1, 2],
            "sequence_id": [10, 11],
            "sign": ["hello", "bye"],
        }
    )
    frame = frame.assign(**replacements)
    frame.to_csv(path, index=False)
    return path


def test_read_train_csv_valid(tmp_path):
    df = read_train_csv(write_train_csv(tmp_path / "train.csv"))
    assert len(df) == 2


def test_read_train_csv_keeps_na_like_signs(tmp_path):
    df = read_train_csv(write_train_csv(tmp_path / "train.csv", sign=["null", "NA"]))
    assert df["sign"].tolist() == ["null", "NA"]


def test_read_train_csv_empty_raises(tmp_path):
    path = tmp_path / "train.csv"
    path.write_text("path,participant_id,sequence_id,sign\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        read_train_csv(path)


def test_read_train_csv_missing_column_raises(tmp_path):
    path = tmp_path / "train.csv"
    pd.DataFrame({"path": ["a"], "sequence_id": [1], "sign": ["x"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="expected columns"):
        read_train_csv(path)


def test_read_train_csv_non_integer_id_raises(tmp_path):
    path = write_train_csv(tmp_path / "train.csv", participant_id=["x", "y"])
    with pytest.raises(ValueError, match="participant_id must be integer"):
        read_train_csv(path)


def test_read_train_csv_duplicate_sequence_raises(tmp_path):
    path = write_train_csv(tmp_path / "train.csv", sequence_id=[10, 10])
    with pytest.raises(ValueError, match="duplicate sequence_id"):
        read_train_csv(path)


def test_read_sign_map_valid(tmp_path):
    path = tmp_path / "map.json"
    path.write_text(json.dumps({"a": 1, "b": 0}), encoding="utf-8")
    assert read_sign_map(path) == {"a": 1, "b": 0}


@pytest.mark.parametrize(
    "payload",
    [{}, ["a"], {"a": "0"}, {"a": 0.0}, {"a": True}, {"a": 0, "b": 0}, {"a": 0, "b": 2}],
)
def test_read_sign_map_rejects_bad_content(tmp_path, payload):
    path = tmp_path / "map.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        read_sign_map(path)


def test_unknown_sign_raises():
    df = pd.DataFrame({"sign": ["hello", "mystery"]})
    with pytest.raises(ValueError, match="mystery"):
        attach_labels(df, {"hello": 0})


def test_load_table_applies_limit_after_validating_everything(fake_dataset):
    cfg = PreprocessConfig(data_root=fake_dataset.root, output_dir=Path("unused"), limit=10)
    table = load_table(cfg)
    assert len(table) == 10
    assert table["label"].between(0, 4).all()


def test_load_table_refuses_participant_ids_that_would_wrap(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    write_train_csv(root / "train.csv", participant_id=[1, 3_000_000_000], sign=["a", "a"])
    (root / "sign_to_prediction_index_map.json").write_text('{"a": 0}', encoding="utf-8")
    cfg = PreprocessConfig(data_root=root, output_dir=tmp_path / "out")
    with pytest.raises(OverflowError, match="participant_id"):
        load_table(cfg)


def test_read_sequence_valid(tmp_path, rng):
    path = tmp_path / "seq.parquet"
    long_frame(landmark_blocks(10, rng)).to_parquet(path, engine="pyarrow", index=False)
    df = read_sequence(path)
    assert set(df.columns) == set(SEQUENCE_COLUMNS)


def test_read_sequence_rejects_missing_column(tmp_path, rng):
    path = tmp_path / "seq.parquet"
    frame = long_frame(landmark_blocks(10, rng)).drop(columns="row_id")
    frame.to_parquet(path, engine="pyarrow", index=False)
    with pytest.raises(ValueError, match="expected columns"):
        read_sequence(path)


def test_read_sequence_rejects_wrong_dtype(tmp_path, rng):
    path = tmp_path / "seq.parquet"
    frame = long_frame(landmark_blocks(10, rng))
    frame["x"] = "text"
    frame.to_parquet(path, engine="pyarrow", index=False)
    with pytest.raises(ValueError, match="column x must be float"):
        read_sequence(path)


def test_corrupted_parquet_is_recorded_as_read_error(fake_dataset):
    outcome = process_row(worker_params(fake_dataset), row_for_case(fake_dataset, "corrupted"))
    assert outcome.features is None
    assert outcome.info["excluded"] is True
    assert re.fullmatch(r"read_error: \w+", outcome.info["reason"])


def test_missing_parquet_is_recorded_as_read_error(fake_dataset):
    outcome = process_row(worker_params(fake_dataset), (1, "train_landmark_files/nope.parquet"))
    assert outcome.info["reason"] == "read_error: FileNotFoundError"


def test_duplicate_rows_in_a_file_are_recorded_not_fatal(tmp_path, rng):
    frame = long_frame(landmark_blocks(10, rng))
    pd.concat([frame, frame.iloc[[0]]]).to_parquet(tmp_path / "dup.parquet", index=False)
    params = WorkerParams(tmp_path, None, 32, False, (0, 11, 12, 13, 14, 15, 16), 3)
    assert process_row(params, (1, "dup.parquet")).info["reason"] == "read_error: ValueError"


def test_to_storage_narrows_dtypes(rng):
    result = preprocess_sequence(long_frame(landmark_blocks(12, rng)))
    outcome = to_storage(5, result)
    assert outcome.features.dtype == np.float16
    assert outcome.presence.dtype == np.uint8
    assert outcome.info["excluded"] is False


def test_to_storage_excludes_what_float16_cannot_hold(rng):
    result = preprocess_sequence(long_frame(landmark_blocks(12, rng)))
    huge = result._replace(features=result.features * 1e5)
    outcome = to_storage(5, huge)
    assert outcome.features is None
    assert outcome.info["excluded"] is True
    assert outcome.info["reason"] == "float16_overflow"
    assert result.info["excluded"] is False


def test_cached_outcome_round_trips(fake_dataset, tmp_path):
    params = worker_params(fake_dataset, cache_dir=tmp_path)
    row = row_for_case(fake_dataset, "both_hands")
    first = process_row(params, row)
    second = process_row(params, row)
    assert (first.from_cache, second.from_cache) == (False, True)
    assert second.info == first.info
    np.testing.assert_array_equal(second.features, first.features)
    assert second.features.dtype == np.float16


def test_excluded_outcomes_are_cached_too(fake_dataset, tmp_path):
    params = worker_params(fake_dataset, cache_dir=tmp_path)
    row = row_for_case(fake_dataset, "corrupted")
    first = process_row(params, row)
    second = process_row(params, row)
    assert second.from_cache is True
    assert second.features is None
    assert second.info == first.info


def test_truncated_cache_entry_is_a_miss(fake_dataset, tmp_path):
    params = worker_params(fake_dataset, cache_dir=tmp_path)
    row = row_for_case(fake_dataset, "both_hands")
    (tmp_path / f"{row[0]}.npz").write_bytes(b"truncated")
    assert _load_cached(tmp_path / f"{row[0]}.npz", row[0]) is None
    assert process_row(params, row).from_cache is False


@pytest.mark.parametrize(
    "change",
    [
        {"num_frames": 16},
        {"use_z": True},
        {"min_hand_frames": 5},
        {"pose_indices": [0, 11, 12]},
        {"data_root": Path("other")},
    ],
)
def test_cache_fingerprint_tracks_transform_settings(change):
    base = {"data_root": Path("in"), "output_dir": Path("out")}
    original = PreprocessConfig(**base)
    assert cache_fingerprint(original) != cache_fingerprint(PreprocessConfig(**{**base, **change}))


@pytest.mark.parametrize("change", [{"num_workers": 1}, {"limit": 5}, {"seed": 1}])
def test_cache_fingerprint_ignores_settings_that_do_not_change_results(change):
    base = {"data_root": Path("in"), "output_dir": Path("out")}
    original = PreprocessConfig(**base)
    assert cache_fingerprint(original) == cache_fingerprint(PreprocessConfig(**{**base, **change}))


def test_changed_settings_do_not_reuse_stale_cache(tmp_path):
    dataset = build_fake_dataset(
        tmp_path / "data",
        participants=(1, 2),
        signs=("a",),
        sequences_per_cell=1,
        special_cases={},
    )
    common = {"data_root": dataset.root, "output_dir": tmp_path / "out", "num_workers": 1}
    first = process_dataset(PreprocessConfig(**common, num_frames=8))
    second = process_dataset(PreprocessConfig(**common, num_frames=16))
    assert first.features.shape[1] == 8
    assert second.features.shape[1] == 16
    assert second.cache_hits == 0


def test_inline_batch_matches_train_csv_order(fake_dataset, tmp_path):
    cfg = PreprocessConfig(
        data_root=fake_dataset.root, output_dir=tmp_path, limit=12, num_workers=1
    )
    batch = process_dataset(cfg, use_cache=False)
    assert (
        batch.manifest["sequence_id"].tolist()
        == fake_dataset.table["sequence_id"].head(12).tolist()
    )
    assert batch.features.shape[0] == (~batch.manifest["excluded"]).sum()
    assert not (tmp_path / "cache").exists()


def test_interrupted_run_resumes_from_the_partial_cache(fake_dataset, tmp_path):
    common = {"data_root": fake_dataset.root, "output_dir": tmp_path, "num_workers": 1}
    partial = process_dataset(PreprocessConfig(**common, limit=40))
    assert (partial.cache_hits, partial.computed) == (0, 40)

    resumed = process_dataset(PreprocessConfig(**common))
    assert (resumed.cache_hits, resumed.computed) == (40, N_SEQUENCES - 40)
    assert len(resumed.manifest) == N_SEQUENCES


def test_progress_is_printed_at_the_configured_interval(
    fake_dataset, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr("edgevision.data.io.PROGRESS_EVERY", 5)
    cfg = PreprocessConfig(
        data_root=fake_dataset.root, output_dir=tmp_path, limit=12, num_workers=1
    )
    process_dataset(cfg, use_cache=False)
    assert capsys.readouterr().out.splitlines() == ["processed 5/12", "processed 10/12"]


def test_collect_run_info_keys():
    info = collect_run_info()
    assert set(info) >= {
        "git_commit",
        "git_dirty",
        "python_version",
        "platform",
        "cpu_model",
        "cpu_logical_cores",
        "cpu_physical_cores",
        "ram_total_bytes",
        "packages",
        "timestamp_utc",
    }
    assert set(info["packages"]) == {"numpy", "pandas", "pyarrow"}
    assert info["ram_total_bytes"] > 0
    assert isinstance(info["cpu_model"], str)


def test_set_seed_is_repeatable(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    set_seed(11)
    first = (random.random(), np.random.random())  # noqa: NPY002
    set_seed(11)
    assert first == (random.random(), np.random.random())  # noqa: NPY002


def run_cli(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def cache_stamps(output_dir):
    return {path: path.stat().st_mtime_ns for path in (output_dir / "cache").rglob("*.npz")}


@pytest.fixture(scope="module")
def cli_run(fake_dataset, tmp_path_factory):
    work = tmp_path_factory.mktemp("cli")
    config_path = work / "config.yaml"
    payload = {
        "preprocess": {
            "data_root": "/nonexistent/overridden/by/flag",
            "output_dir": "/nonexistent/overridden/by/flag",
            "num_workers": 2,
        },
        "split": {},
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    output_dir = work / "processed"
    args = (
        "--config",
        str(config_path),
        "--data-root",
        str(fake_dataset.root),
        "--output-dir",
        str(output_dir),
    )
    completed = run_cli(*args)
    assert completed.returncode == 0, completed.stderr
    return args, output_dir, completed


def test_cli_writes_features(cli_run, fake_dataset):
    _, output_dir, _ = cli_run
    with np.load(output_dir / "features.npz", allow_pickle=False) as data:
        assert set(data.files) == {"X", "presence", "y", "participant_id", "sequence_id"}
        n_kept = N_SEQUENCES - sum(EXPECTED_EXCLUSIONS.values())
        assert data["X"].shape == (n_kept, 32, 49, 2)
        assert data["X"].dtype == np.float16
        assert np.isfinite(data["X"]).all()
        assert data["presence"].shape == (n_kept, 32, 2)
        assert data["presence"].dtype == np.uint8
        assert set(np.unique(data["presence"]).tolist()) <= {0, 1}
        assert data["y"].dtype == np.int16
        assert data["participant_id"].dtype == np.int32
        assert data["sequence_id"].dtype == np.int64
        # Real values above int16 and int32 must arrive unchanged.
        assert set(data["participant_id"].tolist()) == set(PARTICIPANTS)
        expected_ids = fake_dataset.table.set_index("sequence_id")["participant_id"]
        assert data["participant_id"].tolist() == expected_ids.loc[data["sequence_id"]].tolist()


def test_cli_features_align_with_manifest(cli_run):
    _, output_dir, _ = cli_run
    manifest = pd.read_csv(output_dir / "manifest.csv", keep_default_na=False)
    kept = manifest.loc[~manifest["excluded"]]
    with np.load(output_dir / "features.npz", allow_pickle=False) as data:
        assert data["sequence_id"].tolist() == kept["sequence_id"].tolist()
        assert data["y"].tolist() == kept["label"].tolist()


def test_cli_manifest_has_one_row_per_sequence(cli_run, fake_dataset):
    _, output_dir, _ = cli_run
    manifest = pd.read_csv(output_dir / "manifest.csv", keep_default_na=False)
    assert manifest.columns.tolist() == [
        "sequence_id",
        "participant_id",
        "sign",
        "label",
        "excluded",
        "reason",
        "t_raw",
        "t_kept",
        "mirrored",
        "norm_method",
    ]
    assert manifest["sequence_id"].tolist() == fake_dataset.table["sequence_id"].tolist()
    assert "null" in set(manifest["sign"])

    cases = fake_dataset.table.set_index("sequence_id")["case"]
    by_case = manifest.set_index("sequence_id").join(cases)
    reason = by_case.groupby("case")["reason"].agg(lambda values: set(values))
    assert reason["no_hands"] == {"insufficient_hand_frames"}
    assert reason["coincident_shoulders"] == {"degenerate_scale"}
    assert len(reason["corrupted"]) == 1
    assert next(iter(reason["corrupted"])).startswith("read_error: ")
    assert by_case.loc[by_case["case"] == "left_hand_only", "mirrored"].all()
    assert not by_case.loc[by_case["case"] == "right_hand_only", "mirrored"].any()
    assert (by_case.loc[by_case["case"] == "pose_missing", "norm_method"] == "hand_fallback").all()


def test_cli_report_contents(cli_run):
    _, output_dir, completed = cli_run
    report = json.loads((output_dir / "preprocess_report.json").read_text(encoding="utf-8"))
    counts = report["counts"]
    assert counts["total"] == N_SEQUENCES
    assert counts["excluded"] == sum(EXPECTED_EXCLUSIONS.values())
    assert counts["kept"] == N_SEQUENCES - counts["excluded"]

    by_reason = counts["excluded_by_reason"]
    read_errors = [key for key in by_reason if key.startswith("read_error: ")]
    assert len(read_errors) == 1
    assert by_reason[read_errors[0]] == EXPECTED_EXCLUSIONS["read_error"]
    assert by_reason["insufficient_hand_frames"] == EXPECTED_EXCLUSIONS["insufficient_hand_frames"]
    assert by_reason["degenerate_scale"] == EXPECTED_EXCLUSIONS["degenerate_scale"]

    assert report["per_sign_kept"]["min"] <= report["per_sign_kept"]["median"]
    assert report["per_sign_kept"]["median"] <= report["per_sign_kept"]["max"]
    assert 0.0 < report["mirrored_fraction"] < 1.0
    assert sum(report["norm_method_fractions"].values()) == pytest.approx(1.0)
    assert set(report["norm_method_fractions"]) == {"shoulders", "hand_fallback"}
    assert report["cache"] == {"enabled": True, "hits": 0, "computed": N_SEQUENCES}
    assert report["wall_time_s"] > 0.0
    assert report["config"]["preprocess"]["num_workers"] == 2
    assert report["run_info"]["python_version"]
    assert "kept=" in completed.stdout


def test_cli_splits_json(cli_run):
    _, output_dir, _ = cli_run
    splits = json.loads((output_dir / "splits_signer_independent.json").read_text(encoding="utf-8"))
    participants = splits["participants"]
    assert (len(participants["test"]), len(participants["val"])) == (4, 3)
    assert len(participants["train"]) == 1
    assert not (set(participants["train"]) & set(participants["test"]))
    manifest = pd.read_csv(output_dir / "manifest.csv", keep_default_na=False)
    kept_ids = set(manifest.loc[~manifest["excluded"], "sequence_id"])
    assert set(splits["train"]) | set(splits["val"]) | set(splits["test"]) == kept_ids
    assert splits["summary"]["test"]["labels"] == 5


def test_cli_rerun_uses_the_cache(cli_run):
    args, output_dir, _ = cli_run
    with np.load(output_dir / "features.npz", allow_pickle=False) as data:
        before = data["X"].copy()
    stamps = cache_stamps(output_dir)
    assert len(stamps) == N_SEQUENCES

    completed = run_cli(*args)

    assert completed.returncode == 0, completed.stderr
    report = json.loads((output_dir / "preprocess_report.json").read_text(encoding="utf-8"))
    assert report["cache"] == {"enabled": True, "hits": N_SEQUENCES, "computed": 0}
    assert cache_stamps(output_dir) == stamps
    with np.load(output_dir / "features.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["X"], before)


def test_cli_no_cache_recomputes_identical_features(cli_run, fake_dataset, tmp_path, monkeypatch):
    """Calls main() in-process with one worker; the spawned path is covered by cli_run."""
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    _, output_dir, _ = cli_run
    fresh = tmp_path / "fresh"
    config_path = tmp_path / "inline.yaml"
    payload = {
        "preprocess": {
            "data_root": str(fake_dataset.root),
            "output_dir": str(fresh),
            "num_workers": 1,
        },
        "split": {},
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    assert load_cli_module().main(["--config", str(config_path), "--no-cache"]) == 0

    report = json.loads((fresh / "preprocess_report.json").read_text(encoding="utf-8"))
    assert report["cache"] == {"enabled": False, "hits": 0, "computed": N_SEQUENCES}
    assert not (fresh / "cache").exists()
    with np.load(output_dir / "features.npz", allow_pickle=False) as data:
        reference = data["X"].copy()
    with np.load(fresh / "features.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["X"], reference)


def test_cli_help_prints_usage(capsys):
    with pytest.raises(SystemExit) as stopped:
        load_cli_module().parse_args(["--help"])
    assert stopped.value.code == 0
    output = capsys.readouterr().out
    assert "usage:" in output
    assert "--no-cache" in output
