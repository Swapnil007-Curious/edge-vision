from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from conftest import load_cli_module
from edgevision.config import ExperimentConfig, PreprocessConfig, SplitConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_YAML = REPO_ROOT / "experiments" / "preprocess_default.yaml"


def write_yaml(path, payload):
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def minimal():
    return {"preprocess": {"data_root": "in", "output_dir": "out"}, "split": {}}


def test_preprocess_defaults():
    cfg = PreprocessConfig(data_root=Path("in"), output_dir=Path("out"))
    assert cfg.num_frames == 32
    assert cfg.use_z is False
    assert cfg.pose_indices == [0, 11, 12, 13, 14, 15, 16]
    assert cfg.min_hand_frames == 3
    assert cfg.limit is None
    assert cfg.num_workers == 4
    assert cfg.seed == 42


def test_split_defaults():
    cfg = SplitConfig()
    assert cfg.mode == "signer_independent"
    assert (cfg.n_test_signers, cfg.n_val_signers) == (4, 3)
    assert (cfg.val_fraction, cfg.test_fraction) == (0.1, 0.1)
    assert cfg.seed == 42


def test_default_yaml_loads_with_kaggle_paths():
    config = load_config(DEFAULT_YAML)
    assert isinstance(config, ExperimentConfig)
    assert config.preprocess.data_root == Path("/kaggle/input/asl-signs")
    assert config.preprocess.output_dir == Path("/kaggle/working/processed")
    assert config.split.mode == "signer_independent"


@pytest.mark.parametrize("section", ["preprocess", "split"])
def test_unknown_yaml_key_in_a_section_raises(tmp_path, section):
    payload = minimal()
    payload[section]["typo_key"] = 1
    with pytest.raises(ValidationError, match="typo_key"):
        load_config(write_yaml(tmp_path / "c.yaml", payload))


def test_unknown_top_level_key_raises(tmp_path):
    payload = {**minimal(), "extra_section": {}}
    with pytest.raises(ValidationError, match="extra_section"):
        load_config(write_yaml(tmp_path / "c.yaml", payload))


def test_non_mapping_yaml_raises(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_overrides_apply_and_are_validated(tmp_path):
    path = write_yaml(tmp_path / "c.yaml", minimal())
    config = load_config(path, data_root=Path("other"), output_dir=Path("elsewhere"), limit=7)
    assert config.preprocess.data_root == Path("other")
    assert config.preprocess.output_dir == Path("elsewhere")
    assert config.preprocess.limit == 7
    with pytest.raises(ValidationError):
        load_config(path, limit=0)


def test_overrides_absent_keep_yaml_values(tmp_path):
    payload = minimal()
    payload["preprocess"]["limit"] = 5
    config = load_config(write_yaml(tmp_path / "c.yaml", payload))
    assert config.preprocess.data_root == Path("in")
    assert config.preprocess.limit == 5


def test_cli_flags_override_yaml(tmp_path):
    module = load_cli_module()
    path = write_yaml(tmp_path / "c.yaml", minimal())
    args = module.parse_args(
        ["--config", str(path), "--data-root", "a", "--output-dir", "b", "--limit", "9"]
    )
    config = module.build_config(args)
    assert config.preprocess.data_root == Path("a")
    assert config.preprocess.output_dir == Path("b")
    assert config.preprocess.limit == 9
    assert args.no_cache is False
    assert module.parse_args(["--no-cache"]).no_cache is True


@pytest.mark.parametrize(
    "pose_indices",
    [
        [0, 11, 11, 12],  # Duplicate.
        [0, 33],  # Out of range.
        [0, 11, 12, 13],  # Half of the elbow pair: mirroring it would mislabel a side.
    ],
)
def test_invalid_pose_indices_raise(pose_indices):
    with pytest.raises(ValidationError):
        PreprocessConfig(data_root=Path("in"), output_dir=Path("out"), pose_indices=pose_indices)


def test_fractions_must_leave_room_for_train():
    with pytest.raises(ValidationError, match=r"below 1\.0"):
        SplitConfig(val_fraction=0.5, test_fraction=0.5)


def test_config_is_frozen():
    cfg = SplitConfig()
    with pytest.raises(ValidationError):
        cfg.seed = 1
