"""EMA of weights: the update rule, and that validation and best.pt use the EMA weights."""

import copy
import json

import pytest
import torch
import yaml
from pydantic import ValidationError
from torch import nn

import edgevision.train.loop as loop
from edgevision.evaluation.evaluators import TorchCheckpointEvaluator
from edgevision.evaluation.metrics import top_k_accuracy
from edgevision.train.config import load_train_config
from edgevision.train.data import load_split_arrays
from edgevision.train.ema import ModelEMA
from training_support import EXPERIMENTS
from v2_support import build_v2_processed_dir

DECAY = 0.999


def test_update_moves_parameters_by_one_minus_decay_and_copies_buffers():
    torch.manual_seed(0)
    live = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4))
    ema = ModelEMA(live, 0.9)
    start = copy.deepcopy(ema.module.state_dict())
    assert not ema.module.training
    assert not any(parameter.requires_grad for parameter in ema.module.parameters())
    with torch.no_grad():
        for parameter in live.parameters():
            parameter.add_(1.0)
    live.train()(torch.randn(8, 3))  # Moves the BatchNorm running statistics.
    ema.update(live)
    for name, value in ema.module.named_parameters():
        expected = 0.9 * start[name] + 0.1 * dict(live.named_parameters())[name]
        torch.testing.assert_close(value, expected)
    for name, value in ema.module.named_buffers():
        assert torch.equal(value, dict(live.named_buffers())[name])
    assert ema.updates == 1


def test_decay_outside_the_open_interval_is_refused():
    with pytest.raises(ValueError, match="decay"):
        ModelEMA(nn.Linear(2, 2), 1.0)


def test_recipe_v1_refuses_ema_and_v2_augmentations(tmp_path):
    raw = yaml.safe_load((EXPERIMENTS / "train_cnn.yaml").read_text(encoding="utf-8"))
    for change in ({"ema_decay": 0.999}, {"augment": {**raw["augment"], "time_mask_p": 0.5}}):
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump({**raw, **change}), encoding="utf-8")
        with pytest.raises(ValidationError, match="recipe v1"):
            load_train_config(path)


@pytest.fixture(scope="module")
def ema_run(tmp_path_factory):
    """Experiment c (EMA 0.999) for two one-step epochs, recording what the loop builds, steps
    and saves."""
    root = tmp_path_factory.mktemp("ema")
    data = build_v2_processed_dir(root / "processed")
    cfg = load_train_config(
        EXPERIMENTS / "train_c_cnn_v2feat_v2recipe.yaml",
        processed_dir=data.root,
        output_root=root / "runs",
        epochs=2,
    )
    seen = {"saves": []}
    build, save = loop.build_model, loop.save_checkpoint

    def recording_build(*args, **kwargs):
        model = build(*args, **kwargs)
        seen["live"] = model
        seen["initial"] = copy.deepcopy(model.state_dict())
        return model

    def recording_save(path, model, *args, **kwargs):
        live = seen["live"]
        seen["saves"].append(
            {
                "is_live": model is live,
                "saved": copy.deepcopy(model.state_dict()),
                "live": copy.deepcopy(live.state_dict()),
            }
        )
        save(path, model, *args, **kwargs)

    patch = pytest.MonkeyPatch()
    patch.setattr(loop, "build_model", recording_build)
    patch.setattr(loop, "save_checkpoint", recording_save)
    try:
        summary = loop.run_training(cfg, seed=0, device="cpu", log=lambda _: None)
    finally:
        patch.undo()
    return summary, root / "runs" / "c_cnn_v2feat_v2recipe" / "seed_0", seen, data


def test_checkpoint_holds_the_ema_weights(ema_run):
    _, run_dir, seen, _ = ema_run
    first = seen["saves"][0]  # Epoch 1 always improves on the start.
    assert not any(save["is_live"] for save in seen["saves"])
    initial, live, saved = seen["initial"], first["live"], first["saved"]
    for name, value in saved.items():
        if name.endswith(("running_mean", "running_var", "num_batches_tracked")):
            assert torch.equal(value, live[name])  # Buffers are copied, not averaged.
        elif name in ("features.mean", "features.std"):
            assert torch.equal(value, live[name])
        else:  # One optimizer step: ema = decay * initial + (1 - decay) * live.
            expected = DECAY * initial[name] + (1.0 - DECAY) * live[name]
            torch.testing.assert_close(value, expected, rtol=1e-6, atol=1e-7)
    stored = torch.load(run_dir / "best.pt", weights_only=True)["state_dict"]
    last = seen["saves"][-1]["saved"]
    assert stored.keys() == last.keys()
    assert all(torch.equal(stored[name], last[name]) for name in stored)
    moved = [
        name
        for name in live
        if name.endswith("weight") and not torch.equal(live[name], saved[name])
    ]
    assert moved  # The EMA weights are not the live weights.


def test_validation_scores_the_ema_weights(ema_run):
    summary, run_dir, _, data = ema_run
    val = load_split_arrays(data.root, run_dir / "splits_signer_independent.json", ["val"])["val"]
    logits = TorchCheckpointEvaluator(run_dir / "best.pt").predict(val.X)
    assert top_k_accuracy(logits, val.y, 1) == summary["best_val_top1"]


def test_v2_summary_and_config_record_the_recipe(ema_run):
    summary, run_dir, _, _ = ema_run
    assert (summary["experiment"], summary["feature_set"], summary["recipe"]) == (
        "c_cnn_v2feat_v2recipe",
        "v2",
        "v2",
    )
    assert json.loads((run_dir / "summary.json").read_text("utf-8"))["recipe"] == "v2"
    dumped = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    assert (dumped["ema_decay"], dumped["recipe"], dumped["feature_set"]) == (DECAY, "v2", "v2")
    assert dumped["augment"]["frame_dropout_p"] == 0.5
    assert load_train_config(run_dir / "config.yaml").ema_decay == DECAY
