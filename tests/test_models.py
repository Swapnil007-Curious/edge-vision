import numpy as np
import pytest
import torch
from torch import nn

from edgevision.evaluation.metrics import top_k_accuracy
from edgevision.layout import to_model_input
from edgevision.models.config import MODEL_NAMES, CNNConfig, TransformerConfig
from edgevision.models.features import FeatureBlock, compute_feature_stats
from edgevision.models.registry import MAX_PARAMS, MIN_PARAMS, build_model, count_parameters
from edgevision.train.loop import EpochContext, predict_logits, seed_everything, train_one_epoch
from training_support import build_processed_dir

LEARNING_RATES = {"gru": 1e-3, "cnn": 1e-3, "transformer": 5e-4}


def random_input(batch, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 32, 100, generator=generator)
    x[:, :, :42] *= (torch.rand(batch, 32, 42, generator=generator) > 0.3).float()
    x[..., 98:] = (torch.rand(batch, 32, 2, generator=generator) > 0.5).float()
    return x


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_parameter_count_is_in_range_for_250_classes(name):
    count = count_parameters(build_model(name, 250))
    print(f"{name}: {count:,} parameters")
    assert MIN_PARAMS <= count <= MAX_PARAMS


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("batch", [1, 4])
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_returns_logits_per_sample(name, batch, training):
    model = build_model(name, 5).train(training)
    with torch.no_grad():
        logits = model(random_input(batch))
    assert logits.shape == (batch, 5)
    assert torch.isfinite(logits).all()


def test_build_model_rejects_a_config_for_another_model():
    with pytest.raises(ValueError, match="does not match"):
        build_model("gru", 5, CNNConfig())


def test_config_validation():
    with pytest.raises(ValueError, match="divisible"):
        TransformerConfig(d_model=190, num_heads=4)
    with pytest.raises(ValueError, match="odd"):
        CNNConfig(kernel_size=4)


def test_velocity_is_zero_at_the_start_and_at_missing_endpoints():
    x = torch.zeros(1, 32, 100)
    x[0, :, 0] = 0.1 * torch.arange(32.0)  # Point 0 moves along x, present throughout.
    x[0, :, 1] = 1.0
    x[0, :, 2:4] = 0.5  # Point 1 is present except at frame 5.
    x[0, 5, 2:4] = 0.0
    x[0, 7, 2] = 0.9

    features = FeatureBlock().raw_features(x)

    assert features.shape == (1, 32, 198)
    assert torch.equal(features[..., :100], x)
    velocity = features[0, :, 100:]
    assert (velocity[0] == 0.0).all()
    torch.testing.assert_close(velocity[1:, 0], torch.full((31,), 0.1), rtol=0.0, atol=1e-6)
    assert (velocity[:, 1] == 0.0).all()
    assert (velocity[5:7, 2:4] == 0.0).all()  # Frames 5 and 6 each have a missing endpoint.
    assert velocity[7, 2] == pytest.approx(0.4)
    assert velocity[8, 2] == pytest.approx(-0.4)
    assert (velocity[:, 4:] == 0.0).all()  # Never-present points have no velocity.


def test_standardisation_uses_the_buffers_and_floors_std():
    block = FeatureBlock()
    x = random_input(2)
    raw = block.raw_features(x)
    block.mean.fill_(0.5)
    block.std.fill_(2.0)
    torch.testing.assert_close(block(x), (raw - 0.5) / 2.0)
    block.std.zero_()
    torch.testing.assert_close(block(x), (raw - 0.5) / 1e-6)


def test_feature_stats_match_a_direct_computation(monkeypatch):
    monkeypatch.setattr("edgevision.models.features.STATS_CHUNK", 7)  # Several chunks.
    x = random_input(30)
    block = FeatureBlock()
    compute_feature_stats(block, x.numpy())
    raw = block.raw_features(x).reshape(-1, 198).double()
    torch.testing.assert_close(block.mean, raw.mean(dim=0).float())
    expected_std = raw.std(dim=0, unbiased=False).clamp_min(1e-6).float()
    torch.testing.assert_close(block.std, expected_std)
    assert block.std.min() >= 1e-6


def test_feature_stats_refuse_an_empty_split():
    with pytest.raises(ValueError, match="empty"):
        compute_feature_stats(FeatureBlock(), torch.zeros(0, 32, 100))


def test_feature_stats_are_saved_in_the_state_dict():
    model = build_model("cnn", 5)
    assert {"features.mean", "features.std"} <= set(model.state_dict())


@pytest.fixture(scope="module")
def overfit_data(tmp_path_factory):
    data = build_processed_dir(tmp_path_factory.mktemp("overfit"), per_cell=2)
    features = data.features
    x = torch.from_numpy(to_model_input(features["X"][:64], features["presence"][:64]))
    y = torch.from_numpy(features["y"][:64].astype(np.int64))
    return x, y


def epoch_context(model, name, seed=0):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATES[name], weight_decay=0.05)
    return EpochContext(
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
        criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        batch_size=64,
        grad_clip=1.0,
        use_amp=False,
        augment=None,
        order_generator=torch.Generator().manual_seed(seed),
        augment_generator=torch.Generator().manual_seed(seed),
    )


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_overfits_64_samples_within_300_steps(name, overfit_data):
    x, y = overfit_data
    assert len(y) == 64
    seed_everything(0)
    model = build_model(name, 5)
    compute_feature_stats(model.features, x)
    ctx = epoch_context(model, name)
    accuracy, step = 0.0, 0
    while step < 300 and accuracy < 0.95:
        for _ in range(10):
            train_one_epoch(model, x, y, ctx)  # One batch of 64: one step per call.
        step += 10
        accuracy = top_k_accuracy(predict_logits(model, x).numpy(), y.numpy(), 1)
    print(f"{name}: train accuracy {accuracy:.3f} after {step} steps")
    assert accuracy >= 0.95


def test_training_is_repeatable_on_cpu_for_a_seed(overfit_data):
    x, y = overfit_data

    def train_twice():
        seed_everything(5)
        model = build_model("cnn", 5)
        compute_feature_stats(model.features, x)
        ctx = epoch_context(model, "cnn", seed=5)
        ctx.batch_size = 16
        for _ in range(2):
            train_one_epoch(model, x, y, ctx)
        return model.state_dict()

    first, second = train_twice(), train_twice()
    for key, value in first.items():
        assert torch.equal(value, second[key]), key
