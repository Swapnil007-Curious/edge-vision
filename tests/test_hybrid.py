"""The CNN-transformer hybrid: size, shapes, capacity and the late-dropout schedule."""

import numpy as np
import pytest
import torch
from torch import nn

from edgevision.evaluation.metrics import top_k_accuracy
from edgevision.layout import to_model_input
from edgevision.models.config import HybridConfig
from edgevision.models.features import compute_feature_stats
from edgevision.models.hybrid import HybridClassifier, LateDropout
from edgevision.models.registry import (
    HYBRID_MAX_PARAMS,
    HYBRID_MIN_PARAMS,
    build_model,
    count_parameters,
)
from edgevision.train.loop import EpochContext, predict_logits, seed_everything, train_one_epoch
from v2_support import V2_LAYOUT, build_v2_processed_dir

HYBRID_LR = 5e-4


def hybrid(num_classes=5):
    return build_model("hybrid", num_classes, feature_set="v2", layout=V2_LAYOUT)


def test_parameter_count_is_in_range_for_250_classes():
    count = count_parameters(hybrid(250))
    print(f"hybrid: {count:,} parameters")
    assert HYBRID_MIN_PARAMS <= count <= HYBRID_MAX_PARAMS
    assert (HYBRID_MIN_PARAMS, HYBRID_MAX_PARAMS) == (1_500_000, 6_000_000)


def test_structure_follows_the_specification():
    model = hybrid(250)
    assert isinstance(model, HybridClassifier)
    assert model.proj.in_features == 588 and model.proj.out_features == 256
    assert len(model.stages) == 3
    stage = model.stages[0]
    assert len(stage.convs) == 2
    conv = stage.convs[0]
    assert (conv.depthwise.kernel_size, conv.depthwise.groups) == ((17,), 256)
    assert conv.depthwise.padding == (8,)
    assert (conv.expand.out_channels, conv.project.out_channels) == (512, 256)
    assert isinstance(conv.activation, nn.SiLU) and conv.dropout.p == 0.2
    attention = stage.transformer.attention
    assert attention.num_heads == 4 and not isinstance(attention, nn.MultiheadAttention)
    assert stage.transformer.feed_forward[0].out_features == 512
    assert isinstance(stage.transformer.feed_forward[1], nn.GELU)
    assert model.head_dropout.p == 0.4 and model.head.in_features == 256


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("batch", [1, 4])
def test_forward_returns_logits_per_sample(batch, training):
    model = hybrid().train(training)
    generator = torch.Generator().manual_seed(batch)
    x = torch.randn(batch, 32, V2_LAYOUT.frame_dim, generator=generator)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (batch, 5)
    assert torch.isfinite(logits).all()


def test_late_dropout_schedule():
    model = hybrid()
    epochs = 10
    active = []
    for epoch in range(1, epochs + 1):
        model.set_training_progress(epoch, epochs)
        active.append(model.head_dropout.active)
    assert active == [False] * 5 + [True] * 5  # Off for the first 50% of epochs.
    model.set_training_progress(76, 150)
    assert model.head_dropout.active
    model.set_training_progress(75, 150)
    assert not model.head_dropout.active


def test_late_dropout_is_identity_until_active_and_in_eval():
    dropout = LateDropout(0.4).train()
    x = torch.ones(64, 256)
    assert dropout(x) is x
    dropout.active = True
    torch.manual_seed(0)
    dropped = dropout(x)
    assert (dropped == 0.0).float().mean().item() == pytest.approx(0.4, abs=0.05)
    assert dropout.eval()(x) is x


def test_late_dropout_start_is_configurable():
    model = HybridClassifier(5, HybridConfig(late_dropout_start=0.0), hybrid().features)
    model.set_training_progress(1, 10)
    assert model.head_dropout.active


@pytest.fixture(scope="module")
def overfit_data(tmp_path_factory):
    data = build_v2_processed_dir(tmp_path_factory.mktemp("overfit"), per_cell=2)
    features = data.features
    x = to_model_input(features["X"][:64], features["presence"][:64], V2_LAYOUT.num_landmarks)
    return torch.from_numpy(x), torch.from_numpy(features["y"][:64].astype(np.int64))


def test_hybrid_overfits_64_samples_within_400_steps(overfit_data):
    x, y = overfit_data
    assert len(y) == 64
    seed_everything(0)
    model = hybrid()
    compute_feature_stats(model.features, x)
    optimizer = torch.optim.AdamW(model.parameters(), lr=HYBRID_LR, weight_decay=0.05)
    ctx = EpochContext(
        optimizer=optimizer,
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
        criterion=nn.CrossEntropyLoss(label_smoothing=0.1),
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        batch_size=64,
        grad_clip=1.0,
        use_amp=False,
        augment=None,  # Augmentation off.
        order_generator=torch.Generator().manual_seed(0),
        augment_generator=torch.Generator().manual_seed(0),
    )
    accuracy, step = 0.0, 0
    while step < 400 and accuracy < 0.95:
        for _ in range(10):
            train_one_epoch(model, x, y, ctx)  # One batch of 64: one step per call.
        step += 10
        accuracy = top_k_accuracy(predict_logits(model, x).numpy(), y.numpy(), 1)
    print(f"hybrid: train accuracy {accuracy:.3f} after {step} steps")
    assert accuracy >= 0.95
