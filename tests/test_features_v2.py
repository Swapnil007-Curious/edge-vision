"""FeatureBlockV2: layout, the missing-point rule, train-only statistics and ONNX parity."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from torch import nn

from edgevision.export.onnx_export import export_onnx
from edgevision.layout import INPUT_NAME, OUTPUT_NAME, V1_LAYOUT
from edgevision.models.features import compute_feature_stats
from edgevision.models.features_v2 import HAND_PAIRS, FeatureBlockV2, v2_feature_dim
from edgevision.models.registry import build_model, load_checkpoint
from edgevision.train.config import load_train_config
from edgevision.train.data import load_split_arrays
from edgevision.train.loop import run_training
from training_support import EXPERIMENTS
from v2_support import V2_LAYOUT, build_v2_processed_dir

COORDS = V2_LAYOUT.num_coords  # 118
VELOCITY, ACCELERATION = slice(COORDS, 2 * COORDS), slice(2 * COORDS, 3 * COORDS)
HAND = slice(3 * COORDS, 3 * COORDS + 210)
BODY = slice(3 * COORDS + 210, 3 * COORDS + 234)
WRIST, NOSE, MOUTH_LEFT, MOUTH_RIGHT, LEFT_SHOULDER = 21, 42, 51, 52, 53
# Standard-domain ONNX ops the feature block may use; onnxruntime-web's wasm backend runs all.
FEATURE_OPS = {
    "Add",
    "Cast",
    "Concat",
    "Constant",
    "ConstantOfShape",
    "Div",
    "Equal",
    "Gather",
    "Max",
    "Mul",
    "Not",
    "Or",
    "ReduceSum",
    "Reshape",
    "Shape",
    "Slice",
    "Sqrt",
    "Sub",
    "Unsqueeze",
    "Clip",
    "Expand",
    "Where",
}


def v2_input(batch, seed, missing=0.25):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, 32, V2_LAYOUT.frame_dim, generator=generator)
    points = x[..., :COORDS].view(batch, 32, V2_LAYOUT.num_landmarks, 2)
    points[torch.rand(batch, 32, V2_LAYOUT.num_landmarks, generator=generator) < missing] = 0.0
    x[..., COORDS:] = (torch.rand(batch, 32, 2, generator=generator) < 0.6).float()
    return x


def put(x, slot, frames, values):
    x[0, frames, 2 * slot : 2 * slot + 2] = torch.as_tensor(values, dtype=x.dtype)


def test_layout_and_feature_dimension():
    block = FeatureBlockV2(V2_LAYOUT)
    assert len(HAND_PAIRS) == 210
    assert block.output_dim == v2_feature_dim(V2_LAYOUT) == 6 * 59 + 234 == 588
    features = block.raw_features(v2_input(3, 0))
    assert features.shape == (3, 32, 588)
    assert set(block.state_dict()) == {"mean", "std"}  # Index constants are not saved.


def test_v2_needs_face_and_shoulder_landmarks():
    with pytest.raises(ValueError, match="needs pose landmarks"):
        FeatureBlockV2(V1_LAYOUT)


def crafted():
    """Right wrist present throughout: x = 0.05 t^2, y = 1. Point 22 (thumb) present except
    frame 5. Fingertip 4 (slot 25) never present. Nose, mouth and shoulders fixed, except the
    left mouth corner at frame 7."""
    x = torch.zeros(1, 32, V2_LAYOUT.frame_dim, dtype=torch.float64)
    t = torch.arange(32, dtype=torch.float64)
    x[0, :, 2 * WRIST] = 0.05 * t**2
    x[0, :, 2 * WRIST + 1] = 1.0
    put(x, 22, slice(None), (0.5, 0.25))
    put(x, 22, 5, (0.0, 0.0))
    put(x, NOSE, slice(None), (0.0, -1.0))
    put(x, MOUTH_LEFT, slice(None), (0.1, -0.8))
    put(x, MOUTH_RIGHT, slice(None), (-0.1, -0.8))
    put(x, MOUTH_LEFT, 7, (0.0, 0.0))
    put(x, LEFT_SHOULDER, slice(None), (0.5, 0.0))
    return x


def raw(x):
    return FeatureBlockV2(V2_LAYOUT).double().raw_features(x)[0]


def test_coordinates_pass_through_and_velocity_follows_the_missing_rule():
    x = crafted()
    features = raw(x)
    assert torch.equal(features[:, :COORDS], x[0, :, :COORDS])
    velocity = features[:, VELOCITY]
    assert (velocity[0] == 0.0).all()
    t = torch.arange(1, 32, dtype=torch.float64)
    torch.testing.assert_close(velocity[1:, 2 * WRIST], 0.05 * (2 * t - 1))
    assert (velocity[5:7, 44:46] == 0.0).all()  # Point 22: frames 5 and 6 touch frame 5.
    assert (velocity[[4, 7], 44] == 0.0).all()  # Present on both sides, but not moving.
    assert (velocity[:, 50:52] == 0.0).all()  # Slot 25 never present.


def test_acceleration_is_the_second_difference_with_three_present_frames():
    features = raw(crafted())
    acceleration = features[:, ACCELERATION]
    assert (acceleration[:2] == 0.0).all()
    torch.testing.assert_close(
        acceleration[2:, 2 * WRIST], torch.full((30,), 0.1, dtype=torch.float64)
    )
    assert (acceleration[2:, 2 * WRIST + 1] == 0.0).all()  # y is constant.
    assert (acceleration[:, 44:46] == 0.0).all()  # Point 22 is still.
    x = crafted()
    put(x, 22, slice(None), (0.0, 0.0))
    x[0, :, 44] = torch.arange(32, dtype=torch.float64) ** 2  # x = t^2 ...
    x[0, :, 45] = 0.5
    x[0, 5, 44:46] = 0.0  # ... missing at frame 5.
    acceleration = raw(x)[:, ACCELERATION]
    assert (acceleration[5:8, 44] == 0.0).all()  # Every window touching frame 5.
    assert acceleration[4, 44] == pytest.approx(2.0)
    assert acceleration[8, 44] == pytest.approx(2.0)


def test_hand_shape_distances_are_zero_when_either_point_is_missing():
    features = raw(crafted())
    hand = features[:, HAND]
    pair_01 = HAND_PAIRS.index((0, 1))
    t = torch.arange(32, dtype=torch.float64)
    expected = torch.hypot(0.05 * t**2 - 0.5, torch.full_like(t, 0.75))
    expected[5] = 0.0
    torch.testing.assert_close(hand[:, pair_01], expected)
    involving_4 = [index for index, pair in enumerate(HAND_PAIRS) if 4 in pair]
    assert (hand[:, involving_4] == 0.0).all()
    unseen = [index for index, (a, b) in enumerate(HAND_PAIRS) if a > 1 and b > 1]
    assert (hand[:, unseen] == 0.0).all()  # Only points 0 and 1 are ever present.


def test_hand_to_body_distances_use_the_mouth_centre_and_the_missing_rule():
    features = raw(crafted())
    body = features[:, BODY].reshape(32, 6, 4)  # Wrist, tips 4-20 x nose, mouth, shoulders.
    t = torch.arange(32, dtype=torch.float64)
    wrist_x = 0.05 * t**2
    torch.testing.assert_close(body[:, 0, 0], torch.hypot(wrist_x, torch.full_like(t, 2.0)))
    to_mouth = torch.hypot(wrist_x, torch.full_like(t, 1.8))
    to_mouth[7] = 0.0  # A mouth corner is missing.
    torch.testing.assert_close(body[:, 0, 1], to_mouth)
    torch.testing.assert_close(body[:, 0, 2], torch.hypot(wrist_x - 0.5, torch.full_like(t, 1.0)))
    assert (body[:, 0, 3] == 0.0).all()  # The right shoulder is never present.
    assert (body[:, 1:] == 0.0).all()  # No fingertip is present.


def test_feature_stats_match_a_direct_computation():
    x = v2_input(20, 3)
    block = FeatureBlockV2(V2_LAYOUT)
    compute_feature_stats(block, x.numpy())
    values = block.raw_features(x).reshape(-1, 588).double()
    torch.testing.assert_close(block.mean, values.mean(dim=0).float())
    std = values.std(dim=0, unbiased=False).clamp_min(1e-6).float()
    torch.testing.assert_close(block.std, std)


def test_feature_stats_come_from_the_train_split_only(tmp_path):
    data = build_v2_processed_dir(tmp_path / "processed", extreme_eval=True)
    cfg = load_train_config(
        EXPERIMENTS / "train_b_cnn_v2feat_v1recipe.yaml",
        processed_dir=data.root,
        output_root=tmp_path / "runs",
        epochs=1,
    )
    run_training(cfg, seed=0, device="cpu", log=lambda _: None)
    run_dir = tmp_path / "runs" / "b_cnn_v2feat_v1recipe" / "seed_0"
    stored = load_checkpoint(run_dir / "best.pt").model.features
    assert isinstance(stored, FeatureBlockV2)
    arrays = load_split_arrays(
        data.root, run_dir / "splits_signer_independent.json", ["train", "val"]
    )
    expected = FeatureBlockV2(V2_LAYOUT)
    compute_feature_stats(expected, arrays["train"].X)
    torch.testing.assert_close(stored.mean, expected.mean)
    torch.testing.assert_close(stored.std, expected.std)
    leaked = FeatureBlockV2(V2_LAYOUT)
    compute_feature_stats(leaked, np.concatenate([arrays["train"].X, arrays["val"].X]))
    assert (leaked.mean - expected.mean).abs().max() > 100.0


def randomised(module):
    """Non-trivial feature statistics and BatchNorm running statistics."""
    generator = torch.Generator().manual_seed(1)
    features = module.features if hasattr(module, "features") else module
    with torch.no_grad():
        features.mean.copy_(0.5 * torch.randn(588, generator=generator))
        features.std.copy_(0.2 + 1.5 * torch.rand(588, generator=generator))
        for layer in module.modules():
            if isinstance(layer, nn.BatchNorm1d):
                layer.running_mean.copy_(0.1 * torch.randn(layer.num_features, generator=generator))
                layer.running_var.copy_(0.5 + torch.rand(layer.num_features, generator=generator))
    return module.eval()


def build(name):
    torch.manual_seed(0)
    if name == "features":
        return randomised(FeatureBlockV2(V2_LAYOUT))
    return randomised(build_model(name, 250, feature_set="v2", layout=V2_LAYOUT))


@pytest.fixture(scope="module", params=["features", "cnn", "hybrid"])
def exported(request, tmp_path_factory):
    module = build(request.param)
    path = export_onnx(module, tmp_path_factory.mktemp("onnx") / f"{request.param}.onnx")
    return request.param, module, path


@pytest.mark.parametrize("batch", [1, 7])
def test_onnxruntime_matches_torch(exported, batch):
    name, module, path = exported
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = v2_input(batch, seed=batch)
    with torch.no_grad():
        expected = module(x).numpy()
    (actual,) = session.run([OUTPUT_NAME], {INPUT_NAME: x.numpy()})
    assert actual.shape == expected.shape
    difference = float(np.abs(actual - expected).max())
    print(f"{name} batch {batch}: max abs diff {difference:.2e}")
    assert difference < 1e-4


def test_export_uses_only_standard_opset_17_ops(exported):
    name, _, path = exported
    proto = onnx.load(str(path))
    assert [(item.domain, item.version) for item in proto.opset_import] == [("", 17)]
    assert {node.domain for node in proto.graph.node} <= {""}  # No ATen or custom ops.
    input_dims = proto.graph.input[0].type.tensor_type.shape.dim
    assert [dim.dim_value for dim in input_dims[1:]] == [32, 120]
    if name == "features":
        ops = {node.op_type for node in proto.graph.node}
        print(f"feature block ops: {sorted(ops)}")
        assert ops <= FEATURE_OPS
