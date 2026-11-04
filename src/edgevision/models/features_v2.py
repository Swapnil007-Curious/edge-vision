"""v2 feature block: coordinates, velocity, acceleration, dominant-hand shape and hand-to-face
distances, standardised, inside the model so ONNX export carries them.

Input (B, T, 2L + 2) is laid out like v1 (edgevision.layout); presence channels are not
features. A point is missing when x == y == 0.0, and any derived value that involves a missing
point is 0.0. Only Gather, arithmetic, Sqrt, comparisons, Cast and Concat are used, so the block
exports to ONNX opset 17 and runs in onnxruntime-web's wasm backend.
"""

import itertools

import torch

from edgevision.data.landmarks import N_HAND, RIGHT_HAND, pose_slot
from edgevision.layout import InputLayout
from edgevision.models.features import StandardisedFeatures

WRIST = 0
FINGERTIPS = (4, 8, 12, 16, 20)
NOSE, MOUTH_LEFT, MOUTH_RIGHT, LEFT_SHOULDER, RIGHT_SHOULDER = 0, 9, 10, 11, 12
REQUIRED_POSE = (NOSE, MOUTH_LEFT, MOUTH_RIGHT, LEFT_SHOULDER, RIGHT_SHOULDER)
HAND_PAIRS = tuple(itertools.combinations(range(N_HAND), 2))  # 210 pairs.
BODY_POINTS = (WRIST, *FINGERTIPS)  # 6 dominant-hand points ...
BODY_TARGETS = ("nose", "mouth_centre", "left_shoulder", "right_shoulder")  # ... to 4 targets.


def v2_feature_dim(layout: InputLayout) -> int:
    """2L coordinates + 2L velocity + 2L acceleration + 210 hand distances + 24 body distances."""
    return 3 * layout.num_coords + len(HAND_PAIRS) + len(BODY_POINTS) * len(BODY_TARGETS)


class FeatureBlockV2(StandardisedFeatures):
    """(B, T, 2L + 2) -> (B, T, 6L + 234); 588 features for the v2 selection (L = 59)."""

    feature_set = "v2"
    pair_first: torch.Tensor
    pair_second: torch.Tensor
    body_points: torch.Tensor
    target_points: torch.Tensor

    def __init__(self, layout: InputLayout) -> None:
        missing = [index for index in REQUIRED_POSE if index not in layout.pose_indices]
        if missing:
            raise ValueError(
                f"feature set v2 needs pose landmarks {list(REQUIRED_POSE)}; the layout lacks "
                f"{missing} (preprocess with experiments/preprocess_v2.yaml)"
            )
        super().__init__(layout, v2_feature_dim(layout))
        first_slot = RIGHT_HAND.start
        first, second = zip(*HAND_PAIRS, strict=True)
        # Index constants, not state: non-persistent buffers move with the module but are not
        # saved, so checkpoints hold only the standardisation statistics.
        self.register_buffer("pair_first", torch.tensor(first) + first_slot, persistent=False)
        self.register_buffer("pair_second", torch.tensor(second) + first_slot, persistent=False)
        body = [first_slot + point for point in BODY_POINTS]
        self.register_buffer("body_points", torch.tensor(body), persistent=False)
        slots = [pose_slot(layout.pose_indices, index) for index in REQUIRED_POSE]
        self.register_buffer("target_points", torch.tensor(slots), persistent=False)

    def raw_features(self, x: torch.Tensor) -> torch.Tensor:
        layout = self.layout
        frames, coords = layout.num_frames, layout.num_coords
        points = x[..., :coords].reshape(-1, frames, layout.num_landmarks, 2)
        present = ((points[..., 0] != 0.0) | (points[..., 1] != 0.0)).to(points.dtype)

        step = (points[:, 1:] - points[:, :-1]) * (present[:, 1:] * present[:, :-1]).unsqueeze(-1)
        velocity = torch.cat([torch.zeros_like(step[:, :1]), step], dim=1)

        second = points[:, 2:] - 2.0 * points[:, 1:-1] + points[:, :-2]
        all_three = present[:, 2:] * present[:, 1:-1] * present[:, :-2]
        second = second * all_three.unsqueeze(-1)
        acceleration = torch.cat([torch.zeros_like(second[:, :2]), second], dim=1)

        hand_shape = _distances(
            points.index_select(2, self.pair_first),
            points.index_select(2, self.pair_second),
            present.index_select(2, self.pair_first) * present.index_select(2, self.pair_second),
        )

        targets = points.index_select(2, self.target_points)  # nose, mouth L/R, shoulders L/R.
        target_present = present.index_select(2, self.target_points)
        mouth = 0.5 * (targets[:, :, 1:2] + targets[:, :, 2:3])
        mouth_present = target_present[:, :, 1:2] * target_present[:, :, 2:3]
        anchors = torch.cat([targets[:, :, :1], mouth, targets[:, :, 3:]], dim=2)
        anchor_present = torch.cat(
            [target_present[:, :, :1], mouth_present, target_present[:, :, 3:]], dim=2
        )
        hand_points = points.index_select(2, self.body_points)
        hand_present = present.index_select(2, self.body_points)
        to_body = _distances(
            hand_points.unsqueeze(3),
            anchors.unsqueeze(2),
            hand_present.unsqueeze(3) * anchor_present.unsqueeze(2),
        )
        return torch.cat(
            [
                x[..., :coords],
                velocity.reshape(-1, frames, coords),
                acceleration.reshape(-1, frames, coords),
                hand_shape,
                to_body.reshape(-1, frames, len(BODY_POINTS) * len(BODY_TARGETS)),
            ],
            dim=-1,
        )


def _distances(first: torch.Tensor, second: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Euclidean distance over the last axis (x, y), 0.0 where ``valid`` is 0.

    The input carries no gradient (the feature block runs on data), so sqrt at 0 is harmless.
    """
    difference = first - second
    return torch.sqrt((difference * difference).sum(dim=-1)) * valid
