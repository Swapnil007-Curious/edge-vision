"""Model input layout, shared by training, ONNX export and any client of the exported model.

One frame holds FRAME_DIM = 2 * L + 2 floats: the L selected landmarks flattened landmark-major
[l0x, l0y, l1x, l1y, ...], then presence_left and presence_right as 0.0 or 1.0. Landmarks follow
the preprocessing order: left hand 0-20, right hand 0-20, then the pose indices. A point is
missing when both its x and y are exactly 0.0.

The module constants describe the v1 selection (L = 49, FRAME_DIM = 100); InputLayout describes
any selection, such as v2's pose 0-16 (L = 59, FRAME_DIM = 120).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from edgevision.data.landmarks import DEFAULT_POSE_INDICES, num_selected, validate_pose_indices

NUM_FRAMES = 32
POSE_INDICES: tuple[int, ...] = DEFAULT_POSE_INDICES
NUM_LANDMARKS = num_selected(POSE_INDICES)
COORDS_PER_LANDMARK = 2
NUM_COORDS = NUM_LANDMARKS * COORDS_PER_LANDMARK
PRESENCE_LEFT = NUM_COORDS
PRESENCE_RIGHT = NUM_COORDS + 1
FRAME_DIM = NUM_COORDS + 2

INPUT_NAME = "landmarks"
OUTPUT_NAME = "logits"


@dataclass(frozen=True)
class InputLayout:
    """The per-frame layout for one pose selection; the default is the v1 layout."""

    pose_indices: tuple[int, ...] = POSE_INDICES
    num_frames: int = NUM_FRAMES

    def __post_init__(self) -> None:
        validate_pose_indices(self.pose_indices)

    @classmethod
    def for_pose_indices(cls, pose_indices: Sequence[int]) -> "InputLayout":
        return cls(pose_indices=tuple(int(index) for index in pose_indices))

    @property
    def num_landmarks(self) -> int:
        return num_selected(self.pose_indices)

    @property
    def num_coords(self) -> int:
        return self.num_landmarks * COORDS_PER_LANDMARK

    @property
    def frame_dim(self) -> int:
        return self.num_coords + 2

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready description stored with every checkpoint and checked on load."""
        return {
            "num_frames": self.num_frames,
            "num_landmarks": self.num_landmarks,
            "coords_per_landmark": COORDS_PER_LANDMARK,
            "frame_dim": self.frame_dim,
            "coordinate_order": "landmark_major_xy",
            "landmark_order": ["left_hand_0-20", "right_hand_0-20", "pose_indices"],
            "pose_indices": list(self.pose_indices),
            "presence_left_index": self.num_coords,
            "presence_right_index": self.num_coords + 1,
            "missing_point": "x == 0.0 and y == 0.0",
            "input_name": INPUT_NAME,
            "output_name": OUTPUT_NAME,
        }


V1_LAYOUT = InputLayout()


def layout_dict(pose_indices: Sequence[int] = POSE_INDICES) -> dict[str, Any]:
    """JSON-ready description of the layout for ``pose_indices`` (default: v1)."""
    return InputLayout.for_pose_indices(pose_indices).to_dict()


def layout_from_dict(payload: dict[str, Any]) -> InputLayout:
    """The layout a stored description names; refuses one that is not self-consistent."""
    layout = InputLayout.for_pose_indices(payload["pose_indices"])
    if payload != layout.to_dict():
        raise ValueError("input layout description differs from edgevision.layout")
    return layout


def to_model_input(
    features: npt.NDArray[np.float16],
    presence: npt.NDArray[np.uint8],
    num_landmarks: int = NUM_LANDMARKS,
) -> npt.NDArray[np.float32]:
    """Pack (n, 32, L, 2) coordinates and (n, 32, 2) presence into (n, 32, 2L + 2) float32."""
    n = features.shape[0]
    expected = (n, NUM_FRAMES, num_landmarks, COORDS_PER_LANDMARK)
    if features.shape != expected or presence.shape != (n, NUM_FRAMES, 2):
        raise ValueError(
            f"expected coordinates {expected} and presence {(n, NUM_FRAMES, 2)}, "
            f"got {features.shape} and {presence.shape}"
        )
    num_coords = num_landmarks * COORDS_PER_LANDMARK
    packed = np.empty((n, NUM_FRAMES, num_coords + 2), dtype=np.float32)
    packed[..., :num_coords] = features.reshape(n, NUM_FRAMES, num_coords)
    packed[..., num_coords:] = presence
    return packed
