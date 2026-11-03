"""Model input layout, shared by training, ONNX export and any client of the exported model.

One frame holds FRAME_DIM = 100 floats: the 49 selected landmarks flattened landmark-major
[l0x, l0y, l1x, l1y, ..., l48x, l48y], then presence_left and presence_right as 0.0 or 1.0.
Landmarks follow the preprocessing order: left hand 0-20, right hand 0-20, then POSE_INDICES.
A point is missing when both its x and y are exactly 0.0.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from edgevision.data.landmarks import DEFAULT_POSE_INDICES, num_selected

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


def layout_dict() -> dict[str, Any]:
    """JSON-ready description stored with every checkpoint and checked on load."""
    return {
        "num_frames": NUM_FRAMES,
        "num_landmarks": NUM_LANDMARKS,
        "coords_per_landmark": COORDS_PER_LANDMARK,
        "frame_dim": FRAME_DIM,
        "coordinate_order": "landmark_major_xy",
        "landmark_order": ["left_hand_0-20", "right_hand_0-20", "pose_indices"],
        "pose_indices": list(POSE_INDICES),
        "presence_left_index": PRESENCE_LEFT,
        "presence_right_index": PRESENCE_RIGHT,
        "missing_point": "x == 0.0 and y == 0.0",
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
    }


def to_model_input(
    features: npt.NDArray[np.float16], presence: npt.NDArray[np.uint8]
) -> npt.NDArray[np.float32]:
    """Pack (n, 32, 49, 2) coordinates and (n, 32, 2) presence into (n, 32, 100) float32."""
    n = features.shape[0]
    expected = (n, NUM_FRAMES, NUM_LANDMARKS, COORDS_PER_LANDMARK)
    if features.shape != expected or presence.shape != (n, NUM_FRAMES, 2):
        raise ValueError(
            f"expected coordinates {expected} and presence {(n, NUM_FRAMES, 2)}, "
            f"got {features.shape} and {presence.shape}"
        )
    packed = np.empty((n, NUM_FRAMES, FRAME_DIM), dtype=np.float32)
    packed[..., :NUM_COORDS] = features.reshape(n, NUM_FRAMES, NUM_COORDS)
    packed[..., NUM_COORDS:] = presence
    return packed
