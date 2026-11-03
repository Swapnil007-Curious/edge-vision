"""Landmark constants and selection for the asl-signs MediaPipe Holistic output.

The selected order is fixed: left_hand 0-20, right_hand 0-20, then pose landmarks in
``pose_indices`` order. With the default pose indices that gives L = 49. Face landmarks are
dropped.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.intp]

LANDMARK_COUNTS: dict[str, int] = {"face": 468, "left_hand": 21, "pose": 33, "right_hand": 21}
N_HAND = LANDMARK_COUNTS["left_hand"]
N_POSE = LANDMARK_COUNTS["pose"]
N_ALL_LANDMARKS = sum(LANDMARK_COUNTS.values())

# Nose, shoulders, elbows, wrists.
DEFAULT_POSE_INDICES: tuple[int, ...] = (0, 11, 12, 13, 14, 15, 16)
SHOULDERS: tuple[int, int] = (11, 12)
POSE_SWAP_PAIRS: tuple[tuple[int, int], ...] = (SHOULDERS, (13, 14), (15, 16))

# Slices into the selected landmark axis; POSE is open-ended because pose_indices is configurable.
LEFT_HAND = slice(0, N_HAND)
RIGHT_HAND = slice(N_HAND, 2 * N_HAND)
POSE = slice(2 * N_HAND, None)
FIRST_POSE_SLOT = 2 * N_HAND

# Flat landmark ids: TYPE_OFFSETS[TYPE_CODES[type]] + landmark_index, in the range 0-542.
TYPE_CODES: dict[str, int] = {name: code for code, name in enumerate(LANDMARK_COUNTS)}
TYPE_SIZES: IntArray = np.array(list(LANDMARK_COUNTS.values()), dtype=np.intp)
TYPE_OFFSETS: IntArray = np.concatenate(([0], np.cumsum(TYPE_SIZES[:-1]))).astype(np.intp)


def num_selected(pose_indices: Sequence[int] = DEFAULT_POSE_INDICES) -> int:
    return FIRST_POSE_SLOT + len(pose_indices)


def validate_pose_indices(pose_indices: Sequence[int]) -> None:
    """Reject selections the pipeline cannot mirror or address."""
    if len(set(pose_indices)) != len(pose_indices):
        raise ValueError(f"pose_indices must be unique, got {list(pose_indices)}")
    outside = [index for index in pose_indices if not 0 <= index < N_POSE]
    if outside:
        raise ValueError(f"pose_indices outside 0-{N_POSE - 1}: {outside}")
    selected = set(pose_indices)
    for first, second in POSE_SWAP_PAIRS:
        if (first in selected) != (second in selected):
            raise ValueError(f"pose_indices must contain both or neither of ({first}, {second})")


def pose_slot(pose_indices: Sequence[int], landmark: int) -> int:
    """Position of a pose landmark on the selected landmark axis."""
    return FIRST_POSE_SLOT + list(pose_indices).index(landmark)


def shoulder_slots(pose_indices: Sequence[int]) -> tuple[int, int] | None:
    """Slots of the two shoulders, or None if they are not selected."""
    if not set(SHOULDERS) <= set(pose_indices):
        return None
    return pose_slot(pose_indices, SHOULDERS[0]), pose_slot(pose_indices, SHOULDERS[1])


def selection_table(pose_indices: Sequence[int] = DEFAULT_POSE_INDICES) -> IntArray:
    """Map each flat landmark id to its selected slot, or -1 if the landmark is dropped."""
    table = np.full(N_ALL_LANDMARKS, -1, dtype=np.intp)
    hand_points = np.arange(N_HAND)
    table[TYPE_OFFSETS[TYPE_CODES["left_hand"]] + hand_points] = hand_points
    table[TYPE_OFFSETS[TYPE_CODES["right_hand"]] + hand_points] = N_HAND + hand_points
    pose_offset = TYPE_OFFSETS[TYPE_CODES["pose"]]
    for position, landmark in enumerate(pose_indices):
        table[pose_offset + landmark] = FIRST_POSE_SLOT + position
    return table


def mirror_permutation(pose_indices: Sequence[int] = DEFAULT_POSE_INDICES) -> IntArray:
    """Landmark-axis permutation swapping the hand blocks and the selected pose pairs."""
    validate_pose_indices(pose_indices)
    permutation = np.arange(num_selected(pose_indices))
    permutation[LEFT_HAND], permutation[RIGHT_HAND] = (
        permutation[RIGHT_HAND].copy(),
        permutation[LEFT_HAND].copy(),
    )
    for first, second in POSE_SWAP_PAIRS:
        if first in pose_indices:
            slot_a, slot_b = pose_slot(pose_indices, first), pose_slot(pose_indices, second)
            permutation[[slot_a, slot_b]] = permutation[[slot_b, slot_a]]
    return permutation
