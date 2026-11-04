// Assign MediaPipe Tasks hands to the training data's left_hand / right_hand blocks. Pure.
//
// Training data came from MediaPipe Holistic, whose left_hand is the signer's own left hand
// (the hand at pose landmark 15, the left wrist). MediaPipe Tasks' HandLandmarker labels
// handedness assuming a mirrored (selfie) image, while this app feeds it the unmirrored camera
// frame. So the label "Left" is the signer's right hand -> right_hand block, and "Right" is the
// signer's left hand -> left_hand block. The "swap hands" setting reverses this. When both
// detected hands carry the same label, the one further left in the unmirrored image (smaller
// wrist x, the signer's right side when facing the camera) becomes right_hand.

import type { Point2 } from "./preprocess";

/** The category names in hand_landmarker.task's handedness.txt, index 0 and 1. */
export const HANDEDNESS_LABELS = ["Left", "Right"] as const;

export interface DetectedHand {
  landmarks: readonly Point2[];
  /** categoryName of the top handedness category ("Left" or "Right"). */
  label: string;
  score: number;
}

export interface AssignedHands {
  left: readonly Point2[] | null;
  right: readonly Point2[] | null;
  /** True when two hands had the same label and position decided. */
  resolvedByPosition: boolean;
}

/** The training-data block a MediaPipe Tasks label maps to, for an unmirrored frame. */
export function blockForLabel(label: string, swap: boolean): "left" | "right" | null {
  let block: "left" | "right" | null = label === "Left" ? "right" : label === "Right" ? "left" : null;
  if (block !== null && swap) block = block === "left" ? "right" : "left";
  return block;
}

export function assignHands(hands: readonly DetectedHand[], swap: boolean): AssignedHands {
  const result: AssignedHands = { left: null, right: null, resolvedByPosition: false };
  const usable = hands.filter((hand) => hand.landmarks.length > 0).slice(0, 2);
  const blocks = usable.map((hand) => blockForLabel(hand.label, swap));
  const first = usable[0];
  const second = usable[1];
  if (first && second && (blocks[0] === blocks[1] || blocks[0] === null || blocks[1] === null)) {
    const firstWristX = (first.landmarks[0] as Point2).x;
    const secondWristX = (second.landmarks[0] as Point2).x;
    const [imageLeft, imageRight] = firstWristX <= secondWristX ? [first, second] : [second, first];
    result.right = swap ? imageRight.landmarks : imageLeft.landmarks;
    result.left = swap ? imageLeft.landmarks : imageRight.landmarks;
    result.resolvedByPosition = true;
    return result;
  }
  usable.forEach((hand, i) => {
    const block = blocks[i];
    if (block === "left") result.left = hand.landmarks;
    else if (block === "right") result.right = hand.landmarks;
  });
  return result;
}

/**
 * Whether each assigned hand's wrist (hand landmark 0) is nearer the pose wrist of the same
 * side (15 left, 16 right) than the other one. A consistent assignment matches Holistic.
 */
export function wristConsistency(
  assigned: AssignedHands,
  pose: readonly Point2[] | null,
): { checked: number; consistent: number } {
  if (!pose || pose.length < 17) return { checked: 0, consistent: 0 };
  const leftWrist = pose[15] as Point2;
  const rightWrist = pose[16] as Point2;
  const distance = (a: Point2, b: Point2): number => Math.hypot(a.x - b.x, a.y - b.y);
  let checked = 0;
  let consistent = 0;
  for (const [hand, own, other] of [
    [assigned.left, leftWrist, rightWrist],
    [assigned.right, rightWrist, leftWrist],
  ] as const) {
    if (!hand || hand.length === 0) continue;
    checked += 1;
    if (distance(hand[0] as Point2, own) <= distance(hand[0] as Point2, other)) consistent += 1;
  }
  return { checked, consistent };
}
