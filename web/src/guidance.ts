// Camera guidance messages from one frame's landmarks and brightness. Pure.

import type { Point2 } from "./preprocess";

export type GuidanceCode =
  | "no_hands"
  | "hand_out_of_frame"
  | "low_light"
  | "too_close"
  | "too_far"
  | "shoulders_hidden";

export interface GuidanceMessage {
  code: GuidanceCode;
  text: string;
}

export interface GuidanceSettings {
  /** Mean luma (0-255) of a downscaled frame below which the light is too low. */
  lowLightThreshold: number;
  /** Shoulder distance range in normalised image units. */
  minShoulderDistance: number;
  maxShoulderDistance: number;
  /** Consecutive frames without a hand before "no hands" is shown. */
  noHandFrames: number;
}

export const DEFAULT_GUIDANCE_SETTINGS: GuidanceSettings = {
  lowLightThreshold: 50,
  minShoulderDistance: 0.15,
  maxShoulderDistance: 0.7,
  noHandFrames: 15,
};

export interface GuidanceInput {
  hands: readonly (readonly Point2[])[];
  /** Pose shoulders 11 and 12, or null without a pose. */
  shoulders: readonly [Point2, Point2] | null;
  /** Latest mean luma, or null if not measured yet. */
  luminance: number | null;
  framesWithoutHands: number;
}

export const MESSAGES: Record<GuidanceCode, string> = {
  no_hands: "No hands in view.",
  hand_out_of_frame: "A hand is partly out of frame.",
  low_light: "Low light: add light in front of you.",
  too_close: "Too close: move back from the camera.",
  too_far: "Too far: move closer to the camera.",
  shoulders_hidden: "Shoulders not visible: keep your upper body in view.",
};

/** Mean BT.709 luma of RGBA pixels. */
export function meanLuminance(rgba: ArrayLike<number>): number {
  let sum = 0;
  const pixels = Math.floor(rgba.length / 4);
  for (let i = 0; i < pixels; i += 1) {
    sum += 0.2126 * (rgba[i * 4] as number) + 0.7152 * (rgba[i * 4 + 1] as number) + 0.0722 * (rgba[i * 4 + 2] as number);
  }
  return pixels ? sum / pixels : Number.NaN;
}

export function shoulderDistance(shoulders: readonly [Point2, Point2]): number {
  const [a, b] = shoulders;
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function outside(point: Point2): boolean {
  return point.x < 0 || point.x > 1 || point.y < 0 || point.y > 1;
}

export function guidance(input: GuidanceInput, settings: GuidanceSettings): GuidanceMessage[] {
  const codes: GuidanceCode[] = [];
  if (input.hands.length === 0 && input.framesWithoutHands >= settings.noHandFrames) codes.push("no_hands");
  if (input.hands.some((hand) => hand.some(outside))) codes.push("hand_out_of_frame");
  if (input.luminance !== null && input.luminance < settings.lowLightThreshold) codes.push("low_light");
  if (input.shoulders === null) {
    codes.push("shoulders_hidden");
  } else {
    const distance = shoulderDistance(input.shoulders);
    if (distance > settings.maxShoulderDistance) codes.push("too_close");
    else if (distance < settings.minShoulderDistance) codes.push("too_far");
  }
  return codes.map((code) => ({ code, text: MESSAGES[code] }));
}
