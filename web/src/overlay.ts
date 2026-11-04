// Landmark overlay on the preview canvas, in unmirrored frame coordinates; the canvas itself is
// mirrored with CSS so the preview behaves like a mirror.

import type { Point2 } from "./preprocess";

export const COLORS = {
  left: "#00e5ff",
  right: "#ff9e1b",
  pose: "#ffdd00",
} as const;

const POSE_LINES: readonly (readonly [number, number])[] = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
];

export interface OverlayInput {
  left: readonly Point2[] | null;
  right: readonly Point2[] | null;
  pose: readonly Point2[] | null;
  connections: readonly { start: number; end: number }[];
}

function line(context: CanvasRenderingContext2D, a: Point2, b: Point2, w: number, h: number): void {
  context.beginPath();
  context.moveTo(a.x * w, a.y * h);
  context.lineTo(b.x * w, b.y * h);
  context.stroke();
}

function dots(context: CanvasRenderingContext2D, points: readonly Point2[], radius: number, w: number, h: number): void {
  for (const point of points) {
    context.beginPath();
    context.arc(point.x * w, point.y * h, radius, 0, 2 * Math.PI);
    context.fill();
  }
}

export function drawOverlay(context: CanvasRenderingContext2D, input: OverlayInput): void {
  const { width: w, height: h } = context.canvas;
  const scale = Math.max(1, Math.min(w, h) / 240);
  context.lineWidth = 2 * scale;
  for (const [hand, color] of [
    [input.left, COLORS.left],
    [input.right, COLORS.right],
  ] as const) {
    if (!hand) continue;
    context.strokeStyle = color;
    context.fillStyle = color;
    for (const { start, end } of input.connections) {
      const a = hand[start];
      const b = hand[end];
      if (a && b) line(context, a, b, w, h);
    }
    dots(context, hand, 2.5 * scale, w, h);
  }
  if (input.pose) {
    const pose = input.pose;
    context.strokeStyle = COLORS.pose;
    context.fillStyle = COLORS.pose;
    for (const [a, b] of POSE_LINES) {
      const p = pose[a];
      const q = pose[b];
      if (p && q) line(context, p, q, w, h);
    }
    dots(context, [0, 11, 12, 13, 14].map((i) => pose[i]).filter((p): p is Point2 => !!p), 3.5 * scale, w, h);
    // Wrists in the colour of the hand block they should match.
    for (const [index, color] of [
      [15, COLORS.left],
      [16, COLORS.right],
    ] as const) {
      const wrist = pose[index];
      if (!wrist) continue;
      context.fillStyle = color;
      dots(context, [wrist], 6 * scale, w, h);
    }
  }
}
