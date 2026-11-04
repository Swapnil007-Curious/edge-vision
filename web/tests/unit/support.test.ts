// Settings persistence, guidance, statistics, hand assignment, framing and feature detection.

import { describe, expect, it } from "vitest";
import { cropRect } from "../../src/camera";
import { DEFAULT_GUIDANCE_SETTINGS, guidance, meanLuminance } from "../../src/guidance";
import { assignHands, blockForLabel, wristConsistency, type DetectedHand } from "../../src/hands";
import { validateMeta } from "../../src/meta";
import {
  DEFAULT_SETTINGS,
  STORAGE_KEY,
  loadSettings,
  sanitizeSettings,
  saveSettings,
  type KeyValueStore,
} from "../../src/settings";
import { FpsMeter, RollingWindow, percentile } from "../../src/stats";
import { wasmSimdSupported } from "../../src/wasm";
import { readFileSync } from "node:fs";

class MemoryStore implements KeyValueStore {
  values = new Map<string, string>();
  getItem(key: string): string | null {
    return this.values.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.values.set(key, value);
  }
}

const throwingStore: KeyValueStore = {
  getItem() {
    throw new DOMException("blocked", "SecurityError");
  },
  setItem() {
    throw new DOMException("full", "QuotaExceededError");
  },
};

describe("settings", () => {
  it("round-trips through storage", () => {
    const store = new MemoryStore();
    const custom = { ...DEFAULT_SETTINGS, mode: "manual" as const, endFrames: 12, confidenceThreshold: 0.75, swapHands: true };
    expect(saveSettings(store, custom)).toBe(true);
    expect(loadSettings(store)).toEqual(custom);
  });

  it("keeps working when storage throws or is missing", () => {
    expect(loadSettings(throwingStore)).toEqual(DEFAULT_SETTINGS);
    expect(saveSettings(throwingStore, DEFAULT_SETTINGS)).toBe(false);
    expect(loadSettings(null)).toEqual(DEFAULT_SETTINGS);
    expect(saveSettings(null, DEFAULT_SETTINGS)).toBe(false);
  });

  it("replaces invalid or corrupt values with defaults, field by field", () => {
    const store = new MemoryStore();
    store.setItem(STORAGE_KEY, "{not json");
    expect(loadSettings(store)).toEqual(DEFAULT_SETTINGS);
    const sanitized = sanitizeSettings({ mode: "sideways", startFrames: 2.5, endFrames: 10, confidenceThreshold: 2, framing: "portrait", minShoulderDistance: 0.9, maxShoulderDistance: 0.3 });
    expect(sanitized.mode).toBe("auto");
    expect(sanitized.startFrames).toBe(DEFAULT_SETTINGS.startFrames);
    expect(sanitized.endFrames).toBe(10);
    expect(sanitized.confidenceThreshold).toBeNull();
    expect(sanitized.framing).toBe("portrait");
    expect([sanitized.minShoulderDistance, sanitized.maxShoulderDistance]).toEqual([DEFAULT_SETTINGS.minShoulderDistance, DEFAULT_SETTINGS.maxShoulderDistance]);
  });
});

describe("guidance", () => {
  const hand = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5 }));
  const shoulders = [{ x: 0.35, y: 0.6 }, { x: 0.65, y: 0.6 }] as const;

  it("is silent for a well-framed, well-lit signer", () => {
    expect(guidance({ hands: [hand], shoulders, luminance: 120, framesWithoutHands: 0 }, DEFAULT_GUIDANCE_SETTINGS)).toEqual([]);
  });

  it("reports each problem", () => {
    const codes = (input: Parameters<typeof guidance>[0]) => guidance(input, DEFAULT_GUIDANCE_SETTINGS).map((m) => m.code);
    expect(codes({ hands: [], shoulders, luminance: 120, framesWithoutHands: 15 })).toEqual(["no_hands"]);
    expect(codes({ hands: [], shoulders, luminance: 120, framesWithoutHands: 3 })).toEqual([]);
    expect(codes({ hands: [[...hand.slice(1), { x: 1.02, y: 0.5 }]], shoulders, luminance: 120, framesWithoutHands: 0 })).toEqual(["hand_out_of_frame"]);
    expect(codes({ hands: [hand], shoulders, luminance: 20, framesWithoutHands: 0 })).toEqual(["low_light"]);
    expect(codes({ hands: [hand], shoulders: [{ x: 0.1, y: 0.6 }, { x: 0.9, y: 0.6 }], luminance: 120, framesWithoutHands: 0 })).toEqual(["too_close"]);
    expect(codes({ hands: [hand], shoulders: [{ x: 0.45, y: 0.6 }, { x: 0.55, y: 0.6 }], luminance: 120, framesWithoutHands: 0 })).toEqual(["too_far"]);
    expect(codes({ hands: [hand], shoulders: null, luminance: null, framesWithoutHands: 0 })).toEqual(["shoulders_hidden"]);
  });

  it("measures luma", () => {
    expect(meanLuminance([255, 255, 255, 255, 0, 0, 0, 255])).toBeCloseTo(127.5, 10);
  });
});

describe("statistics", () => {
  it("percentile interpolates linearly like numpy", () => {
    expect(percentile([1, 2, 3, 4], 50)).toBe(2.5);
    expect(percentile([10, 0, 20], 95)).toBeCloseTo(19, 10);
    expect(percentile([], 50)).toBeNaN();
  });

  it("keeps only the last 300 samples", () => {
    const window = new RollingWindow(300);
    for (let i = 0; i < 400; i += 1) window.push(i);
    expect(window.size).toBe(300);
    expect(window.summary().p50).toBe(249.5);
  });

  it("counts frames in the last second", () => {
    const meter = new FpsMeter();
    for (let t = 0; t <= 2000; t += 40) meter.tick(t);
    expect(meter.fps(2000)).toBe(25);
    expect(meter.fps(5000)).toBe(0);
  });
});

describe("hand assignment", () => {
  const at = (x: number): { x: number; y: number }[] => Array.from({ length: 21 }, () => ({ x, y: 0.5 }));
  const hand = (label: string, x: number): DetectedHand => ({ label, score: 0.9, landmarks: at(x) });

  it("maps MediaPipe labels for an unmirrored frame, and swaps on request", () => {
    expect(blockForLabel("Left", false)).toBe("right");
    expect(blockForLabel("Right", false)).toBe("left");
    expect(blockForLabel("Left", true)).toBe("left");
    const assigned = assignHands([hand("Left", 0.3), hand("Right", 0.7)], false);
    expect(assigned.right?.[0]?.x).toBe(0.3);
    expect(assigned.left?.[0]?.x).toBe(0.7);
    expect(assignHands([hand("Left", 0.3)], true).left?.[0]?.x).toBe(0.3);
  });

  it("resolves duplicate labels by image position", () => {
    const assigned = assignHands([hand("Left", 0.8), hand("Left", 0.2)], false);
    expect(assigned.resolvedByPosition).toBe(true);
    expect(assigned.right?.[0]?.x).toBe(0.2);
    expect(assigned.left?.[0]?.x).toBe(0.8);
  });

  it("checks hands against the pose wrists", () => {
    const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5 }));
    pose[15] = { x: 0.7, y: 0.5 };
    pose[16] = { x: 0.3, y: 0.5 };
    const consistent = assignHands([hand("Left", 0.3), hand("Right", 0.7)], false);
    expect(wristConsistency(consistent, pose)).toEqual({ checked: 2, consistent: 2 });
    const swapped = assignHands([hand("Left", 0.3), hand("Right", 0.7)], true);
    expect(wristConsistency(swapped, pose)).toEqual({ checked: 2, consistent: 0 });
    expect(wristConsistency(consistent, null)).toEqual({ checked: 0, consistent: 0 });
  });
});

describe("framing and runtime checks", () => {
  it("crops a centred 3:4 region", () => {
    expect(cropRect(640, 480, "full")).toEqual({ sx: 0, sy: 0, sw: 640, sh: 480 });
    expect(cropRect(640, 480, "portrait")).toEqual({ sx: 140, sy: 0, sw: 360, sh: 480 });
    expect(cropRect(480, 960, "portrait")).toEqual({ sx: 0, sy: 160, sw: 480, sh: 640 });
  });

  it("detects WebAssembly SIMD (Node 20 supports it)", () => {
    expect(wasmSimdSupported()).toBe(true);
  });

  it("accepts the committed meta.json and rejects a broken one", () => {
    const raw = JSON.parse(readFileSync(new URL("../../public/model/meta.json", import.meta.url), "utf8"));
    expect(validateMeta(raw).num_classes).toBe(raw.labels.length);
    expect(() => validateMeta({ ...raw, labels: raw.labels.slice(1) })).toThrow(/labels/);
    expect(() => validateMeta({ ...raw, model_sha256: "x" })).toThrow(/sha256/);
  });
});
