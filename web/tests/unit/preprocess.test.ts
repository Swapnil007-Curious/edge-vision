import { describe, expect, it } from "vitest";
import {
  DEFAULT_CONFIG,
  frameCoords,
  frameHasHand,
  interp,
  linspace,
  mirrorPermutation,
  nanMedian,
  pivotRows,
  preprocessCoords,
  preprocessRows,
  rint,
  roundToFloat16,
  snapX,
  stackFrames,
  toModelInput,
  type LongRows,
  type Point2,
  type PreprocessConfig,
} from "../../src/preprocess";
import { preprocessGolden } from "./golden";

const golden = preprocessGolden();
const config: PreprocessConfig = {
  ...DEFAULT_CONFIG,
  numFrames: golden.config.num_frames,
  poseIndices: golden.config.pose_indices,
  minHandFrames: golden.config.min_hand_frames,
};
const FEATURE_TOLERANCE = 1e-5;

/** Spacing of float16 values around `value`. */
function float16Ulp(value: number): number {
  const magnitude = Math.abs(value);
  const exponent = magnitude === 0 ? -14 : Math.max(Math.floor(Math.log2(magnitude)), -14);
  return 2 ** (exponent - 10);
}

describe("preprocess.ts against preprocess_cases.json", () => {
  it("has at least 12 cases covering every branch", () => {
    expect(golden.cases.length).toBeGreaterThanOrEqual(12);
    const infos = golden.cases.map((item) => item.info);
    expect(infos.some((info) => info.mirrored)).toBe(true);
    expect(infos.some((info) => info.norm_method === "hand_fallback" && !info.excluded)).toBe(true);
    expect(new Set(infos.map((info) => info.reason))).toEqual(
      new Set([null, "insufficient_hand_frames", "degenerate_scale"]),
    );
  });

  for (const testCase of golden.cases) {
    it(`matches Python: ${testCase.name}`, () => {
      const result = preprocessRows(testCase.rows, config);
      expect(result.info).toEqual(testCase.info);
      if (testCase.features === null) {
        expect(result.features).toBeNull();
        expect(result.presence).toBeNull();
        return;
      }
      const expected = testCase.features.flat();
      const actual = result.features as Float32Array;
      expect(actual.length).toBe(expected.length);
      let maxDiff = 0;
      for (let i = 0; i < expected.length; i += 1) {
        maxDiff = Math.max(maxDiff, Math.abs((actual[i] as number) - (expected[i] as number)));
      }
      expect(maxDiff).toBeLessThanOrEqual(FEATURE_TOLERANCE);
      expect(Array.from(result.presence as Float32Array)).toEqual((testCase.presence as number[][]).flat());

      const modelInput = toModelInput(actual, config.poseIndices) as Float32Array;
      const expectedInput = (testCase.model_input as number[][]).flat();
      for (let i = 0; i < expectedInput.length; i += 1) {
        const want = expectedInput[i] as number;
        expect(Math.abs((modelInput[i] as number) - want)).toBeLessThanOrEqual(float16Ulp(want));
      }
    });
  }

  it("reports the largest feature difference over all cases", () => {
    let maxDiff = 0;
    let exactModelInput = 0;
    let modelInputValues = 0;
    for (const testCase of golden.cases) {
      if (testCase.features === null) continue;
      const features = preprocessRows(testCase.rows, config).features as Float32Array;
      testCase.features.flat().forEach((want, i) => {
        maxDiff = Math.max(maxDiff, Math.abs((features[i] as number) - want));
      });
      const input = toModelInput(features, config.poseIndices) as Float32Array;
      (testCase.model_input as number[][]).flat().forEach((want, i) => {
        modelInputValues += 1;
        if (input[i] === Math.fround(want)) exactModelInput += 1;
      });
    }
    console.log(`preprocess parity: max |features - python| = ${maxDiff}; model_input exact ${exactModelInput}/${modelInputValues}`);
    expect(maxDiff).toBeLessThanOrEqual(FEATURE_TOLERANCE);
  });

  it("snaps x exactly like snap_x, including ties to even", () => {
    for (const { x, snapped } of golden.snap_x_cases) expect(snapX(x)).toBe(snapped);
  });
});

describe("numerical helpers", () => {
  it("rint rounds half to even", () => {
    expect([0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 2.4, -2.6].map(rint)).toEqual([0, 2, 2, -0, -2, -2, 2, -3]);
  });

  it("linspace pins the endpoint and handles a zero span", () => {
    const positions = linspace(0, 44, 32);
    expect(positions[0]).toBe(0);
    expect(positions[31]).toBe(44);
    expect(positions[1]).toBe(44 / 31);
    expect(Array.from(linspace(0, 0, 4))).toEqual([0, 0, 0, 0]);
  });

  it("interp follows numpy at the edges", () => {
    const xp = [1, 3, 4];
    const fp = [10, 30, 0];
    expect(interp(0.5, xp, fp, Number.NaN, Number.NaN)).toBeNaN();
    expect(interp(4.5, xp, fp, Number.NaN, Number.NaN)).toBeNaN();
    expect(interp(4, xp, fp, Number.NaN, Number.NaN)).toBe(0);
    expect(interp(2, xp, fp, Number.NaN, Number.NaN)).toBe(20);
    expect(interp(3, xp, fp, Number.NaN, Number.NaN)).toBe(30);
    expect(interp(5, [5], [7], -1, -2)).toBe(7);
    expect(interp(4, [5], [7], -1, -2)).toBe(-1);
  });

  it("nanMedian ignores NaN and averages the middle pair", () => {
    expect(nanMedian([3, Number.NaN, 1, 2])).toBe(2);
    expect(nanMedian([4, 1, Number.NaN, 2, 3])).toBe(2.5);
    expect(nanMedian([Number.NaN])).toBeNaN();
  });

  it("rounds to float16 like numpy", () => {
    expect(roundToFloat16(65504)).toBe(65504);
    expect(roundToFloat16(65519)).toBe(65504);
    expect(roundToFloat16(65520)).toBe(Number.POSITIVE_INFINITY);
    expect(roundToFloat16(-70000)).toBe(Number.NEGATIVE_INFINITY);
    expect(roundToFloat16(0.1)).toBe(0.0999755859375);
    expect(roundToFloat16(1 / 3)).toBe(0.333251953125);
    expect(roundToFloat16(2 ** -25)).toBe(0);
    expect(roundToFloat16(3 * 2 ** -25)).toBe(2 ** -23);
    expect(roundToFloat16(1 + 2 ** -11)).toBe(1);
    expect(roundToFloat16(1 + 3 * 2 ** -11)).toBe(1 + 2 ** -9);
  });

  it("excludes a sequence whose coordinates overflow float16", () => {
    const features = new Float32Array(32 * 100);
    features[5] = 1e6;
    expect(toModelInput(features)).toBeNull();
  });

  it("mirror permutation is its own inverse", () => {
    const permutation = mirrorPermutation(DEFAULT_CONFIG.poseIndices);
    permutation.forEach((target, source) => expect(permutation[target]).toBe(source));
    expect(permutation[0]).toBe(21);
    expect(permutation[43]).toBe(44); // shoulders 11 <-> 12
  });
});

describe("live frames", () => {
  function points(n: number, seed: number): Point2[] {
    return Array.from({ length: n }, (_, i) => ({ x: ((seed * 31 + i * 17) % 97) / 97, y: ((seed * 13 + i * 7) % 89) / 89 }));
  }

  function rowsFor(frames: { left: Point2[] | null; right: Point2[] | null; pose: Point2[] | null }[]): LongRows {
    const rows: LongRows & { frame: number[]; type: string[]; landmark_index: number[]; x: (number | null)[]; y: (number | null)[]; z: (number | null)[] } = {
      frame: [],
      type: [],
      landmark_index: [],
      x: [],
      y: [],
      z: [],
    };
    frames.forEach((frame, t) => {
      for (const [type, block, size] of [
        ["left_hand", frame.left, 21],
        ["pose", frame.pose, 33],
        ["right_hand", frame.right, 21],
      ] as const) {
        for (let i = 0; i < size; i += 1) {
          rows.frame.push(t);
          rows.type.push(type);
          rows.landmark_index.push(i);
          rows.x.push(block ? (block[i] as Point2).x : null);
          rows.y.push(block ? (block[i] as Point2).y : null);
          rows.z.push(0);
        }
      }
    });
    return rows;
  }

  it("frameCoords + stackFrames equals pivoting the same landmarks as rows", () => {
    const frames = Array.from({ length: 12 }, (_, t) => ({
      left: t % 3 === 0 ? null : points(21, t),
      right: t % 4 === 1 ? null : points(21, t + 50),
      pose: t === 5 ? null : points(33, t + 100),
    }));
    const live = stackFrames(frames.map((frame) => frameCoords(frame.left, frame.right, frame.pose)));
    const pivoted = pivotRows(rowsFor(frames), DEFAULT_CONFIG.poseIndices);
    expect(Array.from(live.data)).toEqual(Array.from(pivoted.data));
    expect(preprocessCoords(live, config)).toEqual(preprocessRows(rowsFor(frames), config));
  });

  it("frameHasHand needs a complete hand", () => {
    expect(frameHasHand(frameCoords(null, null, points(33, 1)))).toBe(false);
    expect(frameHasHand(frameCoords(points(21, 2), null, null))).toBe(true);
    const partial = points(21, 3);
    partial[4] = { x: Number.NaN, y: 0.5 };
    expect(frameHasHand(frameCoords(partial, null, null))).toBe(false);
  });

  it("rejects malformed rows like pivot_sequence", () => {
    const base = rowsFor([{ left: points(21, 1), right: null, pose: null }]);
    expect(() => pivotRows({ ...base, type: base.type.map((t, i) => (i === 0 ? "hand" : t)) }, DEFAULT_CONFIG.poseIndices)).toThrow(/unknown landmark types/);
    expect(() => pivotRows({ ...base, landmark_index: base.landmark_index.map((v, i) => (i === 0 ? 21 : v)) }, DEFAULT_CONFIG.poseIndices)).toThrow(/outside the valid range/);
    expect(() => pivotRows({ ...base, landmark_index: base.landmark_index.map((v, i) => (i === 1 ? 0 : v)) }, DEFAULT_CONFIG.poseIndices)).toThrow(/duplicate/);
    expect(() => pivotRows({ ...base, z: base.z.map((v, i) => (i === 0 ? Number.POSITIVE_INFINITY : v)) }, DEFAULT_CONFIG.poseIndices)).toThrow(/infinite/);
  });
});
