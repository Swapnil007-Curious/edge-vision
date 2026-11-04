// Per-sequence preprocessing, a step-by-step port of src/edgevision/data/preprocess.py (the
// source of truth). Pure functions: Float64 internally, Float32 outputs, no DOM access.
//
// Internal coordinates are a Float64Array of shape (T, L, 2) holding x, y; z is never used by the
// exported model (use_z is false) and is only checked for infinities, as pivot_sequence does.

export const LANDMARK_COUNTS = { face: 468, left_hand: 21, pose: 33, right_hand: 21 } as const;
export type LandmarkType = keyof typeof LANDMARK_COUNTS;
export const N_HAND = 21;
export const N_POSE = 33;
export const FIRST_POSE_SLOT = 2 * N_HAND;
export const DEFAULT_POSE_INDICES: readonly number[] = [0, 11, 12, 13, 14, 15, 16];
export const SHOULDERS = [11, 12] as const;
export const POSE_SWAP_PAIRS: readonly (readonly [number, number])[] = [
  [11, 12],
  [13, 14],
  [15, 16],
];
export const MIN_SCALE = 1e-6;
export const X_GRID = 2 ** 32;
export const FLOAT16_MAX = 65504;
const COORDS = 2;

export interface PreprocessConfig {
  numFrames: number;
  poseIndices: readonly number[];
  minHandFrames: number;
  minScale: number;
  xGrid: number;
}

export const DEFAULT_CONFIG: PreprocessConfig = {
  numFrames: 32,
  poseIndices: DEFAULT_POSE_INDICES,
  minHandFrames: 3,
  minScale: MIN_SCALE,
  xGrid: X_GRID,
};

export type NormMethod = "shoulders" | "hand_fallback";
export type ExclusionReason = "insufficient_hand_frames" | "degenerate_scale";

export interface SequenceInfo {
  t_raw: number;
  t_kept: number;
  mirrored: boolean;
  norm_method: NormMethod | null;
  excluded: boolean;
  reason: ExclusionReason | null;
}

export interface PreprocessResult {
  /** (numFrames, L * 2 + 2) float32: landmark-major x, y, then presence left and right. */
  features: Float32Array | null;
  /** (numFrames, 2) float32 presence, left then right. */
  presence: Float32Array | null;
  info: SequenceInfo;
}

/** Long-format rows, column-wise, as in web/tests/golden/preprocess_cases.json. */
export interface LongRows {
  frame: readonly number[];
  type: readonly string[];
  landmark_index: readonly number[];
  x: readonly (number | null)[];
  y: readonly (number | null)[];
  z: readonly (number | null)[];
}

export interface Coords {
  frames: number;
  landmarks: number;
  /** (frames, landmarks, 2) x, y; NaN where missing. */
  data: Float64Array;
}

export function numSelected(poseIndices: readonly number[]): number {
  return FIRST_POSE_SLOT + poseIndices.length;
}

export function frameDim(poseIndices: readonly number[]): number {
  return numSelected(poseIndices) * COORDS + 2;
}

/** Same checks as landmarks.validate_pose_indices. */
export function validatePoseIndices(poseIndices: readonly number[]): void {
  if (new Set(poseIndices).size !== poseIndices.length) {
    throw new Error(`pose_indices must be unique, got [${poseIndices.join(", ")}]`);
  }
  const outside = poseIndices.filter((index) => !(index >= 0 && index < N_POSE));
  if (outside.length > 0) throw new Error(`pose_indices outside 0-${N_POSE - 1}: ${outside}`);
  for (const [first, second] of POSE_SWAP_PAIRS) {
    if (poseIndices.includes(first) !== poseIndices.includes(second)) {
      throw new Error(`pose_indices must contain both or neither of (${first}, ${second})`);
    }
  }
}

/** numpy.rint: round half to even; negative values that round to zero give -0. */
export function rint(value: number): number {
  if (!Number.isFinite(value)) return value;
  const floor = Math.floor(value);
  const fraction = value - floor;
  let result: number;
  if (fraction < 0.5) result = floor;
  else if (fraction > 0.5) result = floor + 1;
  else result = floor % 2 === 0 ? floor : floor + 1;
  return result === 0 && value < 0 ? -0 : result;
}

/** snap_x: round to the dyadic grid; adding 0 turns -0 into 0. */
export function snapX(x: number, xGrid: number = X_GRID): number {
  return rint(x * xGrid) / xGrid + 0;
}

/** Map a (type, landmark_index) pair to its selected slot, or -1 if it is dropped. */
export function selectedSlot(
  type: LandmarkType,
  landmarkIndex: number,
  poseIndices: readonly number[],
): number {
  if (type === "left_hand") return landmarkIndex;
  if (type === "right_hand") return N_HAND + landmarkIndex;
  if (type === "pose") {
    const position = poseIndices.indexOf(landmarkIndex);
    return position < 0 ? -1 : FIRST_POSE_SLOT + position;
  }
  return -1;
}

function isLandmarkType(type: string): type is LandmarkType {
  return Object.prototype.hasOwnProperty.call(LANDMARK_COUNTS, type);
}

function nanIfNull(value: number | null): number {
  return value === null ? Number.NaN : value;
}

/** pivot_sequence: dense (T_raw, L, 2) sorted by frame, x snapped. Contract violations throw. */
export function pivotRows(rows: LongRows, poseIndices: readonly number[], xGrid = X_GRID): Coords {
  const n = rows.frame.length;
  for (const column of [rows.type, rows.landmark_index, rows.x, rows.y, rows.z]) {
    if (column.length !== n) throw new Error("row columns differ in length");
  }
  const unknown = new Set<string>();
  for (const type of rows.type) if (!isLandmarkType(type)) unknown.add(type);
  if (unknown.size > 0) throw new Error(`unknown landmark types: ${[...unknown].sort().join(", ")}`);
  for (let i = 0; i < n; i += 1) {
    const type = rows.type[i] as LandmarkType;
    const index = rows.landmark_index[i] as number;
    if (!Number.isInteger(index) || index < 0 || index >= LANDMARK_COUNTS[type]) {
      throw new Error("landmark_index outside the valid range for its type");
    }
  }
  const frames = [...new Set(rows.frame)].sort((a, b) => a - b);
  const frameRow = new Map(frames.map((frame, position) => [frame, position]));
  const seen = new Set<string>();
  for (let i = 0; i < n; i += 1) {
    const key = `${rows.frame[i]}|${rows.type[i]}|${rows.landmark_index[i]}`;
    if (seen.has(key)) throw new Error("duplicate (frame, type, landmark_index) rows");
    seen.add(key);
  }
  for (const column of [rows.x, rows.y, rows.z]) {
    for (const value of column) {
      if (value === Number.POSITIVE_INFINITY || value === Number.NEGATIVE_INFINITY) {
        throw new Error("infinite coordinates");
      }
    }
  }
  const landmarks = numSelected(poseIndices);
  const data = new Float64Array(frames.length * landmarks * COORDS).fill(Number.NaN);
  for (let i = 0; i < n; i += 1) {
    const slot = selectedSlot(rows.type[i] as LandmarkType, rows.landmark_index[i] as number, poseIndices);
    if (slot < 0) continue;
    const offset = ((frameRow.get(rows.frame[i] as number) as number) * landmarks + slot) * COORDS;
    data[offset] = snapX(nanIfNull(rows.x[i] as number | null), xGrid);
    data[offset + 1] = nanIfNull(rows.y[i] as number | null);
  }
  return { frames: frames.length, landmarks, data };
}

/** hand_presence: a hand is present when all 21 of its x, y values are finite. */
export function handPresence(coords: Coords): Uint8Array {
  const presence = new Uint8Array(coords.frames * 2);
  for (let t = 0; t < coords.frames; t += 1) {
    for (let side = 0; side < 2; side += 1) {
      let present = 1;
      const start = (t * coords.landmarks + side * N_HAND) * COORDS;
      for (let k = 0; k < N_HAND * COORDS; k += 1) {
        if (!Number.isFinite(coords.data[start + k] as number)) {
          present = 0;
          break;
        }
      }
      presence[t * 2 + side] = present;
    }
  }
  return presence;
}

export function selectFrames(coords: Coords, keep: readonly boolean[]): Coords {
  const kept = keep.filter(Boolean).length;
  const frameSize = coords.landmarks * COORDS;
  const data = new Float64Array(kept * frameSize);
  let row = 0;
  for (let t = 0; t < coords.frames; t += 1) {
    if (!keep[t]) continue;
    data.set(coords.data.subarray(t * frameSize, (t + 1) * frameSize), row * frameSize);
    row += 1;
  }
  return { frames: kept, landmarks: coords.landmarks, data };
}

/** left_is_dominant: strictly more left-hand frames; ties go to the right. */
export function leftIsDominant(presence: Uint8Array): boolean {
  let left = 0;
  let right = 0;
  for (let t = 0; t < presence.length / 2; t += 1) {
    left += presence[t * 2] as number;
    right += presence[t * 2 + 1] as number;
  }
  return left > right;
}

export function poseSlot(poseIndices: readonly number[], landmark: number): number {
  return FIRST_POSE_SLOT + poseIndices.indexOf(landmark);
}

/** mirror_permutation: swap the hand blocks and the selected pose pairs. */
export function mirrorPermutation(poseIndices: readonly number[]): Int32Array {
  validatePoseIndices(poseIndices);
  const permutation = new Int32Array(numSelected(poseIndices));
  for (let i = 0; i < permutation.length; i += 1) permutation[i] = i;
  for (let i = 0; i < N_HAND; i += 1) {
    permutation[i] = N_HAND + i;
    permutation[N_HAND + i] = i;
  }
  for (const [first, second] of POSE_SWAP_PAIRS) {
    if (!poseIndices.includes(first)) continue;
    const a = poseSlot(poseIndices, first);
    const b = poseSlot(poseIndices, second);
    const saved = permutation[a] as number;
    permutation[a] = permutation[b] as number;
    permutation[b] = saved;
  }
  return permutation;
}

/** mirror_sequence: permute landmarks, then x -> 1 - x. */
export function mirrorCoords(coords: Coords, poseIndices: readonly number[]): Coords {
  const permutation = mirrorPermutation(poseIndices);
  const data = new Float64Array(coords.data.length);
  for (let t = 0; t < coords.frames; t += 1) {
    for (let l = 0; l < coords.landmarks; l += 1) {
      const source = (t * coords.landmarks + (permutation[l] as number)) * COORDS;
      const target = (t * coords.landmarks + l) * COORDS;
      data[target] = 1.0 - (coords.data[source] as number);
      data[target + 1] = coords.data[source + 1] as number;
    }
  }
  return { frames: coords.frames, landmarks: coords.landmarks, data };
}

/** numpy.nanmedian of the non-NaN values; NaN when there are none. */
export function nanMedian(values: readonly number[]): number {
  const finite = Float64Array.from(values.filter((value) => !Number.isNaN(value))).sort();
  const n = finite.length;
  if (n === 0) return Number.NaN;
  const high = finite[n >> 1] as number;
  const low = n % 2 === 1 ? high : (finite[(n >> 1) - 1] as number);
  return (low + high) / 2;
}

export interface NormalizationParams {
  center: [number, number];
  scale: number;
  method: NormMethod;
}

/** normalization_params: shoulders when any frame has both, else the hands' bounding box. */
export function normalizationParams(coords: Coords, poseIndices: readonly number[]): NormalizationParams {
  const { frames, landmarks, data } = coords;
  if (poseIndices.includes(SHOULDERS[0]) && poseIndices.includes(SHOULDERS[1])) {
    const leftSlot = poseSlot(poseIndices, SHOULDERS[0]);
    const rightSlot = poseSlot(poseIndices, SHOULDERS[1]);
    const midX: number[] = [];
    const midY: number[] = [];
    const distance: number[] = [];
    let anyBoth = false;
    for (let t = 0; t < frames; t += 1) {
      const l = (t * landmarks + leftSlot) * COORDS;
      const r = (t * landmarks + rightSlot) * COORDS;
      const lx = data[l] as number;
      const ly = data[l + 1] as number;
      const rx = data[r] as number;
      const ry = data[r + 1] as number;
      if (Number.isFinite(lx) && Number.isFinite(ly) && Number.isFinite(rx) && Number.isFinite(ry)) {
        anyBoth = true;
      }
      midX.push(0.5 * (lx + rx));
      midY.push(0.5 * (ly + ry));
      distance.push(Math.hypot(lx - rx, ly - ry));
    }
    if (anyBoth) {
      return { center: [nanMedian(midX), nanMedian(midY)], scale: nanMedian(distance), method: "shoulders" };
    }
  }
  // Hand points of every frame, flattened frame-major as reshape(-1, 2) does.
  const sum = [0, 0];
  const count = [0, 0];
  const min = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
  const max = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
  for (let t = 0; t < frames; t += 1) {
    for (let l = 0; l < FIRST_POSE_SLOT; l += 1) {
      for (let axis = 0; axis < COORDS; axis += 1) {
        const value = data[(t * landmarks + l) * COORDS + axis] as number;
        if (Number.isNaN(value)) continue;
        sum[axis] = (sum[axis] as number) + value;
        count[axis] = (count[axis] as number) + 1;
        if (value < (min[axis] as number)) min[axis] = value;
        if (value > (max[axis] as number)) max[axis] = value;
      }
    }
  }
  const center: [number, number] = [0, 1].map((axis) =>
    count[axis] ? (sum[axis] as number) / (count[axis] as number) : Number.NaN,
  ) as [number, number];
  const spread = [0, 1].map((axis) =>
    count[axis] ? (max[axis] as number) - (min[axis] as number) : Number.NaN,
  );
  // numpy's max propagates NaN, unlike Math.max on an empty list.
  const scale = Number.isNaN(spread[0] as number) || Number.isNaN(spread[1] as number)
    ? Number.NaN
    : Math.max(spread[0] as number, spread[1] as number);
  return { center, scale, method: "hand_fallback" };
}

/** normalize_coords without z: (p - center) / scale. */
export function normalizeCoords(coords: Coords, params: NormalizationParams): Coords {
  const data = new Float64Array(coords.data.length);
  for (let i = 0; i < data.length; i += 2) {
    data[i] = ((coords.data[i] as number) - params.center[0]) / params.scale;
    data[i + 1] = ((coords.data[i + 1] as number) - params.center[1]) / params.scale;
  }
  return { frames: coords.frames, landmarks: coords.landmarks, data };
}

/** numpy.linspace(start, stop, num) with endpoint=True, step computed as numpy does. */
export function linspace(start: number, stop: number, num: number): Float64Array {
  const result = new Float64Array(num);
  const div = num - 1;
  const delta = stop - start;
  if (div > 0) {
    const step = delta / div;
    for (let i = 0; i < num; i += 1) {
      result[i] = (step === 0 ? (i / div) * delta : i * step) + start;
    }
    result[num - 1] = stop;
  } else {
    for (let i = 0; i < num; i += 1) result[i] = i * delta + start;
  }
  return result;
}

/** Index j with xs[j] <= key < xs[j + 1]; -1 below xs[0]; xs.length above the last value. */
function searchSorted(key: number, xs: readonly number[]): number {
  const n = xs.length;
  if (key > (xs[n - 1] as number)) return n;
  if (key < (xs[0] as number)) return -1;
  let low = 0;
  let high = n - 1;
  while (high - low > 1) {
    const middle = (low + high) >> 1;
    if ((xs[middle] as number) <= key) low = middle;
    else high = middle;
  }
  return (xs[high] as number) <= key ? high : low;
}

/** numpy.interp(x, xp, fp, left, right) for increasing xp, following its C implementation. */
export function interp(
  x: number,
  xp: readonly number[],
  fp: readonly number[],
  left: number,
  right: number,
): number {
  const n = xp.length;
  if (n === 1) {
    const only = xp[0] as number;
    return x < only ? left : x > only ? right : (fp[0] as number);
  }
  if (Number.isNaN(x)) return x;
  const j = searchSorted(x, xp);
  if (j === -1) return left;
  if (j === n) return right;
  if (j === n - 1) return fp[j] as number;
  const x0 = xp[j] as number;
  const y0 = fp[j] as number;
  if (x0 === x) return y0;
  const x1 = xp[j + 1] as number;
  const y1 = fp[j + 1] as number;
  const slope = (y1 - y0) / (x1 - x0);
  let result = slope * (x - x0) + y0;
  if (Number.isNaN(result)) {
    result = slope * (x - x1) + y1;
    if (Number.isNaN(result) && y0 === y1) result = y0;
  }
  return result;
}

/** resample_coords: per coordinate, between finite values only; no extrapolation. */
export function resampleCoords(coords: Coords, numFrames: number): Coords {
  const { frames, landmarks, data } = coords;
  const positions = linspace(0, frames - 1, numFrames);
  const out = new Float64Array(numFrames * landmarks * COORDS).fill(Number.NaN);
  const sourceIndex: number[] = [];
  const sourceValue: number[] = [];
  for (let l = 0; l < landmarks; l += 1) {
    for (let axis = 0; axis < COORDS; axis += 1) {
      sourceIndex.length = 0;
      sourceValue.length = 0;
      for (let t = 0; t < frames; t += 1) {
        const value = data[(t * landmarks + l) * COORDS + axis] as number;
        if (Number.isFinite(value)) {
          sourceIndex.push(t);
          sourceValue.push(value);
        }
      }
      if (sourceIndex.length === 0) continue;
      for (let i = 0; i < numFrames; i += 1) {
        out[(i * landmarks + l) * COORDS + axis] = interp(
          positions[i] as number,
          sourceIndex,
          sourceValue,
          Number.NaN,
          Number.NaN,
        );
      }
    }
  }
  return { frames: numFrames, landmarks, data: out };
}

/** resample_presence: nearest neighbour over the kept-frame index. */
export function resamplePresence(presence: Uint8Array, numFrames: number): Uint8Array {
  const frames = presence.length / 2;
  const positions = linspace(0, frames - 1, numFrames);
  const out = new Uint8Array(numFrames * 2);
  for (let i = 0; i < numFrames; i += 1) {
    const source = Math.floor((positions[i] as number) + 0.5);
    out[i * 2] = presence[source * 2] as number;
    out[i * 2 + 1] = presence[source * 2 + 1] as number;
  }
  return out;
}

/** fill_missing + packing: NaN -> 0, float32, then presence in the last two columns. */
export function packFeatures(coords: Coords, presence: Uint8Array): Float32Array {
  const coordsPerFrame = coords.landmarks * COORDS;
  const width = coordsPerFrame + 2;
  const out = new Float32Array(coords.frames * width);
  for (let t = 0; t < coords.frames; t += 1) {
    for (let k = 0; k < coordsPerFrame; k += 1) {
      const value = coords.data[t * coordsPerFrame + k] as number;
      out[t * width + k] = Number.isNaN(value) ? 0 : value;
    }
    out[t * width + coordsPerFrame] = presence[t * 2] as number;
    out[t * width + coordsPerFrame + 1] = presence[t * 2 + 1] as number;
  }
  for (const value of out) {
    if (!Number.isFinite(value)) throw new Error("non-finite values remain after NaN replacement");
  }
  return out;
}

function excluded(info: SequenceInfo, reason: ExclusionReason): PreprocessResult {
  return { features: null, presence: null, info: { ...info, excluded: true, reason } };
}

/** preprocess_sequence from dense, already snapped coordinates (T_raw, L, 2). */
export function preprocessCoords(raw: Coords, config: PreprocessConfig = DEFAULT_CONFIG): PreprocessResult {
  const rawPresence = handPresence(raw);
  const keep: boolean[] = [];
  for (let t = 0; t < raw.frames; t += 1) {
    keep.push(rawPresence[t * 2] === 1 || rawPresence[t * 2 + 1] === 1);
  }
  let coords = selectFrames(raw, keep);
  let presence = new Uint8Array(coords.frames * 2);
  let row = 0;
  for (let t = 0; t < raw.frames; t += 1) {
    if (!keep[t]) continue;
    presence[row * 2] = rawPresence[t * 2] as number;
    presence[row * 2 + 1] = rawPresence[t * 2 + 1] as number;
    row += 1;
  }
  const info: SequenceInfo = {
    t_raw: raw.frames,
    t_kept: coords.frames,
    mirrored: false,
    norm_method: null,
    excluded: false,
    reason: null,
  };
  if (info.t_kept < config.minHandFrames) return excluded(info, "insufficient_hand_frames");

  if (leftIsDominant(presence)) {
    coords = mirrorCoords(coords, config.poseIndices);
    const swapped = new Uint8Array(presence.length);
    for (let t = 0; t < coords.frames; t += 1) {
      swapped[t * 2] = presence[t * 2 + 1] as number;
      swapped[t * 2 + 1] = presence[t * 2] as number;
    }
    presence = swapped;
    info.mirrored = true;
  }

  const params = normalizationParams(coords, config.poseIndices);
  info.norm_method = params.method;
  if (!Number.isFinite(params.scale) || params.scale < config.minScale) {
    return excluded(info, "degenerate_scale");
  }
  const resampled = resampleCoords(normalizeCoords(coords, params), config.numFrames);
  const resampledPresence = resamplePresence(presence, config.numFrames);
  return {
    features: packFeatures(resampled, resampledPresence),
    presence: Float32Array.from(resampledPresence),
    info,
  };
}

/** preprocess_sequence from long-format rows. */
export function preprocessRows(rows: LongRows, config: PreprocessConfig = DEFAULT_CONFIG): PreprocessResult {
  validatePoseIndices(config.poseIndices);
  return preprocessCoords(pivotRows(rows, config.poseIndices, config.xGrid), config);
}

/** Round to the nearest float16 value (ties to even), as numpy's float32 -> float16 cast does. */
export function roundToFloat16(value: number): number {
  if (!Number.isFinite(value) || value === 0) return value;
  const magnitude = Math.abs(value);
  let exponent = Math.floor(Math.log2(magnitude));
  if (2 ** exponent > magnitude) exponent -= 1;
  else if (2 ** (exponent + 1) <= magnitude) exponent += 1;
  const ulp = 2 ** (Math.max(exponent, -14) - 10);
  const rounded = rint(magnitude / ulp) * ulp;
  const result = rounded > FLOAT16_MAX ? Number.POSITIVE_INFINITY : rounded;
  return value < 0 ? -result : result;
}

/**
 * The storage step training data took (data/io.py to_storage, then layout.to_model_input):
 * coordinates rounded to float16, presence unchanged. Returns null when a value overflows
 * float16, which preprocessing excluded as "float16_overflow".
 */
export function toModelInput(features: Float32Array, poseIndices: readonly number[] = DEFAULT_POSE_INDICES): Float32Array | null {
  const width = frameDim(poseIndices);
  const coordsPerFrame = width - 2;
  const out = new Float32Array(features.length);
  for (let i = 0; i < features.length; i += 1) {
    const value = features[i] as number;
    if (i % width >= coordsPerFrame) {
      out[i] = value;
      continue;
    }
    const rounded = roundToFloat16(value);
    if (!Number.isFinite(rounded)) return null;
    out[i] = rounded;
  }
  return out;
}

export interface Point2 {
  x: number;
  y: number;
}

/**
 * One live frame as dense selected coordinates (L, 2), x snapped like pivot_sequence: left hand,
 * right hand, then pose landmarks in poseIndices order; NaN where a block is missing.
 */
export function frameCoords(
  leftHand: readonly Point2[] | null,
  rightHand: readonly Point2[] | null,
  pose: readonly Point2[] | null,
  poseIndices: readonly number[] = DEFAULT_POSE_INDICES,
  xGrid = X_GRID,
): Float64Array {
  const out = new Float64Array(numSelected(poseIndices) * COORDS).fill(Number.NaN);
  const put = (slot: number, point: Point2 | undefined): void => {
    if (!point) return;
    out[slot * COORDS] = snapX(point.x, xGrid);
    out[slot * COORDS + 1] = point.y;
  };
  if (leftHand) for (let i = 0; i < N_HAND; i += 1) put(i, leftHand[i]);
  if (rightHand) for (let i = 0; i < N_HAND; i += 1) put(N_HAND + i, rightHand[i]);
  if (pose) poseIndices.forEach((landmark, position) => put(FIRST_POSE_SLOT + position, pose[landmark]));
  return out;
}

/** Stack per-frame coordinates (each L * 2 long) into a Coords block. */
export function stackFrames(frames: readonly Float64Array[], poseIndices: readonly number[] = DEFAULT_POSE_INDICES): Coords {
  const landmarks = numSelected(poseIndices);
  const size = landmarks * COORDS;
  const data = new Float64Array(frames.length * size);
  frames.forEach((frame, t) => {
    if (frame.length !== size) throw new Error(`frame ${t} has ${frame.length} values, expected ${size}`);
    data.set(frame, t * size);
  });
  return { frames: frames.length, landmarks, data };
}

/** True when the frame has a full hand, the rule hand_presence applies. */
export function frameHasHand(frame: Float64Array): boolean {
  for (let side = 0; side < 2; side += 1) {
    let present = true;
    for (let k = side * N_HAND * COORDS; k < (side + 1) * N_HAND * COORDS; k += 1) {
      if (!Number.isFinite(frame[k] as number)) {
        present = false;
        break;
      }
    }
    if (present) return true;
  }
  return false;
}
