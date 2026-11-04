// meta.json written by scripts/export_web.py, with the runtime checks the app relies on.

import { DEFAULT_POSE_INDICES, frameDim, validatePoseIndices, type PreprocessConfig } from "./preprocess";

export interface ThresholdBlock {
  computed_on: "val";
  target_precision: number;
  threshold: number;
  coverage: number;
  accuracy_above_threshold: number;
  overall_val_top1: number;
  val_sequences: number;
  method: "target_precision" | "fallback_max_accuracy";
  fallback_min_coverage: number;
  warning: string | null;
}

export interface InputLayout {
  num_frames: number;
  num_landmarks: number;
  coords_per_landmark: number;
  frame_dim: number;
  presence_left_index: number;
  presence_right_index: number;
  input_name: string;
  output_name: string;
  pose_indices: number[];
}

export interface WebMeta {
  schema_version: 1;
  model_name: string;
  variant: "fp32" | "fp16" | "int8_dynamic" | "int8_static";
  data_source: "asl-signs" | "synthetic";
  run: string;
  num_classes: number;
  labels: string[];
  num_frames: number;
  landmark_order: string[];
  pose_indices: number[];
  input_layout: InputLayout;
  preprocessing: {
    num_frames: number;
    use_z: false;
    pose_indices: number[];
    min_hand_frames: number;
    min_scale: number;
    x_grid: number;
    storage_dtype: "float16";
    source: string;
  };
  opset: number;
  model_file: string;
  model_bytes: number;
  model_sha256: string;
  source_checkpoint_sha256: string;
  threshold: ThresholdBlock;
}

function fail(message: string): never {
  throw new Error(`meta.json: ${message}`);
}

/** Check the fields the app depends on; throws with a readable message otherwise. */
export function validateMeta(raw: unknown): WebMeta {
  if (raw === null || typeof raw !== "object") fail("not an object");
  const meta = raw as WebMeta;
  if (meta.schema_version !== 1) fail(`unsupported schema_version ${String(meta.schema_version)}`);
  if (!Array.isArray(meta.labels) || meta.labels.length !== meta.num_classes || meta.num_classes < 2) {
    fail("labels must list num_classes (>= 2) signs");
  }
  if (!/^[0-9a-f]{64}$/.test(meta.model_sha256)) fail("model_sha256 is not a sha256 hex digest");
  const layout = meta.input_layout;
  const pre = meta.preprocessing;
  if (!layout || !pre) fail("input_layout and preprocessing are required");
  if (pre.use_z !== false) fail("use_z must be false");
  validatePoseIndices(pre.pose_indices);
  if (layout.num_frames !== pre.num_frames || meta.num_frames !== pre.num_frames) fail("num_frames differs between fields");
  if (layout.frame_dim !== frameDim(pre.pose_indices)) fail("frame_dim does not match pose_indices");
  if (layout.coords_per_landmark !== 2) fail("coords_per_landmark must be 2");
  const threshold = meta.threshold?.threshold;
  if (typeof threshold !== "number" || threshold < 0 || threshold > 1) fail("threshold must be in [0, 1]");
  return meta;
}

export function preprocessConfig(meta: WebMeta): PreprocessConfig {
  return {
    numFrames: meta.preprocessing.num_frames,
    poseIndices: meta.preprocessing.pose_indices ?? DEFAULT_POSE_INDICES,
    minHandFrames: meta.preprocessing.min_hand_frames,
    minScale: meta.preprocessing.min_scale,
    xGrid: meta.preprocessing.x_grid,
  };
}
