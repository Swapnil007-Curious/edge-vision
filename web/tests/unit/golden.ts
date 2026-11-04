// Loaders for the golden files written by scripts/export_web.py.

import { readFileSync } from "node:fs";
import type { LongRows, SequenceInfo } from "../../src/preprocess";

export interface PreprocessCase {
  name: string;
  description: string;
  rows: LongRows;
  info: SequenceInfo;
  features: number[][] | null;
  presence: number[][] | null;
  model_input: number[][] | null;
}

export interface PreprocessGolden {
  config: { num_frames: number; use_z: false; pose_indices: number[]; min_hand_frames: number };
  snap_x_cases: { x: number; snapped: number }[];
  cases: PreprocessCase[];
}

export interface ModelGolden {
  source: string;
  variant: string;
  model_sha256: string;
  num_classes: number;
  cases: { sequence_id: number; label: number; input: number[][]; logits: number[] }[];
}

function load<T>(name: string): T {
  return JSON.parse(readFileSync(new URL(`../golden/${name}`, import.meta.url), "utf8")) as T;
}

export const preprocessGolden = (): PreprocessGolden => load<PreprocessGolden>("preprocess_cases.json");
export const modelGolden = (): ModelGolden => load<ModelGolden>("model_cases.json");
