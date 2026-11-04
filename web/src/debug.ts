// Test hook, installed only when the page URL has ?debug=1.

import type { WebMeta } from "./meta";
import type { LongRows, SequenceInfo } from "./preprocess";

export interface DebugPreprocessResult {
  features: number[] | null;
  presence: number[] | null;
  modelInput: number[] | null;
  info: SequenceInfo;
}

export interface DebugHook {
  /** Resolves once the classifier is loaded; rejects if loading failed. */
  ready: Promise<void>;
  meta(): WebMeta | null;
  classify(input: number[]): Promise<number[]>;
  preprocess(rows: LongRows): DebugPreprocessResult;
  serviceWorkerControlled(): boolean;
}

declare global {
  interface Window {
    __edgevision?: DebugHook;
  }
}

export function debugEnabled(search: string = location.search): boolean {
  return new URLSearchParams(search).get("debug") === "1";
}

export function installDebugHook(hook: DebugHook): void {
  window.__edgevision = hook;
}
