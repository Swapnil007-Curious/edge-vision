// User settings and their persistence. Every storage access is wrapped: if localStorage is
// missing, blocked or throws, the app keeps working with in-memory settings.

import { DEFAULT_GUIDANCE_SETTINGS } from "./guidance";
import { DEFAULT_SEGMENTER_SETTINGS, type SegmenterMode } from "./segmenter";

export type Framing = "full" | "portrait";
export type Delegate = "CPU" | "GPU";

export interface Settings {
  mode: SegmenterMode;
  startFrames: number;
  endFrames: number;
  maxDurationS: number;
  /** null means the model's preprocessing min_hand_frames from meta.json. */
  minHandFrames: number | null;
  /** null means the threshold calibrated on val and stored in meta.json. */
  confidenceThreshold: number | null;
  framing: Framing;
  swapHands: boolean;
  speak: boolean;
  delegate: Delegate;
  lowLightThreshold: number;
  minShoulderDistance: number;
  maxShoulderDistance: number;
}

export const DEFAULT_SETTINGS: Settings = {
  mode: DEFAULT_SEGMENTER_SETTINGS.mode,
  startFrames: DEFAULT_SEGMENTER_SETTINGS.startFrames,
  endFrames: DEFAULT_SEGMENTER_SETTINGS.endFrames,
  maxDurationS: DEFAULT_SEGMENTER_SETTINGS.maxDurationMs / 1000,
  minHandFrames: null,
  confidenceThreshold: null,
  framing: "full",
  swapHands: false,
  speak: false,
  delegate: "CPU",
  lowLightThreshold: DEFAULT_GUIDANCE_SETTINGS.lowLightThreshold,
  minShoulderDistance: DEFAULT_GUIDANCE_SETTINGS.minShoulderDistance,
  maxShoulderDistance: DEFAULT_GUIDANCE_SETTINGS.maxShoulderDistance,
};

export const STORAGE_KEY = "edge-vision.settings.v1";

interface NumberRule {
  min: number;
  max: number;
  integer: boolean;
}

export const NUMBER_RULES: Record<
  "startFrames" | "endFrames" | "maxDurationS" | "minHandFrames" | "confidenceThreshold" | "lowLightThreshold" | "minShoulderDistance" | "maxShoulderDistance",
  NumberRule
> = {
  startFrames: { min: 1, max: 30, integer: true },
  endFrames: { min: 1, max: 60, integer: true },
  maxDurationS: { min: 0.5, max: 10, integer: false },
  minHandFrames: { min: 1, max: 60, integer: true },
  confidenceThreshold: { min: 0, max: 1, integer: false },
  lowLightThreshold: { min: 0, max: 255, integer: false },
  minShoulderDistance: { min: 0, max: 2, integer: false },
  maxShoulderDistance: { min: 0, max: 2, integer: false },
};

function validNumber(value: unknown, rule: NumberRule): value is number {
  return (
    typeof value === "number" &&
    Number.isFinite(value) &&
    value >= rule.min &&
    value <= rule.max &&
    (!rule.integer || Number.isInteger(value))
  );
}

function pick<T>(value: unknown, allowed: readonly T[], fallback: T): T {
  return allowed.includes(value as T) ? (value as T) : fallback;
}

/** Keep each valid field of `raw`; anything missing or invalid takes its default. */
export function sanitizeSettings(raw: unknown): Settings {
  const source = (raw !== null && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const number = (key: keyof typeof NUMBER_RULES, fallback: number): number =>
    validNumber(source[key], NUMBER_RULES[key]) ? (source[key] as number) : fallback;
  const nullableNumber = (key: "minHandFrames" | "confidenceThreshold"): number | null =>
    validNumber(source[key], NUMBER_RULES[key]) ? (source[key] as number) : null;
  const settings: Settings = {
    mode: pick(source.mode, ["auto", "manual"] as const, DEFAULT_SETTINGS.mode),
    startFrames: number("startFrames", DEFAULT_SETTINGS.startFrames),
    endFrames: number("endFrames", DEFAULT_SETTINGS.endFrames),
    maxDurationS: number("maxDurationS", DEFAULT_SETTINGS.maxDurationS),
    minHandFrames: nullableNumber("minHandFrames"),
    confidenceThreshold: nullableNumber("confidenceThreshold"),
    framing: pick(source.framing, ["full", "portrait"] as const, DEFAULT_SETTINGS.framing),
    swapHands: typeof source.swapHands === "boolean" ? source.swapHands : DEFAULT_SETTINGS.swapHands,
    speak: typeof source.speak === "boolean" ? source.speak : DEFAULT_SETTINGS.speak,
    delegate: pick(source.delegate, ["CPU", "GPU"] as const, DEFAULT_SETTINGS.delegate),
    lowLightThreshold: number("lowLightThreshold", DEFAULT_SETTINGS.lowLightThreshold),
    minShoulderDistance: number("minShoulderDistance", DEFAULT_SETTINGS.minShoulderDistance),
    maxShoulderDistance: number("maxShoulderDistance", DEFAULT_SETTINGS.maxShoulderDistance),
  };
  if (settings.minShoulderDistance >= settings.maxShoulderDistance) {
    settings.minShoulderDistance = DEFAULT_SETTINGS.minShoulderDistance;
    settings.maxShoulderDistance = DEFAULT_SETTINGS.maxShoulderDistance;
  }
  return settings;
}

/** The subset of the Storage interface used here, so tests can pass a stub. */
export interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

/** window.localStorage, or null when even reading the property throws (e.g. blocked storage). */
export function browserStorage(): KeyValueStore | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

export function loadSettings(store: KeyValueStore | null): Settings {
  try {
    const text = store?.getItem(STORAGE_KEY);
    return sanitizeSettings(text ? JSON.parse(text) : null);
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

/** Returns false when the settings could not be stored; they still apply for this session. */
export function saveSettings(store: KeyValueStore | null, settings: Settings): boolean {
  try {
    if (!store) return false;
    store.setItem(STORAGE_KEY, JSON.stringify(settings));
    return true;
  } catch {
    return false;
  }
}
