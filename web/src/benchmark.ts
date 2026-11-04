// The 60-second in-browser benchmark: raw per-frame and per-segment timings plus summaries,
// downloaded as JSON. Nothing is sent anywhere.

import { summarize, type Summary } from "./stats";

export interface FrameTiming {
  t_ms: number;
  hand_ms: number;
  pose_ms: number;
  total_ms: number;
}

export interface SegmentTiming {
  t_ms: number;
  frames: number;
  hand_frames: number;
  preprocess_ms: number;
  classifier_ms: number;
  /** From the segment's last frame arriving to the result being shown. */
  end_to_end_ms: number;
  accepted: boolean;
}

export interface BenchmarkContext {
  camera: { width: number; height: number; frame_rate: number | null; requested: object };
  settings: object;
  model: { name: string; variant: string; sha256: string; data_source: string; threshold: number };
  ort_wasm: object;
  mediapipe: object;
}

export interface BenchmarkReport {
  schema_version: 1;
  started_at: string;
  duration_s: number;
  environment: {
    user_agent: string;
    hardware_concurrency: number | null;
    device_memory_gb: number | null;
    cross_origin_isolated: boolean;
  };
  context: BenchmarkContext;
  frames: { count: number; end_to_end_fps: number; raw: FrameTiming[] };
  segments: { count: number; raw: SegmentTiming[] };
  classifier_fixed_input: { runs: number; raw_ms: number[] };
  summary: Record<string, Summary>;
  notes: string[];
}

export const BENCHMARK_MS = 60_000;

export class BenchmarkRecorder {
  private frames: FrameTiming[] = [];
  private segments: SegmentTiming[] = [];
  private startTime = 0;
  private endTime = 0;
  private startedAt = "";
  private running = false;

  constructor(private readonly durationMs = BENCHMARK_MS) {}

  get isRunning(): boolean {
    return this.running;
  }

  start(now: number): void {
    this.frames = [];
    this.segments = [];
    this.startTime = now;
    this.startedAt = new Date().toISOString();
    this.running = true;
  }

  /** Seconds left, or 0 once the duration is over. */
  remaining(now: number): number {
    return Math.max(0, (this.durationMs - (now - this.startTime)) / 1000);
  }

  done(now: number): boolean {
    return this.running && now - this.startTime >= this.durationMs;
  }

  addFrame(timing: Omit<FrameTiming, "t_ms">, now: number): void {
    if (this.running) this.frames.push({ t_ms: now - this.startTime, ...timing });
  }

  addSegment(timing: Omit<SegmentTiming, "t_ms">, now: number): void {
    if (this.running) this.segments.push({ t_ms: now - this.startTime, ...timing });
  }

  /** Stop recording; the timings are kept for report(). */
  stop(now: number): void {
    if (!this.running) return;
    this.running = false;
    this.endTime = now;
  }

  /** Stop and drop everything recorded. */
  cancel(): void {
    this.running = false;
    this.frames = [];
    this.segments = [];
  }

  report(context: BenchmarkContext, fixedInputMs: number[]): BenchmarkReport {
    const elapsed = (this.endTime - this.startTime) / 1000;
    const nav = navigator as Navigator & { deviceMemory?: number };
    const column = <T>(rows: T[], key: keyof T): number[] => rows.map((row) => row[key] as number);
    const notes: string[] = [];
    if (this.segments.length === 0) notes.push("No segments completed; sign during the benchmark to time preprocessing and classification of real segments.");
    return {
      schema_version: 1,
      started_at: this.startedAt,
      duration_s: elapsed,
      environment: {
        user_agent: navigator.userAgent,
        hardware_concurrency: navigator.hardwareConcurrency ?? null,
        device_memory_gb: nav.deviceMemory ?? null,
        cross_origin_isolated: globalThis.crossOriginIsolated === true,
      },
      context,
      frames: { count: this.frames.length, end_to_end_fps: elapsed > 0 ? this.frames.length / elapsed : 0, raw: this.frames },
      segments: { count: this.segments.length, raw: this.segments },
      classifier_fixed_input: { runs: fixedInputMs.length, raw_ms: fixedInputMs },
      summary: {
        hand_landmarker_ms: summarize(column(this.frames, "hand_ms")),
        pose_landmarker_ms: summarize(column(this.frames, "pose_ms")),
        frame_total_ms: summarize(column(this.frames, "total_ms")),
        preprocess_ms: summarize(column(this.segments, "preprocess_ms")),
        classifier_ms: summarize(column(this.segments, "classifier_ms")),
        segment_end_to_end_ms: summarize(column(this.segments, "end_to_end_ms")),
        classifier_fixed_input_ms: summarize(fixedInputMs),
      },
      notes,
    };
  }
}

export function downloadJson(payload: unknown, fileName: string): void {
  const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export function benchmarkFileName(date = new Date()): string {
  const pad = (value: number): string => String(value).padStart(2, "0");
  return `edge-vision-benchmark-${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}.json`;
}
