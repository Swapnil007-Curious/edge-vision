// Splits the live frame stream into candidate signs. Pure: no timers, no DOM; time comes in
// with every frame.
//
// Auto mode: a segment starts once `startFrames` consecutive frames have any hand (those frames
// are its first frames). It ends after `endFrames` consecutive frames without a hand (the
// trailing hand-less frames are dropped) or when it reaches `maxDurationMs`, whichever comes
// first. Manual mode: frames are recorded while recording is on (Space or the hold button).
// Either way, a segment with fewer hand frames than `minHandFrames` is discarded.

export type SegmenterMode = "auto" | "manual";

export interface SegmenterSettings {
  mode: SegmenterMode;
  startFrames: number;
  endFrames: number;
  maxDurationMs: number;
  minHandFrames: number;
}

export const DEFAULT_SEGMENTER_SETTINGS: SegmenterSettings = {
  mode: "auto",
  startFrames: 3,
  endFrames: 8,
  maxDurationMs: 3000,
  minHandFrames: 3,
};

export type EndReason = "hands_lowered" | "max_length" | "manual_release";

export interface TimedFrame<T> {
  frame: T;
  time: number;
  hasHand: boolean;
}

export interface Segment<T> {
  frames: TimedFrame<T>[];
  startTime: number;
  endTime: number;
  handFrames: number;
  reason: EndReason;
}

export type SegmenterEvent<T> =
  | { type: "segment"; segment: Segment<T> }
  | { type: "discarded"; handFrames: number; reason: EndReason; startTime: number; endTime: number };

export type SegmenterState = "idle" | "arming" | "recording";

export class Segmenter<T> {
  private settings: SegmenterSettings;
  private pending: TimedFrame<T>[] = [];
  private frames: TimedFrame<T>[] = [];
  private recording = false;
  private handFrames = 0;
  private noHandRun = 0;
  private startTime = 0;

  constructor(settings: SegmenterSettings = DEFAULT_SEGMENTER_SETTINGS) {
    this.settings = { ...settings };
  }

  get state(): SegmenterState {
    if (this.recording) return "recording";
    return this.pending.length > 0 ? "arming" : "idle";
  }

  get currentSettings(): SegmenterSettings {
    return { ...this.settings };
  }

  /** Apply new settings; any segment in progress is dropped. */
  configure(settings: SegmenterSettings): void {
    this.settings = { ...settings };
    this.reset();
  }

  reset(): void {
    this.pending = [];
    this.frames = [];
    this.recording = false;
    this.handFrames = 0;
    this.noHandRun = 0;
  }

  /** Feed one frame; returns the event it completes, if any. */
  push(frame: T, hasHand: boolean, time: number): SegmenterEvent<T> | null {
    const timed: TimedFrame<T> = { frame, time, hasHand };
    return this.settings.mode === "auto" ? this.pushAuto(timed) : this.pushManual(timed);
  }

  /** Manual mode only: start (true) or stop (false) recording. Stopping returns the event. */
  setRecording(on: boolean, time: number): SegmenterEvent<T> | null {
    if (this.settings.mode !== "manual") return null;
    if (on) {
      if (!this.recording) {
        this.recording = true;
        this.frames = [];
        this.handFrames = 0;
        this.startTime = time;
      }
      return null;
    }
    if (!this.recording) return null;
    return this.finish(this.frames, "manual_release", time);
  }

  private pushManual(timed: TimedFrame<T>): SegmenterEvent<T> | null {
    if (!this.recording) return null;
    this.frames.push(timed);
    if (timed.hasHand) this.handFrames += 1;
    return null;
  }

  private pushAuto(timed: TimedFrame<T>): SegmenterEvent<T> | null {
    if (!this.recording) {
      if (!timed.hasHand) {
        this.pending = [];
        return null;
      }
      this.pending.push(timed);
      if (this.pending.length < this.settings.startFrames) return null;
      this.recording = true;
      this.frames = this.pending;
      this.pending = [];
      this.handFrames = this.frames.length;
      this.noHandRun = 0;
      this.startTime = (this.frames[0] as TimedFrame<T>).time;
      return this.checkLength(timed.time);
    }

    this.frames.push(timed);
    if (timed.hasHand) {
      this.handFrames += 1;
      this.noHandRun = 0;
    } else {
      this.noHandRun += 1;
      if (this.noHandRun >= this.settings.endFrames) {
        const kept = this.frames.slice(0, this.frames.length - this.noHandRun);
        const last = kept[kept.length - 1];
        return this.finish(kept, "hands_lowered", last ? last.time : timed.time);
      }
    }
    return this.checkLength(timed.time);
  }

  private checkLength(time: number): SegmenterEvent<T> | null {
    if (time - this.startTime < this.settings.maxDurationMs) return null;
    return this.finish(this.frames, "max_length", time);
  }

  private finish(frames: TimedFrame<T>[], reason: EndReason, endTime: number): SegmenterEvent<T> {
    // Trailing hand-less frames are the only ones ever dropped, so the count stays valid.
    const handFrames = this.handFrames;
    const startTime = this.startTime;
    this.reset();
    if (handFrames < this.settings.minHandFrames) {
      return { type: "discarded", handFrames, reason, startTime, endTime };
    }
    return { type: "segment", segment: { frames, startTime, endTime, handFrames, reason } };
  }
}
