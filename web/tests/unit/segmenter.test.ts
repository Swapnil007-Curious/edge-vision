import { describe, expect, it } from "vitest";
import { Segmenter, type SegmenterEvent, type SegmenterSettings } from "../../src/segmenter";

const AUTO: SegmenterSettings = { mode: "auto", startFrames: 3, endFrames: 8, maxDurationMs: 3000, minHandFrames: 3 };
const FRAME_MS = 33;

/** Feed a hand pattern ("1" = hand, "0" = none) at 33 ms per frame; collect events. */
function feed(segmenter: Segmenter<number>, pattern: string, startIndex = 0): SegmenterEvent<number>[] {
  const events: SegmenterEvent<number>[] = [];
  [...pattern].forEach((char, offset) => {
    const index = startIndex + offset;
    const event = segmenter.push(index, char === "1", index * FRAME_MS);
    if (event) events.push(event);
  });
  return events;
}

describe("auto mode", () => {
  it("does not start before 3 consecutive hand frames", () => {
    const segmenter = new Segmenter<number>(AUTO);
    expect(feed(segmenter, "110110")).toEqual([]);
    expect(segmenter.state).toBe("idle");
    feed(segmenter, "11", 6);
    expect(segmenter.state).toBe("arming");
    feed(segmenter, "1", 8);
    expect(segmenter.state).toBe("recording");
  });

  it("starts with the 3 arming frames and ends after 8 hand-less frames, trimming them", () => {
    const segmenter = new Segmenter<number>(AUTO);
    const events = feed(segmenter, "00" + "11111" + "0001" + "00000000" + "0");
    expect(events).toHaveLength(1);
    const event = events[0] as SegmenterEvent<number>;
    expect(event.type).toBe("segment");
    if (event.type !== "segment") return;
    expect(event.segment.reason).toBe("hands_lowered");
    expect(event.segment.frames.map((frame) => frame.frame)).toEqual([2, 3, 4, 5, 6, 7, 8, 9, 10]);
    expect(event.segment.handFrames).toBe(6);
    expect(event.segment.startTime).toBe(2 * FRAME_MS);
    expect(event.segment.endTime).toBe(10 * FRAME_MS);
    expect(segmenter.state).toBe("idle");
  });

  it("a gap shorter than 8 frames does not end the segment", () => {
    const segmenter = new Segmenter<number>(AUTO);
    expect(feed(segmenter, "111" + "0000000" + "111")).toEqual([]);
    expect(segmenter.state).toBe("recording");
  });

  it("ends at the maximum length and then starts afresh", () => {
    const segmenter = new Segmenter<number>(AUTO);
    const events = feed(segmenter, "1".repeat(100));
    // Frame 91 is the first at >= 3000 ms (3003 ms) and ends the segment, so it holds 0..91.
    const first = events[0] as SegmenterEvent<number>;
    expect(first.type).toBe("segment");
    if (first.type !== "segment") return;
    expect(first.segment.reason).toBe("max_length");
    expect(first.segment.frames).toHaveLength(92);
    expect(first.segment.frames[91]?.time).toBe(91 * FRAME_MS);
    expect(first.segment.endTime - first.segment.startTime).toBeGreaterThanOrEqual(3000);
    expect(segmenter.state).toBe("recording");
  });

  it("discards segments with fewer hand frames than min_hand_frames", () => {
    const segmenter = new Segmenter<number>({ ...AUTO, minHandFrames: 5 });
    const events = feed(segmenter, "1111" + "0".repeat(8));
    expect(events).toEqual([{ type: "discarded", handFrames: 4, reason: "hands_lowered", startTime: 0, endTime: 3 * FRAME_MS }]);
  });

  it("ignores manual recording calls", () => {
    const segmenter = new Segmenter<number>(AUTO);
    expect(segmenter.setRecording(true, 0)).toBeNull();
    expect(segmenter.state).toBe("idle");
  });
});

describe("manual mode", () => {
  const MANUAL: SegmenterSettings = { ...AUTO, mode: "manual" };

  it("records only while held and returns the segment on release", () => {
    const segmenter = new Segmenter<number>(MANUAL);
    expect(feed(segmenter, "1111")).toEqual([]);
    expect(segmenter.state).toBe("idle");
    segmenter.setRecording(true, 4 * FRAME_MS);
    feed(segmenter, "0110110", 4);
    const event = segmenter.setRecording(false, 11 * FRAME_MS);
    expect(event?.type).toBe("segment");
    if (event?.type !== "segment") return;
    expect(event.segment.reason).toBe("manual_release");
    expect(event.segment.frames.map((frame) => frame.frame)).toEqual([4, 5, 6, 7, 8, 9, 10]);
    expect(event.segment.handFrames).toBe(4);
    expect(segmenter.state).toBe("idle");
  });

  it("has no maximum length", () => {
    const segmenter = new Segmenter<number>(MANUAL);
    segmenter.setRecording(true, 0);
    expect(feed(segmenter, "1".repeat(200))).toEqual([]);
    const event = segmenter.setRecording(false, 200 * FRAME_MS);
    expect(event?.type === "segment" && event.segment.frames.length).toBe(200);
  });

  it("discards a short hold", () => {
    const segmenter = new Segmenter<number>(MANUAL);
    segmenter.setRecording(true, 0);
    feed(segmenter, "0110");
    expect(segmenter.setRecording(false, 4 * FRAME_MS)).toMatchObject({ type: "discarded", handFrames: 2 });
  });

  it("release without a hold does nothing; configure resets", () => {
    const segmenter = new Segmenter<number>(MANUAL);
    expect(segmenter.setRecording(false, 0)).toBeNull();
    segmenter.setRecording(true, 0);
    segmenter.configure(AUTO);
    expect(segmenter.state).toBe("idle");
  });
});
