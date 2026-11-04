// MediaPipe Tasks HandLandmarker (2 hands) and PoseLandmarker lite in VIDEO mode, loaded from
// this origin's public/mediapipe/ (see scripts/fetch-assets.mjs).

import { HandLandmarker, PoseLandmarker, type NormalizedLandmark } from "@mediapipe/tasks-vision";
import type { DetectedHand } from "./hands";
import type { Delegate } from "./settings";

export interface FrameLandmarks {
  hands: DetectedHand[];
  /** 33 pose landmarks, or null when no pose was found. */
  pose: NormalizedLandmark[] | null;
  handMs: number;
  poseMs: number;
}

export interface Landmarkers {
  delegate: Delegate;
  detect(source: TexImageSource, timestampMs: number): FrameLandmarks;
  close(): void;
}

export const HAND_CONNECTIONS = HandLandmarker.HAND_CONNECTIONS;

export function assetUrl(path: string): string {
  return new URL(`${import.meta.env.BASE_URL}${path}`, document.baseURI).href;
}

export async function createLandmarkers(delegate: Delegate): Promise<Landmarkers> {
  // Only the SIMD build is shipped; FilesetResolver would otherwise probe for a fallback.
  const fileset = {
    wasmLoaderPath: assetUrl("mediapipe/vision_wasm_internal.js"),
    wasmBinaryPath: assetUrl("mediapipe/vision_wasm_internal.wasm"),
  };
  // Created one after the other: each loads the wasm loader script, which sets a global.
  const hand = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: assetUrl("mediapipe/hand_landmarker.task"), delegate },
    runningMode: "VIDEO",
    numHands: 2,
  });
  const pose = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: assetUrl("mediapipe/pose_landmarker_lite.task"), delegate },
    runningMode: "VIDEO",
    numPoses: 1,
    outputSegmentationMasks: false,
  });
  return {
    delegate,
    detect(source: TexImageSource, timestampMs: number): FrameLandmarks {
      const t0 = performance.now();
      const handResult = hand.detectForVideo(source, timestampMs);
      const t1 = performance.now();
      const poseResult = pose.detectForVideo(source, timestampMs);
      const t2 = performance.now();
      const hands: DetectedHand[] = handResult.landmarks.map((landmarks, i) => {
        const category = handResult.handedness[i]?.[0];
        return { landmarks, label: category?.categoryName ?? "", score: category?.score ?? 0 };
      });
      return { hands, pose: poseResult.landmarks[0] ?? null, handMs: t1 - t0, poseMs: t2 - t1 };
    },
    close(): void {
      hand.close();
      pose.close();
    },
  };
}
