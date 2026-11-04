// Camera access and framing. The frame given to MediaPipe is never mirrored; mirroring is a CSS
// transform on the preview canvas only.

import type { Framing } from "./settings";

export const REQUESTED = { width: 640, height: 480, frameRate: 30 } as const;

export interface CropRect {
  sx: number;
  sy: number;
  sw: number;
  sh: number;
}

/** Full frame, or the largest centred 3:4 (portrait) region. */
export function cropRect(width: number, height: number, framing: Framing): CropRect {
  if (framing === "full" || width <= 0 || height <= 0) return { sx: 0, sy: 0, sw: width, sh: height };
  const target = 3 / 4;
  if (width / height > target) {
    const sw = Math.round(height * target);
    return { sx: Math.floor((width - sw) / 2), sy: 0, sw, sh: height };
  }
  const sh = Math.round(width / target);
  return { sx: 0, sy: Math.floor((height - sh) / 2), sw: width, sh };
}

export interface CameraInfo {
  width: number;
  height: number;
  frameRate: number | null;
}

export async function openCamera(video: HTMLVideoElement): Promise<{ stream: MediaStream; info: CameraInfo }> {
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new Error("Camera access is not available in this browser (a secure context is required).");
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      width: { ideal: REQUESTED.width },
      height: { ideal: REQUESTED.height },
      frameRate: { ideal: REQUESTED.frameRate },
      facingMode: "user",
    },
  });
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await video.play();
  const settings = stream.getVideoTracks()[0]?.getSettings() ?? {};
  return {
    stream,
    info: {
      width: settings.width ?? video.videoWidth,
      height: settings.height ?? video.videoHeight,
      frameRate: settings.frameRate ?? null,
    },
  };
}

export function stopCamera(stream: MediaStream | null, video: HTMLVideoElement): void {
  stream?.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
}
