import "./styles.css";

import { BenchmarkRecorder, benchmarkFileName, downloadJson, type BenchmarkContext } from "./benchmark";
import { REQUESTED, cropRect, openCamera, stopCamera, type CameraInfo } from "./camera";
import { createClassifier, sha256Hex, type Classifier } from "./classifier";
import { debugEnabled, installDebugHook } from "./debug";
import { guidance, meanLuminance, shoulderDistance, type GuidanceMessage, type GuidanceSettings } from "./guidance";
import { assignHands, wristConsistency } from "./hands";
import { HAND_CONNECTIONS, createLandmarkers, type Landmarkers } from "./landmarker";
import { preprocessConfig, validateMeta, type WebMeta } from "./meta";
import { drawOverlay } from "./overlay";
import {
  DEFAULT_CONFIG,
  frameCoords,
  frameHasHand,
  preprocessCoords,
  preprocessRows,
  stackFrames,
  toModelInput,
  type Point2,
  type PreprocessConfig,
} from "./preprocess";
import { decide, type Decision } from "./scoring";
import { Segmenter, type SegmenterEvent, type SegmenterSettings } from "./segmenter";
import {
  DEFAULT_SETTINGS,
  NUMBER_RULES,
  browserStorage,
  loadSettings,
  sanitizeSettings,
  saveSettings,
  type Settings,
} from "./settings";
import { FpsMeter, RollingWindow } from "./stats";

const BASE = import.meta.env.BASE_URL;
const HISTORY_LENGTH = 10;
const PERF_REFRESH_MS = 500;
const LUMINANCE_EVERY = 10;
const WRIST_WINDOW = 90;
const FIXED_INPUT_RUNS = 50;

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`missing #${id}`);
  return found as T;
}

const ui = {
  status: element<HTMLParagraphElement>("status"),
  syntheticWarning: element<HTMLParagraphElement>("synthetic-warning"),
  noticeScope: element<HTMLLIElement>("notice-scope"),
  video: element<HTMLVideoElement>("video"),
  preview: element<HTMLCanvasElement>("preview"),
  placeholder: element<HTMLParagraphElement>("preview-placeholder"),
  guidance: element<HTMLUListElement>("guidance"),
  start: element<HTMLButtonElement>("start"),
  stop: element<HTMLButtonElement>("stop"),
  record: element<HTMLButtonElement>("record"),
  recordHelp: element<HTMLParagraphElement>("record-help"),
  segmentState: element<HTMLParagraphElement>("segment-state"),
  resultSign: element<HTMLParagraphElement>("result-sign"),
  resultDetail: element<HTMLParagraphElement>("result-detail"),
  candidates: element<HTMLOListElement>("candidates"),
  announcer: element<HTMLDivElement>("announcer"),
  history: element<HTMLOListElement>("history"),
  settingsForm: element<HTMLFormElement>("settings"),
  resetSettings: element<HTMLButtonElement>("reset-settings"),
  settingsNote: element<HTMLParagraphElement>("settings-note"),
  fps: element<HTMLOutputElement>("fps"),
  perfBody: element<HTMLTableSectionElement>("perf-body"),
  diagCamera: element<HTMLElement>("diag-camera"),
  diagShoulders: element<HTMLElement>("diag-shoulders"),
  diagLight: element<HTMLElement>("diag-light"),
  diagWrists: element<HTMLElement>("diag-wrists"),
  benchmark: element<HTMLButtonElement>("benchmark"),
  benchmarkStatus: element<HTMLParagraphElement>("benchmark-status"),
  modelInfo: element<HTMLParagraphElement>("model-info"),
  offlineStatus: element<HTMLParagraphElement>("offline-status"),
};

const storage = browserStorage();
let settings: Settings = loadSettings(storage);
let meta: WebMeta | null = null;
let classifier: Classifier | null = null;
let landmarkers: Landmarkers | null = null;
let landmarkersLoading: Promise<Landmarkers> | null = null;
let stream: MediaStream | null = null;
let camera: CameraInfo | null = null;
let running = false;
// Incremented on every start and stop, so a frame callback from an earlier run cannot resume.
let loopGeneration = 0;
let lastTimestamp = 0;
let frameCount = 0;
let framesWithoutHands = 0;
let luminance: number | null = null;
let lastShoulderDistance: number | null = null;
let lastGuidance = "";
let lastPerfRefresh = 0;
let lastSegmentInput: Float32Array | null = null;
let classifyChain: Promise<unknown> = Promise.resolve();
const wristWindow: { checked: number; consistent: number }[] = [];

const segmenter = new Segmenter<Float64Array>();
const fpsMeter = new FpsMeter();
const benchmark = new BenchmarkRecorder();
const windows = {
  hand: new RollingWindow(300),
  pose: new RollingWindow(300),
  preprocess: new RollingWindow(300),
  classifier: new RollingWindow(300),
  total: new RollingWindow(300),
};
const PERF_ROWS: [keyof typeof windows, string][] = [
  ["hand", "Hand landmarker (per frame)"],
  ["pose", "Pose landmarker (per frame)"],
  ["preprocess", "Preprocess (per segment)"],
  ["classifier", "Classifier (per segment)"],
  ["total", "Total per frame"],
];

interface HistoryEntry {
  time: Date;
  text: string;
}
const history: HistoryEntry[] = [];

const cropCanvas = document.createElement("canvas");
const cropContext = cropCanvas.getContext("2d") as CanvasRenderingContext2D;
const lumaCanvas = document.createElement("canvas");
lumaCanvas.width = 32;
lumaCanvas.height = 24;
const lumaContext = lumaCanvas.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D;
const previewContext = ui.preview.getContext("2d") as CanvasRenderingContext2D;
ui.preview.hidden = true;

let resolveReady: () => void = () => undefined;
let rejectReady: (error: unknown) => void = () => undefined;
const modelReady = new Promise<void>((resolve, reject) => {
  resolveReady = resolve;
  rejectReady = reject;
});
modelReady.catch(() => undefined);

// ---------------------------------------------------------------------------------------------
// Helpers

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function setStatus(text: string, kind: "info" | "error" = "info"): void {
  ui.status.textContent = text;
  ui.status.dataset.kind = kind;
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function config(): PreprocessConfig {
  const base = meta ? preprocessConfig(meta) : DEFAULT_CONFIG;
  return { ...base, minHandFrames: settings.minHandFrames ?? base.minHandFrames };
}

function threshold(): number {
  return settings.confidenceThreshold ?? meta?.threshold.threshold ?? 1;
}

function segmenterSettings(): SegmenterSettings {
  return {
    mode: settings.mode,
    startFrames: settings.startFrames,
    endFrames: settings.endFrames,
    maxDurationMs: settings.maxDurationS * 1000,
    minHandFrames: config().minHandFrames,
  };
}

function guidanceSettings(): GuidanceSettings {
  return {
    lowLightThreshold: settings.lowLightThreshold,
    minShoulderDistance: settings.minShoulderDistance,
    maxShoulderDistance: settings.maxShoulderDistance,
    noHandFrames: 15,
  };
}

/** Classifier calls run one at a time: an onnxruntime session is not re-entrant. */
function runClassifier(input: Float32Array): Promise<Float32Array> {
  const run = classifyChain.then(async () => {
    if (!classifier) throw new Error("classifier not loaded");
    return classifier.run(input);
  });
  classifyChain = run.catch(() => undefined);
  return run;
}

// ---------------------------------------------------------------------------------------------
// Settings form

const NUMBER_FIELDS = Object.keys(NUMBER_RULES) as (keyof typeof NUMBER_RULES)[];

function fieldElement<T extends HTMLElement>(name: string): T {
  return element<T>(name);
}

function renderSettings(): void {
  for (const name of NUMBER_FIELDS) {
    const input = fieldElement<HTMLInputElement>(name);
    const value = settings[name];
    input.value = value === null ? "" : String(value);
    input.removeAttribute("aria-invalid");
  }
  const placeholderHandFrames = meta ? String(meta.preprocessing.min_hand_frames) : "";
  fieldElement<HTMLInputElement>("minHandFrames").placeholder = placeholderHandFrames;
  fieldElement<HTMLInputElement>("confidenceThreshold").placeholder = meta ? meta.threshold.threshold.toFixed(3) : "";
  fieldElement<HTMLSelectElement>("framing").value = settings.framing;
  fieldElement<HTMLSelectElement>("delegate").value = settings.delegate;
  fieldElement<HTMLInputElement>("swapHands").checked = settings.swapHands;
  fieldElement<HTMLInputElement>("speak").checked = settings.speak;
  for (const radio of document.querySelectorAll<HTMLInputElement>('input[name="mode"]')) {
    radio.checked = radio.value === settings.mode;
  }
  const manual = settings.mode === "manual";
  ui.record.hidden = !manual;
  ui.recordHelp.hidden = !manual;
}

function persist(): void {
  const saved = saveSettings(storage, settings);
  ui.settingsNote.textContent = saved ? "" : "Settings could not be stored in this browser; they apply until the page is closed.";
}

function applySettings(next: Settings, changed: keyof Settings | "all"): void {
  const previousDelegate = settings.delegate;
  settings = next;
  persist();
  renderSettings();
  segmenter.configure(segmenterSettings());
  setRecordingUi(false);
  if ((changed === "delegate" || changed === "all") && previousDelegate !== settings.delegate) {
    void reloadLandmarkers();
  }
}

function readNumberField(name: keyof typeof NUMBER_RULES): number | null | undefined {
  const input = fieldElement<HTMLInputElement>(name);
  const text = input.value.trim();
  const nullable = name === "minHandFrames" || name === "confidenceThreshold";
  if (text === "" && nullable) return null;
  const value = Number(text);
  const rule = NUMBER_RULES[name];
  const valid = text !== "" && Number.isFinite(value) && value >= rule.min && value <= rule.max && (!rule.integer || Number.isInteger(value));
  input.setAttribute("aria-invalid", valid ? "false" : "true");
  return valid ? value : undefined;
}

function bindSettings(): void {
  ui.settingsForm.addEventListener("change", (event) => {
    const target = event.target as HTMLInputElement | HTMLSelectElement;
    const name = target.name as keyof Settings;
    const raw: Record<string, unknown> = { ...settings };
    if ((NUMBER_FIELDS as string[]).includes(name)) {
      const value = readNumberField(name as keyof typeof NUMBER_RULES);
      if (value === undefined) {
        ui.settingsNote.textContent = `${target.labels?.[0]?.textContent ?? name}: value out of range; previous value kept.`;
        return;
      }
      raw[name] = value;
    } else if (target instanceof HTMLInputElement && target.type === "checkbox") {
      raw[name] = target.checked;
    } else {
      raw[name] = target.value;
    }
    const next = sanitizeSettings(raw);
    if (next.minShoulderDistance !== raw.minShoulderDistance || next.maxShoulderDistance !== raw.maxShoulderDistance) {
      ui.settingsNote.textContent = '"Too far" distance must be below "too close" distance; both were reset.';
    }
    applySettings(next, name);
  });
  ui.settingsForm.addEventListener("submit", (event) => event.preventDefault());
  ui.resetSettings.addEventListener("click", () => {
    applySettings({ ...DEFAULT_SETTINGS }, "all");
    ui.settingsNote.textContent = "Settings reset to defaults.";
  });
  for (const radio of document.querySelectorAll<HTMLInputElement>('input[name="mode"]')) {
    radio.addEventListener("change", () => {
      if (radio.checked) applySettings(sanitizeSettings({ ...settings, mode: radio.value }), "mode");
    });
  }
}

// ---------------------------------------------------------------------------------------------
// Results

function speak(text: string): void {
  if (!settings.speak || typeof speechSynthesis === "undefined") return;
  // Network voices would send the text off the device, so only local voices are used.
  const voices = speechSynthesis.getVoices().filter((voice) => voice.localService);
  const voice = voices.find((candidate) => candidate.lang.toLowerCase().startsWith("en")) ?? voices[0];
  if (!voice) {
    ui.settingsNote.textContent = "No on-device voice is available, so results are not spoken.";
    return;
  }
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.voice = voice;
  utterance.lang = voice.lang;
  speechSynthesis.cancel();
  speechSynthesis.speak(utterance);
}

function renderHistory(): void {
  ui.history.replaceChildren(
    ...history.map((entry) => {
      const item = document.createElement("li");
      item.textContent = `${entry.time.toLocaleTimeString()}: ${entry.text}`;
      return item;
    }),
  );
}

function showDecision(decision: Decision): void {
  const candidates = decision.top.map((candidate) => `${candidate.label} ${percent(candidate.probability)}`);
  if (decision.accepted && decision.label !== null) {
    ui.resultSign.textContent = decision.label;
    ui.resultSign.dataset.state = "sure";
    ui.resultDetail.textContent = `${percent(decision.probability)} confident`;
    ui.candidates.hidden = true;
    ui.announcer.textContent = `Sign: ${decision.label}, ${percent(decision.probability)}.`;
    speak(decision.label);
  } else {
    ui.resultSign.textContent = "Not sure";
    ui.resultSign.dataset.state = "unsure";
    ui.resultDetail.textContent = `Top probability ${percent(decision.probability)} is below the threshold ${percent(decision.threshold)}.`;
    ui.candidates.replaceChildren(
      ...decision.top.map((candidate) => {
        const item = document.createElement("li");
        item.textContent = `${candidate.label}: ${percent(candidate.probability)}`;
        return item;
      }),
    );
    ui.candidates.hidden = false;
    ui.announcer.textContent = `Not sure. Candidates: ${candidates.join(", ")}.`;
  }
  history.unshift({
    time: new Date(),
    text: decision.accepted && decision.label !== null ? `${decision.label} (${percent(decision.probability)})` : `Not sure (${candidates.join(", ")})`,
  });
  history.length = Math.min(history.length, HISTORY_LENGTH);
  renderHistory();
}

const REASON_TEXT: Record<string, string> = {
  insufficient_hand_frames: "too few frames with a hand",
  degenerate_scale: "body or hand too small to normalise",
  float16_overflow: "coordinates out of range",
};

async function handleSegmentEvent(event: SegmenterEvent<Float64Array>, completedAt: number): Promise<void> {
  if (event.type === "discarded") {
    ui.segmentState.textContent = `Segment discarded: ${event.handFrames} frames with a hand, ${config().minHandFrames} needed.`;
    return;
  }
  if (!meta) return;
  const { segment } = event;
  const p0 = performance.now();
  const cfg = config();
  const result = preprocessCoords(stackFrames(segment.frames.map((frame) => frame.frame), cfg.poseIndices), cfg);
  const input = result.features ? toModelInput(result.features, cfg.poseIndices) : null;
  const p1 = performance.now();
  if (!input) {
    const reason = result.info.reason ?? "float16_overflow";
    ui.segmentState.textContent = `No sign recognised: ${REASON_TEXT[reason] ?? reason}.`;
    windows.preprocess.push(p1 - p0);
    return;
  }
  lastSegmentInput = input;
  const logits = await runClassifier(input);
  const p2 = performance.now();
  const decision = decide(logits, meta.labels, threshold());
  showDecision(decision);
  const p3 = performance.now();
  windows.preprocess.push(p1 - p0);
  windows.classifier.push(p2 - p1);
  ui.segmentState.textContent = `Recognised a ${((segment.endTime - segment.startTime) / 1000).toFixed(1)} s segment (${segment.frames.length} frames, ${result.info.mirrored ? "mirrored" : "not mirrored"}).`;
  benchmark.addSegment(
    {
      frames: segment.frames.length,
      hand_frames: segment.handFrames,
      preprocess_ms: p1 - p0,
      classifier_ms: p2 - p1,
      end_to_end_ms: p3 - completedAt,
      accepted: decision.accepted,
    },
    p3,
  );
}

// ---------------------------------------------------------------------------------------------
// Frame loop

function renderGuidance(messages: GuidanceMessage[]): void {
  const key = messages.map((item) => item.code).join(",");
  if (key === lastGuidance) return;
  lastGuidance = key;
  ui.guidance.replaceChildren(
    ...messages.map((item) => {
      const li = document.createElement("li");
      li.textContent = item.text;
      return li;
    }),
  );
}

function prepareSource(): TexImageSource {
  const { videoWidth: width, videoHeight: height } = ui.video;
  const rect = cropRect(width, height, settings.framing);
  let source: TexImageSource = ui.video;
  if (settings.framing !== "full") {
    if (cropCanvas.width !== rect.sw || cropCanvas.height !== rect.sh) {
      cropCanvas.width = rect.sw;
      cropCanvas.height = rect.sh;
    }
    cropContext.drawImage(ui.video, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, rect.sw, rect.sh);
    source = cropCanvas;
  }
  if (ui.preview.width !== rect.sw || ui.preview.height !== rect.sh) {
    ui.preview.width = rect.sw;
    ui.preview.height = rect.sh;
  }
  return source;
}

function measureLuminance(source: TexImageSource): number {
  lumaContext.drawImage(source as CanvasImageSource, 0, 0, lumaCanvas.width, lumaCanvas.height);
  return meanLuminance(lumaContext.getImageData(0, 0, lumaCanvas.width, lumaCanvas.height).data);
}

function updateSegmentState(): void {
  if (segmenter.state === "recording") ui.segmentState.textContent = "Capturing a sign…";
}

async function processFrame(t0: number): Promise<void> {
  if (!landmarkers || !meta) return;
  const source = prepareSource();
  const timestamp = Math.max(lastTimestamp + 1, Math.round(t0));
  lastTimestamp = timestamp;
  const detected = landmarkers.detect(source, timestamp);
  const assigned = assignHands(detected.hands, settings.swapHands);
  const pose: Point2[] | null = detected.pose;
  const cfg = config();
  const frame = frameCoords(assigned.left, assigned.right, pose, cfg.poseIndices, cfg.xGrid);
  const hasHand = frameHasHand(frame);
  framesWithoutHands = detected.hands.length > 0 ? 0 : framesWithoutHands + 1;
  if (frameCount % LUMINANCE_EVERY === 0) luminance = measureLuminance(source);
  frameCount += 1;

  const shoulders = pose && pose[11] && pose[12] ? ([pose[11], pose[12]] as const) : null;
  lastShoulderDistance = shoulders ? shoulderDistance(shoulders) : null;
  renderGuidance(
    guidance(
      { hands: detected.hands.map((hand) => hand.landmarks), shoulders, luminance, framesWithoutHands },
      guidanceSettings(),
    ),
  );
  wristWindow.push(wristConsistency(assigned, pose));
  if (wristWindow.length > WRIST_WINDOW) wristWindow.shift();

  previewContext.drawImage(source as CanvasImageSource, 0, 0, ui.preview.width, ui.preview.height);
  drawOverlay(previewContext, { left: assigned.left, right: assigned.right, pose, connections: HAND_CONNECTIONS });

  const event = segmenter.push(frame, hasHand, t0);
  updateSegmentState();
  if (event) await handleSegmentEvent(event, t0);

  const end = performance.now();
  windows.hand.push(detected.handMs);
  windows.pose.push(detected.poseMs);
  windows.total.push(end - t0);
  fpsMeter.tick(end);
  benchmark.addFrame({ hand_ms: detected.handMs, pose_ms: detected.poseMs, total_ms: end - t0 }, end);
  if (end - lastPerfRefresh >= PERF_REFRESH_MS) {
    lastPerfRefresh = end;
    renderPerformance(end);
  }
  if (benchmark.done(end)) void finishBenchmark();
}

function scheduleFrame(generation: number): void {
  if (!running || generation !== loopGeneration) return;
  const video = ui.video as HTMLVideoElement & {
    requestVideoFrameCallback?: (callback: () => void) => number;
  };
  const next = (): void => void onFrame(generation);
  if (typeof video.requestVideoFrameCallback === "function") video.requestVideoFrameCallback(next);
  else requestAnimationFrame(next);
}

async function onFrame(generation: number): Promise<void> {
  if (!running || generation !== loopGeneration) return;
  try {
    await processFrame(performance.now());
  } catch (error) {
    setStatus(`Frame processing failed: ${message(error)}`, "error");
  }
  scheduleFrame(generation);
}

// ---------------------------------------------------------------------------------------------
// Performance panel and benchmark

function format(value: number): string {
  return Number.isFinite(value) ? value.toFixed(1) : "–";
}

function renderPerformance(now: number): void {
  ui.fps.textContent = running ? fpsMeter.fps(now).toFixed(1) : "–";
  ui.perfBody.replaceChildren(
    ...PERF_ROWS.map(([key, label]) => {
      const summary = windows[key].summary();
      const row = document.createElement("tr");
      for (const [i, text] of [label, format(summary.p50), format(summary.p95), String(summary.n)].entries()) {
        const cell = document.createElement(i === 0 ? "th" : "td");
        if (i === 0) cell.setAttribute("scope", "row");
        cell.textContent = text;
        row.append(cell);
      }
      return row;
    }),
  );
  ui.diagCamera.textContent = camera
    ? `${camera.width} × ${camera.height}${camera.frameRate ? ` at ${camera.frameRate.toFixed(0)} fps` : ""}${settings.framing === "portrait" ? ", portrait crop" : ""}`
    : "–";
  ui.diagShoulders.textContent = lastShoulderDistance === null
    ? "no pose"
    : `${lastShoulderDistance.toFixed(2)} (range ${settings.minShoulderDistance}–${settings.maxShoulderDistance})`;
  ui.diagLight.textContent = luminance === null ? "–" : `${luminance.toFixed(0)} / 255 (warning below ${settings.lowLightThreshold})`;
  const checked = wristWindow.reduce((sum, item) => sum + item.checked, 0);
  const consistent = wristWindow.reduce((sum, item) => sum + item.consistent, 0);
  ui.diagWrists.textContent = checked === 0
    ? "no hand and pose seen together yet"
    : `${percent(consistent / checked)} of recent hands${consistent / checked < 0.5 ? ' (try "Swap hands")' : ""}`;
  if (benchmark.isRunning) {
    ui.benchmarkStatus.textContent = `Benchmark running: ${Math.ceil(benchmark.remaining(now))} s left. Sign a few times.`;
  }
}

function benchmarkContext(): BenchmarkContext {
  const activeMeta = meta as WebMeta;
  return {
    camera: {
      width: camera?.width ?? 0,
      height: camera?.height ?? 0,
      frame_rate: camera?.frameRate ?? null,
      requested: { ...REQUESTED },
    },
    settings: { ...settings, effective_threshold: threshold(), effective_min_hand_frames: config().minHandFrames },
    model: {
      name: activeMeta.model_name,
      variant: activeMeta.variant,
      sha256: activeMeta.model_sha256,
      data_source: activeMeta.data_source,
      threshold: activeMeta.threshold.threshold,
    },
    ort_wasm: { execution_provider: "wasm", num_threads: 1, simd: true, proxy: false },
    mediapipe: { delegate: landmarkers?.delegate ?? settings.delegate, hand: "hand_landmarker.task (numHands 2)", pose: "pose_landmarker_lite.task" },
  };
}

async function finishBenchmark(): Promise<void> {
  if (!benchmark.isRunning || !meta) return;
  benchmark.stop(performance.now());
  const size = meta.input_layout.num_frames * meta.input_layout.frame_dim;
  const input = lastSegmentInput ?? new Float32Array(size);
  const fixed: number[] = [];
  for (let i = 0; i < FIXED_INPUT_RUNS; i += 1) {
    const start = performance.now();
    await runClassifier(input);
    fixed.push(performance.now() - start);
  }
  const final = benchmark.report(benchmarkContext(), fixed);
  downloadJson(final, benchmarkFileName());
  ui.benchmark.disabled = !running;
  ui.benchmarkStatus.textContent = `Benchmark done: ${final.frames.count} frames at ${final.frames.end_to_end_fps.toFixed(1)} fps, ${final.segments.count} segments. The JSON file was downloaded.`;
}

// ---------------------------------------------------------------------------------------------
// Camera controls

async function getLandmarkers(): Promise<Landmarkers> {
  if (landmarkers && landmarkers.delegate === settings.delegate) return landmarkers;
  if (!landmarkersLoading) {
    const delegate = settings.delegate;
    landmarkersLoading = createLandmarkers(delegate)
      .then((created) => {
        landmarkers?.close();
        landmarkers = created;
        return created;
      })
      .finally(() => {
        landmarkersLoading = null;
      });
  }
  return landmarkersLoading;
}

async function reloadLandmarkers(): Promise<void> {
  const wasRunning = running;
  if (wasRunning) stopCapture();
  setStatus(`Loading landmark models (${settings.delegate})…`);
  try {
    await getLandmarkers();
    setStatus("Landmark models ready.");
    if (wasRunning) await startCapture();
  } catch (error) {
    setStatus(`Could not load landmark models: ${message(error)}`, "error");
  }
}

async function startCapture(): Promise<void> {
  if (running || !meta) return;
  ui.start.disabled = true;
  try {
    setStatus("Starting camera…");
    const opened = await openCamera(ui.video);
    stream = opened.stream;
    camera = opened.info;
    setStatus("Loading landmark models…");
    await getLandmarkers();
    running = true;
    frameCount = 0;
    framesWithoutHands = 0;
    segmenter.configure(segmenterSettings());
    fpsMeter.clear();
    ui.preview.hidden = false;
    ui.placeholder.hidden = true;
    ui.stop.disabled = false;
    ui.benchmark.disabled = false;
    ui.benchmarkStatus.textContent = "Ready. Sign a few times during the run so whole segments are timed.";
    ui.segmentState.textContent = settings.mode === "auto" ? "Waiting for hands." : "Hold to record a sign.";
    setStatus("Running. Video stays on this device.");
    loopGeneration += 1;
    scheduleFrame(loopGeneration);
  } catch (error) {
    stopCapture();
    const text = error instanceof DOMException && error.name === "NotAllowedError" ? "Camera permission was denied." : message(error);
    setStatus(`Could not start: ${text}`, "error");
  }
}

function stopCapture(): void {
  running = false;
  loopGeneration += 1;
  stopCamera(stream, ui.video);
  stream = null;
  segmenter.reset();
  setRecordingUi(false);
  if (benchmark.isRunning) {
    benchmark.cancel();
    ui.benchmarkStatus.textContent = "Benchmark cancelled because the camera stopped.";
  }
  ui.preview.hidden = true;
  ui.placeholder.hidden = false;
  ui.stop.disabled = true;
  ui.start.disabled = !meta;
  ui.benchmark.disabled = true;
  ui.segmentState.textContent = "Not capturing.";
  renderGuidance([]);
  renderPerformance(performance.now());
}

function setRecordingUi(on: boolean): void {
  ui.record.setAttribute("aria-pressed", String(on));
  ui.record.textContent = on ? "Recording… release to recognise" : "Hold to record a sign";
}

function startManual(): void {
  if (!running || settings.mode !== "manual" || segmenter.state === "recording") return;
  segmenter.setRecording(true, performance.now());
  setRecordingUi(true);
  ui.segmentState.textContent = "Recording…";
}

function stopManual(): void {
  if (settings.mode !== "manual" || segmenter.state !== "recording") return;
  const now = performance.now();
  const event = segmenter.setRecording(false, now);
  setRecordingUi(false);
  if (event) void handleSegmentEvent(event, now).catch((error) => setStatus(message(error), "error"));
}

function isOtherControl(target: EventTarget | null): boolean {
  if (!(target instanceof Element) || target === ui.record) return false;
  return target.closest("input, select, textarea, button, summary, a, [contenteditable]") !== null;
}

function bindControls(): void {
  ui.start.addEventListener("click", () => void startCapture());
  ui.stop.addEventListener("click", () => stopCapture());
  ui.benchmark.addEventListener("click", () => {
    if (!running || benchmark.isRunning) return;
    benchmark.start(performance.now());
    ui.benchmark.disabled = true;
    ui.benchmarkStatus.textContent = "Benchmark running: 60 s left. Sign a few times.";
  });
  ui.record.addEventListener("pointerdown", (event) => {
    ui.record.setPointerCapture(event.pointerId);
    startManual();
  });
  for (const type of ["pointerup", "pointercancel", "lostpointercapture"] as const) {
    ui.record.addEventListener(type, () => stopManual());
  }
  ui.record.addEventListener("click", (event) => event.preventDefault());
  ui.record.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.repeat) {
      event.preventDefault();
      startManual();
    }
  });
  ui.record.addEventListener("keyup", (event) => {
    if (event.key === "Enter") stopManual();
  });
  document.addEventListener("keydown", (event) => {
    if (event.code !== "Space" || settings.mode !== "manual" || !running || isOtherControl(event.target)) return;
    event.preventDefault();
    if (!event.repeat) startManual();
  });
  document.addEventListener("keyup", (event) => {
    if (event.code !== "Space" || settings.mode !== "manual" || isOtherControl(event.target)) return;
    event.preventDefault();
    stopManual();
  });
  window.addEventListener("blur", () => stopManual());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && running) stopCapture();
  });
}

// ---------------------------------------------------------------------------------------------
// Service worker and boot

function updateOfflineStatus(): void {
  const controlled = "serviceWorker" in navigator && navigator.serviceWorker.controller !== null;
  ui.offlineStatus.textContent = controlled
    ? "Offline use: available. The app, model and landmark files are stored on this device."
    : "Offline use: not yet available (stored after the first complete load).";
}

function registerServiceWorker(): void {
  if (!import.meta.env.PROD || !("serviceWorker" in navigator)) {
    updateOfflineStatus();
    return;
  }
  navigator.serviceWorker.addEventListener("controllerchange", updateOfflineStatus);
  navigator.serviceWorker.register(`${BASE}sw.js`, { scope: BASE }).then(
    () => updateOfflineStatus(),
    (error: unknown) => {
      ui.offlineStatus.textContent = `Offline use: unavailable (${message(error)}).`;
    },
  );
}

function renderModelInfo(active: WebMeta): void {
  ui.noticeScope.textContent = `Recognises single isolated signs from a ${active.num_classes}-sign ASL vocabulary. It does not translate sentences.`;
  ui.syntheticWarning.hidden = active.data_source !== "synthetic";
  ui.modelInfo.textContent = `Model: ${active.model_name} (${active.variant}, ${(active.model_bytes / 1e6).toFixed(2)} MB, sha256 ${active.model_sha256.slice(0, 12)}…), confidence threshold ${active.threshold.threshold.toFixed(3)} calibrated on validation data.`;
}

async function fetchOk(url: string): Promise<Response> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response;
}

async function boot(): Promise<void> {
  renderSettings();
  bindSettings();
  bindControls();
  registerServiceWorker();
  renderPerformance(performance.now());
  try {
    setStatus("Loading model…");
    meta = validateMeta(await (await fetchOk(`${BASE}model/meta.json`)).json());
    renderModelInfo(meta);
    renderSettings();
    const bytes = await (await fetchOk(`${BASE}model/${meta.model_file}`)).arrayBuffer();
    const digest = await sha256Hex(bytes);
    if (digest !== meta.model_sha256) throw new Error("model.onnx does not match the sha256 in meta.json");
    classifier = await createClassifier(bytes, meta);
    segmenter.configure(segmenterSettings());
    ui.start.disabled = false;
    document.body.dataset.modelReady = "true";
    setStatus("Model ready. Press Start camera.");
    resolveReady();
    void getLandmarkers().catch(() => undefined);
  } catch (error) {
    setStatus(`Could not load the model: ${message(error)}`, "error");
    rejectReady(error);
  }
}

if (debugEnabled()) {
  installDebugHook({
    ready: modelReady,
    meta: () => meta,
    async classify(input: number[]): Promise<number[]> {
      await modelReady;
      return Array.from(await runClassifier(Float32Array.from(input)));
    },
    preprocess(rows) {
      const cfg = config();
      const result = preprocessRows(rows, cfg);
      const modelInput = result.features ? toModelInput(result.features, cfg.poseIndices) : null;
      return {
        features: result.features ? Array.from(result.features) : null,
        presence: result.presence ? Array.from(result.presence) : null,
        modelInput: modelInput ? Array.from(modelInput) : null,
        info: result.info,
      };
    },
    serviceWorkerControlled: () => "serviceWorker" in navigator && navigator.serviceWorker.controller !== null,
  });
}

void boot();
