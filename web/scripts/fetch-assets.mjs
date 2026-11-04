// Download the MediaPipe runtime files and landmarker models into public/mediapipe/ so the app
// never contacts a CDN at runtime. Idempotent: files whose sha256 already matches are kept.
//
// - The wasm loader and binary come from jsDelivr at the exact @mediapipe/tasks-vision version
//   installed by `npm ci`, and must be byte-identical to the copy inside node_modules.
// - The .task models come from Google's model bucket at a fixed model version and must match
//   the sha256 pinned below, so an upstream change fails loudly instead of silently changing
//   the landmarks the classifier sees.
// - Only the SIMD wasm build is fetched: onnxruntime-web ships SIMD-only binaries, so a browser
//   without WebAssembly SIMD cannot run the classifier anyway.
//
// Writes public/mediapipe/ASSETS.json with url, sha256, bytes and license for every file.

import { createHash } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const WEB_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const OUT_DIR = join(WEB_ROOT, "public", "mediapipe");
const PACKAGE_DIR = join(WEB_ROOT, "node_modules", "@mediapipe", "tasks-vision");
const ATTEMPTS = 3;
const TIMEOUT_MS = 120_000;

const APACHE = "Apache-2.0";
const APACHE_URL = "https://www.apache.org/licenses/LICENSE-2.0";

const MODELS = [
  {
    file: "hand_landmarker.task",
    url: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    sha256: "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1",
    license: APACHE,
    license_url: APACHE_URL,
    model_card:
      "https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Hand%20Tracking%20(Lite_Full)%20with%20Fairness%20Oct%202021.pdf",
  },
  {
    file: "pose_landmarker_lite.task",
    url: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    sha256: "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    license: APACHE,
    license_url: APACHE_URL,
    model_card:
      "https://storage.googleapis.com/mediapipe-assets/Model%20Card%20BlazePose%20GHUM%203D.pdf",
  },
];

const WASM_FILES = ["vision_wasm_internal.js", "vision_wasm_internal.wasm"];

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

async function download(url) {
  let lastError;
  for (let attempt = 1; attempt <= ATTEMPTS; attempt += 1) {
    try {
      const response = await fetch(url, { signal: AbortSignal.timeout(TIMEOUT_MS) });
      if (!response.ok) throw new Error(`HTTP ${response.status} ${response.statusText}`);
      return Buffer.from(await response.arrayBuffer());
    } catch (error) {
      lastError = error;
      console.warn(`  attempt ${attempt}/${ATTEMPTS} failed for ${url}: ${error.message}`);
    }
  }
  throw new Error(`could not download ${url}: ${lastError?.message}`);
}

async function writeAtomic(path, bytes) {
  const temporary = `${path}.tmp`;
  await writeFile(temporary, bytes);
  await rm(path, { force: true });
  await rename(temporary, path);
}

/** Keep an existing file whose hash matches, otherwise download, verify and write it. */
async function ensure(asset) {
  const target = join(OUT_DIR, asset.file);
  if (existsSync(target)) {
    const existing = await readFile(target);
    if (sha256(existing) === asset.sha256) {
      console.log(`ok       ${asset.file} (${existing.length} bytes, already present)`);
      return existing.length;
    }
    console.log(`refetch  ${asset.file} (hash differs from expected)`);
  }
  const bytes = await download(asset.url);
  const actual = sha256(bytes);
  if (actual !== asset.sha256) {
    throw new Error(
      `${asset.file}: sha256 ${actual} does not match expected ${asset.sha256} (${asset.url})`,
    );
  }
  await writeAtomic(target, bytes);
  console.log(`fetched  ${asset.file} (${bytes.length} bytes) from ${asset.url}`);
  return bytes.length;
}

async function main() {
  const packageJson = JSON.parse(await readFile(join(PACKAGE_DIR, "package.json"), "utf8"));
  const version = packageJson.version;
  await mkdir(OUT_DIR, { recursive: true });

  // The wasm files must match the JS API version, so their expected hash is the npm copy's.
  const wasmAssets = [];
  for (const file of WASM_FILES) {
    const local = await readFile(join(PACKAGE_DIR, "wasm", file));
    wasmAssets.push({
      file,
      url: `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${version}/wasm/${file}`,
      sha256: sha256(local),
      license: packageJson.license,
      license_url: APACHE_URL,
      npm_package: `@mediapipe/tasks-vision@${version}`,
    });
  }

  const assets = [...wasmAssets, ...MODELS];
  const records = [];
  for (const asset of assets) {
    const bytes = await ensure(asset);
    records.push({ ...asset, bytes });
  }

  const manifest = {
    description:
      "Third-party files served from this origin by the Edge-Vision web demo; written by npm run fetch-assets.",
    tasks_vision_version: version,
    assets: records,
  };
  const text = `${JSON.stringify(manifest, null, 2)}\n`;
  const manifestPath = join(OUT_DIR, "ASSETS.json");
  const previous = existsSync(manifestPath) ? await readFile(manifestPath, "utf8") : null;
  if (previous !== text) await writeAtomic(manifestPath, Buffer.from(text, "utf8"));
  const total = records.reduce((sum, record) => sum + record.bytes, 0);
  console.log(`ASSETS.json ${previous === text ? "unchanged" : "written"}: ${records.length} files, ${total} bytes`);
}

main().catch((error) => {
  console.error(`fetch-assets failed: ${error.message}`);
  process.exit(1);
});
