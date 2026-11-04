// onnxruntime-web session for the exported classifier.
//
// The wasm execution provider runs single-threaded (numThreads 1): GitHub Pages cannot send the
// COOP/COEP headers that SharedArrayBuffer, and so wasm threads, require. onnxruntime-web 1.31
// ships SIMD-only binaries, so WebAssembly SIMD is required. The .wasm binary is emitted by Vite
// from node_modules and served from this origin; the JS glue is bundled.

import * as ort from "onnxruntime-web/wasm";
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";
import type { WebMeta } from "./meta";
import { wasmSimdSupported } from "./wasm";

export interface Classifier {
  run(input: Float32Array): Promise<Float32Array>;
  release(): Promise<void>;
}

export async function sha256Hex(bytes: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

export async function createClassifier(modelBytes: ArrayBuffer, meta: WebMeta): Promise<Classifier> {
  if (!wasmSimdSupported()) {
    throw new Error("This browser lacks WebAssembly SIMD, which the classifier runtime requires.");
  }
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.simd = true;
  ort.env.wasm.wasmPaths = { wasm: new URL(ortWasmUrl, document.baseURI).href };
  const session = await ort.InferenceSession.create(new Uint8Array(modelBytes), {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  const { input_name: inputName, output_name: outputName, num_frames: frames, frame_dim: width } = meta.input_layout;
  return {
    async run(input: Float32Array): Promise<Float32Array> {
      if (input.length !== frames * width) {
        throw new Error(`classifier input has ${input.length} values, expected ${frames * width}`);
      }
      const tensor = new ort.Tensor("float32", input, [1, frames, width]);
      const outputs = await session.run({ [inputName]: tensor });
      const logits = outputs[outputName];
      if (!logits) throw new Error(`model output ${outputName} missing`);
      return Float32Array.from(logits.data as Float32Array);
    },
    release: () => session.release(),
  };
}
