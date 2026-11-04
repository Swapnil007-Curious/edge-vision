import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";
import { contentSecurityPolicy, serviceWorker } from "./config/plugins.ts";

// GitHub Pages serves the app under /edge-vision/; set BASE_PATH=/edge-vision/ for that build.
const base = process.env.BASE_PATH ?? "/";

export default defineConfig({
  base,
  plugins: [
    contentSecurityPolicy(),
    serviceWorker(fileURLToPath(new URL("./config/service-worker.template.js", import.meta.url))),
  ],
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1500,
  },
  // Pre-bundling would move onnxruntime-web away from its import.meta.url-relative files.
  optimizeDeps: { exclude: ["onnxruntime-web"] },
  preview: { port: 4173, strictPort: true },
  test: {
    include: ["tests/unit/**/*.test.ts"],
    environment: "node",
  },
});
