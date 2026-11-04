// Build-only Vite plugins: a Content-Security-Policy that confines every network request to this
// origin, and the generated service worker that precaches the whole build.

import { createHash } from "node:crypto";
import { readFile, readdir, writeFile } from "node:fs/promises";
import { join, relative, resolve, sep } from "node:path";
import type { Plugin, ResolvedConfig } from "vite";

/**
 * connect-src 'self' makes the browser refuse any request to another origin, whatever a library
 * tries. 'wasm-unsafe-eval' allows WebAssembly compilation only, not JavaScript eval.
 */
export const CONTENT_SECURITY_POLICY = [
  "default-src 'self'",
  "script-src 'self' 'wasm-unsafe-eval'",
  "style-src 'self'",
  "img-src 'self' data: blob:",
  "media-src 'self' blob: mediastream:",
  "connect-src 'self'",
  "worker-src 'self' blob:",
  "manifest-src 'self'",
  "object-src 'none'",
  "base-uri 'self'",
  "form-action 'none'",
].join("; ");

export function contentSecurityPolicy(): Plugin {
  return {
    name: "edge-vision-csp",
    apply: "build",
    transformIndexHtml() {
      return [
        {
          tag: "meta",
          attrs: { "http-equiv": "Content-Security-Policy", content: CONTENT_SECURITY_POLICY },
          injectTo: "head-prepend",
        },
      ];
    },
  };
}

async function listFiles(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map((entry) => {
      const path = join(directory, entry.name);
      return entry.isDirectory() ? listFiles(path) : Promise.resolve([path]);
    }),
  );
  return nested.flat();
}

const EXCLUDED = [/\.map$/, /^sw\.js$/, /(^|\/)\.DS_Store$/];

export function serviceWorker(templatePath: string): Plugin {
  let config: ResolvedConfig;
  return {
    name: "edge-vision-service-worker",
    apply: "build",
    configResolved(resolved) {
      config = resolved;
    },
    async closeBundle() {
      const outDir = resolve(config.root, config.build.outDir);
      const files = (await listFiles(outDir))
        .map((path) => relative(outDir, path).split(sep).join("/"))
        .filter((path) => !EXCLUDED.some((pattern) => pattern.test(path)))
        .sort();
      const template = await readFile(templatePath, "utf8");
      // The cache version changes with any precached file or with the worker's own code.
      const digest = createHash("sha256").update(template);
      for (const path of files) {
        digest.update(path);
        digest.update(await readFile(join(outDir, path)));
      }
      const version = digest.digest("hex").slice(0, 16);
      const source = template
        .replace("__CACHE_VERSION__", version)
        .replace("__PRECACHE_URLS__", JSON.stringify(files, null, 2));
      await writeFile(join(outDir, "sw.js"), source);
      config.logger.info(`service worker: ${files.length} files precached, version ${version}`);
    },
  };
}
