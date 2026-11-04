// Edge-Vision service worker, generated at build time from config/service-worker.template.js.
// Precaches every built file (app shell, model, onnxruntime-web wasm, MediaPipe runtime and
// models) so the app works offline after one complete load. Cache-first for same-origin GETs;
// navigations get the cached index.html. Matching ignores Vary (servers may send Vary: Origin,
// and module scripts carry an Origin header the precache requests lack). Cross-origin requests
// are never made by the app.

const VERSION = "__CACHE_VERSION__";
const CACHE = `edge-vision-${VERSION}`;
const PRECACHE = __PRECACHE_URLS__;
const SCOPE = new URL(self.registration.scope);
const INDEX = new URL("index.html", SCOPE).href;

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const cache = await caches.open(CACHE);
      const requests = PRECACHE.map((path) => new Request(new URL(path, SCOPE).href, { cache: "reload" }));
      await cache.addAll(requests);
      await self.skipWaiting();
    })(),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const names = await caches.keys();
      await Promise.all(names.filter((name) => name.startsWith("edge-vision-") && name !== CACHE).map((name) => caches.delete(name)));
      await self.clients.claim();
    })(),
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== SCOPE.origin) return;
  if (request.mode === "navigate") {
    event.respondWith(
      (async () => {
        const cache = await caches.open(CACHE);
        return (await cache.match(INDEX, { ignoreVary: true })) ?? fetch(request);
      })(),
    );
    return;
  }
  event.respondWith(
    (async () => {
      const cache = await caches.open(CACHE);
      return (await cache.match(request, { ignoreSearch: true, ignoreVary: true })) ?? fetch(request);
    })(),
  );
});
