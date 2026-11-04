// Generate the app icons (SVG and PNG) into public/icons/ without image libraries: a yellow lens
// ring with a white centre on the app's dark background. Run with `npm run icons`.

import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { deflateSync } from "node:zlib";

const OUT_DIR = join(dirname(fileURLToPath(import.meta.url)), "..", "public", "icons");
const BACKGROUND = [0x0b, 0x0c, 0x0c];
const RING = [0xff, 0xdd, 0x00];
const CENTRE = [0xff, 0xff, 0xff];
const SAMPLES = 4;

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function chunk(type, data) {
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([length, body, crc]);
}

function png(size, colourAt) {
  const raw = Buffer.alloc((size * 3 + 1) * size);
  for (let y = 0; y < size; y += 1) {
    raw[y * (size * 3 + 1)] = 0;
    for (let x = 0; x < size; x += 1) {
      const sum = [0, 0, 0];
      for (let sy = 0; sy < SAMPLES; sy += 1) {
        for (let sx = 0; sx < SAMPLES; sx += 1) {
          const colour = colourAt((x + (sx + 0.5) / SAMPLES) / size, (y + (sy + 0.5) / SAMPLES) / size);
          for (let c = 0; c < 3; c += 1) sum[c] += colour[c];
        }
      }
      for (let c = 0; c < 3; c += 1) {
        raw[y * (size * 3 + 1) + 1 + x * 3 + c] = Math.round(sum[c] / (SAMPLES * SAMPLES));
      }
    }
  }
  const header = Buffer.alloc(13);
  header.writeUInt32BE(size, 0);
  header.writeUInt32BE(size, 4);
  header.set([8, 2, 0, 0, 0], 8);
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", header),
    chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

/** Lens design in unit coordinates; `scale` shrinks it for the maskable safe zone. */
function lens(scale) {
  const outer = 0.36 * scale;
  const inner = 0.25 * scale;
  const centre = 0.11 * scale;
  return (u, v) => {
    const r = Math.hypot(u - 0.5, v - 0.5);
    if (r <= centre) return CENTRE;
    if (r <= outer && r >= inner) return RING;
    return BACKGROUND;
  };
}

const hex = (colour) => `#${colour.map((c) => c.toString(16).padStart(2, "0")).join("")}`;

const SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect width="100" height="100" rx="18" fill="${hex(BACKGROUND)}"/>
  <circle cx="50" cy="50" r="30.5" fill="none" stroke="${hex(RING)}" stroke-width="11"/>
  <circle cx="50" cy="50" r="11" fill="${hex(CENTRE)}"/>
</svg>
`;

await mkdir(OUT_DIR, { recursive: true });
await writeFile(join(OUT_DIR, "icon.svg"), SVG);
await writeFile(join(OUT_DIR, "icon-192.png"), png(192, lens(1)));
await writeFile(join(OUT_DIR, "icon-512.png"), png(512, lens(1)));
await writeFile(join(OUT_DIR, "icon-maskable-512.png"), png(512, lens(0.8)));
console.log(`icons written to ${OUT_DIR}`);
