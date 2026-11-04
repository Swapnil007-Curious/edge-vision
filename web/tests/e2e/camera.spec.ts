import { expect, test } from "@playwright/test";

test.use({
  launchOptions: {
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"],
  },
  permissions: ["camera"],
});

test("camera smoke: the pipeline starts and the FPS counter updates", async ({ page, baseURL }) => {
  const origin = new URL(baseURL as string).origin;
  const foreign: string[] = [];
  const cspViolations: string[] = [];
  page.on("request", (request) => {
    const url = request.url();
    if (!url.startsWith(origin) && !url.startsWith("data:") && !url.startsWith("blob:")) foreign.push(url);
  });
  page.on("console", (message) => {
    if (message.text().includes("Content Security Policy")) cspViolations.push(message.text());
  });

  await page.goto("./");
  const start = page.locator("#start");
  await expect(start).toBeEnabled();
  await start.click();
  await expect(page.locator("#status")).toHaveText(/Running/, { timeout: 120_000 });
  await expect
    .poll(async () => Number.parseFloat((await page.locator("#fps").textContent()) ?? ""), { timeout: 60_000 })
    .toBeGreaterThan(0);
  await expect(page.locator("#preview")).toBeVisible();
  // Let a few more frames through so the logged numbers are not from the first frame alone.
  const totalSamples = page.locator("#perf-body tr").last().locator("td").last();
  await expect.poll(async () => Number((await totalSamples.textContent()) ?? 0), { timeout: 60_000 }).toBeGreaterThanOrEqual(20);
  const fps = await page.locator("#fps").textContent();
  const rows = await page.locator("#perf-body tr").allTextContents();
  console.log(`camera smoke: FPS ${fps}; ${rows.join(" | ")}`);

  await page.locator("#stop").click();
  await expect(page.locator("#status")).not.toHaveText(/failed|Could not/);
  expect(foreign, "requests to other origins").toEqual([]);
  expect(cspViolations, "Content-Security-Policy violations").toEqual([]);
});
