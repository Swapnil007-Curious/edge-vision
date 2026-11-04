import { expect, test } from "@playwright/test";
import { classify, openDebugPage, readJson, type ModelGolden } from "./helpers";

test("works offline once the service worker controls the page", async ({ page, context, baseURL }) => {
  const golden = readJson<ModelGolden>("tests/golden/model_cases.json");
  const input = (golden.cases[0] as ModelGolden["cases"][number]).input.flat();

  await openDebugPage(page);
  await page.waitForFunction(() => navigator.serviceWorker?.controller !== null, undefined, { timeout: 120_000 });
  const online = await classify(page, input);

  await context.setOffline(true);
  const origin = new URL(baseURL as string).origin;
  const fromNetwork: string[] = [];
  page.on("response", (response) => {
    if (response.url().startsWith(origin) && !response.fromServiceWorker()) fromNetwork.push(response.url());
  });
  await page.reload();
  await page.waitForFunction(() => window.__edgevision !== undefined);
  await page.evaluate(() => window.__edgevision?.ready);

  await expect(page.locator("#status")).toHaveText(/Model ready/);
  await expect(page.locator("#offline-status")).toHaveText(/available\./);
  expect(await classify(page, input)).toEqual(online);
  expect(await page.evaluate(() => window.__edgevision?.serviceWorkerControlled())).toBe(true);
  expect(fromNetwork, "responses not served by the service worker").toEqual([]);
  await context.setOffline(false);
});
