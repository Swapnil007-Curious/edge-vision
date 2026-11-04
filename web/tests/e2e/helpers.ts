import { readFileSync } from "node:fs";
import type { Page } from "@playwright/test";
import type { DebugPreprocessResult } from "../../src/debug";
import type { LongRows } from "../../src/preprocess";

export function readJson<T>(relativeToWeb: string): T {
  return JSON.parse(readFileSync(new URL(`../../${relativeToWeb}`, import.meta.url), "utf8")) as T;
}

export interface ModelCase {
  sequence_id: number;
  label: number;
  input: number[][];
  logits: number[];
}

export interface ModelGolden {
  source: string;
  variant: string;
  model_sha256: string;
  cases: ModelCase[];
}

export async function openDebugPage(page: Page): Promise<void> {
  await page.goto("./?debug=1");
  await page.waitForFunction(() => window.__edgevision !== undefined);
  await page.evaluate(() => window.__edgevision?.ready);
}

export function classify(page: Page, input: number[]): Promise<number[]> {
  return page.evaluate((values) => (window.__edgevision as NonNullable<typeof window.__edgevision>).classify(values), input);
}

export function preprocess(page: Page, rows: LongRows): Promise<DebugPreprocessResult> {
  return page.evaluate((value) => (window.__edgevision as NonNullable<typeof window.__edgevision>).preprocess(value), rows);
}

export function argmax(values: readonly number[]): number {
  return values.indexOf(Math.max(...values));
}

export function maxAbsDiff(a: readonly number[], b: readonly number[]): number {
  return a.reduce((max, value, i) => Math.max(max, Math.abs(value - (b[i] as number))), 0);
}
