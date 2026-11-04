import { expect, test } from "@playwright/test";
import type { SequenceInfo, LongRows } from "../../src/preprocess";
import { argmax, classify, maxAbsDiff, openDebugPage, preprocess, readJson, type ModelGolden } from "./helpers";

const FLOAT_TOLERANCE = 1e-3;

test("model parity: browser logits match Python onnxruntime", async ({ page }) => {
  const golden = readJson<ModelGolden>("tests/golden/model_cases.json");
  const meta = readJson<{ variant: string; model_sha256: string }>("public/model/meta.json");
  expect(golden.model_sha256).toBe(meta.model_sha256);
  await openDebugPage(page);

  let worst = 0;
  let topMatches = 0;
  for (const item of golden.cases) {
    const logits = await classify(page, item.input.flat());
    expect(logits).toHaveLength(item.logits.length);
    const diff = maxAbsDiff(logits, item.logits);
    worst = Math.max(worst, diff);
    if (argmax(logits) === argmax(item.logits)) topMatches += 1;
    if (meta.variant === "fp32" || meta.variant === "fp16") {
      expect(diff, `sequence ${item.sequence_id}`).toBeLessThanOrEqual(FLOAT_TOLERANCE);
    }
  }
  const summary = `variant ${meta.variant}: top-1 matches ${topMatches}/${golden.cases.length}, max |logit diff| ${worst.toExponential(3)} (source: ${golden.source})`;
  console.log(`model parity: ${summary}`);
  test.info().annotations.push({ type: "model parity", description: summary });
  expect(topMatches).toBe(golden.cases.length);
});

interface PreprocessCase {
  name: string;
  rows: LongRows;
  info: SequenceInfo;
  features: number[][] | null;
  presence: number[][] | null;
}

const BROWSER_CASES = ["left_dominant_both_hands", "pose_nan_hand_fallback", "tie_with_edge_gaps"];

test("preprocess parity through the browser bundle", async ({ page }) => {
  const golden = readJson<{ cases: PreprocessCase[] }>("tests/golden/preprocess_cases.json");
  await openDebugPage(page);
  for (const name of BROWSER_CASES) {
    const item = golden.cases.find((candidate) => candidate.name === name);
    expect(item, name).toBeDefined();
    if (!item || !item.features || !item.presence) continue;
    const result = await preprocess(page, item.rows);
    expect(result.info, name).toEqual(item.info);
    expect(maxAbsDiff(result.features ?? [], item.features.flat()), name).toBeLessThanOrEqual(1e-5);
    expect(result.presence, name).toEqual(item.presence.flat());
  }
});
