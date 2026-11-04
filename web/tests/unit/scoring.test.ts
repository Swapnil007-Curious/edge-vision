import { describe, expect, it } from "vitest";
import { decide, softmax, topK } from "../../src/scoring";
import { modelGolden } from "./golden";

describe("softmax", () => {
  it("sums to one and is shift invariant", () => {
    const p = softmax([1, 2, 3]);
    expect(p.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 12);
    const shifted = softmax([1001, 1002, 1003]);
    p.forEach((value, i) => expect(shifted[i]).toBeCloseTo(value, 12));
    expect(p[2]).toBeCloseTo(Math.exp(3) / (Math.exp(1) + Math.exp(2) + Math.exp(3)), 12);
  });

  it("handles extreme logits without overflow", () => {
    expect(Array.from(softmax([1000, -1000]))).toEqual([1, 0]);
  });
});

describe("topK", () => {
  it("orders by value, ties by index, and caps k", () => {
    expect(topK([0.1, 0.5, 0.2, 0.5], 3)).toEqual([1, 3, 2]);
    expect(topK([0.3, 0.7], 5)).toEqual([1, 0]);
    expect(topK([0.3, 0.7], 0)).toEqual([]);
  });
});

describe("decide", () => {
  const labels = ["apple", "bird", "cat", "dog"];

  it("accepts at or above the threshold", () => {
    const logits = [Math.log(6), Math.log(2), Math.log(1), Math.log(1)]; // p ~ 0.6, 0.2, 0.1, 0.1
    const top = softmax(logits)[0] as number;
    const decision = decide(logits, labels, top);
    expect(decision.accepted).toBe(true);
    expect(decision.label).toBe("apple");
    expect(decision.probability).toBeCloseTo(0.6, 12);
    expect(decide(logits, labels, top + 1e-12).accepted).toBe(false);
  });

  it("returns Not sure with the top-3 candidates below the threshold", () => {
    const decision = decide([0.2, 0.9, 0.1, 0.5], labels, 0.95);
    expect(decision.accepted).toBe(false);
    expect(decision.label).toBeNull();
    expect(decision.top.map((candidate) => candidate.label)).toEqual(["bird", "dog", "apple"]);
    expect(decision.top[0]?.probability).toBeGreaterThan(decision.top[1]?.probability ?? 1);
  });

  it("rejects mismatched label counts", () => {
    expect(() => decide([1, 2], labels, 0.5)).toThrow(/labels/);
  });

  it("agrees with Python's argmax on the golden logits", () => {
    const golden = modelGolden();
    const names = Array.from({ length: golden.num_classes }, (_, i) => `class${i}`);
    for (const item of golden.cases) {
      const pythonTop = item.logits.indexOf(Math.max(...item.logits));
      expect(decide(item.logits, names, 0).top[0]?.index).toBe(pythonTop);
    }
  });
});
