// Softmax, top-k and the confidence decision. Pure.

export interface Candidate {
  index: number;
  label: string;
  probability: number;
}

export interface Decision {
  /** True when the top probability is >= the threshold. */
  accepted: boolean;
  /** The top label when accepted, otherwise null ("Not sure"). */
  label: string | null;
  probability: number;
  threshold: number;
  top: Candidate[];
}

/** Numerically stable softmax in float64. */
export function softmax(logits: ArrayLike<number>): Float64Array {
  const n = logits.length;
  if (n === 0) throw new Error("softmax of an empty vector");
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < n; i += 1) max = Math.max(max, logits[i] as number);
  const out = new Float64Array(n);
  let sum = 0;
  for (let i = 0; i < n; i += 1) {
    const value = Math.exp((logits[i] as number) - max);
    out[i] = value;
    sum += value;
  }
  for (let i = 0; i < n; i += 1) out[i] = (out[i] as number) / sum;
  return out;
}

/** Indices of the k largest values, largest first; ties go to the lower index. */
export function topK(values: ArrayLike<number>, k: number): number[] {
  const indices = Array.from({ length: values.length }, (_, i) => i);
  indices.sort((a, b) => (values[b] as number) - (values[a] as number) || a - b);
  return indices.slice(0, Math.max(0, Math.min(k, values.length)));
}

export function decide(
  logits: ArrayLike<number>,
  labels: readonly string[],
  threshold: number,
  k = 3,
): Decision {
  if (logits.length !== labels.length) {
    throw new Error(`${logits.length} logits for ${labels.length} labels`);
  }
  const probabilities = softmax(logits);
  const top = topK(probabilities, k).map((index) => ({
    index,
    label: labels[index] as string,
    probability: probabilities[index] as number,
  }));
  const best = top[0] as Candidate;
  const accepted = best.probability >= threshold;
  return { accepted, label: accepted ? best.label : null, probability: best.probability, threshold, top };
}
