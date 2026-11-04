// Rolling latency statistics and an FPS meter. Pure.

/** Linear-interpolation percentile (numpy's default method) of an unsorted list. */
export function percentile(values: readonly number[], q: number): number {
  if (values.length === 0) return Number.NaN;
  const sorted = Float64Array.from(values).sort();
  const position = (q / 100) * (sorted.length - 1);
  const low = Math.floor(position);
  const high = Math.ceil(position);
  const a = sorted[low] as number;
  const b = sorted[high] as number;
  return a + (b - a) * (position - low);
}

export interface Summary {
  n: number;
  p50: number;
  p95: number;
  mean: number;
}

export function summarize(values: readonly number[]): Summary {
  const n = values.length;
  const mean = n ? values.reduce((sum, value) => sum + value, 0) / n : Number.NaN;
  return { n, p50: percentile(values, 50), p95: percentile(values, 95), mean };
}

/** The last `capacity` samples. */
export class RollingWindow {
  private readonly values: number[] = [];

  constructor(private readonly capacity = 300) {}

  push(value: number): void {
    this.values.push(value);
    if (this.values.length > this.capacity) this.values.shift();
  }

  get size(): number {
    return this.values.length;
  }

  summary(): Summary {
    return summarize(this.values);
  }

  clear(): void {
    this.values.length = 0;
  }
}

/** Frames per second over the frames of the last `windowMs`. */
export class FpsMeter {
  private readonly times: number[] = [];

  constructor(private readonly windowMs = 1000) {}

  tick(time: number): void {
    this.times.push(time);
    while (this.times.length > 0 && (this.times[0] as number) <= time - this.windowMs) this.times.shift();
  }

  fps(now: number): number {
    while (this.times.length > 0 && (this.times[0] as number) <= now - this.windowMs) this.times.shift();
    return (this.times.length * 1000) / this.windowMs;
  }

  clear(): void {
    this.times.length = 0;
  }
}
