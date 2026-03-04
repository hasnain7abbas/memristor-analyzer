// Frontend smoothing utilities (JS fallback, not normally used — Rust backend is preferred)
export function movingAverage(data: number[], windowSize: number): number[] {
  const half = Math.floor(windowSize / 2);
  return data.map((_, i) => {
    const start = Math.max(0, i - half);
    const end = Math.min(data.length, i + half + 1);
    const window = data.slice(start, end);
    return window.reduce((a, b) => a + b, 0) / window.length;
  });
}
