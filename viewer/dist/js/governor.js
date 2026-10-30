// Quality governor. Pure logic; main.js applies the chosen step (pixel ratio, maxStdDev).
// Rule (FINAL_PLAN 2.9): when fps stays under 40 for 2 s, step down; when the median frame rate stays above 55 for 2 s, step up.

export const SQRT5 = Math.sqrt(5);

/**
 * Quality ladder, best first. Measured on an M1 at 1280x720 with 1.0 M splats (NOTES.md): max std dev sqrt(8) gives
 * 49-53 fps with 33-50 ms p95; sqrt(5) gives 60 fps with 16.8 ms p95. So SD starts at sqrt(5) ("standard") and
 * only the HD preset starts at the scene's own value ("full").
 */
export function buildLadder({ basePixelRatio = 1, fullStd = Math.sqrt(8), stress = false, hd = false, stdOverride = 0, previewFallback = false }) {
  const full = stdOverride || Math.min(fullStd, Math.sqrt(8));
  const std = stdOverride || Math.min(full, SQRT5);
  const steps = [];
  if (stress) {
    steps.push({ name: "stress", pixelRatio: 2, maxStdDev: full });
    steps.push({ name: "stress-mid", pixelRatio: 1.4, maxStdDev: std });
  }
  const pr = stress ? 1 : basePixelRatio;
  if (hd && full > std) steps.push({ name: "full", pixelRatio: pr, maxStdDev: full });
  steps.push({ name: "standard", pixelRatio: pr, maxStdDev: std });
  steps.push({ name: "low", pixelRatio: Math.min(0.85, basePixelRatio), maxStdDev: std });
  steps.push({ name: "min", pixelRatio: Math.min(0.7, basePixelRatio), maxStdDev: std });
  // last resort for weak GPUs: swap to the 200 k-splat preview file (the scene's own, already downloaded)
  if (previewFallback) steps.push({ name: "preview", pixelRatio: Math.min(0.7, basePixelRatio), maxStdDev: std, mesh: "preview" });
  return steps;
}

export class Governor {
  constructor({ ladder, start = 0, onStep, downFps = 40, upFps = 55, windowMs = 2000, upCooldownMs = 8000 }) {
    this.ladder = ladder; this.level = start; this.onStep = onStep || (() => {});
    this.downFps = downFps; this.upFps = upFps; this.windowMs = windowMs; this.upCooldownMs = upCooldownMs;
    this.samples = []; // {t, dt}
    this.enabled = true; this.lastDownAt = -Infinity; this.history = [];
  }
  get step() { return this.ladder[this.level]; }
  /** Call once per rendered frame with the frame time in ms and a monotonic timestamp in ms. */
  push(dtMs, nowMs) {
    if (!this.enabled) return;
    this.samples.push({ t: nowMs, dt: dtMs });
    const cut = nowMs - this.windowMs;
    while (this.samples.length > 3 && this.samples[0].t < cut) this.samples.shift();
    const n = this.samples.length;
    // emergency: two frames in a row slower than 400 ms (software rendering, a very weak GPU) cannot wait out a 2 s window
    if (n >= 2 && this.samples[n - 1].dt > 400 && this.samples[n - 2].dt > 400 && this.level < this.ladder.length - 1) {
      this.go(this.level + 1, nowMs, `frames of ${this.samples[n - 1].dt.toFixed(0)} ms`);
      this.lastDownAt = nowMs;
      return;
    }
    const first = this.samples[0];
    if (!first || nowMs - first.t < this.windowMs * 0.92) return;   // window not full yet
    const meanDt = this.samples.reduce((a, s) => a + s.dt, 0) / n;
    const sorted = this.samples.map((s) => s.dt).sort((a, b) => a - b);
    const medDt = sorted[n >> 1];
    if (1000 / meanDt < this.downFps && this.level < this.ladder.length - 1) {
      this.go(this.level + 1, nowMs, `mean ${(1000 / meanDt).toFixed(1)} fps < ${this.downFps}`);
      this.lastDownAt = nowMs;
    } else if (1000 / medDt > this.upFps && this.level > 0 && nowMs - this.lastDownAt > this.upCooldownMs) {
      this.go(this.level - 1, nowMs, `median ${(1000 / medDt).toFixed(1)} fps > ${this.upFps}`);
    }
  }
  go(level, nowMs, why) {
    const from = this.level; this.level = level; this.samples.length = 0;
    this.history.push({ t: nowMs, from, to: level, why, step: this.step });
    this.onStep(this.step, level, why);
  }
  jumpTo(level) { this.level = Math.max(0, Math.min(this.ladder.length - 1, level)); this.samples.length = 0; this.onStep(this.step, this.level, "set"); }
}
