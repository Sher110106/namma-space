// Quality governor. Pure logic; main.js applies the chosen step (pixel ratio, maxStdDev).
// Rule (FINAL_PLAN 2.9): when fps stays under 40 for 2 s, step down; when the median frame rate stays above 55 for 2 s, step up.
// Added 30 Oct (review finding 15): a step down only stays if it raises the frame rate. A frame rate that does not move when
// quality drops is capped (vsync, battery saver, a background tab), not limited by the GPU, so the step goes back and no new
// step down follows until the frame rate falls well below the capped one.

export const SQRT5 = Math.sqrt(5);

/**
 * Quality ladder, best first. Measured on an M1 at 1280x720 with 1.0 M splats (NOTES.md): max std dev sqrt(8) gives
 * 49-53 fps with 33-50 ms p95; sqrt(5) gives 60 fps with 16.8 ms p95. So SD starts at sqrt(5) ("standard") and
 * only the HD preset starts at the scene's own value ("full").
 * A step that would change nothing (same pixel ratio, same max std dev, same mesh as the step above it) is left out: on a weak GPU
 * the base pixel ratio is already 0.85, so "low" would be "standard" again.
 */
export function buildLadder({ basePixelRatio = 1, fullStd = Math.sqrt(8), stress = false, hd = false, stdOverride = 0, previewFallback = false }) {
  const full = stdOverride || Math.min(fullStd, Math.sqrt(8));
  const std = stdOverride || Math.min(full, SQRT5);
  const steps = [];
  const add = (s) => {
    const p = steps[steps.length - 1];
    if (p && p.pixelRatio === s.pixelRatio && p.maxStdDev === s.maxStdDev && (p.mesh || null) === (s.mesh || null)) return;
    steps.push(s);
  };
  if (stress) {
    add({ name: "stress", pixelRatio: 2, maxStdDev: full });
    add({ name: "stress-mid", pixelRatio: 1.4, maxStdDev: std });
  }
  const pr = stress ? 1 : basePixelRatio;
  if (hd && full > std) add({ name: "full", pixelRatio: pr, maxStdDev: full });
  add({ name: "standard", pixelRatio: pr, maxStdDev: std });
  add({ name: "low", pixelRatio: Math.min(0.85, basePixelRatio), maxStdDev: std });
  add({ name: "min", pixelRatio: Math.min(0.7, basePixelRatio), maxStdDev: std });
  // last resort for weak GPUs: swap to the 200 k-splat preview file (the scene's own, already downloaded)
  if (previewFallback) add({ name: "preview", pixelRatio: Math.min(0.7, basePixelRatio), maxStdDev: std, mesh: "preview" });
  return steps;
}

export class Governor {
  /**
   * onStep(step, level, why, kind) runs when a step is applied; kind is "down" (a trial that is not yet confirmed), "revert", "up",
   * "emergency" or "set". onConfirm(step, level) runs when a trial step down raised the frame rate.
   */
  constructor({ ladder, start = 0, onStep, onConfirm, downFps = 40, upFps = 55, windowMs = 2000, upCooldownMs = 8000, minGain = 1.1, retryDrop = 0.8, emergencyMs = 400 }) {
    this.ladder = ladder; this.level = start; this.onStep = onStep || (() => {}); this.onConfirm = onConfirm || (() => {});
    this.downFps = downFps; this.upFps = upFps; this.windowMs = windowMs; this.upCooldownMs = upCooldownMs;
    this.minGain = minGain; this.retryDrop = retryDrop; this.emergencyMs = emergencyMs;
    this.samples = []; // {t, dt}
    this.enabled = true; this.lastDownAt = -Infinity; this.history = [];
    this.slow = 0;          // frames in a row slower than emergencyMs
    this.trial = null;      // {from, fps0}: a step down that has not yet shown it helps
    this.blockFps = null;   // frame rate at which a step down did not help; no new step down until the rate falls well below it
    this.lastUpAt = -Infinity;
    this.upLimit = 0; this.upLimitUntil = 0;   // a step up that dropped the rate under downFps again is not tried for a minute
    this.chainTop = null;   // the level where the current run of step downs began (null when the rate is steady)
    this.verify = null;     // {ref, back}: after a cap shows up, the top of the run is measured again before the lower level is kept
  }
  get step() { return this.ladder[this.level]; }
  /** Forget what was measured so far (tab hidden then shown, benchmark over, reality check closed): old frames say nothing about now. */
  reset() { this.samples.length = 0; this.slow = 0; }
  setEnabled(on) { this.enabled = !!on; this.reset(); }
  /** Call once per rendered frame with the frame time in ms and a monotonic timestamp in ms. */
  push(dtMs, nowMs) {
    if (!this.enabled) return;
    const last = this.ladder.length - 1;
    this.slow = dtMs > this.emergencyMs ? this.slow + 1 : 0;
    // emergency: two frames in a row slower than 400 ms (software rendering, a very weak GPU) cannot wait out a 2 s window
    if (this.slow >= 2 && this.level < last) {
      this.trial = null; this.verify = null;
      this.go(this.level + 1, nowMs, `frames of ${dtMs.toFixed(0)} ms`, "emergency");
      this.lastDownAt = nowMs;
      return;
    }
    // one frame longer than the window is a pause (hidden tab, debugger, sleep), not a measurement of the GPU
    if (dtMs > this.windowMs) { this.samples.length = 0; return; }
    this.samples.push({ t: nowMs, dt: dtMs });
    const cut = nowMs - this.windowMs;
    while (this.samples.length && this.samples[0].t < cut) this.samples.shift();
    const n = this.samples.length;
    const first = this.samples[0];
    if (!first || nowMs - first.t < this.windowMs * 0.92) return;   // window not full yet
    const meanDt = this.samples.reduce((a, s) => a + s.dt, 0) / n;
    const fps = 1000 / meanDt;
    const sorted = this.samples.map((s) => s.dt).sort((a, b) => a - b);
    const medDt = sorted[n >> 1];
    if (this.verify) {
      // a cap showed up after earlier steps had looked like gains (their first reading may have been a loading hiccup): measure the top again
      const v = this.verify; this.verify = null;
      if (v.ref >= fps * this.minGain) {
        this.blockFps = v.ref; this.lastDownAt = nowMs;      // the top is clearly slower: those steps were real, go back to where we were
        this.go(v.back, nowMs, `top of the run gives ${fps.toFixed(1)} fps, the lower level ${v.ref.toFixed(1)}: the steps help`, "revert");
        return;
      }
      this.blockFps = v.ref; this.chainTop = null;           // same rate at the top: capped, so the top stays
    }
    if (this.trial) {
      const t = this.trial; this.trial = null;
      if (fps < t.fps0 * this.minGain) {
        // the step cost quality and bought nothing: the rate is capped, or the GPU is not what limits it
        this.blockFps = t.fps0; this.lastDownAt = nowMs;
        const top = this.chainTop;
        if (top !== null && top < t.from) {                 // earlier steps of this run were kept on a gain: measure the top again
          this.verify = { ref: Math.max(fps, t.fps0), back: t.from };
          this.go(top, nowMs, `no gain: ${fps.toFixed(1)} fps after the step, ${t.fps0.toFixed(1)} before; checking the top of the run again`, "revert");
        } else {
          this.chainTop = null;
          this.go(t.from, nowMs, `no gain: ${fps.toFixed(1)} fps after the step, ${t.fps0.toFixed(1)} before; going back`, "revert");
        }
        return;
      }
      this.blockFps = null;
      this.onConfirm(this.step, this.level);
    }
    if (fps >= this.downFps) this.chainTop = null;          // a steady rate ends the run of steps
    const allowed = this.blockFps === null || fps < this.blockFps * this.retryDrop;
    if (fps < this.downFps && this.level < last && allowed) {
      if (nowMs - this.lastUpAt < this.windowMs * 3) { this.upLimit = this.level + 1; this.upLimitUntil = nowMs + 60000; }   // undoing a step up: do not climb into it again soon
      this.trial = { from: this.level, fps0: fps };
      if (this.chainTop === null) this.chainTop = this.level;
      this.go(this.level + 1, nowMs, `mean ${fps.toFixed(1)} fps < ${this.downFps}`, "down");
      this.lastDownAt = nowMs;
    } else if (1000 / medDt > this.upFps && this.level > (nowMs < this.upLimitUntil ? this.upLimit : 0) && nowMs - this.lastDownAt > this.upCooldownMs) {
      this.blockFps = null; this.lastUpAt = nowMs; this.chainTop = null;
      this.go(this.level - 1, nowMs, `median ${(1000 / medDt).toFixed(1)} fps > ${this.upFps}`, "up");
    }
  }
  go(level, nowMs, why, kind = "set") {
    const from = this.level; this.level = level; this.samples.length = 0; this.slow = 0;
    this.history.push({ t: nowMs, from, to: level, why, kind, step: this.step });
    this.onStep(this.step, level, why, kind);
  }
  jumpTo(level) {
    this.level = Math.max(0, Math.min(this.ladder.length - 1, level)); this.samples.length = 0; this.slow = 0; this.trial = null; this.verify = null; this.chainTop = null; this.blockFps = null; this.upLimit = 0; this.upLimitUntil = 0;
    this.onStep(this.step, this.level, "set", "set");
  }
}
