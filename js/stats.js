// Frame statistics: fps over 1 s, p95 frame time over the last ~5 s, plus the whole-run aggregate used by the benchmark.

export class FrameStats {
  constructor() { this.dts = []; this.win = []; this.frames = 0; this.accT = 0; this.accN = 0; this.fps = 0; this.last = null; this.runDts = null; }
  /** now: ms (rAF timestamp). Returns dt ms of this frame (0 for the first). */
  tick(now) {
    if (this.last === null) { this.last = now; return 0; }
    const dt = now - this.last; this.last = now;
    this.frames++;
    this.win.push([now, dt]);
    const cut = now - 5000;
    while (this.win.length && this.win[0][0] < cut) this.win.shift();
    this.accT += dt; this.accN++;
    if (this.accT >= 1000) { this.fps = (this.accN * 1000) / this.accT; this.accT = 0; this.accN = 0; }
    if (this.runDts) this.runDts.push(dt);
    return dt;
  }
  get p95() { return percentile(this.win.map((w) => w[1]), 0.95); }
  startRun() { this.runDts = []; }
  endRun() {
    const d = this.runDts || []; this.runDts = null;
    if (!d.length) return null;
    const total = d.reduce((a, b) => a + b, 0);
    return { frames: d.length, secs: total / 1000, fps_mean: (d.length * 1000) / total, p50_ms: percentile(d, 0.5), p95_ms: percentile(d, 0.95), p99_ms: percentile(d, 0.99), max_ms: Math.max(...d) };
  }
}

export function percentile(arr, q) {
  if (!arr.length) return 0;
  const a = [...arr].sort((x, y) => x - y);
  return a[Math.min(a.length - 1, Math.floor(a.length * q))];
}
