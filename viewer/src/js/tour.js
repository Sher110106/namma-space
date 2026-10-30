// Tour path: Catmull-Rom through the scene's tour keyframes. Deterministic in time (position is a function of t only).

function unwrapDeg(vals) {
  const out = [vals[0]];
  for (let i = 1; i < vals.length; i++) {
    let d = vals[i] - vals[i - 1];
    while (d > 180) d -= 360;
    while (d < -180) d += 360;
    out.push(out[i - 1] + d);
  }
  return out;
}
const cr = (p0, p1, p2, p3, u) => 0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u * u + (-p0 + 3 * p1 - 3 * p2 + p3) * u * u * u);

export class TourPath {
  /** @param {{position:number[],yaw_deg:number,pitch_deg:number,t:number}[]} stops @param {number} maxSecs rescale if longer */
  constructor(stops, maxSecs = 60) {
    this.stops = stops.length >= 2 ? stops : [];
    if (!this.stops.length) { this.duration = 0; return; }
    let ts = this.stops.map((s) => s.t ?? 0);
    const t0 = ts[0]; ts = ts.map((t) => t - t0);
    const total = ts[ts.length - 1];
    if (!(total > 0)) ts = ts.map((_, i) => (i * maxSecs) / (ts.length - 1));
    else if (total > maxSecs) ts = ts.map((t) => (t * maxSecs) / total);
    this.t = ts; this.duration = ts[ts.length - 1];
    this.x = this.stops.map((s) => s.position[0]); this.z = this.stops.map((s) => s.position[2]);
    this.yaw = unwrapDeg(this.stops.map((s) => s.yaw_deg)); this.pitch = this.stops.map((s) => s.pitch_deg || 0);
  }
  /** Scale the timeline so the whole tour lasts `secs`. */
  retime(secs) { if (!this.duration) return; const k = secs / this.duration; this.t = this.t.map((t) => t * k); this.duration = secs; }
  at(time) {
    const n = this.t.length;
    const tt = Math.max(0, Math.min(this.duration, time));
    let i = 0;
    while (i < n - 2 && tt > this.t[i + 1]) i++;
    const u = (tt - this.t[i]) / Math.max(1e-6, this.t[i + 1] - this.t[i]);
    const g = (a, k) => a[Math.max(0, Math.min(n - 1, k))];
    const f = (a) => cr(g(a, i - 1), g(a, i), g(a, i + 1), g(a, i + 2), u);
    return { x: f(this.x), z: f(this.z), yawDeg: f(this.yaw), pitchDeg: f(this.pitch) };
  }
}
