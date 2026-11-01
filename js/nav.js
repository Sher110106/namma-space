// Route finding on the walkable grid. Pure logic: no DOM, no three.js, so Node can import it.
//
// Frame as in walker.js: metres, +Y up, grid pixel (row i, col j) is the cell centred at x = ox + (j + .5) cell, z = oz + (i + .5) cell.
// A* runs on 8-connected walkable cells with octile distance; a diagonal step needs both side cells walkable.
// String pulling then drops waypoints that a straight line of sight can skip, keeping a little clearance from walls.

const SQRT2 = Math.SQRT2;
const NB = [[0, -1, 1], [0, 1, 1], [-1, 0, 1], [1, 0, 1], [-1, -1, SQRT2], [-1, 1, SQRT2], [1, -1, SQRT2], [1, 1, SQRT2]];   // di, dj, step cost in cells
const CLEARANCE_M = 0.10;        // string pulling keeps this far from blocked cells when it can
const PEN = new Float32Array(256);   // extra A* cost for a cell by its distance (in cells) to the nearest blocked cell: nudges routes off walls
[0, 0.15, 0.08, 0.03].forEach((v, k) => (PEN[k] = v));

const walk = (g, i, j) => i >= 0 && j >= 0 && i < g.h && j < g.w && g.data[i * g.w + j] === g.walkableValue;

// ---------------------------------------------------------------------------------------------------------------- per-grid caches
const CACHE = new WeakMap();
/** Connected-component labels (0 = not walkable) and the distance in cells to the nearest blocked cell (Chebyshev, capped at 255). */
export function gridInfo(g) {
  let c = CACHE.get(g);
  if (c && c.data === g.data) return c;
  const { w, h } = g, N = w * h;
  const comp = new Int32Array(N); let nComp = 0; const stack = new Int32Array(N);
  for (let s = 0; s < N; s++) {
    if (comp[s] || g.data[s] !== g.walkableValue) continue;
    nComp++; let sp = 0; stack[sp++] = s; comp[s] = nComp;
    while (sp) {
      const u = stack[--sp], ui = (u / w) | 0, uj = u % w;
      for (const [di, dj] of NB) {
        const ni = ui + di, nj = uj + dj;
        if (!walk(g, ni, nj) || (di && dj && !(walk(g, ui + di, uj) && walk(g, ui, uj + dj)))) continue;
        const n = ni * w + nj;
        if (!comp[n]) { comp[n] = nComp; stack[sp++] = n; }
      }
    }
  }
  const clear = new Uint8Array(N);
  for (let i = 0; i < h; i++) for (let j = 0; j < w; j++) {          // forward pass
    const k = i * w + j;
    if (g.data[k] !== g.walkableValue) { clear[k] = 0; continue; }
    let d = (i === 0 || j === 0 || i === h - 1 || j === w - 1) ? 1 : 255;
    if (i > 0) { d = Math.min(d, clear[k - w] + 1); if (j > 0) d = Math.min(d, clear[k - w - 1] + 1); if (j < w - 1) d = Math.min(d, clear[k - w + 1] + 1); }
    if (j > 0) d = Math.min(d, clear[k - 1] + 1);
    clear[k] = Math.min(255, d);
  }
  for (let i = h - 1; i >= 0; i--) for (let j = w - 1; j >= 0; j--) {   // backward pass
    const k = i * w + j; if (!clear[k]) continue;
    let d = clear[k];
    if (i < h - 1) { d = Math.min(d, clear[k + w] + 1); if (j > 0) d = Math.min(d, clear[k + w - 1] + 1); if (j < w - 1) d = Math.min(d, clear[k + w + 1] + 1); }
    if (j < w - 1) d = Math.min(d, clear[k + 1] + 1);
    clear[k] = d;
  }
  c = { data: g.data, comp, nComp, clear };
  CACHE.set(g, c);
  return c;
}

// ---------------------------------------------------------------------------------------------------------------- A*
class MinHeap {
  constructor() { this.k = []; this.v = []; }
  get size() { return this.k.length; }
  push(key, val) {
    let i = this.k.length; this.k.push(key); this.v.push(val);
    while (i > 0) { const p = (i - 1) >> 1; if (this.k[p] <= key) break; this.k[i] = this.k[p]; this.v[i] = this.v[p]; i = p; }
    this.k[i] = key; this.v[i] = val;
  }
  pop() {
    const top = this.v[0], lk = this.k.pop(), lv = this.v.pop(), n = this.k.length;
    if (n > 0) {
      let i = 0;
      for (;;) { let c = 2 * i + 1; if (c >= n) break; if (c + 1 < n && this.k[c + 1] < this.k[c]) c++; if (this.k[c] >= lk) break; this.k[i] = this.k[c]; this.v[i] = this.v[c]; i = c; }
      this.k[i] = lk; this.v[i] = lv;
    }
    return top;
  }
}

/**
 * A* between two cell indices (row * width + col). Returns the cell indices from start to goal, or null.
 * @param {{penalty?:boolean, minClear?:number}} opt  penalty: add the wall-avoidance cost (default true); off gives the plain octile shortest path.
 *   minClear: only use cells at least this many cells from a blocked cell (1 = every walkable cell), except within 3 cells of the start and goal.
 */
export function astar(g, s, t, { penalty = true, minClear = 1 } = {}) {
  const { w, h } = g, N = w * h;
  if (s === t) return [s];
  const { clear } = gridInfo(g);
  const gs = new Float32Array(N).fill(Infinity), prev = new Int32Array(N).fill(-1), closed = new Uint8Array(N);
  const ti = (t / w) | 0, tj = t % w, si = (s / w) | 0, sj = s % w, EX = 3;
  const ok = (ni, nj, n) => clear[n] >= minClear || (Math.abs(ni - si) <= EX && Math.abs(nj - sj) <= EX) || (Math.abs(ni - ti) <= EX && Math.abs(nj - tj) <= EX);
  const hf = (idx) => { const dx = Math.abs((idx % w) - tj), dz = Math.abs(((idx / w) | 0) - ti); return dx + dz + (SQRT2 - 2) * Math.min(dx, dz); };   // octile distance
  const heap = new MinHeap(); gs[s] = 0; heap.push(hf(s), s);
  while (heap.size) {
    const u = heap.pop();
    if (closed[u]) continue;
    closed[u] = 1;
    if (u === t) break;
    const ui = (u / w) | 0, uj = u % w;
    for (const [di, dj, c] of NB) {
      const ni = ui + di, nj = uj + dj;
      if (!walk(g, ni, nj)) continue;
      if (di && dj && !(walk(g, ui + di, uj) && walk(g, ui, uj + dj))) continue;      // no cutting a blocked corner
      const n = ni * w + nj;
      if (closed[n] || !ok(ni, nj, n)) continue;
      const ng = gs[u] + c * (penalty ? 1 + PEN[clear[n]] : 1);
      if (ng < gs[n]) { gs[n] = ng; prev[n] = u; heap.push(ng + hf(n), n); }
    }
  }
  if (!closed[t]) return null;
  const out = [];
  for (let u = t; u !== -1; u = prev[u]) out.push(u);
  return out.reverse();
}

// ---------------------------------------------------------------------------------------------------------------- line of sight and string pulling
/** Exact grid walk (all cells the segment touches, corners included): true when every one is walkable. */
export function segmentClear(g, x0, z0, x1, z1) {
  const u0 = (x0 - g.ox) / g.cell, v0 = (z0 - g.oz) / g.cell, u1 = (x1 - g.ox) / g.cell, v1 = (z1 - g.oz) / g.cell;
  let i = Math.floor(v0), j = Math.floor(u0);
  const iEnd = Math.floor(v1), jEnd = Math.floor(u1);
  if (!walk(g, i, j)) return false;
  const du = u1 - u0, dv = v1 - v0, sj = du > 0 ? 1 : -1, si = dv > 0 ? 1 : -1;
  const tdu = du !== 0 ? Math.abs(1 / du) : Infinity, tdv = dv !== 0 ? Math.abs(1 / dv) : Infinity;
  let tu = du > 0 ? (j + 1 - u0) / du : du < 0 ? (u0 - j) / -du : Infinity;
  let tv = dv > 0 ? (i + 1 - v0) / dv : dv < 0 ? (v0 - i) / -dv : Infinity;
  for (let guard = 0; (i !== iEnd || j !== jEnd) && guard < 100000; guard++) {
    if (Math.abs(tu - tv) < 1e-12) {                                 // the segment passes exactly through a cell corner: both side cells count
      if (!walk(g, i, j + sj) || !walk(g, i + si, j)) return false;
      j += sj; i += si; tu += tdu; tv += tdv;
    } else if (tu < tv) { j += sj; tu += tdu; } else { i += si; tv += tdv; }
    if (!walk(g, i, j)) return false;
  }
  return true;
}
/** Line of sight from a to b ([x, z]) with `r` metres of side clearance (0 = the centre line only). */
export function lineOfSight(g, a, b, r = 0) {
  if (!segmentClear(g, a[0], a[1], b[0], b[1])) return false;
  if (r <= 0) return true;
  const dx = b[0] - a[0], dz = b[1] - a[1], d = Math.hypot(dx, dz);
  if (d < 1e-9) return true;
  const nx = (-dz / d) * r, nz = (dx / d) * r;
  return segmentClear(g, a[0] + nx, a[1] + nz, b[0] + nx, b[1] + nz) && segmentClear(g, a[0] - nx, a[1] - nz, b[0] - nx, b[1] - nz);
}
/** Greedy string pulling: from each anchor jump to the farthest waypoint that can be seen with clearance, then with less. */
export function smoothPath(g, pts, clearance = CLEARANCE_M) {
  const out = [pts[0]], n = pts.length; let a = 0;
  while (a < n - 1) {
    let b = -1;
    for (const r of [clearance, clearance / 2, 0]) {
      for (let k = n - 1; k > a + 1; k--) if (lineOfSight(g, pts[a], pts[k], r)) { b = k; break; }
      if (b > 0) break;
    }
    if (b < 0) b = a + 1;
    out.push(pts[b]); a = b;
  }
  return out;
}

// ---------------------------------------------------------------------------------------------------------------- polylines
export function polylineLength(pts) { let L = 0; for (let k = 1; k < pts.length; k++) L += Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]); return L; }
/** Cumulative arclength at each vertex. */
export function arclengths(pts) { const s = [0]; for (let k = 1; k < pts.length; k++) s.push(s[k - 1] + Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1])); return s; }
/** Closest point on the polyline: {dist, s (arclength of that point), seg}. `fromS` restricts the search to arclength >= fromS - slack (follow mode never snaps backwards). */
export function projectOnPolyline(pts, x, z, fromS = 0, slack = 0.5) {
  let best = { dist: Infinity, s: 0, seg: 0 }, acc = 0;
  for (let k = 1; k < pts.length; k++) {
    const ax = pts[k - 1][0], az = pts[k - 1][1], bx = pts[k][0], bz = pts[k][1], dx = bx - ax, dz = bz - az, L = Math.hypot(dx, dz);
    if (acc + L >= fromS - slack && L > 0) {
      const t = Math.max(0, Math.min(1, ((x - ax) * dx + (z - az) * dz) / (L * L)));
      const d = Math.hypot(x - (ax + t * dx), z - (az + t * dz));
      if (d < best.dist) best = { dist: d, s: acc + t * L, seg: k - 1 };
    }
    acc += L;
  }
  if (best.dist === Infinity && pts.length) best = { dist: Math.hypot(x - pts[pts.length - 1][0], z - pts[pts.length - 1][1]), s: acc, seg: pts.length - 2 };
  return best;
}
export function pointAtS(pts, s) {
  if (pts.length === 1) return pts[0].slice();
  let acc = 0;
  for (let k = 1; k < pts.length; k++) {
    const L = Math.hypot(pts[k][0] - pts[k - 1][0], pts[k][1] - pts[k - 1][1]);
    if (acc + L >= s || k === pts.length - 1) { const t = L > 0 ? Math.max(0, Math.min(1, (s - acc) / L)) : 1; return [pts[k - 1][0] + t * (pts[k][0] - pts[k - 1][0]), pts[k - 1][1] + t * (pts[k][1] - pts[k - 1][1])]; }
    acc += L;
  }
  return pts[pts.length - 1].slice();
}
/** Sample a polyline every `step` metres (end point included). */
export function samplePolyline(pts, step = 0.05) {
  const out = [], total = polylineLength(pts);
  for (let s = 0; s < total; s += step) out.push(pointAtS(pts, s));
  out.push(pts[pts.length - 1].slice());
  return out;
}

// ---------------------------------------------------------------------------------------------------------------- stand spots and routes
const cellIdxAt = (g, x, z) => { const i = g.row(z), j = g.col(x); return i < 0 || j < 0 || i >= g.h || j >= g.w ? -1 : i * g.w + j; };
export const cellCentreOf = (g, idx) => g.cellCentre((idx / g.w) | 0, idx % g.w);

/**
 * The walkable cell centre nearest to (x, z) that is at least `minDist` away from it (items: stand back, do not stand inside the thing),
 * optionally limited to one connected component. Returns {x, z, d, idx} or null.
 */
export function standSpot(g, x, z, { minDist = 0, comp = 0 } = {}) {
  const info = gridInfo(g);
  const rj = Math.ceil((minDist + 8) / g.cell);
  let best = null;
  const scan = (i0, i1, j0, j1) => {
    for (let i = Math.max(0, i0); i <= Math.min(g.h - 1, i1); i++) for (let j = Math.max(0, j0); j <= Math.min(g.w - 1, j1); j++) {
      const k = i * g.w + j;
      if (!info.comp[k] || (comp && info.comp[k] !== comp)) continue;
      const cx = g.ox + (j + 0.5) * g.cell, cz = g.oz + (i + 0.5) * g.cell, d = Math.hypot(cx - x, cz - z);
      if (d < minDist - 1e-9) continue;
      if (!best || d < best.d - 1e-12) best = { x: cx, z: cz, d, idx: k };
    }
  };
  scan(g.row(z) - rj, g.row(z) + rj, g.col(x) - rj, g.col(x) + rj);           // window of minDist + 8 m first
  if (!best || best.d > (rj - 1) * g.cell) scan(0, g.h - 1, 0, g.w - 1);   // nothing near, or the window may have clipped the true nearest
  if (!best && minDist > 0) return standSpot(g, x, z, { minDist: 0, comp });  // nothing far enough: take the nearest
  return best;
}
/** How far back from an item the player stands: its radius plus 0.6 m, 0.8-2.0 m. Rooms and zones: the nearest walkable cell. */
export const standBackFor = (place) => (place.kind === "item" ? Math.max(0.8, Math.min(2.0, (place.radius_m || 0.4) + 0.6)) : 0);

/**
 * Walking route from `from` {x, z} to a place. Returns
 *   {ok:true, points:[[x,z]...], length, seconds, stand:{x,z}, note?}  or  {ok:false, reason}.
 * The route ends at the stand spot (the walkable cell nearest the place), not inside the place.
 */
export function routeToPlace(g, from, place, { walkMps = 1.4, penalty = true, clearance = CLEARANCE_M } = {}) {
  const info = gridInfo(g);
  if (!info.nComp) return { ok: false, reason: "This scene has no walkable floor." };
  const [px, , pz] = place.position;
  let sIdx = cellIdxAt(g, from.x, from.z);
  if (sIdx < 0 || !info.comp[sIdx]) { const n = g.nearestWalkable(from.x, from.z); if (!n) return { ok: false, reason: "You stand outside the walkable floor." }; sIdx = cellIdxAt(g, n[0], n[1]); }
  const comp = info.comp[sIdx], back = standBackFor(place);
  const ideal = standSpot(g, px, pz, { minDist: back });
  let spot = ideal, note;
  if (info.comp[ideal.idx] !== comp) {
    const alt = standSpot(g, px, pz, { minDist: back, comp });
    if (!alt || alt.d > ideal.d + 1.5) return { ok: false, reason: `No walking route: ${place.name} is in an area you cannot reach on foot.` };
    spot = alt; note = `Closest spot you can walk to is ${alt.d.toFixed(1)} m from ${place.name}.`;
  }
  // keep 12 cm from walls when that adds under 3% to the smoothed length, else 7 cm, else hug them (narrow doors)
  const start = info.comp[cellIdxAt(g, from.x, from.z)] ? [from.x, from.z] : cellCentreOf(g, sIdx);
  const build = (cells) => {
    const raw = [start, ...cells.map((c) => cellCentreOf(g, c))];
    const sm = smoothPath(g, raw, clearance);
    return pathWalkable(g, sm) ? sm : raw;                                       // never ship a smoothed path that fails the check
  };
  const plain = astar(g, sIdx, spot.idx, { penalty });
  if (!plain) return { ok: false, reason: `No walking route to ${place.name}.` };
  let pts = build(plain);
  const baseLen = polylineLength(pts);
  for (const minClear of [3, 2]) {
    const c = astar(g, sIdx, spot.idx, { penalty, minClear });
    if (!c) continue;
    const q = build(c);
    if (polylineLength(q) <= 1.03 * baseLen + 0.03) { pts = q; break; }
  }
  pts = pts.filter((p, k) => k === 0 || Math.hypot(p[0] - pts[k - 1][0], p[1] - pts[k - 1][1]) > 1e-6);
  if (pts.length === 1) pts.push(pts[0].slice());
  const length = polylineLength(pts);
  return { ok: true, points: pts, length, seconds: length / walkMps, stand: { x: spot.x, z: spot.z }, note };
}
export function pathWalkable(g, pts, step = 0.02) {
  for (const [x, z] of samplePolyline(pts, step)) if (!g.walkable(x, z)) return false;
  return true;
}

/**
 * One follow-mode steering step. Looks `ahead` metres along the route from the player's projection.
 * Returns {s, total, remaining, target:[x,z], off} (off: metres from the route).
 */
export function steer(pts, x, z, prevS = 0, ahead = 0.6) {
  const total = polylineLength(pts), pr = projectOnPolyline(pts, x, z, prevS, 0.3);
  const s = Math.max(prevS, pr.s);
  return { s, total, remaining: total - s, target: pointAtS(pts, Math.min(total, s + ahead)), off: pr.dist };
}

const wrapPi = (a) => ((((a + Math.PI) % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI)) - Math.PI;
/**
 * One step of follow mode: turns `w.yaw` (radians, same convention as Walker) toward the route and returns the Walker input.
 * st = {s, off, remaining, arrived, done}; the caller then runs w.update(dt, input). After arrival it turns to `faceYaw` (radians) and sets st.done.
 */
export function followStep(pts, w, st, dt, { ahead = 0.6, turnRate = 3.6, arrive = 0.25, faceYaw = null } = {}) {
  const r = steer(pts, w.x, w.z, st.s || 0, ahead);
  st.s = r.s; st.off = r.off; st.remaining = r.remaining;
  const turn = (to) => { const d = wrapPi(to - w.yaw), m = Math.max(-turnRate * dt, Math.min(turnRate * dt, d)); w.yaw += m; return d - m; };
  if (r.remaining <= arrive) {
    st.arrived = true;
    st.done = faceYaw == null || Math.abs(turn(faceYaw)) < 0.02;
    return { forward: 0, strafe: 0, run: false };
  }
  let target = r.target;
  if (w.grid) {         // aim at the farthest route point ahead that the player can walk to in a straight line: no corner cutting into a wall
    target = null;
    for (let a = ahead; a > 0.05 && !target; a -= 0.1) { const c = pointAtS(pts, Math.min(r.total, r.s + a)); if (lineOfSight(w.grid, [w.x, w.z], c, 0.04)) target = c; }
    if (!target) target = pointAtS(pts, Math.min(r.total, r.s + 0.1));
  }
  const dx = target[0] - w.x, dz = target[1] - w.z;
  const left = Math.hypot(dx, dz) > 1e-4 ? turn(Math.atan2(-dx, -dz)) : 0;
  const forward = Math.max(0, Math.cos(left)) * Math.max(0.35, Math.min(1, r.remaining / 0.8));
  return { forward, strafe: 0, run: false };
}
