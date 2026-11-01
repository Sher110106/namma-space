// Headless walking simulations over the real grid and the real Walker. Used by tests H3; also runnable in Node.
import { Walker } from "./walker.js";

function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }

/** n random walks of `seconds` each from random walkable cells; returns counts of samples outside walkable cells. */
export function randomWalks(grid, cfg, { n = 200, seconds = 5, seed = 1 } = {}) {
  const r = rng(seed);
  const walk = [];
  for (let i = 0; i < grid.h; i++) for (let j = 0; j < grid.w; j++) if (grid.data[i * grid.w + j] === grid.walkableValue) walk.push([i, j]);
  if (!walk.length) return { walks: 0, samples: 0, violations: 0, error: "no walkable cells" };
  let samples = 0, violations = 0, blockedFrames = 0, moved = 0;
  const bad = [];
  for (let w = 0; w < n; w++) {
    const [i, j] = walk[Math.floor(r() * walk.length)];
    const [x, z] = grid.cellCentre(i, j);
    const p = new Walker(grid, cfg);
    p.setPose(x, z, r() * 360);
    let t = 0, nextTurn = 0, input = { forward: 1, strafe: 0, run: false };
    const x0 = p.x, z0 = p.z;
    while (t < seconds) {
      if (t >= nextTurn) {
        p.yaw += (r() - 0.5) * 2.5;
        input = { forward: r() < 0.8 ? 1 : (r() < 0.5 ? -1 : 0), strafe: r() < 0.3 ? (r() < 0.5 ? -1 : 1) : 0, run: r() < 0.3 };
        nextTurn = t + 0.3 + r() * 0.7;
      }
      const dt = 1 / 144 + r() * (1 / 12 - 1 / 144);          // 7 ms .. 83 ms frames
      p.update(dt, input); t += dt; samples++;
      if (p.blockedX || p.blockedZ) blockedFrames++;
      if (grid.value(p.x, p.z) !== grid.walkableValue) { violations++; if (bad.length < 5) bad.push([+p.x.toFixed(3), +p.z.toFixed(3)]); }
    }
    moved += Math.hypot(p.x - x0, p.z - z0);
  }
  return { walks: n, seconds, samples, violations, blockedFrames, meanDisplacement: moved / n, firstViolations: bad };
}

/** From the spawn, hold W for `seconds` at each of `headings` equally spaced yaws. */
export function headingPushes(grid, cfg, spawn, { headings = 16, seconds = 6 } = {}) {
  const out = []; let violations = 0, samples = 0;
  for (let h = 0; h < headings; h++) {
    const p = new Walker(grid, cfg);
    p.setPose(spawn.x, spawn.z, (360 * h) / headings);
    const x0 = p.x, z0 = p.z; let t = 0, sawBlock = false, lastMove = 0;
    const dt = 1 / 60;
    while (t < seconds) {
      const px = p.x, pz = p.z;
      p.update(dt, { forward: 1, strafe: 0, run: false }); t += dt; samples++;
      if (p.blockedX || p.blockedZ) sawBlock = true;
      if (Math.hypot(p.x - px, p.z - pz) > 1e-4) lastMove = t;
      if (grid.value(p.x, p.z) !== grid.walkableValue) violations++;
    }
    const dist = Math.hypot(p.x - x0, p.z - z0);
    out.push({ heading: (360 * h) / headings, dist: +dist.toFixed(2), stoppedByWall: sawBlock && p.speed < 0.05 && dist < 0.95 * cfg.walkMps * seconds, sawBlock, still: seconds - lastMove > 0.3 });
  }
  return { headings, seconds, violations, samples, stopped: out.filter((o) => o.stoppedByWall).length, movedHalfMetre: out.filter((o) => o.dist >= 0.5).length, pushes: out };
}

/** Frame-rate independence: the same 3 s walk at several frame rates ends at nearly the same place in open space. */
export function frameRateIndependence(grid, cfg, spawn, yawDeg) {
  const res = {};
  for (const hz of [20, 30, 60, 144]) {
    const p = new Walker(grid, cfg); p.setPose(spawn.x, spawn.z, yawDeg);
    for (let t = 0; t < 1.0; t += 1 / hz) p.update(1 / hz, { forward: 1, strafe: 0, run: false });
    res[hz] = +Math.hypot(p.x - spawn.x, p.z - spawn.z).toFixed(3);
  }
  return res;
}
