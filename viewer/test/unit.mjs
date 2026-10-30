// Unit tests for the pure modules (no browser): walker, governor, tour. Run: node viewer/test/unit.mjs
import assert from "node:assert/strict";
import { Grid, Walker } from "../src/js/walker.js";
import { Governor, buildLadder } from "../src/js/governor.js";
import { TourPath } from "../src/js/tour.js";
import { randomWalks, headingPushes } from "../src/js/sim.js";

let n = 0;
const t = (name, fn) => { fn(); n++; console.log("ok  ", name); };

// 4 m x 3 m room (5 cm cells), a 0.5 m pillar in the middle, a 1 m door gap in the right wall leading to a 1 m x 1 m closet
function room() {
  const cell = 0.05, w = 120, h = 80, data = new Uint8Array(w * h);
  const set = (x0, z0, x1, z1, v) => { for (let i = Math.round(z0 / cell); i < Math.round(z1 / cell); i++) for (let j = Math.round(x0 / cell); j < Math.round(x1 / cell); j++) data[i * w + j] = v; };
  set(0.5, 0.5, 4.5, 3.5, 255);             // room floor: x 0.5..4.5, z 0.5..3.5
  set(2.2, 1.7, 2.7, 2.2, 0);               // pillar (blocked)
  set(4.5, 1.5, 5.5, 2.5, 255);             // door and closet
  return new Grid({ width: w, height: h, cell, origin_xz: [0, 0] }, data);
}
const cfg = { eyeHeight: 1.4, walkMps: 1.4, runMult: 2 };

t("grid cell convention: pixel (i, j) is centred at origin + (j + .5) cell", () => {
  const g = room(); const [x, z] = g.cellCentre(10, 20);
  assert.ok(Math.abs(x - (20 + 0.5) * 0.05) < 1e-9 && Math.abs(z - (10 + 0.5) * 0.05) < 1e-9);
  assert.equal(g.col(x), 20); assert.equal(g.row(z), 10);
});
t("outside the grid counts as blocked", () => { const g = room(); assert.equal(g.value(-1, 1), 0); assert.equal(g.value(99, 99), 0); });
t("yaw 0 looks along -Z, positive yaw turns left (towards -X), strafe right is +X at yaw 0", () => {
  const w = new Walker(null, cfg); w.setPose(0, 0, 0);
  for (let i = 0; i < 120; i++) w.update(1 / 60, { forward: 1, strafe: 0, run: false });
  assert.ok(w.z < -1 && Math.abs(w.x) < 1e-9, `z ${w.z} x ${w.x}`);
  const l = new Walker(null, cfg); l.setPose(0, 0, 90);
  for (let i = 0; i < 120; i++) l.update(1 / 60, { forward: 1, strafe: 0, run: false });
  assert.ok(l.x < -1 && Math.abs(l.z) < 1e-9, `x ${l.x}`);
  const r = new Walker(null, cfg); r.setPose(0, 0, 0);
  for (let i = 0; i < 120; i++) r.update(1 / 60, { forward: 0, strafe: 1, run: false });
  assert.ok(r.x > 1, `strafe x ${r.x}`);
});
t("speed: 1.4 m/s walking, 2.8 m/s running, after the smoothing ramp", () => {
  for (const [run, v] of [[false, 1.4], [true, 2.8]]) {
    const w = new Walker(null, cfg); w.setPose(0, 0, 0);
    for (let i = 0; i < 300; i++) w.update(1 / 60, { forward: 1, strafe: 0, run });   // 5 s
    const z0 = w.z; for (let i = 0; i < 60; i++) w.update(1 / 60, { forward: 1, strafe: 0, run });
    assert.ok(Math.abs(Math.abs(w.z - z0) - v) < 0.02, `${run}: ${Math.abs(w.z - z0)}`);
  }
});
t("diagonal input is not faster than straight", () => {
  const w = new Walker(null, cfg); w.setPose(0, 0, 0);
  for (let i = 0; i < 300; i++) w.update(1 / 60, { forward: 1, strafe: 1, run: false });
  assert.ok(w.speed < 1.41, `speed ${w.speed}`);
});
t("frame-rate independence: 1 s of walking covers the same distance at 20, 60 and 144 Hz", () => {
  const d = [20, 60, 144].map((hz) => { const w = new Walker(null, cfg); w.setPose(0, 0, 0); for (let i = 0; i < hz; i++) w.update(1 / hz, { forward: 1, strafe: 0, run: false }); return -w.z; });
  assert.ok(Math.max(...d) - Math.min(...d) < 0.03, d.join(", "));
});
t("walls stop the player; position never leaves walkable cells (one big step too)", () => {
  const g = room(); const w = new Walker(g, cfg); w.setPose(1, 1, 90);          // facing -X, wall at x = 0.5
  for (let i = 0; i < 300; i++) { w.update(1 / 60, { forward: 1, strafe: 0, run: true }); assert.ok(g.walkable(w.x, w.z)); }
  assert.ok(w.x >= 0.5 && w.x < 0.6, `x ${w.x}`);
  w.setPose(1, 1, 90); w.update(0.1, { forward: 1, strafe: 0, run: true }); w.update(0.1, { forward: 1, strafe: 0, run: true });
  assert.ok(g.walkable(w.x, w.z));
});
t("sliding: pushing diagonally into a wall keeps the parallel motion", () => {
  const g = room(); const w = new Walker(g, cfg); w.setPose(1, 1, 45);            // forward = (-sin45, -cos45): towards -X and -Z
  for (let i = 0; i < 400; i++) w.update(1 / 60, { forward: 1, strafe: 0, run: false });
  assert.ok(w.x < 0.6 && w.z < 0.6, `corner ${w.x}, ${w.z}`);                   // ends in the corner
  const w2 = new Walker(g, cfg); w2.setPose(3, 1, 80);                           // mostly -X, slightly -Z: slides along the north wall? no, hits x wall later
  for (let i = 0; i < 300; i++) w2.update(1 / 60, { forward: 1, strafe: 0, run: false });
  assert.ok(g.walkable(w2.x, w2.z));
  const w3 = new Walker(g, cfg); w3.setPose(1, 2, 120);                          // towards -X and +Z (yaw 120: forward = (-0.87, +0.5))
  for (let i = 0; i < 200; i++) w3.update(1 / 60, { forward: 1, strafe: 0, run: false });
  assert.ok(w3.x < 0.6 && w3.z > 2.4, `slid along the west wall: ${w3.x}, ${w3.z}`);
});
t("the pillar blocks and the door gap lets the player through", () => {
  const g = room(); const w = new Walker(g, cfg); w.setPose(1.5, 1.95, 270);       // facing +X at pillar height
  for (let i = 0; i < 300; i++) w.update(1 / 60, { forward: 1, strafe: 0, run: false });
  assert.ok(w.x < 2.2, `stopped at the pillar: ${w.x}`);
  const d = new Walker(g, cfg); d.setPose(1.5, 2.0, 270);                          // z = 2.0 is outside the pillar's z span? 1.7..2.2 -> inside; move to 2.4
  d.setPose(1.5, 2.4, 270);
  for (let i = 0; i < 400; i++) d.update(1 / 60, { forward: 1, strafe: 0, run: false });
  assert.ok(d.x > 4.6, `through the door: ${d.x}`);
});
t("teleport and spawn push to the nearest walkable cell", () => {
  const g = room(); const w = new Walker(g, cfg);
  w.setPose(2.45, 1.95, 0);                                                         // inside the pillar
  assert.ok(g.walkable(w.x, w.z) && Math.hypot(w.x - 2.45, w.z - 1.95) < 0.4);
  assert.ok(w.teleport(9, 9)); assert.ok(g.walkable(w.x, w.z));
});
t("pitch is limited to +-80 degrees", () => { const w = new Walker(null, cfg); w.look(0, 10); assert.ok(Math.abs(w.pitchDeg - 80) < 1e-9); w.look(0, -10); assert.ok(Math.abs(w.pitchDeg + 80) < 1e-9); });
t("random walks and headings leave no sample in a blocked cell", () => {
  const g = room(); const r = randomWalks(g, cfg, { n: 200, seconds: 5, seed: 3 });
  assert.equal(r.violations, 0); assert.ok(r.blockedFrames > 0);
  const p = headingPushes(g, cfg, { x: 1.5, z: 1.0 }, { headings: 16, seconds: 6 });
  assert.equal(p.violations, 0); assert.ok(p.stopped >= 8 && p.movedHalfMetre >= 12, JSON.stringify([p.stopped, p.movedHalfMetre]));
});
t("fly mode ignores furniture but is clamped to the bounds", () => {
  const g = room(); const w = new Walker(g, cfg); w.fly = true; w.bounds = { min: [0.2, 0.2], max: [4, 3] };
  w.setPose(1.5, 1.95, 270);
  for (let i = 0; i < 600; i++) w.update(1 / 60, { forward: 1, strafe: 0, run: true });
  assert.ok(w.x <= 4 + 1e-9 && w.x > 3.9, `x ${w.x}`);
});

t("governor: steps down after 2 s under 40 fps, up after 2 s over 55, holds off after a step down", () => {
  const steps = []; const g = new Governor({ ladder: buildLadder({ previewFallback: true }), onStep: (s) => steps.push(s.name) });
  let now = 0;
  for (; now < 2200; now += 30) g.push(30, now);                                     // 33 fps
  assert.equal(steps[0], "low");
  for (; now < 4500; now += 30) g.push(30, now);
  assert.equal(steps[1], "min");
  for (; now < 9000; now += 16.7) g.push(16.7, now);                                 // 60 fps but inside the 8 s hold-off
  assert.equal(steps.length, 2);
  for (; now < 14000; now += 16.7) g.push(16.7, now);
  assert.equal(steps[2], "low");                                                     // steps back up
});
t("governor: two frames over 400 ms step down at once (software rendering)", () => {
  const steps = []; const g = new Governor({ ladder: buildLadder({ previewFallback: true }), onStep: (s) => steps.push(s.name) });
  g.push(900, 1000); assert.equal(steps.length, 0); g.push(900, 1900); assert.equal(steps[0], "low");
  g.push(900, 2800); g.push(900, 3700); assert.equal(steps[1], "min");
});
t("governor ladder: SD starts at sqrt(5), HD at the scene value, stress at pixel ratio 2", () => {
  const sd = buildLadder({}), hd = buildLadder({ hd: true }), st = buildLadder({ stress: true });
  assert.ok(Math.abs(sd[0].maxStdDev - Math.sqrt(5)) < 1e-9 && sd[0].pixelRatio === 1);
  assert.ok(Math.abs(hd[0].maxStdDev - Math.sqrt(8)) < 1e-9);
  assert.equal(st[0].pixelRatio, 2); assert.ok(st[1].pixelRatio < 2);
  assert.equal(buildLadder({ basePixelRatio: 0.85 })[0].pixelRatio, 0.85);
});
t("tour: Catmull-Rom passes through its stops, is deterministic, rescales to the cap", () => {
  const stops = [0, 1, 2, 3].map((i) => ({ position: [i, 1.4, i * i], yaw_deg: i * 100, pitch_deg: 0, t: i * 40 }));
  const p = new TourPath(stops, 60);
  assert.equal(p.duration, 60);
  const a = p.at(20); assert.ok(Math.abs(a.x - 1) < 1e-9 && Math.abs(a.z - 1) < 1e-9, JSON.stringify(a));
  assert.deepEqual(p.at(33.3), p.at(33.3));
  const q = new TourPath([{ position: [0, 1, 0], yaw_deg: 170, t: 0 }, { position: [1, 1, 0], yaw_deg: -170, t: 5 }]); // shortest arc across +-180
  assert.ok(Math.abs(q.at(2.5).yawDeg) > 170 || Math.abs(q.at(2.5).yawDeg - 360) < 20, String(q.at(2.5).yawDeg));
  assert.equal(new TourPath([], 60).duration, 0);
});
console.log(`${n} unit tests passed`);
