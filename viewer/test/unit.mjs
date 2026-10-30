// Unit tests for the pure modules (no browser): walker, governor, tour. Run: node viewer/test/unit.mjs
import assert from "node:assert/strict";
import { Grid, Walker } from "../src/js/walker.js";
import { Governor, buildLadder } from "../src/js/governor.js";
import { TourPath } from "../src/js/tour.js";
import { randomWalks, headingPushes } from "../src/js/sim.js";
import { scaleInfo, hdSameAsSd, clearColorOf, DEFAULT_CLEAR } from "../src/js/scene.js";
import { normalisePlace, placesToJson } from "../src/js/places.js";

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

// --- governor -----------------------------------------------------------------------------------------------------------------
// Run the governor against a model machine. frameMs(step, now) is the frame time the machine gives at that step.
function run({ ladder, frameMs, seconds, start = 0, events = {} }) {
  const kinds = [], confirmed = []; let step;
  const g = new Governor({ ladder, start, onStep: (s, l, why, kind) => { kinds.push(`${s.name}:${kind}`); step = s; }, onConfirm: (s) => confirmed.push(s.name) });
  let now = 0;
  while (now < seconds * 1000) {
    if (events[Math.round(now)]) events[Math.round(now)](g);
    const dt = frameMs(g.step, now); now += dt; g.push(dt, now);
  }
  return { g, kinds, confirmed, final: g.step.name, applied: g.history.map((h) => h.step.name) };
}
// A GPU-bound machine: work grows with the pixel count; frames are quantised to the display refresh.
const gpuBound = (workMs, hz = 60) => (s) => { const q = 1000 / hz; return Math.max(q, Math.ceil((workMs * s.pixelRatio * s.pixelRatio * (s.mesh === "preview" ? 0.2 : 1)) / q) * q); };

t("governor: steps down after 2 s under 40 fps when the step helps, and up after 2 s over 55 once the 8 s hold-off ends", () => {
  const ladder = buildLadder({ previewFallback: true });
  const r = run({ ladder, frameMs: (s, now) => (now < 9000 ? gpuBound(45)(s) : 16.7), seconds: 28 });
  assert.equal(r.applied[0], "low");                                                 // first step after about 1.9 s
  assert.ok(r.g.history[0].t > 1800 && r.g.history[0].t < 2300, String(r.g.history[0].t));
  assert.equal(r.g.history[0].kind, "down");
  assert.ok(r.confirmed.includes("low"));                                            // 20 fps -> 30 fps is a gain
  assert.equal(r.final, "standard");                                                 // the load stops at 9 s: back up after the hold-off
  assert.ok(r.g.history.some((h) => h.kind === "up"));
});
t("governor, capped run: a steady 30 fps (battery saver, vsync half rate) goes back to standard and stays there; never the preview", () => {
  for (const ms of [33.3, 1000 / 38, 1000 / 35]) {
    const r = run({ ladder: buildLadder({ previewFallback: true }), frameMs: () => ms, seconds: 40 });
    assert.equal(r.final, "standard", `${ms}: ${r.applied.join(",")}`);
    assert.ok(!r.applied.includes("min") && !r.applied.includes("preview"), `${ms}: ${r.applied.join(",")}`);
    assert.deepEqual(r.applied, ["low", "standard"], `${ms}`);                         // one trial step, then back
    assert.equal(r.confirmed.length, 0);
    assert.ok(r.g.history[1].kind === "revert" && /no gain/.test(r.g.history[1].why));
  }
});
t("governor, capped run with a slow start (a loading hiccup at 24 fps, then 30 fps): the first step looks like a gain, the next shows the cap, and the top of the run is measured again and kept", () => {
  for (const slowUntil of [2600, 3500, 5000]) {
    const r = run({ ladder: buildLadder({ previewFallback: true }), frameMs: (s, now) => (now < slowUntil ? 41.7 : 33.3), seconds: 40 });
    assert.equal(r.final, "standard", `${slowUntil}: ${r.applied.join(",")}`);
    assert.ok(!r.applied.includes("preview"), r.applied.join(","));
    assert.ok(r.g.history.every((h) => h.kind !== "up" || h.t > 0));
    assert.equal(r.g.chainTop, null);
  }
  // and the same shape on a machine that really is GPU-bound: the steps help, so the top is not restored
  const g = run({ ladder: buildLadder({ previewFallback: true }), frameMs: gpuBound(60, 60), seconds: 40 });
  assert.ok(g.g.level >= 1, g.applied.join(","));
});
t("governor, capped run: after the trial the cap does not trigger a second step down, but a real drop to 20 fps does", () => {
  const r = run({ ladder: buildLadder({ previewFallback: true }), frameMs: (s, now) => (now < 20000 ? 33.3 : gpuBound(55)(s)), seconds: 40 });
  assert.deepEqual(r.applied.slice(0, 2), ["low", "standard"]);
  assert.ok(r.applied.length > 2, r.applied.join(","));                              // load rose at 20 s: stepped down again (the step helps)
  assert.ok(r.g.history[2].t > 20000);
});
t("governor, stressed run: a GPU-bound machine keeps stepping down while each step raises the frame rate, and settles where more does not help", () => {
  const ladder = buildLadder({ previewFallback: true });
  const r = run({ ladder, frameMs: gpuBound(52, 144), seconds: 30 });              // 52 ms of work at pixel ratio 1; 144 Hz display
  assert.ok(r.confirmed.length >= 2, r.confirmed.join(","));
  assert.ok(r.applied.includes("min"));
  assert.ok(r.g.level >= 2);
  // the stress preset as in test H6: pixel ratio 2 is the problem, 1.4 fixes it
  const st = run({ ladder: buildLadder({ stress: true }), frameMs: (s) => (s.pixelRatio >= 2 ? 40 : 16.7), seconds: 8 });
  assert.equal(st.applied[0], "stress-mid"); assert.ok(st.g.history[0].t < 2300); assert.deepEqual(st.confirmed, ["stress-mid"]);
});
t("governor, hidden tab: reset() on return, and a single huge frame is not a measurement", () => {
  const ladder = buildLadder({ previewFallback: true });
  // 60 fps, a 30 s pause, 60 fps again. main.js calls reset() on visibilitychange and passes no dt for the first frame back.
  let r = run({ ladder, frameMs: () => 16.7, seconds: 3 });
  const g = r.g; let now = 3000;
  now += 30000; g.reset(); for (let i = 0; i < 300; i++) { now += 16.7; g.push(16.7, now); }
  assert.equal(g.history.length, 0);
  // no reset (a browser that sends no event): the 30 s frame is dropped as a pause
  const g2 = new Governor({ ladder, onStep: () => {} }); now = 0;
  for (let i = 0; i < 120; i++) { now += 16.7; g2.push(16.7, now); }
  now += 30000; g2.push(30000, now);
  for (let i = 0; i < 300; i++) { now += 16.7; g2.push(16.7, now); }
  assert.equal(g2.history.length, 0, JSON.stringify(g2.history.map((h) => h.why)));
  // a frame of 0.2 s once (a hiccup) does not step down either
  const g3 = new Governor({ ladder, onStep: () => {} }); now = 0;
  for (let i = 0; i < 400; i++) { now += i === 100 ? 200 : 16.7; g3.push(i === 100 ? 200 : 16.7, now); }
  assert.equal(g3.history.length, 0);
});
t("governor: two frames over 400 ms step down at once (software rendering), also with a 3 s frame time and with the trial rule on", () => {
  const steps = []; const g = new Governor({ ladder: buildLadder({ previewFallback: true }), onStep: (s) => steps.push(s.name) });
  g.push(900, 1000); assert.equal(steps.length, 0); g.push(900, 1900); assert.equal(steps[0], "low");
  g.push(900, 2800); g.push(900, 3700); assert.equal(steps[1], "min");
  const slow = []; const g2 = new Governor({ ladder: buildLadder({ previewFallback: true }), onStep: (s) => slow.push(s.name) });
  g2.push(3000, 3000); g2.push(3000, 6000); g2.push(3000, 9000); g2.push(3000, 12000);   // frames longer than the window still count
  assert.deepEqual(slow, ["low", "min"]);
  // the emergency rule does not wait for a trial: it steps even after a capped trial set the block
  const g3 = new Governor({ ladder: buildLadder({ previewFallback: true }), onStep: () => {} });
  let now = 0; while (now < 9000) { now += 33.3; g3.push(33.3, now); }
  const before = g3.level; g3.push(700, now + 700); g3.push(700, now + 1400);
  assert.equal(g3.level, before + 1);
});
t("governor: leaving the benchmark or the reality check clears old samples; HD ladder swap resets the trial", () => {
  const g = new Governor({ ladder: buildLadder({}), onStep: () => {} }); let now = 0;
  g.setEnabled(false); for (let i = 0; i < 200; i++) { now += 50; g.push(50, now); }   // slow frames while disabled count for nothing
  assert.equal(g.samples.length, 0);
  g.setEnabled(true); now += 16.7; g.push(16.7, now);
  assert.equal(g.samples.length, 1);
  g.trial = { from: 0, fps0: 20 }; g.blockFps = 20; g.jumpTo(0); assert.equal(g.trial, null); assert.equal(g.blockFps, null);
});
t("governor: after a step up drops the rate under 40 again, it does not climb into that step for a minute (no preview <-> SD flapping)", () => {
  const ladder = buildLadder({ previewFallback: true });
  // machine: fine (60 fps) at the preview and 'min' stays at 30 fps whatever the step; standard/low/min cost 50 ms work, the preview costs 0.2 of it
  const r = run({ ladder, frameMs: (s) => (s.mesh === "preview" ? 7 : gpuBound(80, 144)(s)), seconds: 70 });
  const ups = r.g.history.filter((h) => h.kind === "up").length;
  assert.ok(ups <= 2, `ups ${ups}: ${r.applied.join(",")}`);
});
t("governor ladder: a step that changes nothing is left out (weak GPUs start at pixel ratio 0.85)", () => {
  const weak = buildLadder({ basePixelRatio: 0.85, previewFallback: true }).map((l) => l.name);
  assert.deepEqual(weak, ["standard", "min", "preview"]);
  const soft = buildLadder({ basePixelRatio: 0.5, previewFallback: true }).map((l) => l.name);
  assert.deepEqual(soft, ["standard", "preview"].slice(0, 1).concat(soft.slice(1)));
  assert.equal(soft[0], "standard"); assert.ok(!soft.includes("low") && !soft.includes("min"));
  assert.deepEqual(buildLadder({ previewFallback: true }).map((l) => l.name), ["standard", "low", "min", "preview"]);
  const full = buildLadder({}); for (let i = 1; i < full.length; i++) assert.ok(full[i].pixelRatio !== full[i - 1].pixelRatio || full[i].maxStdDev !== full[i - 1].maxStdDev);
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

// --- wording and scene helpers (review findings 29, 44, 19, 47, 46) ----------------------------------------------------------
t("scale text: an estimate from the camera-height prior is called an estimate; only a stored check is called checked", () => {
  const est = scaleInfo({ pct_68: 12.749685157937574, method: "camera_height", checked_against: null });
  assert.equal(est.short, "est. ±13%"); assert.equal(est.checked, false); assert.match(est.long, /prior on camera height/); assert.match(est.long, /not a measured error/);
  assert.equal(scaleInfo({ pct_68: 8.3 }).short, "est. ±8.3%");
  assert.equal(scaleInfo({ pct_68: 0 }).short, "unchecked"); assert.equal(scaleInfo(undefined).short, "unchecked");
  const chk = scaleInfo({ pct_68: 12.7, checked_against: "tape", check_error_pct: 2.13 });
  assert.equal(chk.checked, true); assert.equal(chk.short, "tape check +2.1%"); assert.equal(chk.pct, 0); assert.match(chk.long, /off by \+2\.1%/);
  assert.equal(scaleInfo({ pct_68: 12.7, checked_against: "tape", check_error_pct: -4 }).short, "tape check -4.0%");
  assert.equal(scaleInfo({ pct_68: 12.7, checked_against: "tape" }).short, "checked (tape)");
  for (const x of [est, chk]) assert.ok(!/we measured|measured error we/i.test(x.short));
});
t("HD is hidden when it is the same file as SD (same URL, or same bytes, count and SH degree), shown when it differs", () => {
  const e = (url, bytes, count, sh) => ({ url, bytes, count, sh_degree: sh });
  assert.equal(hdSameAsSd({ sd: e("a/splat_sd.spz", 16087774, 629993, 3), hd: e("a/splat_hd.spz", 16087774, 629993, 3) }), true);
  assert.equal(hdSameAsSd({ sd: e("a/s.spz", 5, 5, 3), hd: e("a/s.spz", 9, 9, 3) }), true);
  assert.equal(hdSameAsSd({ sd: e("a/splat_sd.spz", 2410000, 131543, 1), hd: e("a/splat_hd.spz", 2840000, 131543, 3) }), false);
  assert.equal(hdSameAsSd({ sd: e("a/s.spz", 5, 5, 1), hd: null }), false); assert.equal(hdSameAsSd({ sd: e("a/s.spz", 5, 5, 1) }), false);
  assert.equal(hdSameAsSd({ sd: e("a/s.spz", 0, 5, 1), hd: e("a/h.spz", 0, 5, 1) }), false);          // unknown sizes: do not guess
});
t("clear colour: read from scene.json when valid, else a neutral grey that is not the old near-black", () => {
  assert.deepEqual(clearColorOf({ clear_color: [120.4, 100, 90] }), [120, 100, 90]);
  for (const bad of [undefined, {}, { clear_color: [1, 2] }, { clear_color: [1, 2, 300] }, { clear_color: "red" }, { clear_color: [NaN, 1, 2] }]) assert.deepEqual(clearColorOf(bad), DEFAULT_CLEAR);
  assert.ok(DEFAULT_CLEAR.reduce((a, b) => a + b) > 3 * 30 && DEFAULT_CLEAR.every((v) => v > 11 + 20));
});
t("places keep an optional notes line through reading and the editor's JSON", () => {
  const p = normalisePlace({ name: "Door", position: [1, 0, 2], notes: "placed by hand with the place editor (build team), 30 Oct" });
  assert.equal(p.notes, "placed by hand with the place editor (build team), 30 Oct");
  assert.equal(normalisePlace({ name: "Sofa", position: [1, 0, 2] }).notes, undefined);
  assert.equal(JSON.parse(placesToJson([p]))[0].notes, p.notes);
});
console.log(`${n} unit tests passed`);
