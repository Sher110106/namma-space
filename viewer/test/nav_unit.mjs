// Unit tests for route finding, fuzzy search and follow steering (no browser). Run: node viewer/test/nav_unit.mjs
// Writes a summary line per check and, with --json FILE, the numbers (used by run_all.sh for verify.json).
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { Grid, Walker } from "../src/js/walker.js";
import { standBackFor, astar, routeToPlace, samplePolyline, pathWalkable, polylineLength, standSpot, gridInfo, lineOfSight, segmentClear, followStep, projectOnPolyline, pointAtS } from "../src/js/nav.js";
import { searchPlaces, readPlaces, editDistance, yawToward, placesToJson, norm } from "../src/js/places.js";
import { readPng } from "./lib/png.mjs";
import { oracle } from "./lib/oracle.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const numbers = {};
let n = 0;
const t = (name, fn) => { fn(); n++; console.log("ok  ", name); };

// ---- test grids
function mulberry(seed) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let x = a; x = Math.imul(x ^ (x >>> 15), x | 1); x ^= x + Math.imul(x ^ (x >>> 7), x | 61); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; }; }
function synthetic(seed, w = 220, h = 160, rects = 38) {
  const rnd = mulberry(seed), data = new Uint8Array(w * h);
  const set = (i0, j0, i1, j1, v) => { for (let i = Math.max(0, i0); i < Math.min(h, i1); i++) for (let j = Math.max(0, j0); j < Math.min(w, j1); j++) data[i * w + j] = v; };
  set(5, 5, h - 5, w - 5, 255);
  for (let k = 0; k < rects; k++) { const i = 8 + Math.floor(rnd() * (h - 30)), j = 8 + Math.floor(rnd() * (w - 40)); set(i, j, i + 3 + Math.floor(rnd() * 22), j + 3 + Math.floor(rnd() * 26), 0); }
  return new Grid({ width: w, height: h, cell: 0.05, origin_xz: [-5.5, -4] }, data);
}
function uWall() {          // a wall with a gap at one end: the route must go round it
  const w = 120, h = 80, data = new Uint8Array(w * h).fill(0);
  for (let i = 5; i < 75; i++) for (let j = 5; j < 115; j++) data[i * w + j] = 255;
  for (let i = 5; i < 60; i++) for (let j = 58; j < 62; j++) data[i * w + j] = 0;        // wall from the top, gap at the bottom
  return new Grid({ width: w, height: h, cell: 0.05, origin_xz: [0, 0] }, data);
}
function honka() {
  const f = path.join(HERE, "fixtures/honka_run/70_grid");
  const sc = JSON.parse(fs.readFileSync(path.join(HERE, "../dist/scenes/honka_demo/scene.json"), "utf8")).grid;
  const png = readPng(path.join(f, "grid.png"));
  return new Grid({ ...sc, width: png.width, height: png.height }, png.data);
}

/** Random same-component pairs; route each; collect the checks. */
function routeBattery(g, seed, pairs, label) {
  const rnd = mulberry(seed), info = gridInfo(g), cells = [];
  for (let k = 0; k < g.data.length; k++) if (info.comp[k]) cells.push(k);
  let bad = 0, worst = 0, sumRatio = 0, cnt = 0, minRatio = 9, samples = 0;
  const failures = [];
  for (let p = 0; p < pairs; p++) {
    const a = cells[Math.floor(rnd() * cells.length)], b = cells[Math.floor(rnd() * cells.length)];
    if (info.comp[a] !== info.comp[b]) continue;
    const [ax, az] = g.cellCentre((a / g.w) | 0, a % g.w), [bx, bz] = g.cellCentre((b / g.w) | 0, b % g.w);
    const place = { name: "target", kind: "zone", position: [bx, 0, bz] };
    const r = routeToPlace(g, { x: ax, z: az }, place);
    assert.ok(r.ok, `${label}: no route between two cells of one component (${ax},${az}) -> (${bx},${bz}): ${r.reason}`);
    for (const [x, z] of samplePolyline(r.points, 0.05)) { samples++; if (!g.walkable(x, z)) { bad++; if (failures.length < 3) failures.push({ a: [ax, az], b: [bx, bz], at: [x, z] }); } }
    const opt = oracle(g, a, b), ratio = opt > 0 ? r.length / opt : 1;
    worst = Math.max(worst, ratio); minRatio = Math.min(minRatio, ratio); if (opt > 0) { sumRatio += ratio; cnt++; }
    const straight = Math.hypot(bx - ax, bz - az);
    assert.ok(r.length >= straight - 1e-6, `${label}: route shorter than the straight line`);
  }
  return { pairs: cnt, samples, blocked_samples: bad, worst_ratio: +worst.toFixed(4), mean_ratio: +(sumRatio / Math.max(1, cnt)).toFixed(4), min_ratio: +minRatio.toFixed(4), failures };
}

// ---------------------------------------------------------------------------------------------------------------- routes
t("A*: octile shortest path equals the independent Dijkstra length (no wall penalty)", () => {
  const g = synthetic(11); const info = gridInfo(g); let checked = 0;
  const rnd = mulberry(5); const cells = []; for (let k = 0; k < g.data.length; k++) if (info.comp[k]) cells.push(k);
  for (let p = 0; p < 60; p++) {
    const a = cells[Math.floor(rnd() * cells.length)], b = cells[Math.floor(rnd() * cells.length)]; if (info.comp[a] !== info.comp[b]) continue;
    const path_ = astar(g, a, b, { penalty: false });
    let L = 0; for (let k = 1; k < path_.length; k++) { const di = Math.abs(((path_[k] / g.w) | 0) - ((path_[k - 1] / g.w) | 0)), dj = Math.abs((path_[k] % g.w) - (path_[k - 1] % g.w)); L += di && dj ? Math.SQRT2 : 1; }
    assert.ok(Math.abs(L * g.cell - oracle(g, a, b)) < 1e-6, `A* ${L * g.cell} vs oracle ${oracle(g, a, b)}`); checked++;
  }
  numbers.astar_vs_dijkstra_pairs = checked;
});
t("routes on 5 random synthetic grids: 0 blocked samples at 5 cm, length within 10% of the grid optimum", () => {
  const rows = [];
  for (const seed of [1, 2, 3, 4, 5]) rows.push(routeBattery(synthetic(seed), 100 + seed, 120, `synthetic ${seed}`));
  const agg = { pairs: 0, samples: 0, blocked: 0, worst: 0, mean: 0 };
  for (const r of rows) { agg.pairs += r.pairs; agg.samples += r.samples; agg.blocked += r.blocked_samples; agg.worst = Math.max(agg.worst, r.worst_ratio); agg.mean += r.mean_ratio / rows.length; }
  numbers.synthetic = { grids: 5, pairs: agg.pairs, samples: agg.samples, blocked_samples: agg.blocked, worst_length_ratio: agg.worst, mean_length_ratio: +agg.mean.toFixed(4) };
  assert.equal(agg.blocked, 0, JSON.stringify(rows.map((r) => r.failures)));
  assert.ok(agg.worst <= 1.10, `worst ratio ${agg.worst}`);
});
t("routes on the honka fixture grid: 0 blocked samples at 5 cm, length within 10% of the grid optimum", () => {
  const g = honka(), r = routeBattery(g, 77, 400, "honka");
  numbers.honka = { pairs: r.pairs, samples: r.samples, blocked_samples: r.blocked_samples, worst_length_ratio: r.worst_ratio, mean_length_ratio: r.mean_ratio, min_length_ratio: r.min_ratio, walkable_m2: +g.walkableArea().toFixed(1) };
  assert.equal(r.blocked_samples, 0, JSON.stringify(r.failures)); assert.ok(r.worst_ratio <= 1.10, `worst ratio ${r.worst_ratio}`);
});
t("a wall with a gap: the route goes through the gap, not the wall", () => {
  const g = uWall(), r = routeToPlace(g, { x: 1.0, z: 1.0 }, { name: "far side", kind: "zone", position: [4.9, 0, 1.0] });
  assert.ok(r.ok); assert.ok(r.length > 5.4 && r.length < 6.2, `length ${r.length}`); assert.ok(pathWalkable(g, r.points, 0.01));
  assert.ok(r.points.some((p) => p[1] > 3.0), "route must pass the gap at the bottom (z > 3 m)");
});
t("start and goal in one cell, or the player already at the place: a route of ~0 length, ok", () => {
  const g = uWall(); const r = routeToPlace(g, { x: 1.02, z: 1.02 }, { name: "here", kind: "zone", position: [1.03, 0, 1.01] });
  assert.ok(r.ok && r.length < 0.2, JSON.stringify(r));
});
t("no-path cases give a clear message", () => {
  // an island: the place sits in a closed pocket
  const w = 80, h = 40, data = new Uint8Array(w * h);
  for (let i = 5; i < 35; i++) for (let j = 5; j < 35; j++) data[i * w + j] = 255;           // main room
  for (let i = 15; i < 25; i++) for (let j = 55; j < 70; j++) data[i * w + j] = 255;         // island far away
  const g = new Grid({ width: w, height: h, cell: 0.05, origin_xz: [0, 0] }, data);
  const r = routeToPlace(g, { x: 1.0, z: 1.0 }, { name: "Pocket", kind: "room", position: [3.4, 0, 1.0] });
  assert.equal(r.ok, false); assert.match(r.reason, /Pocket/); assert.match(r.reason, /No walking route/);
  const near = routeToPlace(g, { x: 1.0, z: 1.0 }, { name: "Edge", kind: "zone", position: [1.9, 0, 1.0] });   // beside the room edge: the closest reachable spot is within 1.5 m of the ideal one
  assert.ok(near.ok);
  const empty = new Grid({ width: 10, height: 10, cell: 0.05, origin_xz: [0, 0] }, new Uint8Array(100));
  const e = routeToPlace(empty, { x: 0.2, z: 0.2 }, { name: "x", kind: "zone", position: [0.3, 0, 0.3] });
  assert.equal(e.ok, false); assert.match(e.reason, /no walkable floor/i);
});
t("a place inside furniture: the route ends on a walkable cell near it; items keep 0.8 m back", () => {
  const g = uWall(); gridInfo(g);
  const item = { name: "Lamp", kind: "item", position: [2.0, 1.0, 1.0], radius_m: 0.4 };
  const s = standSpot(g, 2.0, 1.0, { minDist: 0.8 });
  assert.ok(g.walkable(s.x, s.z) && s.d >= 0.8 - 1e-9 && s.d < 0.86, `stand ${JSON.stringify(s)}`);
  const room = standSpot(g, 3.0, 3.0, {}); assert.ok(room.d < 0.05);
  const wall = { name: "On the wall", kind: "zone", position: [3.0, 0, 1.0] };      // x 3.0 is inside the wall (cols 58-61 = x 2.9-3.1)
  const r = routeToPlace(g, { x: 1.0, z: 3.0 }, wall); assert.ok(r.ok && g.walkable(r.stand.x, r.stand.z));
  assert.ok(routeToPlace(g, { x: 1.0, z: 3.0 }, item).ok);
});
t("stand-back distance: items radius + 0.6 m (0.8 to 2.0), rooms and zones 0", () => {
  assert.equal(standBackFor({ kind: "item", radius_m: 0.4 }), 1.0); assert.equal(standBackFor({ kind: "item", radius_m: 0.1 }), 0.8); assert.equal(standBackFor({ kind: "item", radius_m: 3 }), 2.0);
  assert.equal(standBackFor({ kind: "room", radius_m: 1.5 }), 0); assert.equal(standBackFor({ kind: "zone" }), 0);
});
t("line of sight: exact corner touching is blocked; clearance rejects a close pass", () => {
  const g = uWall();
  assert.equal(segmentClear(g, 1, 1, 4.9, 1), false);                           // through the wall
  assert.equal(segmentClear(g, 1, 3.5, 4.9, 3.5), true);
  assert.equal(lineOfSight(g, [2.0, 3.25], [4.0, 3.25], 0.1), true);              // wall ends at z = 3.0 (row 60); the +/-10 cm side lines stay clear
  assert.equal(lineOfSight(g, [2.0, 3.04], [4.0, 3.04], 0.10), false);            // the +/-10 cm side lines hit the wall end
  assert.equal(lineOfSight(g, [2.0, 3.04], [4.0, 3.04], 0), true);
});
t("string pulling shortens a staircase 8-connected path to a few straight pieces in open floor", () => {
  const g = synthetic(3, 120, 100, 0); const r = routeToPlace(g, { x: -5, z: -3.5 }, { name: "t", kind: "zone", position: [0.0, 0, 0.5] });
  assert.ok(r.ok && r.points.length <= 3, `points ${r.points.length}`);
  assert.ok(Math.abs(r.length - Math.hypot(5, 4.0)) < 0.1);
});
t("time at 1.4 m/s", () => { const g = uWall(); const r = routeToPlace(g, { x: 1, z: 3.5 }, { name: "t", kind: "zone", position: [5, 0, 3.5] }, { walkMps: 1.4 }); assert.ok(Math.abs(r.seconds - r.length / 1.4) < 1e-9 && r.length > 3.9); });
t("route to a place re-plans from a new start (what the viewer does after the player leaves the path)", () => {
  const g = uWall(), p = { name: "far", kind: "zone", position: [5.2, 0, 3.0] };
  const r1 = routeToPlace(g, { x: 1.0, z: 3.5 }, p), pr = projectOnPolyline(r1.points, 1.0, 1.9);
  assert.ok(pr.dist > 0.5, `off-path distance ${pr.dist}`);                         // 1.6 m above the route: the viewer re-plans
  const r2 = routeToPlace(g, { x: 1.0, z: 1.9 }, p); assert.ok(r2.ok && r2.length > r1.length);
});

// ---------------------------------------------------------------------------------------------------------------- follow
t("follow mode: a Walker steered by followStep reaches the stand spot, never leaves walkable cells, at about 1.4 m/s", () => {
  const rows = [];
  for (const [name, g, from, place] of [
    ["gap wall", uWall(), { x: 1.0, z: 1.0 }, { name: "far", kind: "zone", position: [5.0, 0, 1.0] }],
    ["honka", honka(), null, null],
  ]) {
    let start = from, pl = place;
    if (!start) { const info = gridInfo(g); const cells = []; for (let k = 0; k < g.data.length; k++) if (info.comp[k] === 1) cells.push(k); const a = cells[7], b = cells[cells.length - 9]; const [ax, az] = g.cellCentre((a / g.w) | 0, a % g.w), [bx, bz] = g.cellCentre((b / g.w) | 0, b % g.w); start = { x: ax, z: az }; pl = { name: "t", kind: "item", position: [bx, 1, bz], radius_m: 0.4 }; }
    const r = routeToPlace(g, start, pl); assert.ok(r.ok, r.reason);
    const w = new Walker(g, { eyeHeight: 1.42, walkMps: 1.4, runMult: 2 }); w.setPose(start.x, start.z, 0, 0);
    const st = {}; let tt = 0, bad = 0; const dt = 1 / 60;
    const face = (yawToward(r.stand.x, r.stand.z, pl.position[0], pl.position[2]) * Math.PI) / 180;
    while (!st.done && tt < 200) { const input = followStep(r.points, w, st, dt, { faceYaw: face }); w.update(dt, input); if (!g.walkable(w.x, w.z)) bad++; tt += dt; }
    const err = Math.hypot(w.x - r.stand.x, w.z - r.stand.z);
    assert.ok(st.done && err < 0.4 && bad === 0, `${name}: done ${st.done} err ${err.toFixed(2)} bad ${bad}`);
    rows.push({ name, length_m: +r.length.toFixed(2), seconds: +tt.toFixed(1), mean_mps: +(r.length / tt).toFixed(2), end_error_m: +err.toFixed(3), blocked_frames: bad });
  }
  numbers.follow_sim = rows;
  for (const r of rows) assert.ok(r.mean_mps > 0.9 && r.mean_mps < 1.5, JSON.stringify(r));
});
t("yawToward matches the Walker convention (yaw 0 looks along -Z, positive turns left)", () => {
  assert.ok(Math.abs(yawToward(0, 0, 0, -1)) < 1e-9); assert.ok(Math.abs(yawToward(0, 0, -1, 0) - 90) < 1e-9);
  assert.ok(Math.abs(Math.abs(yawToward(0, 0, 0, 1)) - 180) < 1e-9); assert.ok(Math.abs(yawToward(0, 0, 1, 0) + 90) < 1e-9);
  const w = new Walker(null, { eyeHeight: 1.4, walkMps: 1.4, runMult: 2 }); w.setPose(0, 0, yawToward(0, 0, 3, 4)); for (let k = 0; k < 300; k++) w.update(1 / 60, { forward: 1, strafe: 0, run: false });
  assert.ok(Math.abs(w.x / w.z - 3 / 4) < 0.02 && w.x > 0 && w.z > 0, `${w.x} ${w.z}`);
});

// ---------------------------------------------------------------------------------------------------------------- search
const P = readPlaces([
  { id: "kitchen", name: "Kitchen", aliases: ["cooking area", "rasoi"], kind: "room", position: [1, 0, 1] },
  { id: "bedroom", name: "Bedroom", aliases: ["sleeping room"], kind: "room", position: [4, 0, 1] },
  { id: "study-desk", name: "Study desk", aliases: ["work table", "laptop desk"], kind: "item", position: [2, 0.8, 3] },
  { id: "tv", name: "TV", aliases: ["television", "telly"], kind: "item", position: [3, 1, 3] },
  { id: "sofa", name: "Sofa", aliases: ["couch", "settee"], kind: "item", position: [3, 0.4, 2] },
  { id: "balcony", name: "Balcony", aliases: ["terrace"], kind: "zone", position: [0, 0, 4] },
]);
const top = (q) => searchPlaces(P, q)[0]?.place.id;
t("search: finds by name, alias, prefix and word order", () => {
  assert.equal(top("kitchen"), "kitchen"); assert.equal(top("rasoi"), "kitchen"); assert.equal(top("couch"), "sofa"); assert.equal(top("telev"), "tv");
  assert.equal(top("desk study"), "study-desk"); assert.equal(top("work"), "study-desk"); assert.equal(top("  BED  "), "bedroom"); assert.equal(top("tv"), "tv");
});
t("search: one typo still finds it (substitution, deletion, insertion, swap)", () => {
  const cases = { kitchn: "kitchen", kithcen: "kitchen", bedrom: "bedroom", bedrooom: "bedroom", balcny: "balcony", sofaa: "sofa", cuoch: "sofa", televison: "tv", teraace: "balcony" };
  const miss = Object.entries(cases).filter(([q, id]) => top(q) !== id);
  numbers.search_typo_cases = Object.keys(cases).length; assert.deepEqual(miss, []);
});
t("search: nonsense finds nothing; short queries do not match by typo; empty query lists all", () => {
  assert.equal(searchPlaces(P, "zzzzqq").length, 0); assert.equal(searchPlaces(P, "tx").length, 0); assert.equal(searchPlaces(P, "").length, P.length);
  assert.equal(searchPlaces(P, "kitchen sofa").length, 0, "every word must match");
});
t("search: empty query sorts by distance when given a position; editDistance basics; norm strips accents", () => {
  assert.equal(searchPlaces(P, "", { from: { x: 3, z: 3 } })[0].place.id, "tv");
  assert.equal(editDistance("ab", "ba"), 1); assert.equal(editDistance("kitten", "sitting"), 3); assert.equal(norm("Café  Ünit!"), "cafe unit");
});
t("readPlaces: defaults, [x, z] positions, duplicate ids, junk dropped; placesToJson round-trips", () => {
  const l = readPlaces([{ name: "A", position: [1, 2] }, { name: "A", kind: "room", position: [1, 0, 2], aliases: "x, y" }, { name: "", position: [0, 0, 0] }, { name: "B" }, null]);
  assert.equal(l.length, 2); assert.deepEqual(l.map((p) => p.id), ["a", "a-2"]); assert.equal(l[0].position[1], 1.0); assert.deepEqual(l[1].aliases, ["x", "y"]); assert.equal(l[1].radius_m, 1.5);
  assert.deepEqual(readPlaces(JSON.parse(placesToJson(l))), l);
});

console.log(`\n${n} nav/search checks passed`);
const jf = process.argv.indexOf("--json");
if (jf > 0) fs.writeFileSync(process.argv[jf + 1], JSON.stringify(numbers, null, 2) + "\n");
console.log(JSON.stringify(numbers));
