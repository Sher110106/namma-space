// Records the viewer scenes for the demo video, straight from the real viewer, with real key and mouse events.
// usage: node video/rec/record.mjs <run> <outdir> <targets.json> [scene,scene,...]
// targets.json: {"walk_cold": 19.4, "walk": 14, ...}  seconds of footage wanted per scene (from the narration audio).
// Output: <outdir>/<scene>.mp4 (1920x1080, 30 fps). Bench writes bench_a.mp4 and bench_b.mp4. No frame files are written.
import fs from "node:fs";
import path from "node:path";
import { Recorder, openBundle, sleep, HERE } from "./lib.mjs";

const [run, outdir, targetsFile, only] = process.argv.slice(2);
const ROOT = path.resolve(HERE, "../..");
const WEB = path.join(ROOT, "runs", run, "80_web");
const targets = JSON.parse(fs.readFileSync(targetsFile, "utf8"));
fs.mkdirSync(outdir, { recursive: true });
const want = only ? new Set(only.split(",")) : null;
// The page is 1280x720 CSS px at device scale 1.5, so frames come out 1920x1080 and the splat canvas renders at 1280x720 (the M1 keeps 30+ fps while recording).
const VW = 1280, VH = 720, DPR = 1.5, W = 1920, H = 1080;
const report = {};

async function session(extraQuery = "") {
  const b = await openBundle(WEB, { w: VW, h: VH, dpr: DPR });
  await b.pg.goto(b.url(`?scene=${encodeURIComponent(run)}&keys=1&q=sd${extraQuery}`));
  await b.pg.waitFor("window.__ns && (window.__ns.ready||window.__ns.error)", 90000);
  if (await b.pg.eval("__ns.error")) throw new Error("viewer error: " + (await b.pg.eval("__ns.error")));
  await b.pg.waitFor("__ns.S.fullLoaded===true", 90000);
  await sleep(1500);
  return b;
}
const startGame = async (pg) => { await pg.eval("document.getElementById('s-go').click()"); await sleep(900); };
const pose = (pg) => pg.eval("__ns.pose()");

// A smooth mouse-drag turn. Positive degrees turn left. The viewer's drag-look uses 0.0034 rad per pixel.
async function turn(pg, deg, ms = 900) {
  const px = Math.round((Math.abs(deg) * Math.PI / 180) / 0.0034), dir = deg > 0 ? -1 : 1, x0 = VW / 2, y0 = VH / 2;
  const steps = Math.max(8, Math.round(ms / 33));
  await pg.mouse("mousePressed", x0, y0);
  for (let i = 1; i <= steps; i++) { await pg.mouse("mouseMoved", x0 + (dir * px * i) / steps, y0, { buttons: 1 }); await sleep(ms / steps); }
  await pg.mouse("mouseReleased", x0 + dir * px, y0);
}
const norm = (d) => ((d + 540) % 360) - 180;
// Free distance (m) ahead for each heading, measured on the viewer's own walking grid.
async function headings(pg) {
  return pg.eval(`(() => { const g = __ns.grid, p = __ns.pose(), out = [];
    for (let h = -180; h < 180; h += 10) { const r = h * Math.PI / 180, dx = -Math.sin(r), dz = -Math.cos(r); let d = 0;
      while (d < 12 && g.walkable(p.x + dx * (d + 0.05), p.z + dz * (d + 0.05))) d += 0.05; out.push({ h, d }); }
    return { yaw: p.yawDeg, out }; })()`);
}
async function faceFarthest(pg, ms = 1400, maxTurn = 180) {
  const { yaw, out } = await headings(pg);
  let best = null;
  for (const o of out) { const dd = norm(o.h - yaw); if (Math.abs(dd) <= maxTurn && (!best || o.d > best.d)) best = { ...o, dd }; }
  if (Math.abs(best.dd) > 4) await turn(pg, best.dd, ms * Math.min(1.5, Math.abs(best.dd) / 60 + 0.5));
  return best;
}
// Hold W until the walker stops against a wall, then keep holding so the stop is visible.
async function walkToWall(pg, maxS = 8, holdS = 1.8) {
  await pg.key("KeyW");
  let still = 0, last = await pose(pg); const t0 = Date.now();
  while (Date.now() - t0 < maxS * 1000) {
    await sleep(120);
    const p = await pose(pg);
    still = Math.hypot(p.x - last.x, p.z - last.z) < 0.004 ? still + 1 : 0; last = p;
    if (still >= 4) break;
  }
  await sleep(holdS * 1000);
  await pg.key("KeyW", false);
}
async function fill(rec, target) { const left = target - rec.t; if (left > 0) await sleep(left * 1000); }

async function sceneWalk(name, { lead = 1.0, wiggle = true } = {}) {
  const target = targets[name];
  const b = await session(); const pg = b.pg;
  await startGame(pg);
  const rec = await new Recorder(pg, path.join(outdir, name + ".mp4"), { w: W, h: H }).start();
  await sleep(lead * 1000);
  await pg.key("KeyW");
  await sleep(1800);
  if (wiggle) { await turn(pg, 28, 1100); await sleep(500); await turn(pg, -45, 1400); }
  await pg.key("KeyW", false);
  await sleep(500);
  const best = await faceFarthest(pg);
  await sleep(400);
  await walkToWall(pg, 9, 2.0);
  const info = { heading: best.h, free_m: best.d, end: await pose(pg) };
  // spend any time left on a slow look-around so the clip ends on the stopped view
  const left = target - rec.t - 0.3;
  if (left > 2.5) { await turn(pg, -35, Math.min(left * 1000 - 600, 4500)); }
  await fill(rec, target);
  report[name] = { ...(await rec.stop()), ...info, target };
  await b.close();
}

async function sceneMap() {
  const target = targets.map;
  const b = await session(); const pg = b.pg;
  await startGame(pg);
  const rec = await new Recorder(pg, path.join(outdir, "map.mp4"), { w: W, h: H }).start();
  await pg.key("KeyW"); await sleep(1400); await pg.key("KeyW", false); await sleep(600);
  await pg.tap("KeyM"); await sleep(1800);                                   // large map
  const tgt = await pg.eval(`(() => { const S = __ns.S, m = S.minimap, p = __ns.pose(); const pl = (S.data.places || []).map((q) => q.position);
    let best = null; for (const bk of S.data.bookmarks) { const [x, , z] = bk.position; const dp = Math.hypot(x - p.x, z - p.z);
      const near = pl.length ? Math.min(...pl.map((q) => Math.hypot(q[0] - x, q[2] - z))) : 9; if (near < 0.7) continue;
      if (!best || dp > best.dp) best = { x, z, dp }; }
    if (!best) { const [x, , z] = S.data.bookmarks[0].position; best = { x, z, dp: 0 }; }
    const [cx, cz] = m.toCanvas(best.x, best.z), r = m.c.getBoundingClientRect();
    return { px: r.left + (cx / m.c.width) * r.width, py: r.top + (cz / m.c.height) * r.height, x: best.x, z: best.z }; })()`);
  await pg.mouse("mouseMoved", tgt.px - 120, tgt.py - 90, { buttons: 0 }); await sleep(250);
  await pg.mouse("mouseMoved", tgt.px, tgt.py, { buttons: 0 }); await sleep(700);
  const before = await pose(pg);
  await pg.click(tgt.px, tgt.py); await sleep(1500);
  const after = await pose(pg);
  const jumped = Math.hypot(after.x - before.x, after.z - before.z);
  await sleep(600);
  await fill(rec, target);
  report.map = { ...(await rec.stop()), jumped_m: +jumped.toFixed(2), target };
  await b.close();
}

async function sceneReality() {
  const target = targets.reality;
  const b = await session(); const pg = b.pg;
  await startGame(pg);
  await turn(pg, 20, 600);
  const rec = await new Recorder(pg, path.join(outdir, "reality.mp4"), { w: W, h: H }).start();
  await sleep(900);
  await pg.tap("KeyC"); await sleep(1400);
  const sweep = async (secs) => {
    const bx = await pg.eval("__ns.S.reality.box");
    const x = (f) => bx.left + bx.w * f, y = bx.top + bx.h * 0.5;
    await pg.mouse("mouseMoved", x(0.97), y, { buttons: 0 });
    await pg.mouse("mousePressed", x(0.97), y);
    const steps = 40;
    for (let i = 1; i <= steps; i++) { await pg.mouse("mouseMoved", x(0.97 - 0.92 * i / steps), y, { buttons: 1 }); await sleep(secs * 1000 / steps); }
    await sleep(500);
    for (let i = 1; i <= steps / 2; i++) { await pg.mouse("mouseMoved", x(0.05 + 0.45 * i / (steps / 2)), y, { buttons: 1 }); await sleep(secs * 500 / steps); }
    await pg.mouse("mouseReleased", x(0.5), y);
  };
  const pairs = 2;
  const caps = [];
  for (let i = 0; i < pairs; i++) {
    caps.push(await pg.eval("document.getElementById('rc-cap').textContent"));
    await sweep(i === 0 ? 2.2 : 1.6); await sleep(i === pairs - 1 ? 700 : 900);
    if (i < pairs - 1) { await pg.tap("ArrowRight"); await sleep(900); }
  }
  await fill(rec, target);
  report.reality = { ...(await rec.stop()), captions: caps, target };
  await b.close();
}

async function sceneBench() {
  const secs = Math.round(targets.bench_secs || 10);
  const b = await session(`&benchsecs=${secs}`); const pg = b.pg;
  await startGame(pg);
  const a = await new Recorder(pg, path.join(outdir, "bench_a.mp4"), { w: W, h: H }).start();
  await sleep(500);
  await pg.tap("KeyT");
  await sleep(targets.bench_a * 1000 - 500);
  const ra = await a.stop();
  await pg.waitFor("__ns.benchResult", (secs + 20) * 1000, 200);   // wait with nothing recording so the run is not disturbed
  const result = await pg.eval("__ns.benchResult");
  await sleep(500);
  const bb = await new Recorder(pg, path.join(outdir, "bench_b.mp4"), { w: W, h: H }).start();
  await sleep(targets.bench_b * 1000);
  const rb = await bb.stop();
  report.bench = { a: ra, b: rb, result, benchsecs: secs };
  await b.close();
}

async function sceneMeasure() {
  const target = targets.measure;
  const b = await session(); const pg = b.pg;
  await startGame(pg);
  await faceFarthest(pg, 900);
  const rec = await new Recorder(pg, path.join(outdir, "measure.mp4"), { w: W, h: H }).start();
  await sleep(700);
  await pg.tap("KeyK"); await sleep(1000);
  await pg.click(VW * 0.33, VH * 0.52); await sleep(1700);
  await pg.click(VW * 0.67, VH * 0.52); await sleep(2000);
  const res = await pg.eval("document.getElementById('tool-msg')?.textContent || ''");
  await pg.tap("Escape"); await sleep(400);
  const expanded = await pg.eval("document.getElementById('hud').classList.contains('expanded')");
  await pg.tap("KeyH"); await sleep(1300);
  if (expanded) { await pg.tap("KeyH"); } await sleep(200);
  await fill(rec, target);
  report.measure = { ...(await rec.stop()), tool_text: res, target };
  await b.close();
}

async function sceneRoute() {
  const target = targets.route;
  const b = await session(); const pg = b.pg;
  await startGame(pg);
  const pick = await pg.eval(`(() => { const p = __ns.pose(); const pl = __ns.S.data.places || [];
    let best = null; for (const q of pl) { const d = Math.hypot(q.position[0] - p.x, q.position[2] - p.z); if (d >= 3 && (!best || d < best.d)) best = { id: q.id, name: q.name, d }; }
    return best || (pl[0] ? { id: pl[0].id, name: pl[0].name, d: 0 } : null); })()`);
  if (!pick) throw new Error("no places in scene");
  const rec = await new Recorder(pg, path.join(outdir, "route.mp4"), { w: W, h: H }).start();
  await sleep(900);
  await pg.tap("Slash/"); await sleep(900);
  await pg.type(pick.name.toLowerCase().split(" ")[0]); await sleep(1500);
  await pg.tap("Enter"); await sleep(1500);
  await pg.tap("KeyG"); await sleep(2800);
  await pg.tap("Space"); await sleep(1000);
  const t0 = Date.now();
  while (Date.now() - t0 < 14000 && await pg.eval("!!__ns.S.guide.following")) await sleep(200);
  await sleep(1200);
  await fill(rec, target);
  report.route = { ...(await rec.stop()), place: pick, target };
  await b.close();
}

const all = {
  walk_cold: () => sceneWalk("walk_cold", { lead: 1.2 }),
  walk: () => sceneWalk("walk", { lead: 0.5, wiggle: false }),
  map: sceneMap, reality: sceneReality, bench: sceneBench, measure: sceneMeasure, route: sceneRoute,
};
for (const [name, fn] of Object.entries(all)) {
  if (want && !want.has(name)) continue;
  if (name !== "bench" && !(name in targets)) continue;
  const t0 = Date.now();
  try { await fn(); console.log("recorded", name, ((Date.now() - t0) / 1000).toFixed(0) + " s wall"); }
  catch (e) { console.error("FAILED", name, e.message); report[name] = { error: e.message }; }
}
fs.writeFileSync(path.join(outdir, "record_report.json"), JSON.stringify(report, null, 2));
process.exit(0);
