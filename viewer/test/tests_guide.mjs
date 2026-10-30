// Round 2 preview tests: places, search, route, jump, follow, place editor, measure tool, phone layout.
// Registered from tests.mjs. Every test drives the real viewer in headless Chrome (real key and mouse events where it matters).
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { withBrowser, openViewer, waitFullLoad, sceneJson, consoleErrors, sleep } from "./lib/env.mjs";
import { oracle } from "./lib/oracle.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BUNDLE = path.join(HERE, "../tools/bundle.py");
const FIXTURE_RUN = path.join(HERE, "fixtures/honka_run");
const PLACES_FILE = path.join(HERE, "fixtures/honka_places.json");
const r1 = (v) => Math.round(v * 10) / 10;
const r3 = (v) => Math.round(v * 1000) / 1000;
const angDiff = (a, b) => Math.abs(((a - b + 540) % 360) - 180);

function lcg(seed) { let a = seed >>> 0; return () => ((a = (Math.imul(a, 1664525) + 1013904223) >>> 0) / 4294967296); }

/** Open the viewer, press start, wait for the first frame. */
async function ready(ctx, query = "") { await openViewer(ctx, query, { start: true }); await sleep(250); }
const sel = (pg) => pg.eval("__ns.S.guide.sel ? __ns.S.guide.sel.id : null");
const pose = (pg) => pg.eval("__ns.pose()");
async function search(pg, text, { enter = true } = {}) {
  await pg.tap("Slash/"); await sleep(120);
  await pg.type(text); await sleep(120);
  const ids = await pg.eval("[...document.querySelectorAll('#srch-list li[data-id]')].map(li => li.dataset.id)");
  if (enter) { await pg.tap("Enter"); await sleep(150); }
  return ids;
}


// Search cases for the test fixture's own places (the names are fixed there).
const FIXTURE_SEARCH_CASES = [["wall display", "wall-display", "name"], ["tv", "wall-display", "alias"], ["monitor", "wall-display", "alias"], ["sideboard", "coffee-counter", "alias"], ["entrance", "door", "alias"],
  ["meting table", "meeting-table", "typo: deleted letter"], ["cofee counter", "coffee-counter", "typo: deleted letter"], ["windws", "window-blinds", "typo (alias)"], ["armchiar", "armchair", "typo: swap"], ["forest pictur", "forest-picture", "partial"], ["foerst picture", "forest-picture", "typo: swap"]];

/** Search cases built from any scene's places: the name, one alias that no other place shares, and one-letter typos of a long, unshared word of the name. */
function searchCasesFor(places) {
  const words = (str) => String(str).toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
  const wordsOf = (p) => [...words(p.name), ...p.aliases.flatMap(words)];
  const lev = (a, b) => { const d = Array.from({ length: a.length + 1 }, (_, i) => [i, ...Array(b.length).fill(0)]); for (let j = 1; j <= b.length; j++) d[0][j] = j;
    for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++) d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1)); return d[a.length][b.length]; };
  // a word is unshared when no other place has a word that is close to it (one edit), starts with it, or contains it
  const unshared = (w, p) => places.filter((q) => q.id !== p.id).every((q) => wordsOf(q).every((x) => lev(x, w) > 1 && !x.startsWith(w) && !w.startsWith(x) && !x.includes(w)));
  const cases = [];
  for (const p of places) {
    cases.push([p.name.toLowerCase(), p.id, "name"]);
    const al = p.aliases.find((a) => words(a).length && words(a).every((w) => unshared(w, p)));
    if (al) cases.push([al.toLowerCase(), p.id, "alias"]);
  }
  let k = 0;
  for (const p of places) {
    const w = words(p.name).filter((x) => x.length >= 5 && unshared(x, p)).sort((a, b) => b.length - a.length)[0];
    if (!w) continue;
    const swap = k % 2 === 1, i = 2;
    const typo = swap ? w.slice(0, i) + w[i + 1] + w[i] + w.slice(i + 2) : w.slice(0, i) + w.slice(i + 1);
    if (typo === w || !unshared(typo, p)) continue;
    cases.push([p.name.toLowerCase().replace(w, typo), p.id, swap ? "typo: swap" : "typo: deleted letter"]);
    k++;
  }
  return cases;
}

export function registerGuideTests(T) {
  // ------------------------------------------------------------------------------------------------------ R0: Node unit tests
  T("R0", "route, search and follow logic (Node): routes on honka and 5 synthetic grids, A* vs Dijkstra, typo search, follow simulation", async () => {
    const out = path.join(os.tmpdir(), `ns-navunit-${process.pid}.json`);
    const r = spawnSync("node", [path.join(HERE, "nav_unit.mjs"), "--json", out], { encoding: "utf8" });
    let nums = {}; try { nums = JSON.parse(fs.readFileSync(out, "utf8")); fs.rmSync(out); } catch {}
    const passed = (r.stdout.match(/^ok /gm) || []).length;
    const h = nums.honka || {}, s = nums.synthetic || {};
    return { pass: r.status === 0, value: `${passed} checks; honka: ${h.pairs} routes, ${h.blocked_samples} blocked samples of ${h.samples} (5 cm), worst length ${h.worst_length_ratio}x optimum; synthetic: ${s.pairs} routes on ${s.grids} grids, ${s.blocked_samples} blocked of ${s.samples}, worst ${s.worst_length_ratio}x`,
      metrics: nums, detail: r.status === 0 ? "" : (r.stdout + r.stderr).slice(-600) };
  });

  // ------------------------------------------------------------------------------------------------------ R1: routes in the viewer
  T("R1", "routes in the viewer on the fixture grid: every place from 12 start cells; 0 blocked samples at 5 cm; length <= 1.10 x the grid optimum", async (c) => {
    return withBrowser(c.web, async (ctx) => {
      await ready(ctx);
      const n = await ctx.pg.eval("__ns.S.guide.places.length");
      if (!n) return { pass: null, value: "not run: scene has no places" };
      const rnd = lcg(2026);
      const meta = await ctx.pg.eval("({w: __ns.grid.w, h: __ns.grid.h, cell: __ns.grid.cell, ox: __ns.grid.ox, oz: __ns.grid.oz, data: Array.from(__ns.grid.data)})");
      const g = { ...meta, data: Uint8Array.from(meta.data) };
      const walk = []; for (let k = 0; k < g.data.length; k++) if (g.data[k] === 255) walk.push(k);
      const starts = Array.from({ length: 12 }, () => walk[Math.floor(rnd() * walk.length)]);
      const rows = [];
      const ids = await ctx.pg.eval("__ns.S.guide.places.map(p => p.id)");
      for (const id of ids) for (const sIdx of starts) {
        const sx = g.ox + ((sIdx % g.w) + 0.5) * g.cell, sz = g.oz + (Math.floor(sIdx / g.w) + 0.5) * g.cell;
        const res = await ctx.pg.eval(`(async () => { const nav = await import('./js/nav.js'); const G = __ns.S.guide, w = __ns.walker;
          w.teleport(${sx}, ${sz}); G.select(${JSON.stringify(id)}); G.routeOn = true; G.computeRoute(true); const r = G.route;
          if (!r || !r.ok) return {ok: false, reason: r && r.reason};
          let blocked = 0, samples = 0; for (const [x, z] of nav.samplePolyline(r.points, 0.05)) { samples++; if (!__ns.grid.walkable(x, z)) blocked++; }
          return {ok: true, length: r.length, seconds: r.seconds, stand: r.stand, blocked, samples, points: r.points.length}; })()`);
        if (!res.ok) { rows.push({ id, ok: false, reason: res.reason }); continue; }
        const sj = Math.floor((res.stand.x - g.ox) / g.cell), si = Math.floor((res.stand.z - g.oz) / g.cell);
        const opt = oracle(g, sIdx, si * g.w + sj);
        rows.push({ id, ok: true, length: res.length, opt, ratio: opt > 0.2 ? res.length / opt : 1, blocked: res.blocked, samples: res.samples, secs_ok: Math.abs(res.seconds - res.length / 1.4) < 1e-6 });
      }
      const okRows = rows.filter((r) => r.ok), failed = rows.filter((r) => !r.ok);
      const blocked = okRows.reduce((a, r) => a + r.blocked, 0), samples = okRows.reduce((a, r) => a + r.samples, 0), worst = Math.max(...okRows.map((r) => r.ratio));
      const pass = failed.length === 0 && blocked === 0 && worst <= 1.10 && okRows.every((r) => r.secs_ok);
      return { pass, value: `${okRows.length} routes (${ids.length} places x ${starts.length} starts), ${failed.length} failed; ${blocked} blocked of ${samples} samples; worst ${r3(worst)}x optimum; mean ${r3(okRows.reduce((a, r) => a + r.ratio, 0) / okRows.length)}x; walking time = length / 1.4 m/s`,
        metrics: { routes: okRows.length, failed: failed.length, blocked_samples: blocked, samples, worst_ratio: r3(worst), failures: failed.slice(0, 3) } };
    });
  });

  // ------------------------------------------------------------------------------------------------------ R2: jump
  T("R2", "jump lands on a walkable cell, on the nearest one (items stand back), facing the place within 2 degrees", async (c) => {
    return withBrowser(c.web, async (ctx) => {
      await ready(ctx);
      const places = await ctx.pg.eval("__ns.S.guide.places");
      if (!places.length) return { pass: null, value: "not run: scene has no places" };
      const meta = await ctx.pg.eval("({w: __ns.grid.w, h: __ns.grid.h, cell: __ns.grid.cell, ox: __ns.grid.ox, oz: __ns.grid.oz, data: Array.from(__ns.grid.data)})");
      const rows = [];
      for (const p of places) {
        const back = p.kind === "item" ? Math.max(0.8, Math.min(2.0, p.radius_m + 0.6)) : 0;
        await ctx.pg.eval(`__ns.S.guide.select(${JSON.stringify(p.id)}); __ns.S.guide.jump()`); await sleep(120);
        const ps = await pose(ctx.pg);
        const walkable = await ctx.pg.eval(`__ns.grid.walkable(${ps.x}, ${ps.z})`);
        // nearest walkable cell centre at least `back` from the place, by brute force
        let best = Infinity;
        for (let k = 0; k < meta.data.length; k++) if (meta.data[k] === 255) { const x = meta.ox + ((k % meta.w) + 0.5) * meta.cell, z = meta.oz + (Math.floor(k / meta.w) + 0.5) * meta.cell, d = Math.hypot(x - p.position[0], z - p.position[2]); if (d >= back - 1e-9 && d < best) best = d; }
        const d = Math.hypot(ps.x - p.position[0], ps.z - p.position[2]);
        const wantYaw = (Math.atan2(-(p.position[0] - ps.x), -(p.position[2] - ps.z)) * 180) / Math.PI;
        rows.push({ id: p.id, walkable, d: r3(d), nearest: r3(best), yaw_err: r3(angDiff(ps.yawDeg, wantYaw)), near_ok: Math.abs(d - best) < 0.02 });
      }
      const pass = rows.every((r) => r.walkable && r.near_ok && (r.d < 0.3 || r.yaw_err <= 2));
      return { pass, value: `${rows.length} jumps: ${rows.filter((r) => r.walkable).length} on walkable cells; distance to place matches the nearest allowed cell in ${rows.filter((r) => r.near_ok).length}; worst yaw error ${Math.max(...rows.map((r) => r.yaw_err))} deg`, metrics: { rows } };
    });
  });

  // ------------------------------------------------------------------------------------------------------ R3: search
  T("R3", "search box (key /): finds by name, alias and with one typo; nonsense finds nothing; Enter picks the top match", async (c) => {
    return withBrowser(c.web, async (ctx) => {
      await ready(ctx);
      if (!(await ctx.pg.eval("__ns.S.guide.places.length"))) return { pass: null, value: "not run: scene has no places" };
      const places = await ctx.pg.eval("__ns.S.guide.places.map((p) => ({ id: p.id, name: p.name, aliases: p.aliases || [] }))");
      const cases = places.some((p) => p.id === "wall-display") ? FIXTURE_SEARCH_CASES : searchCasesFor(places);
      const rows = [];
      for (const [q, want, kind] of cases) {
        await ctx.pg.eval("__ns.S.guide.clearSelection()");
        const ids = await search(ctx.pg, q);
        const got = await sel(ctx.pg);
        rows.push({ q, want, got, kind, ok: got === want, listed: ids.length });
      }
      const open0 = await ctx.pg.eval("!document.getElementById('srch').hidden");
      await ctx.pg.eval("__ns.S.guide.clearSelection()");
      await ctx.pg.tap("Slash/"); await sleep(100); await ctx.pg.type("zzzqqq"); await sleep(100);
      const none = await ctx.pg.eval("({n: document.querySelectorAll('#srch-list li[data-id]').length, msg: document.querySelector('#srch-list li.none')?.textContent})");
      await ctx.pg.tap("Enter"); await sleep(100);
      const stillNone = (await sel(ctx.pg)) === null;
      await ctx.pg.tap("Escape"); await sleep(100);
      const closed = await ctx.pg.eval("document.getElementById('srch').hidden");
      await ctx.pg.screenshot(path.join(c.shots, "r3_search.png"), { x: 560, y: 0, width: 720, height: 260 });
      const pass = rows.every((r) => r.ok) && none.n === 0 && stillNone && closed && !open0;
      return { pass, value: `${rows.filter((r) => r.ok).length}/${rows.length} queries found the right place (${rows.filter((r) => /typo/.test(r.kind)).length} with one typo); nonsense -> "${none.msg}"; Esc closes: ${closed}`, metrics: { rows, no_match_text: none.msg } };
    });
  });

  // ------------------------------------------------------------------------------------------------------ R4: follow
  T("R4", "follow mode (G then Space) walks the route to the place at walking speed, stays on walkable cells, faces the place; a move key stops it", async (c) => {
    return withBrowser(c.web, async (ctx) => {
      await ready(ctx);
      const places = await ctx.pg.eval("__ns.S.guide.places");
      if (!places.length) return { pass: null, value: "not run: scene has no places" };
      // the place with the longest route from the spawn
      const best = await ctx.pg.eval(`(() => { const G = __ns.S.guide; let b = null; for (const p of G.places) { G.select(p.id); G.routeOn = true; G.computeRoute(true); if (G.route.ok && (!b || G.route.length > b.len)) b = {id: p.id, len: G.route.length}; } G.clearSelection(); return b; })()`);
      await ctx.pg.eval("__ns.S.walker.setPose(__ns.S.data.spawn.position[0], __ns.S.data.spawn.position[2], 0, 0)"); await sleep(100);
      await search(ctx.pg, places.find((p) => p.id === best.id).name);
      await ctx.pg.tap("KeyG"); await sleep(200);
      const routeInfo = await ctx.pg.eval("({ok: __ns.S.guide.route.ok, len: __ns.S.guide.route.length, stand: __ns.S.guide.route.stand, sec: __ns.S.guide.route.seconds})");
      await ctx.pg.screenshot(path.join(c.shots, "r4_route.png"));
      const t0 = Date.now(); await ctx.pg.tap("Space"); await sleep(150);
      const started = await ctx.pg.eval("__ns.S.guide.following");
      let bad = 0, n = 0, maxOff = 0;
      while (Date.now() - t0 < 90000) {
        const s = await ctx.pg.eval("({f: __ns.S.guide.following, ok: __ns.grid.walkable(__ns.walker.x, __ns.walker.z), off: __ns.S.guide.fst ? __ns.S.guide.fst.off : 0})");
        n++; if (!s.ok) bad++; maxOff = Math.max(maxOff, s.off || 0);
        if (!s.f) break; await sleep(100);
      }
      const secs = (Date.now() - t0) / 1000;
      await sleep(200);
      const end = await pose(ctx.pg), pl = places.find((p) => p.id === best.id);
      const dEnd = Math.hypot(end.x - routeInfo.stand.x, end.z - routeInfo.stand.z);
      const standD = Math.hypot(routeInfo.stand.x - pl.position[0], routeInfo.stand.z - pl.position[2]);
      // follow turns to face the place as seen from the stand spot; the player stops up to 0.25 m from that spot, which at a stand-back of
      // 1.8 m alone is 8 degrees seen from the place, so the yaw is judged against the stand spot's direction
      const wantYaw = (Math.atan2(-(pl.position[0] - routeInfo.stand.x), -(pl.position[2] - routeInfo.stand.z)) * 180) / Math.PI;
      const yawErr = angDiff(end.yawDeg, wantYaw);
      const toast = await ctx.pg.eval("document.getElementById('toast').textContent");
      await ctx.pg.screenshot(path.join(c.shots, "r4_arrived.png"));
      // a move key hands control back
      await ctx.pg.eval("__ns.S.walker.setPose(__ns.S.data.spawn.position[0], __ns.S.data.spawn.position[2], 0, 0)"); await sleep(100);
      await ctx.pg.tap("KeyG"); await sleep(100); await ctx.pg.tap("KeyG"); await sleep(100);       // route off, on
      await ctx.pg.tap("Space"); await sleep(1200);
      const mid = await ctx.pg.eval("__ns.S.guide.following");
      await ctx.pg.key("KeyW"); await sleep(120); await ctx.pg.key("KeyW", false); await sleep(100);
      const after = await ctx.pg.eval("__ns.S.guide.following");
      const speed = routeInfo.len / secs;
      const pass = !!started && bad === 0 && dEnd < 0.4 && (standD < 0.3 || yawErr < 6) && speed > 0.8 && speed < 1.6 && mid === true && after === false && /Arrived/.test(toast);
      return { pass, value: `route ${r1(routeInfo.len)} m (${r1(routeInfo.sec)} s at 1.4 m/s) to "${best.id}"; walked it in ${r1(secs)} s (mean ${r1(speed * 10) / 10} m/s incl. start turn); ${bad} of ${n} samples off walkable cells; ended ${r3(dEnd)} m from the stand spot, ${r1(yawErr)} deg off the direction to the place from the stand spot; toast "${toast}"; W key stopped follow: ${mid === true && after === false}`,
        metrics: { place: best.id, route_m: r3(routeInfo.len), seconds: r1(secs), mean_mps: r3(speed), samples: n, blocked_samples: bad, max_off_route_m: r3(maxOff), end_error_m: r3(dEnd), yaw_error_deg: r1(yawErr) } };
    });
  });

  // ------------------------------------------------------------------------------------------------------ R5: route updates and no-path message
  T("R5", "route re-plans after the player leaves the path by > 0.5 m (not for small moves); no-path cases show a clear message", async (c) => {
    return withBrowser(c.web, async (ctx) => {
      await ready(ctx);
      const places = await ctx.pg.eval("__ns.S.guide.places");
      if (!places.length) return { pass: null, value: "not run: scene has no places" };
      await ctx.pg.eval("__ns.S.walker.setPose(__ns.S.data.spawn.position[0], __ns.S.data.spawn.position[2], 0, 0)"); await sleep(100);
      const id = places.find((p) => p.id === "door")?.id || places[0].id;
      await ctx.pg.eval(`__ns.S.guide.select(${JSON.stringify(id)})`); await ctx.pg.tap("KeyG"); await sleep(400);
      const r0 = await ctx.pg.eval("({pts: __ns.S.guide.route.points, id: (window.__rid = Math.random())})");
      // a small step along the route: no re-plan (the route object stays the same)
      await ctx.pg.eval("(() => { window.__r0 = __ns.S.guide.route; })()");
      const near = await ctx.pg.eval(`(() => { const w = __ns.walker, p = __ns.S.guide.route.points[1]; const d = Math.hypot(p[0] - w.x, p[1] - w.z), t = Math.min(0.3, d) / d; w.teleport(w.x + (p[0] - w.x) * t, w.z + (p[1] - w.z) * t); return d; })()`); await sleep(500);
      const same = await ctx.pg.eval("__ns.S.guide.route === window.__r0");
      // a jump 1.5 m off the path: re-planned from the new spot within a second
      const off = await ctx.pg.eval(`(() => { const G = __ns.S.guide, w = __ns.walker; const pts = G.route.points; const n = pts.length - 1; const mx = (pts[0][0] + pts[n][0]) / 2, mz = (pts[0][1] + pts[n][1]) / 2;
        // find a walkable cell 0.7-3 m off the route
        for (let r = 0.8; r < 3; r += 0.2) for (let a = 0; a < 6.28; a += 0.4) { const x = mx + r * Math.cos(a), z = mz + r * Math.sin(a); if (__ns.grid.walkable(x, z)) { let dmin = 1e9; for (let s = 0; s <= 1; s += 0.01) { const px = pts[0][0] + (pts[n][0] - pts[0][0]) * s, pz = pts[0][1] + (pts[n][1] - pts[0][1]) * s; dmin = Math.min(dmin, Math.hypot(px - x, pz - z)); } if (dmin > 0.7) { w.teleport(x, z); return {x, z, dmin}; } } }
        return null; })()`);
      await sleep(900);
      const re = await ctx.pg.eval(`(() => { const G = __ns.S.guide, w = __ns.walker; return {changed: G.route !== window.__r0, start: G.route.points[0], ok: G.route.ok, px: w.x, pz: w.z}; })()`);
      const startErr = Math.hypot(re.start[0] - re.px, re.start[1] - re.pz);
      // no path: wall off the floor, ask for a place beyond it. The fixture keeps its fixed numbers. Any other scene: the first place keeps
      // an island of cells within 0.45 m of its stand spot, every other cell within 2.2 m (plus the stand distance) of the place is blocked,
      // so the nearest cell the player can reach is more than 1.5 m farther from the place than the island (the viewer would otherwise
      // route to a "closest spot" within 1.5 m). The player stands on the walkable cell farthest from the place.
      const blocked = await ctx.pg.eval(`(() => { const g = __ns.grid, G = __ns.S.guide; const d = new Uint8Array(g.data);
        const fixture = G.places.some((p) => p.id === "coffee-counter");
        let id = "coffee-counter", from = [-3.5, 0.8];
        if (fixture) {
          for (let i = 0; i < g.h; i++) for (let j = 0; j < g.w; j++) { const x = g.ox + (j + .5) * g.cell; if (x > -0.5 && x < 2.9) d[i * g.w + j] = 0; }
        } else {
          const place = G.places[0], st = G.standFor(place); if (!st) return { skip: "no stand spot for " + place.id };
          id = place.id; let far = -1; from = null; const R = st.d + 2.2;
          for (let i = 0; i < g.h; i++) for (let j = 0; j < g.w; j++) {
            const x = g.ox + (j + .5) * g.cell, z = g.oz + (i + .5) * g.cell, rp = Math.hypot(x - place.position[0], z - place.position[2]), rs = Math.hypot(x - st.x, z - st.z);
            if (rs <= 0.45) continue;
            if (rp < R) d[i * g.w + j] = 0; else if (g.data[i * g.w + j] === 255 && rp > far) { far = rp; from = [x, z]; }
          }
          if (!from) return { skip: "no walkable cell outside the blocked disc" };
        }
        g.data = d;
        __ns.walker.teleport(from[0], from[1]); G.select(id); G.routeOn = false; G.toggleRoute(); return {ok: G.route.ok, reason: G.route.reason, id, card: document.getElementById('pl-info').textContent, bad: document.getElementById('pl-info').classList.contains('bad'), mesh: !!G.mesh, follow_btn_hidden: document.getElementById('pl-follow').hidden}; })()`);
      if (blocked.skip) return { pass: false, value: `walled-off floor test could not be set up: ${blocked.skip}` };
      const pass = same === true && re.changed === true && re.ok && startErr < 0.15 && off && blocked.ok === false && /No walking route/.test(blocked.reason || "") && /No walking route/.test(blocked.card) && blocked.bad && !blocked.mesh && blocked.follow_btn_hidden;
      return { pass, value: `0.3 m step along the route: route kept (${same}); ${off ? r1(off.dmin * 10) / 10 : "?"} m off the path: re-planned from the new spot (start error ${r3(startErr)} m); walled-off floor: "${blocked.reason}"`, metrics: { same_route_after_small_move: same, replanned_after_off_path: re.changed, start_error_m: r3(startErr), off_path_m: off && r3(off.dmin), no_path: blocked } };
    });
  });

  // ------------------------------------------------------------------------------------------------------ R6: measure tool
  T("R6", "measure tool (K): two clicks give the right distance on a known pair (floor plane), a splat-surface pair is self-consistent, scale interval shown", async (c) => {
    return withBrowser(c.web, async (ctx) => {
      await ready(ctx); await waitFullLoad(ctx.pg); await sleep(1200);
      const sc = await ctx.pg.eval("__ns.S.data.scale");
      const cam = await ctx.pg.eval("({fov: __ns.camera.fov, aspect: __ns.camera.aspect, w: innerWidth, h: innerHeight})");
      const setView = async () => { await ctx.pg.eval("__ns.setPose(__ns.S.data.spawn.position[0], 1.42, __ns.S.data.spawn.position[2], 25, -50)"); await sleep(450); };
      // expected floor point for a pixel, from the pose alone (independent of the viewer's matrices)
      const floorPoint = (p, px, py) => {
        const t = Math.tan((cam.fov * Math.PI) / 360), nx = (px / cam.w) * 2 - 1, ny = -((py / cam.h) * 2 - 1);
        let d = [nx * t * cam.aspect, ny * t, -1];
        const th = (p.pitchDeg * Math.PI) / 180, ps = (p.yawDeg * Math.PI) / 180;
        d = [d[0], d[1] * Math.cos(th) - d[2] * Math.sin(th), d[1] * Math.sin(th) + d[2] * Math.cos(th)];
        d = [d[0] * Math.cos(ps) + d[2] * Math.sin(ps), d[1], -d[0] * Math.sin(ps) + d[2] * Math.cos(ps)];
        const s = -p.y / d[1]; return [p.x + d[0] * s, 0, p.z + d[2] * s];
      };
      // (a) floor plane: raycast off, so the hit is the plane y = 0
      await ctx.pg.eval("__ns.S.guide.useSplats = false");
      await setView();
      const p = await pose(ctx.pg);
      const pxA = [480, 470], pxB = [820, 560];
      await ctx.pg.tap("KeyK"); await sleep(150);
      const toolOn = await ctx.pg.eval("({mode: __ns.S.guide.mode, banner: !document.getElementById('tool').hidden, msg: document.getElementById('tool-msg').textContent})");
      await ctx.pg.click(...pxA); await sleep(150); await ctx.pg.click(...pxB); await sleep(200);
      const m = await ctx.pg.eval("({a: __ns.S.guide.meas.a, b: __ns.S.guide.meas.b, msg: document.getElementById('tool-msg').textContent, res: __ns.S.guide.measureResult()})");
      const A = floorPoint(p, ...pxA), B = floorPoint(p, ...pxB);
      const want = Math.hypot(A[0] - B[0], A[2] - B[2]), got = m.res.d;
      const wantErr = (want * sc.pct_68) / 100;
      // finding 29: the range is an estimate from a prior unless scene.json says it was checked
      const labelOk = /m/.test(m.msg) && (sc.checked_against ? /checked/i.test(m.msg) && !/±/.test(m.msg)
        : sc.pct_68 > 0 ? m.msg.includes(`± ${wantErr.toFixed(2)} m`) && /68%/.test(m.msg) && /estimate/i.test(m.msg) && !/measured error we|from the scale check/i.test(m.msg) : /not checked/.test(m.msg));
      await ctx.pg.screenshot(path.join(c.shots, "r6_measure.png"));
      // (b) splat surfaces: same clicks with the raycast on
      await ctx.pg.eval("__ns.S.guide.useSplats = true");
      await ctx.pg.tap("Digit4"); await sleep(600);                                              // saved view 4 looks at the sideboard and the forest picture: both have splat surfaces
      pxA[0] = 640; pxA[1] = 413; pxB[0] = 208; pxB[1] = 376;
      await ctx.pg.click(...pxA); await sleep(600); await ctx.pg.click(...pxB); await sleep(600);
      const s2 = await ctx.pg.eval(`(() => { const G = __ns.S.guide, m = G.meas; if (!m.a || !m.b) return null; const r = G.measureResult(); const pa = G.project(...m.a.p), pb = G.project(...m.b.p);
        return {src: [m.a.source, m.b.source], d: r.d, direct: Math.hypot(m.a.p[0] - m.b.p[0], m.a.p[1] - m.b.p[1], m.a.p[2] - m.b.p[2]), ea: Math.hypot(pa.x - ${pxA[0]}, pa.y - ${pxA[1]}), eb: Math.hypot(pb.x - ${pxB[0]}, pb.y - ${pxB[1]}), a: m.a.p, b: m.b.p}; })()`);
      // Esc leaves the tool
      await ctx.pg.tap("Escape"); await sleep(100);
      const off = await ctx.pg.eval("({mode: __ns.S.guide.mode, banner: document.getElementById('tool').hidden})");
      const pass = toolOn.mode === "measure" && toolOn.banner && Math.abs(got - want) < 0.01 && labelOk && !!s2 && s2.src.every((x) => x === "splat") && Math.abs(s2.d - s2.direct) < 1e-9 && s2.ea < 1.5 && s2.eb < 1.5 && off.mode === null && off.banner;
      return { pass, value: `floor pair: expected ${r3(want)} m, tool says ${r3(got)} m (error ${r3(Math.abs(got - want) * 100)} cm); label "± ${wantErr.toFixed(2)} m" and 68% interval shown: ${labelOk}; splat pair: ${s2 ? `${s2.src.join("+")} hits, ${r3(s2.d)} m, reprojection error ${r3(s2.ea)}/${r3(s2.eb)} px` : "no hit"}; Esc ends the tool: ${off.mode === null}`,
        metrics: { floor: { expected_m: r3(want), measured_m: r3(got), error_cm: r3(Math.abs(got - want) * 100) }, scale_pct_68: sc.pct_68, label: m.msg, splat: s2 } };
    });
  });

  // ------------------------------------------------------------------------------------------------------ R7: place editor and bundle --places
  T("R7", "place editor (P): click, name, add; search finds it; Copy places JSON; the list survives a reload; bundle.py --places merges it", async (c) => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ns-r7-"));
    try {
      return await withBrowser(c.web, async (ctx) => {
        await ready(ctx); await waitFullLoad(ctx.pg); await sleep(800);
        await ctx.pg.eval("localStorage.clear()");
        const n0 = await ctx.pg.eval("__ns.S.guide.places.length");
        await ctx.pg.tap("KeyP"); await sleep(150);
        const on = await ctx.pg.eval("({mode: __ns.S.guide.mode, banner: !document.getElementById('tool').hidden})");
        await ctx.pg.eval("__ns.setPose(__ns.S.data.spawn.position[0], 1.42, __ns.S.data.spawn.position[2], 25, -20)"); await sleep(400);
        await ctx.pg.click(640, 420); await sleep(700);
        const form = await ctx.pg.eval("({open: !document.getElementById('padd').hidden, pos: __ns.S.guide.pending && __ns.S.guide.pending.pos, src: __ns.S.guide.pending && __ns.S.guide.pending.source})");
        await ctx.pg.type("Test stool"); await sleep(80);
        await ctx.pg.eval("document.getElementById('padd-alias').focus()"); await ctx.pg.type("small seat, footrest"); await sleep(80);
        await ctx.pg.eval("document.querySelector('#padd-kind [data-k=item]').click()");
        await ctx.pg.tap("Enter"); await sleep(250);
        const added = await ctx.pg.eval("({n: __ns.S.guide.places.length, sel: __ns.S.guide.sel && __ns.S.guide.sel.id, p: __ns.S.guide.local[0], stored: localStorage.getItem(__ns.S.guide.key)})");
        // searchable by a second alias
        await ctx.pg.tap("Escape"); await sleep(100);                                          // leaves add mode
        await ctx.pg.eval("__ns.S.guide.clearSelection()");
        await search(ctx.pg, "footrest");
        const found = await sel(ctx.pg);
        // copy JSON (clipboard may be refused in headless: the viewer then shows a box; lastJson is set either way)
        await ctx.pg.tap("KeyP"); await sleep(100);
        await ctx.pg.eval("document.getElementById('tool-copy').click()"); await sleep(400);
        const json = await ctx.pg.eval("__ns.S.guide.lastJson");
        const list = JSON.parse(json);
        const boxText = await ctx.pg.eval("document.getElementById('jsonbox').hidden ? null : document.getElementById('jsonbox-text').value");
        await ctx.pg.eval("document.getElementById('jsonbox').hidden = true");
        fs.writeFileSync(path.join(tmp, "places.json"), json);
        // reload: the local place is still there
        await openViewer(ctx, "", { start: true }); await sleep(300);
        const again = await ctx.pg.eval("({n: __ns.S.guide.places.length, has: __ns.S.guide.places.some(p => p.name === 'Test stool')})");
        // bundle.py --places
        const out = path.join(tmp, "web");
        const b = spawnSync("python3", [BUNDLE, "--run", FIXTURE_RUN, "--out", out, "--name", "r7", "--kind", "fixture", "--places", path.join(tmp, "places.json"), "--verify-dir", path.join(tmp, "v"), "--json"], { encoding: "utf8" });
        let res = {}; try { res = JSON.parse(b.stdout); } catch {}
        const sj = b.status === 0 ? JSON.parse(fs.readFileSync(path.join(out, "scenes/r7/scene.json"), "utf8")) : null;
        const merged = sj?.places?.find((p) => p.name === "Test stool");
        const mergedOk = !!merged && merged.kind === "item" && Math.abs(merged.position[0] - added.p.position[0]) < 1e-3 && merged.aliases.includes("footrest") && sj.places.length === n0 + 1;
        const pass = on.mode === "add" && on.banner && form.open && Array.isArray(form.pos) && added.n === n0 + 1 && added.sel === "test-stool" && added.p.aliases.length === 2 && !!added.stored && found === "test-stool" && list.length === n0 + 1 && !!list.find((p) => p.id === "test-stool")
          && again.has && again.n === n0 + 1 && b.status === 0 && res.ok && mergedOk;
        return { pass, value: `P opened the tool; click picked a ${form.src} point at [${(form.pos || []).map((v) => r1(v * 10) / 10).join(", ")}]; "Test stool" added (${n0} -> ${added.n} places), found by alias "footrest": ${found === "test-stool"}; JSON has ${list.length} places${boxText ? " (clipboard refused: text box shown)" : " (clipboard)"}; after reload ${again.n} places; bundle.py --places: ${b.status === 0 ? `valid, scene.json has ${sj.places.length} places, merged place matches` : "FAILED " + (b.stderr || "").slice(-200)}`,
          metrics: { places_before: n0, places_after: added.n, picked_source: form.src, json_places: list.length, bundle_ok: res.ok, bundle_warnings: res.warnings } };
      });
    } finally { fs.rmSync(tmp, { recursive: true, force: true }); }
  });

  // ------------------------------------------------------------------------------------------------------ R8: minimap
  T("R8", "minimap: place markers drawn; clicking a marker selects the place; the route is drawn on the map", async (c) => {
    return withBrowser(c.web, async (ctx) => {
      await ready(ctx); await sleep(300);
      if (!(await ctx.pg.eval("__ns.S.guide.places.length"))) return { pass: null, value: "not run: scene has no places" };
      const count = () => ctx.pg.eval(`(() => { const c = document.getElementById('map'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let pink = 0, teal = 0; for (let i = 0; i < d.length; i += 4) { if (d[i] > 190 && d[i + 1] < 115 && d[i + 2] > 70 && d[i + 2] < 160) pink++; if (d[i] < 110 && d[i + 1] > 215 && d[i + 2] > 190 && d[i+3] === 255) teal++; } return {pink, teal}; })()`);
      const a = await count();
      // click the marker of a place far from the spawn
      const t = await ctx.pg.eval(`(() => { const G = __ns.S.guide, m = __ns.S.minimap, w = __ns.walker; const r = document.getElementById('map').getBoundingClientRect(), k = r.width / m.c.width;
        let best = null, bd = 0; for (const p of G.places) { const d = Math.hypot(p.position[0] - w.x, p.position[2] - w.z); if (d > bd) { bd = d; best = p; } }
        const [cx, cz] = m.toCanvas(best.position[0], best.position[2]); return {id: best.id, x: r.left + cx * k, y: r.top + cz * k}; })()`);
      const before = await pose(ctx.pg);
      await ctx.pg.click(t.x, t.y); await sleep(250);
      const picked = await sel(ctx.pg), after = await pose(ctx.pg);
      await ctx.pg.tap("KeyG"); await sleep(500);
      const b = await count();
      const moved = Math.hypot(after.x - before.x, after.z - before.z);
      const pass = a.pink >= 20 && picked === t.id && moved < 0.01 && b.teal > a.teal + 40;
      return { pass, value: `place markers: ${a.pink} marker pixels; click on "${t.id}" marker selected "${picked}" (player moved ${r3(moved)} m); route drawn: map route-colour pixels ${a.teal} -> ${b.teal}`, metrics: { marker_pixels: a.pink, route_pixels_before: a.teal, route_pixels_after: b.teal } };
    });
  });

  // ------------------------------------------------------------------------------------------------------ R9: start card, help, URL flags
  T("R9", "start card and ? help list the new keys; ?place=&route=1 opens with the route drawn; no console errors in the whole flow", async (c) => {
    // Place ids come from the scene itself, so the test works on any bundle, not only the fixture.
    const places = (sceneJson(c.web).scene.places || []).map((p) => p.id);
    const [first, second] = [places[0], places[1] || places[0]];
    return withBrowser(c.web, async (ctx) => {
      await openViewer(ctx, first ? `?place=${encodeURIComponent(first)}&route=1` : "", {});
      const card = await ctx.pg.eval("document.getElementById('start').innerText");
      await ctx.pg.screenshot(path.join(c.shots, "r9_start_card.png"));
      await ctx.pg.eval("document.getElementById('s-go').click()"); await sleep(300);
      await ctx.pg.tap("Slash"); await sleep(150);
      const help = await ctx.pg.eval("document.getElementById('help').innerText");
      await ctx.pg.tap("Escape"); await sleep(100);
      const flag = await ctx.pg.eval("({sel: __ns.S.guide.sel && __ns.S.guide.sel.id, route: __ns.S.guide.route && __ns.S.guide.route.ok, on: __ns.S.guide.routeOn, mesh: !!__ns.S.guide.mesh})");
      const errs0 = consoleErrors(ctx.pg);
      const ui = /Search|search/.test(card) && /round 2 preview/i.test(card) && /measure/i.test(card) && /search rooms/.test(help) && /measure/.test(help) && /J G Space/.test(help) && /places JSON/.test(help);
      if (!first) {
        // No places in this scene: the viewer hides the search lines (.needs-places), so check that they are hidden,
        // that the measure tool is still listed, and that the console is clean. The route part cannot run.
        const hidden = await ctx.pg.eval("[...document.querySelectorAll('.needs-places')].every((e) => e.hidden)");
        const okNp = hidden && /measure/i.test(card) && /measure/i.test(help);
        return { pass: okNp && errs0.length === 0, value: `no places: search lines hidden ${hidden}; measure listed on card and help ${/measure/i.test(card) && /measure/i.test(help)}; ${errs0.length} console errors; route part not run`, metrics: { console_errors: errs0 } };
      }
      let swapped = { mesh: false, route: null, on: false };
      if (second !== first) {
        await ctx.pg.eval(`__ns.S.guide.select(${JSON.stringify(second)})`);
        swapped = await ctx.pg.eval("({mesh: !!__ns.S.guide.mesh, route: __ns.S.guide.route, on: __ns.S.guide.routeOn})");      // picking another place clears the old route
      }
      const errs = consoleErrors(ctx.pg);
      const pass = ui && flag.sel === first && flag.route === true && flag.on && flag.mesh && !swapped.mesh && swapped.route === null && !swapped.on && errs.length === 0;
      return { pass, value: `start card mentions search, Round 2 preview and measure; help lists / J G Space K P; ?place=${first}&route=1 -> route drawn (${flag.route}); ${errs.length} console errors`, metrics: { console_errors: errs, flag } };
    });
  });

  // ------------------------------------------------------------------------------------------------------ R10: phone
  T("R10", "phone (390x844 touch): the search button is usable, results are tappable (>= 40 px), card and buttons fit the screen, Route works by touch", async (c) => {
    const places = sceneJson(c.web).scene.places || [];
    if (!places.length) return { pass: null, value: "not run: scene has no places" };
    // Type the start of a real place name from this scene (the fixture used "coffee").
    const query = String(places[0].name || places[0].id).slice(0, 6).toLowerCase();
    return withBrowser(c.web, async (ctx) => {
      await openViewer(ctx, "", {});
      const b = await ctx.pg.eval("(() => { const r = document.getElementById('s-go').getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; })()");
      await ctx.pg.touch("touchStart", [b]); await sleep(60); await ctx.pg.touch("touchEnd", []); await sleep(500);
      const rect = (id) => ctx.pg.eval(`(() => { const e = document.getElementById('${id}'); if (!e || e.hidden) return null; const r = e.getBoundingClientRect(); return {x: r.x, y: r.y, w: r.width, h: r.height, r: r.right, b: r.bottom}; })()`);
      const btn = await rect("srch-btn");
      if (!btn) return { pass: false, value: "search button not visible on the phone" };
      await ctx.pg.touch("touchStart", [[btn.x + btn.w / 2, btn.y + btn.h / 2]]); await sleep(60); await ctx.pg.touch("touchEnd", []); await sleep(300);
      const open = await rect("srch");
      const focused = await ctx.pg.eval("document.activeElement && document.activeElement.id");
      await ctx.pg.type(query); await sleep(200);
      const row = await ctx.pg.eval("(() => { const li = document.querySelector('#srch-list li[data-id]'); if (!li) return null; const r = li.getBoundingClientRect(); return {x: r.x + r.width / 2, y: r.y + r.height / 2, h: r.height, id: li.dataset.id}; })()");
      await ctx.pg.screenshot(path.join(c.shots, "r10_phone_search.png"));
      await ctx.pg.touch("touchStart", [[row.x, row.y]]); await sleep(60); await ctx.pg.touch("touchEnd", []); await sleep(300);
      const card = await rect("place"), picked = await sel(ctx.pg);
      const rb = await rect("pl-route");
      await ctx.pg.touch("touchStart", [[rb.x + rb.w / 2, rb.y + rb.h / 2]]); await sleep(60); await ctx.pg.touch("touchEnd", []); await sleep(500);
      const routeOk = await ctx.pg.eval("!!(__ns.S.guide.route && __ns.S.guide.route.ok)");
      const fb = await rect("pl-follow"); const jb = await rect("pl-jump");
      await ctx.pg.screenshot(path.join(c.shots, "r10_phone_route.png"));
      const W = 390, H = 844;
      const fits = (r) => r && r.x >= -0.5 && r.r <= W + 0.5 && r.y >= -0.5 && r.b <= H + 0.5;
      const pass = !!open && fits(open) && focused === "srch-in" && row && row.h >= 40 && picked === row.id && fits(card) && fits(rb) && fits(jb) && fb && fits(fb) && rb.h >= 36 && routeOk;
      return { pass, value: `search button ${Math.round(btn.w)}x${Math.round(btn.h)} px; box opens and takes focus: ${focused === "srch-in"}; result row ${Math.round(row.h)} px high; tapped "${picked}"; card ${Math.round(card.w)} px wide fits; Route button ${Math.round(rb.h)} px high; route by touch: ${routeOk}`,
        metrics: { search_button: btn, result_row_h: row.h, card, buttons: { route: rb, jump: jb, follow: fb } } };
    }, { page: { width: 390, height: 844, mobile: true, touch: true, dpr: 2 }, chrome: { width: 390, height: 844 } });
  });
}
