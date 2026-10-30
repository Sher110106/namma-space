// Viewer acceptance tests H1-H9 (FINAL_PLAN WP5) plus checks for the P0/P1 features.
// Each test returns {pass, value, detail?, metrics?}; run.mjs writes them to 90_verify/verify.json.
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { startServer } from "./lib/server.mjs";
import { withBrowser, openViewer, waitFullLoad, sceneJson, consoleErrors, sleep } from "./lib/env.mjs";
import { registerGuideTests } from "./tests_guide.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const BUNDLE = path.join(HERE, "../tools/bundle.py");
const ms = (v) => `${v.toFixed(0)} ms`;
const f1 = (v) => (Math.round(v * 10) / 10).toString();

export const tests = [];
const T = (id, name, fn, opts = {}) => tests.push({ id, name, fn, ...opts });

// ---------------------------------------------------------------------------------------------------------- H1
T("H1", "first splat frame <= 5 s from localhost, cold cache; 0 console errors; 0 foreign-origin requests", async (c) => {
  return withBrowser(c.web, async (ctx) => {
    await ctx.pg.send("Network.setCacheDisabled", { cacheDisabled: true });
    const r = await openViewer(ctx, "", {});
    await sleep(800);
    const origin = new URL(ctx.srv.origin).origin;
    const foreign = ctx.pg.requests.map((q) => q.url).filter((u) => !/^(data|blob|about):/.test(u) && new URL(u).origin !== origin);
    const errs = consoleErrors(ctx.pg);
    await ctx.pg.screenshot(path.join(c.shots, "h1_first_frame.png"));
    const full = await ctx.pg.eval("({loaded: __ns.loadedMs, log: __ns.log})");
    const pass = r.ff != null && r.ff <= 5000 && errs.length === 0 && foreign.length === 0;
    return { pass, value: `first frame ${(r.ff / 1000).toFixed(2)} s; ${errs.length} console errors; ${foreign.length} foreign requests`,
      metrics: { first_frame_s: +(r.ff / 1000).toFixed(3), full_loaded_s: full.loaded ? +(full.loaded / 1000).toFixed(3) : null, console_errors: errs, foreign_requests: foreign, n_requests: ctx.pg.requests.length }, detail: full.log.join(" | ") };
  }, { chrome: {} });
});

// ---------------------------------------------------------------------------------------------------------- H2
async function dragTurn(pg, deg) {
  const px = Math.round((Math.abs(deg) * Math.PI / 180) / 0.0034), dir = deg > 0 ? -1 : 1;   // turning left = pointer moves left
  const x0 = 640, y0 = 420;
  await pg.mouse("mousePressed", x0, y0);
  for (let i = 1; i <= 5; i++) await pg.mouse("mouseMoved", x0 + (dir * px * i) / 5, y0, { buttons: 1 });
  await pg.mouse("mouseReleased", x0 + dir * px, y0);
}
T("H2", "scripted 20 s walk with SD: mean fps >= 55, p95 frame <= 20 ms (vsync-capped method)", async (c) => {
  return withBrowser(c.web, async (ctx) => {
    await openViewer(ctx, "?q=sd", { start: true });
    await waitFullLoad(ctx.pg);
    await sleep(2500);                                      // let the sort settle, no screenshots while measuring
    const info = await ctx.pg.eval("({gpu: __ns.gpu, splats: __ns.S.splatCount, level: __ns.governor.level, mb: __ns.S.loadedBytes / 1e6, ua: navigator.userAgent})");
    const load0 = os.loadavg()[0];
    await ctx.pg.eval("__ns.S.stats.startRun()");
    await ctx.pg.key("KeyW");
    const t0 = Date.now(); let k = 0;
    const turns = [40, -25, 55, 30, -45, 35, 50, -30, 40, 25, -35, 45, 30, -20];
    while (Date.now() - t0 < 20000) { await sleep(1300); await dragTurn(ctx.pg, turns[k++ % turns.length]); }
    await ctx.pg.key("KeyW", false);
    const r = await ctx.pg.eval("__ns.S.stats.endRun()");
    const load1 = os.loadavg()[0];
    const gov = await ctx.pg.eval("({hist: __ns.governor.history.map(h => h.step.name + ' ' + h.why), level: __ns.governor.level, pose: __ns.pose()})");
    const pass = r.fps_mean >= 55 && r.p95_ms <= 20;
    return { pass, value: `${f1(r.fps_mean)} fps mean, p95 ${f1(r.p95_ms)} ms, p99 ${f1(r.p99_ms)} ms, max ${f1(r.max_ms)} ms over ${f1(r.secs)} s (${r.frames} frames); ${info.splats} splats`,
      metrics: { ...r, gpu: info.gpu, splats: info.splats, mb: +info.mb.toFixed(2), governor_steps: gov.hist, final_level: gov.level, browser: ctx.br.version, mac_loadavg_1min: { before: +load0.toFixed(1), after: +load1.toFixed(1) }, cpus: os.cpus().length }, detail: `walk ended at ${JSON.stringify(gov.pose)}` };
  });
});

// ---------------------------------------------------------------------------------------------------------- H3
T("H3", "200 random 5 s walks + 16 headings x 6 s: 0 positions in blocked cells; >= 8 wall stops; >= 12 moves >= 0.5 m", async (c) => {
  return withBrowser(c.web, async (ctx) => {
    await openViewer(ctx, "", { start: true });
    const rw = await ctx.pg.eval("__ns.sim.randomWalks({n: 200, seconds: 5, seed: 7})");
    const hp = await ctx.pg.eval("__ns.sim.headingPushes({headings: 16, seconds: 6})");
    const fr = await ctx.pg.eval("__ns.sim.frameRate()");
    // the same through the real keyboard path: hold W at four headings for 4 s, sample cells every 100 ms
    let live = 0, liveBad = 0;
    for (const yaw of [0, 90, 180, 270]) {
      await ctx.pg.eval(`(() => { const s = __ns.data?.spawn; })(); __ns.setPose(__ns.S.data.spawn.position[0], __ns.S.data.spawn.position[1], __ns.S.data.spawn.position[2], ${yaw}, 0)`);
      await sleep(150);
      await ctx.pg.key("KeyW");
      for (let i = 0; i < 40; i++) {
        await sleep(100);
        const ok = await ctx.pg.eval("(() => { const p = __ns.pose(); return __ns.grid.walkable(p.x, p.z); })()");
        live++; if (!ok) liveBad++;
      }
      await ctx.pg.key("KeyW", false);
    }
    const spread = Math.max(...Object.values(fr)) - Math.min(...Object.values(fr));
    const violations = rw.violations + hp.violations + liveBad;
    const pass = violations === 0 && hp.stopped >= 8 && hp.movedHalfMetre >= 12;
    return { pass, value: `${violations} blocked-cell samples of ${rw.samples + hp.samples + live}; wall stops ${hp.stopped}/16; moves >= 0.5 m ${hp.movedHalfMetre}/16`,
      metrics: { walks: rw.walks, walk_samples: rw.samples, walk_violations: rw.violations, blocked_frames: rw.blockedFrames, pushes: 16, push_samples: hp.samples, push_violations: hp.violations, stopped: hp.stopped, moved_half_metre: hp.movedHalfMetre,
        live_samples: live, live_violations: liveBad, frame_rate_distance_1s: fr, frame_rate_spread_m: +spread.toFixed(3), push_detail: hp.pushes }, detail: rw.firstViolations?.length ? `first violations ${JSON.stringify(rw.firstViolations)}` : "" };
  });
});

// ---------------------------------------------------------------------------------------------------------- H4
T("H4", "browser render vs undistorted photo: PSNR >= 18 dB at 3 poses and >= 3 dB above the same poses turned 10 deg", async (c) => {
  const pairsFile = c.pairs || null;
  return withBrowser(c.web, async (ctx) => {
    await openViewer(ctx, "", { start: true });
    await waitFullLoad(ctx.pg);
    const { scene } = sceneJson(c.web, c.scene);
    let pairs, base;
    if (pairsFile) { pairs = JSON.parse(fs.readFileSync(pairsFile, "utf8")); base = ctx.srv.origin + "/__pairs/"; }
    else { pairs = scene.heldout.slice(1, 4); base = ctx.srv.origin + `/scenes/${scene.name}/`; }
    if (!pairs.length) return { pass: null, value: "not run: scene has no pose/photo pairs" };
    const rows = [];
    for (const p of pairs.slice(0, 3)) {
      const h = { ...p, photo: new URL(p.photo, base).href };
      const a = await ctx.pg.eval(`__ns.psnrAgainstPhoto(${JSON.stringify(h)}, {})`);
      const b = await ctx.pg.eval(`__ns.psnrAgainstPhoto(${JSON.stringify(h)}, {yawOffsetDeg: 10})`);
      rows.push({ id: p.id, psnr: +a.psnr.toFixed(2), psnr_rot10: +b.psnr.toFixed(2), gain: +(a.psnr - b.psnr).toFixed(2) });
    }
    const pass = rows.every((r) => r.psnr >= 18 && r.gain >= 3);
    return { pass, value: rows.map((r) => `${r.id}: ${r.psnr} dB (rot10 ${r.psnr_rot10}, +${r.gain})`).join("; "), metrics: { poses: rows, source: pairsFile ? "training poses (pairs file)" : "held-out pairs from scene.json" } };
  }, { server: { extraMounts: pairsFile ? [{ prefix: "/__pairs/", dir: path.dirname(pairsFile) }] : [] } });
});

T("H4b", "browser render vs gsplat render of the same file: PSNR >= 25 dB (needs reference renders)", async (c) => {
  if (!c.parity) return { pass: null, value: "not run: no gsplat reference renders supplied (--parity DIR with parity.json)" };
  return withBrowser(c.web, async (ctx) => {
    await openViewer(ctx, "", { start: true });
    await waitFullLoad(ctx.pg);
    const list = JSON.parse(fs.readFileSync(path.join(c.parity, "parity.json"), "utf8"));
    const rows = [];
    for (const p of list.slice(0, 3)) {
      const h = { ...p, photo: ctx.srv.origin + "/__parity/" + p.render };
      const a = await ctx.pg.eval(`__ns.psnrAgainstPhoto(${JSON.stringify(h)}, {})`);
      rows.push({ id: p.id, psnr: +a.psnr.toFixed(2) });
    }
    return { pass: rows.every((r) => r.psnr >= 25), value: rows.map((r) => `${r.id}: ${r.psnr} dB`).join("; "), metrics: { poses: rows } };
  }, { server: { extraMounts: [{ prefix: "/__parity/", dir: c.parity }] } });
});

// ---------------------------------------------------------------------------------------------------------- H5
T("H5", "touch path moves the player under mobile emulation (left thumb walks, right thumb looks)", async (c) => {
  return withBrowser(c.web, async (ctx) => {
    await openViewer(ctx, "", {});
    const b = await ctx.pg.eval("(() => { const r = document.getElementById('s-go').getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; })()");
    await ctx.pg.touch("touchStart", [b]); await sleep(60); await ctx.pg.touch("touchEnd", []); await sleep(400);
    const started = await ctx.pg.eval("__ns.S.started");
    // face the direction with the most room
    const best = await ctx.pg.eval("(() => { const r = __ns.sim.headingPushes({headings: 16, seconds: 3}); const b = r.pushes.reduce((a, p) => p.dist > a.dist ? p : a); __ns.setPose(__ns.S.data.spawn.position[0], __ns.S.data.spawn.position[1], __ns.S.data.spawn.position[2], b.heading, 0); return b; })()");
    await sleep(200);
    const p0 = await ctx.pg.eval("__ns.pose()");
    await ctx.pg.touch("touchStart", [[160, 300]]);
    for (let i = 1; i <= 12; i++) { await sleep(110); await ctx.pg.touch("touchMove", [[160, 300 - 6 * i]]); }
    const p1 = await ctx.pg.eval("__ns.pose()");
    await ctx.pg.touch("touchEnd", []);
    await sleep(300);
    const yaw0 = (await ctx.pg.eval("__ns.pose()")).yawDeg;
    await ctx.pg.touch("touchStart", [[520, 120]]);
    for (let i = 1; i <= 8; i++) { await sleep(40); await ctx.pg.touch("touchMove", [[520 + 10 * i, 120]]); }
    await ctx.pg.touch("touchEnd", []);
    await sleep(100);
    const yaw1 = (await ctx.pg.eval("__ns.pose()")).yawDeg;
    const moved = Math.hypot(p1.x - p0.x, p1.z - p0.z);
    const turned = Math.abs(((yaw1 - yaw0 + 540) % 360) - 180);
    const dyaw = Math.abs(yaw1 - yaw0);
    const pass = !!started && moved >= 0.5 && dyaw >= 5;
    return { pass, value: `moved ${moved.toFixed(2)} m with the left thumb; turned ${dyaw.toFixed(1)} deg with the right; start button tapped: ${started}`, metrics: { moved_m: +moved.toFixed(3), turned_deg: +dyaw.toFixed(1), heading_deg: best.heading, viewport: "844x390 mobile, touch emulation" } };
  }, { page: { width: 844, height: 390, mobile: true, touch: true, dpr: 2 }, chrome: { width: 844, height: 390 } });
});

// ---------------------------------------------------------------------------------------------------------- H6
T("H6", "?stress=1 (pixel ratio 2): the governor lowers resolution within 3 s", async (c) => {
  return withBrowser(c.web, async (ctx) => {
    const r = await openViewer(ctx, "?stress=1", {});
    const pr0 = await ctx.pg.eval("__ns.S.pr");
    let t = null;
    const t0 = Date.now();
    while (Date.now() - t0 < 9000) {
      t = await ctx.pg.eval("({n: __ns.governor.history.length, pr: __ns.S.pr, h0: __ns.governor.history[0] ? __ns.governor.history[0].t : null, ff: __ns.firstFrameMs})");
      if (t.n > 0) break;
      await sleep(100);
    }
    if (!t || t.n === 0) return { pass: false, value: `no governor step in 9 s at pixel ratio ${pr0}`, metrics: { pr_start: pr0 } };
    const dt = t.h0 - t.ff;
    const hist = await ctx.pg.eval("__ns.governor.history.slice(0, 4).map(h => ({t: Math.round(h.t), step: h.step.name, pr: h.step.pixelRatio, why: h.why}))");
    await sleep(3500);
    const prs = await ctx.pg.eval("({pr: __ns.S.pr, level: __ns.governor.level, name: __ns.governor.step.name})");
    return { pass: pr0 === 2 && t.pr < 2 && dt <= 3000, value: `pixel ratio ${pr0} -> ${t.pr} ${ms(dt)} after the first splat frame; settled at ${prs.name} (${prs.pr})`, metrics: { pr_start: pr0, pr_first_step: t.pr, first_step_after_ms: Math.round(dt), settled: prs, history: hist } };
  });
});

// ---------------------------------------------------------------------------------------------------------- H7
T("H7", "?asset= loads the splat file from a second local origin (CORS), and falls back when that origin has no CORS", async (c) => {
  const { dir, scene } = sceneJson(c.web, c.scene);
  const srvB = await startServer([{ prefix: "/", dir }], { cors: true });
  const srvC = await startServer([{ prefix: "/", dir }], { cors: false });
  try {
    return await withBrowser(c.web, async (ctx) => {
      const urlB = `${srvB.origin}/${scene.splats.sd.url}`;
      const r = await openViewer(ctx, `?asset=${encodeURIComponent(urlB)}`, {});
      await waitFullLoad(ctx.pg);
      const info = await ctx.pg.eval("({kind: __ns.S.active.userData.kind, n: __ns.S.splatCount, urls: [...__ns.S.bytesByUrl.keys()], err: __ns.error})");
      const gotB = srvB.log.some((l) => l.path.endsWith(".spz"));
      const errs = consoleErrors(ctx.pg);
      const posOk = info.kind === "sd" && info.urls.some((u) => u.startsWith(srvB.origin));
      // negative: the second origin sends no CORS header, so the browser blocks it; the viewer must still end on the scene's own file
      const pg2 = await ctx.br.newPage();
      await pg2.goto(ctx.url(`?asset=${encodeURIComponent(`${srvC.origin}/${scene.splats.sd.url}`)}`));
      await pg2.waitFor("window.__ns && (window.__ns.ready || window.__ns.error)", 30000, 100);
      let fell = false;
      try { await pg2.waitFor("__ns.S.fullLoaded === true && __ns.S.active.userData.kind === 'sd'", 30000, 200); fell = true; } catch {}
      const urls2 = fell ? await pg2.eval("[...__ns.S.bytesByUrl.keys()]") : [];
      await pg2.close();
      return { pass: posOk && gotB && fell, value: `cross-origin load ${posOk && gotB ? "ok" : "FAILED"} (${info.n} splats from ${new URL(urlB).host}); no-CORS origin: ${fell ? "fell back to own file" : "no fallback"}`,
        metrics: { cors_origin_requests: srvB.log.length, console_errors: errs.length, fallback_urls: urls2, first_frame_s: +(r.ff / 1000).toFixed(2) }, detail: errs.slice(0, 3).join(" | ") };
    });
  } finally { await srvB.close(); await srvC.close(); }
});

// ---------------------------------------------------------------------------------------------------------- H8
function py(args) { return spawnSync("python3", [BUNDLE, ...args], { encoding: "utf8" }); }
T("H8", "bundle.py output validates: required keys, files present, sizes match; corrupted bundles are rejected", async (c) => {
  const sj = sceneJson(c.web, c.scene);
  const base = py(["--validate", c.web, "--json"]);
  const res = JSON.parse(base.stdout || "{}");
  const okBase = base.status === 0 && res.ok;
  // mutate copies of the scene folder and expect each to fail
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ns-h8-"));
  const muts = [];
  const mutate = (name, fn) => {
    const d = path.join(tmp, name);
    fs.mkdirSync(path.join(d, "scenes"), { recursive: true });
    for (const f of ["index.html", "js", "css", "vendor"]) fs.cpSync(path.join(c.web, f), path.join(d, f), { recursive: true });
    fs.cpSync(path.join(c.web, "scenes"), path.join(d, "scenes"), { recursive: true });
    const sd = path.join(d, "scenes", sj.name);
    fn(sd, path.join(sd, "scene.json"));
    const r = py(["--validate", d, "--json"]);
    const rr = JSON.parse(r.stdout || "{}");
    muts.push({ name, rejected: r.status !== 0 && rr.ok === false, first_error: rr.errors?.[0] });
  };
  const edit = (fn) => (sd, sjp) => { const j = JSON.parse(fs.readFileSync(sjp, "utf8")); fn(j); fs.writeFileSync(sjp, JSON.stringify(j)); };
  mutate("missing_key", (sd, p) => edit((j) => delete j.spawn)(sd, p));
  mutate("bytes_mismatch", (sd, p) => fs.appendFileSync(path.join(sd, sj.scene.splats.sd.url), Buffer.alloc(100)));
  mutate("grid_missing", (sd) => fs.rmSync(path.join(sd, "grid.png")));
  mutate("wrong_count", (sd, p) => edit((j) => (j.splats.sd.count += 1))(sd, p));
  mutate("grid_dims", (sd, p) => edit((j) => (j.grid.width += 1))(sd, p));
  mutate("bad_place", (sd, p) => edit((j) => (j.places = [{ id: "x", name: "", kind: "item", position: [0, 0, 0], radius_m: 0.4 }]))(sd, p));
  mutate("heldout_photo_missing", (sd, p) => { if (sj.scene.heldout.length) fs.rmSync(path.join(sd, sj.scene.heldout[0].photo)); else throw new Error("no heldout"); });
  fs.rmSync(tmp, { recursive: true, force: true });
  const allRejected = muts.every((m) => m.rejected);
  return { pass: okBase && allRejected, value: `valid bundle: ${res.n_checks} checks, ${res.n_failed} failed, ${(res.total_bytes / 1e6).toFixed(1)} MB; ${muts.filter((m) => m.rejected).length}/${muts.length} corrupted copies rejected`,
    metrics: { n_checks: res.n_checks, warnings: res.warnings, mutations: muts, total_bytes: res.total_bytes } };
});

// ---------------------------------------------------------------------------------------------------------- H9
T("H9", "20 Mbps link: SD fully loaded within 18 s; preview first frame within 4 s", async (c) => {
  const { scene } = sceneJson(c.web, c.scene);
  const BPS = 2.5e6;
  return withBrowser(c.web, async (ctx) => {
    await ctx.pg.send("Network.setCacheDisabled", { cacheDisabled: true });
    const t0 = Date.now();
    const r = await openViewer(ctx, "", { timeout: 90000 });
    await waitFullLoad(ctx.pg, 90000);
    const done = await ctx.pg.eval("__ns.loadedMs");
    const hasPreview = !!scene.splats.preview;
    const info = await ctx.pg.eval("({kind: __ns.S.active.userData.kind, n: __ns.S.splatCount})");
    const pass = done / 1000 <= 18 && (hasPreview ? r.ff / 1000 <= 4 : true);
    return { pass, value: `first frame ${(r.ff / 1000).toFixed(2)} s${hasPreview ? " (preview)" : " (no preview in scene)"}; SD loaded ${(done / 1000).toFixed(2)} s at 20 Mbps (${(scene.splats.sd.bytes / 1e6).toFixed(1)} MB)`,
      metrics: { first_frame_s: +(r.ff / 1000).toFixed(2), sd_loaded_s: +(done / 1000).toFixed(2), link_mbps: 20, sd_mb: +(scene.splats.sd.bytes / 1e6).toFixed(2), has_preview: hasPreview, wall_s: +((Date.now() - t0) / 1000).toFixed(1) } };
  }, { server: { bytesPerSec: BPS } });
});

// ---------------------------------------------------------------------------------------------------------- feature checks
T("X1", "WebGL2-missing card shows (and links to the video when configured)", async (c) => {
  return withBrowser(c.web, async (ctx) => {
    await ctx.pg.goto(ctx.url(""));
    await ctx.pg.waitFor("window.__ns && window.__ns.error", 15000, 100);
    const r = await ctx.pg.eval("({err: __ns.error, shown: !document.getElementById('nowebgl').hidden, loaderHidden: document.getElementById('loader').hidden})");
    await ctx.pg.screenshot(path.join(c.shots, "x1_no_webgl.png"));
    return { pass: r.err === "webgl2-missing" && r.shown, value: `error=${r.err}, card shown=${r.shown}` };
  }, { chrome: { extraArgs: ["--disable-gpu", "--disable-3d-apis", "--disable-webgl", "--disable-webgl2"] } });
});

T("X2", "keys: R respawns, H toggles the stats panel, M cycles the map, ? toggles help, Esc closes it", async (c) => {
  return withBrowser(c.web, async (ctx) => {
    await openViewer(ctx, "", { start: true });
    const q = (e) => ctx.pg.eval(e);
    const checks = {};
    await ctx.pg.key("KeyW"); await sleep(900); await ctx.pg.key("KeyW", false); await sleep(200);
    const moved = await q("__ns.pose()"); await ctx.pg.tap("KeyR"); await sleep(150); const back = await q("__ns.pose()");
    const sp = (await q("__ns.S.data.spawn.position"));
    checks.R = Math.hypot(back.x - sp[0], back.z - sp[2]) < 0.2 && Math.hypot(moved.x - sp[0], moved.z - sp[2]) > 0.3;
    const cls0 = await q("document.getElementById('hud').className");
    await ctx.pg.tap("KeyH"); const cls1 = await q("document.getElementById('hud').className");
    await ctx.pg.tap("KeyH"); const cls2 = await q("document.getElementById('hud').className");
    checks.H = cls0 !== cls1 && cls1 !== cls2;
    const m0 = await q("(() => { const w = document.getElementById('mapwrap'); return [w.hidden, w.classList.contains('large')]; })()");
    await ctx.pg.tap("KeyM"); const m1 = await q("(() => { const w = document.getElementById('mapwrap'); return [w.hidden, w.classList.contains('large')]; })()");
    await ctx.pg.tap("KeyM"); const m2 = await q("(() => { const w = document.getElementById('mapwrap'); return [w.hidden, w.classList.contains('large')]; })()");
    await ctx.pg.tap("KeyM");
    checks.M = m0[0] === false && m0[1] === false && m1[1] === true && m2[0] === true;
    await ctx.pg.tap("KeyM"); await q("(() => { const w = document.getElementById('mapwrap'); if (w.classList.contains('large')) w.classList.remove('large'); })()");
    await ctx.pg.tap("Slash"); const h1 = await q("!document.getElementById('help').hidden");
    await ctx.pg.tap("Escape"); const h2 = await q("!document.getElementById('help').hidden");
    checks.help = h1 === true && h2 === false;
    const pass = Object.values(checks).every(Boolean);
    return { pass, value: Object.entries(checks).map(([k, v]) => `${k}:${v ? "ok" : "FAIL"}`).join(" "), metrics: checks };
  });
});

T("X3", "tour benchmark (T / ?bench=1) completes and reports avg fps, p95, splats, MB, GPU", async (c) => {
  return withBrowser(c.web, async (ctx) => {
    await openViewer(ctx, "?bench=1&benchsecs=12", {});
    await ctx.pg.waitFor("__ns.benchResult", 40000, 200);
    const r = await ctx.pg.eval("__ns.benchResult");
    const txt = await ctx.pg.eval("document.getElementById('b-res').textContent");
    await ctx.pg.screenshot(path.join(c.shots, "x3_bench_result.png"));
    const pass = r.frames > 100 && r.avg_fps > 0 && r.splats > 0 && r.mb > 0 && /GPU/.test(txt);
    return { pass, value: `avg ${r.avg_fps} fps, p95 ${r.p95_ms} ms, ${r.splats} splats, ${r.mb} MB, ${r.gpu}`, metrics: r };
  });
});

T("X4", "reality check (C): opens over held-out pairs, slider and arrow keys work, Esc returns", async (c) => {
  return withBrowser(c.web, async (ctx) => {
    await openViewer(ctx, "", { start: true });
    const n = await ctx.pg.eval("__ns.S.data.heldout.length");
    if (!n) return { pass: null, value: "not run: scene has no held-out pairs" };
    await ctx.pg.tap("KeyC"); await sleep(1800);
    const a = await ctx.pg.eval("({mode: __ns.S.mode, i: __ns.S.reality.i, box: __ns.S.reality.box, cap: document.getElementById('rc-cap').textContent, clip: document.getElementById('rc-photo').style.clipPath})");
    await ctx.pg.screenshot(path.join(c.shots, "x4_reality.png"));
    await ctx.pg.eval("(() => { const s = document.getElementById('rc-slider'); s.value = 20; s.dispatchEvent(new Event('input')); })()");
    const clip2 = await ctx.pg.eval("document.getElementById('rc-photo').style.clipPath");
    await ctx.pg.tap("ArrowRight"); await sleep(150);
    const i2 = await ctx.pg.eval("__ns.S.reality.i");
    await ctx.pg.tap("Escape"); await sleep(300);
    const after = await ctx.pg.eval("__ns.S.mode");
    const pass = a.mode === "reality" && clip2 !== a.clip && i2 === 1 && after === "play" && a.box.w > 100;
    return { pass, value: `${n} pairs; mode ${a.mode}; slider clip ${a.clip} -> ${clip2}; right arrow -> photo ${i2 + 1}; Esc -> ${after}; caption "${a.cap}"` };
  });
});

T("X5", "bookmarks 1-5 jump to their saved views", async (c) => {
  return withBrowser(c.web, async (ctx) => {
    await openViewer(ctx, "", { start: true });
    const bm = await ctx.pg.eval("__ns.S.data.bookmarks");
    if (!bm.length) return { pass: null, value: "not run: scene has no bookmarks" };
    const rows = [];
    for (let i = 0; i < Math.min(5, bm.length); i++) {
      await ctx.pg.tap(`Digit${i + 1}`); await sleep(150);
      const p = await ctx.pg.eval("__ns.pose()");
      rows.push({ n: i + 1, err_m: +Math.hypot(p.x - bm[i].position[0], p.z - bm[i].position[2]).toFixed(3) });
    }
    return { pass: rows.every((r) => r.err_m < 0.3), value: rows.map((r) => `${r.n}: ${r.err_m} m`).join(", "), metrics: { rows } };
  });
});

T("X6", "URL flags: ?pose= sets the start pose, ?keys=1 shows the key overlay, ?q=low starts at the low preset", async (c) => {
  return withBrowser(c.web, async (ctx) => {
    const { scene } = sceneJson(c.web, c.scene);
    const sp = scene.spawn.position;
    await openViewer(ctx, `?pose=${sp[0] + 0.1},1.30,${sp[2] + 0.1},33,-10&keys=1&q=low`, { start: true });
    const r = await ctx.pg.eval("({pose: __ns.pose(), keys: !document.getElementById('keyovl').hidden || __ns.flags.keys, gov: __ns.governor.step.name, kind: __ns.S.active.userData.kind})");
    await ctx.pg.key("KeyW"); await sleep(300);
    const on = await ctx.pg.eval("document.querySelectorAll('#keyovl kbd.on').length");
    await ctx.pg.key("KeyW", false);
    const poseOk = Math.abs(r.pose.yawDeg - 33) < 1 && Math.abs(r.pose.pitchDeg + 10) < 1 && Math.abs(r.pose.y - 1.3) < 1e-6;
    const pass = poseOk && r.keys && r.gov === "low" && on >= 1;
    return { pass, value: `pose ${JSON.stringify(r.pose)}; key overlay ${r.keys}, lit ${on}; governor ${r.gov}; splats ${r.kind}` };
  });
});

T("X7", "minimap: view cone and arrow drawn; clicking teleports to the nearest walkable cell", async (c) => {
  return withBrowser(c.web, async (ctx) => {
    await openViewer(ctx, "", { start: true });
    await sleep(500);
    // the minimap image must actually be drawn (warm photo colours; the walkable tint alone is teal)
    const warm = await ctx.pg.eval(`(() => { const c = document.getElementById('map'), d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0; for (let i = 0; i < d.length; i += 4) if (d[i] > d[i + 1] + 25 && d[i] > d[i + 2] + 10) n++; return n; })()`);
    const before = await ctx.pg.eval("__ns.pose()");
    // click a walkable cell far from the spawn: take the walkable cell farthest from the spawn
    const target = await ctx.pg.eval(`(() => { const g = __ns.grid, p = __ns.pose(); let best = null, bd = -1;
      for (let i = 0; i < g.h; i += 4) for (let j = 0; j < g.w; j += 4) if (g.data[i * g.w + j] === 255) { const [x, z] = g.cellCentre(i, j); const d = Math.hypot(x - p.x, z - p.z); if (d > bd) { bd = d; best = [x, z]; } }
      const m = __ns.S.minimap, r = document.getElementById('map').getBoundingClientRect();
      const k = r.width / m.c.width * m.k; return {x: best[0], z: best[1], sx: r.left + (best[0] - g.ox) * k, sy: r.top + (best[1] - g.oz) * k}; })()`);
    await ctx.pg.click(target.sx, target.sy); await sleep(250);
    const after = await ctx.pg.eval("__ns.pose()");
    const err = Math.hypot(after.x - target.x, after.z - target.z);
    return { pass: err < 0.3 && Math.hypot(after.x - before.x, after.z - before.z) > 1 && warm >= 200, value: `minimap image drawn (${warm} warm pixels); teleported ${Math.hypot(after.x - before.x, after.z - before.z).toFixed(1)} m; landing error ${err.toFixed(2)} m` };
  });
});

T("X8", "HD button swaps in splat_hd (a scene without one gets the SD file standing in for HD, to test the mechanism); no button when HD equals SD", async (c) => {
  const { scene, name } = sceneJson(c.web, c.scene);
  const hdOf = (j) => ({ ...j.splats.sd, url: j.splats.sd.url + "?as=hd", sh_degree: 3, bytes: j.splats.sd.bytes + 1 });      // another URL for the same bytes: a separate download, as a real HD file would be; one byte more in the entry, so that it differs from SD even when SD is already SH3
  const withScene = async (edit, fn) => {
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ns-x8-"));
    const j = JSON.parse(JSON.stringify(scene)); edit(j);
    fs.writeFileSync(path.join(tmp, "scene.json"), JSON.stringify(j));
    try { return await withBrowser(c.web, fn, { server: { extraMounts: [{ prefix: `/scenes/${name}/`, dir: tmp }] } }); } finally { fs.rmSync(tmp, { recursive: true, force: true }); }
  };
  const stand = !scene.splats.hd;
  const a = await withScene((j) => { if (!j.splats.hd || j.splats.hd.bytes === j.splats.sd.bytes) j.splats.hd = hdOf(j); }, async (ctx) => {
    await openViewer(ctx, "", { start: true }); await waitFullLoad(ctx.pg);
    const shown = await ctx.pg.eval("!document.getElementById('b-hd').hidden");
    await ctx.pg.eval("document.getElementById('b-hd').click()");
    await ctx.pg.waitFor("__ns.S.active.userData.kind === 'hd'", 60000, 200);
    const n = await ctx.pg.eval("__ns.S.splatCount");
    await ctx.pg.eval("document.getElementById('b-hd').click()");
    await ctx.pg.waitFor("__ns.S.active.userData.kind === 'sd'", 30000, 200);
    return { shown, n };
  });
  const b = await withScene((j) => { j.splats.hd = { ...j.splats.sd, url: "splat_hd_copy.spz" }; }, async (ctx) => {       // identical size, count and SH degree: an old bundle with a copy of SD
    await openViewer(ctx, "", { start: true }); await waitFullLoad(ctx.pg);
    return ctx.pg.eval("({hidden: document.getElementById('b-hd').hidden, hasHd: !!__ns.S.data.splats.hd})");
  });
  return { pass: a.shown && a.n > 0 && b.hidden && b.hasHd, value: `HD button shown ${a.shown}; HD active with ${a.n} splats, then back to SD${stand ? " (mechanism test, SD file as HD)" : ""}; an HD entry equal to SD (same bytes, count, SH degree): button hidden ${b.hidden}`, metrics: { stand_in: stand, equal_hd_hidden: b.hidden } };
});

T("X9", "?scene=<name> loads that scene; an unknown scene shows the error card instead of a blank page", async (c) => {
  const { scene } = sceneJson(c.web, c.scene);
  return withBrowser(c.web, async (ctx) => {
    const r1 = await openViewer(ctx, `?scene=${scene.name}`, {});
    const t1 = await ctx.pg.eval("document.title");
    await ctx.pg.goto(ctx.url("?scene=does_not_exist"));
    await ctx.pg.waitFor("window.__ns && window.__ns.error", 15000, 100);
    const r2 = await ctx.pg.eval("({err: __ns.error, card: !document.getElementById('errcard').hidden})");
    return { pass: r1.err === null && r2.card, value: `known scene ok (title "${t1}"); unknown scene -> error card ${r2.card} (${r2.err})` };
  });
});

T("X10", "fly mode (F) passes furniture but stays inside the room bounds", async (c) => {
  return withBrowser(c.web, async (ctx) => {
    await openViewer(ctx, "", { start: true });
    await ctx.pg.tap("KeyF"); await sleep(100);
    const fly = await ctx.pg.eval("__ns.walker.fly");
    // run in one direction for 12 s of simulated time: must end inside bounds, and may leave the walkable cells
    const r = await ctx.pg.eval(`(() => { const w = __ns.walker, b = __ns.S.data.bounds; let out = 0, offWalk = 0;
      for (const yaw of [0, 90, 180, 270]) { w.setPose(__ns.S.data.spawn.position[0], __ns.S.data.spawn.position[2], yaw, 0); w.fly = true;
        for (let t = 0; t < 12; t += 1/60) { w.update(1/60, {forward: 1, strafe: 0, run: true}); if (w.x < b.min[0] - 1e-6 || w.x > b.max[0] + 1e-6 || w.z < b.min[2] - 1e-6 || w.z > b.max[2] + 1e-6) out++; if (!__ns.grid.walkable(w.x, w.z)) offWalk++; } }
      w.fly = false; w.setPose(__ns.S.data.spawn.position[0], __ns.S.data.spawn.position[2], 0, 0); return {out, offWalk}; })()`);
    return { pass: fly === true && r.out === 0 && r.offWalk > 0, value: `fly on: ${fly}; samples outside bounds ${r.out}; samples past furniture/walls ${r.offWalk}` };
  });
});

T("X11", "weak-GPU fallback: the last governor step swaps to the 200 k-splat preview, and steps back up to SD", async (c) => {
  const { scene } = sceneJson(c.web, c.scene);
  if (!scene.splats.preview) return { pass: null, value: "not run: scene has no preview file" };
  return withBrowser(c.web, async (ctx) => {
    await openViewer(ctx, "", { start: true }); await waitFullLoad(ctx.pg);
    const names = await ctx.pg.eval("__ns.governor.ladder.map(l => l.name)");
    const last = names.length - 1;
    const a = await ctx.pg.eval("({kind: __ns.S.active.userData.kind, n: __ns.S.splatCount})");
    await ctx.pg.eval(`__ns.selectQuality(${last})`); await sleep(1200);
    const b = await ctx.pg.eval("({kind: __ns.S.active.userData.kind, n: __ns.S.splatCount, in_scene: __ns.S.active.parent !== null})");
    await ctx.pg.eval("__ns.selectQuality(0)"); await sleep(1200);
    const d = await ctx.pg.eval("({kind: __ns.S.active.userData.kind, n: __ns.S.splatCount})");
    const pass = names[last] === "preview" && a.kind === "sd" && b.kind === "preview" && b.in_scene && d.kind === "sd";
    return { pass, value: `ladder ${names.join(" > ")}; ${a.kind} ${a.n} -> ${b.kind} ${b.n} -> ${d.kind} ${d.n}` };
  });
});

T("X12", "L copies a spot link; opening it puts the player at the same pose", async (c) => {
  return withBrowser(c.web, async (ctx) => {
    await openViewer(ctx, "", { start: true });
    await ctx.pg.eval("__ns.setPose(__ns.S.data.spawn.position[0] + 0.2, 1.42, __ns.S.data.spawn.position[2] + 0.1, 77, -8)");
    await ctx.pg.tap("KeyL"); await sleep(200);
    const link = await ctx.pg.eval("__ns.lastSpotLink");
    const pg2 = await ctx.br.newPage();
    await pg2.goto(link); await pg2.waitFor("window.__ns && (window.__ns.ready || window.__ns.error)", 30000, 100);
    const p = await pg2.eval("__ns.pose()");
    await pg2.close();
    return { pass: Math.abs(p.yawDeg - 77) < 0.1 && Math.abs(p.pitchDeg + 8) < 0.1, value: `${link.replace(/^https?:\/\/[^/]+/, "")} -> yaw ${p.yawDeg.toFixed(1)}, pitch ${p.pitchDeg.toFixed(1)}` };
  });
});

// ---------------------------------------------------------------------------------------------------------- review fixes (30 Oct)
// A page whose requestAnimationFrame never runs faster than `fps` (battery saver, vsync at half rate), with a stall switch that mimics a hidden tab.
const CAP_SCRIPT = (fps) => `(() => { const raf = window.requestAnimationFrame.bind(window); let last = -1e9; window.__stallUntil = 0; const gap = ${1000 / fps};
  window.requestAnimationFrame = (cb) => raf(() => { const now = performance.now(); const wait = Math.max(gap - (now - last), window.__stallUntil - now);
    if (wait > 1) setTimeout(() => { last = performance.now(); cb(last); }, wait); else { last = now; cb(now); } }); })()`;

T("X13", "frame rate capped at 30 fps (battery saver): the governor may try a step, sees no gain, and ends on standard; never on the preview", async (c) => {
  return withBrowser(c.web, async (ctx) => {
    await ctx.pg.send("Page.addScriptToEvaluateOnNewDocument", { source: CAP_SCRIPT(30) });
    await openViewer(ctx, "?q=sd", { start: true }); await waitFullLoad(ctx.pg);
    await sleep(13000);
    let r;
    for (let i = 0; i < 24; i++) {                                    // let a trial or a re-check that is in flight finish
      r = await ctx.pg.eval("({hist: __ns.governor.history.map(h => h.step.name + ':' + h.kind), level: __ns.governor.level, trial: !!__ns.governor.trial || !!__ns.governor.verify, kind: __ns.S.active.userData.kind, n: __ns.S.splatCount, pr: __ns.S.pr, fps: __ns.S.stats.fps})");
      if (!r.trial) break; await sleep(500);
    }
    const noPreview = !r.hist.some((h) => /^preview/.test(h));
    const pass = r.fps > 24 && r.fps < 36 && noPreview && r.kind === "sd" && r.level === 0 && r.pr === 1;
    return { pass, value: `capped at ${r.fps.toFixed(0)} fps; governor history [${r.hist.join(", ")}]; ends on ${r.kind} ${r.n} splats at pixel ratio ${r.pr}`, metrics: r };
  });
});

T("X14", "a failed or dropped SD download keeps the preview and shows a toast, not the error card; one retry; the error card (with Reload) only when nothing renders", async (c) => {
  const { name, dir, scene } = sceneJson(c.web, c.scene);
  if (!scene.splats.preview) return { pass: null, value: "not run: scene has no preview file" };
  const sd = (p) => /splat_sd\.spz$/.test(p);
  const rows = [];
  const scenario = async (label, server, { noPreview = false, expect }) => {
    let tmp = null; const extraMounts = [];
    if (noPreview) {
      tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ns-x14-")); const j = JSON.parse(JSON.stringify(scene)); j.splats.preview = null;
      fs.writeFileSync(path.join(tmp, "scene.json"), JSON.stringify(j)); extraMounts.push({ prefix: `/scenes/${name}/`, dir: tmp });
    }
    try {
      await withBrowser(c.web, async (ctx) => {
        await openViewer(ctx, "", { start: !noPreview });
        const t0 = Date.now();
        await ctx.pg.waitFor("__ns.log.some(l => /initialised, \\d+ splats/.test(l) && /splats active/.test(__ns.log.join('|'))) || __ns.log.some(l => /failed again/.test(l)) || __ns.error", 45000, 150);
        await sleep(600);
        const r = await ctx.pg.eval(`({err: __ns.error, card: !document.getElementById('errcard').hidden, reload: !!document.querySelector('#errcard button'), kind: __ns.S.active && __ns.S.active.userData.kind, toasts: __ns.toasts || [],
          fullFailed: __ns.fullFailed === true, loaderHidden: document.getElementById('loader').hidden, full: __ns.S.fullLoaded === true, goDisabled: document.getElementById('s-go').disabled, started: !!__ns.S.started, n: __ns.S.splatCount})`);
        let moved = null;
        if (expect === "preview") { const p0 = await ctx.pg.eval("__ns.pose()"); await ctx.pg.key("KeyW"); await sleep(900); await ctx.pg.key("KeyW", false); const p1 = await ctx.pg.eval("__ns.pose()"); moved = Math.hypot(p1.x - p0.x, p1.z - p0.z); }
        const reqs = ctx.srv.log.filter((l) => sd(l.path)).length;
        rows.push({ label, ...r, moved, sd_requests: reqs, ok: expect === "preview" ? !r.card && r.err === null && r.kind === "preview" && r.toasts.some((t) => /did not load/.test(t)) && r.fullFailed && r.loaderHidden && moved > 0.3 && reqs >= 2
          : expect === "sd" ? !r.card && r.kind === "sd" && r.full && reqs >= 2 : r.card && r.reload && r.kind !== "sd" });
      }, { server: { ...server, extraMounts } });
    } finally { if (tmp) fs.rmSync(tmp, { recursive: true, force: true }); }
  };
  await scenario("HTTP 500 on every SD request", { fail: (p) => (sd(p) ? 500 : null) }, { expect: "preview" });
  await scenario("connection dropped after 1 MB, every time", { fail: (p) => (sd(p) ? { dropAfter: 1e6 } : null) }, { expect: "preview" });
  await scenario("first SD request fails, the retry works", { fail: (p, nth) => (sd(p) && nth === 1 ? 500 : null) }, { expect: "sd" });
  await scenario("no preview in the scene and SD fails: error card with a Reload button", { fail: (p) => (sd(p) ? 500 : null) }, { noPreview: true, expect: "card" });
  return { pass: rows.every((r) => r.ok), value: rows.map((r) => `${r.label}: ${r.ok ? "ok" : "FAIL"} (${r.card ? "error card" : r.kind}, ${r.sd_requests} SD requests${r.moved != null ? `, walked ${r.moved.toFixed(1)} m` : ""})`).join("; "), metrics: { rows } };
});

T("X15", "hidden tab: coming back after a long pause does not step the quality down (visibilitychange resets the timer and the governor; no step is judged on the pause)", async (c) => {
  return withBrowser(c.web, async (ctx) => {
    await ctx.pg.send("Page.addScriptToEvaluateOnNewDocument", { source: CAP_SCRIPT(60) });
    await openViewer(ctx, "?q=sd", { start: true }); await waitFullLoad(ctx.pg);
    await sleep(3000);
    const before = await ctx.pg.eval("__ns.governor.history.length");
    // pause the frames for 6 s; the browser fires visibilitychange when the tab is shown again, just before the first frame
    await ctx.pg.eval("window.__stallUntil = performance.now() + 6000; window.__ret = performance.now() + 5900; setTimeout(() => document.dispatchEvent(new Event('visibilitychange')), 5900); 1");
    await sleep(5950);
    const wired = await ctx.pg.eval("({last: __ns.S.stats.last, samples: __ns.governor.samples.length})");
    await sleep(6500);
    // A busy machine may step for its own reasons (the Mac is shared). What must not happen is a step judged on the pause: with the stale
    // samples the first reading after the pause was a mean of well under 15 fps (0.1 to 3.5 fps in the review's runs).
    const r = await ctx.pg.eval(`({n: __ns.governor.history.length, after: __ns.governor.history.filter(h => h.t > window.__ret).map(h => ({kind: h.kind, step: h.step.name, why: h.why, atS: +((h.t - window.__ret) / 1000).toFixed(1)})), level: __ns.governor.level, fps: __ns.S.stats.fps})`);
    const bad = r.after.filter((h) => { const m = /mean ([\d.]+) fps/.exec(h.why); return m && +m[1] < 15; });
    return { pass: wired.last === null && wired.samples === 0 && bad.length === 0, value: `after a 6 s pause: timer cleared ${wired.last === null}, governor samples ${wired.samples}; steps after the pause [${r.after.map((h) => `${h.step}:${h.kind} at +${h.atS} s`).join(", ") || "none"}], steps judged on the pause ${bad.length}; ${r.fps.toFixed(0)} fps`, metrics: { before, ...r, wired, bad } };
  });
});

T("X16", "touchscreen laptop (touch support, mouse as main pointer) gets the desktop path: 'Click to start', pointer lock on a mouse click; a phone still gets 'Tap to start'", async (c) => {
  const laptop = `(() => { Object.defineProperty(navigator, 'maxTouchPoints', { get: () => 5 }); window.ontouchstart = null;
    HTMLCanvasElement.prototype.requestPointerLock = function () { (window.__locks = window.__locks || []).push(1); return Promise.resolve(); }; })()`;
  const lap = await withBrowser(c.web, async (ctx) => {
    await ctx.pg.send("Page.addScriptToEvaluateOnNewDocument", { source: laptop });
    await openViewer(ctx, "", {});
    const sig = await ctx.pg.eval("({mtp: navigator.maxTouchPoints, ots: 'ontouchstart' in window, fine: matchMedia('(pointer: fine)').matches, touchDevice: __ns.S.touchDevice, label: document.getElementById('s-go').textContent, deskHint: !document.getElementById('hint-desk').hidden, measureBtn: !document.getElementById('b-measure').hidden})");
    const b = await ctx.pg.eval("(() => { const r = document.getElementById('s-go').getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; })()");
    await ctx.pg.click(b[0], b[1]); await sleep(300);
    const after = await ctx.pg.eval("({started: __ns.S.started, locks: (window.__locks || []).length, resumePill: !document.getElementById('paused').hidden})");
    return { ...sig, ...after };
  });
  const phone = await withBrowser(c.web, async (ctx) => {
    await openViewer(ctx, "", {});
    return ctx.pg.eval("({touchDevice: __ns.S.touchDevice, label: document.getElementById('s-go').textContent, touchHint: !document.getElementById('hint-touch').hidden})");
  }, { page: { width: 390, height: 844, mobile: true, touch: true, dpr: 2 }, chrome: { width: 390, height: 844 } });
  const pass = lap.mtp === 5 && lap.ots && lap.fine && lap.touchDevice === false && lap.label === "Click to start" && lap.deskHint && !lap.measureBtn && lap.started && lap.locks === 1
    && phone.touchDevice === true && phone.label === "Tap to start" && phone.touchHint;
  return { pass, value: `laptop (maxTouchPoints ${lap.mtp}, pointer fine ${lap.fine}): touch path ${lap.touchDevice}, button "${lap.label}", pointer lock requests after a mouse click ${lap.locks}; phone: touch path ${phone.touchDevice}, button "${phone.label}"`, metrics: { lap, phone } };
});

T("X17", "progress before the 3 MB of JS arrives: the loading text is in the HTML, the bar moves, modulepreload links are present; real progress text follows", async (c) => {
  const html = fs.readFileSync(path.join(c.web, "index.html"), "utf8");
  const staticText = /<div id="loader-sub"[^>]*>([^<]+)</.exec(html)?.[1] || "";
  const preloads = [...html.matchAll(/<link rel="modulepreload" href="([^"]+)"/g)].map((m) => m[1]);
  return withBrowser(c.web, async (ctx) => {
    await ctx.pg.goto(ctx.url(""));
    await sleep(1200);
    const early = await ctx.pg.eval("({sub: document.getElementById('loader-sub').textContent, shown: !document.getElementById('loader').hidden, wait: document.getElementById('loader').classList.contains('wait'), anim: getComputedStyle(document.querySelector('#bar-fill')).animationName, ns: !!window.__ns})");
    await ctx.pg.waitFor("window.__ns && (window.__ns.ready || window.__ns.error)", 60000, 100);
    await waitFullLoad(ctx.pg);
    const late = await ctx.pg.eval("({wait: document.getElementById('loader').classList.contains('wait'), hidden: document.getElementById('loader').hidden, sub: document.getElementById('loader-sub').textContent})");
    const pass = staticText.length > 20 && /viewer/i.test(staticText) && preloads.length >= 4 && early.shown && early.wait && early.anim === "ns-wait" && (late.hidden || !late.wait);
    return { pass, value: `static text "${staticText}"; ${preloads.length} modulepreload links; at 1.2 s: loader shown ${early.shown}, bar animation ${early.anim}, text "${early.sub}"; after load: loader hidden ${late.hidden}`, metrics: { staticText, preloads, early, late } };
  }, { server: { delay: (p) => (/spark\.module\.js$/.test(p) ? 2500 : 0) } });
});

T("X18", "clear colour: dark while loading (the first-frame probe needs it), then the scene's neutral colour; hidden HD when it equals SD", async (c) => {
  const { scene } = sceneJson(c.web, c.scene);
  return withBrowser(c.web, async (ctx) => {
    await openViewer(ctx, "", { start: true }); await waitFullLoad(ctx.pg);
    const r = await ctx.pg.eval(`(() => { const col = new __ns.THREE.Color(); __ns.renderer.getClearColor(col); const c = col.getRGB({}, __ns.THREE.SRGBColorSpace); return { rgb: [c.r, c.g, c.b].map(v => Math.round(v * 255)), ns: __ns.clearColor, hdHidden: document.getElementById('b-hd').hidden, hasHd: !!__ns.S.data.splats.hd }; })()`);
    const want = scene.splats.render?.clear_color;
    const sameAsSd = scene.splats.hd && scene.splats.sd && scene.splats.hd.bytes === scene.splats.sd.bytes && scene.splats.hd.count === scene.splats.sd.count;
    const closeTo = (a, b) => a.every((v, i) => Math.abs(v - b[i]) <= 1);
    const pass = (want ? closeTo(r.rgb, want) : closeTo(r.rgb, [60, 58, 55])) && r.rgb.reduce((a, b) => a + b) > 3 * 30 && (!sameAsSd || r.hdHidden) && (r.hasHd || r.hdHidden);
    return { pass, value: `clear colour after the first splat frame ${JSON.stringify(r.rgb)} (scene.json ${want ? JSON.stringify(want) : "has none, default grey expected"}); HD button hidden ${r.hdHidden} (scene has HD ${r.hasHd}${sameAsSd ? ", identical to SD" : ""})`, metrics: r };
  });
});

// ---------------------------------------------------------------------------------------------------------- Round 2 preview: places, search, route, measure
registerGuideTests(T);
