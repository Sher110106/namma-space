#!/usr/bin/env node
// Viewer test runner. One command:   node viewer/test/run.mjs [--web DIR] [--out DIR] [--only H1,H3] [--pairs FILE] [--parity DIR] [--scene NAME]
// Writes <out>/verify.json (runner format) and <out>/shots/*.png. Exit 0 when no test failed (tests marked not run do not fail).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { tests } from "./tests.mjs";
import { launchChrome } from "./lib/chrome.mjs";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const args = new Map();
for (let i = 2; i < process.argv.length; i++) { const a = process.argv[i]; if (a.startsWith("--")) { args.set(a.slice(2), process.argv[i + 1] && !process.argv[i + 1].startsWith("--") ? process.argv[++i] : "1"); } }
const web = path.resolve(args.get("web") || path.join(HERE, "../dist"));
const out = path.resolve(args.get("out") || (path.basename(web) === "80_web" ? path.join(path.dirname(web), "90_verify") : path.join(HERE, "out")));
const only = args.get("only") ? new Set(args.get("only").split(",")) : null;
const ctx = { web, shots: path.join(out, "shots"), pairs: args.get("pairs") ? path.resolve(args.get("pairs")) : null, parity: args.get("parity") ? path.resolve(args.get("parity")) : null, scene: args.get("scene") || null };
fs.mkdirSync(ctx.shots, { recursive: true });

const results = [];
const probe = await launchChrome(); const browser = probe.version;
const gpuProbe = await (async () => { const pg = await probe.newPage(); await pg.goto("about:blank"); const g = await pg.eval(`(() => { const c = document.createElement('canvas'); const gl = c.getContext('webgl2'); if (!gl) return 'no webgl2'; const d = gl.getExtension('WEBGL_debug_renderer_info'); return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER); })()`); await pg.close(); return g; })();
await probe.close();
console.log(`web ${web}\nout ${out}\nbrowser ${browser}\ngpu ${gpuProbe}\n`);

function write() {
  const byId = Object.fromEntries(results.map((r) => [r.id, r]));
  const h2 = byId.H2, h1 = byId.H1, h3 = byId.H3, x3 = byId.X3;
  const devices = [];
  if (h2?.metrics?.gpu) {
    devices.push({ name: "MacBook (Apple M1), headless Chrome, ANGLE Metal, 1280x720, pixel ratio 1", gpu: h2.metrics.gpu, browser: browser, first_frame_s: h1?.metrics?.first_frame_s ?? null,
      fps_mean: +h2.metrics.fps_mean.toFixed(1), p95_ms: +h2.metrics.p95_ms.toFixed(2), mb: h2.metrics.mb, splats: h2.metrics.splats, source: "H2 scripted 20 s walk" });
  }
  if (x3?.metrics?.avg_fps) {
    devices.push({ name: "same machine, tour benchmark (T)", gpu: x3.metrics.gpu, browser: browser, first_frame_s: h1?.metrics?.first_frame_s ?? null, fps_mean: x3.metrics.avg_fps, p95_ms: x3.metrics.p95_ms, mb: x3.metrics.mb, splats: x3.metrics.splats, source: "X3 benchmark (compressed to 12 s)" });
  }
  const verify = {
    schema: "namma-space/verify@1", date: new Date().toISOString(), web, scene: ctx.scene || null, browser, gpu: gpuProbe, viewport: "1280x720, device pixel ratio 1",
    summary: { passed: results.filter((r) => r.pass === true).length, failed: results.filter((r) => r.pass === false).length, not_run: results.filter((r) => r.pass === null).length },
    tests: results.map(({ id, name, pass, value, metrics, detail, seconds }) => ({ id, name, pass, value, metrics, detail, seconds })),
    viewer: { devices },
    walls: h3?.metrics ? { walks: h3.metrics.walks, pushes: h3.metrics.pushes, penetrations: h3.metrics.walk_violations + h3.metrics.push_violations + h3.metrics.live_violations, wall_stops: h3.metrics.stopped } : null,
  };
  fs.writeFileSync(path.join(out, "verify.json"), JSON.stringify(verify, null, 2) + "\n");
  return verify;
}

for (const t of tests) {
  if (only && !only.has(t.id)) continue;
  const t0 = Date.now();
  let r;
  try { r = await t.fn(ctx); } catch (e) { r = { pass: false, value: `error: ${e.message}`, detail: String(e.stack || e).split("\n").slice(0, 4).join(" | ") }; }
  const seconds = +((Date.now() - t0) / 1000).toFixed(1);
  results.push({ id: t.id, name: t.name, ...r, seconds });
  console.log(`${r.pass === true ? "PASS" : r.pass === false ? "FAIL" : "SKIP"}  ${t.id}  ${r.value}  (${seconds}s)`);
  write();
}
const v = write();
console.log(`\n${v.summary.passed} passed, ${v.summary.failed} failed, ${v.summary.not_run} not run -> ${path.join(out, "verify.json")}`);
process.exit(v.summary.failed ? 1 : 0);
