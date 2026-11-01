// Namma Space viewer. Spark 2.4.0 on three.js 0.180.0, vendored; plain ES modules, no build step.
// Frame: metres, +Y up, yaw 0 looks along -Z, positive yaw turns left (CONTRACTS 2.0 and 2.12).
import * as THREE from "three";
import { SparkRenderer, SplatMesh } from "@sparkjsdev/spark";
import { resolveSceneUrl, fetchJson, validateScene, normaliseScene, scaleInfo, hdSameAsSd, clearColorOf } from "./scene.js";
import { Grid, Walker } from "./walker.js";
import { Governor, buildLadder } from "./governor.js";
import { FrameStats } from "./stats.js";
import { TourPath } from "./tour.js";
import { Minimap } from "./minimap.js";
import { Guide } from "./guide.js";
import * as sim from "./sim.js";

const $ = (id) => document.getElementById(id);
const P = new URLSearchParams(location.search);
const flag = (k) => P.has(k) && P.get(k) !== "0" && P.get(k) !== "false";
const FLAGS = { scene: P.get("scene"), asset: P.get("asset"), pose: P.get("pose"), bench: flag("bench"), stress: flag("stress"), keys: flag("keys"), q: P.get("q"), place: P.get("place"), route: flag("route"), follow: flag("follow") };
const t0 = performance.now();
const ns = (window.__ns = { flags: FLAGS, version: { spark: "2.4.0", three: THREE.REVISION }, log: [], firstFrameMs: null, ready: false, error: null });
const note = (m) => ns.log.push(`${(performance.now() - t0).toFixed(0)}ms ${m}`);

// ---------------------------------------------------------------------------------------------------------------- helpers
function toast(msg, ms = 1800) {
  (ns.toasts = ns.toasts || []).push(msg);                  // kept for the tests: a later toast replaces an earlier one on screen
  const t = $("toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast.id); toast.id = setTimeout(() => (t.hidden = true), ms);
}
const fmtInt = (n) => (n ? Math.round(n).toLocaleString("en-US") : "--");
const mb = (b) => (b / 1e6).toFixed(1);
function showError(title, msg) { $("err-title").textContent = title; $("err-msg").textContent = msg; $("errcard").hidden = false; $("loader").hidden = true; ns.error = `${title}: ${msg}`; }

// ---------------------------------------------------------------------------------------------------------------- WebGL2 gate
function hasWebGL2() {
  try { const c = document.createElement("canvas"); return !!(c.getContext("webgl2")); } catch { return false; }
}
if (!hasWebGL2() || flag("nowebgl")) {
  const url = document.querySelector('meta[name="ns-video"]')?.content;
  if (url) { const a = $("video-link"); a.href = url; a.hidden = false; }
  $("nowebgl-detail").textContent = navigator.userAgent;
  $("nowebgl").hidden = false; $("loader").hidden = true;
  ns.error = "webgl2-missing";
  throw new Error("WebGL2 not available");
}

// ---------------------------------------------------------------------------------------------------------------- renderer
const stage = $("stage");
const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance", alpha: false });
const LOAD_CLEAR = [11, 14, 20];                                  // page background; the first-frame probe compares pixels with this colour
const srgb = (c) => new THREE.Color().setRGB(c[0] / 255, c[1] / 255, c[2] / 255, THREE.SRGBColorSpace);   // 0-255 sRGB, as the canvas shows it
renderer.setClearColor(srgb(LOAD_CLEAR), 1);
// Once the splats show, holes (a wall closer than the model has data for) get a neutral colour from the scene, not near-black (finding 46).
function useSceneClear() { const c = clearColorOf(S.data?.splats?.render); renderer.setClearColor(srgb(c), 1); ns.clearColor = c; }
stage.appendChild(renderer.domElement);
const canvas = renderer.domElement;
const gl = renderer.getContext();
const dbg = gl.getExtension("WEBGL_debug_renderer_info");
const GPU = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
const SOFTWARE = /SwiftShader|llvmpipe|Software|Basic Render/i.test(GPU);
const QEFF = FLAGS.q || (SOFTWARE ? "low" : null);          // no GPU acceleration: start on the small preview file; ?q=sd overrides
const COARSE = matchMedia("(pointer: coarse)").matches;      // phone or tablet; Safari calls every Apple chip "Apple GPU", so only treat it as weak on touch devices
const WEAK = SOFTWARE || /Intel|UHD|Iris|HD Graphics|Mali|Adreno|PowerVR|Vega|Radeon Graphics/i.test(GPU) || (COARSE && (/Apple GPU/.test(GPU) || Math.min(innerWidth, innerHeight) < 900));
const camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.05, 50);
camera.rotation.order = "YXZ";
const scene = new THREE.Scene();
let spark = null;
ns.gpu = GPU; ns.software = SOFTWARE; ns.weak = WEAK;

canvas.addEventListener("webglcontextlost", (e) => { e.preventDefault(); showError("Graphics context lost", "The browser dropped the GPU context (often after sleep or a driver reset). Reload the page to continue."); });

// ---------------------------------------------------------------------------------------------------------------- state
const S = {
  data: null, grid: null, walker: null, minimap: null, tour: null,
  meshes: { preview: null, sd: null, hd: null }, active: null, quality: "sd",
  mode: "boot", startCardOpen: false, pointerLocked: false, dragLook: false,
  eyeY: 1.5, keys: new Set(), touch: { move: null, look: null, fwd: 0, str: 0 },
  loadedBytes: 0, bytesByUrl: new Map(), splatCount: 0,
  stats: new FrameStats(), gov: null, pr: 1, maxStdDev: Math.sqrt(8),
  firstFrame: false, sawSplats: false, frames: 0,
  bench: null, reality: null, hudLast: 0, hudTimer: null, guide: null,
};
ns.S = S; ns.renderer = renderer; ns.camera = camera; ns.THREE = THREE;

function applyPixelRatio(pr) {
  S.pr = pr;
  if (S.mode === "reality") return;
  renderer.setPixelRatio(pr);
  renderer.setSize(innerWidth, innerHeight);
}
function applyStep(step, kind = "set") {
  S.maxStdDev = step.maxStdDev;
  if (spark) spark.maxStdDev = step.maxStdDev;
  applyPixelRatio(step.pixelRatio);
  const want = step.mesh || S.baseKind;                     // the last ladder step swaps in the preview file
  if (S.fullLoaded && want && S.active && S.active.userData.kind !== want && S.meshes[want]?.isInitialized !== false && S.meshes[want]) {
    activate(S.meshes[want]).then(() => { updateHud(true); if (want === "preview" && kind !== "down") toast(`Reduced detail: ${fmtInt(S.splatCount)} splats`, 2500); });
  }
  updateHud(true);
}

// ---------------------------------------------------------------------------------------------------------------- load
async function loadImage(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const blob = await r.blob();
  // colorSpaceConversion:'none' keeps the grid bytes exact; older Safari lacks the option, so fall back step by step
  try { return await createImageBitmap(blob, { colorSpaceConversion: "none" }); } catch {}
  try { return await createImageBitmap(blob); } catch {}
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => rej(new Error(`${url}: cannot decode image`)); im.src = URL.createObjectURL(blob); });
}
async function decodeGrid(meta) {
  const bmp = await loadImage(meta.url);
  const cv = document.createElement("canvas"); cv.width = bmp.width; cv.height = bmp.height;
  const x = cv.getContext("2d", { willReadFrequently: true }); x.drawImage(bmp, 0, 0);
  const rgba = x.getImageData(0, 0, bmp.width, bmp.height).data;
  const data = new Uint8Array(bmp.width * bmp.height);
  for (let i = 0; i < data.length; i++) data[i] = rgba[i * 4];
  if (bmp.width !== meta.width || bmp.height !== meta.height) console.warn(`grid.png is ${bmp.width}x${bmp.height}, scene.json says ${meta.width}x${meta.height}`);
  return new Grid({ ...meta, width: bmp.width, height: bmp.height }, data);
}

function extOf(url) { const m = /\.([a-z0-9]+)(?:[?#]|$)/i.exec(url); return m ? m[1].toLowerCase() : ""; }

function makeMesh(kind, entry) {
  const T = S.data.splats.transform;
  const opts = { url: entry.url, onProgress: (e) => onSplatProgress(kind, entry, e) };
  const ft = extOf(entry.url);
  if (["spz", "ply", "splat", "ksplat", "rad"].includes(ft)) opts.fileType = ft;
  const m = new SplatMesh(opts);
  m.position.fromArray(T.position); m.quaternion.fromArray(T.quaternion); m.scale.setScalar(T.scale ?? 1);
  m.userData.kind = kind;
  return m;
}
function onSplatProgress(kind, entry, e) {
  const loaded = e.loaded || 0, total = e.total || entry.bytes || 0;
  S.bytesByUrl.set(entry.url, loaded);
  S.loadedBytes = [...S.bytesByUrl.values()].reduce((a, b) => a + b, 0);
  if (!S.firstFrame || kind === "hd") {
    $("loader").classList.remove("wait");
    const pct = total ? Math.min(100, (100 * loaded) / total) : null;
    $("bar-fill").style.width = pct == null ? "30%" : pct.toFixed(1) + "%";
    $("loader-sub").textContent = `${kind === "preview" ? "preview" : kind === "hd" ? "HD splats" : "splats"} ${mb(loaded)}${total ? " / " + mb(total) : ""} MB`;
  }
}
const numSplats = (m) => m?.packedSplats?.numSplats ?? m?.numSplats ?? m?.packedSplats?.getNumSplats?.() ?? 0;

/** Swap the visible splat mesh once the new one has produced a few frames. */
async function activate(mesh) {
  const prev = S.active;
  scene.add(mesh);
  S.active = mesh; S.splatCount = numSplats(mesh);
  if (prev && prev !== mesh) {
    await waitFrames(8, 400);
    scene.remove(prev);       // kept in memory (the preview is small): the governor may swap back to it on a weak GPU
  }
}
function waitFrames(n, maxMs) {
  return new Promise((res) => { const f0 = S.frames, t = performance.now(); const id = setInterval(() => { if (S.frames - f0 >= n || performance.now() - t > maxMs) { clearInterval(id); res(); } }, 16); });
}

async function loadSplats() {
  const sp = S.data.splats;
  const want = QEFF === "hd" && sp.hd ? "hd" : QEFF === "low" && sp.preview ? "preview" : "sd";
  S.quality = want === "preview" ? "low" : want;
  S.baseKind = want;
  if (sp.preview && want !== "preview" && want !== "hd") {
    // preview first for a quick first frame, then the full file
    try {
      const pv = makeMesh("preview", sp.preview); S.meshes.preview = pv;
      await pv.initialized; note("preview initialised");
      await activate(pv);
    } catch (e) { console.warn("preview failed, loading the full file directly:", e && e.message || e); S.meshes.preview = null; S.active = null; }
  }
  let entry = want === "hd" ? sp.hd : want === "preview" ? sp.preview : sp.sd;
  const kind = want === "hd" ? "hd" : want === "preview" ? "preview" : "sd";
  let m = S.meshes[kind] || makeMesh(kind, entry); S.meshes[kind] = m;
  // a request that shows no byte of progress for 20 s counts as dead (FINAL_PLAN 2.10: retry after 20 s)
  const watchdog = (mesh, ent) => new Promise((_, rej) => setTimeout(() => { if (!(S.bytesByUrl.get(ent.url) > 0)) rej(new Error(`no data from ${ent.url} after 20 s`)); }, 20000));
  try { await Promise.race([m.initialized, watchdog(m, entry)]); }
  catch (e) {
    // the mirror / override failed (CORS, dead host), the link dropped, the host answered 5xx, or nothing arrived for 20 s:
    // try once more, from the scene's own file when an override was used, else from the same URL
    note(`${kind} failed from ${entry.url}: ${e && e.message || e}`);
    const retryUrl = entry.fallback || entry.url;
    console.warn(`splat file ${entry.url} failed (${e && e.message || e}); trying ${retryUrl}`);
    toast(entry.fallback ? "Mirror failed, loading the bundled copy" : "Download failed, retrying", 3000);
    try { m.dispose(); } catch {}
    S.bytesByUrl.delete(entry.url);
    entry = { ...entry, url: retryUrl, fallback: null };
    m = makeMesh(kind, entry); S.meshes[kind] = m;
    try { await Promise.race([m.initialized, watchdog(m, entry)]); }
    catch (e2) {
      note(`${kind} failed again from ${entry.url}: ${e2 && e2.message || e2}`);
      try { m.dispose(); } catch {}
      S.bytesByUrl.delete(entry.url);
      // a preview is already on screen and playable: keep it, say so, and let the page be reloaded. The error card is for "nothing to show".
      if (S.active && S.active.userData.kind === "preview") {
        S.meshes[kind] = null; S.baseKind = "preview"; S.quality = "low";
        S.fullLoaded = true; ns.fullFailed = true;
        $("loader").hidden = true;
        toast("The full-quality file did not load. Showing the preview (fewer splats). Reload the page to try again.", 9000);
        return;
      }
      throw e2;
    }
  }
  note(`${kind} initialised, ${numSplats(m)} splats`);
  await activate(m);
  S.fullLoaded = true;
  $("loader").hidden = true;
  if (sp.hd && kind !== "hd" && !hdSameAsSd(sp)) { $("b-hd").hidden = false; }      // no HD button when it would load the same file again
  note("splats active");
}

async function toggleHd() {
  const sp = S.data.splats;
  if (!sp.hd) return;
  const btn = $("b-hd");
  if (S.active?.userData.kind === "hd") {
    await activate(S.meshes.sd || (S.meshes.sd = makeMesh("sd", sp.sd)));
    btn.classList.remove("on"); btn.textContent = "HD"; S.quality = "sd"; S.baseKind = "sd"; S.gov.ladder = S.ladders.sd; S.gov.jumpTo(0); toast("SD splats");
  } else {
    btn.textContent = "HD loading…"; $("loader").hidden = false; $("bar-fill").style.width = "0%";
    const hd = S.meshes.hd || makeMesh("hd", sp.hd); S.meshes.hd = hd;
    try { await hd.initialized; await activate(hd); }
    catch (e) {                                              // the SD scene stays on screen
      try { hd.dispose(); } catch {}
      S.meshes.hd = null; S.bytesByUrl.delete(sp.hd.url);
      $("loader").hidden = true; btn.textContent = "HD"; toast("HD file did not load; staying on SD", 4000);
      return;
    }
    $("loader").hidden = true; btn.classList.add("on"); btn.textContent = "HD"; S.quality = "hd"; S.baseKind = "hd"; S.gov.ladder = S.ladders.hd; S.gov.jumpTo(0); toast("HD splats");
  }
  updateHud(true);
}

// ---------------------------------------------------------------------------------------------------------------- pose helpers
function setView(x, z, yawDeg, pitchDeg, y = null) {
  const w = S.walker;
  w.setPose(x, z, yawDeg, pitchDeg);
  S.eyeY = y ?? S.data.camera.eye_height;
}
function respawn() {
  S.guide?.stopFollow(true);
  const s = S.data.spawn;
  setView(s.position[0], s.position[2], s.yaw_deg || 0, s.pitch_deg || 0, s.position[1] || S.data.camera.eye_height);
  S.eyeY = S.data.camera.eye_height;
}
function gotoBookmark(i) {
  const b = S.data.bookmarks[i];
  if (!b) return;
  S.guide?.stopFollow(true);
  const w = S.walker; w.teleport(b.position[0], b.position[2]); w.yaw = (b.yaw_deg || 0) * Math.PI / 180; w.pitch = (b.pitch_deg || 0) * Math.PI / 180;
  S.eyeY = S.data.camera.eye_height; toast(b.name || `View ${i + 1}`);
}
function applyCamera() {
  const w = S.walker;
  camera.position.set(w.x, S.eyeY, w.z);
  camera.rotation.set(w.pitch, w.yaw, 0, "YXZ");
}
function parsePose(str) {
  const v = (str || "").split(",").map(Number);
  if (v.length < 3 || v.some((n) => !Number.isFinite(n))) return null;
  return v.length >= 5 ? { x: v[0], y: v[1], z: v[2], yaw: v[3], pitch: v[4] } : { x: v[0], y: v[1], z: v[2], yaw: v[3] ?? 0, pitch: 0 };
}

// ---------------------------------------------------------------------------------------------------------------- input
const ctrlKeys = new Set(["KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "ShiftLeft", "ShiftRight"]);
const modalOpen = () => !$("start").hidden || !$("help").hidden || !$("bench").hidden || !$("errcard").hidden;

addEventListener("keydown", (e) => {
  if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target?.tagName) && e.code !== "Escape") return;
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (ctrlKeys.has(e.code)) { S.keys.add(e.code); if (S.mode !== "bench") e.preventDefault(); updateKeyOverlay(); }
  if (e.repeat) return;
  if (S.mode === "play" && S.started && $("help").hidden && $("bench").hidden && $("errcard").hidden && S.guide?.onKey(e)) { e.preventDefault(); return; }
  switch (e.code) {
    case "Escape": onEscape(); break;
    case "Enter": case "Space": if (!$("start").hidden && !$("s-go").disabled) { e.preventDefault(); startPlay(); } break;
    case "KeyR": if (S.mode === "play") { respawn(); toast("Respawned"); } break;
    case "KeyH": toggleHud(); break;
    case "KeyM": cycleMap(); break;
    case "KeyT": if (S.mode === "play") startBench(); break;
    case "KeyC": if (S.mode === "play") enterReality(0); else if (S.mode === "reality") exitReality(); break;
    case "KeyF": if (S.mode === "play") toggleFly(); break;
    case "KeyL": if (S.mode === "play") copySpotLink(); break;
    case "Slash": toggleHelp(); break;
    case "ArrowLeft": if (S.mode === "reality") stepReality(-1); break;
    case "ArrowRight": if (S.mode === "reality") stepReality(1); break;
    default:
      if (e.key === "?") { toggleHelp(); break; }
      if (/^Digit[1-5]$/.test(e.code) && S.mode === "play") gotoBookmark(+e.code.slice(5) - 1);
  }
});
addEventListener("keyup", (e) => { S.keys.delete(e.code); updateKeyOverlay(); });
addEventListener("blur", () => { S.keys.clear(); updateKeyOverlay(); });

function onEscape() {
  if (!$("help").hidden) return toggleHelp(false);
  if (S.mode === "reality") return exitReality();
  if (S.mode === "bench") return stopBench(true);
  if (!$("bench").hidden) { $("bench").hidden = true; return; }
  if (S.guide?.escape()) return;
  if (document.pointerLockElement) document.exitPointerLock();
}
function toggleHelp(force) {
  const h = $("help"); const open = force ?? h.hidden;
  h.hidden = !open;
  if (open && document.pointerLockElement) document.exitPointerLock();
}
/** L: copy a link that opens this scene at the current spot (?scene and ?pose). */
async function copySpotLink() {
  const w = S.walker, u = new URL(location.href);
  u.hash = "";
  u.search = `?scene=${encodeURIComponent(S.data.name)}&pose=${[w.x, S.eyeY, w.z, w.yawDeg, w.pitchDeg].map((v) => +v.toFixed(2)).join(",")}`;
  ns.lastSpotLink = u.href;
  try { await navigator.clipboard.writeText(u.href); toast("Link to this spot copied"); } catch { toast("Link: " + u.href, 6000); }
}
function toggleFly() {
  const w = S.walker; w.fly = !w.fly;
  if (!w.fly) w.pushOut();
  toast(w.fly ? "Fly mode: walls only at the room edge" : "Walk mode");
}

const SENS_LOCK = 0.0022, SENS_DRAG = 0.0034;
const look = (dyaw, dpitch) => { if (!S.guide?.following) S.walker.look(dyaw, dpitch); };   // follow mode steers the view itself
canvas.addEventListener("pointerdown", (e) => {
  if (S.mode !== "play" || e.pointerType === "touch") return;
  S.drag = { id: e.pointerId, x: e.clientX, y: e.clientY, moved: 0 };
  canvas.setPointerCapture?.(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (S.mode !== "play") return;
  if (S.pointerLocked) { look(-e.movementX * SENS_LOCK, -e.movementY * SENS_LOCK); return; }
  if (S.drag && S.drag.id === e.pointerId && e.pointerType !== "touch") {
    const dx = e.clientX - S.drag.x, dy = e.clientY - S.drag.y; S.drag.x = e.clientX; S.drag.y = e.clientY; S.drag.moved += Math.abs(dx) + Math.abs(dy);
    look(-dx * SENS_DRAG, -dy * SENS_DRAG);
  }
});
canvas.addEventListener("pointerup", (e) => {
  if (S.mode !== "play" || !S.drag) return;
  const was = S.drag; S.drag = null;
  if (was.moved < 4 && e.pointerType !== "touch" && S.guide?.mode) {      // add-place and measure tools: a click picks a point
    if (!S.guide.pending) S.guide.onCanvasClick(S.pointerLocked ? innerWidth / 2 : e.clientX, S.pointerLocked ? innerHeight / 2 : e.clientY);
    return;
  }
  if (was.moved < 4 && e.pointerType !== "touch" && !S.pointerLocked && canvas.requestPointerLock) {
    try { const p = canvas.requestPointerLock(); p?.catch?.(() => {}); } catch {}
  }
});
const updatePaused = () => { $("paused").hidden = S.pointerLocked || S.mode !== "play" || S.touchDevice || !S.started || !!S.guide?.mode || !!S.guide?.searchOpen; };
document.addEventListener("pointerlockchange", () => { S.pointerLocked = document.pointerLockElement === canvas; updatePaused(); });
document.addEventListener("pointerlockerror", () => { S.pointerLocked = false; toast("Pointer lock unavailable here: drag to look"); });

// touch: left half drags to move (virtual stick), right half drags to look.
// The touch path (tap to start, no pointer lock, thumb hints) is for devices whose main pointer is a finger. A touchscreen laptop
// reports touch support too, but its main pointer is a mouse or trackpad (pointer: fine, hover: hover), so it gets the desktop path (finding 49).
// If a mouse moves on a device that first looked like a touch device (a tablet with a mouse), it switches to the desktop path then.
const FINE_POINTER = matchMedia("(pointer: fine)").matches && matchMedia("(hover: hover)").matches;
S.touchDevice = !FINE_POINTER && (COARSE || "ontouchstart" in window || navigator.maxTouchPoints > 0);
function applyTouchUi() {
  const t = S.touchDevice;
  $("hint-desk").hidden = t; $("hint-touch").hidden = !t;
  $("b-measure").hidden = !t;
  if (!S.started && !$("s-go").disabled) $("s-go").textContent = t ? "Tap to start" : "Click to start";
  updatePaused();
}
addEventListener("pointermove", (e) => { if (S.touchDevice && e.pointerType === "mouse") { S.touchDevice = false; ns.touchSwitched = true; applyTouchUi(); } }, { passive: true });
const STICK_R = 55;
const touchBlocked = (t) => !!(t.target && t.target.closest && t.target.closest("button, #mapwrap, .card, #rc-bar, input, textarea, #hud, #srch, #place, #tool, #padd"));
addEventListener("touchstart", (e) => {
  if (S.mode !== "play") return;
  for (const t of e.changedTouches) {
    if (touchBlocked(t)) continue;
    if (S.guide?.mode) S.tap = { id: t.identifier, x: t.clientX, y: t.clientY, t: performance.now() };      // a short tap with a tool on picks a point
    if (t.clientX < innerWidth / 2 && !S.touch.move) {
      S.touch.move = { id: t.identifier, x: t.clientX, y: t.clientY };
      const st = $("stick"); st.style.left = t.clientX + "px"; st.style.top = t.clientY + "px"; st.firstElementChild.style.transform = ""; $("touch").hidden = false;
    } else if (t.clientX >= innerWidth / 2 && !S.touch.look) {
      S.touch.look = { id: t.identifier, x: t.clientX, y: t.clientY };
    }
  }
  S.touchUsed = true;
  if (e.cancelable && !touchBlocked(e.changedTouches[0])) e.preventDefault();
}, { passive: false });
addEventListener("touchmove", (e) => {
  for (const t of e.changedTouches) {
    const m = S.touch.move, l = S.touch.look;
    if (m && t.identifier === m.id) {
      let dx = t.clientX - m.x, dy = t.clientY - m.y; const r = Math.hypot(dx, dy), k = r > STICK_R ? STICK_R / r : 1;
      dx *= k; dy *= k;
      S.touch.str = dx / STICK_R; S.touch.fwd = -dy / STICK_R;
      $("stick").firstElementChild.style.transform = `translate(${dx}px, ${dy}px)`;
    } else if (l && t.identifier === l.id && S.mode === "play") {
      look(-(t.clientX - l.x) * 0.005, -(t.clientY - l.y) * 0.005); l.x = t.clientX; l.y = t.clientY;
    }
  }
  if (S.touch.move || S.touch.look) e.preventDefault();
}, { passive: false });
const touchEnd = (e) => {
  for (const t of e.changedTouches) {
    const tp = S.tap;
    if (tp && tp.id === t.identifier && e.type === "touchend") {
      S.tap = null;
      if (S.guide?.mode && !S.guide.pending && Math.hypot(t.clientX - tp.x, t.clientY - tp.y) < 12 && performance.now() - tp.t < 500) S.guide.onCanvasClick(t.clientX, t.clientY);
    }
    if (S.touch.move && t.identifier === S.touch.move.id) { S.touch.move = null; S.touch.fwd = S.touch.str = 0; $("touch").hidden = true; }
    if (S.touch.look && t.identifier === S.touch.look.id) S.touch.look = null;
  }
  if (e.touches.length === 0) { S.touch.move = S.touch.look = null; S.touch.fwd = S.touch.str = 0; $("touch").hidden = true; }   // no finger left: drop any stale state
};
addEventListener("touchend", touchEnd); addEventListener("touchcancel", touchEnd);

function readInput() {
  const k = S.keys;
  let fwd = (k.has("KeyW") || k.has("ArrowUp") ? 1 : 0) - (k.has("KeyS") || k.has("ArrowDown") ? 1 : 0);
  let str = (k.has("KeyD") || k.has("ArrowRight") ? 1 : 0) - (k.has("KeyA") || k.has("ArrowLeft") ? 1 : 0);
  fwd += S.touch.fwd; str += S.touch.str;
  return { forward: Math.max(-1, Math.min(1, fwd)), strafe: Math.max(-1, Math.min(1, str)), run: k.has("ShiftLeft") || k.has("ShiftRight") };
}
function updateKeyOverlay() {
  if (!FLAGS.keys) return;
  const o = $("keyovl"); o.hidden = false;
  if (!o.firstChild) o.innerHTML = ["KeyW:W", "KeyA:A", "KeyS:S", "KeyD:D", "ShiftLeft:Shift"].map((s) => `<kbd data-k="${s.split(":")[0]}">${s.split(":")[1]}</kbd>`).join("");
  for (const el of o.children) { const c = el.dataset.k; el.classList.toggle("on", S.keys.has(c) || (c === "KeyW" && S.keys.has("ArrowUp")) || (c === "KeyA" && S.keys.has("ArrowLeft")) || (c === "KeyS" && S.keys.has("ArrowDown")) || (c === "KeyD" && S.keys.has("ArrowRight")) || (c === "ShiftLeft" && S.keys.has("ShiftRight"))); }
}

// ---------------------------------------------------------------------------------------------------------------- HUD
function toggleHud() {
  const h = $("hud"); h.hidden = false;
  h.classList.toggle("expanded"); h.classList.toggle("chip", !h.classList.contains("expanded"));
  clearTimeout(S.hudTimer);
}
$("hud").addEventListener("click", toggleHud);
const scaleText = () => scaleInfo(S.data.scale).short;
function qualityText() {
  const st = S.gov?.step; const lab = S.quality === "hd" ? "HD" : S.quality === "low" ? "low" : "SD";
  return st ? `${lab} · ${st.name} · ${S.pr.toFixed(2)}x` : lab;
}
function updateHud(force = false) {
  const now = performance.now();
  if (!force && now - S.hudLast < 250) return;
  S.hudLast = now;
  const fps = S.stats.fps;
  $("h-fps").textContent = fps ? fps.toFixed(0) : "--";
  $("chip-fps").textContent = fps ? fps.toFixed(0) : "--";
  $("h-p95").textContent = S.stats.win.length > 20 ? `${S.stats.p95.toFixed(1)} ms` : "--";
  $("h-splats").textContent = fmtInt(S.splatCount);
  $("h-mb").textContent = S.loadedBytes ? `${mb(S.loadedBytes)} MB` : "--";
  $("h-q").textContent = qualityText();
  $("h-scale").textContent = scaleText(); $("h-scale").title = scaleInfo(S.data.scale).long;
  $("chip-scale").textContent = ` · scale ${scaleText()}`;
  $("h-gpu").textContent = GPU + (SOFTWARE ? " (software)" : "");
}

// ---------------------------------------------------------------------------------------------------------------- minimap
function cycleMap() {
  const w = $("mapwrap");
  if (w.hidden) { w.hidden = false; w.classList.remove("large"); S.minimap.size(smallMapPx()); }
  else if (!w.classList.contains("large")) { w.classList.add("large"); S.minimap.size(Math.min(innerWidth, innerHeight) * 0.78); }
  else { w.hidden = true; w.classList.remove("large"); }
  $("map-title").textContent = w.classList.contains("large") ? "Map (M to shrink)" : "Map";
}
const smallMapPx = () => Math.round(Math.max(120, Math.min(220, Math.min(innerWidth, innerHeight) * 0.36)));
function mapPick(x, z) {
  if (S.mode !== "play") return;
  S.guide?.stopFollow(true);
  if (S.touchDevice && !$("mapwrap").classList.contains("large")) { cycleMap(); return; }   // phones: tap the small map to enlarge it
  if (S.walker.teleport(x, z)) { toast("Jumped"); if ($("mapwrap").classList.contains("large")) { $("mapwrap").classList.remove("large"); S.minimap.size(smallMapPx()); } }
  else toast("No walkable cell nearby");
}

// ---------------------------------------------------------------------------------------------------------------- tour benchmark
function startBench(auto = false) {
  const tour = S.tour;
  if (!tour || !tour.duration) { toast("This scene has no tour path"); return; }
  const secs = Math.min(60, +P.get("benchsecs") || 60);
  const path = new TourPath(S.data.tour, 60); path.retime(secs);
  if (document.pointerLockElement) document.exitPointerLock();
  S.bench = { path, start: null, secs, auto, gov: S.gov.enabled, level: S.gov.level, saved: { x: S.walker.x, z: S.walker.z, yaw: S.walker.yaw, pitch: S.walker.pitch, eyeY: S.eyeY } };
  S.gov.setEnabled(false);
  S.mode = "bench";
  $("hud").hidden = false; $("hud").classList.add("expanded"); $("hud").classList.remove("chip");
  $("b-title").textContent = "Tour benchmark"; $("b-res").textContent = ""; $("b-close").hidden = true; $("b-copy").hidden = true; $("b-fill").style.width = "0%";
  $("bench").hidden = false; $("bench").style.background = "transparent"; $("bench").style.backdropFilter = "none"; $("bench").style.alignItems = "flex-end"; $("bench").style.pointerEvents = "none";
  ns.benchResult = null;
}
function stepBench(now) {
  const b = S.bench;
  if (b.start === null) { b.start = now; S.stats.startRun(); }
  const t = (now - b.start) / 1000;
  const pose = b.path.at(t);
  S.walker.x = pose.x; S.walker.z = pose.z; S.walker.yaw = pose.yawDeg * Math.PI / 180; S.walker.pitch = pose.pitchDeg * Math.PI / 180; S.eyeY = S.data.camera.eye_height;
  $("b-fill").style.width = Math.min(100, (100 * t) / b.secs) + "%";
  if (t >= b.secs) finishBench();
}
function finishBench() {
  const b = S.bench; const r = S.stats.endRun();
  const res = {
    avg_fps: +r.fps_mean.toFixed(1), p50_ms: +r.p50_ms.toFixed(2), p95_ms: +r.p95_ms.toFixed(2), p99_ms: +r.p99_ms.toFixed(2), max_ms: +r.max_ms.toFixed(1),
    frames: r.frames, secs: +r.secs.toFixed(1), splats: S.splatCount, mb: +mb(S.loadedBytes), gpu: GPU + (SOFTWARE ? " (software)" : ""), software: SOFTWARE,
    quality: S.quality, pixel_ratio: S.pr, max_std_dev: +S.maxStdDev.toFixed(3), viewport: `${innerWidth}x${innerHeight}`, ua: navigator.userAgent, scene: S.data.name, when: new Date().toISOString(),
  };
  ns.benchResult = res;
  $("b-title").textContent = "Tour benchmark done";
  $("b-res").textContent = `avg ${res.avg_fps} fps   p95 ${res.p95_ms} ms   (p50 ${res.p50_ms}, p99 ${res.p99_ms})\n${fmtInt(res.splats)} splats   ${res.mb} MB\nGPU  ${res.gpu}\n${res.viewport}  quality ${res.quality}  pixel ratio ${res.pixel_ratio}  ${res.secs} s, ${res.frames} frames`;
  $("b-close").hidden = false; $("b-copy").hidden = false;
  $("bench").style.pointerEvents = "auto";
  S.gov.setEnabled(b.gov); S.bench = null; S.mode = "play";
  const sv = b.saved; S.walker.x = sv.x; S.walker.z = sv.z; S.walker.yaw = sv.yaw; S.walker.pitch = sv.pitch; S.eyeY = sv.eyeY;
  note("bench done " + JSON.stringify({ fps: res.avg_fps, p95: res.p95_ms }));
}
function stopBench(cancel) {
  const b = S.bench; if (!b) return;
  S.stats.endRun(); S.gov.setEnabled(b.gov); S.bench = null; S.mode = "play";
  const sv = b.saved; S.walker.x = sv.x; S.walker.z = sv.z; S.walker.yaw = sv.yaw; S.walker.pitch = sv.pitch; S.eyeY = sv.eyeY;
  $("bench").hidden = true; if (cancel) toast("Benchmark cancelled");
}
$("b-close").addEventListener("click", () => { $("bench").hidden = true; $("bench").removeAttribute("style"); });
$("b-copy").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(JSON.stringify(ns.benchResult, null, 2)); toast("Result copied"); } catch { toast("Copy failed: select the text instead"); }
});

// ---------------------------------------------------------------------------------------------------------------- reality check
const rcCam = new THREE.PerspectiveCamera(60, 1, 0.05, 50);
function setPrincipalPoint(cam, pp) {
  const e = cam.projectionMatrix.elements;
  e[8] = 1 - 2 * pp[0]; e[9] = 2 * pp[1] - 1;
  cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
}
function poseCamera(cam, h) {
  cam.fov = h.fov_y_deg; cam.aspect = h.aspect; cam.near = S.data.camera.near; cam.far = S.data.camera.far;
  cam.position.fromArray(h.position); cam.quaternion.fromArray(h.quaternion); cam.updateProjectionMatrix();
  if (h.principal_point) setPrincipalPoint(cam, h.principal_point);
  cam.updateMatrixWorld(true);
}
function layoutReality(h) {
  const availW = innerWidth - 24, availH = innerHeight - 130;
  let w = availW, hh = w / h.aspect; if (hh > availH) { hh = availH; w = hh * h.aspect; }
  w = Math.floor(w); hh = Math.floor(hh);
  const left = Math.floor((innerWidth - w) / 2), top = Math.floor(Math.max(8, (innerHeight - 100 - hh) / 2));
  renderer.setPixelRatio(1); renderer.setSize(w, hh);
  Object.assign(canvas.style, { position: "absolute", left: left + "px", top: top + "px" });
  Object.assign($("rc-box").style, { left: left + "px", top: top + "px", width: w + "px", height: hh + "px" });
  S.reality.box = { left, top, w, h: hh };
}
function enterReality(i) {
  const hs = S.data.heldout;
  if (!hs.length) { toast("This scene has no held-out photos"); return; }
  if (document.pointerLockElement) document.exitPointerLock();
  S.reality = { i: 0, slider: 50 };
  S.mode = "reality"; $("reality").hidden = false; $("mapwrap").style.visibility = "hidden"; $("bottom").style.visibility = "hidden";
  setReality(i);
}
function setReality(i) {
  const hs = S.data.heldout, n = hs.length;
  S.reality.i = ((i % n) + n) % n; const h = hs[S.reality.i];
  poseCamera(rcCam, h); layoutReality(h);
  $("rc-photo").src = h.photo;
  const ps = h.psnr > 0 ? ` · PSNR ${h.psnr.toFixed(1)} dB` : "";
  $("rc-cap").textContent = `Held-out photo ${S.reality.i + 1} of ${n}${ps}  (never seen in training)`;
  setSlider(S.reality.slider);
}
const stepReality = (d) => S.mode === "reality" && setReality(S.reality.i + d);
function setSlider(v) {
  S.reality.slider = v; $("rc-slider").value = v;
  $("rc-photo").style.clipPath = `inset(0 ${100 - v}% 0 0)`; $("rc-line").style.left = v + "%";
}
function exitReality() {
  S.mode = "play"; S.reality = null; $("reality").hidden = true; $("mapwrap").style.visibility = ""; $("bottom").style.visibility = ""; S.gov?.reset();
  canvas.style.position = ""; canvas.style.left = ""; canvas.style.top = "";
  renderer.setPixelRatio(S.pr); renderer.setSize(innerWidth, innerHeight);
}
$("rc-prev").addEventListener("click", () => stepReality(-1)); $("rc-next").addEventListener("click", () => stepReality(1));
$("rc-close").addEventListener("click", exitReality);
$("rc-slider").addEventListener("input", (e) => setSlider(+e.target.value));
{
  let dragging = false;
  const move = (e) => { const b = $("rc-box").getBoundingClientRect(); setSlider(Math.max(0, Math.min(100, ((e.clientX - b.left) / b.width) * 100))); };
  $("rc-box").addEventListener("pointerdown", (e) => { dragging = true; $("rc-box").setPointerCapture(e.pointerId); move(e); });
  $("rc-box").addEventListener("pointermove", (e) => dragging && move(e));
  $("rc-box").addEventListener("pointerup", () => (dragging = false));
}

// ---------------------------------------------------------------------------------------------------------------- render loop
function drawSplatProbe() {
  // after render, in the same task: does the canvas hold anything but the clear colour?
  const w = gl.drawingBufferWidth, h = gl.drawingBufferHeight, px = new Uint8Array(4);
  let hits = 0;
  for (let j = 1; j <= 6; j++) for (let i = 1; i <= 8; i++) {
    gl.readPixels(Math.floor((w * i) / 9), Math.floor((h * j) / 7), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    if (Math.abs(px[0] - LOAD_CLEAR[0]) + Math.abs(px[1] - LOAD_CLEAR[1]) + Math.abs(px[2] - LOAD_CLEAR[2]) > 12) hits++;
  }
  return hits >= 6;
}

// A hidden tab stops requestAnimationFrame. The first frame back would show a gap of seconds, so forget the timer and what the governor saw.
document.addEventListener("visibilitychange", () => { S.stats.last = null; S.gov?.reset(); });
addEventListener("pageshow", () => { S.stats.last = null; S.gov?.reset(); });

let lastNow = null;
function frame(now) {
  const dt = S.stats.tick(now);
  S.frames++;
  const dtS = Math.min(0.1, dt / 1000);
  if (S.mode === "play" && S.started && !modalOpen()) {
    const gd = S.guide;
    let inp = readInput();
    if (gd?.following) { if (inp.forward || inp.strafe) gd.stopFollow(); else inp = gd.followInput(dtS); }   // any move key hands control back
    S.walker.update(dtS, inp);
  }
  else if (S.mode === "bench" && S.bench) stepBench(now);
  else if (S.walker && S.mode === "play") S.walker.update(dtS, { forward: 0, strafe: 0, run: false });
  if (S.mode !== "reality") applyCamera();
  renderer.render(scene, S.mode === "reality" ? rcCam : camera);
  if (S.splatCount && !S.sawSplats && S.frames > 2 && drawSplatProbe()) {
    S.sawSplats = true; ns.firstFrameMs = performance.now(); ns.firstSplatCount = S.splatCount; note("first splat frame");
    useSceneClear();
    onFirstFrame();
  }
  if (S.guide) S.guide.update(now, dtS);
  if (S.sawSplats && dt > 0 && S.mode !== "reality" && S.gov) S.gov.push(dt, now);
  if (S.minimap && S.mode === "play" && !$("mapwrap").hidden) S.minimap.draw(S.walker);
  updateHud();
  lastNow = now;
}

function onFirstFrame() {
  $("hud").hidden = false; $("bottom").hidden = false;
  if (!S.started) {
    const b = $("s-go");
    b.disabled = false; b.textContent = S.touchDevice ? "Tap to start" : "Click to start";
  }
  if (!S.hudTimer) S.hudTimer = setTimeout(() => { if (S.mode !== "bench") { $("hud").classList.remove("expanded"); $("hud").classList.add("chip"); } }, 10000);
  if (FLAGS.bench) setTimeout(() => startBench(true), 600);
  ns.ready = true;
}

// ---------------------------------------------------------------------------------------------------------------- start card
function startPlay(e) {
  if (S.started) return;
  S.started = true; $("start").hidden = true;
  const byFinger = e && e.pointerType ? e.pointerType === "touch" : S.touchDevice;     // a mouse click on a touchscreen laptop still locks the pointer
  if (!byFinger) { try { const p = canvas.requestPointerLock?.(); p?.catch?.(() => {}); } catch {} }
  $("bottom").hidden = false;
}
$("s-go").addEventListener("click", startPlay);
$("b-measure").addEventListener("click", (e) => { S.guide?.beginMeasure(); e.currentTarget.blur(); });
$("b-hd").addEventListener("click", () => toggleHd().catch((e) => toast("HD failed: " + e.message)));
const hfovDeg = () => (2 * Math.atan(Math.tan((camera.fov * Math.PI) / 360) * camera.aspect) * 180) / Math.PI;
addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  if (S.minimap) S.minimap.fovDeg = hfovDeg();
  if (S.mode === "reality") layoutReality(S.data.heldout[S.reality.i]); else renderer.setSize(innerWidth, innerHeight);
  if (S.minimap) S.minimap.size($("mapwrap").classList.contains("large") ? Math.min(innerWidth, innerHeight) * 0.78 : smallMapPx());
});
// click on the scene after Esc resumes (pointer lock) via the pointerup handler above

// ---------------------------------------------------------------------------------------------------------------- test hooks
function installHooks() {
  const poseOf = () => ({ x: S.walker.x, y: S.eyeY, z: S.walker.z, yawDeg: S.walker.yawDeg, pitchDeg: S.walker.pitchDeg });
  Object.assign(ns, {
    pose: poseOf,
    setPose(x, y, z, yawDeg, pitchDeg) { setView(x, z, yawDeg, pitchDeg, y); },
    grid: S.grid, walker: S.walker, governor: S.gov, tourPath: S.tour, guide: S.guide,
    sim: {
      randomWalks: (o) => sim.randomWalks(S.grid, walkerCfg(), o),
      headingPushes: (o) => sim.headingPushes(S.grid, walkerCfg(), { x: S.data.spawn.position[0], z: S.data.spawn.position[2] }, o),
      frameRate: () => sim.frameRateIndependence(S.grid, walkerCfg(), { x: S.data.spawn.position[0], z: S.data.spawn.position[2] }, S.data.spawn.yaw_deg || 0),
    },
    /** Render an explicit pose (photo intrinsics) at the photo size and return PSNR against the photo, with an optional yaw offset. */
    async psnrAgainstPhoto(h, { yawOffsetDeg = 0, settleMs = 1500 } = {}) {
      const photo = await loadImage(h.photo);
      const w = photo.width, hh = photo.height;                // render at the photo's own size, so no resampling is involved
      const prev = { mode: S.mode, size: renderer.getSize(new THREE.Vector2()), pr: renderer.getPixelRatio() };
      S.mode = "reality";                                      // the loop renders rcCam
      renderer.setPixelRatio(1); renderer.setSize(w, hh, false);
      poseCamera(rcCam, { ...h, aspect: w / hh });
      if (yawOffsetDeg) { rcCam.rotateOnWorldAxis(new THREE.Vector3(0, 1, 0), (yawOffsetDeg * Math.PI) / 180); rcCam.updateMatrixWorld(true); }
      await new Promise((r) => setTimeout(r, settleMs));
      renderer.render(scene, rcCam);
      const buf = new Uint8Array(w * hh * 4); gl.readPixels(0, 0, w, hh, gl.RGBA, gl.UNSIGNED_BYTE, buf);
      const cv = document.createElement("canvas"); cv.width = w; cv.height = hh; const cx = cv.getContext("2d");
      cx.drawImage(photo, 0, 0, w, hh); const ph = cx.getImageData(0, 0, w, hh).data;
      let se = 0, n = 0;
      for (let y = 0; y < hh; y++) for (let x = 0; x < w; x++) {
        const a = ((hh - 1 - y) * w + x) * 4, b = (y * w + x) * 4;       // gl rows are bottom-up
        for (let c = 0; c < 3; c++) { const d = buf[a + c] - ph[b + c]; se += d * d; n++; }
      }
      const mse = se / n, psnr = 10 * Math.log10((255 * 255) / Math.max(mse, 1e-9));
      // restore
      S.mode = prev.mode; renderer.setPixelRatio(prev.pr); renderer.setSize(prev.size.x, prev.size.y);
      return { psnr, w, h: hh };
    },
    /** Read the current frame as RGBA bytes (call right after a render via renderNow). */
    startBench: () => startBench(true),
    openReality: (i = 0) => { S.started = true; $("start").hidden = true; enterReality(i); },
    closeReality: exitReality,
    selectQuality: (n) => S.gov.jumpTo(n),
  });
}
const walkerCfg = () => ({ eyeHeight: S.data.camera.eye_height, walkMps: S.data.camera.walk_mps, runMult: S.data.camera.run_mult });

// ---------------------------------------------------------------------------------------------------------------- boot
async function boot() {
  try {
    $("loader-sub").textContent = "Viewer code loaded. Reading the scene description.";
    const { url, index } = await resolveSceneUrl(location.href, P);
    note("scene url " + url);
    const raw = await fetchJson(url);
    const problems = validateScene(raw);
    if (problems.length) console.warn("scene.json problems:", problems.join("; "));
    S.data = normaliseScene(raw, url, FLAGS.asset ? new URL(FLAGS.asset, location.href).href : null);
    const d = S.data;
    document.title = `${d.title || d.name} · Namma Space`;
    $("s-title").textContent = d.title || d.name;
    const prov = d.provenance || {};
    $("s-sub").textContent = `${prov.kind === "fixture" ? "Test fixture. " : ""}${d.splats?.sd?.count ? fmtInt(d.splats.sd.count) + " Gaussian splats. " : ""}Walk through it like a game. It is a Gaussian-splat reconstruction${prov.kind === "fixture" ? "" : " of a real room"}.`;
    { const si = scaleInfo(d.scale); $("path-scale").innerHTML = si.checked
      ? `<b>Scale</b>: the stats badge says how the scale was checked (${si.short}). Press <kbd>K</kbd> and click two points to measure.`
      : `<b>Scale</b>: the stats badge shows an estimate of the scale and its 68% range, from camera height. It is not a measured error. Press <kbd>K</kbd> and click two points to measure.`; }
    $("s-prov").textContent = [prov.source, prov.license && `License ${prov.license}`, prov.attribution].filter(Boolean).join(" · ");
    const others = (index?.scenes || []).filter((x) => x.name !== d.name);
    if (others.length) {
      const el = $("s-others"); el.hidden = false;
      el.innerHTML = "Other scenes: " + others.map((x) => `<a href="?scene=${encodeURIComponent(x.name)}">${(x.title || x.name).replace(/[<&]/g, "")}</a>`).join(" &middot; ");
    }
    camera.fov = d.camera.fov_y_deg; camera.near = d.camera.near; camera.far = d.camera.far; camera.updateProjectionMatrix();
    S.eyeY = d.camera.eye_height;
    if (FLAGS.bench) S.started = true;
    $("start").hidden = FLAGS.bench;

    // renderer settings from scene.json
    const r = d.splats.render;
    const stdOverride = +P.get("std") || 0;                                               // ?std= and ?sortms= are tuning flags
    const fullStd = Math.min(r.maxStdDev || Math.sqrt(8), Math.sqrt(8));
    const sparkOpts = { renderer, maxStdDev: stdOverride || Math.min(fullStd, Math.sqrt(5)), preBlurAmount: r.preBlurAmount ?? 0, blurAmount: r.blurAmount ?? 0.3 };
    if (P.has("sortms")) sparkOpts.minSortIntervalMs = +P.get("sortms");
    if (P.has("radial")) sparkOpts.sortRadial = P.get("radial") !== "0";
    spark = new SparkRenderer(sparkOpts);
    scene.add(spark);
    ns.spark = spark;
    const basePR = WEAK ? (SOFTWARE ? 0.5 : 0.85) : 1;
    const lad = (hd) => buildLadder({ basePixelRatio: basePR, fullStd, stress: FLAGS.stress, hd, stdOverride, previewFallback: !!d.splats.preview && QEFF !== "low" });
    S.ladders = { sd: lad(false), hd: lad(true) };
    const ladder = QEFF === "hd" && d.splats.hd ? S.ladders.hd : S.ladders.sd;
    let start = 0;
    if (QEFF === "low") start = Math.max(0, ladder.findIndex((l) => l.name === "low"));          // no "low" step when it would equal "standard"
    // a step down is a trial: it stays only if the frame rate rises (capped or paused frames do not). The toast waits for that.
    S.gov = new Governor({
      ladder, start,
      onStep: (step, level, why, kind) => { applyStep(step, kind); note(`governor -> ${step.name} (${kind}: ${why})`); if (kind !== "down" && kind !== "revert") toast(`Quality: ${step.name}`, 1200); },
      onConfirm: (step) => { note(`governor kept ${step.name}`); toast(step.mesh === "preview" ? `Reduced detail: ${fmtInt(S.splatCount)} splats` : `Quality: ${step.name}`, step.mesh === "preview" ? 2500 : 1200); },
    });
    ns.governor = S.gov;
    applyStep(S.gov.step);

    $("loader-sub").textContent = "Loading the room files.";
    // grid, minimap, splats in parallel
    const gridP = decodeGrid(d.grid);
    const mapP = d.minimap?.url ? loadImage(d.minimap.url).catch(() => null) : Promise.resolve(null);
    S.grid = await gridP;
    S.walker = new Walker(S.grid, { eyeHeight: d.camera.eye_height, walkMps: d.camera.walk_mps, runMult: d.camera.run_mult });
    S.walker.bounds = { min: [d.bounds.min[0], d.bounds.min[2]], max: [d.bounds.max[0], d.bounds.max[2]] };
    const sp0 = d.spawn;
    setView(sp0.position[0], sp0.position[2], sp0.yaw_deg || 0, sp0.pitch_deg || 0, sp0.position[1]);
    S.eyeY = d.camera.eye_height;
    const pose = parsePose(FLAGS.pose);
    if (pose) { setView(pose.x, pose.z, pose.yaw, pose.pitch, pose.y); }
    S.tour = new TourPath(d.tour, 60);
    S.mode = "play";
    S.minimap = new Minimap($("map"), d.minimap || { origin_xz: d.grid.origin_xz, px_per_m: 40 }, S.grid, mapPick);
    S.minimap.fovDeg = hfovDeg();
    S.minimap.bookmarks = d.bookmarks.slice(0, 5).map((b, i) => ({ ...b, n: i + 1 }));
    S.minimap.size(smallMapPx());
    S.guide = new Guide({ S, camera, canvas, scene, toast, updatePaused });
    S.guide.init(d.places);
    applyTouchUi();                                                                          // phones have no K key: the Measure button and the thumb hint
    S.minimap.onPlace = (p) => { if (S.touchDevice && !$("mapwrap").classList.contains("large")) { cycleMap(); return; } S.guide.select(p); };
    if (FLAGS.place && S.guide.select(FLAGS.place)) { if (FLAGS.follow) S.guide.startFollow(); else if (FLAGS.route) S.guide.toggleRoute(); }
    S.minimap.buildTint();
    $("mapwrap").hidden = false;
    S.minimap.draw(S.walker);
    mapP.then((img) => { if (img) { S.minimap.setImage(img); S.minimap.draw(S.walker); } });
    installHooks();
    note("grid ready");
    applyCamera();
    renderer.setAnimationLoop(frame);
    if (SOFTWARE && !FLAGS.q) toast("No GPU acceleration detected: showing the reduced scene. Turn on hardware acceleration for the full one (or add ?q=sd).", 8000);
    await loadSplats();
    ns.loadedMs = performance.now();
  } catch (e) {
    console.error(e);
    showError("Could not load the scene", String(e && e.message || e));
  }
}
boot();
