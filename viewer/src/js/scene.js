// scene.json reader (schema namma-space/scene@1, CONTRACTS 2.12). Plain ES module, no dependencies.

export const REQUIRED_KEYS = ["schema", "name", "title", "provenance", "frame", "splats", "camera", "spawn", "bounds", "grid", "minimap",
  "bookmarks", "tour", "heldout", "places", "scale", "metrics", "build"];

export function validateScene(s) {
  const problems = [];
  for (const k of REQUIRED_KEYS) if (!(k in s)) problems.push(`missing key: ${k}`);
  if (s.schema !== "namma-space/scene@1") problems.push(`unexpected schema: ${s.schema}`);
  if (!s.splats?.sd?.url) problems.push("splats.sd.url missing");
  if (!s.grid?.url) problems.push("grid.url missing");
  return problems;
}

const DEFAULTS = {
  camera: { eye_height: 1.5, fov_y_deg: 70, near: 0.05, far: 50, walk_mps: 1.4, run_mult: 2.0 },
  render: { antialiased: true, preBlurAmount: 0.0, blurAmount: 0.3, maxStdDev: 2.828 },
  transform: { position: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: 1.0 },
};

/** Fill optional fields and make every URL absolute against the scene.json location (or ?asset=). */
export function normaliseScene(s, sceneUrl, assetOverride = null) {
  const out = JSON.parse(JSON.stringify(s));
  out.camera = { ...DEFAULTS.camera, ...(s.camera || {}) };
  out.splats = out.splats || {};
  out.splats.transform = { ...DEFAULTS.transform, ...(s.splats?.transform || {}) };
  out.splats.render = { ...DEFAULTS.render, ...(s.splats?.render || {}) };
  const abs = (u) => new URL(u, sceneUrl).href;
  for (const k of ["sd", "hd", "preview"]) {
    const e = out.splats[k];
    if (e && e.url) e.url = abs(e.url);
  }
  if (assetOverride) {
    // ?asset=<file url> replaces the SD file; ?asset=<dir url>/ replaces the folder that holds all three.
    // The scene's own file is kept as `fallback`, used if the override cannot be fetched (no CORS, dead mirror).
    if (assetOverride.endsWith("/")) {
      for (const k of ["sd", "hd", "preview"]) {
        const e = out.splats[k];
        if (e && e.url) { e.fallback = e.url; e.url = new URL(e.url.split("/").pop(), assetOverride).href; }
      }
    } else if (out.splats.sd) {
      out.splats.sd.fallback = out.splats.sd.url;
      out.splats.sd.url = assetOverride;
    } else {
      out.splats.sd = { url: assetOverride, count: 0, sh_degree: 0, bytes: 0 };
    }
  }
  if (out.grid) out.grid.url = abs(out.grid.url);
  if (out.minimap) out.minimap.url = abs(out.minimap.url);
  out.heldout = (out.heldout || []).map((h) => ({ ...h, photo: abs(h.photo) }));
  out.bookmarks = out.bookmarks || [];
  out.tour = out.tour || [];
  return out;
}

export async function fetchJson(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

/** Resolve which scene to load: ?scene= wins, else scenes/index.json default. Returns the absolute scene.json URL. */
export async function resolveSceneUrl(pageUrl, params) {
  const base = new URL("./scenes/", pageUrl);
  const asked = params.get("scene");
  if (asked) {
    const index = await fetchJson(new URL("index.json", base).href).catch(() => null);   // only used for the "other scenes" links
    return { url: new URL(`${encodeURIComponent(asked)}/scene.json`, base).href, index };
  }
  const index = await fetchJson(new URL("index.json", base).href);
  const name = index.default || index.scenes?.[0]?.name;
  if (!name) throw new Error("scenes/index.json has no default scene");
  return { url: new URL(`${encodeURIComponent(name)}/scene.json`, base).href, index };
}

const fmtPct = (v) => (v < 10 ? v.toFixed(1) : String(Math.round(v)));

/**
 * What the viewer says about the scale (review findings 29 and 44). The 68% range in scene.json comes from a prior on the camera
 * height, not from a measurement, so it is called an estimate. Only a scene whose scale.checked_against is set (a tape check,
 * written by bundle.py from 40_align/rescale.json) gets the word "checked", and then no range is quoted from the prior.
 */
export function scaleInfo(sc) {
  sc = sc || {};
  const pct = +sc.pct_68 > 0 ? +sc.pct_68 : 0;
  if (sc.checked_against) {
    const against = String(sc.checked_against);
    const e = typeof sc.check_error_pct === "number" && Number.isFinite(sc.check_error_pct) ? sc.check_error_pct : null;
    const signed = e === null ? "" : `${e >= 0 ? "+" : "-"}${Math.abs(e).toFixed(1)}%`;
    return { checked: true, pct: 0, short: e === null ? `checked (${against})` : `${against} check ${signed}`,
      long: `Scale checked against a ${against} measure${e === null ? "" : `: the automatic scale was off by ${signed}`}.` };
  }
  if (pct > 0) return { checked: false, pct, short: `est. ±${fmtPct(pct)}%`, long: `Estimated scale, 68% range ±${fmtPct(pct)}%. It comes from a prior on camera height; it is not a measured error.` };
  return { checked: false, pct: 0, short: "unchecked", long: "The scale of this scene was not checked." };
}

/** True when the HD entry would load the same splats as the SD entry (older bundles listed an identical copy; review findings 19 and 47). */
export function hdSameAsSd(sp) {
  if (!sp?.hd || !sp?.sd) return false;
  if (sp.hd.url === sp.sd.url) return true;
  return !!sp.hd.bytes && sp.hd.bytes === sp.sd.bytes && sp.hd.count === sp.sd.count && sp.hd.sh_degree === sp.sd.sh_degree;
}

/** The colour behind the splats where nothing is drawn (holes, a wall closer than the model has data for). [r, g, b] 0-255 from scene.json, else a neutral grey. */
export const DEFAULT_CLEAR = [60, 58, 55];
export function clearColorOf(render) {
  const c = render?.clear_color;
  if (Array.isArray(c) && c.length === 3 && c.every((v) => Number.isFinite(v) && v >= 0 && v <= 255)) return c.map((v) => Math.round(v));
  return DEFAULT_CLEAR.slice();
}
