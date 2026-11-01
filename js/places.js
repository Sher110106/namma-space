// Places: reading scene.json "places", fuzzy search, and the JSON the place editor writes. Pure logic (Node can import it).
//
// place = {id, name, aliases[], kind: "room" | "item" | "zone", position: [x, y, z], yaw_deg, radius_m, notes?}   (CONTRACTS 2.12; notes is an optional extra)

export const KINDS = ["room", "item", "zone"];
export const DEFAULT_RADIUS = { room: 1.5, zone: 1.0, item: 0.4 };

/** Lower case, no accents or punctuation, single spaces. Keeps letters and digits of any script. */
export function norm(s) {
  return String(s ?? "").toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").replace(/[^\p{L}\p{N}\p{M}]+/gu, " ").trim().replace(/\s+/g, " ");
}
export function slug(s) { return norm(s).replace(/ /g, "-").slice(0, 40) || "place"; }

/** Fill defaults and drop broken entries. `y` is the pin height when position has only [x, z]. */
export function normalisePlace(p, i = 0) {
  if (!p || typeof p !== "object") return null;
  const name = String(p.name ?? "").trim();
  const pos = Array.isArray(p.position) ? p.position.map(Number) : [];
  if (!name || pos.length < 2 || pos.some((v) => !Number.isFinite(v))) return null;
  const kind = KINDS.includes(p.kind) ? p.kind : "item";
  const position = pos.length >= 3 ? pos.slice(0, 3) : [pos[0], kind === "item" ? 1.0 : 0.0, pos[1]];
  const aliases = (Array.isArray(p.aliases) ? p.aliases : typeof p.aliases === "string" ? p.aliases.split(",") : []).map((a) => String(a).trim()).filter(Boolean);
  const radius = Number.isFinite(+p.radius_m) && +p.radius_m > 0 ? +p.radius_m : DEFAULT_RADIUS[kind];
  const out = { id: String(p.id || slug(name) || `p${i + 1}`), name, aliases, kind, position, yaw_deg: Number.isFinite(+p.yaw_deg) ? +p.yaw_deg : 0, radius_m: radius };
  if (typeof p.notes === "string" && p.notes.trim()) out.notes = p.notes.trim();       // optional: who placed it and how
  return out;
}
/** Unique ids: a repeated id gets -2, -3, ... */
export function uniqueIds(list) {
  const seen = new Set();
  return list.map((p) => { let id = p.id, n = 2; while (seen.has(id)) id = `${p.id}-${n++}`; seen.add(id); return id === p.id ? p : { ...p, id }; });
}
export function readPlaces(raw) { return uniqueIds((Array.isArray(raw) ? raw : []).map(normalisePlace).filter(Boolean)); }

/** JSON for scene.json "places": 3 decimals, one place per block. */
export function placesToJson(list) {
  const r = (v) => Math.round(v * 1000) / 1000;
  return JSON.stringify(list.map((p) => ({ id: p.id, name: p.name, aliases: p.aliases || [], kind: p.kind, position: p.position.map(r), yaw_deg: r(p.yaw_deg || 0), radius_m: r(p.radius_m), ...(p.notes ? { notes: p.notes } : {}) })), null, 2);
}

/** World yaw in degrees that looks from (fx, fz) toward (tx, tz): yaw 0 looks along -Z, positive turns left. */
export function yawToward(fx, fz, tx, tz) { return (Math.atan2(-(tx - fx), -(tz - fz)) * 180) / Math.PI; }
export function pitchToward(eyeY, dist, targetY, limitDeg = 45) {
  const p = (Math.atan2(targetY - eyeY, Math.max(dist, 0.3)) * 180) / Math.PI;
  return Math.max(-limitDeg, Math.min(limitDeg, p));
}

// ---------------------------------------------------------------------------------------------------------------- fuzzy search
/** Edit distance with adjacent swaps counted as one edit (optimal string alignment). */
export function editDistance(a, b) {
  const n = a.length, m = b.length;
  if (!n || !m) return Math.max(n, m);
  let p2 = new Array(m + 1).fill(0), p1 = Array.from({ length: m + 1 }, (_, j) => j);
  for (let i = 1; i <= n; i++) {
    const cur = [i];
    for (let j = 1; j <= m; j++) {
      const c = a[i - 1] === b[j - 1] ? 0 : 1;
      let v = Math.min(p1[j] + 1, cur[j - 1] + 1, p1[j - 1] + c);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) v = Math.min(v, p2[j - 2] + 1);
      cur.push(v);
    }
    p2 = p1; p1 = cur;
  }
  return p1[m];
}
/** How well query word q matches candidate word c: 0 (no) to 1 (same). One typo is allowed for words of 4+ letters, two for 8+. */
function wordScore(q, c) {
  if (q === c) return 1;
  if (c.startsWith(q)) return 0.9;
  if (q.length >= 3 && c.includes(q)) return 0.6;
  if (q.length >= 4) {
    const tol = q.length >= 8 ? 2 : 1;
    if (editDistance(q, c) <= tol) return 0.75;
    for (const L of [q.length - 1, q.length, q.length + 1]) if (L > 0 && L < c.length && editDistance(q, c.slice(0, L)) <= tol) return 0.6;   // typo while still typing
  }
  return 0;
}
function textScore(qWords, text) {
  const words = text.split(" ");
  let sum = 0;
  for (const q of qWords) { let best = 0; for (const w of words) best = Math.max(best, wordScore(q, w)); if (!best) return 0; sum += best; }
  const s = sum / qWords.length;
  return norm(qWords.join(" ")) === text ? s + 0.2 : s;
}
/**
 * Ranked matches over names and aliases. An empty query lists everything (by distance when `from` is given, else by name).
 * @returns {{place, score, via: string}[]}
 */
export function searchPlaces(places, query, { limit = 8, from = null } = {}) {
  const q = norm(query), qWords = q ? q.split(" ") : [];
  const dist = (p) => (from ? Math.hypot(p.position[0] - from.x, p.position[2] - from.z) : 0);
  if (!qWords.length) {
    return [...places].sort((a, b) => (from ? dist(a) - dist(b) : a.name.localeCompare(b.name))).slice(0, limit).map((place) => ({ place, score: 0, via: place.name }));
  }
  const out = [];
  for (const place of places) {
    let best = textScore(qWords, norm(place.name)), via = place.name;
    for (const a of place.aliases || []) { const s = 0.95 * textScore(qWords, norm(a)); if (s > best) { best = s; via = a; } }
    if (best > 0) out.push({ place, score: best, via });
  }
  out.sort((a, b) => b.score - a.score || (from ? dist(a.place) - dist(b.place) : 0) || a.place.name.localeCompare(b.place.name));
  return out.slice(0, limit);
}
