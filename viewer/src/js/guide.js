// Places, search, route, follow, place editor and measure tool ("Round 2 preview"). Browser side of nav.js and places.js.
// Drawing: the route is a ribbon mesh on the floor (drawn after the splats, no depth test); pins, labels and the measure line are a 2D overlay canvas.
import * as THREE from "three";
import { routeToPlace, standSpot, standBackFor, followStep, projectOnPolyline, arclengths, pointAtS, polylineLength } from "./nav.js";
import { scaleInfo } from "./scene.js";
import { readPlaces, searchPlaces, normalisePlace, uniqueIds, placesToJson, slug, yawToward, pitchToward, DEFAULT_RADIUS } from "./places.js";

const $ = (id) => document.getElementById(id);
const PIN = "#ff5d73", ROUTE = "#5eead4";
const KIND_LABEL = { room: "Room", item: "Item", zone: "Zone" };
const fmtM = (m) => (m < 10 ? m.toFixed(1) : Math.round(m).toString()) + " m";
const fmtT = (s) => (s < 90 ? `${Math.max(1, Math.round(s))} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`);

const RIBBON_VS = `
varying vec2 vUv; varying float vDist;
void main() { vUv = uv; vec4 mv = modelViewMatrix * vec4(position, 1.0); vDist = -mv.z; gl_Position = projectionMatrix * mv; }`;
const RIBBON_FS = `
precision highp float;
uniform float uTime, uS0; uniform vec3 uColor;
varying vec2 vUv; varying float vDist;
void main() {
  float across = abs(vUv.y - 0.5) * 2.0;
  float ahead = smoothstep(uS0 - 0.3, uS0 + 0.4, vUv.x);          // hide the part the player has walked past
  float near = smoothstep(0.35, 1.3, vDist);                        // fade right under the camera
  float far = 1.0 - smoothstep(14.0, 26.0, vDist);
  float p = fract(vUv.x / 0.55 - uTime * 0.9);
  float q = 0.62 - across * 0.40;                                   // chevron with its tip forward
  float d = abs(fract(p - q + 0.5) - 0.5);
  float chev = 1.0 - smoothstep(0.07, 0.13, d);
  float edge = smoothstep(1.0, 0.8, across);
  vec3 col = mix(vec3(0.01, 0.10, 0.10), uColor, chev);
  gl_FragColor = vec4(col, (0.5 + 0.5 * chev) * edge * ahead * near * far);
}`;

export class Guide {
  /**
   * @param {object} c  {S, camera, canvas, scene, toast, onTeleport}  S is main.js state (S.walker, S.grid, S.data, S.active, S.eyeY, S.minimap)
   */
  constructor(c) {
    Object.assign(this, c);
    this.base = []; this.local = []; this.places = [];
    this.sel = null; this.routeOn = false; this.route = null; this.following = false; this.fst = null;
    this.mode = null;                 // null | "add" | "measure"
    this.meas = { a: null, b: null }; this.pending = null; this.added = 0;
    this.ver = 0; this.hi = 0; this.useSplats = true; this.lastRouteAt = -1e9; this.infoAt = 0; this.stuck = null;
    this.raycaster = new THREE.Raycaster();
    this.ov = $("overlay"); this.octx = this.ov.getContext("2d");
    this.mesh = null; this.uni = { uTime: { value: 0 }, uS0: { value: -10 }, uColor: { value: new THREE.Color(ROUTE) } };
    this.v = new THREE.Vector3();
  }

  // ------------------------------------------------------------------------------------------------------------ setup
  init(rawPlaces) {
    this.base = readPlaces(rawPlaces);
    this.local = this.loadLocal();
    this.rebuild();
    this.wire();
    this.resize();
    addEventListener("resize", () => this.resize());
    this.refreshChrome();
  }
  get key() { return `namma-space.places.${this.S.data.name}`; }
  loadLocal() { try { return readPlaces(JSON.parse(localStorage.getItem(this.key) || "[]")); } catch { return []; } }
  saveLocal() { try { localStorage.setItem(this.key, JSON.stringify(this.local)); } catch {} }
  rebuild() {
    const ids = new Set(this.base.map((p) => p.id));
    this.local = this.local.filter((p) => !ids.has(p.id));              // pasted into scene.json already: the scene's copy wins
    this.places = [...this.base, ...this.local];
    this.localIds = new Set(this.local.map((p) => p.id));
    if (this.sel && !this.places.find((p) => p.id === this.sel.id)) this.clearSelection();
    this.bump();
  }
  bump() {
    this.ver++;
    const m = this.S.minimap;
    if (m) { m.places = this.places; m.selId = this.sel?.id ?? null; m.route = this.routeOn && this.route?.ok ? this.route.points : null; m.stand = this.routeOn && this.route?.ok ? this.route.stand : null; m.ver = this.ver; }
  }
  refreshChrome() {
    const has = this.places.length > 0;
    for (const el of document.querySelectorAll(".needs-places")) el.hidden = !has;
    this.refreshChromeTick(this.S.mode === "play");
  }

  wire() {
    const inp = $("srch-in");
    $("srch-btn").addEventListener("click", () => this.openSearch());
    $("srch-x").addEventListener("click", () => this.closeSearch());
    inp.addEventListener("input", () => { this.hi = 0; this.renderList(); });
    inp.addEventListener("keydown", (e) => {
      const n = this.results.length;
      if (e.key === "ArrowDown") { e.preventDefault(); if (n) this.hi = (this.hi + 1) % n; this.renderList(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); if (n) this.hi = (this.hi - 1 + n) % n; this.renderList(); }
      else if (e.key === "Enter") { e.preventDefault(); const r = this.results[this.hi]; if (r) this.choose(r.place); else this.toast("No place matches that"); }
    });
    $("srch-list").addEventListener("pointerdown", (e) => {
      const li = e.target.closest("li[data-id]"); if (!li) return;
      e.preventDefault(); const p = this.places.find((q) => q.id === li.dataset.id); if (p) this.choose(p);
    });
    $("pl-jump").addEventListener("click", () => this.jump());
    $("pl-route").addEventListener("click", () => this.toggleRoute());
    $("pl-follow").addEventListener("click", () => this.toggleFollow());
    $("pl-x").addEventListener("click", () => this.clearSelection());
    $("pl-rm").addEventListener("click", () => this.removeLocal(this.sel));
    $("tool-copy").addEventListener("click", () => this.copyJson());
    $("tool-done").addEventListener("click", () => this.endMode());
    $("tool-clear").addEventListener("click", () => { this.meas = { a: null, b: null }; this.updateToolText(); });
    $("padd-ok").addEventListener("click", () => this.commitAdd());
    $("padd-cancel").addEventListener("click", () => this.cancelAdd());
    $("padd-name").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); this.commitAdd(); } });
    $("padd-alias").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); this.commitAdd(); } });
    for (const b of $("padd-kind").children) b.addEventListener("click", () => { for (const o of $("padd-kind").children) o.classList.toggle("on", o === b); });
    $("jsonbox-close").addEventListener("click", () => ($("jsonbox").hidden = true));
    // a clicked button keeps keyboard focus, and then Space would press it again: let go of it
    for (const id of ["place", "tool", "padd", "jsonbox", "srch"]) $(id).addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setTimeout(() => b.blur(), 0); });
  }
  resize() {
    const d = Math.min(2, devicePixelRatio || 1);
    this.ov.width = Math.round(innerWidth * d); this.ov.height = Math.round(innerHeight * d); this.dpr = d;
  }

  // ------------------------------------------------------------------------------------------------------------ search
  get searchOpen() { return !$("srch").hidden; }
  openSearch() {
    if (!this.places.length) { this.toast("This scene has no places yet"); return; }
    if (document.pointerLockElement) document.exitPointerLock();
    $("srch").hidden = false; $("srch-btn").hidden = true; this.updatePaused?.();
    const inp = $("srch-in"); inp.value = ""; this.hi = 0; this.renderList(); inp.focus();
  }
  closeSearch() { $("srch").hidden = true; $("srch-in").blur(); this.refreshChrome(); this.updatePaused?.(); }
  renderList() {
    const w = this.S.walker, q = $("srch-in").value;
    this.results = searchPlaces(this.places, q, { limit: 8, from: { x: w.x, z: w.z } });
    const ul = $("srch-list");
    ul.innerHTML = "";
    if (!this.results.length) {
      const li = document.createElement("li"); li.className = "none";
      const hint = this.places.slice(0, 3).map((p) => p.name).join(", ");
      li.textContent = `No match. Try ${hint}.`; ul.appendChild(li); return;
    }
    this.results.forEach((r, k) => {
      const p = r.place, li = document.createElement("li");
      li.dataset.id = p.id; li.className = k === this.hi ? "hi" : "";
      const d = Math.hypot(p.position[0] - w.x, p.position[2] - w.z);
      const alias = r.via !== p.name ? ` · ${r.via}` : "";
      li.innerHTML = `<i class="k-${p.kind}"></i><span class="nm"></span><span class="meta"></span>`;
      li.querySelector(".nm").textContent = p.name;
      li.querySelector(".meta").textContent = `${KIND_LABEL[p.kind].toLowerCase()}${alias} · ${fmtM(d)}`;
      ul.appendChild(li);
    });
  }
  choose(place) {
    this.closeSearch();
    this.select(place);
  }

  // ------------------------------------------------------------------------------------------------------------ selection and card
  select(place, { keepRoute = false } = {}) {
    if (typeof place === "string") place = this.places.find((p) => p.id === place);
    if (!place) return null;
    if (!keepRoute) { this.routeOn = false; this.route = null; this.stopFollow(true); this.setMesh(null); }
    this.sel = place;
    $("place").hidden = false;
    $("pl-name").textContent = place.name;
    $("pl-kind").textContent = KIND_LABEL[place.kind];
    $("pl-rm").hidden = !this.localIds.has(place.id);
    this.updateCard(true);
    this.bump();
    return place;
  }
  clearSelection() {
    this.stopFollow(true); this.sel = null; this.routeOn = false; this.route = null; this.setMesh(null);
    $("place").hidden = true; this.bump();
  }
  /** Card text: straight-line distance, or the route's length and walking time (live). */
  updateCard(force = false) {
    const now = performance.now();
    if (!this.sel || (!force && now - this.infoAt < 250)) return;
    this.infoAt = now;
    const w = this.S.walker, p = this.sel, info = $("pl-info");
    let txt, bad = false;
    if (this.routeOn && this.route && !this.route.ok) { txt = this.route.reason; bad = true; }
    else if (this.routeOn && this.route?.ok) {
      const pr = projectOnPolyline(this.route.points, w.x, w.z), left = Math.max(0, this.route.length - pr.s);
      txt = left < 0.4 ? "You are here." : `${fmtM(left)} left · about ${fmtT(left / this.S.data.camera.walk_mps)} on foot`;
      if (this.route.note) txt += ` ${this.route.note}`;
    } else {
      const d = Math.hypot(p.position[0] - w.x, p.position[2] - w.z);
      txt = `${fmtM(d)} away in a straight line`;
    }
    info.textContent = txt; info.classList.toggle("bad", bad);
    $("pl-route").classList.toggle("on", this.routeOn);
    $("pl-route").firstChild.textContent = this.routeOn ? "Hide route " : "Route ";
    $("pl-follow").hidden = !(this.routeOn && this.route?.ok);
    $("pl-follow").classList.toggle("on", this.following);
    $("pl-follow").firstChild.textContent = this.following ? "Stop " : "Follow ";
  }

  // ------------------------------------------------------------------------------------------------------------ jump
  /** Stand spot and view for a place: nearest walkable cell (items: 0.8 m back), facing the place. */
  standFor(place) {
    const g = this.S.grid, [px, py, pz] = place.position;
    const s = standSpot(g, px, pz, { minDist: standBackFor(place) });
    if (!s) return null;
    const yaw = s.d > 0.3 ? yawToward(s.x, s.z, px, pz) : place.yaw_deg || 0;
    const pitch = s.d > 0.3 ? pitchToward(this.S.data.camera.eye_height, s.d, py) : 0;
    return { x: s.x, z: s.z, yaw, pitch, d: s.d };
  }
  jump(place = this.sel) {
    if (!place) return false;
    const st = this.standFor(place); if (!st) { this.toast("No walkable floor to stand on"); return false; }
    this.stopFollow(true);
    const w = this.S.walker;
    w.fly = false; w.teleport(st.x, st.z); w.yaw = (st.yaw * Math.PI) / 180; w.pitch = (st.pitch * Math.PI) / 180;
    this.S.eyeY = this.S.data.camera.eye_height;
    this.toast(`Jumped to ${place.name}`);
    if (this.routeOn) this.computeRoute(true);
    this.updateCard(true);
    return true;
  }

  // ------------------------------------------------------------------------------------------------------------ route
  toggleRoute() {
    if (!this.sel) return;
    if (this.routeOn) { this.stopFollow(true); this.routeOn = false; this.route = null; this.setMesh(null); this.bump(); this.updateCard(true); return; }
    this.routeOn = true; this.computeRoute(true); this.updateCard(true);
    if (this.route && !this.route.ok) this.toast(this.route.reason, 3500);
  }
  computeRoute(force = false) {
    const now = performance.now();
    if (!force && now - this.lastRouteAt < 500) return;
    this.lastRouteAt = now;
    const w = this.S.walker;
    this.route = routeToPlace(this.S.grid, { x: w.x, z: w.z }, this.sel, { walkMps: this.S.data.camera.walk_mps });
    if (this.route.ok) { this.route.s = arclengths(this.route.points); this.setMesh(this.route.points); } else this.setMesh(null);
    this.fst = { s: 0 };
    this.bump();
  }
  setMesh(points) {
    if (this.mesh) { this.scene.remove(this.mesh); this.mesh.geometry.dispose(); this.mesh = null; }
    if (!points || points.length < 2) return;
    const y = (this.S.data.frame?.floor_y ?? 0) + 0.03, hw = 0.09, n = points.length;
    const pos = new Float32Array(n * 6), uv = new Float32Array(n * 4), idx = [];
    let s = 0;
    for (let k = 0; k < n; k++) {
      const p = points[k];
      if (k) s += Math.hypot(p[0] - points[k - 1][0], p[1] - points[k - 1][1]);
      const unit = (a, b) => { const dx = b[0] - a[0], dz = b[1] - a[1], L = Math.hypot(dx, dz) || 1; return [dx / L, dz / L]; };
      const d0 = k ? unit(points[k - 1], p) : unit(p, points[k + 1]), d1 = k < n - 1 ? unit(p, points[k + 1]) : d0;
      const n0 = [-d0[1], d0[0]], n1 = [-d1[1], d1[0]];
      let mx = n0[0] + n1[0], mz = n0[1] + n1[1]; const ml = Math.hypot(mx, mz);
      if (ml < 1e-6) { mx = n0[0]; mz = n0[1]; } else { mx /= ml; mz /= ml; }
      const sc = hw / Math.max(0.5, mx * n0[0] + mz * n0[1]);
      pos.set([p[0] + mx * sc, y, p[1] + mz * sc, p[0] - mx * sc, y, p[1] - mz * sc], k * 6);
      uv.set([s, 0, s, 1], k * 4);
      if (k) { const a = 2 * (k - 1), b = 2 * k; idx.push(a, a + 1, b, a + 1, b + 1, b); }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3)); geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2)); geo.setIndex(idx);
    const mat = new THREE.ShaderMaterial({ uniforms: this.uni, vertexShader: RIBBON_VS, fragmentShader: RIBBON_FS, transparent: true, depthTest: false, depthWrite: false, side: THREE.DoubleSide });
    this.mesh = new THREE.Mesh(geo, mat); this.mesh.frustumCulled = false; this.mesh.renderOrder = 999;
    this.scene.add(this.mesh);
  }

  // ------------------------------------------------------------------------------------------------------------ follow
  toggleFollow() { this.following ? this.stopFollow() : this.startFollow(); }
  startFollow() {
    if (!this.sel) return;
    if (!this.routeOn) { this.routeOn = true; }
    this.computeRoute(true);
    if (!this.route?.ok) { this.updateCard(true); this.toast(this.route?.reason || "No route", 3500); return; }
    const w = this.S.walker; w.fly = false; w.pushOut();
    if (document.pointerLockElement) document.exitPointerLock();
    this.following = true; this.fst = { s: 0 }; this.stuck = { t: performance.now(), x: w.x, z: w.z, tries: 0 };
    this.toast("Following the route. Press any move key to stop.", 2600);
    this.updateCard(true);
  }
  stopFollow(silent = false) {
    if (!this.following) return;
    this.following = false; this.fst = null; this.stuck = null;
    if (!silent) this.toast("Stopped");
    this.updateCard(true);
  }
  /** Walker input while following (replaces the keyboard). Also keeps the route fresh and detects arrival and getting stuck. */
  followInput(dt) {
    const w = this.S.walker, S = this.S;
    const st = this.fst, route = this.route;
    if (!route?.ok) { this.stopFollow(true); return { forward: 0, strafe: 0, run: false }; }
    const face = (yawToward(route.stand.x, route.stand.z, this.sel.position[0], this.sel.position[2]) * Math.PI) / 180;
    const standD = Math.hypot(route.stand.x - this.sel.position[0], route.stand.z - this.sel.position[2]);
    const input = followStep(route.points, w, st, dt, { faceYaw: standD > 0.3 ? face : null });
    const tp = standD > 0.3 ? (pitchToward(S.data.camera.eye_height, standD, this.sel.position[1]) * Math.PI) / 180 : 0;
    const want = st.arrived ? tp : -0.08;
    w.pitch += (want - w.pitch) * Math.min(1, dt * 3);
    if (st.done) {
      this.following = false; this.fst = null; this.stuck = null;
      this.toast(`Arrived: ${this.sel.name}`, 2200); this.updateCard(true);
      return { forward: 0, strafe: 0, run: false };
    }
    const now = performance.now();
    if (st.off > 0.5) this.computeRoute();                                   // pushed off the path: plan again from here
    if (!st.arrived && this.stuck) {
      if (now - this.stuck.t > 2000) {
        if (Math.hypot(w.x - this.stuck.x, w.z - this.stuck.z) < 0.15) {
          if (this.stuck.tries++ >= 1) { this.stopFollow(true); this.toast("The route is blocked here. Stopped.", 3000); return { forward: 0, strafe: 0, run: false }; }
          this.computeRoute(true);
        }
        this.stuck.t = now; this.stuck.x = w.x; this.stuck.z = w.z;
      }
    }
    return input;
  }

  // ------------------------------------------------------------------------------------------------------------ per-frame
  /** Called every frame from main.js after the render. */
  update(now, dt) {
    const S = this.S, on = S.mode === "play";
    this.ov.hidden = !on;
    if (this.mesh) this.mesh.visible = on;
    this.uni.uTime.value = now / 1000;
    this.refreshChromeTick(on);
    if (!on) return;
    if (this.routeOn && this.sel && !this.following && S.started) {                // manual walking: re-plan when the player leaves the path
      const w = S.walker;
      if (this.route?.ok) { const pr = projectOnPolyline(this.route.points, w.x, w.z); this.uni.uS0.value = pr.s; if (pr.dist > 0.5 && !w.fly) this.computeRoute(); }
      else if (!this.route) this.computeRoute();
    } else if (this.routeOn && this.following && this.route?.ok) this.uni.uS0.value = projectOnPolyline(this.route.points, S.walker.x, S.walker.z).s;
    this.updateCard();
    this.draw();
  }
  refreshChromeTick(on) {
    const want = this.places.length > 0 && this.S.started && on && !this.searchOpen && !this.pending;
    if ($("srch-btn").hidden === want) $("srch-btn").hidden = !want;
  }

  // ------------------------------------------------------------------------------------------------------------ overlay drawing
  /** World point -> {x, y, behind, dist} in CSS pixels. */
  project(wx, wy, wz) {
    const cam = this.camera, v = this.v.set(wx, wy, wz).applyMatrix4(cam.matrixWorldInverse);
    const dist = Math.hypot(v.x, v.y, v.z), behind = v.z >= -0.05, W = innerWidth, H = innerHeight;
    if (behind) return { x: W / 2 + v.x * 1000, y: H / 2 - v.y * 1000, behind: true, dist };   // behind the camera: only the side matters (the edge arrow turns you toward it)
    const f = 1 / Math.tan((cam.fov * Math.PI) / 360), k = 1 / -v.z;
    return { x: ((v.x * k * f) / cam.aspect * 0.5 + 0.5) * W, y: (-(v.y * k * f) * 0.5 + 0.5) * H, behind: false, dist };
  }
  draw() {
    const ctx = this.octx, d = this.dpr;
    const busy = !!(this.sel || this.pending || this.meas.a || this.mode === "measure");
    if (!busy) { if (this.drawn) { ctx.setTransform(1, 0, 0, 1, 0, 0); ctx.clearRect(0, 0, this.ov.width, this.ov.height); this.drawn = false; } return; }    // nothing to draw: leave the canvas alone
    this.drawn = true;
    ctx.setTransform(d, 0, 0, d, 0, 0); ctx.clearRect(0, 0, innerWidth, innerHeight);
    this.camera.updateMatrixWorld(); this.camera.matrixWorldInverse.copy(this.camera.matrixWorld).invert();
    this.avoid = ["place", "tool", "padd", "srch"].map((id) => $(id)).filter((e) => e && !e.hidden).map((e) => e.getBoundingClientRect());   // labels move out from under panels
    if (this.sel) { const [x, y, z] = this.sel.position; this.drawPin(ctx, this.project(x, y, z), this.sel.name, fmtM(this.distTo(this.sel)), PIN); }
    if (this.pending) { const [x, y, z] = this.pending.pos; this.drawDot(ctx, this.project(x, y, z), "#fbbf24"); }
    if (this.mode === "measure" || this.meas.a) this.drawMeasure(ctx);
  }
  distTo(p) { const w = this.S.walker; return Math.hypot(p.position[0] - w.x, p.position[2] - w.z); }
  drawDot(ctx, p, color) {
    if (p.behind) return;
    ctx.beginPath(); ctx.arc(p.x, p.y, 6, 0, 7); ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = "#0b0e14"; ctx.stroke();
    ctx.beginPath(); ctx.arc(p.x, p.y, 11, 0, 7); ctx.lineWidth = 1.5; ctx.strokeStyle = "rgba(255,255,255,.85)"; ctx.stroke();
  }
  drawPin(ctx, p, title, sub, color) {
    const W = innerWidth, H = innerHeight, m = 26;
    const sc = Math.max(0.8, Math.min(1.35, 3.2 / Math.max(p.dist, 0.5)));          // nearer = larger, always readable
    const off = p.behind || p.x < m || p.x > W - m || p.y < m || p.y > H - m;
    ctx.save();
    ctx.font = `700 ${Math.round(14 * sc)}px system-ui, sans-serif`;
    const t1 = title, t2 = sub, w1 = ctx.measureText(t1).width;
    ctx.font = `500 ${Math.round(12 * sc)}px system-ui, sans-serif`;
    const w2 = ctx.measureText(t2).width, bw = Math.max(w1, w2) + 20 * sc, bh = 40 * sc;
    if (off) {
      // off screen: an arrow at the edge, pointing toward the place, with the name
      const cx = W / 2, cy = H / 2; let dx = p.x - cx, dy = p.y - cy; const L = Math.hypot(dx, dy) || 1; dx /= L; dy /= L;
      const k = Math.min((W / 2 - 60) / Math.abs(dx || 1e-6), (H / 2 - 70) / Math.abs(dy || 1e-6));
      const ax = cx + dx * k, ay = cy + dy * k;
      ctx.translate(ax, ay);
      ctx.save(); ctx.rotate(Math.atan2(dy, dx)); ctx.beginPath(); ctx.moveTo(18 * sc, 0); ctx.lineTo(-8 * sc, -13 * sc); ctx.lineTo(-3 * sc, 0); ctx.lineTo(-8 * sc, 13 * sc); ctx.closePath();
      ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = "#0b0e14"; ctx.stroke(); ctx.restore();
      const lx = Math.max(-ax + bw / 2 + 8, Math.min(W - ax - bw / 2 - 8, 0)), ly = dy > 0 ? -bh - 16 * sc : 16 * sc;
      this.pill(ctx, lx - bw / 2, ly, bw, bh, t1, t2, sc, null);
    } else {
      ctx.translate(p.x, p.y);
      ctx.beginPath(); ctx.arc(0, 0, 7 * sc, 0, 7); ctx.fillStyle = color; ctx.fill(); ctx.lineWidth = 2.5; ctx.strokeStyle = "#fff"; ctx.stroke();
      ctx.beginPath(); ctx.arc(0, 0, 13 * sc, 0, 7); ctx.lineWidth = 2; ctx.strokeStyle = "rgba(255,93,115,.55)"; ctx.stroke();
      const lx = Math.max(8 - p.x + bw / 2, Math.min(W - 8 - p.x - bw / 2, 0));
      const hit = (y0) => (this.avoid || []).some((r) => p.x + lx - bw / 2 < r.right && p.x + lx + bw / 2 > r.left && p.y + y0 < r.bottom && p.y + y0 + bh + 8 > r.top);
      const above = -22 * sc - bh, below = 22 * sc + 7 * sc;
      if (hit(above) && !hit(below)) this.pill(ctx, lx - bw / 2, below, bw, bh, t1, t2, sc, 0, true);
      else this.pill(ctx, lx - bw / 2, above, bw, bh, t1, t2, sc, 0);
    }
    ctx.restore();
  }
  pill(ctx, x, y, w, h, t1, t2, sc, tipX = 0, tipUp = false) {
    const r = 9 * sc;
    ctx.beginPath(); if (ctx.roundRect) ctx.roundRect(x, y, w, h, r); else ctx.rect(x, y, w, h); ctx.fillStyle = "rgba(14,18,26,.9)"; ctx.fill(); ctx.lineWidth = 1; ctx.strokeStyle = "rgba(255,255,255,.3)"; ctx.stroke();
    if (tipX !== null) {
      const ty = tipUp ? y : y + h, dir = tipUp ? -1 : 1;
      ctx.beginPath(); ctx.moveTo(tipX - 6 * sc, ty); ctx.lineTo(tipX, ty + dir * 7 * sc); ctx.lineTo(tipX + 6 * sc, ty); ctx.closePath(); ctx.fillStyle = "rgba(14,18,26,.9)"; ctx.fill();
    }
    ctx.textAlign = "center"; ctx.textBaseline = "alphabetic";
    ctx.fillStyle = "#fff"; ctx.font = `700 ${Math.round(14 * sc)}px system-ui, sans-serif`; ctx.fillText(t1, x + w / 2, y + 17 * sc);
    ctx.fillStyle = "#9aa6b8"; ctx.font = `500 ${Math.round(12 * sc)}px system-ui, sans-serif`; ctx.fillText(t2, x + w / 2, y + 33 * sc);
  }
  drawMeasure(ctx) {
    const { a, b } = this.meas, pa = a && this.project(...a.p), pb = b && this.project(...b.p);
    if (pa && pb && !pa.behind && !pb.behind) {
      ctx.save(); ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.lineWidth = 5; ctx.strokeStyle = "rgba(11,14,20,.85)"; ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pa.x, pa.y); ctx.lineTo(pb.x, pb.y); ctx.lineWidth = 2.5; ctx.strokeStyle = "#fbbf24"; ctx.stroke(); ctx.restore();
    }
    if (pa) this.drawDot(ctx, pa, "#fbbf24");
    if (pb) this.drawDot(ctx, pb, "#fbbf24");
    if (a && b && pa && pb && !(pa.behind && pb.behind)) {
      const r = this.measureResult(); const mx = (pa.x + pb.x) / 2, my = (pa.y + pb.y) / 2;
      const p = { x: Math.max(60, Math.min(innerWidth - 60, mx)), y: Math.max(70, Math.min(innerHeight - 70, my)), behind: false, dist: 3 };
      this.drawPin(ctx, p, `${r.d.toFixed(2)} m`, r.pm, "#fbbf24");
    }
  }

  // ------------------------------------------------------------------------------------------------------------ picking (add place, measure)
  /** Surface point under a screen position: splat raycast first (nine rays, see surfaceHit), floor plane as fallback. */
  pick(cx, cy) {
    const cv = this.canvas, rect = cv.getBoundingClientRect();
    const ndc = (x, y) => ({ x: ((x - rect.left) / rect.width) * 2 - 1, y: -((y - rect.top) / rect.height) * 2 + 1 });
    this.camera.updateMatrixWorld();
    const mesh = this.S.active;
    if (this.useSplats && mesh) {
      const rays = [[0, 0]], R = 7;
      for (let k = 0; k < 8; k++) rays.push([R * Math.cos((k * Math.PI) / 4), R * Math.sin((k * Math.PI) / 4)]);
      const per = [], keep = mesh.minRaycastOpacity;
      mesh.minRaycastOpacity = 0.12;                         // thin surfaces (screens, glass) hold few opaque splats
      try {
        for (const [dx, dy] of rays) { this.raycaster.setFromCamera(ndc(cx + dx, cy + dy), this.camera); per.push(this.raycaster.intersectObject(mesh, false).map((h) => h.distance)); }
      } catch (e) { per.length = 0; }
      mesh.minRaycastOpacity = keep;
      const h = per.length ? surfaceHit(per) : null;
      if (h) { this.raycaster.setFromCamera(ndc(cx, cy), this.camera); const r = this.raycaster.ray, p = r.origin.clone().addScaledVector(r.direction, h.distance); return { p: [p.x, p.y, p.z], source: "splat", dist: h.distance, support: h.support, hits: per[0].length }; }
    }
    this.raycaster.setFromCamera(ndc(cx, cy), this.camera);
    const ray = this.raycaster.ray, fy = this.S.data.frame?.floor_y ?? 0;
    if (ray.direction.y < -1e-3) {
      const t = (fy - ray.origin.y) / ray.direction.y;
      if (t > 0 && t < 15) return { p: [ray.origin.x + ray.direction.x * t, fy, ray.origin.z + ray.direction.z * t], source: "floor", dist: t, support: 0, hits: 0 };
    }
    return null;
  }
  get toolActive() { return this.mode !== null && !this.pending; }
  /** Canvas click or tap while a tool is active. (clientX, clientY) is the pointer, or the screen centre when the pointer is locked. */
  onCanvasClick(cx, cy) {
    if (!this.mode) return false;
    const hit = this.pick(cx, cy);
    if (!hit) { this.toast("Nothing there. Look at a surface or the floor."); return true; }
    if (this.mode === "measure") {
      if (!this.meas.a || this.meas.b) this.meas = { a: { p: hit.p, source: hit.source }, b: null };
      else this.meas.b = { p: hit.p, source: hit.source };
      this.updateToolText();
    } else if (this.mode === "add") {
      this.pending = { pos: hit.p, yaw: this.S.walker.yawDeg, source: hit.source };
      this.openForm();
    }
    return true;
  }
  measureResult() {
    const { a, b } = this.meas, d = Math.hypot(a.p[0] - b.p[0], a.p[1] - b.p[1], a.p[2] - b.p[2]);
    const si = scaleInfo(this.S.data.scale), pct = si.pct;
    const pm = pct > 0 ? `± ${(d * pct / 100).toFixed(2)} m (scale ${si.short})` : si.checked ? si.short : "scale not checked";
    return { d, pct, pm, checked: si.checked, sources: [a.source, b.source], err: pct > 0 ? (d * pct) / 100 : null };
  }

  // ------------------------------------------------------------------------------------------------------------ tools: banner, add place, measure
  beginMode(mode) {
    if (this.mode === mode) return this.endMode();
    if (this.mode) this.endMode(true);
    if (document.pointerLockElement) document.exitPointerLock();
    this.mode = mode; this.meas = { a: null, b: null }; this.added = 0; this.closeSearch();
    $("tool").hidden = false; this.canvas.classList.add("pickmode");
    this.updateToolText(); this.updatePaused?.();
  }
  beginAdd() { this.beginMode("add"); }
  beginMeasure() { this.beginMode("measure"); }
  endMode(quiet = false) {
    if (!this.mode) return;
    this.mode = null; this.pending = null; $("padd").hidden = true; $("tool").hidden = true; this.canvas.classList.remove("pickmode");
    this.meas = { a: null, b: null };
    if (!quiet) this.toast("Tool off");
    this.refreshChrome(); this.updatePaused?.();
  }
  updateToolText() {
    const msg = $("tool-msg"), add = this.mode === "add";
    $("tool-copy").hidden = !add; $("tool-clear").hidden = add;
    if (add) {
      const n = this.local.length;
      msg.innerHTML = `<b>Add place.</b> Click the spot, then name it. ${n ? `${n} saved in this browser.` : ""}`;
      $("tool-copy").textContent = `Copy places JSON${n ? ` (${this.places.length})` : ""}`;
    } else {
      const { a, b } = this.meas;
      if (!a) msg.innerHTML = "<b>Measure.</b> Click the first point.";
      else if (!b) msg.innerHTML = "<b>Measure.</b> Click the second point.";
      else {
        const r = this.measureResult(), sc = this.S.data.scale || {};
        msg.innerHTML = `<b>${r.d.toFixed(2)} m</b> ${r.pct > 0 ? `<span class="dim">± ${(r.err).toFixed(2)} m</span>` : ""}<br><span class="dim small">${escapeHtml(scaleInfo(sc).long)}${!sc.checked_against && sc.method ? ` Method: ${escapeHtml(sc.method)}.` : ""} Points on ${r.sources.includes("floor") ? "splats or the floor plane" : "splat surfaces"}. Click again to start over.</span>`;
      }
    }
  }
  openForm() {
    $("padd").hidden = false;
    const p = this.pending.pos; $("padd-pos").textContent = `at ${p.map((v) => v.toFixed(2)).join(", ")}`;
    $("padd-name").value = ""; $("padd-alias").value = "";
    for (const o of $("padd-kind").children) o.classList.toggle("on", o.dataset.k === "item");
    $("padd-name").focus();
  }
  cancelAdd() { this.pending = null; $("padd").hidden = true; }
  commitAdd() {
    const name = $("padd-name").value.trim();
    if (!name) { $("padd-name").focus(); this.toast("Type a name first"); return; }
    const kind = [...$("padd-kind").children].find((o) => o.classList.contains("on"))?.dataset.k || "item";
    const raw = { id: slug(name), name, aliases: $("padd-alias").value, kind, position: this.pending.pos.map((v) => Math.round(v * 1000) / 1000), yaw_deg: Math.round(this.pending.yaw * 10) / 10, radius_m: DEFAULT_RADIUS[kind] };
    const p = normalisePlace(raw); if (!p) return;
    const ids = new Set(this.places.map((q) => q.id)); let id = p.id, k = 2; while (ids.has(id)) id = `${p.id}-${k++}`; p.id = id;
    this.local.push(p); this.saveLocal(); this.added++;
    this.pending = null; $("padd").hidden = true; this.rebuild(); this.refreshChrome(); this.updateToolText();
    this.select(p);
    this.toast(`Added ${p.name}. Click another spot, or press Copy places JSON.`, 3000);
  }
  removeLocal(p) {
    if (!p || !this.localIds.has(p.id)) return;
    this.local = this.local.filter((q) => q.id !== p.id); this.saveLocal(); this.clearSelection(); this.rebuild(); this.refreshChrome(); this.updateToolText();
    this.toast(`Removed ${p.name}`);
  }
  async copyJson() {
    const txt = placesToJson(this.places);
    this.lastJson = txt;
    try { await navigator.clipboard.writeText(txt); this.toast(`Copied ${this.places.length} places. Paste them into scene.json as "places".`, 3500); }
    catch { $("jsonbox-text").value = txt; $("jsonbox").hidden = false; $("jsonbox-text").focus(); $("jsonbox-text").select(); }
  }

  // ------------------------------------------------------------------------------------------------------------ keys and escape
  /** Returns true when the key was used. */
  onKey(e) {
    if (e.code === "Slash" && e.key === "/") { this.searchOpen ? this.closeSearch() : this.openSearch(); return true; }
    switch (e.code) {
      case "KeyJ": if (this.sel) this.jump(); else this.toast("Pick a place first: press /"); return true;
      case "KeyG": if (this.sel) this.toggleRoute(); else this.toast("Pick a place first: press /"); return true;
      case "Space": if (this.sel) { this.toggleFollow(); return true; } return false;
      case "KeyP": this.beginAdd(); return true;
      case "KeyK": this.beginMeasure(); return true;
    }
    return false;
  }
  /** Esc: close the innermost thing. Returns true when it closed something. */
  escape() {
    if ($("jsonbox") && !$("jsonbox").hidden) { $("jsonbox").hidden = true; return true; }
    if (this.searchOpen) { this.closeSearch(); return true; }
    if (this.pending) { this.cancelAdd(); return true; }
    if (this.mode) { this.endMode(); return true; }
    if (this.following) { this.stopFollow(); return true; }
    return false;
  }
}

const escapeHtml = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

/**
 * The surface under the centre ray. `per[0]` holds the sorted hit distances of the centre ray, per[1..] those of rays a few pixels around it.
 * The first centre hit that at least `need` of the other rays confirm (a hit within a distance tolerance) is the surface, so a lone floater is ignored.
 * Falls back to the best-supported centre hit with one confirming ray. Returns {distance, support} or null.
 */
export function surfaceHit(per, need = 2, minDist = 0.15) {
  const centre = per[0] || []; let best = null;
  for (const d of centre) {
    if (d < minDist) continue;
    const tol = 0.08 + 0.05 * d;
    let support = 0;
    for (let r = 1; r < per.length; r++) if (per[r].some((x) => Math.abs(x - d) <= tol)) support++;
    if (support >= need) return { distance: d, support };
    if (support >= 1 && (!best || support > best.support)) best = { distance: d, support };
  }
  return best;
}
