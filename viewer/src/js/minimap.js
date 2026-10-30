// Minimap: scene minimap.png as the base, walkable cells tinted, bookmarks, player arrow and view cone. Click to teleport.

export class Minimap {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {{origin_xz:number[],px_per_m:number}} meta  minimap.json block of scene.json
   * @param {import('./walker.js').Grid} grid
   */
  constructor(canvas, meta, grid, onPick) {
    this.c = canvas; this.ctx = canvas.getContext("2d"); this.meta = meta; this.grid = grid; this.onPick = onPick;
    this.img = null; this.tint = null; this.bookmarks = []; this.fovDeg = 70;
    this.worldW = grid.w * grid.cell; this.worldH = grid.h * grid.cell;
    this.size(220);
    canvas.addEventListener("click", (e) => this.click(e));
    canvas.addEventListener("pointerdown", (e) => e.stopPropagation());
  }
  /** Fit the canvas to a long side of `px` CSS pixels, keeping the world aspect. */
  size(px) {
    const a = this.worldW / this.worldH;
    const w = a >= 1 ? px : Math.round(px * a), h = a >= 1 ? Math.round(px / a) : px;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    this.c.width = Math.round(w * dpr); this.c.height = Math.round(h * dpr);
    this.c.style.width = w + "px"; this.c.style.height = h + "px";
    this.k = (this.c.width / this.worldW);   // canvas px per metre
    this.dpr = dpr;
  }
  setImage(img) { this.img = img; this.imgVersion = (this.imgVersion || 0) + 1; this.lastKey = null; }
  buildTint() {
    const g = this.grid, cv = document.createElement("canvas");
    cv.width = g.w; cv.height = g.h;
    const x = cv.getContext("2d"), id = x.createImageData(g.w, g.h);
    for (let i = 0; i < g.data.length; i++) {
      const v = g.data[i], o = i * 4;
      if (v === g.walkableValue) { id.data[o] = 94; id.data[o + 1] = 234; id.data[o + 2] = 212; id.data[o + 3] = 70; }
      else if (v === 0) { id.data[o] = 6; id.data[o + 1] = 8; id.data[o + 2] = 12; id.data[o + 3] = 120; }
    }
    x.putImageData(id, 0, 0);
    this.tint = cv;
  }
  toCanvas(x, z) { return [(x - this.grid.ox) * this.k, (z - this.grid.oz) * this.k]; }
  fromCanvas(px, pz) { return [px / this.k + this.grid.ox, pz / this.k + this.grid.oz]; }
  click(e) {
    const r = this.c.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * this.c.width, pz = ((e.clientY - r.top) / r.height) * this.c.height;
    const [x, z] = this.fromCanvas(px, pz);
    e.stopPropagation();
    this.onPick(x, z);
  }
  /** Base layer (image + walkable tint) is drawn once per size or image change; only the overlay is redrawn per call. */
  baseLayer() {
    const W = this.c.width, H = this.c.height;
    if (this.base && this.base.width === W && this.base.height === H && this.baseVersion === this.imgVersion) return this.base;
    const cv = this.base && this.base.width === W && this.base.height === H ? this.base : document.createElement("canvas");
    cv.width = W; cv.height = H; this.base = cv; this.baseVersion = this.imgVersion;
    const ctx = cv.getContext("2d");
    ctx.fillStyle = "#0a0d13"; ctx.fillRect(0, 0, W, H);
    if (this.img) {
      // place the image by its own origin and px_per_m, so it need not match the grid extent exactly
      const ppm = this.meta.px_per_m || 40, o = this.meta.origin_xz || [this.grid.ox, this.grid.oz];
      const [ix, iz] = this.toCanvas(o[0], o[1]);
      ctx.drawImage(this.img, ix, iz, (this.img.width / ppm) * this.k, (this.img.height / ppm) * this.k);
    }
    if (this.tint) { ctx.imageSmoothingEnabled = false; ctx.drawImage(this.tint, 0, 0, W, H); ctx.imageSmoothingEnabled = true; }
    return cv;
  }
  draw(player, force = false) {
    const key = `${player.x.toFixed(2)},${player.z.toFixed(2)},${player.yaw.toFixed(2)},${this.c.width},${this.imgVersion}`;
    if (!force && key === this.lastKey) return;       // nothing moved: skip the redraw
    this.lastKey = key;
    const ctx = this.ctx, W = this.c.width, H = this.c.height;
    ctx.clearRect(0, 0, W, H);
    ctx.drawImage(this.baseLayer(), 0, 0);
    const d = this.dpr;
    for (const b of this.bookmarks) {
      const [bx, bz] = this.toCanvas(b.position[0], b.position[2]);
      ctx.fillStyle = "rgba(251,191,36,.95)"; ctx.beginPath(); ctx.arc(bx, bz, 4 * d, 0, 7); ctx.fill();
      ctx.fillStyle = "#1a1300"; ctx.font = `700 ${7 * d}px sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(String(b.n), bx, bz + 0.4 * d);
    }
    const [px, pz] = this.toCanvas(player.x, player.z);
    const ang = Math.atan2(-Math.cos(player.yaw), -Math.sin(player.yaw));   // screen angle of the forward vector (x right, z down)
    const half = (this.fovDeg * Math.PI / 180) / 2, len = 2.2 * this.k;
    ctx.fillStyle = "rgba(255,255,255,.22)"; ctx.beginPath(); ctx.moveTo(px, pz); ctx.arc(px, pz, len, ang - half, ang + half); ctx.closePath(); ctx.fill();
    ctx.save(); ctx.translate(px, pz); ctx.rotate(ang + Math.PI / 2);
    ctx.fillStyle = "#fff"; ctx.strokeStyle = "#0b0e14"; ctx.lineWidth = 1.5 * d;
    ctx.beginPath(); ctx.moveTo(0, -7 * d); ctx.lineTo(5 * d, 6 * d); ctx.lineTo(0, 3 * d); ctx.lineTo(-5 * d, 6 * d); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.restore();
  }
}
