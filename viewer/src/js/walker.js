// Player movement on the walkable grid. Pure logic: no DOM, no three.js, so Node can import it.
//
// Frame: world metres, +Y up. yaw 0 looks along -Z, positive yaw turns left (CONTRACTS 2.12).
// Grid pixel (row i, col j) is the cell centred at x = ox + (j + .5) cell, z = oz + (i + .5) cell.

export const WALKABLE = 255;

export class Grid {
  /** @param {{width:number,height:number,cell:number,origin_xz:number[]}} meta @param {Uint8Array} data one byte per cell, row-major */
  constructor(meta, data) {
    this.w = meta.width; this.h = meta.height; this.cell = meta.cell;
    this.ox = meta.origin_xz[0]; this.oz = meta.origin_xz[1];
    this.data = data;
    this.walkableValue = meta.values?.walkable ?? WALKABLE;
  }
  col(x) { return Math.floor((x - this.ox) / this.cell); }
  row(z) { return Math.floor((z - this.oz) / this.cell); }
  value(x, z) {
    const j = this.col(x), i = this.row(z);
    if (i < 0 || j < 0 || i >= this.h || j >= this.w) return 0;
    return this.data[i * this.w + j];
  }
  walkable(x, z) { return this.value(x, z) === this.walkableValue; }
  cellCentre(i, j) { return [this.ox + (j + 0.5) * this.cell, this.oz + (i + 0.5) * this.cell]; }
  /** Nearest walkable cell centre to (x, z), by ring search. Returns null when the grid has none. */
  nearestWalkable(x, z, maxRadiusCells = Math.max(this.w, this.h)) {
    const i0 = this.row(z), j0 = this.col(x);
    const at = (i, j) => i >= 0 && j >= 0 && i < this.h && j < this.w && this.data[i * this.w + j] === this.walkableValue;
    if (at(i0, j0)) return [x, z];
    for (let r = 1; r <= maxRadiusCells; r++) {
      let best = null, bd = Infinity;
      for (let di = -r; di <= r; di++) {
        for (let dj = -r; dj <= r; dj++) {
          if (Math.max(Math.abs(di), Math.abs(dj)) !== r) continue;
          const i = i0 + di, j = j0 + dj;
          if (!at(i, j)) continue;
          const [cx, cz] = this.cellCentre(i, j);
          const d = (cx - x) ** 2 + (cz - z) ** 2;
          if (d < bd) { bd = d; best = [cx, cz]; }
        }
      }
      if (best) return best;
    }
    return null;
  }
  walkableArea() { let n = 0; for (let k = 0; k < this.data.length; k++) if (this.data[k] === this.walkableValue) n++; return n * this.cell * this.cell; }
}

export class Walker {
  /**
   * @param {Grid|null} grid  null = free flight in the plane (no collision)
   * @param {{eyeHeight:number,walkMps:number,runMult:number,smoothTau?:number,pitchLimitDeg?:number}} cfg
   */
  constructor(grid, cfg) {
    this.grid = grid;
    this.eye = cfg.eyeHeight;
    this.walk = cfg.walkMps;
    this.run = cfg.runMult;
    this.tau = cfg.smoothTau ?? 0.075;          // exponential smoothing: ~86 % of target speed after 0.15 s
    this.pitchLimit = (cfg.pitchLimitDeg ?? 80) * Math.PI / 180;
    this.x = 0; this.z = 0; this.yaw = 0; this.pitch = 0; this.vx = 0; this.vz = 0;
    this.blockedX = false; this.blockedZ = false;
    this.fly = false;
    this.bounds = null;                          // {min:[x,z], max:[x,z]} used in fly mode
  }
  /** yaw in radians; deg helpers below. */
  setPose(x, z, yawDeg, pitchDeg = 0) {
    this.x = x; this.z = z; this.yaw = yawDeg * Math.PI / 180; this.pitch = pitchDeg * Math.PI / 180; this.vx = 0; this.vz = 0;
    this.pushOut();
  }
  get yawDeg() { return this.yaw * 180 / Math.PI; }
  get pitchDeg() { return this.pitch * 180 / Math.PI; }
  look(dyaw, dpitch) {
    this.yaw += dyaw;
    this.pitch = Math.max(-this.pitchLimit, Math.min(this.pitchLimit, this.pitch + dpitch));
  }
  /** If the player stands in a blocked cell (spawn, teleport), move to the nearest walkable one. */
  pushOut() {
    if (!this.grid || this.fly) return;
    if (this.grid.walkable(this.x, this.z)) return;
    const p = this.grid.nearestWalkable(this.x, this.z);
    if (p) { this.x = p[0]; this.z = p[1]; this.vx = this.vz = 0; }
  }
  teleport(x, z) {
    if (this.grid && !this.fly) {
      const p = this.grid.nearestWalkable(x, z);
      if (!p) return false;
      x = p[0]; z = p[1];
    }
    this.x = x; this.z = z; this.vx = this.vz = 0; return true;
  }
  /**
   * Advance by dt seconds.
   * @param {number} dt
   * @param {{forward:number,strafe:number,run:boolean}} input  forward/strafe in [-1, 1]
   */
  update(dt, input) {
    if (dt <= 0) return;
    // target velocity in world axes. Forward = (-sin yaw, -cos yaw); right = (cos yaw, -sin yaw).
    const sy = Math.sin(this.yaw), cy = Math.cos(this.yaw);
    let f = input.forward, s = input.strafe;
    const len = Math.hypot(f, s);
    if (len > 1) { f /= len; s /= len; }
    const speed = this.walk * (input.run ? this.run : 1);
    const tx = (-sy * f + cy * s) * speed;
    const tz = (-cy * f - sy * s) * speed;
    const a = 1 - Math.exp(-dt / this.tau);
    const vx0 = this.vx, vz0 = this.vz;
    this.vx += (tx - this.vx) * a;
    this.vz += (tz - this.vz) * a;
    // distance covered this frame uses the mean of the old and new velocity (exact enough at 5-100 ms)
    const dx = 0.5 * (vx0 + this.vx) * dt, dz = 0.5 * (vz0 + this.vz) * dt;
    this.move(dx, dz);
  }
  move(dx, dz) {
    this.blockedX = this.blockedZ = false;
    const g = this.grid;
    if (!g || this.fly) {
      this.x += dx; this.z += dz;
      if (this.bounds) {
        const b = this.bounds, nx = Math.min(b.max[0], Math.max(b.min[0], this.x)), nz = Math.min(b.max[1], Math.max(b.min[1], this.z));
        if (nx !== this.x) { this.blockedX = true; this.vx = 0; }
        if (nz !== this.z) { this.blockedZ = true; this.vz = 0; }
        this.x = nx; this.z = nz;
      }
      return;
    }
    // sub-steps of at most half a cell, x and z resolved separately so the player slides along walls
    const maxStep = 0.5 * g.cell;
    const n = Math.max(1, Math.ceil(Math.max(Math.abs(dx), Math.abs(dz)) / maxStep));
    const sx = dx / n, sz = dz / n;
    for (let k = 0; k < n; k++) {
      if (sx !== 0) { if (g.walkable(this.x + sx, this.z)) this.x += sx; else { this.blockedX = true; this.vx = 0; } }
      if (sz !== 0) { if (g.walkable(this.x, this.z + sz)) this.z += sz; else { this.blockedZ = true; this.vz = 0; } }
    }
  }
  get speed() { return Math.hypot(this.vx, this.vz); }
}
