// Checks test/data/conventions.json against the real three.js (vendored by W5), in Node.
// Run: node test/e2e/conventions.test.mjs
// For each pose: a three.js camera at the expected position/quaternion must look along the expected
// world direction, and a world point must land on the same pixel as the COLMAP pinhole route.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import * as THREE from '../../viewer/vendor/three.module.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const data = JSON.parse(fs.readFileSync(path.join(here, '..', 'data', 'conventions.json'), 'utf8'));
const tol = data.tolerance;
const near = (a, b, t, msg) => assert.ok(Math.abs(a - b) <= t, `${msg}: ${a} vs ${b}`);
let n = 0;

for (const p of data.poses) {
  const [x, y, z] = p.threejs.position;
  const [qx, qy, qz, qw] = p.threejs.quaternion_xyzw;
  const cam = new THREE.PerspectiveCamera(60, 1, 0.05, 50);
  cam.position.set(x, y, z);
  cam.quaternion.set(qx, qy, qz, qw).normalize();
  cam.updateMatrixWorld(true);
  const dir = cam.getWorldDirection(new THREE.Vector3());
  p.threejs.forward_world.forEach((v, i) => near([dir.x, dir.y, dir.z][i], v, 1e-9, `${p.id} forward[${i}]`));

  // COLMAP route: x_cam = R X + t ; three.js route: camera-local = inverse(world matrix) * X, with y and z flipped
  const R = p.colmap.R_w2c, t = p.colmap.t_w2c;
  for (const X of [[0.3, 0.5, -1.0], [1.0, 2.0, 3.0], [-2.0, 0.1, 0.7]]) {
    const xc = [0, 1, 2].map((i) => R[i][0] * X[0] + R[i][1] * X[1] + R[i][2] * X[2] + t[i]);
    const local = new THREE.Vector3(...X).applyMatrix4(cam.matrixWorldInverse);
    near(local.x, xc[0], 1e-9, `${p.id} local.x`);
    near(local.y, -xc[1], 1e-9, `${p.id} local.y`);
    near(local.z, -xc[2], 1e-9, `${p.id} local.z`);
    n++;
  }
}

for (const c of data.t_final_cases) {
  const p = data.poses.find((q) => q.id === c.pose);
  const M = new THREE.Matrix4().set(...c.T_final.flat());
  const pos = new THREE.Vector3(...p.threejs.position).applyMatrix4(M);
  c.expected.position.forEach((v, i) => near([pos.x, pos.y, pos.z][i], v, tol.position_m, `${c.id} position[${i}]`));
  const q = new THREE.Quaternion(...p.threejs.quaternion_xyzw);
  const rot = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().extractRotation(M));
  const got = rot.multiply(q);
  const e = new THREE.Quaternion(...c.expected.quaternion_xyzw);
  near(Math.abs(got.dot(e)), 1, 1e-9, `${c.id} quaternion`);
  n++;
}

for (const k of data.intrinsics) {
  const { width: W, height: H, fx, fy, cx, cy } = k;
  const cam = new THREE.PerspectiveCamera(60, W / H, k.near, k.far);
  cam.projectionMatrix.set(...k.projection_matrix_rowmajor.flat());
  cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
  near(2 * Math.atan(H / (2 * fy)) * 180 / Math.PI, k.fov_y_deg, 1e-9, 'fov');
  for (const [X, Y, Z] of [[0.2, -0.1, 2.0], [-0.5, 0.3, 4.0], [0, 0, 1]]) {
    const u = fx * X / Z + cx, v = fy * Y / Z + cy;                    // COLMAP pixel (y down)
    const ndc = new THREE.Vector3(X, -Y, -Z).applyMatrix4(cam.projectionMatrix); // camera coords: y up, looks -z
    near((ndc.x + 1) * W / 2, u, 1e-6, 'u');
    near((1 - ndc.y) * H / 2, v, 1e-6, 'v');
    n++;
  }
}
console.log(`conventions: ${n} checks passed against three.js r${THREE.REVISION}`);
