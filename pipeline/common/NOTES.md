# pipeline/common: notes (root, WP0)

Standard library only. Three modules other packages may copy-import.

## colmap_io.py

Reads and writes COLMAP binary models (cameras.bin, images.bin, points3D.bin). rigs.bin and frames.bin are carried
through; a similarity transform also updates frames.bin so the model stays consistent. API list is at the top of the file.

Measured on tyrone, host python 3.14, `runs/smoke_honka/clean/sparse/0` (283 images, 21,429 points, 118,536 observations):
- read in 0.57 s including a full reprojection pass; writing the model back is byte-identical for all five files.
- RMS reprojection error recomputed from the keypoints: 0.766 px. COLMAP's own mean error for the same model: 0.5926 px
  (mean and RMS differ; both are correct).
- After a similarity transform (s = 0.5, a permutation rotation, a shift), `colmap model_analyzer` in `namma-colmap:0.1`
  reports the same registered count (283), points (21,429), observations (118,536) and mean reprojection error (0.592629 px)
  as before. This covers test D4's "same count and error" for the transform itself.
- The honka model has one camera of model PINHOLE, so `project()` covers it. Other models: SIMPLE_PINHOLE, SIMPLE_RADIAL,
  RADIAL and OPENCV are implemented; the rest raise NotImplementedError (not needed by our stages).

Decisions:
- Pixel data (2D points, tracks) stays as packed bytes and decodes on demand. That keeps a 283-image model small and makes
  the transform fast. `Image.points2D` and `Point3D.track` give lists when you need them.
- frames.bin: layout read from the honka file (frame id, rig id, quaternion w,x,y,z, translation, one sensor entry of 16
  bytes). The quaternion order matches images.bin. rigs.bin is left alone when every rig has one sensor; a rig with more
  sensors is copied with a warning in `Model.warnings`.
- Text models are refused with the `colmap model_converter` hint. The gsplat pycolmap fork crashes on them.

## conventions.py and test/data/conventions.json

`threejs_from_colmap(R, t, T_final)` and the inverse, `projection_matrix(...)`, `yaw_pitch_deg`, `quat_from_yaw_pitch_deg`.
The JSON is made by `test/e2e/gen_conventions.py` from the three.js side (yaw, pitch, position), without importing
conventions.py, so the Python module and the file are independent. It holds 4 poses and 2 T_final cases and 2 intrinsics.
Checked three ways: Python (`test_conventions.py`), Node against the vendored three.js r180 (`conventions.test.mjs`),
and a pixel test that sends a world point through the COLMAP route and the three.js route and compares the pixels.

Decisions the other packages must know:
- **T_final means X_final = scale * Rf * X + translation.** `parse_t_final` reads every form the packages write (checked
  against grid/frame.py, viewer/tools/bundle.py and export/heldout.py at 04:20): a 4x4 nested list, a flat list of 16
  (row-major), `{"matrix": ...}`, or `{scale, translation|position, quaternion [x,y,z,w]}`. W3's export.json writes the last
  form plus a flat `matrix`. All three readers agree on the meaning, so nothing needs to change.
- Quaternions are [x, y, z, w]; tests compare up to sign.
- COLMAP identity (camera looking along +Z) becomes the three.js quaternion (1, 0, 0, 0): a half turn about X.
- Projection matrix sets NDC x = 2u/W - 1 and NDC y = 1 - 2v/H, so the COLMAP principal point lands on the right pixel.

## stage.py

`atomic_stage` (write `<dir>.tmp`, rename on success; old result moved aside and deleted only after the new one is in
place; a failure leaves `<dir>.failed`), `write_json_atomic`, `read_json`, `Run`, `disk_guard` (exit 20; `NS_FAKE_FREE_GB`
fakes a low disk), `timed`, `stage_context`, `run_main`, plus the stage table (`STAGES`) that the runner and report read.

Decisions:
- The table sets `sfm` to image `namma-colmap:0.2`. `namma-colmap:0.1` has no python3 (checked 03:50: `python3: not found`),
  and W1 built 0.2 with python3 added. The runner takes `--image sfm=...` to override.
- `align` and `grid` run in `namma-space-gpu:0.1` without a GPU; `export` in `namma-space-gpu:0.2` (the table says 0.2;
  W3 may need to change it to 0.1 if 0.2 gives no benefit).
- Under the runner (`NS_RUNNER=1`) `StageContext.log` prints only, since the runner already writes stdout to
  `logs/<stage>.log`. Otherwise it also appends to that file.

## Facts about the host found on the way

- The Mac has little free disk (1.8 GB at 03:45, 0.4 GB at 04:05, falling). `ns pull` sizes each transfer first and refuses
  when it would not fit; `ns doctor` flags it.
- `/w/runs/smoke_honka/clean` is read-only for us; do not write there.
