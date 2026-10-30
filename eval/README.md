# WP7 evaluation

All numbers in `results/*.json` are generated on tyrone with the pinned
`namma-space-gpu:0.1` image.  Each result records its exact command, Git SHA,
date, input digests, pose convention, and metric parameters.  PNGs next to the
JSON files are the corresponding compact figure.

## J1 trainer parity (Mip-NeRF 360 room)

The parity run uses gsplat's shipped trainer and dataset poses:

```sh
nsdocker --gpu w7-mip360 namma-space-gpu:0.1 \
  python /opt/gsplat/examples/simple_trainer.py mcmc \
  --data-dir /w/data/mipnerf360/room --data-factor 2 --max-steps 30000 \
  --eval-steps 30000 --save-steps 30000 --ply-steps 30000 --save-ply \
  --disable-viewer --disable-video --result-dir /w/runs/w7-mip360
```

The held-out split is gsplat's default `test_every=8`, with image names sorted
by the parser.  The published comparison point is 31.36 dB.  SSIM and LPIPS
are copied from the trainer's final evaluation JSON, never estimated from a
different renderer.

The completed run measured **32.482353 dB PSNR, 0.9378889 SSIM, and 0.1323258
AlexNet LPIPS** on 39 held-out images (311 total).

## J2/J4 LiDAR TSDF baseline

For either MuSHRoom room, run inside the image (change `honka` to `vr_room`):

```sh
python -m eval.score_mushroom \
  --dataset /w/data/mushroom/room_datasets/honka \
  --out /w/runs/w7-honka/mushroom_honka_lidar.json
# Host equivalent (the image starts in /w/repo):
ssh -o BatchMode=yes sher@10.1.18.190 \
  '/home/sher/namma-space/bin/nsdocker ns-w7-honka-lidar namma-space-gpu:0.1 \
   python3 -m eval.score_mushroom --dataset /w/data/mushroom/room_datasets/honka \
   --out /w/runs/w7-honka/mushroom_honka_lidar.json'
```

The script reads `iphone/long_capture/transformations.json`, converts its
OpenGL camera convention to OpenCV with `diag(1,-1,-1)`, converts LiDAR PNG
millimetres to metres, and attempts Open3D
`ScalableTSDFVolume(voxel_length=.02, sdf_trunc=.08, depth_trunc=5.0)`.  The
pinned 0.20 wheel returned zero vertices on a valid synthetic plane despite
float32 metres, uint16 millimetres, contiguous uint8 H×W×3 colour, matching
intrinsics, `convert_rgb_to_intensity=False`, inverse c2w extrinsics, and
64×64, 480×640, and 738×994 tests.  `UniformTSDFVolume` returned 1,156
vertices on the same plane.  J2 therefore uses the recorded deterministic
point-surface fallback.  It applies `icp_iphone.json` to score against
`gt_mesh.ply`; accuracy is ours to GT, completeness is GT to ours, and
F-score uses a 5 cm threshold.

The JSON contains both raw and standard cropped scores.  The headline protocol
is `score.cropped`: predicted points are kept inside the Faro mesh bounding box
grown by 5 cm; ground-truth points remain unchanged for completeness.  Honka
raw is accuracy 0.304277 m, completeness 0.025468 m, F@5 cm 0.572772; cropped
is 0.236204 m, 0.025468 m, 0.607190.  The crop removes 15,974 of 196,929
predicted points, supporting the area-coverage explanation for the raw 0.30 m.
Use the cropped protocol for every method.

## J3 phone-only / J4 phone-only VR room

After `rh-honka` or `rh-vrroom` has a final PLY and held-out poses, run:

```sh
python -m eval.score_phone_only \
  --run /w/runs/rh-honka --dataset /w/data/mushroom/room_datasets/honka \
  --out /w/runs/w7-honka/mushroom_honka_ours.json
python -m eval.score_phone_only \
  --run /w/runs/rh-vrroom --dataset /w/data/mushroom/room_datasets/vr_room \
  --out /w/runs/w7-vrroom/mushroom_vr_room_ours.json
python -m eval.score_mushroom \
  --dataset /w/data/mushroom/room_datasets/vr_room \
  --out /w/runs/w7-vrroom/mushroom_vr_room_lidar.json
```

The phone-only commands are run through
`nsdocker --gpu ns-w7-j3-honka namma-space-gpu:0.1` and
`nsdocker --gpu ns-w7-j4-vr namma-space-gpu:0.1`, respectively; the exact
host paths and output names above are the recorded commands.

The scorer renders every matched training camera and held-out frames every 8th
in gsplat's sorted-name convention.  It reports fit RMSE, `s_ours`, `s_true`,
`abs(s_ours/s_true - 1) * 100`, the align 68% interval and containment, depth
error against LiDAR, and raw/cropped Faro geometry.  `score.cropped` is the
headline geometry protocol shared with J2.

Depth PNGs are resized nearest-neighbour from 738×994 to the undistorted
COLMAP raster (735×990 for honka and 735×990 for vr_room) before pixelwise
comparison; no depth interpolation or invalid-pixel fill is used.

Completed results (all values are in the JSON files, with the PNG figures next
to them):

* `mushroom_honka_ours.json`: 40 held-out views, median depth error 4.912543
  cm, 50.5625% within 5 cm and 72.2814% within 10 cm; fit RMSE 0.051374 m
  over all matches (0.010248 m over 221 inliers); automatic scale error
  3.263776%, with `s_true = 0.5091266261` inside the 68% interval
  [0.4662925476, 0.5927740049].  Cropped geometry is accuracy 0.188811 m,
  completeness 0.034617 m, F@5 cm 0.514812.
* `mushroom_vr_room_ours.json`: 53 held-out views, median depth error
  8.485043 cm, 34.3051% within 5 cm and 55.4688% within 10 cm; fit RMSE
  0.014079 m over all matches (0.009684 m over 294 inliers); automatic scale
  error 3.098344%, with `s_true = 0.4401097830` inside
  [0.3782482105, 0.4808477162].  Cropped geometry is accuracy 0.280199 m,
  completeness 0.049527 m, F@5 cm 0.293864.
* `mushroom_vr_room_lidar.json`: LiDAR baseline cropped accuracy 0.257726 m,
  completeness 0.023628 m, F@5 cm 0.530743 (raw accuracy 0.450091 m,
  F@5 cm 0.471344).

Numbers from the smoke model are labelled `smoke model` and are not metric
claims.  If a rehearsal run is absent or a GPU slot is unavailable, its JSON
must contain `status: "not_run"` and the reason instead of an invented value.

## J5 synthetic TSDF check

The unit fixture in `test/unit/geometry/test_tsdf.py` validates projection and
free-space semantics without Open3D.  The full synthetic-room acceptance run
is performed in the GPU image and records the median wall distance; the gate is
`<= 0.02 m`.  The measured four-view box result is
`results/synthetic_room.json` (median 0.013053 m, pass).
