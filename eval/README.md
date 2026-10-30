# WP7 evaluation

All numbers in `results/*.json` are generated on tyrone with the pinned
`namma-space-gpu:0.2` image.  Each result records its exact command, Git SHA,
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
  '/home/sher/namma-space/bin/nsdocker ns-w7-honka-lidar namma-space-gpu:0.2 \
   python3 -m eval.score_mushroom --dataset /w/data/mushroom/room_datasets/honka \
   --out /w/runs/w7-honka/mushroom_honka_lidar.json'
```

The script reads `iphone/long_capture/transformations.json`, converts its
OpenGL camera convention to OpenCV with `diag(1,-1,-1)`, converts LiDAR PNG
millimetres to metres, and uses bounded Open3D
`UniformTSDFVolume(voxel_size=.02, sdf_trunc=.08, depth_trunc=5.0)`. The volume
is a padded cube around the valid world-space depth envelope and is capped at
512³ voxels; the chosen origin and effective resolution are recorded in the
run metadata. `ScalableTSDFVolume` and the deterministic point-surface path
remain recorded fallbacks. It applies `icp_iphone.json` to score against
`gt_mesh.ply`; accuracy is ours to GT, completeness is GT to ours, and F-score
uses a 5 cm threshold.

The JSON contains both raw and standard cropped scores, with 100,000 uniformly
sampled surface points per method. The headline protocol is `score.cropped`:
predicted points are kept inside the Faro mesh bounding box grown by 5 cm;
ground-truth points remain unchanged for completeness. Uniform honka raw is
accuracy 0.211129 m, completeness 0.031016 m, F@5 cm 0.753561; cropped is
0.118976 m, 0.031017 m, 0.802593. Uniform vr_room raw is 0.387771 m,
0.038899 m, 0.657611; cropped is 0.170712 m, 0.038899 m, 0.736644. The old
point-surface scores remain under `fallback_point_surface` in each JSON. Use
the cropped protocol for every method.

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
`nsdocker --gpu ns-w7-j3-honka namma-space-gpu:0.2` and
`nsdocker --gpu ns-w7-j4-vr namma-space-gpu:0.2`, respectively; the exact
host paths and output names above are the recorded commands.

The scorer renders every matched training camera and held-out frames every 8th
in gsplat's sorted-name convention.  It reports fit RMSE, `s_ours`, `s_true`,
`abs(s_ours/s_true - 1) * 100`, the align 68% interval and containment, depth
error against LiDAR, and raw/cropped Faro geometry.  `score.cropped` is the
headline geometry protocol shared with J2.

Depth PNGs are resized nearest-neighbour from 738×994 to the undistorted
COLMAP raster (735×990 for honka and 735×990 for vr_room) before pixelwise
comparison; no depth interpolation or invalid-pixel fill is used.

The tyrone host had no usable NVIDIA driver for these two renderer runs. The
CPU retry failed inside gsplat's CUDA rasterizer, so both JSON files are
`status: "not_run"`; their prior point-surface results are retained under
`fallback_point_surface` and are not new Uniform headline numbers.

* `mushroom_honka_lidar.json`: Uniform cropped accuracy 0.118976 m,
  completeness 0.031017 m, F@5 cm 0.802593.
* `mushroom_vr_room_lidar.json`: Uniform cropped accuracy 0.170712 m,
  completeness 0.038899 m, F@5 cm 0.736644.
Numbers from the smoke model are labelled `smoke model` and are not metric
claims.  If a rehearsal run is absent or a GPU slot is unavailable, its JSON
must contain `status: "not_run"` and the reason instead of an invented value.

## J5 synthetic TSDF check

The unit fixture in `test/unit/geometry/test_tsdf.py` validates projection and
free-space semantics without Open3D.  The full synthetic-room acceptance run
is performed in the GPU image and records the median wall distance; the gate is
`<= 0.02 m`.  The measured four-view box result is
`results/synthetic_room.json` (Uniform, 8,558 vertices, median 0.003638 m, pass).
