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
python eval/score_mushroom.py \
  --dataset /w/data/mushroom/room_datasets/honka \
  --out /w/runs/w7-honka/mushroom_honka_lidar.json
```

The script reads `iphone/long_capture/transformations.json`, converts its
OpenGL camera convention to OpenCV with `diag(1,-1,-1)`, converts LiDAR PNG
millimetres to metres, fuses all views with Open3D
`ScalableTSDFVolume(voxel_length=.02, sdf_trunc=.08, depth_trunc=5.0)`, then
applies `icp_iphone.json` to score against `gt_mesh.ply`.  Accuracy is ours to
GT, completeness is GT to ours, and F-score uses a 5 cm threshold.

The measured honka baseline is in `results/mushroom_honka_lidar.json`:
334 views, accuracy 0.304277 m, completeness 0.025468 m, and F-score
0.572772.  The pinned Open3D 0.20 CPU wheel returned an empty ScalableTSDF
mesh even for a plane; the run therefore used the explicitly recorded
point-surface compatibility fallback after attempting ScalableTSDFVolume.

## J3 phone-only / J4 phone-only VR room

After `rh-honka` or `rh-vrroom` has a final PLY and held-out poses, use the same
`pipeline.geometry.tsdf.render_ply_depth` and `fuse_depth_views` APIs.  The
held-out depth comparison uses the same camera IDs as the LiDAR frames; report
median absolute depth error and shares within 5 cm and 10 cm.  Fit a robust
Umeyama similarity from phone-only camera centres to dataset metric centres;
the scale error is `abs(s_est/s_fit - 1) * 100`.

Numbers from the smoke model are labelled `smoke model` and are not metric
claims.  If a rehearsal run is absent or a GPU slot is unavailable, its JSON
must contain `status: "not_run"` and the reason instead of an invented value.

## J5 synthetic TSDF check

The unit fixture in `test/unit/geometry/test_tsdf.py` validates projection and
free-space semantics without Open3D.  The full synthetic-room acceptance run
is performed in the GPU image and records the median wall distance; the gate is
`<= 0.02 m`.  The measured four-view box result is
`results/synthetic_room.json` (median 0.013053 m, pass).
