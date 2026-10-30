# WP2 ingest and check: notes (W2, 30 Oct 2026)

Two entry points.
- `python3 pipeline/ingest/check_mac.py VIDEO [VIDEO ...] [--out check.json | --run-dir RUN] [--json] [--hwaccel]` (Mac, standard library, ffprobe + ffmpeg). Prints PASS/WARN/FAIL per check with the fix, writes `check.json` `{schema, verdict, warnings[], probe{}, checks[], files[], tool, date}`. Exit 0 for PASS and WARN, 10 for FAIL, 2 for bad usage. `--run-dir` writes `RUN/00_preflight/check.json`.
- `python -m pipeline.ingest --run RUN [--video F ... | --frames DIR] [--take-b | --max-frames N] [--force] [--tonemap auto|zscale|numpy] [--select] [--min-free-gb G]` (image `namma-space-gpu:0.1`). Reads `RUN/input/` (videos, else a `frames/` or `images/` folder, else images in `input/`). Writes `RUN/10_frames/{images/000001.jpg ..., frames.json}` through a temp directory renamed on success. Candidate JPEGs are scratch and deleted before the rename. The last stdout line is `NS_RESULT {json}`.
  - Exit 0 good. Exit 10 (capture problem): video under 45 s ("too short"), under 60 usable frames, or the video cannot be decoded; the message prints `reshoot A s to B s` ranges (for "too few frames" the blurry and duplicate windows) and nothing is written to the run directory. `--force` turns both length checks into warnings. Exit 20: no ffmpeg, or less than 3 GB free (`--min-free-gb`).
  - `frames.json`: `schema, src (video|frames), inputs[], n_cand, n_kept, dropped{blur, dup, thin}, hdr (none|hlg|pq), tonemap (none|zscale|numpy), rotation (cw degrees, or {exif_rotated: n} in frames mode), sample_fps, window, cap, size [w,h], rules{}, kept[{name, t, sharp, src}], dropped_windows[{src, t0, t1, reason}], warnings[], seconds, tool{}`. `t` is seconds from the start of the video (null in frames mode, where `src` is the original file name). Names are consecutive in capture order; with several videos they are joined in file-name order, then thinned together.

## Ingest algorithm as built (FINAL_PLAN 2.2, with two changes)

1. ffprobe: rotation (display matrix or `rotate` tag), `color_transfer` (arib-std-b67 = HLG, smpte2084 = PQ), bit depth, variable frame rate (packet timestamps: more than 5% of gaps off the median gap by over 25%).
2. One ffmpeg pass: autorotate (default), `fps=8`, `showinfo` (gives each candidate's timestamp), scale the long side to at most 1920 (lanczos, never up), tone map if HDR, convert to full-range BT.601 YUV (see below), JPEG q2. **Change 1:** fps and scale come before the tone map (the plan has the tone map first), so the float tone map runs on 8 fps at the final size, not on every 4K frame.
3. Laplacian variance on a 640 px (long side) grey copy; sharpest candidate in each window of 4 (0.5 s).
4. Drop windows under 0.35 x the rolling median (window 31, centred, shrinking at the ends) of the window winners.
5. Drop near-duplicates: mean absolute difference of 64x36 (36x64 portrait) grey thumbnails under 1.5 against the previous kept frame.
6. Thin uniformly to the cap (450; 600 with `--take-b`), joined over all takes.
7. Rename winners `000001.jpg ...`. The winners are the candidate JPEGs themselves (one encode, no recompression).
- **Colour.** JPEG carries no matrix tag and viewers, OpenCV and gsplat's loader assume BT.601. The plan's chain ends in BT.709 YUV; encoding that straight to JPEG shifts hues. The pass ends with `scale=in_color_matrix=auto:out_color_matrix=smpte170m:out_range=pc,format=yuvj420p`. Test B1b: mean RGB of a kept frame is within 4 levels of the video's own decode at the same instant.
- **`--frames` mode.** Sorted by natural file name, `ImageOps.exif_transpose`, EXIF dropped (no `exif=` on save), long side capped at 1920, JPEG q95, all kept up to the cap (no blur or duplicate filter unless `--select`). Mixed sizes are warned. Tested with EXIF orientation 6 (`TestFramesMode`).
- **HDR.** ffmpeg 4.4.2 in `namma-space-gpu:0.1` has `zscale` and `tonemap` (checked with `ffmpeg -filters`), so the numpy fallback is not needed there. It exists anyway (`tonemap_np.py`, `--tonemap numpy`, or automatic when `zscale`/`tonemap` are missing) and was run on b2.
  **Change 2: `tonemap=reinhard`, not `hable`.** b2 is b1 re-encoded as BT.2100 HLG (numpy, diffuse white at signal 0.75), so a good tone map brings b2 back near b1. `python -m pipeline.ingest.tonemap_sweep` on tyrone, mean luma / RMS contrast over 8 frames, b1 reference 139.2 / 58.5:

  | chain (`zscale=...npl,...`) | mean luma | RMS vs b1 | MAE to b1 |
  |---|---|---|---|
  | plan: npl=100, hable, desat=0 | 105.3 | 0.80 | 33.9 |
  | npl=100, reinhard, desat=0 (used) | 137.3 | 0.93 | 7.4 |
  | npl=100, mobius, desat=0 | 146.7 | 1.02 | 11.7 |
  | npl=100, clip | 158.8 | 1.19 | 20.7 |
  | npl=203, hable | 82.0 | 0.65 | 57.2 |
  | npl=1000, hable | 43.6 | 0.36 | 95.6 |

  The plan's hable chain sits exactly on the B2 contrast gate (0.80) and darkens by 25%; reinhard passes with margin. Caveat: b2 has no highlights above diffuse white (its source is SDR), so highlight roll-off is untested; real iPhone HLG with a window in view needs a look. The chain passes the stream's own tags as input (`tin pin min rin`) so untagged primaries do not stop zscale. PQ is encoded in the same way by `make_fixtures.py b2pq` (built, not run through ingest).
- **Timestamps.** `showinfo` `pts_time` after `fps=8`; the test checks each kept frame against the video's own 8 fps stream. `-ss` seeking is not used as a reference: it lands one frame (1/8 s) off the `fps` filter's pick.

## Measured (tyrone container, ffmpeg 4.4.2, OpenCV 4.10, numpy 1.26; Mac numbers in brackets where run)

| Test | Result |
|---|---|
| B1 b1 (60 s 1080p H.264) | 109 frames (100-130), all 1920x1080, times strictly increasing, each frame matches the video's 8 fps stream at round(t*8) (median MAD 1-2 grey levels). 10 windows dropped as blurry, 1 as duplicate; 6 s on the Mac, 8 s in the container |
| B2 b2 (10-bit HEVC HLG) | `hdr: "hlg"`, `tonemap: "zscale"`, mean luma and RMS contrast pass (60-190; >= 0.8 x b1). Numpy fallback also passes (mean 128.6, RMS 59.4 on the Mac) |
| B3 b3 (stored 1920x1080 + rotate 90) | frames 1080x1920; upright (correlation 1.0 with the upright source crops, 0.45 for the 180 deg turn). 100 frames (10 dup: the crop is static) |
| B4 b4 (20 s blurred in ten 2 s segments) | blurred windows with no frame **100%** (>= 75%); sharp windows kept **96%** (>= 90%) |
| B5 b5 (15 s) | exit 10, "too short: b5.mp4 is 15 s, need at least 45 s", `reshoot 0.0 s to 15.0 s`, no `10_frames/` written |
| B6 b6 (VFR 30/20/30 fps, mean 26.7) | runs; `vfr: true`; 108 frames, times in order |
| B7 b7 (3 min 3840x2160 HEVC) | **56 s and 59 s** wall in two container runs (limit 360 s); 305 frames kept of 1440 candidates; 1920x1080 |
| B8 `check_mac` on b1-b7 | b1 WARN, b2 WARN, b3 WARN, b4 WARN, b5 FAIL, b6 WARN, b7 PASS, as built (below). 4, 7-17, 8-12, 5, 1, 5, 22-25 s on the Mac (limit 90 s) |
| frames mode | EXIF orientation 6 applied, EXIF gone, names consecutive |

B8 verdicts follow from the thresholds and how each fixture was made: b1, b3, b6 are 60 s (the 45-90 s WARN band) and nothing else; b2 adds HLG; b4 adds a blur share of 67% (a third of its windows are blurred on purpose; the check measures 67%, in the 60-80% WARN band); b5 is 15 s; b6 adds VFR; b7 is the only PASS.

## Fixtures (tyrone `/home/sher/namma-space/data/fixtures/videos/`, built by `make_fixtures.py`; copies on the Mac in `/tmp/namma-w2/videos`, not in the repo)

b1..b7 as in FINAL_PLAN WP2, plus a `.json` sidecar each. Source: MuSHRoom honka long capture (CC BY 4.0), 335 portrait 738x994 frames held at 5.6 fps and cropped 738x415 with a slow vertical pan (period 10 s; 20 s for b7), scaled to 1920x1080 (3840x2160 for b7). b3 is the portrait crop, not panned.
- **The base clips use the sharp 258 of 335 frames** (Laplacian variance >= 0.5 x median). Reason: 12-18% of the honka frames fall under 0.35 x the median Laplacian variance (some are motion-blurred, some face a plain wall; min/median down to 0.015), whatever the crop. With all 335 frames b1 loses 18 of 120 windows (15%) to the blur rule on its own (102 frames kept), so B4's "sharp segments keep >= 90%" cannot be measured. With the sharp subset b1 loses 10 (109 kept).
- The Laplacian variance of the 640 px grey copy moves by about 30x across a pan from wall to carpet (300 to 11,000 in b1). The 0.35 x rolling-median rule therefore drops about 8-15% of good windows on this footage. It costs frames, not correctness, and a real capture with a long plain wall will lose those windows. Not changed (it is the plan's rule); a content-normalised sharpness (ratio of fine-scale to coarse-scale energy) would cut the false drops. Flagging it for root.
- b2 is encoded in numpy (BT.709 SDR to BT.2100 HLG, scene light 0.265 for SDR white = signal 0.75, BT.2020 primaries, 10-bit HEVC); 3.5 min of CPU.
- ffmpeg 4.4's mp4 muxer ignores `-metadata rotate=90`; `make_fixtures.patch_rotation` writes the display matrix into the `tkhd` box (a=0 b=1 c=-1 d=0), which ffprobe 9 reads as rotation -90. The tag-only legacy form is read by `check_mac.probe_video` too.

## check_mac details and decisions

- Sharpness and luma come from ffmpeg in one decode: `fps=8,scale=640:-2,format=gray`, split; one branch through `convolution` with the Laplacian (the filter clips at 0, so `signalstats` YAVG is half the mean absolute Laplacian), the other through `signalstats` for luma. **Not `blurdetect`** (the plan's choice): it exists in the Mac's ffmpeg 9 but not in the container's 4.4.2, and the Laplacian route works on both and matches the ingest's own measure.
- Blur: share of 0.5 s windows (4 samples) whose best frame is >= 0.5 x the median of the per-window bests. Exposure: median of |luma change| between consecutive 0.5 s window means, as a percent of the median luma (content changes count too: a pan from a bright wall to a dark floor reads as exposure change; b1's 7.1% is that).
- On HDR files the blur and exposure numbers are on the raw HLG/PQ signal (no tone map on the Mac).
- Every non-PASS check carries a `fix` string.
- **The Mac has under 1 GB free** (`df`: 744 MB at 03:50, 117 MB at one point while a 4K ingest wrote candidates). I did not run the 4K ingest there. Tests with big outputs run on tyrone; Mac copies are limited to the videos, a 340 MB venv in `/tmp/namma-w2` and small figures. Root and others should know: a Mac stage that writes a few hundred MB can fail with ENOSPC.
- ffprobe packet timestamps are used for VFR: a 3-min file takes about 1 s.

## Other

- Tyrone queue: three `ns-*` containers at once means each of my jobs waited 3-7 minutes for a slot. I run suites as one container per job.
- Mac venv for development (numpy 2.5.4, OpenCV 5.0): the code also runs there, results agree with the container's (same 102 and 109 frame counts).
- Tests: `test/unit/ingest/test_ingest.py` (B1-B7 + frames mode, needs numpy/OpenCV/ffmpeg: run in the container), `test/unit/ingest/test_check_mac.py` (B8 on the Mac, standard library). Fixtures are found through `NS_FIXTURES`, `/w/data/fixtures/videos` or `/tmp/namma-w2/videos`; a missing fixture skips its test.

## Commands run

```
nsdocker w2-fixtures namma-space-gpu:0.1 python -m pipeline.ingest.make_fixtures --out /w/data/fixtures/videos   # b2 3.5 min, b7 5 min, others < 30 s
nsdocker w2-test-ingest namma-space-gpu:0.1 python -m unittest discover -s test/unit/ingest -p test_ingest.py -v
nsdocker w2-sweep namma-space-gpu:0.1 python -m pipeline.ingest.tonemap_sweep /w/data/fixtures/videos
python3 -m unittest discover -s test/unit/ingest -p test_check_mac.py -v          # Mac
```
