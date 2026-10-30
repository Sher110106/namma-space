# honka60: the smoke-test sample

60 consecutive photos of the MuSHRoom `honka` room (iPhone long capture): `frame_00160.jpg` to `frame_00219.jpg`,
738 x 994 px, copied unchanged from `room_datasets/honka/iphone/long_capture/images/`.

Why these: a first choice (every second frame of 13-131) registered only 81.7% (SfM partial, exit 11) and gave a
grid with no walls (free-fly fallback). These 60 consecutive frames registered 100% (variant D, 0.46 px) and gave a
walkable grid (run `rh-smoke2`, 30 Oct; numbers in `test/expected/report.json`, case `smoke`). They cover one part of
the room, not all of it.

Used by the 10-minute smoke test in `test/README.md`:

    ./ns run --frames test/data/honka60/frames --name smoke --quick --kind public --license "CC BY 4.0" \
        --attribution "MuSHRoom dataset (Ren et al. 2024), room honka"

## Credit and licence

MuSHRoom: Multi-Sensor Hybrid Room Dataset for Joint 3D Reconstruction and Novel View Synthesis.
Xuqian Ren, Wenjia Wang, Dingding Cai, Tuuli Tuominen, Juho Kannala, Esa Rahtu. WACV 2024, arXiv:2311.02778.
Project: https://xuqianren.github.io/publications/MuSHRoom/ ; code and data links: https://github.com/TUTvision/MuSHRoom ;
files from the dataset's Zenodo records.

Licence: CC BY 4.0, as recorded in our plan (FINAL_PLAN section 4) from the Zenodo records. The GitHub README asks
users to cite the paper and does not state a licence; check the Zenodo record's licence field before this folder is
published.
