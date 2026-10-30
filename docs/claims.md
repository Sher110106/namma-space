# Claims ledger

Use a claim only when its evidence file exists and its value is measured. If it does not, use the fallback wording.

| Claim | Evidence | Fallback wording |
|---|---|---|
| Phone video only, with measured fidelity | `report.json`; `50_train/`; `60_export/export.json`; `eval/results/mushroom_*.json`; `eval/results/mip360_room.json`; viewer reality check | “Phone video input; we report held-out image scores and public-room checks.” |
| Close to LiDAR | `eval/results/mushroom_*.json`, with phone-only F-score at the required threshold beside the LiDAR baseline | “Compared with the public-room reference; no LiDAR closeness claim.” |
| Behaves like a place: level frame, scale interval, collision and same-geometry minimap | `40_align/align.json`; `70_grid/grid.json`; `90_verify/verify.json`; `run.json` | “The bundle carries an estimated scale and a grid; manual steps are listed.” |
| One-page SOP, one command and repeatable run | `docs/SOP_card.pdf`; `docs/SOP.md`; `00_preflight/check.json`; `timings.json`; rehearsal log; `test/` | “A documented local pipeline with a logged capture check.” |
| Measured viewer budget | `90_verify/verify.json`, with device, browser, GPU, initial-frame time, frame rate and bytes | “Measured on the listed test device only.” |
| Licence-clean product path | `LICENSES.md`; `docs/licenses.yaml`; `docs/tools/check_licenses.py`; vendor notices | “Licence sources and unresolved items are listed; unverified packages are not presented as checked.” |
| Two connected spaces | one scene from take B with at least `${claims.two_space_threshold}` registered share and a door-crossing test in `report.json`/`90_verify/verify.json` | “One room; the connected-space attempt is recorded in the failure table.” |
| Our SfM linked K rooms of a public apartment | `eval/results/public_apartment_sfm.json` and its log | “Public-room SfM was evaluated without making a room-count claim.” |
| Time to twin: N minutes | `timings.json` and `report.json` | “The run log reports stage times; no single total is claimed until present.” |
| Photoreal | held-out photo/render pairs and PSNR/SSIM/LPIPS in `report.json` | “Visual match shown beside measured image scores.” |
| Smooth on an ordinary laptop | `90_verify/verify.json` on a named laptop; do not generalise from M1 | “Smooth on the measured device; other hardware is unverified.” |
