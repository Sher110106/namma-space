# Namma Space demo narration

This file is the single source of the spoken text. `video/nsvideo.py` fills the `{braces}` from the run's JSON files, speaks each
block with `say`, times captions from the audio, and builds the slides. One block per segment. `Screen:` notes what is shown.
Never type a number here: add a placeholder and a formatter in `nsvideo.py`.

## cold
Screen: real viewer recording. First-person walk, key overlay, minimap, a wall stop.
Say: Built from one phone video. Running in a browser. Nothing to install. This is a Gaussian-splat copy of a room. I walk with the keyboard, and the wall stops me. This dry run uses a public room, MuSHRoom honka. The final cut uses our own room.

## title
Screen: title card, then scope card.
Say: Namma Space. One phone video. One command. A room you can walk through. Round 1 covers one indoor room. We did not build multiple floors, WebXR AR, or plan registration. This is a dry run on a public room: MuSHRoom honka, under CC BY 4.0.

## capture
Screen: capture slides, then the capture footage ({capture_label}).
Say: Capture follows a short routine. One lens, in landscape, with focus and exposure locked. Film at 25 frames per second if the lights flicker. Then three passes. Pass one: an outer loop along the walls. Pass two: an inner loop facing the walls, tilting up at each corner. Pass three: a low pass around the furniture. Cross a door slowly to add a second space. {check_sentence} {capture_sentence}

## pipeline
Screen: the one command, the architecture figure with measured minutes, the stage numbers.
Say: One command runs it all, on a shared RTX A6000 GPU. This room took {t_total}. Frame selection kept {n_kept} of {n_input} frames. Three camera-path variants race. Variant {sfm_variant} won, with {sfm_reg} of frames registered, in {t_sfm}. Align levels the floor and sets scale from camera height, with a 68 percent interval of plus or minus {scale_pct}. Training took {t_train}, and scores {psnr} dB PSNR on held-out photos. Export writes {splats} splats in {export_mb}. The splat is for looks. A light grid, {walk_m2} square metres, is for walking.

## walk
Screen: real viewer recording. W, A, S, D with real key presses, mouse turns, a wall stop.
Say: W, A, S and D walk. The mouse turns. A wall stops you, and you slide along it. In the viewer test, {walks} random walks ended with {penetrations} positions inside a wall.

## map
Screen: real viewer recording. M opens the map. A click jumps to a walkable spot.
Say: M opens the map. Click any walkable spot, and you jump there.

## reality
Screen: real viewer recording. C: held-out photo against the render, three pairs.
Say: C is the reality check. This photo was held out. The model never saw it in training. Drag the line to compare it with the render from the same camera pose. The card shows the PSNR for each pair.

## bench
Screen: real viewer recording. T starts the tour benchmark, cut to {bench_secs} seconds here, then the result card.
Say: T runs the tour benchmark. For this video I cut it to {bench_secs} seconds. It ends with a card: frames per second, p95 frame time, splat count, download size and GPU. It reads lower here, because the Mac was also recording the screen.

## measure
Screen: real viewer recording. K measures two points. H shows the stats panel.
Say: K measures between two points, and shows the scale interval beside the distance. H opens the stats panel.

## route_places
Screen: real viewer recording, labelled Round 2 preview. Search, route, follow.
Say: Round 2 preview. Press slash and search by name. G draws the walking route round the furniture, and Space follows it. The places here were placed by hand.

## route_none
Screen: title card.
Say: Search, jump, route and follow are a Round 2 preview. They will appear in the final cut.

## results
Screen: three result slides, all values read from JSON.
Say: Results, from two public rooms. The automatic scale error is {honka_scale_err} on honka and {vr_scale_err} on vr room, and the true value sits inside the 68 percent interval in both. Depth error against LiDAR, median: {honka_depth} and {vr_depth}. Against a Faro laser scan, F-score at 5 centimetres: phone only {honka_ours_f} and {vr_ours_f}. iPhone LiDAR fusion {honka_lidar_f} and {vr_lidar_f}. We did not match LiDAR. Before scoring, our camera path was similarity-fitted to the dataset's camera path. Our trainer scores {mip_psnr} dB on the Mip-NeRF 360 room, against {mip_pub} published. The viewer ran at {fps} frames per second on an Apple M1 in headless Chrome. Other hardware is untested.

## limits
Screen: limits slide, then links slide with placeholders.
Say: The limits. Not built: multiple floors, WebXR AR, plan registration. We estimate scale from camera height, and did not check it with a tape. Other hardware is untested. The link and team ID slots stay empty until approval to publish. Thank you.
