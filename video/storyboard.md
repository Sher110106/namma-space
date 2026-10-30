# Storyboard

Same storyboard for every run. Run `video/build_video.sh <run>`. Spoken text and screen notes: `narration.md`. Length 3:30 to 4:40, asserted by the build.

| # | Segment | About | Picture | Source |
|---|---|---|---|---|
| 1 | cold | 20 s | First-person walk, key overlay, minimap, wall stop | real viewer recording |
| 2 | title | 18 s | Title card, scope card (in and out of scope) | slides |
| 3 | capture | 35 s | Capture routine (setup, three passes), then capture footage | slides + public-room frames at 16 fps, or `video/inputs/<run>/capture_clip.mp4` at 4x |
| 4 | pipeline | 45 s | `./ns run video.mp4`, architecture with measured times, stage numbers | slides from report.json and timings.json |
| 5a | walk | 13 s | WASD and mouse, wall stop | real viewer recording |
| 5b | map | 7 s | M, click the map, jump | real viewer recording |
| 5c | reality | 13 s | C: held-out photo against render, two pairs | real viewer recording |
| 5d | bench | 16 s | T: tour benchmark (cut to 10 s), result card | real viewer recording |
| 5e | measure | 8 s | K: two points, scale interval; H: stats panel | real viewer recording |
| 5f | route | 11 s | `/` search, G route, Space follow, "Round 2 preview" (title card if the scene has no places) | real viewer recording |
| 6 | results | 49 s | Scale, geometry vs Faro, trainer parity, viewer numbers | slides from eval/results, report.json, verify.json |
| 7 | limits | 16 s | Limits, links with placeholders | slides |

Segment length is the speech length plus 0.6 s, never shorter than its footage. Captions come from the same text and timing as the audio.
