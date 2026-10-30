# Capture SOP v1: one phone, one or two videos

This is the capture procedure for Namma Space. The one-page card is `SOP_card.pdf`. This page gives the reasons.

Scope: one indoor space, or two spaces joined by a door (take B). The method has not been tested on multiple floors, and the SOP says nothing about them.

Sensors: the phone camera only, by choice. No LiDAR, tripod, markers or app.

## 1. Choose the space (2 min)

- Pick the largest space you can film well, with things to look at: shelves, posters, a rug, plants.
- Avoid glass walls, big mirrors, wall-sized windows and bare meeting rooms. A bare room can work, but the camera path is more likely to break.
- Prefer a space with a door into a second space you can film.
- The model will be public. Clear personal papers, photos and screens.

## 2. Prepare (5 min)

Room:
- Nobody in the shot. Pets out.
- Ceiling fans off and stopped. Curtains still.
- Door fully open and wedged.
- All lights on. No direct sun patches.
- Mirrors covered. Wall screens off.

iPhone (setting names vary by model and iOS):
- Settings > Camera > Record Video: 4K at 30 fps.
- HDR Video off.
- Lock Camera on.
- Enhanced Stabilization off, Action mode off, Auto FPS off, if the phone has them.
- If the lights flicker on screen, use 25 fps (turn on Show PAL Formats).
- In the Camera app: Video mode, 1x, landscape. Press and hold on a mid-bright wall until "AE/AF LOCK" shows.

Android (setting names vary by maker):
- 4K at 30 fps, main 1x lens.
- HDR10+ off. Super Steady and other stabilisation off. Auto FPS off.
- Lock focus and exposure (Pro video if the phone has it).

Before the real capture:
- Clean the lens.
- Free at least 5 GB of storage. Charge above 50%.
- Film a 30 s test clip and run `./ns check` on it. Fix what it flags.

## 3. Take A: the room (3-4 min, one continuous video)

Hold the phone at chest height, landscape, with both hands. Turn your body, not your wrist. Turn no faster than one full circle in 12 s.

1. Stand in the doorway facing in. Hold still for 3 s.
2. Outer loop: walk along the walls about 0.5 m from them, with the camera pointing across the room. Walk about one step per second.
3. Inner loop: walk about 1.5 m from the walls, with the camera facing the nearest wall and tilted slightly down so the floor line shows. At each corner, tilt up to the ceiling corner for 2 s.
4. Low pass: phone at waist height, tilted about 30 degrees down, circling the table and the large furniture.
5. Finish where you started.

If the space is too tight for the inner loop (under 1.5 m between wall and furniture), film the walls from where you can stand, 1 to 1.5 m away, and keep moving.

## 4. Take B: the room and the next space (4-5 min, a separate video)

1. Repeat take A's outer loop.
2. Walk slowly out through the door. Keep both sides in view for about 3 s.
3. Do the outer and inner loops in the next space (a corridor or a room).
4. Come back through the same door, slowly, and end inside the room.

## 5. Never

Spin on the spot. Zoom. Switch lens. Film a bare wall from under 1 m. Walk fast. Let anyone into the shot.

## 6. After (3 min)

- A photo of the hand sketch: outline, door, windows, big furniture, the start point.
- Your height in cm.
- Optional, with a tape or the phone's Measure app: the room's length and width wall to wall at floor level, and the door's width. Write down how you measured.

## 7. Why each rule

| Rule | Failure it prevents |
|---|---|
| Move, do not spin | No parallax, so the camera path fails or splits |
| Lock exposure, focus and white balance; HDR off | Colour flicker turns into blotches and floaters; HDR frames decode washed out |
| 1x lens, Lock Camera | A lens switch mid-clip breaks the single-camera model |
| Inner loop facing walls; corner tilt-ups | Walls stay solid when a visitor walks close |
| Low pass | The floor and the space under furniture get covered |
| Cross the door slowly, both ways | Loop detection finds the image pairs that link two spaces |
| Fans off, door wedged, nobody inside | Moving things leave ghosts |
| 25 fps under flickering lights | Banding that shifts from frame to frame (50 Hz mains) |

## 8. The capture check (`./ns check`, Mac, about 1 min per video)

The check runs `ffprobe` and `ffmpeg` on the video and prints PASS, WARN or FAIL with the fix.

| Check | PASS | WARN | FAIL |
|---|---|---|---|
| Duration | 90-420 s | 45-90 s or over 420 s | under 45 s |
| Resolution (short side) | 1080 px or more | 720-1079 px | under 720 px |
| Frame rate | 24-60 | variable rate flagged | under 24 |
| HDR transfer (`arib-std-b67` or `smpte2084`) | absent | present (we tone-map it) | none |
| Blur: share of 0.5 s windows whose sharpest frame is at least 0.5 times the clip median | 80% or more | 60-80% | under 60% |
| Exposure: median luma change between samples | under 8% | 8-15% | over 15% |

A FAIL means reshoot. The check prints the time ranges to redo.

## 9. How the SOP covers the problem statement's four items

| Item on the organizers' slide | Our answer |
|---|---|
| Camera angles | Three passes: outer loop (camera across the room), inner loop (camera at the walls), low pass (camera down at the floor and furniture) |
| Corner shots | Tilt-ups to each ceiling corner, 2 s each |
| Floor plan | A hand sketch goes in. The pipeline computes a plan from the model (the walkable grid and a minimap), levelled and scaled |
| Sensors | Phone camera only, by choice |

## 10. Larger sites (design only, not tested)

For a site with several rooms, the SOP extends by repetition: one take per area, each starting and ending in a doorway it shares with its neighbour, and each crossing the door slowly in both directions. The pipeline then tries to join the takes in one model. Take B tests this on one door. We have no measurement on larger sites yet.
