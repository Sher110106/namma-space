# Namma Space

**One phone video. One command. A room you can walk through.**

Namma Space turns an ordinary indoor phone video into a level, real-scale Gaussian-splat scene with a browser walk controller. The Round 1 path uses local open-source tools and keeps scale, camera poses, collision and the minimap in one frame.

![Walkthrough GIF placeholder](docs/abstract/walkthrough.gif)

## Run it

```bash
./ns run path/to/video.mp4
```

The measured run time is `${timings.total}` seconds on `${hardware.name}` with `${hardware.gpu}`. Values are filled in the submitted report; this screen does not invent a result.

## Smoke test

```bash
./ns test all
```

## Viewer

Viewer link: `${viewer.url}`  
Offline copy: `viewer/dist/` (run `python3 -m http.server 8000` there).

## Docs

- `docs/SOP.md` and `docs/SOP_card.pdf`: capture steps and checks
- `LICENSES.md`: generated licence and credit table
- `docs/abstract/template.html`: technical abstract source
- `docs/poc/00_START_HERE_Proof_of_Concept_and_Links.html`: judge path and links
- `video/storyboard.md`: demo plan and narration

## Licence

Repository code is Apache-2.0. Product and model licences are listed in `LICENSES.md`; unresolved items stay marked as unverified.
