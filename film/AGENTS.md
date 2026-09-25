# Production rules for every agent

You are one member of a 14-agent team making a 4-minute 2D animated short
《一盏灯，一颗星》 (*A Lamp and a Star*) entirely from code. The director (the
orchestrating agent) integrates everything. Project root: `/home/user/chat-clear/film`.

## Read first
- `STYLE.md` — the style bible (palette, character designs, rules). Mandatory.
- `SCREENPLAY.md` — story, shot list, cue names.
- `build/timeline.json` — the real timing (scenes, lines, cues). Generated; do not edit.
  If you need a timing change, say so in your final report.
- `engine/core.py` — helpers (easing, noise, colours, paints, paths, Camera, text).
- `engine/timeline.py` — the `Frame` object `f` passed to scenes (`f.t`, `f.cue()`,
  `f.since()`, `f.line()`, `f.mouth(speaker)`, `f.grade`).

## File ownership (strict — only edit files you own)
| Owner | Files |
|---|---|
| ENV-SKY | `lib/sky.py` |
| ENV-SEA | `lib/sea.py` |
| ROBOT | `lib/robot.py` |
| STAR | `lib/star.py` |
| FX | `lib/fx.py` |
| MUSIC | `audio/music.py`, `build/audio/music*.wav`, `build/music/*` |
| SFX | `audio/sfx.py`, `audio/mix.py`, `build/audio/sfx*.wav`, `build/audio/ambience*.wav`, `build/audio/final_mix.wav` |
| SCENE sXX | `scenes/sXX_*.py` (+ optional private helpers `scenes/_sXX_*.py`) |
| Director | everything else (`engine/*`, `tools/*`, docs, timeline) |

Other people are editing the other files *at the same time*. If an import of a library
fails momentarily, wait a few seconds and retry. Never "fix" someone else's file — if a
library has a bug or is missing something you need, work around it locally in your own
file and mention it in your final report.

## Library contract
Library modules expose the functions/dataclasses already stubbed in them. Library owners
must keep every existing signature working (add keyword args with defaults / new
functions freely) and set `STUB = False` when their art is real. Scene authors should use
the libraries (don't redraw the robot yourself), passing poses/params.

## Rendering & checking your work (you can look at PNG/JPG files with the Read tool!)
Run from `/home/user/chat-clear/film`:
```
python3 -m engine.render --scene s03 --times 0,2.5,7.2 --scale 0.5    # stills -> build/stills/
python3 -m engine.render --scene s03 --sheet 12                       # contact sheet -> build/sheets/s03.jpg
python3 -m engine.render --scene s03 --video --scale 0.4 --workers 1  # preview mp4 w/ dialogue
```
Library owners: write a small test script under `build/test_<yourname>/` that draws your
assets on a canvas and saves PNGs (see engine/core.py for skia usage), then LOOK at them.
Iterate until it looks genuinely beautiful — this film is meant to astonish people.

The machine has only 4 CPU cores shared by 14 agents: render stills at scale ≤ 0.5,
sheets at 0.25, preview videos at scale ≤ 0.4 with `--workers 1`, and prefix heavy
commands with `nice -n 10`. Performance budget: a full 1920x1080 frame should render in
≲ 1.5 s total; keep particle counts sane, cache static geometry at module level.

## Environment
Python 3.11 system interpreter has: skia-python, numpy, scipy, soundfile, pillow, librosa,
pyloudnorm, mido, pretty_midi. `ffmpeg`, `sox`, `fluidsynth` (+ `/usr/share/sounds/sf2/FluidR3_GM.sf2`),
`rubberband` are installed. A venv at `/opt/tts` has torch/kokoro/faster-whisper (voices are
already generated). Noto CJK fonts are installed. Do **not** run apt-get. You may `pip install`
small pure-python packages if truly needed.

## Git
Do NOT commit or push. The director commits.

## Final report
When done, reply with: what you built, the public API (for libraries), known issues, and
the path of 1–3 representative images you rendered.
