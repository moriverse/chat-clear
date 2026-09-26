# 《一盏灯，一颗星》 A Lamp and a Star

A 4-minute (242 s) animated short where **everything was made with code**: every frame,
every voice, every note of music and every sound effect. There are no drawings, recordings
or samples; it's about 25,000 lines of Python.

> An old robot has kept a lighthouse at the edge of the world for 300 years, and no ship
> has ever come. One night a little star falls into the sea. She has to get home before
> sunrise or her light will go out. The robot tries everything, and in the end puts his
> own heart into the lamp to shoot a pillar of light into the sky. Then the sky answers.

## How it was made

| Part | How |
|---|---|
| Story & timing | `SCREENPLAY.md`. `tools/build_timeline.py` turns the beat sheet plus the real voice durations into `build/timeline.json`: scenes, 32 lines and 130+ named cues, shared by picture, music and sound. |
| Voices | `audio/voices.py`: local open-weight **Kokoro** TTS (Mandarin). Voices were cast from the data: `audio/cast.py` synthesised a test line with 100 voices and measured pitch (F0) and intelligibility (Whisper speech recognition). Each character gets its own processing: robot resonance, a child's pitch, a whale's reverb. Every line is checked with Whisper, and the loudness envelope drives lip-sync. |
| Picture | Python + **Skia**, drawn in a 1920×1080 design space at 24 fps. `engine/` has the renderer, camera, bloom, grade, grain and bilingual subtitles. `lib/` has the sky, sea & lighthouse, the robot rig 阿灯, the star 小光 and the constellation whale, and light/particle FX. `scenes/` holds the eight scenes. Every frame is a pure function of time, so frames render in parallel. |
| Music | `audio/music.py`: an original score written as note events at exact cue times, played through FluidSynth (General MIDI) with synthesised pads and swells and convolution reverb. Three leitmotifs (lamp, star, whale/home); key plan D → B minor → D → E → D; 54 hits synced to cues. |
| Sound | `audio/sfx.py`: 216 procedurally synthesised effects on cues, plus ocean and wind beds. `audio/mix.py`: sidechain ducking under dialogue, −16 LUFS, true-peak limiter. |
| Team | 14 Claude agents (5 art libraries, 7 scenes, music, sound), coordinated by a director agent through `AGENTS.md`, `STYLE.md` and the shared timeline. |

## Rebuild

```bash
/opt/tts/bin/python audio/voices.py          # voices (Kokoro + Whisper check)
/opt/tts/bin/python tools/build_timeline.py  # timeline.json, dialogue stem, subtitles.srt
python3 audio/music.py                       # score  -> build/audio/music.wav
python3 audio/sfx.py                         # sound  -> build/audio/sfx.wav, ambience.wav
python3 audio/mix.py                         # mix    -> build/audio/final_mix.wav
python3 -m engine.render --all --video --scale 1 --workers 4   # -> build/film.mp4
python3 -m engine.render --scene s06 --sheet 16                # contact sheet of one scene
```

Needs: Python 3.11, skia-python, numpy, scipy, soundfile, librosa, pyloudnorm, pretty_midi,
ffmpeg, fluidsynth + FluidR3_GM.sf2, rubberband, Noto CJK fonts; kokoro + faster-whisper
for the voices.
