"""Build build/timeline.json + build/audio/dialogue.wav from the beat sheet below
and the real durations of the generated voice lines.

    python tools/build_timeline.py
"""
import json
import os

import numpy as np
import soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUILD = os.path.join(ROOT, 'build')
SR = 48000

# (scene id, duration, crossfade-in seconds, {line_id: local start}, {cue: local time})
BEATS = [
    ('s01', 24.0, 0.0,
     {'N01': 7.2, 'N02': 13.6, 'N03': 20.8},
     {'s01_stars_in': 0.3, 'title_in': 1.5, 'title_out': 6.5, 'tilt_start': 6.0, 'tilt_end': 13.5}),
    ('s02', 27.0, 1.0,
     {'D01': 4.4, 'N04': 11.2, 'D02': 20.3},
     {'s02_climb': 0.0, 's02_arrive': 4.0, 'lamp_ignite': 9.4, 's02_exterior': 10.8,
      'timelapse_start': 11.5, 'timelapse_end': 18.5, 's02_sit': 19.3}),
    ('s03', 20.0, 0.0,
     {'D03': 3.3, 'D04': 8.2},
     {'meteor_start': 2.5, 'impact': 7.0, 'run_down': 10.2, 'boat_out': 13.5}),
    ('s04', 29.0, 0.0,
     {'G01': 2.6, 'D05': 5.8, 'G02': 11.7, 'G03': 16.9, 'D06': 21.9},
     {'lift': 0.8, 'look_sky': 16.4, 'determined': 21.2, 'row_back': 25.5}),
    ('s05', 38.0, 0.8,
     {'D07': 1.2, 'G04': 5.4, 'D08': 7.9, 'G05': 18.4, 'G06': 23.0, 'D10': 26.2, 'G07': 32.2},
     {'throw1': 3.0, 'catch': 5.0, 'stack': 10.3, 'climb': 13.5, 'tower_fall': 16.5, 'splash': 17.3,
      'sit_together': 21.8, 'dawn_warning': 31.3, 'idea': 36.4}),
    ('s06', 48.0, 0.0,
     {'D11': 0.6, 'D12': 13.6, 'G08': 18.0, 'D13': 21.2, 'D14': 26.9, 'G09': 36.8, 'D15': 40.2},
     {'run_up': 6.3, 'aim_up': 9.2, 'beam_up': 11.8, 'touch_heart': 16.6, 'heart_out': 30.8,
      'heart_in': 33.0, 'ignite': 33.8, 'rise': 35.3, 'whale_appear': 39.5, 'beam_fade': 44.0,
      'deng_dark': 45.0}),
    ('s07', 34.0, 0.0,
     {'W01': 13.0, 'G10': 17.2, 'D16': 21.5},
     {'whale_song': 3.0, 'stardust': 6.5, 'reboot': 11.0, 'nuzzle': 16.4, 'farewell': 25.8, 'sunrise': 27.0}),
    ('s08', 22.0, 1.5,
     {'N05': 1.2, 'N06': 9.8},
     {'s08_night': 0.0, 'wave': 9.2, 'end_title': 15.0, 'credits': 18.0, 'fade_out': 21.0}),
]


def word_times(wav, needle):
    """Find start times (s) of each occurrence of `needle` characters via Whisper word timestamps."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return []
    m = WhisperModel('small', device='cpu', compute_type='int8')
    segs, _ = m.transcribe(wav, language='zh', word_timestamps=True, initial_prompt='以下是普通话的句子。')
    words = [w for s in segs for w in s.words]
    out = []
    for i, w in enumerate(words):
        if needle in w.word:
            out.append(w.start)
    return out


def main():
    lines = {ln['id']: ln for ln in json.load(open(os.path.join(ROOT, 'script', 'lines.json')))}
    meta = json.load(open(os.path.join(BUILD, 'voice', 'meta.json')))
    scenes, tl_lines, cues = [], [], {}
    T = 0.0
    problems = []
    for sid, dur, xf, lmap, cmap in BEATS:
        scenes.append(dict(id=sid, start=round(T, 3), end=round(T + dur, 3), xfade=xf))
        prev_end = -1
        for lid, ls in sorted(lmap.items(), key=lambda kv: kv[1]):
            d = meta[lid]['speech']
            ln = dict(lines[lid])
            ln.update(start=round(T + ls, 3), end=round(T + ls + d, 3), wav=f'build/voice/{lid}.wav', tail=meta[lid]['tail'])
            if ls < prev_end + 0.25:
                problems.append(f'{lid} starts {ls:.2f} but previous ends {prev_end:.2f}')
            if ls + d > dur:
                problems.append(f'{lid} ends {ls + d:.2f} > scene {sid} dur {dur}')
            prev_end = ls + d
            tl_lines.append(ln)
        for cname, ct in cmap.items():
            cues[cname] = round(T + ct, 3)
        cues[f'{sid}_start'] = round(T, 3)
        cues[f'{sid}_end'] = round(T + dur, 3)
        for lid, ls in lmap.items():
            cues[f'{lid}_start'] = round(T + ls, 3)
            cues[f'{lid}_end'] = round(T + ls + meta[lid]['speech'], 3)
        T += dur

    # precise "blink, blink" sync in the epilogue narration
    n05 = next(ln for ln in tl_lines if ln['id'] == 'N05')
    ts = word_times(os.path.join(BUILD, 'voice', 'N05.dry.wav'), '闪')
    if len(ts) >= 2:
        cues['blink1'] = round(n05['start'] + ts[0] - 0.03, 3)
        cues['blink2'] = round(n05['start'] + ts[1] - 0.03, 3)
    else:
        cues['blink1'] = round(n05['start'] + 5.3, 3)
        cues['blink2'] = round(n05['start'] + 6.1, 3)

    tl = dict(fps=24, duration=round(T, 3), width=1920, height=1080, scenes=scenes, lines=tl_lines,
              cues=dict(sorted(cues.items(), key=lambda kv: kv[1])))
    json.dump(tl, open(os.path.join(BUILD, 'timeline.json'), 'w'), indent=1, ensure_ascii=False)

    # dialogue stem
    os.makedirs(os.path.join(BUILD, 'audio'), exist_ok=True)
    n = int((T + 2) * SR)
    mixbuf = np.zeros(n, np.float32)
    for ln in tl_lines:
        x, sr = sf.read(os.path.join(ROOT, ln['wav']), dtype='float32')
        i = int(ln['start'] * SR)
        mixbuf[i:i + len(x)] += x[: n - i]
    mixbuf = mixbuf[: int(T * SR)]
    sf.write(os.path.join(BUILD, 'audio', 'dialogue.wav'), np.stack([mixbuf, mixbuf], 1) * 0.9, SR)

    # subtitles (.srt)
    def ts_(s):
        h, m = int(s // 3600), int(s % 3600 // 60)
        return f'{h:02d}:{m:02d}:{s % 60:06.3f}'.replace('.', ',')
    with open(os.path.join(BUILD, 'subtitles.srt'), 'w') as fh:
        for i, ln in enumerate(tl_lines, 1):
            fh.write(f"{i}\n{ts_(ln['start'])} --> {ts_(ln['end'] + 0.3)}\n{ln['text']}\n{ln['en']}\n\n")

    print(f'duration {T:.1f}s, {len(tl_lines)} lines, {len(cues)} cues')
    for p in problems:
        print('PROBLEM:', p)


if __name__ == '__main__':
    main()
