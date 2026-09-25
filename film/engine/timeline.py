"""The timeline is the single source of truth for timing.

build/timeline.json (written by tools/build_timeline.py) contains:
  scenes: [{id, start, end}]           global seconds
  lines:  [{id, scene, speaker, text, en, start, end, wav}]
  cues:   {name: global_seconds}       story beats used by visuals, music and sfx
build/voice/envelopes.npz holds a 100 Hz loudness envelope per line id
(used for lip-sync).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .core import FPS, clamp, smoothstep

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUILD = os.path.join(ROOT, 'build')
ENV_RATE = 100  # envelope samples per second


class Timeline:
    def __init__(self, path=None):
        path = path or os.path.join(BUILD, 'timeline.json')
        with open(path) as fh:
            d = json.load(fh)
        self.data = d
        self.duration = d['duration']
        self.scenes = d['scenes']
        self.lines = d['lines']
        self.cues = d['cues']
        self.by_line = {ln['id']: ln for ln in self.lines}
        self.by_scene = {s['id']: s for s in self.scenes}
        env_path = os.path.join(BUILD, 'voice', 'envelopes.npz')
        self.env = dict(np.load(env_path)) if os.path.exists(env_path) else {}

    def scene_at(self, T):
        for s in self.scenes:
            if s['start'] <= T < s['end']:
                return s
        return self.scenes[-1]

    def mouth(self, speaker, T):
        """Mouth openness 0..1 for `speaker` at global time T (lip-sync)."""
        for ln in self.lines:
            if ln['speaker'] == speaker and ln['start'] - 0.05 <= T <= ln['end'] + 0.05:
                e = self.env.get(ln['id'])
                if e is None:
                    return 0.0
                i = (T - ln['start']) * ENV_RATE
                i0 = int(np.clip(np.floor(i), 0, len(e) - 1))
                i1 = min(i0 + 1, len(e) - 1)
                fr = clamp(i - i0)
                return float(e[i0] * (1 - fr) + e[i1] * fr)
        return 0.0

    def speaking(self, speaker, T, pad=0.0):
        for ln in self.lines:
            if ln['speaker'] == speaker and ln['start'] - pad <= T <= ln['end'] + pad:
                return ln
        return None

    def line_at(self, T):
        for ln in self.lines:
            if ln['start'] <= T <= ln['end']:
                return ln
        return None


@dataclass
class Frame:
    """Context handed to a scene's render(f)."""
    canvas: object          # skia.Canvas, already scaled to design space 1920x1080
    tl: Timeline
    scene: dict
    T: float                # global time (s)
    index: int = 0          # global frame index
    grade: dict = field(default_factory=dict)   # scenes may set post-processing hints

    @property
    def t(self):
        """Local time since scene start (s)."""
        return self.T - self.scene['start']

    @property
    def dur(self):
        return self.scene['end'] - self.scene['start']

    @property
    def p(self):
        return clamp(self.t / self.dur)

    # --- timing helpers in LOCAL scene time ------------------------------
    def line(self, line_id):
        """(start, end) of a dialogue line in local scene time."""
        ln = self.tl.by_line[line_id]
        s = self.scene['start']
        return ln['start'] - s, ln['end'] - s

    def cue(self, name):
        """Local scene time of a named cue (may be negative / beyond dur)."""
        return self.tl.cues[name] - self.scene['start']

    def since(self, name):
        return self.t - self.cue(name)

    def between(self, a, b):
        """0..1 progress between cues/times a and b (names or local seconds)."""
        ta = self.cue(a) if isinstance(a, str) else a
        tb = self.cue(b) if isinstance(b, str) else b
        return clamp((self.t - ta) / (tb - ta)) if tb != ta else float(self.t >= ta)

    def mouth(self, speaker):
        return self.tl.mouth(speaker, self.T)

    def speaking(self, speaker, pad=0.0):
        return self.tl.speaking(speaker, self.T, pad) is not None

    def fade(self, fin=0.5, fout=0.5):
        """Scene-edge fade factor (use for dissolves to/from black)."""
        a = smoothstep(0, fin, self.t) if fin > 0 else 1.0
        b = 1 - smoothstep(self.dur - fout, self.dur, self.t) if fout > 0 else 1.0
        return a * b


def frame_count(tl):
    return int(round(tl.duration * FPS))
