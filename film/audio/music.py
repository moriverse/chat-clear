#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
audio/music.py — MUSIC: the original orchestral score for 《一盏灯，一颗星》 (A Lamp and a Star).

Everything is written by code.  Notes are composed in absolute film seconds, anchored to the
cue / line times in build/timeline.json (so the score follows the cut if the timeline is
regenerated), rendered one instrument per pass with FluidSynth + FluidR3 GM, then processed
in numpy: EQ, panning, dialogue-aware thinning, synthesized layers (analog pads, high
shimmer, star glitter, sub booms, cymbal swells, risers, inharmonic bells, a boing and a
slide whistle for the comedy), a synthetic concert-hall convolution reverb, and mastering.

Leitmotifs
  * Lamp theme  (阿灯)  — warm, a little lonely: 5 | 3' . 2' | 1' . 6 | 5 ...  (music box / piano)
  * Star theme  (小光)  — twinkly & playful: 5 1' 3' 2'3' | 5' ...           (celesta / glock / harp)
  * Whale theme (星鲸)  — broad & awe-filled: 1 . 5 . | 6 . . 5 | 3' 2' 1' | 5  (strings / choir / horns)
Key plan: D major (home) → B minor (mystery) → D → E major for the climax and the miracle
→ D major again for the epilogue.

Usage:   nice -n 10 python3 audio/music.py [--analyze] [--no-cache]
Outputs: build/audio/music.wav (48 kHz / 24-bit stereo, exactly the film length),
         build/audio/music_stems/*.wav (they sum to music.wav),
         build/music/cues.json (sections, leitmotif statements, every hit point),
         build/music/*.png + build/audio/preview_music_dialogue.wav with --analyze.
"""
import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pretty_midi as pm
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / 'build'
MUSDIR = BUILD / 'music'
AUD = BUILD / 'audio'
STEMDIR = AUD / 'music_stems'
CACHE = MUSDIR / 'cache'
SF2 = '/usr/share/sounds/sf2/FluidR3_GM.sf2'
SR = 48000

TL = json.load(open(BUILD / 'timeline.json'))
DUR = float(TL['duration'])
NS = int(round(DUR * SR))
CUE = {k: float(v) for k, v in TL['cues'].items()}
LINES = [(l['id'], float(l['start']), float(l['end']), l['speaker']) for l in TL['lines']]
LINE = {l[0]: (l[1], l[2]) for l in LINES}
SCENE = {s['id']: (float(s['start']), float(s['end'])) for s in TL['scenes']}


def c(name):
    return CUE[name]


def db(x):
    return 10.0 ** (x / 20.0)


def mtof(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


# ============================================================================ music theory
_PC = dict(C=0, D=2, E=4, F=5, G=7, A=9, B=11)


def P(x):
    """'F#5' -> 78; ints pass through."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = x.strip()
    pc = _PC[s[0].upper()]
    i = 1
    while i < len(s) and s[i] in '#b':
        pc += 1 if s[i] == '#' else -1
        i += 1
    return 12 * (int(s[i:]) + 1) + pc


def pc_of(s):
    pc = _PC[s[0].upper()]
    for ch in s[1:]:
        pc += 1 if ch == '#' else (-1 if ch == 'b' else 0)
    return pc % 12


QUAL = {
    '': [0, 4, 7], 'm': [0, 3, 7], '7': [0, 4, 7, 10], 'maj7': [0, 4, 7, 11], 'm7': [0, 3, 7, 10],
    'm9': [0, 3, 7, 10, 14], 'maj9': [0, 4, 7, 11, 14], 'add9': [0, 4, 7, 14], 'madd9': [0, 3, 7, 14],
    'sus2': [0, 2, 7], 'sus4': [0, 5, 7], '7sus4': [0, 5, 7, 10], '6': [0, 4, 7, 9], 'm6': [0, 3, 7, 9],
    'dim': [0, 3, 6], 'dim7': [0, 3, 6, 9], 'm7b5': [0, 3, 6, 10], '9': [0, 4, 7, 10, 14],
    '69': [0, 4, 7, 9, 14], 'maj7#11': [0, 4, 7, 11, 18], 'add9#11': [0, 4, 7, 14, 18], 'aug': [0, 4, 8],
    '7b9': [0, 4, 7, 10, 13], '5': [0, 7],
}


def parse_chord(sym):
    bass = None
    if '/' in sym:
        sym, b = sym.split('/')
        bass = pc_of(b)
    n = 2 if len(sym) > 1 and sym[1] in '#b' else 1
    root = pc_of(sym[:n])
    ivs = QUAL[sym[n:]]
    pcs = [(root + i) % 12 for i in ivs]
    return root, pcs, (root if bass is None else bass)


def chord_pcs(sym):
    return parse_chord(sym)[1]


def bass_of(sym, lo='D2', hi='C#3'):
    b = parse_chord(sym)[2]
    for p in range(P(lo), P(hi) + 1):
        if p % 12 == b:
            return p
    return P(lo) + ((b - P(lo)) % 12)


def voicing(sym, lo, hi, n, prev=None, center=None):
    """Choose n chord tones in [lo, hi] covering the essential tones, voice-led from prev."""
    root, pcs, _ = parse_chord(sym)
    upcs = list(dict.fromkeys(pcs))
    need = list(upcs)
    if len(need) > n and (root + 7) % 12 in need:
        need.remove((root + 7) % 12)
    while len(need) > n:
        need.pop()
    need = set(need)
    lo, hi = P(lo), P(hi)
    cands = [p for p in range(lo, hi + 1) if p % 12 in upcs]
    best, bc = None, 1e18
    ctr = center if center is not None else (lo + hi) / 2.0
    for comb in itertools.combinations(cands, n):
        s = {p % 12 for p in comb}
        if not need <= s:
            continue
        g = np.diff(comb)
        if len(g) and g.max() > 12:
            continue
        cost = 3.0 * np.sum(g == 1) + 1.0 * np.sum(g == 2) * (comb[0] < 57) + 1.5 * (n - len(s))
        if prev is not None and len(prev) == n:
            cost += sum(abs(a - b) for a, b in zip(sorted(prev), comb))
        else:
            cost += 0.5 * abs(np.mean(comb) - ctr)
        if cost < bc:
            bc, best = cost, comb
    if best is None:
        best = tuple(cands[:n])
    return list(best)


def scale_pcs(tonic, mode='major'):
    steps = {'major': [0, 2, 4, 5, 7, 9, 11], 'minor': [0, 2, 3, 5, 7, 8, 10],
             'pent': [0, 2, 4, 7, 9], 'hminor': [0, 2, 3, 5, 7, 8, 11]}[mode]
    t = pc_of(tonic)
    return {(t + s) % 12 for s in steps}


def seq(text, tr=0):
    """'A4:1 | F#5:2 E5:1' -> [(69,1.0), (78,2.0), (76,1.0)];  r = rest."""
    out = []
    for tok in text.split():
        if tok == '|':
            continue
        name, b = tok.split(':')
        out.append((None if name in ('r', '-') else P(name) + tr, float(b)))
    return out


class TM:
    """Tempo map: piecewise-linear beat -> seconds through (beat, time) anchors."""

    def __init__(self, anchors):
        self.b = np.array([a[0] for a in anchors], float)
        self.t = np.array([a[1] for a in anchors], float)

    def __call__(self, beat):
        b, t = self.b, self.t
        if beat <= b[0]:
            return float(t[0] + (beat - b[0]) * (t[1] - t[0]) / (b[1] - b[0]))
        if beat >= b[-1]:
            return float(t[-1] + (beat - b[-1]) * (t[-1] - t[-2]) / (b[-1] - b[-2]))
        return float(np.interp(beat, b, t))


# ---------------------------------------------------------------- leitmotifs (D major)
LAMP_A = "A4:1 | F#5:2 E5:1 | D5:2 B4:1 | A4:3 | B4:1 C#5:1 D5:1 | E5:2 F#5:1 | E5:3"
LAMP_A_CH = ['D', 'G', 'Bm7', 'Em7', 'Asus4', 'A']
LAMP_B = "A4:1 | F#5:2 A5:1 | G5:2 F#5:1 | E5:1 D5:1 B4:1 | A4:2 C#5:1 | D5:3"
LAMP_B_CH = ['D', 'Gmaj7', 'Em7', 'A7', 'D']
STAR = ("A5:.5 D6:.5 F#6:.5 E6:.25 F#6:.25 | A6:1 F#6:.5 D6:.5 | E6:.5 B5:.5 D6:.5 A5:.5 | "
        "B5:.5 C#6:.5 D6:1 | A5:.5 D6:.5 F#6:.5 G6:.25 F#6:.25 | B6:1 A6:.5 F#6:.5 | "
        "G6:.5 E6:.5 C#6:.5 A5:.5 | D6:2")
STAR_HEAD = "A5:.5 D6:.5 F#6:.5 E6:.25 F#6:.25 A6:1"
WHALE_P1 = "E4:2 B4:2 | C#5:3 B4:1 | G#5:2 F#5:1 E5:1 | B4:4"   # E major


# ============================================================================ score container
class Track:
    def __init__(self, name, prog, stem, pan=0.0, send=0.3, space=0.0, gain=0.0, drum=False,
                 hp=None, lp=None, shelf=None, lead=False, duck=-3.0, gate=True, cal=60):
        self.name, self.prog, self.stem, self.pan = name, prog, stem, pan
        self.send, self.space, self.gain, self.drum = send, space, gain, drum
        self.hp, self.lp, self.shelf, self.lead, self.duck = hp, lp, shelf, lead, duck
        self.gate, self.cal = gate, cal
        self.notes, self.bends, self.bumps = [], [], []


class Score:
    def __init__(self):
        self.tr = {}
        self.fx = []           # (kind, stem, send, space, gate, kwargs)
        self.hits = []
        self.motifs = []
        self.sections = []
        self.harmony = []      # (t, chord symbol) — drives the synth pad / shimmer / glitter
        self.curves = {'pad': [], 'shim': [], 'glit': []}

    def track(self, name, *a, **k):
        self.tr[name] = Track(name, *a, **k)

    def n(self, name, p, t, d, v):
        if d <= 0.01 or t >= DUR:
            return
        self.tr[name].notes.append([P(p), float(t), float(d), float(v)])

    def ch(self, name, ps, t, d, v, roll=0.0, vstep=0.0):
        for i, p in enumerate(ps):
            self.n(name, p, t + i * roll, d - i * roll, v + i * vstep)

    def mel(self, name, sq, t0, spb, v, leg=0.97, tr=0, accent=None):
        t = t0
        for i, (p, b) in enumerate(sq):
            d = b * spb
            if p is not None:
                vv = v(i, t) if callable(v) else v
                self.n(name, p + tr, t, d * leg, vv)
            t += d
        return t

    def mel_tm(self, name, sq, tm, b0, v, leg=0.97, tr=0):
        b = b0
        for p, nb in sq:
            if p is not None:
                t0, t1 = tm(b), tm(b + nb)
                self.n(name, p + tr, t0, (t1 - t0) * leg, v)
            b += nb
        return b

    def bump(self, name, pts):
        """Local gain automation (dB) — 0 dB outside the first/last point."""
        self.tr[name].bumps.append(sorted(pts))

    def bend(self, name, pts):
        self.tr[name].bends += pts

    def harm(self, t, sym):
        self.harmony.append((float(t), sym))

    def curve(self, which, pts):
        self.curves[which] += pts

    def fxa(self, kind, stem='fx', send=0.25, space=0.0, gate=True, **kw):
        self.fx.append((kind, stem, send, space, gate, kw))

    def hit(self, name, t, kind, desc):
        self.hits.append(dict(name=name, time=round(float(t), 3), kind=kind, desc=desc))

    def motif(self, name, t, inst, note=''):
        self.motifs.append(dict(motif=name, time=round(float(t), 3), instrument=inst, note=note))

    def section(self, sid, t0, t1, key, feel):
        self.sections.append(dict(scene=sid, start=t0, end=t1, key=key, feel=feel))

    def chord_at(self, t):
        h = sorted(self.harmony)
        cur = h[0][1]
        for tt, s in h:
            if tt <= t + 1e-9:
                cur = s
            else:
                break
        return cur


# ---------------------------------------------------------------- writing helpers
def sustain(S, name, segs, v, tie=True, ext=0.05, vfn=None):
    """segs: [(t0, t1, [pitches])] — ties common tones between consecutive segments."""
    active = {}
    for i, (t0, t1, ps) in enumerate(segs):
        ps = [P(p) for p in ps]
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        for p in ps:
            vv = vfn(t0, v) if vfn else v
            if tie and p in active and abs(active[p][1] - t0) < 1e-6:
                active[p][1] = t1
            else:
                if p in active:
                    a = active.pop(p)
                    S.n(name, p, a[0], a[1] - a[0] + ext, a[2])
                active[p] = [t0, t1, vv]
        for p in list(active):
            if p not in ps or nxt is None or abs(nxt[0] - t1) > 1e-6 or P(p) not in [P(q) for q in nxt[2]]:
                a = active.pop(p)
                S.n(name, p, a[0], a[1] - a[0] + ext, a[2])


def pad_strings(S, segs, v_hi=40, v_lo=40, hi=('G3', 'A4'), n_hi=4, lo=True, lo_rng=('D2', 'C#3'),
                fifth=True, basses=None, choir=None, v_ch=40, ch_rng=('A3', 'E5'), n_ch=3, harm=True):
    """segs: [(t0, t1, sym)] -> sustained string pad (+ optional basses / choir)."""
    prev = None
    hs, ls, bs, cs = [], [], [], []
    prevc = None
    for (t0, t1, sym) in segs:
        if harm:
            S.harm(t0, sym)
        up = voicing(sym, hi[0], hi[1], n_hi, prev)
        prev = up
        hs.append((t0, t1, up))
        b = bass_of(sym, *lo_rng)
        if lo:
            root = parse_chord(sym)[0]
            lp = [b]
            if fifth:
                f5 = b + ((root + 7 - b) % 12)
                if f5 - b < 3:
                    f5 += 12
                lp.append(f5)
            ls.append((t0, t1, lp))
        if basses:
            bs.append((t0, t1, [b - 12]))
        if choir:
            cv = voicing(sym, ch_rng[0], ch_rng[1], n_ch, prevc)
            prevc = cv
            cs.append((t0, t1, cv))
    sustain(S, 'spad_hi', hs, v_hi)
    if lo:
        sustain(S, 'spad_lo', ls, v_lo)
    if basses:
        sustain(S, 'basses', bs, basses)
    if choir:
        sustain(S, choir, cs, v_ch)


def arp(S, name, ps, t0, t1, step, v, pattern='up', ring=1.2, vjit=5, seed=0, accent=0):
    rng = np.random.default_rng(seed)
    ps = [P(p) for p in ps]
    if pattern == 'updown' and len(ps) > 2:
        cyc = ps + ps[-2:0:-1]
    elif pattern == 'down':
        cyc = ps[::-1]
    else:
        cyc = ps
    t, i = t0, 0
    while t < t1 - 1e-6:
        vv = v + rng.integers(-vjit, vjit + 1) + (accent if i % len(cyc) == 0 else 0)
        S.n(name, cyc[i % len(cyc)], t, step * ring, vv)
        t += step
        i += 1


def gliss(S, name, lo, hi, t_end, dur, v0, v1, pcs, ring=1.6, down=False, ease=1.0):
    notes = [p for p in range(P(lo), P(hi) + 1) if p % 12 in pcs]
    if down:
        notes = notes[::-1]
    k = len(notes)
    for i, p in enumerate(notes):
        x = i / max(1, k - 1)
        t = t_end - dur + dur * (x ** ease)
        S.n(name, p, t, ring, v0 + (v1 - v0) * x)


def timp_roll(S, p, t0, t1, v0, v1, r0=14.0, r1=20.0, seed=5):
    rng = np.random.default_rng(seed)
    t = t0
    while t < t1 - 0.02:
        x = (t - t0) / max(1e-6, t1 - t0)
        S.n('timp', p, t, 0.3, (v0 + (v1 - v0) * x) * (0.9 + 0.1 * rng.random()))
        t += 1.0 / (r0 + (r1 - r0) * x)


def sparkle_burst(S, t0, dur, n, pcs, lo, hi, v0, v1, seed, tracks=('celesta', 'glock', 'harp')):
    rng = np.random.default_rng(seed)
    pool = [p for p in range(P(lo), P(hi) + 1) if p % 12 in pcs]
    ts = np.sort(t0 + dur * rng.random(n) ** 1.6)
    for i, t in enumerate(ts):
        x = i / max(1, n - 1)
        S.n(tracks[i % len(tracks)], int(rng.choice(pool)), t, 1.0, v0 + (v1 - v0) * x + rng.integers(-4, 5))


def star_figure(S, name, sym, t0, t1, step, v, lo='A5', seed=0):
    """Star-theme figuration: rising chord tones with a little upper-neighbour flicker."""
    rng = np.random.default_rng(seed)
    pcs = chord_pcs(sym)
    pool = [p for p in range(P(lo), P(lo) + 20) if p % 12 in pcs]
    shape = [0, 1, 2, 3, 2, 3, 4, 2]
    t, i = t0, 0
    while t < t1 - 1e-6:
        idx = min(len(pool) - 1, shape[i % len(shape)])
        S.n(name, pool[idx], t, step * 1.6, v + rng.integers(-5, 6) + (6 if i % 8 == 0 else 0))
        t += step
        i += 1


# ============================================================================ DSP: synth voices
def _t(n):
    return np.arange(n, dtype=np.float64) / SR


def cos_ramp(n):
    return (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, max(1, n)))).astype(np.float32)


def fade_env(n, a, r):
    e = np.ones(n, np.float32)
    na, nr = min(n, int(a * SR)), min(n, int(r * SR))
    if na > 0:
        e[:na] *= cos_ramp(na)
    if nr > 0:
        e[n - nr:] *= cos_ramp(nr)[::-1]
    return e


def _norm(y):
    m = np.abs(y).max()
    return y / m if m > 0 else y


_SAW = {}


def saw_osc(freq, n, phase0, nh):
    nh = int(max(1, min(64, nh)))
    if nh not in _SAW:
        N = 4096
        ph = np.arange(N) / N
        k = np.arange(1, nh + 1)
        tab = (np.sin(2 * np.pi * np.outer(ph, k)) * (((-1.0) ** (k + 1)) / k)).sum(1) * (2 / np.pi)
        _SAW[nh] = np.append(tab, tab[0]).astype(np.float32)
    tab = _SAW[nh]
    if np.isscalar(freq):
        ph = (phase0 + np.arange(n) * (freq / SR)) % 1.0
    else:
        ph = (phase0 + np.cumsum(freq) / SR) % 1.0
    idx = ph * 4096
    i0 = idx.astype(np.int64)
    fr = (idx - i0).astype(np.float32)
    return tab[i0] * (1 - fr) + tab[i0 + 1] * fr


def fx_crash(lvl, decay=2.6, dur=6.0, seed=0, bright=1.0):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = _t(n)
    noise = rng.standard_normal((n, 2))
    b = signal.sosfilt(signal.butter(2, 2600 * bright, 'hp', fs=SR, output='sos'), noise, axis=0)
    m = np.zeros((n, 2))
    for k in range(28):
        f = np.exp(rng.uniform(np.log(1900), np.log(11500)))
        dk = decay * rng.uniform(0.25, 1.0)
        a = rng.uniform(0.3, 1.0)
        for ch in range(2):
            m[:, ch] += a * np.sin(2 * np.pi * f * (1 + 0.002 * ch) * t + rng.uniform(0, 6.28)) * np.exp(-t / dk)
    env = (1 - np.exp(-t / 0.0015)) * (0.5 * np.exp(-t / 0.10) + 0.5 * np.exp(-t / decay))
    y = (_norm(b) * 0.8 + _norm(m) * 0.35) * env[:, None]
    yd = signal.sosfilt(signal.butter(2, 4500, 'lp', fs=SR, output='sos'), y, axis=0)
    w = np.exp(-t / 0.7)[:, None]
    y = y * w + yd * (1 - w) * 1.3
    return (_norm(y) * db(lvl)).astype(np.float32)


def fx_revcym(lvl, length=1.5, seed=1):
    y = fx_crash(0, decay=2.2, dur=max(length, 0.3) + 0.05, seed=seed)[::-1].copy()
    y = y[-int(length * SR):]
    y[-int(0.006 * SR):] *= cos_ramp(int(0.006 * SR))[::-1, None]
    y *= np.linspace(0.0, 1.0, len(y))[:, None] ** 1.5
    return (_norm(y) * db(lvl)).astype(np.float32)


def fx_boom(lvl, f_hi=72, f_lo=30, decay=1.7, dur=6.0, seed=2):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = _t(n)
    f = f_lo + (f_hi - f_lo) * np.exp(-t / 0.22)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * (1 - np.exp(-t / 0.004)) * np.exp(-t / decay)
    body = np.tanh(2.2 * body) / np.tanh(2.2)
    th = signal.sosfilt(signal.butter(2, 160, 'lp', fs=SR, output='sos'), rng.standard_normal(n))
    th = _norm(th) * np.exp(-t / 0.08) * (1 - np.exp(-t / 0.002)) * 0.55
    y = body + th
    y[-int(0.05 * SR):] *= cos_ramp(int(0.05 * SR))[::-1]
    return (np.stack([y, y], 1) * db(lvl) / np.abs(y).max()).astype(np.float32)


def fx_inhale(lvl, length=1.2, seed=3):
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = _t(n)
    x = t / length
    noise = signal.sosfilt(signal.butter(2, [60, 900], 'bp', fs=SR, output='sos'), rng.standard_normal((n, 2)), axis=0)
    sub = np.sin(2 * np.pi * np.cumsum(38 + 30 * x) / SR)
    y = (_norm(noise) * 0.6 + sub[:, None] * 0.6) * (x ** 3)[:, None]
    y[-int(0.01 * SR):] *= cos_ramp(int(0.01 * SR))[::-1, None]
    return (_norm(y) * db(lvl)).astype(np.float32)


def fx_riser(lvl, length, f0=250.0, f1=7000.0, root=None, seed=4):
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    out = np.zeros((n, 2))
    for ch in range(2):
        noise = rng.standard_normal(n + 4096)
        f, tt, Z = signal.stft(noise, fs=SR, nperseg=2048, noverlap=1536)
        prog = np.clip(tt / length, 0, 1)
        fc = f0 * (f1 / f0) ** (prog ** 1.5)
        mask = np.exp(-0.5 * (np.log2((f[:, None] + 20) / fc[None, :]) / 0.5) ** 2)
        _, y = signal.istft(Z * mask, fs=SR, nperseg=2048, noverlap=1536)
        out[:, ch] = y[:n]
    out = _norm(out)
    x = _t(n) / length
    if root is not None:
        ton = np.zeros(n)
        for k, iv in enumerate([0, 7, 12, 16]):
            fr = mtof(root + iv) * 2 ** (x * 1.0)
            ton += np.sin(2 * np.pi * np.cumsum(fr) / SR + k) / (k + 1)
        out = out * 0.75 + _norm(ton)[:, None] * 0.35
    out *= (x ** 2.2)[:, None]
    out[-int(0.012 * SR):] *= cos_ramp(int(0.012 * SR))[::-1, None]
    return (_norm(out) * db(lvl)).astype(np.float32)


BELLP = [(0.5, 0.35, 1.6), (1.0, 1.0, 1.0), (1.19, 0.45, 0.8), (1.56, 0.35, 0.6), (2.0, 0.5, 0.55),
         (2.51, 0.25, 0.4), (2.66, 0.2, 0.35), (3.01, 0.18, 0.3), (4.17, 0.12, 0.2), (5.43, 0.07, 0.15)]


def fx_bells(lvl, midis, decay=3.2, strum=0.012, seed=5):
    rng = np.random.default_rng(seed)
    n = int((decay * 2.2 + 0.3) * SR)
    t = _t(n)
    out = np.zeros((n, 2))
    for i, m in enumerate(midis):
        f0 = mtof(P(m))
        off = int(i * strum * SR)
        y = np.zeros(n)
        for r, a, d in BELLP:
            if f0 * r > 16000:
                continue
            y += a * np.sin(2 * np.pi * f0 * r * t + rng.uniform(0, 6.28)) * np.exp(-t / (decay * d))
        y *= (1 - np.exp(-t / 0.0012))
        pan = -0.5 + i / max(1, len(midis) - 1)
        g = np.array([np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)]) * 1.41
        out[off:] += (y[:n - off, None] * g[None, :])
    return (_norm(out) * db(lvl)).astype(np.float32)


def fx_boing(lvl, f0=190.0, dur=0.9):
    n = int(dur * SR)
    t = _t(n)
    f = f0 * (0.72 + 0.28 * (1 - np.exp(-t / 0.035))) * (1 + 0.16 * np.exp(-t / 0.22) * np.sin(2 * np.pi * 10.5 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    form = 0.55 + 0.45 * np.sin(2 * np.pi * 10.5 * t - 1.0)
    y = (np.sin(ph) + 0.55 * form * np.sin(2 * ph) + 0.3 * form * np.sin(3 * ph) + 0.12 * np.sin(5 * ph))
    y *= (1 - np.exp(-t / 0.003)) * np.exp(-t / 0.3)
    y[-int(0.03 * SR):] *= cos_ramp(int(0.03 * SR))[::-1]
    return (np.stack([y, y * 0.9], 1) * db(lvl) / np.abs(y).max()).astype(np.float32)


def fx_slide(lvl, dur, fa, fb, seed=6):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = _t(n)
    x = t / dur
    s = x * x * (3 - 2 * x)
    f = fa * (fb / fa) ** s * (1 + 0.012 * np.sin(2 * np.pi * 5.5 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    breath = signal.sosfilt(signal.butter(2, 3000, 'hp', fs=SR, output='sos'), rng.standard_normal(n))
    y = np.sin(ph) + 0.08 * np.sin(2 * ph) + 0.05 * _norm(breath) * (0.5 + 0.5 * np.sin(ph))
    y *= fade_env(n, 0.03, 0.07)
    return (np.stack([y, y], 1) * db(lvl) / np.abs(y).max()).astype(np.float32)


def fx_whoosh(lvl, dur, f0=400.0, f1=5000.0, seed=7):
    y = fx_riser(0, dur, f0, f1, seed=seed)
    n = len(y)
    x = np.linspace(0, 1, n)
    y *= (np.sin(np.pi * np.clip(x * 1.1, 0, 1)) / np.maximum(x ** 2.2, 1e-3)).clip(0, 50)[:, None]
    return (_norm(y) * db(lvl)).astype(np.float32)


def fx_ping(lvl, midi, dur=2.5, seed=8):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    t = _t(n)
    f = mtof(P(midi))
    y = (np.sin(2 * np.pi * f * t) * np.exp(-t / 0.9) + 0.35 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-t / 0.45)
         + 0.15 * np.sin(2 * np.pi * 3.01 * f * t) * np.exp(-t / 0.25))
    y *= (1 - np.exp(-t / 0.0015))
    out = np.stack([y, y], 1)
    for k in range(7):          # a few glints around it
        tk = rng.uniform(0.02, 0.35)
        fk = f * rng.choice([1.5, 2.0, 2.5, 3.0])
        if fk > 12000:
            fk /= 2
        o = int(tk * SR)
        m = n - o
        tt = t[:m]
        g = 0.25 * np.sin(2 * np.pi * fk * tt) * np.exp(-tt / rng.uniform(0.08, 0.25)) * (1 - np.exp(-tt / 0.001))
        pan = rng.uniform(-0.8, 0.8)
        out[o:, 0] += g * (1 - pan) / 2 * 2
        out[o:, 1] += g * (1 + pan) / 2 * 2
    return (_norm(out) * db(lvl)).astype(np.float32)


def fx_lowdrone(lvl, midi, dur=4.5):
    n = int(dur * SR)
    t = _t(n)
    f = mtof(P(midi))
    y = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t + 0.3) + 0.12 * np.sin(2 * np.pi * 3 * f * t)
    y *= (1 - np.exp(-t / 0.03)) * np.exp(-t / (dur / 4.2))
    y[-int(0.1 * SR):] *= cos_ramp(int(0.1 * SR))[::-1]
    return (np.stack([y, y], 1) * db(lvl) / np.abs(y).max()).astype(np.float32)


FXGEN = dict(crash=fx_crash, revcym=fx_revcym, boom=fx_boom, inhale=fx_inhale, riser=fx_riser, bells=fx_bells,
             boing=fx_boing, slide=fx_slide, whoosh=fx_whoosh, ping=fx_ping, lowdrone=fx_lowdrone)
# generators whose sound must END at time t (placed backwards)
FX_ENDS = {'revcym', 'inhale'}


# ---------------------------------------------------------------- control curves
CTL_HZ = 100
NCTL = int(DUR * CTL_HZ) + 2
CT = np.arange(NCTL) / CTL_HZ


def curve_ctl(pts, default=-90.0):
    if not pts:
        return np.full(NCTL, default)
    pts = sorted(pts)
    return np.interp(CT, [p[0] for p in pts], [p[1] for p in pts])


def ctl_to_audio(ctl, a=0, b=None):
    b = NS if b is None else b
    ts = np.arange(a, b) / SR
    return np.interp(ts, CT, ctl).astype(np.float32)


def bumps_ctl(bumps):
    out = np.zeros(NCTL)
    for pts in bumps:
        t0, t1 = pts[0][0], pts[-1][0]
        m = (CT >= t0) & (CT <= t1)
        out[m] += np.interp(CT[m], [p[0] for p in pts], [p[1] for p in pts])
    return out


def duck_ctl(pre=0.15, post=0.45):
    d = np.zeros(NCTL)
    for _, s, e, _ in LINES:
        x = np.zeros(NCTL)
        x[(CT >= s) & (CT <= e)] = 1
        r = (CT >= s - pre) & (CT < s)
        x[r] = 0.5 - 0.5 * np.cos(np.pi * (CT[r] - (s - pre)) / pre)
        r = (CT > e) & (CT <= e + post)
        x[r] = 0.5 + 0.5 * np.cos(np.pi * (CT[r] - e) / post)
        d = np.maximum(d, x)
    return d


DUCK = duck_ctl()


def in_line(t, pad=0.1):
    return any(s - pad <= t <= e + pad for _, s, e, _ in LINES)


# ---------------------------------------------------------------- synth layers driven by the harmony
def harmony_segments(S):
    h = sorted(S.harmony)
    ded = []
    for t, s in h:
        if ded and abs(ded[-1][0] - t) < 1e-6:
            ded[-1] = (t, s)
        elif not ded or ded[-1][1] != s:
            ded.append((t, s))
    return [(t, ded[i + 1][0] if i + 1 < len(ded) else DUR, s) for i, (t, s) in enumerate(ded)]


def synth_pad(S, ctl):
    out = np.zeros((NS, 2), np.float32)
    rng = np.random.default_rng(21)
    lp = signal.butter(2, 1100, 'lp', fs=SR, output='sos')
    prev = None
    for a, b, sym in harmony_segments(S):
        i0, i1 = int(a * CTL_HZ), min(NCTL - 1, int(b * CTL_HZ) + 1)
        if ctl[i0:i1 + 1].max() < -65:
            continue
        up = voicing(sym, 'F3', 'D5', 3, prev)
        prev = up
        notes = [bass_of(sym, 'A1', 'G#2')] + up
        n0 = max(0, int((a - 0.35) * SR))
        n1 = min(NS, int((b + 1.4) * SR))
        n = n1 - n0
        if n <= 0:
            continue
        tt = _t(n)
        y = np.zeros((n, 2), np.float32)
        for j, m in enumerate(notes):
            f = mtof(m)
            for k, cents in enumerate((-10.0, -3.0, 4.0, 11.0)):
                ff = f * 2 ** (cents / 1200) * (1 + 0.0012 * np.sin(2 * np.pi * (0.11 + 0.05 * k) * tt + k + j))
                v = saw_osc(ff, n, rng.random(), 16000 / f) * (0.9 if j else 1.3)
                pan = (-0.7, 0.35, -0.35, 0.7)[k]
                y[:, 0] += v * (1 - pan) * 0.5
                y[:, 1] += v * (1 + pan) * 0.5
        y = signal.sosfilt(lp, y, axis=0).astype(np.float32)
        env = np.ones(n, np.float32)
        na = int(0.7 * SR)
        env[:na] = cos_ramp(na)
        rel0 = int((b - (a - 0.35)) * SR)
        if rel0 < n:
            env[rel0:] = cos_ramp(n - rel0)[::-1]
        out[n0:n1] += y * env[:, None] * 0.05
    return out * ctl_to_audio(10 ** (ctl / 20))[:, None]


def synth_shimmer(S, ctl):
    out = np.zeros((NS, 2), np.float32)
    rng = np.random.default_rng(22)
    for a, b, sym in harmony_segments(S):
        i0, i1 = int(a * CTL_HZ), min(NCTL - 1, int(b * CTL_HZ) + 1)
        if ctl[i0:i1 + 1].max() < -65:
            continue
        pcs = chord_pcs(sym)
        ps = [p for p in range(P('C6'), P('F7') + 1) if p % 12 in pcs][:6]
        n0 = max(0, int((a - 0.6) * SR))
        n1 = min(NS, int((b + 1.6) * SR))
        n = n1 - n0
        tt = _t(n)
        y = np.zeros((n, 2))
        for j, m in enumerate(ps):
            f = mtof(m)
            am = 0.55 + 0.45 * np.sin(2 * np.pi * rng.uniform(0.07, 0.3) * tt + rng.uniform(0, 6.28))
            for ch in range(2):
                y[:, ch] += am * (np.sin(2 * np.pi * f * (1 + 0.0025 * ch) * tt + rng.uniform(0, 6.28))
                                  + 0.5 * np.sin(2 * np.pi * f * (1 - 0.002 + 0.004 * ch) * tt)) / (1 + 0.3 * j)
        env = np.ones(n)
        na = int(1.0 * SR)
        env[:na] = cos_ramp(na)
        rel0 = int((b - (a - 0.6)) * SR)
        if rel0 < n:
            env[rel0:] = cos_ramp(n - rel0)[::-1]
        out[n0:n1] += (y * env[:, None] * 0.02).astype(np.float32)
    out = signal.sosfilt(signal.butter(2, 900, 'hp', fs=SR, output='sos'), out, axis=0).astype(np.float32)
    return out * ctl_to_audio(10 ** (ctl / 20))[:, None]


def synth_glitter(S, dens):
    out = np.zeros((NS, 2), np.float32)
    rng = np.random.default_rng(23)
    R = 40.0
    t = 0.0
    while True:
        t += rng.exponential(1.0 / R)
        if t >= DUR - 0.5:
            break
        d = dens[int(t * CTL_HZ)]
        if rng.random() > d / R:
            continue
        pcs = chord_pcs(S.chord_at(t))
        pool = [p for p in range(P('G6'), P('E8') + 1) if p % 12 in pcs]
        m = int(rng.choice(pool))
        f = mtof(m)
        L = rng.uniform(0.12, 0.55)
        n = int(L * 4 * SR)
        n0 = int(t * SR)
        n = min(n, NS - n0)
        tt = _t(n)
        g = (np.sin(2 * np.pi * f * tt) + 0.2 * np.sin(2 * np.pi * 2 * f * tt)) * np.exp(-tt / L) * (1 - np.exp(-tt / 0.002))
        g *= rng.uniform(0.25, 1.0) * 0.03
        pan = rng.uniform(-0.9, 0.9)
        out[n0:n0 + n, 0] += (g * (1 - pan) / 2 * 2).astype(np.float32)
        out[n0:n0 + n, 1] += (g * (1 + pan) / 2 * 2).astype(np.float32)
    return out


# ============================================================================ dialogue helpers
_DLG = None


def dlg_env():
    """Dialogue RMS envelope in dB at 100 Hz (None if the stem is missing)."""
    global _DLG
    if _DLG is None:
        try:
            x, sr = sf.read(str(AUD / 'dialogue.wav'), dtype='float32', always_2d=True)
            m = x.mean(1)
            hop = sr // 100
            n = len(m) // hop
            _DLG = 20 * np.log10(np.sqrt((m[:n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9)
        except Exception:
            _DLG = False
    return _DLG


def dialogue_gaps(t0, t1, min_len=0.14):
    e = dlg_env()
    if e is False:
        return []
    i0, i1 = int(t0 * 100), int(t1 * 100)
    seg = e[i0:i1]
    q = seg < seg.max() - 26
    out, k = [], 0
    while k < len(q):
        if q[k]:
            j = k
            while j < len(q) and q[j]:
                j += 1
            if (j - k) / 100 >= min_len:
                out.append((t0 + k / 100, t0 + j / 100))
            k = j
        else:
            k += 1
    return out


def syllable_onsets(t0, t1, n=4):
    e = dlg_env()
    if e is False:
        return list(np.linspace(t0, t1 - 0.3, n))
    i0 = int(t0 * 100) - 5
    seg = e[i0:int(t1 * 100) + 5]
    on = seg > seg.max() - 14
    ons = []
    for k in range(1, len(on)):
        if on[k] and not on[k - 1]:
            t = (i0 + k) / 100
            if not ons or t - ons[-1] > 0.18:
                ons.append(t)
    if len(ons) < n:
        return list(np.linspace(t0, t1 - 0.3, n))
    return ons[:n]


# ============================================================================ the score
DMAJ, DPENT = scale_pcs('D'), scale_pcs('D', 'pent')
EMAJ, EPENT = scale_pcs('E'), scale_pcs('E', 'pent')


def setup(S):
    T = S.track
    T('piano', 0, 'keys', pan=-0.12, send=0.30, gain=-1, hp=35, lead=True, duck=-6, cal=60)
    T('celesta', 8, 'keys', pan=0.30, send=0.50, space=0.12, gain=-4, hp=250, lead=True, duck=-6, cal=84)
    T('glock', 9, 'keys', pan=0.38, send=0.50, space=0.12, gain=-8, hp=400, lead=True, duck=-7, cal=88)
    T('mbox', 10, 'keys', pan=0.10, send=0.46, space=0.10, gain=-4, hp=300, lead=True, duck=-5, cal=84)
    T('marimba', 12, 'keys', pan=-0.30, send=0.22, gain=-4, hp=80, duck=-4, cal=72)
    T('xylo', 13, 'keys', pan=0.30, send=0.26, gain=-7, hp=200, lead=True, duck=-5, cal=84)
    T('harp', 46, 'keys', pan=-0.38, send=0.42, space=0.05, gain=-2, hp=40, duck=-3, cal=60)
    T('spad_hi', 48, 'strings', pan=-0.22, send=0.40, space=0.05, gain=0, hp=120, shelf=(6000, -3), duck=-3, cal=67)
    T('spad_lo', 48, 'strings', pan=0.25, send=0.34, gain=0, hp=35, duck=-2, cal=48)
    T('vln', 48, 'strings', pan=-0.30, send=0.40, space=0.05, gain=0, hp=150, shelf=(6000, -3), lead=True, duck=-6, cal=76)
    T('cello', 42, 'strings', pan=0.28, send=0.34, gain=-2, hp=40, lead=True, duck=-4, cal=50)
    T('basses', 43, 'strings', pan=0.15, send=0.28, gain=-1, hp=25, lp=2500, duck=-1, cal=36)
    T('trem', 44, 'strings', pan=0.0, send=0.42, gain=-3, hp=60, duck=-3, cal=60)
    T('pizz', 45, 'strings', pan=-0.08, send=0.28, gain=-2, hp=40, duck=-3, cal=55)
    T('wobble', 48, 'strings', pan=0.12, send=0.40, gain=0, hp=100, duck=-3, cal=64)
    T('ostin', 48, 'strings', pan=0.05, send=0.30, gain=0, hp=50, duck=-2, cal=50)
    T('lowsus', 43, 'strings', pan=0.10, send=0.35, gain=0, hp=20, duck=0, gate=False, cal=36)
    T('choir', 52, 'choir', pan=0.0, send=0.50, space=0.25, gain=-2, hp=90, duck=-4, cal=62)
    T('oohs', 53, 'choir', pan=0.05, send=0.55, space=0.30, gain=-2, hp=90, duck=-4, cal=62)
    T('horns', 60, 'brass_winds', pan=-0.20, send=0.45, gain=0, hp=60, lead=True, duck=-5, cal=55)
    T('brass', 61, 'brass_winds', pan=0.15, send=0.40, gain=-2, hp=50, shelf=(5000, -4), duck=-3, cal=60)
    T('trombone', 57, 'brass_winds', pan=0.30, send=0.35, gain=-3, hp=50, duck=-3, cal=50)
    T('tuba', 58, 'brass_winds', pan=0.20, send=0.30, gain=-2, hp=25, duck=-1, cal=36)
    T('flute', 73, 'brass_winds', pan=0.20, send=0.40, gain=-5, hp=250, lead=True, duck=-6, cal=79)
    T('bassoon', 70, 'brass_winds', pan=0.10, send=0.30, gain=-3, hp=50, duck=-3, cal=48)
    T('timp', 47, 'perc', pan=-0.05, send=0.38, gain=-1, hp=30, duck=-1, cal=45)
    T('kit', 0, 'perc', pan=0.0, send=0.35, gain=-8, drum=True, hp=60, duck=-2, cal=49)
    T('tbells', 14, 'perc', pan=0.20, send=0.50, space=0.10, gain=-7, hp=150, duck=-4, cal=72)


def harp_bar(S, sym, t, spb, v, beats=3, lo=('D2', 'C#3')):
    b = bass_of(sym, *lo)
    up = voicing(sym, b + 7, b + 20, 2)
    for j, p in enumerate([b] + up):
        if j < beats:
            S.n('harp', p, t + j * spb, spb * 2.2, v - 4 * j)


def s01(S):
    st, ti, to, tilt = c('s01_stars_in'), c('title_in'), c('title_out'), c('tilt_start')
    end = SCENE['s01'][1]
    S.section('s01', 0.0, end, 'D major', 'hushed stars + shimmer; soft title swell (oohs sing the Whale incipit 1-5); '
              'Lamp theme on music box from the tilt')
    rng = np.random.default_rng(101)
    pool = [P(x) for x in ('D6', 'E6', 'F#6', 'A6', 'B6', 'D7', 'E7', 'F#7', 'A7')]
    tw = list(st + 0.15 + (to - st - 0.3) * np.sort(rng.random(15)) ** 1.25)
    late = np.arange(to + 1.6, end - 1.5, 2.4)
    tw += list(late + rng.uniform(-0.5, 0.5, len(late)))
    for t in tw:
        p = int(rng.choice(pool))
        v = rng.uniform(28, 54) * (0.8 if t > to else 1.0)
        S.n('celesta', p, t, 1.4, v)
        if rng.random() < 0.3:
            S.n('glock', p, t + 0.004, 1.2, v - 14)
    S.hit('s01_stars_in', st, 'texture', 'celesta/glock star twinkles + high shimmer fade in from silence')
    # --- title swell
    spb = (c('s02_climb') - tilt) / 18.0          # Lamp theme: pickup + 5 bars of 3/4 + 2 beats -> pickup lands on s02
    b1 = tilt + spb
    sw = [(ti, ti + 1.9, 'Dadd9'), (ti + 1.9, ti + 3.5, 'G/D'), (ti + 3.5, b1, 'Dadd9')]
    pad_strings(S, sw, v_hi=54, v_lo=50, hi=('A3', 'A4'), n_hi=4)
    for p, t, d, v in (('D4', ti, 1.95, 52), ('A3', ti, 1.95, 44), ('A4', ti + 1.9, 1.65, 56), ('D4', ti + 1.9, 1.65, 46),
                       ('F#4', ti + 3.5, b1 - ti - 3.2, 50), ('A3', ti + 3.5, b1 - ti - 3.2, 42)):
        S.n('oohs', p, t, d, v)
    for tr in ('spad_hi', 'spad_lo', 'oohs'):
        S.bump(tr, [(ti, -20), (ti + 2.7, 0), (to, -7), (b1, -10), (b1 + 1.4, 0)])
    S.ch('harp', ['D2', 'A2', 'D3', 'A3', 'D4', 'E4', 'F#4', 'A4'], ti, 3.0, 44, roll=0.09)
    S.hit('title_in', ti, 'swell', 'strings + oohs swell (Dadd9 - G/D - Dadd9), harp roll as the title assembles')
    gliss(S, 'celesta', 'A5', 'A7', to + 0.75, 0.75, 50, 26, DPENT, ring=1.3)
    S.hit('title_out', to, 'flourish', 'celesta arpeggio drifting up as the letters dissolve into stardust')
    S.motif('whale', ti, 'oohs', 'incipit D-A only (foreshadowing)')
    # --- Lamp theme, music box, one beat per second
    sq = seq(LAMP_A.replace('| E5:3', '| E5:2 A4:1'), tr=12)
    S.mel('mbox', sq, tilt, spb, 60)
    S.motif('lamp', tilt, 'music box', 'phrase A, begins as the camera tilts down')
    S.hit('tilt_start', tilt, 'motif', 'Lamp theme begins (music box pickup A5)')
    bar = 3 * spb
    segs = []
    for k, sym in enumerate(LAMP_A_CH):
        t0 = b1 + k * bar
        t1 = b1 + (k + 1) * bar if k < 5 else c('s02_climb') + (c('s02_arrive') - c('s02_climb')) / 8
        segs.append((t0, t1, sym))
        harp_bar(S, sym, t0, spb, 44)
    pad_strings(S, segs, v_hi=38, v_lo=30, hi=('F#3', 'F#4'), n_hi=3)
    for k, (t0, t1, sym) in enumerate(segs[2:]):
        S.n('cello', bass_of(sym, 'A2', 'G#3'), t0, t1 - t0 + 0.05, 34 + 2 * k)


def s02(S):
    t0, arr, ign = c('s02_climb'), c('s02_arrive'), c('lamp_ignite')
    tl0, tl1, sit = c('timelapse_start'), c('timelapse_end'), c('s02_sit')
    d02 = LINE['D02']
    end = SCENE['s02'][1]
    S.section('s02', t0, end, 'D major', 'cosy climb (piano on the footsteps), lamp-ignite flourish, flowing time-lapse '
              'arpeggios, lonely music box')
    spb = (arr - t0) / 8.0                      # one beat per footstep (walk cadence 1 cycle/s)
    b1 = t0 + spb
    S.mel('piano', seq("F#5:2 A5:1 | G5:2 F#5:1 | E5:1 D5:2"), b1, spb, 54)
    S.motif('lamp', b1, 'piano', 'phrase B head, one beat per footstep; D5 lands on s02_arrive')
    for k, ps in enumerate((['D3', 'A3', 'F#4'], ['G2', 'D3', 'B3'], ['E3', 'B3', 'G4'])):
        for j, p in enumerate(ps):
            S.n('piano', p, b1 + (3 * k + j) * spb, spb * 1.8, 40 - 4 * j)
        S.n('pizz', ps[0] if P(ps[0]) < P('D3') else P(ps[0]) - 12, b1 + 3 * k * spb, 0.4, 44)
    for p, v in (('G2', 40), ('D3', 36), ('B3', 33)):
        S.n('piano', p, arr, 1.8, v)
    pad_strings(S, [(b1, b1 + 3 * spb, 'D'), (b1 + 3 * spb, b1 + 6 * spb, 'Gmaj7'), (b1 + 6 * spb, arr, 'Em7')],
                v_hi=34, v_lo=34, hi=('F#3', 'F#4'), n_hi=3)
    S.hit('s02_arrive', arr, 'landing', 'piano melody lands on D5 (lamp room)')
    # under D01
    ud = [(arr, arr + 1.3, 'G'), (arr + 1.3, arr + 2.5, 'D/F#'), (arr + 2.5, arr + 3.7, 'Em7'),
          (arr + 3.7, ign - 0.7, 'Asus4'), (ign - 0.7, ign, 'A')]
    pad_strings(S, ud, v_hi=36, v_lo=38, hi=('F#3', 'F#4'), n_hi=3)
    for a, b, sym in ud[:4]:
        harp_bar(S, sym, a + 0.05, 0.3, 32)
    # --- the lamp ignites
    gliss(S, 'harp', 'D4', 'D6', ign, 0.45, 34, 60, DMAJ, ring=1.5)
    S.ch('celesta', ['D6', 'F#6', 'A6', 'D7'], ign, 1.8, 60, roll=0.065)
    S.n('glock', 'A6', ign, 1.5, 58)
    S.n('glock', 'D7', ign + 0.2, 1.5, 50)
    S.ch('harp', ['D2', 'A2', 'D3', 'F#3', 'A3'], ign, 3.0, 60, roll=0.025)
    S.n('tbells', 'D5', ign, 3.0, 44)
    pad_strings(S, [(ign, tl0, 'Dadd9')], v_hi=50, v_lo=46, hi=('F#3', 'A4'), n_hi=4, choir='oohs', v_ch=40)
    for tr in ('spad_hi', 'spad_lo', 'oohs'):
        S.bump(tr, [(ign - 0.35, 0), (ign - 0.01, -8), (ign + 0.35, 0), (ign + 1.4, -5), (tl0 - 0.2, -5), (tl0, 0)])
    S.fxa('revcym', t=ign, lvl=-27, length=1.3, seed=12)
    S.fxa('ping', t=ign, lvl=-25, midi='D7', seed=13)
    S.hit('lamp_ignite', ign, 'flourish', 'harp gliss lands on a celesta/glock D-major bloom, tubular bell, soft cymbal swell, sparkle ping')
    arp(S, 'harp', ['D3', 'A3', 'D4', 'E4', 'F#4', 'A4', 'D5'], ign + 0.6, tl0 - 0.1, 0.28, 36, ring=2.0, seed=3)
    # --- time-lapse: eight chords, flowing sextuplets, descending bass
    chs = ['D', 'A/C#', 'Bm', 'F#m/A', 'G', 'D/F#', 'Em7', 'A7sus4']
    cd = (tl1 - tl0) / len(chs)
    segs = [(tl0 + i * cd, tl0 + (i + 1) * cd, s) for i, s in enumerate(chs)]
    pad_strings(S, segs, v_hi=40, v_lo=46, hi=('F#3', 'F#4'), n_hi=3, basses=36, lo_rng=('D2', 'C#3'))
    for i, (a, b, sym) in enumerate(segs):
        tones = voicing(sym, 'D3', 'C5', 4)
        arp(S, 'harp', tones, a, b, cd / 6, 46, pattern='updown', ring=2.2, seed=10 + i, accent=6)
        arp(S, 'piano', [p + 12 for p in voicing(sym, 'F#4', 'E5', 3)], a + cd / 12, b, cd / 3, 32, ring=1.4, seed=20 + i)
        top = max(p for p in range(P('D6'), P('C#7')) if p % 12 in chord_pcs(sym))
        S.n('celesta', top, a + cd * 0.5, 1.2, 30)
    S.hit('timelapse_start', tl0, 'texture', 'flowing harp sextuplets + piano triplets over a descending bass, one chord per 0.875 s')
    # ease back to real time
    pad_strings(S, [(tl1, sit, 'A7')], v_hi=34, v_lo=36, hi=('E3', 'E4'), n_hi=3)
    for p, dt in (('A2', 0.0), ('E3', 0.22), ('G3', 0.48), ('C#4', 0.78)):
        S.n('harp', p, tl1 + dt, 2.0, 40 - 12 * dt)
    S.hit('timelapse_end', tl1, 'texture', 'arpeggios slow down (ritardando) back to real time')
    # --- alone: sparse music box (notes placed in the gaps of D02), low piano
    gaps = [g for g in dialogue_gaps(d02[0] + 0.3, d02[1] - 0.2) if g[1] - g[0] > 0.14]
    t_e6 = gaps[0][0] + 0.02 if gaps else d02[0] + 1.3
    t_d6 = gaps[1][0] + 0.02 if len(gaps) > 1 else d02[0] + 2.6
    t_b5, t_a5 = d02[1] + 0.25, d02[1] + 1.0
    mb = [('A5', sit + 0.15, 1.2, 50), ('F#6', sit + 0.62, 1.5, 54), ('E6', t_e6, 1.2, 50), ('D6', t_d6, 1.4, 50),
          ('B5', t_b5, 0.9, 48), ('A5', t_a5, 2.4, 50)]
    for p, t, d, v in mb:
        S.n('mbox', p, t, d, v)
    S.motif('lamp', sit, 'music box', 'lonely, rubato; notes sit in the breaths of D02')
    hs = [(sit, t_e6, 'Bm7'), (t_e6, t_d6, 'Em9'), (t_d6, t_a5, 'Gmaj7'), (t_a5, end + 0.8, 'Gmaj9')]
    for a, b, sym in hs:
        S.harm(a, sym)
        bb = bass_of(sym, 'E1', 'D#2')
        ps = [bb, bb + 7] + voicing(sym, bb + 12, bb + 26, 2)
        S.ch('piano', ps, a, b - a + 0.4, 36, roll=0.05)
    S.n('piano', 'E5', d02[1] + 1.7, 0.5, 34)
    S.n('piano', 'D5', d02[1] + 2.05, 0.5, 32)
    S.n('piano', 'B4', d02[1] + 2.4, 1.8, 30)
    S.hit('s02_sit', sit, 'thin', 'texture drops to solo music box + low piano')


def s03(S):
    st, ms, imp, rd, bo = c('s03_start'), c('meteor_start'), c('impact'), c('run_down'), c('boat_out')
    end = SCENE['s03'][1]
    S.section('s03', st, end, 'B minor -> D major', 'mystery (tremolo, harp harmonics, odd celesta twinkle); '
              'meteor riser; impact hit; bioluminescent sparkle; running figure; rowing barcarolle')
    S.harm(st, 'Bmadd9')
    sustain(S, 'trem', [(st, ms, ['B2', 'F#3', 'C#4', 'D4'])], 36)
    for p, dt, v in (('F#6', 0.4, 34), ('C#7', 1.3, 30), ('B6', 2.0, 32)):
        S.n('harp', p, st + dt, 2.0, v)
    for p, dt in (('C7', 0.9), ('F#6', 0.99), ('F#6', 1.7), ('C7', 1.78), ('C7', 2.25)):
        S.n('celesta', p, st + dt, 0.5, 40)
    S.hit('s03_start', st, 'mystery', 'tremolo strings Bm(add9), harp harmonics, a tritone "odd twinkle" on celesta')
    # --- meteor: rising line, glissandi, riser
    S.harm(ms, 'Bm')
    L = imp - ms
    for p in ('B2', 'F#3', 'D4'):
        S.n('trem', p, ms, L + 0.05, 42)
    for p, x, v in (('F#4', 0.0, 40), ('G4', 0.2, 46), ('G#4', 0.4, 52), ('A4', 0.58, 58), ('A#4', 0.74, 66), ('B4', 0.88, 74)):
        nx = {0.0: 0.2, 0.2: 0.4, 0.4: 0.58, 0.58: 0.74, 0.74: 0.88, 0.88: 1.0}[x]
        S.n('trem', p, ms + x * L, (nx - x) * L + 0.03, v)
    for te, du, lo, hi, v0, v1 in ((ms + 0.8, 0.8, 'B3', 'B5', 30, 44), (ms + 2.3, 0.9, 'D4', 'D6', 30, 42),
                                   (ms + 3.3, 0.7, 'F#4', 'F#6', 40, 58), (imp, 0.6, 'B4', 'B6', 50, 76)):
        gliss(S, 'harp', lo, hi, te, du, v0, v1, DMAJ, ring=1.4)
    timp_roll(S, 'F#2', imp - 1.8, imp - 0.02, 36, 100)
    S.n('horns', 'F#3', imp - 2.0, 2.0, 66)
    S.bump('horns', [(imp - 2.0, -16), (imp - 0.05, 0)])
    S.fxa('riser', t=ms, lvl=-16, length=L, f0=220, f1=8000, root=P('B3'), seed=31)
    S.fxa('revcym', t=imp, lvl=-13, length=2.2, seed=32)
    S.hit('meteor_start', ms, 'riser', 'chromatic tremolo climb, four widening harp glissandi, noise/tone riser, timpani roll')
    # --- impact
    S.harm(imp, 'Dadd9')
    S.fxa('boom', t=imp, lvl=-3, decay=1.9, seed=33)
    S.fxa('crash', t=imp, lvl=-12, decay=3.0, seed=34)
    S.fxa('bells', t=imp, lvl=-15, midis=['D5', 'A5', 'E6', 'F#6'], decay=3.2, seed=35)
    S.n('kit', 49, imp, 2.0, 100)
    S.n('kit', 57, imp + 0.01, 2.0, 80)
    S.n('timp', 'D2', imp, 2.0, 118)
    S.n('timp', 'A1', imp + 0.005, 2.0, 95)
    S.ch('tbells', ['D5', 'A5'], imp, 4.0, 92)
    S.ch('glock', ['D6', 'E6', 'F#6', 'A6'], imp, 2.5, 82, roll=0.012)
    S.ch('celesta', ['D6', 'F#6', 'A6', 'E7'], imp, 2.5, 80, roll=0.01)
    S.ch('brass', ['D3', 'A3', 'D4', 'F#4'], imp, 2.4, 96)
    S.n('tuba', 'D2', imp, 2.5, 96)
    S.ch('horns', ['A3', 'D4', 'F#4'], imp, 2.6, 92)
    S.ch('ostin', ['D2', 'A2', 'D3', 'F#3', 'A3', 'D4', 'E4', 'F#4', 'A4'], imp, 1.8, 104)
    S.n('basses', 'D1', imp, 2.4, 100)
    S.ch('choir', ['A3', 'D4', 'F#4', 'A4'], imp, 2.6, 86)
    S.ch('harp', ['D1', 'A1', 'D2'], imp, 3.0, 92)
    for tr in ('brass', 'tuba', 'horns', 'choir', 'basses'):
        S.bump(tr, [(imp + 0.3, 0), (imp + 2.4, -14), (imp + 3.5, -14)])
    S.bump('ostin', [(imp + 0.2, 0), (imp + 1.8, -12), (imp + 2.8, -12)])
    S.hit('impact', imp, 'BIG HIT', 'sub boom + cymbal (swell into crash) + timpani + tubular/glock/celesta/synth bell cluster '
          '+ brass/strings/choir D(add9) chord')
    after = [(imp, imp + 1.6, 'Dadd9'), (imp + 1.6, rd, 'Gmaj7#11')]
    pad_strings(S, after, v_hi=44, v_lo=42, hi=('F#3', 'A4'), n_hi=4)
    for k, (dt, nn, v0, v1) in enumerate(((0.0, 14, 74, 50), (0.75, 11, 60, 40), (1.6, 9, 50, 34), (2.5, 7, 42, 30))):
        sparkle_burst(S, imp + dt + 0.02, 0.7, nn, DPENT, 'A5', 'A7', v0, v1, seed=40 + k)
    S.hit('impact+bio', imp + 0.75, 'texture', 'expanding rings of celesta/glock/harp sparkles (bioluminescence)')
    # --- run down the stairs: one pizz per hurried footstep, descending xylophone runs
    p = 1.0 / (2 * 2.2)
    chs = ['Bm', 'G', 'Em', 'F#7']
    k = 0
    while rd + k * p < bo - 0.1:
        t = rd + k * p
        sym = chs[min(3, k // 4)]
        if k % 4 == 0:
            S.harm(t, sym)
            S.ch('ostin', voicing(sym, 'D3', 'D4', 3), t, 0.25, 62)
        r = bass_of(sym, 'E2', 'D#3')
        S.n('pizz', r if k % 2 == 0 else r + 7, t, 0.2, 80 if k % 4 == 0 else 64)
        if k % 2 == 0:
            S.n('kit', 76 if k % 4 == 0 else 77, t, 0.1, 30)
        k += 1
    starts = ['B6', 'G6', 'E6']
    for g in range(3):
        t = rd + g * 4 * p
        sc = [q for q in range(P('B4'), P(starts[g]) + 1) if q % 12 in DMAJ][::-1][:8]
        for j, q in enumerate(sc):
            S.n('xylo', q, t + j * p / 2, 0.18, 72 - j * 2)
            if j % 2 == 0:
                S.n('flute', q - 12, t + j * p / 2, p * 0.45, 46)
    for j, q in enumerate(('F#5', 'A#5', 'C#6', 'F#6')):
        S.n('xylo', q, rd + 12 * p + j * p / 2, 0.2, 64 + 4 * j)
    S.hit('run_down', rd, 'figure', 'pizzicato on every hurried footstep (2.2 cycles/s) + descending xylophone/flute runs (stairs)')
    # --- rowing barcarolle: one 6/8 bar per oar stroke
    bar = 1.6
    e8 = bar / 6
    chs = ['Bmadd9', 'Gmaj7', 'Em9', 'F#7sus4', 'Bmadd9']
    k = 0
    t = bo
    while t < c('lift') + 0.2:
        sym = chs[k % len(chs)]
        S.harm(t, sym)
        b = bass_of(sym, 'E2', 'D#3')
        up = voicing(sym, b + 14, b + 26, 2)
        pat = [b, b + 7, b + 12, up[0], up[1], b + 12]
        last = t + bar > c('lift')
        for j, q in enumerate(pat[:4] if last else pat):
            S.n('harp', q, t + j * e8, e8 * 2.5, [54, 38, 40, 42, 44, 38][j])
        S.ch('spad_lo', [b - 12, b - 5], t, bar + 0.05, 34)
        S.ch('spad_hi', voicing(sym, 'F#3', 'E4', 3), t, bar + 0.05, 30)
        k += 1
        t += bar
    for j, q in enumerate(('F#5', 'B5', 'D6', 'C#6', 'D6', 'F#6')):
        S.n('celesta', q, bo + 2 * bar + j * 0.3, 0.9, 34)
    S.motif('star', bo + 2 * bar, 'celesta', 'foreshadowing, slowed and in B minor (the glow under the water)')
    S.hit('boat_out', bo, 'pulse', 'barcarolle: one 6/8 bar per oar stroke (1.6 s), harp bass on each stroke')


def s04(S):
    st, lift, look, det, rb = c('s04_start'), c('lift'), c('look_sky'), c('determined'), c('row_back')
    g01, d05, g02, g03, d06 = LINE['G01'], LINE['D05'], LINE['G02'], LINE['G03'], LINE['D06']
    end = SCENE['s04'][1]
    S.section('s04', st, end, 'D major', 'tender strings/piano; Star theme born on celesta; worry (borrowed minor); '
              'hopeful horn call; warm row back (Lamp + Star)')
    S.ch('harp', ['D4', 'F#4', 'A4', 'B4', 'D5', 'F#5'], lift, 2.5, 38, roll=0.08)
    S.n('celesta', 'D6', lift + 0.5, 1.5, 30)
    S.hit('lift', lift, 'motif', 'harp up-arpeggio as the dim star is lifted from the water')
    eyes = g02[0] - 0.7
    segs = [(lift, g01[0], 'Gadd9'), (g01[0], g01[0] + 1.2, 'Em9'), (g01[0] + 1.2, d05[0], 'Bm/D'),
            (d05[0], d05[0] + 1.3, 'G'), (d05[0] + 1.3, d05[0] + 2.6, 'D/F#'), (d05[0] + 2.6, d05[0] + 3.9, 'Em7'),
            (d05[0] + 3.9, d05[1] - 0.5, 'A7sus4'), (d05[1] - 0.5, eyes, 'A7')]
    pad_strings(S, segs, v_hi=36, v_lo=38, hi=('G3', 'G4'), n_hi=3)
    for a, b, sym in segs[3:]:
        S.n('cello', bass_of(sym, 'A2', 'G#3'), a, b - a + 0.05, 42)
        harp_bar(S, sym, a + 0.03, 0.3, 30)
    S.n('piano', 'A4', g01[1] + 0.08, 0.5, 36)
    S.n('piano', 'F#5', g01[1] + 0.42, 1.2, 38)
    # --- the Star theme as Guang opens her eyes
    spb = 0.6
    four = "A5:.5 D6:.5 F#6:.5 E6:.25 F#6:.25 | A6:1 F#6:.5 D6:.5 | E6:.5 B5:.5 D6:.5 A5:.5 | B5:.5 C#6:.5 D6:1"
    S.mel('celesta', seq(four), eyes, spb, 54)
    S.mel('glock', seq("A5:.5 D6:.5 F#6:.5 E6:.25 F#6:.25 A6:1"), eyes, spb, 32)
    S.motif('star', eyes, 'celesta (+glock)', 'first statement as Guang opens her eyes')
    S.hit('eyes_open', eyes, 'motif', 'Star theme enters on celesta (0.7 s before G02)')
    sc = [(eyes, eyes + 4 * spb, 'D'), (eyes + 4 * spb, eyes + 5 * spb, 'Em7'), (eyes + 5 * spb, eyes + 7 * spb, 'A7'),
          (eyes + 7 * spb, look, 'D')]
    pad_strings(S, sc, v_hi=30, v_lo=34, hi=('F#3', 'F#4'), n_hi=3)
    for j in range(4):
        tb = eyes + j * 2 * spb
        S.n('harp', ['D3', 'D3', 'E3', 'A2'][j], tb, 1.2, 38)
        S.ch('harp', voicing(['D', 'D', 'Em7', 'A7'][j], 'F#3', 'E4', 3), tb + spb, 0.8, 28)
    # --- the sky is far; worry
    wk = [(look, g03[0] + 1.0, 'Gmaj9'), (g03[0] + 1.0, g03[0] + 2.3, 'Em9'), (g03[0] + 2.3, g03[1] - 0.6, 'Gm6'),
          (g03[1] - 0.6, det, 'Bbmaj7')]
    pad_strings(S, wk, v_hi=42, v_lo=40, hi=('B3', 'D5'), n_hi=4, basses=34, choir='oohs', v_ch=32, ch_rng=('D4', 'D5'))
    S.n('harp', 'G6', look + 0.1, 2.0, 30)
    S.n('harp', 'D7', look + 0.5, 2.0, 28)
    S.hit('look_sky', look, 'swell', 'wide open Gmaj9 (the sky is so far); then Em9 - Gm6 - Bbmaj7 worry under G03')
    # --- determined: horn call on the Lamp head, hopeful progression
    S.n('horns', 'A3', det, 0.32, 70)
    S.n('horns', 'F#4', det + 0.3, 1.3, 74)
    S.n('timp', 'D2', det, 1.0, 55)
    S.ch('harp', ['D2', 'A2', 'D3'], det, 2.0, 50, roll=0.03)
    hp = [(det, d06[0], 'D/A'), (d06[0], d06[0] + 0.8, 'G'), (d06[0] + 0.8, d06[0] + 1.6, 'A'), (d06[0] + 1.6, d06[0] + 2.2, 'Bm'),
          (d06[0] + 2.2, d06[0] + 2.8, 'G'), (d06[0] + 2.8, d06[1], 'A7sus4'), (d06[1], rb, 'A')]
    pad_strings(S, hp, v_hi=46, v_lo=46, hi=('F#3', 'A4'), n_hi=4, basses=40)
    S.motif('lamp', det, 'horn', 'head only (A-F#), a small heroic call')
    S.hit('determined', det, 'lift', 'horn call on the Lamp-theme head, timpani, D major, hopeful progression under D06')
    # --- row back: barcarolle in D, Lamp theme on violins, Star twinkle on celesta
    bar = 1.6
    e8 = bar / 6
    rows = [(rb, 'D'), (rb + bar, 'G'), (rb + 2 * bar, 'A7')]
    for t, sym in rows:
        S.harm(t, sym)
        b = bass_of(sym, 'D2', 'C#3')
        up = voicing(sym, b + 14, b + 26, 2)
        for j, q in enumerate([b, b + 7, b + 12, up[0], up[1], b + 12]):
            if t + j * e8 < end - 0.05:
                S.n('harp', q, t + j * e8, e8 * 2.5, [50, 36, 38, 40, 42, 36][j])
        S.n('pizz', b, t, 0.5, 54)
    pad_strings(S, [(rb, rb + bar, 'D'), (rb + bar, rb + 2 * bar, 'G'), (rb + 2 * bar, end, 'A7')],
                v_hi=42, v_lo=42, hi=('F#3', 'F#4'), n_hi=3, harm=False)
    S.n('vln', 'A4', rb - 0.3, 0.3, 52)
    for p, dt, d in (('F#5', 0.0, 1.0), ('E5', 1.07, 0.5), ('D5', 1.6, 1.0), ('B4', 2.67, 0.5), ('A4', 3.2, end - rb - 3.2 + 0.2)):
        S.n('vln', p, rb + dt, d, 56)
    for p, dt in (('A5', 0), ('D6', 0.2), ('F#6', 0.4), ('E6', 0.6), ('F#6', 0.7), ('A6', 0.8)):
        S.n('celesta', p, rb + bar + dt, 1.0, 44)
    S.motif('lamp', rb, 'violins', 'warm, in 6/8 over the rowing')
    S.motif('star', rb + bar, 'celesta', 'answering twinkle (two lights on the water)')
    S.hit('row_back', rb, 'pulse', 'warm barcarolle, one bar per oar stroke, Lamp theme on violins')


def groove(S, t0, t1, beat, chords, v=1.0, tamb=False, seed=0):
    """Light pizz / marimba / maracas groove; chords change every 2 beats."""
    rng = np.random.default_rng(seed)
    t, k = t0, 0
    while t < t1 - 1e-6:
        sym = chords[(k // 2) % len(chords)]
        root = bass_of(sym, 'D2', 'C#3')
        fifth = root + 7 if root + 7 <= P('A2') else root - 5
        if k % 2 == 0:
            S.harm(t, sym)
        S.n('pizz', root if k % 2 == 0 else fifth, t, 0.3, (72 if k % 2 == 0 else 60) * v)
        S.ch('marimba', voicing(sym, 'D4', 'D5', 3), t + beat / 2, 0.22, (42 + rng.integers(-3, 4)) * v)
        S.n('kit', 70, t, 0.1, 22 * v)
        S.n('kit', 70, t + beat / 2, 0.1, 32 * v)
        if tamb and k % 2 == 1:
            S.n('kit', 54, t, 0.2, 30 * v)
        t += beat
        k += 1


def s05(S):
    st, th, ca, stk, clm = c('s05_start'), c('throw1'), c('catch'), c('stack'), c('climb')
    tf, spl, sit, dw, idea = c('tower_fall'), c('splash'), c('sit_together'), c('dawn_warning'), c('idea')
    d07, g04, d08, g06, d10, g07 = LINE['D07'], LINE['G04'], LINE['D08'], LINE['G06'], LINE['D10'], LINE['G07']
    end = SCENE['s05'][1]
    beat = 0.5
    S.section('s05', st, end, 'D major', 'playful pizz/marimba/glock with comic hits; tender Lamp+Star duet; '
              'B-minor worry at dawn; bright idea sting')
    # --- bouncy intro + counting
    S.harm(st, 'D')
    S.ch('pizz', ['D2', 'D3'], st, 0.3, 66)
    S.ch('marimba', ['D4', 'F#4', 'A4'], st + 0.02, 0.3, 48)
    S.n('pizz', 'A2', st + beat, 0.3, 58)
    S.ch('marimba', ['C#4', 'E4', 'A4'], st + 1.5 * beat, 0.2, 38)
    ons = syllable_onsets(d07[0], d07[1], 4)
    for t, (sym, pz) in zip(ons[:3], (('A', 'A2'), ('Bm', 'B2'), ('A/C#', 'C#3'))):
        S.harm(t, sym)
        S.ch('pizz', [pz, P(pz) + 12], t, 0.3, 60)
        S.ch('marimba', voicing(sym, 'C#4', 'C#5', 3), t, 0.25, 46)
    timp_roll(S, 'A1', ons[2] + 0.1, th - 0.03, 34, 76, 14, 20)
    S.hit('D07_counts', ons[0], 'mickey', 'pizz + marimba stabs on "yi / er / san" (detected in the dialogue), timpani roll to the throw')
    # --- throw: upward run
    S.harm(th, 'Dadd9')
    gliss(S, 'harp', 'D4', 'D7', th + 0.75, 0.75, 48, 78, DMAJ, ring=1.2)
    sc = [q for q in range(P('D5'), P('A6') + 1) if q % 12 in DMAJ]
    for j, q in enumerate(sc):
        S.n('xylo', q, th + 0.65 * j / (len(sc) - 1), 0.2, 60 + 20 * j / len(sc))
    sc = [q for q in range(P('D5'), P('D6') + 1) if q % 12 in DMAJ]
    for j, q in enumerate(sc):
        S.n('flute', q, th + 0.6 * j / (len(sc) - 1), 0.12, 60)
    S.n('pizz', 'D2', th, 0.4, 88)
    S.ch('marimba', ['D4', 'F#4', 'A4', 'D5'], th, 0.4, 62)
    S.fxa('slide', t=th, lvl=-20, dur=0.9, fa=620, fb=1750, seed=51)
    S.fxa('whoosh', t=th, lvl=-25, dur=0.8, seed=52)
    S.hit('throw1', th, 'run', 'upward harp gliss + xylophone/flute scale + slide whistle + whoosh')
    top = th + (ca - th) * 0.47
    for i in range(8):
        S.n('glock', 'A6' if i % 2 == 0 else 'B6', top + i * 0.06, 0.12, 46)
    for j, q in enumerate(('D6', 'F#6', 'A6', 'D7', 'A6', 'F#6')):
        S.n('celesta', q, top + 0.05 + j * 0.07, 0.5, 44)
    S.n('vln', 'A5', top - 0.1, 0.7, 30)
    sc = [q for q in range(P('D5'), P('A6') + 1) if q % 12 in DMAJ][::-1][:10]
    for j, q in enumerate(sc):
        S.n('xylo', q, ca - 0.52 + 0.48 * j / 9, 0.15, 64 - j)
    S.fxa('slide', t=ca - 0.5, lvl=-25, dur=0.45, fa=1500, fb=820, seed=53)
    # --- catch
    S.fxa('boing', t=ca, lvl=-15, f0=196)
    S.n('timp', 'D2', ca, 0.8, 84)
    S.ch('pizz', ['D2', 'A2'], ca, 0.4, 90)
    S.n('kit', 76, ca, 0.2, 80)
    S.n('glock', 'D6', ca, 0.8, 60)
    S.ch('marimba', ['D4', 'F#4', 'A4'], ca, 0.5, 68)
    S.hit('catch', ca, 'accent', 'boing + timpani + pizz + woodblock + glock (lands on his head)')
    # --- groove 1 (G04 / D08)
    groove(S, ca + beat, stk - 0.05, beat, ['D', 'Bm', 'G', 'A'], seed=1)
    for p, dt in (('A5', 0), ('D6', 0.12), ('F#6', 0.24), ('E6', 0.36), ('F#6', 0.42), ('A6', 0.48)):
        S.n('glock', p, g04[1] + 0.06 + dt, 0.6, 50)
    S.motif('star', g04[1] + 0.06, 'glock', 'quick giggle answer')
    # --- stack: three escalating steps (D, E, F#)
    steps = [stk + k * (clm - stk) / 3 for k in range(3)]
    for k, (t, sym) in enumerate(zip(steps, ['D', 'E', 'F#'])):
        r = P('D4') + [0, 2, 4][k]
        S.harm(t, sym)
        for j, q in enumerate([r, r + 4, r + 7]):
            S.n('xylo', q + 12, t - 0.21 + j * 0.07, 0.15, 60 + 5 * k)
        S.ch('brass', [r - 12, r - 5, r, r + 4], t, 0.25, 70 + 8 * k)
        S.n('pizz', r - 24, t, 0.3, 80 + 5 * k)
        S.n('pizz', r - 12, t, 0.3, 70)
        S.n('kit', 77, t, 0.2, 70)
        S.n('timp', r - 24, t, 0.5, 60 + 8 * k)
        tn = steps[k + 1] if k < 2 else clm
        arp(S, 'marimba', voicing(sym, 'D4', 'D5', 3), t + 0.25, tn - 0.1, 0.25, 40 + 4 * k, ring=0.8, seed=60 + k)
        S.hit('stack_%d' % (k + 1), t, 'escalation', 'rising stab #%d (%s major): xylo pickup, brass, pizz, woodblock, timpani' % (k + 1, sym))
    # --- climb: wobbling dominant
    S.harm(clm, 'A7')
    for q in voicing('A7', 'A3', 'G4', 4):
        S.n('wobble', q, clm, tf - clm + 0.06, 58)
    ts = np.arange(clm, tf, 0.01)
    x = (ts - clm) / (tf - clm)
    ph = 2 * np.pi * np.cumsum((3.0 + 4.5 * x) * 0.01)
    S.bend('wobble', list(zip(ts, (0.06 + 0.5 * x ** 1.5) * np.sin(ph))) + [(tf + 0.03, 0.0)])
    S.bump('wobble', [(clm, -10), (tf - 0.2, 0), (tf + 0.4, 0)])
    t = clm
    i = 0
    while t < tf - 0.05:
        S.n('glock', 'E6' if i % 2 == 0 else 'A6', t, 0.1, 34 + 22 * (t - clm) / (tf - clm))
        t += 0.09
        i += 1
    timp_roll(S, 'A1', clm + 1.0, tf - 0.02, 36, 92)
    for j, q in enumerate(('E5', 'F5', 'F#5', 'G5', 'G#5')):
        S.n('vln', q, clm + j * (tf - clm) / 5, (tf - clm) / 5 + 0.02, 48 + 5 * j)
    t = clm
    i = 0
    while t < tf - 0.1:
        S.n('pizz', 'A4' if i % 2 == 0 else 'E5', t, 0.15, 40)
        t += 0.25
        i += 1
    S.hit('climb', clm, 'tension', 'A7 strings with a growing pitch wobble, glock tremolo, chromatic violins, timpani roll')
    # --- tower falls: descending comic slide
    S.fxa('slide', t=tf, lvl=-15, dur=spl - tf - 0.03, fa=1900, fb=330, seed=54)
    S.n('trombone', 'A3', tf, spl - tf + 0.05, 84)
    tb = np.arange(tf, spl, 0.01)
    S.bend('trombone', [(tf - 0.01, 0.0)] + [(t, -12 * ((t - tf) / (spl - tf)) ** 1.3) for t in tb] + [(spl + 0.2, 0.0)])
    gliss(S, 'harp', 'A3', 'A6', spl - 0.05, spl - 0.05 - tf, 70, 50, DMAJ, down=True)
    S.hit('tower_fall', tf, 'slide', 'slide whistle + trombone glissando down an octave + harp gliss down')
    # --- splash
    S.harm(spl, 'D')
    S.n('kit', 49, spl, 2.5, 112)
    S.n('kit', 57, spl + 0.012, 2.5, 92)
    S.fxa('crash', t=spl, lvl=-14, decay=2.2, seed=55)
    S.fxa('boom', t=spl, lvl=-15, f_hi=90, f_lo=40, decay=0.6, dur=3.0, seed=56)
    S.n('timp', 'D2', spl, 1.5, 116)
    S.n('timp', 'A1', spl + 0.005, 1.5, 96)
    S.ch('pizz', ['D2', 'D3'], spl, 0.5, 110)
    S.ch('brass', ['D3', 'F#3', 'A3', 'D4'], spl, 0.4, 104)
    S.n('tuba', 'D2', spl, 0.5, 100)
    S.ch('marimba', ['D4', 'F#4', 'A4', 'D5'], spl, 0.5, 78)
    sparkle_burst(S, spl + 0.05, 0.9, 12, DPENT, 'A5', 'A7', 60, 30, seed=57, tracks=('glock', 'celesta', 'marimba'))
    S.n('bassoon', 'D3', spl + 0.6, 0.12, 72)
    S.n('bassoon', 'A2', spl + 0.8, 0.14, 70)
    S.hit('splash', spl, 'crash', 'cymbals + timpani + brass/pizz/tuba stab on D + droplet sparkles + bassoon "blup blup"')
    # --- laughter groove (G05) -> lands on sit_together
    groove(S, spl + 2 * beat, sit - beat, beat, ['D', 'G', 'A'], tamb=True, seed=2)
    S.harm(sit - beat, 'A7')
    S.ch('pizz', ['A2', 'A3'], sit - beat, 0.3, 64)
    S.ch('marimba', ['C#4', 'G4', 'A4'], sit - beat, 0.3, 50)
    # --- sit together: tender duet in the gaps
    segs = [(sit, g06[0], 'Dadd9'), (g06[0], g06[0] + 1.3, 'Gmaj7'), (g06[0] + 1.3, g06[1], 'Em9'), (g06[1], d10[0], 'A7sus4'),
            (d10[0], d10[0] + 1.1, 'D/F#'), (d10[0] + 1.1, d10[0] + 2.2, 'G'), (d10[0] + 2.2, d10[1] - 0.5, 'A7sus4'),
            (d10[1] - 0.5, d10[1], 'A7'), (d10[1], dw, 'D')]
    pad_strings(S, segs, v_hi=36, v_lo=38, hi=('F#3', 'F#4'), n_hi=3)
    S.n('pizz', 'D2', sit, 0.5, 52)
    S.ch('harp', ['D2', 'A2', 'D3', 'F#3', 'A3'], sit, 2.5, 42, roll=0.05)
    for p, dt, d, v in (('A4', 0.05, 0.5, 44), ('F#5', 0.45, 0.6, 46), ('E5', 1.0, 1.5, 42)):
        S.n('piano', p, sit + dt, d, v)
    for p, dt in (('A5', 0.05), ('D6', 0.22), ('F#6', 0.39)):
        S.n('celesta', p, g06[1] + dt, 1.0, 40)
    for p, dt, d in (('F#5', 0.1, 0.5), ('E5', 0.6, 0.5), ('D5', 1.1, dw - d10[1] - 1.1 + 0.2)):
        S.n('vln', p, d10[1] + dt, d, 46)
    for p, dt in (('A6', 0.2), ('F#6', 0.45), ('D6', 0.7), ('E6', 1.2), ('F#6', 1.35)):
        S.n('celesta', p, d10[1] + dt, 0.9, 38)
    S.ch('piano', ['D2', 'A2'], d10[1], 1.6, 40)
    S.motif('lamp+star', sit, 'piano / celesta / violins', 'intertwined in the gaps of G06 and D10')
    S.hit('sit_together', sit, 'landing', 'groove cadences (A7-D) and dissolves into a soft string pad')
    # --- dawn warning: B minor, texture thins
    dws = [(dw, g07[0] + 0.6, 'Bmadd9'), (g07[0] + 0.6, g07[0] + 2.0, 'Gmaj7'), (g07[0] + 2.0, g07[1] + 0.03, 'Em9'),
           (g07[1] + 0.03, idea - 0.5, 'F#7sus4'), (idea - 0.5, idea, 'F#7')]
    pad_strings(S, dws, v_hi=30, v_lo=30, hi=('F#4', 'D5'), n_hi=2, lo=False)
    S.n('vln', 'F#5', dw, idea - dw - 0.05, 24)
    for dt, v in ((0.0, 40), (1.6, 34), (3.2, 28)):
        S.n('harp', 'B1', dw + dt, 1.6, v)
    S.n('harp', 'F#1', idea - 1.0, 1.0, 28)
    for p, t, v in (('F#6', dw + 0.15, 36), ('D6', dw + 0.55, 32), ('B5', g07[1] + 0.1, 28), ('A#5', idea - 0.8, 26)):
        S.n('celesta', p, t, 1.0, v)
    S.motif('star', dw, 'celesta', 'dimming, falling, slower')
    S.hit('dawn_warning', dw, 'colour', 'B minor; strings thin to a high line + slow low harp pulse; Star motif dims')
    # --- idea!
    gliss(S, 'harp', 'D5', 'D7', idea, 0.35, 40, 70, DMAJ, ring=1.2)
    for j, q in enumerate(('D6', 'E6', 'F#6', 'A6', 'B6')):
        S.n('celesta', q, idea - 0.3 + j * 0.065, 0.3, 50 + 3 * j)
    S.harm(idea, 'Dadd9')
    S.ch('glock', ['D6', 'A6', 'D7'], idea, 1.5, 86)
    S.ch('celesta', ['D6', 'F#6', 'A6', 'D7'], idea + 0.005, 1.5, 76)
    S.ch('tbells', ['D5', 'A5'], idea, 3.0, 62)
    S.n('kit', 81, idea, 2.0, 84)
    S.ch('pizz', ['D2', 'D3'], idea, 0.4, 92)
    S.ch('horns', ['F#4', 'A4', 'D5'], idea, 0.45, 92)
    S.ch('brass', ['D3', 'A3', 'D4'], idea, 0.4, 82)
    S.ch('ostin', ['D3', 'F#3', 'A3', 'D4', 'F#4', 'A4'], idea, 1.4, 88)
    S.bump('ostin', [(idea + 0.15, 0), (idea + 1.4, -12), (idea + 2.0, -12)])
    S.n('timp', 'D2', idea, 1.0, 82)
    S.fxa('ping', t=idea, lvl=-15, midi='D7', seed=58)
    S.fxa('revcym', t=idea, lvl=-26, length=0.8, seed=59)
    S.hit('idea', idea, 'sting', 'bright D-major sting: harp gliss + celesta run into glock/celesta/triangle/bells, horn + brass stab, sparkle ping')
    sustain(S, 'trem', [(idea + 0.3, end + 0.3, ['D3', 'A3', 'D4'])], 40)
    S.bump('trem', [(idea + 0.3, -8), (end, 0)])


def s06(S):
    st, ru, au, bu = c('s06_start'), c('run_up'), c('aim_up'), c('beam_up')
    th, ho, hi, ig = c('touch_heart'), c('heart_out'), c('heart_in'), c('ignite')
    rise, wa, bf, dd = c('rise'), c('whale_appear'), c('beam_fade'), c('deng_dark')
    d11, d12, g08, d13, d14, g09, d15 = (LINE[k] for k in ('D11', 'D12', 'G08', 'D13', 'D14', 'G09', 'D15'))
    end = SCENE['s06'][1]
    S.section('s06', st, end, 'D major / B minor -> E major', 'ostinato + timpani build with accelerando; deflation; '
              'aching solo piano; held breath; huge swell; E-major tutti with the Whale theme; cut to darkness')
    tm = TM([(0, st), (12, ru), (18, au), (24, bu)])
    # --- the build under D11 (beats 0-12)
    plan = [(0, 4, 'D'), (4, 8, 'Bm'), (8, 10, 'G'), (10, 11, 'Asus4'), (11, 12, 'A')]
    for b0, b1, sym in plan:
        r = bass_of(sym, 'A1', 'G#2')
        for k in range(int(round((b1 - b0) * 2))):
            bb = b0 + k * 0.5
            S.n('piano', r if k % 2 == 0 else r + 12, tm(bb), 0.2, 34 + 22 * bb / 12)
    pad_strings(S, [(tm(a), tm(b), s) for a, b, s in plan], v_hi=36, v_lo=40, hi=('F#3', 'F#4'), n_hi=3)
    for tr in ('spad_hi', 'spad_lo'):
        S.bump(tr, [(st, -8), (ru - 0.1, 0)])
    # --- run_up / aim_up: driving ostinato, timpani, heroic horn line (accelerando 114 -> 124 -> 138 bpm)
    plan2 = [(12, 15, 'Bm'), (15, 18, 'G'), (18, 21, 'Em'), (21, 22.5, 'F#sus4'), (22.5, 24, 'F#')]
    pat6 = [0, 12, 7, 12, 0, 7]
    for b0, b1, sym in plan2:
        S.harm(tm(b0), sym)
        r = bass_of(sym, 'E2', 'D#3')
        for k in range(int(round((b1 - b0) * 2))):
            bb = b0 + k * 0.5
            t = tm(bb)
            x = (bb - 12) / 12
            S.n('ostin', r + pat6[int(round((bb - 12) * 2)) % 6], t, (tm(bb + 0.5) - t) * 0.8, 70 + 30 * x + (8 if k % 2 == 0 else 0))
            S.n('piano', r - 12 + (0 if k % 2 == 0 else 12), t, 0.15, 50 + 30 * x)
        for k in range(int(round(b1 - b0))):
            bb = b0 + k
            S.n('timp', bass_of(sym, 'F#1', 'F2') + 12, tm(bb), 0.4, 60 + 35 * (bb - 12) / 12 + (10 if k == 0 else 0))
    S.mel_tm('horns', seq("B3:1.5 D4:.5 F#4:1 | G4:2 F#4:1 | E4:1.5 F#4:.5 G4:1 | A#4:2 C#5:1"), tm, 12, 88)
    S.mel_tm('vln', seq("E5:1.5 F#5:.5 G5:1 | A#5:2 C#6:1"), tm, 18, 72)
    for k in range(6):
        bb = 18 + k
        sym = 'Em' if bb < 21 else 'F#'
        S.n('tuba', bass_of(sym, 'E1', 'D#2'), tm(bb), 0.3, 72 + 4 * k)
        S.n('trombone', bass_of(sym, 'E2', 'D#3'), tm(bb), 0.3, 70 + 4 * k)
    sustain(S, 'choir', [(tm(18), tm(21), voicing('Em', 'B2', 'G4', 4)), (tm(21), tm(22.5), voicing('F#sus4', 'B2', 'G4', 4)),
                         (tm(22.5), tm(24), voicing('F#', 'B2', 'G4', 4))], 66)
    pad_strings(S, [(tm(a), tm(b), s) for a, b, s in plan2], v_hi=56, v_lo=58, hi=('F#3', 'F#4'), n_hi=3, basses=56, harm=False)
    timp_roll(S, 'F#2', tm(23), bu - 0.02, 70, 108)
    gliss(S, 'harp', 'D4', 'D7', bu, 0.5, 50, 84, DMAJ, ring=1.2)
    S.fxa('revcym', t=bu, lvl=-17, length=1.5, seed=61)
    S.hit('run_up', ru, 'drive', 'driving string ostinato + piano + timpani on every beat; horns carry a rising heroic line')
    S.hit('aim_up', au, 'drive', 'accelerando; heavy tuba/trombone heaves on each beat (the iron wheel), choir enters')
    # --- beam up ... and not enough
    S.harm(bu, 'D')
    S.n('horns', 'D5', bu, 1.6, 100)
    S.ch('horns', ['F#4', 'A4'], bu, 1.4, 90)
    S.ch('brass', ['D3', 'F#3', 'A3', 'D4'], bu, 1.2, 100)
    S.n('tuba', 'D2', bu, 1.3, 96)
    S.ch('ostin', ['D2', 'A2', 'D3', 'F#3', 'A3', 'D4', 'F#4', 'A4', 'D5'], bu, 1.1, 100)
    S.ch('choir', ['D3', 'A3', 'D4', 'F#4', 'A4'], bu, 1.3, 96)
    S.n('timp', 'D2', bu, 1.2, 112)
    S.n('kit', 57, bu, 2.0, 84)
    S.ch('glock', ['D6', 'A6'], bu, 1.5, 70)
    S.fxa('boom', t=bu, lvl=-14, decay=1.0, seed=62)
    S.fxa('crash', t=bu, lvl=-20, decay=1.6, seed=63)
    for tr in ('horns', 'brass', 'tuba', 'ostin', 'choir'):
        S.bump(tr, [(bu + 0.4, 0), (bu + 1.6, -14), (bu + 3.0, -14)])
    S.hit('beam_up', bu, 'hit', 'D-major arrival: horns on high D, brass, choir, timpani, cymbal, boom')
    for q in ('D4', 'F#4', 'A4', 'D5'):
        S.n('wobble', q, bu + 0.3, d12[0] + 0.6 - bu - 0.3, 60)
    S.bend('wobble', [(bu + 0.85, 0.0)] + [(bu + 0.9 + k * 0.02, -1.0 * min(1, (k * 0.02) / (d12[0] + 0.4 - bu - 0.9)) ** 1.5)
                                            for k in range(int((d12[0] + 0.6 - bu - 0.9) / 0.02))] + [(d12[0] + 0.9, 0.0)])
    S.bump('wobble', [(bu + 0.3, -6), (bu + 0.9, 0), (d12[0] + 0.6, -26), (d12[0] + 0.9, -26)])
    pad_strings(S, [(bu + 1.0, bu + 1.5, 'G/B'), (bu + 1.5, d12[0] + 0.8, 'Gm/Bb')], v_hi=40, v_lo=40, hi=('D4', 'D5'), n_hi=3)
    S.n('spad_hi', 'D5', d12[0] + 0.8, th - d12[0] - 0.5, 26)
    S.hit('D12_start', d12[0], 'deflate', 'the chord sags (strings bend down a semitone), bass sinks D-B-Bb; near silence')
    # --- touch_heart ... the choice: solo piano in the gaps
    S.harm(th, 'Gmaj7')
    S.ch('piano', ['G2', 'D3', 'F#3', 'B3'], th, 2.5, 36, roll=0.06)
    for p, dt, d, v in (('A4', 0.05, 0.5, 46), ('F#5', 0.5, 0.6, 48), ('E5', 1.05, 1.4, 44)):
        S.n('piano', p, th + dt, d, v)
    S.motif('lamp', th, 'solo piano', 'the heart: head of the theme, alone')
    S.hit('touch_heart', th, 'solo', 'near-silence; solo piano Lamp-theme head over Gmaj7')
    pad_strings(S, [(g08[0], g08[0] + 1.3, 'Em7'), (g08[0] + 1.3, g08[1] + 0.3, 'Cmaj7')], v_hi=30, v_lo=32, hi=('E3', 'E4'), n_hi=3)
    for p, dt, v in (('D5', 0.05, 38), ('B4', 0.4, 40), ('A4', 0.75, 34)):
        S.n('piano', p, g08[1] + dt, 0.9, v)
    dc = [(d13[0], d13[0] + 1.4, 'D/F#'), (d13[0] + 1.4, d13[0] + 2.8, 'G'), (d13[0] + 2.8, d13[0] + 4.0, 'Em7'),
          (d13[0] + 4.0, d13[1] + 0.1, 'A7sus4')]
    pad_strings(S, dc, v_hi=30, v_lo=32, hi=('F#3', 'F#4'), n_hi=3)
    for (a, b, sym), ps in zip(dc, (['F#2', 'D3'], ['G2', 'D3'], ['E2', 'B2'], ['A1', 'E2'])):
        S.ch('piano', ps, a, b - a + 0.3, 30)
    for p, dt, v in (('B4', 0.08, 36), ('C#5', 0.38, 38), ('D5', 0.68, 40)):
        S.n('piano', p, d13[1] + dt, 0.6, v)
    S.n('piano', 'E5', d14[0], 1.5, 36)
    dk = [(d14[0], d14[0] + 1.2, 'Gadd9'), (d14[0] + 1.2, d14[0] + 2.4, 'A'), (d14[0] + 2.4, ho, 'D')]
    pad_strings(S, dk[:2], v_hi=34, v_lo=36, hi=('F#3', 'F#4'), n_hi=3)
    pad_strings(S, dk[2:], v_hi=44, v_lo=44, hi=('F#3', 'A4'), n_hi=4)
    for (a, b, sym), q in zip(dk, ('G2', 'A2', 'D3')):
        S.n('cello', q, a, b - a + 0.05, 36 if q != 'D3' else 44)
    # --- heart_out: held breath
    S.harm(ho, 'Dadd9')
    S.n('wobble', 'A5', ho - 0.25, hi - ho + 0.6, 30)
    S.bump('wobble', [(hi - 0.1, 0), (hi + 0.35, -30), (hi + 0.6, -30)])
    S.hit('heart_out', ho, 'held breath', 'everything drops out; one high violin A5 + faint shimmer')
    # --- heart_in -> ignite: the swell
    S.harm(hi, 'B7sus4')
    S.harm(hi + 0.4, 'B7')
    timp_roll(S, 'F#2', ho + 1.2, hi, 22, 44, 12, 14)
    timp_roll(S, 'B1', hi, ig - 0.015, 50, 122, 16, 26)
    S.fxa('revcym', t=ig, lvl=-9, length=1.9, seed=64)
    S.fxa('inhale', t=ig, lvl=-13, length=1.4, seed=65)
    S.fxa('riser', t=ho + 1.0, lvl=-12, length=ig - ho - 1.0, f0=200, f1=9000, root=P('B3'), seed=66)
    run = [q for q in range(P('B3'), P('B5') + 1) if q % 12 in EMAJ]
    for j, q in enumerate(run):
        S.n('vln', q, hi + (ig - 0.06 - hi) * j / (len(run) - 1), 0.3, 64 + 44 * j / (len(run) - 1))
    for tr, ps, v in (('choir', ['B2', 'F#3', 'B3', 'D#4', 'F#4'], 92), ('brass', ['B2', 'F#3', 'B3', 'D#4'], 96),
                      ('horns', ['F#4', 'B4'], 90), ('spad_lo', ['B1', 'B2'], 84), ('spad_hi', ['F#3', 'A3', 'D#4', 'F#4'], 84),
                      ('tuba', ['B1'], 90)):
        S.ch(tr, ps, hi, ig - hi + 0.02, v)
        S.bump(tr, [(hi, -22), (ig - 0.02, 0)])
    S.hit('heart_in', hi, 'swell', 'huge 0.8 s swell on B7: timpani roll, violin run, choir/brass crescendo, reverse cymbal, inhale, riser')
    # --- IGNITE
    S.harm(ig, 'Eadd9')
    S.fxa('boom', t=ig, lvl=-1, f_hi=80, f_lo=28, decay=2.4, dur=7.0, seed=67)
    S.fxa('crash', t=ig, lvl=-7, decay=3.5, seed=68)
    S.fxa('crash', t=ig + 0.025, lvl=-10, decay=3.0, seed=69, bright=0.8)
    S.fxa('bells', t=ig, lvl=-13, midis=['E5', 'B5', 'E6', 'G#6'], decay=4.0, seed=70)
    S.n('kit', 49, ig, 3.0, 124)
    S.n('kit', 57, ig + 0.015, 3.0, 110)
    S.n('kit', 55, ig + 0.03, 2.0, 90)
    S.n('timp', 'E2', ig, 1.5, 127)
    S.n('timp', 'B1', ig + 0.004, 1.5, 112)
    timp_roll(S, 'E2', ig + 0.15, rise - 0.05, 92, 84, 18, 18, seed=9)
    dur = rise - ig + 0.1
    S.ch('basses', ['E1', 'E2'], ig, dur, 116)
    S.ch('ostin', ['E2', 'B2', 'E3', 'G#3', 'B3', 'E4', 'G#4', 'B4', 'E5'], ig, dur, 118)
    S.ch('spad_lo', ['E2', 'B2', 'E3'], ig, dur, 116)
    S.ch('spad_hi', ['G#3', 'B3', 'E4', 'F#4', 'G#4', 'B4'], ig, dur, 114)
    S.ch('vln', ['E5', 'G#5', 'B5', 'E6'], ig, dur, 108)
    S.ch('brass', ['E3', 'B3', 'E4', 'G#4'], ig, dur, 118)
    S.n('tuba', 'E2', ig, dur, 116)
    S.ch('trombone', ['E3', 'B3'], ig, dur, 112)
    for p, dt, d in (('E4', 0.0, 0.36), ('B4', 0.375, 0.36), ('E5', 0.75, rise - ig - 0.75 + 0.1)):
        S.n('horns', p, ig + dt, d, 120)
    S.ch('choir', ['E3', 'B3', 'E4', 'G#4', 'B4'], ig, dur, 120)
    S.ch('oohs', ['E4', 'G#4', 'B4', 'E5'], ig, dur, 100)
    S.ch('tbells', ['E5', 'B5'], ig, 5.0, 112)
    S.ch('glock', ['E6', 'G#6', 'B6', 'E7'], ig, 3.0, 96, roll=0.01)
    S.ch('celesta', ['E6', 'B6', 'E7'], ig, 3.0, 90)
    gliss(S, 'harp', 'E2', 'E7', ig + 0.9, 0.9, 88, 112, EMAJ, ring=2.5, ease=0.9)
    S.ch('piano', ['E1', 'E2', 'B2'], ig, 3.0, 112)
    S.ch('piano', ['E5', 'G#5', 'B5', 'E6'], ig, 2.5, 100)
    for tr in ('brass', 'trombone', 'tuba', 'choir', 'ostin'):
        S.bump(tr, [(ig + 0.5, 0), (rise - 0.3, -5), (rise, 0)])
    S.hit('ignite', ig, 'MASSIVE TUTTI', 'E major: sub boom, two synth crashes + GM crash/splash, timpani + roll, full strings/brass/'
          'choir, tubular bells, glock/celesta, harp gliss, horn fanfare E-B-E')
    # --- rise: the Whale theme
    wtm = TM([(0, rise), (8, wa), (12, wa + 2.8), (16, bf)])
    S.mel_tm('vln', seq(WHALE_P1, tr=12), wtm, 0, 104)
    S.mel_tm('horns', seq(WHALE_P1), wtm, 0, 108)
    S.mel_tm('cello', seq(WHALE_P1, tr=-12), wtm, 0, 92)
    S.motif('whale', rise, 'violins 8va + horns + cellos', 'full statement, E major, peak note G# on whale_appear')
    wch = [(0, 4, 'E'), (4, 8, 'A'), (8, 10, 'C#m'), (10, 12, 'F#m7'), (12, 14, 'Bsus4'), (14, 16, 'B')]
    wsegs = [(wtm(a), wtm(b), s) for a, b, s in wch]
    pad_strings(S, wsegs, v_hi=96, v_lo=100, hi=('G#3', 'G#4'), n_hi=4, basses=100, choir='choir', v_ch=100,
                ch_rng=('B3', 'E5'), n_ch=4, lo_rng=('A1', 'G#2'))
    for i, (a, b, sym) in enumerate(wsegs):
        arp(S, 'harp', voicing(sym, 'E3', 'E5', 5), a, b, (b - a) / ((wch[i][1] - wch[i][0]) * 4), 62, 'updown', ring=1.4, seed=80 + i)
        sustain(S, 'brass', [(a, b, voicing(sym, 'E3', 'B3', 3))], 86)
        S.n('trombone', bass_of(sym, 'E2', 'D#3'), a, b - a + 0.05, 84)
        S.n('tuba', bass_of(sym, 'E1', 'D#2'), a, b - a + 0.05, 88)
    sustain(S, 'oohs', [(a, b, voicing(s, 'G#4', 'E5', 3)) for a, b, s in wsegs[2:]], 90)
    for bb, p, v in ((0, 'E2', 100), (2, 'B1', 80), (4, 'A1', 96), (6, 'E2', 78), (8, 'C#2', 106)):
        S.n('timp', p, wtm(bb), 1.0, v)
    for t in np.arange(rise, wa - 0.5, 0.9):
        sparkle_burst(S, t, 0.5, 5, EPENT, 'E6', 'E7', 58, 44, seed=int(t * 10), tracks=('glock', 'celesta'))
    S.fxa('revcym', t=wa, lvl=-14, length=2.0, seed=71)
    S.fxa('boom', t=wa, lvl=-13, decay=1.4, seed=72)
    S.fxa('crash', t=wa, lvl=-17, decay=2.8, seed=73)
    S.n('kit', 57, wa, 2.5, 92)
    S.ch('tbells', ['C#5', 'G#5'], wa, 4.0, 96)
    casc = [q for q in range(P('E6'), P('E7') + 1) if q % 12 in EPENT]
    for j, q in enumerate(casc):
        S.n('glock' if j % 2 == 0 else 'celesta', q, wa + j * 1.0 / len(casc), 1.0, 70 - 2 * j)
    S.hit('whale_appear', wa, 'hit', 'theme peaks on G#; choir full, tubular bells, cymbal, boom, star cascade on glock/celesta')
    for tr in ('brass', 'trombone', 'tuba'):
        S.bump(tr, [(d15[0] - 0.3, 0), (d15[0] + 0.2, -12), (d15[1], -12), (d15[1] + 0.8, -2), (bf - 0.05, 0)])
    S.bump('choir', [(d15[0] - 0.3, 0), (d15[0] + 0.2, -5), (d15[1], -5), (d15[1] + 0.8, 0)])
    timp_roll(S, 'B1', d15[1] + 0.3, bf - 0.02, 50, 96)
    S.fxa('revcym', t=bf, lvl=-18, length=1.2, seed=74)
    # --- beam fade: resolution and diminuendo
    S.harm(bf, 'E')
    dur = dd - bf + 0.05
    for tr, ps, v in (('spad_lo', ['E2', 'B2'], 92), ('spad_hi', ['G#3', 'B3', 'E4', 'G#4'], 90), ('basses', ['E1'], 90),
                      ('choir', ['E3', 'B3', 'E4', 'G#4', 'B4'], 92), ('vln', ['E5', 'E6'], 92), ('horns', ['E4', 'G#4'], 72)):
        S.ch(tr, ps, bf, dur, v)
        S.bump(tr, [(bf, 0), (dd - 0.03, -13), (dd + 0.5, -13)])
    S.ch('harp', ['E2', 'B2', 'E3', 'G#3', 'B3', 'E4'], bf, 2.0, 70, roll=0.04)
    S.n('tbells', 'E5', bf, 3.0, 66)
    S.n('timp', 'E2', bf, 1.2, 72)
    S.hit('beam_fade', bf, 'resolution', 'resolves to E major; brass/percussion drop out; diminuendo')
    # --- deng_dark: cut
    S.n('lowsus', 'E1', dd - 0.06, 3.9, 74)
    S.bump('lowsus', [(dd, 0), (dd + 0.8, -4), (dd + 4.2, -40), (dd + 7.0, -40)])
    S.fxa('lowdrone', stem='fx', gate=False, t=dd - 0.04, lvl=-20, midi='E1', dur=4.8)
    S.hit('deng_dark', dd, 'CUT', 'orchestra + reverb gated to silence in 60 ms; only a low E (contrabass + sub drone) fades')


def s07(S):
    st, ws, sd, rb = c('s07_start'), c('whale_song'), c('stardust'), c('reboot')
    nz, fw, sr = c('nuzzle'), c('farewell'), c('sunrise')
    w01, g10, d16 = LINE['W01'], LINE['G10'], LINE['D16']
    end = SCENE['s07'][1]
    S.section('s07', st, end, 'E major', 'silence; reverent Whale chorale; stardust shimmer; music-box reboot; tender strings; '
              'golden sunrise reprise (Lamp phrase B + Star figuration)')
    S.harm(st, 'Eadd9')
    wtm = TM([(0, ws), (16, rb)])
    S.mel_tm('oohs', seq(WHALE_P1), wtm, 0, 64)
    S.mel_tm('vln', seq(WHALE_P1), wtm, 0, 38)
    wch = [(0, 4, 'E'), (4, 8, 'A'), (8, 10, 'C#m'), (10, 12, 'F#m7'), (12, 14, 'Bsus4'), (14, 16, 'B')]
    segs = [(wtm(a), wtm(b), s) for a, b, s in wch]
    pad_strings(S, segs, v_hi=40, v_lo=44, hi=('E3', 'E4'), n_hi=3, basses=44, choir='choir', v_ch=46,
                ch_rng=('E3', 'D#4'), n_ch=3, lo_rng=('A1', 'G#2'))
    for tr in ('oohs', 'choir', 'spad_hi', 'spad_lo', 'basses', 'vln'):
        S.bump(tr, [(ws, -10), (sd, -3), (rb - 1.0, 0)])
    for bb, p in ((0, 'E2'), (4, 'A1'), (8, 'C#2'), (12, 'B1')):
        S.n('harp', p, wtm(bb), 2.0, 44)
    S.motif('whale', ws, 'choir (oohs) + strings', 'reverent chorale as the whale descends')
    S.hit('whale_song', ws, 'entry', 'choir/strings chorale on the Whale theme (pp -> mp)')
    gliss(S, 'harp', 'E5', 'E7', sd + 0.8, 0.8, 56, 40, EMAJ, down=True, ring=2.0)
    rng = np.random.default_rng(71)
    pool = [q for q in range(P('B5'), P('E8') + 1) if q % 12 in EPENT]
    for k, t in enumerate(np.arange(sd + 0.1, rb - 0.3, 0.85)):
        i0 = int(rng.integers(len(pool) // 2, len(pool)))
        for j in range(7):
            if i0 - j < 0:
                break
            S.n('celesta', pool[i0 - j], t + j * 0.07, 0.8, 50 - 2 * j + rng.integers(-3, 4))
            if k % 2 == 0 and j % 2 == 0:
                S.n('glock', pool[i0 - j], t + j * 0.07, 0.6, 32)
    S.hit('stardust', sd, 'shimmer', 'harp gliss down + falling celesta/glock cascades + glitter swell (river of stardust)')
    # --- reboot
    S.harm(rb, 'E')
    for p, dt, v in (('B5', 0.0, 56), ('G#6', 0.62, 50), ('F#6', 1.06, 50), ('E6', 1.85, 56)):
        S.n('mbox', p, rb + dt, 1.2, v)
    pad_strings(S, [(rb, w01[0], 'E')], v_hi=36, v_lo=36, hi=('G#3', 'G#4'), n_hi=3)
    for tr in ('spad_hi', 'spad_lo'):
        S.bump(tr, [(rb, -10), (w01[0], 0)])
    S.fxa('ping', t=rb, lvl=-26, midi='E7', seed=75)
    S.motif('lamp', rb, 'music box', 'waking: the first four notes, hesitant')
    S.hit('reboot', rb, 'chime', 'music-box Lamp-theme head (B-G#-F#-E), soft ping, warm E-major swell')
    # --- tender strings under W01 / G10 / D16
    ts = [(w01[0], w01[0] + 1.4, 'E'), (w01[0] + 1.4, w01[1], 'C#m7'), (w01[1], nz, 'A'), (nz, g10[0], 'E/G#'),
          (g10[0], g10[0] + 1.2, 'A'), (g10[0] + 1.2, g10[0] + 2.3, 'E/G#'), (g10[0] + 2.3, g10[1] - 0.2, 'F#m7'),
          (g10[1] - 0.2, d16[0] - 0.3, 'B7sus4'), (d16[0] - 0.3, d16[0], 'B7'), (d16[0], d16[0] + 1.3, 'E'),
          (d16[0] + 1.3, d16[0] + 2.4, 'C#m7'), (d16[0] + 2.4, d16[0] + 3.3, 'A'), (d16[0] + 3.3, fw, 'B7sus4')]
    pad_strings(S, ts, v_hi=42, v_lo=44, hi=('G#3', 'G#4'), n_hi=3)
    for a, b, sym in ts:
        S.n('cello', bass_of(sym, 'A2', 'G#3'), a, b - a + 0.05, 38)
        up = voicing(sym, 'E3', 'E4', 3)
        for j, q in enumerate(up):
            S.n('harp', q, a + 0.05 + j * 0.12, 1.2, 32)
    for j, q in enumerate(('A2', 'E3', 'A3', 'C#4', 'E4')):
        S.n('harp', q, w01[1] + 0.05 + j * 0.1, 1.5, 40)
    for p, dt in (('B5', 0), ('E6', 0.15), ('G#6', 0.3), ('F#6', 0.45), ('G#6', 0.52), ('B6', 0.6)):
        S.n('celesta', p, nz + dt, 1.0, 48)
    S.motif('star', nz, 'celesta', 'as Guang hugs Deng')
    S.hit('nuzzle', nz, 'motif', 'celesta Star-theme head')
    for p, dt in (('B4', 0.08), ('G#5', 0.3), ('F#5', 0.52)):
        S.n('harp', p, g10[1] + dt, 1.2, 44)
    # --- farewell: lift into the sunrise
    fs = [(fw, fw + 0.6, 'Aadd9'), (fw + 0.6, fw + 0.9, 'B7sus4'), (fw + 0.9, sr, 'B7')]
    pad_strings(S, fs, v_hi=58, v_lo=60, hi=('F#3', 'A4'), n_hi=4, choir='choir', v_ch=58)
    for tr in ('spad_hi', 'spad_lo', 'choir'):
        S.bump(tr, [(fw, -6), (sr - 0.05, 0)])
    gliss(S, 'harp', 'E4', 'E7', sr - 0.1, sr - 0.1 - fw, 50, 84, EMAJ, ring=1.5)
    sc = [q for q in range(P('E4'), P('E5') + 1) if q % 12 in EMAJ]
    for j, q in enumerate(sc):
        S.n('vln', q, fw + (sr - 0.15 - fw) * j / (len(sc) - 1), 0.25, 58 + 28 * j / (len(sc) - 1))
    S.n('horns', 'B3', sr - 0.5, 0.48, 80)
    timp_roll(S, 'B1', fw + 0.4, sr - 0.02, 36, 84)
    S.fxa('revcym', t=sr, lvl=-17, length=1.3, seed=76)
    S.hit('farewell', fw, 'lift', 'harp gliss + rising violin scale + timpani roll + cymbal swell into the sunrise')
    # --- sunrise: golden reprise, Lamp phrase B in E + Star figuration
    spb = (end - sr) / 14.0
    mel = "G#5:2 B5:1 | A5:2 G#5:1 | F#5:1 E5:1 C#5:1 | B4:2 D#5:1 | E5:3"
    S.mel('vln', seq(mel), sr, spb, 98)
    S.mel('horns', seq(mel, tr=-12), sr, spb, 92)
    S.mel('oohs', seq(mel), sr, spb, 70)
    sch = [(0, 3, 'E'), (3, 6, 'Amaj7'), (6, 9, 'F#m7'), (9, 11, 'E/B'), (11, 12, 'B7'), (12, 14, 'E')]
    ssegs = [(sr + a * spb, sr + b * spb, s) for a, b, s in sch]
    pad_strings(S, ssegs, v_hi=86, v_lo=90, hi=('G#3', 'B4'), n_hi=4, basses=84, choir='choir', v_ch=84,
                ch_rng=('B3', 'E5'), n_ch=4, lo_rng=('A1', 'G#2'))
    for i, (a, b, sym) in enumerate(ssegs):
        arp(S, 'harp', voicing(sym, 'E3', 'E5', 5), a, b, spb / 4, 60, 'updown', ring=1.4, seed=90 + i)
        star_figure(S, 'celesta', sym, a, b, spb / 2, 56, lo='B5', seed=95 + i)
        star_figure(S, 'glock', sym, a + spb / 2, b, spb, 40, lo='E6', seed=99 + i)
    for bb, p, v in ((0, 'E2', 84), (3, 'A1', 66), (6, 'F#2', 64), (12, 'E2', 86)):
        S.n('timp', p, sr + bb * spb, 1.2, v)
    timp_roll(S, 'B1', sr + 9 * spb, sr + 12 * spb - 0.02, 48, 80)
    S.n('kit', 57, sr, 2.5, 70)
    S.fxa('crash', t=sr, lvl=-21, decay=2.5, seed=77)
    S.ch('tbells', ['E5', 'B5'], sr, 4.0, 68)
    S.n('celesta', 'B6', sr + 13 * spb, 1.2, 40)
    S.n('celesta', 'E7', sr + 13 * spb + 0.2, 1.2, 36)
    S.motif('lamp+star', sr, 'violins/horns/oohs + celesta/glock', 'golden reprise')
    S.hit('sunrise', sr, 'REPRISE', 'full golden Lamp theme (phrase B, E major) + Star-theme figuration; soft crash + bells')


def s08(S):
    st, b1, b2, wave, et, cr = c('s08_start'), c('blink1'), c('blink2'), c('wave'), c('end_title'), c('credits')
    end = SCENE['s08'][1]
    S.section('s08', st, end, 'D major', 'night again: music-box Lamp theme, two star twinkles, warm cadence on the end title, '
              'celesta coda, fade to silence')
    spb = 0.75
    tp = st + 0.6
    S.mel('mbox', seq("A5:1 | F#6:2 E6:1 | D6:2 B5:1 | A5:2"), tp, spb, 56)
    S.motif('lamp', tp, 'music box', 'night again')
    segs = [(st + 0.2, tp + spb, 'Dadd9'), (tp + spb, tp + 4 * spb, 'D'), (tp + 4 * spb, tp + 7 * spb, 'G'),
            (tp + 7 * spb, b1 - 0.3, 'Bm7'), (b1 - 0.3, wave, 'Gmaj9')]
    pad_strings(S, segs, v_hi=30, v_lo=30, hi=('F#3', 'E4'), n_hi=3)
    for a, b, sym in segs[1:4]:
        harp_bar(S, sym, a, spb, 32)
    for t, p in ((b1, 'A6'), (b2, 'D7')):
        S.n('glock', p, t, 1.2, 72)
        S.n('celesta', p, t, 1.2, 62)
        S.fxa('ping', t=t, lvl=-15, midi=p, seed=int(t))
    S.hit('blink1', b1, 'twinkle', 'glock + celesta A6 + sparkle ping')
    S.hit('blink2', b2, 'twinkle', 'glock + celesta D7 + sparkle ping')
    s2 = [(wave, wave + 1.4, 'Gadd9'), (wave + 1.4, et - 2.0, 'Em7'), (et - 2.0, et - 1.1, 'A7sus4'), (et - 1.1, et, 'A7')]
    s3 = [(et, et + 1.4, 'D'), (et + 1.4, et + 2.5, 'Gmaj7/D'), (et + 2.5, end - 0.3, 'Dadd9')]
    pad_strings(S, s2, v_hi=44, v_lo=46, hi=('F#3', 'F#4'), n_hi=3, basses=40, choir='oohs', v_ch=38)
    pad_strings(S, s3, v_hi=58, v_lo=58, hi=('F#3', 'A4'), n_hi=4, basses=52, choir='oohs', v_ch=50)
    for tr in ('spad_hi', 'spad_lo', 'basses', 'oohs'):
        S.bump(tr, [(wave, -6), (wave + 1.2, 0)])
        S.bump(tr, [(cr + 0.5, 0), (end - 0.4, -40), (end, -60)])
    S.ch('harp', ['G2', 'D3', 'G3', 'B3', 'D4', 'A4'], wave, 2.5, 46, roll=0.05)
    for q, a, d in (('G2', wave, 1.4), ('E2', wave + 1.4, et - 2.0 - wave - 1.4), ('A2', et - 2.0, 2.0), ('D2', et, 3.5)):
        S.n('cello', q, a, d + 0.05, 42)
    for p, t in (('E6', wave + 1.4), ('D6', wave + 2.2), ('B5', wave + 3.0), ('A5', et - 2.0), ('C#6', et - 0.7), ('D6', et)):
        S.n('mbox', p, t, 1.2 if p != 'D6' or t < et else 2.5, 50 if t < et else 56)
    S.hit('wave', wave, 'entry', 'warm strings + harp enter (Gadd9) as Deng waves')
    S.ch('harp', ['D1', 'A1', 'D2', 'A2', 'D3', 'F#3', 'A3', 'D4', 'F#4', 'A4', 'D5'], et, 3.0, 58, roll=0.04)
    S.ch('celesta', ['D6', 'F#6', 'A6'], et, 2.0, 44, roll=0.03)
    S.ch('piano', ['D3', 'A3', 'D4', 'F#4'], et, 3.0, 40)
    S.n('timp', 'D2', et, 1.5, 44)
    S.n('tbells', 'D5', et, 3.0, 40)
    S.hit('end_title', et, 'CADENCE', 'V7-I lands on D major: strings, oohs, harp roll, celesta, music-box D6, soft timpani')
    S.mel('celesta', seq(STAR_HEAD), cr, 0.5, 44)
    S.n('mbox', 'D6', cr + 1.9, 2.0, 38)
    S.n('mbox', 'A5', cr + 1.9, 2.0, 30)
    S.n('harp', 'D5', cr + 2.6, 1.5, 30)
    S.motif('star', cr, 'celesta', 'credits coda')
    S.hit('credits', cr, 'coda', 'celesta Star theme, last music-box D; strings fade to silence by the end')


def synth_plan(S):
    C = c
    S.curve('pad', [
        (0, -90), (C('title_in'), -50), (C('title_in') + 2.8, -30), (C('title_out'), -36), (C('tilt_end'), -33),
        (C('s02_climb'), -34), (C('s02_arrive'), -35), (C('lamp_ignite') - 0.2, -35), (C('lamp_ignite') + 0.6, -27),
        (C('timelapse_start'), -31), (C('timelapse_end'), -33), (C('s02_sit'), -37), (C('s03_start'), -34),
        (C('meteor_start'), -32), (C('impact') - 0.05, -28), (C('impact') + 0.3, -19), (C('run_down') - 0.3, -30),
        (C('run_down'), -46), (C('boat_out'), -31), (C('lift'), -33), (C('D05_start'), -30), (C('G02_start'), -32),
        (C('look_sky'), -27), (C('determined'), -29), (C('row_back'), -30), (C('s05_start'), -44),
        (C('sit_together') - 0.4, -44), (C('sit_together') + 0.5, -31), (C('dawn_warning'), -33), (C('idea'), -29),
        (C('s06_start'), -31), (C('run_up'), -28), (C('beam_up'), -20), (C('D12_start'), -34), (C('touch_heart'), -36),
        (C('D14_start'), -32), (C('heart_out'), -40), (C('heart_in'), -30), (C('ignite') - 0.02, -24),
        (C('ignite') + 0.2, -13), (C('rise'), -16), (C('whale_appear'), -14), (C('beam_fade'), -18),
        (C('deng_dark') - 0.01, -22), (C('deng_dark') + 0.05, -90), (C('s07_start') - 0.3, -90), (C('s07_start') + 1.0, -42),
        (C('whale_song'), -37), (C('stardust'), -31), (C('reboot'), -26), (C('W01_start'), -30), (C('farewell'), -27),
        (C('sunrise'), -18), (C('s07_end') - 1.0, -22), (C('s07_end') + 0.3, -34), (C('wave'), -30),
        (C('end_title'), -24), (C('credits'), -30), (C('fade_out'), -46), (DUR, -90)])
    S.curve('shim', [
        (0, -90), (C('s01_stars_in'), -70), (C('s01_stars_in') + 2.7, -34), (C('title_out'), -37), (C('tilt_end'), -40),
        (C('s02_climb'), -44), (C('lamp_ignite') - 0.1, -44), (C('lamp_ignite') + 0.4, -33), (C('timelapse_start'), -37),
        (C('timelapse_end'), -40), (C('s02_sit'), -46), (C('s03_start'), -40), (C('meteor_start'), -38), (C('impact'), -36),
        (C('impact') + 0.4, -25), (C('run_down') - 0.2, -36), (C('run_down'), -52), (C('boat_out'), -36), (C('lift'), -39),
        (C('G02_start'), -37), (C('look_sky'), -31), (C('determined'), -38), (C('row_back'), -38), (C('s05_start'), -62),
        (C('sit_together'), -62), (C('sit_together') + 1, -41), (C('dawn_warning'), -45), (C('idea'), -33),
        (C('s06_start'), -50), (C('beam_up'), -34), (C('D12_start'), -45), (C('heart_out'), -38), (C('heart_in'), -34),
        (C('ignite'), -23), (C('whale_appear'), -22), (C('beam_fade'), -29), (C('deng_dark') - 0.01, -30),
        (C('deng_dark') + 0.05, -90), (C('s07_start'), -90), (C('s07_start') + 1.0, -50), (C('whale_song'), -44),
        (C('stardust'), -27), (C('reboot'), -31), (C('W01_start'), -37), (C('sunrise'), -27), (C('s07_end'), -34),
        (C('s08_start') + 0.5, -40), (C('blink1') - 0.5, -37), (C('end_title'), -31), (C('fade_out'), -52), (DUR, -90)])
    sp = lambda t, base, peak, tail=1.3: [(t - 0.01, base), (t, peak), (t + tail, base)]
    S.curve('glit', [
        (0, 0), (C('s01_stars_in'), 1), (C('s01_stars_in') + 2.5, 5), *sp(C('title_out'), 3, 11), (C('tilt_end'), 1.5),
        (C('s02_climb'), 0.4), *sp(C('lamp_ignite'), 0.5, 16), (C('timelapse_start'), 4), (C('timelapse_end'), 3),
        (C('s02_sit'), 0.5), (C('s03_start'), 1), (C('meteor_start'), 2), (C('impact') - 0.01, 3), (C('impact'), 28),
        (C('impact') + 3.0, 5), (C('run_down'), 1), (C('boat_out'), 3), (C('lift'), 1.5), (C('G02_start'), 2.5),
        (C('look_sky'), 3), (C('row_back'), 2), (C('s05_start'), 0.3), (C('sit_together'), 1.5), (C('D10_end'), 3.5),
        (C('dawn_warning'), 0.6), *sp(C('idea'), 0.6, 14, 1.2), (C('s06_start'), 0.2), *sp(C('beam_up'), 0.8, 8, 1.5),
        (C('D12_start'), 0.5), (C('heart_out'), 1.2), (C('heart_in'), 2), (C('ignite') - 0.01, 3), (C('ignite'), 26),
        (C('rise'), 14), (C('whale_appear'), 18), (C('beam_fade'), 6), (C('deng_dark') - 0.01, 4), (C('deng_dark') + 0.01, 0),
        (C('whale_song'), 0.8), (C('stardust') - 0.01, 1), (C('stardust'), 20), (C('reboot'), 10), (C('W01_start'), 3),
        (C('nuzzle'), 6), (C('G10_start'), 2), (C('sunrise'), 11), (C('s07_end'), 4), (C('s08_start') + 1, 1.5),
        (C('end_title'), 4), (C('credits'), 3), (C('fade_out'), 0.5), (DUR, 0)])


def compose():
    S = Score()
    setup(S)
    for f in (s01, s02, s03, s04, s05, s06, s07, s08, synth_plan):
        f(S)
    return S


# ============================================================================ rendering
def fluid(mid, wav):
    cmd = ['fluidsynth', '-ni', '-q', '-F', str(wav), '-r', str(SR), '-O', 'float', '-T', 'wav', '-g', '0.5',
           '-R', '0', '-C', '0', '-o', 'synth.polyphony=2048', SF2, str(mid)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def calibrate(S):
    """Measure each GM program's attack latency and level once (cached)."""
    path = MUSDIR / 'calib.json'
    cal = json.load(open(path)) if path.exists() else {}
    for tr in S.tr.values():
        key = '%s%d_%d' % ('d' if tr.drum else 'm', tr.prog, tr.cal)
        if key not in cal:
            m = pm.PrettyMIDI(resolution=1920, initial_tempo=120.0)
            inst = pm.Instrument(program=tr.prog, is_drum=tr.drum)
            inst.notes.append(pm.Note(100, tr.cal, 0.5, 1.7))
            inst.control_changes.append(pm.ControlChange(110, 0, 3.5))
            m.instruments.append(inst)
            mid, wav = CACHE / ('cal_%s.mid' % key), CACHE / ('cal_%s.wav' % key)
            m.write(str(mid))
            fluid(mid, wav)
            x, _ = sf.read(str(wav), dtype='float32', always_2d=True)
            x = x.mean(1)
            hop = 240
            n = len(x) // hop
            e = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1))
            i0 = int(0.5 * SR / hop)
            pk = e[i0:i0 + 100].max()
            lat = (np.argmax(e[i0:] >= 0.5 * pk)) * hop / SR
            lvl = 10 * np.log10((x[int(0.5 * SR):int(2.0 * SR)] ** 2).mean() + 1e-12)
            cal[key] = dict(lat=float(lat), lvl=float(lvl))
        tr.norm = -20.0 - cal[key]['lvl']
        tr.shift = min(0.07, 0.8 * cal[key]['lat'])
    json.dump(cal, open(path, 'w'), indent=1)
    return cal


def write_midi(tr, path):
    dd = c('deng_dark')
    m = pm.PrettyMIDI(resolution=1920, initial_tempo=120.0)
    inst = pm.Instrument(program=tr.prog, is_drum=tr.drum, name=tr.name)
    notes = []
    for p, t, d, v in tr.notes:
        t0 = max(0.0, t - tr.shift)
        t1 = min(DUR - 0.05, t + d - tr.shift)
        if tr.gate and t < dd < t + d:
            t1 = min(t1, dd + 0.02 - tr.shift)
        if tr.lead and in_line(t):
            v *= 0.85
        if t1 - t0 > 0.01:
            notes.append([int(p), t0, t1, int(np.clip(round(v), 1, 127))])
    notes.sort(key=lambda n: (n[1], n[0]))
    last = {}
    for nt in notes:                       # same-pitch overlaps: cut the earlier note
        if nt[0] in last and last[nt[0]][2] > nt[1] - 0.002:
            last[nt[0]][2] = max(last[nt[0]][1] + 0.01, nt[1] - 0.002)
        last[nt[0]] = nt
    for p, t0, t1, v in notes:
        inst.notes.append(pm.Note(v, p, t0, t1))
    if tr.bends:
        for cc, val in ((101, 0), (100, 0), (6, 12), (38, 0), (101, 127), (100, 127)):
            inst.control_changes.append(pm.ControlChange(cc, val, 0.0))
        for t, semis in sorted(tr.bends):
            inst.pitch_bends.append(pm.PitchBend(int(np.clip(semis / 12 * 8192, -8192, 8191)), max(0.0, t - tr.shift)))
    inst.control_changes.append(pm.ControlChange(110, 0, DUR + 1.0))
    m.instruments.append(inst)
    m.write(str(path))


def render_track(tr, use_cache=True):
    mid = CACHE / ('%s.mid' % tr.name)
    write_midi(tr, mid)
    h = hashlib.md5(open(mid, 'rb').read() + str(tr.prog).encode()).hexdigest()[:12]
    wav = CACHE / ('%s_%s.wav' % (tr.name, h))
    if not (use_cache and wav.exists()):
        for old in CACHE.glob('%s_*.wav' % tr.name):
            old.unlink()
        fluid(mid, wav)
    x, _ = sf.read(str(wav), dtype='float32', always_2d=True)
    y = np.zeros((NS, 2), np.float32)
    n = min(NS, len(x))
    y[:n] = x[:n]
    return y


def shelf_sos(f0, gain_db, q=0.707):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    al = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b0 = A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * al)
    b1 = -2 * A * ((A - 1) + (A + 1) * cw)
    b2 = A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * al)
    a0 = (A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * al
    a1 = 2 * ((A - 1) - (A + 1) * cw)
    a2 = (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * al
    return np.array([[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]])


def pan_gains(p):
    return np.array([np.cos((p + 1) * np.pi / 4), np.sin((p + 1) * np.pi / 4)], np.float32) * np.float32(1.4142)


def duck_scale():
    """Softer dialogue thinning inside the climax tutti (the mixer still sidechains)."""
    s = np.ones(NCTL)
    s[(CT > c('ignite')) & (CT < c('beam_fade'))] = 1.0
    return s


DSCALE = duck_scale()


def process_track(tr, x):
    if not tr.notes:
        return None, 0, 0
    a = max(0, int((min(n[1] for n in tr.notes) - 0.2) * SR))
    b = min(NS, int((max(n[1] + n[2] for n in tr.notes) + 6.0) * SR))
    y = x[a:b] * np.float32(db(tr.gain + tr.norm))
    if tr.hp:
        y = signal.sosfilt(signal.butter(2, tr.hp, 'hp', fs=SR, output='sos'), y, axis=0)
    if tr.lp:
        y = signal.sosfilt(signal.butter(2, tr.lp, 'lp', fs=SR, output='sos'), y, axis=0)
    if tr.shelf:
        y = signal.sosfilt(shelf_sos(*tr.shelf), y, axis=0)
    g = bumps_ctl(tr.bumps) + tr.duck * DUCK * DSCALE
    y = y.astype(np.float32) * ctl_to_audio(10 ** (g / 20), a, b)[:, None] * pan_gains(tr.pan)[None, :]
    return y, a, b


def make_ir(rt, length, predelay, seed, hf=1.0):
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = _t(n)
    bands = [(None, 200, 1.15), (200, 800, 1.0), (800, 2500, 0.85 * hf), (2500, 6000, 0.6 * hf), (6000, None, 0.38 * hf)]
    ir = np.zeros((n, 2))
    for ch in range(2):
        noise = rng.standard_normal(n)
        for lo, hi, mul in bands:
            if lo is None:
                sos = signal.butter(4, hi, 'lp', fs=SR, output='sos')
            elif hi is None:
                sos = signal.butter(4, lo, 'hp', fs=SR, output='sos')
            else:
                sos = signal.butter(4, [lo, hi], 'bp', fs=SR, output='sos')
            ir[:, ch] += signal.sosfiltfilt(sos, noise) * np.exp(-6.9078 * t / (rt * mul))
    ir *= (1 - np.exp(-t / 0.012))[:, None]
    for k in range(14):
        tt = rng.uniform(0.005, 0.075)
        amp = 0.6 * np.exp(-tt / 0.04) * rng.choice([-1, 1])
        ch = k % 2
        ir[int(tt * SR), ch] += amp * np.abs(ir).max() * 3
    ir = np.concatenate([np.zeros((int(predelay * SR), 2)), ir])
    ir /= np.sqrt((ir ** 2).sum() / 2)
    return ir.astype(np.float32)


def reverb(x, ir):
    nz = np.flatnonzero(np.abs(x).max(1) > 1e-7)
    y = np.zeros_like(x)
    if len(nz) == 0:
        return y
    a, b = nz[0], nz[-1] + 1
    for ch in range(2):
        r = signal.oaconvolve(x[a:b, ch], ir[:, ch])
        m = min(len(r), NS - a)
        y[a:a + m, ch] += r[:m].astype(np.float32)
    return y


STEMS = ['keys', 'strings', 'brass_winds', 'choir', 'perc', 'fx', 'synth']


def gate_ctl():
    dd = c('deng_dark')
    g = np.ones(NCTL)
    m = (CT >= dd) & (CT < dd + 0.06)
    g[m] = 0.5 + 0.5 * np.cos(np.pi * (CT[m] - dd) / 0.06)
    g[(CT >= dd + 0.06) & (CT < dd + 2.4)] = 0.0
    m = (CT >= dd + 2.4) & (CT < dd + 2.9)
    g[m] = 0.5 - 0.5 * np.cos(np.pi * (CT[m] - dd - 2.4) / 0.5)
    return g


def mix_stems(S, use_cache=True):
    hall = make_ir(2.7, 4.2, 0.024, 1)
    space = make_ir(5.5, 8.0, 0.045, 2, hf=0.75)
    gate = ctl_to_audio(gate_ctl())[:, None]
    duck_a = ctl_to_audio(DUCK)[:, None]
    carve = signal.butter(2, [1300, 3800], 'bp', fs=SR, output='sos')
    out = {}
    for stem in STEMS:
        t0 = time.time()
        dry = np.zeros((NS, 2), np.float32)
        snd = np.zeros((NS, 2), np.float32)
        spc = np.zeros((NS, 2), np.float32)
        ng_dry = np.zeros((NS, 2), np.float32)
        ng_snd = np.zeros((NS, 2), np.float32)
        for tr in S.tr.values():
            if tr.stem != stem or not tr.notes:
                continue
            y, a, b = process_track(tr, render_track(tr, use_cache))
            if tr.gate:
                dry[a:b] += y
                snd[a:b] += y * tr.send
                spc[a:b] += y * tr.space
            else:
                ng_dry[a:b] += y
                ng_snd[a:b] += y * tr.send
        for kind, st, send, sp, gt, kw in S.fx:
            if st != stem:
                continue
            kw = dict(kw)
            t = kw.pop('t')
            y = FXGEN[kind](**kw)
            n0 = int(round((t * SR))) - (len(y) if kind in FX_ENDS else 0)
            if n0 < 0:
                y, n0 = y[-n0:], 0
            n = min(len(y), NS - n0)
            y = y[:n] * ctl_to_audio(10 ** (-2.0 * DUCK * DSCALE / 20), n0, n0 + n)[:, None]
            (dry if gt else ng_dry)[n0:n0 + n] += y
            (snd if gt else ng_snd)[n0:n0 + n] += y * send
            if gt:
                spc[n0:n0 + n] += y * sp
        if stem == 'synth':
            dk = ctl_to_audio(10 ** (-3.0 * DUCK / 20))[:, None]
            pad = synth_pad(S, curve_ctl(S.curves['pad'])) * dk
            shim = synth_shimmer(S, curve_ctl(S.curves['shim'])) * dk
            glit = synth_glitter(S, curve_ctl(S.curves['glit'], 0.0)) * dk
            dry += pad + shim * 0.6 + glit * 0.5
            snd += pad * 0.3 + shim * 0.25 + glit * 0.35
            spc += pad * 0.2 + shim * 0.7 + glit * 0.6
            del pad, shim, glit
        wet = reverb(snd, hall)
        if np.abs(spc).max() > 0:
            wet += reverb(spc, space) * 0.8
        y = (dry + wet) * gate
        y += ng_dry + reverb(ng_snd, hall)
        band = signal.sosfiltfilt(carve, y, axis=0).astype(np.float32)
        y -= band * duck_a * np.float32(1 - db(-4.5))
        out[stem] = y
        print('  stem %-12s %5.1fs  peak %.3f' % (stem, time.time() - t0, np.abs(y).max()), flush=True)
    return out


# ============================================================================ mastering
K_B1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
K_A1 = [1.0, -1.69065929318241, 0.73248077421585]
K_B2 = [1.0, -2.0, 1.0]
K_A2 = [1.0, -1.99004745483398, 0.99007225036621]


def kweight(x):
    return signal.lfilter(K_B2, K_A2, signal.lfilter(K_B1, K_A1, x, axis=0), axis=0)


def loud_curve(x, win=0.4, hop=0.1):
    """Momentary (0.4 s) or short-term (3 s) loudness, LUFS, sampled every hop."""
    k = kweight(x.astype(np.float64)) ** 2
    ms = k.sum(1)
    cs = np.concatenate([[0], np.cumsum(ms)])
    W, H = int(win * SR), int(hop * SR)
    idx = np.arange(0, len(ms) - W, H)
    v = (cs[idx + W] - cs[idx]) / W
    return (idx + W / 2) / SR, -0.691 + 10 * np.log10(v + 1e-12)


def integrated(x):
    import pyloudnorm as pyln
    return pyln.Meter(SR).integrated_loudness(x.astype(np.float64))


def true_peak(x):
    tp = 0.0
    step = SR * 20
    for i in range(0, len(x), step):
        seg = signal.resample_poly(x[i:i + step + 64], 4, 1, axis=0)
        tp = max(tp, np.abs(seg).max())
    return 20 * np.log10(tp + 1e-12)


def limiter_gain(x, ceiling_db):
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    need = np.minimum(1.0, db(ceiling_db) / np.maximum(np.abs(x).max(1), 1e-9))
    g = need
    for w in (int(0.004 * SR) | 1, int(0.06 * SR) | 1):
        g = uniform_filter1d(minimum_filter1d(g, size=w), size=w)
    return np.minimum(g, need).astype(np.float32)


def ride_gain(mix, pivot=-22.0, ratio=1.6):
    """Slow 'gain riding' (like a mixer's hand on the fader): 1.6:1 around the pivot on momentary loudness,
    attack 0.35 s so hits still punch, release 1.8 s; no upward gain for near-silence; the darkness after
    deng_dark is protected; an extra dip under the two climax lines (G09, D15)."""
    tm, lm = loud_curve(mix, 0.4, 0.01)
    L = np.interp(CT, tm, lm)
    boost = (pivot - L) * (1 - 1 / ratio)
    taper = np.clip((L + 56) / 10, 0, 1)
    boost = np.clip(np.where(boost > 0, boost * taper, boost), -8, 10)
    g = np.empty_like(boost)
    g[0] = boost[0]
    aa, ar = np.exp(-1 / (0.35 * CTL_HZ)), np.exp(-1 / (1.8 * CTL_HZ))
    for i in range(1, len(g)):
        k = aa if boost[i] < g[i - 1] else ar
        g[i] = k * g[i - 1] + (1 - k) * boost[i]
    dd, ws = c('deng_dark'), c('whale_song')
    gd = g[int((dd - 0.3) * CTL_HZ)]
    m = (CT >= dd - 0.3) & (CT < ws - 1.0)
    g[m] = np.minimum(g[m], gd)
    m = (CT >= ws - 1.0) & (CT < ws)
    x = (CT[m] - (ws - 1.0))
    g[m] = np.minimum(g[m], gd + (g[m] - gd) * x)
    for lid, dip in (('G09', -6.0), ('D15', -5.0)):
        s0, e0 = LINE[lid]
        pts = [(s0 - 0.35, 0), (s0 - 0.05, dip), (e0, dip), (e0 + 0.2, 0)]
        mm = (CT >= pts[0][0]) & (CT <= pts[-1][0])
        g[mm] += np.interp(CT[mm], [q[0] for q in pts], [q[1] for q in pts])
    g = g - 2.0 * DUCK                     # a touch more room under every line
    np.save(str(MUSDIR / 'ride_gain_db.npy'), g.astype(np.float32))
    return ctl_to_audio(10 ** (g / 20))


def master(stems, target=-20.0, ceiling=-1.3):
    mix = sum(stems.values())
    gc = ride_gain(mix)
    gn = db(target - integrated(mix * gc[:, None]))
    for _ in range(3):
        y = mix * (gc * gn)[:, None]
        gl = limiter_gain(y, ceiling)
        L = integrated(y * gl[:, None])
        gn *= db(target - L)
    fade = np.ones(NS, np.float32)
    f0 = int((DUR - 1.4) * SR)
    fade[f0:] = cos_ramp(NS - f0)[::-1]
    fade[:int(0.01 * SR)] = cos_ramp(int(0.01 * SR))
    y = mix * (gc * gn)[:, None]
    gl = limiter_gain(y, ceiling)
    G = (gc * gn * gl * fade)[:, None]
    stems = {k: v * G for k, v in stems.items()}
    mix = sum(stems.values())
    tp = true_peak(mix)
    if tp > -1.0:
        s = np.float32(db(-1.05 - tp))
        stems = {k: v * s for k, v in stems.items()}
        mix = mix * s
    return mix, stems


# ============================================================================ analysis
def analyze(S):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import librosa
    x, _ = sf.read(str(AUD / 'music.wav'), dtype='float32', always_2d=True)
    d, _ = sf.read(str(AUD / 'dialogue.wav'), dtype='float32', always_2d=True)
    n = min(len(x), len(d))
    rep = {}
    rep['samples'], rep['seconds'] = len(x), len(x) / SR
    rep['peak_dbfs'] = float(20 * np.log10(np.abs(x).max()))
    rep['true_peak_dbfs'] = float(true_peak(x))
    rep['integrated_lufs'] = float(integrated(x))
    # clicks: HF energy spikes relative to local level
    hf = signal.sosfilt(signal.butter(4, 9000, 'hp', fs=SR, output='sos'), x.mean(1))
    e = np.abs(hf)
    from scipy.ndimage import uniform_filter1d
    loc = uniform_filter1d(e, int(0.05 * SR)) + 1e-5
    ratio = e / loc
    rep['max_hf_spike_ratio'] = float(ratio.max())
    top = np.argsort(ratio)[-5:]
    rep['hf_spike_times'] = [round(float(i / SR), 3) for i in top]
    # hit alignment via onset strength
    oenv = librosa.onset.onset_strength(y=x.mean(1), sr=SR, hop_length=240)
    ot = np.arange(len(oenv)) * 240 / SR
    al = []
    for h in S.hits:
        m = (ot > h['time'] - 0.12) & (ot < h['time'] + 0.12)
        if m.any():
            al.append((h['name'], round(float(ot[m][np.argmax(oenv[m])] - h['time']) * 1000)))
    rep['hit_onset_offsets_ms'] = al
    # dialogue balance
    tm, lm = loud_curve(x[:n])
    _, ld = loud_curve(d[:n])
    diffs = []
    for lid, s, e_, _ in LINES:
        mm = (tm >= s) & (tm <= e_) & (ld > -45)
        if mm.any():
            Lm = 10 * np.log10(np.mean(10 ** (lm[mm] / 10)))
            Ld = 10 * np.log10(np.mean(10 ** (ld[mm] / 10)))
            diffs.append((lid, round(float(Ld - Lm), 1)))
    rep['dialogue_minus_music_LU'] = diffs
    rep['dialogue_minus_music_LU_median'] = float(np.median([v for _, v in diffs]))
    prev = x[:n] + d[:n]
    sf.write(str(AUD / 'preview_music_dialogue.wav'), prev / max(1.0, np.abs(prev).max() / 0.95), SR, subtype='PCM_24')
    json.dump(rep, open(MUSDIR / 'analysis.json', 'w'), indent=1)
    # --- plots
    ts3, ls3 = loud_curve(x, 3.0, 0.25)
    marks = ['title_in', 'tilt_start', 'lamp_ignite', 'timelapse_start', 's02_sit', 'meteor_start', 'impact', 'run_down',
             'boat_out', 'G02_start', 'determined', 'throw1', 'catch', 'tower_fall', 'splash', 'sit_together', 'dawn_warning',
             'idea', 'run_up', 'beam_up', 'touch_heart', 'heart_out', 'ignite', 'whale_appear', 'deng_dark', 'whale_song',
             'reboot', 'sunrise', 'blink1', 'end_title']
    fig, ax = plt.subplots(2, 1, figsize=(26, 11), gridspec_kw=dict(height_ratios=[1.3, 1]))
    for sid, (a, b) in SCENE.items():
        for axx in ax:
            axx.axvspan(a, b, color=('#eef2ff' if int(sid[1:]) % 2 else '#fff7e8'), zorder=0)
        ax[0].text(a + 0.5, -12, sid, fontsize=13, weight='bold')
    for _, s, e_, _ in LINES:
        ax[0].axvspan(s, e_, ymin=0, ymax=0.04, color='#c0392b', alpha=0.6)
    ax[0].plot(tm, lm, color='#9db4d8', lw=0.6, label='music momentary (0.4 s)')
    ax[0].plot(ts3, ls3, color='#1f3b73', lw=1.8, label='music short-term (3 s)')
    ax[0].plot(tm, np.where(ld > -60, ld, np.nan), color='#c0392b', lw=0.5, alpha=0.6, label='dialogue momentary')
    for k in marks:
        ax[0].axvline(c(k), color='#666', lw=0.6, ls=':')
        ax[0].text(c(k), -58, k, rotation=90, fontsize=8, va='bottom')
    ax[0].set_ylim(-60, -8)
    ax[0].set_xlim(0, DUR)
    ax[0].set_ylabel('LUFS')
    ax[0].legend(loc='upper left', fontsize=9)
    ax[0].set_title('Score loudness over time  (integrated %.1f LUFS, true peak %.2f dBTP)' % (rep['integrated_lufs'], rep['true_peak_dbfs']))
    M = librosa.feature.melspectrogram(y=x.mean(1)[::2], sr=SR // 2, n_fft=2048, hop_length=1024, n_mels=128, fmax=12000)
    ax[1].imshow(librosa.power_to_db(M, ref=np.max, top_db=80), origin='lower', aspect='auto', extent=[0, DUR, 0, 128], cmap='magma')
    for k in marks:
        ax[1].axvline(c(k), color='w', lw=0.5, ls=':')
    for sid, (a, b) in SCENE.items():
        ax[1].axvline(a, color='c', lw=1.2)
    ax[1].set_ylabel('mel band')
    ax[1].set_xlabel('film seconds')
    fig.tight_layout()
    fig.savefig(str(MUSDIR / 'analysis_overview.png'), dpi=70)
    plt.close(fig)
    zooms = [('impact', 50, 72), ('comedy', 99, 123), ('climax', 164, 190), ('epilogue', 219, 242)]
    fig, ax = plt.subplots(len(zooms), 1, figsize=(20, 13))
    for axx, (nm, a, b) in zip(ax, zooms):
        i0, i1 = int(a * SR), int(b * SR)
        env = np.abs(x[i0:i1].mean(1))
        hop = 240
        k = len(env) // hop
        env = env[:k * hop].reshape(k, hop).max(1)
        tt = a + np.arange(k) * hop / SR
        axx.fill_between(tt, 20 * np.log10(env + 1e-6), -80, color='#1f3b73', alpha=0.8)
        dd_ = np.abs(d[i0:i1].mean(1))[:k * hop].reshape(k, hop).max(1)
        axx.plot(tt, 20 * np.log10(dd_ + 1e-6), color='#c0392b', lw=0.4, alpha=0.7)
        for h in S.hits:
            if a <= h['time'] <= b:
                axx.axvline(h['time'], color='orange', lw=1)
                axx.text(h['time'], -8, h['name'], rotation=90, fontsize=8, va='top')
        axx.set_xlim(a, b)
        axx.set_ylim(-80, 0)
        axx.set_ylabel('dBFS peak')
        axx.set_title(nm)
    fig.tight_layout()
    fig.savefig(str(MUSDIR / 'analysis_zooms.png'), dpi=70)
    plt.close(fig)
    return rep


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--analyze-only', action='store_true')
    args = ap.parse_args()
    for dpath in (MUSDIR, CACHE, STEMDIR):
        dpath.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    S = compose()
    if args.analyze_only:
        S.hits = json.load(open(MUSDIR / 'cues.json'))['hits']
        print(json.dumps(analyze(S), indent=1))
        return
    print('composed: %d notes on %d tracks, %d fx, %d hits' % (sum(len(t.notes) for t in S.tr.values()), len(S.tr), len(S.fx), len(S.hits)))
    calibrate(S)
    stems = mix_stems(S, not args.no_cache)
    mix, stems = master(stems)
    assert len(mix) == NS
    sf.write(str(AUD / 'music.wav'), mix, SR, subtype='PCM_24')
    for k, v in stems.items():
        sf.write(str(STEMDIR / ('%s.wav' % k)), v, SR, subtype='PCM_24')
    cues = dict(film='一盏灯，一颗星 / A Lamp and a Star', duration=DUR, sample_rate=SR,
                key_plan='D major (home) -> B minor (s03 mystery) -> D -> E major (s06 ignite, s07) -> D major (s08)',
                leitmotifs={'lamp': 'Deng: 5 | 3\' . 2\' | 1\' . 6 | 5 (music box / piano / violins)',
                            'star': 'Guang: 5 1\' 3\' 2\'3\' 5\' (celesta / glock / harp)',
                            'whale': 'Star whale / home: 1 . 5 . | 6 . . 5 | 3\' 2\' 1\' | 5 (strings / choir / horns)'},
                stems=['music_stems/%s.wav' % k for k in STEMS],
                sections=S.sections, motif_statements=sorted(S.motifs, key=lambda m: m['time']),
                hits=sorted(S.hits, key=lambda h: h['time']))
    json.dump(cues, open(MUSDIR / 'cues.json', 'w'), indent=1, ensure_ascii=False)
    print('wrote music.wav  %.1f LUFS  peak %.2f dBFS  (%.0fs)' % (integrated(mix), 20 * np.log10(np.abs(mix).max()), time.time() - t0))
    if args.analyze:
        print(json.dumps(analyze(S), indent=1, ensure_ascii=False))


if __name__ == '__main__':
    main()
