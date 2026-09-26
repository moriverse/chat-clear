#!/usr/bin/env python3
"""Sound design for 《一盏灯，一颗星》 (A Lamp and a Star) -- every sound is synthesized.

No samples, no downloads: filtered noise, modal synthesis (metal / wood / glass), FM, additive
bells, polyBLEP oscillators, granular textures, stick-slip friction (creaks, squeaks), bubble
(Minnaert) models for water, formant synthesis (choir, whale) and convolution reverb with
synthetic impulse responses.  Everything is seeded -> the render is deterministic.

Run from the project root (takes ~1-2 min on one core):
    nice -n 10 python3 audio/sfx.py
Outputs (48 kHz stereo, exactly the timeline duration):
    build/audio/ambience.wav   continuous beds (sea, wind, interiors, lamp hum ...)
    build/audio/sfx.wav        spot effects incl. their reverb returns
    build/sfx/events.json      every placed event: sync time (global s), name, gain, ...
All timing is read from build/timeline.json (cue names, scene boundaries, line times).
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.signal as sig
import soundfile as sf
from scipy.interpolate import CubicSpline

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUILD = os.path.join(ROOT, 'build')
SR = 48000
TAU = 2 * np.pi
CR = 400  # control rate for envelopes (Hz)

TL = json.load(open(os.path.join(BUILD, 'timeline.json')))
DUR = float(TL['duration'])
N = int(round(DUR * SR))
CUE = {k: float(v) for k, v in TL['cues'].items()}
LINES = {ln['id']: ln for ln in TL['lines']}
SC = {s['id']: s for s in TL['scenes']}

# Tonal material (sparkles, bells, chimes, choir) lives in D major pentatonic so it sits with a
# warm score; change KEY_ROOT to transpose every pitched effect at once.
KEY_ROOT = 293.66  # D4
PENTA = [0, 2, 4, 7, 9]      # D E F# A B
MAJ = [0, 4, 7]


def cue(name):
    return CUE[name]


def db(x):
    return 10.0 ** (np.asarray(x, dtype=float) / 20.0)


def ns(sec):
    return int(round(sec * SR))


def tvec(dur):
    return np.arange(ns(dur)) / SR


def rng(seed):
    return np.random.default_rng(seed)


def note(semi, root=KEY_ROOT):
    return root * 2.0 ** (semi / 12.0)


def scale_freqs(lo, hi, degrees=PENTA, root=KEY_ROOT):
    out = []
    for octv in range(-4, 6):
        for d in degrees:
            f = note(d + 12 * octv, root)
            if lo <= f <= hi:
                out.append(f)
    return np.array(sorted(out))


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


# ----------------------------------------------------------------------------------------------
# filters
# ----------------------------------------------------------------------------------------------
_SOS = {}


def _sos(kind, f, order):
    f = tuple(float(min(max(v, 5.0), SR * 0.49)) for v in np.atleast_1d(f))
    key = (kind, f, order)
    if key not in _SOS:
        _SOS[key] = sig.butter(order, f if len(f) > 1 else f[0], btype=kind, fs=SR, output='sos')
    return _SOS[key]


def lp(x, f, order=2):
    return sig.sosfilt(_sos('lowpass', f, order), x, axis=0)


def hp(x, f, order=2):
    return sig.sosfilt(_sos('highpass', f, order), x, axis=0)


def bp(x, lo, hi, order=2):
    return sig.sosfilt(_sos('bandpass', [lo, hi], order), x, axis=0)


def rbj(kind, f0, Q=0.707, gain_db=0.0):
    """RBJ cookbook biquad (b, a).  f0/Q may be arrays (returns arrays of coefficients)."""
    f0 = np.clip(np.asarray(f0, float), 10.0, SR * 0.45)
    w0 = TAU * f0 / SR
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * np.asarray(Q, float))
    one = np.ones_like(cw)
    if kind == 'lp':
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == 'hp':
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == 'bp':  # constant 0 dB peak gain
        b = [alpha, 0 * one, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == 'peak':
        A = 10 ** (gain_db / 40.0)
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    else:
        raise ValueError(kind)
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    return b, a


def tvf(x, kind, fc, Q=0.707, block=64):
    """Time-varying biquad (per-block coefficients, state carried across blocks)."""
    x = np.asarray(x, float)
    n = len(x)
    fc = np.broadcast_to(np.asarray(fc, float), (n,))
    Qa = np.broadcast_to(np.asarray(Q, float), (n,))
    idx = np.minimum(np.arange(0, n, block) + block // 2, n - 1)
    B, A = rbj(kind, fc[idx], Qa[idx])
    y = np.empty(n)
    zi = np.zeros(2)
    for j, i in enumerate(range(0, n, block)):
        y[i:i + block], zi = sig.lfilter(B[:, j], A[:, j], x[i:i + block], zi=zi)
    return y


def reson(x, f, Q):
    b, a = rbj('bp', f, Q)
    return sig.lfilter(b, a, x)


def peq(x, f, gain_db, Q=1.0):
    b, a = rbj('peak', f, Q, gain_db)
    return sig.lfilter(b, a, x, axis=0)


# ----------------------------------------------------------------------------------------------
# oscillators, noise, envelopes
# ----------------------------------------------------------------------------------------------
def phase(f, n=None):
    f = np.asarray(f, float)
    if f.ndim == 0:
        f = np.full(n, float(f))
    return np.cumsum(f) / SR


def sine(f, n=None, ph0=0.0):
    return np.sin(TAU * phase(f, n) + ph0)


def saw(f, n=None):
    """PolyBLEP band-limited sawtooth with arbitrary frequency track."""
    f = np.asarray(f, float)
    if f.ndim == 0:
        f = np.full(n, float(f))
    t = np.cumsum(f) / SR % 1.0
    dt = np.clip(f / SR, 1e-7, 0.5)
    y = 2 * t - 1
    m = t < dt
    x = t[m] / dt[m]
    y[m] -= x + x - x * x - 1
    m = t > 1 - dt
    x = (t[m] - 1) / dt[m]
    y[m] -= x * x + x + x + 1
    return y


def white(n, r):
    return r.standard_normal(n)


def colored(n, r, power=1.0):
    """1/f^power noise (power=1 pink, 2 brown), unit std."""
    X = np.fft.rfft(r.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    f[0] = f[1]
    X /= (f / 1000.0) ** (power / 2)
    y = np.fft.irfft(X, n)
    return y / (np.std(y) + 1e-12)


def pink(n, r):
    return colored(n, r, 1.0)


def smooth_noise(n_ctl, rate, r, cr=CR):
    """Smooth random curve (~N(0,1)) with knots at `rate` Hz, sampled at control rate."""
    nk = int(n_ctl / cr * rate) + 4
    xk = np.arange(nk) / rate
    return CubicSpline(xk, r.standard_normal(nk))(np.arange(n_ctl) / cr)


def ctl2audio(ctl, n, cr=CR):
    return np.interp(np.arange(n) * (cr / SR), np.arange(len(ctl)), ctl)


def ar_env(n, attack, release, total=None, shape=1.0):
    """Linear-ish attack, flat, release (seconds).  Rounded with a sine shape."""
    e = np.ones(n)
    a = min(ns(attack), n)
    r_ = min(ns(release), n)
    if a > 0:
        e[:a] = np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    if r_ > 0:
        e[n - r_:] *= np.cos(np.linspace(0, np.pi / 2, r_)) ** 2
    return e ** shape


def perc(n, attack, tau):
    t = np.arange(n) / SR
    e = np.exp(-t / tau)
    a = max(1, ns(attack))
    e[:a] *= np.linspace(0, 1, a)
    return e


def bell_curve(n, peak=0.5, power=1.0):
    """0 -> 1 -> 0 over n samples, maximum at fraction `peak`."""
    u = np.linspace(0, 1, n)
    g = np.log(0.5) / np.log(np.clip(peak, 0.02, 0.98))
    return np.sin(np.pi * u ** g) ** power


def fade(x, fin=0.0015, fout=0.01):
    x = np.array(x, dtype=float, copy=True)
    n = len(x)
    a = min(n, max(1, ns(fin)))
    b = min(n, max(1, ns(fout)))
    ramp_a = np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    ramp_b = np.cos(np.linspace(0, np.pi / 2, b)) ** 2
    if x.ndim == 1:
        x[:a] *= ramp_a
        x[n - b:] *= ramp_b
    else:
        x[:a] *= ramp_a[:, None]
        x[n - b:] *= ramp_b[:, None]
    return x


def pan(x, p=0.0):
    """Equal-power pan; p scalar or per-sample array in [-1, 1]."""
    p = np.clip(np.asarray(p, float), -1, 1)
    th = (p + 1) * np.pi / 4
    return np.stack([x * np.cos(th), x * np.sin(th)], 1)


def mix_into(dst, src, i0, gain=1.0):
    """dst[i0:] += src (both same ndim), clipped at borders."""
    n = len(src)
    a, b = max(0, i0), min(len(dst), i0 + n)
    if b > a:
        dst[a:b] += gain * src[a - i0:b - i0]


def psum(*xs):
    """Sum arrays of different lengths (zero-padded); mono arrays are promoted if mixed with stereo."""
    xs = [np.asarray(x, float) for x in xs]
    st = any(x.ndim == 2 for x in xs)
    xs = [pan(x, 0) * np.sqrt(2) if (st and x.ndim == 1) else x for x in xs]
    n = max(len(x) for x in xs)
    out = np.zeros((n, 2) if st else n)
    for x in xs:
        out[:len(x)] += x
    return out


def norm_peak(x, peak=1.0):
    m = np.max(np.abs(x)) + 1e-12
    return x * (peak / m)


def norm_rms(x, rms_db=-30.0):
    r_ = np.sqrt(np.mean(np.square(x))) + 1e-12
    return x * (db(rms_db) / r_)


# ----------------------------------------------------------------------------------------------
# physical models / building blocks
# ----------------------------------------------------------------------------------------------
def modal(freqs, t60, amps, dur, hard=1.0, r=None, detune=0.0):
    """Sum of exponentially decaying partials (impulse response of a struck object).
    hard<1 softens the strike (low-passes by a short raised-cosine contact pulse)."""
    n = ns(dur)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for i, (f, T6, a) in enumerate(zip(freqs, t60, amps)):
        if f >= SR * 0.45 or a == 0:
            continue
        ph = 0.0 if r is None else r.uniform(0, 0.3)
        y += a * np.exp(-6.9078 * t / T6) * np.sin(TAU * f * t + ph)
        if detune:
            y += 0.5 * a * np.exp(-6.9078 * t / T6) * np.sin(TAU * f * (1 + detune) * t + ph)
    w = int(SR * 0.00025 / max(hard, 0.02))
    if w > 2:
        k = np.hanning(w)
        y = np.convolve(y, k / k.sum())[:n]
    return y


def bubble(f0, tau, rise=0.3, amp=1.0):
    """Minnaert bubble: damped sine with an upward chirp."""
    m = max(8, int(tau * 5 * SR))
    t = np.arange(m) / SR
    f = f0 * (1 + rise * t / (tau * 5))
    y = np.sin(TAU * np.cumsum(f) / SR) * np.exp(-t / tau) * amp
    k = min(m, 24)
    y[:k] *= np.linspace(0, 1, k)
    return y


def noise_burst(dur, tau, lo, hi, r, attack=0.001, order=2):
    n = ns(dur)
    x = bp(white(n, r), lo, hi, order)
    return x * perc(n, attack, tau)


def grain(f, tau, glassy=0.0, attack=0.002, r=None, maxdur=None):
    d = tau * 5 if maxdur is None else min(tau * 5, maxdur)
    m = max(16, ns(d))
    t = np.arange(m) / SR
    ph = 0 if r is None else r.uniform(0, TAU)
    y = np.sin(TAU * f * t + ph) * np.exp(-t / tau)
    if glassy:
        y += glassy * np.sin(TAU * f * 2.756 * t) * np.exp(-t / (tau * 0.35))
    k = max(2, ns(attack))
    y[:k] *= np.sin(np.linspace(0, np.pi / 2, k)) ** 2
    y[-8:] *= np.linspace(1, 0, 8)
    return y


def sparkles(dur, seed, density, freqs=None, f_lo=2000.0, f_hi=8000.0, tau=(0.04, 0.3),
             amp_sd=0.5, pan_fn=None, pitch_fn=None, glassy=0.3, env_fn=None, attack=0.002):
    """Granular sparkle texture (stereo).  density / env_fn / pitch_fn take u in [0,1];
    pan_fn(u, r) -> pan value."""
    r = rng(seed)
    n = ns(dur)
    out = np.zeros((n + ns(2.0), 2))
    dens = density if callable(density) else (lambda u: density + 0 * u)
    grid = np.linspace(0, 1, 400)
    dmax = max(1e-3, float(np.max(dens(grid))))
    t = 0.0
    if freqs is None:
        freqs = scale_freqs(f_lo, f_hi)
    while True:
        t += r.exponential(1.0 / dmax)
        if t >= dur:
            break
        u = t / dur
        if r.uniform() > dens(u) / dmax:
            continue
        f = float(r.choice(freqs)) if len(freqs) else np.exp(r.uniform(np.log(f_lo), np.log(f_hi)))
        if pitch_fn is not None:
            f *= pitch_fn(u)
        f *= 1 + r.normal(0, 0.002)
        ta = r.uniform(*tau)
        g = grain(f, ta, glassy=glassy, attack=attack, r=r)
        a = min(r.lognormal(0, amp_sd), 2.2) * (env_fn(u) if env_fn else 1.0) * (1500.0 / f) ** 0.3
        p = pan_fn(u, r) if pan_fn else r.uniform(-0.8, 0.8)
        th = (np.clip(p, -1, 1) + 1) * np.pi / 4
        i = ns(t)
        m = min(len(g), len(out) - i)
        out[i:i + m, 0] += a * np.cos(th) * g[:m]
        out[i:i + m, 1] += a * np.sin(th) * g[:m]
    return out[:n + ns(1.5)]


def creak(dur, seed, rate=(25, 60), body=(520, 1150, 2300), Q=14, jitter=0.3, noise=0.05,
          gains=None, shape=0.6):
    """Stick-slip friction: irregular impulse train exciting body resonances."""
    r = rng(seed)
    n = ns(dur)
    nc = int(dur * CR) + 4
    rc = np.interp(np.linspace(0, 1, nc), np.linspace(0, 1, len(rate)), rate)
    rc = rc * (1 + 0.35 * jitter * smooth_noise(nc, 9, r))
    rate_a = np.clip(ctl2audio(rc, n), 2, None) * (1 + jitter * 0.5 * r.standard_normal(n))
    ph = np.cumsum(rate_a) / SR
    idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0]
    imp = np.zeros(n)
    imp[idx] = r.uniform(0.4, 1.0, len(idx))
    imp += noise * r.standard_normal(n) * 0.2
    y = np.zeros(n)
    gains = gains or [1.0 / (1 + 0.4 * i) for i in range(len(body))]
    for f, g in zip(body, gains):
        y += g * reson(imp, f * r.uniform(0.97, 1.03), Q)
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** shape
    return norm_peak(y * env)


def servo(dur, f0, f1, seed, bright=1.0, wobble=0.015, gear=0.3, attack=0.03, release=0.06, curve=None):
    """Small electric motor whir with gear whine."""
    r = rng(seed)
    n = ns(dur)
    u = np.linspace(0, 1, n)
    s_ = smoothstep(u) if curve is None else curve(u)
    f = f0 + (f1 - f0) * s_
    f = f * (1 + wobble * ctl2audio(smooth_noise(int(dur * CR) + 4, 12, r), n))
    x = 0.6 * saw(f) + 0.3 * sine(f * 2)
    x = bp(x, 350, 3200 * bright)
    if gear:
        x += gear * sine(f * 6.93) * (0.6 + 0.4 * sine(f * 0.5))
    x += 0.05 * bp(white(n, r), 2000, 7000)
    return x * ar_env(n, attack, release)


def fm_tone(f, dur, ratio=2.0, index=1.5, tau=0.4, idx_tau=0.1, attack=0.003):
    n = ns(dur)
    t = np.arange(n) / SR
    I = index * np.exp(-t / idx_tau)
    y = np.sin(TAU * f * t + I * np.sin(TAU * f * ratio * t))
    return y * perc(n, attack, tau)


def bell(f0, t60=4.0, seed=0, amp_hum=0.5, hard=0.8):
    r = rng(seed)
    parts = [(0.5, 1.0, amp_hum), (1.0, 0.8, 1.0), (1.183, 0.6, 0.45), (1.506, 0.5, 0.4),
             (2.0, 0.45, 0.45), (2.514, 0.35, 0.3), (2.662, 0.3, 0.25), (3.011, 0.25, 0.2),
             (4.166, 0.18, 0.12), (5.433, 0.12, 0.08), (6.796, 0.09, 0.05)]
    fr = [f0 * p * r.uniform(0.998, 1.002) for p, _, _ in parts]
    return modal(fr, [t60 * d for _, d, _ in parts], [a for _, _, a in parts], t60 * 1.05,
                 hard=hard, detune=0.0015)


def crystal(f0, t60=2.4, seed=0, hard=0.6):
    """Glass / crystal 'ting' (free-bar partials, beating doublets)."""
    r = rng(seed)
    parts = [(1.0, 1.0, 1.0), (2.756, 0.45, 0.32), (5.404, 0.22, 0.14), (8.933, 0.1, 0.05)]
    y = modal([f0 * p for p, _, _ in parts], [t60 * d for _, d, _ in parts],
              [a for _, _, a in parts], t60 * 1.1, hard=hard, detune=0.0025, r=r)
    return y


def choir(dur, freqs, seed, vowel='a', vib_rate=5.2, vib_depth=0.006, voices=3, breath=0.08):
    """Formant-filtered detuned polyBLEP voices -> soft 'aah' choir."""
    r = rng(seed)
    n = ns(dur)
    nc = int(dur * CR) + 4
    src = np.zeros(n)
    for f in freqs:
        for v in range(voices):
            det = 1 + r.normal(0, 0.0035)
            vib = vib_depth * np.sin(TAU * r.uniform(vib_rate * 0.85, vib_rate * 1.15) * np.arange(n) / SR
                                     + r.uniform(0, TAU))
            jit = 0.002 * ctl2audio(smooth_noise(nc, 6, r), n)
            src += saw(f * det * (1 + vib + jit)) * r.uniform(0.7, 1.0)
    src += breath * white(n, r) * np.sqrt(len(freqs) * voices)
    forms = {'a': [(800, 80, 1.0), (1150, 90, 0.5), (2900, 120, 0.12), (3900, 130, 0.05)],
             'o': [(450, 70, 1.0), (800, 80, 0.35), (2830, 100, 0.08), (3500, 130, 0.04)],
             'u': [(325, 50, 1.0), (700, 60, 0.25), (2530, 170, 0.05), (3500, 180, 0.03)]}[vowel]
    y = np.zeros(n)
    for fc, bw, g in forms:
        y += g * reson(src, fc, fc / bw)
    return y


def whoosh(dur, f0, f1, seed, peak=0.5, Q=1.2, power=1.5, pan0=0.0, pan1=0.0, color=1.0):
    r = rng(seed)
    n = ns(dur)
    u = np.linspace(0, 1, n)
    fc = f0 * (f1 / f0) ** u
    x = pink(n, r) if color == 1.0 else white(n, r)
    y = tvf(x, 'bp', fc, Q, block=128)
    y *= bell_curve(n, peak, power)
    return pan(y, pan0 + (pan1 - pan0) * u)


def crackle(dur, seed, density, lo=1500, hi=9000, width=0.9, amp_sd=0.7, click_tau=(0.0004, 0.003)):
    """Electrical / spark crackle: sparse filtered clicks (density callable of u or scalar)."""
    r = rng(seed)
    n = ns(dur)
    out = np.zeros((n + ns(0.05), 2))
    dens = density if callable(density) else (lambda u: density + 0 * u)
    dmax = max(1e-3, float(np.max(dens(np.linspace(0, 1, 200)))))
    t = 0.0
    while True:
        t += r.exponential(1.0 / dmax)
        if t >= dur:
            break
        if r.uniform() > dens(t / dur) / dmax:
            continue
        tau = r.uniform(*click_tau)
        m = ns(tau * 6) + 8
        g = r.standard_normal(m) * np.exp(-np.arange(m) / SR / tau)
        a = r.lognormal(0, amp_sd)
        th = (r.uniform(-width, width) + 1) * np.pi / 4
        i = ns(t)
        out[i:i + m, 0] += a * np.cos(th) * g
        out[i:i + m, 1] += a * np.sin(th) * g
    return norm_peak(bp(out, lo, hi))


def splash(dur, seed, size=1.0, bubbles=120, spray=1.0, low=1.0):
    """Water splash: impact thump + broadband crash with falling cutoff + bubbles + spray drops."""
    r = rng(seed)
    n = ns(dur)
    t = np.arange(n) / SR
    y = np.zeros(n)
    # thump (displaced water)
    th = sine(90 * size ** -0.3 * (1 + 1.5 * np.exp(-t / 0.02)), n) * perc(n, 0.004, 0.09 * size)
    y += low * 0.8 * th
    y += low * 0.6 * lp(white(n, r), 350) * perc(n, 0.003, 0.08 * size) * 3
    # crash: noise with cutoff descending
    fc = 1200 + 7000 * np.exp(-t / (0.12 * size))
    crash = tvf(white(n, r), 'lp', fc, 0.8, block=128) * perc(n, 0.006, 0.22 * size)
    crash += 0.5 * bp(white(n, r), 600, 3000) * perc(n, 0.02, 0.45 * size) * (t > 0.03)
    y += 0.9 * crash
    # bubbles
    for _ in range(int(bubbles * size)):
        tb = abs(r.normal(0.05, 0.25 * size))
        f = np.exp(r.uniform(np.log(350), np.log(3200))) / size ** 0.3
        b = bubble(f, r.uniform(0.006, 0.03), r.uniform(0.1, 0.6), r.lognormal(0, 0.5) * 0.12)
        mix_into(y, b, ns(tb))
    # spray falling back: small splats + plinks
    for _ in range(int(40 * spray * size)):
        tb = r.uniform(0.25, 0.9) * size + abs(r.normal(0, 0.3 * size))
        if tb > dur - 0.1:
            continue
        d = noise_burst(0.03, r.uniform(0.003, 0.01), 1500, 9000, r) * r.lognormal(0, 0.5) * 0.12
        mix_into(y, d, ns(tb))
        if r.uniform() < 0.4:
            mix_into(y, bubble(r.uniform(900, 3500), r.uniform(0.005, 0.02), 0.5, 0.05), ns(tb + 0.003))
    return y


def drips(dur, seed, rate_fn, amp=1.0, f=(700, 2600)):
    """Water drops falling into water (plink = small splat + bubble chirp)."""
    r = rng(seed)
    n = ns(dur)
    out = np.zeros(n + ns(0.3))
    grid = np.linspace(0, 1, 200)
    dmax = max(1e-3, float(np.max(rate_fn(grid))))
    t = 0.0
    while True:
        t += r.exponential(1.0 / dmax)
        if t >= dur:
            break
        if r.uniform() > rate_fn(t / dur) / dmax:
            continue
        a = r.lognormal(0, 0.4) * amp
        fb = np.exp(r.uniform(np.log(f[0]), np.log(f[1])))
        b = bubble(fb, r.uniform(0.008, 0.03), r.uniform(0.3, 1.2), 0.6 * a)
        s = noise_burst(0.02, 0.003, 2000, 9000, r) * 0.25 * a
        mix_into(out, s, ns(t))
        mix_into(out, b, ns(t) + ns(0.002))
    return out[:n + ns(0.3)]


def heartbeat(seed, strength=1.0, f=52.0):
    """Soft 'lub-dub' low thump pair (returns ~0.6 s)."""
    n = ns(0.7)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for dt, g in ((0.0, 1.0), (0.24, 0.65)):
        m = n - ns(dt)
        tt = t[:m]
        th = np.sin(TAU * np.cumsum(f * g ** 0.2 * (1 + 0.6 * np.exp(-tt / 0.02))) / SR) * perc(m, 0.006, 0.075)
        y[ns(dt):] += g * th
    return y * strength


# ----------------------------------------------------------------------------------------------
# reverbs (synthetic impulse responses)
# ----------------------------------------------------------------------------------------------
def make_ir(rt, predelay, seed, taps=(), length=None, bright=1.0):
    """Stereo IR with per-band RT60 (rt = (low, lowmid, mid, high) seconds)."""
    r = rng(seed)
    length = length or max(rt) * 1.15 + predelay
    n = ns(length)
    t = np.arange(n) / SR
    edges = [250, 1000, 4000]
    irs = []
    for ch in range(2):
        x = r.standard_normal(n)
        bands = [sig.sosfiltfilt(_sos('lowpass', edges[0], 4), x),
                 sig.sosfiltfilt(_sos('bandpass', [edges[0], edges[1]], 4), x),
                 sig.sosfiltfilt(_sos('bandpass', [edges[1], edges[2]], 4), x),
                 sig.sosfiltfilt(_sos('highpass', edges[2], 4), x) * bright]
        y = sum(b * np.exp(-6.9078 * t / T6) for b, T6 in zip(bands, rt))
        k = ns(0.012)
        y[:k] *= np.linspace(0, 1, k) ** 2
        y /= np.sqrt(np.sum(y ** 2))
        er = np.zeros(n)
        for dt, g in taps:
            j = ns(dt * r.uniform(0.93, 1.07))
            if j < n:
                er[j] += g * (1 if r.uniform() < 0.6 else -1)
        y = y + er
        pd = ns(predelay)
        y = np.concatenate([np.zeros(pd), y])[:n]
        irs.append(y)
    ir = np.stack(irs, 1)
    return ir / np.sqrt(np.mean(np.sum(ir ** 2, 0)))


def _ir_defs():
    tower_taps = [(0.009 * k, 0.5 * 0.8 ** k) for k in range(1, 9)]
    return {
        'tower': make_ir((2.6, 2.2, 1.6, 0.9), 0.012, 11, tower_taps),
        'lamp': make_ir((1.1, 0.95, 0.8, 0.6), 0.004, 12, [(0.006, .5), (0.011, .4), (0.017, .3)]),
        'outdoor': make_ir((0.8, 0.6, 0.45, 0.3), 0.018, 13, [(0.028, .4), (0.047, .3), (0.083, .25)], bright=0.7),
        'magic': make_ir((3.4, 4.0, 4.2, 3.2), 0.035, 14, bright=1.2),
        'sky': make_ir((7.5, 7.5, 6.0, 4.0), 0.07, 15, [(0.42, .35), (0.83, .22), (1.27, .14), (1.71, .08)]),
    }


def convolve_sparse(x, ir):
    """Linear convolution of a long, mostly-silent signal: convolve only the active chunks."""
    n = len(x)
    out = np.zeros(n)
    blk = SR // 2
    act = np.array([np.any(np.abs(x[i:i + blk]) > 1e-9) for i in range(0, n, blk)])
    i = 0
    nb = len(act)
    while i < nb:
        if not act[i]:
            i += 1
            continue
        j = i
        while j < nb and (act[j] or (j + 1 < nb and act[j + 1])):
            j += 1
        a, b = i * blk, min(n, j * blk)
        y = sig.oaconvolve(x[a:b], ir)
        m = min(len(y), n - a)
        out[a:a + m] += y[:m]
        i = j
    return out


# ----------------------------------------------------------------------------------------------
# the SFX bus
# ----------------------------------------------------------------------------------------------
class SfxBus:
    def __init__(self):
        self.dry = np.zeros((N, 2), np.float32)
        self.irs = _ir_defs()
        self.send = {k: np.zeros((N, 2), np.float32) for k in self.irs}
        self.events = []

    def add(self, name, t, x, gain_db=0.0, p=0.0, sync=0.0, rev=None, send_db=-10.0, cut=None,
            fin=0.001, fout=0.012, note_='', norm=True):
        """Place a sound so that its sync point (seconds into x) lands on global time t.
        With norm=True the sound is peak-normalised first, so gain_db == its dry peak level (dBFS)."""
        x = np.asarray(x, float)
        if x.ndim == 1:
            x = pan(x, p) * np.sqrt(2)
        if norm:
            x = norm_peak(x)
        x = fade(x, fin, fout)
        start = ns(t - sync)
        if cut is not None:  # hard picture cut: sound cuts with a 40 ms fade
            k = ns(cut) - start
            if k < len(x):
                f = ns(0.04)
                k0 = max(0, k - f)
                x[k0:k] *= np.linspace(1, 0, k - k0)[:, None]
                x = x[:max(k, 1)]
        g = float(db(gain_db))
        mix_into(self.dry, (g * x).astype(np.float32), start)
        if rev:
            mix_into(self.send[rev], (g * float(db(send_db)) * x).astype(np.float32), start)
        pk = float(np.max(np.abs(x))) * g if len(x) else 0.0
        self.events.append(dict(time=round(float(t), 4), name=name, gain_db=round(float(gain_db), 2),
                                start=round(start / SR, 4), dur=round(len(x) / SR, 3),
                                peak_dbfs=round(20 * np.log10(pk + 1e-12), 1), reverb=rev,
                                note=note_))

    def render(self):
        out = self.dry.astype(float)
        for k, ir in self.irs.items():
            s = self.send[k]
            if not np.any(s):
                continue
            for ch in range(2):
                out[:, ch] += convolve_sparse(s[:, ch].astype(float), ir[:, ch])
        return out


# ----------------------------------------------------------------------------------------------
# AMBIENCE layer generators (each returns stereo (n, 2), roughly unit-ish level)
# ----------------------------------------------------------------------------------------------
def gen_swell(dur, seed):
    """Distant open-ocean surf wash with slow swells."""
    r = rng(seed)
    n = ns(dur)
    nc = int(dur * CR) + 4
    common = smooth_noise(nc, 0.09, r)
    out = np.zeros((n, 2))
    for ch in range(2):
        env = 0.6 + 0.25 * common + 0.12 * smooth_noise(nc, 0.2, r)
        env = np.clip(env, 0.15, 1.3)
        e = ctl2audio(env, n)
        p = pink(n, r)
        low = lp(p, 240, 2)
        mid = bp(p, 240, 1100)
        out[:, ch] = low * (0.5 + 0.5 * e) + 0.45 * mid * e ** 2
    return out


def gen_waves(dur, seed, rate=0.33, slap=1.0, gurgle=1.0, trickle=0.4, pebble=0.0,
              band_gain=(1.0, 0.75, 0.45), rise=(0.5, 1.3), decay=(1.2, 2.8), width=0.8,
              swell_T=11.0, min_gap=0.9, size=1.0):
    """Waves lapping on rocks: per-wave band envelopes + slaps + gurgling bubbles + trickles."""
    r = rng(seed)
    n = ns(dur)
    nc = int(dur * CR) + 4
    tc = np.arange(nc) / CR
    swell = 0.62 + 0.38 * np.sin(TAU * tc / swell_T + r.uniform(0, TAU)) * (0.8 + 0.2 * smooth_noise(nc, 0.1, r))
    env = np.zeros((3, 2, nc))
    grains = np.zeros((n + ns(4), 2))
    times = []
    t = r.uniform(0, 1.0 / rate)
    while t < dur + 1.5:
        times.append(t)
        t += max(min_gap, (1.0 / rate) * r.uniform(0.45, 1.55))
    for ti in times:
        A = float(np.interp(ti, tc, swell)) * r.lognormal(0, 0.3)
        p = r.uniform(-width, width)
        gl, gr = np.cos((p + 1) * np.pi / 4), np.sin((p + 1) * np.pi / 4)
        ri = r.uniform(*rise)
        di = r.uniform(*decay)
        i0 = max(0, int((ti - 3 * ri) * CR))
        i1 = min(nc, int((ti + 6 * di) * CR))
        if i1 <= i0:
            continue
        u = tc[i0:i1] - ti
        shapes = []
        for (ra, dd) in ((ri * 1.1, di * 0.8), (ri * 0.35, di * 0.42), (0.06, di * 0.65)):
            shapes.append(np.where(u < 0, np.exp(-3 * (u / ra) ** 2), np.exp(-np.maximum(u, 0) / dd)))
        for b in range(3):
            env[b, 0, i0:i1] += A * gl * shapes[b]
            env[b, 1, i0:i1] += A * gr * shapes[b]
        # slap(s) of water on the rocks
        for k in range(r.integers(1, 4) if slap > 0 else 0):
            ts = ti + r.uniform(-0.05, 0.35)
            m = ns(0.16)
            burst = bp(white(m, r), 120 / size, 1000 / size) * perc(m, 0.003, r.uniform(0.02, 0.05))
            fclop = r.uniform(170, 420) / size
            tt = np.arange(m) / SR
            clop = np.sin(TAU * np.cumsum(fclop * (1 + 0.5 * np.exp(-tt / 0.01))) / SR) * perc(m, 0.002, 0.03)
            s_ = (burst * 2.2 + 0.7 * clop) * A * slap * r.uniform(0.35, 1.0) * 0.5
            pp = np.clip(p + r.normal(0, 0.15), -1, 1)
            mix_into(grains, pan(s_, pp) * 1.4, ns(ts))
        # gurgling bubbles in the backwash
        for _ in range(r.poisson(max(0.0, A * gurgle * 22))):
            tb = ti + 0.15 + abs(r.normal(0, di * 0.5))
            f = np.exp(r.uniform(np.log(260), np.log(1900))) / size ** 0.5
            b = bubble(f, r.uniform(0.006, 0.035), r.uniform(0.1, 0.5), r.lognormal(0, 0.5) * 0.05 * A)
            mix_into(grains, pan(b, np.clip(p + r.normal(0, 0.2), -1, 1)), ns(tb))
        for _ in range(r.poisson(max(0.0, A * trickle * 8))):
            tb = ti + r.uniform(0.6, 2.2) * di
            b = bubble(np.exp(r.uniform(np.log(1300), np.log(4800))), r.uniform(0.004, 0.015), 0.6,
                       r.lognormal(0, 0.4) * 0.025 * A)
            mix_into(grains, pan(b, np.clip(p + r.normal(0, 0.3), -1, 1)), ns(tb))
        for _ in range(r.poisson(max(0.0, A * pebble * 60))):
            tb = ti + r.uniform(0.5, 1.8)
            m = ns(0.006)
            c = np.sin(TAU * r.uniform(2500, 7000) * np.arange(m) / SR) * np.exp(-np.arange(m) / SR / 0.0012)
            mix_into(grains, pan(c * r.lognormal(0, 0.5) * 0.03 * A, np.clip(p + r.normal(0, 0.3), -1, 1)), ns(tb))
    out = np.zeros((n, 2))
    edges = [(70 / size, 380 / size), (380 / size, 1900), (1900, 9500)]
    for b, (lo, hi) in enumerate(edges):
        for ch in range(2):
            x = bp(white(n, r), lo, hi, 2)
            out[:, ch] += band_gain[b] * x * ctl2audio(env[b, ch], n)
    out += grains[:n]
    return out


def gen_wind(dur, seed, gust=1.0, bright=1.0, whistle=0.3, base=0.4, gust_every=(5, 13)):
    r = rng(seed)
    n = ns(dur)
    nc = int(dur * CR) + 4
    tc = np.arange(nc) / CR
    g = np.full(nc, base) + 0.08 * smooth_noise(nc, 0.5, r) + 0.04 * smooth_noise(nc, 2.5, r)
    t = r.uniform(0, gust_every[1])
    while t < dur + 5:
        a = r.lognormal(-0.4, 0.4) * gust
        ri, fa = r.uniform(1.2, 3.0), r.uniform(1.8, 4.5)
        u = tc - t
        g += a * np.where(u < 0, np.exp(-3 * (u / ri) ** 2), np.exp(-np.maximum(u, 0) / fa))
        t += r.uniform(*gust_every)
    g = np.clip(g, 0.05, None)
    out = np.zeros((n, 2))
    wf = [r.uniform(620, 760), r.uniform(980, 1150), r.uniform(1400, 1650)]
    for ch in range(2):
        gc = np.roll(g, int(r.uniform(0.05, 0.3) * CR) * (1 if ch else 0)) * (1 + 0.06 * smooth_noise(nc, 1.0, r))
        ga = ctl2audio(gc, n)
        p = pink(n, r)
        y = bp(p, 90, 480) * ga + 0.55 * bright * bp(p, 420, 2300) * ga ** 1.6 + 0.12 * bright * bp(p, 2200, 7500) * ga ** 2.4
        if whistle > 0:
            wn = white(n, r)
            for k, f in enumerate(wf):
                w = tvf(wn, 'bp', f * (0.85 + 0.25 * np.clip(ga, 0, 1.6)), 45, block=256)
                y += whistle * 0.9 / (k + 1) * w * np.clip(ga - 0.55, 0, None) ** 1.5
        out[:, ch] = y
    return out


def gen_flutter(dur, seed):
    """Scarf flapping in the wind."""
    r = rng(seed)
    n = ns(dur)
    nc = int(dur * CR) + 4
    g = np.clip(0.55 + 0.35 * smooth_noise(nc, 0.3, r), 0.05, 1.3)
    ga = ctl2audio(g, n)
    out = np.zeros((n, 2))
    for ch in range(2):
        rate = 5 + 6 * ga + 2.5 * ctl2audio(smooth_noise(nc, 4, r), n)
        ph = np.cumsum(np.clip(rate, 2, None)) / SR
        flap = (0.5 + 0.5 * np.sin(TAU * ph)) ** 2
        flap *= np.clip(0.55 + 0.45 * ctl2audio(smooth_noise(nc, 6, r), n), 0.1, 1.2)
        flap = lp(flap, 40)
        x = bp(white(n, r), 250, 3200) * 0.7 + bp(white(n, r), 110, 420) * 1.2
        out[:, ch] = x * flap * ga ** 1.5
    return out


def gen_hull(dur, seed):
    """Little boat: wavelets slapping the wooden hull, water chop, faint creaks."""
    r = rng(seed)
    n = ns(dur)
    out = 0.55 * gen_waves(dur, seed + 1, rate=0.8, slap=0.0, gurgle=1.2, trickle=0.8,
                           band_gain=(0.5, 0.6, 0.35), rise=(0.3, 0.6), decay=(0.5, 1.2), min_gap=0.4)
    out += 0.35 * gen_swell(dur, seed + 2)
    t = 0.0
    while True:
        t += r.exponential(1 / 1.4)
        if t > dur:
            break
        m = ns(0.3)
        burst = bp(white(m, r), 250, 1600) * perc(m, 0.002, r.uniform(0.008, 0.02))
        hull = modal(np.array([160, 290, 470, 720, 1050]) * r.uniform(0.95, 1.05),
                     [0.14, 0.11, 0.08, 0.06, 0.04], [1, .6, .45, .3, .2], 0.3, hard=0.3) * 0.5
        s_ = (burst + hull) * r.lognormal(0, 0.5) * 0.35
        mix_into(out, pan(s_, r.uniform(-0.35, 0.35)), ns(t))
    t = r.uniform(2, 6)
    while t < dur - 1:
        c = creak(r.uniform(0.3, 0.6), int(r.integers(1e6)), rate=(18, 40, 25), body=(380, 820, 1500), Q=10)
        mix_into(out, pan(c * 0.05, r.uniform(-0.3, 0.3)), ns(t))
        t += r.uniform(5, 11)
    return out


def gen_room(dur, seed, low=90.0, moan=(96.0, 191.0)):
    r = rng(seed)
    n = ns(dur)
    nc = int(dur * CR) + 4
    out = np.zeros((n, 2))
    for ch in range(2):
        y = lp(colored(n, r, 2.0), low) * 0.8
        wn = white(n, r)
        for k, f in enumerate(moan):
            am = np.clip(0.5 + 0.5 * ctl2audio(smooth_noise(nc, 0.15, r), n), 0, None)
            y += 0.35 / (k + 1) * reson(wn, f, 18) * am
        y += 0.02 * bp(white(n, r), 800, 5000)
        out[:, ch] = y
    return out


def gen_stairwell(dur, seed):
    """Inside the stone tower: muffled sea through thick walls, low room tone, stairwell reverb."""
    r = rng(seed)
    sea = gen_waves(dur, seed + 1, rate=0.3, slap=1.2, gurgle=0.4, trickle=0.0) + 0.9 * gen_swell(dur, seed + 2)
    sea = lp(sea, 330, 4)
    mid = sea.mean(1, keepdims=True)
    sea = 0.8 * mid + 0.2 * sea
    ir = make_ir((2.6, 2.1, 1.4, 0.8), 0.015, seed + 3)
    wet = np.stack([sig.oaconvolve(sea[:, c], ir[:, c])[:len(sea)] for c in range(2)], 1)
    room = gen_room(dur, seed + 4)
    return norm_rms(sea * 0.6 + wet * 0.5, -30) + norm_rms(room, -40)


def gen_lamproom(dur, seed, hatch=False):
    """Lamp room: sea heard through the glass, wind on the panes; hatch=True opens the roof."""
    sea = gen_waves(dur, seed + 1, rate=0.33) + 0.8 * gen_swell(dur, seed + 2)
    muff = lp(sea, 1000, 2) + 0.05 * hp(sea, 1000)
    wind = gen_wind(dur, seed + 3, whistle=0.8 if not hatch else 0.4, bright=0.6)
    wind = lp(wind, 1800) * 0.6
    room = gen_room(dur, seed + 4, low=140, moan=(233.0,))
    y = norm_rms(muff, -30) + norm_rms(wind, -35) + norm_rms(room, -44)
    if hatch:
        y += norm_rms(gen_wind(dur, seed + 5, bright=1.2, whistle=0.2, base=0.6), -32)
        y += norm_rms(sea, -38)
    ir = make_ir((0.9, 0.8, 0.7, 0.5), 0.005, seed + 6)
    wet = np.stack([sig.oaconvolve(y[:, c], ir[:, c])[:len(y)] for c in range(2)], 1)
    return y + 0.25 * wet


def gen_beamhum(dur, seed, t0):
    """Distant lighthouse lamp + lens-motor hum outside; swells as the beam sweeps past."""
    r = rng(seed)
    n = ns(dur)
    t = np.arange(n) / SR
    T = t0 + t
    y = np.zeros(n)
    for k, a in zip([1, 2, 3, 4, 5, 7], [1, .5, .35, .2, .12, .06]):
        y += a * (np.sin(TAU * 100 * k * t + r.uniform(0, TAU)) + 0.6 * np.sin(TAU * (100 * k + 0.23 * k) * t))
    motor = lp(saw(37.0, n), 260) * 0.5
    air = bp(pink(n, r), 1500, 4200) * 0.25
    sweep = 0.5 + 0.5 * np.cos(0.9 * T)
    y = lp(y, 650) * (0.55 + 0.45 * sweep ** 2) + motor + air * sweep ** 4
    wh = bp(pink(n, r), 300, 1400) * sweep ** 8 * 0.9
    return np.stack([y + wh * 0.8, y + wh * 1.2 * 0.9], 1)


def gen_lamphum(dur, seed):
    """Close warm electrical hum of the lamp + faint filament buzz + lens motor gears."""
    r = rng(seed)
    n = ns(dur)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for k, a in zip([1, 2, 3, 4, 5, 6, 8], [1, .7, .45, .3, .18, .1, .05]):
        y += a * (np.sin(TAU * 100 * k * t + r.uniform(0, TAU)) + 0.5 * np.sin(TAU * (100 * k + 0.31) * t))
    buzz = bp(np.tanh(3.0 * np.sin(TAU * 100 * t)), 300, 2400) * 0.12
    teeth = np.zeros(n)
    teeth[::SR // 26] = 1.0
    gears = reson(teeth, 950, 10) + 0.6 * reson(teeth, 1830, 12)
    y = y * 0.5 + buzz + 0.25 * gears + 0.05 * bp(pink(n, r), 3000, 9000)
    nc = int(dur * CR) + 4
    wob = 1 + 0.08 * ctl2audio(smooth_noise(nc, 0.4, r), n)
    return np.stack([y * wob, y * wob * 0.95 + 0.02 * bp(pink(n, r), 200, 900)], 1)


def gen_lensticks(dur, seed, period=0.75):
    """Slow clockwork of the rotating Fresnel lens: tick ... tock ..."""
    r = rng(seed)
    n = ns(dur)
    out = np.zeros((n + ns(0.3), 2))
    k = 0
    t = 0.05
    while t < dur:
        if k % 2 == 0:
            s_ = modal(np.array([2800, 4150, 6300]) * r.uniform(0.98, 1.02), [0.05, 0.04, 0.03], [1, .6, .3], 0.12)
        else:
            s_ = modal(np.array([1100, 1900, 2750]) * r.uniform(0.98, 1.02), [0.07, 0.05, 0.04], [1, .5, .3], 0.14)
        s_[:ns(0.12)] += 0.4 * modal([180, 330], [0.08, 0.06], [1, .5], 0.12, hard=0.3)
        mix_into(out, pan(s_ * r.uniform(0.8, 1.0), 0.15 if k % 2 else -0.1), ns(t))
        t += period
        k += 1
    return out[:n]


def gen_timelapse(dur, seed):
    """Nights rushing past: cyclic swooshes, sped-up surf, a faint star-trail drone."""
    r = rng(seed)
    n = ns(dur)
    t = np.arange(n) / SR
    rate = 1.3 * smoothstep(t / 1.4) * (1 - smoothstep((t - (dur - 1.4)) / 1.4))
    ph = np.cumsum(rate) / SR
    sw = (0.5 - 0.5 * np.cos(TAU * ph)) ** 2
    x = tvf(pink(n, r), 'bp', 300 * 5 ** sw, 1.3, block=256) * (0.2 + 0.8 * sw)
    pn = np.sin(np.pi * ph)
    out = pan(x, 0.7 * pn)
    surf = gen_waves(dur, seed + 1, rate=2.4, rise=(0.12, 0.3), decay=(0.3, 0.7), min_gap=0.2, slap=0.4,
                     gurgle=0.3)
    out += 0.5 * surf * (0.3 + 0.7 * smoothstep(t / 1.5))[:, None]
    drone = np.zeros(n)
    for f in (note(24), note(31), note(36), note(40)):
        drone += np.sin(TAU * f * (1 + 0.015 * t / dur) * t + r.uniform(0, 6)) * (0.6 + 0.4 * np.sin(TAU * r.uniform(0.2, 0.5) * t))
    out += 0.05 * drone[:, None] * np.array([[1.0, 0.9]])
    return out


def gen_glowsea(dur, seed, density=2.2):
    return sparkles(dur, seed, density, freqs=scale_freqs(2000, 7000), tau=(0.05, 0.3), glassy=0.4,
                    amp_sd=0.6, pan_fn=lambda u, r: r.uniform(-0.95, 0.95))[:ns(dur)]


def gen_calm(dur, seed):
    w = gen_waves(dur, seed, rate=0.2, slap=0.4, gurgle=0.7, trickle=0.6, band_gain=(1.0, 0.55, 0.22),
                  rise=(0.8, 1.6), decay=(1.6, 3.2), min_gap=2.0)
    return lp(w, 2600) + 0.3 * lp(gen_swell(dur, seed + 1), 800)


def gen_dawnwind(dur, seed):
    w = gen_wind(dur, seed, gust=0.6, bright=0.45, whistle=0.0, base=0.55, gust_every=(6, 12))
    r = rng(seed + 9)
    air = hp(np.stack([pink(ns(dur), r), pink(ns(dur), r)], 1), 4000) * 0.04
    return lp(w, 1500) + air


def gen_underglow(dur, seed):
    """Faint deep mysterious drone of the fallen star glowing under the water."""
    r = rng(seed)
    n = ns(dur)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for f, a in ((note(-24), 1.0), (note(-17), 0.5), (note(-12), 0.35), (note(-5), 0.12)):
        y += a * np.sin(TAU * f * t + r.uniform(0, 6)) * (0.7 + 0.3 * np.sin(TAU * r.uniform(0.07, 0.2) * t))
    y += 0.3 * lp(pink(n, r), 200)
    return np.stack([y, np.roll(y, 240)], 1)


LAYERS = {
    'swell': (gen_swell, 101), 'rocks': (gen_waves, 202), 'shore': (None, 303), 'hull': (gen_hull, 404),
    'wind': (gen_wind, 505), 'flutter': (gen_flutter, 606), 'stairwell': (gen_stairwell, 707),
    'lamproom': (gen_lamproom, 808), 'lamproom_open': (None, 909), 'beamhum': (None, 1010),
    'lamphum': (gen_lamphum, 1111), 'lensticks': (gen_lensticks, 1212), 'timelapse': (gen_timelapse, 1313),
    'glowsea': (gen_glowsea, 1414), 'calm': (gen_calm, 1515), 'dawnwind': (gen_dawnwind, 1616),
    'underglow': (gen_underglow, 1717),
}


def layer_audio(name, t0, dur, seed):
    if name == 'shore':
        return gen_waves(dur, seed, rate=0.55, slap=1.3, gurgle=1.6, trickle=1.0, pebble=0.8,
                         band_gain=(0.8, 0.8, 0.55), rise=(0.35, 0.8), decay=(0.8, 1.9), min_gap=0.6, size=0.8)
    if name == 'rocks':
        return gen_waves(dur, seed)
    if name == 'lamproom_open':
        return gen_lamproom(dur, seed, hatch=True)
    if name == 'beamhum':
        return gen_beamhum(dur, seed, t0)
    return LAYERS[name][0](dur, seed)


# per-layer loudness reference (RMS dBFS at 0 dB automation)
LAYER_REF = {'swell': -29, 'rocks': -29, 'shore': -29, 'hull': -30, 'wind': -31, 'flutter': -34,
             'stairwell': -30, 'lamproom': -30, 'lamproom_open': -30, 'beamhum': -36, 'lamphum': -33,
             'lensticks': -36, 'timelapse': -32, 'glowsea': -38, 'calm': -31, 'dawnwind': -31,
             'underglow': -38}

OFF = -90.0


def ambience_plan():
    """Keyframes (global time, dB) per ambience layer, all derived from timeline cues."""
    c = CUE
    S = {k: v['start'] for k, v in SC.items()}
    xf = {k: v['xfade'] for k, v in SC.items()}
    S1, S2, S3, S4, S5, S6, S7, S8 = (S[k] for k in ('s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08'))
    X2, X5, X8 = S2 + max(xf['s02'], 0.05), S5 + max(xf['s05'], 0.05), S8 + max(xf['s08'], 0.05)
    ext2, ts, te, sit = c['s02_exterior'], c['timelapse_start'], c['timelapse_end'], c['s02_sit']
    rd, bo, rb, ru, au, bu = c['run_down'], c['boat_out'], c['row_back'], c['run_up'], c['aim_up'], c['beam_up']
    door = door_time()
    hatch = hatch_time()
    ign, dd, sun, arr = c['ignite'], c['deng_dark'], c['sunrise'], c['s02_arrive']
    e = 0.03  # hard-cut ramp half-width
    P = {}

    def k(layer, *pts):
        P.setdefault(layer, []).extend(pts)

    # ---- s01: title in the sky -> tilt down -> push in to the island -> gallery
    k('swell', (0, OFF), (1.0, -46), (6.0, -22), (c['tilt_end'], -10), (S1 + 20.2, -8), (S2, -8), (X2, OFF))
    k('rocks', (0, OFF), (9.0, OFF), (c['tilt_end'], -26), (S1 + 17, -14), (S1 + 20.2, -6), (S2, -5), (X2, OFF))
    k('wind', (0, OFF), (2.0, -40), (c['tilt_end'], -24), (S1 + 20.2, -11), (S2, -10), (X2, OFF))
    k('beamhum', (0, OFF), (c['tilt_end'], OFF), (c['tilt_end'] + 2, -30), (S1 + 20.2, -18), (S2, -14), (X2, OFF))
    # ---- s02: stairs (interior) -> lamp room -> exterior / time-lapse -> gallery sit
    k('stairwell', (S2, OFF), (X2, -4), (arr - 0.3, -4), (arr + 0.3, OFF))
    k('lamproom', (arr - 0.3, OFF), (arr + 0.3, -3), (ext2 - e, -3), (ext2 + e, OFF))
    k('lamphum', (c['lamp_ignite'], OFF), (c['lamp_ignite'] + 0.5, -2), (ext2 - e, -2), (ext2 + e, OFF))
    k('lensticks', (c['lamp_ignite'] + 0.25, OFF), (c['lamp_ignite'] + 0.6, -2), (ext2 - e, -2), (ext2 + e, OFF))
    k('swell', (ext2 - e, OFF), (ext2 + e, -9), (ts, -9), (ts + 0.8, -15), (te - 0.4, -15), (te + 0.5, -9),
      (sit, -10), (S3, -10))
    k('rocks', (ext2 - e, OFF), (ext2 + e, -7), (ts, -7), (ts + 0.8, -14), (te - 0.4, -14), (te + 0.5, -7),
      (sit, -8), (S3, -8))
    k('wind', (ext2 - e, OFF), (ext2 + e, -16), (te, -16), (sit, -9), (S3, -10))
    k('beamhum', (ext2 - e, OFF), (ext2 + e, -14), (te, -14), (sit, -11), (S3, -12))
    k('flutter', (sit - e, OFF), (sit + e, -13), (S3, -14))
    k('timelapse', (ts - 0.2, OFF), (ts + 0.6, -3), (te - 0.6, -3), (te + 0.3, OFF))
    # ---- s03: gallery -> stairs -> door -> open sea in the boat
    for L_, v in (('swell', -10), ('rocks', -8), ('wind', -10), ('beamhum', -12), ('flutter', -15)):
        k(L_, (rd - e, v), (rd + e, OFF))
    k('glowsea', (c['impact'] + 0.5, OFF), (c['impact'] + 2.5, -14), (rd - e, -14), (rd + e, OFF))
    k('stairwell', (rd - e, OFF), (rd + e, -3), (door - 0.01, -3), (door + 0.05, OFF))
    k('rocks', (door - 0.01, OFF), (door + 0.05, -3), (bo - e, -3), (bo + e, -22), (S4 - 0.05, -24), (S4, OFF))
    k('swell', (door - 0.01, OFF), (door + 0.05, -9), (bo, -6), (S4 - e, -6), (S4 + e, -10))
    k('wind', (door - 0.01, OFF), (door + 0.05, -14), (bo, -19), (S4 - e, -19), (S4 + e, -25))
    k('hull', (bo - e, OFF), (bo + e, -4), (S4 - e, -4), (S4 + e, -2))
    k('glowsea', (bo - e, OFF), (bo + e, -8), (S4 - e, -8), (S4 + e, -12))
    k('underglow', (bo - e, OFF), (bo + 1.5, -8), (S4 - e, -6), (S4 + e, -4), (c['lift'] + 0.4, -6),
      (c['lift'] + 3.0, OFF))
    # ---- s04: intimate close shots at sea -> row back (wide) -> crossfade into s05
    k('hull', (rb - e, -2), (rb + e, -9), (S5, -9), (X5, OFF))
    k('swell', (rb - e, -10), (rb + e, -7), (S5, -8), (X5, -11))
    k('wind', (rb - e, -25), (rb + e, -20), (S5, -20), (X5, -23))
    k('glowsea', (c['lift'] + 3, -16), (c['determined'], -18), (rb - e, -18), (rb + e, -12), (S5, -12), (X5, OFF))
    k('rocks', (rb - e, OFF), (rb + e, -26), (S5, -20), (X5, -15))
    # ---- s05: the island rocks (shore lapping), dawn creeping in
    k('shore', (S5, OFF), (X5, 0), (c['sit_together'], -1), (S6, -2))
    k('beamhum', (S5, OFF), (X5, -26), (S6, -22))
    k('rocks', (c['sit_together'], -16), (S6, -15))
    k('swell', (c['sit_together'], -12), (S6, -11))
    k('dawnwind', (c['dawn_warning'] - 2, OFF), (S6, -18))
    # ---- s06: base (exterior) -> stairs -> lamp room (hatch opens) -> ignition (exterior) -> dark
    for L_, v in (('shore', -2), ('beamhum', -22), ('rocks', -15), ('swell', -11), ('wind', -20), ('dawnwind', -18)):
        k(L_, (ru - e, v), (ru + e, OFF))
    k('wind', (S6, -20))
    k('stairwell', (ru - e, OFF), (ru + e, -4), (au - 0.25, -4), (au + 0.2, OFF))
    k('lamphum', (ru, OFF), (ru + 0.3, -26), (au - 0.1, -12), (au + 0.1, -3), (ign - 0.05, -3), (ign + 0.05, OFF))
    k('lamproom', (au - 0.25, OFF), (au + 0.2, -3), (hatch, -3), (hatch + 0.5, OFF))
    k('lamproom_open', (hatch, OFF), (hatch + 0.5, -4), (ign, -4), (ign + 0.4, -14), (dd, -15), (dd + 1.6, OFF))
    k('rocks', (ign, OFF), (ign + 0.3, -9), (dd, -11), (dd + 1.6, OFF))
    k('swell', (ign, OFF), (ign + 0.3, -9), (dd, -10), (dd + 1.8, -50), (S7, -50))
    k('wind', (ru + e, OFF), (ign, OFF), (ign + 0.3, -13), (dd, -15), (dd + 1.6, OFF))
    # ---- s07: near silence, soft water -> whale -> sunrise (warm wind) -> crossfade to s08 night
    k('swell', (S7 + 0.1, -30), (c['whale_song'], -24), (sun, -20), (sun + 3, -12), (S8, -12))
    k('calm', (S7, OFF), (S7 + 0.25, -10), (c['whale_song'], -7), (sun, -7), (S8, -9), (X8, OFF))
    k('dawnwind', (sun - 0.5, OFF), (sun + 3.5, -8), (S8, -8), (X8, OFF))
    k('rocks', (sun, OFF), (sun + 3, -17), (S8, -16))
    k('flutter', (sun + 1, OFF), (sun + 3, -20), (S8, -20), (X8, OFF))
    # ---- s08: night again, crane up & away, end
    fo, wv, et = c['fade_out'], c['wave'], c['end_title']
    k('swell', (X8, -10), (wv, -10), (et, -15), (fo, -18), (DUR, OFF))
    k('rocks', (X8, -10), (wv, -10), (wv + 5, -19), (et, -22), (fo, -24), (DUR, OFF))
    k('wind', (S8, OFF), (X8, -18), (wv, -18), (wv + 5, -13), (et, -15), (fo, -18), (DUR, OFF))
    k('beamhum', (S8, OFF), (X8, -18), (wv, -18), (wv + 5, -23), (fo, -26), (DUR, OFF))
    for L_ in P:
        P[L_].sort(key=lambda p: p[0])
    return P


def auto_curve(keys, tgrid):
    """Equal-power (power-domain smoothstep) interpolation of (t, dB) keyframes."""
    ts = np.array([k_[0] for k_ in keys])
    pw = np.array([0.0 if k_[1] <= OFF + 1 else 10 ** (k_[1] / 10) for k_ in keys])
    idx = np.clip(np.searchsorted(ts, tgrid, side='right') - 1, 0, len(ts) - 2)
    span = np.maximum(ts[idx + 1] - ts[idx], 1e-9)
    u = smoothstep((tgrid - ts[idx]) / span)
    p = pw[idx] + (pw[idx + 1] - pw[idx]) * u
    p = np.where(tgrid < ts[0], pw[0], np.where(tgrid > ts[-1], pw[-1], p))
    return np.sqrt(np.maximum(p, 0))


def active_spans(curve_ctl, cr, pad=0.6):
    on = curve_ctl > 1e-4
    spans = []
    i = 0
    n = len(on)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < n and on[j]:
            j += 1
        spans.append((max(0.0, i / cr - pad), min(DUR, j / cr + pad)))
        i = j
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1] + 1.0:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return merged


def door_time():
    return min(cue('run_down') + 3.0, cue('boat_out') - 0.3)


def hatch_time():
    return min(cue('aim_up') + 2.1, cue('beam_up') - 0.5)


def render_ambience(log=print):
    plan = ambience_plan()
    out = np.zeros((N, 2))
    tg = np.arange(int(DUR * CR) + 1) / CR
    for name, keys in plan.items():
        curve = auto_curve(keys, tg)
        spans = active_spans(curve, CR)
        for si, (a, b) in enumerate(spans):
            t_ = time.time()
            seed = LAYERS[name][1] * 10 + si
            y = layer_audio(name, a, b - a, seed)
            y = hp(y, 25)
            y = norm_rms(y, LAYER_REF[name])
            i0 = ns(a)
            n = min(len(y), N - i0)
            g = ctl2audio(curve, N)[i0:i0 + n] if False else np.interp(
                (i0 + np.arange(n)) / SR, tg, curve)
            out[i0:i0 + n] += y[:n] * g[:, None]
            log(f'  ambience {name:14s} {a:7.2f}-{b:7.2f}s  ({time.time() - t_:.1f}s)')
    return out


# ----------------------------------------------------------------------------------------------
# SPOT EFFECTS
# ----------------------------------------------------------------------------------------------
IRON = np.array([412, 655, 988, 1433, 2140, 2870, 3720, 5130, 6450])


def footstep(seed, weight=1.0, surface='iron', hurry=False, creaky=False, oilcan=False):
    r = rng(seed)
    dur = 0.9 if surface == 'iron' else 0.4
    n = ns(dur)
    t = np.arange(n) / SR
    f0 = r.uniform(68, 92) * (1.12 if hurry else 1.0)
    y = weight * 0.9 * np.sin(TAU * np.cumsum(f0 * (1 + 0.9 * np.exp(-t / 0.012))) / SR) * perc(n, 0.002, 0.05)
    k = ns(0.006)
    cl = white(k, r) * np.exp(-np.arange(k) / SR / 0.0009)
    y[:k] += 0.35 * hp(cl, 1500)
    if surface == 'iron':
        fr = IRON * r.uniform(0.96, 1.04, len(IRON)) * r.uniform(0.9, 1.1)
        t60 = np.linspace(0.6, 0.12, len(IRON)) * r.uniform(0.8, 1.2)
        amps = r.uniform(0.25, 1.0, len(IRON)) / (1 + np.arange(len(IRON)) * 0.3)
        y += 0.32 * modal(fr, t60, amps, dur, hard=0.7 if not hurry else 1.0, r=r)
        y += 0.22 * modal(np.array([930, 1570, 2320, 3110]) * r.uniform(0.95, 1.05), [0.08, 0.06, 0.05, 0.04],
                          [1, 0.7, 0.5, 0.3], dur)
    elif surface == 'stone':
        m = ns(0.06)
        y[:m] += 0.25 * bp(white(m, r), 1200, 5000) * perc(m, 0.002, 0.018)
        y += 0.3 * np.sin(TAU * 60 * t) * perc(n, 0.002, 0.03)
    elif surface == 'floor':  # riveted iron floor plate of the lamp room
        y += 0.25 * modal(np.array([260, 470, 810, 1330, 2050]) * r.uniform(0.95, 1.05),
                          [0.35, 0.3, 0.2, 0.15, 0.1], [1, .8, .6, .4, .2], dur, hard=0.5, r=r)
    if hurry:
        for _ in range(r.integers(2, 5)):
            tr = r.uniform(0.02, 0.14)
            ping = modal([r.uniform(2000, 6000), r.uniform(3000, 7500)], [0.03, 0.02], [1, 0.5], 0.08)
            mix_into(y, ping * r.uniform(0.08, 0.22), ns(tr))
    if creaky:
        mix_into(y, creak(r.uniform(0.25, 0.4), int(r.integers(1e6)), rate=(40, 22), body=(600, 1300, 2500),
                          Q=16) * 0.25, ns(0.04))
    if oilcan:
        tink = modal([r.uniform(2400, 2600), r.uniform(5200, 5600)], [0.12, 0.06], [1, .4], 0.2)
        mix_into(y, tink * 0.06, ns(r.uniform(0.03, 0.07)))
    return y


def oil_can(seed):
    r = rng(seed)
    n = ns(0.45)
    y = np.zeros(n)
    for dt, f, sgn in ((0.0, 1180, -1), (0.14, 1260, 1)):
        m = ns(0.25)
        tt = np.arange(m) / SR
        f_ = f * (1 + sgn * 0.06 * (1 - np.exp(-tt / 0.012)))
        s_ = (np.sin(TAU * np.cumsum(f_) / SR) + 0.35 * np.sin(TAU * np.cumsum(f_ * 2.31) / SR)) * perc(m, 0.001, 0.045)
        mix_into(y, s_ * (1 if dt == 0 else 0.7), ns(dt))
    sq = hp(lp(white(ns(0.07), r), 7000), 2800) * perc(ns(0.07), 0.005, 0.02) * 0.2
    mix_into(y, sq, ns(0.02))
    return y


def squeak(dur, f0, seed):
    """Mitten polishing glass: pitched stick-slip squeak."""
    r = rng(seed)
    n = ns(dur)
    t = np.arange(n) / SR
    u = t / dur
    f = f0 * (1 + 0.1 * np.sin(np.pi * u) + 0.025 * np.sin(TAU * r.uniform(8, 14) * t))
    x = saw(f)
    rough = 1 + 0.35 * lp(white(n, r), 150) * 4
    x = bp(x * rough, f0 * 0.8, min(f0 * 5, 12000))
    return x * np.sin(np.pi * u) ** 1.2


def lever(seed):
    """Lever pull: ratchet creak then heavy clunk (sync at the clunk, 0.32 s in)."""
    r = rng(seed)
    n = ns(1.5)
    y = np.zeros(n)
    cr_ = creak(0.3, seed + 1, rate=(28, 14), body=(700, 1500, 3100), Q=18)
    mix_into(y, cr_ * 0.35, 0)
    for k in range(3):
        mix_into(y, modal([r.uniform(3000, 3600), 5200], [0.03, 0.02], [1, .4], 0.06) * 0.15, ns(0.05 + 0.08 * k))
    m = ns(1.1)
    tt = np.arange(m) / SR
    th = np.sin(TAU * np.cumsum(62 * (1 + 0.8 * np.exp(-tt / 0.01))) / SR) * perc(m, 0.002, 0.07)
    ring = modal(np.array([182, 391, 718, 1150, 1720, 2610, 3480]) * r.uniform(0.97, 1.03),
                 [0.7, 0.55, 0.45, 0.35, 0.25, 0.18, 0.12], [1, .9, .7, .5, .35, .25, .15], 1.1, hard=0.6)
    mix_into(y, 0.9 * th + 0.45 * ring, ns(0.32))
    return y


def whoomp(seed):
    """Lamp ignition: two igniter sparks then the 'whoomp' (sync = 0.15 s)."""
    r = rng(seed)
    n = ns(2.2)
    y = np.zeros(n)
    for dt in (0.0, 0.07):
        m = ns(0.03)
        sp = bp(white(m, r), 2500, 9000) * perc(m, 0.0005, 0.004)
        zap = saw(r.uniform(90, 140), m) * perc(m, 0.001, 0.01) * 0.3
        mix_into(y, sp * 0.5 + bp(zap, 200, 3000) * 0.4, ns(dt))
    m = ns(2.0)
    tt = np.arange(m) / SR
    fc = 140 + 1600 * np.exp(-((tt - 0.12) / 0.09) ** 2) + 380 * (1 - np.exp(-tt / 0.3))
    fw = tvf(pink(m, r), 'lp', fc, 1.1, block=64) * perc(m, 0.07, 0.4)
    sub = np.sin(TAU * np.cumsum(40 + 22 * (1 - np.exp(-tt / 0.08))) / SR) * perc(m, 0.03, 0.45)
    glow = sum(np.sin(TAU * note(s_ - 12) * tt) for s_ in (0, 7, 12)) * np.clip((tt - 0.1) / 0.6, 0, 1) * np.exp(-tt / 0.9) * 0.08
    mix_into(y, fade(fw * 1.2 + sub * 0.8 + glow, 0.001, 0.3), ns(0.12))
    return y


def sigh_servo(seed):
    r = rng(seed)
    n = ns(1.6)
    t = np.arange(n) / SR
    y = servo(1.2, 470, 235, seed, bright=0.8, curve=lambda u: 1 - (1 - u) ** 2) * bell_curve(ns(1.2), 0.2, 0.8)
    y = np.concatenate([y, np.zeros(n - len(y))])
    breath = bp(white(n, r), 800, 4200) * bell_curve(n, 0.18, 1.5) * 0.35
    y += breath
    thunk = modal([150, 320, 610], [0.12, 0.09, 0.06], [1, .6, .3], 0.3, hard=0.3) * 0.5
    mix_into(y, thunk, ns(0.95))
    return y


def twinkle(f0, seed, warble=0.0, dur=1.4):
    r = rng(seed)
    n = ns(dur)
    t = np.arange(n) / SR
    wob = 1 + warble * np.sin(TAU * 7.5 * t) * np.exp(-t / 0.5)
    y = np.zeros(n)
    for k, (p, a, tau) in enumerate(((1, 1, 0.45), (2.0, 0.3, 0.25), (3.01, 0.12, 0.12))):
        y += a * np.sin(TAU * np.cumsum(f0 * p * wob) / SR) * np.exp(-t / tau)
    y *= perc(n, 0.004, 10)
    sp = sparkles(dur * 0.6, seed + 1, 18, freqs=scale_freqs(f0 * 1.3, f0 * 3.2), tau=(0.02, 0.08),
                  pan_fn=lambda u, r_: r_.uniform(-0.3, 0.3))
    out = pan(y, r.uniform(-0.2, 0.2))
    mix_into(out, sp * 0.25, 0)
    return out


def meteor(dur, seed):
    """Shimmering whoosh that rises in pitch and loudness, crackling sparks; sync = end (impact)."""
    r = rng(seed)
    n = ns(dur)
    t = np.arange(n) / SR
    u = t / dur
    # loudness: soft at first (dialogue D03 sits in the middle), then a big swell into the impact
    L = db(-30 + 12 * smoothstep(u / 0.6) + 18 * smoothstep((u - 0.55) / 0.45) ** 1.5)
    fc = 350 * (3200 / 350) ** (u ** 1.3)
    air = tvf(pink(n, r), 'bp', fc, 1.4, block=128) + 0.5 * tvf(white(n, r), 'bp', fc * 2.2, 2.5, block=128)
    shim = np.zeros(n)
    base = 520 * 2.0 ** (1.3 * u ** 1.4)
    for k, p in enumerate((1, 1.5, 2, 2.5, 3, 4)):
        am = 0.6 + 0.4 * np.sin(TAU * r.uniform(11, 23) * t + r.uniform(0, 6))
        shim += np.sin(TAU * np.cumsum(base * p * (1 + 0.004 * k)) / SR) * am / (1 + 0.3 * k)
    sizzle = hp(white(n, r), 4500) * (0.5 + 0.5 * lp(np.abs(white(n, r)), 30)) * 0.4
    rumble = lp(pink(n, r), 110) * smoothstep((u - 0.6) / 0.4) ** 2 * 3.0
    mono = (air * 1.0 + shim * 0.3 + sizzle * u + rumble) * L
    p_ = -0.65 + 0.9 * smoothstep(u)
    out = pan(mono, p_)
    cr_ = crackle(dur, seed + 1, lambda uu: 4 + 90 * uu ** 2, lo=1800, hi=9000, width=0.25)
    Lc = L[:, None] * np.ones((1, 2))
    m = min(len(cr_), n)
    out[:m] += cr_[:m] * Lc[:m] * 1.4 * pan(np.ones(m), p_[:m]) * 1.4
    k = ns(0.012)
    out[-k:] *= np.linspace(1, 0, k)[:, None]
    return out


def impact_boom(seed):
    r = rng(seed)
    n = ns(4.0)
    t = np.arange(n) / SR
    sub = np.sin(TAU * np.cumsum(32 + 42 * np.exp(-t / 0.35)) / SR) * perc(n, 0.006, 1.1)
    thoom = lp(pink(n, r), 180) * perc(n, 0.004, 0.6) * 2.0
    mid = lp(white(n, r), 700) * perc(n, 0.002, 0.12) * 1.2
    return (sub * 1.0 + thoom * 0.8 + mid * 0.5)


def magic_bells(seed, chord=(0, 7, 12, 16, 19)):
    r = rng(seed)
    n = ns(6.0)
    y = np.zeros(n)
    for i, s_ in enumerate(chord):
        b = bell(note(s_ + 12), t60=r.uniform(3.5, 5.0), seed=seed + i, amp_hum=0.25)
        mix_into(y, b * r.uniform(0.6, 1.0) / (1 + 0.15 * i), ns(0.03 * i + r.uniform(0, 0.02)))
    return y


def biolum_rings(dur, seed, ring_times=(0.15, 0.8, 1.55, 2.4, 3.4)):
    def dens(u):
        tt = u * dur
        d = np.zeros_like(u)
        for rt_ in ring_times:
            d += 55 * np.exp(-((tt - rt_ - 0.25) / 0.3) ** 2) * np.exp(-rt_ / 3.0)
        return d + 3

    def pan_fn(u, r):
        tt = u * dur
        k_ = max([rt_ for rt_ in ring_times if rt_ <= tt] or [0])
        spread = min(1.0, 0.25 + (tt - k_) * 1.2)
        return r.uniform(-spread, spread)

    return sparkles(dur, seed, dens, freqs=scale_freqs(1800, 7500), tau=(0.04, 0.35), glassy=0.45,
                    pan_fn=pan_fn, env_fn=lambda u: np.exp(-u * 1.2))


def bleep(freqs, seed, seg=0.065, gap=0.035, square=0.3, chirp=0.08):
    n = ns(len(freqs) * (seg + gap) + 0.05)
    y = np.zeros(n)
    for i, f in enumerate(freqs):
        m = ns(seg)
        tt = np.arange(m) / SR
        f_ = f * (1 + chirp * (1 - np.exp(-tt / 0.01)))
        ph = TAU * np.cumsum(f_) / SR
        s_ = (np.sin(ph) + square * np.sin(3 * ph) / 3 + square * 0.5 * np.sin(5 * ph) / 5) * ar_env(m, 0.003, 0.02)
        mix_into(y, s_, ns(i * (seg + gap)))
    return y


def door_bang(seed):
    """Wooden door flung open against stone: latch clack, hinge squeal, bang, bounce (sync 0.06 s)."""
    r = rng(seed)
    n = ns(1.6)
    y = np.zeros(n)
    latch = modal([1850, 3100, 4700], [0.05, 0.04, 0.03], [1, .6, .4], 0.1) * 0.35
    mix_into(y, latch, 0)
    hinge = creak(0.12, seed + 1, rate=(420, 380), body=(1400, 2800), Q=25) * 0.25
    mix_into(y, hinge, ns(0.005))
    m = ns(1.2)
    tt = np.arange(m) / SR
    wood = modal(np.array([96, 182, 263, 418, 612, 847, 1210]) * r.uniform(0.97, 1.03),
                 [0.45, 0.35, 0.3, 0.22, 0.16, 0.12, 0.08], [1, .9, .8, .6, .4, .3, .2], 1.2, hard=0.8)
    thud = np.sin(TAU * np.cumsum(70 * (1 + np.exp(-tt / 0.01))) / SR) * perc(m, 0.002, 0.08)
    slap = bp(white(m, r), 500, 4000) * perc(m, 0.001, 0.015)
    mix_into(y, wood * 0.8 + thud * 0.9 + slap * 0.6, ns(0.06))
    for dt, g in ((0.24, 0.3), (0.38, 0.12)):
        mix_into(y, wood[:ns(0.5)] * g * np.linspace(1, 0.3, ns(0.5)), ns(0.06 + dt))
    return y


def oar_stroke(seed, distance=0.0):
    """Blade enters (0 s), pull swirl + oarlock creak, blade exits (~1.0 s) with drips."""
    r = rng(seed)
    n = ns(1.9)
    y = np.zeros(n)
    t = np.arange(n) / SR
    plunk = noise_burst(0.12, 0.03, 180, 1300, r, attack=0.004) * 1.0
    mix_into(y, plunk, 0)
    mix_into(y, bubble(r.uniform(380, 560), 0.03, 0.4, 0.45), ns(0.004))
    for _ in range(8):
        mix_into(y, bubble(r.uniform(500, 1600), r.uniform(0.005, 0.02), 0.4, 0.08), ns(r.uniform(0.01, 0.2)))
    m = ns(0.95)
    swirl = tvf(pink(m, r), 'bp', 380 + 500 * np.sin(np.pi * np.linspace(0, 1, m)), 1.5, block=128)
    mix_into(y, swirl * bell_curve(m, 0.35, 1.2) * 0.45, ns(0.05))
    for _ in range(6):
        mix_into(y, bubble(r.uniform(300, 900), r.uniform(0.01, 0.03), 0.2, 0.06), ns(r.uniform(0.2, 0.8)))
    ck = creak(r.uniform(0.35, 0.5), seed + 7, rate=(35, 60, 30), body=(430, 980, 1900), Q=11)
    mix_into(y, ck * 0.18, ns(0.12))
    knock = modal([150, 280, 460], [0.1, 0.08, 0.06], [1, .6, .4], 0.2, hard=0.25) * 0.25
    mix_into(y, knock, ns(0.3))
    ex = noise_burst(0.2, 0.05, 900, 6000, r, attack=0.01) * 0.35
    mix_into(y, ex, ns(1.0))
    mix_into(y, drips(0.7, seed + 9, lambda u: 14 * (1 - u) + 1, amp=0.35), ns(1.05))
    if distance > 0:
        y = lp(y, 5000 - 3000 * distance) * (1 - 0.4 * distance)
    return y


def heart_hum(dur, seed, beat=True, bpm=62):
    r = rng(seed)
    n = ns(dur)
    t = np.arange(n) / SR
    y = np.zeros(n)
    for s_, a in ((-12, 1.0), (-5, 0.35), (0, 0.4), (4, 0.12), (7, 0.1)):
        f = note(s_ - 12)
        y += a * (np.sin(TAU * f * t + r.uniform(0, 6)) + np.sin(TAU * (f + r.uniform(0.2, 0.5)) * t))
    y = lp(y, 900) * 0.5
    shimmer = sparkles(dur, seed + 1, 1.5, freqs=scale_freqs(1400, 3500), tau=(0.2, 0.5), glassy=0.1,
                       attack=0.05, pan_fn=lambda u, r_: r_.uniform(-0.4, 0.4))[:n]
    out = pan(y, 0) * 1.4 + shimmer * 0.12
    if beat:
        tb = 0.4
        while tb < dur - 0.7:
            hb = heartbeat(seed, 0.35)
            m = min(len(hb), n - ns(tb))
            out[ns(tb):ns(tb) + m] += pan(hb[:m], 0)
            tb += 60.0 / bpm
    return out * ar_env(n, dur * 0.25, dur * 0.35)[:, None]


def glassy_flicker(dur, seed, f0=3150.0, strength=1.0):
    """Thin fragile flicker: glass partials with irregular on/off tremolo + tiny electric ticks."""
    r = rng(seed)
    n = ns(dur)
    nc = int(dur * CR) + 4
    t = np.arange(n) / SR
    tel = (smooth_noise(nc, 14, r) > 0.1).astype(float)
    tel = np.convolve(tel, np.hanning(9) / np.hanning(9).sum(), 'same')
    am = ctl2audio(tel, n) * (0.6 + 0.4 * ctl2audio(smooth_noise(nc, 30, r), n))
    y = np.zeros(n)
    for p, a in ((1.0, 1.0), (1.47, 0.5), (1.93, 0.35), (2.76, 0.2)):
        y += a * np.sin(TAU * f0 * p * (1 - 0.03 * t / dur) * t + r.uniform(0, 6))
    y *= am * np.exp(-t / (dur * 0.8))
    cr_ = crackle(dur, seed + 1, 25, lo=3000, hi=10000, width=0.3)[:n]
    return pan(y * 0.25, r.uniform(-0.1, 0.1)) * strength + cr_ * 0.03 * strength


def boing(seed, f0=330.0):
    r = rng(seed)
    n = ns(0.8)
    t = np.arange(n) / SR
    f = f0 * (1 + 0.28 * np.sin(TAU * 12 * t) * np.exp(-t / 0.22)) * (1 + 0.15 * np.exp(-t / 0.04))
    y = np.sin(TAU * np.cumsum(f) / SR + 1.2 * np.exp(-t / 0.1) * np.sin(TAU * np.cumsum(f * 2.01) / SR))
    return y * perc(n, 0.002, 0.25)


def bonk(seed):
    r = rng(seed)
    return modal(np.array([420, 985, 1650, 2410, 3330]) * r.uniform(0.97, 1.03), [0.3, 0.22, 0.16, 0.12, 0.08],
                 [1, .7, .45, .3, .15], 0.5, hard=0.6)


def wood_thunk(seed, kind='crate'):
    r = rng(seed)
    if kind == 'crate':
        fr = np.array([128, 215, 347, 522, 760, 1120]) * r.uniform(0.9, 1.1)
        y = modal(fr, [0.16, 0.12, 0.1, 0.07, 0.05, 0.04], [1, .8, .7, .5, .3, .2], 0.45, hard=0.5, r=r)
    else:  # barrel: hollow, longer
        fr = np.array([158, 331, 522, 781, 1090]) * r.uniform(0.92, 1.08)
        y = modal(fr, [0.42, 0.32, 0.22, 0.14, 0.1], [1, .8, .5, .35, .2], 0.7, hard=0.4, r=r)
    n = len(y)
    y += 0.4 * lp(white(n, r), 900) * perc(n, 0.002, 0.02)
    return y


def splinter(seed):
    r = rng(seed)
    n = ns(0.35)
    y = np.zeros(n)
    for _ in range(r.integers(5, 12)):
        m = ns(0.03)
        c = bp(white(m, r), 1500, 6500) * perc(m, 0.0005, r.uniform(0.002, 0.008))
        mix_into(y, c * r.lognormal(0, 0.5), ns(abs(r.normal(0, 0.06))))
    return y


def rattle(seed, dur=0.5, rate=20):
    r = rng(seed)
    n = ns(dur)
    y = np.zeros(n + ns(0.2))
    t = 0.0
    while t < dur:
        t += r.exponential(1 / rate)
        y_ = wood_thunk(int(r.integers(1e6)), 'crate')[:ns(0.12)] * r.uniform(0.05, 0.2)
        mix_into(y, y_, ns(t))
    return y


def fm_chime(f, dur=1.2, seed=0):
    return fm_tone(f, dur, ratio=3.5, index=2.0, tau=dur / 3.5, idx_tau=0.05) + \
        0.3 * fm_tone(f * 2, dur, ratio=1.4, index=1.0, tau=dur / 6, idx_tau=0.03)


def gear_grind(dur, seed, rate=(18, 42)):
    r = rng(seed)
    n = ns(dur)
    rt = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(rate)), rate)
    rt *= 1 + 0.08 * ctl2audio(smooth_noise(int(dur * CR) + 4, 5, r), n)
    ph = np.cumsum(rt) / SR
    imp = np.zeros(n)
    idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0]
    imp[idx] = r.uniform(0.5, 1.0, len(idx))
    y = norm_peak(sum(g * reson(imp, f, 14) for f, g in ((720, 1), (1310, .7), (2250, .5), (3400, .3))))
    grind = norm_peak(bp(white(n, r), 900, 4000) * (0.5 + 0.5 * np.sin(TAU * ph)) ** 3)
    return y * 0.8 + grind * 0.3


def ratchet(dur, seed, rate=9):
    r = rng(seed)
    n = ns(dur)
    y = np.zeros(n + ns(0.1))
    t = 0.02
    while t < dur:
        c = modal(np.array([2600, 4100, 5900]) * r.uniform(0.97, 1.03), [0.03, 0.025, 0.02], [1, .6, .3], 0.06)
        c += 0.5 * modal([520, 900], [0.05, 0.04], [1, .5], 0.06, hard=0.4)
        mix_into(y, c * r.uniform(0.7, 1.0), ns(t))
        t += 1 / rate * r.uniform(0.85, 1.15)
    return y[:n]


def iron_groan(dur, seed):
    return creak(dur, seed, rate=(9, 22, 14), body=(92, 171, 310, 520, 890), Q=9, jitter=0.4,
                 gains=[1, .9, .7, .5, .3], shape=0.4)


def hatch_clank(seed):
    """Roof hatch: unlatch clank, hinge groan, swing open thud (sync 0)."""
    r = rng(seed)
    n = ns(1.8)
    y = np.zeros(n)
    mix_into(y, modal(np.array([240, 510, 880, 1390, 2150, 3050]) * r.uniform(.97, 1.03),
                      [0.6, 0.5, 0.35, 0.25, 0.18, 0.1], [1, .9, .7, .5, .3, .2], 1.0, hard=0.8) * 0.7, 0)
    mix_into(y, creak(0.35, seed + 1, rate=(60, 110, 80), body=(640, 1250, 2400), Q=20) * 0.3, ns(0.08))
    m = ns(0.8)
    tt = np.arange(m) / SR
    thud = np.sin(TAU * np.cumsum(55 * (1 + np.exp(-tt / 0.01))) / SR) * perc(m, 0.002, 0.1)
    mix_into(y, thud * 0.8 + 0.3 * modal([190, 400, 720], [0.4, 0.3, 0.2], [1, .7, .4], 0.8, hard=0.5), ns(0.42))
    return y


def electric_rise(seed, dur=2.6):
    """Beam up: rising electric hum + upward 'fwoosh'; sync 0 (peak ~0.8 s)."""
    r = rng(seed)
    n = ns(dur)
    t = np.arange(n) / SR
    u = t / dur
    f = 50 + 70 * smoothstep(t / 0.9)
    ph = TAU * np.cumsum(f) / SR
    hum = sum(a * np.sin(k * ph) for k, a in ((1, 1), (2, .8), (3, .6), (4, .45), (6, .3), (8, .2)))
    hum = np.tanh(1.8 * hum) * 0.6
    env = np.clip(t / 0.7, 0, 1) ** 1.5 * np.exp(-np.maximum(t - 0.9, 0) / 0.9)
    fw = tvf(pink(n, r), 'bp', 250 * 14 ** smoothstep(t / 1.4), 1.3, block=128) * bell_curve(n, 0.3, 1.2)
    zz = crackle(dur, seed + 1, lambda uu: 40 * np.exp(-uu * 3), lo=2000, hi=9000, width=0.5)[:n]
    out = pan(hum * env + fw * 0.8, 0) + zz * 0.15
    return out


def power_down(dur, seed):
    """Slow power-down whine of a weakening robot (with stutters)."""
    r = rng(seed)
    n = ns(dur)
    t = np.arange(n) / SR
    u = t / dur
    f = 880 * (95 / 880) ** (u ** 0.85)
    nc = int(dur * CR) + 4
    f = f * (1 + 0.02 * ctl2audio(smooth_noise(nc, 3, r), n))
    stut = np.clip(1 - 0.8 * (ctl2audio(smooth_noise(nc, 2.2, r), n) > 1.1), 0.2, 1)
    stut = lp(stut, 12)
    x = 0.6 * np.sin(TAU * np.cumsum(f) / SR) + 0.25 * saw(f) + 0.15 * np.sin(TAU * np.cumsum(f * 3.01) / SR)
    x = lp(x, 2500)
    return x * stut * (1 - 0.6 * u) * ar_env(n, 0.8, 1.0)


def ignition(seed, dur=11.0):
    """IGNITION: sub boom, massive whoosh, crackling energy, wind rush, sustained rumble.
    Returns stereo (sync 0).  The choir resonance is a separate event."""
    r = rng(seed)
    n = ns(dur)
    t = np.arange(n) / SR
    out = np.zeros((n, 2))
    sub = np.sin(TAU * np.cumsum(27 + 36 * np.exp(-t / 0.4)) / SR) * perc(n, 0.004, 1.6)
    boom = lp(pink(n, r), 140) * perc(n, 0.003, 0.9) * 2
    out += pan(sub * 1.0 + boom * 0.7, 0)
    for ch in range(2):
        fc = 700 + 13000 * np.exp(-t / 0.7)
        wh = tvf(white(n, r), 'lp', fc, 0.7, block=128) * perc(n, 0.02, 1.4)
        out[:, ch] += wh * 0.55
    crack = hp(white(ns(0.06), r), 1200) * perc(ns(0.06), 0.0005, 0.012)
    mix_into(out, pan(crack * 1.2, 0), 0)
    cr_ = crackle(dur, seed + 1, lambda u: 160 * np.exp(-u * dur / 1.2) + 14 * (u < 10.2 / dur), lo=900,
                  hi=9500, width=0.95)[:n]
    out += cr_ * 0.5
    for k in range(7):
        tz = abs(r.normal(0.4, 0.5))
        m = ns(r.uniform(0.05, 0.14))
        zap = bp(saw(r.uniform(70, 190) * (1 + 0.3 * np.linspace(0, 1, m)), m), 150, 5000) * ar_env(m, 0.003, 0.03)
        mix_into(out, pan(zap * 0.25, r.uniform(-0.8, 0.8)), ns(tz))
    wind = whoosh(4.5, 300, 1500, seed + 2, peak=0.2, Q=0.8, pan0=-0.7, pan1=0.8)
    mix_into(out, wind * 1.2, ns(0.05))
    rum = np.zeros((n, 2))
    for ch in range(2):
        rum[:, ch] = lp(pink(n, r), 95) * 1.6 + 0.35 * np.sin(TAU * 55 * t + ch) * (1 + 0.3 * np.sin(TAU * 0.3 * t))
    re = np.clip(t / 0.3, 0, 1) * (0.55 + 0.45 * np.exp(-t / 2.0))
    out += rum * re[:, None] * 0.5
    return out


def whale_call(dur, contour, seed, formant=((0, 380), (0.5, 900), (1, 520)), sub=0.35, star=0.12,
               vib=0.006, rough=0.0):
    """Celestial whale: formant-filtered polyBLEP moan with pitch glide, sub body and a star-voice."""
    r = rng(seed)
    n = ns(dur)
    t = np.arange(n) / SR
    u = t / dur
    cu, cf = zip(*contour)
    f = CubicSpline(cu, np.log(cf), bc_type='clamped')(u) if len(cu) > 2 else np.interp(u, cu, np.log(cf))
    f = np.exp(f)
    nc = int(dur * CR) + 4
    f *= 1 + vib * np.sin(TAU * 4.3 * t) * smoothstep(u / 0.5) + 0.004 * ctl2audio(smooth_noise(nc, 3, r), n)
    src = 0.22 * lp(saw(f), 2400) + 0.9 * np.sin(TAU * np.cumsum(f) / SR) + \
        0.25 * np.sin(TAU * np.cumsum(f * 2) / SR) + sub * np.sin(TAU * np.cumsum(f * 0.5) / SR)
    if rough:
        src *= 1 + rough * np.sin(TAU * np.cumsum(f * 0.25) / SR)
    fu, ff = zip(*formant)
    F1 = np.exp(np.interp(u, fu, np.log(ff)))
    y = tvf(src, 'bp', F1, 2.5, block=64) * 1.6 + 0.7 * tvf(src, 'bp', F1 * 2.1, 3.5, block=64)
    y += 0.25 * lp(src, 400)
    y += star * (np.sin(TAU * np.cumsum(f * 4) / SR) + 0.5 * np.sin(TAU * np.cumsum(f * 6) / SR)) * \
        (0.5 + 0.5 * np.sin(TAU * 0.7 * t))
    env = np.sin(np.pi * np.clip(u, 0, 1)) ** 0.8 * (0.85 + 0.15 * np.sin(TAU * 1.1 * t))
    return y * env


def stardust(dur, seed):
    def dens(u):
        return 25 + 60 * np.sin(np.pi * np.clip(u, 0, 1)) + 70 * smoothstep((u - 0.75) / 0.25)

    def pan_fn(u, r):
        c = 0.7 * np.sin(TAU * 1.2 * u * dur / 4.0) * (1 - smoothstep((u - 0.7) / 0.3))
        return c + r.normal(0, 0.25 * (1 - 0.7 * u))

    sp = sparkles(dur, seed, dens, freqs=scale_freqs(1500, 9000), tau=(0.05, 0.35), glassy=0.35,
                  pan_fn=pan_fn, pitch_fn=lambda u: 1 + 0.25 * smoothstep((u - 0.75) / 0.25),
                  env_fn=lambda u: 0.5 + 0.5 * smoothstep(u / 0.2))
    r = rng(seed + 1)
    n = len(sp)
    hiss = hp(np.stack([pink(n, r), pink(n, r)], 1), 5000) * 0.05
    hiss *= bell_curve(n, 0.5, 1.0)[:, None]
    return sp + hiss


def rising_sparkle(dur, seed, f_lo=1200, f_hi=6000, dens=45):
    return sparkles(dur, seed, lambda u: dens * (0.4 + 0.6 * np.sin(np.pi * u)), freqs=scale_freqs(f_lo, f_hi),
                    tau=(0.04, 0.25), glassy=0.3, pitch_fn=lambda u: 2 ** (1.2 * u),
                    pan_fn=lambda u, r: r.normal(0, 0.3), env_fn=lambda u: np.sin(np.pi * min(u * 1.2, 1)) + 0.05)


def glow_pad(dur, freqs, seed, attack=1.5, release=2.5, trem=0.35):
    r = rng(seed)
    n = ns(dur)
    t = np.arange(n) / SR
    L = np.zeros(n)
    R = np.zeros(n)
    for i, f in enumerate(freqs):
        a = 1.0 / (1 + 0.3 * i)
        for side, det in ((L, -0.6), (R, 0.6)):
            am = 1 - trem + trem * np.sin(TAU * r.uniform(0.15, 0.5) * t + r.uniform(0, 6))
            side += a * am * (np.sin(TAU * (f + det * r.uniform(0.2, 1)) * t + r.uniform(0, 6)) +
                              0.2 * np.sin(TAU * 2 * f * t + r.uniform(0, 6)))
    env = ar_env(n, attack, release)
    return np.stack([L, R], 1) * env[:, None]


def power_off(seed):
    """Relay clunk + hum collapsing in pitch (CRT-style) -> silence."""
    r = rng(seed)
    n = ns(3.2)
    t = np.arange(n) / SR
    clunk = modal(np.array([140, 310, 560, 980, 1600]) * r.uniform(.97, 1.03), [0.25, 0.2, 0.15, 0.1, 0.07],
                  [1, .8, .5, .3, .15], 0.4, hard=0.5)
    f = 110 * (38 / 110) ** smoothstep(t / 2.4)
    ph = TAU * np.cumsum(f) / SR
    hum = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.3 * np.sin(3 * ph)) * np.exp(-t / 0.9) * 0.5
    whine = np.sin(TAU * np.cumsum(1900 * (0.35 ** (t / 1.2))) / SR) * np.exp(-t / 0.5) * 0.06
    y = hum + whine
    mix_into(y, clunk * 0.8, 0)
    tink = crystal(2600, 0.8, seed + 2) * 0.03
    mix_into(y, tink, ns(1.8))
    return y


def boot_chime(seed, notes=(0, 4, 7, 12, 16)):
    n = ns(2.2)
    y = np.zeros(n)
    for i, s_ in enumerate(notes):
        c = fm_chime(note(s_ + 24), 1.2 if i < len(notes) - 1 else 1.8, seed + i)
        mix_into(y, c * (0.7 + 0.1 * i), ns(0.13 * i))
    return y


# ----------------------------------------------------------------------------------------------
# placing everything
# ----------------------------------------------------------------------------------------------
def voice_gaps(lid, thresh_db=-35, min_gap=0.15):
    x, _ = sf.read(os.path.join(ROOT, LINES[lid]['wav']))
    fr = SR // 100
    m = len(x) // fr
    e = np.sqrt((x[:m * fr].reshape(m, fr) ** 2).mean(1) + 1e-12)
    act = 20 * np.log10(e / e.max()) > thresh_db
    gaps, i = [], 0
    while i < m:
        if not act[i]:
            j = i
            while j < m and not act[j]:
                j += 1
            if (j - i) / 100 >= min_gap:
                gaps.append((LINES[lid]['start'] + i / 100, LINES[lid]['start'] + j / 100))
            i = j
        else:
            i += 1
    return gaps


def place_all(B):
    c = CUE
    S = {k: v['start'] for k, v in SC.items()}
    E = {k: v['end'] for k, v in SC.items()}

    # ======================= s01 opening =======================
    for i, tt in enumerate((0.55, 1.1)):
        B.add('stars_in_twinkle', c['s01_stars_in'] + tt, twinkle(note(33 + 2 * i), 10 + i, dur=1.8), -30,
              rev='magic', send_db=-4)
    ti = c['title_in']
    title = sparkles(4.8, 20, lambda u: 6 + 70 * smoothstep(u / 0.4) * (1 - 0.8 * smoothstep((u - 0.45) / 0.55)),
                     freqs=scale_freqs(1200, 7000), tau=(0.05, 0.4), glassy=0.25,
                     pan_fn=lambda u, r: r.normal(0, 0.9 * (1 - 0.7 * min(u / 0.4, 1)) + 0.05),
                     pitch_fn=lambda u: 2 ** (-0.3 * (1 - min(u / 0.4, 1))))
    B.add('title_in_shimmer', ti, title, -20, rev='magic', send_db=-2, note_='star particles assemble')
    bloom = glow_pad(4.5, [note(s_ + 12) for s_ in (0, 7, 12, 16, 19)], 21, attack=1.4, release=2.6)
    B.add('title_in_glow', ti + 0.6, bloom, -34, rev='magic', send_db=-6)
    B.add('title_formed_chime', ti + 1.7, psum(crystal(note(33), 3.0, 22), 0.6 * crystal(note(40), 2.4, 23)), -30,
          p=0.0, rev='magic', send_db=-3)
    to = c['title_out']
    diss = sparkles(4.0, 24, lambda u: 60 * np.exp(-u * 3) + 3, freqs=scale_freqs(2000, 9000), tau=(0.04, 0.3),
                    glassy=0.3, pan_fn=lambda u, r: r.normal(0, 0.3 + 0.6 * u), pitch_fn=lambda u: 2 ** (0.8 * u))
    B.add('title_out_dissolve', to, diss, -22, rev='magic', send_db=-3)
    B.add('title_out_whoosh', to, whoosh(2.2, 900, 5000, 25, peak=0.25, Q=1.0, pan0=0, pan1=0.2)[:, :], -38,
          rev='magic', send_db=-8)

    # ======================= s02 evening =======================
    t0, arr = c['s02_climb'], c['s02_arrive']
    xf_end = S['s02'] + SC['s02']['xfade']
    k = 0
    while t0 + k * 0.5 <= arr + 1e-6:
        tt = t0 + k * 0.5
        g = -13 + 20 * np.log10(max(0.15, min(1.0, (tt - S['s02'] + 0.15) / max(xf_end - S['s02'], 1e-3))))
        last = abs(tt - arr) < 1e-6
        fs = footstep(1000 + k, weight=1.0, surface='floor' if last else 'iron', creaky=(k % 3 == 1),
                      oilcan=(k % 2 == 0))
        B.add('step_stairs' if not last else 'step_lamproom', tt, fs, g - (3 if last else 0),
              p=0.25 * np.sin(k * 1.3), rev='tower', send_db=-6)
        if not last:
            B.add('knee_servo', tt + 0.12, servo(0.22, 260, 380, 1100 + k, gear=0.2), g - 17,
                  p=0.2 * np.sin(k * 1.3), rev='tower', send_db=-10)
        k += 1
    # polishing: oil can + squeaks in the gaps of D01 (and softly under it)
    d01 = LINES['D01']
    B.add('oil_can', arr + 0.15, oil_can(1200), -24, p=0.2, rev='lamp', send_db=-8)
    gaps = [g_ for g_ in voice_gaps('D01') if g_[1] - g_[0] > 0.2]
    sq_times = [arr + 0.32]
    for a, b in gaps:
        if b <= d01['end'] + 0.6:
            sq_times += [a + 0.03, a + 0.03 + min(0.16, (b - a) / 2)]
    sq_times += [d01['end'] + 0.02]
    for i, tt in enumerate(sq_times):
        if tt > c['lamp_ignite'] - 0.5:
            continue
        B.add('polish_squeak', tt, squeak(0.12 + 0.04 * (i % 2), 1750 + 260 * (i % 3), 1300 + i), -27,
              p=-0.1, rev='lamp', send_db=-8)
    for i, tt in enumerate(np.linspace(d01['start'] + 0.4, d01['end'] - 0.4, 4)):
        B.add('polish_rub_soft', tt, squeak(0.1, 2100 + 150 * i, 1350 + i), -40, p=-0.1, rev='lamp', send_db=-10)
    li = c['lamp_ignite']
    B.add('lever_clunk', li - 0.35, lever(1400), -12, p=0.15, sync=0.32, rev='lamp', send_db=-6)
    B.add('lamp_whoomp', li, whoomp(1500), -9, sync=0.15, rev='lamp', send_db=-6)
    B.add('ignite_sparkle', li + 0.05, sparkles(1.6, 1501, lambda u: 40 * np.exp(-u * 4), freqs=scale_freqs(1500, 6000),
                                                  tau=(0.05, 0.3)), -28, rev='lamp', send_db=-6)
    sit = c['s02_sit']
    d02 = LINES['D02']
    B.add('sigh_servo', d02['end'] + 0.35, sigh_servo(1600), -24, p=0.0, rev='outdoor', send_db=-10,
          cut=E['s02'])

    # ======================= s03 the falling star =======================
    ms, im = c['meteor_start'], c['impact']
    B.add('odd_twinkle', ms - 1.6, twinkle(note(38), 1700, warble=0.02), -26, p=-0.5, rev='magic', send_db=-4)
    B.add('odd_twinkle2', ms - 0.75, twinkle(note(40), 1701, warble=0.06), -24, p=-0.55, rev='magic', send_db=-4)
    md = im - ms
    B.add('meteor_whoosh', im, meteor(md, 1800), -4, sync=md, rev='outdoor', send_db=-6, fout=0.005)
    B.add('impact_boom', im, impact_boom(1900), -5, p=0.2, rev='outdoor', send_db=-10)
    B.add('impact_splash', im + 0.02, splash(3.5, 1901, size=1.6, bubbles=160, spray=1.4), -7, p=0.25,
          rev='outdoor', send_db=-6)
    B.add('impact_bells', im + 0.04, magic_bells(1902), -20, p=0.15, rev='magic', send_db=0)
    B.add('impact_flash_shimmer', im + 0.02, sparkles(2.0, 1903, lambda u: 90 * np.exp(-u * 3), freqs=scale_freqs(2500, 9000),
                                                        tau=(0.03, 0.2)), -24, rev='magic', send_db=-3)
    B.add('biolum_rings', im + 0.1, biolum_rings(5.0, 1904), -22, rev='magic', send_db=-5)
    B.add('antenna_bleep', im + 0.62, bleep([1900, 2500], 1905), -24, p=0.05, rev='outdoor', send_db=-10)
    B.add('startle_servo', im + 0.85, servo(0.25, 300, 620, 1906, gear=0.35), -26, rev='outdoor', send_db=-12)
    rd, bo = c['run_down'], c['boat_out']
    door = door_time()
    k = 0
    step = 1 / (2 * 2.2)
    while rd + k * step < bo - 1e-6:
        tt = rd + k * step
        surf = 'iron' if tt < door else 'stone'
        B.add('run_step_down' if surf == 'iron' else 'run_step_rock', tt,
              footstep(2000 + k, weight=0.8, surface=surf, hurry=True), -14 if surf == 'iron' else -18,
              p=0.35 * np.sin(k * 2.1), rev='tower' if surf == 'iron' else 'outdoor', send_db=-7, cut=bo)
        k += 1
    B.add('door_bang', door, door_bang(2100), -9, sync=0.06, p=0.1, rev='outdoor', send_db=-8)
    # rowing
    tt = bo
    k = 0
    while tt < E['s03'] - 0.05:
        B.add('oar_stroke', tt, oar_stroke(2200 + k), -15, p=0.25 * (-1) ** k, rev='outdoor', send_db=-14,
              cut=E['s03'])
        tt += 1.6
        k += 1

    # ======================= s04 the meeting =======================
    lift = c['lift']
    B.add('hands_dip', lift, splash(1.2, 2300, size=0.45, bubbles=40, spray=0.2, low=0.3), -20, p=0.0,
          rev='outdoor', send_db=-14)
    pour = bp(white(ns(0.9), rng(2301)), 700, 5000) * bell_curve(ns(0.9), 0.15, 1.0) * 0.25
    pour += drips(0.9, 2302, lambda u: 60 * (1 - u) + 5, amp=0.6)[:ns(0.9)]
    B.add('lift_pour', lift + 0.45, pour, -22, rev='outdoor', send_db=-14)
    B.add('lift_drips', lift + 1.1, drips(5.0, 2303, lambda u: 9 * np.exp(-u * 2.2) + 0.3, amp=0.7), -24,
          rev='outdoor', send_db=-12)
    g01, d05, g02, g03 = LINES['G01'], LINES['D05'], LINES['G02'], LINES['G03']
    hh = heart_hum(g02['start'] + 1.0 - (d05['start'] - 1.2), 2400)
    B.add('heart_hum_warm', d05['start'] - 1.2, hh, -24, rev='outdoor', send_db=-14)
    B.add('glow_returns', d05['end'] - 1.8, rising_sparkle(2.2, 2401, 1500, 4500, dens=18), -32, rev='magic',
          send_db=-5)
    B.add('eyes_open_ting', g02['start'] - 0.5, crystal(note(31), 1.4, 2402), -34, rev='magic', send_db=-6)
    B.add('flicker_fade', g03['end'] - 1.1, glassy_flicker(1.4, 2403, strength=0.8), -28, rev='magic',
          send_db=-8)
    det = c['determined']
    B.add('determined_servo', det + 0.1, servo(0.3, 320, 700, 2404, gear=0.4), -26, rev='outdoor', send_db=-12)
    B.add('determined_ching', det + 0.38, fm_chime(note(26), 0.9) * 0.6, -30, rev='magic', send_db=-6)
    d06 = LINES['D06']
    B.add('glow_bloom', d06['end'] + 0.05, rising_sparkle(1.8, 2405, 1500, 5000, dens=30), -28, rev='magic',
          send_db=-4)
    B.add('glow_bloom_pad', d06['end'] + 0.05, glow_pad(3.0, [note(s_ + 12) for s_ in (0, 4, 7)], 2406, 0.6, 2.0),
          -40, rev='magic', send_db=-6)
    rb = c['row_back']
    tt = rb
    k = 0
    while tt < E['s04'] - 0.05:
        B.add('oar_stroke_far', tt, oar_stroke(2500 + k, distance=0.6), -21, p=0.1, rev='outdoor', send_db=-10)
        tt += 1.6
        k += 1

    # ======================= s05 trying =======================
    th, ca = c['throw1'], c['catch']
    B.add('throw_whoosh', th, whoosh(1.1, 400, 2600, 2600, peak=0.3, Q=1.2, pan0=0, pan1=0.1), -16, sync=0.2,
          rev='outdoor', send_db=-10)
    trail = sparkles(ca - th, 2601, lambda u: 60 * (1 - 0.5 * np.sin(np.pi * u)), freqs=scale_freqs(1800, 7000),
                     tau=(0.04, 0.25), pitch_fn=lambda u: 2 ** (0.9 * np.sin(np.pi * u) - 0.2),
                     pan_fn=lambda u, r: r.normal(0, 0.2))
    B.add('throw_sparkle_trail', th, trail, -22, rev='magic', send_db=-5)
    B.add('apex_twirl', th + 0.48 * (ca - th), twinkle(note(36), 2602, warble=0.03), -30, rev='magic', send_db=-5)
    B.add('fall_whistle', ca - 0.75, pan(sine(2400 * 0.55 ** np.linspace(0, 1, ns(0.75)) ** 1.0) *
                                         bell_curve(ns(0.75), 0.5, 1.0) * 0.3, 0), -34, rev='magic', send_db=-8)
    B.add('catch_bonk', ca, bonk(2603), -16, p=0.0, rev='outdoor', send_db=-12)
    B.add('catch_boing', ca + 0.01, boing(2604, 300), -20, p=0.0, rev='outdoor', send_db=-12)
    st, cl, tf, spl = c['stack'], c['climb'], c['tower_fall'], c['splash']
    for i in range(3):
        tt = st + (cl - st) * (i + 0.15) / 3
        B.add('stack_whoosh', tt - 0.12, whoosh(0.3, 600, 1800, 2700 + i, peak=0.6)[:, :], -30, rev='outdoor',
              send_db=-12)
        B.add('stack_servo', tt - 0.18, servo(0.2, 300 + 60 * i, 520 + 80 * i, 2710 + i, gear=0.3), -30,
              rev='outdoor', send_db=-14)
        B.add('stack_thunk_crate', tt, wood_thunk(2720 + i, 'crate'), -15, p=0.25 - 0.2 * i, rev='outdoor', send_db=-10)
        B.add('stack_thunk_barrel', tt + 0.22, wood_thunk(2730 + i, 'barrel'), -18, p=0.2 - 0.2 * i, rev='outdoor',
              send_db=-10)
    wob_d = tf - cl
    wob = creak(wob_d, 2800, rate=(14, 30, 55), body=(310, 690, 1350, 2100), Q=12, jitter=0.5, shape=0.2)
    wob *= np.linspace(0.15, 1.0, len(wob)) ** 1.6 * (0.7 + 0.3 * np.sin(TAU * np.cumsum(np.linspace(0.9, 2.2, len(wob))) / SR))
    B.add('tower_creaks', cl, wob, -17, p=0.1, rev='outdoor', send_db=-10)
    for i, tt in enumerate(np.cumsum(np.linspace(0.9, 0.3, 7)) + cl):
        if tt < tf:
            B.add('crate_rattle', tt, rattle(2810 + i, 0.2, 25), -22 + 1.2 * i, p=0.1 * np.sin(i), rev='outdoor',
                  send_db=-12)
    B.add('tiptoe_servo', cl + 0.2, servo(1.2, 900, 1150, 2820, gear=0.1, wobble=0.04), -38, rev='outdoor',
          send_db=-14)
    B.add('topple_groan', tf, creak(0.55, 2830, rate=(60, 25, 8), body=(210, 470, 980), Q=10), -12,
          rev='outdoor', send_db=-10)
    B.add('fall_whoosh', tf + 0.2, whoosh(0.8, 1400, 350, 2831, peak=0.55, pan0=-0.2, pan1=0.3), -20,
          rev='outdoor', send_db=-10)
    r5 = rng(2840)
    for i in range(12):
        tt = tf + 0.25 + (spl - tf - 0.2) * (i / 11) ** 0.8 + r5.uniform(-0.03, 0.03)
        kind = 'barrel' if i % 3 == 1 else 'crate'
        B.add(f'crash_{kind}', tt, wood_thunk(2850 + i, kind), -10 - r5.uniform(0, 5), p=r5.uniform(-0.7, 0.7),
              rev='outdoor', send_db=-8)
        if i % 4 == 0:
            B.add('crash_splinter', tt + 0.01, splinter(2870 + i), -16, p=r5.uniform(-0.6, 0.6), rev='outdoor',
                  send_db=-10)
    B.add('comedic_splash', spl, splash(3.0, 2900, size=1.4, bubbles=180, spray=1.6), -6, p=0.0, rev='outdoor',
          send_db=-8)
    B.add('sploosh_bloop', spl + 0.03, pan(sine(420 * 0.3 ** np.linspace(0, 1, ns(0.35)), ns(0.35)) *
                                           perc(ns(0.35), 0.005, 0.12), 0), -18, rev='outdoor', send_db=-12)
    B.add('pop_up_deng', spl + 0.6, bubble(160, 0.06, 2.2, 1.0), -18, p=-0.1, rev='outdoor', send_db=-12)
    B.add('pop_up_guang', spl + 0.9, bubble(420, 0.04, 2.5, 1.0), -22, p=0.2, rev='outdoor', send_db=-12)
    B.add('pop_up_sparkle', spl + 0.9, twinkle(note(33), 2901), -26, p=0.2, rev='magic', send_db=-5)
    B.add('after_drips', spl + 0.7, drips(4.5, 2902, lambda u: 10 * np.exp(-u * 1.5) + 0.5, amp=0.8), -22,
          rev='outdoor', send_db=-12)
    r5 = rng(2903)
    for i in range(5):
        B.add('floating_crate_knock', spl + 1.5 + i * r5.uniform(0.6, 1.1), wood_thunk(2910 + i, 'crate') * 0.5,
              -30, p=r5.uniform(-0.6, 0.6), rev='outdoor', send_db=-12)
    g05 = LINES['G05']
    B.add('happy_antenna_beep', g05['end'] + 0.12, bleep([note(24), note(28), note(31)], 2950, seg=0.07, gap=0.03,
                                                          chirp=0.03), -25, rev='outdoor', send_db=-10)
    d10 = LINES['D10']
    B.add('happy_glow', d10['end'] + 0.1, rising_sparkle(1.6, 2960, 1500, 5000, dens=22), -30, rev='magic',
          send_db=-5)
    dw = c['dawn_warning']
    B.add('dawn_warning_flicker', dw, glassy_flicker(1.6, 2970, strength=1.0), -22, rev='magic', send_db=-6)
    B.add('dawn_warning_flicker2', dw + 1.9, glassy_flicker(1.0, 2971, f0=2800, strength=0.6), -30, rev='magic',
          send_db=-8)
    idea = c['idea']
    B.add('beam_pass_whoosh', idea - 0.35, whoosh(1.4, 250, 900, 2980, peak=0.4, Q=1.0, pan0=-0.6, pan1=0.6), -24,
          rev='outdoor', send_db=-10)
    B.add('idea_bleep', idea, fm_chime(note(26), 1.0) * 0.6 + np.concatenate([np.zeros(ns(0.09)), fm_chime(note(33), 0.91)])[:ns(1.0)],
          -20, rev='outdoor', send_db=-8)
    B.add('idea_sparkle', idea + 0.05, sparkles(1.0, 2981, lambda u: 50 * np.exp(-u * 3), freqs=scale_freqs(2500, 8000),
                                                  tau=(0.03, 0.2)), -28, rev='magic', send_db=-5)

    # ======================= s06 the beam =======================
    ru, au, bu = c['run_up'], c['aim_up'], c['beam_up']
    k = 0
    while ru + k * step < au - 1e-6:
        tt = ru + k * step
        B.add('run_step_up', tt, footstep(3000 + k, weight=0.85, hurry=True), -14, p=0.35 * np.sin(k * 2.1),
              rev='tower', send_db=-7, cut=au)
        k += 1
    B.add('guang_weak_flicker', ru + 0.8, glassy_flicker(1.6, 3050, f0=2900, strength=0.5), -36, rev='tower',
          send_db=-10)
    crank_d = hatch_time() - au
    B.add('wheel_grab', au, modal([210, 480, 930], [0.4, 0.3, 0.2], [1, .6, .3], 0.6, hard=0.4), -18,
          rev='lamp', send_db=-8)
    B.add('effort_servo', au + 0.1, servo(crank_d - 0.2, 900, 1500, 3100, gear=0.5, wobble=0.05,
                                           curve=lambda u: u ** 0.7), -28, rev='lamp', send_db=-8)
    B.add('iron_wheel_groan', au + 0.15, iron_groan(crank_d, 3101), -12, p=-0.1, rev='lamp', send_db=-6)
    B.add('gear_grinding', au + 0.25, gear_grind(crank_d - 0.1, 3102) * ar_env(ns(crank_d - 0.1), 0.2, 0.2), -17,
          p=0.15, rev='lamp', send_db=-6)
    B.add('ratchet', au + 0.3, ratchet(crank_d - 0.2, 3103, rate=8), -18, p=0.3, rev='lamp', send_db=-8)
    B.add('lens_tilt_rumble', au + 0.4, pan(lp(pink(ns(crank_d), rng(3104)), 120) * ar_env(ns(crank_d), 0.4, 0.3), 0),
          -16, rev='lamp', send_db=-8)
    hatch = hatch_time()
    B.add('lens_lock_clunk', hatch - 0.25, lever(3105)[ns(0.3):], -16, rev='lamp', send_db=-6)
    B.add('roof_hatch_clank', hatch, hatch_clank(3106), -11, p=0.0, rev='lamp', send_db=-6)
    B.add('hatch_wind_in', hatch + 0.3, whoosh(1.5, 300, 1200, 3107, peak=0.3), -26, rev='lamp', send_db=-10)
    B.add('beam_up_hum', bu, electric_rise(3108), -11, rev='outdoor', send_db=-6)
    B.add('beam_up_whoosh', bu + 0.05, whoosh(2.4, 200, 3500, 3109, peak=0.35, Q=1.1), -18, rev='magic',
          send_db=-8)
    th_ = c['touch_heart']
    B.add('heart_glow_hum', th_, heart_hum(4.2, 3110, beat=True, bpm=58), -27, rev='lamp', send_db=-10)
    B.add('glass_touch', th_ + 0.9, crystal(3900, 0.5, 3111, hard=0.3) * 0.4, -34, rev='lamp', send_db=-8)
    for i, g_ in enumerate(('G08', 'D13', 'D14')):
        ln = LINES[g_]
        B.add('guang_faint_flicker', ln['start'] - 0.55 if i == 0 else ln['end'] + 0.1,
              glassy_flicker(0.6, 3120 + i, f0=2700, strength=0.4), -38, rev='lamp', send_db=-10)
    ho, hi_, ig = c['heart_out'], c['heart_in'], c['ignite']
    B.add('porthole_creak', ho, creak(0.5, 3200, rate=(90, 160, 110), body=(780, 1450, 2600), Q=22), -22, p=0.0,
          rev='lamp', send_db=-8)
    B.add('porthole_glass_clink', ho + 0.5, crystal(4300, 0.7, 3201, hard=0.8), -26, rev='lamp', send_db=-8)
    B.add('heart_embers', ho + 0.6, psum(crackle(hi_ - ho - 0.4, 3202, 18, lo=1200, hi=6000, width=0.2),
               pan(lp(pink(ns(hi_ - ho - 0.4), rng(3203)), 400) * 0.05, 0)), -30, rev='lamp', send_db=-10)
    B.add('heart_lift_swell', ho + 0.6, glow_pad(hi_ - ho, [note(s_) for s_ in (0, 7, 12)], 3204, 0.8, 1.0), -40,
          rev='lamp', send_db=-8)
    dd = c['deng_dark']
    pdw = power_down(dd - ho, 3205)
    pdw *= np.interp(np.arange(len(pdw)) / SR + ho, [ho, ig - 0.1, ig + 0.5, dd - 2.5, dd], [1, 1, 0.5, 0.45, 0.2])
    B.add('power_down_whine', ho, pdw, -30, rev='lamp', send_db=-10)
    B.add('heart_in_click', hi_, modal([3300, 4900, 6800], [0.05, 0.03, 0.02], [1, .5, .3], 0.12) +
          0.5 * modal([520, 900], [0.08, 0.05], [1, .5], 0.12, hard=0.5), -18, rev='lamp', send_db=-8)
    B.add('heart_in_zzt', hi_ + 0.06, bp(saw(118, ns(0.12)) * ar_env(ns(0.12), 0.003, 0.04), 200, 4000), -30,
          rev='lamp', send_db=-8)
    pre = ig - hi_ - 0.15
    if pre > 0.2:
        m = ns(pre)
        u = np.linspace(0, 1, m)
        suck = tvf(pink(m, rng(3210)), 'bp', 400 * 12 ** u ** 2, 1.5, block=64) * u ** 3
        suck += 0.3 * np.sin(TAU * np.cumsum(note(24) * (1 + 0.5 * u ** 2)) / SR) * u ** 4
        B.add('pre_ignite_inhale', ig, suck, -16, sync=pre, rev='magic', send_db=-8, fout=0.003)
    B.add('IGNITION', ig, ignition(3300), -3, rev='outdoor', send_db=-10)
    ch_d = c['beam_fade'] + 1.6 - ig
    chv = choir(ch_d, [note(s_) for s_ in (-12, -5, 0, 4, 7, 12, 16)], 3301)
    tt = np.arange(len(chv)) / SR + ig
    chv *= np.interp(tt, [ig, ig + 0.7, ig + 2.0, ig + 3.2, c['beam_fade'], c['beam_fade'] + 1.6],
                     [0, 1, 0.8, 0.35, 0.3, 0])
    B.add('ignition_choir', ig + 0.05, pan(chv, 0) * np.array([[1.0, 0.92]]), -17, rev='magic', send_db=-2)
    shim = glow_pad(ch_d, [note(s_ + 24) for s_ in (0, 4, 7, 11, 14)], 3302, attack=0.6, release=1.6, trem=0.6)
    B.add('ignition_shimmer', ig + 0.1, shim, -30, rev='magic', send_db=-3)
    rs = c['rise']
    B.add('rise_sparkle', rs, rising_sparkle(3.2, 3400, 1200, 6000, dens=50), -20, rev='magic', send_db=-3)
    B.add('rise_color_chime', rs + 0.9, magic_bells(3401, chord=(0, 4, 7, 12)), -28, rev='magic', send_db=-3)
    wa = c['whale_appear']
    d15 = LINES['D15']
    w1 = whale_call(max(0.6, min(1.3, d15['start'] - wa + 0.2)), [(0, 150), (0.6, 260), (1, 230)], 3500,
                    formant=((0, 350), (0.6, 800), (1, 600)))
    B.add('whale_appear_call', wa, w1, -13, rev='sky', send_db=0, p=-0.1)
    w2d = min(2.6, c['beam_fade'] + 1.5 - (d15['end'] + 0.15))
    if w2d > 0.8:
        w2 = whale_call(w2d, [(0, 200), (0.3, 170), (0.7, 290), (1, 250)], 3501,
                        formant=((0, 420), (0.5, 950), (1, 500)))
        B.add('whale_gather_call', d15['end'] + 0.15, w2, -13, rev='sky', send_db=0, p=0.1)
    bf = c['beam_fade']
    B.add('beam_fade_winddown', bf, sparkles(2.5, 3600, lambda u: 40 * (1 - u), freqs=scale_freqs(1500, 6000),
                                             tau=(0.05, 0.3), pitch_fn=lambda u: 2 ** (-1.0 * u)), -26,
          rev='magic', send_db=-4)
    B.add('power_off', dd, power_off(3700), -16, rev='lamp', send_db=-8)

    # ======================= s07 the answer =======================
    ws = c['whale_song']
    sd = c['stardust']
    rb_ = c['reboot']
    song = [(0.0, 2.6, [(0, 110), (0.35, 190), (0.8, 170), (1, 150)], ((0, 300), (0.5, 800), (1, 450)), -12),
            (2.9, 1.9, [(0, 230), (0.5, 420), (1, 380)], ((0, 500), (0.6, 1300), (1, 700)), -15),
            (5.2, 2.6, [(0, 300), (0.3, 320), (1, 140)], ((0, 900), (0.5, 600), (1, 350)), -13)]
    for i, (dt, d_, cont, form, g_) in enumerate(song):
        if ws + dt + d_ > rb_ + 0.3:
            continue
        B.add(f'whale_song_{i + 1}', ws + dt, whale_call(d_, cont, 3800 + i, formant=form, rough=0.15 if i == 2 else 0),
              g_, p=-0.15 + 0.15 * i, rev='sky', send_db=0)
    B.add('whale_descend_air', ws + 0.5, pan(lp(pink(ns(4.0), rng(3810)), 90) * bell_curve(ns(4.0), 0.5, 1.0), 0),
          -22, rev='sky', send_db=-6)
    B.add('stardust_stream', sd, stardust(rb_ - sd + 0.2, 3900), -18, rev='magic', send_db=-3)
    B.add('reboot_crackle', rb_, crackle(0.5, 4000, lambda u: 120 * (1 - u) + 10, lo=1200, hi=9000, width=0.4), -22,
          rev='lamp', send_db=-8)
    B.add('reboot_heartbeat', rb_ + 0.15, heartbeat(4001, 1.0, 50), -12, rev='lamp', send_db=-10)
    B.add('reboot_heartbeat2', rb_ + 0.95, heartbeat(4002, 0.8, 52), -15, rev='lamp', send_db=-10)
    B.add('reboot_star_heart', rb_ + 0.2, glow_pad(2.4, [note(s_ + 12) for s_ in (0, 7, 12)], 4003, 0.3, 1.5), -32,
          rev='magic', send_db=-4)
    B.add('boot_chime', rb_ + 0.7, boot_chime(4004), -22, rev='magic', send_db=-5)
    for i, dt in enumerate((1.45, 1.7)):
        B.add('eye_blink_pip', rb_ + dt, bleep([note(36 + 2 * i)], 4010 + i, seg=0.04, chirp=0.2), -32,
              rev='lamp', send_db=-10)
    w01 = LINES['W01']
    sdur = max(0.3, min(0.9, w01['start'] - (rb_ + 1.5) + 0.2))
    B.add('servo_stretch', rb_ + 1.5, servo(sdur, 280, 520, 4020, gear=0.3, curve=lambda u: np.sin(np.pi * u * 0.8)),
          -30, rev='lamp', send_db=-10)
    nz = c['nuzzle']
    g10 = LINES['G10']
    B.add('nuzzle_sparkle', nz, sparkles(max(0.5, g10['start'] - nz + 0.3), 4100, 30, freqs=scale_freqs(1000, 3200),
                                         tau=(0.08, 0.4), glassy=0.1, attack=0.01,
                                         pan_fn=lambda u, r: r.normal(0, 0.25)), -26, rev='magic', send_db=-4)
    B.add('nuzzle_warm', nz, glow_pad(1.6, [note(s_) for s_ in (12, 16, 19)], 4101, 0.2, 1.2), -38, rev='magic',
          send_db=-6)
    fw, su = c['farewell'], c['sunrise']
    B.add('farewell_whoosh_up', fw, whoosh(1.4, 300, 3000, 4200, peak=0.35, Q=1.2, pan0=0, pan1=-0.2), -17,
          rev='magic', send_db=-6)
    B.add('farewell_trail', fw, rising_sparkle(2.0, 4201, 1500, 6000, dens=40), -22, rev='magic', send_db=-3)
    B.add('farewell_whale_call', fw + 0.5, whale_call(3.0, [(0, 170), (0.4, 330), (0.8, 300), (1, 360)], 4202,
                                                      formant=((0, 400), (0.5, 1100), (1, 700))), -12, p=-0.1,
          rev='sky', send_db=0)
    B.add('farewell_whale_call2', fw + 3.9, whale_call(2.4, [(0, 260), (0.5, 420), (1, 390)], 4203,
                                                       formant=((0, 600), (0.5, 1300), (1, 900))), -17, p=-0.3,
          rev='sky', send_db=0)
    sun_d = E['s07'] + SC['s08']['xfade'] - su
    B.add('sunrise_shimmer', su, glow_pad(sun_d, [note(s_ + 12) for s_ in (0, 4, 7, 12, 14, 19)], 4300,
                                          attack=3.5, release=3.0, trem=0.45), -30, rev='magic', send_db=-4)
    B.add('sunrise_sparkles', su + 0.5, sparkles(sun_d - 0.5, 4301, lambda u: 5 + 14 * np.sin(np.pi * u),
                                                 freqs=scale_freqs(1800, 7000), tau=(0.1, 0.5), glassy=0.2,
                                                 attack=0.01), -30, rev='magic', send_db=-3)
    B.add('whale_dissolve', su + 2.5, sparkles(3.5, 4302, lambda u: 50 * np.exp(-u * 1.5), freqs=scale_freqs(2500, 9000),
                                               tau=(0.03, 0.2), pitch_fn=lambda u: 2 ** (0.5 * u),
                                               pan_fn=lambda u, r: r.uniform(-0.2, 0.9)), -26, rev='magic',
          send_db=-3)

    # ======================= s08 epilogue =======================
    for i, (cn, s_) in enumerate((('blink1', 33), ('blink2', 38))):
        tb = c[cn]
        ting = crystal(note(s_), 2.6, 4400 + i, hard=0.9)
        B.add(f'{cn}_crystal_ting', tb, ting, -17, p=0.05, rev='magic', send_db=-3)
        B.add(f'{cn}_glints', tb + 0.01, sparkles(0.7, 4410 + i, 30, freqs=scale_freqs(4000, 10000), tau=(0.02, 0.1),
                                                  pan_fn=lambda u, r: r.normal(0.05, 0.25)), -30, rev='magic',
              send_db=-4)
    n06 = LINES['N06']
    B.add('answer_twinkle', n06['end'] + 0.3, twinkle(note(40), 4420, dur=1.2), -34, p=0.1, rev='magic', send_db=-3)
    et = c['end_title']
    et_d = min(4.5, DUR - et - 0.2)
    end = sparkles(et_d, 4500, lambda u: 5 + 45 * smoothstep(u / 0.35) * (1 - 0.8 * smoothstep((u - 0.4) / 0.6)),
                   freqs=scale_freqs(1200, 7000), tau=(0.05, 0.4), glassy=0.25,
                   pan_fn=lambda u, r: r.normal(0, 0.8 * (1 - 0.7 * min(u / 0.35, 1)) + 0.05))
    B.add('end_title_shimmer', et, end, -24, rev='magic', send_db=-2)
    B.add('end_title_glow', et + 0.5, glow_pad(min(5.5, DUR - et - 0.6), [note(s_ + 12) for s_ in (0, 7, 12, 16)],
                                               4501, attack=1.5, release=2.5), -38, rev='magic', send_db=-5)
    B.add('end_title_chime', et + 1.4, psum(crystal(note(33), 3.0, 4502), 0.5 * crystal(note(40), 2.2, 4503)), -32,
          rev='magic', send_db=-3)


def final_fade(x):
    """Everything fades to silence between cue fade_out and the end."""
    fo = cue('fade_out') if 'fade_out' in CUE else DUR - 1.0
    t = np.arange(len(x)) / SR
    g = 1 - smoothstep((t - fo) / max(DUR - fo - 0.05, 0.05))
    return x * g[:, None]


def write_wav(path, x):
    x = np.asarray(x, np.float64)
    assert len(x) == N
    pk = np.max(np.abs(x))
    if pk > 0.98:
        print(f'  ! {os.path.basename(path)} peak {pk:.3f} -> scaled to 0.98')
        x = x * (0.98 / pk)
    sf.write(path, x.astype(np.float32), SR, subtype='PCM_24')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['ambience', 'sfx'], default=None)
    a = ap.parse_args()
    os.makedirs(os.path.join(BUILD, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(BUILD, 'sfx'), exist_ok=True)
    t_ = time.time()
    if a.only in (None, 'ambience'):
        print('rendering ambience ...')
        amb = render_ambience()
        amb = final_fade(hp(amb, 20))
        write_wav(os.path.join(BUILD, 'audio', 'ambience.wav'), amb)
        print(f'  ambience done ({time.time() - t_:.0f}s)')
    if a.only in (None, 'sfx'):
        print('rendering sfx ...')
        B = SfxBus()
        place_all(B)
        print(f'  placed {len(B.events)} events ({time.time() - t_:.0f}s); convolving reverbs ...')
        out = final_fade(hp(B.render(), 18))
        write_wav(os.path.join(BUILD, 'audio', 'sfx.wav'), out)
        ev = sorted(B.events, key=lambda e: e['time'])
        json.dump(dict(sr=SR, duration=DUR, count=len(ev), events=ev), open(os.path.join(BUILD, 'sfx', 'events.json'), 'w'),
                  indent=1, ensure_ascii=False)
        print(f'  sfx done ({time.time() - t_:.0f}s)')


if __name__ == '__main__':
    main()
