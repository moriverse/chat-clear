#!/usr/bin/env python3
"""Final re-recording mix for 《一盏灯，一颗星》.

    nice -n 10 python3 audio/mix.py                 # full mix + stems + report + analysis PNGs
    nice -n 10 python3 audio/mix.py --no-plots      # faster
    nice -n 10 python3 audio/mix.py --music-db -2 --sfx-db +1   # rebalance (dB offsets per bus)

Inputs  (missing ones are treated as silence, with a warning):
    build/voice/<ID>.wav         dialogue lines, placed at their timeline start times
                                 (falls back to build/audio/dialogue.wav if a voice file is missing)
    build/audio/music.wav        score (any sample rate / channel count; resampled, padded/trimmed)
    build/audio/ambience.wav     beds    (audio/sfx.py)
    build/audio/sfx.wav          effects (audio/sfx.py)
Outputs:
    build/audio/final_mix.wav    48 kHz stereo 24-bit, exactly the timeline duration,
                                 -16 LUFS integrated, true peak <= -1.0 dBTP
    build/audio/stem_{dialogue,music,ambience,sfx}.wav
                                 post-fader, post-ducking, with the master gain *and* the limiter's
                                 gain curve applied, so the four stems sum to final_mix.wav
    build/sfx/mix_report.json    loudness numbers, per-line dialogue margins, checks
    build/sfx/analysis_*.png     loudness-over-time, spectrogram, per-scene SFX/event views

Mix chain: dialogue bus (HPF, gentle 2:1 compressor, presence lift) -> centre (~-17 LUFS).
Music / ambience / sfx are first brought to fixed integrated targets (-22.5 / -31 / -25.5 LUFS,
plus the --*-db offsets) so a re-rendered stem keeps its place in the balance.
Music bus (HPF, -2 dB dip at 2.8 kHz for speech, sidechain duck ~-7 dB under dialogue).
Ambience bus (HPF, sidechain duck ~-4 dB).  SFX bus (HPF).  Automatic dialogue protection:
for every line where the non-dialogue buses are closer than --margin LU to the dialogue, they are
pulled down smoothly for that line only.  Master: gentle 1.3:1 glue compression, loudness
normalisation (pyloudnorm, BS.1770-4) and a 4x-oversampled look-ahead true-peak limiter.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pyloudnorm as pyln
import scipy.signal as sig
import soundfile as sf
from scipy.ndimage import maximum_filter1d, uniform_filter1d

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
BUILD = os.path.join(ROOT, 'build')
AUD = os.path.join(BUILD, 'audio')
SR = 48000
TL = json.load(open(os.path.join(BUILD, 'timeline.json')))
DUR = float(TL['duration'])
N = int(round(DUR * SR))
BUSES = ('dialogue', 'music', 'ambience', 'sfx')


def db(x):
    return 10.0 ** (np.asarray(x, float) / 20.0)


def todb(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-12))


def log(*a):
    print(*a, flush=True)


# ----------------------------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------------------------
def load_stereo(path, label):
    if not os.path.exists(path):
        log(f'  - {label}: {os.path.relpath(path, ROOT)} not found -> silence')
        return None
    x, sr = sf.read(path, dtype='float64', always_2d=True)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, 1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    if sr != SR:
        from math import gcd
        g = gcd(int(sr), SR)
        x = sig.resample_poly(x, SR // g, int(sr) // g, axis=0)
        log(f'  - {label}: resampled {sr} -> {SR} Hz')
    if len(x) < N:
        x = np.concatenate([x, np.zeros((N - len(x), 2))])
    elif len(x) > N:
        k = min(len(x) - N, int(0.05 * SR))
        x = x[:N].copy()
        if k:
            x[N - k:] *= np.linspace(1, 0, k)[:, None]
    log(f'  - {label}: {os.path.relpath(path, ROOT)}  ({len(x) / SR:.2f}s)')
    return x


def build_dialogue():
    """Place every line's processed voice file at its timeline start (mono -> centre)."""
    d = np.zeros(N)
    missing = []
    for ln in TL['lines']:
        p = os.path.join(ROOT, ln['wav'])
        if not os.path.exists(p):
            missing.append(ln['id'])
            continue
        x, sr = sf.read(p, dtype='float64', always_2d=True)
        x = x.mean(1)
        if sr != SR:
            x = sig.resample_poly(x, SR, sr)
        i = int(round(ln['start'] * SR))
        m = min(len(x), N - i)
        d[i:i + m] += x[:m]
    if missing:
        stem = load_stereo(os.path.join(AUD, 'dialogue.wav'), 'dialogue stem (fallback)')
        if stem is not None:
            log(f'  - voice files missing for {missing}: using dialogue.wav for everything')
            return stem.mean(1) / 0.9
    return d


def write(path, x, subtype='PCM_24'):
    x = np.asarray(x, np.float64)
    assert x.shape == (N, 2), x.shape
    if subtype == 'PCM_24' and np.max(np.abs(x)) >= 1.0:
        subtype = 'FLOAT'
    sf.write(path, x.astype(np.float32 if subtype == 'FLOAT' else np.float64), SR, subtype=subtype)


# ----------------------------------------------------------------------------------------------
# DSP helpers
# ----------------------------------------------------------------------------------------------
def sos(kind, f, order=2):
    return sig.butter(order, f, btype=kind, fs=SR, output='sos')


def peq(x, f0, gain_db, Q=1.0):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    al = np.sin(w0) / (2 * Q)
    b = np.array([1 + al * A, -2 * np.cos(w0), 1 - al * A])
    a = np.array([1 + al / A, -2 * np.cos(w0), 1 - al / A])
    return sig.lfilter(b / a[0], a / a[0], x, axis=0)


def shelf_hi(x, f0, gain_db):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    al = np.sin(w0) / 2 * np.sqrt(2)
    cw = np.cos(w0)
    b = np.array([A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * al), -2 * A * ((A - 1) + (A + 1) * cw),
                  A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * al)])
    a = np.array([(A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * al, 2 * ((A - 1) - (A + 1) * cw),
                  (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * al])
    return sig.lfilter(b / a[0], a / a[0], x, axis=0)


CRC = 1000  # control rate for gain curves (Hz)
HOP = SR // CRC


def rms_ctl(x):
    """RMS (both channels) per 1 ms control frame."""
    m = x if x.ndim == 1 else np.sqrt(np.mean(x ** 2, 1))
    k = len(m) // HOP
    return np.sqrt(np.mean(m[:k * HOP].reshape(k, HOP) ** 2, 1) + 1e-20)


def smooth_ar(v, attack, release):
    """Asymmetric one-pole smoothing on a control-rate curve (rising -> attack)."""
    aa = np.exp(-1.0 / (attack * CRC))
    ar = np.exp(-1.0 / (release * CRC))
    out = np.empty_like(v)
    y = v[0]
    for i, x in enumerate(v):
        c = aa if x > y else ar
        y = c * y + (1 - c) * x
        out[i] = y
    return out


def ctl_to_audio(g):
    return np.interp(np.arange(N) / SR, (np.arange(len(g)) + 0.5) / CRC, g)


def compressor_gain(x, thresh_db, ratio, attack, release, knee=6.0, rms_win=0.03):
    """Gain curve (audio rate, linear) of a feed-forward RMS compressor."""
    e = rms_ctl(x)
    e = np.sqrt(np.maximum(uniform_filter1d(e ** 2, max(1, int(rms_win * CRC))), 1e-20))
    lv = todb(e)
    over = lv - thresh_db
    gr = np.where(over <= -knee / 2, 0.0,
                  np.where(over >= knee / 2, over * (1 - 1 / ratio),
                           (1 - 1 / ratio) * (over + knee / 2) ** 2 / (2 * knee)))
    gr = smooth_ar(gr, attack, release)
    return ctl_to_audio(db(-gr))


def dialogue_key(d, thresh_db=-42.0, hold=0.25, lookahead=0.12, attack=0.12, release=0.6):
    """0..1 curve that is 1 while dialogue is speaking (with look-ahead, hold, smooth A/R)."""
    e = todb(rms_ctl(d))
    act = (e > thresh_db).astype(float)
    act = maximum_filter1d(act, int(2 * hold * CRC) + 1)
    la = int(lookahead * CRC)
    act = np.concatenate([act[la:], np.zeros(la)])
    k = smooth_ar(act, attack, release)
    k = np.clip(k * 1.15, 0, 1)
    k = np.concatenate([k, np.zeros(max(0, int(DUR * CRC) + 2 - len(k)))])
    return k


# K-weighting (BS.1770) at 48 kHz for loudness curves
_KB1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
_KA1 = [1.0, -1.69065929318241, 0.73248077421585]
_KB2 = [1.0, -2.0, 1.0]
_KA2 = [1.0, -1.99004745483398, 0.99007225036621]


def kweight(x):
    return sig.lfilter(_KB2, _KA2, sig.lfilter(_KB1, _KA1, x, axis=0), axis=0)


def loudness_curve(x, win=3.0, hop=0.1):
    """Short-term (win=3) / momentary (win=0.4) loudness (LUFS) every hop seconds."""
    y = kweight(x)
    p = np.sum(y ** 2, 1) if y.ndim == 2 else y ** 2
    c = np.concatenate([[0], np.cumsum(p)])
    w = int(win * SR)
    h = int(hop * SR)
    idx = np.arange(0, len(p) - w + 1, h)
    ms = (c[idx + w] - c[idx]) / w
    t = (idx + w / 2) / SR
    return t, -0.691 + 10 * np.log10(ms + 1e-20)


METER = pyln.Meter(SR)


def lufs(x):
    if len(x) < int(0.45 * SR):
        return -np.inf
    with np.errstate(divide='ignore'):
        v = METER.integrated_loudness(x)
    return v


def true_peak_env(x, os_=4, chunk=SR * 10, pad=256):
    """Per-sample true-peak magnitude (max over channels of the 4x-oversampled signal)."""
    n = len(x)
    env = np.zeros(n)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        a0, b0 = max(0, a - pad), min(n, b + pad)
        y = sig.resample_poly(x[a0:b0], os_, 1, axis=0)
        m = np.abs(y).max(1).reshape(-1, os_).max(1)
        env[a:b] = m[a - a0:a - a0 + (b - a)]
    return np.maximum(env, np.abs(x).max(1))


def limiter_gain(x, ceiling_db=-1.0, lookahead=0.0025, release=0.12):
    """Look-ahead true-peak limiter gain curve (linear, audio rate)."""
    thr = db(ceiling_db)
    tp = true_peak_env(x)
    gr = np.maximum(0.0, todb(tp) - todb(thr))
    if not np.any(gr > 0):
        return np.ones(len(x)), 0.0
    W = max(2, int(lookahead * SR))
    grm = maximum_filter1d(gr, 2 * W + 1)
    k = len(grm) // HOP
    blk = grm[:k * HOP].reshape(k, HOP).max(1)
    a = np.exp(-1.0 / (release * CRC))
    rel = np.empty_like(blk)
    y = 0.0
    for i, v in enumerate(blk):
        y = max(v, y * a)
        rel[i] = y
    rel_a = np.interp(np.arange(len(grm)), np.arange(k) * HOP + HOP / 2, rel)
    grm = np.maximum(grm, rel_a)
    grs = uniform_filter1d(grm, W)
    grs = np.maximum(grs, gr)  # safety
    return db(-grs), float(gr.max())


# ----------------------------------------------------------------------------------------------
# main mix
# ----------------------------------------------------------------------------------------------
def line_windows():
    return [(ln['id'], ln['start'], ln['end']) for ln in TL['lines']]


def protect_dialogue(dlg, rest, margin, max_cut=14.0, iters=3):
    """Smooth per-line attenuation of the non-dialogue buses so dialogue stays >= margin LU on top.
    Returns an audio-rate gain curve for the rest (1.0 where untouched) and a per-line report."""
    g_ctl = np.ones(int(DUR * CRC) + 2)
    cuts = {}
    for _ in range(iters):
        g = ctl_to_audio(g_ctl)
        changed = False
        for lid, s, e in line_windows():
            a, b = int(s * SR), int(e * SR)
            ld = lufs(dlg[a:b])
            lr = lufs(rest[a:b] * g[a:b, None])
            if not np.isfinite(ld) or not np.isfinite(lr):
                continue
            need = margin - (ld - lr)
            if need > 0.05:
                cut = min(max_cut, need + 0.3)
                cuts[lid] = cuts.get(lid, 0.0) + cut
                t = np.arange(len(g_ctl)) / CRC
                ramp_in = np.clip((t - (s - 0.45)) / 0.35, 0, 1)
                ramp_out = np.clip(((e + 0.5) - t) / 0.5, 0, 1)
                w = np.minimum(ramp_in, ramp_out)
                w = w * w * (3 - 2 * w)
                g_ctl = np.minimum(g_ctl, 1 - w * (1 - db(-cuts[lid])))
                changed = True
        if not changed:
            break
    return ctl_to_audio(g_ctl), cuts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--music', default=os.path.join(AUD, 'music.wav'))
    ap.add_argument('--dialogue-db', type=float, default=0.0)
    ap.add_argument('--music-db', type=float, default=0.0)
    ap.add_argument('--ambience-db', type=float, default=0.0)
    ap.add_argument('--sfx-db', type=float, default=0.0)
    ap.add_argument('--music-duck', type=float, default=7.0, help='dB of music ducking under dialogue')
    ap.add_argument('--amb-duck', type=float, default=4.0, help='dB of ambience ducking under dialogue')
    ap.add_argument('--margin', type=float, default=9.0, help='min LU of dialogue above the rest per line')
    ap.add_argument('--lufs', type=float, default=-16.0)
    ap.add_argument('--tp', type=float, default=-1.0)
    ap.add_argument('--no-plots', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    os.makedirs(os.path.join(BUILD, 'sfx'), exist_ok=True)

    log('loading ...')
    d = build_dialogue()
    mus = load_stereo(a.music, 'music')
    amb = load_stereo(os.path.join(AUD, 'ambience.wav'), 'ambience')
    fx = load_stereo(os.path.join(AUD, 'sfx.wav'), 'sfx')
    have = dict(music=mus is not None, ambience=amb is not None, sfx=fx is not None)
    Z = np.zeros((N, 2))
    mus = Z.copy() if mus is None else mus
    amb = Z.copy() if amb is None else amb
    fx = Z.copy() if fx is None else fx

    # ---------------- bus processing ----------------
    log('bus processing ...')
    # dialogue: HPF 70, gentle compression, a touch of presence; mono -> centre (-3 dB pan law)
    d = sig.sosfilt(sos('highpass', 70), d)
    d = d * compressor_gain(d, -26.0, 2.0, 0.005, 0.15)
    d = shelf_hi(d, 4500, 1.0)
    dlg = np.stack([d, d], 1) * db(a.dialogue_db)  # phantom centre (voices are -20 LUFS mono -> ~-17 stereo)
    # static level targets (integrated, pre-master) so a changed music/sfx render keeps its balance
    TGT = dict(dialogue=None, music=-22.5, ambience=-31.0, sfx=-25.5)
    mus = sig.sosfilt(sos('highpass', 30), mus, axis=0)
    mus = peq(mus, 2800, -2.0, 0.9)
    mus = mus * compressor_gain(mus, -20.0, 1.5, 0.03, 0.4)[:, None]
    amb = sig.sosfilt(sos('highpass', 30), amb, axis=0)
    fx = sig.sosfilt(sos('highpass', 20), fx, axis=0)
    levels_in = {}
    for name, x, off in (('music', mus, a.music_db), ('ambience', amb, a.ambience_db), ('sfx', fx, a.sfx_db)):
        L = lufs(x) if np.any(x) else -np.inf
        levels_in[name] = L
        if np.isfinite(L):
            x *= db(TGT[name] - L + off)
    # ---------------- sidechain ducking ----------------
    key = dialogue_key(d)
    key_a = ctl_to_audio(key)
    duck_m = db(-a.music_duck * key_a)
    duck_a = db(-a.amb_duck * key_a)
    mus *= duck_m[:, None]
    amb *= duck_a[:, None]
    # ---------------- dialogue protection ----------------
    log('dialogue protection ...')
    rest = mus + amb + fx
    g_prot, cuts = protect_dialogue(dlg, rest, a.margin)
    for x in (mus, amb, fx):
        x *= g_prot[:, None]
    if cuts:
        log('  extra per-line cuts on non-dialogue buses (dB): ' +
            ', '.join(f'{k} {v:.1f}' for k, v in sorted(cuts.items())))
    stems = dict(dialogue=dlg, music=mus, ambience=amb, sfx=fx)
    mix = dlg + mus + amb + fx
    # ---------------- master ----------------
    log('master ...')
    glue = compressor_gain(mix, -14.0, 1.3, 0.03, 0.35, knee=8.0)
    mix_g = mix * glue[:, None]
    L0 = lufs(mix_g)
    gain = a.lufs - L0
    lim_max = 0.0
    for it in range(4):
        pre = mix_g * db(gain)
        lg, lim_max = limiter_gain(pre, a.tp - 0.15)
        out = pre * lg[:, None]
        L1 = lufs(out)
        err = a.lufs - L1
        if abs(err) < 0.05:
            break
        gain += err
    total_gain = glue * db(gain) * lg
    out = mix * total_gain[:, None]
    # fix length / safety
    tp = true_peak_env(out)
    tp_db = float(todb(tp.max()))
    if tp_db > a.tp:
        out *= db(a.tp - tp_db - 0.02)
        total_gain *= db(a.tp - tp_db - 0.02)
        tp_db = float(todb(true_peak_env(out).max()))
    final_L = lufs(out)
    write(os.path.join(AUD, 'final_mix.wav'), out)
    for k, x in stems.items():
        write(os.path.join(AUD, f'stem_{k}.wav'), x * total_gain[:, None])
    log(f'  final: {final_L:.2f} LUFS, true peak {tp_db:.2f} dBTP, master gain {gain:+.2f} dB, '
        f'limiter max GR {lim_max:.2f} dB')

    # ---------------- checks / report ----------------
    log('checks ...')
    S = {k: x * total_gain[:, None] for k, x in stems.items()}
    rest_f = S['music'] + S['ambience'] + S['sfx']
    per_line = []
    for lid, s, e in line_windows():
        i, j = int(s * SR), int(e * SR)
        ld, lr = lufs(S['dialogue'][i:j]), lufs(rest_f[i:j])
        per_line.append(dict(id=lid, start=s, end=e, dialogue_lufs=round(ld, 2),
                             rest_lufs=round(lr, 2) if np.isfinite(lr) else None,
                             margin_lu=round(ld - lr, 2) if np.isfinite(lr) else 99.0,
                             extra_cut_db=round(cuts.get(lid, 0.0), 2)))
    min_margin = min(p_['margin_lu'] for p_ in per_line)
    stem_L = {k: (round(lufs(x), 2) if np.any(x) else None) for k, x in S.items()}
    dc = {k: float(np.abs(x.mean(0)).max()) for k, x in S.items()}
    dc['final'] = float(np.abs(out.mean(0)).max())
    rep = dict(duration_s=len(out) / SR, samples=len(out), sr=SR, integrated_lufs=round(final_L, 2),
               true_peak_dbtp=round(tp_db, 2), sample_peak_dbfs=round(float(todb(np.abs(out).max())), 2),
               loudness_range_lu=round(float(METER_LRA(out)), 1), master_gain_db=round(gain, 2),
               limiter_max_gr_db=round(lim_max, 2), inputs_present=have,
               input_integrated_lufs={k: (round(v, 2) if np.isfinite(v) else None) for k, v in levels_in.items()},
               stem_integrated_lufs=stem_L, dc_offset=dc, music_duck_db=a.music_duck, amb_duck_db=a.amb_duck,
               min_dialogue_margin_lu=round(min_margin, 2), lines=per_line,
               bus_offsets_db=dict(dialogue=a.dialogue_db, music=a.music_db, ambience=a.ambience_db, sfx=a.sfx_db))
    json.dump(rep, open(os.path.join(BUILD, 'sfx', 'mix_report.json'), 'w'), indent=1)
    log(f'  stems LUFS: {stem_L}')
    log(f'  min dialogue margin over the rest: {min_margin:.1f} LU  (per line in build/sfx/mix_report.json)')
    log(f'  DC offsets max {max(dc.values()):.2e}')
    if not a.no_plots:
        log('plots ...')
        plots(S, out, key, rep)
    log(f'done in {time.time() - t0:.0f}s -> build/audio/final_mix.wav')


def METER_LRA(x):
    """Loudness range (EBU 3342) from short-term loudness."""
    t, st = loudness_curve(x, 3.0, 1.0)
    st = st[st > -70]
    if len(st) < 3:
        return 0.0
    rel = 10 * np.log10(np.mean(10 ** (st / 10))) - 20
    st = st[st > rel]
    return np.percentile(st, 95) - np.percentile(st, 10)


# ----------------------------------------------------------------------------------------------
# analysis plots
# ----------------------------------------------------------------------------------------------
MAJOR = ['title_in', 'title_out', 'lamp_ignite', 'meteor_start', 'impact', 'run_down', 'boat_out', 'lift',
         'row_back', 'throw1', 'catch', 'stack', 'tower_fall', 'splash', 'dawn_warning', 'idea', 'run_up', 'aim_up',
         'beam_up', 'heart_out', 'heart_in', 'ignite', 'rise', 'whale_appear', 'beam_fade', 'deng_dark',
         'whale_song', 'stardust', 'reboot', 'nuzzle', 'farewell', 'sunrise', 'blink1', 'blink2', 'end_title']
COL = dict(dialogue='#e8e4d8', music='#9fc4ff', ambience='#2e9e8e', sfx='#ffb347', mix='#ff6f91')


def _decorate(ax, ymin, ymax, labels=True, t0=0, t1=None):
    t1 = DUR if t1 is None else t1
    for s in TL['scenes']:
        if t0 <= s['start'] <= t1:
            ax.axvline(s['start'], color='w', lw=0.8, alpha=0.6)
            if labels:
                ax.text(s['start'] + 0.3, ymax, s['id'], color='w', fontsize=8, va='top')
    for c in MAJOR:
        if c in TL['cues'] and t0 <= TL['cues'][c] <= t1:
            ax.axvline(TL['cues'][c], color='#ffd36b', lw=0.5, alpha=0.45, ls='--')
            if labels:
                ax.text(TL['cues'][c], ymin, c, color='#ffd36b', fontsize=5.5, rotation=90, va='bottom',
                        ha='right', alpha=0.9)
    for ln in TL['lines']:
        if ln['end'] >= t0 and ln['start'] <= t1:
            ax.axvspan(ln['start'], ln['end'], color='w', alpha=0.07, lw=0)


def plots(S, out, key, rep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams.update({'axes.facecolor': '#0b1026', 'figure.facecolor': '#070b1f', 'text.color': 'w',
                         'axes.labelcolor': 'w', 'xtick.color': 'w', 'ytick.color': 'w', 'axes.edgecolor': '#445'})
    outdir = os.path.join(BUILD, 'sfx')
    # 1) loudness over time (short-term 3 s + momentary for dialogue) per stem
    fig, axes = plt.subplots(2, 1, figsize=(26, 10), gridspec_kw=dict(height_ratios=[3, 1]), sharex=True)
    ax = axes[0]
    for k in ('ambience', 'music', 'sfx', 'dialogue'):
        if np.any(S[k]):
            t, L = loudness_curve(S[k], 0.4 if k == 'dialogue' else 1.0, 0.05)
            ax.plot(t, L, color=COL[k], lw=1.0 if k != 'dialogue' else 0.8, label=f'{k} ({rep["stem_integrated_lufs"][k]} LUFS)')
    t, L = loudness_curve(out, 3.0, 0.1)
    ax.plot(t, L, color=COL['mix'], lw=2.0, label=f'final mix short-term ({rep["integrated_lufs"]} LUFS int.)')
    ax.axhline(rep['integrated_lufs'], color=COL['mix'], lw=0.6, ls=':')
    ax.set_ylim(-60, -4)
    ax.set_ylabel('loudness (LUFS)  [dialogue: momentary 0.4 s, others 1 s]')
    _decorate(ax, -59, -5)
    ax.legend(loc='lower right', fontsize=9, facecolor='#0b1026', labelcolor='w')
    ax.set_title(f'《一盏灯，一颗星》 final mix — {rep["integrated_lufs"]} LUFS, TP {rep["true_peak_dbtp"]} dBTP, '
                 f'LRA {rep["loudness_range_lu"]} LU, min dialogue margin {rep["min_dialogue_margin_lu"]} LU',
                 color='w', fontsize=12)
    ax2 = axes[1]
    tk = np.arange(len(key)) / 1000.0
    ax2.plot(tk, -rep['music_duck_db'] * key, color=COL['music'], lw=1, label='music duck (dB)')
    ax2.plot(tk, -rep['amb_duck_db'] * key, color=COL['ambience'], lw=1, label='ambience duck (dB)')
    for p_ in rep['lines']:
        if p_['extra_cut_db'] > 0:
            ax2.plot([p_['start'], p_['end']], [-p_['extra_cut_db']] * 2, color=COL['sfx'], lw=3)
    for p_ in rep['lines']:
        ax2.text((p_['start'] + p_['end']) / 2, 1.0, f"{p_['margin_lu']:.0f}", color='w', fontsize=6, ha='center')
    ax2.set_ylim(-15, 2.5)
    ax2.set_ylabel('gain (dB) / margin')
    ax2.set_xlabel('time (s)')
    _decorate(ax2, -14.5, 2, labels=False)
    ax2.legend(loc='lower right', fontsize=8, facecolor='#0b1026', labelcolor='w')
    ax2.set_xlim(0, DUR)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'analysis_loudness.png'), dpi=80)
    plt.close(fig)
    # 2) spectrogram of final mix, 4 rows
    m = out.mean(1)
    rows = 4
    seg = DUR / rows
    fig, axes = plt.subplots(rows, 1, figsize=(26, 16))
    for r_ in range(rows):
        a0, a1 = r_ * seg, (r_ + 1) * seg
        x = m[int(a0 * SR):int(a1 * SR)]
        f, tt, Sx = sig.spectrogram(x, SR, nperseg=2048, noverlap=1536)
        ax = axes[r_]
        Sd = 10 * np.log10(Sx + 1e-14)
        ax.pcolormesh(tt + a0, f, Sd, vmin=-120, vmax=-30, shading='auto', cmap='magma')
        ax.set_yscale('symlog', linthresh=400)
        ax.set_ylim(30, 20000)
        ax.set_xlim(a0, a1)
        _decorate(ax, 35, 18000, t0=a0, t1=a1)
        ax.set_ylabel('Hz')
    axes[0].set_title('final mix spectrogram (scene boundaries white, cues dashed gold, dialogue shaded)', color='w')
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'analysis_spectrogram.png'), dpi=70)
    plt.close(fig)
    # 3) per-scene SFX stem spectrogram with events
    evp = os.path.join(outdir, 'events.json')
    ev = json.load(open(evp))['events'] if os.path.exists(evp) else []
    sc = TL['scenes']
    fig, axes = plt.subplots(len(sc), 1, figsize=(24, 4.2 * len(sc)))
    fxm = S['sfx'].mean(1) + 0.5 * S['ambience'].mean(1)
    for i, s in enumerate(sc):
        ax = axes[i]
        a0, a1 = s['start'], s['end']
        x = fxm[int(a0 * SR):int(a1 * SR)]
        f, tt, Sx = sig.spectrogram(x, SR, nperseg=1024, noverlap=768)
        ax.pcolormesh(tt + a0, f, 10 * np.log10(Sx + 1e-14), vmin=-125, vmax=-35, shading='auto', cmap='magma')
        ax.set_yscale('symlog', linthresh=400)
        ax.set_ylim(30, 20000)
        ax.set_xlim(a0, a1)
        for c, v in TL['cues'].items():
            if a0 <= v <= a1 and not c.endswith(('_start', '_end')):
                ax.axvline(v, color='#6ff0ff', lw=0.8, alpha=0.8)
                ax.text(v, 19000, c, color='#6ff0ff', fontsize=7, rotation=90, va='top', ha='right')
        lastx = -9
        for k_, e in enumerate(ev):
            if a0 <= e['time'] <= a1:
                ax.plot([e['time']], [45 + 20 * (k_ % 4)], marker='^', color='#ffd36b', ms=5)
                if e['time'] - lastx > 0.35:
                    ax.text(e['time'], 60 + 30 * (k_ % 3), e['name'], color='#ffd36b', fontsize=5.5, rotation=90,
                            va='bottom')
                    lastx = e['time']
        for ln in TL['lines']:
            if ln['scene'] == s['id']:
                ax.axvspan(ln['start'], ln['end'], color='w', alpha=0.08, lw=0)
                ax.text(ln['start'], 25000, ln['id'], color='w', fontsize=7)
        ax.set_title(f"{s['id']}  sfx + ambience stems  (events ▲, cues cyan, dialogue shaded)", color='w', fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, 'analysis_sfx_scenes.png'), dpi=60)
    plt.close(fig)
    log('  wrote build/sfx/analysis_loudness.png, analysis_spectrogram.png, analysis_sfx_scenes.png')


if __name__ == '__main__':
    main()
