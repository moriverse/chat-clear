"""Generate every dialogue line with Kokoro (local, open-weight TTS), give each
character a distinct sound, compute lip-sync envelopes, and verify each line
with Whisper speech recognition.

Run with the TTS venv:  /opt/tts/bin/python audio/voices.py
Outputs: build/voice/<ID>.wav (48 kHz mono, processed), build/voice/<ID>.dry.wav,
         build/voice/envelopes.npz, build/voice/meta.json
"""
import difflib
import json
import os
import sys

import librosa
import numpy as np
import pyloudnorm as pyln
import pyrubberband as pyrb
import scipy.signal as sig
import soundfile as sf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUT = os.path.join(ROOT, 'build', 'voice')
os.makedirs(OUT, exist_ok=True)
SR_TTS, SR = 24000, 48000

# Data-driven casting (see build/cast_*.jsonl): pitch + Whisper intelligibility
CAST = {
    'narrator': dict(voice='zf_019', speed=0.92, pitch=0.0),
    'deng':     dict(voice='zm_034', speed=0.90, pitch=-0.5),
    'guang':    dict(voice='zf_074', speed=1.00, pitch=2.0),
    'whale':    dict(voice='zf_046', speed=0.82, pitch=-3.0),
}
# per-line delivery tweaks (speed multipliers)
LINE_SPEED = {
    'G01': 0.90, 'G07': 0.85, 'G08': 0.95, 'D13': 0.88, 'D14': 0.80, 'D15': 0.85,
    'G04': 1.08, 'G05': 1.08, 'D07': 1.05, 'D03': 0.95, 'D04': 1.1, 'D16': 0.9,
    'N06': 0.9, 'G09': 0.9, 'W01': 0.9,
}


def reverb_ir(seconds, sr, decay=3.0, seed=0, bright=0.5):
    r = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    ir = r.normal(0, 1, n) * np.exp(-decay * t)
    b, a = sig.butter(1, 2000 + 6000 * bright, fs=sr)
    ir = sig.lfilter(b, a, ir)
    ir[: int(0.01 * sr)] *= np.linspace(0, 1, int(0.01 * sr))
    return ir / np.sqrt(np.sum(ir ** 2))


def add_reverb(x, sr, wet, seconds=1.2, decay=4.0, seed=0, bright=0.5, predelay=0.02):
    ir = reverb_ir(seconds, sr, decay, seed, bright)
    y = sig.fftconvolve(x, ir)[: len(x) + len(ir)]
    y = np.concatenate([np.zeros(int(predelay * sr)), y])
    out = np.zeros(len(y))
    out[: len(x)] += x * (1 - wet * 0.3)
    out += y * wet * 0.35
    return out


def robotize(x, sr):
    """Gentle 'old tin robot': short metallic comb resonance + faint servo buzz, fully intelligible."""
    d = int(0.0045 * sr)
    comb = np.copy(x)
    for k in range(1, 4):
        comb[d * k:] += x[:-d * k] * (0.45 ** k)
    t = np.arange(len(x)) / sr
    ring = x * np.sin(2 * np.pi * 70 * t)
    b, a = sig.butter(2, [300, 5500], btype='band', fs=sr)
    y = 0.75 * x + 0.35 * sig.lfilter(b, a, comb) + 0.06 * ring
    return y


def process(speaker, x):
    if speaker == 'deng':
        x = robotize(x, SR)
        x = add_reverb(x, SR, 0.25, 0.8, 5.0, seed=1)
    elif speaker == 'guang':
        # tiny sparkle: very light shimmer via short chorus
        d = int(0.012 * SR)
        mod = (np.sin(np.arange(len(x)) / SR * 2 * np.pi * 0.8) * 0.5 + 0.5) * d
        idx = np.clip(np.arange(len(x)) - mod.astype(int), 0, len(x) - 1)
        x = 0.85 * x + 0.2 * x[idx]
        x = add_reverb(x, SR, 0.3, 1.0, 4.5, seed=2, bright=0.8)
    elif speaker == 'whale':
        low = pyrb.pitch_shift(x, SR, -12)
        x = x + 0.12 * low
        x = add_reverb(x, SR, 0.6, 3.0, 1.6, seed=3, bright=0.4, predelay=0.05)
    elif speaker == 'narrator':
        x = add_reverb(x, SR, 0.18, 1.0, 5.0, seed=4)
    return x


def envelope(x, sr, rate=100):
    hop = sr // rate
    n = len(x) // hop + 1
    pad = np.pad(x, (0, n * hop - len(x) + hop))
    rms = np.sqrt(np.mean(pad[: n * hop].reshape(n, hop) ** 2, axis=1))
    db = 20 * np.log10(rms + 1e-6)
    ref = np.percentile(db, 95)
    e = np.clip((db - (ref - 30)) / 30, 0, 1) ** 1.3
    # attack/release smoothing
    out = np.zeros_like(e)
    v = 0.0
    for i, s in enumerate(e):
        v = v + (s - v) * (0.6 if s > v else 0.25)
        out[i] = v
    return out.astype(np.float32)


def cer(a, b):
    f = lambda s: ''.join(ch for ch in s if '一' <= ch <= '鿿')
    return 1 - difflib.SequenceMatcher(None, f(a), f(b)).ratio()


def main():
    only = set(sys.argv[1:])
    from kokoro import KModel, KPipeline
    repo_id = 'hexgrad/Kokoro-82M-v1.1-zh'
    model = KModel(repo_id=repo_id).to('cpu').eval()
    zh = KPipeline(lang_code='z', repo_id=repo_id, model=model)
    lines = json.load(open(os.path.join(ROOT, 'script', 'lines.json')))
    meter = pyln.Meter(SR)
    meta_path = os.path.join(OUT, 'meta.json')
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    envs = dict(np.load(os.path.join(OUT, 'envelopes.npz'))) if os.path.exists(os.path.join(OUT, 'envelopes.npz')) else {}
    for ln in lines:
        if only and ln['id'] not in only:
            continue
        cast = CAST[ln['speaker']]
        speed = cast['speed'] * LINE_SPEED.get(ln['id'], 1.0)
        audio = np.concatenate([r.audio.numpy() for r in zh(ln['text'], voice=cast['voice'], speed=speed)])
        x = librosa.resample(audio.astype(np.float64), orig_sr=SR_TTS, target_sr=SR)
        x, _ = librosa.effects.trim(x, top_db=42)
        if cast['pitch']:
            x = pyrb.pitch_shift(x, SR, cast['pitch'])
        x = np.concatenate([np.zeros(int(0.03 * SR)), x, np.zeros(int(0.05 * SR))])
        loud = meter.integrated_loudness(x)
        x = pyln.normalize.loudness(x, loud, -20.0)
        sf.write(os.path.join(OUT, f"{ln['id']}.dry.wav"), x.astype(np.float32), SR)
        env = envelope(x, SR)
        y = process(ln['speaker'], x)
        y = y / max(1.0, np.max(np.abs(y)) / 0.95)
        sf.write(os.path.join(OUT, f"{ln['id']}.wav"), y.astype(np.float32), SR)
        envs[ln['id']] = env
        meta[ln['id']] = dict(speaker=ln['speaker'], speech=len(x) / SR, tail=len(y) / SR, speed=speed)
        print(ln['id'], ln['speaker'], f"{len(x) / SR:.2f}s", flush=True)
    np.savez(os.path.join(OUT, 'envelopes.npz'), **envs)
    json.dump(meta, open(meta_path, 'w'), indent=1, ensure_ascii=False)

    # --- verification with Whisper -------------------------------------------------
    from faster_whisper import WhisperModel
    asr = WhisperModel('small', device='cpu', compute_type='int8')
    report = []
    for ln in lines:
        if only and ln['id'] not in only:
            continue
        segs, _ = asr.transcribe(os.path.join(OUT, f"{ln['id']}.wav"), language='zh',
                                 initial_prompt='以下是普通话的句子。')
        txt = ''.join(s.text for s in segs)
        e = cer(ln['text'], txt)
        report.append(dict(id=ln['id'], cer=round(e, 3), heard=txt, text=ln['text']))
        print(f"  ASR {ln['id']} cer={e:.2f} heard={txt}", flush=True)
    json.dump(report, open(os.path.join(OUT, 'asr_report.json'), 'w'), indent=1, ensure_ascii=False)


if __name__ == '__main__':
    main()
