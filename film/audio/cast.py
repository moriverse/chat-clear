"""Data-driven voice casting: synthesize a test line with every Kokoro zh voice,
measure pitch (F0) and intelligibility (Whisper CER), write a table."""
import json, sys, os, time
import numpy as np, soundfile as sf, librosa
from kokoro import KModel, KPipeline
from faster_whisper import WhisperModel

OUT = os.path.join(os.path.dirname(__file__), '..', 'build', 'cast')
os.makedirs(OUT, exist_ok=True)
repo_id = 'hexgrad/Kokoro-82M-v1.1-zh'
model = KModel(repo_id=repo_id).to('cpu').eval()
zh = KPipeline(lang_code='z', repo_id=repo_id, model=model)
asr = WhisperModel('small', device='cpu', compute_type='int8')
TEXT = "别怕，别怕。我叫阿灯。天亮以前，我一定要回家。"
voices = sys.argv[1].split(',')

def cer(a, b):
    import difflib
    a = ''.join(ch for ch in a if '一' <= ch <= '鿿')
    b = ''.join(ch for ch in b if '一' <= ch <= '鿿')
    sm = difflib.SequenceMatcher(None, a, b)
    return 1 - sm.ratio()

rows = []
for v in voices:
    try:
        a = np.concatenate([r.audio.numpy() for r in zh(TEXT, voice=v, speed=1.0)])
        sf.write(f'{OUT}/{v}.wav', a, 24000)
        f0, vf, _ = librosa.pyin(a, fmin=60, fmax=600, sr=24000, frame_length=1024)
        f0m = float(np.nanmedian(f0)); f0s = float(np.nanstd(f0))
        segs, _ = asr.transcribe(f'{OUT}/{v}.wav', language='zh', initial_prompt='以下是普通话的句子。')
        txt = ''.join(s.text for s in segs)
        row = dict(voice=v, dur=len(a)/24000, f0=f0m, f0std=f0s, cer=cer(TEXT, txt), asr=txt)
    except Exception as e:
        row = dict(voice=v, error=str(e))
    rows.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
