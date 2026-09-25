"""Renderer CLI.

Examples (run from film/):
  # a few stills of one scene at half resolution
  python -m engine.render --scene s03 --times 0,2.5,5 --scale 0.5
  # contact sheet: 12 evenly spaced frames of a scene -> build/sheets/s03.jpg
  python -m engine.render --scene s03 --sheet 12
  # preview video of a scene with its dialogue audio -> build/preview/s03.mp4
  python -m engine.render --scene s03 --video --scale 0.5
  # full film (final quality)
  python -m engine.render --all --video --scale 1 --workers 4
"""
from __future__ import annotations

import argparse
import glob
import importlib
import math
import os
import subprocess
import sys
import time
import traceback

import numpy as np
import skia

from . import post, subs
from .core import FPS, H, W, col, font, smoothstep
from .timeline import BUILD, ROOT, Frame, Timeline

sys.path.insert(0, ROOT)
_MODULES = {}


def scene_module(sid):
    if sid not in _MODULES:
        files = sorted(glob.glob(os.path.join(ROOT, 'scenes', f'{sid}_*.py')))
        if not files:
            _MODULES[sid] = None
        else:
            name = os.path.splitext(os.path.basename(files[0]))[0]
            _MODULES[sid] = importlib.import_module(f'scenes.{name}')
    return _MODULES[sid]


def _draw_scene(canvas, tl, scene, T, index, strict):
    f = Frame(canvas=canvas, tl=tl, scene=scene, T=T, index=index)
    mod = scene_module(scene['id'])
    try:
        if mod is None:
            raise RuntimeError(f"no scenes/{scene['id']}_*.py yet")
        canvas.save()
        mod.render(f)
        canvas.restore()
    except Exception as e:  # never kill a long render in dev mode
        if strict:
            raise
        canvas.restoreToCount(1)
        canvas.clear(col('#401020'))
        fnt = font(34)
        msg = traceback.format_exc().strip().splitlines()[-4:]
        for i, s in enumerate([f"{scene['id']} t={f.t:.2f}"] + msg):
            canvas.drawString(s[:110], 40, 80 + 44 * i, fnt, skia.Paint(Color=col('#ffffff')))
    return f.grade


def render_frame(tl, index, scale=0.5, subtitles=True, strict=False):
    """Render global frame `index` -> skia.Image (W*scale x H*scale)."""
    T = index / FPS
    w, h = int(round(W * scale)), int(round(H * scale))
    surface = skia.Surface(w, h)
    c = surface.getCanvas()
    c.clear(col('#000000'))
    c.save()
    c.scale(scale, scale)
    scene = tl.scene_at(T)
    grade = _draw_scene(c, tl, scene, T, index, strict)
    c.restore()

    # optional crossfade from the previous scene
    xf = scene.get('xfade', 0)
    si = tl.scenes.index(scene)
    if xf and si > 0 and T < scene['start'] + xf:
        prev = tl.scenes[si - 1]
        a = 1 - smoothstep(scene['start'], scene['start'] + xf, T)
        s2 = skia.Surface(w, h)
        c2 = s2.getCanvas()
        c2.clear(col('#000000'))
        c2.save()
        c2.scale(scale, scale)
        g2 = _draw_scene(c2, tl, prev, T, index, strict)
        c2.restore()
        img2 = s2.makeImageSnapshot()
        p = skia.Paint()
        p.setAlphaf(a)
        c.drawImage(img2, 0, 0, skia.SamplingOptions(), p)
        for k, v in g2.items():  # blend numeric grade values
            if isinstance(v, (int, float)) and isinstance(grade.get(k, post.DEFAULT.get(k)), (int, float)):
                base = grade.get(k, post.DEFAULT.get(k))
                grade[k] = base * (1 - a) + v * a

    post.apply(surface, scale, grade, index)
    c.save()
    c.scale(scale, scale)
    subs.draw(c, tl, T, subtitles)
    c.restore()
    return surface.makeImageSnapshot()


def to_rgb_bytes(img):
    arr = img.toarray(colorType=skia.kRGBA_8888_ColorType)
    return np.ascontiguousarray(arr[:, :, :3]).tobytes()


# ----------------------------------------------------------------------------
def _encode_range(args):
    start, end, scale, path, subtitles, strict, crf = args
    tl = Timeline()
    w, h = int(round(W * scale)), int(round(H * scale))
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium',
           '-crf', str(crf), '-pix_fmt', 'yuv420p', path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(start, end):
        img = render_frame(tl, i, scale, subtitles, strict)
        proc.stdin.write(to_rgb_bytes(img))
        if (i - start) % 48 == 0:
            el = time.time() - t0
            print(f'  [{os.path.basename(path)}] frame {i} ({i - start + 1}/{end - start}) {el / (i - start + 1):.2f}s/f', flush=True)
    proc.stdin.close()
    proc.wait()
    return path


def render_video(tl, frames, scale, out, workers, subtitles, strict, audio=None, crf=18):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmpdir = out + '.parts'
    os.makedirs(tmpdir, exist_ok=True)
    start, end = frames
    n = end - start
    workers = max(1, min(workers, n // 24 or 1))
    chunk = math.ceil(n / workers)
    jobs = []
    for k in range(workers):
        a, b = start + k * chunk, min(end, start + (k + 1) * chunk)
        if a < b:
            jobs.append((a, b, scale, os.path.join(tmpdir, f'part{k:03d}.mp4'), subtitles, strict, crf))
    if workers == 1:
        parts = [_encode_range(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.get_context('fork').Pool(workers) as pool:
            parts = pool.map(_encode_range, jobs)
    lst = os.path.join(tmpdir, 'list.txt')
    with open(lst, 'w') as fh:
        for p in parts:
            fh.write(f"file '{os.path.abspath(p)}'\n")
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', lst]
    if audio and os.path.exists(audio):
        cmd += ['-ss', f'{start / FPS:.3f}', '-t', f'{n / FPS:.3f}', '-i', audio,
                '-map', '0:v', '-map', '1:a', '-c:a', 'aac', '-b:a', '192k']
    cmd += ['-c:v', 'copy', '-movflags', '+faststart', out]
    subprocess.run(cmd, check=True)
    print('wrote', out)


def contact_sheet(tl, scene, n, scale, out, subtitles):
    cols = 4
    rows = math.ceil(n / cols)
    tw, th = int(W * scale), int(H * scale)
    sheet = skia.Surface(cols * tw, rows * (th + 28))
    c = sheet.getCanvas()
    c.clear(col('#202020'))
    fnt = font(20)
    s0 = int(math.ceil(scene['start'] * FPS))
    s1 = int(scene['end'] * FPS) - 1
    for k in range(n):
        i = int(round(s0 + (s1 - s0) * k / max(1, n - 1)))
        img = render_frame(tl, i, scale, subtitles)
        x, y = (k % cols) * tw, (k // cols) * (th + 28)
        c.drawImage(img, x, y)
        c.drawString(f"{scene['id']}  t={i / FPS - scene['start']:.2f}s  (T={i / FPS:.2f})", x + 8, y + th + 21, fnt,
                     skia.Paint(Color=col('#dddddd'), AntiAlias=True))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sheet.makeImageSnapshot().save(out, skia.kJPEG, 88)
    print('wrote', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', action='append', help='scene id (repeatable)')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--times', help='comma list of LOCAL scene times (s) for stills')
    ap.add_argument('--global-times', help='comma list of GLOBAL times (s) for stills')
    ap.add_argument('--sheet', type=int, default=0, help='contact sheet with N frames per scene')
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--scale', type=float, default=None)
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-subs', action='store_true')
    ap.add_argument('--strict', action='store_true')
    ap.add_argument('--audio', default=None, help='audio file to mux (default: best available mix)')
    ap.add_argument('--crf', type=int, default=18)
    ap.add_argument('--range', help='global seconds a,b for video')
    a = ap.parse_args()

    tl = Timeline()
    subtitles = not a.no_subs
    scenes = tl.scenes if a.all else [tl.by_scene[s] for s in (a.scene or [])]
    if not scenes and not a.global_times and not a.range:
        ap.error('give --scene, --all, --global-times or --range')

    if a.global_times:
        scale = a.scale or 0.5
        outdir = a.out or os.path.join(BUILD, 'stills')
        os.makedirs(outdir, exist_ok=True)
        for T in [float(x) for x in a.global_times.split(',')]:
            img = render_frame(tl, int(round(T * FPS)), scale, subtitles, a.strict)
            p = os.path.join(outdir, f'T{T:07.2f}.png')
            img.save(p, skia.kPNG)
            print('wrote', p)
        return

    audio = a.audio
    if audio is None:
        for cand in ('audio/final_mix.wav', 'audio/dialogue.wav'):
            if os.path.exists(os.path.join(BUILD, cand)):
                audio = os.path.join(BUILD, cand)
                break

    if a.range:
        s0, s1 = (float(x) for x in a.range.split(','))
        out = a.out or os.path.join(BUILD, 'preview', f'range_{s0:.0f}_{s1:.0f}.mp4')
        render_video(tl, (int(s0 * FPS), int(s1 * FPS)), a.scale or 0.5, out, a.workers, subtitles, a.strict, audio, a.crf)
        return

    for sc in scenes:
        if a.times:
            scale = a.scale or 0.5
            outdir = a.out or os.path.join(BUILD, 'stills')
            os.makedirs(outdir, exist_ok=True)
            for t in [float(x) for x in a.times.split(',')]:
                i = int(round((sc['start'] + t) * FPS))
                img = render_frame(tl, i, scale, subtitles, a.strict)
                p = os.path.join(outdir, f"{sc['id']}_t{t:06.2f}.png")
                img.save(p, skia.kPNG)
                print('wrote', p)
        if a.sheet:
            contact_sheet(tl, sc, a.sheet, a.scale or 0.25, a.out or os.path.join(BUILD, 'sheets', f"{sc['id']}.jpg"), subtitles)
        if a.video and not a.all:
            out = a.out or os.path.join(BUILD, 'preview', f"{sc['id']}.mp4")
            frames = (int(math.ceil(sc['start'] * FPS)), int(math.ceil(sc['end'] * FPS)))
            render_video(tl, frames, a.scale or 0.5, out, a.workers, subtitles, a.strict, audio, a.crf)

    if a.video and a.all:
        out = a.out or os.path.join(BUILD, 'film.mp4')
        render_video(tl, (0, int(round(tl.duration * FPS))), a.scale or 1.0, out, a.workers, subtitles, a.strict, audio, a.crf)


if __name__ == '__main__':
    main()
