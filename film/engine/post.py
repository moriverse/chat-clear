"""Whole-frame post-processing: bloom, colour grade, vignette, grain, fades.

Scenes can steer it per frame through `f.grade`, e.g.
    f.grade.update(bloom=1.4, exposure=1.1, sat=0.8, tint='#ffd9a0', tint_amt=0.1,
                   vignette=0.5, black=0.0)
Defaults are in DEFAULT below.
"""
from __future__ import annotations

import numpy as np
import skia

from .core import clamp, hexrgb

DEFAULT = dict(
    bloom=1.0,        # bloom strength multiplier (0 disables)
    bloom_thr=0.55,   # luminance threshold for bloom
    exposure=1.0,
    sat=1.0,
    tint=None,        # '#rrggbb' colour multiplied in
    tint_amt=0.0,
    vignette=0.45,
    grain=0.05,
    black=0.0,        # 0..1 fade to black
    white=0.0,        # 0..1 fade to white (flash)
)

_GRAIN = {}


def _grain_image(seed):
    if seed not in _GRAIN:
        g = np.random.default_rng(seed).normal(0.5, 0.18, (256, 256)).clip(0, 1)
        a = (g * 255).astype(np.uint8)
        rgba = np.dstack([a, a, a, np.full_like(a, 255)])
        _GRAIN[seed] = skia.Image.fromarray(rgba, colorType=skia.kRGBA_8888_ColorType)
    return _GRAIN[seed]


def _matrix(exposure, sat, tint, tint_amt):
    # saturation matrix (Rec.709 luma)
    lr, lg, lb = 0.2126, 0.7152, 0.0722
    s = sat
    m = np.array([
        [lr * (1 - s) + s, lg * (1 - s), lb * (1 - s)],
        [lr * (1 - s), lg * (1 - s) + s, lb * (1 - s)],
        [lr * (1 - s), lg * (1 - s), lb * (1 - s) + s],
    ])
    if tint and tint_amt:
        tr = np.array(hexrgb(tint))
        m = m * (1 - tint_amt + tint_amt * tr)[:, None]
    m = m * exposure
    out = []
    for r in range(3):
        out += [float(m[r, 0]), float(m[r, 1]), float(m[r, 2]), 0.0, 0.0]
    out += [0, 0, 0, 1, 0]
    return out


def apply(surface, scale, grade, frame_index):
    g = dict(DEFAULT)
    g.update({k: v for k, v in grade.items() if v is not None})
    c = surface.getCanvas()
    w, h = surface.width(), surface.height()

    # --- bloom (two radii, computed at 1/4 resolution) -------------------------
    if g['bloom'] > 0:
        snap = surface.makeImageSnapshot()
        sw, sh = max(1, w // 4), max(1, h // 4)
        thr = g['bloom_thr']
        k = 1.0 / max(1e-3, 1 - thr)
        thr_m = [k, 0, 0, 0, -thr * k,
                 0, k, 0, 0, -thr * k,
                 0, 0, k, 0, -thr * k,
                 0, 0, 0, 1, 0]
        small = skia.Surface(sw, sh)
        p = skia.Paint(ColorFilter=skia.ColorFilters.Matrix(thr_m))
        small.getCanvas().drawImageRect(snap, skia.Rect.MakeWH(sw, sh),
                                        skia.SamplingOptions(skia.FilterMode.kLinear), p)
        bright = small.makeImageSnapshot()
        glow = skia.Surface(sw, sh)
        gc = glow.getCanvas()
        for sigma, amt in ((3.0 * scale, 0.9), (12.0 * scale, 0.8), (30.0 * scale, 0.6)):
            bp = skia.Paint(ImageFilter=skia.ImageFilters.Blur(sigma, sigma, skia.TileMode.kClamp))
            bp.setBlendMode(skia.BlendMode.kPlus)
            bp.setAlphaf(clamp(amt))
            gc.drawImage(bright, 0, 0, skia.SamplingOptions(), bp)
        gimg = glow.makeImageSnapshot()
        add = skia.Paint(BlendMode=skia.BlendMode.kPlus)
        add.setAlphaf(clamp(0.55 * g['bloom']))
        c.drawImageRect(gimg, skia.Rect.MakeWH(w, h), skia.SamplingOptions(skia.FilterMode.kLinear), add)
        if g['bloom'] > 1.8:  # extra pass for very bright moments
            add.setAlphaf(clamp(0.55 * (g['bloom'] - 1.8)))
            c.drawImageRect(gimg, skia.Rect.MakeWH(w, h), skia.SamplingOptions(skia.FilterMode.kLinear), add)

    # --- colour grade ------------------------------------------------------------
    if g['exposure'] != 1.0 or g['sat'] != 1.0 or (g['tint'] and g['tint_amt']):
        snap = surface.makeImageSnapshot()
        p = skia.Paint(ColorFilter=skia.ColorFilters.Matrix(_matrix(g['exposure'], g['sat'], g['tint'], g['tint_amt'])))
        p.setBlendMode(skia.BlendMode.kSrc)
        c.drawImage(snap, 0, 0, skia.SamplingOptions(), p)

    # --- vignette ----------------------------------------------------------------
    if g['vignette'] > 0:
        r = (w * w + h * h) ** 0.5 / 2
        vp = skia.Paint()
        vp.setShader(skia.GradientShader.MakeRadial(
            skia.Point(w / 2, h / 2), r,
            [skia.Color4f(0, 0, 0, 0).toColor(), skia.Color4f(0, 0, 0, 0).toColor(),
             skia.Color4f(0, 0, 0.02, clamp(g['vignette'])).toColor()],
            [0.0, 0.55, 1.0]))
        c.drawRect(skia.Rect.MakeWH(w, h), vp)

    # --- grain -------------------------------------------------------------------
    if g['grain'] > 0:
        img = _grain_image(frame_index % 7)
        m = skia.Matrix()
        m.setScale(scale * 1.5, scale * 1.5)
        m.postTranslate((frame_index * 97) % 256, (frame_index * 61) % 256)
        gp = skia.Paint()
        gp.setShader(img.makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat,
                                    skia.SamplingOptions(skia.FilterMode.kLinear), m))
        gp.setBlendMode(skia.BlendMode.kOverlay)
        gp.setAlphaf(clamp(g['grain'] * 2))
        c.drawRect(skia.Rect.MakeWH(w, h), gp)

    # --- fades -------------------------------------------------------------------
    if g['white'] > 0:
        c.drawColor(skia.Color4f(1, 0.98, 0.92, clamp(g['white'])).toColor())
    if g['black'] > 0:
        c.drawColor(skia.Color4f(0, 0, 0, clamp(g['black'])).toColor())
