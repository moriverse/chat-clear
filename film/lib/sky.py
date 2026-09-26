"""SKY library — owner: ENV-SKY agent.

Night sky, stars, Milky Way, moon, clouds, dawn and sunrise, plus the special
sky effects of the film (star trails, the hero star, the sky vortex and the
cloud ring blown open by the light pillar).

All functions draw in world/design units onto a skia canvas `c` that the
caller has already transformed (e.g. inside `cam.apply(c, depth)`). Every
function is a pure function of its arguments (and `t`); module-level caches
only hold precomputed, deterministic data.

Recommended draw order for a sky backdrop (or just call `backdrop()`):

    sky.sky(c, rect, horizon_y, dawn, sunrise)
    sky.dawn_glow(c, horizon_y, amount, sun_x)       # when dawn > 0
    sky.milky_way(c, t, ..., horizon_y=horizon_y)
    sky.stars(c, t, rect, bright=..., horizon_y=horizon_y)
    sky.moon(...) / sky.sun(...)
    sky.clouds(c, t, y=..., dawn=..., sunrise=...)
    ... then the sea, which hides everything below horizon_y.

Performance (1920x1080, full frame of sky): sky ~15 ms, milky way ~40 ms,
stars ~25 ms, one cloud band ~40 ms.

CONTRACT: keep every signature below working (you may add keyword args with
defaults and new functions). STUB=True means placeholder art.
"""
from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import skia

from engine.core import (ADD, H, TAU, W, at, clamp, col, fill, hexrgb, lerp, linear, mix, nprng, radial,
                         smoothstep, soft_glow)

STUB = False

# Palette (see STYLE.md)
NIGHT_TOP = '#070b1f'
NIGHT_MID = '#16224d'
NIGHT_HORIZON = '#2c3f7a'
NIGHT_TEAL = '#2e6f8e'
DAWN_TOP = '#27306a'
DAWN_MID = '#8a6fb0'
DAWN_HORIZON = '#f5a878'
SUN_HORIZON = '#ffd08a'
STAR_WARM = '#fff6e0'
STAR_COOL = '#cfe3ff'

DEFAULT_RECT = (-2000, -3000, 5920, 5160)

# Default Milky Way geometry (shared so scenes can reuse it)
MW_CENTER = (960, 250)
MW_ANGLE = -0.35
MW_LENGTH = 3000
MW_WIDTH = 520

_SAMPLING = skia.SamplingOptions(skia.FilterMode.kLinear)
_SAMPLING_MIP = skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear)


# ============================================================================
# small helpers
# ============================================================================
def _rgb(c):
    return hexrgb(c) if isinstance(c, str) else tuple(c[:3])


def _c4(rgb, a=1.0):
    r, g, b = rgb
    return skia.Color4f(clamp(r), clamp(g), clamp(b), clamp(a))


def _device_scale(c):
    """Device pixels per local unit (geometric mean of the matrix axes)."""
    m = c.getTotalMatrix()
    d = abs(m.getScaleX() * m.getScaleY() - m.getSkewX() * m.getSkewY())
    return math.sqrt(d) if d > 1e-12 else 1.0


def _render_scale(c):
    """Resolution scale of the output surface (1.0 = 1920 wide)."""
    try:
        w = c.getBaseLayerSize().width()
        return w / W if w > 0 else 1.0
    except Exception:  # pragma: no cover
        return 1.0


def _zoom_factor(c, size_zoom):
    """World-size multiplier so that point sizes grow like zoom**size_zoom."""
    dev = _device_scale(c)
    zoom = dev / max(1e-6, _render_scale(c))
    return dev, zoom ** (size_zoom - 1.0)


def _visible(c, rect, margin=0.0):
    """Intersection of `rect` (x, y, w, h) with the local clip, grown by margin."""
    b = c.getLocalClipBounds()
    x, y, w, h = rect
    x0, y0 = max(b.left(), x), max(b.top(), y)
    x1, y1 = min(b.right(), x + w), min(b.bottom(), y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0 - margin, y0 - margin, x1 + margin, y1 + margin)


def _hash01(i, j, seed):
    """Vectorised integer hash -> [0,1)."""
    v = (i * 374761393 + j * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    v = ((v ^ (v >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((v ^ (v >> 16)) & 0xFFFFFF) / 16777216.0


def _vnoise2(x, y, seed=0):
    """Smooth 2-D value noise in [0,1] (numpy, vectorised)."""
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    fx = x - xi
    fy = y - yi
    ux = fx * fx * (3 - 2 * fx)
    uy = fy * fy * (3 - 2 * fy)
    a = _hash01(xi, yi, seed)
    b = _hash01(xi + 1, yi, seed)
    cc = _hash01(xi, yi + 1, seed)
    d = _hash01(xi + 1, yi + 1, seed)
    return (a + (b - a) * ux) * (1 - uy) + (cc + (d - cc) * ux) * uy


def _spectral_pair(h, w, beta, seed, ax=1.0, ay=1.0):
    """Two independent periodic gaussian random fields with a 1/f^beta power
    spectrum (spectral synthesis). ax/ay > 1 suppress variation along that
    axis (-> features elongated along it). Returns float32 arrays, std 1."""
    r = np.random.default_rng(seed)
    fy = np.fft.fftfreq(h)[:, None] * ay
    fx = np.fft.fftfreq(w)[None, :] * ax
    k = np.sqrt(fx * fx + fy * fy)
    k[0, 0] = 1.0
    amp = k ** (-beta / 2.0)
    amp[0, 0] = 0.0
    spec = (r.standard_normal((h, w)) + 1j * r.standard_normal((h, w))) * amp
    f = np.fft.ifft2(spec)
    out = []
    for a in (f.real, f.imag):
        a = a - a.mean()
        out.append((a / (a.std() + 1e-12)).astype(np.float32))
    return out


def _smooth1d(n, beta, seed):
    r = np.random.default_rng(seed)
    f = np.fft.rfftfreq(n)
    f[0] = 1.0
    amp = f ** (-beta / 2.0)
    amp[0] = 0.0
    spec = (r.standard_normal(len(f)) + 1j * r.standard_normal(len(f))) * amp
    a = np.fft.irfft(spec, n)
    a -= a.mean()
    return (a / (a.std() + 1e-12)).astype(np.float32)


def _sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _to_image(rgb, alpha=None, seed=0, premul=True):
    """float rgb (h,w,3) in 0..1 -> skia.Image (premultiplied, dithered)."""
    h, w = rgb.shape[:2]
    if alpha is None:
        alpha = np.clip(rgb.max(axis=2), 0, 1)
    dn = np.random.default_rng(seed).random((h, w, 1), dtype=np.float32) - 0.5
    rgba = np.empty((h, w, 4), np.uint8)
    rgba[..., :3] = np.clip(rgb * 255.0 + dn, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.clip(alpha * 255.0 + dn[..., 0], 0, 255).astype(np.uint8)
    rgba[..., 3] = np.maximum(rgba[..., 3], rgba[..., :3].max(axis=2))
    return skia.Image.fromarray(rgba, colorType=skia.kRGBA_8888_ColorType,
                                alphaType=skia.kPremul_AlphaType if premul else skia.kUnpremul_AlphaType)


def _points(xs, ys):
    return [skia.Point(a, b) for a, b in zip(xs.tolist(), ys.tolist())]


_EFFECTS = {}


def _effect(name, src):
    e = _EFFECTS.get(name)
    if e is None:
        e = skia.RuntimeEffect.MakeForShader(src)
        if e is None:  # pragma: no cover
            raise RuntimeError(f'SkSL compile failed: {name}')
        _EFFECTS[name] = e
    return e


def _shader(name, src, uniforms, children=()):
    b = skia.RuntimeShaderBuilder(_effect(name, src))
    for k, v in uniforms.items():
        if isinstance(v, (tuple, list)):
            if len(v) == 2:
                b.setUniform(k, skia.V2(*[float(x) for x in v]))
            elif len(v) == 3:
                b.setUniform(k, skia.V3(*[float(x) for x in v]))
            else:
                b.setUniform(k, skia.V4(*[float(x) for x in v]))
        else:
            b.setUniform(k, float(v))
    for k, sh in children:
        b.setChild(k, sh)
    return b.makeShader()


# ============================================================================
# SKY GRADIENT
# ============================================================================
# stops: distance ABOVE the horizon (world units) -> colour, for 3 palettes.
_SKY_POS = (-220, 0, 30, 110, 260, 520, 900, 2600)
_SKY_NIGHT = ('#22386c', '#2a5580', '#2b5f84', '#2b4a7e', NIGHT_HORIZON, NIGHT_MID, NIGHT_TOP, '#03061a')
_SKY_DAWN = ('#e89a7c', '#ffb88a', '#f2a07a', '#d98a86', '#a8739e', '#6c5a9a', '#3a3a7c', '#1f2760')
_SKY_SUN = ('#ffd89a', '#fff0c0', SUN_HORIZON, '#f9c592', '#e8b9a0', '#a9b4cc', '#6f9ad0', '#4a7cc0')


def sky_color(height_above_horizon, dawn=0.0, sunrise=0.0):
    """Sky colour (r,g,b) at a given height above the horizon — handy for
    fog / atmospheric perspective in other modules."""
    hgt = float(height_above_horizon)
    cols = _sky_stops(dawn, sunrise)
    pos = _SKY_POS
    if hgt <= pos[0]:
        return cols[0]
    for i in range(len(pos) - 1):
        if hgt <= pos[i + 1]:
            u = (hgt - pos[i]) / (pos[i + 1] - pos[i])
            return mix(cols[i], cols[i + 1], u)
    return cols[-1]


def _sky_stops(dawn, sunrise):
    d, s = clamp(dawn), clamp(sunrise)
    out = []
    for n, dw, sn in zip(_SKY_NIGHT, _SKY_DAWN, _SKY_SUN):
        out.append(mix(mix(n, dw, d), sn, s))
    return out


def sky(c, rect=DEFAULT_RECT, horizon_y=700, dawn=0.0, sunrise=0.0, airglow=1.0):
    """Fill `rect` (x, y, w, h in world units) with the sky gradient.

    horizon_y: world y of the sea horizon (the gradient is anchored to it;
               the deep zenith colour is reached ~2600 units above it).
    dawn:      0 = deep night (indigo with a faint teal band at the horizon),
               1 = pre-dawn (violet sky, peach horizon).
    sunrise:   0..1 on top of dawn: golden horizon, pale blue top.
    airglow:   0..1 strength of the faint teal/green airglow band just above
               the horizon at night.
    The gradient is dithered, so dark skies do not band.
    """
    x, y, w, h = rect
    cols = _sky_stops(dawn, sunrise)
    top = horizon_y - _SKY_POS[-1]
    bot = horizon_y - _SKY_POS[0]
    span = bot - top
    pos = [(bot - (horizon_y - p) + 0.0) for p in _SKY_POS]  # placeholder (overwritten below)
    pos = [((horizon_y - p) - top) / span for p in _SKY_POS]
    order = sorted(range(len(pos)), key=lambda i: pos[i])
    p = skia.Paint(AntiAlias=False)
    p.setShader(skia.GradientShader.MakeLinear(
        [skia.Point(0, top), skia.Point(0, bot)],
        [col(cols[i]) for i in order], [pos[i] for i in order]))
    p.setDither(True)
    r = skia.Rect.MakeXYWH(x, y, w, h)
    c.drawRect(r, p)
    # faint airglow: a soft teal-green veil hugging the horizon (night only)
    ag = airglow * (1 - clamp(dawn)) * (1 - clamp(sunrise))
    if ag > 0.01:
        gp = linear((0, horizon_y - 420), (0, horizon_y + 10),
                    [(0, '#1f6f7a', 0.0), (0.55, '#2b8a8a', 0.025 * ag), (0.9, '#3a9a9a', 0.05 * ag),
                     (1, '#3a9a9a', 0.0)])
        gp.setBlendMode(ADD)
        gp.setDither(True)
        c.drawRect(skia.Rect.MakeLTRB(x, max(y, horizon_y - 420), x + w, min(y + h, horizon_y + 10)), gp)


def grade_hint(dawn=0.0, sunrise=0.0):
    """Recommended `f.grade` values for bright skies. The engine's bloom
    thresholds each colour channel at 0.55, so a bright dawn/sunrise sky would
    bloom into neon pinks/whites; raise the threshold as the sky brightens:
        f.grade.update(sky.grade_hint(dawn, sunrise))
    """
    k = max(clamp(dawn) * 0.75, clamp(sunrise))
    return dict(bloom_thr=0.55 + 0.30 * k, bloom=1.0 - 0.25 * k)


# ============================================================================
# STARS
# ============================================================================
_CELL = 1024.0
_CELLS = OrderedDict()
_STAR_COLS = [hexrgb(STAR_COOL), hexrgb('#f2f5ff'), hexrgb(STAR_WARM), hexrgb('#ffd8a6')]
_STAR_COL_P = [0.36, 0.30, 0.26, 0.08]
_POOL = 1.5  # generated pool relative to density=1


def _cell_stars(seed, i, j):
    key = (seed, i, j)
    d = _CELLS.get(key)
    if d is not None:
        _CELLS.move_to_end(key)
        return d
    r = np.random.default_rng([seed & 0x7FFFFFFF, i + 4096, j + 4096])
    nf, nm, nb = r.poisson(1050 * _POOL), r.poisson(175 * _POOL), r.poisson(13 * _POOL)
    n = nf + nm + nb
    x = i * _CELL + r.uniform(0, _CELL, n)
    y = j * _CELL + r.uniform(0, _CELL, n)
    cls = np.concatenate([np.zeros(nf, np.int8), np.ones(nm, np.int8), np.full(nb, 2, np.int8)])
    u = r.random(n)
    size = np.where(cls == 0, 0.85 + 0.45 * u, np.where(cls == 1, 1.3 + 0.8 * u, 2.1 + 1.1 * u))
    u2 = r.random(n)
    alpha = np.where(cls == 0, 0.10 + 0.40 * u2 ** 1.7,
                     np.where(cls == 1, 0.45 + 0.45 * u2, 0.88 + 0.12 * u2))
    # patchy distribution: thin out faint stars in "voids" of a smooth noise
    patch = _vnoise2(x / 700.0, y / 700.0, seed + 5) * 0.65 + _vnoise2(x / 230.0, y / 230.0, seed + 9) * 0.35
    keep = r.random(n) < np.where(cls == 0, 0.25 + 1.0 * patch, 1.0)
    colc = r.choice(4, n, p=_STAR_COL_P).astype(np.int8)
    f1 = r.uniform(0.9, 3.2, n)
    f2 = r.uniform(2.5, 6.0, n)
    p1 = r.uniform(0, TAU, n)
    p2 = r.uniform(0, TAU, n)
    depth = np.where(cls == 0, r.uniform(0.25, 0.6, n), np.where(cls == 1, r.uniform(0.2, 0.45, n),
                                                                 r.uniform(0.12, 0.3, n)))
    glint = (cls == 2) & (r.random(n) < 0.6)
    glen = r.uniform(9, 30, n) * (size / 2.6)
    rank = r.random(n)  # density subsetting: keep rank < density / POOL
    sel = keep
    d = dict(x=x[sel], y=y[sel], size=size[sel], alpha=alpha[sel], cls=cls[sel], colc=colc[sel], f1=f1[sel],
             f2=f2[sel], p1=p1[sel], p2=p2[sel], depth=depth[sel], glint=glint[sel], glen=glen[sel],
             rank=rank[sel])
    _CELLS[key] = d
    while len(_CELLS) > 400:
        _CELLS.popitem(last=False)
    return d


def _gather_stars(seed, box, density):
    x0, y0, x1, y1 = box
    i0, i1 = int(math.floor(x0 / _CELL)), int(math.floor(x1 / _CELL))
    j0, j1 = int(math.floor(y0 / _CELL)), int(math.floor(y1 / _CELL))
    if (i1 - i0 + 1) * (j1 - j0 + 1) > 160:  # absurdly zoomed out: thin the field
        return None
    parts = [_cell_stars(seed, i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]
    thr = min(1.0, density / _POOL)
    out = {}
    for k in parts[0]:
        out[k] = np.concatenate([p[k] for p in parts])
    m = (out['x'] >= x0) & (out['x'] <= x1) & (out['y'] >= y0) & (out['y'] <= y1) & (out['rank'] < thr)
    return {k: v[m] for k, v in out.items()}


_PT_PAINT = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style)
_PT_PAINT.setStrokeCap(skia.Paint.kRound_Cap)
_NA = 40  # alpha quantisation levels (sqrt space)


def _draw_points(c, xs, ys, dsize, alpha, colc, dev, cols=_STAR_COLS, blend=ADD):
    """Batched round points. dsize = diameter in DEVICE pixels, alpha 0..1,
    colc = index into `cols`. Sub-pixel stars are drawn 1 px wide with their
    alpha scaled by area, so previews match a downsampled final render."""
    if len(xs) == 0:
        return
    small = dsize < 1.0
    alpha = np.where(small, alpha * dsize * dsize, alpha)
    dsize = np.maximum(dsize, 1.0)
    sb = np.clip(np.round(np.log2(dsize) * 4.0), 0, 15).astype(np.int32)
    ab = np.clip(np.round(np.sqrt(np.clip(alpha, 0, 1)) * _NA), 0, _NA).astype(np.int32)
    ok = ab > 0
    if not ok.any():
        return
    xs, ys, sb, ab, cc = xs[ok], ys[ok], sb[ok], ab[ok], colc[ok].astype(np.int32)
    key = (cc * 16 + sb) * (_NA + 1) + ab
    order = np.argsort(key, kind='stable')
    ks = key[order]
    cuts = np.flatnonzero(np.diff(ks)) + 1
    starts = np.concatenate([[0], cuts])
    ends = np.concatenate([cuts, [len(ks)]])
    xs, ys = xs[order], ys[order]
    p = _PT_PAINT
    p.setBlendMode(blend)
    for s, e in zip(starts.tolist(), ends.tolist()):
        k = int(ks[s])
        a = (k % (_NA + 1)) / _NA
        k //= (_NA + 1)
        sbin = k % 16
        ci = k // 16
        p.setColor4f(_c4(cols[ci], a * a))
        p.setStrokeWidth((2.0 ** (sbin / 4.0)) / dev)
        c.drawPoints(skia.Canvas.kPoints_PointMode, _points(xs[s:e], ys[s:e]), p)


_GLOW_SHADERS = {}


def _glow_paint(rgb, kind='glow'):
    key = (tuple(round(v, 3) for v in rgb), kind)
    p = _GLOW_SHADERS.get(key)
    if p is None:
        if kind == 'glow':
            stops = [(0.0, 1.0), (0.08, 0.55), (0.25, 0.16), (0.55, 0.04), (1.0, 0.0)]
        elif kind == 'spike':
            stops = [(0.0, 1.0), (0.06, 0.6), (0.22, 0.2), (0.55, 0.05), (1.0, 0.0)]
        elif kind == 'core':
            stops = [(0.0, 1.0), (0.45, 0.9), (1.0, 0.0)]
        else:  # ring
            stops = [(0.0, 0.0), (0.45, 0.0), (0.78, 0.35), (0.9, 1.0), (1.0, 0.0)]
        shader = skia.GradientShader.MakeRadial(
            skia.Point(0, 0), 1.0, [_c4(rgb, a).toColor() for _, a in stops], [s for s, _ in stops])
        p = skia.Paint(AntiAlias=True)
        p.setShader(shader)
        p.setBlendMode(ADD)
        _GLOW_SHADERS[key] = p
    return p


def _blob(c, x, y, rx, ry, paint, alpha, rot=0.0):
    """Draw a unit radial-gradient paint as an ellipse (rx, ry) at (x, y)."""
    if alpha <= 0.003 or rx <= 0 or ry <= 0:
        return
    paint.setAlphaf(clamp(alpha))
    c.save()
    c.translate(x, y)
    if rot:
        c.rotate(math.degrees(rot))
    c.scale(rx, ry)
    c.drawCircle(0, 0, 1.0, paint)
    c.restore()


def _star_glints(c, x, y, rgb, a, glen, thick, glow_r, diag=0.0, rot=0.0):
    """Soft glow + 4-point cross glint (+ optional diagonal pair)."""
    _blob(c, x, y, glow_r, glow_r, _glow_paint(rgb, 'glow'), 0.55 * a)
    if glen > 0:
        sp = _glow_paint(rgb, 'spike')
        _blob(c, x, y, glen, thick, sp, 0.85 * a, rot)
        _blob(c, x, y, thick, glen, sp, 0.85 * a, rot)
        if diag > 0:
            _blob(c, x, y, glen * 0.55 * diag, thick * 0.8, sp, 0.5 * a * diag, rot + math.pi / 4)
            _blob(c, x, y, thick * 0.8, glen * 0.55 * diag, sp, 0.5 * a * diag, rot + math.pi / 4)


def _twinkle(d, t, twinkle):
    s = 0.6 * np.sin(d['f1'] * t + d['p1']) + 0.4 * np.sin(d['f2'] * t + d['p2'])
    return 1.0 - d['depth'] * np.clip(twinkle, 0, 2) * (0.5 + 0.5 * s)


def stars(c, t, rect=DEFAULT_RECT, density=1.0, bright=1.0, twinkle=1.0, seed=7, horizon_y=None,
          fade=320.0, size=1.0, glints=1.0, size_zoom=0.5, warm=0.0):
    """Starfield covering `rect` (x, y, w, h world units; stars exist everywhere,
    generated lazily per 1024-unit cell, so any rect works).

    t:         time (s) — stars twinkle gently.
    density:   0..1.5 fraction of stars (1 = thousands of faint, hundreds of
               medium, a few dozen bright per 1920x1080 screen).
    bright:    global brightness multiplier (0 = invisible, e.g. at sunrise).
    twinkle:   twinkle depth multiplier (0 = steady).
    seed:      star layout seed (deterministic).
    horizon_y: if given, stars fade out over `fade` units above that line.
    size:      star size multiplier.
    glints:    0..1 strength of the glow + 4-point cross glints on bright stars.
    size_zoom: how much stars grow when the camera zooms (device size grows
               like zoom**size_zoom; 1 = like any world object, 0 = constant).
    warm:      -1..1 shifts the star colours cooler / warmer.
    Drawn additively with batched drawPoints (brightness buckets); vectors,
    so they stay crisp at any zoom.
    """
    if bright <= 0.002:
        return
    box = _visible(c, rect, margin=40)
    if box is None:
        return
    d = _gather_stars(seed, box, density)
    if d is None or len(d['x']) == 0:
        return
    dev, kz = _zoom_factor(c, size_zoom)
    a = d['alpha'] * _twinkle(d, t, twinkle) * bright
    if horizon_y is not None:
        hf = np.clip((horizon_y - d['y']) / max(1.0, fade), 0, 1)
        a = a * hf * hf * (3 - 2 * hf)
    cols = _STAR_COLS
    if warm:
        wt = clamp(abs(warm))
        tgt = hexrgb('#ffd9a0') if warm > 0 else hexrgb('#b8d4ff')
        cols = [mix(cc, tgt, wt * 0.6) for cc in _STAR_COLS]
    ws = d['size'] * size * kz  # world size
    _draw_points(c, d['x'], d['y'], ws * dev, a, d['colc'], dev, cols)
    # glow + glints for bright stars
    if glints > 0:
        idx = np.flatnonzero((d['cls'] == 2) & (a > 0.05))
        tw = 0.8 + 0.2 * np.sin(d['f1'] * 1.7 * t + d['p2'])
        for i in idx.tolist():
            rgb = cols[int(d['colc'][i])]
            ai = float(a[i]) * glints
            s = float(ws[i])
            gl = float(d['glen'][i]) * kz * size * float(tw[i]) if d['glint'][i] else 0.0
            thick = max(0.55 * s, 1.0 / dev)
            _star_glints(c, float(d['x'][i]), float(d['y'][i]), rgb, ai, gl, thick, s * 4.2)


# ============================================================================
# MILKY WAY
# ============================================================================
_MW = {}
_MW_TW, _MW_TH = 2048, 512


def _mw_data():
    """Precompute the Milky Way band texture (numpy spectral noise) once."""
    if 'img' in _MW:
        return _MW
    TW, TH = _MW_TW, _MW_TH
    u = ((np.arange(TW, dtype=np.float32) + 0.5) / TW)[None, :]
    v = ((np.arange(TH, dtype=np.float32) + 0.5) / TH * 2 - 1)[:, None]
    a1, a2 = _spectral_pair(TH, TW, 3.4, 101, ax=3.0, ay=1.0)   # large mottling (elongated)
    b1, b2 = _spectral_pair(TH, TW, 3.0, 202, ax=3.5, ay=1.0)   # dust lanes
    c1, c2 = _spectral_pair(TH, TW, 2.7, 303, ax=1.6, ay=1.0)   # star-cloud texture
    g1, _g2 = _spectral_pair(TH, TW, 0.8, 304, ax=1.0, ay=0.72)  # grain
    m1 = _smooth1d(TW, 4.0, 404)[None, :]
    m2 = _smooth1d(TW, 3.5, 405)[None, :]
    m3 = _smooth1d(TW, 3.0, 406)[None, :]
    m4 = _smooth1d(TW, 3.0, 407)[None, :]
    bulge = np.exp(-((u - 0.42) / 0.14) ** 2)
    along = (0.8 + 0.22 * m4) * (1.0 + 0.3 * bulge)
    vc = 0.05 * m1 - 0.02 * bulge
    wm = (1.0 + 0.14 * m2) * (1.0 + 0.2 * bulge)
    vv = (v - vc) / wm
    core = np.exp(-(vv / 0.14) ** 2)
    halo = np.exp(-(vv / 0.36) ** 2)
    outer = np.exp(-(v / 0.70) ** 2)
    mott = np.exp(0.42 * a1 + 0.42 * c1)
    # dust: a central rift that wanders and breaks up, smooth lanes, patches
    rc = 0.02 + 0.045 * m3 + 0.025 * a2
    rw = 0.07 + 0.03 * m2 + 0.015 * c2
    rift = np.exp(-((vv - rc) / rw) ** 2) * (_sstep(0.1, 0.28, u) * _sstep(0.9, 0.62, u))
    rift *= 0.5 + 0.5 * np.tanh(1.3 * b2 + 0.7)
    lanes = _sstep(0.55, 1.0, 1.0 - np.abs(b1)) ** 2 * (0.25 + 0.75 * halo) * _sstep(-0.8, 0.8, a2)
    patch = _sstep(0.3, 2.0, a2 + 0.4 * c2) * halo
    dust = 1.0 - (1.0 - 0.88 * rift) * (1.0 - 0.7 * lanes) * (1.0 - 0.55 * patch)
    i_core = core * along * mott * (1.0 - dust)
    i_halo = halo * np.sqrt(along) * np.exp(0.25 * a1) * (1.0 - 0.4 * dust) * (1.0 - 0.6 * rift)
    i_glow = np.exp(-(vv / 0.22) ** 2) * along * (1.0 - 0.5 * dust) * (1.0 - 0.8 * rift)
    i_outer = outer * (0.9 + 0.1 * a1)
    grain = _sstep(1.6, 3.2, g1) * (core + 0.4 * halo) * (1.0 - dust)
    knots = _sstep(2.2, 3.0, c2 + 0.5 * a1) * halo * (1 - dust)
    end = (_sstep(0.0, 0.2, u) * _sstep(1.0, 0.8, u))
    warmth = (np.clip(i_core, 0, 1) ** 0.7)[..., None]
    lav = np.array([0.62, 0.58, 0.98], np.float32)
    cream = np.array([1.00, 0.80, 0.63], np.float32)
    rgb = (0.045 * i_outer[..., None] * np.array([0.24, 0.32, 0.70], np.float32)
           + 0.065 * i_halo[..., None] * np.array([0.46, 0.50, 0.92], np.float32)
           + 0.13 * i_core[..., None] * (lav * (1 - warmth) + cream * warmth)
           + 0.05 * i_glow[..., None] * np.array([0.80, 0.70, 0.95], np.float32)
           + 0.07 * grain[..., None] * np.array([0.95, 0.95, 1.0], np.float32)
           + 0.06 * knots[..., None] * np.array([1.0, 0.48, 0.70], np.float32))
    rgb = rgb * end[..., None]
    rgb = 1.0 - np.exp(-rgb * 1.2)
    _MW['img'] = _to_image(rgb, seed=7).withDefaultMipmaps()
    # star-cloud probability map (for band stars)
    dens = (0.2 * halo + i_core) * (1 - dust) * end
    _MW['dens'] = (dens / dens.max()).astype(np.float32)
    return _MW


_MW_CACHE = OrderedDict()


def _band_geom(center, angle, length, width, bend):
    """Returns pos(s, v) mapping band coords (s in 0..1 along, v in -1..1
    across, v=+1 on the upper side) -> world (x, y) arrays."""
    dx, dy = math.cos(angle), math.sin(angle)
    nx, ny = -dy, dx
    if ny > 0:
        nx, ny = -nx, -ny  # normal pointing up on screen
    cx, cy = center

    def pos(s, v):
        s = np.asarray(s, np.float64)
        v = np.asarray(v, np.float64)
        q = 2 * s - 1
        # arch: the middle stays at `center`, the ends drop by bend*length
        px = cx + (s - 0.5) * length * dx - bend * length * q * q * nx
        py = cy + (s - 0.5) * length * dy - bend * length * q * q * ny
        tx = dx - bend * 4 * q * nx
        ty = dy - bend * 4 * q * ny
        tl = np.sqrt(tx * tx + ty * ty)
        mx, my = -ty / tl, tx / tl
        sgn = np.where(mx * nx + my * ny >= 0, 1.0, -1.0)
        return px + v * width * mx * sgn, py + v * width * my * sgn

    return pos


def _mw_band(center, angle, length, width, bend, horizon_y, fade, seed):
    key = (tuple(round(float(v), 2) for v in center), round(angle, 4), round(length, 1), round(width, 1),
           round(bend, 4), None if horizon_y is None else round(horizon_y, 1), round(fade, 1), seed)
    b = _MW_CACHE.get(key)
    if b is not None:
        _MW_CACHE.move_to_end(key)
        return b
    mw = _mw_data()
    pos = _band_geom(center, angle, length, width, bend)
    ns, nv = 72, 9
    S, V = np.meshgrid(np.linspace(0, 1, ns + 1), np.linspace(-1, 1, nv))
    PX, PY = pos(S, V)
    pts = _points(PX.ravel(), PY.ravel())
    texs = _points((S.ravel() * _MW_TW), ((V.ravel() + 1) * 0.5 * _MW_TH))
    if horizon_y is not None:
        hf = np.clip((horizon_y - PY.ravel()) / max(1.0, fade), 0, 1)
        hf = hf * hf * (3 - 2 * hf)
    else:
        hf = np.ones(PX.size)
    cols = [skia.Color4f(float(a), float(a), float(a), float(a)).toColor() for a in hf]
    idx = []
    for j in range(nv - 1):
        for i in range(ns):
            a0 = j * (ns + 1) + i
            a1, b0 = a0 + 1, a0 + ns + 1
            b1 = b0 + 1
            idx += [a0, a1, b1, a0, b1, b0]
    verts = skia.Vertices(skia.Vertices.kTriangles_VertexMode, pts, texs, cols, idx)
    # band stars: rejection-sampled against the star-cloud density map
    r = np.random.default_rng(seed)
    area = length * width * 2
    n = int(min(40000, 26000 * area / (3000 * 1040)))
    s = r.random(n * 2)
    v = np.clip(r.normal(0, 0.33, n * 2), -1, 1)
    dens = mw['dens']
    ix = np.clip((s * _MW_TW).astype(int), 0, _MW_TW - 1)
    iy = np.clip(((v + 1) * 0.5 * _MW_TH).astype(int), 0, _MW_TH - 1)
    keep = r.random(n * 2) < (dens[iy, ix] * 1.4 + 0.02)
    s, v = s[keep][:n], v[keep][:n]
    m = len(s)
    sx, sy = pos(s, v)
    dd = dens[np.clip(((v + 1) * 0.5 * _MW_TH).astype(int), 0, _MW_TH - 1), np.clip((s * _MW_TW).astype(int), 0, _MW_TW - 1)]
    u = r.random(m)
    bright_ = u > 0.985
    size = np.where(bright_, 1.3 + 0.8 * r.random(m), 0.7 + 0.5 * r.random(m))
    alpha = np.where(bright_, 0.5 + 0.4 * r.random(m), 0.08 + 0.32 * r.random(m) ** 1.5) * (0.6 + 0.4 * dd)
    if horizon_y is not None:
        hf2 = np.clip((horizon_y - sy) / max(1.0, fade), 0, 1)
        alpha = alpha * hf2 * hf2 * (3 - 2 * hf2)
    colc = r.choice(4, m, p=[0.34, 0.36, 0.24, 0.06]).astype(np.int8)
    st = dict(x=sx, y=sy, size=size, alpha=alpha, colc=colc, f1=r.uniform(0.9, 3.0, m),
              f2=r.uniform(2.5, 6, m), p1=r.uniform(0, TAU, m), p2=r.uniform(0, TAU, m),
              depth=r.uniform(0.2, 0.55, m))
    b = dict(verts=verts, stars=st)
    _MW_CACHE[key] = b
    while len(_MW_CACHE) > 8:
        _MW_CACHE.popitem(last=False)
    return b


def milky_way(c, t, center=MW_CENTER, angle=MW_ANGLE, length=MW_LENGTH, width=MW_WIDTH, alpha=1.0,
              bend=0.0, horizon_y=None, fade=420.0, stars=1.0, twinkle=1.0, seed=11, size_zoom=0.5):
    """The Milky Way: a luminous band with nebulous structure (star clouds,
    dark dust lanes and a central rift, soft blue/violet halo, peach core,
    faint pink knots) plus thousands of tiny band stars. Drawn additively,
    as a textured mesh (precomputed 2048x512 texture), so it is cheap and
    stays put in world space while the camera moves.

    center:    world point of the band's middle.
    angle:     direction of the band (radians, 0 = pointing right).
    length:    band length in world units (the ends fade out softly).
    width:     half-width of the band's glow region (core ~0.3*width wide).
    alpha:     overall brightness 0..1.
    bend:      arch the band (0 = straight; 0.1 = ends curve 0.1*length down
               relative to the middle — like the Milky Way over a wide lens).
    horizon_y: if given, the band fades out over `fade` units above it.
    stars:     density multiplier for the band's tiny stars (0 = none).
    t, twinkle: band stars twinkle.
    Suggested for a full-sky tilt shot (s01): center=(960,-700), angle=-1.05,
    length=6200, width=700, bend=0.06.
    """
    if alpha <= 0.002:
        return
    mw = _mw_data()
    b = _mw_band(center, angle, length, width, bend, horizon_y, fade, seed)
    if 'shader' not in mw:
        mw['shader'] = mw['img'].makeShader(skia.TileMode.kClamp, skia.TileMode.kClamp, _SAMPLING_MIP)
    p = skia.Paint(AntiAlias=True, Shader=mw['shader'], BlendMode=ADD)
    p.setAlphaf(clamp(alpha))
    c.drawVertices(b['verts'], p, skia.BlendMode.kModulate)
    if stars > 0:
        st = b['stars']
        box = _visible(c, DEFAULT_RECT if False else (-1e7, -1e7, 2e7, 2e7), margin=10)
        if box is None:
            return
        x0, y0, x1, y1 = box
        m = (st['x'] >= x0) & (st['x'] <= x1) & (st['y'] >= y0) & (st['y'] <= y1)
        if stars < 1:
            m &= (np.arange(len(st['x'])) % 100) < stars * 100
        if not m.any():
            return
        d = {k: v[m] for k, v in st.items()}
        dev, kz = _zoom_factor(c, size_zoom)
        a = d['alpha'] * _twinkle(d, t, twinkle) * clamp(alpha)
        _draw_points(c, d['x'], d['y'], d['size'] * kz * dev, a, d['colc'], dev)


# ============================================================================
# MOON
# ============================================================================
def _moon_lit_path(r, phase, n=72):
    ph = phase % 1.0
    waning = ph > 0.5
    k = math.cos(TAU * ph)
    pts = []
    for i in range(n + 1):
        th = math.pi * i / n
        pts.append((r * math.sin(th), -r * math.cos(th)))
    for i in range(n, -1, -1):
        th = math.pi * i / n
        pts.append((r * k * math.sin(th), -r * math.cos(th)))
    if waning:
        pts = [(-x, y) for x, y in pts]
    path = skia.Path()
    path.moveTo(*pts[0])
    for q in pts[1:]:
        path.lineTo(*q)
    path.close()
    return path


def moon(c, x, y, r=60, phase=0.15, glow=1.0, angle=0.5, earthshine=0.22, sky_col=None):
    """Moon (default: a thin waxing crescent) with a cool halo.

    (x, y): centre of the full lunar disc; r: its radius (world units).
    phase:  0 = new, 0.25 = first quarter, 0.5 = full, 0.75 = last quarter.
    glow:   halo strength 0..1+ (0 = no halo).
    angle:  rotation of the lit side (radians; 0 = lit on the right).
    earthshine: faint visibility of the dark part of the disc (0..1).
    sky_col: colour used to hide stars behind the dark part (default night mid).
    """
    lit_frac = (1 - math.cos(TAU * (phase % 1.0))) / 2
    side = 1 if (phase % 1.0) <= 0.5 else -1
    gx = x + math.cos(angle) * side * r * (1 - lit_frac) * 0.55
    gy = y + math.sin(angle) * side * r * (1 - lit_frac) * 0.55
    g = glow * (0.35 + 0.65 * math.sqrt(lit_frac + 0.02))
    lit = _moon_lit_path(r, phase)
    with at(c, x, y, rot=angle):
        # dark disc hides stars behind it, lit faintly by earthshine
        c.drawCircle(0, 0, r * 0.995, fill(sky_col or '#141d42', 1.0))
        c.drawCircle(0, 0, r, fill('#3a4a7a', earthshine))
        lp = radial((r * 0.25 * side, -r * 0.2), r * 1.25,
                    [(0, '#fffdf2', 1), (0.6, '#f4efdc', 1), (1, '#d8d4c8', 1)])
        c.drawPath(lit, lp)
        c.save()
        c.clipPath(lit, doAntiAlias=True)
        for (mx, my, mr, ma) in ((0.25, -0.3, 0.22, 0.10), (0.45, 0.15, 0.18, 0.08), (0.1, 0.35, 0.26, 0.07),
                                 (0.6, -0.1, 0.12, 0.06), (0.72, 0.42, 0.10, 0.05)):
            c.drawCircle(side * mx * r, my * r, mr * r, fill('#a9a896', ma, blur=mr * r * 0.35))
        c.restore()
        # bright limb rim
        rim = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style, StrokeWidth=max(1.0, r * 0.04))
        rim.setColor4f(_c4(hexrgb('#ffffff'), 0.35))
        rim.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, r * 0.03))
        rim.setBlendMode(ADD)
        c.save()
        c.clipPath(lit, doAntiAlias=True)
        c.drawCircle(0, 0, r * 0.985, rim)
        c.restore()
    # halo last: atmospheric glow in front of the (dark part of the) disc
    if g > 0:
        soft_glow(c, gx, gy, r * 9, '#8fb0ff', 0.10 * g)
        soft_glow(c, gx, gy, r * 3.2, '#bcd0ff', 0.20 * g)
        soft_glow(c, gx, gy, r * 1.5, '#dfe8ff', 0.18 * g)


# ============================================================================
# CLOUDS (baked band textures with optical-depth lighting, tinted per call)
# ============================================================================
_CLOUD = {}
_CL_W, _CL_H = 4096, 256
_CL_UX = 1.5  # world units per texel along x (at scale 1) -> 6144-unit period


def _cloud_band(variant):
    """Band texture (periodic in x) of picture-book clouds: clusters of soft
    round puffs on flat-ish bases plus faint wisps (metaballs, perturbed by
    noise). Channels: R = soft density (0.5 = default edge), G = light
    transmitted from above (exp(-optical depth)), B = light from below."""
    img = _CLOUD.get(variant)
    if img is not None:
        return img
    h, w = _CL_H, _CL_W
    uy = 300.0 / h                      # world units per texel vertically
    r = np.random.default_rng(1200 + variant)
    field = np.zeros((h, w), np.float32)
    ys = ((np.arange(h, dtype=np.float32) + 0.5) * uy)[:, None]   # world y within band (0..300)

    def blob(cx, cy, rx, ry, amp, base=None):
        # cx, rx in world units along x (periodic), cy, ry in world units
        x0 = int(math.floor((cx - 3 * rx) / _CL_UX))
        x1 = int(math.ceil((cx + 3 * rx) / _CL_UX))
        y0 = max(0, int((cy - 3 * ry) / uy))
        y1 = min(h, int((cy + 3 * ry) / uy) + 1)
        if y1 <= y0:
            return
        xs = (np.arange(x0, x1) + 0.5) * _CL_UX
        yy = ys[y0:y1]
        g = amp * np.exp(-(((xs[None, :] - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2))
        if base is not None:  # flatten the bottom
            g = g * _sstep(base + ry * 0.35, base - ry * 0.1, yy)
        cols = np.arange(x0, x1) % w       # wrap around: the band tiles in x
        # soft union: keeps every puff's round bump visible (a plain sum melts them)
        cur = field[y0:y1, cols]
        field[y0:y1, cols] = cur + g - cur * g

    ncl = 16
    xs_ = np.sort(r.uniform(0, w * _CL_UX, ncl))
    for i in range(ncl):
        cx = xs_[i] + r.uniform(-100, 100)
        small = r.random() < 0.35
        wd = r.uniform(120, 260) if small else r.uniform(260, 660)   # cluster width (world)
        base = r.uniform(160, 238)           # bottom line (world y inside the 300-unit band)
        hgt = wd * r.uniform(0.13, 0.22) + 14
        blob(cx, base - hgt * 0.3, wd * 0.5, hgt * 0.3, 0.85, base)   # flat body
        npf = int(2 + wd / 120)
        for j in range(npf):                 # main puffs along the top, bigger in the middle
            u = (j + 0.5) / npf * 2 - 1 + r.uniform(-0.1, 0.1)
            prof = math.sqrt(max(0.0, 1 - u * u))
            pr = (0.45 + 0.55 * prof) * hgt * r.uniform(0.5, 0.72) + 8
            px = cx + u * wd * 0.4
            py = base - hgt * 0.22 - prof * hgt * 0.32
            blob(px, py, pr * 1.1, pr * 0.95, r.uniform(0.78, 0.95), base)
        for j in range(r.integers(1, 3)):    # one or two taller crown puffs
            u = r.uniform(-0.45, 0.45)
            pr = hgt * r.uniform(0.3, 0.45) + 6
            blob(cx + u * wd * 0.4, base - hgt * 0.62, pr * 1.1, pr * 0.95, r.uniform(0.75, 0.9), base)
    from scipy.ndimage import gaussian_filter1d
    a, _b = _spectral_pair(h, w, 2.6, 1300 + variant, ax=1.0, ay=0.9)
    dens = np.clip((field - 0.5) * 1.5 + 0.5 + 0.05 * a, 0, 1)
    alpha = _sstep(0.33, 0.67, dens)
    k = 0.055
    top = np.exp(-k * np.cumsum(alpha, axis=0))
    bot = np.exp(-k * np.cumsum(alpha[::-1], axis=0))[::-1]
    top = gaussian_filter1d(top, 5.0, axis=1, mode='wrap')
    bot = gaussian_filter1d(bot, 5.0, axis=1, mode='wrap')
    rgba = np.empty((h, w, 4), np.uint8)
    rgba[..., 0] = np.clip(dens * 255 + 0.5, 0, 255)
    rgba[..., 1] = np.clip(top * 255 + 0.5, 0, 255)
    rgba[..., 2] = np.clip(bot * 255 + 0.5, 0, 255)
    rgba[..., 3] = 255
    img = skia.Image.fromarray(rgba, colorType=skia.kRGBA_8888_ColorType).withDefaultMipmaps()
    _CLOUD[variant] = img
    return img


def _cloud_field():
    """Periodic isotropic fbm texture (R, G = two fields), used by the swirl."""
    if 'img' in _CLOUD:
        return _CLOUD
    N = 512
    a, b = _spectral_pair(N, N * 2, 3.7, 777)
    c1, _ = _spectral_pair(N, N * 2, 2.9, 778)
    f = 0.8 * a + 0.2 * c1
    f = (f - f.min()) / (f.max() - f.min())
    g = (b - b.min()) / (b.max() - b.min())
    rgba = np.empty((N, N * 2, 4), np.uint8)
    rgba[..., 0] = np.clip(f * 255 + 0.5, 0, 255).astype(np.uint8)
    rgba[..., 1] = np.clip(g * 255 + 0.5, 0, 255).astype(np.uint8)
    rgba[..., 2] = 0
    rgba[..., 3] = 255
    _CLOUD['img'] = skia.Image.fromarray(rgba, colorType=skia.kRGBA_8888_ColorType)
    _CLOUD['size'] = (N * 2, N)
    return _CLOUD


def _cloud_colors(color, dawn, sunrise, light, under):
    """-> (lit, shadow, under) colours for the cloud shading formula
    colour = shadow + (lit - shadow) * T_top + under * T_bottom."""
    d, s = clamp(dawn), clamp(sunrise)
    base = _rgb(color)
    lit_n = mix(base, '#c0cff6', 0.62)
    sh_n = tuple(v * 0.6 for v in base)
    lit = mix(mix(lit_n, '#c4a6dc', d), '#fff2dc', s)
    shadow = mix(mix(sh_n, '#4a3d78', d), '#c9959c', s)
    if light is not None:
        lit = _rgb(light)
    und = _rgb(under) if under is not None else mix('#000000', mix('#ff9a78', '#ffc066', s), max(d * 0.9, s))
    return lit, shadow, und


def _cloud_filter(lit, shadow, und, alpha, cover, soft=0.2):
    thr = 0.5 + (0.5 - clamp(cover)) * 0.8
    g = 1.0 / soft
    m = [0, lit[0] - shadow[0], und[0], 0, shadow[0],
         0, lit[1] - shadow[1], und[1], 0, shadow[1],
         0, lit[2] - shadow[2], und[2], 0, shadow[2],
         g * alpha, 0, 0, 0, (0.5 - thr * g) * alpha]
    return skia.ColorFilters.Matrix(m)


def clouds(c, t, y=500, alpha=0.5, color='#34467f', seed=3, speed=6.0, scale=1.0, rect_x=(-2000, 3920),
           dawn=0.0, sunrise=0.0, cover=0.5, thickness=150.0, layers=2, light=None, under=None,
           streak=0.0, haze=None):
    """A drifting band of soft, layered clouds centred on world y.

    alpha:     opacity 0..1 (mapped to 1-(1-alpha)^2 so bodies stay defined;
               0.2..0.35 = faint, 0.5 = solid silhouettes).
    color:     body colour at night (moonlit blue-grey); dawn/sunrise re-tint it.
    seed:      layout variation (picks texture variant + offsets).
    speed:     drift speed in world units / s (positive = to the right).
    scale:     size of the cloud shapes and of the band.
    rect_x:    horizontal extent (x0, x1) in world units.
    dawn, sunrise: 0..1 tint (violet bodies with peach undersides at dawn,
               warm peach/gold at sunrise).
    cover:     0..1 coverage (0.5 = broken clouds, 0.8 = nearly overcast).
    thickness: half-height of the band in world units (times scale).
    layers:    1..3 parallax sub-layers (back layers lower, smaller, paler,
               slower — atmospheric perspective).
    light:     override colour of the lit tops (default moonlight/dawn/sun).
    under:     override colour of the light on the undersides.
    streak:    0..1 long-exposure look for time-lapses (clouds smear along x).
    haze:      colour back layers fade towards (default: the sky colour).
    """
    if alpha <= 0.003:
        return
    b = c.getLocalClipBounds()
    x0, x1 = max(rect_x[0], b.left()), min(rect_x[1], b.right())
    if x1 <= x0:
        return
    lit, shadow, und = _cloud_colors(color, dawn, sunrise, light, under)
    rng_ = nprng(seed * 31 + 5)
    layers = max(1, min(3, int(layers)))
    st = clamp(streak)
    for k in reversed(range(layers)):
        # k = 0 front layer; higher k = further away: lower, smaller, paler
        sk = scale * (1.0 - 0.28 * k)
        hh = thickness * sk
        yc = y + k * thickness * scale * 0.42
        ox = rng_.uniform(0, _CL_W * _CL_UX)
        y0, y1 = max(yc - hh, b.top()), min(yc + hh, b.bottom())
        if y1 <= y0:
            continue
        hz = _rgb(haze) if haze is not None else sky_color(max(0.0, 700 - yc + 200), dawn, sunrise)
        fk = 0.3 * k
        l_k, s_k, u_k = mix(lit, hz, fk * 0.8), mix(shadow, hz, fk), tuple(v * (1 - fk) for v in und)
        m = skia.Matrix()
        sx = _CL_UX * sk * (1.0 + 2.5 * st)
        m.setScale(sx, 2 * hh / _CL_H)
        m.postTranslate(ox + t * speed * (1.0 - 0.3 * k) - 3000, yc - hh)
        img = _cloud_band((seed + k * 7) % 5)
        sh = img.makeShader(skia.TileMode.kRepeat, skia.TileMode.kDecal, _SAMPLING_MIP, m)
        p = skia.Paint(AntiAlias=False, Shader=sh)  # (setShader is very slow in skia-python)
        a_k = clamp(alpha * (1 - 0.25 * k) * (1 - 0.5 * st))
        a_k = 1 - (1 - a_k) ** 2        # denser bodies, so edges stay defined
        p.setColorFilter(_cloud_filter(l_k, s_k, u_k, a_k, cover,
                                        0.2 + 0.5 * st))
        p.setDither(True)
        c.drawRect(skia.Rect.MakeLTRB(x0, y0, x1, y1), p)


# ============================================================================
# DAWN & SUN
# ============================================================================
def dawn_glow(c, horizon_y=700, amount=0.0, sun_x=960, width=1.0):
    """Warm pre-dawn bloom sitting on the horizon (use with sky(dawn>0)).

    amount: 0..1 strength.   sun_x: where the sun will rise (glow centre).
    width:  horizontal spread multiplier.
    """
    if amount <= 0.002:
        return
    a = clamp(amount)
    wide = 1500 * width
    # wide, flat peach/rose glow
    p = radial((0, 0), 1.0, [(0, '#ffb98a', 0.30 * a), (0.25, '#ff9f86', 0.16 * a), (0.55, '#c9789c', 0.06 * a),
                             (1, '#8a5a9c', 0.0)])
    p.setBlendMode(ADD)
    p.setDither(True)
    c.save()
    c.translate(sun_x, horizon_y)
    c.scale(wide, wide * 0.30)
    c.drawCircle(0, 0, 1.0, p)
    c.restore()
    # hot core right where the sun will appear
    q = radial((0, 0), 1.0, [(0, '#fff0c8', 0.30 * a), (0.3, '#ffcf98', 0.13 * a), (1, '#ffa070', 0.0)])
    q.setBlendMode(ADD)
    q.setDither(True)
    c.save()
    c.translate(sun_x, horizon_y)
    c.scale(420 * width, 150)
    c.drawCircle(0, 0, 1.0, q)
    c.restore()


def sun(c, x, y, r=90, amount=1.0, t=0.0, rays=1.0, flare=1.0):
    """Rising sun: bright disc with limb glow, big warm halo, soft slowly
    turning god rays and a faint horizontal flare streak.

    (x, y): disc centre (put it near horizon_y; the sea drawn afterwards
    hides the lower part).  r: disc radius.  amount: 0..1 fade.
    t: time (rays turn / shimmer).  rays: god-ray strength.  flare: streak.
    """
    if amount <= 0.002:
        return
    a = clamp(amount)
    soft_glow(c, x, y, r * 12, '#ffb870', 0.09 * a)
    soft_glow(c, x, y, r * 5, '#ffcf80', 0.22 * a)
    soft_glow(c, x, y, r * 2.2, '#ffe6b0', 0.38 * a)
    if rays > 0:
        # soft god rays: long thin radial-gradient ellipses through the sun
        sp = _glow_paint(hexrgb('#ffe9c0'), 'spike')
        n = 9
        for i in range(n):
            ang = math.pi * i / n + 0.21 * math.sin(i * 12.9898) + t * 0.015
            ln = r * (9 + 6 * (0.5 + 0.5 * math.sin(i * 3.7 + 1.3)))
            wd = r * (0.35 + 0.35 * (0.5 + 0.5 * math.sin(i * 7.31 + t * 0.35)))
            al = a * rays * (0.10 + 0.07 * math.sin(i * 5.1 + t * 0.5))
            _blob(c, x, y, ln, wd, sp, al, ang)
    if flare > 0:
        fp = radial((0, 0), 1.0, [(0, '#ffe8c0', 0.35 * a * flare), (0.3, '#ffd090', 0.12 * a * flare),
                                  (1, '#ffb070', 0)])
        fp.setBlendMode(ADD)
        c.save()
        c.translate(x, y)
        c.scale(r * 11, r * 0.45)
        c.drawCircle(0, 0, 1.0, fp)
        c.restore()
    dp = radial((x - r * 0.15, y - r * 0.15), r * 1.1,
                [(0, '#fffef6', 1), (0.7, '#fff6d8', 1), (1, '#ffe2a0', 1)])
    dp.setAlphaf(a)
    c.drawCircle(x, y, r, dp)
    soft_glow(c, x, y, r * 1.35, '#fff4d8', 0.45 * a)


def shooting_star_sky_streak(c, x, y, angle, length, alpha, width=2.5, color='#ffffff'):
    """A thin background meteor streak (decoration): bright head at (x, y),
    tapered tail of `length` pointing away along `angle` + 180 degrees."""
    if alpha <= 0.003:
        return
    with at(c, x, y, rot=angle):
        path = skia.Path()
        path.moveTo(0, -width * 0.5)
        path.lineTo(-length, -0.2)
        path.lineTo(-length, 0.2)
        path.lineTo(0, width * 0.5)
        path.close()
        p = linear((0, 0), (-length, 0), [(0, color, alpha), (0.25, '#cfe0ff', alpha * 0.5), (1, '#9ab8ff', 0)])
        p.setBlendMode(ADD)
        c.drawPath(path, p)
        _blob(c, 0, 0, width * 4, width * 4, _glow_paint(hexrgb('#dfe8ff'), 'glow'), alpha)


# ============================================================================
# STAR TRAILS (s02 time-lapse)
# ============================================================================
_TRAILS = OrderedDict()
_TR_N = 4096  # angular resolution


def _trail_tex(center, rmax, seed, density):
    """Polar prefix-sum texture of the star field around `center`: for each
    radius row, the cumulative brightness of the stars along the angle. A
    trail of any arc length is then S(b) - S(a): two texture reads."""
    key = (round(center[0], 1), round(center[1], 1), int(rmax), seed, round(density, 3))
    d = _TRAILS.get(key)
    if d is not None:
        _TRAILS.move_to_end(key)
        return d
    cx, cy = center
    st = _gather_stars(seed, (cx - rmax, cy - rmax, cx + rmax, cy + rmax), density)
    rows = int(rmax) + 4
    grid = np.zeros((rows, _TR_N + 1), np.float32)
    if st is not None and len(st['x']):
        dx, dy = st['x'] - cx, st['y'] - cy
        rr = np.sqrt(dx * dx + dy * dy)
        m = (rr < rmax) & (rr > 2)
        rr, dx, dy = rr[m], dx[m], dy[m]
        br = (st['alpha'][m] ** 1.4 * np.clip(st['size'][m] / 1.25, 0.6, 1.6)
              * np.where(st['cls'][m] == 0, 0.45, 1.0)).astype(np.float32)
        th = np.mod(np.arctan2(dy, dx), TAU)
        colf = th / TAU * _TR_N
        c0 = np.floor(colf).astype(int)
        fc = (colf - c0).astype(np.float32)
        sig = np.clip(st['size'][m] * 0.42, 0.55, 1.4)
        for off in (-2, -1, 0, 1, 2):
            row = np.floor(rr).astype(int) + off
            wgt = np.exp(-0.5 * ((row + 0.5 - rr) / sig) ** 2).astype(np.float32) * br
            ok = (row >= 0) & (row < rows)
            np.add.at(grid, (row[ok], c0[ok] % _TR_N), wgt[ok] * (1 - fc[ok]))
            np.add.at(grid, (row[ok], (c0[ok] + 1) % _TR_N), wgt[ok] * fc[ok])
    S = np.zeros_like(grid)
    S[:, 1:] = np.cumsum(grid[:, :-1], axis=1)
    # pack 3 radius rows per texel row (RGB), alpha = 1
    nr = (rows + 2) // 3
    pad = np.zeros((nr * 3, _TR_N + 1), np.float32)
    pad[:rows] = S
    tex = np.ones((nr, _TR_N + 1, 4), np.float32)
    tex[..., 0] = pad[0::3]
    tex[..., 1] = pad[1::3]
    tex[..., 2] = pad[2::3]
    img = skia.Image.fromarray(np.ascontiguousarray(tex), colorType=skia.kRGBA_F32_ColorType,
                               alphaType=skia.kPremul_AlphaType)
    d = dict(img=img, rows=rows)
    _TRAILS[key] = d
    while len(_TRAILS) > 3:
        _TRAILS.popitem(last=False)
    return d


_TRAIL_SKSL = """
uniform shader tex;
uniform float2 ctr;
uniform float spin;
uniform float arc;
uniform float gain;
uniform float rows;
uniform float ncol;
uniform float hy;
uniform float hfade;
uniform float rref;
uniform float3 cwarm;
uniform float3 ccool;

float S(float row, float x) {
    float j = floor(row / 3.0);
    float ch = row - j * 3.0;
    float4 s = tex.eval(float2(x, j + 0.5));
    return ch < 0.5 ? s.r : (ch < 1.5 ? s.g : s.b);
}

float rowval(float row, float xb, float xa, float wrap) {
    if (row < 0.0 || row >= rows) { return 0.0; }
    float tot = S(row, ncol + 0.5);
    return S(row, xb) - S(row, xa) + wrap * tot;
}

half4 main(float2 p) {
    float2 d = p - ctr;
    float r = length(d);
    float th = atan(d.y, d.x) + spin;
    th = th - 6.2831853 * floor(th / 6.2831853);
    float a = th - arc;
    float wrap = a < 0.0 ? 1.0 : 0.0;
    a = a + wrap * 6.2831853;
    float k = ncol / 6.2831853;
    float xb = th * k + 0.5;
    float xa = a * k + 0.5;
    float rf = r - 0.5;
    float r0 = floor(rf);
    float fr = rf - r0;
    float v = mix(rowval(r0, xb, xa, wrap), rowval(r0 + 1.0, xb, xa, wrap), fr);
    float h = fract(sin(r0 * 12.9898) * 43758.5453);
    float3 cc = mix(ccool, cwarm, h);
    float hf = clamp((hy - p.y) / hfade, 0.0, 1.0);
    // outer stars sweep faster -> dimmer trails (keeps the average even)
    float rfo = pow(rref / max(r, rref * 0.75), 0.8);
    v = v * gain * rfo * hf * hf * (3.0 - 2.0 * hf);
    v = clamp(v, 0.0, 1.0);
    return half4(cc * v, v);
}
"""


def star_trails(c, t, center=(960, 200), amount=1.0, spin=0.0, rect=DEFAULT_RECT, arc=None, seed=7,
                density=1.0, bright=1.0, horizon_y=None, fade=320.0, gain=0.6, pole=1.0):
    """Time-lapse star trails around the pole star (s02 "300 years pass").

    center: world position of the pole star (rotation centre).
    amount: 0..1 trail visibility (0 = exactly the normal stars() look,
            rotated by `spin`).
    spin:   accumulated sky rotation in radians (positive = counter-clockwise
            on screen, like the northern sky). Stars are drawn rotated by it;
            end a time-lapse on a multiple of 2*pi (or keep calling this with
            amount=0 and the final spin) to return seamlessly to stars().
    arc:    trail arc length (radians); default min(|spin|, 2*pi) so trails
            grow with the rotation into full circles.
    rect, seed, density, bright, horizon_y, fade: as in stars().
    gain:   trail brightness.  pole: brightness of the pole star highlight.
    The trails are rendered by an SkSL shader from a polar prefix-sum
    texture, so thousands of full-circle trails cost the same as short ones.
    """
    if bright <= 0.002:
        return
    cx, cy = center
    amt = clamp(amount)
    # 1) trails
    if amt > 0.002:
        box = _visible(c, rect)
        if box is not None:
            x0, y0, x1, y1 = box
            far = max(math.hypot(x0 - cx, y0 - cy), math.hypot(x1 - cx, y0 - cy),
                      math.hypot(x0 - cx, y1 - cy), math.hypot(x1 - cx, y1 - cy))
            rmax = min(2600, int(math.ceil((far + 20) / 256.0)) * 256)
            tr = _trail_tex(center, rmax, seed, density)
            a_arc = abs(spin) if arc is None else abs(arc)
            a_arc = min(a_arc, TAU * 0.9999)
            if a_arc > 1e-4:
                sgn = 1.0 if spin >= 0 else -1.0
                tsh = tr['img'].makeShader(skia.TileMode.kClamp, skia.TileMode.kClamp, _SAMPLING)
                sh = _shader('trails', _TRAIL_SKSL, dict(
                    ctr=(cx, cy), spin=sgn * spin if sgn > 0 else -spin, arc=a_arc,
                    gain=gain * amt * bright, rows=float(tr['rows']), ncol=float(_TR_N),
                    hy=float(horizon_y if horizon_y is not None else 1e9), hfade=max(1.0, fade), rref=380.0,
                    cwarm=hexrgb('#ffe2b8'), ccool=hexrgb('#d8e4ff')), [('tex', tsh)])
                p = skia.Paint(AntiAlias=False, Shader=sh)  # (setShader is very slow in skia-python)
                p.setBlendMode(ADD)
                if sgn < 0:
                    # clockwise spin: mirror the angle direction
                    c.save()
                    c.translate(cx, cy)
                    c.scale(1, -1)
                    c.translate(-cx, -cy)
                    c.drawRect(skia.Rect.MakeLTRB(x0, 2 * cy - y1, x1, 2 * cy - y0), p)
                    c.restore()
                else:
                    c.drawRect(skia.Rect.MakeLTRB(x0, y0, x1, y1), p)
    # 2) the stars themselves, rotated by spin (heads dimmed while trailing)
    c.save()
    c.translate(cx, cy)
    c.rotate(-math.degrees(spin))
    c.translate(-cx, -cy)
    if horizon_y is None:
        stars(c, t, rect=(-1e6, -1e6, 2e6, 2e6), density=density, bright=bright * (1 - 0.45 * amt), seed=seed,
              twinkle=1.0 - 0.8 * amt, glints=1.0 - 0.6 * amt)
    else:
        # horizon fade must stay in screen space: approximate by fading the heads with the trails
        stars(c, t, rect=(-1e6, -1e6, 2e6, 2e6), density=density, bright=bright * (1 - 0.45 * amt), seed=seed,
              twinkle=1.0 - 0.8 * amt, glints=1.0 - 0.6 * amt, horizon_y=None)
    c.restore()
    # 3) the pole star
    if pole > 0:
        hero_star(c, cx, cy, size=0.75, t=t, warmth=0.0, intensity=pole * bright)


# ============================================================================
# HERO STAR (the star that answers Deng)
# ============================================================================
def hero_flare(t_since, dur=0.9):
    """Convenience envelope for one 'blink': 0 -> 1 (fast) -> 0 over `dur` s."""
    if t_since < 0 or t_since > dur:
        return 0.0
    u = t_since / dur
    return (1 - math.exp(-u * 18)) * (1 - u) ** 1.6


def hero_star(c, x, y, size=1.0, flare=0.0, t=0.0, warmth=1.0, intensity=1.0, flare_age=None, seed=5):
    """One special star (Guang, answering Deng from the sky in the epilogue).

    size:      overall scale (1 = a bit bigger than the brightest stars).
    flare:     0..1 sparkle burst: bright bloom, long 4-point cross glints,
               diagonal glints, a soft ring and little sparkles flying out.
               Drive it with hero_flare(t - blink_time) for "blink, blink".
    t:         time (gentle idle twinkle, sparkle motion).
    warmth:    0 = white-blue star, 1 = warm gold (Guang's colour).
    intensity: brightness multiplier.
    flare_age: optional seconds since the flare started; if given, the ring
               expands with it (otherwise the ring grows with `flare`).
    """
    if intensity <= 0.002:
        return
    f = clamp(flare)
    warm = mix('#dfeaff', '#ffe29a', warmth)
    core_c = mix('#ffffff', '#fffbea', warmth)
    tw = 0.9 + 0.1 * math.sin(t * 2.3 + 1.1) * (0.7 + 0.3 * math.sin(t * 5.1))
    k = intensity * tw
    s = size
    # soft glow and bloom
    _blob(c, x, y, 70 * s * (1 + 0.8 * f), 70 * s * (1 + 0.8 * f), _glow_paint(mix(warm, '#8fb0ff', 0.25), 'glow'),
          0.16 * k + 0.2 * f)
    _blob(c, x, y, 30 * s * (1 + 2.2 * f), 30 * s * (1 + 2.2 * f), _glow_paint(warm, 'glow'), 0.6 * k + 0.4 * f)
    if f > 0:
        _blob(c, x, y, 90 * s * (0.6 + f), 90 * s * (0.6 + f), _glow_paint(mix(warm, '#ffd070', 0.3), 'glow'), 0.35 * f)
    # glints
    gl = 44 * s * (1 + 0.12 * math.sin(t * 3.1)) * (1 + 3.5 * f)
    th = 2.0 * s * (1 + 0.6 * f)
    sp = _glow_paint(core_c, 'spike')
    _blob(c, x, y, gl, th, sp, 0.9 * k + 0.1 * f)
    _blob(c, x, y, th, gl, sp, 0.9 * k + 0.1 * f)
    dg = 0.3 + 0.7 * f
    _blob(c, x, y, gl * 0.45 * dg, th * 0.8, sp, (0.35 * k + 0.4 * f), math.pi / 4)
    _blob(c, x, y, th * 0.8, gl * 0.45 * dg, sp, (0.35 * k + 0.4 * f), math.pi / 4)
    # core
    _blob(c, x, y, 3.6 * s * (1 + 0.6 * f), 3.6 * s * (1 + 0.6 * f), _glow_paint(core_c, 'core'), 1.0)
    # ring
    if f > 0.01:
        if flare_age is not None:
            rr = s * (26 + 150 * (1 - math.exp(-flare_age * 3.5)))
        else:
            rr = s * (30 + 90 * f)
        _blob(c, x, y, rr, rr, _glow_paint(mix(warm, '#ffffff', 0.3), 'ring'), 0.2 * f)
        # sparkles flying out
        r_ = nprng(seed)
        n = 10
        base = (flare_age if flare_age is not None else f * 0.6)
        for i in range(n):
            ang = TAU * i / n + r_.uniform(-0.25, 0.25)
            dist = s * (20 + r_.uniform(60, 150) * (1 - math.exp(-base * 4.0)) * (0.6 + 0.6 * f))
            px, py = x + math.cos(ang) * dist, y + math.sin(ang) * dist
            sa = f * (0.5 + 0.5 * math.sin(t * 9 + i * 2.1))
            sz = s * r_.uniform(9, 17)
            _blob(c, px, py, sz, sz * 0.14, sp, sa)
            _blob(c, px, py, sz * 0.14, sz, sp, sa)
            _blob(c, px, py, sz * 0.6, sz * 0.6, _glow_paint(warm, 'glow'), sa * 0.5)


# ============================================================================
# SKY SWIRL (s06 whale_appear)
# ============================================================================
_SWIRL_SKSL = """
uniform shader band;
uniform shader field;
uniform float2 ctr;
uniform float R;
uniform float amt;
uniform float t;
uniform float2 bsz;

float3 arm(float th, float lr, float r, float ph, float wind, float spin) {
    float dth = th - (wind * lr + spin + ph);
    dth = dth - 6.2831853 * floor((dth + 3.14159265) / 6.2831853);
    float v = dth / 1.35;
    float s = 0.12 + 0.7 * clamp(1.0 - r / 1.25, 0.0, 1.0) + 0.1 * ph;
    return band.eval(float2(s * bsz.x, (v * 0.5 + 0.5) * bsz.y)).rgb;
}

half4 main(float2 p) {
    float2 d = p - ctr;
    float r = length(d) / R;
    if (r > 1.4) { return half4(0.0); }
    float th = atan(d.y, d.x);
    float lr = log(r + 0.03);
    float wind = 1.2 + 2.6 * amt;
    float spin = -t * 0.22 * (0.3 + 0.7 * amt);
    float3 neb = arm(th, lr, r, 0.0, wind, spin) + arm(th, lr, r, 3.14159265, wind, spin);
    // extra swirling nebulosity from an fbm field, rotated with the vortex
    float rot = th - wind * lr * 0.9 - spin * 1.4;
    float2 q = float2(cos(rot), sin(rot)) * r * 260.0;
    float n1 = field.eval(q + float2(311.0, 97.0)).r;
    float n2 = field.eval(q * 2.1 + float2(37.0, 401.0)).g;
    float mist = smoothstep(0.35, 0.8, n1 * 0.7 + n2 * 0.3);
    float fall = smoothstep(1.4, 0.35, r);
    float3 mistc = mix(float3(0.42, 0.36, 0.95), float3(0.35, 0.72, 1.0), smoothstep(0.1, 1.0, r));
    mistc = mix(mistc, float3(1.0, 0.72, 0.78), smoothstep(0.7, 0.95, n2) * 0.5);
    float3 col = neb * (0.8 + 0.45 * amt) * fall + mistc * mist * 0.13 * fall;
    // the eye: where the whale emerges
    float core = exp(-r * r / 0.0025) * 0.85 + exp(-r * r / 0.03) * 0.2;
    float ring = exp(-pow((r - 0.2 - 0.015 * sin(t * 1.3)) / 0.03, 2.0)) * 0.12;
    col += float3(0.82, 0.9, 1.0) * core + float3(0.62, 0.8, 1.0) * ring * fall;
    col *= amt;
    col = 1.0 - exp(-col * 1.1);
    float a = clamp(max(col.r, max(col.g, col.b)), 0.0, 1.0);
    return half4(col, a);
}
"""


def sky_swirl(c, t, x, y, radius, amount, seed=21, stars_n=700, streaks=1.0):
    """The Milky Way / stars swirling into a vortex high in the sky, from
    which the constellation whale emerges (s06 `whale_appear`).

    (x, y):  vortex centre (world units).  radius: its size (arms reach
             ~1.3*radius).
    amount:  0..1 — fades in AND winds up the spiral (0 = nothing drawn).
    t:       time (the vortex turns; stars spiral inwards).
    stars_n: number of spiralling stars.  streaks: motion-streak strength.
    Drawn additively (SkSL nebula + batched star points + streaks).
    """
    a = clamp(amount)
    if a <= 0.003:
        return
    cf = _cloud_field()
    mw = _mw_data()
    fsh = cf['img'].makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat, _SAMPLING)
    bsh = mw['img'].makeShader(skia.TileMode.kClamp, skia.TileMode.kClamp, _SAMPLING)
    sh = _shader('swirl', _SWIRL_SKSL, dict(ctr=(x, y), R=float(radius), amt=a, t=float(t),
                                            bsz=(_MW_TW, _MW_TH)), [('band', bsh), ('field', fsh)])
    p = skia.Paint(AntiAlias=False, Shader=sh)  # (setShader is very slow in skia-python)
    p.setBlendMode(ADD)
    rr = radius * 1.42
    b = c.getLocalClipBounds()
    box = skia.Rect.MakeLTRB(max(x - rr, b.left()), max(y - rr, b.top()), min(x + rr, b.right()),
                             min(y + rr, b.bottom()))
    if box.isEmpty():
        return
    c.drawRect(box, p)
    # spiralling stars: they ride the arms inwards (log-spiral paths)
    r_ = nprng(seed)
    n = int(stars_n)
    r0 = r_.random(n) ** 0.8 * 1.25
    onarm = r_.random(n) < 0.7
    ph = np.where(onarm, r_.integers(0, 2, n) * math.pi + r_.normal(0, 0.28, n), r_.uniform(0, TAU, n))
    wind = 1.2 + 2.6 * a
    spd = 0.3 + 0.7 * a
    inflow = 0.045 * spd

    def pos(tt):
        rr = np.mod(r0 - tt * inflow, 1.25) + 0.04
        th = wind * np.log(rr + 0.03) - tt * 0.22 * spd + ph
        return x + np.cos(th) * rr * radius, y + np.sin(th) * rr * radius, rr

    px, py, rr_ = pos(t)
    fade = np.clip((1.29 - rr_) / 0.3, 0, 1) * np.clip((rr_ - 0.05) / 0.12, 0, 1)
    br = r_.random(n)
    alpha = (0.2 + 0.8 * br ** 2) * fade * a
    size = 0.9 + 1.9 * br ** 3
    colc = r_.choice(4, n, p=[0.5, 0.3, 0.15, 0.05]).astype(np.int8)
    dev, kz = _zoom_factor(c, 0.5)
    _draw_points(c, px, py, size * kz * dev, alpha, colc, dev)
    if streaks > 0:
        sel = np.flatnonzero(br > 0.62)
        segs, dt = 5, 0.28 * streaks
        sp = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style)
        sp.setStrokeCap(skia.Paint.kRound_Cap)
        sp.setBlendMode(ADD)
        prev = (px[sel], py[sel], rr_[sel])
        for k in range(segs):
            cur = pos(t - dt * (k + 1))
            cx_, cy_, cr_ = cur[0][sel], cur[1][sel], cur[2][sel]
            ok = cr_ >= prev[2] - 1e-6   # skip wrap-around jumps
            al = alpha[sel] * 0.42 * (1 - (k + 0.5) / segs) ** 1.5
            sp.setStrokeWidth(max(1.1 / dev, 1.5 * (1 - 0.6 * k / segs)))
            abin = np.clip(np.round(al * 12), 0, 12).astype(int)
            for lv in range(1, 13):
                m = ok & (abin == lv)
                if not m.any():
                    continue
                sp.setColor4f(_c4(hexrgb('#dfe9ff'), lv / 12.0))
                seg = []
                for q in zip(prev[0][m].tolist(), prev[1][m].tolist(), cx_[m].tolist(), cy_[m].tolist()):
                    seg.append(skia.Point(q[0], q[1]))
                    seg.append(skia.Point(q[2], q[3]))
                c.drawPoints(skia.Canvas.kLines_PointMode, seg, sp)
            prev = (cx_, cy_, cr_)


# ============================================================================
# CLOUD RING (s06 ignition)
# ============================================================================
_RING_SKSL = """
uniform shader field;
uniform float2 ctr;
uniform float squash;
uniform float hole;
uniform float R;
uniform float fsc;
uniform float2 foff;
uniform float2 foff2;
uniform float hh;
uniform float thr;
uniform float soft;
uniform float3 lit;
uniform float3 shadow;
uniform float3 glow;
uniform float alpha;
uniform float glowamt;

float2 push(float2 q, out float comp) {
    // the material that was inside the hole is squeezed into a rim of width w
    float r = length(q);
    float w = 0.45 * hole + 30.0;
    float u = clamp((r - hole) / w, 0.0, 1.0);
    float r0 = r >= hole + w ? r : mix(0.3 * (hole + w), hole + w, u);
    comp = hole > 1.0 ? (1.0 - u) * (1.0 - u) : 0.0;
    return r > 0.001 ? q * (r0 / r) : q;
}

float dens(float2 p) {
    float2 q = float2(p.x - ctr.x, (p.y - ctr.y) / squash);
    float comp;
    float2 q0 = push(q, comp);
    float2 f = q0 * fsc;
    float n = field.eval(f + foff).r * 0.9 + field.eval(f * 1.7 + foff2).g * 0.1;
    float e = (p.y - ctr.y) / hh;
    float env = clamp(1.0 - e * e, 0.0, 1.0);
    return n + 0.5 * env - 0.5 + 0.1 * comp;
}

half4 main(float2 p) {
    float2 q = float2(p.x - ctr.x, (p.y - ctr.y) / squash);
    float r = length(q);
    float d = dens(p);
    float ang = atan(q.y, q.x);
    float wob = 1.0 + 0.07 * (field.eval(float2(ang * 60.0 + 300.0, 40.0)).g - 0.5) * 2.0;
    float a = smoothstep(thr - soft, thr + soft, d) * smoothstep(hole * wob * 0.99, hole * wob * 1.05 + 4.0, r);
    if (a < 0.003) { return half4(0.0); }
    float du = dens(p - float2(0.0, 5.0));
    float top = clamp((d - du) * 12.0 + 0.3, 0.0, 1.0);
    float e = clamp((p.y - ctr.y) / hh, -1.0, 1.0);
    float3 cc = mix(shadow, lit, top * (0.75 - 0.25 * e));
    float w = 0.45 * hole + 30.0;
    float rimz = hole > 1.0 ? exp(-max(r - hole, 0.0) / (0.5 * w + 1.0)) : 0.0;
    float far = 1.0 / (1.0 + pow(max(r - hole, 0.0) / (0.22 * R + 1.0), 2.0));
    cc += glow * glowamt * (far * 0.35 + rimz * 0.9);
    a *= alpha;
    return half4(cc * a, a);
}
"""


def cloud_ring(c, t, x, y, radius, open_amount, rect_x=(-2000, 3920), thickness=150.0, squash=0.22,
               alpha=0.6, color='#34467f', glow_color='#ffcf7a', glow=1.0, cover=0.55, seed=4,
               speed=4.0, scale=1.0, dawn=0.0, sunrise=0.0):
    """A cloud deck at altitude, blown open in a ring around the vertical
    light pillar (s06 ignition). With open_amount=0 it is simply a full cloud
    band (so it can be drawn before the ignition too) that matches clouds().

    (x, y):      where the pillar pierces the deck (hole centre, world units).
    radius:      final hole radius (horizontal, world units).
    open_amount: 0..1 — the hole opens outward, the clouds are pushed away
                 (area-preserving radial push) and bunch up into a rim that
                 is lit gold from inside. Use an ease-out curve over ~1-2 s.
    rect_x:      horizontal extent of the deck.
    thickness:   vertical half-extent of the deck on screen (world units).
    squash:      vertical foreshortening of the deck plane (0.22 = seen from
                 far below at a low angle).
    alpha, color, cover, speed, scale, dawn, sunrise: as in clouds().
    glow_color, glow: the pillar's light on the cloud walls (scale `glow`
                 with the pillar's intensity; 0 = unlit).
    """
    if alpha <= 0.003:
        return
    b = c.getLocalClipBounds()
    x0, x1 = max(rect_x[0], b.left()), min(rect_x[1], b.right())
    hh = thickness * scale
    y0, y1 = max(y - hh, b.top()), min(y + hh, b.bottom())
    if x1 <= x0 or y1 <= y0:
        return
    lit, shadow, _und = _cloud_colors(color, dawn, sunrise, None, None)
    oa = clamp(open_amount)
    cf = _cloud_field()
    tw, th = cf['size']
    rng_ = nprng(seed * 31 + 5)
    ox, oy = rng_.uniform(0, tw), rng_.uniform(0, th)
    ox2, oy2 = rng_.uniform(0, tw), rng_.uniform(0, th)
    fsc = 1.0 / (12.0 * scale)
    fsh = cf['img'].makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat, _SAMPLING)
    thr = 0.5 + (0.5 - clamp(cover)) * 0.5
    sh = _shader('ring', _RING_SKSL, dict(
        ctr=(x, y), squash=squash, hole=radius * oa, R=float(radius), fsc=fsc,
        foff=(ox - t * speed * fsc, oy), foff2=(ox2 - t * speed * fsc * 2.3 * 1.2, oy2), hh=hh, thr=thr,
        soft=0.08, lit=lit, shadow=shadow, glow=_rgb(glow_color), alpha=clamp(alpha),
        glowamt=glow * smoothstep(0.0, 0.06, oa)), [('field', fsh)])
    p = skia.Paint(AntiAlias=False, Shader=sh)  # (setShader is very slow in skia-python)
    p.setDither(True)
    c.drawRect(skia.Rect.MakeLTRB(x0, y0, x1, y1), p)


# ============================================================================
# CONVENIENCE: the whole backdrop in the right order
# ============================================================================
def backdrop(c, t, rect=DEFAULT_RECT, horizon_y=700, dawn=0.0, sunrise=0.0, milky=1.0, star_bright=None,
             mw=None, sun_x=960, sun_y=None, sun_r=90, moon_at=None, cloud_y=None, cloud_alpha=0.35,
             cloud_speed=6.0, seed=7):
    """Draw sky + dawn glow + Milky Way + stars (+ optional moon, sun and a
    cloud band) in the right order.

    milky:       Milky Way brightness (auto-faded by dawn/sunrise).
    star_bright: star brightness (default: fades out with dawn and sunrise).
    mw:          dict of milky_way() kwargs (center, angle, length, width, bend).
    sun_y:       if given, draws the sun at (sun_x, sun_y) with radius sun_r.
    moon_at:     (x, y, r[, phase]) to draw a moon.
    cloud_y:     if given, draws clouds() centred there.
    """
    sky(c, rect, horizon_y, dawn, sunrise)
    if dawn > 0 or sunrise > 0:
        dawn_glow(c, horizon_y, max(dawn * 0.8, sunrise), sun_x)
    night = (1 - 0.75 * clamp(dawn)) * (1 - clamp(sunrise))
    if milky * night > 0.01:
        kw = dict(mw or {})
        kw.setdefault('horizon_y', horizon_y)
        milky_way(c, t, alpha=milky * night * (1 - 0.5 * clamp(dawn)), **kw)
    sb = night if star_bright is None else star_bright
    stars(c, t, rect, bright=sb, horizon_y=horizon_y, seed=seed)
    if moon_at is not None:
        mx, my, mr = moon_at[:3]
        moon(c, mx, my, mr, *(moon_at[3:4] or ()), glow=night)
    if sun_y is not None:
        sun(c, sun_x, sun_y, sun_r, amount=max(sunrise, 0.0) if sunrise > 0 else 1.0, t=t)
    if cloud_y is not None:
        clouds(c, t, y=cloud_y, alpha=cloud_alpha, speed=cloud_speed, dawn=dawn, sunrise=sunrise)
