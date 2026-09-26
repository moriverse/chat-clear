"""SEA & SET library — owner: ENV-SEA agent.

Ocean, island, lighthouse (exterior + interior), boat and small props.
Draw in world units on an already-transformed canvas.

CONTRACT: keep every signature below working (you may add keyword args with
defaults and new functions). STUB=True means placeholder art.
"""
from __future__ import annotations

import math
import struct
from contextlib import contextmanager

import numpy as np
import skia

from engine.core import (ADD, H, W, at, clamp, col, fill, hexrgb, lerp, linear, mix, noise1, nprng, poly, radial,
                         rrect, saved, smooth_path, smoothstep, soft_glow, stroke)

STUB = False

SEA_DEEP = '#081631'
SEA_MID = '#123060'
SEA_HI = '#5c8fd6'
FOAM = '#bfd8ff'

WARM = '#ffd36b'
WARM2 = '#ffb347'
COOL = '#9fc4ff'

# ============================================================================
# infrastructure: runtime shaders + precomputed textures (module level cache)
# ============================================================================


def _rgb(c):
    return hexrgb(c) if isinstance(c, str) else tuple(c[:3])


class _Shader:
    """Tiny wrapper around an SkSL runtime shader with named float uniforms."""

    def __init__(self, src):
        self.eff = skia.RuntimeEffect.MakeForShader(src)
        self.layout = []
        off = 0
        sizes = {0: 1, 1: 2, 2: 3, 3: 4}
        for u in self.eff.uniforms():
            n = sizes[int(u.type)]
            self.layout.append((u.name, off, n))
            off += n
        self.nfloats = off

    def make(self, children=(), matrix=None, **vals):
        buf = [0.0] * self.nfloats
        for name, off, n in self.layout:
            v = vals.get(name, 0.0)
            if n == 1:
                buf[off] = float(v)
            else:
                if isinstance(v, str):
                    v = hexrgb(v)
                for i in range(n):
                    buf[off + i] = float(v[i])
        data = skia.Data.MakeWithCopy(struct.pack(f'{self.nfloats}f', *buf))
        if children:
            ch = skia.VectorRuntimeEffectChildPtr([skia.RuntimeEffectChildPtr(s) for s in children])
            if matrix is not None:
                return self.eff.makeShader(data, skia.SpanRuntimeEffectChildPtr(ch), matrix)
            return self.eff.makeShader(data, skia.SpanRuntimeEffectChildPtr(ch))
        if matrix is not None:
            return self.eff.makeShader(data, skia.SpanRuntimeEffectChildPtr(skia.VectorRuntimeEffectChildPtr([])), matrix)
        return self.eff.makeShader(data)


def _px(c):
    """World units per device pixel for the canvas' current transform."""
    m = c.getTotalMatrix()
    sx = math.hypot(m.getScaleX(), m.getSkewY())
    return 1.0 / max(sx, 1e-6)


def _device_scale(c):
    m = c.getTotalMatrix()
    return math.hypot(m.getScaleX(), m.getSkewY())


def _draw_lowres(c, rect, factor, fn, blend=None, alpha=1.0):
    """Render fn(c2) (design coordinates) into an offscreen surface at `factor` x the current
    device resolution, then draw it back into `rect` (x0, y0, x1, y1) with linear filtering.
    Used for smooth, expensive layers (light maps, the view through the glass)."""
    ds = _device_scale(c) * factor
    x0, y0, x1, y1 = rect
    w = max(2, int(math.ceil((x1 - x0) * ds)))
    h = max(2, int(math.ceil((y1 - y0) * ds)))
    surf = skia.Surface(w, h)
    c2 = surf.getCanvas()
    c2.clear(skia.ColorTRANSPARENT)
    c2.scale(w / (x1 - x0), h / (y1 - y0))
    c2.translate(-x0, -y0)
    fn(c2)
    img = surf.makeImageSnapshot()
    p = skia.Paint(AntiAlias=True)
    if blend is not None:
        p.setBlendMode(blend)
    if alpha < 1.0:
        p.setAlphaf(clamp(alpha))
    c.drawImageRect(img, skia.Rect.MakeLTRB(x0, y0, x1, y1), skia.SamplingOptions(skia.FilterMode.kLinear), p)


def _fft_noise(n, seed, kpeak, bw, aniso=1.0):
    """Tileable band-limited noise (n x n) normalised to zero mean / unit std.
    Returns (h, dh/dy) as float32 arrays."""
    r = nprng(seed)
    F = np.fft.fft2(r.normal(size=(n, n)))
    ky = np.fft.fftfreq(n)[:, None] * n
    kx = np.fft.fftfreq(n)[None, :] * n
    k = np.sqrt((kx * aniso) ** 2 + ky ** 2)
    filt = np.exp(-((k - kpeak) / bw) ** 2) * (k > 0.5)
    F = F * filt
    h = np.real(np.fft.ifft2(F))
    dy = np.real(np.fft.ifft2(F * (2j * np.pi * ky / n)))
    s = h.std() + 1e-9
    return (h / s).astype(np.float32), (dy / (dy.std() + 1e-9)).astype(np.float32)


def _to_u8(a, lo=-2.5, hi=2.5):
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _image(rgba):
    rgba = np.ascontiguousarray(rgba)
    return skia.Image.fromarray(rgba, colorType=skia.kRGBA_8888_ColorType)


_TEXN = 512
_TEX = {}


def _tex():
    """Precomputed tileable wave textures (built once per process)."""
    if not _TEX:
        n = _TEXN
        h1, s1 = _fft_noise(n, 11, 18, 7, aniso=1.7)
        hb, _ = _fft_noise(n, 12, 70, 25)
        # crest-sharpened slope: waves with sharper crests read as painted strokes
        r = _to_u8(h1)
        g = _to_u8(s1)
        b = _to_u8(hb)
        wave = np.dstack([r, g, b, np.full_like(r, 255)])
        img = _image(wave)
        _TEX['wave_img'] = img
        _TEX['wave'] = img.makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat,
                                      skia.SamplingOptions(skia.FilterMode.kLinear))
        # soft blotch noise for rocks / wood / clouds of dust (larger features)
        h2, _ = _fft_noise(256, 21, 6, 4)
        h3, _ = _fft_noise(256, 22, 24, 10)
        h4, _ = _fft_noise(256, 23, 60, 20)
        m = np.dstack([_to_u8(h2), _to_u8(h3), _to_u8(h4), np.full((256, 256), 255, np.uint8)])
        img2 = _image(m)
        _TEX['blot_img'] = img2
        _TEX['blot'] = img2.makeShader(skia.TileMode.kRepeat, skia.TileMode.kRepeat,
                                       skia.SamplingOptions(skia.FilterMode.kLinear))
    return _TEX


# ============================================================================
# OCEAN
# ============================================================================
_OCEAN_SRC = """
uniform shader wave;
uniform float t;
uniform float hy;
uniform float cx;
uniform float pxw;
uniform float calm;
uniform float fz;
uniform float3 cHor;
uniform float3 cMid;
uniform float3 cDeep;
uniform float3 cHi;
uniform float3 cHaze;

float fadeF(float fp, float a, float b) { return 1.0 - smoothstep(a, b, fp); }

half4 main(float2 p) {
    float d = p.y - hy;
    float dd = max(d, 0.35);
    float Z = fz / dd;
    float X = (p.x - cx) / dd;
    float fpv = fz * pxw / (dd * dd);   // plane-units per pixel (vertical)

    // --- three wave layers sampled from one tileable texture --------------
    float kv1 = 7.0;
    float2 uv1 = float2(X * 3.0 + t * 1.3, Z * kv1 + t * 5.5);
    half4 s1 = wave.eval(uv1);
    float w1 = fadeF(fpv * kv1, 5.0, 14.0);

    float kv2 = 26.0;
    float2 uv2 = float2(X * 11.0 - t * 3.0 + 131.0, Z * kv2 + t * 17.0 + 57.0);
    half4 s2 = wave.eval(uv2);
    float w2 = fadeF(fpv * kv2, 4.0, 12.0);

    float kv3 = 80.0;
    float2 uv3 = float2(X * 40.0 + t * 9.0 + 311.0, Z * kv3 + t * 36.0 + 229.0);
    half4 s3 = wave.eval(uv3);
    float w3 = fadeF(fpv * kv3, 3.5, 10.0);

    // swell bands rolling towards camera
    float sw = sin(Z * 1.6 - t * 1.1 + sin(X * 0.35 + t * 0.15) * 1.2);
    float wsw = fadeF(fpv * 1.6 * 6.0, 1.0, 3.0);

    float S = (float(s1.g) - 0.5) * 0.9 * w1
            + (float(s2.g) - 0.5) * 0.8 * w2
            + (float(s3.g) - 0.5) * 0.5 * w3;
    S = S * calm + sw * 0.05 * wsw * calm;

    // --- base gradient (nearness m: 0 at horizon, ~1 at the bottom) ---------
    float m = 1.0 / (1.0 + Z * 0.075);
    float3 base = mix(cHor, cMid, smoothstep(0.02, 0.42, m));
    base = mix(base, cDeep, smoothstep(0.40, 0.95, m));

    // large soft value variations (painted look)
    float blot = (float(s1.r) - 0.5) * w1 * 0.16 + sw * 0.045 * wsw;
    base *= 1.0 + blot * calm;

    float hi = smoothstep(0.10, 0.26, S);
    float hi2 = smoothstep(0.20, 0.34, S);
    float dk = smoothstep(0.06, 0.30, -S);
    float3 colr = base * (1.0 - 0.20 * dk * smoothstep(0.02, 0.2, m));
    colr += cHi * (hi * 0.22 + hi2 * 0.18) * (0.55 + 0.45 * m);

    // near-horizon fine shimmer (stands in for detail that has been faded out)
    float far = (1.0 - w2) * smoothstep(0.0, 0.05, m);
    colr += cHi * far * 0.035 * (0.5 + 0.5 * sin(Z * 0.9 + t * 0.7 + X * 0.05));

    // horizon haze: thin luminous line
    float hz = exp(-d / (3.0 + 2.0 * pxw)) * 0.40 + exp(-d / 30.0) * 0.12;
    colr += cHaze * hz;

    float a = smoothstep(-pxw, pxw * 0.5, d);
    return half4(half3(colr) * a, a);
}
"""

_REFL_SRC = """
uniform shader wave;
uniform float t;
uniform float hy;
uniform float cx;
uniform float pxw;
uniform float fz;
uniform float lx;
uniform float y0;
uniform float ym;
uniform float sig;
uniform float inten;
uniform float w0;
uniform float wk;
uniform float calm;
uniform float yend;
uniform float far;
uniform float3 lc;

half4 main(float2 p) {
    float d = max(p.y - hy, 0.35);
    float dy = max(p.y - y0, 0.0);
    float Z = fz / d;
    float X = (p.x - cx) / d;
    float fpv = fz * pxw / (d * d);
    float w = w0 + dy * wk;

    // wobbling, broken horizontal dashes
    float2 uvA = float2(X * 13.0 + t * 2.0, Z * 30.0 + t * 15.0);
    float2 uvB = float2(X * 30.0 - t * 5.0 + 91.0, Z * 85.0 + t * 31.0 + 17.0);
    half4 a = wave.eval(uvA);
    half4 b = wave.eval(uvB);
    float fa = 1.0 - smoothstep(4.0, 12.0, fpv * 30.0);
    float fb = 1.0 - smoothstep(3.0, 9.0, fpv * 85.0);
    float wob = ((float(a.g) - 0.5) * 1.8 * fa + (float(b.g) - 0.5) * 1.1 * fb) * calm;
    float dx = (p.x - lx) / w + wob;
    float lat = exp(-dx * dx * 1.4);
    float n = mix(0.5, float(a.r), fa) * 0.55 + mix(0.5, float(b.r), fb) * 0.45;
    float th = mix(0.47, 0.64, clamp(abs(dx) * 0.7, 0.0, 1.0));
    float soft = 0.035 + (1.0 - fb) * 0.2;
    float mask = smoothstep(th - soft, th + soft, n);
    // vertical profile
    float v;
    if (far > 0.5) {
        v = 0.30 + 0.70 * exp(-dy / sig);
    } else {
        float g = (p.y - ym) / sig;
        v = exp(-g * g) * 0.7 + 0.3 * exp(-dy / sig);
    }
    v *= 1.0 - smoothstep(yend * 0.6, yend, dy);
    v *= smoothstep(0.0, 3.0 + 4.0 * pxw, dy);
    float sp = smoothstep(0.60, 0.68, n) * lat;
    float k = inten * v * (lat * mask * 0.9 + lat * lat * 0.12 + sp * 0.7);
    return half4(half3(lc * k), 0.0);
}
"""

_OCEAN = None
_REFL = None


def _ocean_shaders():
    global _OCEAN, _REFL
    if _OCEAN is None:
        _OCEAN = _Shader(_OCEAN_SRC)
        _REFL = _Shader(_REFL_SRC)
    return _OCEAN, _REFL


def sea_palette(dawn=0.0, sunrise=0.0):
    """Colours used by the ocean for a given dawn / sunrise amount (rgb tuples)."""
    hor = mix(mix('#20355f', '#8f7aa6', dawn), '#f0b77c', sunrise)
    mid = mix(mix(SEA_MID, '#3a3f78', dawn), '#b98464', sunrise)
    deep = mix(mix(SEA_DEEP, '#161a44', dawn), '#4a3a5e', sunrise)
    hi = mix(mix('#4f7fc8', '#b79ac8', dawn), '#ffd9a0', sunrise)
    gl = mix(mix('#cfe3ff', '#ffe6d6', dawn), '#fff1c4', sunrise)
    haze = mix(mix('#3a5aa0', '#e0a0a0', dawn), '#ffe0a8', sunrise)
    return dict(hor=hor, mid=mid, deep=deep, hi=hi, gl=gl, haze=haze)


FOCAL = 1000.0   # virtual focal length of the sea's perspective (world units)


def ocean(c, t, horizon_y=700, rect_x=(-2000, 3920), bottom=2200, lights=(), dawn=0.0, sunrise=0.0, calm=1.0,
          glints=1.0, cx=960.0, reflect=1.0, focal=FOCAL):
    """Animated night sea from `horizon_y` down to `bottom`, covering x in `rect_x`.

    Rendered by an SkSL shader in WORLD coordinates (the camera transform on the canvas
    is respected, pans/zooms stay consistent). Waves are thin & dense near the horizon and
    grow towards the camera (true perspective, anti-aliased by distance).

    t:         time in seconds (use global f.T so motion is continuous across cuts)
    horizon_y: world y of the horizon line (sea starts here)
    rect_x:    (x0, x1) world range to cover (make it wide when the camera pans)
    bottom:    world y where the sea rect ends
    lights:    iterable of light tuples ABOVE the sea, each
               (x, y, colour, intensity[, y_base[, width]])
               -> a shimmering broken reflection column is drawn under x.
               y_base: water y directly under the light (default: horizon_y — far lights such
               as the lighthouse lamp or the sun). Give it for near lights, e.g. the boat
               lantern: (lx, ly, '#ffc86b', 1.0, waterline_y).
               width: column width multiplier (default 1; sun ~3).
    dawn:      0..1 pre-dawn violet/peach tint;  sunrise: 0..1 golden sea
    calm:      wave amplitude multiplier (0 = mirror-flat, 1 = normal, 1.5 = choppy)
    glints:    starlight sparkle amount (0..2)
    cx:        x of the perspective vanishing point (normally screen centre)
    reflect:   global multiplier for the light reflection columns
    """
    x0, x1 = rect_x
    if bottom <= horizon_y:
        return
    tex = _tex()
    osh, rsh = _ocean_shaders()
    pal = sea_palette(dawn, sunrise)
    pxw = _px(c)
    sh = osh.make([tex['wave']], t=t, hy=horizon_y, cx=cx, pxw=pxw, calm=calm, fz=focal,
                  cHor=pal['hor'], cMid=pal['mid'], cDeep=pal['deep'], cHi=pal['hi'], cHaze=pal['haze'])
    p = skia.Paint(Shader=sh)
    c.drawRect(skia.Rect.MakeLTRB(x0, horizon_y - 2 * pxw, x1, bottom), p)
    if glints > 0:
        sea_glints(c, t, horizon_y, (x0, x1), bottom, amount=glints * (1 - 0.5 * sunrise), color=pal['gl'], cx=cx)
    for L in lights:
        water_reflection(c, t, L, horizon_y=horizon_y, bottom=bottom, calm=calm, cx=cx, strength=reflect,
                         focal=focal)


def water_reflection(c, t, light, horizon_y=700, bottom=2200, calm=1.0, cx=960.0, strength=1.0, focal=FOCAL):
    """Draw ONE light's shimmering reflection column on the water (additive).
    light = (x, y, colour, intensity[, y_base[, width]]) — see ocean()."""
    lx, ly, lc, li = light[:4]
    if li * strength <= 0.003:
        return
    yb = light[4] if len(light) > 4 and light[4] is not None else horizon_y
    wmul = light[5] if len(light) > 5 else 1.0
    tex = _tex()
    _, rsh = _ocean_shaders()
    pxw = _px(c)
    yb = max(yb, horizon_y)
    h_above = max(yb - ly, 8.0)
    ym = yb + h_above              # mirror point
    far = yb <= horizon_y + 1.0
    if far:
        sig = 260.0
        yend = max(bottom - yb, 10.0) * 1.5
        w0 = 3.0 * wmul
        wk = 0.26 * wmul
    else:
        sig = max(h_above * 0.8, 20.0)
        yend = h_above * 3.2
        w0 = max(h_above * 0.10, 4.0) * wmul
        wk = 0.30 * wmul
    y_stop = min(bottom, yb + yend)
    wmax = w0 + (y_stop - yb) * wk
    sh = rsh.make([tex['wave']], t=t, hy=horizon_y, cx=cx, pxw=pxw, fz=focal, lx=lx, y0=yb, ym=ym, sig=sig,
                  inten=li * strength, w0=w0, wk=wk, calm=max(calm, 0.15), yend=yend, far=1.0 if far else 0.0,
                  lc=_rgb(lc))
    p = skia.Paint(Shader=sh)
    p.setBlendMode(ADD)
    k0, k1 = w0 * 1.9 + 3, wmax * 1.9 + 3
    c.drawPath(poly([(lx - k0, yb), (lx + k0, yb), (lx + k1, y_stop), (lx - k1, y_stop)]), p)


def sea_glints(c, t, horizon_y, rect_x, bottom, amount=1.0, color='#cfe3ff', cx=960.0, seed=5):
    """Twinkling starlight glints on the water (deterministic particles, world coords).
    Called by ocean(); exposed so scenes can add extra sparkle."""
    if amount <= 0:
        return
    try:
        cb = c.getLocalClipBounds()
        vx0, vx1, vy1 = max(rect_x[0], cb.left()), min(rect_x[1], cb.right()), min(bottom, cb.bottom())
    except Exception:
        vx0, vx1, vy1 = rect_x[0], rect_x[1], bottom
    if vx1 <= vx0 or vy1 <= horizon_y:
        return
    pxw = _px(c)
    dmax = vy1 - horizon_y
    d0 = max(1.5, 1.2 * pxw)
    ratio = 1.16
    nb = int(math.log(max(dmax / d0, 1.0001)) / math.log(ratio)) + 1
    ks = np.arange(nb)
    dk = d0 * ratio ** ks                    # top of each band (distance below horizon)
    cw = np.maximum(dk * 0.9, 3.0)           # cell width in world units
    xs, ys, al, sz = [], [], [], []
    rate = 1.3
    for k in range(nb):
        i0 = int(math.floor((vx0 - cx) / cw[k])) - 1
        i1 = int(math.floor((vx1 - cx) / cw[k])) + 1
        ii = np.arange(i0, i1 + 1)
        if len(ii) == 0:
            continue
        hsh = (ii * 73856093 + k * 19349663 + seed * 83492791) & 0x7fffffff
        ph = (hsh % 1000) / 1000.0
        ep = np.floor(t * rate + ph)                          # epoch -> re-roll position
        h2 = ((hsh ^ (ep.astype(np.int64) * 2654435761)) & 0x7fffffff)
        u1 = (h2 % 997) / 997.0
        u2 = ((h2 // 997) % 991) / 991.0
        u3 = ((h2 // 988027) % 983) / 983.0
        keep = u3 < 0.42
        if not keep.any():
            continue
        life = (t * rate + ph) - ep                            # 0..1 within epoch
        env = np.sin(np.pi * life) ** 3
        xx = cx + (ii + u1) * cw[k]
        dd = dk[k] * (1 + (ratio - 1) * u2)
        near = dd / (dd + 160.0)
        a = env * (0.35 + 0.65 * u3 / 0.42) * (0.25 + 0.75 * near) * amount
        m = keep & (a > 0.04)
        xs.append(xx[m]); ys.append(horizon_y + dd[m] * np.ones(m.sum())); al.append(a[m])
        sz.append(np.maximum(0.8 * pxw, dd[m] * 0.010 + 0.6))
    if not xs:
        return
    xs = np.concatenate(xs); ys = np.concatenate(ys); al = np.concatenate(al); sz = np.concatenate(sz)
    p = skia.Paint(AntiAlias=True, Color=col(color))
    p.setStrokeCap(skia.Paint.kRound_Cap)
    p.setBlendMode(ADD)
    lv = np.clip((al * 4).astype(int), 0, 4)
    sl = np.clip((np.log2(np.maximum(sz, 0.5)) * 2).astype(int), -2, 8)
    for lvl in range(1, 5):
        for s_ in np.unique(sl[lv == lvl]):
            m = (lv == lvl) & (sl == s_)
            if not m.any():
                continue
            w = 2 ** (s_ / 2.0)
            p.setAlphaf(min(1.0, lvl / 4.0))
            p.setStrokeWidth(w * 1.1)
            pts = [skia.Point(float(a), float(b)) for a, b in zip(xs[m], ys[m])]
            # horizontal dash: draw as short lines (glints are stretched sideways)
            segs = []
            for q in pts:
                segs.append(skia.Point(q.x() - w * 1.2, q.y()))
                segs.append(skia.Point(q.x() + w * 1.2, q.y()))
            c.drawPoints(skia.Canvas.kLines_PointMode, segs, p)


# ----------------------------------------------------------------------------
# placeholder implementations below are replaced step by step
# ----------------------------------------------------------------------------


# ============================================================================
# shared lighting environment (moonlit night / dawn / sunrise)
# ============================================================================
def _norm(v):
    l = math.sqrt(sum(a * a for a in v)) or 1.0
    return tuple(a / l for a in v)


def _vmix(a, b, k):
    return tuple(x + (y - x) * k for x, y in zip(a, b))


def _vscale(a, k):
    return tuple(x * k for x in a)


def light_env(dawn=0.0, sunrise=0.0, sun_dir=1.0):
    """Light colours shared by all set pieces so they sit in the same world.
    Returns dict: moon_dir, moon, amb, rim, sun_dir, sun (rgb tuples)."""
    moon = _vmix(_vmix((0.50, 0.55, 0.68), (0.55, 0.50, 0.64), dawn), (0.30, 0.29, 0.36), sunrise)
    amb = _vmix(_vmix((0.11, 0.13, 0.23), (0.25, 0.21, 0.33), dawn), (0.42, 0.35, 0.38), sunrise)
    rim = _vmix(_vmix((0.22, 0.30, 0.56), (0.55, 0.38, 0.50), dawn), (0.60, 0.45, 0.35), sunrise)
    sun = _vscale((1.0, 0.72, 0.45), sunrise * 0.95 + dawn * 0.10)
    return dict(moon_dir=_norm((-0.80, 0.42, 0.46)), moon=moon, amb=amb, rim=rim,
                sun_dir=_norm((0.78 * sun_dir, 0.18, 0.60)), sun=sun)


def _lit(albedo, env, k_moon=0.8, k_amb=1.0):
    """Flat-shaded colour of a surface roughly facing the camera under `env`."""
    a = _rgb(albedo)
    return tuple(clamp(a[i] * (env['amb'][i] * k_amb + env['moon'][i] * k_moon + env['sun'][i] * 0.7)) for i in range(3))


# ============================================================================
# LIGHTHOUSE (exterior)
# ============================================================================
_REV_SRC = """
uniform shader blot;
uniform float pxw;
uniform float yA;
uniform float yB;
uniform float rA;
uniform float rB;
uniform float pw;
uniform float dome;
uniform float ybase;
uniform float e;
uniform float4 bandsA;
uniform float4 bandsB;
uniform float3 cA;
uniform float3 cB;
uniform float3 Lm;
uniform float3 cm;
uniform float3 amb;
uniform float3 rimc;
uniform float3 Ls;
uniform float3 cs;
uniform float3 cw;
uniform float wTop;
uniform float wFall;
uniform float streak;
uniform float ao;
uniform float aoFall;
uniform float spec;

float band(float Y, float lo, float hi, float aa) {
    return smoothstep(lo - aa, lo + aa, Y) * (1.0 - smoothstep(hi - aa, hi + aa, Y));
}

half4 main(float2 p) {
    float f = clamp((p.y - yA) / (yB - yA), 0.0, 1.0);
    float r;
    float3 n;
    float u;
    float nz;
    if (dome > 0.5) {
        float q = sqrt(max(1.0 - f * f, 0.0));
        r = mix(rB, rA, q);
        u = clamp(p.x / max(r, 0.001), -1.0, 1.0);
        nz = sqrt(max(1.0 - u * u, 0.0));
        n = normalize(float3(u * q, f * 1.2 + 0.1, nz * q));
    } else {
        r = mix(rA, rB, pow(f, pw));
        u = clamp(p.x / max(r, 0.001), -1.0, 1.0);
        nz = sqrt(max(1.0 - u * u, 0.0));
        float tilt = (rA - rB) / abs(yB - yA);
        n = normalize(float3(u, tilt, nz));
    }
    float Yh = (ybase - p.y) - e * r * nz;
    float aa = pxw * 0.8;
    float red = band(Yh, bandsA.x, bandsA.y, aa) + band(Yh, bandsA.z, bandsA.w, aa)
              + band(Yh, bandsB.x, bandsB.y, aa) + band(Yh, bandsB.z, bandsB.w, aa);
    float3 alb = mix(cA, cB, clamp(red, 0.0, 1.0));
    float ang = asin(u);
    half4 bt = blot.eval(float2(ang * 30.0 + 17.0, Yh * 0.05 + 40.0));
    half4 bt2 = blot.eval(float2(ang * 70.0 + 3.0, Yh * 0.4 + 9.0));
    alb *= 1.0 - streak * ((float(bt.g) - 0.5) * 0.30 + (float(bt2.b) - 0.5) * 0.10 + (float(bt.r) - 0.5) * 0.12);
    float dm = clamp(dot(n, Lm) * 0.62 + 0.38, 0.0, 1.0);
    dm = dm * dm * (3.0 - 2.0 * dm);
    float ds = clamp(dot(n, Ls) * 0.6 + 0.4, 0.0, 1.0);
    ds = ds * ds;
    float3 light = amb * (0.8 + 0.3 * n.y) + cm * dm + cs * ds;
    light += rimc * pow(max(u, 0.0), 5.0) * (0.35 + 0.65 * nz);
    light += cw * exp(-max(wTop - Yh, 0.0) / wFall) * (0.3 + 0.7 * nz);
    float occ = 1.0 - ao * exp(-max(p.y - yB, 0.0) / aoFall);
    float3 colr = alb * light * occ;
    // soft moon specular sheen (glossy paint / iron)
    float3 hv = normalize(Lm + float3(0.0, 0.0, 1.0));
    colr += cm * pow(max(dot(n, hv), 0.0), 24.0) * spec;
    return half4(half3(colr), 1.0);
}
"""
_REV = None


def _rev_shader():
    global _REV
    if _REV is None:
        _REV = _Shader(_REV_SRC)
    return _REV


def _rev_paint(env, pxw, yA, yB, rA, rB, cA, cB=None, bands=(), dome=False, pw=1.0, e=0.09, ybase=None,
               warm=(0, 0, 0), wtop=0.0, wfall=40.0, streak=1.0, ao=0.0, aofall=10.0, spec=0.0):
    b = list(bands)[:4] + [(-9e5, -9e5)] * (4 - len(list(bands)[:4]))
    sh = _rev_shader().make([_tex()['blot']], pxw=pxw, yA=yA, yB=yB, rA=rA, rB=rB, pw=pw, dome=1.0 if dome else 0.0,
                            ybase=yA if ybase is None else ybase, e=e,
                            bandsA=(b[0][0], b[0][1], b[1][0], b[1][1]), bandsB=(b[2][0], b[2][1], b[3][0], b[3][1]),
                            cA=_rgb(cA), cB=_rgb(cB or cA), Lm=env['moon_dir'], cm=env['moon'], amb=env['amb'],
                            rimc=env['rim'], Ls=env['sun_dir'], cs=env['sun'], cw=warm, wTop=wtop, wFall=wfall,
                            streak=streak, ao=ao, aoFall=aofall, spec=spec)
    return skia.Paint(AntiAlias=True, Shader=sh)


def _arc(r, y0, e, a0=-math.pi / 2, a1=math.pi / 2, n=24, x0=0.0):
    """Points of a horizontal circle (radius r, height y0) seen from slightly below:
    theta=0 is the front (raised by e*r).  Returns list of (x, y)."""
    pts = []
    for i in range(n + 1):
        a = a0 + (a1 - a0) * i / n
        pts.append((x0 + r * math.sin(a), y0 - e * r * math.cos(a)))
    return pts


def _band_path(r_bot, y_bot, r_top, y_top, e, n=24):
    """Closed outline of a cylinder/cone section between two horizontal circles (front arcs)."""
    bot = _arc(r_bot, y_bot, e, n=n)
    top = _arc(r_top, y_top, e, n=n)[::-1]
    path = skia.Path()
    path.moveTo(*bot[0])
    for q in bot[1:]:
        path.lineTo(*q)
    for q in top:
        path.lineTo(*q)
    path.close()
    return path


# geometry (local units at s=1, y up is negative, origin = base centre on the rock)
LH = dict(
    plinth_y=-24.0, plinth_r=(80.0, 75.0),
    tower_y=(-24.0, -384.0), tower_r=(64.0, 44.0),
    corbel_y=(-384.0, -399.0), corbel_r=(44.0, 67.0),
    floor_y=-400.0, deck_r=69.0,
    rail_y=-427.0, rail_mid=-414.0, rail_r=66.0,
    base_y=(-400.0, -414.0), base_r=42.0,
    glass_y=(-414.0, -474.0), glass_r=40.0,
    eave_y=(-474.0, -481.0), eave_r=49.0,
    dome_y=(-481.0, -513.0), dome_r=(46.0, 5.0),
    ball_y=-518.5, ball_r=6.5, rod_top=-541.0,
    lamp_y=-445.0,
    bands=((64.0, 116.0), (168.0, 220.0), (272.0, 324.0)),
)
LH_HEIGHT = 520  # lighthouse height at s=1 from base to lamp top (vent ball ~ 525, rod tip 541)
LH_WHITE = '#ece7da'
LH_RED = '#bf4238'
LH_IRON = '#2a2f45'
LH_STONE = '#8d8f9e'


def lighthouse_anchors(x, y, s=1.0):
    """World coordinates of useful lighthouse points for a lighthouse drawn at (x, y, s).
    keys: lamp (x,y) — lens centre / beam origin; gallery_y — walking surface of the balcony;
    gallery_x (left, right) — balcony extent; rail_top_y — top of the railing;
    glass (left, top, right, bottom) — lamp-room glazing box; door (x, y) — bottom centre of
    the door (threshold); door_size (w, h); dome_top (x, y) — top of the vent ball;
    rod_top (x, y); tower_top_y; base_y; deng_scale — robot scale that fits the gallery
    (Deng ≈ 36*s tall, head above the rail); hatch (x, y) — roof hatch centre (beam_up)."""
    g = LH
    e = 0.09
    return dict(
        lamp=(x, y + g['lamp_y'] * s),
        gallery_y=y + g['floor_y'] * s,
        gallery_x=(x - g['rail_r'] * s * 0.97, x + g['rail_r'] * s * 0.97),
        rail_top_y=y + (g['rail_y'] - e * g['rail_r']) * s,
        glass=(x - g['glass_r'] * s, y + g['glass_y'][1] * s, x + g['glass_r'] * s, y + g['glass_y'][0] * s),
        door=(x, y + (g['plinth_y'] - e * g['tower_r'][0]) * s),
        door_size=(26 * s, 44 * s),
        dome_top=(x, y + (g['ball_y'] - g['ball_r']) * s),
        rod_top=(x, y + g['rod_top'] * s),
        tower_top_y=y + g['tower_y'][1] * s,
        base_y=y,
        deng_scale=0.10 * s,
        hatch=(x, y + (g['dome_y'][0] + 0.55 * (g['dome_y'][1] - g['dome_y'][0])) * s),
    )


def _tower_r(yy):
    (y0, y1), (r0, r1) = LH['tower_y'], LH['tower_r']
    f = clamp((yy - y0) / (y1 - y0))
    return r0 + (r1 - r0) * f


def lighthouse(c, x, y, s=1.0, lamp=1.0, dawn=0.0, lit_windows=1.0, t=0.0, beam_up=0.0, beam_angle=None,
               sunrise=0.0, sun_dir=1.0, parts='all', lamp_color=WARM, view_e=0.09, glow=1.0):
    """Lighthouse exterior. (x, y) = base centre on the rock. s=1 → ~520 tall (rod tip 541).

    lamp:        0..1 brightness of the lamp room (0 = dark lighthouse)
    dawn/sunrise: lighting of the tower (sunrise: warm key light from side `sun_dir` = +1 right / -1 left)
    lit_windows: 0..1 warm tower windows + little lamp over the door
    t:           time (lamp shimmer, hatch animation)
    beam_up:     0..1 climax: roof hatch swings open, glow pours upward
    beam_angle:  optional sweep angle (same value you pass to beam()); the lamp room flares
                 when the beam faces the camera
    parts:       'all' | 'back' | 'front' — 'back' draws everything except the front railing
                 and lamp glow; 'front' draws only those. Draw a character standing on the
                 gallery between the two calls so he stands *behind* the front railing.
    view_e:      ellipse factor of horizontal circles (0 = level view, 0.09 default: seen from
                 slightly below; ~0.2 for a low close camera)
    glow:        multiplier for the additive lamp glow halo
    Returns world (lx, ly) of the lamp centre (for beams/glows).
    """
    g = LH
    e = view_e
    env = light_env(dawn, sunrise, sun_dir)
    lamp = clamp(lamp)
    flare = 0.0
    if beam_angle is not None and beam_up < 0.5:
        flare = max(0.0, math.sin(beam_angle)) ** 6
    shimmer = 1.0 + 0.03 * noise1(t * 7.0, 3) + 0.02 * noise1(t * 17.0, 5)
    lampk = lamp * shimmer
    warm = _vscale(_rgb(lamp_color), 0.55 * lampk)
    lx, ly = x, y + g['lamp_y'] * s
    with at(c, x, y, sx=s):
        pxw = _px(c)
        if parts in ('all', 'back'):
            _lh_body(c, env, pxw, e, lampk, warm, lit_windows, t, dawn, sunrise)
            _lh_gallery_back(c, env, pxw, e, lampk, warm)
            _lh_lamp_room(c, env, pxw, e, lampk, flare, t, beam_up, lamp_color)
            _lh_roof(c, env, pxw, e, lampk, warm, beam_up, t, lamp_color)
        if parts in ('all', 'front'):
            _lh_gallery_front(c, env, pxw, e, lampk, warm)
    if parts in ('all', 'front') and lamp > 0 and glow > 0:
        k = lampk * glow
        soft_glow(c, lx, ly, 230 * s, lamp_color, 0.32 * k * (1 + 1.5 * flare) * (1 - 0.5 * beam_up))
        soft_glow(c, lx, ly, 70 * s, '#fff4d8', 0.55 * k * (1 + flare))
        if flare > 0.01:
            soft_glow(c, lx, ly, 520 * s, lamp_color, 0.35 * flare * k)
        if beam_up > 0:
            hx, hy = x, y + (g['dome_y'][0] - 20) * s
            soft_glow(c, hx, hy - 30 * s, 160 * s, lamp_color, 0.5 * k * beam_up)
    return lx, ly


def _lh_body(c, env, pxw, e, lampk, warm, lit_windows, t, dawn, sunrise):
    g = LH
    # --- plinth (granite base course) ---------------------------------------
    r0, r1 = g['plinth_r']
    py = g['plinth_y']
    c.drawPath(_band_path(r0, 2.0, r1, py, e), _rev_paint(env, pxw, 2.0, py, r0, r1, LH_STONE, streak=2.2, e=e,
                                                          warm=_vscale(warm, 0.0), spec=0.0))
    # stone block joints on the plinth
    jp = stroke(_lit('#3a3d52', env, 0.5), max(0.7, 0.6 * pxw), 0.6)
    c.drawPath(_path_from(_arc(r0 * 0.99, py * 0.5, e, n=20)), jp)
    for a in (-1.2, -0.75, -0.3, 0.15, 0.6, 1.05):
        xa = r0 * math.sin(a)
        ya0 = 2.0 - e * r0 * math.cos(a)
        ya1 = py - e * r1 * math.cos(a)
        ym = (ya0 + ya1) / 2
        if a in (-0.75, 0.15, 1.05):
            c.drawLine(xa, ya0, xa, ym, jp)
        else:
            c.drawLine(xa * 0.99, ym, xa * 0.97, ya1, jp)
    # --- tower -----------------------------------------------------------------
    (ty0, ty1), (tr0, tr1) = g['tower_y'], g['tower_r']
    top_warm = warm
    paint = _rev_paint(env, pxw, ty0, ty1, tr0, tr1, LH_WHITE, LH_RED, bands=g['bands'], e=e, ybase=ty0,
                       warm=top_warm, wtop=(ty0 - ty1), wfall=38.0, streak=1.0, ao=0.45, aofall=9.0, spec=0.10)
    c.drawPath(_band_path(tr0, ty0, tr1, ty1, e), paint)
    # soft contact shadow where tower meets plinth
    sp = linear((0, ty0), (0, ty0 - 16), [(0, '#05060f', 0.35), (1, '#05060f', 0.0)])
    c.drawPath(_band_path(tr0, ty0 + 1, _tower_r(ty0 - 16), ty0 - 16, e), sp)
    # --- door --------------------------------------------------------------------
    _lh_door(c, env, pxw, e, lit_windows, t)
    # --- windows -----------------------------------------------------------------
    for (Y, u) in ((142.0, 0.30), (246.0, -0.26), (338.0, 0.12)):
        _lh_window(c, env, pxw, e, ty0 - Y, u, lit_windows, t, Y)


def _path_from(pts, close=False):
    path = skia.Path()
    path.moveTo(*pts[0])
    for q in pts[1:]:
        path.lineTo(*q)
    if close:
        path.close()
    return path


def _lh_window(c, env, pxw, e, yy, u, lit, t, seed):
    r = _tower_r(yy)
    nz = math.sqrt(max(0.0, 1 - u * u))
    xx = u * r
    yc = yy - e * r * nz
    w, h = 9.5 * (0.35 + 0.65 * nz), 15.0
    # reveal (deep-set stone opening)
    frame = rrect(xx - w / 2 - 1.8, yc - h / 2 - 2.0, w + 3.6, h + 3.5, w / 2 + 1.5)
    c.drawRRect(frame, fill(_lit('#b9b4a6', env, 0.7)))
    glass = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(xx - w / 2, yc - h / 2, w, h), w / 2, w / 2)
    fl = 0.92 + 0.08 * noise1(t * 3.0 + seed, 7)
    if lit > 0:
        gp = linear((0, yc - h / 2), (0, yc + h / 2), [(0, '#ffe7a8', 1), (1, '#ffae52', 1)])
        c.drawRRect(glass, fill('#141a2c'))
        gp.setAlphaf(clamp(lit * fl))
        c.drawRRect(glass, gp)
    else:
        c.drawRRect(glass, linear((0, yc - h / 2), (0, yc + h / 2), [(0, '#2a3552', 1), (1, '#141a2c', 1)]))
    # mullion cross
    mp = stroke('#3a2a24' if lit > 0 else '#1a1f30', max(0.9, 0.7 * pxw))
    c.drawLine(xx, yc - h / 2 + 1, xx, yc + h / 2, mp)
    c.drawLine(xx - w / 2, yc + 1.5, xx + w / 2, yc + 1.5, mp)
    # sill
    c.drawRRect(rrect(xx - w / 2 - 3, yc + h / 2 + 0.5, w + 6, 2.6, 1.2), fill(_lit('#d8d2c2', env, 0.9)))
    if lit > 0:
        soft_glow(c, xx, yc, 26, '#ffb85a', 0.30 * lit * fl)


def _lh_door(c, env, pxw, e, lit, t):
    g = LH
    tr0 = g['tower_r'][0]
    yb = g['plinth_y'] - e * tr0 + 0.5
    w, h = 26.0, 44.0
    # stone surround
    sur = skia.Path()
    sur.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(-w / 2 - 4, yb - h - 4, w + 8, h + 4), w / 2 + 4, w / 2 + 4))
    c.drawPath(sur, fill(_lit('#a9a497', env, 0.75)))
    # keystone
    c.drawPath(poly([(-3.2, yb - h - 5.5), (3.2, yb - h - 5.5), (2.2, yb - h + 1.5), (-2.2, yb - h + 1.5)]),
               fill(_lit('#c9c3b3', env, 0.8)))
    door = skia.Path()
    door.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(-w / 2, yb - h, w, h + 0.5), w / 2, w / 2))
    base = _lit('#7a5236', env, 0.85)
    dark = _lit('#4a3020', env, 0.6)
    c.drawPath(door, linear((-w / 2, 0), (w / 2, 0), [(0, base, 1), (1, dark, 1)]))
    with saved_clip(c, door):
        pl = stroke(_lit('#3a2418', env, 0.6), max(0.8, 0.6 * pxw), 0.85)
        for xx in (-6.5, 0.0, 6.5):
            c.drawLine(xx, yb - h, xx, yb, pl)
        hp = stroke(_lit('#23232e', env, 0.8), 2.2)
        c.drawLine(-w / 2, yb - h * 0.72, w / 2 - 4, yb - h * 0.72, hp)
        c.drawLine(-w / 2, yb - h * 0.25, w / 2 - 4, yb - h * 0.25, hp)
        # moon rim on the left edge
        c.drawPath(door, stroke(env['moon'], 1.6, 0.25))
    c.drawCircle(w / 2 - 5, yb - h * 0.48, 1.6, fill('#d8b25a'))
    # threshold step
    c.drawRRect(rrect(-w / 2 - 7, yb - 1.0, w + 14, 5.0, 2.0), fill(_lit('#9a9aa6', env, 0.9)))
    # little lamp over the door
    ly = yb - h - 13
    c.drawLine(0, ly - 5, 0, ly + 1, stroke('#1e2233', 1.4))
    c.drawRRect(rrect(-3, ly - 1, 6, 7, 2), fill('#1e2233'))
    if lit > 0:
        fl = 0.9 + 0.1 * noise1(t * 5.0, 9)
        c.drawRRect(rrect(-2, ly, 4, 5, 1.5), fill('#ffe2a0', lit * fl))
        soft_glow(c, 0, ly + 3, 40, '#ffbf66', 0.30 * lit * fl)
        # warm pool on the door and threshold
        pl = radial((0, ly + 22), 50, [(0, '#ffb45a', 0.16 * lit), (1, '#ffb45a', 0.0)])
        pl.setBlendMode(ADD)
        c.drawCircle(0, ly + 20, 55, pl)


@contextmanager
def saved_clip(c, path):
    c.save()
    c.clipPath(path, skia.ClipOp.kIntersect, True)
    try:
        yield
    finally:
        c.restore()


def _rail_posts(c, env, pxw, e, lampk, front):
    g = LH
    R = g['rail_r']
    n = 22
    wpx = max(2.0, 0.9 * pxw)
    for i in range(n):
        a = (i + 0.5) * math.tau / n
        ca = math.cos(a)
        if (ca > 0) != front:
            continue
        xx = R * math.sin(a)
        yb = g['floor_y'] - e * R * ca
        yt = g['rail_y'] - e * R * ca
        side = math.sin(a)
        colr = _lit(LH_IRON, env, 0.6 + 0.5 * max(0, -side))
        if not front:
            colr = _vmix(colr, (0.02, 0.03, 0.06), 0.35)
        c.drawLine(xx, yb, xx, yt, stroke(colr, wpx, cap=skia.Paint.kButt_Cap))
        if front and lampk > 0 and abs(side) < 0.9:
            c.drawLine(xx + 0.4, yb, xx + 0.4, yt, stroke(WARM, wpx * 0.4, 0.18 * lampk, cap=skia.Paint.kButt_Cap))


def _rail_line(c, env, pxw, e, y0, w, lampk, front, lit_top=False):
    g = LH
    R = g['rail_r']
    if front:
        pts = _arc(R, y0, e, -math.pi / 2, math.pi / 2, n=28)
    else:
        pts = _arc(R, y0, e, math.pi / 2, 3 * math.pi / 2, n=28)
    colr = _lit(LH_IRON, env, 0.9)
    if not front:
        colr = _vmix(colr, (0.02, 0.03, 0.06), 0.35)
    c.drawPath(_path_from(pts), stroke(colr, max(w, 0.9 * pxw)))
    if front and lit_top:
        # moonlit top edge + warm lamp kiss
        hi = _path_from([(q[0], q[1] - w * 0.3) for q in pts[:18]])
        c.drawPath(hi, stroke(env['moon'], max(w * 0.4, 0.6 * pxw), 0.45))
        if lampk > 0:
            c.drawPath(_path_from([(q[0], q[1] - w * 0.3) for q in pts[6:23]]), stroke(WARM, max(w * 0.35, 0.5 * pxw), 0.35 * lampk))


def _lh_gallery_back(c, env, pxw, e, lampk, warm):
    g = LH
    # corbel (flared support under the balcony)
    (cy0, cy1), (cr0, cr1) = g['corbel_y'], g['corbel_r']
    c.drawPath(_band_path(cr0, cy0, cr1, cy1, e),
               _rev_paint(env, pxw, cy0, cy1, cr0, cr1, LH_WHITE, pw=2.2, e=e, warm=_vscale(warm, 0.3),
                          wtop=0, wfall=10, streak=0.8))
    # little brackets under the deck
    bp = fill(_lit('#9aa0b8', env, 0.35))
    for i in range(12):
        a = (i + 0.5) * math.tau / 12 - math.pi / 2
        if math.cos(a) <= 0.05:
            continue
        ca = math.cos(a)
        xx0 = cr0 * math.sin(a)
        xx1 = (cr1 - 3) * math.sin(a)
        y0 = cy0 - e * cr0 * ca
        y1 = cy1 - e * cr1 * ca + 1
        wv = 2.4 * ca + 0.6
        c.drawPath(poly([(xx0 - wv, y0), (xx0 + wv, y0), (xx1 + wv, y1), (xx1 - wv, y1)]), bp)
    # back railing (seen behind the lamp room)
    _rail_line(c, env, pxw, e, g['rail_y'], 2.4, lampk, False)
    _rail_line(c, env, pxw, e, g['rail_mid'], 1.3, lampk, False)
    _rail_posts(c, env, pxw, e, lampk, False)
    # deck slab edge
    R = g['deck_r']
    fy = g['floor_y']
    deck = _band_path(R, fy + 6, R, fy, e)
    c.drawPath(deck, _rev_paint(env, pxw, fy + 6, fy, R, R, LH_IRON, e=e, warm=_vscale(warm, 0.8), wtop=0,
                                wfall=30, streak=0.5, spec=0.25))
    # top face of the deck is hidden (seen from below) — thin moonlit lip
    c.drawPath(_path_from(_arc(R, fy, e, n=28)), stroke(env['moon'], max(0.8, 0.6 * pxw), 0.35))


def _lh_gallery_front(c, env, pxw, e, lampk, warm):
    g = LH
    _rail_posts(c, env, pxw, e, lampk, True)
    _rail_line(c, env, pxw, e, g['rail_mid'], 1.3, lampk, True)
    _rail_line(c, env, pxw, e, g['rail_y'], 2.6, lampk, True, lit_top=True)


def _lh_lamp_room(c, env, pxw, e, lampk, flare, t, beam_up, lamp_color):
    g = LH
    # lamp-room base wall (red iron drum with vents)
    (by0, by1), br = g['base_y'], g['base_r']
    c.drawPath(_band_path(br, by0, br, by1, e),
               _rev_paint(env, pxw, by0, by1, br, br, LH_RED, e=e, warm=_vscale(_rgb(lamp_color), 0.25 * lampk),
                          wtop=0, wfall=20, streak=0.6, spec=0.3))
    for a in (-0.9, -0.3, 0.3, 0.9):
        xx = br * math.sin(a)
        yy = (by0 + by1) / 2 - e * br * math.cos(a)
        c.drawOval(skia.Rect.MakeXYWH(xx - 2.2 * math.cos(a), yy - 2.2, 4.4 * math.cos(a), 4.4), fill('#1a1420', 0.8))
    # glazing
    (gy0, gy1), gr = g['glass_y'], g['glass_r']
    gpath = _band_path(gr, gy0, gr, gy1, e)
    lyc = g['lamp_y']
    k = clamp(lampk)
    # dark glass base
    c.drawPath(gpath, linear((0, gy1), (0, gy0), [(0, '#1b2546', 1), (0.6, '#0f1630', 1), (1, '#0b1026', 1)]))
    with saved_clip(c, gpath):
        # back mullions seen through the glass
        for i in range(8):
            a = (i + 0.5) * math.tau / 8 + 0.2
            if math.cos(a) >= 0:
                continue
            xx = gr * math.sin(a)
            c.drawLine(xx, gy0, xx, gy1, stroke('#0a0d1a', 1.5, 0.8))
        if k > 0:
            lc = _rgb(lamp_color)
            gp = radial((0, lyc), 46, [(0, '#fff6dc', 1.0 * k), (0.35, lamp_color, 0.95 * k),
                                      (1.0, _vmix(lc, (0.6, 0.25, 0.1), 0.5), 0.55 * k)])
            c.drawPath(gpath, gp)
        # the lens (beehive) inside
        lw, lh = 21.0, 25.0
        lens = skia.Path()
        lens.addRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(-lw, lyc - lh, 2 * lw, 2 * lh), 10, 10))
        if k > 0:
            c.drawPath(lens, radial((0, lyc), 28, [(0, '#ffffff', k), (0.5, '#fff0c0', k * 0.9), (1, lamp_color, k * 0.6)]))
            lp = stroke('#b8742a', 0.8, 0.45 * k)
            for j in range(-5, 6):
                yy = lyc + j * 4.4
                c.drawPath(_path_from(_arc(lw * math.sqrt(max(0.0, 1 - (j / 6.5) ** 2)), yy, e * 0.5, n=10)), lp)
            c.drawLine(0, lyc - lh, 0, lyc + lh, stroke('#c07a30', 1.0, 0.4 * k))
        else:
            c.drawPath(lens, linear((-lw, 0), (lw, 0), [(0, '#6f7fa8', 0.55), (0.4, '#3a4670', 0.5), (1, '#1c2444', 0.5)]))
            lp = stroke('#9fb4e0', 0.7, 0.35)
            for j in range(-5, 6):
                yy = lyc + j * 4.4
                c.drawPath(_path_from(_arc(lw * math.sqrt(max(0.0, 1 - (j / 6.5) ** 2)), yy, e * 0.5, n=10)), lp)
        # pedestal under the lens
        c.drawRect(skia.Rect.MakeLTRB(-8, lyc + lh, 8, gy0), fill('#1a1424' if k <= 0 else '#5a3a20'))
        # cool reflection of the sky on the glass (moon side)
        rp = linear((-gr, 0), (gr, 0), [(0, env['moon'], 0.0), (0.12, env['moon'], 0.35 * (1 - 0.7 * k)),
                                        (0.28, env['moon'], 0.0), (0.75, env['moon'], 0.0),
                                        (0.9, env['rim'], 0.18), (1, env['rim'], 0.0)])
        rp.setBlendMode(ADD)
        c.drawPath(gpath, rp)
    # front mullions (iron frame)
    for i in range(8):
        a = (i + 0.5) * math.tau / 8 + 0.2
        ca = math.cos(a)
        if ca < 0:
            continue
        xx = gr * math.sin(a)
        wv = 1.2 + 2.2 * ca
        y0 = gy0 - e * gr * ca
        y1 = gy1 - e * gr * ca
        c.drawLine(xx, y0, xx, y1, stroke(_lit(LH_IRON, env, 0.5), max(wv, 0.8 * pxw), cap=skia.Paint.kButt_Cap))
        if k > 0:
            c.drawLine(xx - wv * 0.45, y0, xx - wv * 0.45, y1, stroke(lamp_color, max(0.5, wv * 0.25), 0.35 * k,
                                                                     cap=skia.Paint.kButt_Cap))
    # horizontal astragal + sill & head rings
    ip = stroke(_lit(LH_IRON, env, 0.7), max(1.1, 0.8 * pxw))
    c.drawPath(_path_from(_arc(gr, (gy0 + gy1) / 2 + 6, e, n=24)), stroke(_lit(LH_IRON, env, 0.6), max(0.9, 0.7 * pxw)))
    c.drawPath(_path_from(_arc(gr + 0.5, gy0, e, n=24)), stroke(_lit(LH_IRON, env, 0.7), max(3.0, 0.8 * pxw)))
    c.drawPath(_path_from(_arc(gr + 0.5, gy1, e, n=24)), stroke(_lit(LH_IRON, env, 0.7), max(2.4, 0.8 * pxw)))
    del ip
    if k > 0 and flare > 0.01:
        fp = radial((0, lyc), 40, [(0, '#ffffff', 0.9 * flare * k), (1, lamp_color, 0.0)])
        fp.setBlendMode(ADD)
        c.drawCircle(0, lyc, 40, fp)


def _lh_roof(c, env, pxw, e, lampk, warm, beam_up, t, lamp_color):
    g = LH
    (ey0, ey1), er = g['eave_y'], g['eave_r']
    c.drawPath(_band_path(er, ey0, er, ey1, e),
               _rev_paint(env, pxw, ey0, ey1, er, er, LH_IRON, e=e, warm=_vscale(_rgb(lamp_color), 0.5 * lampk),
                          wtop=8, wfall=6, streak=0.3, spec=0.4))
    # eave underside lit by the lamp
    if lampk > 0:
        c.drawPath(_path_from(_arc(er - 1, ey0 + 0.8, e, n=24)), stroke(lamp_color, 1.4, 0.5 * clamp(lampk)))
    (dy0, dy1), (dr0, dr1) = g['dome_y'], g['dome_r']
    cut_f = 0.55
    open_a = ease_io(clamp(beam_up)) * 1.9
    dome_path = skia.Path()
    n = 26
    pts_l, pts_r = [], []
    for i in range(n + 1):
        f = i / n
        q = math.sqrt(max(0.0, 1 - f * f))
        r = dr1 + (dr0 - dr1) * q
        yy = dy0 + (dy1 - dy0) * f
        pts_r.append((r, yy - e * r))
        pts_l.append((-r, yy - e * r))
    base = _arc(dr0, dy0, e, n=24)
    dome_path.moveTo(*base[0])
    for q in base[1:]:
        dome_path.lineTo(*q)
    for q in pts_r[1:]:
        dome_path.lineTo(q[0], q[1])
    for q in reversed(pts_l[1:]):
        dome_path.lineTo(q[0], q[1])
    dome_path.close()
    dpaint = _rev_paint(env, pxw, dy0, dy1, dr0, dr1, LH_RED, dome=True, e=e * 0.0, streak=0.5, spec=0.55,
                        warm=_vscale(_rgb(lamp_color), 0.18 * lampk), wtop=0, wfall=6)
    ycut = dy0 + (dy1 - dy0) * cut_f
    rcut = dr1 + (dr0 - dr1) * math.sqrt(1 - cut_f ** 2)
    if open_a <= 0.001:
        c.drawPath(dome_path, dpaint)
        # seams
        sp = stroke(_lit('#5a1e22', env, 0.8), max(0.7, 0.5 * pxw), 0.7)
        for a in (-1.0, -0.45, 0.1, 0.65, 1.2):
            pts = []
            for i in range(0, n + 1, 2):
                f = i / n
                q = math.sqrt(max(0.0, 1 - f * f))
                r = dr1 + (dr0 - dr1) * q
                pts.append((r * math.sin(a), dy0 + (dy1 - dy0) * f - e * r * math.cos(a)))
            c.drawPath(_path_from(pts), sp)
        _vent_ball(c, env, pxw, 0.0, 0.0)
        return
    # --- hatch open (climax) -------------------------------------------------
    with saved(c):
        c.clipRect(skia.Rect.MakeLTRB(-200, ycut, 200, 50))
        c.drawPath(dome_path, dpaint)
    # opening
    op = skia.Rect.MakeLTRB(-rcut, ycut - 3.5, rcut, ycut + 3.5)
    c.drawOval(op, fill('#1a0d0a'))
    if lampk > 0:
        c.drawOval(op, radial((0, ycut), rcut, [(0, '#ffffff', clamp(lampk)), (0.6, '#fff0b8', clamp(lampk)),
                                               (1, lamp_color, 0.8 * clamp(lampk))]))
    # lid swinging open around the right hinge
    with saved(c):
        c.translate(rcut, ycut)
        c.rotate(math.degrees(open_a))
        c.translate(-rcut, -ycut)
        with saved(c):
            c.clipRect(skia.Rect.MakeLTRB(-200, -900, 200, ycut))
            c.drawPath(dome_path, dpaint)
        c.drawOval(skia.Rect.MakeLTRB(-rcut, ycut - 2.0, rcut, ycut + 2.0), fill('#3a1216'))
        _vent_ball(c, env, pxw, 0.0, 0.0)


def _vent_ball(c, env, pxw, dx, dy):
    g = LH
    by, br = g['ball_y'] + dy, g['ball_r']
    c.drawRect(skia.Rect.MakeLTRB(dx - 2.5, by + br - 1, dx + 2.5, g['dome_y'][1] + dy + 1), fill(_lit(LH_IRON, env, 0.7)))
    c.drawLine(dx, by, dx, g['rod_top'] + dy, stroke(_lit(LH_IRON, env, 0.9), max(1.2, 0.7 * pxw)))
    c.drawCircle(dx, g['rod_top'] + dy, 1.4, fill(_lit(LH_IRON, env, 0.9)))
    bp = radial((dx - br * 0.4, by - br * 0.45), br * 1.5,
                [(0, _lit('#6a7090', env, 1.2), 1), (0.5, _lit(LH_IRON, env, 0.7), 1), (1, _lit('#10121e', env, 0.5), 1)])
    c.drawCircle(dx, by, br, bp)
    c.drawCircle(dx + br * 0.2, by, br, stroke(env['rim'], 0.8, 0.4))


def ease_io(x):
    x = clamp(x)
    return x * x * (3 - 2 * x)


_BEAM_SRC = """
uniform shader blot;
uniform float t;
uniform float3 lc;
uniform float inten;
uniform float dust;
uniform float occl;
uniform float tail;
uniform float lend;

half4 main(float2 p) {
    float L = p.x / 1000.0;
    float v = p.y / 100.0;
    float av = abs(v);
    float edge = 1.0 - smoothstep(0.55, 1.0, av);
    float body = exp(-v * v * 5.0);
    float core = exp(-v * v * 40.0);
    float skirt = (1.0 - av) * (1.0 - av);
    float lat = body * 0.55 + core * 0.35 + skirt * 0.18;
    lat *= edge;
    float along = pow(clamp(1.0 - L / lend, 0.0, 1.0), tail);
    along *= smoothstep(0.0, 0.015, L);
    along *= mix(1.0, smoothstep(0.02, 0.12, L), occl);
    // drifting mist and lengthwise light shafts
    half4 m1 = blot.eval(float2(L * 420.0 - t * 14.0, v * 34.0 + 50.0 + t * 1.5));
    half4 m2 = blot.eval(float2(v * 90.0 + 13.0, L * 22.0 - t * 0.35));
    float mist = 1.0 + dust * ((float(m1.r) - 0.5) * 0.9 + (float(m2.g) - 0.5) * 0.6);
    float a = inten * along * lat * max(mist, 0.0);
    return half4(half3(lc * a), 0.0);
}
"""
_BEAMSH = None


def _beam_shader():
    global _BEAMSH
    if _BEAMSH is None:
        _BEAMSH = _Shader(_BEAM_SRC)
    return _BEAMSH


def beam_geometry(x, y, angle, length=2600, width0=30, width1=520, mode='auto', elev=None, horizon_y=None,
                  dist=0.9, n=30):
    """Projected centre line of the beam: list of (sx, sy, half_width, L) for L in 0..1, plus facing
    (-1 away .. +1 towards the camera). See beam() for the parameters."""
    screen = (mode == 'screen') or (mode == 'auto' and elev is None and abs(angle + math.pi / 2) < 1e-9)
    out = []
    if screen:
        dx, dy = math.cos(angle), math.sin(angle)
        for i in range(n + 1):
            L = (i / n) ** 1.35
            out.append((x + dx * length * L, y + dy * length * L, (width0 + (width1 - width0) * L) / 2, L))
        return out, 0.0
    el = elev or 0.0
    ce = math.cos(el)
    dxz, dyz, dzz = ce * math.cos(angle), math.sin(el), ce * math.sin(angle)
    ell = dist
    F = length / max(ell, 1e-3)
    hy = horizon_y if horizon_y is not None else y + 0.10 * length
    hL = (hy - y) / F
    zmin = 0.16
    for i in range(n + 1):
        L = (i / n) ** 1.35
        Z = 1.0 - ell * L * dzz
        if Z < zmin:
            break
        X = ell * L * dxz
        Y = hL + ell * L * dyz
        sx = x + F * X / Z
        sy = hy - F * Y / Z
        hw = (width0 + (width1 - width0) * L) / 2 / Z
        out.append((sx, sy, hw, L))
    return out, dzz


def beam(c, x, y, angle, length=2600, width0=30, width1=520, intensity=1.0, color='#ffe7a0', mode='auto',
         elev=None, horizon_y=None, t=0.0, dist=0.9, dust=1.0, flare=1.0):
    """Volumetric lighthouse beam from the lamp at (x, y) (additive light).

    angle: in the default 'sweep' interpretation this is the AZIMUTH of the rotating lens,
           i.e. pass the continuous sweep angle `f.T * 0.9`: 0 = pointing right, pi/2 = towards
           the camera (beam widens, glare flares at the lamp), pi = left, 3pi/2 = away (beam
           shortens towards the horizon and hides behind the lantern).
    mode:  'sweep' | 'screen' | 'auto' (default). 'screen' treats angle as a flat 2D screen
           direction (0 right, -pi/2 up). 'auto' = sweep, except the legacy call with angle
           exactly -pi/2 (straight up), which is drawn as a vertical screen-space beam.
    elev:  elevation above the horizontal in sweep mode (pi/2 = straight up; e.g. for the
           climax tilt animate elev 0 -> pi/2).
    length: on-screen reach (world units) when the beam is side-on; width0/width1: beam width
           at the lamp / at the far end.
    horizon_y: world y of the horizon (gives the right perspective dip when the beam points
           away); default ~0.1*length below the lamp.
    dist:  beam length relative to the camera distance (bigger = more dramatic perspective).
    intensity: 0..2; color: beam colour; dust: mist/shaft modulation amount; flare: glare
           multiplier when the beam faces the camera; t: time for drifting mist.
    Returns the facing value (-1 away .. +1 towards camera).
    """
    if intensity <= 0:
        return 0.0
    pts, facing = beam_geometry(x, y, angle, length, width0, width1, mode, elev, horizon_y, dist)
    if len(pts) < 2:
        return facing
    f01 = max(0.0, facing)
    inten = intensity * (0.42 + 0.30 * (facing + 1) / 2 + 0.25 * f01 ** 2)
    occl = max(0.0, -facing)
    # triangle mesh: 5 lateral columns per row
    cols = (-1.0, -0.5, 0.0, 0.5, 1.0)
    pos, tex = [], []
    m = len(pts)
    for i, (sx, sy, hw, L) in enumerate(pts):
        j0, j1 = max(0, i - 1), min(m - 1, i + 1)
        tx, ty = pts[j1][0] - pts[j0][0], pts[j1][1] - pts[j0][1]
        tl = math.hypot(tx, ty) or 1.0
        nx, ny = -ty / tl, tx / tl
        hw2 = hw * 1.25 + 2.0
        for v in cols:
            pos.append(skia.Point(sx + nx * hw2 * v, sy + ny * hw2 * v))
            tex.append(skia.Point(L * 1000.0, v * 100.0))
    idx = []
    k = len(cols)
    for i in range(m - 1):
        for j in range(k - 1):
            a, b = i * k + j, i * k + j + 1
            c2, d = (i + 1) * k + j, (i + 1) * k + j + 1
            idx += [a, b, c2, b, d, c2]
    lend = pts[-1][3] if pts[-1][3] < 0.999 else 1.0
    tail = 1.5 if facing > -0.2 else 1.1
    sh = _beam_shader().make([_tex()['blot']], t=t, lc=_rgb(color), inten=inten, dust=dust, occl=occl,
                             tail=tail, lend=max(lend, 0.05) if lend < 1.0 else 1.0)
    p = skia.Paint(AntiAlias=True, Shader=sh)
    p.setBlendMode(ADD)
    verts = skia.Vertices(skia.Vertices.kTriangles_VertexMode, pos, tex, None, idx)
    c.drawVertices(verts, p, skia.BlendMode.kModulate)
    # glare at the lamp when facing the camera
    if f01 > 0.02 and flare > 0:
        g = f01 ** 4 * intensity * flare
        soft_glow(c, x, y, width0 * 7.0, color, 0.55 * g)
        soft_glow(c, x, y, width0 * 2.0, '#ffffff', 0.7 * g)
        # anamorphic streak
        sp = linear((x - width0 * 14, y), (x + width0 * 14, y), [(0, color, 0.0), (0.5, '#fff8e0', 0.5 * g), (1, color, 0.0)])
        sp.setBlendMode(ADD)
        c.drawRect(skia.Rect.MakeLTRB(x - width0 * 14, y - width0 * 0.08 - 1, x + width0 * 14, y + width0 * 0.08 + 1), sp)
    return facing


def beam_light(x, y, angle, horizon_y=700, length=2600, intensity=1.0, color='#ffe7a0', dist=0.9, width0=30,
               width1=520):
    """A light tuple for ocean(lights=[...]) that makes the sweeping beam shimmer on the sea.
    Pass the same values as to beam()."""
    pts, facing = beam_geometry(x, y, angle, length, width0, width1, 'sweep', None, horizon_y, dist)
    if not pts:
        return (x, y, color, 0.0)
    q = pts[min(len(pts) - 1, len(pts) // 3)]
    f01 = max(0.0, facing)
    k = intensity * (0.18 + 0.25 * (facing + 1) / 2 + 0.35 * f01 ** 2)
    return (q[0], min(q[1], horizon_y - 5), color, k, None, 1.6 + 1.5 * f01)


def beam_timelapse(c, x, y, amount=1.0, radius=900, color='#ffe7a0', squash=0.16):
    """Time-lapse blur of a sweeping beam: a flattened luminous disc around the lamp (s02)."""
    if amount <= 0:
        return
    with saved(c):
        c.translate(x, y - radius * squash * 0.25)
        c.scale(1.0, squash)
        p = radial((0, 0), radius, [(0, '#fff8e0', 0.55 * amount), (0.12, color, 0.35 * amount),
                                    (0.45, color, 0.12 * amount), (1, color, 0.0)])
        p.setBlendMode(ADD)
        c.drawCircle(0, 0, radius, p)
    soft_glow(c, x, y, radius * 0.25, color, 0.35 * amount)


# ============================================================================
# ISLAND
# ============================================================================
_ROCK_SRC = """
uniform shader blot;
uniform float3 cTop;
uniform float3 cBot;
uniform float yT;
uniform float yB;
uniform float nscale;
uniform float3 warm;
uniform float2 wpos;
uniform float wrad;
uniform float wetY;
uniform float3 sheen;
uniform float2 lightDir;
uniform float3 lightC;

half4 main(float2 p) {
    float g = clamp((p.y - yT) / (yB - yT), 0.0, 1.0);
    float3 c = mix(cTop, cBot, g * g * (3.0 - 2.0 * g));
    half4 n = blot.eval(p * nscale + float2(37.0, 11.0));
    half4 n2 = blot.eval(p * nscale * 2.2 + float2(5.0, 71.0));
    c *= 1.0 + (float(n.r) - 0.5) * 0.28 + (float(n.g) - 0.5) * 0.10 + (float(n2.g) - 0.5) * 0.05;
    float2 d = (p - wpos) / wrad;
    c += warm * exp(-dot(d, d) * 2.2);
    float wet = smoothstep(wetY - 22.0, wetY - 2.0, p.y);
    c *= 1.0 - 0.35 * wet;
    return half4(half3(c), 1.0);
}
"""
_ROCK = None


def _rock_shader():
    global _ROCK
    if _ROCK is None:
        _ROCK = _Shader(_ROCK_SRC)
    return _ROCK


def _rock_paint(top, bot, yT, yB, env, nscale=0.012, warm=(0, 0, 0), wpos=(0, 0), wrad=1.0, wet_y=1e5, local_scale=1.0):
    sh = _rock_shader().make([_tex()['blot']], cTop=_rgb(top), cBot=_rgb(bot), yT=yT, yB=yB, nscale=nscale * 256 / 256,
                             warm=warm, wpos=wpos, wrad=wrad, wetY=wet_y, sheen=env['moon'], lightDir=(0, 0),
                             lightC=env['moon'])
    return skia.Paint(AntiAlias=True, Shader=sh)


# hand-authored island geometry (local units at s=1; (0,0) = where the lighthouse stands)
_ISL_WL = 140.0   # waterline (local y)
_ISL_BACK = [(-440, 150), (-420, 105), (-396, 66), (-374, 36), (-354, 50), (-322, 40), (-275, 26), (-200, 4),
             (-125, -12), (-45, -22), (40, -26), (118, -26), (160, -36), (196, -50), (232, -52), (266, -34),
             (300, -4), (345, 40), (400, 90), (450, 150)]
_ISL_TOP = [(-300, 74), (-262, 44), (-190, 16), (-110, 2), (0, -2), (112, 2), (196, 14), (268, 38), (322, 74),
            (262, 66), (160, 56), (60, 52), (-50, 54), (-150, 60), (-236, 70)]
_ISL_FACE = [(-338, 150), (-318, 96), (-300, 74), (-236, 70), (-150, 60), (-50, 54), (60, 52), (160, 56),
             (262, 66), (322, 74), (346, 104), (372, 150)]
_ISL_BOULDERS = [(-372, 138, 58, 40), (-262, 146, 50, 30), (-150, 146, 62, 34), (-28, 150, 52, 28),
                 (92, 148, 60, 32), (214, 144, 56, 36), (322, 140, 64, 42), (420, 150, 40, 22)]
_ISL_STEPS = [(-292, 86), (-262, 74), (-230, 62), (-196, 52), (-160, 43), (-124, 35), (-90, 27), (-58, 20),
              (-30, 14), (-8, 10)]


def island_anchors(x, y, s=1.0):
    """World coordinates of island points for an island drawn at (x, y, s).
    keys: top (x,y) = lighthouse spot; waterline_y; plateau (x0, x1, y) — flat walkable top
    (characters stand at y..y+40*s in front of the lighthouse); jetty (x0, x1, deck_y);
    mooring (x, y) bollard top; boat (x, y) waterline point where a moored boat sits;
    seat (x, y) — a low rock at the water's edge to sit on (s05); shallows (x, y) —
    shallow water next to the rocks (splash spot); crates (x, y) — flat spot right of the
    lighthouse for props / the crate tower."""
    def P(px, py):
        return (x + px * s, y + py * s)
    return dict(
        top=(x, y), waterline_y=y + _ISL_WL * s,
        plateau=(x - 210 * s, x + 230 * s, y + 20 * s),
        jetty=(x - 500 * s, x - 300 * s, y + 98 * s),
        mooring=P(-478, 90), boat=P(-420, _ISL_WL + 12),
        seat=P(250, 108), shallows=P(310, _ISL_WL + 30),
        crates=P(165, 30),
    )


def island(c, x, y, s=1.0, dawn=0.0, t=0.0, sunrise=0.0, sun_dir=1.0, lamp=1.0, door_light=1.0, props=True,
           jetty=True, reflection=True, foam=1.0):
    """Rocky island; (x, y) = top-centre of the rock where the lighthouse stands. s=1 → ~900 wide.
    Waterline at y + 140*s. Draw it after ocean() and before lighthouse().

    t:          time — animates the foam lapping at the waterline
    dawn/sunrise/sun_dir: lighting (see light_env)
    lamp:       lighthouse lamp brightness (warm kiss on the plateau)
    door_light: warm light pool from the little lamp over the door (0 = off)
    props:      draw the crates/barrel/rope near the lighthouse
    jetty:      draw the wooden jetty on the left (boat mooring)
    reflection: draw the island's dark reflection on the water
    foam:       foam amount
    See island_anchors() for jetty/boat/seat positions."""
    env = light_env(dawn, sunrise, sun_dir)
    with at(c, x, y, sx=s):
        pxw = _px(c)
        wl = _ISL_WL
        if reflection:
            _isl_reflection(c, t, env, wl)
        warm = _vscale((1.0, 0.7, 0.35), 0.22 * door_light)
        # back ridge (far plane: bluer, lower contrast)
        back = smooth_path(_ISL_BACK, close=True, tension=0.8)
        c.drawPath(back, _rock_paint(_lit('#34406f', env, 0.5), _lit('#1b2242', env, 0.4), -30, wl, env,
                                     nscale=0.35, wet_y=wl))
        _facets(c, env, _ISL_BACK_FACETS, -30, wl, 0.35, contrast=0.55)
        c.drawPath(smooth_path(_ISL_BACK[1:-1], tension=0.8), stroke(env['rim'], 2.2, 0.30))
        # main front face, split into planes
        face = smooth_path(_ISL_FACE, close=True, tension=0.6)
        c.drawPath(face, _rock_paint(_lit('#2c355c', env, 0.5), _lit('#141930', env, 0.35), 50, wl, env,
                                     nscale=0.45, warm=warm, wpos=(0, 40), wrad=120, wet_y=wl))
        with saved_clip(c, face):
            _facets(c, env, _ISL_FACE_FACETS, 50, wl, 0.45, warm=warm, wpos=(0, 40), wrad=120, wet_y=wl)
        # top surface (plateau) — lighter, moonlit
        top = smooth_path(_ISL_TOP, close=True, tension=0.6)
        warm_top = _vscale((1.0, 0.72, 0.4), 0.30 * door_light + 0.05 * lamp)
        c.drawPath(top, _rock_paint(_lit('#525c8a', env, 0.75), _lit('#363f6c', env, 0.6), -5, 70, env,
                                    nscale=0.5, warm=warm_top, wpos=(0, 16), wrad=90))
        with saved_clip(c, top):
            _facets(c, env, _ISL_TOP_FACETS, -5, 70, 0.5, warm=warm_top, wpos=(0, 16), wrad=90, contrast=0.6)
        c.drawPath(smooth_path(_ISL_TOP[:9], tension=0.6), stroke(env['moon'], 1.6, 0.30))
        _isl_moss(c, env)
        _isl_path(c, env, door_light)
        for i, (bx, by, bw, bh) in enumerate(_ISL_BOULDERS):
            _boulder(c, env, bx, by, bw, bh, i, wl)
        if jetty:
            _jetty(c, env, pxw, t, wl)
        if props:
            _isl_props(c, env, pxw, dawn, sunrise, sun_dir)
        _isl_grass(c, env, t)
        if foam > 0:
            _isl_foam(c, t, env, wl, foam)


# facet polygons: (points, light) — light > 0 faces the moon (upper-left), < 0 turned away
_ISL_FACE_FACETS = [
    ([(-340, 150), (-318, 96), (-300, 74), (-262, 70), (-250, 110), (-270, 152)], 0.7),
    ([(-262, 70), (-190, 62), (-170, 100), (-196, 152), (-270, 152), (-250, 110)], -0.3),
    ([(-190, 62), (-100, 56), (-110, 96), (-90, 152), (-196, 152), (-170, 100)], 0.45),
    ([(-100, 56), (10, 53), (0, 90), (30, 152), (-90, 152), (-110, 96)], -0.15),
    ([(10, 53), (110, 54), (130, 100), (110, 152), (30, 152), (0, 90)], 0.3),
    ([(110, 54), (210, 60), (230, 96), (220, 152), (110, 152), (130, 100)], -0.5),
    ([(210, 60), (322, 74), (346, 104), (374, 152), (220, 152), (230, 96)], -0.8),
]
_ISL_TOP_FACETS = [
    ([(-300, 74), (-262, 44), (-190, 16), (-150, 10), (-165, 60), (-236, 70)], 0.8),
    ([(-150, 10), (-110, 2), (0, -2), (112, 2), (140, 8), (120, 55), (-50, 54), (-165, 60)], 0.25),
    ([(140, 8), (196, 14), (268, 38), (322, 74), (262, 66), (160, 56), (120, 55)], -0.55),
]
_ISL_BACK_FACETS = [
    ([(-440, 150), (-420, 105), (-396, 66), (-374, 36), (-354, 50), (-322, 40), (-275, 26), (-200, 4), (-230, 90),
      (-300, 150)], 0.6),
    ([(118, -26), (160, -36), (196, -50), (214, -10), (196, 60), (140, 40)], 0.55),
    ([(196, -50), (232, -52), (266, -34), (300, -4), (345, 40), (400, 90), (450, 150), (300, 150), (214, -10)], -0.65),
]


def _facets(c, env, facets, yT, yB, nscale, warm=(0, 0, 0), wpos=(0, 0), wrad=1.0, wet_y=1e5, contrast=1.0):
    for pts, lv in facets:
        path = smooth_path(pts, close=True, tension=0.35)
        k = lv * contrast
        if k > 0:
            col_ = env['moon']
            p = linear((pts[0][0], yT), (pts[len(pts) // 2][0], yB), [(0, col_, 0.20 * k), (1, col_, 0.04 * k)])
            p.setBlendMode(ADD)
        else:
            p = linear((0, yT), (0, yB), [(0, '#050714', -0.45 * k), (1, '#050714', -0.25 * k)])
        c.drawPath(path, p)


def _boulder(c, env, bx, by, bw, bh, i, wl):
    r = nprng(100 + i)
    pts = []
    n = 9
    for k in range(n):
        a = math.pi + k / (n - 1) * math.pi
        rr = 1.0 + r.uniform(-0.12, 0.12)
        pts.append((bx + math.cos(a) * bw * rr, by + math.sin(a) * bh * rr))
    pts = [(bx - bw, wl + 12)] + pts + [(bx + bw, wl + 12)]
    path = smooth_path(pts, close=True, tension=0.7)
    c.drawPath(path, _rock_paint(_lit('#2e3862', env, 0.55), _lit('#10152c', env, 0.3), by - bh, wl + 5, env,
                                 nscale=0.6, wet_y=wl))
    with saved_clip(c, path):
        # shadowed right half
        sx = bx + bw * r.uniform(0.0, 0.3)
        c.drawPath(poly([(sx, by - bh * 1.3), (bx + bw * 1.2, by - bh * 1.3), (bx + bw * 1.2, wl + 14),
                         (sx - bw * 0.25, wl + 14)]), fill('#04060f', 0.35))
        # moonlit top-left plane
        lp = linear((bx - bw, by - bh), (bx, by), [(0, env['moon'], 0.22), (1, env['moon'], 0.0)])
        lp.setBlendMode(ADD)
        c.drawPath(poly([(bx - bw * 1.2, by - bh * 1.3), (sx, by - bh * 1.3), (sx - bw * 0.3, by - bh * 0.1),
                         (bx - bw * 1.2, by + bh * 0.2)]), lp)
    cap = smooth_path(pts[1:5], tension=0.7)
    c.drawPath(cap, stroke(env['moon'], 1.8, 0.25))


def _isl_moss(c, env):
    """Soft moss cushions draped along the upper rock edges."""
    mc = _lit('#3a5f55', env, 0.8)
    hc = _lit('#6d9c7c', env, 1.0)
    r = nprng(31)
    edge = _ISL_TOP[:9]
    spots = []
    for i in range(len(edge) - 1):
        (x0, y0), (x1, y1) = edge[i], edge[i + 1]
        for k in range(2):
            u = r.uniform(0.1, 0.9)
            spots.append((x0 + (x1 - x0) * u, y0 + (y1 - y0) * u + r.uniform(2, 9), r.uniform(9, 20)))
    for (px, py) in ((-300, 74), (322, 74)):
        spots.append((px + 12 * (1 if px < 0 else -1), py + 4, 12))
    for (mx, my, mw) in spots:
        if abs(mx) < 95:
            continue      # keep the lighthouse plinth area clean
        pts = []
        for k in range(10):
            a = k / 10 * math.tau
            rr = 1 + r.uniform(-0.3, 0.3)
            pts.append((mx + math.cos(a) * mw * rr, my + math.sin(a) * mw * 0.32 * rr))
        path = smooth_path(pts, close=True, tension=0.8)
        c.drawPath(path, fill(mc, 0.8))
        c.drawPath(smooth_path(pts[5:10], tension=0.8), stroke(hc, 1.2, 0.35))


def _isl_path(c, env, door_light):
    r = nprng(41)
    for i, (px, py) in enumerate(_ISL_STEPS):
        w = 17 - i * 0.6 + r.uniform(-2, 2)
        h = 5.5 - i * 0.12
        rr = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(px - w / 2, py - h / 2, w, h), h / 2, h / 2)
        c.drawRRect(rr, fill(_lit('#7c80a0', env, 0.9)))
        c.drawRRect(rrect(px - w / 2, py - h / 2 + h * 0.55, w, h * 0.45, h / 4), fill(_lit('#3a3f60', env, 0.6), 0.8))
        if door_light > 0:
            k = clamp(1 - (abs(px) + abs(py - 10) * 2) / 160)
            if k > 0:
                c.drawRRect(rr, fill('#ffb45a', 0.35 * k * door_light, blend=ADD))


def _isl_grass(c, env, t):
    gp = stroke(_lit('#3f6b58', env, 0.9), 1.6, 0.9)
    for (gx, gy, n, seed) in ((-235, 58, 5, 1), (-140, 42, 4, 2), (205, 42, 5, 3), (265, 58, 4, 4), (-310, 80, 4, 5),
                              (120, 44, 3, 6)):
        r = nprng(200 + seed)
        for k in range(n):
            xx = gx + k * 3.2 - n * 1.6
            hgt = r.uniform(7, 13)
            sway = math.sin(t * 1.7 + seed + k * 0.7) * 1.6
            p = skia.Path()
            p.moveTo(xx, gy)
            p.quadTo(xx + sway * 0.4, gy - hgt * 0.6, xx + sway + r.uniform(-2, 2), gy - hgt)
            c.drawPath(p, gp)


def _isl_foam(c, t, env, wl, amt, x0=-445, x1=455):
    """Foam lapping at the waterline: a soft broken band that breathes up and down."""
    fcol = FOAM
    xs = np.arange(x0, x1 + 1, 6.0)
    ph = t * 1.25 + xs * 0.021
    lap = np.sin(ph)
    yc = wl + 8 - 3.2 * lap
    nz = np.array([noise1(v * 0.035 + t * 0.25, 8) for v in xs])
    th = np.clip(2.2 + 3.0 * (nz + 0.3) + 1.5 * (0.5 + 0.5 * lap), 0.0, None)
    top = [(float(a), float(b - h * 0.6)) for a, b, h in zip(xs, yc, th)]
    bot = [(float(a), float(b + h * 0.4)) for a, b, h in zip(xs, yc, th)][::-1]
    band = _path_from(top + bot, close=True)
    c.drawPath(band, fill(fcol, 0.30 * amt))
    # brighter clumps where the band is thick
    for i in range(0, len(xs), 3):
        if nz[i] > 0.18:
            w = 8 + 22 * nz[i]
            c.drawOval(skia.Rect.MakeXYWH(xs[i] - w / 2, yc[i] - 1.3, w, 2.6), fill('#eef6ff', clamp(0.35 * nz[i]) * amt))
    # thin line where rock meets water
    c.drawPath(_path_from(top[::2]), stroke('#e8f2ff', 0.9, 0.18 * amt))
    # outgoing ripple rings
    for k in range(4):
        ph = (t * 0.35 + k * 0.25) % 1.0
        rx = (x1 - x0) / 2 + 20 + ph * 110
        ry = 14 + ph * 22
        a = (1 - ph) ** 1.5 * 0.22 * amt
        rp = stroke(fcol, 1.3, a)
        cxm = (x0 + x1) / 2
        c.drawArc(skia.Rect.MakeLTRB(cxm - rx, wl + 6 - ry, cxm + rx, wl + 6 + ry), 10, 160, False, rp)


def _isl_reflection(c, t, env, wl):
    pts = [(px, wl + (wl - py) * 0.55) for px, py in _ISL_BACK]
    path = smooth_path(pts, close=True, tension=0.8)
    p = linear((0, wl), (0, wl + 110), [(0, '#050818', 0.75), (1, '#050818', 0.0)])
    c.drawPath(path, p)
    # horizontal breakup by wave lines
    for k in range(9):
        yy = wl + 8 + k * 11 + 2 * math.sin(t * 1.1 + k)
        xw = 380 - k * 18
        c.drawLine(-xw + 20 * math.sin(t * 0.7 + k * 2), yy, xw, yy, stroke(SEA_HI, 1.3, 0.10))


def _jetty(c, env, pxw, t, wl):
    deck_y = 98.0
    x0, x1 = -505.0, -300.0
    wood = _lit('#8a6444', env, 0.8)
    wood_d = _lit('#4a3322', env, 0.6)
    # posts (pilings) into the water
    for px in (-492, -430, -368, -318):
        c.drawRect(skia.Rect.MakeLTRB(px - 5, deck_y, px + 5, wl + 6), linear((px - 5, 0), (px + 5, 0), [(0, wood, 1), (1, wood_d, 1)]))
        c.drawOval(skia.Rect.MakeXYWH(px - 9, wl + 3, 18, 5), fill(FOAM, 0.35 + 0.15 * math.sin(t * 2 + px)))
        # wobbly reflection of the post
        for k in range(4):
            yy = wl + 10 + k * 8
            dx = 2.5 * math.sin(t * 2.2 + k * 1.7 + px)
            c.drawRect(skia.Rect.MakeXYWH(px - 4 + dx, yy, 8, 5), fill('#0b0f22', 0.45 * (1 - k / 4)))
    # cross brace
    c.drawLine(-492, deck_y + 10, -430, wl - 4, stroke(wood_d, 3.0))
    c.drawLine(-430, deck_y + 10, -368, wl - 4, stroke(wood_d, 3.0))
    # deck: front face + thin top surface
    c.drawRect(skia.Rect.MakeLTRB(x0, deck_y - 3, x1, deck_y + 7), linear((0, deck_y - 3), (0, deck_y + 7), [(0, wood, 1), (1, wood_d, 1)]))
    c.drawRect(skia.Rect.MakeLTRB(x0, deck_y - 5, x1, deck_y - 2), fill(_lit('#b08a66', env, 0.95)))
    pl = stroke(_lit('#3a281c', env, 0.5), max(0.7, 0.6 * pxw), 0.8)
    xx = x0 + 10
    while xx < x1:
        c.drawLine(xx, deck_y - 3, xx, deck_y + 7, pl)
        xx += 11
    # bollard + rope coil
    bx = -478
    c.drawRRect(rrect(bx - 5, deck_y - 18, 10, 16, 3), fill(_lit('#2a2f45', env, 0.8)))
    c.drawRRect(rrect(bx - 7, deck_y - 20, 14, 4, 2), fill(_lit('#3a4060', env, 0.9)))
    rp = stroke(_lit('#c8a870', env, 0.9), 1.8)
    for k in range(3):
        c.drawOval(skia.Rect.MakeXYWH(bx - 6 - k, deck_y - 14 + k * 3.0, 12 + 2 * k, 3.4), rp)
    # ladder down to the water
    lp = stroke(wood_d, 2.4)
    c.drawLine(-452, deck_y, -452, wl + 2, lp)
    c.drawLine(-440, deck_y, -440, wl + 2, lp)
    for k in range(4):
        yy = deck_y + 9 + k * 9
        c.drawLine(-452, yy, -440, yy, stroke(wood_d, 1.6))
    # a lantern hook post at the jetty root
    c.drawLine(-312, deck_y - 4, -312, deck_y - 52, stroke(_lit('#2a2f45', env, 0.8), 3.0))
    c.drawLine(-312, deck_y - 50, -300, deck_y - 50, stroke(_lit('#2a2f45', env, 0.8), 2.0))


def _isl_props(c, env, pxw, dawn, sunrise, sun_dir):
    crate(c, 150, 44, 0.30, kind='crate', dawn=dawn, sunrise=sunrise, seed=3)
    crate(c, 186, 46, 0.25, kind='crate', dawn=dawn, sunrise=sunrise, seed=5)
    crate(c, 166, 11, 0.22, kind='crate', dawn=dawn, sunrise=sunrise, seed=7)
    crate(c, 214, 50, 0.26, kind='barrel', dawn=dawn, sunrise=sunrise, seed=2)
    # coiled rope on the ground
    rp = stroke(_lit('#c8a870', env, 0.9), 1.6)
    for k in range(3):
        c.drawOval(skia.Rect.MakeXYWH(-128 - k * 2, 38 - k * 1.2, 22 + k * 4, 6 + k * 1.2), rp)


# ============================================================================
# LAMP ROOM (interior, full screen 1920x1080)
# ============================================================================
LR = dict(
    hy=520.0,          # eye level / outside horizon
    f=900.0,           # focal length of the room's cylinder projection
    top_h=360.0,       # glazing head height above eye (screen units at centre)
    sill_h=-190.0,     # sill height (below eye)
    floor_h=-290.0,    # wall/floor junction
    lens=(960.0, 470.0, 170.0),
    wheel=(1372.0, 640.0, 104.0),
    lever=(826.0, 772.0),
    hatch=(960.0, 58.0, 200.0, 44.0),
)


def _lr_y(x, h):
    """Screen y of a horizontal circle on the room wall at height h (relative to eye) at screen x."""
    u = (x - 960.0) / LR['f']
    return LR['hy'] - h * math.sqrt(1.0 + u * u)


def _lr_curve(h, x0=-300, x1=2220, n=48):
    return [(x0 + (x1 - x0) * i / n, _lr_y(x0 + (x1 - x0) * i / n, h)) for i in range(n + 1)]


def _lr_mullion_xs(n=16, offset=0.0):
    """Screen x of window mullions (angles around the back wall) with a width factor."""
    out = []
    for k in range(-n // 2, n // 2 + 1):
        a = (k + 0.5) * math.tau / n + offset
        if abs(a) < math.radians(66):
            out.append((960.0 + LR['f'] * math.tan(a), math.cos(a)))
    return out


_LRLIGHT_SRC = """
uniform float lamp;
uniform float2 lc;
uniform float3 amb;
uniform float3 win;
uniform float3 warm;
uniform float rot;
uniform float up;
uniform float hy;
uniform float fz;
uniform float sweep;

half4 main(float2 p) {
    float u = (p.x - 960.0) / fz;
    float k = sqrt(1.0 + u * u);
    float h = (hy - p.y) / k;          // wall height coordinate
    // window light: strongest in the glazing band, spilling onto floor & ceiling
    float wb = smoothstep(-330.0, -190.0, h) * (1.0 - smoothstep(360.0, 470.0, h));
    float3 L = amb + win * (0.35 + 0.65 * wb);
    // warm lens light
    float2 d = (p - lc) / float2(760.0, 620.0);
    float r2 = dot(d, d);
    float fall = 1.0 / (1.0 + r2 * 3.2);
    // rotating bullseye beams sweeping around the walls
    float ang = atan(u);
    float bs = 0.0;
    for (int i = 0; i < 6; i++) {
        float a = rot + float(i) * 1.0471976;
        float dd = sin((ang - a) * 0.5);
        bs += exp(-dd * dd * 90.0);
    }
    float onwall = smoothstep(80.0, 260.0, length(p - lc));
    float3 W = warm * lamp * (fall * (0.95 - 0.25 * up) + bs * sweep * onwall * (1.0 - up) * 0.45);
    // floor pool right under the lens
    float2 fp = (p - float2(lc.x, lc.y + 420.0)) / float2(520.0, 170.0);
    W += warm * lamp * 0.45 * exp(-dot(fp, fp) * 1.3);
    L += W;
    return half4(half3(min(L, float3(1.0))), 1.0);
}
"""
_LRLIGHT = None


def _lr_light_shader():
    global _LRLIGHT
    if _LRLIGHT is None:
        _LRLIGHT = _Shader(_LRLIGHT_SRC)
    return _LRLIGHT


def lamp_room(c, t, lamp=1.0, lens_rot=0.0, dawn=0.0, view_offset=0.0, pointing_up=0.0, wheel_rot=0.0, lever=0.0,
              hatch=None, sunrise=0.0, outside=True, lamp_color=WARM, beam_sweep=1.0, dust=1.0):
    """Full-screen INTERIOR of the lamp room at the top of the lighthouse (1920x1080 frame):
    curved glazing showing the night sea & sky, brass/iron window frames, a big rotating
    Fresnel lens on a cast-iron pedestal with a tilting yoke, a plank floor, an iron
    hand-wheel with gears (right of the pedestal), a brass ignition lever (left of the
    pedestal) and a roof hatch.

    t:           time (s)
    lamp:        0..1 lamp brightness (warm flood light, glowing prism rings, rotating light)
    lens_rot:    rotation phase of the lens (radians; e.g. t*0.6 while lit)
    dawn/sunrise: outside sky/sea tint (+ cool→warm fill light)
    view_offset: horizontal camera offset in screen units (interior parallax; outside moves 25%)
    pointing_up: 0..1 lens assembly tilted to aim straight up (climax); also opens the hatch
    wheel_rot:   rotation of the hand-wheel (radians); gears follow
    lever:       0..1 ignition lever position (0 = up/off, 1 = pulled down)
    hatch:       0..1 roof hatch opening (default = pointing_up)
    outside:     draw the outside view (sky/stars/sea) through the glass
    beam_sweep:  strength of the lens beams sweeping across walls
    Deng (360 tall) stands on the floor at y≈900, left of the lens (x≈560–700).
    Returns (lens_x, lens_y, lens_r) — the centre/radius of the front bullseye (the burner
    / heart slot sits at the centre)."""
    g = LR
    lamp = clamp(lamp)
    pu = clamp(pointing_up)
    hatch = pu if hatch is None else clamp(hatch)
    ox = -view_offset
    lx, ly, lr = g['lens']
    lxs = lx + ox
    with saved(c):
        # ---------------- outside view through the glazing ----------------
        band = _lr_glazing_path(ox)
        with saved(c):
            c.clipPath(band, skia.ClipOp.kIntersect, True)
            if outside:
                ytop = max(-10.0, _lr_y(-300, LR['top_h']))
                ybot = _lr_y(-300, LR['sill_h']) + 10
                _draw_lowres(c, (-10, ytop, W + 10, min(H, ybot)), 0.5,
                             lambda c2: _lr_outside(c2, t, dawn, sunrise, view_offset * 0.25, ybot))
            else:
                c.drawRect(skia.Rect.MakeLTRB(-10, -10, W + 10, H + 10), fill('#0b1230'))
            _lr_glass(c, t, lamp, lxs, ly, lamp_color)
        # ---------------- interior (albedo layer x light map) ----------------
        c.saveLayer(None, None)
        with saved(c):
            c.translate(ox, 0)
            _lr_ceiling(c, hatch)
            _lr_frames(c)
            _lr_walls(c)
            _lr_floor(c)
            _lr_pedestal(c, t, lever)
            _lr_wheel(c, wheel_rot)
        env_amb = _vmix(_vmix((0.24, 0.28, 0.46), (0.36, 0.30, 0.46), dawn), (0.62, 0.52, 0.48), sunrise)
        env_win = _vmix(_vmix((0.16, 0.20, 0.34), (0.30, 0.24, 0.30), dawn), (0.40, 0.32, 0.22), sunrise)
        lp = skia.Paint(Shader=_lr_light_shader().make(
            lamp=lamp, lc=(lxs, ly), amb=env_amb, win=env_win, warm=_vmix(_rgb(lamp_color), (1.0, 0.9, 0.75), 0.35),
            rot=lens_rot, up=pu, hy=g['hy'], fz=g['f'], sweep=beam_sweep))
        _draw_lowres(c, (-8, -8, W + 8, H + 8), 0.125, lambda c2: c2.drawPaint(lp), blend=skia.BlendMode.kModulate)
        c.restore()
        # hatch opening sky (emissive, not multiplied)
        if hatch > 0.01:
            _lr_hatch_sky(c, t, hatch, ox, lamp, pu, lamp_color)
        # ---------------- the lens (drawn lit, not multiplied) ----------------
        with saved(c):
            c.translate(ox, 0)
            _lr_yoke(c, lamp)
            _lr_lens(c, t, lamp, lens_rot, pu, lamp_color)
            _lr_trunnions(c, lamp)
        # ---------------- additive light ----------------
        if lamp > 0:
            _lr_glow(c, t, lamp, lens_rot, pu, lxs, ly, lamp_color, dust)
    ly2 = ly - 150 * math.sin(pu * math.pi / 2 * 0.95) * 0.0
    return lxs, ly2, lr


def _lr_glazing_path(ox):
    top = _lr_curve(LR['top_h'])
    bot = _lr_curve(LR['sill_h'])
    pts = [(x + ox, y) for x, y in top] + [(x + ox, y) for x, y in bot[::-1]]
    return _path_from(pts, close=True)


def _lr_outside(c, t, dawn, sunrise, off, ybot=1100):
    hy = LR['hy']
    c.save()
    c.translate(-off, 0)
    ok = False
    try:
        from lib import sky as _sky
        _sky.sky(c, rect=(-400, -400, 2800, hy + 402), horizon_y=hy, dawn=dawn, sunrise=sunrise)
        _sky.stars(c, t, rect=(-400, -400, 2800, hy + 400), horizon_y=hy, bright=1 - sunrise)
        ok = True
    except Exception:
        ok = False
    if not ok:
        c.drawRect(skia.Rect.MakeLTRB(-400, -400, 2400, hy + 2),
                   linear((0, hy - 700), (0, hy), [(0, '#070b1f', 1), (0.6, '#16224d', 1), (1, '#2c3f7a', 1)]))
        r = nprng(77)
        for i in range(160):
            sx, sy = r.uniform(-300, 2300), r.uniform(-300, hy - 20)
            a = 0.35 + 0.35 * math.sin(t * 2 + i)
            c.drawCircle(sx, sy, r.uniform(0.8, 2.0), fill('#fff6e0', a * clamp((hy - sy) / 200)))
    ocean(c, t, horizon_y=hy, rect_x=(-400, 2400), bottom=ybot, dawn=dawn, sunrise=sunrise, calm=0.7,
          glints=0.8, focal=700.0)
    c.restore()


def _lr_glass(c, t, lamp, lx, ly, lamp_color):
    # faint glass tint & diagonal sheen
    c.drawRect(skia.Rect.MakeLTRB(-10, -10, W + 10, H + 10), fill('#1a2a50', 0.18))
    for (x0, w, a) in ((260, 90, 0.05), (420, 30, 0.04), (1480, 120, 0.04), (1660, 40, 0.035)):
        p = skia.Path()
        p.moveTo(x0, -20)
        p.lineTo(x0 + w, -20)
        p.lineTo(x0 + w - 260, 900)
        p.lineTo(x0 - 260, 900)
        p.close()
        c.drawPath(p, fill('#bcd4ff', a, blend=ADD))
    if lamp > 0:
        # reflection of the glowing lens in the curved glass (left & right panes)
        for (rx, sc) in ((lx - 560, 0.55), (lx + 600, 0.5)):
            rp = radial((rx, ly - 10), 150 * sc * 1.6, [(0, lamp_color, 0.30 * lamp), (1, lamp_color, 0.0)])
            rp.setBlendMode(ADD)
            with saved(c):
                c.translate(rx, ly - 10)
                c.scale(0.45, 1.0)
                c.translate(-rx, -(ly - 10))
                c.drawCircle(rx, ly - 10, 150 * sc * 1.6, rp)


def _lr_frames(c):
    """Window frames: mullions, astragal, head & sill rings (albedo; lit by the light map)."""
    iron = '#3a3f58'
    brass = '#d6ad62'
    # head ring (thick) and sill ring
    for (h, th, colr) in ((LR['top_h'] + 6, 30, iron), (LR['sill_h'] - 2, 22, iron)):
        a = _lr_curve(h + th / 2)
        b = _lr_curve(h - th / 2)
        c.drawPath(_path_from(a + b[::-1], close=True), fill(colr))
        c.drawPath(_path_from(_lr_curve(h - th / 2 + 2)), stroke(brass, 3.0))
        c.drawPath(_path_from(_lr_curve(h + th / 2 - 2)), stroke('#6a6f8a', 2.0))
    # mid astragal
    c.drawPath(_path_from(_lr_curve(92)), stroke(iron, 7.0))
    c.drawPath(_path_from(_lr_curve(95)), stroke(brass, 1.6))
    # mullions (curved glazing bars)
    for (mx, ca) in _lr_mullion_xs():
        w = 7 + 16 * ca
        y0 = _lr_y(mx, LR['top_h'])
        y1 = _lr_y(mx, LR['sill_h'])
        c.drawRect(skia.Rect.MakeLTRB(mx - w / 2, y0, mx + w / 2, y1), fill(iron))
        c.drawRect(skia.Rect.MakeLTRB(mx - w / 2, y0, mx - w / 2 + 2.5, y1), fill(brass))
        c.drawRect(skia.Rect.MakeLTRB(mx + w / 2 - 3, y0, mx + w / 2, y1), fill('#1c2030'))
        for h in (LR['top_h'] - 12, 92, LR['sill_h'] + 10):
            c.drawCircle(mx, _lr_y(mx, h), 2.6 + 1.4 * ca, fill(brass))


def _lr_walls(c):
    """Lower wainscot wall between sill and floor, with panels and a brass handrail."""
    top = _lr_curve(LR['sill_h'] - 13)
    bot = _lr_curve(LR['floor_h'])
    c.drawPath(_path_from(top + bot[::-1], close=True),
               linear((0, 700), (0, 880), [(0, '#6e4834', 1), (1, '#4a2e22', 1)]))
    # panel dividers aligned with mullions
    for (mx, ca) in _lr_mullion_xs():
        y0 = _lr_y(mx, LR['sill_h'] - 16)
        y1 = _lr_y(mx, LR['floor_h'])
        w = 3 + 6 * ca
        c.drawRect(skia.Rect.MakeLTRB(mx - w / 2, y0, mx + w / 2, y1), fill('#3a2418'))
        c.drawRect(skia.Rect.MakeLTRB(mx - w / 2, y0, mx - w / 2 + 1.5, y1), fill('#8a6040'))
    # raised panel mouldings
    c.drawPath(_path_from(_lr_curve(LR['sill_h'] - 40)), stroke('#8a5e40', 2.0))
    c.drawPath(_path_from(_lr_curve(LR['floor_h'] + 22)), stroke('#2e1c14', 3.0))
    # brass handrail on posts
    rail = _lr_curve(LR['sill_h'] - 58)
    c.drawPath(_path_from(rail), stroke('#b88c48', 7.0))
    c.drawPath(_path_from([(x, y - 2) for x, y in rail]), stroke('#f0cc80', 2.0))
    for (mx, ca) in _lr_mullion_xs():
        c.drawCircle(mx, _lr_y(mx, LR['sill_h'] - 58), 4.5, fill('#e0b870'))
    # skirting
    sk_t = _lr_curve(LR['floor_h'] + 12)
    c.drawPath(_path_from(sk_t + bot[::-1], close=True), fill('#2a1a12'))


def _lr_floor(c):
    fy = _lr_curve(LR['floor_h'])
    path = _path_from(fy + [(2220, 1200), (-300, 1200)], close=True)
    c.drawPath(path, linear((0, 800), (0, 1080), [(0, '#6a4630', 1), (1, '#9a6a44', 1)]))
    with saved_clip(c, path):
        vx, vy = 960.0, LR['hy']
        r = nprng(55)
        # boards converging to the vanishing point
        for k in range(-26, 27):
            xb = 960 + k * 118
            c.drawLine(vx + (xb - vx) * 0.12, vy + (1180 - vy) * 0.12, xb, 1180, stroke('#3a2416', 2.2, 0.8))
            tone = r.uniform(-0.035, 0.035)
            if abs(tone) > 0.015:
                p = poly([(vx + (xb - vx) * 0.12, vy + (1180 - vy) * 0.12), (vx + (xb + 118 - vx) * 0.12, vy + (1180 - vy) * 0.12),
                          (xb + 118, 1180), (xb, 1180)])
                c.drawPath(p, fill('#ffffff' if tone > 0 else '#000000', abs(tone)))
            # butt joints
            for j in range(3):
                tt = r.uniform(0.3, 0.95)
                yy = vy + (1180 - vy) * tt
                xa = vx + (xb - vx) * tt
                xb2 = vx + (xb + 118 - vx) * tt
                c.drawLine(xa, yy, xb2, yy, stroke('#3a2416', 1.4, 0.6))
        # grain sheen
        c.drawRect(skia.Rect.MakeLTRB(-300, 790, 2220, 1180), linear((0, 790), (0, 900), [(0, '#000000', 0.35), (1, '#000000', 0.0)]))


def _lr_ceiling(c, hatch):
    top = _lr_curve(LR['top_h'] + 20)
    path = _path_from([(-300, -50), (2220, -50)] + top[::-1], close=True)
    c.drawPath(path, linear((0, -50), (0, 200), [(0, '#241c26', 1), (1, '#3e3040', 1)]))
    hx, hy, hrx, hry = LR['hatch']
    with saved_clip(c, path):
        # radial ribs converging on the hatch
        for k in range(-9, 10):
            xe = 960 + k * 150
            ye = _lr_y(xe, LR['top_h'] + 20)
            xs = hx + k * hrx / 9.5
            ys = hy + hry * 0.2
            c.drawLine(xs, ys, xe, ye + 4, stroke('#1a141c', 9.0))
            c.drawLine(xs - 3, ys, xe - 3, ye + 4, stroke('#6a5060', 2.0))
        # hatch collar
        c.drawOval(skia.Rect.MakeLTRB(hx - hrx - 26, hy - hry - 12, hx + hrx + 26, hy + hry + 12), fill('#2a2230'))
        c.drawOval(skia.Rect.MakeLTRB(hx - hrx - 26, hy - hry - 12, hx + hrx + 26, hy + hry + 12), stroke('#d6ad62', 3.0))
        if hatch < 0.999:
            # closed / partly closed doors
            c.drawOval(skia.Rect.MakeLTRB(hx - hrx, hy - hry, hx + hrx, hy + hry), fill('#3a2e3a'))
            for k in range(-4, 5):
                c.drawCircle(hx + k * hrx / 5, hy + hry * 0.1 * math.cos(k / 4), 2.4, fill('#c9a15b'))
            c.drawLine(hx, hy - hry, hx, hy + hry, stroke('#16101a', 3.0))


def _lr_hatch_sky(c, t, hatch, ox, lamp, pu, lamp_color):
    hx, hy, hrx, hry = LR['hatch']
    hx += ox
    o = ease_io(hatch)
    rect = skia.Rect.MakeLTRB(hx - hrx, hy - hry, hx + hrx, hy + hry)
    op = skia.Path()
    op.addOval(rect)
    with saved_clip(c, op):
        c.drawRect(rect, linear((0, hy - hry), (0, hy + hry), [(0, '#060a1c', 1), (1, '#1a2550', 1)]))
        r = nprng(91)
        for i in range(40):
            sx, sy = hx + r.uniform(-hrx, hrx), hy + r.uniform(-hry, hry)
            c.drawCircle(sx, sy, r.uniform(0.7, 1.8), fill('#fff6e0', 0.5 + 0.4 * math.sin(t * 2.3 + i)))
        # the two doors swinging up (seen edge-on as they rise)
        for side in (-1, 1):
            w = hrx * (1 - o)
            if w > 1:
                x0 = hx + side * hrx - (side * w if side > 0 else -w) * 0
                xa, xb = (hx - hrx, hx - hrx + w) if side < 0 else (hx + hrx - w, hx + hrx)
                c.drawRect(skia.Rect.MakeLTRB(xa, hy - hry, xb, hy + hry), fill('#2e2430'))
                del x0
    # doors standing up at the rim
    for side in (-1, 1):
        k = o
        xh = hx + side * hrx
        top = hy - hry - 70 * k
        xt = xh - side * hrx * (1 - k) * 0.9
        c.drawPath(poly([(xh, hy + 4), (xh - side * 2, hy - 10), (xt, top), (xt + side * 6, top + 6)]),
                   fill('#2a2230', 1.0))
    if lamp > 0 and pu > 0.3:
        k = lamp * smoothstep(0.3, 1.0, pu) * o
        bp = linear((0, hy + 380), (0, -200), [(0, lamp_color, 0.0), (0.4, '#fff4d0', 0.55 * k), (1, lamp_color, 0.25 * k)])
        bp.setBlendMode(ADD)
        c.drawPath(poly([(hx - 120, 420), (hx + 120, 420), (hx + hrx * 0.95, hy), (hx + hrx * 1.1, -60),
                         (hx - hrx * 1.1, -60), (hx - hrx * 0.95, hy)]), bp)


# ---------------------------------------------------------------------------- lens (3D)
_LENS_N = 6
_LENS_RB = 175.0
_LENS_MODEL = {}


def _lens_model():
    if not _LENS_MODEL:
        rings = []
        for yl in np.arange(131.0, 232.0, 11.0):
            rings.append((yl, _LENS_RB * math.cos((yl - 120.0) / 118.0 * 1.02), 'up'))
        for yl in np.arange(-131.0, -204.0, -11.0):
            rings.append((yl, _LENS_RB - (abs(yl) - 120.0) * 0.72, 'lo'))
        _LENS_MODEL['rings'] = rings
        th = np.linspace(0, math.tau, 73)
        _LENS_MODEL['th'] = th
        _LENS_MODEL['bull'] = [118.0 * math.sqrt(j / 9.0) for j in range(1, 10)]
    return _LENS_MODEL


def _lens_xf(P, rot, beta, cx, cy):
    """Rotate local lens points (N,3) about Y by rot, tilt about X by beta, project."""
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    cr, sr = math.cos(rot), math.sin(rot)
    x1 = x * cr + z * sr
    z1 = -x * sr + z * cr
    cb, sb = math.cos(beta), math.sin(beta)
    y2 = y * cb + z1 * sb
    z2 = -y * sb + z1 * cb
    ps = 1.0 / (1.0 - z2 / 2600.0)
    return np.stack([cx + x1 * ps, cy - y2 * ps, z2], axis=1)


def _lens_dir(v, rot, beta):
    return _lens_xf(np.array([v], float), rot, beta, 0.0, 0.0)[0]


def _segments(pts, mask):
    """Split projected points into runs where mask is True -> list of arrays."""
    runs, cur = [], []
    for p, m in zip(pts, mask):
        if m:
            cur.append(p)
        elif cur:
            if len(cur) > 1:
                runs.append(cur)
            cur = []
    if len(cur) > 1:
        runs.append(cur)
    return runs


def _hull(points):
    pts = sorted(set((round(float(a), 2), round(float(b), 2)) for a, b in points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _lr_lens(c, t, lamp, rot, pu, lamp_color):
    m = _lens_model()
    cx, cy, _ = LR['lens']
    beta = pu * math.radians(62)
    th = m['th']
    N = _LENS_N
    lit = lamp
    lc = _rgb(lamp_color)
    # ---- all rings (prisms + belt edges) projected
    ring_defs = [(yl, r, kind) for (yl, r, kind) in m['rings']] + [(120.0, _LENS_RB, 'belt'), (-120.0, _LENS_RB, 'belt')]
    proj = []
    allpts = []
    for (yl, r, kind) in ring_defs:
        P = np.stack([r * np.sin(th), np.full_like(th, yl), r * np.cos(th)], axis=1)
        Q = _lens_xf(P, rot, beta, cx, cy)
        Nn = _lens_xf(np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], axis=1), rot, beta, 0, 0)
        proj.append((yl, r, kind, Q, Nn[:, 2]))
        allpts += [(q[0], q[1]) for q in Q[::3]]
    hull = _hull(allpts)
    hpath = _path_from(hull, close=True)
    # ---- glass body
    if lit > 0:
        c.drawPath(hpath, radial((cx, cy), 260, [(0, '#ffdc90', 0.40 * lit), (0.5, _vmix(lc, (0.8, 0.45, 0.15), 0.4), 0.32 * lit),
                                                (1, _vmix(lc, (0.5, 0.2, 0.1), 0.7), 0.30 * lit)]))
    c.drawPath(hpath, linear((cx - 200, 0), (cx + 200, 0), [(0, '#9fb8e8', 0.20), (0.35, '#5a6c9a', 0.14), (1, '#2a3050', 0.22)]))
    # ---- back halves of rings (seen through the glass)
    bcol = _vmix((0.62, 0.72, 0.95), lc, lit)
    for (yl, r, kind, Q, nz) in proj:
        for run in _segments([(q[0], q[1]) for q in Q], nz < 0):
            c.drawPath(_path_from(run), stroke(bcol, 1.2, 0.18 + 0.2 * lit))
    # ---- burner / lamp source at the centre
    _lens_burner(c, t, cx, cy, lit, lamp_color, beta)
    # ---- bullseye panels
    order = []
    for k in range(N):
        phi = k * math.tau / N
        n = _lens_dir((math.sin(phi), 0.0, math.cos(phi)), rot, beta)
        order.append((n[2], k, phi))
    order.sort()
    ap = _LENS_RB * math.cos(math.pi / N)
    hw = _LENS_RB * math.sin(math.pi / N)
    for nzk, k, phi in order:
        if nzk <= 0.02:
            continue
        cpt = np.array([ap * math.sin(phi), 0.0, ap * math.cos(phi)])
        tv = np.array([math.cos(phi), 0.0, -math.sin(phi)])
        up = np.array([0.0, 1.0, 0.0])
        quad = np.array([cpt - hw * tv - 118 * up, cpt + hw * tv - 118 * up, cpt + hw * tv + 118 * up, cpt - hw * tv + 118 * up])
        Qq = _lens_xf(quad, rot, beta, cx, cy)
        qpath = _path_from([(q[0], q[1]) for q in Qq], close=True)
        face = nzk
        blaze = lit * face ** 3
        with saved_clip(c, qpath):
            # panel glass
            c.drawPath(qpath, fill(_vmix((0.55, 0.66, 0.9), lc, lit), 0.06 + 0.10 * face))
            for j, rho in enumerate(m['bull']):
                circ = cpt[None, :] + rho * (np.cos(th)[:, None] * tv[None, :] + np.sin(th)[:, None] * up[None, :])
                Qc = _lens_xf(circ, rot, beta, cx, cy)
                path = _path_from([(q[0], q[1]) for q in Qc], close=True)
                w = 2.6 * (0.4 + 0.6 * face)
                if lit > 0:
                    c.drawPath(path, stroke(_vmix(lc, (0.55, 0.28, 0.08), 0.55), w * 1.5, clamp(0.35 * lit)))
                    c.drawPath(path, stroke(_vmix(lc, (1, 0.97, 0.9), 0.25 * face), w * 0.8, clamp(0.25 + 0.5 * lit * (0.35 + 0.65 * face))))
                    hp = stroke('#fffaf0', w * 0.35, clamp(blaze * 0.35))
                    hp.setBlendMode(ADD)
                    c.drawPath(path, hp)
                else:
                    c.drawPath(path, stroke('#0a0f22', w * 1.6, 0.45))
                    c.drawPath(path, stroke('#8aa2d2', w * 0.9, 0.30))
                    c.drawPath(path, stroke('#dfeaff', w * 0.4, 0.35 + 0.3 * face))
            # central bullseye disc
            cen = _lens_xf(cpt[None, :], rot, beta, cx, cy)[0]
            rr = m['bull'][0] * 1.05
            if lit > 0:
                gp = radial((cen[0], cen[1]), rr * 1.6, [(0, '#ffffff', clamp(0.15 + 0.45 * blaze)),
                                                         (0.5, '#fff0c0', clamp(0.10 + 0.3 * blaze)), (1, lamp_color, 0.0)])
                gp.setBlendMode(ADD)
                c.drawCircle(cen[0], cen[1], rr * 1.8, gp)
            else:
                c.drawCircle(cen[0] - rr * 0.3, cen[1] - rr * 0.35, rr * 0.35, fill('#dfe9ff', 0.25 * face))
    # ---- front halves of prism rings
    for (yl, r, kind, Q, nz) in proj:
        for run in _segments([(q[0], q[1]) for q in Q], nz >= 0):
            if kind == 'belt':
                continue
            if lit > 0:
                c.drawPath(_path_from([(a, b + 2) for a, b in run]), stroke(_vmix(lc, (0.5, 0.25, 0.1), 0.6), 3.0, 0.5 * lit))
                c.drawPath(_path_from(run), stroke(_vmix(lc, (1, 0.95, 0.85), 0.35), 3.4, clamp(0.35 + 0.45 * lit)))
                hp = stroke('#fff8e0', 1.1, clamp(0.35 * lit))
                hp.setBlendMode(ADD)
                c.drawPath(_path_from([(a, b - 1) for a, b in run]), hp)
            else:
                c.drawPath(_path_from([(a, b + 2) for a, b in run]), stroke('#0a0f20', 4.0, 0.5))
                c.drawPath(_path_from(run), stroke('#7f98c8', 3.0, 0.35))
                c.drawPath(_path_from([(a, b - 1.4) for a, b in run]), stroke('#d4e4ff', 1.1, 0.55))
    # ---- brass frame: vertical bars between panels + horizontal rings
    brass = _vmix((0.62, 0.50, 0.30), (1.0, 0.80, 0.45), lit)
    brass_hi = _vmix((0.75, 0.82, 0.95), (1.0, 0.95, 0.75), lit)
    prof = [(yl, r) for (yl, r, kind) in sorted(ring_defs, key=lambda q: q[0])]
    for k in range(N):
        phi = (k + 0.5) * math.tau / N
        P = np.array([[r * math.sin(phi), yl, r * math.cos(phi)] for yl, r in prof])
        Q = _lens_xf(P, rot, beta, cx, cy)
        n = _lens_dir((math.sin(phi), 0.0, math.cos(phi)), rot, beta)
        if n[2] < -0.05:
            c.drawPath(_path_from([(q[0], q[1]) for q in Q]), stroke(brass, 3.0, 0.25))
            continue
        w = 5.0 + 5.0 * n[2]
        c.drawPath(_path_from([(q[0], q[1]) for q in Q]), stroke(_vscale(brass, 0.7), w + 2))
        c.drawPath(_path_from([(q[0], q[1]) for q in Q]), stroke(brass, w))
        c.drawPath(_path_from([(q[0] - w * 0.25, q[1]) for q in Q]), stroke(brass_hi, w * 0.28, 0.8))
    for (yl, r, th_w) in ((120.0, _LENS_RB + 2, 9.0), (-120.0, _LENS_RB + 2, 9.0), (232.0, 92.0, 10.0),
                          (-205.0, 122.0, 12.0)):
        P = np.stack([r * np.sin(th), np.full_like(th, yl), r * np.cos(th)], axis=1)
        Q = _lens_xf(P, rot, beta, cx, cy)
        nz = _lens_xf(np.stack([np.sin(th), np.zeros_like(th), np.cos(th)], axis=1), rot, beta, 0, 0)[:, 2]
        for run in _segments([(q[0], q[1]) for q in Q], nz >= 0):
            c.drawPath(_path_from(run), stroke(_vscale(brass, 0.65), th_w + 2))
            c.drawPath(_path_from(run), stroke(brass, th_w))
            c.drawPath(_path_from([(a, b - th_w * 0.25) for a, b in run]), stroke(brass_hi, th_w * 0.3, 0.8))
    # ---- crown finial
    top = _lens_xf(np.array([[0.0, 245.0, 0.0], [0.0, 275.0, 0.0]]), rot, beta, cx, cy)
    c.drawLine(top[0][0], top[0][1], top[1][0], top[1][1], stroke(brass, 14))
    c.drawCircle(top[1][0], top[1][1], 11, fill(brass))
    c.drawCircle(top[1][0] - 3, top[1][1] - 3, 4, fill(brass_hi, 0.8))


def _lens_burner(c, t, cx, cy, lit, lamp_color, beta):
    # brass burner column
    c.drawRRect(rrect(cx - 16, cy + 18, 32, 90, 6), fill(_vmix((0.35, 0.30, 0.25), (0.85, 0.62, 0.30), lit)))
    c.drawRRect(rrect(cx - 26, cy + 10, 52, 12, 5), fill(_vmix((0.45, 0.38, 0.30), (0.95, 0.72, 0.38), lit)))
    if lit > 0:
        fl = 1.0 + 0.04 * noise1(t * 9.0, 2) + 0.03 * noise1(t * 23.0, 4)
        k = lit * fl
        c.drawOval(skia.Rect.MakeLTRB(cx - 16, cy - 32, cx + 16, cy + 12), fill('#fff2c8', clamp(k)))
        gp = radial((cx, cy - 10), 55, [(0, '#ffffff', clamp(0.7 * k)), (0.3, '#fff1c0', clamp(0.35 * k)), (1, lamp_color, 0.0)])
        gp.setBlendMode(ADD)
        c.drawCircle(cx, cy - 8, 70, gp)
    else:
        c.drawOval(skia.Rect.MakeLTRB(cx - 18, cy - 30, cx + 18, cy + 12), fill('#6a6e80', 0.55))
        c.drawOval(skia.Rect.MakeLTRB(cx - 10, cy - 24, cx - 2, cy - 10), fill('#c8d6f0', 0.35))


def _lr_pedestal(c, t, lever):
    """Cast-iron pedestal with brass bands + ignition lever (albedo layer)."""
    cx = 960.0
    iron = '#454c68'
    iron_d = '#262a3c'
    brass = '#d6ad62'
    # base plinth on the floor
    c.drawOval(skia.Rect.MakeLTRB(cx - 230, 818, cx + 230, 884), fill('#1a1420', 0.6))
    c.drawPath(_path_from([(cx - 200, 850), (cx - 190, 812), (cx + 190, 812), (cx + 200, 850)] +
                          [(cx + 200 * math.cos(a), 850 + 24 * math.sin(a)) for a in np.linspace(0, math.pi, 20)], close=True),
               linear((cx - 200, 0), (cx + 200, 0), [(0, iron, 1), (0.35, '#6a7294', 1), (1, iron_d, 1)]))
    c.drawOval(skia.Rect.MakeLTRB(cx - 190, 800, cx + 190, 824), fill('#5a6282'))
    # fluted column
    col_path = _path_from([(cx - 92, 812), (cx - 72, 712), (cx + 72, 712), (cx + 92, 812)], close=True)
    c.drawPath(col_path, linear((cx - 92, 0), (cx + 92, 0), [(0, iron, 1), (0.3, '#707a9e', 1), (0.65, iron, 1), (1, iron_d, 1)]))
    with saved_clip(c, col_path):
        for k in range(-5, 6):
            u = k / 5.5
            c.drawLine(cx + u * 72, 712, cx + u * 92, 812, stroke(iron_d, 3.0, 0.6))
            c.drawLine(cx + u * 72 - 3, 712, cx + u * 92 - 3, 812, stroke('#8a94b8', 1.2, 0.35))
    # brass bands
    for (yy, rx, th) in ((716, 80, 10), (806, 96, 9)):
        c.drawOval(skia.Rect.MakeLTRB(cx - rx, yy - th, cx + rx, yy + th), fill(brass))
        c.drawOval(skia.Rect.MakeLTRB(cx - rx + 6, yy - th + 2, cx + rx - 6, yy + th - 5), fill('#f0d090', 0.5))
    # top turntable plate
    c.drawOval(skia.Rect.MakeLTRB(cx - 170, 682, cx + 170, 722), fill(iron_d))
    c.drawOval(skia.Rect.MakeLTRB(cx - 170, 676, cx + 170, 712), linear((cx - 170, 0), (cx + 170, 0), [(0, '#5a6282', 1), (0.4, '#8a92b0', 1), (1, iron_d, 1)]))
    c.drawOval(skia.Rect.MakeLTRB(cx - 170, 676, cx + 170, 712), stroke(brass, 3))
    for k in range(12):
        a = k / 12 * math.tau
        c.drawCircle(cx + 158 * math.sin(a), 694 + 14 * math.cos(a), 3, fill(brass))
    # clockwork box on the left with the ignition lever
    bx, by = LR['lever']
    c.drawRRect(rrect(bx - 58, by - 44, 76, 80, 8), linear((bx - 58, 0), (bx + 18, 0), [(0, '#5a6282', 1), (1, iron_d, 1)]))
    c.drawRRect(rrect(bx - 58, by - 44, 76, 80, 8), stroke(brass, 3))
    c.drawCircle(bx - 20, by - 4, 18, fill('#e8dcc0'))
    c.drawCircle(bx - 20, by - 4, 18, stroke(brass, 3))
    a = -0.6 + t * 0.0
    c.drawLine(bx - 20, by - 4, bx - 20 + 13 * math.cos(a), by - 4 + 13 * math.sin(a), stroke('#2a2020', 2))
    ang = lerp(-2.0, -3.7, ease_io(lever))
    L = 96.0
    ex, ey = bx + L * math.cos(ang), by + L * math.sin(ang)
    c.drawLine(bx, by, ex, ey, stroke('#b08440', 9))
    c.drawLine(bx, by, ex, ey, stroke('#f0cc80', 3, 0.8))
    c.drawCircle(ex, ey, 13, fill('#8a2a26'))
    c.drawCircle(ex - 4, ey - 4, 4, fill('#ff9a80', 0.7))
    c.drawCircle(bx, by, 10, fill(brass))


def _lr_yoke(c, lamp):
    """Iron U-yoke carrying the lens on trunnions (drawn over the albedo layer, lit directly)."""
    cx, cy, _ = LR['lens']
    k = lamp
    iron = _vmix((0.20, 0.23, 0.36), (0.48, 0.36, 0.28), k)
    hi = _vmix((0.45, 0.52, 0.75), (1.0, 0.78, 0.45), k)
    for side in (-1, 1):
        x0, y0 = cx + side * 140, 696
        x1, y1 = cx + side * 214, cy
        p = skia.Path()
        p.moveTo(x0, y0)
        p.cubicTo(x0 + side * 70, y0 - 40, x1 + side * 20, y1 + 120, x1, y1)
        c.drawPath(p, stroke(_vscale(iron, 0.6), 30))
        c.drawPath(p, stroke(iron, 24))
        p2 = skia.Path()
        p2.moveTo(x0 - side * 6, y0)
        p2.cubicTo(x0 + side * 64, y0 - 40, x1 + side * 14, y1 + 120, x1 - side * 6, y1)
        c.drawPath(p2, stroke(hi, 3.5, 0.55))


def _lr_trunnions(c, lamp):
    cx, cy, _ = LR['lens']
    k = lamp
    brass = _vmix((0.62, 0.50, 0.30), (1.0, 0.80, 0.45), k)
    for side in (-1, 1):
        x = cx + side * 214
        c.drawCircle(x, cy, 22, fill(_vscale(brass, 0.6)))
        c.drawCircle(x, cy, 17, fill(brass))
        c.drawCircle(x - 5, cy - 5, 6, fill('#fff0c8', 0.6))


def _gear(c, x, y, r, teeth, rot, colr, hub_col, tooth=None):
    tooth = tooth or r * 0.16
    pts = []
    n = teeth * 4
    for i in range(n):
        a = rot + i / n * math.tau
        phase = i % 4
        rr = r + (tooth if phase in (1, 2) else 0.0)
        pts.append((x + rr * math.cos(a), y + rr * math.sin(a)))
    c.drawPath(_path_from(pts, close=True), fill(colr))
    c.drawCircle(x, y, r * 0.78, fill(_vscale(_rgb(colr), 0.75)))
    for k in range(5):
        a = rot + k / 5 * math.tau
        c.drawCircle(x + r * 0.48 * math.cos(a), y + r * 0.48 * math.sin(a), r * 0.16, fill(_vscale(_rgb(colr), 0.45)))
    c.drawCircle(x, y, r * 0.22, fill(hub_col))


def _lr_wheel(c, wheel_rot):
    """Iron hand-wheel on a stand with a gear train up to the lens trunnion (albedo layer)."""
    cx, cy, _ = LR['lens']
    wx, wy, wr = LR['wheel']
    iron = '#4a5070'
    brass = '#d6ad62'
    # gear train: trunnion gear (r70) <- idler (r52) <- pinion on wheel axle (r30)
    tg = (cx + 214.0, cy)
    ig = (tg[0] + 122 * math.cos(math.radians(40)), tg[1] + 122 * math.sin(math.radians(40)))
    pg = (wx, wy)
    # stand
    c.drawPath(poly([(wx - 12, wy), (wx + 12, wy), (wx + 70, 905), (wx + 44, 905)]), fill('#30364e'))
    c.drawPath(poly([(wx - 12, wy), (wx + 12, wy), (wx - 44, 905), (wx - 70, 905)]), fill('#3a4060'))
    c.drawOval(skia.Rect.MakeLTRB(wx - 90, 895, wx + 90, 915), fill('#1a1420', 0.5))
    # support bracket to the idler
    c.drawLine(ig[0], ig[1], wx + 20, wy + 90, stroke('#30364e', 14))
    _gear(c, tg[0], tg[1], 70, 24, wheel_rot * 30.0 / 70.0, '#6a6070', brass)
    _gear(c, ig[0], ig[1], 52, 18, -wheel_rot * 30.0 / 52.0 + 0.09, '#5a5a70', brass)
    _gear(c, pg[0], pg[1], 30, 11, wheel_rot, '#7a6a50', brass)
    # the wheel: rim, spokes, hub, handle
    for k in range(6):
        a = wheel_rot + k / 6 * math.tau
        x1, y1 = wx + (wr - 8) * math.cos(a), wy + (wr - 8) * math.sin(a)
        p = skia.Path()
        p.moveTo(wx, wy)
        p.quadTo(wx + (wr * 0.5) * math.cos(a + 0.25), wy + (wr * 0.5) * math.sin(a + 0.25), x1, y1)
        c.drawPath(p, stroke('#262a3c', 11))
        c.drawPath(p, stroke(iron, 7))
    c.drawCircle(wx, wy, wr, stroke('#262a3c', 20))
    c.drawCircle(wx, wy, wr, stroke(iron, 15))
    c.drawArc(skia.Rect.MakeLTRB(wx - wr, wy - wr, wx + wr, wy + wr), 180, 110, False, stroke('#9aa2c0', 4, 0.8))
    for k in range(12):
        a = wheel_rot + (k + 0.5) / 12 * math.tau
        c.drawCircle(wx + wr * math.cos(a), wy + wr * math.sin(a), 3.2, fill(brass))
    c.drawCircle(wx, wy, 22, fill(brass))
    c.drawCircle(wx, wy, 22, stroke('#8a6a30', 3))
    c.drawCircle(wx - 6, wy - 6, 7, fill('#fff0c0', 0.7))
    ha = wheel_rot + 0.6
    hx, hy = wx + wr * math.cos(ha), wy + wr * math.sin(ha)
    c.drawCircle(hx, hy, 14, fill('#6a3a28'))
    c.drawCircle(hx - 4, hy - 4, 5, fill('#c08060', 0.8))


def _lr_glow(c, t, lamp, rot, pu, lx, ly, lamp_color, dust):
    k = lamp
    gp = radial((lx, ly), 1100, [(0, '#fff0c0', 0.16 * k), (0.06, '#fff0c0', 0.12 * k), (0.2, lamp_color, 0.08 * k),
                                 (0.5, lamp_color, 0.035 * k), (1, lamp_color, 0.0)])
    gp.setBlendMode(ADD)
    c.drawRect(skia.Rect.MakeLTRB(max(-10, lx - 1100), -10, min(W + 10, lx + 1100), H + 10), gp)
    beta = pu * math.radians(62)
    # beams from the bullseye panels sweeping across the room
    N = _LENS_N
    for i in range(N):
        phi = i * math.tau / N
        n = _lens_dir((math.sin(phi), 0.0, math.cos(phi)), rot, beta)
        side = math.hypot(n[0], n[1])
        if side < 0.08:
            continue
        dx, dy = n[0] / side, -n[1] / side
        L = 1500 * side
        a = 0.10 * k * (1 - 0.6 * abs(n[2])) * (1 if n[2] > -0.3 else 0.5) * (1 - 0.8 * pu)
        px, py = -dy, dx
        w0, w1 = 70, 70 + 520 * side
        x0, y0 = lx + dx * 120 * side, ly + dy * 120 * side
        x1, y1 = x0 + dx * L, y0 + dy * L
        bp = linear((x0, y0), (x1, y1), [(0, '#fff4d0', a), (0.5, lamp_color, a * 0.45), (1, lamp_color, 0.0)])
        bp.setBlendMode(ADD)
        c.drawPath(poly([(x0 + px * w0, y0 + py * w0), (x1 + px * w1, y1 + py * w1), (x1 - px * w1, y1 - py * w1),
                         (x0 - px * w0, y0 - py * w0)]), bp)
    # dust motes drifting in the light
    if dust > 0:
        r = nprng(313)
        n = 70
        bx = r.uniform(-700, 700, n)
        by = r.uniform(-420, 420, n)
        sp = r.uniform(4, 14, n)
        ph = r.uniform(0, math.tau, n)
        sz = r.uniform(1.0, 2.6, n)
        p = fill('#ffe9b8', 1.0, blend=ADD)
        for i in range(n):
            x = lx + bx[i] + math.sin(t * 0.3 + ph[i]) * 30
            y = ly + ((by[i] - t * sp[i] + 420) % 840) - 420
            d = math.hypot(x - lx, y - ly)
            a = k * dust * 0.55 * clamp(1 - d / 800) * (0.5 + 0.5 * math.sin(t * 1.7 + ph[i]))
            if a > 0.02:
                p.setAlphaf(a)
                c.drawCircle(x, y, sz[i], p)


# ============================================================================
# SPIRAL STAIRCASE (interior, full screen)
# ============================================================================
ST = dict(
    f=2400.0, D=2400.0,       # camera focal length / distance from the tower axis
    Rw=930.0,                 # inner wall radius
    rc=78.0,                  # central column radius
    Rs=820.0,                 # outer radius of the treads
    n=14,                     # steps per turn
    rise=46.0,                # height per step
    thick=30.0,               # tread slab thickness
    a0=-2.2,                  # angle of step 0 (0 = nearest the camera, +pi/2 = right)
    e=0.30,                   # obliqueness (how much of the treads' tops we see)
    h0=360.0,                 # camera height at scroll=0
)

_WALL_SRC = """
uniform shader blot;
uniform float f;
uniform float D;
uniform float Rw;
uniform float hcam;
uniform float2 lp;
uniform float lrad;
uniform float3 warm;
uniform float3 amb;
uniform float3 cool;
uniform float t;

half4 main(float2 p) {
    float dx = (p.x - 960.0) / f;
    float dy = -(p.y - 540.0) / f;
    float A = dx * dx + 1.0;
    float disc = D * D - A * (D * D - Rw * Rw);
    float tt = (D + sqrt(max(disc, 0.0))) / A;
    float X = dx * tt;
    float Z = -D + tt;
    float Y = hcam + dy * tt;
    float ang = atan(X, Z);
    float s = ang * Rw;
    // masonry courses
    float ch = 58.0;
    float row = floor(Y / ch);
    float fy = fract(Y / ch);
    float bw = 118.0;
    float off = mod(row, 2.0) * 0.5;
    float fx = fract(s / bw + off + sin(row * 1.7) * 0.13);
    float bid = floor(s / bw + off + sin(row * 1.7) * 0.13) + row * 17.0;
    float px = 2.0 / (tt / f);     // pixels -> units scale
    float mort = min(min(fy, 1.0 - fy) * ch, min(fx, 1.0 - fx) * bw);
    float m = smoothstep(1.2, 3.8, mort);
    half4 n = blot.eval(float2(s * 0.35, Y * 0.35));
    float tone = fract(sin(bid * 12.9898) * 43758.5453);
    float3 stone = mix(float3(0.50, 0.44, 0.40), float3(0.62, 0.56, 0.50), tone);
    stone *= 0.85 + 0.35 * float(n.r) + 0.12 * float(n.g);
    // block bevel: lighter top edge, darker bottom edge
    stone *= 1.0 + 0.10 * smoothstep(0.75, 0.97, fy) - 0.12 * smoothstep(0.25, 0.03, fy);
    float3 alb = mix(float3(0.16, 0.14, 0.14), stone, m);
    // lighting: warm pool (screen space, falls off softly) + ambient + cool from above
    float2 d = (p - lp) / lrad;
    float wl = exp(-dot(d, d) * 1.6) * 1.0 + exp(-dot(d, d) * 0.35) * 0.25;
    float3 L = amb + warm * wl + cool * (0.4 + 0.6 * smoothstep(-600.0, 900.0, Y - hcam));
    // side walls fade darker (curving towards camera)
    L *= 0.55 + 0.45 * smoothstep(-1.0, 0.3, cos(ang));
    return half4(half3(alb * L), 1.0);
}
"""
_WALLSH = None


def _wall_shader():
    global _WALLSH
    if _WALLSH is None:
        _WALLSH = _Shader(_WALL_SRC)
    return _WALLSH


def _st_proj(r, a, Y, hcam):
    """3D (radius, angle, height) -> screen (x, y, scale, Z)."""
    g = ST
    X = r * math.sin(a)
    Z = -r * math.cos(a)
    k = g['f'] / (Z + g['D'])
    # oblique view from slightly above: nearer points sit lower on screen, so every tread
    # shows its top surface
    return 960.0 + X * k, 540.0 - (Y - hcam) * k - Z * g['e'] * k, k, Z


def _st_windows(hcam):
    """Round windows on the wall near the visible height range: list of (angle, Y)."""
    out = []
    i0 = int(math.floor((hcam - 1200) / 620.0))
    for i in range(i0, i0 + 6):
        Y = 380 + i * 620.0
        a = math.pi + 0.75 * math.sin(i * 2.31 + 0.4)
        out.append((a, Y, i))
    return out


def stairs_interior(c, t, scroll=0.0, light=0.6, light_pos=None, layer='all', robot_k=None, dawn=0.0,
                    light_color=WARM2, moon=1.0):
    """Full-screen view inside the lighthouse tower: a stone spiral staircase winding around a
    central column (cutaway view — the near half of the wall is removed), masonry wall with
    small round windows showing the night, warm local light.

    t:          time (s)
    scroll:     vertical camera travel in world units (increase to climb; ~46 per step,
                644 per full turn)
    light:      0..1 overall warm light level (lantern/porthole glow)
    light_pos:  (x, y) screen position of the robot's chest light -> warm light pool on walls,
                steps and column (default: centre of frame)
    layer:      'all' | 'back' | 'front' — split drawing so a character on step `robot_k` is
                correctly occluded: draw layer='back', then the robot at step_pos(robot_k),
                then layer='front' (same other arguments).
    robot_k:    the (float) step index the character stands on (needed for back/front split)
    moon:       cool moonlight through the windows
    Returns step_pos(k) -> (x, y, depth_scale): screen position of the middle of the tread
    of step k (float k allowed; feet go here) and the perspective scale there (≈0.75 far side
    … 1.0 at the column plane … 1.5 nearest). Extra helpers on the function object:
    step_pos.front(k) -> +1 nearest the camera .. -1 behind the column;
    step_pos.dir(k) -> +1 if climbing moves right on screen at k, -1 if left;
    step_pos.hidden(k) -> True if the column hides that spot."""
    g = ST
    hcam = g['h0'] + scroll
    lp = light_pos if light_pos is not None else (960.0, 560.0)
    light = clamp(light)
    rmid = (g['rc'] + g['Rs']) * 0.52
    da = math.tau / g['n']

    def step_pos(k):
        a = g['a0'] + k * da
        x, y, sc, Z = _st_proj(rmid, a, k * g['rise'], hcam)
        return x, y, sc

    step_pos.front = lambda k: math.cos(g['a0'] + k * da)
    step_pos.dir = lambda k: 1.0 if math.cos(g['a0'] + k * da) >= 0 else -1.0

    def _hidden(k):
        a = g['a0'] + k * da
        if math.cos(a) >= 0:
            return False
        x, y, sc = step_pos(k)
        cxr = g['rc'] * g['f'] / g['D'] + 30
        return abs(x - 960) < cxr
    step_pos.hidden = _hidden

    warm = _vscale(_rgb(light_color), 1.15 * light)
    amb = _vmix((0.10, 0.10, 0.17), (0.20, 0.16, 0.22), dawn)
    cool = _vscale((0.10, 0.13, 0.24), moon)
    if layer in ('all', 'back'):
        sh = _wall_shader().make([_tex()['blot']], f=g['f'], D=g['D'], Rw=g['Rw'], hcam=hcam, lp=lp, lrad=620.0,
                                 warm=warm, amb=amb, cool=cool, t=t)
        c.drawRect(skia.Rect.MakeLTRB(-10, -10, W + 10, H + 10), skia.Paint(Shader=sh))
        _st_draw_windows(c, t, hcam, moon, dawn)
    # visible step range
    kmin = int(math.floor((hcam - 900) / g['rise'])) - 2
    kmax = int(math.ceil((hcam + 900) / g['rise'])) + 2
    items = []
    for k in range(kmin, kmax + 1):
        a = g['a0'] + k * da
        Zc = -rmid * math.cos(a)
        items.append((Zc, 'step', k))
    items.append((0.0, 'col', None))
    if robot_k is not None and layer != 'all':
        ar = g['a0'] + robot_k * da
        Zr = -rmid * math.cos(ar)

        def in_front(it):
            Zi, kind, k = it
            if kind == 'col':
                return Zr > 0
            return k > robot_k + 0.5 and Zi < Zr
        items = [it for it in items if in_front(it) == (layer == 'front')]
    elif layer == 'front':
        items = []
    items.sort(key=lambda it: -it[0])
    for Zi, kind, k in items:
        if kind == 'col':
            _st_column(c, t, hcam, lp, warm, amb, cool, light)
        else:
            _st_step(c, k, hcam, lp, warm, amb, cool, light, da)
    if layer in ('all', 'front'):
        # moonlight shafts + dust in the air
        _st_air(c, t, hcam, lp, light, moon, light_color)
    return step_pos


def _st_draw_windows(c, t, hcam, moon, dawn):
    g = ST
    for (a, Y, i) in _st_windows(hcam):
        pts_o, pts_i = [], []
        for j in range(28):
            th = j / 28 * math.tau
            # window: circle on the wall, radius 62 (outer reveal 84)
            for (rr, lst) in ((84.0, pts_o), (60.0, pts_i)):
                aa = a + rr * math.cos(th) / g['Rw']
                yy = Y + rr * math.sin(th)
                x, y, sc, Z = _st_proj(g['Rw'], aa, yy, hcam)
                lst.append((x, y))
        if max(p[1] for p in pts_o) < -50 or min(p[1] for p in pts_o) > H + 50:
            continue
        c.drawPath(_path_from(pts_o, close=True), fill('#1a1618'))
        c.drawPath(_path_from(pts_o[14:], close=False), stroke('#6a5e58', 3.0, 0.5))
        ip = _path_from(pts_i, close=True)
        cx = sum(p[0] for p in pts_i) / len(pts_i)
        cy = sum(p[1] for p in pts_i) / len(pts_i)
        c.drawPath(ip, radial((cx, cy - 20), 90, [(0, mix('#2a3e7a', '#8a6fb0', dawn), 1), (1, mix('#0c1433', '#3a3060', dawn), 1)]))
        with saved_clip(c, ip):
            r = nprng(500 + i)
            for s_ in range(5):
                sx, sy = cx + r.uniform(-45, 45), cy + r.uniform(-45, 45)
                c.drawCircle(sx, sy, r.uniform(0.8, 1.8), fill('#fff6e0', 0.5 + 0.4 * math.sin(t * 2 + s_ + i)))
            # glass glint + cross bar
            c.drawLine(cx - 70, cy, cx + 70, cy, stroke('#1a1618', 4.0))
            c.drawLine(cx, cy - 70, cx, cy + 70, stroke('#1a1618', 4.0))
        c.drawPath(ip, stroke('#9fc4ff', 2.0, 0.25 * moon))
        soft_glow(c, cx, cy, 150, COOL, 0.10 * moon)


def _st_light_at(x, y, lp, warm, amb, cool, light, lrad=560.0):
    d2 = ((x - lp[0]) ** 2 + (y - lp[1]) ** 2) / (lrad * lrad)
    wl = math.exp(-d2 * 1.6) + 0.25 * math.exp(-d2 * 0.35)
    return tuple(amb[i] * 1.2 + cool[i] * 0.8 + warm[i] * wl for i in range(3))


def _st_step(c, k, hcam, lp, warm, amb, cool, light, da):
    g = ST
    a = g['a0'] + k * da
    a1, a2 = a - da / 2, a + da / 2
    Y = k * g['rise']
    Yb = Y - g['thick']
    rc, Rs = g['rc'], g['Rs']
    nseg = 6
    # vertices
    def P(r, aa, yy):
        x, y, sc, Z = _st_proj(r, aa, yy, hcam)
        return (x, y)
    outer_top = [P(Rs, a1 + (a2 - a1) * i / nseg, Y) for i in range(nseg + 1)]
    outer_bot = [P(Rs, a1 + (a2 - a1) * i / nseg, Yb) for i in range(nseg + 1)]
    in_t1, in_t2 = P(rc, a1, Y), P(rc, a2, Y)
    in_b1, in_b2 = P(rc, a1, Yb), P(rc, a2, Yb)
    top = [in_t1] + outer_top + [in_t2]
    bot = [in_b1] + outer_bot + [in_b2]
    ys = [p[1] for p in top + bot]
    if max(ys) < -40 or min(ys) > H + 40:
        return
    xm, ym = (outer_top[nseg // 2][0] + in_t1[0]) / 2, (outer_top[nseg // 2][1] + in_t1[1]) / 2
    L = _st_light_at(xm, ym, lp, warm, amb, cool, light)
    stone_top = (0.66, 0.58, 0.50)
    stone_side = (0.46, 0.40, 0.36)
    stone_under = (0.30, 0.27, 0.27)

    def lit(alb, kk=1.0):
        return tuple(clamp(alb[i] * L[i] * kk) for i in range(3))

    def area(pts):
        s = 0.0
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            s += x1 * y2 - x2 * y1
        return s / 2
    faces = []
    # top (seen when camera above)
    faces.append((top, lit(stone_top), 'top'))
    faces.append((bot[::-1], lit(stone_under, 0.8), 'bot'))
    # riser at a1 (towards lower steps) and back face at a2
    faces.append(([in_t1, outer_top[0], outer_bot[0], in_b1], lit(stone_side, 0.95), 'r1'))
    faces.append(([in_t2, in_b2, outer_bot[-1], outer_top[-1]], lit(stone_side, 0.75), 'r2'))
    # outer end (cut section, visible for near steps)
    faces.append((outer_top + outer_bot[::-1], lit(stone_side, 0.65), 'out'))
    ref = area(top)
    for pts, colr, name in faces:
        a_ = area(pts)
        if name == 'top':
            vis = True
        elif name == 'bot':
            vis = False
        else:
            vis = a_ * (1 if ref >= 0 else -1) < 0 if name in ('r1', 'out') else a_ * (1 if ref >= 0 else -1) > 0
            vis = abs(a_) > 1.0
        if not vis:
            continue
        c.drawPath(_path_from(pts, close=True), fill(colr))
    # nosing highlight on the front edge of the tread
    hl = _st_light_at(xm, ym, lp, warm, amb, cool, light)
    c.drawLine(in_t1[0], in_t1[1], outer_top[0][0], outer_top[0][1],
               stroke(tuple(clamp(0.9 * hl[i]) for i in range(3)), 2.0, 0.7))
    # iron handrail post + rail segment near the wall
    rr = Rs - 40
    p0 = P(rr, a, Y)
    p1 = P(rr, a, Y + 150)
    rail0 = P(rr, a1, Y + 150 - g['rise'] / 2)
    rail1 = P(rr, a2, Y + 150 + g['rise'] / 2)
    ic = tuple(clamp(0.35 * hl[i] + 0.03) for i in range(3))
    sc = _st_proj(rr, a, Y, hcam)[2]
    c.drawLine(p0[0], p0[1], p1[0], p1[1], stroke(ic, 4.0 * sc))
    c.drawLine(rail0[0], rail0[1], rail1[0], rail1[1], stroke(ic, 7.0 * sc))
    c.drawLine(rail0[0], rail0[1] - 2 * sc, rail1[0], rail1[1] - 2 * sc,
               stroke(tuple(clamp(0.9 * hl[i]) for i in range(3)), 1.6 * sc, 0.6))


def _st_column(c, t, hcam, lp, warm, amb, cool, light):
    g = ST
    k = g['f'] / g['D']
    hw = g['rc'] * k
    x0, x1 = 960 - hw, 960 + hw
    # horizontal shading: cylinder
    L = _st_light_at(960, lp[1], lp, warm, amb, cool, light)
    base = (0.52, 0.47, 0.44)
    lc = tuple(clamp(base[i] * L[i] * 1.1) for i in range(3))
    dc = tuple(clamp(base[i] * L[i] * 0.35) for i in range(3))
    side = 1.0 if lp[0] < 960 else -1.0
    stops = [(0, lc if side > 0 else dc, 1), (0.35, lc, 1), (1, dc if side > 0 else lc, 1)]
    c.drawRect(skia.Rect.MakeLTRB(x0, -10, x1, H + 10), linear((x0, 0), (x1, 0), stops))
    # vertical falloff of the warm light along the column
    vp = linear((0, lp[1] - 700), (0, lp[1] + 700), [(0, '#000000', 0.55), (0.5, '#000000', 0.0), (1, '#000000', 0.55)])
    c.drawRect(skia.Rect.MakeLTRB(x0, -10, x1, H + 10), vp)
    # stone drums
    ch = 58.0
    y0 = hcam - 700
    j = math.floor(y0 / ch)
    while j * ch < hcam + 700:
        Y = j * ch
        yy = 540 - (Y - hcam) * k
        c.drawLine(x0, yy, x1, yy, stroke('#1a1414', 1.6, 0.6))
        j += 1
    c.drawLine(x0 + 2, -10, x0 + 2, H + 10, stroke(COOL, 2.0, 0.08))


def _st_air(c, t, hcam, lp, light, moon, light_color):
    # a soft warm halo around the light source (volumetric air)
    if light > 0:
        soft_glow(c, lp[0], lp[1], 380, light_color, 0.22 * light)
    # moonlight shafts from the windows
    for (a, Y, i) in _st_windows(hcam):
        x, y, sc, Z = _st_proj(ST['Rw'], a, Y, hcam)
        if -200 < y < H + 200:
            p = skia.Path()
            p.moveTo(x - 50, y - 40)
            p.lineTo(x + 50, y + 40)
            p.lineTo(x + 520, y + 760)
            p.lineTo(x + 260, y + 820)
            p.close()
            sp = linear((x, y), (x + 400, y + 800), [(0, COOL, 0.10 * moon), (1, COOL, 0.0)])
            sp.setBlendMode(ADD)
            c.drawPath(p, sp)
    # dust motes near the light
    r = nprng(611)
    n = 40
    for i in range(n):
        bx, by = r.uniform(-500, 500), r.uniform(-400, 400)
        sp_ = r.uniform(5, 15)
        ph = r.uniform(0, math.tau)
        x = lp[0] + bx + math.sin(t * 0.4 + ph) * 25
        y = lp[1] + ((by - t * sp_ + 400) % 800) - 400
        d = math.hypot(x - lp[0], y - lp[1])
        a_ = light * 0.5 * clamp(1 - d / 550) * (0.5 + 0.5 * math.sin(t * 1.9 + ph))
        if a_ > 0.02:
            c.drawCircle(x, y, r.uniform(1.0, 2.2), fill('#ffe2b0', a_, blend=ADD))


def gallery_railing(c, x0, x1, y, s=1.0):
    """Foreground railing of the outside balcony (the 'gallery') around the lamp room."""
    c.drawRect(skia.Rect.MakeLTRB(x0, y, x1, y + 10 * s), fill('#2a2f45'))


def _boat_curve(u, pts):
    """Piecewise Catmull-Rom sample of a list of (x, y) at u in [0,1]."""
    n = len(pts) - 1
    k = min(n - 1, int(u * n))
    lu = u * n - k
    p0 = pts[max(0, k - 1)]
    p1, p2 = pts[k], pts[k + 1]
    p3 = pts[min(n, k + 2)]
    l2, l3 = lu * lu, lu * lu * lu
    return tuple(0.5 * ((2 * p1[i]) + (-p0[i] + p2[i]) * lu + (2 * p0[i] - 5 * p1[i] + 4 * p2[i] - p3[i]) * l2 +
                        (-p0[i] + 3 * p1[i] - 3 * p2[i] + p3[i]) * l3) for i in range(2))


# hull profile (local units, bow at +x, waterline y=0; gunwale top ≈ -30 like the old stub)
_BOAT_SHEER = [(-160, -30), (-90, -24), (0, -22), (90, -27), (150, -40), (176, -60)]   # top edge
_BOAT_KEEL = [(-150, 22), (-80, 30), (0, 31), (80, 26), (140, 8), (176, -60)]          # bottom edge


def boat(c, x, y, s=1.0, rock=0.0, lantern=1.0, t=0.0, dawn=0.0, sunrise=0.0, wet=0.0):
    """Small wooden rowing boat seen from the side, bow to the right (use a negative `s`
    to mirror). Draw it AFTER the robot so the hull hides his legs. (x,y) = waterline
    centre; s=1 → ~340 long; gunwale top at local y≈-30, oarlock at local (12, -26).
    rock: tilt (radians). lantern: 0..1 brightness of the bow lantern (on a curved pole).
    Returns the world (x, y) of the lantern flame."""
    env = light_env(dawn, sunrise)
    wood = (0.56, 0.37, 0.23)
    rail = (0.66, 0.46, 0.30)
    lk = clamp(lantern)

    def shade(base, k_moon, warm=0.0):
        cc = _lit(base, env, k_moon)
        return tuple(clamp(cc[i] + base[i] * warm * (1.0, 0.72, 0.42)[i] * 0.55) for i in range(3))

    N = 28
    top = [_boat_curve(i / N, _BOAT_SHEER) for i in range(N + 1)]
    bot = [_boat_curve(i / N, _BOAT_KEEL) for i in range(N + 1)]

    def strake(a, b):  # band between fractions a..b from gunwale (0) to keel (1)
        up = [(tx + (bx - tx) * a, ty + (by - ty) * a) for (tx, ty), (bx, by) in zip(top, bot)]
        dn = [(tx + (bx - tx) * b, ty + (by - ty) * b) for (tx, ty), (bx, by) in zip(top, bot)]
        return poly(up + dn[::-1])

    sway = 0.06 * math.sin(t * 1.9) + rock * 1.5
    with at(c, x, y, rot=rock, sx=s):
        pxw = _px(c)
        # --- hull strakes: lighter at the top (moonlit), darker to the keel -------------------
        bands = [(0.0, 0.3, 1.05), (0.3, 0.58, 0.78), (0.58, 0.8, 0.55), (0.8, 1.0, 0.36)]
        for a, b, km in bands:
            p = linear((150, 0), (-160, 0), [(0, shade(wood, km, 0.9 * lk), 1), (0.55, shade(wood, km, 0.25 * lk), 1),
                                             (1, shade(wood, km * 0.9), 1)])
            c.drawPath(strake(a, b), p)
        # plank seams + nails
        seam = stroke(shade(wood, 0.12), max(1.2, 0.9 * pxw), 0.85)
        for a in (0.3, 0.58, 0.8):
            pts = [(tx + (bx - tx) * a, ty + (by - ty) * a) for (tx, ty), (bx, by) in zip(top, bot)]
            c.drawPath(smooth_path(pts), seam)
            for i in range(2, N - 2, 3):
                nx, ny = pts[i]
                c.drawCircle(nx, ny - 3, 1.4, fill(shade(wood, 0.1)))
        # ribs showing as faint vertical lines
        for i in range(4, N - 3, 4):
            (tx, ty), (bx, by) = top[i], bot[i]
            c.drawLine(tx, ty + 6, tx + (bx - tx) * 0.96, ty + (by - ty) * 0.96,
                       stroke(shade(wood, 0.2), max(1.0, 0.7 * pxw), 0.35))
        # transom (stern board) and stem post (bow)
        c.drawPath(poly([(-164, -33), (-150, -34), (-138, 22), (-152, 24)]), fill(shade(wood, 0.5)))
        stem = skia.Path()
        stem.moveTo(170, -66)
        stem.cubicTo(178, -40, 162, 0, 140, 12)
        c.drawPath(stem, stroke(shade(rail, 0.7, 0.8 * lk), 7, 1.0))
        # gunwale rail with a moonlit top edge
        rp = smooth_path(top)
        c.drawPath(rp, stroke(shade(rail, 0.95, 0.9 * lk), 9, 1.0))
        c.drawPath(rp, stroke(env['moon'], 2.2, 0.55))
        # oarlock
        c.drawPath(poly([(6, -24), (18, -24), (16, -34), (8, -34)]), fill(shade((0.45, 0.45, 0.5), 0.8)))
        c.drawCircle(12, -36, 5, stroke(shade((0.6, 0.55, 0.45), 1.0, 0.5 * lk), 2.5))
        # wet sheen stripes
        c.drawPath(strake(0.05, 0.12), fill(env['moon'], 0.10 + 0.12 * wet))
        # --- the waterline: submerged part fades into the sea + a foam line ------------------
        c.save()
        c.clipPath(strake(0.0, 1.0), skia.ClipOp.kIntersect, True)
        uw = linear((0, 3), (0, 30), [(0, '#123060', 0.55), (1, '#0a1a38', 0.95)])
        c.drawRect(skia.Rect.MakeLTRB(-180, 3, 185, 45), uw)
        c.restore()
        for k in range(9):  # broken foam dashes lapping at the waterline
            xx = -140 + k * 34 + 8 * math.sin(t * 1.3 + k)
            ww = 16 + 8 * math.sin(k * 2.1 + t * 2.0)
            yy = 3 + 1.2 * math.sin(xx * 0.08 + t * 3.1)
            c.drawLine(xx, yy, xx + ww, yy, stroke(FOAM, 2.0, 0.35))
        c.drawLine(-150, 3, 150, 3, stroke(FOAM, 6.0, 0.08, blur=3))
        # --- lantern pole (curved iron) and the lantern ----------------------------------------
        pole = skia.Path()
        pole.moveTo(112, -30)
        pole.cubicTo(108, -90, 118, -134, 132, -142)
        c.drawPath(pole, stroke(shade((0.30, 0.30, 0.36), 0.9), 5, 1.0))
        c.save()
        c.translate(132, -142)
        c.rotate(math.degrees(sway))
        c.drawLine(0, 0, 0, 10, stroke('#3a3a44', 2.0))
        # lantern body: cap, glass, base
        glass = rrect(-11, 10, 22, 26, 5)
        if lk > 0:
            gp = radial((0, 23), 20, [(0, '#fff6d0', 1.0 * lk), (0.5, '#ffc86b', 0.9 * lk), (1, '#b86a2a', 0.6 * lk)])
            c.drawRRect(glass, gp)
            fl_ = radial((0, 24), 7, [(0, '#ffffff', lk), (1, '#ffd36b', 0)])
            fl_.setBlendMode(ADD)
            c.drawCircle(0, 24, 7, fl_)
        else:
            c.drawRRect(glass, fill('#2a3350', 0.9))
        for xx in (-11, 0, 11):
            c.drawLine(xx, 10, xx, 36, stroke('#6b5a3a', 1.6, 0.9))
        c.drawPath(poly([(-14, 11), (14, 11), (6, 2), (-6, 2)]), fill(shade((0.55, 0.45, 0.30), 0.9, 0.6 * lk)))
        c.drawRRect(rrect(-13, 34, 26, 5, 2), fill(shade((0.55, 0.45, 0.30), 0.8, 0.6 * lk)))
        c.restore()
        # warm pool of lantern light on the bow planks
        if lk > 0:
            wp = radial((125, -40), 170, [(0, '#ffc86b', 0.22 * lk), (1, '#ffc86b', 0)])
            wp.setBlendMode(ADD)
            c.drawPath(strake(0.0, 1.0), wp)
    # world position of the flame (same transform maths as above)
    fxl, fyl = 132 - math.sin(sway) * 24, -142 + math.cos(sway) * 24
    cr, sr = math.cos(rock), math.sin(rock)
    wx = x + (fxl * cr - fyl * sr) * s
    wy = y + (fxl * sr + fyl * cr) * abs(s)
    if lk > 0:
        soft_glow(c, wx, wy, 110 * abs(s), '#ffc86b', 0.8 * lk)
    return wx, wy


def crate(c, x, y, s=1.0, rot=0.0, kind='crate', dawn=0.0, sunrise=0.0, sun_dir=1.0, seed=0, warm=0.0,
          wet=0.0):
    """Wooden crate / barrel prop. (x,y) = bottom centre. kind in {'crate','barrel'}. s=1 → ~120 tall.

    rot:   rotation in radians about the bottom centre (tumbling tower)
    seed:  varies wood tone / plank layout
    warm:  0..1 warm lamplight tint (e.g. near the lit door)
    wet:   0..1 darker + glossy (after the splash)
    Crates are drawn in a gentle 3/4 view: front face 120x120, a side face on the right and
    the top face visible."""
    env = light_env(dawn, sunrise, sun_dir)
    r = nprng(700 + seed)
    tone = r.uniform(-0.08, 0.08)
    base = (0.62 + tone, 0.44 + tone * 0.8, 0.28 + tone * 0.5)
    if kind == 'barrel':
        base = (0.55 + tone, 0.36 + tone * 0.8, 0.22 + tone * 0.5)
    base = _vmix(base, (0.25, 0.18, 0.13), 0.45 * wet)
    wk = _vscale((1.0, 0.72, 0.42), 0.35 * warm)

    def shade(k_moon, k_add=0.0):
        cc = _lit(base, env, k_moon)
        return tuple(clamp(cc[i] + wk[i] * base[i] + k_add) for i in range(3))
    with at(c, x, y, rot=rot, sx=s):
        pxw = _px(c)
        if kind == 'barrel':
            _barrel(c, env, shade, pxw, r, wet)
        else:
            _crate(c, env, shade, pxw, r, wet)


def _crate(c, env, shade, pxw, r, wet):
    d, lift = 26.0, 16.0            # side depth / vertical skew of the receding faces
    front = shade(0.75)
    side = shade(0.30)
    top = shade(1.10)
    dark = shade(0.18)
    edge = shade(0.95)
    # side face (right)
    c.drawPath(poly([(60, 0), (60 + d, -lift), (60 + d, -120 - lift), (60, -120)]), fill(side))
    # top face
    c.drawPath(poly([(-60, -120), (60, -120), (60 + d, -120 - lift), (-60 + d, -120 - lift)]), fill(top))
    # front face
    c.drawRect(skia.Rect.MakeLTRB(-60, -120, 60, 0), fill(front))
    gap = stroke(dark, max(1.6, 0.8 * pxw), 0.9, cap=skia.Paint.kButt_Cap)
    # planks on the front (3 horizontal boards)
    ys = [-120, -80 + r.uniform(-4, 4), -40 + r.uniform(-4, 4), 0]
    for yy in ys[1:-1]:
        c.drawLine(-60, yy, 60, yy, gap)
    # grain
    gp = stroke(dark, max(0.7, 0.5 * pxw), 0.35)
    for k in range(9):
        yy = -114 + k * 13 + r.uniform(-3, 3)
        p = skia.Path()
        p.moveTo(-54, yy)
        p.cubicTo(-20, yy + r.uniform(-3, 3), 20, yy + r.uniform(-3, 3), 54, yy + r.uniform(-2, 2))
        c.drawPath(p, gp)
    # frame boards + diagonal brace
    fr = shade(0.9)
    for (x0, y0, x1, y1) in ((-60, -120, -46, 0), (46, -120, 60, 0), (-60, -120, 60, -108), (-60, -12, 60, 0)):
        c.drawRect(skia.Rect.MakeLTRB(x0, y0, x1, y1), fill(fr))
    c.drawPath(poly([(-46, -12), (-34, -12), (46, -96), (46, -108), (34, -108), (-46, -24)]), fill(fr))
    ol = stroke(dark, max(1.0, 0.6 * pxw), 0.7)
    c.drawRect(skia.Rect.MakeLTRB(-46, -108, 46, -12), ol)
    # nails
    for (nx, ny) in ((-53, -114), (53, -114), (-53, -6), (53, -6), (-53, -60), (53, -60)):
        c.drawCircle(nx, ny, 1.6, fill(shade(0.2)))
        c.drawCircle(nx - 0.5, ny - 0.5, 0.7, fill(env['moon'], 0.6))
    # side face planks
    for k in (1, 2):
        yy = -120 + k * 40
        c.drawLine(60, yy, 60 + d, yy - lift, stroke(dark, max(1.2, 0.7 * pxw), 0.8))
    # top face boards
    for k in (1, 2):
        xx = -60 + k * 40
        c.drawLine(xx, -120, xx + d, -120 - lift, stroke(shade(0.6), max(1.0, 0.6 * pxw), 0.8))
    # moonlit rims (upper-left edges)
    c.drawLine(-60, -120, -60, 0, stroke(env['moon'], 1.6, 0.35))
    c.drawLine(-60, -120, 60, -120, stroke(edge, 1.6, 0.8))
    c.drawLine(-60 + d, -120 - lift, 60 + d, -120 - lift, stroke(env['moon'], 1.2, 0.3))
    if wet > 0:
        c.drawRect(skia.Rect.MakeLTRB(-50, -110, -40, -20), fill(env['moon'], 0.18 * wet))


def _barrel(c, env, shade, pxw, r, wet):
    hw, hb = 45.0, 38.0         # half width at belly / at the ends
    top_e = 9.0
    path = skia.Path()
    path.moveTo(-hb, -top_e * 0.2)
    path.cubicTo(-hw - 6, -40, -hw - 6, -80, -hb, -120)
    path.lineTo(hb, -120)
    path.cubicTo(hw + 6, -80, hw + 6, -40, hb, -top_e * 0.2)
    path.quadTo(0, top_e, -hb, -top_e * 0.2)
    path.close()
    lc = shade(1.0)
    mc = shade(0.6)
    dc = shade(0.15)
    p = linear((-hw, 0), (hw, 0), [(0, mc, 1), (0.25, lc, 1), (0.55, mc, 1), (1, dc, 1)])
    c.drawPath(path, p)
    with saved_clip(c, path):
        sp = stroke(dc, max(1.0, 0.6 * pxw), 0.6)
        for k in range(-3, 4):
            u = k / 3.6
            q = skia.Path()
            q.moveTo(u * hb, 0)
            q.cubicTo(u * (hw + 5), -40, u * (hw + 5), -80, u * hb, -120)
            c.drawPath(q, sp)
        # iron hoops
        for yy in (-16, -38, -82, -104):
            belly = 1 - ((yy + 60) / 60) ** 2
            ww = hb + (hw + 4 - hb) * belly
            hp = skia.Path()
            hp.moveTo(-ww - 2, yy)
            hp.quadTo(0, yy + 6, ww + 2, yy)
            c.drawPath(hp, stroke(_lit('#2a2f45', env, 0.8), 4.5, cap=skia.Paint.kButt_Cap))
            hp2 = skia.Path()
            hp2.moveTo(-ww, yy - 1.6)
            hp2.quadTo(-ww * 0.3, yy + 2.5, 0, yy + 1.4)
            c.drawPath(hp2, stroke(env['moon'], 1.0, 0.4))
    # lid
    lid = skia.Rect.MakeLTRB(-hb, -120 - top_e, hb, -120 + top_e)
    c.drawOval(lid, fill(shade(1.1)))
    c.drawOval(skia.Rect.MakeLTRB(-hb + 5, -120 - top_e + 2.5, hb - 5, -120 + top_e - 2.5), fill(shade(0.7)))
    c.drawLine(-hb * 0.6, -120, hb * 0.6, -120, stroke(dc, 1.0, 0.5))
    c.drawOval(lid, stroke(_lit('#2a2f45', env, 0.9), 2.5))
    if wet > 0:
        c.drawRect(skia.Rect.MakeLTRB(-30, -100, -24, -20), fill(env['moon'], 0.2 * wet))


def rocks_foreground(c, t, y=980, seed=1, dawn=0.0):
    """Dark foreground rocks along the bottom of the frame (placeholder)."""
    c.drawPath(poly([(-100, H + 50), (-100, y), (300, y - 40), (700, y + 20), (1200, y - 10), (1700, y + 30), (2100, y), (2100, H + 50)]), fill('#0c1022'))
