"""FX library — owner: FX agent.

Light, particles and magic. Every effect is a PURE FUNCTION of its inputs and
time (no per-frame state), so frames can be rendered in any order / in parallel.
Seeded randomness only (hash of particle index + seed).

CONTRACT: keep the signatures below working (add kwargs with defaults and new
functions freely). STUB=True means placeholder art.

Implementation notes
--------------------
* All coordinates are design units (the renderer scales the canvas).
* Light is drawn additively (engine.core.ADD) so it stacks and blooms.
* Big soft shapes (glows, beams, ribbons, rings, rays) are triangle meshes with
  per-vertex colours (`canvas.drawVertices`) — ~3x cheaper per pixel than
  gradient shaders on the CPU rasteriser and free to shape.
* Small particles are drawn from precomputed mip-mapped sprites
  (soft glow, hot core, 4-point glint, streak, droplet, bokeh…) with ONE
  `drawAtlas` call per sprite type, colours tinted per particle.
* Everything is culled against the canvas clip, so effects that extend far
  off-screen (the pillar reaches y=-3000) cost only what is visible.

Public API (all take the skia canvas `c` first):
  glow, sparkles, meteor, splash, ripples, bioluminescence, light_pillar,
  stardust_stream, shockwave, lens_flare, god_rays, dust_motes, heart_core,
  embers, speed_lines, sparkle_burst, water_drips, impact_flash, light_trail,
  glint (single 4-point star glint), bezier_point (path helper).
"""
from __future__ import annotations

import math

import numpy as np
import skia

from engine.core import (ADD, at, clamp, col, ease_in_out, ease_out, fill, hexrgb, lerp, linear, mix, noise1,
                         nprng, poly, radial, smoothstep, soft_glow, stroke)

STUB = False
TAU = math.tau
_WHITE = np.array([1.0, 1.0, 1.0])


# =============================================================================
# low level helpers
# =============================================================================
def _c3(c):
    """colour ('#rrggbb' or tuple) -> np.array([r,g,b]) floats."""
    if c is None:
        return None
    if isinstance(c, str):
        return np.array(hexrgb(c))
    if isinstance(c, np.ndarray):
        return c.astype(float)
    return np.array(c[:3], dtype=float)


def _mixc(a, b, t):
    a, b = _c3(a), _c3(b)
    t = np.asarray(t, dtype=float)
    if t.ndim == 0:
        return a + (b - a) * float(np.clip(t, 0, 1))
    t = np.clip(t, 0, 1)[..., None]
    return a + (b - a) * t


def _pack(rgb, a):
    """rgb (3,) or (...,3) floats and alpha scalar or (...) -> list of ARGB ints."""
    a = np.clip(np.asarray(a, dtype=np.float64), 0.0, 1.0)
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    A = (a * 255.0 + 0.5).astype(np.uint32)
    if rgb.ndim == 1:
        base = (int(rgb[0] * 255 + .5) << 16) | (int(rgb[1] * 255 + .5) << 8) | int(rgb[2] * 255 + .5)
        out = (A << np.uint32(24)) | np.uint32(base)
    else:
        q = (rgb * 255.0 + 0.5).astype(np.uint32)
        out = (A << np.uint32(24)) | (q[..., 0] << np.uint32(16)) | (q[..., 1] << np.uint32(8)) | q[..., 2]
    return np.broadcast_to(out, np.broadcast_shapes(A.shape, out.shape)).ravel().tolist()


def _hash(i, seed=0):
    """Vectorised integer hash -> [0,1) (same recipe as engine.core._hash)."""
    x = (np.asarray(i).astype(np.int64) * 374761393 + int(seed) * 668265263) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((x ^ (x >> 16)) & 0xFFFFFF) / float(0xFFFFFF)


def _h2(i, k, seed=0):
    """hash of (particle i, cycle k)."""
    return _hash(np.asarray(i, dtype=np.int64) * 7919 + np.asarray(k, dtype=np.int64) * 104729, seed)


def _noise(x, seed=0):
    """Vectorised smooth value noise in [-1,1]."""
    x = np.asarray(x, dtype=np.float64)
    i = np.floor(x)
    f = x - i
    u = f * f * (3 - 2 * f)
    a = _hash(i, seed)
    b = _hash(i + 1, seed)
    return (a + (b - a) * u) * 2 - 1


def _fbm(x, seed=0, octaves=3):
    v = 0.0
    amp, fr = 0.5, 1.0
    for o in range(octaves):
        v = v + amp * _noise(x * fr, seed + o * 17)
        amp *= 0.5
        fr *= 2.03
    return v / 0.875


def _sstep(a, b, x):
    t = np.clip((np.asarray(x, dtype=float) - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _clipb(c, pad=0.0):
    """Local clip bounds (L, T, R, B) padded."""
    r = c.getLocalClipBounds()
    return r.left() - pad, r.top() - pad, r.right() + pad, r.bottom() + pad


def _paint(blend=ADD, alpha=1.0):
    p = skia.Paint()
    if blend is not None:
        p.setBlendMode(blend)
    if alpha < 1.0:
        p.setAlphaf(clamp(alpha))
    return p


# ---------------------------------------------------------------------------
# meshes
# ---------------------------------------------------------------------------
_TRI = skia.Vertices.kTriangles_VertexMode
_IDX = {}


def _grid_idx(rows, cols, wrap=False):
    key = (rows, cols, wrap)
    idx = _IDX.get(key)
    if idx is None:
        r = np.arange(rows - 1)[:, None]
        k = np.arange(cols if wrap else cols - 1)[None, :]
        a = r * cols + k
        b = r * cols + (k + 1) % cols
        cc = a + cols
        d = b + cols
        idx = np.stack([a, b, cc, b, d, cc], -1).ravel().tolist()
        _IDX[key] = idx
    return idx


def _mesh(c, X, Y, colors, idx, blend=ADD):
    pts = list(zip(np.ravel(X).tolist(), np.ravel(Y).tolist()))
    v = skia.Vertices.MakeCopy(_TRI, pts, None, colors, idx)
    c.drawVertices(v, _paint(blend), skia.BlendMode.kDst)


def _multi_mesh(c, parts, blend=ADD):
    """Draw several (X, Y, colors(list), idx) grids in one drawVertices call."""
    xs, ys, cs, ids = [], [], [], []
    off = 0
    for X, Y, C, I in parts:
        X = np.ravel(X)
        xs.append(X)
        ys.append(np.ravel(Y))
        cs.extend(C)
        if off:
            ids.extend((np.asarray(I) + off).tolist())
        else:
            ids.extend(I)
        off += len(X)
    if not off:
        return
    _mesh(c, np.concatenate(xs), np.concatenate(ys), cs, ids, blend)


# radial disc geometry (unit) & profiles
_RR = np.array([0, .012, .03, .055, .085, .12, .165, .22, .285, .36, .445, .54, .645, .76, .88, 1.0])


def _profile(kind, rr=_RR):
    w = (1 - rr ** 2) ** 2
    if kind == 'soft':
        f = 0.4 * np.exp(-(rr / 0.16) ** 2) + 0.6 * np.exp(-(rr / 0.5) ** 2)
    elif kind == 'hot':   # sharp hot centre + long tail (like real light scatter)
        f = 0.6 / (1 + (rr / 0.05) ** 2) + 0.4 * np.exp(-(rr / 0.35) ** 2)
    elif kind == 'gauss':
        f = np.exp(-(rr / 0.42) ** 2)
    elif kind == 'flat':  # soft-edged disc (mist, foam)
        f = 1 - _sstep(0.45, 1.0, rr)
    elif kind == 'rim':   # disc brighter towards its edge (lens ghosts, bubbles)
        f = (0.35 + 0.65 * _sstep(0.4, 0.9, rr)) * (1 - _sstep(0.9, 1.0, rr))
        w = 1
    else:
        raise ValueError(kind)
    f = f * w
    return f / f[0] if f[0] > 0 else f


_PROF = {k: _profile(k) for k in ('soft', 'hot', 'gauss', 'flat', 'rim')}
_ANG = {}


def _angles(segs):
    a = _ANG.get(segs)
    if a is None:
        th = np.linspace(0, TAU, segs, endpoint=False)
        a = _ANG[segs] = (np.cos(th), np.sin(th))
    return a


def _disc_part(x, y, r, rgb, a, prof='soft', sy=1.0, core_rgb=None, segs=None, rr=None, pr=None):
    """Mesh part for a radial glow disc (use with _multi_mesh)."""
    if segs is None:
        segs = int(clamp(r * 0.3, 20, 72))
    rr = _RR if rr is None else rr
    pr = _PROF[prof] if pr is None else pr
    ca, sa = _angles(segs)
    R = rr * r
    X = x + R[:, None] * ca[None, :]
    Y = y + R[:, None] * sa[None, :] * sy
    A = np.broadcast_to((a * pr)[:, None], X.shape)
    rgb = _c3(rgb)
    if core_rgb is not None:
        k = _sstep(0.0, 0.35, rr)
        C = _mixc(core_rgb, rgb, k)[:, None, :]
        C = np.broadcast_to(C, X.shape + (3,))
    else:
        C = rgb
    return X, Y, _pack(C, A), _grid_idx(len(rr), segs, True)


def _disc(c, x, y, r, rgb, a=1.0, prof='soft', sy=1.0, core_rgb=None, segs=None, blend=ADD):
    if a <= 0.002 or r <= 0.3:
        return
    L, T, Rr, B = _clipb(c)
    if x + r < L or x - r > Rr or y + r * sy < T or y - r * sy > B:
        return
    _mesh(c, *_disc_part(x, y, r, rgb, a, prof, sy, core_rgb, segs), blend=blend)


def _ring_part(x, y, R, offs, alphas, rgb, segs=96, sy=1.0, amod=None, rot=0.0, rgbs=None):
    """Annulus mesh: radii R+offs (K,), alpha profile (K,), optional per-segment alpha mod (segs,)."""
    th = np.linspace(0, TAU, segs, endpoint=False) + rot
    ca, sa = np.cos(th), np.sin(th)
    rad = np.maximum(0.0, R + np.asarray(offs, dtype=float))
    X = x + rad[:, None] * ca[None, :]
    Y = y + rad[:, None] * sa[None, :] * sy
    A = np.asarray(alphas, dtype=float)[:, None] * (np.ones(segs) if amod is None else amod)[None, :]
    C = _c3(rgb) if rgbs is None else np.broadcast_to(np.asarray(rgbs)[:, None, :], X.shape + (3,))
    return X, Y, _pack(C, A), _grid_idx(len(rad), segs, True)


def _ribbon_part(P, w, offs, prof, rgb, a_along, normals=None):
    """Ribbon mesh along polyline P (M,2) with half-width w (M,), cross offsets offs (K,)
    in [-1,1] and cross alpha profile prof (K,). rgb (3,) or (M,3) or (M,K,3)."""
    P = np.asarray(P, dtype=float)
    if normals is None:
        normals = _normals(P)
    offs = np.asarray(offs, dtype=float)
    X = P[:, 0, None] + normals[:, 0, None] * w[:, None] * offs[None, :]
    Y = P[:, 1, None] + normals[:, 1, None] * w[:, None] * offs[None, :]
    A = np.asarray(a_along, dtype=float)[:, None] * np.asarray(prof, dtype=float)[None, :]
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim == 2:
        rgb = np.broadcast_to(rgb[:, None, :], X.shape + (3,))
    return X, Y, _pack(rgb, A), _grid_idx(len(P), len(offs))


def _normals(P):
    T = np.gradient(P, axis=0) if len(P) > 2 else np.repeat((P[1:] - P[:1]), len(P), 0)
    n = np.hypot(T[:, 0], T[:, 1])
    n[n < 1e-9] = 1.0
    return np.stack([-T[:, 1] / n, T[:, 0] / n], -1)


# cross-section profiles for ribbons
_X5 = np.array([-1, -.5, 0, .5, 1.0])
_P5 = np.array([0, .45, 1, .45, 0])
_X7 = np.array([-1, -.62, -.3, 0, .3, .62, 1.0])
_P7 = np.array([0, .16, .6, 1, .6, .16, 0])
_X3 = np.array([-1, 0, 1.0])
_P3 = np.array([0, 1.0, 0])


# ---------------------------------------------------------------------------
# sprites
# ---------------------------------------------------------------------------
class _Spr:
    __slots__ = ('img', 'w', 'h', 'rect')

    def __init__(self, arr_a, shade=None):
        a = np.clip(arr_a, 0, 1)
        s = a if shade is None else a * np.clip(shade, 0, 1)
        a8 = (a * 255 + 0.5).astype(np.uint8)
        s8 = (s * 255 + 0.5).astype(np.uint8)
        rgba = np.ascontiguousarray(np.dstack([s8, s8, s8, a8]))
        img = skia.Image.fromarray(rgba, colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kPremul_AlphaType)
        self.img = img.withDefaultMipmaps()
        self.h, self.w = a.shape
        self.rect = skia.Rect.MakeWH(self.w, self.h)


def _grid01(w, h=None):
    h = h or w
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    return (x + 0.5) / w * 2 - 1, (y + 0.5) / h * 2 - 1


def _make_sprite(name):
    if name == 'glow':       # soft gaussian-ish blob
        u, v = _grid01(64)
        r = np.hypot(u, v)
        return _Spr(np.exp(-(r / 0.45) ** 2) * np.clip(1 - r * r, 0, 1) ** 2)
    if name == 'core':       # small hot dot with a halo
        u, v = _grid01(64)
        r = np.hypot(u, v)
        a = 0.62 * np.exp(-(r / 0.13) ** 2) + 0.38 * np.exp(-(r / 0.42) ** 2)
        return _Spr(a * np.clip(1 - r * r, 0, 1) ** 2 / 1.0)
    if name in ('glint', 'glint8'):   # 4-point (or 8-point) star glint
        n = 160
        u, v = _grid01(n)
        au, av = np.abs(u), np.abs(v)
        r = np.hypot(u, v)
        # concave-sided star body (astroid) + thin long spikes + hot core
        s = (np.sqrt(au) + np.sqrt(av)) ** 2
        body = np.clip(1 - s / 0.55, 0, 1) ** 2.2
        spike = (np.exp(-(v / 0.022) ** 2) * np.clip(1 - au, 0, 1) ** 2.0 +
                 np.exp(-(u / 0.022) ** 2) * np.clip(1 - av, 0, 1) ** 2.0)
        core = np.exp(-(r / 0.07) ** 2)
        halo = 0.22 * np.exp(-(r / 0.22) ** 2)
        a = body * 0.9 + spike * 0.75 + core + halo
        if name == 'glint8':
            d1, d2 = (u + v) / 1.4142, (u - v) / 1.4142
            a += 0.45 * (np.exp(-(d2 / 0.02) ** 2) * np.clip(1 - np.abs(d1) / 0.6, 0, 1) ** 2.5 +
                         np.exp(-(d1 / 0.02) ** 2) * np.clip(1 - np.abs(d2) / 0.6, 0, 1) ** 2.5)
        a *= np.clip(1 - r ** 8, 0, 1)
        return _Spr(np.clip(a, 0, 1))
    if name in ('streak', 'streak8'):  # capsule streak, aspect 4 / 8 (long axis = x)
        asp = 4 if name == 'streak' else 8
        w, h = 32 * asp, 32
        u, v = _grid01(w, h)
        ux = np.clip(np.abs(u) * asp - (asp - 1), 0, None)   # distance past the segment, in v units
        d = np.hypot(ux, v)
        along = 1 - 0.55 * np.abs(u)                          # slightly brighter middle
        return _Spr(np.clip(1 - d, 0, 1) ** 1.8 * along)
    if name == 'drop':       # water droplet: soft disc with highlight shading
        u, v = _grid01(48)
        r = np.hypot(u, v)
        a = 1 - _sstep(0.62, 0.9, r)
        hl = np.exp(-(((u + 0.28) / 0.2) ** 2 + ((v + 0.3) / 0.16) ** 2))
        shade = 0.72 + 0.28 * (1 - r) + 0.9 * hl
        return _Spr(a, shade)
    if name == 'bokeh':      # out-of-focus disc, brighter rim
        u, v = _grid01(64)
        r = np.hypot(u, v)
        a = (0.55 + 0.45 * _sstep(0.5, 0.82, r)) * (1 - _sstep(0.8, 0.95, r))
        return _Spr(a)
    if name == 'ring':
        u, v = _grid01(128)
        r = np.hypot(u, v)
        return _Spr(np.exp(-((r - 0.8) / 0.05) ** 2) + 0.15 * np.exp(-((r - 0.8) / 0.14) ** 2))
    if name == 'hex':        # lens ghost
        u, v = _grid01(96)
        th = np.arctan2(v, u)
        r = np.hypot(u, v)
        k = np.cos(np.pi / 6) / np.cos((th % (np.pi / 3)) - np.pi / 6)
        rn = r / (0.86 * k)
        a = (0.4 + 0.6 * _sstep(0.55, 0.95, rn)) * (1 - _sstep(0.93, 1.0, rn))
        return _Spr(a)
    if name == 'burst':      # starburst (many thin rays) for lens flares
        n = 384
        u, v = _grid01(n)
        r = np.hypot(u, v) + 1e-6
        th = np.arctan2(v, u)
        g = np.random.default_rng(77)
        a = np.zeros_like(r)
        for k in range(60):
            ang = g.uniform(0, TAU)
            wd = g.uniform(0.0025, 0.007)          # half-width in sprite units (constant along the ray)
            ln = g.uniform(0.3, 1.0)
            amp = g.uniform(0.15, 0.8)
            d = np.angle(np.exp(1j * (th - ang))) * r
            a += amp * np.exp(-(d / wd) ** 2) * np.clip(1 - r / ln, 0, 1) ** 2.2 * _sstep(0.02, 0.12, r)
        a = a * 0.8 + 0.5 * np.exp(-(r / 0.05) ** 2)
        a *= np.clip(1 - r, 0, 1)
        return _Spr(np.clip(a, 0, 1))
    raise ValueError(name)


_SPR = {}


def _spr(name):
    s = _SPR.get(name)
    if s is None:
        s = _SPR[name] = _make_sprite(name)
    return s


_SAMP = skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear)


def _blit(c, name, x, y, size, rgb, a, rot=0.0, blend=ADD, bounds=None):
    """Draw N sprites of type `name` centred at (x,y), `size` = drawn width in
    design units, rotation `rot` (radians), tint rgb (3,) or (N,3), alpha (N,)."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    n = x.shape[0]
    if n == 0:
        return
    y = np.broadcast_to(np.asarray(y, dtype=float), (n,))
    size = np.broadcast_to(np.asarray(size, dtype=float), (n,))
    a = np.broadcast_to(np.asarray(a, dtype=float), (n,))
    rot = np.broadcast_to(np.asarray(rot, dtype=float), (n,))
    rgb = np.asarray(rgb, dtype=float)
    L, T, R, B = bounds if bounds is not None else _clipb(c)
    hs = size * 0.5
    m = (a > 0.004) & (size > 0.15) & (x + hs > L) & (x - hs < R) & (y + hs > T) & (y - hs < B)
    if not m.any():
        return
    x, y, size, a, rot = x[m], y[m], size[m], a[m], rot[m]
    if rgb.ndim == 2:
        rgb = rgb[m]
    s = _spr(name)
    sc = size / s.w
    cs, sn = sc * np.cos(rot), sc * np.sin(rot)
    hw, hh = s.w * 0.5, s.h * 0.5
    tx = x - (cs * hw - sn * hh)
    ty = y - (sn * hw + cs * hh)
    xf = [skia.RSXform(p, q, r_, t_) for p, q, r_, t_ in zip(cs.tolist(), sn.tolist(), tx.tolist(), ty.tolist())]
    c.drawAtlas(s.img, xf, [s.rect] * len(xf), _pack(rgb, a), skia.BlendMode.kModulate, _SAMP, None, _paint(blend))


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
def bezier_point(p0, p1, u, ctrl=None):
    """Point on a straight line p0->p1 (or quadratic bezier through ctrl) at u in [0,1]."""
    u = np.asarray(u, dtype=float)
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    if ctrl is None:
        return p0 + (p1 - p0) * u[..., None]
    c_ = np.asarray(ctrl, float)
    v = 1 - u
    return (v * v)[..., None] * p0 + (2 * v * u)[..., None] * c_ + (u * u)[..., None] * p1


def _catmull(points, n=160):
    """Catmull-Rom sample through points -> (n,2), resampled uniformly by arc length."""
    P = np.asarray(points, dtype=float)
    if len(P) < 2:
        return np.repeat(P[:1], n, 0)
    if len(P) == 2:
        u = np.linspace(0, 1, n)[:, None]
        return P[0] + (P[1] - P[0]) * u
    Pp = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    segs = len(P) - 1
    per = max(8, int(math.ceil(n * 2 / segs)))
    out = []
    t = np.linspace(0, 1, per, endpoint=False)[:, None]
    t2, t3 = t * t, t * t * t
    for i in range(segs):
        p0, p1, p2, p3 = Pp[i], Pp[i + 1], Pp[i + 2], Pp[i + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    out.append(P[-1:])
    Q = np.vstack(out)
    return _resample(Q, n)


def _resample(Q, n):
    d = np.hypot(*np.diff(Q, axis=0).T)
    s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] <= 1e-9:
        return np.repeat(Q[:1], n, 0)
    su = np.linspace(0, s[-1], n)
    return np.stack([np.interp(su, s, Q[:, 0]), np.interp(su, s, Q[:, 1])], -1)


def _arclen_table(P):
    d = np.hypot(*np.diff(P, axis=0).T)
    return np.concatenate([[0], np.cumsum(d)])


def _at_arclen(P, s_tab, s):
    s = np.clip(s, 0, s_tab[-1])
    return np.stack([np.interp(s, s_tab, P[:, 0]), np.interp(s, s_tab, P[:, 1])], -1)


# =============================================================================
# glow & glints
# =============================================================================
def glow(c, x, y, r, color='#ffd86b', intensity=1.0, core=True, core_color=None, sy=1.0, flicker=0.0, t=0.0,
         blend=ADD, halo=1.0):
    """Soft layered additive light blob.

    x, y, r     centre and outer radius of the halo (design units)
    color       halo colour; intensity 0..~2 (above 1 the core whitens / blooms more)
    core        draw a small white-hot core (radius 0.22 r) mixed from `core_color`
                (default: colour mixed towards white)
    sy          vertical squash (e.g. 0.3 for a light pool on water)
    flicker     0..1 amount of living flicker (uses t)
    halo        multiplier for the wide halo layer
    """
    if intensity <= 0 or r <= 0:
        return
    I = intensity
    if flicker:
        I *= 1 + flicker * (0.18 * noise1(t * 7.3, 5) + 0.08 * noise1(t * 23.0, 9))
    L, T, R, B = _clipb(c)
    if x + r < L or x - r > R or y + r * sy < T or y - r * sy > B:
        return
    rgb = _c3(color)
    parts = [_disc_part(x, y, r, rgb, min(1.0, 0.6 * I * halo), 'soft', sy)]
    if core:
        cr = _c3(core_color) if core_color is not None else _mixc(rgb, _WHITE, 0.55 + 0.2 * clamp(I - 1))
        parts.append(_disc_part(x, y, r * 0.3, cr, min(1.0, 0.8 * I), 'hot', sy, core_rgb=_WHITE if I > 1 else None))
    _multi_mesh(c, parts, blend)


def glint(c, x, y, size, color='#ffffff', intensity=1.0, rot=0.0, eight=False, halo=True):
    """A single 4-point (or 8-point) star glint of total width `size`."""
    if intensity <= 0 or size <= 0:
        return
    rgb = _c3(color)
    if halo:
        _blit(c, 'glow', x, y, size * 0.7, _mixc(rgb, rgb, 0), 0.35 * min(1, intensity))
    _blit(c, 'glint8' if eight else 'glint', x, y, size, _mixc(rgb, _WHITE, 0.35), min(1, intensity), rot)


def sparkles(c, t, x, y, radius=120, count=30, color='#fff4c0', seed=1, intensity=1.0, rise=20.0, size=1.0,
             radius_y=None, life=(0.9, 2.4), jitter=0.3, twinkle=1.0, rot=0.0, halo=0.5, drift=0.0):
    """Twinkling 4-point star glints scattered around (x, y), slowly rising.

    radius / radius_y   horizontal / vertical extent of the scatter ellipse
    count               number of simultaneous sparkle slots (each re-spawns at a new
                        random place every cycle)
    rise                distance each glint rises over its lifetime (design units)
    size                size multiplier (base glints ~8..46 units)
    life                (min, max) lifetime in seconds
    jitter              colour jitter (0 = all `color`; 1 = strong warm/cool/white variation)
    twinkle             amount of fast shimmer
    rot                 base rotation of the glints (radians)
    halo                strength of the coloured soft halo behind each glint
    drift               horizontal wander (units)
    """
    if intensity <= 0 or count <= 0:
        return
    n = int(count)
    i = np.arange(n)
    P = life[0] + (life[1] - life[0]) * _hash(i, seed * 31 + 1)
    ph = _hash(i, seed * 31 + 2)
    cyc = t / P + ph
    k = np.floor(cyc)
    u = cyc - k
    ry = radius if radius_y is None else radius_y
    h1, h2, h3, h4 = _h2(i, k, seed), _h2(i, k, seed + 101), _h2(i, k, seed + 202), _h2(i, k, seed + 303)
    rr = np.sqrt(h1) * (0.35 + 0.65 * h1 ** 0.3)
    ang = TAU * h2
    px = x + np.cos(ang) * rr * radius + drift * np.sin(t * 0.7 + i) * u
    py = y + np.sin(ang) * rr * ry - u * rise
    # envelope: quick flare, then slow decay; fast shimmer on top
    env = _sstep(0, 0.12, u) * (1 - _sstep(0.3, 1.0, u)) ** 1.3
    env = env * (1 - twinkle * 0.35 + twinkle * 0.35 * np.sin(t * (9 + 8 * h3) + TAU * h4))
    base = (14 + 52 * h3 ** 2.6) * size
    sz = base * (0.55 + 0.45 * env)
    # colour jitter: toward white, warm or cool
    rgb = _c3(color)
    warm, cool = _c3('#ffc978'), _c3('#bfe0ff')
    tint = np.where(h4[:, None] < 0.5, warm, cool)
    C = rgb + (tint - rgb) * (jitter * np.abs(h4 - 0.5) * 2)[:, None]
    C = C + (1 - C) * (0.25 + 0.3 * h1)[:, None]
    a = intensity * env
    r_ = rot + (h2 - 0.5) * 0.35 + 0.15 * np.sin(t * 0.6 + i)
    B = _clipb(c, 60)
    if halo > 0:
        _blit(c, 'glow', px, py, sz * 1.1, C, a * 0.5 * halo, bounds=B)
    _blit(c, 'glint', px, py, sz, C, a, r_, bounds=B)


def sparkle_burst(c, x, y, t_since, scale=1.0, color='#fff4c0', seed=15, count=10, life=0.9, intensity=1.0,
                  ring=1.0):
    """A quick radial twinkle burst ("blink" of a star, a catch, a landing).

    t_since     seconds since the burst (nothing drawn outside 0..life)
    scale       overall size (1 = central flare ~200 units wide)
    count       satellite glints flying outward
    ring        strength of the faint expanding ring (0 disables)
    """
    if t_since < 0 or t_since > life or intensity <= 0:
        return
    u = t_since / life
    rgb = _c3(color)
    # central flare: fast attack, exponential decay
    env = _sstep(0, 0.06 / life, u) * math.exp(-t_since * 5.0 / life)
    _disc(c, x, y, 120 * scale * (0.6 + 0.6 * env), rgb, 0.6 * env * intensity, 'soft')
    _blit(c, 'glint', x, y, 230 * scale * (0.4 + 0.8 * env), _mixc(rgb, _WHITE, 0.5), env * intensity,
          0.25 * t_since)
    _blit(c, 'glint8', x, y, 130 * scale * (0.3 + 0.7 * env), _mixc(rgb, _WHITE, 0.3), 0.7 * env * intensity,
          math.pi / 4 - 0.4 * t_since)
    if ring:
        rr = scale * (25 + 150 * ease_out(min(1.0, u * 1.4)))
        ra = ring * 0.22 * intensity * (1 - min(1.0, u * 1.4)) ** 2 * _sstep(0, 0.05, u)
        if ra > 0.004:
            w = 5 * scale * (1 + 1.5 * u)
            _mesh(c, *_ring_part(x, y, rr, [-3 * w, -w, 0, w * 0.6], np.array([0, .35, 1, 0]) * ra,
                                 _mixc(rgb, _WHITE, .4), 64))
    # satellites
    n = int(count)
    if n:
        i = np.arange(n)
        ang = TAU * (i + 0.6 * _hash(i, seed)) / n
        dist = scale * (60 + 110 * _hash(i, seed + 1)) * ease_out(min(1, u * 1.6))
        delay = 0.12 * _hash(i, seed + 2)
        uu = np.clip((u - delay) / (1 - delay), 0, 1)
        e = _sstep(0, 0.1, uu) * (1 - uu) ** 1.6 * (0.7 + 0.3 * np.sin(t_since * 40 + i * 2.1))
        sz = scale * (18 + 26 * _hash(i, seed + 3)) * (0.6 + 0.4 * e)
        px, py = x + np.cos(ang) * dist, y + np.sin(ang) * dist
        _blit(c, 'glint', px, py, sz, _mixc(rgb, _WHITE, 0.4), e * intensity, 0.3 * _hash(i, seed + 4))


def impact_flash(c, x, y, t_since, radius=200, color='#fff6e0', glow_color='#ffd36b', intensity=1.0, life=1.4,
                 streak=True, sy=1.0):
    """Brief blinding flash + a fading warm halo, a cross glint and an anamorphic streak.

    radius      size of the flash core region (halo reaches ~3x)
    life        total duration in seconds
    sy          vertical squash for the halo (e.g. 0.6 when it sits on the sea)
    """
    if t_since < 0 or t_since > life or intensity <= 0:
        return
    I = intensity
    fl = _sstep(0, 0.035, t_since) * math.exp(-t_since * 9.0)          # blinding flash
    ha = _sstep(0, 0.05, t_since) * math.exp(-t_since * 2.4) * (1 - _sstep(life * 0.6, life, t_since))
    rgb, grgb = _c3(color), _c3(glow_color)
    hrgb = _mixc(grgb, '#ffa040', 0.3)
    parts = []
    if ha > 0.003:
        parts.append(_disc_part(x, y, radius * (2.2 + 1.2 * t_since), hrgb, min(1, 0.6 * ha * I), 'soft', sy))
    if fl > 0.003:
        parts.append(_disc_part(x, y, radius * (0.5 + 0.9 * ease_out(t_since / 0.15)), rgb, min(1, fl * I * 1.2), 'hot',
                                sy, core_rgb=_WHITE))
    if streak and (fl + ha) > 0.01:
        e = fl * 0.9 + ha * 0.35
        wv = radius * 5.5 * (0.7 + 0.5 * e)
        xs = x + np.array([-1, -.5, -.2, -.06, 0, .06, .2, .5, 1]) * wv
        prof_x = np.array([0, .07, .3, .7, 1, .7, .3, .07, 0])
        hh = radius * 0.06 + 2
        ys = y + np.array([-1, -.4, 0, .4, 1]) * hh
        X, Y = np.meshgrid(xs, ys)
        A = np.array([0, .5, 1, .5, 0])[:, None] * prof_x[None, :] * min(1, e * I)
        parts.append((X, Y, _pack(_mixc(grgb, _WHITE, 0.6), A), _grid_idx(5, 9)))
    _multi_mesh(c, parts)
    e = fl + 0.4 * ha
    if e > 0.01:
        _blit(c, 'glint', x, y, radius * 3.2 * (0.5 + 0.7 * e), _mixc(grgb, _WHITE, 0.6), min(1, e * I), 0.0)


# =============================================================================
# meteor
# =============================================================================
def meteor(c, x0, y0, x1, y1, p, color='#fff1c8', tail=500, size=1.0, t=0.0, ctrl=None, duration=4.0,
           sparks=1.0, glow_color='#ffd36b', head_color='#ffffff', curl=1.0, seed=3, intensity=1.0, grow=1.0,
           linger=0.0, wisps=True):
    """Shooting / falling star travelling from (x0,y0) to (x1,y1); p = 0..1 progress along
    the path (arc-length, so constant speed for linear p). Bright white-gold head with glow
    and a cross glint, a long tapered tail that curls and flows behind it, shedding sparks
    that fall with gravity and fade.

    ctrl        optional (cx, cy) control point: the path becomes a quadratic bezier
                (e.g. curving down into the sea)
    tail        tail length along the path (design units)
    size        width multiplier (head glow ~150*size radius)
    t           scene time (animates the curl/flicker)
    duration    seconds the scene takes for p 0->1 (used to age the shed sparks)
    sparks      spark amount multiplier (0 disables)
    curl        amount of tail curl / smoke-like flow
    grow        head size at p=1 relative to p=0 (e.g. 2.5 = grows as it approaches)
    linger      seconds after p=1 during which the tail retracts into the end point
                (p may then exceed 1); 0 = draw nothing once p > 1 (old behaviour)
    """
    ext = linger / max(duration, 1e-3)
    if p < 0 or p > 1 + ext or intensity <= 0:
        return
    retract = 0.0
    if p > 1:
        retract = (p - 1) / max(ext, 1e-6)
        p = 1.0
    sz = size * lerp(1.0, grow, p)
    # path table
    u = np.linspace(0, 1, 200)
    Pp = bezier_point((x0, y0), (x1, y1), u, ctrl)
    s_tab = _arclen_table(Pp)
    Ltot = s_tab[-1]
    sh = p * Ltot
    Lt = min(tail * (1 - retract), sh)
    rgb, grgb, hrgb = _c3(color), _c3(glow_color), _c3(head_color)
    I = intensity * (1 - retract)
    parts = []
    if Lt > 4:
        M = 56
        d = np.linspace(0, Lt, M) ** 1.0
        dn = d / max(tail, 1)
        P = _at_arclen(Pp, s_tab, sh - d)
        nrm = _normals(P)
        # curl: displacement attached to world path position, growing with age
        sabs = sh - d
        amp = curl * sz * (6 + 70 * dn ** 1.4)
        off = amp * (_noise(sabs * 0.0045 + t * 0.35, seed) * 0.8 + _noise(sabs * 0.011 - t * 0.9, seed + 3) * 0.35)
        P = P + nrm * off[:, None]
        nrm = _normals(P)
        # when the tail is clamped at the start of the path, fade it out there
        endf = 1 - _sstep(Lt * 0.55, Lt, d) if Lt < tail * (1 - retract) - 1 else np.ones(M)
        fin = _sstep(0, 34 * sz, d)       # fade the ribbons in behind the head (no flat cut)
        # outer warm glow tail
        w_o = sz * (64 * (1 - dn) ** 0.8 + 8)
        a_o = 0.3 * I * (1 - dn) ** 1.3 * endf * fin
        parts.append(_ribbon_part(P, w_o, _X7, _P7, grgb, a_o, nrm))
        # main tail
        w_m = sz * (26 * (1 - dn) ** 1.1 + 2.0)
        a_m = 0.8 * I * (1 - dn) ** 1.6 * (0.85 + 0.15 * _noise(d * 0.05 - t * 12, seed + 7)) * endf * (0.3 + 0.7 * fin)
        cm = _mixc(rgb, grgb, np.clip(dn * 1.6, 0, 1) * 0.7)
        parts.append(_ribbon_part(P, w_m, _X5, _P5, cm, a_m, nrm))
        # white-hot core, shorter
        cl = np.clip(d / max(Lt * 0.5, 1), 0, 1)
        w_c = sz * (10 * (1 - cl) ** 1.2 + 1.2)
        parts.append(_ribbon_part(P, w_c, _X5, _P5, hrgb, I * (1 - cl) ** 2.2 * endf, nrm))
        # flowing wisps that peel off the tail
        if wisps and Lt > 60:
            for j in range(2):
                m2 = 30
                d2 = np.linspace(Lt * (0.08 + 0.1 * j), Lt, m2)
                dn2 = d2 / max(tail, 1)
                P2 = _at_arclen(Pp, s_tab, sh - d2)
                n2 = _normals(P2)
                s2 = sh - d2
                amp2 = curl * sz * (10 + 120 * ((d2 - d2[0]) / max(tail, 1)) ** 1.2)
                o2 = amp2 * _noise(s2 * 0.006 + t * 0.5 + j * 3.7, seed + 11 + j)
                o2 += (j - 1) * sz * 8 * (d2 / max(tail, 1))
                P2 = P2 + n2 * o2[:, None]
                wv = sz * (6.0 - j) * (1 - dn2 * 0.5)
                av = 0.24 * I * _sstep(d2[0], d2[0] + 40 * sz, d2) * (1 - dn2) ** 2 * (1 - _sstep(Lt * 0.6, Lt, d2))
                parts.append(_ribbon_part(P2, wv, _X5, _P5, _mixc(rgb, grgb, 0.4), av))
    _multi_mesh(c, parts)

    hx, hy = _at_arclen(Pp, s_tab, sh)
    # shed sparks: emitted uniformly in p, aged by (p - p_i) * duration
    ns = int(120 * sparks)
    if ns > 0:
        i = np.arange(ns)
        pe = (i + _hash(i, seed + 21)) / ns
        age = (p - pe) * duration + retract * linger
        life = 0.8 + 1.3 * _hash(i, seed + 22)
        m = (age >= 0) & (age < life)
        if m.any():
            i, pe, age, life = i[m], pe[m], age[m], life[m]
            se = pe * Ltot
            p0 = _at_arclen(Pp, s_tab, se)
            p1 = _at_arclen(Pp, s_tab, se + 2.0)
            tg = (p1 - p0)
            tg /= np.maximum(np.hypot(tg[:, 0], tg[:, 1]), 1e-6)[:, None]
            vpath = Ltot / max(duration, 1e-3)
            ang = TAU * _hash(i, seed + 23)
            sp = (40 + 150 * _hash(i, seed + 24)) * sz
            v0 = tg * (vpath * (0.05 + 0.2 * _hash(i, seed + 25)))[:, None] + np.stack([np.cos(ang), np.sin(ang)], -1) * sp[:, None]
            kd = 2.2
            dec = (1 - np.exp(-kd * age)) / kd
            g = 200 * sz
            pos = p0 + v0 * dec[:, None]
            pos[:, 1] += 0.5 * g * age * age
            vel = v0 * np.exp(-kd * age)[:, None]
            vel[:, 1] += g * age
            uu = age / life
            e = (1 - uu) ** 1.1 * _sstep(0, 0.06, age) * (0.6 + 0.4 * np.sin(age * 38 + i * 1.7))
            cs = _mixc(hrgb, grgb, np.clip(uu * 1.5, 0, 1))
            cs = cs + (_c3('#ff9a4a') - cs) * np.clip(uu * 1.2 - 0.4, 0, 1)[:, None]
            ssz = sz * (6 + 12 * _hash(i, seed + 26) ** 2) * (1 - 0.5 * uu)
            spd = np.hypot(vel[:, 0], vel[:, 1])
            B = _clipb(c, 50)
            _blit(c, 'glow', pos[:, 0], pos[:, 1], ssz * 3.0, cs, e * I * 0.35, bounds=B)
            _blit(c, 'streak', pos[:, 0], pos[:, 1], ssz * (1.2 + np.clip(spd / 120, 0, 3)), cs, e * I,
                  np.arctan2(vel[:, 1], vel[:, 0]), bounds=B)
            gl = _hash(i, seed + 27) < 0.3
            if gl.any():
                _blit(c, 'glint', pos[gl, 0], pos[gl, 1], ssz[gl] * 3.2, cs[gl], e[gl] * I, 0.2, bounds=B)
    # head
    hI = I * (1 + 0.06 * noise1(t * 17, seed))
    _multi_mesh(c, [_disc_part(hx, hy, 190 * sz, grgb, min(1, 0.45 * hI), 'soft'),
                    _disc_part(hx, hy, 64 * sz, rgb, min(1, 0.75 * hI), 'soft'),
                    _disc_part(hx, hy, 20 * sz, hrgb, min(1, hI), 'hot', core_rgb=_WHITE)])
    fl = 1 + 0.15 * noise1(t * 11, seed + 1)
    _blit(c, 'glint', hx, hy, 150 * sz * fl, _mixc(hrgb, rgb, 0.3), min(1, 0.95 * hI), 0.08 * noise1(t * 2, seed))


# =============================================================================
# water: splash, ripples, drips, bioluminescence
# =============================================================================
def splash(c, x, y, t_since, scale=1.0, color='#dff2ff', glow_color='#ffe7a0', seed=2, persp=0.3,
           intensity=1.0, height=1.0, life=3.2, mist=1.0, crown=1.0):
    """Water splash + (optional) burst of light where something falls into the sea at (x, y).

    t_since     seconds since impact (nothing drawn if < 0 or > life)
    scale       overall size (1 = spray column ~300 units high, crown ~200 units wide)
    color       water/spray colour (drawn with normal alpha blending)
    glow_color  colour of the magic light inside the splash; None = ordinary splash
                (no flash, no additive light) e.g. for the comedic tower-fall splash
    persp       vertical squash of horizontal motion (camera angle over the sea)
    height      column height multiplier;  crown = crown-finger amount
    mist        amount of hanging mist
    """
    ts = t_since
    if ts < 0 or ts > life or intensity <= 0:
        return
    s = scale
    g = 1500.0 * s
    magic = glow_color is not None
    wrgb = _c3(color)
    grgb = _c3(glow_color) if magic else None
    deep = _mixc(grgb, '#ff9d3a', 0.55) if magic else None
    B = _clipb(c, 120 * s)
    if x + 500 * s < B[0] or x - 500 * s > B[2] or y + 200 * s < B[1] or y - 600 * s > B[3]:
        return
    I = intensity
    foam = _mixc(wrgb, _WHITE, 0.45)

    # --- light: flash + light pool on the water ----------------------------------------
    if magic:
        impact_flash(c, x, y - 20 * s, ts, radius=120 * s, color=_mixc(grgb, _WHITE, 0.6), glow_color=deep, intensity=I,
                     life=min(life, 1.8), sy=0.85)
        pool = I * _sstep(0, 0.06, ts) * math.exp(-ts * 0.8)
        _multi_mesh(c, [_disc_part(x, y, 560 * s, deep, 0.5 * pool, 'soft', sy=persp),
                        _disc_part(x, y, 200 * s, grgb, 0.5 * pool, 'soft', sy=persp)])

    # --- jet column body -----------------------------------------------------------
    Hc = 300 * s * height
    v0 = math.sqrt(2 * g * Hc)
    top = v0 * ts - 0.5 * g * ts * ts
    ca = I * _sstep(0, 0.04, ts) * (1 - _sstep(0.25, 1.5, ts))
    if top > 4 * s and ca > 0.01:
        M = 12
        hv = np.linspace(0, 1, M)
        hh = hv * top
        wdt = s * (22 + 30 * (1 - hv) ** 1.5) * (1 + 0.9 * ts) * (1 - 0.6 * hv ** 3)
        wx = x + s * 7 * _noise(hh * 0.02 + ts * 2.5, seed + 3) * hv
        P = np.stack([wx, y - hh], -1)
        nrm = np.tile([[1.0, 0.0]], (M, 1))
        aa = ca * (0.35 + 0.65 * (1 - hv)) * (1 - _sstep(0.75, 1.0, hv))
        _mesh(c, *_ribbon_part(P, wdt, _X5, np.array([0, .55, .8, .55, 0]), foam, aa * (0.45 if magic else 0.75), nrm),
              blend=None)
        if magic:
            la = I * math.exp(-ts * 1.4) * _sstep(0, 0.05, ts)
            _multi_mesh(c, [_ribbon_part(P, wdt * 2.8, _X7, _P7, deep, aa * 0.4 * la / max(ca, 1e-3), nrm),
                            _ribbon_part(P, wdt * 0.6, _X5, _P5, grgb, aa * 0.45 * la / max(ca, 1e-3), nrm)])

    # --- crown fingers -------------------------------------------------------------------
    if crown > 0 and ts < 1.3:
        nf = 22
        k = np.arange(nf)
        th = TAU * (k + 0.5 * _hash(k, seed + 50)) / nf
        vr = s * (140 + 120 * _hash(k, seed + 51))
        vy = s * (330 + 260 * _hash(k, seed + 52)) * crown ** 0.5
        tau = ts - 0.03
        r0 = 30 * s
        parts, dx_, dy_, da_ = [], [], [], []
        for q in range(nf):
            hq = vy[q] * tau - 0.5 * g * tau * tau
            if tau <= 0 or hq < -2 * s:
                continue
            rad = r0 + vr[q] * tau
            ct, st = math.cos(th[q]), math.sin(th[q])
            bx, by = x + ct * (r0 + 0.35 * vr[q] * tau), y + st * (r0 + 0.35 * vr[q] * tau) * persp
            tx, ty = x + ct * rad, y + st * rad * persp - max(hq, 0)
            mx, my = bx + (tx - bx) * 0.3, by + (ty - by) * 0.75 - 0.15 * max(hq, 0)
            u = np.linspace(0, 1, 7)[:, None]
            P = (1 - u) ** 2 * np.array([bx, by]) + 2 * (1 - u) * u * np.array([mx, my]) + u * u * np.array([tx, ty])
            front = 0.55 + 0.45 * st
            fa = I * crown * front * (1 - _sstep(0.5, 1.2, ts)) * _sstep(0, 0.05, tau)
            wv = s * (9 * (1 - u[:, 0]) ** 0.8 + 2.2) * (1 - 0.3 * ts)
            parts.append(_ribbon_part(P, wv, _X5, np.array([0, .6, 1, .6, 0]), foam, fa * 0.7 * (0.5 + 0.5 * u[:, 0])))
            dx_.append(tx)
            dy_.append(ty)
            da_.append(fa)
        if parts:
            _multi_mesh(c, parts, blend=None)
            _blit(c, 'drop', np.array(dx_), np.array(dy_), 11 * s, foam, np.array(da_), blend=None, bounds=B)
            if magic:
                _blit(c, 'core', np.array(dx_), np.array(dy_), 16 * s, grgb, np.array(da_) * 0.5 * math.exp(-ts), bounds=B)

    # --- droplets: column + crown spray ------------------------------------------------
    rng_i = np.arange(240)
    kind = rng_i < 120                       # True = column spray, False = crown spray
    h = [_hash(rng_i, seed * 13 + k) for k in range(8)]
    ang = TAU * h[0]
    vy = np.where(kind, (480 + 540 * h[1] ** 0.7) * height, 240 + 300 * h[1]) * s
    vr = np.where(kind, 30 + 120 * h[2], 160 + 240 * h[2]) * s
    delay = np.where(kind, 0.02 + 0.12 * h[3], 0.03 + 0.1 * h[3])
    tt = ts - delay
    dx = np.cos(ang) * vr * tt
    dz = np.sin(ang) * vr * tt
    hh = vy * tt - 0.5 * g * tt * tt
    alive = (tt > 0) & (hh > -3 * s)
    r0 = np.where(kind, 14, 34) * s
    px = x + dx + np.cos(ang) * r0
    py = y - hh + (dz + np.sin(ang) * r0) * persp
    vx = np.cos(ang) * vr
    vyy = -(vy - g * tt) + np.sin(ang) * vr * persp
    spd = np.hypot(vx, vyy)
    dsz = s * np.where(kind, 5 + 8 * h[4], 4 + 7 * h[4])
    fade = (1 - _sstep(life * 0.6, life, ts)) * _sstep(-3 * s, 8 * s, hh)
    a = I * fade * (0.6 + 0.4 * h[5])
    m = alive & (a > 0.01)
    if m.any():
        rot = np.arctan2(vyy, vx)
        stretch = 1 + np.clip(spd / (450 * s), 0, 2.5)
        if magic:
            lit = np.exp(-np.hypot(dx, hh * 0.6) / (200 * s)) * math.exp(-ts * 0.5)
            dc = wrgb + (deep - wrgb) * np.clip(lit * 1.3, 0, 1)[:, None]
        else:
            lit = np.zeros_like(dx)
            dc = np.broadcast_to(foam, (len(px), 3))
        _blit(c, 'streak', px[m], py[m], (dsz * stretch * 1.4)[m], dc[m], a[m] * 0.85, rot[m], blend=None, bounds=B)
        big = m & (h[6] < 0.35)
        _blit(c, 'drop', px[big], py[big], dsz[big] * 1.5, dc[big], a[big], blend=None, bounds=B)
        if magic:
            gm = m & (h[6] < 0.55)
            _blit(c, 'core', px[gm], py[gm], dsz[gm] * 3.2, deep, a[gm] * 0.45 * (0.3 + lit[gm]), blend=ADD, bounds=B)
            gl = m & (h[7] < 0.14)
            _blit(c, 'glint', px[gl], py[gl], dsz[gl] * 6, _mixc(grgb, _WHITE, 0.5),
                  a[gl] * (0.5 + 0.5 * np.sin(ts * 30 + rng_i[gl])), 0, bounds=B)

    # --- foam on the water: patchy expanding ring + central boil ------------------------
    fa = I * _sstep(0.03, 0.25, ts) * (1 - _sstep(0.9, life, ts))
    if fa > 0.01:
        R = s * (45 + 200 * ease_out(clamp(ts / 2.4)))
        nfo = 30
        k = np.arange(nfo)
        th = TAU * (k + 0.7 * _hash(k, seed + 60)) / nfo
        rr = R * (0.85 + 0.25 * _hash(k, seed + 61))
        fx_ = x + np.cos(th) * rr
        fy_ = y + np.sin(th) * rr * persp
        fs = s * (40 + 50 * _hash(k, seed + 62)) * (0.7 + 0.5 * ts)
        fal = fa * (0.18 + 0.14 * _hash(k, seed + 63)) * (0.6 + 0.4 * np.sin(th))
        w = s * (8 + 12 * ts)
        amod = 0.55 + 0.45 * _noise(np.linspace(0, TAU, 64, endpoint=False) * 4 + seed, seed + 9)
        _mesh(c, *_ring_part(x, y, R, [-2.5 * w, -w, 0, w * 0.5], np.array([0, 0.2, 0.35, 0]) * fa, foam, 64, persp, amod),
              blend=None)
        _blit(c, 'glow', fx_, fy_, fs, foam, fal, blend=None, bounds=B)
        boil = I * _sstep(0.05, 0.3, ts) * (1 - _sstep(0.6, 2.4, ts))
        if boil > 0.01:
            _disc(c, x, y, s * (60 + 50 * ts), foam, 0.3 * boil, 'soft', sy=persp * 1.1, blend=None)

    # --- mist ----------------------------------------------------------------------
    if mist > 0:
        n = 16
        i = np.arange(n)
        mh = [_hash(i, seed * 7 + 40 + k) for k in range(4)]
        mt = ts - 0.1 - 0.35 * mh[0]
        mm = mt > 0
        if mm.any():
            ex = 1 - np.exp(-np.maximum(mt, 0) * 1.1)
            mx = x + (mh[1] - 0.5) * 300 * s * ex + 16 * s * mt
            my = y - (30 + 200 * mh[2] * height) * s * ex - 12 * s * mt
            msz = s * (120 + 180 * mh[3]) * (0.45 + 0.9 * ex)
            ma = I * mist * 0.085 * _sstep(0, 0.35, mt) * (1 - _sstep(0.6, life - 0.2, mt))
            _blit(c, 'glow', mx[mm], my[mm], msz[mm], foam, ma[mm], blend=None, bounds=B)
            if magic:
                _blit(c, 'glow', mx[mm], my[mm], msz[mm] * 0.8, deep, ma[mm] * 1.2 * math.exp(-ts * 0.7), bounds=B)


def ripples(c, x, y, t_since, color='#bfe0ff', intensity=1.0, persp=0.25, count=4, speed=120.0, life=4.0,
            width=5.0, interval=0.45, glow=True, seed=5, echo=True):
    """Expanding elliptical ripples on the water surface centred at (x, y).

    persp       vertical squash of the ellipses (0.25 = seen at a low angle)
    count       number of rings (one every `interval` seconds)
    speed       expansion speed (units/s, decelerating slightly)
    life        seconds each ring lives
    width       crest thickness (grows as the ring expands)
    glow        add a soft glow band around each crest
    echo        add a faint secondary trough/crest inside each ring
    """
    if t_since < 0 or intensity <= 0:
        return
    rgb = _c3(color)
    parts = []
    for k in range(int(count)):
        tt = t_since - k * interval
        if not (0 < tt < life):
            continue
        rr = speed * tt * (1 - 0.18 * tt / life) + 6
        a = intensity * (1 - tt / life) ** 1.4 * _sstep(0, 0.2, tt) * (1 - 0.12 * k)
        if a < 0.004:
            continue
        w = width * (1 + rr / 260)
        segs = int(clamp(rr * 0.35, 40, 150))
        th = np.linspace(0, TAU, segs, endpoint=False)
        amod = (0.55 + 0.45 * np.sin(th)) * (0.8 + 0.2 * _noise(th * 5 + k * 3.1 + t_since * 0.5, seed + k))
        parts.append(_ring_part(x, y, rr, [-1.6 * w, -0.6 * w, 0, 0.5 * w, 1.2 * w],
                                np.array([0, .45, 1, .35, 0]) * a, rgb, segs, persp, amod))
        if glow:
            parts.append(_ring_part(x, y, rr, [-5 * w, -2 * w, 0, 2 * w, 5 * w], np.array([0, .12, .2, .1, 0]) * a,
                                    rgb, segs, persp, amod))
        if echo and rr > 3 * w:
            parts.append(_ring_part(x, y, rr * 0.8, [-w, 0, w], np.array([0, .35, 0]) * a, rgb, segs, persp, amod))
    _multi_mesh(c, parts)


def water_drips(c, t, x, y, spread=30, rate=2.0, seed=17, fall=160, size=1.0, color='#cfe6ff', glint=None,
                intensity=1.0, spots=4):
    """Drips falling from a wet object whose bottom edge is at (x ± spread, y).

    rate        drops per second (all spots together)
    fall        distance a drop falls before vanishing (e.g. to the water)
    size        drop size multiplier (1 ≈ 7 units)
    color       water colour; glint = optional light colour for a sparkle on each drop
                (e.g. '#ffd36b' next to Guang or a lantern)
    spots       number of drip points along the bottom edge
    """
    if intensity <= 0 or rate <= 0:
        return
    n = max(1, int(spots))
    j = np.arange(n)
    sx = x + spread * (2 * _hash(j, seed) - 1)
    P = n / rate * (0.7 + 0.6 * _hash(j, seed + 1))
    ph = _hash(j, seed + 2)
    g = 1600.0 * size
    rgb = _c3(color)
    px, py, pr, pa, pst = [], [], [], [], []
    for back in (0, 1):
        cyc = t / P + ph
        k = np.floor(cyc) - back
        u = cyc - k                       # may be > 1 for the previous cycle
        form = 0.55
        rd = 4.5 * size * (0.7 + 0.6 * _h2(j, k, seed + 3))
        # forming bead
        if back == 0:
            m = u < form
            px += list(sx[m])
            py += list((y + rd * (u / form) * 0.9)[m])
            pr += list((rd * (0.35 + 0.65 * (u / form) ** 0.7))[m])
            pa += [1.0] * int(m.sum())
            pst += [1.0] * int(m.sum())
        tau = (u - form) * P
        d = 0.5 * g * tau * tau
        m = (tau > 0) & (d < fall)
        if m.any():
            v = g * tau
            px += list(sx[m])
            py += list((y + rd * 0.9 + d)[m])
            pr += list(rd[m])
            pa += list((1 - _sstep(fall * 0.75, fall, d))[m])
            pst += list((1 + np.clip(v / (900 * size), 0, 1.8))[m])
    if not px:
        return
    px, py, pr, pa, pst = map(np.asarray, (px, py, pr, pa, pst))
    for i in range(len(px)):
        with at(c, px[i], py[i]):
            r = pr[i]
            st = pst[i]
            # teardrop: round bottom, pointed top that stretches with speed
            path = skia.Path()
            path.moveTo(0, -r * (1.2 + 1.3 * (st - 1)) - r * 0.4)
            path.cubicTo(r * 0.35, -r * 0.9 * st, r, -r * 0.35, r, r * 0.1)
            path.cubicTo(r, r * 0.75, r * 0.5, r, 0, r)
            path.cubicTo(-r * 0.5, r, -r, r * 0.75, -r, r * 0.1)
            path.cubicTo(-r, -r * 0.35, -r * 0.35, -r * 0.9 * st, 0, -r * (1.2 + 1.3 * (st - 1)) - r * 0.4)
            path.close()
            c.drawPath(path, fill(rgb, 0.6 * pa[i] * intensity))
            c.drawPath(path, stroke('#ffffff', max(0.6, r * 0.18), 0.35 * pa[i] * intensity))
            c.drawCircle(-r * 0.35, -r * 0.1, r * 0.3, fill('#ffffff', 0.85 * pa[i] * intensity))
    if glint is not None:
        _blit(c, 'glint', px - pr * 0.3, py - pr * 0.4, pr * 5, _mixc(glint, _WHITE, 0.4),
              pa * intensity * (0.5 + 0.5 * np.sin(t * 9 + np.arange(len(px)))))


def bioluminescence(c, t, x, y, radius=400, intensity=1.0, persp=0.25, seed=4, color='#6ff0ff', t_since=None,
                    life=8.0, gold='#ffd36b', count=None, size=1.0, rings=3, horizon=None, patchy=1.0, glow=1.0):
    """Glowing plankton lighting up on the sea surface around (x, y) (elliptical area).

    t_since     time since the trigger (impact). None = whole field already lit (steady
                twinkling). Otherwise a bright ring front expands outward from the centre
                (reaching `radius` at ~0.35*life), each plankton flashes gold-white as the
                front passes then settles to twinkling cyan, then everything fades by `life`.
    life        seconds until the field has faded out again
    persp       vertical squash at the centre (0.25 = low angle over the sea)
    horizon     optional screen y of the horizon: enables true perspective (the far half
                compresses toward the horizon, far plankton get smaller, nothing is drawn
                above the horizon)
    color/gold  cool and warm plankton colours (cyan-gold)
    count       particle count (default scales with radius, ~1100 at 400)
    rings       number of secondary ring pulses that follow the front
    patchy      0..1 organic clumping of the plankton (drifting patches & filaments)
    glow        strength of the broad soft glow on the water
    """
    if intensity <= 0:
        return
    n = int(count if count is not None else clamp(1300 * (radius / 400) ** 0.8, 300, 2600))
    i = np.arange(n)
    h = [_hash(i, seed * 11 + k) for k in range(8)]
    persp_mode = horizon is not None and horizon < y
    if persp_mode:
        # sample in screen rows so the visible water is evenly populated
        D = max(1.0, (y - horizon) / max(persp, 1e-3))
        yrow = horizon + (y - horizon) * 9.0 * h[0] ** 1.4 + 0.5
        depth0 = (y - horizon) / (yrow - horizon)
        oz = D * (1 - depth0)
        chord = np.sqrt(np.maximum(radius * radius - oz * oz, 0))
        ox = (2 * h[1] - 1) * chord
        d = np.hypot(ox, oz)
        d = np.where(chord > 0, d, radius * 2)
    else:
        d = radius * np.sqrt(h[0]) ** 0.95
        th = TAU * h[1]
        ox, oz = np.cos(th) * d, np.sin(th) * d          # offsets on the water plane (z>0 = nearer)
    crgb, grgb = _c3(color), _c3(gold)

    def proj(ox_, oz_):
        if not persp_mode:
            return x + ox_, y + oz_ * persp, np.ones_like(ox_)
        D = max(1.0, (y - horizon) / max(persp, 1e-3))
        depth = 1.0 - oz_ / D                      # >1 farther, <1 nearer (0 = at the camera)
        depth = np.maximum(depth, 0.1)
        return x + ox_ / depth, horizon + (y - horizon) / depth, 1.0 / depth

    px, py, dscale = proj(ox, oz)
    ok = np.ones(n, bool) if not persp_mode else ((py > horizon + 1) & (d < radius))
    dep = np.minimum(dscale, 3.0) ** 0.8 if persp_mode else 1 + np.clip(oz / max(radius, 1), -1, 1) * 0.35
    Tf = 0.35 * life
    front_r = None
    if t_since is None:
        age = np.full(n, 10.0)
        act = np.ones(n)
        fade = 1.0
    else:
        ts = t_since
        if ts < 0 or ts > life:
            return
        front_r = radius * min(1.0, (ts / Tf)) ** 0.7 if ts < Tf else radius * (1 + 0.15 * (ts - Tf) / Tf)
        ta = Tf * (d / radius) ** (1 / 0.7)
        age = ts - ta
        act = (age > 0).astype(float)
        fade = 1 - _sstep(life * (0.45 + 0.3 * h[2]), life * (0.8 + 0.2 * h[2]), ts)
    # organic clumping: slowly drifting noise field
    if patchy > 0:
        fld = (_noise(ox * 0.006 + t * 0.05 + 3.1, seed + 1) * 0.6 + _noise(oz * 0.009 - t * 0.04 + ox * 0.002, seed + 2) * 0.4)
        clump = np.clip(0.55 + 0.9 * fld, 0.08, 1.4) ** 1.5
        clump = 1 - patchy + patchy * clump
    else:
        clump = 1.0
    tw = 0.68 + 0.32 * np.sin(t * (2 + 4 * h[3]) + TAU * h[4])
    flash = _sstep(0, 0.1, age) * (0.75 + 1.2 * np.exp(-np.maximum(age, 0) * 2.2))
    ring_boost = 0
    if front_r is not None and rings:
        for jj in range(1, int(rings) + 1):
            tsj = t_since - jj * 0.6
            if tsj <= 0:
                continue
            rj = radius * min(1.0, tsj / Tf) ** 0.7 if tsj < Tf else radius * (1 + 0.15 * (tsj - Tf) / Tf)
            ring_boost = ring_boost + 0.8 * np.exp(-((d - rj) / (0.035 * radius)) ** 2) * (1 - jj / (rings + 1)) * \
                np.exp(-np.maximum(age, 0) * 0.35)
    edge = 1 - _sstep(0.75, 1.0, d / radius)
    a = intensity * act * fade * edge * clump * (flash * tw + ring_boost) * ok
    warmish = h[5] < 0.2
    k_gold = np.where(warmish, 0.8, np.exp(-np.maximum(age, 0) * 1.1))
    C = crgb + (grgb - crgb) * np.clip(k_gold, 0, 1)[:, None]
    C = C + (1 - C) * (0.35 * np.exp(-np.maximum(age, 0) * 3))[:, None]
    sz = size * (3.5 + 10 * h[6] ** 2.2) * dep * (0.8 + 0.4 * np.exp(-np.maximum(age, 0) * 2))
    B = _clipb(c, 40)
    # broad glows on the water (drawn first)
    if glow > 0:
        if persp_mode:
            # screen-aligned perspective grid: rows = depth, cols = world x across the disc
            dfar, dnear = 1 + radius / D, max(0.12, 1 - radius / D)
            rows_ = 40
            inv = np.geomspace(1 / dfar, 1 / dnear, rows_)         # screen-y is linear in 1/depth
            dep_r = 1 / inv
            oxs = radius * np.sin(np.linspace(-np.pi / 2, np.pi / 2, 41))
            OZ = (D * (1 - dep_r))[:, None] * np.ones((1, len(oxs)))
            OX = np.ones((rows_, 1)) * oxs[None, :]
            GX = x + OX / dep_r[:, None]
            GY = np.broadcast_to((horizon + (y - horizon) / dep_r)[:, None], GX.shape)
            dd = np.hypot(OX, OZ) / radius
            gd = np.broadcast_to((1 / dep_r)[:, None], GX.shape)
            gidx = _grid_idx(rows_, len(oxs))
        else:
            rs = np.array([0, .1, .22, .36, .5, .64, .78, .9, 1.0])
            segs = 64
            thg = np.linspace(0, TAU, segs, endpoint=False)
            R2 = rs[:, None] * radius
            GX, GY, gd = proj(R2 * np.cos(thg)[None, :], R2 * np.sin(thg)[None, :])
            dd = rs[:, None] * np.ones((1, segs))
            gidx = _grid_idx(len(rs), segs, True)
        if front_r is None:
            A = 0.1 * intensity * glow * np.clip(1 - dd ** 2, 0, 1) ** 2
            Cg = crgb
        else:
            fr = front_r / radius
            band = np.exp(-((dd - fr) / 0.08) ** 2) * (1 - _sstep(Tf * 0.8, Tf * 1.7, t_since))
            inner = (dd < fr) * 1.0 * np.exp(-t_since * 0.25) * (1 - _sstep(life * 0.4, life, t_since))
            A = intensity * glow * (0.3 * band + 0.08 * inner * np.clip(1 - dd ** 2, 0, 1)) * _sstep(0, 0.1, t_since)
            A = A + intensity * glow * 0.22 * math.exp(-t_since * 0.7) * np.exp(-(dd / 0.3) ** 2) * _sstep(0, 0.1, t_since)
            Cg = _mixc(grgb, crgb, np.clip(dd / 0.4 + (t_since / life), 0, 1))
        A = A * (1 - _sstep(0.85, 1.0, dd))
        if persp_mode:
            A = A * np.minimum(1.0, 1.0 / gd) ** 1.2          # the spread-out near side is dimmer
        _mesh(c, GX, GY, _pack(Cg, A), gidx)
    _blit(c, 'glow', px, py, sz * 3.4, C, a * 0.32, bounds=B)
    _blit(c, 'core', px, py, sz, C, a, bounds=B)
    gl = (h[3] < 0.07)
    if gl.any():
        _blit(c, 'glint', px[gl], py[gl], sz[gl] * 4.5, C[gl], (a * (0.5 + 0.5 * np.sin(t * 5 + i)))[gl], 0, bounds=B)


# =============================================================================
# the climax pillar
# =============================================================================
def light_pillar(c, t, x, y_base, y_top=-3000, width=160, intensity=1.0, color='#fff0b8', core='#ffffff', seed=6,
                 birth=1.0, glow_color='#ffd36b', turbulence=1.0, motes=1.0, rings=1.0, flare=1.0, streaks=1.0,
                 top_fade=0.3, spread=0.15, atmosphere=1.0, outer_color='#ffb347'):
    """THE climax beam: a colossal vertical pillar of golden-white light shooting from
    (x, y_base) up to y_top: white-hot core, soft wide atmospheric falloff, rippling
    turbulent edges, upward-streaming filaments and bright motes, rising light rings and
    a flare at the base where it leaves the lamp.

    width       half-width of the main body at intensity 1 (design units)
    intensity   0..~2: build-up (<1: thinner, fainter), full (1), overdrive (>1: wider,
                whiter, blooms hard); fade by animating it down to 0
    birth       0..1 the moment the pillar shoots upward: the top travels from y_base to
                y_top (with a blazing tip) — keep 1 once it has arrived
    turbulence  edge ripple amount;  motes / rings / streaks / flare / atmosphere: layer
                amounts (0 disables a layer)
    top_fade    fraction of the height over which the pillar dissolves at the top
    spread      how much wider the pillar gets towards the top (0.15 = 15 %)
    outer_color colour of the far atmospheric glow (deeper gold reads warmer over blue)
    """
    I = float(intensity)
    if I <= 0.002 or birth <= 0:
        return
    Iv, Ix = min(I, 1.0), max(0.0, I - 1.0)
    be = 1 - (1 - clamp(birth)) ** 2.2
    yt = y_base + (y_top - y_base) * be
    Hh = max(1.0, y_base - yt)
    L, T, R, B = _clipb(c, 40)
    wb = width * (0.25 + 0.75 * Iv ** 0.8) * (1 + 0.3 * Ix)
    grgb, crgb, corergb, orgb = _c3(glow_color), _c3(color), _c3(core), _c3(outer_color)
    parts = []
    tv = t
    ya, yb = max(yt, T - 200), min(y_base, B + 200)
    if yb > ya and x + wb * 9 > L and x - wb * 9 < R:
        n = int(clamp((yb - ya) / 8, 24, 170))
        ys = np.linspace(ya, yb, n)
        hf = np.clip((y_base - ys) / Hh, 0, 1)                   # 0 base .. 1 top
        if birth >= 1:
            vfade = 1 - _sstep(1 - top_fade, 1.0, hf)
        else:
            vfade = 1 - 0.35 * _sstep(0.7, 1.0, hf)
        dy = y_base - ys
        vfade = vfade * _sstep(-0.1 * width, 0.12 * width, dy)
        vb = vfade * (1 - 0.2 * hf)
        # beam leaves the lens narrower, then opens up to full width
        wy = wb * (1 + spread * hf) * (0.3 + 0.7 * _sstep(-0.1 * width, 1.3 * width, dy) ** 0.7)
        wob = turbulence * wb * (0.07 * _noise((ys + tv * 520) * 0.0032, seed) + 0.03 * _noise((ys + tv * 900) * 0.01, seed + 1))
        # per-side edge ripple (monotonic in the offset -> no fold-over)
        rip_l = turbulence * (0.16 * _noise((ys + tv * 760) * 0.008, seed + 20) + 0.08 * _noise((ys + tv * 1400) * 0.021, seed + 23))
        rip_r = turbulence * (0.16 * _noise((ys + tv * 800) * 0.008, seed + 27) + 0.08 * _noise((ys + tv * 1450) * 0.021, seed + 30))
        # --- A: atmospheric falloff (very wide, faint) --------------------------------
        if atmosphere > 0:
            oa = np.array([-1, -.66, -.42, -.26, -.15, -.07, 0, .07, .15, .26, .42, .66, 1.0])
            pa = 1 / (1 + (np.abs(oa) / 0.14) ** 1.6)
            pa = (pa - pa[0]) / (1 - pa[0])
            wa = wy * 7.0
            X = x + wob[:, None] * 0.3 + oa[None, :] * wa[:, None]
            Y = np.broadcast_to(ys[:, None], X.shape)
            A = atmosphere * (0.34 * Iv + 0.16 * Ix) * vb[:, None] * pa[None, :]
            Ca = _mixc(grgb, orgb, np.clip(np.abs(oa) * 2.2, 0, 1))
            parts.append((X, Y, _pack(np.broadcast_to(Ca[None], X.shape + (3,)), A), _grid_idx(n, len(oa))))
            # mid halo hugging the body
            X = x + wob[:, None] * 0.8 + _X7[None, :] * (wy * 2.4)[:, None]
            A = atmosphere * (0.3 * Iv + 0.1 * Ix) * vb[:, None] * _P7[None, :]
            parts.append((X, np.broadcast_to(ys[:, None], X.shape), _pack(grgb, A), _grid_idx(n, 7)))
        # --- B: body with rippling edges ----------------------------------------------
        ob = np.linspace(-1, 1, 13)
        sc_ = 1 + np.where(ob[None, :] < 0, rip_l[:, None], rip_r[:, None]) * np.abs(ob[None, :]) ** 1.5
        X = x + wob[:, None] + ob[None, :] * wy[:, None] * sc_
        Y = np.broadcast_to(ys[:, None], X.shape)
        pb = np.exp(-(ob / 0.52) ** 2) * (1 - np.abs(ob) ** 4)
        # upward flowing shimmer + faint twisting bands
        shimmer = (0.8 + 0.2 * _noise((ys[:, None] + tv * 1500) * 0.018 + ob[None, :] * 2.3, seed + 40)) * \
                  (0.88 + 0.12 * np.sin((ys[:, None] + tv * 900) * 0.021 + ob[None, :] * 2.4))
        A = (0.5 * Iv + 0.2 * Ix) * vb[:, None] * pb[None, :] * shimmer
        Cb = _mixc(grgb, crgb, np.exp(-(ob / 0.4) ** 2))
        parts.append((X, Y, _pack(np.broadcast_to(Cb[None, :, :], X.shape + (3,)), A), _grid_idx(n, len(ob))))
        # --- C: white-hot core ---------------------------------------------------------
        wc = wy * (0.32 + 0.14 * Ix) * (1 + 0.08 * _noise((ys + tv * 2000) * 0.03, seed + 50))
        X = x + wob[:, None] * 0.7 + _X7[None, :] * wc[:, None]
        Y = np.broadcast_to(ys[:, None], X.shape)
        A = min(1.0, 0.9 * Iv + 0.3 * Ix) * vfade[:, None] * _P7[None, :]
        parts.append((X, Y, _pack(corergb, A), _grid_idx(n, 7)))
        # --- D: upward streaming filaments ---------------------------------------------
        ns = int(30 * streaks)
        if ns:
            j = np.arange(ns)
            hs = [_hash(j, seed + 60 + k) for k in range(6)]
            lane = (hs[0] * 2 - 1) * np.abs(hs[0] * 2 - 1) ** 0.3 * 0.7
            ln = 160 + 620 * hs[1]
            sp = 700 + 1500 * hs[2]
            per = Hh + ln
            yh = y_base - ((tv * sp + hs[3] * per) % per)
            m_ = 7
            fr = np.linspace(0, 1, m_)
            for q in range(ns):
                yy = yh[q] + fr * ln[q]
                if yy[-1] < max(ya, yt) or yy[0] > yb:
                    continue
                yy = np.clip(yy, yt, y_base)
                hfq = (y_base - yy) / Hh
                wq = wb * (1 + spread * hfq)
                ripq = np.interp(yy, ys, rip_l if lane[q] < 0 else rip_r)
                xq = x + lane[q] * wq * (1 + ripq * abs(lane[q]) ** 1.5) + np.interp(yy, ys, wob)
                P = np.stack([xq, yy], -1)
                ww = np.full(m_, 2.5 + 6 * hs[4][q]) * (1 + 0.5 * Ix)
                aq = (0.4 + 0.3 * hs[5][q]) * Iv * _sstep(0, 0.15, fr) * (1 - fr) ** 1.1 * \
                    (1 - _sstep(1 - top_fade, 1, hfq)) * (1 - abs(lane[q]) * 0.6) * _sstep(0, 0.1 * width, y_base - yy)
                parts.append(_ribbon_part(P, ww, _X3, _P3, _mixc(crgb, _WHITE, 0.5), aq, np.tile([[1.0, 0.0]], (m_, 1))))
    # --- rising rings --------------------------------------------------------------
    if rings > 0 and birth >= 0.6:
        Tr, gap = 2.6, 0.65
        for k in range(5):
            age = (t % gap) + k * gap
            if age > Tr:
                continue
            hr = Hh * 0.85 * (age / Tr) ** 1.8
            yr = y_base - hr
            ra = rings * 0.32 * Iv * (1 - age / Tr) ** 1.3 * _sstep(0, 0.25, age)
            if ra < 0.01 or yr < T - 100 or yr > B + 100:
                continue
            rr = wb * (1.2 + 1.6 * age / Tr)
            wr = 7 + 12 * age / Tr
            th = np.linspace(0, TAU, 72, endpoint=False)
            amod = (0.3 + 0.7 * (0.5 + 0.5 * np.sin(th))) * (0.75 + 0.25 * _noise(th * 3 + k * 5.3, seed + 70))
            parts.append(_ring_part(x, yr, rr, [-5 * wr, -2 * wr, -0.6 * wr, 0, 0.6 * wr, 2 * wr, 5 * wr],
                                    np.array([0, .18, .6, 1, .6, .18, 0]) * ra, _mixc(grgb, _WHITE, 0.3), 72, 0.2, amod))
    # --- base flare ------------------------------------------------------------------
    if flare > 0 and y_base > T - 800 and y_base < B + 800:
        fa = flare * (0.7 * Iv + 0.3 * Ix)
        parts.append(_disc_part(x, y_base, wb * 3.2, grgb, min(1, 0.42 * fa), 'soft'))
        parts.append(_disc_part(x, y_base, wb * 4.5, orgb, min(1, 0.3 * fa), 'soft', 0.22))
        parts.append(_disc_part(x, y_base, wb * 0.8, corergb, min(1, 0.95 * fa), 'hot', core_rgb=_WHITE))
        # horizontal anamorphic flare
        xs = x + np.array([-1, -.45, -.18, -.06, 0, .06, .18, .45, 1]) * wb * 8
        px_ = np.array([0, .1, .32, .7, 1, .7, .32, .1, 0])
        ys_ = y_base + np.array([-1, -.35, 0, .35, 1]) * (wb * 0.06 + 3)
        X, Y = np.meshgrid(xs, ys_)
        A = np.array([0, .5, 1, .5, 0])[:, None] * px_[None, :] * min(1, 0.55 * fa)
        parts.append((X, Y, _pack(_mixc(grgb, _WHITE, 0.6), A), _grid_idx(5, 9)))
    # --- blazing tip while shooting up ----------------------------------------------
    if birth < 1:
        ta = Iv * (1 - _sstep(0.85, 1.0, birth))
        parts.append(_disc_part(x, yt, wb * 3.0, grgb, 0.5 * ta, 'soft'))
        parts.append(_disc_part(x, yt, wb * 1.0, corergb, ta, 'hot', core_rgb=_WHITE))
    _multi_mesh(c, parts)
    if birth < 1:
        _blit(c, 'glint', x, yt, wb * 5, _mixc(grgb, _WHITE, 0.6), Iv * (1 - _sstep(0.85, 1.0, birth)), 0)
    # --- motes spiralling upward -----------------------------------------------------
    nm = int(90 * motes)
    if nm and birth > 0.2:
        j = np.arange(nm)
        hm = [_hash(j, seed + 90 + k) for k in range(8)]
        sp = 200 + 1100 * hm[0] ** 1.5
        yy = y_base - ((t * sp + hm[1] * Hh) % Hh)
        hfm = (y_base - yy) / Hh
        orb = wb * (0.25 + 2.6 * hm[2] ** 1.2) * (1 + spread * hfm)
        ph = TAU * hm[3] + t * (0.5 + 1.0 * hm[4]) * np.where(hm[5] < 0.5, 1, -1)
        mx = x + orb * np.sin(ph)
        front = 0.5 + 0.5 * np.cos(ph)
        ma = motes * Iv * (0.25 + 0.75 * front) * (1 - _sstep(1 - top_fade, 1, hfm)) * _sstep(0, 0.03, hfm) * \
            (0.5 + 0.5 * hm[6]) * (yy > yt)
        msz = 5 + 16 * hm[7] ** 2
        mc = _mixc(crgb, _WHITE, 0.4)
        Bd = (L, T, R, B)
        _blit(c, 'glow', mx, yy, msz * 3.0, grgb, ma * 0.3, bounds=Bd)
        _blit(c, 'streak', mx, yy, msz * (1.4 + sp / 1400), mc, ma * 0.9, -math.pi / 2, bounds=Bd)
        gl = hm[6] > 0.75
        _blit(c, 'glint', mx[gl], yy[gl], msz[gl] * 3.0, mc, ma[gl] * (0.6 + 0.4 * np.sin(t * 13 + j[gl])), 0, bounds=Bd)


# =============================================================================
# stardust stream
# =============================================================================
def _stream_frame(points, swirl_end, swirl_len, swirl_persp, swirl_radius, width, K=240):
    """centre line (K,2) of the stream incl. optional end spiral around the last point."""
    if swirl_end <= 0:
        return _catmull(points, K)
    E = np.asarray(points[-1], dtype=float)
    # the approach curve ends on the spiral's outer radius, then the spiral winds inward
    R0 = swirl_radius if swirl_radius is not None else 2.6 * width
    C = _catmull(points, K)
    dist = np.hypot(C[:, 0] - E[0], (C[:, 1] - E[1]) / max(swirl_persp, 0.05))
    inside = np.nonzero(dist < R0)[0]
    ke = int(inside[0]) if len(inside) else K - 1
    ke = max(ke, 8)
    A = _resample(C[:ke + 1], K)
    S0 = A[-1]
    v0 = S0 - E
    phi0 = math.atan2(v0[1] / max(swirl_persp, 0.05), v0[0])
    r0 = max(1.0, math.hypot(v0[0], v0[1] / max(swirl_persp, 0.05)))
    tg = A[-1] - A[-4]
    cr = v0[0] * tg[1] - v0[1] * tg[0]
    dirn = 1.0 if cr >= 0 else -1.0
    # spiral: length-balanced so particles keep a steady speed
    ns = K
    k = np.linspace(0, 1, ns)
    rad = r0 * (1 - k) ** 1.15
    phi = phi0 + dirn * TAU * swirl_end * (1 - (1 - k) ** 1.5)
    Sx = E[0] + rad * np.cos(phi)
    Sy = E[1] + rad * np.sin(phi) * swirl_persp
    S = np.stack([Sx, Sy], -1)
    # split the total length between approach and spiral by swirl_len (fraction of the path)
    la = _arclen_table(A)[-1]
    ls = _arclen_table(S)[-1]
    na = max(8, int(K * (1 - swirl_len)))
    nsp = max(8, K - na)
    return np.vstack([_resample(A, na), _resample(S, nsp)[1:]]), la, ls


def stardust_stream(c, t, points, density=1.0, color='#cfe8ff', width=40, speed=0.25, seed=8, intensity=1.0,
                    size=1.0, swirl_end=0.0, swirl_len=0.35, swirl_persp=0.45, swirl_radius=None, taper=0.35,
                    turbulence=1.0, twinkle=1.0, gold='#fff0b0', ribbon=1.0, helix=1.0, head=1.0):
    """A river of stardust flowing along the curve through `points` [(x,y), ...] from the
    first to the last point (e.g. from the whale down into Deng's chest).

    density     particle amount multiplier (1 ≈ 420 particles)
    width       half-width of the stream at its start (tapers to `taper`*width at the end)
    speed       fraction of the path travelled per second
    swirl_end   number of turns the stream spirals around the last point before pouring
                into it (0 = no spiral). swirl_len = fraction of the path (by particle
                time) spent in the spiral; swirl_persp = vertical squash of the spiral
                (a ring seen at an angle); swirl_radius = outer spiral radius
                (default 2.6*width)
    turbulence  sideways noise wander;  helix = strands twisting around the centre line
    twinkle     fraction/strength of glint twinkles
    gold        second colour mixed into the particles;  ribbon = glowing body amount
    head        glow at the destination point (0 disables)
    """
    if intensity <= 0 or len(points) < 2:
        return
    K = 240
    fr = _stream_frame(points, swirl_end, swirl_len, swirl_persp, swirl_radius, width, K)
    Cl = fr[0] if isinstance(fr, tuple) else fr
    Kc = len(Cl)
    # parameter u in [0,1] is by index (time), not by length: the spiral part gets
    # `swirl_len` of the time whatever its length
    uu = np.linspace(0, 1, Kc)
    nrm = _normals(Cl)
    crgb, grgb = _c3(color), _c3(gold)
    us = 1 - swirl_len if swirl_end > 0 else 1.0

    def wfun(u):
        w = width * (1 + (taper - 1) * np.clip(u / max(us, 1e-3), 0, 1))
        if swirl_end > 0:
            w = w * (1 - 0.75 * _sstep(us, 1.0, u))
        return w

    def at_u(u):
        f = np.clip(u, 0, 1) * (Kc - 1)
        return (np.stack([np.interp(f, np.arange(Kc), Cl[:, 0]), np.interp(f, np.arange(Kc), Cl[:, 1])], -1),
                np.interp(f, np.arange(Kc), nrm[:, 0]), np.interp(f, np.arange(Kc), nrm[:, 1]))

    if ribbon > 0:
        wv = wfun(uu)
        s_tab = _arclen_table(Cl)
        flow = 0.7 + 0.3 * np.sin(s_tab * 0.025 - t * 7.0)
        a = intensity * ribbon * flow * _sstep(0, 0.06, uu) * (1 - _sstep(0.92, 1.0, uu))
        _multi_mesh(c, [_ribbon_part(Cl, wv * 2.4, _X7, _P7, _mixc(crgb, grgb, 0.2), a * 0.13, nrm),
                        _ribbon_part(Cl, wv * 0.9, _X5, _P5, _mixc(crgb, _WHITE, 0.4), a * 0.2, nrm)])
    n = int(420 * density)
    if n <= 0:
        return
    i = np.arange(n)
    h = [_hash(i, seed * 17 + k) for k in range(11)]
    spf = 0.8 + 0.4 * h[0]
    u = (h[1] + t * speed * spf) % 1.0
    P, Nx, Ny = at_u(u)
    w = wfun(u)
    lane = np.clip(np.sqrt(-2 * np.log(np.maximum(h[2], 1e-6))) * np.cos(TAU * h[3]) * 0.42, -1.5, 1.5)
    hph = TAU * h[4] + u * TAU * 3 * helix + t * 1.3 * helix
    off = w * (lane * (1 - 0.6 * helix) + 0.6 * helix * np.sign(lane + 1e-9) * np.abs(lane) ** 0.5 * np.sin(hph))
    off = off + turbulence * w * 0.45 * _noise(u * 9 + t * 0.8 + i * 0.37, seed + 3)
    px = P[:, 0] + Nx * off
    py = P[:, 1] + Ny * off
    depth = (0.7 + 0.3 * np.cos(hph)) if helix else 1.0
    env = _sstep(0, 0.05, u) * (1 - _sstep(0.95, 1.0, u))
    tw = 1 - 0.4 * twinkle + 0.4 * twinkle * np.sin(t * (6 + 9 * h[5]) + TAU * h[6])
    a = intensity * env * depth * tw * (0.55 + 0.45 * h[7])
    sz = size * (4 + 11 * h[8] ** 2.5) * (1 - 0.35 * u)
    Cc = crgb + (grgb - crgb) * (h[9] ** 2)[:, None]
    Cc = Cc + (1 - Cc) * 0.25
    B = _clipb(c, 50)
    # direction of motion for short flow streaks
    P2, _, _ = at_u(np.minimum(u + 0.004, 1))
    ang = np.arctan2(P2[:, 1] - P[:, 1], P2[:, 0] - P[:, 0])
    _blit(c, 'glow', px, py, sz * 3.6, Cc, a * 0.3, bounds=B)
    st = h[10] < 0.2
    _blit(c, 'streak', px[st], py[st], (sz * 2.6)[st], Cc[st], a[st] * 0.8, ang[st], bounds=B)
    _blit(c, 'core', px[~st], py[~st], (sz * 1.5)[~st], Cc[~st], a[~st], bounds=B)
    gl = h[5] < 0.12 * twinkle
    if gl.any():
        flash = np.clip(np.sin(t * (2 + 3 * h[6]) + TAU * h[7]), 0, 1) ** 4
        _blit(c, 'glint', px[gl], py[gl], (sz * 6)[gl], Cc[gl], (a * flash)[gl], 0, bounds=B)
    if swirl_end > 0 and head > 0:
        E = points[-1]
        glow(c, E[0], E[1], width * 2.2, color, 0.5 * intensity * head, core=True)


# =============================================================================
# rings, flares, rays
# =============================================================================
def shockwave(c, x, y, t_since, color='#fff4d0', max_r=900, life=1.2, persp=1.0, width=None, intensity=1.0,
              seed=12, chroma=True, echo=True):
    """Expanding ring of light (ignition moments) with a sharp bright leading edge, a soft
    trailing wake inside it and a faint chromatic fringe (reads like air distortion).

    persp       vertical squash (1 = circle facing camera, 0.25 = ring lying on the sea)
    width       edge thickness (default 3% of max_r, grows as it expands)
    """
    if not (0 <= t_since <= life) or intensity <= 0:
        return
    u = t_since / life
    R = max_r * (1 - (1 - u) ** 2.6)
    a = intensity * (1 - u) ** 1.3 * _sstep(0, 0.08, u)
    w = (width if width is not None else 0.03 * max_r) * (0.5 + 1.0 * u)
    w = min(w, R / 9.0 + 1.0)
    rgb = _c3(color)
    segs = int(clamp(R * 0.3, 48, 180))
    th = np.linspace(0, TAU, segs, endpoint=False)
    amod = 0.72 + 0.28 * _noise(th * 6 + seed, seed) * 0.5 + 0.28 * _noise(th * 17 + t_since * 3, seed + 1) * 0.5
    parts = [_ring_part(x, y, R, [-7 * w, -3.5 * w, -1.6 * w, -0.6 * w, 0, 0.35 * w, 1.0 * w],
                        np.array([0, .07, .2, .55, 1, .45, 0]) * a, _mixc(rgb, _WHITE, 0.3), segs, persp, amod,
                        rgbs=np.array([_c3('#ffb347'), _c3('#ffc870'), rgb, rgb, _mixc(rgb, _WHITE, .7), _c3('#bfe6ff'), _c3('#9fc4ff')]) if chroma else None)]
    if echo and u > 0.05:
        u2 = (t_since - 0.12 * life) / life
        if u2 > 0:
            R2 = max_r * 0.86 * (1 - (1 - u2) ** 2.6)
            a2 = a * 0.35
            parts.append(_ring_part(x, y, R2, [-2 * w, -0.5 * w, 0, 0.5 * w], np.array([0, .5, 1, 0]) * a2, rgb, segs,
                                    persp, amod))
    _multi_mesh(c, parts)


def lens_flare(c, x, y, intensity=1.0, color='#ffe7b0', cx=960, cy=540, t=0.0, streak=1.0, ghosts=1.0,
               starburst=1.0, halo=1.0, streak_color=None, streak_len=1400, rot=0.0, seed=13):
    """Screen-space flare for a very bright light at (x, y): horizontal anamorphic streak,
    a starburst, a faint halo ring and a chain of ghosts mirrored through (cx, cy).
    Draw it in SCREEN coordinates (outside Camera.apply), after the scene.

    streak / ghosts / starburst / halo   layer amounts (0 disables)
    streak_color   default: colour mixed toward cool white
    streak_len     half-length of the anamorphic streak
    """
    if intensity <= 0:
        return
    I = intensity
    rgb = _c3(color)
    srgb = _c3(streak_color) if streak_color is not None else _mixc(rgb, _c3('#dfe8ff'), 0.45)
    parts = []
    if streak > 0:
        for (hh, aa, ln) in ((3.0, 0.9, 1.0), (16, 0.28, 0.75), (60, 0.08, 0.5)):
            xs = x + np.array([-1, -.55, -.28, -.12, -.04, 0, .04, .12, .28, .55, 1]) * streak_len * ln
            px_ = np.array([0, .08, .22, .5, .85, 1, .85, .5, .22, .08, 0])
            ys = y + np.array([-1, -.4, 0, .4, 1]) * hh
            X, Y = np.meshgrid(xs, ys)
            A = np.array([0, .45, 1, .45, 0])[:, None] * px_[None, :] * min(1, aa * I * streak)
            parts.append((X, Y, _pack(srgb, A), _grid_idx(5, 11)))
    if halo > 0:
        parts.append(_ring_part(x, y, 230 * (0.8 + 0.2 * I), [-40, -12, 0, 12, 40], np.array([0, .3, 1, .3, 0]) * 0.045 * I * halo,
                                _mixc(rgb, _c3('#c8e0ff'), 0.4), 96))
    parts.append(_disc_part(x, y, 260 * (0.6 + 0.4 * I), rgb, min(1, 0.35 * I), 'soft'))
    _multi_mesh(c, parts)
    if starburst > 0:
        _blit(c, 'burst', x, y, 620 * (0.6 + 0.4 * min(I, 2)), _mixc(rgb, _WHITE, 0.5), min(1, 0.5 * I * starburst),
              rot + 0.0006 * (x + y) + 0.03 * t)
    if ghosts > 0:
        vx, vy = cx - x, cy - y
        g = [(0.45, 60, '#ffcf80', 'hex', 0.06), (0.7, 30, '#bfe6ff', 'bokeh', 0.08), (1.05, 110, '#c8a0ff', 'hex', 0.035),
             (1.35, 48, '#9fffe0', 'bokeh', 0.05), (1.7, 160, '#ffd36b', 'ring', 0.05), (2.0, 24, '#ffffff', 'bokeh', 0.1),
             (-0.25, 40, '#ffb0a0', 'bokeh', 0.05)]
        for f_, sz, cc, kind, aa in g:
            _blit(c, kind, x + vx * f_, y + vy * f_, sz * 2, _c3(cc), min(1, aa * I * ghosts), rot)


def god_rays(c, t, x, y, count=12, length=1400, spread=math.pi * 2, angle=0.0, color='#fff0c0', intensity=1.0,
             seed=9, width=1.0, start=0.0, rot_speed=0.015, shimmer=1.0, sy=1.0, source=True):
    """Soft radiating shafts of light from (x, y) (divine / reboot / sunrise moments),
    slowly rotating and shimmering, broad and overlapping like light through haze.

    count       number of shafts (each ray is a soft wedge; a few thin bright ones are mixed in)
    spread      angular range (2π = all around; e.g. π/2 fan centred on `angle`)
    angle       centre direction (radians, 0 = +x, -π/2 = up)
    width       shaft width multiplier;  start = inner radius where rays begin
    rot_speed   slow rotation (rad/s);  shimmer = amount of slow flicker per shaft
    sy          vertical squash (rays over a surface)
    source      add a soft glow at the source
    """
    if intensity <= 0:
        return
    n = int(count)
    i = np.arange(n)
    h = [_hash(i, seed + k) for k in range(7)]
    full = spread >= TAU - 1e-3
    base = angle - spread / 2 + spread * (i + 0.5 + (h[0] - 0.5) * 0.9) / n
    wob = np.array([noise1(t * 0.22 + k * 3.3, seed) for k in range(n)])
    ang = base + t * rot_speed + 0.05 * wob
    rgb = _c3(color)
    rows = np.array([0, .03, .08, .16, .27, .4, .55, .7, .85, 1.0])
    parts = []
    thin = h[5] < 0.25
    hw = np.where(thin, 0.018 + 0.02 * h[1], 0.07 + 0.14 * h[1]) * width * (max(spread, 0.3) / TAU) ** 0.35
    ln = length * (0.6 + 0.4 * h[2]) * (1 + 0.08 * np.array([noise1(t * 0.4 + k, seed + 5) for k in range(n)]))
    sh = np.array([noise1(t * (0.3 + 0.35 * h[3][k]) + k * 7.1, seed + 9) for k in range(n)])
    al = intensity * np.where(thin, 0.14 + 0.1 * h[4], 0.09 + 0.1 * h[4]) * (1 - shimmer * 0.6 + shimmer * 0.6 * (0.5 + 0.5 * sh))
    if not full:
        edge = np.abs((base - angle) / (spread / 2))
        al = al * (1 - _sstep(0.6, 1.0, edge))
    cross = np.array([0, .35, .8, 1, .8, .35, 0])
    offs0 = np.array([-1, -0.62, -0.3, 0, 0.3, 0.62, 1.0])
    for k in range(n):
        r = start + rows * (ln[k] - start)
        along = _sstep(0.0, 0.2, rows) * (1 - rows) ** 1.4 * (0.85 + 0.15 * np.sin(rows * 9 + t * 0.7 + k))
        A = ang[k] + offs0 * hw[k]
        X = x + r[:, None] * np.cos(A)[None, :]
        Y = y + r[:, None] * np.sin(A)[None, :] * sy
        parts.append((X, Y, _pack(rgb, al[k] * along[:, None] * cross[None, :]), _grid_idx(len(rows), 7)))
    if source:
        parts.append(_disc_part(x, y, max(length * 0.32, start * 1.5), rgb, 0.3 * intensity, 'soft', sy))
    _multi_mesh(c, parts)


# =============================================================================
# small particles: dust, embers, speed lines
# =============================================================================
def dust_motes(c, t, rect=(0, 0, 1920, 1080), count=60, color='#ffe9b0', intensity=0.5, seed=10, size=1.0,
               light=None, bokeh=0.12, drift=(5.0, -2.0), twinkle=1.0):
    """Slow floating dust in a light beam / interior.

    rect        (x, y, w, h) area the motes live in (they wrap around inside it)
    light       optional (lx, ly, radius): motes are brighter near this light
    bokeh       fraction of big out-of-focus foreground motes
    drift       mean drift velocity (units/s)
    twinkle     how strongly motes glint as they tumble through the light
    """
    if intensity <= 0 or count <= 0:
        return
    n = int(count)
    i = np.arange(n)
    h = [_hash(i, seed + k) for k in range(10)]
    x0, y0, w, hh = rect
    px = (h[0] * w + t * (drift[0] * (0.5 + h[1])) + 22 * np.sin(t * (0.2 + 0.3 * h[2]) + TAU * h[3])
          + 14 * _noise(t * 0.15 + i * 1.7, seed)) % w + x0
    py = (h[4] * hh + t * (drift[1] * (0.5 + h[5])) + 16 * np.sin(t * (0.15 + 0.25 * h[6]) + TAU * h[7])
          + 10 * _noise(t * 0.13 + i * 2.3, seed + 1)) % hh + y0
    glint_ = np.clip(np.sin(t * (0.8 + 1.6 * h[8]) + TAU * h[9]), 0, 1) ** 6
    a = intensity * (0.45 + 0.45 * h[2] + twinkle * 1.0 * glint_)
    if light is not None:
        lx, ly, lr = light
        a = a * (0.08 + 0.92 * np.exp(-((px - lx) ** 2 + (py - ly) ** 2) / (lr * lr)))
    big = h[6] < bokeh
    rgb = _c3(color)
    B = _clipb(c, 60)
    sm = ~big
    sz = size * (3.0 + 6.0 * h[3] ** 2)
    _blit(c, 'glow', px[sm], py[sm], sz[sm] * 4.0, rgb, a[sm] * 0.25, bounds=B)
    _blit(c, 'core', px[sm], py[sm], sz[sm] * 2.6, rgb, a[sm], bounds=B)
    if big.any():
        bs = size * (20 + 34 * h[1][big])
        _blit(c, 'bokeh', px[big], py[big], bs, rgb, a[big] * 0.14, bounds=B)
    gl = sm & (glint_ > 0.3) & (h[0] < 0.3)
    if gl.any():
        _blit(c, 'glint', px[gl], py[gl], sz[gl] * 7, rgb, a[gl] * glint_[gl] * 0.8, 0, bounds=B)


def embers(c, t, x, y, count=20, color='#ffa040', seed=11, intensity=1.0, spread=60, rise=180, size=1.0,
           life=(1.2, 2.6), wind=0.0, hot='#ffe7a0'):
    """Rising warm sparks / light specks (heart removal, lamp ignition, flames).

    spread      horizontal spawn half-width;  rise = height travelled over a lifetime
    life        (min, max) seconds;  wind = sideways drift (units over a lifetime)
    hot         colour of a freshly born ember (they cool to `color`, then deep red)
    """
    if intensity <= 0 or count <= 0:
        return
    n = int(count)
    i = np.arange(n)
    P = life[0] + (life[1] - life[0]) * _hash(i, seed + 1)
    cyc = t / P + _hash(i, seed + 2)
    k = np.floor(cyc)
    u = cyc - k
    h1, h2, h3 = _h2(i, k, seed + 3), _h2(i, k, seed + 4), _h2(i, k, seed + 5)
    rise_i = rise * (0.6 + 0.6 * h2)
    yy = y - rise_i * (1 - (1 - u) ** 1.7)
    sway = 14 * size * np.sin(t * (1.5 + 2 * h3) + TAU * h1) * u
    xx = x + spread * (2 * h1 - 1) * (1 + 0.5 * u) + sway + wind * u ** 1.5 + 10 * _noise(t * 0.9 + i * 3.1, seed) * u
    vy = -rise_i * 1.7 * (1 - u) ** 0.7 / P
    vx = wind * 1.5 * np.sqrt(u) / P + 14 * size * np.cos(t * (1.5 + 2 * h3) + TAU * h1) * (1.5 + 2 * h3) * u
    fl = 0.6 + 0.4 * _noise(t * 14 + i * 5.3, seed + 7)
    a = intensity * _sstep(0, 0.07, u) * (1 - u) ** 1.1 * fl
    C = _mixc(hot, color, np.clip(u * 1.6, 0, 1))
    C = C + (_c3('#e0461e') - C) * np.clip(u * 1.5 - 0.6, 0, 1)[:, None]
    sz = size * (6 + 8 * h2 ** 1.5) * (1 - 0.45 * u)
    spd = np.hypot(vx, vy)
    B = _clipb(c, 40)
    _blit(c, 'glow', xx, yy, sz * 5.5, C, a * 0.45, bounds=B)
    _blit(c, 'core', xx, yy, sz * 2.4, _mixc(C, _WHITE, 0.25), a, bounds=B)
    _blit(c, 'streak', xx, yy, sz * (1.1 + np.clip(spd / 140, 0, 1.6)), C, a * 0.8,
          np.arctan2(vy, vx), bounds=B)


def speed_lines(c, t, direction=0.0, intensity=1.0, rect=(0, 0, 1920, 1080), count=30, color='#fff4e0', seed=14,
                length=(220, 700), width=(3.0, 10.0), speed=3200.0, center=None, blend=ADD, alpha=0.55, smears=6):
    """Comic motion smears (the hurried stairs run).

    direction   direction of MOTION: angle in radians (0 = moving right, π/2 = moving down)
                or a (dx, dy) vector. Lines streak parallel to it and stream backwards.
    rect        (x, y, w, h) area filled with lines
    center      optional (cx, cy): radial "focus lines" bursting outward from this point
                instead of parallel lines
    blend       ADD (glowy, default) or None for plain translucent strokes
    alpha       peak opacity of the lines (times intensity)
    smears      number of wide faint motion-blur bands mixed in (parallel mode)
    """
    if intensity <= 0 or count <= 0:
        return
    if isinstance(direction, (tuple, list)):
        direction = math.atan2(direction[1], direction[0])
    n = int(count)
    i = np.arange(n)
    per = 0.22 + 0.3 * _hash(i, seed)
    cyc = t / per + _hash(i, seed + 1)
    k = np.floor(cyc)
    u = cyc - k
    h1, h2, h3, h4 = _h2(i, k, seed + 2), _h2(i, k, seed + 3), _h2(i, k, seed + 4), _h2(i, k, seed + 5)
    x0, y0, w, hh = rect
    ln = length[0] + (length[1] - length[0]) * h3
    wd = width[0] + (width[1] - width[0]) * h4
    rgb = _c3(color)
    parts = []
    env = np.sin(np.pi * u) ** 0.8
    m_ = 7
    fr = np.linspace(0, 1, m_)
    for q in range(n):
        if center is None:
            dx, dy = math.cos(direction), math.sin(direction)
            cxr, cyr = x0 + w * h1[q], y0 + hh * h2[q]
            # travel backwards across the rect
            trav = (u[q] - 0.5) * speed * per[q]
            hx, hy = cxr - dx * trav, cyr - dy * trav
            tx, ty = hx - dx * ln[q], hy - dy * ln[q]
        else:
            cx_, cy_ = center
            ang = TAU * h1[q]
            dx, dy = math.cos(ang), math.sin(ang)
            r0 = 0.25 * max(w, hh) * (0.6 + 0.8 * h2[q]) + u[q] * speed * per[q] * 0.3
            tx, ty = cx_ + dx * r0, cy_ + dy * r0
            hx, hy = tx + dx * ln[q], ty + dy * ln[q]
        P = np.stack([hx + (tx - hx) * fr, hy + (ty - hy) * fr], -1)
        ww = wd[q] * np.sin(np.pi * np.clip(fr * 0.85 + 0.08, 0, 1)) ** 0.7 + 0.8
        aq = intensity * alpha * env[q] * (1 - fr) ** 0.8 * _sstep(0, 0.12, fr + 0.02)
        nx, ny = -(ty - hy), (tx - hx)
        nn = math.hypot(nx, ny) or 1
        parts.append(_ribbon_part(P, ww, _X3, _P3, rgb, aq, np.tile([[nx / nn, ny / nn]], (m_, 1))))
    if smears and center is None:
        dx, dy = math.cos(direction), math.sin(direction)
        diag = math.hypot(w, hh)
        for q in range(int(smears)):
            per_s = 0.35 + 0.3 * _hash(q, seed + 40)
            cy_ = t / per_s + _hash(q, seed + 41)
            ks = math.floor(cy_)
            us = cy_ - ks
            hs1, hs2 = float(_h2(q, ks, seed + 42)), float(_h2(q, ks, seed + 43))
            cx0, cy0 = x0 + w * hs1, y0 + hh * hs2
            trav = (us - 0.5) * speed * per_s * 0.7
            hx, hy = cx0 - dx * trav, cy0 - dy * trav
            L_ = diag * (0.35 + 0.3 * hs2)
            P = np.stack([hx - dx * L_ * fr, hy - dy * L_ * fr], -1)
            ww = (25 + 45 * hs1) * np.sin(np.pi * np.clip(fr, 0.02, 0.98)) ** 0.5
            aq = intensity * alpha * 0.2 * math.sin(math.pi * us) * np.sin(np.pi * fr)
            parts.append(_ribbon_part(P, ww, _X5, _P5, rgb, aq, np.tile([[-dy, dx]], (m_, 1))))
    with_clip = skia.Rect.MakeXYWH(*rect)
    c.save()
    c.clipRect(with_clip)
    _multi_mesh(c, parts, blend)
    c.restore()


# =============================================================================
# heart core (standalone object) & light trail
# =============================================================================
def _star_path(R, r_in, rot=-math.pi / 2, round_=0.35, n=5):
    """Rounded n-point star path centred at 0,0."""
    pts = []
    for k in range(n * 2):
        a = rot + k * math.pi / n
        rr = R if k % 2 == 0 else r_in
        pts.append((math.cos(a) * rr, math.sin(a) * rr))
    path = skia.Path()
    m = len(pts)
    for k in range(m):
        p0 = pts[k - 1]
        p1 = pts[k]
        p2 = pts[(k + 1) % m]
        a0 = (p1[0] + (p0[0] - p1[0]) * round_, p1[1] + (p0[1] - p1[1]) * round_)
        a1 = (p1[0] + (p2[0] - p1[0]) * round_, p1[1] + (p2[1] - p1[1]) * round_)
        if k == 0:
            path.moveTo(*a0)
        else:
            path.lineTo(*a0)
        path.quadTo(p1[0], p1[1], a1[0], a1[1])
    path.close()
    return path


def _flame_path(r, t, seed, scale=1.0, sway=0.0):
    """Teardrop flame path (base at +0.45r, tip up) with living wobble."""
    n = 26
    s = np.linspace(0, 1, n)
    tip = 0.95 * r * scale * (1 + 0.12 * noise1(t * 6.1, seed) + 0.06 * noise1(t * 15.0, seed + 1))
    base = 0.42 * r * scale
    hw = 0.46 * r * scale * np.sin(np.pi * np.clip(s, 0, 1) ** 0.62) ** 0.9 * (1 - 0.35 * s)
    yy = base - s * (base + tip)
    wob = r * scale * (0.07 * np.sin(s * 7 - t * 9 + seed) + 0.05 * np.sin(s * 13 - t * 15)) * s
    sw = sway * s ** 2 * r * scale
    xl = -hw + wob + sw
    xr = hw + wob + sw
    pts = [(float(a), float(b)) for a, b in zip(xr, yy)] + [(float(a), float(b)) for a, b in zip(xl[::-1], yy[::-1])]
    from engine.core import smooth_path
    return smooth_path(pts, close=True)


def heart_core(c, t, x, y, r=18, intensity=1.0, kind='flame', cage=True, flicker=1.0, sparks=True, rot=0.0,
               glass=True, halo=1.0):
    """The robot's heart core as a standalone glowing object (taken out of the chest and
    carried to the lamp, s06; the new star heart in s07).

    r           radius of the glass sphere (flame) / star (design units). Detailed enough
                for close-ups (r ~ 60-120) and still reads at r ~ 18.
    intensity   0..2 brightness (0 = dead/cold)
    kind        'flame': an amber living flame inside a small glass sphere held by a brass
                cage, flickering, with embers.  'star': the new star-shaped heart, white-gold,
                gently pulsing, with glints.
    cage        draw the brass cage (flame kind)
    flicker     flame liveliness;  sparks = rising embers / orbiting sparkles
    rot         slow rotation of the cage (radians) e.g. while being carried
    halo        multiplier for the outer glow
    """
    I = max(0.0, intensity)
    if kind == 'star':
        pul = 1 + 0.05 * math.sin(t * 2.4) + 0.02 * noise1(t * 3, 4)
        Ic = clamp(I)
        glow(c, x, y, r * 7 * halo, '#ffc860', 0.45 * I, core=False)
        _disc(c, x, y, r * 4.5, _c3('#9fe6ff'), 0.07 * I, 'soft')
        _blit(c, 'glint', x, y, r * 6.5 * pul, _c3('#fff2c8'), min(1, 0.45 * I), rot + 0.1 * math.sin(t * 0.7))
        with at(c, x, y, rot=rot + 0.05 * math.sin(t * 1.3), sx=pul):
            sp = _star_path(r * 1.15, r * 0.56, round_=0.32)
            # soft glow skirt hugging the star shape
            c.drawPath(sp, fill('#ffd86b', 0.5 * Ic, blur=r * 0.25, blend=ADD))
            p = radial((0, -r * 0.08), r * 1.2, [(0, '#fffdf2', 1), (0.3, '#fff3c0', 1), (0.7, '#ffe07a', 1), (1, '#ffbf3f', 1)])
            p.setAlphaf(clamp(0.25 + 0.75 * Ic))
            c.drawPath(sp, p)
            # rim light + inner facet highlight
            c.drawPath(sp, stroke('#fff6d0', max(0.8, r * 0.06), 0.7 * Ic))
            hp = _star_path(r * 0.55, r * 0.27, round_=0.4)
            c.drawPath(hp, fill('#ffffff', 0.6 * Ic, blur=r * 0.08, blend=ADD))
        glow(c, x, y, r * 0.9, '#fff6dc', 0.5 * I, core=False)
        if sparks and I > 0.05:
            n = 7
            k = np.arange(n)
            ang = t * (0.9 + 0.25 * _hash(k, 3)) + TAU * k / n
            rr = r * (1.9 + 0.5 * _hash(k, 4))
            e = (0.5 + 0.5 * np.sin(t * (3 + 2 * _hash(k, 5)) + k)) ** 3
            _blit(c, 'glint', x + np.cos(ang) * rr, y + np.sin(ang) * rr * 0.55, r * (0.9 + 0.7 * _hash(k, 6)),
                  _c3('#fff4c0'), e * Ic, 0)
        return

    # ---------------- flame in a caged glass sphere ----------------
    fl = 1 + flicker * (0.12 * noise1(t * 8.3, 7) + 0.06 * noise1(t * 21.0, 8))
    Ie = I * fl
    glow(c, x, y, r * 7.5 * halo, '#ffb347', 0.5 * Ie, core=False)
    glow(c, x, y, r * 2.8, '#ffc870', 0.55 * Ie, core=False)
    # glass: dark-amber tinted sphere, lit from inside
    if glass:
        gp = radial((x, y + r * 0.1), r, [(0, '#ffcf80', 0.45 * clamp(I)), (0.65, '#c8702a', 0.28 + 0.1 * clamp(I)),
                                          (1, '#5a3420', 0.55)])
        c.drawCircle(x, y, r, gp)
    # flame layers
    sway = flicker * 0.25 * noise1(t * 2.2, 11)
    with at(c, x, y + r * 0.12):
        for sc_, colr, a_, bl in ((1.0, '#ff6a1e', 0.85, 0.1), (0.78, '#ffa436', 0.9, 0.08),
                                  (0.55, '#ffd878', 0.95, 0.06), (0.3, '#fffbe8', 1.0, 0.05)):
            p = fill(colr, clamp(a_ * Ie), blur=max(0.4, r * bl), blend=ADD)
            c.drawPath(_flame_path(r, t, 3, sc_, sway), p)
    glow(c, x, y + r * 0.05, r * 0.8, '#ffe2a0', 0.35 * Ie, core=False)
    if glass:
        # rim + specular highlights
        c.drawCircle(x, y, r * 0.97, stroke('#ffd9a0', max(0.6, r * 0.06), 0.35 * clamp(0.4 + I), blur=r * 0.03))
        with at(c, x - r * 0.38, y - r * 0.45, rot=-0.6):
            c.drawOval(skia.Rect.MakeXYWH(-r * 0.28, -r * 0.11, r * 0.56, r * 0.22), fill('#ffffff', 0.55, blur=r * 0.04))
        c.drawCircle(x + r * 0.42, y + r * 0.42, r * 0.07, fill('#ffffff', 0.45, blur=r * 0.02))
    if cage:
        brass, lit, dark = '#c9a15b', '#ffe0a0', '#6a4a26'
        bw = max(0.9, r * 0.075)
        # meridian bars (rotating globe cage); back bars dimmer, drawn first
        for k in range(3):
            ph = rot + k * math.pi / 3
            cx_ = math.cos(ph)
            front = math.sin(ph) > 0
            rx = abs(cx_) * r * 1.04
            rect = skia.Rect.MakeXYWH(x - rx, y - r * 1.04, 2 * rx, 2 * r * 1.04)
            a_ = 1.0 if front else 0.35
            if rx < bw * 0.6:
                c.drawLine(x, y - r * 1.04, x, y + r * 1.04, stroke(brass, bw, a_))
            else:
                c.drawOval(rect, stroke(dark if not front else brass, bw, a_))
                if front:
                    c.drawOval(rect, stroke(lit, bw * 0.35, 0.5 * clamp(I)))
        # equator band
        eq = skia.Rect.MakeXYWH(x - r * 1.06, y - r * 0.22, r * 2.12, r * 0.44)
        c.drawOval(eq, stroke(dark, bw * 1.3, 0.9))
        c.drawArc(eq, 0, 180, False, stroke(brass, bw * 1.3, 1.0))
        c.drawArc(eq, 20, 140, False, stroke(lit, bw * 0.45, 0.55 * clamp(I)))
        # caps + little loop
        for sgn in (-1, 1):
            cap = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x - r * 0.34, y + sgn * r * 1.02 - r * 0.13, r * 0.68, r * 0.26),
                                        r * 0.1, r * 0.1)
            c.drawRRect(cap, fill(brass))
            c.drawRRect(cap, stroke(lit, max(0.5, r * 0.035), 0.6))
        c.drawCircle(x, y - r * 1.32, r * 0.17, stroke(brass, bw * 0.9, 1.0))
        # rivets
        for k in range(-2, 3):
            ax = x + k * r * 0.42
            c.drawCircle(ax, y + r * 0.2 * math.sqrt(max(0, 1 - (k * 0.42 / 1.06) ** 2)), max(0.6, r * 0.05), fill(lit, 0.8))
    if sparks and I > 0.05:
        embers(c, t, x, y - r * 0.8, count=10, spread=r * 0.35, rise=r * 3.2, size=max(0.3, r / 45), intensity=clamp(I),
               seed=31, life=(0.9, 1.8))


def light_trail(c, points, width=14, color='#ffd86b', intensity=1.0, t=0.0, core_color='#fffbea', sparkle=0.6,
                taper=True, seed=16):
    """A glowing ribbon along a path (e.g. Guang's flight trail).

    points      [(x, y), ...] from the OLDEST (tail end, fades out) to the NEWEST (head)
    width       half-width at the head
    sparkle     amount of twinkling glints scattered along the trail (uses t)
    taper       narrow the trail towards its tail
    """
    if intensity <= 0 or len(points) < 2:
        return
    K = 90
    P = _catmull(points, K)
    s = np.linspace(0, 1, K)        # 0 tail -> 1 head
    wv = width * (s ** 0.7 if taper else np.ones(K)) + 0.8
    a = intensity * s ** 1.4 * (1 - _sstep(0.97, 1.0, s) * 0.5)
    rgb, crgb = _c3(color), _c3(core_color)
    nrm = _normals(P)
    wig = width * 0.5 * _noise(s * 6 - t * 1.5, seed) * (1 - s)
    P = P + nrm * wig[:, None]
    _multi_mesh(c, [_ribbon_part(P, wv * 3.0, _X7, _P7, rgb, a * 0.28, nrm),
                    _ribbon_part(P, wv, _X5, _P5, rgb, a * 0.7, nrm),
                    _ribbon_part(P, wv * 0.35, _X5, _P5, crgb, a * 0.95, nrm)])
    if sparkle > 0:
        n = int(26 * sparkle)
        i = np.arange(n)
        per = 0.5 + 0.7 * _hash(i, seed + 1)
        cyc = t / per + _hash(i, seed + 2)
        k = np.floor(cyc)
        u = cyc - k
        su = _h2(i, k, seed + 3) ** 0.7
        idx = np.clip((su * (K - 1)).astype(int), 0, K - 1)
        off = (2 * _h2(i, k, seed + 4) - 1) * wv[idx] * 2.2
        px = P[idx, 0] + nrm[idx, 0] * off
        py = P[idx, 1] + nrm[idx, 1] * off + u * 14
        e = np.sin(np.pi * u) ** 2 * su
        _blit(c, 'glint', px, py, width * (1.2 + 1.8 * _h2(i, k, seed + 5)), _mixc(rgb, _WHITE, 0.5), e * intensity, 0)
