"""Private helpers for s07 (owner: SCENE-S07).

Defensive wrappers so the scene keeps working while the libraries are being
upgraded in parallel (unknown kwargs are dropped, new pose fields are only set
when they exist), plus small curve / layer utilities.
"""
from __future__ import annotations

import dataclasses
import inspect
import math

import skia

from engine.core import H, TAU, W, clamp, lerp

# ----------------------------------------------------------------------------
# tolerant calls
# ----------------------------------------------------------------------------
_SIG = {}


def accepts(fn):
    """Set of keyword names `fn` accepts, or None if it takes **kwargs / unknown."""
    key = (getattr(fn, '__module__', ''), getattr(fn, '__qualname__', repr(fn)), id(fn))
    if key not in _SIG:
        try:
            ps = inspect.signature(fn).parameters
            if any(p.kind == p.VAR_KEYWORD for p in ps.values()):
                _SIG[key] = None
            else:
                _SIG[key] = frozenset(ps)
        except (TypeError, ValueError):
            _SIG[key] = None
    return _SIG[key]


def has_kw(fn, name):
    ok = accepts(fn)
    return ok is not None and name in ok


def call(fn, *a, **kw):
    """Call a library function, silently dropping kwargs it does not know (yet)."""
    ok = accepts(fn)
    if ok is not None:
        kw = {k: v for k, v in kw.items() if k in ok}
    return fn(*a, **kw)


_FIELDS = {}


def fields(cls):
    if cls not in _FIELDS:
        _FIELDS[cls] = frozenset(fl.name for fl in dataclasses.fields(cls))
    return _FIELDS[cls]


def mk(cls, **kw):
    """Build a pose dataclass, ignoring fields the library does not have (yet)."""
    names = fields(cls)
    return cls(**{k: v for k, v in kw.items() if k in names})


def to_hex(rgb):
    return '#%02x%02x%02x' % tuple(int(round(clamp(v) * 255)) for v in rgb[:3])


# ----------------------------------------------------------------------------
# camera helpers
# ----------------------------------------------------------------------------
def to_layer(cam, X, Y, depth=1.0):
    """Inverse of Camera.to_screen: screen point -> coordinates of layer `depth`."""
    z = 1.0 + (cam.zoom - 1.0) * depth if depth != 1.0 else cam.zoom
    cx = W / 2 + (cam.x - W / 2) * depth
    cy = H / 2 + (cam.y - H / 2) * depth
    return cx + (X - W / 2) / z, cy + (Y - H / 2) / z


def layer_zoom(cam, depth=1.0):
    return 1.0 + (cam.zoom - 1.0) * depth if depth != 1.0 else cam.zoom


# ----------------------------------------------------------------------------
# curves
# ----------------------------------------------------------------------------
def cr_sample(pts, per=12):
    """Catmull-Rom spline through pts, sampled `per` points per segment."""
    out = []
    n = len(pts)
    if n < 2:
        return list(pts)
    for i in range(n - 1):
        p0 = pts[max(i - 1, 0)]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[min(i + 2, n - 1)]
        for k in range(per):
            u = k / per
            u2, u3 = u * u, u * u * u
            q = []
            for d in (0, 1):
                q.append(0.5 * (2 * p1[d] + (-p0[d] + p2[d]) * u
                                + (2 * p0[d] - 5 * p1[d] + 4 * p2[d] - p3[d]) * u2
                                + (-p0[d] + 3 * p1[d] - 3 * p2[d] + p3[d]) * u3))
            out.append((q[0], q[1]))
    out.append(tuple(pts[-1]))
    return out


def _cum(pts):
    L = [0.0]
    for (ax, ay), (bx, by) in zip(pts, pts[1:]):
        L.append(L[-1] + math.hypot(bx - ax, by - ay))
    return L


def length(pts):
    return _cum(pts)[-1]


def point_at(pts, u, L=None):
    """Point at arc-length fraction u (0..1) along polyline."""
    L = L or _cum(pts)
    tot = L[-1]
    if tot <= 0:
        return pts[0]
    d = clamp(u) * tot
    for i in range(1, len(L)):
        if L[i] >= d:
            seg = L[i] - L[i - 1]
            k = (d - L[i - 1]) / seg if seg > 0 else 0.0
            (ax, ay), (bx, by) = pts[i - 1], pts[i]
            return ax + (bx - ax) * k, ay + (by - ay) * k
    return pts[-1]


def resample(pts, n, a=0.0, b=1.0):
    """n+1 points evenly spaced (arc length) between fractions a..b of the polyline."""
    L = _cum(pts)
    return [point_at(pts, lerp(a, b, i / n), L) for i in range(n + 1)]


def spath(pts):
    p = skia.Path()
    if not pts:
        return p
    p.moveTo(*pts[0])
    for q in pts[1:]:
        p.lineTo(*q)
    return p


def bezier(p0, p1, p2, p3, u):
    v = 1 - u
    return (v * v * v * p0[0] + 3 * v * v * u * p1[0] + 3 * v * u * u * p2[0] + u * u * u * p3[0],
            v * v * v * p0[1] + 3 * v * v * u * p1[1] + 3 * v * u * u * p2[1] + u * u * u * p3[1])


def helix(cx, y_top, y_bot, r0, r1, turns, phase, n=90, persp=0.3):
    """Spiral around a vertical axis at cx, from y_top (radius r0) to y_bot (radius r1).
    Returns (points, front_flags) — front = the half of each loop nearer the camera."""
    pts, front = [], []
    for i in range(n + 1):
        s = i / n
        a = phase + s * turns * TAU
        r = lerp(r0, r1, s ** 0.75)
        pts.append((cx + r * math.cos(a), lerp(y_top, y_bot, s) + r * persp * math.sin(a)))
        front.append(math.sin(a) > 0)
    return pts, front


def runs(pts, flags):
    """Split a polyline into consecutive runs of equal flag -> [(flag, pts)]."""
    out = []
    cur = [pts[0]]
    fl = flags[0]
    for p, f in zip(pts[1:], flags[1:]):
        cur.append(p)
        if f != fl:
            out.append((fl, cur))
            cur = [p]
            fl = f
    if len(cur) > 1:
        out.append((fl, cur))
    return out


# ----------------------------------------------------------------------------
# picture recording (know a character's anchors before compositing it)
# ----------------------------------------------------------------------------
def record(draw_fn, cam=None, depth=1.0):
    """Run draw_fn(canvas) into a Picture (optionally inside cam.apply(depth)).
    Returns (picture, draw_fn result)."""
    rec = skia.PictureRecorder()
    cc = rec.beginRecording(skia.Rect.MakeLTRB(-6000, -6000, 8000, 8000))
    if cam is not None:
        with cam.apply(cc, depth):
            res = draw_fn(cc)
    else:
        res = draw_fn(cc)
    return rec.finishRecordingAsPicture(), res


def put(c, pic, mul=None, add=(0.0, 0.0, 0.0), bounds=None, alpha=1.0):
    """Composite a recorded picture, optionally through a colour-matrix light tint."""
    if mul is None and alpha >= 1.0:
        c.drawPicture(pic)
        return
    p = skia.Paint()
    if mul is not None:
        m = [mul[0], 0, 0, 0, add[0],
             0, mul[1], 0, 0, add[1],
             0, 0, mul[2], 0, add[2],
             0, 0, 0, 1, 0]
        p.setColorFilter(skia.ColorFilters.Matrix(m))
    p.setAlphaf(clamp(alpha))
    c.saveLayer(bounds, p)
    c.drawPicture(pic)
    c.restore()
