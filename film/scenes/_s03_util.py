"""Private helpers for s03 (owner: SCENE-S03).

Library-compat shims (call lib functions with only the kwargs they accept),
a tiny perspective model of the sea plane, the curved meteor, and the
bioluminescent plankton field. Everything is a pure function of its inputs.
"""
from __future__ import annotations

import dataclasses
import inspect
import math

import numpy as np
import skia

from engine.core import (ADD, H, W, clamp, col, fill, lerp, linear, mix, noise1, nprng, radial, smoothstep,
                         soft_glow, stroke)

TAU = math.tau

# ----------------------------------------------------------------------------
# library compatibility
# ----------------------------------------------------------------------------
_SIG = {}


def _params(fn):
    key = id(fn)
    if key not in _SIG:
        try:
            ps = inspect.signature(fn).parameters
            varkw = any(p.kind == p.VAR_KEYWORD for p in ps.values())
            _SIG[key] = (set(ps), varkw)
        except (TypeError, ValueError):
            _SIG[key] = (set(), True)
    return _SIG[key]


def has_kw(fn, name):
    names, varkw = _params(fn)
    return name in names


def call(fn, *args, **kw):
    """Call a library function passing only the keyword args it accepts."""
    names, varkw = _params(fn)
    if not varkw:
        kw = {k: v for k, v in kw.items() if k in names}
    return fn(*args, **kw)


_WARNED = set()


def safe(fn, *args, **kw):
    """Like call(), but a library that is mid-edit (raising) must not kill the frame."""
    try:
        return call(fn, *args, **kw)
    except Exception as e:  # noqa: BLE001
        key = (getattr(fn, '__name__', '?'), type(e).__name__)
        if key not in _WARNED:
            _WARNED.add(key)
            import sys
            print(f'[s03] library call {key[0]} failed: {e!r}', file=sys.stderr)
        return None


def mkpose(cls, **kw):
    """Build a dataclass pose, silently dropping fields the library doesn't have (yet)."""
    fields = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in kw.items() if k in fields})


def pose_has(cls, name):
    return name in {f.name for f in dataclasses.fields(cls)}


def h01(i, seed=0):
    """Deterministic hash -> [0,1)."""
    x = (int(i) * 374761393 + int(seed) * 668265263) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((x ^ (x >> 16)) & 0xFFFFFF) / float(0x1000000)


# ----------------------------------------------------------------------------
# perspective sea plane
# ----------------------------------------------------------------------------
class SeaView:
    """Pinhole camera looking at a flat sea. Ground point (X, Z) -> screen (x, y).
    hz: screen y of the horizon, F: focal length (px), cam_h: camera height,
    cx: screen x of the optical axis, cam_x: lateral camera position."""

    def __init__(self, hz, F=1000.0, cam_h=50.0, cx=W / 2, cam_x=0.0):
        self.hz, self.F, self.cam_h, self.cx, self.cam_x = hz, F, cam_h, cx, cam_x

    def proj(self, X, Z):
        return self.cx + (X - self.cam_x) * self.F / Z, self.hz + self.cam_h * self.F / Z

    def ground(self, x, y):
        Z = self.cam_h * self.F / max(0.5, (y - self.hz))
        X = self.cam_x + (x - self.cx) * Z / self.F
        return X, Z

    def scale(self, Z):
        return self.F / Z


class ground:
    """Context manager: local coords become ground-plane (X, Z) coords of SeaView `sv`
    (perspective homography), clipped to zmin..zmax so nothing wraps behind the camera."""

    def __init__(self, c, sv, zmin=6.0, zmax=60000.0, xspan=80000.0):
        self.c, self.sv, self.zmin, self.zmax, self.xspan = c, sv, zmin, zmax, xspan

    def __enter__(self):
        sv = self.sv
        m = skia.Matrix.MakeAll(sv.F, sv.cx, -sv.F * sv.cam_x, 0, sv.hz, sv.cam_h * sv.F, 0, 1, 0)
        self.c.save()
        self.c.concat(m)
        self.c.clipRect(skia.Rect.MakeLTRB(sv.cam_x - self.xspan, self.zmin, sv.cam_x + self.xspan, self.zmax))
        return self.c

    def __exit__(self, *e):
        self.c.restore()


def bio_front(ts, radius, life):
    """Ring-front radius of fx.bioluminescence (same law as the library)."""
    Tf = 0.35 * life
    if ts <= 0:
        return 0.0
    return radius * (ts / Tf) ** 0.7 if ts < Tf else radius * (1 + 0.15 * (ts - Tf) / Tf)


def ground_ring_path(sv, X0, Z0, R, n=96, zmin=2.0):
    """Closed path of a circle of radius R on the sea plane (clipped in front of camera)."""
    path = skia.Path()
    first = True
    for i in range(n + 1):
        a = TAU * i / n
        X, Z = X0 + math.cos(a) * R, Z0 + math.sin(a) * R
        if Z < zmin:
            first = True
            continue
        x, y = sv.proj(X, Z)
        if first:
            path.moveTo(x, y)
            first = False
        else:
            path.lineTo(x, y)
    return path


# ----------------------------------------------------------------------------
# plankton field (world-anchored points on the sea plane)
# ----------------------------------------------------------------------------
class Plankton:
    """N glowing specks scattered uniformly in *screen* space for a reference view,
    stored as ground coordinates so they stay put when the camera tilts/pans."""

    def __init__(self, sv, n=900, seed=41, x_range=(-300, W + 300), y_top_pad=2.0):
        r = nprng(seed)
        ys = sv.hz + y_top_pad + (H + 120 - sv.hz - y_top_pad) * r.uniform(0, 1, n) ** 1.15
        xs = r.uniform(x_range[0], x_range[1], n)
        self.Z = sv.cam_h * sv.F / (ys - sv.hz)
        self.X = sv.cam_x + (xs - sv.cx) * self.Z / sv.F
        self.ph = r.uniform(0, TAU, n)
        self.om = r.uniform(1.2, 4.0, n)
        self.sz = r.uniform(0.6, 1.4, n)
        self.warm = r.uniform(0, 1, n) < 0.22
        self.jit = r.uniform(0, 1, n)
        self.n = n

    def draw(self, c, sv, t, bright, size=1.0, max_px=7.0, cool='#6ff0ff', warm='#ffd36b', clip_y=None):
        """bright: array (n,) of per-particle brightness 0..~2 (already includes story logic)."""
        x, y = sv.proj(self.X, self.Z)
        s = np.clip((0.9 + sv.F / self.Z * 0.35) * self.sz * size, 0.7, max_px)
        tw = 0.6 + 0.4 * np.sin(t * self.om + self.ph)
        a = np.clip(bright * tw, 0, 1.5)
        vis = (a > 0.02) & (x > -20) & (x < W + 20) & (y < H + 20) & (y > sv.hz)
        if clip_y is not None:
            vis &= y < clip_y
        idx = np.nonzero(vis)[0]
        if len(idx) == 0:
            return
        pc = fill(cool, 1.0, blend=ADD)
        pw = fill(warm, 1.0, blend=ADD)
        for i in idx:
            p = pw if self.warm[i] else pc
            ai = float(a[i])
            si = float(s[i])
            xi, yi = float(x[i]), float(y[i])
            if si > 2.2 and ai > 0.25:  # soft halo on the bigger / brighter ones
                p.setAlphaf(clamp(ai * 0.16))
                c.drawOval(skia.Rect.MakeXYWH(xi - si * 4, yi - si * 1.6, si * 8, si * 3.2), p)
            p.setAlphaf(clamp(ai))
            c.drawOval(skia.Rect.MakeXYWH(xi - si, yi - si * 0.55, si * 2, si * 1.1), p)


def ring_brightness(pl, X0, Z0, ts, radius=870.0, life=12.0, band=55.0, sustain=0.55, decay=7.0):
    """Story logic for the plankton ring (synced with fx.bioluminescence's front law):
    a bright flash when the front passes, then a sustained twinkling glow slowly decaying."""
    if ts <= 0:
        return np.zeros(pl.n)
    Tf = 0.35 * life
    R = bio_front(ts, radius, life)
    d = np.hypot(pl.X - X0, pl.Z - Z0)
    front = np.exp(-((d - R) / band) ** 2)
    inside = d < R
    t_pass = Tf * np.clip(d / radius, 0, 1) ** (1 / 0.7)
    age = np.clip(ts - t_pass, 0, None)
    glow = np.where(inside, sustain * np.exp(-age / decay) * (0.45 + 0.55 * np.exp(-age / 1.0)), 0.0)
    glow = glow * (0.35 + 0.65 * pl.jit)
    return front * 1.6 + glow


# ----------------------------------------------------------------------------
# curved meteor
# ----------------------------------------------------------------------------
class Bezier:
    """Quadratic (3 pts) or cubic (4 pts) bezier with an arc-length table."""

    def __init__(self, pts, n=600):
        self.pts = pts
        us = np.linspace(0, 1, n)
        if len(pts) == 3:
            (x0, y0), (x1, y1), (x2, y2) = pts
            a, b, cc = (1 - us) ** 2, 2 * (1 - us) * us, us ** 2
            self.x = a * x0 + b * x1 + cc * x2
            self.y = a * y0 + b * y1 + cc * y2
        else:
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = pts
            a = (1 - us) ** 3
            b = 3 * (1 - us) ** 2 * us
            cc = 3 * (1 - us) * us ** 2
            d = us ** 3
            self.x = a * x0 + b * x1 + cc * x2 + d * x3
            self.y = a * y0 + b * y1 + cc * y2 + d * y3
        seg = np.hypot(np.diff(self.x), np.diff(self.y))
        self.cum = np.concatenate([[0], np.cumsum(seg)])
        self.length = float(self.cum[-1])

    def at(self, s):
        """Point at arc-length fraction s (0..1, extrapolates linearly outside)."""
        L = s * self.length
        if s < 0:
            dx, dy = self.x[1] - self.x[0], self.y[1] - self.y[0]
            k = L / max(1e-6, math.hypot(dx, dy))
            return self.x[0] + dx * k, self.y[0] + dy * k
        L = min(L, self.length)
        return float(np.interp(L, self.cum, self.x)), float(np.interp(L, self.cum, self.y))

    def tangent(self, s, ds=0.004):
        a = self.at(s - ds)
        b = self.at(s + ds)
        d = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
        return (b[0] - a[0]) / d, (b[1] - a[1]) / d


def ribbon(pts, widths):
    """Filled tapered ribbon along pts with half-widths `widths`."""
    n = len(pts)
    left, right = [], []
    for i in range(n):
        x, y = pts[i]
        a = pts[max(0, i - 1)]
        b = pts[min(n - 1, i + 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        d = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / d, dx / d
        w = widths[i]
        left.append((x + nx * w, y + ny * w))
        right.append((x - nx * w, y - ny * w))
    path = skia.Path()
    path.moveTo(*left[0])
    for p in left[1:]:
        path.lineTo(*p)
    for p in reversed(right):
        path.lineTo(*p)
    path.close()
    return path


def glint(c, x, y, size, a, rot=0.0, color='#fff6e0', thin=0.08):
    """Four-point cross glint (star twinkle)."""
    if a <= 0.01 or size <= 0.5:
        return
    p = fill(color, clamp(a), blend=ADD)
    c.save()
    c.translate(x, y)
    c.rotate(math.degrees(rot))
    for k in range(2):
        path = skia.Path()
        L, w = size * (1.0 if k == 0 else 0.72), size * thin
        path.moveTo(-L, 0)
        path.quadTo(0, -w, L, 0)
        path.quadTo(0, w, -L, 0)
        path.close()
        c.drawPath(path, p)
        c.rotate(90)
    c.restore()


def draw_meteor_tail(c, pts, size, intensity, color='#fff1c8', hot='#ffffff', edge='#9fd8ff'):
    """pts: head first. Layered additive tapered ribbons."""
    n = len(pts)
    if n < 3:
        return
    layers = ((1.0, 7.0, 0.07, edge, 10), (0.85, 3.2, 0.16, color, 3), (0.6, 1.5, 0.42, color, 0),
              (0.32, 0.6, 0.85, hot, 0))
    for frac, wm, a, colr, blur in layers:
        m = max(3, int(n * frac))
        sub = pts[:m]
        widths = [size * wm * (1 - k / (m - 1)) ** 0.9 + 0.3 for k in range(m)]
        p = fill(colr, clamp(a * intensity), blur=blur * size if blur else 0.0, blend=ADD)
        c.drawPath(ribbon(sub, widths), p)


def meteor_sparks(c, t, t0, pos_at, tan_at, size_at, rate=60.0, life=1.1, seed=77, intensity=1.0, t_end=None):
    """Sparks shed from the meteor head. pos_at(time)->(x,y), tan_at(time)->(dx,dy) screen.
    Each spark is born at a fixed time -> pure function of t."""
    if t < t0:
        return
    i_hi = int((t - t0) * rate)
    i_lo = max(0, int((t - life - t0) * rate))
    if t_end is not None:
        i_hi = min(i_hi, int((t_end - t0) * rate))
    pw = fill('#fff4c0', 1.0, blend=ADD)
    po = fill('#ffb867', 1.0, blend=ADD)
    for i in range(i_lo, i_hi + 1):
        tb = t0 + i / rate
        age = t - tb
        if age < 0 or age > life:
            continue
        lf = life * (0.5 + 0.5 * h01(i, seed + 1))
        if age > lf:
            continue
        bx, by = pos_at(tb)
        tx, ty = tan_at(tb)
        sz = size_at(tb)
        # eject sideways & backwards, then drift down with drag
        side = (h01(i, seed + 2) - 0.5) * 2
        back = 0.3 + 0.7 * h01(i, seed + 3)
        v = (60 + 140 * h01(i, seed + 4)) * sz
        vx = -tx * v * back + -ty * v * side * 0.8
        vy = -ty * v * back + tx * v * side * 0.8
        k = (1 - math.exp(-age * 2.2)) / 2.2
        px = bx + vx * k
        py = by + vy * k + 40 * age * age * sz
        fade = (1 - age / lf) ** 1.5
        r = (1.2 + 2.2 * h01(i, seed + 5)) * min(2.2, 0.6 + sz * 0.5) * (0.4 + 0.6 * fade)
        p = pw if h01(i, seed + 6) < 0.6 else po
        tw = 0.6 + 0.4 * math.sin(age * 30 + i)
        p.setAlphaf(clamp(fade * intensity * tw))
        c.drawCircle(px, py, r, p)
        if r > 2.4 and fade > 0.4:
            p.setAlphaf(clamp(fade * intensity * 0.18))
            c.drawCircle(px, py, r * 3.5, p)


# ----------------------------------------------------------------------------
# lighting helpers
# ----------------------------------------------------------------------------
def lit(c, bounds, draw_fn, washes):
    """Draw `draw_fn()` inside a layer, then wash it with directional light gradients
    clipped to what was drawn (SrcATop). washes: list of (p0, p1, colour, alpha)
    where p0 is the lit side (alpha) fading to 0 at p1. Returns draw_fn's result."""
    washes = [w for w in washes if w[3] > 0.004]
    if not washes:
        return draw_fn()
    c.saveLayer(bounds, None)
    res = draw_fn()
    for p0, p1, colr, a in washes:
        p = linear(p0, p1, [(0, colr, clamp(a)), (1, colr, 0)])
        p.setBlendMode(skia.BlendMode.kSrcATop)
        c.drawRect(bounds, p)
    c.restore()
    return res
