"""Private helpers for scene s06 (owner: SCENE-S06).

- tolerant calls into the libraries (kwargs a library does not know yet are dropped)
- camera inverse / visible rect helpers
- robot arm IK (arm geometry probed from lib.robot once, so it adapts to the real art)
- "lit" robot drawing: extra light multiplied onto the robot only (arithmetic filter)
- LOCAL FALLBACKS for things the libraries may not have yet: cloud deck blown open
  into a ring, Milky Way swirl, the lamp room's iron wheel, eye glints, light streaks.
Everything is a pure function of its inputs (frames render out of order).
"""
from __future__ import annotations

import inspect
import math
from dataclasses import fields

import numpy as np
import skia

from engine.core import (ADD, H, TAU, W, at, clamp, col, fill, lerp, linear, mix, noise1, nprng, poly,
                         radial, rrect, smoothstep, soft_glow, stroke)
from lib import fx, robot, sea, sky, star

# ----------------------------------------------------------------------------
# tolerant library calls
# ----------------------------------------------------------------------------
_SIG = {}


def _accepts(fn):
    if fn not in _SIG:
        try:
            ps = inspect.signature(fn).parameters
            if any(p.kind == p.VAR_KEYWORD for p in ps.values()):
                _SIG[fn] = None
            else:
                _SIG[fn] = set(ps)
        except (TypeError, ValueError):
            _SIG[fn] = None
    return _SIG[fn]


def accepts(fn, name):
    ok = _accepts(fn)
    return ok is None or name in ok


def call(fn, *args, **kw):
    """Call a library function, silently dropping kwargs it does not (yet) accept."""
    ok = _accepts(fn)
    if ok is not None:
        kw = {k: v for k, v in kw.items() if k in ok}
    return fn(*args, **kw)


def lib_fn(mod, name):
    return getattr(mod, name, None)


def _dc(cls, kw):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in kw.items() if k in names})


def rpose(**kw):
    return _dc(robot.RobotPose, kw)


def spose(**kw):
    return _dc(star.StarPose, kw)


# ----------------------------------------------------------------------------
# camera helpers
# ----------------------------------------------------------------------------
def cam_inv(cam, X, Y, depth=1.0):
    z = 1.0 + (cam.zoom - 1.0) * depth if depth != 1.0 else cam.zoom
    cx = W / 2 + (cam.x - W / 2) * depth
    cy = H / 2 + (cam.y - H / 2) * depth
    return cx + (X - W / 2) / z, cy + (Y - H / 2) / z


def visible(cam, depth=1.0, margin=260):
    x0, y0 = cam_inv(cam, -margin, -margin, depth)
    x1, y1 = cam_inv(cam, W + margin, H + margin, depth)
    return (x0, y0, x1 - x0, y1 - y0)


def loglerp(a, b, u):
    return math.exp(lerp(math.log(a), math.log(b), u))


# ----------------------------------------------------------------------------
# robot arm geometry + IK
# ----------------------------------------------------------------------------
_ARM = {}


def _probe_arms():
    """Estimate shoulder position and the two arm segment lengths of lib.robot (scale 1)."""
    if _ARM:
        return _ARM
    geo = {'r': (62.0, -225.0, 70.0, 70.0), 'l': (-62.0, -225.0, 70.0, 70.0)}
    try:
        rec = skia.PictureRecorder()
        cc = rec.beginRecording(skia.Rect.MakeXYWH(-2000, -2000, 4000, 4000))

        def hands(a, e):
            p = rpose(x=0.0, y=0.0, scale=1.0, facing=1.0, lean=0.0, arm_l=(a, e), arm_r=(a, e),
                      walk=0.0, sit=0.0, shake=0.0, head_dy=0.0)
            return robot.draw_robot(cc, p, 0.0)

        h0, h1, h2 = hands(0.0, 0.0), hands(math.pi / 2, 0.0), hands(math.pi / 2, math.pi / 2)
        rec.finishRecordingAsPicture()
        for side in ('r', 'l'):
            sx, sy = h0[side][0], h1[side][1]
            l1 = h2[side][0] - sx
            l2 = sy - h2[side][1]
            if 15 < l1 < 400 and 15 < l2 < 400:
                geo[side] = (sx, sy, l1, l2)
    except Exception:
        pass
    _ARM.update(geo)
    return _ARM


def arm_ik(pose_kw, side, target, elbow=1.0):
    """Arm angles (shoulder, elbow) so the `side` hand of a robot posed with pose_kw
    (x, y, scale, facing, lean) reaches world `target`."""
    sx, sy, l1, l2 = _probe_arms()[side]
    s = pose_kw.get('scale', 1.0)
    fc = pose_kw.get('facing', 1.0)
    ln = pose_kw.get('lean', 0.0)
    qx = (target[0] - pose_kw.get('x', 960.0)) / (s * fc)
    qy = (target[1] - pose_kw.get('y', 900.0)) / s
    cs, sn = math.cos(-ln), math.sin(-ln)
    px, py = qx * cs - qy * sn, qx * sn + qy * cs
    X, Y = px - sx, py - sy
    d = max(1e-3, math.hypot(X, Y))
    dmax = (l1 + l2) * 0.995
    dmin = abs(l1 - l2) + 1.0
    if d > dmax:
        X, Y, d = X * dmax / d, Y * dmax / d, dmax
    elif d < dmin:
        X, Y, d = X * dmin / d, Y * dmin / d, dmin
    ce = clamp((d * d - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1.0, 1.0)
    e = elbow * math.acos(ce)
    a = math.atan2(X, Y) - math.atan2(l2 * math.sin(e), l1 + l2 * math.cos(e))
    return (a, e)


def arm_reach(pose_kw, side):
    return sum(_probe_arms()[side][2:]) * pose_kw.get('scale', 1.0)


# ----------------------------------------------------------------------------
# lighting the robot: light multiplied onto its pixels only
# ----------------------------------------------------------------------------
def robot_bounds(pose_kw, pad=1.25):
    s = pose_kw.get('scale', 1.0)
    x, y = pose_kw.get('x', 960.0), pose_kw.get('y', 900.0)
    return skia.Rect.MakeLTRB(x - 300 * s * pad, y - 440 * s * pad, x + 300 * s * pad, y + 60 * s)


def draw_robot_lit(c, pose_kw, t, lights=(), gain=1.0):
    """Draw Deng; `lights` = list of skia.Paint (gradient shaders, colour = light amount)
    whose light is multiplied into his colours (robot pixels only). Returns hands dict."""
    pose = rpose(**pose_kw)
    lights = [p for p in lights if p is not None]
    if not lights or gain <= 0:
        return robot.draw_robot(c, pose, t)
    b = robot_bounds(pose_kw)
    rec = skia.PictureRecorder()
    cc = rec.beginRecording(b)
    for p in lights:
        p.setBlendMode(ADD)
        cc.drawRect(b, p)
    pic = rec.finishRecordingAsPicture()
    filt = skia.ImageFilters.Arithmetic(1.6 * gain, 1.0, 0.0, 0.0, True, skia.ImageFilters.Picture(pic, b))
    c.saveLayer(b, skia.Paint(ImageFilter=filt))
    try:
        res = robot.draw_robot(c, pose, t)
    finally:
        c.restore()
    return res


def light_band(p0, p1, color, amt, width=0.18, center=0.5):
    """A soft linear band of light between p0 and p1 (for a beam sweeping across)."""
    if amt <= 0:
        return None
    a = max(0.0, center - width)
    b = min(1.0, center + width)
    stops = [(0.0, color, 0.0)]
    if a > 0:
        stops.append((a, color, 0.0))
    stops.append((clamp(center, 0.001, 0.999), color, amt))
    if b < 1:
        stops.append((b, color, 0.0))
    stops.append((1.0, color, 0.0))
    return linear(p0, p1, stops)


def light_radial(x, y, r, color, amt):
    if amt <= 0:
        return None
    return radial((x, y), r, [(0.0, color, amt), (0.5, color, amt * 0.45), (1.0, color, 0.0)])


def light_linear(p0, p1, color, a0, a1):
    return linear(p0, p1, [(0.0, color, a0), (1.0, color, a1)])


# ----------------------------------------------------------------------------
# small FX (scene-local)
# ----------------------------------------------------------------------------
def glint(c, x, y, r, color='#fff6d8', a=1.0, rot=0.0):
    """4-point cross glint (eye sparkle / star flare)."""
    if a <= 0 or r <= 0:
        return
    with at(c, x, y, rot=rot):
        p = fill(color, a, blend=ADD)
        for k in range(2):
            w = r * 0.11
            path = poly([(-r, 0), (0, -w), (r, 0), (0, w)]) if k == 0 else poly([(0, -r), (w, 0), (0, r), (-w, 0)])
            c.drawPath(path, p)
    soft_glow(c, x, y, r * 0.8, color, a * 0.7)


_STREAKS = nprng(606).uniform(0, 1, (70, 4))


def light_streaks(c, t, x0, x1, y0, y1, speed=1800.0, color='#fff3c8', intensity=1.0, length=260.0, width=3.0):
    """Streaks of light flowing UP inside a column [x0,x1] x [y0,y1] (world units)."""
    if intensity <= 0:
        return
    span = (y1 - y0) + length
    for u, v, sp, ph in _STREAKS:
        x = lerp(x0, x1, u)
        yy = y1 - ((v * span + t * speed * (0.6 + 0.8 * sp)) % span)
        ln = length * (0.5 + sp)
        a = intensity * (0.35 + 0.65 * (0.5 + 0.5 * math.sin(ph * TAU + t * 3.0)))
        p = linear((x, yy), (x, yy + ln), [(0, color, a), (1, color, 0)])
        p.setBlendMode(ADD)
        p.setStyle(skia.Paint.kStroke_Style)
        p.setStrokeWidth(width * (0.6 + sp))
        p.setStrokeCap(skia.Paint.kRound_Cap)
        c.drawLine(x, yy, x, yy + ln, p)


# ----------------------------------------------------------------------------
# FALLBACK: cloud deck that the pillar blows open into a ring
# ----------------------------------------------------------------------------
_DECK = nprng(2106)
_DECK_N = 64
_DECK_U = _DECK.uniform(-1, 1, _DECK_N)            # across
_DECK_V = _DECK.uniform(0, 1, _DECK_N)             # depth (0 near .. 1 far)
_DECK_R = _DECK.uniform(0.6, 1.3, _DECK_N)
_DECK_P = _DECK.uniform(0, TAU, _DECK_N)


def cloud_deck(c, t, cx, y, width=5600, thick=150, open_=0.0, alpha=0.55, lit=0.0, color='#2b3970',
               lit_color='#ffcf7a', ring_r=1500, drift=8.0, ring_thick=1.0):
    """A thin layer of soft clouds seen from far below (a band at world y). open_ 0..1 blows a
    ring-shaped hole around x=cx (the pillar), pushing the clouds outward into a billowing
    ring that is lit warm on its inside edge by `lit`."""
    if alpha <= 0:
        return
    R = ring_r * (1 - (1 - clamp(open_)) ** 2.2) if open_ > 0 else 0.0
    K = 0.12   # vertical squash of the horizontal cloud plane (seen from far below)
    items = []
    for i in range(_DECK_N):
        u, v, rs, ph = _DECK_U[i], _DECK_V[i], _DECK_R[i], _DECK_P[i]
        px = cx + u * width / 2 + ((t * drift * (1.2 - v)) % 200.0) - 100.0
        pz = (v - 0.5) * width * 0.75          # plane depth coordinate (+ = far)
        rx = (200 + 240 * rs) * (1.25 - 0.5 * v)
        dx, dz = px - cx, pz
        d = math.hypot(dx, dz)
        push = 0.0
        if R > 0 and d < R + rx * 0.6:
            nd = R + rx * 0.35 + (d / (R + rx)) * rx * 0.3
            k = nd / max(d, 1.0)
            dx, dz = dx * k, dz * k
            push = clamp((nd - d) / 600.0)
        sx = cx + dx
        sy = y + dz * K + math.sin(ph + t * 0.3) * 6
        items.append((v, sx, sy, rx, push, math.hypot(dx, dz)))
    items.sort(key=lambda it: -it[0])  # far first
    for v, sx, sy, rx, push, dd in items:
        ry = rx * (0.28 + 0.06 * (1 - v))
        a = alpha * (0.55 + 0.45 * (1 - v)) * (1 - 0.35 * push)
        warm = clamp(lit * math.exp(-max(0.0, dd - R) / 520.0)) if lit > 0 else 0.0
        cc = mix(color, lit_color, warm * 0.85)
        with at(c, sx, sy, sx=1.0, sy=ry / rx):
            c.drawCircle(0, 0, rx, radial((0, 0), rx, [(0, cc, a), (0.55, cc, a * 0.6), (1, cc, 0)]))
            if warm > 0.02:
                c.drawCircle(0, 0, rx * 0.8, radial((0, rx * 0.3), rx * 0.8,
                                                  [(0, lit_color, 0.5 * warm * alpha), (1, lit_color, 0)],
                                                  paint=skia.Paint(AntiAlias=True, BlendMode=ADD)))
    # the billowing torus rim of the ring, lit from inside by the pillar
    if R > 0:
        nrim = 30
        aa = alpha * 0.85 * clamp(open_ * 3)
        for k in range(nrim):
            a0 = k / nrim * TAU + 0.3
            rr = R * (1.0 + 0.08 * math.sin(k * 2.3))
            px = cx + math.cos(a0) * rr
            py = y + math.sin(a0) * rr * K
            rx = (260 + 60 * math.sin(k * 1.7)) * ring_thick * (1.0 - 0.25 * math.sin(a0))
            near = math.sin(a0) < 0
            cc = mix(color, lit_color, clamp(lit) * (0.45 if near else 0.9))
            with at(c, px, py, sy=0.34):
                c.drawCircle(0, 0, rx, radial((0, 0), rx, [(0, cc, aa), (0.6, cc, aa * 0.5), (1, cc, 0)]))


# ----------------------------------------------------------------------------
# FALLBACK: swirling Milky Way (whale's arrival)
# ----------------------------------------------------------------------------
_SW = nprng(3107)
_SW_N = 900
_SW_ARM = _SW.integers(0, 3, _SW_N)
_SW_R = np.sqrt(_SW.uniform(0.02, 1.0, _SW_N))
_SW_J = _SW.normal(0, 0.22, _SW_N)
_SW_S = _SW.uniform(0.8, 3.2, _SW_N)
_SW_B = _SW.uniform(0.3, 1.0, _SW_N)
_SW_C = _SW.integers(0, 3, _SW_N)


def sky_swirl(c, t, cx, cy, radius=1600, amount=1.0, spin=0.0, persp=0.55):
    """A great spiral of stars and nebula turning around (cx, cy). spin: rotation (radians)."""
    if amount <= 0:
        return
    # nebula glow
    for rr, colr, a in ((radius * 1.1, '#3a3f9a', 0.30), (radius * 0.7, '#7a58c8', 0.22),
                        (radius * 0.35, '#9fc4ff', 0.28), (radius * 0.12, '#e8f2ff', 0.5)):
        with at(c, cx, cy, sy=persp):
            p = radial((0, 0), rr, [(0, colr, a * amount), (0.5, colr, a * 0.4 * amount), (1, colr, 0)])
            p.setBlendMode(ADD)
            c.drawCircle(0, 0, rr, p)
    # spiral arms as soft dust lanes
    for arm in range(3):
        pts = []
        for k in range(40):
            r = 0.08 + k / 39 * 0.95
            th = arm * TAU / 3 + spin * (1.6 - r) + 2.6 * math.log(r * 8 + 1)
            pts.append((cx + math.cos(th) * r * radius, cy + math.sin(th) * r * radius * persp))
        path = skia.Path()
        path.moveTo(*pts[0])
        for q in pts[1:]:
            path.lineTo(*q)
        p = stroke('#8fa8ff', radius * 0.16, 0.10 * amount, blur=radius * 0.05)
        p.setBlendMode(ADD)
        c.drawPath(path, p)
    # stars along the arms (differential rotation => swirl)
    r = _SW_R
    th = _SW_ARM * TAU / 3 + spin * (1.6 - r) + 2.6 * np.log(r * 8 + 1) + _SW_J
    xs = cx + np.cos(th) * r * radius
    ys = cy + np.sin(th) * r * radius * persp
    tw = 0.6 + 0.4 * np.sin(t * 2.3 + np.arange(_SW_N) * 1.7)
    cols = ('#ffffff', '#cfe3ff', '#fff6e0')
    for ci in range(3):
        for sb in (0, 1):
            m = (_SW_C == ci) & ((_SW_S > 2.0) == bool(sb))
            if not m.any():
                continue
            a = float(np.mean(_SW_B[m] * tw[m])) * amount
            p = skia.Paint(AntiAlias=True, Color=col(cols[ci], clamp(a)), StrokeWidth=3.6 if sb else 2.0,
                           StrokeCap=skia.Paint.kRound_Cap, BlendMode=ADD)
            pts = [skia.Point(float(x), float(y)) for x, y in zip(xs[m], ys[m])]
            c.drawPoints(skia.Canvas.kPoints_PointMode, pts, p)


# ----------------------------------------------------------------------------
# FALLBACK: the big iron wheel in the lamp room (if lib.sea.lamp_room has no wheel)
# ----------------------------------------------------------------------------
def iron_wheel(c, x, y, r, rot, light=0.4):
    """Iron hand-wheel with spokes, brass hub, handles; a gear meshing behind it."""
    # gear behind (turns opposite)
    gx, gy, gr = x - r * 0.95, y + r * 0.55, r * 0.55
    with at(c, gx, gy, rot=-rot * 1.8):
        teeth = []
        n = 14
        for k in range(n * 2):
            a = k / (n * 2) * TAU
            rr = gr if k % 2 == 0 else gr * 0.84
            teeth.append((math.cos(a) * rr, math.sin(a) * rr))
        c.drawPath(poly(teeth), fill('#1f2336'))
        c.drawCircle(0, 0, gr * 0.72, fill('#262b40'))
        c.drawCircle(0, 0, gr * 0.2, fill('#8a6a3a'))
    # axle bracket
    c.drawPath(poly([(x - 30, y), (x + 30, y), (x + 60, y + r * 1.6), (x - 60, y + r * 1.6)]), fill('#1b1f30'))
    with at(c, x, y, rot=rot):
        c.drawCircle(0, 0, r, stroke('#2a2f45', r * 0.16))
        c.drawCircle(0, 0, r * 1.02, stroke('#4a5270', r * 0.03, 0.8))
        for k in range(8):
            a = k / 8 * TAU
            c.drawLine(0, 0, math.cos(a) * r, math.sin(a) * r, stroke('#2a2f45', r * 0.07))
            hx, hy = math.cos(a) * r * 1.16, math.sin(a) * r * 1.16
            c.drawLine(math.cos(a) * r, math.sin(a) * r, hx, hy, stroke('#2a2f45', r * 0.07))
            c.drawCircle(hx, hy, r * 0.075, fill('#8a6a3a'))
        c.drawCircle(0, 0, r * 0.2, fill('#c9a15b'))
        c.drawCircle(-r * 0.05, -r * 0.05, r * 0.08, fill('#ecd08a', 0.8))
    if light > 0:
        c.drawArc(skia.Rect.MakeXYWH(x - r, y - r, 2 * r, 2 * r), 190, 80, False,
                  stroke('#ffd36b', r * 0.05, 0.35 * light))
