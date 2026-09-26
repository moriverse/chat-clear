"""Private helpers for s02 (owner: SCENE-S02).

- future-proof library calls (only pass kwargs a function accepts; transient library bugs
  cost one element instead of the whole frame)
- cheap "ghost" beams for the motion-blurred sweep and a fallback time-lapse disc
- star-trail fallback (used only if lib.sky has no star_trails)
Everything is a pure function of its arguments.
"""
from __future__ import annotations

import dataclasses
import inspect
import math
import sys
import traceback

import numpy as np
import skia

from engine.core import ADD, TAU, clamp, hexrgb, linear, nprng, radial
from lib import robot, sea

# ----------------------------------------------------------------------------
# signature-safe calls
# ----------------------------------------------------------------------------
_SIG = {}


def accepts(fn, name):
    key = id(fn)
    if key not in _SIG:
        try:
            ps = inspect.signature(fn).parameters
            var = any(p.kind == p.VAR_KEYWORD for p in ps.values())
            _SIG[key] = (set(ps), var)
        except (TypeError, ValueError):
            _SIG[key] = (set(), True)
    names, var = _SIG[key]
    return var or name in names


def call(fn, *args, **kw):
    """Call fn passing only the keyword args it accepts."""
    kw = {k: v for k, v in kw.items() if accepts(fn, k)}
    return fn(*args, **kw)


_WARNED = set()


def safe(fn, *args, **kw):
    """Like call(), but a library exception only costs that element (warned once on stderr).
    Libraries are rewritten in parallel; a transient bug must not kill the frame."""
    try:
        return call(fn, *args, **kw)
    except Exception as e:  # noqa: BLE001
        name = getattr(fn, '__name__', str(fn))
        if name not in _WARNED:
            _WARNED.add(name)
            print(f'[s02] library call {name} failed: {e!r}', file=sys.stderr)
            traceback.print_exc(limit=3)
        return None


_POSE_FIELDS = None


def pose(**kw):
    """RobotPose with only the fields the current lib.robot knows about."""
    global _POSE_FIELDS
    if _POSE_FIELDS is None:
        _POSE_FIELDS = {fl.name for fl in dataclasses.fields(robot.RobotPose)}
    return robot.RobotPose(**{k: v for k, v in kw.items() if k in _POSE_FIELDS})


def anchors(kw, t):
    """Robot anchor points without drawing (falls back to a throw-away recording)."""
    p = pose(**kw)
    if hasattr(robot, 'anchors'):
        return robot.anchors(p, t)
    rec = skia.PictureRecorder()
    cc = rec.beginRecording(skia.Rect.MakeLTRB(-20000, -20000, 20000, 20000))
    out = robot.draw_robot(cc, p, t) or {}
    rec.finishRecordingAsPicture()
    return out


def hand_rest(kw, side, t):
    """World position of a hand with the arm override removed (for blending into a reach)."""
    k = dict(kw)
    k.pop('arm_%s_override' % side, None)
    return anchors(k, t)[side]


# ----------------------------------------------------------------------------
# beams
# ----------------------------------------------------------------------------
BEAM_COL = '#ffe7a0'


def beam_ghost(c, x, y, az, alpha, length, w0, w1, horizon_y, color=BEAM_COL):
    """Cheap flat copy of the sweeping beam (same projected geometry as sea.beam) used for
    the motion-blur samples of a fast-spinning beam."""
    if alpha <= 0.005:
        return
    geo = getattr(sea, 'beam_geometry', None)
    if geo is None:
        return
    pts, facing = call(geo, x, y, az, length=length, width0=w0, width1=w1, mode='sweep', horizon_y=horizon_y)
    if len(pts) < 2:
        return
    left, right = [], []
    m = len(pts)
    for i, (sx, sy, hw, L) in enumerate(pts):
        j0, j1 = max(0, i - 1), min(m - 1, i + 1)
        tx, ty = pts[j1][0] - pts[j0][0], pts[j1][1] - pts[j0][1]
        tl = math.hypot(tx, ty) or 1.0
        nx, ny = -ty / tl, tx / tl
        left.append((sx + nx * hw, sy + ny * hw))
        right.append((sx - nx * hw, sy - ny * hw))
    path = skia.Path()
    path.moveTo(*left[0])
    for q in left[1:]:
        path.lineTo(*q)
    for q in reversed(right):
        path.lineTo(*q)
    path.close()
    k = alpha * (0.5 + 0.3 * max(0.0, facing))
    p = linear(pts[0][:2], pts[-1][:2], [(0, color, 0.55 * k), (0.4, color, 0.25 * k), (1, color, 0.0)])
    p.setBlendMode(ADD)
    p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, 10))
    c.drawPath(path, p)


def beam_disc(c, x, y, R, ry, intensity=1.0, color=BEAM_COL, droop=0.0):
    """Fallback: the spinning beam seen in a long exposure — a flat luminous disc."""
    if intensity <= 0.01:
        return
    c.save()
    c.translate(x, y + droop)
    c.scale(1.0, ry / R)
    p = radial((0, 0), R, [(0.0, '#ffffff', 0.6 * intensity), (0.03, color, 0.5 * intensity),
                           (0.12, color, 0.34 * intensity), (0.35, color, 0.18 * intensity),
                           (0.7, color, 0.07 * intensity), (1.0, color, 0.0)])
    p.setBlendMode(ADD)
    c.drawCircle(0, 0, R, p)
    c.restore()


# ----------------------------------------------------------------------------
# star trails fallback
# ----------------------------------------------------------------------------
_TRAILS = None
_TRAIL_COLS = ('#fff6e0', '#cfe3ff', '#ffd9a8', '#a9c8ff')


def _trail_data():
    global _TRAILS
    if _TRAILS is None:
        r = nprng(20202)
        n = 700
        rad = np.sqrt(r.uniform(0.0, 1.0, n)) * 2300 + 6
        ang = r.uniform(0, TAU, n)
        mag = r.power(3.2, n)
        cls = r.integers(0, len(_TRAIL_COLS), n)
        _TRAILS = (rad, ang, mag, cls)
    return _TRAILS


def star_trails(c, cx, cy, theta, sweep, alpha=1.0, horizon_y=None):
    """Concentric star trails around (cx, cy); theta = sky rotation (CCW), sweep = trail arc."""
    if alpha <= 0.01:
        return
    rad, ang, mag, cls = _trail_data()
    sweep = clamp(sweep, 0.0, TAU * 0.985)
    sdeg = math.degrees(sweep)
    frac = min(0.999, max(0.002, sweep / TAU))
    shaders = []
    for cc in _TRAIL_COLS:
        rgb = hexrgb(cc)
        shaders.append(skia.GradientShader.MakeSweep(
            cx, cy, [skia.Color4f(*rgb, 1.0).toColor(), skia.Color4f(*rgb, 0.0).toColor(),
                     skia.Color4f(*rgb, 0.0).toColor()], [0.0, frac, 1.0]))
    p = skia.Paint(AntiAlias=True, Style=skia.Paint.kStroke_Style)
    p.setStrokeCap(skia.Paint.kRound_Cap)
    p.setBlendMode(ADD)
    hmax = 1e9 if horizon_y is None else horizon_y
    for i in range(len(rad)):
        R = rad[i]
        if cy - R > hmax or sdeg < 0.4:
            continue
        head = ang[i] - theta
        p.setShader(shaders[cls[i]].makeWithLocalMatrix(
            skia.Matrix.RotateDeg(math.degrees(head), skia.Point(cx, cy))))
        p.setStrokeWidth(0.7 + 2.4 * mag[i] * mag[i])
        p.setAlphaf(clamp(alpha * (0.18 + 0.82 * mag[i])))
        c.drawArc(skia.Rect.MakeLTRB(cx - R, cy - R, cx + R, cy + R), math.degrees(head), sdeg, False, p)
