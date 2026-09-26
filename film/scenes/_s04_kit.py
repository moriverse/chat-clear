"""Private helpers for s04 (The meeting): robot-arm IK, local fallbacks for library
features that may not exist yet, and small scene-specific drawing (underwater view,
water drips, oars).  Everything is a pure function of its inputs.
"""
from __future__ import annotations

import dataclasses
import inspect
import math

import skia

from engine.core import (ADD, TAU, at, clamp, col, fill, lerp, linear, mix, noise1, nprng,
                         poly, radial, smooth_path, soft_glow, stroke)
from lib import fx, robot, sea, star


# ----------------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------------
def has_field(cls, name):
    try:
        return name in cls.__dataclass_fields__
    except Exception:
        return False


def accepts(fn, name):
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


_WARNED = set()


def call_kw(fn, *args, _default=None, **kw):
    """Call a library function with only the keyword args it accepts (libraries evolve
    in parallel).  A library exception is reported once and the call is skipped, so a
    half-edited library never takes the whole scene down."""
    try:
        params = inspect.signature(fn).parameters
        if not any(p.kind == p.VAR_KEYWORD for p in params.values()):
            kw = {k: v for k, v in kw.items() if k in params}
    except (TypeError, ValueError):
        pass
    try:
        return fn(*args, **kw)
    except Exception as e:  # noqa: BLE001
        key = (getattr(fn, '__module__', '?'), getattr(fn, '__name__', '?'))
        if key not in _WARNED:
            _WARNED.add(key)
            import sys
            print(f'[s04] library call {key[0]}.{key[1]} failed: {e!r}', file=sys.stderr)
        return _default


def make_pose(cls, **kw):
    """Build a dataclass pose, silently dropping fields the library doesn't know yet."""
    fields = getattr(cls, '__dataclass_fields__', {})
    return cls(**{k: v for k, v in kw.items() if k in fields})


def expr_at(t, keys, fade=0.25):
    """Expression keyframes [(time, name), ...] -> (eyes, eyes2, eyes_mix) with crossfades."""
    from engine.core import smoothstep
    prev, cur, t0 = keys[0][1], keys[0][1], keys[0][0]
    for (tk, name) in keys[1:]:
        if t >= tk:
            prev, cur, t0 = cur, name, tk
        else:
            break
    k = smoothstep(t0, t0 + fade, t)
    if k >= 1 or prev == cur:
        return cur, cur, 0.0
    return prev, cur, k


def bump(t, a, b, fade_in=0.25, fade_out=0.35):
    """Smooth 0→1→0 envelope: rises at a, falls at b."""
    from engine.core import smoothstep
    return smoothstep(a, a + fade_in, t) * (1 - smoothstep(b, b + fade_out, t))


def hop(t, t0, dur=0.35, amp=1.0):
    """A single sine hop starting at t0."""
    u = (t - t0) / dur
    if u <= 0 or u >= 1:
        return 0.0
    return math.sin(u * math.pi) * amp


# ----------------------------------------------------------------------------
# robot arm IK
# ----------------------------------------------------------------------------
_NULL_BOUNDS = skia.Rect.MakeWH(4000, 3000)


def _probe(pose, t):
    rec = skia.PictureRecorder()
    cv = rec.beginRecording(skia.Rect.MakeXYWH(-4000, -4000, 12000, 12000))
    out = robot.draw_robot(cv, pose, t)
    rec.finishRecordingAsPicture()
    return out


def robot_rig(pose, t):
    """Measure the robot's arm rig for `pose` (all fields except arms matter).

    Returns dict with per-side shoulder S, basis vectors (down, fwd) scaled to full arm
    length, segment fractions, plus the probe's 'heart'/'head'/'eyes'/'antenna' points.
    Uses the documented arm convention: shoulder 0 = down, +pi/2 = forward, +pi = up."""
    if has_field(type(pose), 'arm_l_override'):
        a = anchors(pose, t)
        rig = dict(a)
        rig.setdefault('head_top', (a['head'][0], a['head'][1] - 60 * pose.scale))
        return rig
    P = dataclasses.replace
    r_down = _probe(P(pose, arm_l=(0.0, 0.0), arm_r=(0.0, 0.0)), t)
    r_up = _probe(P(pose, arm_l=(math.pi, 0.0), arm_r=(math.pi, 0.0)), t)
    r_fwd = _probe(P(pose, arm_l=(math.pi / 2, 0.0), arm_r=(math.pi / 2, 0.0)), t)
    r_bent = _probe(P(pose, arm_l=(0.0, math.pi / 2), arm_r=(0.0, math.pi / 2)), t)
    rig = {k: r_down.get(k) for k in ('heart', 'head', 'eyes', 'antenna')}
    for side in ('l', 'r'):
        hd, hu, hf, hb = r_down[side], r_up[side], r_fwd[side], r_bent[side]
        S = ((hd[0] + hu[0]) / 2, (hd[1] + hu[1]) / 2)
        D = (hd[0] - S[0], hd[1] - S[1])      # full arm pointing down
        F = (hf[0] - S[0], hf[1] - S[1])      # full arm pointing forward
        # hb - S = a*D + b*F  ->  a = l1/L, b = l2/L
        det = D[0] * F[1] - D[1] * F[0]
        vx, vy = hb[0] - S[0], hb[1] - S[1]
        if abs(det) < 1e-6:
            a, b = 0.5, 0.5
        else:
            a = (vx * F[1] - vy * F[0]) / det
            b = (D[0] * vy - D[1] * vx) / det
        rig[side] = dict(S=S, D=D, F=F, a=max(0.05, a), b=max(0.05, b), det=det)
    return rig


def ik_arm(rig_side, target, bend=1.0, reach=0.985):
    """Solve (shoulder, elbow) so the hand lands on world `target`.
    bend=+1 → natural elbow (forearm rotates towards forward), -1 → other way."""
    S, D, F, det = rig_side['S'], rig_side['D'], rig_side['F'], rig_side['det']
    l1, l2 = rig_side['a'], rig_side['b']
    vx, vy = target[0] - S[0], target[1] - S[1]
    if abs(det) < 1e-6:
        return (0.0, 0.0)
    # express target in (down, fwd) basis, units of full arm length
    y = (vx * F[1] - vy * F[0]) / det     # along "down"
    x = (D[0] * vy - D[1] * vx) / det     # along "forward"
    d = math.hypot(x, y)
    dmax = (l1 + l2) * reach
    dmin = abs(l1 - l2) + 1e-3
    if d > dmax:
        x, y, d = x * dmax / d, y * dmax / d, dmax
    elif d < dmin:
        if d < 1e-6:
            x, y, d = dmin, 0.0, dmin
        else:
            x, y, d = x * dmin / d, y * dmin / d, dmin
    ce = clamp((d * d - l1 * l1 - l2 * l2) / (2 * l1 * l2), -1.0, 1.0)
    e = math.acos(ce) * (1 if bend >= 0 else -1)
    phi = math.atan2(x, y)            # angle from "down" towards "forward"
    a = phi - math.atan2(l2 * math.sin(e), l1 + l2 * math.cos(e))
    return (a, e)


def pose_with_hands(pose, t, target_l=None, target_r=None, bend_l=1.0, bend_r=1.0, rig=None):
    """Return (pose', rig) with arms reaching the given world targets.  Uses the robot
    library's own IK overrides when available (arm_*_override), else a local 2-bone IK."""
    if has_field(type(pose), 'arm_l_override'):
        kw = {}
        if target_l is not None:
            kw['arm_l_override'] = (float(target_l[0]), float(target_l[1]))
        if target_r is not None:
            kw['arm_r_override'] = (float(target_r[0]), float(target_r[1]))
        return dataclasses.replace(pose, **kw), rig
    rig = rig or robot_rig(pose, t)
    kw = {}
    if target_l is not None:
        kw['arm_l'] = ik_arm(rig['l'], target_l, bend_l)
    if target_r is not None:
        kw['arm_r'] = ik_arm(rig['r'], target_r, bend_r)
    return dataclasses.replace(pose, **kw), rig


def shoulder(rig, side):
    if 'shoulder_' + side in rig:
        return rig['shoulder_' + side]
    return rig[side]['S']


def overdraw_hand(c, pose, t, hand_xy, s, r=10.5):
    """Re-draw only the palm of a robot hand (clipped disc inside the palm) so it sits in
    front of something drawn after the robot (Guang cradled in his mittens)."""
    path = skia.Path()
    path.addCircle(hand_xy[0], hand_xy[1], r * s)
    c.save()
    c.clipPath(path, skia.ClipOp.kIntersect, True)
    robot.draw_robot(c, pose, t)
    c.restore()


def anchors(pose, t):
    """World anchor points of the robot for `pose` without drawing it."""
    if hasattr(robot, 'anchors'):
        try:
            return robot.anchors(pose, t)
        except Exception:  # noqa: BLE001
            pass
    return _probe(pose, t)


# ----------------------------------------------------------------------------
# underwater split view (shot 1) — local art, the sea library has no underwater view
# ----------------------------------------------------------------------------
def surface_y(x, t, y_s, amp=1.0, disturb=()):
    """Water-line height at x.  disturb: iterable of (x0, t_since, strength)."""
    y = y_s + amp * (4.0 * math.sin(x * 0.011 + t * 1.7) + 2.5 * math.sin(x * 0.027 - t * 2.3)
                     + 1.5 * math.sin(x * 0.051 + t * 3.1))
    for (x0, ts, k) in disturb:
        if 0 <= ts < 3.0:
            dx = abs(x - x0)
            wave_r = 40 + ts * 260
            env = math.exp(-((dx - wave_r) / 70.0) ** 2) * (1 - ts / 3.0) ** 2
            y += k * 16 * env * math.sin(dx * 0.05 - ts * 11)
            # the central hump where something breaks the surface
            y -= k * 10 * math.exp(-(dx / 50.0) ** 2) * max(0.0, 1 - ts * 2)
    return y


def surface_path(t, y_s, x0, x1, bottom, amp=1.0, disturb=(), step=24):
    pts = []
    x = x0
    while x <= x1 + step:
        pts.append((x, surface_y(x, t, y_s, amp, disturb)))
        x += step
    path = skia.Path()
    path.moveTo(*pts[0])
    for p in pts[1:]:
        path.lineTo(*p)
    line = skia.Path(path)
    path.lineTo(x1 + step, bottom)
    path.lineTo(x0, bottom)
    path.close()
    return path, line


def underwater_back(c, t, y_s, x0, x1, bottom, glow_xy=None, glow_amt=0.0, seed=41):
    """Underwater backdrop below the surface line: depth gradient, light shafts, plankton."""
    c.drawRect(skia.Rect.MakeLTRB(x0, y_s - 30, x1, bottom),
               linear((0, y_s), (0, y_s + 900), [(0, '#1b3f78', 1), (0.35, '#0e2550', 1), (1, '#040a1c', 1)]))
    # soft slanted light shafts from the surface (moonlight)
    r = nprng(seed)
    for i in range(7):
        bx = x0 + (x1 - x0) * (i + r.uniform(0.1, 0.9)) / 7 + 40 * math.sin(t * 0.3 + i)
        w0 = r.uniform(40, 110)
        ln = r.uniform(420, 760)
        sl = 0.28
        a = 0.06 + 0.04 * math.sin(t * 0.7 + i * 1.7)
        p = linear((bx, y_s), (bx + sl * ln, y_s + ln), [(0, '#9fc4ff', a), (1, '#9fc4ff', 0)])
        p.setBlendMode(ADD)
        c.drawPath(poly([(bx - w0 / 2, y_s), (bx + w0 / 2, y_s),
                         (bx + sl * ln + w0 * 0.9, y_s + ln), (bx + sl * ln - w0 * 0.9, y_s + ln)]), p)
    # plankton specks
    for i in range(70):
        px = x0 + (r.uniform(0, 1) * (x1 - x0) + t * r.uniform(-8, 8)) % (x1 - x0)
        py = y_s + 20 + (r.uniform(0, 1) * (bottom - y_s - 20) - t * r.uniform(2, 10)) % max(1, bottom - y_s - 20)
        tw = 0.5 + 0.5 * math.sin(t * r.uniform(1.5, 3.5) + i)
        near = 0.0
        if glow_xy is not None:
            near = max(0.0, 1 - math.hypot(px - glow_xy[0], py - glow_xy[1]) / 500) * glow_amt
        colr = mix('#6ff0ff', '#ffd86b', near)
        c.drawCircle(px, py, r.uniform(1.2, 2.8), fill(colr, (0.25 + 0.9 * near) * tw, blend=ADD))


def bubbles(c, t, x, y, y_s, count=10, spread=40.0, seed=12, rate=1.0, alpha=0.7):
    """Tiny bubbles rising from (x, y) to the surface y_s (pure function of t)."""
    r = nprng(seed)
    h = max(10.0, y - y_s)
    for i in range(count):
        speed = r.uniform(60, 120)
        per = h / speed + r.uniform(0.2, 0.9)
        ph = (t * rate + r.uniform(0, per)) % per
        by = y - ph * speed
        if by < y_s + 4:
            continue
        bx = x + r.uniform(-spread, spread) * 0.4 + math.sin(t * 3 + i) * 5 + (y - by) * r.uniform(-0.1, 0.1)
        rr = r.uniform(1.5, 4.0) * (1 + (y - by) / h * 0.4)
        c.drawCircle(bx, by, rr, stroke('#cfe8ff', 1.0, alpha * 0.7))
        c.drawCircle(bx - rr * 0.35, by - rr * 0.35, rr * 0.3, fill('#ffffff', alpha * 0.8))


def underwater_front(c, t, y_s, x0, x1, bottom, disturb=(), glow_xy=None, glow_amt=0.0, a=0.5):
    """Tint everything below the water line (drawn over submerged objects) + surface lines."""
    path, line = surface_path(t, y_s, x0, x1, bottom, disturb=disturb)
    tint = linear((0, y_s), (0, y_s + 700), [(0, '#123a70', a * 0.75), (1, '#061230', a)])
    c.drawPath(path, tint)
    if glow_xy is not None and glow_amt > 0:
        gx, gy = glow_xy
        with at(c):
            c.clipPath(path, doAntiAlias=True)
            soft_glow(c, gx, gy, 460, '#6fdcff', 0.14 * glow_amt)
            soft_glow(c, gx, gy, 200, '#ffd86b', 0.16 * glow_amt)
            # light hitting the underside of the surface above the glow
            soft_glow(c, gx, y_s + 12, 380, '#9fe6ff', 0.14 * glow_amt)
    # underside of the surface: bright silvery band + thin meniscus line
    c.drawPath(line, stroke('#6f9fe0', 9, 0.35, blur=5))
    c.drawPath(line, stroke('#cfe6ff', 2.2, 0.75))


def above_water_sheen(c, t, y_s, x0, x1, lights=()):
    """Thin horizontal glints on the water line from lights above (lantern, star)."""
    for (lx, ly, colr, k) in lights:
        for j in range(5):
            w = 60 + j * 38
            yy = y_s - 2 + j * 1.5
            xx = lx + 14 * math.sin(t * 2.1 + j * 1.3)
            p = linear((xx - w, 0), (xx + w, 0), [(0, colr, 0), (0.5, colr, 0.35 * k * (1 - j / 5)), (1, colr, 0)])
            p.setBlendMode(ADD)
            c.drawRect(skia.Rect.MakeXYWH(xx - w, yy, 2 * w, 3), p)


# ----------------------------------------------------------------------------
# drips (fallback for fx.water_drips)
# ----------------------------------------------------------------------------
def drips(c, t, src_fn, t_start, t_end, water_y, width=60.0, rate=9.0, seed=5, colr='#dff2ff',
          glow_col='#ffe7a0', ripple=True, g=1500.0):
    """Drops that detach from a moving source and fall into the water.
    src_fn(time) -> (x, y) of the dripping bottom edge at that time.
    Emission between t_start and t_end, with rate decaying over time."""
    r = nprng(seed)
    n = int((t_end - t_start) * rate) + 1
    ems = []
    for i in range(n):
        te = t_start + (i + r.uniform(0, 0.9)) / rate
        ems.append((te, r.uniform(-1, 1), r.uniform(0.6, 1.25), r.uniform(0, 1)))
    for (te, off, size, keep) in ems:
        if te > t_end or te > t:
            continue
        # fewer drops later in the window
        if keep > 1.0 - 0.75 * ((te - t_start) / max(1e-3, t_end - t_start)) ** 0.6 and te > t_start + 0.5:
            continue
        age = t - te
        sx, sy = src_fn(te)
        x = sx + off * width * 0.5
        y0 = sy + 4
        y = y0 + 0.5 * g * age * age
        if y < water_y - 3:
            if age < 0.08:  # hanging, stretching
                s = age / 0.08
                c.drawOval(skia.Rect.MakeXYWH(x - 3 * size, y0 - 2, 6 * size, (4 + 8 * s) * size), fill(colr, 0.85))
            else:
                v = g * age
                ln = clamp(v * 0.012, 4, 22) * size
                c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x - 2.6 * size, y - ln, 5.2 * size, ln + 5 * size),
                                                  2.6 * size, 2.6 * size), fill(colr, 0.8))
                c.drawCircle(x - 1 * size, y - 1 * size, 1.3 * size, fill('#ffffff', 0.9))
                soft_glow(c, x, y, 10 * size, glow_col, 0.25)
        elif ripple:
            # hit time
            t_hit = math.sqrt(max(0.0, 2 * (water_y - y0) / g))
            ts = age - t_hit
            if 0 <= ts < 1.2:
                rr = 6 + ts * 55 * size
                a = (1 - ts / 1.2) ** 1.5 * 0.7
                c.drawOval(skia.Rect.MakeXYWH(x - rr, water_y - rr * 0.22, 2 * rr, 2 * rr * 0.22), stroke(colr, 1.8, a))
                if ts < 0.25:  # tiny crown
                    k = ts / 0.25
                    for j in (-1, 1):
                        c.drawCircle(x + j * (4 + 10 * k), water_y - 10 * math.sin(k * math.pi) * size, 1.6 * size,
                                     fill(colr, 0.8 * (1 - k)))


def splash_burst(c, t_since, x, y, scale=1.0, seed=3, colr='#dff2ff', up=1.0):
    """Small local splash when something breaks the water surface."""
    if t_since < 0 or t_since > 1.4:
        return
    r = nprng(seed)
    for i in range(26):
        ang = -math.pi / 2 + r.uniform(-1.0, 1.0)
        v = r.uniform(180, 520) * scale * up
        px = x + math.cos(ang) * v * t_since * 0.8 + r.uniform(-30, 30) * scale
        py = y + math.sin(ang) * v * t_since + 0.5 * 1500 * t_since * t_since
        if py < y + 6:
            sz = r.uniform(1.5, 4.5) * scale * (1 - t_since / 1.6)
            c.drawCircle(px, py, sz, fill(colr, 0.85 * (1 - t_since / 1.4)))


# ----------------------------------------------------------------------------
# oars (fallback when sea.boat does not draw them)
# ----------------------------------------------------------------------------
def oar(c, lock, handle_dir_angle, length_in, length_out, blade_depth_y=None, s=1.0, front=True):
    """An oar pivoting in an oarlock at `lock`.  Angle is of the blade side (radians)."""
    lx, ly = lock
    ca, sa = math.cos(handle_dir_angle), math.sin(handle_dir_angle)
    hx, hy = lx - ca * length_in, ly - sa * length_in
    bx, by = lx + ca * length_out, ly + sa * length_out
    c.drawLine(hx, hy, bx, by, stroke('#8a6440', 7 * s))
    c.drawLine(hx, hy, bx, by, stroke('#b08a5a', 2.5 * s, 0.6))
    with at(c, bx, by, rot=handle_dir_angle):
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(-34 * s, -9 * s, 50 * s, 18 * s), 8 * s, 8 * s), fill('#9a7048'))
    return (hx, hy), (bx, by)
