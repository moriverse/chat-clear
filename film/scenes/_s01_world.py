"""Shared exterior-set helpers for the bookend scenes s01 (opening) and s08 (epilogue).

* spline / log-spline keyframing with continuous velocity (smooth camera moves)
* the lighthouse "group" drawn with a dolly model: lighthouse base (x_b, y_b) in
  screen units and a scale s — the sky and sea stay at their designed scale while
  the lighthouse grows, like a real camera moving towards it
* a fake-3D sweeping beam (angle = T*0.9 by convention) with lamp flare and
  shimmering reflections on the water
* Deng standing on the gallery (lib.robot), with an extra glow for his heart
* hero star fallback (lib.sky.hero_star if the sky library provides one)
"""
from __future__ import annotations

import math

import skia

from engine.core import (ADD, TAU, clamp, col, fill, lerp, linear, mix, noise1, nprng, radial,
                         smoothstep, soft_glow, stroke)
from lib import fx, robot, sea, sky

WARM = '#ffd36b'
BEAM = '#ffe7a0'


# ----------------------------------------------------------------------------
# keyframing
# ----------------------------------------------------------------------------
def spline(t, keys):
    """Cubic Hermite through (time, value) keys, Catmull-Rom tangents, zero velocity
    at the first/last key. Values may be numbers or tuples."""
    ts = [k[0] for k in keys]
    if t <= ts[0]:
        return keys[0][1]
    if t >= ts[-1]:
        return keys[-1][1]
    i = 0
    while t > ts[i + 1]:
        i += 1
    t0, t1 = ts[i], ts[i + 1]
    h = t1 - t0
    u = (t - t0) / h
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2

    def comp(j):
        v = [k[1] if j is None else k[1][j] for k in keys]
        m0 = 0.0 if i == 0 else (v[i + 1] - v[i - 1]) / (ts[i + 1] - ts[i - 1])
        m1 = 0.0 if i + 1 == len(keys) - 1 else (v[i + 2] - v[i]) / (ts[i + 2] - ts[i])
        return h00 * v[i] + h10 * h * m0 + h01 * v[i + 1] + h11 * h * m1

    if isinstance(keys[0][1], (tuple, list)):
        return tuple(comp(j) for j in range(len(keys[0][1])))
    return comp(None)


def log_spline(t, keys):
    return math.exp(spline(t, [(k, math.log(v)) for k, v in keys]))


# ----------------------------------------------------------------------------
# lighthouse geometry (robust to library changes)
# ----------------------------------------------------------------------------
def gallery_local():
    """(floor_y, half_width) of the gallery in lighthouse-local units at s=1 (y up = negative)."""
    fy = getattr(sea, 'GALLERY_Y', None)
    hw = getattr(sea, 'GALLERY_HALF_W', None)
    if fy is None:
        fy = -405.0
    elif fy > 0:
        fy = -fy
    return fy, (hw if hw is not None else 64.0)


def lamp_local():
    ly = getattr(sea, 'LAMP_Y', None)
    if ly is None:
        return -445.0
    return -abs(ly)


# ----------------------------------------------------------------------------
# beam
# ----------------------------------------------------------------------------
def beam_state(T, phase=None):
    """Fake-3D beam: phase = T*0.9. Returns (screen_angle, length_factor, toward) where
    toward in -1..1 (+1 = beam pointing at the camera)."""
    ph = T * 0.9 if phase is None else phase
    dx, dz = math.cos(ph), math.sin(ph)      # dz > 0: towards camera
    ang = math.atan2(0.10 * dz, dx)
    lf = math.hypot(dx, 0.10 * dz)
    return ang, lf, dz


def draw_beam(c, T, lx, ly, s, part, intensity=1.0, reach=None, phase=None):
    """part='back' draws the beam if it points away from the camera (call before the
    lighthouse), part='front' if towards the camera (call after). The flare is part of
    'front'."""
    if intensity <= 0:
        return
    ang, lf, dz = beam_state(T, phase)
    towards = dz > 0
    if (part == 'back') == towards:
        return
    L = (reach if reach is not None else 950 + 2600 * s) * max(0.08, lf)
    w0 = 26 * s + 4
    w1 = (110 + 150 * s) * (1 + 0.9 * max(0.0, dz))
    inten = intensity * (0.4 + 0.3 * max(0.0, dz)) * (0.35 + 0.65 * smoothstep(0.0, 0.5, lf))
    sea.beam(c, lx, ly, ang, length=L, width0=w0, width1=w1, intensity=inten, color=BEAM)
    # soft wider halo around the cone
    sea.beam(c, lx, ly, ang, length=L * 0.7, width0=w0 * 3, width1=w1 * 1.8, intensity=inten * 0.25, color=WARM)


def lamp_flare(c, T, lx, ly, s, intensity=1.0, phase=None):
    """Big glow when the beam faces the camera + a constant lamp glow."""
    if intensity <= 0:
        return 0.0
    ang, lf, dz = beam_state(T, phase)
    f = max(0.0, dz) ** 6
    sk = s ** 0.6
    soft_glow(c, lx, ly, 45 * sk + 25, '#fff4d0', 0.45 * intensity)
    if f > 0.01:
        soft_glow(c, lx, ly, (170 * sk + 90) * (0.6 + 0.4 * f), '#ffe7b0', 0.45 * f * intensity)
        fx.lens_flare(c, lx, ly, intensity=0.6 * f * intensity)
    return f


def beam_glints(c, T, lx, base_y, s, horizon, intensity=1.0, seed=21, phase=None):
    """Shimmering light on the water under the beam's path + lamp reflection sparkle."""
    if intensity <= 0:
        return
    ang, lf, dz = beam_state(T, phase)
    ph = T * 0.9 if phase is None else phase
    dx = math.cos(ph)
    r = nprng(seed)
    n = 70
    p = fill(BEAM, 1.0, blend=ADD)
    reach = 1000 + 2400 * s
    us = r.uniform(0.05, 1.0, n)
    jit = r.normal(0, 1, n)
    ph2 = r.uniform(0, TAU, n)
    for i in range(n):
        u = us[i]
        d = u * reach
        x = lx + dx * d * lf / max(lf, 1e-3) * 1.0
        y = base_y + (horizon - base_y) * (u ** 0.5) * 0.9 + dz * d * 0.05
        y = max(horizon + 2, y + jit[i] * 3)
        tw = 0.5 + 0.5 * math.sin(T * 7 + ph2[i])
        a = intensity * (1 - u) ** 1.5 * tw * 0.5 * (0.4 + 0.6 * abs(dx))
        if a < 0.02:
            continue
        p.setAlphaf(clamp(a))
        w = 3 + 9 * (1 - u) * (0.5 + s)
        c.drawOval(skia.Rect.MakeXYWH(x - w, y - 0.8, 2 * w, 1.6 + (1 - u) * 1.5), p)


# ----------------------------------------------------------------------------
# hero star (the one that answers)
# ----------------------------------------------------------------------------
def hero_star(c, T, x, y, flare=0.0, scale=1.0, intensity=1.0, flare_age=None, warmth=1.0):
    """The answering star. Uses lib.sky.hero_star when available, otherwise a local one."""
    fn = getattr(sky, 'hero_star', None)
    if fn is not None:
        try:
            return fn(c, x, y, size=scale, flare=flare, t=T, warmth=warmth, intensity=intensity,
                      flare_age=flare_age)
        except TypeError:
            pass
    _hero_local(c, T, x, y, flare, scale, intensity)


def _hero_local(c, T, x, y, flare, scale, intensity):
    tw = 0.85 + 0.15 * math.sin(T * 2.3) + 0.06 * noise1(T * 3.0, 77)
    k = scale * intensity
    soft_glow(c, x, y, 70 * scale * (1 + 1.4 * flare), '#cfe3ff', 0.5 * k * tw + 0.5 * flare)
    soft_glow(c, x, y, 22 * scale * (1 + flare), '#fff8e8', 0.9 * k)
    # cross glints (4-point), longer when flaring
    for ang, ln in ((0.0, 1.0), (math.pi / 2, 1.0), (math.pi / 4, 0.45), (-math.pi / 4, 0.45)):
        L = (38 + 260 * flare) * scale * ln * tw
        wdt = (1.6 + 2.5 * flare) * scale * (1 if ln == 1.0 else 0.7)
        for sgn in (1, -1):
            p = linear((x, y), (x + math.cos(ang) * L * sgn, y + math.sin(ang) * L * sgn),
                       [(0, '#ffffff', min(1.0, 0.8 * k + flare)), (1, '#cfe3ff', 0.0)])
            p.setBlendMode(ADD)
            p.setStyle(skia.Paint.kStroke_Style)
            p.setStrokeWidth(wdt)
            p.setStrokeCap(skia.Paint.kRound_Cap)
            c.drawLine(x, y, x + math.cos(ang) * L * sgn, y + math.sin(ang) * L * sgn, p)
    c.drawCircle(x, y, 3.2 * scale * (1 + 0.6 * flare), fill('#ffffff', 1.0))


def flare_env(t, cue_t, attack=0.05, decay=0.55):
    """Sharp attack, smooth decay flare envelope around cue time."""
    d = t - cue_t
    if d < -attack:
        return 0.0
    if d < 0:
        return smoothstep(-attack, 0, d)
    return math.exp(-d / decay) * (1 - smoothstep(decay * 5, decay * 6, d))


def star_burst(c, T, x, y, since, scale=1.0, seed=40):
    """A little sparkle burst radiating from a star (since = seconds since the flare)."""
    if since < 0 or since > 1.6:
        return
    r = nprng(seed)
    n = 22
    for i in range(n):
        a = r.uniform(0, TAU)
        v = r.uniform(60, 170) * scale
        d = v * (1 - math.exp(-since * 3.0))
        px, py = x + math.cos(a) * d, y + math.sin(a) * d + 18 * since * since * scale
        al = (1 - since / 1.6) ** 1.5 * (0.6 + 0.4 * math.sin(since * 20 + i))
        sz = r.uniform(1.2, 2.6) * scale * (1 - since / 2)
        c.drawCircle(px, py, max(0.4, sz), fill('#fff4d6', clamp(al), blend=ADD))
        if i % 3 == 0:
            L = 7 * scale * (1 - since / 1.6)
            q = stroke('#ffffff', 0.9 * scale, clamp(al * 0.8))
            q.setBlendMode(ADD)
            c.drawLine(px - L, py, px + L, py, q)
            c.drawLine(px, py - L, px, py + L, q)


# ----------------------------------------------------------------------------
# the lighthouse group
# ----------------------------------------------------------------------------
def _anchors(x_b, y_b, s):
    fn = getattr(sea, 'lighthouse_anchors', None)
    if fn is not None:
        try:
            return fn(x_b, y_b, s)
        except Exception:
            pass
    fy, hw = gallery_local()
    return dict(lamp=(x_b, y_b + lamp_local() * s), gallery_y=y_b + fy * s,
                gallery_x=(x_b - hw * s, x_b + hw * s), deng_scale=0.11 * s, legacy=True)


def _lighthouse(c, T, x_b, y_b, s, lamp, parts, phase, glow=1.0):
    """Call sea.lighthouse with as many of the newer kwargs as it accepts."""
    try:
        return sea.lighthouse(c, x_b, y_b, s, lamp=lamp, t=T, beam_angle=phase, parts=parts, glow=glow), True
    except TypeError:
        if parts == 'front':
            return None, False
        return sea.lighthouse(c, x_b, y_b, s, lamp=lamp), False


def rail_seat(x_b, y_b, s, deng_x):
    """World (x, y) of the top of the front railing at relative position deng_x (-1..1)."""
    L = getattr(sea, 'LH', None) or {}
    r = L.get('rail_r', 66.0)
    ry = L.get('rail_y', -427.0)
    e = 0.09
    u = max(-0.98, min(0.98, deng_x * 0.97))
    ca = math.sqrt(1 - u * u)
    return x_b + u * r * s, y_b + (ry - e * r * ca) * s


def draw_lighthouse_group(c, T, x_b, y_b, s, horizon, pose_fn=None, lamp=1.0, beam=1.0,
                          deng_x=-0.3, deng_scale=None, glints=1.0, beam_phase=None,
                          island=True, deng_mode='behind'):
    """Island + lighthouse + beam + Deng on the gallery, with the lighthouse base at screen
    (x_b, y_b) and scale s. deng_x: position along the gallery (-1 left end .. 1 right end).
    pose_fn(pose) may modify the RobotPose (called with a default pose already placed on the
    gallery). Returns dict(lamp=(x,y), hands=..., deng=pose, floor_y=...)."""
    out = {}
    phase = T * 0.9 if beam_phase is None else beam_phase
    an = _anchors(x_b, y_b, s)
    lx, lamp_y = an['lamp']
    # reflection glints under the beam (on the water, behind everything else)
    if glints > 0 and beam > 0:
        beam_glints(c, T, lx, y_b, s, horizon, intensity=glints * beam, phase=phase)
    draw_beam(c, T, lx, lamp_y, s, 'back', intensity=beam, phase=phase)
    if island:
        sea.island(c, x_b, y_b, s * 0.55)
    res, split = _lighthouse(c, T, x_b, y_b, s, lamp, 'back', phase)
    if isinstance(res, (tuple, list)) and len(res) >= 2:
        lx, lamp_y = res[0], res[1]
    out['lamp'] = (lx, lamp_y)
    floor_y = an['gallery_y']
    gx0, gx1 = an['gallery_x']
    ds = an.get('deng_scale', 0.1 * s) if deng_scale is None else deng_scale * s
    front_glow = 1.0 / (1.0 + 0.35 * max(0.0, s - 1.0))

    def _deng():
        if pose_fn is None or ds * robot.HEIGHT <= 2.0:
            return
        if deng_mode == 'on_rail':
            px, py = rail_seat(x_b, y_b, s, deng_x)
            pose = robot.RobotPose(x=px, y=py, scale=ds, sit=1.0)
        else:
            pose = robot.RobotPose(x=lerp((gx0 + gx1) / 2, gx1 if deng_x > 0 else gx0, abs(deng_x)),
                                   y=floor_y, scale=ds)
        pose.rim_color = '#ffd9a0'
        pose.rim = 0.55
        pose.rim_angle = math.atan2(lamp_y - (pose.y - 200 * ds), lx - pose.x)
        pose.ambient = '#b8c2e8'
        pose.shadow = 0.0
        ctx = dict(s=s, lamp=(lx, lamp_y), rail_y=rail_seat(x_b, y_b, s, deng_x)[1], floor_y=floor_y,
                   x_b=x_b, y_b=y_b)
        pose = pose_fn(pose, ctx) or pose
        hands = robot.draw_robot(c, pose, T)
        out['hands'] = hands
        out['deng'] = pose
        # extra warm heart glow so it reads even when small
        if isinstance(hands, dict) and 'heart' in hands and pose.heart > 0 and pose.heart_present:
            hx, hy = hands['heart']
            col_h = '#ffc86b' if pose.heart_star < 0.5 else '#fff0b0'
            soft_glow(c, hx, hy, 70 * ds + 6, col_h, 0.35 * min(1.5, pose.heart))

    if deng_mode != 'on_rail' or not split:
        _deng()
    if split:
        _lighthouse(c, T, x_b, y_b, s, lamp, 'front', phase, glow=front_glow)
    if deng_mode == 'on_rail' and split:
        _deng()
    elif pose_fn is not None and an.get('legacy'):
        rs = 0.16 * s
        try:
            sea.gallery_railing(c, gx0, gx1, floor_y - 140 * rs, rs)
        except Exception:
            pass
    draw_beam(c, T, lx, lamp_y, s, 'front', intensity=beam, phase=phase)
    out['flare'] = lamp_flare(c, T, lx, lamp_y, s, intensity=beam * lamp * (0.5 if split else 1.0), phase=phase)
    out['floor_y'] = floor_y
    out['anchors'] = an
    return out
