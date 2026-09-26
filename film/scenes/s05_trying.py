"""s05 · 想办法 Trying — 38 s. Owner: SCENE-S05.

Playful, funny, warm; then worry creeps in. Crossfades in from s04 (0.8 s).

Shots (local time, all from cues):
  SH1  0 → stack          THE THROW. Island rocks by the lighthouse base. Deng crouches with
                           Guang and dips on each count of D07 (syllable onsets read from the
                           voice envelope), flings her up at `throw1`; the camera tilts to follow
                           (ease-out rise, hang + spin, ease-in fall); she lands ON HIS HEAD at
                           `catch` (squash both, sparkle burst, head-bonk wobble). G04 she laughs
                           on his head; D08 he rolls up imaginary sleeves (IK hand rubs the other
                           forearm) and flexes.
  SH2  stack → climb      MONTAGE, 3 hard cuts: runs in with a crate overhead; heaves a crate up
                           from a barrel step; climbs the 7-high wobbly tower, Guang on his head.
  SH3  climb → splash+.55 TOP OF THE TOWER: tiptoes, arms up, Guang held high on his fingertips;
                           the camera pulls back to show the stars are still impossibly far; the
                           sway grows; `tower_fall`: uh-oh, topple (closed-form rigid rotation,
                           then per-crate ballistic arcs with spin); `splash` into shallow water.
  SH4  → sit_together     IN THE WATER: Deng pops up with seaweed on his head, Guang pops out
                           laughing and sparkling (G05); Deng sheepish, then laughs too.
  SH5a sit_together → dawn_warning-0.7   WIDE: on a rock at the water's edge, Guang on his
                           knee, reflections, beam sweeping. G06, D10; she glows brighter.
  SH5b → end              MEDIUM: dawn begins to pale the horizon; Guang flickers & dims (G07,
                           sleepy); Deng alarmed, gathers her to his heart; at `idea` the beam
                           sweeps down onto them and stops on his face; antenna flash: an idea!
"""
from __future__ import annotations

import inspect
import math
from dataclasses import replace

import skia

from engine.core import (H, TAU, W, Camera, clamp, ease_in, ease_in_out, ease_out, ease_out_back, fill, keyframes,
                         lerp, noise1, saved, smoothstep, soft_glow)
from lib import fx, robot, sea, sky, star
from lib.robot import RobotPose, auto_blink
from lib.star import StarPose

from . import _s05_util as U

R_STAR = getattr(star, 'RADIUS', 55)
BEAM_RATE = 0.9  # rad/s — shared convention (global time)


# ----------------------------------------------------------------------------
# small compat helpers
# ----------------------------------------------------------------------------
_SIG = {}


def _accepts(fn, kw):
    key = (id(fn), kw)
    if key not in _SIG:
        try:
            _SIG[key] = kw in inspect.signature(fn).parameters
        except Exception:
            _SIG[key] = False
    return _SIG[key]


def _kw(fn, **kw):
    return {k: v for k, v in kw.items() if _accepts(fn, k)}


def island(c, x, y, s, dawn=0.0):
    sea.island(c, x, y, s, **_kw(sea.island, dawn=dawn))


def lighthouse(c, x, y, s, T, dawn=0.0, phi=None, lamp=1.0):
    return sea.lighthouse(c, x, y, s, lamp=lamp, dawn=dawn, **_kw(sea.lighthouse, t=T, beam_angle=phi))


# Top edge of sea.island() in its local units (current library art). If the island art
# changes, only this profile (or a library helper) is needed.
_ISLAND_TOP = [(-460, 120), (-330, 20), (-160, -6), (0, -12), (170, -4), (340, 30), (470, 130)]


def island_top(x, ix, iy, s):
    fn = getattr(sea, 'island_top', None)
    if fn is not None:
        try:
            return fn(x, ix, iy, s)
        except Exception:
            pass
    lx = (x - ix) / s
    pts = _ISLAND_TOP
    if lx <= pts[0][0]:
        return iy + pts[0][1] * s
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if lx <= x1:
            u = (lx - x0) / (x1 - x0)
            return iy + (y0 + (y1 - y0) * u) * s
    return iy + pts[-1][1] * s


def _cues(f):
    C = dict(throw1=f.cue('throw1'), catch=f.cue('catch'), stack=f.cue('stack'), climb=f.cue('climb'),
             fall=f.cue('tower_fall'), splash=f.cue('splash'), sit=f.cue('sit_together'),
             warn=f.cue('dawn_warning'), idea=f.cue('idea'))
    for lid in ('D07', 'G04', 'D08', 'G05', 'G06', 'D10', 'G07'):
        C[lid] = f.line(lid)
    return C


def star_pose(**kw):
    return U.make_pose(StarPose, **kw)


def robot_pose(**kw):
    return U.make_pose(RobotPose, **kw)


def deng_idle(T, seed=0):
    return robot.idle(T, seed) if hasattr(robot, 'idle') else dict(head_dy=0.0, lean=0.0, head_tilt=0.0)


def draw_deng(c, pose, T):
    return robot.draw_robot(c, pose, T)


def draw_guang(c, pose, T):
    return star.draw_star(c, pose, T)


_LIB_SEAWEED = 'seaweed' in getattr(RobotPose, '__dataclass_fields__', {})
_LIB_ANT_FLASH = 'antenna_flash' in getattr(RobotPose, '__dataclass_fields__', {})


def antenna_flash_local(c, res, amount):
    """Fallback flash of the antenna bulb when the robot library cannot do it."""
    if _LIB_ANT_FLASH or amount <= 0.01 or 'antenna' not in res:
        return
    ax, ay = res['antenna']
    soft_glow(c, ax, ay, 70 * amount + 10, '#ff7a5a', 0.9 * amount)
    soft_glow(c, ax, ay, 22 + 10 * amount, '#fff0d0', 0.8 * amount)


# ----------------------------------------------------------------------------
# shared backdrop
# ----------------------------------------------------------------------------
def star_camera(f, cam, y_par=1.0, x_par=0.06, z_par=0.05):
    return Camera(x=W / 2 + (cam.x - W / 2) * x_par, y=H / 2 + (cam.y - H / 2) * y_par,
                  zoom=1 + (cam.zoom - 1) * z_par, shake=cam.shake * 0.3, t=f.T)


def backdrop(c, f, cam, horizon, dawn=0.0, lights=(), y_par=1.0, milky=((700, 150), -0.32, 1.0),
             star_bright=1.0, hero=(), sun_x=1500, calm=1.0, ref_cam=None):
    """Sky gradient + sea in the world camera; Milky Way/stars in a (nearly) fixed star camera,
    so zooming the world never brings the stars any closer.
    milky / hero positions are SCREEN points as seen from `ref_cam` (default: `cam`)."""
    scam = star_camera(f, cam, y_par=y_par)
    rcam = star_camera(f, ref_cam or cam, y_par=y_par)

    def S(sx, sy):  # screen point at the reference camera -> star-camera world point
        return ((sx - W / 2) / rcam.zoom + rcam.x, (sy - H / 2) / rcam.zoom + rcam.y)
    with cam.apply(c):
        sky.sky(c, rect=(cam.x - 12000, cam.y - 12000, 24000, 24000), horizon_y=horizon, dawn=dawn)
        if dawn > 0:
            sky.dawn_glow(c, horizon_y=horizon, amount=dawn, sun_x=sun_x, **_kw(sky.dawn_glow, width=2.0))
    hs = cam.to_screen(0, horizon)[1]
    hy_star = (hs - H / 2) / scam.zoom + scam.y
    with scam.apply(c):
        (msx, msy), mang, malpha = milky
        sky.milky_way(c, f.T, center=S(msx, msy), angle=mang, alpha=malpha * (1 - dawn * 1.4),
                      **_kw(sky.milky_way, horizon_y=hy_star))
        sky.stars(c, f.T, rect=(scam.x - 1400, scam.y - 900, 2800, 1800), horizon_y=hy_star,
                  bright=star_bright * (1 - dawn * 1.2))
        for (hx, hy, hs_) in hero:
            px, py = S(hx, hy)
            if hasattr(sky, 'hero_star'):
                sky.hero_star(c, px, py, size=hs_, t=f.T, intensity=1 - dawn)
            else:
                U.sparkle_shape(c, px, py, 9 * hs_, '#fff6e0', 0.9 * (1 - dawn))
    with cam.apply(c):
        sea.ocean(c, f.T, horizon_y=horizon, lights=lights, dawn=dawn, calm=calm,
                  rect_x=(cam.x - 9000, cam.x + 9000), bottom=cam.y + 9000,
                  **_kw(sea.ocean, cx=cam.x))
    return scam


# ============================================================================
# World for the island shots: island + lighthouse at (0, 0) with s=10, which makes the
# lighthouse ~14x Deng (sea.lighthouse_anchors: deng_scale = 0.1*s) and its door Deng-sized.
# Deng scale 1. The "front yard" plateau left of the lighthouse faces the open sea.
# ============================================================================
S_ISL = 10.0
LAMP = (0.0, -445.0 * S_ISL)
HOR1 = -60
D1X, D1Y = -1450, 450      # Deng's throw spot (on the plateau)
TX, TY = -2000, 430        # crate tower base (left edge of the plateau)
CAM1 = (-1250, 150)


def _anchors():
    fn = getattr(sea, 'island_anchors', None)
    if fn is not None:
        try:
            return fn(0.0, 0.0, S_ISL)
        except Exception:
            pass
    return dict(seat=(2500.0, 1080.0), waterline_y=1400.0, plateau=(-2100.0, 2300.0, 200.0))


def ground1(x):
    """Walking surface for the plateau (flat) and the steps down to the jetty."""
    if x >= -2400:
        return None
    return lerp(470, 820, clamp((-2400 - x) / 600))


def lamp1():
    return LAMP


def _pile_items():
    return [dict(kind='crate', s=1.36, x=-1170, y=335, rot=-0.02, seed=11),
            dict(kind='barrel', s=1.2, x=-1010, y=345, rot=0.0, seed=12),
            dict(kind='crate', s=1.05, x=-1162, y=335 - 1.36 * 120, rot=0.05, seed=13)]


def draw_island_set(c, f, dawn=0.0, with_pile=True, pile_n=3):
    phi = f.T * BEAM_RATE
    sea.island(c, 0.0, 0.0, S_ISL, **_kw(sea.island, dawn=dawn, t=f.T, props=False))
    lx, ly = lighthouse(c, 0.0, 0.0, S_ISL, f.T, dawn=dawn, phi=phi)
    U.draw_beam(c, lx, ly, phi, length=9000, intensity=0.85, horizon_y=HOR1, t=f.T, width1=2400)
    if with_pile:
        for it in _pile_items()[:pile_n]:
            U.draw_item(c, it)
    return lx, ly


def _flight(C, t, p0, apex, p1):
    """Guang's throw trajectory: ease-out rise, hang, ease-in fall. Returns (x, y, phase)."""
    t0, t1 = C['throw1'], C['catch']
    ta = t0 + 0.88
    tb = t0 + 1.28
    if t <= ta:
        u = (t - t0) / (ta - t0)
        return lerp(p0[0], apex[0], ease_out(u)), lerp(p0[1], apex[1], ease_out(u)), 'up'
    if t <= tb:
        u = (t - ta) / (tb - ta)
        return apex[0] + 14 * math.sin(u * math.pi), apex[1] - 14 * math.sin(u * math.pi), 'hang'
    u = (t - tb) / (t1 - tb)
    return (lerp(apex[0], p1[0], ease_in_out(u) * 0.5 + ease_in(u) * 0.5), lerp(apex[1], p1[1], ease_in(u)),
            'down')


def _sleeve_targets(pose, T, tau, which):
    """IK target for the rubbing hand: slides along the other arm's forearm with a little roll."""
    a = U.measure_robot(pose, T)
    if which == 'r':        # near hand rubs the far forearm
        hx, hy = a['l']
        ex, ey = a.get('elbow_l', a['l'])
    else:
        hx, hy = a['r']
        ex, ey = a.get('elbow_r', a['r'])
    w = TAU * 2.6 * tau
    k = 0.35 + 0.3 * (0.5 + 0.5 * math.sin(w))
    px, py = lerp(hx, ex, k), lerp(hy, ey, k)
    dx, dy = ex - hx, ey - hy
    d = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / d, dx / d
    r = 7 * math.cos(w)
    return px + nx * r, py + ny * r - 4


def _deng_pose_sh1(f, C, t):
    T = f.T
    idl = deng_idle(T)
    t0, tc = C['throw1'], C['catch']
    d7s, d7e = C['D07']
    ons = U.syllable_onsets(f, 'D07', 4)
    x = D1X
    squash = 0.10
    lean = 0.05
    head_dy = idl['head_dy']
    head_tilt = idl['head_tilt']
    arm = (0.95, 0.55)
    eyes, look, smile = 'happy', (0.5, 0.2), 0.6
    walk, walk_phase, shake = 0.0, 0.0, 0.0
    hand_open = 0.6
    extra = {}
    # a little "ready?" nod before counting
    head_tilt += 0.1 * U.bump(t - 0.35, 0.2, 0.35)
    # the three counts — each a deeper dip
    for k, amp in zip(range(3), (0.12, 0.17, 0.23)):
        b = U.bump(t - ons[k] + 0.04, 0.09, 0.28)
        squash += amp * b
        lean += 0.10 * b
        head_dy += 6 * b
        arm = (arm[0] - 0.35 * b, arm[1])
    if d7s - 0.2 < t:
        eyes, look, smile = 'determined', (0.1, -0.6), 0.15
    # wind-up on "飞" and the fling
    wind0 = ons[3] - 0.05
    if t >= wind0:
        u = smoothstep(wind0, t0 - 0.12, t)
        squash = lerp(squash, 0.42, u)
        lean = lerp(lean, 0.26, u)
        arm = (lerp(arm[0], 0.2, u), lerp(arm[1], 0.35, u))
    if t >= t0 - 0.12:
        u = clamp((t - (t0 - 0.12)) / 0.2)
        e = ease_in(u)
        squash = lerp(0.42, -0.28, ease_out(u))
        lean = lerp(0.26, -0.12, e)
        arm = (lerp(0.2, 2.95, e), lerp(0.35, 0.05, e))
        eyes, look = 'determined', (0.0, -1.0)
    if t >= t0 + 0.08:
        tau = t - (t0 + 0.08)
        squash = -0.28 * U.damped(tau, 2.2, 5.0)
        lean = -0.12 * math.exp(-4 * tau)
        a0 = keyframes(t, [(t0 + 0.08, 2.95), (t0 + 0.5, 2.8), (t0 + 1.0, 2.25), (4.25, 2.3), (4.45, 2.75)])
        a1 = keyframes(t, [(t0 + 0.08, 0.05), (t0 + 1.0, 0.45), (4.25, 0.4), (4.45, 0.12)])
        arm = (a0, a1)
        eyes = 'surprised' if t < t0 + 0.35 else 'happy'
        look = (0.0, -1.0)
        smile = 0.8
        hand_open = 1.0
    # scramble under her
    if t >= 4.2:
        x = keyframes(t, [(4.2, D1X), (4.45, D1X - 26), (4.7, D1X + 30), (4.92, D1X + 12)], ease_in_out)
        walk = U.bump(t - 4.2, 0.1, 0.6)
        walk_phase = (t - 4.2) * 3.0
        eyes, look = 'surprised', (0.0, -1.0)
    # bonk!
    if t >= tc:
        tau = t - tc
        x = D1X + 12
        walk = 0.0
        squash = 0.34 * U.damped(tau, 2.4, 4.5)
        head_dy += 18 * math.exp(-7 * tau)
        head_tilt += 0.22 * U.damped_sin(tau, 2.0, 3.2)
        arm = (lerp(2.75, 0.2, ease_out(clamp(tau / 0.4))), lerp(0.12, 0.25, clamp(tau / 0.4)))
        eyes = 'surprised' if tau < 0.6 else 'happy'
        look = (0.0, -1.0)
        smile = 0.5 if tau < 0.6 else 0.8
        shake = 0.6 * math.exp(-5 * tau)
    # G04: chuckling, looking up at her
    g4s, g4e = C['G04']
    if t >= g4s:
        head_dy += -3 * abs(math.sin((t - g4s) * 7.0)) * (1 - smoothstep(g4e - 0.3, g4e, t))
        arm = (0.25 + 0.05 * math.sin(T * 1.5), 0.25)
    # thinking beat
    d8s, d8e = C['D08']
    if g4e < t < d8s + 0.1:
        eyes, look, smile = 'open', (0.5, -0.3), 0.4
        head_tilt += 0.08 * smoothstep(g4e, g4e + 0.3, t)
    base = dict(x=x, y=D1Y, scale=1.0, facing=1.0, lean=lean + idl['lean'], head_tilt=head_tilt,
                head_dy=head_dy, arm_l=arm, arm_r=arm, hand_l_open=hand_open, hand_r_open=hand_open,
                walk=walk, walk_phase=walk_phase, squash=squash, eyes=eyes, blink=auto_blink(T), look=look,
                mouth=f.mouth('deng'), smile=smile, shake=shake, scarf_wind=0.5)
    # D08: roll up sleeves, then flex
    if t >= d8s - 0.1:
        tau = t - (d8s - 0.1)
        roll1 = smoothstep(0, 0.25, tau) * (1 - smoothstep(0.85, 1.05, tau))
        roll2 = smoothstep(0.85, 1.05, tau) * (1 - smoothstep(1.55, 1.75, tau))
        flex = smoothstep(1.55, 1.8, tau)
        fwd = (1.25, 0.55)
        rest = (0.25, 0.25)
        fl = (1.3, 1.85 + 0.12 * math.sin(tau * 9))
        arm_l = tuple(lerp(a, b, roll1) for a, b in zip(rest, fwd))
        arm_r = tuple(lerp(a, b, roll2) for a, b in zip(rest, fwd))
        arm_l = tuple(lerp(a, b, flex) for a, b in zip(arm_l, fl))
        arm_r = tuple(lerp(a, b, flex) for a, b in zip(arm_r, fl))
        base.update(arm_l=arm_l, arm_r=arm_r, eyes='determined', smile=0.35 + 0.2 * flex,
                    hand_l_open=1 - flex, hand_r_open=1 - flex, squash=squash - 0.08 * flex,
                    head_tilt=head_tilt + 0.12 * (roll1 - roll2) - 0.06 * flex,
                    head_dy=head_dy + 6 * U.bump(tau - 1.6, 0.1, 0.25),
                    look=(0.6 * (roll1 + roll2) + 0.3 * flex, 0.45 * (roll1 + roll2) - 0.3 * flex))
        pose = robot_pose(**base)
        if roll1 > 0.02 and 'arm_r_override' in RobotPose.__dataclass_fields__:
            tgt = _sleeve_targets(pose, T, tau, 'r')
            nat = U.measure_robot(pose, T)['r']
            pose = replace(pose, arm_r_override=(lerp(nat[0], tgt[0], roll1), lerp(nat[1], tgt[1], roll1)))
        if roll2 > 0.02 and 'arm_l_override' in RobotPose.__dataclass_fields__:
            tgt = _sleeve_targets(pose, T, tau, 'l')
            nat = U.measure_robot(pose, T)['l']
            pose = replace(pose, arm_l_override=(lerp(nat[0], tgt[0], roll2), lerp(nat[1], tgt[1], roll2)),
                           arm_l_front=True)
        return pose
    return robot_pose(**base)


def _guang_on_head(res, pose, gs, lift=0.0):
    (tx, ty), (ux, uy) = U.head_frame(res, pose)
    d = R_STAR * gs * 0.72 + lift
    return tx + ux * d, ty + uy * d, math.atan2(ux, -uy)


def shot_throw(f, C):
    c, t, T = f.canvas, f.t, f.T
    t0, tc = C['throw1'], C['catch']
    gs = 1.0
    # --- where is Guang? (needed for the camera) ---
    p0 = U.hands_mid(U.measure_robot(_deng_pose_sh1(f, C, t0), T - t + t0))
    p0 = (p0[0], p0[1] - R_STAR * gs * 0.55)
    pc = _deng_pose_sh1(f, C, tc - 1e-3)
    rc = U.measure_robot(pc, T - t + tc)
    p1x, p1y, _ = _guang_on_head(rc, pc, gs)
    apex = (p0[0] + 70, p0[1] - 930)
    pose = _deng_pose_sh1(f, C, t)
    if t < t0:
        gx, gy, phase = None, p0[1], 'held'
    elif t < tc:
        gx, gy, phase = _flight(C, t, p0, apex, (p1x, p1y))
    else:
        gx, gy, phase = None, p1y, 'head'

    # --- camera: follows her up with a soft limit ---
    ref_y = gy if phase in ('up', 'hang', 'down') else D1Y + 800
    cy = U.smin(CAM1[1], ref_y + 420, 70)
    z = 1 + 0.07 * smoothstep(C['D07'][0], C['D07'][1] + 0.1, t) * (1 - smoothstep(t0 - 0.1, t0 + 0.4, t))
    cx = CAM1[0]
    if t > tc:
        u = ease_in_out(clamp((t - (tc + 0.5)) / 3.6))
        z = lerp(1.0, 1.28, u)
        cx = lerp(CAM1[0], CAM1[0] - 70, u)
        cy = lerp(cy, CAM1[1] + 50, u)
    shake = 7 * math.exp(-8 * (t - tc)) if t >= tc else 0.0
    cam = Camera(x=cx, y=cy, zoom=z, shake=shake, t=T)

    lx, ly = lamp1()
    backdrop(c, f, cam, HOR1, lights=[(lx, ly, '#ffd36b', 0.6)], milky=((620, 120), -0.3, 1.0),
             hero=[(1250, 70, 1.1)], ref_cam=Camera(x=CAM1[0], y=CAM1[1]))
    with cam.apply(c):
        draw_island_set(c, f)
        res = draw_deng(c, pose, T)
        if phase == 'held':
            hx, hy = U.hands_mid(res)
            ons = U.syllable_onsets(f, 'D07', 4)
            cheer = sum(U.bump(t - o + 0.02, 0.08, 0.25) for o in ons[:3])
            sp = star_pose(x=hx, y=hy - R_STAR * gs * 0.55, scale=gs, eyes='happy', smile=0.9,
                           arm_l=0.5 + 0.6 * cheer, arm_r=0.5 + 0.6 * cheer, blink=auto_blink(T, 3, 4.3),
                           mouth=f.mouth('guang'), rot=0.05 * math.sin(T * 2), squash=0.12 * cheer, halo=0.8)
            draw_guang(c, sp, T)
        elif phase in ('up', 'hang', 'down'):
            dt = 1 / 48
            ax_, ay_, _ = _flight(C, t - dt, p0, apex, (p1x, p1y))
            bx_, by_, _ = _flight(C, t + dt, p0, apex, (p1x, p1y))
            vx, vy = (bx_ - ax_) / (2 * dt), (by_ - ay_) / (2 * dt)
            spd = math.hypot(vx, vy)
            if phase == 'up':
                rot = 0.35 * smoothstep(t0, t0 + 0.88, t)
                eyes, arms, mouth = 'surprised', 1.0, 0.35
            elif phase == 'hang':
                u = (t - (t0 + 0.88)) / 0.4
                rot = 0.35 + TAU * ease_in_out(u)
                eyes, arms, mouth = 'happy', 1.3, 0.6
            else:
                u = (t - (t0 + 1.28)) / (tc - t0 - 1.28)
                rot = 0.35 + TAU - 0.35 * u + 0.25 * math.sin(u * 9)
                eyes, arms, mouth = 'surprised', 0.8 + 0.5 * math.sin(t * 30), 0.5
            sp = star_pose(x=gx, y=gy, scale=gs, glow=1.1, eyes=eyes, smile=0.8, rot=rot, arm_l=arms, arm_r=arms,
                           trail=clamp(spd / 1200), vx=vx, vy=vy, squash=-clamp(spd / 6000) * 0.5, mouth=mouth)
            draw_guang(c, sp, T)
        else:
            tau = t - tc
            sq = 0.55 * U.damped(tau, 2.6, 6.0)
            g4s, g4e = C['G04']
            laugh = smoothstep(g4s - 0.1, g4s + 0.1, t) * (1 - smoothstep(g4e - 0.1, g4e + 0.3, t))
            hop = 9 * abs(math.sin((t - g4s) * math.pi * 3.8)) * laugh
            x, y, rot = _guang_on_head(res, pose, gs, lift=hop)
            d8s, d8e = C['D08']
            cheer = smoothstep(d8s + 1.5, d8s + 1.8, t)
            eyes = 'closed' if tau < 0.35 else 'happy'
            look = (0.0, 0.0)
            if g4e + 0.3 < t < d8s + 1.5:
                eyes, look = 'open', (-0.1, 0.8)
            arms = 0.2 + laugh * (0.9 + 0.45 * math.sin(t * 11)) + cheer * (1.3 + 0.2 * math.sin(t * 9))
            arms_r = 0.2 + laugh * (0.9 + 0.45 * math.sin(t * 11 + 1.3)) + cheer * (1.3 + 0.2 * math.sin(t * 9 + 1))
            sp = star_pose(x=x, y=y, scale=gs, glow=1.0 + 0.15 * laugh, eyes=eyes if cheer < 0.5 else 'happy',
                           smile=0.9, rot=rot + 0.12 * math.sin(t * 7) * laugh, arm_l=arms, arm_r=arms_r,
                           squash=sq, squash_pivot=1.0, mouth=f.mouth('guang'), blink=auto_blink(T, 3, 4.3),
                           look=look)
            draw_guang(c, sp, T)
            U.sparkle_burst(c, p1x, p1y + R_STAR * 0.5, t - tc, count=12, seed=51, size=0.55, life=0.8)
            if laugh > 0:
                U.twinkles(c, T, x, y, radius=95, count=6, seed=52, intensity=laugh * 0.8)


# ============================================================================
# SH2 — stacking montage
# ============================================================================
def shot_stack(f, C):
    c, t, T = f.canvas, f.t, f.T
    b0, b3 = C['stack'], C['climb']
    L = (b3 - b0) / 3
    beat = min(2, max(0, int((t - b0) // L)))
    tb = t - (b0 + beat * L)  # time within beat
    u = clamp(tb / L)
    if beat == 0:
        cam = Camera(x=lerp(-1480, -1560, u), y=120, zoom=0.95, t=T)
    elif beat == 1:
        cam = Camera(x=-1830, y=lerp(-40, -70, u), zoom=0.9, t=T)
    else:
        cam = Camera(x=-1930, y=lerp(180, -560, ease_in_out(u)), zoom=0.88, t=T)
    ref = Camera(x=cam.x, y=[120, -40, 180][beat], zoom=cam.zoom)
    lx, ly = lamp1()
    backdrop(c, f, cam, HOR1, lights=[(lx, ly, '#ffd36b', 0.6)], milky=((700, 120), -0.3, 1.0), y_par=1.0,
             hero=[(1250, 80, 1.0)], ref_cam=ref)
    with cam.apply(c):
        draw_island_set(c, f, with_pile=False)
        pile = _pile_items()
        if beat == 0:
            items, top = U.tower_chain(2, TX, TY, bend=0.01 * math.sin(T * 3))
            for it in items:
                U.draw_item(c, it)
            for it in pile[:2]:
                U.draw_item(c, it)
            # Guang cheering on the pile
            p0 = pile[0]
            gx, gy = p0['x'], p0['y'] - p0['s'] * U.ITEM_H - R_STAR * 0.72
            ch = math.sin(T * 12)
            draw_guang(c, star_pose(x=gx, y=gy - 10 * abs(ch), scale=0.95, eyes='happy', smile=1.0,
                                    arm_l=1.2 + 0.4 * ch, arm_r=1.2 - 0.4 * ch, mouth=0.4, rot=0.1 * ch), T)
            # Deng hurrying left with the little crate overhead (fast motion)
            x = lerp(-1250, -1760, ease_in_out(u) * 0.3 + u * 0.7)
            ph = (t - b0) * 2.2
            pose = robot_pose(x=x, y=D1Y, scale=1.0, facing=-1.0, walk=1.0, walk_phase=ph, run=0.6, lean=0.1,
                              arm_l=(3.0, 0.15), arm_r=(3.0, 0.15), hand_l_open=0.9, hand_r_open=0.9,
                              eyes='happy', look=(0.5, -0.2), smile=0.8,
                              head_dy=-4 * abs(math.sin(ph * math.pi)), blink=0.0, scarf_wind=1.0, scarf_dir=1)
            res = draw_deng(c, pose, T)
            hx, hy = U.hands_mid(res)
            top_y = min(hy, res.get('head_top', res['head'])[1] - 4)
            U.draw_item(c, dict(pile[2], x=hx, y=top_y + 6, rot=0.04 * math.sin(ph * TAU)))
            U.speed_lines(c, x + 120, D1Y - 200, -1, T, n=6, length=170, spread=260, a=0.45)
            U.dust_puffs(c, x + 20, D1Y - 5, T, a=0.6, direction=-1)
        elif beat == 1:
            items, top = U.tower_chain(3, TX, TY, bend=0.015 * math.sin(T * 3))
            for it in items:
                U.draw_item(c, it)
            U.draw_item(c, pile[1])
            # barrel step to the right of the tower
            sx = TX + 240
            step = dict(kind='barrel', s=1.15, x=sx, y=TY + 20, rot=0.0, seed=21)
            U.draw_item(c, step)
            stand_y = step['y'] - step['s'] * U.ITEM_H
            lift = ease_out(clamp(u / 0.6))
            pose = robot_pose(x=sx + 5, y=stand_y, scale=1.0, facing=-1.0, lean=-0.05 + 0.05 * lift,
                              arm_l=(lerp(2.3, 3.05, lift), 0.1), arm_r=(lerp(2.3, 3.05, lift), 0.1),
                              squash=lerp(0.15, -0.1, lift), eyes='determined', eyes2='happy', eyes_mix=0.4,
                              look=(0.3, -0.8), smile=0.2, shake=0.8 * (1 - lift), blink=0.0, mouth=0.25,
                              shadow=0.0)
            res = draw_deng(c, pose, T)
            hx, hy = U.hands_mid(res)
            hy = min(hy, res.get('head_top', res['head'])[1] - 4)
            # the crate travels from his hands onto the tower top
            k = smoothstep(0.55, 0.92, u)
            nx = TOWER_NEXT = U.TOWER[3]
            cx_ = lerp(hx, top[0], k)
            cy_ = lerp(hy + 6, top[1] + 2, k * k)
            U.draw_item(c, dict(kind=nx[0], s=nx[1], x=cx_, y=cy_, rot=lerp(0.1, top[2], k), i=3))
            # Guang cheering from below
            gx = TX + 520
            ch = math.sin(T * 11)
            gy = TY + 40 - R_STAR * 0.8
            draw_guang(c, star_pose(x=gx, y=gy - 22 * abs(ch), scale=0.95, eyes='happy', smile=1.0,
                                    arm_l=1.3 + 0.3 * ch, arm_r=1.3 - 0.3 * ch, mouth=0.5, rot=0.15 * ch), T)
            U.twinkles(c, T, gx, gy - 20, radius=70, count=4, seed=21, intensity=0.6)
        else:
            sway = 0.02 * math.sin(T * 2.6)
            items, top = U.tower_chain(7, TX, TY, bend=sway)
            for it in items:
                U.draw_item(c, it)
            # Deng climbing the right side of the tower, Guang riding on his head
            k = ease_in_out(u) * 0.8 + u * 0.2
            lvl = lerp(1.6, 5.4, k)
            i0 = min(len(items) - 1, int(lvl))
            it = items[i0]
            fr = lvl - i0
            ex = it['x'] + it['h'] * fr * math.sin(it['rot']) + 62 * it['s'] * math.cos(it['rot'])
            ey = it['y'] - it['h'] * fr * math.cos(it['rot'])
            ph = (t - (b0 + 2 * L)) * 2.2
            swing = math.sin(ph * TAU)
            pose = robot_pose(x=ex + 40, y=ey + 60, scale=1.0, facing=-1.0, lean=-0.08,
                              arm_l=(2.75 + 0.25 * swing, 0.2), arm_r=(2.75 - 0.25 * swing, 0.2), walk=1.0,
                              walk_phase=ph, eyes='determined', eyes2='happy', eyes_mix=0.4, look=(0.2, -0.9),
                              smile=0.3, blink=0.0, head_dy=-3 * abs(swing), shadow=0.0)
            res = draw_deng(c, pose, T)
            gx, gy, rot = _guang_on_head(res, pose, 0.95)
            draw_guang(c, star_pose(x=gx, y=gy, scale=0.95, rot=rot + 0.1 * swing, eyes='happy', smile=0.9,
                                    arm_l=1.4 + 0.3 * swing, arm_r=1.4 - 0.3 * swing, mouth=0.3), T)
    # quick exposure pop on each cut (montage punctuation)
    f.grade['exposure'] = 1.0 + 0.12 * (1 - smoothstep(0, 0.12, tb))


# ============================================================================
# SH3 — top of the tower, reveal, topple, splash
# ============================================================================
G_FALL = 11000.0
SPX, SPY = -3700.0, 1650.0       # where Deng hits the water (in front of the jetty)
GSPX, GSPY = -3480.0, 1680.0     # where Guang hits the water
WATER_FALL_Y = 1620.0
SHORE_X = -3050.0


def _sway(C, t):
    c0 = C['climb']
    tau = max(0.0, min(t, C['fall']) - c0)
    span = max(0.5, C['fall'] - c0)
    A = 0.018 + 0.11 * (tau / span) ** 1.7
    ph = TAU * (0.9 * tau + 0.075 * tau * tau + 0.1)
    return A * math.sin(ph)


def _topple(C, t):
    """(rigid rotation Phi about the pivot, extra bend) after tower_fall."""
    tau = t - C['fall']
    if tau <= 0:
        return 0.0, 0.0
    phi = -(0.22 * tau + 0.5 * 7.5 * tau * tau)
    bend = -0.16 * smoothstep(0, 0.35, tau)
    return phi, bend


def _tower_state(C, t):
    """Items (world bottom-centre + rot) and the top frame, with sway & rigid topple applied."""
    tg = TY
    phi, bend2 = _topple(C, t)
    bend = _sway(C, t) if t < C['fall'] else _sway(C, C['fall']) + bend2
    items, top = U.tower_chain(7, TX, tg, bend=bend)
    piv = (items[1]['x'], items[1]['y'])
    if phi != 0:
        out = []
        for it in items:
            if it['i'] == 0:
                out.append(dict(it, rot=it['rot'] + phi * 0.05))
                continue
            x, y = U.rotate_about(piv[0], piv[1], it['x'], it['y'], phi)
            out.append(dict(it, x=x, y=y, rot=it['rot'] + phi))
        items = out
        tx, ty = U.rotate_about(piv[0], piv[1], top[0], top[1], phi)
        top = (tx, ty, top[2] + phi)
    return items, top


REL_CHAR = 0.22


def _item_release(i):
    return 0.18 + 0.045 * (6 - i)


def _falling_items(C, t):
    """Per-item positions during/after the topple, closed form. Returns (items, splashes)."""
    items_now, _ = _tower_state(C, t)
    tau = t - C['fall']
    if tau <= 0:
        return items_now, []
    out, splashes = [], []
    dt = 1 / 60
    for it in items_now:
        i = it['i']
        if i == 0:
            out.append(it)
            continue
        tr = _item_release(i)
        if tau <= tr:
            out.append(it)
            continue
        ta, _ = _tower_state(C, C['fall'] + tr - dt)
        tb_, _ = _tower_state(C, C['fall'] + tr + dt)
        a, b = ta[i], tb_[i]
        ca, cb = U.item_center(a), U.item_center(b)
        cx0, cy0 = (ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2
        vx, vy = (cb[0] - ca[0]) / (2 * dt) - 500 * (i / 6.0), (cb[1] - ca[1]) / (2 * dt)
        w = (b['rot'] - a['rot']) / (2 * dt) * 0.8 + (2.5 if i % 2 else -3.0)
        r0 = (a['rot'] + b['rot']) / 2
        s = tau - tr
        wl = WATER_FALL_Y + (i % 3) * 30
        yl = wl - it['h'] * 0.18
        disc = vy * vy + 2 * G_FALL * max(0.0, yl - cy0)
        tl = (-vy + math.sqrt(disc)) / G_FALL
        xl = cx0 + vx * tl
        over_land = xl > SHORE_X
        if over_land:
            yl = (ground1(xl) or TY + 20) - it['h'] * 0.5
            disc = vy * vy + 2 * G_FALL * max(0.0, yl - cy0)
            tl = (-vy + math.sqrt(disc)) / G_FALL
            xl = cx0 + vx * tl
        if s < tl:
            cx, cy = cx0 + vx * s, cy0 + vy * s + 0.5 * G_FALL * s * s
            rot = r0 + w * s
            state = 'air'
        else:
            s2 = s - tl
            rl = r0 + w * tl
            tgt = round(rl / (math.pi / 2)) * (math.pi / 2)
            rot = tgt + (rl - tgt) * math.exp(-6 * s2)
            if over_land:
                cx = xl + vx * 0.08 * (1 - math.exp(-5 * s2))
                cy = yl
            else:
                cx = xl + vx * 0.05 * (1 - math.exp(-3 * s2))
                cy = yl + 10 * math.sin(s2 * 3 + i) * math.exp(-0.5 * s2) + 18 * (1 - math.exp(-4 * s2))
                splashes.append((xl, wl, s2, i))
            state = 'land' if over_land else 'water'
        bx = cx - it['h'] / 2 * math.sin(rot)
        by = cy + it['h'] / 2 * math.cos(rot)
        out.append(dict(it, x=bx, y=by, rot=rot, state=state, wl=wl))
    return out, splashes


def _char_fall(t, p0, p1, tr_abs, t1_abs):
    """Ballistic from p0 at tr_abs to p1 at t1_abs (solved in closed form)."""
    Tt = max(0.05, t1_abs - tr_abs)
    vx = (p1[0] - p0[0]) / Tt
    vy = (p1[1] - p0[1] - 0.5 * G_FALL * Tt * Tt) / Tt
    s = t - tr_abs
    return p0[0] + vx * s, p0[1] + vy * s + 0.5 * G_FALL * s * s


def _deng_top_pose(f, C, t, top):
    T = f.T
    tx, ty, ta = top
    c0, fall = C['climb'], C['fall']
    grow = smoothstep(c0, fall, t)
    eyes, look, smile = 'determined', (0.0, -1.0), 0.0
    if t > c0 + 1.1:
        eyes, smile = 'wonder' if _robot_has_eye('wonder') else 'open', 0.2
    if t > fall - 0.55:
        eyes, look, smile = 'worried', (0.2, 0.8), -0.3
    if t > fall:
        eyes, look, smile = 'surprised', (0.0, 0.2), -0.3
    reach = 0.06 * math.sin(T * 5.0)
    return robot_pose(x=tx, y=ty, scale=1.0, facing=1.0, lean=ta * 0.35, squash=-0.14 + 0.03 * math.sin(T * 7),
                      arm_l=(3.1 + reach, 0.0), arm_r=(3.1 - reach, 0.0), hand_l_open=1.0, hand_r_open=1.0,
                      eyes=eyes, look=look, smile=smile, shake=0.3 + 1.2 * grow ** 2, blink=auto_blink(T),
                      mouth=0.0, head_tilt=-0.1, shadow=0.0)


def _robot_has_eye(name):
    fn = getattr(robot, '_expr', None)
    if fn is None:
        return False
    try:
        fn(name)
        return True
    except Exception:
        return False


def _held_high(res, gs):
    hx, hy = U.hands_mid(res)
    top = res.get('head_top', res['head'])[1]
    return hx, min(hy, top) - R_STAR * gs * 0.82


def shot_climb(f, C):
    c, t, T = f.canvas, f.t, f.T
    c0, fall, spl = C['climb'], C['fall'], C['splash']
    items, splashes = _falling_items(C, t)
    _, top_rigid = _tower_state(C, t)
    top_static_y = U.tower_chain(7, TX, TY)[1][1]
    cam_keys = [
        (c0, (TX + 20, top_static_y - 230, 1.45)),
        (c0 + 0.9, (TX + 20, top_static_y - 250, 1.4)),
        (fall - 0.3, (-2150, 120, 0.3)),
        (fall + 0.1, (-2200, 150, 0.3)),
        (spl, (-2500, 400, 0.33)),
        (spl + 0.6, (-2550, 430, 0.34)),
    ]
    cx, cy, z = keyframes(t, cam_keys, ease_in_out)
    shake = 10 * math.exp(-6 * (t - spl)) if t >= spl else 0.0
    cam = Camera(x=cx, y=cy, zoom=z, shake=shake, t=T)
    lx, ly = lamp1()
    backdrop(c, f, cam, HOR1, lights=[(lx, ly, '#ffd36b', 0.6)], y_par=0.05, milky=((760, 170), -0.28, 1.1),
             hero=[(930, 110, 1.4)], ref_cam=Camera(x=-2150, y=120, zoom=0.3))
    with cam.apply(c):
        draw_island_set(c, f, with_pile=True, pile_n=2)
        for it in items:
            if it.get('state') == 'water':
                U.in_water(c, lambda cc, it=it: U.draw_item(cc, it), it['wl'], sub_alpha=0.2)
            else:
                U.draw_item(c, it)
        gs = 0.9
        tr = fall + REL_CHAR
        if t < tr:
            pose = _deng_top_pose(f, C, t, top_rigid)
            res = draw_deng(c, pose, T)
            hx, hy = _held_high(res, gs)
            reachg = math.sin(T * 6)
            eyes = 'open' if t < fall - 0.55 else ('surprised' if t > fall else 'open')
            look = (0.0, -1.0) if t < fall - 0.55 else (0.2, 0.6)
            draw_guang(c, star_pose(x=hx, y=hy, scale=gs, arm_l=1.45 + 0.2 * reachg, arm_r=1.45 - 0.2 * reachg,
                                    eyes=eyes, look=look, smile=0.6 if t < fall - 0.55 else -0.1,
                                    rot=top_rigid[2] * 0.5, blink=auto_blink(T, 3, 4.3), squash=-0.08), T)
            if t > fall - 0.1:
                # "uh-oh" sweat drop
                a = smoothstep(fall - 0.1, fall + 0.05, t)
                ex, ey = res['head']
                c.drawCircle(ex + 70, ey - 30, 7, fill('#cfe6ff', 0.8 * a))
        elif t < spl + 0.02:
            # both fly off and tumble towards the water
            pose0 = _deng_top_pose(f, C, tr, _tower_state(C, tr)[1])
            r0 = U.measure_robot(pose0, T)
            dcx0 = (pose0.x + r0['head'][0]) / 2
            dcy0 = (pose0.y + r0['head'][1]) / 2
            dx, dy = _char_fall(t, (dcx0, dcy0), (SPX, SPY - 60), tr, spl)
            rot = pose0.lean - 5.0 * (t - tr)
            flail = math.sin(t * 38)
            pose = robot_pose(x=0.0, y=0.0, scale=1.0, facing=1.0, arm_l=(2.6 + 0.5 * flail, 0.4),
                              arm_r=(2.2 - 0.5 * flail, 0.6), eyes='surprised', look=(0, 0), smile=-0.4,
                              mouth=0.8, walk=1.0, walk_phase=t * 4, shadow=0.0)
            rr = U.measure_robot(pose, T)
            ocx, ocy = rr['head'][0] / 2, rr['head'][1] / 2

            def _dr(cc, pose=pose):
                with saved(cc):
                    cc.translate(dx, dy)
                    cc.rotate(math.degrees(rot))
                    cc.translate(-ocx, -ocy)
                    draw_deng(cc, pose, T)
            U.in_water(c, _dr, SPY, sub_alpha=0.15)
            hx0, hy0 = _held_high(r0, gs)
            gx, gy = _char_fall(t, (hx0, hy0), (GSPX, GSPY - 30), tr, spl)

            def _dg(cc):
                draw_guang(cc, star_pose(x=gx, y=gy, scale=gs, rot=9 * (t - tr), eyes='surprised', smile=-0.2,
                                         mouth=0.7, arm_l=1.6, arm_r=1.6, trail=0.8, vx=-1500, vy=1500), T)
            U.in_water(c, _dg, GSPY, sub_alpha=0.2)
        ts = t - spl
        if ts >= 0:
            _splash(c, SPX, SPY, ts, 2.4, seed=61)
            _splash(c, GSPX, GSPY, ts - 0.03, 1.3, seed=62)
            fx.ripples(c, SPX, SPY, ts, intensity=0.8, persp=0.2, count=4, speed=380)
        for (x, y, s2, i) in splashes:
            _splash(c, x, y, s2, 0.9, seed=70 + i)
            fx.ripples(c, x, y, s2, intensity=0.5, persp=0.2, count=2, speed=160)


def _splash(c, x, y, ts, scale, seed=2, mist=0.35):
    """Ordinary (non-magic) water splash."""
    if ts < 0 or ts > 3.2:
        return
    try:
        fx.splash(c, x, y, ts, scale=scale, color='#dff2ff', glow_color=None, seed=seed, mist=mist)
    except TypeError:
        fx.splash(c, x, y, ts, scale=scale, glow_color='#5a78a8')


# ============================================================================
# SH4 — in the water
# ============================================================================
HOR4 = 560
WL4 = 800
D4X, D4S = 790, 1.05
G4X = 1060


def shot_water(f, C):
    c, t, T = f.canvas, f.t, f.T
    spl = C['splash']
    tp = spl + 0.72     # Deng pops up
    tgp = spl + 1.0     # Guang pops out
    g5s, g5e = C['G05']
    cam = Camera(x=lerp(930, 950, f.between(spl, C['sit'])), y=560,
                 zoom=lerp(1.0, 1.06, ease_in_out(f.between(spl + 0.5, C['sit']))), t=T)
    phi = T * BEAM_RATE
    ILX, ILY, ILS = 1580, 650, 0.95
    LX, LY, LS = 1590, 642, 1.2
    lamp = (LX, LY - 445 * LS)
    gu = ease_out_back(clamp((t - tgp) / 0.5), 1.5)
    gy = lerp(WL4 + 50, 648, gu)
    lights = [(lamp[0], lamp[1], '#ffd36b', 0.7)]
    if t > tgp:
        lights.append((G4X, gy, '#ffd86b', 0.5, WL4))
    backdrop(c, f, cam, HOR4, lights=lights, milky=((980, 150), -0.3, 1.0), hero=[(640, 110, 0.9)])
    with cam.apply(c):
        island(c, ILX, ILY, ILS)
        lx, ly = lighthouse(c, LX, LY, LS, T, phi=phi)
        U.draw_beam(c, lx, ly, phi, length=3000, intensity=0.8, horizon_y=HOR4, t=T)
        # floating debris from the tower
        floats = [('barrel', 360, WL4 - 70, 0.55, math.pi / 2 + 0.05), ('crate', 1760, WL4 - 40, 0.55, -0.25),
                  ('crate', 1360, WL4 + 30, 0.85, 0.32)]
        for k, (kind, x, wl, s, rot) in enumerate(floats):
            bob = 5 * math.sin(T * 1.7 + k * 2)
            rr = rot + 0.05 * math.sin(T * 1.3 + k)
            h = U.ITEM_H * s
            it = dict(kind=kind, s=s, x=x + 6 * math.sin(T * 0.5 + k), y=wl + h * 0.45 + bob, rot=rr)
            U.in_water(c, lambda cc, it=it: U.draw_item(cc, it), wl + bob * 0.3, sub_alpha=0.22)
            U.waterline_ring(c, x, wl + bob * 0.3, h * 1.1, T, a=0.45, seed=k)
        # --- Deng ---
        up = ease_out_back(clamp((t - tp) / 0.42), 1.6)
        feet = lerp(WL4 + 560, WL4 + 150 * D4S, up)
        idl = deng_idle(T)
        eyes, look, smile = 'closed', (0, 0), 0.0
        arm_l, arm_r = (0.5, 0.4), (0.5, 0.4)
        head_dy, head_tilt, squash, mouth = idl['head_dy'], idl['head_tilt'], 0.0, f.mouth('deng')
        blink = auto_blink(T, 1)
        ant = 0.0
        lean = idl['lean']
        tau = t - tp
        if tau > 0.45:
            eyes, look = 'open', (0.6, -0.1)
        sheep0, sheep1 = g5s + 0.1, g5e - 1.0
        scratch = 0.0
        if sheep0 < t < sheep1:
            eyes, look, smile = 'sad', (-0.7, 0.35), 0.15
            head_tilt += 0.14
            scratch = smoothstep(sheep0, sheep0 + 0.3, t) * (1 - smoothstep(sheep1 - 0.3, sheep1, t))
        laugh0 = sheep1 + 0.35
        if sheep1 <= t < laugh0:
            eyes, look, smile = 'open', (0.7, -0.1), 0.5
            blink = max(blink, U.bump(t - sheep1 - 0.05, 0.06, 0.08))
        if t >= laugh0:
            k = t - laugh0
            eyes, smile = 'happy', 1.0
            bb = abs(math.sin(k * math.pi * 4.6))
            mouth = 0.35 + 0.5 * bb
            head_dy += -8 * bb
            squash = 0.06 * math.sin(k * TAU * 4.6)
            arm_l, arm_r = (0.9, 1.3 + 0.2 * bb), (0.9, 1.3 + 0.2 * bb)
            head_tilt += 0.08 * math.sin(k * 3) - 0.06
            lean -= 0.05
            ant = 0.6 + 0.4 * math.sin(k * TAU * 3.2)
        pose = robot_pose(x=D4X, y=feet, scale=D4S, facing=1.0, eyes=eyes, look=look, smile=smile, arm_l=arm_l,
                          arm_r=arm_r, head_dy=head_dy, head_tilt=head_tilt, squash=squash, mouth=mouth,
                          blink=blink, scarf_wind=0.2, lean=lean, seaweed=1.0 if t > tp else 0.0,
                          antenna_flash=ant * 0.6, antenna_rate=3.0 if ant > 0 else 1.0, shadow=0.0,
                          rim=0.5, rim_color='#ffe0a0', rim_angle=0.0)
        if scratch > 0 and 'arm_r_override' in RobotPose.__dataclass_fields__:
            a = U.measure_robot(pose, T)
            (hx, hy), (ux, uy) = a['head'], U.head_frame(a, pose)[1]
            tx = hx - 62 * D4S + 6 * math.sin(t * 24)
            ty = hy - 20 * D4S + 5 * math.cos(t * 24)
            nat = a['r']
            pose = replace(pose, arm_r_override=(lerp(nat[0], tx, scratch), lerp(nat[1], ty, scratch)))
        res_box = {}

        def _dd(cc):
            r = draw_deng(cc, pose, T)
            res_box['r'] = r
            if t > tp and not _LIB_SEAWEED:
                top, upv = U.head_frame(r, pose)
                U.seaweed(cc, top, upv, D4S, T, drip=1 - smoothstep(tp + 1.5, tp + 3.5, t),
                          sway=math.sin(t * 9) * 0.6 * (t >= laugh0), side=-1.0)
        if t > tp - 0.3:
            U.in_water(c, _dd, WL4, sub_alpha=0.3)
            antenna_flash_local(c, res_box['r'], ant)
            U.waterline_ring(c, D4X, WL4, 170 * D4S, T, a=0.8 * clamp((t - tp) / 0.2))
            # water streaming off him
            if hasattr(fx, 'water_drips') and not getattr(fx, 'STUB', True):
                amt = 1 - smoothstep(tp + 1.0, tp + 3.0, t)
                if amt > 0.02:
                    hx, hy = res_box['r']['head']
                    fx.water_drips(c, T, hx, hy + 40 * D4S, spread=60 * D4S, rate=3.0 * amt, seed=17,
                                   fall=WL4 - hy - 40 * D4S)
            else:
                for k in range(6):
                    ph = ((t - tp) * 1.6 + k / 6.0) % 1.0
                    a = (1 - smoothstep(tp + 0.5, tp + 2.5, t)) * (1 - ph)
                    x = D4X + (k - 2.5) * 30 * D4S
                    y0 = WL4 - 150 * D4S - (k % 3) * 40
                    c.drawCircle(x, y0 + ph * ph * 160, 3.5, fill('#cfe6ff', 0.7 * a))
        if t < tp + 0.1:
            soft_glow(c, D4X, WL4 + 60, 120, '#ffb347', 0.25)  # heart glow under water
        U.bubbles(c, T, D4X, WL4, t - spl, count=14, spread=70, rise=40, seed=81, a=0.8, life=0.8)
        fx.ripples(c, D4X, WL4, t - tp, intensity=0.6, persp=0.16, count=3, speed=150)
        # --- Guang ---
        if t < tgp:
            soft_glow(c, G4X, WL4 + 30, 110, '#ffd86b', 0.35 + 0.15 * math.sin(T * 9))
            U.bubbles(c, T, G4X, WL4, t - spl - 0.2, count=8, spread=40, rise=30, seed=82, a=0.7, life=0.8)
        else:
            laugh = smoothstep(g5s - 0.1, g5s + 0.1, t)
            k = t - g5s
            hop = 10 * abs(math.sin(k * math.pi * 3.6)) * laugh
            rot = 0.22 * math.sin(k * 6.5) * laugh
            gx = G4X + 6 * math.sin(T * 1.4)
            gyy = gy - hop + 4 * math.sin(T * 2.2)
            sp = star_pose(x=gx, y=gyy, scale=1.0, glow=1.1 + 0.1 * laugh, eyes='happy', smile=1.0,
                           rot=rot, arm_l=0.9 + 0.5 * math.sin(k * 10) * laugh,
                           arm_r=1.1 + 0.4 * math.sin(k * 10 + 1.4), mouth=f.mouth('guang'),
                           wet=1 - smoothstep(tgp, tgp + 2.6, t), blush=0.9, sparkle=1.5)
            U.in_water(c, lambda cc: draw_guang(cc, sp, T), WL4, sub_alpha=0.3)
            U.sparkle_burst(c, G4X, WL4 - 20, t - tgp, count=14, seed=83, size=0.7)
            U.twinkles(c, T, gx, gyy, radius=100, count=8, seed=84, intensity=0.9 * laugh)
            fx.ripples(c, G4X, WL4, t - tgp, intensity=0.6, persp=0.16, count=3, speed=130)
        _splash(c, D4X, WL4, t - spl, 1.3, seed=91, mist=0.15)


# ============================================================================
# SH5 — together on the rock; dawn; the idea
# ============================================================================
HOR5 = 620
LI5 = (560, 706, 0.55)
LH5 = (560, 700, 0.9)
ROCK5 = (1180, 850, 0.45)
DS5 = 0.7
GS5 = 0.5


def _rock_top():
    return island_top(ROCK5[0] + 8, *ROCK5)


def _rock_wl():
    return ROCK5[1] + 122 * ROCK5[2]


def _li_wl():
    return LI5[1] + 122 * LI5[2]


def lamp5():
    return LH5[0], LH5[1] - 445 * LH5[2]


def _dawn(C, t):
    return 0.35 * smoothstep(C['warn'] - 1.3, C['idea'] + 1.6, t)


def _sit_state(f, C, t):
    """Acting for the sitting pair (both SH5 shots)."""
    T = f.T
    idl = deng_idle(T, 2)
    g6s, g6e = C['G06']
    d10s, d10e = C['D10']
    g7s, g7e = C['G07']
    warn, idea = C['warn'], C['idea']
    wonder = 'wonder' if _robot_has_eye('wonder') else 'open'
    # ---- Deng ----
    eyes, look, smile = wonder, (0.45, -0.85), 0.45
    head_tilt, lean, shake = idl['head_tilt'] - 0.08, 0.0, 0.0
    ant = 0.0
    look_at_her = (0.55, 0.75)
    blink_extra = U.bump(t - (g6s + 1.9), 0.12, 0.2) if g6s + 1.8 < t < d10s else 0.0
    if d10s <= t:
        a = smoothstep(d10s, d10s + 0.4, t)
        eyes = 'sad' if t < d10s + 1.2 else 'open'
        look = (lerp(0.45, 0.3, a), lerp(-0.85, 0.3, a))
        head_tilt += 0.1 * U.bump(t - d10s - 0.2, 0.25, 0.5)
        smile = 0.2
    turn_t = d10s + 1.45
    if t >= turn_t:
        a = smoothstep(turn_t, turn_t + 0.45, t)
        look = tuple(lerp(p, q, a) for p, q in zip(look, look_at_her))
        head_tilt += 0.12 * a
        lean += 0.04 * a
    if t >= turn_t + 0.9:
        eyes, smile = 'happy', 0.85
    hug = 0.0
    if t >= warn + 0.2:
        eyes, look, smile = 'surprised', look_at_her, 0.1
    if t >= g7s + 0.4:
        eyes, smile = 'worried', -0.2
        lean += 0.05
    hug_t = g7s + 1.5
    hug = smoothstep(hug_t, hug_t + 0.7, t)
    if t >= g7e:
        k = t - g7e
        eyes, smile = 'worried', -0.4
        shake = 0.35
        look = (0.55 * math.sin(k * 4.0) + 0.1, 0.1 - 0.3 * abs(math.sin(k * 2.0)))
    if t >= idea:
        k = t - idea
        a = ease_out(clamp(k / 0.35))
        look = tuple(lerp(p, q, a) for p, q in zip(look, (0.5, -1.0)))
        head_tilt = lerp(head_tilt, 0.12, a)
        eyes = 'surprised'
        shake = 0.35 * (1 - a)
        if k > 0.62:
            eyes, smile = 'determined', 0.35
        ant = U.bump(k - 0.5, 0.05, 0.9) + (0.25 if k > 0.5 else 0)
    # ---- Guang ----
    geyes, glook, gsmile = 'open', (-0.3, -0.9), 0.35
    glow, flicker = 1.0, 0.0
    garm_l, garm_r = 0.1, 0.1
    blush = 0.6
    grot = 0.0
    if g6s - 0.3 <= t:
        a = smoothstep(g6s - 0.3, g6s + 0.2, t)
        glook = (lerp(-0.3, 0.6, a), lerp(-0.9, -0.7, a))
        gsmile = 0.15
    happy_t = d10s + 2.6
    if t >= happy_t:
        a = smoothstep(happy_t, happy_t + 1.2, t)
        glow = 1.0 + 0.4 * a
        geyes, gsmile, blush = 'happy', 0.9, 0.6 + 0.4 * a
        garm_l = garm_r = 0.5 * a
        grot = -0.08 * a
    if t >= warn:
        k = t - warn
        dim = smoothstep(0, 1.8, k)
        glow = lerp(1.4, 0.45, dim)
        flicker = 0.9 * (1 - smoothstep(1.2, 3.0, k)) + 0.45
        fl = noise1(T * 16.0, 7) * 0.5 + noise1(T * 5.0, 9) * 0.5
        glow *= 1 - 0.3 * flicker * max(0.0, fl)
        if k < 0.25:
            glow *= 1.25  # a last bright sputter before dimming
        geyes, glook, gsmile = 'surprised', (0.0, 0.4), 0.0
        garm_l = garm_r = 0.1
        grot = 0.0
        blush = 0.4
    if t >= g7s - 0.3:
        geyes, glook, gsmile = 'sleepy', (0.5, -0.4), -0.1
        grot = 0.1 * smoothstep(g7s, g7e, t)
        garm_l = garm_r = -0.2
    return dict(eyes=eyes, look=look, smile=smile, head_tilt=head_tilt, lean=lean + idl['lean'], shake=shake,
                ant=ant, blink_extra=blink_extra, head_dy=idl['head_dy'], hug=hug,
                geyes=geyes, glook=glook, gsmile=gsmile, glow=glow, flicker=flicker, garm_l=garm_l, garm_r=garm_r,
                blush=blush, grot=grot)


def _pair_pose(f, st, T, beam_lit=0.0):
    rt = _rock_top()
    hug = st['hug']
    return robot_pose(x=ROCK5[0] + 8, y=rt + 4, scale=DS5, facing=-1.0, sit=1.0, sit_dangle=1.0,
                      eyes=st['eyes'], look=st['look'], smile=st['smile'], head_tilt=st['head_tilt'],
                      lean=st['lean'], shake=st['shake'], arm_l=(0.3, 0.15), arm_r=(0.75, 1.1), hand_r_open=0.8,
                      hug=hug, blink=max(auto_blink(T, 2), st['blink_extra']), mouth=f.mouth('deng'),
                      head_dy=st['head_dy'], scarf_wind=0.35, antenna_flash=clamp(st['ant']),
                      antenna_rate=1.0, rim=0.35 + 0.6 * beam_lit, rim_color='#ffe0a0' if beam_lit > 0.1 else '#9fc4ff',
                      rim_angle=-2.4 if beam_lit > 0.1 else -math.pi / 2, shadow=0.25)


def _draw_pair(c, f, st, T, beam_lit=0.0):
    pose = _pair_pose(f, st, T, beam_lit)
    res = draw_deng(c, pose, T)
    kx, ky = U.knee_point(res, pose)
    bob = 2.5 * math.sin(T * 1.9) * (1 - st['hug'])
    lap = (kx - 6 * pose.facing, ky - R_STAR * GS5 * 0.8 + bob)
    hx, hy = res['heart']
    held = (hx + 4, hy + 6)
    h = ease_in_out(st['hug'])
    gp = star_pose(x=lerp(lap[0], held[0], h), y=lerp(lap[1], held[1], h) - 14 * math.sin(math.pi * h),
                   scale=GS5, glow=st['glow'], flicker=st['flicker'], eyes=st['geyes'], look=st['glook'],
                   smile=st['gsmile'], arm_l=st['garm_l'], arm_r=st['garm_r'], blush=st['blush'], rot=st['grot'],
                   blink=auto_blink(T, 3, 4.3), mouth=f.mouth('guang'), leg_l=0.3, leg_r=-0.1,
                   halo=1.0 - 0.3 * h)
    draw_guang(c, gp, T)
    return res, pose, gp


def _beam_aimed(C, t, ang_hit, k=0.62):
    """2D beam angle that sweeps like the rotating lens, then eases to rest on Deng at `idea`
    (story-driven stop). Returns (angle, on_target 0..1)."""
    t_ref = C['idea'] - 0.7
    u = BEAM_RATE * (t - t_ref)
    A = 0.5
    du = u if u < -A else -A * math.exp(-(u + A) / A)
    ang = ang_hit + du * k
    return ang, math.exp(-(du * k / 0.09) ** 2)


def shot_sit(f, C, medium=False):
    c, t, T = f.canvas, f.t, f.T
    dawn = _dawn(C, t)
    sit, warn, idea = C['sit'], C['warn'], C['idea']
    t_med = warn - 0.7
    st = _sit_state(f, C, t)
    face = U.measure_robot(_pair_pose(f, st, T), T)['head']
    if not medium:
        u = ease_in_out(f.between(sit, t_med))
        cam = Camera(x=lerp(1010, 1050, u), y=lerp(660, 680, u), zoom=lerp(1.32, 1.42, u), t=T)
    else:
        u = ease_in_out(f.between(t_med, idea))
        cx, cy, z = lerp(1170, 1165, u), lerp(760, 752, u), lerp(2.2, 2.3, u)
        pi = ease_in_out(clamp((t - idea - 0.45) / 1.0))
        cx, cy, z = lerp(cx, face[0] - 30, pi), lerp(cy, face[1] + 45, pi), lerp(z, 3.0, pi)
        cam = Camera(x=cx, y=cy, zoom=z, t=T)
    lx, ly = lamp5()
    rt = _rock_top()
    lights = [(lx, ly, '#ffd36b', 0.7), (ROCK5[0] - 20, rt - 40, '#ffd86b', 0.3 * st['glow'], _rock_wl())]
    backdrop(c, f, cam, HOR5, dawn=dawn, lights=lights, y_par=0.2 if medium else 0.5,
             milky=((1080, 130), -0.34, 1.15), sun_x=1650, hero=[(820, 150, 1.0)])
    lit = 0.0
    ang = None
    if medium:
        ang_hit = math.atan2(face[1] - ly, face[0] - lx)
        ang, lit = _beam_aimed(C, t, ang_hit)
    with cam.apply(c):
        # reflections first (on the water, under the solid props)
        def _refl_rock(cc):
            island(cc, *ROCK5, dawn=dawn)
            _draw_pair(cc, f, st, T, lit)

        def _refl_lh(cc):
            island(cc, *LI5, dawn=dawn)
            lighthouse(cc, *LH5, T, dawn=dawn, phi=T * BEAM_RATE)
        U.reflection(c, _li_wl(), _refl_lh, T, alpha=0.35, amp=5.0)
        U.reflection(c, _rock_wl(), _refl_rock, T, alpha=0.5, amp=4.0)
        island(c, *LI5, dawn=dawn)
        lighthouse(c, *LH5, T, dawn=dawn, phi=None if medium else T * BEAM_RATE)
        if not medium:
            U.draw_beam(c, lx, ly, T * BEAM_RATE, length=2600, intensity=0.8, horizon_y=HOR5, t=T)
        island(c, *ROCK5, dawn=dawn)
        res, pose, gp = _draw_pair(c, f, st, T, lit)
        if medium:
            U.draw_beam_screen(c, lx, ly, ang, length=2200, width1=560, intensity=0.75, t=T)
            if lit > 0.01:
                hx, hy = res['head']
                soft_glow(c, hx, hy, 130, '#ffe7a0', 0.35 * lit)
                soft_glow(c, hx - 20, hy + 60, 240, '#ffd36b', 0.18 * lit)
        antenna_flash_local(c, res, st['ant'])
        if t >= idea + 0.5:
            ax, ay = res['antenna']
            U.sparkle_burst(c, ax, ay, t - idea - 0.5, count=10, life=0.8, color='#ffe7b0', seed=97, size=0.35)
        happy = smoothstep(C['D10'][0] + 2.6, C['D10'][0] + 3.4, t) * (1 - smoothstep(warn, warn + 0.5, t))
        if happy > 0:
            U.twinkles(c, T, gp.x, gp.y, radius=55, count=7, seed=98, intensity=happy, size=0.6)
    g = dict(vignette=0.5)
    if medium:
        g['sat'] = 1.0 - 0.12 * smoothstep(warn, warn + 2.5, t)
        g['exposure'] = 1.0 + 0.06 * lit
        g['bloom'] = 1.0 + 0.3 * lit
    else:
        g['bloom'] = 1.05
    f.grade.update(g)


# ============================================================================
def render(f):
    C = _cues(f)
    t = f.t
    f.grade.update(vignette=0.48)
    if t < C['stack']:
        shot_throw(f, C)
    elif t < C['climb']:
        shot_stack(f, C)
    elif t < C['splash'] + 0.55:
        shot_climb(f, C)
    elif t < C['sit']:
        shot_water(f, C)
    elif t < C['warn'] - 0.7:
        shot_sit(f, C, medium=False)
    else:
        shot_sit(f, C, medium=True)
