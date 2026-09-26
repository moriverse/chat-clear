"""s03 · 流星 The falling star (20 s) — owner: SCENE-S03.

Shots (boundaries derived from cues / line times, never hard-coded):
  S1  0 → D03 start         wide, behind Deng on the gallery: huge sky, one star twinkles oddly;
                            at `meteor_start` it breaks loose.
  S2  D03 → impact-1.3      medium on Deng (low angle), the meteor crossing the sky, D03.
  S3  impact-1.3 → +1.0     over the railing, camera tilts down with the plunge; impact: flash,
                            splash, shockwave, plankton ring; Deng springs up.
  S4  impact+1.0 → run_down exterior wide: the ring of light races across the whole sea past the
                            rim-lit lighthouse; D04; Deng dashes inside.
  S5  run_down → door-0.12  comic: clattering down the spiral stairs (hurry cadence 2.2 c/s from
                            run_down: foot contacts at walk_phase multiples of 0.5).
  S6  → boat_out            the door bangs open at DOOR_BANG = boat_out - 1.0, Deng shoots out.
  S7  boat_out → end        wide low angle: rowing (one stroke / 1.6 s from boat_out, blade enters
                            the water at each stroke start) over the glowing sea towards a faint glow
                            under the water.
"""
from __future__ import annotations

import math

import numpy as np
import skia

from engine.core import (ADD, H, W, Camera, at, clamp, ease_in, ease_in_out, ease_out, ease_out_back, fill,
                         keyframes, lerp, linear, mix, noise1, poly, pulse, radial, smoothstep, soft_glow, stroke)
from lib import fx, robot, sea, sky

from scenes import _s03_util as U

TAU = math.tau
HZW = 700.0            # world y of the horizon for the gallery shots (S1-S3)
HEAD_UP = -1.0         # sign of RobotPose.head_tilt that tips a right-facing head back (look up)
WALK_HURRY = 2.2       # cycles / s (shared convention with SFX)
STROKE = 1.6           # s per oar stroke (shared convention with SFX)
BIO_R, BIO_LIFE = 870.0, 12.0   # plankton ring (ground units): radius reached at 0.35*life

GOLD = '#ffd36b'
CYAN = '#6ff0ff'
STARWARM = '#ffe58a'

# meteor path (gallery-world coords, quadratic like fx.meteor's `ctrl`): breaks loose high in
# the sky, shoots right, arcs over and plunges into the sea far out
MP0, MCTRL, MP1 = (700.0, 240.0), (1650.0, 70.0), (1620.0, HZW + 92.0)
MPATH = U.Bezier([MP0, MCTRL, MP1])
MW_S13 = dict(center=(1150, 170), angle=0.75, length=4600, width=540, bend=0.04)


# ============================================================================
# timing
# ============================================================================
class Cues:
    def __init__(self, f):
        self.ms = f.cue('meteor_start')
        self.imp = f.cue('impact')
        self.run = f.cue('run_down')
        self.boat = f.cue('boat_out')
        self.d03 = f.line('D03')
        self.d04 = f.line('D04')
        self.s2 = self.d03[0]
        self.s3 = self.imp - 1.3
        self.s4 = self.imp + 1.0
        self.door = self.boat - 1.0          # DOOR BANG instant (for SFX)
        self.s6 = self.door - 0.12


def meteor_s(cu, t):
    """Arc-length fraction along MPATH (accelerating)."""
    u = (t - cu.ms) / (cu.imp - cu.ms)
    if u <= 0:
        return 0.0
    u = min(u, 1.0)
    return u * (0.5 + 0.5 * u ** 1.25)


def meteor_size(cu, t):
    u = clamp((t - cu.ms) / (cu.imp - cu.ms))
    return 0.22 + 1.1 * u ** 2.0


def meteor_head(cu, t):
    return MPATH.at(meteor_s(cu, t))


# ============================================================================
# shared drawing
# ============================================================================
def view_rect(cam, pad=260):
    return (cam.x - W / 2 / cam.zoom - pad, cam.y - H / 2 / cam.zoom - pad, W / cam.zoom + 2 * pad,
            H / cam.zoom + 2 * pad)


def far_sky(c, T, cam, mw=MW_S13, mw_alpha=0.9, stars_bright=1.0, clouds=0.0, hz=HZW):
    rect = view_rect(cam)
    U.call(sky.sky, c, rect=rect, horizon_y=hz)
    U.safe(sky.milky_way, c, T, alpha=mw_alpha, horizon_y=hz, **mw)
    U.call(sky.stars, c, T, rect=rect, horizon_y=hz, bright=stars_bright, seed=7)
    if clouds > 0:
        U.safe(sky.clouds, c, T, y=hz - 90, alpha=clouds, speed=5.0, scale=1.0, cover=0.35, layers=2,
               rect_x=(rect[0], rect[0] + rect[2]))


def far_sea(c, T, cam, lights=(), hz=HZW, calm=0.8):
    U.call(sea.ocean, c, T, horizon_y=hz, rect_x=(cam.x - 1400, cam.x + 1400), bottom=cam.y + 900,
           lights=lights, calm=calm, cx=cam.x)


def odd_star(c, t, T, x, y, cu):
    """The star that twinkles oddly before breaking loose (it is Guang)."""
    k = smoothstep(0.0, 1.6, t)
    fl = 0.8 + 0.2 * noise1(t * 7.0, 3) + 0.25 * max(0.0, noise1(t * 19.0, 5))
    flare = 0.0
    for tb, amp in ((0.55, 0.25), (1.25, 0.45), (1.62, 0.3), (2.0, 0.6), (2.28, 0.5)):
        flare = max(flare, amp * U.call(sky.hero_flare, t - tb, dur=0.45) if hasattr(sky, 'hero_flare') else 0.0)
    pre = smoothstep(cu.ms - 0.8, cu.ms, t)
    jx, jy = noise1(t * 11.0, 21) * 2.5 * pre, noise1(t * 11.0, 22) * 2.5 * pre
    if hasattr(sky, 'hero_star'):
        U.call(sky.hero_star, c, x + jx, y + jy, size=0.8 + 0.5 * k + 0.4 * pre, flare=clamp(flare + 0.35 * pre),
               t=T, warmth=0.55 + 0.45 * pre, intensity=(0.55 + 0.45 * k) * fl, flare_age=0.03)
    else:
        fx.glow(c, x, y, 40, STARWARM, 0.8 * fl)


def draw_meteor(c, cu, t, T, intensity=1.0):
    """Library meteor on the curved path (gallery-world coords, inside the far camera)."""
    if t < cu.ms or t > cu.imp + 0.02:
        return
    s = min(1.0, meteor_s(cu, t))
    sz = meteor_size(cu, t)
    u = clamp((t - cu.ms) / (cu.imp - cu.ms))
    fx.meteor(c, MP0[0], MP0[1], MP1[0], MP1[1], s, ctrl=MCTRL, tail=260 + 700 * u, size=sz, t=T,
              duration=cu.imp - cu.ms, sparks=1.3, grow=1.0, intensity=intensity, curl=0.8)
    # break-loose flare at the start
    fl = pulse(t, cu.ms, cu.ms + 0.05, 0.25)
    if fl > 0.01:
        fx.impact_flash(c, MP0[0], MP0[1], t - cu.ms + 0.05, radius=60, intensity=0.8, life=0.7)


def gallery_floor(c, floor_y, x0=-400, x1=W + 400, warm=1.0, cool=1.0):
    """Iron gallery floor behind Deng (near layer)."""
    c.drawRect(skia.Rect.MakeLTRB(x0, floor_y, x1, H + 600),
               linear((0, floor_y), (0, floor_y + 260), [(0, '#1c2140', 1), (1, '#080a16', 1)]))
    p = stroke('#2a3158', 2.0, 0.45)
    for k in range(1, 8):
        yy = floor_y + (k ** 1.7) * 9
        c.drawLine(x0, yy, x1, yy, p)
    c.drawRect(skia.Rect.MakeLTRB(x0, floor_y - 2, x1, floor_y + 3), fill('#4a5a8e', 0.55 * cool))
    if warm > 0:  # lamp-room light spilling from behind (off-screen left)
        fx.glow(c, -150, floor_y + 220, 1100, '#ffb347', 0.35 * warm, core=False, sy=0.45)


def beam_3d(c, T, hz, az0, strength=0.5):
    """The lighthouse beam seen from the gallery: a cone from behind the camera sweeping out to
    the horizon. Same angle convention as exterior shots (f.T * 0.9)."""
    az = (T * 0.9 - az0 + math.pi) % TAU - math.pi
    if abs(az) > 1.1:
        return
    vis = (1 - abs(az) / 1.1) ** 1.5
    vx = W / 2 + math.tan(az) * 900
    near = W / 2 + math.tan(az) * 2600
    path = poly([(vx - 6, hz - 3), (vx + 6, hz - 3), (near + 900, -300), (near - 900, -300)])
    p = linear((vx, hz), (near, -300), [(0, '#ffe7a0', 0.0), (0.25, '#ffe7a0', 0.08 * strength * vis),
                                         (1, '#ffe7a0', 0.22 * strength * vis)])
    p.setBlendMode(ADD)
    c.drawPath(path, p)
    fx.glow(c, vx, hz, 140, '#ffe7a0', 0.3 * strength * vis, core=False, sy=0.35)


def deng(c, pose_kw, T, bounds=None, washes=()):
    pose = U.mkpose(robot.RobotPose, **pose_kw)
    if bounds is None:
        s = pose.scale
        bounds = skia.Rect.MakeLTRB(pose.x - 360 * s, pose.y - 480 * s, pose.x + 360 * s, pose.y + 80 * s)
    return U.lit(c, bounds, lambda: robot.draw_robot(c, pose, T), list(washes))


def antenna_flare(c, res, amt, t, t_since=None):
    if amt <= 0.01 or not res or 'antenna' not in res:
        return
    x, y = res['antenna']
    fx.glow(c, x, y, 90 * amt + 12, '#ff6b5b', 0.9 * amt)
    if t_since is not None:
        fx.sparkle_burst(c, x, y, t_since, scale=0.35, color='#ffd0c0', count=8, life=0.6)


def look_at(eye, target):
    dx, dy = target[0] - eye[0], target[1] - eye[1]
    d = math.hypot(dx, dy) or 1.0
    return (clamp(dx / d, -1, 1), clamp(dy / d, -1, 1))


# ============================================================================
# S1 — the odd star, it breaks loose
# ============================================================================
NIGHT_AMB = '#c9d2f2'


def shot_sky(f, cu):
    c, t, T = f.canvas, f.t, f.T
    drift = ease_in_out(t / max(0.1, cu.s2))
    cam = Camera(x=980 + 30 * drift, y=HZW - 360 + 10 * drift, zoom=1.0 + 0.015 * drift)
    hz = cam.to_screen(0, HZW)[1]
    mx, my = meteor_head(cu, t)
    msx, msy = cam.to_screen(mx, my)
    with cam.apply(c, 1.0):
        far_sky(c, T, cam)
        lights = [(mx, my, '#ffe7a0', 0.6 * meteor_size(cu, t))] if t > cu.ms else []
        far_sea(c, T, cam, lights)
        if t < cu.ms + 0.1:
            odd_star(c, t, T, *MP0, cu)
        draw_meteor(c, cu, t, T)
    # near layer: gallery + Deng seen from behind (over the shoulder)
    ncam = Camera(x=960 + 10 * drift, y=540 + 4 * drift, zoom=1.0 + 0.03 * drift)
    with ncam.apply(c, 1.0):
        floor_y = 1010
        U.call(sea.gallery_railing, c, -300, W + 300, floor_y - 130, 0.93)
        gallery_floor(c, floor_y)
        # chin in hands (as s02 left him) -> notices the twinkle -> head up -> it shoots
        notice = smoothstep(1.1, 1.9, t)
        shoot = smoothstep(cu.ms, cu.ms + 0.4, t)
        rx, ry, s = 400, floor_y + 40, 1.45
        ex, ey = ncam.to_screen(rx, ry - 250 * s)
        tgt = (msx, msy) if t > cu.ms else cam.to_screen(*MP0)
        lk = look_at((ex, ey), tgt)
        idl = robot.idle(T)
        rim_a = math.atan2(tgt[1] - ey, tgt[0] - ex)
        pose = dict(x=rx, y=ry, scale=s, facing=1, back=True, sit=1.0, sit_dangle=0.0,
                    chin_hands=1 - notice, head_tilt=0.28 * notice * lk[0] + idl['head_tilt'] + 0.1 * shoot,
                    head_dy=idl['head_dy'] + 5 * (1 - notice) - 6 * notice,
                    lean=0.05 * (1 - notice) - 0.06 * shoot, arm_l=(0.35, 0.5), arm_r=(0.35, 0.5),
                    scarf_wind=0.45, ambient=NIGHT_AMB,
                    rim=0.45 + 0.6 * shoot * meteor_size(cu, t), rim_color='#fff1c8' if shoot > 0.5 else '#9fc4ff',
                    rim_angle=rim_a if shoot > 0.5 else -math.pi / 2 + 0.3)
        washes = [((rx - 260, ry + 80), (rx + 80, ry - 250), '#ffb347', 0.22)]
        deng(c, pose, T, washes=washes)
    f.grade.update(vignette=0.55, bloom=1.05)


# ============================================================================
# S2 — medium on Deng, D03
# ============================================================================
def shot_medium(f, cu):
    c, t, T = f.canvas, f.t, f.T
    k = (t - cu.s2) / max(0.1, cu.s3 - cu.s2)
    cam = Camera(x=1010 + 50 * k, y=HZW - 290 + 30 * ease_in_out(k), zoom=1.0)
    mx, my = meteor_head(cu, t)
    sz = meteor_size(cu, t)
    msx, msy = cam.to_screen(mx, my)
    with cam.apply(c, 1.0):
        far_sky(c, T, cam)
        far_sea(c, T, cam, [(mx, my, '#ffe7a0', 0.6 * sz)])
        draw_meteor(c, cu, t, T)

    # near layer: railing to his right, Deng bigger, low angle
    ncam = Camera(x=960 - 25 * k, y=540, zoom=1.0 + 0.05 * ease_in_out(k))
    with ncam.apply(c, 1.0):
        U.call(sea.gallery_railing, c, 700, W + 400, 860, 1.9)
        gallery_floor(c, 860 + 266, x0=560, warm=0.6)
        rx, ry, s = 420, 1260, 2.15
        ex, ey = ncam.to_screen(rx + 20 * s, ry - 250 * s)
        lk = look_at((ex, ey), (msx, msy))
        elev = clamp(-lk[1], -0.3, 1.0)
        mouth = f.mouth('deng')
        lean_in = smoothstep(cu.s2, cu.s2 + 0.6, t)
        idl = robot.idle(T)
        pose = dict(x=rx, y=ry, scale=s, facing=1, sit=1.0, turn=0.15,
                    head_tilt=HEAD_UP * (0.1 + 0.3 * elev) + idl['head_tilt'] + 0.04 * math.sin(t * 5) * mouth,
                    head_dy=idl['head_dy'] - 6 * lean_in, lean=0.08 * lean_in,
                    arm_l=(1.35, 0.35), arm_r=(1.25 + 0.2 * lean_in, 0.3), hand_r_open=0.2, hand_l_open=0.2,
                    eyes='surprised', blink=robot.auto_blink(T, every=4.3) if t > cu.s2 + 0.8 else 0.0,
                    look=lk, mouth=mouth, smile=0.0, scarf_wind=0.55, ambient=NIGHT_AMB,
                    rim=clamp(0.4 + 0.5 * sz), rim_color='#fff1c8', rim_angle=math.atan2(msy - ey, msx - ex))
        a = clamp(0.06 + 0.16 * sz)
        washes = [((msx, msy), (rx - 200, ry - 200), '#fff1c8', a),
                  ((rx - 500, ry), (rx, ry - 300), '#ffb347', 0.14)]
        deng(c, pose, T, washes=washes)
    f.grade.update(vignette=0.5, bloom=1.05 + 0.35 * k)


# ============================================================================
# S3 — the plunge & impact (over the railing)
# ============================================================================
S3_CAM0, S3_CAM1 = HZW - 250, HZW + 70
S3_H = 55.0


def s3_sv(cam):
    return U.SeaView(hz=cam.to_screen(0, HZW)[1], F=1000.0, cam_h=S3_H, cx=cam.to_screen(960, 0)[0], cam_x=0.0)


_PL = {}


def plankton(key, sv_factory, **kw):
    if key not in _PL:
        _PL[key] = U.Plankton(sv_factory(), **kw)
    return _PL[key]


def sea_impact_fx(c, sv, ts, X0, Z0, pl, T, strength=1.0, pl_size=1.0):
    """Everything on the water after the impact: ground-plane effects through the perspective
    homography (true perspective rings) + screen-space sparkles + the vertical splash."""
    if ts < -0.02:
        return
    ts = max(0.0, ts)
    ix, iy = sv.proj(X0, Z0)
    sc = sv.scale(Z0)
    with U.ground(c, sv):
        fx.bioluminescence(c, T, X0, Z0, radius=BIO_R, persp=1.0, t_since=ts, life=BIO_LIFE, size=1.3,
                           count=1500, intensity=strength)
        fx.ripples(c, X0, Z0, ts, color='#bfe0ff', intensity=0.9 * strength, persp=1.0, count=5, speed=150,
                   life=3.6, width=5)
        fx.shockwave(c, X0, Z0, ts, color='#fff4d0', max_r=900, life=1.3, persp=1.0, intensity=strength)
    br = U.ring_brightness(pl, X0, Z0, ts, radius=BIO_R, life=BIO_LIFE) * strength
    pl.draw(c, sv, T, br, size=pl_size)
    # glowing ring fronts sweeping across the water (true perspective)
    R = U.bio_front(ts, BIO_R, BIO_LIFE)
    for j, (dr, a0) in enumerate(((0.0, 1.0), (-0.07, 0.45), (-0.15, 0.25))):
        Rj = R * (1 + dr)
        if Rj < 4:
            continue
        fade = strength * a0 * (0.35 + 0.65 * math.exp(-ts / 2.5))
        path = U.ground_ring_path(sv, X0, Z0, Rj, n=160)
        c.drawPath(path, stroke(CYAN, 14, 0.22 * fade, blur=9))
        c.drawPath(path, stroke('#bff8ff', 3.0, 0.5 * fade, blur=1.2))
        if j == 0:
            c.drawPath(path, stroke(GOLD, 6, 0.25 * fade * math.exp(-ts / 1.5), blur=3))
    # the vertical burst: impact flash, splash of light, column
    fx.impact_flash(c, ix, iy, ts, radius=120 * sc / 1.8, intensity=strength, life=1.6, sy=0.5)
    fx.splash(c, ix, iy, ts, scale=0.9 * sc / 1.8, glow_color='#ffe7a0', persp=S3_H / Z0 * 2.5,
              intensity=strength, height=1.3)
    if ts < 1.5:
        k = ts / 1.5
        hcol = 700 * sc / 1.8 * ease_out(min(1, ts / 0.2)) * (1 - k) ** 0.6
        wcol = 40 * sc / 1.8 * (1 - 0.5 * k)
        p = linear((ix, iy), (ix, iy - hcol), [(0, '#ffffff', 0.9 * (1 - k) * strength),
                                               (0.35, '#ffe7a0', 0.5 * (1 - k) * strength), (1, '#ffe7a0', 0)])
        p.setBlendMode(ADD)
        c.drawPath(poly([(ix - wcol, iy), (ix + wcol, iy), (ix + wcol * 0.15, iy - hcol),
                         (ix - wcol * 0.15, iy - hcol)]), p)
    # the star, now under the water: a faint golden glow that stays
    uw = smoothstep(0.5, 1.8, ts)
    if uw > 0:
        flat_glow(c, ix, iy + 3, 90 * sc, GOLD, 0.5 * uw * (0.85 + 0.15 * math.sin(T * 2.3)), sy=0.22)
        flat_glow(c, ix, iy + 3, 30 * sc, '#fff3cf', 0.6 * uw, sy=0.3)


def shot_impact(f, cu):
    c, t, T = f.canvas, f.t, f.T
    k = ease_in_out((t - cu.s3) / (cu.imp - cu.s3))
    ts = t - cu.imp
    shake = 16 * math.exp(-ts * 2.6) if ts > 0 else 0.0
    c.save()
    c.translate(shake * noise1(t * 13, 11), shake * noise1(t * 13, 29))
    cam = Camera(x=1000, y=lerp(S3_CAM0, S3_CAM1, k), zoom=1.0)
    mx, my = meteor_head(cu, t)
    msx, msy = cam.to_screen(mx, my)
    ix_s, iy_s = cam.to_screen(*MP1)
    with cam.apply(c, 1.0):
        far_sky(c, T, cam)
        if ts < 0:
            lights = [(mx, my, '#ffe7a0', 0.7 * meteor_size(cu, t))]
        else:
            lights = [(MP1[0], MP1[1] - 200, '#ffe7a0', 1.2 * math.exp(-ts * 1.4), MP1[1])]
        far_sea(c, T, cam, lights)
    sv = s3_sv(cam)
    X0, Z0 = sv.ground(ix_s, iy_s)
    pl = plankton('s3', lambda: s3_sv(Camera(x=1000, y=S3_CAM1)), n=1800, seed=31)
    sea_impact_fx(c, sv, ts, X0, Z0, pl, T)
    with cam.apply(c, 1.0):
        draw_meteor(c, cu, t, T)
    # near layer (partial parallax with the tilt)
    ny = 540 + (cam.y - S3_CAM0) * 0.3
    ncam = Camera(x=960, y=ny, zoom=1.06)
    flash_a = math.exp(-max(0.0, ts) * 1.6) if ts >= 0 else 0.0
    ring_a = 0.0
    if ts > 0:
        ring_a = 0.4 * smoothstep(0.3, 1.2, ts)
    pre = 0.15 + 0.35 * meteor_size(cu, t) if ts < 0 else 0.0
    with ncam.apply(c, 1.0):
        floor_y = 1060
        bnd = skia.Rect.MakeLTRB(-400, floor_y - 200, W + 400, floor_y + 5)
        U.lit(c, bnd, lambda: U.call(sea.gallery_railing, c, -400, W + 400, floor_y - 140, 1.0),
              [((ix_s, iy_s - 200), (ix_s - 900, floor_y), '#fff4d0', 0.7 * flash_a + 0.3 * pre),
               ((ix_s, floor_y - 300), (ix_s - 1400, floor_y), CYAN, ring_a)])
        gallery_floor(c, floor_y)
        if flash_a > 0.01:
            c.drawRect(skia.Rect.MakeLTRB(-400, floor_y - 2, W + 400, floor_y + 4), fill('#fff4d0', 0.8 * flash_a))
        # Deng from behind: watching, startled by the flash, springs up
        rx, ry, s = 520, floor_y - 4, 1.1
        jump = ts - 0.22
        sit, squash, y_off, lean = 1.0, 0.0, 0.0, 0.0
        arm_l = arm_r = (0.5, 0.6)
        flare = 0.0
        if ts > 0:
            flinch = pulse(ts, 0.0, 0.1, 0.1)
            lean = -0.12 * flinch
            squash = -0.25 * flinch
        if jump > 0:
            # squash (anticipation) -> stretch up -> land standing -> settle
            sit = 1 - ease_out(clamp(jump / 0.25))
            y_off = -120 * math.sin(math.pi * clamp(jump / 0.42)) * s
            squash = keyframes(jump, [(0.0, -0.35), (0.1, 0.45), (0.38, 0.2), (0.46, -0.3), (0.7, 0.05), (0.9, 0.0)])
            arm_up = pulse(jump, 0.05, 0.45, 0.12)
            arm_l = (lerp(0.5, 2.8, arm_up), lerp(0.6, 0.3, arm_up))
            arm_r = (lerp(0.5, 2.6, arm_up), lerp(0.6, 0.2, arm_up))
            if jump > 0.5:
                pt = smoothstep(0.5, 0.8, jump)
                arm_r = (lerp(arm_r[0], 1.5, pt), lerp(arm_r[1], 0.1, pt))
                arm_l = (lerp(arm_l[0], 0.9, pt), lerp(arm_l[1], 0.6, pt))
            flare = pulse(jump, 0.05, 0.6, 0.3)
            lean = lerp(lean, 0.1, smoothstep(0.4, 0.8, jump))
        ex, ey = ncam.to_screen(rx, ry - 250 * s + y_off)
        tgt = (msx, msy) if ts < 0 else (ix_s, iy_s)
        lk = look_at((ex, ey), tgt)
        idl = robot.idle(T)
        rim_col = mix('#fff1c8', CYAN, smoothstep(0.4, 1.4, ts)) if ts > 0 else '#fff1c8'
        pose = dict(x=rx, y=ry + y_off, scale=s, facing=1, back=True, sit=sit, sit_dangle=0.0, squash=squash,
                    head_tilt=0.3 * lk[0] + idl['head_tilt'], head_dy=idl['head_dy'] - 8 * clamp(-lk[1]),
                    lean=lean, arm_l=arm_l, arm_r=arm_r, hand_l_open=0.9 if jump > 0 else 0.3,
                    hand_r_open=0.9 if jump > 0 else 0.3, scarf_wind=0.6 + 0.4 * flash_a, ambient=NIGHT_AMB,
                    rim=clamp(0.5 + 0.5 * pre + 1.0 * flash_a + ring_a), rim_color=rim_col,
                    rim_angle=math.atan2(tgt[1] - ey, tgt[0] - ex), antenna_flash=flare)
        washes = [((ix_s, iy_s), (rx - 100, ry - 200), '#fff4d0', 0.55 * flash_a),
                  ((ix_s, ry - 600), (rx - 200, ry), CYAN, ring_a * 0.6),
                  ((rx - 400, ry + 60), (rx + 60, ry - 250), '#ffb347', 0.16)]
        if ts < 0:
            washes.append(((msx, msy), (rx - 100, ry), '#fff1c8', 0.5 * pre))
        res = deng(c, pose, T, washes=washes)
        antenna_flare(c, res, flare * 0.6, t, jump - 0.05 if jump > 0.05 else None)
    c.restore()
    g = {'vignette': 0.5}
    if ts >= 0:
        g['white'] = 0.85 * math.exp(-ts / 0.09) + 0.1 * math.exp(-ts / 0.6)
        g['bloom'] = 1.2 + 1.5 * math.exp(-ts / 0.8)
        g['exposure'] = 1.0 + 0.2 * math.exp(-ts / 0.5)
    else:
        pre_ = smoothstep(cu.imp - 0.6, cu.imp, t)
        g['bloom'] = 1.2 + 0.5 * pre_
        g['exposure'] = 1.0 + 0.08 * pre_
    f.grade.update(g)


# ============================================================================
# S4 — exterior at gallery height: D04 on the gallery; the ring races across the sea below
# ============================================================================
S4_HZ = 560.0
S4_H = 70.0
S4_LH = (300.0, 3480.0, 7.0)     # lighthouse base x, y (far below the frame), scale


def s4_sv(cam=None):
    if cam is None:
        return U.SeaView(hz=S4_HZ, F=1000.0, cam_h=S4_H)
    return U.SeaView(hz=cam.to_screen(0, S4_HZ)[1], F=1000.0 * cam.zoom, cam_h=S4_H, cx=cam.to_screen(960, 0)[0])


S4_IMPACT = s4_sv().ground(1530, 640)
S4_LH_GROUND = (-20.0, 45.0)     # where the tower stands on the sea plane (near the camera)


def shot_exterior(f, cu):
    c, t, T = f.canvas, f.t, f.T
    ts = t - cu.imp
    k = ease_in_out((t - cu.s4) / max(0.1, cu.run - cu.s4))
    shake = 6 * math.exp(-(ts - 1.0) * 1.8)
    c.save()
    c.translate(shake * noise1(t * 11, 3), shake * noise1(t * 11, 4))
    cam = Camera(x=960 + 20 * k, y=540 + 8 * k, zoom=1.0 + 0.025 * k)
    lhx, lhy, lhs = S4_LH
    A = U.call(sea.lighthouse_anchors, lhx, lhy, lhs) if hasattr(sea, 'lighthouse_anchors') else None
    lamp_xy = A['lamp'] if A else (lhx, lhy - 445 * lhs)
    with cam.apply(c, 1.0):
        rect = view_rect(cam, 400)
        U.call(sky.sky, c, rect=rect, horizon_y=S4_HZ)
        U.safe(sky.milky_way, c, T, center=(1250, 120), angle=0.24, length=3600, width=420, alpha=0.6,
               horizon_y=S4_HZ, bend=0.03)
        U.call(sky.stars, c, T, rect=rect, horizon_y=S4_HZ, seed=9)
        U.call(sea.ocean, c, T, horizon_y=S4_HZ, rect_x=(rect[0], rect[0] + rect[2]), bottom=H + 400, calm=0.8,
               cx=cam.x, lights=[(1530, 440, '#ffe7a0', 0.8 * math.exp(-ts * 0.7), 640)])
    sv = s4_sv(cam)
    X0, Z0 = S4_IMPACT
    sea_impact_fx(c, sv, ts, X0, Z0, plankton('s4', s4_sv, n=2200, seed=44, x_range=(-200, W + 200)), T,
                  pl_size=1.0)
    ixs, iys = sv.proj(X0, Z0)
    Xi, Zi = S4_LH_GROUND
    R = U.bio_front(ts, BIO_R, BIO_LIFE)
    dist = math.hypot(Xi - X0, Zi - Z0)
    near = smoothstep(dist - 450, dist - 60, R)       # the ring front arriving below the tower
    flash_a = 0.4 * math.exp(-(ts - 1.0) * 1.4)
    ring_col = CYAN
    with cam.apply(c, 1.0):
        beam_ang = T * 0.9
        U.call(sea.beam, c, lamp_xy[0], lamp_xy[1], beam_ang, intensity=0.45, horizon_y=S4_HZ, t=T,
               length=3000, width0=30 * lhs, width1=520, flare=0.4)
        bnd = skia.Rect.MakeLTRB(lhx - 120 * lhs, lhy - 560 * lhs, lhx + 120 * lhs, H + 400)
        washes = [((ixs, iys - 150), (lhx - 60 * lhs, lhy - 450 * lhs), '#fff4d0', flash_a),
                  ((lhx + 80 * lhs, H + 200), (lhx - 40 * lhs, lhy - 470 * lhs), ring_col, 0.55 * near)]
        U.lit(c, bnd, lambda: U.call(sea.lighthouse, c, lhx, lhy, lhs, lamp=0.75, t=T, beam_angle=beam_ang,
                                     parts='back', view_e=0.03, glow=0.35), washes)
        # Deng on the gallery: points at the sea during D04, then dashes round to the door
        gy = A['gallery_y'] if A else lhy - 400 * lhs
        gx0, gx1 = A['gallery_x'] if A else (lhx - 66 * lhs, lhx + 66 * lhs)
        ds = (A['deng_scale'] if A else 0.1 * lhs) * 1.05
        d04s, d04e = cu.d04
        t_turn = d04e + 0.05
        run_u = clamp((t - t_turn) / max(0.1, cu.run - t_turn))
        stand_x = gx1 - 16 * lhs
        x = stand_x - run_u ** 1.6 * 150 * lhs
        facing = 1 if t < t_turn else -1
        talk = pulse(t, d04s, d04e, 0.15)
        hop = abs(math.sin((t - d04s) * 6.5)) * 10 * talk
        point = smoothstep(d04s - 0.1, d04s + 0.25, t) * (1 - smoothstep(t_turn - 0.1, t_turn + 0.1, t))
        settle = smoothstep(cu.s4, cu.s4 + 0.35, t)
        tgt = (ixs, iys)
        mouth = f.mouth('deng')
        idl = robot.idle(T)
        pose = dict(x=x, y=gy - hop, scale=ds, facing=facing, sit=0.0,
                    eyes='surprised' if t < d04s + 0.9 else 'determined', mouth=mouth,
                    look=(0.7, 0.35) if facing > 0 else (0.6, 0.0),
                    arm_r=(lerp(2.2, 1.45, settle) * point + 0.2 * (1 - point), 0.05 + 0.3 * (1 - point)),
                    arm_l=(lerp(2.6, 0.9, settle) * point + 0.25 * (1 - point), 0.5),
                    hand_r_open=1.0, walk=smoothstep(0, 0.15, run_u), run=1.0,
                    walk_phase=(t - t_turn) * WALK_HURRY, lean=0.12 * point + 0.25 * smoothstep(0, 0.2, run_u),
                    head_tilt=0.08 * point + idl['head_tilt'] + 0.05 * math.sin(t * 9) * mouth,
                    squash=0.15 * math.sin((t - d04s) * 13) * talk,
                    blink=robot.auto_blink(T, seed=3), scarf_wind=0.8, ambient=NIGHT_AMB,
                    rim=clamp(0.5 + flash_a + 0.8 * near), rim_color=mix('#fff1c8', CYAN, smoothstep(1.2, 2.2, ts)),
                    rim_angle=math.atan2(tgt[1] - gy, tgt[0] - x), antenna_flash=0.4 * talk)
        behind = x < lhx + 30 * lhs and facing < 0
        if not behind:
            deng(c, pose, T, washes=[((ixs, iys), (x - 60 * ds * 3, gy - 300 * ds), CYAN, 0.25 * near)])
        U.call(sea.lighthouse, c, lhx, lhy, lhs, lamp=0.75, t=T, beam_angle=beam_ang, parts='front', view_e=0.03,
               glow=0.3)
    c.restore()
    f.grade.update(bloom=1.15 + 0.5 * math.exp(-(ts - 1) * 1.5), vignette=0.5)


# ============================================================================
# S5 — the stairs (comic, fast)
# ============================================================================
_STEP0 = {}
STEPS_PER_FOOT = 2.0


def _step_pos0():
    """step_pos(k) at scroll=0, obtained by recording (not rasterising) the library call once."""
    if 'fn' not in _STEP0:
        rec = skia.PictureRecorder()
        cc = rec.beginRecording(skia.Rect.MakeWH(W, H))
        _STEP0['fn'] = U.call(sea.stairs_interior, cc, 0.0, scroll=0.0)
        rec.finishRecordingAsPicture()
    return _STEP0['fn']


class alpha_layer:
    def __init__(self, c, a):
        self.c, self.a = c, a

    def __enter__(self):
        p = skia.Paint()
        p.setAlphaf(clamp(self.a))
        self.c.saveLayer(None, p)

    def __exit__(self, *e):
        self.c.restore()


def shot_stairs(f, cu):
    c, t, T = f.canvas, f.t, f.T
    tr = t - cu.run
    phase = tr * WALK_HURRY
    k0 = 34.0
    kk = k0 - phase * 2 * STEPS_PER_FOOT          # step index (descending)
    sp0 = _step_pos0()
    x0, y0, d0 = sp0(kk)
    scroll = 760 - y0
    step_pos = U.call(sea.stairs_interior, c, T, scroll=scroll, light=0.75)
    if step_pos is None:
        step_pos = lambda kx: (sp0(kx)[0], sp0(kx)[1] + scroll, sp0(kx)[2])
    x, y, d = step_pos(kk)
    xa, ya, _ = step_pos(kk + 0.25)
    facing = 1 if x > xa else -1
    fx.speed_lines(c, t, direction=(x - xa, y - ya + 40), intensity=0.9, count=28, alpha=0.35,
                   rect=(0, 0, W, H), color='#ffe9c0')
    s = 0.95 * d
    bob = abs(math.sin(phase * math.pi)) * 14
    for gk, ga in ((0.9, 0.16), (0.45, 0.28)):
        gx, gy, gd = step_pos(kk + gk)
        pose = dict(x=gx, y=gy - bob, scale=0.95 * gd, facing=facing, walk=1.0, walk_phase=phase - gk * 0.25,
                    lean=0.25, eyes='determined', arm_l=(2.4, 0.4), arm_r=(0.2, 1.3), scarf_wind=1.0)
        with alpha_layer(c, ga):
            robot.draw_robot(c, U.mkpose(robot.RobotPose, **pose), T)
    pose = dict(x=x, y=y - bob, scale=s, facing=facing, walk=1.0, walk_phase=phase, lean=0.25,
                eyes='determined', arm_l=(2.5 + 0.3 * math.sin(phase * TAU), 0.4),
                arm_r=(0.3 + 0.3 * math.sin(phase * TAU), 1.2), mouth=0.0,
                blink=0.0, scarf_wind=1.0, squash=0.12 * math.cos(phase * TAU * 2))
    deng(c, pose, T, washes=[((x - 300, y - 400), (x + 100, y), '#ffb347', 0.18)])
    # comic clatter ticks at each foot contact (walk_phase multiples of 0.5)
    for n in range(int(phase * 2) - 2, int(phase * 2) + 1):
        if n < 0:
            continue
        age = tr - n / 2 / WALK_HURRY
        if 0 <= age < 0.3:
            px, py, pd = step_pos(k0 - n * STEPS_PER_FOOT)
            a = 1 - age / 0.3
            for j in range(4):
                ang = math.pi * 1.1 + j * 0.27
                r = 30 + 80 * age / 0.3
                c.drawLine(px + math.cos(ang) * r * 0.5, py - 6 + math.sin(ang) * r * 0.4,
                           px + math.cos(ang) * r, py - 6 + math.sin(ang) * r * 0.8, stroke('#ffe9b0', 3, 0.6 * a))
    f.grade.update(bloom=1.0, vignette=0.6, tint='#ffd9a0', tint_amt=0.08)


# ============================================================================
# S6 — the door bangs open, Deng shoots out onto the rocks
# ============================================================================
S6_HZ = 520.0
S6_H = 25.0
S6_LH = (560.0, 1010.0, 5.6)


def s6_sv():
    return U.SeaView(hz=S6_HZ, F=1000.0, cam_h=S6_H)


def shot_door(f, cu):
    c, t, T = f.canvas, f.t, f.T
    td = t - cu.door
    if td > 0:
        shake = 14 * math.exp(-td * 9)
    else:
        shake = 2.0 * math.sin(t * 95)          # door rattling just before the bang
    c.save()
    c.translate(shake * noise1(t * 30, 1), shake * noise1(t * 30, 2) * 0.5)
    rect = (-300, -300, W + 600, S6_HZ + 400)
    U.call(sky.sky, c, rect=rect, horizon_y=S6_HZ)
    U.safe(sky.milky_way, c, T, center=(1500, 120), angle=0.75, length=2800, width=380, alpha=0.7,
           horizon_y=S6_HZ)
    U.call(sky.stars, c, T, rect=rect, horizon_y=S6_HZ, seed=11)
    U.call(sea.ocean, c, T, horizon_y=S6_HZ, rect_x=(-300, W + 300), bottom=H + 300, calm=0.8,
           lights=[(1700, 700, '#ffc86b', 0.9, 860)])
    sv = s6_sv()
    pl = plankton('s6', s6_sv, n=700, seed=61)
    pl.draw(c, sv, T, 0.5 * (0.4 + 0.6 * pl.jit))
    with U.ground(c, sv):
        fx.bioluminescence(c, T, 60.0, 160.0, radius=260, persp=1.0, t_since=None, size=0.3, count=900,
                           intensity=0.7)
    U.call(sea.boat, c, 1700, 862, 0.7, rock=0.03 * math.sin(T * 1.3), lantern=0.9, t=T)
    lbx, lby, ls = S6_LH
    U.call(sea.island, c, 900, lby + 10, 2.6)
    U.call(sea.lighthouse, c, lbx, lby, ls, lamp=1.0, t=T, view_e=0.12)
    A = U.call(sea.lighthouse_anchors, lbx, lby, ls) if hasattr(sea, 'lighthouse_anchors') else None
    dx, dy = A['door'] if A else (lbx, lby - 30 * ls)
    dw, dh = A['door_size'] if A else (26 * ls, 44 * ls)
    door(c, t, td, dx, dy, dw, dh)
    # Deng bursting out, running to the boat (hurry cadence continues from run_down)
    phase = (t - cu.run) * WALK_HURRY
    if td > 0.02:
        u = td - 0.02
        k = clamp(u / (cu.boat - cu.door))
        x = dx + 1750 * ease_in(k) * 0.35 + 1750 * k * 0.65
        y = dy + 10 + 150 * ease_out(k)
        sc = lerp(0.55, 0.8, k)
        bob = abs(math.sin(phase * math.pi)) * 16 * sc
        for gk, ga in ((0.08, 0.14), (0.04, 0.26)):
            kg = clamp((u - gk) / (cu.boat - cu.door))
            gx = dx + 1750 * ease_in(kg) * 0.35 + 1750 * kg * 0.65
            gy = dy + 10 + 150 * ease_out(kg)
            with alpha_layer(c, ga):
                robot.draw_robot(c, U.mkpose(robot.RobotPose, x=gx, y=gy - bob, scale=lerp(0.55, 0.8, kg), facing=1,
                                             walk=1.0, run=1.0, walk_phase=phase - gk * 4, lean=0.3,
                                             eyes='determined', scarf_wind=1.0, ambient=NIGHT_AMB, shadow=0.0), T)
        pose = dict(x=x, y=y - bob, scale=sc, facing=1, walk=1.0, run=1.0, walk_phase=phase, lean=0.3,
                    eyes='determined', scarf_wind=1.0, blink=0.0, ambient=NIGHT_AMB,
                    rim=0.9 * clamp(1 - k * 1.5) + 0.3, rim_color='#ffcf7a', rim_angle=math.pi)
        deng(c, pose, T, washes=[((dx, dy - 100), (x + 80 * sc, y - 100), '#ffcf7a', 0.35 * clamp(1 - k * 1.5))])
        for j in range(8):   # dust puff at the threshold
            a = clamp(1 - td / 0.8) * 0.28
            px = dx + (j - 3.5) * 22 + td * 140 * (U.h01(j, 3) + 0.2)
            py = dy - 10 - td * 70 * U.h01(j, 4)
            c.drawCircle(px, py, 16 + 50 * td, fill('#c8c0b0', a, blur=10))
        fx.speed_lines(c, t, direction=(1.0, 0.1), intensity=clamp(1.2 - k), rect=(0, y - 360, W, 330), count=16,
                       alpha=0.35)
    U.call(sea.rocks_foreground, c, T, y=1010, seed=3)
    c.restore()
    f.grade.update(bloom=1.15, vignette=0.5)


def door(c, t, td, x, y, w, h):
    """The door in the lighthouse base (x, y = threshold centre) swinging open with a bang.
    Local prop: the library draws a closed door; we paint the opening over it."""
    opened = 0.0
    if td > 0:
        opened = clamp(ease_out_back(clamp(td / 0.1), 2.4))
        if td > 0.1:
            opened -= 0.14 * math.exp(-td * 7) * math.sin(td * 38)
    x0, y0 = x - w / 2, y - h
    r = w / 2
    if opened > 0:
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x0, y0, w, h + 1), r, r),
                    linear((x, y0), (x, y), [(0, '#ffcf7a', 1), (1, '#ffe7b0', 1)]))
        # light spilling out onto the rocks and a warm glow around the doorway
        sp = poly([(x0, y), (x0 + w, y), (x0 + w + 3.2 * w, y + 1.4 * h), (x0 - 1.2 * w, y + 1.4 * h)])
        p = linear((x, y), (x + 0.6 * w, y + 1.4 * h), [(0, '#ffcf7a', 0.55 * opened), (1, '#ffcf7a', 0)])
        p.setBlendMode(ADD)
        c.drawPath(sp, p)
        fx.glow(c, x, y - h * 0.45, h * 1.5, '#ffb347', 0.5 * opened, core=False)
    else:
        rim = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x0 - 1.5, y0 - 1.5, w + 3, h + 3), r + 1.5, r + 1.5)
        c.drawRRect(rim, stroke('#ffcf7a', 3, 0.85, blur=1.5))
    # the door leaf, hinged on the left, swinging out towards the camera and banging the wall
    pw = w * (1 - 1.3 * opened)
    if abs(pw) > 1.5:
        xa, xb = sorted((x0, x0 + pw))
        rr = min(r, abs(pw) / 2)
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(xa, y0 - (4 if pw < 0 else 0), xb, y), rr, r),
                    fill('#6a4630' if pw > 0 else '#8a5e3e'))
        c.drawRect(skia.Rect.MakeLTRB(xa, y0 + h * 0.35, xb, y0 + h * 0.4), fill('#2a2f45', 0.7))
        c.drawRect(skia.Rect.MakeLTRB(xa, y0 + h * 0.75, xb, y0 + h * 0.8), fill('#2a2f45', 0.7))
    # comic bang lines
    if 0 < td < 0.25:
        a = 1 - td / 0.25
        p = stroke('#fff0c8', 5, 0.8 * a)
        for j in range(7):
            ang = -math.pi + j * math.pi / 6
            r0 = h * (0.6 + 0.5 * td / 0.25)
            cx_, cy_ = x, y - h * 0.5
            c.drawLine(cx_ + math.cos(ang) * r0, cy_ + math.sin(ang) * r0 * 0.9,
                       cx_ + math.cos(ang) * (r0 + h * 0.25), cy_ + math.sin(ang) * (r0 + h * 0.25) * 0.9, p)


# ============================================================================
# S7 — rowing across the glowing sea (wide, low angle)
# ============================================================================
S7_HZ = 700.0
S7_F, S7_H = 1000.0, 8.0
BOAT_Y = 872.0
BOAT_S = 1.9                     # boat scale
DENG_S7 = 0.78
GLOW_WX, GLOW_WY = 1480.0, 792.0 # the faint glow under the water (world)


def s7_sv(cam=None):
    if cam is None:
        return U.SeaView(hz=S7_HZ, F=S7_F, cam_h=S7_H)
    Zb = S7_H * S7_F / (BOAT_Y - S7_HZ)
    return U.SeaView(hz=cam.to_screen(0, S7_HZ)[1], F=S7_F * cam.zoom, cam_h=S7_H,
                     cx=W / 2, cam_x=(cam.x - W / 2) * Zb / S7_F)


def boat_x(cu, t):
    """Boat position: steady glide + a surge on every drive (one stroke per STROKE s)."""
    tb = max(0.0, t - cu.boat)
    n = math.floor(tb / STROKE)
    u = tb / STROKE - n
    return 330 + 30 * tb + 62 * (n + ease_out(clamp(u / 0.45)))


_ARM = {}


def arm_geom(side, sit):
    """Measure shoulder position & reach of the library robot's arm (scale 1, facing +1, at the
    origin) with robot.anchors (or a recorded draw). Cached. Fallback IK only."""
    key = (side, round(sit, 2))
    if key not in _ARM:
        def anc(arm):
            pose = U.mkpose(robot.RobotPose, x=0, y=0, scale=1.0, facing=1, sit=sit, **{'arm_' + side: arm})
            if hasattr(robot, 'anchors'):
                return robot.anchors(pose, 0.0)
            rec = skia.PictureRecorder()
            cc = rec.beginRecording(skia.Rect.MakeXYWH(-1000, -1000, 2000, 2000))
            r = robot.draw_robot(cc, pose, 0.0)
            rec.finishRecordingAsPicture()
            return r
        h0, h9 = anc((0.0, 0.0))[side], anc((math.pi / 2, 0.0))[side]
        sx, sy = h0[0], h9[1]
        _ARM[key] = (sx, sy, max(40.0, h0[1] - sy))
    return _ARM[key]


def ik_arm(side, sit, rx, ry, s, target):
    """(shoulder_angle, elbow_bend) that puts the hand at `target` (world), elbow down."""
    sx, sy, L = arm_geom(side, sit)
    Sx, Sy = rx + sx * s, ry + sy * s
    dx, dy = target[0] - Sx, target[1] - Sy
    d = clamp(math.hypot(dx, dy), 0.3 * L * s, 0.985 * L * s)
    l1 = l2 = L * s / 2
    base = math.atan2(dx, dy)
    alpha = math.acos(clamp((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d), -1, 1))
    bend = math.pi - math.acos(clamp((l1 * l1 + l2 * l2 - d * d) / (2 * l1 * l2), -1, 1))
    return (base - alpha, bend)


OAR_WATER = 30.0   # waterline below the oarlock (boat units)


def oar_blade(u):
    """Blade tip relative to the oarlock (boat units). Reaches the water surface exactly at
    u = 0 (stroke start), sweeps back under water, lifts out, recovers forward in the air."""
    D, air = 120.0, -12.0
    if u < 0.45:                       # drive: in the water, sweeping back
        k = u / 0.45
        dx = D * math.cos(math.pi * ease_in_out(k))
        dy = OAR_WATER + 9 * math.sin(math.pi * min(1.0, k * 1.6)) * (1 - 0.3 * k)
    elif u < 0.55:                     # release: lift out
        k = (u - 0.45) / 0.1
        dx = -D
        dy = lerp(OAR_WATER, air, ease_out(k))
    elif u < 0.92:                     # recovery: forward in the air
        k = (u - 0.55) / 0.37
        dx = -D * math.cos(math.pi * ease_in_out(k))
        dy = air - 8 * math.sin(math.pi * k)
    else:                              # square & drop to the water surface
        k = (u - 0.92) / 0.08
        dx = D
        dy = lerp(air, OAR_WATER, ease_in(k))
    return dx, dy


def draw_oar(c, pivot, blade, hand, s, color='#8a6440', a=1.0, sub=0.0):
    """Oar shaft hand -> oarlock -> blade tip; the blade part below the water (y > sub) is dimmed."""
    px, py = pivot
    bx, by = blade
    hx, hy = hand
    c.drawLine(hx, hy, px, py, stroke(color, 6 * s, a))
    c.drawLine(px, py, bx, by, stroke(color, 6 * s, a))
    ang = math.atan2(by - py, bx - px)
    with at(c, bx, by, rot=ang):
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(-40 * s, -8 * s, 48 * s, 16 * s), 7 * s, 7 * s),
                    fill(color, a))
    if sub and by > sub:  # water covers the submerged part
        c.drawRect(skia.Rect.MakeLTRB(bx - 60 * s, sub, bx + 60 * s, by + 12 * s), fill('#0c1f45', 0.55))


def shot_boat(f, cu):
    c, t, T = f.canvas, f.t, f.T
    tb = t - cu.boat
    k = clamp(tb / max(0.1, f.dur - cu.boat))
    bxw = boat_x(cu, t)
    cam = Camera(x=900 + (bxw - 330) * 0.55, y=540 - 10 * ease_in_out(k), zoom=1.0 + 0.05 * ease_in_out(k))
    hz = cam.to_screen(0, S7_HZ)[1]
    sv = s7_sv(cam)
    bs = BOAT_S
    bob = 4 * math.sin(T * 1.7)
    rock = 0.022 * math.sin(T * 1.3 + 0.5)
    far = Camera(x=960 + (cam.x - 960) * 0.06, y=cam.y, zoom=1.0 + (cam.zoom - 1) * 0.2)
    with far.apply(c, 1.0):
        rect = (far.x - 1300, -500, 2600, S7_HZ + 600)
        U.call(sky.sky, c, rect=rect, horizon_y=S7_HZ)
        U.safe(sky.milky_way, c, T, center=(1050, 300), angle=-0.5, length=4200, width=560, alpha=0.95,
               horizon_y=S7_HZ, bend=0.05)
        U.call(sky.stars, c, T, rect=rect, horizon_y=S7_HZ, seed=7)
        lx0, ly0, ls0 = 190, S7_HZ + 5, 0.28
        U.call(sea.island, c, lx0, ly0 + 3, 0.28)
        lamp = U.call(sea.lighthouse, c, lx0, ly0, ls0, lamp=1.0, t=T, beam_angle=T * 0.9)
        lpx, lpy = lamp if lamp else (lx0, ly0 - 445 * ls0)
    lant_w = (bxw + 130 * bs, BOAT_Y - 120 * bs)       # stub geometry; replaced by sea.boat's return below
    lant_s = cam.to_screen(*lant_w)
    lamp_s = far.to_screen(lpx, lpy)
    U.call(sea.ocean, c, T, horizon_y=hz, rect_x=(-300, W + 300), bottom=H + 300, calm=0.6, cx=W / 2,
           lights=[(lant_s[0], lant_s[1], '#ffc86b', 1.0, cam.to_screen(0, BOAT_Y)[1]),
                   (lamp_s[0], lamp_s[1], GOLD, 0.45)])
    with far.apply(c, 1.0):
        U.call(sea.beam, c, lpx, lpy, T * 0.9, length=1500, width0=8, width1=220, intensity=0.5,
               horizon_y=S7_HZ, t=T)
    # plankton: the whole sea still glows; brighter around the underwater glow and in the wake
    gsx, gsy = cam.to_screen(GLOW_WX, GLOW_WY)
    Xg, Zg = sv.ground(gsx, gsy)
    bsx, bsy = cam.to_screen(bxw, BOAT_Y)
    Xb, Zb = sv.ground(bsx, bsy)
    n_now = math.floor(max(0.0, tb) / STROKE)
    with U.ground(c, sv):
        fx.bioluminescence(c, T, Xg, Zg, radius=26, persp=1.0, t_since=None, size=0.05, count=700,
                           intensity=0.9)
        fx.bioluminescence(c, T, Xg, Zg, radius=6, persp=1.0, t_since=None, size=0.03, count=160,
                           intensity=1.0, color=GOLD, seed=9)
        # oar entries: ripples + a little flash of plankton, world-anchored
        for n in range(max(0, n_now - 3), n_now + 1):
            te = cu.boat + n * STROKE
            age = t - te
            if age < 0:
                continue
            ex_s = cam.to_screen(boat_x(cu, te) + 130 * bs, BOAT_Y)
            Xe, Ze = sv.ground(ex_s[0], ex_s[1] + 8)
            fx.ripples(c, Xe, Ze - 0.3, age, persp=1.0, count=2, speed=1.4, life=2.5, width=0.07, intensity=0.7)
            fx.bioluminescence(c, T, Xe, Ze - 0.3, radius=1.6, persp=1.0, t_since=age, life=4.0, size=0.02,
                               count=60, rings=1)
    pl = plankton('s7', s7_sv, n=1400, seed=71, x_range=(-600, W + 600))
    dg = np.hypot(pl.X - Xg, pl.Z - Zg)
    br = 0.28 * (0.35 + 0.65 * pl.jit) + 1.0 * np.exp(-(dg / 12.0) ** 2)
    behind = (pl.X < Xb) & (np.abs(pl.Z - Zb) < 0.6 + (Xb - pl.X) * 0.12)
    br = br + np.where(behind, 1.0 * np.exp(-np.clip(Xb - pl.X, 0, None) / 10.0), 0.0)
    pl.draw(c, sv, T, br, size=0.9)
    underwater_glow(c, T, gsx, gsy, 1.0 + 0.25 * k)
    with cam.apply(c, 1.0):
        for n in range(max(0, n_now - 1), n_now + 1):   # entry splashes (vertical, screen space)
            te = cu.boat + n * STROKE
            age = t - te
            if 0 <= age < 1.0:
                fx.splash(c, boat_x(cu, te) + 130 * bs, BOAT_Y + 8, age, scale=0.2, glow_color=CYAN, persp=0.2,
                          intensity=0.8, height=0.6, life=0.9, mist=0.3)
        # the boat + Deng rowing
        u = (max(0.0, tb) / STROKE) % 1.0
        rx, ry, rs = bxw - 15 * bs, BOAT_Y + bob - 16 * bs, DENG_S7
        pivot = (bxw + 12 * bs, BOAT_Y + bob - OAR_WATER * bs)
        bdx, bdy = oar_blade(u)
        blade = (pivot[0] + bdx * bs, pivot[1] + bdy * bs)
        lin = 0.45
        handle = (pivot[0] - bdx * bs * lin, pivot[1] - bdy * bs * lin - 16 * bs)
        far_pivot = (pivot[0] + 6 * bs, pivot[1] - 5 * bs)
        far_blade = (far_pivot[0] + bdx * bs * 0.9, far_pivot[1] + bdy * bs * 0.9)
        far_handle = (far_pivot[0] - bdx * bs * lin, far_pivot[1] - bdy * bs * lin - 18 * bs)
        draw_oar(c, far_pivot, far_blade, far_handle, bs * 0.85, color='#5a4030', sub=BOAT_Y + bob)
        look_x = 0.6 + 0.3 * smoothstep(3.0, 5.5, tb)
        pose = dict(x=rx, y=ry, scale=rs, facing=1, sit=1.0, sit_dangle=0.0, lean=0.1 * math.sin(u * TAU) + 0.05,
                    hand_r_open=0.1, hand_l_open=0.1,
                    eyes='determined' if tb < 2.5 else ('wonder' if tb > 4.2 else 'open'),
                    blink=robot.auto_blink(T, seed=2), look=(look_x, 0.3 + 0.3 * smoothstep(3.5, 6.0, tb)),
                    mouth=0.0, smile=0.1, head_tilt=0.06 + 0.08 * smoothstep(3.5, 6.0, tb),
                    scarf_wind=0.55, ambient=NIGHT_AMB, rim=0.75, rim_color='#ffc86b',
                    rim_angle=math.atan2(lant_w[1] - (ry - 150 * rs), lant_w[0] - rx))
        if U.pose_has(robot.RobotPose, 'arm_r_override'):
            pose.update(arm_r_override=handle, arm_l_override=far_handle, arm_l_front=False)
        else:
            pose.update(arm_r=ik_arm('r', 1.0, rx, ry, rs, handle), arm_l=ik_arm('l', 1.0, rx, ry, rs, far_handle))
        washes = [((lant_w[0], lant_w[1]), (rx - 200, ry - 150), '#ffc86b', 0.25),
                  ((rx, ry + 200), (rx, ry - 250), CYAN, 0.12)]
        res = deng(c, pose, T, washes=washes)
        U.call(sea.boat, c, bxw, BOAT_Y + bob, bs, rock=rock, lantern=1.0, t=T)
        hand = res.get('r', handle) if res else handle
        draw_oar(c, pivot, blade, hand, bs, sub=BOAT_Y + bob)
        if 0.47 < u < 0.9:
            fx.water_drips(c, T, blade[0], blade[1] + 6 * bs, spread=10 * bs, rate=6,
                           fall=max(10, BOAT_Y + bob - blade[1]), size=0.7, glint=CYAN, spots=2)
    f.grade.update(bloom=1.25, vignette=0.55, tint='#bfe8ff', tint_amt=0.04)


def flat_glow(c, x, y, r, color, a, sy=0.22):
    """Additive radial glow squashed vertically (a light pool on / under the water)."""
    if a <= 0.003:
        return
    c.save()
    c.translate(x, y)
    c.scale(1.0, sy)
    soft_glow(c, 0, 0, r, color, a)
    c.restore()


def underwater_glow(c, T, x, y, k=1.0):
    """The fallen star under the surface: soft gold glow, pulsing, caustic flickers above it."""
    pl_ = (0.8 + 0.2 * math.sin(T * 2.1) + 0.06 * noise1(T * 3, 7)) * k
    flat_glow(c, x, y + 6, 560, '#ffb347', 0.28 * pl_, sy=0.2)
    flat_glow(c, x, y + 6, 260, GOLD, 0.55 * pl_, sy=0.22)
    flat_glow(c, x, y + 8, 90, '#fff3cf', 0.75 * pl_, sy=0.3)
    p = fill('#fff0b0', 1.0, blend=ADD)
    for i in range(24):
        a = U.h01(i, 91) * TAU
        rr = (0.2 + 0.9 * U.h01(i, 92)) * 230
        px, py = x + math.cos(a) * rr, y + 6 + math.sin(a) * rr * 0.2
        tw = max(0.0, math.sin(T * (2 + 3 * U.h01(i, 93)) + i)) ** 3
        p.setAlphaf(clamp(0.8 * tw * pl_))
        c.drawOval(skia.Rect.MakeXYWH(px - 11, py - 2, 22, 4), p)
    fx.sparkles(c, T, x, y - 30, radius=140, radius_y=40, count=8, color='#ffe9a0', intensity=0.6 * k, rise=60,
                size=0.5, seed=33)


# ============================================================================
def render(f):
    cu = Cues(f)
    t = f.t
    if t < cu.s2:
        shot_sky(f, cu)
    elif t < cu.s3:
        shot_medium(f, cu)
    elif t < cu.s4:
        shot_impact(f, cu)
    elif t < cu.run:
        shot_exterior(f, cu)
    elif t < cu.s6:
        shot_stairs(f, cu)
    elif t < cu.boat:
        shot_door(f, cu)
    else:
        shot_boat(f, cu)
