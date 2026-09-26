"""s04 · 相遇 The meeting (29 s) — the tender heart of the film.

Shots (all timing from cues / line times, local seconds):
  A  0 → G01_end+0.3   Split-level shot at the water line beside the boat: a faint glow
                       under the surface; at `lift` Deng's mitten hands plunge in and lift
                       out a small dim star, dripping, curled, shivering (G01).
  B  → D05_end         Medium close-up in the boat: Deng hugs her to his chest porthole;
                       the heart warms her, her glow returns a little (D05, head tilted).
  C  → row_back        Closer two-shot: she opens big shiny eyes, sniffles, embarrassed
                       smile, gestures (G02).  `look_sky`: both look up, the camera tilts up
                       to the vast sky and back down (G03, she flickers on "熄灭").
                       `determined`: Deng's eyes turn determined, fist raised (D06); she
                       beams and hops onto his head.
  D  row_back → end    Wide: the little boat with two lights rowing back to the
                       lighthouse across the softly glowing sea.
"""
from __future__ import annotations

import dataclasses
import math

import skia

from engine.core import (ADD, TAU, Camera, H, W, at, clamp, col, ease_in, ease_in_out, ease_out,
                         ease_out_back, fill, keyframes, lerp, linear, mix, noise1, pulse, radial,
                         smoothstep, soft_glow, spring, stroke)
from lib import fx, robot, sea, sky, star
from scenes import _s04_kit as K

RobotPose = robot.RobotPose
StarPose = star.StarPose


# ----------------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------------
class Tm:
    def __init__(self, f):
        self.lift = f.cue('lift')
        self.g1s, self.g1e = f.line('G01')
        self.d5s, self.d5e = f.line('D05')
        self.g2s, self.g2e = f.line('G02')
        self.look = f.cue('look_sky')
        self.g3s, self.g3e = f.line('G03')
        self.det = f.cue('determined')
        self.d6s, self.d6e = f.line('D06')
        self.row = f.cue('row_back')
        self.cut_warm = self.g1e + 0.3
        self.cut_close = self.d5e + 0.02
        # hand/star choreography in shot A
        self.dip = self.lift              # hands start down
        self.enter = self.lift + 0.34     # hands break the surface
        self.cup = self.lift + 0.62       # hands under her
        self.rise = self.lift + 0.8       # lifting starts
        self.out = self.lift + 1.05       # (refined in shot A) she breaks the surface
        self.held = self.lift + 1.7       # at face height
        # the flicker on "熄灭" near the end of G03
        self.xm0 = self.g3e - 0.95
        self.xm1 = self.g3e + 0.25
        # fist pump on "想办法！"
        self.fist_ant = self.d6s + 1.15
        self.fist_up = self.d6s + 1.55
        self.beam = self.d6s + 1.75
        self.hop = self.d6e - 0.55


def auto_blink(T, seed=0, every=3.7):
    return robot.auto_blink(T, seed=seed, every=every)


def star_blink(T, seed=3, every=3.1):
    ph = (T + seed * 0.91) % every
    return max(0.0, 1 - abs(ph - 0.07) / 0.07) if ph < 0.14 else 0.0


def shiver(t, amp):
    """Small fast trembling offsets (dx, dy, rot)."""
    if amp <= 0:
        return 0.0, 0.0, 0.0
    return (amp * 2.6 * math.sin(t * 47.0) + amp * 1.2 * noise1(t * 20, 3),
            amp * 1.4 * math.sin(t * 39.0 + 1.3),
            amp * 0.035 * math.sin(t * 43.0 + 0.4))


def draw_star(c, pose, t):
    return star.draw_star(c, pose, t)


def star_pose(**kw):
    return K.make_pose(StarPose, **kw)


def robot_pose(**kw):
    return K.make_pose(RobotPose, **kw)


def sky_backdrop(c, cam, t, T, horizon_y, sky_depth=0.2, mw_center=(960, -120), mw_alpha=0.9,
                 star_density=1.0):
    """Sky + stars + Milky Way on a parallax layer."""
    with cam.apply(c, sky_depth):
        sky.sky(c, rect=(-3000, -4000, 7920, 8000), horizon_y=horizon_y)
        K.call_kw(sky.milky_way, c, T, center=mw_center, angle=-0.42, length=4200, width=640, alpha=mw_alpha,
                  horizon_y=horizon_y, bend=0.04)
        K.call_kw(sky.stars, c, T, rect=(-3000, -4000, 7920, 4000 + horizon_y), density=star_density,
                  horizon_y=horizon_y)


# ----------------------------------------------------------------------------
# SHOT A — rescue at the water line
# ----------------------------------------------------------------------------
YS_A = 600.0                                   # water surface (world) in shot A
DENG_A = dict(x=560.0, y=612.0, scale=1.6)     # seat point in the boat
BOAT_A = dict(x=420.0, y=624.0, s=2.4)
STAR_A_SCALE = 1.25
UNDER_A = (862.0, YS_A + 168.0)                # where the dim star floats under water
CUP_A = (848.0, YS_A + 64.0)                   # where his hands scoop her
HOLD_A = (772.0, 396.0)                        # held up in front of his face


def a_lean(t, tm):
    """Deng leans over the gunwale; anticipation, reach, then straightens as he lifts her."""
    return keyframes(t, [(-10, 0.56), (tm.dip - 0.3, 0.56), (tm.dip, 0.5), (tm.cup, 1.0), (tm.rise, 1.0),
                         (tm.held, 0.24), (tm.held + 0.6, 0.2), (tm.cut_warm + 1, 0.18)], ease_in_out)


def a_deng_pose(f, tm, t):
    T = f.T
    idl = robot.idle(T, 1)
    lean = a_lean(t, tm) + idl['lean']
    if t < tm.dip:
        eyes, eyes2, mix_ = 'worried', 'surprised', 0.0
    elif t < tm.held - 0.2:
        eyes, eyes2, mix_ = 'surprised', 'worried', smoothstep(tm.rise, tm.held, t)
    else:
        eyes, eyes2, mix_ = 'worried', 'sad', 0.35 * K.bump(t, tm.g1s, tm.g1e, 0.4, 0.5)
    lk_down = (0.55, 0.95)
    lk_her = (0.75, 0.2)
    k = smoothstep(tm.rise, tm.held, t)
    look = (lerp(lk_down[0], lk_her[0], k), lerp(lk_down[1], lk_her[1], k))
    tilt = 0.14 * smoothstep(tm.held, tm.held + 0.9, t)
    rim_k = smoothstep(tm.out - 0.2, tm.held, t)
    return robot_pose(x=DENG_A['x'], y=DENG_A['y'], scale=DENG_A['scale'], facing=1.0, sit=1.0,
                      sit_dangle=0.0, lean=lean, head_tilt=tilt + idl['head_tilt'], head_dy=idl['head_dy'],
                      eyes=eyes, eyes2=eyes2, eyes_mix=mix_, blink=auto_blink(T, 2), look=look,
                      mouth=f.mouth('deng'), smile=lerp(0.05, -0.25, k), scarf_wind=0.35, scarf_dir=-1.0,
                      hand_l_open=0.9, hand_r_open=0.9, rim=0.55,
                      rim_color=mix('#8fd8ff', '#ffe2a0', rim_k), rim_angle=lerp(1.2, 0.2, rim_k),
                      shadow=0.0)


def _a_star_raw(t, tm):
    ux, uy = UNDER_A
    bob = (4 * math.sin(t * 1.3), 5 * math.sin(t * 0.9 + 1))
    if t <= tm.dip:
        return (ux + bob[0], uy + bob[1] - 22 * smoothstep(-1.0, tm.dip, t))
    if t <= tm.cup:
        k = ease_in_out((t - tm.dip) / (tm.cup - tm.dip))
        k2 = 1 - smoothstep(tm.dip, tm.cup, t)
        return (lerp(ux, CUP_A[0], k) + bob[0] * k2, lerp(uy - 22, CUP_A[1], k) + bob[1] * k2)
    if t <= tm.rise:
        return CUP_A[0], CUP_A[1] + 4 * math.sin((t - tm.cup) / (tm.rise - tm.cup) * math.pi)
    u = (t - tm.rise) / (tm.held - tm.rise)
    if u < 1:
        e = ease_in_out(u)
        x = lerp(CUP_A[0], HOLD_A[0], e) + math.sin(e * math.pi) * 36
        y = lerp(CUP_A[1], HOLD_A[1], e)
        return (x, y)
    s_ = t - tm.held
    ov = math.exp(-5 * s_) * math.sin(s_ * 9) * 10
    return (HOLD_A[0] + 2 * math.sin(t * 1.1), HOLD_A[1] + ov + 3 * math.sin(t * 1.7))


def a_star_pos(t, tm):
    """Star centre in world coords during shot A (pure function of t)."""
    return _a_star_raw(t, tm)


def a_out_time(tm):
    """When she breaks the surface (solve y(t) = YS_A on the rise)."""
    lo, hi = tm.rise, tm.held
    for _ in range(30):
        mid = (lo + hi) / 2
        if _a_star_raw(mid, tm)[1] > YS_A:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def shot_rescue(f, tm):
    c, t, T = f.canvas, f.t, f.T
    tm.out = a_out_time(tm)
    # camera: close on the water and the glow beneath, then ease up to Deng's face and the star
    cx = keyframes(t, [(-5, 846), (tm.dip, 850), (tm.rise, 840), (tm.held + 0.4, 706), (tm.cut_warm, 700)])
    cy = keyframes(t, [(-5, 616), (tm.dip, 614), (tm.rise, 590), (tm.held + 0.4, 412), (tm.cut_warm, 404)])
    z = keyframes(t, [(-5, 1.3), (tm.dip, 1.3), (tm.rise, 1.28), (tm.held + 0.4, 1.42), (tm.cut_warm, 1.48)])
    cam = Camera(x=cx, y=cy, zoom=z, t=T)

    # --- above water: sky, far horizon with the tiny lighthouse --------------
    sky_backdrop(c, cam, t, T, horizon_y=YS_A + 2, sky_depth=0.25, mw_center=(700, 0), mw_alpha=0.7)
    with cam.apply(c, 0.35):
        lhx, lhy = 40, YS_A + 1          # far behind the boat (continuity with s03): only its beam shows
        K.call_kw(sea.island, c, lhx, lhy + 4, 0.13)
        lx, ly = K.call_kw(sea.lighthouse, c, lhx, lhy, 0.13, lamp=1.0, t=T, _default=(lhx, lhy - 58))
        K.call_kw(sea.beam, c, lx, ly, T * 0.9, length=900, width0=6, width1=160, intensity=0.5)

    x0, x1, bottom = -800, 3000, 2400
    sp = a_star_pos(t, tm)
    sr = star.RADIUS * STAR_A_SCALE
    submerged = sp[1] > YS_A + 8
    disturb = [(CUP_A[0] - 10, t - tm.enter, 0.9), (CUP_A[0] - 5, t - tm.out, 1.3)]
    glow = 0.2 + 0.05 * math.sin(t * 2.3)
    with cam.apply(c, 1.0):
        # underwater backdrop lit by the dim star
        K.underwater_back(c, t, YS_A, x0, x1, bottom, glow_xy=sp if submerged else None, glow_amt=1.0)

        # --- Deng -------------------------------------------------------------
        pose = a_deng_pose(f, tm, t)
        rest_l, rest_r = (800.0, 552.0), (738.0, 556.0)          # mittens on the gunwale
        cup_l = (sp[0] + sr * 0.62, sp[1] + sr * 0.5)            # far hand: her right side, below
        cup_r = (sp[0] - sr * 0.66, sp[1] + sr * 0.45)           # near hand: her left side, below
        if t < tm.dip:
            lift = 12 * smoothstep(tm.dip - 0.3, tm.dip, t)      # anticipation: hands lift off the rim
            tl_, tr_ = (rest_l[0], rest_l[1] - lift), (rest_r[0], rest_r[1] - lift)
        else:
            k = ease_in_out((t - tm.dip) / (tm.cup - tm.dip))
            tl_ = (lerp(rest_l[0], cup_l[0], k), lerp(rest_l[1] - 12, cup_l[1], k))
            tr_ = (lerp(rest_r[0], cup_r[0], k), lerp(rest_r[1] - 12, cup_r[1], k))
        pose, _ = K.pose_with_hands(pose, T, tl_, tr_)
        drawn = robot.draw_robot(c, pose, T)

        # --- the star (drawn after the robot: sits in his cupped mittens) --------
        wet = smoothstep(tm.out - 0.05, tm.out + 0.05, t)
        shiv_amp = smoothstep(tm.held - 0.4, tm.held + 0.2, t) * (0.8 + 0.3 * pulse(t, tm.g1s, tm.g1e, 0.2))
        crying = K.bump(t, tm.g1s - 0.1, tm.g1e, 0.2, 0.4)
        sob = (0.5 + 0.5 * math.sin((t - tm.g1s) * 7.0)) * crying
        spose = star_pose(x=sp[0], y=sp[1] + 2 * sob, scale=STAR_A_SCALE, rot=-0.1 + 0.03 * sob,
                          glow=glow, flicker=0.35, squash=0.1 + 0.05 * sob,
                          arm_l=-0.45, arm_r=-0.4, leg_l=-0.45, leg_r=-0.4, hug=0.35,
                          eyes='closed', mouth=f.mouth('guang'), smile=-0.4 - 0.4 * crying,
                          blush=0.25, tears=0.9 * crying, wet=wet, shiver=min(1.0, shiv_amp),
                          halo=0.45, sparkle=0.15, brow=-0.8, rim=0.5, rim_color='#9fc4ff')
        draw_star(c, spose, T)
        if t > tm.cup:     # the near mitten's palm in front of her
            K.overdraw_hand(c, pose, T, drawn['r'], DENG_A['scale'])
        if t < tm.out + 0.2:
            K.bubbles(c, T, sp[0], sp[1] - sr * 0.3, YS_A, count=9, spread=sr, seed=12,
                      alpha=0.6 * (1 - smoothstep(tm.out - 0.2, tm.out + 0.2, t)))

        # --- boat hull in front of Deng's legs --------------------------------
        K.call_kw(sea.boat, c, BOAT_A['x'], BOAT_A['y'], BOAT_A['s'], rock=0.015 * math.sin(t * 1.1),
                  lantern=0.0, t=T)

        # --- water tint over everything submerged, surface line, glints ------
        K.underwater_front(c, t, YS_A, x0, x1, bottom, disturb=disturb,
                           glow_xy=sp if submerged else None, glow_amt=1.0)
        K.above_water_sheen(c, t, YS_A, x0, x1, lights=[(sp[0], sp[1], '#ffd86b', 0.25 if submerged else 0.5)])

        # splashes where the hands enter and where she comes out
        K.splash_burst(c, t - tm.enter, CUP_A[0] - 10, YS_A, 0.8, seed=4)
        K.splash_burst(c, t - tm.out, CUP_A[0], YS_A, 1.1, seed=9, up=1.2)
        K.call_kw(fx.ripples, c, CUP_A[0] - 10, YS_A + 3, t - tm.enter, color='#bfe0ff', intensity=0.6,
                  persp=0.1, count=3, speed=150, life=2.5)
        K.call_kw(fx.ripples, c, CUP_A[0], YS_A + 3, t - tm.out, color='#bfe0ff', intensity=0.7,
                  persp=0.1, count=4, speed=140, life=3.0)
        # drips from her and his mittens once she is out, falling back into the sea
        if t > tm.out:
            def src(te, tm=tm):
                p = a_star_pos(te, tm)
                return (p[0], p[1] + sr * 0.55)
            K.drips(c, t, src, tm.out, tm.cut_warm + 1.0, YS_A, width=sr * 1.5, rate=11.0, seed=7,
                    glow_col='#ffe7a0')
            def src_h(te, tm=tm):
                p = a_star_pos(te, tm)
                return (p[0] + sr * 0.62, p[1] + sr * 0.75)
            K.drips(c, t, src_h, tm.out, tm.cut_warm, YS_A, width=24, rate=4.0, seed=8)
        # her faint light on Deng's face once she is out
        if not submerged:
            K.call_kw(fx.glow, c, sp[0], sp[1], sr * 3.0, color='#ffd86b', intensity=0.12, core=False)

    f.grade.update(sat=0.92, vignette=0.55, tint='#b8ccff', tint_amt=0.06, bloom=1.05)


# ----------------------------------------------------------------------------
# SHOTS B / C — in the boat, close
# ----------------------------------------------------------------------------
HZ_BC = 470.0
BC_SKY_DEPTH = 0.3
DENG_BC = dict(x=780.0, y=975.0, scale=2.5)     # seat point
BOAT_BC = dict(x=830.0, y=992.0, s=3.4)
STAR_BC_SCALE = 1.9


def bc_background(c, cam, t, T, blur=0.0, shoot=None):
    """Sky, sea and the softly glowing plankton behind the close shots.
    blur: background defocus (design units) for an intimate shallow depth of field."""
    if blur > 0.3:
        p = skia.Paint()
        p.setImageFilter(skia.ImageFilters.Blur(blur, blur, skia.TileMode.kClamp))
        c.saveLayer(None, p)
    sky_backdrop(c, cam, t, T, horizon_y=HZ_BC, sky_depth=BC_SKY_DEPTH, mw_center=(1000, -450), mw_alpha=0.8)
    with cam.apply(c, BC_SKY_DEPTH):
        # a tiny shooting star while we look up at the sky
        if shoot is not None:
            u = (t - shoot) / 0.7
            if 0 < u < 1:
                hx, hy = 700 + 900 * u, -700 + 260 * u
                K.call_kw(sky.shooting_star_sky_streak, c, hx, hy, math.atan2(260, 900), 60 + 260 * math.sin(u * math.pi),
                          0.9 * math.sin(u * math.pi), width=2.5)
        K.call_kw(sea.ocean, c, T, horizon_y=HZ_BC, rect_x=(-3000, 5000), bottom=3000)
        # the sea still glows softly from the fallen star (bioluminescence)
        for i, (bx, by, rr) in enumerate(((300, 640, 380), (1500, 590, 460), (2100, 720, 520), (-300, 760, 480))):
            K.call_kw(fx.bioluminescence, c, T, bx, by, radius=rr, intensity=0.55, persp=0.18, seed=20 + i)
    if blur > 0.3:
        c.restore()


def bc_state(f, tm):
    """Choreography of shots B and C as a pure function of time."""
    t, T = f.t, f.T
    st = {}
    # ---- Deng body ---------------------------------------------------------
    idl = robot.idle(T, 4)
    rock = 0.03 * math.sin((t - tm.cut_warm) * 2.1) * K.bump(t, tm.cut_warm + 0.6, tm.d5s + 2.4, 0.8, 1.2)
    look_up = K.bump(t, tm.look, tm.xm0 - 0.15, 0.45, 0.45)            # both look at the sky
    glance_sky = K.bump(t, tm.d5s + 3.35, tm.d5s + 3.95, 0.25, 0.35)    # "从天上"
    follow_pt = K.bump(t, tm.g2s + 1.25, tm.g2s + 2.2, 0.25, 0.3)       # follows her pointing
    up_head = smoothstep(tm.hop + 0.2, tm.hop + 0.7, t)                 # she's on his head
    nod = K.hop(t, tm.d5s + 1.75, 0.45) + K.hop(t, tm.det + 0.45, 0.4)
    lean = 0.03 + rock + idl['lean'] - 0.07 * look_up + 0.05 * K.bump(t, tm.det, tm.d6e, 0.4, 0.5)
    head_tilt = idl['head_tilt']
    head_tilt += 0.13 * K.bump(t, tm.cut_warm + 0.3, tm.d5s + 3.2, 0.7, 0.5)
    head_tilt += 0.2 * K.bump(t, tm.d5s + 3.9, tm.d5e + 0.8, 0.35, 0.6)   # curious tilt on the question
    head_tilt += 0.1 * K.bump(t, tm.cut_close + 0.4, tm.g2e, 0.5, 0.5)
    head_tilt -= 0.2 * look_up + 0.12 * up_head
    head_tilt += 0.07 * nod
    head_dy = idl['head_dy'] + 6 * nod
    expr = [(-99, 'worried'), (tm.d5s - 0.3, 'open'), (tm.cut_close + 0.45, 'happy'), (tm.g2s + 0.6, 'open'),
            (tm.g2s + 2.7, 'happy'), (tm.look + 0.05, 'wonder'), (tm.xm0, 'worried'),
            (tm.det + 0.1, 'determined'), (tm.beam + 0.55, 'happy')]
    eyes, eyes2, emix = K.expr_at(t, expr, fade=0.22)
    # look direction: at her most of the time
    k_c = smoothstep(tm.cut_close + 0.3, tm.cut_close + 1.4, t)
    look = (lerp(0.5, 0.8, k_c), lerp(0.7, 0.45, k_c))
    for amt, tgt in ((glance_sky, (0.3, -0.9)), (follow_pt, (0.85, -0.85)), (look_up, (0.25, -1.0)),
                     (up_head, (0.2, -1.0))):
        look = (lerp(look[0], tgt[0], amt), lerp(look[1], tgt[1], amt))
    smile = 0.45 + 0.1 * K.bump(t, tm.d5s + 1.6, tm.d5s + 2.6, 0.2, 0.4)
    smile -= 0.6 * K.bump(t, tm.xm0 - 0.1, tm.det + 0.1, 0.3, 0.3)
    smile += 0.35 * smoothstep(tm.beam, tm.beam + 0.3, t)
    shake = 0.5 * K.bump(t, tm.fist_up + 0.05, tm.fist_up + 0.4, 0.05, 0.2)
    blush = 0.5 * K.bump(t, tm.g2s + 2.7, tm.g2e + 0.4, 0.4, 0.6) + 0.4 * K.bump(t, tm.beam, tm.row, 0.4, 0.5)
    fist = K.bump(t, tm.fist_ant - 0.2, tm.d6e + 0.1, 0.15, 0.4)
    st['deng'] = dict(x=DENG_BC['x'], y=DENG_BC['y'] - 5 * K.hop(t, tm.fist_up, 0.35), scale=DENG_BC['scale'],
                      facing=1.0, sit=1.0, sit_dangle=0.0, lean=lean, head_tilt=head_tilt, head_dy=head_dy,
                      eyes=eyes, eyes2=eyes2, eyes_mix=emix,
                      blink=auto_blink(T, 5) if eyes not in ('happy',) else 0.0, look=look,
                      mouth=f.mouth('deng'), smile=smile, scarf_wind=0.45, scarf_dir=-1.0, shake=shake,
                      blush=blush, hand_l_open=0.8, hand_r_open=lerp(0.8, 0.0, fist), heart=1.0,
                      rim=0.45, shadow=0.0, antenna_flash=0.8 * K.bump(t, tm.fist_up, tm.fist_up + 0.3, 0.05, 0.3))

    # ---- Guang ---------------------------------------------------------------
    warm = smoothstep(tm.d5s + 0.2, tm.d5e - 0.4, t)                 # heart warming her
    glow = lerp(0.25, 0.6, warm)
    # flicker + glow dip on "熄灭"
    xm = K.bump(t, tm.xm0, tm.xm1, 0.15, 0.6)
    glow -= 0.28 * xm * (0.6 + 0.4 * abs(math.sin(T * 23.0)) * (1 if noise1(T * 9, 2) > -0.2 else 0.2))
    # beam: glow blooms up
    beamk = smoothstep(tm.beam, tm.beam + 0.45, t)
    glow += beamk * (0.55 + 0.12 * math.exp(-3 * max(0, t - tm.beam - 0.45)) + 0.04 * math.sin(T * 3))
    flicker = 0.3 * (1 - warm) + 0.9 * xm
    shiv = (1 - smoothstep(tm.d5s + 0.8, tm.d5s + 3.6, t)) * 0.8
    shiv += 0.25 * xm
    # eyes: closed until she opens them after D05
    open_t = tm.cut_close + 0.35
    if t < open_t:
        eyes, blink = 'closed', 0.0
    else:
        eyes = 'open'
        # flutter open: two quick blinks
        b1 = K.bump(t, open_t, open_t + 0.08, 0.02, 0.08)
        b2 = K.bump(t, open_t + 0.32, open_t + 0.36, 0.04, 0.06)
        blink = max(1 - smoothstep(open_t, open_t + 0.14, t), b1 * 0.0, b2)
        blink = max(blink, star_blink(T))
    cover = K.bump(t, tm.g2s + 2.55, tm.g2e + 0.1, 0.3, 0.45)            # covers face
    point = K.bump(t, tm.g2s + 1.05, tm.g2s + 2.3, 0.3, 0.3)              # points at sky
    if tm.g2s + 2.6 < t < tm.g2e + 0.3:
        eyes = 'happy'   # embarrassed squint behind her arms
    if tm.look + 0.1 < t < tm.xm0:
        eyes = 'open'
    if tm.xm0 <= t < tm.det:
        eyes = 'sad'
    if t >= tm.beam:
        eyes = 'happy'
    if tm.d6s - 0.4 < t < tm.beam:
        eyes = 'open'
    # look: up at Deng's face / sky
    g_look = (-0.55, -0.75)
    g_look = (lerp(g_look[0], 0.35, point), lerp(g_look[1], -1.0, point))
    g_look = (lerp(g_look[0], 0.0, look_up), lerp(g_look[1], -1.0, look_up))
    smile_g = lerp(-0.5, 0.1, warm)
    smile_g += 0.5 * smoothstep(tm.g2s - 0.3, tm.g2s + 0.2, t)     # small embarrassed smile
    smile_g -= 0.7 * K.bump(t, tm.xm0 - 0.3, tm.det + 0.4, 0.3, 0.4)
    smile_g += 0.6 * beamk
    # sniffle: two tiny squash pulses just after she opens her eyes
    sn = K.hop(t, open_t + 0.45, 0.18) + K.hop(t, open_t + 0.75, 0.2)
    squash = 0.12 * (1 - warm) - 0.08 * sn + 0.1 * K.hop(t, tm.beam, 0.3) - 0.08 * K.hop(t, tm.beam + 0.3, 0.3)
    # arms: curled in -> relaxed -> gestures
    arm_l = lerp(-0.45, 0.0, warm)
    arm_r = lerp(-0.4, 0.0, warm)
    arm_r += 1.45 * point + 0.12 * math.sin((t - tm.g2s) * 9) * point   # little jabs at the sky
    arm_l += 1.1 * beamk * (1 - smoothstep(tm.hop, tm.hop + 0.3, t)) + 0.2 * math.sin(T * 5) * beamk
    arm_r += 1.1 * beamk * (1 - smoothstep(tm.hop, tm.hop + 0.3, t)) - 0.2 * math.sin(T * 5) * beamk
    blush = lerp(0.3, 0.7, warm) + 0.35 * cover
    tears = 0.8 * (1 - smoothstep(tm.d5s, tm.d5s + 2.5, t)) + 0.4 * K.bump(t, tm.xm0, tm.det, 0.3, 0.5)
    rot = -0.05 * (1 - warm) + 0.06 * point - 0.12 * cover + 0.05 * math.sin(T * 1.3) * warm
    st['star'] = dict(glow=glow, flicker=flicker, shiver=shiv, eyes=eyes, blink=blink, look=g_look,
                      smile=smile_g, squash=squash, arm_l=arm_l, arm_r=arm_r, blush=blush, tears=tears,
                      rot=rot, mouth=f.mouth('guang'), wet=1 - smoothstep(tm.d5s, tm.d5e, t),
                      curl=1 - warm, sniff=sn, cover=cover,
                      leg_l=lerp(-0.45, 0.1, warm), leg_r=lerp(-0.4, 0.05, warm),
                      hug=0.35 * (1 - warm))
    st['warm'] = warm
    st['xm'] = xm
    st['beamk'] = beamk
    st['look_up'] = look_up
    return st


_REF = {}


def ref_rig(x, y, scale, facing=1.0, sit=1.0):
    """Rig of Deng's neutral pose (no idle motion) — stable anchors for camera framing."""
    key = (x, y, scale, facing, sit)
    if key not in _REF:
        _REF[key] = K.robot_rig(robot_pose(x=x, y=y, scale=scale, facing=facing, sit=sit), 0.0)
    return _REF[key]


def bc_star_pos(t, tm, heart, head, s):
    """Guang's centre in shots B/C relative to Deng's (live) heart & head."""
    pre = (heart[0] + 92 * s, heart[1] - 42 * s)          # continuity: held out in front (shot A)
    hug = (heart[0] + 7 * s, heart[1] - 5 * s)             # cuddled against the porthole
    front = (heart[0] + 50 * s, heart[1] - 30 * s)         # held forward so they can look at each other
    k1 = ease_in_out((t - tm.cut_warm - 0.05) / 0.9)
    x, y = lerp(pre[0], hug[0], k1), lerp(pre[1], hug[1], k1)
    k2 = ease_in_out((t - tm.cut_close - 0.5) / 1.0)
    x, y = lerp(x, front[0], k2), lerp(y, front[1], k2) - 10 * s * math.sin(k2 * math.pi)
    return x, y


def shot_close(f, tm):
    """Shots B (warmth) and C (Guang / sky / promise) share one set-up."""
    c, t, T = f.canvas, f.t, f.T
    st = bc_state(f, tm)
    s = DENG_BC['scale']
    sr = star.RADIUS * STAR_BC_SCALE
    ref = ref_rig(DENG_BC['x'], DENG_BC['y'], s)
    rh, rt = ref['head'], ref['heart']

    # ---- camera (framed on the neutral rig so idle motion never jitters it) ----------
    if t < tm.cut_close:
        u = ease_in_out((t - tm.cut_warm) / max(0.1, tm.cut_close - tm.cut_warm))
        cx0, cy0 = rh[0] + 40 * s, rh[1] + 60 * s
        cam = Camera(x=cx0 - 5 * u, y=cy0 + 4 * u, zoom=lerp(1.2, 1.3, u), t=T)
    else:
        front_ref = (rt[0] + 50 * s, rt[1] - 30 * s)
        mid = ((rh[0] + front_ref[0]) / 2, (rh[1] + front_ref[1]) / 2)
        base = (mid[0] - 10, mid[1] + 50, 1.46)
        base2 = (mid[0], mid[1] + 10, 1.5)
        after = (mid[0] - 5, mid[1] - 22, 1.2)
        high = (mid[0] + 60, -2350, 0.9)
        up0, up1 = tm.look + 0.3, tm.g3s + 1.75
        dn0, dn1 = tm.g3s + 1.95, tm.xm0 + 0.3
        if t < up0:
            k = ease_in_out((t - tm.cut_close - 0.3) / 1.3)
            cx, cy, z = lerp(base[0], base2[0], k), lerp(base[1], base2[1], k), lerp(base[2], base2[2], k)
            z += 0.025 * smoothstep(tm.cut_close + 1.5, up0, t)
        elif t < up1:
            k = ease_in_out((t - up0) / (up1 - up0))
            cx, cy, z = lerp(base2[0], high[0], k), lerp(base2[1], high[1], k), lerp(base2[2] + 0.025, high[2], k)
        elif t < dn0:
            k = (t - up1) / (dn0 - up1)
            cx, cy, z = high[0], high[1] - 30 * k, high[2] - 0.01 * k
        elif t < dn1:
            k = ease_in_out((t - dn0) / (dn1 - dn0))
            cx, cy, z = lerp(high[0], after[0], k), lerp(high[1] - 30, after[1], k), lerp(high[2] - 0.01, after[2], k)
        else:
            k = (t - dn1) / max(0.1, tm.row - dn1)
            cx, cy, z = after[0], after[1] - 15 * k, after[2] + 0.03 * k
        cam = Camera(x=cx, y=cy, zoom=z, t=T)

    # shallow depth of field; racks to the sky while the camera tilts up
    blur = 5.0 * (1 - K.bump(t, tm.look + 0.3, tm.g3s + 2.2, 1.0, 1.0))
    bc_background(c, cam, t, T, blur=blur, shoot=tm.g3s + 0.95)

    with cam.apply(c, 1.0):
        # ---- Deng ---------------------------------------------------------------
        pose = robot_pose(**st['deng'])
        rig = K.robot_rig(pose, T)
        heart, head = rig['heart'], rig['head']
        sd = st['star']
        sx, sy = bc_star_pos(t, tm, heart, head, s)
        # hop onto his head at the end
        hop_k = ease_in_out((t - tm.hop) / 0.75)
        if hop_k > 0:
            top = (head[0] + 8 * s, head[1] - 70 * s - sr * 0.5)
            sx = lerp(sx, top[0], hop_k) + math.sin(hop_k * math.pi) * 90 * s
            sy = lerp(sy, top[1], hop_k) - math.sin(hop_k * math.pi) * 45 * s
        # bounce when beaming
        sy -= 7 * s * K.hop(t, tm.beam, 0.4)
        spos = (sx, sy)
        # hands: cradle her from below (far hand = her right side, near hand = her left side);
        # the near hand pats her gently on "别怕，别怕"
        pat = K.hop(t, tm.d5s + 0.25, 0.3) + K.hop(t, tm.d5s + 0.85, 0.3)
        sy_h = sy if hop_k <= 0 else bc_star_pos(tm.hop, tm, heart, head, s)[1]
        sx_h = sx if hop_k <= 0 else bc_star_pos(tm.hop, tm, heart, head, s)[0]
        hold_l = (sx_h + sr * 0.6, sy_h + sr * 0.42)
        hold_r = (sx_h - sr * 0.62, sy_h + sr * 0.5 - 7 * s * pat)
        if hop_k > 0:   # after she hops off, the far hand drops to rest
            rest_l = (heart[0] + 30 * s, heart[1] + 55 * s)
            kk = smoothstep(0.1, 0.7, hop_k)
            hold_l = (lerp(hold_l[0], rest_l[0], kk), lerp(hold_l[1], rest_l[1], kk))
        fist = K.bump(t, tm.fist_ant - 0.2, tm.d6e + 0.1, 0.15, 0.4)
        # fist pump for D06 with the near arm (anticipation → up with overshoot → settle)
        if t > tm.fist_ant - 0.35:
            ant = K.bump(t, tm.fist_ant - 0.25, tm.fist_up - 0.05, 0.25, 0.08)
            up = ease_out_back(clamp((t - tm.fist_up) / 0.3), 2.2)
            fist_rest = (heart[0] - 40 * s, heart[1] + 30 * s + 16 * s * ant)
            fist_top = (head[0] - 92 * s, head[1] + 6 * s)
            fx_ = lerp(fist_rest[0], fist_top[0], up)
            fy_ = lerp(fist_rest[1], fist_top[1], up)
            fy_ += 3 * s * math.sin((t - tm.fist_up) * 16) * K.bump(t, tm.fist_up + 0.25, tm.fist_up + 0.6, 0.05, 0.2)
            down = smoothstep(tm.d6e - 0.2, tm.d6e + 0.5, t)      # lower it gently after the line
            fx_ = lerp(fx_, heart[0] - 30 * s, down)
            fy_ = lerp(fy_, heart[1] + 60 * s, down)
            blend_in = smoothstep(tm.fist_ant - 0.35, tm.fist_ant, t)
            hold_r = (lerp(hold_r[0], fx_, blend_in), lerp(hold_r[1], fy_, blend_in))
        pose, rig = K.pose_with_hands(pose, T, hold_l, hold_r, rig=rig)
        drawn = robot.draw_robot(c, pose, T)
        # warm light from the heart spilling onto her (with a soft heartbeat)
        hug_on = smoothstep(tm.cut_warm, tm.cut_warm + 1.2, t) * (1 - 0.5 * smoothstep(tm.look - 0.5, tm.look + 0.5, t))
        beat = 0.5 + 0.5 * max(K.hop((t % 1.25), 0.0, 0.22), 0.7 * K.hop((t % 1.25), 0.28, 0.25))
        K.call_kw(fx.glow, c, heart[0], heart[1], 120 * s, color='#ffb347',
                  intensity=0.3 * hug_on * (0.75 + 0.25 * beat), core=False)
        # tiny warm motes drifting up from the heart while it warms her
        warm_motes = K.bump(t, tm.d5s, tm.cut_close + 1.0, 0.8, 1.0)
        if warm_motes > 0:
            K.call_kw(fx.embers, c, T, heart[0] + 10 * s, heart[1] - 10 * s, count=12, color='#ffc070', seed=21,
                      intensity=0.55 * warm_motes, spread=45 * s, rise=110 * s, size=0.9)

        # ---- Guang ----------------------------------------------------------------
        spin = -TAU * ease_in_out(clamp((t - tm.hop - 0.05) / 0.65))     # a happy twirl on the way up
        spose = star_pose(x=spos[0], y=spos[1], scale=STAR_BC_SCALE, rot=sd['rot'] + spin, glow=sd['glow'],
                          flicker=sd['flicker'], squash=sd['squash'], arm_l=sd['arm_l'], arm_r=sd['arm_r'],
                          eyes=sd['eyes'], blink=sd['blink'], look=sd['look'], mouth=sd['mouth'],
                          smile=sd['smile'], blush=sd['blush'], tears=sd['tears'], wet=sd['wet'],
                          shiver=min(1.0, sd['shiver']), cover=sd['cover'], leg_l=sd['leg_l'],
                          leg_r=sd['leg_r'], hug=sd['hug'], halo=lerp(0.5, 0.9, st['warm']) + 0.3 * st['beamk'],
                          rim=0.6, rim_color='#ffb347',
                          rim_dir=math.atan2(heart[1] - spos[1], heart[0] - spos[0]),
                          trail=0.6 * math.sin(hop_k * math.pi) if 0 < hop_k < 1 else 0.0)
        draw_star(c, spose, T)
        # the near mitten's palm in front of her (she sits IN his hands, not on top of them)
        if hop_k < 0.15 and fist < 0.5:
            K.overdraw_hand(c, pose, T, drawn['r'], s)
        if st['beamk'] > 0:
            K.call_kw(fx.sparkles, c, T, spos[0], spos[1], radius=sr * 2.0, count=22, color='#fff4c0',
                      seed=31, intensity=st['beamk'] * (0.7 + 0.3 * math.exp(-2 * max(0, t - tm.beam))),
                      rise=40, size=1.2)
            K.call_kw(fx.sparkle_burst, c, spos[0], spos[1], t - tm.beam, scale=0.9, color='#fff4c0', seed=33, ring=False)

        # ---- boat gunwale in the foreground ----------------------------------------
        K.call_kw(sea.boat, c, BOAT_BC['x'], BOAT_BC['y'], BOAT_BC['s'], rock=0.012 * math.sin(T * 0.9),
                  lantern=0.25, t=T)

    # ---- grade ---------------------------------------------------------------------
    hugk = smoothstep(tm.cut_warm, tm.d5s + 1.5, t)
    tint_amt = 0.05 + 0.1 * hugk
    tint_amt -= 0.08 * st['look_up']
    f.grade.update(tint='#ffd9a0', tint_amt=max(0.0, tint_amt), exposure=1.0 + 0.04 * hugk - 0.05 * st['xm'],
                   bloom=1.0 + 0.3 * st['beamk'], vignette=0.5, sat=0.95 + 0.07 * st['beamk'])


# ----------------------------------------------------------------------------
# SHOT D — wide: rowing back to the lighthouse
# ----------------------------------------------------------------------------
HZ_D = 600.0
BOAT_D_S = 1.0
DENG_D_SCALE = 0.74
STAR_D_SCALE = 0.62
STROKE = 1.6          # one oar stroke every 1.6 s from `row_back` (shared convention)
PULL = 0.45           # fraction of the stroke with the blade in the water


def d_boat_x(t, tm):
    s_ = max(0.0, t - tm.row)
    ph = s_ / STROKE
    # steady glide + a surge after each catch
    return 470 + 34 * s_ + 10 * math.sin(TAU * ph - 1.2)


def d_oar(u, lock, sb):
    """Oar geometry in side view for stroke phase u (0 = catch).  Returns handle, blade, dip."""
    if u < PULL:
        v = ease_in_out(u / PULL)
        th = lerp(0.75, -0.6, v)
        dip = 8 * sb
    else:
        v = ease_in_out((u - PULL) / (1 - PULL))
        th = lerp(-0.6, 0.75, v)
        dip = -34 * sb * math.sin(math.pi * v) ** 0.8
    L_out, L_in = 150 * sb, 70 * sb
    bx = lock[0] + L_out * math.sin(th)
    by = lock[1] + L_out * math.cos(th) * 0.5 + dip + 16 * sb
    dx, dy = bx - lock[0], by - lock[1]
    hx, hy = lock[0] - dx * L_in / L_out, lock[1] - dy * L_in / L_out
    return (hx, hy), (bx, by), th


def draw_oar(c, handle, blade, water_y, sb, dark=0.0, feather=0.0):
    """Wooden oar with a paddle blade; the part below the water line is faint (submerged)."""
    shaft = mix('#9a7048', '#3a2a24', dark)
    hi = mix('#d8b07a', '#5a4234', dark)
    ang = math.atan2(blade[1] - handle[1], blade[0] - handle[0])

    def _draw(a):
        c.drawLine(handle[0], handle[1], blade[0], blade[1], stroke(shaft, 6.5 * sb, a))
        c.drawLine(handle[0], handle[1], blade[0], blade[1], stroke(hi, 2.0 * sb, 0.5 * a))
        with at(c, blade[0], blade[1], rot=ang):
            w = 11 * sb * (1 - 0.6 * feather)
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(-36 * sb, -w, 50 * sb, 2 * w), w, w), fill(shaft, a))
            c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(-32 * sb, -w * 0.8, 42 * sb, w * 0.7), w * 0.4, w * 0.4),
                        fill(hi, 0.35 * a))
        c.drawCircle(handle[0], handle[1], 4.5 * sb, fill(mix(shaft, '#000000', 0.2), a))
    c.save()
    c.clipRect(skia.Rect.MakeLTRB(-5000, -5000, 9000, water_y))
    _draw(1.0)
    c.restore()
    c.save()
    c.clipRect(skia.Rect.MakeLTRB(-5000, water_y, 9000, 9000))
    _draw(0.28)
    c.restore()


def shot_row(f, tm):
    c, t, T = f.canvas, f.t, f.T
    s_ = t - tm.row
    cam = Camera(x=W - lerp(880, 960, ease_in_out(s_ / 5.0)), y=lerp(565, 540, ease_in_out(s_ / 5.0)),
                 zoom=lerp(1.14, 1.0, ease_out(s_ / 5.0)), t=T)

    ph = max(0.0, s_) / STROKE
    k_stroke = math.floor(ph)
    u = ph - k_stroke                    # 0 = blade enters the water (catch)
    sb = BOAT_D_S
    bx = d_boat_x(t, tm)
    by = 812 + 3 * math.sin(T * 1.4)
    pull = math.sin(math.pi * clamp(u / PULL)) if u < PULL else 0.0
    rock = 0.015 * math.sin(T * 1.1) - 0.012 * pull
    gunwale = by - 30 * sb

    # ---- sky, sea & distant island ----------------------------------------------
    # Screen direction: in s03 Deng rowed screen-right away from the lighthouse; rowing BACK
    # he travels screen-left towards it.  The foreground is laid out moving right and then
    # mirrored about x = W/2 (MIR); background stays unmirrored so the beam sweeps correctly.
    def MIR(x):
        return W - x
    sky_backdrop(c, cam, t, T, horizon_y=HZ_D, sky_depth=0.12, mw_center=(820, 120), mw_alpha=1.0)
    ix, iy = 330, HZ_D + 16
    lamp_guess = (ix, iy - 222)
    head_guess = (MIR(bx), by - 250)
    with cam.apply(c, 1.0):
        lantern_guess = (MIR(bx + 130 * sb), by - 120 * sb)
        K.call_kw(sea.ocean, c, T, horizon_y=HZ_D, rect_x=(-2400, 4400), bottom=2400,
                  lights=[(lamp_guess[0], lamp_guess[1], '#ffd36b', 0.7),
                          (lantern_guess[0], lantern_guess[1], '#ffc86b', 0.8, by),
                          (head_guess[0], head_guess[1], '#ffd86b', 0.7, by)])
        for i, (gx, gy, rr) in enumerate(((300, 700, 420), (1250, 690, 520), (800, 940, 600), (1700, 860, 480),
                                          (-100, 900, 500))):
            K.call_kw(fx.bioluminescence, c, T, gx, gy, radius=rr, intensity=0.6, persp=0.2, seed=40 + i)
        K.call_kw(sea.island, c, ix, iy, 0.5)
        lx, ly = K.call_kw(sea.lighthouse, c, ix, iy, 0.5, lamp=1.0, t=T, _default=lamp_guess)
        K.call_kw(sea.beam, c, lx, ly, T * 0.9, length=2600, width0=24, width1=460, intensity=0.9)

    with cam.apply(c, 1.0):
        c.translate(W, 0)
        c.scale(-1, 1)
        # wake: fading ripples left behind every stroke + glowing plankton stirred by the oars
        for j in range(0, 4):
            kk = k_stroke - j
            if kk < 0:
                continue
            t0 = tm.row + kk * STROKE
            wx = d_boat_x(t0, tm) - 150 * sb
            K.call_kw(fx.ripples, c, wx, by + 10, t - t0, color='#bfe0ff', intensity=0.3, persp=0.16,
                      count=2, speed=70, life=4.0)
            lock0 = (d_boat_x(t0, tm) + 12 * sb, by - 30 * sb)
            _, blade0, _ = d_oar(0.0, lock0, sb)
            K.call_kw(fx.ripples, c, blade0[0], by + 14 * sb, t - t0, color='#cfe8ff', intensity=0.6, persp=0.2,
                      count=3, speed=80, life=2.4)
            K.call_kw(fx.bioluminescence, c, T, blade0[0] - 40 * sb, by + 16 * sb, radius=110 * sb,
                      intensity=0.9, persp=0.25, seed=60 + int(kk) % 7, t_since=t - t0, life=3.2)

        # ---- far oar (behind the boat) ----------------------------------------------
        lock_f = (bx + 22 * sb, gunwale - 4 * sb)
        hf, bf, _ = d_oar(u, lock_f, sb)
        bf = (bf[0] + 6 * sb, bf[1] - 26 * sb)     # the far blade dips on the other side of the hull
        draw_oar(c, hf, bf, by - 2, sb, dark=0.45)

        # ---- Deng in the boat, rowing ----------------------------------------------
        lock_n = (bx + 12 * sb, gunwale)
        hn, bn, th = d_oar(u, lock_n, sb)
        idl = robot.idle(T, 7)
        push = (hn[0] - lock_n[0]) / (70 * sb)          # -1..1 handle travel
        pose = robot_pose(x=bx - 34 * sb, y=gunwale + 10 * sb, scale=DENG_D_SCALE, facing=1.0, sit=1.0,
                          sit_dangle=0.0, lean=0.1 + 0.16 * push + idl['lean'], eyes='happy', smile=0.75,
                          blink=0.0, look=(0.4, -0.3), scarf_wind=0.7, scarf_dir=-1.0,
                          hand_l_open=0.15, hand_r_open=0.15, head_tilt=-0.05 + idl['head_tilt'],
                          head_dy=idl['head_dy'], rim=0.5, shadow=0.0)
        pose, _ = K.pose_with_hands(pose, T, (hf[0], hf[1]), (hn[0], hn[1]))
        drawn = robot.draw_robot(c, pose, T)
        # Guang sitting on his head, swinging her legs, glowing happily
        top = drawn.get('head_top', (drawn['head'][0], drawn['head'][1] - 57 * DENG_D_SCALE))
        gy = top[1] - star.RADIUS * STAR_D_SCALE * 0.62 + 1.5 * math.sin(T * 2.2)
        gp = star_pose(x=top[0] - 4, y=gy, scale=STAR_D_SCALE, rot=0.06 * math.sin(T * 1.6), glow=1.2,
                       eyes='happy', smile=0.85, blush=0.75, arm_l=0.5 + 0.25 * math.sin(T * 2.4), arm_r=0.35,
                       leg_l=0.4 + 0.3 * math.sin(T * 4.0), leg_r=0.4 - 0.3 * math.sin(T * 4.0),
                       look=(0.6, -0.1), squash_pivot=1.0, squash=0.04 * math.sin(T * 4.4))
        draw_star(c, gp, T)
        K.call_kw(fx.sparkles, c, T, top[0], gy, radius=70, count=8, color='#fff4c0', seed=51,
                  intensity=0.55, rise=40, size=0.7)
        # boat hull & lantern
        K.call_kw(sea.boat, c, bx, by, sb, rock=rock, lantern=1.0, t=T)
        # near oar in front of the hull
        draw_oar(c, hn, bn, by + 4, sb, feather=1.0 - pull if u >= PULL else 0.0)
        # drips from the blade during the recovery
        if u > PULL + 0.05:
            K.call_kw(fx.water_drips, c, T, bn[0], bn[1] + 6 * sb, spread=10 * sb, rate=6.0, seed=70,
                      fall=max(4.0, by + 10 - bn[1]), size=0.6, glint='#ffd86b', intensity=0.8, spots=3)

    f.grade.update(vignette=0.5, tint='#c8d4ff', tint_amt=0.04, bloom=1.1)


# ----------------------------------------------------------------------------
def render(f):
    tm = Tm(f)
    t = f.t
    if t < tm.cut_warm:
        shot_rescue(f, tm)
    elif t < tm.row:
        shot_close(f, tm)
    else:
        shot_row(f, tm)
