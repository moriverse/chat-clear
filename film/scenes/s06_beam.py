"""s06 · 灯塔的光 The beam — ★ CLIMAX (48 s).  Owner: SCENE-S06.

Shot list (all cut points derive from timeline cues / line times):
 1  [ .. run_up)          IDEA  low-angle close-up of Deng on the rocks, the lighthouse beam
                          sweeping overhead and washing over his face at "…照到天上！"; his eyes
                          light up with hope (antenna flash on the idea).
 2  [run_up .. aim_up)    RUN   up the spiral stairs, 2.2 walk cycles/s from run_up, dim Guang.
                          Two angles (cut at the midpoint) so he stays on the near side.
 3  [aim_up .. beam_up)   AIM   lamp room: three heaves on the iron wheel, lens tilts up, hatch.
 4  [beam_up .. D12)      WEAK  wide exterior: a thin vertical beam that dies long before the sky.
 5  [D12 .. touch_heart)  D12   medium on Deng: looks up at the beam, then down at grey Guang.
 6  [touch .. G08)        HEART close on his chest: the heart core glows, his hand on the glass.
 7  [G08 .. D13+0.3)      G08   two-shot, Guang in his palm reaching up; his soft smile.
 8  [.. D14-0.4)          D13   wide lamp room, Deng at the glass, the empty sea. Slow push.
 9  [.. heart_out)        D14   warm close-up, his eyes on her.
10a [heart_out .. +1.6)   OUT   he opens the porthole, lifts the heart out; eyes flicker, sags.
10b [.. ignite)           IN    closer on the lens: heart placed at heart_in, energy gathers.
11  [ignite .. +2.2)      IGNITION wide: flash, shockwaves, the pillar births to the top of the
                          sky, clouds blown into a ring, shake; camera tilts up the pillar.
12  [.. whale_appear)     RISE  Guang regains her colour and floats up the pillar; G09 as the
                          camera pulls far back to the tiny lighthouse below.
13  [whale .. +1.0)       WHALE the Milky Way swirls, the constellation whale circles down.
14  [.. +2.9)             D15   Deng in the blazing lamp room, dimming, a weak wave.
15  [.. deng_dark)        FADE  the pillar thins and fades; the whale gathers Guang up.
16  [deng_dark .. +1.6)   DARK  Deng's eyes go out, he slumps against the lamp.
17  [.. end]              HOLD  the dark lighthouse, silence.
"""
from __future__ import annotations

import math

import skia

from engine.core import (ADD, H, TAU, W, Camera, clamp, col, ease_in, ease_in_out, ease_out, ease_out_back,
                         keyframes, lerp, noise1, pulse, radial, smoothstep)
from lib import fx, robot, sea, sky, star
from scenes import _s06_kit as K

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
EXT_HZ = 800                      # exterior horizon (world y)
LH_X, LH_Y, LH_S = 960.0, 874.0, 1.0
CLOUD_Y = -250.0
SWIRL_C = (960.0, -3150.0)
PILLAR_TOP = -4700.0              # dissolves into the heart of the swirl
GUANG_DX = 150.0                  # Guang drifts out to the pillar's edge so she reads against it

# lamp room layout (design units of lib.sea.lamp_room's 1920x1080 frame)
DENG_LR = dict(x=1130.0, y=935.0, scale=1.35)
WHEEL = (1455.0, 690.0, 150.0)

WARM = '#ffd36b'
WARM2 = '#ffb347'
BEAMC = '#ffe7a0'
COOL = '#9fc4ff'
GOLD = '#fff0b8'


def _anchors():
    try:
        return sea.lighthouse_anchors(LH_X, LH_Y, LH_S)
    except Exception:
        return dict(lamp=(LH_X, LH_Y - 445 * LH_S), hatch=(LH_X, LH_Y - 500 * LH_S))


ANCH = _anchors()
LAMP0 = tuple(ANCH['lamp'])
HATCH = tuple(ANCH.get('hatch', LAMP0))


# ----------------------------------------------------------------------------
# timing
# ----------------------------------------------------------------------------
class Q:
    """Local cue times for this scene (+ derived cut points)."""

    def __init__(self, f):
        cu = f.cue
        self.idea = cu('idea')
        self.run = cu('run_up')
        self.aim = cu('aim_up')
        self.bup = cu('beam_up')
        self.touch = cu('touch_heart')
        self.hout = cu('heart_out')
        self.hin = cu('heart_in')
        self.ign = cu('ignite')
        self.rise = cu('rise')
        self.whale = cu('whale_appear')
        self.fade = cu('beam_fade')
        self.dark = cu('deng_dark')
        self.d11 = f.line('D11')
        self.d12 = f.line('D12')
        self.g08 = f.line('G08')
        self.d13 = f.line('D13')
        self.d14 = f.line('D14')
        self.g09 = f.line('G09')
        self.d15 = f.line('D15')
        self.end = f.dur
        self.run_mid = (self.run + self.aim) / 2
        self.c_d13 = self.d13[0] + 0.3
        self.c_d14 = self.d14[0] - 0.4
        self.c_in = self.hout + 1.6
        self.c_rise = self.ign + 2.2
        self.c_d15 = self.whale + 1.0
        self.c_fade = self.c_d15 + 2.9
        self.c_hold = self.dark + 1.6
        self.lift = self.hout + 0.75      # the heart leaves the chest


def spline(t, keys):
    """Catmull-Rom through (time, value) keys, clamped at the ends (smooth velocities)."""
    if t <= keys[0][0]:
        return keys[0][1]
    if t >= keys[-1][0]:
        return keys[-1][1]
    for i in range(len(keys) - 1):
        t0, v0 = keys[i]
        t1, v1 = keys[i + 1]
        if t <= t1:
            vm, tm = (keys[i - 1][1], keys[i - 1][0]) if i > 0 else (v0 - (v1 - v0), t0 - (t1 - t0))
            v2, t2 = (keys[i + 2][1], keys[i + 2][0]) if i + 2 < len(keys) else (v1 + (v1 - v0), t1 + (t1 - t0))
            h = t1 - t0
            m0 = (v1 - vm) / (t1 - tm) * h
            m1 = (v2 - v0) / (t2 - t0) * h
            u = (t - t0) / h
            u2, u3 = u * u, u * u * u
            return ((2 * u3 - 3 * u2 + 1) * v0 + (u3 - 2 * u2 + u) * m0 +
                    (-2 * u3 + 3 * u2) * v1 + (u3 - u2) * m1)
    return keys[-1][1]


def dawn_at(t):
    return 0.36 + 0.06 * clamp(t / 48.0)


def dropout(t, seed, rate=9.0, thresh=0.25):
    """1 normally, 0 during brief deterministic flicker dropouts."""
    return 0.0 if noise1(t * rate, seed) < -1 + 2 * thresh else 1.0


def deng_power(t, q):
    """Electric life of Deng (lib.robot adds its own flicker when power < 0.5)."""
    if t < q.lift:
        return 1.0
    base = keyframes(t, [(q.lift, 1.0), (q.lift + 0.9, 0.36), (q.fade, 0.3), (q.dark, 0.24)])
    strong = 1 - smoothstep(q.lift + 0.2, q.lift + 1.6, t)
    p = base * (1 - 0.7 * strong * (1 - dropout(t, 71, 11.0, 0.3)))
    if t >= q.dark:  # stutter out: on, off, on... dark
        u = t - q.dark
        if u < 0.55:
            on = (u < 0.12) or (0.22 < u < 0.3) or (0.4 < u < 0.44)
            p = base * (0.9 if on else 0.0) * (1 - u / 0.55)
        else:
            p = 0.0
    return clamp(p)


def guang_glow(t, q):
    return spline(t, [(-2.0, 0.34), (q.d12[0], 0.26), (q.touch, 0.17), (q.g08[0], 0.15), (q.ign, 0.12),
                      (q.ign + 1.2, 0.5), (q.rise + 0.4, 0.8), (q.rise + 1.8, 1.8), (q.end + 2, 1.85)])


def guang_flicker(t, q):
    return keyframes(t, [(0, 0.4), (q.d12[0], 0.55), (q.ign, 0.75), (q.ign + 1.0, 0.3), (q.rise + 1.2, 0.0)])


def guang_y(t, q):
    """Guang's world altitude in the exterior 'pillar world'."""
    return spline(t, [(q.rise, 400.0), (q.rise + 0.7, 330.0), (q.rise + 1.7, 120.0), (q.rise + 2.7, -260.0),
                      (q.whale, -1100.0), (q.whale + 1.5, -1850.0), (q.fade - 0.7, -2300.0),
                      (q.fade + 0.3, -2420.0), (q.end + 2, -2700.0)])


def guang_x(t, q):
    return LH_X + GUANG_DX * smoothstep(q.rise, q.rise + 1.6, t) + 14 * math.sin(t * 1.3)


def guang_v(t, q):
    dt = 1 / 48
    return ((guang_x(t + dt, q) - guang_x(t - dt, q)) / (2 * dt),
            (guang_y(t + dt, q) - guang_y(t - dt, q)) / (2 * dt))


# ----------------------------------------------------------------------------
# common drawing helpers
# ----------------------------------------------------------------------------
def draw_guang(c, f, x, y, scale, glow, flicker=0.0, **kw):
    kw.setdefault('blink', robot.auto_blink(f.T, seed=3, every=4.3))
    kw.setdefault('mouth', f.mouth('guang'))
    pose = K.spose(x=x, y=y, scale=scale, glow=glow, flicker=flicker, **kw)
    return star.draw_star(c, pose, f.T)


def draw_stars(c, f, cam, depth=0.15, bright=1.0, hz_world=None):
    with cam.apply(c, depth):
        rect = K.visible(cam, depth, 80)
        hz = None
        if hz_world is not None:
            sy = cam.to_screen(0, hz_world, 1.0)[1]
            hz = K.cam_inv(cam, 0, sy, depth)[1]
        K.call(sky.stars, c, f.T, rect=rect, bright=bright, horizon_y=hz)


def cradle(pose, g):
    """Hand IK targets (near r, far l) cupping Guang centred at g."""
    s = pose['scale']
    fc = pose.get('facing', 1.0)
    pose['arm_r_override'] = (g[0] + fc * 30 * s, g[1] + 40 * s)
    pose['arm_l_override'] = (g[0] - fc * 32 * s, g[1] + 44 * s)
    pose.setdefault('hand_r_open', 0.9)
    pose.setdefault('hand_l_open', 0.9)
    return pose


def robot_anchor(pose, key, t):
    try:
        return robot.anchors(K.rpose(**pose), t)[key]
    except Exception:
        s = pose['scale']
        off = dict(heart=-195, head=-300, eyes=-305)
        return pose['x'], pose['y'] + off.get(key, -200) * s


# ----------------------------------------------------------------------------
# exterior "pillar world": sky, sea, island, lighthouse, beam / pillar, clouds, whale
# ----------------------------------------------------------------------------
def whale_state(t, q):
    th = spline(t, [(q.whale - 0.5, -2.45), (q.whale, -2.2), (q.fade + 0.3, 1.22), (q.end + 2, 4.3)])
    oy = spline(t, [(q.whale - 0.5, -3000.0), (q.whale, -2900.0), (q.fade + 0.3, -2560.0), (q.end + 2, -3500.0)])
    x = LH_X + 1050 * math.cos(th)
    y = oy + 230 * math.sin(th)
    sc = 0.6 * (1 + 0.16 * math.sin(th))
    sn = -math.sin(th)
    facing = math.copysign(max(0.3, abs(sn)), sn if abs(sn) > 1e-6 else 1.0)
    alpha = smoothstep(q.whale - 0.2, q.whale + 0.9, t) * (1 - 0.6 * smoothstep(q.dark, q.end + 1, t))
    rot = 0.1 * math.cos(th) * (1 if facing > 0 else -1)
    return dict(x=x, y=y, scale=sc, facing=facing, alpha=alpha, rot=rot, near=math.sin(th) > 0, th=th)


def _draw_whale(c, T, ws):
    return K.call(star.draw_whale, c, T, ws['x'], ws['y'], scale=ws['scale'], alpha=ws['alpha'],
                  facing=ws['facing'], swim=T * 0.5, rot=ws['rot'], swim_rate=0.5, glow=1.1)


def exterior(f, c, cam, q, *, dawn, lamp=1.0, beam_up=0.0, beam=0.0, beam_len=0.0, pillar=0.0, birth=1.0,
             pillar_w=150.0, ring=0.0, cloud_glow=0.0, swirl=0.0, whale=False, guang=None,
             shock=None, stars_bright=1.0, sea_light=0.0, cloud_alpha=0.5):
    """Draw the wide exterior world. guang: dict for draw_guang (+ 'x','y', 'gather') or None."""
    T = f.T
    t = f.t
    with cam.apply(c, 1.0):
        K.call(sky.sky, c, rect=K.visible(cam, 1.0), horizon_y=EXT_HZ, dawn=dawn)
        if dawn > 0:
            K.call(sky.dawn_glow, c, horizon_y=EXT_HZ, amount=dawn * 0.8, sun_x=1500)
    mw_a = 0.8 * stars_bright * (1 - clamp(swirl * 1.4))   # the band is pulled into the vortex
    if mw_a > 0.02:
        with cam.apply(c, 0.3):
            mw_hz = K.cam_inv(cam, 0, cam.to_screen(0, EXT_HZ, 1.0)[1], 0.3)[1]
            K.call(sky.milky_way, c, T, center=(760, -620), angle=-1.05, length=6200, width=700, bend=0.06,
                   alpha=mw_a, horizon_y=mw_hz)
    draw_stars(c, f, cam, 0.15, stars_bright, EXT_HZ)

    ws = whale_state(t, q) if whale else None
    whale_res = None
    lx, ly = LAMP0
    hx, hy = HATCH
    with cam.apply(c, 1.0):
        vis = K.visible(cam, 1.0)
        x0, y0, vw, vh = vis
        # --- the great swirl & the far side of the whale's orbit ----------------------------
        if swirl > 0:
            if hasattr(sky, 'sky_swirl'):
                K.call(sky.sky_swirl, c, T, SWIRL_C[0], SWIRL_C[1], 1500, swirl, streaks=1.0)
            else:
                K.sky_swirl(c, T, SWIRL_C[0], SWIRL_C[1], 1500, swirl, 0.2 * T)
        if ws and ws['alpha'] > 0 and not ws['near']:
            whale_res = _draw_whale(c, T, ws)
        # --- clouds (a thin deck; the pillar blows it open into a ring) ------------------------
        if cloud_alpha > 0 and y0 < CLOUD_Y + 500 and y0 + vh > CLOUD_Y - 500:
            if hasattr(sky, 'cloud_ring'):
                K.call(sky.cloud_ring, c, T, LH_X, CLOUD_Y, 1500.0, ring, rect_x=(x0, x0 + vw), thickness=210.0,
                       squash=0.2, alpha=cloud_alpha, glow=cloud_glow, dawn=dawn * 0.6, cover=0.55,
                       color='#3e5190')
            else:
                K.cloud_deck(c, T, LH_X, CLOUD_Y, open_=ring, alpha=cloud_alpha, lit=cloud_glow)
        # --- sea ------------------------------------------------------------------------------
        if y0 + vh > EXT_HZ:
            lights = []
            if lamp > 0:
                lights.append((lx, ly, WARM, 0.8 * lamp))
            if sea_light > 0:
                lights.append((lx, ly, GOLD, sea_light, None, 3.0))
            K.call(sea.ocean, c, T, horizon_y=EXT_HZ, rect_x=(x0, x0 + vw), bottom=y0 + vh + 50,
                   lights=lights, dawn=dawn)
            if shock is not None:  # shockwave racing across the water
                K.call(fx.shockwave, c, LH_X, EXT_HZ + 60, shock, color='#ffe9b0', max_r=3600, life=2.0,
                       persp=0.07, intensity=0.8)
        # --- island & lighthouse ------------------------------------------------------------------
        if y0 < LH_Y + 250 and y0 + vh > LH_Y - 700:
            K.call(sea.island, c, LH_X, LH_Y + 6, LH_S, dawn=dawn)
            r = K.call(sea.lighthouse, c, LH_X, LH_Y, LH_S, lamp=clamp(lamp), dawn=dawn, lit_windows=clamp(lamp),
                       t=T, beam_up=beam_up)
            if isinstance(r, (tuple, list)) and len(r) >= 2:
                lx, ly = float(r[0]), float(r[1])
        # --- the weak beam (screen-space vertical) --------------------------------------------------
        if beam > 0 and beam_len > 0:
            K.call(sea.beam, c, hx, hy + 10, -math.pi / 2, length=beam_len, width0=44, width1=420,
                   intensity=beam, color=BEAMC, t=T)
            K.call(fx.glow, c, hx, hy, 200, color=WARM, intensity=0.45 * beam)
        # --- the pillar ------------------------------------------------------------------------------
        if pillar > 0:
            kw = dict(width=pillar_w, intensity=pillar, color=GOLD, core='#ffffff')
            if K.accepts(fx.light_pillar, 'birth'):
                K.call(fx.light_pillar, c, T, hx, hy + 10, y_top=PILLAR_TOP, birth=birth, **kw)
            else:
                K.call(fx.light_pillar, c, T, hx, hy + 10, y_top=lerp(hy, PILLAR_TOP, birth), **kw)
        # --- near side of the whale ---------------------------------------------------------------------
        if ws and ws['alpha'] > 0 and ws['near']:
            whale_res = _draw_whale(c, T, ws)
        # --- Guang ---------------------------------------------------------------------------------------
        if guang is not None:
            g = dict(guang)
            gather = g.pop('gather', 0.0)
            gx, gy = g.pop('x'), g.pop('y')
            if gather > 0 and whale_res:
                ht = whale_res.get('head_top', whale_res.get('eye'))
                tx, ty = ht[0], ht[1] - star.RADIUS * g['scale'] * 0.9
                gx, gy = lerp(gx, tx, gather), lerp(gy, ty, gather)
            draw_guang(c, f, gx, gy, **g)
            guang['_pos'] = (gx, gy)
        # --- ignition shockwave in the air ---------------------------------------------------------------
        if shock is not None:
            K.call(fx.shockwave, c, hx, hy, shock, color='#fff4d0', max_r=3400, life=1.4, persp=1.0, intensity=0.3,
                   chroma=False)
            K.call(fx.shockwave, c, hx, hy, shock - 0.15, color='#ffd98a', max_r=2400, life=1.3, persp=0.35,
                   intensity=0.35, chroma=False)
    return (lx, ly), ws, whale_res


# ----------------------------------------------------------------------------
# lamp room
# ----------------------------------------------------------------------------
def lens_of(res):
    if isinstance(res, dict):
        return res.get('lens', (960.0, 470.0, 170.0))
    if isinstance(res, (tuple, list)) and len(res) >= 3:
        return float(res[0]), float(res[1]), float(res[2])
    return 960.0, 470.0, 170.0


def wheel_of(res):
    if isinstance(res, dict) and 'wheel' in res:
        w = res['wheel']
        return float(w[0]), float(w[1]), float(w[2]) if len(w) > 2 else WHEEL[2]
    return WHEEL


def lamp_room(f, c, cam, q, *, lamp=1.0, pointing_up=0.0, wheel_rot=0.0, lens_rot=None, beam=0.0,
              blaze=0.0, core=0.0, core_hot=0.0):
    """Lamp room interior + (fallback) wheel + beam up the hatch + blazing pillar.
    Returns (lens(x,y,r), wheel(x,y,r))."""
    T = f.T
    t = f.t
    if lens_rot is None:
        T_aim = T - t + q.aim
        lens_rot = T * 0.9 if t < q.aim else T_aim * 0.9 + 0.35 * ease_out((t - q.aim) / 0.6)
    with cam.apply(c, 1.0):
        res = K.call(sea.lamp_room, c, T, lamp=clamp(lamp), lens_rot=lens_rot, dawn=dawn_at(t),
                     pointing_up=pointing_up, wheel_rot=wheel_rot)
        lx, ly, lr = lens_of(res)
        wh = wheel_of(res)
        if not (K.accepts(sea.lamp_room, 'wheel_rot') or K.accepts(sea.lamp_room, 'wheel')):
            K.iron_wheel(c, wh[0], wh[1], wh[2], wheel_rot, light=clamp(lamp))
        if core > 0:  # the heart core sitting in the lens
            K.call(fx.heart_core, c, T, lx, ly, r=22 + 12 * core_hot, intensity=min(2.0, core), kind='flame')
            K.call(fx.glow, c, lx, ly, lr * (1.2 + 2.2 * core_hot), color=WARM2, intensity=0.35 * core + 0.7 * core_hot)
        if beam > 0:  # the (weak) beam through the open hatch
            K.call(sea.beam, c, lx, ly - lr * 0.4, -math.pi / 2, length=1300, width0=lr * 0.5, width1=lr * 2.0,
                   intensity=beam, color=BEAMC, t=T)
        if blaze > 0:
            K.call(fx.light_pillar, c, T, lx, ly - lr * 0.2, y_top=-900, width=lr * 0.9, intensity=blaze,
                   color=GOLD, core='#ffffff', rings=0.4, flare=0.6)
            K.call(fx.glow, c, lx, ly, lr * 4.0, color='#fff0c0', intensity=0.6 * blaze)
    return (lx, ly, lr), wh


def lr_pose(f, t, facing=-1.0, lens=(960.0, 470.0, 170.0), lamp_amt=1.0, **kw):
    """Base pose for Deng in the lamp room: idle, lip-sync, warm rim from the lens."""
    T = f.T
    idl = robot.idle(T, 4)
    lx, ly, _ = lens
    ang = math.atan2(ly - (DENG_LR['y'] - 250 * DENG_LR['scale']), lx - DENG_LR['x'])
    p = dict(DENG_LR, facing=facing, lean=idl['lean'], head_dy=idl['head_dy'], head_tilt=idl['head_tilt'],
             mouth=f.mouth('deng'), blink=robot.auto_blink(T, 1), scarf_wind=0.2, heart=1.0,
             rim=min(0.6, 0.4 * lamp_amt + 0.1), rim_color=WARM if lamp_amt > 0.05 else COOL,
             rim_angle=ang if lamp_amt > 0.05 else -math.pi / 2, shadow=0.4)
    p.update(kw)
    return p


def lamp_fill(lens, amt):
    """Extra warm fill light (multiplied onto Deng) from the glowing lens."""
    lx, ly, lr = lens
    return [K.light_radial(lx, ly, 1000, WARM, 0.4 * amt)]


# ----------------------------------------------------------------------------
# SHOT 1 — IDEA (low-angle close-up, beam sweeping overhead)
# ----------------------------------------------------------------------------
def shot_idea(f, c, t, q):
    T = f.T
    dawn = dawn_at(t)
    u = clamp(t / q.run)
    cam = Camera(x=lerp(760, 800, ease_in_out(u)), y=lerp(560, 500, ease_in_out(u)),
                 zoom=lerp(1.0, 1.11, ease_in_out(u)), t=T)
    hz = 1010.0
    # beam azimuth (lib.sea sweep convention: pi/2 = towards camera). It rotates at 0.9 rad/s and is
    # phased so it faces us exactly one period after s05's `idea` (and again on "…天上！").
    T_pass = T - t + q.idea + TAU / 0.9
    ang = 0.9 * (T - T_pass) + math.pi / 2
    with cam.apply(c, 0.0):
        K.call(sky.sky, c, rect=(-300, -300, W + 600, H + 600), horizon_y=hz, dawn=dawn)
        K.call(sky.dawn_glow, c, horizon_y=hz, amount=dawn * 0.8, sun_x=300)
    with cam.apply(c, 0.1):
        K.call(sky.milky_way, c, T, center=(700, 150), angle=-0.55, length=4200, width=560, alpha=0.75)
    draw_stars(c, f, cam, 0.08, 1.0, None)
    with cam.apply(c, 0.3):
        K.call(sea.ocean, c, T, horizon_y=hz, rect_x=(-600, 2600), bottom=1500,
               lights=[(1560, 300, WARM, 0.6)], dawn=dawn)
    with cam.apply(c, 0.55):
        r = K.call(sea.lighthouse, c, 1560, 1360, 2.4, lamp=1.0, dawn=dawn, lit_windows=1.0, t=T,
                   beam_angle=ang, view_e=0.2)
        lx, ly = (float(r[0]), float(r[1])) if isinstance(r, (tuple, list)) else (1560.0, 292.0)
        facing = K.call(sea.beam, c, lx, ly, ang, length=2600, width0=46, width1=760, intensity=0.75,
                        color=BEAMC, mode='sweep', horizon_y=hz, t=T, dist=0.9, flare=0.6)
    try:
        facing = float(facing)
    except (TypeError, ValueError):
        facing = max(0.0, math.sin(ang))
    wash = max(0.0, facing) ** 5
    lamp_scr = cam.to_screen(lx, ly, 0.55)
    # --- Deng -----------------------------------------------------------------------------
    s = 2.7
    idl = robot.idle(T, 2)
    lean = 0.1 * smoothstep(q.run - 0.45, q.run - 0.05, t)
    pose = dict(x=700.0, y=1190.0, scale=s, facing=1.0, lean=lean + idl['lean'], head_dy=idl['head_dy'],
                head_tilt=-0.1 - 0.08 * smoothstep(4.4, 5.2, t) + idl['head_tilt'],
                mouth=f.mouth('deng'), blink=robot.auto_blink(T, 1), scarf_wind=0.7, scarf_dir=-1.0,
                rim=0.5, rim_color='#ffe2b0', rim_angle=-0.75, shadow=0.0, heart=1.0,
                antenna_flash=pulse(t, 3.8, 4.25, 0.12))
    if t < 1.7:
        eyes, look, smile = ('worried' if t < 0.6 else 'open'), (0.45, -0.75), 0.05
    elif t < 3.75:
        eyes, look, smile = 'wonder', (0.5 - 0.35 * math.cos(ang), -0.8), 0.15
    elif t < 4.55:
        eyes, look, smile = 'surprised', (0.3, -0.9), 0.3
    else:
        eyes, look, smile = 'wonder', (0.2, -1.0), 0.7
        pose.update(eyes2='determined', eyes_mix=smoothstep(5.6, 6.1, t))
    pose.update(eyes=eyes, look=look, smile=smile, squash=-0.04 * smoothstep(q.run - 0.4, q.run, t))
    if 3.72 <= t < 3.95:
        pose['blink'] = 0.0
    g = (pose['x'] + 72 * s, pose['y'] - 168 * s + 3 * math.sin(T * 1.6))
    cradle(pose, g)
    band_c = 0.5 + 0.9 * math.cos(ang)
    lights = [K.light_band((pose['x'] - 520, 0), (pose['x'] + 520, 0), BEAMC, 0.5 * wash, 0.32, band_c),
              K.light_radial(g[0], g[1], 300, '#ffd86b', 0.18)]
    with cam.apply(c, 1.0):
        res = K.draw_robot_lit(c, pose, T, lights)
        draw_guang(c, f, g[0], g[1], s * 0.8, guang_glow(t, q), guang_flicker(t, q), eyes='sleepy',
                   look=(-0.3, -0.6), smile=0.1, rot=-0.15, halo=0.6, rim=0.4, rim_color='#ffe2b0',
                   rim_dir=-0.75)
        if wash > 0.05 and t > 4.5:  # hope: glints in his eyes as the light hits
            ex, ey = res['eyes']
            for sx in (-26, 26):
                K.glint(c, ex + sx * s, ey - 8 * s, 34 * s * wash, '#fff6d8', 0.7 * wash, rot=T * 0.5)
    if wash > 0.01:  # the beam washing over the camera
        p = radial(lamp_scr, 1600, [(0, BEAMC, 0.1 * wash), (1, BEAMC, 0.02 * wash)])
        p.setBlendMode(ADD)
        c.drawRect(skia.Rect.MakeWH(W, H), p)
        K.call(fx.lens_flare, c, lamp_scr[0], lamp_scr[1], intensity=0.3 * wash, color='#ffe7b0', t=T)
    f.grade.update(bloom=1.2 + 0.15 * wash, vignette=0.5, tint='#cfd8ff', tint_amt=0.05 * (1 - wash))


# ----------------------------------------------------------------------------
# SHOT 2 — RUN up the stairs
# ----------------------------------------------------------------------------
_STAIR_WIN = {}


def _stair_windows():
    """Two step-index windows where the stairs face the camera (largest depth scale)."""
    if _STAIR_WIN:
        return _STAIR_WIN['a'], _STAIR_WIN['b']
    ka, kb = 0.0, 3.0
    try:
        rec = skia.PictureRecorder()
        cc = rec.beginRecording(skia.Rect.MakeWH(W, H))
        sp = K.call(sea.stairs_interior, cc, 0.0, scroll=0.0, light=0.6)
        rec.finishRecordingAsPicture()
        n = 6.4
        ks = [i * 0.25 for i in range(0, 200)]
        dep = [sp(k)[2] for k in ks]

        def score(k0):
            vals = [d for k, d in zip(ks, dep) if k0 <= k <= k0 + n]
            return sum(vals) / max(1, len(vals))

        ka = max([i * 0.25 for i in range(0, 64)], key=score)
        kb_abs = max([ka + n + i * 0.25 for i in range(0, 80)], key=score)
        kb = kb_abs - n
    except Exception:
        pass
    _STAIR_WIN['a'], _STAIR_WIN['b'] = ka, kb
    return ka, kb


def shot_run(f, c, t, q):
    T = f.T
    ph = max(0.0, t - q.run) * 2.2          # walk phase: foot contacts at multiples of 0.5
    steps = ph * 2.0
    ka, kb = _stair_windows()
    second = t >= q.run_mid
    k = (kb if second else ka) + steps
    rec = skia.PictureRecorder()           # probe where the step is with scroll=0 ...
    rc = rec.beginRecording(skia.Rect.MakeWH(W, H))
    sp0 = K.call(sea.stairs_interior, rc, T, scroll=0.0, light=0.6)
    rec.finishRecordingAsPicture()
    _, y0, _ = sp0(k)
    scroll = (900.0 if second else 860.0) - y0   # ... then scroll so Deng stays framed
    cam = Camera(t=T, zoom=1.04 if second else 1.0)
    with cam.apply(c, 1.0):
        sp = K.call(sea.stairs_interior, c, T, scroll=scroll, light=0.65)
        x, y, d = sp(k)
        xa, _, _ = sp(k + 0.25)
        facing = 1.0 if xa >= x else -1.0
        s = 1.25 * d
        bob = abs(math.sin(ph * math.pi * 2)) * 6 * s
        pose = dict(x=x, y=y, scale=s, facing=facing, walk=1.0, walk_phase=ph, run=1.0, lean=0.16,
                    eyes='determined', look=(0.6, -0.4), smile=-0.1, mouth=0.0,
                    blink=robot.auto_blink(T, 1), scarf_wind=1.0, shake=0.3,
                    rim=0.45, rim_color=WARM2, rim_angle=math.pi / 2)
        g = (x + facing * 72 * s, y - 178 * s - bob * 0.6)
        cradle(pose, g)
        K.draw_robot_lit(c, pose, T, [K.light_radial(x, y - 200 * s, 520 * s, WARM2, 0.2)])
        draw_guang(c, f, g[0], g[1], s * 0.8, guang_glow(t, q), guang_flicker(t, q) + 0.1,
                   eyes='sleepy', rot=0.2 * math.sin(ph * math.pi * 2), squash=0.05 * math.sin(ph * TAU * 2),
                   halo=0.6)
    f.grade.update(bloom=1.15, vignette=0.62, tint='#ffd9a0', tint_amt=0.06)


# ----------------------------------------------------------------------------
# SHOT 3 — AIM: heaving the wheel
# ----------------------------------------------------------------------------
HEAVES = ((0.35, 1.0), (1.1, 1.65), (1.8, 2.4))  # (start, end) seconds after aim_up


def wheel_progress(t, q):
    u = t - q.aim
    return sum(ease_in_out(clamp((u - a) / (b - a))) for a, b in HEAVES) / len(HEAVES)


def shot_aim(f, c, t, q):
    T = f.T
    u = t - q.aim
    prog = wheel_progress(t, q)
    wrot = -2.6 * prog
    pu = smoothstep(0.02, 1.0, prog)
    grind = sum(pulse(u, a + 0.1, b - 0.05, 0.1) for a, b in HEAVES)
    beam = smoothstep(q.bup - 0.25, q.bup - 0.02, t)
    cam = Camera(x=lerp(1090, 1110, u / 2.6), y=lerp(520, 505, u / 2.6), zoom=lerp(1.12, 1.18, u / 2.6),
                 shake=2.5 * grind + 10 * pulse(u, 2.35, 2.45, 0.08), t=T)
    lens, wh = lamp_room(f, c, cam, q, lamp=1.0, pointing_up=pu, wheel_rot=wrot, beam=beam)
    wx, wy, wr = wh
    heave = max(pulse(u, a, b, 0.12) for a, b in HEAVES)
    lean = -0.2 * heave + 0.08 * (1 - heave) * smoothstep(0.0, 0.3, u)
    s = DENG_LR['scale']
    done = u >= 2.35
    pose = lr_pose(f, t, 1.0, lens, 1.0, lean=lean, shake=0.8 * heave, eyes='determined', smile=-0.35,
                   mouth=0.0, blink=0.0 if heave > 0.3 else robot.auto_blink(T, 1),
                   look=(0.6, -0.1) if not done else (0.2, -1.0),
                   head_tilt=-0.06 * heave if not done else -0.15, squash=-0.06 * heave,
                   hand_r_open=0.15, hand_l_open=0.15)
    # grip points ride the rim during a pull, then re-grip between pulls
    local = 0.0
    for a, b in HEAVES:
        if a <= u <= b + 0.1:
            local = ease_in_out(clamp((u - a) / (b - a))) * (2.6 / len(HEAVES))
    base_a = math.pi * 0.92
    ga_r, ga_l = base_a - 0.25 - local, base_a + 0.28 - local
    tr = (wx + math.cos(ga_r) * wr, wy + math.sin(ga_r) * wr)
    tl = (wx + math.cos(ga_l) * wr, wy + math.sin(ga_l) * wr)
    if u < 0.3:
        e = ease_out(u / 0.3)
        tr = (lerp(pose['x'] + 90, tr[0], e), lerp(pose['y'] - 250, tr[1], e))
        tl = (lerp(pose['x'] + 60, tl[0], e), lerp(pose['y'] - 240, tl[1], e))
    pose.update(arm_r_override=tr, arm_l_override=tl)
    with cam.apply(c, 1.0):
        res = K.draw_robot_lit(c, pose, T, lamp_fill(lens, 0.5), gain=0.7)
        hx, hy = res.get('head_top', res['head'])
        gg = s * 0.8
        draw_guang(c, f, hx + 30 * s, hy - star.RADIUS * gg * 0.75 + 4 * heave * math.sin(T * 30), gg,
                   guang_glow(t, q), guang_flicker(t, q), eyes='sleepy' if u < 2.3 else 'surprised',
                   look=(0.2, -0.8), rot=0.25 + 0.08 * heave * math.sin(T * 25), halo=0.7, rim=0.5,
                   rim_color=WARM, rim_dir=math.pi)
        if grind > 0.05:  # grinding sparks at the gear
            K.call(fx.sparkles, c, T * 3, wx - wr * 0.7, wy + wr * 0.4, radius=40, count=14, color='#ffc070',
                   seed=int(T * 8) % 5 + 40, intensity=grind, rise=-30.0, size=0.8)
        K.call(fx.glow, c, lens[0], 40, 500, color=COOL, intensity=0.25 * pu, core=False)
        K.call(fx.glow, c, lens[0], lens[1], lens[2] * 2.5, color='#fff0c0', intensity=0.7 * beam)
    f.grade.update(bloom=1.3 + 0.5 * beam, vignette=0.55, tint='#ffd9a0', tint_amt=0.07)


# ----------------------------------------------------------------------------
# SHOT 4 — the weak beam, wide exterior
# ----------------------------------------------------------------------------
def shot_weak(f, c, t, q):
    T = f.T
    u = t - q.bup
    grow = ease_out(clamp(u / 0.8))
    strain = 0.85 + 0.15 * noise1(T * 7, 5)
    e = ease_in_out(clamp((u - 0.25) / 1.5))
    cam = Camera(x=960, y=lerp(600, 290, e), zoom=lerp(0.95, 0.86, e), t=T)
    exterior(f, c, cam, q, dawn=dawn_at(t), lamp=1.0, beam_up=smoothstep(0, 0.4, u) * 0.8,
             beam=1.15 * strain, beam_len=lerp(60, 820, grow), cloud_alpha=0.7)
    f.grade.update(bloom=1.35, vignette=0.5)


# ----------------------------------------------------------------------------
# SHOTS 5..9 — lamp room drama (D12, touch_heart, G08, D13, D14)
# ----------------------------------------------------------------------------
def palm_guang_pos(pose, lift=0.0):
    s = pose['scale']
    return (pose['x'] + pose['facing'] * 80 * s, pose['y'] - (150 + lift) * s)


def shot_lamproom_drama(f, c, t, q):
    T = f.T
    beam = 0.65 * (0.85 + 0.15 * noise1(T * 6, 9))
    s = DENG_LR['scale']
    hp = 1.0 + 0.15 * math.sin(T * 5.0)
    gl, gfl = guang_glow(t, q), guang_flicker(t, q)
    lens0 = (960.0, 470.0, 170.0)
    if t < q.touch:  # ------------------------------------------------ SHOT 5: D12
        u = t - q.d12[0]
        e = ease_in_out(clamp(u / 3.0))
        cam = Camera(x=lerp(1070, 1085, e), y=lerp(545, 560, e), zoom=lerp(1.72, 1.86, e), t=T)
        ld = smoothstep(0.5, 1.3, u)
        pose = lr_pose(f, t, -1.0, lens0, 1.0, eyes='worried' if u < 1.1 else 'sad',
                       look=(lerp(0.2, 0.3, ld), lerp(-1.0, 0.8, ld)), head_tilt=lerp(-0.18, 0.12, ld), smile=-0.4)
        g = palm_guang_pos(pose, 10)
        gk = dict(eyes='sleepy', look=(0.2, -0.5), rot=0.4)
        touch = 0.0
    elif t < q.g08[0]:  # ------------------------------------------- SHOT 6: touch_heart
        u = t - q.touch
        cam = Camera(x=lerp(1110, 1105, u / 1.4), y=lerp(640, 630, u / 1.4), zoom=lerp(2.75, 2.95, u / 1.4), t=T)
        pose = lr_pose(f, t, -1.0, lens0, 1.0, eyes='sad', look=(0.1, 1.0), head_tilt=0.2, smile=-0.15)
        g = palm_guang_pos(pose, -10)
        gk = dict(eyes='sleepy', look=(0.3, -0.6), rot=0.6)
        hp *= 1.0 + 0.3 * smoothstep(0.2, 0.9, u)
        touch = smoothstep(0.25, 0.85, u)
    elif t < q.c_d13:  # -------------------------------------------- SHOT 7: G08 two-shot + smile
        u = t - q.g08[0]
        e = ease_in_out(clamp(u / (q.c_d13 - q.g08[0])))
        cam = Camera(x=lerp(1085, 1095, e), y=lerp(610, 590, e), zoom=lerp(2.1, 2.3, e), t=T)
        sm = smoothstep(q.g08[1] - 0.1, q.g08[1] + 0.6, t)
        pose = lr_pose(f, t, -1.0, lens0, 1.0, eyes='sad', eyes2='happy', eyes_mix=sm,
                       look=(0.35, 0.9), head_tilt=0.14, smile=lerp(-0.2, 0.55, sm))
        g = palm_guang_pos(pose, 20)
        reach = pulse(t, q.g08[0] + 1.0, q.g08[1] + 0.2, 0.4)
        gk = dict(eyes='teary' if t < q.g08[1] + 0.2 else 'sad', look=(0.3, -1.0), rot=0.15, tears=0.6,
                  smile=-0.3, arm_l=0.35 + (0.8 + 0.15 * math.sin(T * 3)) * reach, shiver=0.3)
        touch = 1 - smoothstep(0.0, 0.6, u)
    elif t < q.c_d14:  # -------------------------------------------- SHOT 8: D13, the empty sea
        u = t - q.c_d13
        e = ease_in_out(clamp(u / (q.c_d14 - q.c_d13)))
        cam = Camera(x=lerp(990, 1080, e), y=lerp(530, 520, e), zoom=lerp(1.02, 1.3, e), t=T)
        pose = lr_pose(f, t, 1.0, lens0, 0.8, eyes='open', look=(0.9, -0.05), head_tilt=-0.03,
                       smile=0.25 + 0.1 * smoothstep(q.d13[1] - 0.8, q.d13[1], t))
        g = palm_guang_pos(pose, 0)
        gk = dict(eyes='sleepy', look=(0.4, -0.7), rot=-0.3)
        touch = 0.0
    else:  # ---------------------------------------------------------- SHOT 9: D14 close-up
        u = t - q.c_d14
        e = ease_in_out(clamp(u / (q.hout - q.c_d14)))
        cam = Camera(x=lerp(1185, 1190, e), y=lerp(545, 530, e), zoom=lerp(2.45, 2.7, e), t=T)
        pose = lr_pose(f, t, 1.0, lens0, 0.8, eyes='happy', look=(0.6, 0.55), head_tilt=0.1, smile=0.75,
                       blush=0.35 * smoothstep(q.d14[0] + 1.5, q.d14[1], t))
        g = palm_guang_pos(pose, 105)
        g = (g[0] + 10 * s, g[1])
        gk = dict(eyes='teary' if t < q.d14[1] else 'happy', look=(-0.5, -0.7), rot=-0.2, tears=0.5,
                  smile=0.2 if t < q.d14[1] else 0.5, arm_l=0.3)
        hp *= 1.1
        touch = 0.0
    pose['heart'] = clamp(hp)
    cradle(pose, g)
    hx, hy = robot_anchor(pose, 'heart', T)
    if touch > 0:
        tr = pose['arm_r_override']
        pose['arm_r_override'] = (lerp(tr[0], hx + pose['facing'] * 12 * s, touch), lerp(tr[1], hy + 4 * s, touch))
    lens, _ = lamp_room(f, c, cam, q, lamp=1.0, pointing_up=1.0, wheel_rot=-2.6, beam=beam)
    with cam.apply(c, 1.0):
        L = lamp_fill(lens, 0.3) + [K.light_radial(hx, hy, 260 * s, WARM2, 0.12 * hp)]
        res = K.draw_robot_lit(c, pose, T, L, gain=0.6)
        draw_guang(c, f, g[0], g[1], s * 0.8, gl, gfl, halo=0.6, rim=0.5, rim_color=WARM2, rim_dir=math.pi / 2,
                   **gk)
        if q.touch <= t < q.g08[0]:  # heart light spilling through the glass under his hand
            hh = res.get('heart', (hx, hy))
            K.call(fx.glow, c, hh[0], hh[1], 120 * s, color=WARM2, intensity=0.4 * hp)
        if t >= q.c_d14:  # a few warm motes around the two of them
            K.call(fx.dust_motes, c, T, rect=(900, 300, 600, 500), count=22, color='#ffe9b0', intensity=0.3, seed=61)
    f.grade.update(bloom=1.3, vignette=0.58, tint='#ffd9a0', tint_amt=0.1 if t >= q.c_d14 else 0.07,
                   sat=0.92 if (q.touch <= t < q.c_d13) else 1.0)


# ----------------------------------------------------------------------------
# SHOT 10 — the heart: out of the chest, into the lamp
# ----------------------------------------------------------------------------
def shot_heart(f, c, t, q):
    T = f.T
    s = DENG_LR['scale']
    pw = deng_power(t, q)
    sag = smoothstep(q.lift, q.lift + 1.2, t)
    core_in = smoothstep(q.hin, q.hin + 0.15, t)
    gather = smoothstep(q.hin, q.ign, t)
    if t < q.c_in:  # ------------------------------------------ 10a: medium, porthole opens
        e = ease_in_out(clamp((t - q.hout) / 1.6))
        cam = Camera(x=lerp(1060, 1075, e), y=lerp(590, 600, e), zoom=lerp(1.55, 1.68, e), t=T)
    else:  # --------------------------------------------------- 10b: closer on the lens
        e = ease_in_out(clamp((t - q.c_in) / (q.ign - q.c_in)))
        cam = Camera(x=lerp(1030, 1000, e), y=lerp(540, 505, e), zoom=lerp(2.0, 2.45, ease_in(e)),
                     shake=9 * gather ** 2, t=T)
    lens, _ = lamp_room(f, c, cam, q, lamp=1.0, pointing_up=1.0, wheel_rot=-2.6, beam=0.5 * (1 - core_in),
                        core=core_in * (1 + gather), core_hot=gather)
    lx, ly, lr = lens
    closed = dropout(t, 5, 6.0, 0.15) < 0.5
    pose = lr_pose(f, t, -1.0, lens, 1.0 + gather, power=pw,
                   heart_open=ease_out_back(clamp((t - q.hout) / 0.45), 1.2), heart_present=t < q.lift,
                   heart=1.0, lean=0.1 * sag + 0.02 * math.sin(T * 0.8), head_dy=14 * sag, head_tilt=0.16 * sag,
                   smile=0.35 - 0.2 * sag, shake=0.4 * sag,
                   eyes='happy' if t < q.lift else ('closed' if closed else 'sleepy'),
                   look=(0.0, 0.8) if t < q.lift + 0.3 else (-0.6, -0.2))
    hx, hy = robot_anchor(pose, 'heart', T)
    g = palm_guang_pos(pose, -5)
    g = (g[0] + 60 * s, g[1] + 15 * s)      # far hand holds Guang a little aside
    ho = q.hout
    if t < ho + 0.3:
        tr = (hx - 70 * s, hy + 60 * s)
    elif t < q.lift:
        e = ease_in_out((t - ho - 0.3) / (q.lift - ho - 0.3))
        tr = (lerp(hx - 70 * s, hx - 6 * s, e), lerp(hy + 60 * s, hy, e))
    elif t < q.lift + 0.45:
        e = ease_out((t - q.lift) / 0.45)
        tr = (lerp(hx - 6 * s, hx - 95 * s, e), lerp(hy, hy - 70 * s, e))
    elif t < q.hin:
        e = ease_in_out((t - q.lift - 0.45) / (q.hin - q.lift - 0.45))
        tr = (lerp(hx - 95 * s, lx + 10, e), lerp(hy - 70 * s, ly + 22, e))
    else:
        e = ease_in_out((t - q.hin - 0.1) / 0.5)
        tr = (lerp(lx + 10, lx + 80, e), lerp(ly + 22, ly + 100, e))
    pose.update(arm_r_override=tr, arm_l_override=(g[0] + 28 * s, g[1] + 42 * s),
                hand_r_open=0.35 if q.lift <= t < q.hin else 0.85, hand_l_open=0.9)
    with cam.apply(c, 1.0):
        res = K.draw_robot_lit(c, pose, T, lamp_fill(lens, 0.35 + 0.8 * gather), gain=0.6)
        draw_guang(c, f, g[0], g[1], s * 0.8, guang_glow(t, q) + 0.25 * gather, guang_flicker(t, q),
                   eyes='teary' if t < q.hin else 'surprised', look=(-0.8, -0.5), tears=0.5, rot=0.3,
                   arm_l=0.6, halo=0.6, rim=0.6, rim_color=WARM2, rim_dir=math.pi)
        if q.lift <= t < q.hin:  # the core in his hand
            hpos = res.get('r', tr)
            cx, cy = hpos[0] - 2, hpos[1] - 18 * s
            if hasattr(robot, 'draw_heart_core'):
                K.call(fx.glow, c, cx, cy, 200 * s, color=WARM2, intensity=0.45)
                K.call(robot.draw_heart_core, c, cx, cy, scale=s * 1.15, brightness=1.0, t=T)
            else:
                K.call(fx.heart_core, c, T, cx, cy, r=18 * s, intensity=1.0, kind='flame')
            K.call(fx.embers, c, T, cx, cy, count=18, color='#ffb060', seed=12, intensity=0.9, spread=24 * s,
                   rise=150 * s)
        if gather > 0:  # energy gathering into the lens (implosion of light motes)
            n = 26
            for i in range(n):
                a = i * 2.39996 + 0.4
                ph = (T * 0.9 + i / n) % 1.0
                rr = lr * (2.6 - 2.4 * ph)
                K.glint(c, lx + math.cos(a) * rr, ly + math.sin(a) * rr, 10 + 14 * ph, '#fff0c0', gather * ph, rot=a)
            K.call(fx.god_rays, c, T, lx, ly, count=14, length=lr * 5, color='#fff0c0',
                   intensity=0.6 * gather ** 1.5, seed=13)
    suck = pulse(t, q.ign - 0.18, q.ign - 0.02, 0.06)
    f.grade.update(bloom=1.3 + 0.7 * gather, vignette=0.58 + 0.1 * sag, tint='#ffd9a0', tint_amt=0.08,
                   sat=1.0 - 0.15 * sag, exposure=1.0 + 0.2 * gather ** 2 - 0.35 * suck)


# ----------------------------------------------------------------------------
# SHOT 11 — IGNITION (wide exterior)
# ----------------------------------------------------------------------------
def pillar_state(t, q):
    u = t - q.ign
    birth = ease_out(clamp(u / 0.65))
    burst = math.exp(-max(0.0, u) * 3.0)
    width = 140 + 120 * burst
    inten = 1.15 + 0.45 * burst + 0.06 * math.sin(t * 9)
    fade = smoothstep(q.fade, q.dark + 0.2, t)
    return birth, width * (1 - 0.8 * fade), inten * (1 - fade)


def shot_ignite(f, c, t, q):
    T = f.T
    u = t - q.ign
    birth, pw, pi_ = pillar_state(t, q)
    e = ease_in_out(clamp((u - 0.45) / 1.75))
    cam = Camera(x=960, y=lerp(330, -950, e), zoom=lerp(0.76, 0.58, e), shake=42 * math.exp(-u * 2.6) + 4, t=T)
    guang = None
    if t >= q.rise - 0.1:
        vx, vy = guang_v(t, q)
        guang = dict(x=guang_x(t, q), y=guang_y(t, q), scale=0.3, glow=guang_glow(t, q),
                     flicker=guang_flicker(t, q), eyes='surprised', rot=0.6 * (t - q.rise), trail=1.0, vx=vx, vy=vy)
    exterior(f, c, cam, q, dawn=dawn_at(t), lamp=1.0, beam_up=1.0, pillar=pi_, birth=birth, pillar_w=pw,
             ring=ease_out(clamp((u - 0.05) / 1.8)), cloud_glow=1.2 * birth, shock=u, sea_light=1.0, guang=guang,
             cloud_alpha=0.5)
    lsx, lsy = cam.to_screen(*HATCH)
    K.call(fx.god_rays, c, T, lsx, lsy, count=16, length=1700, color='#fff0c0',
           intensity=0.55 * math.exp(-u * 1.6), seed=31)
    K.call(fx.lens_flare, c, lsx, lsy, intensity=0.7 * math.exp(-u * 1.3) + 0.15, color='#ffe7b0', t=T)
    white = 0.82 * math.exp(-u * 4.5) if u >= 0 else 0.0
    f.grade.update(white=white, bloom=2.3 - 0.6 * smoothstep(0.3, 2.2, u), exposure=1.04, vignette=0.4,
                   tint='#ffe2b0', tint_amt=0.06)


# ----------------------------------------------------------------------------
# SHOT 12 — RISE: Guang floats up the pillar, G09, pull back to the tiny lighthouse
# ----------------------------------------------------------------------------
def shot_rise(f, c, t, q):
    T = f.T
    u = t - q.c_rise
    gx, gy = guang_x(t, q), guang_y(t, q)
    birth, pw, pi_ = pillar_state(t, q)
    pb_start = q.g09[0] + 0.55
    pb = ease_in_out(clamp((t - pb_start) / (q.whale - 0.1 - pb_start)))
    z_close = lerp(3.1, 2.8, clamp(u / 1.4))
    z_far = 760.0 / (LH_Y - guang_y(q.whale, q))
    zoom = K.loglerp(z_close, z_far, pb)
    far_c = (LH_X, LH_Y - 410.0 / z_far)
    near_c = (gx - 20.0, gy + 12.0)
    cam = Camera(x=lerp(near_c[0], far_c[0], pb), y=lerp(near_c[1], far_c[1], pb), zoom=zoom,
                 shake=2.0 * (1 - pb), t=T)
    gs = lerp(0.62, 1.05, pb)
    ld = smoothstep(q.g09[0] + 0.2, q.g09[0] + 0.7, t)
    spin = 0.9 * (t - q.rise) * (1 - ld)
    rot = lerp((spin + math.pi) % TAU - math.pi, 0.25, ld)
    vx, vy = guang_v(t, q)
    guang = dict(x=gx, y=gy, scale=gs, glow=guang_glow(t, q), flicker=guang_flicker(t, q),
                 eyes='happy' if ld < 0.5 else 'teary', look=(0.0, lerp(-0.6, 1.0, ld)), rot=rot, trail=1.0,
                 vx=vx * lerp(0.3, 1.0, pb), vy=vy * lerp(0.3, 1.0, pb), arm_r=lerp(0.3, -0.5, ld) + 0.15 * ld * math.sin(T * 4),
                 arm_l=lerp(0.3, 0.2, ld), tears=0.6 * ld, smile=lerp(0.6, -0.1, ld), rim=0.6, rim_color=GOLD,
                 rim_dir=math.pi)
    # the pillar is dimmed in the close-up so she reads against it; full strength once far
    exterior(f, c, cam, q, dawn=dawn_at(t), lamp=1.0, beam_up=1.0, pillar=pi_ * lerp(0.45, 1.0, pb ** 0.7),
             birth=birth, pillar_w=pw, ring=1.0, cloud_glow=1.2, sea_light=1.0, guang=guang, cloud_alpha=0.5)
    with cam.apply(c, 1.0):  # light flowing up the pillar around her
        vis = K.visible(cam, 1.0, 0)
        K.light_streaks(c, T, LH_X - pw * 0.8, LH_X + pw * 0.8, vis[1], vis[1] + vis[3], speed=1500.0,
                        intensity=0.22 * (1 - pb), length=260 / max(zoom, 0.5), width=3.0 / max(zoom, 0.3))
        K.call(fx.sparkles, c, T, gx, gy + 60, radius=90, count=10, color='#fff4c0', seed=91,
               intensity=0.8 * (1 - pb), rise=-160.0, size=0.5)
    f.grade.update(bloom=1.7 - 0.2 * pb, vignette=0.45, tint='#ffe2b0', tint_amt=0.05 * (1 - pb))


# ----------------------------------------------------------------------------
# SHOTS 13 & 15 — the whale high above; the pillar fades, Guang gathered up
# ----------------------------------------------------------------------------
def shot_sky(f, c, t, q, fade_shot=False):
    T = f.T
    birth, pw, pi_ = pillar_state(t, q)
    if not fade_shot:
        e = ease_in_out(clamp((t - q.whale) / 1.0))
        cam = Camera(x=960, y=lerp(-2150, -2300, e), zoom=lerp(0.52, 0.5, e), t=T)
    else:
        e = ease_in_out(clamp((t - q.c_fade) / (q.dark - q.c_fade)))
        cam = Camera(x=lerp(1010, 980, e), y=lerp(-2380, -2460, e), zoom=lerp(0.64, 0.7, e), t=T)
    swirl = smoothstep(q.whale - 0.4, q.whale + 1.2, t) * (1 - 0.5 * smoothstep(q.dark, q.end, t))
    gather = smoothstep(q.fade + 0.05, q.fade + 0.6, t)
    vx, vy = guang_v(t, q)
    guang = dict(x=guang_x(t, q), y=guang_y(t, q), scale=1.05, glow=guang_glow(t, q), flicker=0.0,
                 eyes='happy' if gather > 0.5 else 'open', look=(0.4, -0.6), rot=0.3 * math.sin(T * 0.8),
                 trail=1.0 - gather, vx=vx, vy=vy, smile=0.6, arm_l=0.7, arm_r=0.7, gather=gather)
    lampv = 1.0 - smoothstep(q.fade, q.dark, t)
    exterior(f, c, cam, q, dawn=dawn_at(t), lamp=lampv, beam_up=1.0, pillar=pi_ * 0.65, birth=birth,
             pillar_w=pw, ring=1.0, cloud_glow=1.2 * lampv, swirl=swirl, whale=True, guang=guang,
             cloud_alpha=0.5)
    if fade_shot and gather > 0 and '_pos' in guang:
        with cam.apply(c, 1.0):
            gp = guang['_pos']
            burst = pulse(t, q.fade + 0.45, q.fade + 0.6, 0.25)
            K.call(fx.sparkles, c, T, gp[0], gp[1], radius=170, count=26, color='#dff0ff', seed=77,
                   intensity=burst, rise=40.0, size=1.2)
            K.call(fx.glow, c, gp[0], gp[1], 300, color='#cfe8ff', intensity=0.5 * burst)
    f.grade.update(bloom=1.35, vignette=0.45, tint='#cfe0ff', tint_amt=0.05)


# ----------------------------------------------------------------------------
# SHOTS 14 & 16 — Deng in the lamp room: the weak wave; the eyes go dark
# ----------------------------------------------------------------------------
def shot_deng_end(f, c, t, q, dark=False):
    T = f.T
    pw = deng_power(t, q)
    birth, pwid, pi_ = pillar_state(t, q)
    lampv = 1.0 - smoothstep(q.fade, q.dark, t)
    lens0 = (960.0, 470.0, 170.0)
    if not dark:
        blaze = pi_ / 1.3
        e = ease_in_out(clamp((t - q.c_d15) / (q.c_fade - q.c_d15)))
        cam = Camera(x=lerp(1050, 1070, e), y=lerp(520, 540, e), zoom=lerp(1.42, 1.55, e), t=T)
        wave = pulse(t, q.d15[0] + 0.7, q.d15[1] + 0.6, 0.4)
        pose = lr_pose(f, t, -1.0, lens0, 1.6, power=pw, heart_open=1.0, heart_present=False,
                       eyes='happy', look=(-0.3, -1.0), head_tilt=-0.2, smile=0.5, lean=-0.05, head_dy=8,
                       shake=0.25, wave=0.55 * wave, wave_side='r', wave_speed=0.45,
                       arm_l=(0.2, 0.3), hand_l_open=0.4)
    else:
        blaze = 0.0
        u = t - q.dark
        slump = ease_in_out(clamp((u - 0.45) / 0.9))
        cam = Camera(x=lerp(1060, 1050, u / 1.6), y=lerp(590, 600, u / 1.6), zoom=lerp(1.5, 1.56, u / 1.6), t=T)
        pose = lr_pose(f, t, -1.0, lens0, 0.0, power=pw, heart_open=1.0, heart_present=False,
                       eyes='closed' if u > 0.55 else 'sad', look=(-0.2, 0.3),
                       lean=lerp(0.0, 0.32, slump), head_dy=lerp(8, 34, slump), head_tilt=lerp(0.0, 0.35, slump),
                       sit=0.55 * ease_in(clamp((u - 0.8) / 0.8)), sit_dangle=0.0, smile=0.2,
                       arm_r=(lerp(0.4, 1.3, slump), lerp(0.3, 0.8, slump)), arm_l=(lerp(0.25, 0.1, slump), 0.2),
                       hand_r_open=0.8, hand_l_open=0.6, rim=0.3)
        pose['x'] = DENG_LR['x'] - 25 * slump
    lens, _ = lamp_room(f, c, cam, q, lamp=lampv, pointing_up=1.0, wheel_rot=-2.6, beam=0.0, blaze=blaze,
                        core=lampv * (1.2 if not dark else 0.3), core_hot=0.4 * lampv)
    with cam.apply(c, 1.0):
        K.draw_robot_lit(c, pose, T, lamp_fill(lens, 1.4 * blaze + 0.2) if not dark else [])
        if not dark:
            K.call(fx.dust_motes, c, T, rect=(lens[0] - 400, 0, 800, 800), count=36, color='#fff0c0',
                   intensity=0.5, seed=64)
    if not dark:
        f.grade.update(bloom=1.7, vignette=0.5, tint='#ffe2b0', tint_amt=0.08, exposure=1.03)
    else:
        f.grade.update(bloom=1.0, vignette=0.65, sat=0.75, exposure=0.85, tint='#a8b8e8', tint_amt=0.1)


# ----------------------------------------------------------------------------
# SHOT 17 — HOLD on the dark lighthouse
# ----------------------------------------------------------------------------
def shot_hold(f, c, t, q):
    T = f.T
    e = ease_in_out(clamp((t - q.c_hold) / 3.0))
    cam = Camera(x=960, y=lerp(560, 520, e), zoom=lerp(1.0, 1.06, e), t=T)
    exterior(f, c, cam, q, dawn=dawn_at(t), lamp=0.0, beam_up=1.0 - smoothstep(q.c_hold, q.end, t),
             cloud_alpha=0.45, stars_bright=0.85)
    f.grade.update(bloom=0.9, vignette=0.6, sat=0.8, exposure=0.85, tint='#a8b8e8', tint_amt=0.1)


# ----------------------------------------------------------------------------
def render(f):
    c = f.canvas
    t = f.t
    q = Q(f)
    c.drawColor(col('#05070f'))
    if t < q.run:
        shot_idea(f, c, t, q)
    elif t < q.aim:
        shot_run(f, c, t, q)
    elif t < q.bup:
        shot_aim(f, c, t, q)
    elif t < q.d12[0]:
        shot_weak(f, c, t, q)
    elif t < q.hout:
        shot_lamproom_drama(f, c, t, q)
    elif t < q.ign:
        shot_heart(f, c, t, q)
    elif t < q.c_rise:
        shot_ignite(f, c, t, q)
    elif t < q.whale:
        shot_rise(f, c, t, q)
    elif t < q.c_d15:
        shot_sky(f, c, t, q, fade_shot=False)
    elif t < q.c_fade:
        shot_deng_end(f, c, t, q, dark=False)
    elif t < q.dark:
        shot_sky(f, c, t, q, fade_shot=True)
    elif t < q.c_hold:
        shot_deng_end(f, c, t, q, dark=True)
    else:
        shot_hold(f, c, t, q)
