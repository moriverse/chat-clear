"""s02 · 阿灯的夜晚 Deng's evening — 27 s.  Owner: SCENE-S02.

Shots (all timing from the timeline; cuts are instant, at the cue times):
  1. climb      0 → s02_arrive           spiral stairs, oil can, chest light on the walls
  2. lamp room  s02_arrive → s02_exterior  D01: greets the sea, sets the can down, polishes
                                           the lens, pulls the lever; lamp_ignite: flash, lens
                                           turns, warm flood, dust motes, warm rim on Deng
  3. exterior   s02_exterior → s02_sit     wide lighthouse, sweeping beam, N04;
                                           timelapse_start→end: 300 years — star trails around
                                           the pole, the beam blurs into a disc, clouds streak,
                                           nights flicker; eases back to real time
  4. alone      s02_sit → end              Deng on the gallery lip, chin in hands, beam
                                           passing behind him, D02 (sad), a sigh, pull back
"""
from __future__ import annotations

import math

import skia

from engine.core import (ADD, H, TAU, W, Camera, _hash, at, clamp, ease_in, ease_in_out, ease_out,
                         ease_out_back, fill, lerp, linear, mix, noise1, poly, pulse, rrect, smoothstep,
                         soft_glow, stroke)
from lib import fx, robot, sea, sky
from scenes import _s02_kit as kit

WALK_CADENCE = 1.0          # cycles / s (shared convention with SFX)
BIG = skia.Rect.MakeLTRB(-20000, -20000, 20000, 20000)
WARM = '#ffd36b'
BEAM = '#ffe7a0'


def render(f):
    t = f.t
    if t < f.cue('s02_arrive'):
        shot_climb(f)
    elif t < f.cue('s02_exterior'):
        shot_lamp_room(f)
    elif t < f.cue('s02_sit'):
        shot_exterior(f)
    else:
        shot_alone(f)


def _walk_speed(s):
    if hasattr(robot, 'walk_speed'):
        return robot.walk_speed(s, WALK_CADENCE)
    return 60.0 * s * WALK_CADENCE


def _draw_deng(c, kw, t):
    return robot.draw_robot(c, kit.pose(**kw), t) or {}


# =============================================================================
# small props (not library assets): oil can, ignition lever
# =============================================================================
CAN_H = 64.0  # handle-top to bottom, at scale 1


def draw_oil_can(c, x, y, s=1.0, rot=0.0, warm=0.6, facing=1.0, ambient=1.0):
    """Little brass oil can hanging from its handle top at (x, y)."""
    k = ambient
    with at(c, x, y, rot=rot, sx=s * facing, sy=s):
        hp = skia.Path()
        hp.addArc(skia.Rect.MakeLTRB(-13, 1, 13, 27), 200, 140)
        c.drawPath(hp, stroke(_dim('#7a6038', k), 5))
        body = skia.Path()
        body.moveTo(-22, 28)
        body.cubicTo(-26, 40, -28, 54, -26, 62)
        body.lineTo(26, 62)
        body.cubicTo(28, 54, 26, 40, 22, 28)
        body.cubicTo(14, 20, -14, 20, -22, 28)
        body.close()
        c.drawPath(body, fill(_dim('#9c7a45', k)))
        c.drawPath(body, linear((-26, 0), (26, 0), [(0, '#4a3620', 0.0), (0.55, '#4a3620', 0.0), (1, '#4a3620', 0.7)]))
        c.drawRRect(rrect(-27, 56, 54, 8, 3), fill(_dim('#6e5230', k)))
        c.drawOval(skia.Rect.MakeXYWH(-19, 32, 9, 22), fill('#ffd9a0', 0.35 * warm))
        c.drawPath(poly([(18, 34), (52, 6), (56, 9), (24, 42)]), fill(_dim('#8a6a3c', k)))
        c.drawCircle(54, 7, 3.2, fill(_dim('#c9a15b', k)))
        c.drawRRect(rrect(-7, 18, 14, 7, 3), fill(_dim('#c9a15b', k)))


def _dim(h, k):
    return mix('#000000', h, clamp(k)) if k < 1 else h


def draw_lever(c, pivot, L, angle, s, floor_y, lamp=0.0, ambient=1.0):
    """Brass ignition lever on an iron console. Returns the knob position."""
    px, py = pivot
    k = ambient
    # console
    c.drawRRect(rrect(px - 26 * s, py - 6 * s, 52 * s, floor_y - py + 6 * s, 10 * s), fill(_dim('#2a2f45', k)))
    c.drawRRect(rrect(px - 26 * s, py - 6 * s, 12 * s, floor_y - py + 6 * s, 6 * s), fill(_dim('#3e4668', k), 0.8))
    c.drawRRect(rrect(px - 40 * s, floor_y - 14 * s, 80 * s, 16 * s, 6 * s), fill(_dim('#232839', k)))
    c.drawCircle(px, py, 30 * s, fill(_dim('#3a3450', k)))
    c.drawCircle(px, py, 25 * s, fill(_dim('#8a6a3a', k)))
    # status lamp on the console
    c.drawCircle(px + 12 * s, py + 40 * s, 6 * s, fill('#ffd36b' if lamp > 0 else _dim('#4a2a22', k)))
    if lamp > 0:
        soft_glow(c, px + 12 * s, py + 40 * s, 30 * s, '#ffd36b', 0.6 * lamp)
    kx, ky = px + math.cos(angle) * L, py + math.sin(angle) * L
    c.drawLine(px, py, kx, ky, stroke(_dim('#b08a4a', k), 10 * s))
    c.drawLine(px, py, kx, ky, stroke('#f0d49a', 2.5 * s, 0.45 * k))
    c.drawCircle(px, py, 10 * s, fill(_dim('#c9a15b', k)))
    c.drawCircle(kx, ky, 15 * s, fill(_dim('#b8413a', k)))
    c.drawCircle(kx - 4 * s, ky - 5 * s, 5 * s, fill('#ff8a7a', 0.55 * k))
    return kx, ky


# =============================================================================
# 1. CLIMB
# =============================================================================
CLIMB_STEPS_PER_CYCLE = 2      # a foot lands on a new step every half cycle
CLIMB_SCALE = 1.0
CLIMB_Y = 880.0                # screen y where Deng's feet stay (the view rises with him)
_STAIR = {}


def _stairs_probe():
    """Probe the lib stairs geometry once: step_pos at scroll 0, the scroll response, and the
    run of steps that stays on the front of the spiral and on screen."""
    if 'k0' in _STAIR:
        return _STAIR
    rec = skia.PictureRecorder()
    cc = rec.beginRecording(BIG)
    sp0 = kit.call(sea.stairs_interior, cc, 0.0, scroll=0.0)
    sp1 = kit.call(sea.stairs_interior, cc, 0.0, scroll=100.0)
    rec.finishRecordingAsPicture()
    dy = (sp1(3.0)[1] - sp0(3.0)[1]) / 100.0 or 1.0
    n = int(2 * (4.2 * CLIMB_STEPS_PER_CYCLE * WALK_CADENCE + 1))
    best = None
    for i in range(0, 161):
        k0 = i * 0.25
        ps = [sp0(k0 - 0.5 + j * 0.5) for j in range(n + 1)]
        on = all(260 < p[0] < W - 260 for p in ps)
        score = min(p[2] for p in ps) + (0.0 if on else -1.0)
        if best is None or score > best[0] + 1e-6:
            best = (score, k0)
    _STAIR.update(sp0=sp0, dy=dy, k0=best[1])
    return _STAIR


def shot_climb(f):
    c = f.canvas
    t = f.t
    st = _stairs_probe()
    sp0, dyf, k0 = st['sp0'], st['dy'], st['k0']
    phase = t * WALK_CADENCE
    kf = phase * CLIMB_STEPS_PER_CYCLE
    k_body = k0 + kf + 0.10 * math.sin(TAU * kf) / TAU      # a touch faster just after each contact
    k_cam = k0 + kf
    scroll = (CLIMB_Y - sp0(k_cam)[1]) / dyf
    x, y, d = sp0(k_body)
    y += scroll * dyf
    facing = 1.0 if sp0(k_body + 0.2)[0] >= x else -1.0
    s = CLIMB_SCALE * max(0.55, d)

    idl = robot.idle(f.T)
    flick = 0.93 + 0.07 * noise1(f.T * 3.0, 5)
    kw = dict(x=x, y=y, scale=s, facing=facing, walk=1.0, walk_phase=phase, lean=0.09 + idl['lean'],
              head_dy=idl['head_dy'], head_tilt=idl['head_tilt'] - 0.04,
              arm_r=(0.1, 0.35), hand_r_open=0.05, eyes='open', blink=robot.auto_blink(f.T, seed=2),
              look=(0.5, -0.55), smile=0.4, scarf_wind=0.2, ambient='#c8b8a8', rim=0.35,
              rim_color='#9fc4ff', rim_angle=-math.pi / 2 - 0.4 * facing, glow_cast=1.3, shadow=0.3,
              heart=flick)
    a = kit.anchors(kw, f.T)
    heart = a.get('heart', (x, y - 190 * s))

    kit.call(sea.stairs_interior, c, f.T, scroll=scroll, light=0.5, light_pos=heart,
             light_color='#ffb347', light_amount=flick)
    if not kit.accepts(sea.stairs_interior, 'light_pos'):
        soft_glow(c, heart[0], heart[1], 760, '#ffb347', 0.30 * flick)
    soft_glow(c, heart[0], heart[1], 300, '#ffc870', 0.16 * flick)

    pts = _draw_deng(c, kw, f.T)
    hr = pts.get('r')
    if hr:
        rot = 0.22 * math.sin(TAU * phase - 0.9) * -facing
        draw_oil_can(c, hr[0], hr[1] - 4 * s, s * 0.9, rot, warm=0.9, facing=facing, ambient=0.85)

    f.grade.update(vignette=0.62, bloom=1.05, tint='#ffcf99', tint_amt=0.05)


# =============================================================================
# 2. LAMP ROOM
# =============================================================================
LR_Y = 960.0          # Deng's floor contact
LR_S = 1.2


def _lr_lamp(u):
    """Lamp brightness after ignition (u = seconds since lamp_ignite): sputter, then bloom."""
    if u < 0:
        return 0.0
    rise = ease_out(u / 0.45)
    sput = math.exp(-u * 3.5) * (0.5 + 0.5 * math.sin(u * 47.0)) * 0.55
    return clamp(rise * (1 - sput))


def _lens_rot(u):
    if u <= 0:
        return 0.0
    r = 1.6
    if u < r:
        v = u / r
        return 0.9 * r * (v ** 3 - v ** 4 / 2)
    return 0.9 * (r / 2 + (u - r))


def _walk_int(u, a, b):
    """Integral of the walk amount (1 until a, smooth fade to 0 at b)."""
    if u <= 0:
        return 0.0
    if u < a:
        return u
    if u < b:
        v = (u - a) / (b - a)
        return a + (b - a) * (v - (v ** 3 - v ** 4 / 2))
    return a + (b - a) / 2


def _lr_layout(lx, ly, lr, s):
    """Where Deng stands so his near hand can polish the lens, and the lever geometry.
    Everything is derived from the lamp_room lens + measured robot anchors."""
    reach = (62.0 + 69.0) * s
    a0 = kit.anchors(dict(x=0.0, y=LR_Y, scale=s, facing=1.0), 0.0)
    sh = a0.get('shoulder_r', (32 * s, LR_Y - 182 * s))
    sy = sh[1]
    py = clamp(max(ly + lr * 0.35, sy - 0.74 * reach), ly - lr * 0.9, ly + lr * 0.95)
    px = lx - math.sqrt(max(0.0, lr * lr - (py - ly) ** 2)) + 8 * s
    dxr = math.sqrt(max(0.0, (0.8 * reach) ** 2 - (sy - py) ** 2))
    x_stop = px - dxr - sh[0]
    shoulder = (x_stop + sh[0], sy)
    pivot = (shoulder[0] + 165 * s / 1.12, shoulder[1] + 72 * s / 1.12)
    return dict(x_stop=x_stop, polish=(px, py), shoulder=shoulder, pivot=pivot, lever_len=100 * s / 1.12,
                reach=reach)


def shot_lamp_room(f):
    c = f.canvas
    t = f.t
    t_arr = f.cue('s02_arrive')
    t_ign = f.cue('lamp_ignite')
    d01s, d01e = f.line('D01')
    u_ign = t - t_ign
    lamp = _lr_lamp(u_ign)
    lens_rot = _lens_rot(u_ign)
    ua = t - t_arr
    s = LR_S

    # ---- camera: gentle push-in on the ritual, small jolt + ease back at ignition ---------
    push = ease_in_out((t - t_arr) / (t_ign - t_arr))
    back = ease_in_out((t - t_ign) / 1.4)
    zoom = max(1.0, lerp(1.0, 1.12, push) - 0.06 * back)
    cx = lerp(W / 2, 860, push) + 50 * back
    cy = lerp(H / 2, 575, push) - 25 * back
    mx, my = (W - W / zoom) / 2, (H - H / zoom) / 2
    cam = Camera(x=clamp(cx, W / 2 - mx, W / 2 + mx), y=clamp(cy, H / 2 - my, H / 2 + my), zoom=zoom,
                 shake=8.0 * math.exp(-max(0.0, u_ign) * 3.0) * (u_ign >= 0), t=f.T)

    with cam.apply(c, 1.0):
        res = kit.call(sea.lamp_room, c, f.T, lamp=lamp, lens_rot=lens_rot)
        try:
            lx, ly, lr = res
        except (TypeError, ValueError):
            lx, ly, lr = 960, 470, 170
        L = _lr_layout(lx, ly, lr, s)

        # ---- beats ------------------------------------------------------------------------
        t_greet0, t_greet1 = d01s, d01s + 1.65
        t_can0, t_can1 = t_greet1 - 0.05, t_greet1 + 0.8           # set the can down
        t_can_rel = (t_can0 + t_can1) / 2
        t_pol0, t_pol1 = t_can1 + 0.05, t_ign - 0.85                # polish the lens
        t_grab = t_ign - 0.34                                       # hand on the lever knob
        mouth = f.mouth('deng')
        idl = robot.idle(f.T, seed=1)

        # ---- walk in (feet do not slide: x advances at walk_speed) ------------------------
        wa, wb = 0.55, 1.05
        v = _walk_speed(s)
        x = L['x_stop'] - v * (_walk_int(1e9, wa, wb) - _walk_int(ua, wa, wb))
        walk_amt = 1.0 - smoothstep(wa, wb, ua)

        crouch = pulse(t, t_can0 + 0.2, t_can1 - 0.2, 0.22)
        greet = pulse(t, t_greet0 - 0.1, t_greet1 - 0.1, 0.3)
        pol = pulse(t, t_pol0, t_pol1, 0.35)
        pull = clamp((t - t_grab) / (t_ign - t_grab))
        joy = ease_out_back(clamp((u_ign - 0.15) / 0.7)) if u_ign >= 0 else 0.0

        lean = 0.05 * walk_amt + 0.26 * crouch + 0.07 * pol + idl['lean']
        squash = 0.10 * crouch
        head_tilt = idl['head_tilt'] + 0.08 * greet + 0.14 * pol
        head_dy = idl['head_dy'] - (2.0 * mouth if f.speaking('deng') else 0.0)
        eyes, look, smile = 'open', (0.55, -0.25), 0.45
        if greet > 0.05:
            look = (0.85, -0.45)
            smile = 0.65
            eyes = 'happy' if pulse(t, t_greet0 + 0.85, t_greet1, 0.05) > 0.5 else 'open'
        if crouch > 0.3:
            look = (0.5, 0.6)
        if pol > 0.3:
            eyes, smile, look = 'happy', 0.8, (0.7, -0.35)
        if t_pol1 < t < t_ign + 0.12:
            eyes, smile, look = 'determined', 0.35, (0.7, 0.0)
            lean += 0.04 * (1 - pull) * smoothstep(t_pol1, t_grab, t) - 0.12 * ease_in(pull)
            squash += 0.06 * ease_in(pull)
        if u_ign >= 0:
            eyes = 'surprised' if u_ign < 0.4 else 'happy'
            smile = lerp(0.4, 0.95, joy)
            look = (0.55, -0.7)
            lean += -0.08 * joy - 0.07 * math.exp(-u_ign * 4.0)
            head_tilt += -0.06 * joy

        ambient = mix('#8f9ccc', '#fff0dc', lamp)
        kw = dict(x=x, y=LR_Y, scale=s, facing=1.0, lean=lean, squash=squash, head_tilt=head_tilt,
                  head_dy=head_dy, walk=walk_amt, walk_phase=t * WALK_CADENCE, eyes=eyes,
                  blink=robot.auto_blink(f.T, seed=3) if eyes in ('open', 'determined') else 0.0,
                  look=look, mouth=mouth, smile=smile, scarf_wind=0.15, heart=1.0,
                  arm_l=(0.15, 0.25), arm_r=(0.12, 0.35), hand_r_open=0.15,
                  ambient=ambient, rim=lerp(0.35, 0.95, lamp), rim_color=mix('#9fc4ff', '#ffd58a', lamp),
                  rim_angle=lerp(-math.pi / 2, -0.35, lamp), glow_cast=1.2, shadow=0.4,
                  wave=greet, wave_side='l', wave_speed=0.9)

        # ---- the lever ----------------------------------------------------------------------
        a_off, a_on = -2.0, -3.3
        if t < t_ign:
            lev_ang = lerp(a_off, a_on, ease_in(pull))
        else:
            lev_ang = a_on + 0.10 * math.exp(-u_ign * 6) * math.sin(u_ign * 32)
        piv, LL = L['pivot'], L['lever_len']
        knob = (piv[0] + math.cos(lev_ang) * LL, piv[1] + math.sin(lev_ang) * LL)

        # ---- near arm: can -> set down -> polish -> lever -> joy -------------------------------
        can_floor = (x + 88 * s, LR_Y + 4)
        can_handle = (can_floor[0], can_floor[1] - CAN_H * s * 0.9)
        target, k_t = None, 0.0
        if t_can0 <= t < t_pol0:
            target, k_t = can_handle, pulse(t, t_can0, t_can1, 0.3)
        elif t_pol0 <= t < t_ign:
            ph = (t - t_pol0) * 2.3
            pr = 20 * s * smoothstep(t_pol0 + 0.25, t_pol0 + 0.6, t)
            pp = L['polish']
            pol_pt = (pp[0] + math.cos(TAU * ph) * pr, pp[1] + math.sin(TAU * ph) * pr * 0.7)
            k_lv = smoothstep(t_pol1 - 0.05, t_grab, t)
            target = (lerp(pol_pt[0], knob[0], k_lv), lerp(pol_pt[1], knob[1], k_lv))
            k_t = smoothstep(t_pol0, t_pol0 + 0.45, t)
        elif t >= t_ign:
            target, k_t = knob, 1.0 - smoothstep(0.25, 0.9, u_ign)
        if target is not None and k_t > 0.001:
            rest = kit.hand_rest(kw, 'r', f.T)
            kw['arm_r_override'] = (lerp(rest[0], target[0], k_t), lerp(rest[1], target[1], k_t))
        if u_ign >= 0:
            kw['hug'] = 0.55 * smoothstep(0.35, 1.0, u_ign)
        kw['hand_r_open'] = 0.1 if (t < t_can_rel or t_grab - 0.05 < t < t_ign + 0.3) else 0.6
        heart = kit.anchors(kw, f.T).get('heart', (x, LR_Y - 190 * s))

        # ---- light in the room (behind Deng) ---------------------------------------------------
        if lamp > 0:
            soft_glow(c, lx, ly, 1500, '#ffcf7a', 0.28 * lamp)
            kit.safe(fx.god_rays, c, f.T, lx, ly, count=14, length=1500, color='#fff0c0',
                     intensity=0.30 * lamp, angle=lens_rot * 0.5, start=lr * 0.8, source=False)
        soft_glow(c, heart[0], heart[1], 420, '#ffb347', 0.16 * (1 - 0.6 * lamp))
        if lamp > 0:  # long soft shadow thrown away from the lens
            c.save()
            c.translate(x, LR_Y)
            c.skew(1.5, 0)
            c.scale(1.0, 0.2)
            c.drawOval(skia.Rect.MakeXYWH(-85 * s, -340 * s, 170 * s, 340 * s), fill('#05070f', 0.30 * lamp, blur=16))
            c.restore()

        cam_amb = mix('#000000', '#ffffff', lerp(0.7, 1.0, lamp))
        draw_lever(c, piv, LL, lev_ang, s, LR_Y - 26, lamp, ambient=lerp(0.7, 1.0, lamp))
        released = t >= t_can_rel
        if released:
            draw_oil_can(c, can_handle[0], can_handle[1], s * 0.9, 0.0, warm=0.4 + 0.6 * lamp,
                         ambient=lerp(0.75, 1.0, lamp))

        # ---- Deng ---------------------------------------------------------------------------------
        pts = _draw_deng(c, kw, f.T)
        hand_r = pts.get('r', (x, LR_Y - 150 * s))
        if not released:
            rot = 0.18 * math.sin(TAU * t * WALK_CADENCE - 0.9) * walk_amt
            draw_oil_can(c, hand_r[0], hand_r[1] - 4 * s, s * 0.9, -rot, warm=0.8, ambient=0.8)

        # polishing glints on the glass
        if t_pol0 + 0.35 < t < t_pol1:
            gl = 0.5 + 0.5 * math.sin(t * 9.0)
            soft_glow(c, hand_r[0] + 12, hand_r[1] - 12, 70, '#dff0ff', 0.30 * gl)
            kit.safe(fx.sparkles, c, f.T, hand_r[0] + 10, hand_r[1] - 20, radius=50, count=6,
                     color='#eaf6ff', seed=21, intensity=0.9, rise=25, size=0.7)
        # dust motes dancing in the new light, a few embers from the burner
        if lamp > 0:
            kit.safe(fx.dust_motes, c, f.T, rect=(lx - 1000, ly - 460, 1700, 980), count=80,
                     color='#ffe9b0', intensity=0.8 * lamp, seed=12, light=(lx, ly, 900))
            kit.safe(fx.embers, c, f.T, lx, ly + lr * 0.2, count=14, color='#ffd08a', seed=13,
                     intensity=0.6 * lamp * math.exp(-max(0.0, u_ign - 0.3) * 1.2), spread=lr * 0.7, rise=260)
        if 0 <= u_ign < 1.3:
            kit.safe(fx.shockwave, c, lx, ly, u_ign, color='#fff4d0', max_r=1300, life=1.1, intensity=0.6)

    # screen-space flare at the moment the lamp catches
    if 0 <= u_ign < 2.5:
        sx, sy = cam.to_screen(lx, ly)
        kit.safe(fx.lens_flare, c, sx, sy, intensity=0.9 * math.exp(-u_ign * 1.8) + 0.15 * lamp,
                 color='#ffe7b0', t=f.T)

    flash = math.exp(-u_ign * 5.5) if u_ign >= 0 else 0.0
    pre = smoothstep(-0.10, 0.0, u_ign) * (u_ign < 0)
    f.grade.update(white=0.55 * flash + 0.2 * pre, bloom=1.0 + 0.4 * lamp + 1.6 * flash,
                   exposure=lerp(0.95, 1.04, lamp) + 0.25 * flash, tint='#ffd9a0', tint_amt=0.10 * lamp,
                   vignette=lerp(0.62, 0.5, lamp), sat=lerp(0.95, 1.05, lamp))


# =============================================================================
# 3. EXTERIOR & TIME-LAPSE
# =============================================================================
EX_HORIZON = 730.0
EX_LH = (800.0, 812.0)       # lighthouse base on the rock
EX_LH_S = 0.86
EX_POLE = (905.0, 175.0)     # pole star (screen space, depth 0)
TL_RAMP_UP = 2.3
TL_RAMP_DN = 2.4
STAR_OMEGA = 2.6             # peak sky rotation (rad/s)
TRAIL_WINDOW = 3.2           # "exposure" of the trails (s)
BEAM_BOOST = 200.0           # nominal peak beam speed-up
CLOUD_BOOST = 350.0


def _env_int(t, a, b, ru, rd):
    """Integral from a to t of the time-lapse envelope (smoothstep up, hold, smoothstep down)."""
    hold = (b - rd) - (a + ru)
    if t <= a:
        return 0.0
    if t < a + ru:
        u = (t - a) / ru
        return ru * (u ** 3 - u ** 4 / 2)
    if t < b - rd:
        return ru / 2 + (t - a - ru)
    if t < b:
        v = (t - (b - rd)) / rd
        return ru / 2 + hold + rd * (v - (v ** 3 - v ** 4 / 2))
    return ru / 2 + hold + rd / 2


def _env(t, a, b, ru, rd):
    return smoothstep(a, a + ru, t) * (1 - smoothstep(b - rd, b, t))


def _beam_boost(a, b):
    tot = _env_int(b + 1, a, b, TL_RAMP_UP, TL_RAMP_DN)
    n = round(0.9 * BEAM_BOOST * tot / TAU)
    return TAU * n / (0.9 * tot)     # whole extra turns -> the beam is back in sync afterwards


def _sky_horizon(cam, world_h, d_ref, d):
    """World y at parallax depth d that lands on the same screen row as world_h at d_ref."""
    hy = cam.to_screen(0, world_h, d_ref)[1]
    z = 1.0 + (cam.zoom - 1.0) * d
    cy = H / 2 + (cam.y - H / 2) * d
    return cy + (hy - H / 2) / z


def shot_exterior(f):
    c = f.canvas
    t = f.t
    T = f.T
    t0 = f.cue('s02_exterior')
    ta, tb = f.cue('timelapse_start'), f.cue('timelapse_end')
    env = _env(t, ta, tb, TL_RAMP_UP, TL_RAMP_DN)
    I = _env_int(t, ta, tb, TL_RAMP_UP, TL_RAMP_DN)
    u0 = t - t0

    # camera: starts close on the lamp, pulls back to reveal, then creeps in during the lapse
    anc = kit.call(sea.lighthouse_anchors, *EX_LH, EX_LH_S) if hasattr(sea, 'lighthouse_anchors') else {}
    lamp_w = anc.get('lamp', (EX_LH[0], EX_LH[1] - 445 * EX_LH_S))
    reveal = ease_in_out(u0 / 2.6)
    creep = ease_in_out((t - ta) / (tb - ta))
    zoom = lerp(1.55, 1.0, reveal) + 0.05 * creep
    cam = Camera(x=lerp(lamp_w[0], W / 2, reveal), y=lerp(lamp_w[1] + 40, H / 2, reveal) - 20 * creep,
                 zoom=zoom, t=T)

    # sky rotation and trail length (a long exposure whose window closes as we slow down)
    theta = STAR_OMEGA * I
    win = TRAIL_WINDOW * (1 - smoothstep(tb - 2.3, tb, t))
    sweep = STAR_OMEGA * (I - _env_int(t - win, ta, tb, TL_RAMP_UP, TL_RAMP_DN))
    hflick = _hash(f.index, 77)
    flick = env * hflick
    pole = EX_POLE

    d_sky, d_cloud, d_sea = 0.08, 0.22, 0.75
    hz_sky = _sky_horizon(cam, EX_HORIZON, d_sea, d_sky)
    hz_cloud = _sky_horizon(cam, EX_HORIZON, d_sea, d_cloud)
    with cam.apply(c, d_sky):
        kit.safe(sky.sky, c, horizon_y=hz_sky, dawn=0.10 * flick * flick)
    # the sky wheel: everything celestial turns about the pole
    with cam.apply(c, 0.0):
        with at(c, pole[0], pole[1], rot=-theta):
            c.translate(-pole[0], -pole[1])
            kit.safe(sky.milky_way, c, T, center=(1180, 300), angle=-0.55, length=3400, width=560,
                     alpha=0.55 * (1 - 0.35 * env), horizon_y=hz_sky + 60)
        if hasattr(sky, 'star_trails'):
            arc = clamp(sweep, 1e-4, TAU * 0.995)
            amt = smoothstep(0.0, 0.25, sweep)
            kit.safe(sky.star_trails, c, T, center=pole, amount=amt, spin=theta, arc=arc, horizon_y=hz_sky,
                     gain=lerp(0.9, 0.24, (arc / TAU) ** 0.7), density=lerp(1.0, 0.38, amt), pole=1.0,
                     fade=520.0)
        else:
            pa = 1.0 - smoothstep(0.0, 0.3, sweep)
            with at(c, pole[0], pole[1], rot=-theta):
                c.translate(-pole[0], -pole[1])
                if pa > 0.01:
                    kit.safe(sky.stars, c, T, bright=pa, horizon_y=hz_sky)
            kit.star_trails(c, pole[0], pole[1], theta, sweep, alpha=smoothstep(0.03, 0.35, sweep),
                            horizon_y=hz_sky)
            soft_glow(c, pole[0], pole[1], 46, '#dfeaff', 0.5)
        # meteors flashing through the centuries
        if env > 0.2:
            for m in range(2):
                if _hash(f.index * 3 + m, 91) < 0.30 * env:
                    mx = 100 + _hash(f.index * 5 + m, 92) * 1700
                    my = 60 + _hash(f.index * 7 + m, 93) * 420
                    ma = 0.35 + _hash(f.index * 11 + m, 94) * 0.9
                    kit.safe(sky.shooting_star_sky_streak, c, mx, my, ma, 120 + 160 * _hash(f.index, 95), 0.7)
    # clouds: drifting, then streaking past in long exposure
    with cam.apply(c, d_cloud):
        kit.safe(sky.clouds, c, t + CLOUD_BOOST * I, y=hz_cloud - 210, alpha=0.22 + 0.08 * env,
                 color='#3a4c86', seed=3, speed=7.0, scale=1.1, layers=1, cover=0.4, streak=env)

    # beam azimuth: global (continuous across cuts), plus whole extra turns during the lapse
    kboost = _beam_boost(ta, tb)
    az = 0.9 * T + 0.9 * kboost * I
    delta = 0.9 * (1 + kboost * env) / 48.0         # rotation during a 180-degree shutter
    disc = smoothstep(0.7, 3.0, delta)
    blen, bw0, bw1 = 2700, 30, 430

    with cam.apply(c, d_sea):
        lights = [(lamp_w[0], lamp_w[1], WARM, 0.5 + 0.8 * disc)]
        if disc < 0.95 and hasattr(sea, 'beam_light'):
            lights.append(kit.call(sea.beam_light, lamp_w[0], lamp_w[1], az, horizon_y=EX_HORIZON,
                                   length=blen, intensity=1.0 - disc, width0=bw0, width1=bw1))
        kit.call(sea.ocean, c, T, horizon_y=EX_HORIZON, lights=lights, calm=1.0 - 0.8 * env)
    with cam.apply(c, 1.0):
        kit.call(sea.island, c, EX_LH[0], EX_LH[1], 0.78, t=T)
        res = kit.call(sea.lighthouse, c, EX_LH[0], EX_LH[1], EX_LH_S, lamp=1.0, t=T,
                       beam_angle=az if disc < 0.5 else None)
        try:
            lx, ly = res
        except (TypeError, ValueError):
            lx, ly = lamp_w
        # the beam, sampled over the shutter; it becomes a luminous disc as it spins up
        if disc < 0.99:
            kit.call(sea.beam, c, lx, ly, az, length=blen, width0=bw0, width1=bw1, intensity=1.0 - disc,
                     horizon_y=EX_HORIZON, t=T)
            m = int(clamp(math.ceil(delta / 0.12), 1, 10))
            for k in range(1, m):
                kit.beam_ghost(c, lx, ly, az - delta * k / (m - 1), (1 - disc) * 0.9 / m ** 0.6,
                               blen, bw0, bw1, EX_HORIZON)
        if disc > 0.01:
            kit.beam_disc(c, lx, ly, 1650, 62, intensity=0.9 * disc, droop=4)
            kit.beam_disc(c, lx, ly, 700, 34, intensity=0.5 * disc, droop=2)
            soft_glow(c, lx, ly, 120, '#fff4d8', 0.4 * disc)

    f.grade.update(vignette=0.5, bloom=1.0 + 0.2 * env, exposure=1.0 + 0.07 * (hflick - 0.5) * env,
                   sat=1.0 + 0.06 * env)


# =============================================================================
# 4. ALONE ON THE GALLERY
# =============================================================================
AL_HORIZON = 600.0
AL_S = 1.15                # Deng's scale
AL_LH_X = 90.0             # lighthouse axis (lamp room at the left edge)
AL_GALLERY_Y = 735.0       # where the gallery floor sits on screen (world, depth 1)
AL_PHI = 1.08              # where on the gallery ring Deng sits (0 = front centre, + = right)
AL_VIEW_E = 0.04


def shot_alone(f):
    c = f.canvas
    t = f.t
    T = f.T
    t0 = f.cue('s02_sit')
    d02s, d02e = f.line('D02')

    # lighthouse scaled so Deng fits its gallery (lib: deng_scale = 0.10 * s)
    lh_s = AL_S / 0.10
    lh_y = AL_GALLERY_Y + 400.0 * lh_s
    anc = kit.call(sea.lighthouse_anchors, AL_LH_X, lh_y, lh_s) if hasattr(sea, 'lighthouse_anchors') else {}
    gal_y = anc.get('gallery_y', AL_GALLERY_Y)
    deck_r = sea.LH['deck_r'] * lh_s if hasattr(sea, 'LH') else 690.0
    seat_x = AL_LH_X + deck_r * math.sin(AL_PHI) * 0.98
    seat_y = gal_y - AL_VIEW_E * deck_r * math.cos(AL_PHI) + 2

    # camera: settle, slow push during the line, then drift back to leave him small
    push = ease_in_out((t - t0) / (d02e - t0))
    back = ease_in_out((t - (d02e + 0.5)) / (f.dur - d02e - 0.3))
    zoom = lerp(1.10, 1.17, push) - 0.24 * back
    cam = Camera(x=lerp(seat_x - 20, seat_x - 40, push) + 90 * back,
                 y=lerp(seat_y - 170, seat_y - 180, push) - 60 * back, zoom=zoom, t=T)

    az = 0.9 * T
    d_sky, d_sea = 0.05, 0.3
    hz_sky = _sky_horizon(cam, AL_HORIZON, d_sea, d_sky)
    with cam.apply(c, d_sky):
        kit.safe(sky.sky, c, horizon_y=hz_sky)
        kit.safe(sky.milky_way, c, T, center=(1300, 250), angle=-0.62, length=3600, width=600, alpha=0.7,
                 horizon_y=hz_sky)
        kit.safe(sky.stars, c, T, horizon_y=hz_sky, bright=1.0)
    with cam.apply(c, d_sea):
        kit.call(sea.ocean, c, T, horizon_y=AL_HORIZON, calm=0.8,
                 lights=[(AL_LH_X, AL_HORIZON - 30, WARM, 0.25)])

    # sigh / acting curves
    hope = pulse(t, d02s - 0.2, d02s + 1.5, 0.35)            # "要是有人，" lifts his head
    glance = pulse(t, d02s + 1.75, d02s + 2.95, 0.3)         # "能看见这束光，" looks back at the light
    inhale = pulse(t, d02e + 0.25, d02e + 0.8, 0.25)
    slump = smoothstep(d02e + 0.65, d02e + 1.7, t)          # the sigh: shoulders drop, head lowers

    with cam.apply(c, 1.0):
        lamp_xy = anc.get('lamp', (AL_LH_X, gal_y - 45 * lh_s))
        res = kit.call(sea.lighthouse, c, AL_LH_X, lh_y, lh_s, lamp=0.62, t=T, beam_angle=az,
                       view_e=AL_VIEW_E, glow=0.35)
        try:
            lx, ly = res
        except (TypeError, ValueError):
            lx, ly = lamp_xy
        facing_cam = kit.call(sea.beam, c, lx, ly, az, length=3400, width0=36 * lh_s / 10, width1=1100,
                              intensity=0.7, horizon_y=AL_HORIZON, t=T, dist=0.9, flare=0.5)
        fl = max(0.0, math.sin(az)) ** 4

        # ---- Deng --------------------------------------------------------------------------
        mouth = f.mouth('deng')
        idl = robot.idle(T, seed=4)
        head_dy = idl['head_dy'] * 0.6 - 8 * hope - 5 * inhale + 13 * slump
        head_tilt = idl['head_tilt'] + 0.05 * hope - 0.03 * inhale + 0.12 * slump
        lean = 0.05 + 0.5 * idl['lean'] - 0.04 * hope - 0.02 * inhale + 0.08 * slump
        squash = -0.03 * inhale + 0.05 * slump
        look = (lerp(0.55, 0.75, hope), lerp(0.05, -0.35, hope))
        if glance > 0:
            look = (lerp(look[0], -0.85, glance), lerp(look[1], -0.45, glance))
        look = (look[0], lerp(look[1], 0.55, slump))
        blink = robot.auto_blink(T, seed=5)
        if slump > 0:
            blink = max(blink, 0.38 * slump)
        kw = dict(x=seat_x, y=seat_y, scale=AL_S, facing=1.0, sit=1.0, sit_dangle=1.0, lean=lean,
                  squash=squash, head_dy=head_dy, head_tilt=head_tilt, head_turn=-0.35 * glance,
                  eyes='sad', blink=blink, look=look, mouth=mouth, smile=-0.3 - 0.15 * slump,
                  scarf_wind=0.75, scarf_dir=-1.0, heart=0.85, chin_hands=1.0 - 0.25 * hope,
                  ambient='#b4bfe6', rim=0.55 + 0.4 * fl, rim_color=mix('#ffcf80', '#fff0c0', fl),
                  rim_angle=math.pi + 0.5, shadow=0.0, glow_cast=1.0, sit_kick=0.0)
        _draw_deng(c, kw, T)

    f.grade.update(vignette=lerp(0.52, 0.64, slump), bloom=1.0 + 0.25 * fl, sat=lerp(0.96, 0.86, slump),
                   tint='#9fc4ff', tint_amt=lerp(0.03, 0.08, slump), exposure=lerp(1.0, 0.95, slump))
