"""s07 · 回答 The answer (34 s) — grief → miracle → joy → golden morning.

Owner: SCENE-S07.  render(f) draws local time f.t (and keeps working past the end,
under the 1.5 s crossfade into s08).  Every cut is derived from timeline cues:

  S1  0 → stardust+1.3     EXT wide: the dark lighthouse under quiet stars (continues the
                           held exterior that ends s06). whale_song: the constellation whale
                           gathers out of scattered starlight and glides down with tiny bright
                           Guang on her head. stardust: she breathes a river of stardust that
                           pours into the lamp room (which glows starlight-blue).
  S2  → W01-0.15           INT lamp room: dead Deng slumped on the floor against the lens;
                           the river pours through the open roof hatch, swirls round him and
                           into his empty chest. reboot: flash, a star-shaped heart, god rays,
                           eyes flicker on (blink, blink), he sits up and looks down at his
                           chest in wonder.
  S3  → nuzzle             EXT gallery: the whale's enormous head right beside the gallery,
                           her kind eye on Deng. W01 "谢谢你，守灯人。" (f.mouth('whale')).
                           Guang, floating by her mother's eye, takes off.
  S4a → D16-0.25           Two-shot: Guang swoops down and hugs Deng's face. G10.
  S4b → farewell           Closer on Deng: D16, his star heart pulses on "为你发光";
                           a last cheek-nuzzle.
  S5a → sunrise+1.9        Guang flies up to her mother, who rises nose-up into the sky;
                           dawn begins on the right (east); Deng waves.
  S5b → end (+1.5)         The wide: sun rising over a gold sea, stars fading, the whale
                           swimming up into the sunrise and dissolving into sparkles, Guang
                           left behind as one bright star, Deng waving from the gallery.
Geography: the whale, and later the sun, are always screen-right of the lighthouse;
Deng faces right in every exterior shot.
"""
from __future__ import annotations

import math

import skia

from engine.core import (ADD, H, TAU, W, Camera, clamp, ease_in, ease_in_out, ease_out, ease_out_back, fill,
                         keyframes, lerp, linear, mix, nprng, pulse, smoothstep, spring)
from lib import fx, robot, sea, sky, star
from scenes._s07_util import (bezier, call, cr_sample, fields, has_kw, mk, put, record, resample, spath,
                              to_hex, to_layer)

DUST_COOL = '#cfe8ff'
DUST_GOLD = '#fff0b0'
WHALE_LIGHT = '#8fb8ff'
WARM = '#ffd36b'
SUN_COL = '#ffcf80'
NIGHT_AMB = '#aebadc'        # Deng's ambient tint under the whale's starlight
SUN_AMB = '#ffd9a8'          # ... in the sunrise
DENG_RATIO = 0.15            # Deng scale per lighthouse scale in exterior shots (head clears the rail)


# =============================================================================
# timing
# =============================================================================
class Cues:
    def __init__(self, f):
        self.song = f.cue('whale_song')
        self.dust = f.cue('stardust')
        self.reboot = f.cue('reboot')
        self.w01 = f.line('W01')
        self.nuzzle = f.cue('nuzzle')
        self.g10 = f.line('G10')
        self.d16 = f.line('D16')
        self.farewell = f.cue('farewell')
        self.sunrise = f.cue('sunrise')
        self.end = f.dur
        # cuts
        self.c2 = self.dust + 1.3
        self.c3 = self.w01[0] - 0.15
        self.c4a = self.nuzzle
        self.c4b = self.d16[0] - 0.25
        self.c5a = self.farewell
        self.c5b = self.sunrise + 1.9


def daylight(t, C):
    """Continuous time of day: (dawn, sunrise, star brightness)."""
    dawn = 0.42 + 0.58 * ease_in_out((t - (C.farewell - 0.8)) / (C.sunrise + 3.0 - C.farewell + 0.8))
    sunr = ease_in_out((t - C.sunrise) / 6.5)
    stars_b = keyframes(t, [(0, 0.5), (C.song, 0.5), (C.song + 2.5, 0.9), (C.sunrise - 0.5, 0.85),
                            (C.sunrise + 4.5, 0.0)])
    return dawn, sunr, stars_b


def tri(x, c, w=0.07):
    return max(0.0, 1 - abs(x - c) / w)


def piecewise(x, keys):
    if x <= keys[0][0]:
        return keys[0][1]
    for (a, va), (b, vb) in zip(keys, keys[1:]):
        if x <= b:
            return va + (vb - va) * ((x - a) / (b - a) if b > a else 1.0)
    return keys[-1][1]


def blink(T, seed=0):
    return robot.auto_blink(T, seed)


def side_wave(T, amount=1.0, speed=1.3):
    """Far-arm (arm_l) wave out to the side: the hand stays clear of the head."""
    ph = TAU * speed * T
    return (lerp(0.15, 2.05 + 0.06 * math.sin(ph * 0.5), amount), lerp(0.2, 0.15 + 0.45 * math.sin(ph), amount))


def vel(pos_fn, t, dt=0.04):
    x0, y0 = pos_fn(t - dt)
    x1, y1 = pos_fn(t)
    return (x1 - x0) / dt, (y1 - y0) / dt


# =============================================================================
# library wrappers (tolerant of libraries still being upgraded)
# =============================================================================
_WOFF = {}


def whale_off():
    """Unit-whale anchor offsets (scale 1, facing +1, rot 0), measured once."""
    if not _WOFF:
        try:
            _, res = record(lambda cc: call(star.draw_whale, cc, 0.0, 0.0, 0.0, scale=1.0, alpha=0.0, facing=1.0))
        except Exception:
            res = {}
        for k, v in (res or {}).items():
            try:
                _WOFF[k] = (float(v[0]), float(v[1]))
            except (TypeError, IndexError, ValueError):
                pass
        _WOFF.setdefault('eye', (450.0, -40.0))
        _WOFF.setdefault('mouth', (650.0, 40.0))
        _WOFF.setdefault('tail', (-700.0, 0.0))
        eye = _WOFF['eye']
        _WOFF.setdefault('head_top', (eye[0] - 70.0, eye[1] - 110.0))
    return _WOFF


def whale_pt(name, x, y, scale, facing=1.0, rot=0.0):
    ox, oy = whale_off().get(name, (0.0, 0.0))
    lx, ly = ox * scale * facing, oy * scale
    cs, sn = math.cos(rot), math.sin(rot)
    return x + lx * cs - ly * sn, y + lx * sn + ly * cs


def whale_center_for(name, px, py, scale, facing=1.0, rot=0.0):
    """Body centre that puts anchor `name` at (px, py)."""
    ax, ay = whale_pt(name, 0.0, 0.0, scale, facing, rot)
    return px - ax, py - ay


def draw_whale(c, T, x, y, scale, facing=1.0, rot=0.0, alpha=1.0, swim=0.0, mouth=0.0, eye=1.0,
               dissolve=0.0, glow=1.0, solid=0.22, dust=1.0, swim_rate=0.5, look=0.0):
    native = has_kw(star.draw_whale, 'dissolve')
    kw = dict(scale=scale, alpha=alpha, facing=facing, swim=swim, mouth=mouth, eye=eye, rot=rot, glow=glow,
              solid=solid, dust=dust, swim_rate=swim_rate, look=look)
    if native:
        kw['dissolve'] = dissolve
    else:
        kw['alpha'] = alpha * (1 - smoothstep(0.1, 0.95, dissolve))
    res = call(star.draw_whale, c, T, x, y, **kw) or {}
    A = {k: whale_pt(k, x, y, scale, facing, rot) for k in whale_off()}
    if not star.STUB:
        for k, v in res.items():
            A[k] = v
    return A


def draw_guang(c, T, **kw):
    kw.setdefault('glow', 1.3)
    return star.draw_star(c, mk(star.StarPose, **kw), T)


def flight_trail(c, T, pos_fn, t, span=0.6, width=10.0, n=14, intensity=1.0, color='#ffd86b'):
    """Glowing ribbon + glints along the path Guang just flew (oldest → newest)."""
    if intensity <= 0.01:
        return
    pts = [pos_fn(t - span + span * i / n) for i in range(n + 1)]
    L = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    if L < 6:
        return
    call(fx.light_trail, c, pts, width=width, color=color, intensity=intensity * clamp(L / 120), t=T)


def deng_pose(**kw):
    return mk(robot.RobotPose, **kw)


def deng_anchors(pose, T):
    if hasattr(robot, 'anchors'):
        try:
            return robot.anchors(pose, T)
        except Exception:
            pass
    s = pose.scale
    return dict(heart=(pose.x + 23 * s * pose.facing, pose.y - 150 * s), head=(pose.x, pose.y - 271 * s),
                eyes=(pose.x, pose.y - 272 * s))


def with_hand_on_heart(pose, T, amount=1.0, dx=6.0, dy=10.0):
    """Near hand resting on the porthole (IK override), blended by `amount`."""
    if amount <= 0.01 or 'arm_r_override' not in fields(robot.RobotPose):
        return pose
    a = deng_anchors(pose, T)
    hx, hy = a['heart']
    s = pose.scale
    tx, ty = hx + dx * s * pose.facing, hy + dy * s
    if amount < 0.99:
        rx, ry = a.get('r', (tx, ty))
        tx, ty = lerp(rx, tx, amount), lerp(ry, ty, amount)
    pose.arm_r_override = (tx, ty)
    return pose


def deng_bounds(cam, pose, depth=1.0):
    s = pose.scale
    x0, y0 = cam.to_screen(pose.x - 330 * s, pose.y - 470 * s, depth)
    x1, y1 = cam.to_screen(pose.x + 330 * s, pose.y + 60 * s, depth)
    return skia.Rect.MakeLTRB(x0, y0, x1, y1)


def deng_record(cam, pose, T, depth=1.0):
    return record(lambda cc: robot.draw_robot(cc, pose, T), cam, depth)


def deng_put(c, cam, pic, pose, mul=None, add=(0.0, 0.0, 0.0), depth=1.0):
    """Composite Deng. If the robot library ever lacks lighting inputs, fake the scene light
    with a colour matrix on his layer."""
    if 'ambient' in fields(robot.RobotPose):
        mul = None
    put(c, pic, mul=mul, add=add, bounds=deng_bounds(cam, pose, depth) if mul is not None else None)


def lh_anchors(x, y, s):
    if hasattr(sea, 'lighthouse_anchors'):
        try:
            return sea.lighthouse_anchors(x, y, s)
        except Exception:
            pass
    return dict(lamp=(x, y - 445 * s), gallery_y=y - 400 * s, gallery_x=(x - 64 * s, x + 64 * s),
                glass=(x - 40 * s, y - 474 * s, x + 40 * s, y - 414 * s), deng_scale=0.10 * s,
                hatch=(x, y - 498 * s))


def lighthouse(c, x, y, s, **kw):
    return call(sea.lighthouse, c, x, y, s, **kw)


def stardust(c, T, pts, **kw):
    kw.setdefault('color', DUST_COOL)
    kw.setdefault('gold', DUST_GOLD)
    call(fx.stardust_stream, c, T, pts, **kw)


# =============================================================================
# environment
# =============================================================================
def env(c, f, cam, HZ, dsea, dawn, sunr, stars_b, milky=0.0, mw=((700, 120), -0.45, 3000, 520), sun=None,
        lights=(), clouds=0.0, dstar=0.08, glow_amt=None):
    """Sky (screen space, anchored to the sea horizon), dawn glow, Milky Way, stars, sun,
    clouds, sea.  sun = (x, y, r, amount) in the sea layer."""
    Hs = cam.to_screen(0, HZ, dsea)[1]
    call(sky.sky, c, rect=(-100, -100, W + 200, H + 200), horizon_y=Hs, dawn=dawn, sunrise=sunr)
    with cam.apply(c, dsea):
        sx = sun[0] if sun else 1500
        ga = max(dawn * 0.55, sunr * 0.6) if glow_amt is None else glow_amt
        if ga > 0.02:
            call(sky.dawn_glow, c, horizon_y=HZ, amount=ga, sun_x=sx)
    with cam.apply(c, dstar):
        hy = to_layer(cam, 0, Hs, dstar)[1]
        if milky > 0.01:
            call(sky.milky_way, c, f.T, center=mw[0], angle=mw[1], length=mw[2], width=mw[3], alpha=milky,
                 horizon_y=hy)
        if stars_b > 0.005:
            call(sky.stars, c, f.T, bright=stars_b, horizon_y=hy)
    with cam.apply(c, dsea):
        if sun is not None and sun[3] > 0:
            call(sky.sun, c, sun[0], sun[1], r=sun[2], amount=sun[3], t=f.T, rays=0.7, flare=0.7)
        if clouds > 0:
            call(sky.clouds, c, f.T, y=HZ - 250, alpha=0.45 * clouds, seed=21, speed=5.0, scale=1.2, dawn=dawn,
                 sunrise=sunr, cover=0.35, layers=2)
        call(sea.ocean, c, f.T, horizon_y=HZ, lights=tuple(lights), dawn=dawn, sunrise=sunr)
    return Hs


# =============================================================================
# S1 · EXT wide — darkness; the whale descends; the river of stardust
# (opens on exactly the framing s06 holds on: lighthouse (960, 874), horizon 800,
#  camera (960, 520) zoom 1.06, dawn ≈ 0.42)
# =============================================================================
LH1 = (960.0, 874.0, 1.0)    # lighthouse base x, y, scale (as s06)
HZ1 = 800.0
D_WHALE1 = 0.8
MW1 = dict(center=(760, -620), angle=-1.05, length=6200, width=700, bend=0.06)


def whale1(t, C):
    """Whale body centre / scale / rot / appear / exhale for shot 1 (layer D_WHALE1)."""
    u = ease_in_out(clamp((t - C.song) / 3.6))
    x, y = bezier((180, -330), (300, -250), (380, 40), (430, 110), u)
    sc = lerp(0.5, 0.7, u)
    rot = lerp(0.32, -0.03, ease_out(u))
    y += 9 * math.sin(0.8 * t) * u
    rot += 0.02 * math.sin(0.6 * t + 1.0) * u
    ex = pulse(t, C.dust - 0.2, C.dust + 1.4, 0.5)
    rot -= 0.05 * ex
    appear = smoothstep(C.song - 0.2, C.song + 2.6, t)
    return x, y, sc, rot, appear, ex


def shot1(f, c, t, C):
    dawn, sunr, stars_b = daylight(t, C)
    hold = ease_in_out(clamp(t / C.song))
    tilt = ease_in_out((t - C.song + 0.2) / 4.0)
    cam = Camera(x=lerp(960, 900, tilt), y=lerp(520, 512, hold) - 80 * tilt,
                 zoom=lerp(1.06, 1.075, hold) - 0.07 * tilt, t=f.T)
    Hs = cam.to_screen(0, HZ1, 1.0)[1]

    wx, wy, ws, wrot, appear, exhale = whale1(t, C)
    swim = f.T * 0.5 + 0.8 * smoothstep(C.dust - 0.3, C.dust + 1.0, t)
    lx0, ly0, ls = LH1
    LA = lh_anchors(lx0, ly0, ls)
    head = ease_out(clamp((t - C.dust) / 1.5))
    arrive = smoothstep(0.8, 1.0, head) if t >= C.dust else 0.0

    with cam.apply(c, 1.0):
        call(sky.sky, c, rect=(cam.x - 1400, cam.y - 1000, 2800, 2000), horizon_y=HZ1, dawn=dawn)
        call(sky.dawn_glow, c, horizon_y=HZ1, amount=dawn * 0.8, sun_x=1500)
    with cam.apply(c, 0.3):
        call(sky.milky_way, c, f.T, alpha=(0.55 + 0.35 * appear) * stars_b / 0.9,
             horizon_y=to_layer(cam, 0, Hs, 0.3)[1], **MW1)
    with cam.apply(c, 0.15):
        call(sky.stars, c, f.T, bright=stars_b, horizon_y=to_layer(cam, 0, Hs, 0.15)[1])
    with cam.apply(c, 1.0):
        call(sky.clouds, c, f.T, y=-250, alpha=0.4, dawn=dawn * 0.6, cover=0.5, thickness=210)

    # the whale gathers out of scattered starlight (reverse dissolve), Guang on her head
    A = None
    with cam.apply(c, D_WHALE1):
        if appear > 0:
            fx.glow(c, wx, wy, 1000 * ws, '#5f8fff', 0.12 * appear, core=False)
            A = draw_whale(c, f.T, wx, wy, ws, facing=1.0, rot=wrot, alpha=clamp(appear * 1.6),
                           dissolve=1 - appear, swim=swim, mouth=0.45 * exhale, eye=1.0, glow=0.9, solid=0.25,
                           look=0.6)
            gs = 0.3
            hx, hy = A['head_top']
            draw_guang(c, f.T, x=hx, y=hy - star.RADIUS * gs * 0.9 + 2 * math.sin(f.T * 3), scale=gs,
                       glow=1.6, alpha=smoothstep(0.55, 1.0, appear), eyes='happy', smile=0.8,
                       arm_r=0.35 * math.sin(f.T * 5), rot=wrot * 0.5)

    with cam.apply(c, 1.0):
        lights = []
        if appear > 0:
            lights.append((wx, HZ1 - 300, WHALE_LIGHT, 0.4 * appear))
        call(sea.ocean, c, f.T, horizon_y=HZ1, rect_x=(cam.x - 1500, cam.x + 1500), bottom=cam.y + 900,
             lights=tuple(lights), dawn=dawn)
        call(sea.island, c, lx0, ly0 + 6, ls, dawn=dawn, t=f.T, lamp=0.0, door_light=0.0)
        # dark lamp room ... that fills with starlight as the river pours in
        lighthouse(c, lx0, ly0, ls, lamp=0.55 * arrive, dawn=dawn, lit_windows=0.0, t=f.T,
                   lamp_color='#cfe6ff', glow=0.7)
    lx, ly = LA['lamp']
    hatch = LA.get('hatch', (lx, ly - 50 * ls))

    if t >= C.dust - 0.1 and A is not None:
        Mx, My = cam.to_screen(*A['mouth'], D_WHALE1)
        Lx, Ly = cam.to_screen(*hatch, 1.0)
        key = [(Mx, My), (Mx + 110, My - 70), (lerp(Mx, Lx, 0.6), min(My, Ly) - 120),
               (Lx + 60, Ly - 120), (Lx, Ly)]
        path = cr_sample(key, 10)
        inten = smoothstep(C.dust - 0.1, C.dust + 0.5, t)
        z = cam.zoom
        if head > 0.01:
            pts = resample(path, 50, 0.0, head)
            done = head > 0.97
            stardust(c, f.T, pts, density=0.9 * head, width=22 * z, speed=0.3, seed=71, intensity=inten,
                     size=0.8 * z, swirl_end=1.3 if done else 0.0, swirl_len=0.28, swirl_persp=0.4,
                     swirl_radius=55 * z, taper=0.5, head=1.0)
            call(fx.sparkles, c, f.T, Mx, My, radius=60, count=14, color='#e8f4ff', seed=73,
                 intensity=inten * (1 - 0.6 * head), rise=30, size=0.6)
            if not done:
                hx_, hy_ = pts[-1]
                fx.glow(c, hx_, hy_, 70 * z, '#dff0ff', 0.6 * inten)
        if arrive > 0:
            gx_, gy_ = cam.to_screen(lx, ly, 1.0)
            fx.glow(c, gx_, gy_, 400 * z, WHALE_LIGHT, 0.14 * arrive, core=False)


# =============================================================================
# S2 · INT lamp room — the stardust fills his chest; reboot
# (Deng exactly as s06 leaves him: slumped against the lamp, porthole open, dark)
# =============================================================================
DENG2 = dict(x=1105.0, y=935.0, scale=1.35)


def power_curve(x):
    if x < 0.30:
        return 0.0
    return piecewise(x, [(0.30, 0.0), (0.33, 0.85), (0.41, 0.05), (0.54, 0.0), (0.57, 0.95), (0.65, 0.3),
                         (0.73, 1.0)])


def shot2(f, c, t, C):
    dawn, _, _ = daylight(t, C)
    x = t - C.reboot                     # time since reboot
    push = ease_in_out(clamp((t - C.c2) / (C.reboot + 0.2 - C.c2)))
    back = ease_in_out(clamp((x - 0.2) / 2.0))
    cam = Camera(x=lerp(990, 1060, push) - 15 * back, y=lerp(520, 600, push) - 10 * back,
                 zoom=lerp(1.0, 1.4, push) - 0.06 * back, t=f.T)
    if x > 0:
        cam.shake = 7 * math.exp(-5 * x)

    I = smoothstep(C.c2 - 0.5, C.c2 + 0.2, t) * (1 - smoothstep(C.reboot - 0.25, C.reboot + 0.3, t))
    charge = smoothstep(C.c2, C.reboot, t)
    with cam.apply(c, 1.0):
        lens = call(sea.lamp_room, c, f.T, lamp=0.0, lens_rot=0.0, dawn=dawn, pointing_up=1.0, wheel_rot=-2.6,
                    beam_sweep=0.0)
    try:
        lx, ly, lr = lens
    except (TypeError, ValueError):
        lx, ly, lr = 960.0, 470.0, 170.0

    # ---- Deng: slumped against the lamp, dead ... then alive
    p = power_curve(x) if x > 0 else 0.0
    hs = ease_out(clamp(x / 0.45)) if x > 0 else 0.0
    lift = ease_out_back(clamp((x - 0.35) / 0.9)) if x > 0 else 0.0
    down = ease_in_out(clamp((x - 1.35) / 0.55))
    touch = ease_in_out(clamp((x - 1.45) / 0.6))
    bl = max(tri(x, 1.05), tri(x, 1.38)) if x < 1.8 else blink(f.T, 3)
    eyes = 'closed' if p <= 0.01 else ('surprised' if x < 1.3 else 'wonder')
    br = 0.5 + 0.5 * math.sin(f.T * 1.6)
    amb = mix('#5a6488', '#b8c4e4', clamp(0.5 * charge * I + smoothstep(0.0, 1.0, x)))
    pose = deng_pose(
        x=DENG2['x'] + 10 * lift, y=DENG2['y'], scale=DENG2['scale'], facing=-1.0,
        sit=0.55, sit_dangle=0.0, lean=lerp(0.32, 0.04, lift),
        head_dy=lerp(34, 4, lift) + 5 * down + (1 - lift) * 1.2 * br,
        head_tilt=lerp(0.35, -0.03, lift) + 0.14 * down,
        arm_l=(lerp(0.1, 0.15, lift), lerp(0.2, 0.3, lift)), arm_r=(lerp(1.3, 0.25, lift), lerp(0.8, 0.3, lift)),
        hand_l_open=0.6, hand_r_open=lerp(0.8, 0.9, touch), heart_open=1.0 - smoothstep(2.3, 2.9, x),
        eyes=eyes, eyes2='happy', eyes_mix=smoothstep(2.1, 2.6, x), blink=bl,
        look=(0.1 * down, 0.95 * down) if x > 0 else (-0.2, 0.3), smile=lerp(0.2, 0.7, smoothstep(1.9, 2.5, x)),
        power=p, heart=hs, heart_present=x >= 0, heart_star=hs, antenna_blink=p > 0.9,
        scarf_wind=0.2 + 0.25 * I, ambient=to_hex(amb), rim=0.3 + 0.5 * I, rim_color='#bfe0ff',
        rim_angle=-math.pi / 2, mouth=0.0, shadow=0.4)
    pose = with_hand_on_heart(pose, f.T, touch)
    pic, res = deng_record(cam, pose, f.T)
    hx, hy = res.get('heart', (pose.x, pose.y - 150 * pose.scale))

    # ---- the river of stardust through the open roof hatch, spiralling into the chest
    tight = smoothstep(C.reboot - 1.8, C.reboot, t)
    pts = [(lx - 30, -260), (lx - 10, 40), (lx + 60, hy - 430), (hx - 260, hy - 140), (hx, hy)]
    skw = dict(density=1.3, width=58, speed=0.26, seed=81, intensity=I, size=1.5, swirl_end=2.3 + tight,
               swirl_len=0.42, swirl_persp=0.3, swirl_radius=lerp(280, 170, tight), taper=0.35, head=0.0)

    def stream_half(front):
        c.save()
        c.clipRect(skia.Rect.MakeLTRB(-1e5, hy if front else -1e5, 1e5, 1e5 if front else hy))
        stardust(c, f.T, pts, **skw)
        c.restore()

    with cam.apply(c, 1.0):
        if I > 0:
            # starlight falling through the hatch and filling the room
            col_p = linear((0, -100), (0, hy + 250), [(0, '#9fc4ff', 0.2 * I), (1, '#9fc4ff', 0.0)])
            col_p.setBlendMode(ADD)
            c.drawPath(spath([(lx - 150, -140), (lx + 150, -140), (hx + 300, hy + 250), (hx - 300, hy + 250)]),
                       col_p)
            fx.glow(c, hx, hy - 60, 620, WHALE_LIGHT, 0.2 * I, core=False)
            call(fx.dust_motes, c, f.T, rect=(lx - 400, 0, 1100, 1000), count=45, color='#cfe3ff',
                 intensity=0.55 * I, seed=77, light=(hx, hy, 500))
            stream_half(False)

    deng_put(c, cam, pic, pose)

    with cam.apply(c, 1.0):
        if I > 0:
            stream_half(True)
            # the chest drinks the light
            fx.glow(c, hx, hy, 90 + 80 * charge, '#e6f3ff', (0.35 + 0.8 * charge) * I)
            call(fx.sparkles, c, f.T, hx, hy, radius=60 + 50 * charge, count=12, color='#eaf6ff', seed=131,
                 intensity=I * charge, rise=-25, size=0.7)
        if x > -0.05:
            xx = max(0.0, x)
            # the miracle: flash, rays, ring, the new warm heart light
            call(fx.impact_flash, c, hx, hy, xx, radius=380, color='#fff6e0', glow_color=WARM, intensity=1.1)
            call(fx.god_rays, c, f.T, hx, hy, count=16, length=1150, color='#fff0c0',
                 intensity=keyframes(xx, [(0, 0.0), (0.08, 1.0), (1.6, 0.45), (4.0, 0.3)]), seed=9, start=30)
            call(fx.shockwave, c, hx, hy, xx, color='#fff4d0', max_r=720, life=1.1, persp=0.85)
            call(fx.sparkle_burst, c, hx, hy, xx, scale=0.9, color='#fff4c0', seed=141, count=12, life=1.2)
            fx.glow(c, hx, hy, 230, WARM, 0.45 * hs, core=False)
            fx.glow(c, hx, hy, 560, '#ffb347', 0.12 * hs, core=False)
            if robot.STUB:
                call(fx.heart_core, c, f.T, hx, hy, r=15 * pose.scale, intensity=hs, kind='star')

# =============================================================================
# gallery shots (S3, S4a, S4b, S5a): camera outside, at gallery height
# =============================================================================
class Gallery:
    """Lighthouse top at the scale matching Deng (ds), Deng standing on the right-hand side
    of the balcony at (deng_x, floor_y)."""

    def __init__(self, ds, deng_x, floor_y, side=0.74, view_e=0.035):
        a1 = lh_anchors(0.0, 0.0, 1.0)
        self.S = ds / DENG_RATIO
        half = (a1['gallery_x'][1] - a1['gallery_x'][0]) / 2 * self.S
        self.x = deng_x - side * half
        self.base = floor_y - a1['gallery_y'] * self.S
        self.view_e = view_e

    def draw(self, c, cam, dawn, sunr=0.0, parts='all'):
        with cam.apply(c, 1.0):
            lighthouse(c, self.x, self.base, self.S, lamp=0.0, dawn=dawn, sunrise=sunr, sun_dir=1.0,
                       lit_windows=0.0, parts=parts, view_e=self.view_e, glow=0.0)
            if parts == 'front' and not has_kw(sea.lighthouse, 'parts'):
                pass


def gallery_whale(c, f, cam, ex, ey, ws, depth=0.9, glow=0.7, mouth=0.0, eye=1.0, look=0.7):
    """The whale's head seen close, placed by her eye position (she faces left, at Deng)."""
    wrot = 0.025 * math.sin(f.T * 0.5)
    wx, wy = whale_center_for('eye', ex, ey, ws, -1.0, wrot)
    with cam.apply(c, depth):
        fx.glow(c, ex, ey, 1100, '#5f8fff', 0.14, core=False)
        A = draw_whale(c, f.T, wx, wy, ws, facing=-1.0, rot=wrot, alpha=1.0, swim=f.T * 0.25, mouth=mouth,
                       eye=eye, glow=glow, solid=0.4, dust=0.5, swim_rate=0.25, look=look)
    return A


def shot3(f, c, t, C):
    dawn, sunr, stars_b = daylight(t, C)
    u = ease_in_out(clamp((t - C.c3) / (C.nuzzle - C.c3)))
    cam = Camera(x=960 + 30 * u, y=540 - 10 * u, zoom=1.0 + 0.04 * u, t=f.T)
    DX, FY, DS = 600.0, 1000.0, 1.25
    HZ = 650.0
    w1, w2 = C.w01

    ex, ey = 1400 + 8 * math.sin(f.T * 0.7), 380 + 7 * math.sin(f.T * 0.9)
    lights = [(to_layer(cam, cam.to_screen(ex, ey, 0.9)[0], 0, 0.25)[0], HZ - 100, WHALE_LIGHT, 0.6)]
    env(c, f, cam, HZ, 0.25, dawn, sunr, stars_b * 0.85, milky=0.35, mw=((700, 60), -0.3, 3000, 500),
        lights=lights)
    eye_open = 1 - 0.9 * pulse(t, w2 + 0.25, w2 + 0.42, 0.13)
    gallery_whale(c, f, cam, ex, ey, 4.2, glow=0.75, mouth=f.mouth('whale'), eye=eye_open, look=0.8)
    G = Gallery(DS, DX, FY)
    G.draw(c, cam, dawn, parts='back')

    # Deng on the balcony, hand on his new heart, looking up into her eye
    surprise = 1 - smoothstep(w1 + 0.6, w1 + 1.2, t)
    pose = deng_pose(
        x=DX, y=FY, scale=DS, facing=1.0, lean=-0.04 + 0.01 * math.sin(f.T * 0.8),
        head_dy=1.5 * math.sin(f.T * 1.6), head_tilt=-0.08, arm_l=(0.12, 0.25), hand_r_open=0.9,
        eyes='surprised' if surprise > 0.5 else 'wonder', eyes2='happy', eyes_mix=smoothstep(w2 - 1.2, w2 - 0.4, t),
        blink=blink(f.T, 5), look=(0.65, -0.55), smile=lerp(0.2, 0.75, smoothstep(w1 + 1.0, w2, t)),
        power=1.0, heart=1.0, heart_present=True, heart_star=1.0, scarf_wind=0.45, scarf_dir=-1.0, mouth=0.0,
        ambient=NIGHT_AMB, rim=0.8, rim_color='#9fd0ff', rim_angle=-0.45, shadow=0.0)
    pose = with_hand_on_heart(pose, f.T)
    pic, res = deng_record(cam, pose, f.T)
    deng_put(c, cam, pic, pose)
    hx, hy = cam.to_screen(*res.get('heart', (DX, FY - 150 * DS)))
    fx.glow(c, hx, hy, 140, WARM, 0.4 + 0.08 * math.sin(f.T * 3), core=False)
    G.draw(c, cam, dawn, parts='front')

    # Guang floats by her mother's eye; waves; then swoops off towards Deng
    face = cam.to_screen(*res.get('head', (DX, FY - 271 * DS)))
    target = to_layer(cam, face[0] + 80, face[1] - 10, 0.95)
    launch = C.nuzzle - 0.55
    off = f.T - t

    def gpos(tt):
        k = ease_in(clamp((tt - launch) / 1.0))
        r_ = (ex - 170 + 8 * math.sin((tt + off) * 1.1), ey - 200 + 10 * math.sin((tt + off) * 2.0))
        return (lerp(r_[0], target[0], k), lerp(r_[1], target[1], k) - 110 * math.sin(k * math.pi))

    gx, gy = gpos(t)
    flying = t > launch
    vx, vy = vel(gpos, t) if flying else (0.0, 0.0)
    waving = smoothstep(w2 - 0.9, w2 - 0.4, t) * (1 - smoothstep(launch - 0.1, launch, t))
    with cam.apply(c, 0.95):
        flight_trail(c, f.T, gpos, t, span=0.4, width=9, intensity=1.0 if flying else 0.0)
        draw_guang(c, f.T, x=gx, y=gy, scale=0.8, glow=1.6, eyes='happy' if t < w2 else 'open', smile=0.9,
                   arm_r=waving * (0.6 + 0.45 * math.sin(f.T * 12)), look=(-0.8, 0.5),
                   trail=0.7 if flying else 0.0, vx=vx, vy=vy, rot=-0.3 * flying, rim=0.6, rim_color='#9fc4ff',
                   rim_dir=0.0)


def shot4a(f, c, t, C):
    dawn, sunr, stars_b = daylight(t, C)
    tt = t - C.nuzzle
    u = ease_in_out(clamp(tt / (C.c4b - C.nuzzle)))
    cam = Camera(x=960 - 20 * u, y=540 - 15 * u, zoom=1.0 + 0.05 * u, t=f.T)
    DX, FY, DS = 760.0, 1250.0, 2.35
    head = (DX, FY - 271 * DS)
    HZ = head[1] + 30
    ex, ey = 1680 + 10 * math.sin(f.T * 0.6), 230 + 8 * math.sin(f.T * 0.8)
    lights = [(to_layer(cam, ex, 0, 0.25)[0], HZ - 100, WHALE_LIGHT, 0.5)]
    env(c, f, cam, HZ, 0.25, dawn, sunr, stars_b * 0.8, milky=0.3, mw=((900, 40), -0.3, 3000, 500), lights=lights)
    gallery_whale(c, f, cam, ex, ey, 5.2, glow=0.55, look=0.9, eye=1 - 0.9 * pulse(tt, 3.2, 3.36, 0.12))
    G = Gallery(DS, DX, FY, view_e=0.03)
    G.draw(c, cam, dawn, parts='back')

    # Guang's flight: swoop in → hug → pull back and talk
    g1, g2 = C.g10
    t_hug, t_rel, t_set = 0.55, 1.55, 2.25
    hug_pt = (head[0] + 66 * DS, head[1] + 4 * DS)
    talk_pt = (head[0] + 310, head[1] - 90)

    def gpos(x):
        if x < t_hug:
            k = ease_out(clamp(x / t_hug))
            return bezier((1560, -160), (1500, 150), (1150, 420), hug_pt, k)
        if x < t_rel:
            w = x - t_hug
            return (hug_pt[0] + 5 * math.sin(TAU * 3.0 * w) * smoothstep(0.1, 0.3, w),
                    hug_pt[1] + 2 * math.sin(TAU * 1.5 * w))
        k = ease_in_out(clamp((x - t_rel) / (t_set - t_rel)))
        return (lerp(hug_pt[0], talk_pt[0], k),
                lerp(hug_pt[1], talk_pt[1], k) - 40 * math.sin(k * math.pi) + 7 * math.sin((x + C.nuzzle) * 2.4) * k)

    gx, gy = gpos(tt)
    hugging = t_hug <= tt < t_rel + 0.15
    contact = tt - t_hug
    squash = 0.28 * math.exp(-7 * contact) * math.cos(contact * 20) if contact > 0 else 0.0
    point_up = pulse(t, g1 + 1.9, g1 + 2.9, 0.3)

    # Deng: surprised, then melting into the hug
    tilt = (0.1 * smoothstep(0, 0.2, contact) * (1 - smoothstep(t_rel - 0.1, t_rel + 0.4, tt))
            + 0.04 * spring(contact, 2.5, 5) * (contact > 0))
    cradle = pulse(tt, t_hug + 0.1, t_rel + 0.2, 0.3)
    pose = deng_pose(
        x=DX, y=FY, scale=DS, facing=1.0, lean=-0.03 + 0.03 * tilt,
        head_dy=1.5 * math.sin(f.T * 1.6) + 6 * squash, head_tilt=tilt,
        arm_r=(0.2, 0.25), hand_r_open=0.9, arm_l=(0.15, 0.25),
        eyes='surprised' if tt < t_hug + 0.12 else 'happy', blink=0.0 if hugging else blink(f.T, 7),
        look=(0.6, -0.25) if tt > t_rel else (0.3, -0.5), smile=0.9, mouth=f.mouth('deng'),
        blush=0.8 * pulse(tt, t_hug, t_set + 0.5, 0.3),
        power=1.0, heart=1.0, heart_present=True, heart_star=1.0, scarf_wind=0.45, scarf_dir=-1.0,
        ambient=NIGHT_AMB, rim=0.7, rim_color='#9fd0ff', rim_angle=-0.5, shadow=0.0)
    if cradle > 0.01 and 'arm_r_override' in fields(robot.RobotPose):
        a = deng_anchors(pose, f.T)
        r0 = a.get('r', (DX + 40 * DS, FY - 60 * DS))
        pose.arm_r_override = (lerp(r0[0], gx - 20 * DS, cradle), lerp(r0[1], gy + 40 * DS, cradle))
    pic, res = deng_record(cam, pose, f.T)
    deng_put(c, cam, pic, pose)
    hx, hy = cam.to_screen(*res.get('heart', (DX, FY - 150 * DS)))
    fx.glow(c, hx, hy, 240, WARM, 0.4 + 0.35 * cradle, core=False)
    G.draw(c, cam, dawn, parts='front')

    vx, vy = vel(gpos, tt)
    with cam.apply(c, 1.0):
        flight_trail(c, f.T, gpos, tt, span=0.4, width=16, intensity=1 - smoothstep(t_hug - 0.05, t_hug + 0.25, tt))
        if hugging:
            call(fx.sparkles, c, f.T, gx, gy, radius=130, count=10, color='#fff4c0', seed=151,
                 intensity=0.8, rise=40, size=0.9)
            call(fx.sparkle_burst, c, gx - 60, gy, contact, scale=0.6, color='#fff4c0', seed=152, count=8)
        draw_guang(c, f.T, x=gx, y=gy, scale=1.45, glow=1.45, squash=squash,
                   rot=(-0.28 if hugging else 0.08 * math.sin(f.T * 1.3)) + 0.05 * math.sin(f.T * 4) * hugging,
                   eyes='happy' if (hugging or t > g2 - 0.2) else 'open', hug=1.0 if hugging else 0.0,
                   arm_l=0.1 * math.sin(f.T * 2), arm_r=1.2 * point_up,
                   look=(0.2, -0.8) if point_up > 0.5 else (-0.7, 0.1), mouth=f.mouth('guang'), smile=0.9,
                   blush=0.9, trail=0.8 if tt < t_hug else 0.0, vx=vx, vy=vy, rim=0.5, rim_color='#9fc4ff',
                   rim_dir=0.0)


def shot4b(f, c, t, C):
    dawn, sunr, stars_b = daylight(t, C)
    tt = t - C.c4b
    u = ease_in_out(clamp(tt / (C.farewell - C.c4b)))
    cam = Camera(x=960 + 10 * u, y=540 - 10 * u, zoom=1.0 + 0.035 * u, t=f.T)
    DX, FY, DS = 780.0, 1480.0, 3.05
    head = (DX, FY - 271 * DS)
    HZ = head[1] + 40
    ex, ey = 1830 + 10 * math.sin(f.T * 0.6), 170 + 8 * math.sin(f.T * 0.8)
    lights = [(to_layer(cam, 1500, 0, 0.25)[0], HZ - 100, WHALE_LIGHT, 0.45)]
    env(c, f, cam, HZ, 0.25, dawn, sunr, stars_b * 0.75, milky=0.25, mw=((1000, 20), -0.25, 3000, 500),
        lights=lights)
    gallery_whale(c, f, cam, ex, ey, 6.5, glow=0.45, look=0.9)
    G = Gallery(DS, DX, FY, view_e=0.03)
    G.draw(c, cam, dawn, parts='back')

    d1, d2 = C.d16
    glow_phrase = pulse(t, d2 - 1.6, d2 + 0.2, 0.35)
    nod = math.sin(clamp((t - d1) / 0.45) * math.pi) if d1 <= t <= d1 + 0.45 else 0.0
    kiss_a, kiss_b = d2 - 0.05, d2 + 0.3
    kiss = pulse(t, kiss_a, kiss_b, 0.12)
    pose = deng_pose(
        x=DX, y=FY, scale=DS, facing=1.0, lean=-0.02 + 0.01 * math.sin(f.T * 0.8),
        head_dy=1.5 * math.sin(f.T * 1.6) + 10 * nod, head_tilt=0.05 * nod + 0.08 * kiss,
        arm_r=(0.3, 0.3), arm_l=(0.12, 0.25),
        eyes='happy', blink=0.0 if kiss > 0.2 else blink(f.T, 9), look=(0.55, -0.2), smile=0.85,
        blush=0.5 + 0.4 * kiss, mouth=f.mouth('deng'), power=1.0, heart=1.0, heart_present=True, heart_star=1.0,
        scarf_wind=0.5, scarf_dir=-1.0, ambient=NIGHT_AMB, rim=0.7, rim_color='#9fd0ff', rim_angle=-0.5,
        shadow=0.0)
    pic, res = deng_record(cam, pose, f.T)
    deng_put(c, cam, pic, pose)
    hx, hy = cam.to_screen(*res.get('heart', (DX, FY - 150 * DS)))
    fx.glow(c, hx, hy, 300, WARM, 0.4 + 0.5 * glow_phrase, core=False)
    if glow_phrase > 0:
        call(fx.god_rays, c, f.T, hx, hy, count=12, length=700, color='#fff0c0', intensity=0.4 * glow_phrase,
             seed=19, start=40)
        call(fx.sparkles, c, f.T, hx, hy, radius=150, count=12, color='#fff4c0', seed=161,
             intensity=glow_phrase, rise=60, size=0.9)
    G.draw(c, cam, dawn, parts='front')

    # Guang floating, listening, then a quick cheek-nuzzle and up she goes
    cheek = (head[0] + 72 * DS, head[1] + 6 * DS)

    def gpos(x):
        tg = C.c4b + x
        rest = (1330 + 6 * math.sin(tg * 1.1), 450 + 10 * math.sin(tg * 2.2))
        k = pulse(tg, kiss_a, kiss_b, 0.12)
        up = ease_in(clamp((tg - kiss_b) / 0.5))
        return (lerp(rest[0], cheek[0], k) + 260 * up, lerp(rest[1], cheek[1], k) - 700 * up)

    gx, gy = gpos(tt)
    up = ease_in(clamp((t - kiss_b) / 0.5))
    vx, vy = vel(gpos, tt)
    with cam.apply(c, 1.0):
        flight_trail(c, f.T, gpos, tt, span=0.35, width=20, intensity=up)
        if kiss > 0.5:
            call(fx.sparkle_burst, c, gx - 60, gy, t - kiss_a - 0.1, scale=0.7, color='#fff4c0', seed=171, count=8)
        draw_guang(c, f.T, x=gx, y=gy, scale=1.85, glow=1.45, rot=0.06 * math.sin(f.T * 1.3) - 0.25 * kiss,
                   eyes='happy', hug=kiss, look=(-0.7, 0.15), smile=0.95, blush=1.0,
                   arm_l=0.15 * math.sin(f.T * 2), arm_r=0.15 * math.sin(f.T * 2 + 1),
                   mouth=f.mouth('guang'), trail=0.8 * up, vx=vx, vy=vy, rim=0.5, rim_color='#9fc4ff', rim_dir=0.0)


# =============================================================================
# S5a · farewell — Guang flies up to her mother, who rises; dawn begins in the east
# =============================================================================
def whale5a(t, C):
    k = ease_in_out(clamp((t - C.farewell - 0.2) / 5.5))
    x, y = bezier((1700, 330), (1560, 120), (1420, -150), (1300, -700), k)
    rot = lerp(0.06, 1.1, ease_in_out(clamp((t - C.farewell - 0.1) / 3.2)))
    return x, y + 8 * math.sin(t * 0.9), 1.5, rot


def shot5a(f, c, t, C):
    dawn, sunr, stars_b = daylight(t, C)
    tt = t - C.c5a
    u = ease_in_out(clamp(tt / (C.c5b - C.c5a)))
    cam = Camera(x=960 + 20 * u, y=540 - 50 * u, zoom=1.0 + 0.02 * u, t=f.T)
    DX, FY, DS = 600.0, 1060.0, 1.05
    HZ = 860.0
    sun_amt = smoothstep(C.sunrise - 0.3, C.sunrise + 2.0, t)
    env(c, f, cam, HZ, 0.2, dawn, sunr, stars_b, milky=0.3 * (1 - sunr), mw=((1000, 100), -0.4, 3000, 500),
        sun=(1560, HZ + 90, 46, sun_amt * 0.7), lights=[(1560, HZ, SUN_COL, 0.5 * sun_amt, None, 2.5)])

    wx, wy, ws, wrot = whale5a(t, C)
    with cam.apply(c, 0.9):
        fx.glow(c, wx, wy, 1100, '#5f8fff', 0.12 * (1 - 0.5 * sunr), core=False)
        draw_whale(c, f.T, wx, wy, ws, facing=-1.0, rot=wrot, alpha=1.0, swim=f.T * 0.6, eye=1.0, glow=0.8,
                   solid=0.25, look=0.8, swim_rate=0.6)
    G = Gallery(DS, DX, FY)
    G.draw(c, cam, dawn, sunr, parts='back')

    # Guang: from Deng's cheek up to her mother's head
    face = (DX + 72 * DS, FY - 271 * DS)
    s0 = to_layer(cam, *cam.to_screen(*face, 1.0), 0.9)

    def gpos(x):
        k = ease_in_out(clamp((x - C.farewell) / 1.5))
        xw, yw, sw, rw = whale5a(x, C)
        ht = whale_pt('head_top', xw, yw, sw, -1.0, rw)
        ht = (ht[0] - math.sin(rw) * star.RADIUS * 0.6 * 0.9, ht[1] - math.cos(rw) * star.RADIUS * 0.6 * 0.9)
        return (lerp(s0[0], ht[0], k) + 120 * math.sin(k * math.pi), lerp(s0[1], ht[1], k))

    gx, gy = gpos(t)
    gs = lerp(1.0, 0.6, ease_in_out(clamp(tt / 1.5)))
    flying = tt < 1.6
    vx, vy = vel(gpos, t)
    with cam.apply(c, 0.9):
        flight_trail(c, f.T, gpos, t, span=0.6, width=11, intensity=1.0 if flying else 0.0)
        draw_guang(c, f.T, x=gx, y=gy, scale=gs, glow=1.6, eyes='happy', smile=0.9,
                   arm_r=(0.5 + 0.5 * math.sin(f.T * 11)) if not flying else 0.3, rot=0.5 * wrot * (not flying),
                   look=(-0.3, 0.6), trail=0.8 if flying else 0.0, vx=vx, vy=vy)

    # Deng waves goodbye, watching her go, in the first warm light
    wave = smoothstep(C.farewell + 0.4, C.farewell + 0.9, t)
    gsx, gsy = cam.to_screen(gx, gy, 0.9)
    fsx, fsy = cam.to_screen(*face)
    dx, dy = gsx - fsx, gsy - fsy
    nrm = math.hypot(dx, dy) or 1.0
    warm = smoothstep(C.sunrise, C.sunrise + 2.0, t)
    pose = deng_pose(
        x=DX, y=FY, scale=DS, facing=1.0, lean=-0.05, head_dy=1.5 * math.sin(f.T * 1.6), head_tilt=-0.1,
        arm_r=(0.15, 0.2), arm_l=side_wave(t - C.farewell, wave), hand_l_open=1.0,
        eyes='happy', blink=blink(f.T, 11), look=(clamp(dx / nrm, -1, 1), clamp(dy / nrm, -1, 1)), smile=0.9,
        power=1.0, heart=1.0, heart_present=True, heart_star=1.0, scarf_wind=0.65, scarf_dir=-1.0,
        ambient=to_hex(mix(NIGHT_AMB, SUN_AMB, warm)), rim=0.8,
        rim_color=to_hex(mix('#9fd0ff', '#ffc98a', warm)), rim_angle=lerp(-1.2, 0.15, warm), shadow=0.0)
    pic, res = deng_record(cam, pose, f.T)
    deng_put(c, cam, pic, pose)
    hx, hy = cam.to_screen(*res.get('heart', (DX, FY - 150 * DS)))
    fx.glow(c, hx, hy, 130, WARM, 0.4, core=False)
    G.draw(c, cam, dawn, sunr, parts='front')


# =============================================================================
# S5b · the sunrise wide
# =============================================================================
LH5 = (560.0, 850.0, 1.2)
HZ5 = 705.0
D_SEA5 = 0.5
D_WHALE5 = 0.6
SUN_X5 = 1330.0


def whale5b(t, C):
    k = ease_in_out(clamp((t - C.c5b + 0.3) / 5.4))
    x, y = bezier((1300, 470), (1150, 390), (930, 330), (780, 290), k)
    return x, y, lerp(0.6, 0.46, k), lerp(0.95, 0.5, k)


def shot5b(f, c, t, C):
    dawn, sunr, stars_b = daylight(t, C)
    tt = t - C.c5b
    u = ease_in_out(clamp(tt / 5.6))
    lx0, ly0, ls = LH5
    LA = lh_anchors(lx0, ly0, ls)
    top = LA['lamp']
    cam = Camera(x=lerp(top[0] + 80, 960, u), y=lerp(top[1] + 20, 545, u) - 10 * smoothstep(5.6, 8.0, tt),
                 zoom=lerp(1.6, 1.0, u), t=f.T)

    rise = ease_out(clamp((t - C.sunrise) / 7.5))
    sun_y = HZ5 + 50 - 130 * rise
    sun_amt = smoothstep(C.sunrise - 0.3, C.sunrise + 2.0, t)
    env(c, f, cam, HZ5, D_SEA5, dawn, sunr, stars_b, milky=0.25 * (1 - sunr), mw=((900, 100), -0.4, 3000, 520),
        sun=(SUN_X5, sun_y, 44, sun_amt), lights=[(SUN_X5, sun_y, SUN_COL, 0.9 * sun_amt, None, 3.0)],
        clouds=sun_amt)

    # the whale swims up into the sunrise and dissolves into sparkles
    wx, wy, ws, wrot = whale5b(t, C)
    dis = smoothstep(C.c5b + 0.8, C.c5b + 5.0, t)
    g_alpha = 1 - smoothstep(C.c5b + 4.2, C.c5b + 5.6, t)
    with cam.apply(c, D_WHALE5):
        fx.glow(c, wx, wy, 900 * ws, '#8fb0ff', 0.1 * (1 - dis), core=False)
        A = draw_whale(c, f.T, wx, wy, ws, facing=-1.0, rot=wrot, alpha=1.0, swim=f.T * 0.5, eye=1.0,
                       dissolve=dis, glow=0.85, solid=0.15 * (1 - dis), look=0.8)
        # Guang rides on — and stays behind as one bright star in the morning sky
        gs = 0.3
        end = whale5b(C.c5b + 4.6, C)
        star_pt = whale_pt('head_top', end[0], end[1], end[2], -1.0, end[3])
        k = smoothstep(C.c5b + 3.8, C.c5b + 4.6, t)
        hx, hy = A['head_top']
        gx, gy = lerp(hx, star_pt[0], k), lerp(hy, star_pt[1], k) - star.RADIUS * gs * 0.9
        if g_alpha > 0:
            draw_guang(c, f.T, x=gx, y=gy, scale=gs, glow=1.7, alpha=g_alpha, eyes='happy', smile=0.9,
                       arm_r=0.5 + 0.5 * math.sin(f.T * 10), rot=0.3, rim=0.8, rim_color='#ffd36b', rim_dir=0.3)
        if g_alpha < 1:
            fl = sky.hero_flare(t - (C.c5b + 5.0), 1.2) if hasattr(sky, 'hero_flare') else 0.0
            if hasattr(sky, 'hero_star'):
                sky.hero_star(c, gx, gy, size=1.4, flare=fl, t=f.T, warmth=1.0, intensity=1 - g_alpha)
            else:
                call(fx.glint, c, gx, gy, 60, color='#fff6d8', intensity=1 - g_alpha)

    # island, lighthouse in golden light, and Deng waving from the gallery (behind the rail)
    DS = DENG_RATIO * ls
    gxr = LA['gallery_x'][1]
    pose = deng_pose(
        x=lerp(lx0, gxr, 0.74), y=LA['gallery_y'], scale=DS, facing=1.0, lean=-0.05, head_tilt=-0.08,
        arm_r=(0.15, 0.2), arm_l=side_wave(t - C.farewell), hand_l_open=1.0,
        eyes='happy', blink=blink(f.T, 13), look=(-0.2, -0.9), smile=0.9,
        power=1.0, heart=1.0, heart_present=True, heart_star=1.0, scarf_wind=0.85, scarf_dir=-1.0,
        ambient=SUN_AMB, rim=0.9, rim_color='#ffcf80', rim_angle=0.1, shadow=0.0)
    with cam.apply(c, 1.0):
        call(sea.island, c, lx0, ly0 + 8, ls * 0.9, dawn=dawn, sunrise=sunr, sun_dir=1.0, t=f.T, lamp=0.0,
             door_light=0.0)
        lighthouse(c, lx0, ly0, ls, lamp=0.0, dawn=dawn, sunrise=sunr, sun_dir=1.0, lit_windows=0.0, t=f.T,
                   parts='back')
    pic, res = deng_record(cam, pose, f.T)
    deng_put(c, cam, pic, pose)
    hx, hy = cam.to_screen(*res.get('heart', (pose.x, pose.y - 150 * DS)))
    fx.glow(c, hx, hy, 40 * cam.zoom, WARM, 0.6)
    with cam.apply(c, 1.0):
        lighthouse(c, lx0, ly0, ls, lamp=0.0, dawn=dawn, sunrise=sunr, sun_dir=1.0, lit_windows=0.0, t=f.T,
                   parts='front')
    ssx, ssy = cam.to_screen(SUN_X5, sun_y, D_SEA5)
    if ssy < cam.to_screen(0, HZ5, D_SEA5)[1] + 10:
        call(fx.lens_flare, c, ssx, ssy, intensity=0.25 * sun_amt * rise, color='#ffe7b0', t=f.T, halo=0.4,
             ghosts=0.5, starburst=0.5)


# =============================================================================
# grade & dispatch
# =============================================================================
def grade(f, t, C):
    x = t - C.reboot
    sat = keyframes(t, [(0, 0.8), (C.song, 0.72), (C.dust + 1, 0.8), (C.reboot, 0.82), (C.reboot + 0.8, 1.0),
                        (C.sunrise, 1.0), (C.sunrise + 5, 1.06)])
    expo = keyframes(t, [(0, 0.85), (C.song, 0.86), (C.dust + 1, 0.95), (C.reboot + 0.6, 1.0), (C.sunrise, 1.0),
                         (C.sunrise + 5, 0.97)])
    bloom = keyframes(t, [(0, 0.9), (C.song, 0.9), (C.dust + 1, 1.2), (C.reboot - 0.1, 1.4), (C.reboot + 0.15, 2.3),
                          (C.reboot + 1.6, 1.2), (C.sunrise, 1.1), (C.sunrise + 4, 0.95)])
    thr = keyframes(t, [(C.sunrise, 0.55), (C.sunrise + 4, 0.7)])
    vig = keyframes(t, [(0, 0.6), (C.dust, 0.55), (C.reboot + 1, 0.45), (C.sunrise + 3, 0.36)])
    cool = 1 - smoothstep(C.reboot, C.reboot + 1.5, t)
    warm = smoothstep(C.sunrise - 0.5, C.sunrise + 4.5, t)
    if warm > 0:
        tint, amt = to_hex(mix('#ffe6c8', '#ffd49a', warm)), lerp(0.03, 0.08, warm)
    else:
        tint, amt = to_hex(mix('#ffe6c8', '#a8b8e8', cool)), lerp(0.03, 0.1, cool)
    white = 0.5 * math.exp(-7.0 * x) if x >= 0 else 0.2 * smoothstep(-0.25, 0.0, x)
    f.grade.update(sat=sat, exposure=expo, bloom=bloom, bloom_thr=thr, vignette=vig, tint=tint, tint_amt=amt,
                   white=white)


def render(f):
    c = f.canvas
    t = f.t
    C = Cues(f)
    c.clear(skia.Color4f(0, 0, 0, 1).toColor())
    grade(f, t, C)
    if t < C.c2:
        shot1(f, c, t, C)
    elif t < C.c3:
        shot2(f, c, t, C)
    elif t < C.c4a:
        shot3(f, c, t, C)
    elif t < C.c4b:
        shot4a(f, c, t, C)
    elif t < C.c5a:
        shot4b(f, c, t, C)
    elif t < C.c5b:
        shot5a(f, c, t, C)
    else:
        shot5b(f, c, t, C)
