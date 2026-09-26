"""阿灯 (Deng) — the old lighthouse-keeper robot. Owner: ROBOT agent.

A fully posable 2D rig drawn with skia: tapered teal lighthouse-barrel body with a red band,
rivets, rust and a riveted patch; brass porthole with a flickering amber heart core (door on
a hinge, empty socket, or the new star heart); bendy noodle tube arms with brass joints and
three-fingered mittens; stubby legs with round boots; an old-TV/diving-helmet head whose dark
glass screen shows his glowing eyes and mouth; blinking antenna bulb; knitted red scarf whose
tails flutter procedurally.  No outlines, soft gradients, rim light, AO, additive glows.
Level of detail adapts to on-screen size (tiny on the gallery .. full-frame face close-up).

Usage in a scene:
    from lib.robot import RobotPose, draw_robot, auto_blink, idle
    pose = RobotPose(x=900, y=820, scale=1.0, facing=1, mouth=f.mouth('deng'),
                     blink=auto_blink(f.T), arm_r=(1.2, 0.4), **idle(f.T))
    a = draw_robot(c, pose, t=f.T)
    a['r']        # world (x, y) of the right (near) hand — put the star there

    # holding Guang with his hands IN FRONT of her:
    a = draw_robot(c, pose, f.T, part='back'); star.draw_star(...); draw_robot(c, pose, f.T, part='front')

CONVENTIONS (all unchanged from the original stub):
  * (x, y) = world ground point between the feet.  When sit=1 it is the SEAT point
    (the bottom of his body rests on y; the legs dangle forward/down past it).
  * scale 1 -> ~360 units from boot soles to antenna tip.
  * facing +1 faces screen-right, -1 mirrors to face screen-left.
  * lean > 0 tilts forward (towards the facing direction), pivoting at the pelvis with the
    feet planted; head_tilt > 0 rolls the top of the head towards the facing direction.
  * arms: (shoulder_angle, elbow_bend).  shoulder 0 = hanging down, +pi/2 = pointing
    forward (facing direction), +pi = straight up, negative = swung back.  elbow_bend > 0
    bends the forearm further in the same rotational direction (a natural elbow flex).
    arm_r is the NEAR arm (drawn in front of the body), arm_l the FAR arm (behind).
    A near arm raised above the head is drawn behind the head so the face stays readable
    (arm_r_behind_head overrides).  For a wave use pose.wave=1 (or wave_arm(T)).
  * look = (x, y), x > 0 looks forward (facing direction), y < 0 looks up.
  * walk_phase in cycles; a foot touches down at phase 0.0 (near foot) and 0.5 (far foot).
    Move pose.x by walk_speed(scale, cadence, run) * facing units/s so feet do not slide.
  * rim_angle: screen-space direction TOWARDS the light (0 = light to the right,
    -pi/2 = light above, pi/2 = below).  WORLD space (not mirrored by facing).
  * Everything is a pure function of (pose, t) — seeded noise only.

RETURNED ANCHORS (world coords, also available without drawing via anchors(pose, t)):
  'l','r' hand centres (far/near) · 'heart' porthole centre · 'head' head centre ·
  'eyes' between the eyes · 'antenna' bulb centre · 'mouth' · 'chin' · 'head_top' (where
  Guang can sit) · 'lap' (knee top when sitting, belly front when standing) · 'feet' ·
  'shoulder_l','shoulder_r','elbow_l','elbow_r'.

HELPERS: auto_blink(T, seed, every) · idle(T, seed) -> dict(head_dy, lean, head_tilt) ·
  walk_pose(pose, phase, amount=1, run=None) · walk_speed(scale, cadence, run) ·
  wave_arm(T, amount, speed) -> arm tuple · hand_pos(pose, side, t) · anchors(pose, t) ·
  draw_heart_core(c, x, y, scale, brightness, t) (the flame core, e.g. lifted out in his
  hand in s06) · draw_star_heart(c, x, y, scale, amount, t).
  Ambient presets: AMBIENT_NIGHT, AMBIENT_LAMP, AMBIENT_SUNRISE.

CONTRACT: keep RobotPose fields and draw_robot signature working (add fields with
defaults freely).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Optional

import skia

from engine.core import (ADD, SCREEN, _hash, clamp, col, ease_out_back, hexrgb, lerp,
                         mix, noise1, smooth_path, smoothstep)

# ---- cached paint helpers (same semantics as engine.core.fill/stroke/linear/radial) -------
_col_cached = lru_cache(maxsize=8192)(col)


def _cc(c, a=1.0):
    try:
        return _col_cached(c, round(a, 3))
    except TypeError:          # unhashable (list) colour
        return col(tuple(c), a)


def fill(c, a=1.0, blur=0.0, blend=None):
    p = skia.Paint(AntiAlias=True, Color=_cc(c, a))
    if blur > 0:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    if blend is not None:
        p.setBlendMode(blend)
    return p


def stroke(c, w=2.0, a=1.0, blur=0.0):
    p = skia.Paint(AntiAlias=True, Color=_cc(c, a), Style=skia.Paint.kStroke_Style, StrokeWidth=w)
    p.setStrokeCap(skia.Paint.kRound_Cap)
    p.setStrokeJoin(skia.Paint.kRound_Join)
    if blur > 0:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    return p


def linear(p0, p1, stops):
    return skia.Paint(AntiAlias=True, Shader=skia.GradientShader.MakeLinear(
        [skia.Point(*p0), skia.Point(*p1)], [_cc(c, a) for _, c, a in stops], [s for s, _, _ in stops]))


def radial(center, r, stops):
    return skia.Paint(AntiAlias=True, Shader=skia.GradientShader.MakeRadial(
        skia.Point(*center), max(r, 0.01), [_cc(c, a) for _, c, a in stops], [s for s, _, _ in stops]))

STUB = False
HEIGHT = 360  # design units, feet to antenna tip at scale=1

PI = math.pi

# ----------------------------------------------------------------------------------------
# rig geometry (rig-local units at scale 1, y down, origin = ground point between feet)
# ----------------------------------------------------------------------------------------
ANKLE_Y = -17.0
HIP_Y = -57.0
PIVOT_Y = -62.0                       # lean pivot (pelvis)
PLINTH_T, PLINTH_B, PLINTH_HALF = -82.0, -61.0, 76.0
BODY_T, BODY_B = -201.0, -76.0
HALF_T, HALF_B = 53.0, 70.0
RY = 0.13                             # vertical/horizontal ratio of ring ellipses (view from a bit above)
STRIPE_Y0, STRIPE_Y1 = -121.0, -101.0
PORT_Y0, PORT_R = -158.0, 30.0
SHOULDER_Y = -182.0
NECK_Y = -212.0                       # head pivot (bottom of head)
HEAD_OFF = 59.0                       # head centre above pivot
HW, HH, HR = 76.0, 57.0, 40.0         # head half width / half height / corner radius
ANT_LEN = 24.0
BULB_R = 7.5
L_UP, L_FORE, HAND_OFF = 62.0, 54.0, 15.0
HAND_SCALE = 1.22
L_LOW = L_FORE + HAND_OFF             # elbow -> hand centre
SIT_DROP = 51.0
STRIDE_WALK, STRIDE_RUN = 15.0, 23.0  # half stride (units at scale 1)

# emissive colours (not affected by ambient)
EYE = '#ffe9a8'
EYE_CORE = '#fffbe8'
EYE_EDGE = '#ffc45e'
EYE_GLOW = '#ffd36b'
HEART_CORE = '#fff0c8'
HEART_MID = '#ffd36b'
HEART = '#ffb347'
HEART_DEEP = '#ff7a2a'
STAR_CORE = '#ffffff'
STAR_BODY = '#fff3b0'
STAR_EDGE = '#ffc94a'
STAR_GLOW = '#ffd86b'
BULB_ON = '#ff5b4a'
BULB_HOT = '#ffd0c0'
TEAR = '#9fe6ff'
BLUSH = '#ff8f7a'

AMBIENT_NIGHT = '#aab6e0'
AMBIENT_LAMP = '#ffe7c4'
AMBIENT_SUNRISE = '#ffd9a8'

_MAT = {
    'body': '#4f8a9a', 'body_sh': '#3b6e7e', 'body_lt': '#7fb5c2', 'body_dk': '#2a5564',
    'head': '#56909f', 'head_lt': '#78afbc', 'head_sh': '#37687a',
    'brass': '#c9a15b', 'brass_lt': '#ecd291', 'brass_dk': '#80602e',
    'rust': '#9a5b3c', 'rust_dk': '#6a3924', 'rust_lt': '#bb7d50',
    'red': '#c8453c', 'red_dk': '#8f2e28', 'red_lt': '#e6735f', 'cream': '#ecdcb6',
    'screen': '#10202a', 'screen_edge': '#060d12', 'screen_mid': '#16303c',
    'iron': '#2e4854', 'iron_lt': '#4d7684', 'iron_dk': '#1a2a33',
    'boot': '#35535f', 'boot_lt': '#6690a0', 'boot_dk': '#1d323b',
    'patch': '#6f9094', 'patch_lt': '#9ab8b9', 'patch_dk': '#46666c',
    'hand': '#6199a8', 'hand_lt': '#86b9c5', 'hand_sh': '#3c6f7d',
    'tube': '#437a8a', 'tube_lt': '#79afbc', 'tube_sh': '#2b5664',
    'bulb_off': '#5e2522', 'cavity': '#1a0f0b', 'cavity_lt': '#3a2418',
    'weed': '#3f7d3c', 'weed_dk': '#22472a', 'weed_lt': '#93c96f',
    'shadow': '#05080f',
}


@lru_cache(maxsize=64)
def _palette(ambient):
    try:
        amb = hexrgb(ambient) if isinstance(ambient, str) else tuple(ambient)[:3]
    except Exception:
        amb = (1.0, 1.0, 1.0)
    return {k: tuple(clamp(v * a) for v, a in zip(hexrgb(h), amb)) for k, h in _MAT.items()}


# ----------------------------------------------------------------------------------------
# pose
# ----------------------------------------------------------------------------------------
@dataclass
class RobotPose:
    x: float = 960.0          # world x of the point between the feet (ground contact)
    y: float = 900.0          # world y of the ground contact (seat point when sit=1)
    scale: float = 1.0        # 1.0 → ~360 units tall
    facing: float = 1.0       # +1 faces screen-right, -1 faces screen-left (mirror)
    turn: float = 0.0         # fake yaw: -0.5 = full front view, 0 = 3/4 (designed), +1 ≈ profile
    lean: float = 0.0         # whole body tilt (radians, + = lean forward in facing direction)
    head_tilt: float = 0.0    # radians (+ = top of head towards facing direction)
    head_dy: float = 0.0      # extra head offset (bob), units (+ = down)
    # arms: (shoulder_angle, elbow_bend) radians. shoulder 0 = hanging down,
    #       +pi/2 = pointing forward (facing direction), +pi = straight up.
    arm_l: tuple = (0.15, 0.2)   # far arm
    arm_r: tuple = (0.15, 0.2)   # near arm
    hand_l_open: float = 0.5     # 0 fist .. 1 open
    hand_r_open: float = 0.5
    walk: float = 0.0            # 0..1 amount of walk cycle applied
    walk_phase: float = 0.0      # cycles (1.0 = one full stride pair)
    sit: float = 0.0             # 0 standing .. 1 sitting (legs forward)
    squash: float = 0.0          # -1..1 squash/stretch (landing, jumping)
    # face (drawn on the dark screen of the head)
    eyes: str = 'open'           # 'open','happy','sad','closed','surprised','determined','worried'
                                 # (+ 'sleepy', 'wonder')
    blink: float = 0.0           # 0 open .. 1 closed (overrides)
    look: tuple = (0.0, 0.0)     # eye offset -1..1 (x + = forward, y - = up)
    mouth: float = 0.0           # lip-sync openness 0..1 (pass f.mouth('deng'))
    smile: float = 0.3           # -1 frown .. 1 big smile (expressions add a bias)
    # lights / story state
    power: float = 1.0           # 0..1 overall electric life: eyes, antenna, screen (0 = dead)
    heart: float = 1.0           # 0..1 brightness of the heart core in the chest porthole
    heart_open: float = 0.0      # 0..1 chest porthole door swung open
    heart_present: bool = True   # False = empty chest (after giving the heart away)
    heart_star: float = 0.0      # 0..1 the NEW star-shaped heart (ending) replaces the old core
    scarf_wind: float = 0.5      # 0..1 wind strength on the red scarf
    antenna_blink: bool = True
    shake: float = 0.0           # tremble amplitude 0..1+ (effort / fear)
    # ---- additions -------------------------------------------------------------------
    run: float = 0.0             # 0..1 turns the walk cycle into a hurry/run (bigger stride, pumping arms)
    head_turn: float = 0.0       # extra yaw of the head only (same units as turn)
    back: bool = False           # True = seen from behind (no face / porthole; scarf knot at back)
    eyes2: str = ''              # optional second expression to blend towards ...
    eyes_mix: float = 0.0        # ... 0 = eyes, 1 = eyes2
    blush: float = 0.0           # 0..1 glowing blush marks under the eyes
    tear: float = 0.0            # 0..1 a glowing tear under the near eye
    hand_l_rot: float = 0.0      # extra wrist rotation (radians)
    hand_r_rot: float = 0.0
    hand_l_cup: float = 0.0      # >0.5 = palm-up cupped hand (fingers curl up, towards the other hand)
    hand_r_cup: float = 0.0      #        e.g. Guang lying in his palms (s04/s06)
    arm_l_override: Optional[tuple] = None   # world (x, y) IK target for the far hand
    arm_r_override: Optional[tuple] = None   # world (x, y) IK target for the near hand
    arm_l_front: Optional[bool] = None       # force far arm in front of body (None = auto)
    arm_r_behind_head: Optional[bool] = None  # force raised near arm behind the head (None = auto)
    hug: float = 0.0             # 0..1 both arms curl in to hold something against the porthole
    chin_hands: float = 0.0      # 0..1 both hands cup the chin (sad sitting pose)
    wave: float = 0.0            # 0..1 wave the arm given by wave_side (animated with t)
    wave_side: str = 'r'
    wave_speed: float = 1.0
    sit_dangle: float = 1.0      # 1 = legs dangle over a ledge, 0 = legs straight out on flat ground
    sit_kick: float = 0.0        # 0..1 happy alternating leg swing while sitting
    rim: float = 0.35            # rim light strength 0..1
    rim_color: str = '#9fc4ff'   # rim light colour (cool moonlight by default)
    rim_angle: float = -PI / 2   # direction towards the light, screen space (-pi/2 = from above)
    ambient: str = '#ffffff'     # multiply tint on all non-glowing materials ('#ffd9a8' sunrise)
    shadow: float = 0.35         # soft contact shadow under the feet (0 = none)
    antenna_rate: float = 1.0    # antenna blink speed multiplier
    antenna_flash: float = 0.0   # 0..1 extra flash of the antenna bulb ("idea!")
    scarf_dir: Optional[float] = None  # world wind direction: +1 blows to screen right, -1 left, None = trails behind
    glow_cast: float = 1.0       # how much the heart light warms his own body/hands
    seaweed: float = 0.0         # 0..1 a strand of seaweed draped over his head (after the splash)
    detail: Optional[int] = None  # force level of detail 0/1/2 (None = auto from on-screen size)


# ----------------------------------------------------------------------------------------
# helpers for scenes
# ----------------------------------------------------------------------------------------
def auto_blink(T, seed=0, every=3.7):
    """Natural blink value 0..1 at time T (use for pose.blink). Irregular intervals and the
    occasional double blink, fully deterministic."""
    T = T + seed * 1.37
    k = math.floor(T / every)
    v = 0.0
    for kk in (k - 1, k):
        start = kk * every + _hash(kk, 911 + seed) * every * 0.55
        for extra in ((0.0, 0.3) if _hash(kk, 57 + seed) < 0.22 else (0.0,)):
            ph = T - start - extra
            if 0.0 <= ph < 0.17:
                # fast close, slightly slower open
                v = max(v, ph / 0.06 if ph < 0.06 else 1 - (ph - 0.06) / 0.11)
    return clamp(v)


def idle(T, seed=0):
    """Small breathing / servo idle offsets: returns dict(head_dy, lean, head_tilt)."""
    br = math.sin(T * 1.55 + seed)
    return dict(head_dy=br * 1.6 + noise1(T * 0.9, 13 + seed) * 0.6,
                lean=math.sin(T * 0.77 + seed) * 0.008 + noise1(T * 0.35, 17 + seed) * 0.01,
                head_tilt=noise1(T * 0.4, 31 + seed) * 0.035)


def walk_pose(pose, phase, amount=1.0, run=None):
    """Copy of `pose` walking at `phase` (cycles). Foot contacts at phase 0.0 and 0.5."""
    kw = dict(walk=amount, walk_phase=phase)
    if run is not None:
        kw['run'] = run
    return replace(pose, **kw)


def walk_speed(scale=1.0, cadence=1.0, run=0.0):
    """Ground speed (units/s) at which the planted foot does not slide."""
    return 4.0 * lerp(STRIDE_WALK, STRIDE_RUN, clamp(run)) * scale * cadence


def wave_arm(T, amount=1.0, speed=1.0):
    """Arm tuple for a friendly wave (lerped from hanging by `amount`)."""
    ph = T * TAU_WAVE * speed
    sh = 3.3 + 0.06 * math.sin(ph * 0.5 + 0.8)
    el = -0.1 + 0.42 * math.sin(ph)
    base = (0.15, 0.2)
    return (lerp(base[0], sh, amount), lerp(base[1], el, amount))


TAU_WAVE = math.tau * 1.5   # waves per second * tau


def hand_pos(pose, side='r', t=0.0):
    """World (x, y) of a hand centre ('l' far / 'r' near) without drawing."""
    return anchors(pose, t)[side]


def anchors(pose, t=0.0):
    """All anchor points (world coords) exactly as draw_robot would return them."""
    return _Rig(pose, t).anchors()


# ----------------------------------------------------------------------------------------
# small geometry utils
# ----------------------------------------------------------------------------------------
def _half(y):
    v = clamp((y - BODY_T) / (BODY_B - BODY_T))
    return lerp(HALF_T, HALF_B, v) + 3.5 * math.sin(PI * v)


def _surf(theta, y, yaw, half=None):
    """Point on the body cylinder at surface angle theta (0 = his front) and height y.
    Returns (x, y, visibility cos)."""
    h = _half(y) if half is None else half
    ph = theta + yaw
    return h * math.sin(ph), y + RY * h * math.cos(ph), math.cos(ph)


def _bez(p0, c1, c2, p3, n):
    out = []
    for i in range(n + 1):
        t = i / n
        mt = 1 - t
        a, b, cc, d = mt * mt * mt, 3 * mt * mt * t, 3 * mt * t * t, t * t * t
        out.append((a * p0[0] + b * c1[0] + cc * c2[0] + d * p3[0],
                    a * p0[1] + b * c1[1] + cc * c2[1] + d * p3[1]))
    return out


def _tube_path(pts, r0, r1, radii=None):
    """Closed outline of a tapered tube with round caps along polyline pts."""
    n = len(pts)
    left, right, nrm = [], [], []
    for i, (x, y) in enumerate(pts):
        if i == 0:
            dx, dy = pts[1][0] - x, pts[1][1] - y
        elif i == n - 1:
            dx, dy = x - pts[i - 1][0], y - pts[i - 1][1]
        else:
            dx, dy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / L, dx / L
        r = radii[i] if radii else lerp(r0, r1, i / (n - 1))
        left.append((x + nx * r, y + ny * r))
        right.append((x - nx * r, y - ny * r))
        nrm.append((nx, ny, r))
    p = skia.Path()
    p.moveTo(*left[0])
    for q in left[1:]:
        p.lineTo(*q)
    ex, ey = pts[-1]
    nx, ny, r = nrm[-1]
    p.arcTo(skia.Rect.MakeLTRB(ex - r, ey - r, ex + r, ey + r), math.degrees(math.atan2(ny, nx)), -180, False)
    for q in reversed(right):
        p.lineTo(*q)
    sx, sy = pts[0]
    nx, ny, r = nrm[0]
    p.arcTo(skia.Rect.MakeLTRB(sx - r, sy - r, sx + r, sy + r), math.degrees(math.atan2(-ny, -nx)), -180, False)
    p.close()
    return p


def _polyline(pts):
    p = skia.Path()
    p.moveTo(*pts[0])
    for q in pts[1:]:
        p.lineTo(*q)
    return p


def _offset_path(path, dx, dy):
    q = skia.Path(path)
    q.offset(dx, dy)
    return q


def _rim(c, path, lx, ly, wd, color, a):
    """Crisp inner rim light on the side of `path` facing direction (lx, ly)."""
    if a <= 0.004:
        return
    for k, aa in ((3.2, 0.3), (1.0, 1.0)):
        sh = _offset_path(path, -lx * wd * k, -ly * wd * k)
        c.save()
        c.clipPath(path, skia.ClipOp.kIntersect, True)
        c.clipPath(sh, skia.ClipOp.kDifference, True)
        p = fill(color, clamp(a * aa))
        p.setBlendMode(ADD)
        c.drawPaint(p)
        c.restore()


def _glow(c, x, y, r, color, a):
    """Additive radial glow (cheap)."""
    if a <= 0.003 or r <= 0.1:
        return
    p = radial((x, y), r, [(0.0, color, clamp(a)), (0.22, color, clamp(a * 0.5)),
                           (0.55, color, clamp(a * 0.14)), (1.0, color, 0.0)])
    p.setBlendMode(ADD)
    c.drawCircle(x, y, r, p)


def _power_level(p, t):
    """Effective electric level with deterministic flicker when 0 < power < 0.5."""
    p = clamp(p)
    if p <= 0.0:
        return 0.0
    if p < 0.5:
        k = (0.5 - p) / 0.5
        n = 0.6 * noise1(t * 17.0, 71) + 0.4 * noise1(t * 41.0, 73)
        v = p * (1.0 - 0.55 * k * max(0.0, n + 0.1))
        if _hash(int(math.floor(t * 13.0)), 75) < k * 0.35:   # dropouts
            v *= 0.12
        if _hash(int(math.floor(t * 5.0)), 77) < k * 0.15:
            v = min(1.0, v * 1.8)                              # brief surges
        return clamp(v)
    return p


# static body silhouettes ----------------------------------------------------------------
@lru_cache(maxsize=1)
def _body_path():
    n = 22
    right = [(_half(lerp(BODY_T, BODY_B, i / n)), lerp(BODY_T, BODY_B, i / n)) for i in range(n + 1)]
    p = skia.Path()
    p.moveTo(-HALF_T, BODY_T)
    p.quadTo(0, BODY_T - 2 * RY * HALF_T, HALF_T, BODY_T)
    for q in right[1:]:
        p.lineTo(*q)
    hb = _half(BODY_B)
    m = 26
    for i in range(1, m + 1):
        ph = PI / 2 - PI * i / m
        p.lineTo(hb * math.sin(ph), BODY_B + RY * hb * math.cos(ph))
    for q in reversed(right[:-1]):
        p.lineTo(-q[0], q[1])
    p.close()
    return p


@lru_cache(maxsize=1)
def _plinth_path():
    h = PLINTH_HALF
    p = skia.Path()
    m = 26
    p.moveTo(-h + 3, PLINTH_T)
    for i in range(0, m + 1):
        ph = -PI / 2 + PI * i / m
        p.lineTo((h - 3) * math.sin(ph), PLINTH_T - RY * (h - 3) * math.cos(ph))
    p.lineTo(h, PLINTH_B - 4)
    for i in range(0, m + 1):
        ph = PI / 2 - PI * i / m
        p.lineTo(h * math.sin(ph), PLINTH_B + RY * h * math.cos(ph))
    p.lineTo(-h, PLINTH_B - 4)
    p.close()
    return p


@lru_cache(maxsize=16)
def _band_path(y0, y1):
    """Ring band on the body between heights y0 (top) and y1 (bottom), front half."""
    m = 28
    p = skia.Path()
    for i in range(m + 1):
        ph = -PI / 2 + PI * i / m
        h = _half(y0) + 2
        (p.moveTo if i == 0 else p.lineTo)(h * math.sin(ph), y0 + RY * h * math.cos(ph))
    for i in range(m + 1):
        ph = PI / 2 - PI * i / m
        h = _half(y1) + 2
        p.lineTo(h * math.sin(ph), y1 + RY * h * math.cos(ph))
    p.close()
    return p


# rust blobs: (theta, y, radius, seed)
_RUST = [(-1.05, -142, 7.5, 1), (0.95, -88, 5.0, 2), (-0.45, -186, 4.5, 3), (1.15, -165, 6.0, 4),
         (-0.62, -84, 6.0, 5), (2.5, -135, 8.0, 6), (-2.7, -168, 6.5, 7), (0.25, -91, 3.5, 8),
         (1.7, -110, 5.0, 9)]


@lru_cache(maxsize=16)
def _blob(seed, n=9):
    import random
    r = random.Random(seed * 7919 + 3)
    return [(math.cos(i / n * math.tau) * (0.72 + 0.4 * r.random()),
             math.sin(i / n * math.tau) * (0.72 + 0.4 * r.random())) for i in range(n)]


@lru_cache(maxsize=1)
def _star_path(ro=1.0, ri=0.5):
    pts = []
    for i in range(10):
        a = -PI / 2 + i * PI / 5
        r = ro if i % 2 == 0 else ri
        pts.append((math.cos(a) * r, math.sin(a) * r))
    return smooth_path(pts, close=True, tension=0.28)


# ----------------------------------------------------------------------------------------
# expressions (parametric eyes -> smooth blending between any two)
# ----------------------------------------------------------------------------------------
_E0 = dict(w=21.0, h=29.0, top=0.0, tilt=0.0, bot=0.0, arc=0.0, close=0.0, brow=0.0, btilt=0.0,
           by=0.0, dy=0.0, hl=1.0, o=0.0, smile=0.0, spark=0.0, sep=0.0)
_EXPR = {
    'open': {},
    'happy': dict(arc=1.0, smile=0.5, dy=-1.0, hl=0.0),
    'sad': dict(w=21.0, h=26.0, top=0.36, tilt=-0.55, bot=0.12, dy=4.0, hl=0.55, smile=-0.85,
                brow=0.0, sep=1.5),
    'closed': dict(close=1.0, smile=0.0, hl=0.0),
    'surprised': dict(w=26.0, h=34.0, o=1.0, brow=0.85, by=-2.0, smile=0.0, sep=1.0, dy=1.0),
    'determined': dict(w=23.0, h=26.0, top=0.28, tilt=0.6, hl=0.75, smile=-0.45, dy=1.0, sep=-1.0),
    'worried': dict(w=20.0, h=27.0, top=0.1, tilt=-0.3, brow=1.0, btilt=0.5, by=0.0, dy=2.0,
                    smile=-0.65, hl=0.8),
    'sleepy': dict(h=27.0, top=0.52, tilt=-0.12, hl=0.3, smile=0.0, dy=2.0),
    'wonder': dict(w=25.0, h=33.0, hl=1.0, spark=1.0, smile=0.15, brow=0.0, dy=1.0),
}


_ALIAS = {'teary': 'sad', 'cry': 'sad', 'angry': 'determined', 'shock': 'surprised', 'shocked': 'surprised',
          'smile': 'happy', 'joy': 'happy', 'laugh': 'happy', 'blink': 'closed', 'off': 'closed', 'dim': 'sleepy',
          'tired': 'sleepy', 'scared': 'worried', 'afraid': 'worried', 'fear': 'worried', 'awe': 'wonder',
          'amazed': 'wonder', 'neutral': 'open', 'normal': 'open'}


def _expr(name):
    d = dict(_E0)
    name = (name or 'open').lower() if isinstance(name, str) else 'open'
    d.update(_EXPR.get(_ALIAS.get(name, name), {}))
    return d


def _expr_mix(a, b, t):
    ea = _expr(a)
    if not b or t <= 0:
        return ea
    eb = _expr(b)
    t = clamp(t)
    return {k: lerp(ea[k], eb[k], t) for k in ea}


# ----------------------------------------------------------------------------------------
# IK
# ----------------------------------------------------------------------------------------
def _ik(sx, sy, tx, ty, l1, l2):
    """2-bone IK in rig convention: returns (shoulder, elbow, stretch)."""
    dx, dy = tx - sx, ty - sy
    d = math.hypot(dx, dy)
    stretch = 1.0
    if d > (l1 + l2) * 0.999:
        stretch = min(d / ((l1 + l2) * 0.999), 1.3)
    l1, l2 = l1 * stretch, l2 * stretch
    d = clamp(d, abs(l1 - l2) + 1e-3, (l1 + l2) * 0.9995)
    at = math.atan2(dx, dy)
    ca = clamp((l1 * l1 + d * d - l2 * l2) / (2 * l1 * d), -1, 1)
    ce = clamp((l1 * l1 + l2 * l2 - d * d) / (2 * l1 * l2), -1, 1)
    alpha = math.acos(ca)
    el = PI - math.acos(ce)
    return at - alpha, el, stretch


def _lerp_arm(a, b, t):
    return (lerp(a[0], b[0], t), lerp(a[1], b[1], t))


# ----------------------------------------------------------------------------------------
# the rig: all geometry for a pose (no drawing)
# ----------------------------------------------------------------------------------------
class _Rig:
    def __init__(self, pose: RobotPose, t: float):
        self.p = p = pose
        self.t = t
        s = p.scale
        f = 1.0 if p.facing >= 0 else -1.0
        self.f = f
        self.sit = sit = smoothstep(0.0, 1.0, clamp(p.sit))
        w = clamp(p.walk) * (1 - sit)
        rk = clamp(p.run)
        self.w, self.rk = w, rk
        ph = p.walk_phase
        self.ph = ph
        self.yaw = clamp(0.38 + 0.76 * p.turn, -1.25, 1.25)
        self.hyaw = clamp(self.yaw + 0.76 * p.head_turn, -1.25, 1.25)

        # ---- walk cycle ---------------------------------------------------------------
        A = lerp(STRIDE_WALK, STRIDE_RUN, rk)
        lift = lerp(11.0, 19.0, rk)
        feet = []
        for c0 in (0.0, 0.5):   # near (contact 0.0), far (contact 0.5)
            q = (ph - c0) % 1.0
            if q < 0.5:
                fx = A * (1 - 4 * q)
                fy = 0.0
                ang = -0.2 * max(0.0, 1 - q / 0.07)
            else:
                v = (q - 0.5) / 0.5
                fx = -A * math.cos(PI * v)
                fy = -lift * math.sin(PI * v) ** 1.3
                ang = 0.35 * math.sin(PI * v) * (1 - v) - 0.25 * math.sin(PI * v) * v
            feet.append((fx * w, fy * w, ang * w))
        self.feet_walk = feet
        bob = lerp(4.5, 9.0, rk) * w * math.sin(TAU_1 * (ph - 0.06)) ** 2
        contact = math.exp(-((ph % 0.5) / 0.055)) * w * lerp(0.07, 0.12, rk)
        self.walk_roll = w * (0.022 * math.sin(2 * TAU_1 * ph + 1.0) + lerp(0.045, 0.17, rk))
        self.head_walk_dy = -lerp(3.0, 5.0, rk) * w * math.sin(TAU_1 * (ph - 0.14)) ** 2 + bob * 0.4
        # ---- root transform -------------------------------------------------------------
        sq = clamp(p.squash + contact, -1.0, 1.0)
        sqy = 1 - 0.22 * sq
        sqx = 1 + 0.16 * sq
        shx = shy = 0.0
        self.shake_rot = 0.0
        if p.shake > 0:
            shx = p.shake * 2.6 * noise1(t * 31.0, 501)
            shy = p.shake * 1.6 * noise1(t * 29.0, 503)
            self.shake_rot = p.shake * 0.012 * noise1(t * 23.0, 507)
        M = skia.Matrix()
        M.preTranslate(p.x, p.y)
        M.preScale(s * f, s)
        M.preTranslate(shx, shy)
        M.preScale(sqx, sqy)
        self.M_root = M
        # body (relative to root)
        self.body_dy = SIT_DROP * sit - bob
        self.body_rot = p.lean + self.walk_roll + self.shake_rot
        B = skia.Matrix()
        B.preTranslate(0, self.body_dy)
        B.preRotate(math.degrees(self.body_rot), 0, PIVOT_Y)
        self.B = B
        self.M_body = skia.Matrix.Concat(M, B)
        # head (relative to body)
        hroll = -0.45 * self.walk_roll + 0.01 * noise1(t * 0.5, 91)
        Hm = skia.Matrix()
        Hm.preTranslate(0, NECK_Y + p.head_dy + self.head_walk_dy)
        Hm.preRotate(math.degrees(p.head_tilt + hroll * (1 if w > 0 else 0)))
        Hm.preTranslate(0, -HEAD_OFF)
        self.Hm = Hm
        self.M_head = skia.Matrix.Concat(self.M_body, Hm)
        inv = skia.Matrix()
        self.M_body.invert(inv)
        self.M_body_inv = inv

        # ---- legs (root-local) ----------------------------------------------------------
        sy_ = math.sin(self.yaw)
        self.legs = []
        for i, side in enumerate(('near', 'far')):
            hip_b = (-21.0 + 5 * sy_, HIP_Y) if side == 'near' else (21.0 + 5 * sy_, HIP_Y - 2)
            hip = B.mapXY(*hip_b)
            hip = (hip.fX, hip.fY)
            fx, fy, fang = feet[i]
            base_x = -19.0 + 4 * sy_ if side == 'near' else 21.0 + 4 * sy_
            base_y = 0.0 if side == 'near' else -3.0
            ank_s = (base_x + fx, base_y + ANKLE_Y + fy)
            knee_s = ((hip[0] + ank_s[0]) / 2 + 4, (hip[1] + ank_s[1]) / 2)
            ang_s = fang
            # sitting (body-local then mapped)
            dg = clamp(p.sit_dangle)
            kick = p.sit_kick * 11.0 * math.sin(t * 4.2 + (0 if side == 'near' else PI)) if sit > 0 else 0.0
            kb = (hip_b[0] + lerp(25, 21, dg), hip_b[1] + lerp(3, 6, dg))
            ab = (kb[0] + lerp(27, 4 + kick, dg), kb[1] + lerp(-2, 25 - abs(kick) * 0.2, dg))
            kn = B.mapXY(*kb)
            an = B.mapXY(*ab)
            ang_sit = lerp(-1.35, -0.1 + kick * 0.02, dg) + self.body_rot
            knee = (lerp(knee_s[0], kn.fX, sit), lerp(knee_s[1], kn.fY, sit))
            ank = (lerp(ank_s[0], an.fX, sit), lerp(ank_s[1], an.fY, sit))
            ang = lerp(ang_s, ang_sit, sit)
            self.legs.append(dict(hip=hip, knee=knee, ankle=ank, ang=ang, side=side))

        # ---- arms (body-local) ------------------------------------------------------------
        hs = _half(SHOULDER_Y)
        self.sh_r = (-hs * 0.93 + 3 * sy_, SHOULDER_Y)
        self.sh_l = (hs * (0.95 - 0.4 * max(0.0, sy_)) + 2 * sy_, SHOULDER_Y - 2)
        swing = lerp(0.32, 0.8, rk) * w
        elb = lerp(0.12, 1.35, rk) * w
        cph = math.cos(TAU_1 * ph)
        arm_r = (p.arm_r[0] - swing * cph, p.arm_r[1] + elb + 0.1 * w * math.sin(TAU_1 * ph))
        arm_l = (p.arm_l[0] + swing * cph, p.arm_l[1] + elb - 0.1 * w * math.sin(TAU_1 * ph))
        open_r, open_l = p.hand_r_open, p.hand_l_open
        heart_c = self.heart_center()
        # hug: IK both hands to the porthole
        if p.hug > 0:
            k = smoothstep(0, 1, p.hug)
            a, e, _ = _ik(*self.sh_r, heart_c[0] - 8, heart_c[1] + 12, L_UP, L_LOW)
            arm_r = _lerp_arm(arm_r, (a, e), k)
            a, e, _ = _ik(*self.sh_l, heart_c[0] + 18, heart_c[1] - 2, L_UP, L_LOW)
            arm_l = _lerp_arm(arm_l, (a, e), k)
            open_r = lerp(open_r, 0.75, k)
            open_l = lerp(open_l, 0.75, k)
        if p.wave > 0:
            k = smoothstep(0, 1, p.wave)
            wv = wave_arm(t, 1.0, p.wave_speed)
            if p.wave_side == 'l':
                arm_l = _lerp_arm(arm_l, wv, k)
                open_l = lerp(open_l, 1.0, k)
            else:
                arm_r = _lerp_arm(arm_r, wv, k)
                open_r = lerp(open_r, 1.0, k)
        st_r = st_l = 1.0
        if p.arm_r_override is not None:
            q = inv.mapXY(*p.arm_r_override)
            a, e, st_r = _ik(*self.sh_r, q.fX, q.fY, L_UP, L_LOW)
            arm_r = (a, e)
        if p.arm_l_override is not None:
            q = inv.mapXY(*p.arm_l_override)
            a, e, st_l = _ik(*self.sh_l, q.fX, q.fY, L_UP, L_LOW)
            arm_l = (a, e)
        self.arm_r = self._fk(self.sh_r, arm_r, st_r, open_r, p.hand_r_rot)
        self.arm_l = self._fk(self.sh_l, arm_l, st_l, open_l, p.hand_l_rot)
        self._chin(p, Hm, B, sit, sy_)
        for arm, cup, other in ((self.arm_r, p.hand_r_cup, self.arm_l), (self.arm_l, p.hand_l_cup, self.arm_r)):
            dx = other['hand'][0] - arm['hand'][0]
            arm['cup'] = clamp(cup)
            arm['cup_dir'] = 1.0 if (dx >= 0 or abs(dx) < 5) else -1.0
        # near arm raised above the shoulders goes behind the head (keeps the face readable)
        ar = self.arm_r
        if p.arm_r_behind_head is not None:
            self.near_behind_head = bool(p.arm_r_behind_head)
        else:
            self.near_behind_head = (p.hug < 0.25 and p.chin_hands < 0.25 and ar['a'] > 2.35 and
                                     ar['hand'][1] < SHOULDER_Y - 75 and ar['hand'][0] < 45)
        # far arm layering
        if p.arm_l_front is not None:
            self.far_front = bool(p.arm_l_front)
        else:
            al = self.arm_l
            self.far_front = (p.hug > 0.25 or p.chin_hands > 0.25 or
                              (arm_l[0] > 0.45 and al['hand'][0] < self.sh_l[0] - 12 and
                               al['hand'][1] < SHOULDER_Y + 105))

    # ------------------------------------------------------------------------------------
    def _fk(self, sh, arm, stretch, open_, rot):
        a, e = arm
        l1, l2 = L_UP * stretch, L_FORE * stretch
        ex, ey = sh[0] + l1 * math.sin(a), sh[1] + l1 * math.cos(a)
        fa = a + e
        dx, dy = math.sin(fa), math.cos(fa)
        wx, wy = ex + l2 * dx, ey + l2 * dy
        hx, hy = wx + HAND_OFF * dx, wy + HAND_OFF * dy
        return dict(sh=sh, elbow=(ex, ey), wrist=(wx, wy), hand=(hx, hy), dir=(dx, dy),
                    ang=math.atan2(dy, dx) + rot, open=clamp(open_), a=a)

    def _chin(self, p, Hm, B, sit, sy_):
        if p.chin_hands <= 0:
            return
        k = smoothstep(0, 1, p.chin_hands)
        chin = Hm.mapXY(8 * math.sin(self.hyaw), HH + 1)
        binv = skia.Matrix()
        B.invert(binv)
        for key, dxh, li, rot, ov in (('arm_r', -15, 0, p.hand_r_rot, p.arm_r_override),
                                      ('arm_l', 15, 1, p.hand_l_rot, p.arm_l_override)):
            if ov is not None:
                continue
            arm = getattr(self, key)
            kn = binv.mapXY(*self.legs[li]['knee'])
            belly = (12 + 20 * sy_ + (-14 if li == 0 else 14), -108)
            el_t = (lerp(belly[0], kn.fX + 2, sit), lerp(belly[1], kn.fY - 11, sit))
            hd_t = (chin.fX + dxh, chin.fY + 5)
            el = (lerp(arm['elbow'][0], el_t[0], k), lerp(arm['elbow'][1], el_t[1], k))
            hd = (lerp(arm['hand'][0], hd_t[0], k), lerp(arm['hand'][1], hd_t[1], k))
            setattr(self, key, self._pos_arm(arm['sh'], el, hd, lerp(arm['open'], 0.85, k), rot))

    def _pos_arm(self, sh, el, hd, open_, rot):
        dx, dy = hd[0] - el[0], hd[1] - el[1]
        L = math.hypot(dx, dy) or 1.0
        dx, dy = dx / L, dy / L
        wr = (hd[0] - dx * HAND_OFF, hd[1] - dy * HAND_OFF)
        return dict(sh=sh, elbow=el, wrist=wr, hand=hd, dir=(dx, dy), ang=math.atan2(dy, dx) + rot,
                    open=clamp(open_), a=math.atan2(el[0] - sh[0], el[1] - sh[1]))

    def heart_center(self):
        x, y, _ = _surf(0.0, PORT_Y0 + 8, self.yaw)
        return (x, PORT_Y0 + RY * _half(PORT_Y0 + 8) * math.cos(self.yaw))

    def light_local(self):
        """Rim light direction in body-local coordinates (unit)."""
        a = self.p.rim_angle
        v = self.M_body_inv.mapVector(math.cos(a), math.sin(a))
        L = math.hypot(v.fX, v.fY) or 1.0
        return v.fX / L, v.fY / L

    def eyes_local(self):
        sx = 20 * math.sin(self.hyaw)
        lk = self.p.look
        return (sx + clamp(lk[0], -1.5, 1.5) * 9, 4 - 5 + clamp(lk[1], -1.5, 1.5) * 6)

    def anchors(self):
        Mb, Mh, Mr = self.M_body, self.M_head, self.M_root

        def wb(x, y):
            q = Mb.mapXY(x, y)
            return (q.fX, q.fY)

        def wh(x, y):
            q = Mh.mapXY(x, y)
            return (q.fX, q.fY)

        def wr(x, y):
            q = Mr.mapXY(x, y)
            return (q.fX, q.fY)

        hy = self.hyaw
        ex, ey = self.eyes_local()
        ax = -8 * math.sin(hy)
        sway = self.antenna_sway()
        near_leg = self.legs[0]
        lap = (lerp(near_leg['hip'][0], near_leg['knee'][0], 0.7), near_leg['knee'][1] - 12)
        lap_stand = self.B.mapXY(24 + 18 * math.sin(self.yaw), -100)
        lap = (lerp(lap_stand.fX, lap[0], self.sit), lerp(lap_stand.fY, lap[1], self.sit))
        return {
            'l': wb(*self.arm_l['hand']), 'r': wb(*self.arm_r['hand']),
            'heart': wb(*self.heart_center()),
            'head': wh(0, 0),
            'eyes': wh(ex, ey),
            'antenna': wh(ax + sway * ANT_LEN, -HH - ANT_LEN - 2),
            'mouth': wh(20 * math.sin(hy) + self.p.look[0] * 4, 4 + 17),
            'chin': wh(8 * math.sin(hy), HH),
            'head_top': wh(20 + 12 * math.sin(hy), -HH - 2),
            'lap': wr(*lap),
            'feet': (self.p.x, self.p.y),
            'shoulder_l': wb(*self.sh_l), 'shoulder_r': wb(*self.sh_r),
            'elbow_l': wb(*self.arm_l['elbow']), 'elbow_r': wb(*self.arm_r['elbow']),
        }

    def antenna_sway(self):
        t = self.t
        return (0.07 * noise1(t * 0.8, 41) + 0.03 * noise1(t * 2.3, 43)
                + 0.14 * self.w * math.sin(2 * TAU_1 * self.ph - 1.9)
                - 0.6 * (self.p.head_tilt) * 0.3 + 0.06 * self.p.shake * noise1(t * 20, 45))


TAU_1 = math.tau


# ----------------------------------------------------------------------------------------
# drawing parts
# ----------------------------------------------------------------------------------------
def _draw_shadow(c, rig):
    p = rig.p
    a = p.shadow * (1 - rig.sit)
    if a <= 0.01:
        return
    s = p.scale
    lift = min(rig.feet_walk[0][1], rig.feet_walk[1][1])
    rx, ry = 78 * s, 13 * s
    paint = radial((0, 0), 1.0, [(0.0, '#000000', clamp(a)), (0.55, '#000000', clamp(a * 0.6)),
                                 (1.0, '#000000', 0.0)])
    c.save()
    c.translate(p.x + 4 * s * rig.f, p.y + 1 * s)
    c.scale(rx, ry)
    c.drawCircle(0, 0, 1.0, paint)
    c.restore()
    del lift


def _draw_tube(c, pts, r0, r1, pal, lod, cols=('tube_sh', 'tube', 'tube_lt'), ribs=True, light=(0.35, -0.94), rib_a=0.55):
    path = _tube_path(pts, r0, r1)
    c.drawPath(path, fill(pal[cols[0]]))
    if lod == 0:
        q = _offset_path(path, light[0] * r0 * 0.4, light[1] * r0 * 0.4)
        c.save()
        c.clipPath(path, skia.ClipOp.kIntersect, True)
        c.drawPath(q, fill(pal[cols[1]]))
        c.restore()
        return path
    c.save()
    c.clipPath(path, skia.ClipOp.kIntersect, True)
    q = _offset_path(path, light[0] * r0 * 0.5, light[1] * r0 * 0.5)
    c.drawPath(q, fill(pal[cols[1]]))
    hl = [(x + light[0] * r0 * 0.5, y + light[1] * r0 * 0.5) for x, y in pts]
    c.drawPath(_polyline(hl), stroke(pal[cols[2]], (r0 + r1) * 0.28, a=0.75))
    if ribs:
        rp = skia.Path()
        n = len(pts)
        for i in range(1, n - 1, 1 if lod == 2 else 2):
            x, y = pts[i]
            dx, dy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
            L = math.hypot(dx, dy) or 1
            dx, dy = dx / L, dy / L
            r = lerp(r0, r1, i / (n - 1)) * 1.05
            nx, ny = -dy, dx
            rp.moveTo(x + nx * r, y + ny * r)
            rp.quadTo(x + dx * r * 0.45, y + dy * r * 0.45, x - nx * r, y - ny * r)
        c.drawPath(rp, stroke(pal[cols[0]], max(0.9, r0 * 0.14), a=rib_a))
    c.restore()
    return path


def _draw_boot(c, ankle, ang, pal, lod, dark=0.0):
    with_ = c.save()
    c.translate(*ankle)
    c.rotate(math.degrees(ang))
    base = mix(pal['boot'], pal['boot_dk'], dark)
    lt = mix(pal['boot_lt'], pal['boot'], dark * 0.8)
    # sole
    sole = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-19, 10, 27, 17.5), 5, 5)
    c.drawRRect(sole, fill(mix(pal['iron_dk'], pal['shadow'], dark * 0.5)))
    up = skia.Path()
    up.moveTo(-19, 12)
    up.cubicTo(-22, -6, -6, -11, 6, -6)
    up.cubicTo(18, -2, 29, 2, 28, 12)
    up.close()
    c.drawPath(up, radial((2, -8), 30, [(0.0, lt, 1.0), (0.5, base, 1.0), (1.0, mix(base, pal['boot_dk'], 0.6), 1.0)]))
    if lod >= 1:
        # brass toe cap + ankle ring
        cap = skia.Path()
        cap.moveTo(14, -1)
        cap.cubicTo(22, 1, 29, 4, 28, 12)
        cap.lineTo(16, 12)
        cap.cubicTo(18, 7, 17, 2, 14, -1)
        c.drawPath(cap, fill(mix(pal['brass'], pal['brass_dk'], 0.35 + dark * 0.4), 0.9))
        c.drawOval(skia.Rect.MakeLTRB(-8, -9, 11, -3), fill(mix(pal['brass_dk'], pal['shadow'], dark * 0.4)))
        c.drawCircle(18, 4, 2.2, fill(pal['brass_lt'], 0.5 * (1 - dark)))
    c.restoreToCount(with_)


def _draw_leg(c, rig, leg, pal, lod, dark=0.0):
    hip, knee, ank = leg['hip'], leg['knee'], leg['ankle']
    c1 = (lerp(hip[0], knee[0], 0.8), lerp(hip[1], knee[1], 0.8))
    c2 = (lerp(ank[0], knee[0], 0.8), lerp(ank[1], knee[1], 0.8))
    pts = _bez(hip, c1, c2, ank, 8 if lod else 4)
    cols = ('tube_sh', 'tube', 'tube_lt') if dark < 0.2 else ('iron_dk', 'tube_sh', 'tube')
    _draw_tube(c, pts, 13.0, 11.5, pal, lod, cols=cols, ribs=lod >= 1, rib_a=0.45)
    _draw_boot(c, ank, leg['ang'], pal, lod, dark)


def _draw_cup_hand(c, arm, pal, lod, near):
    """Palm-up cupped mitten (in body-local axes, fingers curl up towards cup_dir)."""
    hx, hy = arm['hand']
    sd = arm.get('cup_dir', 1.0)
    dk = 0.0 if near else 0.25
    base = mix(pal['hand'], pal['hand_sh'], dk)
    lt = mix(pal['hand_lt'], pal['hand'], dk)
    sh = mix(pal['hand_sh'], pal['iron_dk'], dk)
    o = arm['open']
    sv0 = c.save()
    c.translate(hx, hy)
    c.scale(HAND_SCALE, HAND_SCALE)
    c.translate(-hx, -hy)
    # far-side finger first (darker), palm bowl, near finger + thumb on top
    f1 = _tube_path([(hx + sd * 7, hy + 3), (hx + sd * 13, hy + 1), (hx + sd * 16, hy - 5 - 3 * o)], 5.4, 5.0)
    c.drawPath(f1, fill(sh))
    bowl = skia.Path()
    bowl.moveTo(hx - sd * 15, hy - 3)
    bowl.cubicTo(hx - sd * 14, hy + 13, hx + sd * 12, hy + 14, hx + sd * 15, hy - 1)
    bowl.cubicTo(hx + sd * 8, hy + 2, hx - sd * 8, hy + 1, hx - sd * 15, hy - 3)
    bowl.close()
    c.drawPath(bowl, radial((hx - sd * 3, hy + 2), 18, [(0, lt, 1), (0.6, base, 1), (1, sh, 1)]))
    # inside of the palm (slightly darker) visible from above
    c.drawOval(skia.Rect.MakeLTRB(hx - 12, hy - 4, hx + 12, hy + 3), fill(sh, 0.55))
    f2 = _tube_path([(hx + sd * 5, hy + 6), (hx + sd * 12, hy + 5), (hx + sd * 18, hy - 1 - 3 * o)], 5.6, 5.2)
    c.drawPath(f2, fill(base))
    if lod >= 1:
        c.drawPath(_offset_path(f2, -0.6, -1.0), fill(lt, 0.5))
    th = _tube_path([(hx - sd * 12, hy + 1), (hx - sd * 14, hy - 5), (hx - sd * 12, hy - 10)], 5.2, 4.8)
    c.drawPath(th, fill(base))
    c.restoreToCount(sv0)
    # wrist cuff along the forearm
    sv = c.save()
    wx, wy = arm['wrist']
    c.translate(wx, wy)
    c.rotate(math.degrees(math.atan2(arm['dir'][1], arm['dir'][0])))
    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-4, -9, 3, 9), 3, 3),
                linear((0, -9), (0, 9), [(0, pal['brass_lt'], 1), (0.5, pal['brass'], 1), (1, pal['brass_dk'], 1)]))
    c.restoreToCount(sv)


def _draw_hand(c, arm, pal, lod, glow_a, heart_col, near=True):
    hx, hy = arm['hand']
    o = arm['open']
    if arm.get('cup', 0.0) > 0.5:
        _draw_cup_hand(c, arm, pal, lod, near)
        if glow_a > 0.01:
            _glow(c, hx, hy, 24, heart_col, glow_a)
        return
    sv = c.save()
    c.translate(hx, hy)
    c.rotate(math.degrees(arm['ang']))
    c.scale(HAND_SCALE, HAND_SCALE)
    dk = 0.0 if near else 0.25
    base = mix(pal['hand'], pal['hand_sh'], dk)
    lt = mix(pal['hand_lt'], pal['hand'], dk)
    sh = mix(pal['hand_sh'], pal['iron_dk'], dk)
    fp = fill(sh)
    fl = fill(base)
    # fingers (two) + thumb, capsules from palm
    fingers = []
    for i, (by, a_open, a_fist) in enumerate(((-3.5, -0.12, 0.9), (4.5, 0.38, 1.25))):
        ang = lerp(a_fist, a_open, o)
        ln = lerp(3.0, 12.5, o)
        bx = 5.0
        ex, ey = bx + math.cos(ang) * ln, by + math.sin(ang) * ln
        fingers.append(((bx, by), (ex, ey)))
    for (b, e) in fingers:
        pth = _tube_path([b, ((b[0] + e[0]) / 2, (b[1] + e[1]) / 2), e], 5.6, 5.2)
        c.drawPath(pth, fp)
        if lod >= 1:
            c.drawPath(_offset_path(pth, -0.8, -1.2), fl)
    # palm
    c.drawOval(skia.Rect.MakeLTRB(-11, -11.5, 13, 12), radial((-1, -5), 16, [(0.0, lt, 1.0), (0.55, base, 1.0), (1.0, sh, 1.0)]))
    # thumb on top (thumb side = -y)
    ta = lerp(-1.05, -0.75, o)
    tl = lerp(4.0, 10.0, o)
    tb = (1.5, -6.5)
    te = (tb[0] + math.cos(ta) * tl, tb[1] + math.sin(ta) * tl)
    tp = _tube_path([tb, ((tb[0] + te[0]) / 2, (tb[1] + te[1]) / 2), te], 5.4, 5.0)
    c.drawPath(tp, fill(base))
    if lod >= 1:
        c.drawPath(_offset_path(tp, -0.6, -1.0), fill(lt, 0.55))
        # knuckle highlight
        c.drawCircle(1, -6, 3.2, fill(pal['hand_lt'], 0.35 * (1 - dk)))
    # wrist cuff (brass)
    cuff = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-16, -9, -9, 9), 3, 3)
    c.drawRRect(cuff, linear((0, -9), (0, 9), [(0, pal['brass_lt'], 1), (0.5, pal['brass'], 1), (1, pal['brass_dk'], 1)]))
    c.restoreToCount(sv)
    if glow_a > 0.01:
        _glow(c, hx, hy, 24, heart_col, glow_a)


def _draw_arm(c, rig, arm, pal, lod, near, glow_col, glow_amt, heart_c):
    sh, el, wr = arm['sh'], arm['elbow'], arm['wrist']
    c1 = (lerp(sh[0], el[0], 0.8), lerp(sh[1], el[1], 0.8))
    c2 = (lerp(wr[0], el[0], 0.8), lerp(wr[1], el[1], 0.8))
    n = (16 if lod == 2 else 11) if lod else 6
    pts = _bez(sh, c1, c2, wr, n)
    dk = 0.0 if near else 0.3
    cols = ('tube_sh', 'tube', 'tube_lt') if near else ('tube_sh', 'tube_sh', 'tube')
    # shoulder socket
    c.drawCircle(sh[0], sh[1], 13.5, fill(mix(pal['brass_dk'], pal['iron_dk'], dk)))
    _draw_tube(c, pts, 9.0, 7.0, pal, lod, cols=cols, ribs=lod >= 1)
    # shoulder cap
    c.drawCircle(sh[0], sh[1], 10.5, radial((sh[0] - 2, sh[1] - 4), 12,
                 [(0, mix(pal['brass_lt'], pal['brass'], dk), 1), (0.7, mix(pal['brass'], pal['brass_dk'], dk), 1),
                  (1, pal['brass_dk'], 1)]))
    # elbow ring at the curve's middle
    mid = pts[n // 2]
    a, b = pts[n // 2 - 1], pts[n // 2 + 1]
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    sv = c.save()
    c.translate(*mid)
    c.rotate(math.degrees(ang))
    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-4, -10, 4, 10), 3, 3),
                linear((0, -10), (0, 10), [(0, mix(pal['brass_lt'], pal['brass'], dk), 1), (0.55, mix(pal['brass'], pal['brass_dk'], dk), 1),
                                           (1, pal['brass_dk'], 1)]))
    c.restoreToCount(sv)
    # heart light on hands
    hx, hy = arm['hand']
    d = math.hypot(hx - heart_c[0], hy - heart_c[1])
    ga = glow_amt * 0.3 * clamp(1 - d / 170.0) ** 1.5
    _draw_hand(c, arm, pal, lod, ga, glow_col, near)


def _draw_scarf_tail(c, root, length, width, seed, rig, pal, lod, dirx, a_hang, front):
    p, t = rig.p, rig.t
    we = clamp(p.scarf_wind * 0.95 + 0.35 * rig.w + 0.45 * rig.w * rig.rk, 0.0, 1.2)
    we_e = we ** 1.1 if we < 1 else 1.0
    n = 11 if lod else 6
    seg = length / n
    a_stream = dirx * (PI / 2 - 0.1 - 0.08 * min(we, 1))
    base = lerp(a_hang, a_stream, clamp(we_e))
    freq = 1.1 + 2.6 * we
    amp0 = 0.14 + 0.5 * we
    pts = [root]
    widths = [width]
    shade = [0.0]
    x, y = root
    for i in range(1, n + 1):
        s = i / n
        ai = base * (1 - 0.3 * s * (1 - clamp(we)))
        amp = amp0 * s ** 0.75
        ai += amp * (noise1(t * freq - s * 2.3, seed) + 0.5 * noise1(t * freq * 2.1 - s * 4.1, seed + 1))
        ai -= rig.body_rot * 0.8
        x += math.sin(ai) * seg
        y += math.cos(ai) * seg
        pts.append((x, y))
        tw = 1 - 0.45 * max(0.0, noise1(t * freq * 0.55 - s * 1.7, seed + 5)) * s * clamp(we + 0.2)
        widths.append(width * (1 - 0.12 * s) * tw)
        shade.append(1 - tw)
    # outline
    left, right = [], []
    for i, (px, py) in enumerate(pts):
        if i == 0:
            dx, dy = pts[1][0] - px, pts[1][1] - py
        elif i == n:
            dx, dy = px - pts[i - 1][0], py - pts[i - 1][1]
        else:
            dx, dy = pts[i + 1][0] - pts[i - 1][0], pts[i + 1][1] - pts[i - 1][1]
        L = math.hypot(dx, dy) or 1
        nx, ny = -dy / L, dx / L
        wv = widths[i] / 2
        left.append((px + nx * wv, py + ny * wv))
        right.append((px - nx * wv, py - ny * wv))
    path = skia.Path()
    path.moveTo(*left[0])
    for q in left[1:]:
        path.lineTo(*q)
    for q in reversed(right):
        path.lineTo(*q)
    path.close()
    dk = 0.0 if front else 0.3
    base_c = mix(pal['red'], pal['red_dk'], dk)
    c.drawPath(path, fill(base_c))
    c.save()
    c.clipPath(path, skia.ClipOp.kIntersect, True)
    # twist shading per segment
    for i in range(n):
        if shade[i + 1] > 0.05:
            q = skia.Path()
            q.moveTo(*left[i]); q.lineTo(*left[i + 1]); q.lineTo(*right[i + 1]); q.lineTo(*right[i]); q.close()
            c.drawPath(q, fill(pal['red_dk'], clamp(shade[i + 1] * 1.3)))
    # soft highlight along one edge / shade along the other
    hl = [(lerp(l[0], r[0], 0.25), lerp(l[1], r[1], 0.25)) for l, r in zip(left, right)]
    sh = [(lerp(l[0], r[0], 0.85), lerp(l[1], r[1], 0.85)) for l, r in zip(left, right)]
    c.drawPath(_polyline(hl), stroke(pal['red_lt'], width * 0.28, a=0.35 * (1 - dk)))
    c.drawPath(_polyline(sh), stroke(pal['red_dk'], width * 0.32, a=0.45))
    if lod >= 1:
        # knit ribs across + cream stripes near the end
        rp = skia.Path()
        m = 3 if lod == 2 else 2
        for i in range(n):
            for j in range(m):
                u = (j + 0.5) / m
                a0 = (lerp(left[i][0], left[i + 1][0], u), lerp(left[i][1], left[i + 1][1], u))
                b0 = (lerp(right[i][0], right[i + 1][0], u), lerp(right[i][1], right[i + 1][1], u))
                rp.moveTo(*a0)
                rp.lineTo(*b0)
        c.drawPath(rp, stroke(pal['red_dk'], max(0.8, width * 0.07), a=0.35))
        for s0, s1 in ((0.72, 0.77), (0.83, 0.88)):
            i0, i1 = int(s0 * n), min(n, int(math.ceil(s1 * n)))
            u0, u1 = s0 * n - i0, s1 * n - (i1 - 1)
            q = skia.Path()
            la = (lerp(left[i0][0], left[i0 + 1][0], u0), lerp(left[i0][1], left[i0 + 1][1], u0))
            ra = (lerp(right[i0][0], right[i0 + 1][0], u0), lerp(right[i0][1], right[i0 + 1][1], u0))
            lb = (lerp(left[i1 - 1][0], left[i1][0], u1), lerp(left[i1 - 1][1], left[i1][1], u1))
            rb = (lerp(right[i1 - 1][0], right[i1][0], u1), lerp(right[i1 - 1][1], right[i1][1], u1))
            q.moveTo(*la); q.lineTo(*lb); q.lineTo(*rb); q.lineTo(*ra); q.close()
            c.drawPath(q, fill(mix(pal['cream'], pal['red_dk'], dk), 0.72))
    c.restore()
    # fringe
    if lod >= 1:
        ex, ey = pts[-1]
        dx, dy = pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]
        L = math.hypot(dx, dy) or 1
        dx, dy = dx / L, dy / L
        nx, ny = -dy, dx
        fp = skia.Path()
        for k in range(4):
            u = (k / 3 - 0.5) * widths[-1] * 0.85
            wig = 0.35 * noise1(t * freq * 1.3 + k * 1.7, seed + 9) * (0.3 + we)
            fdx, fdy = dx * math.cos(wig) - dy * math.sin(wig), dx * math.sin(wig) + dy * math.cos(wig)
            fp.moveTo(ex + nx * u, ey + ny * u)
            fp.lineTo(ex + nx * u + fdx * 8.5, ey + ny * u + fdy * 8.5)
        c.drawPath(fp, stroke(mix(pal['red'], pal['red_dk'], 0.3 + dk), 2.4))
    return path


def _scarf_wrap_path():
    return _scarf_wrap_cached()


@lru_cache(maxsize=1)
def _scarf_wrap_cached():
    m = 24
    rt, rb = 45.0, 55.0
    yt, yb = -223.0, -197.0
    p = skia.Path()
    for i in range(m + 1):
        ph = -PI / 2 + PI * i / m
        (p.moveTo if i == 0 else p.lineTo)(rt * math.sin(ph), yt + 0.1 * rt * math.cos(ph))
    p.cubicTo(rt + 10, yt + 2, rb + 7, yb - 8, rb, yb)
    for i in range(m + 1):
        ph = PI / 2 - PI * i / m
        p.lineTo(rb * math.sin(ph), yb + 0.17 * rb * math.cos(ph))
    p.cubicTo(-rb - 7, yb - 8, -rt - 10, yt + 2, -rt, yt)
    p.close()
    return p


def _draw_scarf_wrap(c, rig, pal, lod, knot_x):
    path = _scarf_wrap_path()
    yaw = rig.yaw
    c.drawPath(path, linear((0, -226), (0, -186), [(0, pal['red_lt'], 1), (0.35, pal['red'], 1), (1, pal['red_dk'], 1)]))
    c.save()
    c.clipPath(path, skia.ClipOp.kIntersect, True)
    # cylinder shading
    c.drawRect(skia.Rect.MakeLTRB(-64, -230, 64, -180), linear((-60, 0), (60, 0), [
        (0, '#000000', 0.45), (0.25, '#000000', 0.12), (0.5 + 0.3 * math.sin(yaw) - 0.1, '#000000', 0.0),
        (0.85, '#000000', 0.12), (1, '#000000', 0.4)]))
    if lod >= 1:
        # knit stitches: rows of little V's following the ring curvature
        vp = skia.Path()
        rows = 3
        for r in range(rows):
            v = (r + 0.5) / rows
            y0 = lerp(-221.0, -199.0, v)
            rr = lerp(46.0, 54.0, v)
            ry = lerp(0.1, 0.17, v) * rr
            step = 0.22 if lod == 2 else 0.3
            th = -PI
            while th < PI:
                ph = th + yaw + (step * 0.5 if r % 2 else 0)
                cs = math.cos(ph)
                if cs > 0.12:
                    x = rr * math.sin(ph)
                    y = y0 + ry * cs
                    vw = 2.6 * cs
                    vp.moveTo(x - vw, y - 2.4)
                    vp.lineTo(x, y + 1.0)
                    vp.lineTo(x + vw, y - 2.4)
                th += step
        c.drawPath(vp, stroke(pal['red_dk'], 1.1, a=0.5))
        c.drawPath(_offset_path(vp, 0, -1.1), stroke(pal['red_lt'], 0.7, a=0.28))
    # head shadow on scarf top
    c.drawRect(skia.Rect.MakeLTRB(-64, -228, 64, -205), linear((0, -226), (0, -208), [(0, '#000000', 0.35), (1, '#000000', 0.0)]))
    c.restore()
    # knot
    kx, ky = knot_x, -198
    c.drawOval(skia.Rect.MakeLTRB(kx - 11, ky - 11, kx + 11, ky + 10),
               radial((kx - 3, ky - 5), 14, [(0, pal['red_lt'], 1), (0.6, pal['red'], 1), (1, pal['red_dk'], 1)]))
    if lod >= 1:
        kp = skia.Path()
        kp.moveTo(kx - 7, ky - 4)
        kp.quadTo(kx, ky + 1, kx + 6, ky - 6)
        c.drawPath(kp, stroke(pal['red_dk'], 1.4, a=0.6))


def _draw_body(c, rig, pal, lod, glow_col, glow_amt):
    p, yaw, t = rig.p, rig.yaw, rig.t
    back = p.back
    yv = yaw + (PI if back else 0.0)
    # plinth
    pl = _plinth_path()
    c.drawPath(pl, linear((-PLINTH_HALF, 0), (PLINTH_HALF, 0), [
        (0, pal['body_dk'], 1), (0.35, pal['body_sh'], 1), (0.62, pal['body'], 1), (1, pal['body_dk'], 1)]))
    c.save()
    c.clipPath(pl, skia.ClipOp.kIntersect, True)
    c.drawRect(skia.Rect.MakeLTRB(-80, -72, 80, -40), linear((0, -70), (0, -48), [(0, '#000000', 0.0), (1, '#000000', 0.35)]))
    # brass trim line on plinth
    tp = skia.Path()
    for i in range(25):
        ph = -PI / 2 + PI * i / 24
        (tp.moveTo if i == 0 else tp.lineTo)(PLINTH_HALF * math.sin(ph), -66 + RY * PLINTH_HALF * math.cos(ph))
    c.drawPath(tp, stroke(pal['brass_dk'], 2.2, a=0.8))
    if lod >= 1:
        _rivet_ring(c, -71.5, yv, pal, lod, half=PLINTH_HALF - 1, step=0.34)
    c.restore()
    if p.rim > 0:
        lx, ly = rig.light_local()
        _rim(c, pl, lx, ly, max(3.0, 1.6 / max(0.05, rig.px)), p.rim_color, 0.4 * p.rim)

    # barrel
    bp = _body_path()
    c.drawPath(bp, fill(pal['body']))
    c.save()
    c.clipPath(bp, skia.ClipOp.kIntersect, True)
    # red stripe (lighthouse band)
    c.drawPath(_band_path(STRIPE_Y0, STRIPE_Y1), fill(pal['red']))
    if lod >= 1:
        c.drawPath(_band_path(STRIPE_Y0, STRIPE_Y0 + 2.2), fill(pal['red_lt'], 0.5))
    # rust
    rust_list = _RUST if lod >= 1 else _RUST[:3]
    for th, y, r, sd in rust_list:
        x, yy, cs = _surf(th, y, yv)
        if cs < 0.08:
            continue
        pts = [(x + bx * r * cs * 1.1, yy + by * r) for bx, by in _blob(sd)]
        path = smooth_path(pts, close=True, tension=0.5)
        c.drawPath(path, radial((x, yy), r * 1.25, [(0, pal['rust_dk'], 0.85), (0.6, pal['rust'], 0.7), (1, pal['rust'], 0.0)]))
        if lod >= 1 and sd % 2 == 0:
            # drip streak
            c.drawRect(skia.Rect.MakeLTRB(x - 1.2 * cs, yy + r * 0.5, x + 1.2 * cs, yy + r * 2.8),
                       linear((0, yy + r * 0.5), (0, yy + r * 2.8), [(0, pal['rust'], 0.55), (1, pal['rust'], 0.0)]))
    # riveted patch
    if not back:
        _draw_patch(c, yv, pal, lod)
    else:
        _draw_hatch(c, yv, pal, lod)
    # vertical panel seam with rivets
    if lod >= 1:
        for th in (-1.45, 1.6):
            x0, _, cs = _surf(th, -190, yv)
            if cs < 0.15:
                continue
            sp = skia.Path()
            first = True
            for i in range(12):
                y = lerp(-196, -80, i / 11)
                x, yy, _ = _surf(th, y, yv)
                (sp.moveTo if first else sp.lineTo)(x, yy - RY * _half(y) * math.cos(th + yv))
                first = False
            c.drawPath(sp, stroke(pal['body_dk'], 1.3, a=0.55 * cs))
            c.drawPath(_offset_path(sp, 1.3, 0), stroke(pal['body_lt'], 0.9, a=0.35 * cs))
    # rivet rings along the stripe
    if lod >= 1:
        _rivet_ring(c, STRIPE_Y0 - 5, yv, pal, lod, step=0.36)
        _rivet_ring(c, STRIPE_Y1 + 5, yv, pal, lod, step=0.36)
    # cylinder shading (one overlay for everything painted on the barrel)
    hx = 0.5 + 0.34 * math.sin(yaw)
    c.drawRect(skia.Rect.MakeLTRB(-80, -210, 80, -60), linear((-72, 0), (72, 0), [
        (0, '#000000', 0.5), (0.16, '#000000', 0.25), (clamp(hx - 0.2, 0.2, 0.9), '#000000', 0.0),
        (clamp(hx + 0.22, 0.25, 0.95), '#000000', 0.04), (1, '#000000', 0.42)]))
    hp = linear((-72, 0), (72, 0), [(0, '#ffffff', 0.0), (clamp(hx - 0.14, 0.01, 0.98), '#ffffff', 0.0),
                                    (clamp(hx - 0.02, 0.02, 0.99), '#ffffff', 0.13), (clamp(hx + 0.1, 0.03, 0.995), '#ffffff', 0.0),
                                    (1, '#ffffff', 0.0)])
    hp.setBlendMode(SCREEN)
    c.drawRect(skia.Rect.MakeLTRB(-80, -210, 80, -60), hp)
    # AO under scarf/head, bottom darkening
    c.drawRect(skia.Rect.MakeLTRB(-80, -210, 80, -160), linear((0, -200), (0, -168), [(0, '#000000', 0.5), (1, '#000000', 0.0)]))
    c.drawRect(skia.Rect.MakeLTRB(-80, -110, 80, -60), linear((0, -100), (0, -66), [(0, '#000000', 0.0), (1, '#000000', 0.3)]))
    # near-arm shadow on the body (ambient occlusion)
    for arm, on in ((rig.arm_r, True), (rig.arm_l, rig.far_front)):
        if not on:
            continue
        sh, el, wr = arm['sh'], arm['elbow'], arm['wrist']
        c1 = (lerp(sh[0], el[0], 0.8) + 4, lerp(sh[1], el[1], 0.8) + 6)
        c2 = (lerp(wr[0], el[0], 0.8) + 4, lerp(wr[1], el[1], 0.8) + 6)
        ap = _polyline(_bez((sh[0] + 3, sh[1] + 5), c1, c2, (wr[0] + 4, wr[1] + 6), 8))
        c.drawPath(ap, stroke('#000000', 22, a=0.22, blur=5.0 if lod else 0.0))
        hx_, hy_ = arm['hand']
        c.drawCircle(hx_ + 4, hy_ + 6, 14, fill('#000000', 0.2, blur=5.0 if lod else 0.0))
    # heart light cast on the belly
    if glow_amt > 0.01 and not back:
        hc = rig.heart_center()
        g = radial((hc[0], hc[1] + 4), 110, [(0, glow_col, clamp(0.15 * glow_amt)), (0.45, glow_col, clamp(0.06 * glow_amt)),
                                           (1, glow_col, 0.0)])
        g.setBlendMode(ADD)
        c.drawCircle(hc[0], hc[1] + 4, 110, g)
    c.restore()
    # rim light
    if p.rim > 0:
        lx, ly = rig.light_local()
        wd = max(3.0, 1.6 / max(0.05, rig.px))
        _rim(c, bp, lx, ly, wd, p.rim_color, 0.55 * p.rim)
    return bp


def _rivet_ring(c, y, yaw, pal, lod, half=None, step=0.35):
    sp, bp_, hp = skia.Path(), skia.Path(), skia.Path()
    th = -PI
    h = _half(y) if half is None else half
    r0 = 1.9 if lod == 2 else 2.2
    while th < PI:
        ph = th + yaw
        cs = math.cos(ph)
        if cs > 0.18:
            x = h * math.sin(ph)
            yy = y + RY * h * cs
            rw = r0 * (0.35 + 0.65 * cs)
            sp.addOval(skia.Rect.MakeLTRB(x - rw, yy - r0 + 0.9, x + rw, yy + r0 + 0.9))
            bp_.addOval(skia.Rect.MakeLTRB(x - rw, yy - r0, x + rw, yy + r0))
            hp.addCircle(x - rw * 0.3, yy - r0 * 0.35, r0 * 0.38)
        th += step
    c.drawPath(sp, fill('#000000', 0.35))
    c.drawPath(bp_, fill(pal['body_lt']))
    if lod == 2:
        c.drawPath(hp, fill('#ffffff', 0.45))


def _draw_patch(c, yaw, pal, lod):
    th, y = -0.98, -110.0
    x, yy, cs = _surf(th, y, yaw)
    if cs < 0.1:
        return
    sv = c.save()
    c.translate(x, yy)
    c.rotate(7)
    c.scale(max(0.25, cs), 1)
    r = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-16, -15, 16, 15), 4, 4)
    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-15, -13, 18, 18), 4, 4), fill('#000000', 0.28, blur=1.5 if lod else 0))
    c.drawRRect(r, linear((-16, -15), (16, 15), [(0, pal['patch_lt'], 1), (0.4, pal['patch'], 1), (1, pal['patch_dk'], 1)]))
    if lod >= 1:
        # bent edge highlight / rivets
        c.drawLine(-13, -14, 13, -14, stroke(pal['patch_lt'], 1.2, a=0.8))
        for (px, py) in ((-11, -10), (11, -10), (-11, 10), (11, 10), (0, -11), (0, 11)):
            c.drawCircle(px, py + 0.8, 2.1, fill('#000000', 0.35))
            c.drawCircle(px, py, 2.0, fill(pal['patch_lt']))
            if lod == 2:
                c.drawCircle(px - 0.6, py - 0.7, 0.8, fill('#ffffff', 0.55))
        # a rust edge on the patch
        c.drawCircle(12, 12, 5, radial((12, 12), 6, [(0, pal['rust'], 0.7), (1, pal['rust'], 0.0)]))
    c.restoreToCount(sv)


def _draw_hatch(c, yaw, pal, lod):
    """Back view: a riveted service hatch where the porthole would be."""
    x, yy, cs = _surf(PI, PORT_Y0 + 8, yaw)
    if cs < 0.1:
        return
    sv = c.save()
    c.translate(x, yy + 10)
    c.scale(max(0.25, cs), 1)
    r = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-26, -30, 26, 30), 9, 9)
    c.drawRRect(r, fill(pal['body_dk']))
    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-23, -27, 23, 27), 7, 7), linear((0, -27), (0, 27), [
        (0, pal['body_lt'], 1), (0.5, pal['body'], 1), (1, pal['body_sh'], 1)]))
    for i in range(4):
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-14, -16 + i * 9, 14, -12 + i * 9), 2, 2), fill(pal['body_dk'], 0.8))
    if lod >= 1:
        for (px, py) in ((-19, -23), (19, -23), (-19, 23), (19, 23)):
            c.drawCircle(px, py, 2.2, fill(pal['brass']))
    c.restoreToCount(sv)


def draw_heart_core(c, x, y, scale=1.0, brightness=1.0, t=0.0, lod=1):
    """The flame-like amber heart core, centred at (x, y) (world or local units).
    scale 1 = the size it has inside Deng's porthole at robot scale 1. Scenes can use this
    to draw the core in his hand after he lifts it out (s06)."""
    b = clamp(brightness)
    if b <= 0.005:
        return
    sv = c.save()
    c.translate(x, y)
    c.scale(scale, scale)
    k = 0.6 + 0.4 * b
    fl = 1.0 + 0.07 * noise1(t * 7.0, 11) + 0.04 * noise1(t * 19.0, 12)
    sway = 2.6 * noise1(t * 2.1, 5) + 1.1 * noise1(t * 5.9, 9)
    hgt = (15.0 + 2.2 * noise1(t * 4.3, 3)) * k
    rr = 10.0 * k
    _glow(c, 0, 3, 34, HEART, 0.5 * b * fl)
    outer = _flame_path(sway, hgt, rr)
    dim = 1 - b
    c0 = mix(HEART_CORE, HEART, dim * 0.8)
    c1 = mix(HEART_MID, HEART_DEEP, dim * 0.8)
    c2 = mix(HEART, '#a0401c', dim * 0.8)
    c.drawPath(outer, radial((0, 7), rr * 1.9, [(0, c0, clamp(b * 1.3)), (0.4, c1, clamp(b * 1.2)),
                                                 (0.8, c2, clamp(b * 1.1)), (1, HEART_DEEP, clamp(b * 0.9))]))
    inner = _flame_path(sway * 0.6, hgt * 0.55, rr * 0.55, oy=9.0)
    ip = fill(EYE_CORE if b > 0.5 else HEART_MID, clamp(b * 1.1 * fl))
    ip.setBlendMode(ADD)
    c.drawPath(inner, ip)
    if lod >= 1 and b > 0.2:
        # tiny embers rising
        for i in range(3):
            u = (t * (0.45 + 0.1 * i) + _hash(i, 211)) % 1.0
            ex = (_hash(i, 213) - 0.5) * 12 + 2.5 * math.sin(t * 3 + i * 2)
            ey = 10 - u * 26
            a = b * math.sin(PI * u) * 0.9
            ep = fill(HEART_CORE, clamp(a))
            ep.setBlendMode(ADD)
            c.drawCircle(ex, ey, 1.1 + 0.4 * (1 - u), ep)
    c.restoreToCount(sv)


def _flame_path(sway, hgt, r, oy=6.0):
    tipx, tipy = sway, oy - r - hgt
    p = skia.Path()
    p.moveTo(tipx, tipy)
    p.cubicTo(tipx + 2.5, tipy + hgt * 0.45, r, oy - r * 0.9, r, oy)
    p.cubicTo(r, oy + r * 0.56, r * 0.56, oy + r, 0, oy + r)
    p.cubicTo(-r * 0.56, oy + r, -r, oy + r * 0.56, -r, oy)
    p.cubicTo(-r, oy - r * 0.9, tipx - 2.5, tipy + hgt * 0.45, tipx, tipy)
    p.close()
    return p


def draw_star_heart(c, x, y, scale=1.0, amount=1.0, t=0.0, lod=1):
    """The new star-shaped heart (ending), centred at (x, y). amount 0..1 grows it in with a
    little birth flash around amount ~0.35; at 1 it glows steadily with a soft lub-dub pulse."""
    u = clamp(amount)
    if u <= 0.003:
        return
    sv = c.save()
    c.translate(x, y)
    sz = (0.3 + 0.7 * ease_out_back(smoothstep(0.05, 0.85, u))) * scale
    al = smoothstep(0.0, 0.4, u)
    flash = math.sin(PI * clamp((u - 0.12) / 0.55)) ** 2
    # heartbeat: lub-dub
    ph = (t % 1.15) / 1.15
    beat = math.exp(-((ph - 0.05) / 0.05) ** 2) + 0.6 * math.exp(-((ph - 0.22) / 0.05) ** 2)
    pul = 1 + 0.07 * beat
    c.scale(sz, sz)
    if lod == 0:
        _glow(c, 0, 0, 90, STAR_GLOW, 0.5 * al)
    _glow(c, 0, 0, 110, STAR_GLOW, (0.12 + 0.25 * flash) * al * pul)
    _glow(c, 0, 0, 46, STAR_GLOW, (0.42 + 0.3 * flash) * al * pul)
    if flash > 0.02 and lod >= 1:
        # birth sparkle rays
        rp = stroke('#fff6d8', 1.3, a=clamp(0.55 * flash))
        rp.setBlendMode(ADD)
        for i in range(10):
            ang = i / 10 * math.tau + 0.3 + t * 0.4
            r0 = 22 + 6 * (i % 2)
            r1 = r0 + (12 + 12 * (i % 2)) * flash
            c.drawLine(math.cos(ang) * r0, math.sin(ang) * r0, math.cos(ang) * r1, math.sin(ang) * r1, rp)
    c.save()
    c.rotate(math.degrees(0.07 * math.sin(t * 0.9)))
    c.scale(pul * 18.5, pul * 18.5)
    sp = _star_path()
    c.drawPath(sp, radial((0, -0.05), 1.05, [(0, STAR_CORE, al), (0.3, '#fff6cf', al), (0.68, '#ffdd6e', al),
                                              (1.0, '#ffb53c', al)]))
    ip = fill('#fffdf2', clamp(al * (0.45 + 0.3 * beat + 0.4 * flash)))
    ip.setBlendMode(ADD)
    c.scale(0.42, 0.42)
    c.drawPath(sp, ip)
    c.restore()
    # twinkle glints
    if lod >= 1:
        for (gx, gy, gs, ph0) in ((13, -13, 8, 0.0), (-12, 10, 5.5, 2.1)):
            k = (0.5 + 0.5 * math.sin(t * 2.3 + ph0)) * al
            if k < 0.05:
                continue
            gp = stroke('#ffffff', 1.3, a=clamp(k))
            gp.setBlendMode(ADD)
            L = gs * (0.6 + 0.6 * k)
            c.drawLine(gx - L, gy, gx + L, gy, gp)
            c.drawLine(gx, gy - L, gx, gy + L, gp)
    c.restoreToCount(sv)


def _draw_porthole(c, rig, pal, lod, heart_amt, star_amt, core_amt):
    p, t = rig.p, rig.t
    hx, hy = rig.heart_center()
    cs = math.cos(rig.yaw)
    fx = max(0.8, cs * 0.9 + 0.12)
    sv = c.save()
    c.translate(hx, hy)
    c.scale(fx, 1.0)
    R = PORT_R
    # AO ring on body + flange
    c.drawCircle(0, 2.5, R + 3, fill('#000000', 0.3, blur=2.5 if lod else 0))
    c.drawCircle(0, 0, R, linear((-R * 0.6, -R), (R * 0.6, R), [(0, pal['brass_lt'], 1), (0.45, pal['brass'], 1), (1, pal['brass_dk'], 1)]))
    c.drawCircle(0, 0, R - 5.5, fill(pal['brass_dk']))
    if lod >= 1:
        bp = skia.Path()
        hp = skia.Path()
        for i in range(8):
            a = i / 8 * math.tau + PI / 8
            bx, by = math.cos(a) * (R - 2.8), math.sin(a) * (R - 2.8)
            bp.addCircle(bx, by, 1.8)
            hp.addCircle(bx - 0.5, by - 0.6, 0.8)
        c.drawPath(bp, fill(pal['brass_dk']))
        c.drawPath(hp, fill(pal['brass_lt'], 0.8))
    # cavity
    r_in = R - 7.0
    c.drawCircle(0, 0, r_in, radial((0, 3), r_in, [(0, mix(pal['cavity_lt'], HEART_DEEP, 0.25 * heart_amt), 1), (1, pal['cavity'], 1)]))
    if lod >= 1 and core_amt < 0.99:
        # cradle prongs / contacts (visible when empty or dim)
        pr = skia.Path()
        for a in (-2.4, -0.74, PI / 2):
            pr.moveTo(math.cos(a) * r_in, math.sin(a) * r_in)
            pr.lineTo(math.cos(a) * (r_in - 7), math.sin(a) * (r_in - 7))
        c.drawPath(pr, stroke(pal['brass_dk'], 2.4))
        c.drawCircle(0, 0, 5, fill(pal['iron_dk']))
        c.drawCircle(-1, -1, 2, fill(pal['brass_dk'], 0.8))
    c.save()
    c.clipPath(_circle_path(r_in), skia.ClipOp.kIntersect, True)
    if core_amt > 0.005:
        draw_heart_core(c, 0, 0, 1.0, heart_amt * core_amt, t, lod)
    c.restore()
    if star_amt > 0.003:
        draw_star_heart(c, 0, 0, 1.0, star_amt, t, lod)
    # the glass door on its hinge (left side)
    th = clamp(p.heart_open) * 1.95
    hinge_x = -(R - 3.0)
    ksc = math.cos(th)
    if th > 0.02:
        # shadow of the door on the body
        c.drawOval(skia.Rect.MakeLTRB(hinge_x - (R - 4) * math.sin(th) * 1.6 - 4, -R + 2, hinge_x + 2, R + 2),
                   fill('#000000', 0.25 * math.sin(min(th, PI / 2)), blur=3 if lod else 0))
    c.save()
    c.translate(hinge_x, 0)
    c.scale(ksc if abs(ksc) > 0.02 else 0.02, 1.0 + 0.06 * math.sin(th))
    c.translate(-hinge_x, 0)
    backside = ksc < 0
    rd = r_in + 2.5
    c.drawCircle(0, 0, rd, stroke(pal['brass_dk'] if backside else pal['brass'], 4.0))
    glass_a = 0.22 if not backside else 0.35
    c.drawCircle(0, 0, rd - 1.5, fill('#9fd0e0' if not backside else '#40606a', glass_a * (0.6 + 0.4 * (1 - heart_amt))))
    if not backside:
        # reflections
        rp = skia.Path()
        rp.addArc(skia.Rect.MakeLTRB(-rd + 4, -rd + 4, rd - 4, rd - 4), 195, 80)
        c.drawPath(rp, stroke('#ffffff', 3.0, a=0.5))
        c.drawCircle(-rd * 0.45, -rd * 0.55, 1.8, fill('#ffffff', 0.8))
        c.drawCircle(rd * 0.3, rd * 0.45, 5, fill('#ffffff', 0.06))
        # latch knob on the right
        c.drawCircle(R - 2, 0, 3.4, fill(pal['brass_dk']))
        c.drawCircle(R - 2.5, -0.8, 2.4, fill(pal['brass_lt']))
    c.restore()
    # hinge knuckles
    for yy in (-10, 10):
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(hinge_x - 3, yy - 4.5, hinge_x + 3, yy + 4.5), 2, 2),
                    linear((hinge_x - 3, 0), (hinge_x + 3, 0), [(0, pal['brass_dk'], 1), (0.5, pal['brass_lt'], 1), (1, pal['brass_dk'], 1)]))
    c.restoreToCount(sv)


@lru_cache(maxsize=8)
def _circle_path(r):
    p = skia.Path()
    p.addCircle(0, 0, r)
    return p


def _head_path():
    return _head_path_cached()


@lru_cache(maxsize=1)
def _head_path_cached():
    p = skia.Path()
    # slightly wider at the top: a friendly old-TV / diving-helmet silhouette
    tw, bw = HW, HW - 5
    r = HR
    p.moveTo(-tw + r, -HH)
    p.lineTo(tw - r, -HH)
    p.cubicTo(tw - r * 0.35, -HH, tw, -HH + r * 0.35, tw, -HH + r)
    p.cubicTo(tw + 1.5, -HH + r + 25, bw + 1, HH - r - 5, bw, HH - r * 0.8)
    p.cubicTo(bw, HH - r * 0.2, bw - r * 0.35, HH, bw - r, HH)
    p.lineTo(-bw + r, HH)
    p.cubicTo(-bw + r * 0.35, HH, -bw, HH - r * 0.2, -bw, HH - r * 0.8)
    p.cubicTo(-bw - 1, HH - r - 5, -tw - 1.5, -HH + r + 25, -tw, -HH + r)
    p.cubicTo(-tw, -HH + r * 0.35, -tw + r * 0.35, -HH, -tw + r, -HH)
    p.close()
    return p


def _screen_geom(hy):
    scx = 20 * math.sin(hy)
    sw = 57 * (0.92 + 0.08 * math.cos(hy))
    sh = 41.0
    return scx, 4.0, sw, sh


def _draw_head(c, rig, pal, lod, P, Pvis):
    p, t, hy = rig.p, rig.t, rig.hyaw
    back = p.back
    shp = _head_path()
    sn = math.sin(hy)
    # ear knobs (behind shell)
    knob_x = -HW - 1 if sn >= 0 else HW + 1
    far_x = (HW - 7 - 10 * sn) if sn >= 0 else (-HW + 7 - 10 * sn)
    for kx, dk in ((far_x, 0.45), (knob_x, 0.0)):
        c.drawCircle(kx, 8, 12, fill(mix(pal['brass_dk'], pal['shadow'], dk * 0.5)))
        c.drawCircle(kx, 8, 9, radial((kx - 2, 5), 11, [(0, mix(pal['brass_lt'], pal['brass'], dk), 1), (1, mix(pal['brass_dk'], pal['shadow'], dk * 0.4), 1)]))
        if lod >= 1:
            c.drawLine(kx, 2, kx, 14, stroke(pal['brass_dk'], 2.0, a=0.8))
    # shell
    c.drawPath(shp, linear((0, -HH), (0, HH), [(0, pal['head_lt'], 1), (0.35, pal['head'], 1), (1, pal['head_sh'], 1)]))
    c.save()
    c.clipPath(shp, skia.ClipOp.kIntersect, True)
    # side plane (3/4): darker band on the turned-away side
    side = -1 if sn >= 0 else 1
    if back:
        side = -side
        c.drawPath(shp, fill('#000000', 0.14))
    wside = 16 + 34 * abs(sn)
    x0 = side * HW
    c.drawRect(skia.Rect.MakeLTRB(-HW - 4, -HH - 4, HW + 4, HH + 4), linear((x0, 0), (x0 - side * wside, 0), [
        (0, '#000000', 0.42), (0.7, '#000000', 0.12), (1, '#000000', 0.0)]))
    c.drawRect(skia.Rect.MakeLTRB(-HW - 4, -HH - 4, HW + 4, HH + 4), linear((-x0, 0), (-x0 + side * 22, 0), [
        (0, '#000000', 0.25), (1, '#000000', 0.0)]))
    # top highlight
    hp = radial((10 * sn, -HH + 8), 70, [(0, '#ffffff', 0.16), (1, '#ffffff', 0.0)])
    hp.setBlendMode(SCREEN)
    c.drawOval(skia.Rect.MakeLTRB(-70 + 10 * sn, -HH - 10, 70 + 10 * sn, -HH + 30), hp)
    # brass base band
    c.drawRect(skia.Rect.MakeLTRB(-HW, HH - 11, HW, HH + 2), linear((-HW, 0), (HW, 0), [
        (0, pal['brass_dk'], 1), (0.5 + 0.3 * sn, pal['brass_lt'], 1), (1, pal['brass_dk'], 1)]))
    c.drawRect(skia.Rect.MakeLTRB(-HW, HH - 12.5, HW, HH - 10.5), fill(pal['brass_dk'], 0.7))
    if lod >= 1:
        # rust freckle on the helmet + scratch
        rx = -44 + 10 * sn
        c.drawCircle(rx, 34, 6, radial((rx, 34), 7, [(0, pal['rust_dk'], 0.7), (1, pal['rust'], 0.0)]))
        if lod == 2:
            sp = skia.Path()
            sp.moveTo(38 + 8 * sn, -40)
            sp.quadTo(46 + 8 * sn, -36, 52 + 8 * sn, -38)
            c.drawPath(sp, stroke(pal['head_lt'], 1.0, a=0.6))
            bpp = skia.Path()
            for bx in (-60, -30, 0, 30, 60):
                bpp.addCircle(bx + 6 * sn, HH - 5, 1.6)
            c.drawPath(bpp, fill(pal['brass_lt'], 0.7))
    c.restore()

    if back:
        _draw_head_back(c, rig, pal, lod)
    else:
        _draw_screen(c, rig, pal, lod, P, Pvis)
    # rim light on shell
    if p.rim > 0:
        a = p.rim_angle
        inv = skia.Matrix()
        rig.M_head.invert(inv)
        v = inv.mapVector(math.cos(a), math.sin(a))
        L = math.hypot(v.fX, v.fY) or 1
        wd = max(3.0, 1.6 / max(0.05, rig.px))
        _rim(c, shp, v.fX / L, v.fY / L, wd, p.rim_color, 0.45 * p.rim)
    # antenna
    _draw_antenna(c, rig, pal, lod, P)
    if p.seaweed > 0:
        _draw_seaweed(c, rig, pal, lod)


def _draw_head_back(c, rig, pal, lod):
    sn = math.sin(rig.hyaw)
    cx = -16 * sn
    r = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(cx - 36, -30, cx + 36, 26), 14, 14)
    c.drawRRect(r, fill(pal['head_sh']))
    c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(cx - 33, -27, cx + 33, 23), 12, 12), linear((0, -27), (0, 23), [
        (0, pal['head_lt'], 1), (1, pal['head'], 1)]))
    for i in range(5):
        y = -18 + i * 9
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(cx - 22, y, cx + 22, y + 3.5), 1.7, 1.7), fill(pal['head_sh']))
    for (px, py) in ((-28, -22), (28, -22), (-28, 18), (28, 18)):
        c.drawCircle(cx + px, py, 2.2, fill(pal['brass']))


def _draw_screen(c, rig, pal, lod, P, Pvis):
    p, t, hy = rig.p, rig.t, rig.hyaw
    scx, scy, sw, sh = _screen_geom(hy)
    sr = 25.0
    bez = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(scx - sw - 7.5, scy - sh - 7.5, scx + sw + 7.5, scy + sh + 7.5), sr + 7, sr + 7)
    c.drawRRect(bez, fill('#000000', 0.3, blur=3 if lod else 0))
    c.drawRRect(bez, linear((0, scy - sh - 8), (0, scy + sh + 8), [(0, pal['brass_lt'], 1), (0.4, pal['brass'], 1), (1, pal['brass_dk'], 1)]))
    scr_rr = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(scx - sw, scy - sh, scx + sw, scy + sh), sr, sr)
    # inner bevel
    inner = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(scx - sw - 2.2, scy - sh - 2.2, scx + sw + 2.2, scy + sh + 2.2), sr + 2, sr + 2)
    c.drawRRect(inner, fill(pal['brass_dk']))
    if lod >= 1:
        bp = skia.Path()
        hp = skia.Path()
        for (bx, by) in ((-sw - 3.8, -sh + 6), (sw + 3.8, -sh + 6), (-sw - 3.8, sh - 6), (sw + 3.8, sh - 6),
                         (0, -sh - 3.8), (0, sh + 3.8)):
            bp.addCircle(scx + bx, scy + by, 1.9)
            hp.addCircle(scx + bx - 0.5, scy + by - 0.6, 0.8)
        c.drawPath(bp, fill(pal['brass_dk']))
        c.drawPath(hp, fill(pal['brass_lt'], 0.9))
    # glass (dark, not ambient-tinted much: it is glass)
    scr = hexrgb('#10202a')
    c.drawRRect(scr_rr, radial((scx - 8, scy - 6), sw * 1.25, [(0, '#18313d', 1), (0.6, scr, 1), (1, '#060d12', 1)]))
    sv = c.save()
    c.clipRRect(scr_rr, skia.ClipOp.kIntersect, True)
    if P > 0.002:
        # CRT backlight
        g = radial((scx, scy), sw * 1.2, [(0, '#3b8c9c', 0.2 * P), (0.6, '#1f5a68', 0.1 * P), (1, '#1f5a68', 0.0)])
        g.setBlendMode(ADD)
        c.drawRRect(scr_rr, g)
        _draw_face(c, rig, pal, lod, P, Pvis, scx, scy, sw, sh)
        if lod == 2:
            lp = skia.Path()
            y = scy - sh
            while y < scy + sh:
                lp.moveTo(scx - sw, y)
                lp.lineTo(scx + sw, y)
                y += 3.2
            c.drawPath(lp, stroke('#000000', 0.9, a=0.07))
    # inset shadow along the top of the glass
    c.drawRect(skia.Rect.MakeLTRB(scx - sw, scy - sh, scx + sw, scy - sh + 22),
               linear((0, scy - sh), (0, scy - sh + 20), [(0, '#000000', 0.5), (1, '#000000', 0.0)]))
    # glass reflections (not electric)
    amb = _palette(p.ambient)['head_lt']
    rp = skia.Path()
    rp.moveTo(scx - sw + 7, scy + sh * 0.1)
    rp.cubicTo(scx - sw + 5, scy - sh * 0.7, scx - sw * 0.55, scy - sh + 5, scx + sw * 0.05, scy - sh + 5)
    rp.cubicTo(scx - sw * 0.45, scy - sh + 11, scx - sw + 13, scy - sh * 0.5, scx - sw + 7, scy + sh * 0.1)
    rp.close()
    c.drawPath(rp, fill(mix(amb, '#ffffff', 0.6), 0.16))
    c.drawCircle(scx - sw + 14, scy - sh + 12, 2.6, fill('#ffffff', 0.55))
    c.drawCircle(scx - sw + 20.5, scy - sh + 9, 1.3, fill('#ffffff', 0.4))
    br = skia.Path()
    br.moveTo(scx + sw * 0.55, scy + sh - 4)
    br.quadTo(scx + sw - 6, scy + sh - 5, scx + sw - 4, scy + sh * 0.35)
    c.drawPath(br, stroke('#ffffff', 2.2, a=0.1))
    c.restoreToCount(sv)


def _draw_face(c, rig, pal, lod, P, Pvis, scx, scy, sw, sh):
    p, t, hy = rig.p, rig.t, rig.hyaw
    E = _expr_mix(p.eyes, p.eyes2, p.eyes_mix)
    ex, ey = rig.eyes_local()
    ey += E['dy']
    sn = math.sin(hy)
    boost = 1.22 if lod == 0 else 1.0
    sep = 23.5 + E['sep']
    blink = clamp(p.blink)
    for side in (-1, 1):
        persp = 1 - 0.16 * sn * side  # far eye narrower
        _draw_eye(c, ex + side * sep * (1 - 0.06 * abs(sn)), ey, side, E, blink, Pvis, P, persp * boost, boost, lod, t)
    # mouth
    mx = scx + clamp(p.look[0], -1.5, 1.5) * 4
    my = scy + 17 + clamp(p.look[1], -1.5, 1.5) * 3 + E['dy'] * 0.3
    _draw_mouth(c, mx, my, p.mouth, clamp(p.smile + E['smile'], -1, 1), E['o'], Pvis, P, boost, sn)
    if p.blush > 0:
        for side in (-1, 1):
            bx = ex + side * (sep + 7)
            by = ey + 19
            g = radial((bx, by), 13, [(0, BLUSH, 0.55 * p.blush * Pvis), (1, BLUSH, 0.0)])
            g.setBlendMode(ADD)
            c.drawOval(skia.Rect.MakeLTRB(bx - 13, by - 7, bx + 13, by + 7), g)
            if lod >= 1:
                lp = stroke(BLUSH, 1.6, a=0.7 * p.blush * Pvis)
                lp.setBlendMode(ADD)
                for k in (-1, 0, 1):
                    c.drawLine(bx + k * 5 - 1.5, by + 2.5, bx + k * 5 + 1.5, by - 2.5, lp)
    if p.tear > 0:
        tx, ty = ex - sep - 5, ey + 19 + 2 * math.sin(t * 1.3)
        tp = skia.Path()
        tp.moveTo(tx, ty - 7)
        tp.cubicTo(tx + 1.5, ty - 3, tx + 4.5, ty, tx + 4.5, ty + 2.5)
        tp.cubicTo(tx + 4.5, ty + 5.5, tx - 4.5, ty + 5.5, tx - 4.5, ty + 2.5)
        tp.cubicTo(tx - 4.5, ty, tx - 1.5, ty - 3, tx, ty - 7)
        c.drawPath(tp, fill(TEAR, clamp(p.tear * Pvis)))
        _glow(c, tx, ty + 1, 14, TEAR, 0.35 * p.tear * P)


def _eye_paint(x, y, r, a):
    return radial((x, y), r, [(0, EYE_CORE, a), (0.55, EYE, a), (1, EYE_EDGE, a)])


def _draw_eye(c, ex, ey, side, E, blink, Pvis, P, wmul, boost, lod, t):
    inner = -side
    w = E['w'] * wmul
    h = E['h'] * boost
    cl = clamp(max(E['close'], blink))
    arc = clamp(E['arc'])
    shimmer = 1.0 + 0.04 * noise1(t * 3.0, 300 + side)
    a_open = (1 - arc) * Pvis
    # glow halo
    vis = (1 - cl * 0.7) * (1 - E['top'] * 0.5)
    _glow(c, ex, ey, max(w, h) * 1.35 + 6, EYE_GLOW, 0.33 * P * vis * shimmer)
    if a_open > 0.004:
        hh = h * (1 - cl)
        cy = ey + h * 0.2 * cl
        oval_a = a_open * smoothstep(2.5, 6.5, hh)
        line_a = a_open * (1 - smoothstep(2.5, 6.5, hh))
        if oval_a > 0.004:
            rect = skia.Rect.MakeLTRB(ex - w / 2, cy - hh / 2, ex + w / 2, cy + hh / 2)
            top, tilt, bot = E['top'], E['tilt'], E['bot']
            clip = top > 0.005 or bot > 0.005 or abs(tilt) > 0.01
            if clip:
                c.save()
                base = cy - hh / 2 + top * hh
                cp = skia.Path()
                xs = (ex - w, ex + w)
                ytop = [base + tilt * hh * 0.5 * ((x - ex) * inner / (w / 2)) for x in xs]
                cp.moveTo(xs[0], ytop[0])
                cp.lineTo(xs[1], ytop[1])
                for k in range(6):
                    u = 1 - k / 5
                    x = lerp(ex - w, ex + w, u)
                    dxn = (x - ex) / (w / 2)
                    yb = cy + hh / 2 - bot * hh * (0.55 + 0.45 * max(0.0, 1 - dxn * dxn))
                    cp.lineTo(x, yb)
                cp.close()
                c.clipPath(cp, skia.ClipOp.kIntersect, True)
            c.drawOval(rect, _eye_paint(ex, cy - hh * 0.12, max(w, hh) * 0.62, clamp(oval_a * shimmer)))
            if E['hl'] > 0.01 and hh > 8:
                hp = fill('#ffffff', clamp(E['hl'] * oval_a * 0.9))
                c.drawOval(skia.Rect.MakeXYWH(ex + w * 0.08 * side * 0 + w * 0.05, cy - hh * 0.36, w * 0.3, w * 0.3 * 1.1), hp)
                c.drawCircle(ex - w * 0.18, cy + hh * 0.2, w * 0.07, fill('#ffffff', clamp(E['hl'] * oval_a * 0.6)))
                if E['spark'] > 0.01:
                    sp = stroke('#ffffff', 1.2, a=clamp(E['spark'] * oval_a))
                    gx, gy, L = ex - w * 0.15, cy - hh * 0.05, w * 0.22
                    c.drawLine(gx - L, gy, gx + L, gy, sp)
                    c.drawLine(gx, gy - L, gx, gy + L, sp)
            if clip:
                c.restore()
        if line_a > 0.004:
            lp = skia.Path()
            lp.moveTo(ex - w * 0.52, cy - 1)
            lp.quadTo(ex, cy + 5, ex + w * 0.52, cy - 1)
            c.drawPath(lp, stroke(EYE, 4.2 * boost, a=clamp(line_a)))
    if arc > 0.004:
        ap = skia.Path()
        ap.moveTo(ex - w * 0.58, ey + 5)
        ap.cubicTo(ex - w * 0.45, ey - h * 0.42, ex + w * 0.45, ey - h * 0.42, ex + w * 0.58, ey + 5)
        c.drawPath(ap, stroke(EYE, 5.6 * boost, a=clamp(arc * Pvis)))
    if E['brow'] > 0.01:
        by = ey - h / 2 - 6 + E['by']
        bw = w * 0.6
        xi, xo = ex + inner * bw, ex - inner * bw
        yi = by - E['btilt'] * 7
        yo = by + E['btilt'] * 3
        bp = skia.Path()
        bp.moveTo(xi, yi)
        bp.quadTo((xi + xo) / 2, (yi + yo) / 2 - 3.5, xo, yo)
        c.drawPath(bp, stroke(EYE, 3.6 * boost, a=clamp(E['brow'] * Pvis)))


def _draw_mouth(c, mx, my, m, k, o, Pvis, P, boost, sn):
    m = clamp(m)
    w = 27.0 * (1 - 0.08 * abs(sn)) * boost
    col_a = clamp(Pvis)
    _glow(c, mx, my + m * 6, 26 + m * 10, EYE_GLOW, 0.2 * P)
    if o > 0.01:
        rw, rh = 6 + m * 3, 7.5 + m * 5
        c.drawOval(skia.Rect.MakeLTRB(mx - rw, my - rh + 4, mx + rw, my + rh + 4), stroke(EYE, 3.4 * boost, a=clamp(col_a * o)))
        if o > 0.99:
            return
        col_a *= (1 - o)
    if m < 0.06:
        p = skia.Path()
        p.moveTo(mx - w / 2, my - k * 2.5)
        p.quadTo(mx, my + k * 10, mx + w / 2, my - k * 2.5)
        c.drawPath(p, stroke(EYE, 3.8 * boost, a=col_a))
        return
    w2 = w / 2 * (1 - 0.22 * m)
    top = k * 5
    bot = k * 5 + m * 17 + 3
    p = skia.Path()
    p.moveTo(mx - w2, my - k * 2.5)
    p.quadTo(mx, my + top, mx + w2, my - k * 2.5)
    p.quadTo(mx + w2 * 0.9, my + bot * 0.7, mx, my + bot * 0.95)
    p.quadTo(mx - w2 * 0.9, my + bot * 0.7, mx - w2, my - k * 2.5)
    p.close()
    c.drawPath(p, fill(EYE, col_a * 0.32))
    # little tongue-glow at the bottom when open
    if m > 0.2:
        c.save()
        c.clipPath(p, skia.ClipOp.kIntersect, True)
        c.drawOval(skia.Rect.MakeLTRB(mx - w2 * 0.6, my + bot * 0.5, mx + w2 * 0.6, my + bot * 1.25),
                   fill(EYE_EDGE, col_a * 0.75 * smoothstep(0.2, 0.6, m)))
        c.restore()
    c.drawPath(p, stroke(EYE, 3.4 * boost, a=col_a))


def _draw_antenna(c, rig, pal, lod, P):
    p, t = rig.p, rig.t
    sn = math.sin(rig.hyaw)
    ax = -8 * sn
    by = -HH - 1
    sway = rig.antenna_sway()
    tx, ty = ax + sway * ANT_LEN, by - ANT_LEN - 1
    # base cap
    c.drawOval(skia.Rect.MakeLTRB(ax - 11, by - 5, ax + 11, by + 5), linear((0, by - 5), (0, by + 5), [(0, pal['brass_lt'], 1), (1, pal['brass_dk'], 1)]))
    rod = skia.Path()
    rod.moveTo(ax, by - 3)
    rod.quadTo(ax + sway * ANT_LEN * 0.25, by - ANT_LEN * 0.55, tx, ty + BULB_R * 0.6)
    c.drawPath(rod, stroke(pal['brass_dk'], 3.6))
    c.drawPath(_offset_path(rod, -0.7, 0), stroke(pal['brass_lt'], 1.2, a=0.8))
    if lod >= 1:
        # spring coil at the base
        sp = skia.Path()
        for i in range(5):
            yy = by - 5 - i * 2.2
            sp.moveTo(ax - 3.2, yy)
            sp.lineTo(ax + 3.2, yy - 1.2)
        c.drawPath(sp, stroke(pal['brass'], 1.5))
    # bulb
    if p.antenna_blink:
        ph = (t * p.antenna_rate / 2.2 + 0.3) % 1.0
        v = smoothstep(0.0, 0.04, ph) * (1 - smoothstep(0.24, 0.34, ph))
        lvl = 0.16 + 0.84 * v
    else:
        lvl = 0.85
    lvl = clamp(lvl * P + p.antenna_flash * 1.2 * max(P, 0.3))
    c.drawCircle(tx, ty, BULB_R, radial((tx - 2, ty - 2.5), BULB_R * 1.3, [
        (0, mix(pal['bulb_off'], BULB_HOT, lvl), 1), (0.5, mix(pal['bulb_off'], BULB_ON, lvl), 1),
        (1, mix(pal['bulb_off'], '#b8302a', lvl * 0.8), 1)]))
    c.drawCircle(tx - 2.4, ty - 2.6, 2.0, fill('#ffffff', 0.55 + 0.3 * lvl))
    if lvl > 0.02:
        _glow(c, tx, ty, 22 + 16 * p.antenna_flash, BULB_ON, 0.55 * lvl)
        if p.antenna_flash > 0.05:
            rp = stroke('#ffe0d0', 1.6, a=clamp(p.antenna_flash))
            rp.setBlendMode(ADD)
            for i in range(8):
                a = i / 8 * math.tau + t
                r0, r1 = BULB_R + 4, BULB_R + 4 + 9 * p.antenna_flash
                c.drawLine(tx + math.cos(a) * r0, ty + math.sin(a) * r0, tx + math.cos(a) * r1, ty + math.sin(a) * r1, rp)


def _draw_seaweed(c, rig, pal, lod):
    """A comic strand of seaweed draped over his helmet (after the splash in s05)."""
    p, t = rig.p, rig.t
    a = clamp(p.seaweed)
    sw = 1.5 * math.sin(t * 1.7)
    strands = (
        # (start, over-the-top control, end, width, seed)
        ((10, -HH - 6), (-46, -HH - 22), (-HW - 10 + sw, 14), 13.0, 1),
        ((16, -HH - 5), (58, -HH - 16), (HW - 14 + sw, -8), 11.0, 2),
    )
    for (p0, cp, p1, wd, sd) in strands:
        pts = []
        for i in range(13):
            u = i / 12
            x = (1 - u) ** 2 * p0[0] + 2 * (1 - u) * u * cp[0] + u * u * p1[0]
            y = (1 - u) ** 2 * p0[1] + 2 * (1 - u) * u * cp[1] + u * u * p1[1]
            x += 2.2 * math.sin(u * 9 + sd + t * 1.3) * u
            pts.append((x, y))
        path = _tube_path(pts, wd * 0.6, wd * 0.35, radii=[wd * (0.5 + 0.14 * math.sin(i * 1.7 + sd)) * (1 - 0.35 * i / 12)
                                                           for i in range(13)])
        c.drawPath(path, fill(pal['weed_dk'], a))
        c.drawPath(_offset_path(path, -0.8, -1.2), fill(pal['weed'], a))
        if lod >= 1:
            c.drawPath(_polyline(pts), stroke(pal['weed_lt'], 1.1, a=0.55 * a))
            # little blades
            for i in (3, 6, 9):
                x, y = pts[i]
                ang = math.atan2(pts[i + 1][1] - pts[i - 1][1], pts[i + 1][0] - pts[i - 1][0]) + (0.9 if i % 2 else -0.9)
                c.save()
                c.translate(x, y)
                c.rotate(math.degrees(ang))
                c.drawOval(skia.Rect.MakeLTRB(0, -3.2, 14, 3.2), fill(pal['weed'], a))
                c.restore()
        # a drip at the end
        ex, ey = pts[-1]
        dy = (t * 0.9 + sd * 0.37) % 1.0
        c.drawCircle(ex, ey + 4 + dy * 14, 1.8 * (1 - dy), fill('#bfe4ff', 0.7 * a * (1 - dy)))


# ----------------------------------------------------------------------------------------
# main entry
# ----------------------------------------------------------------------------------------
def _lod(c, pose):
    if pose.detail is not None:
        return int(pose.detail), 1.0
    try:
        k = c.getTotalMatrix().getMinScale()
    except Exception:
        k = 1.0
    px = pose.scale * (k if k > 0 else 1.0)
    return (0 if px < 0.42 else (1 if px < 1.25 else 2)), px


def draw_robot(c, pose: RobotPose, t: float = 0.0, part: str = 'all'):
    """Draw Deng. Returns dict with world positions:
    {'l': hand_l, 'r': hand_r, 'heart': heart_center, 'head': head_center, 'eyes': (x, y),
     'antenna': tip, 'mouth', 'chin', 'head_top', 'lap', 'feet', 'shoulder_l', 'shoulder_r',
     'elbow_l', 'elbow_r'}.

    part: 'all' (default) | 'back' (everything except the arms that pass in front of his
    body) | 'front' (only those front arms/hands). Draw 'back', then a held prop (Guang),
    then 'front' to put his mitten hands in front of what he holds."""
    if not (abs(pose.scale) > 1e-4):
        return {k: (pose.x, pose.y) for k in ('l', 'r', 'heart', 'head', 'eyes', 'antenna', 'mouth', 'chin',
                                                'head_top', 'lap', 'feet', 'shoulder_l', 'shoulder_r',
                                                'elbow_l', 'elbow_r')}
    rig = _Rig(pose, t)
    lod, px = _lod(c, pose)
    rig.px = px
    pal = _palette(pose.ambient)
    P = _power_level(pose.power, t)
    Pvis = clamp(P ** 0.55) if P > 0 else 0.0
    back = pose.back
    do_back = part in ('all', 'back')
    do_front = part in ('all', 'front')
    # heart state
    star = clamp(pose.heart_star)
    core_amt = (1.0 if pose.heart_present else 0.0) * (1 - smoothstep(0.0, 0.7, star))
    heart_amt = clamp(pose.heart)
    open_boost = 1 + 0.4 * clamp(pose.heart_open)
    glow_core = heart_amt * core_amt * open_boost
    glow_star = smoothstep(0.0, 0.5, star) * 1.1
    glow_amt = clamp((glow_core + glow_star) * pose.glow_cast, 0, 1.6)
    glow_col = mix(HEART, '#ffe7a0', clamp(star)) if glow_amt > 0 else hexrgb(HEART)
    hc = rig.heart_center()
    near_behind = rig.near_behind_head

    sv = c.save()
    if do_back:
        _draw_shadow(c, rig)
        # ---------- legs behind (root space) -------------------------------------------
        legs_front = rig.sit > 0.5
        c.save()
        c.concat(rig.M_root)
        if not legs_front:
            _draw_leg(c, rig, rig.legs[1], pal, lod, dark=0.3)
            _draw_leg(c, rig, rig.legs[0], pal, lod, dark=0.0)
            if rig.sit < 0.5:
                c.save()
                c.concat(rig.B)
                k = 1 - rig.sit * 2
                ao = radial((0, 0), 1.0, [(0, '#000000', 0.55 * k), (0.6, '#000000', 0.25 * k), (1, '#000000', 0.0)])
                c.translate(0, HIP_Y + 4)
                c.scale(64, 22)
                c.drawCircle(0, 0, 1.0, ao)
                c.restore()
        c.restore()

        # ---------- body space -----------------------------------------------------------
        c.save()
        c.concat(rig.M_body)
        sn = math.sin(rig.yaw)
        knot_x = -8.0 if back else -24.0 + 8 * sn
        if pose.scarf_dir is not None:
            dirx = (1.0 if pose.scarf_dir >= 0 else -1.0) * rig.f
        else:
            dirx = -1.0
        tail_b_root = (knot_x + 4, -200.0) if back else (44.0 * dirx, -209.0)
        tail_a_root = (knot_x - 1, -195.0)
        if not back:
            _draw_scarf_tail(c, tail_b_root, 94, 18, 7, rig, pal, lod, dirx, dirx * 0.1, front=False)
        if not rig.far_front:
            _draw_arm(c, rig, rig.arm_l, pal, lod, False, glow_col, glow_amt * 0.6, hc)
        _draw_body(c, rig, pal, lod, glow_col, glow_amt)
        if not back:
            _draw_porthole(c, rig, pal, lod, heart_amt, star, core_amt)
        c.restore()

        # ---------- legs in front when sitting ------------------------------------------
        if legs_front:
            c.save()
            c.concat(rig.M_root)
            _draw_leg(c, rig, rig.legs[1], pal, lod, dark=0.3)
            _draw_leg(c, rig, rig.legs[0], pal, lod, dark=0.0)
            c.restore()

        # ---------- neck, scarf wrap, raised near arm (behind head) --------------------------
        c.save()
        c.concat(rig.M_body)
        c.drawRRect(skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(-24, -232, 24, -205), 8, 8), fill(pal['iron_dk']))
        _draw_scarf_wrap(c, rig, pal, lod, knot_x)
        if back:
            _draw_scarf_tail(c, tail_b_root, 94, 18, 7, rig, pal, lod, dirx, dirx * 0.05, front=True)
        _draw_scarf_tail(c, tail_a_root, 74, 17, 3, rig, pal, lod, dirx, -dirx * 0.07, front=True)
        kx, ky = knot_x, -198
        c.drawOval(skia.Rect.MakeLTRB(kx - 10, ky - 10, kx + 10, ky + 9),
                   radial((kx - 3, ky - 4), 13, [(0, pal['red_lt'], 1), (0.6, pal['red'], 1), (1, pal['red_dk'], 1)]))
        if near_behind:
            _draw_arm(c, rig, rig.arm_r, pal, lod, True, glow_col, glow_amt, hc)
            _arm_rim(c, rig, rig.arm_r, px)
        c.restore()

        # ---------- head ---------------------------------------------------------------------
        c.save()
        c.concat(rig.M_head)
        _draw_head(c, rig, pal, lod, P, Pvis)
        if glow_amt > 0.01 and not back:
            g = radial((10 * math.sin(rig.hyaw), HH + 6), 60, [(0, glow_col, 0.28 * glow_amt), (1, glow_col, 0.0)])
            g.setBlendMode(ADD)
            c.save()
            c.clipPath(_head_path(), skia.ClipOp.kIntersect, True)
            c.drawCircle(10 * math.sin(rig.hyaw), HH + 6, 60, g)
            c.restore()
        c.restore()

    # ---------- arms in front ---------------------------------------------------------------
    if do_front:
        c.save()
        c.concat(rig.M_body)
        if rig.far_front:
            _draw_arm(c, rig, rig.arm_l, pal, lod, False, glow_col, glow_amt, hc)
        if not near_behind:
            _draw_arm(c, rig, rig.arm_r, pal, lod, True, glow_col, glow_amt, hc)
            _arm_rim(c, rig, rig.arm_r, px)
        # heart halo on top (lights up whatever passes in front of the porthole)
        if not back and glow_core > 0.01:
            r = 62 * (1.5 if lod == 0 else 1.0)
            _glow(c, hc[0], hc[1], r, HEART, 0.24 * clamp(glow_core))
            if lod == 0:
                _glow(c, hc[0], hc[1], 20, HEART_CORE, 0.6 * clamp(glow_core))
        c.restore()

    c.restoreToCount(sv)
    return rig.anchors()


def _arm_rim(c, rig, arm, px):
    p = rig.p
    if p.rim <= 0:
        return
    lx, ly = rig.light_local()
    wd = max(2.6, 1.4 / max(0.05, px))
    sh, el, wr = arm['sh'], arm['elbow'], arm['wrist']
    c1 = (lerp(sh[0], el[0], 0.8), lerp(sh[1], el[1], 0.8))
    c2 = (lerp(wr[0], el[0], 0.8), lerp(wr[1], el[1], 0.8))
    _rim(c, _tube_path(_bez(sh, c1, c2, wr, 10), 9.0, 7.0), lx, ly, wd * 0.8, p.rim_color, 0.45 * p.rim)
