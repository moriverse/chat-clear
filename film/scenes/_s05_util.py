"""Private helpers for s05 (想办法 Trying). Owner: SCENE-S05.

Everything here is a pure function of its inputs (frames render out of order).
Contains local fallbacks for effects the shared libraries may not (yet) provide,
plus scene-specific props (seaweed, the crate tower physics, water occlusion,
water reflections, the fake-3D lighthouse beam).
"""
from __future__ import annotations

import math

import numpy as np
import skia

from engine.core import (ADD, H, W, TAU, Camera, at, clamp, col, ease_in_out, fill, lerp, linear, mix, noise1,
                         poly, radial, rng, saved, smooth_path, smoothstep, soft_glow, stroke)
from lib import fx, robot, sea, star

SEA_MID = '#123060'
SEA_DEEP = '#081631'


# ----------------------------------------------------------------------------
# timing helpers
# ----------------------------------------------------------------------------
_ONSETS = {}


def syllable_onsets(f, line_id, n=4):
    """Local times of the first `n` syllable onsets of a dialogue line, found in the
    lip-sync loudness envelope (so the acting stays synced even if the voice changes).
    Falls back to evenly spaced beats."""
    s, e = f.line(line_id)
    key = (line_id, n)
    if key not in _ONSETS:
        rel = []
        env = f.tl.env.get(line_id)
        if env is not None and len(env) > 4:
            env = np.asarray(env, dtype=float)
            low = True
            for i, v in enumerate(env):
                if low and v > 0.5:
                    rel.append(i / 100.0)
                    low = False
                elif not low and v < 0.22:
                    low = True
        if len(rel) < n:
            dur = e - s
            rel = [dur * k / n for k in range(n)]
        _ONSETS[key] = rel[:n]
    return [s + r for r in _ONSETS[key]]


def smin(a, b, k=60.0):
    """Smooth minimum."""
    m = min(a, b)
    return m - k * math.log(math.exp(-(a - m) / k) + math.exp(-(b - m) / k))


def bump(x, rise=0.1, fall=0.3):
    """0 -> 1 -> 0 pulse starting at x=0 (seconds)."""
    if x <= 0:
        return 0.0
    if x < rise:
        return smoothstep(0, rise, x)
    return 1 - smoothstep(rise, rise + fall, x)


def damped(x, freq=3.0, decay=5.0):
    """Damped oscillation starting at 1 when x=0, 0 before."""
    if x < 0:
        return 0.0
    return math.exp(-decay * x) * math.cos(TAU * freq * x)


def damped_sin(x, freq=3.0, decay=5.0):
    if x < 0:
        return 0.0
    return math.exp(-decay * x) * math.sin(TAU * freq * x)


# ----------------------------------------------------------------------------
# robot / star wrappers (tolerant of library upgrades)
# ----------------------------------------------------------------------------
def make_pose(cls, **kw):
    """Construct a dataclass pose, silently dropping fields it does not know."""
    fields = getattr(cls, '__dataclass_fields__', {})
    good = {k: v for k, v in kw.items() if k in fields}
    return cls(**good)


def measure_robot(pose, T):
    """World anchor points of a robot pose without drawing it on screen."""
    fn = getattr(robot, 'anchors', None)
    if fn is not None:
        try:
            return fn(pose, T)
        except Exception:
            pass
    rec = skia.PictureRecorder()
    cc = rec.beginRecording(skia.Rect(-10, -10, 10, 10))
    res = robot.draw_robot(cc, pose, T)
    rec.finishRecordingAsPicture()
    return res


def head_frame(res, pose):
    """(head_top_xy, up_vector) of the robot from draw_robot's result."""
    hx, hy = res['head']
    if 'head_top' in res:
        tx, ty = res['head_top']
        ux, uy = tx - hx, ty - hy
        d = math.hypot(ux, uy) or 1.0
        return (tx, ty), (ux / d, uy / d)
    ax, ay = res.get('antenna', (hx, hy - 88 * pose.scale))
    ux, uy = ax - hx, ay - hy
    d = math.hypot(ux, uy) or 1.0
    ux, uy = ux / d, uy / d
    k = 0.57 * d
    return (hx + ux * k, hy + uy * k), (ux, uy)


def knee_point(res, pose):
    for key in ('lap', 'knee', 'knee_r', 'knee_l'):
        if key in res:
            return res[key]
    s = pose.scale
    return pose.x + pose.facing * 48 * s, pose.y - 78 * s


def hands_mid(res):
    (lx, ly), (rx, ry) = res['l'], res['r']
    return (lx + rx) / 2, (ly + ry) / 2


# ----------------------------------------------------------------------------
# sparkles (local fallback for fx.sparkle_burst)
# ----------------------------------------------------------------------------
def sparkle_shape(c, x, y, r, color='#fff4c0', a=1.0, rot=0.0):
    if a <= 0.01 or r <= 0.2:
        return
    k = 0.2
    pts = []
    for i in range(8):
        ang = rot + i * math.pi / 4
        rr = r if i % 2 == 0 else r * k
        pts.append((x + math.cos(ang) * rr, y + math.sin(ang) * rr))
    c.drawPath(poly(pts), fill(color, a, blend=ADD))
    soft_glow(c, x, y, r * 1.7, color, a * 0.35)


def _local_burst(c, x, y, t_since, count=16, speed=320.0, life=0.9, color='#fff4c0', seed=5, size=1.0):
    if not (0 <= t_since <= life):
        return
    r = rng(seed)
    u = t_since / life
    soft_glow(c, x, y, 150 * size * (0.6 + 0.6 * u), color, 0.8 * (1 - u) ** 2)
    for i in range(count):
        ang = r.uniform(0, TAU)
        v = speed * size * r.uniform(0.45, 1.0)
        # decelerating radial motion: d = v * (1 - e^{-k t}) / k
        kk = 3.2
        d = v * (1 - math.exp(-kk * t_since)) / kk
        px = x + math.cos(ang) * d
        py = y + math.sin(ang) * d + 40 * size * t_since * t_since
        tw = 0.6 + 0.4 * math.sin(t_since * 25 + i * 1.7)
        a = (1 - u) ** 1.3 * tw
        sparkle_shape(c, px, py, (7 + 6 * r.uniform(0, 1)) * size * (1 - 0.5 * u), color, a, rot=r.uniform(0, 0.8))


def sparkle_burst(c, x, y, t_since, count=16, speed=320.0, life=0.9, color='#fff4c0', seed=5, size=1.0,
                  ring=False, intensity=1.0):
    """Radial burst of 4-point sparkles. Uses lib.fx.sparkle_burst when available."""
    if t_since < 0 or t_since > life + 0.05:
        return
    fn = getattr(fx, 'sparkle_burst', None)
    if fn is not None and not getattr(fx, 'STUB', True):
        try:
            fn(c, x, y, t_since, scale=size, color=color, seed=seed, count=count, life=life, intensity=intensity,
               ring=ring)
            return
        except TypeError:
            pass
    _local_burst(c, x, y, t_since, count, speed, life, color, seed, size)


def twinkles(c, t, x, y, radius=80, count=8, color='#fff4c0', seed=3, intensity=1.0, size=1.0):
    """A few twinkling 4-point sparkles hovering around a point (laughter / joy)."""
    if intensity <= 0:
        return
    r = rng(seed)
    for i in range(count):
        ph = r.uniform(0, 1)
        sp = r.uniform(0.6, 1.2)
        life = (t * sp + ph) % 1.0
        ang = r.uniform(0, TAU)
        d = radius * (0.5 + 0.5 * r.uniform(0, 1))
        px = x + math.cos(ang) * d
        py = y + math.sin(ang) * d * 0.8 - life * 30
        a = intensity * math.sin(life * math.pi) ** 2
        sparkle_shape(c, px, py, (6 + 5 * r.uniform(0, 1)) * size, color, a, rot=life * 1.2)


# ----------------------------------------------------------------------------
# lighthouse beam in fake 3D
# ----------------------------------------------------------------------------
def beam_screen_angle(phi, persp=0.3, tilt=0.0):
    """Horizontal-plane beam direction phi (0 = screen right, +pi/2 = towards camera) ->
    2D drawing angle and projected length factor."""
    dx, dz = math.cos(phi), math.sin(phi)
    ang = math.atan2(dz * persp, dx) + tilt * abs(dx)
    k = math.hypot(dx, dz * persp)
    return ang, k, dz


_BEAM_SWEEP = None


def _beam_has_sweep():
    global _BEAM_SWEEP
    if _BEAM_SWEEP is None:
        try:
            import inspect
            _BEAM_SWEEP = 'mode' in inspect.signature(sea.beam).parameters
        except Exception:
            _BEAM_SWEEP = False
    return _BEAM_SWEEP


def draw_beam(c, lx, ly, phi, length=2600, persp=0.3, tilt=0.0, intensity=1.0, width1=520, horizon_y=None, t=0.0,
              dist=0.9):
    """Rotating lighthouse beam with azimuth `phi` (0 = right, pi/2 = towards camera).
    Uses sea.beam's own fake-3D sweep when available, else a local projection."""
    if _beam_has_sweep():
        sea.beam(c, lx, ly, phi, length=length, width1=width1, intensity=intensity, mode='sweep',
                 horizon_y=horizon_y, t=t, dist=dist)
        return phi
    ang, k, dz = beam_screen_angle(phi, persp, tilt)
    inten = intensity * (0.55 + 0.45 * max(0.0, dz)) * (0.35 + 0.65 * k)
    L = length * (0.25 + 0.75 * k)
    sea.beam(c, lx, ly, ang, length=L, width0=26, width1=width1 * (0.6 + 0.4 * k), intensity=inten)
    if dz > 0.5:
        soft_glow(c, lx, ly, 260, '#ffe7a0', intensity * ((dz - 0.5) / 0.5) ** 3 * 0.9)
    return ang


def draw_beam_screen(c, lx, ly, ang, length=2400, width1=600, intensity=1.0, t=0.0):
    """Beam along a fixed 2D screen direction (story-aimed beam)."""
    if _beam_has_sweep():
        sea.beam(c, lx, ly, ang, length=length, width1=width1, intensity=intensity, mode='screen', t=t)
    else:
        sea.beam(c, lx, ly, ang, length=length, width0=26, width1=width1, intensity=intensity)


# ----------------------------------------------------------------------------
# water helpers
# ----------------------------------------------------------------------------
def in_water(c, draw_fn, wl, x0=-4000, x1=6000, sub_alpha=0.28, tint=SEA_MID, top=-6000, bottom=8000):
    """Draw something partly submerged: full above the waterline `wl` (world y),
    faint and blue-tinted below it."""
    with saved(c):
        c.clipRect(skia.Rect.MakeLTRB(x0, top, x1, wl))
        draw_fn(c)
    if sub_alpha > 0:
        with saved(c):
            c.clipRect(skia.Rect.MakeLTRB(x0, wl, x1, bottom))
            p = skia.Paint(AntiAlias=True)
            p.setAlphaf(clamp(sub_alpha))
            c.saveLayer(None, p)
            draw_fn(c)
            tp = fill(tint, 0.55)
            tp.setBlendMode(skia.BlendMode.kSrcATop)
            c.drawRect(skia.Rect.MakeLTRB(x0, wl, x1, bottom), tp)
            c.restore()


def waterline_ring(c, x, y, w, t, a=0.6, seed=0):
    """Bright foam/specular ring where an object pierces the water surface."""
    if a <= 0:
        return
    wob = 1 + 0.06 * math.sin(t * 3.1 + seed)
    rw, rh = w * 0.62 * wob, w * 0.1
    c.drawOval(skia.Rect.MakeXYWH(x - rw, y - rh, rw * 2, rh * 2), stroke('#cfe6ff', 3.0, a * 0.55))
    c.drawOval(skia.Rect.MakeXYWH(x - rw * 0.8, y - rh * 0.35, rw * 1.6, rh * 0.9), fill('#bfd8ff', a * 0.18))


def bubbles(c, t, x, y, t_since, count=10, spread=60, rise=180, seed=7, a=0.7, life=1.2):
    if t_since < 0:
        return
    r = rng(seed)
    for i in range(count):
        d = r.uniform(0, life * 0.8)
        tt = t_since - d
        if not (0 < tt < 0.6):
            continue
        px = x + r.uniform(-spread, spread) + math.sin(tt * 12 + i) * 5
        py = y - tt / 0.6 * rise * r.uniform(0.3, 1.0)
        rad = r.uniform(3, 8)
        c.drawCircle(px, py, rad, stroke('#cfe6ff', 1.5, a * (1 - tt / 0.6)))


def reflection(c, axis_y, draw_fn, t, alpha=0.45, amp=6.0, fade=0.85, rgb=(0.55, 0.65, 0.9), band=3.0):
    """Mirror `draw_fn` about the world line y=axis_y into the water below it, with a
    rippling horizontal wobble. Renders the mirrored content once into an offscreen
    image (device space) and redraws it in thin wavy strips (cheap)."""
    M = c.getTotalMatrix()
    size = c.getBaseLayerSize()
    dw, dh = size.width(), size.height()
    sc = M.getScaleY() or 1.0
    ay = M.mapXY(0, axis_y).y()
    ay0 = int(max(0, math.floor(ay)))
    hh = dh - ay0
    if hh <= 2 or ay > dh:
        return
    surf = skia.Surface(dw, hh)
    rc = surf.getCanvas()
    rc.clear(skia.ColorTRANSPARENT)
    rc.translate(0, -ay0)
    rc.concat(M)
    rc.translate(0, axis_y)
    rc.scale(1, -1)
    rc.translate(0, -axis_y)
    draw_fn(rc)
    img = surf.makeImageSnapshot()
    dev_scale = M.getScaleX()
    cm = [rgb[0], 0, 0, 0, 0,
          0, rgb[1], 0, 0, 0,
          0, 0, rgb[2], 0, 0,
          0, 0, 0, 1, 0]
    with saved(c):
        c.resetMatrix()
        p = skia.Paint(AntiAlias=False)
        p.setColorFilter(skia.ColorFilters.Matrix(cm))
        step = max(1, int(round(band * dev_scale)))
        y = 0
        samp = skia.SamplingOptions(skia.FilterMode.kLinear)
        while y < hh:
            sh = min(step, hh - y)
            u = y / max(1.0, hh)
            wy = (y / dev_scale) / sc  # world-ish distance below axis
            dx = amp * dev_scale * (0.35 + 1.2 * u) * (math.sin(wy * 0.21 + t * 2.3) * 0.7 +
                                                     math.sin(wy * 0.057 - t * 1.3) * 0.3)
            p.setAlphaf(clamp(alpha * (1 - fade * u)))
            c.drawImageRect(img, skia.Rect.MakeXYWH(0, y, dw, sh), skia.Rect.MakeXYWH(dx, ay0 + y, dw, sh),
                            samp, p)
            y += sh


# ----------------------------------------------------------------------------
# scene props
# ----------------------------------------------------------------------------
def seaweed(c, top, up, s, t, drip=1.0, sway=0.0, side=-1.0):
    """A limp strand of seaweed draped over the robot's head.
    top: head-top world point; up: head up-vector; s: robot scale; side: -1 drapes over
    screen-left side of the head, +1 right."""
    ux, uy = up
    rx, ry = -uy, ux  # head 'right' vector
    if side < 0:
        rx, ry = -rx, -ry

    def P(a, b):  # a along right, b along up (head units, scale s)
        return (top[0] + (rx * a + ux * b) * s, top[1] + (ry * a + uy * b) * s)

    w1 = math.sin(t * 2.3) * 3 + sway * 10
    w2 = math.sin(t * 2.3 + 0.8) * 6 + sway * 18
    # main strand: from slightly past centre over the top, down the side of the head
    cl = [(-22, 2), (8, 7), (40, 5), (62, -8), (74 + w1 * 0.3, -34), (78 + w1, -62), (76 + w2, -92), (70 + w2 * 1.2, -118)]
    wid = [7, 9, 10, 10, 9, 8, 6, 3]
    left, right = [], []
    for i, (a, b) in enumerate(cl):
        a0, b0 = cl[max(0, i - 1)]
        a1, b1 = cl[min(len(cl) - 1, i + 1)]
        ta, tb = a1 - a0, b1 - b0
        d = math.hypot(ta, tb) or 1.0
        na, nb = -tb / d, ta / d
        hw = wid[i] * (1 + 0.15 * math.sin(i * 1.9))
        left.append(P(a + na * hw, b + nb * hw))
        right.append(P(a - na * hw, b - nb * hw))
    path = smooth_path(left + right[::-1], close=True, tension=0.5)
    c.drawPath(path, fill('#1f4a35'))
    # lighter inner vein / wet sheen
    vein = smooth_path([P(a, b + 1.5) for a, b in cl[1:-1]], close=False)
    c.drawPath(vein, stroke('#4f9a66', 2.6 * s, 0.8))
    # a second, shorter frond flopping over the other way
    fr = [(-10, 4), (-34, 8), (-52, -2), (-58 - w1 * 0.4, -22)]
    fl, frr = [], []
    for i, (a, b) in enumerate(fr):
        hw = [6, 7, 5, 2][i]
        fl.append(P(a, b + hw))
        frr.append(P(a, b - hw))
    c.drawPath(smooth_path(fl + frr[::-1], close=True, tension=0.5), fill('#2a5e40'))
    c.drawPath(smooth_path([P(a, b + 1) for a, b in fr], close=False), stroke('#5aa874', 2.0 * s, 0.6))
    # little leaf blobs
    for (a, b, rr) in ((46, 4, 8), (77 + w1, -60, 7), (-40, 6, 6)):
        x, y = P(a, b)
        c.drawCircle(x, y, rr * s, fill('#2f6a48'))
        c.drawCircle(x - 2 * s, y - 2 * s, rr * 0.45 * s, fill('#6cbf86', 0.5))
    # drips from the strand tip
    if drip > 0:
        tx, ty = P(70 + w2 * 1.2, -118)
        for k in range(3):
            ph = (t * 1.3 + k / 3.0) % 1.0
            dy = ph * ph * 90 * s
            c.drawCircle(tx, ty + dy, 3.2 * s * (1 - ph * 0.5), fill('#cfe6ff', drip * 0.8 * (1 - ph)))


def speed_lines(c, x, y, direction, t, n=6, length=160, spread=140, a=0.5, seed=4):
    r = rng(seed)
    for i in range(n):
        yy = y + r.uniform(-spread / 2, spread / 2)
        off = (t * 900 + r.uniform(0, 400)) % 300
        x0 = x - direction * (40 + off * 0.3)
        c.drawLine(x0, yy, x0 - direction * length * r.uniform(0.5, 1.0), yy,
                   stroke('#cfe0ff', 3.0, a * r.uniform(0.4, 1.0)))


def dust_puffs(c, x, y, t, a=0.5, seed=2, n=4, direction=1):
    r = rng(seed)
    for i in range(n):
        life = (t * 2.2 + i / n) % 1.0
        px = x - direction * (life * 70 + r.uniform(0, 20))
        py = y - life * 25
        c.drawCircle(px, py, 10 + life * 22, fill('#8a93b8', a * (1 - life) * 0.5, blur=6))


# ----------------------------------------------------------------------------
# crate tower
# ----------------------------------------------------------------------------
ITEM_H = 120.0
# kind, scale, dx (bottom offset along previous item's x axis), rotation jitter
TOWER = [
    ('crate', 1.40, 0.0, 0.0),
    ('crate', 1.34, 10.0, -0.03),
    ('barrel', 1.30, -8.0, 0.035),
    ('crate', 1.28, 12.0, -0.045),
    ('barrel', 1.22, -13.0, 0.05),
    ('crate', 1.18, 9.0, -0.055),
    ('crate', 1.02, -6.0, 0.075),
]


def tower_chain(n, bx, by, bend=0.0, lean0=0.0):
    """Stack the first `n` tower items from bottom-centre (bx, by).
    bend: angle (rad) added progressively up the tower (sway). Returns (items, top)
    items: list of dict(kind, s, x, y, rot, h) (bottom-centre + rotation);
    top: (x, y, rot) top-centre of the last item."""
    items = []
    x, y, a_prev = bx, by, lean0
    N = len(TOWER)
    for i in range(min(n, N)):
        kind, s, dx, rj = TOWER[i]
        a = lean0 + rj + bend * (i + 0.5) / N
        if i > 0:
            x += math.cos(a_prev) * dx * s
            y += math.sin(a_prev) * dx * s
        h = ITEM_H * s
        items.append(dict(kind=kind, s=s, x=x, y=y, rot=a, h=h, i=i))
        x, y = x + h * math.sin(a), y - h * math.cos(a)
        a_prev = a
    return items, (x, y, a_prev)


_CRATE_KW = None


def draw_item(c, it, dawn=0.0, wet=0.0):
    global _CRATE_KW
    if _CRATE_KW is None:
        try:
            import inspect
            _CRATE_KW = set(inspect.signature(sea.crate).parameters)
        except Exception:
            _CRATE_KW = set()
    kw = {}
    if 'seed' in _CRATE_KW:
        kw['seed'] = it.get('seed', it.get('i', 0) * 3 + 1)
    if 'dawn' in _CRATE_KW:
        kw['dawn'] = dawn
    if 'wet' in _CRATE_KW and wet:
        kw['wet'] = wet
    sea.crate(c, it['x'], it['y'], it['s'], rot=it['rot'], kind=it['kind'], **kw)


def rotate_about(px, py, qx, qy, ang):
    cs, sn = math.cos(ang), math.sin(ang)
    dx, dy = qx - px, qy - py
    return px + dx * cs - dy * sn, py + dx * sn + dy * cs


def item_center(it):
    hx, hy = it['h'] / 2 * math.sin(it['rot']), -it['h'] / 2 * math.cos(it['rot'])
    return it['x'] + hx, it['y'] + hy
