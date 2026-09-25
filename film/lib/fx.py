"""FX library — owner: FX agent.

Light, particles and magic. Every effect is a PURE FUNCTION of its inputs and
time (no per-frame state), so frames can be rendered in any order / in parallel.
Seeded randomness only (engine.core.nprng / rng).

CONTRACT: keep the signatures below working (add kwargs with defaults and new
functions freely). STUB=True means placeholder art.
"""
from __future__ import annotations

import math

import skia

from engine.core import (ADD, at, col, fill, linear, mix, noise1, nprng, poly, radial, soft_glow, stroke)

STUB = True


def glow(c, x, y, r, color='#ffd86b', intensity=1.0):
    """Soft additive light blob."""
    soft_glow(c, x, y, r, color, intensity)


def sparkles(c, t, x, y, radius=120, count=30, color='#fff4c0', seed=1, intensity=1.0, rise=20.0, size=1.0):
    """Twinkling 4-point sparkles scattered around (x,y), slowly rising."""
    r = nprng(seed)
    for i in range(count):
        ph = r.uniform(0, 1)
        life = (t * 0.6 + ph) % 1.0
        px = x + r.uniform(-radius, radius)
        py = y + r.uniform(-radius, radius) - life * rise
        a = intensity * math.sin(life * math.pi)
        c.drawCircle(px, py, 2.5 * size, fill(color, a, blend=ADD))


def meteor(c, x0, y0, x1, y1, p, color='#fff1c8', tail=500, size=1.0, t=0.0):
    """Shooting star travelling from (x0,y0) to (x1,y1); p = 0..1 progress along the path.
    Bright head, long glowing tail, shedding sparks."""
    if not (0 <= p <= 1):
        return
    hx, hy = x0 + (x1 - x0) * p, y0 + (y1 - y0) * p
    ang = math.atan2(y1 - y0, x1 - x0)
    tx, ty = hx - math.cos(ang) * tail, hy - math.sin(ang) * tail
    lp = linear((hx, hy), (tx, ty), [(0, color, 1), (1, color, 0)])
    lp.setStyle(skia.Paint.kStroke_Style)
    lp.setStrokeWidth(8 * size)
    lp.setStrokeCap(skia.Paint.kRound_Cap)
    lp.setBlendMode(ADD)
    c.drawLine(hx, hy, tx, ty, lp)
    soft_glow(c, hx, hy, 60 * size, color, 1.0)


def splash(c, x, y, t_since, scale=1.0, color='#dff2ff', glow_color='#ffe7a0', seed=2):
    """Water splash + burst of light where something falls into the sea at (x,y).
    t_since: seconds since impact (draw nothing if < 0 or > ~3)."""
    if t_since < 0 or t_since > 3:
        return
    r = nprng(seed)
    for i in range(40):
        a = -math.pi / 2 + r.uniform(-1.1, 1.1)
        v = r.uniform(200, 600) * scale
        px = x + math.cos(a) * v * t_since
        py = y + math.sin(a) * v * t_since + 600 * t_since * t_since
        if py < y + 10:
            c.drawCircle(px, py, 4 * scale, fill(color, 1 - t_since / 3))
    soft_glow(c, x, y, 300 * scale, glow_color, max(0, 1 - t_since))


def ripples(c, x, y, t_since, color='#bfe0ff', intensity=1.0, persp=0.25, count=4, speed=120.0, life=4.0):
    """Expanding elliptical ripples on the water surface centred at (x,y).
    persp: vertical squash of the ellipses (0.25 = seen at a low angle)."""
    if t_since < 0:
        return
    for k in range(count):
        tt = t_since - k * 0.45
        if 0 < tt < life:
            rr = tt * speed
            a = intensity * (1 - tt / life)
            c.drawOval(skia.Rect.MakeXYWH(x - rr, y - rr * persp, rr * 2, rr * 2 * persp), stroke(color, 3, a))


def bioluminescence(c, t, x, y, radius=400, intensity=1.0, persp=0.25, seed=4, color='#6ff0ff'):
    """Glowing plankton lighting up on the sea surface around (x, y) (elliptical area)."""
    r = nprng(seed)
    for i in range(120):
        a = r.uniform(0, math.tau)
        d = math.sqrt(r.uniform(0, 1)) * radius
        px, py = x + math.cos(a) * d, y + math.sin(a) * d * persp
        tw = 0.5 + 0.5 * math.sin(t * 3 + i)
        c.drawCircle(px, py, 2.2, fill(color, intensity * tw * (1 - d / radius), blend=ADD))


def light_pillar(c, t, x, y_base, y_top=-3000, width=160, intensity=1.0, color='#fff0b8', core='#ffffff', seed=6):
    """THE climax beam: a colossal vertical pillar of light shooting from (x, y_base) up to
    y_top, with a white-hot core, rippling edges, spiralling motes and rising rings."""
    if intensity <= 0:
        return
    p = linear((x - width, 0), (x + width, 0), [(0, color, 0), (0.5, color, 0.8 * intensity), (1, color, 0)])
    p.setBlendMode(ADD)
    c.drawRect(skia.Rect.MakeLTRB(x - width, y_top, x + width, y_base), p)
    q = linear((x - width * 0.2, 0), (x + width * 0.2, 0), [(0, core, 0), (0.5, core, intensity), (1, core, 0)])
    q.setBlendMode(ADD)
    c.drawRect(skia.Rect.MakeLTRB(x - width * 0.2, y_top, x + width * 0.2, y_base), q)


def stardust_stream(c, t, points, density=1.0, color='#cfe8ff', width=40, speed=0.25, seed=8, intensity=1.0, size=1.0):
    """Particles flowing along a polyline `points` [(x,y),...] from first to last point
    (e.g. stardust pouring from the sky into the robot's chest)."""
    if intensity <= 0 or len(points) < 2:
        return
    r = nprng(seed)
    segs = list(zip(points, points[1:]))
    n = int(80 * density)
    for i in range(n):
        u = (r.uniform(0, 1) + t * speed) % 1.0
        k = min(len(segs) - 1, int(u * len(segs)))
        lu = u * len(segs) - k
        (ax, ay), (bx, by) = segs[k]
        px = ax + (bx - ax) * lu + r.normal(0, width / 3)
        py = ay + (by - ay) * lu + r.normal(0, width / 3)
        c.drawCircle(px, py, 2.5 * size, fill(color, intensity, blend=ADD))


def shockwave(c, x, y, t_since, color='#fff4d0', max_r=900, life=1.2, persp=1.0):
    """Expanding ring of light (ignition moments)."""
    if not (0 <= t_since <= life):
        return
    rr = max_r * (t_since / life) ** 0.6
    a = 1 - t_since / life
    c.drawOval(skia.Rect.MakeXYWH(x - rr, y - rr * persp, 2 * rr, 2 * rr * persp), stroke(color, 10 * a + 1, a))


def lens_flare(c, x, y, intensity=1.0, color='#ffe7b0', cx=960, cy=540):
    """Anamorphic streak + ghosts for a bright light at (x,y) (screen coords)."""
    if intensity <= 0:
        return
    p = linear((x - 700, y), (x + 700, y), [(0, color, 0), (0.5, color, 0.5 * intensity), (1, color, 0)])
    p.setBlendMode(ADD)
    c.drawRect(skia.Rect.MakeXYWH(x - 700, y - 3, 1400, 6), p)


def god_rays(c, t, x, y, count=12, length=1400, spread=math.pi * 2, angle=0.0, color='#fff0c0', intensity=1.0, seed=9):
    """Radiating soft rays from (x,y) (divine / reboot moments)."""
    if intensity <= 0:
        return
    r = nprng(seed)
    for i in range(count):
        a = angle - spread / 2 + spread * (i + 0.5) / count + 0.05 * math.sin(t + i)
        w = r.uniform(0.02, 0.07)
        pts = [(x, y), (x + math.cos(a - w) * length, y + math.sin(a - w) * length),
               (x + math.cos(a + w) * length, y + math.sin(a + w) * length)]
        p = radial((x, y), length, [(0, color, 0.35 * intensity), (1, color, 0)])
        p.setBlendMode(ADD)
        c.drawPath(poly(pts), p)


def dust_motes(c, t, rect=(0, 0, 1920, 1080), count=60, color='#ffe9b0', intensity=0.5, seed=10):
    """Slow floating dust in a light beam / interior."""
    r = nprng(seed)
    x0, y0, w, h = rect
    for i in range(count):
        px = x0 + (r.uniform(0, w) + t * r.uniform(-6, 6)) % w
        py = y0 + (r.uniform(0, h) + t * r.uniform(-4, 2)) % h
        c.drawCircle(px, py, r.uniform(0.8, 2.2), fill(color, intensity * (0.5 + 0.5 * math.sin(t + i)), blend=ADD))


def heart_core(c, t, x, y, r=18, intensity=1.0, kind='flame'):
    """The robot's heart core as a standalone glowing object (when taken out of the chest
    and carried to the lamp). kind: 'flame' (old amber core) or 'star' (new heart)."""
    soft_glow(c, x, y, r * 5, '#ffb347', intensity)
    c.drawCircle(x, y, r, fill('#fff0c8', intensity))


def embers(c, t, x, y, count=20, color='#ffb060', seed=11, intensity=1.0, spread=60, rise=180):
    """Rising warm embers / light specks (heart removal, lamp ignition)."""
    r = nprng(seed)
    for i in range(count):
        life = (t * 0.7 + r.uniform(0, 1)) % 1.0
        px = x + r.uniform(-spread, spread) + math.sin(t * 2 + i) * 10
        py = y - life * rise
        c.drawCircle(px, py, 2.5, fill(color, intensity * (1 - life), blend=ADD))
