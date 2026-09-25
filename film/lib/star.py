"""小光 (Guang) the fallen star child, and the constellation whale (her mother).
Owner: STAR agent.

CONTRACT: keep StarPose fields, draw_star and draw_whale signatures working
(add fields/kwargs with defaults freely). STUB=True means placeholder art.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import skia

from engine.core import (ADD, at, col, fill, linear, mix, noise1, poly, radial, smooth_path, soft_glow, stroke)

STUB = True
RADIUS = 55  # outer point radius at scale=1


@dataclass
class StarPose:
    x: float = 960.0
    y: float = 540.0
    scale: float = 1.0          # 1.0 → ~110 units across
    rot: float = 0.0            # body rotation radians
    glow: float = 1.0           # 0 = almost extinguished (grey-blue, no halo) .. 1 healthy .. 2 super bright
    flicker: float = 0.0        # 0..1 amount of unstable flicker (when dying)
    squash: float = 0.0         # -1..1 squash/stretch
    # "arms" = the two upper side points; wave/gesture angles in radians (0 = rest)
    arm_l: float = 0.0
    arm_r: float = 0.0
    # face
    eyes: str = 'open'          # 'open','happy','sad','closed','teary','surprised','sleepy'
    blink: float = 0.0
    look: tuple = (0.0, 0.0)
    mouth: float = 0.0          # lip-sync 0..1 (f.mouth('guang'))
    smile: float = 0.4          # -1..1
    blush: float = 0.6
    tears: float = 0.0          # 0..1 visible tears
    trail: float = 0.0          # 0..1 sparkle trail strength (when moving/flying)
    vx: float = 0.0             # velocity hint (units/s) for trail direction & stretch
    vy: float = 0.0
    wet: float = 0.0            # 0..1 just out of the water: drips


def draw_star(c, pose: StarPose, t: float = 0.0):
    """Draw Guang centred at (x, y). Returns dict(center=(x,y), top=(x,y))."""
    s = pose.scale
    g = max(0.0, pose.glow)
    body = mix('#8d97b8', '#ffe58a', min(1.0, g))
    if g > 0.05:
        soft_glow(c, pose.x, pose.y, RADIUS * s * (2.5 + g), '#ffd86b', 0.6 * min(1.5, g))
    with at(c, pose.x, pose.y, rot=pose.rot, sx=s):
        pts = []
        for i in range(10):
            a = -math.pi / 2 + i * math.pi / 5
            r = RADIUS if i % 2 == 0 else RADIUS * 0.5
            pts.append((math.cos(a) * r, math.sin(a) * r))
        c.drawPath(smooth_path(pts, close=True, tension=0.2), fill(body))
        e = 1 - pose.blink
        c.drawOval(skia.Rect.MakeXYWH(-16, -10, 9, 13 * e + 1), fill('#2a2140'))
        c.drawOval(skia.Rect.MakeXYWH(7, -10, 9, 13 * e + 1), fill('#2a2140'))
        c.drawOval(skia.Rect.MakeXYWH(-5, 8, 10, 3 + 8 * pose.mouth), fill('#6a3a4a'))
    return dict(center=(pose.x, pose.y), top=(pose.x, pose.y - RADIUS * s))


def draw_whale(c, t, x, y, scale=1.0, alpha=1.0, facing=1.0, swim=0.0, mouth=0.0, eye=1.0, rot=0.0):
    """The constellation whale (Guang's mother) made of stars & light lines, huge and gentle.
    (x, y) = body centre. scale=1 → ~1400 units long. swim: phase (tail/fin undulation driver,
    e.g. t*0.5). mouth: 0..1 (for her one line of dialogue). Returns dict(eye=(x,y),
    mouth=(x,y), tail=(x,y), belly=(x,y)) in world coords."""
    if alpha <= 0:
        return dict(eye=(x, y), mouth=(x, y), tail=(x, y), belly=(x, y))
    with at(c, x, y, rot=rot, sx=scale * facing, sy=scale):
        p = fill('#3d7bd9', 0.35 * alpha)
        c.drawOval(skia.Rect.MakeXYWH(-700, -180, 1400, 360), p)
        for i in range(24):
            a = i / 24 * math.tau
            c.drawCircle(math.cos(a) * 690, math.sin(a) * 175, 5, fill('#ffffff', alpha))
    return dict(eye=(x + 450 * scale * facing, y - 40 * scale), mouth=(x + 650 * scale * facing, y + 40 * scale),
                tail=(x - 700 * scale * facing, y), belly=(x, y + 170 * scale))


def draw_star_family(c, t, cx, cy, spread=600, alpha=1.0, n=7, seed=5):
    """Other little star children twinkling/waving high in the sky (welcome party)."""
    for i in range(n):
        a = i / n * math.tau
        draw_star(c, StarPose(x=cx + math.cos(a) * spread, y=cy + math.sin(a) * spread * 0.4, scale=0.35, glow=alpha), t)
