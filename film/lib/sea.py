"""SEA & SET library — owner: ENV-SEA agent.

Ocean, island, lighthouse (exterior + interior), boat and small props.
Draw in world units on an already-transformed canvas.

CONTRACT: keep every signature below working (you may add keyword args with
defaults and new functions). STUB=True means placeholder art.
"""
from __future__ import annotations

import math

import skia

from engine.core import (ADD, H, W, at, col, fill, linear, mix, noise1, poly, radial, rrect, soft_glow, stroke)

STUB = True

SEA_DEEP = '#081631'
SEA_MID = '#123060'
SEA_HI = '#5c8fd6'
FOAM = '#bfd8ff'


def ocean(c, t, horizon_y=700, rect_x=(-2000, 3920), bottom=2200, lights=(), dawn=0.0, sunrise=0.0, calm=1.0):
    """Animated sea from horizon_y down to `bottom`.

    lights: iterable of (x, y, colour, intensity) light sources ABOVE the sea;
            draw shimmering vertical reflection streaks under each (x) on the water.
    dawn/sunrise: tint the water to match the sky (0..1).
    calm: wave amplitude multiplier.
    """
    x0, x1 = rect_x
    top = mix(mix(SEA_MID, '#6a5c8e', dawn), '#e0a86a', sunrise)
    p = linear((0, horizon_y), (0, horizon_y + 700), [(0, top, 1), (1, mix(SEA_DEEP, '#3a4a7a', sunrise), 1)])
    c.drawRect(skia.Rect.MakeLTRB(x0, horizon_y, x1, bottom), p)
    # wave lines
    for k in range(18):
        yy = horizon_y + (k + 1) ** 1.6 * 6
        a = 0.18
        pts = []
        for i in range(60):
            xx = x0 + (x1 - x0) * i / 59
            pts.append((xx, yy + math.sin(xx * 0.01 + t * 1.3 + k) * (1 + k * 0.4) * calm))
        path = skia.Path()
        path.moveTo(*pts[0])
        for q in pts[1:]:
            path.lineTo(*q)
        c.drawPath(path, stroke(SEA_HI, 1 + k * 0.15, a))
    for (lx, ly, lc, li) in lights:
        for k in range(10):
            yy = horizon_y + 10 + k * 26
            wv = 30 + k * 10
            c.drawRect(skia.Rect.MakeXYWH(lx - wv / 2 + math.sin(t * 2 + k) * 8, yy, wv, 5), fill(lc, 0.35 * li * (1 - k / 10)))


def island(c, x, y, s=1.0, dawn=0.0):
    """Rocky island; (x, y) = top-centre of the rock where the lighthouse stands. s=1 → ~900 wide."""
    with at(c, x, y, sx=s):
        c.drawPath(poly([(-460, 120), (-330, 20), (-160, -6), (0, -12), (170, -4), (340, 30), (470, 130)]), fill('#1d2440'))
        c.drawPath(poly([(-300, 40), (-150, 8), (40, 0), (220, 20), (300, 60)]), fill('#2a3358'))


LH_HEIGHT = 520  # lighthouse height at s=1 from base to lamp top


def lighthouse(c, x, y, s=1.0, lamp=1.0, dawn=0.0, lit_windows=1.0):
    """Lighthouse exterior. (x, y) = base centre on the rock. s=1 → ~520 tall.
    lamp: 0..1 brightness of the lamp room.
    Returns world (lx, ly) of the lamp centre (for beams/glows)."""
    with at(c, x, y, sx=s):
        c.drawPath(poly([(-70, 0), (70, 0), (46, -400), (-46, -400)]), fill('#d9d4c6'))
        for k in range(3):
            y0 = -60 - k * 120
            c.drawPath(poly([(-68 + k * 8, y0), (68 - k * 8, y0), (64 - k * 8, y0 - 40), (-64 + k * 8, y0 - 40)]), fill('#a8403a'))
        c.drawRect(skia.Rect.MakeLTRB(-62, -410, 62, -396), fill('#2a2f45'))
        c.drawRect(skia.Rect.MakeLTRB(-40, -480, 40, -410), fill('#ffd36b', 0.3 + 0.7 * lamp))
        c.drawPath(poly([(-50, -480), (50, -480), (0, -520)]), fill('#2a2f45'))
    lx, ly = x, y - 445 * s
    if lamp > 0:
        soft_glow(c, lx, ly, 160 * s, '#ffd36b', 0.8 * lamp)
    return lx, ly


def beam(c, x, y, angle, length=2600, width0=30, width1=520, intensity=1.0, color='#ffe7a0'):
    """Sweeping lighthouse beam cone from (x,y). angle in radians (0 = pointing right,
    -pi/2 = straight up). Use additive light. Screen-space fake 3D is fine."""
    if intensity <= 0:
        return
    with at(c, x, y, rot=angle):
        p = linear((0, 0), (length, 0), [(0, color, 0.55 * intensity), (1, color, 0.0)])
        p.setBlendMode(ADD)
        c.drawPath(poly([(0, -width0 / 2), (length, -width1 / 2), (length, width1 / 2), (0, width0 / 2)]), p)


def lamp_room(c, t, lamp=1.0, lens_rot=0.0, dawn=0.0, view_offset=0.0, pointing_up=0.0):
    """Full-screen INTERIOR of the lamp room at the top of the lighthouse (1920x1080 frame):
    glass panes with the night sea/sky outside, brass frame, central big Fresnel lens on a
    rotating base, floor. lamp: lens light 0..1. lens_rot: rotation phase of the lens.
    pointing_up: 0..1 the lens mechanism tilted to aim straight up (climax).
    Returns (lens_x, lens_y, lens_r) so scenes can place the heart core / glow."""
    c.drawRect(skia.Rect.MakeWH(W, H), linear((0, 0), (0, H), [(0, '#0d1433', 1), (1, '#1a2450', 1)]))
    for i in range(5):
        c.drawRect(skia.Rect.MakeXYWH(100 + i * 380, 120, 12, 640), fill('#8a6a3a'))
    c.drawRect(skia.Rect.MakeXYWH(0, 760, W, 320), fill('#3a2e2a'))
    lx, ly, lr = 960, 470, 170
    c.drawCircle(lx, ly, lr, fill('#b8c8e8', 0.4))
    if lamp > 0:
        soft_glow(c, lx, ly, lr * 3, '#ffd36b', lamp)
    return lx, ly, lr


def stairs_interior(c, t, scroll=0.0, light=0.6):
    """Full-screen view inside the lighthouse tower: a spiral staircase curling up around a
    central column, small round windows showing the night. scroll: vertical camera travel in
    world units (increase to climb). Returns a function step_pos(k) -> (x, y, depth_scale)
    giving the screen position of the k-th step (float k allowed) for placing the robot."""
    c.drawRect(skia.Rect.MakeWH(W, H), fill('#1b1a2e'))
    c.drawRect(skia.Rect.MakeXYWH(900, 0, 120, H), fill('#2d2b40'))

    def step_pos(k):
        a = k * 0.5
        x = 960 + math.cos(a) * 420
        y = 900 - k * 40 + scroll
        return x, y, 0.8 + 0.2 * math.sin(a)

    for k in range(-5, 60):
        x, y, d = step_pos(k)
        if -100 < y < H + 100:
            c.drawRect(skia.Rect.MakeXYWH(x - 80 * d, y, 160 * d, 16), fill('#5a4a3e'))
    return step_pos


def gallery_railing(c, x0, x1, y, s=1.0):
    """Foreground railing of the outside balcony (the 'gallery') around the lamp room."""
    c.drawRect(skia.Rect.MakeLTRB(x0, y, x1, y + 10 * s), fill('#2a2f45'))
    n = int((x1 - x0) / (60 * s))
    for i in range(n + 1):
        xx = x0 + i * 60 * s
        c.drawRect(skia.Rect.MakeXYWH(xx, y, 6 * s, 120 * s), fill('#2a2f45'))
    c.drawRect(skia.Rect.MakeLTRB(x0, y + 120 * s, x1, y + 140 * s), fill('#2a2f45'))


def boat(c, x, y, s=1.0, rock=0.0, lantern=1.0, t=0.0):
    """Small wooden rowing boat on the water. (x,y) = waterline centre. rock: tilt radians.
    lantern: brightness of a lantern hanging on a pole at the bow. Returns lantern (x, y)."""
    with at(c, x, y, rot=rock, sx=s):
        c.drawPath(poly([(-160, -30), (160, -30), (120, 20), (-120, 20)]), fill('#6b4a33'))
    lx, ly = x + 130 * s, y - 120 * s
    c.drawLine(x + 130 * s, y - 30 * s, lx, ly, stroke('#3a2a20', 5 * s))
    soft_glow(c, lx, ly, 90 * s, '#ffc86b', lantern)
    return lx, ly


def crate(c, x, y, s=1.0, rot=0.0, kind='crate'):
    """Wooden crate / barrel prop. (x,y) = bottom centre. kind in {'crate','barrel'}. s=1 → ~120 tall."""
    with at(c, x, y, rot=rot, sx=s):
        if kind == 'barrel':
            c.drawRRect(rrect(-45, -120, 90, 120, 20), fill('#7a5236'))
        else:
            c.drawRect(skia.Rect.MakeXYWH(-60, -120, 120, 120), fill('#8a6440'))


def rocks_foreground(c, t, y=980, seed=1, dawn=0.0):
    """Dark foreground rocks along the bottom of the frame (for depth)."""
    c.drawPath(poly([(-100, H + 50), (-100, y), (300, y - 40), (700, y + 20), (1200, y - 10), (1700, y + 30), (2100, y), (2100, H + 50)]), fill('#0c1022'))
