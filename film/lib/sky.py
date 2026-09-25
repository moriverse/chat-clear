"""SKY library — owner: ENV-SKY agent.

Night sky, stars, Milky Way, moon, clouds, dawn and sunrise.
All functions draw in world/design units onto a skia canvas `c` that the
caller has already transformed (e.g. inside `cam.apply(c, depth)`).

CONTRACT: keep every signature below working (you may add keyword args with
defaults and new functions). STUB=True means placeholder art.
"""
from __future__ import annotations

import math

import skia

from engine.core import (ADD, H, W, col, fill, linear, mix, nprng, radial, soft_glow)

STUB = True

# Palette (see STYLE.md)
NIGHT_TOP = '#070b1f'
NIGHT_MID = '#16224d'
NIGHT_HORIZON = '#2c3f7a'
DAWN_TOP = '#27306a'
DAWN_MID = '#8a6fb0'
DAWN_HORIZON = '#f5a878'
SUN_HORIZON = '#ffd08a'


def sky(c, rect=(-2000, -3000, 5920, 5160), horizon_y=700, dawn=0.0, sunrise=0.0):
    """Fill `rect` (x, y, w, h in world units) with the sky gradient.

    horizon_y: world y of the sea horizon (gradient is anchored to it).
    dawn: 0 = deep night, 1 = pre-dawn purple/peach glow at the horizon.
    sunrise: 0..1 on top of dawn: warm golden morning, stars gone.
    """
    x, y, w, h = rect
    top = mix(mix(NIGHT_TOP, DAWN_TOP, dawn), '#6fa6d8', sunrise)
    mid = mix(mix(NIGHT_MID, DAWN_MID, dawn), '#f3c8a0', sunrise)
    hor = mix(mix(NIGHT_HORIZON, DAWN_HORIZON, dawn), SUN_HORIZON, sunrise)
    p = linear((0, horizon_y - 900), (0, horizon_y), [(0, top, 1), (0.6, mid, 1), (1, hor, 1)])
    c.drawRect(skia.Rect.MakeXYWH(x, y, w, h), p)


_STARS = None


def stars(c, t, rect=(-2000, -3000, 5920, 5160), density=1.0, bright=1.0, twinkle=1.0, seed=7, horizon_y=None):
    """Starfield covering `rect`. Deterministic for a given seed; twinkles with time t.
    bright: global multiplier (0 = invisible, e.g. at sunrise).
    horizon_y: if given, stars fade out towards the horizon line.
    """
    global _STARS
    if bright <= 0:
        return
    if _STARS is None:
        r = nprng(seed)
        n = 2600
        _STARS = (r.uniform(-2000, 3920, n), r.uniform(-3000, 2160, n), r.uniform(0.6, 2.4, n), r.uniform(0, 6.28, n))
    xs, ys, ss, ph = _STARS
    x0, y0, w, h = rect
    p = fill('#fff6e0')
    for i in range(int(len(xs) * min(1.0, density))):
        x, y = xs[i], ys[i]
        if not (x0 <= x <= x0 + w and y0 <= y <= y0 + h):
            continue
        a = bright * (0.55 + 0.45 * math.sin(t * 2.0 * twinkle + ph[i]))
        if horizon_y is not None:
            a *= max(0.0, min(1.0, (horizon_y - y) / 250))
        p.setAlphaf(max(0.0, min(1.0, a)))
        c.drawCircle(x, y, ss[i], p)


def milky_way(c, t, center=(960, 250), angle=-0.35, length=3000, width=520, alpha=1.0):
    """Soft band of the Milky Way through `center` at `angle` (radians)."""
    if alpha <= 0:
        return
    with_ = skia.Paint(AntiAlias=True)
    c.save()
    c.translate(*center)
    c.rotate(math.degrees(angle))
    p = linear((0, -width / 2), (0, width / 2), [(0, '#6c7fd0', 0), (0.5, '#9aa8e8', 0.22 * alpha), (1, '#6c7fd0', 0)])
    c.drawRect(skia.Rect.MakeXYWH(-length / 2, -width / 2, length, width), p)
    c.restore()


def moon(c, x, y, r=60, phase=0.25, glow=1.0):
    """Crescent moon. phase 0 = new, 0.5 = full."""
    soft_glow(c, x, y, r * 4, '#bcd0ff', 0.25 * glow)
    c.drawCircle(x, y, r, fill('#f4f1e2'))
    c.drawCircle(x + r * (1 - 2 * phase) * 1.2, y - r * 0.1, r * 0.95, fill(NIGHT_MID))


def clouds(c, t, y=500, alpha=0.5, color='#34467f', seed=3, speed=6.0, scale=1.0, rect_x=(-2000, 3920)):
    """A drifting band of soft layered clouds centred on world y."""
    r = nprng(seed)
    for i in range(14):
        cx = rect_x[0] + ((r.uniform(0, 6000) + t * speed) % (rect_x[1] - rect_x[0]))
        cy = y + r.uniform(-60, 60) * scale
        c.drawOval(skia.Rect.MakeXYWH(cx, cy, 380 * scale, 70 * scale), fill(color, alpha, blur=18))


def dawn_glow(c, horizon_y=700, amount=0.0, sun_x=960):
    """Warm bloom sitting on the horizon (use with dawn>0)."""
    if amount <= 0:
        return
    soft_glow(c, sun_x, horizon_y, 900, '#ffb07a', 0.5 * amount)


def sun(c, x, y, r=90, amount=1.0):
    """Rising sun disc with glow (y near horizon_y, partially hidden by sea drawn after)."""
    if amount <= 0:
        return
    soft_glow(c, x, y, r * 6, '#ffcf80', 0.6 * amount)
    c.drawCircle(x, y, r, fill('#fff1c4', amount))


def shooting_star_sky_streak(c, x, y, angle, length, alpha):
    """A thin background meteor streak (decoration)."""
    c.save()
    c.translate(x, y)
    c.rotate(math.degrees(angle))
    p = linear((0, 0), (-length, 0), [(0, '#ffffff', alpha), (1, '#9ab8ff', 0)])
    p.setStrokeWidth(2.5)
    p.setStyle(skia.Paint.kStroke_Style)
    c.drawLine(0, 0, -length, 0, p)
    c.restore()
