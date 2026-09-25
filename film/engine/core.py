"""Core helpers shared by every scene and library.

Everything is drawn in a fixed *design space* of 1920x1080 units. The renderer
scales the canvas for previews, so never hard-code pixel sizes other than in
design units.
"""
from __future__ import annotations

import math
import random
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import skia

W, H = 1920, 1080
FPS = 24
TAU = math.tau


# ----------------------------------------------------------------------------
# math / easing
# ----------------------------------------------------------------------------
def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    return a + (b - a) * t


def invlerp(a, b, x):
    return 0.0 if a == b else clamp((x - a) / (b - a))


def remap(x, a0, a1, b0, b1, clip=True):
    t = (x - a0) / (a1 - a0) if a1 != a0 else 0.0
    if clip:
        t = clamp(t)
    return b0 + (b1 - b0) * t


def smoothstep(a, b, x):
    t = invlerp(a, b, x)
    return t * t * (3 - 2 * t)


def smootherstep(a, b, x):
    t = invlerp(a, b, x)
    return t * t * t * (t * (t * 6 - 15) + 10)


def ease_in(t):
    t = clamp(t)
    return t * t * t


def ease_out(t):
    t = clamp(t)
    return 1 - (1 - t) ** 3


def ease_in_out(t):
    t = clamp(t)
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def ease_out_back(t, s=1.70158):
    t = clamp(t) - 1
    return t * t * ((s + 1) * t + s) + 1


def ease_out_elastic(t):
    t = clamp(t)
    if t in (0, 1):
        return t
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * (TAU / 3)) + 1


def pulse(t, a, b, fade=0.3):
    """1 inside [a,b], smooth 0 outside, with `fade` seconds of ramp."""
    return smoothstep(a - fade, a, t) * (1 - smoothstep(b, b + fade, t))


def spring(t, freq=2.0, damp=4.0):
    """Damped overshoot 0->1 as t goes 0->inf (seconds)."""
    if t <= 0:
        return 0.0
    return 1 - math.exp(-damp * t) * math.cos(TAU * freq * t)


def keyframes(t, keys, ease=ease_in_out):
    """Interpolate a list of (time, value) pairs; values may be numbers or tuples."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:]):
        if t <= t1:
            u = ease((t - t0) / (t1 - t0)) if t1 > t0 else 1.0
            if isinstance(v0, (tuple, list)):
                return tuple(lerp(a, b, u) for a, b in zip(v0, v1))
            return lerp(v0, v1, u)
    return keys[-1][1]


# ----------------------------------------------------------------------------
# deterministic noise (pure functions of inputs -> frames can render in any order)
# ----------------------------------------------------------------------------
def _hash(i, seed=0):
    x = (int(i) * 374761393 + int(seed) * 668265263) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((x ^ (x >> 16)) & 0xFFFFFF) / float(0xFFFFFF)


def noise1(x, seed=0):
    """Smooth value noise in [-1, 1]."""
    i = math.floor(x)
    f = x - i
    u = f * f * (3 - 2 * f)
    return lerp(_hash(i, seed), _hash(i + 1, seed), u) * 2 - 1


def fbm1(x, seed=0, octaves=4):
    v, a, fr = 0.0, 0.5, 1.0
    for o in range(octaves):
        v += a * noise1(x * fr, seed + o * 17)
        a *= 0.5
        fr *= 2.0
    return v


def rng(seed):
    """Seeded python RNG. Use at module level / per object, never global random."""
    return random.Random(seed)


def nprng(seed):
    return np.random.default_rng(seed)


# ----------------------------------------------------------------------------
# colour
# ----------------------------------------------------------------------------
def hexrgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def col(c, a=1.0):
    """Colour from '#rrggbb' or (r,g,b) floats 0..1 -> skia Color4f-compatible int."""
    if isinstance(c, str):
        c = hexrgb(c)
    r, g, b = (clamp(v) for v in c[:3])
    return skia.Color4f(r, g, b, clamp(a)).toColor()


def mix(c1, c2, t):
    """Mix two colours (hex or tuples) -> (r,g,b) tuple."""
    a = hexrgb(c1) if isinstance(c1, str) else c1
    b = hexrgb(c2) if isinstance(c2, str) else c2
    return tuple(lerp(x, y, clamp(t)) for x, y in zip(a[:3], b[:3]))


def scale_rgb(c, k):
    a = hexrgb(c) if isinstance(c, str) else c
    return tuple(clamp(v * k) for v in a[:3])


# ----------------------------------------------------------------------------
# paints
# ----------------------------------------------------------------------------
def fill(c, a=1.0, blur=0.0, blend=None):
    p = skia.Paint(AntiAlias=True, Color=col(c, a))
    if blur > 0:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    if blend is not None:
        p.setBlendMode(blend)
    return p


def stroke(c, w=2.0, a=1.0, blur=0.0, cap=skia.Paint.kRound_Cap, join=skia.Paint.kRound_Join):
    p = skia.Paint(AntiAlias=True, Color=col(c, a), Style=skia.Paint.kStroke_Style, StrokeWidth=w)
    p.setStrokeCap(cap)
    p.setStrokeJoin(join)
    if blur > 0:
        p.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    return p


ADD = skia.BlendMode.kPlus
SCREEN = skia.BlendMode.kScreen


def linear(p0, p1, stops, paint=None):
    """stops: list of (pos, colour, alpha)."""
    paint = paint or skia.Paint(AntiAlias=True)
    paint.setShader(skia.GradientShader.MakeLinear(
        [skia.Point(*p0), skia.Point(*p1)],
        [col(c, a) for _, c, a in stops], [s for s, _, _ in stops]))
    return paint


def radial(center, r, stops, paint=None):
    paint = paint or skia.Paint(AntiAlias=True)
    paint.setShader(skia.GradientShader.MakeRadial(
        skia.Point(*center), max(r, 0.01),
        [col(c, a) for _, c, a in stops], [s for s, _, _ in stops]))
    return paint


def soft_glow(c, x, y, r, color, a=1.0, add=True):
    """Radial glow disc that falls off to 0 (cheap, no blur)."""
    p = radial((x, y), r, [(0.0, color, a), (0.25, color, a * 0.45), (0.6, color, a * 0.12), (1.0, color, 0.0)])
    if add:
        p.setBlendMode(ADD)
    c.drawCircle(x, y, r, p)


# ----------------------------------------------------------------------------
# paths
# ----------------------------------------------------------------------------
def poly(points, close=True):
    path = skia.Path()
    path.moveTo(*points[0])
    for pt in points[1:]:
        path.lineTo(*pt)
    if close:
        path.close()
    return path


def smooth_path(points, close=False, tension=0.5):
    """Catmull-Rom spline through points -> skia.Path of cubic beziers."""
    pts = list(points)
    n = len(pts)
    path = skia.Path()
    if n < 2:
        return path
    path.moveTo(*pts[0])
    rng_ = range(n) if close else range(n - 1)
    for i in rng_:
        p0 = pts[(i - 1) % n] if (close or i > 0) else pts[i]
        p1 = pts[i]
        p2 = pts[(i + 1) % n]
        p3 = pts[(i + 2) % n] if (close or i + 2 < n) else p2
        k = tension / 3.0 * 2
        c1 = (p1[0] + (p2[0] - p0[0]) * k / 2, p1[1] + (p2[1] - p0[1]) * k / 2)
        c2 = (p2[0] - (p3[0] - p1[0]) * k / 2, p2[1] - (p3[1] - p1[1]) * k / 2)
        path.cubicTo(*c1, *c2, *p2)
    if close:
        path.close()
    return path


def rrect(x, y, w, h, r):
    return skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y, w, h), r, r)


# ----------------------------------------------------------------------------
# transforms / camera
# ----------------------------------------------------------------------------
@contextmanager
def saved(c):
    c.save()
    try:
        yield c
    finally:
        c.restore()


@contextmanager
def at(c, x=0.0, y=0.0, rot=0.0, sx=1.0, sy=None):
    """Translate to (x,y), rotate `rot` radians, scale."""
    c.save()
    c.translate(x, y)
    if rot:
        c.rotate(math.degrees(rot))
    if sx != 1.0 or (sy is not None and sy != 1.0):
        c.scale(sx, sx if sy is None else sy)
    try:
        yield c
    finally:
        c.restore()


@contextmanager
def layer(c, alpha=1.0, blend=None, bounds=None):
    """Offscreen layer composited with alpha / blend mode."""
    p = skia.Paint(AntiAlias=True)
    p.setAlphaf(clamp(alpha))
    if blend is not None:
        p.setBlendMode(blend)
    c.saveLayer(bounds, p)
    try:
        yield c
    finally:
        c.restore()


@dataclass
class Camera:
    """A 2D camera. (x, y) is the world point shown at screen centre.

    zoom=1 shows exactly 1920x1080 world units. `apply(c, depth)` pushes the
    transform for a parallax layer: depth=1 is the focal plane, depth=0 is
    infinitely far (does not move/zoom), depth>1 is foreground (moves more).
    """
    x: float = W / 2
    y: float = H / 2
    zoom: float = 1.0
    rot: float = 0.0
    shake: float = 0.0  # amplitude in design units
    t: float = 0.0      # time, used for shake noise

    @contextmanager
    def apply(self, c, depth=1.0):
        c.save()
        sx = self.shake * noise1(self.t * 13.0, 11) * depth
        sy = self.shake * noise1(self.t * 13.0, 29) * depth
        c.translate(W / 2 + sx, H / 2 + sy)
        if self.rot:
            c.rotate(math.degrees(self.rot * depth))
        z = 1.0 + (self.zoom - 1.0) * depth if depth != 1.0 else self.zoom
        c.scale(z, z)
        c.translate(-(W / 2 + (self.x - W / 2) * depth), -(H / 2 + (self.y - H / 2) * depth))
        try:
            yield c
        finally:
            c.restore()

    def to_screen(self, x, y, depth=1.0):
        z = 1.0 + (self.zoom - 1.0) * depth if depth != 1.0 else self.zoom
        cx = W / 2 + (self.x - W / 2) * depth
        cy = H / 2 + (self.y - H / 2) * depth
        return (W / 2 + (x - cx) * z, H / 2 + (y - cy) * z)


# ----------------------------------------------------------------------------
# text
# ----------------------------------------------------------------------------
_FONTS = {}
FONT_SANS = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_SANS_BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
FONT_SERIF = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
FONT_SERIF_BOLD = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc'


def font(size, path=FONT_SANS, index=2):
    """index 2 in the Noto CJK .ttc files is Simplified Chinese (SC)."""
    key = (path, index)
    if key not in _FONTS:
        _FONTS[key] = skia.Typeface.MakeFromFile(path, index)
    f = skia.Font(_FONTS[key], size)
    f.setEdging(skia.Font.Edging.kAntiAlias)
    f.setSubpixel(True)
    return f


def text_width(s, f):
    return f.measureText(s)


def draw_text(c, s, x, y, f, paint, align='center'):
    w = f.measureText(s)
    if align == 'center':
        x -= w / 2
    elif align == 'right':
        x -= w
    c.drawString(s, x, y, f, paint)
