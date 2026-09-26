"""Shared particle-title machinery for s01 (opening title) and s08 (end title).

    from scenes._title import TITLE_MAIN, draw_title
    draw_title(c, TITLE_MAIN, t, cx, cy, t_in=..., t_out=None, ...)

The Chinese title 一盏灯，一颗星 is rasterised once, its glyph fill sampled into
~2000 points. Each point is a star particle that
  * rests as a faint star somewhere in the sky before `t_in`,
  * flows along a graceful arc into its place in the glyphs (left-to-right sweep,
    per-particle delay, ease in-out, little arrival flare),
  * shimmers on top of the crisp serif title, which is revealed by a soft
    wipe that follows the arrival front,
  * after `t_out` is released by a second wipe, rises and drifts (curl-ish sway)
    to a destination (e.g. inside the Milky Way band) where it stays as a star.

Everything is a pure function of time. Geometry is cached at module level.
"""
from __future__ import annotations

import math

import numpy as np
import skia

from engine.core import (ADD, FONT_SERIF, FONT_SERIF_BOLD, clamp, col, font, nprng, smoothstep)

# ----------------------------------------------------------------------------
PALETTE = ('#fff6e0', '#ffe6b0', '#cfe3ff')     # warm white, soft gold, blue-white
PAL_W = (0.62, 0.22, 0.16)
TEXT_TOP = '#fff4dc'
TEXT_BOT = '#f2cc84'
GLOW = '#ffd68a'


def _smooth(u):
    """Vectorised smootherstep for numpy arrays (already clipped 0..1)."""
    return u * u * u * (u * (u * 6 - 15) + 10)


def _ease_io(u):
    return np.where(u < 0.5, 4 * u ** 3, 1 - (-2 * u + 2) ** 3 / 2)


class GlyphText:
    """A line of text + its glyph-fill point samples, centred on its ink box."""

    def __init__(self, text, size, path=FONT_SERIF_BOLD, spacing=4.2, max_pts=2000, seed=11, tracking=0.0):
        self.text = text
        self.size = size
        self.font = font(size, path)
        self.tracking = tracking
        f = self.font
        m = f.getMetrics()
        asc, desc = -m.fAscent, m.fDescent
        glyphs = f.textToGlyphs(text)
        widths = f.getWidths(glyphs)
        # glyph x positions (with optional tracking)
        xs = [0.0]
        for w_ in widths[:-1]:
            xs.append(xs[-1] + w_ + tracking * size)
        self.glyphs = glyphs
        self.gx = xs
        width = xs[-1] + widths[-1]
        pad = int(size * 0.3)
        sw, sh = int(width + 2 * pad), int(asc + desc + 2 * pad)
        surf = skia.Surface(sw, sh)
        cv = surf.getCanvas()
        cv.clear(skia.Color4f(0, 0, 0, 0).toColor())
        blob = self._blob(pad, pad + asc)
        cv.drawTextBlob(blob, 0, 0, skia.Paint(AntiAlias=True, Color=col('#ffffff')))
        a = surf.makeImageSnapshot().toarray()[..., 3].astype(np.float32) / 255.0
        ys_, xs_ = np.nonzero(a > 0.5)
        x0, x1, y0, y1 = xs_.min(), xs_.max(), ys_.min(), ys_.max()
        cxm, cym = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        self.ink_w, self.ink_h = float(x1 - x0), float(y1 - y0)
        # text origin (for drawing the crisp blob) relative to ink centre
        self.ox = pad - cxm
        self.oy = pad + asc - cym
        # jittered-grid sampling of the fill
        r = nprng(seed)
        gxs = np.arange(x0, x1 + 1, spacing)
        gys = np.arange(y0, y1 + 1, spacing)
        X, Y = np.meshgrid(gxs, gys)
        X = X.ravel() + r.uniform(-0.42, 0.42, X.size) * spacing
        Y = Y.ravel() + r.uniform(-0.42, 0.42, Y.size) * spacing
        xi = np.clip(np.round(X).astype(int), 0, sw - 1)
        yi = np.clip(np.round(Y).astype(int), 0, sh - 1)
        keep = a[yi, xi] > 0.55
        P = np.stack([X[keep] - cxm, Y[keep] - cym], 1)
        if len(P) > max_pts:
            P = P[r.choice(len(P), max_pts, replace=False)]
        self.pts = P.astype(np.float64)
        self.bounds = skia.Rect.MakeLTRB(-self.ink_w / 2 - size * 0.25, -self.ink_h / 2 - size * 0.25,
                                         self.ink_w / 2 + size * 0.25, self.ink_h / 2 + size * 0.25)

    def _blob(self, ox, oy):
        return skia.TextBlob.MakeFromPosTextH(self.text, [ox + x for x in self.gx], oy, self.font)

    def blob(self):
        if not hasattr(self, '_blob_c'):
            self._blob_c = self._blob(self.ox, self.oy)
        return self._blob_c

    def draw(self, c, x, y, paint):
        c.drawTextBlob(self.blob(), x, y, paint)


class ParticleTitle:
    """Precomputed particle system for one title (main line + optional sub line)."""

    def __init__(self, main, sub=None, sub_gap=150, seed=3, spread=(2300, 1250), sub_particles=True):
        self.main = main
        self.sub = sub
        self.sub_gap = sub_gap
        r = nprng(seed)
        pts = [main.pts]
        kinds = [np.zeros(len(main.pts))]
        if sub is not None and sub_particles:
            sp = sub.pts + np.array([0.0, sub_gap])
            pts.append(sp)
            kinds.append(np.ones(len(sp)))
        P = np.concatenate(pts, 0)
        self.kind = np.concatenate(kinds)
        n = len(P)
        self.n = n
        self.tx, self.ty = P[:, 0], P[:, 1]
        half = main.ink_w / 2 + 1e-3
        self.txn = np.clip((self.tx + half) / (2 * half), 0, 1)     # 0..1 across title
        # starting positions: scattered over the whole sky around the title
        sx = r.uniform(-spread[0] / 2, spread[0] / 2, n)
        sy = r.uniform(-spread[1] * 0.55, spread[1] * 0.45, n)
        # bias: a third of them start closer (smaller arcs)
        near = r.uniform(0, 1, n) < 0.18
        sx = np.where(near, self.tx + r.normal(0, 420, n), sx)
        sy = np.where(near, self.ty + r.normal(0, 320, n), sy)
        self.sx, self.sy = sx, sy
        # arc control point: perpendicular offset, mostly one swirl direction
        dx, dy = self.tx - sx, self.ty - sy
        swirl = r.normal(0.42, 0.12, n)
        self.cx = (sx + self.tx) / 2 - dy * swirl
        self.cy = (sy + self.ty) / 2 + dx * swirl
        # timing (seconds relative to t_in)
        self.delay = 0.05 + 0.95 * self.txn + r.uniform(0, 0.4, n) + self.kind * 0.9
        self.dur = r.uniform(1.15, 1.7, n)
        # dissolve timing (relative to t_out)
        self.rel = 1.25 * self.txn + r.uniform(0, 0.3, n) - self.kind * 0.5
        self.rel = np.maximum(self.rel, 0.0)
        self.rdur = r.uniform(2.6, 4.2, n)
        # look
        self.size = np.where(self.kind > 0, r.uniform(0.9, 1.4, n), r.uniform(1.25, 2.25, n))
        self.star_size = r.uniform(0.8, 1.9, n)
        self.star_bright = r.uniform(0.15, 0.7, n) ** 1.5
        self.ph = r.uniform(0, math.tau, n)
        self.tw = r.uniform(1.5, 4.0, n)
        cidx = r.choice(3, n, p=PAL_W)
        self.cidx = cidx
        self.sway = r.uniform(18, 60, n)
        self.sway_f = r.uniform(0.5, 1.2, n)
        self.kick_x = r.normal(0, 150, n)
        self.kick_y = r.normal(0, 90, n)
        # rest-star fade-in time (before t_in) relative to t_in
        self.appear = r.uniform(-1.3, 0.2, n)
        self.dest = None

    # ------------------------------------------------------------------
    def set_source_point(self, x, y, frac=0.35, seed=9, jitter=14.0):
        """Make a fraction of the particles emanate from one point (title-local coords),
        e.g. the answering star: they stream out of it into the letters."""
        r = nprng(seed)
        n = self.n
        m = r.uniform(0, 1, n) < frac
        self.sx = np.where(m, x + r.normal(0, jitter, n), self.sx)
        self.sy = np.where(m, y + r.normal(0, jitter, n), self.sy)
        dx, dy = self.tx - self.sx, self.ty - self.sy
        swirl = r.normal(0.32, 0.18, n)
        self.cx = (self.sx + self.tx) / 2 - dy * swirl
        self.cy = (self.sy + self.ty) / 2 + dx * swirl
        # emitted ones leave the star in a stream (staggered), rest stars wait longer
        self.delay = np.where(m, r.uniform(0.0, 1.3, n), self.delay)
        self.appear = np.where(m, 99.0, self.appear)     # invisible until emitted
        self.emitted = m
        return self

    def set_dest_band(self, cx, cy, angle, width, seed=5, lift=260):
        """Dissolve destination: points inside a band (e.g. the Milky Way) given in
        title-local coordinates (title centre = 0,0). Each particle goes to the band
        point roughly above it."""
        r = nprng(seed)
        n = self.n
        ca, sa = math.cos(angle), math.sin(angle)
        across = r.normal(0, width * 0.24, n)
        if abs(sa) > 0.5:
            # steep band: pick a height above each particle, find the band there
            yd = self.ty - r.uniform(lift * 0.8, lift * 2.4, n)
            along = (yd - cy) / sa
        else:
            along = (self.tx - cx) * ca * 1.6 + r.normal(0, 520, n)
        dx = cx + along * ca - across * sa
        dy = cy + along * sa + across * ca
        self.dest = (dx, dy)
        self.lift = lift
        return self

    def set_dest_rise(self, height=520, spread=500, seed=5):
        r = nprng(seed)
        n = self.n
        self.dest = (self.tx + r.normal(0, spread, n), self.ty - height + r.normal(0, height * 0.35, n))
        self.lift = height * 0.5
        return self


def _draw_points(c, xs, ys, rs, als, cidx, blur=0.0, size_mul=1.0, alpha_mul=1.0, palette=PALETTE, blend=ADD):
    """Bucketed batched point drawing (colour x alpha x size)."""
    als = als * alpha_mul
    m = als > 0.02
    if not np.any(m):
        return
    xs, ys, rs, als, cidx = xs[m], ys[m], rs[m] * size_mul, als[m], cidx[m]
    ai = np.clip(np.ceil(als * 7).astype(int), 1, 7)
    si = np.clip(np.floor(rs / 0.8).astype(int), 0, 6)
    key = cidx * 100 + ai * 10 + si
    order = np.argsort(key, kind='stable')
    key_s = key[order]
    xs_s, ys_s = xs[order], ys[order]
    bounds = np.flatnonzero(np.diff(key_s)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(key_s)]])
    mf = skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur) if blur > 0 else None
    for a0, b0 in zip(starts, ends):
        k = int(key_s[a0])
        ci, al, sz = k // 100, (k // 10) % 10, k % 10
        p = skia.Paint(AntiAlias=True, Color=col(palette[ci], al / 7.0))
        p.setStrokeWidth(max(0.6, (sz + 0.5) * 0.8 * 2))
        p.setStrokeCap(skia.Paint.kRound_Cap)
        if blend is not None:
            p.setBlendMode(blend)
        if mf is not None:
            p.setMaskFilter(mf)
        pts = list(map(skia.Point, xs_s[a0:b0].tolist(), ys_s[a0:b0].tolist()))
        c.drawPoints(skia.Canvas.kPoints_PointMode, pts, p)


def _draw_lines(c, x0, y0, x1, y1, als, widths, cidx, palette=PALETTE):
    m = als > 0.03
    if not np.any(m):
        return
    x0, y0, x1, y1, als, widths, cidx = x0[m], y0[m], x1[m], y1[m], als[m], widths[m], cidx[m]
    ai = np.clip(np.ceil(als * 5).astype(int), 1, 5)
    wi = np.clip(np.floor(widths / 0.8).astype(int), 0, 4)
    key = cidx * 100 + ai * 10 + wi
    for k in np.unique(key):
        sel = key == k
        ci, al, wv = int(k) // 100, (int(k) // 10) % 10, int(k) % 10
        p = skia.Paint(AntiAlias=True, Color=col(palette[ci], al / 5.0 * 0.6))
        p.setStrokeWidth((wv + 0.5) * 0.8 * 1.3)
        p.setStrokeCap(skia.Paint.kRound_Cap)
        p.setBlendMode(ADD)
        pts = []
        for a, b, cc, d in zip(x0[sel].tolist(), y0[sel].tolist(), x1[sel].tolist(), y1[sel].tolist()):
            pts.append(skia.Point(a, b))
            pts.append(skia.Point(cc, d))
        c.drawPoints(skia.Canvas.kLines_PointMode, pts, p)


def _text_layer(c, gt, x, y, alpha, wipe=None, erase=None, glow=0.7, sheen=None):
    """Crisp serif text with warm gradient + soft halo, optionally revealed by a
    left-to-right wipe front (wipe = (front_x, softness)) and erased by another
    (erase = (front_x, softness)); fronts are in title-local x."""
    if alpha <= 0.003:
        return
    b = gt.bounds.makeOffset(x, y)
    b = skia.Rect.MakeLTRB(b.left() - 60, b.top() - 60, b.right() + 60, b.bottom() + 60)
    lp = skia.Paint()
    lp.setAlphaf(clamp(alpha))
    c.saveLayer(b, lp)
    # halo
    if glow > 0:
        hp = skia.Paint(AntiAlias=True, Color=col(GLOW, 0.55 * glow))
        hp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, gt.size * 0.12))
        gt.draw(c, x, y, hp)
        hp2 = skia.Paint(AntiAlias=True, Color=col(GLOW, 0.35 * glow))
        hp2.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, gt.size * 0.35))
        gt.draw(c, x, y, hp2)
    tp = skia.Paint(AntiAlias=True)
    tp.setShader(skia.GradientShader.MakeLinear(
        [skia.Point(x, y - gt.ink_h / 2), skia.Point(x, y + gt.ink_h / 2)],
        [col(TEXT_TOP), col(TEXT_BOT)], [0.0, 1.0]))
    gt.draw(c, x, y, tp)
    # a soft specular sheen gliding across the letters (sheen = 0..1 position)
    if sheen is not None and 0.0 < sheen < 1.0:
        half = gt.ink_w / 2
        sx_ = x - half - 200 + (2 * half + 400) * sheen
        k = math.sin(math.pi * sheen)
        sp = skia.Paint(AntiAlias=True)
        sp.setShader(skia.GradientShader.MakeLinear(
            [skia.Point(sx_ - 70, y + gt.ink_h * 0.4), skia.Point(sx_ + 70, y - gt.ink_h * 0.4)],
            [col('#ffffff', 0.0), col('#ffffff', 1.0 * k), col('#ffffff', 0.0)], [0.0, 0.5, 1.0]))
        sp.setBlendMode(skia.BlendMode.kSrcATop)
        c.drawRect(b, sp)
    # wipes (destination-in masks)
    for spec, reveal in ((wipe, True), (erase, False)):
        if spec is None:
            continue
        fx, soft = spec
        a_left, a_right = (1.0, 0.0) if reveal else (0.0, 1.0)
        mp = skia.Paint()
        # slightly diagonal front for elegance
        p0 = skia.Point(x + fx - soft, y + gt.ink_h * 0.25)
        p1 = skia.Point(x + fx + soft, y - gt.ink_h * 0.25)
        mp.setShader(skia.GradientShader.MakeLinear(
            [p0, p1], [skia.Color4f(0, 0, 0, a_left).toColor(), skia.Color4f(0, 0, 0, a_right).toColor()], [0.0, 1.0]))
        mp.setBlendMode(skia.BlendMode.kDstIn)
        c.drawRect(b, mp)
    c.restore()


def draw_title(c, pt: ParticleTitle, t, x, y, t_in, t_out=None, alpha=1.0, rest_alpha=1.0,
               sub_alpha=None, sub_in=None, text_glow=0.7, shimmer=1.0, start_offset=(0.0, 0.0), sheen=True):
    """Draw the particle title centred at (x, y) (current canvas coords).

    t: current time (s); t_in: assembly cue; t_out: dissolve cue (None = hold).
    rest_alpha: brightness of the waiting particles before they start to move
    (use the starfield fade-in). sub_in: time the sub line (English / 完) fades in
    (default t_in + 2.9). Returns dict(text=0..1 visibility of crisp text).
    """
    n = pt.n
    main = pt.main
    lt = t - t_in
    half = main.ink_w / 2
    # ----------------------------------------------------------- assembly
    u = np.clip((lt - pt.delay) / pt.dur, 0.0, 1.0)
    e = _ease_io(u)
    sx = pt.sx + start_offset[0]
    sy = pt.sy + start_offset[1]
    omt = 1 - e
    px = omt * omt * sx + 2 * omt * e * pt.cx + e * e * pt.tx
    py = omt * omt * sy + 2 * omt * e * pt.cy + e * e * pt.ty
    # velocity (units/s) for streaks
    de = np.where(u < 0.5, 12 * u ** 2, 3 * (-2 * u + 2) ** 2) / pt.dur * ((u > 0) & (u < 1))
    vx = (2 * omt * (pt.cx - sx) + 2 * e * (pt.tx - pt.cx)) * de
    vy = (2 * omt * (pt.cy - sy) + 2 * e * (pt.ty - pt.cy)) * de
    tw = 0.5 + 0.5 * np.sin(t * pt.tw + pt.ph)
    # waiting stars
    wait_a = 0.6 * rest_alpha * pt.star_bright * (0.55 + 0.45 * tw) * np.clip((lt - pt.appear) / 1.2, 0, 1)
    moving_a = 0.45 + 0.4 * np.sin(np.pi * u)             # brighten mid-flight
    arrive_t = lt - (pt.delay + pt.dur)
    flare = np.where(arrive_t > 0, np.exp(-np.maximum(arrive_t, 0) * 4.0), 0.0)
    # crisp text reveal front follows arrivals
    front = -half - 140 + (half * 2 + 280) * clamp((lt - 1.25) / 1.75)
    text_vis = clamp((lt - 1.2) / 1.9)
    settled_a = (0.45 + 0.3 * tw) * (1 - 0.45 * text_vis * shimmer) + flare * 0.6
    a = np.where(u <= 0, wait_a, np.where(u >= 1, settled_a, np.maximum(wait_a, moving_a)))
    rs = np.where(u <= 0, pt.star_size * (0.7 + 0.3 * tw), pt.size * (1 + 0.25 * np.sin(np.pi * u)) + flare * 0.9)
    rs = np.where(u <= 0, rs, np.where(u >= 1, rs, rs))
    # ----------------------------------------------------------- dissolve
    erase = None
    if t_out is not None and pt.dest is not None:
        lo = t - t_out
        v = lo - pt.rel
        u2 = np.clip(v / pt.rdur, 0.0, 1.0)
        rel = v > 0
        if np.any(rel):
            e2 = _ease_io(u2)
            dx, dy = pt.dest
            ax, ay = px, py
            c1x = ax + (dx - ax) * 0.15 + pt.kick_x
            c1y = ay - pt.lift * 0.7 + pt.kick_y
            o2 = 1 - e2
            qx = o2 * o2 * ax + 2 * o2 * e2 * c1x + e2 * e2 * dx
            qy = o2 * o2 * ay + 2 * o2 * e2 * c1y + e2 * e2 * dy
            sway = pt.sway * np.sin(v * pt.sway_f * 2.2 + pt.ph) * np.sin(np.pi * u2)
            qx = qx + sway
            rflare = np.where(rel, np.exp(-np.maximum(v, 0) * 4.0), 0.0) * 0.5
            end_a = pt.star_bright * (0.55 + 0.45 * tw)
            fade_in_flight = np.minimum(1.0, 0.55 + 0.45 * np.exp(-np.maximum(v, 0) * 2.5))
            ra = ((1 - e2) * (0.42 + 0.25 * tw) + e2 * end_a) * fade_in_flight + rflare * 0.5
            rr = (1 - e2) * pt.size + e2 * pt.star_size * (0.7 + 0.3 * tw) + rflare * 1.0
            dq = np.where(u2 < 0.5, 12 * u2 ** 2, 3 * (-2 * u2 + 2) ** 2) / pt.rdur * ((u2 > 0) & (u2 < 1))
            rvx = (2 * o2 * (c1x - ax) + 2 * e2 * (dx - c1x)) * dq
            rvy = (2 * o2 * (c1y - ay) + 2 * e2 * (dy - c1y)) * dq
            px = np.where(rel, qx, px)
            py = np.where(rel, qy, py)
            a = np.where(rel, ra, a)
            rs = np.where(rel, rr, rs)
            vx = np.where(rel, rvx, vx)
            vy = np.where(rel, rvy, vy)
        efront = -half - 140 + (half * 2 + 280) * clamp((lo + 0.05) / 1.35)
        if lo > -0.2:
            erase = (efront, 150)
    a = a * alpha
    X = px + x
    Y = py + y
    # streaks for moving particles
    sp = np.hypot(vx, vy)
    mv = sp > 5
    if np.any(mv):
        L = np.minimum(sp * 0.03, 26.0)
        nx = np.where(mv, vx / np.maximum(sp, 1e-6), 0)
        ny = np.where(mv, vy / np.maximum(sp, 1e-6), 0)
        _draw_lines(c, X[mv], Y[mv], (X - nx * L)[mv], (Y - ny * L)[mv], (a * 0.32)[mv], (rs * 0.7)[mv], pt.cidx[mv])
    # halo pass for bright/flaring ones
    _draw_points(c, X, Y, rs, a * 0.28, pt.cidx, blur=3.0, size_mul=2.6)
    _draw_points(c, X, Y, rs, a, pt.cidx)
    # crisp text
    ta = alpha * text_vis
    if t_out is not None:
        ta *= 1 - clamp((t - t_out - 1.3) / 0.3)
    sheen_p = (lt - 3.25) / 1.3 if sheen else None
    _text_layer(c, main, x, y, ta, wipe=(front, 140) if text_vis < 1 or lt < 3.2 else None, erase=erase,
                glow=text_glow, sheen=sheen_p)
    # sub line (English / 完 · The End): gentle fade + tracking breathe
    if pt.sub is not None:
        s_in = t_in + 2.9 if sub_in is None else sub_in
        sa = smoothstep(s_in, s_in + 1.2, t)
        if t_out is not None:
            sa *= 1 - smoothstep(t_out - 0.1, t_out + 0.7, t)
        if sub_alpha is not None:
            sa *= sub_alpha
        if sa > 0.003:
            _sub_text(c, pt.sub, x, y + pt.sub_gap, sa * alpha, clamp((t - s_in) / 3.0))
    return dict(text=ta)


def _sub_text(c, gt, x, y, a, p):
    lp = skia.Paint()
    lp.setAlphaf(clamp(a))
    b = gt.bounds.makeOffset(x, y)
    c.saveLayer(skia.Rect.MakeLTRB(b.left() - 80, b.top() - 40, b.right() + 80, b.bottom() + 40), lp)
    hp = skia.Paint(AntiAlias=True, Color=col('#cfe3ff', 0.35))
    hp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, gt.size * 0.3))
    # rise gently into place
    dy = (1 - (1 - (1 - p) ** 3)) * 10
    gt.draw(c, x, y + dy, hp)
    gt.draw(c, x, y + dy, skia.Paint(AntiAlias=True, Color=col('#e9eefc', 0.92)))
    c.restore()


# ----------------------------------------------------------------------------
# cached title instances (built lazily: fonts must load in each worker)
_CACHE = {}


def title_main(which='s01'):
    """The shared title system. which='s01' (English sub) or 's08' (完 · The End)."""
    if which not in _CACHE:
        main = GlyphText('一盏灯，一颗星', 132, FONT_SERIF_BOLD, spacing=4.0, max_pts=2100, seed=11, tracking=0.06)
        if which == 's01':
            sub = GlyphText('A Lamp and a Star', 40, FONT_SERIF, spacing=3.0, max_pts=260, seed=12, tracking=0.12)
            pt = ParticleTitle(main, sub, sub_gap=128, seed=3, sub_particles=False)
        else:
            sub = GlyphText('完 · The End', 38, FONT_SERIF, spacing=3.0, max_pts=200, seed=13, tracking=0.1)
            pt = ParticleTitle(main, sub, sub_gap=124, seed=31, sub_particles=False)
        _CACHE[which] = pt
    return _CACHE[which]
