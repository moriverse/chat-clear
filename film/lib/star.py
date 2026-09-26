"""小光 (Guang) the fallen star child, and the constellation whale (her mother).
Owner: STAR agent.

CONTRACT: keep StarPose fields, draw_star and draw_whale signatures working
(add fields/kwargs with defaults freely). STUB=True means placeholder art.

Quick reference
---------------
    from lib.star import StarPose, draw_star, draw_whale, draw_star_family

    a = draw_star(c, StarPose(x, y, scale=1.0, glow=1.0, eyes='happy',
                              mouth=f.mouth('guang'), arm_r=0.8), f.T)
    a['center'], a['top'], a['bottom'], a['hand_l'], a['hand_r']     # world coords

    w = draw_whale(c, f.T, x, y, scale=0.8, facing=-1, swim=f.T * 0.5, mouth=f.mouth('whale'))
    w['eye'], w['mouth'], w['tail'], w['belly'], w['head_top']        # world coords

Everything is a pure function of the inputs and `t` (seeded randomness only).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import skia

from engine.core import (ADD, at, clamp, col, fill, hexrgb, lerp, linear, mix, noise1, nprng, poly, radial,
                         smooth_path, smoothstep, soft_glow, stroke)

STUB = False
RADIUS = 55  # outer point radius at scale=1

TAU = math.tau

# ----------------------------------------------------------------------------
# palette (STYLE.md)
# ----------------------------------------------------------------------------
G_CORE = '#fffbea'
G_BODY = '#ffe58a'
G_EDGE = '#ffc94a'
G_GLOW = '#ffd86b'
G_WARM = '#ffb347'
G_CHEEK = '#ff9fb2'
G_EYE = '#2a2140'
G_DIM_BODY = '#8d97b8'
G_DIM_EDGE = '#6b7599'
G_DIM_CORE = '#a3acc8'
# The engine bloom adds up to ~+0.5 per channel over bright areas, which clips the STYLE gold
# (#ffe58a) to lemon. These pre-compensated values read as the STYLE gold *after* bloom.
G_R_CORE = '#fff0cc'
G_R_BODY = '#ffcf6a'
G_R_EDGE = '#ffa83e'
G_R_DEEP = '#f98a2e'

W_BLUE = '#3d7bd9'
W_VIOLET = '#8a5bd9'
W_LINE = '#9fd8ff'
W_NODE = '#ffffff'


def _rgb(c):
    return hexrgb(c) if isinstance(c, str) else tuple(c[:3])


def _hh(i, seed=0):
    """Deterministic hash -> [0, 1)."""
    x = (int(i) * 374761393 + int(seed) * 668265263 + 0x9E3779B9) & 0xFFFFFFFF
    x = ((x ^ (x >> 13)) * 1274126177) & 0xFFFFFFFF
    x = ((x ^ (x >> 16)) * 2246822519) & 0xFFFFFFFF
    return ((x ^ (x >> 15)) & 0xFFFFFF) / float(0x1000000)


def _frac(x):
    return x - math.floor(x)


def _ctm_scale(c):
    m = c.getTotalMatrix()
    return math.sqrt(abs(m.getScaleX() * m.getScaleY() - m.getSkewX() * m.getSkewY())) or 1.0


# ----------------------------------------------------------------------------
# shared sparkle glint
# ----------------------------------------------------------------------------
def _make_glint_path():
    p = skia.Path()
    k = 0.09
    p.moveTo(0, -1)
    p.quadTo(k, -k, 1, 0)
    p.quadTo(k, k, 0, 1)
    p.quadTo(-k, k, -1, 0)
    p.quadTo(-k, -k, 0, -1)
    p.close()
    return p


_GLINT = _make_glint_path()


def glint(c, x, y, size, color='#fff6d8', alpha=1.0, rot=0.0, core=True):
    """A soft 4-point twinkle (additive). size = half length of the long rays."""
    if alpha <= 0.004 or size <= 0.05:
        return
    a = clamp(alpha)
    c.save()
    c.translate(x, y)
    if rot:
        c.rotate(math.degrees(rot))
    c.scale(size, size)
    c.drawPath(_GLINT, fill(color, a, blend=ADD))
    c.restore()
    if core:
        p = radial((x, y), size * 0.42, [(0, '#ffffff', a), (0.35, color, a * 0.55), (1, color, 0)])
        p.setBlendMode(ADD)
        c.drawCircle(x, y, size * 0.42, p)


def _dot(c, x, y, r, color, a):
    """Additive soft dot."""
    if a <= 0.004 or r <= 0.05:
        return
    p = radial((x, y), r, [(0, color, a), (0.3, color, a * 0.5), (1, color, 0)])
    p.setBlendMode(ADD)
    c.drawCircle(x, y, r, p)


# ============================================================================
# 小光 GUANG
# ============================================================================
@dataclass
class StarPose:
    x: float = 960.0
    y: float = 540.0
    scale: float = 1.0          # 1.0 → ~110 units across
    rot: float = 0.0            # body rotation radians
    glow: float = 1.0           # 0 = almost extinguished (grey-blue, no halo) .. 1 healthy .. 2 super bright
    flicker: float = 0.0        # 0..1 amount of unstable flicker (when dying)
    squash: float = 0.0         # -1..1 squash/stretch (+ = squashed flat & wide, - = stretched tall)
    # "arms" = the two upper side points; angles in radians (0 = rest).
    # POSITIVE = raise the point upward (towards the head point), negative = droop down.
    # arm_l = the screen-left point, arm_r = the screen-right point.  ~0.8 wave, ~1.3 point up.
    arm_l: float = 0.0
    arm_r: float = 0.0
    # face
    eyes: str = 'open'          # 'open','happy','sad','closed','teary','surprised','sleepy'
    blink: float = 0.0          # 0 open .. 1 closed
    look: tuple = (0.0, 0.0)    # -1..1 gaze offset (x, y)
    mouth: float = 0.0          # lip-sync 0..1 (f.mouth('guang'))
    smile: float = 0.4          # -1..1
    blush: float = 0.6
    tears: float = 0.0          # 0..1 visible tear streams
    trail: float = 0.0          # 0..1 sparkle trail strength (when moving/flying)
    vx: float = 0.0             # velocity hint (units/s) for trail direction & stretch
    vy: float = 0.0
    wet: float = 0.0            # 0..1 just out of the water: drips + sheen
    # --- additions (all optional) ---
    hug: float = 0.0            # 0..1 both arm points wrap forward/inward (hugging Deng's face)
    cover: float = 0.0          # 0..1 arm points fold over the eyes (shy / covering face)
    leg_l: float = 0.0          # lower points; + = swing outward/up, - = inward (e.g. sitting, kicking)
    leg_r: float = 0.0
    rim: float = 0.0            # 0..1 rim light strength
    rim_color: str = '#9fc4ff'  # cool moon rim by default; use '#ffd36b' near the lamp
    rim_dir: float = -math.pi / 2   # world direction the rim light comes FROM (default: above)
    color: Optional[str] = None     # tint override (star children); None = Guang's gold
    halo: float = 1.0           # multiplier for halo + rays (lower it when she's cupped in hands)
    sparkle: float = 1.0        # multiplier for idle twinkles around her
    brow: Optional[float] = None    # tiny brows: None=auto per expression, -1 worried .. +1 determined
    squash_pivot: float = 0.0   # 0 = squash about centre, 1 = about the bottom (landing)
    alpha: float = 1.0          # overall opacity (fade in/out)
    shiver: float = 0.0         # 0..1 cold trembling (deterministic jitter)


# --- geometry constants (local units, scale 1) ---
_R = float(RADIUS)
_RI = 0.5 * _R          # valley radius (plump)
_RT = 0.165 * _R        # tip radius
_RP = 12.0              # limb pivot distance from centre
_BULGE = 2.4
_LIMB_A = [-math.pi / 2 + k * TAU / 5 for k in range(5)]   # 0 top, 1 up-right, 2 low-right, 3 low-left, 4 up-left
_L0 = _R - _RT - _RP
# half width at the pivot so that the rest edge passes through the valley point
_AX_V = _RI * math.cos(math.pi / 5)
_LAT_V = _RI * math.sin(math.pi / 5)
_SLOPE = (_LAT_V - _RT) / ((_R - _RT) - _AX_V)
_W0 = _LAT_V + _SLOPE * (_AX_V - _RP)
_NS = 13                # samples per limb edge
_S = np.linspace(0.0, 1.0, _NS)
_SF = 0.46              # where valley fillets attach on the edges

EYE_X = 14.8
EYE_Y = -1.0
MOUTH_Y = 12.2


class _Limb:
    """One rounded point of the star, built along a (possibly bent) centre line.
    rot: rigid rotation at the pivot; curl: extra bending towards the tip (profile s**cp)."""
    __slots__ = ('s', 'cx', 'cy', 'th', 'w', 'left', 'right', 'tip', 'tr', 'th_tip', 'a')

    def __init__(self, a, rot=0.0, curl=0.0, lscale=1.0, cp=1.0, pivot=None, direction=None, widths=None):
        self.a = a
        th0 = (a if direction is None else direction) + rot
        px, py = pivot if pivot is not None else (_RP * math.cos(a), _RP * math.sin(a))
        L = _L0 * lscale
        s = _S
        th = th0 + curl * s ** cp
        ds = L / (_NS - 1)
        co, sn = np.cos(th), np.sin(th)
        cx = px + np.concatenate([[0.0], np.cumsum((co[:-1] + co[1:]) * 0.5 * ds)])
        cy = py + np.concatenate([[0.0], np.cumsum((sn[:-1] + sn[1:]) * 0.5 * ds)])
        w = self.width(s) if widths is None else widths
        self.s, self.cx, self.cy, self.th, self.w = s, cx, cy, th, w
        nx, ny = np.sin(th), -np.cos(th)          # normal towards the (visually) counter-clockwise side
        self.left = np.stack([cx + nx * w, cy + ny * w], 1)
        self.right = np.stack([cx - nx * w, cy - ny * w], 1)
        self.tip = (float(cx[-1]), float(cy[-1]))
        self.tr = float(w[-1])
        self.th_tip = float(th[-1])

    @staticmethod
    def width(s):
        s = np.asarray(s, dtype=float)
        return _W0 + (_RT - _W0) * s + _BULGE * np.sin(np.pi * s) * (1 - 0.3 * s)

    def edge_pt(self, s, side):
        cx = np.interp(s, self.s, self.cx)
        cy = np.interp(s, self.s, self.cy)
        th = np.interp(s, self.s, self.th)
        w = np.interp(s, self.s, self.w)
        sg = 1.0 if side == 'l' else -1.0
        return (float(cx + sg * math.sin(th) * w), float(cy - sg * math.cos(th) * w))

    def add_to(self, path, s_from=0):
        """Closed contour of this limb (clockwise), starting at sample index s_from."""
        L, R_ = self.left, self.right
        path.moveTo(float(L[s_from, 0]), float(L[s_from, 1]))
        for i in range(s_from + 1, _NS):
            path.lineTo(float(L[i, 0]), float(L[i, 1]))
        tx, ty, r = self.tip[0], self.tip[1], self.tr
        path.arcTo(skia.Rect.MakeLTRB(tx - r, ty - r, tx + r, ty + r),
                   math.degrees(self.th_tip) - 90.0, 180.0, False)
        for i in range(_NS - 1, s_from - 1, -1):
            path.lineTo(float(R_[i, 0]), float(R_[i, 1]))
        path.close()


def _line_isect(p1, d1, p2, d2):
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
    return (p1[0] + d1[0] * t, p1[1] + d1[1] * t)


def _limb_params(pose, t):
    """-> (kwargs for the 5 silhouette limbs, kwargs for arms drawn in front of the face)."""
    sway = 0.045 * math.sin(t * 2.1) + 0.03 * noise1(t * 0.9, 17)
    if pose.shiver > 0:
        sway += pose.shiver * 0.06 * noise1(t * 31.0, 23)
    hug = clamp(pose.hug)
    cover = clamp(pose.cover)
    ce = cover * cover * (3 - 2 * cover)
    out, front = [], []
    for k in range(5):
        a = _LIMB_A[k]
        kw = dict(a=a)
        if k in (1, 4):
            sg = -1.0 if k == 1 else 1.0          # screen-right point raises counter-clockwise
            A = (pose.arm_r if k == 1 else pose.arm_l) + sway * 0.7 * (-sg)
            if A >= 0:
                T = 0.9 * A
                rot, curl, cp, ls = 0.22 * T, 0.78 * T, 2.0, 1.0 + 0.08 * min(A, 1.5)
            else:
                rot, curl, cp, ls = 0.55 * A, 0.35 * A, 1.0, 1.0
            if hug > 0:
                rot += 0.28 * hug
                curl += -2.1 * hug
                cp = lerp(cp, 1.5, hug)
            if ce > 0:
                # fold forward: the point foreshortens to a nub, then re-appears in front over the eye
                ls = lerp(ls, 0.5, min(1.0, ce * 2))
                rot += 0.12 * min(1.0, ce * 2)
                curl *= 1 - min(1.0, ce * 2)
                if ce > 0.5:
                    u = (ce - 0.5) / 0.5
                    u = u * u * (3 - 2 * u)
                    ex, ey = -sg * (EYE_X - 1.0), EYE_Y + 1.0
                    px_, py_ = 30.0 * math.cos(a), 30.0 * math.sin(a)
                    dist = math.hypot(ex - px_, ey - py_)
                    front.append(dict(a=a, pivot=(px_, py_), direction=math.atan2(ey - py_, ex - px_) - sg * 0.35,
                                      curl=sg * 0.7, lscale=max(0.02, dist * u / _L0),
                                      widths=np.linspace(11.5, 11.0, _NS)))
            kw.update(rot=sg * rot, curl=sg * curl, cp=cp, lscale=ls)
        elif k in (2, 3):
            lg = pose.leg_r if k == 2 else pose.leg_l
            sg = -1.0 if k == 2 else 1.0
            kw.update(rot=sg * 0.6 * lg, curl=sg * 0.3 * lg)
        else:
            kw.update(rot=0.02 * math.sin(t * 1.7 + 1.0))
        out.append(kw)
    return out, front


def _body_path(limbs):
    path = skia.Path()
    path.setFillType(skia.PathFillType.kWinding)
    # core: straight across each limb base, rounded fillets in the valleys
    fp = []
    for k in range(5):
        lb = limbs[k]
        fp.append((lb.edge_pt(_SF, 'l'), lb.edge_pt(_SF, 'r'),
                   lb.edge_pt(_SF + 0.06, 'l'), lb.edge_pt(_SF + 0.06, 'r')))
    path.moveTo(*fp[0][0])
    for k in range(5):
        l0, r0, l1, r1 = fp[k]
        nl0, _, nl1, _ = fp[(k + 1) % 5]
        path.lineTo(*r0)
        # control point = intersection of both edge tangents (inward directions)
        d1 = (r0[0] - r1[0], r0[1] - r1[1])
        d2 = (nl0[0] - nl1[0], nl0[1] - nl1[1])
        v = _line_isect(r0, d1, nl0, d2)
        mx, my = (r0[0] + nl0[0]) / 2, (r0[1] + nl0[1]) / 2
        if v is None or math.hypot(v[0] - mx, v[1] - my) > 14 or math.hypot(*v) < 12:
            rm = math.hypot(mx, my) or 1
            v = (mx / rm * (rm - 2.5), my / rm * (rm - 2.5))
        path.quadTo(v[0], v[1], *nl0)
    path.close()
    for lb in limbs:
        lb.add_to(path)
    return path


def _flicker(t, amt):
    if amt <= 0:
        return 1.0
    n1 = 0.5 + 0.5 * noise1(t * 7.3, 91)
    n2 = 0.5 + 0.5 * noise1(t * 21.7, 92)
    dip = smoothstep(0.25, 0.75, noise1(t * 2.3, 93))
    dip2 = smoothstep(0.6, 0.9, noise1(t * 11.0, 94))
    k = 0.3 * n1 + 0.15 * n2 + 0.55 * dip + 0.4 * dip2
    return clamp(1.0 - amt * min(1.0, k))


def _palette(pose, ge):
    """-> dict of rgb tuples for the current glow level."""
    k = clamp(ge) ** 1.2
    kc = clamp(ge) ** 1.8
    if pose.color:
        base = _rgb(pose.color)
        core_f = mix(base, '#ffffff', 0.8)
        edge_f = tuple(clamp(v * 0.86 - 0.02) for v in base)
        body = mix(G_DIM_BODY, base, k)
        edge = mix(G_DIM_EDGE, edge_f, k)
        core = mix(mix(G_DIM_BODY, '#b8c0da', 0.6), core_f, kc)
        glow = base
        warm = mix(base, '#ffffff', 0.25)
        deep = tuple(v * 0.8 for v in edge)
    else:
        body = mix(G_DIM_BODY, G_R_BODY, k)
        edge = mix(G_DIM_EDGE, G_R_EDGE, k)
        core = mix(mix(G_DIM_BODY, '#b8c0da', 0.6), G_R_CORE, kc)
        deep = mix(G_DIM_EDGE, G_R_DEEP, k)
        glow = _rgb(G_GLOW)
        warm = _rgb(G_WARM)
        if ge > 1:
            u = clamp(ge - 1)
            core = mix(core, '#ffffff', u * 0.8)
            body = mix(body, '#ffe08a', u * 0.45)
            edge = mix(edge, '#ffc45a', u * 0.35)
    return dict(core=core, body=body, edge=edge, deep=deep, glow=glow, warm=warm, k=k)


def _star_matrix(pose, t):
    s = pose.scale
    m = skia.Matrix()
    jx = jy = 0.0
    if pose.shiver > 0:
        jx = pose.shiver * 1.3 * s * noise1(t * 40.0, 31)
        jy = pose.shiver * 0.8 * s * noise1(t * 43.0, 37)
    m.setTranslate(pose.x + jx, pose.y + jy)
    sp = math.hypot(pose.vx, pose.vy)
    if sp > 1:
        st = clamp(sp / 2200.0, 0, 0.22)
        ang = math.degrees(math.atan2(pose.vy, pose.vx))
        m.preRotate(ang)
        m.preScale(1 + st, 1 - st * 0.55)
        m.preRotate(-ang)
    if pose.rot:
        m.preRotate(math.degrees(pose.rot))
    sq = clamp(pose.squash, -1, 1)
    if sq:
        py = _R * 0.81 * s * clamp(pose.squash_pivot)
        m.preTranslate(0, py)
        m.preScale(1 + 0.28 * sq, 1 - 0.28 * sq)
        m.preTranslate(0, -py)
    m.preScale(s, s)
    return m


def _draw_face(c, pose, pal, t, px):
    """Eyes, cheeks, mouth, tears. Local star units. px = one device pixel in local units."""
    ex0, ey0 = EYE_X, EYE_Y
    lx, ly = pose.look
    lx, ly = clamp(lx, -1, 1), clamp(ly, -1, 1)
    fx, fy = lx * 2.6, ly * 2.0          # whole-face shift with gaze (fake 3D)
    expr = pose.eyes or 'open'
    op = clamp(1.0 - pose.blink)
    eye_col = G_EYE
    dim = 1.0 - pal['k']

    # --- cheeks ---------------------------------------------------------------
    if pose.blush > 0:
        ba = clamp(pose.blush) * (0.72 - 0.25 * dim)
        for sx in (-1, 1):
            cx = sx * 22.8 + fx * 0.6
            p = fill(G_CHEEK, ba, blur=max(2.4, 1.2 * px))
            c.drawOval(skia.Rect.MakeXYWH(cx - 6.2, 6.4 + fy * 0.5, 12.4, 7.0), p)

    # --- brows ------------------------------------------------------------------
    brow = pose.brow
    if brow is None:
        brow = {'sad': -0.8, 'teary': -0.9, 'surprised': 0.0, 'sleepy': None}.get(expr, None)
        if expr == 'surprised':
            brow = 0.01
    if brow is not None:
        bcol = mix(pal['edge'], '#b26a1e', 0.55)
        lift = 3.0 if expr == 'surprised' else 0.0
        for sx in (-1, 1):
            bx = sx * ex0 + fx
            by = ey0 - 12.8 + fy - lift
            tilt = -brow * 2.4          # worried: inner end up
            x_in, x_out = bx - sx * 4.2, bx + sx * 3.4
            y_in, y_out = by + tilt * 0.5 * (-1), by + tilt * 0.5
            y_in = by - (-brow) * 1.5 if brow < 0 else by + brow * 1.5
            y_out = by + (-brow) * 1.0 if brow < 0 else by - brow * 1.0
            bp = skia.Path()
            bp.moveTo(x_in, y_in)
            bp.quadTo((x_in + x_out) / 2, min(y_in, y_out) - 1.2, x_out, y_out)
            c.drawPath(bp, stroke(bcol, max(1.7, 0.9 * px), 0.85))

    # --- eyes -------------------------------------------------------------------
    for sx in (-1, 1):
        ex = sx * ex0 + fx
        ey = ey0 + fy
        if expr == 'happy':
            p = skia.Path()
            w = 13.2
            p.moveTo(ex - w * 0.55, ey + 2.6)
            p.quadTo(ex, ey - 9.4, ex + w * 0.55, ey + 2.6)
            c.drawPath(p, stroke(eye_col, max(3.4, 1.1 * px)))
            continue
        if expr == 'closed' or op < 0.14:
            p = skia.Path()
            w = 12.8
            p.moveTo(ex - w * 0.55, ey + 0.8)
            p.quadTo(ex, ey + 6.2, ex + w * 0.55, ey + 0.8)
            c.drawPath(p, stroke(eye_col, max(2.9, 1.0 * px)))
            continue
        w, h = 12.6, 16.2
        if expr == 'teary':
            w, h = 13.4, 17.4
        elif expr == 'surprised':
            w, h = 14.0, 16.6
        elif expr == 'sad':
            w, h = 12.2, 15.0
        hh = h * op
        cyy = ey + (h - hh) * 0.3
        rect = skia.Rect.MakeXYWH(ex - w / 2, cyy - hh / 2, w, hh)
        c.save()
        # lids
        if expr == 'sleepy':
            lid = cyy - hh / 2 + hh * 0.52
            lp = skia.Path()
            lp.moveTo(ex - w, lid - 0.8)
            lp.quadTo(ex, lid + 1.6, ex + w, lid - 0.8)
            lp.lineTo(ex + w, cyy + h)
            lp.lineTo(ex - w, cyy + h)
            lp.close()
            c.clipPath(lp, doAntiAlias=True)
        elif expr in ('sad',):
            top = cyy - hh / 2
            lp = skia.Path()
            x_in, x_out = ex - sx * w * 0.7, ex + sx * w * 0.7
            lp.moveTo(x_in, top - 2)
            lp.lineTo(x_in, top + hh * 0.08)
            lp.quadTo(ex, top + hh * 0.12, x_out, top + hh * 0.42)
            lp.lineTo(x_out, top - 2)
            lp.close()
            c.clipPath(lp, skia.ClipOp.kDifference, doAntiAlias=True)
        # iris
        ip = linear((ex, cyy - hh / 2), (ex, cyy + hh / 2),
                    [(0, '#171029', 1), (0.55, eye_col, 1), (1, '#54428a', 1)])
        ip.setAntiAlias(True)
        c.drawOval(rect, ip)
        # highlights (stay fixed-ish, the eye moves under them a little)
        hl = 1.0
        wob = 0.0
        if expr == 'teary':
            wob = 0.35 * math.sin(t * 9.0) + 0.25 * math.sin(t * 13.7 + sx)
        r1 = w * (0.27 if expr != 'surprised' else 0.22)
        hx = ex - w * 0.2 - lx * 0.8 + wob * 0.4
        hy = cyy - hh * 0.2 + wob * 0.3
        c.drawOval(skia.Rect.MakeXYWH(hx - r1, hy - r1 * min(1, op * 1.1), 2 * r1, 2 * r1 * min(1, op * 1.1)),
                   fill('#ffffff', 0.97 * hl))
        r2 = w * 0.12
        c.drawCircle(ex + w * 0.2 - lx * 0.5, cyy + hh * 0.22, r2, fill('#ffffff', 0.9))
        if expr == 'teary':
            # watery bottom crescent + extra sparkle
            wp = skia.Path()
            wp.addOval(rect)
            c.save()
            c.clipPath(wp, doAntiAlias=True)
            c.drawOval(skia.Rect.MakeXYWH(ex - w * 0.62, cyy + hh * 0.12, w * 1.24, hh * 0.7),
                       fill('#9fd8ff', 0.45))
            c.restore()
            c.drawCircle(ex + w * 0.05 + wob * 0.3, cyy + hh * 0.33, w * 0.07, fill('#ffffff', 0.95))
            c.drawCircle(ex - w * 0.3, cyy + hh * 0.05, w * 0.06, fill('#ffffff', 0.7))
        # warm reflected light in the lower iris
        c.drawOval(skia.Rect.MakeXYWH(ex - w * 0.3, cyy + hh * 0.1, w * 0.6, hh * 0.3),
                   fill(pal['glow'], 0.12 * pal['k'], blur=1.2))
        c.restore()
        if expr == 'sleepy':
            lid = cyy - hh / 2 + hh * 0.52
            lp = skia.Path()
            lp.moveTo(ex - w * 0.55, lid - 0.2)
            lp.quadTo(ex, lid + 1.4, ex + w * 0.55, lid - 0.2)
            c.drawPath(lp, stroke(eye_col, max(1.6, 0.9 * px)))

    # --- tears ------------------------------------------------------------------
    tears = clamp(pose.tears)
    if tears > 0:
        for sx in (-1, 1):
            x0 = sx * (ex0 + 2.8) + fx
            y0 = ey0 + 7.0 + fy
            x1, y1 = x0 + sx * 2.6, y0 + 8 + 12 * tears
            tp = skia.Path()
            tp.moveTo(x0, y0)
            tp.cubicTo(x0 + sx * 1.8, y0 + 3, x0 + sx * 1.2, y1 - 6, x1, y1)
            c.drawPath(tp, stroke('#8fd0ff', max(3.4, px) * (0.6 + 0.4 * tears), 0.42 * tears))
            hl = skia.Path(tp)
            hl.offset(-sx * 0.7, 0)
            c.drawPath(hl, stroke('#ffffff', max(0.9, 0.6 * px), 0.6 * tears))
            # drops sliding down and hanging at the end
            for j in range(2):
                ph = _frac(t * 0.75 + j * 0.5 + (0.3 if sx > 0 else 0))
                dx = lerp(x0, x1, ph * ph)
                dy = lerp(y0, y1, ph)
                rr = (1.2 + 1.3 * ph) * (0.7 + 0.3 * tears)
                c.drawCircle(dx, dy, rr, fill('#bfe6ff', 0.6 * tears))
                c.drawCircle(dx - rr * 0.3, dy - rr * 0.35, rr * 0.38, fill('#ffffff', 0.9 * tears))
            rr = 2.0 + 0.6 * math.sin(t * 3 + sx)
            c.drawOval(skia.Rect.MakeXYWH(x1 - rr, y1 - rr * 0.8, rr * 2, rr * 2.2), fill('#bfe6ff', 0.55 * tears))
            c.drawCircle(x1 - rr * 0.3, y1 - rr * 0.1, rr * 0.35, fill('#ffffff', 0.85 * tears))

    # --- mouth ------------------------------------------------------------------
    m = clamp(pose.mouth)
    sm = clamp(pose.smile, -1, 1)
    mx, my = fx * 0.9, MOUTH_Y + fy * 0.8
    mcol = '#5e1f33'
    if expr == 'surprised':
        ow, oh = 3.0 + 0.8 * m, 3.6 + 3.2 * m
        c.drawOval(skia.Rect.MakeXYWH(mx - ow, my - oh * 0.6, ow * 2, oh * 1.6 * 0.75), fill(mcol))
    elif m < 0.06:
        hw = 5.0 + 1.4 * abs(sm)
        p = skia.Path()
        p.moveTo(mx - hw, my - 1.4 * sm)
        p.quadTo(mx, my + 4.4 * sm, mx + hw, my - 1.4 * sm)
        c.drawPath(p, stroke(mcol, max(1.9, 0.9 * px)))
    else:
        hw = 4.2 + 1.9 * m + 1.3 * max(sm, 0)
        hh = 1.8 + 7.4 * m
        cy_ = -1.6 * sm
        p = skia.Path()
        p.moveTo(mx - hw, my + cy_)
        p.quadTo(mx, my + 0.9 - 0.8 * sm + cy_ * 0.2, mx + hw, my + cy_)
        p.quadTo(mx + hw * 0.9, my + hh * 1.25 + max(sm, 0) * 1.5, mx, my + hh * 1.25 + max(sm, 0) * 1.5)
        p.quadTo(mx - hw * 0.9, my + hh * 1.25 + max(sm, 0) * 1.5, mx - hw, my + cy_)
        p.close()
        c.drawPath(p, fill(mcol))
        c.save()
        c.clipPath(p, doAntiAlias=True)
        tw = hw * 0.62
        c.drawOval(skia.Rect.MakeXYWH(mx - tw, my + hh * 0.72, tw * 2, hh * 0.9), fill('#ff8fa0', 0.95))
        c.restore()


def _draw_arm_front(c, lb, shader, pal, px, amt, s_from=2):
    """Draw an arm point in front of the face (cover/hug): soft cast shadow, own volume shading."""
    p = skia.Path()
    lb.add_to(p, s_from)
    tx, ty, tr = lb.tip[0], lb.tip[1], lb.tr
    # soft shadow cast on the body/face
    sh = skia.Path(p)
    sh.offset(0.8, 2.8)
    c.drawPath(sh, fill(mix(pal['deep'], '#7a3a08', 0.55), 0.5 * amt, blur=2.4))
    # base fill = the body shader (so the root melts into the body) ...
    bp = skia.Paint(AntiAlias=True)
    bp.setShader(shader)
    c.drawPath(p, bp)
    # ... plus its own round volume towards the tip
    ax, ay = float(lb.cx[0]), float(lb.cy[0])
    vol = radial((tx - tr * 0.25, ty - tr * 0.35), tr * 1.6,
                 [(0, mix(pal['core'], pal['body'], 0.45), 0.55 * amt), (0.45, pal['body'], 0.5 * amt),
                  (0.85, pal['edge'], 0.6 * amt), (1, pal['deep'], 0.0)])
    c.save()
    c.clipPath(p, doAntiAlias=True)
    c.drawCircle(tx, ty, tr * 1.6, vol)
    # darker lower rim so it separates from the face behind
    rim = skia.Path(p)
    rim.offset(-0.4, -1.8)
    c.clipPath(rim, skia.ClipOp.kDifference, doAntiAlias=True)
    lg = linear((ax, ay), (tx, ty), [(0, pal['deep'], 0), (0.5, pal['deep'], 0.8 * amt), (1, pal['deep'], 0.9 * amt)])
    c.drawPath(p, lg)
    c.restore()
    c.drawCircle(tx - tr * 0.3, ty - tr * 0.4, tr * 0.28, fill('#ffffff', 0.45 * amt, blur=1.2))


def draw_star(c, pose: StarPose, t: float = 0.0):
    """Draw Guang centred at (x, y).
    Returns dict(center, top, bottom, hand_l, hand_r, foot_l, foot_r) in world coords."""
    s = pose.scale
    g = max(0.0, pose.glow)
    ge = g * _flicker(t, pose.flicker)
    pal = _palette(pose, ge)
    M = _star_matrix(pose, t)
    params, fparams = _limb_params(pose, t)
    limbs = [_Limb(**kw) for kw in params]
    flimbs = [_Limb(**kw) for kw in fparams]
    body = _body_path(limbs)

    def W(px_, py_):
        pt = M.mapXY(px_, py_)
        return (pt.x(), pt.y())

    anchors = dict(center=(pose.x, pose.y), top=W(*limbs[0].tip),
                   bottom=W(0.0, _R * 0.81), hand_r=W(*limbs[1].tip), hand_l=W(*limbs[4].tip),
                   foot_r=W(*limbs[2].tip), foot_l=W(*limbs[3].tip))
    anchors['top'] = W(limbs[0].tip[0], limbs[0].tip[1] - _RT)
    if pose.alpha <= 0.003 or s <= 0:
        return anchors

    ctm = _ctm_scale(c)
    layered = pose.alpha < 0.999
    if layered:
        rr = _R * s * 5.5
        lp = skia.Paint()
        lp.setAlphaf(clamp(pose.alpha))
        c.saveLayer(skia.Rect.MakeLTRB(pose.x - rr, pose.y - rr, pose.x + rr, pose.y + rr), lp)

    halo_k = max(0.0, pose.halo)
    pulse = 1.0 + 0.045 * math.sin(t * 2.3) + 0.02 * math.sin(t * 5.1 + 1.3)
    speed = math.hypot(pose.vx, pose.vy)

    # ------------------------------------------------------------------ world FX
    if pose.trail > 0 and speed > 1:
        _draw_trail(c, pose, pal, t, ge, speed)

    c.save()
    c.concat(M)
    px = 1.0 / max(1e-6, ctm * s)      # one device pixel in local units

    # --- halo ------------------------------------------------------------------
    dev = ctm * s                      # device pixels per local unit
    # in close-ups the halo grows slower than the body (keeps the frame readable and cheap)
    hsc = 1.0 if dev <= 1.2 else (1.2 / dev) ** 0.45
    if ge > 0.02 and halo_k > 0:
        hg = min(ge, 2.2)
        rh = _R * (1.7 + 1.1 * min(hg, 1) + 1.3 * max(hg - 1, 0)) * pulse * max(hsc, 0.45 if hg <= 1 else 0.4)
        ah = halo_k * (0.42 * min(hg, 1) ** 1.3 + 0.25 * max(hg - 1, 0))
        # cheap 2-stop outer glow + richer small inner glow
        ho = radial((0, 0), rh, [(0, pal['warm'], ah * 0.3), (1, pal['warm'], 0)])
        ho.setBlendMode(ADD)
        c.drawCircle(0, 0, rh, ho)
        ri = max(_R * 1.25, rh * 0.5)
        hp = radial((0, 0), ri, [(0, pal['glow'], ah * 0.75), (0.45, pal['glow'], ah * 0.35), (1, pal['glow'], 0)])
        hp.setBlendMode(ADD)
        c.drawCircle(0, 0, ri, hp)
        # rays
        ar = halo_k * (0.16 * smoothstep(0.5, 1.0, hg) + 0.3 * clamp(hg - 1))
        if ar > 0.01:
            _draw_rays(c, t, pal, ar, _R * (1.9 + 1.6 * clamp(hg - 1)) * pulse * max(hsc, 0.55))
        # star-shaped glow hugging the silhouette
        c.drawPath(body, fill(pal['glow'], halo_k * 0.55 * min(hg, 1.3), blur=_R * 0.13, blend=ADD))
    elif ge <= 0.02 and halo_k > 0:
        # nearly extinguished: the faintest cold glow
        _dot(c, 0, 0, _R * 1.4, '#8d97b8', 0.06 * halo_k)

    # --- body ------------------------------------------------------------------
    sh = skia.GradientShader.MakeRadial(
        skia.Point(-_R * 0.06, -_R * 0.16), _R * 1.1,
        [col(pal['core']), col(mix(pal['core'], pal['body'], 0.55)), col(pal['body']),
         col(pal['edge']), col(pal['deep'])],
        [0.0, 0.26, 0.56, 0.84, 1.0])
    if pose.wet > 0:
        wet = clamp(pose.wet)
    else:
        wet = 0.0
    bp = skia.Paint(AntiAlias=True)
    bp.setShader(sh)
    c.drawPath(body, bp)

    c.save()
    c.clipPath(body, doAntiAlias=True)
    # lower volume shade
    shade = mix(pal['edge'], '#c9731c', 0.45) if not pose.color else tuple(v * 0.8 for v in pal['edge'])
    c.drawRect(skia.Rect.MakeLTRB(-_R * 1.5, -_R * 1.5, _R * 1.5, _R * 1.5),
               linear((0, 2), (0, _R * 0.95), [(0, shade, 0), (1, shade, 0.42 + 0.1 * (1 - pal['k']))]))
    # warm inner edge
    c.drawPath(body, stroke(pal['edge'], 7.0, 0.55, blur=3.0))
    # rim light
    if pose.rim > 0:
        ang = pose.rim_dir - pose.rot
        dx, dy = -math.cos(ang) * 2.8, -math.sin(ang) * 2.8
        sp_ = skia.Path(body)
        sp_.offset(dx, dy)
        c.save()
        c.clipPath(sp_, skia.ClipOp.kDifference, doAntiAlias=True)
        c.drawPath(body, fill(pose.rim_color, 0.8 * clamp(pose.rim), blur=1.6, blend=ADD))
        c.restore()
    # wet: cool water film
    if wet > 0:
        c.drawRect(skia.Rect.MakeLTRB(-_R * 1.5, -_R * 1.5, _R * 1.5, _R * 1.5), fill('#bcd6ff', 0.16 * wet))
    # inner core glow
    if ge > 0.05:
        ig = min(ge, 1.8)
        _dot(c, 0, -_R * 0.2, _R * 0.75, pal['core'], 0.1 * ig)
    # glossy sheen top-left
    c.save()
    c.translate(-_R * 0.28, -_R * 0.42)
    c.rotate(-32)
    c.drawOval(skia.Rect.MakeXYWH(-9, -3.6, 18, 7.2), fill('#ffffff', 0.5 + 0.3 * wet, blur=2.6))
    c.restore()
    c.drawCircle(-_R * 0.43, -_R * 0.2, 1.9, fill('#ffffff', 0.55 + 0.35 * wet, blur=0.6))
    if wet > 0:
        # water beads & streaks catching the light
        for i in range(6):
            bx = (_hh(i, 5) - 0.5) * _R * 1.1
            by = (_hh(i, 6) - 0.5) * _R * 1.0
            if abs(bx) < 20 and -12 < by < 16:
                continue   # keep the face clear
            rb = 1.2 + 1.3 * _hh(i, 7)
            c.drawCircle(bx, by, rb, fill('#eaf6ff', 0.55 * wet))
            c.drawCircle(bx - rb * 0.3, by - rb * 0.35, rb * 0.4, fill('#ffffff', 0.9 * wet))
        c.save()
        c.translate(_R * 0.3, -_R * 0.1)
        c.rotate(58)
        c.drawOval(skia.Rect.MakeXYWH(-10, -1.5, 20, 3), fill('#ffffff', 0.35 * wet, blur=1.2))
        c.restore()
    c.restore()

    # --- face ------------------------------------------------------------------
    _draw_face(c, pose, pal, t, px)

    # --- arms in front (cover / hug) ------------------------------------------------
    if pose.hug > 0.02:
        for k in (1, 4):
            _draw_arm_front(c, limbs[k], sh, pal, px, smoothstep(0.0, 0.35, pose.hug), 2)
    for lb in flimbs:
        _draw_arm_front(c, lb, sh, pal, px, 1.0, 0)

    # --- twinkles around her ----------------------------------------------------------
    if pose.sparkle > 0 and ge > 0.3:
        _draw_idle_sparkles(c, t, pal, ge, pose.sparkle, px)

    c.restore()

    # ------------------------------------------------------------------ drips (world)
    if pose.wet > 0.01:
        _draw_drips(c, pose, limbs, M, t, s)

    if layered:
        c.restore()
    return anchors


def _draw_rays(c, t, pal, a, length):
    """Soft rotating light rays (radial-gradient wedges, a wide faint + a narrow bright pass)."""
    n = 10
    base = t * 0.12
    for wk, ak in ((1.9, 0.45), (0.8, 0.8)):
        p = skia.Path()
        for i in range(n):
            ang = base + i * TAU / n + (TAU / 20)
            ln = length * (0.75 + 0.35 * (i % 2 == 0)) * (0.85 + 0.15 * math.sin(t * 1.3 + i * 2.1))
            hw = (0.07 if i % 2 == 0 else 0.045) * wk
            r0 = _R * 0.35
            p.moveTo(math.cos(ang - hw) * r0, math.sin(ang - hw) * r0)
            p.quadTo(math.cos(ang - hw * 0.35) * ln * 0.55, math.sin(ang - hw * 0.35) * ln * 0.55,
                     math.cos(ang) * ln, math.sin(ang) * ln)
            p.quadTo(math.cos(ang + hw * 0.35) * ln * 0.55, math.sin(ang + hw * 0.35) * ln * 0.55,
                     math.cos(ang + hw) * r0, math.sin(ang + hw) * r0)
            p.close()
        rp = radial((0, 0), length * 1.05, [(0, pal['core'], a * ak), (0.3, pal['glow'], a * 0.6 * ak),
                                           (1, pal['glow'], 0)])
        rp.setBlendMode(ADD)
        c.drawPath(p, rp)


def _draw_idle_sparkles(c, t, pal, ge, amt, px):
    n = 5 + int(round(5 * clamp(ge - 1)))
    base_a = amt * smoothstep(0.3, 1.0, ge) * (1 + 0.3 * clamp(ge - 1))
    for i in range(n):
        per = 1.7 + 1.1 * _hh(i, 41)
        ph = t / per + _hh(i, 42)
        cyc = math.floor(ph)
        life = ph - cyc
        ang = TAU * _hh(cyc * 7 + i, 43)
        rad = _R * (1.05 + 0.75 * _hh(cyc * 7 + i, 44))
        x = math.cos(ang) * rad
        y = math.sin(ang) * rad - life * 8
        a = base_a * math.sin(math.pi * life) ** 1.5
        size = (5.5 + 4.5 * _hh(cyc * 7 + i, 45)) * (0.6 + 0.4 * math.sin(math.pi * life))
        size = max(size, 2.2 * px)
        glint(c, x, y, size, mix(pal['core'], pal['glow'], 0.3), a)


def _draw_trail(c, pose, pal, t, ge, speed):
    s = pose.scale
    tr = clamp(pose.trail)
    dx, dy = -pose.vx / speed, -pose.vy / speed
    nx, ny = -dy, dx
    L = min(speed * 0.32, 560.0) * max(0.45, s)
    x0, y0 = pose.x, pose.y
    bright = max(0.25, min(ge, 1.6))
    # comet streak
    w0 = _R * 0.62 * s
    p = skia.Path()
    p.moveTo(x0 + nx * w0, y0 + ny * w0)
    p.quadTo(x0 + dx * L * 0.35 + nx * w0 * 0.6, y0 + dy * L * 0.35 + ny * w0 * 0.6, x0 + dx * L, y0 + dy * L)
    p.quadTo(x0 + dx * L * 0.35 - nx * w0 * 0.6, y0 + dy * L * 0.35 - ny * w0 * 0.6, x0 - nx * w0, y0 - ny * w0)
    p.close()
    lp = linear((x0, y0), (x0 + dx * L, y0 + dy * L),
                [(0, pal['glow'], 0.5 * tr * bright), (0.4, pal['warm'], 0.18 * tr * bright), (1, pal['warm'], 0)])
    lp.setBlendMode(ADD)
    lp.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, max(2.0, 6 * s)))
    c.drawPath(p, lp)
    # sparkles flowing back
    n = 26
    for i in range(n):
        per = 0.55 + 0.55 * _hh(i, 61)
        ph = t / per + _hh(i, 62)
        cyc = math.floor(ph)
        age = ph - cyc
        j = cyc * 13 + i
        dist = age * L * (0.55 + 0.55 * _hh(j, 63))
        side = (_hh(j, 64) - 0.5) * _R * s * 1.3 * (0.35 + age)
        px_ = x0 + dx * dist + nx * side
        py_ = y0 + dy * dist + ny * side + age * age * 18 * s
        a = tr * (1 - age) ** 1.2 * (0.65 + 0.35 * math.sin(t * 23 + i * 1.7)) * bright
        sz = s * (4.0 + 5.0 * _hh(j, 65)) * (1 - age * 0.7)
        colr = mix(pal['core'], pal['glow'], _hh(j, 66))
        if i % 3 == 0:
            glint(c, px_, py_, sz * 1.5, colr, a)
        else:
            _dot(c, px_, py_, sz, colr, a * 0.9)
            c.drawCircle(px_, py_, max(0.6, sz * 0.22), fill('#ffffff', a, blend=ADD))


def _draw_drips(c, pose, limbs, M, t, s):
    wet = clamp(pose.wet)
    pts = []
    for k in range(5):
        lb = limbs[k]
        tx, ty = lb.tip
        # lowest point of the tip cap in world space
        best = None
        for aa in np.linspace(0, TAU, 12, endpoint=False):
            q = M.mapXY(tx + math.cos(aa) * lb.tr, ty + math.sin(aa) * lb.tr)
            if best is None or q.y() > best[1]:
                best = (q.x(), q.y())
        pts.append(best)
    q = M.mapXY(0, _R * 0.55)
    pts.append((q.x(), q.y()))
    cy0 = pose.y
    grav = 1600.0 * max(0.35, s)
    for i, (x, y) in enumerate(pts):
        if y < cy0 - _R * s * 0.25:
            continue   # tips pointing up don't drip
        per = 0.8 + 0.6 * _hh(i, 71)
        ph = t / per + _hh(i, 72)
        life = ph - math.floor(ph)
        grow = 0.55
        if _hh(math.floor(ph) * 5 + i, 73) > 0.35 + 0.6 * wet:
            continue
        if life < grow:
            r = s * (1.0 + 2.2 * life / grow)
            dy = r * 0.8
            c.drawOval(skia.Rect.MakeXYWH(x - r, y - r * 0.3, 2 * r, 2.1 * r), fill('#cfe8ff', 0.7 * wet))
            c.drawCircle(x - r * 0.3, y + dy * 0.5, r * 0.35, fill('#ffffff', 0.9 * wet))
        else:
            tau = (life - grow) * per
            fy = y + 0.5 * grav * tau * tau + s * 2
            r = s * 2.6 * (1 - 0.3 * (life - grow) / (1 - grow))
            a = wet * (1 - smoothstep(0.75, 1.0, life))
            c.drawOval(skia.Rect.MakeXYWH(x - r * 0.8, fy - r * 1.3, r * 1.6, r * 2.6), fill('#d8eeff', 0.75 * a))
            c.drawCircle(x - r * 0.25, fy - r * 0.4, r * 0.35, fill('#ffffff', 0.95 * a))
            _dot(c, x, fy, r * 3, '#ffe9a8', 0.25 * a)


# ============================================================================
# 星鲸 THE CONSTELLATION WHALE
# ============================================================================
_TEX = {}
_TW, _TH = 1024, 256


def _zoom_noise(rng_, gh, gw, H, W):
    from scipy.ndimage import zoom
    g = rng_.random((gh + 3, gw + 3))
    z = zoom(g, ((H + 3 * H / gh) / (gh + 3), (W + 3 * W / gw) / (gw + 3)), order=3)
    oy, ox = int(H / gh), int(W / gw)
    return z[oy:oy + H, ox:ox + W]


def _fbm2(seed, H, W, base=(3, 12), octaves=5):
    r = nprng(seed)
    acc = np.zeros((H, W))
    amp, tot = 1.0, 0.0
    gh, gw = base
    for o in range(octaves):
        acc += amp * _zoom_noise(r, gh, gw, H, W)
        tot += amp
        amp *= 0.55
        gh *= 2
        gw *= 2
    acc /= tot
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-9)
    return acc


def _nebula_texture():
    if 'img' in _TEX:
        return _TEX['img']
    from scipy.ndimage import map_coordinates
    H, W = _TH, _TW
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    wx = _fbm2(702, H, W, (2, 5), 3) - 0.5
    wy = _fbm2(703, H, W, (2, 5), 3) - 0.5

    def warped(seed, base, octv, k=1.0):
        n = _fbm2(seed, H, W, base, octv)
        return map_coordinates(n, [np.clip(yy + wy * 50 * k, 0, H - 1), np.clip(xx + wx * 140 * k, 0, W - 1)],
                               order=1)

    neb = warped(701, (3, 9), 6)
    neb = (neb - neb.min()) / (neb.max() - neb.min())
    clouds = _sstep(0.3, 0.85, neb)
    lanes = _sstep(0.5, 0.72, warped(707, (4, 12), 5, 0.7))
    clouds = clouds * (1 - 0.65 * lanes)
    fine = _fbm2(708, H, W, (8, 32), 4)
    clouds = np.clip(clouds * (0.75 + 0.5 * fine), 0, 1)
    fil = warped(704, (3, 11), 5, 1.2)
    ridge = 1.0 - np.abs(fil * 2 - 1)
    ridge = np.clip((ridge - 0.84) / 0.16, 0, 1) ** 2.0 * clouds
    hue = _fbm2(706, H, W, (2, 6), 3)
    u = xx / (W - 1)
    v = yy / (H - 1)
    blue = np.array(hexrgb(W_BLUE))
    vio = np.array(hexrgb(W_VIOLET))
    tcol = np.clip((u - 0.05) / 0.9 + (hue - 0.5) * 1.1, 0, 1)[..., None]
    tcol = tcol * tcol * (3 - 2 * tcol)
    base = blue * (1 - tcol) + vio * tcol
    # bright cloud cores: cyan-white towards the head, lilac/pink towards the tail
    core_col = np.array(hexrgb('#b8e2ff')) * (1 - tcol) + np.array(hexrgb('#e6c2ff')) * tcol
    belly = np.clip((v - 0.62) / 0.38, 0, 1)[..., None]
    base = base * (1 - 0.35 * belly) + np.array(hexrgb('#8fc4ff')) * 0.35 * belly
    rimv = np.exp(-((v - 0.0) / 0.08) ** 2) + 0.7 * np.exp(-((1.0 - v) / 0.1) ** 2)
    rimu = np.exp(-((u - 0.0) / 0.03) ** 2)
    rim = np.clip(rimv + rimu, 0, 1.2)
    inten = (0.05 + 0.5 * clouds ** 1.3) * (0.75 + 0.25 * np.sin(np.pi * v)) + ridge * 0.18 + rim * 0.58
    cmix = np.clip(clouds ** 2 * 0.75 + ridge * 0.5 + rim * 0.5, 0, 0.9)[..., None]
    rgb = base * (1 - cmix) + core_col * cmix
    # embedded tiny stars
    r = nprng(705)
    ns = 1500
    sy = r.integers(0, H - 1, ns)
    sx = r.integers(0, W - 1, ns)
    sb = r.random(ns) ** 3
    inten[sy, sx] += 0.45 + 0.9 * sb
    rgb[sy, sx] = rgb[sy, sx] * 0.3 + 0.7
    big = sb > 0.5
    for dy_, dx_ in ((0, 1), (1, 0), (1, 1)):
        inten[sy[big] + dy_, sx[big] + dx_] += 0.5 * sb[big]
        rgb[sy[big] + dy_, sx[big] + dx_] = rgb[sy[big] + dy_, sx[big] + dx_] * 0.4 + 0.6
    alpha = np.clip(inten, 0, 1)
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[..., :3] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    img = skia.Image.fromarray(rgba, colorType=skia.kRGBA_8888_ColorType, alphaType=skia.kUnpremul_AlphaType)
    _TEX['img'] = img
    # (mipmapped sampling renders identically here but is ~2x faster in skia-python's raster pipeline)
    _TEX['shader'] = img.withDefaultMipmaps().makeShader(
        skia.TileMode.kClamp, skia.TileMode.kClamp, skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear))
    return img


def _tex_shader():
    _nebula_texture()
    return _TEX['shader']


# ---- whale geometry ----------------------------------------------------------
_WN = 60
_WX = np.linspace(0.0, 1.0, _WN)
_WU = _WX ** 1.45                 # dense near the head for a round snout
_WM = 9                            # rows across the body
_WV = np.linspace(0.0, 1.0, _WM)


def _quarter(u, u1):
    x = np.clip(u / u1, 0.0, 1.0)
    return np.sqrt(np.clip(x * (2 - x), 0, 1))


def _sstep(a, b, x):
    tt = np.clip((x - a) / (b - a), 0, 1)
    return tt * tt * (3 - 2 * tt)


def _profile(ctrl, u):
    from scipy.interpolate import PchipInterpolator
    cu, cv = zip(*ctrl)
    return PchipInterpolator(cu, cv)(u)


# half-thickness above (top) and below (bottom) the spine, u = 0 snout .. 1 tail stock
_TOP_CTRL = [(0.0, 0.0), (0.003, 21), (0.012, 43), (0.04, 70), (0.1, 98), (0.19, 124), (0.3, 140), (0.42, 142),
             (0.54, 127), (0.66, 97), (0.77, 66), (0.87, 40), (0.94, 27), (1.0, 20)]
_BOT_CTRL = [(0.0, 0.0), (0.003, 21), (0.012, 43), (0.04, 72), (0.1, 114), (0.19, 156), (0.29, 176), (0.39, 176),
             (0.5, 156), (0.62, 118), (0.74, 78), (0.85, 46), (0.93, 28), (1.0, 20)]
_W_TOP = _profile(_TOP_CTRL, _WU) + 26.0 * np.exp(-((_WU - 0.70) / 0.03) ** 2) * (1 + 0.5 * np.tanh((_WU - 0.70) / 0.02))
_W_BOT = _profile(_BOT_CTRL, _WU)


def _spine(u, swim):
    """Spine points (x, y) for u in [0,1] (0 = snout)."""
    u = np.asarray(u, dtype=float)
    X = 690.0 - 1190.0 * u
    env = u ** 1.8
    Y = 50.0 * env * np.sin(TAU * (swim - 0.75 * u)) \
        - 6.0 * np.sin(TAU * swim + 0.8) * (1 - u) \
        - 22.0 * np.sin(np.pi * u) + 26.0 * u * u
    return X, Y


def _frames(X, Y):
    dx = np.gradient(X)
    dy = np.gradient(Y)
    ln = np.hypot(dx, dy) + 1e-9
    tx, ty = dx / ln, dy / ln
    # up-normal (towards -y when heading towards -x)
    return tx, ty, -ty, tx


def _strip_mesh(cx, cy, nx, ny, w_up, w_dn, u0, u1, rows=_WM):
    """Mesh + outline for a strip along a centreline. Returns (positions (n,rows,2), texcoords)."""
    v = np.linspace(0.0, 1.0, rows)
    off = w_up[:, None] * (1 - v[None, :]) - w_dn[:, None] * v[None, :]
    px_ = cx[:, None] + nx[:, None] * off
    py_ = cy[:, None] + ny[:, None] * off
    n = len(cx)
    tu = np.linspace(u0, u1, n)[:, None] * np.ones((1, rows))
    tv = np.ones((n, 1)) * v[None, :]
    return np.stack([px_, py_], -1), np.stack([tu * (_TW - 1), tv * (_TH - 1)], -1)


_IDX_CACHE = {}


def _indices(n, m):
    key = (n, m)
    if key not in _IDX_CACHE:
        ii, jj = np.mgrid[0:n - 1, 0:m - 1]
        a0 = (ii * m + jj).ravel()
        _IDX_CACHE[key] = np.stack([a0, a0 + m, a0 + 1, a0 + 1, a0 + m, a0 + m + 1], 1).ravel().tolist()
    return _IDX_CACHE[key]


_FADE_CACHE = {}


def _root_fade(n, m, s0):
    """Vertex colours (white, alpha ramping in over the first s0 of the strip)."""
    key = (n, m, s0)
    if key not in _FADE_CACHE:
        s = np.linspace(0, 1, n)
        a = _sstep(0.0, s0, s)
        cols = []
        for i in range(n):
            ai = int(round(255 * a[i]))
            cols += [(ai << 24) | 0xFFFFFF] * m
        _FADE_CACHE[key] = cols
    return _FADE_CACHE[key]


def _mesh_vertices(pos, tex, fade=0.0):
    n, m = pos.shape[:2]
    P = pos.reshape(-1, 2)
    T = tex.reshape(-1, 2)
    pts = [skia.Point(float(a), float(b)) for a, b in P]
    tps = [skia.Point(float(a), float(b)) for a, b in T]
    cols = _root_fade(n, m, fade) if fade > 0 else None
    return skia.Vertices.MakeCopy(skia.Vertices.kTriangles_VertexMode, pts, tps, cols, _indices(n, m))


def _outline_path(pos, close=True):
    """Outline of a strip mesh: top row forward, bottom row back."""
    top = pos[:, 0, :]
    bot = pos[:, -1, :]
    pts = np.concatenate([top, bot[::-1]], 0)
    p = skia.Path()
    p.moveTo(float(pts[0, 0]), float(pts[0, 1]))
    for a, b in pts[1:]:
        p.lineTo(float(a), float(b))
    if close:
        p.close()
    return p


def _poly_path(pts):
    p = skia.Path()
    p.moveTo(float(pts[0][0]), float(pts[0][1]))
    for a, b in pts[1:]:
        p.lineTo(float(a), float(b))
    p.close()
    return p


def _fin_geom(root, ang, curl, length, wlead, wtrail, n=18, bumps=0.0, phase=0.0):
    """Centreline strip for a fin/fluke lobe. Returns cx, cy, nx, ny, w_up, w_dn."""
    s = np.linspace(0.0, 1.0, n)
    th = ang + curl * s
    if abs(curl) < 1e-4:
        cx = root[0] + length * s * math.cos(ang)
        cy = root[1] + length * s * math.sin(ang)
    else:
        cx = root[0] + length / curl * (np.sin(th) - math.sin(ang))
        cy = root[1] - length / curl * (np.cos(th) - math.cos(ang))
    nx, ny = np.sin(th), -np.cos(th)
    wl = wlead(s)
    wt = wtrail(s)
    if bumps:
        wl = wl + bumps * np.maximum(0, np.sin(s * np.pi * 9 + phase)) ** 2 * (1 - s) * (s > 0.08)
    return cx, cy, nx, ny, wl, wt


def _whale_geometry(swim, mouth):
    u = _WU
    X, Y = _spine(u, swim)
    tx, ty, nx, ny = _frames(X, Y)
    top = _W_TOP.copy()
    bot = _W_BOT.copy()
    # lower jaw drops a touch when speaking
    bot = bot + mouth * 10.0 * np.sin(np.pi * np.clip(u / 0.3, 0, 1)) ** 1.5
    geo = dict(u=u, X=X, Y=Y, tx=tx, ty=ty, nx=nx, ny=ny, top=top, bot=bot)
    pos, tex = _strip_mesh(X, Y, nx, ny, top, bot, 0.0, 1.0)
    geo['body'] = (pos, tex)

    # --- pectoral fins (long, humpback-like) ---
    def fin(side):
        near = side == 'near'
        ua = 0.335 if near else 0.31
        i = int(np.searchsorted(u, ua))
        vdown = 0.66 if near else 0.42
        rx = X[i] - nx[i] * bot[i] * vdown
        ry = Y[i] - ny[i] * bot[i] * vdown
        flap = math.sin(TAU * swim + (0.4 if near else 1.3))
        base_ang = math.atan2(ty[i], tx[i])      # pointing to the tail
        ang = base_ang - (0.78 if near else 0.62) - 0.2 * flap
        curl = 0.42 + 0.16 * flap
        L = 450.0 if near else 380.0
        shape = lambda s: (1 - s) ** 0.75 * (0.62 + 0.38 * np.sin(np.pi * np.clip(s * 2.2, 0, 1) * 0.5)) + 0.07
        g = _fin_geom((rx, ry), ang, curl, L, lambda s: 46 * shape(s), lambda s: 36 * shape(s) * (1 - 0.3 * s),
                      bumps=5.0, phase=0.0)
        return g

    geo['fin_near'] = fin('near')
    geo['fin_far'] = fin('far')
    for key, (u0, u1) in (('fin_near', (0.25, 0.6)), ('fin_far', (0.25, 0.6))):
        cx, cy, fnx, fny, wl, wt = geo[key]
        geo[key + '_mesh'] = _strip_mesh(cx, cy, fnx, fny, wl, wt, u0, u1, rows=5)

    # --- flukes (two swept lobes, pitching with the stroke) ---
    ex, ey = float(X[-1]), float(Y[-1])
    tang = math.atan2(ty[-1], tx[-1])
    pitch = 0.32 * math.cos(TAU * (swim - 0.75) - 0.9)
    fl = 1.0 + 0.14 * math.sin(TAU * swim - 0.5)
    lead = lambda s: 44 * (1 - s) ** 0.55 * (0.7 + 0.3 * np.sin(np.pi * np.clip(s * 1.6, 0, 1))) + 2.5
    trail = lambda s: 26 * (1 - s) ** 1.6 + 2.0
    lobes = []
    for sg in (-1, 1):   # -1 upper lobe, +1 lower lobe
        L = 215.0 * (fl if sg < 0 else 2.0 - fl)
        ang = tang + pitch - sg * 1.02
        curl = sg * 0.8
        root = (ex + 12 * math.cos(tang), ey + 12 * math.sin(tang))
        if sg < 0:
            g = _fin_geom(root, ang, curl, L, trail, lead, n=16)
        else:
            g = _fin_geom(root, ang, curl, L, lead, trail, n=16)
        lobes.append(g)
    geo['flukes'] = lobes
    geo['fluke_mesh'] = [_strip_mesh(*g, 0.85, 1.0, rows=5) for g in lobes]
    geo['tail_end'] = (ex, ey)
    return geo


def _body_pt(geo, uq, vq):
    """Point on the body: vq = -1 top edge, 0 spine, +1 bottom edge."""
    u = geo['u']
    X = np.interp(uq, u, geo['X'])
    Y = np.interp(uq, u, geo['Y'])
    nx = np.interp(uq, u, geo['nx'])
    ny = np.interp(uq, u, geo['ny'])
    top = np.interp(uq, u, geo['top'])
    bot = np.interp(uq, u, geo['bot'])
    off = -vq * top if vq < 0 else -vq * bot
    return (float(X + nx * off), float(Y + ny * off))


def _fin_pt(g, s, side):
    cx, cy, nx, ny, wl, wt = g
    n = len(cx)
    fi = s * (n - 1)
    i = int(min(n - 2, math.floor(fi)))
    f = fi - i
    x = cx[i] * (1 - f) + cx[i + 1] * f
    y = cy[i] * (1 - f) + cy[i + 1] * f
    nxx = nx[i] * (1 - f) + nx[i + 1] * f
    nyy = ny[i] * (1 - f) + ny[i + 1] * f
    if side == 0:
        return (float(x), float(y))
    w = (wl[i] * (1 - f) + wl[i + 1] * f) if side > 0 else -(wt[i] * (1 - f) + wt[i + 1] * f)
    return (float(x + nxx * w * 0.8), float(y + nyy * w * 0.8))


_EYE_U = 0.25
_EYE_V = -0.2
_MOUTH_U = 0.225


def _mouth_line(geo, mouth, lower=False, n=16):
    pts = []
    u = geo['u']
    for i in range(n):
        f = i / (n - 1)
        uq = 0.003 + (_MOUTH_U - 0.003) * f
        # from the snout tip the lips arc down along the jaw, then curl up into a smile below the eye
        vq = 0.07 + 0.3 * math.sin(math.pi * f * 0.86) ** 0.8 - 0.07 * smoothstep(0.7, 1.0, f)
        x, y = _body_pt(geo, uq, vq)
        if lower:
            gap = mouth * 18.0 * math.sin(math.pi * min(1.0, f * 1.08)) ** 0.7
            nx = float(np.interp(uq, u, geo['nx']))
            ny = float(np.interp(uq, u, geo['ny']))
            x, y = x - nx * gap, y - ny * gap
        pts.append((x, y))
    return pts


def _node_list(geo):
    """(x, y, size, importance) for every constellation star + line index pairs."""
    nodes = []
    idx = {}

    def add(name, p, size=1.0):
        idx[name] = len(nodes)
        nodes.append((p[0], p[1], size))

    add('snout', _body_pt(geo, 0.004, 0.0), 1.1)
    for i, uq in enumerate((0.075, 0.17, 0.30, 0.44, 0.57, 0.655, 0.77, 0.88)):
        add(f'd{i}', _body_pt(geo, uq, -0.9 if i != 5 else -0.98), 1.2 if i in (2, 5) else 0.9)
    for i, uq in enumerate((0.1, 0.22, 0.345, 0.47, 0.6, 0.72, 0.85)):
        add(f'b{i}', _body_pt(geo, uq, 0.88), 1.0 if i in (2,) else 0.85)
    add('mc', _mouth_line(geo, 0.0)[-1], 0.8)
    add('blow', _body_pt(geo, 0.12, -0.98), 0.7)
    for i, uq in enumerate((0.36, 0.53, 0.71)):
        add(f's{i}', _body_pt(geo, uq, -0.05 + 0.1 * (i % 2)), 0.75)
    ex, ey = geo['tail_end']
    add('tail', (ex, ey), 1.0)
    fn = geo['fin_near']
    add('fr0', _fin_pt(fn, 0.02, 1), 0.8)
    add('fr1', _fin_pt(fn, 0.45, 1), 0.9)
    add('ft', _fin_pt(fn, 1.0, 0), 1.2)
    add('fr2', _fin_pt(fn, 0.55, -1), 0.8)
    ff = geo['fin_far']
    add('gg0', _fin_pt(ff, 0.5, 0), 0.6)
    add('gg1', _fin_pt(ff, 1.0, 0), 0.8)
    up, dn = geo['flukes']
    add('ut', _fin_pt(up, 1.0, 0), 1.2)
    add('um', _fin_pt(up, 0.5, 1), 0.8)
    add('dt', _fin_pt(dn, 1.0, 0), 1.2)
    add('dm', _fin_pt(dn, 0.5, -1), 0.8)
    lines = []
    faint = []

    def chain(names, dst=lines):
        for a_, b_ in zip(names, names[1:]):
            dst.append((idx[a_], idx[b_]))

    chain(['snout'] + [f'd{i}' for i in range(8)] + ['tail'])
    chain(['snout'] + [f'b{i}' for i in range(7)] + ['tail'])
    chain(['b2', 'fr0', 'fr1', 'ft', 'fr2', 'b3'])
    chain(['tail', 'um', 'ut'])
    chain(['tail', 'dm', 'dt'])
    chain(['s0', 's1', 's2', 'tail'], faint)
    chain(['gg0', 'gg1'], faint)
    lines = (lines, faint)
    return nodes, lines, idx


def _whale_world(x, y, scale, facing, rot):
    m = skia.Matrix()
    m.setTranslate(x, y)
    if rot:
        m.preRotate(math.degrees(rot))
    m.preScale(scale * facing, scale)
    return m


def draw_whale(c, t, x, y, scale=1.0, alpha=1.0, facing=1.0, swim=0.0, mouth=0.0, eye=1.0, rot=0.0,
               dissolve=0.0, glow=1.0, solid=0.22, dust=1.0, swim_rate=0.5, look=0.0):
    """The constellation whale (Guang's mother) made of stars & light lines, huge and gentle.
    (x, y) = body centre. scale=1 → ~1400 units long. facing=+1 swims to screen-right, -1 left.
    swim: undulation phase in cycles (e.g. t*0.5 -> one tail beat every 2 s).
    mouth: 0..1 (for her one line of dialogue). eye: 1 open .. 0 closed (blink).
    dissolve: 0..1 body fades and the star nodes scatter into sparkles.
    glow: brightness multiplier. solid: 0..1 normal-blend dark body underlayer that hides the
    background stars behind her (keeps her readable over the Milky Way). dust: stardust amount.
    swim_rate: d(swim)/dt estimate, used so the stardust trail follows past tail positions.
    Returns dict(eye, mouth, tail, belly, head_top, snout, fin_tip, center) in world coords
    (head_top is on her back above the eye: put Guang ~RADIUS*scale above it)."""
    M = _whale_world(x, y, scale, facing, rot)
    mouth = clamp(mouth)
    d = clamp(dissolve)
    geo = _whale_geometry(swim, mouth)

    def Wp(p):
        q = M.mapXY(p[0], p[1])
        return (q.x(), q.y())

    eye_p = _body_pt(geo, _EYE_U, _EYE_V)
    ml = _mouth_line(geo, 0.0)
    anchors = dict(eye=Wp(eye_p), mouth=Wp(ml[len(ml) // 2]), tail=Wp(geo['tail_end']),
                   belly=Wp(_body_pt(geo, 0.42, 1.0)), head_top=Wp(_body_pt(geo, 0.2, -1.0)),
                   snout=Wp(_body_pt(geo, 0.0, 0.0)), fin_tip=Wp(_fin_pt(geo['fin_near'], 1.0, 0)),
                   center=(x, y))
    if alpha <= 0.003 or scale <= 0:
        return anchors

    c.save()
    c.concat(M)
    ctm = _ctm_scale(c)
    px = 1.0 / max(1e-6, ctm)          # one device pixel in local units
    A = clamp(alpha) * max(0.0, glow)
    body_a = clamp(alpha) * (1 - smoothstep(0.0, 0.75, d))
    line_a = A * (1 - smoothstep(0.0, 0.45, d))

    # silhouette (union of body, fins, flukes)
    body_pos = geo['body'][0]
    sil = skia.Path()
    sil.setFillType(skia.PathFillType.kWinding)
    sil.addPath(_outline_path(body_pos))
    fin_paths = []
    for key in ('fin_near_mesh',):
        fin_paths.append(_outline_path(geo[key][0]))
    for fm in geo['fluke_mesh']:
        fin_paths.append(_outline_path(fm[0]))
    far_path = _outline_path(geo['fin_far_mesh'][0])
    sil_all = skia.Path(sil)
    for fp in fin_paths:
        sil_all = skia.Op(sil_all, fp, skia.PathOp.kUnion_PathOp) or sil_all

    # --- 1. dark underlayer (normal blend) + outer aura ---
    if body_a > 0.003:
        if solid > 0:
            c.drawPath(sil_all, fill('#0a1238', clamp(solid) * body_a))
            c.drawPath(far_path, fill('#0a1238', clamp(solid) * body_a * 0.6))
        aura = 0.22 * body_a * glow * (1 - smoothstep(1.3, 2.1, ctm))   # skip the costly blur in close-ups
        if aura > 0.004:
            c.drawPath(sil_all, fill('#3f6fe0', aura, blur=48.0, blend=ADD))

    # --- 2. nebula body (texture meshes, additive) ---
    if body_a > 0.003:
        sh = _tex_shader()
        # NB: Paint(Shader=...) — Paint.setShader() with an image shader is ~70 ms in skia-python
        mp = skia.Paint(AntiAlias=True, Shader=sh, BlendMode=ADD)
        # far fin, dimmer
        mp.setAlphaf(clamp(0.45 * body_a * glow))
        c.drawVertices(_mesh_vertices(*geo['fin_far_mesh'], fade=0.3), mp, skia.BlendMode.kModulate)
        mp.setAlphaf(clamp(0.85 * body_a * glow))
        c.drawVertices(_mesh_vertices(*geo['body']), mp)
        for fm in geo['fluke_mesh']:
            c.drawVertices(_mesh_vertices(*fm, fade=0.12), mp, skia.BlendMode.kModulate)
        c.drawVertices(_mesh_vertices(*geo['fin_near_mesh'], fade=0.22), mp, skia.BlendMode.kModulate)
        # glowing edges
        gl_ = stroke('#7fc4ff', 9.0, 0.1 * body_a * glow)
        gl_.setBlendMode(ADD)
        c.drawPath(sil_all, gl_)
        ep = stroke(W_LINE, max(1.4, 0.8 * px), 0.35 * body_a * glow)
        ep.setBlendMode(ADD)
        c.drawPath(sil_all, ep)
        fe = stroke(W_LINE, max(1.2, 0.7 * px), 0.18 * body_a * glow)
        fe.setBlendMode(ADD)
        c.save()
        c.clipPath(sil_all, skia.ClipOp.kDifference, doAntiAlias=True)
        c.drawPath(far_path, fe)
        c.restore()
        # throat grooves (ventral pleats)
        gp = skia.Path()
        for k in range(5):
            vq = 0.55 + 0.1 * k
            pts = [_body_pt(geo, uq, vq) for uq in np.linspace(0.07 + 0.02 * k, 0.42 - 0.02 * k, 10)]
            gp.addPoly([skia.Point(*p) for p in pts], False)
        gs = stroke('#a8d8ff', max(1.3, 0.7 * px), 0.13 * body_a * glow)
        gs.setBlendMode(ADD)
        c.drawPath(gp, gs)
        # twinkling stars inside the body
        _whale_inner_stars(c, t, geo, body_a * glow, px)

    # --- 3. constellation lines & nodes ---
    nodes, lines, nidx = _node_list(geo)
    if d > 0:
        nodes = _scatter_nodes(nodes, d)
    if line_a > 0.003:
        main, faint = lines
        for group, k in ((main, 1.0), (faint, 0.45)):
            lp = skia.Path()
            for a_, b_ in group:
                lp.moveTo(nodes[a_][0], nodes[a_][1])
                lp.lineTo(nodes[b_][0], nodes[b_][1])
            wide = stroke(W_LINE, max(7.0, 3.0 * px), 0.1 * line_a * k)
            wide.setBlendMode(ADD)
            c.drawPath(lp, wide)
            core = stroke('#cfeeff', max(1.8, 0.9 * px), 0.65 * line_a * k)
            core.setBlendMode(ADD)
            c.drawPath(lp, core)
        # mouth line (lips)
        up_ = _mouth_line(geo, mouth)
        mp_ = skia.Path()
        mp_.addPoly([skia.Point(*p) for p in up_], False)
        if mouth > 0.02:
            lo_ = _mouth_line(geo, mouth, lower=True)
            gap = skia.Path()
            gap.addPoly([skia.Point(*p) for p in up_ + lo_[::-1]], True)
            c.drawPath(gap, fill('#050a24', 0.55 * body_a))
            c.drawPath(gap, fill('#6f8fff', 0.18 * body_a, blend=ADD, blur=4))
            mp_.addPoly([skia.Point(*p) for p in lo_], False)
        ms = stroke('#dff4ff', max(2.2, 0.9 * px), 0.75 * line_a)
        ms.setBlendMode(ADD)
        c.drawPath(mp_, stroke(W_LINE, max(8, 3 * px), 0.12 * line_a))
        c.drawPath(mp_, ms)
    if A > 0.003:
        _whale_nodes(c, t, nodes, A, d, px)

    # --- 4. the eye ---
    if A * (1 - d) > 0.003:
        _whale_eye(c, t, eye_p, clamp(eye), A * (1 - smoothstep(0.2, 0.8, d)), px, facing, look)

    # --- 5. stardust from the tail ---
    if dust > 0:
        _whale_dust(c, t, geo, swim, swim_rate, A * dust, px)
    if d > 0:
        _whale_dissolve_sparkles(c, t, geo, d, clamp(alpha) * max(0.0, glow), px)
    c.restore()
    return anchors


def _whale_inner_stars(c, t, geo, a, px):
    n = 34
    for i in range(n):
        uq = 0.04 + 0.9 * _hh(i, 201)
        vq = -0.85 + 1.7 * _hh(i, 202)
        x, y = _body_pt(geo, uq, vq)
        tw = 0.5 + 0.5 * math.sin(t * (1.1 + 1.7 * _hh(i, 203)) + TAU * _hh(i, 204))
        r = (1.4 + 1.8 * _hh(i, 205))
        r = max(r, 0.8 * px)
        c.drawCircle(x, y, r, fill('#eaf6ff', a * (0.25 + 0.6 * tw), blend=ADD))
        if _hh(i, 206) > 0.75:
            glint(c, x, y, r * 4.5 * (0.5 + tw), '#cfe9ff', a * tw * 0.6, core=False)


def _scatter_nodes(nodes, d):
    out = []
    k = d ** 1.3
    for i, (x, y, s) in enumerate(nodes):
        ang = TAU * _hh(i, 301)
        dist = (180 + 420 * _hh(i, 302)) * k
        out.append((x + math.cos(ang) * dist, y + math.sin(ang) * dist * 0.6 - 220 * k * _hh(i, 303), s))
    return out


def _whale_nodes(c, t, nodes, a, d, px):
    fade = 1 - smoothstep(0.62, 1.0, d)
    flash = 1 + 0.9 * math.sin(math.pi * clamp(d * 1.4))
    for i, (x, y, s) in enumerate(nodes):
        tw = 0.8 + 0.2 * math.sin(t * (1.3 + 2.1 * _hh(i, 101)) + TAU * _hh(i, 102))
        aa = a * tw * fade * (flash if d > 0 else 1.0)
        if aa <= 0.003:
            continue
        rg = 22 * s
        _dot(c, x, y, rg, '#bfe4ff', 0.55 * aa)
        rc = max(3.2 * s, 1.1 * px)
        c.drawCircle(x, y, rc, fill('#ffffff', min(1, aa), blend=ADD))
        # occasional glints
        gl = smoothstep(0.55, 1.0, math.sin(t * (0.7 + 0.9 * _hh(i, 103)) + TAU * _hh(i, 104)))
        if s >= 1.0:
            gl = max(gl, 0.35)
        if gl > 0.01:
            glint(c, x, y, max(16 * s, 5 * px) * (0.6 + 0.6 * gl), '#e8f6ff', aa * gl, core=False)


def _whale_eye(c, t, p, op, a, px, facing, look):
    """Bright, kind star-eye: glowing orb, deep blue pupil with highlights, a gently lowered
    upper lid and a smiling lower lid. op = openness (blink)."""
    x, y = p
    r = 21.0
    _dot(c, x, y, r * 5.5, '#8fd0ff', 0.32 * a)
    # soft dark socket so the eye reads as an eye
    c.drawOval(skia.Rect.MakeXYWH(x - r * 2.1, y - r * 1.7, r * 4.2, r * 3.4), fill('#0a1238', 0.42 * a, blur=r * 0.5))
    # upper lid curve (front corner -> back corner); it lowers as the eye closes
    drop = (1 - op) * r * 1.35
    f0 = (x + r * 1.2, y - r * 0.05)
    b0 = (x - r * 1.25, y + r * 0.1)
    cp1 = (x + r * 0.7, y - r * 1.22 + drop)
    cp2 = (x - r * 0.75, y - r * 1.3 + drop)
    lid = skia.Path()
    lid.moveTo(*f0)
    lid.cubicTo(*cp1, *cp2, *b0)
    if op > 0.1:
        # orb clipped under the lid
        clip = skia.Path(lid)
        clip.lineTo(x - r * 1.4, y + r * 1.4)
        clip.lineTo(x + r * 1.4, y + r * 1.4)
        clip.close()
        c.save()
        c.clipPath(clip, doAntiAlias=True)
        oy = y + r * 0.08
        orb = radial((x + r * 0.15, oy - r * 0.3), r * 1.25,
                     [(0, '#ffffff', a), (0.4, '#e2f5ff', a), (0.78, '#9fd4ff', a), (1, '#5b8fe6', a)])
        orb.setAntiAlias(True)
        c.drawCircle(x, oy, r, orb)
        lx = x + look * 3.5 + r * 0.1
        pr = r * 0.6
        c.drawCircle(lx, oy + r * 0.08, pr, fill('#274aa8', 0.9 * a))
        c.drawCircle(lx, oy + r * 0.08, pr * 0.6, fill('#0e1a52', 0.95 * a))
        c.drawCircle(lx + pr * 0.38, oy - pr * 0.18, pr * 0.34, fill('#ffffff', a))
        c.drawCircle(lx - pr * 0.4, oy + pr * 0.5, pr * 0.14, fill('#ffffff', 0.85 * a))
        # lid shadow on the orb
        c.drawPath(lid, stroke('#3a5fc0', r * 0.5, 0.35 * a, blur=r * 0.15))
        c.restore()
        ls = stroke('#d8f0ff', max(r * 0.17, px), 0.9 * a)
        ls.setBlendMode(ADD)
        c.drawPath(lid, ls)
        low = skia.Path()
        low.moveTo(x - r * 0.95, y + r * 1.05)
        low.quadTo(x + r * 0.1, y + r * 1.5, x + r * 1.1, y + r * 0.8)
        lo = stroke('#9fd8ff', max(r * 0.1, 0.8 * px), 0.5 * a)
        lo.setBlendMode(ADD)
        c.drawPath(low, lo)
        glint(c, x + r * 0.4, y - r * 0.2, r * (1.3 + 0.15 * math.sin(t * 1.7)), '#eaf8ff', 0.45 * a * op, core=False)
    else:
        cl = skia.Path()
        cl.moveTo(*f0)
        cl.quadTo(x, y + r * 0.8, b0[0], b0[1])
        cs = stroke('#dff4ff', max(r * 0.18, px), 0.9 * a)
        cs.setBlendMode(ADD)
        c.drawPath(cl, cs)


def _whale_dust(c, t, geo, swim, swim_rate, a, px):
    n = 56
    ex, ey = geo['tail_end']
    for i in range(n):
        per = 2.2 + 1.6 * _hh(i, 401)
        ph = t / per + _hh(i, 402)
        cyc = math.floor(ph)
        age = ph - cyc
        j = cyc * 17 + i
        # tail position at birth (so the trail follows the tail's past strokes)
        born = swim - age * per * swim_rate
        _, Yb = _spine(np.array([1.0]), born)
        spread = (_hh(j, 403) - 0.5) * 2
        sx = ex - 60 - 120 * _hh(j, 404)
        sy = float(Yb[0]) + spread * 150 * (0.3 + 0.7 * _hh(j, 405))
        drift = age * (260 + 240 * _hh(j, 406))
        x = sx - drift
        y = sy + age * 60 * (_hh(j, 407) - 0.3) + 12 * math.sin(t * 1.3 + i)
        aa = a * math.sin(math.pi * min(1.0, age * 1.3)) ** 1.2 * (1 - age) * \
            (0.6 + 0.4 * math.sin(t * 7 + i * 1.3))
        if aa <= 0.004:
            continue
        r = (2.0 + 3.5 * _hh(j, 408)) * (1 - 0.5 * age)
        colr = '#dff0ff' if _hh(j, 409) > 0.4 else '#e6d4ff'
        if i % 4 == 0:
            glint(c, x, y, max(r * 4, 3 * px), colr, aa)
        else:
            _dot(c, x, y, r * 3.2, colr, aa * 0.6)
            c.drawCircle(x, y, max(r * 0.45, 0.7 * px), fill('#ffffff', aa, blend=ADD))


def _whale_dissolve_sparkles(c, t, geo, d, a, px):
    n = 110
    k = d ** 1.2
    env = math.sin(math.pi * clamp(d * 1.15)) ** 0.7
    for i in range(n):
        uq = 0.02 + 0.95 * _hh(i, 501)
        vq = -0.9 + 1.8 * _hh(i, 502)
        x, y = _body_pt(geo, uq, vq)
        ang = TAU * _hh(i, 503)
        dist = (80 + 380 * _hh(i, 504)) * k
        x += math.cos(ang) * dist
        y += math.sin(ang) * dist * 0.7 - 260 * k * (0.4 + _hh(i, 505))
        tw = 0.55 + 0.45 * math.sin(t * (5 + 4 * _hh(i, 506)) + i)
        aa = a * env * tw
        if aa <= 0.004:
            continue
        colr = mix('#dff0ff', '#fff1c4', _hh(i, 507))
        r = 2.0 + 3.0 * _hh(i, 508)
        if i % 3 == 0:
            glint(c, x, y, max(r * 4.5, 3 * px), colr, aa)
        else:
            _dot(c, x, y, r * 3, colr, aa * 0.7)
            c.drawCircle(x, y, max(0.7 * px, r * 0.45), fill('#ffffff', aa, blend=ADD))


# ============================================================================
# the star children (welcome party)
# ============================================================================
_FAMILY_COLORS = ['#ffb8d6', '#a8f0d0', '#a8d8ff', '#d4bcff', '#ffcfa3', '#fff3a0', '#9ff0f0', '#ffc2c2']


def draw_star_family(c, t, cx, cy, spread=600, alpha=1.0, n=7, seed=5, scale=0.34, wave=1.0):
    """Other little star children twinkling/waving high in the sky (welcome party).
    Arranged in a gentle arc around (cx, cy); each bobs, waves, blinks and twinkles."""
    if alpha <= 0.003:
        return
    for i in range(n):
        h1, h2, h3, h4 = (_hh(i, seed * 10 + k) for k in range(4))
        f = (i + 0.5) / n
        ang = math.pi * (1.08 + 0.84 * f)            # upper arc
        rx = cx + math.cos(ang) * spread + (h1 - 0.5) * spread * 0.15
        ry = cy + math.sin(ang) * spread * 0.38 + (h2 - 0.5) * spread * 0.1
        ry += 9 * math.sin(t * (1.2 + 0.5 * h3) + TAU * h4)
        rx += 5 * math.sin(t * 0.7 + TAU * h1)
        sc = scale * (0.75 + 0.5 * h3)
        # wave: one arm waves while the other sways
        wv = wave * (0.7 + 0.5 * math.sin(t * (5.0 + 2 * h2) + TAU * h1))
        side = h4 > 0.5
        bl_ph = _frac(t / (2.5 + 2 * h1) + h2)
        blink = 1.0 if bl_ph < 0.05 else 0.0
        pose = StarPose(x=rx, y=ry, scale=sc, rot=0.12 * math.sin(t * 0.9 + TAU * h3),
                        glow=1.0 + 0.3 * math.sin(t * 2 + TAU * h2), color=_FAMILY_COLORS[(i + seed) % len(_FAMILY_COLORS)],
                        eyes='happy' if h3 > 0.6 else 'open', blink=blink, smile=0.7, blush=0.7,
                        arm_r=wv if side else 0.15, arm_l=0.15 if side else wv,
                        halo=0.9, sparkle=0.7, alpha=alpha, mouth=0.25 if h2 > 0.7 else 0.0)
        draw_star(c, pose, t + 10 * h1)
