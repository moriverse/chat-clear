"""s01 · 开场 Opening (24 s) — owner: SCENE-BOOKENDS.

1. 0–6.5   black → stars fade in; the title 一盏灯，一颗星 assembles from drifting star
           particles, then dissolves back into stardust that rises into the Milky Way.
2. 6–13.5  slow tilt down (sky/stars at parallax depth 0.4, sea/horizon at 1.0) until
           the horizon of a calm night sea sits at 2/3 frame height; distant meteors.
3. 13.5–20.2 dolly towards the tiny lighthouse island, beam sweeping, reflections.
4. 20.2–24 closer: Deng on the gallery, heart glowing, antenna blinking (N03), then
           push into the lamp-room glow for the crossfade into s02 (render works past 24 s).
"""
from __future__ import annotations

import math

import skia

from engine.core import (Camera, H, W, clamp, col, ease_in_out, fill, lerp, smoothstep, soft_glow)
from lib import fx, robot, sea, sky

from scenes import _title
from scenes._s01_world import draw_lighthouse_group, log_spline, spline

HZ = 720.0                 # final horizon (world == screen once the tilt is done)
HZ_START = 2280.0          # horizon far below the frame at the start (looking up)
D_SKY = 0.4                # parallax depth of stars / Milky Way / title
TITLE_Y = 400.0            # title centre on screen at the start
# Milky Way in final (sky-layer) coords: rises steeply from the horizon just behind the
# lighthouse, and crosses the right of the frame beside the title at the start.
MW = dict(cx=1560.0, cy=-330.0, angle=-1.05, width=700.0, length=6200.0, bend=0.05)

# The sweep follows the angle = T*0.9 convention with a constant phase offset (s01 only
# crossfades into an interior, so nothing to match): the beam faces the camera at ~17.3 s
# (wide shot) and ~24.3 s (into the lamp glow), and is turned away while we meet Deng.
BEAM_OFFSET = 4.83

_PT = None


def _title_sys():
    global _PT
    if _PT is None:
        pt = _title.title_main('s01')
        sky_off = (HZ_START - HZ) * D_SKY
        ty = TITLE_Y - sky_off                      # title centre in sky-layer coords
        pt.set_dest_band(MW['cx'] - 960, MW['cy'] - ty, MW['angle'], MW['width'], lift=240)
        _PT = (pt, ty)
    return _PT


def _horizon(f):
    t = f.t
    tilt = ease_in_out(f.between('tilt_start', 'tilt_end'))
    h = lerp(HZ_START, HZ, tilt)
    te = f.cue('tilt_end')
    if t > te:
        h = spline(t, [(te, HZ), (f.cue('N02_end'), 716.0), (f.cue('N03_end'), 700.0), (te + 12.0, 690.0)])
    return h


def _lighthouse_frame(f, hz):
    """Returns (x_b, y_b, s): lighthouse base on screen and its scale (dolly model)."""
    t = f.t
    te = f.cue('tilt_end')
    t3 = f.cue('N02_end')          # ~20.2  "push closer"
    t4 = f.cue('N03_end')          # ~23.2
    tend = f.dur + 1.6
    s0 = 0.16
    if t <= te:
        return 1250.0, hz + 20.0, s0
    n3 = f.cue('N03_start')        # ~20.8
    s = log_spline(t, [(te, s0), (te + 3.5, 0.42), (t3, 1.3), (n3 + 1.6, 5.6), (t4 + 0.2, 6.6), (tend, 14.0)])
    F = spline(t, [(te, (0.0, -300.0)), (t3, (0.0, -390.0)), (n3 + 1.6, (-19.0, -421.0)),
                   (t4 + 0.2, (-18.0, -423.0)), (tend, (0.0, -448.0))])
    P = spline(t, [(te, (1250.0, hz + 20.0 - 300.0 * s0)), (t3, (1060.0, 470.0)),
                   (n3 + 1.6, (930.0, 560.0)), (t4 + 0.2, (925.0, 555.0)), (tend, (960.0, 520.0))])
    return P[0] - F[0] * s, P[1] - F[1] * s, s


def _deng_pose(f):
    t = f.t
    n3 = f.cue('N03_start')

    def pose_fn(p, ctx):
        idl = robot.idle(f.T, seed=1)
        p.facing = -1.0
        p.turn = 0.1
        p.head_dy = idl['head_dy']
        p.lean = idl['lean'] + 0.03
        # looks out over the empty sea, then (on "阿灯") lifts his head to the stars
        up = smoothstep(n3 + 1.0, n3 + 2.1, t)
        p.head_tilt = idl['head_tilt'] - 0.2 * up
        p.look = (0.45 * (1 - up) + 0.15 * up, -0.05 - 0.8 * up)
        p.eyes = 'open'
        p.eyes2 = 'wonder'
        p.eyes_mix = 0.6 * up
        # both mittens resting on the top rail
        ry = ctx['rail_y'] + 4 * p.scale
        p.arm_r_override = (p.x + 48 * p.scale, ry)
        p.arm_l_override = (p.x - 36 * p.scale, ry - 2 * p.scale)
        p.hand_l_open = 0.3
        p.hand_r_open = 0.3
        p.blink = robot.auto_blink(f.T, seed=2)
        p.smile = 0.15 + 0.15 * up
        p.mouth = f.mouth('deng')
        p.scarf_wind = 0.65
        p.heart = 0.92 + 0.08 * math.sin(f.T * 2.2)
        p.antenna_flash = 0.8 * math.exp(-((t - n3 - 1.45) / 0.18) ** 2)
        return p
    return pose_fn


def render(f):
    c = f.canvas
    t = f.t
    T = f.T
    hz = _horizon(f)
    cam = Camera(x=W / 2, y=H / 2 - (hz - HZ), t=T)
    # subtle zoom on the sky layer during the push-in (stars barely scale)
    te = f.cue('tilt_end')
    push = clamp((t - te) / (f.dur + 1.5 - te))
    sky_cam = Camera(x=W / 2, y=H / 2 - (hz - HZ) * 1.0, zoom=1.0 + 0.12 * push * push, t=T)

    stars_in = smoothstep(f.cue('s01_stars_in'), f.cue('s01_stars_in') + 2.6, t)

    # ---------------------------------------------------------------- sky gradient
    with cam.apply(c, 1.0):
        sky.sky(c, horizon_y=HZ)
    if stars_in < 1:
        c.drawRect(skia.Rect.MakeWH(W, H), fill('#000000', 1 - stars_in))

    # ---------------------------------------------------------------- far layer
    hz_sky = HZ + (hz - HZ) * (1 - D_SKY)            # horizon in sky-layer coords
    with sky_cam.apply(c, D_SKY):
        mw_a = 0.6 * smoothstep(0.8, 4.5, t)
        sky.milky_way(c, T, center=(MW['cx'], MW['cy']), angle=MW['angle'], length=MW['length'],
                      width=MW['width'], alpha=mw_a, bend=MW['bend'], horizon_y=hz_sky)
        sky.stars(c, T, bright=stars_in, horizon_y=hz_sky)
        # distant meteors during the tilt
        for (tm, dur, x0, y0, x1, y1) in ((8.3, 0.75, 1560, -330, 1180, -120),
                                         (11.5, 0.6, 520, 170, 250, 320)):
            p = (t - tm) / dur
            if 0 <= p <= 1:
                fx.meteor(c, x0, y0, x1, y1, p, tail=240, size=0.32, t=T)
        # the title
        if t < 16:
            pt, ty = _title_sys()
            _title.draw_title(c, pt, t, 960.0, ty, t_in=f.cue('title_in'), t_out=f.cue('title_out'),
                              rest_alpha=stars_in)

    # ---------------------------------------------------------------- sea + lighthouse
    if hz < H + 60:
        x_b, y_b, s = _lighthouse_frame(f, hz)
        lamp_y = y_b - 445 * s
        with cam.apply(c, 1.0):
            off = hz - HZ
            lights = [(x_b, lamp_y - off, '#ffd36b', 0.9, y_b - off)]
            sea.ocean(c, T, horizon_y=HZ, lights=lights)
        # the lighthouse is drawn in screen space (dolly model)
        c.save()
        draw_lighthouse_group(c, T, x_b, y_b, s, hz, pose_fn=_deng_pose(f), lamp=1.0, beam=1.0,
                              beam_phase=T * 0.9 + BEAM_OFFSET, deng_scale=0.155, deng_x=-0.3)
        c.restore()

    # ---------------------------------------------------------------- push into the lamp glow
    t4 = f.cue('N03_end')
    glow = smoothstep(t4 - 0.2, f.dur + 0.9, t)
    if glow > 0:
        soft_glow(c, W / 2, H / 2 - 40, 1300, '#ffc877', 0.45 * glow)
        c.drawRect(skia.Rect.MakeWH(W, H), fill('#3a2410', 0.35 * glow * glow))

    # ---------------------------------------------------------------- grade
    f.grade.update(
        bloom=1.15 + 0.2 * glow,
        vignette=0.5 + 0.1 * glow,
        grain=0.05,
    )
