"""阿灯 (Deng) — the old lighthouse-keeper robot. Owner: ROBOT agent.

Usage in a scene:
    from lib.robot import RobotPose, draw_robot, auto_blink
    pose = RobotPose(x=900, y=820, scale=1.0, facing=1, mouth=f.mouth('deng'),
                     blink=auto_blink(f.T), arm_r=(-1.2, 0.4))
    hands = draw_robot(c, pose, t=f.T)
    hands['r']  # world (x, y) of right hand — put the star there

CONTRACT: keep RobotPose fields and draw_robot signature working (add fields with
defaults freely). STUB=True means placeholder art.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import skia

from engine.core import (ADD, at, col, fill, linear, noise1, poly, radial, rrect, soft_glow, stroke)

STUB = True
HEIGHT = 360  # design units, feet to antenna tip at scale=1


@dataclass
class RobotPose:
    x: float = 960.0          # world x of the point between the feet (ground contact)
    y: float = 900.0          # world y of the ground contact
    scale: float = 1.0        # 1.0 → ~360 units tall
    facing: float = 1.0       # +1 faces screen-right, -1 faces screen-left (mirror)
    turn: float = 0.0         # -1..1 fake 3/4 head/body turn (0 = front-ish 3/4 view as designed)
    lean: float = 0.0         # whole body tilt (radians, + = lean forward in facing direction)
    head_tilt: float = 0.0    # radians
    head_dy: float = 0.0      # extra head offset (bob), units
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
    blink: float = 0.0           # 0 open .. 1 closed (overrides)
    look: tuple = (0.0, 0.0)     # pupil offset -1..1
    mouth: float = 0.0           # lip-sync openness 0..1 (pass f.mouth('deng'))
    smile: float = 0.3           # -1 frown .. 1 big smile
    # lights / story state
    power: float = 1.0           # 0..1 overall electric life: eyes, antenna, screen (0 = dead)
    heart: float = 1.0           # 0..1 brightness of the heart core in the chest porthole
    heart_open: float = 0.0      # 0..1 chest porthole door swung open
    heart_present: bool = True   # False = empty chest (after giving the heart away)
    heart_star: float = 0.0      # 0..1 the NEW star-shaped heart (ending) replaces the old core
    scarf_wind: float = 0.5      # 0..1 wind strength on the red scarf
    antenna_blink: bool = True
    shake: float = 0.0           # tremble amplitude (effort / fear)


def auto_blink(T, seed=0, every=3.7):
    """Natural blink value 0..1 at time T (use for pose.blink)."""
    ph = (T + seed * 1.37) % every
    return max(0.0, 1 - abs(ph - 0.08) / 0.08) if ph < 0.16 else 0.0


def idle(T, seed=0):
    """Small breathing / servo idle offsets: returns dict(head_dy, lean, head_tilt)."""
    return dict(head_dy=math.sin(T * 1.6 + seed) * 2.0, lean=math.sin(T * 0.8 + seed) * 0.01,
                head_tilt=noise1(T * 0.4, 31 + seed) * 0.03)


def draw_robot(c, pose: RobotPose, t: float = 0.0):
    """Draw Deng. Returns dict with world positions:
    {'l': hand_l, 'r': hand_r, 'heart': heart_center, 'head': head_center, 'eyes': (x, y),
     'antenna': tip}. """
    s = pose.scale
    out = {}
    with at(c, pose.x, pose.y, sx=s * pose.facing, sy=s):
        c.rotate(math.degrees(pose.lean))
        # legs
        c.drawRRect(rrect(-50, -80, 36, 80, 12), fill('#2f5560'))
        c.drawRRect(rrect(14, -80, 36, 80, 12), fill('#2f5560'))
        # body
        c.drawPath(poly([(-75, -80), (75, -80), (60, -250), (-60, -250)]), fill('#4f8a9a'))
        c.drawRect(skia.Rect.MakeLTRB(-72, -150, 72, -128), fill('#b8413a'))
        c.drawCircle(0, -195, 32, fill('#c9a15b'))
        c.drawCircle(0, -195, 25, fill('#1a2030'))
        if pose.heart_present and pose.heart > 0:
            c.drawCircle(0, -195, 18, fill('#ffb347', pose.heart))
        # head
        c.drawRRect(rrect(-70, -350, 140, 100, 26), fill('#5b98a8'))
        c.drawRRect(rrect(-56, -338, 112, 76, 18), fill('#10202a'))
        e = 1 - pose.blink
        c.drawOval(skia.Rect.MakeXYWH(-36, -318, 22, 26 * e + 1), fill('#ffe9a8', pose.power))
        c.drawOval(skia.Rect.MakeXYWH(14, -318, 22, 26 * e + 1), fill('#ffe9a8', pose.power))
        c.drawRRect(rrect(-14, -282, 28, 4 + 12 * pose.mouth, 3), fill('#ffe9a8', pose.power))
        c.drawLine(0, -350, 0, -385, stroke('#c9a15b', 4))
        c.drawCircle(0, -388, 7, fill('#ff6b5b', pose.power))
        # arms (simple)
        for side, (sh, el), sx in (('l', pose.arm_l, -1), ('r', pose.arm_r, 1)):
            x0, y0 = 62 * sx if side == 'r' else -62, -225
            a1 = sh
            x1 = x0 + math.sin(a1) * 70
            y1 = y0 + math.cos(a1) * 70
            x2 = x1 + math.sin(a1 + el) * 70
            y2 = y1 + math.cos(a1 + el) * 70
            c.drawLine(x0, y0, x1, y1, stroke('#3b6e7e', 14))
            c.drawLine(x1, y1, x2, y2, stroke('#3b6e7e', 14))
            c.drawCircle(x2, y2, 14, fill('#c9a15b'))
            m = c.getTotalMatrix()
            out[side] = (x2, y2)
        tm = c.getTotalMatrix()
    # convert local points to world by re-applying the same transform maths
    def w(px, py):
        cs, sn = math.cos(pose.lean), math.sin(pose.lean)
        lx, ly = px * cs - py * sn, px * sn + py * cs
        return pose.x + lx * s * pose.facing, pose.y + ly * s
    res = {k: w(*v) for k, v in out.items()}
    res['heart'] = w(0, -195)
    res['head'] = w(0, -300)
    res['eyes'] = w(0, -305)
    res['antenna'] = w(0, -388)
    return res
