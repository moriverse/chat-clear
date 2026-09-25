"""PLACEHOLDER scene — to be replaced by its scene agent."""
from engine.core import Camera, W, H, col, font, draw_text
import skia
from lib import sky, sea, robot, star


def render(f):
    c = f.canvas
    cam = Camera(t=f.T)
    with cam.apply(c, 1.0):
        sky.sky(c, horizon_y=700)
        sky.stars(c, f.T, horizon_y=700)
        sea.ocean(c, f.T, horizon_y=700)
        sea.island(c, 1300, 760, 0.6)
        lx, ly = sea.lighthouse(c, 1300, 760, 0.6)
        robot.draw_robot(c, robot.RobotPose(x=700, y=900, mouth=f.mouth('deng'), blink=robot.auto_blink(f.T)), f.T)
        star.draw_star(c, star.StarPose(x=900, y=600, mouth=f.mouth('guang')), f.T)
    draw_text(c, f"{f.scene['id']} placeholder  t={f.t:.1f}", W / 2, 120, font(40), skia.Paint(Color=col('#ffffff'), AntiAlias=True))
