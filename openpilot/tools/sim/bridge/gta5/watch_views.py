#!/usr/bin/env python3
"""Shows the road and wide camera streams side by side in a resizable window (watch3 lays out a comma device's screen)."""
import pyray as rl

from openpilot.cereal.visionipc import VisionStreamType
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.onroad.cameraview import CameraView

ASPECT = 1928 / 1208

if __name__ == "__main__":
  gui_app._width = gui_app._scaled_width = 1920
  gui_app._height = gui_app._scaled_height = 602
  gui_app._scale = 1.0
  gui_app.init_window("road | wide")
  rl.set_window_state(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
  road = CameraView("camerad", VisionStreamType.VISION_STREAM_NARROW_ROAD)
  wide = CameraView("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD)
  for _ in gui_app.render():
    w, h = rl.get_screen_width(), rl.get_screen_height()
    # two views across if the window is wide, else stacked; each as large as fits at the cameras' aspect
    across = w / h > ASPECT
    vw = min(w / 2, h * ASPECT) if across else min(w, h / 2 * ASPECT)
    vh = vw / ASPECT
    x0, y0 = (w - (2 * vw if across else vw)) / 2, (h - (vh if across else 2 * vh)) / 2
    road.render(rl.Rectangle(x0, y0, vw, vh))
    wide.render(rl.Rectangle(x0 + vw, y0, vw, vh) if across else rl.Rectangle(x0, y0 + vh, vw, vh))
