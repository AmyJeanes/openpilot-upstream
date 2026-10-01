#!/usr/bin/env python3
"""Shows the full road and wide camera frames the bridge sends openpilot; its UI only shows a crop of them.
Run with the same OPENPILOT_PREFIX as openpilot."""
import os

os.environ.setdefault("BIG", "1")  # the larger window size; must be set before the UI library is imported

import pyray as rl

from openpilot.cereal.visionipc import VisionStreamType
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.onroad.cameraview import CameraView


if __name__ == "__main__":
  gui_app.init_window("slowroads cameras")
  road = CameraView("camerad", VisionStreamType.VISION_STREAM_NARROW_ROAD)
  wide = CameraView("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD)
  for _ in gui_app.render():
    half = gui_app.width / 2
    road.render(rl.Rectangle(0, 0, half, gui_app.height))
    wide.render(rl.Rectangle(half, 0, half, gui_app.height))
