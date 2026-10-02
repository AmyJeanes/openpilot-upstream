#!/usr/bin/env python3
"""Shows the road and wide camera streams side by side in a resizable window (watch3 lays out a comma device's screen),
outlining the part of each that the driving model takes in."""
import numpy as np
import pyray as rl

import openpilot.cereal.messaging as messaging
from openpilot.cereal.visionipc import VisionStreamType
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import MEDMODEL_INPUT_SIZE, SBIGMODEL_INPUT_SIZE, get_warp_matrix
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.onroad.cameraview import CameraView

CAM_W, CAM_H = 1928, 1208
ASPECT = CAM_W / CAM_H
OUTLINE = rl.Color(0, 255, 120, 255)


def model_outline(warp: np.ndarray, size: tuple[int, int]) -> np.ndarray:
  """The model frame's corners in camera pixels: modeld's warp maps model frame pixels to camera pixels."""
  w, h = size
  corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=np.float64).T
  cam = warp @ corners
  return (cam[:2] / cam[2]).T


def draw_outline(points: np.ndarray, rect: rl.Rectangle):
  sx, sy = rect.width / CAM_W, rect.height / CAM_H
  screen = [rl.Vector2(rect.x + x * sx, rect.y + y * sy) for x, y in points]
  for a, b in zip(screen, screen[1:] + screen[:1], strict=True):
    rl.draw_line_ex(a, b, 2, OUTLINE)


if __name__ == "__main__":
  gui_app._width = gui_app._scaled_width = 1920
  gui_app._height = gui_app._scaled_height = 602
  gui_app._scale = 1.0
  gui_app.init_window("road | wide")
  rl.set_window_state(rl.ConfigFlags.FLAG_WINDOW_RESIZABLE)
  road = CameraView("camerad", VisionStreamType.VISION_STREAM_NARROW_ROAD)
  wide = CameraView("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD)
  sm = messaging.SubMaster(['extrinsicsCalibration', 'deviceState', 'narrowRoadCameraState'])
  outlines = None
  for _ in gui_app.render():
    sm.update(0)
    if sm.updated['extrinsicsCalibration'] and sm.seen['deviceState'] and sm.seen['narrowRoadCameraState']:
      rpy = np.array(sm['extrinsicsCalibration'].rpyCalib)
      if len(rpy) == 3:
        dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['narrowRoadCameraState'].sensor))]
        # as modeld frames them: the road camera in the medium model frame, the wide one in the big model frame
        outlines = (model_outline(get_warp_matrix(rpy, dc.narrow_road.intrinsics, False), MEDMODEL_INPUT_SIZE),
                    model_outline(get_warp_matrix(rpy, dc.wide_road.intrinsics, True), SBIGMODEL_INPUT_SIZE))

    w, h = rl.get_screen_width(), rl.get_screen_height()
    # two views across if the window is wide, else stacked; each as large as fits at the cameras' aspect
    across = w / h > ASPECT
    vw = min(w / 2, h * ASPECT) if across else min(w, h / 2 * ASPECT)
    vh = vw / ASPECT
    x0, y0 = (w - (2 * vw if across else vw)) / 2, (h - (vh if across else 2 * vh)) / 2
    road_rect = rl.Rectangle(x0, y0, vw, vh)
    wide_rect = rl.Rectangle(x0 + vw, y0, vw, vh) if across else rl.Rectangle(x0, y0 + vh, vw, vh)
    road.render(road_rect)
    wide.render(wide_rect)
    if outlines is not None:
      draw_outline(outlines[0], road_rect)
      draw_outline(outlines[1], wide_rect)
