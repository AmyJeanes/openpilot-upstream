import threading

import numpy as np
import pyray as rl

from openpilot.cereal import custom
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.reproject_c4 import calib_from_rotvec, debug_outlines
from openpilot.selfdrive.ui.onroad.alert_renderer import ALERT_HEIGHTS, ALERT_MARGIN
from openpilot.selfdrive.ui.onroad.hud_renderer import UI_CONFIG, COLORS
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.text_measure import measure_text_cached

FRAME_HEADINGS = {"device_narrow": "DEVICE NARROW CAM", "device_wide": "DEVICE WIDE CAM", "c4_narrow": "C4 NARROW CAM",
                  "c4_wide": "C4 WIDE CAM"}
GREEN, AMBER, RED = rl.Color(128, 216, 166, 255), rl.Color(255, 200, 80, 255), rl.Color(255, 90, 90, 255)
BLUE, PURPLE, GREY, WHITE = rl.Color(110, 180, 255, 255), rl.Color(215, 140, 255, 255), rl.Color(170, 170, 170, 255), rl.Color(240, 240, 240, 255)
OUTLINE_COLORS = {'amber': AMBER, 'green': GREEN, 'blue': BLUE, 'purple': PURPLE}


class ReprojectDebug:
  """The 3X->comma 4 reprojection's debug view (ShowReprojectionDebug: 0 none, 1 visual, 2 stats, 3 all), drawn over the
  road view: the geometry's outlines on the frame being shown, a legend of what is drawn, and the timings along the bottom."""
  def __init__(self):
    self._latency_ms = 0.0
    self._ui_ms = 50.0
    self._font_bold: rl.Font = gui_app.font(FontWeight.BOLD)
    self._font_medium: rl.Font = gui_app.font(FontWeight.MEDIUM)
    self._screen: tuple = (None, [])  # the outlines on screen, for one set of outlines, frame and transform
    self._outlines_key: tuple | None = None
    self._outlines_thread: threading.Thread | None = None
    self._outlines_checked = 0.0
    self._outline_sets: tuple = (0, {})  # (generation, debug_outlines result)

  def render(self, rect: rl.Rectangle, frame: str, image_to_screen: np.ndarray | None) -> None:
    """frame: the road view's frame (device/c4, narrow/wide); image_to_screen: its image px -> screen."""
    sm = ui_state.sm
    if sm.updated['modelV2'] and sm['modelV2'].timestampEof:
      # camera frame end to model output: the stage, the model and the handoffs between them
      self._latency_ms += 0.1 * ((sm.logMonoTime['modelV2'] - sm['modelV2'].timestampEof) * 1e-6 - self._latency_ms)
    self._ui_ms += 0.1 * (rl.get_frame_time() * 1e3 - self._ui_ms)
    mode = ui_state.reproject_debug
    if not mode or not sm.seen['reprojectCalibration']:
      return
    visual, stats = mode in (1, 3), mode in (2, 3)
    def grade(v, ok, warn):
      return GREEN if v < ok else AMBER if v < warn else RED

    legend = []
    if visual:
      self._update_outlines()
    if visual and image_to_screen is not None:
      for name, col, fill, outline in self._outlines(frame, image_to_screen):
        if outline is None:
          rl.draw_triangle_strip(fill, len(fill), rl.Color(col.r, col.g, col.b, 110))
        else:
          rl.draw_triangle_fan(fill, len(fill), rl.Color(col.r, col.g, col.b, 40))
          rl.draw_spline_linear(outline, len(outline), 4.0, col)
        legend.append((name, col))

    fit = sm['reprojectCalibration']
    fitted = fit.status == custom.ReprojectCalibration.Status.fitted
    rot = np.degrees(fit.applied) if len(fit.applied) == 3 else [0, 0, 0]
    tiles = [  # label, value, color
      ("LATENCY ms", f"{self._latency_ms:.0f}", grade(self._latency_ms, 52, 56)),
      ("UI ms", f"{self._ui_ms:.0f}", grade(self._ui_ms, 55, 75)),
      ("ROTATION", f"{rot[0]:+.2f}/{rot[1]:+.2f}" if fitted else "FITTING", GREEN if fitted else AMBER),
    ]
    big, small, pad, gap, th = 72, 26, 16, 12, 6
    def measure():
      return [int(max(measure_text_cached(self._font_bold, value, big).x, measure_text_cached(self._font_medium, label, small).x) + 2 * pad)
              for label, value, _ in tiles]
    widths = measure()
    avail = rect.width - 2 * 260  # the driver-monitoring icon takes a bottom corner (left- or right-hand drive)
    if sum(widths) + gap * (len(tiles) - 1) > avail:
      s = avail / (sum(widths) + gap * (len(tiles) - 1))
      big, small, pad = int(big * s), max(20, int(small * s)), int(pad * s)
      widths = measure()
    h = big + small + 2 * pad + th if stats else 0
    # centered along the bottom, stepping up above the alert bar (the calibration bar) when one is showing; the corners
    # keep the set-speed box, experimental button and driver-monitoring icon
    alert_size = sm['selfdriveState'].alertSize.raw if sm.recv_frame['selfdriveState'] >= ui_state.started_frame else 0
    above = ALERT_HEIGHTS[alert_size] - ALERT_MARGIN + gap if alert_size in ALERT_HEIGHTS else 0
    x = int(rect.x + (rect.width - (sum(widths) + gap * (len(tiles) - 1))) / 2)
    y = int(rect.y + rect.height - UI_CONFIG.border_size - h - above)
    for (label, value, col), w in zip(tiles if stats else [], widths, strict=False):
      rl.draw_rectangle(x, y, w, h, COLORS.BLACK_TRANSLUCENT)
      rl.draw_rectangle(x, y, w, th, col)
      rl.draw_text_ex(self._font_medium, label, rl.Vector2(x + pad, y + th + pad - 4), small, 0, GREY)
      rl.draw_text_ex(self._font_bold, value, rl.Vector2(x + pad, y + th + pad + small), big, 0, col)
      x += w + gap
    if not visual:
      return
    # the legend row above the tiles: the frame on screen, then a swatch and name per outline drawn
    lsize, lpad, lgap, sw = 30, 20, 40, 24
    items = [(FRAME_HEADINGS.get(frame, ""), None)] + legend
    def item_w(n, c):
      return measure_text_cached(self._font_bold if c is None else self._font_medium, n, lsize).x + (0 if c is None else sw + 20)
    tw = sum(item_w(n, c) for n, c in items) + lgap * (len(items) - 1) + 2 * lpad
    lh = lsize + 26
    lx = int(rect.x + (rect.width - tw) / 2)
    ly = y - (gap if stats else 0) - lh
    rl.draw_rectangle(lx, ly, int(tw), lh, COLORS.BLACK_TRANSLUCENT)
    cx = lx + lpad
    sy = ly + (lh - sw) / 2
    for n, c in items:
      font = self._font_bold if c is None else self._font_medium
      if c is not None:
        rl.draw_rectangle(int(cx), int(sy), sw, sw, rl.Color(c.r, c.g, c.b, 90))
        rl.draw_rectangle_lines_ex(rl.Rectangle(cx, sy, sw, sw), 3, c)
      ty = ly + (lh - measure_text_cached(font, n, lsize).y) / 2  # text and swatches centered on the same line
      rl.draw_text_ex(font, n, rl.Vector2(cx + (0 if c is None else sw + 20), ty), lsize, 0, WHITE)
      cx += item_w(n, c) + lgap

  def _update_outlines(self) -> None:
    """Recompute the outlines (debug_outlines) for the applied rotation and the model's calibration when either moves,
    checked once a second, in a thread: it is ~60 ms of numpy."""
    sm = ui_state.sm
    now = rl.get_time()
    if now - self._outlines_checked < 1.0 or (self._outlines_thread is not None and self._outlines_thread.is_alive()):
      return
    self._outlines_checked = now
    rotation = tuple(sm['reprojectCalibration'].applied)
    calib = sm['extrinsicsCalibration'].rpyCalib if sm.seen['extrinsicsCalibration'] else []
    rpy = tuple(calib) if len(calib) == 3 else (0.0, 0.0, 0.0)
    key = (tuple(np.round(rotation, 5)), tuple(np.round(rpy, 4)))
    if len(rotation) != 3 or key == self._outlines_key:
      return
    self._outlines_key = key
    def run():
      try:
        self._outline_sets = (self._outline_sets[0] + 1, debug_outlines(calib_from_rotvec(rotation), np.array(rpy, np.float32)))
      except Exception:
        cloudlog.exception("ui: reprojection debug outlines failed")
    self._outlines_thread = threading.Thread(target=run, daemon=True)
    self._outlines_thread.start()

  def _outlines(self, frame: str, T: np.ndarray) -> list:
    """This frame's outlines as (name, color, fill, outline) in screen px: a band's fill is one triangle strip around the
    ring (outline None), a polygon's a triangle fan from its center plus its closed outline. Transformed once per set of
    outlines, frame and transform, not per rendered frame."""
    generation, sets = self._outline_sets
    key = (generation, frame, T.tobytes())
    if key == self._screen[0]:
      return self._screen[1]
    def to_screen(xy):
      pts = np.asarray(xy, np.float32).reshape(-1, 2)
      scr = np.c_[pts, np.ones(len(pts), np.float32)] @ T.T
      return [rl.Vector2(x, y) for x, y in scr[:, :2].tolist()], pts
    out = []
    for it in sets.get(frame, []):
      if not it['visible'] or len(it['points']) < 2:
        continue
      col = OUTLINE_COLORS.get(it['color'], GREY)
      scr, pts = to_screen(it['points'])
      # raylib culls clockwise faces: the strip and the fan run the ring the other way round
      q = np.roll(pts, -1, 0)
      area = float((pts[:, 0] * q[:, 1] - q[:, 0] * pts[:, 1]).sum())
      if it.get('inner'):
        inner, _ = to_screen(it['inner'])
        if len(inner) != len(scr):
          continue
        order = range(len(scr)) if area > 0 else range(len(scr) - 1, -1, -1)
        strip = [v for i in order for v in (scr[i], inner[i])]
        strip += strip[:2]
        out.append((it['name'], col, strip, None))
      else:
        c = pts.mean(0)
        center, _ = to_screen(c)
        ring = scr[::-1] if area > 0 else scr
        out.append((it['name'], col, center + ring + ring[:1], scr + scr[:1]))
    self._screen = (key, out)
    return out
