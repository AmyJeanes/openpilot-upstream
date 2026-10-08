"""Navigation's speed cap on the onroad HUD: while navigation holds the car below the set speed (navSpeed: for a turn, a
bend, a lower limit ahead, a lane change or the arrival), a box under the MAX box shows the speed it allows, with an icon
for why, in navigation's blue. The MAX box keeps the driver's set speed, which stays the car's."""
import math
import time
from functools import lru_cache

import pyray as rl

from openpilot.common.constants import CV
from openpilot.selfdrive.ui.nav.draw import Shape, arc, clamp01, draw_maneuver, max_blend, premul, q, stroke_tris, with_alpha
from openpilot.selfdrive.ui.onroad.nav_panel import LIT_BLUE
from openpilot.selfdrive.ui.ui_state import UIStatus, ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

HEIGHT = 180
ICON = 76
FONT_SIZE = 80
LABEL_SIZE = 40  # the MAX box's label
BOX = rl.Color(0, 0, 0, 166)  # the MAX box's
FRESH_S = 1.0  # navSpeed older than this: navigation stopped sending
FADE_S = 0.25


def bend_paths() -> list[list[tuple[float, float]]]:
  """A curve ahead to the right, as the warning sign's arrow, drawn as the card's turn arrow: a stem bending away along
  a wide arc, and the same arrowhead."""
  cx, cy, r, end = 32.0, 28.0, 16.0, 260.0  # the arc from the stem's top, round to `end` deg (screen angles, y down)
  curve = [(cx - r, 42.0)] + arc(cx, cy, r, 180.0, end, 12)
  dx, dy = -math.sin(math.radians(end)), math.cos(math.radians(end))  # the way the arc ends
  tip = (curve[-1][0] + dx, curve[-1][1] + dy)
  wings = [(tip[0] - 7 * dx + 7 * sx * dy, tip[1] - 7 * dy - 7 * sx * dx) for sx in (1, -1)]
  return [curve, [wings[0], tip, wings[1]]]


def icon_paths(reason: str) -> list[list[tuple[float, float]]]:
  """The icons with no maneuver of the card's to borrow, as strokes in a 48-unit box (y down)."""
  if reason in ("bend", "bendLeft", "bendRight"):
    paths = bend_paths()
    return [[(48 - x, y) for x, y in p] for p in paths] if reason == "bendLeft" else paths
  # a lane change, for room to change lanes or into a turn's bay
  return [[(16, 44), (16, 34), (32, 18), (32, 10)], [(25, 15), (32, 8), (39, 15)], [(6, 8), (6, 16)], [(6, 26), (6, 34)],
          [(42, 26), (42, 34)], [(42, 40), (42, 46)]]


@lru_cache(maxsize=16)
def _icon(reason: str, cx: float, cy: float, size: float) -> Shape:
  s = size / 48
  paths = [[(cx - size / 2 + x * s, cy - size / 2 + y * s) for x, y in p] for p in icon_paths(reason)]
  return Shape(stroke_tris(paths, 4.5 * s))


def draw_reason(reason: str, cx: float, cy: float, size: float, color: rl.Color) -> None:
  if reason in ("turnLeft", "turnRight"):
    draw_maneuver(cx, cy, size, "right", reason == "turnLeft", color)
  elif reason == "arrival":
    draw_maneuver(cx, cy, size, "arrive", False, color)
  else:
    _icon(reason, q(cx), q(cy), q(size)).draw(color)


class NavSpeedSign:
  """Drawn by the HUD under its MAX box: update() each frame with the set speed as shown, then draw()."""
  def __init__(self):
    self._font = gui_app.font(FontWeight.BOLD)
    self._label_font = gui_app.font(FontWeight.SEMI_BOLD)
    self.cap = 0.0  # the speed nav allows, in the display unit
    self.reason = "none"
    self.alpha = 0.0
    self._t: float | None = None

  def update(self, set_speed: float, cruise_set: bool) -> None:
    sm = ui_state.sm
    now = time.monotonic()
    ns = sm['navSpeed']
    fresh = sm.seen['navSpeed'] and now - sm.recv_time['navSpeed'] < FRESH_S
    cap = ns.speedCap * (CV.MS_TO_KPH if ui_state.is_metric else CV.MS_TO_MPH)
    active = (fresh and cruise_set and ui_state.status != UIStatus.DISENGAGED and ns.speedCap > 0
              and round(cap) < round(set_speed))
    if active:
      self.cap, self.reason = cap, str(ns.reason)
    dt = 0.0 if self._t is None else max(now - self._t, 0.0)
    self._t = now
    step = dt / FADE_S
    self.alpha = clamp01(self.alpha + step) if active else clamp01(self.alpha - step)

  def draw(self, x: float, y: float, width: float) -> None:
    a = self.alpha
    if a <= 0.01:
      return
    rect = rl.Rectangle(x, y, width, HEIGHT)
    rl.draw_rectangle_rounded(rect, 0.35, 10, with_alpha(BOX, a))
    rl.draw_rectangle_rounded_lines_ex(rect, 0.35, 10, 6, with_alpha(LIT_BLUE, a))
    if self.reason == "speedLimit":  # worded as the MAX box: a limit sign's number under its word
      label = tr("LIMIT")
      size = measure_text_cached(self._label_font, label, LABEL_SIZE)
      rl.draw_text_ex(self._label_font, label, rl.Vector2(x + (width - size.x) / 2, y + 30), LABEL_SIZE, 0, with_alpha(LIT_BLUE, a))
    else:
      with max_blend():  # the icon's strokes fade as one, without doubling where they join
        draw_reason(self.reason, x + width / 2, y + 54, ICON, premul(LIT_BLUE, a))
    text = str(max(round(self.cap), 0))
    size = measure_text_cached(self._font, text, FONT_SIZE)
    rl.draw_text_ex(self._font, text, rl.Vector2(x + (width - size.x) / 2, y + 92), FONT_SIZE, 0, with_alpha(rl.WHITE, a))
