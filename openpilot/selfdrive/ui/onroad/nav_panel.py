"""The navigation panel beside the onroad camera view on the comma 3X: the next maneuver, the lanes for it, a map of the
route ahead, and the arrival. Lane change requests show as openpilot's own alerts in the camera view."""
import pyray as rl

from openpilot.selfdrive.ui.nav.draw import draw_dashed_rect, draw_lane_arrow, draw_maneuver_icon
from openpilot.selfdrive.ui.nav.nav_map import NavMap
from openpilot.selfdrive.ui.nav.nav_state import Guidance, Lane, NavState, format_arrival, format_distance, format_duration
from openpilot.selfdrive.ui.nav.text import lane_caption, maneuver_road, then_text
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.lib.wrap_text import wrap_text
from openpilot.system.ui.widgets import Widget

GAP = 18
RADIUS = 42  # px, the cards' corners
PAD_X = 39

CARD = rl.Color(0x17, 0x18, 0x1A, 0xFF)
TEXT = rl.WHITE
TEXT_SECONDARY = rl.Color(0xB4, 0xB4, 0xBB, 0xFF)
TEXT_ROAD = rl.Color(0xD4, 0xD4, 0xD8, 0xFF)
LANE_BG = rl.Color(0x26, 0x27, 0x2B, 0xFF)
LANE_ONCOMING_BG = rl.Color(0x0C, 0x0C, 0x0D, 0xFF)
LANE_ONCOMING_EDGE = rl.Color(0x3A, 0x3A, 0x40, 0xFF)
LANE_ONCOMING_ARROW = rl.Color(0x5B, 0x5B, 0x63, 0xFF)
LANE_ARROW = rl.Color(0x8C, 0x8C, 0x96, 0xFF)
LANE_TARGET = rl.Color(0x2F, 0x63, 0xD8, 0xFF)
CHIP = rl.Color(0x17, 0x18, 0x1A, 0xEB)

# px; secondary text stays at least 48 px, to read on the 6" screen
DISTANCE_SIZE = 96
ROAD_SIZE = 52
LANE_TITLE_SIZE = 54
LANE_CAPTION_SIZE = 48
CHIP_SIZE = 48
FOOTER_VALUE_SIZE = 60
FOOTER_LABEL_SIZE = 48
DESTINATION_SIZE = 48

ICON_SIZE = 132
LANE_W, LANE_H, LANE_GAP, LANE_MIN_W = 96, 114, 15, 60
MANEUVER_H = 210
LANES_H = 168
FOOTER_H = 168
FOOTER_GAP = 40

def fit_text(font: rl.Font, text: str, size: int, width: float) -> str:
  """The text cut short with ".." to fit the width (the UI fonts have no ellipsis glyph)."""
  if measure_text_cached(font, text, size).x <= width:
    return text
  n = len(text)
  while n > 1 and measure_text_cached(font, text[:n] + "..", size).x > width:
    n -= 1
  return text[:n].rstrip() + ".."


class NavPanel(Widget):
  def __init__(self, nav: NavState):
    super().__init__()
    self.nav = nav
    self.map = NavMap(fps=gui_app.target_fps)
    self._bold = gui_app.font(FontWeight.BOLD)
    self._semi_bold = gui_app.font(FontWeight.SEMI_BOLD)
    self._medium = gui_app.font(FontWeight.MEDIUM)

  def _render(self, rect: rl.Rectangle):
    g = self.nav.guidance
    metric = ui_state.is_metric
    y = rect.y
    self._draw_maneuver(rl.Rectangle(rect.x, y, rect.width, MANEUVER_H), g, metric)
    y += MANEUVER_H + GAP
    if g.show_lanes:
      self._draw_lanes(rl.Rectangle(rect.x, y, rect.width, LANES_H), g, metric)
      y += LANES_H + GAP
    map_h = rect.y + rect.height - FOOTER_H - GAP - y
    self._draw_map(rl.Rectangle(rect.x, y, rect.width, map_h), g, metric)
    self._draw_footer(rl.Rectangle(rect.x, rect.y + rect.height - FOOTER_H, rect.width, FOOTER_H), g, metric)

  @staticmethod
  def _card(rect: rl.Rectangle, color: rl.Color = CARD) -> None:
    rl.draw_rectangle_rounded(rect, RADIUS / (min(rect.width, rect.height) / 2), 12, color)

  def _text(self, font: rl.Font, text: str, size: int, x: float, y: float, color: rl.Color) -> None:
    rl.draw_text_ex(font, text, rl.Vector2(x, y), size, 0, color)

  def _draw_maneuver(self, rect: rl.Rectangle, g: Guidance, metric: bool) -> None:
    self._card(rect)
    m = g.maneuver
    if m is None:
      return
    icon = rl.Rectangle(rect.x + PAD_X, rect.y + (rect.height - ICON_SIZE) / 2, ICON_SIZE, ICON_SIZE)
    draw_maneuver_icon(icon, m.type, m.modifier, TEXT, 12.0)
    x = icon.x + ICON_SIZE + 33
    width = rect.x + rect.width - PAD_X - x
    dist = format_distance(m.distance, metric) if m.type != "arrive" or m.distance > 30 else tr("Now")
    road = fit_text(self._semi_bold, maneuver_road(g), ROAD_SIZE, width)
    d_h, r_h = measure_text_cached(self._bold, dist, DISTANCE_SIZE).y, measure_text_cached(self._semi_bold, road, ROAD_SIZE).y
    top = rect.y + (rect.height - d_h - r_h) / 2
    self._text(self._bold, dist, DISTANCE_SIZE, x, top, TEXT)
    self._text(self._semi_bold, road, ROAD_SIZE, x, top + d_h, TEXT_ROAD)

  def _draw_lanes(self, rect: rl.Rectangle, g: Guidance, metric: bool) -> None:
    self._card(rect)
    n = len(g.lanes)
    max_w = rect.width * 0.55
    w = min(LANE_W, (max_w - LANE_GAP * (n - 1)) / max(n, 1))
    w = max(w, LANE_MIN_W)
    x, y = rect.x + PAD_X, rect.y + (rect.height - LANE_H) / 2
    for lane in g.lanes:
      self._draw_lane(rl.Rectangle(x, y, w, LANE_H), lane)
      x += w + LANE_GAP
    x += 12
    title, sub = lane_caption(g, metric)
    width = rect.x + rect.width - PAD_X - x
    title = fit_text(self._bold, title, LANE_TITLE_SIZE, width)
    sub = fit_text(self._medium, sub, LANE_CAPTION_SIZE, width)
    t_h = measure_text_cached(self._bold, title, LANE_TITLE_SIZE).y
    s_h = measure_text_cached(self._medium, sub, LANE_CAPTION_SIZE).y if sub else 0
    top = rect.y + (rect.height - t_h - s_h) / 2
    self._text(self._bold, title, LANE_TITLE_SIZE, x, top, TEXT)
    if sub:
      self._text(self._medium, sub, LANE_CAPTION_SIZE, x, top + t_h, TEXT_SECONDARY)

  def _draw_lane(self, rect: rl.Rectangle, lane: Lane) -> None:
    roundness = 21 / (min(rect.width, rect.height) / 2)
    side = min(rect.width, rect.height) * 0.55
    arrow = rl.Rectangle(rect.x + (rect.width - side) / 2, rect.y + (rect.height - side) / 2, side, side)
    if lane.oncoming:
      rl.draw_rectangle_rounded(rect, roundness, 8, LANE_ONCOMING_BG)
      draw_dashed_rect(rect, 21, 9, 3, LANE_ONCOMING_EDGE)
      draw_lane_arrow(arrow, "straight", LANE_ONCOMING_ARROW, 4.0, oncoming=True)
      return
    rl.draw_rectangle_rounded(rect, roundness, 8, LANE_TARGET if lane.active else LANE_BG)
    if lane.current:
      rl.draw_rectangle_rounded_lines_ex(rl.Rectangle(rect.x + 2, rect.y + 2, rect.width - 4, rect.height - 4), roundness, 8, 4, TEXT)
    bright = lane.active or lane.current
    for d in lane.directions:
      if lane.active and d == lane.active_direction:
        continue
      draw_lane_arrow(arrow, d, TEXT if bright and not lane.active else LANE_ARROW, 4.5)
    if lane.active:
      draw_lane_arrow(arrow, lane.active_direction, TEXT, 4.8)

  def _draw_map(self, rect: rl.Rectangle, g: Guidance, metric: bool) -> None:
    style = self.map.style
    self._card(rect, style.background)
    rl.begin_scissor_mode(int(rect.x), int(rect.y), int(rect.width), int(rect.height))
    self.map.render(rect, self.nav, ui_state.sm["carState"].vEgo)
    chip = then_text(g, metric)
    if chip:
      size = measure_text_cached(self._bold, chip, CHIP_SIZE)
      box = rl.Rectangle(rect.x + 24, rect.y + 21, min(size.x + 54, rect.width - 48), size.y + 30)
      rl.draw_rectangle_rounded(box, 24 / (box.height / 2), 10, CHIP)
      self._text(self._bold, fit_text(self._bold, chip, CHIP_SIZE, box.width - 54), CHIP_SIZE, box.x + 27, box.y + 15, TEXT)
    rl.end_scissor_mode()
    # round the map's corners: cover what the square clip let past them, within the gap around the card
    rl.draw_rectangle_rounded_lines_ex(rect, RADIUS / (min(rect.width, rect.height) / 2), 12, GAP - 1, rl.BLACK)

  def _draw_footer(self, rect: rl.Rectangle, g: Guidance, metric: bool) -> None:
    """Arrival time, then the time and distance left, and the destination in what's left of the width."""
    self._card(rect)
    x, right = rect.x + PAD_X, rect.x + rect.width - PAD_X
    for value, label in ((format_arrival(g.time_remaining), tr("arrival")),
                         (format_duration(g.time_remaining), format_distance(g.distance_remaining, metric))):
      v = measure_text_cached(self._bold, value, FOOTER_VALUE_SIZE)
      lab = measure_text_cached(self._medium, label, FOOTER_LABEL_SIZE)
      top = rect.y + (rect.height - v.y - lab.y) / 2
      self._text(self._bold, value, FOOTER_VALUE_SIZE, x, top, TEXT)
      self._text(self._medium, label, FOOTER_LABEL_SIZE, x, top + v.y, TEXT_SECONDARY)
      x += max(v.x, lab.x) + FOOTER_GAP
    # the destination's name, right-aligned on up to two lines in the width left
    lines = wrap_text(self._bold, g.destination or tr("Destination"), DESTINATION_SIZE, int(right - x))
    if len(lines) > 2:
      lines = [lines[0], fit_text(self._bold, " ".join(lines[1:]), DESTINATION_SIZE, right - x)]
    sizes = [measure_text_cached(self._bold, line, DESTINATION_SIZE) for line in lines]
    y = rect.y + (rect.height - sum(s.y for s in sizes)) / 2
    for line, size in zip(lines, sizes, strict=True):
      self._text(self._bold, line, DESTINATION_SIZE, right - size.x, y, TEXT)
      y += size.y
