"""The comma four's navigation page, next to the driving view in the pager while a route is active: the map, the turn
card with the street (read in a longer look, so it's here and not on the driving view), the lanes over the map's top
on a fade, the trip row (arrival, time and distance left) and the road button (Navigate on openpilot; slide to end)."""
import time

import pyray as rl

from openpilot.selfdrive.ui.mici.onroad.alert_renderer import AlertRenderer
from openpilot.selfdrive.ui.mici.widgets.road_button import TOUCH_D, SLIDE_D, RoadButton
from openpilot.selfdrive.ui.nav.draw import (Anim, draw_car, draw_maneuver, lane_arrow, mask_corners, mix, maneuver_kind, split_arrow,
                                             with_alpha)
from openpilot.selfdrive.ui.nav.nav_map import MapStyle, NavMap
from openpilot.selfdrive.ui.nav.nav_state import (CardLane, NavState, card_lanes, format_arrival, format_distance, format_duration,
                                                  format_trip_distance, maneuver_phase)
from openpilot.selfdrive.ui.nav.session import NavSession
from openpilot.selfdrive.ui.nav.text import maneuver_road
from openpilot.selfdrive.ui.mici.onroad.nav_card import LANE_DIM, LIT_BLUE, LIT_S, NOW_BLUE, cap_top, fs, lanes_lit
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached
from openpilot.system.ui.widgets import Widget

MAP_BG = rl.Color(0x0e, 0x10, 0x13, 255)
SUB = rl.Color(0x23, 0x24, 0x29, 255)  # the cards over the map, as the 3X's turn section
TEXT2 = rl.Color(0xd4, 0xd4, 0xd8, 255)
INSET, GAP, PAD, RADIUS = 8.0, 8.0, 12.0, 16.0
TURN_W = 228.0
ICON, DIST_EM, DIST_EM_MIN, UNIT_K = 48.0, 56.0, 40.0, 0.6
STREET_EM, STREET_LINE = 34.0, 38.0  # 10 arcmin: for a second look, as stock mici's secondary alert text
TRIP_W, TRIP_H, TRIP_EM, TRIP_EM_MIN = 300.0, 52.0, 34.0, 26.0
LANE_CELL, LANE_ARROW, LANE_CAR = 46.0, 36.0, 20.0
# the map drawn for this page: thinner strokes than the 3X's, and the car low and right of centre, between the trip row
# and the road button, so the route ahead runs up the map's open middle
MICI_MAP = MapStyle(road_px_min=5.0, road_px_per_m=0.35, road_px_max=13.0, route_px=8.0, car_size=13.0, marker_radius=6.0,
                    car_at=0.86, car_x=0.70, top_margin=110.0, view_min=120.0, view_max=650.0)


def wrap_lines(font: rl.Font, s: str, size: int, width: float, max_lines: int) -> list[str]:
  """Words wrapped to width; the last line cut short with '..' (the UI fonts have no ellipsis glyph) if it runs over."""
  lines, cur = [], ""
  words = s.split()
  for i, word in enumerate(words):
    trial = f"{cur} {word}".strip()
    if cur and measure_text_cached(font, trial, size).x > width:
      lines.append(cur)
      cur = word
      if len(lines) == max_lines - 1:
        cur = " ".join(words[i:])
        break
    else:
      cur = trial
  if cur:
    lines.append(cur)
  if lines and measure_text_cached(font, lines[-1], size).x > width:
    last = lines[-1]
    while len(last) > 1 and measure_text_cached(font, last + "..", size).x > width:
      last = last[:-1]
    lines[-1] = last.rstrip() + ".."
  return lines[:max_lines]


class NavMapPage(Widget):
  def __init__(self, nav: NavState, session: NavSession, on_end):
    super().__init__()
    self.nav, self.session = nav, session
    self.map = NavMap(style=MICI_MAP, fps=gui_app.target_fps)
    self.button = self._child(RoadButton(session, on_end))
    self.button.set_touch_valid_callback(lambda: self._touch_valid())
    self._alerts = AlertRenderer()  # openpilot's own alerts only: navigation's lane request is the lanes here
    self._bold = gui_app.font(FontWeight.BOLD)
    self._semi = gui_app.font(FontWeight.SEMI_BOLD)
    self._lit = Anim(0.0)
    self.phase = "cruise"
    self._cache: dict[tuple, object] = {}

  def _cached(self, key: tuple, make):
    if key not in self._cache:
      if len(self._cache) > 32:
        self._cache.clear()
      self._cache[key] = make()
    return self._cache[key]

  def _update_state(self) -> None:
    now = time.monotonic()
    v = ui_state.sm["carState"].vEgo
    self.phase = maneuver_phase(self.nav.guidance, v, self.phase != "cruise") if self.session.route_on else "cruise"
    self._lit.set(1.0 if lanes_lit(self.session.noo, ui_state.status) else 0.0, now, LIT_S)

  def _render(self, rect: rl.Rectangle) -> None:
    now = time.monotonic()
    rl.draw_rectangle_rec(rect, rl.BLACK)
    m = rl.Rectangle(rect.x + INSET, rect.y + INSET, rect.width - 2 * INSET, rect.height - 2 * INSET)
    rl.draw_rectangle_rounded(m, 2 * RADIUS / m.height, 16, MAP_BG)
    rl.begin_scissor_mode(int(m.x), int(m.y), int(m.width), int(m.height))
    self.map.render(m, self.nav)
    rl.end_scissor_mode()
    mask_corners(m, RADIUS, rl.BLACK)

    keep = 1.0 - min(1.0, self.button.slide.offset / 30.0)  # the trip row gives way to the slide track
    lit = mix(rl.WHITE, LIT_BLUE, self._lit.value(now))
    turn = self._draw_turn(m)
    lanes, here = card_lanes(self.nav.guidance, self.nav.car_lane, ui_state.nav_show_lanes_always)
    if lanes:
      self._draw_lanes(m, turn, lanes, here, lit)
    if keep > 0.01:
      self._draw_trip(m, keep)

    # the road button in the map's bottom right corner; its track runs left to the map's inset
    bcx, bcy = m.x + m.width - GAP - SLIDE_D / 2, m.y + m.height - GAP - SLIDE_D / 2
    self.button.track_left = m.x + GAP
    self.button.render(rl.Rectangle(bcx - TOUCH_D / 2, bcy - TOUCH_D / 2, TOUCH_D, TOUCH_D))

    self._alerts.render(rect)

  def _draw_turn(self, m: rl.Rectangle) -> "rl.Rectangle | None":
    """The next maneuver: its icon and distance, and the street it leads onto (the destination, arriving)."""
    g = self.nav.guidance
    if g.maneuver is None:
      return None
    metric = ui_state.is_metric
    dist = tr("Now") if self.phase == "turn" else format_distance(g.maneuver.distance, metric)
    street = maneuver_road(g)
    inner = TURN_W - 2 * PAD
    lines = self._cached(("street", street), lambda: wrap_lines(self._semi, street, fs(STREET_EM), inner, 2)) if street else []
    h = PAD + 56 + STREET_LINE * len(lines) + (8 if lines else PAD - 2)
    r = rl.Rectangle(m.x + GAP, m.y + GAP, TURN_W, h)
    rl.draw_rectangle_rounded(r, 2 * RADIUS / h, 16, with_alpha(SUB, 0.97))
    kind, left = maneuver_kind(g.maneuver.type, g.maneuver.modifier)
    ymid = r.y + PAD + 26
    draw_maneuver(r.x + PAD + ICON / 2, ymid, ICON, kind, left, rl.WHITE)
    num, _, unit = dist.partition(" ")
    width = inner - ICON - 10
    em = self._cached(("dist", dist, width), lambda: self._fit_dist(num, unit, width))
    color = NOW_BLUE if self.phase == "turn" else rl.WHITE
    tx, ty = r.x + PAD + ICON + 10, cap_top(ymid, em)
    rl.draw_text_ex(self._bold, num, rl.Vector2(tx, ty), fs(em), 0, color)
    if unit:
      uem = em * UNIT_K
      ux = tx + measure_text_cached(self._bold, num, fs(em)).x + 0.14 * em
      rl.draw_text_ex(self._semi, unit, rl.Vector2(ux, ty + 0.969 / 1.211 * (em - uem)), fs(uem), 0, with_alpha(color, 0.8))
    for i, line in enumerate(lines):
      rl.draw_text_ex(self._semi, line, rl.Vector2(r.x + PAD, r.y + PAD + 60 + STREET_LINE * i), fs(STREET_EM), 0, TEXT2)
    return r

  def _fit_dist(self, num: str, unit: str, width: float) -> float:
    em = DIST_EM
    while em > DIST_EM_MIN:
      w = measure_text_cached(self._bold, num, fs(em)).x
      w += 0.14 * em + measure_text_cached(self._semi, unit, fs(em * UNIT_K)).x if unit else 0.0
      if w <= width:
        break
      em -= 1
    return em

  def _draw_lanes(self, m: rl.Rectangle, turn: "rl.Rectangle | None", lanes: list[CardLane], here: int | None, lit_col) -> None:
    """Over the map's top right, beside the turn card, on a fade into the map (as the 3X floats them over its map)."""
    x0 = (turn.x + turn.width if turn is not None else m.x) + GAP
    x1 = m.x + m.width - GAP
    cell = min(LANE_CELL, (x1 - x0) / len(lanes))
    size = min(LANE_ARROW, cell * 0.78)
    bx, bw = int(x0 - GAP), int(m.x + m.width - (x0 - GAP))
    rl.draw_rectangle_gradient_v(bx, int(m.y), bw, 70, with_alpha(MAP_BG, 0.95), with_alpha(MAP_BG, 0.8))
    rl.draw_rectangle_gradient_v(bx, int(m.y) + 70, bw, 26, with_alpha(MAP_BG, 0.8), with_alpha(MAP_BG, 0.0))
    mask_corners(m, RADIUS, rl.BLACK)
    lx = x0 + (x1 - x0 - cell * len(lanes)) / 2
    for i, lane in enumerate(lanes):
      cx, cy = lx + cell * (i + 0.5), m.y + 14 + size / 2
      if lane.arrow in ("upright", "upleft"):
        split_arrow(cx, cy, size, lane.arrow == "upright", lit_col if lane.straight_lit else LANE_DIM,
                    lit_col if lane.turn_lit else LANE_DIM)
      else:
        lane_arrow(cx, cy, size, lane.arrow, lit_col if lane.straight_lit or lane.turn_lit else LANE_DIM)
      if i == here:
        sure = self.nav.car_lane is not None and self.nav.car_lane.sure
        draw_car(cx, m.y + 14 + size + 4 + LANE_CAR / 2, LANE_CAR, with_alpha(rl.WHITE, 1.0 if sure else 0.4))

  def _draw_trip(self, m: rl.Rectangle, alpha: float) -> None:
    """Arrival, time left and distance left: three centred values without labels, shrunk together to fit."""
    g = self.nav.guidance
    if g.time_remaining <= 0 and g.distance_remaining <= 0:
      return
    values = (format_arrival(g.time_remaining), format_duration(g.time_remaining),
              format_trip_distance(g.distance_remaining, ui_state.is_metric))
    r = rl.Rectangle(m.x + GAP, m.y + m.height - GAP - TRIP_H, TRIP_W, TRIP_H)
    rl.draw_rectangle_rounded(r, 2 * RADIUS / TRIP_H, 16, with_alpha(SUB, 0.97 * alpha))

    def layout():
      em = TRIP_EM
      while em > TRIP_EM_MIN and max(measure_text_cached(self._bold, v, fs(em)).x for v in values) > r.width / 3 - 14:
        em -= 1
      return em, [measure_text_cached(self._bold, v, fs(em)).x for v in values]
    em, widths = self._cached(("trip", values), layout)
    for i, (v, vw) in enumerate(zip(values, widths, strict=True)):
      rl.draw_text_ex(self._bold, v, rl.Vector2(r.x + r.width * (i + 0.5) / 3 - vw / 2, cap_top(r.y + TRIP_H / 2, em)), fs(em), 0,
                      with_alpha(rl.WHITE, alpha))
