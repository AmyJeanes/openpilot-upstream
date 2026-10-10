"""Navigation on the comma four's driving view: a turn card at the top right of the camera, clear of driver monitoring
(top left) and of the path's far end (the middle). Far from the next maneuver it's a chip, the maneuver's icon and
distance; as the maneuver nears (maneuver_phase's approach, about 300 m at 20 m/s) it grows and shows the lanes with
the car under its lane; in the junction a "Now" chip; for the arrival a pin chip. While navigation needs the car in
another lane mici's own alert asks for it ("move right" / "signal to confirm") and the card drops under the alert as
the lanes only. The street name is left out: on this screen it can't be read at a glance (it's on the map page).
Navigate on openpilot shows by colour: the lanes that take the maneuver blue while openpilot makes the move, white
while navigation only guides."""
from dataclasses import dataclass

import pyray as rl

from openpilot.selfdrive.ui.mici.onroad.alert_renderer import Alert, AlertSize, AlertStatus
from openpilot.selfdrive.ui.nav.draw import (Anim, draw_car, draw_maneuver, lane_arrow, lerp, max_blend, maneuver_kind, mix, premul,
                                             split_arrow, with_alpha)
from openpilot.selfdrive.ui.nav.nav_state import CardLane, Guidance, NavState, card_lanes, format_distance, maneuver_phase
from openpilot.selfdrive.ui.nav.session import NavSession
from openpilot.selfdrive.ui.ui_state import UIStatus, ui_state
from openpilot.system.ui.lib.application import FONT_SCALE, FontWeight, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

CARD_BG = rl.Color(0x10, 0x11, 0x13, 232)  # near opaque, to read against a bright sky
LIT_BLUE = rl.Color(0x5b, 0x8d, 0xff, 255)  # as the 3X: openpilot will make this move
LANE_DIM = rl.Color(0x5d, 0x5d, 0x66, 255)  # a lane that won't take the maneuver
NOW_BLUE = rl.Color(0x9a, 0xb8, 0xff, 255)

ICON_FOLLOWS_NOO = False  # the maneuver icon stays white; True lights it blue with the lanes
REQUEST_NEEDS_ENGAGED = True  # "signal to confirm" only means something while openpilot steers

MARGIN, PAD, RADIUS = 12.0, 12.0, 16.0
MAX_W = 210.0  # the card stays right of the path's far end, which is mid-screen
UNDER_ALERT_Y = 140.0  # the lanes-only card's top: under the small (lane change) alert, 140 px on mici
ICON_CHIP, ICON_BIG = 42.0, 48.0
EM_CHIP, EM_BIG, UNIT_K = 48.0, 56.0, 0.6  # distance capitals 14.5 and 17 arcmin (MICI notes); the unit at 0.6 of it
LANE_CELL, LANE_ARROW, LANE_CAR = 44.0, 34.0, 22.0
LANES_H = 6 + LANE_ARROW + 4 + LANE_CAR + 2  # the lane row under the turn row: arrows, then the car
ANIM_S = 0.35  # the card's moves between its shapes
LIT_S = 0.5  # the lanes' colour as Navigate on openpilot changes, as the 3X road icon
REQUEST_ON_S, REQUEST_OFF_S = 0.6, 1.0  # a lane request holds this long before it shows, and after it clears
SMALL_ALERTS = ("preLaneChangeLeft", "preLaneChangeRight", "laneChange", "navLaneChangeLeft", "navLaneChangeRight")


def fs(em: float) -> int:
  """A text em in screen px as a size for draw_text_ex, which scales by FONT_SCALE."""
  return max(1, int(round(em / FONT_SCALE)))


def cap_top(y_mid: float, em: float) -> float:
  """The y to draw a line of Inter at for its capitals to centre on y_mid (its line box is 1.211 em, the baseline at
  0.969 em, capitals 0.728 em)."""
  return y_mid - (0.969 / 1.211 * em - 0.728 * em / 2)


def lanes_lit(noo: bool, status: UIStatus) -> bool:
  """The lanes that take the maneuver light blue: openpilot will make the move (Navigate on openpilot on, engaged)."""
  return noo and status != UIStatus.DISENGAGED


def card_mode(g: Guidance, phase: str, has_lanes: bool, alert: Alert | None) -> str:
  """hidden, chip (far), card (near: bigger, with the lanes), now (in the junction) or lanes (under a lane change
  alert). Any other alert takes the card away: alerts always win."""
  if g.maneuver is None:
    return "hidden"
  if alert is not None:
    small = alert.alert_type.split("/")[0] in SMALL_ALERTS
    return "lanes" if small and has_lanes else "hidden"
  if phase == "turn":
    return "now"
  if phase == "approach" and g.maneuver.type != "arrive":
    return "card"
  return "chip"


def lane_request(g: Guidance, lanes: list[CardLane], here: int | None, phase: str) -> str | None:
  """'left' or 'right' while the maneuver is near (the card's approach), the car (by the model's lane head) isn't in
  a lane that takes it, and one is open to move into; None otherwise, and whenever the car's lane is unknown."""
  if here is None or not lanes or phase != "approach" or g.lane_open_distance > 0:
    return None
  lit = [i for i, lane in enumerate(lanes) if lane.straight_lit or lane.turn_lit]
  if not lit or here in lit:
    return None
  return "right" if min(lit, key=lambda i: abs(i - here)) > here else "left"


def request_alert(side: str) -> Alert:
  """Navigation's lane change request, as mici draws a lane change prompt (the turn signal icon, the small height)."""
  text = tr("move right") if side == "right" else tr("move left")
  return Alert(text, tr("signal to confirm"), AlertSize.small, AlertStatus.normal,
               alert_type=f"navLaneChange{side.capitalize()}/warning")


_NOTHING = object()


class Debounce:
  """A value that changes only once a new one has held for on_s (or None for off_s)."""
  def __init__(self, on_s: float, off_s: float):
    self.on_s, self.off_s = on_s, off_s
    self.value = None
    self._pending, self._since = _NOTHING, 0.0

  def update(self, value, now: float):
    if value == self.value:
      self._pending = _NOTHING
    elif value != self._pending:
      self._pending, self._since = value, now
    elif now - self._since >= (self.off_s if value is None else self.on_s):
      self.value, self._pending = value, _NOTHING
    return self.value


@dataclass
class Layout:
  w: float
  h: float
  y: float
  big: float  # 0 the chip's icon and text, 1 the card's
  row: float  # the turn row shown
  lanes: float  # the lane row shown


def card_layout(mode: str, row_w_chip: float, row_w_big: float, lanes_w: float) -> Layout:
  """The card's size and contents for a mode; row_w_*: the turn row's width at each size."""
  if mode == "card":
    w = max(row_w_big, lanes_w) + 2 * PAD
    return Layout(w, PAD + ICON_BIG + (LANES_H if lanes_w else -2) + PAD, MARGIN, 1.0, 1.0, 1.0 if lanes_w else 0.0)
  if mode == "lanes":
    return Layout(lanes_w + 2 * PAD, 8 + LANES_H + 6, UNDER_ALERT_Y, 0.0, 0.0, 1.0)
  return Layout(row_w_chip + 2 * PAD, PAD + ICON_CHIP + PAD - 2, MARGIN, 0.0, 1.0, 0.0)


class NavTurnCard:
  """Drawn by the onroad view: update() each frame it shows, render() over the camera, before the alerts."""
  def __init__(self, nav: NavState, session: NavSession):
    self.nav, self.session = nav, session
    self._bold = gui_app.font(FontWeight.BOLD)
    self._semi = gui_app.font(FontWeight.SEMI_BOLD)
    self.phase = "cruise"
    self.mode = "hidden"  # as drawn this frame
    self._shape = "chip"  # the last shown mode, kept while the card fades away
    self._request = Debounce(REQUEST_ON_S, REQUEST_OFF_S)
    self._anims = {k: Anim(0.0) for k in ("w", "h", "y", "big", "row", "lanes", "shown", "lit")}
    self._lanes_key: tuple = ()
    self._lanes: tuple[list[CardLane], int | None] = ([], None)
    self._now = 0.0

  def lanes(self) -> tuple[list[CardLane], int | None]:
    key = (self.nav.updates, self.nav.car_lane, ui_state.nav_show_lanes_always)
    if key != self._lanes_key:
      self._lanes_key = key
      self._lanes = card_lanes(self.nav.guidance, self.nav.car_lane, ui_state.nav_show_lanes_always)
    return self._lanes

  def update(self, now: float) -> None:
    self._now = now
    on = self.session.route_on
    v = ui_state.sm["carState"].vEgo
    self.phase = maneuver_phase(self.nav.guidance, v, self.phase != "cruise") if on else "cruise"
    lanes, here = self.lanes() if on else ([], None)
    want = lane_request(self.nav.guidance, lanes, here, self.phase) if on else None
    if self.nav.car_lane is None or not self.nav.car_lane.sure:  # never ask to move on the model's guess
      want = None
    if REQUEST_NEEDS_ENGAGED and ui_state.status == UIStatus.DISENGAGED:
      want = None
    self._request.update(want, now)
    self._anims["lit"].set(1.0 if lanes_lit(self.session.noo, ui_state.status) else 0.0, now, LIT_S)

  def request_alert(self) -> Alert | None:
    """Navigation's lane request for the alert renderer, when openpilot has no alert of its own."""
    side = self._request.value
    return request_alert(side) if side is not None and self.session.route_on else None

  def _dist_text(self) -> tuple[str, str]:
    m = self.nav.guidance.maneuver
    if m is None:
      return "", ""
    if self.phase == "turn":
      return tr("Now"), ""
    num, _, unit = format_distance(m.distance, ui_state.is_metric).partition(" ")
    return num, unit

  def _dist_w(self, num: str, unit: str, em: float) -> float:
    w = measure_text_cached(self._bold, num, fs(em)).x
    return w + (0.14 * em + measure_text_cached(self._semi, unit, fs(em * UNIT_K)).x if unit else 0.0)

  def _lane_cell(self, n: int) -> float:
    return min(LANE_CELL, (MAX_W - 2 * PAD) / n) if n else LANE_CELL

  def render(self, content: rl.Rectangle, alert: Alert | None) -> None:
    now = self._now
    g = self.nav.guidance
    lanes, here = self.lanes() if self.session.route_on else ([], None)
    mode = card_mode(g, self.phase, bool(lanes), alert) if self.session.route_on else "hidden"
    num, unit = self._dist_text()
    cell = self._lane_cell(len(lanes))
    lanes_w = cell * len(lanes)
    self.mode = mode
    if mode != "hidden":
      self._shape = mode
    lay = card_layout(self._shape, ICON_CHIP + 10 + self._dist_w(num, unit, EM_CHIP),
                      ICON_BIG + 10 + self._dist_w(num, unit, EM_BIG), lanes_w if self._shape in ("card", "lanes") else 0.0)
    a = self._anims
    first = a["shown"].value(now) <= 0.001  # appearing: straight into its shape, no move from the last one's
    for k in ("w", "h", "y", "big", "row", "lanes"):
      a[k].set(getattr(lay, k), now, ANIM_S)
      if first:
        a[k].hold_at(getattr(lay, k), now - ANIM_S)
    a["shown"].set(0.0 if mode == "hidden" else 1.0, now, ANIM_S)
    shown = a["shown"].value(now)
    if shown <= 0.01:
      return

    w, h, y, big, row, la = (a[k].value(now) for k in ("w", "h", "y", "big", "row", "lanes"))
    r = rl.Rectangle(content.x + content.width - MARGIN - w, content.y + y, w, h)
    rl.draw_rectangle_rounded(r, min(1.0, 2 * RADIUS / max(1.0, min(w, h))), 16, with_alpha(CARD_BG, shown))
    rl.begin_scissor_mode(int(r.x), int(r.y), int(r.width + 1), int(r.height + 1))
    lit = a["lit"].value(now)
    lane_lit = mix(rl.WHITE, LIT_BLUE, lit)

    # the turn row: the maneuver's icon and its distance, centred
    icon, em = lerp(ICON_CHIP, ICON_BIG, big), round(lerp(EM_CHIP, EM_BIG, big))
    row_a = shown * row
    if row_a > 0.01 and g.maneuver is not None:
      row_w = icon + 10 + self._dist_w(num, unit, em)
      x0, ymid = r.x + (w - row_w) / 2, r.y + PAD + max(icon, 0.8 * em) / 2
      kind, left = maneuver_kind(g.maneuver.type, g.maneuver.modifier)
      draw_maneuver(x0 + icon / 2, ymid, icon, kind, left, with_alpha(lane_lit if ICON_FOLLOWS_NOO else rl.WHITE, row_a))
      ty = cap_top(ymid, em)
      tx = x0 + icon + 10
      color = NOW_BLUE if self.phase == "turn" else rl.WHITE
      rl.draw_text_ex(self._bold, num, rl.Vector2(tx, ty), fs(em), 0, with_alpha(color, row_a))
      if unit:
        uem = em * UNIT_K
        ux = tx + measure_text_cached(self._bold, num, fs(em)).x + 0.14 * em
        uy = ty + 0.969 / 1.211 * (em - uem)  # on the number's baseline
        rl.draw_text_ex(self._semi, unit, rl.Vector2(ux, uy), fs(uem), 0, with_alpha(color, 0.8 * row_a))

    # the lanes: those that take the maneuver lit, the rest dim, the car under its lane (faint while the model's unsure)
    if la * shown > 0.01 and lanes:
      ly = r.y + lerp(8, PAD + max(icon, 0.8 * em) + 6, row)
      self._draw_lanes(r.x + (w - lanes_w) / 2, ly, lanes, here, cell, la * shown, lane_lit)
    rl.end_scissor_mode()

  def _draw_lanes(self, x: float, y: float, lanes: list[CardLane], here: int | None, cell: float, alpha: float, lit_col):
    size = min(LANE_ARROW, cell * 0.77)
    lit, dim = premul(lit_col, alpha), premul(LANE_DIM, alpha)
    with max_blend():  # strokes faded one by one would double up where they cross; over the dark card this fades as one
      for i, lane in enumerate(lanes):
        cx, cy = x + cell * (i + 0.5), y + size / 2
        if lane.arrow in ("upright", "upleft"):  # a shared lane lights only the branch the route takes
          split_arrow(cx, cy, size, lane.arrow == "upright", lit if lane.straight_lit else dim, lit if lane.turn_lit else dim)
        else:
          lane_arrow(cx, cy, size, lane.arrow, lit if lane.straight_lit or lane.turn_lit else dim)
        if i == here:
          sure = self.nav.car_lane is not None and self.nav.car_lane.sure
          draw_car(cx, y + size + 4 + LANE_CAR / 2, LANE_CAR, premul(rl.WHITE, alpha if sure else 0.4 * alpha))
