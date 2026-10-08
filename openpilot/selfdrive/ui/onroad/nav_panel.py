"""Navigation on the comma 3X (tizi): a card over the camera view that grows out of the experimental button's spot when
a route starts, or an always-on split with the camera beside a panel. The card shows the next maneuver and the trip;
open, a map with the lanes for the maneuver floating over its top. The top-right button becomes Navigate on openpilot's
toggle while a route is active, and slides left to end the route.

The card opens by itself for a maneuver ahead, and by hand (a tap, or a drag down that follows the finger); a hand
open or close lasts until nav would next change it, and a hand open lapses after OVERRIDE_OPEN_S. The map's own buttons
(the layout pin) show while the card was opened by hand or the map is tapped, and fade after MAP_CONTROLS_S."""
import math
import time

import pyray as rl

from openpilot.common.params import Params
from openpilot.selfdrive.ui import UI_BORDER_SIZE
from openpilot.selfdrive.ui.nav.draw import (clamp01, draw_car, draw_maneuver, draw_pin, draw_road_icon, ease, lane_arrow, lerp,
                                             mask_corners, max_blend, maneuver_kind, mix, premul, smootherstep, smoothstep,
                                             split_arrow, with_alpha, xfade)
from openpilot.selfdrive.ui.nav.nav_map import NavMap
from openpilot.selfdrive.ui.nav.nav_state import (NavState, card_lanes, format_arrival, format_distance, format_duration,
                                                  format_trip_distance, maneuver_phase)
from openpilot.selfdrive.ui.nav.text import maneuver_road
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import FONT_SCALE, FontWeight, MouseEvent, gui_app
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.lib.text_measure import measure_text_cached

BORDER = UI_BORDER_SIZE
BLACK_TRANSLUCENT = rl.Color(0, 0, 0, 166)  # the stock experimental button's disc
NAV_DISC = rl.Color(0, 0, 0, 110)  # the nav button's disc in the card
SLIDE_TRACK_BG = rl.Color(0x0c, 0x0c, 0x0e, 255)
SLIDE_DISC = rl.Color(0x2a, 0x2b, 0x30, 255)
CARD = rl.Color(0x17, 0x18, 0x1a, 255)
SUB = rl.Color(0x23, 0x24, 0x29, 255)
MAP_BG = rl.Color(0x0e, 0x10, 0x13, 255)
MAP_BUTTON = rl.Color(0x23, 0x24, 0x29, 235)
DIM = rl.Color(0x8c, 0x8c, 0x96, 255)  # the nav button's road and car with Navigate on openpilot off
LIT_BLUE = rl.Color(0x5b, 0x8d, 0xff, 255)  # ... and on; the lanes that take the turn with it on
TEXT2 = rl.Color(0xd4, 0xd4, 0xd8, 255)
NOW = rl.Color(0x9a, 0xb8, 0xff, 255)
RULE = rl.Color(255, 255, 255, 36)
LANE_DIM = rl.Color(0x5d, 0x5d, 0x66, 255)  # a lane arrow that won't take the turn
END_RED = rl.Color(0xC9, 0x22, 0x31, 255)  # the nav button as it becomes the end X: red, as alerts

STOCK_BUTTON = 192  # the experimental button: its size and the nav button's touch area
STOCK_ICON = 144
NAV_BUTTON = 136.0  # the nav button as drawn in the card
CARD_W, PANEL_W, PANEL_GAP = 615.0, 660.0, 12.0  # the card over the camera; the split's panel, and its gap to the camera
CARD_MARGIN = 22.0  # the card in from the camera view's content edges
SLIDE_TRACK = 420.0  # px the button slides left along its "slide to end" track, when nothing sets its left end
OVERRIDE_OPEN_S = 15.0  # a card opened by hand closes again after this, if nav hasn't opened it itself
DRAG_SNAP = 0.06  # let go after dragging the card this much of the way and it snaps open or closed
SLIDE_END = 0.75  # let go past this much of the track to end the route
MAP_CONTROLS_S = 4.0  # the map's buttons fade away after this long untouched; a tap on the map brings them back
EXPAND_S = 0.5  # the card's expand/collapse, the lanes' slide, the route's start and end and the layout switch
NOO_HOLD_S = 2.0  # a tapped Navigate on openpilot shows as set while the param catches up
LANES_H = 168.0
NOWHERE = rl.Rectangle(0, 0, 0, 0)  # a hit area with nothing drawn
END_MOVED = 50.0  # m the destination moves for a route that was ended to count as a new one


def fs(px: float) -> int:
  """A text size in screen px, as a size for draw_text_ex (which scales by FONT_SCALE)."""
  return int(round(px / FONT_SCALE))


class Anim:
  """A value easing towards its target over `dur` seconds."""
  def __init__(self, v: float):
    self.a = self.b = v
    self.t0 = -1e9
    self.dur = 0.7

  def set(self, target: float, now: float, dur: float):
    if target != self.b:
      self.a, self.b, self.t0, self.dur = self.value(now), target, now, dur

  def value(self, now: float) -> float:
    return lerp(self.a, self.b, ease((now - self.t0) / self.dur))

  def hold_at(self, v: float, now: float):
    """Restart from v (where a drag left it) towards the current target."""
    self.a, self.t0 = v, now

  def linear(self, now: float) -> float:
    """The move on plain time, not eased: for crossfades, which an ease-out would rush through in a frame or two."""
    return lerp(self.a, self.b, clamp01((now - self.t0) / self.dur))

  def fade(self, now: float) -> float:
    return lerp(self.a, self.b, clamp01((now - self.t0) / (0.6 * self.dur)))


def wrap(font: rl.Font, s: str, size: int, width: float, max_lines: int) -> list[str]:
  lines, cur = [], ""
  for word in s.split():
    trial = f"{cur} {word}".strip()
    if cur and measure_text_cached(font, trial, size).x > width:
      lines.append(cur)
      cur = word
    else:
      cur = trial
  lines.append(cur)
  return lines[:max_lines]


def in_circle(hit: tuple[float, float, float], x: float, y: float) -> bool:
  cx, cy, rad = hit
  return rad > 0 and (x - cx) ** 2 + (y - cy) ** 2 <= rad ** 2


def in_rect(r: rl.Rectangle, x: float, y: float) -> bool:
  return rl.check_collision_point_rec(rl.Vector2(x, y), r)


class NavCard:
  """Drawn by NavSplitView: update() before the camera view renders (touches, animations, the camera's rect), draw()
  from the camera view's overlay hook, over the HUD and under the alerts."""
  def __init__(self, nav: NavState):
    self.nav = nav
    self.map = NavMap(fps=gui_app.target_fps)
    self._params = Params()
    self._bold = gui_app.font(FontWeight.BOLD)
    self._semi = gui_app.font(FontWeight.SEMI_BOLD)
    self._wheel = gui_app.texture("icons/chffr_wheel.png", STOCK_ICON, STOCK_ICON)
    self._exp_icon = gui_app.texture("icons/experimental.png", STOCK_ICON, STOCK_ICON)

    self.split_on = False  # the layout pin: the always-on split, else the card over the camera
    self.noo = False  # Navigate on openpilot, as shown
    self._noo_set_t = -1e9
    self.route_on = False  # a route to show: nav has one, and the driver hasn't ended it here
    self._ended: tuple | None = None  # the route the driver slid to end, until nav drops it or has a new destination
    self.phase = "cruise"
    self.want_open = False
    # a hand open or close of the card overrides nav's own choice until nav would next change it
    self.override: str | None = None
    self.override_t, self.last_auto = 0.0, False
    self.claimed = False  # the touch going on began on the card or its button, so the camera view doesn't take it

    self.hit_exp = (0.0, 0.0, 0.0)  # centre and touch radius of the top-right button, as last drawn
    self.hit_card = rl.Rectangle(0, 0, 0, 0)
    self.hit_eta = rl.Rectangle(0, 0, 0, 0)
    self.hit_map = rl.Rectangle(0, 0, 0, 0)
    self.hit_layout = (0.0, 0.0, 0.0)
    self.press_x: float | None = None  # where a press on the button began, while it's down
    self.slide = 0.0  # px the button is slid left (springs back when let go short of the end)
    self.dragged = False  # the press moved enough to be a slide, so its release isn't a tap
    self.slide_target = 0.0  # where the finger has the button, which it eases towards
    self.slide_len = SLIDE_TRACK
    self.confirm_t: float | None = None  # let go past the end point: the button finishes its travel, then the route ends
    self.vdrag: tuple[float, float] | None = None  # (y, open amount) where a drag on the card began
    self.o_live: float | None = None  # the open amount while the finger has the card
    self.drag_range = 700.0

    self.open = Anim(0.0)
    self.lanes = Anim(0.0)
    self.split = Anim(0.0)
    self.present = Anim(0.0)  # the route's card on screen: 0 none, 1 there
    self.noo_lit = Anim(0.0)  # the nav button's road: 0 grey, 1 lit blue
    self.map_ctrl = Anim(0.0)  # the map's own buttons (the pin; later heading lock and the like): 0 hidden, 1 shown
    self.map_poke_t = -1e9
    self.inner_r = 0.0
    self.clip = rl.Rectangle(0, 0, 0, 0)  # the card's visible rect while it grows or shrinks
    self._now = 0.0
    self._rect = rl.Rectangle(0, 0, 0, 0)
    # what's drawn from each navInstruction, and the text's layout, kept until what they're made from changes
    self._words_key: tuple = ()
    self._words: tuple = ()
    self._lanes_key = -1
    self._lanes_cache: tuple = ([], None)
    self._layout_cache: dict[tuple, object] = {}
    self.camera_rect = rl.Rectangle(0, 0, 0, 0)

  @property
  def stock_button_shown(self) -> bool:
    """The HUD's own experimental button: only with no route at all, the card has it otherwise."""
    return self.present.value(self._now) <= 0.001

  def _route_key(self) -> tuple:
    return self.nav.guidance.destination, self.nav.route_end

  def _new_destination(self, ended: tuple) -> bool:
    (name, end), (name0, end0) = self._route_key(), ended
    if name != name0 or (end is None) != (end0 is None):
      return True
    if end is None or end0 is None:
      return False
    dy, dx = (end[0] - end0[0]) * 111319.0, (end[1] - end0[1]) * 111319.0 * math.cos(math.radians(end[0]))
    return math.hypot(dx, dy) > END_MOVED

  def set_noo(self, on: bool):
    self.noo, self._noo_set_t = on, time.monotonic()
    self._params.put_bool("NavigateOnOpenpilot", on)

  # *** input ***

  def on_button(self, x: float, y: float) -> bool:
    return in_circle(self.hit_exp, x, y)

  def can_drag_card(self, x: float, y: float) -> bool:
    """Like a maps app's sheet: the small card drags open from anywhere on it, the open card closes from its trip
    row, never from the button."""
    if self.split_on or not self.route_on or self.on_button(x, y):
      return False
    return in_rect(self.hit_eta if self.open.b > 0.5 else self.hit_card, x, y)

  def _claims(self, x: float, y: float) -> bool:
    if not self.route_on:
      return False
    return (self.on_button(x, y) or in_rect(self.hit_card, x, y) or in_rect(self.hit_map, x, y) or
            in_circle(self.hit_layout, x, y))

  def tap(self, x: float, y: float):
    now = time.monotonic()
    if self.on_button(x, y):
      if self.route_on:
        self.set_noo(not self.noo)
    elif in_circle(self.hit_layout, x, y):
      self.map_poke_t = now
      leaving_split = self.split_on
      self.split_on = not self.split_on
      self.override = None
      if leaving_split:  # stay open, as if pulled open by hand, until nav would next close it (or the timeout)
        self.override, self.override_t = "open", now
        self.last_auto = self.phase != "cruise"
    elif in_rect(self.hit_map, x, y):  # the map keeps the card open; shows its buttons
      self.map_poke_t = now
      if self.override == "open":  # still in use: a hand-opened card's timeout starts again
        self.override_t = now
    elif not self.split_on and self.route_on and in_rect(self.hit_card, x, y):
      self.override, self.override_t = ("closed" if self.open.b > 0.5 else "open"), now  # by hand

  def _handle_event(self, ev: MouseEvent):
    x, y = ev.pos.x, ev.pos.y
    if ev.left_pressed:
      self.claimed = self._claims(x, y)
      if self.route_on and self.on_button(x, y):
        self.press_x, self.dragged = x, False
      elif self.can_drag_card(x, y):
        self.vdrag = (y, self.open.value(time.monotonic()))
    elif ev.left_down and self.press_x is not None and self.route_on:
      self.slide_target = min(self.slide_len, max(0.0, self.press_x - x))
      self.dragged = self.dragged or self.slide_target > 24
    elif ev.left_down and self.vdrag is not None:
      y0, o0 = self.vdrag
      if self.o_live is not None or abs(y - y0) > 12:  # moved enough to be a drag, not a tap
        self.o_live = clamp01(o0 + (y - y0) / max(1.0, self.drag_range))
    if ev.left_released:
      now = time.monotonic()
      if self.press_x is not None and self.dragged:
        if self.slide_target >= SLIDE_END * self.slide_len:  # slid far enough: finish the travel, then end the route
          self.confirm_t = now
      elif self.o_live is not None and self.vdrag is not None:  # let go of a drag: past a small threshold snap that way
        o0 = self.vdrag[1]
        if self.o_live - o0 > DRAG_SNAP:
          self.override, self.override_t = "open", now
        elif o0 - self.o_live > DRAG_SNAP:
          self.override, self.override_t = "closed", now
        self.open.hold_at(self.o_live, now)  # carry on from where the finger left it
        if self.phase == "approach" and self._lanes()[0]:
          self.lanes.hold_at(self.o_live, now)
        self.o_live = None
      elif self.claimed:
        self.tap(x, y)
      self.press_x, self.vdrag = None, None
      self.claimed = False

  def _end_route(self):
    self._ended = self._route_key()
    self._params.remove("NavDestination")  # as a phone or the map would: navd drops the destination

  # *** state ***

  def _lanes(self):
    if not self.route_on:
      return [], None
    if self._lanes_key != self.nav.updates:
      self._lanes_key, self._lanes_cache = self.nav.updates, card_lanes(self.nav.guidance)
    return self._lanes_cache

  def _guidance_words(self) -> tuple:
    """(distance, road, icon kind, mirrored, trip row) for the latest navInstruction."""
    metric = ui_state.is_metric
    key = (self.nav.updates, self.phase, metric)
    if key != self._words_key:
      g = self.nav.guidance
      m = g.maneuver
      dist = tr("Now") if self.phase == "turn" else (format_distance(m.distance, metric) if m is not None else "")
      kind, left = maneuver_kind(m.type, m.modifier) if m is not None else ("straight", False)
      eta = (format_arrival(g.time_remaining), format_duration(g.time_remaining), format_trip_distance(g.distance_remaining, metric))
      self._words_key, self._words = key, (dist, maneuver_road(g), kind, left, eta)
    return self._words

  def _cached(self, key: tuple, make):
    if key not in self._layout_cache:
      if len(self._layout_cache) > 64:  # sizes in between, from animations
        self._layout_cache.clear()
      self._layout_cache[key] = make()
    return self._layout_cache[key]

  def update(self, rect: rl.Rectangle):
    now = self._now = time.monotonic()
    self._rect = rect
    if self._ended is not None and (not self.nav.active or self._new_destination(self._ended)):
      self._ended = None
    route_on = self.nav.active and self._ended is None
    if route_on and not self.route_on:  # each new route starts with Navigate on openpilot off: guidance until tapped on
      self.set_noo(False)
      self.override = None
    self.route_on = route_on
    if now - self._noo_set_t > NOO_HOLD_S:
      self.noo = ui_state.navigate_on_openpilot
    v = ui_state.sm["carState"].vEgo
    self.phase = maneuver_phase(self.nav.guidance, v, self.phase != "cruise") if route_on else "cruise"

    for ev in gui_app.mouse_events:
      if ev.slot == 0:
        self._handle_event(ev)
    if self.press_x is not None:
      self.slide += (self.slide_target - self.slide) * 0.5
    if self.confirm_t is not None:
      self.slide = min(self.slide_len, self.slide + max(30.0, (self.slide_len - self.slide) * 0.35))
      if now - self.confirm_t > 0.35:
        self._end_route()
        self.route_on = False
        self.slide, self.slide_target, self.confirm_t = 0.0, 0.0, None
    elif self.press_x is None and self.slide > 0:  # let go short of the end: spring back
      self.slide = max(0.0, self.slide - max(30.0, self.slide * 0.25))
      self.slide_target = 0.0

    split_on = self.split_on and self.route_on
    auto_open = split_on or self.phase != "cruise"
    if auto_open != self.last_auto:  # nav changed its mind: a hand override is done
      self.override, self.last_auto = None, auto_open
    if self.override == "open" and now - self.override_t > OVERRIDE_OPEN_S:
      self.override = None
    self.want_open = auto_open if self.override is None or split_on else self.override == "open"
    if self.want_open and self.open.b < 0.5 and self.override == "open":  # opened by hand: show the map's buttons a while
      self.map_poke_t = now
    self.open.set(1.0 if self.want_open else 0.0, now, EXPAND_S)
    self.map_ctrl.set(1.0 if now - self.map_poke_t < MAP_CONTROLS_S else 0.0, now, 0.3)
    has_lanes = bool(self._lanes()[0])
    self.lanes.set(1.0 if self.phase == "approach" and has_lanes and self.want_open else 0.0, now, EXPAND_S)
    self.split.set(1.0 if self.split_on else 0.0, now, EXPAND_S)
    self.noo_lit.set(1.0 if self.noo else 0.0, now, EXPAND_S)
    self.present.set(1.0 if self.route_on else 0.0, now, EXPAND_S)

    # the camera narrows inside its border for the always-on split's panel
    sp, pr = self.split.value(now), self.present.value(now)
    cam_w = lerp(rect.width, rect.width - PANEL_W - PANEL_GAP, sp * pr)
    self.camera_rect = rl.Rectangle(rect.x, rect.y, cam_w, rect.height)

  # *** drawing ***

  def draw(self, content: rl.Rectangle) -> rl.Rectangle:
    """Draws the card and its button; returns the camera view's rect for the alerts (beside the open card)."""
    now, rect = self._now, self._rect
    pr = self.present.value(now)
    if pr <= 0.001:  # no card: the HUD draws its own button (still, on a route's first frame)
      self.hit_exp = self.hit_layout = (0.0, 0.0, 0.0)
      self.hit_card = self.hit_eta = self.hit_map = NOWHERE
      return content
    dist, road, kind, left, eta = self._guidance_words()
    has_maneuver = self.nav.guidance.maneuver is not None
    lanes, here = self._lanes()
    has_lanes = bool(lanes)

    # over the camera the card grows and folds on a slow-start curve, so its contents can fade while it has barely
    # moved and are never seen cut off by its edges
    lt = self.present.linear(now)
    grow = smootherstep(lt)
    o, la, sp = self.open.value(now), self.lanes.value(now), self.split.value(now)
    o_fade, la_fade = self.open.fade(now), self.lanes.fade(now)
    if self.o_live is not None:  # following the finger, the lanes too (when there are any to show)
      o = o_fade = self.o_live
      if self.phase == "approach" and has_lanes:
        la = la_fade = self.o_live

    # sp blends the card over the camera (0) into the always-on split (1): the camera narrows inside its border,
    # the card slides into the panel and loses its background, the button moves to its stock spot in the camera
    W, H = rect.width, rect.height
    card_radius = 0.12 * (H - 2 * BORDER) / 2  # the status border's inner corner radius, shared by the card
    cw = lerp(CARD_W, PANEL_W, sp)
    full = rl.Rectangle(rect.x + BORDER, rect.y + BORDER, W - 2 * BORDER, H - 2 * BORDER)

    # layout of the card's contents first, so the small card fits them exactly
    pad_x, pad_top, pad_bot = lerp(33, 18, o) * (1 - sp), lerp(27, 18, o) * (1 - sp), lerp(24, 21, o) * (1 - sp)
    self.inner_r = lerp(card_radius - 18, card_radius, sp)
    road_size = 39
    # in the card the top-right button keeps one spot at the card's top right, drawn 136 px but tapped over the stock
    # 192 px circle, lining up with the turn section's padding in each state; both spots sit inside the stock button's
    # own touch area. In the split it's back in its stock spot in the camera, so the card's text gets the room back.
    exp_d = NAV_BUTTON
    exp_right = pad_x + lerp(0, 24, o)  # the button's right edge, in from the card's: the turn section's padding
    icon, dist_size, sec_pad = lerp(96, 100, o), lerp(78, 90, o), lerp(0, 24, o)
    tx_off = sec_pad + icon + 20
    text_w_max = (cw - exp_right - exp_d - 12) - (pad_x + tx_off)
    road_lines = self._cached(("road", road, round(text_w_max)), lambda: wrap(self._semi, road, fs(road_size), text_w_max, 3))
    text_h = dist_size + 6 + 46 * len(road_lines)
    sec_h = max(max(icon, text_h) + 2 * sec_pad, exp_d + 2 * lerp(0, 12, o))
    eta_h = lerp(2 + 21 + 48 + 6, 110, sp)
    small_h = 27 + max(96.0, 78 + 6 + 46 * len(road_lines), exp_d) + 21 + eta_h + 24
    self.drag_range = (full.height - 2 * CARD_MARGIN) - small_h  # px of finger travel from closed to fully open
    over = rl.Rectangle(full.x + full.width - CARD_MARGIN - CARD_W, full.y + CARD_MARGIN, CARD_W,
                        lerp(small_h, full.height - 2 * CARD_MARGIN, o))
    panel = rl.Rectangle(rect.x + W - PANEL_W + (1 - pr) * (PANEL_W + PANEL_GAP), rect.y, PANEL_W, H)  # slides in from the right
    card = rl.Rectangle(lerp(over.x, panel.x, sp), lerp(over.y, panel.y, sp), cw, lerp(over.height, panel.height, sp))
    # over the camera the card grows out of the experimental button's stock spot (and shrinks back into it): the
    # contents are laid out on the full card and revealed by the growing rect
    stock_btn = rl.Rectangle(full.x + full.width - BORDER - STOCK_BUTTON, full.y + BORDER, STOCK_BUTTON, STOCK_BUTTON)
    pr_h = grow * min(1.0, grow / 0.5)  # the height closes in sooner, so even a tall open card lands as a circle
    grown = rl.Rectangle(lerp(stock_btn.x, over.x, grow), lerp(stock_btn.y, over.y, grow), lerp(STOCK_BUTTON, over.width, grow),
                         lerp(STOCK_BUTTON, over.height, pr_h))
    vis = rl.Rectangle(lerp(grown.x, panel.x, sp), lerp(grown.y, panel.y, sp), lerp(grown.width, cw, sp),
                       lerp(grown.height, panel.height, sp))

    k = min(1.0, grow / 0.35)  # near the stock spot: a circle the button's size, folding into its disc
    vis_round = min(1.0, 2 * lerp(min(vis.width, vis.height) / 2, card_radius, k) / min(vis.width, vis.height))
    bg_a = min(1.0, grow / 0.12)  # the card's colour holds until the very end of the fold, then fades under the button
    if sp < 0.999:
      rl.draw_rectangle_rounded(vis, vis_round, 24, with_alpha(CARD, (1 - sp) * bg_a))
    # the contents fade out before the corners start rounding into the button's circle, then aren't drawn at all:
    # they're clipped to the card's square, which would show past the rounded corners (an open card's map does)
    gone = (1 - clamp01((lt - 0.75) / 0.25)) * (1 - sp)
    # and kept off the left and bottom corners (where the card cuts across them) so they never square those off
    inset = 0.0 if sp > 0.5 else min(vis.width, vis.height) / 2 * vis_round * min(1.0, (1 - grow) * 10)
    self.clip = (rl.Rectangle(vis.x + inset, vis.y, vis.width - inset, vis.height - inset) if gone < 0.99
                 else rl.Rectangle(vis.x, vis.y, 0, 0))
    rl.begin_scissor_mode(int(self.clip.x), int(self.clip.y), int(self.clip.width), int(self.clip.height))
    inner = rl.Rectangle(card.x + pad_x, card.y + pad_top, card.width - 2 * pad_x, card.height - pad_top - pad_bot)

    # turn section: plain in the small card, its own lighter card when open
    sec = rl.Rectangle(inner.x, inner.y, inner.width, sec_h)
    if o_fade > 0.01:
      rl.draw_rectangle_rounded(sec, min(1.0, 2 * self.inner_r / sec_h), 16, with_alpha(SUB, o_fade))
    # the turn's icon and text fade out while the slide-to-end track is showing (a long road name is taller than it)
    keep = 1.0 - (min(1.0, self.slide / 30.0) if self.route_on and sp < 0.5 else 0.0)
    if has_maneuver:
      draw_maneuver(sec.x + sec_pad + icon / 2, sec.y + sec_h / 2, icon, kind, left, with_alpha(rl.WHITE, keep))
    ty = sec.y + (sec_h - text_h) / 2
    tx = sec.x + tx_off
    self._text(self._bold, dist, tx, ty, fs(dist_size), with_alpha(NOW if self.phase == "turn" else rl.WHITE, keep))
    for i, line in enumerate(road_lines):
      self._text(self._semi, line, tx, ty + dist_size + 6 + 46 * i, fs(road_size), with_alpha(TEXT2, keep))

    # trip info, the same small and open: a rule, then arrival, time left, distance left (its own card in the split)
    eta_y = inner.y + inner.height - eta_h
    ex0, ew = inner.x + lerp(0, 15, o), inner.width - 2 * lerp(0, 15, o)
    if sp > 0.001:
      rl.draw_rectangle_rounded(rl.Rectangle(inner.x, eta_y, inner.width, eta_h), min(1.0, 2 * self.inner_r / eta_h), 16,
                                with_alpha(SUB, sp))
    if sp < 0.999:
      rl.draw_rectangle(int(inner.x), int(eta_y), int(inner.width), 2, with_alpha(RULE, 1 - sp))
    row_top, row_bot = lerp(eta_y + 2, eta_y, sp), lerp(card.y + card.height, eta_y + eta_h, sp)
    self.hit_eta = rl.Rectangle(inner.x, row_top - 20, inner.width, card.y + card.height - row_top + 20)
    size, sizes = self._cached(("eta", eta, round(ew)), lambda: self._eta_layout(eta, ew))
    for i, (v, vz) in enumerate(zip(eta, sizes, strict=True)):
      self._text(self._bold, v, ex0 + ew * (i + 0.5) / 3 - vz.x / 2, row_top + (row_bot - row_top - vz.y) / 2, fs(size), rl.WHITE)

    # map between them, with the lanes floating over its top so the map never moves
    gap = lerp(0, 18, o)  # the same gap between cards in both layouts
    map_r = rl.Rectangle(inner.x, sec.y + sec_h + gap, inner.width, max(0.0, eta_y - lerp(21, 18, sp) - (sec.y + sec_h + gap)))
    behind = mix(CARD, rl.BLACK, sp)
    self._draw_map(map_r, o_fade, behind)
    self.hit_card = card
    self.hit_layout = (0.0, 0.0, 0.0)
    self.hit_map = map_r if o_fade > 0.5 and map_r.height > 160 else NOWHERE
    ctrl = self.map_ctrl.value(now)
    if o_fade * ctrl > 0.01 and map_r.height > 160:  # switch between the card and the always-on split
      lcx, lcy = map_r.x + map_r.width - 24 - 48, map_r.y + map_r.height - 24 - 48
      rl.draw_circle(int(lcx), int(lcy), 48, with_alpha(MAP_BUTTON, o_fade * ctrl))
      draw_pin(lcx, lcy, sp, with_alpha(rl.WHITE, o_fade * ctrl))
      if ctrl > 0.5 and o_fade > 0.5:
        self.hit_layout = (lcx, lcy, 70)
    if map_r.height >= LANES_H:  # only once the map has room for them
      self._draw_lanes(map_r, lanes, here, la_fade * min(1.0, (map_r.height - LANES_H) / 120), lerp(-36, 0, la))
      mask_corners(map_r, self.inner_r, behind)
    if 0.01 < gone < 0.99:  # the contents fading as the card folds; its colour stays
      rl.draw_rectangle_rounded(vis, vis_round, 24, with_alpha(CARD, gone * bg_a))
    rl.end_scissor_mode()
    self.clip = rl.Rectangle(rect.x, rect.y, W, H)
    self._draw_button(now, pr, sp, grow, card, sec, sec_h, exp_right, stock_btn, content)

    # the alerts: across the camera, or beside the open card
    return rl.Rectangle(content.x, content.y, min(content.width, lerp(content.width, card.x - content.x, o)), content.height)

  def _eta_layout(self, eta: tuple, ew: float) -> tuple:
    """One size for all three, shrunk together until the widest fits its third (down to 36 px), and their sizes."""
    size = 48
    while size > 36 and max(measure_text_cached(self._bold, v, fs(size)).x for v in eta) > ew / 3 - 16:
      size -= 1
    return size, [measure_text_cached(self._bold, v, fs(size)) for v in eta]

  def _text(self, font: rl.Font, s: str, x: float, y: float, size: int, color: rl.Color):
    rl.draw_text_ex(font, s, rl.Vector2(x, y), size, 0, color)

  def _draw_map(self, m: rl.Rectangle, alpha: float, behind: rl.Color):
    if m.height < 4 or alpha <= 0.01:
      return
    rl.draw_rectangle_rounded(m, min(1.0, 2 * self.inner_r / min(m.width, m.height)), 16, with_alpha(MAP_BG, alpha))
    c = self.clip
    x0, y0 = max(m.x, c.x), max(m.y, c.y)
    x1, y1 = min(m.x + m.width, c.x + c.width), min(m.y + m.height, c.y + c.height)
    rl.begin_scissor_mode(int(x0), int(y0), int(max(0, x1 - x0)), int(max(0, y1 - y0)))
    self.map.render(m, self.nav, alpha)
    rl.end_scissor_mode()
    rl.begin_scissor_mode(int(c.x), int(c.y), int(c.width), int(c.height))  # back to the card's own clip
    mask_corners(m, self.inner_r, behind)

  def _draw_lanes(self, m: rl.Rectangle, lanes, here: int | None, a: float, slide: float):
    """Floating over the map's top on a fade: the lanes that take the turn lit (route blue with Navigate on openpilot
    on, white when it's only guiding), the rest dim, and the car under the lane we're in."""
    if a <= 0.01 or not lanes:
      return
    r = rl.Rectangle(m.x, m.y + slide, m.width, LANES_H)
    solid = int(r.height * 0.75)
    rl.draw_rectangle_gradient_v(int(r.x), int(r.y), int(r.width), solid, with_alpha(MAP_BG, 0.95 * a), with_alpha(MAP_BG, 0.75 * a))
    rl.draw_rectangle_gradient_v(int(r.x), int(r.y) + solid, int(r.width), int(r.height) - solid, with_alpha(MAP_BG, 0.75 * a),
                                 with_alpha(MAP_BG, 0.0))
    lit, dim = premul(LIT_BLUE if self.noo else rl.WHITE, a), premul(LANE_DIM, a)
    n = len(lanes)
    cw = min(126.0, (r.width - 24) / n)
    x0 = (r.width - n * cw) / 2
    size = min(66.0, cw * 0.62)
    with max_blend():  # the arrows' strokes overlap: faded one by one they'd double up where they cross
      for i, lane in enumerate(lanes):
        cx, cy = r.x + x0 + cw * (i + 0.5), r.y + 64
        if lane.arrow in ("upright", "upleft"):  # a shared lane lights only the branch the route takes
          split_arrow(cx, cy, size, lane.arrow == "upright", lit if lane.straight_lit else dim, lit if lane.turn_lit else dim)
        else:
          lane_arrow(cx, cy, size, lane.arrow, lit if lane.straight_lit or lane.turn_lit else dim)
        if i == here:
          draw_car(cx, r.y + 124, 44, premul(rl.WHITE, a))

  def _draw_button(self, now, pr, sp, grow, card, sec, sec_h, exp_right, stock_btn, content):
    """The top-right button, drawn after the lanes so its slide track covers what's under it."""
    exp_d = NAV_BUTTON
    c_x, c_y = card.x + card.width - exp_right - exp_d / 2, sec.y + sec_h / 2
    c_left = sec.x + (sec.x + sec.width) - (c_x + exp_d / 2)  # the track's left end mirrors the button's gap
    mix_t = self.present.linear(now)
    cam = (content.x + content.width - BORDER - STOCK_BUTTON / 2, content.y + BORDER + STOCK_BUTTON / 2)  # the stock spot
    if self.present.b >= 0.5:  # starting: the experimental button becomes the nav one
      # over the camera both icons ride one path from the stock spot to the card's, at their own (near equal) icon
      # sizes, crossfading over one disc that settles from the stock one into the card's; in the split the
      # experimental one rides the narrowing camera left while the nav one comes in with the panel
      mv = smoothstep(mix_t)  # the move spans the crossfade (the card's eased grow is over too soon)
      path = (lerp(stock_btn.x + STOCK_BUTTON / 2, c_x, mv), lerp(stock_btn.y + STOCK_BUTTON / 2, c_y, mv))
      ox, oy = (lerp(a, b, sp) for a, b in zip(path, cam, strict=True))
      bx, by = (lerp(a, b, sp) for a, b in zip(path, (c_x, c_y), strict=True))
      bd = lerp(lerp(STOCK_BUTTON, exp_d, mv), exp_d, sp)  # both shrink together from the stock size to the card's
      old_a, new_a = xfade(mix_t)
      if old_a > 0.01:
        if sp < 0.5:  # one disc under both icons
          rl.draw_circle(int(bx), int(by), bd / 2, mix(BLACK_TRANSLUCENT, NAV_DISC, new_a))
        else:
          rl.draw_circle(int(ox), int(oy), STOCK_BUTTON / 2, with_alpha(BLACK_TRANSLUCENT, old_a))
        self._draw_exp_icon(ox, oy, lerp(bd, STOCK_BUTTON, sp), old_a)
      if new_a > 0.01:
        self._draw_nav_button(bx, by, bd, old_a <= 0.01 or sp >= 0.5, c_left, new_a)
      self.hit_exp = (bx, by, STOCK_BUTTON / 2)
    else:  # ending: the slide to end already took the nav button away; the experimental one fades back in
      back = clamp01((0.45 - grow) / 0.3)  # on the card's own curve: fully back before the card's colour goes
      if back > 0.01:
        rl.draw_circle(int(cam[0]), int(cam[1]), STOCK_BUTTON / 2, with_alpha(BLACK_TRANSLUCENT, back))
        self._draw_exp_icon(cam[0], cam[1], STOCK_BUTTON, back)
      self.hit_exp = (0.0, 0.0, 0.0)

  def _slide_label_layout(self, label: str, width: float) -> tuple:
    px = 36
    while px > 28 and measure_text_cached(self._semi, label, fs(px)).x > width:  # ends before the button
      px -= 1
    return px, measure_text_cached(self._semi, label, fs(px))

  def _draw_exp_icon(self, cx: float, cy: float, d: float, alpha: float):
    tex = self._exp_icon if ui_state.sm["selfdriveState"].experimentalMode else self._wheel
    k = d * 0.72 / tex.width
    rl.draw_texture_ex(tex, rl.Vector2(cx - tex.width * k / 2, cy - tex.height * k / 2), 0, k, with_alpha(rl.WHITE, alpha))

  def _draw_nav_button(self, cx: float, cy: float, d: float, with_bg: bool, track_left: float, fade: float):
    """The nav button, or while slid left a "slide to end" track behind it, the button turning red and its road into
    an X as it nears the end."""
    self.slide_len = (cx - d / 2) - track_left
    frac = min(1.0, self.slide / self.slide_len)
    to_x = min(1.0, self.slide / (SLIDE_END * self.slide_len))  # 1 from the end point on: let go now and it ends
    shown = min(1.0, self.slide / 30.0) if self.route_on else 0.0
    if shown > 0:
      full = rl.Rectangle(cx - self.slide_len - d / 2, cy - d / 2, self.slide_len + d, d)
      rl.draw_rectangle_rounded(full, 1.0, 24, with_alpha(SLIDE_TRACK_BG, shown))
      label = tr("slide to end navigation")
      px, lz = self._cached(("slide", round(full.width - d * 1.5)), lambda: self._slide_label_layout(label, full.width - d * 1.5))
      lx = full.x + d * 0.4  # left aligned, in from the track's rounded end
      self._text(self._semi, label, lx, cy - lz.y / 2, fs(px), with_alpha(rl.WHITE, shown * 0.8 * (1 - frac)))
      cx -= self.slide
    if self.press_x is not None and self.dragged:
      d *= 1.06  # held: a touch bigger, as mici's slider
    if with_bg or to_x > 0:
      base = SLIDE_DISC if shown > 0 else (NAV_DISC if with_bg else BLACK_TRANSLUCENT)
      rl.draw_circle(int(cx), int(cy), d / 2, with_alpha(base, fade))
    lit = self.noo_lit.value(self._now)
    step = lit if self.noo_lit.b > 0.5 else 0.0  # turning on, the dashes move one step toward us; turning off, still
    draw_road_icon(cx, cy, d, with_alpha(mix(DIM, LIT_BLUE, lit), fade), step, to_x, END_RED, rl.WHITE)
