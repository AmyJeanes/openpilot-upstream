"""The comma four nav UI's state logic: the driving view's card by distance, its lanes and lane requests, the road
button's tap and slide, and the pager's map page. The whole UI against scripted navigation is
selfdrive/ui/tests/mici_nav_player.py (its scenarios check the same in place, with touches)."""
from openpilot.selfdrive.ui.mici.layouts.main import MAP_PAGE_TIMEOUT, keep_view_shift, onroad_tap_target, page_timeout
from openpilot.selfdrive.ui.mici.onroad.alert_renderer import Alert, AlertSize
from openpilot.selfdrive.ui.mici.onroad.nav_card import (EM_BIG, MARGIN, REQUEST_OFF_S, REQUEST_ON_S, UNDER_ALERT_Y, Debounce, card_layout,
                                                         card_mode, lane_request, lanes_lit, request_alert)
from openpilot.selfdrive.ui.mici.widgets.road_button import FINISH_S, GROW_PX, SLIDE_END, TAP_SLOP, SlideToEnd
from openpilot.selfdrive.ui.nav.nav_state import CarLane, Guidance, Lane, Maneuver, card_lanes, maneuver_phase
from openpilot.selfdrive.ui.ui_state import UIStatus

RIGHT_TURN_LANES = [Lane(["straight"], False, "none", True), Lane(["straight"], False, "none", True),
                    Lane(["straight"], False, "none", False), Lane(["straight"], False, "none", False),
                    Lane(["straight", "right"], True, "right", False), Lane(["right"], True, "right", False)]
PRE_LANE_CHANGE = Alert("Steer Right", "Confirm Lane Change", AlertSize.mid, 0, alert_type="preLaneChangeRight/warning")
TAKE_CONTROL = Alert("take control", "turn exceeds limit", AlertSize.mid, 1, alert_type="steerSaturated/warning")


def guidance(distance: float | None, kind: str = "turn", lanes: bool = True) -> Guidance:
  g = Guidance()
  if distance is not None:
    g.maneuver = Maneuver(distance, kind, "right", "Vespucci Blvd")
  if lanes:
    g.lanes, g.show_lanes = RIGHT_TURN_LANES, True
  return g


def mode_at(distance: float | None, v: float = 20.0, kind: str = "turn", alert: Alert | None = None, was: str = "cruise") -> str:
  g = guidance(distance, kind)
  phase = maneuver_phase(g, v, was != "cruise")
  return card_mode(g, phase, bool(card_lanes(g)[0]), alert)


def test_card_by_distance():
  assert mode_at(None) == "hidden"  # a route with no maneuver: nothing to show
  assert mode_at(1.5 * 1609.0) == "chip"
  assert mode_at(0.4 * 1609.0) == "chip"
  assert mode_at(300.0) == "card"  # 15 s at 20 m/s
  assert mode_at(320.0) == "chip"
  assert mode_at(320.0, was="approach") == "card"  # it closes a little farther out than it opens, so it doesn't flap
  assert mode_at(150.0, v=5.0) == "card"  # slow: from 200 m
  assert mode_at(100.0) == "card"
  assert mode_at(15.0, v=7.0) == "now"


def test_arrival_is_a_chip():
  assert mode_at(300.0, kind="arrive") == "chip"
  assert mode_at(10.0, v=5.0, kind="arrive") == "now"


def test_alerts_take_the_card():
  assert mode_at(300.0, alert=PRE_LANE_CHANGE) == "lanes"  # a lane change prompt: the lanes only, under it
  assert mode_at(300.0, alert=request_alert("right")) == "lanes"
  assert mode_at(300.0, alert=TAKE_CONTROL) == "hidden"  # any other alert: no card
  g = guidance(300.0, lanes=False)
  assert card_mode(g, "approach", False, PRE_LANE_CHANGE) == "hidden"  # nothing to show under the prompt


def test_card_layout():
  chip = card_layout("chip", 120.0, 140.0, 0.0)
  card = card_layout("card", 120.0, 140.0, 176.0)
  lanes = card_layout("lanes", 120.0, 140.0, 176.0)
  assert (chip.big, chip.lanes, chip.y) == (0.0, 0.0, MARGIN)
  assert card.big == 1.0 and card.lanes == 1.0 and card.w > chip.w and card.h > chip.h
  assert card_layout("card", 120.0, 140.0, 0.0).lanes == 0.0  # no lanes for this maneuver: just the bigger turn row
  assert (lanes.row, lanes.lanes, lanes.y) == (0.0, 1.0, UNDER_ALERT_Y)
  assert EM_BIG == 56.0  # the one glance-size line on this screen (17 arcmin capitals)


def test_lane_highlighting():
  lanes, here = card_lanes(guidance(300.0), CarLane(1, 4, True))
  assert [lane.arrow for lane in lanes] == ["up", "up", "upright", "right"]  # ours only, left to right
  assert [lane.turn_lit for lane in lanes] == [False, False, True, True]
  assert not lanes[2].straight_lit  # the shared lane lights only its turning branch
  assert here == 1
  assert card_lanes(guidance(300.0), CarLane(1, 3, True))[1] is None  # the model counts other lanes: no car
  assert lanes_lit(True, UIStatus.ENGAGED)
  assert lanes_lit(True, UIStatus.OVERRIDE)
  assert not lanes_lit(True, UIStatus.DISENGAGED)  # openpilot won't make the move: white
  assert not lanes_lit(False, UIStatus.ENGAGED)  # guidance only: white


def test_lane_request():
  g = guidance(250.0)
  lanes, _ = card_lanes(g)
  assert lane_request(g, lanes, 0, "approach") == "right"
  assert lane_request(g, lanes, 1, "approach") == "right"
  assert lane_request(g, lanes, 2, "approach") is None  # in a lane that takes it
  assert lane_request(g, lanes, None, "approach") is None  # lane unknown: never ask
  assert lane_request(g, lanes, 0, "cruise") is None  # not near yet
  assert lane_request(g, lanes, 0, "turn") is None  # too late
  g.lane_open_distance = 40.0
  assert lane_request(g, lanes, 0, "approach") is None  # the turn lane hasn't opened yet
  left = Guidance(maneuver=Maneuver(250.0, "turn", "left", ""), show_lanes=True,
                  lanes=[Lane(["left"], True, "left", False), Lane(["straight"], False, "none", False),
                         Lane(["straight"], False, "none", False)])
  assert lane_request(left, card_lanes(left)[0], 2, "approach") == "left"


def test_request_alert():
  a = request_alert("right")
  assert (a.text1, a.text2, a.size) == ("move right", "signal to confirm", AlertSize.small)
  assert a.alert_type.startswith("navLaneChangeRight")  # the turn signal icon and the small height, as mici's prompts
  assert request_alert("left").alert_type.startswith("navLaneChangeLeft")


def test_request_debounce():
  d = Debounce(REQUEST_ON_S, REQUEST_OFF_S)
  assert d.update("right", 0.0) is None
  assert d.update("right", REQUEST_ON_S - 0.05) is None
  assert d.update("right", REQUEST_ON_S + 0.01) == "right"
  assert d.update(None, 2.0) == "right"  # a moment's dropout doesn't take it away
  assert d.update("right", 2.1) == "right"
  assert d.update(None, 3.0) == "right"
  assert d.update(None, 3.0 + REQUEST_OFF_S + 0.01) is None


def test_slide_tap():
  s = SlideToEnd()
  s.length = 400.0
  s.press(472.0)
  s.move(472.0 - TAP_SLOP + 2)  # the button follows the finger, but within the slop it's still a tap
  assert s.release(1.0) == "tap"
  for i in range(30):
    s.step(1.0 + i / 30, 1 / 30)
  assert s.offset == 0.0


def test_slide_springs_back():
  s = SlideToEnd()
  s.length = 400.0
  s.press(472.0)
  s.move(472.0 - 0.5 * 400.0)
  assert s.offset == 200.0 and s.grow == 1.0 and 0.6 < s.morph < 0.7
  assert s.release(1.0) == "back"
  for i in range(30):
    assert not s.step(1.0 + i / 30, 1 / 30)
  assert s.offset == 0.0 and s.grow == 0.0


def test_slide_ends_past_the_end_point():
  s = SlideToEnd()
  s.length = 400.0
  s.press(472.0)
  s.move(472.0 - GROW_PX / 2)
  assert 0.4 < s.grow < 0.6  # it grows as the slide starts
  s.move(472.0 - SLIDE_END * 400.0)
  assert s.morph == 1.0  # the X complete at the end point
  assert s.release(1.0) == "end"
  s.press(300.0)
  assert not s.held  # no new press while it finishes
  ended = [s.step(1.0 + i / 30, 1 / 30) for i in range(1, int(FINISH_S * 30) + 3)]
  assert ended.count(True) == 1
  assert s.offset == 0.0


def test_slide_needs_a_press():
  s = SlideToEnd()
  s.move(100.0)
  assert s.release(0.0) == "" and s.offset == 0.0


def test_pager():
  assert onroad_tap_target(True) == "map"
  assert onroad_tap_target(False) == "home"
  assert page_timeout(True) == MAP_PAGE_TIMEOUT == 15
  assert page_timeout(False) is None
  # pages [alerts 0][home 1][map 2][driving 3]: the map page coming or going keeps the driving view still
  assert keep_view_shift(2, 3, appeared=True, width=536) == -536
  assert keep_view_shift(2, 3, appeared=False, width=536) == 536
  assert keep_view_shift(2, 1, appeared=True, width=536) == 0  # home, before it: nothing moves
  assert keep_view_shift(2, 2, appeared=False, width=536) == 0  # it was in view: driving slides into its place
