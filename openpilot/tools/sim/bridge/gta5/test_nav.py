import numpy as np

import openpilot.tools.sim.bridge.gta5.navd.planner as nav_mod
from openpilot.tools.sim.bridge.gta5.navd.planner import Fork, Planner, find_turn, lane_plan
from openpilot.tools.sim.bridge.gta5.gta5_driver import HumanDriver
from openpilot.tools.sim.bridge.gta5.gta5_navd import bridge_tune, nav_inputs
from openpilot.tools.sim.bridge.gta5.map.paths import Link
from openpilot.tools.sim.bridge.gta5.map.router import Route


class Clock:
  def __init__(self):
    self.t = 100.0

  def monotonic(self):
    return self.t


def route_to_turn(before: float, side: str = "right", after: float = 100.0) -> np.ndarray:
  """Points every 5 m north for `before` m, then east (right) or west for `after` m."""
  ahead = [(0.0, y) for y in np.arange(0.0, before, 5.0)]
  sign = 1.0 if side == "right" else -1.0
  return np.array(ahead + [(sign * x, before) for x in np.arange(0.0, after, 5.0)])


class Drive:
  """navd's planner driven along a route by a car whose driver does as it's asked, northwards at a steady speed."""
  def __init__(self, lane, v=8.0):
    self.clock = Clock()
    self.sent, self.desires, self.events = [], [], []
    self.indicator = None
    self.nav = Planner(bridge_tune())
    self.human = HumanDriver(self._send, lambda: None)
    self.lane, self.v = lane, v
    self.drives = True  # Navigate on openpilot
    self.stage = 3  # PLAN D's: auto-approved, as GTA runs
    self.request = None  # nav's request to the driver, as of the last step

  def update(self, state: dict, engaged: bool, indicator: str | None, desire: dict) -> bool:
    """A step from the bridge's state, as the world takes it: the driver's game commands to `sent`, NavDesire to
    `desires`, and the game's waypoint off on arriving."""
    out = self.nav.update(nav_inputs(state, engaged, indicator, desire, self.clock.t, self.drives, self.stage))
    self.request = out.request
    self.human.act(out)
    self.desires += out.desires
    self.events += out.events
    if out.arrived:
      self._send({"type": "waypoint", "off": True})
    return out.arrived

  def _send(self, m):
    self.sent.append(m)
    self.indicator = m.get("side") if m["type"] == "setIndicator" else None if m["type"] == "indicatorOff" else self.indicator

  def step(self, route, y, extra=None, dt=0.05):
    self.clock.t += dt
    state = {"vEgo": self.v, "pos": [0.0, y, 0.0], "heading": 0.0, "yawRate": 0.0, "lane": list(self.lane),
             "route": [[0.0, 0.0]] + [p for p in (route - [0.0, y]).tolist() if p[1] > 0.0 or p[0] != 0.0], **(extra or {})}
    return self.update(state, True, self.indicator, {})


def signals(d: Drive):
  return [m["side"] for m in d.sent if m["type"] == "setIndicator"]


def test_find_turn():
  t = find_turn(route_to_turn(100.0, "left"))
  assert t is not None and t.side == "left" and abs(t.dist - 100.0) < 6
  assert find_turn(np.array([(0.0, y) for y in range(0, 200, 5)], dtype=float)) is None


def test_find_turn_on_its_last_segment():
  # nodes 10 m apart: from 15 m out, the turn's node is the route's first point ahead
  for before in (25.0, 15.0, 12.0, 8.0, 6.0):
    t = find_turn(np.array([(0.0, 0.0), (0.0, before), (30.0, before), (60.0, before)]), 5.0)
    assert t is not None and t.side == "right" and abs(t.dist - before) < 0.1
  assert find_turn(np.array([(0.0, 0.0), (0.0, 4.0), (30.0, 4.0), (60.0, 4.0)]), 5.0) is None


def test_find_turn_off_a_junction_link():
  """A jog across a junction (a turn onto its short link, then the other way off it) is no turn where the way on, from
  the road before the link, goes straight on or the other way; a real turn off the link still is."""
  def jog(first: float, second: float, link: float = 10.0) -> np.ndarray:
    pts, h = [np.array([0.0, 0.0]), np.array([0.0, 20.0])], 0.0
    for turn, length in ((first, link), (second, 100.0)):
      h += turn
      pts.append(pts[-1] + length * np.array([-np.sin(np.radians(h)), np.cos(np.radians(h))]))
    return np.array(pts)
  assert find_turn(jog(-90.0, 90.0), 5.0) is None  # straight across
  assert find_turn(jog(-90.0, 53.0), 5.0) is None  # bearing right, which a left off the link would miss
  t = find_turn(jog(-25.0, 78.0), 5.0)  # a left turn, past a bear right onto the link
  assert t is not None and t.side == "left"
  t = find_turn(jog(-90.0, 53.0, 40.0), 5.0)  # a link too long to be across a junction: two turns
  assert t is not None and t.side == "right"


def test_entry_signal_past_the_entry():
  # sparse approach nodes, signalling 3 m past the default entry (12 m out): the turn must still be found there
  route = np.array([(0.0, y) for y in np.arange(0.0, 200.0, 20.0)] + [(x, 200.0) for x in np.arange(0.0, 100.0, 10.0)])
  d = Drive((1, 2), v=5.0)
  d.nav.tune = bridge_tune("")
  d.nav.tune.values.update(signal_mode="entry", signal_entry_offset=3.0)
  y = 0.0
  while y < 200.0 and not d.nav.signaling:
    d.step(route, y, {"routeEnd": 300.0 - y})
    y += d.v * 0.05
  assert signals(d) == ["right"] and 200.0 - y < 13.0


def test_turn_from_its_lane():
  route = route_to_turn(300.0)
  d = Drive((1, 2))
  y = 0.0
  while y < 300.0 and not d.nav.signaling:
    d.step(route, y)
    y += d.v * 0.05
  assert signals(d) == ["right"]


def test_lane_change_early_enough_for_two():
  route = route_to_turn(400.0)
  d = Drive((0, 3), v=15.0)
  y, started = 0.0, None
  while y < 380.0:
    d.step(route, y)
    if d.nav.changing and started is None:
      started = 400.0 - y
    y += d.v * 0.05
  # two changes of 8 s at 15 m/s, plus 4 s, back from 30 m before the turn
  assert started is not None and started >= 2 * 8 * 15 + 30 - 5


def test_wrong_lane_turn_left_to_route():
  route = route_to_turn(40.0)
  d = Drive((0, 2), v=3.0)  # too near to change lanes for it
  d.nav.tune.values["turn_into_lane"] = 0.0  # today's rule; turn_into_lane signals it for n16 (test_slow_beside_the_turn_signals_it)
  y, turned = 0.0, False
  while y < 38.0:
    d.step(route, y, {"laneFrac": 0.0})  # surely in the left lane
    turned |= d.nav.turn is not None
    y += d.v * 0.05
  assert not turned and len(d.nav.skipped) == 1


def test_fork_lanes():
  assert Fork(100.0, "right", 1, 2, True).lanes(2) == (1, 1)
  assert Fork(100.0, "left", 2, 3, True).lanes(3) == (0, 1)
  assert Fork(100.0, "left", 1, 1, False).lanes(2) == (0, 1)  # one lane in: any lane
  assert Fork(100.0, "left", 4, 4, True, other=1).lanes(4) == (0, 1)  # GTA counts the exit's lane on top
  assert Fork(100.0, "right", 2, 2, False, slip=True).lanes(2) == (0, 1)  # a bay opening: any lane
  assert not Fork(100.0, "left", 2, 2, True, other=1).keep  # staying on the road past a smaller branch: no keep desire
  assert Fork(100.0, "right", 1, 3, True, other=2).keep  # leaving it


def test_link_lanes():
  two_way = Link([0, 0x60, (2 << 5) | (2 << 2)])  # 2+2, offset 6/7 of half a lane
  assert two_way.lane(2.0) == -1 or two_way.inner > 2.0
  assert two_way.lane(two_way.inner + 1.0) == 0 and two_way.lane(two_way.inner + 6.0) == 1
  assert two_way.lane(-6.0) == -2
  one_way = Link([0, 0, 2 << 5])
  assert one_way.lane(-2.0) == 0 and one_way.lane(2.0) == 1 and one_way.lane(-20.0) == 0


def test_lane_change_param_before_blinker():
  # openpilot reads NavDesire every 0.2 s: the blinker before it would ask for a turn below 19 mph
  route = route_to_turn(200.0)
  d = Drive((0, 2), v=7.0)
  d.nav.tune.values["model_lc_speed"] = 0.0  # today's timing: these mechanics run below n16's lane change speed
  events, last_desire, last_ind = [], "", None
  y = 0.0
  while y < 190.0:
    d.step(route, y)
    if d.nav.desire != last_desire:
      events.append((d.clock.t, "desire", d.nav.desire))
      last_desire = d.nav.desire
    if d.indicator != last_ind:
      events.append((d.clock.t, "blinker", d.indicator))
      last_ind = d.indicator
    if d.nav.changing and d.indicator and y > 50:
      d.indicator = None  # the bridge cancels it once the lane change is done
      d.lane = (1, 2)
    y += d.v * 0.05
  on = next(t for t, kind, v in events if kind == "desire" and v == "laneChange")
  blink = next(t for t, kind, v in events if kind == "blinker" and v == "right")
  assert blink - on >= 0.35
  off = next(t for t, kind, v in events if kind == "desire" and v == "" and t > blink)
  blink_off = next(t for t, kind, v in events if kind == "blinker" and v is None and t > blink)
  assert off - blink_off >= 0.45


def test_lane_change_into_turn():
  # a lane change for a left turn still going lane_change_last before it: the blinker stays on, now for the turn
  route = route_to_turn(120.0, "left")
  d = Drive((0, 2), v=5.0)
  d.nav.tune.values["lane_change_into_turn"] = 30.0
  d.nav.tune.values["model_lc_speed"] = 0.0  # today's timing: these mechanics run below n16's lane change speed

  y, changed_at, turned_at = 0.0, None, None
  while y < 100.0:
    if 120.0 - y < 47.0:
      d.lane = (1, 3)  # a lane for the turn opens on the left
    d.step(route, y)
    if d.nav.changing and changed_at is None:
      changed_at = 120.0 - y
    if d.nav.turn is not None and turned_at is None:
      turned_at = 120.0 - y
      assert d.indicator == "left" and d.nav.desire == ""
    y += d.v * 0.05
  assert changed_at is not None and turned_at is not None and 25.0 < turned_at <= 31.0
  assert signals(d) == ["left", "left"] and not any(m["type"] == "indicatorOff" for m in d.sent)


def test_turn_straight_after_a_turn():
  # left, then right 35 m on: the second is signalled without waiting out a cooldown
  first = route_to_turn(100.0, "left", after=0.0)
  route = np.concatenate((first, [(-x, 100.0) for x in np.arange(0.0, 35.0, 5.0)], [(-35.0, 100.0 + y) for y in np.arange(5.0, 100.0, 5.0)]))
  sides = []
  d = Drive((0, 1), v=5.0)
  pos, heading = np.array([0.0, 0.0]), 0.0
  for _ in range(2000):
    d.clock.t += 0.05
    along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))))
    s = along[int(np.argmin(np.hypot(*(route - pos).T)))]
    ahead = [pos.tolist()] + [p.tolist() for p, a in zip(route, along, strict=True) if a > s + 0.5]
    state = {"vEgo": d.v, "pos": [*pos, 0.0], "heading": heading, "yawRate": 0.0, "lane": [0, 1], "route": ahead, "routeEnd": along[-1] - s}
    d.update(state, True, d.indicator, {})
    if d.nav.signaled and (not sides or sides[-1] != d.nav.signaled):
      sides.append(d.nav.signaled)
    # follow the route, heading along it
    nxt = np.array(ahead[1]) if len(ahead) > 1 else pos
    step = nxt - pos
    if np.hypot(*step) < 1e-6:
      break
    heading = float(np.degrees(np.arctan2(-step[0], step[1])))
    pos = pos + step / np.hypot(*step) * min(d.v * 0.05, np.hypot(*step))
  assert sides == ["left", "right"]


def test_left_turn_into_its_bay():
  # a bay opening 30 m before a left turn, from the inside lane: into the bay, then signal the turn (at 6 m/s, below
  # n16's lane change speed, only with model_lc_speed off: test_slow_bay_left_to_the_turn_signal)
  route = route_to_turn(200.0, "left")
  d = Drive((0, 2), v=6.0)
  d.nav.tune.values["model_lc_speed"] = 0.0
  y, order = 0.0, []
  while y < 195.0:
    bay_at = 170.0 - y
    forks = [[bay_at, "right", 2, 2, False, 0, True]] if bay_at > 0 else []
    d.step(route, y, {"forks": forks, "routeEnd": 300.0 - y})
    if d.nav.changing == "left" and "bay" not in order:
      order.append("bay")
      assert 200.0 - y > 18.0
    if d.nav.changing and d.indicator == "left" and y > 0 and "bay" in order and 200.0 - y < 26:
      d.indicator, d.lane = None, (-1, 2)  # in the bay
    if d.nav.signaled == "left" and "turn" not in order:
      order.append("turn")
    y += d.v * 0.05
  assert order == ["bay", "turn"]


def test_slow_bay_left_to_the_turn_signal():
  """The same bay at 6 m/s with n16's lane change speed (10 m/s): no lane change into it (n16 would take the desire
  like a turn's, at the wrong place); the turn's signal comes, its desire takes the car over."""
  route = route_to_turn(200.0, "left")
  d = Drive((0, 2), v=6.0)
  y, changed, signalled = 0.0, False, False
  while y < 195.0:
    bay_at = 170.0 - y
    d.step(route, y, {"forks": [[bay_at, "right", 2, 2, False, 0, True]] if bay_at > 0 else [], "routeEnd": 300.0 - y})
    changed |= d.nav.changing is not None
    signalled |= d.nav.signaled == "left"
    y += d.v * 0.05
  assert not changed and signalled


def test_no_keep_left_on_a_two_way_road():
  # a left fork on a two-way road (L2): keepLeft would take the car over into the oncoming lanes
  route = route_to_turn(300.0)
  for two_way, expect in ((True, False), (False, True)):
    d = Drive((0, 2), v=8.0)
    asked = False
    for k in range(60):
      d.step(route, k * 0.4, {"forks": [[50.0 - k * 0.4, "left", 1, 2, True, 1, False]], "twoWay": two_way, "routeEnd": 400.0})
      asked |= d.nav.desire == "keepLeft"
    assert asked == expect


def test_doubtful_lane_still_signals():
  # nav's lane says the car is beside the right turn's lane, but its position is between the two (the model ended the
  # lane change early, R1): signal the turn rather than leave it
  route = route_to_turn(60.0)
  for frac, signalled in ((0.75, True), (0.0, False)):
    d = Drive((0, 2), v=5.0)
    d.nav.tune.values["turn_into_lane"] = 0.0  # today's rule; turn_into_lane signals it for n16 (test_slow_beside_the_turn_signals_it)
    y = 0.0
    while y < 45.0:
      d.step(route, y, {"laneFrac": frac, "routeEnd": 200.0})
      y += d.v * 0.05
    assert (d.nav.turn is not None or d.nav.signaled is not None or not d.nav.skipped) == signalled


def test_turn_done_at_its_way_out_without_repulse():
  # once the car has turned, the blinker drops at the way out even while still yawing, and isn't pulsed again
  route = route_to_turn(40.0, "left")
  d = Drive((0, 1), v=5.0)
  d.nav.tune = bridge_tune("")
  d.nav.tune.values.update(left_signal_max_entry=0.0)  # signalled at 25 m, without a stop line
  for k in range(60):
    d.step(route, k * 0.25)
  assert d.nav.signaled == "left"
  repeats = d.nav.repeat_t
  # the car turns: heading 0 -> 85 (way out 90), yawing 0.5 rad/s; the route ahead now runs west from the car
  for k, h in enumerate(np.linspace(0, 85, 40)):
    d.clock.t += 0.1
    state = {"vEgo": 5.0, "pos": [0.0, 40.0, 0.0], "heading": float(h), "yawRate": 0.5, "lane": [0, 1], "routeEnd": 200.0,
             "route": [[-x, 40.0] for x in np.arange(0.0, 100.0, 5.0)]}
    d.update(state, True, d.indicator, {"left": 1.0 if k < 30 else 0.05})
  assert d.nav.signaled is None and d.nav.repeat_t == repeats
  assert d.nav.cue == "keepRight"



def test_exit_cue_stacked_within_the_turn():
  # exit_cue_at: the keepRight starts halfway round, stacked on the turn while its blinker is on, then plain past it
  route = route_to_turn(40.0, "left")
  d = Drive((0, 1), v=5.0)
  d.nav.tune = bridge_tune("")
  d.nav.tune.values.update(exit_cue_at=0.5, left_signal_max_entry=0.0)
  for k in range(60):
    d.step(route, k * 0.25)
  assert d.nav.signaled == "left" and "+keepRight" not in d.desires
  for k, h in enumerate(np.linspace(0, 85, 40)):
    d.clock.t += 0.1
    state = {"vEgo": 5.0, "pos": [0.0, 40.0, 0.0], "heading": float(h), "yawRate": 0.5, "lane": [0, 1], "routeEnd": 200.0,
             "route": [[-x, 40.0] for x in np.arange(0.0, 100.0, 5.0)]}
    d.update(state, True, d.indicator, {"left": 1.0})
    if h < 40:
      assert d.nav.desire != "+keepRight"
  assert "+keepRight" in d.desires and d.nav.signaled is None and d.desires[-1] == "keepRight"


def test_left_signal_waits_for_the_stop_line():
  # left_signal_max_entry: a left turn's signal waits until its stop line (20 m before the turn) is 3 m off; never sooner
  route = route_to_turn(100.0, "left")
  at = {}
  for wait in (0.0, 3.0):
    d = Drive((0, 2), v=5.0)
    d.nav.tune = bridge_tune("")
    d.nav.tune.values.update(left_signal_max_entry=wait)
    y = 0.0
    while y < 100.0 and not signals(d):
      d.step(route, y, {"stops": [80.0 - y], "routeEnd": 300.0 - y})
      y += d.v * 0.05
    at[wait] = 80.0 - y
  assert at[0.0] > 5.0 and 2.0 < at[3.0] <= 3.0
  # not held just after a lane change towards it, whose desire would carry on into the junction instead
  d = Drive((0, 2), v=5.0)
  d.nav.tune = bridge_tune("")
  d.nav.tune.values.update(left_signal_max_entry=3.0)
  y = 0.0
  while y < 100.0 and not signals(d):
    d.nav.change_side, d.nav.change_t = "left", d.clock.t - 2.0
    d.step(route, y, {"stops": [80.0 - y], "routeEnd": 300.0 - y})
    y += d.v * 0.05
  assert abs((80.0 - y) - at[0.0]) < 1.0


def test_unturned_turn_given_up_past_its_point():
  # unturned_cancel: a car that drives straight on past the turn gets no more pulses past its point, and loses the
  # signal this far past it instead of at MISSED_BY
  route = route_to_turn(40.0, "left")
  gone = {}
  for cancel in (0.0, 10.0):
    d = Drive((0, 1), v=5.0)
    d.nav.tune = bridge_tune("")
    d.nav.tune.values.update(unturned_cancel=cancel, unturned_keep_pulses=False, left_signal_max_entry=0.0)
    y, pulses = 0.0, []
    while y < 100.0 and (d.nav.signaled is not None or not signals(d)):
      d.step(route, y, {"routeEnd": 300.0 - y})
      if d.nav.signaled is not None and y > 40.0:
        pulses.append(d.nav.repeat_t)
      y += d.v * 0.05
    gone[cancel] = y - 40.0
    if cancel:
      assert len(set(pulses)) <= 1
  assert 9.0 < gone[10.0] < 12.0 and gone[0.0] > 30.0, gone


def test_unturned_cancel_can_keep_pulses():
  route = route_to_turn(40.0, "left")
  d = Drive((0, 1), v=5.0)
  d.nav.tune = bridge_tune("")
  d.nav.tune.values.update(unturned_cancel=25.0, unturned_keep_pulses=True, left_signal_max_entry=0.0)
  y, pulses = 0.0, []
  while y < 100.0 and (d.nav.signaled is not None or not signals(d)):
    d.step(route, y, {"routeEnd": 300.0 - y})
    if d.nav.signaled is not None and y > 40.0:
      pulses.append(d.nav.repeat_t)
    y += d.v * 0.05
  assert 24.0 < y - 40.0 < 27.0 and len(set(pulses)) > 1, (y, pulses)


def test_junction_entry():
  from openpilot.tools.sim.bridge.gta5.navd.planner import junction_entry
  assert junction_entry(100.0, [30.0, 85.0], [100.0]) == (85.0, "stop line")
  assert junction_entry(100.0, [], [65.0, 82.0, 92.0, 100.0]) == (82.0, "junction")  # back while the nodes are close
  assert junction_entry(100.0, [], [100.0]) == (85.0, "default")


def test_tune_reloads(tmp_path=None):
  import json as _json
  import os as _os
  import tempfile
  from openpilot.tools.sim.bridge.gta5.navd.planner import Tune
  path = _os.path.join(tmp_path or tempfile.mkdtemp(), "tune.json")
  t = Tune(path)
  assert t.signal_mode == "time" and t.changed() == {}
  with open(path, "w") as f:
    _json.dump({"signal_mode": "entry", "turn_speed_square": 4.0, "nonsense": 1}, f)
  assert t.refresh(10.0) and t.signal_mode == "entry" and t.turn_speed_square == 4.0
  assert t.changed() == {"signal_mode": "entry", "turn_speed_square": 4.0}
  assert not t.refresh(10.5)  # checked at most every second
  _os.utime(path, (1, 1))
  with open(path, "w") as f:
    _json.dump({}, f)
  assert t.refresh(12.0) and t.changed() == {}


def test_signal_at_the_junction_entry():
  # signal_mode entry: the blinker comes on as the car passes the stop line (15 m before the turn's node), not 5 s out
  import os as _os
  import tempfile
  import json as _json
  path = _os.path.join(tempfile.mkdtemp(), "tune.json")
  with open(path, "w") as f:
    _json.dump({"signal_mode": "entry", "signal_entry_offset": 1.0}, f)
  route = route_to_turn(120.0)
  d = Drive((1, 2), v=5.0)
  d.nav.tune = bridge_tune(path)
  y, at = 0.0, None
  while y < 118.0 and at is None:
    d.step(route, y, {"stops": [105.0 - y], "junctions": [120.0 - y], "routeEnd": 300.0 - y})
    if d.nav.signaled:
      at = 120.0 - y
    y += d.v * 0.05
  assert at is not None and 12.0 < at < 14.5, at


def lane_route(points, links: list) -> Route:
  r = Route(np.array(points, dtype=float), [])
  r.links = links if isinstance(links, list) else [links] * (len(points) - 1)
  return r


def lane_line(r: Route, lane, v=8.0) -> np.ndarray:
  return r.lane_line(lane_plan(r.rest(), [], lane, r.lanes_at, v))


def x_at(line: np.ndarray, y: float) -> float:
  return float(line[np.argmin(np.abs(line[:, 1] - y) + 1000 * (np.abs(line[:, 0]) > 20))][0])


def test_lane_line_moves_over_for_a_right_turn():
  two = Link([0, 0, 2 << 5])  # one-way, two lanes centred on the line
  line = lane_line(lane_route(route_to_turn(200.0), two), (0, 2))
  assert abs(x_at(line, 20.0) + 2.75) < 0.1  # the left lane
  assert abs(x_at(line, 175.0) - 2.75) < 0.1  # the right one, changed by 30 m before
  east = line[line[:, 0] > 40.0]
  assert len(east) and np.allclose(east[:, 1], 200.0 - 2.75, atol=0.1)  # and out of the turn in the right lane


def test_lane_line_arrives_left_from_a_left_turn():
  two = Link([0, 0, 2 << 5])
  line = lane_line(lane_route(route_to_turn(200.0, "left"), two), (0, 2))
  assert abs(x_at(line, 100.0) + 2.75) < 0.1  # already in the turn's lane
  west = line[line[:, 0] < -40.0]
  assert len(west) and np.allclose(west[:, 1], 200.0 - 2.75, atol=0.1)


def test_lane_line_across_links_without_lanes():
  # a junction's insides between a two-lane road and a three-lane one: the offset runs evenly, not to the road's line
  pts = [(0.0, y) for y in np.arange(0.0, 200.0, 5.0)]
  links = [Link([0, 0, 2 << 5])] * 10 + [None] * 5 + [Link([0, 0, 3 << 5])] * (len(pts) - 16)
  line = lane_line(lane_route(pts, links), (1, 2))
  xs = [x_at(line, y) for y in (40.0, 50.0, 60.0, 65.0, 75.0, 100.0)]
  assert abs(xs[0] - 2.75) < 0.1 and abs(xs[-1]) < 0.1  # lane 1 of 2, then of 3
  assert abs(xs[2] - 1.65) < 0.1 and abs(xs[3] - 1.1) < 0.1
  assert all(a >= b - 1e-6 for a, b in zip(xs, xs[1:], strict=False))


def test_turn_from_beside_its_lanes():
  # turn_from_beside: one lane off the turn's lanes where it would be left to the route, it is signalled from there;
  # two lanes off it is still left
  route = route_to_turn(40.0)
  for lane, beside, want in (((0, 2), False, False), ((0, 2), True, True), ((0, 3), True, False)):
    d = Drive(lane, v=3.0)
    d.nav.tune = bridge_tune("")
    d.nav.tune.values.update(turn_from_beside=beside, turn_into_lane=0.0)  # today's rule (turn_into_lane: n16's)
    y, turned = 0.0, False
    while y < 38.0:
      d.step(route, y, {"laneFrac": 0.0})  # surely in the left lane
      turned |= d.nav.turn is not None
      y += d.v * 0.05
    assert turned == want and (not d.nav.skipped) == want, (lane, beside)


def left_turn_then_on(d: Drive, after: list[tuple[float, float]], wait: float = 0.0) -> list[tuple[float, str | None]]:
  """Signals the left turn of route_to_turn(40, "left"), drives it round to 85 deg (way out 90) at 0.5 rad/s, waits
  `wait` s straight, then follows `after`, (heading, yaw rate) every 0.1 s; returns (heading, cue) per step."""
  route = route_to_turn(40.0, "left")
  for k in range(60):
    d.step(route, k * 0.25)
  assert d.nav.signaled == "left"
  west = [[-x, 40.0] for x in np.arange(0.0, 100.0, 5.0)]
  seq = [(float(h), 0.5) for h in np.linspace(0, 85, 40)] + [(85.0, 0.0)] * int(wait * 10) + after
  cues = []
  for h, yaw in seq:
    d.clock.t += 0.1
    state = {"vEgo": 5.0, "pos": [0.0, 40.0, 0.0], "heading": h, "yawRate": yaw, "lane": [0, 1], "routeEnd": 200.0, "route": west}
    d.update(state, True, d.indicator, {"left": 1.0})
    cues.append((h, d.nav.cue))
  return cues


def test_exit_watch_counter_cue_past_the_way_out():
  # exit_watch: past the left turn's way out and still coming round while its pulse may be queued: keepRight; not once
  # the pulse has aged out of the queue, and not without the flag
  round_on = [(85.0 + 2.9 * k, 0.5) for k in range(1, 21)]  # on to 143 deg in 2 s
  for watch, wait, want in ((False, 0.0, False), (True, 0.0, True), (True, 8.0, False)):
    d = Drive((0, 1), v=5.0)
    d.nav.tune = bridge_tune("")
    d.nav.tune.values.update(left_signal_max_entry=0.0, exit_cue_left=False, exit_watch=watch)
    cues = left_turn_then_on(d, round_on, wait)
    assert d.nav.signaled is None
    late = [c for h, c in cues[-20:] if h > 90.0 + nav_mod.EXIT_WATCH_PAST + 3.0]
    assert late and all(c == ("keepRight" if want else None) for c in late), (watch, wait, late)
    assert all(c is None for _, c in cues[:40])  # nothing during the turn itself


def test_turned_unwrapped_and_no_repulse_past_the_way_out():
  # a car come round 350 deg with the turn still signalled reads 10 deg turned wrapped, so it is pulsed again;
  # turned_unwrapped reads 350, and no_repulse_past_exit holds it back as past the way out
  route = route_to_turn(40.0, "left")
  for unwrapped, past_exit, want in ((False, False, True), (True, False, False), (False, True, False)):
    d = Drive((0, 1), v=5.0)
    d.nav.tune = bridge_tune("")
    d.nav.tune.values.update(left_signal_max_entry=0.0, turned_unwrapped=unwrapped, no_repulse_past_exit=past_exit)
    k = 0
    while d.nav.signaled is None:
      d.step(route, k * 0.25)
      k += 1
    for _ in range(10):
      d.step(route, k * 0.25)  # the plugin shows the indicator
    assert d.nav.turn is not None and d.nav.shown
    d.nav.swept, d.nav.swept_heading = 350.0, 350.0
    repeats = d.nav.repeat_t
    d.clock.t += 3.0
    state = {"vEgo": 5.0, "pos": [0.0, k * 0.25, 0.0], "heading": 350.0, "yawRate": 0.0, "lane": [0, 1],
             "route": [[0.0, 0.0]] + [p for p in (route - [0.0, k * 0.25]).tolist() if p[1] > 0.0 or p[0] != 0.0]}
    d.update(state, True, d.indicator, {"left": 0.05})
    assert d.nav.turn is not None
    assert (d.nav.repeat_t != repeats) == want, (unwrapped, past_exit)


def test_blinker_steady_when_openpilot_refreshes_the_turn():
  # with TurnDesireRefresh openpilot asks for the turn again itself, so the blinker no longer drops to repeat it
  route = route_to_turn(40.0, "left")
  for refresh in (False, True):
    d = Drive((0, 1), v=5.0)
    d.nav.refresh = refresh
    d.nav.tune = bridge_tune("")
    d.nav.tune.values.update(left_signal_max_entry=0.0)
    y, gaps = 0.0, 0
    while y < 100.0 and (d.nav.signaled is not None or not signals(d)):
      d.step(route, y, {"routeEnd": 300.0 - y})
      gaps += d.nav.signaled is not None and d.nav.blinker_gap(d.clock.t)
      y += d.v * 0.05
    assert (gaps == 0) == refresh, (refresh, gaps)


def test_keep_right_out_of_oncoming_by_the_map():
  # the map's lanes (the bridge's laneMap) see a one-way driven the wrong way, which the plugin's and route's readings miss
  route = np.array([(0.0, y) for y in np.arange(0.0, 400.0, 5.0)])
  wrong = {"lane": -1, "lanes": 0, "kind": "wrong-way", "bay": False, "oncoming": True, "areas": True}

  def desires(lane_map, extra=None, **tune):
    d = Drive((0, 2))
    d.nav.tune = bridge_tune("")
    d.nav.tune.values.update(tune)
    for k in range(40):  # 2 s
      d.step(route, k * 0.4, {"laneMap": lane_map, "routeEnd": 400.0, **(extra or {})})
    return d.desires
  assert "keepRight" in desires(wrong)
  assert "keepRight" not in desires(None) and "keepRight" not in desires({**wrong, "kind": "own", "oncoming": False})
  assert "keepRight" not in desires(wrong, oncoming_map=False)
  # by a junction's node: the plugin's reading follows GTA's diagonal links there, and so does the map's, until the
  # bridge has the junctions' areas to leave it out in
  near = {"junctions": [5.0]}
  assert "keepRight" not in desires({**wrong, "areas": False}, near)
  assert "keepRight" in desires(wrong, near)
  assert "keepRight" in desires(None, {"lanePlugin": [-1, 2]}) and "keepRight" not in desires(None, {"lanePlugin": [-1, 2], **near})


def test_lane_change_for_a_lane_that_ends():
  # straight on through a junction where the left lane ends (laneDrops: only the right one of two carries on)
  route = np.array([(0.0, y) for y in np.arange(0.0, 400.0, 5.0)])
  d = Drive((0, 2))
  d.nav.tune.values["model_lc_speed"] = 0.0  # today's timing: these mechanics run below n16's lane change speed
  y, started = 0.0, None
  while y < 150.0 and started is None:
    d.step(route, y, {"laneDrops": [[150.0 - y, 1, 1, 2]], "routeEnd": 400.0 - y})
    started = d.nav.changing and (d.nav.changing, 150.0 - y)
    y += d.v * 0.05
  assert started and started[0] == "right" and started[1] > nav_mod.LANE_CHANGE_DIST
  d = Drive((0, 2))
  d.nav.tune = bridge_tune("")
  d.nav.tune.values.update(lane_drops=False)
  for k in range(100):
    d.step(route, k * 0.4, {"laneDrops": [[150.0 - k * 0.4, 1, 1, 2]], "routeEnd": 400.0})
  assert d.nav.changing is None  # off, for an A/B
  d = Drive((1, 2))
  for k in range(100):
    d.step(route, k * 0.4, {"laneDrops": [[150.0 - k * 0.4, 1, 1, 2]], "routeEnd": 400.0})
  assert d.nav.changing is None  # already in it


def test_guidance_only_drives_nothing():
  # Navigate on openpilot off: no signals, lane changes or NavDesire on the way to a turn at a junction; turned on, nav
  # drives as before, and turned off again mid-way it cancels what it asked for
  route = route_to_turn(200.0)
  d = Drive((0, 2), v=7.0)
  d.drives = False
  y = 0.0
  while y < 100.0:
    d.step(route, y, {"junctions": [200.0 - y]})
    y += d.v * 0.05
  assert d.sent == [] and d.desires == []
  d.drives = True
  while y < 190.0 and not d.sent:
    d.step(route, y)
    y += d.v * 0.05
  assert d.sent and (signals(d) or "laneChange" in d.desires)
  d.drives = False
  for _ in range(20):  # NavDesire is held a moment past a lane change, as openpilot reads it every 0.2 s
    d.step(route, y)
  assert d.nav.desire == "" and d.nav.changing is None and d.nav.signaled is None


def jogging_route() -> Route:
  """North 300 m with a 52 deg bend 2 m long half way (GTA's links jogging through a junction), then a right turn,
  which Valhalla calls one."""
  pts = [(0.0, y) for y in (0.0, 60.0, 120.0, 148.0)] + [(1.6, 149.2)] + [(1.6, y) for y in (180.0, 240.0, 300.0)]
  pts += [(x, 300.0) for x in (40.0, 80.0, 120.0)]
  return Route(np.array(pts), [{'type': 1, 'begin_shape_index': 0}, {'type': 10, 'begin_shape_index': 7},  # Valhalla's right
                                {'type': 4, 'begin_shape_index': len(pts) - 1}])


def replay_turns(r: Route, upto: float, own: bool) -> list:
  """The next turn nav finds as the car drives along the route every 2 m, as the bridge gives it the route (points
  every 5 m from the car), with (own) or without the route's own turns: [(m along the car, m along the turn)]."""
  p = Planner(bridge_tune())
  p.min_ahead = nav_mod.MIN_AHEAD_MAP
  out = []
  for s in np.arange(0.0, upto, 2.0):
    pos = np.array([np.interp(s, r.along, r.points[:, k]) for k in (0, 1)])
    r.locate(pos)
    t = p._next_turn(r.ahead(1000.0, 5.0), r.turns(r.length) if own else None)
    out.append((r.at, None if t is None else r.at + t.dist))
  return out


def test_route_turns_hold_still():
  # the turn ahead stays where it is along the route as the car drives towards it; found again on the points from the
  # car every step, it moves about with them
  r = jogging_route()
  turns = r.turns(r.length)
  assert [(t[1], round(t[3])) for t in turns] == [('right', 90)]
  found = replay_turns(r, 280.0, own=True)
  assert all(t is not None and t > at and abs(t - turns[0][0]) < 0.11 for at, t in found)


def test_signal_point_holds_still():
  # the overlay's signal point is where the signal comes on at the turn's own speed, whatever the car's now
  r = jogging_route()
  r.locate(np.array([0.0, 100.0]))
  route, turns = r.ahead(1000.0, 5.0), r.turns(r.length)
  p = Planner(bridge_tune())
  p.min_ahead = nav_mod.MIN_AHEAD_MAP
  points = []
  for v in (4.0, 9.0, 15.0):
    p.v = v
    points.append(p.turn_points(route, [], [], [], turns))
  assert all(np.allclose(a, points[0][0]) and np.allclose(b, points[0][1]) for a, b in points)


def test_route_68_turns_hold_still():
  # Amy's Route 68 replay from (-1401.93, 2416.59), heading 288.9, to (-813.6, 179.5): found on the points from the car
  # every step, the turn ahead jumped between a 50 deg bend at ~350 m and the left at ~490 m every few metres
  import os
  import urllib.request
  if not os.path.exists(os.path.expanduser('~/gta5map_lanes/gta5.osm.pbf')):
    return
  try:
    urllib.request.urlopen('http://127.0.0.1:8002/status', timeout=2.0).read()
  except OSError:
    return
  from openpilot.tools.sim.bridge.gta5.junction_plan import make_router
  router = make_router(os.path.expanduser('~/gta5map_lanes'), None, None)
  router.osm = router.roads
  r = router.route(np.array([-1401.93, 2416.59]), (-288.9) % 360, np.array([-813.603, 179.474]), 27.17)
  ends = [t[0] for t in r.turns(r.length)]
  found = replay_turns(r, 480.0, own=True)
  assert all(t is not None and t > at and min(abs(t - e) for e in ends) < 0.11 for at, t in found)
  assert len({round(t / 10.0) for _, t in found}) == 2  # the right at the start, then the left


def test_city_turn_changes_drawn_where_nav_starts_them():
  # a right turn 400 m on from the left of two lanes at 10 m/s: the lane plan draws the change where nav starts it
  # (8 s + 4 s before the last place, 30 m before the turn), not as late as allowed; not across a stop line on the way
  route = route_to_turn(400.0)
  turn = find_turn(route, nav_mod.MIN_AHEAD_MAP).dist

  def change(**kw):
    keys = lane_plan(route, [], [0, 2], lambda d, after=False: 2, 10.0, kw.pop('tune', None), maps=[], **kw)
    return next((a[0], b[0]) for a, b in zip(keys, keys[1:]) if b[0] > a[0] and b[1] != a[1])
  start, end = change()
  assert abs(start - (turn - 30.0 - 120.0)) < 1.0 and abs(end - start - nav_mod.LANE_LINE_CHANGE) < 1.0
  start2, _ = change(crossings=[start + 10.0])
  assert abs(start2 - (start + 15.0)) < 1.0  # from just past the stop line
  late = nav_mod.Tune()
  late.values['plan_city_early'] = False
  assert abs(change(tune=late)[1] - (turn - 30.0)) < 1.0


def test_turn_lane_changes_not_held_by_junction_nodes():
  # a right turn 250 m on from the left of three lanes at 8 m/s, GTA's junction nodes every 20 m on the way (side
  # streets) and a stop line 16 m before the turn: the changes start where nav's lead says (2 x 8 s + 4 s at 8 m/s
  # before 30 m short of the turn), not only once no junction node is within 40 m; and the overlay's signal marker is
  # where the first change's blinker comes on
  route = route_to_turn(250.0)
  turn = find_turn(route, nav_mod.MIN_AHEAD_MAP).dist
  d = Drive((0, 3), v=8.0)
  d.nav.tune.values["model_lc_speed"] = 0.0  # today's timing: these mechanics run below n16's lane change speed
  y, started, marker = 0.0, None, None
  while y < 200.0 and started is None:
    extra = {"stops": [turn - 16.0 - y], "junctions": [j - y for j in np.arange(20.0, turn, 20.0) if j > y]}
    d.step(route, y, extra)
    if d.nav.changing:
      started = turn - y
    if marker is None and d.nav.lane is not None:  # once nav has the car's lane
      ahead = np.array([p for p in (route - [0.0, y]).tolist() if p[1] >= 0.0])
      marker = d.nav.turn_points(ahead, [], extra["stops"], extra["junctions"])[1][1] + y
    y += d.v * 0.05
  lead = 30.0 + (2 * nav_mod.LANE_CHANGE_TIME + nav_mod.LANE_CHANGE_EARLY) * 8.0
  assert started is not None and started >= lead - 20.0, started
  assert marker is not None and abs(marker - (turn - lead)) < 5.0, marker


def first_change(route, lane, v, extra=None, upto=None, tune=None):
  """Where (m before route_to_turn's turn) nav first asks for a lane change, and the drive."""
  turn = find_turn(route, nav_mod.MIN_AHEAD_MAP).dist
  d = Drive(lane, v=v)
  d.nav.tune.values.update(tune or {})
  y = 0.0
  while y < (upto if upto is not None else turn - 10.0):
    d.step(route, y, {k: (v_(y) if callable(v_) else v_) for k, v_ in (extra or {}).items()})
    if d.nav.changing:
      return turn - y, d
    y += d.v * 0.05
  return None, d


def test_city_change_before_the_slowing():
  """model_lc_speed: a change for a turn with lights before it ends before the car slows for them (queue_room) or the
  turn, timed at the road's limit: it starts further out than today's (city_lead at the car's speed from 30 m short)."""
  route = route_to_turn(400.0)
  turn = find_turn(route, nav_mod.MIN_AHEAD_MAP).dist
  extra = {"stops": lambda y: [turn - 20.0 - y], "stopKinds": ["lights"], "limits": [[0.0, 13.3]], "routeEnd": lambda y: 600.0 - y}
  old, _ = first_change(route, (0, 2), 10.5, extra, tune={"model_lc_speed": 0.0})
  new, _ = first_change(route, (0, 2), 10.5, extra)
  assert old is not None and new is not None
  lead = nav_mod.city_lead(1, 13.3, nav_mod.TUNE)
  assert abs(old - (nav_mod.LANE_CHANGE_DIST + nav_mod.city_lead(1, 10.5, nav_mod.TUNE))) < 5.0
  assert abs(new - (20.0 + 30.0 + lead)) < 5.0 and new > old + 50.0


def test_slow_city_change_waits_for_the_traffic():
  """Slower than model_lc_speed (n16's 10 m/s) the change isn't asked for, also between lc_wait of it and it (9 m/s:
  n16 takes a lane change desire that slow like a turn's); at speed it is."""
  route = route_to_turn(300.0)
  slow, _ = first_change(route, (0, 2), 5.0, {"routeEnd": 500.0}, upto=290.0)
  under, _ = first_change(route, (0, 2), 9.0, {"routeEnd": 500.0}, upto=290.0)
  fast, _ = first_change(route, (0, 2), 11.0, {"routeEnd": 500.0}, upto=290.0)
  assert slow is None and under is None and fast is not None
  # today's planner asks at 5 m/s
  old, _ = first_change(route, (0, 2), 5.0, {"routeEnd": 500.0}, upto=290.0, tune={"model_lc_speed": 0.0})
  assert old is not None


def test_change_not_started_withdrawn_when_slowing():
  """Asked for at 11 m/s, the car slowing to 8 before openpilot starts it (below n16's 10 m/s): withdrawn, its blinker
  never on (gave_up), and not asked again until the car is faster."""
  route = route_to_turn(300.0)
  at, d = first_change(route, (0, 2), 11.0, {"routeEnd": 500.0}, upto=290.0)
  assert at is not None
  y = 300.0 - at
  d.v = 8.0
  for _ in range(40):
    y += d.v * 0.05
    d.step(route, y, {"routeEnd": 500.0 - y})
    assert d.nav.changing is None or d.nav.change_went
  assert not signals(d)
  assert [e["outcome"] for e in d.events if e.get("event") == "navRequest"] == ["gave_up"]


def test_hovering_at_the_lane_change_speed_asks_once():
  """Asked at 10.4 m/s and held by the blind spot, the car hovering about n16's 10 m/s (9.3-10.4) before openpilot can
  start it: the one request stays (no ask, withdraw, ask); a margin (model_lc_margin) below the speed it's withdrawn.
  With no margin it's asked, withdrawn and asked again."""
  route = route_to_turn(400.0)
  extra = {"routeEnd": lambda y: 600.0 - y, "blindspot": [False, True]}

  def hover(tune):
    at, d = first_change(route, (0, 2), 10.4, extra, upto=300.0, tune=tune)
    assert at is not None
    y = 400.0 - at
    for k in range(160):  # 8 s
      d.v = 9.85 + 0.55 * (1 if k % 40 < 20 else -1)
      d.step(route, y, {k_: v_(y) if callable(v_) else v_ for k_, v_ in extra.items()})
      y += d.v * 0.05
    return d, y
  d, y = hover({})
  asks = [e for e in d.events if e.get("event") == "navRequest"]
  assert d.nav.changing == "right" and not d.nav.change_went and asks == []
  d.v = 8.9
  for _ in range(10):
    d.step(route, y, {k_: v_(y) if callable(v_) else v_ for k_, v_ in extra.items()})
    y += d.v * 0.05
  assert d.nav.changing is None and [e["outcome"] for e in d.events if e.get("event") == "navRequest"] == ["gave_up"]
  d, _ = hover({"model_lc_margin": 0.0})
  assert [e["outcome"] for e in d.events if e.get("event") == "navRequest"].count("gave_up") >= 2


def test_too_slow_near_the_turn_isnt_forced():
  """Two lanes off a turn's lanes near it, slower than the model changes at: nothing asked, the turn left to the route."""
  route = route_to_turn(120.0, "left")
  d = Drive((2, 3), v=4.0)
  y, asked = 0.0, False
  while y < 110.0:
    d.step(route, y, {"laneFrac": 2.0, "routeEnd": 300.0})
    asked |= d.nav.changing is not None
    y += d.v * 0.05
  assert not asked and d.nav.skipped and not signals(d)


def test_slow_beside_the_turn_signals_it():
  """One lane beside a right turn's lane, slow, 70 m out (a bay opened late): nav signals the turn, whose desire takes
  the model over, rather than asking for a lane change it can't make; today's planner asks for the change."""
  route = route_to_turn(120.0)
  for tune, want in (({}, "signal"), ({"model_lc_speed": 0.0}, "change")):
    d = Drive((0, 2), v=5.0)
    d.nav.tune.values.update(tune)
    y, got = 50.0, None
    while y < 110.0 and got is None:
      d.step(route, y, {"laneFrac": 0.0, "routeEnd": 300.0})
      got = "signal" if d.nav.signaling else "change" if d.nav.changing else None
      y += d.v * 0.05
    assert got == want and 120.0 - y > 40.0, (tune, got, 120.0 - y)


def test_turn_gap_keeps_the_turn_for_its_signal():
  """One lane beside a right turn's lane, between the turn signal's speed (19 mph) and n16's lane change speed when
  the last place for a change goes by: no change, no signal, and not left to the route; once the car slows below
  19 mph its signal (turn_into_lane) takes the model over. turn_gap off: left to the route at that speed."""
  route = route_to_turn(120.0)
  for tune, want in (({}, "signal"), ({"turn_gap": False}, "skipped")):
    d = Drive((0, 2), v=9.3)
    d.nav.tune.values.update(tune)
    y, got, blinkers = 40.0, None, set()
    while y < 110.0 and got is None:
      d.v = 9.3 if 120.0 - y > 22.0 else 7.0
      d.step(route, y, {"laneFrac": 0.0, "routeEnd": 300.0})
      got = "signal" if d.nav.signaling else "skipped" if d.nav.skipped else None
      if d.indicator is not None:
        blinkers.add(d.v)
      y += d.v * 0.05
    assert got == want, (tune, got, 120.0 - y)
    assert d.nav.changing is None and all(v < 19 * 0.44704 for v in blinkers), blinkers
    if want == "signal":
      assert 120.0 - y < 22.0 and not d.nav.skipped


def test_turn_into_after_a_change_that_came_back():
  """A change towards a right turn that openpilot started but didn't finish (the car back in the lane beside): slower
  near the turn, its signal from beside still takes over (the change no longer under way), not left to the route."""
  route = route_to_turn(300.0)
  at, d = first_change(route, (0, 2), 11.0, {"routeEnd": 500.0}, upto=280.0)
  assert at is not None
  y = 300.0 - at
  for _ in range(30):  # the blinker on and the side clear: openpilot starts it
    d.step(route, y, {"routeEnd": 500.0 - y, "laneFrac": 0.0})
    y += d.v * 0.05
  assert d.nav.change_went
  d._send({"type": "indicatorOff"})  # it gave up and came back: the stalk off
  d.v = 6.0
  y = max(y, 300.0 - 75.0)
  while y < 290.0 and not d.nav.signaling and not d.nav.skipped:
    d.step(route, y, {"routeEnd": 500.0 - y, "laneFrac": 0.0})
    y += d.v * 0.05
  assert d.nav.signaling and not d.nav.skipped and 300.0 - y > 40.0, (300.0 - y, d.nav.skipped)


def test_no_lane_end_ask_from_a_reading_before_the_lanes_changed():
  """Just past a lane drop (3 lanes to 2), the lane reading still the one before it (lane 2 of 3, held LANE_STEADY): the
  lane maps ahead start from 2 lanes, the right one ending 150 m on. Its count isn't theirs, so which lane ends is a guess
  (lanes_for would count the lane that carries on from the left in the 3: lane 1 only, and ask the car over): nothing
  asked. Read as lane 2 of 2 (the one that ends), asked left (laneEnds)."""
  route = np.array([(0.0, y) for y in np.arange(0.0, 1000.0, 5.0)])

  def asks(lane):
    d = Drive(lane, v=12.0)
    y = 0.0
    while y < 100.0:
      d.step(route, y, {"laneMaps": [[150.0 - y, [0, None], 1]], "laneCounts": [[0.0, 2]], "routeEnd": 900.0 - y})
      y += d.v * 0.05
    return [(r["reason"], r["side"]) for r in d.events if r.get("event") == "navRequest"] + (
      [(d.request["reason"], d.request["side"])] if d.request else [])
  assert asks((1, 3)) == []
  assert asks((1, 2)) == [("laneEnds", "left")]


def test_request_lifecycle():
  """A change for a turn: the request (navInstruction.request's content) says what, which side, why, by when and the
  stage; approved once the blinker is on its side; logged with its outcome once over (a navRequest event)."""
  route = route_to_turn(400.0)
  d = Drive((0, 2), v=15.0)
  y = 0.0
  while d.request is None and y < 300.0:
    d.step(route, y)
    y += d.v * 0.05
  r = d.request
  assert r is not None and (r["kind"], r["side"], r["reason"], r["laneChanges"], r["stage"], r["id"]) == ("laneChange", "right", "turn", 1, 3, 1)
  assert 0.0 < r["distance"] < 400.0 - y and not r["approved"]
  for _ in range(20):  # the driver's blinker comes on (stage 3: at once) and openpilot starts the change
    d.step(route, y)
    y += d.v * 0.05
  assert d.request["approved"] and d.request["distance"] < r["distance"]
  d.lane = (1, 2)
  d._send({"type": "indicatorOff"})  # done: the driver's stalk off
  d.step(route, y)
  assert d.request is None
  ev = [e for e in d.events if e["event"] == "navRequest"]
  assert len(ev) == 1 and ev[0]["outcome"] == "done" and ev[0]["lane"] == [0, 2] and ev[0]["reason"] == "turn"
  assert ev[0]["approved_after"] is not None and ev[0]["approved_after"] < 1.0 and ev[0]["target"] > 0
  assert d.nav.request_log == ev


def test_request_declined_and_turn_request():
  """The driver's blinker off before openpilot could start it: declined. A turn's signal is a request too, closed when
  nav is disengaged."""
  route = route_to_turn(400.0)
  d = Drive((0, 2), v=15.0)
  y = 0.0
  while d.indicator is None and y < 300.0:
    d.step(route, y, {"blindspot": [False, True]})  # held by the blind spot
    y += d.v * 0.05
  d.step(route, y, {"blindspot": [False, True]})  # nav sees the blinker on
  d._send({"type": "indicatorOff"})
  d.step(route, y, {"blindspot": [False, True]})
  assert [e["outcome"] for e in d.events if e["event"] == "navRequest"] == ["declined"]
  d = Drive((1, 2), v=5.0)
  route = route_to_turn(60.0)
  y = 0.0
  while not d.nav.signaling and y < 50.0:
    d.step(route, y)
    y += d.v * 0.05
  assert d.request is not None and (d.request["kind"], d.request["side"], d.request["reason"]) == ("turn", "right", "turn")
  d.update({"vEgo": d.v, "pos": [0.0, y, 0.0], "heading": 0.0, "lane": [1, 2], "route": []}, False, d.indicator, {})
  assert d.request is None and [e["outcome"] for e in d.events if e["event"] == "navRequest"] == ["cancelled"]


def test_sided_desire_when_the_driver_approves():
  """Stages 1-2 (the driver approves): NavDesire names the request's side, so only that side's blinker is a lane
  change at any speed (desire_helper); stage 3 (auto-approved, GTA) keeps today's "laneChange"."""
  route = route_to_turn(400.0)
  for stage, want in ((1, "laneChange:right"), (2, "laneChange:right"), (3, "laneChange")):
    d = Drive((0, 2), v=15.0)
    d.stage = stage
    y = 0.0
    while d.request is None and y < 300.0:
      d.step(route, y)
      y += d.v * 0.05
    assert d.nav.desire == want and d.request["stage"] == stage, stage
