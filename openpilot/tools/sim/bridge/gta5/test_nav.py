import numpy as np

import openpilot.selfdrive.navd.planner as nav_mod
from openpilot.selfdrive.navd.planner import Fork, Planner, find_turn, lane_plan
from openpilot.tools.sim.bridge.gta5.gta5_driver import Driver
from openpilot.tools.sim.bridge.gta5.gta5_navd import nav_inputs
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
    self.sent, self.desires = [], []
    self.indicator = None
    self.nav = Planner()
    self.driver = Driver(self._send, lambda: None)
    self.lane, self.v = lane, v
    self.drives = True  # Navigate on openpilot

  def update(self, state: dict, engaged: bool, indicator: str | None, desire: dict) -> tuple[float, bool]:
    """A step from the bridge's state, as the world takes it: the driver's game commands to `sent`, NavDesire to
    `desires`, and the game's waypoint off on arriving."""
    out = self.nav.update(nav_inputs(state, engaged, indicator, desire, self.clock.t, self.drives))
    self.driver.act(out)
    self.desires += out.desires
    if out.arrived:
      self._send({"type": "waypoint", "off": True})
    return out.cap, out.arrived

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


def test_entry_signal_past_the_entry():
  # sparse approach nodes, signalling 3 m past the default entry (12 m out): the turn must still be found there
  route = np.array([(0.0, y) for y in np.arange(0.0, 200.0, 20.0)] + [(x, 200.0) for x in np.arange(0.0, 100.0, 10.0)])
  d = Drive((1, 2), v=5.0)
  d.nav.tune = nav_mod.Tune("")
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
  # a bay opening 30 m before a left turn, from the inside lane: into the bay, then signal the turn
  route = route_to_turn(200.0, "left")
  d = Drive((0, 2), v=6.0)
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


def test_uncapped_tune_into_a_bay():
  # the bay test's drive: by default nav slows the car for the bay and the turn; with the turn, lane change and bay
  # speeds lifted (an end-to-end longitudinal model sets its own speed) it still changes into the bay and signals
  route = route_to_turn(200.0, "left")
  for uncapped in (False, True):
    d = Drive((0, 2), v=6.0)
    d.nav.tune = nav_mod.Tune("")
    if uncapped:
      d.nav.tune.values.update(turn_speed_soft=40.0, turn_speed_square=40.0, turn_speed_sharp=40.0,
                               lane_change_min_speed=40.0, bay_speed=0.0)
    y, caps, order = 0.0, [], []
    while y < 195.0:
      bay_at = 170.0 - y
      forks = [[bay_at, "right", 2, 2, False, 0, True]] if bay_at > 0 else []
      cap, _ = d.step(route, y, {"forks": forks, "routeEnd": 300.0 - y})
      caps.append(cap)
      if d.nav.changing == "left" and "bay" not in order:
        order.append("bay")
      if d.nav.changing and d.indicator == "left" and "bay" in order and 200.0 - y < 26:
        d.indicator, d.lane = None, (-1, 2)
      if d.nav.signaled == "left" and "turn" not in order:
        order.append("turn")
      y += d.v * 0.05
    assert order == ["bay", "turn"]
    slowed = [c for c in caps if 0 < c < d.v]
    assert (not slowed) if uncapped else min(slowed) <= nav_mod.BAY_SPEED


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
    y = 0.0
    while y < 45.0:
      d.step(route, y, {"laneFrac": frac, "routeEnd": 200.0})
      y += d.v * 0.05
    assert (d.nav.turn is not None or d.nav.signaled is not None or not d.nav.skipped) == signalled


def test_turn_done_at_its_way_out_without_repulse():
  # once the car has turned, the blinker drops at the way out even while still yawing, and isn't pulsed again
  route = route_to_turn(40.0, "left")
  d = Drive((0, 1), v=5.0)
  d.nav.tune = nav_mod.Tune("")
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
  d.nav.tune = nav_mod.Tune("")
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
    d.nav.tune = nav_mod.Tune("")
    d.nav.tune.values.update(left_signal_max_entry=wait)
    y = 0.0
    while y < 100.0 and not signals(d):
      d.step(route, y, {"stops": [80.0 - y], "routeEnd": 300.0 - y})
      y += d.v * 0.05
    at[wait] = 80.0 - y
  assert at[0.0] > 5.0 and 2.0 < at[3.0] <= 3.0
  # not held just after a lane change towards it, whose desire would carry on into the junction instead
  d = Drive((0, 2), v=5.0)
  d.nav.tune = nav_mod.Tune("")
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
    d.nav.tune = nav_mod.Tune("")
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
  d.nav.tune = nav_mod.Tune("")
  d.nav.tune.values.update(unturned_cancel=25.0, unturned_keep_pulses=True, left_signal_max_entry=0.0)
  y, pulses = 0.0, []
  while y < 100.0 and (d.nav.signaled is not None or not signals(d)):
    d.step(route, y, {"routeEnd": 300.0 - y})
    if d.nav.signaled is not None and y > 40.0:
      pulses.append(d.nav.repeat_t)
    y += d.v * 0.05
  assert 24.0 < y - 40.0 < 27.0 and len(set(pulses)) > 1, (y, pulses)


def test_junction_entry():
  from openpilot.selfdrive.navd.planner import junction_entry
  assert junction_entry(100.0, [30.0, 85.0], [100.0]) == (85.0, "stop line")
  assert junction_entry(100.0, [], [65.0, 82.0, 92.0, 100.0]) == (82.0, "junction")  # back while the nodes are close
  assert junction_entry(100.0, [], [100.0]) == (85.0, "default")


def test_tune_reloads(tmp_path=None):
  import json as _json
  import os as _os
  import tempfile
  from openpilot.selfdrive.navd.planner import Tune
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
  from openpilot.selfdrive.navd.planner import Tune
  path = _os.path.join(tempfile.mkdtemp(), "tune.json")
  with open(path, "w") as f:
    _json.dump({"signal_mode": "entry", "signal_entry_offset": 1.0}, f)
  route = route_to_turn(120.0)
  d = Drive((1, 2), v=5.0)
  d.nav.tune = Tune(path)
  y, at = 0.0, None
  while y < 118.0 and at is None:
    d.step(route, y, {"stops": [105.0 - y], "junctions": [120.0 - y], "routeEnd": 300.0 - y})
    if d.nav.signaled:
      at = 120.0 - y
    y += d.v * 0.05
  assert at is not None and 12.0 < at < 14.5, at


def test_turn_speed_held_through_the_arc():
  # the turn's speed holds until the car heads its way out and is straight, not when the turn counts as done
  route = route_to_turn(40.0, "left")
  d = Drive((0, 1), v=5.0)
  d.nav.tune = nav_mod.Tune("")
  d.nav.tune.values.update(left_signal_max_entry=0.0)  # signalled at 25 m, without a stop line
  for k in range(60):
    d.step(route, k * 0.25)
  assert d.nav.signaled == "left"
  ahead = [[-x, 40.0] for x in np.arange(0.0, 300.0, 5.0)]
  caps = []
  for h, yaw in [(hh, 0.5) for hh in np.linspace(0, 80, 30)] + [(80.0, 0.3)] * 5 + [(88.0, 0.02)] * 30:
    d.clock.t += 0.1
    cap, _ = d.update({"vEgo": 5.0, "pos": [0.0, 40.0, 0.0], "heading": float(h), "yawRate": yaw, "lane": [0, 1],
                           "routeEnd": 500.0, "route": ahead}, True, d.indicator, {})
    caps.append(cap)
  held = d.nav.tune.turn_speed_square
  assert d.nav.signaled is None  # done at 25 deg from the way out
  assert all(abs(c - held) < 1e-6 for c in caps[:35])  # but held through the arc
  assert caps[-1] > held + 1.0  # and lifted gently once straight


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
    d.nav.tune = nav_mod.Tune("")
    d.nav.tune.values.update(turn_from_beside=beside)
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
    d.nav.tune = nav_mod.Tune("")
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
    d.nav.tune = nav_mod.Tune("")
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
    d.nav.tune = nav_mod.Tune("")
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
    d.nav.tune = nav_mod.Tune("")
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
  y, started = 0.0, None
  while y < 150.0 and started is None:
    d.step(route, y, {"laneDrops": [[150.0 - y, 1, 1, 2]], "routeEnd": 400.0 - y})
    started = d.nav.changing and (d.nav.changing, 150.0 - y)
    y += d.v * 0.05
  assert started and started[0] == "right" and started[1] > nav_mod.LANE_CHANGE_DIST
  d = Drive((0, 2))
  d.nav.tune = nav_mod.Tune("")
  d.nav.tune.values.update(lane_drops=False)
  for k in range(100):
    d.step(route, k * 0.4, {"laneDrops": [[150.0 - k * 0.4, 1, 1, 2]], "routeEnd": 400.0})
  assert d.nav.changing is None  # off, for an A/B
  d = Drive((1, 2))
  for k in range(100):
    d.step(route, k * 0.4, {"laneDrops": [[150.0 - k * 0.4, 1, 1, 2]], "routeEnd": 400.0})
  assert d.nav.changing is None  # already in it


def test_guidance_only_drives_nothing():
  # Navigate on openpilot off: no signals, lane changes, NavDesire or speed caps on the way to a turn; turned on,
  # nav drives as before, and turned off again mid-way it cancels what it asked for
  route = route_to_turn(200.0)
  d = Drive((0, 2), v=7.0)
  d.drives = False
  y, caps = 0.0, []
  while y < 100.0:
    caps.append(d.step(route, y)[0])
    y += d.v * 0.05
  assert d.sent == [] and d.desires == [] and not any(caps)
  d.drives = True
  while y < 190.0 and not d.sent:
    d.step(route, y)
    y += d.v * 0.05
  assert d.sent and (signals(d) or "laneChange" in d.desires)
  d.drives = False
  for _ in range(20):  # NavDesire is held a moment past a lane change, as openpilot reads it every 0.2 s
    d.step(route, y)
  assert d.nav.desire == "" and d.nav.changing is None and d.nav.signaled is None
