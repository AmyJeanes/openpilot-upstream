import numpy as np

import openpilot.tools.sim.bridge.gta5.gta5_nav as nav_mod
from openpilot.tools.sim.bridge.gta5.gta5_nav import Fork, Nav, find_turn, lane_plan
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
  """Nav driven along a route by a car that does as it's asked, northwards at a steady speed."""
  def __init__(self, lane, v=8.0):
    self.clock = Clock()
    nav_mod.time = self.clock
    self.sent, self.desires = [], []
    self.indicator = None
    self.nav = Nav(self._send, self.desires.append)
    self.lane, self.v = lane, v

  def _send(self, m):
    self.sent.append(m)
    self.indicator = m.get("side") if m["type"] == "setIndicator" else None if m["type"] == "indicatorOff" else self.indicator

  def step(self, route, y, extra=None, dt=0.05):
    self.clock.t += dt
    state = {"vEgo": self.v, "pos": [0.0, y, 0.0], "heading": 0.0, "yawRate": 0.0, "lane": list(self.lane),
             "route": [[0.0, 0.0]] + [p for p in (route - [0.0, y]).tolist() if p[1] > 0.0 or p[0] != 0.0], **(extra or {})}
    return self.nav.update(state, True, self.indicator, {})


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
    d.nav.update(state, True, d.indicator, {})
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
  for k in range(60):
    d.step(route, k * 0.25)
  assert d.nav.signaled == "left"
  repeats = d.nav.repeat_t
  # the car turns: heading 0 -> 85 (way out 90), yawing 0.5 rad/s; the route ahead now runs west from the car
  for k, h in enumerate(np.linspace(0, 85, 40)):
    d.clock.t += 0.1
    state = {"vEgo": 5.0, "pos": [0.0, 40.0, 0.0], "heading": float(h), "yawRate": 0.5, "lane": [0, 1], "routeEnd": 200.0,
             "route": [[-x, 40.0] for x in np.arange(0.0, 100.0, 5.0)]}
    d.nav.update(state, True, d.indicator, {"left": 1.0 if k < 30 else 0.05})
  assert d.nav.signaled is None and d.nav.repeat_t == repeats
  assert d.nav.cue == "keepRight"



def test_exit_cue_stacked_within_the_turn():
  # exit_cue_at: the keepRight starts halfway round, stacked on the turn while its blinker is on, then plain past it
  route = route_to_turn(40.0, "left")
  d = Drive((0, 1), v=5.0)
  d.nav.tune = nav_mod.Tune("")
  d.nav.tune.values.update(exit_cue_at=0.5)
  for k in range(60):
    d.step(route, k * 0.25)
  assert d.nav.signaled == "left" and "+keepRight" not in d.desires
  for k, h in enumerate(np.linspace(0, 85, 40)):
    d.clock.t += 0.1
    state = {"vEgo": 5.0, "pos": [0.0, 40.0, 0.0], "heading": float(h), "yawRate": 0.5, "lane": [0, 1], "routeEnd": 200.0,
             "route": [[-x, 40.0] for x in np.arange(0.0, 100.0, 5.0)]}
    d.nav.update(state, True, d.indicator, {"left": 1.0})
    if h < 40:
      assert d.nav.desire != "+keepRight"
  assert "+keepRight" in d.desires and d.nav.signaled is None and d.desires[-1] == "keepRight"

def test_junction_entry():
  from openpilot.tools.sim.bridge.gta5.gta5_nav import junction_entry
  assert junction_entry(100.0, [30.0, 85.0], [100.0]) == (85.0, "stop line")
  assert junction_entry(100.0, [], [65.0, 82.0, 92.0, 100.0]) == (82.0, "junction")  # back while the nodes are close
  assert junction_entry(100.0, [], [100.0]) == (85.0, "default")


def test_tune_reloads(tmp_path=None):
  import json as _json
  import os as _os
  import tempfile
  from openpilot.tools.sim.bridge.gta5.gta5_nav import Tune
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
  from openpilot.tools.sim.bridge.gta5.gta5_nav import Tune
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
  for k in range(60):
    d.step(route, k * 0.25)
  assert d.nav.signaled == "left"
  ahead = [[-x, 40.0] for x in np.arange(0.0, 300.0, 5.0)]
  caps = []
  for h, yaw in [(hh, 0.5) for hh in np.linspace(0, 80, 30)] + [(80.0, 0.3)] * 5 + [(88.0, 0.02)] * 30:
    d.clock.t += 0.1
    cap, _ = d.nav.update({"vEgo": 5.0, "pos": [0.0, 40.0, 0.0], "heading": float(h), "yawRate": yaw, "lane": [0, 1],
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
