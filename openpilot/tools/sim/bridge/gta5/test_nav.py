import numpy as np

import openpilot.tools.sim.bridge.gta5.gta5_nav as nav_mod
from openpilot.tools.sim.bridge.gta5.gta5_nav import Fork, Nav, find_turn
from openpilot.tools.sim.bridge.gta5.map.paths import Link


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
    d.step(route, y)
    turned |= d.nav.turn is not None
    y += d.v * 0.05
  assert not turned and len(d.nav.skipped) == 1


def test_fork_lanes():
  assert Fork(100.0, "right", 1, 2, True).lanes(2) == (1, 1)
  assert Fork(100.0, "left", 2, 3, True).lanes(3) == (0, 1)
  assert Fork(100.0, "left", 1, 1, False).lanes(2) == (0, 1)  # one lane in: any lane
  assert Fork(100.0, "left", 4, 4, True, other=1).lanes(4) == (0, 1)  # GTA counts the exit's lane on top
  assert Fork(100.0, "right", 2, 2, False, slip=True).lanes(2) == (0, 1)  # a bay opening: any lane


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
