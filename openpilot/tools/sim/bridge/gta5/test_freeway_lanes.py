"""nav's lane changes for freeway exits and forks (planner fwy_schedule), live and on its lane plan, and the lane it
plans from (lane_source). No pytest needed: `python test_freeway_lanes.py` runs them all."""
import numpy as np

import openpilot.selfdrive.navd.planner as nav_mod
from openpilot.selfdrive.navd.planner import lane_plan
from openpilot.tools.sim.bridge.gta5.test_nav import Drive

V = 29.0
CHANGE_S = 5.0  # s the driver's lane change takes


class Freeway:
  """A straight freeway north of `lanes` lanes, the route leaving it on the right at `exit_at` m (a one-lane ramp off
  the right lane), with `extra` moves: forks [m, side, lanes, lanes in, keep, other, slip] and lane maps
  [m, map, how many after] (m along); the road class `cls` throughout."""
  def __init__(self, lanes=4, exit_at=4000.0, cls="motorway", forks=(), maps=(), length=None, exit_lanes=None):
    self.n, self.exit_at, self.cls = lanes, exit_at, cls
    self.length = length or exit_at + 300.0
    k = exit_lanes or lanes
    self.forks = sorted([[exit_at, "right", 1, k, True, k - 1, False], *forks])
    self.maps = sorted([[exit_at, [None] * (k - 1) + [0], 1], *maps])
    self.route = np.array([(0.0, y) for y in np.arange(0.0, self.length, 5.0)])

  def extra(self, y: float) -> dict:
    return {"forks": [[d - y, *rest] for d, *rest in self.forks if d - y > -50.0],
            "laneMaps": [[d - y, m, k] for d, m, k in self.maps if d - y > 0.0],
            "roadClasses": [[0.0, self.cls]] if self.cls else None, "routeEnd": self.length - y, "limits": []}

  def ahead(self, y: float) -> np.ndarray:
    """The route on from the car at y, in map coordinates (as the bridge gives it: points held still)."""
    return np.vstack([[0.0, y], self.route[self.route[:, 1] > y]])

  def lanes_at(self, y: float):
    """lane_plan's lanes_at for the car at y."""
    def at(d, after=False):
      s = y + d + (1.0 if after else -1.0)
      k = self.n
      for dm, _m, n in self.maps:
        if dm <= s:
          k = n
      return k
    return at

  def ribbon(self, y: float, lane: int, n: int, fwy=None, v=V) -> list[tuple[float, float, float]]:
    """nav's lane plan from the car at y in lane `lane` of n: [(m along from, to, lanes moved)]."""
    e = self.extra(y)
    keys = lane_plan(self.ahead(y), e["forks"], [lane, n], self.lanes_at(y), v, None,
                     maps=e["laneMaps"], classes=e["roadClasses"], limits=e["limits"], fwy=fwy, turns=[])
    return [(a[0] + y, b[0] + y, b[1] - a[1]) for a, b in zip(keys, keys[1:], strict=False)
            if b[0] > a[0] + 0.01 and abs(b[1] - a[1]) > 0.01]

  def drive(self, lane: int, start=0.0, upto=None, v=V, d: Drive | None = None, model=None, tune=None):
    """The live planner driven from `start` in lane `lane`: [(m along where each change was asked for, side)], the
    Drive, and the ribbon drawn at the start."""
    d = d or Drive((lane, self.n), v=v)
    if tune:
      d.nav.tune = nav_mod.Tune("")
      d.nav.tune.values.update(tune)
    y, changes, done_at = start, [], None
    ribbon = None
    while y < (upto or self.exit_at - 20.0):
      n = d.lane[1]
      for dm, m, k in self.maps:
        if y - V * 0.05 < dm <= y:
          d.lane = (nav_mod.carry(m, d.lane[0]), k)
      extra = {**self.extra(y), "laneFrac": float(d.lane[0])}
      if model is not None:
        extra["modelLane"] = model(y, d)
      d.clock.t += 0.05
      d.update({"vEgo": v, "pos": [0.0, y, 0.0], "heading": 0.0, "yawRate": 0.0, "lane": list(d.lane),
                "route": self.ahead(y).tolist(), **extra}, True, d.indicator, {})
      if ribbon is None and d.nav.lane is not None:
        ribbon = self.ribbon(y, d.nav.lane[0], d.nav.lane[1], d.nav.fwy_state(), v)
      if d.nav.changing and done_at is None and d.indicator == d.nav.changing:
        changes.append((round(y), d.nav.changing))
        done_at = d.clock.t + CHANGE_S
      if done_at is not None and d.clock.t >= done_at:
        d.lane = (d.lane[0] + (1 if d.indicator == "right" else -1), n)
        d.indicator, done_at = None, None
      y += v * 0.05
    return changes, d, ribbon


def test_far_ahead_with_room():
  # 4 lanes, from the left lane, the exit 4 km on: 30 s a lane at 29 m/s, the last done 300 m before the gore
  f = Freeway()
  each = 30.0 * V
  first = f.exit_at - 300.0 - 3 * each
  changes, d, ribbon = f.drive(0)
  assert [s for _, s in changes] == ["right"] * 3, changes
  starts = [c[0] for c in changes]
  assert abs(starts[0] - first) < 40.0, (starts, first)
  assert all(abs(b - a - each) < 40.0 for a, b in zip(starts, starts[1:], strict=False)), starts
  assert starts[-1] + CHANGE_S * V < f.exit_at - 300.0
  # the lane plan drawn from the start has the same changes
  assert [round(r[2]) for r in ribbon] == [1, 1, 1], ribbon
  assert all(abs(r[0] - s) < 40.0 for r, s in zip(ribbon, starts, strict=True)), (ribbon, starts)
  assert d.lane[0] == 3


def test_short_room_spread_evenly():
  # starting 1.5 km before the exit: the three changes spread evenly over the 1.2 km to 300 m before it
  f = Freeway()
  start = f.exit_at - 1500.0
  changes, _, ribbon = f.drive(0, start=start)
  starts = [c[0] for c in changes]
  assert len(starts) == 3 and starts[0] - start < 60.0, starts
  gaps = np.diff(starts)
  assert abs(gaps[0] - gaps[1]) < 40.0 and 300.0 < gaps[0] < 450.0, starts
  assert all(abs(r[0] - s) < 40.0 for r, s in zip(ribbon, starts, strict=True)), (ribbon, starts)


def test_city_roads_keep_their_timing():
  # the same road as a primary road, or without road classes: nav's changes as before, (8 s each + 12 s) back from the end
  for cls in ("primary", None):
    f = Freeway(cls=cls)
    changes, _, ribbon = f.drive(0, start=2000.0)
    last = max(nav_mod.FORK_LAST_DIST, nav_mod.FORK_LAST * V)
    early = (3 * nav_mod.LANE_CHANGE_TIME + nav_mod.LANE_CHANGE_EARLY + nav_mod.FAST_EARLY) * V
    assert abs(changes[0][0] - (f.exit_at - last - early)) < 60.0, (cls, changes)
    # the lane plan draws them where nav starts them (plan_city_early), each over LANE_LINE_CHANGE m
    assert len(ribbon) == 1 and abs(ribbon[0][0] - changes[0][0]) < 60.0, (cls, ribbon, changes)
    assert abs(ribbon[0][1] - ribbon[0][0] - 3 * nav_mod.LANE_LINE_CHANGE) < 1.0 and ribbon[0][2] == 3.0, (cls, ribbon)
  # and the tune turns it off
  changes, _, _ = Freeway().drive(0, start=2000.0, tune={"fwy_lane_time": 0.0})
  assert abs(changes[0][0] - (4000.0 - last - early)) < 60.0, changes


def test_not_into_a_lane_leaving_the_route_first():
  # an exit the route doesn't take at 3200 m, off the right lane, which ends there (a drop lane, beginning again at
  # 3300 m): the first two changes on time, the third, into the right lane, only once it carries on past the exit
  f = Freeway(forks=[[3200.0, "left", 3, 4, False, 0, False]], maps=[[3200.0, [0, 1, 2, None], 3], [3300.0, [0, 1, 2], 4]])
  changes, d, ribbon = f.drive(0)
  starts = [c[0] for c in changes]
  first, each = f.exit_at - 300.0 - 3 * 30.0 * V, 30.0 * V
  assert len(starts) == 3 and abs(starts[0] - first) < 40.0 and abs(starts[1] - first - each) < 40.0, starts
  assert 3300.0 <= starts[2] < 3400.0, starts
  assert len(ribbon) == 3 and ribbon[1][1] < 3200.0 and 3300.0 <= ribbon[2][0] < 3400.0, ribbon
  assert all(abs(r[0] - s) < 50.0 for r, s in zip(ribbon, starts, strict=True)), (ribbon, starts)
  assert d.lane[0] == 3


def test_settles_past_the_maneuver_before():
  # the route keeps left at a split at 2000 m, taking the left two of four lanes, then exits off the right of those:
  # no change for the exit until fwy_settle s past the split, though its budget would start it before
  f = Freeway(exit_at=3000.0, exit_lanes=2, forks=[[2000.0, "left", 2, 4, True, 2, False]], maps=[[2000.0, [0, 1, None, None], 2]])
  changes, _, ribbon = f.drive(0)
  settle = 2000.0 + 5.0 * V
  assert [c[1] for c in changes] == ["right"] and settle - 5.0 <= changes[0][0] < settle + 60.0, changes
  assert len(ribbon) == 1 and abs(ribbon[0][0] - changes[0][0]) < 40.0, (ribbon, changes)


def model_lane(idx, count, prob=0.9):
  return [idx, count, prob]


def lanes_of(d: Drive, model, steps=60, lane=(1, 3)):
  out = []
  d.lane = lane
  for k in range(steps):
    m = model(k)
    d.step(np.array([(0.0, y) for y in np.arange(0.0, 500.0, 5.0)]), 0.0, {"modelLane": m, "routeEnd": 500.0})
    out.append((d.nav.lane, d.nav.lane_src))
  return out


def test_fused_lane_source():
  # sure, held a second, as many lanes as the map: the model's
  d = Drive((1, 3), v=0.0)
  got = lanes_of(d, lambda k: model_lane(2, 3))
  assert got[15] == ((1, 3), "map") and got[-1] == ((2, 3), "model"), got[::10]
  # a lane that flickers never holds a second: the map's
  d = Drive((1, 3), v=0.0)
  got = lanes_of(d, lambda k: model_lane(2 if k % 8 < 4 else 0, 3))
  assert all(g == ((1, 3), "map") for g in got[15:]), got[::10]
  # unsure, or counting another number of lanes: the map's
  d = Drive((1, 3), v=0.0)
  assert lanes_of(d, lambda k: model_lane(2, 3, 0.5))[-1] == ((1, 3), "map")
  d = Drive((1, 3), v=0.0)
  assert lanes_of(d, lambda k: model_lane(2, 4))[-1] == ((1, 3), "map")
  # a brief flip after it's held stays the held lane
  d = Drive((1, 3), v=0.0)
  got = lanes_of(d, lambda k: model_lane(0 if 40 <= k < 48 else 2, 3))
  assert got[-1] == ((2, 3), "model") and all(g[0] == (2, 3) for g in got[25:]), got[20:]
  # map alone, or the model alone
  for src, want in (("map", ((1, 3), "map")), ("model", ((2, 4), "model"))):
    d = Drive((1, 3), v=0.0)
    d.nav.tune = nav_mod.Tune("")
    d.nav.tune.values.update(lane_source=src)
    assert lanes_of(d, lambda k: model_lane(2, 4))[-1] == want, src


def test_model_lane_lag_and_flicker_dont_add_changes():
  # by the model's lane, which reads each lane 2.5 s late and flickers to the left lane for a moment: three changes
  # right, close together for lack of room, but no extra one for a lane the car has already left
  f = Freeway()
  seen: list[tuple[float, int]] = []

  def model(y, d):
    seen.append((d.clock.t, d.lane[0]))
    idx = next((lane for t, lane in reversed(seen) if t <= d.clock.t - 2.5), seen[0][1])
    if 3600.0 < y < 3620.0:
      idx = 0
    return model_lane(idx, d.lane[1])
  changes, d, _ = f.drive(0, start=f.exit_at - 900.0, model=model)
  assert [c[1] for c in changes] == ["right"] * 3 and d.lane[0] == 3, changes
  assert d.nav.lane_src == "model"


if __name__ == "__main__":
  import sys
  tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
  only = sys.argv[1:]
  for name, fn in tests:
    if only and name not in only:
      continue
    fn()
    print(f"ok {name}")
