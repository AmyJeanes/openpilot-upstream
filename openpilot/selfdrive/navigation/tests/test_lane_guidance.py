"""Step 1 lane guidance (lane_guidance.py) on Mapbox routes: made ones along the equator, and the real fixtures."""
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from openpilot.selfdrive.navigation import lane_guidance as lg
from openpilot.selfdrive.navigation.route import METERS_PER_DEGREE

FIXTURES = Path(__file__).parent / "fixtures" / "mapbox"
DEG = 1 / METERS_PER_DEGREE  # degrees of longitude per m at the equator


def lane(ind, valid=False):
  out = {"indications": [ind] if isinstance(ind, str) else list(ind), "valid": valid, "active": valid}
  if valid:
    out["valid_indication"] = ind if isinstance(ind, str) else ind[0]
  return out


def made_route(maneuvers, length=3000.0, kmh=48, motorway=False, signals=(), extra=()):
  """A straight route east along the equator, `length` m, with maneuvers [(m along, type, modifier, lanes or None)],
  maxspeed `kmh` throughout; intersections with traffic signals at `signals` (m along), and others with lanes at
  `extra` [(m along, lanes)]."""
  n = int(length // 10) + 1
  coords = [[i * 10.0 * DEG, 0.0] for i in range(n)]
  cuts = [0.0] + [m[0] for m in maneuvers] + [length]
  steps = []
  for k in range(len(cuts) - 1):
    a, b = cuts[k], cuts[k + 1]
    i0, i1 = int(a // 10), int(b // 10)
    mtype, mod, lanes = ("depart", "", None) if k == 0 else maneuvers[k - 1][1:]
    it = {"location": coords[i0], "geometry_index": i0, "bearings": [90, 270], "entry": [True, False], "in": 1, "out": 0,
          "mapbox_streets_v8": {"class": "motorway" if motorway else "street"}}
    if lanes is not None:
      it["lanes"] = lanes
    more = [{"location": coords[int(x // 10)], "geometry_index": int(x // 10), "bearings": [90, 270], "entry": [True, False],
             "in": 1, "out": 0, **({"traffic_signal": True} if x in signals else {}), **({"lanes": ls} if ls else {})}
            for x, ls in sorted([(x, None) for x in signals] + list(extra)) if a < x < b]
    steps.append({"distance": b - a, "duration": (b - a) / 10.0, "name": f"road {k}",
                  "geometry": {"type": "LineString", "coordinates": coords[i0:i1 + 1]},
                  "maneuver": {"type": mtype, "modifier": mod, "location": coords[i0]} if mod else {"type": mtype, "location": coords[i0]},
                  "intersections": [it, *more], "bannerInstructions": []})
  steps.append({"distance": 0.0, "duration": 0.0, "name": "", "geometry": {"type": "LineString", "coordinates": [coords[-1]] * 2},
                "maneuver": {"type": "arrive", "location": coords[-1]}, "intersections": [{"location": coords[-1],
                "geometry_index": n - 1, "bearings": [270], "entry": [False], "in": 0}], "bannerInstructions": []})
  return {"geometry": {"type": "LineString", "coordinates": coords}, "distance": length,
          "legs": [{"steps": steps, "annotation": {"maxspeed": [{"speed": kmh, "unit": "km/h"}] * (n - 1)}}]}


class Drive:
  """The guidance stepped at 10 Hz along a route at a speed, in a lane the model reads surely."""
  def __init__(self, route, tune=None, stage=3):
    self.g = lg.LaneGuidance(tune)
    self.g.set_route(route)
    self.t, self.along, self.stage = 0.0, 0.0, stage
    self.blinker = None
    self.log: list = []

  def step(self, v, lane, changing=False, blinker="auto"):
    out = self.g.update(lg.GuidanceInputs(self.t, self.along, v, [lane[0], lane[1], 0.95], self.blinker if blinker == "auto" else blinker,
                                          changing, stage=self.stage))
    for r in out.requests:
      self.blinker = None if r == "cancelSignal" else ("left" if r.endswith("Left") else "right")
    self.log.append((self.t, self.along, out))
    self.t += 0.1
    self.along += v * 0.1
    return out

  def run_until(self, v, lane, cond, max_s=200.0):
    for _ in range(int(max_s * 10)):
      out = self.step(v, lane)
      if cond(out):
        return out
    return None


def first_request(d: Drive, v: float, lane, max_s=300.0):
  out = d.run_until(v, lane, lambda o: o.request is not None and o.request["kind"] == "laneChange", max_s)
  return None if out is None else (d.along, out)


def turn_right_two_lanes():
  return made_route([(2000.0, "turn", "right", [lane("straight"), lane("right", True)])])


def test_timing_by_speed_and_lanes_away():
  """Asked once the time left (to done_before before the turn) at the planning speed is lanes away x lc_time + margin:
  further out at speed, and twice as far again for two lanes."""
  t = lg.Tune()
  vmax = 48 / 3.6
  for v, n, lanes_ in ((12.0, 1, [lane("straight"), lane("right", True)]),
                       (20.0, 1, [lane("straight"), lane("right", True)]),
                       (12.0, 2, [lane("straight"), lane("straight"), lane("right", True)])):
    d = Drive(made_route([(2000.0, "turn", "right", lanes_)]))
    got = first_request(d, v, (0, n + 1))
    assert got is not None
    at, out = got
    vp = max(v, t.maxspeed_share * vmax)
    left = 2000.0 - t.done_before_turn - at
    assert left == pytest.approx(max(vp * (n * t.lc_time + t.margin), n * t.min_dist), abs=2 * v * 0.1 + 0.5)
    assert out.request["laneChanges"] == n and out.request["side"] == "right" and out.request["reason"] == "turn"


def test_slowing_traffic_doesnt_put_it_off():
  """Below the road's limit the planning speed is its share of maxspeed: on a 40 mph road (share 14.2 m/s) asked as far
  out at 11 m/s as at 14; faster, further."""
  def at(v):
    return first_request(Drive(made_route([(2000.0, "turn", "right", [lane("straight"), lane("right", True)])], kmh=64)), v, (0, 2))[0]
  assert abs(at(11.0) - at(14.0)) < 3.0
  assert at(14.0) - at(20.0) > 50.0


def test_freeway_exit_longer_lc_time():
  """A motorway exit three lanes over at 70 mph: asked lanes x lc_time_fwy + margin out, km ahead."""
  t = lg.Tune()
  route = made_route([(5000.0, "off ramp", "slight right", [lane("straight")] * 3 + [lane("slight right", True)])],
                     length=6000.0, kmh=113, motorway=True)
  d = Drive(route)
  at, out = first_request(d, 30.0, (0, 4))
  left = 5000.0 - t.done_before_fork - at
  assert left == pytest.approx(30.0 * (3 * t.lc_time_fwy + t.margin), abs=4.0)
  assert out.request["reason"] == "exit" and out.request["laneChanges"] == 3


def test_bay_not_open_counts_from_the_turns_side():
  """A left turn (right-hand traffic) into a bay: Mapbox lists 3 lanes, the model counts 2 before it opens. From the left
  the car in the left lane is in place (no request) and the turn's signal takes it over as it slows; one lane right of
  that, one change left."""
  route = made_route([(1500.0, "turn", "left", [lane("left", True), lane("straight"), lane("straight")])])
  d = Drive(route, {"turn_into": 80.0})
  assert d.run_until(12.0, (0, 2), lambda o: o.request is not None, 120.0) is None  # in place: nothing asked
  d.along = 1500.0 - 70.0
  out = d.run_until(6.0, (0, 2), lambda o: o.request is not None, 10.0)
  assert out is not None and out.request["kind"] == "turn" and out.request["side"] == "left"
  d2 = Drive(route)
  at, out = first_request(d2, 12.0, (1, 2))
  assert out.request["laneChanges"] == 1 and out.request["side"] == "left"


def test_two_close_maneuvers_planned_together():
  """Two left lanes for a left turn, a right turn 60 m on: the left turn's target narrows to its right-hand lane."""
  route = made_route([(1500.0, "turn", "left", [lane("left", True), lane("left", True), lane("straight")]),
                      (1560.0, "turn", "right", None)])
  _, out = first_request(Drive(route), 12.0, (0, 3))
  assert out.request["side"] == "right" and out.request["laneChanges"] == 1
  got = first_request(Drive(route), 12.0, (1, 3))
  assert got is None or got[0] > 1500.0  # already in its lane: nothing asked before the first turn


def test_no_lanes_falls_back_to_the_outermost():
  route = made_route([(1500.0, "turn", "right", None)])
  _, out = first_request(Drive(route), 12.0, (0, 3))
  assert out.request["side"] == "right" and out.request["laneChanges"] == 2
  assert first_request(Drive(route, {"fallback": False}), 12.0, (0, 3)) is None


def test_withdrawn_when_too_slow():
  d = Drive(turn_right_two_lanes())
  first_request(d, 12.0, (0, 2))
  out = d.run_until(7.0, (0, 2), lambda o: o.request is None or o.request["kind"] != "laneChange", 5.0)
  assert out is not None and out.request is None
  assert d.g.request_log[-1]["outcome"] == "gave_up"
  assert "cancelSignal" in out.requests or d.blinker is None
  assert d.g.desire == "" or d.t > 0


def test_request_and_sided_desire():
  """Stage 1: NavDesire laneChange:<side>, the blinker the driver's (approved once on); stage 3: bare laneChange, nav's
  blinker after openpilot has read it; done, the next waits change_gap."""
  for stage, desire in ((1, "laneChange:right"), (3, "laneChange")):
    d = Drive(made_route([(2000.0, "turn", "right", [lane("straight"), lane("straight"), lane("right", True)])]), stage=stage)
    at, out = first_request(d, 12.0, (0, 3))
    assert d.g.desire == desire and not out.request["approved"]
    out = d.run_until(12.0, (0, 3), lambda o: "laneChangeRight" in o.requests, 1.0)
    assert out is not None  # the blinker after PARAM_LEAD
    out = d.step(12.0, (0, 3))
    assert out.request["approved"]
    d.step(12.0, (0, 3), changing=True)
    out = d.step(12.0, (0, 3), changing=False)
    assert d.g.request_log[-1]["outcome"] == "done" and d.g.lane == (1, 3)
    out = d.run_until(12.0, (1, 3), lambda o: o.request is not None and o.request["kind"] == "laneChange", 5.0)
    assert out is not None and out.request["laneChanges"] == 1 and d.t - d.g.request_log[-1]["t"] >= lg.Tune().change_gap


def test_lanes_shown_only_under_guidance():
  route = made_route([(2000.0, "turn", "right", [lane("straight"), lane(("straight", "right"), True)])])
  d = Drive(route)
  out = d.step(12.0, (1, 2))
  assert out.lanes is None  # 2 km out
  d.along = 2000.0 - 300.0
  out = d.step(12.0, (1, 2))
  assert [x["directions"] for x in out.lanes] == [["straight"], ["straight", "right"]]
  assert [x["active"] for x in out.lanes] == [False, True] and out.lanes[1]["activeDirection"] == "straight"
  assert d.log and out.lanes[1]["current"] or d.g.lane is None


def test_lane_reader_holds():
  r = lg.LaneReader(lg.Tune())
  assert r.update([1, 3, 0.9], 0.0) is None
  assert r.update([1, 3, 0.9], 1.0) == (1, 3)
  assert r.update([2, 3, 0.5], 1.2) == (1, 3)  # unsure: held
  assert r.update(None, 3.5) == (1, 3)  # a dropout under lane_stale
  assert r.update(None, 4.5) is None


def fixture_route(name: str) -> dict:
  return json.loads((FIXTURES / f"{name}.json").read_text())["response"]["routes"][0]


def test_real_bay_dale_road():
  """Winchester Rd into Dale Rd (left-hand traffic, the right turn's bay): counted from the right, in place in the right
  lane of 2 before the third appears."""
  mr = lg.MapboxRoute(fixture_route("soton_uni_to_general"))
  m = next(x for x in mr.moves if x.name == "Dale Road")
  assert m.kind == "turn" and m.side == "right" and m.signal
  assert lg.targets(m, 2) == [0] and lg.targets(m, 3) == [0]


def test_real_exit_m27_j3():
  """M27 J3: the exit lane exists only at the gore (4 lanes there, 3 before): the left lane of 3 is in place."""
  mr = lg.MapboxRoute(fixture_route("m27_j5_west_exit_j3"))
  m = next(x for x in mr.moves if x.kind == "exit" and x.side == "left")
  assert m.freeway and len(m.lanes) == 4 and lg.targets(m, 3) == [0]


def test_navd_runs_it(monkeypatch):
  """comma navd with NAVD_LANE_GUIDANCE=1: the request and lanes on navInstruction, NavDesire set."""
  try:
    import openpilot.common.params  # noqa: F401  (native: fails where it isn't built)
  except OSError:
    stub = types.ModuleType("openpilot.common.params")
    stub.Params = MagicMock
    monkeypatch.setitem(sys.modules, "openpilot.common.params", stub)
  from openpilot.selfdrive.navigation import navd
  values = {"NavLaneChangeStage": 1, "LanguageSetting": "main_en"}
  params = MagicMock()
  params.get.side_effect = values.get
  params.put.side_effect = values.__setitem__
  params.remove.side_effect = lambda k: values.pop(k, None)
  route = made_route([(2000.0, "turn", "right", [lane("straight"), lane("right", True)])])
  sm = MagicMock()
  sm.alive = {"modelV2": True}
  sm.valid = {"modelV2": True}
  cs = SimpleNamespace(vEgo=12.0, leftBlinker=False, rightBlinker=False, leftBlindspot=False, rightBlindspot=False)
  md = SimpleNamespace(laneHead=SimpleNamespace(laneIdx=0, laneCount=2, prob=0.95), meta=SimpleNamespace(laneChangeState="off"))
  gps = SimpleNamespace(hasFix=True, latitude=0.0, longitude=1880.0 * DEG, bearingDeg=90.0)
  sm.__getitem__.side_effect = lambda k: {"carState": cs, "modelV2": md}.get(k, gps)
  with patch.dict("os.environ", {"NAVD_LANE_GUIDANCE": "1", "MAPBOX_HOST": "http://127.0.0.1:1"}), patch.object(navd, "Api"):
    engine = navd.RouteEngine(sm, MagicMock(), params=params)
  response = MagicMock(status_code=200)
  response.json.return_value = {"routes": [route]}
  engine.last_position, engine.last_bearing = (1880.0 * DEG, 0.0), 90.0
  with patch("requests.get", return_value=response) as get:
    engine.calculate_route((3000.0 * DEG, 0.0))
  assert get.call_args.kwargs["params"]["annotations"] == "maxspeed,congestion"
  clock = iter(i * 0.1 for i in range(1000))
  with patch.object(navd, "latest_gps_source", return_value="gpsLocation"), patch.object(navd.time, "monotonic", lambda: next(clock)):
    for _ in range(25):
      engine.update(full=False)
  msg = next(c.args[1] for c in reversed(engine.pm.send.call_args_list) if c.args[0] == "navInstruction")
  ins = msg.navInstruction
  assert str(ins.request.kind) == "laneChange" and str(ins.request.side) == "right" and ins.request.laneChanges == 1
  assert [bool(x.active) for x in ins.lanes] == [False, True] and ins.laneDistance > 0
  assert values.get("NavDesire") == "laneChange:right"


def test_lights_before_the_turn_bring_the_deadline_forward():
  """A traffic signal 80 m before the turn: the changes are to be done signal_room before it (a queue there)."""
  t = lg.Tune()
  lanes_ = [lane("straight"), lane("right", True)]
  plain = first_request(Drive(made_route([(2000.0, "turn", "right", lanes_)])), 12.0, (0, 2))[0]
  lit = first_request(Drive(made_route([(2000.0, "turn", "right", lanes_)], signals=(1920.0,))), 12.0, (0, 2))
  assert lit[1].request["distance"] == pytest.approx(1920.0 - t.signal_room - lit[0], abs=3.0)
  assert plain - lit[0] == pytest.approx(80.0 + t.signal_room - t.done_before_turn, abs=3.0)


def test_blinker_off_before_the_other_side():
  """The blinker still on the other side (the driver's, a turn's): off first, then the change's own after a gap, as
  openpilot starts a change on the blinker's edge."""
  d = Drive(turn_right_two_lanes())
  d.blinker = "left"
  out = d.run_until(12.0, (0, 2), lambda o: o.requests, 300.0)
  assert out.requests == ["cancelSignal"]
  out = d.run_until(12.0, (0, 2), lambda o: o.requests, 2.0)
  assert out.requests == ["laneChangeRight"]


def test_approach_lanes_not_another_maneuvers():
  """A maneuver whose own intersection has no lanes takes the approach's, not those of the maneuver just before it."""
  exit_lanes = [lane("straight")] * 3 + [lane("slight right", True)]
  route = made_route([(1500.0, "off ramp", "slight right", exit_lanes), (1550.0, "fork", "slight left", None)],
                     extra=[(1530.0, [lane("slight left", True), lane("straight")])])
  mr = lg.MapboxRoute(route)
  fork = next(m for m in mr.moves if m.kind == "fork")
  assert [x["indications"] for x in fork.lanes] == [["slight left"], ["straight"]]
  route = made_route([(1500.0, "off ramp", "slight right", exit_lanes), (1550.0, "fork", "slight left", None)])
  fork = next(m for m in lg.MapboxRoute(route).moves if m.along == 1550.0)
  assert fork.lanes is None and lg.targets(fork, 2) == [0]  # the fallback, not the exit's lanes
