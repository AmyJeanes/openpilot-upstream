import os
from types import SimpleNamespace

import numpy as np

import openpilot.tools.sim.bridge.gta5.gta5_brain as brain_mod
from openpilot.tools.sim.bridge.gta5.gta5_brain import Brain


class FakeIO:
  """An OpenpilotIO (gta5_opio.py) that keeps what the brain tells openpilot."""
  def __init__(self):
    self.calls: list[tuple] = []
    self.inputs: dict = {}
    self.params = SimpleNamespace(get=lambda k: None, get_bool=lambda k: True, put=lambda k, v: None, remove=lambda k: None)

  def __getattr__(self, op):
    return lambda *args: self.calls.append((op, *args))

  def set_inputs(self, **inputs):
    self.inputs.update(inputs)

  def sent(self, op: str) -> list:
    return [c[1:] for c in self.calls if c[0] == op]


def brain() -> Brain:
  b = Brain.__new__(Brain)
  b.io = FakeIO()
  return b


def test_lane_slots_preview():
  from openpilot.selfdrive.navd import lane_slots as ls
  from openpilot.selfdrive.navd import route_input as ri
  from openpilot.tools.sim.bridge.gta5.test_lane_slots import RIGHT, fixture, route
  osm, nodes = fixture('lht.osm', drive_on_right=False)
  r = route(osm, nodes, [1, 2, 4], {1: RIGHT})
  r.at, r.off = 20.0, 1.0
  w = brain()
  w.noo, w.noo_t = True, float("inf")  # Navigate on openpilot on, the param read
  w.route_input, w.route = None, r
  w.navigator = SimpleNamespace(router=SimpleNamespace(osm=osm))
  state = {"vEgo": 10.0, "heading": 0.0}
  w._write_route_input(state)
  w._write_lane_slots(state)
  enc = w.route_input[1]
  w._write_lane_slots(state)
  assert w.route_input[1] is enc  # once per route, for the model's input and the preview
  writes, inputs = [c[0] for c in w.io.sent("lane_slots")], [c[0] for c in w.io.sent("route_input")]
  assert writes[-1][ls.SIDE] == -1.0 and ls.describe(writes[-1][:ls.LANE_SLOTS_LEN]) == 'here aTo..... out ao...... | from 0 by 50 m'
  np.testing.assert_array_equal(inputs[-1][ri.LANES], writes[-1][:ls.LANE_SLOTS_LEN])
  r.off = brain_mod.OFF_ROUTE_INPUT + 1.0
  w._write_lane_slots(state)
  w._write_route_input(state)
  writes, inputs = [c[0] for c in w.io.sent("lane_slots")], [c[0] for c in w.io.sent("route_input")]
  assert len(writes) == 3 and not writes[-1].any() and not inputs[-1].any()


def test_map_lane_for_nav():
  from openpilot.tools.sim.bridge.gta5.map.test_lane_match import matcher
  w = brain()
  w.lane_matcher, w.junction_areas = matcher(), None
  state = {"pos": [-3.0, -150.0, 0.5], "heading": 0.0}  # northbound in the divided road's southbound lanes
  read = w._map_lane(state)
  assert {k: v for k, v in read.items() if k not in ("way", "right")} == {"lane": -1, "lanes": 2, "kind": "oncoming", "bay": False,
                                                                        "oncoming": True, "beside": ["oncoming", "own"], "areas": False}
  assert isinstance(read["way"], int) and read["right"] < 0.0  # left of the way's line, in its southbound lanes
  w.junction_areas = SimpleNamespace(inside=lambda x, y, z: False)
  assert w._map_lane(state)["areas"] is True
  w.junction_areas = SimpleNamespace(inside=lambda x, y, z: True)
  assert w._map_lane(state) is None  # in a junction's area
  w.lane_matcher = None
  assert w._map_lane(state) is None


def e2e_module():
  """e2e.py, without the live stack's OPENPILOT_PREFIX it defaults to reaching the other tests."""
  had = "OPENPILOT_PREFIX" in os.environ
  from openpilot.tools.sim.bridge.gta5 import e2e
  if not had:
    os.environ.pop("OPENPILOT_PREFIX", None)
  return e2e


def test_e2e_oncoming_times():
  e2e = e2e_module()
  pts = [{"t": 0.5 * k, "v": 5.0, "lane": [0, 2], "mlane": [0, 2, "own"]} for k in range(20)]
  for k in range(4, 10):  # 3 s the wrong way on a one-way, which the game's reading misses
    pts[k].update(lane=None, mlane=[-1, 0, "wrong-way"])
  for k in range(10, 14):  # on through a junction, where the map can't say, and the game's reading sees it the last 1 s
    pts[k].update(lane=[-1, 2] if k >= 12 else None, mlane=None)
  t = e2e.oncoming_times(pts)
  assert t == {"oncoming_s": 1.0, "oncoming_max": 1.0, "oncoming_map_s": 3.0, "oncoming_map_max": 3.0, "oncoming_any_s": 4.0,
               "oncoming_any_max": 4.0}
  assert not e2e.safe({"outcome": "arrived", **t})
  assert e2e.safe({"outcome": "arrived", "oncoming_s": 0.5, "oncoming_max": 0.5})  # results from before the map's reading
  pts[5]["v"] = 0.5  # stopped: not counted, and not over
  assert e2e.oncoming_times(pts)["oncoming_map_max"] == 2.5


def test_e2e_lane_map():
  from openpilot.tools.sim.bridge.gta5.map.lane_match import JunctionAreas
  from openpilot.tools.sim.bridge.gta5.map.test_junctions import make
  from openpilot.tools.sim.bridge.gta5.map.test_lane_match import HEIGHTS, NODES, WAYS
  e2e = e2e_module()
  lm = e2e.LaneMap(make(NODES, WAYS, HEIGHTS), JunctionAreas(np.zeros((0, 2)), [], np.zeros(0)))
  assert lm.read(-3.0, -150.0, 0.5, 0.0) == [-1, 2, "oncoming"]
  assert lm.read(0.6, -70.0, 0.5, 0.0) == [-1, 0, "bay"]  # the southbound left turn bay in the median
  assert lm.read(6.0, -150.0, 0.5, 180.0) == [-2, 2, "oncoming"]  # southbound in the far northbound lane
  # a trip's start: the middle of its lane, not the road's line
  x, y, h = lm.lane_start(0.0, -150.0, 0.5, 0.0, 9)
  assert abs(x - 6.25) < 1e-6 and abs(y + 150.0) < 1e-6 and abs(h) < 1e-6
  assert abs(lm.lane_start(0.0, -150.0, 0.5, 180.0, 0)[0] + 2.75) < 1e-6
  lm.areas = SimpleNamespace(inside=lambda x, y, z: True)
  assert lm.read(-3.0, -150.0, 0.5, 0.0) is None


def test_lane_slots_failure_keeps_v1():
  from openpilot.selfdrive.navd import route_input as ri
  from openpilot.tools.sim.bridge.gta5.test_lane_slots import RIGHT, fixture, route
  osm, nodes = fixture('lht.osm', drive_on_right=False)
  r = route(osm, nodes, [1, 2, 4], {1: RIGHT})
  r.at, r.off = 20.0, 1.0
  w = brain()
  w.noo, w.noo_t = True, float("inf")
  w.route_input, w.route = None, r
  w.navigator = SimpleNamespace(router=SimpleNamespace(osm=osm))
  w._route_encoder(r).slots.encode = None  # broken: calling it raises
  w._write_route_input({"vEgo": 10.0, "heading": 0.0})
  inputs = [c[0] for c in w.io.sent("route_input")]
  assert w.route_input[1].slots is None and inputs[-1][ri.PRESENT] == 1.0 and not inputs[-1][ri.LANES].any()


def test_nav_speed_cap(monkeypatch):
  """nav's cap goes to openpilot's planner as navSpeed (the car's set speed stays the driver's), at most every
  NAV_SPEED_EVERY unless its reason changes; with GTA5_NAV_SPEED=can, the simulated car shows it as its set speed."""
  now = [100.0]
  monkeypatch.setattr(brain_mod.time, "monotonic", lambda: now[0])
  w = brain()
  w.cap, w.cap_reason, w.next_nav_speed = 0.0, "", 0.0
  w._set_cap(6.0, "turnLeft")
  assert w.io.sent("nav_speed") == [(6.0, "turnLeft")] and w.io.inputs == {"cruise_cap": 6.0, "cap_set_speed": False}
  now[0] += 0.01
  w._set_cap(5.9, "turnLeft")
  assert len(w.io.sent("nav_speed")) == 1 and w.cap == 5.9 and w.io.inputs["cruise_cap"] == 5.9
  w._set_cap(5.8, "arrival")
  assert w.io.sent("nav_speed")[-1] == (5.8, "arrival")
  now[0] += brain_mod.NAV_SPEED_EVERY
  w._set_cap(0.0, "")
  assert w.io.sent("nav_speed")[-1] == (0.0, "none") and len(w.io.sent("nav_speed")) == 3

  monkeypatch.setattr(brain_mod, "NAV_SPEED", "can")
  w._set_cap(7.0, "bendLeft")
  assert len(w.io.sent("nav_speed")) == 3 and w.io.inputs == {"cruise_cap": 7.0, "cap_set_speed": True}


def test_mark_key(monkeypatch, tmp_path):
  """Each new mark press sends a userBookmark at once and, MARK_AFTER s later, is a line in GTA5_MARKS: where, what drove,
  the recording's segment and frame, the map driver's plan and the route about the car, and the car's track from
  MARK_BEFORE s before the press; the first state seen and a reloaded core's count from 0 again are no mark."""
  import json
  now = [1000.0]
  monkeypatch.setattr(brain_mod.time, "monotonic", lambda: now[0])
  marks = tmp_path / "marks.jsonl"
  monkeypatch.setattr(brain_mod, "MARKS", str(marks))
  w = brain()
  w.presses = {}
  w.track, w.lane_lines, w.marks_due = brain_mod.deque(maxlen=200), brain_mod.deque(maxlen=20), []
  w.lane_line = (None, 0.0, [[0.0, 0.0], [1.0, 1.0]])
  w.openpilot = w.driver = w.driving = SimpleNamespace(name="openpilot")
  w.expert = SimpleNamespace(md=None, md_driving=False, source="none")
  pts = np.array([[0.0, float(y), 0.0] for y in range(0, 1001, 10)])
  w.route = SimpleNamespace(at=500.04, length=1000.0, points=pts, along=pts[:, 1].copy())
  w.recorder = SimpleNamespace(segment=SimpleNamespace(path=brain_mod.Path("/rec/data/6acb00000012"), rows=[0] * 345),
                               route_name="2026-10-12--15-00-00")
  state = {"t": 5.0, "pos": [1.234, 2.0, 30.0], "heading": 90.0, "vEgo": 8.0, "street": "Elgin Ave", "markPresses": 3}

  def drive(seconds, **kw):
    for _ in range(int(seconds / 0.05)):
      now[0] += 0.05
      w._update_buttons({**state, "pos": [0.0, (now[0] - 1000.0) * 8.0, 30.0], **kw})

  drive(40.0)
  assert not marks.exists() and not w.io.sent("bookmark")
  drive(0.05, markPresses=4)
  assert w.io.sent("bookmark") == [()] and not marks.exists()
  drive(brain_mod.MARK_AFTER + 0.1, markPresses=4)
  m = json.loads(marks.read_text())
  assert m["driver"] == "openpilot" and m["street"] == "Elgin Ave" and m["segment"] == "6acb00000012" and m["frame"] == 345
  assert m["route"] == {"at": 500.0, "length": 1000.0, "dest": [0.0, 1000.0]} and m["plan"] is None
  assert len(m["route_line"]) == 38 and m["route_line"][0] == [0.0, 210.0] and m["lane_line"] == [[0.0, 0.0], [1.0, 1.0]]
  t = np.array(m["track"])
  assert abs(t[0, 0] - (m["mono"] - brain_mod.MARK_BEFORE)) < 0.25 and abs(t[-1, 0] - (m["mono"] + brain_mod.MARK_AFTER)) < 0.25
  drive(0.05, markPresses=0)
  md = SimpleNamespace(info=lambda: {"phase": "drive"}, s=400.0, s_path=np.arange(0.0, 1000.0), path=np.c_[np.zeros(1000), np.arange(1000.0)])
  w.expert = w.driving = SimpleNamespace(md=md, md_driving=True, source="map")
  w.recorder = None
  drive(0.05, markPresses=1)
  w._flush_marks()  # as the bridge closes: a mark still waiting is written with the track it has
  lines = marks.read_text().splitlines()
  m = json.loads(lines[-1])
  assert len(lines) == 2 and m["driver"] == "map" and m["map"] == {"phase": "drive"} and m["segment"] is None
  assert m["plan"][0] == [0.0, 100.0] and m["plan"][-1] == [0.0, 480.0]
