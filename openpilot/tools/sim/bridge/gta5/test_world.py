import os
from types import SimpleNamespace

import numpy as np

import openpilot.tools.sim.bridge.gta5.gta5_world as world_mod
from openpilot.tools.sim.bridge.gta5.gta5_world import GTA5World


def test_lane_slots_preview():
  from openpilot.selfdrive.navd import lane_slots as ls
  from openpilot.selfdrive.navd import route_input as ri
  from openpilot.tools.sim.bridge.gta5.test_lane_slots import RIGHT, fixture, route
  osm, nodes = fixture('lht.osm', drive_on_right=False)
  r = route(osm, nodes, [1, 2, 4], {1: RIGHT})
  r.at, r.off = 20.0, 1.0
  w = GTA5World.__new__(GTA5World)
  w.noo, w.noo_t = True, float("inf")  # Navigate on openpilot on, the param read
  writes, inputs = [], []
  w.lanes_writer = SimpleNamespace(write=writes.append)
  w.route_writer = SimpleNamespace(write=inputs.append)
  w.route_input, w.route = None, r
  w.navigator = SimpleNamespace(router=SimpleNamespace(osm=osm))
  state = {"vEgo": 10.0, "heading": 0.0}
  w._write_route_input(state)
  w._write_lane_slots(state)
  enc = w.route_input[1]
  w._write_lane_slots(state)
  assert w.route_input[1] is enc  # once per route, for the model's input and the preview
  assert writes[-1][ls.SIDE] == -1.0 and ls.describe(writes[-1][:ls.LANE_SLOTS_LEN]) == 'here aTo..... out ao...... | from 0 by 50 m'
  np.testing.assert_array_equal(inputs[-1][ri.LANES], writes[-1][:ls.LANE_SLOTS_LEN])
  r.off = world_mod.OFF_ROUTE_INPUT + 1.0
  w._write_lane_slots(state)
  w._write_route_input(state)
  assert len(writes) == 3 and not writes[-1].any() and not inputs[-1].any()


def test_map_lane_for_nav():
  from openpilot.tools.sim.bridge.gta5.map.test_lane_match import matcher
  w = GTA5World.__new__(GTA5World)
  w.lane_matcher, w.junction_areas = matcher(), None
  state = {"pos": [-3.0, -150.0, 0.5], "heading": 0.0}  # northbound in the divided road's southbound lanes
  assert w._map_lane(state) == {"lane": -1, "lanes": 2, "kind": "oncoming", "bay": False, "oncoming": True,
                                "beside": ["oncoming", "own"], "areas": False}
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
  w = GTA5World.__new__(GTA5World)
  w.noo, w.noo_t = True, float("inf")
  inputs = []
  w.route_writer = SimpleNamespace(write=inputs.append)
  w.route_input, w.route = None, r
  w.navigator = SimpleNamespace(router=SimpleNamespace(osm=osm))
  w._route_encoder(r).slots.encode = None  # broken: calling it raises
  w._write_route_input({"vEgo": 10.0, "heading": 0.0})
  assert w.route_input[1].slots is None and inputs[-1][ri.PRESENT] == 1.0 and not inputs[-1][ri.LANES].any()


def test_nav_speed_cap(monkeypatch):
  """nav's cap goes to openpilot's planner as navSpeed (the car's set speed stays the driver's), at most every
  NAV_SPEED_EVERY unless its reason changes; with GTA5_NAV_SPEED=can, the simulated car shows it as its set speed."""
  from openpilot.tools.sim.lib.common import SimulatorState
  now = [100.0]
  monkeypatch.setattr(world_mod.time, "monotonic", lambda: now[0])
  sent = []
  w = GTA5World.__new__(GTA5World)
  w.cap, w.cap_reason, w.next_nav_speed = 0.0, "", 0.0
  w.nav_speed_pm = SimpleNamespace(send=lambda s, m: sent.append((s, round(m.navSpeed.speedCap, 2), str(m.navSpeed.reason))))
  s = SimulatorState()
  w._set_cap(s, 6.0, "turnLeft")
  assert sent == [("navSpeed", 6.0, "turnLeft")] and s.cruise_cap == 6.0 and not s.cap_set_speed
  now[0] += 0.01
  w._set_cap(s, 5.9, "turnLeft")
  assert len(sent) == 1 and w.cap == 5.9
  w._set_cap(s, 5.8, "arrival")
  assert sent[-1] == ("navSpeed", 5.8, "arrival")
  now[0] += world_mod.NAV_SPEED_EVERY
  w._set_cap(s, 0.0, "")
  assert sent[-1] == ("navSpeed", 0.0, "none") and len(sent) == 3

  w.nav_speed_pm = None  # GTA5_NAV_SPEED=can
  w._set_cap(s, 7.0, "bendLeft")
  assert len(sent) == 3 and s.cruise_cap == 7.0 and s.cap_set_speed
