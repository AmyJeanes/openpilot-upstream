import os
from types import SimpleNamespace

import numpy as np
from cereal import log

import openpilot.tools.sim.bridge.gta5.gta5_world as world_mod
from openpilot.tools.sim.bridge.gta5.gta5_world import NUDGE_TIMEOUT, NUDGE_TORQUE, GTA5World

LaneChangeState = log.LaneChangeState


class Clock:
  def __init__(self):
    self.t = 100.0

  def monotonic(self):
    return self.t


def indicator_world(lane_change):
  w = GTA5World.__new__(GTA5World)
  w.indicator, w.indicator_t, w.indicator_heading, w.lane_changing = None, 0.0, 0.0, False
  w.sm = {'modelV2': SimpleNamespace(meta=SimpleNamespace(laneChangeState=lane_change))}
  w.nav = SimpleNamespace(signaling=False)
  w.sent = []
  w._send = w.sent.append
  return w


def test_nudge_held_every_step_until_the_timeout():
  # the car thread samples the torque at its own phase, so the nudge must be the step's whole value, not a moment of it
  assert GTA5World.sets_torque
  clock = Clock()
  world_mod.time = clock
  w = indicator_world(LaneChangeState.preLaneChange)
  torques = []
  while clock.t < 100.0 + NUDGE_TIMEOUT + 1.0:
    torques.append((clock.t - 100.0, w._update_indicator("right", 0.0, 0.0)))
    clock.t += 0.01
  assert all(tq == -NUDGE_TORQUE for t, tq in torques if t < NUDGE_TIMEOUT - 0.01)
  assert all(tq == 0 for t, tq in torques if t > NUDGE_TIMEOUT + 0.01)
  assert indicator_world(LaneChangeState.preLaneChange)._update_indicator("left", 0.0, 0.0) == NUDGE_TORQUE


def test_no_nudge_once_the_lane_change_starts_or_for_a_turn():
  world_mod.time = Clock()
  assert indicator_world(LaneChangeState.laneChangeStarting)._update_indicator("left", 0.0, 0.0) == 0
  w = indicator_world(LaneChangeState.preLaneChange)
  w.nav.signaling = True
  assert w._update_indicator("left", 0.0, 0.0) == 0
  assert indicator_world(LaneChangeState.preLaneChange)._update_indicator(None, 0.0, 0.0) == 0


def test_lane_slots_preview():
  from openpilot.tools.sim.bridge.gta5 import gta5_lane_slots as ls
  from openpilot.tools.sim.bridge.gta5.test_lane_slots import RIGHT, fixture, route
  osm, nodes = fixture('lht.osm', drive_on_right=False)
  r = route(osm, nodes, [1, 2, 4], {1: RIGHT})
  r.at, r.off = 20.0, 1.0
  w = GTA5World.__new__(GTA5World)
  writes = []
  w.lanes_writer = SimpleNamespace(write=writes.append)
  w.lane_slots, w.route = None, r
  w.navigator = SimpleNamespace(router=SimpleNamespace(osm=osm))
  w._write_lane_slots({"vEgo": 10.0})
  slots = w.lane_slots[1]
  w._write_lane_slots({"vEgo": 10.0})
  assert w.lane_slots[1] is slots  # once per route
  assert writes[-1][ls.SIDE] == -1.0 and ls.describe(writes[-1][:ls.LANE_SLOTS_LEN]) == 'here aTo..... out ao...... | from 0 by 50 m'
  r.off = world_mod.OFF_ROUTE_INPUT + 1.0
  w._write_lane_slots({"vEgo": 10.0})
  assert len(writes) == 3 and not writes[-1].any()


def test_map_lane_for_nav():
  from openpilot.tools.sim.bridge.gta5.map.test_lane_match import matcher
  w = GTA5World.__new__(GTA5World)
  w.lane_matcher, w.junction_areas = matcher(), None
  state = {"pos": [-3.0, -150.0, 0.5], "heading": 0.0}  # northbound in the divided road's southbound lanes
  assert w._map_lane(state) == {"lane": -1, "lanes": 2, "kind": "oncoming", "bay": False, "oncoming": True, "areas": False}
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
