"""The GTA layer around navd: the simulated driver and the destination."""
import numpy as np
from cereal import log

import openpilot.tools.sim.bridge.gta5.gta5_driver as driver_mod
from openpilot.selfdrive.navd.inputs import NavOutputs
from openpilot.tools.sim.bridge.gta5.gta5_driver import GO_GAS, NUDGE_TIMEOUT, NUDGE_TORQUE, Driver, PullAway
from openpilot.tools.sim.bridge.gta5.gta5_navd import CANCELLED_FROM, Destination
from openpilot.tools.sim.bridge.gta5.gta5_world import GTA5World
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game

LaneChangeState = log.LaneChangeState


class Clock:
  def __init__(self):
    self.t = 100.0

  def monotonic(self):
    return self.t



def driver():
  sent = []
  return Driver(sent.append, lambda: sent.append("cancel")), sent


def test_nudge_held_every_step_until_the_timeout():
  # the car thread samples the torque at its own phase, so the nudge must be the step's whole value, not a moment of it
  assert GTA5World.sets_torque
  clock = Clock()
  driver_mod.time = clock
  d, _ = driver()
  torques = []
  while clock.t < 100.0 + NUDGE_TIMEOUT + 1.0:
    torques.append((clock.t - 100.0, d.stalk("right", 0.0, 0.0, LaneChangeState.preLaneChange, False)))
    clock.t += 0.01
  assert all(tq == -NUDGE_TORQUE for t, tq in torques if t < NUDGE_TIMEOUT - 0.01)
  assert all(tq == 0 for t, tq in torques if t > NUDGE_TIMEOUT + 0.01)
  assert driver()[0].stalk("left", 0.0, 0.0, LaneChangeState.preLaneChange, False) == NUDGE_TORQUE


def test_no_nudge_once_the_lane_change_starts_or_for_a_turn():
  driver_mod.time = Clock()
  assert driver()[0].stalk("left", 0.0, 0.0, LaneChangeState.laneChangeStarting, False) == 0
  assert driver()[0].stalk("left", 0.0, 0.0, LaneChangeState.preLaneChange, True) == 0
  assert driver()[0].stalk(None, 0.0, 0.0, LaneChangeState.preLaneChange, False) == 0


def test_stalk_cancels_after_the_turn_and_after_a_lane_change():
  clock = Clock()
  driver_mod.time = clock
  d, sent = driver()
  d.stalk("left", 0.0, 0.0, LaneChangeState.off, True)
  d.stalk("left", 45.0, 0.3, LaneChangeState.off, True)
  assert sent == []  # still turning
  d.stalk("left", 80.0, 0.05, LaneChangeState.off, True)
  assert sent == [{"type": "indicatorOff"}]
  d, sent = driver()
  d.stalk("right", 0.0, 0.0, LaneChangeState.laneChangeStarting, False)
  d.stalk("right", 0.0, 0.0, LaneChangeState.laneChangeFinishing, False)
  assert sent == [{"type": "indicatorOff"}]


def test_requests_become_the_stalk_and_arrival_cancels():
  d, sent = driver()
  d.act(NavOutputs(0.0, False, ["laneChangeRight", "cancelSignal", "signalTurnLeft"], []))
  assert sent == [{"type": "setIndicator", "side": "right"}, {"type": "indicatorOff"}, {"type": "setIndicator", "side": "left"}]
  d.act(NavOutputs(3.0, True, [], []))
  assert sent[-1] == "cancel"
  d.indicator = "left"
  assert d.blinkers(False) == (True, False) and d.blinkers(True) == (False, False)


def test_pull_away_at_green():
  clock = Clock()
  driver_mod.time = clock
  sent = []
  p = PullAway(sent.append)
  for k in range(200):
    red = k < 40
    p.update({"vEgo": 0.0, "traffic": {"red": int(red)}}, True)
    clock.t += 0.05
  assert sent[0] == {"type": "gas", "secs": GO_GAS} and len(sent) == 3  # every GO_EVERY s, GO_TRIES times
  sent.clear()
  p = PullAway(sent.append)
  for k in range(100):
    p.update({"vEgo": 0.0, "traffic": {"red": int(k < 40)}, "vehicleAhead": 8.0}, True)
    clock.t += 0.05
  assert sent == []  # led away by the car ahead


class Params:
  def __init__(self):
    self.values = {}

  def put(self, key, value):
    self.values[key] = value

  def remove(self, key):
    self.values.pop(key, None)


def test_destination_from_the_game_and_the_map_view():
  params, sent = Params(), []
  dest = Destination(params, sent.append)
  car = np.array([0.0, 0.0])
  assert dest.update(None, car, None) is None and "NavDestination" not in params.values
  assert list(dest.update([1000.0, 2000.0], car, None)) == [1000.0, 2000.0]
  lat, lon = params.values["NavDestination"]["latitude"], params.values["NavDestination"]["longitude"]
  assert np.allclose(to_game(lat, lon), (1000.0, 2000.0))
  # GTA clears its waypoint near it: the destination stays; cleared from further off, it goes
  near = np.array([1000.0, 2000.0 - CANCELLED_FROM + 10.0])
  assert dest.update(None, near, None) is not None
  assert dest.update(None, car, None) is None and "NavDestination" not in params.values
  # a map view pick replaces it, and arriving clears it and the game's waypoint
  assert list(dest.update(None, car, ([5.0, 6.0],))) == [5.0, 6.0]
  dest.arrived()
  assert dest.dest is None and sent == [{"type": "waypoint", "off": True}] and "NavDestination" not in params.values
