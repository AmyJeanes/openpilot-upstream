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

  def get(self, key):
    return self.values.get(key)

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


def test_destination_ended_by_the_driver():
  # the UI's slide to end removes NavDestination: the destination goes and the game's waypoint (left set in the game)
  # is ignored, until the player sets another
  params, sent = Params(), []
  dest = Destination(params, sent.append)
  dest.check_every = 0.0
  car, wp = np.array([0.0, 0.0]), [1000.0, 2000.0]
  assert dest.update(wp, car, None) is not None and "NavDestination" in params.values
  params.remove("NavDestination")
  assert dest.update(wp, car, None) is None and "NavDestination" not in params.values and sent == []
  assert dest.update(wp, car, None) is None
  assert list(dest.update([1500.0, 2000.0], car, None)) == [1500.0, 2000.0] and "NavDestination" in params.values
  # a map view pick after an ended route counts too, and so does the old waypoint once the game has cleared it
  params.remove("NavDestination")
  assert dest.update([1500.0, 2000.0], car, None) is None
  assert list(dest.update(None, car, ([5.0, 6.0],))) == [5.0, 6.0]


def test_destination_not_ended_when_nav_drives_regardless():
  # GTA5_NOO=on (test runs): a NavDestination the manager clears doesn't end the drive
  params, sent = Params(), []
  dest = Destination(params, sent.append, ends=False)
  dest.check_every = 0.0
  car, wp = np.array([0.0, 0.0]), [1000.0, 2000.0]
  assert dest.update(wp, car, None) is not None
  params.remove("NavDestination")
  assert list(dest.update(wp, car, None)) == wp


def test_destination_from_a_mission_route():
  # with no waypoint, a mission's GPS route's end is the destination; the player's waypoint wins while there is one
  params, sent = Params(), []
  dest = Destination(params, sent.append)
  car, wp, mission = np.array([0.0, 0.0]), [1000.0, 2000.0], [-500.0, 300.0]
  assert list(dest.update(None, car, None, mission)) == mission and "NavDestination" in params.values
  assert list(dest.update(wp, car, None, mission)) == wp
  # the player clears it far from it: back to the mission's
  assert list(dest.update(None, car, None, mission)) == mission
  # a blip on a moving car: small moves keep the route, a big one is a new destination
  assert list(dest.update(None, car, None, [-510.0, 310.0])) == mission
  assert list(dest.update(None, car, None, [-560.0, 300.0])) == [-560.0, 300.0]
  # arriving leaves the mission's blip alone, and the same blip doesn't come back as a new destination
  dest.arrived()
  assert dest.dest is None and sent == [] and "NavDestination" not in params.values
  assert dest.update(None, car, None, [-560.0, 300.0]) is None
  # the mission takes its blip away from afar: the destination goes; a new one counts
  assert list(dest.update(None, car, None, [100.0, 100.0])) == [100.0, 100.0]
  assert dest.update(None, car, None, None) is None
  assert list(dest.update(None, car, None, [100.0, 100.0])) == [100.0, 100.0]


def test_destination_waypoint_cleared_on_arrival_with_a_mission():
  # GTA clears the waypoint as the car nears it: the mission's route waits until the car has arrived
  params, sent = Params(), []
  dest = Destination(params, sent.append)
  wp, mission = [1000.0, 2000.0], [-500.0, 300.0]
  assert list(dest.update(wp, np.array([0.0, 0.0]), None, mission)) == wp
  near = np.array([1000.0, 2000.0 - CANCELLED_FROM + 10.0])
  assert list(dest.update(None, near, None, mission)) == wp
  dest.arrived()
  assert sent == [{"type": "waypoint", "off": True}]
  assert list(dest.update(None, near, None, mission)) == mission


def test_destination_mission_route_ended_by_the_driver():
  # slide to end: the mission's route is ignored until it moves on (the mission isn't ours to end)
  params, sent = Params(), []
  dest = Destination(params, sent.append)
  dest.check_every = 0.0
  car, mission = np.array([0.0, 0.0]), [-500.0, 300.0]
  assert dest.update(None, car, None, mission) is not None
  params.remove("NavDestination")
  assert dest.update(None, car, None, mission) is None and sent == []
  assert dest.update(None, car, None, [-505.0, 300.0]) is None
  assert list(dest.update(None, car, None, [-600.0, 300.0])) == [-600.0, 300.0]


def test_destination_waypoint_set_on_foot():
  # the bridge sees no states on foot; a waypoint set meanwhile is new when the car's states come again
  params, sent = Params(), []
  dest = Destination(params, sent.append)
  car = np.array([0.0, 0.0])
  assert dest.update(None, car, None) is None
  assert list(dest.update([1000.0, 2000.0], car, None)) == [1000.0, 2000.0]
  # and one changed on foot replaces the old
  assert list(dest.update([1500.0, 2000.0], car, None)) == [1500.0, 2000.0]
