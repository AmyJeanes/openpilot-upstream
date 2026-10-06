from types import SimpleNamespace

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
