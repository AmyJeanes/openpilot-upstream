"""Approving navigation's lane change request (PLAN D): the driver's blinker on the requested side fires the lane change
through DesireHelper at any speed (NavDesire "laneChange:<side>"), with the nudge and blind spot rules as for any lane
change; the other side's blinker stays a turn below the lane turn speed."""
import sys
import types
from types import SimpleNamespace

try:
  from openpilot.common.params import Params  # noqa: F401
except OSError:  # params not built (a bare checkout): a dict stands in for NavDesire
  stub = types.ModuleType("openpilot.common.params")

  class Params:
    store: dict = {}

    def get(self, key):
      return self.store.get(key)

    def get_bool(self, key):
      return bool(self.store.get(key))

    def put(self, key, value):
      self.store[key] = value

  stub.Params = Params
  sys.modules["openpilot.common.params"] = stub

from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib import desire_helper as dh

LCS = log.LaneChangeState


def car_state(v=5.0, left=False, right=False, torque=0.0, left_bs=False, right_bs=False):
  return SimpleNamespace(vEgo=v, leftBlinker=left, rightBlinker=right, steeringPressed=abs(torque) > 0,
                         steeringTorque=torque, leftBlindspot=left_bs, rightBlindspot=right_bs)


class Helper:
  """A DesireHelper with NavDesire held as given."""
  def __init__(self, nav: str):
    self.h = dh.DesireHelper()
    self.h.nav.get = lambda fresh=False: nav

  def run(self, secs: float, **cs) -> None:
    for _ in range(round(secs / DT_MDL)):
      self.h.update(car_state(**cs), True, 0.5)


def test_nav_lane_change_sides():
  assert dh.nav_lane_change("laneChange", True, False) and dh.nav_lane_change("laneChange", False, True)
  assert dh.nav_lane_change("laneChange:right", False, True)
  assert not dh.nav_lane_change("laneChange:right", True, False)
  assert not dh.nav_lane_change("laneChange:left", False, False)
  assert not dh.nav_lane_change("keepLeft", True, False) and not dh.nav_lane_change("", True, False)


def test_driver_blinker_on_the_requested_side_approves():
  """Below 20 mph, nav asking for a change right: the right blinker waits for the nudge (preLaneChange), the nudge
  starts it, and the model gets laneChangeRight."""
  d = Helper("laneChange:right")
  d.run(0.5, v=6.0)
  d.run(0.5, v=6.0, right=True)
  assert d.h.lane_change_state == LCS.preLaneChange and d.h.desire == log.Desire.none
  d.run(0.2, v=6.0, right=True, torque=-2.0)
  assert d.h.lane_change_state == LCS.laneChangeStarting and d.h.desire == log.Desire.laneChangeRight


def test_other_side_is_still_a_turn():
  d = Helper("laneChange:right")
  d.run(0.5, v=6.0)
  d.run(0.5, v=6.0, left=True)
  assert d.h.lane_change_state == LCS.off and d.h.desire == log.Desire.turnLeft


def test_blind_spot_holds_the_approval():
  """Approved, but a vehicle in the blind spot that side: the nudge doesn't start it until it's been clear
  BLINDSPOT_CLEAR_TIME."""
  d = Helper("laneChange:left")
  d.run(0.5, v=6.0)
  d.run(0.5, v=6.0, left=True, left_bs=True)
  d.run(0.5, v=6.0, left=True, torque=2.0, left_bs=True)
  assert d.h.lane_change_state == LCS.preLaneChange
  d.run(dh.BLINDSPOT_CLEAR_TIME - 0.1, v=6.0, left=True, torque=2.0)
  assert d.h.lane_change_state == LCS.preLaneChange
  d.run(0.2, v=6.0, left=True, torque=2.0)
  assert d.h.lane_change_state == LCS.laneChangeStarting and d.h.desire == log.Desire.laneChangeLeft


def test_no_request_below_the_lane_change_speed_is_a_turn():
  d = Helper("")
  d.run(0.5, v=6.0)
  d.run(0.5, v=6.0, right=True, torque=-2.0)
  assert d.h.lane_change_state == LCS.off and d.h.desire == log.Desire.turnRight
