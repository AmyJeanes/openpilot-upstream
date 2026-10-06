import math

from openpilot.common.test import OpenpilotTestCase

from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.modeld.constants import ModelConstants


class TestTurnDesireRefresh(OpenpilotTestCase):

  def setUp(self):
    super().setUp()
    self.DH = DesireHelper()
    self.DH.turn_refresh = True
    self.t = 0.0

  def drive(self, seconds, v_ego=5.0, yaw_rate=0.0, turn_prob=1.0, blinker=True):
    """Times (s) the turn was asked for again over these seconds."""
    CS = car.CarState.new_message(vEgo=v_ego, leftBlinker=blinker)
    desire_state = [0.0] * ModelConstants.DESIRE_LEN
    desire_state[log.Desire.turnLeft] = turn_prob
    refreshed = []
    for _ in range(round(seconds / DT_MDL)):
      self.DH.update(CS, True, 0.0, yaw_rate, desire_state)
      self.t += DT_MDL
      if self.DH.refresh:
        refreshed.append(self.t)
    return refreshed

  def test_off(self):
    self.DH.turn_refresh = False
    assert self.drive(20.0) == []
    assert self.DH.desire == log.Desire.turnLeft

  def test_every_few_seconds_until_turning(self):
    refreshed = self.drive(10.0)
    assert len(refreshed) == 3
    assert all(abs(b - a - 2.55) < 0.06 for a, b in zip(refreshed, refreshed[1:], strict=False))
    # 15 deg turned at 0.05 rad/s takes ~5 s, after which it isn't asked for again
    self.drive(1.0, blinker=False)
    assert len(self.drive(20.0, yaw_rate=0.05)) == 2

  def test_sooner_once_the_model_forgets(self):
    refreshed = self.drive(10.0, turn_prob=0.05)
    assert len(refreshed) == 4
    assert abs(refreshed[0] - 2.05) < 0.06

  def test_moving_off_after_a_stop(self):
    self.drive(1.0, yaw_rate=math.radians(20.0))  # 20 deg turned, then a long stop
    assert self.drive(30.0, v_ego=0.0) == []
    assert len(self.drive(0.5)) == 1

  def test_not_after_a_stop_well_into_the_turn(self):
    self.drive(1.0, yaw_rate=math.radians(40.0))
    self.drive(30.0, v_ego=0.0)
    assert self.drive(5.0) == []

  def test_new_turn_starts_afresh(self):
    self.drive(1.0, yaw_rate=math.radians(40.0))
    self.drive(1.0, blinker=False)
    assert len(self.drive(3.0)) == 1
