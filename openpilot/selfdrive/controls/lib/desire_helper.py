import time

from openpilot.cereal import log
from openpilot.common.constants import CV
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL

LaneChangeState = log.LaneChangeState
LaneChangeDirection = log.LaneChangeDirection

LANE_CHANGE_SPEED_MIN = 20 * CV.MPH_TO_MS
LANE_CHANGE_TIME_MAX = 10.
LANE_CHANGE_START_TIME = 0.5
# below this a blinker asks the model to take the next turn instead, as sunnypilot's lane turn desire does
LANE_TURN_SPEED = 19 * CV.MPH_TO_MS
# a navigation source's request (the NavDesire param): "laneChange" makes the blinker ask for a lane change at any speed
NAV_LANE_CHANGE = "laneChange"
# or one of the model's keep desires, for a fork, which openpilot itself never asks for
NAV_KEEP = {"keepLeft": log.Desire.keepLeft, "keepRight": log.Desire.keepRight}
NAV_READ_EVERY = 0.2  # s


class NavDesire:
  def __init__(self):
    self.params = Params()
    self.value = ""
    self.t = 0.0

  def get(self) -> str:
    now = time.monotonic()
    if now - self.t > NAV_READ_EVERY:
      self.value, self.t = self.params.get("NavDesire") or "", now
    return self.value


def lane_turn_desire(CS, nav: str = "") -> int:
  """The turn a blinker asks for at low speed, unless the blind spot that way is occupied; overrides any lane change."""
  if CS.vEgo < LANE_TURN_SPEED and nav != NAV_LANE_CHANGE:
    if CS.leftBlinker and not CS.rightBlinker and not CS.leftBlindspot:
      return log.Desire.turnLeft
    if CS.rightBlinker and not CS.leftBlinker and not CS.rightBlindspot:
      return log.Desire.turnRight
  return log.Desire.none


class DesireHelper:
  def __init__(self):
    self.lane_change_state = LaneChangeState.off
    self.lane_change_direction = LaneChangeDirection.none
    self.lane_change_timer = 0.0
    self.prev_one_blinker = False
    self.desire = log.Desire.none
    self.nav = NavDesire()

  @staticmethod
  def get_lane_change_direction(CS):
    return LaneChangeDirection.left if CS.leftBlinker else LaneChangeDirection.right

  def update(self, carstate, lateral_active, lane_change_prob):
    v_ego = carstate.vEgo
    one_blinker = carstate.leftBlinker != carstate.rightBlinker
    nav = self.nav.get()
    below_lane_change_speed = v_ego < LANE_CHANGE_SPEED_MIN and nav != NAV_LANE_CHANGE

    if not lateral_active or self.lane_change_timer > LANE_CHANGE_TIME_MAX:
      self.lane_change_state = LaneChangeState.off
      self.lane_change_direction = LaneChangeDirection.none
      self.lane_change_timer = 0.0
    else:
      if self.lane_change_state == LaneChangeState.off and one_blinker and not self.prev_one_blinker and not below_lane_change_speed:
        self.lane_change_state = LaneChangeState.preLaneChange
        self.lane_change_timer = 0.0
        # Initialize lane change direction to prevent UI alert flicker
        self.lane_change_direction = self.get_lane_change_direction(carstate)

      elif self.lane_change_state == LaneChangeState.preLaneChange:
        # Update lane change direction
        self.lane_change_direction = self.get_lane_change_direction(carstate)

        torque_applied = carstate.steeringPressed and \
                         ((carstate.steeringTorque > 0 and self.lane_change_direction == LaneChangeDirection.left) or
                          (carstate.steeringTorque < 0 and self.lane_change_direction == LaneChangeDirection.right))

        blindspot_detected = ((carstate.leftBlindspot and self.lane_change_direction == LaneChangeDirection.left) or
                              (carstate.rightBlindspot and self.lane_change_direction == LaneChangeDirection.right))

        if not one_blinker or below_lane_change_speed:
          self.lane_change_state = LaneChangeState.off
          self.lane_change_direction = LaneChangeDirection.none
          self.lane_change_timer = 0.0
        elif torque_applied and not blindspot_detected:
          self.lane_change_state = LaneChangeState.laneChangeStarting
          self.lane_change_timer = 0.0

      elif self.lane_change_state == LaneChangeState.laneChangeStarting:
        self.lane_change_timer += DT_MDL

        if lane_change_prob < 0.02 and self.lane_change_timer >= LANE_CHANGE_START_TIME:
          self.lane_change_timer = 0.0
          if one_blinker:
            self.lane_change_state = LaneChangeState.preLaneChange
            self.lane_change_direction = self.get_lane_change_direction(carstate)
          else:
            self.lane_change_state = LaneChangeState.off
            self.lane_change_direction = LaneChangeDirection.none

    self.prev_one_blinker = one_blinker and lateral_active

    self.desire = lane_turn_desire(carstate, nav)
    if self.desire == log.Desire.none and self.lane_change_state == LaneChangeState.laneChangeStarting:
      if self.lane_change_direction == LaneChangeDirection.left:
        self.desire = log.Desire.laneChangeLeft
      elif self.lane_change_direction == LaneChangeDirection.right:
        self.desire = log.Desire.laneChangeRight
    if self.desire == log.Desire.none:
      self.desire = NAV_KEEP.get(nav, log.Desire.none)
