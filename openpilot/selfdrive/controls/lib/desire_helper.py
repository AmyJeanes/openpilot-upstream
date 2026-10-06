import math
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
# after a "+", a keep desire given alongside whatever else the model gets ("+keepRight" with a turn, "laneChange+keepRight");
# the model was trained on one desire at a time, so two at once is out of its training distribution
NAV_STACK = "+"
NAV_READ_EVERY = 0.2  # s
# The model takes a desire as a pulse on its rising edge, and forgets a turn after several seconds or at a stop. With
# TurnDesireRefresh, a turn is asked for again until the car has started turning: on moving off after a stop, and every
# few seconds, sooner once the model no longer expects it. Asked many times within one turn, the model takes it badly.
TURN_REFRESH_EVERY = 2.5  # s
TURN_REFRESH_MIN = 2.0  # s between requests
TURN_REFRESH_BELOW = 0.1  # the model's probability of the turn
TURN_STARTED = math.radians(15.)  # turned since asked for: no more, as a pulse late in a turn swings the car round past it
TURN_STOP_STARTED = math.radians(30.)  # after a stop, asked again unless turned this far already
TURN_STOPPED_SPEED = 0.3  # m/s
TURN_MOVING_SPEED = 1.0  # m/s


class NavDesire:
  def __init__(self):
    self.params = Params()
    self.value = ""
    self.t = 0.0

  def get(self, fresh: bool = False) -> str:
    now = time.monotonic()
    if fresh or now - self.t > NAV_READ_EVERY:
      self.value, self.t = self.params.get("NavDesire") or "", now
    return self.value


def nav_split(nav: str) -> tuple[str, int]:
  """NavDesire's own request, and the keep desire stacked on it if any."""
  base, _, stacked = nav.partition(NAV_STACK)
  return base, NAV_KEEP.get(stacked, log.Desire.none)


def lane_turn_desire(CS, nav: str = "") -> int:
  """The turn a blinker asks for at low speed, unless the blind spot that way is occupied; overrides any lane change."""
  if CS.vEgo < LANE_TURN_SPEED and nav_split(nav)[0] != NAV_LANE_CHANGE:
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
    self.stacked = log.Desire.none  # a second desire for the model, only when a navigation source stacks one
    self.nav = NavDesire()
    self.turn_refresh = self.nav.params.get_bool("TurnDesireRefresh")
    self.refresh = False  # for modeld to give the model the turn desire's pulse again
    self.refresh_count = 0  # in the turn desire under way
    self.turn_timer = 0.0  # since the turn was last asked for
    self.turn_angle = 0.0  # rad turned since it began
    self.turn_stopped = False

  @staticmethod
  def get_lane_change_direction(CS):
    return LaneChangeDirection.left if CS.leftBlinker else LaneChangeDirection.right

  def update_turn_refresh(self, v_ego, yaw_rate, turn_prob, new_turn):
    if new_turn:
      self.turn_timer, self.turn_angle, self.turn_stopped, self.refresh_count = 0.0, 0.0, False, 0
      return False
    self.turn_timer += DT_MDL
    self.turn_angle += yaw_rate * DT_MDL
    if v_ego < TURN_STOPPED_SPEED:
      self.turn_stopped = True
      return False
    if v_ego < TURN_MOVING_SPEED or self.turn_timer < TURN_REFRESH_MIN:
      return False
    turned = abs(self.turn_angle)
    fading = turn_prob < TURN_REFRESH_BELOW or self.turn_timer > TURN_REFRESH_EVERY
    refresh = (turned < TURN_STARTED and fading) or (self.turn_stopped and turned < TURN_STOP_STARTED)
    self.turn_stopped = False
    if refresh:
      self.turn_timer = 0.0
      self.refresh_count += 1
    return refresh

  def update(self, carstate, lateral_active, lane_change_prob, yaw_rate=0.0, desire_state=None):
    v_ego = carstate.vEgo
    one_blinker = carstate.leftBlinker != carstate.rightBlinker
    # read afresh as a blinker comes on, which means a turn or a lane change by it
    nav, stacked = nav_split(self.nav.get(one_blinker and not self.prev_one_blinker))
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

    prev_desire = self.desire
    self.desire = lane_turn_desire(carstate, nav)
    self.refresh = False
    if self.turn_refresh and self.desire in (log.Desire.turnLeft, log.Desire.turnRight):
      turn_prob = desire_state[self.desire] if desire_state is not None and len(desire_state) > self.desire else 1.0
      self.refresh = self.update_turn_refresh(v_ego, yaw_rate, turn_prob, self.desire != prev_desire)
    if self.desire == log.Desire.none and self.lane_change_state == LaneChangeState.laneChangeStarting:
      if self.lane_change_direction == LaneChangeDirection.left:
        self.desire = log.Desire.laneChangeLeft
      elif self.lane_change_direction == LaneChangeDirection.right:
        self.desire = log.Desire.laneChangeRight
    if self.desire == log.Desire.none:
      self.desire = NAV_KEEP.get(nav, stacked)
    self.stacked = stacked if stacked != self.desire else log.Desire.none
