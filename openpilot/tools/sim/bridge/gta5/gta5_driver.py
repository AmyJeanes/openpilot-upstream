"""The simulated driver: what a person in the seat does, as game commands. navd asks (NavOutputs.requests) and the
driver works the stalk: the game's indicator is the stalk, which the car's blinkers follow. It also nudges the wheel to
start a lane change openpilot waits on, cancels the indicator once the car has come round a turn, as a car's stalk
does, cancels cruise on arriving, and presses the gas when a light the car waits at turns green (PullAway).

It approves every request at once, as the bridge did before navd asked rather than acted."""
import os
import time

from openpilot.cereal import log
from openpilot.selfdrive.navd.inputs import CANCEL_SIGNAL, LANE_CHANGE_LEFT, LANE_CHANGE_RIGHT, SIGNAL_TURN_LEFT, \
  SIGNAL_TURN_RIGHT, NavOutputs

DEBUG = bool(os.getenv("GTA5_DEBUG"))
# openpilot starts a signaled lane change on a steering nudge towards it; give that nudge for the driver.
# Positive is left, and it must exceed the simulated Honda's steeringPressed threshold.
NUDGE_TORQUE = 2000
NUDGE_TIMEOUT = 3.0  # s after the indicator comes on
TURN_CANCEL_DEG = 60.0  # heading change since the indicator came on that counts as a turn taken
TURN_CANCEL_YAW_RATE = 0.1  # rad/s, straightened out
# the driver's gas press that gets the car moving again, which the model won't do itself once stopped
GO_AFTER = 1.0  # s stopped
GO_GREEN = 0.4  # s since traffic last showed red
GO_GAS = 0.5  # s
GO_EVERY = 3.0  # s
GO_TRIES = 3
GO_CLEAR = 15.0  # m: a vehicle nearer ahead leads the car away
INDICATOR = {SIGNAL_TURN_LEFT: "left", SIGNAL_TURN_RIGHT: "right", LANE_CHANGE_LEFT: "left", LANE_CHANGE_RIGHT: "right"}
LaneChangeState = log.LaneChangeState


class Driver:
  def __init__(self, send, cancel):
    self.send = send  # to the plugin
    self.cancel = cancel  # cruise cancel, as the driver's button
    self.indicator: str | None = None  # the game's indicator, as of the last step
    self.indicator_t = 0.0
    self.indicator_heading = 0.0
    self.lane_changing = False
    self.pull_away = PullAway(send)

  def act(self, out: NavOutputs):
    """navd's requests, approved: the stalk for each, in order; on arriving, cruise cancelled."""
    for r in out.requests:
      self.send({"type": "indicatorOff"} if r == CANCEL_SIGNAL else {"type": "setIndicator", "side": INDICATOR[r]})
    if out.arrived:
      self.cancel()

  def blinkers(self, gap: bool) -> tuple[bool, bool]:
    """The blinkers openpilot sees: the game's indicator, but off during navd's repeat gap (Planner.blinker_gap)."""
    return self.indicator == "left" and not gap, self.indicator == "right" and not gap

  def stalk(self, indicator: str | None, heading: float, yaw_rate: float, lane_change, nav_signaling: bool) -> float:
    """The plugin's indicator is the blinker stalk: nudge the wheel to start the lane change, and cancel the indicator
    once it is done, as a car's stalk would. Returns the driver's steering torque: the nudge, else 0. lane_change is
    openpilot's laneChangeState; nav_signaling whether navd's request on the stalk is a turn's (as of its last step)."""
    now = time.monotonic()
    if indicator != self.indicator:
      self.indicator, self.indicator_t, self.lane_changing = indicator, now, False
      self.indicator_heading = heading
    if indicator is None:
      return 0.0
    # a turn: cancel once the car has come round and straightened out
    turned = abs((heading - self.indicator_heading + 180) % 360 - 180)
    if turned > TURN_CANCEL_DEG and abs(yaw_rate) < TURN_CANCEL_YAW_RATE:
      self.send({"type": "indicatorOff"})
      self.indicator_heading = heading  # once, until the plugin's indicator goes off
      return 0.0
    if lane_change == LaneChangeState.laneChangeStarting:
      self.lane_changing = True
    elif self.lane_changing:
      self.lane_changing = False
      if not nav_signaling:  # a lane change still going as the blinker became the turn's
        self.send({"type": "indicatorOff"})
    elif (lane_change == LaneChangeState.preLaneChange and now - self.indicator_t < NUDGE_TIMEOUT
          and not nav_signaling):  # a turn on the route, not a lane change
      return NUDGE_TORQUE if indicator == "left" else -NUDGE_TORQUE
    return 0.0


class PullAway:
  """Presses the gas, as a driver would, when AI traffic waiting with the car at a red light shows it has turned green."""
  def __init__(self, send):
    self.send = send
    self.stopped_t: float | None = None
    self.red_t: float | None = None  # when traffic last showed a red light during this stop
    self.go_t = 0.0
    self.tries = 0

  def update(self, state: dict, engaged: bool):
    now = time.monotonic()
    if not engaged or state.get("vEgo", 0.0) > 0.3:
      self.stopped_t, self.red_t, self.tries = None, None, 0
      return
    if self.stopped_t is None:
      self.stopped_t = now
    traffic = state.get("traffic", {})
    if traffic.get("red", 0):
      self.red_t = now
      return
    user = state.get("user") or {}
    ahead = state.get("vehicleAhead", 0.0)
    blocked = traffic.get("crossing", 0) or traffic.get("peds", 0) or 0 < ahead < GO_CLEAR or user.get("gas") or user.get("brake")
    if (self.red_t is not None and now - self.red_t > GO_GREEN and not blocked and now - self.stopped_t > GO_AFTER and now - self.go_t > GO_EVERY
        and self.tries < GO_TRIES):
      self.go_t, self.tries = now, self.tries + 1
      if DEBUG:
        print(f"nav: green, pulling away ({self.tries})")
      self.send({"type": "gas", "secs": GO_GAS})
