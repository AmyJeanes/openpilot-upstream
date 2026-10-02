"""Navigation: drives the GPS route to the game map's waypoint, as a driver would with openpilot's lane turn desire. It
moves into the lane for each turn on the route, slows for it, signals it, and stops at the waypoint. PullAway pulls away
from a light once it turns green."""
import math
import os
import time

import numpy as np

from openpilot.common.constants import CV

DEBUG = bool(os.getenv("GTA5_DEBUG"))
TURN_ANGLE = 50.0  # deg of heading change along TURN_WINDOW m of route that makes a turn, not a bend
TURN_WINDOW = 30.0  # m
TURN_HOLDS = 20.0  # m past the turn where the route still heads the new way: a jog between lanes comes back
U_TURN_ANGLE = 135.0  # deg: the model can't make a U-turn, so drive on until GTA routes round instead
TURN_SPEED = 12 * CV.MPH_TO_MS  # the lane turn desire works below 19 mph
TURN_SLOW_BY = 15.0  # m before the turn
SLOW_DECEL = 0.8  # m/s^2, the approach the cruise cap allows
LOOKAHEAD = 150.0  # m, turns further on don't cap the speed yet
SIGNAL_DIST = 50.0  # m: signal from here once below the lane change speed, which would ask for a lane change instead
SIGNAL_ALWAYS_DIST = 30.0  # m: signal anyway
DONE_HEADING = 20.0  # deg from the turn's exit heading, straightened out
DONE_YAW_RATE = 0.1  # rad/s
MISSED_BY = 40.0  # m driven past a signaled turn that didn't happen: GTA reroutes
# The model takes a turn request as a pulse when the blinker comes on, and decides itself when the turn is done; after a
# stop, or once it no longer expects the turn, the blinker drops for openpilot (not the car's lights) to ask again.
REPEAT_GAP = 0.3  # s without the blinker
REPEAT_EVERY = 2.0  # s at most
REPEAT_BELOW = 0.1  # the model's probability of the turn
MIN_AHEAD = 20.0  # m: the route starts with a jog from the car's lane to GTA's road nodes, which isn't a turn
CONFIRM = 0.5  # s a turn must stay in the route before it counts, past the route's jitter at junctions
BEHIND_BY = 2.0  # m
COOLDOWN = 3.0  # s after a turn, while the rest of it may still look like a turn ahead
ARRIVE_DECEL = 0.7  # m/s^2
ARRIVE_LAG = 1.0  # s: openpilot eases into a lower set speed
ARRIVE_KEEP = 100.0  # m: GTA clears the waypoint as the car nears it, so stop on the straight distance left to it
STOP_BEFORE = 8.0  # m before the waypoint
ARRIVED_DIST = 15.0  # m: disengage once nearly stopped
ROUTE_POINTS = 101  # the plugin's route covers 500 m; fewer points means it ends there
LANE_CHANGE_DIST = 40.0  # m before a turn, the last a lane change for it starts
LANE_CHANGE_SPEED = 2.0  # m/s
LANE_CHANGE_TIMEOUT = 10.0  # s
LANE_CHANGE_GAP = 2.0  # s between lane changes
WRONG_SIDE_FOR = 1.0  # s in the oncoming lanes before moving back over
# The model has no desire for straight on, and sometimes turns where the route doesn't, as from a lane that becomes a
# turn lane; a keep desire away from that side (meant for forks) holds it on the road.
KEEP_ABOVE = 0.25  # the model's probability of a turn the route doesn't take
KEEP_CLEAR = 80.0  # m: no turn on the route nearer than this
KEEP_UNTIL = 0.05  # probability of the turn, once past where it was
KEEP_FOR = 30.0  # m driven at least
KEEP_GAP = 0.5  # s off to repeat it: openpilot reads the desire every 0.2 s
# the driver's gas press that gets the car moving again, which the model won't do itself once stopped
GO_AFTER = 1.0  # s stopped
GO_GREEN = 0.4  # s since traffic last showed red
GO_GAS = 0.5  # s
GO_EVERY = 3.0  # s
GO_TRIES = 3
GO_CLEAR = 15.0  # m: a vehicle nearer ahead leads the car away


def wrap(deg: float) -> float:
  return (deg + 180) % 360 - 180


class Turn:
  def __init__(self, dist: float, side: str, exit_heading: float):
    self.dist = dist  # m along the route to the turn
    self.side = side  # "left" or "right"
    self.exit_heading = exit_heading  # game heading (deg counterclockwise from north) after the turn


def find_turn(route: np.ndarray, after: float = 0.0) -> Turn | None:
  """The first turn on the route beyond `after` m: where its heading changes by TURN_ANGLE within TURN_WINDOW m."""
  seg = np.diff(route, axis=0)
  lengths = np.hypot(seg[:, 0], seg[:, 1])
  keep = lengths > 0.5
  seg, lengths = seg[keep], lengths[keep]
  if len(seg) < 2:
    return None
  starts = np.concatenate(([0.0], np.cumsum(lengths)[:-1]))
  # game headings: counterclockwise from north, from x east and y north
  headings = np.degrees(np.arctan2(seg[:, 1], seg[:, 0])) - 90
  for i in range(len(seg)):
    if starts[i] < after:
      continue
    j = int(np.searchsorted(starts, starts[i] + TURN_WINDOW, side='right')) - 1
    if j <= i:
      break
    change = wrap(headings[j] - headings[i])
    if abs(change) < TURN_ANGLE:
      continue
    # the turn is at its sharpest vertex in the window
    steps = [abs(wrap(headings[k + 1] - headings[k])) for k in range(i, j)]
    k = i + int(np.argmax(steps)) + 1
    held = int(np.searchsorted(starts, starts[k] + TURN_HOLDS, side='right')) - 1
    exit_change = abs(wrap(headings[max(held, j)] - headings[i]))
    if exit_change < TURN_ANGLE:
      continue
    if exit_change > U_TURN_ANGLE:
      return None
    return Turn(starts[k], "left" if change > 0 else "right", headings[max(held, j)])
  return None


class Nav:
  def __init__(self, send, set_desire):
    self.send = send  # to the plugin
    self.set_desire = set_desire  # openpilot's NavDesire
    self.turn: Turn | None = None  # the one being signaled
    self.cooldown_until = 0.0
    self.signaled: str | None = None
    self.shown = False  # whether the plugin's indicator has come on for our signal yet
    self.driven = 0.0  # m since signaling
    self.last_t = 0.0
    self.route_end: float | None = None  # m to the waypoint, once within the route's 500 m
    self.dest: np.ndarray | None = None  # where the waypoint was
    self.repeat_t = 0.0  # when the blinker last dropped to repeat the turn
    self.stopped = False
    self.seen: tuple[str, float] | None = None  # the turn ahead's side, and since when
    self.changing: str | None = None  # the side of nav's lane change under way
    self.change_shown = False
    self.change_t = 0.0  # when it started, or the last ended
    self.wrong_side_t: float | None = None  # since when the car has been in the oncoming lanes
    self.keeping: str | None = None  # the keep desire held against a turn the route doesn't take
    self.keep_from = 0.0  # self.driven when it started
    self.keep_gap_until = 0.0  # repeating it after a stop: off until then
    self.keep_stopped = False

  def update(self, state: dict, engaged: bool, indicator: str | None, desire: dict[str, float]) -> tuple[float, bool]:
    """Returns the cruise cap (m/s, 0 for none) and whether to disengage, having arrived. `desire` is the model's
    probability of each turn."""
    now = time.monotonic()
    v = state.get("vEgo", 0.0)
    step = v * min(now - self.last_t, 0.1)
    self.last_t = now
    self.driven += step
    route = state.get("route")
    if not engaged:
      self._cancel(indicator)
      self._end_change(indicator)
      self._end_keep()
      self.route_end, self.dest = None, None
      return 0.0, False
    self._watch_change(indicator, now)
    self._keep_right(state.get("lane"), v, now)
    if not route:
      self._cancel(indicator)
      self._end_keep()
      left = None if self.dest is None else float(np.hypot(*(self.dest - np.array(state["pos"][:2]))))
      if left is not None and left < ARRIVE_KEEP:
        # counted down by the distance driven, which doesn't grow again if the car runs past it
        self.route_end = left if self.route_end is None else min(self.route_end - step, left)
        return self._arrive(v)
      self.route_end, self.dest = None, None
      return 0.0, False
    route = np.array(route, dtype=float)
    waypoint = np.array(state.get("waypoint", (0.0, 0.0)), dtype=float)
    self.dest = waypoint if waypoint.any() else route[-1]  # (0, 0) as GTA clears it
    self.route_end = None
    if len(route) < ROUTE_POINTS:
      # along the route to its point nearest the waypoint: past that it sometimes runs on
      along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))))
      near = int(np.argmin(np.hypot(*(route - self.dest).T)))
      self.route_end = float(along[near] + np.hypot(*(route[near] - self.dest)))
    heading, yaw_rate = state["heading"], state["yawRate"]

    # the driver's own indicator takes over from ours, once ours has shown
    self.shown |= self.signaled is not None and indicator == self.signaled
    if self.signaled is not None and self.shown and indicator != self.signaled:
      self.turn, self.signaled = None, None
      self.cooldown_until = now + COOLDOWN

    if self.turn is not None:
      done = (self.driven > self.turn.dist - TURN_WINDOW / 2 and abs(wrap(heading - self.turn.exit_heading)) < DONE_HEADING
              and abs(yaw_rate) < DONE_YAW_RATE)
      if done or self.driven > self.turn.dist + MISSED_BY:
        if DEBUG:
          print(f"nav: turn {'done' if done else 'missed'} after {self.driven:.0f} m (car {heading:.0f})")
        self._cancel(indicator)
        self.cooldown_until = now + COOLDOWN

    turn = find_turn(route, MIN_AHEAD) if self.turn is None else None
    if turn is not None and self._behind(route, heading):
      turn = None  # GTA routes on once the car has passed somewhere to turn around
    if turn is None:
      self.seen = None
    elif self.seen is None or self.seen[0] != turn.side:
      self.seen = (turn.side, now)
    if turn is not None and now - self.seen[1] < CONFIRM:
      turn = None
    self._change_lane(turn, state.get("lane"), indicator, v, now)
    cap = 0.0
    if turn is not None and turn.dist < LOOKAHEAD:
      cap = math.sqrt(TURN_SPEED ** 2 + 2 * SLOW_DECEL * max(0.0, turn.dist - TURN_SLOW_BY))
      signal = turn.dist < SIGNAL_ALWAYS_DIST or (turn.dist < SIGNAL_DIST and v < 19 * CV.MPH_TO_MS and self.changing is None)
      if now >= self.cooldown_until and signal:
        self._end_change(indicator)
        self.turn, self.signaled, self.driven, self.shown = turn, turn.side, 0.0, False
        if DEBUG:
          print(f"nav: signal {turn.side} in {turn.dist:.0f} m, exit heading {turn.exit_heading % 360:.0f} (car {heading:.0f}, {v:.1f} m/s)")
        self.send({"type": "setIndicator", "side": turn.side})
    if self.turn is not None:
      cap = TURN_SPEED
      if v < 0.3:
        self.stopped = True
      elif v > 1.0 and self.shown and now - self.repeat_t > REPEAT_EVERY and (self.stopped or desire.get(self.turn.side, 1.0) < REPEAT_BELOW):
        self.repeat_t, self.stopped = now, False
    self._keep_straight(turn, desire, v, now)

    if self.route_end is not None:
      stop, arrived = self._arrive(v)
      return min(cap, stop) if cap else stop, arrived
    return max(cap, 0.5) if cap else 0.0, False

  def _keep_straight(self, turn: Turn | None, desire: dict[str, float], v: float, now: float):
    if self.turn is not None or self.changing is not None:
      self._end_keep()
      return
    if self.keeping is None:
      side = max(("left", "right"), key=lambda k: desire.get(k, 0.0))
      if desire.get(side, 0.0) > KEEP_ABOVE and (turn is None or turn.dist > KEEP_CLEAR):
        self.keeping, self.keep_from, self.keep_stopped = "keepRight" if side == "left" else "keepLeft", self.driven, False
        if DEBUG:
          print(f"nav: {self.keeping}, the model expecting a {side} turn ({desire[side]:.2f}) the route doesn't take")
        self.set_desire(self.keeping)
      return
    if self.driven - self.keep_from > KEEP_FOR and max(desire.get("left", 0.0), desire.get("right", 0.0)) < KEEP_UNTIL:
      self._end_keep()
      return
    # the model forgets it at a stop, as a turn: off briefly as the car pulls away, so it sees it again
    if v < 0.3:
      self.keep_stopped = True
    elif v > 1.0 and self.keep_stopped:
      self.keep_stopped, self.keep_gap_until = False, now + KEEP_GAP
      self.set_desire("")
    elif self.keep_gap_until and now > self.keep_gap_until:
      self.keep_gap_until = 0.0
      self.set_desire(self.keeping)

  def _end_keep(self):
    if self.keeping is not None and self.changing is None:  # a lane change's desire replaces it
      self.set_desire("")
    self.keeping, self.keep_gap_until = None, 0.0

  def _keep_right(self, lane: list[int] | None, v: float, now: float):
    """Back over from the oncoming lanes, which the model sometimes drifts into on wide roads."""
    if not lane or lane[0] >= 0:
      self.wrong_side_t = None
      return
    self.wrong_side_t = self.wrong_side_t or now
    if (self.changing is None and self.turn is None and now - self.wrong_side_t > WRONG_SIDE_FOR and v > LANE_CHANGE_SPEED
        and now - self.change_t > LANE_CHANGE_GAP):
      self._start_change("right", f"out of oncoming lane {-lane[0]}")

  def _start_change(self, side: str, why: str):
    self.changing, self.change_shown, self.change_t = side, False, time.monotonic()
    if DEBUG:
      print(f"nav: lane change {side}, {why}")
    self.set_desire("laneChange")
    self.send({"type": "setIndicator", "side": side})

  def _change_lane(self, turn: Turn | None, lane: list[int] | None, indicator: str | None, v: float, now: float):
    """Into the turn's lane: the leftmost for a left turn, the rightmost for a right."""
    if self.changing is not None or turn is None or self.turn is not None or not lane or not LANE_CHANGE_DIST < turn.dist < LOOKAHEAD:
      return
    i, n = lane
    want = 0 if turn.side == "left" else n - 1
    if i == want or v < LANE_CHANGE_SPEED or now - self.change_t < LANE_CHANGE_GAP:
      return
    self._start_change("left" if want < i else "right", f"from lane {i + 1} of {n} for the {turn.side} turn in {turn.dist:.0f} m")

  def _watch_change(self, indicator: str | None, now: float):
    if self.changing is None:
      return
    self.change_shown |= indicator == self.changing
    # the bridge cancels the indicator once the lane change is done, as does the driver to stop it
    if (self.change_shown and indicator != self.changing) or now - self.change_t > LANE_CHANGE_TIMEOUT:
      self._end_change(indicator)

  def _end_change(self, indicator: str | None):
    if self.changing is None:
      return
    if indicator == self.changing:
      self.send({"type": "indicatorOff"})
    self.set_desire("")
    self.changing, self.change_t = None, time.monotonic()

  def _arrive(self, v: float) -> tuple[float, bool]:
    stop = max(math.sqrt(2 * ARRIVE_DECEL * max(0.0, self.route_end - STOP_BEFORE - v * ARRIVE_LAG)), 0.5)
    # or stopped near it, short of the cap's aim (not at lights further back)
    arrived = v < 2.0 and self.route_end < ARRIVED_DIST or v < 0.3 and self.route_end < 2 * ARRIVED_DIST
    if arrived:
      self.send({"type": "waypoint", "off": True})
      self.route_end, self.dest = None, None
    return stop, arrived

  @staticmethod
  def _behind(route: np.ndarray, heading: float) -> bool:
    """Whether the route leads back the way the car came: its points 10 and 15 m on are behind the car."""
    h = math.radians(heading)
    forward = np.array([-math.sin(h), math.cos(h)])
    ahead = (route[2:4] - route[0]) @ forward
    return len(ahead) == 2 and bool(np.all(ahead < -BEHIND_BY))

  def _cancel(self, indicator: str | None):
    if self.signaled is not None and (indicator == self.signaled or not self.shown):
      self.send({"type": "indicatorOff"})
    self.turn, self.signaled = None, None

  @property
  def blinker_gap(self) -> bool:
    """Whether openpilot shouldn't see the blinker just now, to repeat the turn request."""
    return time.monotonic() - self.repeat_t < REPEAT_GAP

  @property
  def signaling(self) -> bool:
    return self.signaled is not None


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
