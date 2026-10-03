"""Navigation: drives the GPS route to the game map's waypoint, as a driver would with openpilot's lane turn desire. It
moves into the lane for each turn and fork on the route, slows for it, signals it, and stops at the waypoint. PullAway
pulls away from a light once it turns green."""
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
TURN_SPEED = 12 * CV.MPH_TO_MS  # for a square turn; the lane turn desire works below 19 mph
SOFT_TURN_SPEED = 16 * CV.MPH_TO_MS  # for a turn of TURN_ANGLE
TURN_SLOW_BY = 25.0  # m before the turn
SLOW_DECEL = 0.6  # m/s^2, the approach the cruise cap allows
SLOW_LAG = 1.0  # s: openpilot eases into a lower set speed
LOOKAHEAD = 250.0  # m, turns further on don't cap the speed yet
SIGNAL_DIST = 50.0  # m: signal from here once below the lane change speed, which would ask for a lane change instead
SIGNAL_ALWAYS_DIST = 30.0  # m: signal anyway, if in the turn's lane, else leave it unless changing into it
SIGNAL_LAST_DIST = 15.0  # m: the last chance, while a lane change into the turn's lane finishes
DONE_HEADING = 20.0  # deg from the turn's exit heading, straightened out
DONE_YAW_RATE = 0.1  # rad/s
MISSED_BY = 40.0  # m driven past a signaled turn that didn't happen: GTA reroutes
# The model takes a turn request as a pulse when the blinker comes on, and decides itself when the turn is done; it
# forgets the pulse after several seconds (the big model ~6.6 s), and at a stop. Until the turn starts, after a stop, or
# once it no longer expects the turn, the blinker drops for openpilot (not the car's lights) to ask again.
REPEAT_GAP = 0.3  # s without the blinker
REPEAT_EVERY = 2.0  # s at most
REPEAT_BELOW = 0.1  # the model's probability of the turn
PULSE_EVERY = 2.5  # s, until the turn starts, and for keep desires while held
TURN_STARTED = 15.0  # deg turned since signaling
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
# Lanes: a lane change takes the model several seconds, and the next needs a gap; plan them back from the last place
# one may start, and slow down when there isn't room left. A turn or fork that can't be reached from the car's lane is
# left for the route to come round again: a missed turn beats one across lanes, which the model won't take anyway.
LANE_CHANGE_DIST = 30.0  # m before a turn, the last a lane change for it starts
LANE_CHANGE_TIME = 8.0  # s each, with the gap before the next
LANE_CHANGE_EARLY = 4.0  # s more
LANE_CHANGE_SPEED = 2.0  # m/s
LANE_CHANGE_MIN_SPEED = 5.0  # m/s, slowing down for room to change lanes for a turn
FORK_MIN_SPEED = 10.0  # m/s, for a fork, as on a freeway
LANE_CHANGE_TIMEOUT = 10.0  # s
LANE_CHANGE_GAP = 2.0  # s between lane changes
LANE_CHANGE_TRIES = 3  # for one turn or fork, without getting nearer its lane
LANE_STEADY = 0.5  # s a lane reading must hold
LANE_STALE = 3.0  # s without a steady reading
WRONG_SIDE_FOR = 1.0  # s in the oncoming lanes before moving back over
SKIP_PAST = 60.0  # m: a turn or fork left to the route is forgotten this far behind
FORK_TURN = 80.0  # m: a turn this soon after a fork left to the route goes with it, as from a road's lanes split at a junction
# Forks: the route's branch and its lanes come from the map; the keep desire towards it (the model's fork desires)
# holds the model to that side through the split.
FORK_LOOKAHEAD = 1000.0  # m
FORK_LAST = 2.0  # s before a fork, the last a lane change for it starts
FORK_LAST_DIST = 40.0  # m, at least
FORK_KEEP = 4.0  # s before a fork that the keep desire starts
FORK_KEEP_DIST = 60.0  # m, at least
FORK_KEEP_PAST = 40.0  # m past it
# Bends and ramps: no faster than this sideways acceleration, measured over CURVE_WINDOW m of route
CURVE_ACCEL = 2.5  # m/s^2
CURVE_WINDOW = 30.0  # m
CURVE_LOOKAHEAD = 300.0  # m
LIMIT_DECEL = 0.8  # m/s^2, slowing for a lower speed limit ahead
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
  def __init__(self, dist: float, side: str, exit_heading: float, angle: float = 90.0):
    self.dist = dist  # m along the route to the turn
    self.side = side  # "left" or "right"
    self.exit_heading = exit_heading  # game heading (deg counterclockwise from north) after the turn
    self.angle = angle  # deg turned

  @property
  def speed(self) -> float:
    return float(np.interp(self.angle, [TURN_ANGLE, 90.0], [SOFT_TURN_SPEED, TURN_SPEED]))

  def lanes(self, n: int) -> tuple[int, int]:
    """The lanes (from the left) of n to take it from."""
    return (0, 0) if self.side == "left" else (n - 1, n - 1)


class Fork:
  def __init__(self, dist: float, side: str, lanes: int, lanes_in: int, keep: bool):
    self.dist = dist  # m along the route
    self.side = side  # the branch the route takes
    self.ours, self.lanes_in = lanes, lanes_in  # lanes on the route's branch, and on the road before
    self.keep = keep  # whether to hold the keep desire through it

  def lanes(self, n: int) -> tuple[int, int]:
    if self.lanes_in <= 1 or self.ours >= self.lanes_in:
      return 0, n - 1  # every lane goes the route's way
    ours = max(1, min(self.ours, n))
    return (0, ours - 1) if self.side == "left" else (n - ours, n - 1)


def headings(route: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Each segment's game heading (counterclockwise from north) and the distance to its start, skipping tiny ones."""
  seg = np.diff(route, axis=0)
  lengths = np.hypot(seg[:, 0], seg[:, 1])
  keep = lengths > 0.5
  seg, lengths = seg[keep], lengths[keep]
  starts = np.concatenate(([0.0], np.cumsum(lengths)[:-1]))
  return np.degrees(np.arctan2(seg[:, 1], seg[:, 0])) - 90, starts


def find_turn(route: np.ndarray, after: float = 0.0) -> Turn | None:
  """The first turn on the route beyond `after` m: where its heading changes by TURN_ANGLE within TURN_WINDOW m."""
  heads, starts = headings(route)
  if len(heads) < 2:
    return None
  ends = np.searchsorted(starts, starts + TURN_WINDOW, side='right') - 1
  idx = np.arange(len(heads))
  change = (heads[ends] - heads + 180) % 360 - 180
  candidates = np.nonzero((starts >= after) & (ends > idx) & (np.abs(change) >= TURN_ANGLE))[0]
  for i in candidates:
    j = int(ends[i])
    # the turn is at its sharpest vertex in the window
    steps = np.abs((np.diff(heads[i:j + 1]) + 180) % 360 - 180)
    k = i + int(np.argmax(steps)) + 1
    held = int(np.searchsorted(starts, starts[k] + TURN_HOLDS, side='right')) - 1
    exit_change = abs(wrap(heads[max(held, j)] - heads[i]))
    if exit_change < TURN_ANGLE:
      continue
    if exit_change > U_TURN_ANGLE:
      return None
    return Turn(float(starts[k]), "left" if change[i] > 0 else "right", float(heads[max(held, j)]), exit_change)
  return None


def curve_cap(route: np.ndarray, v: float) -> float:
  """The speed now that leaves room to slow for the bends ahead, 0 for none."""
  heads, starts = headings(route)
  if len(heads) < 3:
    return 0.0
  ends = np.searchsorted(starts, starts + CURVE_WINDOW, side='right') - 1
  span = starts[ends] - starts
  ok = (span > CURVE_WINDOW / 2) & (starts < CURVE_LOOKAHEAD)
  if not ok.any():
    return 0.0
  curvature = np.radians(np.abs((heads[ends] - heads + 180) % 360 - 180)) / np.maximum(span, 1.0)
  # the turns themselves have their own speed
  speed = np.maximum(np.sqrt(CURVE_ACCEL / np.maximum(curvature, 1e-4)), TURN_SPEED)
  # from the middle of the window, where the bend is
  dist = np.maximum(starts + span / 2 - v * SLOW_LAG, 0.0)
  return float(np.min(np.sqrt(speed ** 2 + 2 * SLOW_DECEL * dist)[ok]))


def limit_cap(limits: list, v: float) -> float:
  """The speed now that leaves room to slow for lower speed limits ahead ([[m ahead, m/s or 0 unknown], ...]), 0 for
  none; the set speed follows the limit where the car is."""
  caps = [math.sqrt(limit ** 2 + 2 * LIMIT_DECEL * max(0.0, d - v * SLOW_LAG)) for d, limit in limits if limit > 0 and d > 0]
  return min(caps) if caps else 0.0


def slow_for(speed: float, dist: float, v: float) -> float:
  return math.sqrt(speed ** 2 + 2 * SLOW_DECEL * max(0.0, dist - v * SLOW_LAG))


class Nav:
  def __init__(self, send, set_desire):
    self.send = send  # to the plugin
    self.set_desire = set_desire  # openpilot's NavDesire
    self.desire = ""  # what NavDesire is set to
    self.turn: Turn | None = None  # the one being signaled
    self.cooldown_until = 0.0
    self.signaled: str | None = None
    self.shown = False  # whether the plugin's indicator has come on for our signal yet
    self.driven = 0.0  # m since starting
    self.turn_from = 0.0  # self.driven when signaling
    self.last_t = 0.0
    self.route_end: float | None = None  # m to the waypoint, once within the route's 500 m
    self.dest: np.ndarray | None = None  # where the waypoint was
    self.repeat_t = 0.0  # when the blinker last dropped to repeat the turn
    self.signal_heading = 0.0  # the car's heading when signaling
    self.stopped = False
    self.seen: tuple[str, float] | None = None  # the turn ahead's side, and since when
    self.changing: str | None = None  # the side of nav's lane change under way
    self.change_shown = False
    self.change_t = 0.0  # when it started, or the last ended
    self.change_tries: dict[tuple, int] = {}  # lane changes for each turn or fork (by where it is), without progress
    self.change_from: tuple[int, int] | None = None  # the lane the last change started from
    self.lane: tuple[int, int] | None = None  # the car's lane [i from the left, of n], once it has held LANE_STEADY
    self.lane_seen: tuple[tuple[int, int] | None, float] = (None, 0.0)
    self.lane_t = 0.0  # when the lane reading last held
    self.skipped: list[np.ndarray] = []  # where turns and forks left to the route are
    self.skip_turns_to = 0.0  # self.driven to which turns are left to the route, after a fork that was
    self.v = 0.0
    self.wrong_side_t: float | None = None  # since when the car has been in the oncoming lanes
    self.keeping: str | None = None  # the keep desire held against a turn the route doesn't take
    self.keep_from = 0.0  # self.driven when it started
    self.keep_t = 0.0  # when it was last asked for again
    self.fork_keep: str | None = None  # the keep desire held for a fork on the route
    self.fork_keep_t = 0.0  # when it was last asked for again
    self.fork_keep_to = 0.0  # self.driven until which it's held, past the fork
    self.keep_gap_until = 0.0  # repeating a keep desire: off until then
    self.keep_stopped = False

  def update(self, state: dict, engaged: bool, indicator: str | None, desire: dict[str, float]) -> tuple[float, bool]:
    """Returns the cruise cap (m/s, 0 for none) and whether to disengage, having arrived. `desire` is the model's
    probability of each turn ("left", "right") and keep ("keepLeft", "keepRight")."""
    now = time.monotonic()
    v = self.v = state.get("vEgo", 0.0)
    step = v * min(now - self.last_t, 0.1)
    self.last_t = now
    self.driven += step
    route = state.get("route")
    pos = np.array(state.get("pos", (0.0, 0.0))[:2], dtype=float)
    self._read_lane(state.get("lane"), now)
    if not engaged:
      self._cancel(indicator)
      self._end_change(indicator)
      self.keeping, self.fork_keep = None, None
      self._set_desire(now)
      self.route_end, self.dest = None, None
      self.skipped.clear()
      return 0.0, False
    self._watch_change(indicator, now)
    self._keep_right(v, now)
    if not route:
      self._cancel(indicator)
      self.keeping, self.fork_keep = None, None
      self._set_desire(now)
      left = None if self.dest is None else float(np.hypot(*(self.dest - pos)))
      if left is not None and left < ARRIVE_KEEP:
        # counted down by the distance driven, which doesn't grow again if the car runs past it
        self.route_end = left if self.route_end is None else min(self.route_end - step, left)
        return self._arrive(v)
      self.route_end, self.dest = None, None
      return 0.0, False
    route = np.array(route, dtype=float)
    waypoint = np.array(state.get("waypoint") or (0.0, 0.0), dtype=float)
    self.dest = waypoint if waypoint.any() else route[-1]  # (0, 0) as GTA clears it
    self.route_end = None
    if state.get("routeEnd") is not None:
      if state["routeEnd"] < ARRIVE_KEEP * 5:
        self.route_end = float(state["routeEnd"])
    elif len(route) < ROUTE_POINTS:
      # along the route to its point nearest the waypoint: past that it sometimes runs on
      along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))))
      near = int(np.argmin(np.hypot(*(route - self.dest).T)))
      self.route_end = float(along[near] + np.hypot(*(route[near] - self.dest)))
    heading, yaw_rate = state["heading"], state["yawRate"]
    self.skipped = [p for p in self.skipped if np.hypot(*(p - pos)) < SKIP_PAST + 2 * FORK_LOOKAHEAD
                    and not self._passed(p, pos, heading)]

    # the driver's own indicator takes over from ours, once ours has shown
    self.shown |= self.signaled is not None and indicator == self.signaled
    if self.signaled is not None and self.shown and indicator != self.signaled:
      self.turn, self.signaled = None, None
      self.cooldown_until = now + COOLDOWN

    if self.turn is not None:
      done = (self.driven - self.turn_from > self.turn.dist - TURN_WINDOW / 2 and abs(wrap(heading - self.turn.exit_heading)) < DONE_HEADING
              and abs(yaw_rate) < DONE_YAW_RATE)
      if done or self.driven - self.turn_from > self.turn.dist + MISSED_BY:
        if DEBUG:
          print(f"nav: turn {'done' if done else 'missed'} after {self.driven - self.turn_from:.0f} m (car {heading:.0f})")
        self._cancel(indicator)
        self.cooldown_until = now + COOLDOWN

    turn = self._next_turn(route) if self.turn is None else None
    if turn is not None and self._behind(route, heading):
      turn = None  # GTA routes on once the car has passed somewhere to turn around
    if turn is None:
      self.seen = None
    elif self.seen is None or self.seen[0] != turn.side:
      self.seen = (turn.side, now)
    if turn is not None and now - self.seen[1] < CONFIRM:
      turn = None
    forks = self._forks(state.get("forks"), route, turn)
    caps = [self._change_lane(forks + ([turn] if turn is not None else []), route, v, now)]
    if turn is not None and turn.dist < LOOKAHEAD:
      caps.append(slow_for(turn.speed, turn.dist - TURN_SLOW_BY, v))
      self._signal(turn, route, indicator, v, now, heading)
    if self.turn is not None:
      caps.append(self.turn.speed)
      if v < 0.3:
        self.stopped = True
      elif v > 1.0 and self.shown and now - self.repeat_t > REPEAT_EVERY:
        started = abs(wrap(heading - self.signal_heading)) > TURN_STARTED
        if self.stopped or desire.get(self.turn.side, 1.0) < REPEAT_BELOW or (not started and now - self.repeat_t > PULSE_EVERY):
          self.repeat_t, self.stopped = now, False
    caps.append(curve_cap(route, v))
    caps.append(limit_cap(state.get("limits") or [], v))
    self._keep_fork(forks[0] if forks else None, route, desire, v, now)
    self._keep_straight(turn, desire, v, now)
    self._set_desire(now)

    caps = [c for c in caps if c > 0]
    cap = max(min(caps), 0.5) if caps else 0.0
    if self.route_end is not None:
      stop, arrived = self._arrive(v)
      return min(cap, stop) if cap else stop, arrived
    return cap, False

  def _next_turn(self, route: np.ndarray) -> Turn | None:
    """The first turn ahead not left to the route."""
    turn = find_turn(route, max(MIN_AHEAD, self.skip_turns_to - self.driven))
    while turn is not None and self._is_skipped(route, turn.dist):
      turn = find_turn(route, turn.dist + TURN_HOLDS)
    return turn

  def _forks(self, forks: list | None, route: np.ndarray, turn: Turn | None) -> list[Fork]:
    """The forks ahead, before any turn."""
    out = []
    for d, side, lanes, lanes_in, keep in forks or []:
      if d > FORK_LOOKAHEAD or (turn is not None and d > turn.dist):
        break
      if d > 0 and not self._is_skipped(route, d):
        out.append(Fork(d, side, lanes, lanes_in, keep))
    return out

  @staticmethod
  def _point(route: np.ndarray, dist: float) -> np.ndarray:
    along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))))
    return np.array([np.interp(dist, along, route[:, 0]), np.interp(dist, along, route[:, 1])])

  def _is_skipped(self, route: np.ndarray, dist: float) -> bool:
    if not self.skipped:
      return False
    p = self._point(route, max(dist, 0.0))
    return any(np.hypot(*(p - s)) < TURN_HOLDS for s in self.skipped)

  def _skip(self, route: np.ndarray, dist: float, what: str):
    if DEBUG:
      print(f"nav: not in lane {self.lane} for the {what} in {dist:.0f} m; leaving it to the route")
    self.skipped.append(self._point(route, max(dist, 0.0)))

  @staticmethod
  def _passed(p: np.ndarray, pos: np.ndarray, heading: float) -> bool:
    h = math.radians(heading)
    return bool((p - pos) @ np.array([-math.sin(h), math.cos(h)]) < -SKIP_PAST)

  def _read_lane(self, lane: list[int] | None, now: float):
    """The lane reading, once it has held for a moment: it's noisy where roads join. None once it hasn't for a while."""
    reading = tuple(lane) if lane else None
    if reading != self.lane_seen[0]:
      self.lane_seen = (reading, now)
    elif now - self.lane_seen[1] >= LANE_STEADY:
      self.lane, self.lane_t = reading, now
    if now - self.lane_t > LANE_STALE:
      self.lane = None

  def _signal(self, turn: Turn, route: np.ndarray, indicator: str | None, v: float, now: float, heading: float):
    """Signals the turn once near and slow enough, from its lane; one that can't be taken from the car's lane is left."""
    if now < self.cooldown_until:
      return
    slow = v < 19 * CV.MPH_TO_MS and self.changing is None
    if not (turn.dist < SIGNAL_ALWAYS_DIST or (turn.dist < SIGNAL_DIST and slow)):
      return
    if self.lane is not None and self.lane[0] >= 0:
      lo, hi = turn.lanes(self.lane[1])
      if not lo <= self.lane[0] <= hi:
        if turn.dist > LANE_CHANGE_DIST or (self.changing is not None and turn.dist > SIGNAL_LAST_DIST):
          return  # a lane change towards it may still come or finish
        self._end_change(indicator)
        self._skip(route, turn.dist, f"{turn.side} turn")
        return
    self._end_change(indicator)
    self.turn, self.signaled, self.turn_from, self.shown = turn, turn.side, self.driven, False
    self.signal_heading, self.repeat_t = heading, now
    if DEBUG:
      print(f"nav: signal {turn.side} in {turn.dist:.0f} m, exit heading {turn.exit_heading % 360:.0f} (car {heading:.0f}, {v:.1f} m/s, lane {self.lane})")
    self.send({"type": "setIndicator", "side": turn.side})

  def _change_lane(self, ahead: list, route: np.ndarray, v: float, now: float) -> float:
    """Into the lanes for the turns and forks ahead (nearest first), early enough for the changes the first the car isn't
    in the lanes for needs, without leaving the lanes for those before it; returns a cruise cap that leaves room for
    them (0 for none)."""
    if self.turn is not None or self.lane is None or self.lane[0] < 0:
      return 0.0
    i, n = self.lane
    allowed = (0, n - 1)
    for m in ahead:
      lo, hi = m.lanes(n)
      if lo <= i <= hi:
        allowed = (max(allowed[0], lo), min(allowed[1], hi))
        continue
      if (i > hi and i <= allowed[0]) or (i < lo and i >= allowed[1]):
        return 0.0  # not until past the one before, whose lanes the car is in
      return self._change_for(m, lo, hi, route, v, now)
    return 0.0

  def _change_for(self, m: Turn | Fork, lo: int, hi: int, route: np.ndarray, v: float, now: float) -> float:
    i, n = self.lane
    changes = lo - i if i < lo else i - hi
    fork = isinstance(m, Fork)
    last = max(FORK_LAST_DIST, FORK_LAST * v) if fork else LANE_CHANGE_DIST
    where = self._point(route, max(m.dist, 0.0))
    key = (fork, round(float(where[0])), round(float(where[1])))
    if self.change_tries.get(key, 0) >= LANE_CHANGE_TRIES:
      return 0.0  # the model won't: there's no lane there
    room = m.dist - last
    need = changes * LANE_CHANGE_TIME
    if fork and room < 0 and self.changing is None:
      self._skip(route, m.dist, f"{m.side} fork")
      self.skip_turns_to = self.driven + m.dist + FORK_TURN
      return 0.0
    cap = max(FORK_MIN_SPEED if fork else LANE_CHANGE_MIN_SPEED, room / need) if room < need * v else 0.0
    if (self.changing is None and room > 0 and room < (need + LANE_CHANGE_EARLY) * max(v, LANE_CHANGE_MIN_SPEED)
        and v > LANE_CHANGE_SPEED and now - self.change_t > LANE_CHANGE_GAP and now >= self.cooldown_until):
      if self.change_from == self.lane:
        self.change_tries[key] = self.change_tries.get(key, 0) + 1  # the last change didn't get anywhere
      self.change_from = self.lane
      what = f"{m.side} {'fork' if fork else 'turn'}"
      self._start_change("left" if i > hi else "right", f"from lane {i + 1} of {n} for the {what} in {m.dist:.0f} m ({changes} to go)")
    return cap

  def _keep_fork(self, fork: Fork | None, route: np.ndarray, desire: dict[str, float], v: float, now: float):
    """The keep desire towards the route's branch, from a little before the fork to past it; repeated if the model
    stops expecting it."""
    want = self.fork_keep if self.driven < self.fork_keep_to and self.turn is None else None
    if fork is not None and fork.keep and self.turn is None and fork.dist < max(FORK_KEEP_DIST, FORK_KEEP * v):
      lo, hi = fork.lanes(self.lane[1]) if self.lane is not None else (0, 0)
      if self.lane is None or lo <= self.lane[0] <= hi:
        want = "keepLeft" if fork.side == "left" else "keepRight"
        self.fork_keep_to = self.driven + fork.dist + FORK_KEEP_PAST
    if want != self.fork_keep:
      if DEBUG and want:
        print(f"nav: {want} for the fork in {self.fork_keep_to - self.driven - FORK_KEEP_PAST:.0f} m")
      self.fork_keep, self.fork_keep_t = want, now
    elif want and v > 1.0 and now - self.fork_keep_t > PULSE_EVERY and self.driven < self.fork_keep_to - FORK_KEEP_PAST:
      self.fork_keep_t, self.keep_gap_until = now, now + KEEP_GAP

  def _keep_straight(self, turn: Turn | None, desire: dict[str, float], v: float, now: float):
    if self.turn is not None or self.changing is not None:
      self.keeping = None
      return
    if self.keeping is None:
      side = max(("left", "right"), key=lambda k: desire.get(k, 0.0))
      if desire.get(side, 0.0) > KEEP_ABOVE and (turn is None or turn.dist > KEEP_CLEAR):
        self.keeping, self.keep_from, self.keep_t = "keepRight" if side == "left" else "keepLeft", self.driven, now
        if DEBUG:
          print(f"nav: {self.keeping}, the model expecting a {side} turn ({desire[side]:.2f}) the route doesn't take")
      return
    if self.driven - self.keep_from > KEEP_FOR and max(desire.get("left", 0.0), desire.get("right", 0.0)) < KEEP_UNTIL:
      self.keeping = None
    elif self.fork_keep is None and v > 1.0 and now - self.keep_t > PULSE_EVERY:
      self.keep_t, self.keep_gap_until = now, now + KEEP_GAP

  def _set_desire(self, now: float):
    """NavDesire: a lane change under way, else a keep desire; the model forgets a keep desire at a stop, as a turn, so
    it goes off briefly as the car pulls away to be seen again."""
    want = "laneChange" if self.changing is not None else self.fork_keep or self.keeping or ""
    if want.startswith("keep"):
      if self.v < 0.3:
        self.keep_stopped = True
      elif self.v > 1.0 and self.keep_stopped:
        self.keep_stopped, self.keep_gap_until = False, now + KEEP_GAP
    if now < self.keep_gap_until and want.startswith("keep"):
      want = ""
    if want != self.desire:
      self.desire = want
      self.set_desire(want)

  def _keep_right(self, v: float, now: float):
    """Back over from the oncoming lanes, which the model sometimes drifts into on wide roads."""
    lane = self.lane
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
    self._set_desire(self.change_t)
    self.send({"type": "setIndicator", "side": side})

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
    self.changing, self.change_t = None, time.monotonic()
    self._set_desire(self.change_t)

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
