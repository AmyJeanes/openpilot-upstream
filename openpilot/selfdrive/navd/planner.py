"""navd's planner: drives the route to the destination, as a driver would with openpilot's lane turn desire. It moves
into the lane for each turn and fork on the route, slows for it, has it signalled, and stops at the destination. Each
step (NavInputs) gives a cruise cap, NavDesire's changes and requests to the driver (NavOutputs): signal a turn, change
lanes, cancel the signal; the driver works the stalk.

Game-agnostic: nothing here reads a game. Until navd localizes itself and reads the lane from perception, the car's
pose and lane in its inputs are the simulator's truth (inputs.py)."""
import json
import math
import os

import numpy as np

from openpilot.common.constants import CV
from openpilot.selfdrive.navd.inputs import NavInputs, NavOutputs
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import turn_targets  # the map stack moves into navd with the OSM-only Route

# the GTA5_ names are the bridge's, while navd runs inside it
DEBUG = bool(os.getenv("NAVD_DEBUG") or os.getenv("GTA5_DEBUG"))
TURN_ANGLE = 50.0  # deg of heading change along TURN_WINDOW m of route that makes a turn, not a bend
TURN_WINDOW = 30.0  # m
TURN_HOLDS = 20.0  # m past the turn where the route still heads the new way: a jog between lanes comes back
JOG_LINK = 25.0  # m: a turn off a link this short across a junction (a jog between offset roads) is one from the road before it
U_TURN_ANGLE = 135.0  # deg: the model can't make a U-turn, so drive on until GTA routes round instead
TURN_SPEED = 4.5  # m/s for a square turn: slower turns tighter, and the lane turn desire works below 19 mph
SHARP_TURN_SPEED = 4.0  # m/s beyond sharp_angle
SOFT_TURN_SPEED = 6.0  # m/s for a turn of TURN_ANGLE
TURN_SLOW_BY = 25.0  # m before the turn
SLOW_DECEL = 0.6  # m/s^2, the approach the cruise cap allows
SLOW_LAG = 1.0  # s: openpilot eases into a lower set speed
LOOKAHEAD = 250.0  # m, turns further on don't cap the speed yet
NO_CAP = (0.0, "")
# Signal SIGNAL_TIME before a turn (no further out than SIGNAL_DIST) once below the lane change speed, which would ask
# for a lane change instead: further out, the model stops short with a turn asked for and no turning to take.
SIGNAL_TIME = 5.0  # s
SIGNAL_DIST = 50.0  # m
SIGNAL_MIN = 28.0  # m: signal anyway, if in the turn's lane, else leave it unless changing into it
SIGNAL_LAST = 8.0  # m: signalling at the junction's entry, signal by here whatever the speed
# How far the model turns depends on its speed in the turn: the turn's speed holds through the arc, until the car heads
# its way out and is straight, and lifts gently after
HOLD_RELEASE = 10.0  # deg from the way out
HOLD_RELEASE_M = 40.0  # m past the turn's point, at the latest
HOLD_RELEASE_ACCEL = 1.0  # m/s^2
HOLD_ARC = 10.0  # deg turned: in the arc, for the speeds logged
HOLD_DONE = 40.0  # m/s: lifted past anything
SIGNAL_LAST_DIST = 26.0  # m: the last chance, while a lane change into the turn's lane finishes (the turn finder looks from MIN_AHEAD on)
DONE_HEADING = 20.0  # deg from the turn's exit heading, straightened out
DONE_YAW_RATE = 0.1  # rad/s
MISSED_BY = 40.0  # m driven past a signaled turn that didn't happen: GTA reroutes
# The model takes a turn request as a pulse when the blinker comes on, and decides itself when the turn is done; it
# forgets the pulse after several seconds (the big model ~6.6 s), and at a stop. Until the turn starts, after a stop, or
# once it no longer expects the turn, the blinker drops for openpilot (not the car's lights) to ask again, unless
# openpilot asks again itself (TurnDesireRefresh).
REPEAT_GAP = 0.3  # s without the blinker
REPEAT_EVERY = 2.0  # s at most
REPEAT_BELOW = 0.1  # the model's probability of the turn
PULSE_EVERY = 2.5  # s, until the turn starts, and for keep desires while held
TURN_STARTED = 15.0  # deg turned since signaling: no more pulses, as one late in the turn swings the car round past it
STOP_REPEAT_TURNED = 30.0  # deg: after a stop, asked again unless turned this far already
EXIT_DONE = 25.0  # deg from the turn's way out: done, the blinker off, whatever the yaw rate
# Coming round past a left turn's way out, the model is held back by keepRight for a moment (not keepLeft after a right
# turn, which takes the car into the oncoming lanes)
EXIT_CUE = os.getenv("NAVD_EXIT_CUE", os.getenv("GTA5_EXIT_CUE", "1")) != "0"
EXIT_CUE_FOR = 2.0  # s at most
EXIT_CUE_YAW = 0.15  # rad/s still turning
EXIT_WATCH_QUEUE = 6.6  # s a pulse stays in the big model's desire queue (exit_watch)
EXIT_WATCH_PAST = 10.0  # deg past the turn's way out
ONCOMING_JUNCTION = 15.0  # m from a junction or stop-line node, inside which the lane reading follows GTA's diagonal links
MIN_AHEAD = 20.0  # m: GTA's route starts with a jog from the car's lane to its road nodes, which isn't a turn
MIN_AHEAD_MAP = 5.0  # m: our map's routes start at the car
TURN_PAST = 8.0  # m past a turn, heading within TURN_PAST_HEADING of its way out: done, as for a turn straight after
TURN_PAST_HEADING = 45.0  # deg
CONFIRM = 0.5  # s a turn must stay in the route before it counts, past the route's jitter at junctions
BEHIND_BY = 2.0  # m
COOLDOWN = 3.0  # s after a turn without lane changes, while the car straightens out
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
FAST, FAST_EARLY = 18.0, 8.0  # m/s, and s more at that speed, as on a freeway, where the exits come up fast
LANE_CHANGE_SPEED = 2.0  # m/s
LANE_CHANGE_MIN_SPEED = 5.0  # m/s, slowing down for room to change lanes for a turn
FORK_MIN_SPEED = 10.0  # m/s, for a fork, as on a freeway
LANE_CHANGE_TIMEOUT = 10.0  # s
LANE_CHANGE_GAP = 2.0  # s between lane changes
LANE_CHANGE_TRIES = 3  # for one turn or fork, without getting nearer its lane
LANE_LINE_CHANGE = 40.0  # m a lane change takes on the map's lane line
LANE_STEADY = 0.5  # s a lane reading must hold
LANE_STALE = 3.0  # s without a steady reading
SKIP_SURE = 0.6  # lanes from the turn's lanes, by the car's position on the route, to leave the turn to the route
WRONG_SIDE_FOR = 1.0  # s in the oncoming lanes before moving back over
TURNING = 0.1  # rad/s: no lane change while turning
# The blinker means a lane change below 19 mph only once openpilot has read NavDesire (every 0.2 s), and a turn
# otherwise: the param is set this long before the blinker comes on, and kept this long after it goes off.
PARAM_LEAD = 0.4  # s
PARAM_HOLD = 0.5  # s
SKIP_PAST = 60.0  # m: a turn or fork left to the route is forgotten this far behind
SKIP_NEAR = 20.0  # m: a turn or fork found this near one left to the route is the same
TAKEN_NEAR = 12.0  # m, or near a turn taken, as a turn may follow straight after
FORK_TURN = 80.0  # m: a turn this soon after a fork left to the route goes with it, as from a road's lanes split at a junction
# Forks: the route's branch and its lanes come from the map; the keep desire towards it (the model's fork desires)
# holds the model to that side through the split.
FORK_LOOKAHEAD = 1000.0  # m
FORK_LAST = 2.0  # s before a fork, the last a lane change for it starts
FORK_LAST_DIST = 40.0  # m, at least
FORK_KEEP = 4.0  # s before a fork that the keep desire starts
FORK_KEEP_DIST = 60.0  # m, at least
FORK_KEEP_PAST = 40.0  # m past it
# A slip lane or turn bay (GTA's centre turn bays are the road's median) opening this near before a turn: the turn is
# signalled, and slowed for, from where it opens, so that the model takes it into the bay
BAY_BEFORE = 70.0  # m
BAY_SIGNAL = 5.0  # m before it opens
# and the car changes lanes into it as it opens, from the lane beside it, then signals the turn (by BAY_LAST at the
# latest): from the lane beside a bay the model carries straight on
BAY_CHANGE = os.getenv("NAVD_BAY", os.getenv("GTA5_BAY", "1")) != "0"
BAY_OPEN = 3.0  # m before it opens
BAY_LAST = 18.0  # m before the turn
BAY_SPEED = 4.5  # m/s, changing into it
# Bends and ramps: no faster than this sideways acceleration, measured over CURVE_WINDOW m of route. The Tesla's
# steering is limited to 3.6 m/s^2 (3 m/s^2 and road roll) and in our drives saturated at about 3.
CURVE_ACCEL = 2.0  # m/s^2
CURVE_WINDOW = 20.0  # m
CURVE_STEP = 2.5  # m
CURVE_LOOKAHEAD = 300.0  # m
LIMIT_DECEL = 0.8  # m/s^2, slowing for a lower speed limit ahead
# The model has no desire for straight on, and sometimes turns where the route doesn't, as from a lane that becomes a
# turn lane; a keep desire away from that side (meant for forks) holds it on the road.
KEEP_ABOVE = 0.25  # the model's probability of a turn the route doesn't take
KEEP_CLEAR = 80.0  # m: no turn on the route nearer than this
KEEP_UNTIL = 0.05  # probability of the turn, once past where it was
KEEP_FOR = 30.0  # m driven at least
KEEP_GAP = 0.5  # s off to repeat it: openpilot reads the desire every 0.2 s
KEEP_AFTER_TURN = 60.0  # m: the model's expectation of the turn just taken fades only after it
KEEP_MIN_SPEED = 3.0  # m/s: pulling away, its turn probabilities are noise
# A turn's junction entry: GTA's stop line (11-22 m before the turn's node in the junction), else its junction nodes
ENTRY_STOP_BEFORE = 50.0  # m before the turn
ENTRY_JUNCTION_BEFORE = 40.0  # m
ENTRY_GAP = 15.0  # m between a junction's nodes
ENTRY_PAST = 3.0  # m past the turn's point, a junction node of it
ENTRY_MIN = 5.0  # m before the turn: nearer, the junction nodes say nothing more than the turn
ENTRY_DEFAULT = 15.0  # m before the turn
LEFT_SIGNAL_AFTER_CHANGE = 8.0  # s after a lane change towards a left turn its signal isn't held for the stop line
# The map's turn arrows (turn:lanes) say which lanes a turn or fork is taken from: those of the lanes into its junction,
# ending this near the turn's point (in the junction)
ARROWS_BEFORE, ARROWS_AFTER = 30.0, 5.0  # m
THROUGH_TURN = 20.0  # deg at most the route turns through a junction it goes straight on through, where the arrows don't say


def wrap(deg: float) -> float:
  return (deg + 180) % 360 - 180


class Tune:
  """Nav's turn parameters, from a JSON file (the bridge's GTA5_NAVTUNE) read again whenever it changes, so a test harness
  can sweep them without restarting; the defaults are those above, and without a file. Keys left out keep their
  defaults."""
  DEFAULTS = {
    "turn_speed_soft": SOFT_TURN_SPEED,  # m/s for a turn of TURN_ANGLE, between it and a square turn by angle
    "turn_speed_square": TURN_SPEED,  # m/s for a square turn
    "turn_speed_square_right": 0.0,  # m/s for a square right turn (0: turn_speed_square)
    "turn_speed_sharp": SHARP_TURN_SPEED,  # m/s beyond sharp_angle
    "sharp_angle": 110.0,  # deg
    "slow_decel": SLOW_DECEL,  # m/s^2
    "slow_from": LOOKAHEAD,  # m before a turn the slowing for it may start
    "slow_done": TURN_SLOW_BY,  # m before slow_ref the car is down to the turn's speed
    "slow_ref": "turn",  # "turn" (its node, in the junction) or "entry" (the junction's entry: the stop line)
    "signal_mode": "time",  # "time": SIGNAL_TIME ahead; "entry": at signal_entry_offset from the junction's entry
    "signal_time": SIGNAL_TIME,  # s
    "signal_min": SIGNAL_MIN,  # m
    "signal_max": SIGNAL_DIST,  # m
    "signal_entry_offset": 0.0,  # m past the junction's entry (negative before it)
    # m before the junction's entry the time mode signals by whatever the speed (0: off); the route's turn point is
    # the junction's centre, and too close to a right turn's corner the model goes straight on
    "signal_min_entry": 0.0,
    # m before the junction's entry a left turn's time-mode signal waits for (0: off); signalled well before its stop
    # line, the model turns in early and can cut into the near, oncoming half of a split junction
    "left_signal_max_entry": 0.5,
    # m past a turn's point it is given up if the car hasn't started turning (0: MISSED_BY); a pulse still in the
    # model's memory after a miss turns it at the next opening, a wall as often as a road
    "unturned_cancel": 25.0,
    "unturned_keep_pulses": True,  # False: no pulses past the turn's point, which starves late turn-ins
    "repulse_every": PULSE_EVERY,  # s
    "repulse_until_turned": TURN_STARTED,  # deg
    "repulse_after_stop_until": STOP_REPEAT_TURNED,  # deg
    "repulse_below_prob": REPEAT_BELOW,
    "lane_change_time": LANE_CHANGE_TIME,  # s each
    "lane_change_early": LANE_CHANGE_EARLY,  # s
    "lane_change_fast_early": FAST_EARLY,  # s more above FAST m/s
    "lane_change_last": LANE_CHANGE_DIST,  # m before a turn
    # m before a turn a lane change towards it still under way becomes the turn's signal (0: it's ended, though
    # openpilot's lane change carries on into the junction until the model is done with it)
    "lane_change_into_turn": 0.0,
    "lane_change_min_speed": LANE_CHANGE_MIN_SPEED,  # m/s, slowing down for a lane change for a turn to finish before it
    "bay_speed": BAY_SPEED,  # m/s while changing into a turn's bay (0: no cap)
    "curve_accel": CURVE_ACCEL,  # m/s^2
    # held from signalling a turn through its arc, until the car heads within turn_release deg of its way out and is
    # straight, or is turn_release_m past it; then lifted at release_accel
    "turn_hold_speed": 0.0,  # m/s, 0: the turn's approach speed
    "turn_release": HOLD_RELEASE,  # deg
    "turn_release_m": HOLD_RELEASE_M,  # m
    "release_accel": HOLD_RELEASE_ACCEL,  # m/s^2
    "exit_cue_right": True,  # keepRight out of a right turn too, which can end across the new road's centre line
    "exit_cue_left": True,
    # share of a turn's angle come round by which its keepRight starts, stacked on the turn while its blinker is on (1:
    # only once out of it; 0: with the turn's signal)
    "exit_cue_at": 1.0,
    # keepRight while the plugin reads the car in the oncoming lanes outside a junction, where the route's reading
    # disagrees and nav has no lane to change back from
    "oncoming_keep": True,
    # and while the map's lanes (laneMap) put it in an oncoming lane, on a one-way the wrong way or in the other
    # direction's turn bay, which the plugin and route readings miss
    "oncoming_map": True,
    # going straight on through a junction, out of lanes that end there (laneDrops)
    "lane_drops": True,
    # a turn the car is one lane beside the lanes for, at the last place a lane change for it starts, is signalled from
    # there rather than left to the route: on the traffic-free bench the model turns across the lane, and turn bays
    # that open late (2 lanes becoming 3) leave the car one lane off after its planned change
    "turn_from_beside": False,
    # after a turn is over, while its last pulse can still be in the model's queue (EXIT_WATCH_QUEUE), the counter keep
    # desire (keepRight after a left turn, keepLeft after a right one, which two-way roads drop) whenever the car
    # is past the turn's way out and still yawing that way: a stale pulse keeps a slow turn going round
    "exit_watch": False,
    # the heading turned since signalling, accumulated (not wrapped, which reads ~0 again after a full circle), for the
    # re-pulse and unturned_cancel checks
    "turned_unwrapped": False,
    # no re-pulse of a turn once the car has come round past its way out (as after a stop mid-turn)
    "no_repulse_past_exit": False,
  }
  CHECK_EVERY = 1.0  # s

  def __init__(self, path: str | None = None):
    self.path = path
    self.values = dict(self.DEFAULTS)
    self.mtime: float | None = None
    self.next_check = 0.0
    self.refresh(0.0)

  def __getattr__(self, key):
    try:
      return self.__dict__["values"][key]
    except KeyError:
      raise AttributeError(key) from None

  def refresh(self, now: float) -> bool:
    """Reads the file again if it changed; returns whether it did."""
    if not self.path or now < self.next_check:
      return False
    self.next_check = now + self.CHECK_EVERY
    try:
      mtime = os.path.getmtime(self.path)
      if mtime == self.mtime:
        return False
      with open(self.path) as f:
        given = json.load(f)
    except (OSError, ValueError) as e:
      if self.mtime is not None or os.path.exists(self.path):
        print(f"nav: tune {self.path}: {e}")
      self.mtime = None
      return False
    self.mtime = mtime
    unknown = sorted(set(given) - set(self.DEFAULTS))
    if unknown:
      print(f"nav: tune ignores {unknown}")
    self.values = {**self.DEFAULTS, **{k: v for k, v in given.items() if k in self.DEFAULTS}}
    return True

  def changed(self) -> dict:
    return {k: v for k, v in self.values.items() if v != self.DEFAULTS[k]}


TUNE = Tune("")  # the defaults


def remap(targets: tuple[int, int, int], n: int, left: bool) -> tuple[int, int]:
  """Target lanes lo-hi of m lanes as lanes of n, counted from the left (else from the right) where they differ."""
  lo, hi, m = targets
  if not left:
    lo, hi = lo + n - m, hi + n - m
  return min(max(lo, 0), n - 1), min(max(hi, 0), n - 1)


def parse_arrows(arrows) -> list[tuple]:
  """Route.info's laneArrows: [(m ahead to where they end, each lane's arrows from the left[, the route's move through
  the junction as they call it: 'left', 'through' or 'right'])]."""
  return [(float(d), [frozenset(a.split(";")) - {""} for a in lanes], *move) for d, lanes, *move in arrows or []]


def arrows_entry(arrows: list, dist: float) -> tuple | None:
  """The arrows entry (parse_arrows') of the junction of a turn or fork `dist` m on, None for none."""
  near = [(abs(a[0] - dist), a) for a in arrows if dist - ARROWS_BEFORE <= a[0] <= dist + ARROWS_AFTER]
  return min(near, key=lambda n: n[0])[1] if near else None


def arrows_at(arrows: list, dist: float) -> list[frozenset[str]] | None:
  """The arrows of the lanes into the junction of a turn or fork `dist` m on (parse_arrows'), None for none."""
  entry = arrows_entry(arrows, dist)
  return entry[1] if entry is not None else None


def aim(m, arrows: list):
  """A turn or fork's target lanes from the map's turn arrows, where its junction has them. The move they're for is the
  one the arrows call the route's (a turn of a skewed junction's road on is through), else the turn or fork's side."""
  entry = arrows_entry(arrows, m.dist)
  lanes = entry[1] if entry is not None else None
  move = entry[2] if entry is not None and len(entry) > 2 else None
  if lanes and move is not None and (isinstance(m, Turn) and move == "through" or getattr(m, "junction", False)):
    found = turn_targets(lanes, move)
  else:
    found = turn_targets(lanes, m.side, isinstance(m, Fork)) if lanes else []
  m.targets = (min(found), max(found), len(lanes)) if found else None


class Turn:
  def __init__(self, dist: float, side: str, exit_heading: float, angle: float = 90.0):
    self.dist = dist  # m along the route to the turn
    self.side = side  # "left" or "right"
    self.exit_heading = exit_heading  # game heading (deg counterclockwise from north) after the turn
    self.angle = angle  # deg turned
    self.targets: tuple[int, int, int] | None = None  # lanes lo-hi of n the map's arrows allow it from

  def speed(self, tune: Tune | None = None) -> float:
    t = tune or TUNE
    if self.angle > t.sharp_angle:
      return t.turn_speed_sharp
    square = t.turn_speed_square_right if self.side == "right" and t.turn_speed_square_right > 0 else t.turn_speed_square
    return float(np.interp(self.angle, [TURN_ANGLE, 90.0], [t.turn_speed_soft, square]))

  def lanes(self, n: int) -> tuple[int, int]:
    """The lanes (from the left) of n to take it from: those the map's arrows allow, else the outside lane."""
    if self.targets is not None:
      return remap(self.targets, n, self.side == "left")
    return (0, 0) if self.side == "left" else (n - 1, n - 1)


class Fork:
  def __init__(self, dist: float, side: str, lanes: int, lanes_in: int, keep: bool, other: int = 0, slip: bool = False):
    self.dist = dist  # m along the route
    self.side = side  # the branch the route takes
    self.ours, self.lanes_in = lanes, lanes_in  # lanes on the route's branch, and on the road before
    self.other = other  # lanes on the other branches, which GTA sometimes counts on top of the road's
    # whether to hold the keep desire through it: a fork in the road, where the route leaves the road for the other
    # branch, not where it stays on it past a smaller one (L2: keepLeft there took the car into the oncoming lanes)
    main = other > 0 and lanes > other and lanes >= lanes_in - other
    self.keep = keep and not main
    self.slip = slip  # the other branch opens a slip lane or turn bay
    self.junction = False  # the route's move through a junction rather than a fork in the road: aimed as the arrows call it
    self.targets: tuple[int, int, int] | None = None  # lanes lo-hi of n the map's arrows allow it from

  def lanes(self, n: int) -> tuple[int, int]:
    if self.targets is not None:
      return remap(self.targets, n, self.side == "left")
    ours = min(self.ours, self.lanes_in - self.other) if 0 < self.other < self.lanes_in else self.ours
    if self.other and ours >= 3:
      ours -= 1  # and not the lane beside the other branch on a wide road, which drifts into it
    if self.lanes_in <= 1 or ours >= self.lanes_in:
      return 0, n - 1  # every lane goes the route's way
    ours = max(1, min(ours, n))
    return (0, ours - 1) if self.side == "left" else (n - ours, n - 1)


class Through:
  """Straight on through a junction where some of the lanes into it only turn, or end there: the lanes that go on."""
  def __init__(self, dist: float, targets: tuple[int, int, int]):
    self.dist = dist  # m along the route to the end of the lanes' arrows
    self.side = "through"
    self.targets = targets

  def lanes(self, n: int) -> tuple[int, int]:
    return remap(self.targets, n, self.targets[0] == 0)


def throughs(route: np.ndarray, arrows: list, moves: list, drops: list | None = None) -> list[Through]:
  """The junctions the route goes straight on through (no turn or fork near) where the arrows leave some lanes out, or
  where some lanes end (Route.info's laneDrops: [m ahead, first and last lane that carry on, of how many]). Straight on
  is as the arrows call the route's move, else by how far it turns."""
  out = []
  along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T)))) if len(route) >= 2 else np.zeros(1)

  def clear(d):
    return d > 0 and not any(d - ARROWS_AFTER <= m.dist <= d + ARROWS_BEFORE for m in moves) and d + 25.0 <= along[-1] and d >= 15.0
  for d, lo, hi, n in drops or []:
    if clear(d):
      out.append(Through(d, (lo, hi, n)))
  for d, lanes, *move in arrows:
    if not clear(d):
      continue
    found = turn_targets(lanes, "through")
    if not found or len(found) == len(lanes):
      continue
    if move:  # as the arrows call the route's move, also a skewed junction's road on
      if move[0] != "through":
        continue
    else:
      p = [np.array([np.interp(v, along, route[:, 0]), np.interp(v, along, route[:, 1])]) for v in (d - 15.0, d, d + 25.0)]
      h_in, h_out = (math.degrees(math.atan2(*(b - a)[::-1])) for a, b in ((p[0], p[1]), (p[1], p[2])))
      if abs(wrap(h_out - h_in)) > THROUGH_TURN:
        continue
    out.append(Through(d, (min(found), max(found), len(lanes))))
  return out


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
  # from the segment `after` falls in, as the one into a turn's node starts behind the car near the junction
  reach = np.append(starts[1:], np.inf)
  candidates = np.nonzero((reach > after) & (ends > idx) & (np.abs(change) >= TURN_ANGLE))[0]
  for i in candidates:
    j = int(ends[i])
    # the turn is at its sharpest vertex in the window
    steps = np.abs((np.diff(heads[i:j + 1]) + 180) % 360 - 180)
    k = i + int(np.argmax(steps)) + 1
    held = int(np.searchsorted(starts, starts[k] + TURN_HOLDS, side='right')) - 1
    exit_change = abs(wrap(heads[max(held, j)] - heads[i]))
    if exit_change < TURN_ANGLE:
      continue
    jog = 0 < i < len(heads) - 1 and starts[i + 1] - starts[i] < JOG_LINK
    if jog and np.sign(change[i]) * wrap(heads[max(held, j)] - heads[i - 1]) <= THROUGH_TURN:
      continue  # off a junction's link (a jog): from the road before it, the way on is straight on, or the other way
    if exit_change > U_TURN_ANGLE:
      return None
    return Turn(float(starts[k]), "left" if change[i] > 0 else "right", float(heads[max(held, j)]), exit_change)
  return None


def curve_cap(route: np.ndarray, v: float, tune: Tune | None = None) -> float:
  """The speed now that leaves room to slow for the bends ahead, 0 for none. A bend's curvature is its heading change
  over CURVE_WINDOW m, but no more than over twice or four times the window: GTA's lanes jog sideways through junctions
  and where a freeway splits, turning one way and back, which isn't a bend."""
  heads, starts = headings(route)
  if len(heads) < 3:
    return 0.0
  h = np.unwrap(np.radians(heads))
  mids = starts + np.diff(np.append(starts, starts[-1] + 5.0)) / 2
  s = np.arange(0.0, min(CURVE_LOOKAHEAD, float(mids[-1])), CURVE_STEP)
  if len(s) == 0:
    return 0.0
  w = CURVE_WINDOW / 2

  def turned(a, b):
    return np.abs(np.interp(b, mids, h) - np.interp(a, mids, h))
  curvature = np.minimum.reduce([turned(s - k * w, s + k * w) for k in (1, 2, 4)]) / CURVE_WINDOW
  # the turns themselves have their own speed
  t = tune or TUNE
  speed = np.maximum(np.sqrt(t.curve_accel / np.maximum(curvature, 1e-4)), t.turn_speed_square)
  return float(np.min(np.sqrt(speed ** 2 + 2 * t.slow_decel * np.maximum(s - w - v * SLOW_LAG, 0.0))))


def limit_cap(limits: list, v: float) -> float:
  """The speed now that leaves room to slow for lower speed limits ahead ([[m ahead, m/s or 0 unknown], ...]), 0 for
  none; the set speed follows the limit where the car is."""
  caps = [math.sqrt(limit ** 2 + 2 * LIMIT_DECEL * max(0.0, d - v * SLOW_LAG)) for d, limit in limits if limit > 0 and d > 0]
  return min(caps) if caps else 0.0


def lowest(caps: list[tuple[float, str]]) -> tuple[float, str]:
  """The lowest of the caps (m/s, 0 for none) with its reason, no lower than a crawl; NO_CAP with none."""
  caps = [c for c in caps if c[0] > 0]
  if not caps:
    return NO_CAP
  cap, reason = min(caps, key=lambda c: c[0])
  return max(cap, 0.5), reason


def turn_reason(side: str) -> str:
  return "turnLeft" if side == "left" else "turnRight"


def slow_for(speed: float, dist: float, v: float, decel: float = SLOW_DECEL) -> float:
  return math.sqrt(speed ** 2 + 2 * decel * max(0.0, dist - v * SLOW_LAG))


def lane_plan(route: np.ndarray, forks: list, lane, lanes_at, v: float, tune: Tune | None = None,
              arrows: list | None = None, drops: list | None = None) -> list[tuple[float, float]]:
  """The lanes nav aims for along the whole route, for the map: [(m along, lane from the left)], ramping between each
  two. From the car's lane (out of the oncoming lanes first), it changes only for a turn or fork whose lanes it isn't
  in (by the map's turn arrows where it has them, Route.info's laneArrows) or to go straight on past lanes that only
  turn or end (laneDrops), by where nav's changes for it must have ended, and arrives from a turn in its side's outside
  lane. lanes_at(m, after) is the lanes the car's way just before (after: past) a point."""
  t = tune or TUNE
  arrows = parse_arrows(arrows)
  ahead: list[Turn | Fork | Through] = []
  turn = find_turn(route, MIN_AHEAD_MAP)
  while turn is not None:
    ahead.append(turn)
    turn = find_turn(route, turn.dist + TURN_HOLDS)
  ahead += [Fork(d, side, *rest) for d, side, *rest in forks if d > 0]
  for m in ahead:
    aim(m, arrows)
  ahead += throughs(route, arrows, ahead, drops if t.lane_drops else None)
  ahead.sort(key=lambda m: m.dist)
  cur = lane[0] if lane else 0
  keys = [(0.0, float(cur))]
  free = 0.0  # m along from which the next change may start: past the turn or fork before
  if cur < 0:
    free, cur = -cur * LANE_LINE_CHANGE, 0
    keys.append((free, 0.0))
  for m in ahead:
    n = lanes_at(m.dist, False)
    if n <= 0:
      continue
    cur = min(cur, n - 1)
    lo, hi = m.lanes(n)
    if not lo <= cur <= hi:
      want = lo if cur < lo else hi
      last = max(FORK_LAST_DIST, FORK_LAST * v) if isinstance(m, Fork) else t.lane_change_last
      end = min(max(m.dist - last, free), m.dist)
      start = min(max(end - abs(want - cur) * LANE_LINE_CHANGE, free), end)
      keys += [(start, float(cur)), (end, float(want))]
      cur = want
    if isinstance(m, Turn):
      new = 0 if m.side == "left" else max(lanes_at(m.dist, True) - 1, 0)
      free = m.dist + TURN_HOLDS
    elif isinstance(m, Through):
      new = min(cur, max(lanes_at(m.dist, True) - 1, 0))
      free = m.dist
    else:
      new = max(cur - max(n - m.ours, 0), 0) if m.side == "right" else cur  # numbered from the branch's own left lane
      free = m.dist
    keys += [(m.dist, float(cur)), (m.dist, float(new))]
    cur = new
  return keys


def junction_entry(turn_dist: float, stops: list, junctions: list) -> tuple[float, str]:
  """Where the car enters a turn's junction (m ahead, from the same point as turn_dist): its stop line within
  ENTRY_STOP_BEFORE m before the turn, else the first of the junction's nodes leading up to it, else ENTRY_DEFAULT m
  before it; and which of those it is."""
  before = [d for d in stops if turn_dist - ENTRY_STOP_BEFORE < d <= turn_dist]
  if before:
    return max(before), "stop line"
  run = sorted((d for d in junctions if turn_dist - ENTRY_JUNCTION_BEFORE < d <= turn_dist + ENTRY_PAST), reverse=True)
  first = None
  for d in run:  # back from the turn while the nodes stay close together
    if first is not None and first - d > ENTRY_GAP:
      break
    first = d
  if first is not None and first < turn_dist - ENTRY_MIN:
    return first, "junction"
  return turn_dist - ENTRY_DEFAULT, "default"


class Planner:
  def __init__(self, tune: Tune | None = None, refresh: bool = False):
    self.tune = tune or Tune()
    self.requests: list[str] = []  # this step's requests to the driver (NavOutputs)
    self.desires: list[str] = []  # this step's NavDesire changes
    self.now = 0.0
    self.refresh = refresh  # openpilot asks for the turn again itself (TurnDesireRefresh): the blinker stays on
    self.was_engaged = False
    self.desire = ""  # what NavDesire is set to
    self.entry, self.entry_kind = 0.0, ""  # m to the junction entry of the turn ahead, and what it is
    self.hold: dict | None = None  # the speed held through the turn signalled last, and the speeds through it
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
    self.swept = 0.0  # deg the heading has changed since signaling, accumulated (left positive)
    self.swept_heading = 0.0  # the heading it was last accumulated at
    self.watch: dict | None = None  # the turn just over, while its pulse may be queued (exit_watch)
    self.stopped = False
    self.seen: tuple[str, float] | None = None  # the turn ahead's side, and since when
    self.changing: str | None = None  # the side of nav's lane change under way
    self.change_turn = False  # whether it's into the lanes for a turn
    self.change_side: str | None = None  # the last lane change's side, under way or ended
    self.change_shown = False
    self.change_t = 0.0  # when it started, or the last ended
    self.change_tries: dict[tuple, int] = {}  # lane changes for each turn or fork (by where it is), without progress
    self.change_from: tuple[int, int] | None = None  # the lane the last change started from
    self.change_send_at = 0.0  # when to put the blinker on for it, once openpilot has read NavDesire; 0 once on
    self.change_hold_until = 0.0  # NavDesire stays a lane change until then, after the blinker went off
    self.turn_point: np.ndarray | None = None  # where the signaled turn is
    self.signaled_at: np.ndarray | None = None  # where the car was on the route when it signaled it
    self.turned_at = -1e9  # self.driven when the last turn was done
    self.min_ahead = MIN_AHEAD
    self.bay_to = 0.0  # self.driven to which the car is changing into, or is in, a turn bay
    self.yaw = 0.0
    self.lane_frac: float | None = None  # the car's lane from its position on the route's link, between lanes as it changes
    self.one_way = False  # whether the road is one-way, as a keepLeft on a two-way road drifts into the oncoming lanes
    self.cue: str | None = None  # the keep desire held against swinging round past a turn's way out
    self.cue_until = 0.0
    self.cue_hold = False  # held its time even once the car is straight
    self.cue_staged = False  # the keepRight started within the turn (exit_cue_at), stacked on it
    self.recover: str | None = None  # keepRight out of the oncoming lanes (oncoming_keep)
    self.recover_t = 0.0
    self.oncoming_since: float | None = None
    self.lane: tuple[int, int] | None = None  # the car's lane [i from the left, of n], once it has held LANE_STEADY
    self.lane_seen: tuple[tuple[int, int] | None, float] = (None, 0.0)
    self.lane_t = 0.0  # when the lane reading last held
    self.skipped: list[np.ndarray] = []  # where turns and forks left to the route are
    self.taken: list[np.ndarray] = []  # where turns taken are
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

  def update(self, inp: NavInputs) -> NavOutputs:
    """The step's cruise cap, arrival, requests to the driver and NavDesire's changes."""
    self.requests, self.desires = [], []
    (cap, reason), arrived = self._update(inp)
    return NavOutputs(cap, arrived, self.requests, self.desires, reason)

  def _update(self, inp: NavInputs) -> tuple[tuple[float, str], bool]:
    """The cruise cap (m/s, 0 for none) and its reason, and whether the car has arrived."""
    now = self.now = inp.t
    # guidance only (Navigate on openpilot off) takes no nav actions, as when disengaged: signals cancelled, no lane
    # changes, turn or keep desires, turn slowing or arrival; but what keeps the car on its road stays (_guide)
    engaged, indicator, desire = inp.engaged and inp.drive, inp.blinker, inp.desire
    if self.tune.refresh(now) or engaged and not self.was_engaged:
      print(f"nav: tune {json.dumps(self.tune.changed())} from {self.tune.path or 'defaults'}")
    self.was_engaged = engaged
    t = self.tune
    v = self.v = inp.v
    step = v * min(now - self.last_t, 0.1)
    self.last_t = now
    self.driven += step
    route = inp.route
    pos = np.array(inp.pos[:2], dtype=float)
    self._read_lane(inp.truth_lane, now)
    if not engaged:
      self._cancel(indicator)
      self._end_change(indicator)
      self.keeping, self.fork_keep = None, None
      cap = self._guide(inp, now) if inp.engaged else NO_CAP
      self._set_desire(now)
      self.route_end, self.dest = None, None
      self.skipped.clear()
      self.taken.clear()
      self.hold = None
      return cap, False
    self._watch_change(indicator, now)
    if self.changing is not None and self.change_send_at and now >= self.change_send_at:
      self.change_send_at = 0.0
      self.requests.append("laneChange" + self.changing.capitalize())
    self.yaw = inp.yaw_rate
    self.lane_frac, self.one_way = inp.truth_lane_frac, inp.two_way is False
    self._keep_right(v, now)
    near = [d for d in (inp.stops or []) + (inp.junctions or []) if abs(d) < ONCOMING_JUNCTION]
    self._oncoming_keep(inp.truth_lane_plugin, bool(near), now, inp.truth_lane_map)
    if not route:
      self._cancel(indicator)
      self.keeping, self.fork_keep = None, None
      self._set_desire(now)
      left = None if self.dest is None else float(np.hypot(*(self.dest - pos)))
      if left is not None and left < ARRIVE_KEEP:
        # counted down by the distance driven, which doesn't grow again if the car runs past it
        self.route_end = left if self.route_end is None else min(self.route_end - step, left)
        stop, arrived = self._arrive(v)
        return (stop, "arrival"), arrived
      self.route_end, self.dest = None, None
      return NO_CAP, False
    route = np.array(route, dtype=float)
    waypoint = np.array(inp.dest or (0.0, 0.0), dtype=float)
    self.dest = waypoint if waypoint.any() else route[-1]  # (0, 0) as GTA clears it
    self.route_end = None
    self.min_ahead = MIN_AHEAD_MAP if inp.route_end is not None else MIN_AHEAD
    if inp.route_end is not None:
      if inp.route_end < ARRIVE_KEEP * 5:
        self.route_end = float(inp.route_end)
    elif len(route) < ROUTE_POINTS:
      # along the route to its point nearest the waypoint: past that it sometimes runs on
      along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))))
      near = int(np.argmin(np.hypot(*(route - self.dest).T)))
      self.route_end = float(along[near] + np.hypot(*(route[near] - self.dest)))
    heading, yaw_rate = inp.heading, inp.yaw_rate
    if self.turn is not None:
      self.swept += wrap(heading - self.swept_heading)
      self.swept_heading = heading
    self.skipped = [p for p in self.skipped if np.hypot(*(p - pos)) < SKIP_PAST + 2 * FORK_LOOKAHEAD
                    and not self._passed(p, pos, heading)]
    self.taken = [p for p in self.taken if np.hypot(*(p - pos)) < SKIP_PAST]

    # the bridge cancels the indicator once the car has turned, as does the driver to take over from ours, once shown
    self.shown |= self.signaled is not None and indicator == self.signaled
    if self.signaled is not None and self.shown and indicator != self.signaled:
      self._turn_over(now)
      self.turn, self.signaled = None, None

    if self.turn is not None:
      along, off = self.driven - self.turn_from, abs(wrap(heading - self.turn.exit_heading))
      out = along > self.turn.dist - TURN_WINDOW / 2 and off < EXIT_DONE
      done = out or along > self.turn.dist + TURN_PAST and off < TURN_PAST_HEADING and abs(yaw_rate) < DONE_YAW_RATE
      turning = yaw_rate if self.turn.side == "left" else -yaw_rate
      cued = t.exit_cue_right if self.turn.side == "right" else t.exit_cue_left
      if EXIT_CUE and cued and t.exit_cue_at < 1.0 and not self.cue_staged and not out:
        angle = max(abs(wrap(self.turn.exit_heading - self.signal_heading)), 1.0)
        if t.exit_cue_at <= 0.0 or 1.0 - off / angle >= t.exit_cue_at:
          self.cue, self.cue_until, self.cue_hold, self.cue_staged = "keepRight", float("inf"), True, True
          if DEBUG:
            print(f"nav: keepRight stacked on the {self.turn.side} turn, {100 * (1.0 - off / angle):.0f}% round")
      if out and self.cue_staged and self.cue_until == float("inf"):
        self.cue_until = now + EXIT_CUE_FOR
      if out and EXIT_CUE and self.turn.side == "right" and t.exit_cue_right:
        self.cue, self.cue_until, self.cue_hold = "keepRight", now + EXIT_CUE_FOR, True
        if DEBUG:
          print("nav: keepRight out of the right turn")
      if out and EXIT_CUE and self.turn.side == "left" and t.exit_cue_left and turning > EXIT_CUE_YAW:
        self.cue, self.cue_until, self.cue_hold = "keepRight", now + EXIT_CUE_FOR, False
        if DEBUG:
          print(f"nav: keepRight as the car comes round past the left turn ({turning:.2f} rad/s)")
      unturned = (t.unturned_cancel > 0 and along > self.turn.dist + t.unturned_cancel
                  and self._turned(heading) < TURN_STARTED)
      if done or along > self.turn.dist + MISSED_BY or unturned:
        if DEBUG:
          print(f"nav: turn {'done' if done else 'missed'} after {along:.0f} m (car {heading:.0f})")
        self._turn_over(now)
        self._cancel(indicator)

    turn = self._next_turn(route) if self.turn is None else None
    if turn is not None and self._behind(route, heading):
      turn = None  # GTA routes on once the car has passed somewhere to turn around
    if turn is None:
      self.seen = None
    elif self.seen is None or self.seen[0] != turn.side:
      self.seen = (turn.side, now)
    if turn is not None and now - self.seen[1] < CONFIRM:
      turn = None
    forks = self._forks(inp.forks, route, turn)
    arrows = parse_arrows(inp.lane_arrows)
    moves: list = forks + ([turn] if turn is not None else [])
    for m in moves:
      aim(m, arrows)
    drops = inp.lane_drops if t.lane_drops else None
    ahead = moves + [m for m in throughs(route, arrows, moves, drops) if turn is None or m.dist < turn.dist]
    caps = [(self._change_lane(sorted(ahead, key=lambda m: m.dist), route, v, now), "laneChange")]
    if turn is not None:
      self.entry, self.entry_kind = junction_entry(turn.dist, inp.stops or [], inp.junctions or [])
    if turn is not None and turn.dist < t.slow_from:
      bay = self._bay(forks, turn)
      ref = self.entry if t.slow_ref == "entry" else turn.dist
      caps.append((slow_for(turn.speed(t), min(ref - t.slow_done, turn.dist - bay - BAY_SIGNAL), v, t.slow_decel), turn_reason(turn.side)))
      self._enter_bay(turn, bay, v, now)
      if self.changing is not None and self.driven < self.bay_to and t.bay_speed > 0:
        caps.append((t.bay_speed, "bay"))
      self._signal(turn, route, indicator, v, now, heading, bay)
    caps.append((self._hold_cap(heading, yaw_rate, v, now), turn_reason(self.hold['side']) if self.hold else ""))
    if self.turn is not None:
      if v < 0.3:
        self.stopped = True
      elif (v > 1.0 and self.shown and now - self.repeat_t > REPEAT_EVERY
            and (t.unturned_cancel <= 0 or t.unturned_keep_pulses or self.driven - self.turn_from < self.turn.dist)):
        turned = self._turned(heading)
        fading = desire.get(self.turn.side, 1.0) < t.repulse_below_prob or now - self.repeat_t > t.repulse_every
        past = t.no_repulse_past_exit and self._past_exit() > 0
        if not past and (turned < t.repulse_until_turned and fading or self.stopped and turned < t.repulse_after_stop_until):
          self.repeat_t = now
        self.stopped = False
    self._exit_watch(heading, yaw_rate, now)
    caps.append((curve_cap(route, v, t), "bend"))
    caps.append((limit_cap(inp.limits or [], v), "speedLimit"))
    self._keep_fork(forks[0] if forks else None, turn, desire, v, now)
    self._keep_straight(turn, desire, v, now)
    self._set_desire(now)

    cap = lowest(caps)
    if self.route_end is not None:
      stop, arrived = self._arrive(v)
      return (stop, "arrival") if not cap[0] or stop <= cap[0] else cap, arrived
    return cap, False

  def _guide(self, inp: NavInputs, now: float) -> tuple[float, str]:
    """Guidance only, engaged: the cruise cap for the road's bends and speed limits ahead, the bends only up to the next
    turn at a junction (the driver may not take it, and turns aren't slowed for), and keepRight out of the oncoming
    lanes."""
    self.yaw, self.one_way = inp.yaw_rate, inp.two_way is False
    near = [d for d in (inp.stops or []) + (inp.junctions or []) if abs(d) < ONCOMING_JUNCTION]
    self._oncoming_keep(inp.truth_lane_plugin, bool(near), now, inp.truth_lane_map)
    caps = [(limit_cap(inp.limits or [], inp.v), "speedLimit")]
    if inp.route:
      route = np.array(inp.route, dtype=float)
      turn = find_turn(route, MIN_AHEAD_MAP if inp.route_end is not None else MIN_AHEAD)
      while turn is not None and not any(abs(d - turn.dist) < ENTRY_JUNCTION_BEFORE / 2 for d in inp.junctions or []):
        turn = find_turn(route, turn.dist + TURN_HOLDS)  # a bend of the road, not a turn: slowed for
      if turn is not None:
        along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))))
        route = np.vstack([route[along < turn.dist], self._point(route, turn.dist)])
      caps.append((curve_cap(route, inp.v, self.tune), "bend"))
    return lowest(caps)

  def _hold_cap(self, heading: float, yaw_rate: float, v: float, now: float) -> float:
    """The turn's speed, held through its arc and lifted gently once the car is out of it (0 for none); logs the
    speeds through the turn as it lifts."""
    h = self.hold
    if h is None:
      return 0.0
    t = self.tune
    if h['entry_v'] is None and self.driven >= h['entry_at']:
      h['entry_v'] = v
    turned = abs(wrap(heading - h['heading']))
    if h['released'] is None:
      if turned > HOLD_ARC:
        h['arc'].append(v)
      straight = abs(wrap(heading - h['exit'])) < t.turn_release and abs(yaw_rate) < DONE_YAW_RATE
      if straight and h['arc'] or self.driven > h['end_at']:
        h['released'] = now
        if DEBUG:
          arc = h['arc'] or [v]
          entry = "-" if h['entry_v'] is None else f"{h['entry_v']:.1f}"
          speeds = f"entry {entry}, arc min {min(arc):.1f} mean {np.mean(arc):.1f}, exit {v:.1f} m/s"
          print(f"nav: turn speeds {h['side']} {h['angle']:.0f} deg: {speeds}, turned {turned:.0f} deg, held {h['speed']:.1f}")
      return h['speed']
    cap = h['speed'] + t.release_accel * (now - h['released'])
    if cap > HOLD_DONE:
      self.hold = None
      return 0.0
    return cap

  def _turned(self, heading: float) -> float:
    """deg the car has turned since signaling, either way: accumulated with turned_unwrapped, else wrapped."""
    return abs(self.swept) if self.tune.turned_unwrapped else abs(wrap(heading - self.signal_heading))

  def _past_exit(self) -> float:
    """deg the car has come round past the signaled turn's way out, by the accumulated heading (< 0: short of it)."""
    sgn = 1.0 if self.turn.side == "left" else -1.0
    return sgn * self.swept - abs(wrap(self.turn.exit_heading - self.signal_heading))

  def _exit_watch(self, heading: float, yaw_rate: float, now: float):
    """exit_watch: the counter keep desire while the turn just over may still be queued as a pulse and the car keeps
    coming round past its way out."""
    w = self.watch
    if w is None:
      return
    if self.turn is not None or now > w["until"]:
      self.watch = None
      return
    w["past"] += w["sgn"] * wrap(heading - w["heading"])
    w["heading"] = heading
    if w["past"] > EXIT_WATCH_PAST and w["sgn"] * yaw_rate > EXIT_CUE_YAW:
      if DEBUG and self.cue != w["cue"]:
        print(f"nav: {w['cue']}, {w['past']:.0f} deg past the {w['side']} turn's way out with its pulse maybe still queued")
      self.cue, self.cue_until, self.cue_hold = w["cue"], now + EXIT_CUE_FOR, False

  def _turn_over(self, now: float):
    if self.turn is not None and self.tune.exit_watch:
      sgn = 1.0 if self.turn.side == "left" else -1.0
      self.watch = {"side": self.turn.side, "sgn": sgn, "cue": "keepRight" if sgn > 0 else "keepLeft",
                    "past": sgn * wrap(self.swept_heading - self.turn.exit_heading), "heading": self.swept_heading,
                    "until": self.repeat_t + REPEAT_GAP + EXIT_WATCH_QUEUE}
    if self.turn_point is not None:
      self.taken.append(self.turn_point)  # the rest of it can look like a turn ahead
    self.cooldown_until, self.turned_at, self.turn_point = now + COOLDOWN, self.driven, None
    if self.cue_staged and self.cue_until == float("inf"):
      self.cue = None
    self.cue_staged = False

  def _next_turn(self, route: np.ndarray) -> Turn | None:
    """The first turn ahead not left to the route."""
    turn = find_turn(route, max(self.min_ahead, self.skip_turns_to - self.driven))
    while turn is not None and self._is_skipped(route, turn.dist):
      turn = find_turn(route, turn.dist + TURN_HOLDS)
    return turn

  def _forks(self, forks: list | None, route: np.ndarray, turn: Turn | None) -> list[Fork]:
    """The forks ahead, before any turn."""
    out = []
    for d, side, *rest in forks or []:
      if d > FORK_LOOKAHEAD or (turn is not None and d > turn.dist):
        break
      if d > 0 and not self._is_skipped(route, d):
        out.append(Fork(d, side, *rest))
    return out

  @staticmethod
  def _bay(forks: list[Fork], turn: Turn | None) -> float:
    """How far before the turn ahead a slip lane or turn bay for it opens, 0 for none."""
    if turn is None:
      return 0.0
    bays = [turn.dist - f.dist for f in forks if f.slip and f.side != turn.side and turn.dist - f.dist < BAY_BEFORE]
    return max(bays) if bays else 0.0

  @staticmethod
  def _point(route: np.ndarray, dist: float) -> np.ndarray:
    along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))))
    return np.array([np.interp(dist, along, route[:, 0]), np.interp(dist, along, route[:, 1])])

  def _is_skipped(self, route: np.ndarray, dist: float) -> bool:
    if not self.skipped and not self.taken:
      return False
    p = self._point(route, max(dist, 0.0))
    return any(np.hypot(*(p - s)) < SKIP_NEAR for s in self.skipped) or any(np.hypot(*(p - s)) < TAKEN_NEAR for s in self.taken)

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

  def _signal(self, turn: Turn, route: np.ndarray, indicator: str | None, v: float, now: float, heading: float, bay: float):
    """Signals the turn once near and slow enough, from its lane; one that can't be taken from the car's lane is left."""
    t = self.tune
    into = t.lane_change_into_turn > 0 and self.changing == turn.side and self.change_turn
    if into and turn.dist > t.lane_change_into_turn:
      return
    if into:
      # the blinker stays on and, with NavDesire cleared, asks for the turn, which overrides the lane change
      if DEBUG:
        print(f"nav: lane change {turn.side} becomes the turn in {turn.dist:.0f} m")
      self.changing, self.change_t, self.change_send_at = None, now, 0.0
      self._set_desire(now)
    elif not self._signal_due(turn, route, indicator, v, now, bay):
      return
    self.turn, self.signaled, self.turn_from, self.shown = turn, turn.side, self.driven, False
    t = self.tune
    self.hold = {'side': turn.side, 'angle': turn.angle, 'exit': turn.exit_heading, 'heading': heading,
                 'speed': t.turn_hold_speed or turn.speed(t), 'entry_at': self.driven + self.entry, 'entry_v': None,
                 'end_at': self.driven + turn.dist + t.turn_release_m, 'arc': [], 'released': None}
    self.turn_point = self._point(route, turn.dist)
    self.signaled_at = route[0].copy()
    self.signal_heading, self.repeat_t = heading, now
    self.swept, self.swept_heading = 0.0, heading
    if DEBUG:
      lane = f"lane {self.lane} at {self.lane_frac}"
      entry = f"{self.entry_kind} in {self.entry:.0f} m"
      print(f"nav: signal {turn.side} in {turn.dist:.0f} m, {entry}, exit heading {turn.exit_heading % 360:.0f} (car {heading:.0f}, {v:.1f} m/s, {lane})")
    self.requests.append("signalTurn" + turn.side.capitalize())

  def _signal_window(self, v: float, bay: float) -> float:
    """m before the turn the time mode signals from, once slow enough."""
    t = self.tune
    return max(t.signal_min, min(t.signal_max, v * t.signal_time), bay + BAY_SIGNAL)

  def signal_from(self, turn: Turn, entry: float, v: float, bay: float) -> float:
    """m along the route where _signal_due's distances first allow the turn's signal (it also waits to be slow enough
    and in the turn's lanes), the junction's entry `entry` m along."""
    t = self.tune
    if t.signal_mode == "entry" and self.min_ahead == MIN_AHEAD_MAP:
      return min(entry + t.signal_entry_offset, turn.dist - SIGNAL_LAST)
    at = turn.dist - self._signal_window(v, bay)
    if t.signal_min_entry > 0:
      at = min(at, entry - t.signal_min_entry)
    if turn.side == "left" and t.left_signal_max_entry > 0:
      at = max(at, min(entry - t.left_signal_max_entry, turn.dist - SIGNAL_LAST))
    return at

  def turn_points(self, route: np.ndarray, forks: list | None, stops: list | None,
                  junctions: list | None) -> tuple[np.ndarray, np.ndarray | None] | None:
    """For the map overlay: where the turn nav signals next is and where its signal comes on by distance; once
    signalled, where the car was when it did."""
    if self.turn is not None and self.turn_point is not None:
      return self.turn_point, self.signaled_at
    turn = self._next_turn(route)
    if turn is None:
      return None
    bay = self._bay(self._forks(forks, route, turn), turn)
    entry, _ = junction_entry(turn.dist, stops or [], junctions or [])
    return self._point(route, turn.dist), self._point(route, max(self.signal_from(turn, entry, self.v, bay), 0.0))

  def _signal_due(self, turn: Turn, route: np.ndarray, indicator: str | None, v: float, now: float, bay: float) -> bool:
    """Whether to signal the turn now, ending any lane change towards it, or leaving it to the route."""
    t = self.tune
    slow = v < 19 * CV.MPH_TO_MS and self.changing is None
    if t.signal_mode == "entry" and self.min_ahead == MIN_AHEAD_MAP:
      # at the junction's entry, as a driver does (too early, the model takes it for a lane change or turns short)
      due = self.entry <= -t.signal_entry_offset or turn.dist < SIGNAL_LAST
      if not (due and (slow or turn.dist < SIGNAL_LAST)):
        return False
    else:
      window = self._signal_window(v, bay)
      early = t.signal_min_entry > 0 and self.entry < t.signal_min_entry
      if not (turn.dist < t.signal_min or ((early or turn.dist < window) and slow)):
        return False
      # not after a lane change towards it: held, its lane change desire carries on into the junction
      changed = self.change_side == "left" and (self.changing is not None or now - self.change_t < LEFT_SIGNAL_AFTER_CHANGE)
      if (turn.side == "left" and t.left_signal_max_entry > 0 and self.entry > t.left_signal_max_entry and not changed
          and turn.dist >= SIGNAL_LAST):
        return False
    if self._bay_beside(turn, bay) and self.driven >= self.bay_to and turn.dist > BAY_LAST:
      return False  # into the turn bay first
    if self.lane is not None and self.lane[0] >= 0:
      lo, hi = turn.lanes(self.lane[1])
      if not lo <= self.lane[0] <= hi:
        if turn.dist > t.lane_change_last or (self.changing is not None and turn.dist > SIGNAL_LAST_DIST):
          return False  # a lane change towards it may still come or finish
        beside = t.turn_from_beside and self.lane[0] in (lo - 1, hi + 1) and (
          self.lane_frac is None or lo - 1 - SKIP_SURE <= self.lane_frac <= hi + 1 + SKIP_SURE)
        if self._sure_wrong(lo, hi) and not beside:
          self._end_change(indicator)
          self._skip(route, turn.dist, f"{turn.side} turn")
          return False
        if beside and DEBUG:
          print(f"nav: not in lane {self.lane} for the {turn.side} turn in {turn.dist:.0f} m; signalling it from beside its lanes")
    if self.changing is not None:
      if self.driven < self.bay_to and turn.dist > BAY_LAST:
        return False  # into the turn bay first
      self._end_change(indicator)
    # openpilot is to read NavDesire cleared before the blinker comes on for the turn
    return now >= self.change_hold_until + PARAM_LEAD

  def _enter_bay(self, turn: Turn, bay: float, v: float, now: float):
    """Into the turn bay or slip lane for the turn ahead as it opens, from the lane beside it."""
    if not self._bay_beside(turn, bay) or self.driven < self.bay_to or self.changing is not None:
      return
    if turn.dist - bay > BAY_OPEN or turn.dist < BAY_LAST or v < LANE_CHANGE_SPEED:
      return
    self.bay_to = self.driven + turn.dist + TURN_HOLDS
    self._start_change(turn.side, f"into the bay for the {turn.side} turn in {turn.dist:.0f} m")

  def _sure_wrong(self, lo: int, hi: int) -> bool:
    """Whether the car is surely out of lanes lo-hi, by its position on the route's link (not just between lanes, as
    the model ends lane changes early on GTA's wide lanes), the plugin not disagreeing."""
    return self.lane_frac is not None and (lo - self.lane_frac > SKIP_SURE or self.lane_frac - hi > SKIP_SURE)

  def _bay_beside(self, turn: Turn, bay: float) -> bool:
    """Whether the car is in the lane beside the turn ahead's bay, to change into it."""
    if not BAY_CHANGE or not bay or self.lane is None:
      return False
    return self.lane[0] == (0 if turn.side == "left" else self.lane[1] - 1)

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

  def _change_for(self, m: Turn | Fork | Through, lo: int, hi: int, route: np.ndarray, v: float, now: float) -> float:
    i, n = self.lane
    changes = lo - i if i < lo else i - hi
    fork = isinstance(m, Fork)
    last = max(FORK_LAST_DIST, FORK_LAST * v) if fork else self.tune.lane_change_last
    where = self._point(route, max(m.dist, 0.0))
    key = (fork, round(float(where[0])), round(float(where[1])))
    if self.change_tries.get(key, 0) >= LANE_CHANGE_TRIES:
      return 0.0  # the model won't: there's no lane there
    room = m.dist - last
    need = changes * self.tune.lane_change_time
    if fork and room < 0 and self.changing is None:
      if not self._sure_wrong(lo, hi):
        return 0.0
      self._skip(route, m.dist, f"{m.side} fork")
      self.skip_turns_to = self.driven + m.dist + FORK_TURN
      return 0.0
    cap = max(FORK_MIN_SPEED if fork else self.tune.lane_change_min_speed, room / need) if room < need * v else 0.0
    early = self.tune.lane_change_early + (self.tune.lane_change_fast_early if v > FAST else 0.0)
    if (self.changing is None and room > 0 and room < (need + early) * max(v, LANE_CHANGE_MIN_SPEED)
        and v > LANE_CHANGE_SPEED and now - self.change_t > LANE_CHANGE_GAP and now >= self.cooldown_until and abs(self.yaw) < TURNING):
      if self.change_from == self.lane:
        self.change_tries[key] = self.change_tries.get(key, 0) + 1  # the last change didn't get anywhere
      self.change_from = self.lane
      what = "way straight on" if isinstance(m, Through) else f"{m.side} {'fork' if fork else 'turn'}"
      self._start_change("left" if i > hi else "right", f"from lane {i + 1} of {n} for the {what} in {m.dist:.0f} m ({changes} to go)",
                         turn=isinstance(m, Turn))
    return cap

  def _keep_fork(self, fork: Fork | None, turn: Turn | None, desire: dict[str, float], v: float, now: float):
    """The keep desire towards the route's branch, from a little before the fork to past it, but not against a turn
    the other way soon after; repeated as the model forgets it."""
    want = self.fork_keep if self.driven < self.fork_keep_to and self.turn is None else None
    against = fork is not None and turn is not None and turn.side != fork.side and turn.dist - fork.dist < FORK_TURN
    if fork is not None and fork.keep and not against and self.turn is None and fork.dist < max(FORK_KEEP_DIST, FORK_KEEP * v):
      lo, hi = fork.lanes(self.lane[1]) if self.lane is not None else (0, 0)
      if self.lane is None or lo <= self.lane[0] <= hi:
        want = "keepLeft" if fork.side == "left" and self.one_way else "keepRight" if fork.side == "right" else None
        self.fork_keep_to = self.driven + fork.dist + FORK_KEEP_PAST
    if want != self.fork_keep:
      if DEBUG and want:
        print(f"nav: {want} for the fork in {self.fork_keep_to - self.driven - FORK_KEEP_PAST:.0f} m")
      self.fork_keep, self.fork_keep_t = want, now
    elif want and v > 1.0 and now - self.fork_keep_t > self.tune.repulse_every and self.driven < self.fork_keep_to - FORK_KEEP_PAST:
      self.fork_keep_t, self.keep_gap_until = now, now + KEEP_GAP

  def _keep_straight(self, turn: Turn | None, desire: dict[str, float], v: float, now: float):
    if self.turn is not None or self.changing is not None:
      self.keeping = None
      return
    if self.keeping is None:
      if (self.driven - self.turned_at < KEEP_AFTER_TURN or abs(self.yaw) > TURNING or v < KEEP_MIN_SPEED
          or (self.lane is not None and self.lane[1] < 2)):
        return  # the turn just taken, or one lane: nothing to keep from
      side = max(("left", "right"), key=lambda k: desire.get(k, 0.0))
      if desire.get(side, 0.0) > KEEP_ABOVE and (turn is None or turn.dist > KEEP_CLEAR) and (side == "left" or self.one_way):
        self.keeping, self.keep_from, self.keep_t = "keepRight" if side == "left" else "keepLeft", self.driven, now
        if DEBUG:
          print(f"nav: {self.keeping}, the model expecting a {side} turn ({desire[side]:.2f}) the route doesn't take")
      return
    if self.driven - self.keep_from > KEEP_FOR and max(desire.get("left", 0.0), desire.get("right", 0.0)) < KEEP_UNTIL:
      self.keeping = None
    elif self.fork_keep is None and v > 1.0 and now - self.keep_t > self.tune.repulse_every:
      self.keep_t, self.keep_gap_until = now, now + KEEP_GAP

  def _set_desire(self, now: float):
    """NavDesire: a lane change under way, else a keep desire; the model forgets a keep desire at a stop, as a turn, so
    it goes off briefly as the car pulls away to be seen again."""
    if self.cue and (now > self.cue_until or abs(self.yaw) < TURNING and not self.cue_hold):
      self.cue = None
    keep = self.cue or self.recover or self.fork_keep or self.keeping or ""
    want = "laneChange" if self.changing is not None or now < self.change_hold_until else keep
    if want == "keepLeft" and not self.one_way:
      want = ""  # on a two-way road, towards the oncoming lanes
    if want.startswith("keep"):
      if self.v < 0.3:
        self.keep_stopped = True
      elif self.v > 1.0 and self.keep_stopped:
        self.keep_stopped, self.keep_gap_until = False, now + KEEP_GAP
    if now < self.keep_gap_until and want.startswith("keep"):
      want = ""
    if want.startswith("keep") and self.cue_staged and self.signaled is not None:
      want = "+" + want  # given alongside the turn the blinker asks for (openpilot's NavDesire stacking)
    if want != self.desire:
      self.desire = want
      self.desires.append(want)

  def _keep_right(self, v: float, now: float):
    """Back over from the oncoming lanes, which the model sometimes drifts into on wide roads."""
    lane = self.lane
    if not lane or lane[0] >= 0 or self.driven < self.bay_to:  # a centre turn bay is beyond the inside lane
      self.wrong_side_t = None
      return
    self.wrong_side_t = self.wrong_side_t or now
    if (self.changing is None and self.turn is None and now - self.wrong_side_t > WRONG_SIDE_FOR and v > LANE_CHANGE_SPEED
        and now - self.change_t > LANE_CHANGE_GAP):
      self._start_change("right", f"out of oncoming lane {-lane[0]}")

  def _oncoming_keep(self, plugin_lane: list[int] | None, near_junction: bool, now: float, map_lane: dict | None = None):
    """keepRight while the plugin's own lane reading has held in the oncoming lanes (oncoming_keep): it's right outside
    junctions, but nav has no lane, so no lane change back, where the route's reading disagrees. Also while the map's
    lanes have the car oncoming (oncoming_map), as on a one-way the wrong way, where the plugin reads no lane."""
    plugin = bool(plugin_lane) and plugin_lane[0] < 0 and not self.one_way and not near_junction
    # the bridge leaves the map's reading out in junctions' areas once it has them ("areas"), else near ones
    by_map = (self.tune.oncoming_map and bool(map_lane) and bool(map_lane.get("oncoming"))
              and (bool(map_lane.get("areas")) or not near_junction))
    on = self.tune.oncoming_keep and (plugin or by_map) and self.turn is None and self.driven >= self.bay_to and self.v > 1.0
    if not on:
      self.oncoming_since, self.recover = None, None
      return
    self.oncoming_since = self.oncoming_since or now
    if now - self.oncoming_since < WRONG_SIDE_FOR:
      return
    if self.recover is None:
      self.recover_t = now
      if DEBUG:
        what = f"lane {-plugin_lane[0]}" if plugin else f"({'turn bay' if map_lane.get('bay') else map_lane.get('kind')} by the map)"
        print(f"nav: keepRight out of oncoming {what}")
    elif now - self.recover_t > self.tune.repulse_every:
      self.recover_t, self.keep_gap_until = now, now + KEEP_GAP
    self.recover = "keepRight"

  def _start_change(self, side: str, why: str, turn: bool = False):
    self.changing, self.change_shown, self.change_t, self.change_turn = side, False, self.now, turn
    self.change_side = side
    if DEBUG:
      print(f"nav: lane change {side}, {why}")
    self._set_desire(self.change_t)
    self.change_send_at = self.change_t + PARAM_LEAD

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
      self.requests.append("cancelSignal")
    self.changing, self.change_t, self.change_send_at = None, self.now, 0.0
    self.change_hold_until = self.change_t + PARAM_HOLD
    self._set_desire(self.change_t)

  def _arrive(self, v: float) -> tuple[float, bool]:
    stop = max(math.sqrt(2 * ARRIVE_DECEL * max(0.0, self.route_end - STOP_BEFORE - v * ARRIVE_LAG)), 0.5)
    # or stopped near it, short of the cap's aim (not at lights further back)
    arrived = v < 2.0 and self.route_end < ARRIVED_DIST or v < 0.3 and self.route_end < 2 * ARRIVED_DIST
    if arrived:
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
      self.requests.append("cancelSignal")
    self.turn, self.signaled = None, None

  def blinker_gap(self, now: float) -> bool:
    """Whether openpilot shouldn't see the blinker just now, to repeat the turn request."""
    return not self.refresh and now - self.repeat_t < REPEAT_GAP

  @property
  def signaling(self) -> bool:
    return self.signaled is not None

