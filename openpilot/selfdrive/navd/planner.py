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
FORK_LOOKAHEAD = 3500.0  # m, as far as a freeway move's changes may start (fwy_lane_time)
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
# A turn's junction entry: its stop line (the map's, at the junction's mouth or behind its crossing), else GTA's junction nodes
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
# Freeways (Route.info's roadClasses): exits and forks come up fast across several lanes, so nav moves over for them
# far ahead, by the Tune's fwy_ settings: that many s of travel per lane at the road's limit (the car's speed if
# higher), the changes evenly spread and done by fwy_clear m before the gore; from fwy_settle s past the maneuver
# before (a fork or keep, a turn, an on-ramp's merge), and never into lanes that leave the route on the way
FREEWAY = frozenset({"motorway", "motorway_link", "trunk", "trunk_link"})
MAINLINE = frozenset({"motorway", "trunk"})
FWY_RAMP_TIME = 4.0  # s a lane change takes on the lane line
FWY_NEAR = 20.0  # m: a barrier or a move's frozen spacing (FwyState) this near a move is its own
FWY_MARKS_BEHIND = 1000.0  # m: barriers further behind are forgotten
# by the model's lane, which may lag a lane change: no change after one until it reads the new lane or this long has
# passed, nor one back the other way this soon after
MODEL_CATCHUP = 4.0  # s
MODEL_REVERSE_GAP = 10.0  # s


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
    # line, the model turns in early and can cut into the near, oncoming half of a split junction. Distances are from
    # the car's centre, so 3 m signals about as the front reaches the line
    "left_signal_max_entry": 3.0,
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
    # freeway exits and forks (FREEWAY): s of travel per lane to cross, at the road's limit or the car's speed if
    # higher (0: as on any road); the changes done this far before the gore where there's room; no sooner than this
    # long past the maneuver before; and at least this long apart where the room is short
    "fwy_lane_time": 30.0,  # s
    "fwy_clear": 300.0,  # m
    "fwy_settle": 5.0,  # s
    "fwy_min_gap": LANE_CHANGE_TIME,  # s
    # the lane nav plans from: "map" (the bridge's reading on the route, a stand-in for the truth), "model" (the driving
    # model's current-lane head, modelV2.laneHead, as a real car has, once sure), or "fused" (the model's once sure and
    # counting as many lanes as the map, else the map's)
    "lane_source": os.getenv("NAVD_LANE_SOURCE", "fused"),
    "model_lane_prob": 0.7,  # the model's lane counts once its probability is at least this
    "model_lane_hold": 1.0,  # s, holding the same lane that sure
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
    self.every = False  # every lane carries on along the route's branch, by the map's lanes (aim_fork)

  def lanes(self, n: int) -> tuple[int, int]:
    if self.targets is not None:
      return remap(self.targets, n, self.side == "left")
    if self.every:
      return 0, n - 1
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


class Ends:
  """Where some of our lanes end along the road (a lane drop, or a split the route takes the other branch of, its gore
  beginning there): the lanes that carry on (Route.info's laneMaps), to be in by then rather than cross over at it."""
  def __init__(self, dist: float, mapping):
    self.dist = dist
    self.side = "through"
    kept = [i for i, j in enumerate(mapping) if j is not None]
    self.targets = (min(kept), max(kept), len(mapping))

  def lanes(self, n: int) -> tuple[int, int]:
    return remap(self.targets, n, self.targets[0] == 0)


def lane_ends(maps: list, moves: list) -> list[Ends]:
  """Ends for the lane maps ([(m ahead, map, how many after)]) where some of our lanes end, but none at a turn's
  junction, where nav picks the lane out."""
  turns = [m.dist for m in moves if isinstance(m, Turn)]
  return [Ends(d, mp) for d, mp, _ in maps if d > 0.0 and None in mp and any(j is not None for j in mp)
          and not any(t - MAP_AT <= d <= t + TURN_HOLDS for t in turns)]


class Opening:
  """Lanes beginning on the left of ours along the road (a turn bay opening): the lane the car is in carries on as the
  one that many further right."""
  def __init__(self, dist: float, extra: int):
    self.dist = dist  # m along the route to where the road's lane count rises
    self.extra = extra


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


def next_turn(route: np.ndarray, after: float, turns: list | None = None) -> Turn | None:
  """find_turn(route, after); with the route's own turns (Route.info's turns, found once along its shape: [[m ahead,
  side, exit heading, deg turned], ...]), the first of them past `after`, which hold still as the car drives on, where
  the turns found on the points from the car move about with them."""
  if turns is None:
    return find_turn(route, after)
  for d, side, exit_heading, angle in turns:
    if d > after:
      return Turn(float(d), side, float(exit_heading), float(angle))
  return None


def curve_cap(route: np.ndarray, v: float, tune: Tune | None = None) -> float:
  """The speed now that leaves room to slow for the bends ahead, 0 for none. A bend's curvature is its heading change
  over CURVE_WINDOW m, but no more than over twice or four times the window: GTA's lanes jog sideways through junctions
  and where a freeway splits, turning one way and back, which isn't a bend."""
  return bend_cap(route, v, tune)[0]


def bend_cap(route: np.ndarray, v: float, tune: Tune | None = None) -> tuple[float, str]:
  """curve_cap, with the reason for it: the way the bend that sets it turns ("bendLeft" or "bendRight")."""
  heads, starts = headings(route)
  if len(heads) < 3:
    return NO_CAP
  h = np.unwrap(np.radians(heads))
  mids = starts + np.diff(np.append(starts, starts[-1] + 5.0)) / 2
  s = np.arange(0.0, min(CURVE_LOOKAHEAD, float(mids[-1])), CURVE_STEP)
  if len(s) == 0:
    return NO_CAP
  w = CURVE_WINDOW / 2

  def turned(a, b):
    return np.abs(np.interp(b, mids, h) - np.interp(a, mids, h))
  curvature = np.minimum.reduce([turned(s - k * w, s + k * w) for k in (1, 2, 4)]) / CURVE_WINDOW
  # the turns themselves have their own speed
  t = tune or TUNE
  speed = np.maximum(np.sqrt(t.curve_accel / np.maximum(curvature, 1e-4)), t.turn_speed_square)
  caps = np.sqrt(speed ** 2 + 2 * t.slow_decel * np.maximum(s - w - v * SLOW_LAG, 0.0))
  i = int(np.argmin(caps))
  left = np.interp(s[i] + w, mids, h) > np.interp(s[i] - w, mids, h)  # headings turn counterclockwise to the left
  return float(caps[i]), "bendLeft" if left else "bendRight"


def limit_cap(limits: list, v: float) -> float:
  """The speed now: the limit where the car is, leaving room to slow for lower limits ahead ([[m ahead, m/s or 0
  unknown], ...]); 0 for none. The set speed is the driver's, above or below it."""
  caps = [math.sqrt(limit ** 2 + 2 * LIMIT_DECEL * max(0.0, d - v * SLOW_LAG)) for d, limit in limits if limit > 0]
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


def value_at(changes: list | None, d: float, default=None):
  """A per-stretch value d m on, from [[m ahead, value], ...] where it changes (Route.info's limits, roadClasses)."""
  out = default
  for at, value in changes or []:
    if at > d:
      break
    out = value
  return out


def freeway_move(m, classes: list | None) -> bool:
  """Whether a move ahead (not a turn) is on a freeway, by the road just before it; never without the road classes."""
  return not isinstance(m, Turn) and value_at(classes, m.dist - 1.0, "") in FREEWAY


def barrier(m) -> float | None:
  """Where past a maneuver the route takes (a turn, or a fork or keep it takes a branch of) changes for a freeway move
  after it may start, before fwy_settle; None for the others, which only keep the car out of lanes leaving the route."""
  if isinstance(m, Turn):
    return m.dist + TURN_HOLDS
  return m.dist if isinstance(m, Fork) and m.keep else None


def merges(classes: list | None) -> list[float]:
  """m ahead where the route joins a freeway's mainline from another road, as from an on-ramp (Route.info's
  roadClasses)."""
  out, prev = [], ""
  for d, c in classes or []:
    if d > 0 and c in MAINLINE and prev and prev not in MAINLINE:
      out.append(d)
    prev = c or prev
  return out


def fwy_free(dist: float, barriers: list[float], v: float, t: Tune) -> float:
  """m on from which changes for a freeway move `dist` m on may start: fwy_settle past the last of the barriers (m on,
  behind the car too) before it."""
  before = [b for b in barriers if b < dist - FWY_NEAR]
  return max(0.0, max(before) + t.fwy_settle * max(v, LANE_CHANGE_MIN_SPEED)) if before else 0.0


def fwy_start(dist: float, total: int, pos: float, barriers: list[float], v: float, last: float, t: Tune) -> float:
  """m on from which `total` changes for a freeway move `dist` m on may start, no sooner than `pos`: settled past the
  barriers before it, but no later than leaves room for them, one lane change and gap each, by `last` m before it."""
  latest = dist - last - total * (FWY_RAMP_TIME + LANE_CHANGE_GAP) * max(v, LANE_CHANGE_MIN_SPEED)
  return max(pos, min(fwy_free(dist, barriers, v, t), latest))


def fwy_speed(v: float, limits: list | None, d: float = 0.0) -> float:
  """The speed a freeway move's changes are timed at: the road's limit d m on, or the car's speed if higher."""
  return max(v, value_at(limits, d, 0.0) or 0.0, LANE_CHANGE_MIN_SPEED)


def fwy_schedule(dist: float, n: int, start: float, v: float, vref: float, last: float, t: Tune,
                 slot: float | None = None) -> tuple[float, float]:
  """(end, slot) for n lane changes for a freeway move `dist` m on, the first no sooner than `start` m on: the k-th
  (from 0) starts at end - (n - k) * slot, the last done by end. Each lane gets fwy_lane_time s of travel at vref,
  ending fwy_clear m before the move; where there isn't that room, spread evenly over what there is, at least
  fwy_min_gap apart (as late as `last` m before it, at the latest). `slot`: the spacing frozen as the first began."""
  gap = t.fwy_min_gap * max(v, LANE_CHANGE_MIN_SPEED)
  end = min(dist - last, max(dist - t.fwy_clear, start + n * gap))
  each = min(t.fwy_lane_time * vref, max(end - start, 0.0) / max(n, 1))
  if slot is not None:
    each = slot
  return end, each


class FwyState:
  """What the live planner keeps for freeway changes, for the lane plan to time them as it does: the barriers it has
  seen (m on from the car, negative behind it, as an on-ramp's merge just passed), and the spacing of each move's
  changes ([(x, y, m)] by the move's point), frozen as the first began so that the rest stay evenly spread."""
  def __init__(self, barriers: list | None = None, slots: list | None = None):
    self.barriers = barriers or []
    self.slots = slots or []

  def slot(self, point: np.ndarray) -> float | None:
    near = [s for x, y, s in self.slots if math.hypot(x - point[0], y - point[1]) < FWY_NEAR]
    return near[0] if near else None


MAP_AT = 0.5  # m: a lane map this near a maneuver is its junction's or fork's own, past it (Route.info rounds to 0.1 m)


def carry(mapping, lane: int) -> int:
  """The lane `lane` carries on as through one of Route.info's laneMaps' maps; one that ends, as the nearest that
  carries on (the one it merges into), the left one where they're as near."""
  n = len(mapping)
  if not n:
    return int(lane)
  lane = min(max(int(lane), 0), n - 1)
  if mapping[lane] is not None:
    return mapping[lane]
  for d in range(1, n):
    for k in (lane - d, lane + d):
      if 0 <= k < n and mapping[k] is not None:
        return mapping[k]
  return 0


def carry_back(mapping, lane: int) -> int:
  """The lane before a lane map that carries on as `lane`; for a lane that begins there, the one beside it."""
  came = [i for i, j in enumerate(mapping) if j is not None and j == lane]
  if came:
    return came[0]
  near = [(abs(j - lane), i) for i, j in enumerate(mapping) if j is not None]
  return min(near)[1] if near else min(max(int(lane), 0), max(len(mapping) - 1, 0))


class LaneMaps:
  """Route.info's laneMaps ([[m ahead, the lane each lane before carries on as, how many after]]): where our lanes
  change along the route, to follow one lane across the changes as its number from the left changes."""
  def __init__(self, maps):
    self.maps = sorted(((float(d), list(m), int(n)) for d, m, n in maps or []), key=lambda m: m[0])

  def between(self, s0: float, s1: float) -> list[tuple[float, list, int]]:
    """The maps after s0 and before s1 (not at it: a junction's own are past its maneuver)."""
    return [m for m in self.maps if s0 < m[0] < s1 - MAP_AT]

  def carry(self, lane: int, s0: float, s1: float) -> int:
    for _, m, _ in self.between(s0, s1):
      lane = carry(m, lane)
    return lane

  def lanes(self, lane: int, n: int, s1: float, lo: int, hi: int) -> tuple[int, int] | None:
    """The lanes of the n at the car (lane() numbers them) that carry on into lanes lo-hi s1 m on; where none do (a
    bay opening on the way), the one nearest them. None where the maps don't start from n lanes (the reading is
    from before a change)."""
    first = self.between(-1.0, s1)
    if first and len(first[0][1]) != n:
      return None
    ends = [self.carry(i, -1.0, s1) for i in range(n)]
    into = [i for i, e in enumerate(ends) if lo <= e <= hi]
    if into:
      return min(into), max(into)
    near = min(range(n), key=lambda i: (min(abs(ends[i] - lo), abs(ends[i] - hi)), abs(i - lane)))
    return near, near


def lane_plan(route: np.ndarray, forks: list, lane, lanes_at, v: float, tune: Tune | None = None,
              arrows: list | None = None, drops: list | None = None, opens: list | None = None,
              maps: list | None = None, turns: list | None = None, classes: list | None = None,
              limits: list | None = None, fwy: FwyState | None = None) -> list[tuple[float, float]]:
  """The lanes nav aims for along the whole route, for the map: [(m along, lane from the left)], ramping between each
  two. From the car's lane (out of the oncoming lanes first), it changes only for a turn or fork whose lanes it isn't
  in (by the map's turn arrows where it has them, Route.info's laneArrows) or to go straight on past lanes that only
  turn or end (laneDrops), by where nav's changes for it must have ended, and arrives from a turn in its side's outside
  lane. Where our lanes change along the road (`maps`, Route.info's laneMaps: lanes beginning or ending on either side,
  at a node or across a junction) it keeps to its lane as its number changes, a lane that ends merging into the
  nearest. Without maps, only where lanes begin on the left of ours (`opens`: [m ahead, how many]) is it renumbered.
  lanes_at(m, after) is the lanes the car's way just before (after: past) a point. `turns`: the route's own turns
  (next_turn), else found on `route`. With the maps, freeway moves' changes are timed as the live planner times them
  (fwy_schedule), by the road classes and speed limits ahead (Route.info's roadClasses, limits) and what the planner
  keeps for them (`fwy`, Planner.fwy_state)."""
  t = tune or TUNE
  arrows = parse_arrows(arrows)
  ahead: list[Turn | Fork | Through] = []
  turn = next_turn(route, MIN_AHEAD_MAP, turns)
  while turn is not None:
    ahead.append(turn)
    turn = next_turn(route, turn.dist + TURN_HOLDS, turns)
  ahead += [Fork(d, side, *rest) for d, side, *rest in forks if d > 0]
  for m in ahead:
    aim(m, arrows)
  # with the maps, lanes that end are theirs (lane_ends), at junctions too
  ahead += throughs(route, arrows, ahead, drops if t.lane_drops and maps is None else None)
  if maps is not None:
    maps = LaneMaps(maps)
    for m in ahead:
      aim_fork(m, maps.maps)
    ahead += lane_ends(maps.maps, ahead)
    return _mapped_plan(sorted(ahead, key=lambda m: m.dist), maps, lane, lanes_at, v, t, route, classes, limits, fwy)
  ahead += [Opening(float(d), int(extra)) for d, extra in opens or [] if d > 0]
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
    if isinstance(m, Opening):
      # renumbered where the road widens: a renumbering, not a change, so also while changes are held off (free)
      at = max(m.dist, keys[-1][0])
      keys += [(at, float(cur)), (at, float(cur + m.extra))]
      cur += m.extra
      free = max(free, m.dist)
      continue
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
      # numbered from the branch's own left lane, of the lanes the map has on past it (a bay GTA forks off as its own link
      # is a lane of the road there, which carries on)
      after = lanes_at(m.dist, True)
      new = max(cur - max(n - (after if after > 0 else m.ours), 0), 0) if m.side == "right" else cur
      free = m.dist
    keys += [(m.dist, float(cur)), (m.dist, float(new))]
    cur = new
  return keys


FORK_MAP_NEAR = 15.0  # m past a fork in GTA's roads that the map's lanes may change for it


def aim_fork(m, maps: list | None):
  """A fork's lanes where the map's arrows don't give them, from where the map's lanes change for it (lane maps
  [(m along, map, how many after)] from it to FORK_MAP_NEAR m on): the lanes before that carry on, all of them where
  none end there, but the one beside the other branch on a road of three or more. GTA's own counts at a fork don't
  always add up (a link into the gore counted on top of the road's lanes); without maps, they're all there is."""
  if not isinstance(m, Fork) or m.targets is not None or m.junction or maps is None:
    return
  near = [mp for d, mp, _ in maps if m.dist - MAP_AT <= d <= m.dist + FORK_MAP_NEAR]
  kept = [i for i, j in enumerate(near[0]) if j is not None] if near else []
  if not near or not kept or len(kept) == len(near[0]):
    m.every = True
    return
  if m.other and len(kept) >= 3:
    kept = kept[:-1] if m.side == "left" else kept[1:]
  m.targets = (min(kept), max(kept), len(near[0]))


def bay_opens(maps: list, lane: int) -> float | None:
  """Where lane `lane` (numbered as past the maps) begins on the way, as a turn bay opens: the last of the maps (laneMaps'
  [(m ahead, map, how many after)]) no lane before carries on into; None where it's there all along."""
  for d, m, _ in reversed(maps):
    if lane not in m:
      return d
    lane = carry_back(m, lane)
  return None


def _mapped_plan(ahead: list, maps: LaneMaps, lane, lanes_at, v: float, t: Tune, route: np.ndarray, classes: list | None = None,
                 limits: list | None = None, fwy: FwyState | None = None) -> list[tuple[float, float]]:
  """lane_plan with the lane maps: the lane followed across each change of the road's lanes by a step in its number
  where it changes (as keys at one place, which lane_line puts either side of the node), also part way through a lane
  change."""
  cur = lane[0] if lane else 0
  keys = [(0.0, float(cur))]
  todo = list(maps.maps)  # the maps not yet passed, nearest first
  here = [m for m in todo if m[0] <= 0.0]
  todo = [m for m in todo if m[0] > 0.0]
  free = 0.0  # m along from which the next change may start: past the turn or fork before
  if cur < 0:
    free, cur = -cur * LANE_LINE_CHANGE, 0
    keys.append((free, 0.0))
  elif here and lane and len(here[0][1]) == lane[1]:  # the car's lanes as the route's road numbers them there
    for _, m, _ in here:
      cur = carry(m, cur)
    keys = [(0.0, float(cur))]
  pos = free  # keys are laid to here; cur is the lane there

  def upto(s: float) -> int:
    return next((k for k, m in enumerate(todo) if m[0] >= s - MAP_AT), len(todo))

  def hold(lane: int, s: float, mark: bool = True) -> int:
    """On in the lane to s, through the maps before it, which are passed."""
    for d, m, _ in todo[:upto(s)]:
      new = carry(m, lane)
      if mark:
        keys.extend([(d, float(lane)), (d, float(new))])
      lane = new
    del todo[:upto(s)]
    return lane

  def ramp(x: int, s0: float, y: int, s1: float):
    """From lane x at s0 to y at s1 (each numbered as the road is there), across the changes between."""
    inside = todo[:upto(s1)]
    del todo[:len(inside)]
    xs = [x]
    for _, m, _ in inside:
      xs.append(carry(m, xs[-1]))
    ys = [y]
    for _, m, _ in reversed(inside):
      ys.insert(0, carry_back(m, ys[0]))
    keys.append((s0, float(x)))
    for k, (d, _, _) in enumerate(inside):
      f = min(max((d - s0) / max(s1 - s0, 1e-6), 0.0), 1.0)
      keys.extend([(d, xs[k] + (ys[k] - xs[k]) * f), (d, xs[k + 1] + (ys[k + 1] - xs[k + 1]) * f)])
    keys.append((s1, float(y)))

  def along(lane: int, s0: float, s1: float) -> int:
    """The lane numbered as the road is at s0 (not before pos), carried on to s1 through the maps not yet passed."""
    for _, mp, _ in todo[upto(s0):upto(s1)]:
      lane = carry(mp, lane)
    return lane

  def count_at(s: float) -> int:
    k = upto(s)
    return len(todo[k][1]) if k < len(todo) else lanes_at(s, False)

  # freeway moves (FwyState, fwy_schedule): the barriers before them, and the moves the car keeps to the lanes of, not
  # yet laid past, as their changes may start before those but not into lanes leaving the route there: (m, lo, hi)
  fwy_on = classes is not None and t.fwy_lane_time > 0
  barriers = (list(fwy.barriers) if fwy is not None else []) + merges(classes)
  keeps: list[tuple[float, int, int]] = []

  def fwy_changes(m, at: int, want: int, last: float):
    """Lays the changes from lane `at` to `want` (numbered as at m) for a freeway move, as the live planner times them;
    those it can't fit before `last` m before the move are left to the usual late changes."""
    nonlocal cur, pos
    total, d = abs(want - at), 1 if want > at else -1
    start = fwy_start(m.dist, total, pos, barriers, v, last, t)
    slot = fwy.slot(Planner._point(route, m.dist)) if fwy is not None and len(route) >= 2 else None
    end, each = fwy_schedule(m.dist, total, start, v, fwy_speed(v, limits, m.dist - 1.0), last, t, slot)
    length = max(LANE_LINE_CHANGE, FWY_RAMP_TIME * v)
    done_at = -math.inf
    for k in range(total):
      s = max(end - (total - k) * each, pos, done_at + LANE_CHANGE_GAP * v)
      while s < m.dist - last:
        x = along(cur, pos, s)
        y = x + d
        n_s = count_at(s)
        if n_s > 0 and not 0 <= y < n_s:  # the lane it moves into only begins further on (a lane opening)
          nxt = upto(s)
          s = todo[nxt][0] + 2 * MAP_AT if nxt < len(todo) and todo[nxt][0] < m.dist else math.inf
          continue
        # not into lanes leaving the route before the move: past them first
        bad = [c for c, lo_c, hi_c in keeps if c > s and not lo_c <= along(y, s, c) <= hi_c]
        if not bad:
          break
        s = max(bad) + 2 * MAP_AT  # past its lane maps too
      if s >= m.dist - last:
        return
      e = min(s + length, m.dist - last)
      cur = hold(cur, s)
      y = along(cur, s, e) + d
      ramp(cur, s, y, e)
      cur, pos, done_at = y, e, e

  if pos > 0.0:
    cur = hold(cur, pos, mark=False)  # out of the oncoming lanes meanwhile
  for m in ahead:
    if barrier(m) is not None:
      barriers.append(barrier(m))  # past it, so only for the moves after it
    n = lanes_at(m.dist, False)
    if n <= 0:
      continue
    at = cur
    for _, mp, _ in todo[:upto(m.dist)]:
      at = carry(mp, at)
    at = min(at, n - 1)
    lo, hi = m.lanes(n)
    if isinstance(m, Fork) and (lo, hi) == (0, n - 1):
      continue  # any lane will do: nothing to keep changes for the next move from starting before it
    fwy_move = fwy_on and freeway_move(m, classes)
    if fwy_move and not lo <= at <= hi:
      fwy_changes(m, at, lo if at < lo else hi, max(FORK_LAST_DIST, FORK_LAST * v) if isinstance(m, Fork) else t.lane_change_last)
      at = min(along(cur, pos, m.dist), n - 1)
    elif fwy_move and lo <= at <= hi and barrier(m) is None:
      keeps.append((m.dist, lo, hi))  # laid past later, as changes for a freeway move after it may start before it
      free = max(free, m.dist)
      continue
    if not lo <= at <= hi:
      want = lo if at < lo else hi
      last = max(FORK_LAST_DIST, FORK_LAST * v) if isinstance(m, Fork) else t.lane_change_last
      end = max(min(max(m.dist - last, free), m.dist), pos)
      opens = bay_opens(todo[:upto(m.dist)], want)
      if opens is not None and opens >= end:
        # the lane only begins once the changes would have ended: into it from there, by the move
        start = max(opens, pos)
        end = max(min(start + abs(want - at) * LANE_LINE_CHANGE, m.dist), start)
      elif opens is not None and isinstance(m, Turn) and max(pos, free) <= opens:
        # into a turn bay as it opens, from the lane beside it (the changes to that one first, as late as they may)
        beside = want
        for _, mp, _ in reversed(todo[upto(opens):upto(m.dist)]):
          beside = carry_back(mp, beside)
        here = cur
        for _, mp, _ in todo[:upto(opens)]:
          here = carry(mp, here)
        if beside != here:
          start = max(min(max(opens - abs(beside - here) * LANE_LINE_CHANGE, free), opens), pos)
          cur = hold(cur, start)
          ramp(cur, start, beside, opens)
          cur, pos = beside, opens
        start = max(pos, opens)
        end = max(min(start + LANE_LINE_CHANGE, end), start)
      else:
        start = max(min(max(end - abs(want - at) * LANE_LINE_CHANGE, free), end), pos)
      cur = hold(cur, start)
      to = want
      for _, mp, _ in reversed(todo[upto(end):upto(m.dist)]):
        to = carry_back(mp, to)
      ramp(cur, start, to, end)
      cur, pos = to, end
    cur = min(hold(cur, m.dist), n - 1)
    pos = m.dist
    keeps = [c for c in keeps if c[0] > pos]
    if isinstance(m, Turn):
      # out of the junction in its side's outside lane, of the road out past any of the junction's own links
      out = max(lanes_at(m.dist, True), lanes_at(m.dist + TURN_HOLDS, False))
      new = 0 if m.side == "left" else max(out - 1, 0)
      keys += [(m.dist, float(cur)), (m.dist, float(new))]
      cur = new
      del todo[:upto(m.dist + TURN_HOLDS)]  # the junction's: nav picks the lane out
      free = m.dist + TURN_HOLDS
    else:
      free = m.dist
  hold(cur, math.inf)
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
    self.maps: LaneMaps | None = None  # where the road's lanes change ahead (NavInputs.lane_maps)
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
    self.classes: list | None = None  # the road classes ahead (NavInputs.road_classes)
    self.limits: list | None = None
    self.fwy_marks: list[float] = []  # self.driven at the barriers seen (fwy_free), behind the car too
    self.fwy_slots: list[tuple[float, float, float]] = []  # FwyState.slots
    # the car's lane by each source (lane_source): the map's once steady, the model's once sure and held
    self.map_lane: tuple[int, int] | None = None
    self.model_lane: tuple[int, int] | None = None
    self.model_seen: tuple[tuple[int, int] | None, float] = (None, 0.0)
    self.model_raw: list | None = None  # this step's laneHead reading [index, count, prob]
    self.lane_src = ""  # where self.lane is from: "map" or "model"
    self.lane_log = None  # NAVD_LANE_LOG's file, opened on the first step
    self.lane_disagree: bool | None = None

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
    self._read_lane(inp.truth_lane, now, inp.model_lane)
    self.maps = LaneMaps(inp.lane_maps) if inp.lane_maps is not None else None
    self.classes, self.limits = inp.road_classes, inp.limits
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

    turn = self._next_turn(route, inp.turns) if self.turn is None else None
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
      aim_fork(m, self.maps.maps if self.maps is not None else None)
    drops = inp.lane_drops if t.lane_drops and self.maps is None else None
    ahead = moves + [m for m in throughs(route, arrows, moves, drops) if turn is None or m.dist < turn.dist]
    if self.maps is not None:
      ahead += [m for m in lane_ends(self.maps.maps, moves) if turn is None or m.dist < turn.dist]
    self._mark_barriers(ahead, pos, heading)
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
    caps.append(bend_cap(route, v, t))
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
      turn = next_turn(route, MIN_AHEAD_MAP if inp.route_end is not None else MIN_AHEAD, inp.turns)
      while turn is not None and not any(abs(d - turn.dist) < ENTRY_JUNCTION_BEFORE / 2 for d in inp.junctions or []):
        turn = next_turn(route, turn.dist + TURN_HOLDS, inp.turns)  # a bend of the road, not a turn: slowed for
      if turn is not None:
        along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(route, axis=0).T))))
        route = np.vstack([route[along < turn.dist], self._point(route, turn.dist)])
      caps.append(bend_cap(route, inp.v, self.tune))
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

  def _next_turn(self, route: np.ndarray, turns: list | None = None) -> Turn | None:
    """The first turn ahead not left to the route (of the route's own turns where given, next_turn)."""
    turn = next_turn(route, max(self.min_ahead, self.skip_turns_to - self.driven), turns)
    while turn is not None and self._is_skipped(route, turn.dist):
      turn = next_turn(route, turn.dist + TURN_HOLDS, turns)
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

  def _read_lane(self, lane: list[int] | None, now: float, model: list | None = None):
    """The car's lane by the lane source (Tune's lane_source) from the map's reading, once it has held for a moment
    (it's noisy where roads join; None once it hasn't for a while), and the model's (_read_model_lane)."""
    reading = tuple(lane) if lane else None
    if reading != self.lane_seen[0]:
      self.lane_seen = (reading, now)
    elif now - self.lane_seen[1] >= LANE_STEADY:
      self.map_lane, self.lane_t = reading, now
    if now - self.lane_t > LANE_STALE:
      self.map_lane = None
    self._read_model_lane(model, now)
    self.lane, self.lane_src = self._pick_lane()
    self._log_lane(now)

  def _read_model_lane(self, model: list | None, now: float):
    """The model's lane (laneHead's [index, count, prob]) once it has been sure of it for model_lane_hold s, and kept
    until it has been sure of another, or unsure, that long: a flicker doesn't move it."""
    t = self.tune
    self.model_raw = list(model) if model else None
    sure = None
    if model and 0 <= int(model[0]) < int(model[1]) and float(model[2]) >= t.model_lane_prob:
      sure = (int(model[0]), int(model[1]))
    if sure != self.model_seen[0]:
      self.model_seen = (sure, now)
    if now - self.model_seen[1] >= t.model_lane_hold:
      self.model_lane = sure

  def _pick_lane(self) -> tuple[tuple[int, int] | None, str]:
    """The lane nav plans from, and its source: the map's, or the model's where lane_source allows it ("fused": once
    the map has the car in one of as many lanes of ours as the model counts)."""
    src, mapped, model = self.tune.lane_source, self.map_lane, self.model_lane
    if src == "model":
      return model, "model" if model is not None else ""
    if src == "fused" and model is not None and mapped is not None and mapped[0] >= 0 and mapped[1] == model[1]:
      return model, "model"
    return mapped, "map" if mapped is not None else ""

  def _log_lane(self, now: float):
    """Each step's lane by each source to NAVD_LANE_LOG (JSON lines), to score the model's against the map's; and with
    DEBUG, where they start or stop disagreeing."""
    mapped, model = self.map_lane, self.model_lane
    disagree = None if mapped is None or model is None else tuple(mapped) != tuple(model)
    if DEBUG and disagree is not None and disagree != self.lane_disagree:
      print(f"nav: lane by the model {model} {'disagrees with' if disagree else 'agrees with'} the map's {mapped}")
    self.lane_disagree = disagree
    path = os.getenv("NAVD_LANE_LOG")
    if not path:
      return
    if self.lane_log is None:
      try:
        self.lane_log = open(path, "a", buffering=1)
      except OSError as e:
        print(f"nav: lane log {path}: {e}")
        os.environ.pop("NAVD_LANE_LOG", None)
        return
    raw = self.model_raw
    self.lane_log.write(json.dumps({
      "t": round(now, 3), "map": mapped and list(mapped), "model": model and list(model),
      "raw": raw and [int(raw[0]), int(raw[1]), round(float(raw[2]), 3)], "used": self.lane and list(self.lane),
      "src": self.lane_src, "disagree": disagree, "changing": self.changing}) + "\n")

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

  def turn_points(self, route: np.ndarray, forks: list | None, stops: list | None, junctions: list | None,
                  turns: list | None = None) -> tuple[np.ndarray, np.ndarray | None] | None:
    """For the map overlay: where the turn nav signals next is (of the route's own turns where given), and where its
    signal comes on by distance at the turn's own speed, which nav slows to for it and signals at, so the point holds
    still while the car slows; once signalled, where the car was when it did."""
    if self.turn is not None and self.turn_point is not None:
      return self.turn_point, self.signaled_at
    turn = self._next_turn(route, turns)
    if turn is None:
      return None
    bay = self._bay(self._forks(forks, route, turn), turn)
    entry, _ = junction_entry(turn.dist, stops or [], junctions or [])
    at = self.signal_from(turn, entry, turn.speed(self.tune), bay)
    return self._point(route, turn.dist), self._point(route, max(at, 0.0))

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
      lo, hi = self._lanes_for(turn)
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
    if turn.dist - bay > BAY_OPEN or turn.dist < BAY_LAST or v < LANE_CHANGE_SPEED or not self._clear_to_change(turn.side):
      return
    self.bay_to = self.driven + turn.dist + TURN_HOLDS
    self._start_change(turn.side, f"into the bay for the {turn.side} turn in {turn.dist:.0f} m")

  def _lanes_for(self, m) -> tuple[int, int]:
    """The lanes of the car's road (self.lane's) to be in for a turn, fork or way straight on: its own lanes at its
    junction, followed back to the car across where the road's lanes change (laneMaps); without them, or with a
    lane reading from before a change, its lanes counted from its side."""
    i, n = self.lane
    if self.maps is not None:
      before = self.maps.between(-1.0, m.dist)
      lo, hi = m.lanes(before[-1][2] if before else n)
      found = self.maps.lanes(i, n, m.dist, lo, hi)
      if found is not None:
        return found
    return m.lanes(n)

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
      lo, hi = self._lanes_for(m)
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
    side = "left" if i > hi else "right"
    if self.tune.fwy_lane_time > 0 and freeway_move(m, self.classes):
      first = self.maps.between(-1.0, math.inf) if self.maps is not None else []
      if first and len(first[0][1]) != n:
        return cap  # a lane reading from before the road's lanes changed (just past a split): its targets are a guess
      total = self._lanes_to(m, changes)
      start = fwy_start(m.dist, total, 0.0, [b - self.driven for b in self.fwy_marks], v, last, self.tune)
      slot = self._fwy_slot(where)
      end, each = fwy_schedule(m.dist, total, start, v, fwy_speed(v, self.limits, m.dist - 1.0), last, self.tune, slot)
      due = end - total * each <= 0.0
    else:
      early = self.tune.lane_change_early + (self.tune.lane_change_fast_early if v > FAST else 0.0)
      due, each = room < (need + early) * max(v, LANE_CHANGE_MIN_SPEED), None
    if (self.changing is None and room > 0 and due and v > LANE_CHANGE_SPEED and now - self.change_t > LANE_CHANGE_GAP
        and now >= self.cooldown_until and abs(self.yaw) < TURNING and self._lane_settled(side, now)
        and self._clear_to_change(side)):
      if self.change_from == self.lane:
        self.change_tries[key] = self.change_tries.get(key, 0) + 1  # the last change didn't get anywhere
      self.change_from = self.lane
      if each is not None and self._fwy_slot(where) is None:
        self.fwy_slots.append((float(where[0]), float(where[1]), each))
      what = "way straight on" if isinstance(m, Through) else "lane ending" if isinstance(m, Ends) else f"{m.side} {'fork' if fork else 'turn'}"
      self._start_change(side, f"from lane {i + 1} of {n} for the {what} in {m.dist:.0f} m ({changes} to go)",
                         turn=isinstance(m, Turn))
    return cap

  def _lanes_to(self, m, changes: int) -> int:
    """The lane changes into a move's lanes from the car's, counting those into lanes still to open on the way (where
    _lanes_for gives the lane beside them)."""
    i, n = self.lane
    if self.maps is None:
      return changes
    first = self.maps.between(-1.0, m.dist)
    if first and len(first[0][1]) != n:
      return changes
    lo, hi = m.lanes(first[-1][2] if first else n)
    at = self.maps.carry(i, -1.0, m.dist)
    return max(changes, lo - at if at < lo else at - hi if at > hi else 0)

  def _fwy_slot(self, where: np.ndarray) -> float | None:
    return FwyState(slots=self.fwy_slots).slot(where)

  def _mark_barriers(self, ahead: list, pos: np.ndarray, heading: float):
    """Remembers where the barriers ahead (fwy_free) are, by distance driven, to settle past them once behind the car;
    and forgets the freeway moves' frozen spacing once past them."""
    for b in [barrier(m) for m in ahead] + merges(self.classes):
      if b is None or b <= 0.0:
        continue
      at = self.driven + b
      near = [k for k, mk in enumerate(self.fwy_marks) if abs(mk - at) < FWY_NEAR]
      if near:
        self.fwy_marks[near[0]] = at
      else:
        self.fwy_marks.append(at)
    self.fwy_marks = [mk for mk in self.fwy_marks if mk > self.driven - FWY_MARKS_BEHIND]
    self.fwy_slots = [s for s in self.fwy_slots if not self._passed(np.array(s[:2]), pos, heading)]

  def fwy_state(self) -> FwyState:
    """What the live planner keeps for freeway moves, for lane_plan."""
    return FwyState([mk - self.driven for mk in self.fwy_marks], list(self.fwy_slots))

  def _lane_settled(self, side: str, now: float) -> bool:
    """Whether a lane change for a turn or fork may go by the model's lane: not while it may still be reading the lane
    before the last change, nor back the other way soon after one (a lane flipping across the target's)."""
    if self.lane_src != "model":
      return True
    if self.change_from == self.lane and now - self.change_t < MODEL_CATCHUP:
      return False
    return not (self.change_side is not None and side != self.change_side and now - self.change_t < MODEL_REVERSE_GAP)

  def _clear_to_change(self, side: str) -> bool:
    """Whether nav may start a lane change to `side` now: every change it starts (for a turn or fork, into a bay, out
    of the oncoming lanes) asks here first, and asks again each step while it isn't."""
    return True

  def _keep_fork(self, fork: Fork | None, turn: Turn | None, desire: dict[str, float], v: float, now: float):
    """The keep desire towards the route's branch, from a little before the fork to past it, but not against a turn
    the other way soon after; repeated as the model forgets it."""
    want = self.fork_keep if self.driven < self.fork_keep_to and self.turn is None else None
    against = fork is not None and turn is not None and turn.side != fork.side and turn.dist - fork.dist < FORK_TURN
    if fork is not None and fork.keep and not against and self.turn is None and fork.dist < max(FORK_KEEP_DIST, FORK_KEEP * v):
      lo, hi = self._lanes_for(fork) if self.lane is not None else (0, 0)
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
        and now - self.change_t > LANE_CHANGE_GAP and self._clear_to_change("right")):
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

