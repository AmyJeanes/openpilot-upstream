"""Route input v2's lane slots (gta5test notes/osm_lanes.md, section 5), from a router.Route and the car's place on it:
the road the car is on and the road out of the next maneuver, each as SLOTS lanes counted from the kerb on the traffic
side (the right kerb where traffic drives on the right), each one-hot (oncoming, allowed, target) and all zero for no
lane; and how far on the car must be in a target lane. There is no car lane: the model reads its lane from vision.

A target is set only where the route needs particular lanes: the lanes its move is taken from (the map's turn arrows,
else nav's fallbacks: gta5_nav Turn, Fork and Through .lanes) are a strict subset of the lanes allowed. In LANES_HERE
it shows from where nav would start the lane changes for it from the farthest allowed lane until the junction, and
TARGET_DIST counts down to where those changes must have ended (nav's last change point, moved back to the start of a
stretch where change:lanes bans changing towards the target). LANES_EXIT always shows the lanes the next maneuver goes
into: those the target lanes lead to, nearest first, narrowed to the following move's lanes where its window has begun.

Layout (LANE_SLOTS_LEN = 49 floats): [0:24] LANES_HERE and [24:48] LANES_EXIT, slot k at 3k (oncoming), 3k + 1
(allowed), 3k + 2 (target); [48] TARGET_DIST m / 100, clipped -0.5..3, 0 with no target. The bridge writes it with
[49] the traffic side (1 right, -1 left, 0 no route) to its own shared-memory file for watch_route.py's preview, apart
from the model's route input.
"""
import os

import numpy as np

from openpilot.common.hardware.hw import Paths
from openpilot.tools.sim.bridge.gta5 import gta5_expert, gta5_nav
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import LEFTS, RIGHTS, Section

SLOTS = 8
ONCOMING, ALLOWED, TARGET = 0, 1, 2
LANES_HERE = slice(0, 3 * SLOTS)
LANES_EXIT = slice(3 * SLOTS, 6 * SLOTS)
TARGET_DIST = 6 * SLOTS
LANE_SLOTS_LEN = TARGET_DIST + 1
SIDE = LANE_SLOTS_LEN  # the preview's traffic side
PREVIEW_LEN = SIDE + 1
DIST_UNIT = 100.0  # m
NEAR = 60.0  # m on to look for a road's lanes from a segment without (inside a junction)
EXIT_PAST = 1.0  # m past a maneuver's point where its road out begins
FORK_NEAR = 15.0  # m between a keep maneuver and GTA's fork in the road
BAN_BACK = 600.0  # m before a move to look for a stretch where change:lanes bans changing towards its lanes
# NEXT's choice of the next maneuver (gta5_route_input)
NEXT_PAST, NEXT_AHEAD, NEXT_SOON = 20.0, 500.0, 60.0


def lane_slots_path() -> str:
  return os.path.join(Paths.shm_path(), "route_lanes" + os.environ.get("OPENPILOT_PREFIX", ""))


class Move:
  """A turn, fork or straight on through a junction, and the lanes of the road into it the route needs."""
  def __init__(self, along: float, nav, sec: Section | None, maneuver: bool):
    self.along = along
    self.nav = nav  # gta5_nav Turn, Fork or Through, aimed by the map's arrows
    self.turn = isinstance(nav, gta5_nav.Turn)
    self.fork = isinstance(nav, gta5_nav.Fork)
    self.maneuver = maneuver  # one of NEXT's (gta5_expert.maneuvers), not a Through
    self.n = sec.lanes if sec is not None else 0
    self.targets: set[int] = set()  # of the lanes into its junction, numbered from the left of ours
    self.centre = False  # the target is the centre turn lane
    self.worst = 0  # lane changes into the targets from the farthest allowed lane
    self.free = 0.0  # m along from which its changes may start: past the move before
    self.bans: list[tuple[float, float]] = []  # m along: stretches where changing towards its lanes is banned
    if sec is not None and self.n:
      self.targets, self.centre = targets(sec, nav)
      allowed = allowed_lanes(sec)
      if self.targets:
        self.worst = max((min(abs(i - t) for t in self.targets) for i in allowed - self.targets), default=0)
      elif self.centre:
        self.worst = max((abs(i - centre_side(sec)) for i in allowed), default=0) + 1

  @property
  def need(self) -> bool:
    return bool(self.targets) or self.centre

  def window(self, v: float, after_free: bool = True) -> tuple[float, float]:
    """(start, end) m along: where its target begins to show and where the changes into it must have ended; with
    after_free, not before the move before is past (as nav's lane plan)."""
    t = gta5_nav.TUNE
    free = self.free if after_free else -np.inf
    last = max(gta5_nav.FORK_LAST_DIST, gta5_nav.FORK_LAST * v) if self.fork else t.lane_change_last
    end = self.along - last
    for a, b in self.bans:
      if a <= end < b:
        end = a
    end = min(max(end, free), self.along)
    early = t.lane_change_early + (t.lane_change_fast_early if v > gta5_nav.FAST else 0.0)
    lead = max(self.worst * gta5_nav.LANE_LINE_CHANGE,
               (self.worst * t.lane_change_time + early) * max(v, gta5_nav.LANE_CHANGE_MIN_SPEED))
    return min(max(end - lead, free), end), end


def allowed_lanes(sec: Section) -> set[int]:
  return {i for i, s in enumerate(sec.ours) if s.lane.access}


def centre_side(sec: Section) -> int:
  """Our lane next to the centre turn lane (or the oncoming lanes)."""
  return 0 if sec.first > 0 else sec.lanes - 1


def centre_lane(sec: Section, nav) -> bool:
  """Whether a turn towards the road's middle is taken from its centre turn lane (lanes:both_ways with that arrow)."""
  if not isinstance(nav, gta5_nav.Turn) or sec.lanes == 0 or all(s.heading == 0 for s in sec.spans):
    return False
  middle = sec.spans[:sec.first] if sec.first > 0 else sec.spans[sec.first + sec.lanes:]
  side = 'left' if sec.first > 0 else 'right'  # where the oncoming lanes are
  arrows = LEFTS if side == 'left' else RIGHTS
  return nav.side == side and any(s.heading == 0 and s.lane.turns & arrows for s in middle)


def targets(sec: Section, nav) -> tuple[set[int], bool]:
  """The lanes of ours a move needs on this road ((set, False)), or its centre turn lane (empty, True); (empty, False)
  where any allowed lane will do."""
  if centre_lane(sec, nav):
    return set(), True
  allowed = allowed_lanes(sec)
  lo, hi = nav.lanes(sec.lanes)
  want = set(range(lo, hi + 1)) & allowed
  return (want, False) if want and want != allowed else (set(), False)


def banned(sec: Section, want: set[int]) -> bool:
  """Whether change:lanes keeps the farthest allowed lane from changing into `want`."""
  allowed = allowed_lanes(sec)
  if not want or not allowed:
    return False
  ours = sec.ours
  lo, hi = min(want), max(want)
  right = [i for i in allowed if i > hi]
  left = [i for i in allowed if i < lo]
  return (bool(right) and not all(ours[j].lane.change_left for j in range(hi + 1, max(right) + 1))) or \
         (bool(left) and not all(ours[j].lane.change_right for j in range(min(left), lo)))


def into(move: Move, sec_out: Section) -> set[int]:
  """The lanes of the road out a move's target lanes lead to, nearest in order (all its lanes where any would do)."""
  n_out = sec_out.lanes
  if not n_out:
    return set()
  want = move.targets or set(range(move.n))
  if move.turn:
    k = 1 if move.centre else min(len(want), n_out)
    return set(range(k)) if move.nav.side == 'left' else set(range(n_out - k, n_out))
  shift = move.n - n_out if move.fork and move.nav.side == 'right' else 0  # a right branch's lanes from its own left
  return {min(max(i - shift, 0), n_out - 1) for i in want}


class LaneSlots:
  """Encodes one route (router.Route); the moves along it are worked out once here, so `encode` per frame is cheap."""

  def __init__(self, route, drive_on_right: bool = True):
    self.lanes = route.lanes
    self.drive_on_right = drive_on_right
    self.points = np.asarray(route.points, float)
    self.moves: list[Move] = []
    self.along = np.zeros(0)
    self.next: list[int] = []
    if self.lanes is None:
      return
    arrows = self.lanes.arrows
    gta = any(link is not None for link in route.links)  # GTA's road data, with its forks in the road
    navs = []
    for m in gta5_expert.maneuvers(route):
      side = 'left' if m.desire.endswith('Left') else 'right'
      n_in = self._lanes(m.along, False)
      if m.turn:
        nav = gta5_nav.Turn(m.along, side, m.exit_heading)
      else:
        forks = [f for f in route.forks if abs(f.along - m.along) < FORK_NEAR]
        if forks:
          f = min(forks, key=lambda f: abs(f.along - m.along))
          nav = gta5_nav.Fork(m.along, f.side, f.lanes, f.lanes_in, f.keep, f.other, f.slip)
        elif gta:  # GTA has no fork here: any lane, as nav, unless the map's arrows say
          nav = gta5_nav.Fork(m.along, side, n_in, n_in, False)
        else:  # a map's route alone: by the branch's lane count
          nav = gta5_nav.Fork(m.along, side, self._lanes(m.along, True), n_in, True)
      gta5_nav.aim(nav, arrows)
      navs.append(nav)
    navs += gta5_nav.throughs(self.points, arrows, navs)
    navs.sort(key=lambda nav: nav.dist)
    maneuvers = {id(nav) for nav in navs if not isinstance(nav, gta5_nav.Through)}
    free = 0.0
    for nav in navs:
      move = Move(nav.dist, nav, self._section(nav.dist - EXIT_PAST, False), id(nav) in maneuvers)
      move.free = free
      if move.need and not move.centre:
        move.bans = self._bans(move)
      free = nav.dist + (gta5_nav.TURN_HOLDS if move.turn else 0.0)
      self.moves.append(move)
    self.along = np.array([m.along for m in self.moves])
    self.next = [k for k, m in enumerate(self.moves) if m.maneuver]

  def _section(self, s: float, ahead: bool = True) -> Section | None:
    """The road's lanes at s m along, else the nearest within NEAR m on (ahead) or back."""
    lanes = self.lanes
    k = lanes.segment(s)
    step = 1 if ahead else -1
    while 0 <= k < len(lanes.sections) and (lanes.along[k] - s if ahead else s - lanes.along[k + 1]) <= NEAR:
      sec = lanes.sections[k]
      if sec is not None and sec.lanes:
        return sec
      k += step
    return None

  def _lanes(self, s: float, after: bool) -> int:
    sec = self._section(s + (EXIT_PAST if after else -EXIT_PAST), after)
    return sec.lanes if sec is not None else 0

  def _bans(self, move: Move) -> list[tuple[float, float]]:
    lanes, out = self.lanes, []
    k = lanes.segment(move.along - EXIT_PAST)
    while k >= 0 and lanes.along[k + 1] > move.along - BAN_BACK:
      sec = lanes.sections[k]
      if sec is not None and sec.lanes and banned(sec, targets(sec, move.nav)[0]):
        a, b = float(lanes.along[k]), float(lanes.along[k + 1])
        out = [(a, out[0][1])] + out[1:] if out and abs(out[0][0] - b) < 1e-6 else [(a, b), *out]
      k -= 1
    return out

  def encode(self, s: float, v: float = 0.0) -> np.ndarray:
    """[LANE_SLOTS_LEN] for the car s m along the route at v m/s."""
    out = np.zeros(LANE_SLOTS_LEN, np.float32)
    if self.lanes is None:
      return out
    here = self._section(s)
    k = int(np.searchsorted(self.along, s, side='right')) if self.moves else 0
    if here is not None and k < len(self.moves) and self.moves[k].need:
      move = self.moves[k]
      start, end = move.window(v)
      want, centre = targets(here, move.nav)
      if start <= s and (want or centre):
        self._fill(out[LANES_HERE], here, want, centre)
        out[TARGET_DIST] = np.clip((end - s) / DIST_UNIT, -0.5, 3.0)
        here = None
    if here is not None:
      self._fill(out[LANES_HERE], here, set(), False)
    nxt = self._next(s)
    if nxt is not None:
      move = self.moves[nxt]
      sec_out = self._section(move.along + EXIT_PAST)
      if sec_out is not None:
        want = into(move, sec_out)
        if nxt + 1 < len(self.moves):  # the move after needs a side from where this one comes out
          after = self.moves[nxt + 1]
          if after.need and after.window(v, after_free=False)[0] <= move.along + EXIT_PAST:
            then = targets(sec_out, after.nav)[0]
            if then and want:
              want = (want & then) or {min(then, key=lambda t: min(abs(t - i) for i in want))}
            elif then:
              want = then
        allowed = allowed_lanes(sec_out)
        want &= allowed
        self._fill(out[LANES_EXIT], sec_out, want if want != allowed else set(), False)
    return out

  def _next(self, s: float) -> int | None:
    """The maneuver NEXT shows: until NEXT_PAST m past it, unless the one after is within NEXT_SOON m; within NEXT_AHEAD."""
    ahead = [k for k in self.next if -NEXT_PAST < self.moves[k].along - s < NEXT_AHEAD]
    if len(ahead) > 1 and self.moves[ahead[0]].along < s and self.moves[ahead[1]].along - s < NEXT_SOON:
      ahead = ahead[1:]
    return ahead[0] if ahead else None

  def _fill(self, out: np.ndarray, sec: Section, want: set[int], centre: bool):
    """A road's slots, kerb first: its lanes from the right where traffic drives on the right, else from the left."""
    spans = list(enumerate(sec.spans))
    if self.drive_on_right:
      spans.reverse()
    ours = range(sec.first, sec.first + sec.lanes)
    slots = out.reshape(SLOTS, 3)
    for slot, (i, span) in enumerate(spans[:SLOTS]):
      if i in ours:
        if span.lane.access:
          slots[slot, TARGET if i - sec.first in want else ALLOWED] = 1.0
      elif span.heading == 0:  # a centre turn lane
        slots[slot, TARGET if centre and span.lane.turns else ONCOMING] = 1.0
      else:
        slots[slot, ONCOMING] = 1.0


def preview(slots: LaneSlots | None, s: float, v: float) -> np.ndarray:
  """[PREVIEW_LEN] for watch_route.py: the slots and the traffic side; zero without a route."""
  out = np.zeros(PREVIEW_LEN, np.float32)
  if slots is not None:
    out[:LANE_SLOTS_LEN] = slots.encode(s, v)
    out[SIDE] = 1.0 if slots.drive_on_right else -1.0
  return out


def decode(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, float | None]:
  """(here, exit) as [SLOTS] states (-1 no lane, else ONCOMING / ALLOWED / TARGET) and TARGET_DIST in m (None: no target)."""
  vec = np.asarray(vec, np.float32)

  def states(block):
    b = block.reshape(SLOTS, 3)
    return np.where(b.any(axis=1), b.argmax(axis=1), -1)
  here = states(vec[LANES_HERE])
  dist = float(vec[TARGET_DIST] * DIST_UNIT) if (here == TARGET).any() else None
  return here, states(vec[LANES_EXIT]), dist


def describe(vec: np.ndarray) -> str:
  """One line, kerb first: o oncoming, a allowed, T target, . none."""
  here, out, dist = decode(vec)

  def row(states):
    return ''.join('.oaT'[k + 1] for k in states)
  return f"here {row(here)} out {row(out)}" + (f" by {dist:.0f} m" if dist is not None else "")
