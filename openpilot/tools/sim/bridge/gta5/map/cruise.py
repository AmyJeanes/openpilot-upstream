"""Cruising: a route with no destination, as openpilot drives without one. The map driver drives it as any route; it runs
on along the road the car is on, taking each junction the way a driver carries on without a destination, and is
walked on further well before the car reaches its end (Cruise), so the car never stops for the end of it.

The way on is walked along the map's roads (walk), as Valhalla doesn't say where "straight on" goes: from the car's
way and direction (its lane by the map's lanes, lane_match.py) along each way, and at a node with more than one way on,
through the junction to each way out of it as its turn:lanes call them (osm_lanes.junction_move's moves: the
junction's short straight links crossed), it takes:
1. at the first junction, where the car's lane is turn-only (its turn:lanes arrows have no through), that turn;
2. else the road it's on carried on (the same name, or unnamed the same class) turning no more than SAME_ROAD_TURN;
3. else the way out turning least.
Ways out it never takes: against a one-way, U-turns (more than MOVE_U_TURN), lanes the map closes to cars, moves the
map's turn restrictions forbid (Restrictions), banned ones (below), and while there's another, dead ends and minor
roads (service roads, tracks, GTA's switched-off and off-road links) off a road that isn't one. A dead end is a pocket
(Roaming): road from which no drive without a U-turn gets on to road it could drive round for ever (a cul-de-sac, a
whole estate behind one road in, a car park's loop), found once for the map; where that isn't known, a way out with no
LEADS_ON m of road beyond. Starting in a pocket, it drives out of it if it can, else to its end: the map driver can't
turn the car round (no U-turns; the router keeps them off routes for the same reason), so there it stops.

Valhalla then routes the walk (Router.via): from the car's place on it through a location just past each junction where
the walk had more than one way out to choose from, and to its end. That keeps the route on the walk's roads, as a route
Valhalla would drive (its own turn restrictions and costs), with its maneuvers, speed limits and the rest of a route as
the bridge routes a destination. Where Valhalla's route still leaves the walk, the way out it refused is banned (at a
junction the walk chose at) or the junction is given a location to go through (elsewhere, where Valhalla found a way
round), and the walk taken again from there; one of GTA's shortcut links across a painted separation the walk takes
(Router.crossings) is banned likewise.

The route walked on is swapped in once the car is clear of a junction (JUNCTION_SPAN), so the map driver splices it
into its path on a straight rather than in a turn.

The player's arrow keys steer it (Cruise.prompt): the next junction within PROMPT_SECS s with a way out that side (at a
fork, every way out within FORK_TURN, the one furthest that way) is taken that way, walked on from there and routed
again; not by one of GTA's shortcut links across a painted separation, which the route would ban (at a motorway's
exits, the hop across the gore to a ramp or collector road already split off). One nearer than the car can still turn
at is too late (prompt_need: by its speed and the lane changes into a lane for the turn, lanes_over). That route goes in
at once but for while the car's in a turn, as a city's junctions are too close together to wait for a gap between them.
The other arrow undoes it, walked on from there as before. A route that timed out is asked for again, and timing out
again it's the router being busy, not a turn it can't make; a way out the router wouldn't take stays banned only while
the car's near it (PROMPT_BAN_AWAY). Each prompt is logged (Cruise.take_log): the junctions looked at and why each was
or wasn't taken, the route, and whether the car then took the turn."""
import math
import threading
import time
import urllib.error

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.lane_match import LaneMatcher
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, FORWARD, LEFTS, MOVE_HEADING, MOVE_SKEW, MOVE_SPAN, \
  MOVE_THROUGH, MOVE_U_TURN, RIGHTS, OsmLanes, Section, heading_of, oneway_of

AHEAD = 1500.0  # m of road the route runs on past the car
FIRST_AHEAD = 500.0  # m of it when cruising starts: fewer junctions for Valhalla to route through, so it comes sooner
EXTEND_AT = 700.0  # m left of it when it's walked on further: well before the map driver slows for its end
MAX_THROUGH = 18  # through locations a route takes at most: Valhalla's max_locations (20) less the start and the end
LANE_REACH = 150.0  # m: the car's lane chooses the way out at the first junction within this
SAME_ROAD_TURN = 60.0  # deg: the road the car is on carried on through a junction turns no more than this
PASSED_M = 2000.0  # m of the links a walk was trimmed of that it remembers, so as not to go round a loop again
LEADS_ON = 250.0  # m of road on past a way out for it not to be a dead end
LEADS_ON_STEPS = 2000  # nodes looked at for that at most, beyond which it leads on
MINOR_CLASSES = frozenset({'service', 'track'})
MINOR_TAGS = ('gta:switched_off', 'gta:no_gps', 'gta:offroad')
DIVERGE = 15.0  # m Valhalla's route may stray from the walk's roads
DIVERGE_FROM = 30.0  # m along the route before that counts
JUNCTION_NEAR = 40.0  # m along the walk between where Valhalla leaves it and the junction whose way out it refused
THROUGH_MIN = 3.0  # m: a through location goes on the first link this long from the way out on
THROUGH_AHEAD = 5.0  # m past the car a through location goes at the least
END_BACK = 10.0  # m short of the walk's last node its route ends
JUNCTION_SPAN = (-15.0, 30.0)  # m about a junction (behind, ahead of the car) a route walked on waits to be swapped in
SWAP_BY = 400.0  # m left of the route when the next is swapped in all the same
SWAP_SEARCH = 300.0  # m on from where a route was made the car's looked for on it as it's swapped in
ROAM_MIN = 2000.0  # m of road a set of links a car can drive round for ever has, not to be a pocket
ROUNDS = 6  # routes again after the walk is changed (a ban, a junction to go through), at most
PROMPT_SECS = 12.0  # s of driving ahead an arrow key's turn may be at, and PROMPT_MIN m at the least
PROMPT_MIN = 120.0
# how near a turn may be (prompt_need): m short of its junction's node at the least (its stop line and corner), s of
# driving to route it and splice it in, braking for it (m/s^2, to m/s), and each lane change into a lane for it (s of
# driving, a squeezed one's, and m at the least)
PROMPT_NEAR = 15.0
PROMPT_LAG = 2.0
PROMPT_DECEL = 3.0
PROMPT_TURN_V = 6.0
PROMPT_LC_S = 1.5
PROMPT_LC_MIN = 15.0
FORK_TURN = 45.0  # deg: a junction whose ways out all turn less than this is a fork, where an arrow keeps to its side
PROMPT_TRACE = 250.0  # m ahead the log's trace of a prompt shows junctions to, past those within reach
WATCH_PAST = 50.0  # m past a forced turn's junction the car's watched for on either way out (the log's prompt_outcome)
WATCH_NEAR = 12.0  # m from one of those points the car is on that way out
PROMPT_TRIES = 2  # routes an arrow key's turn is tried for, where the router times out
PROMPT_BAN_AWAY = 300.0  # m from a junction its ways out the router wouldn't take for an arrow key stay banned within


def timed_out(e: BaseException) -> bool:
  """Whether a request to the router failed by timing out (connecting, or waiting for its answer)."""
  return isinstance(e, TimeoutError) or (isinstance(e, urllib.error.URLError) and isinstance(e.reason, TimeoutError))


def turned(h_in: float, h_out: float) -> float:
  """Degrees from heading h_in to h_out (radians counterclockwise from east), left positive."""
  return math.degrees((h_out - h_in + math.pi) % (2 * math.pi) - math.pi)


def bearing(osm: OsmLanes, a: int, b: int) -> float:
  """The bearing (degrees clockwise from north) from node a to node b, as Valhalla takes a location's heading."""
  d = osm.node_xy(b) - osm.node_xy(a)
  return math.degrees(math.atan2(d[0], d[1])) % 360


class Roaming:
  """The links (node, next node) from which a car can drive on for ever without a U-turn, and not round and round a
  car park: those in a strongly connected set of the map's drivable links (linked where a car can go from one on to the
  next: into a node and out by another link, turning no more than MOVE_U_TURN, where the turn restrictions at the node
  allow it) with at least ROAM_MIN m of road. The
  rest are pockets: dead ends, road that leads only into them, and small loops. Found once per map (a few s)."""
  _found: dict[int, 'Roaming'] = {}

  def __init__(self, links: set[tuple[int, int]]):
    self.links = links

  def __contains__(self, link: tuple[int, int]) -> bool:
    return link in self.links

  @classmethod
  def of(cls, osm: OsmLanes) -> 'Roaming':
    if id(osm) not in cls._found:
      cls._found = {id(osm): cls.find(osm)}  # one map at a time
    return cls._found[id(osm)]

  @classmethod
  def find(cls, osm: OsmLanes, restrictions: 'Restrictions | None' = None) -> 'Roaming':
    walker = Walker(osm)
    rules = restrictions if restrictions is not None else Restrictions.of(osm)
    ids: dict[tuple[int, int], int] = {}
    links: list[tuple[int, int]] = []
    for a, b in osm.pairs:
      if walker.drivable(a, b):
        ids[(a, b)] = len(links)
        links.append((a, b))
    heading = [walker.heading(a, b) for a, b in links]
    length = [float(np.hypot(*(osm.node_xy(b) - osm.node_xy(a)))) for a, b in links]
    out: list[list[int]] = [[] for _ in links]
    for k, (a, b) in enumerate(links):
      for m in osm.links.get(b, ()):
        j = ids.get((b, m))
        if j is not None and m != a and abs(turned(heading[k], heading[j])) <= MOVE_U_TURN and \
           rules.allows([osm.pairs[(a, b)][0]], b, osm.pairs[(b, m)][0]):
          out[k].append(j)
    found = set()
    for comp in strongly_connected(out):
      if sum(length[k] for k in comp) >= ROAM_MIN:
        found.update(links[k] for k in comp)
    return cls(found)


def strongly_connected(out: list[list[int]]) -> list[list[int]]:
  """A directed graph's strongly connected sets (out: each vertex's successors), by Tarjan's algorithm without
  recursion."""
  index, low, on, stack, comps = [-1] * len(out), [0] * len(out), [False] * len(out), [], []
  n = 0
  for root in range(len(out)):
    if index[root] >= 0:
      continue
    work = [(root, 0)]
    while work:
      v, i = work.pop()
      if i == 0:
        index[v] = low[v] = n
        n += 1
        stack.append(v)
        on[v] = True
      for k in range(i, len(out[v])):
        w = out[v][k]
        if index[w] < 0:
          work.append((v, k + 1))
          work.append((w, 0))
          break
        if on[w]:
          low[v] = min(low[v], index[w])
      else:
        if low[v] == index[v]:
          comp = []
          while True:
            w = stack.pop()
            on[w] = False
            comp.append(w)
            if w == v:
              break
          comps.append(comp)
        if work:
          u = work[-1][0]
          low[u] = min(low[u], low[v])
  return comps


class Restrictions:
  """The map's turn restrictions (type=restriction relations, no_* ones: the generator writes no only_* ones), for the
  walk to keep to as Valhalla's routes do: by the way into a via node, the ways out it may not take; by the way out of
  a run of via ways, the runs of ways (the from way and the via ways) it may not be taken after."""
  _loaded: dict[str, 'Restrictions'] = {}

  def __init__(self, relations: dict):
    self.at_node: dict[tuple[int, int], set[int]] = {}  # (from way, via node) -> to ways
    self.after: dict[int, list[tuple[int, ...]]] = {}  # to way -> (from way, via ways...)
    for tags, members in relations.values():
      if tags.get('type') != 'restriction' or not tags.get('restriction', '').startswith('no_'):
        continue
      wf = [r for t, r, role in members if t == 'w' and role == 'from']
      wt = [r for t, r, role in members if t == 'w' and role == 'to']
      vn = [r for t, r, role in members if t == 'n' and role == 'via']
      vw = [r for t, r, role in members if t == 'w' and role == 'via']
      if len(wf) != 1 or len(wt) != 1:
        continue
      if vn and not vw:
        self.at_node.setdefault((wf[0], vn[0]), set()).add(wt[0])
      elif vw and not vn:
        self.after.setdefault(wt[0], []).append((wf[0], *vw))

  @classmethod
  def of(cls, osm: OsmLanes) -> 'Restrictions':
    """The restrictions of the map file osm was loaded from (read once per file), none for a map without one."""
    if osm.path is None:
      return cls({})
    if osm.path not in cls._loaded:
      from openpilot.tools.sim.bridge.gta5.map import osm_pbf
      cls._loaded[osm.path] = cls(osm_pbf.read(osm.path, relations=('restriction',)).relations)
    return cls._loaded[osm.path]

  def allows(self, ways: list[int], node: int, to: int) -> bool:
    """Whether a car on the last of `ways` (the ways it came along, in order, each once) may turn at `node` on to way
    `to`."""
    if not ways or ways[-1] == to:
      return True
    if to in self.at_node.get((ways[-1], node), ()):
      return False
    return not any(tuple(ways[-len(run):]) == run for run in self.after.get(to, ()))


class Exit:
  """A way out of a junction: the nodes from the junction's node to its first node past it, how far it turns, and
  whether there's a way on from its end the turn restrictions allow, by the ways taken to it (some forbid the moves out
  of a link only for a car that came along a certain way, as through a median)."""
  def __init__(self, turn: float, path: list[int], on: bool = True):
    self.turn = turn
    self.path = path
    self.on = on

  @property
  def edge(self) -> tuple[int, int]:
    return self.path[-2], self.path[-1]


class Walker:
  """The roads a walk may take on a map, and which ways out of a junction it takes (see the module's doc)."""
  def __init__(self, osm: OsmLanes, banned: set | None = None, restrictions: Restrictions | None = None,
               roaming: Roaming | None = None):
    self.osm = osm
    self.banned: set[tuple[int, int]] = banned if banned is not None else set()  # links (node, next node) never taken
    self.restrictions = restrictions if restrictions is not None else Restrictions({})
    self.roaming = roaming  # the links off which a car can drive on for ever, where known
    self._open: dict[tuple[int, bool], bool] = {}

  def way(self, a: int, b: int) -> int:
    return self.osm.pairs[(a, b)][0]

  def drivable(self, a: int, b: int) -> bool:
    """Whether a car may drive from node a to node b: not against a one-way, with a lane open to cars that way."""
    pair = self.osm.pairs.get((a, b))
    if pair is None:
      return False
    w, fwd = pair
    one = oneway_of(self.osm.ways[w][0])
    if one != 0 and (one == 1) != fwd:
      return False
    if (w, fwd) not in self._open:
      sec = Section.of(self.osm.lanes(w), FORWARD if fwd else BACKWARD)
      self._open[(w, fwd)] = not sec.ours or any(s.lane.access for s in sec.ours)
    return self._open[(w, fwd)]

  def heading(self, a: int, b: int) -> float:
    return float(heading_of(self.osm.node_xy(b) - self.osm.node_xy(a)))

  def minor(self, w: int) -> bool:
    tags = self.osm.ways[w][0]
    return tags.get('highway') in MINOR_CLASSES or any(tags.get(t) == 'yes' for t in MINOR_TAGS)

  def name(self, w: int) -> str:
    return self.osm.ways[w][0].get('name', '')

  def road_class(self, w: int) -> str:
    return self.osm.ways[w][0].get('highway', '').removesuffix('_link')

  def exits(self, prev: int, j: int, h_in: float, ways: list[int]) -> list[Exit]:
    """The ways out of the junction at node j, arrived at from prev heading h_in along `ways` (each once, the last
    prev's to j): across its short straight links (within MOVE_THROUGH of h_in, MOVE_SPAN m from j), each first node
    past it, as junction_move finds them; none the map's turn restrictions forbid."""
    out: list[Exit] = []
    seen = {prev, j}
    stack: list[tuple[int, float, list[int], list[int]]] = [(j, 0.0, [j], ways)]
    while stack:
      n, dist, path, came = stack.pop()
      for m in self.osm.links.get(n, ()):
        if m in seen or not self.drivable(n, m):
          continue
        w = self.way(n, m)
        if not self.restrictions.allows(came, n, w):
          continue
        seen.add(m)
        on = came if came and came[-1] == w else came + [w]
        d = self.osm.node_xy(m) - self.osm.node_xy(n)
        t = turned(h_in, float(heading_of(d)))
        length = float(np.hypot(*d))
        if abs(t) <= MOVE_THROUGH and dist + length <= MOVE_SPAN and \
           any(q != n and self.drivable(m, q) for q in self.osm.links.get(m, ())):
          stack.append((m, dist + length, path + [m], on))
        elif abs(t) <= MOVE_U_TURN:
          out.append(Exit(t, path + [m], any(q != n and self.drivable(m, q) and self.restrictions.allows(on, m, self.way(m, q))
                                             for q in self.osm.links.get(m, ()))))
    return out

  def leads_on(self, a: int, b: int, reach: float = LEADS_ON) -> bool:
    """Whether there's `reach` m of road on from the link a -> b without turning back: not a dead end."""
    stack = [(a, b, float(np.hypot(*(self.osm.node_xy(b) - self.osm.node_xy(a)))))]
    seen: set[tuple[int, int]] = set()
    while stack:
      if len(seen) > LEADS_ON_STEPS:
        return True
      p, n, dist = stack.pop()
      if dist >= reach:
        return True
      h = self.heading(p, n)
      for m in self.osm.links.get(n, ()):
        if m == p or (n, m) in seen or not self.drivable(n, m) or abs(turned(h, self.heading(n, m))) > MOVE_U_TURN:
          continue
        seen.add((n, m))
        stack.append((n, m, dist + float(np.hypot(*(self.osm.node_xy(m) - self.osm.node_xy(n))))))
    return False

  @staticmethod
  def move(e: Exit, exits: list[Exit]) -> str:
    """'left', 'through' or 'right', as the junction's turn:lanes call a way out (junction_move)."""
    nearest = min(abs(x.turn) for x in exits)
    if abs(e.turn) <= MOVE_THROUGH or (abs(e.turn) == nearest and nearest <= MOVE_SKEW):
      return 'through'
    return 'left' if e.turn > 0 else 'right'

  def candidates(self, approach: int, exits: list[Exit]) -> list[Exit]:
    """The ways out it may take: none across a banned link (inside the junction too), and while there's another, not one
    the turn restrictions leave no way on from, nor into a pocket (else a dead end), nor (off a road that isn't one) a
    minor road."""
    cands = [e for e in exits if not any(link in self.banned for link in zip(e.path, e.path[1:], strict=False))]
    cands = [e for e in cands if e.on] or cands
    on = [e for e in cands if (e.edge in self.roaming if self.roaming is not None else self.leads_on(*e.edge))]
    cands = on or cands
    if not self.minor(approach):
      cands = [e for e in cands if not self.minor(self.way(*e.edge))] or cands
    return cands

  def choose(self, road: str, approach: int, exits: list[Exit], cands: list[Exit], turn: str | None = None) -> Exit:
    """The way out it takes among cands: the turn `turn` ('left' or 'right') where given and there is one, else the
    road (`road`: its name, '' for none) carried on, else the one turning least."""
    if turn is not None:
      ours = [e for e in cands if self.move(e, exits) == turn]
      if ours:
        return min(ours, key=lambda e: abs(e.turn))

    # off a link (a ramp, GTA's hop across to a road alongside) its class isn't the road it leads onto: the next link
    # back would pass for the road carried on
    link = self.osm.ways[approach][0].get('highway', '').endswith('_link')

    def same(e: Exit) -> bool:
      w = self.way(*e.edge)
      return self.name(w) == road if road else not link and not self.name(w) and self.road_class(w) == self.road_class(approach)
    carried = [e for e in cands if abs(e.turn) <= SAME_ROAD_TURN and same(e)]
    return min(carried or cands, key=lambda e: abs(e.turn))

  def side_exit(self, side: str, exits: list[Exit], cands: list[Exit]) -> tuple[Exit | None, bool]:
    """The way out an arrow key (`side`) asks for among cands, and whether it's a fork's: at a fork the one furthest
    that way, else the turn that way turning least; None for none."""
    if len(cands) > 1 and all(abs(e.turn) < FORK_TURN for e in cands):
      return (max if side == 'left' else min)(cands, key=lambda e: e.turn), True
    ours = [e for e in cands if self.move(e, exits) == side]
    return (min(ours, key=lambda e: abs(e.turn)) if ours else None), False


class Walk:
  """A walk along the map's roads: its nodes, m along it to each, and the junctions it chose a way out of among others
  (`choices`: (the index in `nodes` of the junction's node, of the first node past the junction)), which Valhalla's
  route goes through."""
  def __init__(self, walker: Walker, prev: int, node: int, turn: str | None = None, car: float = 0.0):
    self.walker, self.osm = walker, walker.osm
    self.nodes = [prev, node]
    self.along = [0.0, self._length(prev, node)]
    self.ways = [walker.way(prev, node)]  # the ways it went along, each once in turn (for the turn restrictions)
    self.choices: list[tuple[int, int]] = []
    self.turn = turn  # the car's turn-only lane's turn, for the first junction
    self.car = car  # m along it to the car, when it was walked from there
    self.ended = False  # no way on: the road ends
    self.passed: list[tuple[tuple[int, int], float]] = []  # links trimmed off its start (the latest PASSED_M m), and m each

  def _length(self, a: int, b: int) -> float:
    return float(np.hypot(*(self.osm.node_xy(b) - self.osm.node_xy(a))))

  def _add(self, n: int):
    w = self.walker.way(self.nodes[-1], n)
    if w != self.ways[-1]:
      self.ways.append(w)
    self.along.append(self.along[-1] + self._length(self.nodes[-1], n))
    self.nodes.append(n)

  def points(self, k0: int = 0) -> np.ndarray:
    return np.array([self.osm.node_xy(n) for n in self.nodes[k0:]])

  def heading_in(self) -> float:
    """The walk's heading into its last node, over its last MOVE_HEADING m."""
    j, s = len(self.nodes) - 1, self.along[-1] - MOVE_HEADING
    k = j - 1
    while k > 0 and self.along[k] > s:
      k -= 1
    return self.walker.heading(self.nodes[k], self.nodes[j])

  def road(self) -> str:
    """The name of the road it's on: of the last named way within the last 200 m."""
    for k in range(len(self.nodes) - 1, 0, -1):
      if self.along[-1] - self.along[k] > 200.0:
        break
      name = self.walker.name(self.walker.way(self.nodes[k - 1], self.nodes[k]))
      if name:
        return name
    return ''

  def extend(self, until: float, max_choices: int, start: float = 0.0):
    """Walks on until `until` m along it, or until it has chosen at max_choices junctions past `start` m, or the road
    ends."""
    w = self.walker
    while not self.ended and self.along[-1] < until and sum(self.along[k] > start for _, k in self.choices) < max_choices:
      prev, j = self.nodes[-2], self.nodes[-1]
      outs = [m for m in self.osm.links.get(j, ()) if m != prev and w.drivable(j, m) and (j, m) not in w.banned and
              w.restrictions.allows(self.ways, j, w.way(j, m))]
      h_in = self.heading_in()
      if len(outs) == 1 and abs(turned(h_in, w.heading(j, outs[0]))) <= MOVE_U_TURN:
        self._add(outs[0])
        continue
      exits = w.exits(prev, j, h_in, self.ways) if outs else []
      approach = w.way(prev, j)
      cands = w.candidates(approach, exits)
      if not cands:
        self.ended = True
        break
      # not round a loop it has already been round (a service road about a building), while there's a way off it
      cands = [e for e in cands if not any(self.walked(link) for link in zip(e.path, e.path[1:], strict=False))] or cands
      turn = self.turn if self.along[-1] - self.car <= LANE_REACH else None
      self.turn = None  # the car's lane is for the first junction only
      e = w.choose(self.road(), approach, exits, cands, turn)
      at = len(self.nodes) - 1
      for n in e.path[1:]:
        self._add(n)
      if len(cands) > 1:
        self.choices.append((at, len(self.nodes) - 1))

  def force(self, k: int, e: Exit):
    """Back to its junction at node k, taking way out e there."""
    self.cut(k)
    for n in e.path[1:]:
      self._add(n)
    self.choices.append((k, len(self.nodes) - 1))

  def has(self, link: tuple[int, int]) -> bool:
    return any(self.nodes[k] == link[0] and self.nodes[k + 1] == link[1] for k in range(len(self.nodes) - 1))

  def walked(self, link: tuple[int, int]) -> bool:
    """Whether it takes the link, or was trimmed of it lately (passed)."""
    return self.has(link) or any(link == p for p, _ in self.passed)

  def junctions(self, k0: int, upto: float) -> list[tuple[int, list[Exit], list[Exit], Exit | None]]:
    """The junctions it passes from node k0 (> 0) to `upto` m along: (the index in nodes of the junction's node, its
    ways out, those it may take, the one it takes)."""
    w, out, k = self.walker, [], max(k0, 1)
    while k < len(self.nodes) - 1 and self.along[k] <= upto:
      prev, j = self.nodes[k - 1], self.nodes[k]
      if sum(m != prev and w.drivable(j, m) for m in self.osm.links.get(j, ())) < 2:
        k += 1
        continue
      head = _copy(self, w)
      head.cut(k)
      exits = w.exits(prev, j, head.heading_in(), head.ways)
      taken = next((e for e in exits if self.nodes[k:k + len(e.path)] == e.path), None)
      out.append((k, exits, w.candidates(w.way(prev, j), exits), taken))
      k += len(taken.path) - 1 if taken is not None else 1
    return out

  def cut(self, k: int):
    """Back to node k (its first link kept), the walk's choices past it undone."""
    k = max(k, 1)
    self.nodes, self.along = self.nodes[:k + 1], self.along[:k + 1]
    self.choices = [(j, c) for j, c in self.choices if c <= k]
    self.ended = False
    self.ways = []
    for a, b in zip(self.nodes, self.nodes[1:], strict=False):
      w = self.walker.way(a, b)
      if not self.ways or self.ways[-1] != w:
        self.ways.append(w)

  def trim(self, k: int):
    """From node k on: the nodes before it dropped."""
    if k <= 0:
      return
    k = min(k, len(self.nodes) - 2)
    self.passed += [((a, b), self.along[i + 1] - self.along[i]) for i, (a, b) in enumerate(zip(self.nodes[:k], self.nodes[1:k + 1]))]
    total = sum(m for _, m in self.passed)
    while self.passed and total > PASSED_M:
      total -= self.passed.pop(0)[1]
    base = self.along[k]
    self.nodes, self.along = self.nodes[k:], [a - base for a in self.along[k:]]
    self.choices = [(max(j - k, 0), c - k) for j, c in self.choices if c - k >= 1]
    self.turn = None

  def cap(self, after: float, n: int):
    """Ends it short of its junction past `after` m where it would choose more than n times."""
    ahead = [(j, c) for j, c in self.choices if self.along[c] > after]
    if len(ahead) > n:
      self.cut(ahead[n][0])

  def locate(self, pos: np.ndarray) -> tuple[int, float]:
    """The segment of the walk nearest pos, and m along the walk to pos there."""
    pts = self.points()
    a, ab = pts[:-1], np.diff(pts, axis=0)
    t = np.clip(np.einsum('ij,ij->i', pos - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-9), 0.0, 1.0)
    d = np.hypot(*(a + ab * t[:, None] - pos).T)
    k = int(np.argmin(d))
    return k, float(self.along[k] + t[k] * np.hypot(*ab[k]))

  def through(self, after: float) -> list[tuple[np.ndarray, float]]:
    """Where Valhalla's route goes through the walk's choices: a point (game metres) on the middle of the first link at
    least THROUGH_MIN m long from each way out on, and its bearing; those past `after` m along it."""
    out = []
    for _, c in self.choices:
      k = c
      while k + 1 < len(self.nodes) and self.along[k] - self.along[k - 1] < THROUGH_MIN:
        k += 1
      if (self.along[k - 1] + self.along[k]) / 2 > after:
        a, b = self.nodes[k - 1], self.nodes[k]
        out.append(((self.osm.node_xy(a) + self.osm.node_xy(b)) / 2, bearing(self.osm, a, b)))
    return out

  def point(self, s: float) -> tuple[np.ndarray, float]:
    """The point `s` m along it (game metres), and its bearing there: where its route starts, the car's place on it
    (its start link is the car's, by the lane matcher, at the car's height: not a road over or under it)."""
    k = min(max(int(np.searchsorted(self.along, s, side='right')) - 1, 0), len(self.nodes) - 2)
    a, b = self.nodes[k], self.nodes[k + 1]
    p, q = self.osm.node_xy(a), self.osm.node_xy(b)
    t = min(max((s - self.along[k]) / max(self.along[k + 1] - self.along[k], 1e-6), 0.0), 1.0)
    return p + (q - p) * t, bearing(self.osm, a, b)

  def end(self) -> tuple[np.ndarray, float]:
    """Its end, as Valhalla's route ends it: END_BACK m short of its last node (on its last link, not at the node, where
    Valhalla could take another road meeting there; half way along a shorter one), and its bearing."""
    a, b = self.nodes[-2], self.nodes[-1]
    p, q = self.osm.node_xy(a), self.osm.node_xy(b)
    length = max(float(np.hypot(*(q - p))), 1e-6)
    return q + (p - q) * min(END_BACK, length / 2) / length, bearing(self.osm, a, b)


def start_link(osm: OsmLanes, walker: Walker, pos: np.ndarray, heading: float, z: float | None,
               matcher: LaneMatcher) -> tuple[int, int, str | None] | None:
  """The link (node, next node) the car is driving along, from its position and game heading (degrees counterclockwise
  from north), at its height where known; and the turn its lane only turns, by the map's turn:lanes (None for none, or
  for through). None off the map's roads, or driving against a one-way."""
  reading = matcher.match(float(pos[0]), float(pos[1]), math.radians(heading + 90.0), z)
  if reading is None:
    return None
  refs = osm.ways[reading.way][1]
  pts = osm.way_points(reading.way)
  a, ab = pts[:-1], np.diff(pts, axis=0)
  t = np.clip(np.einsum('ij,ij->i', pos - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-9), 0.0, 1.0)
  i = int(np.argmin(np.hypot(*(a + ab * t[:, None] - pos).T)))
  u, v = (refs[i], refs[i + 1]) if reading.direction == FORWARD else (refs[i + 1], refs[i])
  if not walker.drivable(u, v):
    return None
  # the car's lane's arrows where they're painted, at its way's end (the car's offset there, the way's line the same)
  sec = osm.end_section(reading.way, refs[-1] if reading.direction == FORWARD else refs[0], reading.direction)
  turn = None
  if sec.lanes and reading.lane >= 0:
    arrows = sec.turns[min(sec.lane(reading.right), sec.lanes - 1)]
    if arrows and 'through' not in arrows:
      lefts, rights = bool(arrows & LEFTS), bool(arrows & RIGHTS)
      turn = 'left' if lefts and not rights else 'right' if rights and not lefts else None
  return u, v, turn


def lanes_over(walk: Walk, k: int, side: str, lane: list[int] | None) -> int:
  """Lanes the car crosses from its lane (lane: [i from the left, of n]; one where it isn't known) into one that turns
  `side` at the walk's junction k: by the turn:lanes arrows on the road into it, else its outermost lane that side; lanes
  it has more than the car's road taken as opened on that side (a turn pocket)."""
  if lane is None or lane[0] < 0:
    return 1
  osm = walk.osm
  a, b = walk.nodes[k - 1], walk.nodes[k]
  w, fwd = osm.pairs[(a, b)]
  sec = osm.end_section(w, b, FORWARD if fwd else BACKWARD)
  if not sec.lanes:
    return 0
  arrows = LEFTS if side == 'left' else RIGHTS
  ok = [i for i, t in enumerate(sec.turns) if t & arrows] or [0 if side == 'left' else sec.lanes - 1]
  i, n = lane
  here = min(max(i + (sec.lanes - n) if side == 'left' else i, 0), sec.lanes - 1)
  return min(abs(here - j) for j in ok)


def prompt_need(v: float, lanes: int, fork: bool) -> float:
  """m short of a junction's node an arrow key's turn is still made at v m/s with `lanes` lane changes into it: its
  route made and spliced in, braking to PROMPT_TURN_V for a turn (not a fork), and each lane change."""
  brake = 0.0 if fork else max(v * v - PROMPT_TURN_V ** 2, 0.0) / (2 * PROMPT_DECEL)
  return PROMPT_NEAR + PROMPT_LAG * v + brake + lanes * max(PROMPT_LC_MIN, PROMPT_LC_S * v)


def strays(points: np.ndarray, line: np.ndarray, start: float = DIVERGE_FROM, far: float = DIVERGE) -> int | None:
  """The first of a route's points past `start` m along it further than `far` m from a polyline; None for none."""
  along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))))
  a, ab = line[:-1], np.diff(line, axis=0)
  for k in np.nonzero(along > start)[0]:
    p = points[k]
    t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-9), 0.0, 1.0)
    if float(np.hypot(*(a + ab * t[:, None] - p).T).min()) > far:
      return int(k)
  return None


class Cruise:
  """Keeps a cruise route from the car: walked when it starts, walked on and routed again from the car once there's less
  than EXTEND_AT m of it left (the roads it chose kept), and walked afresh where the car leaves it. Routing runs on its
  own thread, so update() never waits on the server."""
  OFF_ROUTE = 15.0  # m from the route
  OFF_FOR = 1.5  # s
  RETRY = 3.0  # s between attempts

  def __init__(self, router, matcher=None):
    """router: a Router with the map's roads; matcher() the brain's LaneMatcher where it has one (else one is made)."""
    self.router = router
    self.matcher = matcher or (lambda: None)
    self._own_matcher: LaneMatcher | None = None
    self.route = None
    self.walk: Walk | None = None
    self.pending: tuple | None = None  # (route, walk) walked on, waiting for the car to be clear of a junction
    self.banned: set[tuple[int, int]] = set()
    self.prompt_bans: dict[tuple[int, int], np.ndarray] = {}  # of those, ways out refused for an arrow key: where
    self.off_since: float | None = None
    self.next_try = 0.0
    self.busy = 0  # routes being made on their threads
    self.epoch = 0  # reset() and an arrow key move it on: a route being made before is dropped
    self.lock = threading.Lock()
    self.forced: dict | None = None  # the arrow key's turn the walk takes: side, node (the junction's), link, road
    self.notes: list[tuple[str, str, str]] = []  # (kind, side, road) of an arrow key's turn that couldn't be routed
    self.log: list[dict] = []  # what the arrow keys' prompts did, in detail, for the brain's log
    self.watching: dict | None = None  # a forced turn being watched for whether the car takes it (_watch)
    self.prompted = None  # the route an arrow key's prompt made, until the car's first located on it

  @property
  def osm(self) -> OsmLanes | None:
    return self.router.osm if self.router.osm is not None else self.router.roads

  def reset(self):
    """No cruising: the route and its walk go (a destination's route drives instead, or the map driver is off)."""
    with self.lock:
      if self.route is None and self.walk is None and self.off_since is None and self.pending is None:
        return
      self.route, self.walk, self.off_since, self.next_try, self.pending = None, None, None, 0.0, None
      self.forced, self.notes, self.watching, self.prompted = None, [], None, None
      self.epoch += 1

  def update(self, pos: np.ndarray, bearing: float, now: float, z: float | None = None):
    """The cruise route, the car located on it (pos and bearing, clockwise from north, the car's); None until made."""
    if self.osm is None:
      return None
    with self.lock:
      self._unban(pos)
      route, walk, pending = self.route, self.walk, self.pending
      swapped = False
      if pending is not None and pending[0] is self.prompted and self._too_late(pos):
        pending = self.pending = self.prompted = None
      if route is not None and pending is not None and (not self._held(route, pending[0]) or route.length - route.at < SWAP_BY):
        route, walk = self.route, self.walk = pending
        self.pending, swapped = None, True
    if route is not None:
      # a route held a while was located where the car was when it was made: found again further on
      off = route.locate(pos, z, -bearing, search=SWAP_SEARCH) if swapped else route.locate(pos, z, -bearing)
      with self.lock:
        if route is self.prompted:
          self.prompted = None
          self._swapped(route, pos)
        self._watch_step(pos)
      self.off_since = None if off < self.OFF_ROUTE or route.on_road() else (self.off_since or now)
      if self.off_since is not None and now - self.off_since > self.OFF_FOR and now >= self.next_try:
        self._start(pos, bearing, z, now, fresh=True)
      elif self.off_since is None and pending is None and route.length - route.at < EXTEND_AT and walk is not None and \
          not walk.ended and now >= self.next_try:
        self._start(pos, bearing, z, now, fresh=False)
    elif now >= self.next_try:
      self._start(pos, bearing, z, now, fresh=True)
    return route

  @staticmethod
  def _at_junction(route, span: tuple[float, float] = JUNCTION_SPAN) -> bool:
    """Whether the car is at one of the route's junctions (span: m behind and ahead of the car), where the map driver
    splicing a new route into its path could kink it through the turn under way."""
    lo, hi = span
    found = list(route.junctions) + (list(route.lanes.junctions) if route.lanes is not None else [])
    return any(lo <= j - route.at <= hi for j in found)

  def _too_late(self, pos) -> bool:
    """Whether an arrow key's route came too late for its turn (routing it can take seconds): the car's nearer its
    junction than it needs, less the time it allowed to route it (the watch's late). Then it's dropped, said so."""
    w = self.watching
    if w is None or float(np.hypot(*(np.asarray(pos[:2], float) - w['xy']))) >= w['late']:
      return False
    self.log.append({'event': 'prompt_route', 'late': True, 'node': w['node']})
    if self.forced is not None:
      self.notes.append(('late', self.forced['side'], ''))
    self.forced = self.watching = None
    return True

  def _unban(self, pos):
    """Frees the ways out the router wouldn't take for an arrow key once the car's PROMPT_BAN_AWAY m from them: till
    then their junction answers at once, and coming back later they're asked for again."""
    p = np.asarray(pos[:2], float)
    for link, xy in list(self.prompt_bans.items()):
      if float(np.hypot(*(p - xy))) > PROMPT_BAN_AWAY:
        del self.prompt_bans[link]
        self.banned.discard(link)

  def _held(self, route, new) -> bool:
    """Whether a new route waits to be swapped in: one walked on while the car's at a junction (_at_junction); an arrow
    key's only while the car's in a turn (within JUNCTION_SPAN of one) or in a junction (within JUNCTION_SPAN's first m
    of its node, either side), where a plan spliced in can kink: junctions are too close together in a city for it to
    wait for a gap between them and still come before its turn. Should it then be too late, it's dropped (_too_late)."""
    if new is not self.prompted:
      return self._at_junction(route)
    lo, hi = JUNCTION_SPAN
    return bool(route.turns(hi, behind=-lo)) or self._at_junction(route, (lo, -lo))

  def _start(self, pos: np.ndarray, bearing: float, z: float | None, now: float, fresh: bool):
    if self.busy:
      return
    self.busy, self.next_try = self.busy + 1, now + self.RETRY
    threading.Thread(target=self._make, args=(np.array(pos, float), bearing, z, fresh, self.epoch), daemon=True).start()

  def _matcher(self) -> LaneMatcher:
    m = self.matcher()
    if m is None:
      if self._own_matcher is None or self._own_matcher.osm is not self.osm:
        self._own_matcher = LaneMatcher(self.osm)
      m = self._own_matcher
    return m

  def _make(self, pos: np.ndarray, bearing: float, z: float | None, fresh: bool, epoch: int):
    try:
      route, walk = self.make(pos, bearing, z, None if fresh else self.walk)
      if route is not None:
        route.locate(pos, z, -bearing)
      with self.lock:
        if epoch == self.epoch and route is not None and not fresh and self.route is not None:
          self.pending = route, walk  # walked on: swapped in once the car's clear of a junction (update)
        elif epoch == self.epoch and route is not None:
          self.route, self.walk, self.off_since, self.pending = route, walk, None, None
          self.forced = None  # walked afresh
        elif epoch == self.epoch and walk is not None and walk.ended:
          self.walk = walk  # the road ends: no further to walk
    except (OSError, ValueError, KeyError) as e:
      print(f"cruise: {e}")
    finally:
      with self.lock:
        self.busy -= 1

  def prompt(self, side: str, pos: np.ndarray, bearing: float, z: float | None, v: float) -> tuple[str, str]:
    """An arrow key (`side`: 'left' or 'right') while cruising, the car at pos (bearing clockwise from north) at v m/s:
    (kind, road), kind one of
    - 'turn' or 'keep': the next junction (or fork) within PROMPT_SECS s with a way out that side is taken that way (or
      already was), routed on its thread; road is the way out's name;
    - 'cancel': the other side's turn undone, the walk taking its junction as it would have;
    - 'late': the only such junctions are nearer than the car can still turn at (prompt_need), or the turn to undo is;
      'forbidden': only turn restrictions, dead ends or shortcut links across a painted separation that way; 'none': no junction that way ahead, nor a route to
      steer (a lane change instead)."""
    with self.lock:
      walk = self.pending[1] if self.pending is not None else self.walk
      if walk is None or self.route is None or self.osm is None:
        return 'none', ''
      walker = Walker(self.osm, self.banned, Restrictions.of(self.osm), Roaming.of(self.osm))
      walk = _copy(walk, walker)
      k, at = walk.locate(np.asarray(pos[:2], float))
      k = min(k, len(walk.nodes) - 2)
      at -= walk.along[k]
      walk.trim(k)
      forced = self.forced
      if forced is not None:
        ahead = [i for i in range(1, len(walk.nodes) - 1) if walk.nodes[i] == forced['node'] and walk.along[i] > at]
        if not ahead:
          forced = self.forced = None  # passed
        elif forced['side'] != side:
          # within JUNCTION_SPAN of its turn the undoing route would be held (_held) till the car had made the turn
          if walk.along[ahead[0]] - at < max(prompt_need(v, 0, True), JUNCTION_SPAN[1] + PROMPT_LAG * v):
            self.log.append({'event': 'prompt', 'kind': 'late', 'side': side, 'node': forced['node'], 'cancel': True})
            return 'late', ''  # the turn's under way: no time to be routed and spliced back
          walk.cut(ahead[0])  # walked on from the junction as it would have been
          self.forced = self.watching = None
          self.log.append({'event': 'prompt', 'kind': 'cancel', 'side': side, 'node': forced['node']})
          self._prompt(walk, at, None, pos, bearing, z)
          return 'cancel', forced['road']
      upto = at + max(PROMPT_MIN, PROMPT_SECS * v)
      lane = self.route.lane() if self.route.on_road() else None
      trace = {'side': side, 'v': round(v, 1), 'pos': [round(float(pos[0]), 1), round(float(pos[1]), 1)],
               'upto': round(upto - at), 'lane': lane, 'junctions': []}
      late = forbidden = False
      for kj, exits, cands, taken in walk.junctions(1, max(upto, at + PROMPT_TRACE)):
        if walk.along[kj] < at:
          continue
        # a way out by one of GTA's shortcut links across a painted separation, which the route would ban (Router.crossings)
        crossing = {x.edge for x in cands if x is not taken and self._crosses(x)}
        across = walker.side_exit(side, exits, cands)[0] if crossing else None
        cands = [x for x in cands if x.edge not in crossing]
        seen = self._trace_junction(walk, kj, at, exits, cands, taken, crossing)
        trace['junctions'].append(seen)
        e, fork = walker.side_exit(side, exits, cands)
        if walk.along[kj] > upto:
          seen['why'] = 'beyond' if e is not None and (taken is None or e.edge != taken.edge) else 'beyond_no'
          continue
        if e is None:
          ruled = (across is not None and across.edge in crossing) or self._ruled_out(walk, kj, side, cands)
          forbidden = forbidden or ruled
          seen['why'] = 'ruled_out' if ruled else 'no_way'
          continue
        road = walker.name(walker.way(*e.edge))
        seen['exit'] = e.edge[1]
        if taken is not None and e.edge == taken.edge:
          if fork:
            seen['why'] = 'fork_kept'
            forbidden = forbidden or (across is not None and across.edge in crossing)
            continue  # keeping to that side already: a junction further on, or a lane change
          seen['why'] = 'already'
          self._trace(trace, 'turn')
          return 'turn', road
        over = lanes_over(walk, kj, side, lane)
        need = prompt_need(v, over, fork)
        seen.update(over=over, need=round(need, 1))
        if walk.along[kj] - at < need:
          seen['why'] = 'late'
          late = True
          continue
        before = _copy(walk, walker)
        walk.force(kj, e)
        self.forced = {'side': side, 'node': walk.nodes[kj], 'link': e.edge, 'road': road}
        self._watch(before, walk, kj, need, need - PROMPT_LAG * v)
        seen['why'] = 'chosen'
        self._trace(trace, 'keep' if fork else 'turn')
        # the junction's other ways out that side (at a fork, further that way than the one the walk took), in turn, where
        # the router won't take this one
        sign = 1 if side == 'left' else -1
        if fork:
          others = sorted((x for x in cands if x is not e and (taken is None or sign * (x.turn - taken.turn) > 0)),
                          key=lambda x: -sign * x.turn)
        else:
          others = sorted((x for x in cands if x is not e and walker.move(x, exits) == side), key=lambda x: abs(x.turn))
        self._prompt(walk, at, e.edge, pos, bearing, z, (before, kj, others))
        return ('keep' if fork else 'turn'), road
      kind = 'late' if late else 'forbidden' if forbidden else 'none'
      self._trace(trace, kind)
      return kind, ''

  def _trace_junction(self, walk: Walk, k: int, at: float, exits: list[Exit], cands: list[Exit], taken: Exit | None,
                      crossing: set[tuple[int, int]]) -> dict:
    """A junction an arrow key looked at, for the log: m to it, and each way out (turn, move, why it's ruled out)."""
    w = walk.walker
    out = {'node': walk.nodes[k], 'm': round(walk.along[k] - at, 1), 'taken': taken.edge[1] if taken is not None else None,
           'exits': []}
    ok = {e.edge for e in cands}
    for e in exits:
      way = w.way(*e.edge)
      out['exits'].append({'to': e.edge[1], 'turn': round(e.turn), 'move': w.move(e, exits), 'cand': e.edge in ok,
                           'banned': e.edge in w.banned, 'on': e.on, 'minor': w.minor(way), 'road': w.name(way),
                           'roam': None if w.roaming is None else e.edge in w.roaming, 'len': len(e.path),
                           'crossing': e.edge in crossing})
    return out

  def _crosses(self, e: Exit) -> bool:
    """Whether way out e takes one of GTA's shortcut links across a painted separation (Router.crossings)."""
    return bool(self.router.crossings(np.array([self.osm.node_xy(n) for n in e.path])))

  def _trace(self, trace: dict, kind: str):
    self.log.append({'event': 'prompt', 'kind': kind, **trace})

  def _watch(self, before: Walk, forced: Walk, k: int, need: float, late: float):
    """Watches for whether the car takes a forced turn: which of the forced walk's and the walk before's points WATCH_PAST
    m past the junction it comes to first (update)."""
    s = forced.along[k] + WATCH_PAST
    self.watching = {'node': forced.nodes[k], 'xy': forced.osm.node_xy(forced.nodes[k]).tolist(),
                     'want': forced.point(min(s, forced.along[-1]))[0].tolist(),
                     'other': before.point(min(s, before.along[-1]))[0].tolist() if before.along[-1] > before.along[k] else None,
                     'swapped': None, 'need': need, 'late': late}

  def _watch_step(self, pos: np.ndarray):
    """The watched forced turn's outcome, once the car's near either way out (or well away from both)."""
    w = self.watching
    if w is None:
      return
    if self.pending is not None and self.pending[0] is self.prompted:
      return  # its route not in yet (held): the car's on the old one, and _too_late still needs the watch
    p = np.asarray(pos[:2], float)
    d_want = float(np.hypot(*(p - w['want'])))
    d_other = float(np.hypot(*(p - w['other']))) if w['other'] is not None else math.inf
    d_node = float(np.hypot(*(p - w['xy'])))
    w['closest'] = min(w.get('closest', math.inf), d_node)
    outcome = ('taken' if d_want <= d_other else 'missed') if min(d_want, d_other) < WATCH_NEAR else \
      'away' if d_node > 2 * WATCH_PAST and w['closest'] < WATCH_PAST else None
    if outcome is not None:
      self.log.append({'event': 'prompt_outcome', 'outcome': outcome, 'node': w['node'], 'swapped': w['swapped'],
                       'pos': [round(float(p[0]), 1), round(float(p[1]), 1)]})
      self.watching = None
      if self.forced is not None and self.forced['node'] == w['node']:
        self.forced = None  # passed: not to be undone should the walk come round to its junction again

  def _swapped(self, route, pos=None):
    """Logs where the car is, relative to the watched junction, as a forced turn's route goes in."""
    w = self.watching
    if w is None or w['swapped'] is not None:
      return
    w['swapped'] = {'at': round(route.at, 1)}
    if pos is not None:
      w['swapped']['m'] = round(float(np.hypot(*(np.asarray(pos[:2], float) - w['xy']))), 1)

  def take_log(self) -> list[dict]:
    with self.lock:
      log, self.log = self.log, []
    return log

  @staticmethod
  def _ruled_out(walk: Walk, k: int, side: str, cands: list[Exit]) -> bool:
    """Whether junction k has a way out `side` (not a minor road) that turn restrictions or a dead end rule out."""
    w = walk.walker
    free = Walker(w.osm, roaming=w.roaming)
    head = _copy(walk, free)
    head.cut(k)
    exits = free.exits(walk.nodes[k - 1], walk.nodes[k], head.heading_in(), head.ways)
    ok = {e.edge for e in cands}
    return any(e.edge not in ok and free.move(e, exits) == side and not w.minor(w.way(*e.edge)) for e in exits)

  def take_notes(self) -> list[tuple[str, str, str]]:
    with self.lock:
      notes, self.notes = self.notes, []
    return notes

  def _prompt(self, walk: Walk, at: float, keep: tuple[int, int] | None, pos, bearing: float, z: float | None,
              others: tuple[Walk, int, list[Exit]] | None = None):
    """Routes an arrow key's walk (forced at its junction, or cut back to it) on its own thread, in place of any route
    being made (the lock held); `others`: the walk before it was forced, its junction, and the other ways out to force
    there in turn where the router won't take the first."""
    self.epoch += 1
    self.pending, self.busy = None, self.busy + 1
    threading.Thread(target=self._make_prompt, args=(walk, at, keep, np.array(pos, float), bearing, z, self.epoch, others),
                     daemon=True).start()

  def _make_prompt(self, walk: Walk, at: float, keep: tuple[int, int] | None, pos: np.ndarray, bearing: float,
                   z: float | None, epoch: int, others: tuple[Walk, int, list[Exit]] | None = None):
    route, t0, banned0, err, slow = None, time.monotonic(), set(self.banned), None, False
    asked = {keep, *(alt.edge for alt in (others[2] if others is not None else []))}
    start, first = _copy(walk, walk.walker), keep
    try:
      for _ in range(PROMPT_TRIES):
        try:
          route, walk, keep = self._route_prompt(_copy(start, start.walker), at, first, epoch, others)
          slow = False
          break
        except OSError as e:
          if not timed_out(e) or epoch != self.epoch:
            raise
          slow, err = True, repr(e)  # the router's busy, not refusing the turn: asked again
      if route is not None:
        route.prompt = True  # the map driver keeps less of its old path: the turn may be only a few s on
        route.locate(pos, z, -bearing)
    except (OSError, ValueError, KeyError, urllib.error.URLError) as e:
      err = repr(e)
      print(f"cruise: {e}")
    finally:
      with self.lock:
        self.busy -= 1
        held = False
        for link in (self.banned - banned0) & asked:
          self.prompt_bans[link] = self.osm.node_xy(link[0])
        if epoch == self.epoch and route is not None:
          self.prompted = route
          if self.route is not None:  # swapped in by update, once the car's out of a turn and if it's still in time
            self.pending = route, walk
            held = self._held(self.route, route)
          else:
            self.route, self.walk, self.off_since, self.pending = route, walk, None, None
        elif epoch == self.epoch and keep is not None and self.forced is not None:
          self.notes.append(('router_busy' if slow else 'forbidden', self.forced['side'], self.forced['road']))
          self.forced = None
          self.watching = None
        self.log.append({'event': 'prompt_route', 'ok': route is not None, 'secs': round(time.monotonic() - t0, 2),
                         'stale': epoch != self.epoch, 'held': held, 'keep': list(keep) if keep else None,
                         'has_keep': keep is None or walk.has(keep), 'banned': sorted(self.banned - banned0), 'err': err,
                         'timeout': slow})

  def _route_prompt(self, walk: Walk, at: float, keep: tuple[int, int] | None, epoch: int,
                    others: tuple[Walk, int, list[Exit]] | None):
    """An arrow key's walk routed, else in turn its junction's other ways out that side (`others`): (route, walk, the
    way out it takes); (None, walk, keep) where none can be."""
    route, walk = self._route(walk, at, AHEAD, 0, keep)
    base, k, alts = others if others is not None else (None, 0, [])
    for alt in alts:
      if route is not None:
        break
      forced = _copy(base, walk.walker)
      forced.force(k, alt)
      route, walk = self._route(forced, at, AHEAD, 0, alt.edge)
      if route is not None:
        keep = alt.edge
        with self.lock:
          if epoch == self.epoch and self.forced is not None:
            self.forced = {**self.forced, 'link': alt.edge, 'road': walk.walker.name(walk.walker.way(*alt.edge))}
            w = self.watching or {'need': 0.0, 'late': 0.0}
            self._watch(base, forced, k, w['need'], w['late'])
    return route, walk, keep

  def make(self, pos: np.ndarray, bearing: float, z: float | None, walk: Walk | None = None):
    """A cruise route from the car (pos, bearing clockwise from north, height z): along a new walk, or along `walk`
    walked on past the car's place on it; and its walk. (None, walk) where none can be made."""
    osm = self.osm
    walker = Walker(osm, self.banned, Restrictions.of(osm), Roaming.of(osm))
    fresh = walk is None
    if fresh:
      found = start_link(osm, walker, pos[:2], -bearing, z, self._matcher())
      if found is None:
        raise ValueError("no road the car's way to cruise on")
      walk = Walk(walker, found[0], found[1], found[2])
      _, at = walk.locate(pos[:2])
      walk.car, ahead = at, FIRST_AHEAD
    else:
      walk, ahead = _copy(walk, walker), AHEAD  # walked on as a copy: the one routed stays until this one is
      k, at = walk.locate(pos[:2])
      k = min(k, len(walk.nodes) - 2)
      at -= walk.along[k]
      walk.trim(k)
    return self._route(walk, at, ahead, 0 if fresh else len(walk.nodes))  # an extension's nodes so far

  def _route(self, walk: Walk, at: float, ahead: float, known: int, keep: tuple[int, int] | None = None):
    """The walk walked on to `ahead` m past the car (`at` m along it) and routed: (route, walk); (None, walk) where none
    can be made, it's no further on than `known` nodes, or it no longer takes the link `keep`."""
    pinned: set[int] = set()  # nodes the route was made to go through as it left the walk there
    grow = True
    for _ in range(ROUNDS):
      if grow:
        walk.extend(at + ahead, MAX_THROUGH, at)
      walk.cap(at, MAX_THROUGH)
      if walk.along[-1] - at < 1.0 or len(walk.nodes) <= known:
        return None, walk  # nowhere on, or no further than the route it has
      crossing = self._crossing(walk)
      if crossing is not None:
        self.banned.add((walk.nodes[crossing - 1], walk.nodes[crossing]))
        walk.cut(crossing - 1)
        continue
      try:
        route = self.router.via(walk.point(at), walk.through(at + THROUGH_AHEAD), walk.end())
      except urllib.error.HTTPError:
        # no route through them all (one past a turn the restrictions forbid, whose way round dead-ends): short of the last
        if not walk.choices:
          raise
        walk.cut(walk.choices[-1][0])
        grow = False  # walked on again by the next extension
        continue
      k = strays(route.points, walk.points())
      if k is None:
        if keep is not None and not walk.has(keep):
          return None, walk
        route.cruise = True
        return route, walk
      # Valhalla left the walk: at a junction it chose at, the way out is one it can't take from there (the turn
      # restrictions); at another, it found a way round, which a location just past the junction keeps it off
      j = self._left_at(walk, route.points, k)
      if j is None:
        return None, walk
      c = next((c for jj, c in walk.choices if jj == j), None)
      if c is None and j not in pinned:
        pinned.add(j)
        walk.choices = sorted(walk.choices + [(j, j + 1)])
        continue
      c = c if c is not None else j + 1
      self.banned.add((walk.nodes[c - 1], walk.nodes[c]))
      walk.cut(j)
    return None, walk

  @staticmethod
  def _left_at(walk: Walk, points: np.ndarray, k: int) -> int | None:
    """The junction node (its index in the walk) where a route's points, k of them the first far from the walk, left
    it: the walk's node with another way on nearest before the last of them on it; None for none within JUNCTION_NEAR."""
    line = walk.points()
    i = k
    while i > 0 and strays(points[i:i + 1], line, start=-1.0, far=2.0) is not None:
      i -= 1
    _, s = walk.locate(points[i])
    def ways_on(j):
      n = walk.nodes[j]
      return sum(walk.walker.drivable(n, m) for m in walk.osm.links.get(n, ()) if m != walk.nodes[j - 1])
    near = [j for j in range(1, len(walk.nodes) - 1) if s - JUNCTION_NEAR <= walk.along[j] <= s + 5.0 and ways_on(j) > 1]
    return min(near, key=lambda j: abs(walk.along[j] - s)) if near else None

  def _crossing(self, walk: Walk) -> int | None:
    """The first node past one of GTA's shortcut links across a painted separation the walk takes (Router.crossings)."""
    spots = self.router.crossings(walk.points())
    if not spots:
      return None
    pts = walk.points()
    mids = (pts[:-1] + pts[1:]) / 2
    return int(np.argmin(np.hypot(*(mids - spots[0]).T))) + 1


def _copy(walk: Walk, walker: Walker) -> Walk:
  out = Walk.__new__(Walk)
  out.walker, out.osm = walker, walker.osm
  out.nodes, out.along, out.choices, out.ways = list(walk.nodes), list(walk.along), list(walk.choices), list(walk.ways)
  out.turn, out.ended, out.car, out.passed = walk.turn, walk.ended, walk.car, list(walk.passed)
  return out
