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
into its path on a straight rather than in a turn."""
import math
import threading
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
ROAM_MIN = 2000.0  # m of road a set of links a car can drive round for ever has, not to be a pocket
ROUNDS = 6  # routes again after the walk is changed (a ban, a junction to go through), at most


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
    """The ways out it may take: not banned, and while there's another, not one the turn restrictions leave no way on
    from, nor into a pocket (else a dead end), nor (off a road that isn't one) a minor road."""
    cands = [e for e in exits if e.edge not in self.banned]
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

    def same(e: Exit) -> bool:
      w = self.way(*e.edge)
      return self.name(w) == road if road else not self.name(w) and self.road_class(w) == self.road_class(approach)
    carried = [e for e in cands if abs(e.turn) <= SAME_ROAD_TURN and same(e)]
    return min(carried or cands, key=lambda e: abs(e.turn))


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
      turn = self.turn if self.along[-1] - self.car <= LANE_REACH else None
      self.turn = None  # the car's lane is for the first junction only
      e = w.choose(self.road(), approach, exits, cands, turn)
      at = len(self.nodes) - 1
      for n in e.path[1:]:
        self._add(n)
      if len(cands) > 1:
        self.choices.append((at, len(self.nodes) - 1))

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
    self.off_since: float | None = None
    self.next_try = 0.0
    self.busy = False
    self.epoch = 0  # reset() moves it on: a route being made before is dropped
    self.lock = threading.Lock()

  @property
  def osm(self) -> OsmLanes | None:
    return self.router.osm if self.router.osm is not None else self.router.roads

  def reset(self):
    """No cruising: the route and its walk go (a destination's route drives instead, or the map driver is off)."""
    with self.lock:
      if self.route is None and self.walk is None and self.off_since is None and self.pending is None:
        return
      self.route, self.walk, self.off_since, self.next_try, self.pending = None, None, None, 0.0, None
      self.epoch += 1

  def update(self, pos: np.ndarray, bearing: float, now: float, z: float | None = None):
    """The cruise route, the car located on it (pos and bearing, clockwise from north, the car's); None until made."""
    if self.osm is None:
      return None
    with self.lock:
      route, walk, pending = self.route, self.walk, self.pending
      if route is not None and pending is not None and (not self._at_junction(route) or route.length - route.at < SWAP_BY):
        route, walk = self.route, self.walk = pending
        self.pending = None
    if route is not None:
      off = route.locate(pos, z, -bearing)
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
  def _at_junction(route) -> bool:
    """Whether the car is at one of the route's junctions (JUNCTION_SPAN), where the map driver splicing a new route
    into its path could kink it through the turn under way."""
    lo, hi = JUNCTION_SPAN
    found = list(route.junctions) + (list(route.lanes.junctions) if route.lanes is not None else [])
    return any(lo <= j - route.at <= hi for j in found)

  def _start(self, pos: np.ndarray, bearing: float, z: float | None, now: float, fresh: bool):
    if self.busy:
      return
    self.busy, self.next_try = True, now + self.RETRY
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
        elif epoch == self.epoch and walk is not None and walk.ended:
          self.walk = walk  # the road ends: no further to walk
    except (OSError, ValueError, KeyError) as e:
      print(f"cruise: {e}")
    finally:
      self.busy = False

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
    known = 0 if fresh else len(walk.nodes)  # an extension's nodes so far
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
  out.turn, out.ended, out.car = walk.turn, walk.ended, walk.car
  return out
