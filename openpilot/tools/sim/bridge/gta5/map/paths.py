"""GTA's own road data (ynddump's paths.jsonl, map/README.md) for our map's routes: the height of each road, its lanes
each way as the game lays them out, and the roads branching off it. Our map's ways are GTA's links, so a route's shape
points are GTA's nodes."""
import heapq
import json
import math
from collections import Counter, defaultdict

import numpy as np

# m: CodeWalker draws lanes this wide, out from the link's line by its offset (half a lane at most), or centred on a
# one-way link; where cars drive in the game matches
LANE_WIDTH, NARROW_LANE_WIDTH = 5.5, 4.0
CAR_HEIGHT = 0.6  # m: the car's position is this far above the road's nodes
CELL = 20.0  # m
PED_SPECIALS = {14, 18}
SLIP_LANE, LEFT_TURN_ONLY = 1, 128  # node flags 1 and 4
STOP_LINES = {15, 16}  # specials: a traffic light's stop line, a stop junction's
JUNCTION = 4  # node flags 2
# GTA's stop lines (and its no left / no right turn flags) don't say which way they face: they're for the junction
# ahead within STOP_REACH (GTA's are 12-24 m before its node), or within STOP_REACH_FAR if that's nearer than the one
# behind. Where a road has junctions about as near both ways (within STOP_TIE), the one with more stop lines.
STOP_REACH = 30.0  # m
STOP_REACH_FAR = 50.0  # m
STOP_TIE = 5.0  # m


def wrap(deg: float) -> float:
  return (deg + 180) % 360 - 180


def heading(dx: float, dy: float) -> float:
  """Game heading, counterclockwise from north."""
  return math.degrees(math.atan2(-dx, dy))


def roads_cross(headings) -> bool:
  """Whether a junction node's links (their headings from it) cross, as where roads meet: GTA also flags the nodes where
  a road's lanes split as junctions."""
  return any(45.0 <= abs(wrap(h - g)) <= 135.0 for n, h in enumerate(headings) for g in headings[n + 1:])


def nearest_junction(i, firsts, step, length, junction, reach: float):
  """The nodes from i, by one of `firsts`, to the nearest junction node within `reach` m along step(node) (the nodes on
  from it), and that distance; None if none."""
  best = {j: length(i, j) for j in firsts}
  prev = dict.fromkeys(firsts, i)
  heap = [(d, n, j) for n, (j, d) in enumerate(best.items()) if d <= reach]
  heapq.heapify(heap)
  done, n = {i}, len(heap)
  while heap:
    d, _, k = heapq.heappop(heap)
    if k in done:
      continue
    done.add(k)
    if junction(k):
      path = [k]
      while path[-1] != i:
        path.append(prev[path[-1]])
      return path[::-1], d
    for m in step(k):
      dm = d + length(k, m)
      if m not in done and dm <= reach and dm < best.get(m, math.inf):
        best[m], prev[m], n = dm, k, n + 1
        heapq.heappush(heap, (dm, n, m))
  return None


def junction_scores(stops, out, length, junction) -> Counter:
  """How many of the stop line nodes have each junction node nearest ahead within STOP_REACH one way or another."""
  score: Counter = Counter()
  for i in stops:
    near = {found[0][-1] for j in out(i) if (found := nearest_junction(i, [j], out, length, junction, STOP_REACH))}
    score.update(near)
  return score


def toward_junction(i, out, into, length, link_heading, junction, score: Counter) -> dict:
  """The ways on from node i towards the junction it stands before, as a stop line or a turn flag does (see
  STOP_REACH): {next node: the nodes from i to the junction's node}; empty where none is ahead."""
  found = {}
  for j in out(i):
    ahead = nearest_junction(i, [j], out, length, junction, STOP_REACH_FAR)
    if ahead is None:
      continue
    if ahead[1] > STOP_REACH and nearest_junction(i, [p for p in into(i) if p != j], into, length, junction, ahead[1]):
      continue  # nearer the junction behind it, which it's past
    found[j] = ahead
  if not found:
    return {}

  def same_way(j, best):  # lanes splitting off towards the same junction, not the road back
    return abs(wrap(link_heading(i, j) - link_heading(i, best))) < 90

  best = min(found, key=lambda j: found[j][1])
  back = [j for j in found if not same_way(j, best) and found[j][1] - found[best][1] < STOP_TIE]
  if back:
    rival = min(back, key=lambda j: found[j][1])
    mine, theirs = score[found[best][0][-1]], score[found[rival][0][-1]]
    if theirs == mine:
      return {j: path for j, (path, _) in found.items()}  # can't tell
    if theirs > mine:
      best = rival
  return {j: path for j, (path, _) in found.items() if same_way(j, best)}


class Link:
  __slots__ = ('lanes', 'back', 'inner', 'width', 'shortcut', 'no_nav')

  def __init__(self, f: list[int]):
    self.lanes, self.back = (f[2] >> 5) & 7, (f[2] >> 2) & 7  # this way, and the other
    self.width = NARROW_LANE_WIDTH if f[1] & 2 else LANE_WIDTH
    offset = ((f[1] >> 4) & 7) / 7 * (-0.5 if f[1] & 128 else 0.5) * self.width
    # m right of the link's line to the left edge of the lanes this way
    self.inner = offset if self.back else offset - self.lanes * self.width / 2
    self.shortcut = bool(f[2] & 2)
    self.no_nav = bool(f[2] & 1)  # GTA's "don't use for navigation", as on the links splitting a road's lanes at a junction

  def lane(self, right: float) -> int:
    """The lane (from the left; negative across in the oncoming lanes) of a point `right` m right of the link's line."""
    i = math.floor((right - self.inner) / self.width)
    return max(i, -self.back) if i < 0 else min(i, self.lanes - 1)


class Paths:
  def __init__(self, path: str):
    xs, zs, flags, street, index, raw = [], [], [], [], {}, []
    with open(path) as f:
      for line in f:
        d = json.loads(line)
        if d['t'] == 'n':
          index[(d['a'], d['i'])] = len(xs)
          xs.append((d['x'], d['y']))
          zs.append(d['z'])
          flags.append(d['f'])
          street.append(d['st'])
        elif d['t'] == 'l':
          raw.append(((d['a'], d['i']), (d['ta'], d['ti']), d['f']))
    self.xy = np.array(xs, dtype=float)
    self.z = np.array(zs, dtype=float)
    self.flags = flags
    self.street = street
    self.links: dict[tuple[int, int], Link] = {}
    self.out: dict[int, list[int]] = defaultdict(list)  # node -> nodes a car can drive on to
    self.into: dict[int, list[int]] = defaultdict(list)  # node -> nodes a car can come from
    for a, b, f in raw:
      if a not in index or b not in index:
        continue
      i, j = index[a], index[b]
      if self.ped(i) or self.ped(j):
        continue
      link = Link(f)
      self.links[(i, j)] = link
      if link.lanes:
        self.out[i].append(j)
        self.into[j].append(i)
    self.exact: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, (x, y) in enumerate(self.xy):
      self.exact[(round(x), round(y))].append(i)
    self.cells: dict[tuple[int, int], set[tuple[int, int]]] | None = None
    self._toward: dict[int, set[int]] | None = None  # stop line -> the nodes on from it towards its junction

  def stop_line(self, i: int) -> bool:
    return (self.flags[i][1] >> 3) in STOP_LINES

  def stop_for(self, i: int, nxt: int) -> bool:
    """Whether stop line i stops the way on to node nxt: towards the junction it stands before, not away from it."""
    if self._toward is None:
      def out(k):
        return self.out.get(k, ())

      def into(k):
        return self.into.get(k, ())

      def length(a, b):
        return float(np.hypot(*(self.xy[b] - self.xy[a])))

      def junction(k):
        return self.junction(k) and roads_cross([self.link_heading(k, j) for j in {*out(k), *into(k)}])
      stops = [k for k in range(len(self.xy)) if self.stop_line(k) and not junction(k)]
      score = junction_scores(stops, out, length, junction)
      self._toward = {k: set(toward_junction(k, out, into, length, self.link_heading, junction, score)) for k in stops}
    return nxt in self._toward[i] if i in self._toward else True

  def junction(self, i: int) -> bool:
    return bool(self.flags[i][2] & JUNCTION)

  def ped(self, i: int) -> bool:
    return (self.flags[i][1] >> 3) in PED_SPECIALS

  def nodes_at(self, p, tol: float = 0.5) -> list[int]:
    """The nodes at a point of a route's shape, nearest first: more than one where roads pass over each other."""
    x, y = round(p[0]), round(p[1])
    found = []
    for dx in (-1, 0, 1):
      for dy in (-1, 0, 1):
        for i in self.exact.get((x + dx, y + dy), ()):
          d = math.hypot(*(self.xy[i] - p[:2]))
          if d < tol:
            found.append((d, i))
    return [i for _, i in sorted(found)]

  def route_nodes(self, points) -> list[int | None]:
    """The node at each point of a route's shape, None between nodes, choosing among nodes at the same place those
    linked along the route."""
    found = [self.nodes_at(p) for p in points]
    out: list[int | None] = []
    for k, cands in enumerate(found):
      prev = out[-1] if out else None
      nxt = found[k + 1] if k + 1 < len(found) else []
      linked = [c for c in cands if (prev is not None and (prev, c) in self.links) or any((c, n) in self.links for n in nxt)]
      out.append((linked or cands or [None])[0])
    return out

  def link_heading(self, i: int, j: int) -> float:
    return heading(*(self.xy[j] - self.xy[i]))

  def index(self):
    """Links by the cells they pass through, for looking them up near a point."""
    self.cells = defaultdict(set)
    for (i, j) in self.links:
      if j < i and (j, i) in self.links:
        continue
      p, q = self.xy[i], self.xy[j]
      steps = max(2, int(np.hypot(*(q - p)) / (CELL / 2)) + 2)
      for t in np.linspace(0.0, 1.0, steps):
        c = p + (q - p) * t
        self.cells[(int(c[0] // CELL), int(c[1] // CELL))].add((i, j))

  def _near(self, pos):
    """The links (both ways) with a cell next to pos's."""
    if self.cells is None:
      self.index()
    assert self.cells is not None
    cx, cy = int(pos[0] // CELL), int(pos[1] // CELL)
    seen = set()
    for dx in (-1, 0, 1):
      for dy in (-1, 0, 1):
        for i, j in self.cells.get((cx + dx, cy + dy), ()):
          if (i, j) not in seen:
            seen.add((i, j))
            yield i, j

  def _along(self, a: int, b: int, pos) -> tuple[float, float, float]:
    """How far along link a-b pos is (0-1), how far right of its line, and the road's height there."""
    p, q = self.xy[a], self.xy[b]
    ab = q - p
    length2 = max(float(ab @ ab), 1e-6)
    t = float(np.clip((pos[:2] - p) @ ab / length2, 0.0, 1.0))
    right = float((pos[0] - p[0]) * ab[1] - (pos[1] - p[1]) * ab[0]) / math.sqrt(length2)
    return t, right, float(self.z[a] + (self.z[b] - self.z[a]) * t)

  def snap(self, pos, z: float, car_heading: float, max_off: float = 15.0, max_dz: float = 3.0, max_turn: float = 35.0):
    """The point on the line of the link the car is on, at its height and heading, the link's heading and its next
    node; None if none is near. Where roads pass over each other, the one at the car's height."""
    best, best_d = None, max_off
    for i, j in self._near(pos):
      for a, b in ((i, j), (j, i)):
        link = self.links.get((a, b))
        if link is None or not link.lanes or link.shortcut or np.hypot(*(self.xy[b] - self.xy[a])) < 1.0:
          continue
        h = self.link_heading(a, b)
        if abs(wrap(car_heading - h)) > max_turn:
          continue
        t, right, road_z = self._along(a, b, pos)
        if abs(z - CAR_HEIGHT - road_z) > max_dz:
          continue
        near = self.xy[a] + (self.xy[b] - self.xy[a]) * t
        # from the lanes this way, not the line, so a car in an outside lane isn't nearer a road beside it
        lo, hi = link.inner, link.inner + link.lanes * link.width
        d = max(lo - right, right - hi, 0.0) + float(np.hypot(*(pos[:2] - near))) * 0.01
        if d < best_d:
          best, best_d = (a, b, t, road_z), d
    if best is None:
      return None
    a, b, t, _ = best
    return self.xy[a] + (self.xy[b] - self.xy[a]) * t, self.link_heading(a, b), b

  def forks(self, i_prev: int, i: int, i_next: int, spread: float) -> list[tuple[float, int]]:
    """The other roads a car can take on from node i, arriving from i_prev, within `spread` deg of straight on: their
    heading relative to the way in (left positive) and their next node."""
    h_in = self.link_heading(i_prev, i)
    out = []
    for j in self.out.get(i, ()):
      if j in (i_next, i_prev) or self.links[(i, j)].shortcut:
        continue
      rel = wrap(self.link_heading(i, j) - h_in)
      if abs(rel) < spread:
        out.append((rel, j))
    return out
