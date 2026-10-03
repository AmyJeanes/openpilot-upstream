"""GTA's own road data (ynddump's paths.jsonl, map/README.md) for our map's routes: the height of each road, its lanes
each way as the game lays them out, and the roads branching off it. Our map's ways are GTA's links, so a route's shape
points are GTA's nodes."""
import json
import math
from collections import defaultdict

import numpy as np

# m: CodeWalker draws lanes this wide, out from the link's line by its offset (half a lane at most), or centred on a
# one-way link; where cars drive in the game matches
LANE_WIDTH, NARROW_LANE_WIDTH = 5.5, 4.0
CAR_HEIGHT = 0.6  # m: the car's position is this far above the road's nodes
CELL = 20.0  # m
PED_SPECIALS = {14, 18}
SLIP_LANE, LEFT_TURN_ONLY = 1, 128  # node flags 1 and 4


def wrap(deg: float) -> float:
  return (deg + 180) % 360 - 180


def heading(dx: float, dy: float) -> float:
  """Game heading, counterclockwise from north."""
  return math.degrees(math.atan2(-dx, dy))


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
