"""The map's stop lines for nav: junctions.py's (from the map's `highway=traffic_signals` / `stop` / `give_way` nodes and
their direction tags, placed on the game's paint by stop_paint.py), one per approach, across the lanes towards the
junction. Each is kept as the links a route crosses it on, from node a to node b towards its junction, and how far past a
it is: the link of the road it's drawn on, and the links of roads beside it that the line reaches across, as where GTA
lays an approach as a link per lane with the line on one of them. A route whose shape points are the map's nodes crosses
it where it drives one of those links that way, and only that way, at the place the map view and the overlay draw it.
Working out the junctions takes about half a minute on the whole map, so they're kept in a cache beside the junction
areas (lane_match.py), built with them."""
import math
import os

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes, oneway_of

KINDS = ('stop', 'lights', 'give_way')  # a stop sign's, a traffic light's, a give way sign's
STOPS_VERSION = 1  # of the cache: bump when what goes into it changes
ON_LINK = 1.0  # m from a link: a route's end point part way along it is on it
NEAR_LINE = 6.0  # m: a route's point off the road's nodes is at a stop line this near it
REACH = 5.0  # m past a stop line's ends a road beside the one it's drawn on still crosses it (a lane's width)
ON_REACH = 1.5  # m past its ends a road crosses the line itself
SQUARE = math.cos(math.radians(45.0))  # a road beside crosses a stop line within this of square ...
ALIGN = math.cos(math.radians(30.0))  # ... running within this of the line's own road ...
ALIGN_ON = math.cos(math.radians(50.0))  # ... or within this where it crosses the line itself (a lane change link across)
LEVEL = 4.0  # m: ... at its height, and has no line of its own
# m along a route: a line crossed on a road beside its own gives way to a line of the route's own road this near, and to
# another such line crossed before it this near (GTA lays some approaches as a link per lane, each with its own line,
# staggered along the road)
BESIDE_SPAN = 15.0
SAME_LINE = 10.0  # m along a route: crossings of one line this near are one
SAME_STOP = 1.0  # m along a route: crossings of lines this near are one
CELL = 25.0  # m, of the links' grid while they're worked out
MIN_LINK = 0.5  # m: a shorter link (a stop line's node split beside GTA's) says nothing of the way it's driven


class StopLines:
  def __init__(self, a, b, offset, kind, line, own, group):
    self.a = np.asarray(a, np.int64)
    self.b = np.asarray(b, np.int64)  # the link's nodes, b nearer the junction
    # m from a towards b to the line; negative past the road's nodes junctions.py followed out from the junction (its
    # ARM_LENGTH), where the line is that far before a along the road
    self.offset = np.asarray(offset, float)
    self.kind = np.asarray(kind, np.int8)  # into KINDS
    self.line = np.asarray(line, float).reshape(-1, 2, 2)  # across the lanes, right to left arriving
    self.own = np.asarray(own, bool)  # the road it's drawn on, else one beside it
    self.group = np.asarray(group, np.int64)  # which line: the same for its own road's link and those beside
    self.into: dict[int, list[int]] = {}  # by b
    self.out: dict[int, list[int]] = {}  # by a
    for i, (a_, b_) in enumerate(zip(self.a.tolist(), self.b.tolist(), strict=True)):
      self.into.setdefault(b_, []).append(i)
      self.out.setdefault(a_, []).append(i)

  def __len__(self) -> int:
    return len(self.offset)

  @classmethod
  def from_junctions(cls, junctions) -> 'StopLines':
    from openpilot.tools.sim.bridge.gta5.map.lane_match import node_heights
    osm = junctions.osm
    links = _Links(osm, node_heights(osm))
    rows = []
    group = 0
    for j in junctions.junctions:
      for s in j.stops:
        own = _link(osm, s.member.nodes, s.along)
        if own is None:
          continue
        kind = KINDS.index('lights') if s.signal else KINDS.index(s.kind)
        rows.append((*own, kind, s.line, True, group))
        arriving = -s.member.line.tangent(s.along)
        for a, b, offset in links.across(s.line, arriving, own[:2]):
          rows.append((a, b, offset, kind, s.line, False, group))
        group += 1
    owned = {(r[0], r[1]) for r in rows if r[5]}
    rows = [r for r in rows if r[5] or (r[0], r[1]) not in owned]
    cols = list(zip(*rows, strict=True)) if rows else [()] * 7
    return cls(np.array(cols[0], np.int64), np.array(cols[1], np.int64), np.array(cols[2], float), np.array(cols[3], np.int8),
               np.array(cols[4], float).reshape(-1, 2, 2), np.array(cols[5], bool), np.array(cols[6], np.int64))

  def save(self, path: str):
    np.savez(path, a=self.a, b=self.b, offset=self.offset, kind=self.kind, line=self.line, own=self.own, group=self.group)

  @classmethod
  def load(cls, path: str) -> 'StopLines':
    with np.load(path) as f:
      return cls(f['a'], f['b'], f['offset'], f['kind'], f['line'], f['own'], f['group'])

  @staticmethod
  def cache_file(osm_path: str, cache_dir: str | None = None) -> str:
    """Where a map's stop lines are kept, by the map file's contents (beside its junction areas)."""
    from openpilot.tools.sim.bridge.gta5.map.lane_match import CACHE_DIR, map_hash
    return os.path.join(cache_dir or CACHE_DIR, f"stop_lines-{map_hash(osm_path, STOPS_VERSION)}.npz")

  @classmethod
  def cached(cls, osm: OsmLanes, cache_dir: str | None = None, build: bool = True, junctions=None) -> 'StopLines | None':
    """A map's stop lines (osm read from a file) from the cache; if they aren't there yet and `build`, worked out (from
    `junctions` where given) and kept there, else None. A map not read from a file is worked out each time."""
    if osm.path is None:
      return cls._build(osm, junctions) if build else None
    path = cls.cache_file(osm.path, cache_dir)
    try:
      return cls.load(path)
    except (OSError, ValueError, KeyError):
      if not build:
        return None
    from openpilot.tools.sim.bridge.gta5.map.lane_match import keep
    lines = cls._build(osm, junctions)
    keep(lines, path)
    return lines

  @classmethod
  def _build(cls, osm: OsmLanes, junctions=None) -> 'StopLines':
    from openpilot.tools.sim.bridge.gta5.map.junctions import Junctions
    return cls.from_junctions(junctions if junctions is not None else Junctions(osm))

  @classmethod
  def of(cls, osm: OsmLanes, build: bool = True) -> 'StopLines | None':
    """A map's stop lines, read from the cache (or worked out) once per OsmLanes; None if they aren't kept yet and not
    `build`."""
    lines = getattr(osm, '_stop_lines', None)
    if lines is None:
      lines = cls.cached(osm, build=build)
      if lines is not None:
        osm._stop_lines = lines
    return lines

  def on_route(self, points, along, osm: OsmLanes) -> list[tuple[float, str]]:
    """The stop lines a route crosses towards their junctions: [(m along it, kind)], in order. It crosses one where it
    drives one of the line's links towards the junction, or starts or ends part way along one on the line's far or
    near side. A route point is at every map node at its place (MATCH_TOL), so which of nodes over each other, or of
    a stop line's node and GTA's beside it, the route is said to pass doesn't matter."""
    pts = np.asarray(points, float)[:, :2]
    along = np.asarray(along, float)
    at_nodes = [set(osm.nodes_at(p)) for p in pts]
    n = len(pts)
    hits: list[tuple[float, int]] = []
    for k in range(n - 1):
      here, nxt = at_nodes[k], at_nodes[k + 1]
      if here and nxt and along[k + 1] - along[k] < MIN_LINK:
        continue  # nodes at one place, both at both points
      if here and nxt:
        for a in here:
          for i in self.out.get(a, ()):
            if int(self.b[i]) in nxt:
              at = float(along[k] + self.offset[i])
              if self.offset[i] >= 0.0 or (at >= along[0] and self._near(pts, along, at, i)):
                hits.append((at, i))
      elif k == 0 and nxt:  # the route starts part way along a link into the next point's node
        for b in nxt:
          for i in self.into.get(b, ()):
            past = _along_link(pts[0], osm, int(self.a[i]), b)  # m from the link's start to the route's
            if past is not None and past <= self.offset[i]:
              hits.append((float(along[0] + self.offset[i] - past), i))
      elif k == n - 2 and here:  # the route ends part way along a link out of this point's node
        for a in here:
          for i in self.out.get(a, ()):
            past = _along_link(pts[-1], osm, a, int(self.b[i]))
            if past is not None and 0.0 <= self.offset[i] <= past:
              hits.append((float(along[k] + self.offset[i]), i))
    hits.sort()
    lines: list[tuple[float, int]] = []
    for at, i in hits:  # one crossing of each line, on its own road's link where the route takes it
      same = next((n for n, (s, j) in enumerate(lines) if self.group[j] == self.group[i] and at - s < SAME_LINE), None)
      if same is None:
        lines.append((at, i))
      elif self.own[i] and not self.own[lines[same][1]]:
        lines[same] = (at, i)
    owns = [s for s, i in lines if self.own[i]]
    out: list[tuple[float, str]] = []
    beside = -math.inf
    for at, i in sorted(lines):
      if not self.own[i]:
        if at - beside < BESIDE_SPAN or any(abs(at - s) < BESIDE_SPAN for s in owns):
          continue
        beside = at
      kind = KINDS[int(self.kind[i])]
      if out and at - out[-1][0] < SAME_STOP:
        if kind == 'lights':
          out[-1] = (out[-1][0], kind)
        continue
      out.append((at, kind))
    return out

  def _near(self, pts, along, s: float, i: int) -> bool:
    """Whether the route's point s m along it is at stop line i: the route runs on the road's line, the left edge of the
    lanes the line crosses, or beside them across a median."""
    p = np.array([np.interp(s, along, pts[:, 0]), np.interp(s, along, pts[:, 1])])
    a, ab = self.line[i, 0], self.line[i, 1] - self.line[i, 0]
    t = float(np.clip(np.dot(p - a, ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0))
    return float(np.hypot(*(a + ab * t - p))) < NEAR_LINE


class _Links:
  """The map's links (each way's node to node), in a grid, to find those a stop line reaches across."""
  def __init__(self, osm: OsmLanes, heights: dict[int, float]):
    self.osm, self.heights = osm, heights
    self.rows = [(a, b, oneway_of(tags)) for tags, refs in osm.ways.values() for a, b in zip(refs, refs[1:], strict=False)]
    self.cells: dict[tuple[int, int], list[int]] = {}
    for k, (a, b, _) in enumerate(self.rows):
      pa, pb = osm.node_xy(a), osm.node_xy(b)
      lo, hi = np.minimum(pa, pb), np.maximum(pa, pb)
      for cx in range(int(lo[0] // CELL), int(hi[0] // CELL) + 1):
        for cy in range(int(lo[1] // CELL), int(hi[1] // CELL) + 1):
          self.cells.setdefault((cx, cy), []).append(k)

  def across(self, line, arriving, own: tuple[int, int]) -> list[tuple[int, int, float]]:
    """(a, b, m from a to the line) for the links other than `own` that cross the line carried REACH m past its ends,
    near square to it, running within ALIGN of `arriving` (the line's road's direction towards its junction; ALIGN_ON
    where they cross the line itself) at its own link's height, each the way it may be driven towards the junction."""
    p0, p1 = np.asarray(line[0], float), np.asarray(line[1], float)
    u = p1 - p0
    width = float(np.hypot(*u))
    if width < 1e-6:
      return []
    u /= width
    p0, p1 = p0 - u * REACH, p1 + u * REACH
    zs = [self.heights[n] for n in own if n in self.heights]
    z = float(np.mean(zs)) if zs else None
    lo, hi = np.minimum(p0, p1), np.maximum(p0, p1)
    seen, out = set(), []
    for cx in range(int(lo[0] // CELL), int(hi[0] // CELL) + 1):
      for cy in range(int(lo[1] // CELL), int(hi[1] // CELL) + 1):
        for k in self.cells.get((cx, cy), ()):
          if k in seen:
            continue
          seen.add(k)
          a, b, one = self.rows[k]
          if {a, b} == set(own):
            continue
          pa, pb = self.osm.node_xy(a), self.osm.node_xy(b)
          d = pb - pa
          seg = float(np.hypot(*d))
          if seg < 1e-6 or abs(float(d @ u)) / seg > SQUARE:
            continue
          if seg < MIN_LINK:
            continue
          ahead = float(d @ arriving) / seg
          if abs(ahead) < ALIGN_ON:
            continue
          if ahead < 0:  # driven towards the junction from b to a
            if one == 1:
              continue
            a, b, pa, d = b, a, pb, -d
          elif one == -1:
            continue
          if z is not None and any(abs(self.heights[n] - z) > LEVEL for n in (a, b) if n in self.heights):
            continue
          hit = _intersect(pa, d, p0, p1 - p0)
          if hit is None:
            continue
          on = REACH - ON_REACH <= float((pa + d * hit[0] - p0) @ u) <= REACH + width + ON_REACH
          if on or abs(ahead) >= ALIGN:
            out.append((a, b, hit[0] * seg))
    return out


def _intersect(p, d, q, e) -> tuple[float, float] | None:
  """(t, v) where p + t d crosses q + v e, both 0..1; None where they don't."""
  den = d[0] * e[1] - d[1] * e[0]
  if abs(den) < 1e-12:
    return None
  w = q - p
  t = (w[0] * e[1] - w[1] * e[0]) / den
  v = (w[0] * d[1] - w[1] * d[0]) / den
  return (float(t), float(v)) if 0.0 <= t <= 1.0 and 0.0 <= v <= 1.0 else None


def _link(osm: OsmLanes, nodes, along: float) -> tuple[int, int, float] | None:
  """(a, b, m from a towards b) of the point `along` m out along a road's nodes from its junction's node: b the node
  nearer the junction, on a link at least MIN_LINK long; past the last node, the last link and a negative offset."""
  if len(nodes) < 2:
    return None
  pts = osm.xy[osm.data.index(nodes)]
  cum = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))))
  m = max(min(int(np.searchsorted(cum, along, side='left')), len(nodes) - 1), 1)  # between nodes m - 1 and m
  stepped = False
  while cum[m] - cum[m - 1] < MIN_LINK and m > 1:  # at the node: the link on in from it
    m, stepped = m - 1, True
  while cum[m] - cum[m - 1] < MIN_LINK and m < len(nodes) - 1:  # at the junction's own node: the link in to it
    m, stepped = m + 1, True
  offset = float(cum[m] - along)
  return int(nodes[m]), int(nodes[m - 1]), max(offset, 0.0) if stepped else offset


def _along_link(p, osm: OsmLanes, a: int, b: int) -> float | None:
  """m along the link from node a to node b to point p, None where p isn't on it."""
  pa, pb = osm.node_xy(a), osm.node_xy(b)
  ab = pb - pa
  length = float(np.hypot(*ab))
  if length < 1e-6:
    return None
  t = float(np.dot(np.asarray(p) - pa, ab)) / length ** 2
  if not -0.01 <= t <= 1.01 or float(np.hypot(*(pa + ab * np.clip(t, 0.0, 1.0) - p))) > ON_LINK:
    return None
  return t * length
