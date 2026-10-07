"""Junction geometry from OpenStreetMap ways and their lane cross-sections (osm_lanes.py), as osm2streets (A/B Street)
draws it: each road into a junction is trimmed back to where its kerbs meet its neighbours' kerbs, the junction's area
is the polygon from the trimmed road ends with a rounded kerb at each corner, and the stop lines go across the lanes
into it at its traffic signals, stop and give way signs. Only standard tags are read, so it works the same on any OSM
map as on our GTA V one.

- Junction nodes are where three or more roads meet, and only where they share the node: roads crossing over each
  other never make one. A node where a road's lanes part into ways side by side (fewer than three arms) isn't one.
- Junction nodes joined by a road shorter than CLUSTER_LINK (and their widest road's width), or whose trimmed ends
  would overlap along a road shorter than MERGE_LINK, make one junction, as where a divided road crosses another
  (osm2streets merges such short roads into the junction); the roads between them are inside it. Junctions further
  apart are trimmed back less, to fit.
- An arm is the road out of a junction, followed through nodes where only two roads meet. Arms that run side by side
  and overlap, as a turn lane mapped as its own way beside its road, are one arm: its kerbs are the outer ones.
- The area fans out from the junction's centre to its outline, which can fold back on itself round a junction of many
  nodes (fill the triangles, in_fan). Its kerbs go round the outside of the roads inside it.
- A stop line is at its node (`highway=traffic_signals` / `stop` / `give_way`), across the lanes towards the junction
  the node's direction tag (`traffic_signals:direction`, `direction`) faces, or towards the nearest junction without
  one; no nearer the junction than its mouth, and behind a crossing (`footway=crossing`) near it. Signals on a
  junction's own node stop every way into it at the mouth.

Geometry is in metres with y 90 degrees left of x; arms are sorted counterclockwise. "Left" of an arm is on the left
looking out of the junction along it.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, FORWARD, OsmLanes, offset_line, oneway_of

CLUSTER_LINK = 15.0  # m: junction nodes joined by a road this short (and shorter than their widest road is wide) are one junction
ARM_LENGTH = 60.0  # m of each road out of a junction that its geometry is worked out over
BACK = 30.0  # m each kerb is carried on straight back into the junction, to find where it meets the next
MAX_TRIM = 40.0  # m: a road is trimmed back no further than this from its junction node
CORNER_RADIUS = 6.0  # m: a kerb's radius round a junction's corner, at most ...
MIN_RADIUS = 2.0  # m: ... and at least, half the narrower road's width between
MAX_TANGENT = 9.0  # m from where two kerbs would meet to where the rounded corner starts, at most (acute corners)
MAX_PUSH = 6.0  # m a kerb is moved out at most to go round the roads inside a junction
BUNDLE_ANGLE = 12.0  # deg: arms heading within this of each other ...
BUNDLE_GAP = 0.5  # m: ... and less than this apart (or overlapping) are one road
STOP_REACH = 40.0  # m beyond a junction's mouth its stop lines can be
STOP_SETBACK = 0.5  # m: a stop line at the mouth is this far out from it
MERGE_GAP = 2.0  # m: junctions whose trimmed ends come closer than this along the road between them ...
MERGE_LINK = 20.0  # m: ... are one where that road is shorter than this; else both are trimmed less
MAX_SPAN = 60.0  # m across a junction's nodes, at most, from merging
CROSSING_WIDTH = 3.0  # m: a pedestrian crossing's painted width
CROSSING_REACH = 8.0  # m out from a junction's mouth: a crossing this near goes between its stop lines and it
CELL = 50.0  # m
STOPS = {'traffic_signals': 'stop', 'stop': 'stop', 'give_way': 'give_way'}
FREEWAY = frozenset({'motorway', 'motorway_link'})
U_TURN = 160.0  # deg: a move turning back more than this is a U-turn, left out
STRAIGHT = 30.0  # deg: a move turning less than this goes through
# the deg turned (left positive) along which each turn:lanes arrow points
ARROWS = {'through': (-STRAIGHT, STRAIGHT), 'slight_left': (10.0, 60.0), 'left': (STRAIGHT, 150.0), 'sharp_left': (110.0, U_TURN),
          'slight_right': (-60.0, -10.0), 'right': (-150.0, -STRAIGHT), 'sharp_right': (-U_TURN, -110.0),
          'merge_to_left': (-STRAIGHT, STRAIGHT), 'merge_to_right': (-STRAIGHT, STRAIGHT)}


class Poly:
  """A polyline carried on straight `back` m before its start and on past its end, by arc length s (0 at its start)."""
  def __init__(self, points, back: float = BACK, ahead: float = 200.0):
    p = np.asarray(points, float)[:, :2]
    p = p[np.concatenate(([True], np.hypot(*np.diff(p, axis=0).T) > 1e-6))]
    if len(p) < 2:
      raise ValueError('a polyline needs two distinct points')
    u0, u1 = _unit(p[1] - p[0]), _unit(p[-1] - p[-2])
    self.p = np.vstack([p[0] - u0 * back, p, p[-1] + u1 * ahead])
    seg = np.hypot(*np.diff(self.p, axis=0).T)
    self.s = np.concatenate(([0.0], np.cumsum(seg))) - back
    self.length = float(self.s[-2])  # of the polyline itself

  def at(self, s: float) -> np.ndarray:
    return np.array([np.interp(s, self.s, self.p[:, 0]), np.interp(s, self.s, self.p[:, 1])])

  def tangent(self, s: float) -> np.ndarray:
    k = int(np.clip(np.searchsorted(self.s, s, side='right') - 1, 0, len(self.p) - 2))
    return _unit(self.p[k + 1] - self.p[k])

  def project(self, q) -> float:
    """s of the nearest point to q."""
    a, ab = self.p[:-1], np.diff(self.p, axis=0)
    ab2 = np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-12)
    t = np.clip(np.einsum('ij,ij->i', np.asarray(q, float) - a, ab) / ab2, 0.0, 1.0)
    k = int(np.argmin(np.hypot(*(a + ab * t[:, None] - q).T)))
    return float(self.s[k] + t[k] * math.sqrt(ab2[k]))

  def between(self, s0: float, s1: float) -> np.ndarray:
    """The polyline from s0 to s1 (backwards where s1 < s0)."""
    lo, hi = min(s0, s1), max(s0, s1)
    inner = self.p[(self.s > lo) & (self.s < hi)]
    out = np.vstack([self.at(lo), inner, self.at(hi)])
    return out if s0 <= s1 else out[::-1]


def _unit(v) -> np.ndarray:
  return np.asarray(v, float) / max(float(np.hypot(*v)), 1e-9)


def _left(u) -> np.ndarray:
  return np.array([-u[1], u[0]])


def _cross(a, b) -> float:
  return float(a[0] * b[1] - a[1] * b[0])


def crossing(a: Poly, b: Poly, a_max: float, b_max: float) -> tuple[float, float, np.ndarray] | None:
  """Where two polylines cross (both no further than their max s): (s on a, s on b, point), the one nearest their
  starts; None where they don't."""
  ka, kb = np.flatnonzero(a.s[:-1] <= a_max), np.flatnonzero(b.s[:-1] <= b_max)
  if not len(ka) or not len(kb):
    return None
  a0, d1 = a.p[ka], a.p[ka + 1] - a.p[ka]
  b0, d2 = b.p[kb], b.p[kb + 1] - b.p[kb]
  r = b0[None] - a0[:, None]
  den = d1[:, None, 0] * d2[None, :, 1] - d1[:, None, 1] * d2[None, :, 0]
  ok = np.abs(den) > 1e-9
  den = np.where(ok, den, 1.0)
  t = (r[..., 0] * d2[None, :, 1] - r[..., 1] * d2[None, :, 0]) / den
  u = (r[..., 0] * d1[:, None, 1] - r[..., 1] * d1[:, None, 0]) / den
  hit = ok & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
  if not hit.any():
    return None
  i, j = np.nonzero(hit)
  sa = a.s[ka[i]] + t[i, j] * np.hypot(*d1[i].T)
  sb = b.s[kb[j]] + u[i, j] * np.hypot(*d2[j].T)
  keep = (sa <= a_max) & (sb <= b_max)
  if not keep.any():
    return None
  n = int(np.argmin(np.where(keep, sa + sb, np.inf)))
  return float(sa[n]), float(sb[n]), a0[i[n]] + d1[i[n]] * t[i[n], j[n]]


def in_fan(points, centre, polygon) -> np.ndarray:
  """Which points [K, 2] are in a junction's area: the triangles from its centre to each edge of its polygon (which
  can fold back on itself, round a junction of many nodes)."""
  p = np.asarray(points, float).reshape(-1, 2) - centre
  a, b = polygon - centre, np.roll(polygon, -1, axis=0) - centre
  ca = a[None, :, 0] * p[:, None, 1] - a[None, :, 1] * p[:, None, 0]  # side of each triangle's edges the points are on
  cb = p[:, None, 0] * b[None, :, 1] - p[:, None, 1] * b[None, :, 0]
  ab = (b - a)[None]
  cc = ab[..., 0] * (p[:, None, 1] - a[None, :, 1]) - ab[..., 1] * (p[:, None, 0] - a[None, :, 0])
  sign = np.sign(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])[None]
  return ((ca * sign >= 0) & (cb * sign >= 0) & (cc * sign >= 0) & (sign != 0)).any(1)


def clip_outside(line, areas: list[tuple[np.ndarray, np.ndarray]], min_len: float = 0.3) -> list[np.ndarray]:
  """The runs of a polyline [N, 2] outside all the areas [(centre, polygon)] (in_fan)."""
  line = np.asarray(line, float)[:, :2]
  if not areas or len(line) < 2:
    return [line] if len(line) >= 2 else []
  # where the line can cross an area's edge: its polygon's edges, and the spokes from its centre where it folds back
  edges_a = np.vstack([q for c, p in areas for q in (p, np.broadcast_to(c, p.shape))])
  edges_b = np.vstack([q for c, p in areas for q in (np.roll(p, -1, axis=0), p)])
  a, d = line[:-1], np.diff(line, axis=0)
  e = edges_b - edges_a
  r = edges_a[None] - a[:, None]
  den = d[:, None, 0] * e[None, :, 1] - d[:, None, 1] * e[None, :, 0]
  ok = np.abs(den) > 1e-12
  den = np.where(ok, den, 1.0)
  t = (r[..., 0] * e[None, :, 1] - r[..., 1] * e[None, :, 0]) / den
  u = (r[..., 0] * d[:, None, 1] - r[..., 1] * d[:, None, 0]) / den
  hit = ok & (t > 0) & (t < 1) & (u >= 0) & (u <= 1)
  # every point where the line crosses an edge, as (segment, t), then each piece between two kept where it's outside
  cuts = [(k, 0.0) for k in range(len(a))] + [(int(k), float(t[k, j])) for k, j in zip(*np.nonzero(hit), strict=True)]
  cuts.append((len(a) - 1, 1.0))
  cuts.sort()
  pts = np.array([a[k] + d[k] * tk for k, tk in cuts])
  mids = (pts[:-1] + pts[1:]) / 2
  out_mask = ~np.any([in_fan(mids, c, p) for c, p in areas], axis=0)
  pieces, run = [], []
  for i, keep in enumerate(out_mask):
    if keep:
      if not run:
        run = [pts[i]]
      run.append(pts[i + 1])
    elif run:
      pieces.append(np.array(run))
      run = []
  if run:
    pieces.append(np.array(run))
  return [p for p in pieces if np.hypot(*np.diff(p, axis=0).T).sum() >= min_len]


def densify(line: np.ndarray, step: float = 1.0) -> np.ndarray:
  """A polyline with points no more than step apart."""
  out = [line[:1]]
  for a, b in zip(line[:-1], line[1:], strict=True):
    n = max(int(np.ceil(np.hypot(*(b - a)) / step)), 1)
    out.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
  return np.vstack(out)


def push(kerb: np.ndarray, centre: np.ndarray, samples: np.ndarray, reach: float = 0.75) -> np.ndarray:
  """A kerb (its ends kept) moved out from the centre as far as any of the samples in the same direction (within
  `reach` m sideways), up to MAX_PUSH: round the outside of roads it would otherwise cut across."""
  k = densify(kerb)
  d, s = k - centre, samples - centre
  r, rs = np.hypot(*d.T), np.hypot(*s.T)
  a, as_ = np.arctan2(d[:, 1], d[:, 0]), np.arctan2(s[:, 1], s[:, 0])
  diff = np.abs((as_[None] - a[:, None] + np.pi) % (2 * np.pi) - np.pi)
  near = diff < reach / np.maximum(r, 1.0)[:, None]
  far = np.where(near, rs[None], 0.0).max(1)
  scale = np.where(far > r, np.minimum(far, r + MAX_PUSH) / np.maximum(r, 1e-6), 1.0)
  if len(scale) > 4:  # smoothed, so the kerb sweeps round them rather than following every corner
    padded = np.pad(scale, 2, mode='edge')
    scale = np.max([padded[s:s + len(scale)] for s in range(5)], axis=0)
    scale = np.convolve(np.pad(scale, 3, mode='edge'), np.ones(7) / 7, mode='valid')
  scale[0] = scale[-1] = 1.0
  return centre + d * scale[:, None]


def hull(pts: np.ndarray) -> np.ndarray:
  """The convex hull of points [K, 2], counterclockwise."""
  pts = np.unique(np.round(pts, 4), axis=0)
  if len(pts) < 3:
    return pts

  def half(points):
    out: list = []
    for q in points:
      while len(out) >= 2 and _cross(out[-1] - out[-2], q - out[-2]) <= 0:
        out.pop()
      out.append(q)
    return out[:-1]
  return np.array(half(pts) + half(pts[::-1]))


def bezier(p0, c, p1, step: float = 0.75) -> np.ndarray:
  n = max(int((np.hypot(*(c - p0)) + np.hypot(*(p1 - c))) / step), 2)
  t = np.linspace(0, 1, n + 1)[:, None]
  return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1


@dataclass
class Member:
  """One road out of a junction: the ways along it from the junction node [(way, along its direction)], its nodes, its
  line, its kerbs, and where it's trimmed (m along its line from its junction node)."""
  start: int
  ways: list[tuple[int, bool]]
  nodes: list[int]
  line: Poly
  left: Poly
  right: Poly
  edges: tuple[float, float]  # its kerbs at the junction, m right of its line looking out
  trim: float = 0.0

  @property
  def heading(self) -> float:
    d = min(10.0, self.line.length)
    return math.atan2(*(self.line.at(d) - self.line.at(0.0))[::-1])


@dataclass
class Arm:
  """Members side by side as one road out of a junction: its line is the widest member's, its kerbs the outer ones."""
  members: list[Member]
  line: Poly
  left: Poly
  right: Poly
  heading: float
  width: float  # m, its widest member's
  trim: float = 0.0
  cap: float = MAX_TRIM  # m: trimmed no further than this, short of the next junction

  def mouth(self) -> tuple[np.ndarray, np.ndarray]:
    """Its kerbs where it's trimmed: (right, left)."""
    p = self.line.at(self.trim)
    return self.right.at(self.right.project(p)), self.left.at(self.left.project(p))


@dataclass
class Stop:
  kind: str  # 'stop' or 'give_way'
  line: np.ndarray  # [2, 2] across the lanes towards the junction, right to left seen arriving
  member: Member
  along: float  # m along the member's line from its junction node
  area: np.ndarray  # the whole road from the junction's mouth out to the stop line, where no lane lines are painted
  signal: bool = False  # a traffic light's


@dataclass
class Junction:
  nodes: list[int]
  arms: list[Arm]
  polygon: np.ndarray  # [N, 2], counterclockwise
  kerbs: list[np.ndarray]  # the kerb round each corner, from one arm's mouth to the next's
  inside: set[int]  # ways inside it
  centre: np.ndarray  # its nodes' middle, which its area fans out from (in_fan)
  stops: list[Stop] = field(default_factory=list)

  @property
  def ways(self) -> set[int]:
    return {w for arm in self.arms for m in arm.members for w, _ in m.ways[:1]}


class Junctions:
  """The junctions of a map (OsmLanes): `junctions`, and `trims` {(way, node): m}, how far each way is trimmed back from
  its end at a junction node, along the arm it starts; `inside` the ways inside junctions. `roads(tags)` says which
  ways count (default: all the map's roads)."""
  def __init__(self, osm: OsmLanes, roads=None):
    self.osm = osm
    keep = roads or (lambda tags: True)
    self.ways = {w: v for w, v in osm.ways.items() if keep(v[0]) and len(v[1]) >= 2}
    self.steps: dict[int, list[tuple[int, int, bool]]] = {}  # node -> [(way, next node, along the way)]
    for wid, (_, refs) in self.ways.items():
      for i, n in enumerate(refs):
        if i > 0:
          self.steps.setdefault(n, []).append((wid, refs[i - 1], False))
        if i + 1 < len(refs):
          self.steps.setdefault(n, []).append((wid, refs[i + 1], True))
    self.junction_nodes = {n for n, s in self.steps.items() if len(s) >= 3}
    self.crossings = [osm.xy[osm.data.index(refs)] for tags, refs in osm.data.ways.values()
                      if tags.get('footway') == 'crossing' and len(refs) >= 2]
    self._crossing_cells = self._cells(self.crossings)
    self.junctions: list[Junction] = []
    self.trims: dict[tuple[int, int], float] = {}
    self.inside: set[int] = set()
    self._rules: tuple[dict, list] | None = None  # read from the map's relations when first needed
    self._build()

  @staticmethod
  def _cells(lines: list[np.ndarray]) -> dict[tuple[int, int], list[int]]:
    out: dict[tuple[int, int], list[int]] = {}
    for n, pts in enumerate(lines):
      lo, hi = pts.min(0) // CELL, pts.max(0) // CELL
      for cx in range(int(lo[0]), int(hi[0]) + 1):
        for cy in range(int(lo[1]), int(hi[1]) + 1):
          out.setdefault((cx, cy), []).append(n)
    return out

  @staticmethod
  def _near(lines, cells, p) -> list[np.ndarray]:
    return [lines[n] for n in cells.get((int(p[0] // CELL), int(p[1] // CELL)), ())]

  def crossing_lines(self) -> list[np.ndarray]:
    """The crossings (`footway=crossing`) where they're on the road: between the kerbs of the roads they cross."""
    ids = list(self.ways)
    lines = [self.osm.way_points(w) for w in ids]
    cells = self._cells(lines)
    out = []
    for c in self.crossings:
      cp = Poly(c, 0.0, 0.0)
      near = {n for p in c for n in cells.get((int(p[0] // CELL), int(p[1] // CELL)), ())}
      spans = []
      for n in sorted(near):
        road = self.osm.lanes(ids[n])
        hits = []
        for off in road.edges(FORWARD):
          try:
            kerb = Poly(offset_line(lines[n], off), 0.0, 0.0)
          except ValueError:
            break
          hit = crossing(cp, kerb, math.inf, math.inf)
          if hit is None:
            break
          hits.append(hit[0])
        if len(hits) == 2:
          spans.append((min(hits), max(hits)))
      if not spans:
        out.append(c)
        continue
      spans.sort()
      merged = [list(spans[0])]
      for a, b in spans[1:]:
        if a <= merged[-1][1] + 0.5:
          merged[-1][1] = max(merged[-1][1], b)
        else:
          merged.append([a, b])
      out += [cp.between(a, b) for a, b in merged if b - a > 0.5]
    return out

  def widest(self, node: int) -> float:
    return max(self.osm.lanes(w).width for w, _, _ in self.steps[node])

  def joint(self, node: int, reach: float = 0.6) -> np.ndarray | None:
    """Where roads meet at a node outside any junction, as where a road's width changes: the hull of their ends, which
    fills the wedge between flat-ended roads meeting at an angle (counterclockwise polygon), None for a dead end."""
    steps = self.steps.get(node, ())
    if len(steps) < 2:
      return None
    p, pts = self.osm.node_xy(node), []
    for w, nxt, fwd in steps:
      u = _unit(self.osm.node_xy(nxt) - p)
      lo, hi = self.osm.lanes(w).edges(FORWARD if fwd else BACKWARD)
      r = -_left(u)
      pts += [p + r * lo, p + r * hi, p + u * reach + r * lo, p + u * reach + r * hi]
    return hull(np.array(pts))

  # *** arms ***

  def walk(self, node: int, step: tuple[int, int, bool], max_len: float) -> tuple[list[tuple[int, bool]], list[int], float]:
    """The ways on from a node by a step, through nodes where only two roads meet, to a junction or max_len."""
    ways, nodes, length, cur = [], [node], 0.0, node
    w, nxt, fwd = step
    while True:
      ways.append((w, fwd))
      nodes.append(nxt)
      length += float(np.hypot(*(self.osm.node_xy(nxt) - self.osm.node_xy(cur))))
      on = [s for s in self.steps.get(nxt, ()) if not (s[0] == w and s[1] == cur)]
      if length >= max_len or nxt in self.junction_nodes or len(on) != 1 or on[0][1] in nodes:
        return ways, nodes, length
      cur, (w, nxt, fwd) = nxt, on[0]

  def member(self, node: int, step) -> Member | None:
    ways, nodes, _ = self.walk(node, step, ARM_LENGTH)
    pts = self.osm.xy[self.osm.data.index(nodes)]
    if len(pts) < 2 or np.hypot(*(pts[-1] - pts[0])) < 1e-3:
      return None
    edges = [self.osm.lanes(w).edges(FORWARD if fwd else BACKWARD) for w, fwd in ways]

    def kerb(side):  # each run of ways as wide offset on its own, stepping where the width changes
      out, k = [], 0
      while k < len(edges):
        k1 = k
        while k1 + 1 < len(edges) and abs(edges[k1 + 1][side] - edges[k][side]) < 0.01:
          k1 += 1
        out.append(offset_line(pts[k:k1 + 2], edges[k][side]))
        k = k1 + 1
      return np.vstack(out)
    try:
      line, left, right = Poly(pts), Poly(kerb(0)), Poly(kerb(1))
    except ValueError:
      return None
    return Member(node, ways, nodes, line, left, right, edges[0])

  # *** junctions ***

  def _build(self):
    # a node where a road's lanes split into ways side by side, with fewer than three arms, isn't a junction
    cache: dict[tuple, Junction | None] = {((n,), ()): self.junction([n]) for n in self.junction_nodes}
    self.junction_nodes = {n for n in self.junction_nodes if cache[((n,), ())] is not None}
    parent = {n: n for n in self.junction_nodes}

    def find(n):
      while parent[n] != n:
        parent[n] = parent[parent[n]]
        n = parent[n]
      return n

    for n in self.junction_nodes:
      for step in self.steps[n]:
        _, nodes, length = self.walk(n, step, CLUSTER_LINK)
        if nodes[-1] in self.junction_nodes and nodes[-1] != n and length < min(CLUSTER_LINK, max(self.widest(n), self.widest(nodes[-1]))):
          parent[find(n)] = find(nodes[-1])
    # junctions whose trimmed ends overlap are one where the road between them is short; else both are trimmed less
    caps: dict[tuple[int, int], float] = {}  # (node, way) -> how far at most the road out of the node is trimmed
    for _ in range(4):
      groups: dict[int, list[int]] = {}
      for n in self.junction_nodes:
        groups.setdefault(find(n), []).append(n)
      built = {}
      for k, v in groups.items():
        mine = {key: c for key, c in caps.items() if key[0] in v}
        key = (tuple(sorted(v)), tuple(sorted(mine.items())))
        if key not in cache:
          cache[key] = self.junction(sorted(v), mine)
        built[k] = cache[key]
      ends = {(m.start, m.ways[0][0]): m for j in built.values() if j for arm in j.arms for m in arm.members}
      changed = False
      for k, j in built.items():
        for arm in (j.arms if j else []):
          for m in arm.members:
            end = m.nodes[-1]
            if end not in parent or find(end) == find(k):
              continue
            other = ends.get((end, m.ways[-1][0]))
            room = m.line.length - MERGE_GAP
            if other is None or m.trim + other.trim <= room + 0.01:
              continue
            a, b = find(k), find(end)
            both = np.array([self.osm.node_xy(q) for q in groups[a] + groups[b]])
            if m.line.length < MERGE_LINK and np.hypot(*(both.max(0) - both.min(0))) < MAX_SPAN:
              parent[b] = a
              groups[a] = groups[a] + groups.pop(b)
            else:
              share = max(room, 0.0) / (m.trim + other.trim)
              caps[(m.start, m.ways[0][0])] = m.trim * share
              caps[(end, m.ways[-1][0])] = other.trim * share
            changed = True
      if not changed:
        break
    self.junctions = [j for j in built.values() if j]
    for j in self.junctions:
      self.inside |= j.inside
      for arm in j.arms:
        for m in arm.members:
          self.trims[(m.ways[0][0], m.start)] = m.trim
    self._stops()

  def junction(self, nodes: list[int], caps: dict[tuple[int, int], float] | None = None) -> Junction | None:
    """A junction of these nodes, or None where its roads make fewer than three arms (as where a road's lanes split).
    `caps` {(node, way): m} limits how far the roads out of it are trimmed."""
    inner = set(nodes)
    members, links, seen = [], [], set()
    for n in nodes:
      for step in self.steps[n]:
        ways, chain, _ = self.walk(n, step, ARM_LENGTH)
        if chain[-1] in inner and chain[-1] in self.junction_nodes:
          links.append((ways, chain))
          continue
        key = (n, step[0], step[1])
        if key in seen:
          continue
        seen.add(key)
        m = self.member(n, step)
        if m is not None:
          members.append(m)
    if len(members) < 3 or all(self.ways[m.ways[0][0]][0].get('highway') in FREEWAY for m in members):
      return None  # freeways only merge and part, with no junction between
    centre = np.mean([self.osm.node_xy(n) for n in nodes], axis=0)

    def bearing(m):
      p = m.line.at(min(10.0, m.line.length))
      return math.atan2(p[1] - centre[1], p[0] - centre[0])
    members.sort(key=bearing)
    arms = self.bundle(members)
    if len(arms) < 3:
      return None
    for arm in arms:
      arm.cap = min([MAX_TRIM] + [caps[k] for m in arm.members if (k := (m.start, m.ways[0][0])) in (caps or {})])
    polygon, kerbs = self.outline(arms, self.shape(arms))
    for arm in arms:
      p = arm.line.at(arm.trim)
      for m in arm.members:
        m.trim = max(m.line.project(p), 0.0)
    # a road between the junction's own nodes is inside it, unless it runs mostly outside its area, as a slip road
    # round a corner: that one is drawn as a road, its kerbs cut where it's inside
    inside = set()
    for ways, chain in links:
      pts = self.osm.xy[self.osm.data.index(chain)]
      mids = np.vstack([pts[:-1] + (pts[1:] - pts[:-1]) * t for t in (0.25, 0.5, 0.75)])
      if in_fan(mids, centre, polygon).mean() >= 0.5:
        inside |= {w for w, _ in ways}
    if inside:  # the kerbs go round the roads inside where they reach out past the corners
      samples = []
      for w in inside:
        pts = self.osm.way_points(w)
        for off in self.osm.lanes(w).edges(FORWARD):
          samples.append(densify(offset_line(pts, off)))
      kerbs = [push(k, centre, np.vstack(samples)) for k in kerbs]
      polygon = np.vstack([np.vstack([kerbs[i - 1][-1:], kerbs[i][:-1]]) for i in range(len(kerbs))])
    return Junction(nodes, arms, polygon, kerbs, inside, centre)

  def bundle(self, members: list[Member]) -> list[Arm]:
    """Groups members (counterclockwise) running side by side into arms."""
    n = len(members)
    joined = [self.side_by_side(members[i], members[(i + 1) % n]) for i in range(n)]
    if all(joined):
      return []
    first = next(i for i in range(n) if not joined[i])  # start after a break, so groups don't wrap round
    groups, cur = [], []
    for k in range(n):
      i = (first + 1 + k) % n
      cur.append(members[i])
      if not joined[i]:
        groups.append(cur)
        cur = []
    arms = []
    for g in groups:
      widest = max(g, key=lambda m: m.edges[1] - m.edges[0])
      d = min(10.0, widest.line.length)
      p, nl = widest.line.at(d), _left(widest.line.tangent(d))

      def lateral(poly, p=p, nl=nl):
        return float((poly.at(poly.project(p)) - p) @ nl)
      left = max(g, key=lambda m: lateral(m.left)).left
      right = min(g, key=lambda m: lateral(m.right)).right
      arms.append(Arm(g, widest.line, left, right, widest.heading, widest.edges[1] - widest.edges[0]))
    return arms

  @staticmethod
  def side_by_side(a: Member, b: Member) -> bool:
    turn = abs((b.heading - a.heading + math.pi) % (2 * math.pi) - math.pi)
    if math.degrees(turn) > BUNDLE_ANGLE:
      return False
    d = min(10.0, a.line.length)
    p = a.left.at(a.left.project(a.line.at(d)))
    q = b.right.at(b.right.project(p))
    return float((q - p) @ _left(a.line.tangent(d))) < BUNDLE_GAP

  @staticmethod
  def shape(arms: list[Arm]) -> list[tuple[float, float, np.ndarray] | None]:
    """Sets each arm's trim; returns each corner's kerb, between an arm and the next counterclockwise: (where it rounds
    from on this arm's left kerb, where to on the next's right kerb, the point the two kerbs meet at), or None where
    they don't meet ahead of the junction and the kerb between them is straight."""
    n = len(arms)
    trims = [0.0] * n
    corners: list[tuple[float, float, np.ndarray] | None] = []
    for i in range(n):
      a, b = arms[i], arms[(i + 1) % n]
      gap = (b.heading - a.heading) % (2 * math.pi)
      hit = crossing(a.left, b.right, MAX_TRIM, MAX_TRIM) if 1e-3 < gap < math.pi - 1e-3 else None
      if hit is None:
        corners.append(None)
        continue
      sa, sb, x = hit
      radius = min(max(min(a.width, b.width) / 2, MIN_RADIUS), CORNER_RADIUS)
      t = min(radius / math.tan(gap / 2), MAX_TANGENT, a.cap - a.line.project(x), b.cap - b.line.project(x))
      ta, tb = sa + max(t, 0.0), sb + max(t, 0.0)
      trims[i] = max(trims[i], a.line.project(a.left.at(ta)))
      trims[(i + 1) % n] = max(trims[(i + 1) % n], b.line.project(b.right.at(tb)))
      corners.append((ta, tb, x))
    for arm, t in zip(arms, trims, strict=True):
      arm.trim = min(max(t, 0.0), arm.cap)
    return corners

  @staticmethod
  def outline(arms: list[Arm], corners) -> tuple[np.ndarray, list[np.ndarray]]:
    """The junction's polygon, and its kerb round each corner from one arm's mouth to the next's."""
    n = len(arms)
    mouths = [arm.mouth() for arm in arms]
    pieces, kerbs = [], []
    for i in range(n):
      a, b = arms[i], arms[(i + 1) % n]
      l_pt, r_next = mouths[i][1], mouths[(i + 1) % n][0]
      corner = corners[i]
      if corner is None:
        kerb = np.vstack([l_pt, r_next])
      else:
        ta, tb, x = corner
        sl, sr = a.left.project(l_pt), b.right.project(r_next)
        kerb = np.vstack([a.left.between(sl, min(ta, sl)), bezier(a.left.at(ta), x, b.right.at(tb))[1:-1],
                          b.right.between(min(tb, sr), sr)])
      kerbs.append(kerb)
      pieces += [mouths[i][0][None], kerb[:-1]]
    return np.vstack(pieces), kerbs

  # *** stop lines ***

  def _stops(self):
    at: dict[int, list[tuple[Junction, Member, int]]] = {}
    for j in self.junctions:
      for arm in j.arms:
        for m in arm.members:
          for k, node in enumerate(m.nodes):
            at.setdefault(node, []).append((j, m, k))
    tags_of = self.osm.data.node_tags
    for node, tags in tags_of.items():
      kind = STOPS.get(tags.get('highway', ''))
      if kind is None:
        continue
      facing = tags.get('traffic_signals:direction') or tags.get('direction')
      found = []
      for j, m, k in at.get(node, ()):
        if k == 0:  # on the junction's own node: every way in, at its mouth
          continue
        along = self._along(m, k)
        if along > m.trim + STOP_REACH:
          continue
        if facing in ('forward', 'backward'):
          wanted = facing == 'backward'  # the way runs out of the junction where traffic towards it goes backward
          if not any(m.ways[q][1] == wanted for q in (k - 1, k) if q < len(m.ways)):
            continue
        found.append((along, j, m))
      for along, j, m in sorted(found, key=lambda f: f[0])[:1 if facing not in ('forward', 'backward') else None]:
        self._add_stop(j, m, kind, along, tags['highway'] == 'traffic_signals')
    for j in self.junctions:
      tags = next((tags_of[n] for n in j.nodes if STOPS.get(tags_of.get(n, {}).get('highway', ''))), None)
      if tags is not None:
        for arm in j.arms:
          for m in arm.members:
            if not any(s.member is m for s in j.stops):
              self._add_stop(j, m, STOPS[tags['highway']], 0.0, tags['highway'] == 'traffic_signals')

  @staticmethod
  def _along(m: Member, k: int) -> float:
    pts = m.line.p[1:-1]
    return float(np.hypot(*np.diff(pts[:k + 1], axis=0).T).sum()) if k else 0.0

  def _add_stop(self, j: Junction, m: Member, kind: str, along: float, signal: bool = False):
    w, fwd = m.ways[0]
    spans = self.osm.lanes(w).ours(BACKWARD if fwd else FORWARD)  # the lanes towards the junction
    if not spans:
      return
    s = max(along, m.trim + STOP_SETBACK)
    for c in self._near(self.crossings, self._crossing_cells, m.line.at(s)):  # a stop line goes before a crossing
      hit = crossing(Poly(c, 0.0, 0.0), m.line, math.inf, max(s, m.trim + CROSSING_REACH) + CROSSING_WIDTH)
      if hit is not None and hit[1] > m.trim - CROSSING_WIDTH:
        s = max(s, hit[1] + CROSSING_WIDTH / 2 + STOP_SETBACK)
    p, u = m.line.at(s), m.line.tangent(s)
    right = _left(u)  # right of the direction of travel into the junction
    lo, hi = min(sp.left for sp in spans), max(sp.right for sp in spans)
    line = np.array([p + right * hi, p + right * lo])
    e_lo, e_hi = m.edges  # looking out: its right kerb is on the left arriving
    q = m.line.at(m.trim)
    uq = _left(m.line.tangent(m.trim))
    area = np.array([q - uq * e_hi, p + right * (-e_hi), p + right * (-e_lo), q - uq * e_lo])
    j.stops.append(Stop(kind, line, m, s, area, signal))

  # *** movements ***

  def movements(self, j: Junction) -> list['Movement']:
    """Every lane-to-lane move through a junction, with its path (see Movement)."""
    if self._rules is None:
      self._rules = rules(self.osm.data.relations)
    restricted, connects = self._rules
    stop_at = {id(s.member): s.along for s in j.stops}
    arm_of = {id(m): k for k, arm in enumerate(j.arms) for m in arm.members}
    members = [m for arm in j.arms for m in arm.members]
    out = []
    for m_in in members:
      w_in, fwd_in = m_in.ways[0]
      lanes_in = self.osm.lanes(w_in).ours(BACKWARD if fwd_in else FORWARD)  # towards the junction
      if not lanes_in:
        continue
      reach = self._reachable(j, m_in.start)
      heading_in = -m_in.line.tangent(0.0)
      exits = []  # (deg turned, member, its lanes out)
      for m_out in members:
        if arm_of[id(m_out)] == arm_of[id(m_in)] or m_out.start not in reach:
          continue
        w_out, fwd_out = m_out.ways[0]
        lanes_out = self.osm.lanes(w_out).ours(FORWARD if fwd_out else BACKWARD)
        if not lanes_out:
          continue
        u = m_out.line.tangent(0.0)
        turned = math.degrees(math.atan2(_cross(heading_in, u), float(heading_in @ u)))  # left positive
        if abs(turned) > U_TURN or forbidden(restricted, m_in, m_out):
          continue
        exits.append((turned, m_out, lanes_out))
      if not exits:
        continue
      pairs = lane_moves([s.lane.turns for s in lanes_in], [(t, len(lo)) for t, _, lo in exits])
      ways_in = {w for w, _ in m_in.ways}
      for k, (turned, m_out, lanes_out) in enumerate(exits):
        given = next((c for f, t, c in connects if f in ways_in and t == m_out.ways[0][0]), None)
        for a, b in (given if given is not None else pairs[k]):
          if a < len(lanes_in) and b < len(lanes_out):
            out.append(self._movement(m_in, a, lanes_in[a], stop_at.get(id(m_in), m_in.trim), m_out, b, lanes_out[b], turned))
    return out

  def _reachable(self, j: Junction, node: int) -> set[int]:
    """The junction's nodes a car at one of them can drive to along the roads inside it, keeping to their one-way rules."""
    seen, stack = {node}, [node]
    while stack:
      n = stack.pop()
      for w, nxt, along in self.steps.get(n, ()):
        oneway = oneway_of(self.ways[w][0])
        if w in j.inside and nxt not in seen and (oneway == 0 or (oneway == 1) == along):
          seen.add(nxt)
          stack.append(nxt)
    return seen

  @staticmethod
  def _movement(m_in: Member, a: int, span_in, s_in: float, m_out: Member, b: int, span_out, turned: float) -> 'Movement':
    u_in, u_out = m_in.line.tangent(s_in), m_out.line.tangent(m_out.trim)
    p0 = m_in.line.at(s_in) + _left(u_in) * span_in.centre  # arriving, the lane's right is left of the road looking out
    p1 = m_out.line.at(m_out.trim) - _left(u_out) * span_out.centre
    kind = 'through' if abs(turned) <= STRAIGHT else 'left' if turned > 0 else 'right'
    return Movement(m_in, a, m_out, b, kind, turned, lane_curve(p0, -u_in, p1, u_out))


@dataclass
class Movement:
  """A move from lane `lane_in` (of those arriving on `into`, numbered from the left) to lane `lane_out` (of those
  leaving on `out`): 'left', 'through' or 'right' (`turned` deg, left positive), and its path, the lane centre from the
  incoming lane's stop line (or the junction's mouth) to the outgoing lane at the mouth."""
  into: Member
  lane_in: int
  out: Member
  lane_out: int
  kind: str
  turned: float
  path: np.ndarray


def rules(relations: dict) -> tuple[dict, list]:
  """Turn restrictions {from way: [(restriction, to way)]} and connectivity [(from way, to way, [(lane in, lane out)])]
  (lanes from 0 at the left; optional lanes, `(n)`, count; both-ways lanes, `bw`, are left out)."""
  restricted: dict[int, list[tuple[str, int]]] = {}
  connects = []
  for tags, members in relations.values():
    wf = [r for t, r, role in members if t == 'w' and role == 'from']
    wt = [r for t, r, role in members if t == 'w' and role == 'to']
    if len(wf) != 1 or len(wt) != 1:
      continue
    if tags.get('type') == 'restriction' and tags.get('restriction', '').startswith(('no_', 'only_')):
      restricted.setdefault(wf[0], []).append((tags['restriction'], wt[0]))
    elif tags.get('type') == 'connectivity':
      pairs = []
      for group in tags.get('connectivity', '').split('|'):
        a, _, bs = group.partition(':')
        a = a.strip('()')
        for b in bs.split(','):
          b = b.strip('()')
          if a.isdigit() and b.isdigit():
            pairs.append((int(a) - 1, int(b) - 1))
      if pairs:
        connects.append((wf[0], wt[0], pairs))
  return restricted, connects


def forbidden(restricted: dict, m_in: Member, m_out: Member) -> bool:
  """Whether a turn restriction from a way along the road in forbids the move to a way along the road out (by any via)."""
  outs = {w for w, _ in m_out.ways}
  for w, _ in m_in.ways:
    for kind, to in restricted.get(w, ()):
      if kind.startswith('no_') and to in outs:
        return True
      if kind.startswith('only_') and to not in outs:
        return True
  return False


def lane_moves(turns: list[frozenset[str]], exits: list[tuple[float, int]]) -> list[list[tuple[int, int]]]:
  """For each way out ((deg turned, its lanes)), the moves [(lane in, lane out)] to it of the lanes in, given each lane's
  turn:lanes arrows (empty for none). With arrows, a lane takes the ways out they point along, or the nearest one on
  that side where none does. Without, the leftmost lane also turns left, the rightmost right, and the rest go through
  (at a T, half each way). The lanes taking a way out go to its lanes in order from the side they turn to (through,
  spread evenly)."""
  n = len(turns)
  straight = [k for k, (t, _) in enumerate(exits) if abs(t) <= STRAIGHT]
  lefts = [k for k, (t, _) in enumerate(exits) if t > STRAIGHT]
  rights = [k for k, (t, _) in enumerate(exits) if t < -STRAIGHT]
  take: list[set[int]] = [set() for _ in range(n)]
  if any(turns):
    for i, ts in enumerate(turns):
      for turn in (ts or {'through'}) & ARROWS.keys():
        lo, hi = ARROWS[turn]
        hit = [k for k, (t, _) in enumerate(exits) if lo <= t <= hi]
        if not hit:
          side = [k for k, (t, _) in enumerate(exits) if (t > 0) == (lo + hi > 0)]
          hit = [min(side, key=lambda k: abs(exits[k][0] - (lo + hi) / 2))] if side else []
        take[i] |= set(hit)
  elif n == 1:
    take[0] = set(range(len(exits)))
  else:
    for i in range(n):
      take[i] |= set(straight)
    if straight:
      take[0] |= set(lefts)
      take[-1] |= set(rights)
    else:
      half = (n + 1) // 2 if lefts and rights else n
      for i in range(n):
        take[i] |= set(lefts) if (i < half and lefts) or not rights else set(rights)
  out = []
  for k, (turned, n_out) in enumerate(exits):
    ins = [i for i in range(n) if k in take[i]]
    m = len(ins)
    if turned > STRAIGHT:  # from the left
      out.append([(i, min(q, n_out - 1)) for q, i in enumerate(ins)])
    elif turned < -STRAIGHT:  # from the right
      out.append([(i, max(n_out - m + q, 0)) for q, i in enumerate(ins)])
    else:
      out.append([(i, round(q * (n_out - 1) / (m - 1)) if m > 1 else min(i, n_out - 1)) for q, i in enumerate(ins)])
  return out


def lane_curve(p0, u0, p1, u1, step: float = 1.0) -> np.ndarray:
  """A smooth path from p0 heading u0 to p1 heading u1 (unit vectors): a cubic Bezier whose handles reach 0.55 of the way
  towards where the two headings' lines meet (close to a circular arc round a right angle), or a third of the way
  across where they don't meet ahead."""
  d = float(np.hypot(*(p1 - p0)))
  ka = kb = d / 3
  if abs(_cross(u0, u1)) > 0.05:
    a, b = np.linalg.solve(np.array([u0, u1]).T, p1 - p0)  # p0 + a u0 = corner = p1 - b u1
    if a > 0 and b > 0:
      ka, kb = 0.55 * a, 0.55 * b
  c0, c1 = p0 + u0 * ka, p1 - u1 * kb
  t = np.linspace(0, 1, max(int(d / step), 4) + 1)[:, None]
  return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * c0 + 3 * (1 - t) * t ** 2 * c1 + t ** 3 * p1
