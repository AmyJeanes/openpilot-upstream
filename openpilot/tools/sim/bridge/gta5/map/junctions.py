"""Junction geometry from OpenStreetMap ways and their lane cross-sections (osm_lanes.py), as osm2streets (A/B Street)
draws it: each road into a junction is trimmed back to where its kerbs meet its neighbours' kerbs, the junction's area
is the polygon from the trimmed road ends with a rounded kerb at each corner, and the stop lines go across the lanes
into it at its traffic signals, stop and give way signs. Only standard tags are read, so it works the same on any OSM
map as on our GTA V one.

- Junction nodes are where three or more roads meet, and only where they share the node: roads crossing over each
  other never make one, nor GTA's two-way links across a motorway's median or between its parallel chains (crossovers).
  A node where a road's lanes part into ways side by side (fewer than three arms) isn't one, nor
  one where a one-way lane only leaves a two-way road or joins it on its own side, at a shallow angle (LANE_SPLIT), and
  stays beside it (BESIDE_REACH), as GTA lays a turn bay as a way of its own with lane changes to and from it.
- Junction nodes joined by a road shorter than CLUSTER_LINK (and their widest road's width), or whose trimmed ends
  would overlap along a road shorter than MERGE_LINK, make one junction, as where a divided road crosses another
  (osm2streets merges such short roads into the junction); the roads between them are inside it. Junctions further
  apart are trimmed back less, to fit. So do the nodes on a divided road's two carriageways joined by a two-way road
  across its median shorter than MEDIAN_LINK, as where a side road meets it at a gap in the median, and three junction
  nodes joined each to each by roads shorter than TRIANGLE_LINK, two of them one-way along some of their length, that
  make a junction of three arms, as GTA's triangle of one-way slips where a side road meets a main road (as junctions
  of their own, their areas lay across each other and their kerbs looped through it).
- An arm is the road out of a junction, followed through nodes where only two roads meet. Arms that run side by side
  and overlap, as a turn lane mapped as its own way beside its road, are one arm: its kerbs are the outer ones.
- A divided road's two carriageways out of a junction side by side, one in and one out, are trimmed back as far as each
  other, so the area ends square across both at its median's nose.
- The area fans out from the junction's centre to its outline, which can fold back on itself round a junction of many
  nodes (fill the triangles, in_fan). Its kerbs go round the outside of the roads inside it.
- A stop line is at its node (`highway=traffic_signals` / `stop` / `give_way`), across the lanes towards the junction
  the node's direction tag (`traffic_signals:direction`, `direction`) faces, or towards the nearest junction without
  one (nearest past its mouth), and for one junction only: where the tag faces two (the node's ways drawn opposite ways
  out of it), the one whose mouth is nearer; no nearer the junction than its mouth, and behind a crossing
  (`footway=crossing`) near it. Signals on a
  junction's own node stop every way into it at the mouth (a directed one, those it faces), but for a directed one
  already placed on another junction's approach (GTA's light where a slip parts, for the junction ahead). A stop line surveyed where it's painted
  (`source:position=survey`) is drawn at its node, and its road is trimmed back no further than that; one nearer
  than where its kerbs meet the next roads' (painted level with the road it meets) is inside the junction's area,
  shaped as without it, and lies along that road's edge.
- Kerbs aren't drawn within ISLAND_REACH of a painted island (`traffic_calming=painted_island`) or a road surface
  area (`area:highway`) at their height (off_islands): not of a road above or below (a ramp over a motorway).
- A road carried straight on through a junction (`Junction.through`): where the junction has no traffic signals, the
  road has no stop or give way line or crossing into it, nothing of a higher class and wider meets it there, and its
  lines meet the same lines of the road on the far side at the junction's node (or, between two of its nodes, of the
  ways inside it it runs on along, as through a slip triangle) (a turn lane's line, which doesn't, is
  left out). One road per junction, the highest class, then the widest; none where another as important crosses it (a
  crossroads of equals). Where it's a priority road both sides (`priority_road=designated` / `yes_unposted`), or only
  minor roads meet it (service roads, tracks, one-lane slips, narrower than MINOR_SHARE of it), its lines
  are painted on across the junction (`Junction.carried`, with the ways after its first that the area still reaches
  into), as a main road's centre line runs on past a side road. The
  junction's area stays the whole of where its roads meet, the through road's lanes too: traffic turning out of a side
  road crosses them, and the moves, trims and stop lines are worked out over it.

Geometry is in metres with y 90 degrees left of x; arms are sorted counterclockwise. "Left" of an arm is on the left
looking out of the junction along it.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, CENTRE, DIVIDER, FORWARD, MEDIAN, OsmLanes, offset_line, oneway_of

CLUSTER_LINK = 15.0  # m: junction nodes joined by a road this short (and shorter than their widest road is wide) are one junction
MEDIAN_LINK = 25.0  # m: a two-way road this short across a divided road's median joins the junctions on its carriageways
ARM_LENGTH = 60.0  # m of each road out of a junction that its geometry is worked out over
BACK = 30.0  # m each kerb is carried on straight back into the junction, to find where it meets the next
MAX_TRIM = 40.0  # m: a road is trimmed back no further than this from its junction node
CORNER_RADIUS = 6.0  # m: a kerb's radius round a junction's corner, at most ...
MIN_RADIUS = 2.0  # m: ... and at least, half the narrower road's width between
MAX_TANGENT = 9.0  # m from where two kerbs would meet to where the rounded corner starts, at most (acute corners)
MAX_PUSH = 6.0  # m a kerb is moved out at most to go round the roads inside a junction
BUNDLE_ANGLE = 12.0  # deg: arms heading within this of each other ...
BUNDLE_GAP = 0.5  # m: ... and less than this apart (or overlapping) are one road
MEDIAN_REACH = 25.0  # m apart at most: a divided road's carriageways out of a junction, its median's nose squared across
STOP_REACH = 40.0  # m beyond a junction's mouth its stop lines can be
STOP_SETBACK = 0.5  # m: a stop line at the mouth is this far out from it
STOP_SKEW = 40.0  # deg off square across its road a stop line along the road it meets may be
MERGE_GAP = 2.0  # m: junctions whose trimmed ends come closer than this along the road between them ...
MERGE_LINK = 20.0  # m: ... are one where that road is shorter than this; else both are trimmed less
MAX_SPAN = 60.0  # m across a junction's nodes, at most, from merging
TRIANGLE_LINK = 40.0  # m: three junction nodes joined each to each by roads shorter than this, two partly one-way ...
TRIANGLE_SPAN = 40.0  # m: ... and no further apart than this, are one junction
CROSSING_WIDTH = 3.0  # m: a pedestrian crossing's painted width
CROSSING_REACH = 8.0  # m out from a junction's mouth: a crossing this near goes between its stop lines and it
ISLAND_REACH = 1.0  # m round a painted island (traffic_calming=painted_island) within which no kerb is drawn
ISLAND_LEVEL = 3.0  # m: an island or road surface further above or below a kerb is another level's
CELL = 50.0  # m
STOPS = {'traffic_signals': 'stop', 'stop': 'stop', 'give_way': 'give_way'}
FREEWAY = frozenset({'motorway', 'motorway_link'})
CROSSOVER_M = 60.0  # m: a two-way link between motorway carriageways no longer than this is a crossover, not a road
MERGE_FLOW = 45.0  # deg: one-way roads all running within this of one heading only merge and part, with no junction
LANE_SPLIT = 50.0  # deg: a one-way lane leaving or joining a two-way road within this of its traffic, on its side ...
BESIDE_REACH = 30.0  # m: ... and beside the road this far along (within its kerb and the lane's width), only parts or merges
FREEWAY_FLOW = 60.0  # deg: ... or of this where one is a freeway's (GTA lays a diverge's lane changes as short links across it)
U_TURN = 160.0  # deg: a move turning back more than this is a U-turn, left out
LANE_CHANGE_M = 40.0  # m: GTA's lane changes between carriageways are 10-25 m long
BRANCH = 10.0  # deg more a lane change turns off a carriageway than the carriageway turns going on
SAME_WAY = 75.0  # deg between two ways' headings running the same way (GTA's lane changes cut across at up to ~60)
MEET = 0.6  # m: lines of a road either side of a junction this near each other at its node are one line carried across
CLASSES = ('motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'living_street', 'service', 'track')
PRIORITY = ('designated', 'yes_unposted')  # priority_road=*: a priority road, signed or not
MINOR = ('service', 'track')  # roads whose junctions with a through road leave its lines painted on across
MINOR_SHARE = 0.4  # of the through road's width a minor road meeting it is at most
FLIPPED = {'dashed_solid': 'solid_dashed', 'solid_dashed': 'dashed_solid'}  # a line's halves seen from the other way
STRAIGHT = 30.0  # deg: a move turning less than this goes through
# the deg turned (left positive) along which each turn:lanes arrow points
ARROWS = {'through': (-STRAIGHT, STRAIGHT), 'slight_left': (10.0, 60.0), 'left': (STRAIGHT, 150.0), 'sharp_left': (110.0, U_TURN),
          'slight_right': (-60.0, -10.0), 'right': (-150.0, -STRAIGHT), 'sharp_right': (-U_TURN, -110.0),
          'merge_to_left': (-STRAIGHT, STRAIGHT), 'merge_to_right': (-STRAIGHT, STRAIGHT)}


def lane_changes(osm: OsmLanes, ways) -> set[int]:
  """The one-way ways among `ways` up to LANE_CHANGE_M long that leave a carriageway going on through their first node
  and join one going on through their last (GTA's lane changes between links side by side): at each, another one-way
  way in and another out run on straighter (by BRANCH) than this one turns off them, and within SAME_WAY of it. Where
  carriageways braid, both are such ways' neighbours."""
  first, last, ends_at, starts_at = {}, {}, {}, {}
  for wid in ways:
    tags, refs = osm.ways[wid]
    if oneway_of(tags) != 1 or len(refs) < 2:
      continue
    first[wid], last[wid] = refs[0], refs[-1]
    ends_at.setdefault(refs[-1], []).append(wid)
    starts_at.setdefault(refs[0], []).append(wid)

  def heading(wid, at_end):
    pts = osm.xy[osm.data.index(osm.ways[wid][1])]
    return _unit(pts[-1] - pts[-2] if at_end else pts[1] - pts[0])

  def angle(u, v):
    return float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))

  out = set()
  for wid in first:
    pts = osm.xy[osm.data.index(osm.ways[wid][1])]
    if float(np.hypot(*np.diff(pts, axis=0).T).sum()) > LANE_CHANGE_M:
      continue
    ends = []
    for node, u, leaving in ((first[wid], heading(wid, False), True), (last[wid], heading(wid, True), False)):
      ins = [heading(w, True) for w in ends_at.get(node, ()) if w != wid]
      outs = [heading(w, False) for w in starts_at.get(node, ()) if w != wid]
      if not ins or not outs:
        ends.append(False)
        continue
      on = min(angle(a, b) for a in ins for b in outs)  # the carriageway going on through the node
      off = min(angle(a, u) for a in ins) if leaving else min(angle(u, b) for b in outs)
      ends.append(off > on + BRANCH and off < SAME_WAY)
    if all(ends):
      out.add(wid)
  return out


DRAWN_NEAR = 0.6  # m: a painted island's outline this near a line the road already draws there is that line


def apart(line: np.ndarray, segs: np.ndarray, dist: float = DRAWN_NEAR, step: float = 0.25, min_len: float = 1.0) -> list[np.ndarray]:
  """The pieces of a line [N, 2] further than dist from every segment [S, 2, 2]: a painted island's outline but where
  the road's own lines draw it (a hatched median's edges are its centre lines)."""
  pts = densify(line, step)
  if len(segs):
    lo, hi = pts.min(0) - dist, pts.max(0) + dist
    s = segs[((segs.max(1) >= lo) & (segs.min(1) <= hi)).all(1)]
  else:
    s = segs
  if not len(s):
    return [line]
  a, ab = s[:, 0], s[:, 1] - s[:, 0]
  near = np.zeros(len(pts), bool)
  for b in range(0, len(pts), 256):
    q = pts[b:b + 256]
    t = np.clip(np.einsum('qsk,sk->qs', q[:, None] - a[None], ab) / np.maximum(np.einsum('sk,sk->s', ab, ab), 1e-12)[None], 0.0, 1.0)
    near[b:b + 256] = np.hypot(*(a[None] + ab[None] * t[..., None] - q[:, None]).transpose(2, 0, 1)).min(1) < dist
  out, k = [], 0
  while k < len(pts):
    if near[k]:
      k += 1
      continue
    e = k
    while e < len(pts) and not near[e]:
      e += 1
    if e - k >= 2 and float(np.hypot(*np.diff(pts[k:e], axis=0).T).sum()) >= min_len:
      out.append(pts[k:e])
    k = e
  return out


class Islands:
  """Outlines [K, 2] (closed) kerbs keep off, by CELL squares, with their heights (nan unknown)."""
  def __init__(self, polys: list[np.ndarray], heights: list[float] | None = None):
    self.polys = polys
    self.z = np.array(heights if heights is not None else [np.nan] * len(polys), float)
    self.cells: dict[tuple[int, int], list[int]] = {}
    for n, p in enumerate(polys):
      lo, hi = p.min(0) // CELL, p.max(0) // CELL
      for cx in range(int(lo[0]), int(hi[0]) + 1):
        for cy in range(int(lo[1]), int(hi[1]) + 1):
          self.cells.setdefault((cx, cy), []).append(n)

  def __len__(self) -> int:
    return len(self.polys)

  def near(self, lo: np.ndarray, hi: np.ndarray, z: float | None = None) -> list[np.ndarray]:
    """The outlines in a box, those at height z (within ISLAND_LEVEL) where both are known."""
    found = {n for cx in range(int(lo[0] // CELL), int(hi[0] // CELL) + 1) for cy in range(int(lo[1] // CELL), int(hi[1] // CELL) + 1)
             for n in self.cells.get((cx, cy), ())}
    level = z is None or not np.isfinite(z)
    return [self.polys[n] for n in sorted(found) if (self.polys[n].max(0) >= lo).all() and (self.polys[n].min(0) <= hi).all()
            and (level or not np.isfinite(self.z[n]) or abs(self.z[n] - z) <= ISLAND_LEVEL)]


def off_islands(line: np.ndarray, islands, reach: float = ISLAND_REACH, step: float = 0.25,
                min_len: float = 0.3, z=None) -> list[np.ndarray]:
  """The pieces of a line [N, 2] not inside, nor within reach of, any of the outlines (Islands, or a list of [K, 2]); of
  Islands only those at the line's height `z` (m, or its points' heights), where given."""
  if not len(islands):
    return [line]
  lo, hi = line.min(0) - reach, line.max(0) + reach
  if z is not None and np.ndim(z):
    z = float(np.nanmean(z)) if np.isfinite(z).any() else None
  near = islands.near(lo, hi, z) if isinstance(islands, Islands) else \
    [p for p in islands if (p.max(0) >= lo).all() and (p.min(0) <= hi).all()]
  if not near:
    return [line]
  pts = densify(line, step)
  out = np.zeros(len(pts), bool)
  for p in near:
    ring = np.vstack([p, p[:1]]) if np.hypot(*(p[-1] - p[0])) > 1e-6 else p
    a, ab = ring[:-1], np.diff(ring, axis=0)
    t = np.clip(np.einsum('qsk,sk->qs', pts[:, None] - a[None], ab) / np.maximum(np.einsum('sk,sk->s', ab, ab), 1e-12)[None], 0.0, 1.0)
    dist = np.hypot(*(a[None] + ab[None] * t[..., None] - pts[:, None]).transpose(2, 0, 1)).min(axis=1)
    x, y = pts[:, None, 0], pts[:, None, 1]  # inside: even-odd, as an outline needn't be convex
    spans = (a[None, :, 1] > y) != (ring[None, 1:, 1] > y)
    dy = np.where(ab[:, 1] == 0.0, 1e-12, ab[:, 1])[None]
    cross = x < a[None, :, 0] + (y - a[None, :, 1]) * ab[None, :, 0] / dy
    out |= (dist < reach) | ((spans & cross).sum(axis=1) % 2 == 1)
  pieces, k = [], 0
  while k < len(pts):
    if out[k]:
      k += 1
      continue
    e = k
    while e < len(pts) and not out[e]:
      e += 1
    piece = pts[k:e]
    if len(piece) >= 2 and float(np.hypot(*np.diff(piece, axis=0).T).sum()) >= min_len:
      pieces.append(piece)
    k = e
  return pieces


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


def crossovers(osm: OsmLanes, ways: dict) -> set[int]:
    """GTA's links across a motorway's median or between its parallel chains (emergency crossovers, shortcuts): chains
    of two-way ways, no longer than CROSSOVER_M, whose ends meet only motorway carriageways (one-way motorway ways and
    their links). They're no roads to draw, and make no junctions."""
    at: dict[int, list[int]] = {}
    for w, (_, refs) in ways.items():
      at.setdefault(refs[0], []).append(w)
      at.setdefault(refs[-1], []).append(w)

    def carriageway(w):
      tags = ways[w][0]
      return tags.get('highway') in FREEWAY and oneway_of(tags) != 0

    def candidate(w):
      return oneway_of(ways[w][0]) == 0 and not carriageway(w)
    out: set[int] = set()
    seen: set[int] = set()
    for w in ways:
      if w in seen or not candidate(w):
        continue
      group, todo, ends = {w}, [w], []  # the candidates joined end to end at nodes only they meet at
      while todo:
        v = todo.pop()
        for n in (ways[v][1][0], ways[v][1][-1]):
          others = [x for x in at[n] if x not in group]
          if not others and len(at[n]) > 1:
            continue  # between two of the group
          if others and all(candidate(x) for x in others) and not any(carriageway(x) for x in at[n]):
            group.update(others)
            todo += others
          elif n not in ends:
            ends.append(n)
      seen |= group
      length = sum(float(np.hypot(*np.diff(osm.way_points(v), axis=0).T).sum()) for v in group)
      if len(ends) >= 2 and length <= CROSSOVER_M * len(ends) / 2 and \
          all(any(carriageway(x) for x in at[n]) and all(carriageway(x) or x in group for x in at[n]) for n in ends):
        out |= group
    return out


def _faces(tags: dict, m: 'Member', k: int) -> bool:
  """Whether a stop line node, the member's k-th, is for traffic towards its junction by its direction tag (none: either)."""
  facing = tags.get('traffic_signals:direction') or tags.get('direction')
  if facing not in ('forward', 'backward'):
    return True
  wanted = facing == 'backward'  # the way runs out of the junction where traffic towards it goes backward
  return any(m.ways[q][1] == wanted for q in (k - 1, k) if 0 <= q < len(m.ways))


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
  stop: float = math.inf  # m: back to its surveyed stop line, where that's out past where its kerbs meet the next arms'
  meets: tuple = (None, None)  # where its right and left kerbs meet the next arms' (shape), else None

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
  node: int | None = None  # the node tagged with it


@dataclass
class Junction:
  nodes: list[int]
  arms: list[Arm]
  polygon: np.ndarray  # [N, 2], counterclockwise
  kerbs: list[np.ndarray]  # the kerb round each corner, from one arm's mouth to the next's
  inside: set[int]  # ways inside it
  centre: np.ndarray  # its nodes' middle, which its area fans out from (in_fan)
  stops: list[Stop] = field(default_factory=list)
  through: dict[int, set[float]] = field(default_factory=dict)  # the road carried on through: way -> its lines (m right, to cm)
  carried: dict[int, set[float]] = field(default_factory=dict)  # those painted on across it (a priority road's)
  minor: bool = False  # only minor roads meet its through road (_minor_sides): its lines go on across

  @property
  def ways(self) -> set[int]:
    return {w for arm in self.arms for m in arm.members for w, _ in m.ways[:1]}

  @property
  def roads(self) -> set[int]:
    """The ways of its roads that reach into its area: those starting short of where their road is trimmed."""
    return {w for arm in self.arms for m in arm.members for k, (w, _) in enumerate(m.ways) if k == 0 or Junctions._along(m, k) < m.trim}


class Junctions:
  """The junctions of a map (OsmLanes): `junctions`, and `trims` {(way, node): m}, how far each way is trimmed back from
  its end at a junction node, along the arm it starts; `inside` the ways inside junctions. `roads(tags)` says which
  ways count (default: all the map's roads)."""
  def __init__(self, osm: OsmLanes, roads=None):
    self.osm = osm
    keep = roads or (lambda tags: True)
    self.ways = {w: v for w, v in osm.ways.items() if keep(v[0]) and len(v[1]) >= 2}
    self.crossovers = crossovers(osm, self.ways)
    for w in self.crossovers:
      del self.ways[w]
    self.steps: dict[int, list[tuple[int, int, bool]]] = {}  # node -> [(way, next node, along the way)]
    for wid, (_, refs) in self.ways.items():
      for i, n in enumerate(refs):
        if i > 0:
          self.steps.setdefault(n, []).append((wid, refs[i - 1], False))
        if i + 1 < len(refs):
          self.steps.setdefault(n, []).append((wid, refs[i + 1], True))
    self.junction_nodes = {n for n, s in self.steps.items() if len(s) >= 3}
    self.lane_changes = lane_changes(osm, self.ways)
    # painted islands: road surface, the kerbs GTA's links round them suggest aren't there
    painted = [(osm.xy[osm.data.index(refs)], tags.get('colour')) for tags, refs in osm.data.ways.values()
               if tags.get('traffic_calming') == 'painted_island' and len(refs) >= 4]
    surfaces = [refs for tags, refs in osm.data.ways.values() if 'area:highway' in tags and len(refs) >= 4]
    self.painted_islands = [xy for xy, _ in painted]  # their tips end junctions' corners (shape)

    def height(refs):
      ele = [float(t['ele']) for n in refs if 'ele' in (t := osm.data.node_tags.get(n, {}))]
      return float(np.mean(ele)) if ele else np.nan
    island_refs = [refs for tags, refs in osm.data.ways.values() if tags.get('traffic_calming') == 'painted_island' and len(refs) >= 4]
    self.islands = Islands(self.painted_islands + [osm.xy[osm.data.index(refs)] for refs in surfaces],  # road surface: no kerb in it
                           [height(refs) for refs in island_refs + surfaces])
    self.island_outlines = [(xy, colour) for xy, colour in painted if colour in ('white', 'yellow')]  # drawn as painted
    self.crossings = [osm.xy[osm.data.index(refs)] for tags, refs in osm.data.ways.values()
                      if tags.get('footway') == 'crossing' and len(refs) >= 2]
    self._crossing_cells = self._cells(self.crossings)
    self.junctions: list[Junction] = []
    self.trims: dict[tuple[int, int], float] = {}
    self.inside: set[int] = set()
    self.triangle_sides: set[frozenset[int]] = set()  # the node pairs of the slip triangles merged into junctions
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

  def carriageways(self, node: int) -> list[np.ndarray]:
    """The directions of the carriageways going on through a node: a one-way way (not a freeway's) into it and one out
    of it, running within STRAIGHT of each other."""
    ins, outs = [], []
    for w, nxt, fwd in self.steps.get(node, ()):
      tags = self.ways[w][0]
      way = oneway_of(tags)
      if not way or self.freeway(tags):
        continue
      u = _unit(self.osm.node_xy(nxt) - self.osm.node_xy(node))
      (outs if (way == 1) == fwd else ins).append(u)
    return [(o - i) / 2 for i in ins for o in outs if float(-i @ o) > math.cos(math.radians(STRAIGHT))]

  def across_median(self, ways: list[tuple[int, bool]], nodes: list[int]) -> bool:
    """Whether a road (its ways from nodes[0] to nodes[-1]) runs two-way across a divided road's median, between its
    two carriageways: one going on through either end, running opposite ways, the road crossing them."""
    if any(oneway_of(self.ways[w][0]) for w, _ in ways) or len(nodes) < 2:
      return False
    across = _unit(self.osm.node_xy(nodes[-1]) - self.osm.node_xy(nodes[0]))
    return any(float(_unit(u) @ _unit(v)) < -math.cos(math.radians(STRAIGHT)) and
               abs(float(_unit(u) @ across)) < math.cos(math.radians(45.0))
               for u in self.carriageways(nodes[0]) for v in self.carriageways(nodes[-1]))

  def straight_on(self, ways: list[tuple[int, bool]], chain: list[int], other: list[int]) -> bool:
    """Whether a two-way road (its ways along chain) is the road `other` (a chain of nodes) or runs straight on from it."""
    if chain == other or chain == other[::-1]:
      return True
    if any(oneway_of(self.ways[w][0]) for w, _ in ways) or not {chain[0], chain[-1]} & {other[0], other[-1]}:
      return False
    u = _unit(self.osm.node_xy(chain[-1]) - self.osm.node_xy(chain[0]))
    v = _unit(self.osm.node_xy(other[-1]) - self.osm.node_xy(other[0]))
    return abs(float(u @ v)) > math.cos(math.radians(STRAIGHT))

  def member(self, node: int, step) -> Member | None:
    ways, nodes, _ = self.walk(node, step, ARM_LENGTH)
    pts = self.osm.xy[self.osm.data.index(nodes)]
    if len(pts) < 2 or np.hypot(*(pts[-1] - pts[0])) < 1e-3:
      return None
    edges = [self.osm.lanes(w).edges(FORWARD if fwd else BACKWARD) for w, fwd in ways]

    try:  # the kerbs as the roads' edges are drawn (OsmLanes.line_geometry), stepping only where they don't meet
      line, left, right = Poly(pts), Poly(self.osm.kerb_line(ways, nodes, 0)), Poly(self.osm.kerb_line(ways, nodes, 1))
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
        ways, nodes, length = self.walk(n, step, MEDIAN_LINK)
        if nodes[-1] in self.junction_nodes and nodes[-1] != n and \
            (length < min(CLUSTER_LINK, max(self.widest(n), self.widest(nodes[-1]))) or self.across_median(ways, nodes)):
          parent[find(n)] = find(nodes[-1])
    near: dict[int, set[int]] = {n: set() for n in self.junction_nodes}
    slips: set[frozenset[int]] = set()  # pairs of them joined by a road one-way somewhere along it
    for n in self.junction_nodes:
      for step in self.steps[n]:
        ways, nodes, length = self.walk(n, step, TRIANGLE_LINK)
        if nodes[-1] in near and nodes[-1] != n and length < TRIANGLE_LINK:
          near[n].add(nodes[-1])
          near[nodes[-1]].add(n)
          if any(oneway_of(self.ways[w][0]) for w, _ in ways):
            slips.add(frozenset((n, nodes[-1])))
    for a in self.junction_nodes:
      for b in near[a]:
        for c in near[a] & near[b]:
          xy = np.array([self.osm.node_xy(q) for q in (a, b, c)])
          sides = {frozenset((a, b)), frozenset((b, c)), frozenset((a, c))}
          if len(sides & slips) >= 2 and np.hypot(*(xy.max(0) - xy.min(0))) < TRIANGLE_SPAN and \
              (one := self.junction(sorted({a, b, c}))) is not None and len(one.arms) == 3:  # a side road joining
            parent[find(b)] = find(a)
            parent[find(c)] = find(a)
            self.triangle_sides |= sides
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
    for j in self.junctions:
      j.through = self._through(j)
      j.minor = self._minor_sides(j)
      if all(self.ways[w][0].get('priority_road') in PRIORITY for w in j.through) or j.minor:
        j.carried = self._carried_on(j)
    self.osm.carry_across([tuple(j.through) for j in self.junctions if j.carried and len(j.through) == 2])

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
    if len(members) < 3 or all(self.freeway(self.ways[m.ways[0][0]][0]) for m in members) or \
        self.merges(members, [w for ways, _ in links for w, _ in ways]) or (len(nodes) == 1 and self.lane_split(members)):
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
      arm.stop = min([along - STOP_SETBACK for m in arm.members for along in self._surveyed_stops(m)], default=math.inf)
    corners = self.shape(arms, self._islands_near(nodes))
    self.square(arms)
    polygon, kerbs = self.outline(arms, corners)
    for arm in arms:
      p = arm.line.at(arm.trim)
      for m in arm.members:
        m.trim = max(m.line.project(p), 0.0)
    # a road between the junction's own nodes is inside it, unless it runs mostly outside its area, as a slip road
    # round a corner: that one is drawn as a road, its kerbs cut where it's inside. A road across a divided road's
    # median is inside it whatever its area, with the road it runs straight on into across the median
    inside = set()
    across = [chain for ways, chain in links if self.across_median(ways, chain)]  # and through a node in the median
    across += [c1 + c2[1:] for w1, c1 in links for w2, c2 in links if c1[-1] == c2[0] and c1[0] != c2[-1] and
               self.across_median(w1 + w2, c1 + c2[1:])]
    for ways, chain in links:
      pts = self.osm.xy[self.osm.data.index(chain)]
      mids = np.vstack([pts[:-1] + (pts[1:] - pts[:-1]) * t for t in (0.25, 0.5, 0.75)])
      if in_fan(mids, centre, polygon).mean() >= 0.5 or any(self.straight_on(ways, chain, c) for c in across) or \
          frozenset((chain[0], chain[-1])) in self.triangle_sides:  # a slip triangle's are its own, not roads round it
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

  @staticmethod
  def freeway(tags: dict) -> bool:
    """A motorway's, or a one-way trunk road's: a divided highway's carriageway, or a lane change across one."""
    return tags.get('highway') in FREEWAY or (tags.get('highway', '').removesuffix('_link') == 'trunk' and oneway_of(tags) != 0)

  def lane_split(self, members: list[Member]) -> bool:
    """Whether the roads out of a node are a two-way road carried straight on through it and one-way ways only leaving
    it or joining it, each on the side its traffic that way drives on and within LANE_SPLIT of that traffic's heading: a
    lane parting from or merging into it, with no traffic crossing."""
    two = [m for m in members if not oneway_of(self.ways[m.ways[0][0]][0])]
    one = [m for m in members if oneway_of(self.ways[m.ways[0][0]][0])]
    if len(two) != 2 or not one or math.cos(two[0].heading - two[1].heading) > -math.cos(math.radians(MERGE_FLOW)):
      return False
    for m in one:
      w, along = m.ways[0]
      flow = m.heading if (oneway_of(self.ways[w][0]) == 1) == along else m.heading + math.pi
      road = max(two, key=lambda r: math.cos(r.heading - flow)).heading  # the traffic it leaves with or joins
      if math.cos(flow - road) < math.cos(math.radians(LANE_SPLIT)):
        return False
      left_of = math.sin(m.heading - road) > 0  # its way out from the node lies left of that traffic's line
      if left_of == self.osm.drive_on_right:
        return False
      # beside the road: not a slip road away to another one
      side = max(two, key=lambda r: math.cos(r.heading - m.heading))
      p, u = side.line.at(0.0), side.line.tangent(0.0)
      q = m.line.at(min(BESIDE_REACH, m.line.length))
      lo, hi = self.osm.lanes(w).edges(FORWARD)
      if abs(_cross(u, q - p)) > max(abs(e) for e in side.edges) + (hi - lo) + 1.0:
        return False
    return True

  def merges(self, members: list[Member], inside: list[int]) -> bool:
    """Whether the roads out of a junction are all one-way and all run within MERGE_FLOW (FREEWAY_FLOW where one is a
    motorway's or its link's) of one heading, and the roads between its nodes (`inside`) are one-way: lanes merging,
    parting or changing across one carriageway (as lane changes laid as links of their own), with no traffic crossing.
    GTA's lane changes between the carriageways (`lane_changes`) cut across at a steeper angle: the carriageways'
    flows alone say it, where two or more are left."""
    if any(not oneway_of(self.ways[w][0]) for w in inside):
      return False
    flows = []
    for m in members:
      w, along = m.ways[0]
      way = oneway_of(self.ways[w][0])
      if not way:
        return False
      flows.append((m.heading if (way == 1) == along else m.heading + math.pi, w in self.lane_changes))
    if sum(not change for _, change in flows) >= 2:
      flows = [f for f in flows if not f[1]]
    flows = [f for f, _ in flows]
    mean = math.atan2(sum(math.sin(f) for f in flows), sum(math.cos(f) for f in flows))
    spread = FREEWAY_FLOW if any(self.ways[m.ways[0][0]][0].get('highway') in FREEWAY for m in members) else MERGE_FLOW
    return all(math.cos(f - mean) > math.cos(math.radians(spread)) for f in flows)

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

  def flows(self, arm: Arm) -> set[bool]:
    """Which ways an arm's one-way roads run: out of the junction (True), into it (False); none if any is two-way."""
    out = set()
    for m in arm.members:
      w, along = m.ways[0]
      way = oneway_of(self.ways[w][0])
      if not way:
        return set()
      out.add((way == 1) == along)
    return out

  def square(self, arms: list[Arm]):
    """Trims each of a divided road's carriageways out of the junction, one in and one out side by side (with any turn
    lanes beside them), back at least as far as the other: one trimmed short would leave the junction's area reaching
    out along the other only."""
    flows = [self.flows(arm) for arm in arms]
    for i, a in enumerate(arms):
      for j, b in enumerate(arms):
        if i == j or not flows[i] or not flows[j] or len(flows[i] | flows[j]) < 2:
          continue
        if math.degrees(abs((b.heading - a.heading + math.pi) % (2 * math.pi) - math.pi)) > BUNDLE_ANGLE:
          continue
        p = b.line.at(b.trim)
        s = a.line.project(p)
        if np.hypot(*(a.line.at(s) - p)) <= MEDIAN_REACH:
          a.trim = min(max(a.trim, s), a.cap, a.stop)

  @staticmethod
  def side_by_side(a: Member, b: Member) -> bool:
    turn = abs((b.heading - a.heading + math.pi) % (2 * math.pi) - math.pi)
    if math.degrees(turn) > BUNDLE_ANGLE:
      return False
    d = min(10.0, a.line.length)
    p = a.left.at(a.left.project(a.line.at(d)))
    q = b.right.at(b.right.project(p))
    return float((q - p) @ _left(a.line.tangent(d))) < BUNDLE_GAP

  def _islands_near(self, nodes: list[int]) -> list[np.ndarray]:
    """The painted islands within MAX_TRIM of a junction's nodes."""
    pts = self.osm.xy[self.osm.data.index(nodes)]
    lo, hi = pts.min(0) - MAX_TRIM, pts.max(0) + MAX_TRIM
    return [p for p in self.painted_islands if (p.max(0) >= lo).all() and (p.min(0) <= hi).all()]

  @staticmethod
  def _tip(a: 'Arm', b: 'Arm', x: np.ndarray, islands: list[np.ndarray]) -> np.ndarray | None:
    """Where a painted island lying in the corner between two arms (its points between their lines, nearer the
    junction than where their kerbs meet, x) comes nearest the junction: its tip, the corner's point instead of x."""
    best = None
    sx = max(a.line.project(x), b.line.project(x))
    for p in islands:
      for q in p:
        sa, sb = a.line.project(q), b.line.project(q)
        if not (0.0 < sa < sx and 0.0 < sb < sx):
          continue
        # left of a's line looking out and right of b's: between them
        if float((q - a.line.at(sa)) @ _left(a.line.tangent(sa))) <= 0.0 or float((q - b.line.at(sb)) @ _left(b.line.tangent(sb))) >= 0.0:
          continue
        if best is None or min(sa, sb) < best[0]:
          best = (min(sa, sb), q)
    return None if best is None else best[1]

  @staticmethod
  def shape(arms: list[Arm], islands: list[np.ndarray] = ()) -> list[tuple[float, float, np.ndarray] | None]:
    """Sets each arm's trim; returns each corner's kerb, between an arm and the next counterclockwise: (where it rounds
    from on this arm's left kerb, where to on the next's right kerb, the point the two kerbs meet at), or None where
    they don't meet ahead of the junction and the kerb between them is straight. A corner rounds no further out than a
    surveyed stop line beyond where the kerbs meet; one nearer the junction than that (painted level with the road it
    meets, which the class layout draws a little wider or off where GTA's links run) is inside its area, and the road
    is trimmed as if it had none: cut back to it, its kerbs would run out across the other road's lanes. A painted
    island in the corner (traffic_calming=painted_island: road surface, no kerb) ends it at the island's tip, square:
    two roads parting at a narrow angle with a painted gore between would else trim back to MAX_TRIM over it."""
    n = len(arms)
    trims = [0.0] * n
    hits = []
    for i in range(n):
      a, b = arms[i], arms[(i + 1) % n]
      gap = (b.heading - a.heading) % (2 * math.pi)
      # arms heading within BUNDLE_ANGLE of each other run side by side: their kerbs, carried back, cross only deep inside
      hit = crossing(a.left, b.right, MAX_TRIM, MAX_TRIM) if math.radians(BUNDLE_ANGLE) < gap < math.pi - 1e-3 else None
      if hit is not None and islands and (tip := Junctions._tip(a, b, hit[2], islands)) is not None:
        hit = (a.left.project(tip), b.right.project(tip), tip, True)
      hits.append(hit)
    meet = [0.0] * n  # m out along each arm to where its kerbs meet its neighbours'
    for i, hit in enumerate(hits):
      if hit is not None:
        a, b = arms[i], arms[(i + 1) % n]
        meet[i], meet[(i + 1) % n] = max(meet[i], a.line.project(hit[2])), max(meet[(i + 1) % n], b.line.project(hit[2]))
        a.meets, b.meets = (a.meets[0], hit[2]), (hit[2], b.meets[1])
    for arm, m in zip(arms, meet, strict=True):
      if arm.stop < m:
        arm.stop = math.inf
    corners: list[tuple[float, float, np.ndarray] | None] = []
    for i, hit in enumerate(hits):
      if hit is None:
        corners.append(None)
        continue
      a, b = arms[i], arms[(i + 1) % n]
      gap = (b.heading - a.heading) % (2 * math.pi)
      sa, sb, x = hit[:3]
      radius = min(max(min(a.width, b.width) / 2, MIN_RADIUS), CORNER_RADIUS)
      t = min(radius / math.tan(gap / 2), MAX_TANGENT, min(a.cap, a.stop) - a.line.project(x), min(b.cap, b.stop) - b.line.project(x))
      ta, tb = (sa, sb) if len(hit) > 3 else (sa + max(t, 0.0), sb + max(t, 0.0))  # (square at an island's tip)
      trims[i] = max(trims[i], a.line.project(a.left.at(ta)))
      trims[(i + 1) % n] = max(trims[(i + 1) % n], b.line.project(b.right.at(tb)))
      corners.append((ta, tb, x))
    for arm, t in zip(arms, trims, strict=True):
      arm.trim = min(max(t, 0.0), arm.cap, arm.stop)
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
    placed = set()  # stop line nodes placed on a junction's approach
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
        if not _faces(tags, m, k):
          continue
        found.append((along, j, m))
      if not found:
        continue
      # one junction's: a node between two junctions, its ways drawn opposite ways out of it, faces both by its tag
      # (forward along one is backward along the other); the line is the one nearest its own junction's mouth
      best = min(found, key=lambda f: f[0] - f[2].trim)
      for along, j, m in [best] if facing not in ('forward', 'backward') else [f for f in found if f[1] is best[1]]:
        self._add_stop(j, m, kind, along, tags['highway'] == 'traffic_signals', node)
        placed.add(node)
    for j in self.junctions:
      # a directed one already placed on another junction's approach is that one's (GTA's light where a slip parts,
      # for the junction ahead), not every way's into this one; one placed nowhere stops the ways in it faces
      node = next((n for n in j.nodes if STOPS.get(tags_of.get(n, {}).get('highway', '')) and
                   not (n in placed and (tags_of[n].get('traffic_signals:direction') or tags_of[n].get('direction')))), None)
      if node is not None:
        tags = tags_of[node]
        for arm in j.arms:
          for m in arm.members:
            if not any(s.member is m for s in j.stops) and _faces(tags, m, 0):
              self._add_stop(j, m, STOPS[tags['highway']], 0.0, tags['highway'] == 'traffic_signals', node)

  def _surveyed_stops(self, m: Member) -> list[float]:
    """m along the member to the stop lines on it towards its junction that were surveyed where they're painted
    (`source:position=survey`): its road ends there, and its stop line is drawn there."""
    tags_of = self.osm.data.node_tags
    out = []
    for k, node in enumerate(m.nodes[1:], 1):
      tags = tags_of.get(node, {})
      if tags.get('source:position') == 'survey' and STOPS.get(tags.get('highway', '')) and _faces(tags, m, k):
        if (along := self._along(m, k)) <= MAX_TRIM + STOP_SETBACK:
          out.append(along)
    return out

  @staticmethod
  def _along(m: Member, k: int) -> float:
    pts = m.line.p[1:-1]
    return float(np.hypot(*np.diff(pts[:k + 1], axis=0).T).sum()) if k else 0.0

  def _add_stop(self, j: Junction, m: Member, kind: str, along: float, signal: bool = False, node: int | None = None):
    w, fwd = m.ways[0]
    spans = self.osm.lanes(w).ours(BACKWARD if fwd else FORWARD)  # the lanes towards the junction
    if not spans:
      return
    surveyed = node is not None and self.osm.data.node_tags.get(node, {}).get('source:position') == 'survey'
    s = along if surveyed else max(along, m.trim + STOP_SETBACK)  # surveyed: where painted, even inside the area
    for c in [] if surveyed else self._near(self.crossings, self._crossing_cells, m.line.at(s)):  # a stop line goes before a crossing
      hit = crossing(Poly(c, 0.0, 0.0), m.line, math.inf, max(s, m.trim + CROSSING_REACH) + CROSSING_WIDTH)
      if hit is not None and hit[1] > m.trim - CROSSING_WIDTH:
        s = max(s, hit[1] + CROSSING_WIDTH / 2 + STOP_SETBACK)
    p, u = m.line.at(s), m.line.tangent(s)
    right = _left(u)  # right of the direction of travel into the junction
    lo, hi = min(sp.left for sp in spans), max(sp.right for sp in spans)
    line = np.array([p + right * hi, p + right * lo])
    arm = next((a for a in j.arms if any(x is m for x in a.members)), None)
    if surveyed and s < m.trim and arm is not None and all(x is not None for x in arm.meets):
      # inside the area, level with the road it meets: along that road's edge, between where this road's kerbs meet it
      e = _unit(arm.meets[1] - arm.meets[0])
      if (c := float(e @ right)) > math.cos(math.radians(STOP_SKEW)):
        line = np.array([p + e * hi / c, p + e * lo / c])
    e_lo, e_hi = m.edges  # looking out: its right kerb is on the left arriving
    q = m.line.at(m.trim)
    uq = _left(m.line.tangent(m.trim))
    area = np.array([q - uq * e_hi, p + right * (-e_hi), p + right * (-e_lo), q - uq * e_lo])
    j.stops.append(Stop(kind, line, m, s, area, signal, node))

  # *** a road carried on through ***

  def _through(self, j: Junction) -> dict[int, set[float]]:
    """The road carried straight on through a junction, as its lines that meet across it: {way: {m right of its line,
    rounded to cm}}, for the way either side; empty where there's none."""
    tags_of = self.osm.data.node_tags
    if any(s.signal for s in j.stops) or any(tags_of.get(n, {}).get('highway') == 'traffic_signals' for n in j.nodes):
      return {}

    def rank(m):
      return CLASSES.index(c) if (c := self.ways[m.ways[0][0]][0].get('highway', '').removesuffix('_link')) in CLASSES else len(CLASSES)
    stopped = {id(s.member) for s in j.stops}
    found = []
    for i, a in enumerate(j.arms):
      for b in j.arms[i + 1:]:
        turn = abs(math.degrees((b.heading - a.heading) % (2 * math.pi) - math.pi))
        if turn > STRAIGHT or len(a.members) != 1 or len(b.members) != 1:
          continue
        ma, mb = a.members[0], b.members[0]
        if id(ma) in stopped or id(mb) in stopped:
          continue
        lines = self._meeting(ma, mb) if ma.start == mb.start else self._meeting_across(j, ma, mb)
        if not lines:
          continue
        pair_rank, width = max(rank(ma), rank(mb)), min(a.width, b.width)
        ok = not any(self.osm.taper(m.ways[0][0]) is not None or self._crossed(m) for m in (ma, mb)) and \
          not any(rank(m) < pair_rank and arm.width > width for arm in j.arms if arm is not a and arm is not b for m in arm.members)
        found.append(((pair_rank, -width, turn), {id(a), id(b)}, lines, ok))
    if not found:
      return {}
    found.sort(key=lambda f: f[0])
    best = found[0]
    # where a road as important crosses it with its lines, neither has the way: a crossroads of equals
    if not best[3] or any(f[0][0] == best[0][0] and not f[1] & best[1] for f in found[1:]):
      return {}
    return best[2]

  def _minor_sides(self, j: Junction) -> bool:
    """Whether every road meeting the through road at a junction is a minor one: a service road or track (a driveway,
    a car park's way in), or a one-way lane (a slip into or out of it), each narrower than MINOR_SHARE of the through
    road: its lines are painted on across, as the game paints a main road's lanes on past them."""
    if not j.through:
      return False
    width = min(arm.width for arm in j.arms if any(m.ways[0][0] in j.through for m in arm.members))
    for arm in j.arms:
      for m in arm.members:
        w = m.ways[0][0]
        if w in j.through:
          continue
        tags = self.ways[w][0]
        minor = tags.get('highway') in MINOR or (oneway_of(tags) and len(self.osm.lanes(w).lanes) == 1)
        if not minor or arm.width > MINOR_SHARE * width:
          return False
    return True

  def _carried_on(self, j: Junction) -> dict[int, set[float]]:
    """The through road's lines painted on across the junction: its first way's either side, and the ways after them
    that still reach into its area (the area of a side road meeting at a skew reaches past a short way), all their
    lines between lanes."""
    out = dict(j.through)
    for arm in j.arms:
      for m in arm.members:
        if m.ways[0][0] not in j.through:
          continue
        for k, (w, _) in enumerate(m.ways[1:], 1):
          if self._along(m, k) >= m.trim:
            break
          if (road := self.osm.lanes(w)).markings:
            out[w] = {round(ln.offset, 2) for ln in road.lines(FORWARD) if ln.kind in (CENTRE, DIVIDER, MEDIAN)}
    return out

  def _crossed(self, m: Member) -> bool:
    """Whether a pedestrian crossing (`footway=crossing`) crosses the road near its junction: its lines stop there."""
    reach = m.trim + CROSSING_REACH + CROSSING_WIDTH
    for c in self._near(self.crossings, self._crossing_cells, m.line.at(min(m.trim, m.line.length))):
      hit = crossing(Poly(c, 0.0, 0.0), m.line, math.inf, reach)
      if hit is not None and hit[1] >= 0.0:
        return True
    return False

  def _meeting_across(self, j: Junction, ma: Member, mb: Member) -> dict[int, set[float]]:
    """_meeting for two roads out of a junction of several nodes, from two of them: the road runs on between them along
    ways inside it (as a through road between the corners of a slip triangle), its lines meeting at each node."""
    from types import SimpleNamespace
    inside = [w for w in j.inside if len(self.ways[w][1]) >= 2]
    paths, frontier = [], [(ma.start, [])]
    for _ in range(3):  # up to three ways inside
      nxt = []
      for node, path in frontier:
        for w in inside:
          refs = self.ways[w][1]
          if w in [p for p, _ in path] or node not in (refs[0], refs[-1]):
            continue
          far = refs[-1] if refs[0] == node else refs[0]
          (paths if far == mb.start else nxt).append((far, path + [(w, refs[0] == node)]))
      frontier = nxt
    def out_of(node, w, fwd):  # the inside way as a road out of node
      refs = self.ways[w][1]
      chain = refs if fwd else refs[::-1]
      return SimpleNamespace(start=node, ways=[(w, fwd)], line=Poly(self.osm.xy[self.osm.data.index(chain)]))

    def length(path):
      return sum(float(np.hypot(*np.diff(self.osm.way_points(w), axis=0).T).sum()) for w, _ in path)
    for _, path in sorted(paths, key=lambda p: length(p[1])):  # the shortest whose lines meet all along
      out: dict[int, set[float]] = {}
      node, prev = ma.start, ma
      for w, fwd in [*path, (None, None)]:
        lines = self._meeting(prev, out_of(node, w, fwd) if w is not None else mb)
        if not lines:
          break
        for k, v in lines.items():
          out.setdefault(k, set()).update(v)
        if w is None:
          return out
        node = self.ways[w][1][-1] if fwd else self.ways[w][1][0]
        prev = out_of(node, w, not fwd)
    return {}

  def _meeting(self, ma: Member, mb: Member) -> dict[int, set[float]]:
    """The lines of two roads out of a node that meet each other there, same kind and style: {way: {m right}}."""
    def lines(m):
      w, fwd = m.ways[0]
      road = self.osm.lanes(w)
      if not road.markings:
        return []
      p, right = self.osm.node_xy(m.start), -_left(m.line.tangent(0.0))
      return [(ln, p + right * ln.offset, w, fwd) for ln in road.lines(FORWARD if fwd else BACKWARD) if ln.kind in (CENTRE, DIVIDER, MEDIAN)]
    out: dict[int, set[float]] = {}
    theirs = lines(mb)
    for ln, p, w, fwd in lines(ma):
      for ln2, q, w2, fwd2 in theirs:
        if ln2.kind == ln.kind and FLIPPED.get(ln2.style, ln2.style) == ln.style and np.hypot(*(p - q)) < MEET:
          out.setdefault(w, set()).add(round(ln.offset if fwd else -ln.offset, 2))  # offsets along the way's own direction
          out.setdefault(w2, set()).add(round(ln2.offset if fwd2 else -ln2.offset, 2))
          break
    return out

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
  that side where none does (through: the nearest straight, as on a skewed junction). Without, the leftmost lane also
  turns left, the rightmost right, and the rest go through
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
          side = [k for k, (t, _) in enumerate(exits) if lo + hi == 0 or (t > 0) == (lo + hi > 0)]
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
