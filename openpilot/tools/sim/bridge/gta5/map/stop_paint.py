"""Stop lines where the game files paint them (polylines.jsonl): a map's stop line nodes (`highway=traffic_signals` /
`stop` with their direction tag, as junctions.py reads them) moved to the painted line across their approach, and
added on approaches with a painted stop line and none in the map.

- A painted stop line is a thick (MIN_WIDTH), straight white line, solid or tiled, across an approach's lanes towards
  its junction (junctions.py's members): within ANGLE of square to the road, at its height (DZ), covering most of
  those lanes (COVER), from the junction's node out to STOP_REACH past its mouth. Lines covering only the other
  direction's lanes are that approach's, not this one's, and a line across the whole road with another junction's area
  behind it about as near as this junction's mouth ahead (OTHER_BEHIND, MOUTH_TIE) is that junction's: traffic
  leaving a junction doesn't stop as it leaves.
- GTA paints the far edge of a crossing at a junction as its stop line: from the first line out from the junction, the
  lines behind it within PAIR_GAP, or within CROSSING_SPAN with a painted crossing between or both across the whole
  road (a crossing's edges painted without stripes), go with it, and the outermost is the stop line. Pieces of one
  line laid end to end count as one.
- A stop line node the map has is moved there when it's within MAX_MOVE (its signal or sign kept, with its direction
  now said by the way it's on). An approach without one gets one where the line covers only its own lanes (a line
  across the whole road is a crossing's edge) with no painted crossing just ahead of it: `traffic_signals` at a
  junction with signals, else `stop`. Elsewhere the map's own stop lines stay as they are.
- The node goes on the approach's way where the line crosses its line: a new node splitting the way (ids from
  `node_ids`, as for tapers) unless one of its nodes is within SNAP. The pieces keep the way's tags, a widening
  lane's start and end widths taken at the cut, and turning back at the new node on a two-way way is forbidden, as
  GTA's links allow none part way along.
"""
import json
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.junctions import FREEWAY, STOP_REACH, Junctions, _left, in_fan
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, FORWARD, OsmLanes, oneway_of

MIN_WIDTH = 0.25  # m: stop lines are painted this thick at least (lane lines 0.1-0.2)
MIN_LENGTH, MAX_LENGTH = 2.0, 40.0  # m
STRAIGHT = 0.9  # of a line's length between its ends at least
ANGLE = 30.0  # deg off square across the road at most
DZ = 2.0  # m between the paint's height and the road's
COVER = 0.6  # of the approach's lanes' width a stop line covers at least
OWN_SIDE = 1.5  # m a new stop line reaches past its lanes at most into the other direction's
PAIR_GAP = 4.0  # m: lines this near behind the first one out from a junction go with it (a stop line behind a crossing)
CROSSING_SPAN = 12.0  # m: as do lines this near with a painted crossing between (its two edges)
MAX_MOVE = 15.0  # m a map stop line is moved at most to the paint
NEW_REACH = 15.0  # m past a junction's mouth a new stop line can be
OTHER_BEHIND = 3.0  # m behind a line across the whole road that another junction's area is looked for
MOUTH_TIE = 0.5  # m: that area about this much further from the line than this junction's mouth still claims it
CROSSING_AHEAD = 6.0  # m: a painted crossing this near ahead of a line makes it the crossing's edge, not a new stop line
MERGE_ALONG, MERGE_GAP = 0.6, 1.5  # m: painted pieces this near along the road, and end to end, are one line
SNAP = 0.5  # m: a stop line this near a node of its way goes on that node
CELL = 25.0  # m


@dataclass
class Line:
  id: int
  pts: np.ndarray  # [n, 2]
  z: float


@dataclass
class Placed:
  way: int
  t: float  # along the way from its first node, 0-1
  node: int | None  # an existing node it goes on, else None for a new one
  tags: dict
  moved_from: list[int]  # the nodes whose stop lines it was
  line: int  # the painted line's id
  xy: np.ndarray
  z: float


def painted_lines(path) -> list[Line]:
  out = []
  with open(path) as f:
    for row in f:
      p = json.loads(row)
      if p['colour'] != 'white' or p['width'] < MIN_WIDTH or not MIN_LENGTH <= p['len'] <= MAX_LENGTH:
        continue
      pts = np.array(p['pts'], float)
      if np.hypot(*(pts[-1, :2] - pts[0, :2])) < STRAIGHT * p['len']:
        continue
      out.append(Line(p['id'], pts[:, :2], float(pts[:, 2].mean())))
  return out


def painted_crossings(path) -> list[tuple[np.ndarray, float]]:
  """The painted crossings' outlines (features.jsonl), with their heights."""
  out = []
  with open(path) as f:
    for row in f:
      d = json.loads(row)
      if d['kind'] == 'crossing' and len(d.get('poly') or ()) >= 3 and 'rail' not in d.get('tex', ''):
        out.append((np.array(d['poly'], float), float(d['z'])))
  return out


def inside(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
  """Which points [K, 2] are inside a polygon (even-odd: the painted crossings' outlines aren't convex)."""
  a, b = polygon, np.roll(polygon, -1, axis=0)
  x, y = points[:, None, 0], points[:, None, 1]
  spans = (a[None, :, 1] > y) != (b[None, :, 1] > y)
  dy = np.where(b[:, 1] == a[:, 1], 1e-12, b[:, 1] - a[:, 1])[None]
  cross = x < a[None, :, 0] + (y - a[None, :, 1]) * (b[None, :, 0] - a[None, :, 0]) / dy
  return (spans & cross).sum(1) % 2 == 1


def _grid(items, points) -> dict[tuple[int, int], list[int]]:
  cells = defaultdict(list)
  for n, item in enumerate(items):
    for c in {(int(x // CELL), int(y // CELL)) for x, y in points(item)}:
      cells[c].append(n)
  return cells


def _cells_along(pts: np.ndarray, pad: float):
  lo, hi = pts.min(0) - pad, pts.max(0) + pad
  return [(cx, cy) for cx in range(int(lo[0] // CELL), int(hi[0] // CELL) + 1) for cy in range(int(lo[1] // CELL), int(hi[1] // CELL) + 1)]


class StopPaint:
  def __init__(self, junctions: Junctions, lines: list[Line], crossings: list[tuple[np.ndarray, float]] | None = None):
    self.j, self.osm = junctions, junctions.osm
    self.lines = lines
    self.line_cells = _grid(lines, lambda line: line.pts)
    self.crossings = crossings or []
    self.crossing_cells = _grid(self.crossings, lambda c: c[0])
    self.ele = {n: float(t['ele']) for n, t in self.osm.data.node_tags.items() if 'ele' in t}
    self.areas = [(j, float(np.mean([self.ele.get(n, 0.0) for n in j.nodes]))) for j in junctions.junctions]
    self.area_cells = _grid([j for j, _ in self.areas], lambda j: j.polygon)

  def _z(self, m, s: float) -> float | None:
    pts = m.line.p[1:-1]
    along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))))
    z = [self.ele.get(n, math.nan) for n in m.nodes]
    v = float(np.interp(s, along, z))
    return None if math.isnan(v) else v

  def _way_at(self, m, s: float) -> tuple[int, float, float]:
    """The member's way at s m from its junction node: (index in m.ways, m along the member to its first node, its length)."""
    pts = m.line.p[1:-1]
    seg = np.hypot(*np.diff(pts, axis=0).T)
    along = np.concatenate(([0.0], np.cumsum(seg)))
    k = int(np.clip(np.searchsorted(along, s, side='right') - 1, 0, len(seg) - 1))
    return k, float(along[k]), float(seg[k])

  def _lanes_at(self, m, k: int) -> tuple[float, float] | None:
    """The lanes towards the junction on the member's k-th way, m right of its line seen arriving."""
    w, fwd = m.ways[k]
    spans = self.osm.lanes(w).ours(BACKWARD if fwd else FORWARD)
    if not spans:
      return None
    return min(sp.left for sp in spans), max(sp.right for sp in spans)

  def candidates(self, m, s_max: float) -> list[tuple[float, Line, float, float]]:
    """Painted stop lines across the member's lanes towards its junction, out to s_max: [(s, line, m it reaches past
    the lanes' left edge, m past their right edge)] by s."""
    pts = m.line.p[1:-1]
    seg = np.hypot(*np.diff(pts, axis=0).T)
    along = np.concatenate(([0.0], np.cumsum(seg)))
    samples = np.array([m.line.at(d) for d in np.arange(0.0, s_max + CELL / 2, CELL / 2)])
    near = {n for c in _cells_along(samples, CELL / 2) for n in self.line_cells.get(c, ())}
    hits = []
    for n in near:
      line = self.lines[n]
      c, d = line.pts.mean(0), line.pts[-1] - line.pts[0]
      d = d / max(np.hypot(*d), 1e-9)
      for k in range(len(seg)):
        if along[k] > s_max or seg[k] < 1e-6:
          continue
        a, u = pts[k], (pts[k + 1] - pts[k]) / seg[k]
        if abs(float(u @ d)) > math.sin(math.radians(ANGLE)):
          continue
        # where the paint's line crosses the way's
        den = u[0] * d[1] - u[1] * d[0]
        if abs(den) < 1e-9:
          continue
        r = c - a
        t = (r[0] * d[1] - r[1] * d[0]) / den
        if not -1e-6 <= t <= seg[k] + 1e-6 or not 0.0 <= along[k] + t <= s_max:
          continue
        s = float(along[k] + t)
        z = self._z(m, s)
        if z is not None and abs(z - line.z) > DZ:  # (a map without heights: any)
          continue
        lanes = self._lanes_at(m, k)
        if lanes is None:
          continue
        right = _left(u)  # left looking out: right of travel towards the junction
        lat = (line.pts - (a + u * t)) @ right
        hits.append((s, line, float(lat.min()), float(lat.max()), lanes))
        break
    # pieces of one line (a line laid as several decals): at the same place, end to end
    hits.sort(key=lambda h: (h[0], h[2]))
    groups = []
    for h in hits:
      g = next((g for g in groups if abs(g[0] - h[0]) <= MERGE_ALONG and h[2] <= g[3] + MERGE_GAP and h[3] >= g[2] - MERGE_GAP), None)
      if g is None:
        groups.append(list(h))
      else:
        g[2], g[3] = min(g[2], h[2]), max(g[3], h[3])
        if h[3] - h[2] > np.ptp(g[1].pts, axis=0).max():  # (the longest piece stands for it)
          g[1] = h[1]
    out = []
    for s, line, lat_lo, lat_hi, (lo, hi) in groups:
      if min(lat_hi, hi) - max(lat_lo, lo) >= COVER * (hi - lo):
        out.append((s, line, lo - lat_lo, lat_hi - hi))
    return sorted(out, key=lambda c: c[0])

  def _crossing_between(self, m, s0: float, s1: float, z: float) -> bool:
    """Whether a painted crossing lies across the member's line between s0 and s1 m from its junction node."""
    p0, p1 = m.line.at(max(s0, 0.0) + 0.3), m.line.at(s1 - 0.3)
    samples = np.array([p0 + (p1 - p0) * f for f in np.linspace(0, 1, 7)])
    for c in _cells_along(samples, 1.0):
      for n in self.crossing_cells.get(c, ()):
        poly, zc = self.crossings[n]
        if abs(zc - z) <= DZ and inside(samples, poly).any():
          return True
    return False

  def _other_junction_behind(self, j, m, s: float, z: float) -> float | None:
    """How far behind s on the member (out from its junction) another junction's area is, within OTHER_BEHIND."""
    ds = np.linspace(0.1, OTHER_BEHIND, 28)
    samples = np.array([m.line.at(s + d) for d in ds])
    best = None
    for n in {n for c in _cells_along(samples, 1.0) for n in self.area_cells.get(c, ())}:
      other, zo = self.areas[n]
      if other is not j and abs(zo - z) <= DZ + 1.0 and (hit := in_fan(samples, other.centre, other.polygon)).any():
        d = float(ds[np.argmax(hit)])
        best = d if best is None else min(best, d)
    return best

  def _one_crossing(self, m, a, b) -> bool:
    """Whether candidate b, behind a, goes with it: near it, or a crossing's far edge (a crossing painted between, or
    both lines across the whole road, as GTA paints a crossing's two edges without stripes)."""
    gap = b[0] - a[0]
    if gap <= PAIR_GAP:
      return True
    if gap > CROSSING_SPAN:
      return False
    whole = all(c[2] > OWN_SIDE or c[3] > OWN_SIDE for c in (a, b))
    return whole or self._crossing_between(m, a[0], b[0], a[1].z)

  def place(self) -> tuple[list[Placed], dict]:
    """Where each approach's stop line goes, and counts."""
    tags_of = self.osm.data.node_tags
    out, counts = [], defaultdict(int)
    replaced = defaultdict(int)  # node -> its stop lines moved
    stops_from = defaultdict(int)  # node -> its stop lines
    for j in self.j.junctions:
      for st in j.stops:
        if st.node is not None:
          stops_from[st.node] += 1
    for j in self.j.junctions:
      signals = any(st.signal for st in j.stops)
      for arm in j.arms:
        for m in arm.members:
          if self.j.ways[m.ways[0][0]][0].get('highway') in FREEWAY or self._lanes_at(m, 0) is None:
            continue
          mine = [st for st in j.stops if st.member is m]
          # from GTA's stop line node: the line nearest it; at the junction's own node (every way in), none of ours
          if any(st.node is not None and st.node in j.nodes for st in mine):
            continue
          at_node = [st for st in mine if st.node is not None]
          s_max = min(m.line.length, m.trim + (STOP_REACH if at_node else NEW_REACH))
          found = self.candidates(m, s_max)
          # a line across the whole road as near another junction's area behind it as this one's mouth is that one's
          mouths = [c for c in found if (c[2] > OWN_SIDE or c[3] > OWN_SIDE) and
                    (d := self._other_junction_behind(j, m, c[0], c[1].z)) is not None and d < c[0] - m.trim + MOUTH_TIE]
          if mouths:
            counts["lines at another junction's mouth left out"] += len(mouths)
            found = [c for c in found if c not in mouths]
          if not found:
            counts['no paint' if at_node else 'no stop line, no paint'] += 1
            continue
          k = 0
          while k + 1 < len(found) and self._one_crossing(m, found[k], found[k + 1]):
            k += 1
          s, line, past_left, past_right = found[k]
          if at_node:
            st = min(at_node, key=lambda st: abs(s - self.j._along(m, m.nodes.index(st.node))))
            if abs(s - self.j._along(m, m.nodes.index(st.node))) > MAX_MOVE:
              counts['paint too far from the stop line'] += 1
              continue
            highway = tags_of[st.node]['highway']
          else:
            z = self._z(m, s)
            z = line.z if z is None else z
            if past_left > OWN_SIDE or past_right > OWN_SIDE:
              counts['new: line across the whole road'] += 1
              continue
            if self._crossing_between(m, s - CROSSING_AHEAD, s, z):
              counts['new: a crossing ahead'] += 1
              continue
            highway = 'traffic_signals' if signals else 'stop'
          kw, a0, length = self._way_at(m, s)
          w, outward = m.ways[kw]
          frac = min(max((s - a0) / max(length, 1e-6), 0.0), 1.0)
          t = frac if outward else 1.0 - frac  # along the way's own direction
          moved = [st.node for st in at_node]
          node = None
          if (s - a0) <= SNAP and kw > 0:
            node = m.nodes[kw]
          elif a0 + length - s <= SNAP and kw + 1 < len(m.nodes) - 1:
            node = m.nodes[kw + 1]
          if node is not None and (len(self.j.steps.get(node, ())) != 2 or (node in stops_from and node not in moved)):
            node = None  # (not where other roads join, nor on another approach's stop line)
          if node is None and min(frac, 1.0 - frac) * length < 0.05:
            counts['at a node where roads join'] += 1
            continue
          direction = 'forward' if not outward else 'backward'  # the way runs out of the junction: traffic in goes backward
          tags = {'highway': highway, 'traffic_signals:direction' if highway == 'traffic_signals' else 'direction': direction,
                  'source:position': 'survey'}
          for n in moved:
            replaced[n] += 1
          z = self._z(m, s)
          out.append(Placed(w, t, node, tags, moved, line.id, m.line.at(s), line.z if z is None else z))
          counts['moved to the paint' if at_node else f'new ({highway})'] += 1
    # a node's tag goes only once all its stop lines have moved
    keep = {n for n in replaced if replaced[n] < stops_from[n]}
    for p in out:
      if set(p.moved_from) & keep:
        p.moved_from = [n for n in p.moved_from if n not in keep]
        counts['moved, its node keeping its tag too'] += 1
    return out, dict(counts)


SPLIT_KEYS = (':start', ':end')


def _interpolate(tags: dict, t: float) -> tuple[dict, dict]:
  """A way's tags for its two pieces split at t: widths at its ends (`...:start` / `...:end`) taken at the cut."""
  first, second = dict(tags), dict(tags)
  for key in [k for k in tags if k.endswith(':start')]:
    base = key[:-len(':start')]
    if f'{base}:end' not in tags:
      continue
    a, b = tags[key].split('|'), tags[f'{base}:end'].split('|')
    if len(a) != len(b):
      continue
    try:
      mid = '|'.join(f'{float(x) + (float(y) - float(x)) * t:.2f}'.rstrip('0').rstrip('.') for x, y in zip(a, b, strict=True))
    except ValueError:
      continue
    first[f'{base}:end'] = mid
    second[key] = mid
  return first, second


def rewrite(path: str, placed: list[Placed], new_node_id, to_lat_lon, remap_restrictions) -> tuple[dict, dict]:
  """Writes the map at `path` again with the stop lines `placed`: tags moved, new nodes splitting their ways. Returns
  {new node id: (x, y, z)} and {way id: [(piece id, first node, last node)]} for the ways split."""
  import osmium
  nodes, ways, rels = [], {}, []
  for o in osmium.FileProcessor(path):
    if o.is_node():
      nodes.append([o.id, o.location.lon, o.location.lat, dict(o.tags)])
    elif o.is_way():
      ways[o.id] = ([n.ref for n in o.nodes], dict(o.tags))
    elif o.is_relation():
      rels.append((o.id, dict(o.tags), [(m.type, m.ref, m.role) for m in o.members]))
  node_tags = {n[0]: n[3] for n in nodes}
  for p in placed:
    for n in p.moved_from:
      for key in ('highway', 'traffic_signals:direction', 'direction', 'source:position'):
        node_tags[n].pop(key, None)
  cuts = defaultdict(list)
  new_nodes = {}
  for p in placed:
    if p.node is not None:
      node_tags[p.node].update(p.tags)
      continue
    nid = new_node_id()
    new_nodes[nid] = (float(p.xy[0]), float(p.xy[1]), float(p.z))
    lat, lon = to_lat_lon(float(p.xy[0]), float(p.xy[1]))
    nodes.append([nid, lon, lat, {**p.tags, 'ele': f'{p.z:.1f}'}])
    node_tags[nid] = nodes[-1][3]
    cuts[p.way].append((p.t, nid))
  next_way = max(ways) + 1
  pieces, ends_of, u_turns = {}, {}, []
  for wid, (refs, tags) in list(ways.items()):
    ends_of[wid] = (refs[0], refs[-1])
    if wid not in cuts or len(refs) != 2:
      continue
    ids, rows, rest, at = [wid], [], tags, 0.0
    for t, _ in sorted(cuts[wid]):
      first, rest = _interpolate(rest, (t - at) / max(1.0 - at, 1e-9))
      rows.append(first)
      ids.append(next_way)
      next_way += 1
      at = t
    rows.append(rest)
    chain = [refs[0], *[nid for _, nid in sorted(cuts[wid])], refs[1]]
    pieces[wid] = []
    two_way = not oneway_of(tags)
    for n, (pid, t) in enumerate(zip(ids, rows, strict=True)):
      ways[pid] = ([chain[n], chain[n + 1]], t)
      ends_of[pid] = (chain[n], chain[n + 1])
      pieces[wid].append((pid, chain[n], chain[n + 1]))
      if two_way:
        for end in (chain[n], chain[n + 1]):
          if end in new_nodes:
            u_turns.append(('no_u_turn', pid, [('n', end)], pid))
  restrictions, other = [], []
  for rid, tags, members in rels:
    if tags.get('type') == 'restriction':
      wf = next(r for t, r, role in members if role == 'from')
      wt = next(r for t, r, role in members if role == 'to')
      via = [('n' if t == 'n' else 'w', r) for t, r, role in members if role == 'via']
      restrictions.append((rid, tags, (tags['restriction'], wf, via, wt)))
    else:
      other.append((rid, tags, members))
  ids = {wid: [p[0] for p in ps] for wid, ps in pieces.items()}
  remapped = remap_restrictions([r[2] for r in restrictions], ends_of, ids) if ids else [r[2] for r in restrictions]
  w = osmium.SimpleWriter(path, overwrite=True)
  for nid, lon, lat, tags in sorted(nodes, key=lambda n: n[0]):
    w.add_node(osmium.osm.mutable.Node(id=nid, version=1, location=(lon, lat), tags=tags))
  for wid in sorted(ways):
    refs, tags = ways[wid]
    w.add_way(osmium.osm.mutable.Way(id=wid, version=1, nodes=refs, tags=tags))
  next_rel = max([r[0] for r in rels], default=0) + 1
  for (rid, tags, _), (_, wf, via, wt) in zip(restrictions, remapped, strict=True):
    w.add_relation(osmium.osm.mutable.Relation(id=rid, version=1, tags=tags, members=[
      ('w', wf, 'from'), *[(t, ref, 'via') for t, ref in via], ('w', wt, 'to')]))
  for rid, tags, members in other:
    members = [(t, p, role) for t, ref, role in members for p in (ids.get(ref, [ref]) if t == 'w' else [ref])]
    w.add_relation(osmium.osm.mutable.Relation(id=rid, version=1, tags=tags, members=members))
  for kind, wf, via, wt in u_turns:
    w.add_relation(osmium.osm.mutable.Relation(id=next_rel, version=1, tags={'type': 'restriction', 'restriction': kind},
                                               members=[('w', wf, 'from'), *[(t, ref, 'via') for t, ref in via], ('w', wt, 'to')]))
    next_rel += 1
  w.close()
  return new_nodes, pieces


def stop_lines(path: str, lines_path: str, features_path: str | None, project) -> tuple[list[Placed], dict]:
  """The stop lines of the map at `path` placed by the game files' paint (StopPaint.place)."""
  osm = OsmLanes.load(path, project)
  junctions = Junctions(osm)
  crossings = painted_crossings(features_path) if features_path else []
  return StopPaint(junctions, painted_lines(lines_path), crossings).place()
