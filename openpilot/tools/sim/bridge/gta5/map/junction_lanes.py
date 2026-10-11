"""Lanes across GTA's junction links that bend off their road's line to its junction node, put back on the line.

GTA meets every link of a junction at one node. Where a divided road's carriageways cross another road at a node in
their middle (the tram crossing on Innocence Blvd, Las Lagunas Blvd), each carriageway's last link before the node
and its first after it angle across to the node, so the lanes laid along them jog towards it and back (8 m at
Innocence Blvd, over both of its raised islands), while the lanes painted run straight on through the junction.

- A bend: a one-way way into a junction node and one out of it (not a motorway's), each followed out from the node
  through nodes where the road just carries on to the first node within STRETCH_MAX where its heading turns BEND_TURN or
  more (the bend), the road beyond straight (STRAIGHT_TOL over LINE_M). The two roads beyond run on into each other:
  within HEADING_TOL of one heading and LINE_TOL of one line, the line through the two bends. The node lies BEND_MIN or
  more off that line, and the ways in and out have as many lanes our way.
- The ways from each bend to the node get `placement:offset:start` / `:end` (osm_lanes: the offset varying along the
  way) moving their lanes across onto the line, square across each way: nothing at the bend, at the node as far as it
  lies off the line, linearly between. Each way's own `placement:offset` is added, and `placement:offset` itself is
  their mean, for readers of one cross-section a way.
- Only where the game files say so (`Evidence`): their paint lines along the ways (polylines.jsonl) run over the
  middle of the moved lanes (INSIDE from their edges) at most OVER_KEPT as often as over the lanes as GTA laid them, on
  OVER_BETTER of the cross-sections fewer, and no more often yellow; the lanes' edges are on average no more than
  PAINT_WORSE further from the paint lines; and their kerbs and raised islands (roadpaint's tiles) are under no more of
  the lanes (RAISED_WORSE).
"""
import json
import math
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, FORWARD, OsmLanes, Section, oneway_of, signed_metres

BEND_TURN = 5.0  # deg: a node where a road out of a junction node turns this much is where it bends to the node
STRETCH_MAX = 45.0  # m from the junction node to its bend, at most
LINE_M = 15.0  # m of road beyond the bend its line is taken over ...
STRAIGHT_TOL = 4.0  # deg: ... bending no more than this from it
HEADING_TOL = 10.0  # deg between the roads beyond the bends in and out, at most
LINE_TOL = 1.5  # m between their lines, at most
BEND_MIN = 1.0  # m the junction node lies off the line at least
FREEWAYS = ('motorway', 'motorway_link')
SAMPLE = 1.0  # m between the cross-sections the evidence is read at
REACH = 10.0  # m either side of a way the paint is read
ALONG = 25.0  # deg: paint lines running within this of the way's heading are lines along it
PAINT_CAP = 2.0  # m: a paint line further than this from the lanes' edges counts as this far
PAINT_WORSE = 0.25  # m further, on average, the paint lines along the ways may sit from the moved lanes' edges
RAISED_WORSE = 0.01  # of the lanes' area the move may put over raised surface (kerb, pavement), at most
INSIDE = 0.6  # m from every lane edge a paint line along a way is over a lane, not on its lines
OVER_KEPT = 0.5  # of the paint over the lanes as laid that may be left over the moved ones, at most
OVER_BETTER = 0.2  # of the cross-sections read the paint over the lanes must drop by, at least
LANE_INSET = 0.4  # m in from each lane's edges its area is read over (a car keeps off its lines)


@dataclass
class Bend:
  node: int  # the junction node
  ins: list[int]  # the ways from the bend in to the node, in order of travel
  outs: list[int]  # and from the node to the bend out
  line: tuple[np.ndarray, np.ndarray]  # the road's line through the junction: a point (the bend in) and its heading
  off: float  # m the node lies right of the line


def _unit(v) -> np.ndarray:
  return np.asarray(v, float) / max(float(np.hypot(*v)), 1e-9)


def _right_of(p, a, u) -> float:
  d = np.asarray(p, float) - a
  return float(d[0] * u[1] - d[1] * u[0])


def _chain(osm: OsmLanes, node: int, way: int, reach: float) -> tuple[list[tuple[int, int, int]], np.ndarray]:
  """From a node outward along a way and on through nodes where the road just carries on: [(way, its node nearer the
  start, its far node)], and the points from the node."""
  out, pts, cur, length = [], [osm.node_xy(node)], node, 0.0
  for _ in range(32):
    refs = osm.ways[way][1]
    if cur not in (refs[0], refs[-1]):
      break
    seq = refs if refs[0] == cur else refs[::-1]
    for n in seq[1:]:
      q = osm.node_xy(n)
      length += float(np.hypot(*(q - pts[-1])))
      pts.append(q)
    out.append((way, cur, seq[-1]))
    cur = seq[-1]
    nxt = osm._other(way, cur)
    if length >= reach or nxt is None or nxt[0] == way:
      break
    way = nxt[0]
  return out, np.array(pts)


def _bend(pts: np.ndarray) -> tuple[int, np.ndarray] | None:
  """(the index of the bend among the points out from the node, the road's heading beyond it, outward)."""
  s = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))))
  for k in range(1, len(pts) - 1):
    if s[k] > STRETCH_MAX:
      return None
    a, b = _unit(pts[k] - pts[k - 1]), _unit(pts[k + 1] - pts[k])
    if math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0)))) < BEND_TURN:
      continue
    if s[-1] - s[k] < 0.6 * LINE_M:
      return None
    far = min(int(np.searchsorted(s, s[k] + LINE_M)), len(pts) - 1)
    u = _unit(pts[far] - pts[k])
    if any(math.degrees(math.acos(float(np.clip(_unit(pts[j + 1] - pts[j]) @ u, -1.0, 1.0)))) > STRAIGHT_TOL for j in range(k, far)):
      return None
    return k, u
  return None


def _ours(osm: OsmLanes, way: int, node: int, into: bool) -> int:
  refs = osm.ways[way][1]
  d = FORWARD if (refs[-1] == node) == into else BACKWARD
  return Section.of(osm.lanes(way), d).lanes


def bends(osm: OsmLanes) -> list[Bend]:
  """The bends at the map's junction nodes (see the module's docstring)."""
  out = []
  for node in sorted(n for n, d in osm.degree.items() if d >= 3):
    ins, outs = [], []
    for w in osm._ends_at(node):
      tags, refs = osm.ways[w]
      one = oneway_of(tags)
      if not one or tags.get('highway') in FREEWAYS:
        continue
      chain, pts = _chain(osm, node, w, STRETCH_MAX + LINE_M + 5.0)
      if (found := _bend(pts)) is None:
        continue
      into = (refs[-1] == node) == (one == 1)
      (ins if into else outs).append((w, chain[:found[0]], pts[found[0]], found[1]))
    for wi, chain_in, b, ui in ins:
      for wo, chain_out, e, uo in outs:
        t_in, t_out = -ui, uo  # the roads' headings, travelling
        if math.degrees(math.acos(float(np.clip(t_in @ t_out, -1.0, 1.0)))) > HEADING_TOL:
          continue
        if abs(_right_of(e, b, t_in)) > LINE_TOL or abs(_right_of(b, e, t_out)) > LINE_TOL:
          continue
        u = _unit(e - b)
        off = _right_of(osm.node_xy(node), b, u)
        if abs(off) < BEND_MIN or _ours(osm, wi, node, True) != _ours(osm, wo, node, False):
          continue
        out.append(Bend(node, [c[0] for c in chain_in][::-1], [c[0] for c in chain_out], (b, u), off))
  return out


def shifts(osm: OsmLanes, found: list[Bend]) -> dict[int, tuple[float, float]]:
  """{way: (placement:offset at its first node, at its last)} moving each bend's ways' lanes onto its line. A way two
  bends would move differently is left out."""
  out: dict[int, tuple[float, float]] = {}
  clash: set[int] = set()
  for bend in found:
    a, u = bend.line
    n_line = np.array([u[1], -u[0]])  # right of the line
    for w in bend.ins + bend.outs:
      tags, refs = osm.ways[w]
      p = osm.way_points(w)
      r = np.array([_unit(p[-1] - p[0])[1], -_unit(p[-1] - p[0])[0]])  # right of the way, along its refs
      across = float(r @ n_line)
      if abs(across) < 0.5:
        continue
      base = signed_metres(tags.get('placement:offset')) or 0.0
      # square across the way onto the line at each end (the lane line through the node, mitred there, falls about
      # (1 - cos) of the turn between the ways in and out off it: under a metre at Innocence Blvd's 38 degrees)
      ends = [round(base - _right_of(osm.node_xy(n), a, u) / across, 2) for n in (refs[0], refs[-1])]
      prev = out.get(w)
      if prev is not None and max(abs(prev[0] - ends[0]), abs(prev[1] - ends[1])) > 0.05:
        clash.add(w)
      out[w] = (ends[0], ends[1])
  for w in clash:
    del out[w]
  return out


def tags_for(osm: OsmLanes, moved: dict[int, tuple[float, float]]) -> dict[int, dict[str, str]]:
  return {w: {'placement:offset': f'{(s + e) / 2:g}', 'placement:offset:start': f'{s:g}', 'placement:offset:end': f'{e:g}'}
          for w, (s, e) in moved.items()}


class Evidence:
  """The game files' paint lines (polylines.jsonl) and surfaces (roadpaint's tiles), read along ways."""
  CELL = 32.0

  def __init__(self, lines_path, tiles_dir):
    sys.path.insert(0, str(pathlib.Path(__file__).parent / 'roadpaint'))
    import rp
    self.rp = rp
    self.tiles = rp.Tiles(tiles_dir, cache=64)
    segs, yellow = [], []
    with open(lines_path) as f:
      for row in f:
        p = json.loads(row)
        if p['colour'] not in ('white', 'yellow'):
          continue
        pts = np.array(p['pts'], float)
        for a, b in zip(pts[:-1], pts[1:], strict=True):
          segs.append((*a, *b))
          yellow.append(p['colour'] == 'yellow')
    self.segs = np.array(segs, float).reshape(-1, 6)
    self.yellow = np.array(yellow, bool)
    self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    lo, hi = np.minimum(self.segs[:, :2], self.segs[:, 3:5]), np.maximum(self.segs[:, :2], self.segs[:, 3:5])
    for k in range(len(self.segs)):
      for cx in range(int(lo[k, 0] // self.CELL), int(hi[k, 0] // self.CELL) + 1):
        for cy in range(int(lo[k, 1] // self.CELL), int(hi[k, 1] // self.CELL) + 1):
          self.cells[(cx, cy)].append(k)

  def _near(self, lo, hi) -> np.ndarray:
    ids = {k for cx in range(int(lo[0] // self.CELL), int(hi[0] // self.CELL) + 1)
           for cy in range(int(lo[1] // self.CELL), int(hi[1] // self.CELL) + 1) for k in self.cells.get((cx, cy), ())}
    ids = np.array(sorted(ids), np.int64)
    return (self.segs[ids], ids) if len(ids) else (np.zeros((0, 6)), ids)

  def read(self, osm: OsmLanes, way: int, offsets: tuple[float, float] | None) -> tuple:
    """With the way's placement:offset `offsets` at its ends (None: as tagged), read every SAMPLE m: (summed m from
    each paint line along the way to the nearest lane edge, how many, its lanes' samples over raised surface, how
    many samples, white and yellow paint lines over the middle of a lane, how many cross-sections)."""
    tags, refs = osm.ways[way]
    p = osm.way_points(way)
    z = np.array([float(osm.data.node_tags.get(n, {}).get('ele', 'nan')) for n in refs])
    z = np.where(np.isfinite(z), z, np.nanmean(z) if np.isfinite(z).any() else 0.0)
    along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))))
    length = float(along[-1])
    if length < 2 * SAMPLE:
      return 0.0, 0, 0.0, 0, 0, 0, 0
    base = signed_metres(tags.get('placement:offset')) or 0.0
    sec = Section.of(osm.lanes(way), FORWARD)
    edges = sorted({round(e, 3) for sp in sec.spans for e in (sp.left, sp.right)})
    bands = [(sp.left + LANE_INSET, sp.right - LANE_INSET) for sp in sec.spans if sp.right - sp.left > 2 * LANE_INSET]
    s = np.arange(SAMPLE / 2, length, SAMPLE)
    k = np.clip(np.searchsorted(along, s, side='right') - 1, 0, len(p) - 2)
    u = (p[k + 1] - p[k]) / np.maximum(np.hypot(*(p[k + 1] - p[k]).T), 1e-9)[:, None]
    q = p[k] + u * (s - along[k])[:, None]
    r = np.column_stack([u[:, 1], -u[:, 0]])
    zz = np.interp(s, along, z)
    shift = np.zeros(len(s)) if offsets is None else np.interp(s, [0.0, length], offsets) - base
    lo, hi = q.min(0) - REACH, q.max(0) + REACH
    segs, ids = self._near(lo, hi)
    paint, n_paint = 0.0, 0
    inside = [0, 0]  # paint lines along the way over its lanes: white, yellow
    if len(segs):
      a, d = segs[:, :2], segs[:, 3:5] - segs[:, :2]
      dl = np.maximum(np.hypot(*d.T), 1e-9)
      for i in range(len(s)):
        along_ok = np.abs((d @ u[i]) / dl) >= math.cos(math.radians(ALONG))
        c0 = q[i] - r[i] * REACH
        e = r[i] * 2 * REACH
        den = d[:, 0] * e[1] - d[:, 1] * e[0]
        ok = along_ok & (np.abs(den) > 1e-9)
        den = np.where(ok, den, 1.0)
        t = ((c0[0] - a[:, 0]) * e[1] - (c0[1] - a[:, 1]) * e[0]) / den
        v = ((c0[0] - a[:, 0]) * d[:, 1] - (c0[1] - a[:, 1]) * d[:, 0]) / den
        zs = segs[:, 2] + t * (segs[:, 5] - segs[:, 2])
        hit = ok & (t >= 0) & (t <= 1) & (v >= 0) & (v <= 1) & (np.abs(zs - zz[i]) < 2.0)
        for off, yellow in zip(-REACH + 2 * REACH * v[hit], self.yellow[ids[hit]], strict=True):
          near = min(abs(off - (x + shift[i])) for x in edges)
          paint += min(near, PAINT_CAP)
          n_paint += 1
          if near > INSIDE and edges[0] + shift[i] < off < edges[-1] + shift[i]:  # over a lane, not on its lines
            inside[int(yellow)] += 1
    raised, n_raised = 0.0, 0
    if bands:
      across = np.array([x for lo_, hi_ in bands for x in np.linspace(lo_, hi_, 5)])
      pts = q[:, None] + r[:, None] * (across[None] + shift[:, None])[..., None]
      code, _ = self.tiles.sample(pts[..., 0].ravel(), pts[..., 1].ravel(), np.repeat(zz, len(across)), zwin=1.5)
      cls = (code & 7).reshape(len(s), len(across))
      raised = float(np.isin(cls, (self.rp.KERB, self.rp.WALK)).sum())
      n_raised = int(cls.size)
    return paint, n_paint, raised, n_raised, inside[0], inside[1], len(s)

  def judge(self, osm: OsmLanes, ways: list[int], moved: dict[int, tuple[float, float]]) -> tuple[bool, dict]:
    """Whether the game files back moving these ways' lanes: (verdict, the readings as laid and as moved)."""
    before, after = np.zeros(7), np.zeros(7)
    for w in ways:
      before += self.read(osm, w, None)
      after += self.read(osm, w, moved[w])
    paint_b = before[0] / before[1] if before[1] else None
    paint_a = after[0] / after[1] if after[1] else None
    raised_b = before[2] / before[3] if before[3] else 0.0
    raised_a = after[2] / after[3] if after[3] else 0.0
    over_b, over_a, sections = before[4] + before[5], after[4] + after[5], max(before[6], 1.0)
    clearer = over_a <= OVER_KEPT * over_b and over_b - over_a >= OVER_BETTER * sections
    ok = clearer and raised_a <= raised_b + RAISED_WORSE and after[5] <= before[5] and \
      not (paint_b is not None and paint_a is not None and paint_a > paint_b + PAINT_WORSE)
    return bool(ok), {'paint': [paint_b and round(float(paint_b), 2), paint_a and round(float(paint_a), 2), int(before[1])],
                      'raised': [round(float(raised_b), 3), round(float(raised_a), 3)],
                      'over': [int(before[4]), int(before[5]), int(after[4]), int(after[5]), int(before[6])]}


def placed(osm: OsmLanes, lines_path, tiles_dir) -> tuple[dict[int, dict[str, str]], list[dict]]:
  """The tags moving the lanes of the map's bends the game files back ({way: tags}), and a row per bend: where, how
  far, the verdict and the readings."""
  found = bends(osm)
  evidence = Evidence(lines_path, tiles_dir)
  rows, kept = [], {}
  for bend in found:
    moved = shifts(osm, [bend])
    ways = [w for w in bend.ins + bend.outs if w in moved]
    if not ways:
      continue
    ok, why = evidence.judge(osm, ways, moved)
    rows.append({'node': bend.node, 'xy': [round(float(v), 1) for v in osm.node_xy(bend.node)], 'off': round(bend.off, 2),
                 'ins': bend.ins, 'outs': bend.outs, 'moved': {w: moved[w] for w in ways}, 'ok': ok, **why})
    if ok:
      kept.update({w: moved[w] for w in ways})
  # a way two kept bends move differently is left as it was
  again = shifts(osm, [b for b, row in zip(found, rows, strict=False) if row['ok']]) if kept else {}
  kept = {w: v for w, v in kept.items() if w in again}
  return tags_for(osm, kept), rows


def add(path: str, tags: dict[int, dict[str, str]]) -> int:
  """Writes the map at `path` again with the ways' tags updated; returns how many ways changed."""
  import osmium
  nodes, ways, rels = [], [], []
  for o in osmium.FileProcessor(path):
    if o.is_node():
      nodes.append(osmium.osm.mutable.Node(id=o.id, version=1, location=(o.location.lon, o.location.lat), tags=dict(o.tags)))
    elif o.is_way():
      t = dict(o.tags)
      if o.id in tags:
        t.update(tags[o.id])
      ways.append(osmium.osm.mutable.Way(id=o.id, version=1, nodes=[n.ref for n in o.nodes], tags=t))
    elif o.is_relation():
      rels.append(osmium.osm.mutable.Relation(id=o.id, version=1, tags=dict(o.tags), members=[(m.type, m.ref, m.role) for m in o.members]))
  w = osmium.SimpleWriter(path, overwrite=True)
  for n in nodes:
    w.add_node(n)
  for way in ways:
    w.add_way(way)
  for r in rels:
    w.add_relation(r)
  w.close()
  return sum(1 for way in ways if way.id in tags)
