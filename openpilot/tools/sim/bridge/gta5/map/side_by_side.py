"""One-way ways side by side on one road surface, as GTA draws a freeway (a link per lane or two, with lane changes as
links between them) or as a turn lane mapped as its own way beside its road: a kerb is drawn only where the surface
ends. A way's kerb is left out where it lies on, or within SHARED of, the carriageway of another one-way way running the
same way on its level, other than the ways it carries on from or into. Where two such ways meet edge to edge, the way on
the left draws the line between them as a lane line: dashed, solid where change:lanes says either outer lane there may
not cross it. Only standard tags are read. Real maps don't split a carriageway without a physical separation between
its parts, so there it rarely applies.

Where a one-way way parts into several (a diverge: GTA starts each branch from the middle of the carriageway it leaves,
not from its own lanes there), the branches' kerbs are left out where they lie inside that carriageway carried on
straight past its end (by GORE m, and SHARED in from its edges, so a branch running on along its edge keeps its kerb):
they cross its lanes until the branches part, at the gore. Likewise where several merge into one, back from its start.

On a freeway (junctions.Junctions.freeway: a motorway or one-way trunk road, or its link) a kerb within GORE_SHARED of
another carriageway running its way on its level (but not on it, or within SHARED) faces a flush shoulder or gore between
them, not a barrier: it's drawn as the solid line painted at the lanes' edge there, not as a kerb.

A one-way way's kerb is also left out on a two-way road's carriageway, running along it either way: GTA lays a turn bay
beside a two-way road as a one-way way leaving it from its middle, with lane changes to and from it across its lanes
(junctions.py's lane_split: no junction there), and their kerbs crossed its lanes.

GTA also lays lane changes (junctions.lane_changes: a one-way way up to LANE_CHANGE_M long turning off a carriageway
that goes on, into another that came from elsewhere, all running about one way) across the painted gore where two carriageways part or meet, and
their kerbs crossed it. One with a stretch over GAP_M on no other carriageway (`across`) is part of the surface: it has
no kerbs, covers no other way's kerb and parts no carriageway, so the gore's edges are the carriageways' own kerbs.
"""
from collections import defaultdict

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.junctions import SAME_WAY, Junctions, densify, in_fan, lane_changes
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import FORWARD, oneway_of

# m: a kerb this near another way's carriageway, or on it, is inside the road surface (lanes at their class
# width often fall short of the paint between GTA's links)
SHARED = 1.5
MEETS = 1.5  # m a kerb may be on the carriageway beside it (lanes placed to a lane's edge or middle) and still be its edge
PARALLEL = 10.0  # deg: a way meeting another this near parallel shares a lane line with it, rather than crossing to it
LEVEL = 2.5  # m of height between ways on one level
STEP = 0.5  # m: kerbs are cut to this
MIN_PIECE = 0.3  # m
CELL = 25.0  # m
GORE = 40.0  # m a carriageway that parts or merges is carried on past its end, or back from its start
GAP_M = 1.0  # m of a lane change on no carriageway (but other lane changes'): it crosses a gore
CHANGE_M = 40.0  # m: a city lane change between carriageways is no longer than this
JOIN_TURN = 20.0  # deg: a way leaving a run's start or joining its end at more than this crosses its kerb only there
GORE_SHARED = 5.0  # m: a freeway's kerb this near another carriageway running its way is the edge of a gore or shoulder


class SideBySide:
  """The one-way ways among `ways` (way ids of an osm_lanes.OsmLanes) and their carriageways, a quadrilateral for each
  segment; `layer_of(way)` is its layer."""
  def __init__(self, osm, ways, layer_of):
    self.osm, self.layer_of = osm, layer_of
    self.first, self.last = {}, {}
    self._crosses: dict[int, bool] = {}
    self.two_way: set[int] = set()  # the two-way roads among them, whose carriageways a one-way way's kerb can lie on
    # each segment's way, middle, carriageway widened by SHARED, the strip along its left edge, unit heading, layer, height
    self.quads: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float | None]] = []
    self._cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    self.ends_at, self.starts_at = defaultdict(list), defaultdict(list)  # node -> the one-way ways ending, starting there
    # node -> the carriageways that part or merge there, carried on past it: (inner quad, unit heading, layer, height)
    self.gores: dict[int, list[tuple[np.ndarray, np.ndarray, int, float | None]]] = defaultdict(list)
    for wid in ways:
      tags, refs = osm.ways[wid]
      if oneway_of(tags) != 1 or len(refs) < 2:
        continue
      self.first[wid], self.last[wid] = refs[0], refs[-1]
      self.ends_at[refs[-1]].append(wid)
      self.starts_at[refs[0]].append(wid)
    changes = lane_changes(osm, list(self.first))
    for wid in self.first:
      if wid not in changes:
        self._add_quads(wid)
    # the lane changes across a gap between the carriageways (a gore) rather than over their lanes
    self.across = {w for w in changes if self._gap(w) > GAP_M}
    for wid in sorted(changes - self.across):
      self._add_quads(wid)
    for wid in self.first:
      refs = osm.ways[wid][1]
      lo, hi = osm.lanes(wid).edges(FORWARD)
      if hi - lo <= 2 * SHARED or wid in self.across:
        continue
      pts, z = osm.xy[osm.data.index(refs)], self._heights(refs)
      for node, (p, q), zz in ((refs[-1], pts[-2:], None if z is None else z[-1]), (refs[0], pts[1::-1], None if z is None else z[0])):
        parts = [w for w in (self.starts_at[node] if node == refs[-1] else self.ends_at[node]) if w not in self.across]
        if len(parts) < 2 or np.hypot(*(q - p)) < 0.1:
          continue
        u = (q - p) / np.hypot(*(q - p))  # out past the node, the way it's carried on
        r = np.array([u[1], -u[0]]) * (1.0 if node == refs[-1] else -1.0)  # right of its direction of travel
        far = q + u * GORE
        near = q - u * SHARED  # from just before its end: a branch's kerb starts beside its first node
        inner = np.array([near + r * (lo + SHARED), far + r * (lo + SHARED), far + r * (hi - SHARED), near + r * (hi - SHARED)])
        self.gores[node].append((inner, u if node == refs[-1] else -u, layer_of(wid), None if zz is None else float(zz)))
    for wid in ways:
      tags, refs = osm.ways[wid]
      if oneway_of(tags) == 0 and len(refs) >= 2:
        self.two_way.add(wid)
        self._add_quads(wid)
    # a two-way road that a turn bay parts from (or joins) where it carries on as another, narrower: carried on past
    # the node both ways, as a one-way carriageway that parts is (GTA starts the bay from the road's middle)
    at = defaultdict(list)
    for wid in ways:
      refs = osm.ways[wid][1]
      at[refs[0]].append(wid)
      at[refs[-1]].append(wid)
    self.node_ways = {n: len(v) for n, v in at.items()}
    for wid in self.two_way:
      refs = osm.ways[wid][1]
      lo, hi = osm.lanes(wid).edges(FORWARD)
      pts, z = osm.xy[osm.data.index(refs)], self._heights(refs)
      for node, (p, q), zz in ((refs[-1], pts[-2:], None if z is None else z[-1]), (refs[0], pts[1::-1], None if z is None else z[0])):
        others = [w for w in at[node] if w != wid]
        if sum(w in self.two_way for w in others) != 1 or not any(w in self.first for w in others) or np.hypot(*(q - p)) < 0.1:
          continue
        u = (q - p) / np.hypot(*(q - p))  # out past the node
        r = np.array([u[1], -u[0]]) * (1.0 if node == refs[-1] else -1.0)  # right of the way's direction
        far, near = q + u * GORE, q - u * SHARED
        inner = np.array([near + r * (lo + SHARED), far + r * (lo + SHARED), far + r * (hi - SHARED), near + r * (hi - SHARED)])
        for heading in (u, -u):
          self.gores[node].append((inner, heading, layer_of(wid), None if zz is None else float(zz)))

  def _add_quads(self, wid: int):
    refs = self.osm.ways[wid][1]
    pts = self.osm.xy[self.osm.data.index(refs)]
    z = self._heights(refs)
    lo, hi = self.osm.lanes(wid).edges(FORWARD)
    for k in range(len(pts) - 1):
      p, q = pts[k], pts[k + 1]
      length = float(np.hypot(*(q - p)))
      if length < 0.1:
        continue
      u = (q - p) / length
      r = np.array([u[1], -u[0]])  # right of the direction of travel

      def quad(a, b, p=p, q=q, r=r):
        return np.array([p + r * a, q + r * a, q + r * b, p + r * b])
      surface, strip = quad(lo - SHARED, hi + SHARED), quad(lo - SHARED, lo + MEETS)
      self.quads.append((wid, surface.mean(0), surface, strip, u, self.layer_of(wid), None if z is None else float(z[k:k + 2].mean()),
                         quad(lo - GORE_SHARED, hi + GORE_SHARED)))
      n = len(self.quads) - 1
      lo_c, hi_c = self.quads[-1][7].min(0) // CELL, self.quads[-1][7].max(0) // CELL
      for cx in range(int(lo_c[0]), int(hi_c[0]) + 1):
        for cy in range(int(lo_c[1]), int(hi_c[1]) + 1):
          self._cells[(cx, cy)].append(n)

  def _gap(self, wid: int) -> float:
    """m of the longest stretch of a way's line on no carriageway of the ways with quads so far."""
    refs = self.osm.ways[wid][1]
    pts, covered, _, _ = self._cover(self.osm.xy[self.osm.data.index(refs)], self._heights(refs), self.layer_of(wid), {wid}, False,
                                     gores=False)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    return max((float(seg[a:b].sum()) for a, b in _runs(~covered)), default=0.0)

  def _heights(self, refs) -> np.ndarray | None:
    try:
      return np.array([float(self.osm.data.node_tags[n]['ele']) for n in refs])
    except (KeyError, ValueError):
      return None

  def kerb(self, line, z, layer: int, ways, right: bool) -> tuple[list[np.ndarray], list[tuple[np.ndarray, str]]]:
    """A kerb of the ways `ways` (joined end to end, on `layer`): a polyline [N, 2] along their direction, with heights
    [N] or None. Returns the pieces of it that are kerbs, and, for a right-hand kerb, the pieces that are the lane line
    between them and a way beside them, with its style ('dashed' / 'solid'). A lane change across other ways (crosses)
    has no lane lines of its own."""
    line = np.asarray(line, float)[:, :2]
    ways = set(ways)
    if len(line) >= 2 and ways and ways <= self.two_way:
      return self._two_way_kerb(line, z, layer, ways, right)
    if len(line) < 2 or not all(w in self.first for w in ways):  # roads not given keep their kerbs
      return ([line] if len(line) >= 2 else []), []
    if ways <= self.across or all(self._city_change(w) for w in ways):
      return [], []
    pts, covered, beside, edge = self._cover(line, z, layer, ways, right and not any(self.crosses(w) for w in ways))
    edge &= ~covered
    kept = [pts[a:b + 1] for a, b in _runs(~covered & ~edge)]
    lines = [(pts[a:b + 1], 'solid') for a, b in _runs(edge)]
    if right:
      mine = all(self.osm.lanes(w).lanes[-1].change_right for w in ways)
      for wid in sorted(set(beside[beside >= 0].tolist())):
        if self.crosses(wid):
          continue
        style = 'dashed' if mine and self.osm.lanes(wid).lanes[0].change_left else 'solid'
        lines += [(pts[a:b + 1], style) for a, b in _runs(beside == wid)]
    keep = [p for p in kept if _length(p) >= MIN_PIECE]
    return keep, [(p, s) for p, s in lines if _length(p) >= MIN_PIECE]

  def _two_way_kerb(self, line, z, layer: int, ways: set, right: bool) -> tuple[list[np.ndarray], list[tuple[np.ndarray, str]]]:
    """A two-way road's kerb (kerb's arguments): left out on a one-way way's carriageway running the way the traffic on
    that side does (a turn bay laid as a way of its own beside it), and where it meets that way's left edge the solid line
    painted between them."""
    flip = not right  # traffic on the left kerb's side runs against the way
    kerb_line = line[::-1] if flip else line
    kz = None if z is None else (np.asarray(z, float)[::-1] if flip else np.asarray(z, float))
    pts, covered, beside, _ = self._cover(kerb_line, kz, layer, ways, True, gores=False, oneway_only=True)
    _, gore, _, _ = self._cover(kerb_line, kz, layer, ways, False, gores=True, oneway_only=True, quads=False)
    if not covered.any() and not gore.any():
      return [line], []  # as given: densified, every road's kerbs would double the lines
    kept = [pts[a:b + 1] for a, b in _runs(~covered & ~gore)]  # inside a wider road it carries on from: the bay's line
    lines = [(pts[a:b + 1], 'solid') for a, b in _runs((beside >= 0) | (gore & ~covered))]
    return [p for p in kept if _length(p) >= MIN_PIECE], [(p, st) for p, st in lines if _length(p) >= MIN_PIECE]

  def _city_change(self, wid: int) -> bool:
    """Whether a way off the freeways is a lane change between carriageways (as GTA lays them to and from a turn bay):
    shorter than CHANGE_M, from a node where others meet to one where others meet, and lying mostly on other ways'
    carriageways (crosses). Its kerbs are inside the road surface."""
    tags, refs = self.osm.ways[wid]
    if Junctions.freeway(tags) or self.node_ways.get(refs[0], 0) < 3 or self.node_ways.get(refs[-1], 0) < 3:
      return False
    if _length(self.osm.xy[self.osm.data.index(refs)]) > CHANGE_M:
      return False
    return self.crosses(wid)

  def _turns(self, wid: int, first: bool, u: np.ndarray) -> float:
    """Deg between a way's heading at its first (else last) node and u."""
    p = self.osm.xy[self.osm.data.index(self.osm.ways[wid][1])]
    d = p[1] - p[0] if first else p[-1] - p[-2]
    c = float(d @ u) / max(float(np.hypot(*d)), 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

  def crosses(self, wid: int) -> bool:
    """Whether a one-way way's line lies mostly on other ways' carriageways running its way: a lane change across them,
    as GTA lays them between its freeway links, rather than a way of lanes of its own."""
    if wid not in self._crosses:
      refs = self.osm.ways[wid][1]
      _, covered, _, _ = self._cover(self.osm.xy[self.osm.data.index(refs)], self._heights(refs), self.layer_of(wid), {wid}, False,
                                     gores=False)
      self._crosses[wid] = bool(covered.mean() > 0.5) if len(covered) else False
    return self._crosses[wid]

  def _cover(self, line, z, layer: int, ways: set, right: bool, gores: bool = True, oneway_only: bool = False, quads: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                                                            np.ndarray]:
    """The line densified (points [N, 2]), whether each of its segments is on the carriageway of a way beside `ways`,
    (`right`) the way whose left edge each meets, else -1, and (on a freeway) whether it's within GORE_SHARED of one. With `gores`, inside a carriageway the run parts from or
    merges into counts too (not for a way's own line: that isn't a lane change across others)."""
    refs = [self.osm.ways[w][1] for w in ways]
    starts, ends = {r[0] for r in refs}, {r[-1] for r in refs}
    begin, finish = starts - ends, ends - starts  # the ends of the run of ways
    pts = densify(line, STEP)
    mids = (pts[:-1] + pts[1:]) / 2
    seg = np.diff(pts, axis=0)
    u = seg / np.maximum(np.hypot(*seg.T), 1e-9)[:, None]
    zm = None
    if z is not None and len(line) >= 2:
      s_line = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))))
      s_pts = np.concatenate(([0.0], np.cumsum(np.hypot(*seg.T))))
      zm = np.interp((s_pts[:-1] + s_pts[1:]) / 2, s_line, np.asarray(z, float))
    tests = defaultdict(list)  # quad -> the kerb's segments in its cells
    cells = np.floor(mids / CELL).astype(int)
    keys, of = np.unique(cells, axis=0, return_inverse=True)
    for c, (cx, cy) in enumerate(keys):
      here = np.nonzero(of.ravel() == c)[0]
      for n in self._cells.get((int(cx), int(cy)), ()):
        tests[n].append(here)
    covered = np.zeros(len(mids), bool)
    edge = np.zeros(len(mids), bool)  # near a freeway carriageway beside it: a gore's or shoulder's painted edge
    freeway = any(Junctions.freeway(self.osm.ways[w][0]) for w in ways)
    beside = np.full(len(mids), -1)  # the way whose left edge this right-hand kerb meets
    cos = np.cos(np.radians(SAME_WAY))
    # inside the carriageway this run parts from, or merges into, carried on past it
    for node in begin | finish if gores else ():
      for inner, heading, gore_layer, gore_z in self.gores.get(node, ()):
        if gore_layer != layer:
          continue
        sel = u @ heading >= cos
        if zm is not None and gore_z is not None:
          sel &= np.abs(zm - gore_z) < LEVEL
        idx = np.nonzero(sel)[0]
        if len(idx):
          covered[idx[in_fan(mids[idx], inner.mean(0), inner)]] = True
    for n, idx in tests.items() if quads else ():
      wid, centre, surface, strip, heading, quad_layer, quad_z, wide = self.quads[n]
      two_way = wid in self.two_way
      if wid in ways or quad_layer != layer or (two_way and oneway_only):
        continue
      if not two_way and (self.last[wid] in begin or self.first[wid] in finish or
                          (oneway_only and (self.first[wid] in begin | finish or self.last[wid] in begin | finish))):
        continue  # a way carrying on from or into this run isn't beside it, nor one parting from or joining a two-way road
      if not two_way and not freeway and ((self.first[wid] in begin and self._turns(wid, True, u[0]) > JOIN_TURN) or
                          (self.last[wid] in finish and self._turns(wid, False, u[-1]) > JOIN_TURN)):
        continue  # nor (off the freeways) a way turning off its start or onto its end: it crosses its kerb only there
      idx = np.concatenate(idx)
      if not two_way and (freeway or Junctions.freeway(self.osm.ways[wid][0])):
        lo, hi = wide.min(0), wide.max(0)
        near = idx[(u[idx] @ heading >= cos) & (mids[idx] >= lo).all(1) & (mids[idx] <= hi).all(1)]
        if zm is not None and quad_z is not None:
          near = near[np.abs(zm[near] - quad_z) < LEVEL]
        if len(near):
          edge[near[in_fan(mids[near], centre, wide)]] = True
      lo, hi = surface.min(0), surface.max(0)
      along = np.abs(u[idx] @ heading) if two_way else u[idx] @ heading
      idx = idx[(along >= cos) & (mids[idx] >= lo).all(1) & (mids[idx] <= hi).all(1)]
      if zm is not None and quad_z is not None:
        idx = idx[np.abs(zm[idx] - quad_z) < LEVEL]
      if not len(idx):
        continue
      on = in_fan(mids[idx], centre, surface)
      covered[idx[on]] = True
      if right and not two_way:
        meets = idx[on & in_fan(mids[idx], strip.mean(0), strip) & (u[idx] @ heading >= np.cos(np.radians(PARALLEL)))]
        beside[meets] = wid
    return pts, covered, beside, edge


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
  """[(first, last + 1)] of each run of True segments, as point indices of the pieces they make."""
  out, start = [], None
  for k, v in enumerate(np.append(mask, False)):
    if v and start is None:
      start = k
    elif not v and start is not None:
      out.append((start, k))
      start = None
  return out


def _length(p: np.ndarray) -> float:
  return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) >= 2 else 0.0
