"""One-way ways side by side on one road surface, as GTA draws a freeway (a link per lane or two, with lane changes as
links between them) or as a turn lane mapped as its own way beside its road: a kerb is drawn only where the surface
ends. A way's kerb is left out where it lies on, or within SHARED of, the carriageway of another one-way way running the
same way on its level, other than the ways it carries on from or into. Where two such ways meet edge to edge, the way on
the left draws the line between them as a lane line: dashed, solid where change:lanes says either outer lane there may
not cross it. Only standard tags are read. Real maps don't split a carriageway without a physical separation between
its parts, so there it rarely applies.
"""
from collections import defaultdict

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.junctions import densify, in_fan
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import FORWARD, oneway_of

# m: a kerb this near another way's carriageway, or on it, is inside the road surface (lanes at their class
# width often fall short of the paint between GTA's links)
SHARED = 1.5
MEETS = 1.5  # m a kerb may be on the carriageway beside it (lanes placed to a lane's edge or middle) and still be its edge
SAME_WAY = 75.0  # deg between two ways' headings running the same way (GTA's lane changes cut across at up to ~60)
PARALLEL = 10.0  # deg: a way meeting another this near parallel shares a lane line with it, rather than crossing to it
LEVEL = 2.5  # m of height between ways on one level
STEP = 0.5  # m: kerbs are cut to this
MIN_PIECE = 0.3  # m
CELL = 25.0  # m


class SideBySide:
  """The one-way ways among `ways` (way ids of an osm_lanes.OsmLanes) and their carriageways, a quadrilateral for each
  segment; `layer_of(way)` is its layer."""
  def __init__(self, osm, ways, layer_of):
    self.osm, self.layer_of = osm, layer_of
    self.first, self.last = {}, {}
    self._crosses: dict[int, bool] = {}
    # each segment's way, middle, carriageway widened by SHARED, the strip along its left edge, unit heading, layer, height
    self.quads: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, float | None]] = []
    self._cells: dict[tuple[int, int], list[int]] = defaultdict(list)
    for wid in ways:
      tags, refs = osm.ways[wid]
      if oneway_of(tags) != 1 or len(refs) < 2:
        continue
      self.first[wid], self.last[wid] = refs[0], refs[-1]
      pts = osm.xy[osm.data.index(refs)]
      z = self._heights(refs)
      lo, hi = osm.lanes(wid).edges(FORWARD)
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
        self.quads.append((wid, surface.mean(0), surface, strip, u, layer_of(wid), None if z is None else float(z[k:k + 2].mean())))
        n = len(self.quads) - 1
        lo_c, hi_c = surface.min(0) // CELL, surface.max(0) // CELL
        for cx in range(int(lo_c[0]), int(hi_c[0]) + 1):
          for cy in range(int(lo_c[1]), int(hi_c[1]) + 1):
            self._cells[(cx, cy)].append(n)

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
    if len(line) < 2 or not all(w in self.first for w in ways):  # two-way roads keep their kerbs
      return ([line] if len(line) >= 2 else []), []
    pts, covered, beside = self._cover(line, z, layer, ways, right and not any(self.crosses(w) for w in ways))
    kept = [pts[a:b + 1] for a, b in _runs(~covered)]
    lines = []
    if right:
      mine = all(self.osm.lanes(w).lanes[-1].change_right for w in ways)
      for wid in sorted(set(beside[beside >= 0].tolist())):
        if self.crosses(wid):
          continue
        style = 'dashed' if mine and self.osm.lanes(wid).lanes[0].change_left else 'solid'
        lines += [(pts[a:b + 1], style) for a, b in _runs(beside == wid)]
    keep = [p for p in kept if _length(p) >= MIN_PIECE]
    return keep, [(p, s) for p, s in lines if _length(p) >= MIN_PIECE]

  def crosses(self, wid: int) -> bool:
    """Whether a one-way way's line lies mostly on other ways' carriageways running its way: a lane change across them,
    as GTA lays them between its freeway links, rather than a way of lanes of its own."""
    if wid not in self._crosses:
      refs = self.osm.ways[wid][1]
      _, covered, _ = self._cover(self.osm.xy[self.osm.data.index(refs)], self._heights(refs), self.layer_of(wid), {wid}, False)
      self._crosses[wid] = bool(covered.mean() > 0.5) if len(covered) else False
    return self._crosses[wid]

  def _cover(self, line, z, layer: int, ways: set, right: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The line densified (points [N, 2]), whether each of its segments is on the carriageway of a way beside `ways`,
    and (`right`) the way whose left edge each meets, else -1."""
    starts, ends = {self.first[w] for w in ways if w in self.first}, {self.last[w] for w in ways if w in self.last}
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
    beside = np.full(len(mids), -1)  # the way whose left edge this right-hand kerb meets
    cos = np.cos(np.radians(SAME_WAY))
    for n, idx in tests.items():
      wid, centre, surface, strip, heading, quad_layer, quad_z = self.quads[n]
      if wid in ways or self.last[wid] in begin or self.first[wid] in finish or quad_layer != layer:
        continue  # a way carrying on from or into this run isn't beside it
      idx = np.concatenate(idx)
      lo, hi = surface.min(0), surface.max(0)
      idx = idx[(u[idx] @ heading >= cos) & (mids[idx] >= lo).all(1) & (mids[idx] <= hi).all(1)]
      if zm is not None and quad_z is not None:
        idx = idx[np.abs(zm[idx] - quad_z) < LEVEL]
      if not len(idx):
        continue
      on = in_fan(mids[idx], centre, surface)
      covered[idx[on]] = True
      if right:
        meets = idx[on & in_fan(mids[idx], strip.mean(0), strip) & (u[idx] @ heading >= np.cos(np.radians(PARALLEL)))]
        beside[meets] = wid
    return pts, covered, beside


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
