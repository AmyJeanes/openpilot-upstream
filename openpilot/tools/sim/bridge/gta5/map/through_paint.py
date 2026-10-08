"""Priority roads where the game files paint a road's lines on across a junction (polylines.jsonl): a road carried
straight on through a junction (junctions.py's `Junction.through`) whose lines the game paints across the junction's
area, as a main road's centre line runs on past a side road, gets `priority_road=yes_unposted` on its ways either side,
so junctions.py carries its lines across (`Junction.carried`). Where the paint stops at the junction's mouth, or the
game paints none there, the road's lines end at the junction as before.

- A line is painted across where paint of either colour runs within NEAR of it (at its height, DZ) along at least
  COVER of its length inside the area: the colour is the road's own business, and where the map's colour is wrong the
  line still runs on.
- GTA's links are ways of their own, so the tag stays on the stretch next to the junction.
"""
import json
from collections import defaultdict

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.junctions import Junctions, densify, in_fan
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes, offset_line
from openpilot.tools.sim.bridge.gta5.map.osm_to_roads import drawn

NEAR = 0.6  # m from a line to its paint
DZ = 2.0  # m between the paint's height and the junction's
COVER = 0.5  # of a road's lines inside the junction's area painted, at least
MIN_INSIDE = 2.0  # m of line inside the area at least, to judge
STEP = 0.5  # m between the points a line is checked at
CELL = 2.0  # m
TAG = {'priority_road': 'yes_unposted'}


class Paint:
  """The game files' painted lines, as points every STEP (dashed lines end to end), in CELL squares."""
  def __init__(self, path: str):
    self.cells: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    with open(path) as f:
      for row in f:
        q = np.array(json.loads(row)['pts'], float)
        if len(q) < 2:
          continue
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(q[:, :2], axis=0).T))]
        t = np.linspace(0.0, s[-1], max(int(s[-1] / STEP) + 2, 2))
        pts = np.stack([np.interp(t, s, q[:, k]) for k in range(3)], axis=1)
        for key, part in self._split(pts):
          self.cells[key].append(part)
    self.cells = {k: np.vstack(v) for k, v in self.cells.items()}

  @staticmethod
  def _split(pts: np.ndarray):
    keys = np.floor(pts[:, :2] / CELL).astype(int)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    keys, pts = keys[order], pts[order]
    edges = np.flatnonzero(np.any(np.diff(keys, axis=0) != 0, axis=1)) + 1
    for a, b in zip(np.r_[0, edges], np.r_[edges, len(pts)], strict=True):
      yield (int(keys[a, 0]), int(keys[a, 1])), pts[a:b]

  def near(self, p: np.ndarray, z: float) -> bool:
    cx, cy = int(p[0] // CELL), int(p[1] // CELL)
    for dx in (-1, 0, 1):
      for dy in (-1, 0, 1):
        q = self.cells.get((cx + dx, cy + dy))
        if q is not None and ((np.hypot(q[:, 0] - p[0], q[:, 1] - p[1]) < NEAR) & (np.abs(q[:, 2] - z) < DZ)).any():
          return True
    return False


def painted_across(osm: OsmLanes, junctions: Junctions, paint: Paint) -> tuple[set[int], dict]:
  """The ways of the roads carried through junctions whose lines the game paints across them, and counts."""
  ways, counts = set(), defaultdict(int)
  for j in junctions.junctions:
    if not j.through:
      continue
    z = float(np.mean([float(osm.data.node_tags.get(n, {}).get('ele', 'nan')) for n in j.nodes]))
    inside = painted = 0
    for w, offsets in j.through.items():
      pts = osm.way_points(w)
      for off in offsets:
        line = densify(offset_line(pts, off), STEP)
        line = line[in_fan(line, j.centre, j.polygon)]
        inside += len(line)
        painted += sum(paint.near(p, z) for p in line)
    if inside * STEP < MIN_INSIDE or np.isnan(z):
      counts['too little inside'] += 1
    elif painted >= COVER * inside:
      counts['painted across'] += 1
      ways |= set(j.through)
    else:
      counts['not painted across'] += 1
  return ways, dict(counts)


def retag(path: str, ways: set[int], tags: dict):
  """Writes the map at `path` again with `tags` added to these ways."""
  import osmium
  nodes, rows, rels = [], [], []
  for o in osmium.FileProcessor(path):
    if o.is_node():
      nodes.append((o.id, (o.location.lon, o.location.lat), dict(o.tags)))
    elif o.is_way():
      rows.append((o.id, [n.ref for n in o.nodes], {**dict(o.tags), **(tags if o.id in ways else {})}))
    elif o.is_relation():
      rels.append((o.id, dict(o.tags), [(m.type, m.ref, m.role) for m in o.members]))
  w = osmium.SimpleWriter(path, overwrite=True)
  for nid, loc, t in nodes:
    w.add_node(osmium.osm.mutable.Node(id=nid, version=1, location=loc, tags=t))
  for wid, refs, t in rows:
    w.add_way(osmium.osm.mutable.Way(id=wid, version=1, nodes=refs, tags=t))
  for rid, t, members in rels:
    w.add_relation(osmium.osm.mutable.Relation(id=rid, version=1, tags=t, members=members))
  w.close()


def priority_roads(path: str, lines_path: str, project) -> dict:
  """Tags the map at `path` in place (painted_across); returns the counts."""
  osm = OsmLanes.load(path, project)
  ways, counts = painted_across(osm, Junctions(osm, drawn), Paint(lines_path))
  retag(path, ways, TAG)
  counts['ways tagged'] = len(ways)
  return counts
