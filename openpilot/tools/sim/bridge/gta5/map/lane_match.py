"""The car's lane from a map's lane tags alone (osm_lanes.py), whatever the game or a route says: the way it is on, its
lane there, and whether that's an oncoming lane or a one-way driven the wrong way, from its position, heading and
height. junctions.py's junction areas say where a reading means nothing: inside a junction the car is on the moves
through it, not on the ways meeting there.

Positions are the map's metres (x east, y north), headings radians counterclockwise from x."""
import hashlib
import math
import os
from typing import NamedTuple

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, FORWARD, OsmLanes, Section

ALIGN = math.cos(math.radians(40.0))  # a way runs within this of the car's heading, either way
MAX_DZ = 4.0  # m between the car and the way's height
REACH = 25.0  # m either side of a way's line it is looked for
BEYOND = 8.0  # m past a way's end
MAX_OUT = 3.0  # m outside a way's kerbs: off it
BAY_SCORE = 3.0  # how much worse a two-way way's match may be than a one-way's for the car to be in its median
CELL, PAD = 20.0, 15.0  # m: a grid of the ways, each cell with the segments within PAD of it
OWN, ONCOMING, WRONG_WAY, MEDIAN, CENTRE = 'own', 'oncoming', 'wrong-way', 'median', 'centre'
JUNCTION_DZ = 6.0  # m between the car and a junction's height: a road passing over or under it
JUNCTION_CANDIDATES = 6
CACHE_DIR = os.path.expanduser(os.getenv("GTA5_LANE_CACHE", "~/.cache/gta5_lanes"))  # junction areas, by map
AREAS_VERSION = 2  # of junction areas' cache: bump when what goes into them changes


class LaneReading(NamedTuple):
  way: int
  direction: int  # FORWARD / BACKWARD along the way
  lane: int  # Section's numbering: 0 to lanes - 1 ours from the left, negative left of them; -1 on a one-way driven against
  lanes: int
  kind: str  # OWN, ONCOMING, WRONG_WAY, MEDIAN (between the directions) or CENTRE (a centre turn lane)
  bay: bool  # a one-way driven against inside a two-way road's median: the other direction's turn bay
  right: float  # m right of the way's line
  out: float  # m outside its kerbs

  @property
  def oncoming(self) -> bool:
    return self.kind in (ONCOMING, WRONG_WAY)


def node_heights(osm: OsmLanes) -> dict[int, float]:
  """The nodes' heights from their `ele` tags."""
  out = {}
  for node, tags in osm.data.node_tags.items():
    try:
      out[node] = float(tags['ele'])
    except (KeyError, ValueError):
      pass
  return out


class LaneMatcher:
  """Finds the car's lane on a map's ways (OsmLanes), from its position, heading and, where the map has them, the
  heights of the ways' nodes (default: their `ele` tags), so a road passing over or under the car isn't it."""
  def __init__(self, osm: OsmLanes, heights: dict[int, float] | None = None):
    self.osm = osm
    heights = node_heights(osm) if heights is None else heights
    rows = []
    for wid, (_, refs) in osm.ways.items():
      pts = osm.way_points(wid)
      for i in range(len(refs) - 1):
        rows.append((wid, *pts[i], heights.get(refs[i], np.nan), *pts[i + 1], heights.get(refs[i + 1], np.nan)))
    s = np.array(rows, dtype=float).reshape(-1, 7)
    self.way = s[:, 0].astype(np.int64)
    self.a, self.az, self.b, self.bz = s[:, 1:3], s[:, 3], s[:, 4:6], s[:, 6]
    d = self.b - self.a
    self.len = np.hypot(d[:, 0], d[:, 1])
    self.u = d / np.maximum(self.len, 1e-9)[:, None]
    self.cells: dict[tuple[int, int], np.ndarray] = {}
    lo, hi = np.minimum(self.a, self.b) - PAD, np.maximum(self.a, self.b) + PAD
    cells: dict[tuple[int, int], list[int]] = {}
    for k in range(len(s)):
      for i in range(int(lo[k, 0] // CELL), int(hi[k, 0] // CELL) + 1):
        for j in range(int(lo[k, 1] // CELL), int(hi[k, 1] // CELL) + 1):
          cells.setdefault((i, j), []).append(k)
    self.cells = {k: np.array(v) for k, v in cells.items()}
    self._sections: dict[tuple[int, int], Section] = {}

  def section(self, way: int, direction: int) -> Section:
    if (way, direction) not in self._sections:
      self._sections[(way, direction)] = Section.of(self.osm.lanes(way), direction)
    return self._sections[(way, direction)]

  def _candidates(self, x: float, y: float, heading: float, z: float | None) -> list[tuple]:
    """The ways running the car's way near it, best first: [(score, way, direction, Section, m right of its line, m
    outside its kerbs, the spans the car is in, segment)]."""
    idx = self.cells.get((int(x // CELL), int(y // CELL)))
    if idx is None:
      return []
    a, u, ln = self.a[idx], self.u[idx], self.len[idx]
    rel = np.array([x, y]) - a
    t = np.einsum('ij,ij->i', rel, u)
    beyond = np.maximum(np.maximum(-t, t - ln), 0.0)
    r = rel[:, 0] * u[:, 1] - rel[:, 1] * u[:, 0]  # m right of the way's line
    if z is None:
      dz = np.zeros(len(idx))
    else:
      frac = np.clip(t / np.maximum(ln, 1e-9), 0, 1)
      zs = self.az[idx] + (self.bz[idx] - self.az[idx]) * frac
      dz = np.where(np.isnan(zs), 0.0, np.abs(zs - z))
    align = u @ np.array([math.cos(heading), math.sin(heading)])
    ok = (np.abs(align) > ALIGN) & (dz < MAX_DZ) & (np.abs(r) < REACH) & (beyond < BEYOND)
    cands = []
    for k in np.nonzero(ok)[0]:
      way = int(self.way[idx[k]])
      direction = FORWARD if align[k] > 0 else BACKWARD
      sec = self.section(way, direction)
      if not sec.spans:
        continue
      right = float(r[k] if direction == FORWARD else -r[k])
      lo, hi = sec.edges
      out = max(lo - right, right - hi, 0.0)
      inside = [i for i, s in enumerate(sec.spans) if s.left - 1e-6 <= right <= s.right + 1e-6]
      # m to the nearest lane: a turn bay laid as its own one-way way inside a two-way road's median is more likely
      # the way the car is on than the road whose median it would otherwise be in
      gap = 0.0 if inside else min(min(abs(right - s.left), abs(right - s.right)) for s in sec.spans)
      cands.append((out + gap + float(beyond[k]) + 0.2 * float(dz[k]), way, direction, sec, right, out, inside, int(idx[k])))
    return sorted(cands, key=lambda c: c[0])

  def match(self, x: float, y: float, heading: float, z: float | None = None) -> LaneReading | None:
    """The car's lane at (x, y) heading `heading` (radians from x, counterclockwise), at height z if known; None off
    the map's roads or across them."""
    cands = self._candidates(x, y, heading, z)
    if not cands:
      return None
    score, way, direction, sec, right, out, inside, _ = cands[0]
    if out > MAX_OUT:
      return None
    spans = sec.spans
    i = inside[0] if inside else min(range(len(spans)), key=lambda q: min(abs(right - spans[q].left), abs(right - spans[q].right)))
    lane = i - sec.first
    in_median = not inside and any(p.right <= right <= q.left for p, q in zip(spans, spans[1:], strict=False))
    if all(s.heading == -1 for s in spans):
      kind, lane = WRONG_WAY, -1
    elif in_median:
      kind = MEDIAN
    elif spans[i].heading == -1:
      kind = ONCOMING
    elif spans[i].heading == 0 and not all(s.heading == 0 for s in spans):
      kind = CENTRE
    else:
      kind = OWN
    bay = kind == WRONG_WAY and any(c[3].two_way and not c[6] and c[5] == 0 and c[0] - score < BAY_SCORE for c in cands[1:])
    return LaneReading(way, direction, lane, sec.lanes, kind, bay, round(right, 2), round(out, 2))

  def lane_centre(self, x: float, y: float, heading: float, lane: int, z: float | None = None) -> tuple[float, float, float] | None:
    """The middle of lane `lane` (from the left, clamped to the lanes there) of the road at (x, y) running the way
    `heading` does, level with the point: (x, y, the road's heading that way, radians); None off the roads."""
    cands = [c for c in self._candidates(x, y, heading, z) if c[3].lanes and c[5] <= MAX_OUT]
    if not cands:
      return None
    _, _, direction, sec, right, _, _, seg = cands[0]
    d = self.u[seg] * direction
    shift = sec.ours[min(max(lane, 0), sec.lanes - 1)].centre - right
    return x + d[1] * shift, y - d[0] * shift, math.atan2(d[1], d[0])


def map_hash(osm_path: str, version: int) -> str:
  """A map file's caches' key: its contents and the version of what's kept."""
  h = hashlib.sha1(str(version).encode())
  with open(osm_path, 'rb') as f:
    h.update(f.read())
  return h.hexdigest()[:16]


def keep(obj, path: str):
  """Saves obj (with a save(path)) to a cache file whole, so a reader never finds it half written."""
  os.makedirs(os.path.dirname(path), exist_ok=True)
  tmp = f"{path}.{os.getpid()}.tmp.npz"
  obj.save(tmp)
  os.replace(tmp, path)


class JunctionAreas:
  """The areas of a map's junctions (junctions.py), with their heights, to tell whether a point is in one."""
  def __init__(self, centres: np.ndarray, polygons: list[np.ndarray], heights: np.ndarray):
    self.centres = np.asarray(centres, float).reshape(-1, 2)
    self.polygons = [np.asarray(p, float) for p in polygons]
    self.heights = np.asarray(heights, float)
    self.radii = np.array([float(np.max(np.hypot(*(p - c).T))) if len(p) else 0.0 for p, c in zip(self.polygons, self.centres, strict=True)])

  @classmethod
  def from_junctions(cls, junctions, heights: dict[int, float] | None = None) -> 'JunctionAreas':
    """From a Junctions; heights (default: the nodes' `ele` tags) give each junction its nodes' mean height."""
    heights = node_heights(junctions.osm) if heights is None else heights
    js = [j for j in junctions.junctions if not j.minor]  # a driveway's across a main road: still on its lanes
    zs = [float(np.mean(h)) if (h := [heights[n] for n in j.nodes if n in heights]) else np.nan for j in js]
    return cls(np.array([j.centre for j in js]).reshape(-1, 2), [j.polygon for j in js], np.array(zs))

  def save(self, path: str):
    sizes = np.array([len(p) for p in self.polygons], np.int64)
    points = np.concatenate(self.polygons) if self.polygons else np.zeros((0, 2))
    np.savez(path, centres=self.centres, heights=self.heights, sizes=sizes, points=points)

  @classmethod
  def load(cls, path: str) -> 'JunctionAreas':
    with np.load(path) as f:
      ends = np.cumsum(f['sizes'])
      return cls(f['centres'], np.split(f['points'], ends[:-1]) if len(ends) else [], f['heights'])

  @staticmethod
  def cache_file(osm_path: str, cache_dir: str = CACHE_DIR) -> str:
    """Where a map's junction areas are kept, by the map file's contents."""
    return os.path.join(cache_dir, f"junction_areas-{map_hash(osm_path, AREAS_VERSION)}.npz")

  @classmethod
  def cached(cls, osm: OsmLanes, cache_dir: str = CACHE_DIR, build: bool = True, junctions=None) -> 'JunctionAreas | None':
    """A map's junction areas (osm read from a file) from the cache; if they aren't there yet and `build`, built (about
    half a minute on the whole GTA map, or from `junctions` where given) and kept there, else None."""
    path = cls.cache_file(osm.path, cache_dir)
    try:
      return cls.load(path)
    except (OSError, ValueError, KeyError):
      if not build:
        return None
    from openpilot.tools.sim.bridge.gta5.map.junctions import Junctions
    areas = cls.from_junctions(junctions if junctions is not None else Junctions(osm))
    keep(areas, path)
    return areas

  def inside(self, x: float, y: float, z: float | None = None) -> bool:
    from openpilot.tools.sim.bridge.gta5.map.junctions import in_fan
    if not len(self.centres):
      return False
    d = np.hypot(self.centres[:, 0] - x, self.centres[:, 1] - y)
    for k in np.argsort(d)[:JUNCTION_CANDIDATES]:
      if d[k] > self.radii[k] + 1.0:
        break
      if z is not None and not np.isnan(self.heights[k]) and abs(self.heights[k] - z) > JUNCTION_DZ:
        continue
      if in_fan([x, y], self.centres[k], self.polygons[k])[0]:
        return True
    return False


def main():
  """`lane_match.py MAP.osm.pbf`: builds a map's junction areas and stop lines (stop_lines.py) into the cache, as the
  bridge does in the background."""
  import argparse
  from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
  from openpilot.tools.sim.bridge.gta5.map.stop_lines import StopLines
  ap = argparse.ArgumentParser(description=main.__doc__)
  ap.add_argument("osm")
  ap.add_argument("--cache", default=CACHE_DIR)
  args = ap.parse_args()
  osm = OsmLanes.load(args.osm, to_game)
  areas, stops = JunctionAreas.cached(osm, args.cache, build=False), StopLines.cached(osm, args.cache, build=False)
  if areas is None or stops is None:  # the junctions worked out once for both
    from openpilot.tools.sim.bridge.gta5.map.junctions import Junctions
    junctions = Junctions(osm)
    if areas is None:
      areas = JunctionAreas.cached(osm, args.cache, junctions=junctions)
    if stops is None:
      stops = StopLines.cached(osm, args.cache, junctions=junctions)
  print(f"{len(areas.centres)} junction areas in {JunctionAreas.cache_file(args.osm, args.cache)}")
  print(f"{len(stops)} stop lines in {StopLines.cache_file(args.osm, args.cache)}")


if __name__ == "__main__":
  main()
