"""Raised islands: kerbed islands standing a little above the road with road all round them (the D-shaped islands with
a lamp post on a tram crossing's centre line, slip and corner islands, median noses). GTA lays its links over many of
them where a road's links all meet at one junction node, so the map's lanes ran through them. They're mapped as
`area=yes` + `traffic_calming=island` closed ways (no highway tag: the router reads past them), drawn as kerbs, and
nav and the driver keep the car's lanes off them.

Found from roadpaint's height-layered tiles (5 cm; road, kerb, pavement or gutter at each height), along the map's
roads: each lane's centre and BODY m either side (a car's width), one-lane roads too, END_SKIP clear of each way's
ends, read every STEP m; a run of RUN_MIN m over kerb or pavement is a candidate. Most such runs are noise: concrete
roads, yards and tracks the tiles call pavement, driveways across a dropped kerb, pavements a road's lanes overlap at a
corner. A candidate is an island where
- it's real kerb: KERB_MIN kerb cells within KERB_NEAR of it, and the raised surface around it stands DZ above the road
  and gutter there (a concrete road is level and kerbless, walls, planters and decks higher);
- the raised surface it's on is closed: it doesn't reach the edge of a WINDOW m square round it (a pavement, a long
  median), its area (with what it encloses, as a grass bed inside a kerb) is AREA, and RING of the cells just outside
  it are road or gutter (the rest: no surface at that height, as a tram track's rails);
- a road other than a service road runs over it: a service road's island is mostly a pavement a driveway cuts into a
  closed piece.
Its outline is the raised surface's, traced on a PIXEL grid and simplified by SIMPLIFY; its height the road's round it.
"""
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.painted_islands import _erode, _label, _outline, _rdp, area

ROADS = ('motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'living_street', 'service',
         'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link')
STEP = 0.5  # m between the points read along a lane
BODY = 0.9  # m either side of a lane's centre read too
END_SKIP = 3.0  # m from a way's ends left unread (a corner's pavement under the lanes GTA lays to its node)
RUN_MIN = 1.5  # m of kerb or pavement in a row under a lane
KERB_MIN = 20  # kerb cells (KERB_CELL m) ...
KERB_NEAR = 3.0  # m: ... within this of the run
KERB_WINDOW, KERB_CELL = 12.0, 0.1  # m: the square read round a run for its kerb and height
DZ = (0.05, 0.4)  # m the raised surface stands above the road round it
WINDOW, PIXEL = 40.0, 0.2  # m: the square a raised surface is traced in, and its grid
AREA = (2.0, 300.0)  # m^2: smaller is a post's or a bollard's base
RING = 0.5  # of the cells just outside it road or gutter, at least
CLOSE = 3  # pixels a raised surface is grown by to join its pieces (gaps in a kerb up to 1.2 m), then shrunk back
SIMPLIFY = 0.2  # m an outline's dropped points may be off it
ZWIN = 1.5  # m from a road's height the tiles' surfaces are read within
# raised surface the tiles read where the game draws road, checked in its top-down shots: a kerbed crescent over a
# lane (with a U-turn arrow painted on it) and the pavement's corner by (22, -750)
NOT_ISLANDS = ((22.1, -750.7),)
NOT_REACH = 5.0  # m from one of those to an island's middle


@dataclass
class Raised:
  ring: np.ndarray  # [N, 3] at the road's height, its first point repeated last
  area: float  # m^2
  dz: float  # m above the road
  centre: np.ndarray  # [2]
  ways: dict[int, str] = field(default_factory=dict)  # the ways whose lanes run over it: their highway class
  centre_hit: bool = False  # a lane's centre runs over it, not only a car's width beside it


def _tiles(tiles_dir):
  sys.path.insert(0, str(pathlib.Path(__file__).parent / 'roadpaint'))
  import rp
  return rp, rp.Tiles(tiles_dir, cache=256)


def candidates(osm, tiles, rp, ways=None) -> list[tuple[float, float, float, int, bool]]:
  """Runs of RUN_MIN over kerb or pavement under the roads' lanes: (x, y, road z, way, under a lane's centre)."""
  from openpilot.tools.sim.bridge.gta5.map.osm_lanes import FORWARD, offset_line
  out = []
  for w in (osm.ways if ways is None else ways):
    tags, refs = osm.ways[w]
    if tags.get('highway') not in ROADS or tags.get('area') == 'yes' or len(refs) < 2:
      continue
    z = np.array([float(osm.data.node_tags.get(r, {}).get('ele', 'nan')) for r in refs])
    if not np.isfinite(z).any():
      continue
    pts = osm.way_points(w)
    along = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    if along[-1] < 2 * END_SKIP + RUN_MIN:
      continue
    try:
      spans = osm.lanes(w).section(FORWARD)
    except (ValueError, KeyError):
      continue
    s = np.arange(END_SKIP, along[-1] - END_SKIP, STEP)
    zs = np.interp(s, along[np.isfinite(z)], z[np.isfinite(z)])
    for sp in spans:
      for side in (-BODY, 0.0, BODY):
        line = offset_line(pts, (sp.left + sp.right) / 2 + side)
        la = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))])
        if len(line) < 2 or la[-1] < 1e-6:
          continue
        t = s * la[-1] / along[-1]  # (an offset line is longer or shorter round bends)
        q = np.column_stack([np.interp(t, la, line[:, 0]), np.interp(t, la, line[:, 1])])
        code, _ = tiles.sample(q[:, 0], q[:, 1], zs, zwin=ZWIN)
        raised = np.isin(code & 7, (rp.KERB, rp.WALK))
        k = 0
        while k < len(raised):
          if not raised[k]:
            k += 1
            continue
          e = k
          while e + 1 < len(raised) and raised[e + 1]:
            e += 1
          if (e - k + 1) * STEP >= RUN_MIN:
            i = (k + e) // 2
            out.append((float(q[i, 0]), float(q[i, 1]), float(zs[i]), w, side == 0.0))
          k = e + 1
  return out


def kerbed(tiles, rp, x: float, y: float, z: float) -> float | None:
  """How far the raised surface round a point stands above the road there, where it's real kerb (KERB_MIN kerb cells
  within KERB_NEAR, standing DZ up); else None."""
  g = (np.arange(int(KERB_WINDOW / KERB_CELL)) + 0.5) * KERB_CELL - KERB_WINDOW / 2
  gx, gy = np.meshgrid(g, g)
  code, zz = tiles.sample(x + gx.ravel(), y + gy.ravel(), z, zwin=ZWIN)
  cls = code & 7
  raised, road = np.isin(cls, (rp.KERB, rp.WALK)), np.isin(cls, (rp.ROAD, rp.GUTTER))
  if (cls[np.hypot(gx, gy).ravel() <= KERB_NEAR] == rp.KERB).sum() < KERB_MIN or not raised.any() or not road.any():
    return None
  dz = float(np.nanmedian(zz[raised]) - np.nanmedian(zz[road]))
  return dz if DZ[0] <= dz <= DZ[1] else None


def _grow(m: np.ndarray, n: int) -> np.ndarray:
  for _ in range(n):
    g = m.copy()
    g[1:] |= m[:-1]
    g[:-1] |= m[1:]
    g[:, 1:] |= m[:, :-1]
    g[:, :-1] |= m[:, 1:]
    m = g
  return m


def blob(tiles, rp, x: float, y: float, z: float):
  """The raised surface nearest a point, traced in a WINDOW square: (outline [N, 2] closed, area, share of its ring road
  or gutter, road height round it), or None where there's none near or it's open (reaches the square's edge)."""
  n = int(WINDOW / PIXEL)
  xs = x - WINDOW / 2 + (np.arange(n) + 0.5) * PIXEL
  ys = y - WINDOW / 2 + (np.arange(n) + 0.5) * PIXEL  # row 0 south
  gx, gy = np.meshgrid(xs, ys)
  code, zz = tiles.sample(gx.ravel(), gy.ravel(), z, zwin=ZWIN)
  cls, zz = (code & 7).reshape(n, n), zz.reshape(n, n)
  raised = np.isin(cls, (rp.KERB, rp.WALK))
  c, r = n // 2, int(round(1.0 / PIXEL))
  near = np.argwhere(raised[c - r:c + r + 1, c - r:c + r + 1])
  if not len(near):
    return None
  r0, c0 = near[np.argmin(np.hypot(*(near - r).T))] + c - r
  lab = _label(_grow(raised, CLOSE))  # a thin kerb round a bed reads in pieces
  comp = lab == lab[r0, c0]
  if comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any():
    return None
  holes = _label(~comp)  # a kerb round a grass bed: the bed is the island too
  border = np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]]))
  comp = _erode(comp | ((holes >= 0) & ~np.isin(holes, border)), CLOSE)  # shrunk back to the kerb's outer edge
  grown = comp.copy()
  grown[1:] |= comp[:-1]
  grown[:-1] |= comp[1:]
  grown[:, 1:] |= comp[:, :-1]
  grown[:, :-1] |= comp[:, 1:]
  ring = grown & ~comp
  road = np.isin(cls[ring], (rp.ROAD, rp.GUTTER))
  edge = _outline(comp)
  xy = np.column_stack([xs[0] - PIXEL / 2 + edge[:, 0] * PIXEL, ys[0] - PIXEL / 2 + edge[:, 1] * PIXEL])
  xy = _rdp(np.vstack([xy, xy[:1]]), SIMPLIFY)
  zroad = float(np.nanmedian(zz[ring][road])) if road.any() else z
  return xy, float(comp.sum() * PIXEL ** 2), float(road.mean()), zroad


def islands(osm, tiles_dir, ways=None, stats=None) -> list[Raised]:
  """The raised islands under the map's roads' lanes (OsmLanes). `stats`, a dict, takes how many runs there were and
  how many places (1 m squares, a run's middle) each test dropped."""
  from openpilot.tools.sim.bridge.gta5.map.stop_paint import inside
  rp, tiles = _tiles(tiles_dir)
  stats = {} if stats is None else stats
  cands = candidates(osm, tiles, rp, ways)
  stats['runs'] = len(cands)
  found: list[Raised] = []
  rejected: dict[tuple[int, int], str] = {}  # (by a 1 m grid) what a run there was dropped for

  def drop(key, why):
    rejected[key] = why
    stats[why] = stats.get(why, 0) + 1
  for x, y, z, w, centre in cands:
    hit = next((f for f in found if (f.ring[:-1, :2].min(0) - 0.5 <= (x, y)).all() and ((x, y) <= f.ring[:-1, :2].max(0) + 0.5).all()
                and inside(np.array([[x, y]]), f.ring[:, :2])[0]), None)
    if hit is None:
      hit = next((f for f in found if np.hypot(*(np.array([x, y]) - f.centre)) < 15.0 and
                  np.hypot(*(f.ring[:-1, :2] - (x, y)).T).min() < 0.5), None)
    if hit is not None:
      hit.ways[w] = osm.ways[w][0]['highway']
      hit.centre_hit |= centre
      continue
    key = (int(np.floor(x)), int(np.floor(y)))
    if key in rejected:
      continue
    dz = kerbed(tiles, rp, x, y, z)
    if dz is None:
      drop(key, 'no kerb or level')
      continue
    b = blob(tiles, rp, x, y, z)
    if b is None:
      drop(key, 'open')
      continue
    xy, size, road, zroad = b
    if not AREA[0] <= size <= AREA[1]:
      drop(key, 'area')
      continue
    if road < RING:
      drop(key, 'not ringed by road')
      continue
    same = next((f for f in found if np.hypot(*(xy[:-1].mean(0) - f.centre)) < 1.0), None)  # traced again from beside it
    if same is not None:
      same.ways[w] = osm.ways[w][0]['highway']
      same.centre_hit |= centre
      continue
    ring = np.column_stack([xy, np.full(len(xy), zroad)])
    found.append(Raised(ring, round(size, 1), round(dz, 3), xy[:-1].mean(0), {w: osm.ways[w][0]['highway']}, centre))
  out = [f for f in found if any(c != 'service' for c in f.ways.values())]
  stats['service roads only'] = len(found) - len(out)
  kept = [f for f in out if all(np.hypot(*(f.centre - q)) > NOT_REACH for q in NOT_ISLANDS)]
  stats['road the game draws'] = len(out) - len(kept)
  return kept


def areas(found: list[Raised]) -> list[tuple[np.ndarray, dict]]:
  """The islands as painted_islands.add's extra areas: (ring [N, 3] unclosed, tags)."""
  return [(f.ring[:-1], {'area': 'yes', 'traffic_calming': 'island'}) for f in found]


def main():
  import json
  from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
  from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes
  pbf, tiles_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
  osm = OsmLanes.load(pbf, to_game)
  stats: dict = {}
  found = islands(osm, tiles_dir, stats=stats)
  rows = [{'centre': [round(float(v), 1) for v in f.centre], 'area': f.area, 'dz': f.dz, 'z': round(float(f.ring[0, 2]), 2),
           'centre_hit': f.centre_hit, 'ways': {str(w): c for w, c in sorted(f.ways.items())},
           'names': sorted({osm.ways[w][0].get('name') for w in f.ways} - {None}),
           'no_nav': any(osm.ways[w][0].get('gta:no_nav') == 'yes' for w in f.ways),
           'ring': np.round(f.ring[:, :2], 2).tolist(), 'ring_area': round(area(f.ring[:-1, :2]), 1)} for f in found]
  with open(out, 'w') as fh:
    json.dump({'stats': stats, 'islands': rows}, fh, indent=1)
  print(len(found), 'raised islands;', stats)


if __name__ == '__main__':
  main()
