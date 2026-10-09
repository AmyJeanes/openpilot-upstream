"""Painted islands: flat painted areas on the road surface with road all round them (a painted triangle where a side
road's two directions part, a chevron-hatched gore or median between GTA's carriageways). GTA lays its links round them
as round a raised island, so the map's kerbs went round and through them; they're mapped as `area=yes` +
`traffic_calming=painted_island` closed ways with the `colour` of the paint round them, kerbs aren't drawn within
junctions.ISLAND_REACH of one (junctions.off_islands), and the map view and the overlay draw its outline in that colour.

- An outline is a white or yellow solid polyline closed on itself, or up to three of a colour meeting end to end into
  a ring (JOIN), of AREA m^2.
- It's painted, not raised, where roadpaint's height-layered tiles (5 cm; road, kerb, pavement or gutter at each
  height) read it as road inside (INSIDE of it) and in a band BAND m outside its outline (AROUND of that), at its
  height: a raised island is pavement or kerb, and an outline at the road's edge has kerb or pavement beside it.
- A flush edge (`flush_strips`): a way's edge facing another carriageway within GAP_BEYOND with road beyond it
  (FLUSH_OUT), as where GTA's links run side by side round paint or a slip parts, the class layout's edges inside
  the asphalt: a strip of `area:highway=<its class>` (road surface) along each run of FLUSH_RUN at least, which no
  kerb is drawn in either.
- A painted gap (`gaps`) is road surface the tiles read between carriageways that no way's lanes cover (but GTA's lane
  changes, which cut across the paint as its AI drives), enclosed: the road all round it, no kerb, pavement or edge
  beside it (EDGE_NEIGHBOURS), of GAP_AREA and wider than GAP_WIDTH somewhere. It's looked for only where a one-way
  way's edge has another carriageway within GAP_BEYOND (the tiles are 10 GB), on a PIXEL grid; its outline is traced
  round the pixels and simplified. Its colour is the paint's the tiles read along its edge (OUTLINE_PAINT of it at
  least, the more of white and yellow), else it has none (paint stripes or chevrons only, drawn as no line).
"""
import json
import pathlib
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

JOIN = 0.5  # m between polylines' ends that meet
AREA = (8.0, 600.0)  # m^2: smaller rings are boxes and symbols
INSIDE = 0.9  # of the outline's inside read as road at least
AROUND = 0.85  # of the band outside it read as road at least
BAND = (0.5, 1.5)  # m out from the outline
STEP = 0.5  # m between the points sampled
OUTLINE_PAINT = 0.15  # of a gap's edge pixels painted at least for it to be outlined in that paint


@dataclass
class Island:
  ring: np.ndarray  # [N, 3], its first point repeated last
  colour: str | None  # its outline's paint, 'white' or 'yellow'; None unpainted


def outlines(path) -> list[Island]:
  """The white or yellow solid rings among the game files' polylines (polylines.jsonl)."""
  rows, colours = [], []
  with open(path) as f:
    for row in f:
      p = json.loads(row)
      if p['colour'] in ('white', 'yellow') and p['style'] != 'dashed' and p['len'] < 150.0:
        rows.append(np.array(p['pts'], float))
        colours.append(p['colour'])
  ends = defaultdict(list)

  def key(q):
    return round(float(q[0]) / JOIN), round(float(q[1]) / JOIN)
  for n, pts in enumerate(rows):
    for e in (0, -1):
      ends[key(pts[e])].append((n, e))

  def meeting(q, used):
    k = key(q)
    return [(m, e) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for m, e in ends.get((k[0] + dx, k[1] + dy), ())
            if m not in used and colours[m] == colours[used[0]] and np.hypot(*(rows[m][e, :2] - q[:2])) < JOIN]

  out, seen = [], set()
  for n, pts in enumerate(rows):
    if n in seen:
      continue
    chain, used = [pts], [n]
    while np.hypot(*(chain[-1][-1, :2] - pts[0, :2])) >= JOIN and len(used) < 3:
      nxt = meeting(chain[-1][-1], used)
      if len(nxt) != 1:
        break
      m, e = nxt[0]
      chain.append(rows[m] if e == 0 else rows[m][::-1])
      used.append(m)
    ring = np.vstack(chain)
    if np.hypot(*(ring[-1, :2] - ring[0, :2])) < JOIN and AREA[0] <= area(ring) <= AREA[1]:
      out.append(Island(np.vstack([ring, ring[:1]]), colours[n]))
      seen.update(used)
  return out


def area(ring: np.ndarray) -> float:
  x, y = ring[:, 0], ring[:, 1]
  return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _band(ring: np.ndarray) -> np.ndarray:
  pts = []
  for a, b in zip(ring[:-1, :2], ring[1:, :2], strict=True):
    size = float(np.hypot(*(b - a)))
    if size < 1e-6:
      continue
    n = np.array([(b - a)[1], -(b - a)[0]]) / size
    for f in np.linspace(0.0, 1.0, max(2, int(size / STEP) + 1)):
      for d in np.arange(BAND[0], BAND[1] + 1e-6, STEP):
        pts += [a + (b - a) * f + n * d, a + (b - a) * f - n * d]
  return np.array(pts)


def painted(found: list[Island], tiles_dir) -> list[Island]:
  """The outlines that are flat paint on the road (roadpaint's tiles, `rp.Tiles`, read at each one's height)."""
  from openpilot.tools.sim.bridge.gta5.map.stop_paint import inside
  sys.path.insert(0, str(pathlib.Path(__file__).parent / 'roadpaint'))
  import rp
  tiles = rp.Tiles(tiles_dir, cache=16)
  out = []
  for island in found:
    ring = island.ring
    z = float(ring[:, 2].mean())
    lo, hi = ring[:, :2].min(0), ring[:, :2].max(0)
    grid = np.stack(np.meshgrid(np.arange(lo[0], hi[0], STEP), np.arange(lo[1], hi[1], STEP)), -1).reshape(-1, 2)
    grid = grid[inside(grid, ring[:, :2])]
    band = _band(ring)
    band = band[~inside(band, ring[:, :2])]
    if len(grid) < 4 or len(band) < 4:
      continue
    code_in, _ = tiles.sample(grid[:, 0], grid[:, 1], z)
    code_out, _ = tiles.sample(band[:, 0], band[:, 1], z)
    if ((code_in & 7) == rp.ROAD).mean() >= INSIDE and ((code_out & 7) == rp.ROAD).mean() >= AROUND:
      out.append(island)
  return out


def islands(lines_path, tiles_dir, osm=None) -> list[Island]:
  """The painted islands: outlines on flat road, and (with the map, osm_lanes.OsmLanes) the painted gaps between its
  carriageways (gaps) but those inside an outline already."""
  from openpilot.tools.sim.bridge.gta5.map.stop_paint import inside
  found = painted(outlines(lines_path), tiles_dir)
  for g in gaps(osm, tiles_dir) if osm is not None else []:
    c = g.ring[:-1, :2].mean(0)
    if not any(inside(c[None], f.ring[:, :2])[0] for f in found):
      found.append(g)
  return found


def add(path: str, found: list[Island], new_node_id, to_lat_lon, surfaces=()) -> int:
  """Writes the map at `path` again with each island, and each road surface strip [(outline [N, 2] at z, class, z)]
  (flush_strips), as a closed way of new nodes; returns how many."""
  import osmium
  nodes, ways, rels = [], [], []
  for o in osmium.FileProcessor(path):
    if o.is_node():
      nodes.append(osmium.osm.mutable.Node(id=o.id, version=1, location=(o.location.lon, o.location.lat), tags=dict(o.tags)))
    elif o.is_way():
      ways.append(osmium.osm.mutable.Way(id=o.id, version=1, nodes=[n.ref for n in o.nodes], tags=dict(o.tags)))
    elif o.is_relation():
      rels.append(osmium.osm.mutable.Relation(id=o.id, version=1, tags=dict(o.tags),
                                              members=[(m.type, m.ref, m.role) for m in o.members]))
  next_way = max(w.id for w in ways) + 1
  areas = [(island.ring[:-1], {'area': 'yes', 'traffic_calming': 'painted_island',
                                **({'colour': island.colour} if island.colour else {})}) for island in found]
  areas += [(np.column_stack([xy, np.full(len(xy), z)]), {'area:highway': cls}) for xy, cls, z in surfaces]
  for ring, tags in areas:
    refs = []
    for x, y, z in ring:
      nid = new_node_id()
      lat, lon = to_lat_lon(float(x), float(y))
      nodes.append(osmium.osm.mutable.Node(id=nid, version=1, location=(lon, lat), tags={'ele': f'{z:.1f}'}))
      refs.append(nid)
    ways.append(osmium.osm.mutable.Way(id=next_way, version=1, nodes=[*refs, refs[0]], tags=tags))
    next_way += 1
  w = osmium.SimpleWriter(path, overwrite=True)
  for n in sorted(nodes, key=lambda n: n.id):
    w.add_node(n)
  for way in sorted(ways, key=lambda way: way.id):
    w.add_way(way)
  for r in rels:
    w.add_relation(r)
  w.close()
  return len(areas)


# *** painted gaps between carriageways ***

GAP_BEYOND = (1.0, 8.0)  # m out from a one-way way's edge another carriageway is looked for: a gap between them
PIXEL = 0.25  # m: the gaps are found on a grid of this
WINDOW = 100.0  # m: the gaps are looked for in squares this big, overlapping by WINDOW_PAD
WINDOW_PAD = 15.0  # m
GAP_AREA = 6.0  # m^2 at least
GAP_WIDTH = 0.5  # m: a gap narrower all along is a gap between quads, not paint
EDGE_NEIGHBOURS = 0.03  # of a gap's neighbouring pixels off the road (kerb, pavement, none) at most: it's enclosed
SIMPLIFY = 0.2  # m an outline's dropped points may be off it
QCELL = 25.0  # m: the quads' index cells


class Quads:
  """Ways' carriageways as quadrilaterals [4, 2] segment by segment, between their edges (OsmLanes.lanes.edges),
  indexed by QCELL squares for testing many points at once."""
  def __init__(self, osm, ways):
    from openpilot.tools.sim.bridge.gta5.map.osm_lanes import FORWARD
    quads, owner = [], []
    for w in ways:
      pts = osm.way_points(w)
      lo, hi = osm.lanes(w).edges(FORWARD)
      for a, b in zip(pts[:-1], pts[1:], strict=True):
        size = float(np.hypot(*(b - a)))
        if size < 0.05:
          continue
        r = np.array([(b - a)[1], -(b - a)[0]]) / size  # right of its direction
        quads.append([a + r * lo, b + r * lo, b + r * hi, a + r * hi])
        owner.append(w)
    self.q = np.array(quads, float).reshape(-1, 4, 2)
    self.owner = np.array(owner, np.int64)
    self.cells = defaultdict(list)
    for k, q in enumerate(self.q):
      lo, hi = q.min(0) // QCELL, q.max(0) // QCELL
      for cx in range(int(lo[0]), int(hi[0]) + 1):
        for cy in range(int(lo[1]), int(hi[1]) + 1):
          self.cells[(cx, cy)].append(k)

  def inside(self, pts: np.ndarray, but: np.ndarray | None = None) -> np.ndarray:
    """Which points [P, 2] lie in a quad (of a way other than but[P], where given)."""
    out = np.zeros(len(pts), bool)
    keys = np.floor(pts / QCELL).astype(np.int64)
    order = np.lexsort((keys[:, 1], keys[:, 0]))
    ks = keys[order]
    starts = np.flatnonzero(np.r_[True, (np.diff(ks, axis=0) != 0).any(1)])
    for s, e in zip(starts, np.r_[starts[1:], len(ks)], strict=True):
      idx = self.cells.get((int(ks[s, 0]), int(ks[s, 1])))
      if not idx:
        continue
      sel = order[s:e]
      p = pts[sel]
      q = self.q[idx]  # [Q, 4, 2]
      a, b = q, np.roll(q, -1, axis=1)
      side = (b[None, :, :, 0] - a[None, :, :, 0]) * (p[:, None, None, 1] - a[None, :, :, 1]) - \
             (b[None, :, :, 1] - a[None, :, :, 1]) * (p[:, None, None, 0] - a[None, :, :, 0])  # [P, Q, 4]
      hit = (side >= 0).all(2) | (side <= 0).all(2)
      if but is not None:
        hit &= self.owner[idx][None] != but[sel][:, None]
      out[sel] = hit.any(1)
    return out


def gap_places(osm, quads: Quads) -> np.ndarray:
  """Points [K, 2] just off one-way ways' edges with another carriageway GAP_BEYOND out and nothing between."""
  from openpilot.tools.sim.bridge.gta5.map.osm_lanes import FORWARD, oneway_of
  starts, normals, ways = [], [], []
  for w, (tags, refs) in osm.ways.items():
    if not oneway_of(tags) or len(refs) < 2:
      continue
    pts = osm.way_points(w)
    lo, hi = osm.lanes(w).edges(FORWARD)
    for a, b in zip(pts[:-1], pts[1:], strict=True):
      size = float(np.hypot(*(b - a)))
      if size < 1.0:
        continue
      r = np.array([(b - a)[1], -(b - a)[0]]) / size
      for f in np.arange(0.5, size, 2.0) / size:
        p = a + (b - a) * f
        for edge, sgn in ((hi, 1.0), (lo, -1.0)):
          starts.append(p + r * edge)
          normals.append(r * sgn)
          ways.append(w)
  if not starts:
    return np.zeros((0, 2))
  starts, normals, ways = np.array(starts), np.array(normals), np.array(ways, np.int64)
  dists = np.array([GAP_BEYOND[0], *np.arange(2.0, GAP_BEYOND[1] + 0.1, 1.0)])
  probes = starts[:, None] + normals[:, None] * dists[None, :, None]  # [K, D, 2]
  hit = quads.inside(probes.reshape(-1, 2), np.repeat(ways, len(dists))).reshape(len(starts), len(dists))
  keep = ~hit[:, 0] & hit[:, 1:].any(1)
  return probes[keep, 0]


def _label(mask: np.ndarray) -> np.ndarray:
  """4-connected components of a boolean grid: a label per pixel (-1 off the mask)."""
  h, w = mask.shape
  idx = np.arange(mask.size).reshape(mask.shape)
  pairs = [np.stack([idx[:, :-1][mask[:, :-1] & mask[:, 1:]], idx[:, 1:][mask[:, :-1] & mask[:, 1:]]], 1),
           np.stack([idx[:-1][mask[:-1] & mask[1:]], idx[1:][mask[:-1] & mask[1:]]], 1)]
  e = np.concatenate(pairs) if any(len(p) for p in pairs) else np.zeros((0, 2), np.int64)
  parent = np.arange(mask.size)
  while True:
    pa, pb = parent[e[:, 0]], parent[e[:, 1]]
    m = np.minimum(pa, pb)
    before = parent.copy()
    np.minimum.at(parent, pa, m)
    np.minimum.at(parent, pb, m)
    parent = parent[parent]
    if (parent == before).all():
      break
  return np.where(mask, parent.reshape(mask.shape), -1)


def _outline(comp: np.ndarray) -> np.ndarray:
  """The outer boundary of a component [rows, cols] (Moore neighbour tracing), as pixel centres' (col, row)."""
  rows, cols = comp.shape
  p = np.argwhere(comp)
  start = tuple(p[np.lexsort((p[:, 1], p[:, 0]))][0])
  dirs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
  path, cur, back = [start], start, 6  # came from the left

  def on(q):
    return 0 <= q[0] < rows and 0 <= q[1] < cols and comp[q]
  for _ in range(4 * int(comp.sum()) + 8):
    for k in range(8):
      d = (back + 1 + k) % 8
      q = (cur[0] + dirs[d][0], cur[1] + dirs[d][1])
      if on(q):
        back = (d + 4) % 8
        cur = q
        break
    else:
      break
    if cur == start:
      break
    path.append(cur)
  return np.array([(c + 0.5, r + 0.5) for r, c in path], float)


def _rdp(pts: np.ndarray, tol: float) -> np.ndarray:
  if len(pts) < 3:
    return pts
  a, b = pts[0], pts[-1]
  d = b - a
  n = float(np.hypot(*d))
  off = np.abs((pts[:, 0] - a[0]) * d[1] - (pts[:, 1] - a[1]) * d[0]) / n if n > 1e-9 else np.hypot(*(pts - a).T)
  k = int(np.argmax(off))
  if off[k] <= tol:
    return np.array([a, b])
  return np.vstack([_rdp(pts[:k + 1], tol)[:-1], _rdp(pts[k:], tol)])


def _erode(m: np.ndarray, n: int) -> np.ndarray:
  for _ in range(n):
    c = m.copy()
    c[1:] &= m[:-1]
    c[:-1] &= m[1:]
    c[:, 1:] &= m[:, :-1]
    c[:, :-1] &= m[:, 1:]
    c[0], c[-1], c[:, 0], c[:, -1] = False, False, False, False
    m = c
  return m


def gaps(osm, tiles_dir, ways=None) -> list[Island]:
  """The road surface roadpaint's tiles read between carriageways, enclosed by them and the road (no kerb, pavement
  or edge beside it), that no way's lanes cover: flat painted gores and medians GTA lays its links round."""
  from openpilot.tools.sim.bridge.gta5.map.junctions import lane_changes
  sys.path.insert(0, str(pathlib.Path(__file__).parent / 'roadpaint'))
  import rp
  tiles = rp.Tiles(tiles_dir, cache=32)
  ways = list(osm.ways) if ways is None else ways
  changes = lane_changes(osm, ways)  # GTA's lane changes cross the paint, as its AI drives: they cover none of it
  quads = Quads(osm, [w for w in ways if w not in changes])
  places = gap_places(osm, quads)
  if not len(places):
    return []
  ele_xy, ele_z = [], []
  for w in ways:
    for n in osm.ways[w][1]:
      t = osm.data.node_tags.get(n, {})
      if 'ele' in t:
        ele_xy.append(osm.node_xy(n))
        ele_z.append(float(t['ele']))
  ele_xy, ele_z = np.array(ele_xy), np.array(ele_z)
  ecells = defaultdict(list)
  for k, q in enumerate(ele_xy):
    ecells[(int(q[0] // QCELL), int(q[1] // QCELL))].append(k)
  out, seen = [], []
  for key in sorted({(int(x // WINDOW), int(y // WINDOW)) for x, y in places}):
    lo = np.array(key, float) * WINDOW - WINDOW_PAD
    hi = lo + WINDOW + 2 * WINDOW_PAD
    xs, ys = np.arange(lo[0], hi[0], PIXEL) + PIXEL / 2, np.arange(lo[1], hi[1], PIXEL) + PIXEL / 2
    gx, gy = np.meshgrid(xs, ys)
    pix = np.column_stack([gx.ravel(), gy.ravel()])
    covered = quads.inside(pix).reshape(gx.shape)
    near = [k for cx in range(int(lo[0] // QCELL) - 1, int(hi[0] // QCELL) + 2)
            for cy in range(int(lo[1] // QCELL) - 1, int(hi[1] // QCELL) + 2) for k in ecells.get((cx, cy), ())]
    if not near:
      continue
    # each pixel's road height: the nearest node's, on a coarse grid
    coarse = np.column_stack([gx[::8, ::8].ravel(), gy[::8, ::8].ravel()])
    nz = ele_z[near][np.argmin(np.hypot(*(coarse[:, None] - ele_xy[near][None]).transpose(2, 0, 1)), axis=1)]
    zref = np.repeat(np.repeat(nz.reshape(gx[::8, ::8].shape), 8, 0), 8, 1)[:gx.shape[0], :gx.shape[1]].ravel()
    code, z = tiles.sample(pix[:, 0], pix[:, 1], zref)
    code = code.reshape(gx.shape)
    road = (code & 7) == rp.ROAD
    lab = _label(road & ~covered)
    ids, counts = np.unique(lab[lab >= 0], return_counts=True)
    for k in ids[counts * PIXEL ** 2 >= GAP_AREA]:
      comp = lab == k
      if comp[0].any() or comp[-1].any() or comp[:, 0].any() or comp[:, -1].any():
        continue
      r, c = np.nonzero(comp)
      r0, r1, c0, c1 = max(r.min() - 2, 0), r.max() + 3, max(c.min() - 2, 0), c.max() + 3
      sub = comp[r0:r1, c0:c1]
      grown = sub.copy()
      grown[1:] |= sub[:-1]
      grown[:-1] |= sub[1:]
      grown[:, 1:] |= sub[:, :-1]
      grown[:, :-1] |= sub[:, 1:]
      ring_px = grown & ~sub
      if (ring_px & ~covered[r0:r1, c0:c1] & ~road[r0:r1, c0:c1]).sum() > EDGE_NEIGHBOURS * ring_px.sum():
        continue
      if not _erode(sub, int(round(GAP_WIDTH / 2 / PIXEL))).any():
        continue
      edge = _outline(sub)
      xy = np.column_stack([lo[0] + (c0 + edge[:, 0]) * PIXEL, lo[1] + (r0 + edge[:, 1]) * PIXEL])
      centre = xy.mean(0)
      if any(np.hypot(*(centre - s)) < 1.0 for s in seen):  # (found again in the window beside)
        continue
      xy = _rdp(np.vstack([xy, xy[:1]]), SIMPLIFY)[:-1]
      if len(xy) < 3:
        continue
      seen.append(centre)
      zz = float(np.nanmean(z.reshape(gx.shape)[comp]))
      ring = np.column_stack([xy, np.full(len(xy), zz)])
      paint = (code[r0:r1, c0:c1] >> 3)[sub & ~_erode(sub, 2)]  # its edge: within 0.5 m of its outline
      colour = max((1, 2), key=lambda p: int((paint == p).sum()))
      painted_edge = len(paint) and (paint == colour).mean() >= OUTLINE_PAINT
      out.append(Island(np.vstack([ring, ring[:1]]), {1: 'white', 2: 'yellow'}[colour] if painted_edge else None))
  return out


# *** flush edges: a way's edge with road surface on beyond it ***

FLUSH_OUT = (0.6, 1.4)  # m beyond an edge the tiles are read: road at both, and it's no kerb
FLUSH_STEP = 0.5  # m between the points read along an edge
FLUSH_RUN = 2.0  # m of edge flush at least
FLUSH_STRIP = (0.4, 0.9)  # m inside and beyond the edge the strip marking it as road surface covers


def flush_strips(osm, tiles_dir, ways=None) -> list[tuple[np.ndarray, str, float]]:
  """Where a way's edge has road on beyond it (roadpaint's tiles) and faces another carriageway within GAP_BEYOND
  (GTA's links side by side, slips and turn lanes parting, where the class layout's edges fall inside the asphalt):
  strips [N, 2] along those runs, with the way's class, as road surface (area:highway) no kerb is drawn in. Only
  there: an edge with no carriageway beyond is the road's own edge, kept wherever the layout puts it."""
  from openpilot.tools.sim.bridge.gta5.map.junctions import lane_changes
  from openpilot.tools.sim.bridge.gta5.map.osm_lanes import FORWARD
  sys.path.insert(0, str(pathlib.Path(__file__).parent / 'roadpaint'))
  import rp
  tiles = rp.Tiles(tiles_dir, cache=64)
  ways = list(osm.ways) if ways is None else ways
  changes = lane_changes(osm, ways)
  quads = Quads(osm, [w for w in ways if w not in changes])
  ele = {n: float(t['ele']) for n, t in osm.data.node_tags.items() if 'ele' in t}
  runs = []  # per way side: (way, points on the edge [K, 2], out [K, 2], z [K])
  for w in ways:
    if w in changes:
      continue
    refs = osm.ways[w][1]
    pts = osm.way_points(w)
    lo, hi = osm.lanes(w).edges(FORWARD)
    seg = np.hypot(*np.diff(pts, axis=0).T)
    along = np.concatenate(([0.0], np.cumsum(seg)))
    if along[-1] < FLUSH_RUN:
      continue
    s = np.arange(FLUSH_STEP / 2, along[-1], FLUSH_STEP)
    k = np.clip(np.searchsorted(along, s, side='right') - 1, 0, len(seg) - 1)
    u = (pts[k + 1] - pts[k]) / np.maximum(seg[k], 1e-9)[:, None]
    p = pts[k] + u * (s - along[k])[:, None]
    r = np.column_stack([u[:, 1], -u[:, 0]])
    z = np.interp(s, [0.0, along[-1]], [ele.get(refs[0], np.nan), ele.get(refs[-1], np.nan)])
    for edge, sgn in ((hi, 1.0), (lo, -1.0)):
      runs.append((w, p + r * edge, r * sgn, z))
  if not runs:
    return []
  at_edge = np.concatenate([e for _, e, _, _ in runs])
  out_dir = np.concatenate([o for _, _, o, _ in runs])
  z = np.concatenate([zz for _, _, _, zz in runs])
  owner = np.concatenate([np.full(len(e), w, np.int64) for w, e, _, _ in runs])
  # facing another carriageway: within GAP_BEYOND beyond
  dists = np.arange(GAP_BEYOND[0], GAP_BEYOND[1] + 0.1, 1.0)
  probes = at_edge[:, None] + out_dir[:, None] * dists[None, :, None]
  facing = quads.inside(probes.reshape(-1, 2), np.repeat(owner, len(dists))).reshape(len(at_edge), len(dists)).any(1)
  cand = np.flatnonzero(facing & np.isfinite(z))
  flush = np.zeros(len(at_edge), bool)
  if len(cand):
    ok = np.ones(len(cand), bool)
    for d in FLUSH_OUT:
      q = at_edge[cand] + out_dir[cand] * d
      order = np.lexsort((np.floor(q[:, 1] / rp.TILE), np.floor(q[:, 0] / rp.TILE)))  # tile by tile
      code = np.zeros(len(cand), np.uint8)
      code[order] = tiles.sample(q[order, 0], q[order, 1], z[cand][order])[0]
      ok &= (code & 7) == rp.ROAD
    flush[cand] = ok
  strips, start = [], 0
  for w, e, _, _ in runs:  # runs of flush points along each way's side, FLUSH_RUN long at least
    f = flush[start:start + len(e)]
    k = 0
    while k < len(f):
      if not f[k]:
        k += 1
        continue
      j = k
      while j + 1 < len(f) and f[j + 1]:
        j += 1
      if (j - k + 1) * FLUSH_STEP >= FLUSH_RUN:
        pe, po = at_edge[start + k:start + j + 1], out_dir[start + k:start + j + 1]
        u = np.array([po[0, 1], -po[0, 0]]) * (1.0 if np.dot(pe[-1] - pe[0], [po[0, 1], -po[0, 0]]) >= 0 else -1.0)
        pe = np.vstack([pe[:1] - u * FLUSH_STEP / 2, pe, pe[-1:] + u * FLUSH_STEP / 2])  # the whole run's length
        po = np.vstack([po[:1], po, po[-1:]])
        strips.append((np.vstack([pe - po * FLUSH_STRIP[0], (pe + po * FLUSH_STRIP[1])[::-1]]),
                       osm.ways[w][0].get('highway', 'road'), float(np.nanmean(z[start + k:start + j + 1]))))
      k = j + 1
    start += len(e)
  return strips
