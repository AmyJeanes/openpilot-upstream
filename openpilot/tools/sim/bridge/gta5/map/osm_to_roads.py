#!/usr/bin/env python3
"""Writes the roads of an OSM file as the map view's roads.json: polylines in metres (x east, y north), each road's
middle, with its width kerb to kerb, how far each is trimmed back at its junctions, and the junctions' areas
(junctions.py); and with --lanes, the lines painted on them as lanes.json (osm_lanes.py, from the map's lane tags):
kerbs, carried round the junctions' corners (none between one-way ways side by side on one surface: a lane line,
side_by_side.py), white lines between lanes one way (dashed, solid where change:lanes
forbids crossing), yellow lines between the directions, stop and give way lines, and crossings (`footway=crossing`).
No lines are painted inside junctions but those of a road carried straight on through one (Junction.carried), nor
between a stop line and its junction; the moves through them from lane to lane (Junctions.movements: turn:lanes,
connectivity, restrictions) are guides the view can show there.

Where the map's nodes have heights (`ele`, as ynd_to_osm's do), each road, line, junction and signal has its height,
so the view can tell the level the car is on from the roads over and under it.

`--frame gta5` gives game coordinates (for ynd_to_osm's maps); otherwise metres from the file's centre.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from openpilot.tools.sim.bridge.gta5.map import osm_pbf
from openpilot.tools.sim.bridge.gta5.map.gta5_map import METRES_PER_DEGREE, to_game
from openpilot.tools.sim.bridge.gta5.map.junctions import Junctions, _left, _unit, apart, clip_outside, hull, off_islands
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, CENTRE, DIVIDER, EDGE, EDGE_LINE, FORWARD, MEDIAN, PARKING, OsmLanes, \
  offset_line
from openpilot.tools.sim.bridge.gta5.map.side_by_side import SideBySide

ROAD_CLASSES = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'service', 'track']
# lanes.json's kinds: a road's edge (kerb), white lines between lanes one way, yellow lines between the directions,
# stop lines, give way lines, crossings, the paths of the moves through junctions from lane to lane by their turn, and
# parking lanes on the carriageway (along their middle)
KINDS = ['edge', 'dashed', 'solid', 'centre', 'centre_dashed', 'stop', 'give_way', 'crossing', 'guide_left', 'guide_through',
         'guide_right', 'parking']
PARKING_STRIP = 'parking_strip'  # a parking lane's middle, in a road's lines
DOUBLE = 0.15  # m from a double line's middle to each of its lines
PAINTED = (DIVIDER, CENTRE, MEDIAN)  # the lines between lanes, which a road carried on through a junction keeps across it
CELL = 50.0  # m
NODE_REACH = 2.0  # m out from a junction node along each road that where they start reaches (node_ends) ...
NODE_PAD = 0.3  # m: ... and this much wider all round


def drawn(tags: dict) -> bool:
  """Whether a way is one of the roads drawn."""
  return tags.get('highway', '').removesuffix('_link') in ROAD_CLASSES


def join(ways):
  """Joins ways end to end where only two of the same kind meet, so the view draws long lines rather than many short
  ones. `ways` is [(kind, oneway, node ids)]; a one-way line keeps its direction."""
  ends = defaultdict(list)  # node -> indices of the ways ending there
  for i, (*_, nodes) in enumerate(ways):
    ends[nodes[0]].append(i)
    ends[nodes[-1]].append(i)
  done = [False] * len(ways)
  out = []
  for i, (kind, oneway, nodes) in enumerate(ways):
    if done[i]:
      continue
    done[i] = True
    line = list(nodes)
    for forward in (True, False):
      last = i
      while True:
        node = line[-1] if forward else line[0]
        if len(ends[node]) != 2:
          break
        j = ends[node][0] if ends[node][1] == last else ends[node][1]
        if done[j] or ways[j][:2] != (kind, oneway):
          break
        nxt = ways[j][2]
        if (nxt[0] == node) != forward:  # runs the other way
          if oneway:
            break
          nxt = nxt[::-1]
        done[j] = True
        last = j
        line = line + nxt[1:] if forward else nxt[:-1] + line
    out.append((kind, oneway, line))
  return out


JOINED = frozenset(KINDS.index(k) for k in ('edge', 'dashed', 'solid', 'centre', 'centre_dashed', 'parking'))
JOIN_TOL = 0.06  # m between two pieces' ends that are one point (a double line's two, offset again at a bend, end apart)


def join_lines(lines: list, layers: list, zs: list | None) -> tuple[list, list, list | None]:
  """lanes.json's lines [kind, x0, y0, ...] joined end to end where two pieces of one kind on one layer, and no
  other, end at the same point, as where a road carries on from one way to the next: one line, so its dashes run on."""
  ends = defaultdict(list)  # (layer, kind, cell) -> [(piece, its last point (1) or first (0))]
  for n, line in enumerate(lines):
    if line[0] in JOINED and len(line) >= 5:
      for e, (x, y) in ((0, line[1:3]), (1, line[-2:])):
        ends[(layers[n], line[0], round(x / JOIN_TOL), round(y / JOIN_TOL))].append((n, e))

  def partner(n, e):  # the one other piece ending where piece n's end e is
    line = lines[n]
    x, y = line[1:3] if e == 0 else line[-2:]
    cx, cy = round(x / JOIN_TOL), round(y / JOIN_TOL)
    found = {(m, f) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for m, f in ends.get((layers[n], line[0], cx + dx, cy + dy), ())
             if (m, f) != (n, e) and np.hypot(*(np.array(lines[m][1:3] if f == 0 else lines[m][-2:]) - (x, y))) <= JOIN_TOL}
    return next(iter(found)) if len(found) == 1 else None

  def heights(n):
    z = zs[n]
    return [z] * ((len(lines[n]) - 1) // 2) if isinstance(z, int) else list(z)
  done = [False] * len(lines)
  out, out_layers, out_z = [], [], []
  for n in range(len(lines)):
    if done[n]:
      continue
    done[n] = True
    pts = np.array(lines[n][1:], float).reshape(-1, 2)
    z = heights(n) if zs is not None else None
    for e in (1, 0):  # on from its end, then back from its start
      at, end = n, e
      while lines[at][0] in JOINED and (found := partner(at, end)) is not None and not done[found[0]] and partner(*found) == (at, end):
        m, f = found
        done[m] = True
        more = np.array(lines[m][1:], float).reshape(-1, 2)
        mz = heights(m) if zs is not None else None
        if (f == 1) == (e == 1):  # runs the other way
          more, mz = more[::-1], mz[::-1] if mz is not None else None
        if e == 1:
          pts = np.vstack([pts, more[1:]])
          z = z + mz[1:] if z is not None else None
        else:
          pts = np.vstack([more[:-1], pts])
          z = mz[:-1] + z if z is not None else None
        at, end = m, 1 - f
    out.append([lines[n][0]] + [round(float(v), 2) for v in pts.ravel()])
    out_layers.append(layers[n])
    if zs is not None:
      out_z.append(packed(z))
  return out, out_layers, out_z if zs is not None else None


def z_along(piece: np.ndarray, line: np.ndarray, z: np.ndarray) -> np.ndarray:
  """The heights [P] of the points of a piece of a polyline [N, 2] that has heights z [N], by where each falls along it."""
  if len(line) < 2:
    return np.full(len(piece), float(z[0]))
  a, ab = line[:-1], np.diff(line, axis=0)
  ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-12)
  out = np.empty(len(piece))
  for s in range(0, len(piece), 64):  # a block of points at a time, so a long line's [P, N] arrays stay small
    q = piece[s:s + 64]
    t = np.clip(np.einsum("pij,ij->pi", q[:, None] - a[None], ab) / ab2, 0.0, 1.0)
    k = np.argmin(np.hypot(*(a[None] + ab[None] * t[..., None] - q[:, None]).transpose(2, 0, 1)), axis=1)
    tk = t[np.arange(len(q)), k]
    out[s:s + 64] = z[k] + (z[k + 1] - z[k]) * tk
  return out


def packed(z) -> int | list[int]:
  """Heights to the metre, as one number where they're all the same."""
  z = np.rint(np.atleast_1d(z)).astype(int).tolist()
  return z[0] if min(z) == max(z) else z


def marks(style: str | None, white: bool) -> list[tuple[int, float]]:
  """A line's style as lanes.json kinds and offsets from it, white or yellow: [(kind, m right)]."""
  dashed, solid = (1, 2) if white else (4, 3)
  return {'dashed': [(dashed, 0.0)], 'solid': [(solid, 0.0)], 'double_solid': [(solid, -DOUBLE), (solid, DOUBLE)],
          'dashed_solid': [(dashed, -DOUBLE), (solid, DOUBLE)], 'solid_dashed': [(solid, -DOUBLE), (dashed, DOUBLE)]}.get(style or '', [])


def level(tags: dict) -> tuple[int, int]:
  """A way's layer (`layer`, else 1 on a bridge, -1 in a tunnel) and whether it's on the ground (0), a bridge (1) or in a
  tunnel (2)."""
  bridge = tags.get('bridge', 'no') != 'no'
  tunnel = tags.get('tunnel', 'no') not in ('no', 'building_passage')
  try:
    layer = max(min(int(tags['layer']), 5), -5)
  except (KeyError, ValueError):
    layer = 1 if bridge else -1 if tunnel else 0
  return layer, 1 if bridge else 2 if tunnel else 0


def strip(osm: OsmLanes, wid: int, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray] | None:
  """A way whose kerbs move along it (OsmLanes.taper, .blend): its left and right kerbs [N, 2] seen along it, as
  line_geometry draws them, trimmed back t0 m from its first node and t1 m from its last; None where they don't."""
  found = osm.taper(wid) or osm.blend(wid)
  if found is None:
    return None
  edges = [g for line, g in osm.line_geometry(wid) if line.kind == EDGE]
  left, right = edges[0], edges[-1]
  if found[0] == BACKWARD:
    left, right = right[::-1], left[::-1]
  if len(left) != len(right) or len(left) < 2:
    return None
  along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff((left + right) / 2, axis=0).T))))
  if t0 + t1 >= along[-1] - 0.05:
    return None
  s = np.unique(np.concatenate(([t0], along[(along > t0) & (along < along[-1] - t1)], [along[-1] - t1])))

  def at(line):
    return np.stack([np.interp(s, along, line[:, 0]), np.interp(s, along, line[:, 1])], axis=1)
  kerbs = left, right
  left, right = at(left), at(right)
  # a trimmed end on the kerbs where the junction's mouth meets them: each kerb's nearest point to the way's line there
  line, way_along = osm.way_points(wid), osm.way_along(wid)
  for t, end in ((t0, 0), (t1, -1)):
    if t > 0:
      v = t if end == 0 else way_along[-1] - t
      p = np.array([np.interp(v, way_along, line[:, 0]), np.interp(v, way_along, line[:, 1])])
      left[end], right[end] = nearest(kerbs[0], p), nearest(kerbs[1], p)
  return left, right


def nearest(line: np.ndarray, p: np.ndarray) -> np.ndarray:
  """The point of a polyline [N, 2] nearest p."""
  a, ab = line[:-1], np.diff(line, axis=0)
  t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-12), 0.0, 1.0)
  q = a + ab * t[:, None]
  return q[int(np.argmin(np.hypot(*(q - p).T)))]


def joint(osm: OsmLanes, junctions: Junctions, node: int, reach: float = 0.6) -> np.ndarray | None:
  """Junctions.joint from the kerbs as drawn (OsmLanes.edges_at), where flat-ended roads meet outside junctions: the
  hull of their ends, and where a road just carries on, of its kerbs' mitred corner too, so nothing is left between a
  road piece's flat end and the next one's."""
  steps = junctions.steps.get(node, ())
  if len(steps) < 2:
    return None
  p, pts = osm.node_xy(node), []
  for w, nxt, fwd in steps:
    u = _unit(osm.node_xy(nxt) - p)
    refs = osm.ways[w][1]
    lo, hi = osm.edges_at(w, float(osm.way_along(w)[refs.index(node)]), FORWARD if fwd else BACKWARD)
    r = -_left(u)
    pts += [p + r * lo, p + r * hi, p + u * reach + r * lo, p + u * reach + r * hi]
  if len(steps) == 2:  # the corner where the kerbs of the way in and the way out meet
    (_, a, _), (w, b, fwd) = steps
    refs = osm.ways[w][1]
    line = np.array([osm.node_xy(a), p, osm.node_xy(b)])
    for off in osm.edges_at(w, float(osm.way_along(w)[refs.index(node)]), FORWARD if fwd else BACKWARD):
      pts.append(offset_line(line, off)[1])
  return hull(np.array(pts))


def node_ends(members, reach: float = NODE_REACH, pad: float = NODE_PAD) -> np.ndarray:
  """Where the roads out of one junction node start: the hull of their kerbs from the node out to `reach` (or to where
  each is trimmed, if nearer), `pad` wider all round but no further out than that along each road."""
  pts = np.array([kerb.at(s) for m in members for kerb in (m.left, m.right) for s in (0.0, max(min(m.trim, reach) - pad, 0.0))])
  out = hull(pts)
  centre = out.mean(0)
  away = out - centre
  return out + away / np.maximum(np.hypot(*away.T), 1e-9)[:, None] * pad


class PaintAreas:
  """Where no lines are painted: each junction's area and the road from each stop line in to it. An area cuts the lines
  on its junction's layer (its roads' highest) and those of the roads meeting it on any layer, as a bridge starting
  there: layer tags alone can't tell a road meeting a junction from one passing under it. A junction's area doesn't cut
  the lines of the road carried on through it (Junction.carried). Where the roads out of a junction's node start
  (node_ends) also cuts the lines of those trimmed back from it: a road meeting the junction at a slant has its lines'
  ends at the node reaching out of the area, across the mouth of a road beside it trimmed little or not at all."""
  def __init__(self, junctions: Junctions):
    js = junctions.junctions
    self.layer = [max(level(junctions.ways[w][0])[0] for w in j.ways) for j in js]
    self.areas = [(j.centre, j.polygon) for j in js] + [(s.area.mean(0), s.area) for j in js for s in j.stops]
    of = list(range(len(js))) + [n for n, j in enumerate(js) for _ in j.stops]  # each area's junction
    self._own: dict[int, set[int]] = {}  # a node_ends area -> the ways whose lines it cuts
    for n, j in enumerate(js):
      for node in j.nodes:
        members = [m for arm in j.arms for m in arm.members if m.start == node]
        ways = {w for m in members if m.trim > 0.0 for k, (w, _) in enumerate(m.ways) if k == 0 or Junctions._along(m, k) < m.trim}
        if len(members) < 2 or not ways:
          continue
        self._own[len(self.areas)] = ways
        outline = node_ends(members)
        self.areas.append((outline.mean(0), outline))
        of.append(n)
    self._of, self._roads, self._carried = of, [j.roads for j in js], [j.carried for j in js]
    self._index: dict[tuple[int, int], list[int]] = defaultdict(list)
    for n, (_, a) in enumerate(self.areas):
      lo, hi = a.min(0) // CELL, a.max(0) // CELL
      for cx in range(int(lo[0]), int(hi[0]) + 1):
        for cy in range(int(lo[1]), int(hi[1]) + 1):
          self._index[(cx, cy)].append(n)

  def near(self, pts: np.ndarray, layer: int, ways=frozenset(), kerbs_only: bool = False, but: int | None = None,
           offset: float | None = None) -> list:
    """The areas [(centre, polygon)] that cut a line near points [K, 2] on a layer, along these ways; for a kerb only
    the junctions' areas, but for junction `but`'s. `offset`: the line's m right of its ways, where it's one painted
    between lanes (Junction.carried)."""
    lo, hi = pts.min(0) // CELL, pts.max(0) // CELL
    found = {n for cx in range(int(lo[0]), int(hi[0]) + 1) for cy in range(int(lo[1]), int(hi[1]) + 1) for n in self._index.get((cx, cy), ())}
    key = None if offset is None else round(offset, 2)
    out = []
    for n in sorted(found):
      j = self._of[n]
      carried = key is not None and any(key in self._carried[j].get(w, ()) for w in ways)
      if n in self._own:
        if self._own[n] & ways and j != but and not carried:
          out.append(self.areas[n])
      elif (self.layer[j] == layer or self._roads[j] & ways) and n != but and (not kerbs_only or n < len(self.layer)) \
          and not (carried and n < len(self.layer)):
        out.append(self.areas[n])
    return out


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('osm')
  p.add_argument('out')
  p.add_argument('--lanes', help="the lines painted on the roads, from the map's lane tags (lanes.json)")
  p.add_argument('--frame', choices=['gta5', 'local'], default='gta5')
  args = p.parse_args()

  data = osm_pbf.read(args.osm, relations=('restriction', 'connectivity'))
  if args.frame == 'gta5':
    project = to_game
  else:
    lat0, lon0 = (data.lat.min() + data.lat.max()) / 2, (data.lon.min() + data.lon.max()) / 2

    def project(lat, lon):
      return (lon - lon0) * METRES_PER_DEGREE * np.cos(np.radians(lat0)), (lat - lat0) * METRES_PER_DEGREE
  osm = OsmLanes(data, project)

  def points(nodes):
    return osm.xy[data.index(nodes)]

  def ele(node) -> float:
    return float(data.node_tags[node]['ele'])

  def heights(nodes) -> np.ndarray:
    return np.array([ele(n) for n in nodes])
  junctions = Junctions(osm, drawn)
  carried_inside = {w: lines for j in junctions.junctions for w, lines in j.carried.items() if w in junctions.inside}
  levels = {wid: level(tags) for wid, (tags, _) in junctions.ways.items()}
  try:  # the nodes' heights, where the map has them all
    for _, refs in junctions.ways.values():
      heights(refs)
    high = True
  except (KeyError, ValueError):
    high = False
  roads, layouts, tapered = [], [], []
  strips, strip_z = [], []  # roads whose kerbs move along them, drawn zoomed in between their kerbs as drawn
  for wid, (tags, refs) in junctions.ways.items():
    road = osm.lanes(wid)
    lo, hi = road.edges(FORWARD)
    lanes = tags.get('lanes', '1').split(';')[0]
    c = ROAD_CLASSES.index(tags['highway'].removesuffix('_link'))
    kerbs = None if wid in junctions.inside else       strip(osm, wid, junctions.trims.get((wid, refs[0]), 0.0), junctions.trims.get((wid, refs[-1]), 0.0))
    if kerbs is not None:
      strips.append([c, *levels[wid], len(kerbs[0])] + [round(float(v), 2) for v in np.concatenate(kerbs).ravel()])
      if high:
        strip_z.append(packed(z_along((kerbs[0] + kerbs[1]) / 2, points(refs), heights(refs))))
    # a strip's way is drawn whole only zoomed out, as a way inside a junction (trims -1)
    kind = (c, int(lanes) if lanes.isdigit() else 1, round(hi - lo, 1), round((lo + hi) / 2, 1),
            wid in junctions.inside or kerbs is not None, *levels[wid])
    roads.append((kind, tags.get('oneway') == 'yes', refs))
    if wid not in junctions.inside and (osm.taper(wid) is not None or osm.blend(wid) is not None):
      tapered.append(wid)  # its lines move along it: drawn on their own
    elif wid not in junctions.inside:
      lines = tuple((ln.kind, round(ln.offset, 2), ln.style, ln.white) for ln in road.lines(FORWARD) if ln.kind == EDGE or (road.markings
                    and ln.kind != PARKING)) + tuple((PARKING_STRIP, round((a + b) / 2, 2), None, True) for a, b in road.parking_lanes(FORWARD))
      layouts.append(((levels[wid][0], lines), True, refs))  # lines are offsets along a way's direction: join only ways going on
    elif carried_inside.get(wid) and road.markings:  # inside a junction, only the lines carried across it
      lines = tuple((ln.kind, round(ln.offset, 2), ln.style, ln.white) for ln in road.lines(FORWARD)
                    if ln.kind in (CENTRE, DIVIDER, MEDIAN) and round(ln.offset, 2) in carried_inside[wid])
      if lines:
        layouts.append(((levels[wid][0], lines), True, refs))

  def trim(a, b):  # how far the road from node a on to b is trimmed back at a
    return round(junctions.trims.get((osm.pairs[(a, b)][0], a), 0.0), 1)

  out_ways, out_widths, out_trims, out_levels, out_heights = [], [], [], [], []
  ends: dict[int, int] = {}  # node -> the best class of the road pieces ending there
  for (c, lanes, width, middle, inside, layer, structure), oneway, nodes in join(roads):
    pts = points(nodes)
    pts = offset_line(pts, middle) if middle else pts
    out_ways.append([c, lanes, int(oneway)] + [round(float(v), 1) for v in pts.ravel()])
    out_widths.append(width)
    out_trims += [-1, -1] if inside else [trim(nodes[0], nodes[1]), trim(nodes[-1], nodes[-2])]
    out_levels += [layer, structure]
    if high:
      z = heights(nodes)
      out_heights.append(packed(z_along(pts, points(nodes), z) if middle else z))
    for n in (nodes[0], nodes[-1]):
      ends[n] = min(ends.get(n, c), c)

  def layer_at(ways):
    return max(levels[w][0] for w in ways)
  paint = PaintAreas(junctions)
  area_layer = paint.layer
  areas, area_z = [], []
  junction_z = [float(np.mean(heights(j.nodes))) for j in junctions.junctions] if high else []
  for n, j in enumerate(junctions.junctions):
    c = min(ROAD_CLASSES.index(junctions.ways[w][0]['highway'].removesuffix('_link')) for w in j.ways)
    areas.append([c, area_layer[n]] + [round(float(v), 2) for v in np.concatenate((j.centre, j.polygon.ravel()))])
    if high:
      area_z.append(packed(junction_z[n]))
  in_junctions = {n for j in junctions.junctions for n in j.nodes}
  for n, c in ends.items():  # where flat-ended road pieces meet outside junctions
    if n not in in_junctions and (area := joint(osm, junctions, n)) is not None and len(area) >= 3:
      areas.append([c, layer_at({w for w, _, _ in junctions.steps[n]})] +
                   [round(float(v), 2) for v in np.concatenate((osm.node_xy(n), area.ravel()))])
      if high:
        area_z.append(packed(ele(n)))
  signals = [n for n, tags in data.node_tags.items() if tags.get('highway') == 'traffic_signals']
  out = {
    'classes': ROAD_CLASSES,
    'ways': out_ways,  # [class, lanes, oneway, x0, y0, x1, y1, ...]: the road's middle
    'widths': out_widths,  # m, kerb to kerb
    # m each way is trimmed back from its first and last point, where it meets a junction; -1, -1 inside a junction
    'trims': out_trims,
    'levels': out_levels,  # each way's layer and whether it's on the ground (0), a bridge (1) or in a tunnel (2)
    # [class (its main road's), layer, cx, cy, x0, y0, ...]: each junction's area round its centre (it can fold back on
    # itself: fill the triangles from the centre to each edge), and where road pieces meet elsewhere
    'junctions': areas,
    'signals': [[round(float(v), 1) for v in osm.node_xy(n)] for n in signals],
    # [class, layer, structure, n, left kerb x0, y0, ... (n points), right kerb x0, y0, ...]: roads whose kerbs move
    # along them (OsmLanes.taper, .blend), trimmed back at junctions, drawn between their kerbs zoomed in
    'strips': strips,
  }
  if high:  # m: each way's height at each of its points (one number where it's level), each junction's, each signal's
    out.update(heights=out_heights, junction_heights=area_z, signal_heights=[packed(ele(n)) for n in signals],
               strip_heights=strip_z)
  with open(args.out, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
  print(f"{len(out['ways'])} ways, {len(areas)} junctions, {len(out['signals'])} signals -> {args.out}")

  if not args.lanes:
    return
  if not osm.tagged:
    print(f"no lane tags in {args.osm}: no {args.lanes}")
    return
  # no lines inside junctions, nor from a stop line in to its junction (PaintAreas)
  lines, line_layers, line_z = [], [], []

  def add(k, piece, layer, z):
    lines.append([k] + [round(float(v), 2) for v in piece.ravel()])
    line_layers.append(layer)
    if high:
      line_z.append(packed(z))
  # kerbs only where the road surface ends, not between one-way ways side by side (side_by_side.py)
  side = SideBySide(osm, [w for w in junctions.ways if w not in junctions.inside], lambda w: levels[w][0], paint)
  for (layer, sig), _, nodes in join(layouts):
    pts = points(nodes)
    z = heights(nodes) if high else None
    ways = {osm.pairs[(a, b)][0] for a, b in zip(nodes[:-1], nodes[1:], strict=True)}
    right = max((offset for kind, offset, *_ in sig if kind == EDGE), default=None)
    painted_edges = {offset for kind, offset, *_ in sig if kind == EDGE_LINE}
    for kind, offset, style, white in sig:
      for k, off in [(0, 0.0)] if kind == EDGE else [(KINDS.index('parking'), 0.0)] if kind == PARKING_STRIP else \
          marks(style, white):
        geom = osm.offset_nodes(nodes, offset + off)
        for piece in clip_outside(geom, paint.near(geom, layer, ways, kind == EDGE, offset=offset if kind in PAINTED else None)):
          if kind != EDGE:
            add(k, piece, layer, z_along(piece, pts, z) if high else None)
            continue
          kerbs, between = side.kerb(piece, z_along(piece, pts, z) if high else None, layer, ways, offset == right)
          for p in (q for kerb in kerbs for q in off_islands(kerb, junctions.islands)):
            add(k, p, layer, z_along(p, pts, z) if high else None)
          for p, s in between if offset not in painted_edges else ():  # its own line is in its paint
            add(KINDS.index(s), p, layer, z_along(p, pts, z) if high else None)
  for wid in tapered:  # lanes opening or closing along it (osm_lanes.OsmLanes.taper)
    refs, road, layer = junctions.ways[wid][1], osm.lanes(wid), levels[wid][0]
    pts = points(refs)
    z = heights(refs) if high else None
    geometry = osm.line_geometry(wid)
    right = max((line.offset for line, _ in geometry if line.kind == EDGE), default=None)
    painted_edges = {round(line.offset, 2) for line, _ in geometry if line.kind == EDGE_LINE}
    for line, geom in geometry:
      if line.kind != EDGE and (not road.markings or line.kind == PARKING):
        continue
      for k, off in [(0, 0.0)] if line.kind == EDGE else marks(line.style, line.white):
        g = offset_line(geom, off) if off else geom
        # a blend's lines move only at its other end: at a junction they're where its lanes put them
        for piece in clip_outside(g, paint.near(g, layer, {wid}, line.kind == EDGE, offset=line.offset if line.kind in PAINTED else None)):
          if line.kind != EDGE:
            add(k, piece, layer, z_along(piece, pts, z) if high else None)
            continue
          kerbs, between = side.kerb(piece, z_along(piece, pts, z) if high else None, layer, {wid}, line.offset == right)
          for p in (q for kerb in kerbs for q in off_islands(kerb, junctions.islands)):  # as for the ways above
            add(k, p, layer, z_along(p, pts, z) if high else None)
          for p, st in between if round(line.offset, 2) not in painted_edges else ():
            add(KINDS.index(st), p, layer, z_along(p, pts, z) if high else None)
    for a, b in road.parking_lanes(FORWARD):
      g = offset_line(pts, (a + b) / 2)
      for piece in clip_outside(g, paint.near(g, layer, {wid})):
        add(KINDS.index('parking'), piece, layer, z_along(piece, pts, z) if high else None)
  for n, j in enumerate(junctions.junctions):
    jz = junction_z[n] if high else None
    for kerb in j.kerbs:  # where junctions overlap, neither's kerb crosses the other
      for piece in clip_outside(kerb, paint.near(kerb, area_layer[n], kerbs_only=True, but=n)):
        for p in off_islands(piece, junctions.islands):
          add(KINDS.index('edge'), p, area_layer[n], jz)
    for s in j.stops:
      add(KINDS.index(s.kind), s.line, area_layer[n], jz)
    for mv in junctions.movements(j):
      add(KINDS.index(f'guide_{mv.kind}'), mv.path, area_layer[n], jz)
  if junctions.island_outlines:  # painted islands' outlines, in their paint, at their nearest road node's height
    road_nodes = sorted({n for _, refs in junctions.ways.values() for n in refs})
    near_xy, near_z = points(road_nodes), heights(road_nodes) if high else None
    # (but where the road's own lines draw them already)
    kinds_of = {'yellow': {KINDS.index(k) for k in ('centre', 'centre_dashed')}, 'white': {KINDS.index(k) for k in ('solid', 'dashed', 'edge')}}
    drawn_segs = {c: np.concatenate([np.stack([p[:-1], p[1:]], 1) for ln in lines if ln[0] in ks
                                      for p in [np.array(ln[1:]).reshape(-1, 2)]] or [np.zeros((0, 2, 2))]) for c, ks in kinds_of.items()}
    for xy, colour in junctions.island_outlines:
      ring = np.vstack([xy, xy[:1]]) if np.hypot(*(xy[-1] - xy[0])) > 1e-6 else xy
      z = near_z[np.argmin(np.hypot(*(near_xy - ring.mean(0)).T))] if high else None
      for piece in apart(ring, drawn_segs[colour]):
        add(KINDS.index('centre' if colour == 'yellow' else 'solid'), piece, 0, None if z is None else np.full(len(piece), z))
  crossings = junctions.crossing_lines()
  if high:  # a crossing's height is its nearest road node's
    road_nodes = sorted({n for _, refs in junctions.ways.values() for n in refs})
    road_xy, road_z = points(road_nodes), heights(road_nodes)
  for c in crossings:
    add(KINDS.index('crossing'), c, 0, road_z[np.argmin(np.hypot(*(road_xy - c.mean(0)).T))] if high else None)
  lines, line_layers, line_z = join_lines(lines, line_layers, line_z if high else None)
  out = {'kinds': KINDS, 'lines': lines, 'layers': line_layers}
  if high:
    out['heights'] = line_z  # m: each line's height at each of its points, one number where it's level
  with open(args.lanes, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
  stops = sum(len(j.stops) for j in junctions.junctions)
  print(f"{len(lines)} lines, {stops} stop lines, {len(crossings)} crossings -> {args.lanes}")


if __name__ == '__main__':
  main()
