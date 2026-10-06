#!/usr/bin/env python3
"""Writes the roads of an OSM file as the map view's roads.json: polylines in metres (x east, y north), each road's
middle, with its width kerb to kerb, how far each is trimmed back at its junctions, and the junctions' areas
(junctions.py); and with --lanes, the lines painted on them as lanes.json (osm_lanes.py, from the map's lane tags):
kerbs, carried round the junctions' corners, white lines between lanes one way (dashed, solid where change:lanes
forbids crossing), yellow lines between the directions, stop and give way lines, and crossings (`footway=crossing`).
No lines are painted inside junctions, nor between a stop line and its junction; the moves through them from lane to
lane (Junctions.movements: turn:lanes, connectivity, restrictions) are guides the view can show there.

`--frame gta5` gives game coordinates (for ynd_to_osm's maps); otherwise metres from the file's centre.
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from openpilot.tools.sim.bridge.gta5.map import osm_pbf
from openpilot.tools.sim.bridge.gta5.map.gta5_map import METRES_PER_DEGREE, to_game
from openpilot.tools.sim.bridge.gta5.map.junctions import Junctions, clip_outside
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import DIVIDER, EDGE, FORWARD, MEDIAN, OsmLanes, offset_line

ROAD_CLASSES = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'service', 'track']
# lanes.json's kinds: a road's edge (kerb), white lines between lanes one way, yellow lines between the directions,
# stop lines, give way lines, crossings, and the paths of the moves through junctions from lane to lane by their turn
KINDS = ['edge', 'dashed', 'solid', 'centre', 'centre_dashed', 'stop', 'give_way', 'crossing', 'guide_left', 'guide_through',
         'guide_right']
DOUBLE = 0.15  # m from a double line's middle to each of its lines
CELL = 50.0  # m


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


def marks(style: str | None, divider: bool) -> list[tuple[int, float]]:
  """A line's style as lanes.json kinds and offsets from it: [(kind, m right)]."""
  dashed, solid = (1, 2) if divider else (4, 3)
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

  def drawn(tags):
    return tags.get('highway', '').removesuffix('_link') in ROAD_CLASSES
  junctions = Junctions(osm, drawn)
  levels = {wid: level(tags) for wid, (tags, _) in junctions.ways.items()}
  roads, layouts = [], []
  for wid, (tags, refs) in junctions.ways.items():
    road = osm.lanes(wid)
    lo, hi = road.edges(FORWARD)
    lanes = tags.get('lanes', '1').split(';')[0]
    kind = (ROAD_CLASSES.index(tags['highway'].removesuffix('_link')), int(lanes) if lanes.isdigit() else 1,
            round(hi - lo, 1), round((lo + hi) / 2, 1), wid in junctions.inside, *levels[wid])
    roads.append((kind, tags.get('oneway') == 'yes', refs))
    if wid not in junctions.inside:
      lines = tuple((ln.kind, round(ln.offset, 2), ln.style) for ln in road.lines(FORWARD) if ln.kind == EDGE or road.markings)
      layouts.append(((levels[wid][0], lines), True, refs))  # lines are offsets along a way's direction: join only ways going on

  def trim(a, b):  # how far the road from node a on to b is trimmed back at a
    return round(junctions.trims.get((osm.pairs[(a, b)][0], a), 0.0), 1)

  out_ways, out_widths, out_trims, out_levels = [], [], [], []
  ends: dict[int, int] = {}  # node -> the best class of the road pieces ending there
  for (c, lanes, width, middle, inside, layer, structure), oneway, nodes in join(roads):
    pts = points(nodes)
    pts = offset_line(pts, middle) if middle else pts
    out_ways.append([c, lanes, int(oneway)] + [round(float(v), 1) for v in pts.ravel()])
    out_widths.append(width)
    out_trims += [-1, -1] if inside else [trim(nodes[0], nodes[1]), trim(nodes[-1], nodes[-2])]
    out_levels += [layer, structure]
    for n in (nodes[0], nodes[-1]):
      ends[n] = min(ends.get(n, c), c)

  def layer_at(ways):
    return max(levels[w][0] for w in ways)
  areas, area_layer = [], []
  for j in junctions.junctions:
    c = min(ROAD_CLASSES.index(junctions.ways[w][0]['highway'].removesuffix('_link')) for w in j.ways)
    area_layer.append(layer_at(j.ways))
    areas.append([c, area_layer[-1]] + [round(float(v), 2) for v in np.concatenate((j.centre, j.polygon.ravel()))])
  in_junctions = {n for j in junctions.junctions for n in j.nodes}
  for n, c in ends.items():  # where flat-ended road pieces meet outside junctions
    if n not in in_junctions and (joint := junctions.joint(n)) is not None and len(joint) >= 3:
      areas.append([c, layer_at({w for w, _, _ in junctions.steps[n]})] +
                   [round(float(v), 2) for v in np.concatenate((osm.node_xy(n), joint.ravel()))])
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
  }
  with open(args.out, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
  print(f"{len(out['ways'])} ways, {len(areas)} junctions, {len(out['signals'])} signals -> {args.out}")

  if not args.lanes:
    return
  if not osm.tagged:
    print(f"no lane tags in {args.osm}: no {args.lanes}")
    return
  # no lines inside junctions, nor from a stop line in to its junction; each area cuts only its own layer's lines
  kerb_areas = [(j.centre, j.polygon) for j in junctions.junctions]
  paint_areas = kerb_areas + [(s.area.mean(0), s.area) for j in junctions.junctions for s in j.stops]
  paint_layer = area_layer + [area_layer[n] for n, j in enumerate(junctions.junctions) for _ in j.stops]
  index: dict[tuple[int, int], list[int]] = defaultdict(list)
  for n, (_, a) in enumerate(paint_areas):
    lo, hi = a.min(0) // CELL, a.max(0) // CELL
    for cx in range(int(lo[0]), int(hi[0]) + 1):
      for cy in range(int(lo[1]), int(hi[1]) + 1):
        index[(cx, cy)].append(n)

  def near(pts, layer, kerbs_only, but=None):
    lo, hi = pts.min(0) // CELL, pts.max(0) // CELL
    found = {n for cx in range(int(lo[0]), int(hi[0]) + 1) for cy in range(int(lo[1]), int(hi[1]) + 1) for n in index.get((cx, cy), ())}
    return [paint_areas[n] for n in sorted(found) if paint_layer[n] == layer and n != but and (not kerbs_only or n < len(kerb_areas))]

  lines, line_layers = [], []

  def add(k, piece, layer):
    lines.append([k] + [round(float(v), 2) for v in piece.ravel()])
    line_layers.append(layer)
  for (layer, sig), _, nodes in join(layouts):
    pts = points(nodes)
    for kind, offset, style in sig:
      # a median's edges one line each, as maps paint it, rather than the divider's double line on both
      for k, off in [(0, 0.0)] if kind == EDGE else marks('solid' if kind == MEDIAN else style, kind == DIVIDER):
        geom = offset_line(pts, offset + off)
        for piece in clip_outside(geom, near(geom, layer, kind == EDGE)):
          add(k, piece, layer)
  for n, j in enumerate(junctions.junctions):
    for kerb in j.kerbs:  # where junctions overlap, neither's kerb crosses the other
      for piece in clip_outside(kerb, near(kerb, area_layer[n], True, but=n)):
        add(KINDS.index('edge'), piece, area_layer[n])
    for s in j.stops:
      add(KINDS.index(s.kind), s.line, area_layer[n])
    for mv in junctions.movements(j):
      add(KINDS.index(f'guide_{mv.kind}'), mv.path, area_layer[n])
  crossings = junctions.crossing_lines()
  for c in crossings:
    add(KINDS.index('crossing'), c, 0)
  with open(args.lanes, 'w') as f:
    json.dump({'kinds': KINDS, 'lines': lines, 'layers': line_layers}, f, separators=(',', ':'))
  stops = sum(len(j.stops) for j in junctions.junctions)
  print(f"{len(lines)} lines, {stops} stop lines, {len(crossings)} crossings -> {args.lanes}")


if __name__ == '__main__':
  main()
