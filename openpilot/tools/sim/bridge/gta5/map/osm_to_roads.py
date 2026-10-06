#!/usr/bin/env python3
"""Writes the roads of an OSM file as the map view's roads.json: polylines in metres (x east, y north), each road's
middle, with its width kerb to kerb; and with --lanes, the lines painted on them as lanes.json (osm_lanes.py, from the
map's lane tags): road edges, white lines between lanes one way (dashed, solid where change:lanes forbids crossing),
yellow lines between the directions; none inside junctions (where roads cross) and only edges on the ways into them.

`--frame gta5` gives game coordinates (for ynd_to_osm's maps); otherwise metres from the file's centre.
"""
import argparse
import json
import math
from collections import defaultdict

import numpy as np

from openpilot.tools.sim.bridge.gta5.map import osm_pbf
from openpilot.tools.sim.bridge.gta5.map.gta5_map import METRES_PER_DEGREE, to_game
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import DIVIDER, EDGE, FORWARD, OsmLanes, offset_line

ROAD_CLASSES = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'service', 'track']
# lanes.json's kinds: a road's edge, white lines between lanes one way, yellow lines between the directions
KINDS = ['edge', 'dashed', 'solid', 'centre', 'centre_dashed']
DOUBLE = 0.15  # m from a double line's middle to each of its lines
CROSS = (45.0, 135.0)  # deg between two ways at a node where roads cross
CELL = 50.0  # m
JUNCTION_REACH = 0.8  # of the widest road at a node where roads cross: a junction can spread over several nodes
JUNCTION_LINK = 15.0  # m: a way this short into a node where roads cross, or between two, is inside the junction


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


def outside(line: np.ndarray, circles: list[tuple[np.ndarray, float]], min_len: float = 0.3) -> list[np.ndarray]:
  """The parts of a polyline outside the circles [(centre, radius)]."""
  pieces: list[list] = []
  run: list = []
  for a, b in zip(line[:-1], line[1:], strict=True):
    d = b - a
    dd = float(d @ d)
    inside = []
    for c, r in circles:
      t = float((c - a) @ d) / max(dd, 1e-9)
      h2 = (r * r - float(np.sum((a + d * t - c) ** 2))) / max(dd, 1e-9)
      if h2 > 0 and t + math.sqrt(h2) > 0 and t - math.sqrt(h2) < 1:
        inside.append((max(t - math.sqrt(h2), 0.0), min(t + math.sqrt(h2), 1.0)))
    t = 0.0
    for lo, hi in sorted(inside) + [(1.0, 1.0)]:
      if lo > t:
        if t > 0 or not run:
          if run:
            pieces.append(run)
          run = [a + d * t]
        run.append(a + d * lo)
      if lo < 1.0 and run:
        pieces.append(run)
        run = []
      t = max(t, hi)
  if run:
    pieces.append(run)
  return [np.array(p) for p in pieces if len(p) >= 2 and np.hypot(*np.diff(np.array(p), axis=0).T).sum() >= min_len]


def crossings(osm: OsmLanes, widths: dict[int, float]) -> dict[int, tuple[np.ndarray, float]]:
  """The nodes where roads cross: {node: circle round it, of JUNCTION_REACH times the widest road there}."""
  heads = defaultdict(list)
  for wid, (_, refs) in osm.ways.items():
    if len(refs) < 2:
      continue
    for i, j in ((0, 1), (-1, -2)):
      d = osm.node_xy(refs[j]) - osm.node_xy(refs[i])
      heads[refs[i]].append((math.degrees(math.atan2(d[1], d[0])), widths[wid]))
  out = {}
  for node, hs in heads.items():
    if len(hs) >= 3 and any(CROSS[0] <= abs((a - b + 180) % 360 - 180) <= CROSS[1] for n, (a, _) in enumerate(hs) for b, _ in hs[n + 1:]):
      out[node] = (osm.node_xy(node), JUNCTION_REACH * max(w for _, w in hs))
  return out


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('osm')
  p.add_argument('out')
  p.add_argument('--lanes', help="the lines painted on the roads, from the map's lane tags (lanes.json)")
  p.add_argument('--frame', choices=['gta5', 'local'], default='gta5')
  args = p.parse_args()

  data = osm_pbf.read(args.osm, relations=())
  if args.frame == 'gta5':
    project = to_game
  else:
    lat0, lon0 = (data.lat.min() + data.lat.max()) / 2, (data.lon.min() + data.lon.max()) / 2

    def project(lat, lon):
      return (lon - lon0) * METRES_PER_DEGREE * np.cos(np.radians(lat0)), (lat - lat0) * METRES_PER_DEGREE
  osm = OsmLanes(data, project)

  def points(nodes):
    return osm.xy[data.index(nodes)]

  roads, widths, layouts = [], {}, []
  for wid in osm.ways:
    lo, hi = osm.lanes(wid).edges(FORWARD)
    widths[wid] = hi - lo
  circles = crossings(osm, widths)
  for wid, (tags, refs) in osm.ways.items():
    lo, hi = osm.lanes(wid).edges(FORWARD)
    cls = tags.get('highway', '').removesuffix('_link')
    if cls not in ROAD_CLASSES or len(refs) < 2:
      continue
    lanes = tags.get('lanes', '1').split(';')[0]
    kind = (ROAD_CLASSES.index(cls), int(lanes) if lanes.isdigit() else 1, round(hi - lo, 1), round((lo + hi) / 2, 1))
    roads.append((kind, tags.get('oneway') == 'yes', refs))
    road = osm.lanes(wid)
    ends = (refs[0] in circles) + (refs[-1] in circles)
    inside = ends == 2 or ends and float(np.hypot(*np.diff(points(refs), axis=0).T).sum()) < JUNCTION_LINK
    lines = tuple((ln.kind, round(ln.offset, 2), ln.style) for ln in road.lines(FORWARD)
                  if not inside and (ln.kind == EDGE or road.markings and not ends))
    layouts.append((lines, True, refs))  # lines are offsets along a way's direction: join only ways going on

  out_ways, out_widths = [], []
  for (c, lanes, width, middle), oneway, nodes in join(roads):
    pts = points(nodes)
    pts = offset_line(pts, middle) if middle else pts
    out_ways.append([c, lanes, int(oneway)] + [round(float(v), 1) for v in pts.ravel()])
    out_widths.append(width)
  signals = [n for n, tags in data.node_tags.items() if tags.get('highway') == 'traffic_signals']
  out = {
    'classes': ROAD_CLASSES,
    'ways': out_ways,  # [class, lanes, oneway, x0, y0, x1, y1, ...]: the road's middle
    'widths': out_widths,  # m, kerb to kerb
    'signals': [[round(float(v), 1) for v in osm.node_xy(n)] for n in signals],
  }
  with open(args.out, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
  print(f"{len(out['ways'])} ways, {len(out['signals'])} signals -> {args.out}")

  if not args.lanes:
    return
  if not osm.tagged:
    print(f"no lane tags in {args.osm}: no {args.lanes}")
    return
  cells = defaultdict(list)
  for c in circles.values():
    cells[(int(c[0][0] // CELL), int(c[0][1] // CELL))].append(c)
  lines = []
  for sig, _, nodes in join(layouts):
    pts = points(nodes)
    lo, hi = (pts.min(0) - CELL) // CELL, (pts.max(0) + CELL) // CELL
    near = [c for cx in range(int(lo[0]), int(hi[0]) + 1) for cy in range(int(lo[1]), int(hi[1]) + 1) for c in cells.get((cx, cy), ())]
    for kind, offset, style in sig:
      for k, off in [(0, 0.0)] if kind == EDGE else marks(style, kind == DIVIDER):
        geom = offset_line(pts, offset + off)
        for piece in outside(geom, near):
          lines.append([k] + [round(float(v), 1) for v in piece.ravel()])
  with open(args.lanes, 'w') as f:
    json.dump({'kinds': KINDS, 'lines': lines}, f, separators=(',', ':'))
  print(f"{len(lines)} lines, {len(circles)} crossings -> {args.lanes}")


if __name__ == '__main__':
  main()
