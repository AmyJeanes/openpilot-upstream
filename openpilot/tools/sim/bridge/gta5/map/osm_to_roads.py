#!/usr/bin/env python3
"""Writes the roads of an OSM file as the map view's roads.json: polylines in metres (x east, y north).

`--frame gta5` gives game coordinates (for ynd_to_osm's maps); otherwise metres from the file's centre.
"""
import argparse
import json
import math
from collections import defaultdict

import osmium

from openpilot.tools.sim.bridge.gta5.map.gta5_map import METRES_PER_DEGREE, to_game

ROAD_CLASSES = ['motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'service', 'track']


def join(ways):
  """Joins ways end to end where only two of the same kind meet, so the view draws long lines rather than many short ones."""
  ends = defaultdict(list)  # node -> indices of the ways ending there
  for i, (*_, pts) in enumerate(ways):
    ends[pts[0][0]].append(i)
    ends[pts[-1][0]].append(i)
  done = [False] * len(ways)
  out = []
  for i, (c, lanes, oneway, pts) in enumerate(ways):
    if done[i]:
      continue
    done[i] = True
    line = list(pts)
    for forward in (True, False):
      last = i
      while True:
        node = line[-1][0] if forward else line[0][0]
        if len(ends[node]) != 2:
          break
        j = ends[node][0] if ends[node][1] == last else ends[node][1]
        if done[j] or ways[j][:3] != (c, lanes, oneway):
          break
        nxt = ways[j][3]
        joins = nxt[0][0] == node if forward else nxt[-1][0] == node  # continues in the line's direction
        if not joins:
          if oneway:
            break  # one-way lines keep their direction
          nxt = nxt[::-1]
        done[j] = True
        last = j
        line = line + nxt[1:] if forward else nxt[:-1] + line
    out.append((c, lanes, oneway, line))
  return out


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('osm')
  p.add_argument('out')
  p.add_argument('--frame', choices=['gta5', 'local'], default='gta5')
  args = p.parse_args()

  ways, signals = [], []
  for o in osmium.FileProcessor(args.osm).with_locations():
    if o.is_node() and o.tags.get('highway') == 'traffic_signals':
      signals.append((o.location.lat, o.location.lon))
    elif o.is_way():
      cls = o.tags.get('highway', '').removesuffix('_link')
      if cls not in ROAD_CLASSES:
        continue
      lanes = int(o.tags.get('lanes', '1').split(';')[0] or 1)
      ways.append((ROAD_CLASSES.index(cls), lanes, o.tags.get('oneway') == 'yes', [(n.ref, n.lat, n.lon) for n in o.nodes]))
  ways = [(c, lanes, oneway, [(lat, lon) for _, lat, lon in pts]) for c, lanes, oneway, pts in join(ways)]

  if args.frame == 'gta5':
    def project(lat, lon):
      return to_game(lat, lon)
  else:
    lats = [lat for *_, pts in ways for lat, _ in pts]
    lons = [lon for *_, pts in ways for _, lon in pts]
    lat0, lon0 = (min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2
    def project(lat, lon):
      return (lon - lon0) * METRES_PER_DEGREE * math.cos(math.radians(lat0)), (lat - lat0) * METRES_PER_DEGREE

  out = {
    'classes': ROAD_CLASSES,
    # [class, lanes, oneway, x0, y0, x1, y1, ...]
    'ways': [[c, lanes, int(oneway)] + [round(v, 1) for pt in pts for v in project(*pt)] for c, lanes, oneway, pts in ways],
    'signals': [[round(v, 1) for v in project(*pt)] for pt in signals],
  }
  with open(args.out, 'w') as f:
    json.dump(out, f, separators=(',', ':'))
  print(f"{len(out['ways'])} ways, {len(out['signals'])} signals -> {args.out}")


if __name__ == '__main__':
  main()
