#!/usr/bin/env python3
"""Checks that osm_lanes.py reads the lane tags ynd_to_osm.py writes back to GTA's own lane layout (paths.Link) on every
way, both ways: the same lanes, edges within TOL. Prints the ways that differ by kind (two-way or one-way, offset in
fourteenths of a lane, lanes each way, how they differ).

  python lane_parity.py paths.jsonl gta5.osm.pbf
"""
import argparse
import json
from collections import Counter

import osmium

from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, FORWARD, WayLanes
from openpilot.tools.sim.bridge.gta5.map.paths import Link

TOL = 0.05  # m


def node_id(a, i):  # as ynd_to_osm.node_id
  return a * 65536 + i + 1


def swapped(f):
  """A link record's flags as the record the other way: its lane counts swapped."""
  f = list(f)
  f[2] = (f[2] & ~0xFC) | (((f[2] >> 2) & 7) << 5) | (((f[2] >> 5) & 7) << 2)
  return f


def expected(ab: Link, ba: Link):
  """GTA's lanes a -> b and b -> a, left to right, m right of the link's line seen from a."""
  ours = [(ab.inner + k * ab.width, ab.inner + (k + 1) * ab.width) for k in range(ab.lanes)]
  oncoming = [(-(ba.inner + (k + 1) * ba.width), -(ba.inner + k * ba.width)) for k in reversed(range(ba.lanes))]
  return ours, oncoming


def kind(f):
  steps = ((f[1] >> 4) & 7) * (-1 if f[1] & 128 else 1)
  fwd, back = (f[2] >> 5) & 7, (f[2] >> 2) & 7
  return ('two-way' if fwd and back else 'one-way', steps, fwd, back)


def compare(road: WayLanes, direction: int, ab: Link, ba: Link) -> str | None:
  ours, oncoming = expected(ab, ba)
  got_ours = [(s.left, s.right) for s in road.ours(direction)]
  got_oncoming = [(s.left, s.right) for s in road.oncoming(direction)]
  if road.single_track:
    got_oncoming = got_ours  # its one lane is both ways'
  if (len(got_ours), len(got_oncoming)) != (len(ours), len(oncoming)):
    return f'lanes {len(got_ours)}+{len(got_oncoming)}, GTA {len(ours)}+{len(oncoming)}'
  err = max((abs(x - y) for got, want in ((got_ours, ours), (got_oncoming, oncoming))
             for g, e in zip(got, want, strict=True) for x, y in zip(g, e, strict=True)), default=0.0)
  return f'edges off by up to {err:.2f} m' if err > TOL else None


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('dump', help="ynddump's paths.jsonl")
  p.add_argument('osm', help="ynd_to_osm.py's map")
  args = p.parse_args()

  records, twice = {}, set()  # paths.Paths keeps the last record where GTA has two for a link
  with open(args.dump) as f:
    for line in f:
      d = json.loads(line)
      if d['t'] == 'l':
        key = (node_id(d['a'], d['i']), node_id(d['ta'], d['ti']))
        if key in records and records[key] != d['f']:
          twice.add(key)
        records[key] = d['f']

  checked, bad = 0, Counter()
  for way in osmium.FileProcessor(args.osm, osmium.osm.WAY):
    a, b = way.nodes[0].ref, way.nodes[-1].ref
    if (a, b) not in records and (b, a) not in records:
      bad[('no GTA link', 'way', way.id)] += 1
      continue
    road = WayLanes.from_tags(dict(way.tags))
    for direction, (p, q) in ((FORWARD, (a, b)), (BACKWARD, (b, a))):
      fp = records.get((p, q)) or swapped(records[(q, p)])
      ab, ba = Link(fp), Link(records.get((q, p)) or swapped(fp))
      if not ab.lanes:
        continue
      checked += 1
      if (why := compare(road, direction, ab, ba)) is not None:
        bad[(*kind(fp), why + (', GTA has two differing records' if {(p, q), (q, p)} & twice else ''))] += 1
  print(f"{checked} directed ways checked, {sum(bad.values())} differ from GTA's layout")
  for k, n in sorted(bad.items(), key=lambda kv: -kv[1]):
    print(f"  {n:6d}  {k}")
  return 1 if bad else 0


if __name__ == '__main__':
  raise SystemExit(main())
