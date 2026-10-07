#!/usr/bin/env python3
"""Checks that osm_lanes.py reads the lane tags ynd_to_osm.py writes back to the layout of GTA's links as painted
(ynd_to_osm.layout) on every way, both ways: the same lanes, edges within TOL. Prints the ways that differ by kind
(two-way or one-way, offset in steps of GTA's field, lanes each way, how they differ).

  python lane_parity.py paths.jsonl gta5.osm.pbf
"""
import argparse
import json
from collections import Counter

import osmium

from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, FORWARD, WayLanes
from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import FREEWAY, layout

TOL = 0.05  # m


def node_id(a, i):  # as ynd_to_osm.node_id
  return a * 65536 + i + 1


def swapped(f):
  """A link record's flags as the record the other way: its lane counts swapped."""
  f = list(f)
  f[2] = (f[2] & ~0xFC) | (((f[2] >> 2) & 7) << 5) | (((f[2] >> 5) & 7) << 2)
  return f


def lanes(f) -> tuple[int, int]:
  """A link record's lanes its way and back."""
  return (f[2] >> 5) & 7, (f[2] >> 2) & 7


def expected(ab, ba, freeway: bool = False):
  """The lanes a -> b and b -> a (the link records' flags), left to right, m right of the link's line seen from a; on a
  `freeway` link, wider where one-way."""
  def run(f):  # one direction's lanes: how many, how wide, m right of the line to their left edge
    n, back = lanes(f)
    w, offset = layout(f, back, freeway and n >= 2)
    return n, w, offset if back else offset - n * w / 2
  n, w, inner = run(ab)
  m, v, inner_back = run(ba)
  ours = [(inner + k * w, inner + (k + 1) * w) for k in range(n)]
  oncoming = [(-(inner_back + (k + 1) * v), -(inner_back + k * v)) for k in reversed(range(m))]
  return ours, oncoming


def kind(f):
  steps = ((f[1] >> 4) & 7) * (-1 if f[1] & 128 else 1)
  fwd, back = lanes(f)
  return ('two-way' if fwd and back else 'one-way', steps, fwd, back)


def compare(road: WayLanes, direction: int, ab, ba, freeway: bool = False) -> str | None:
  ours, oncoming = expected(ab, ba, freeway)
  got_ours = [(s.left, s.right) for s in road.ours(direction)]
  got_oncoming = [(s.left, s.right) for s in road.oncoming(direction)]
  if road.single_track:
    got_oncoming = got_ours  # its one lane is both ways'
  # a turn bay folded into the road (ynd_to_osm.detached_bays): one more lane on the inside of a direction, beside its
  # lanes as GTA has them, in a two-way road's median or left of a one-way road's
  if ours and len(got_ours) == len(ours) + 1 and abs(got_ours[0][1] - ours[0][0]) <= TOL:
    got_ours = got_ours[1:]
  if oncoming and len(got_oncoming) == len(oncoming) + 1 and abs(got_oncoming[-1][0] - oncoming[-1][1]) <= TOL:
    got_oncoming = got_oncoming[:-1]
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

  records, twice, freeway = {}, set(), set()  # paths.Paths keeps the last record where GTA has two for a link
  with open(args.dump) as f:
    for line in f:
      d = json.loads(line)
      if d['t'] == 'n' and d['f'][2] & FREEWAY:
        freeway.add(node_id(d['a'], d['i']))
      elif d['t'] == 'l':
        key = (node_id(d['a'], d['i']), node_id(d['ta'], d['ti']))
        if key in records and records[key] != d['f']:
          twice.add(key)
        records[key] = d['f']

  checked, surveyed, bad = 0, 0, Counter()
  for way in osmium.FileProcessor(args.osm, osmium.osm.WAY):
    a, b = way.nodes[0].ref, way.nodes[-1].ref
    if (a, b) not in records and (b, a) not in records:
      bad[('no GTA link', 'way', way.id)] += 1
      continue
    if way.tags.get('source:width') == 'survey':  # measured paint, not the layout
      surveyed += 1
      continue
    road = WayLanes.from_tags(dict(way.tags))
    for direction, (p, q) in ((FORWARD, (a, b)), (BACKWARD, (b, a))):
      fp = records.get((p, q)) or swapped(records[(q, p)])
      if not lanes(fp)[0]:
        continue
      checked += 1
      if (why := compare(road, direction, fp, records.get((q, p)) or swapped(fp), {p, q} <= freeway)) is not None:
        bad[(*kind(fp), why + (', GTA has two differing records' if {(p, q), (q, p)} & twice else ''))] += 1
  print(f"{checked} directed ways checked ({surveyed} ways with surveyed widths left out), {sum(bad.values())} differ from GTA's links' layout")
  for k, n in sorted(bad.items(), key=lambda kv: -kv[1]):
    print(f"  {n:6d}  {k}")
  return 1 if bad else 0


if __name__ == '__main__':
  raise SystemExit(main())
