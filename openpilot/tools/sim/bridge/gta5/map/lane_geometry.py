#!/usr/bin/env python3
"""The lane geometry fitted to the paint (osm_lanes.GEOMETRY_FILE) baked into the map's .osm.pbf as its own tags.

GEOMETRY_FILE is the reviewable list of changes, keyed by GTA's node ids since the pbf's own ids change with each build:
- `ways`: {"a>b": {"tags": {...}}}, the way from node a to node b (its end nodes) gets these tags over its own (null:
  the tag goes): width:lanes, width and placement:offset from lane_fit.py.
- `diverges`: {"a>b": {"keep": true}}, the way travelled from node a to node b gets osm_lanes.RUN_ON_TAG=no: the branch
  keeps its lanes on its own line there (diverge_report.py --keep).

Baked in, the map is an ordinary OSM map that anything reading its lane tags sees whole; nothing reads GEOMETRY_FILE at
run time. Fit and keep against a map built without it (ynd_to_osm.py without --lanegeom), then bake it in:

  lane_geometry.py MAP/gta5.osm.pbf MAP/lanegeom.json [--out OUT.osm.pbf]
"""
import argparse
import json

from openpilot.tools.sim.bridge.gta5.map.osm_lanes import RUN_ON_NO, RUN_ON_TAG, travelled


def bake(path: str, geometry: dict, out: str | None = None) -> dict:
  """Writes the map at `path` (or to `out`) with `geometry`'s tags on its ways; returns what was and wasn't found."""
  import osmium
  over = {k: v['tags'] for k, v in geometry.get('ways', {}).items()}
  keep = {tuple(k.split('>')) for k, v in geometry.get('diverges', {}).items() if v.get('keep')}
  gta: dict[int, str] = {}
  nodes, ways, rels = [], [], []
  for o in osmium.FileProcessor(path):
    if o.is_node():
      if 'gta:node' in o.tags:
        gta[o.id] = o.tags['gta:node']
      nodes.append(osmium.osm.mutable.Node(id=o.id, version=1, location=(o.location.lon, o.location.lat), tags=dict(o.tags)))
    elif o.is_way():
      ways.append((o.id, [n.ref for n in o.nodes], dict(o.tags)))
    elif o.is_relation():
      rels.append(osmium.osm.mutable.Relation(id=o.id, version=1, tags=dict(o.tags), members=[(m.type, m.ref, m.role) for m in o.members]))
  tagged, kept = set(), set()
  for _, refs, tags in ways:
    key = f'{gta.get(refs[0])}>{gta.get(refs[-1])}'
    if key in over:
      tags.update(over[key])
      for k in [k for k, v in tags.items() if v is None]:
        del tags[k]
      tagged.add(key)
    links = {(gta.get(p), gta.get(q)) for p, q in travelled(tags, refs)} & keep
    if links:
      tags[RUN_ON_TAG] = RUN_ON_NO
      kept |= links
  w = osmium.SimpleWriter(out or path, overwrite=True)
  for n in nodes:
    w.add_node(n)
  for wid, refs, tags in ways:
    w.add_way(osmium.osm.mutable.Way(id=wid, version=1, nodes=refs, tags=tags))
  for r in rels:
    w.add_relation(r)
  w.close()
  return {'ways': len(tagged), 'ways_not_found': sorted(set(over) - tagged),
          'diverges': len(kept), 'diverges_not_found': sorted('>'.join(k) for k in keep - kept)}


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('osm', help='the .osm.pbf, written again in place unless --out')
  p.add_argument('geometry', help='its lane geometry file (lanegeom.json)')
  p.add_argument('--out', help='write the baked map here instead')
  args = p.parse_args()
  with open(args.geometry) as f:
    found = bake(args.osm, json.load(f), args.out)
  print(f"lane geometry baked in: {found['ways']} ways' tags, {found['diverges']} diverges kept "
        f"({len(found['ways_not_found'])} ways and {len(found['diverges_not_found'])} diverges not found)")
  for k in ('ways_not_found', 'diverges_not_found'):
    if found[k]:
      print(f"  {k}: {' '.join(found[k][:20])}")


if __name__ == '__main__':
  main()
