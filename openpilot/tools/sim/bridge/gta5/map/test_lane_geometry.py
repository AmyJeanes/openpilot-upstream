"""lane_geometry.py's bake on a made map. No pytest needed: `python test_lane_geometry.py`."""
import os
import tempfile

from openpilot.tools.sim.bridge.gta5.map.osm_lanes import RUN_ON_NO, RUN_ON_TAG

NODES = {101: ('1:1', 0.0), 102: ('1:2', 60.0), 103: ('1:3', 120.0), 104: ('2:7', 180.0), 105: ('1:5', 240.0)}
WAYS = {11: ([101, 102], {'highway': 'motorway', 'lanes': '2', 'oneway': 'yes', 'width': '11', 'placement': 'transition'}),
        12: ([102, 103, 104], {'highway': 'motorway_link', 'lanes': '1', 'oneway': 'yes'}),
        13: ([104, 105], {'highway': 'trunk', 'lanes': '2'})}


def made(d: str) -> str:
  import osmium
  from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon
  path = os.path.join(d, 'gta5.osm.pbf')
  w = osmium.SimpleWriter(path)
  for n, (gta, y) in NODES.items():
    lat, lon = to_lat_lon(0.0, y)
    w.add_node(osmium.osm.mutable.Node(id=n, version=1, location=(lon, lat), tags={'gta:node': gta}))
  for wid, (refs, tags) in WAYS.items():
    w.add_way(osmium.osm.mutable.Way(id=wid, version=1, nodes=refs, tags=tags))
  w.close()
  return path


def test_ways_tags_and_kept_diverges_baked_in():
  # a way's tags over its own by its end nodes' GTA ids (null takes one away), a kept diverge's link tagged on the way
  # travelled along it (a two-way way either way, a one-way only its own), and keys the map hasn't; the map read back is
  # an ordinary OSM map that says all that by its tags
  try:
    import osmium  # noqa: F401
  except ImportError:
    print('test_ways_tags_and_kept_diverges_baked_in skipped: no pyosmium')
    return
  from openpilot.tools.sim.bridge.gta5.map import lane_geometry, osm_pbf
  from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
  from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes
  geometry = {'ways': {'1:1>1:2': {'tags': {'width:lanes': '5.0|6.0', 'placement:offset': '0.3', 'placement': None}},
                       '9:9>9:8': {'tags': {'width': '7'}}},
              'diverges': {'1:2>1:3': {'keep': True}, '1:3>2:7': {'keep': False}, '1:5>2:7': {'keep': True},
                           '2:7>1:3': {'keep': True}, '5:5>5:6': {'keep': True}}}
  with tempfile.TemporaryDirectory() as d:
    path = made(d)
    found = lane_geometry.bake(path, geometry)
    assert found == {'ways': 1, 'ways_not_found': ['9:9>9:8'], 'diverges': 2, 'diverges_not_found': ['2:7>1:3', '5:5>5:6']}, found
    data = osm_pbf.read(path)
    osm = OsmLanes.load(path, to_game)
  t11, t12, t13 = data.ways[11][0], data.ways[12][0], data.ways[13][0]
  assert t11['width:lanes'] == '5.0|6.0' and t11['placement:offset'] == '0.3' and 'placement' not in t11, t11
  assert t11['width'] == '11' and RUN_ON_TAG not in t11
  assert t12.get(RUN_ON_TAG) == RUN_ON_NO and t13.get(RUN_ON_TAG) == RUN_ON_NO
  assert osm.kept_diverge(102, 103) and not osm.kept_diverge(101, 102) and not osm.kept_diverge(103, 102)
  assert osm.kept_diverge(105, 104) and osm.kept_diverge(104, 105)


if __name__ == "__main__":
  for name, fn in sorted(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"ok {name}")
