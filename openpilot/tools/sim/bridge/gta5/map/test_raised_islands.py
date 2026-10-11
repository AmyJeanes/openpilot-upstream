"""raised_islands.py's raised islands from roadpaint's tiles made here, under a road's lanes. No pytest needed: `python
test_raised_islands.py` runs them all."""
import os
import struct
import tempfile
import zlib

import numpy as np

from openpilot.tools.sim.bridge.gta5.map import raised_islands

ROAD, KERB, WALK = 1, 2, 3


def write_surface(d, code_z):
  """Tile 0_0, one surface: code_z(x, y) gives each pixel's (code, height)."""
  n = 1280
  xs = (np.arange(n) + 0.5) * 0.05
  x, y = np.meshgrid(xs, 64.0 - xs)
  code, z = code_z(x, y)
  body = struct.pack('<iiiifi', 0, 0, 0, n, 0.05, 1)
  for arr in (code.astype(np.uint8)[None], z.astype(np.float32)[None]):
    c = zlib.compress(arr.tobytes())
    body += struct.pack('<i', len(c)) + c
  with open(os.path.join(d, '0_0.bin'), 'wb') as f:
    f.write(body)


def surface(x, y):
  """Road at 0 m along y = 32, with (a) a kerbed island 0.15 m up on its right lane at x = 15, (b) a flat pavement-class
  patch on its left lane at x = 30 (a concrete road), (c) a kerbed strip 0.15 m up across its right lane at x = 46
  running on north out of the tile (a pavement)."""
  code, z = np.full(x.shape, ROAD), np.zeros(x.shape)
  r = np.hypot(x - 15.0, y - 29.25)
  island = r < 2.0
  patch = (abs(x - 30.0) < 1.5) & (abs(y - 34.75) < 1.5)
  strip = (abs(x - 46.0) < 1.2) & (y > 27.0)
  code[island | strip], z[island | strip] = WALK, 0.15
  code[(island & (r > 1.7)) | (strip & (abs(x - 46.0) > 0.9))] = KERB
  code[patch] = WALK
  return code, z


def lanes(highway='residential'):
  from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game, to_lat_lon
  from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes
  from openpilot.tools.sim.bridge.gta5.map.osm_pbf import OsmData
  ll = np.array([to_lat_lon(x, 32.0) for x in (2.0, 62.0)])
  return OsmLanes(OsmData(np.array([1, 2]), ll[:, 0], ll[:, 1], {1: {'ele': '0'}, 2: {'ele': '0'}},
                          {1: ({'highway': highway, 'lanes': '2', 'width': '11'}, [1, 2])}), to_game)


def test_kerbed_island_only():
  with tempfile.TemporaryDirectory() as d:
    write_surface(d, surface)
    stats = {}
    found = raised_islands.islands(lanes(), d, stats=stats)
  assert len(found) == 1, (found, stats)
  f = found[0]
  assert np.hypot(*(f.centre - (15.0, 29.25))) < 0.3 and 10.0 < f.area < 14.0 and abs(f.dz - 0.15) < 0.02
  assert f.centre_hit and f.ways == {1: 'residential'} and np.allclose(f.ring[0], f.ring[-1])
  assert stats['open'] >= 1 and stats['no kerb or level'] >= 1  # the pavement strip, the concrete patch


def test_service_roads_island_left_out():
  with tempfile.TemporaryDirectory() as d:
    write_surface(d, surface)
    stats = {}
    assert raised_islands.islands(lanes('service'), d, stats=stats) == [] and stats['service roads only'] == 1


def test_kerb_round_a_bed():
  # a kerb ring 0.2-0.4 m wide round a bed with no surface (grass), with a gap in it: the island is the whole bed
  def bed(x, y):
    code, z = np.full(x.shape, ROAD), np.zeros(x.shape)
    r = np.hypot((x - 20.0) / 2.0, y - 29.25)  # an oval 8 m by 4 m
    kerb = (r > 1.8) & (r < 2.0) & ~((abs(y - 29.25) < 0.3) & (x > 20.0))
    code[r <= 1.8], code[kerb], z[kerb] = 0, KERB, 0.15
    return code, z
  with tempfile.TemporaryDirectory() as d:
    write_surface(d, bed)
    rp, tiles = raised_islands._tiles(d)
    b = raised_islands.blob(tiles, rp, 20.0, 31.25, 0.0)
  assert b is not None and 20.0 < b[1] < 28.0  # pi * 4 * 2 = 25 m^2


def test_areas_written():
  try:
    import osmium
  except ImportError:
    print('test_areas_written skipped: no pyosmium')
    return
  from openpilot.tools.sim.bridge.gta5.map import osm_pbf, painted_islands
  from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'map.osm.pbf')
    w = osmium.SimpleWriter(path)
    for n, (x, y) in {1: (0.0, 0.0), 2: (100.0, 0.0)}.items():
      lat, lon = to_lat_lon(x, y)
      w.add_node(osmium.osm.mutable.Node(id=n, version=1, location=(lon, lat), tags={}))
    w.add_way(osmium.osm.mutable.Way(id=1, version=1, nodes=[1, 2], tags={'highway': 'residential'}))
    w.close()
    ring = np.array([[10.0, 1.0, 5.0], [14.0, 1.0, 5.0], [12.0, 4.0, 5.0], [10.0, 1.0, 5.0]])
    island = raised_islands.Raised(ring, 6.0, 0.15, ring[:-1, :2].mean(0), {1: 'residential'}, True)
    assert painted_islands.add(path, [], to_lat_lon, (), raised_islands.areas([island])) == 1
    data = osm_pbf.read(path)
  (tags, refs), = [v for wid, v in data.ways.items() if wid != 1]
  assert tags == {'area': 'yes', 'traffic_calming': 'island'} and len(refs) == 4 and refs[0] == refs[-1]
  assert all(data.node_tags[n]['ele'] == '5.0' for n in refs)


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
