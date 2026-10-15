"""build_pmtiles.py's tiles as comma's unmodified reader and renderer take them (selfdrive/navigation), the local tile
source, and frame.py. Made-up roads near GTA's origin; with GTA5_MAP set (the map's folder, with gta5.osm.pbf), also
the whole map. No pytest needed: `python test_map_frames.py` runs them all."""
import math
import os
import tempfile
from pathlib import Path

import numpy as np

from openpilot.selfdrive.navigation.local_tiles import LocalTiles
from openpilot.selfdrive.navigation.pmtiles import PMTiles
from openpilot.selfdrive.navigation.renderer import MAJOR, MEDIUM, MINOR, STREET, TRACK, EGO, MapRenderer, project, mercator, road_style
from openpilot.selfdrive.navigation.vector_tile import decode_roads
from openpilot.tools.sim.bridge.gta5.navmodel import build_pmtiles as bp
from openpilot.tools.sim.bridge.gta5.navmodel.frame import LAT0, game, lat_lon, tiles_lat0

# GTA's road classes, as the map guesses them (README: Road classes), and the renderer's style for each
STYLES = {"motorway": MAJOR, "trunk": MAJOR, "primary": MAJOR, "secondary": MEDIUM, "tertiary": MEDIUM,
          "motorway_link": STREET, "trunk_link": STREET, "primary_link": STREET, "residential": STREET,
          "unclassified": STREET, "service": MINOR, "track": TRACK, "living_street": STREET}


def archive(roads, lat0: float = 0.0) -> str:
  """A PMTiles file of `roads`: ((x, y) game metres, tags) each."""
  out = []
  for xy, tags in roads:
    lat, lon = lat_lon(np.asarray(xy)[:, 0], np.asarray(xy)[:, 1], lat0)
    out.append((np.column_stack((lon, lat)), bp.road_properties(tags)))
  tiles = {k: bp.encode_tile(v) for k, v in bp.tile_features(out).items()}
  path = tempfile.mktemp(suffix=".pmtiles")
  bp.write_pmtiles(path, tiles, lat0=lat0)
  return path


def test_classes_get_the_renderers_styles():
  for highway, style in STYLES.items():
    path = archive([([(0, 0), (50, 0)], {"highway": highway})])
    a = PMTiles(LocalTiles(path))
    assert (a.min_zoom, a.max_zoom) == (15, 15)
    roads = [r for xy in [(16384, 16383), (16384, 16384), (16383, 16383), (16383, 16384)] if (t := a.get(15, *xy)) for r in decode_roads(t)]
    assert roads and all(road_style(r) == style for r in roads), (highway, [(r.kind, r.detail) for r in roads])
    a.close()
  assert bp.road_properties({"highway": "footway"})["kind"] == "path"  # in the layer, left out by the renderer
  assert bp.road_properties({"highway": "track"}, tracks="path")["kind"] == "path"
  assert bp.road_properties({"building": "yes"}) is None


def test_bridge_flags_and_tile_edges():
  """A bridge keeps is_bridge; a road across a tile edge is in both tiles, overlapping by the buffer."""
  x_edge = 0.0  # tile x 16384 starts at lon 0
  path = archive([([(x_edge - 300, 10), (x_edge + 300, 10)], {"highway": "primary", "bridge": "yes", "layer": "1"})])
  a = PMTiles(LocalTiles(path))
  west, east = decode_roads(a.get(15, 16383, 16383)), decode_roads(a.get(15, 16384, 16383))
  assert all(r.bridge and not r.tunnel for r in west + east)
  assert west[0].points[:, 0].max() > 1.0 and east[0].points[:, 0].min() < 0.0  # past the edge, inside the buffer
  assert west[0].points[:, 0].max() <= 1 + bp.BUFFER / bp.EXTENT + 1e-9


def test_a_bridge_cuts_the_road_under_it():
  """As comma's test_crossings: a ground road along the frame, a bridge across it; the cut is black either side of the
  bridge's stroke, the bridge itself white, and the road beyond the cut white again."""
  path = archive([([(0, -300), (0, 300)], {"highway": "residential"}),
                  ([(-300, 40), (300, 40)], {"highway": "primary", "bridge": "yes"})])
  r = MapRenderer(PMTiles(LocalTiles(path)))
  lat, lon = lat_lon(0.0, 0.0)
  route = np.array([lat_lon(0.0, -100.0)[::-1], lat_lon(0.0, 100.0)[::-1]])
  image, valid = r.render(lon, lat, 0.0, route)
  assert valid
  y_bridge = int(round(EGO[1] - 40))
  half = 19.0 / 2  # MAJOR at the equator's style zoom
  assert image[y_bridge, 120] == 255  # on the bridge, beside the route
  assert image[int(y_bridge + half + 1), 124] == 0  # the cut across the road (off the route), below the bridge's stroke
  assert image[int(y_bridge - half - 1), 124] == 0  # and above it
  assert image[int(y_bridge + half + 6), 124] == 255  # the road beyond the cut


def test_one_game_metre_is_a_pixel_in_either_frame():
  for lat0 in (0.0, 34.0):
    lat, lon = lat_lon(np.array([0.0, 100.0, 0.0]), np.array([0.0, 0.0, 100.0]), lat0)
    px = project(mercator(np.column_stack((lon, lat))), float(lon[0]), float(lat[0]), 0.0)
    assert abs(np.hypot(*(px[1] - px[0])) - 100) < 0.05 and abs(np.hypot(*(px[2] - px[0])) - 100) < 0.05, lat0
    assert px[1][0] > px[0][0] and px[2][1] < px[0][1]  # x east is right, y north is up
    x, y = game(lat, lon, lat0)
    np.testing.assert_allclose(x, [0, 100, 0], atol=1e-6)
    np.testing.assert_allclose(y, [0, 0, 100], atol=1e-6)
  lat, lon = lat_lon(3000.0, 7000.0, 34.0)  # the far side of the map: within 0.2 % still
  near = np.array(lat_lon(np.array([3000.0, 3100.0]), np.array([7000.0, 7000.0]), 34.0)).T
  px = project(mercator(near[:, ::-1]), lon, lat, 0.0)
  assert abs(np.hypot(*(px[1] - px[0])) - 100) < 0.2
  assert math.isclose(lat_lon(1.0, 2.0)[0], lat_lon(1.0, 2.0, 0.0)[0])


def test_tiles_carry_their_frame():
  for lat0 in (0.0, LAT0):
    assert tiles_lat0(archive([([(0, 0), (50, 0)], {"highway": "service"})], lat0)) == lat0


def test_local_tiles_read_and_close():
  path = archive([([(0, 0), (50, 0)], {"highway": "service"})])
  src = LocalTiles(path)
  assert src.size == os.path.getsize(path) and src.read(0, 7) == b"PMTiles"
  src.prefetch([(0, 7)])
  src.close()
  assert src.file.closed


def test_whole_map():
  folder = os.getenv("GTA5_MAP")
  if not folder or not (Path(folder) / "gta5.osm.pbf").exists():
    print("  skipped: GTA5_MAP not set")
    return
  out = tempfile.mktemp(suffix=".pmtiles")
  stats = bp.build(str(Path(folder) / "gta5.osm.pbf"), out)
  assert stats["tiles"] > 40 and stats["bridges"] > 1000, stats
  assert tiles_lat0(out) == LAT0
  r = MapRenderer(PMTiles(LocalTiles(out)))
  lat, lon = lat_lon(215.0, -810.0, LAT0)  # Mission Row
  route = np.array([lat_lon(215.0, -900.0, LAT0)[::-1], lat_lon(215.0, -700.0, LAT0)[::-1]])
  image, valid = r.render(lon, lat, 0.0, route)
  assert valid and (image == 255).mean() > 0.1


if __name__ == "__main__":
  for name, f in list(globals().items()):
    if name.startswith("test_") and callable(f):
      f()
      print("ok", name)
