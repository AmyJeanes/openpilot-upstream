"""Where the nav model's map frame puts GTA: game metres to the (lat, lon) the map tiles, the route and the car's fix are
given in for comma's renderer (selfdrive/navigation/renderer.py), and back.

lat0 = 0 is the bridge's own frame (map/gta5_map.py: about (0, 0), on the equator), the one its Valhalla, navRoute and
GNSS use. The renderer draws roads by a style zoom that depends on the latitude, wider at the equator than where comma's
cars drive (about 10 % wider than at 34 N), so the map frame defaults to Los Angeles' latitude: Web Mercator about
(LAT0, 0) scaled so that a game metre is still a metre (a pixel) there. Every input of one renderer must be in the
same frame: build_pmtiles.py stores its lat0 in the tiles' metadata, and the route and the fix take it from there
(tiles_lat0).
"""
import json
import math

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game, to_lat_lon

EARTH_CIRCUMFERENCE = 40075016.68557849  # equatorial, as the renderer's Web Mercator
LAT0 = 34.0  # Los Angeles, GTA's Los Santos


def tiles_lat0(path: str) -> float:
  """The frame latitude a build_pmtiles.py archive was built in (0 for archives from before it was stored)."""
  from pmtiles.reader import MmapSource, Reader
  with open(path, "rb") as f:
    metadata = Reader(MmapSource(f)).metadata()
  if isinstance(metadata, (str, bytes)):
    metadata = json.loads(metadata)
  return float(metadata.get("lat0", 0.0))


def lat_lon(x, y, lat0: float = 0.0):
  """(lat, lon) degrees of game metres (x east, y north), arrays or floats."""
  if lat0 == 0.0:
    return to_lat_lon(x, y)
  k = EARTH_CIRCUMFERENCE * math.cos(math.radians(lat0))  # metres per unit of normalized mercator at lat0
  my0 = (1.0 - math.asinh(math.tan(math.radians(lat0))) / math.pi) / 2.0
  mx, my = 0.5 + np.asarray(x, dtype=np.float64) / k, my0 - np.asarray(y, dtype=np.float64) / k
  lon = mx * 360.0 - 180.0
  lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * my))))
  return (float(lat), float(lon)) if np.ndim(lat) == 0 else (lat, lon)


def game(lat, lon, lat0: float = 0.0):
  """Game metres (x, y) of (lat, lon) in the frame about lat0."""
  if lat0 == 0.0:
    return to_game(lat, lon)
  k = EARTH_CIRCUMFERENCE * math.cos(math.radians(lat0))
  my0 = (1.0 - math.asinh(math.tan(math.radians(lat0))) / math.pi) / 2.0
  mx = (np.asarray(lon, dtype=np.float64) + 180.0) / 360.0
  my = (1.0 - np.arcsinh(np.tan(np.radians(np.asarray(lat, dtype=np.float64)))) / np.pi) / 2.0
  x, y = (mx - 0.5) * k, (my0 - my) * k
  return (float(x), float(y)) if np.ndim(x) == 0 else (x, y)


def projector(lat0: float = 0.0):
  """lat_lon for one frame, as nav_input.ModelNav's `project`."""
  return lambda x, y: lat_lon(x, y, lat0)
