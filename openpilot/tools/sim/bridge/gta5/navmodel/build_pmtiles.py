#!/usr/bin/env python3
"""GTA's roads as a local PMTiles file of Protomaps vector tiles, the form comma's nav map renderer reads
(selfdrive/navigation/{pmtiles,vector_tile,renderer}.py, comma nav-model-integration afbf2e5), from the map's
gta5.osm.pbf:

  python build_pmtiles.py ~/gta5map/gta5.osm.pbf ~/gta5map/gta5_roads.pmtiles [--lat0 0]

What the renderer reads, as Protomaps' basemap gives it: layer `roads`, line features, gzip MVT tiles at zoom 15
(extent 4096, a 64-unit buffer) in a gzip PMTiles v3 archive, with
- kind: highway (motorway, motorway_link), major_road (trunk, primary, secondary, tertiary and their links),
  minor_road (residential, unclassified, road, service, track), other (living_street, pedestrian, raceway, ...),
  path (footway, path, cycleway, steps, bridleway), which the renderer leaves out;
- kind_detail: the OSM highway (for service roads, their service=*, when tagged);
- is_bridge / is_tunnel: booleans, only where true; level: 1 on bridges, -1 in tunnels;
- name and oneway, as Protomaps has them (the renderer doesn't read them).
Tracks are minor roads here, so the renderer draws them in its track style (`--tracks path` leaves them out, as a
basemap that files tracks under paths would).

The roads are taken where the pbf has them, in the bridge's frame (map/gta5_map.py), or with --lat0 moved into the
frame about that latitude (frame.py), for the renderer's road widths there.
"""
import argparse
import gzip
import math
from collections import defaultdict

import numpy as np

ZOOM = 15
EXTENT = 4096
BUFFER = 64  # tile units each side, as Protomaps (4 px of a 256 px tile)
KINDS = {
  "motorway": "highway", "motorway_link": "highway",
  "trunk": "major_road", "trunk_link": "major_road", "primary": "major_road", "primary_link": "major_road",
  "secondary": "major_road", "secondary_link": "major_road", "tertiary": "major_road", "tertiary_link": "major_road",
  "residential": "minor_road", "unclassified": "minor_road", "road": "minor_road", "service": "minor_road",
  "track": "minor_road",
  "living_street": "other", "pedestrian": "other", "raceway": "other", "busway": "other", "bus_guideway": "other",
  "footway": "path", "path": "path", "cycleway": "path", "steps": "path", "bridleway": "path", "corridor": "path",
}


def mercator(lon, lat) -> np.ndarray:
  """Normalized Web Mercator, x east / y south, as the renderer's."""
  y = (1.0 - np.arcsinh(np.tan(np.deg2rad(lat))) / np.pi) / 2.0
  return np.column_stack(((np.asarray(lon) + 180.0) / 360.0, y))


def road_properties(tags: dict, tracks: str = "minor_road") -> dict | None:
  """A way's Protomaps roads properties, or None for one that isn't a road."""
  highway = tags.get("highway")
  kind = KINDS.get(highway) if highway else None
  if kind is None:
    return None
  if highway == "track":
    kind = tracks
  props = {"kind": kind, "kind_detail": tags.get("service", "service") if highway == "service" else highway}
  if tags.get("bridge", "no") not in ("no", ""):
    props.update(is_bridge=True, level=1)
  elif tags.get("tunnel", "no") not in ("no", ""):
    props.update(is_tunnel=True, level=-1)
  if tags.get("name"):
    props["name"] = tags["name"]
  if tags.get("oneway") in ("yes", "1", "-1"):
    props["oneway"] = "yes"
  return props


def read_roads(pbf: str, lat0: float = 0.0, tracks: str = "minor_road") -> list[tuple[np.ndarray, dict]]:
  """The pbf's roads: each one's (lon, lat) points (in the frame about lat0) and its properties."""
  import osmium

  from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
  from openpilot.tools.sim.bridge.gta5.navmodel.frame import lat_lon

  roads = []
  for w in osmium.FileProcessor(pbf, osmium.osm.NODE | osmium.osm.WAY).with_locations():
    if not w.is_way():
      continue
    props = road_properties(dict(w.tags), tracks)
    if props is None:
      continue
    pts = np.array([(n.lon, n.lat) for n in w.nodes if n.location.valid()], dtype=np.float64)
    if len(pts) < 2:
      continue
    if lat0 != 0.0:
      x, y = to_game(pts[:, 1], pts[:, 0])
      lat, lon = lat_lon(x, y, lat0)
      pts = np.column_stack((lon, lat))
    roads.append((pts, props))
  return roads


def tile_features(roads: list[tuple[np.ndarray, dict]], zoom: int = ZOOM) -> dict[tuple[int, int], list[dict]]:
  """The roads cut into tiles: per (x, y), MVT features in tile units (y down), clipped BUFFER outside the tile."""
  from shapely.geometry import LineString, box

  clip = box(-BUFFER, -BUFFER, EXTENT + BUFFER, EXTENT + BUFFER)
  count = 1 << zoom
  tiles: dict[tuple[int, int], list[dict]] = defaultdict(list)
  pad = BUFFER / EXTENT
  for pts, props in roads:
    t = mercator(pts[:, 0], pts[:, 1]) * count  # tiles
    lo, hi = np.floor(t.min(axis=0) - pad).astype(int), np.floor(t.max(axis=0) + pad).astype(int)
    for tx in range(lo[0], hi[0] + 1):
      for ty in range(lo[1], hi[1] + 1):
        local = (t - (tx, ty)) * EXTENT
        line = LineString(local).intersection(clip)
        for part in getattr(line, "geoms", [line]):
          if part.geom_type == "LineString" and len(part.coords) >= 2:
            tiles[(tx % count, ty)].append({"geometry": part, "properties": props})
  return tiles


def encode_tile(features: list[dict]) -> bytes:
  """One gzip MVT tile with the `roads` layer."""
  import mapbox_vector_tile

  data = mapbox_vector_tile.encode([{"name": "roads", "features": features}],
                                   default_options={"extents": EXTENT, "y_coord_down": True})
  return gzip.compress(data, mtime=0)


def write_pmtiles(path: str, tiles: dict[tuple[int, int], bytes], zoom: int = ZOOM, attribution: str = "") -> None:
  from pmtiles.tile import Compression, TileType, zxy_to_tileid
  from pmtiles.writer import Writer

  count = 1 << zoom
  xs, ys = [x for x, _ in tiles], [y for _, y in tiles]

  def lon(x):
    return x / count * 360.0 - 180.0

  def lat(y):
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / count))))

  with open(path, "wb") as f:
    w = Writer(f)
    for (x, y), data in sorted(tiles.items(), key=lambda kv: zxy_to_tileid(zoom, *kv[0])):
      w.write_tile(zxy_to_tileid(zoom, x, y), data)
    bounds = (lon(min(xs)), lat(max(ys) + 1), lon(max(xs) + 1), lat(min(ys)))
    w.finalize({
      "tile_type": TileType.MVT, "tile_compression": Compression.GZIP,
      "min_lon_e7": int(bounds[0] * 1e7), "min_lat_e7": int(bounds[1] * 1e7),
      "max_lon_e7": int(bounds[2] * 1e7), "max_lat_e7": int(bounds[3] * 1e7),
      "center_zoom": zoom, "center_lon_e7": int((bounds[0] + bounds[2]) / 2 * 1e7),
      "center_lat_e7": int((bounds[1] + bounds[3]) / 2 * 1e7),
    }, {"name": "gta5 roads", "type": "baselayer", "format": "pbf", "attribution": attribution,
        "vector_layers": [{"id": "roads", "fields": {"kind": "String", "kind_detail": "String", "is_bridge": "Boolean",
                                                     "is_tunnel": "Boolean", "level": "Number", "name": "String",
                                                     "oneway": "String"}, "minzoom": zoom, "maxzoom": zoom}]})


def build(pbf: str, out: str, lat0: float = 0.0, tracks: str = "minor_road") -> dict:
  roads = read_roads(pbf, lat0, tracks)
  tiles = {xy: encode_tile(feats) for xy, feats in tile_features(roads).items()}
  write_pmtiles(out, tiles, attribution=f"GTA V roads from {pbf.rsplit('/', 1)[-1]}, frame lat0={lat0}")
  kinds: dict[str, int] = defaultdict(int)
  for _, p in roads:
    kinds[p["kind"]] += 1
  return {"roads": len(roads), "bridges": sum(bool(p.get("is_bridge")) for _, p in roads), "tiles": len(tiles),
          "bytes": sum(len(t) for t in tiles.values()), "kinds": dict(kinds)}


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("pbf")
  ap.add_argument("out")
  ap.add_argument("--lat0", type=float, default=0.0, help="frame latitude (frame.py); 0: the bridge's own")
  ap.add_argument("--tracks", choices=("minor_road", "path"), default="minor_road", help="tracks' kind (path: not drawn)")
  args = ap.parse_args()
  print(build(args.pbf, args.out, args.lat0, args.tracks))


if __name__ == "__main__":
  main()
