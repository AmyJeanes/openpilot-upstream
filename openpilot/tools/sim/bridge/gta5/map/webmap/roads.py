"""Roads from the pbf -> multi-zoom Protomaps-style `roads` MVT in the lat0 frame (wraps navmodel/build_pmtiles.py logic)."""
import argparse
import gzip
import math
import os
import sys
from collections import defaultdict

import mapbox_vector_tile
import numpy as np
import osmium
from pmtiles.tile import Compression, TileType
from shapely.geometry import LineString, box

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gframe  # noqa: E402
import pmw  # noqa: E402

METRES_PER_DEGREE = 6378137.0 * math.pi / 180   # gta5_map.to_game
EXTENT, BUFFER = 4096, 64
KINDS = {"motorway": "highway", "motorway_link": "highway",
         "trunk": "major_road", "trunk_link": "major_road", "primary": "major_road", "primary_link": "major_road",
         "secondary": "major_road", "secondary_link": "major_road", "tertiary": "major_road", "tertiary_link": "major_road",
         "residential": "minor_road", "unclassified": "minor_road", "road": "minor_road", "service": "minor_road",
         "track": "minor_road",
         "living_street": "other", "pedestrian": "other", "raceway": "other", "busway": "other", "bus_guideway": "other",
         "footway": "path", "path": "path", "cycleway": "path", "steps": "path", "bridleway": "path", "corridor": "path"}
MINZ = {"highway": 12, "major_road": 12, "minor_road": 13, "other": 15, "path": 99}


def props_of(tags):
    hw = tags.get("highway")
    kind = KINDS.get(hw) if hw else None
    if kind is None:
        return None
    p = {"kind": kind, "kind_detail": tags.get("service", "service") if hw == "service" else hw}
    if tags.get("bridge", "no") not in ("no", ""):
        p.update(is_bridge=True, level=1)
    elif tags.get("tunnel", "no") not in ("no", ""):
        p.update(is_tunnel=True, level=-1)
    if tags.get("name"):
        p["name"] = tags["name"]
    if tags.get("oneway") in ("yes", "1", "-1"):
        p["oneway"] = "yes"
    return p


def read_roads(pbf):
    roads = []
    for w in osmium.FileProcessor(pbf, osmium.osm.NODE | osmium.osm.WAY).with_locations():
        if not w.is_way():
            continue
        p = props_of(dict(w.tags))
        if p is None:
            continue
        pts = np.array([(n.lon, n.lat) for n in w.nodes if n.location.valid()], dtype=np.float64)
        if len(pts) < 2:
            continue
        x, y = pts[:, 0] * METRES_PER_DEGREE, pts[:, 1] * METRES_PER_DEGREE   # GTA metres
        mx, my = gframe.merc(x, y)
        roads.append((np.column_stack((mx, my)), p))
    return roads


def build(pbf, pm_path, dir_path, zmin, zmax):
    roads = read_roads(pbf)
    print("roads", len(roads), flush=True)
    sink = pmw.Sink(pm_path, dir_path, TileType.MVT, Compression.GZIP, "mvt")
    clip = box(-BUFFER, -BUFFER, EXTENT + BUFFER, EXTENT + BUFFER)
    for z in range(zmin, zmax + 1):
        n = 1 << z
        tiles = defaultdict(list)
        pad = BUFFER / EXTENT
        for m, p in roads:
            if MINZ[p["kind"]] > z:
                continue
            t = m * n
            lo, hi = np.floor(t.min(0) - pad).astype(int), np.floor(t.max(0) + pad).astype(int)
            for tx in range(lo[0], hi[0] + 1):
                for ty in range(lo[1], hi[1] + 1):
                    ls = LineString((t - (tx, ty)) * EXTENT)
                    if z < 15:
                        ls = ls.simplify(2.0 * (16 - z) / 4)
                    g = ls.intersection(clip)
                    for part in getattr(g, "geoms", [g]):
                        if part.geom_type == "LineString" and len(part.coords) >= 2:
                            tiles[(tx, ty)].append({"geometry": part, "properties": p})
        for (x, y), f in tiles.items():
            d = mapbox_vector_tile.encode([{"name": "roads", "features": f}],
                                          default_options={"extents": EXTENT, "y_coord_down": True})
            sink.add(z, x, y, gzip.compress(d, mtime=0))
        print("roads z", z, len(tiles), "tiles", flush=True)
    md = {"name": "gta5 roads", "format": "pbf", "lat0": gframe.LAT0,
          "vector_layers": [{"id": "roads", "fields": {"kind": "String", "kind_detail": "String", "is_bridge": "Boolean",
                                                       "is_tunnel": "Boolean", "level": "Number", "name": "String",
                                                       "oneway": "String"}, "minzoom": zmin, "maxzoom": zmax}]}
    b = sink.finish(md)
    print("bounds", b, "bytes", sum(len(v) for v in sink.tiles.values()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pbf")
    ap.add_argument("--pmtiles")
    ap.add_argument("--dir")
    ap.add_argument("--zmin", type=int, default=12)
    ap.add_argument("--zmax", type=int, default=16)
    a = ap.parse_args()
    build(a.pbf, a.pmtiles, a.dir, a.zmin, a.zmax)
