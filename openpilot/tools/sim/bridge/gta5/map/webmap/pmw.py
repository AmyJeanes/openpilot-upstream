"""Tile sink writing PMTiles and/or a z/x/y directory."""
import math
import os

from pmtiles.tile import Compression, TileType, zxy_to_tileid
from pmtiles.writer import Writer


class Sink:
    def __init__(self, pm_path=None, dir_path=None, tile_type=TileType.WEBP, compression=Compression.NONE, ext="webp"):
        self.pm_path, self.dir_path, self.tile_type, self.comp, self.ext = pm_path, dir_path, tile_type, compression, ext
        self.tiles = {}   # (z,x,y)->bytes; prototype-size only (a full build would spool to disk)

    def add(self, z, x, y, data):
        self.tiles[(z, x, y)] = data

    def finish(self, metadata, center_zoom=None):
        zs = sorted({k[0] for k in self.tiles})

        def lon(x, z):
            return x / (1 << z) * 360 - 180

        def lat(y, z):
            return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / (1 << z)))))

        zt = zs[-1]
        xs = [k[1] for k in self.tiles if k[0] == zt]
        ys = [k[2] for k in self.tiles if k[0] == zt]
        b = (lon(min(xs), zt), lat(max(ys) + 1, zt), lon(max(xs) + 1, zt), lat(min(ys), zt))
        if self.pm_path:
            with open(self.pm_path, "wb") as f:
                w = Writer(f)
                for k, d in sorted(self.tiles.items(), key=lambda kv: zxy_to_tileid(*kv[0])):
                    w.write_tile(zxy_to_tileid(*k), d)
                cz = center_zoom or zs[len(zs) // 2]
                w.finalize({"tile_type": self.tile_type, "tile_compression": self.comp,
                            "min_zoom": zs[0], "max_zoom": zs[-1],
                            "min_lon_e7": int(b[0] * 1e7), "min_lat_e7": int(b[1] * 1e7),
                            "max_lon_e7": int(b[2] * 1e7), "max_lat_e7": int(b[3] * 1e7),
                            "center_zoom": cz, "center_lon_e7": int((b[0] + b[2]) / 2 * 1e7),
                            "center_lat_e7": int((b[1] + b[3]) / 2 * 1e7)}, metadata)
        if self.dir_path:
            for (z, x, y), d in self.tiles.items():
                p = os.path.join(self.dir_path, str(z), str(x))
                os.makedirs(p, exist_ok=True)
                with open(os.path.join(p, f"{y}.{self.ext}"), "wb") as f:
                    f.write(d)
        return b
