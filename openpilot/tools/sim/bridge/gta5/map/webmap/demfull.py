"""Elevation tiles for the web map's 3D view: every cell's height tile (the top surface: ground, roofs, trees, the water
surface) averaged to z18 (~0.5 m/px) in the lat0 Web Mercator frame, painted open-water cells at their body's surface
height, gaps inside the captures filled from the nearest height; terrarium-encoded 256 px PNG tiles z18..zmin as
dem.pmtiles (MapLibre raster-dem, encoding "terrarium").

  demfull.py <out dir> [--zmax 18] [--zmin 10]
"""
import argparse
import io
import json
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gframe  # noqa: E402

F = "/mnt/e/gta5_ortho_20261018"
FILL = f"{F}/full_water/fill"
CW, CH, CX0, CY0 = 2304, 1296, 128, 72
PPM = 100.0 / 3.0
SURFACE = {"sea": 0.0, "lago_zancudo": 0.0, "alamo": 30.0, "vinewood_res": 196.409, "land_act": 160.339}  # water.xml heights


def centre(i, j):
  return -4300 + 69.12 * i, -4300 + 38.88 * j


def terrarium(h):
  v = np.clip(h, -32768, 32767) + 32768.0
  r = np.floor(v / 256)
  g = np.floor(v) - r * 256
  b = np.floor((v - np.floor(v)) * 256)
  return np.dstack((r, g, b)).astype(np.uint8)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--zmax", type=int, default=18)
  ap.add_argument("--zmin", type=int, default=10)
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)
  S = 256 * (1 << a.zmax)
  r = gframe.px_per_m(a.zmax) / PPM  # canvas px per capture px
  cells = sorted((int(f[1:4]), int(f[5:8])) for f in os.listdir(f"{F}/depth/tiles") if f.endswith("_h.png"))
  fill = json.load(open(f"{FILL}/cells.json"))["filled"] if os.path.exists(f"{FILL}/cells.json") else []
  allc = [(i, j) for i, j in cells] + [tuple(c["cell"]) for c in fill]
  # canvas: every cell's z{zmax} rect, on whole tiles
  def rect(i, j):
    xc, yc = centre(i, j)
    mx0, my0 = gframe.merc(xc - CW / PPM / 2, yc + CH / PPM / 2)
    mx1, my1 = gframe.merc(xc + CW / PPM / 2, yc - CH / PPM / 2)
    return float(mx0) * S, float(my0) * S, float(mx1) * S, float(my1) * S
  rs = [rect(i, j) for i, j in allc]
  step = 256
  X0 = int(min(q[0] for q in rs) // step * step)
  Y0 = int(min(q[1] for q in rs) // step * step)
  W = int(math.ceil((max(q[2] for q in rs) - X0) / step) * step)
  H = int(math.ceil((max(q[3] for q in rs) - Y0) / step) * step)
  print(f"{len(cells)} height tiles, {len(fill)} painted cells, canvas {W}x{H} at z{a.zmax}", flush=True)
  num = np.zeros((H, W), np.float32)
  den = np.zeros((H, W), np.float32)
  covered = np.zeros((H, W), bool)
  for k, (i, j) in enumerate(cells):
    s = json.load(open(f"{F}/depth/tiles/o{i:03d}_{j:03d}.json"))["height"]
    v = np.array(Image.open(f"{F}/depth/tiles/o{i:03d}_{j:03d}_h.png"))[CY0:CY0 + CH, CX0:CX0 + CW]
    ok = v != s["nodata"]
    h = np.where(ok, s["base_m"] + v.astype(np.float32) * s["scale_m"], 0).astype(np.float32)
    x0, y0, x1, y1 = rect(i, j)
    u0, v0 = int(round(x0 - X0)), int(round(y0 - Y0))
    w, hh = int(round((x1 - x0))), int(round((y1 - y0)))
    hs = cv2.resize(h * ok, (w, hh), interpolation=cv2.INTER_AREA)
    ms = cv2.resize(ok.astype(np.float32), (w, hh), interpolation=cv2.INTER_AREA)
    num[v0:v0 + hh, u0:u0 + w] += hs
    den[v0:v0 + hh, u0:u0 + w] += ms
    covered[v0:v0 + hh, u0:u0 + w] = True
    if k % 2000 == 0:
      print(f"{k} of {len(cells)}", flush=True)
  for c in fill:
    i, j = c["cell"]
    x0, y0, x1, y1 = rect(i, j)
    u0, v0 = int(round(x0 - X0)), int(round(y0 - Y0))
    w, hh = int(round((x1 - x0))), int(round((y1 - y0)))
    blank = den[v0:v0 + hh, u0:u0 + w] == 0
    num[v0:v0 + hh, u0:u0 + w][blank] = SURFACE.get(c["body"], 0.0)
    den[v0:v0 + hh, u0:u0 + w][blank] = 1.0
    covered[v0:v0 + hh, u0:u0 + w] = True
  ok = den > 0.05
  hgt = np.where(ok, num / np.maximum(den, 1e-6), 0).astype(np.float32)
  del num
  gap = covered & ~ok
  print(f"gaps inside the captures: {gap.sum()} px; filling from the nearest height", flush=True)
  if gap.any():
    idx = ndi.distance_transform_edt(~ok, return_distances=False, return_indices=True)
    hgt[gap] = hgt[idx[0][gap], idx[1][gap]]
    del idx
  hgt[~covered] = 0.0  # the sea past the captures
  from pmtiles.tile import Compression, TileType, zxy_to_tileid
  from pmtiles.writer import Writer
  tiles = {}
  cv, ox, oy, cov = hgt, X0 // 256, Y0 // 256, covered
  for z in range(a.zmax, a.zmin - 1, -1):
    ny, nx = cv.shape[0] // 256, cv.shape[1] // 256
    any_cov = cov.reshape(ny, 256, nx, 256).any(axis=(1, 3))
    for ty in range(ny):
      for tx in range(nx):
        if any_cov[ty, tx]:
          b = io.BytesIO()
          Image.fromarray(terrarium(cv[ty * 256:(ty + 1) * 256, tx * 256:(tx + 1) * 256]), "RGB").save(b, "PNG", optimize=False, compress_level=6)
          tiles[zxy_to_tileid(z, ox + tx, oy + ty)] = b.getvalue()
    print(f"z{z}: {int(any_cov.sum())} tiles", flush=True)
    if z > a.zmin:
      # even tile origin and size, so each level halves onto whole tiles
      if ox % 2:
        cv, cov, ox = np.pad(cv, ((0, 0), (256, 0))), np.pad(cov, ((0, 0), (256, 0))), ox - 1
      if oy % 2:
        cv, cov, oy = np.pad(cv, ((256, 0), (0, 0))), np.pad(cov, ((256, 0), (0, 0))), oy - 1
      if (cv.shape[1] // 256) % 2:
        cv, cov = np.pad(cv, ((0, 0), (0, 256))), np.pad(cov, ((0, 0), (0, 256)))
      if (cv.shape[0] // 256) % 2:
        cv, cov = np.pad(cv, ((0, 256), (0, 0))), np.pad(cov, ((0, 256), (0, 0)))
      cv = cv2.resize(cv, (cv.shape[1] // 2, cv.shape[0] // 2), interpolation=cv2.INTER_AREA)
      cov = cov.reshape(cov.shape[0] // 2, 2, cov.shape[1] // 2, 2).any(axis=(1, 3))
      ox, oy = ox // 2, oy // 2

  def lon(x, z):
    return x / (1 << z) * 360 - 180

  def lat(y, z):
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / (1 << z)))))

  bx0, by0, bx1, by1 = X0 / 256, Y0 / 256, (X0 + W) / 256, (Y0 + H) / 256
  bnd = (lon(bx0, a.zmax), lat(by1, a.zmax), lon(bx1, a.zmax), lat(by0, a.zmax))
  with open(f"{a.out}/dem.pmtiles.tmp", "wb") as f:
    w = Writer(f)
    for tid in sorted(tiles):
      w.write_tile(tid, tiles[tid])
    w.finalize({"tile_type": TileType.PNG, "tile_compression": Compression.NONE, "min_zoom": a.zmin, "max_zoom": a.zmax,
                "min_lon_e7": int(bnd[0] * 1e7), "min_lat_e7": int(bnd[1] * 1e7), "max_lon_e7": int(bnd[2] * 1e7),
                "max_lat_e7": int(bnd[3] * 1e7), "center_zoom": 14, "center_lon_e7": int((bnd[0] + bnd[2]) / 2 * 1e7),
                "center_lat_e7": int((bnd[1] + bnd[3]) / 2 * 1e7)},
               {"name": "gta5 elevation", "format": "png", "encoding": "terrarium", "lat0": gframe.LAT0,
                "attribution": "GTA V height captures (internal)"})
  os.replace(f"{a.out}/dem.pmtiles.tmp", f"{a.out}/dem.pmtiles")
  print(f"wrote {len(tiles)} tiles, {sum(len(v) for v in tiles.values()) / 1e6:.0f} MB, heights {hgt[covered].min():.1f}..{hgt[covered].max():.1f} m", flush=True)


if __name__ == "__main__":
  main()
