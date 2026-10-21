"""Full-map satellite tiler: every ortho cell (water tiles first, then the filled ones, then the raw albedo; centre
crop 2304x1296) -> 256 px WebP tiles z22..zmin in the lat0 Web Mercator frame, as one PMTiles file.

Works in z16-tile blocks (64x64 z22 tiles, ~0.5 km): each block's cells are mosaicked and warped to z22, tiled down to z16
and spooled to disk; the z16 images are then assembled for z15..zmin. Resumable: a block with a finished spool is kept.

  satfull.py <out dir> [--quality 70] [--zmin 10] [--workers 3] [--blocks N (N blocks mid-map only, a test)]
"""
import argparse
import io
import json
import math
import multiprocessing as mp
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gframe  # noqa: E402

F = "/mnt/e/gta5_ortho_20261018"
FILL = f"{F}/full_water/fill"  # open-water cells the capture skipped, painted the body's colour
SRC = [f"{F}/full_water/tiles", f"{F}/full_filled", f"{F}/full/tiles", FILL]
CW, CH, CX0, CY0 = 2304, 1296, 128, 72
PPM = 100.0 / 3.0
Z, B = 22, 16                      # top zoom, block zoom
S22 = 256 * (1 << Z)
BPX = 256 << (Z - B)               # block size in z22 px
QUALITY = 70
SEA = np.array([0x30, 0x5d, 0x63], np.float32)  # the open sea's colour (imgcheck/water/params.json)


def centre(i, j):
  return -4300 + 69.12 * i, -4300 + 38.88 * j


def z22(x, y):
  mx, my = gframe.merc(x, y)
  return float(mx) * S22, float(my) * S22


def load(i, j):
  for d in SRC:
    p = f"{d}/o{i:03d}_{j:03d}.png"
    if os.path.exists(p):
      im = cv2.imread(p, cv2.IMREAD_COLOR)
      if im is not None:
        return im[CY0:CY0 + CH, CX0:CX0 + CW]
  return None


def encode(t):
  a = t[..., 3]
  if not a.any():
    return None
  rgb = t[..., :3]
  if a.min() < 255:  # the captures' edge, laid on the colour the page shows past them so the two meet without a line
    rgb = np.clip(rgb + SEA * (1 - a[..., None] / 255.0), 0, 255).astype(np.uint8)
  near = (np.abs(rgb.astype(np.int16) - SEA.astype(np.int16)) <= 3).all(-1)  # resampling ripples on the open sea
  if near.any():
    rgb = np.where(near[..., None], SEA.astype(np.uint8), rgb)
  b = io.BytesIO()
  # WebP decodes SEA as #305d62 at any quality (index.html's background); open sea is stored finer, as at the default
  # quality a flat tile gets specks at its corners
  Image.fromarray(rgb, "RGB").save(b, "WEBP", quality=95 if near.mean() > 0.5 else QUALITY, method=4)
  return b.getvalue()


def halve(cv):
  return cv2.resize(cv, (cv.shape[1] // 2, cv.shape[0] // 2), interpolation=cv2.INTER_AREA)


def block(args):
  bx, by, cells, out = args
  spool = f"{out}/spool/{bx}_{by}"
  if os.path.exists(spool + ".idx"):
    return bx, by, np.load(spool + ".z16.npy"), 0
  X0, Y0 = bx * BPX, by * BPX
  # the cells' mosaic, its north-west corner at the westmost, northmost cell's
  i0, i1 = min(c[0] for c in cells), max(c[0] for c in cells)
  j0, j1 = min(c[1] for c in cells), max(c[1] for c in cells)
  M = np.zeros(((j1 - j0 + 1) * CH, (i1 - i0 + 1) * CW, 4), np.uint8)
  for i, j in cells:
    im = load(i, j)
    if im is None:
      continue
    r0, c0 = (j1 - j) * CH, (i - i0) * CW
    M[r0:r0 + CH, c0:c0 + CW, :3] = im[..., ::-1]
    M[r0:r0 + CH, c0:c0 + CW, 3] = np.where((im != 0).any(axis=2), 255, 0)
  xc, yc = centre(i0, j1)
  Xw, Yn = z22(xc - CW / PPM / 2, yc + CH / PPM / 2)
  r = PPM / gframe.px_per_m(Z)
  Mx = np.array([[r, 0, (X0 - Xw) * r], [0, r, (Y0 - Yn) * r]], np.float64)
  cv = cv2.warpAffine(M, Mx, (BPX, BPX), flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT,
                      borderValue=(0, 0, 0, 0))
  del M
  idx, nt = [], 0
  with open(spool + ".dat.tmp", "wb") as f:
    for z in range(Z, B - 1, -1):
      k = cv.shape[0] // 256
      tx0, ty0 = bx << (z - B), by << (z - B)
      al = cv[..., 3].reshape(k, 256, k, 256).max(axis=(1, 3))
      for ty in range(k):
        for tx in range(k):
          if not al[ty, tx]:
            continue
          d = encode(cv[ty * 256:(ty + 1) * 256, tx * 256:(tx + 1) * 256])
          if d:
            idx.append((z, tx0 + tx, ty0 + ty, f.tell(), len(d)))
            f.write(d)
            nt += 1
      if z > B:
        cv = halve(cv)
  np.save(spool + ".z16.npy", cv)
  os.replace(spool + ".dat.tmp", spool + ".dat")
  with open(spool + ".idx.tmp", "w") as f:
    json.dump(idx, f)
  os.replace(spool + ".idx.tmp", spool + ".idx")
  return bx, by, cv, nt


def write(out, keys, low):
  from pmtiles.tile import Compression, TileType, zxy_to_tileid
  from pmtiles.writer import Writer
  entries = []  # (tileid, block or None, offset, length, z, x, y)
  for bx, by in keys:
    for z, x, y, off, ln in json.load(open(f"{out}/spool/{bx}_{by}.idx")):
      entries.append((zxy_to_tileid(z, x, y), (bx, by), off, ln, z, x, y))
  lowd = {}
  for (z, x, y), d in low:
    tid = zxy_to_tileid(z, x, y)
    lowd[tid] = d
    entries.append((tid, None, 0, len(d), z, x, y))
  entries.sort(key=lambda e: e[0])
  zs = [e[4] for e in entries]
  zt = max(zs)
  xs = [e[5] for e in entries if e[4] == zt]
  ys = [e[6] for e in entries if e[4] == zt]

  def lon(x, z):
    return x / (1 << z) * 360 - 180

  def lat(y, z):
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / (1 << z)))))

  bnd = (lon(min(xs), zt), lat(max(ys) + 1, zt), lon(max(xs) + 1, zt), lat(min(ys), zt))
  files, total = {}, 0
  tmp = f"{out}/sat.pmtiles.tmp"
  with open(tmp, "wb") as f:
    w = Writer(f)
    for tid, key, off, ln, *_ in entries:
      if key is None:
        d = lowd[tid]
      else:
        if key not in files:
          files[key] = open(f"{out}/spool/{key[0]}_{key[1]}.dat", "rb")
        fh = files[key]
        fh.seek(off)
        d = fh.read(ln)
      w.write_tile(tid, d)
      total += ln
    w.finalize({"tile_type": TileType.WEBP, "tile_compression": Compression.NONE, "min_zoom": min(zs), "max_zoom": zt,
                "min_lon_e7": int(bnd[0] * 1e7), "min_lat_e7": int(bnd[1] * 1e7), "max_lon_e7": int(bnd[2] * 1e7),
                "max_lat_e7": int(bnd[3] * 1e7), "center_zoom": 14, "center_lon_e7": int((bnd[0] + bnd[2]) / 2 * 1e7),
                "center_lat_e7": int((bnd[1] + bnd[3]) / 2 * 1e7)},
               {"name": "gta5 satellite", "format": "webp", "lat0": gframe.LAT0, "z22_cm_per_px": 100 / gframe.px_per_m(Z),
                "attribution": "GTA V orthographic captures (internal)"})
  for fh in files.values():
    fh.close()
  os.replace(tmp, f"{out}/sat.pmtiles")
  print(f"wrote {len(entries)} tiles, {total / 1e9:.2f} GB -> {out}/sat.pmtiles", flush=True)


def main():
  global QUALITY
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--quality", type=int, default=70)
  ap.add_argument("--zmin", type=int, default=10)
  ap.add_argument("--workers", type=int, default=3)
  ap.add_argument("--blocks", type=int, default=0)
  ap.add_argument("--redo-fill", action="store_true", help="rebuild the blocks that hold painted open-water cells")
  ap.add_argument("--redo-cells", default="", help="rebuild the blocks that hold these cells: i,j;i,j")
  a = ap.parse_args()
  QUALITY = a.quality
  os.makedirs(f"{a.out}/spool", exist_ok=True)
  cv2.setNumThreads(2)
  cells = {(int(f[1:4]), int(f[5:8])) for f in os.listdir(f"{F}/full/tiles") if f.endswith(".json")}
  fill = {(int(f[1:4]), int(f[5:8])) for f in os.listdir(FILL) if f.endswith(".png")} if os.path.isdir(FILL) else set()
  cells = sorted(cells | fill)
  blocks = {}
  for i, j in cells:
    xc, yc = centre(i, j)
    Xa, Ya = z22(xc - CW / PPM / 2, yc + CH / PPM / 2)
    Xb, Yb = z22(xc + CW / PPM / 2, yc - CH / PPM / 2)
    for bx in range(int(Xa // BPX), int(Xb // BPX) + 1):
      for by in range(int(Ya // BPX), int(Yb // BPX) + 1):
        blocks.setdefault((bx, by), []).append((i, j))
  have = set(cells)
  for k, cs in blocks.items():  # a cell of margin all round, for the resampling kernel
    blocks[k] = sorted({(i + di, j + dj) for i, j in cs for di in (-1, 0, 1) for dj in (-1, 0, 1)} & have)
  keys = sorted(blocks)
  redo = fill if a.redo_fill else {tuple(int(v) for v in c.split(",")) for c in a.redo_cells.split(";") if c}
  if redo:
    for k in keys:
      if redo & set(blocks[k]):
        for ext in (".idx", ".dat", ".z16.npy"):
          p = f"{a.out}/spool/{k[0]}_{k[1]}{ext}"
          if os.path.exists(p):
            os.remove(p)
  if a.blocks:
    mid = len(keys) // 2
    keys = keys[mid:mid + a.blocks]
  print(f"{len(cells)} cells, {len(keys)} blocks", flush=True)
  step = 1 << (B - a.zmin)
  bx0, by0 = min(k[0] for k in keys), min(k[1] for k in keys)
  bx0 -= bx0 % step
  by0 -= by0 % step
  nx, ny = max(k[0] for k in keys) - bx0 + 1, max(k[1] for k in keys) - by0 + 1
  nx += (-nx) % step
  ny += (-ny) % step
  C16 = np.zeros((ny * 256, nx * 256, 4), np.uint8)
  t0, done, ntiles = time.time(), 0, 0
  with mp.get_context("fork").Pool(a.workers, maxtasksperchild=8) as pool:
    for bx, by, img, nt in pool.imap_unordered(block, [(bx, by, blocks[(bx, by)], a.out) for bx, by in keys]):
      C16[(by - by0) * 256:(by - by0 + 1) * 256, (bx - bx0) * 256:(bx - bx0 + 1) * 256] = img
      done += 1
      ntiles += nt
      if done % 20 == 0 or done == len(keys):
        el = time.time() - t0
        print(f"{done} of {len(keys)} blocks, {ntiles} tiles this run, {el / 60:.1f} min, ~{el / done * (len(keys) - done) / 60:.0f} min left",
              flush=True)
  low, cv = [], C16
  for z in range(B - 1, a.zmin - 1, -1):
    cv = halve(cv)
    k0x, k0y = bx0 >> (B - z), by0 >> (B - z)
    al = cv[..., 3].reshape(cv.shape[0] // 256, 256, cv.shape[1] // 256, 256).max(axis=(1, 3))
    for ty in range(al.shape[0]):
      for tx in range(al.shape[1]):
        if al[ty, tx]:
          d = encode(cv[ty * 256:(ty + 1) * 256, tx * 256:(tx + 1) * 256])
          if d:
            low.append(((z, k0x + tx, k0y + ty), d))
  print(f"low zooms: {len(low)} tiles", flush=True)
  write(a.out, keys, low)


if __name__ == "__main__":
  main()
