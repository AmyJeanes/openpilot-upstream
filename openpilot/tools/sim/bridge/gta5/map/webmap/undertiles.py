"""Under-deck imagery for the map's "beneath decks" view: every passing under-deck shot (ortho_depth.py under) cut to
what the top imagery can't show, as transparent WebP tiles z22..zmin over sat.pmtiles, plus their outline.

A shot's pixel is kept where its own height is below the top-level height tile's (depth/) by more than --dz m: the deck,
its parapets and anything else over the camera hide it from above. Lower levels are drawn over higher ones, so the view
shows the lowest level everywhere (the ground beneath a stack of decks, not the road between two of them). The albedo matches the top tiles' exactly, so no colour matching.

  undertiles.py <out dir> [--shots <tiles dir> ...] [--quality 70] [--zmin 10] [--dz 2] [--workers 6] [--names o061_079_L1,...]
"""
import argparse
import io
import json
import math
import multiprocessing as mp
import os
import sys

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import Polygon, mapping
from shapely.ops import transform, unary_union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gframe  # noqa: E402
from satfull import F, Z, S22, halve  # noqa: E402

SHOTS = [f"{F}/under/tiles", f"{F}/under_gaps/tiles"]  # the crossings' shots, then the gap planner's (imgcheck/under_gaps.py)
DEPTH = f"{F}/depth/tiles"
GRID_X0, GRID_Y0, STEP_X, STEP_Y = -4300.0, -4300.0, 69.12, 38.88  # ortho.py's grid: cell centres
EDGE = 8            # px dropped at a shot's border
MIN_M2 = 2.0        # smallest kept patch
ROUGH_M = 0.15      # top height's local spread (45 cm box) over a tree crown
TREE_M2 = 40.0      # a rough patch smaller than this is a lone crown, not a deck (car parks and roads under woods are rough too)
QUALITY, DZ = 70, 2.0
FOLDER: dict[str, str] = {}  # shot name -> its tiles dir


def height(path, m):
  v = cv2.imread(path, cv2.IMREAD_UNCHANGED)
  h = m["base_m"] + v.astype(np.float32) * m["scale_m"]
  h[v == m["nodata"]] = np.nan
  return h


def top(d, W, H):
  """The top-level heights in the shot's pixels (same scale), each pixel from the height tile of its own grid cell: a
  shot off the grid's centres spans several. None without any."""
  M = d["world_to_px"]
  ci = np.round(((np.arange(W) + 0.5 - M[0][2]) / M[0][0] - GRID_X0) / STEP_X).astype(int)
  cj = np.round(((np.arange(H) + 0.5 - M[1][2]) / M[1][1] - GRID_Y0) / STEP_Y).astype(int)
  out, found = np.full((H, W), np.nan, np.float32), False
  for i in np.unique(ci):
    for j in np.unique(cj):
      tp = f"{DEPTH}/o{i:03d}_{j:03d}.json"
      if not os.path.exists(tp):
        continue
      t = json.load(open(tp))
      ox, oy = (M[k][2] - t["world_to_px"][k][2] for k in (0, 1))
      ht = cv2.warpAffine(height(f"{DEPTH}/{t['height']['png']}", t["height"]), np.float32([[1, 0, ox], [0, 1, oy]]), (W, H),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
      k = np.ix_(cj == j, ci == i)
      out[k], found = ht[k], True
  return out if found else None


def mask(d, folder):
  """The shot's albedo and its kept pixels (uint8 0/255), or None without a top height tile."""
  al = cv2.imread(f"{folder}/{d['images']['albedo']}", cv2.IMREAD_COLOR)
  hu = height(f"{folder}/{d['height']['png']}", d["height"])
  H, W = hu.shape
  ht = top(d, W, H)
  if ht is None:
    return None, None
  with np.errstate(invalid="ignore"):
    m = (ht - hu > DZ) & (al != 0).any(axis=2)
  m[:EDGE], m[-EDGE:], m[:, :EDGE], m[:, -EDGE:] = False, False, False, False
  m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
  n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
  area = st[:, cv2.CC_STAT_AREA] * (d["cm_per_px"] / 100) ** 2
  drop = area < MIN_M2
  drop[0] = True
  # lone tree crowns: the top surface over them is rough, a deck's is smooth
  h0 = np.nan_to_num(ht, nan=0.0)
  rough = np.sqrt(np.maximum(cv2.blur(h0 * h0, (15, 15)) - cv2.blur(h0, (15, 15)) ** 2, 0)) > ROUGH_M
  drop |= (np.bincount(lab.ravel(), rough.ravel(), n) > 0.35 * st[:, cv2.CC_STAT_AREA]) & (area < TREE_M2)
  m[drop[lab]] = 0
  return al, m * 255


def fill_black(al):
  """Black (no albedo) pixels take their neighbours' colour, so resampling doesn't pull a dark rim into kept pixels."""
  ok = (al != 0).any(axis=2).astype(np.float32)
  if ok.all():
    return al
  out = al.astype(np.float32)
  for k in (9, 33, 129):
    w = cv2.blur(ok, (k, k))
    c = cv2.blur(out * ok[..., None], (k, k))
    hole = (ok == 0) & (w > 0)
    out[hole] = c[hole] / w[hole][:, None]
    ok = np.maximum(ok, hole.astype(np.float32))
  return np.clip(out, 0, 255).astype(np.uint8)


def z22(x, y):
  mx, my = gframe.merc(x, y)
  return float(mx) * S22, float(my) * S22


def shot(name):
  """One shot warped to z22: {(tx, ty): RGBA tile}, its outline (world polygons) and kept px."""
  d = json.load(open(f"{FOLDER[name]}/{name}.json"))
  al, m = mask(d, FOLDER[name])
  if al is None or not m.any():
    return name, {}, [], 0 if al is None else -1
  M = np.array(d["world_to_px"], np.float64)
  # z22 px -> world -> shot px
  s = S22 / gframe.K
  A = np.array([[M[0, 0] / s, 0, M[0, 2] - M[0, 0] * (0.5 * S22) / s],
                [0, -M[1, 1] / s, M[1, 2] + M[1, 1] * gframe.MY0 * S22 / s]])
  H, W = m.shape
  xw0, yn = (0 - M[0, 2]) / M[0, 0], (0 - M[1, 2]) / M[1, 1]
  xw1, ys = (W - M[0, 2]) / M[0, 0], (H - M[1, 2]) / M[1, 1]
  X0, Y0 = z22(xw0, yn)
  X1, Y1 = z22(xw1, ys)
  tx0, ty0, tx1, ty1 = int(X0 // 256), int(Y0 // 256), int(X1 // 256), int(Y1 // 256)
  ox, oy = tx0 * 256, ty0 * 256
  Ab = A.copy()
  Ab[:, 2] += A[:, 0] * ox + A[:, 1] * oy
  size = ((tx1 - tx0 + 1) * 256, (ty1 - ty0 + 1) * 256)
  rgb = cv2.warpAffine(fill_black(al), Ab, size, flags=cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_REPLICATE)
  a = cv2.warpAffine(m, Ab, size, flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
  out = {}
  for ty in range(ty1 - ty0 + 1):
    for tx in range(tx1 - tx0 + 1):
      ta = a[ty * 256:(ty + 1) * 256, tx * 256:(tx + 1) * 256]
      if ta.any():
        t = np.empty((256, 256, 4), np.uint8)
        t[..., :3] = rgb[ty * 256:(ty + 1) * 256, tx * 256:(tx + 1) * 256, ::-1]
        t[..., 3] = ta
        out[(tx0 + tx, ty0 + ty)] = t
  # outline at 12 cm, in world metres
  k = 4
  small = cv2.resize(m, (W // k, H // k), interpolation=cv2.INTER_AREA) > 127
  cs, hier = cv2.findContours(small.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  polys = []
  if hier is not None:
    def world(c):
      c = c[:, 0, :].astype(np.float64) * k + k / 2
      return np.c_[(c[:, 0] - M[0, 2]) / M[0, 0], (c[:, 1] - M[1, 2]) / M[1, 1]]
    for n, c in enumerate(cs):
      if hier[0][n][3] != -1 or len(c) < 3:
        continue
      holes = [world(cs[h]) for h in range(len(cs)) if hier[0][h][3] == n and len(cs[h]) >= 3]
      p = Polygon(world(c), holes).buffer(0)
      if not p.is_empty:
        polys.append(p)
  return name, out, polys, int(m.sum() // 255)


def over(dst, src):
  """src over dst, straight alpha."""
  if dst is None:
    return src
  sa, da = src[..., 3:].astype(np.float32) / 255, dst[..., 3:].astype(np.float32) / 255
  oa = sa + da * (1 - sa)
  rgb = (src[..., :3] * sa + dst[..., :3] * da * (1 - sa)) / np.maximum(oa, 1e-6)
  return np.dstack([np.clip(rgb + 0.5, 0, 255), oa * 255 + 0.5]).astype(np.uint8)


def encode(t):
  b = io.BytesIO()
  Image.fromarray(t, "RGBA").save(b, "WEBP", quality=QUALITY, method=4)
  return b.getvalue()


def down(tiles):
  """One zoom lower: each 2x2 block of tiles halved, in premultiplied alpha (no dark rim at the edges)."""
  par = {}
  for tx, ty in tiles:
    par.setdefault((tx >> 1, ty >> 1), []).append((tx, ty))
  out = {}
  for (px, py), kids in par.items():
    c = np.zeros((512, 512, 4), np.float32)
    for tx, ty in kids:
      t = tiles[(tx, ty)].astype(np.float32)
      t[..., :3] *= t[..., 3:] / 255
      c[(ty & 1) * 256:(ty & 1) * 256 + 256, (tx & 1) * 256:(tx & 1) * 256 + 256] = t
    h = halve(c)
    a = h[..., 3:]
    h[..., :3] = np.where(a > 0, h[..., :3] * 255 / np.maximum(a, 1e-6), 0)
    t = np.clip(h + 0.5, 0, 255).astype(np.uint8)
    if t[..., 3].any():
      out[(px, py)] = t
  return out


def write(path, enc):
  from pmtiles.tile import Compression, TileType, zxy_to_tileid
  from pmtiles.writer import Writer
  zs = sorted({k[0] for k in enc})
  zt = zs[-1]
  xs = [k[1] for k in enc if k[0] == zt]
  ys = [k[2] for k in enc if k[0] == zt]

  def lon(x, z):
    return x / (1 << z) * 360 - 180

  def lat(y, z):
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / (1 << z)))))

  bnd = (lon(min(xs), zt), lat(max(ys) + 1, zt), lon(max(xs) + 1, zt), lat(min(ys), zt))
  with open(path + ".tmp", "wb") as f:
    w = Writer(f)
    for k in sorted(enc, key=lambda k: zxy_to_tileid(*k)):
      w.write_tile(zxy_to_tileid(*k), enc[k])
    w.finalize({"tile_type": TileType.WEBP, "tile_compression": Compression.NONE, "min_zoom": zs[0], "max_zoom": zt,
                "min_lon_e7": int(bnd[0] * 1e7), "min_lat_e7": int(bnd[1] * 1e7), "max_lon_e7": int(bnd[2] * 1e7),
                "max_lat_e7": int(bnd[3] * 1e7), "center_zoom": 16, "center_lon_e7": int((bnd[0] + bnd[2]) / 2 * 1e7),
                "center_lat_e7": int((bnd[1] + bnd[3]) / 2 * 1e7)},
               {"name": "gta5 under-deck", "format": "webp", "lat0": gframe.LAT0, "z22_cm_per_px": 100 / gframe.px_per_m(Z),
                "attribution": "GTA V orthographic captures (internal)"})
  os.replace(path + ".tmp", path)


def main():
  global QUALITY, DZ
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--quality", type=int, default=70)
  ap.add_argument("--zmin", type=int, default=10)
  ap.add_argument("--dz", type=float, default=2.0, help="m the top surface must be above the shot's to keep a pixel")
  ap.add_argument("--workers", type=int, default=6)
  ap.add_argument("--names", default="", help="only these shots (a test)")
  ap.add_argument("--shots", nargs="+", default=SHOTS, help="the passing under-deck shots' tiles dirs (missing ones skipped)")
  a = ap.parse_args()
  QUALITY, DZ = a.quality, a.dz
  sides = {}
  for folder in a.shots:
    for f in sorted(os.listdir(folder)) if os.path.isdir(folder) else []:
      if f.endswith(".json"):
        sides[f[:-5]] = json.load(open(f"{folder}/{f}"))
        FOLDER[f[:-5]] = folder
  names = a.names.split(",") if a.names else [n for n, d in sides.items() if d.get("pass")]
  names.sort(key=lambda n: (-sides[n]["camera"]["cam_z"], n))  # the lowest level drawn last, over the others
  print(f"{len(sides)} shots, {sum(not d.get('pass') for d in sides.values())} failed their checks, using {len(names)}", flush=True)
  cv2.setNumThreads(2)
  tiles, polys, used, empty, notop = {}, [], 0, [], []
  with mp.get_context("fork").Pool(a.workers) as pool:
    for n, (name, ts, ps, px) in enumerate(pool.imap(shot, names, chunksize=1)):
      if px == 0:
        notop.append(name)
      elif px < 0:
        empty.append(name)
      else:
        used += 1
        for k, t in ts.items():
          tiles[k] = over(tiles.get(k), t)
        polys += ps
      if (n + 1) % 50 == 0:
        print(f"{n + 1} of {len(names)} shots, {len(tiles)} z{Z} tiles", flush=True)
  print(f"used {used}; nothing beneath a deck: {len(empty)} {empty}; no top height tile: {len(notop)} {notop}", flush=True)
  enc = {}
  for z in range(Z, a.zmin - 1, -1):
    for (x, y), t in tiles.items():
      enc[(z, x, y)] = encode(t)
    print(f"z{z}: {len(tiles)} tiles", flush=True)
    if z > a.zmin:
      tiles = down(tiles)
  write(f"{a.out}/under.pmtiles", enc)
  print(f"wrote {len(enc)} tiles, {sum(map(len, enc.values())) / 1e6:.1f} MB -> {a.out}/under.pmtiles", flush=True)
  u = unary_union(polys).simplify(0.2)
  geoms = list(getattr(u, "geoms", [u]))
  geoms = [g for g in geoms if g.area >= MIN_M2]

  def ll(x, y, z=None):
    lat, lon = gframe.lat_lon(np.asarray(x), np.asarray(y))
    return np.round(lon, 7), np.round(lat, 7)

  fc = {"type": "FeatureCollection", "features": [
    {"type": "Feature", "properties": {"kind": "under", "area_m2": round(g.area, 1)}, "geometry": mapping(transform(ll, g))}
    for g in geoms]}
  with open(f"{a.out}/under_footprint.geojson.tmp", "w") as f:
    json.dump(fc, f, separators=(",", ":"))
  os.replace(f"{a.out}/under_footprint.geojson.tmp", f"{a.out}/under_footprint.geojson")
  print(f"footprint: {len(geoms)} polygons, {sum(g.area for g in geoms):.0f} m2", flush=True)


if __name__ == "__main__":
  main()
