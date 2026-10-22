#!/usr/bin/env python3
"""Under-deck gaps: road the top imagery hides that no passing under-deck shot shows, and a plan of shots to fill it.
The crossings' plan (ortho_depth.py underplan) misses road under anything that isn't a road (rail viaducts, gantries) and
decks running on past the cell holding the crossing; this plans from what is actually hidden.

hidden: the game-file road surface (roadpaint raster, any layer, road or gutter) more than DZ m below the top-level height
tile (depth/), on a 25 cm world grid, per 64 m raster tile -> <out>/hidden/<tx>_<ty>.npz (see hidden_tile).
reach: how far from its centre a passing shot shows the hidden road on its level (sets REACH_X, REACH_Y).
plan: the hidden road no passing shot keeps (undertiles' cut), less foliage, cover too low for the camera, tunnels,
buildings' floors and slivers, covered greedily by shots centred on it, each at one camera height (ortho_depth.py under's
plan format, names oIII_JJJ_G<n>) -> <out>/plan.json and report.json (areas by what becomes of them, shots, hours).
measure: a box's hidden road by what shows it, as numbers and a PNG.

  python under_gaps.py hidden <out> [--box x0 x1 y0 y1]
  python under_gaps.py reach <out> [--shots <tiles dir> ...]
  python under_gaps.py plan <out> [--shots <tiles dir> ...] [--box x0 x1 y0 y1]
  python under_gaps.py measure <out> <png> --box x0 x1 y0 y1 [--shots ...] [--extra ...] [--plan plan.json]
"""
import argparse
import collections
import functools
import json
import math
import multiprocessing as mp
import os

import cv2
import numpy as np

from openpilot.tools.sim.bridge.gta5.map.imgcheck import ortho, ortho_depth
from openpilot.tools.sim.bridge.gta5.map.imgcheck.ortho import H_M, W_M
from openpilot.tools.sim.bridge.gta5.map.roadpaint import rp
from openpilot.tools.sim.bridge.gta5.map.webmap import undertiles

F = "/mnt/e/gta5_ortho_20261018"
RASTER = "/mnt/e/gta5_scratch/gamefiles_v2/rp_all/tiles"
DEPTH, ALBEDO = f"{F}/depth/tiles", f"{F}/full/tiles"
RES = 0.25  # m: the world grid
K = round(RES / rp.RES)  # raster pixels per grid step
G = round(rp.TILE / RES)  # grid steps per raster tile
DZ = 2.0  # m the top surface must be over the road to hide it (as undertiles)
DRIVEABLE = (rp.ROAD, rp.GUTTER)
ROUGH_M = 0.15  # top height's local spread (45 cm box) over a tree crown (as undertiles)
SHOTS = [f"{F}/under/tiles", f"{F}/under_gaps/tiles"]
PED_SPECIALS = {10, 14, 15, 16}  # path node specials that aren't roads
# plan (the rules for covers that aren't road decks set from contact sheets of random planned shots, 22 Oct: without
# them most were trees, parked trailers and buildings' floors)
LOW_M = 4.0  # m from the road up to its lowest cover's top under which the camera can't fit between them (underplan's DZ)
OTHER_LOW_M = 6.0  # m: the same under a cover that isn't a road deck; lower ones are parked trailers and containers
TUNNEL_M = 30.0  # m from the nearest road the top imagery shows beyond which hidden road is a tunnel or long covered way
NARROW_M, WIDE_SHARE = 12.0, 0.2  # a patch not under a road deck with over this share further than this (m) from road
# seen from above is under a building: a viaduct or gantry over a road is narrower
LINK_M = 4.0  # m of the game's road links a patch not under a road deck needs: else it's a building's floor, not a road
ROUGH_SHARE = 0.35  # a patch not under a road deck with a rough top over more than this share is under trees (undertiles')
SLIVER, OTHER_SLIVER = 5, 9  # grid steps (1.25, 2.25 m; odd, so opening stays inside): narrower patches (deck edges,
# eaves) aren't worth a shot
MIN_GAP_M2, MIN_OTHER_M2, MIN_SHOT_M2 = 4.0, 20.0, 5.0  # smallest patch planned under a deck, under anything else; least new
# road a shot must show
# m from a shot's centre within which it shows its level: passing shots keep >= 90% of it out to here (reach over the
# 397 crossings' and 548 covered-road shots, 22 Oct); it falls to ~80% by the frame's last 2 m (its edge cut and opening)
REACH_X, REACH_Y = 34.0, 18.0
CAM_ABOVE, CAM_UNDER, CAM_STEP = 2.0, 1.8, 0.5  # m: camera over the road (U_MIN_ABOVE), under the cover's top (deck + 0.3)
CROSSINGS = 10  # points per shot for the clearance rays (3 rays each, 30 at most)
RSEG_M, RSEG_W = 4.0, 2.0  # m: length and width of the raster's stand-in links (checks a and b)
BLOCK = 8  # raster tiles per planning block side
S_PER_SHOT = 11.5  # s: the first gap pass (345 shots, 22 Oct, retries included; 88% passed)
HID: dict = {}
SHOWN: dict = {}
LINKS = np.zeros((0, 7))


# ------------------------------------------------------------------------------------------------------- hidden road
@functools.lru_cache(maxsize=16)
def depth_tile(i: int, j: int):
  """A height tile's heights, rough mask and world_to_px, or None."""
  p = f"{DEPTH}/o{i:03d}_{j:03d}.json"
  if not os.path.exists(p):
    return None
  s = json.load(open(p))
  m = s["height"]
  v = cv2.imread(f"{DEPTH}/{m['png']}", cv2.IMREAD_UNCHANGED)
  h = m["base_m"] + v.astype(np.float32) * m["scale_m"]
  h[v == m["nodata"]] = np.nan
  h0 = np.nan_to_num(h, nan=0.0)
  rough = np.sqrt(np.maximum(cv2.blur(h0 * h0, (15, 15)) - cv2.blur(h0, (15, 15)) ** 2, 0)) > ROUGH_M
  return h, rough, np.array(s["world_to_px"])


@functools.lru_cache(maxsize=4)
def green_tile(i: int, j: int):
  """The share of green albedo pixels within 1 m of each pixel (classify.py's foliage test), or None."""
  im = cv2.imread(f"{ALBEDO}/o{i:03d}_{j:03d}.png", cv2.IMREAD_COLOR)
  if im is None:
    return None
  b, g, r = (im[..., k].astype(np.float32) for k in range(3))
  green = ((g > r * 1.05) & (g > b * 1.1) & (im.max(axis=2) > 25)).astype(np.float32)
  return cv2.blur(green, (31, 31))


def cells_of(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  return np.round((x - ortho.GRID_X0) / ortho.STEP_X).astype(int), np.round((y - ortho.GRID_Y0) / ortho.STEP_Y).astype(int)


def grid_xy(tx: int, ty: int) -> tuple[np.ndarray, np.ndarray]:
  """World x, y of a raster tile's grid points (G x G, row 0 north)."""
  c = (np.arange(G) + 0.5) * RES
  return np.meshgrid(tx * rp.TILE + c, ty * rp.TILE + rp.TILE - c)


def top_at(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """The top-level height and its roughness at world points, each from the height tile of the point's own cell."""
  top, rough = np.full(x.shape, np.nan, np.float32), np.zeros(x.shape, bool)
  ci, cj = cells_of(x, y)
  for i, j in set(zip(ci.ravel().tolist(), cj.ravel().tolist(), strict=True)):
    t = depth_tile(i, j)
    if t is None:
      continue
    h, r, M = t
    k = (ci == i) & (cj == j)
    c = np.clip((M[0, 0] * x[k] + M[0, 2]).astype(int), 0, h.shape[1] - 1)
    w = np.clip((M[1, 1] * y[k] + M[1, 2]).astype(int), 0, h.shape[0] - 1)
    top[k], rough[k] = h[w, c], r[w, c]
  return top, rough


def green_at(x: np.ndarray, y: np.ndarray) -> np.ndarray:
  """The greenness (0-1) of the top albedo at world points."""
  gr = np.zeros(x.shape, np.float32)
  ci, cj = cells_of(x, y)
  for i, j in set(zip(ci.tolist(), cj.tolist(), strict=True)):
    if (gt := green_tile(i, j)) is None:
      continue
    M = depth_tile(i, j)[2]
    k = (ci == i) & (cj == j)
    c = np.clip((M[0, 0] * x[k] + M[0, 2]).astype(int), 0, gt.shape[1] - 1)
    w = np.clip((M[1, 1] * y[k] + M[1, 2]).astype(int), 0, gt.shape[0] - 1)
    gr[k] = gt[w, c]
  return gr


def hidden_tile(key: tuple[int, int], out: str) -> tuple[tuple[int, int], int]:
  """One raster tile's road -> <out>/hidden/<tx>_<ty>.npz (G x G each): z the lowest hidden road layer (NaN where none
  is hidden), top the top-level height over it, seen the road the top imagery shows, deck a raster surface at the top's
  height (a road deck over the hidden road), rough + green the top over it (foliage). The number of hidden points."""
  tx, ty = key
  code, z = rp.read_tile(f"{RASTER}/{tx}_{ty}.bin")
  code, z = code[:, K // 2::K, K // 2::K], z[:, K // 2::K, K // 2::K]
  drive = np.isin(code & 7, DRIVEABLE)
  if not drive.any():
    return key, 0
  X, Y = grid_xy(tx, ty)
  top, rough = top_at(X, Y)
  with np.errstate(invalid="ignore"):
    hid = drive & (z < top[None] - DZ)
    seen = (drive & (z >= top[None] - DZ)).any(0)
    deck = ((code & 7) > 0) & (np.abs(z - top[None]) < 1.0)
  h2 = hid.any(0)
  green = np.zeros(h2.shape, np.float32)
  if (h2 & rough).any():
    green[h2 & rough] = green_at(X[h2 & rough], Y[h2 & rough])
  low = np.where(hid, z, np.inf).min(0)
  low[~h2] = np.nan
  # the lowest raster surface over it (a road deck under a higher one): the camera must stay under that
  with np.errstate(invalid="ignore"):
    over = np.where(((code & 7) > 0) & (z > low[None] + DZ), z, np.inf).min(0)
  over = np.where(np.isfinite(over), over, top)
  np.savez_compressed(f"{out}/hidden/{tx}_{ty}.npz", z=low.astype(np.float32), top=np.where(h2, top, np.nan).astype(np.float32),
                      over=np.where(h2, over, np.nan).astype(np.float32), seen=seen, deck=deck.any(0) & h2, rough=rough & h2,
                      green=np.round(green * 100).astype(np.uint8))
  return key, int(h2.sum())


def hidden(a) -> None:
  os.makedirs(f"{a.out}/hidden", exist_ok=True)
  keys = sorted(tuple(map(int, f[:-4].split("_"))) for f in os.listdir(RASTER) if f.endswith(".bin"))
  if a.box:
    x0, x1, y0, y1 = a.box
    keys = [(tx, ty) for tx, ty in keys if tx * rp.TILE < x1 and (tx + 1) * rp.TILE > x0 and ty * rp.TILE < y1 and (ty + 1) * rp.TILE > y0]
  keys = [k for k in keys if not os.path.exists(f"{a.out}/hidden/{k[0]}_{k[1]}.npz")] if a.resume else keys
  total = 0
  with mp.get_context("fork").Pool(a.workers) as pool:
    for n, (_, h) in enumerate(pool.imap(functools.partial(hidden_tile, out=a.out), keys, chunksize=8)):
      total += h
      if (n + 1) % 500 == 0:
        print(f"{n + 1} of {len(keys)} raster tiles, {total * RES * RES:.0f} m2 hidden so far", flush=True)
  print(f"{len(keys)} raster tiles: {total * RES * RES:.0f} m2 of road hidden -> {a.out}/hidden", flush=True)


def load_hidden(out: str) -> dict[tuple[int, int], dict]:
  h = {}
  for f in os.listdir(f"{out}/hidden"):
    if f.endswith(".npz"):
      with np.load(f"{out}/hidden/{f}") as d:
        h[tuple(map(int, f[:-4].split("_")))] = dict(d)
  return h


def foliage(d: dict) -> np.ndarray:
  return d["rough"] & (d["green"] >= 50)


# -------------------------------------------------------------------------------------------------------- shown road
def passing(folders: list[str]) -> list[tuple[str, dict]]:
  """(tiles dir, sidecar) of every passing under-deck shot in these dirs."""
  out = []
  for d in folders:
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
      if f.endswith(".json"):
        s = json.load(open(f"{d}/{f}"))
        if s.get("pass"):
          out.append((d, s))
  return out


def frame_tiles(s: dict):
  """The raster tiles a shot's frame touches: key, their grid points' world x, y, the shot's pixel column and row of
  each, and which are in the frame."""
  M, (W, H) = np.array(s["world_to_px"]), s["size"]
  x0, x1 = -M[0, 2] / M[0, 0], (W - M[0, 2]) / M[0, 0]
  y1, y0 = -M[1, 2] / M[1, 1], (H - M[1, 2]) / M[1, 1]
  for tx in range(math.floor(x0 / rp.TILE), math.floor(x1 / rp.TILE) + 1):
    for ty in range(math.floor(y0 / rp.TILE), math.floor(y1 / rp.TILE) + 1):
      X, Y = grid_xy(tx, ty)
      c, r = np.floor(M[0, 0] * X + M[0, 2]).astype(int), np.floor(M[1, 1] * Y + M[1, 2]).astype(int)
      k = (c >= 0) & (c < W) & (r >= 0) & (r < H)
      if k.any():
        yield (tx, ty), X, Y, c, r, k


def shown_shot(item) -> tuple[str, dict]:
  """One shot's kept pixels (undertiles' cut) on the world grid: its name, {(tx, ty): G x G bool}."""
  cv2.setNumThreads(1)
  folder, s = item
  _, m = undertiles.mask(s, folder)
  out = {}
  if m is not None:
    for key, _, _, c, r, k in frame_tiles(s):
      g = np.zeros((G, G), bool)
      g[k] = m[r[k], c[k]] > 0
      if g.any():
        out[key] = g
  return s["name"], out


def shown(folders: list[str], workers: int, each: bool = False, box=None):
  """The union of the passing shots' kept pixels on the grid ({(tx, ty): G x G bool}); with each, also every shot's own.
  box (x0 x1 y0 y1): only the shots whose frames touch it."""
  union, per = {}, {}
  shots = passing(folders)
  if box:
    shots = [(f, s) for f, s in shots if box[0] - W_M / 2 < s["x"] < box[1] + W_M / 2 and box[2] - H_M / 2 < s["y"] < box[3] + H_M / 2]
  with mp.get_context("fork").Pool(workers) as pool:
    for name, g in pool.imap_unordered(shown_shot, shots, chunksize=2):
      for key, v in g.items():
        union[key] = union[key] | v if key in union else v
      if each:
        per[name] = g
  print(f"{len(shots)} passing shots in {', '.join(folders)}", flush=True)
  return (union, per) if each else union


# ------------------------------------------------------------------------------------------------------------ reach
def reach(a) -> None:
  """How far from its centre a passing shot shows the hidden road it could: of the hidden road in its frame on its level
  (within 1 m of its lower road's height range, the top over the camera by over 1 m, not foliage), the share it keeps by
  distance from the centre east-west and north-south (2 m bins) -> <out>/reach.json."""
  hid = load_hidden(a.out)
  shots = {s["name"]: s for _, s in passing(a.shots)}
  _, per = shown(a.shots, a.workers, each=True)
  nx, ny = math.ceil(W_M / 2 / 2), math.ceil(H_M / 2 / 2)
  tot, kept = np.zeros((ny, nx)), np.zeros((ny, nx))
  depth_tot, depth_kept = collections.Counter(), collections.Counter()
  for name, g in per.items():
    s = shots[name]
    lo, hi, cz = s["road"]["z_lo"] - 1.0, s["road"]["z_hi"] + 1.0, s["camera"]["cam_z"]
    for key, X, Y, _, _, k in frame_tiles(s):
      if key not in hid:
        continue
      d = hid[key]
      with np.errstate(invalid="ignore"):
        e = k & np.isfinite(d["z"]) & (d["z"] >= lo) & (d["z"] <= hi) & (d["top"] > cz + 1.0) & ~foliage(d)
      if not e.any():
        continue
      bx = np.minimum((np.abs(X[e] - s["x"]) / 2).astype(int), nx - 1)
      by = np.minimum((np.abs(Y[e] - s["y"]) / 2).astype(int), ny - 1)
      kk = g.get(key, np.zeros((G, G), bool))[e]
      np.add.at(tot, (by, bx), 1)
      np.add.at(kept, (by, bx), kk)
      dz = np.minimum(((cz - d["z"][e]) / 2).astype(int), 5)
      for v, q in zip(dz.tolist(), kk.tolist(), strict=True):
        depth_tot[v] += 1
        depth_kept[v] += q
  share = np.where(tot > 0, kept / np.maximum(tot, 1), np.nan)
  print("share kept by |dx| (columns, 2 m bins) and |dy| (rows); hidden m2 per row on the right")
  print("      " + " ".join(f"{2 * i:>4d}" for i in range(nx)))
  for j in range(ny):
    print(f"{2 * j:>4d}  " + " ".join(f"{v:4.2f}" if np.isfinite(v) else "   -" for v in share[j]) + f"  {tot[j].sum() * RES * RES:8.0f}")
  print("by camera height over the road (2 m bins):", {2 * k: round(depth_kept[k] / depth_tot[k], 3) for k in sorted(depth_tot)})
  json.dump({"shots": len(per), "bin_m": 2, "share": np.where(np.isfinite(share), share, -1).round(3).tolist(),
             "hidden_m2": (tot * RES * RES).round(1).tolist(), "by_camera_height": {2 * k: [depth_kept[k], depth_tot[k]] for k in sorted(depth_tot)}},
            open(f"{a.out}/reach.json", "w"))


# ------------------------------------------------------------------------------------------------------------- plan
def road_links(paths: str) -> np.ndarray:
  """The game's vehicle links (no ped, boat or laneless ones): x0 y0 z0 x1 y1 z1 width per row."""
  nodes, segs, seen = {}, [], set()
  rows = [json.loads(line) for line in open(paths)]
  for r in rows:
    if r["t"] == "n" and (r["f"][1] >> 3) not in PED_SPECIALS and not r["f"][2] & 32:
      nodes[(r["a"], r["i"])] = (r["x"], r["y"], r["z"])
  for r in rows:
    if r["t"] != "l":
      continue
    a, b = (r["a"], r["i"]), (r["ta"], r["ti"])
    f2 = (r.get("f") or [0, 0, 0])[2]
    if a in nodes and b in nodes and (b, a) not in seen and ((f2 >> 5) & 7) + ((f2 >> 2) & 7) > 0:
      seen.add((a, b))
      segs.append(nodes[a] + nodes[b] + (ortho_depth.lanes_width(f2),))
  return np.array(segs, np.float64)


def clip(s: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> list | None:
  """A link cut to a box (Liang-Barsky), its height carried along; None when it misses."""
  ax, ay, az, bx, by, bz, w = s
  t0, t1 = 0.0, 1.0
  for p, q in ((-(bx - ax), ax - x0), (bx - ax, x1 - ax), (-(by - ay), ay - y0), (by - ay, y1 - ay)):
    if abs(p) < 1e-12:
      if q < 0:
        return None
      continue
    t = q / p
    if p < 0:
      t0 = max(t0, t)
    else:
      t1 = min(t1, t)
  if t1 - t0 < 1e-3:
    return None
  def at(t):
    return ax + t * (bx - ax), ay + t * (by - ay), az + t * (bz - az)
  return [round(float(v), 2) for v in (*at(t0), *at(t1), w)]


def spread(p: np.ndarray, n: int) -> np.ndarray:
  """n points of p (rows x, y, ...) spread out: the first nearest the middle, then each farthest from those chosen."""
  k = [int(np.argmin(np.hypot(p[:, 0] - p[:, 0].mean(), p[:, 1] - p[:, 1].mean())))]
  d = np.hypot(p[:, 0] - p[k[0], 0], p[:, 1] - p[k[0], 1])
  while len(k) < min(n, len(p)):
    k.append(int(np.argmax(d)))
    d = np.minimum(d, np.hypot(p[:, 0] - p[k[-1], 0], p[:, 1] - p[k[-1], 1]))
  return p[k]


def raster_segs(p: np.ndarray, cross: np.ndarray) -> list:
  """Stand-ins for the lower road's links where it has none (car parks and service ways under decks): through each
  crossing, RSEG_M along the hidden road's run there (its points' main axis within 3 m), at the raster's heights."""
  out = []
  for x, y, z, _ in cross:
    q = p[np.hypot(p[:, 0] - x, p[:, 1] - y) < 3.0]
    if len(q) < 8:
      continue
    w, v = np.linalg.eigh(np.cov((q[:, :2] - q[:, :2].mean(0)).T))
    ux, uy = v[:, 1] * RSEG_M / 2
    ends = []
    for ex, ey in ((x - ux, y - uy), (x + ux, y + uy)):
      e = p[np.hypot(p[:, 0] - ex, p[:, 1] - ey) < 0.75]
      ends.append((ex, ey, float(np.median(e[:, 2])) if len(e) else z))
    out.append([round(float(v), 2) for v in (*ends[0], *ends[1], RSEG_W)])
  return out


def block_keys(bx: int, by: int) -> list[tuple[int, int]]:
  return [(tx, ty) for tx in range(bx * BLOCK - 1, bx * BLOCK + BLOCK + 1) for ty in range(by * BLOCK - 1, by * BLOCK + BLOCK + 1)]


def plan_block(b: tuple[int, int]) -> tuple[dict, list]:
  """One block of BLOCK x BLOCK raster tiles (a tile of margin around it for the filters): the hidden road's areas by
  what becomes of it, and shots covering what's left to capture (only the block's own points counted)."""
  bx, by = b
  n = (BLOCK + 2) * G
  ar = {k: np.full((n, n), np.nan, np.float32) for k in ("z", "top", "over")}
  ar.update({k: np.zeros((n, n), bool) for k in ("seen", "deck", "rough", "fol", "shown")})
  for tx, ty in block_keys(bx, by):
    d = HID.get((tx, ty))
    if d is None:
      continue
    c, r = (tx - (bx * BLOCK - 1)) * G, ((by * BLOCK + BLOCK) - ty) * G
    for k in ("z", "top", "over", "seen", "deck", "rough"):
      ar[k][r:r + G, c:c + G] = d[k]
    ar["fol"][r:r + G, c:c + G] = foliage(d)
    if (tx, ty) in SHOWN:
      ar["shown"][r:r + G, c:c + G] = SHOWN[(tx, ty)]
  x0, y1 = (bx * BLOCK - 1) * rp.TILE, (by * BLOCK + BLOCK + 1) * rp.TILE  # the mosaic's west and north edges
  core = np.zeros((n, n), bool)
  core[G:-G, G:-G] = True
  hid = np.isfinite(ar["z"])
  if not (hid & core).any():
    return {}, []
  with np.errstate(invalid="ignore"):
    gap = hid & ~ar["shown"] & ~ar["fol"]
    low = gap & (ar["over"] - ar["z"] < LOW_M)
    far = cv2.distanceTransform((~ar["seen"]).astype(np.uint8), cv2.DIST_L2, 5) * RES
    tunnel = gap & ~low & (far > TUNNEL_M)
  cand = gap & ~low & ~tunnel
  deck = cand & ar["deck"]
  # under a road deck: patches 1 m wide and up; under anything else (viaducts, gantries, canopies, trees, buildings'
  # eaves), 2 m wide and up, and whole patches judged together
  opened = cv2.morphologyEx(deck.astype(np.uint8), cv2.MORPH_OPEN, np.ones((SLIVER, SLIVER), np.uint8)) > 0
  with np.errstate(invalid="ignore"):
    low |= cand & ~deck & (ar["over"] - ar["z"] < OTHER_LOW_M)
  cand &= ~low
  other = cv2.morphologyEx((cand & ~deck).astype(np.uint8), cv2.MORPH_OPEN, np.ones((OTHER_SLIVER, OTHER_SLIVER), np.uint8)) > 0
  nl, lab, st, _ = cv2.connectedComponentsWithStats(other.astype(np.uint8), connectivity=8)
  size = np.maximum(st[:, cv2.CC_STAT_AREA], 1)
  # a rough top over much of it: tree crowns. Much of it further than NARROW_M from road seen from above: a building's
  # floor (sheds, warehouses, car parks: interiors the camera doesn't draw) or a covered way
  trees = np.bincount(lab[other & ar["rough"]], minlength=nl) > ROUGH_SHARE * size
  wide = np.bincount(lab[other & (far > NARROW_M)], minlength=nl) > WIDE_SHARE * size
  # a road under a viaduct or gantry has the game's links along it; a building's floor (garages, warehouses, yards
  # under a roof) has none
  link = np.zeros((n, n), bool)
  x1, y0 = x0 + n * RES, y1 - n * RES
  for sx, sy, sz, ex, ey, ez, _ in LINKS[(np.maximum(LINKS[:, 0], LINKS[:, 3]) > x0) & (np.minimum(LINKS[:, 0], LINKS[:, 3]) < x1) &
                                         (np.maximum(LINKS[:, 1], LINKS[:, 4]) > y0) & (np.minimum(LINKS[:, 1], LINKS[:, 4]) < y1)]:
    t = np.linspace(0, 1, max(int(math.hypot(ex - sx, ey - sy) / RES), 1) + 1)
    c, r = ((sx + t * (ex - sx) - x0) / RES).astype(int), ((y1 - (sy + t * (ey - sy))) / RES).astype(int)
    k = (c >= 0) & (c < n) & (r >= 0) & (r < n)
    c, r, lz = c[k], r[k], (sz + t * (ez - sz))[k]
    with np.errstate(invalid="ignore"):
      on = np.abs(ar["z"][r, c] - lz) < 2.0
    link[r[on], c[on]] = True
  wide |= np.bincount(lab[other & link], minlength=nl) * RES < LINK_M
  small = size * RES * RES < MIN_OTHER_M2
  for v in (trees, wide, small):
    v[0] = False
  nd, labd, std, _ = cv2.connectedComponentsWithStats(opened.astype(np.uint8), connectivity=8)
  dsmall = std[:, cv2.CC_STAT_AREA] * RES * RES < MIN_GAP_M2
  dsmall[0] = True
  target = (opened & ~dsmall[labd]) | (other & ~(trees | wide | small)[lab])
  def area(m):
    return float((m & core).sum() * RES * RES)
  stats = {"hidden": area(hid), "shown": area(hid & ar["shown"]), "foliage": area(hid & ~ar["shown"] & ar["fol"]) + area(other & trees[lab]),
           "low": area(low), "tunnel": area(tunnel), "sliver": area(cand & ~opened & ~other),
           "small": area(opened & dsmall[labd]) + area(other & small[lab] & ~trees[lab] & ~wide[lab]),
           "wide_cover": area(other & wide[lab] & ~trees[lab]), "target": area(target), "target_deck": area(target & ar["deck"])}
  # greedy cover: each shot the centre and camera height covering the most of what's left (a point at camera height c
  # when its road is CAM_ABOVE under c and its lowest cover's top CAM_UNDER over it), within REACH of the centre
  rr, cc = np.nonzero(target & core)
  if not len(rr):
    stats["leftover"] = 0.0
    return stats, []
  z, ov = ar["z"][rr, cc], ar["over"][rr, cc]
  lo, hi = z + CAM_ABOVE, ov - CAM_UNDER
  cams = np.unique(np.round((lo + np.maximum(hi, lo)) / 2 / CAM_STEP) * CAM_STEP)
  live = np.ones(len(rr), bool)
  m = n // 4  # the score grid: 1 m
  gy, gx = rr // 4, cc // 4
  kx, ky = 2 * int(REACH_X) + 1, 2 * int(REACH_Y) + 1
  cache: dict[float, tuple[float, int, np.ndarray]] = {}

  def score(c: float):
    e = live & (lo - CAM_STEP / 2 <= c) & (c <= np.maximum(hi, lo) + CAM_STEP / 2)
    cnt = np.zeros((m, m), np.float32)
    np.add.at(cnt, (gy[e], gx[e]), 1.0)
    box = cv2.boxFilter(cnt, -1, (kx, ky), normalize=False, borderType=cv2.BORDER_CONSTANT)
    s = box + 0.01 * cv2.GaussianBlur(cnt, (0, 0), REACH_Y / 2)  # among equals, the centre nearest the patch's middle
    k = int(np.argmax(s))
    return float(box.flat[k]), k, e

  shots, left = [], 0.0
  while live.any():
    for c in cams:
      if c not in cache:
        cache[c] = score(c)
    c = max(cache, key=lambda v: cache[v][0])
    best, k, e = cache[c]
    if best * RES * RES < MIN_SHOT_M2:
      left = float(live.sum() * RES * RES)
      break
    cy, cx = divmod(k, m)
    px, py = x0 + (cx + 0.5) * 4 * RES, y1 - (cy + 0.5) * 4 * RES
    X, Y = x0 + (cc + 0.5) * RES, y1 - (rr + 0.5) * RES
    took = e & (np.abs(gx - cx) <= REACH_X) & (np.abs(gy - cy) <= REACH_Y)
    live &= ~took
    p = np.c_[X[took], Y[took], z[took], ov[took]]  # crossings as underplan's: x, y, the road, the top of what's over it
    cross = spread(p, CROSSINGS)
    shots.append({"x": round(px, 2), "y": round(py, 2), "camera": round(float(c), 2), "z_lo": round(float(z[took].min()), 2),
                  "z_hi": round(float(z[took].max()), 2), "gap_m2": round(float(took.sum() * RES * RES), 1),
                  "deck_share": round(float(ar["deck"][rr[took], cc[took]].mean()), 2),
                  "crossings": [[round(float(v), 2) for v in q] for q in cross], "raster_segs": raster_segs(p, cross)})
    for v in list(cache):  # rescore the camera heights that shared these points
      if (cache[v][2] & took).any():
        del cache[v]
  stats["leftover"] = left
  stats["shots"] = len(shots)
  return stats, shots


def plan(a) -> None:
  global HID, SHOWN, LINKS
  HID = load_hidden(a.out)
  blocks = sorted({(tx // BLOCK, ty // BLOCK) for tx, ty in HID if np.isfinite(HID[(tx, ty)]["z"]).any()})
  span, box = BLOCK * rp.TILE, None
  if a.box:
    x0, x1, y0, y1 = a.box
    blocks = [(bx, by) for bx, by in blocks if bx * span < x1 and (bx + 1) * span > x0 and by * span < y1 and (by + 1) * span > y0]
    box = (min(b[0] for b in blocks) * span - rp.TILE, (max(b[0] for b in blocks) + 1) * span + rp.TILE,
           min(b[1] for b in blocks) * span - rp.TILE, (max(b[1] for b in blocks) + 1) * span + rp.TILE)
  SHOWN = shown(a.shots, a.workers, box=box)
  LINKS = segs = road_links(a.paths)
  total, shots = collections.Counter(), []
  with mp.get_context("fork").Pool(a.workers) as pool:
    for st, sh in pool.imap_unordered(plan_block, blocks):
      total.update(st)
      shots += sh
  # per shot: its cell, the lower road's links on its level cut to its reach (checks a and b), a name
  per_cell = collections.Counter()
  for s in shots:
    s["i"], s["j"] = (int(v) for v in cells_of(np.array(s["x"]), np.array(s["y"])))
    x0, x1, y0, y1 = s["x"] - REACH_X, s["x"] + REACH_X, s["y"] - REACH_Y, s["y"] + REACH_Y
    near = segs[(np.maximum(segs[:, 0], segs[:, 3]) > x0) & (np.minimum(segs[:, 0], segs[:, 3]) < x1) &
                (np.maximum(segs[:, 1], segs[:, 4]) > y0) & (np.minimum(segs[:, 1], segs[:, 4]) < y1)]
    cut = [q for q in (clip(v, x0, x1, y0, y1) for v in near) if q]
    s["segs"] = [q for q in cut if min(q[2], q[5]) > s["z_lo"] - 1.5 and max(q[2], q[5]) < s["z_hi"] + 1.5]
    if not s["segs"]:  # no link on its level: the raster's road instead
      s["segs"], s["no_links"] = s["raster_segs"], True
    del s["raster_segs"]
  shots = [s for s in shots if s["segs"]]  # nothing to check a shot by: not planned
  order = {c: k for k, c in enumerate(ortho.blocks({(s["i"], s["j"]) for s in shots}))}
  shots.sort(key=lambda s: (order[(s["i"], s["j"])], s["camera"]))
  for s in shots:
    per_cell[(s["i"], s["j"])] += 1
    s["name"] = f"o{s['i']:03d}_{s['j']:03d}_G{per_cell[(s['i'], s['j'])]}"
    s["level"] = per_cell[(s["i"], s["j"])]
  report = {k: round(v, 1) for k, v in total.items()}
  report.update(shots=len(shots), shots_without_links=sum(bool(s.get("no_links")) for s in shots), est_hours=round(len(shots) * S_PER_SHOT / 3600, 1),
                shots_from=a.shots, box=a.box, reach_m=[REACH_X, REACH_Y])
  json.dump({"cm_per_px": ortho.CM_PER_PX, "near": ortho_depth.U_NEAR, "far": ortho_depth.U_FAR, "kind": "gaps", "report": report, "tiles": shots},
            open(f"{a.out}/plan.json", "w"))
  json.dump(report, open(f"{a.out}/report.json", "w"), indent=1)
  print(json.dumps(report, indent=1))


# ---------------------------------------------------------------------------------------------------------- measure
def measure(a) -> None:
  """The hidden road in a box by what shows it: the shots (green), only the extra shots (yellow), nothing (red; foliage
  dark green), the road the top imagery shows grey; a PNG at 25 cm/px, north up, with a plan's shots' reach in white.
  Areas printed and saved beside the PNG (.json)."""
  x0, x1, y0, y1 = a.box
  hid = load_hidden(a.out)
  sh = shown(a.shots, a.workers, box=a.box)
  ex = shown(a.extra, a.workers, box=a.box) if a.extra else {}
  W, H = round((x1 - x0) / RES), round((y1 - y0) / RES)
  img = np.zeros((H, W, 3), np.uint8)
  none = np.zeros((G, G), bool)
  area = collections.Counter()
  for key, d in hid.items():
    if not (key[0] * rp.TILE < x1 and (key[0] + 1) * rp.TILE > x0 and key[1] * rp.TILE < y1 and (key[1] + 1) * rp.TILE > y0):
      continue
    X, Y = grid_xy(*key)
    col, row = np.floor((X - x0) / RES).astype(int), np.floor((y1 - Y) / RES).astype(int)
    inb = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    h = np.isfinite(d["z"]) & inb
    s, e, f = sh.get(key, none), ex.get(key, none), foliage(d)
    cats = {"seen": d["seen"] & inb & ~h, "shown": h & s, "extra_only": h & ~s & e, "foliage": h & ~s & ~e & f, "neither": h & ~s & ~e & ~f}
    for name, colour in (("seen", (60, 60, 60)), ("shown", (0, 200, 0)), ("extra_only", (0, 200, 255)), ("foliage", (0, 90, 0)), ("neither", (0, 0, 255))):
      m = cats[name]
      img[row[m], col[m]] = colour
      if name != "seen":
        area[name] += float(m.sum() * RES * RES)
  area["hidden"] = sum(area.values())
  def p(x, y):
    return int((x - x0) / RES), int((y1 - y) / RES)
  if a.plan:
    for t in json.load(open(a.plan))["tiles"]:
      cv2.rectangle(img, p(t["x"] - REACH_X, t["y"] + REACH_Y), p(t["x"] + REACH_X, t["y"] - REACH_Y), (255, 255, 255), 1)
      cv2.circle(img, p(t["x"], t["y"]), 3, (255, 255, 255), -1)
  cv2.imwrite(a.png, img)
  out = {k: round(v) for k, v in area.items()}
  json.dump({"box": a.box, "shots": a.shots, "extra": a.extra, "m2": out}, open(os.path.splitext(a.png)[0] + ".json", "w"), indent=1)
  print(json.dumps(out))


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  h = sub.add_parser("hidden")
  h.add_argument("out")
  h.add_argument("--box", type=float, nargs=4, help="x0 x1 y0 y1: only the raster tiles touching this box")
  h.add_argument("--workers", type=int, default=12)
  h.add_argument("--resume", action="store_true")
  r = sub.add_parser("reach")
  r.add_argument("out")
  r.add_argument("--shots", nargs="+", default=[f"{F}/under/tiles"])
  r.add_argument("--workers", type=int, default=12)
  p = sub.add_parser("plan")
  p.add_argument("out")
  p.add_argument("--shots", nargs="+", default=SHOTS, help="the passing shots' tiles dirs: what they show isn't planned")
  p.add_argument("--box", type=float, nargs=4, help="x0 x1 y0 y1: only the blocks touching this box")
  p.add_argument("--paths", default=os.path.expanduser("~/gta5map/paths.jsonl"))
  p.add_argument("--workers", type=int, default=12)
  m = sub.add_parser("measure")
  m.add_argument("out")
  m.add_argument("png")
  m.add_argument("--box", type=float, nargs=4, required=True, help="x0 x1 y0 y1")
  m.add_argument("--shots", nargs="+", default=SHOTS)
  m.add_argument("--extra", nargs="*", default=[], help="more shots' dirs, counted apart (yellow)")
  m.add_argument("--plan", help="a plan.json whose shots' reach is drawn")
  m.add_argument("--workers", type=int, default=12)
  a = ap.parse_args()
  if a.cmd == "hidden":
    hidden(a)
  elif a.cmd == "reach":
    reach(a)
  elif a.cmd == "plan":
    plan(a)
  elif a.cmd == "measure":
    measure(a)


if __name__ == "__main__":
  main()
