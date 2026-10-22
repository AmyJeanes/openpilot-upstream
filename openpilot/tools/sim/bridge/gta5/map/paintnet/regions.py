"""regions.py x0 x1 y0 y1 out.npz: the road surface cut into regions by its paint, level by level.

The game files' surface raster (roadpaint tiles: per 5 cm pixel up to 3 surfaces at different heights, each its class
and paint) is read at RES m. Walls: every painted pixel (lines, chevrons, crossings; arrows and text are masked out with
the markings list so they don't hole lanes) plus every lengthwise paint line drawn whole (closing a dashed line's gaps),
each on the surface its height matches. Free space: road surface that isn't wall. Regions: free space joined between
neighbouring pixels on surfaces within DZ m in height, so a road passing under a deck stays apart from it.
Out: labels per pixel and surface layer (0 = none), each layer's height, and the wall paint colour per pixel/layer."""
import json, math, sys
import numpy as np
import cv2
from scipy import sparse
from scipy import ndimage as ndi
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, "/home/amy/git/openpilot-slowroads/openpilot/tools/sim/bridge/gta5/map/roadpaint")
import rp

TILES = "/mnt/e/gta5_scratch/gamefiles_v2/rp_all/tiles"
PAINT = "/mnt/e/gta5_scratch/gamefiles_v2/rp_all/polylines.jsonl"
FEATS = "/mnt/e/gta5_scratch/gamefiles_v2/rp_all/features.jsonl"
RES, DZ, LINE_ZWIN = 0.1, 0.25, 0.6
MASKED = ("arrow", "text_", "parking_marker", "disabled", "pedestrian", "bicycle", "keep_clear")

x0, x1, y0, y1 = map(float, sys.argv[1:5])
out = sys.argv[5]
W, H = int(round((x1 - x0) / RES)), int(round((y1 - y0) / RES))
xs = x0 + (np.arange(W) + 0.5) * RES
ys = y1 - (np.arange(H) + 0.5) * RES
L = 3
code = np.zeros((L, H, W), np.uint8)
zz = np.full((L, H, W), np.nan, np.float32)
for tx in range(int(math.floor(x0 / rp.TILE)), int(math.floor(x1 / rp.TILE)) + 1):
  for ty in range(int(math.floor(y0 / rp.TILE)), int(math.floor(y1 / rp.TILE)) + 1):
    try:
      c, z = rp.read_tile(f"{TILES}/{tx}_{ty}.bin")
    except FileNotFoundError:
      continue
    mc = (xs >= tx * rp.TILE) & (xs < tx * rp.TILE + rp.TILE)
    mr = (ys >= ty * rp.TILE) & (ys < ty * rp.TILE + rp.TILE)
    col = ((xs[mc] - tx * rp.TILE) / rp.RES).astype(int)
    row = ((ty * rp.TILE + rp.TILE - ys[mr]) / rp.RES).astype(int)
    for l in range(c.shape[0]):
      code[l][np.ix_(mr, mc)] = c[l][np.ix_(row, col)]
      zz[l][np.ix_(mr, mc)] = z[l][np.ix_(row, col)]
cls = code & 7
paint = code >> 3          # 1 white, 2 yellow
present = cls > 0
zz[~present] = np.nan

# arrows and text are paint inside lanes, not walls
feats = [json.loads(l) for l in open(FEATS)]
masked = np.zeros((L, H, W), bool)
for f in feats:
  if not f["kind"].startswith(MASKED) or not (x0 - 10 <= f["x"] <= x1 + 10 and y0 - 10 <= f["y"] <= y1 + 10):
    continue
  r = max(f.get("len", 0), f.get("wid", 0)) / 2 + 0.3
  c0, r0 = (f["x"] - x0) / RES, (y1 - f["y"]) / RES
  disc = np.zeros((H, W), np.uint8)
  cv2.circle(disc, (int(round(c0)), int(round(r0))), int(math.ceil(r / RES)), 1, -1)
  for l in range(L):
    masked[l] |= (disc > 0) & (np.abs(zz[l] - f["z"]) < 1.5)
wall_col = np.where(masked, 0, paint)

# lengthwise lines drawn whole, on the surface whose height matches (a dashed line's gaps are walls too)
for line in open(PAINT):
  r = json.loads(line)
  P = np.array(r["pts"], float)
  if not ((P[:, 0] >= x0 - 5) & (P[:, 0] <= x1 + 5) & (P[:, 1] >= y0 - 5) & (P[:, 1] <= y1 + 5)).any():
    continue
  colour = 2 if r["colour"] == "yellow" else 1
  for (ax, ay, az), (bx, by, bz) in zip(P[:-1], P[1:]):
    n = max(2, int(math.hypot(bx - ax, by - ay) / (RES / 2)))
    t = np.linspace(0, 1, n)
    cc = np.round((ax + (bx - ax) * t - x0) / RES - 0.5).astype(int)
    rr = np.round((y1 - (ay + (by - ay) * t)) / RES - 0.5).astype(int)
    pz = az + (bz - az) * t
    ok = (cc >= 0) & (cc < W) & (rr >= 0) & (rr < H)
    cc, rr, pz = cc[ok], rr[ok], pz[ok]
    for l in range(L):
      on = np.abs(zz[l, rr, cc] - pz) < LINE_ZWIN
      wall_col[l, rr[on], cc[on]] = np.maximum(wall_col[l, rr[on], cc[on]], colour)
wall = wall_col > 0
free = present & (cls == 1) & ~wall
# lane space reaches at most LANE_REACH m from paint (lane regions measured 5-7.5 m wide, each beside a painted line):
# unmarked surface beyond (a plaza, driveways, a car park joined to a street with no kerb) is cut away so lanes don't
# leak into it; it stays road, just not lane space
LANE_REACH = 7.5
near_paint = np.zeros_like(free)
for l in range(L):
  near_paint[l] = ndi.distance_transform_edt(~wall[l]) * RES <= LANE_REACH
offlane = free & ~near_paint
free &= near_paint

# regions: free pixels joined to 4-neighbours on surfaces within DZ in height
N = L * H * W
idx = np.arange(N).reshape(L, H, W)
rows, cols = [], []
for dr, dc in ((0, 1), (1, 0)):
  for la in range(L):
    for lb in range(L):
      a = free[la, :H - dr, :W - dc]
      b = free[lb, dr:, dc:]
      m = a & b & (np.abs(zz[la, :H - dr, :W - dc] - zz[lb, dr:, dc:]) < DZ)
      rows.append(idx[la, :H - dr, :W - dc][m])
      cols.append(idx[lb, dr:, dc:][m])
rows, cols = np.concatenate(rows), np.concatenate(cols)
g = sparse.coo_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(N, N))
n, lab = connected_components(g, directed=False)
lab = lab.reshape(L, H, W) + 1
lab[~free] = 0
# number regions by size, small ids for big regions; drop specks
u, cnt = np.unique(lab[lab > 0], return_counts=True)
order = np.argsort(-cnt)
remap = np.zeros(lab.max() + 1, np.int32)
remap[u[order]] = np.arange(1, len(u) + 1)
lab = remap[lab]
np.savez_compressed(out, lab=lab, z=zz, wall=wall_col, cls=cls, offlane=offlane, x0=x0, y1=y1, res=RES)
print(f"{W}x{H} px, {int(free.sum())} free px, {len(u)} regions; over 2 m2: {int((cnt * RES * RES > 2).sum())}; over 20 m2: {int((cnt * RES * RES > 20).sum())}")
