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
# the markings list misses many arrows: paint shaped like one is masked too
sys.path.insert(0, sys.path[0])
import arrows
masked |= arrows.find((paint > 0) & ~masked, zz, x0, y1, RES)[0]
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

# a line's end carries on along its own direction across the gap to the next paint, so lanes don't join through it:
# to the next dash of the same line within JOIN_DASH m (gaps.py on the interchange: dash gaps 7-8 m, none 9-20 m),
# to any other paint or the road's edge within JOIN_NEAR m (line ends before a crossing or stop line: up to 2.5 m,
# a dip at 2.5-3 m). Ends are skeleton ends of paint no wider than LINE_W; direction from the last END_RUN m.
JOIN_DASH, JOIN_NEAR, LINE_W, END_RUN, SAME_DIR = 9.0, 2.5, 0.6, 1.5, math.cos(math.radians(15))
sys.path.insert(0, sys.path[0])
from skel import thin, degree, walk
paintw = wall_col > 0
ends = []  # (layer, row, col, unit dir (dr, dc), colour)
for l in range(L):
  sk = thin(paintw[l])
  dist = ndi.distance_transform_edt(paintw[l])
  for r, c in zip(*np.nonzero(degree(sk) == 1)):
    p = walk(sk, (r, c), int(END_RUN / RES))
    if len(p) < int(END_RUN / RES) // 2:
      continue
    pr, pc = np.array(p).T
    if 2 * np.median(dist[pr, pc]) * RES > LINE_W:
      continue
    d = np.array([r - pr[-1], c - pc[-1]], float)
    ends.append((l, r, c, d / np.linalg.norm(d), int(wall_col[l, r, c])))
end_at = {}
for i, e in enumerate(ends):
  end_at.setdefault((e[0], e[1] // 5, e[2] // 5), []).append(i)


def facing(l, r, c, d):
  """an end of another line near (r, c) on layer l pointing back along d: the next dash."""
  for dr in (-1, 0, 1):
    for dc in (-1, 0, 1):
      for i in end_at.get((l, r // 5 + dr, c // 5 + dc), ()):
        e = ends[i]
        if abs(e[1] - r) <= 6 and abs(e[2] - c) <= 6 and e[3] @ d < -SAME_DIR:
          return True
  return False


joined = 0
for l, r, c, d, colour in ends:
  zr = zz[l, r, c]
  path, left, la = [], False, l
  for i in range(1, int(2 * JOIN_DASH / RES)):
    fr, fc = r + d[0] * i / 2, c + d[1] * i / 2
    rr, cc = int(round(fr)), int(round(fc))
    if not (0 <= rr < H and 0 <= cc < W):
      break
    dz = np.abs(zz[:, rr, cc] - zr)
    dz = np.where(np.isnan(dz), np.inf, dz)
    la = int(dz.argmin())
    gone = dz[la] > DZ * 2 or cls[la, rr, cc] != 1
    hit = not gone and paintw[la, rr, cc]
    if not left:
      left = not hit
      if gone:
        break
      continue
    run = i / 2 * RES
    if gone or hit:
      if run <= JOIN_NEAR or (hit and facing(la, rr, cc, d)):
        for pl, pr_, pc_ in path:
          wall_col[pl, pr_, pc_] = max(wall_col[pl, pr_, pc_], colour)
        joined += 1
      break
    zr = zz[la, rr, cc]
    path.append((la, rr, cc))
print(f"{len(ends)} line ends, {joined} carried across a gap")
wall = wall_col > 0
free = present & (cls == 1) & ~wall
# lane space reaches at most LANE_REACH m from paint (lane regions measured 5-7.5 m wide, each beside a painted line):
# unmarked surface beyond (a plaza, driveways, a car park joined to a street with no kerb) is cut away so lanes don't
# leak into it; it stays road, just not lane space
LANE_REACH = 7.5
# (paint on the same surface: layers hold whichever surfaces a pixel has, so a layer can switch between a street and the
# deck above it; the nearest paint on each layer must be within PAINT_DZ m of the pixel's height)
PAINT_DZ = 1.0
near_paint = np.zeros_like(free)
for l2 in range(L):
  d, (ir, ic) = ndi.distance_transform_edt(~wall[l2], return_indices=True)
  zw = zz[l2, ir, ic]
  for l in range(L):
    near_paint[l] |= (d * RES <= LANE_REACH) & (np.abs(zz[l] - zw) < PAINT_DZ)
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
