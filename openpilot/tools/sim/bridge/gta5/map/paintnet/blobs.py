"""blobs.py x0 x1 y0 y1: raster paint pieces by shape (length and width along/across their main axis), each marked
whether a markings-list arrow or text sits on it, to tell arrows from line paint by shape.
From gaps.py: x0 x1 y0 y1: how far each painted stroke's end is from the next paint straight ahead, to set how far a
line is carried across a gap. Strokes: raster paint pieces (per surface layer, arrows/text masked) that are long and
thin. From each end a ray runs on along the stroke's own direction over road at the stroke's height; the first paint
it meets is a hit, collinear when that piece is a stroke running the same way (a dashed line's next dash)."""
import json, math, sys
import numpy as np
import cv2
sys.path.insert(0, "/home/amy/git/openpilot-slowroads/openpilot/tools/sim/bridge/gta5/map/roadpaint")
import rp

TILES = "/mnt/e/gta5_scratch/gamefiles_v2/rp_all/tiles"
FEATS = "/mnt/e/gta5_scratch/gamefiles_v2/rp_all/features.jsonl"
RES = 0.1
MASKED = ("arrow", "text_", "parking_marker", "disabled", "pedestrian", "bicycle", "keep_clear")
x0, x1, y0, y1 = map(float, sys.argv[1:5])
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
cls, paint = code & 7, code >> 3
masked = np.zeros((L, H, W), bool)
for line in open(FEATS):
  f = json.loads(line)
  if not f["kind"].startswith(MASKED) or not (x0 - 10 <= f["x"] <= x1 + 10 and y0 - 10 <= f["y"] <= y1 + 10):
    continue
  r = max(f.get("len", 0), f.get("wid", 0)) / 2 + 0.3
  disc = np.zeros((H, W), np.uint8)
  cv2.circle(disc, (int(round((f["x"] - x0) / RES)), int(round((y1 - f["y"]) / RES))), int(math.ceil(r / RES)), 1, -1)
  for l in range(L):
    masked[l] |= (disc > 0) & (np.abs(zz[l] - f["z"]) < 1.5)
pnt = paint > 0  # arrows left in, to measure them

strokes = []  # (layer, comp, cx, cy, ux, uy, half_len, width) in pixels
comps = []
for l in range(L):
  n, cl, st, _ = cv2.connectedComponentsWithStats(pnt[l].astype(np.uint8), connectivity=8)
  comps.append(cl)
  for k in range(1, n):
    if st[k, cv2.CC_STAT_AREA] < 10:
      continue
    x, y, w, h = st[k, :4]
    rr, cc = np.nonzero(cl[y:y + h, x:x + w] == k)
    P = np.stack([cc + x, rr + y], 1).astype(float)
    m = P.mean(0)
    ev, evec = np.linalg.eigh(np.cov((P - m).T))
    u = evec[:, 1]
    t = (P - m) @ u
    s = (P - m) @ evec[:, 0]
    ln, wd = (t.max() - t.min()) * RES, (s.max() - s.min()) * RES
    strokes.append((l, k, m[0], m[1], u[0], u[1], t.min(), t.max(), ln, wd, bool(masked[l][cl == k].mean() > 0.5), st[k, cv2.CC_STAT_AREA] * RES * RES))
S = np.array([s[8] for s in strokes]), np.array([s[9] for s in strokes])
print(len(strokes), "pieces; length m:", np.histogram(S[0], [0, .5, 1, 2, 3, 4, 6, 10, 20, 50, 1e3])[0].tolist(),
      "width m:", np.histogram(S[1], [0, .1, .2, .3, .4, .6, 1, 2, 1e3])[0].tolist())
for s in sorted(strokes, key=lambda s: s[8]):
  if 3 <= s[8] <= 9 and s[9] >= 0.5:
    print(f"len {s[8]:5.2f} wid {s[9]:4.2f} area {s[11]:5.2f} fill {s[11] / (s[8] * s[9]):.2f} listed {s[10]} at {x0 + s[2] * RES:.1f},{y1 - s[3] * RES:.1f} L{s[0]}")
