"""fill_cells.py <out dir>: tiles for the grid cells the capture skipped as open water (the middle of the Alamo Sea, the
port basins, Lago Zancudo, the Land Act reservoir): cells missing from the grid but with captured cells on both sides
both ways or cut off from the grid's edge, painted the deep colour of the nearest body of water (bodies.npz, within 300 m). Writes <out>/oIII_JJJ.png
(2560x1440, the capture size) and cells.json."""
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

S = os.path.dirname(os.path.abspath(__file__))
F = "/mnt/e/gta5_ortho_20261018"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
P = json.load(open(f"{S}/params.json"))["bodies"]
z = np.load(os.environ.get("BODIES", "/mnt/e/gta5_audit/water_full_20261019/bodies.npz"))
body, X0, Y0 = z["body"], int(z["x0"]), int(z["y0"])
names = json.load(open(os.path.join(os.path.dirname(os.environ.get("BODIES", "/mnt/e/gta5_audit/water_full_20261019/bodies.npz")), "bodies.json")))["bodies"]
cells = {(int(f[1:4]), int(f[5:8])) for f in os.listdir(f"{F}/full/tiles") if f.endswith(".json")}
I, J = [c[0] for c in cells], [c[1] for c in cells]
miss = []
for i in range(min(I), max(I) + 1):
  for j in range(min(J), max(J) + 1):
    if (i, j) in cells:
      continue
    lr = any((i - d, j) in cells for d in range(1, 15)) and any((i + d, j) in cells for d in range(1, 15))
    ud = any((i, j - d) in cells for d in range(1, 25)) and any((i, j + d) in cells for d in range(1, 25))
    if lr and ud:
      miss.append((i, j))
# and holes too wide for that (the Alamo Sea): missing cells not connected to the grid's edge
grid = np.zeros((max(J) - min(J) + 3, max(I) - min(I) + 3), bool)  # True = missing, one cell of border
for i in range(min(I) - 1, max(I) + 2):
  for j in range(min(J) - 1, max(J) + 2):
    grid[j - min(J) + 1, i - min(I) + 1] = (i, j) not in cells
lab, _ = ndi.label(grid)
outside = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
inner = {(i, j) for j, i in zip(*np.nonzero(grid & ~np.isin(lab, list(outside))))}
inner_cells = {(int(i) + min(I) - 1, int(j) + min(J) - 1) for i, j in inner}
miss = sorted(set(miss) | inner_cells)
# the nearest water point's body, for every grid point
idx = ndi.distance_transform_edt(body == 0, return_distances=False, return_indices=True)
dist = ndi.distance_transform_edt(body == 0)
out, skipped = [], []
for i, j in miss:
  x, y = -4300 + 69.12 * i, -4300 + 38.88 * j
  gj, gi = int(y - Y0), int(x - X0)
  if dist[gj, gi] > (3000 if (i, j) in inner_cells else 300):  # a cell deep inside a lake is still that lake
    skipped.append([i, j])
    continue
  b = names[body[idx[0][gj, gi], idx[1][gj, gi]]]
  t = P[b]["teal"]
  rgb = [int(t[k:k + 2], 16) for k in (0, 2, 4)]
  Image.new("RGB", (2560, 1440), tuple(rgb)).save(f"{OUT}/o{i:03d}_{j:03d}.png")
  out.append({"cell": [i, j], "body": b, "dist_m": round(float(dist[gj, gi]), 1)})
json.dump({"filled": out, "skipped_no_water_near": skipped}, open(f"{OUT}/cells.json", "w"), indent=1)
from collections import Counter
print(len(out), "filled", Counter(o["body"] for o in out), "|", len(skipped), "skipped (no water within 300 m)")
