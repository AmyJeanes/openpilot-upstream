"""rtab.py regions.npz rstats.json: every region over 15 m2 with its class, measures and centre (game x, y)."""
import json, sys
import numpy as np
from scipy import ndimage as ndi

R = np.load(sys.argv[1])
lab, res, x0, y1 = R["lab"], float(R["res"]), float(R["x0"]), float(R["y1"])
src = open(sys.path[0] + "/axes.py").read()
ns = {}
exec(src[src.index("CELL_AREA,"):src.index("R = np.load")], ns)
exec(src[src.index("def classify"):src.index("def lonlat")], ns)
objs = ndi.find_objects(lab)
rows = []
for o in json.load(open(sys.argv[2])):
  if o["area"] < 15:
    continue
  k = o["id"]
  sl = objs[k - 1]
  rr, cc = np.nonzero((lab[sl] == k).any(0))
  x = x0 + (cc.mean() + sl[2].start + .5) * res
  y = y1 - (rr.mean() + sl[1].start + .5) * res
  rows.append((ns["classify"](o), k, o["area"], o["widest"], o["white"], o["yellow"], o["edge"], o["other"], o["z"], round(x), round(y)))
print("class id area widest white yellow edge other z x y")
for r in sorted(rows):
  print(*r)
