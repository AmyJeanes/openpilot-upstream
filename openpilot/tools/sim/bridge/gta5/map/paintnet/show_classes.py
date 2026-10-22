"""show_classes.py regions.npz rstats.json out.png: regions coloured by axes.py's class, ids at their centres; walls
white/yellow, other road grey. Left the top surface at each pixel, right the one below it."""
import json, sys
import numpy as np
import cv2
from scipy import ndimage as ndi

R = np.load(sys.argv[1])
lab, z, wall, cls = R["lab"], R["z"], R["wall"], R["cls"]
src = open(sys.path[0] + "/axes.py").read()
ns = {}
exec(src[src.index("CELL_AREA,"):src.index("R = np.load")], ns)
exec(src[src.index("def classify"):src.index("def lonlat")], ns)
stats = {o["id"]: o for o in json.load(open(sys.argv[2]))}
COL = {"lane": (80, 180, 60), "area": (200, 80, 200), "open": (40, 40, 220), "shoulder": (200, 160, 60),
       "median": (0, 140, 255), "cell": (120, 120, 120)}
pal = np.zeros((lab.max() + 1, 3), np.uint8)
for k, o in stats.items():
  pal[k] = COL[ns["classify"](o)]
L, H, W = lab.shape
zk = np.where(np.isnan(z), -np.inf, z)
order = np.argsort(-zk, axis=0)
objs = ndi.find_objects(lab)
ims = []
for rank in (0, 1):
  li = order[rank]
  pick = lambda a: np.take_along_axis(a, li[None], 0)[0]
  lb, wl, cl, valid = pick(lab), pick(wall), pick(cls), pick(zk) > -np.inf
  im = np.full((H, W, 3), 0, np.uint8)
  im[valid & (cl == 1)] = (70, 70, 70)
  im[valid & (cl > 1)] = (35, 35, 35)
  im[lb > 0] = pal[lb[lb > 0]]
  im[wl == 1] = (255, 255, 255)
  im[wl == 2] = (0, 220, 255)
  for k, o in stats.items():
    if o["area"] < 15:
      continue
    m = lb == k
    if m.sum() < 300:
      continue
    d = ndi.distance_transform_edt(m)
    r, c = np.unravel_index(d.argmax(), d.shape)
    cv2.putText(im, str(k), (int(c) - 12, int(r) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(im, str(k), (int(c) - 12, int(r) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
  ims.append(im)
cv2.imwrite(sys.argv[3], np.vstack([ims[0], np.full((8, W, 3), 255, np.uint8), ims[1]]))
