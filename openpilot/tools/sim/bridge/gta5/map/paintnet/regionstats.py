"""regionstats.py regions.npz out.json: what bounds and fills each region, to classify them from measured values.

Per region: area (m2), height, and around its edge (pixels just outside it on the same surface, within 0.5 m in
height): shares that are white wall, yellow wall, the road's edge (kerb, pavement, gutter or no surface) and other
free road (the same space carrying on across a surface seam). Also the widest point (2x the largest distance from
its edge, m) and its length (pixels of its medial axis, roughly)."""
import json, sys
import numpy as np
from scipy import ndimage as ndi

R = np.load(sys.argv[1])
lab, z, wall, cls = R["lab"], R["z"], R["wall"], R["cls"]
res = float(R["res"])
L, H, W = lab.shape
n = int(lab.max())
objs = ndi.find_objects(lab)
out = []
for k in range(1, n + 1):
  sl = objs[k - 1]
  if sl is None:
    continue
  # a margin around the region's box for its edge
  s = (slice(None), slice(max(sl[1].start - 2, 0), sl[1].stop + 2), slice(max(sl[2].start - 2, 0), sl[2].stop + 2))
  m = lab[s] == k
  area = int(m.sum()) * res * res
  if area < 0.5:
    continue
  zl = z[s]
  zr = float(np.nanmean(zl[m]))
  foot = m.any(0)
  ring = ndi.binary_dilation(foot) & ~foot
  # what lies on the ring, on the surface nearest the height of the region beside it (not the region's mean: ramps
  # climb metres along their length)
  zk = np.nanmean(np.where(m, zl, np.nan), 0)
  _, (ir, ic) = ndi.distance_transform_edt(~foot, return_indices=True)
  zz = np.where(np.isnan(zl), np.inf, np.abs(zl - zk[ir, ic][None]))
  li = zz.argmin(0)
  pick = lambda a: np.take_along_axis(a, li[None], 0)[0]
  near = pick(zz) < 0.5
  wl, cl, lb = pick(wall[s]), pick(cls[s]), pick(lab[s])
  rw = ring & near
  tot = max(int(ring.sum()), 1)
  white = int((rw & (wl == 1)).sum()) / tot
  yellow = int((rw & (wl == 2)).sum()) / tot
  edge = int((ring & (~near | ((cl != 1) & (wl == 0)))).sum()) / tot
  other = int((rw & (wl == 0) & (lb > 0) & (lb != k)).sum()) / tot
  widest = 2 * float(ndi.distance_transform_edt(foot).max()) * res
  out.append({"id": k, "area": round(area, 1), "z": round(zr, 1), "white": round(white, 2), "yellow": round(yellow, 2),
              "edge": round(edge, 2), "other": round(other, 2), "widest": round(widest, 2)})
json.dump(out, open(sys.argv[2], "w"))
A = np.array([o["area"] for o in out])
h, b = np.histogram(A, bins=[0.5, 1, 2, 4, 8, 15, 30, 60, 120, 250, 500, 1000, 1e6])
print(len(out), "regions; area m2:", " ".join(f"{b[i]:g}-{b[i+1]:g}:{h[i]}" for i in range(len(h))))
Wd = np.array([o["widest"] for o in out])
h, b = np.histogram(Wd, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 30, 100])
print("widest m:", " ".join(f"{b[i]:g}-{b[i+1]:g}:{h[i]}" for i in range(len(h))))
for o in sorted(out, key=lambda o: -o["area"])[:30]:
  print(o)
