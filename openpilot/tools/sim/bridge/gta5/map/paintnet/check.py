"""check.py regions.npz rstats.json groups.json dir.json out.geojson [out.png]: the pilot's lanes checked against
regions.py's paint regions, which come from the surface raster rather than the line list, so they fail differently.

Flags (points, with why and the lane or region named):
  unmapped: lane space (a lane or area region) not under any pilot feature, at least UNMAPPED_A m2 and UNMAPPED_W m wide
  crosses: a lane with paint inside it running its way (a line it should stop at) for CROSS_RUN m
  offroad: a lane over kerb, pavement or no surface at its height for OFF_RUN m
  overlap: two lanes on the same ground for more than OVERLAP_A m2 (band ends and bends leave up to ~8 m2)
Each lane is sampled every metre: across its middle INNER of its width, against the surface layer nearest its height."""
import json, math, os, sys
import numpy as np
import cv2
from scipy import ndimage as ndi
sys.path.insert(0, "/home/amy/git/op-ortho/openpilot/tools/sim/bridge/gta5/map/webmap")
import gframe

UNMAPPED_A, UNMAPPED_W, CROSS_RUN, OFF_RUN, OVERLAP_A, INNER, ZTOL = 5.0, 1.5, 3.0, 3.0, 20.0, 0.6, 1.0
R = np.load(sys.argv[1])
lab, z, wall, cls = R["lab"], R["z"], R["wall"], R["cls"]
res, x0, y1 = float(R["res"]), float(R["x0"]), float(R["y1"])
L, H, W = lab.shape
src = open(os.path.dirname(os.path.abspath(__file__)) + "/axes.py").read()
ns = {}
exec(src[src.index("CELL_AREA,"):src.index("R = np.load")], ns)
exec(src[src.index("def classify"):src.index("def lonlat")], ns)
kind = {o["id"]: ns["classify"](o) for o in json.load(open(sys.argv[2]))}
G = json.load(open(sys.argv[3]))
others = [r for r in json.load(open(sys.argv[4])) if r["kind"] != "lane"]
zk = np.where(np.isnan(z), np.inf, z)


def px(x, y):
  return (x - x0) / res - 0.5, (y1 - y) / res - 0.5


def layer_at(r, c, zq):
  d = np.abs(zk[:, r, c] - zq)
  l = int(d.argmin())
  return l if d[l] < ZTOL else -1


def paint(A, into, value=1):
  """fill a run's band (centre +- half width, sample by sample) into per-layer masks, on the layer nearest its height."""
  for (ax, ay, az, aw), (bx, by, bz, bw) in zip(A[:-1], A[1:]):
    d = np.array([bx - ax, by - ay]); n = np.linalg.norm(d)
    if n < 1e-6:
      continue
    d /= n
    nn = np.array([d[1], -d[0]])
    P = [np.array([ax, ay]) - d * 0.5 + nn * aw / 2, np.array([bx, by]) + d * 0.5 + nn * bw / 2,
         np.array([bx, by]) + d * 0.5 - nn * bw / 2, np.array([ax, ay]) - d * 0.5 - nn * aw / 2]
    poly = np.array([px(*p) for p in P])
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    rr, cc = np.nonzero(m)
    if not len(rr):
      continue
    dz = np.abs(zk[:, rr, cc] - (az + bz) / 2)
    l = dz.argmin(0)
    ok = dz[l, np.arange(len(l))] < ZTOL
    np.add.at(into, (l[ok], rr[ok], cc[ok]), value)


flags = []


def flag(why, x, y, name):
  flags.append({"why": why, "x": round(float(x), 1), "y": round(float(y), 1), "name": name})


# where the pilot puts something, and how many lanes on each pixel
lanes_on = np.zeros((L, H, W), np.uint8)
anything = np.zeros((L, H, W), np.uint8)
for k, r in enumerate(G["lanes"]):
  A = np.array(r["pts"])
  one = np.zeros_like(lanes_on)  # a lane's own samples overlap by design: count each lane once per pixel
  paint(A, one)
  lanes_on += one > 0
  paint(A, anything)
for r in others:
  paint(np.array(r["pts"]), anything)

# unmapped lane space
space = np.isin(lab, [k for k, v in kind.items() if v in ("lane", "area")])
covered = ndi.binary_dilation(anything > 0, np.ones((1, 7, 7), bool))
gap = space & ~covered
for l in range(L):
  n, cl = cv2.connectedComponents(gap[l].astype(np.uint8), connectivity=4)
  for k in range(1, n):
    m = cl == k
    a = m.sum() * res * res
    if a < UNMAPPED_A:
      continue
    d = ndi.distance_transform_edt(m)
    if 2 * d.max() * res < UNMAPPED_W:
      continue
    r, c = np.unravel_index(d.argmax(), d.shape)
    reg = int(np.bincount(lab[l][m]).argmax())
    flag(f"unmapped: {a:.0f} m2 of lane space (region {reg}, {kind.get(reg)}) with no lane", x0 + (c + .5) * res, y1 - (r + .5) * res, f"region {reg}")

# overlaps
n, cl = cv2.connectedComponents(((lanes_on > 1).any(0)).astype(np.uint8))
for k in range(1, n):
  m = cl == k
  if m.sum() * res * res > OVERLAP_A:
    r, c = np.argwhere(m).mean(0)
    flag(f"overlap: lanes share {m.sum() * res * res:.0f} m2", x0 + (c + .5) * res, y1 - (r + .5) * res, "")

# lines inside lanes, lanes off the road
for k, rcd in enumerate(G["lanes"]):
  A = np.array(rcd["pts"])
  lid = f"{rcd['left']}|{rcd['right']}|{k}"
  d = np.gradient(A[:, :2], axis=0)
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  nrm = np.stack([d[:, 1], -d[:, 0]], 1)
  offs = np.linspace(-INNER / 2, INNER / 2, 13)  # fractions of the width across the lane's middle
  hit = np.zeros((len(A), len(offs)), bool)
  off = np.zeros(len(A), bool)
  for i, (x, y, zq, w) in enumerate(A):
    for j, f in enumerate(offs):
      c, r = px(x + nrm[i, 0] * f * w, y + nrm[i, 1] * f * w)
      r, c = int(round(r)), int(round(c))
      if not (0 <= r < H and 0 <= c < W):
        continue
      l = layer_at(r, c, zq)
      if l < 0 or cls[l, r, c] != 1:
        if j == len(offs) // 2:
          off[i] = True
        continue
      # a line is 0.1-0.4 m wide: look a pixel either side square to the lane so a thin line isn't stepped over
      for s in (-1, 0, 1):
        rs, cs = int(round(r - nrm[i, 1] * s)), int(round(c + nrm[i, 0] * s))
        if 0 <= rs < H and 0 <= cs < W and wall[l, rs, cs] > 0:
          hit[i, j] = True
  # paint running the lane's way: the same place across it hit on CROSS_RUN m of samples in a row (a stop line or
  # crossing is hit across the whole width, but on one or two samples only)
  run = np.zeros(len(offs), int)
  best = np.zeros(len(offs), int)
  at = np.zeros(len(offs), int)
  for i in range(len(A)):
    run = np.where(hit[i], run + 1, 0)
    up = run > best
    best = np.where(up, run, best)
    at = np.where(up, i, at)
  j = int(best.argmax())
  if best[j] >= CROSS_RUN:
    i = at[j] - best[j] // 2
    flag(f"crosses: paint runs {best[j]} m inside the lane", A[i, 0], A[i, 1], lid)
  r_, best_off, at_off = 0, 0, 0
  for i in range(len(A)):
    r_ = r_ + 1 if off[i] else 0
    if r_ > best_off:
      best_off, at_off = r_, i
  if best_off >= OFF_RUN:
    i = at_off - best_off // 2
    flag(f"offroad: lane centre off the road for {best_off} m", A[i, 0], A[i, 1], lid)

feats = []
for f in flags:
  lat, lon = gframe.lat_lon(np.array([f["x"]]), np.array([f["y"]]))
  feats.append({"type": "Feature", "properties": {"kind": "check", **f}, "geometry": {"type": "Point", "coordinates": [round(float(lon[0]), 8), round(float(lat[0]), 8)]}})
json.dump({"type": "FeatureCollection", "features": feats}, open(sys.argv[5], "w"))
for f in flags:
  lat, lon = gframe.lat_lon(np.array([f["x"]]), np.array([f["y"]]))
  print(f"{f['why']:60s} {f['name']:22s} ({f['x']}, {f['y']})  #21/{lat[0]:.7f}/{lon[0]:.7f}")
print(len(flags), "flags ->", sys.argv[5])

if len(sys.argv) > 6:
  order = np.argsort(-np.where(np.isnan(z), -np.inf, z), 0)[0]
  pick = lambda a: np.take_along_axis(a, order[None], 0)[0]
  im = np.zeros((H, W, 3), np.uint8)
  im[pick(cls) == 1] = (60, 60, 60)
  im[pick(lanes_on) > 0] = (60, 140, 60)
  im[pick(lanes_on) > 1] = (0, 0, 255)
  im[pick(gap)] = (200, 60, 200)
  im[pick(wall) == 1] = (255, 255, 255)
  im[pick(wall) == 2] = (0, 220, 255)
  for i, f in enumerate(flags):
    c, r = px(f["x"], f["y"])
    cv2.circle(im, (int(c), int(r)), 14, (0, 0, 255), 3)
    cv2.putText(im, str(i), (int(c) + 16, int(r) + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
  cv2.imwrite(sys.argv[6], im)
