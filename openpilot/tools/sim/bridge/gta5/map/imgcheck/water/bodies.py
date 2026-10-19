"""bodies.py <out dir>: every height tile's water probe (<cell>_w.npz, 1 m grid) on one world grid, each water point
assigned a body: water.xml water (the NO_WAVES height answered) by its surface height and place, model water (lakes,
rivers, tanks, pools) by connected piece, place and size. Writes bodies.npz (grid: body id, surface height) and
bodies.json (id -> name, points, pieces)."""
import glob, json, os, sys
import numpy as np
from scipy import ndimage as ndi

OUT = sys.argv[1]
T = "/mnt/e/gta5_ortho_20261018/depth/tiles"
X0, Y0, NX, NY = -4400, -4400, 9000, 12800  # 1 m cells: x -4400..4600, y -4400..8400
c = np.zeros((NY, NX), np.uint8)  # 0 none, 1 water.xml, 2 model water
W = np.full((NY, NX), np.nan, np.float32)
n = 0
for side in glob.glob(f"{T}/*.json"):
  s = json.load(open(side))
  if not s.get("water", {}).get("water_pts"):
    continue
  z = np.load(f"{T}/{s['water']['npz']}")
  wat = z["c"] == 1
  if not wat.any():
    continue
  j0, i0 = int(z["y0"]) - Y0, int(z["x0"]) - X0
  h, w = wat.shape
  sl = (slice(j0, j0 + h), slice(i0, i0 + w))
  c[sl] = np.where(wat, np.where(z["q"], 1, 2), c[sl])
  W[sl] = np.where(wat, z["h"], W[sl])
  n += 1
# the NO_WAVES height doesn't answer over much of water.xml (whole reservoirs, the port): its surface heights are exact
XML_Z = [(0.0, 1.0), (30.0, 2.0), (196.409, 0.5), (160.339, 0.5)]  # sea, Alamo, Vinewood and Land Act reservoirs
known = np.zeros(c.shape, bool)
for z0, tol in XML_Z:
  known |= (c > 0) & (np.abs(W - z0) < tol)
known &= ~((np.abs(W - 30.0) < 2.0) & (np.arange(NX)[None, :] + X0 < -1500))  # not the tanks
c[known] = 1
print(n, "tiles with water;", int((c == 1).sum()), "water.xml m2,", int((c == 2).sum()), "model water m2", flush=True)

BODIES = ["none", "sea", "alamo", "vinewood_res", "land_act", "lago_zancudo", "pond", "mirror_park", "tanks", "ls_river",
          "zancudo_river", "pool"]
bid = {k: i for i, k in enumerate(BODIES)}
body = np.zeros((NY, NX), np.uint8)
yy, xx = np.nonzero(c == 1)
wz = W[yy, xx]
x, y = xx + X0 + 0.5, yy + Y0 + 0.5
b = np.full(len(yy), bid["pond"], np.uint8)
b[np.abs(wz) < 1.0] = bid["sea"]
b[(np.abs(wz - 30.0) < 2.0) & (x > -1500)] = bid["alamo"]  # the treatment tanks sit at 31.7 m too
b[np.abs(wz - 196.409) < 1.0] = bid["vinewood_res"]
b[np.abs(wz - 160.339) < 1.0] = bid["land_act"]
# Lago Zancudo and the lower river: sea-level water.xml inland of the coast (the opaque alpha-80 quads span x -2668..-1796)
b[(np.abs(wz) < 1.0) & (x > -2700) & (x < -1000) & (y > 2350) & (y < 3200)] = bid["lago_zancudo"]
body[yy, xx] = b

lab, nlab = ndi.label(c == 2, structure=np.ones((3, 3)))
objs = ndi.find_objects(lab)
area = np.bincount(lab.ravel())
pieces = []
for k, sl in enumerate(objs, 1):
  m = lab[sl] == k
  jj, ii = np.nonzero(m)
  cx, cy = float(ii.mean() + sl[1].start + X0), float(jj.mean() + sl[0].start + Y0)
  a = int(area[k])
  wm = float(np.nanmedian(W[sl][m]))
  if abs(cx - 1079) < 150 and abs(cy + 643) < 150 and a > 2000:
    name = "mirror_park"
  elif abs(cx + 2138) < 120 and abs(cy - 3425) < 120:
    name = "tanks"
  elif 300 < cx < 1400 and -2900 < cy < 900 and a >= 300:
    name = "ls_river"
  elif -3100 < cx < 0 and 1600 < cy < 4600 and a >= 300:
    name = "zancudo_river"
  elif a < 400:
    name = "pool"
  else:  # golf course and park lakes
    name = "pond"
  body[sl][m] = bid[name]
  pieces.append({"id": k, "body": name, "m2": a, "x": round(cx, 1), "y": round(cy, 1), "w": round(wm, 2)})
np.savez_compressed(f"{OUT}/bodies.npz", body=body, W=W.astype(np.float16), x0=X0, y0=Y0, step=1.0)
summary = {name: {"m2": int((body == i).sum())} for i, name in enumerate(BODIES) if i}
big = sorted(pieces, key=lambda p: -p["m2"])
json.dump({"bodies": BODIES, "summary": summary, "model_pieces": len(pieces), "biggest_pieces": big[:60],
           "ponds": [p for p in pieces if p["body"] == "pond"]}, open(f"{OUT}/bodies.json", "w"), indent=1)
for k, v in summary.items():
  print(f"{k:14s} {v['m2']:>10d} m2")
print("model pieces", len(pieces), "| ponds:", [(p["x"], p["y"], p["m2"]) for p in pieces if p["body"] == "pond"][:20])
