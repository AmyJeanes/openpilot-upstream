"""axes.py regions.npz rstats.json out.geojson: regions classified and lane centre lines traced, for the web map.

Classes, from regionstats.py's measures on the interchange (lane regions are 5-7 m at their widest, walled mostly
in paint; shoulders 1-1.6 m beside the road's edge; chevron cells under 15 m2, walled all round; unpainted floors
nearly all edge):
  cell: under CELL_AREA m2 and walled over CELL_WALL of its edge (a gore's chevrons cut it into these)
  median: yellow over MEDIAN_YELLOW of its edge
  shoulder: at most SHOULDER_W m wide at its widest, touching the road's edge
  open: road's edge over OPEN_EDGE of its edge (junction floors, car parks, unmarked roads)
  area: wider than AREA_W m at its widest (a junction's box, lanes running together where a line is missing)
  lane: the rest (one or more lanes; their centre lines say how many and where they part)
Centre lines: each lane region thinned to its medial axis (Zhang-Suen), split at forks, spurs shorter than the local
half width dropped; width along it is twice the distance to the region's edge."""
import json, sys
import numpy as np
import cv2
from scipy import ndimage as ndi
sys.path.insert(0, "/home/amy/git/op-ortho/openpilot/tools/sim/bridge/gta5/map/webmap")
import gframe

CELL_AREA, CELL_WALL, MEDIAN_YELLOW, SHOULDER_W, OPEN_EDGE = 15.0, 0.8, 0.7, 3.0, 0.7
AREA_W = 9.0  # m: lane regions measured 5-7.5 m at their widest; wider ones are junction boxes and lanes run together
HOLE = 15.0  # m2: islands of paint inside a lane (an arrow's mask missing its tip, a speck): lanes have no islands
R = np.load(sys.argv[1])
lab, z = R["lab"], R["z"]
res, x0, y1 = float(R["res"]), float(R["x0"]), float(R["y1"])
stats = {o["id"]: o for o in json.load(open(sys.argv[2]))}
L, H, W = lab.shape


def classify(o):
  walls = o["white"] + o["yellow"]
  if o["area"] < CELL_AREA and walls >= CELL_WALL:
    return "cell"
  if o["yellow"] >= MEDIAN_YELLOW:
    return "median"
  if o["widest"] <= SHOULDER_W and o["edge"] > 0.2:
    return "shoulder"
  if o["edge"] >= OPEN_EDGE:
    return "open"
  if o["widest"] > AREA_W:
    return "area"
  return "lane"


def thin(img):
  """Zhang-Suen thinning of a 0/1 image."""
  im = np.pad(img.astype(np.uint8), 1)
  changed = True
  while changed:
    changed = False
    for step in (0, 1):
      P = [im[1:-1, 1:-1]]
      n = [im[:-2, 1:-1], im[:-2, 2:], im[1:-1, 2:], im[2:, 2:], im[2:, 1:-1], im[2:, :-2], im[1:-1, :-2], im[:-2, :-2]]
      B = sum(n)
      seq = n + [n[0]]
      A = sum(((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8) for k in range(8))
      p2, p4, p6, p8 = n[0], n[2], n[4], n[6]
      if step == 0:
        c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
      else:
        c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
      rm = (P[0] == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
      if rm.any():
        im[1:-1, 1:-1][rm] = 0
        changed = True
  return im[1:-1, 1:-1]


NB = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def paths(sk):
  """the skeleton's pixel paths between ends and forks, as lists of (row, col)."""
  pts = set(zip(*np.nonzero(sk)))
  deg = {p: sum((p[0] + dr, p[1] + dc) in pts for dr, dc in NB) for p in pts}
  nodes = {p for p, d in deg.items() if d != 2}
  seen, out = set(), []
  for s in nodes:
    for dr, dc in NB:
      q = (s[0] + dr, s[1] + dc)
      if q not in pts or (s, q) in seen:
        continue
      path, prev, cur = [s], s, q
      while True:
        seen.add((prev, cur)); seen.add((cur, prev))
        path.append(cur)
        if cur in nodes:
          break
        nxt = [(cur[0] + dr2, cur[1] + dc2) for dr2, dc2 in NB if (cur[0] + dr2, cur[1] + dc2) in pts and (cur[0] + dr2, cur[1] + dc2) != prev]
        nxt = [p for p in nxt if (cur, p) not in seen]
        if not nxt:
          break
        prev, cur = cur, nxt[0]
      out.append(path)
  if not nodes and pts:  # a loop
    out.append(sorted(pts))
  return out, nodes, deg


def lonlat(cols, rows):
  x = x0 + (np.asarray(cols) + 0.5) * res
  y = y1 - (np.asarray(rows) + 0.5) * res
  lat, lon = gframe.lat_lon(x, y)
  return np.stack([lon, lat], 1).round(8).tolist()


feats = []
counts = {}
objs = ndi.find_objects(lab)
for k, o in stats.items():
  sl = objs[k - 1]
  if sl is None:
    continue
  r0, c0 = max(sl[1].start - 2, 0), max(sl[2].start - 2, 0)
  m3 = lab[:, r0:sl[1].stop + 2, c0:sl[2].stop + 2] == k
  foot = m3.any(0)
  kind = classify(o)
  counts[kind] = counts.get(kind, 0) + 1
  cs, _ = cv2.findContours(foot.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  for cnt in cs:
    if len(cnt) < 3:
      continue
    ring = lonlat(cnt[:, 0, 0] + c0, cnt[:, 0, 1] + r0)
    ring.append(ring[0])
    feats.append({"type": "Feature", "properties": {"kind": "region", "class": kind, "id": k, **o},
                  "geometry": {"type": "Polygon", "coordinates": [ring]}})
  if kind not in ("lane", "area"):
    continue
  # specks inside a lane (worn paint, a manhole) would loop the axis: fill holes under HOLE m2
  holes, nh = ndi.label(ndi.binary_fill_holes(foot) & ~foot)
  if nh:
    sizes = ndi.sum(np.ones_like(holes), holes, np.arange(1, nh + 1)) * res * res
    foot = foot | np.isin(holes, np.flatnonzero(sizes < HOLE) + 1)
  dist = ndi.distance_transform_edt(foot)
  sk = thin(foot)
  # spurs (from bumps in the edge) pruned until none is shorter than the lane's width where it grows
  for _ in range(20):
    ps, nodes, deg = paths(sk)
    cut = False
    for p in ps:
      ends = [q for q in (p[0], p[-1]) if deg.get(q, 2) == 1]
      if not ends or len(ps) == 1:
        continue
      rr = np.array([q[0] for q in p]); cc = np.array([q[1] for q in p])
      if len(p) * res < 2 * dist[rr, cc].max() * res:
        inner = [q for q in p if deg.get(q, 2) <= 2]  # keep the fork pixel
        for q in inner:
          sk[q] = 0
        cut = True
    if not cut:
      break
    sk = thin(sk)
  ps, nodes, deg = paths(sk)
  for p in ps:
    rr = np.array([q[0] for q in p]); cc = np.array([q[1] for q in p])
    w = 2 * dist[rr, cc] * res
    if len(p) < 3:
      continue
    # thin the pixel path to every ~0.5 m and smooth a little
    step = max(1, int(0.5 / res))
    sel = np.r_[np.arange(0, len(p), step), len(p) - 1]
    sx = ndi.gaussian_filter1d(cc[sel].astype(float), 1, mode="nearest") + c0
    sy = ndi.gaussian_filter1d(rr[sel].astype(float), 1, mode="nearest") + r0
    feats.append({"type": "Feature", "properties": {"kind": "axis", "region": k, "width": round(float(np.median(w)), 2),
                                                    "z": o["z"], "len": round(len(p) * res, 1)},
                  "geometry": {"type": "LineString", "coordinates": lonlat(sx, sy)}})
json.dump({"type": "FeatureCollection", "features": feats}, open(sys.argv[3], "w"))
print(counts, sum(f["properties"]["kind"] == "axis" for f in feats), "axis pieces ->", sys.argv[3])
