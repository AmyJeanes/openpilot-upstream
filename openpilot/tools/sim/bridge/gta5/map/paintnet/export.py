"""export.py dir.json groups.json out.geojson: the pilot's lanes for the web map, in its frame (webmap/gframe.py).

Features (properties.kind): lane (area between its boundaries, coloured by direction group, with "dir" known or not),
centre (a lane's centre line in its travel direction, for arrows); lanes and areas are cut into pieces in the open
or under a deck ("covered", for the page's level switch), gore, shoulder, crossing, not_traffic (areas), and
flag (a point per flagged group, with why). Every lane has an id (its two boundary line ids and run) so it can be
named in a correction."""
import json, math, sys
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, "/home/amy/git/op-ortho/openpilot/tools/sim/bridge/gta5/map/webmap")
import gframe
sys.path.insert(0, "/home/amy/git/openpilot-slowroads/openpilot/tools/sim/bridge/gta5/map/roadpaint")
import rp

T = rp.Tiles("/mnt/e/gta5_scratch/gamefiles_v2/rp_all/tiles", cache=256)


def covered(A):
  """per point: another surface in the game files more than 2.5 m above it (under a bridge or deck)."""
  out = np.zeros(len(A), bool)
  for k, (x, y, z) in enumerate(A[:, :3]):
    tx, ty = int(np.floor(x / rp.TILE)), int(np.floor(y / rp.TILE))
    t = T._get(tx, ty)
    if t is None:
      continue
    c, zz = t
    col = min(int((x - tx * rp.TILE) / rp.RES), rp.N - 1)
    row = min(int((ty * rp.TILE + rp.TILE - y) / rp.RES), rp.N - 1)
    out[k] = bool(((c[:, row, col] > 0) & (zz[:, row, col] > z + 2.5)).any())
  return out

dirj, grpj, out = sys.argv[1:4]


def lonlat(xy):
  lat, lon = gframe.lat_lon(xy[:, 0], xy[:, 1])
  return np.stack([lon, lat], 1).round(8).tolist()


def area(A):
  """polygon of a run: centre +- half its width square to it, reaching half a sample past each end (each sample stands
  for the metre around it, so runs that meet leave no gap)."""
  xy = A[:, :2].copy()
  d = np.gradient(xy, axis=0)
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  xy[0] -= d[0] * 0.5
  xy[-1] += d[-1] * 0.5
  n = np.stack([d[:, 1], -d[:, 0]], 1)
  h = A[:, 3:4] / 2
  ring = np.vstack([xy - n * h, (xy + n * h)[::-1], (xy - n * h)[:1]])
  return [lonlat(ring)]


# direction groups: colours clear of the other kinds' (gore magenta, crossing cyan, median yellow, turn lane orange,
# no-traffic and flag red, grey)
PALETTE = ["#43a047", "#1e88e5", "#8e24aa", "#00897b", "#6d4c41", "#3949ab", "#7cb342", "#5e35b1", "#0277bd", "#2e7d32"]


def colour(g):
  return PALETTE[g % len(PALETTE)]


def pieces(A):
  """(start, end, covered) stretches of a run in the open or under a deck, at least 2 samples each. An open gap under
  15 m between covered stretches (the strip between a motorway's two carriageways) counts as beneath: display only."""
  cv = covered(A)
  cv = cv | ndi.binary_closing(np.r_[False, cv, False], np.ones(15, bool))[1:-1]
  out, k0 = [], 0
  for k in range(1, len(A) + 1):
    if k == len(A) or cv[k] != cv[k0]:
      if k - k0 >= 2:
        out.append((k0, k, bool(cv[k0])))
      elif out:  # a lone sample joins the stretch before it
        out[-1] = (out[-1][0], k, out[-1][2])
      k0 = k
  return out or [(0, len(A), bool(cv[0]))]


feats = []
G = json.load(open(grpj))
for k, r in enumerate(G["lanes"]):
  A = np.array(r["pts"])
  lid = f"{r['left']}|{r['right']}|{k}"
  props = {"kind": "lane", "id": lid, "group": r["group"], "col": colour(r["group"]), "dir": r["dir"],
           "width": round(float(np.median(A[:, 3])), 2), "z": round(float(np.median(A[:, 2])), 1),
           "arrows": r.get("arrow_votes", 0), "links": r.get("link_votes", 0)}
  for a, b, cov in pieces(A):  # one piece per stretch in the open or under a deck (the level switch, arrows)
    feats.append({"type": "Feature", "properties": {**props, "covered": cov}, "geometry": {"type": "Polygon", "coordinates": area(A[a:b])}})
    if r["dir"] != "?":
      feats.append({"type": "Feature", "properties": {**props, "kind": "centre", "covered": cov},
                    "geometry": {"type": "LineString", "coordinates": lonlat(A[a:b, :2])}})
for k, r in enumerate(json.load(open(dirj))):
  if r["kind"] in ("lane",):
    continue
  A = np.array(r["pts"])
  for a, b, cov in pieces(A):
    feats.append({"type": "Feature", "properties": {"kind": r["kind"], "id": f"{r['left']}|{r['right']}|{r['kind'][0]}{k}",
                                                    "width": round(float(np.median(A[:, 3])), 2), "z": round(float(np.median(A[:, 2])), 1),
                                                    "covered": cov},
                  "geometry": {"type": "Polygon", "coordinates": area(A[a:b])}})
lanes_by = {}
for r in G["lanes"]:
  lanes_by.setdefault(r["group"], []).append(np.array(r["pts"]))
for f in G["flags"]:
  P = np.vstack(lanes_by.get(f["group"], [np.zeros((1, 4))]))
  m = P[len(P) // 2]
  feats.append({"type": "Feature", "properties": {"kind": "flag", "group": f["group"], "why": "; ".join(f["why"]), "covered": False},
                "geometry": {"type": "Point", "coordinates": lonlat(m[None, :2])[0]}})
json.dump({"type": "FeatureCollection", "features": feats}, open(out, "w"))
print(len(feats), "features ->", out)
