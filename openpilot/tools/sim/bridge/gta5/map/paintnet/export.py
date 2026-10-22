"""export.py dir.json groups.json out.geojson [links.json]: the pilot's lanes for the web map, in its frame (webmap/gframe.py).

Features (properties.kind): lane (area between its boundaries, coloured by direction group, with "dir" known or not),
centre (a lane's centre line in its travel direction, for arrows); lanes and areas are cut into pieces in the open
or under a deck ("covered", for the page's level switch), gore, shoulder, crossing, not_traffic (areas), and
flag (a point per flagged group, with why); from connect.py's links: connector (the strip filling a break between a
lane and the one it carries on into) and link (a line from a lane's end to its successor's start; "type" split when a
lane carries on into two, merge when two carry on into one, else continue) with linkpt, a point marker halfway along it; edge (a lane piece's sides, and a line across at the lane's real ends).
Every piece says whether it runs under a deck ("covered"), whether it is one ("deck"), and whether its lane is one
anywhere ("on_deck"), for the page's level switch.
Every lane has an id (its two boundary line ids and run) so it can be
named in a correction."""
import json, math, sys
import numpy as np
from scipy import ndimage as ndi

sys.path.insert(0, "/home/amy/git/op-ortho/openpilot/tools/sim/bridge/gta5/map/webmap")
import gframe
sys.path.insert(0, "/home/amy/git/openpilot-slowroads/openpilot/tools/sim/bridge/gta5/map/roadpaint")
import rp

T = rp.Tiles("/mnt/e/gta5_scratch/gamefiles_v2/rp_all/tiles", cache=256)


def covered(A, sign=1):
  """per point: another surface in the game files more than 2.5 m above it (under a bridge or deck); with sign -1, more
  than 2.5 m below it (on a deck)."""
  out = np.zeros(len(A), bool)
  for k, (x, y, z) in enumerate(A[:, :3]):
    tx, ty = int(np.floor(x / rp.TILE)), int(np.floor(y / rp.TILE))
    t = T._get(tx, ty)
    if t is None:
      continue
    c, zz = t
    col = min(int((x - tx * rp.TILE) / rp.RES), rp.N - 1)
    row = min(int((ty * rp.TILE + rp.TILE - y) / rp.RES), rp.N - 1)
    out[k] = bool(((c[:, row, col] > 0) & (sign * (zz[:, row, col] - z) > 2.5)).any())
  return out

dirj, grpj, out = sys.argv[1:4]
linkj = sys.argv[4] if len(sys.argv) > 4 else None


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


def edges(A, start, end):
  """a run piece's two sides as lines, plus a line across it at the run's real start or end (pieces are cut where a
  lane goes on or off a deck, and those cuts aren't lane ends)."""
  xy = A[:, :2].copy()
  d = np.gradient(xy, axis=0)
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  xy[0] -= d[0] * 0.5
  xy[-1] += d[-1] * 0.5
  n = np.stack([d[:, 1], -d[:, 0]], 1)
  h = A[:, 3:4] / 2
  left, right = xy - n * h, xy + n * h
  out = [left, right]
  if start:
    out.append(np.array([left[0], right[0]]))
  if end:
    out.append(np.array([left[-1], right[-1]]))
  return [lonlat(e) for e in out]


# direction groups: colours clear of the other kinds' (gore magenta, crossing cyan, median yellow, turn lane orange,
# no-traffic and flag red, grey)
PALETTE = ["#43a047", "#1e88e5", "#8e24aa", "#00897b", "#6d4c41", "#3949ab", "#7cb342", "#5e35b1", "#0277bd", "#2e7d32"]


def colour(g):
  return PALETTE[g % len(PALETTE)]


def pieces(A):
  """(start, end, covered, deck) stretches of a run under a deck or not and on a deck or not, at least 2 samples each.
  An open gap under 15 m between covered (or deck) stretches (the strip between a motorway's two carriageways) counts
  as one: display only."""
  st = []
  for sign in (1, -1):
    cv = covered(A, sign)
    st.append(cv | ndi.binary_closing(np.r_[False, cv, False], np.ones(15, bool))[1:-1])
  cv = st[0] + 2 * st[1]
  out, k0 = [], 0
  for k in range(1, len(A) + 1):
    if k == len(A) or cv[k] != cv[k0]:
      if k - k0 >= 2:
        out.append((k0, k, bool(cv[k0] & 1), bool(cv[k0] & 2)))
      elif out:  # a lone sample joins the stretch before it
        out[-1] = (out[-1][0], k, out[-1][2], out[-1][3])
      k0 = k
  return out or [(0, len(A), bool(cv[0] & 1), bool(cv[0] & 2))]


feats = []
lane_deck = {}
G = json.load(open(grpj))
LK = json.load(open(linkj)) if linkj else {"links": [], "pts": {}}
for k, p in LK.get("pts", {}).items():  # openings reshaped by connect.py
  G["lanes"][int(k)]["pts"] = p
lid = lambda k: f"{G['lanes'][k]['left']}|{G['lanes'][k]['right']}|{k}"
nsucc, npred = {}, {}
for i, j, g, *br in LK["links"]:
  if not br:  # a branch leaves a lane that carries on; it doesn't make that lane's own carrying on a split
    nsucc[i] = nsucc.get(i, 0) + 1
  npred[j] = npred.get(j, 0) + 1
# a lane carrying on into exactly one, which nothing else carries on into, is the same lane (its lines are stored in
# pieces, or one stops for a stretch): drawn as one, its break filled
nxt = {i: j for i, j, g, *br in LK["links"] if not br and nsucc[i] == 1 and npred[j] == 1}
heads = [k for k in range(len(G["lanes"])) if k not in set(nxt.values())]


def fill(A, B):
  """samples running from A's end to B's start, one a metre."""
  g = np.linalg.norm(B[0, :2] - A[-1, :2])
  t = np.arange(1, max(int(round(g)), 1))[:, None] / max(round(g), 1)
  return A[-1] + (B[0] - A[-1]) * t


for k in heads:
  chain = [k]
  while chain[-1] in nxt:
    chain.append(nxt[chain[-1]])
  A = np.array(G["lanes"][k]["pts"])
  for q in chain[1:]:
    B = np.array(G["lanes"][q]["pts"])
    A = np.vstack([A, fill(A, B), B])
  r = G["lanes"][k]
  name = " > ".join(lid(q) for q in chain)
  props = {"kind": "lane", "id": name, "group": r["group"], "col": colour(r["group"]), "dir": r["dir"],
           "width": round(float(np.median(A[:, 3])), 2), "z": round(float(np.median(A[:, 2])), 1),
           "arrows": sum(G["lanes"][q].get("arrow_votes", 0) for q in chain), "links": sum(G["lanes"][q].get("link_votes", 0) for q in chain)}
  pcs = pieces(A)
  props["on_deck"] = any(p[3] for p in pcs)  # the lowest-level view hides a lane on a deck anywhere whole
  for q in chain:
    lane_deck[q] = props["on_deck"]
  for a, b, cov, deck in pcs:  # one piece per stretch under a deck, on one, or neither (the level switch, arrows)
    feats.append({"type": "Feature", "properties": {**props, "covered": cov, "deck": deck}, "geometry": {"type": "Polygon", "coordinates": area(A[a:b])}})
    feats.append({"type": "Feature", "properties": {"kind": "edge", "id": name, "covered": cov, "deck": deck, "on_deck": props["on_deck"]},
                  "geometry": {"type": "MultiLineString", "coordinates": edges(A[a:b], a == 0, b == len(A))}})
    if r["dir"] != "?":
      feats.append({"type": "Feature", "properties": {**props, "kind": "centre", "covered": cov, "deck": deck},
                    "geometry": {"type": "LineString", "coordinates": lonlat(A[a:b, :2])}})
for k, r in enumerate(json.load(open(dirj))):
  if r["kind"] in ("lane",):
    continue
  A = np.array(r["pts"])
  pcs = pieces(A)
  on_deck = any(p[3] for p in pcs)
  for a, b, cov, deck in pcs:
    feats.append({"type": "Feature", "properties": {"kind": r["kind"], "on_deck": on_deck, "id": f"{r['left']}|{r['right']}|{r['kind'][0]}{k}",
                                                    "width": round(float(np.median(A[:, 3])), 2), "z": round(float(np.median(A[:, 2])), 1),
                                                    "covered": cov, "deck": deck},
                  "geometry": {"type": "Polygon", "coordinates": area(A[a:b])}})
for i, j, g, *br in LK["links"]:
  if nxt.get(i) == j:  # drawn as one lane
    continue
  A, B = np.array(G["lanes"][i]["pts"]), np.array(G["lanes"][j]["pts"])
  if br == ["merge"]:  # a lane ending beside one that carries on: marked where it ends
    a, b0 = A[-1], B[np.linalg.norm(B[:, :2] - A[-1, :2], axis=1).argmin()]
  else:
    a = A[np.linalg.norm(A[:, :2] - B[0, :2], axis=1).argmin()] if br else A[-1]  # a branch leaves from beside its start
    b0 = B[0]
  kind = "merge" if br == ["merge"] or (not br and npred[j] > 1) else "split" if br or nsucc[i] > 1 else "continue"
  if not br and g > 0.5:  # (a merge's or branch's lanes touch)
    # the break before a split or merge filled with samples running from the lane's end to its successor's start
    F = fill(A, B)
    if len(F):
      to = G["lanes"][j]
      mid = F[len(F) // 2][None, :3]
      cov, deck = bool(covered(mid)[0]), bool(covered(mid, -1)[0])
      props = {"kind": "lane", "connector": True, "id": f"{lid(i)} > {lid(j)}", "group": to["group"], "col": colour(to["group"]),
               "dir": to["dir"], "z": round(float(mid[0, 2]), 1), "covered": cov, "deck": deck, "on_deck": lane_deck[i] or lane_deck[j]}
      feats.append({"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": area(F)}})
      feats.append({"type": "Feature", "properties": {"kind": "edge", "id": props["id"], "covered": cov, "deck": deck, "on_deck": props["on_deck"]},
                    "geometry": {"type": "MultiLineString", "coordinates": edges(F, False, False)[:2]}})
  m = (a[:2] + b0[:2]) / 2
  feats.append({"type": "Feature", "properties": {"kind": "linkpt", "type": kind, "from": lid(i), "to": lid(j), "gap": g, "on_deck": lane_deck[i] or lane_deck[j],
                                                  "covered": bool(covered(np.array([a[:3]]))[0]), "deck": bool(covered(np.array([a[:3]]), -1)[0])},
                "geometry": {"type": "Point", "coordinates": lonlat(m[None])[0]}})
lanes_by = {}
for r in G["lanes"]:
  lanes_by.setdefault(r["group"], []).append(np.array(r["pts"]))
for f in G["flags"]:
  P = np.vstack(lanes_by.get(f["group"], [np.zeros((1, 4))]))
  m = P[len(P) // 2]
  feats.append({"type": "Feature", "properties": {"kind": "flag", "group": f["group"], "why": "; ".join(f["why"]), "covered": False, "deck": False, "on_deck": False},
                "geometry": {"type": "Point", "coordinates": lonlat(m[None, :2])[0]}})
json.dump({"type": "FeatureCollection", "features": feats}, open(out, "w"))
print(len(feats), "features ->", out)
