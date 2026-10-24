"""export.py dir.json groups.json out.geojson [links.json [junctions.json]] [--box=x0,x1,y0,y1]: the pilot's lanes for the web map, in its frame (webmap/gframe.py).

Features (properties.kind): lane (area between its boundaries, coloured by its heading, north/east/south/west, with "dir"
known or not),
centre (a lane's centre line in its travel direction, for arrows); lanes and areas are cut into pieces in the open
or under a deck ("covered", for the page's level switch), gore, shoulder, crossing (its outline, where it has one),
not_traffic (areas; not those in a junction's box, between a stop line and its crossing), and
flag (a point per flagged group, with why), junction (junctions.py's box, its lanes inside named), guide (a lane inside
a junction's box: its lines carried through it, guiding traffic, not a lane), turns (a lane arriving at a junction: its
arrows' turns, at its stop line), path (junctions.py's: an arriving lane's way across the box to a leaving lane, by
its turn); from connect.py's links: connector (the strip filling a break between a
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

BOX = next((list(map(float, a[6:].split(","))) for a in sys.argv if a.startswith("--box=")), None)  # x0 x1 y0 y1
sys.argv = [a for a in sys.argv if not a.startswith("--box=")]
dirj, grpj, out = sys.argv[1:4]
linkj = sys.argv[4] if len(sys.argv) > 4 else None


def clip(A):
  """the longest stretch of A's samples inside the pilot's box (lines are only read up to 5 m past it, so runs fray
  beyond it), and where it starts and stops in A; None if under 2 samples."""
  if BOX is None:
    return A, 0, len(A)
  inb = (A[:, 0] >= BOX[0]) & (A[:, 0] <= BOX[1]) & (A[:, 1] >= BOX[2]) & (A[:, 1] <= BOX[3])
  best, k0 = (0, 0), None
  for k in range(len(A) + 1):
    if k < len(A) and inb[k]:
      k0 = k if k0 is None else k0
    elif k0 is not None:
      best = max(best, (k0, k), key=lambda r: r[1] - r[0])
      k0 = None
  return (A[best[0]:best[1]], *best) if best[1] - best[0] >= 2 else (None, 0, 0)


def lonlat(xy):
  lat, lon = gframe.lat_lon(xy[:, 0], xy[:, 1])
  return np.stack([lon, lat], 1).round(8).tolist()


def meet(a, d, cut):
  """where the line through a along d meets cut ([x, y, ux, uy]: a stop line's point and way), or None if they run
  nearly alike."""
  q, u = np.array(cut[:2]), np.array(cut[2:])
  den = d[0] * -u[1] + u[0] * d[1]
  if abs(den) < 0.3:
    return None
  s = ((q - a)[0] * -u[1] + u[0] * (q - a)[1]) / den
  return a + d * s


def sides(A, cut0=None, cut1=None):
  """a run's two sides (centre +- half its width square to it), reaching half a sample past each end (each sample stands
  for the metre around it, so runs that meet leave no gap), or at an end with a cut (its stop line) along that."""
  xy = A[:, :2].copy()
  d = np.gradient(xy, axis=0)
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  n = np.stack([d[:, 1], -d[:, 0]], 1)
  h = A[:, 3:4] / 2
  left, right = xy - n * h, xy + n * h
  for k, sg, cut in ((0, -1, cut0), (-1, 1, cut1)):
    ends = [meet(e[k], d[k], cut) for e in (left, right)] if cut else [None]
    if all(m is not None for m in ends) and all(np.linalg.norm(m - e[k]) < 3 for m, e in zip(ends, (left, right))):
      left[k], right[k] = ends
    else:
      left[k] = left[k] + sg * d[k] * 0.5
      right[k] = right[k] + sg * d[k] * 0.5
  return left, right


def area(A, cut0=None, cut1=None):
  """polygon of a run (see sides)."""
  left, right = sides(A, cut0, cut1)
  return [lonlat(np.vstack([left, right[::-1], left[:1]]))]


def edges(A, start, end, cut0=None, cut1=None):
  """a run piece's two sides as lines, plus a line across it at the run's real start or end (pieces are cut where a
  lane goes on or off a deck, and those cuts aren't lane ends)."""
  left, right = sides(A, cut0, cut1)
  out = [left, right]
  if start:
    out.append(np.array([left[0], right[0]]))
  if end:
    out.append(np.array([left[-1], right[-1]]))
  return [lonlat(e) for e in out]


# lanes coloured by which way they head, in quarters round the compass (clear of the other kinds' colours: gore
# magenta, crossing cyan, median yellow, turn lane orange, no-traffic and flag red, grey)
HEADING = {"north": "#1e88e5", "east": "#43a047", "south": "#8e24aa", "west": "#00897b"}


def heading(A):
  """the quarter of the compass a run heads in overall (its mean direction)."""
  d = np.diff(A[:, :2], axis=0).sum(0)
  a = math.degrees(math.atan2(d[1], d[0])) % 360
  return ["east", "north", "west", "south"][int(((a + 45) % 360) // 90)]


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
STOPS = LK.get("stops", {})
JUNCTIONS = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else []
guide = {i: J["id"] for J in JUNCTIONS for i in J["inside"]}  # lane paint carried through a junction's box
GUIDE_MIN = 6.0  # m: shorter pieces (beside a corner's arc) guide nothing (lanes.py's MINLEN)
lid = lambda k: f"{G['lanes'][k]['left']}|{G['lanes'][k]['right']}|{k}"
nsucc, npred = {}, {}
for i, j, g, *br in LK["links"]:
  if not br:  # a branch leaves a lane that carries on; it doesn't make that lane's own carrying on a split
    nsucc[i] = nsucc.get(i, 0) + 1
  if br != ["merge"]:  # nor does a lane merging into one make its carrying on a merge
    npred[j] = npred.get(j, 0) + 1
# a lane carrying on into exactly one, which nothing else carries on into, is the same lane (its lines are stored in
# pieces, or one stops for a stretch): drawn as one, its break filled
nxt = {i: j for i, j, g, *br in LK["links"] if not br and nsucc[i] == 1 and npred[j] == 1}
heads = [k for k in range(len(G["lanes"])) if k not in set(nxt.values())]


def fill(A, B):
  """samples running from A's end to B's start, one a metre: straight, or curving from A's direction to B's where they
  differ (a free turn lane carrying on round a traffic island)."""
  g = np.linalg.norm(B[0, :2] - A[-1, :2])
  t = np.arange(1, max(int(round(g)), 1))[:, None] / max(round(g), 1)
  F = A[-1] + (B[0] - A[-1]) * t
  if len(A) > 1 and len(B) > 1:
    dp, dq = A[-1, :2] - A[max(len(A) - 4, 0), :2], B[min(3, len(B) - 1), :2] - B[0, :2]
    dp, dq = dp / max(np.linalg.norm(dp), 1e-9), dq / max(np.linalg.norm(dq), 1e-9)
    if dp @ dq < math.cos(math.radians(10)):
      F[:, :2] = (2 * t ** 3 - 3 * t ** 2 + 1) * A[-1, :2] + (t ** 3 - 2 * t ** 2 + t) * g * dp + (3 * t ** 2 - 2 * t ** 3) * B[0, :2] + (t ** 3 - t ** 2) * g * dq
  return F


for k in heads:
  chain = [k]
  while chain[-1] in nxt:
    chain.append(nxt[chain[-1]])
  A = np.array(G["lanes"][k]["pts"])
  if all(q in guide for q in chain):  # (a guide: where traffic through the junction drives, not a lane)
    if len(A) >= GUIDE_MIN:
      feats.append({"type": "Feature", "properties": {"kind": "guide", "id": lid(k), "junction": guide[k], "z": round(float(np.median(A[:, 2])), 1),
                                                      "covered": False, "deck": False, "on_deck": False}, "geometry": {"type": "Polygon", "coordinates": area(A)}})
    continue
  for q in chain[1:]:
    B = np.array(G["lanes"][q]["pts"])
    A = np.vstack([A, fill(A, B), B])
  n_all = len(A)
  A, i0, i1 = clip(A)
  if A is None:
    continue
  r = G["lanes"][k]
  name = " > ".join(lid(q) for q in chain)
  props = {"kind": "lane", "id": name, "group": r["group"], "heading": heading(A), "col": HEADING[heading(A)], "dir": r["dir"],
           "width": round(float(np.median(A[:, 3])), 2), "z": round(float(np.median(A[:, 2])), 1),
           "arrows": sum(G["lanes"][q].get("arrow_votes", 0) for q in chain), "links": sum(G["lanes"][q].get("link_votes", 0) for q in chain)}
  pcs = pieces(A)
  props["on_deck"] = any(p[3] for p in pcs)  # the lowest-level view hides a lane on a deck anywhere whole
  for q in chain:
    lane_deck[q] = props["on_deck"]
  stop0 = STOPS.get(str(chain[0]), {}).get("start") if i0 == 0 else None
  stop1 = STOPS.get(str(chain[-1]), {}).get("end") if i1 == n_all else None
  for a, b, cov, deck in pcs:  # one piece per stretch under a deck, on one, or neither (the level switch, arrows)
    cut = (stop0 if a == 0 else None, stop1 if b == len(A) else None)
    zp = round(float(np.median(A[a:b, 2])), 1)  # (the piece's own height: the map's height band)
    feats.append({"type": "Feature", "properties": {**props, "covered": cov, "deck": deck, "z": zp}, "geometry": {"type": "Polygon", "coordinates": area(A[a:b], *cut)}})
    feats.append({"type": "Feature", "properties": {"kind": "edge", "id": name, "covered": cov, "deck": deck, "on_deck": props["on_deck"], "z": zp},
                  "geometry": {"type": "MultiLineString", "coordinates": edges(A[a:b], a == 0, b == len(A), *cut)}})
    if r["dir"] != "?":
      feats.append({"type": "Feature", "properties": {**props, "kind": "centre", "covered": cov, "deck": deck},
                    "geometry": {"type": "LineString", "coordinates": lonlat(A[a:b, :2])}})
in_box = {k for J in JUNCTIONS for k in J["strips"]}
for k, r in enumerate(json.load(open(dirj))):
  if r["kind"] in ("lane",):
    continue
  if r["kind"] == "not_traffic" and k in in_box:  # (between a stop line and its crossing: the junction's box, driven over)
    continue
  A, i0, i1 = clip(np.array(r["pts"]))
  if A is None:
    continue
  st = r.get("stops", {})  # (a strip's end cut along the stop line or crossing it runs to)
  st = {"start": st.get("start") if i0 == 0 else None, "end": st.get("end") if i1 == len(r["pts"]) else None}
  pcs = pieces(A)
  on_deck = any(p[3] for p in pcs)
  if r.get("outline"):  # a crossing cut to the road's surface: one area
    O = np.array(r["outline"] + r["outline"][:1])
    mid = A[len(A) // 2][None, :3]
    feats.append({"type": "Feature", "properties": {"kind": r["kind"], "on_deck": on_deck, "id": f"{r['left']}|{r['right']}|{r['kind'][0]}{k}",
                                                    "width": round(float(np.median(A[:, 3])), 2), "z": round(float(np.median(A[:, 2])), 1),
                                                    "covered": bool(covered(mid)[0]), "deck": bool(covered(mid, -1)[0])},
                  "geometry": {"type": "Polygon", "coordinates": [lonlat(O)]}})
    continue
  for a, b, cov, deck in pcs:
    feats.append({"type": "Feature", "properties": {"kind": r["kind"], "on_deck": on_deck, "id": f"{r['left']}|{r['right']}|{r['kind'][0]}{k}",
                                                    "width": round(float(np.median(A[:, 3])), 2), "z": round(float(np.median(A[:, 2])), 1),
                                                    "covered": cov, "deck": deck},
                  "geometry": {"type": "Polygon", "coordinates": area(A[a:b], st["start"] if a == 0 else None, st["end"] if b == len(A) else None)}})
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
    if len(F) >= 2:  # (one sample: a gap the lanes' own half-sample ends already close)
      to = G["lanes"][j]
      mid = F[len(F) // 2][None, :3]
      cov, deck = bool(covered(mid)[0]), bool(covered(mid, -1)[0])
      hd = heading(B)  # (the lane it leads into)
      props = {"kind": "lane", "connector": True, "id": f"{lid(i)} > {lid(j)}", "group": to["group"], "heading": hd, "col": HEADING[hd],
               "dir": to["dir"], "z": round(float(mid[0, 2]), 1), "covered": cov, "deck": deck, "on_deck": lane_deck.get(i, False) or lane_deck.get(j, False)}
      feats.append({"type": "Feature", "properties": props, "geometry": {"type": "Polygon", "coordinates": area(F)}})
      feats.append({"type": "Feature", "properties": {"kind": "edge", "id": props["id"], "covered": cov, "deck": deck, "on_deck": props["on_deck"], "z": props["z"]},
                    "geometry": {"type": "MultiLineString", "coordinates": edges(F, False, False)[:2]}})
  m = (a[:2] + b0[:2]) / 2
  feats.append({"type": "Feature", "properties": {"kind": "linkpt", "type": kind, "from": lid(i), "to": lid(j), "gap": g, "z": round(float(a[2]), 1), "on_deck": lane_deck.get(i, False) or lane_deck.get(j, False),
                                                  "covered": bool(covered(np.array([a[:3]]))[0]), "deck": bool(covered(np.array([a[:3]]), -1)[0])},
                "geometry": {"type": "Point", "coordinates": lonlat(m[None])[0]}})
# each lane arriving at a junction: its turns from its arrows, drawn just before its stop line as arrows pointing the
# way it travels (? where an arrow is painted but its type unknown)
GLYPH = {"left": "←", "through": "↑", "right": "→"}
turns_of = {}
for J in JUNCTIONS:
  for i, which, tu in J["mouths"]:
    if which != "end":
      continue
    turns_of[i] = tu
    A = np.array(G["lanes"][i]["pts"])
    e = A[-1, :2] - A[max(len(A) - 4, 0), :2]
    e /= max(np.linalg.norm(e), 1e-9)
    p = A[-1, :2] - e * 3.0
    feats.append({"type": "Feature", "properties": {"kind": "turns", "id": lid(i), "turns": tu or "none",
                                                    "text": "?" if tu == "?" else "".join(GLYPH[t] for t in (tu or "").split(";") if t),
                                                    "bearing": round(math.degrees(math.atan2(e[0], e[1])), 1), "z": J["z"],
                                                    "covered": False, "deck": False, "on_deck": False},
                  "geometry": {"type": "Point", "coordinates": lonlat(p[None])[0]}})
if JUNCTIONS:
  for J in JUNCTIONS:
    feats.append({"type": "Feature", "properties": {"kind": "junction", "id": f"junction {J['id']}", "inside": ", ".join(lid(i) for i in J["inside"]),
                                                    "z": J["z"], "covered": False, "deck": False, "on_deck": False},
                  "geometry": {"type": "Polygon", "coordinates": [lonlat(np.array(J["ring"] + J["ring"][:1]))]}})
    # its paths: an arriving lane's end to a leaving lane's start, by its turn; an arrow with no road its way, and a
    # leaving lane nothing reaches, flagged
    flat = {"covered": False, "deck": False, "on_deck": False}
    for p in J.get("paths", []):
      if p["to"] is None:
        e = np.array(G["lanes"][p["from"]]["pts"])[-1]
        feats.append({"type": "Feature", "properties": {"kind": "flag", "group": None, "why": f"{lid(p['from'])}: turns {p['turn']}, no road leaves that way", "z": round(float(e[2]), 1), **flat},
                      "geometry": {"type": "Point", "coordinates": lonlat(e[None, :2])[0]}})
        continue
      P = np.array(p["pts"])
      feats.append({"type": "Feature", "properties": {"kind": "path", "id": f"{lid(p['from'])} > {lid(p['to'])}", "turn": p["turn"],
                                                      "guide": lid(p["guide"]) if p["guide"] is not None else "", "flags": ", ".join(p["flags"]),
                                                      "junction": J["id"], "z": round(float(np.median(P[:, 2])), 1), **flat},
                    "geometry": {"type": "LineString", "coordinates": lonlat(P[:, :2])}})
    for j in J.get("unreached", []):
      s = np.array(G["lanes"][j]["pts"])[0]
      feats.append({"type": "Feature", "properties": {"kind": "flag", "group": None, "why": f"{lid(j)}: leaves junction {J['id']}, no path reaches it", "z": round(float(s[2]), 1), **flat},
                    "geometry": {"type": "Point", "coordinates": lonlat(s[None, :2])[0]}})
lanes_by = {}
for r in G["lanes"]:
  lanes_by.setdefault(r["group"], []).append(np.array(r["pts"]))
for f in G["flags"]:
  P = np.vstack(lanes_by.get(f["group"], [np.zeros((1, 4))]))
  m = P[len(P) // 2]
  feats.append({"type": "Feature", "properties": {"kind": "flag", "group": f["group"], "why": "; ".join(f["why"]), "z": round(float(m[2]), 1), "covered": False, "deck": False, "on_deck": False},
                "geometry": {"type": "Point", "coordinates": lonlat(m[None, :2])[0]}})
json.dump({"type": "FeatureCollection", "features": feats}, open(out, "w"))
print(len(feats), "features ->", out)
