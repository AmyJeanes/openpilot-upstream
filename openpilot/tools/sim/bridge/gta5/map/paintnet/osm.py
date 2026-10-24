"""osm.py x0 x1 y0 y1 groups.json links.json dir.json junctions.json out.osm out.geojson: the pilot's lanes as
OpenStreetMap ways, tagged as LA's own map tags its roads (survey of downtown LA's OSM, Oct 2026).

A way is a carriageway: the lanes side by side with no kerb between them, one direction or both where only paint
separates them (a centre line, a painted median, a two-way left-turn lane), drawn down the middle of its outer lane
edges. Lanes first become tracks (a lane and the lanes it carries on into, by connect.py's links, the breaks between
them filled in). At every track sample its neighbours either side are found (bands touching within ADJ m, same deck,
parallel); a gore between two lanes parts them, as a kerb does, except a gore whose wide end meets a junction (a
painted island in a junction approach, which only stops lane changes). The cross-section at a sample is the chain
of neighbours; a section is a stretch where it holds the same lanes, and a way is a section cut where its layer, its
lanes' change rules change or its lanes stray WIDTH_STEP from a straight taper, each piece with its own lanes' widths
(width:lanes:start / :end where they narrow or widen along it). Sections that carry on from one another share a node at the end of the one with
most lanes (a fork's or merge's trunk, so at the gore's tip); the others ease into it, a gore's branches over the
gore's length, other ways at TAPER m per m sideways. A one-lane branch of a fork or merge beside a branch of two lanes
or more (a ramp, a free turn lane round a junction) is a <class>_link of the road it leaves or joins, unnamed.
Junctions are junctions.py's boxes: lanes inside are guide lines (dropped); the ways at their mouths run on to one
node where the arms' lines (through their mouth lanes' ends) best cross. Crossings become footway=crossing ways,
noded where they cross a road. Signals (GTA's traffic light nodes in the box) go on each arm at its stop line with
traffic_signals:direction (stop_paint.py's tags, as on the live map); the crossings there are crossing=traffic_signals.
Restriction relations at the junction's node: no_u_turn back into a two-way arm, and no_left_turn / no_right_turn /
no_straight_on for a move between arms no lane makes (junctions.py's paths); likewise where roads fork or merge.
Tags: lanes (+:forward/:backward/:both_ways); turn:lanes from junctions.py's turns at the mouths ("none": no arrow),
merge_to_left/right on a lane merging into another, elsewhere listed arrows (a lane whose arrow is unknown leaves the
tag off and is flagged); change:lanes from the paint between lanes, width:lanes (:start / :end on a taper), width (kerb to kerb, gutter in; none on a taper),
shoulder (+:width), divider=solid_line for a painted median (and a raised island in one: the way isn't split there),
bridge/layer where the way's centre has another surface DECK m below. GTA path nodes only hint: highway class,
maxspeed and street name, each kept the same along a road's lanes. Every node has ele (z).
Anything undecided goes in each way's fixme tag and the geojson's "issues"."""
import json, math, os, sys, collections
from xml.sax.saxutils import quoteattr
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from shapely.geometry import Polygon, Point, LineString
from shapely.prepared import prep
import scan
import arrows

sys.path.insert(0, "/home/amy/git/op-ortho/openpilot/tools/sim/bridge/gta5/map/webmap")
import gframe

x0, x1, y0, y1 = scan.x0, scan.x1, scan.y0, scan.y1
grpj, linkj, dirj, junj, out_osm, out_geo = sys.argv[5:11]

ADJ = 0.5  # m between two lanes' bands that still share a line: measured 0 +- 0.3 at shared lines (connect.py's EDGE)
PARALLEL, DZ = 30.0, 1.5  # deg, m: a neighbour runs alongside on this deck (direction.py's FLOW_ANG, connect.py's deck)
MINLEN, GAP = 6.0, 3.0  # m: lanes.py's shortest line, and its longest break inside one lane
NARROW = 3.5  # m: no lane is narrower (measured 4.1 m up), no shoulder wider than 3.2 (direction.py: none in 3.5-4.5)
SHOULDER = 4.0  # m: direction.py's widest shoulder; a kerb further out than this is not this carriageway's
ATTACH = 8.0  # m from a way's end to its junction's strips (0-1.8 m on the interchange; the nearest other ends 24 m)
# m along per m sideways a way's centre moves into a shared node: lanes open and close at 0.20-0.46 m of width per m
# on the interchange (motorway 0.2, city bays 0.3-0.46; median 0.34), moving the carriageway's centre half that
TAPER = 6.0
# a two-way road parting into two one-way roads round a kerbed island: each eases in at this m along per m sideways
# (its own lanes part at the island's nose, not over a lane's opening)
DUAL_TAPER = 2.0
ARM = 10.0  # m: a junction arm's heading is its last ARM m (connect.py's frame looks 10 m along)
DECK, DECK_CLOSE = 2.5, 15  # m: a surface this far below is a road under a deck (export.py's covered), gaps closed
ARROW_ALIGN, ARROW_DZ, ARROW_SPILL = 30.0, 2.5, 0.5  # deg, m, m outside a lane: an arrow is in it (ynd_to_osm.py's)
EDGE_BOX = 6.0  # m: ends this near the box's edge are cut by it (lanes.py samples lines 5 m past it)
TAPER_POINT = 1.5  # m: a lane narrower than this all along a way is a taper's point there, not yet (or no longer) a lane
# a lane narrowing or widening along a way is width:lanes:start / :end on it (OSM wiki Key:lanes: a way is split where
# a lane starts or ends, not as its width moves), the way cut where its lanes stray WIDTH_STEP m from a straight taper
# between its ends' widths (each the median of its END_N end samples); under TAPER_MIN m of change is no taper
WIDTH_STEP, TAPER_MIN, END_N = 1.0, 0.5, 3
SMOOTH = 5  # samples: a way's centre averaged over this (one sample jumps where a lane's band widens at a junction)
SIMPLIFY = 0.1  # m: way nodes dropped while the line stays this near them (under the 0.3 m spread of shared edges)
# mph by class and GTA speed class (slow, normal, fast, faster): ynd_to_osm.py's LIMITS, a hint only
LIMITS = {"motorway": (25, 40, 55, 65), "trunk": (25, 45, 50, 55), "primary": (25, 40, 45, 50), "residential": (20, 30, 50, 50)}
FREEWAY, TRAFFIC_LIGHT = 64, 15  # GTA node flags 2 bit; node special (flags 1 >> 3)


def unit(v):
  return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def frame(A):
  """unit travel direction (over +-2 samples: a single metre wobbles) and the unit normal to its right."""
  k = np.arange(len(A))
  d = unit(A[np.minimum(k + 2, len(A) - 1), :2] - A[np.maximum(k - 2, 0), :2])
  return d, np.stack([d[:, 1], -d[:, 0]], 1)


def band(A):
  """a run's area: its centre +- half its width, half a sample past each end (as export.py's area)."""
  d, n = frame(A)
  xy = A[:, :2].copy()
  xy[0] -= d[0] * 0.5
  xy[-1] += d[-1] * 0.5
  h = A[:, 3:4] / 2
  if len(A) < 2:
    return Point(xy[0]).buffer(max(float(h[0, 0]), 0.3))
  return Polygon(np.vstack([xy - n * h, (xy + n * h)[::-1]])).buffer(0)


def covered(P, sign=1):
  """per point: another surface more than DECK m above it (under a deck); sign -1: below it (on a deck). export.py's."""
  out = np.zeros(len(P), bool)
  for k, (x, y, z) in enumerate(P[:, :3]):
    tx, ty = int(np.floor(x / scan.rp.TILE)), int(np.floor(y / scan.rp.TILE))
    t = scan.T._get(tx, ty)
    if t is None:
      continue
    c, zz = t
    col = min(int((x - tx * scan.rp.TILE) / scan.rp.RES), scan.rp.N - 1)
    row = min(int((ty * scan.rp.TILE + scan.rp.TILE - y) / scan.rp.RES), scan.rp.N - 1)
    out[k] = bool(((c[:, row, col] > 0) & (sign * (zz[:, row, col] - z) > DECK)).any())
  return out


def runs_of(v, minlen):
  """[start, end, value) stretches of equal values; one shorter than minlen joins the stretch before it (the first,
  the one after) and takes its value."""
  r = []
  for k in range(len(v)):
    if r and v[k] == r[-1][2]:
      r[-1][1] = k + 1
    else:
      r.append([k, k + 1, v[k]])
  while len(r) > 1:
    q = next((q for q in range(len(r)) if r[q][1] - r[q][0] < minlen), None)
    if q is None:
      break
    if q > 0:
      r[q - 1][1] = r[q][1]
    else:
      r[1][0] = r[0][0]
    del r[q]
    m = [r[0]]
    for e in r[1:]:
      if e[2] == m[-1][2]:
        m[-1][1] = e[1]
      else:
        m.append(e)
    r = m
  return [tuple(e) for e in r]


issues_free = []  # (x, y, why) not tied to a way
G = json.load(open(grpj))
LK = json.load(open(linkj))
for k, p in LK.get("pts", {}).items():  # openings reshaped by connect.py
  G["lanes"][int(k)]["pts"] = p
lanes = G["lanes"]
LA = [np.array(r["pts"], float) for r in lanes]
strips = collections.defaultdict(list)  # kind -> [(index in dir.json, pts)]
for k, r in enumerate(json.load(open(dirj))):
  if r["kind"] != "lane":
    strips[r["kind"]].append((k, np.array(r["pts"], float)))
at_edge = lambda p: min(p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1]) < EDGE_BOX

# --- junctions: junctions.py's boxes, the lanes ending and starting at their mouths, and the lanes inside them
dir_at = {k: (kind, A) for kind, L_ in strips.items() for k, A in L_}
junctions = []
for r in json.load(open(junj)):
  junctions.append({"hull": Polygon(r["ring"]).buffer(0), "z": r["z"], "strips": [(dir_at[k][0], k) for k in r["strips"]],
                    "crossings": [dir_at[k][1] for k in r["strips"] if dir_at[k][0] == "crossing"], "mouths": r["mouths"],
                    "inside": set(r["inside"]), "paths": r.get("paths", []), "ends": [], "issues": []})
end_turns = {i: tu for J in junctions for i, which, tu in J["mouths"] if which == "end"}
merge_into = {i: j for i, j, g, *br in LK["links"] if br == ["merge"]}

# --- lanes kept: known direction, MINLEN long, not painted through a junction
keep = []
for i, A in enumerate(LA):
  if lanes[i]["dir"] == "?":
    issues_free.append((*A[len(A) // 2, :2], f"lane {i}: no direction"))
    continue
  if len(A) < MINLEN:
    continue  # stubs between a stop line and a crossing
  if any(i in J["inside"] for J in junctions):
    continue  # a guide line through a junction
  keep.append(i)

# --- tracks: lanes chained by their links (a break filled in), and by an end meeting a start within GAP m in line
succ, pred, branch_links, other_links = collections.defaultdict(list), collections.defaultdict(list), [], []
for i, j, g, *br in LK["links"]:
  if i not in keep or j not in keep:
    continue
  if br:
    branch_links.append((i, j))
  else:
    succ[i].append(j)
    pred[j].append(i)
joins, implied = {}, []
for i, js_ in succ.items():
  if len(js_) == 1 and len(pred[js_[0]]) == 1:
    joins[i] = js_[0]
  else:
    other_links += [(i, j) for j in js_]
branch_to = {j for _, j in branch_links}
for i in keep:
  if i in succ:
    continue
  A = LA[i]
  da, _ = frame(A)
  for j in keep:
    if j == i or j in pred or j in branch_to or j in joins.values():
      continue
    B = LA[j]
    db, nb = frame(B)
    along = (B[0, :2] - A[-1, :2]) @ da[-1]
    o = (B[0, :2] - A[-1, :2]) @ nb[0]
    overlap = min(A[-1, 3] / 2, o + B[0, 3] / 2) - max(-A[-1, 3] / 2, o - B[0, 3] / 2)  # the bands side by side
    if 0 < along <= GAP and overlap >= min(A[-1, 3], B[0, 3]) / 2 and abs(A[-1, 2] - B[0, 2]) <= DZ and da[-1] @ db[0] > math.cos(math.radians(PARALLEL)):
      joins[i] = j
      implied.append((i, j, round(float(A[-1, 3]), 1), round(float(B[0, 3]), 1)))
      break
starts = [i for i in keep if i not in joins.values()]
tracks = []  # {"lanes": [...], "X": N x 4, "src": lane per sample (-1 filled)}
for i in starts:
  chain = [i]
  while chain[-1] in joins:
    chain.append(joins[chain[-1]])
  X, src = [LA[chain[0]]], [np.full(len(LA[chain[0]]), chain[0])]
  for a, b in zip(chain, chain[1:]):
    p, q = X[-1][-1], LA[b][0]
    m = int(np.linalg.norm(q[:2] - p[:2]))
    if m > 1:
      t = np.arange(1, m)[:, None] / m
      X.append(p + (q - p) * t)
      src.append(np.full(m - 1, -1))
    X.append(LA[b])
    src.append(np.full(len(LA[b]), b))
  tracks.append({"lanes": chain, "X": np.vstack(X), "src": np.concatenate(src)})
lane_track = {l: (t, int(np.flatnonzero(T["src"] == l)[0]), int(np.flatnonzero(T["src"] == l)[-1])) for t, T in enumerate(tracks) for l in T["lanes"]}
# a narrow strip that carries on into no lane is the road's shoulder, not a lane (a lane narrowing in a track stays)
narrow_tracks = [t for t, T in enumerate(tracks) if np.median(T["X"][:, 3]) < NARROW and len(T["lanes"]) == 1]
for t in narrow_tracks:
  strips["shoulder"].append((-1, tracks[t]["X"]))
  issues_free.append((*tracks[t]["X"][len(tracks[t]["X"]) // 2, :2], f"lane {tracks[t]['lanes'][0]}: {np.median(tracks[t]['X'][:, 3]):.1f} m wide, taken as shoulder"))
for T in tracks:
  T["d"], T["n"] = frame(T["X"])

# --- neighbours: the nearest band either side; joined unless a kerb, or a gore not at a junction, lies between
gore_at_j = []
for k, A in strips["gore"]:
  ends = [Point(A[0, :2]), Point(A[-1, :2])]
  wide = ends[int(A[-1, 3] > A[0, 3])]
  gore_at_j.append(any(J["hull"].distance(wide) <= ATTACH and abs(J["z"] - A[0, 2]) < 2 for J in junctions))
gores = [(prep(band(A).buffer(0.3)), atj, np.median(A[:, 2])) for (k, A), atj in zip(strips["gore"], gore_at_j)]
# (a raised island in a median keeps the road one way: it is the median carried on, kerbed)
centres = [(prep(band(A).buffer(0.3)), "median" if kind == "island" else kind, np.median(A[:, 2])) for kind in ("median", "twltl", "island") for k, A in strips[kind]]
live = [t for t in range(len(tracks)) if t not in narrow_tracks]
S_xy = np.vstack([tracks[t]["X"][:, :2] for t in live])
S_id = np.concatenate([np.stack([np.full(len(tracks[t]["X"]), t), np.arange(len(tracks[t]["X"]))], 1) for t in live])
tree = cKDTree(S_xy)


def between(p, z):
  """what lies at p between two bands on this deck: ("gore", at a junction?), ("median"|"twltl",) or None."""
  pt = Point(p)
  for g, atj, zm in gores:
    if abs(zm - z) < 2 and g.contains(pt):
      return ("gore", atj)
  for g, kind, zm in centres:
    if abs(zm - z) < 2 and g.contains(pt):
      return (kind,)
  return None


NB = {}
for t in live:
  T = tracks[t]
  NB[t] = []
  for k, (x, y, z, w) in enumerate(T["X"]):
    c, d, n = np.array([x, y]), T["d"][k], T["n"][k]
    best = {}
    for idx in tree.query_ball_point(c, 15.0):
      t2, m = S_id[idx]
      if t2 == t:
        continue
      B = tracks[t2]
      off = B["X"][m, :2] - c
      along = off @ d
      if abs(along) > 1.0 or abs(B["X"][m, 2] - z) > DZ or abs(d @ B["d"][m]) < math.cos(math.radians(PARALLEL)):
        continue
      o = off @ n
      side = "R" if o > 0 else "L"
      gap = abs(o) - B["X"][m, 3] / 2 - w / 2
      if side not in best or gap < best[side][2] - 1e-6 or (abs(gap - best[side][2]) < 1e-6 and abs(along) < best[side][4]):
        best[side] = (int(t2), int(m), gap, bool(d @ B["d"][m] > 0), abs(along))
    nb = {}
    for side, (t2, m, gap, same, _) in best.items():
      sg = 1 if side == "R" else -1
      mid = c + n * sg * (w / 2 + max(gap, 0) / 2)
      what = between(mid, z)
      if gap <= ADJ and what is None:
        joint = "line"
      elif what and what[0] == "gore":
        joint = "gore" if what[1] else None
      elif what and gap > ADJ:
        joint = what[0]
      elif gap <= NARROW and scan.extent(c + n * sg * w / 2, n * sg, z, 1)[0] >= gap - 0.1:
        joint = "open"  # unpainted road between (a junction's mouth): no kerb, so one carriageway
      else:
        joint = None
      if joint:
        nb[side] = (t2, m, joint, same)
    NB[t].append(nb)


def section_at(t, k):
  """the cross-section through track t's sample k: [(track, sample, offset right of it, width, sign)] left to right,
  and the joint between each pair."""
  T = tracks[t]
  c, n = T["X"][k, :2], T["n"][k]
  mem, joints = [(t, k, 0.0, T["X"][k, 3], 1)], []
  for side in ("R", "L"):
    cur, s = (t, k), 1
    seen = {t}
    while True:
      look = side if s > 0 else ("L" if side == "R" else "R")
      nb = NB[cur[0]][cur[1]].get(look)
      if not nb or nb[0] in seen or nb[0] not in NB:
        break
      t2, m, joint, same = nb
      s = s if same else -s
      seen.add(t2)
      B = tracks[t2]["X"][m]
      e = (t2, m, float((B[:2] - c) @ n), float(B[3]), s)
      if side == "R":
        mem.append(e)
        joints.append(joint)
      else:
        mem.insert(0, e)
        joints.insert(0, joint)
      cur = (t2, m)
  return mem, joints


XS = {t: [section_at(t, k) for k in range(len(tracks[t]["X"]))] for t in live}

# --- sections: stretches of each track with the same lanes alongside, joined across the tracks they hold
run_of = {}
runs = []
for t in live:
  keys = [frozenset(m[0] for m in XS[t][k][0]) for k in range(len(XS[t]))]
  for a, b, _ in runs_of(keys, MINLEN):
    for k in range(a, b):
      run_of[t, k] = len(runs)
    runs.append((t, a, b))
rp_ = list(range(len(runs)))


def rroot(i):
  while rp_[i] != i:
    rp_[i] = rp_[rp_[i]]
    i = rp_[i]
  return i


# a run is joined to the runs alongside its middle (its ends may be a metre or two off its neighbours')
for i, (t, a, b) in enumerate(runs):
  for t2, m, *_ in XS[t][(a + b) // 2][0]:
    ra, rb = rroot(i), rroot(run_of[t2, m])
    if ra != rb:
      rp_[rb] = ra
sec_runs = collections.defaultdict(list)
for i in range(len(runs)):
  sec_runs[rroot(i)].append(i)
sections = []
for members in sec_runs.values():
  ref = max(members, key=lambda i: runs[i][2] - runs[i][1])
  t, a, b = runs[ref]
  T = tracks[t]
  mid = (a + b) // 2
  mem, joints = XS[t][mid]
  flip = sum(e[4] < 0 for e in mem) > sum(e[4] > 0 for e in mem)
  # the way's line (tagged placement where it isn't the middle of all its lanes, as LA maps it): a two-way road's
  # centre line between its directions; a one-way road's middle of its full lanes, a lane opening or merging beside
  # them (narrower than any lane somewhere along here: a taper, a slip lane's merge, a motorway's acceleration lane)
  # counted in its lanes but not moving its line
  signs = [e[4] for e in mem]
  span_ = collections.defaultdict(list)
  for i in members:
    span_[runs[i][0]] += [runs[i][1], runs[i][2]]
  side = {t2 for t2, v in span_.items() if tracks[t2]["X"][min(v):max(v), 3].min() < NARROW}
  two_way = len(set(signs)) == 2 and sum(signs[q] != signs[q + 1] for q in range(len(signs) - 1)) == 1 and "twltl" not in joints
  C, mids, W = [], [], {}
  for k in range(a, b):
    m = XS[t][k][0]
    W[k] = {e[0]: e[3] for e in m}  # each lane's width here
    lo = min(e[2] - e[3] / 2 for e in m)
    hi = max(e[2] + e[3] / 2 for e in m)
    mid = None  # (a sample or two without both directions, or without a full lane: a short run merged in)
    if two_way:
      q = next((q for q in range(len(m) - 1) if m[q][4] != m[q + 1][4]), None)
      if q is not None:
        mid = (m[q][2] + m[q][3] / 2 + m[q + 1][2] - m[q + 1][3] / 2) / 2
    else:
      core = [e for e in m if e[0] not in side]
      if core:
        mid = (min(e[2] - e[3] / 2 for e in core) + max(e[2] + e[3] / 2 for e in core)) / 2
    mids.append(mid)
    C.append([T["X"][k, 2], hi - lo, k])
  # those samples keep the way where it is beside them
  known = [q for q, v in enumerate(mids) if v is not None]
  for q in range(len(mids)):
    if mids[q] is None:
      mids[q] = mids[min(known, key=lambda r: abs(r - q))] if known else 0.0
  C = np.array([[*(T["X"][k, :2] + T["n"][k] * md), z, w, k] for (z, w, k), md in zip(C, mids)])
  C[:, :2] = ndi.uniform_filter1d(C[:, :2], SMOOTH, axis=0, mode="nearest")
  if flip:
    C = C[::-1]
  sections.append({"ref": (t, a, b), "flip": flip, "runs": members, "tracks": {runs[i][0] for i in members}, "C": C,
                   "mem": mem, "joints": joints, "issues": [], "two_way": two_way, "side": side, "W": W})
sec_of_run = {i: s for s, S in enumerate(sections) for i in S["runs"]}

# --- section ends that meet: a track carrying on from one section into the next, or a link between tracks
port = {}
pparent = {}
made = set()  # (section end traffic leaves by, section end it carries on into): the moves lanes make where sections meet


def pfind(p):
  while pparent.get(p, p) != p:
    p = pparent[p]
  return p


def end_near(s, q):
  C = sections[s]["C"]
  return (s, int(np.linalg.norm(C[-1, :2] - q) < np.linalg.norm(C[0, :2] - q)))


def meet(p1, p2):
  a, b = pfind(p1), pfind(p2)
  if a != b:
    pparent[b] = a


for t in live:
  X = tracks[t]["X"]
  for k in range(1, len(X)):
    r0, r1 = run_of[t, k - 1], run_of[t, k]
    s0, s1 = sec_of_run[r0], sec_of_run[r1]
    if s0 != s1:
      q = X[k, :2]
      meet(end_near(s0, q), end_near(s1, q))
      made.add((end_near(s0, q), end_near(s1, q)))
for i, j in other_links + branch_links:
  ti, a_, b_ = lane_track[i]
  tj, aj, _ = lane_track[j]
  if ti not in XS or tj not in XS:
    continue
  X = tracks[ti]["X"]
  if merge_into.get(i) == j:
    # a lane merging into another meets it where it ends; nothing to join where the two run in one section (an
    # acceleration lane, a slip lane's merge: tagged merge_to_*)
    if any(ti in S["tracks"] and tj in S["tracks"] for S in sections):
      continue
    p = X[b_, :2]
    kj = int(np.linalg.norm(tracks[tj]["X"][:, :2] - p, axis=1).argmin())
    si, sj = sec_of_run[run_of[ti, b_]], sec_of_run[run_of[tj, kj]]
    if si != sj:
      meet(end_near(si, p), end_near(sj, p))
      made.add((end_near(si, p), end_near(sj, p)))
    continue
  ki = int(np.linalg.norm(X[:, :2] - tracks[tj]["X"][aj, :2], axis=1).argmin()) if (i, j) in branch_links else b_
  si, sj = sec_of_run[run_of[ti, ki]], sec_of_run[run_of[tj, aj]]
  if si != sj:
    meet(end_near(si, tracks[tj]["X"][aj, :2]), end_near(sj, tracks[tj]["X"][aj, :2]))
    made.add((end_near(si, tracks[tj]["X"][aj, :2]), end_near(sj, tracks[tj]["X"][aj, :2])))
clusters = collections.defaultdict(list)
for s in range(len(sections)):
  for e in (0, 1):
    clusters[pfind((s, e))].append((s, e))

# --- lanes of each section in way order, and their tags
arrow_list = [a for a in scan.src.features() if a["kind"] == "arrow" and scan.inbox(a["x"], a["y"], 10)]
code_, zz_, _, shapes = arrows.box(x0 - 20, x1 + 20, y0 - 20, y1 + 20, 0.1)
shapes = [s for s in shapes if all(math.hypot(s[0] - a["x"], s[1] - a["y"]) > 3.0 for a in arrow_list)]


def in_lane(x, y, z, head=None):
  """(track, sample) whose band holds a point on its deck (and, given a heading, runs that way), or None."""
  best = None
  for idx in tree.query_ball_point([x, y], 6.0):
    t, k = S_id[idx]
    X = tracks[t]["X"][k]
    off = np.array([x, y]) - X[:2]
    if abs(off @ tracks[t]["n"][k]) > X[3] / 2 + ARROW_SPILL or abs(off @ tracks[t]["d"][k]) > 1.0 or abs(X[2] - z) > ARROW_DZ:
      continue
    if head is not None and np.array([math.cos(math.radians(head)), math.sin(math.radians(head))]) @ tracks[t]["d"][k] < math.cos(math.radians(ARROW_ALIGN)):
      continue
    dd = np.linalg.norm(off)
    if best is None or dd < best[0]:
      best = (dd, int(t), int(k))
  return best and best[1:]


arrows_at = collections.defaultdict(list)  # track -> [(sample, type)]
for a in arrow_list:
  hit = in_lane(a["x"], a["y"], a["z"], a["heading"])
  if hit:
    arrows_at[hit[0]].append((hit[1], a["arrow"]))
# an arrow found only by its shape takes the type its paint matches, as junctions.py's (else it's of unknown type)
TMPL = arrows.templates(scan.T, scan.src.features(), os.path.expanduser("~/.cache/paintnet/arrow_templates.npz"))
unknown_at = collections.defaultdict(list)
for x, y, z, _ in shapes:
  hit = in_lane(x, y, z)
  if hit:
    kind = arrows.kind(scan.T, x, y, z, tracks[hit[0]]["d"][hit[1]], TMPL)
    if kind:
      arrows_at[hit[0]].append((hit[1], kind))
    else:
      unknown_at[hit[0]].append(hit[1])


def line_styles(c, n, z, e):
  """the paint lines within 0.6 m of offset e across a section (a double line's two halves), left to right."""
  return [x for x in scan.crossings(c, n, z, e - 0.6, e + 0.6)]


for s, S in enumerate(sections):
  t0, a, b = S["ref"]
  flip = S["flip"]
  order = S["mem"] if not flip else S["mem"][::-1]  # way order, left to right
  joints = S["joints"] if not flip else S["joints"][::-1]
  sg = -1 if flip else 1
  S["order"] = [(e[0], e[4] * sg) for e in order]  # (track, +1 forward / -1 backward)
  S["joints_way"] = joints
  rng = {}
  for i in S["runs"]:
    t, ra, rb = runs[i]
    rng.setdefault(t, [ra, rb])
    rng[t] = [min(rng[t][0], ra), max(rng[t][1], rb)]
  S["rng"] = rng
  S["width_lanes"] = {t: float(np.median(tracks[t]["X"][rng[t][0]:rng[t][1], 3])) for t, _ in S["order"] if t in rng}
  # per sample along the way: lane change rules at each joint between same-direction lanes
  X = tracks[t0]["X"]
  ks = S["C"][:, 4].astype(int)
  ch = []
  prev = None
  for k in ks:
    mem, jn = XS[t0][k]
    allow = {t: [True, True] for t, _ in S["order"]}  # own left, own right; lanes not alongside here keep yes
    seen_line = False
    c, nrm, z = X[k, :2], tracks[t0]["n"][k], X[k, 2]
    for q in range(len(mem) - 1):
      A_, B_ = mem[q], mem[q + 1]
      if A_[4] != B_[4]:
        continue
      if jn[q] == "gore":
        okL = okR = False
      else:
        e = (A_[2] + A_[3] / 2 + B_[2] - B_[3] / 2) / 2
        ls = line_styles(c, nrm, z, e)
        if not ls:
          continue
        okL, okR = ls[0][3] == "dashed", ls[-1][3] == "dashed"  # left lane crosses the line nearest it first
      if A_[0] not in allow or B_[0] not in allow:
        continue
      seen_line = True
      # A_ is on the ref's left of B_: A_ crosses toward ref-right, B_ toward ref-left
      allow[A_[0]][1 if A_[4] > 0 else 0] = okL
      allow[B_[0]][0 if B_[4] > 0 else 1] = okR
    v = tuple("yes" if l and r else "no" if not l and not r else "not_left" if r else "not_right" for l, r in (allow[tt] for tt, _ in S["order"]))
    v = v if seen_line or not prev else prev  # no line read here (between dashes of a gap): as before
    ch.append(v)
    prev = v
  S["change"] = ch
  # on a deck: gaps under DECK_CLOSE m between deck stretches closed (export.py's pieces)
  dk = covered(S["C"][:, :3], -1)
  S["deck"] = dk | ndi.binary_closing(np.r_[False, dk, False], np.ones(DECK_CLOSE, bool))[1:-1]

# --- nodes where sections meet: at the end of the section with most lanes (a fork's trunk: the gore's tip)
nodes = {}  # id -> [x, y, z, tags]
nid = [0]


def new_node(x, y, z, tags=None):
  nid[0] -= 1
  nodes[nid[0]] = [float(x), float(y), float(z), dict(tags or {})]
  return nid[0]


port_node, port_reach, port_rate = {}, {}, {}
for cl, ports in clusters.items():
  if len(ports) < 2:
    continue
  nl = [len(sections[s]["order"]) for s, e in ports]
  top = [p for p, n_ in zip(ports, nl) if n_ == max(nl)]
  P = np.mean([sections[s]["C"][-1 if e else 0, :3] for s, e in top], axis=0)
  nd = new_node(*P)
  # a fork or merge at a gore's tip: its branches come off the node along the gore, reaching their own lanes' middle
  # where it ends (the gore's length); elsewhere they ease in at TAPER
  tips = [len(A) for _, A in strips["gore"] if abs(A[0, 2] - P[2]) < 2 and
          min(np.linalg.norm(A[0, :2] - P[:2]), np.linalg.norm(A[-1, :2] - P[:2])) <= max(sections[s]["C"][:, 3].max() for s, _ in ports) / 2]
  dual = all(sections[s]["two_way"] for s, _ in top)
  # a one-lane branch meeting its road at a junction's mouth (a free turn lane round the corner) follows its own lane
  # and meets the road's node in one straight stretch, as mapped: eased in, its whole length would leave its lane
  at_mouth = any(abs(J["z"] - P[2]) < 2 and J["hull"].distance(Point(P[:2])) <= ATTACH for J in junctions)
  for p in ports:
    port_node[p] = nd
    port_reach[p] = float(max(tips)) if tips and p not in top else None
    port_rate[p] = DUAL_TAPER if dual and p not in top and len({d for _, d in sections[p[0]]["order"]}) == 1 else TAPER
    if at_mouth and p not in top and len(sections[p[0]]["order"]) == 1:
      port_rate[p] = port_reach[p] = None


def blend(C, e, q, share, reach=None, rate=TAPER):
  """move a way's end e (0 start, 1 end) onto q, easing in over `reach` m, or `rate` m per m it moves sideways within
  `share` of the way's length."""
  C = C.copy()
  i = -1 if e else 0
  delta = q[:2] - C[i, :2]
  dist = np.linalg.norm(delta)
  if dist < 1e-3:
    C[i, :2] = q[:2]
    return C
  s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C[:, :2], axis=0), axis=1))]
  s = s if not e else s[-1] - s
  L = min(reach, s.max()) if reach else min(rate * dist, s.max() * share if s.max() > 0 else 1)
  w = np.clip(1 - s / max(L, 1e-6), 0, 1)
  C[:, :2] += w[:, None] * delta
  return C


for s, S in enumerate(sections):
  moves = [np.linalg.norm(np.array(nodes[port_node[(s, e)]][:2]) - S["C"][-1 if e else 0, :2]) if (s, e) in port_node else 0 for e in (0, 1)]
  for e in (0, 1):
    if (s, e) in port_node and port_rate[(s, e)] is not None:
      q = np.array(nodes[port_node[(s, e)]][:3])
      S["C"] = blend(S["C"], e, q, 0.5 if moves[1 - e] > ADJ else 1.0, port_reach[(s, e)], port_rate[(s, e)])  # the whole way when its other end stays

# --- junctions: the ways at a junction's mouths run on to one node where the arms' lines best cross
gta_nodes = [n for n in scan.nodes.values() if scan.inbox(n["x"], n["y"], 20)]
def way_end(s, e):
  """a section's end point and the way it heads out there (over its last ARM m)."""
  C = sections[s]["C"]
  p = C[-1 if e else 0]
  s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C[:, :2], axis=0), axis=1))]
  back = C[np.searchsorted(s_, s_[-1] - ARM)] if e else C[min(np.searchsorted(s_, ARM), len(C) - 1)]
  return p, unit(p[:2] - back[:2])


# the section ends at a junction's mouths, then any other dead end ATTACH m short of a box it heads into. Each mouth
# lane's track offers the end nearest the mouth of the sections holding it; the end holding most of the mouths' lanes
# goes first, and an end whose mouth lanes are all reached already isn't an arm (a stretch where only an arm's leaving
# lanes have started, short of its arriving lanes' stop line, is dropped: the arm's way runs on to the junction, both
# ways). A mouth's end may also be where another section meets it (a free turn lane leaving or joining at the mouth):
# the arm still runs on to the junction, the other way meeting it at its end
dead = [(s, e) for s in range(len(sections)) for e in (0, 1) if (s, e) not in port_node]
ext = {}  # (s, e) -> junction node
attached, met = set(), set()
for J in junctions:
  mouth_t = {lane_track[i][0]: (i, which) for i, which, _ in J["mouths"] if i in lane_track and lane_track[i][0] in XS}
  cand = set()
  for t, (i, which) in mouth_t.items():
    _, a_, b_ = lane_track[i]
    q = tracks[t]["X"][b_ if which == "end" else a_, :2]
    cand.add(min(((s, e) for s, S in enumerate(sections) if t in S["tracks"] for e in (0, 1)),
                 key=lambda se: np.linalg.norm(sections[se[0]]["C"][-1 if se[1] else 0, :2] - q)))
  cov = {se: {t for t in mouth_t if t in sections[se[0]]["tracks"]} for se in cand}
  gap = {se: J["hull"].distance(Point(sections[se[0]]["C"][-1 if se[1] else 0, :2])) for se in cand}
  reached = set()
  for s, e in sorted(cand, key=lambda se: (-len(cov[se]), round(gap[se], 1), -len(sections[se[0]]["order"]))):
    if (s, e) in attached or pfind((s, e)) in met or gap[s, e] > ATTACH:
      continue
    if cov[s, e] <= reached:
      if sections[s]["tracks"] <= set(mouth_t):
        sections[s]["drop"] = True
      continue
    if (s, e) in port_node:
      met.add(pfind((s, e)))
    attached.add((s, e))
    reached |= cov[s, e]
    J["ends"].append((s, e, *way_end(s, e)))
    J.setdefault("arm_lanes", {})[s, e] = [mouth_t[t] for t in cov[s, e]]
    # the arm's line for the junction's node: through its mouth lanes' ends, the way they run there (the way's own end
    # leans where a lane opens or merges beside it)
    M = [(LA[i][-1 if which == "end" else 0], frame(LA[i])[0][-1 if which == "end" else 0]) for t, (i, which) in mouth_t.items() if t in cov[s, e]]
    J.setdefault("arms", []).append((np.mean([m[0][:2] for m in M], 0), unit(np.sum([m[1] * np.sign(m[1] @ M[0][1]) for m in M], 0))))
for s, e in dead:
  if (s, e) in attached or sections[s].get("drop"):
    continue
  p, out_dir = way_end(s, e)
  best = None
  for j, J in enumerate(junctions):
    if abs(J["z"] - p[2]) > 2:
      continue
    dd = J["hull"].distance(Point(p[:2]))
    ahead = (np.array(J["hull"].centroid.coords[0]) - p[:2]) @ out_dir > 0
    if dd <= ATTACH and ahead and (best is None or dd < best[0]):
      best = (dd, j)
  if best:
    junctions[best[1]]["ends"].append((s, e, p, out_dir))
  elif at_edge(p[:2]):
    pass
  else:
    sections[s]["issues"].append(f"dead end at its {'end' if e else 'start'}: no link and no junction near")
for J in junctions:
  if len({s for s, *_ in J["ends"]}) < 2:
    if J["ends"]:
      for s, e, p, _ in J["ends"]:
        sections[s]["issues"].append(f"ends at a junction with only this arm in the box: not connected")
    continue
  M, bb = np.zeros((2, 2)), np.zeros(2)
  for p, u in J.get("arms") or [(p, u) for s, e, p, u in J["ends"]]:
    Q = np.eye(2) - np.outer(u, u)
    M += Q
    bb += Q @ p[:2]
  if np.linalg.eigvalsh(M)[0] > 0.3 and J["hull"].buffer(5).contains(Point(np.linalg.solve(M, bb))):
    xy = np.linalg.solve(M, bb)
  else:
    xy = np.array(J["hull"].centroid.coords[0])
    J["issues"].append("arms' lines don't cross inside it: node at the strips' centre")
  sig = [n for n in gta_nodes if (n["f"][1] >> 3) == TRAFFIC_LIGHT and J["hull"].buffer(5).contains(Point(n["x"], n["y"]))]
  gn = min(gta_nodes, key=lambda n: math.hypot(n["x"] - xy[0], n["y"] - xy[1]) if abs(n["z"] - J["z"]) < 3 else 1e9)
  J["signals"] = bool(sig)  # tagged on each arm at its stop line, below
  J["node"] = new_node(*xy, np.mean([p[2] for _, _, p, _ in J["ends"]]))
  J["gta_offset"] = round(math.hypot(gn["x"] - xy[0], gn["y"] - xy[1]), 1)
  for s, e, p, u in J["ends"]:
    ext[(s, e)] = J["node"]

# --- highway class and speed hints from the GTA links along each section
raw = {}
for (ka, kb), l in scan.links.items():
  raw[(ka, kb)] = l


def gta_hint(C):
  """(class, speed class, street name) from the GTA links along a way, or None."""
  fw, lanes_, sp, st = [], [], [], []
  for p in C[::4]:
    best = None
    for ka, kb, A, B in scan.L:
      if abs((A[2] + B[2]) / 2 - p[2]) > 3:
        continue
      ab = B[:2] - A[:2]
      Lb = np.linalg.norm(ab)
      if Lb < 1:
        continue
      tt = np.clip((p[:2] - A[:2]) @ ab / Lb ** 2, 0, 1)
      dd = np.linalg.norm(A[:2] + tt * ab - p[:2])
      if dd <= p[3] / 2 + 3 and (best is None or dd < best[0]):
        best = (dd, ka, kb)
    if best:
      _, ka, kb = best
      na, nb = scan.nodes[ka], scan.nodes[kb]
      l = raw.get((ka, kb)) or raw.get((kb, ka))
      fw.append(bool(na["f"][2] & nb["f"][2] & FREEWAY))
      lanes_.append(max((l["f"][2] >> 5) & 7, (l["f"][2] >> 2) & 7))
      sp.append(max(na.get("sp", 1), nb.get("sp", 1)))
      st += [scan.streets[n["st"]] for n in (na, nb) if n.get("st") in scan.streets]
  if not fw:
    return None
  cls = "motorway" if np.mean(fw) >= 0.5 else "primary" if max(lanes_) >= 2 else "residential"
  return cls, collections.Counter(sp).most_common(1)[0][0], collections.Counter(st).most_common(1)[0][0] if st else None


hints = [gta_hint(S["C"]) for S in sections]
# a one-lane branch of a fork or merge, beside a branch of two lanes or more, is a link (a motorway's ramp, a free turn
# lane round a junction); so on through 1:1 joins. Its class is the road's it leaves or joins (the most important).
# (GTA doesn't mark them: ynd_to_osm.py's ramps())
LINKS = ("motorway", "trunk", "primary", "secondary", "tertiary")  # classes with a _link, most important first
link_of = {}  # section -> the class it's a link of
for cl, ports in clusters.items():
  if len(ports) >= 3 or cl in met:  # (at a mouth the junction is the other branch)
    nl = {s: len(sections[s]["order"]) for s, _ in ports}
    trunk = max(nl, key=nl.get)
    if hints[trunk] and hints[trunk][0] in LINKS and min(nl.values()) == 1 and max(nl.values()) >= 2:
      for s in (s for s, n_ in nl.items() if n_ == 1):
        link_of[s] = min(link_of.get(s, hints[trunk][0]), hints[trunk][0], key=LINKS.index)
grow = True
while grow:
  grow = False
  for cl, ports in clusters.items():
    ss = {s for s, _ in ports}
    if len(ports) == 2 and cl not in met and len(ss & set(link_of)) == 1:
      cls = link_of[next(iter(ss & set(link_of)))]
      link_of.update({s: cls for s in ss})
      grow = True
# a road keeps its class, speed and name along its lanes: each track takes the value most of its length has, a section
# the value most of its tracks' length has (GTA's links beside a bay or across a junction's mouth read another road's).
# A track's stretch on a link votes apart from its stretch on the road (a ramp's lane carries on from the motorway's).
span = {s: {(t, s in link_of): r[1] - r[0] for t, r in S["rng"].items()} for s, S in enumerate(sections)}
track_len = collections.Counter()
for sp_ in span.values():
  track_len.update(sp_)
for q in range(3):
  vote = collections.defaultdict(collections.Counter)
  for s, h in enumerate(hints):
    if h and h[q] is not None:
      for t, L_ in span[s].items():
        vote[t][h[q]] += L_
  for s, S in enumerate(sections):
    tally = collections.Counter()
    for t in span[s]:
      if vote[t]:
        tally[vote[t].most_common(1)[0][0]] += track_len[t]
    S.setdefault("hint", [None] * 3)[q] = tally.most_common(1)[0][0] if tally else None
for S in sections:
  S["hint"] = tuple(S["hint"]) if S["hint"][0] else None
# a named road keeps one class: the most important along it (GTA lays an arterial as a link per lane in places, which
# read as a one-lane road), with the speed most of its length at that class has
RANK = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential"]
by_name = collections.defaultdict(list)
for s, S in enumerate(sections):
  if S["hint"] and S["hint"][2] and s not in link_of:
    by_name[S["hint"][2]].append(s)
for name, ss in by_name.items():
  cls = min((sections[s]["hint"][0] for s in ss), key=lambda c: RANK.index(c) if c in RANK else len(RANK))
  sp = collections.Counter()
  for s in ss:
    if sections[s]["hint"][0] == cls:
      sp[sections[s]["hint"][1]] += len(sections[s]["C"])
  for s in ss:
    sections[s]["hint"] = (cls, sp.most_common(1)[0][0], name)


def turn_value(S, t, sign):
  """a track's turns in this section: where it ends here at a junction's mouth, junctions.py's ("none": no arrow);
  where it ends merging into another lane, "merge" (its side is worked out by the caller); else the listed arrow in
  its stretch (the last along travel if several), "?" if an arrow of unknown type is painted there, None if none."""
  a, b = S["rng"][t]
  last = tracks[t]["lanes"][-1]
  if b == len(tracks[t]["X"]) and last in end_turns:
    return end_turns[last] or "none", None
  if b == len(tracks[t]["X"]) and last in merge_into:
    return "merge", None
  hits = sorted((k, v) for k, v in arrows_at.get(t, []) if a <= k < b)
  if hits:
    vals = {v for _, v in hits}
    return hits[-1][1], (f"lane arrows differ along it: {sorted(vals)}" if len(vals) > 1 else None)
  if any(a <= k < b for k in unknown_at.get(t, [])):
    return "?", None
  return None, None


TURN_ORDER = ["reverse", "sharp_left", "left", "slight_left", "through", "slight_right", "right", "sharp_right", "merge_to_left",
              "merge_to_right", "none"]  # within a lane, left to right (as mapped in LA)


def tags_for(S, s, change, widths, ends=None):
  """ends: the lanes' widths at the way's first and last node where they narrow or widen along it, else None."""
  tg, iss = {}, []
  hint = S["hint"]
  if hint:
    cls, sp, name = hint
    link = s in link_of
    tg["highway"] = link_of[s] + "_link" if link else cls
    tg["maxspeed"] = f"{LIMITS[link_of[s] if link and link_of[s] in LIMITS else cls][sp]} mph"
    if name and not link:  # a link turns off the road (ynd_to_osm.py's naming)
      tg["name"] = name
  else:
    tg["highway"] = "road"
    iss.append("no GTA link alongside: class unknown")
  # a lane only a taper's point here (narrower than TAPER_POINT all along) isn't counted yet, or any more
  span = {t: max(w, *(e.get(t, 0.0) for e in ends)) if ends else w for t, w in widths.items()}
  gone = {t for t, w in span.items() if w < TAPER_POINT and any(w2 >= TAPER_POINT for w2 in span.values())}
  order = [(t, d) for t, d in S["order"] if t not in gone]
  fwd = [t for t, d in order if d > 0]
  bwd = [t for t, d in order if d < 0][::-1]  # left to right as their drivers see them
  twltl = "twltl" in S["joints_way"]
  oneway = not bwd
  if oneway:
    tg["oneway"] = "yes"
  tg["lanes"] = str(len(fwd) + len(bwd) + int(twltl))
  # where the way's line is among its lanes (the sections' line above)
  core = [q for q, t in enumerate(fwd) if t not in S["side"]]
  if not oneway and S["two_way"]:
    tg["placement:forward"] = "left_of:1"
  elif oneway and core and len(core) < len(fwd) and core == list(range(core[0], core[-1] + 1)):
    m_ = core[len(core) // 2] + 1
    tg["placement"] = f"middle_of:{m_}" if len(core) % 2 else f"left_of:{m_}"
  if not oneway:
    tg["lanes:forward"], tg["lanes:backward"] = str(len(fwd)), str(len(bwd))
    if twltl:
      tg["lanes:both_ways"], tg["turn:lanes:both_ways"] = "1", "left"
    if any(j == "median" for j in S["joints_way"]):
      tg["divider"] = "solid_line"
  for grp, key in ((fwd, "" if oneway else ":forward"), (bwd, ":backward")):
    if not grp:
      continue
    vals = [turn_value(S, t, 1) for t in grp]
    iss += [v[1] for v in vals if v[1]]
    tv = [v[0] for v in vals]
    for q, v in enumerate(tv):
      if v == "merge":
        into = lane_track.get(merge_into[tracks[grp[q]]["lanes"][-1]], (None,))[0]
        tv[q] = "none" if into not in grp else "merge_to_left" if grp.index(into) < q else "merge_to_right"
    tv = [";".join(sorted(v.split(";"), key=lambda x: TURN_ORDER.index(x) if x in TURN_ORDER else 99)) if v else v for v in tv]
    if all(v != "?" for v in tv) and any(v not in (None, "none") for v in tv):
      tg["turn:lanes" + key] = "|".join(v or "none" for v in tv)
    elif "?" in tv:
      iss.append(f"turn:lanes{key} partly known: " + "|".join(v if v and v != "?" else "?" for v in tv) +
                 (" (arrow painted, type unlisted)" if "?" in tv else ""))
    if ends:
      for e, end in zip(ends, ("start", "end")):
        tg[f"width:lanes{key}:{end}"] = "|".join(f"{e.get(t, 0.0):.1f}" for t in grp)
    else:
      tg["width:lanes" + key] = "|".join(f"{widths.get(t, S['width_lanes'][t]):.1f}" for t in grp)
    for t in grp:
      if np.median(tracks[t]["X"][:, 3]) < NARROW:
        iss.append(f"lane {'+'.join(map(str, tracks[t]['lanes']))} {np.median(tracks[t]['X'][:, 3]):.1f} m wide: narrower than any lane")
  if change:
    cf = [change[i] for i, (t, d) in enumerate(S["order"]) if d > 0 and t not in gone]
    cb = [change[i] for i, (t, d) in enumerate(S["order"]) if d < 0 and t not in gone][::-1]
    for grp, key in ((cf, "" if oneway else ":forward"), (cb, ":backward")):
      if grp and any(v != "yes" for v in grp) and len(grp) > 1:
        tg["change:lanes" + key] = "|".join(grp)
  return tg, iss


# shoulders and kerb-to-kerb width, per section
shoulder_polys = [prep(band(A).buffer(0.3)) for _, A in strips["shoulder"]]
shoulder_w = [A for _, A in strips["shoulder"]]
for s, S in enumerate(sections):
  t0, a, b = S["ref"]
  X, nn = tracks[t0]["X"], tracks[t0]["n"]
  found = {"L": [], "R": []}
  widths, open_side = [], collections.Counter()
  for q, k in enumerate(range(a, b)):
    mem = XS[t0][k][0]
    lo = min(e[2] - e[3] / 2 for e in mem)
    hi = max(e[2] + e[3] / 2 for e in mem)
    for side, o, sg in (("L", lo, -1), ("R", hi, 1)):
      p = X[k, :2] + nn[k] * (o + sg * 0.6)
      hit = [i for i, g in enumerate(shoulder_polys) if g.contains(Point(p)) and abs(np.median(shoulder_w[i][:, 2]) - X[k, 2]) < 2]
      found[side].append(hit[0] if hit else None)
    if q % 3 == 0:
      c = X[k, :2] + nn[k] * (lo + hi) / 2
      ext_ = []
      for side, o, sg in (("L", lo, -1), ("R", hi, 1)):
        reach, end = scan.extent(c, nn[k], X[k, 2], sg)
        cap = abs(o - (lo + hi) / 2) + SHOULDER
        if reach > cap:
          open_side[side] += 1
          reach = None
        ext_.append(reach)
      if None not in ext_:
        widths.append(sum(ext_))
        S.setdefault("width_k", {})[k] = sum(ext_)
  n_ = b - a
  sh = {}
  for side in ("L", "R"):
    hits = [h for h in found[side] if h is not None]
    if len(hits) > n_ / 2:
      i = collections.Counter(hits).most_common(1)[0][0]
      sh[side] = float(np.median(shoulder_w[i][:, 3]))
  if S["flip"]:
    sh = {{"L": "R", "R": "L"}[k]: v for k, v in sh.items()}
    open_side = collections.Counter({{"L": "R", "R": "L"}[k]: v for k, v in open_side.items()})
  S["shoulder"] = sh
  S["width"] = float(np.median(widths)) if len(widths) >= max(1, (n_ // 3) // 2) else None
  for side, v in open_side.items():
    if v > (n_ // 3) / 2:
      S["issues"].append(f"no kerb within {SHOULDER:.0f} m of the {'left' if side == 'L' else 'right'} lane edge (gore or another carriageway): width from the other samples" if S["width"] else
                         f"no kerb within {SHOULDER:.0f} m of the {'left' if side == 'L' else 'right'} lane edge: width unknown")

# --- ways: each section cut where its layer or lane change rules change
ways = []
# an arm's way running on to its junction's node from a fork or merge (a free turn lane leaving or joining just short of
# the mouth) carries on with only the lanes reaching the mouth, as its own way
arm_tracks = {se: {lane_track[i][0] for i, _ in ml} for J in junctions for se, ml in J.get("arm_lanes", {}).items()}
for s, S in enumerate(sections):
  if S.get("drop"):
    continue
  C = S["C"]
  ks = C[:, 4].astype(int)
  Wm = np.array([[S["W"][k].get(t, np.nan) for t, _ in S["order"]] for k in ks])  # (nan: no sample of the lane there)

  def ends_of(a, b):
    """each lane's width at [a, b)'s first and last node: the median of its END_N samples nearest that end (0: none)."""
    out = []
    for rows in (Wm[a:b], Wm[a:b][::-1]):
      out.append(np.array([np.median(c[~np.isnan(c)][:END_N]) if (~np.isnan(c)).any() else 0.0 for c in rows.T]))
    return tuple(out)

  dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C[:, :2], axis=0), axis=1))]

  def straight(a, b):
    """[a, b) cut where its lanes' widths stray WIDTH_STEP from a straight taper between its ends'."""
    w0, w1 = ends_of(a, b)
    f = ((dist[a:b] - dist[a]) / max(dist[b - 1] - dist[a], 1e-9))[:, None]
    dev = np.nan_to_num(np.abs(Wm[a:b] - (w0 + (w1 - w0) * f))).max(1)
    dev[:int(MINLEN)], dev[len(dev) - int(MINLEN):] = 0, 0  # (no piece shorter than MINLEN)
    if dev.max() <= WIDTH_STEP:
      return [(a, b)]
    m = a + int(dev.argmax())
    return straight(a, m) + straight(m, b)
  pieces = [(a_, b_, v) for a, b, v in runs_of([(bool(d), c) for d, c in zip(S["deck"], S["change"])], MINLEN) for a_, b_ in straight(a, b)]
  start_node = port_node.get((s, 0))
  for q, (a, b, (on_deck, change)) in enumerate(pieces):
    last = q == len(pieces) - 1
    P = C[a:b + (0 if last else 1), :3]  # pieces share their cut sample
    nd = []
    for k, p in enumerate(P):
      if k == 0 and start_node is not None:
        nd.append(start_node)
      elif k == len(P) - 1 and last and (s, 1) in port_node:
        nd.append(port_node[(s, 1)])
      else:
        nd.append(new_node(*p))
    stubs = []
    for e, here in ((0, q == 0), (1, last)):
      if here and (s, e) in ext:
        keep = arm_tracks.get((s, e))
        if (s, e) in port_node and keep is not None and {t for t, _ in S["order"]} - keep:
          stubs.append((e, keep))
        elif e:
          nd.append(ext[(s, 1)])
        else:
          nd.insert(0, ext[(s, 0)])
    start_node = nd[-1]
    widths = {t: float(np.median(v)) for t in S["width_lanes"] if (v := [S["W"][k][t] for k in ks[a:b] if t in S["W"][k]])}
    w0, w1 = ends_of(a, b)
    ends = None
    if np.abs(w1 - w0).max() >= TAPER_MIN:
      ends = tuple({t: float(w[i]) for i, (t, _) in enumerate(S["order"])} for w in (w0, w1))
    tg, iss = tags_for(S, s, change, widths, ends)
    if on_deck:
      tg["bridge"], tg["layer"] = "yes", "1"
    kerb = [S.get("width_k", {}).get(k) for k in ks[a:b]]
    kerb_at = [w for w in kerb if w is not None]
    # a tapering way has no one kerb-to-kerb width and OSM has no settled tag for one that changes: its lanes' and
    # shoulders' widths give its cross-section
    if S["width"] and not ends:
      tg["width"] = f"{np.median(kerb_at) if len(kerb_at) >= 2 else S['width']:.1f}"  # (kerb to kerb every third sample)
    sh = S["shoulder"]
    if sh:
      tg["shoulder"] = "both" if len(sh) == 2 else "left" if "L" in sh else "right"
      if len(sh) == 2 and abs(sh["L"] - sh["R"]) > 0.3:
        tg["shoulder:left:width"], tg["shoulder:right:width"] = f"{sh['L']:.1f}", f"{sh['R']:.1f}"
      else:
        tg["shoulder:width"] = f"{np.mean(list(sh.values())):.1f}"
    lanes_src = [tracks[t]["lanes"] for t, _ in S["order"]]
    tg["paintnet:lanes"] = ";".join("+".join(map(str, l)) for l in lanes_src)
    ways.append({"nodes": nd, "tags": tg, "issues": S["issues"] + iss, "section": s})
    for e, keep in stubs:
      S2 = {**S, "order": [(t, d) for t, d in S["order"] if t in keep]}
      change2 = tuple(c for c, (t, _) in zip(change, S["order"]) if t in keep) if change else change
      tg2, iss2 = tags_for(S2, s, change2, {t: float((w1 if e else w0)[i]) for i, (t, _) in enumerate(S["order"]) if t in keep})
      if on_deck:
        tg2["bridge"], tg2["layer"] = "yes", "1"
      tg2["paintnet:lanes"] = ";".join("+".join(map(str, tracks[t]["lanes"])) for t, _ in S2["order"])
      ways.append({"nodes": [port_node[(s, 1)], ext[(s, 1)]] if e else [ext[(s, 0)], port_node[(s, 0)]], "tags": tg2,
                   "issues": S["issues"] + iss2, "section": s, "stub": True})

# --- crossings: a footway along each crossing strip, noded where it crosses a way on its deck
def node_xy(i):
  return np.array(nodes[i][:2])


def cut_way(w, line, z, tags, near=0.3):
  """node way w where it crosses `line` (on its deck), tagging that node, or the node within `near` m of it; the
  node, or None if it doesn't cross."""
  for q in range(len(w["nodes"]) - 1):
    a_, b_ = w["nodes"][q], w["nodes"][q + 1]
    if abs(nodes[a_][2] - z) > 2:
      continue
    hit = LineString([node_xy(a_), node_xy(b_)]).intersection(line)
    if hit.is_empty or hit.geom_type != "Point":
      continue
    p = np.array(hit.coords[0])
    for i in (a_, b_):
      if np.linalg.norm(p - node_xy(i)) < near:
        nodes[i][3].update(tags)
        return i
    zc = nodes[a_][2] + (nodes[b_][2] - nodes[a_][2]) * np.linalg.norm(p - node_xy(a_)) / max(np.linalg.norm(node_xy(b_) - node_xy(a_)), 1e-9)
    cn = new_node(*p, zc, tags)
    w["nodes"].insert(q + 1, cn)
    return cn
  return None


# signals where traffic stops for them: on each arm's way at its stop line (the line across its arriving lanes' ends),
# tagged with the way's direction it applies to (stop_paint.py's tags, as on the live map)
for J in junctions:
  if not J.get("signals"):
    continue
  for s, e, p, u in J["ends"]:
    ends = [LA[i][-1] for i, which, _ in J["mouths"] if which == "end" and i in lane_track and lane_track[i][0] in sections[s]["tracks"]]
    piece = [w for w in ways if w["section"] == s]
    if not ends or not piece:
      continue
    E = np.array(ends)
    nrm = np.array([-u[1], u[0]])
    o = (E[:, :2] - E[:, :2].mean(0)) @ nrm
    mid = E[:, :2].mean(0)
    reach = max(np.abs(o).max() + E[:, 3].max(), sections[s]["C"][:, 3].max()) / 2 + 2
    line = LineString([mid - nrm * reach, mid + nrm * reach])
    tags = {"highway": "traffic_signals", "traffic_signals:direction": "forward" if e else "backward", "source:position": "survey",
            "stop_line:bearing": f"{math.degrees(math.atan2(nrm[0], nrm[1])) % 180:.0f}"}
    # (on the arm's last way, or the one before where the arm ends in a run-on past a fork)
    if all(cut_way(w, line, float(E[:, 2].mean()), tags) is None for w in (piece[::-1] if e else piece)[:3]):
      J["issues"].append(f"stop line of way {s} not on it")

for J in junctions:
  ctl = "traffic_signals" if J.get("signals") else "marked"  # a crossing at a signalled junction has its own lights
  sigs = {"crossing:signals": "yes"} if J.get("signals") else {}
  for A in J["crossings"]:
    foot = LineString(A[:, :2])
    cuts = []
    for w in ways:
      for q in range(len(w["nodes"]) - 1):
        a_, b_ = w["nodes"][q], w["nodes"][q + 1]
        if abs(nodes[a_][2] - np.median(A[:, 2])) > 2:
          continue
        seg = LineString([node_xy(a_), node_xy(b_)])
        hit = seg.intersection(foot)
        if hit.is_empty or hit.geom_type != "Point":
          continue
        p = np.array(hit.coords[0])
        if min(np.linalg.norm(p - node_xy(a_)), np.linalg.norm(p - node_xy(b_))) < 0.3:
          continue
        zc = nodes[a_][2] + (nodes[b_][2] - nodes[a_][2]) * np.linalg.norm(p - node_xy(a_)) / max(np.linalg.norm(node_xy(b_) - node_xy(a_)), 1e-9)
        cn = new_node(*p, zc, {"highway": "crossing", "crossing": ctl, "crossing:markings": "yes", **sigs})
        w["nodes"].insert(q + 1, cn)
        cuts.append((foot.project(Point(p)), cn))
        break
    fn = []
    pts = [(foot.project(Point(p)), None, p) for p in A[:, :3]] + [(d, cn, None) for d, cn in cuts]
    for d, cn, p in sorted(pts, key=lambda v: v[0]):
      fn.append(cn if cn is not None else new_node(*p))
    ways.append({"nodes": fn, "tags": {"highway": "footway", "footway": "crossing", "crossing": ctl, "crossing:markings": "yes", **sigs},
                 "issues": [] if cuts else ["crosses no way"], "section": None})

# turn restrictions at a junction's node (ynd_to_osm.py's, as on the live map): no U-turn back into the arm a two-way
# arm's traffic arrives on, and none of the moves between arms that no lane makes (junctions.py's paths: a free turn
# lane round the box makes it instead, or no arrow turns that way)
TURN_ANG, UTURN = 45.0, 150.0  # deg: junctions.py's
restrictions = []  # (kind, from way, via node, to way)
for J in junctions:
  if "node" not in J:
    continue
  moves = {(p_["from"], p_["to"]) for p_ in J["paths"]}
  arms = []
  for (s, e), ml in J.get("arm_lanes", {}).items():
    w = [w for w in ways if w["section"] == s and J["node"] in (w["nodes"][0], w["nodes"][-1])]
    if w:
      o = np.mean([LA[i][-1 if which == "end" else 0, :2] for i, which in ml], 0) - node_xy(J["node"])
      arms.append((w[0], {i for i, which in ml if which == "end"}, {i for i, which in ml if which == "start"}, math.degrees(math.atan2(o[1], o[0]))))
  for wa, arr, _, oa in arms:
    for wb, _, lea, ob in arms:
      if not arr or not lea or (wa is not wb and any((i, j) in moves for i in arr for j in lea)):
        continue
      th = (ob - oa) % 360 - 180  # the turn from arriving (heading -oa) to leaving (heading ob), left positive
      kind = "no_u_turn" if wa is wb or abs(th) > UTURN else "no_straight_on" if abs(th) <= TURN_ANG else "no_left_turn" if th > 0 else "no_right_turn"
      restrictions.append((kind, wa, J["node"], wb))

# and where sections meet away from a junction (a fork or merge), none of the moves no lane makes (into the other
# direction of a two-way road, from one carriageway back into the other); an arm's run-on stub carries only its own
# section's lanes on to the mouth, so trades no traffic with the other ways there
end_at = collections.defaultdict(list)  # node -> [(way, the node is its last, its section's end there)]
for w in ways:
  s = w["section"]
  if s is None:
    continue
  for last in (False, True):
    n = w["nodes"][-1 if last else 0]
    e = next((e for e in (0, 1) if port_node.get((s, e)) == n), None)
    if e is not None:
      end_at[n].append((w, last, e))
junction_nodes = {J["node"] for J in junctions if "node" in J}
for n, E in end_at.items():
  if n in junction_nodes or len({w["section"] for w, *_ in E}) < 2:
    continue
  for wa, la, ea in E:
    if "oneway" in wa["tags"] and not la:
      continue
    for wb, lb, eb in E:
      if wb["section"] == wa["section"] or ("oneway" in wb["tags"] and lb):
        continue
      if not (wa.get("stub") or wb.get("stub")) and ((wa["section"], ea), (wb["section"], eb)) in made:
        continue
      ha = node_xy(wa["nodes"][-1 if la else 0]) - node_xy(wa["nodes"][-2 if la else 1])
      hb = node_xy(wb["nodes"][-2 if lb else 1]) - node_xy(wb["nodes"][-1 if lb else 0])
      th = math.degrees(math.atan2(ha[0] * hb[1] - ha[1] * hb[0], ha @ hb))
      restrictions.append(("no_u_turn" if abs(th) > UTURN else "no_straight_on" if abs(th) <= TURN_ANG else
                           "no_left_turn" if th > 0 else "no_right_turn", wa, n, wb))

for J in junctions:
  if "node" in J:
    nodes[J["node"]][3]["paintnet:junction"] = f"{len(J['ends'])} ends; GTA's node {J['gta_offset']} m away"
    if J["issues"]:
      nodes[J["node"]][3]["fixme"] = "; ".join(J["issues"])

# --- drop way nodes the line passes within SIMPLIFY m of (in height too), keeping shared and tagged ones
use = collections.Counter(i for w in ways for i in w["nodes"])


def keep_mask(P):
  k = np.zeros(len(P), bool)
  k[[0, -1]] = True
  stack = [(0, len(P) - 1)]
  while stack:
    a, b = stack.pop()
    if b - a < 2:
      continue
    ab = P[b] - P[a]
    L = max(np.linalg.norm(ab[:2]), 1e-9)
    t = np.clip((P[a + 1:b, :2] - P[a, :2]) @ ab[:2] / L ** 2, 0, 1)
    off = np.linalg.norm(P[a + 1:b, :2] - (P[a, :2] + t[:, None] * ab[:2]), axis=1)
    dz = np.abs(P[a + 1:b, 2] - (P[a, 2] + t * ab[2]))
    e = np.maximum(off, dz)
    q = int(e.argmax())
    if e[q] > SIMPLIFY:
      k[a + 1 + q] = True
      stack += [(a, a + 1 + q), (a + 1 + q, b)]
  return k


for w in ways:
  fixed = [q for q, i in enumerate(w["nodes"]) if use[i] > 1 or nodes[i][3] or q in (0, len(w["nodes"]) - 1)]
  out = []
  for a, b in zip(fixed, fixed[1:]):
    P = np.array([nodes[i][:3] for i in w["nodes"][a:b + 1]])
    k = keep_mask(P)
    out += [i for i, kk in zip(w["nodes"][a:b], k[:-1]) if kk]
  out.append(w["nodes"][-1])
  for i in set(w["nodes"]) - set(out):
    if use[i] == 1:
      del nodes[i]
  w["nodes"] = out

# --- write
for nd in nodes.values():
  nd[3]["ele"] = f"{nd[2]:.1f}"
ids = sorted(nodes, reverse=True)
lat, lon = gframe.lat_lon(np.array([nodes[i][0] for i in ids]), np.array([nodes[i][1] for i in ids]))
ll = {i: (float(a), float(o)) for i, a, o in zip(ids, lat, lon)}
wid0 = min(nodes) - 1
with open(out_osm, "w") as f:
  f.write("<?xml version='1.0' encoding='UTF-8'?>\n<osm version='0.6' generator='paintnet osm.py' upload='never'>\n")
  for i in ids:
    tg = nodes[i][3]
    f.write(f"  <node id='{i}' lat='{ll[i][0]:.9f}' lon='{ll[i][1]:.9f}'" + (">\n" if tg else "/>\n"))
    for k, v in tg.items():
      f.write(f"    <tag k={quoteattr(k)} v={quoteattr(str(v))}/>\n")
    if tg:
      f.write("  </node>\n")
  for q, w in enumerate(ways):
    w["id"] = wid0 - q
    f.write(f"  <way id='{w['id']}'>\n")
    for i in w["nodes"]:
      f.write(f"    <nd ref='{i}'/>\n")
    tg = dict(w["tags"])
    if w["issues"]:
      tg["fixme"] = "; ".join(w["issues"])
    for k, v in tg.items():
      f.write(f"    <tag k={quoteattr(k)} v={quoteattr(str(v))}/>\n")
    f.write("  </way>\n")
  for q, (kind, wa, via, wb) in enumerate(restrictions):
    f.write(f"  <relation id='{-1 - q}'>\n    <member type='way' ref='{wa['id']}' role='from'/>\n    <member type='node' ref='{via}' role='via'/>\n"
            f"    <member type='way' ref='{wb['id']}' role='to'/>\n    <tag k='type' v='restriction'/>\n    <tag k='restriction' v='{kind}'/>\n  </relation>\n")
  f.write("</osm>\n")
feats = []
for w in ways:
  coords = [[round(ll[i][1], 8), round(ll[i][0], 8), round(nodes[i][2], 1)] for i in w["nodes"]]
  feats.append({"type": "Feature", "properties": {"osm_type": "way", "id": w["id"], **w["tags"], "issues": "; ".join(w["issues"])},
                "geometry": {"type": "LineString", "coordinates": coords}})
for i in ids:
  tg = {k: v for k, v in nodes[i][3].items() if k != "ele"}
  if tg:
    feats.append({"type": "Feature", "properties": {"osm_type": "node", "id": i, **tg, "issues": tg.get("fixme", "")},
                  "geometry": {"type": "Point", "coordinates": [round(ll[i][1], 8), round(ll[i][0], 8), round(nodes[i][2], 1)]}})
for x, y, why in issues_free:
  la, lo = gframe.lat_lon(np.array([x]), np.array([y]))
  feats.append({"type": "Feature", "properties": {"osm_type": "flag", "issues": why}, "geometry": {"type": "Point", "coordinates": [round(float(lo[0]), 8), round(float(la[0]), 8)]}})
json.dump({"type": "FeatureCollection", "features": feats}, open(out_geo, "w"))

# --- report
print(len(tracks), "tracks;", len(sections), "sections;", len(ways), "ways;", len(nodes), "nodes;", len(restrictions), "restrictions ->", out_osm, out_geo)
for kind, wa, via, wb in restrictions:
  print(f" restriction {kind}: way {wa['id']} -> {wb['id']} at node {via}")
for i, j, wa, wb in implied:
  print(f" carries on without a link: lane {i} -> {j} ({wa} -> {wb} m wide)")
for J in junctions:
  print(" junction of", ", ".join(f"{k} {i}" for k, i in J["strips"]) + ":", f"node {J['node']}, {len(J['ends'])} ends, GTA's node {J['gta_offset']} m off" if "node" in J else "not built", "; ".join(J["issues"]))
for w in ways:
  C = np.array([nodes[i][:2] for i in w["nodes"]])
  L = np.linalg.norm(np.diff(C, axis=0), axis=1).sum()
  print(f" way {w['id']} {L:.0f} m ({C[0, 0]:.0f}, {C[0, 1]:.0f}) -> ({C[-1, 0]:.0f}, {C[-1, 1]:.0f})", " ".join(f"{k}={v}" for k, v in w["tags"].items()), "| " + "; ".join(w["issues"]) if w["issues"] else "")
for x, y, why in issues_free:
  print(" flag", round(x), round(y), why)
