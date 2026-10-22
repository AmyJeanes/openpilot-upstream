"""osm.py x0 x1 y0 y1 groups.json links.json dir.json out.osm out.geojson: the pilot's lanes as OpenStreetMap ways.

A way is a carriageway: the lanes side by side with no kerb between them, one direction or both where only paint
separates them (a centre line, a painted median, a two-way left-turn lane), drawn down the middle of its outer lane
edges. Lanes first become tracks (a lane and the lanes it carries on into, by connect.py's links, the breaks between
them filled in). At every track sample its neighbours either side are found (bands touching within ADJ m, same deck,
parallel); a gore between two lanes parts them, as a kerb does, except a gore whose wide end meets a junction (a
painted island in a junction approach, which only stops lane changes). The cross-section at a sample is the chain
of neighbours; a section is a stretch where it holds the same lanes, and a way is a section cut where its layer or
its lanes' change rules change. Sections that carry on from one another share a node at the end of the one with
most lanes (a fork's or merge's trunk, so at the gore's tip); the others ease into it, a gore's branches over the
gore's length, other ways at TAPER m per m sideways.
Junctions are the crossings and no-traffic strips (stop line to crossing) lying together on a deck; lanes inside
their hull are guide lines through it (dropped), and way ends near it run on to one junction node where the arms'
centre lines best cross. Crossings become footway=crossing ways, noded where they cross a road.
Tags: lanes (+:forward/:backward/:both_ways), turn:lanes from listed arrows only (a lane whose arrow is unknown leaves
the tag off and is flagged), change:lanes from the paint between lanes, width:lanes, width (kerb to kerb, gutter in),
shoulder (+:width), divider=solid_line for a painted median, bridge/layer where the way's centre has another surface
DECK m below. GTA path nodes only hint: highway class, maxspeed, traffic signals. Every node has ele (z).
Anything undecided goes in each way's fixme tag and the geojson's "issues"."""
import json, math, sys, collections
from xml.sax.saxutils import quoteattr
import numpy as np
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
from shapely.prepared import prep
import scan
import arrows

sys.path.insert(0, "/home/amy/git/op-ortho/openpilot/tools/sim/bridge/gta5/map/webmap")
import gframe

x0, x1, y0, y1 = scan.x0, scan.x1, scan.y0, scan.y1
grpj, linkj, dirj, out_osm, out_geo = sys.argv[5:10]

ADJ = 0.5  # m between two lanes' bands that still share a line: measured 0 +- 0.3 at shared lines (connect.py's EDGE)
PARALLEL, DZ = 30.0, 1.5  # deg, m: a neighbour runs alongside on this deck (direction.py's FLOW_ANG, connect.py's deck)
MINLEN, GAP = 6.0, 3.0  # m: lanes.py's shortest line, and its longest break inside one lane
NARROW = 3.5  # m: no lane is narrower (measured 4.1 m up), no shoulder wider than 3.2 (direction.py: none in 3.5-4.5)
SHOULDER = 4.0  # m: direction.py's widest shoulder; a kerb further out than this is not this carriageway's
JCLUST = 10.0  # m between a junction's crossings and stop-line strips (2.0-5.3 m in the interchange's; next 140 m off)
ATTACH = 8.0  # m from a way's end to its junction's strips (0-1.8 m on the interchange; the nearest other ends 24 m)
# m along per m sideways a way's centre moves into a shared node: lanes open and close at 0.20-0.46 m of width per m
# on the interchange (motorway 0.2, city bays 0.3-0.46; median 0.34), moving the carriageway's centre half that
TAPER = 6.0
ARM = 10.0  # m: a junction arm's heading is its last ARM m (connect.py's frame looks 10 m along)
DECK, DECK_CLOSE = 2.5, 15  # m: a surface this far below is a road under a deck (export.py's covered), gaps closed
ARROW_ALIGN, ARROW_DZ, ARROW_SPILL = 30.0, 2.5, 0.5  # deg, m, m outside a lane: an arrow is in it (ynd_to_osm.py's)
EDGE_BOX = 6.0  # m: ends this near the box's edge are cut by it (lanes.py samples lines 5 m past it)
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

# --- junctions: crossings and stop-line strips lying together on a deck
js = [(kind, k, A) for kind in ("crossing", "not_traffic") for k, A in strips[kind]]
par = list(range(len(js)))


def root(i):
  while par[i] != i:
    i = par[i]
  return i


for a in range(len(js)):
  for b in range(a + 1, len(js)):
    A, B = js[a][2], js[b][2]
    if abs(np.median(A[:, 2]) - np.median(B[:, 2])) > 2:
      continue
    if np.linalg.norm(A[:, None, :2] - B[None, :, :2], axis=2).min() <= JCLUST:
      par[root(b)] = root(a)
junctions = []
for r in sorted({root(i) for i in range(len(js))}):
  mem = [js[i] for i in range(len(js)) if root(i) == r]
  hull = unary_union([band(A) for _, _, A in mem]).convex_hull
  junctions.append({"hull": hull, "z": float(np.median(np.vstack([A for _, _, A in mem])[:, 2])), "strips": [(kd, k) for kd, k, _ in mem],
                    "crossings": [A for kd, _, A in mem if kd == "crossing"], "ends": [], "issues": []})

# --- lanes kept: known direction, MINLEN long, not painted through a junction
keep = []
for i, A in enumerate(LA):
  if lanes[i]["dir"] == "?":
    issues_free.append((*A[len(A) // 2, :2], f"lane {i}: no direction"))
    continue
  if len(A) < MINLEN:
    continue  # stubs between a stop line and a crossing
  inside = [J for J in junctions if abs(J["z"] - np.median(A[:, 2])) < 2 and
            np.mean([J["hull"].contains(Point(p)) for p in A[:, :2]]) >= 0.5]
  if inside:
    inside[0]["issues"].append(f"lane {i} painted through the junction: not a way")
    continue
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
gores = [(prep(band(A).buffer(0.3)), atj, A) for (k, A), atj in zip(strips["gore"], gore_at_j)]
centres = [(prep(band(A).buffer(0.3)), kind, A) for kind in ("median", "twltl") for k, A in strips[kind]]
live = [t for t in range(len(tracks)) if t not in narrow_tracks]
S_xy = np.vstack([tracks[t]["X"][:, :2] for t in live])
S_id = np.concatenate([np.stack([np.full(len(tracks[t]["X"]), t), np.arange(len(tracks[t]["X"]))], 1) for t in live])
tree = cKDTree(S_xy)


def between(p, z):
  """what lies at p between two bands on this deck: ("gore", at a junction?), ("median"|"twltl",) or None."""
  pt = Point(p)
  for g, atj, A in gores:
    if abs(np.median(A[:, 2]) - z) < 2 and g.contains(pt):
      return ("gore", atj)
  for g, kind, A in centres:
    if abs(np.median(A[:, 2]) - z) < 2 and g.contains(pt):
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
  C = []
  for k in range(a, b):
    m = XS[t][k][0]
    lo = min(e[2] - e[3] / 2 for e in m)
    hi = max(e[2] + e[3] / 2 for e in m)
    C.append([*(T["X"][k, :2] + T["n"][k] * (lo + hi) / 2), T["X"][k, 2], hi - lo, k])
  C = np.array(C)
  C[:, :2] = ndi.uniform_filter1d(C[:, :2], SMOOTH, axis=0, mode="nearest")
  if flip:
    C = C[::-1]
  sections.append({"ref": (t, a, b), "flip": flip, "runs": members, "tracks": {runs[i][0] for i in members}, "C": C,
                   "mem": mem, "joints": joints, "issues": []})
sec_of_run = {i: s for s, S in enumerate(sections) for i in S["runs"]}

# --- section ends that meet: a track carrying on from one section into the next, or a link between tracks
port = {}
pparent = {}


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
for i, j in other_links + branch_links:
  ti, a_, b_ = lane_track[i]
  tj, aj, _ = lane_track[j]
  if ti not in XS or tj not in XS:
    continue
  X = tracks[ti]["X"]
  ki = int(np.linalg.norm(X[:, :2] - tracks[tj]["X"][aj, :2], axis=1).argmin()) if (i, j) in branch_links else b_
  si, sj = sec_of_run[run_of[ti, ki]], sec_of_run[run_of[tj, aj]]
  if si != sj:
    meet(end_near(si, tracks[tj]["X"][aj, :2]), end_near(sj, tracks[tj]["X"][aj, :2]))
clusters = collections.defaultdict(list)
for s in range(len(sections)):
  for e in (0, 1):
    clusters[pfind((s, e))].append((s, e))

# --- lanes of each section in way order, and their tags
arrow_list = [a for a in scan.src.features() if a["kind"] == "arrow" and scan.inbox(a["x"], a["y"], 10)]
code_, zz_ = arrows.load(x0 - 20, x1 + 20, y0 - 20, y1 + 20, 0.1)
shapes = arrows.find((code_ >> 3) > 0, zz_, x0 - 20, y1 + 20, 0.1)[1]
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
unknown_at = collections.defaultdict(list)
for x, y, z, _ in shapes:
  hit = in_lane(x, y, z)
  if hit:
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


port_node, port_reach = {}, {}
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
  for p in ports:
    port_node[p] = nd
    port_reach[p] = float(max(tips)) if tips and p not in top else None


def blend(C, e, q, share, reach=None):
  """move a way's end e (0 start, 1 end) onto q, easing in over `reach` m, or TAPER m per m it moves sideways within
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
  L = min(reach, s.max()) if reach else min(TAPER * dist, s.max() * share if s.max() > 0 else 1)
  w = np.clip(1 - s / max(L, 1e-6), 0, 1)
  C[:, :2] += w[:, None] * delta
  return C


for s, S in enumerate(sections):
  moves = [np.linalg.norm(np.array(nodes[port_node[(s, e)]][:2]) - S["C"][-1 if e else 0, :2]) if (s, e) in port_node else 0 for e in (0, 1)]
  for e in (0, 1):
    if (s, e) in port_node:
      q = np.array(nodes[port_node[(s, e)]][:3])
      S["C"] = blend(S["C"], e, q, 0.5 if moves[1 - e] > ADJ else 1.0, port_reach[(s, e)])  # the whole way when its other end stays

# --- junctions: dead ends near a junction's strips run on to one node where the arms' lines best cross
gta_nodes = [n for n in scan.nodes.values() if scan.inbox(n["x"], n["y"], 20)]
dead = [(s, e) for s in range(len(sections)) for e in (0, 1) if (s, e) not in port_node]
ext = {}  # (s, e) -> junction node
for s, e in dead:
  C = sections[s]["C"]
  p = C[-1 if e else 0]
  s_ = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C[:, :2], axis=0), axis=1))]
  back = C[np.searchsorted(s_, s_[-1] - ARM)] if e else C[min(np.searchsorted(s_, ARM), len(C) - 1)]
  out_dir = unit(p[:2] - back[:2])
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
  for s, e, p, u in J["ends"]:
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
  tags = {"highway": "traffic_signals"} if sig else {}
  J["node"] = new_node(*xy, np.mean([p[2] for _, _, p, _ in J["ends"]]), tags)
  J["gta_offset"] = round(math.hypot(gn["x"] - xy[0], gn["y"] - xy[1]), 1)
  for s, e, p, u in J["ends"]:
    ext[(s, e)] = J["node"]

# --- highway class and speed hints from the GTA links along each section
raw = {}
for (ka, kb), l in scan.links.items():
  raw[(ka, kb)] = l


def gta_hint(C):
  fw, lanes_, sp = [], [], []
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
  if not fw:
    return None
  cls = "motorway" if np.mean(fw) >= 0.5 else "primary" if max(lanes_) >= 2 else "residential"
  return cls, collections.Counter(sp).most_common(1)[0][0]


for s, S in enumerate(sections):
  S["hint"] = gta_hint(S["C"])
# a one-lane branch of a motorway fork or merge, beside a branch of two lanes or more, is a link; so on through 1:1 joins
link_secs = set()  # (GTA doesn't mark ramps: ynd_to_osm.py's ramps())
for ports in clusters.values():
  if len(ports) >= 3:
    nl = {s: len(sections[s]["order"]) for s, _ in ports}
    trunk = max(nl, key=nl.get)
    if sections[trunk]["hint"] and sections[trunk]["hint"][0] == "motorway" and min(nl.values()) == 1 and max(nl.values()) >= 2:
      link_secs |= {s for s, n_ in nl.items() if n_ == 1}
grow = True
while grow:
  grow = False
  for ports in clusters.values():
    if len(ports) == 2 and len({s for s, _ in ports} & link_secs) == 1:
      link_secs |= {s for s, _ in ports}
      grow = True


def turn_value(S, t, sign):
  """the listed arrow in track t's stretch of this section (the last along travel if several), "?" if an arrow of
  unknown type is painted there, None if none."""
  a, b = S["rng"][t]
  hits = sorted((k, v) for k, v in arrows_at.get(t, []) if a <= k < b)
  if hits:
    vals = {v for _, v in hits}
    return hits[-1][1], (f"lane arrows differ along it: {sorted(vals)}" if len(vals) > 1 else None)
  if any(a <= k < b for k in unknown_at.get(t, [])):
    return "?", None
  return None, None


def tags_for(S, s, change):
  tg, iss = {}, []
  hint = S["hint"]
  if hint:
    cls, sp = hint
    tg["highway"] = "motorway_link" if s in link_secs else cls
    tg["maxspeed"] = f"{LIMITS['motorway' if s in link_secs else cls][sp]} mph"
  else:
    tg["highway"] = "road"
    iss.append("no GTA link alongside: class unknown")
  fwd = [t for t, d in S["order"] if d > 0]
  bwd = [t for t, d in S["order"] if d < 0][::-1]  # left to right as their drivers see them
  twltl = "twltl" in S["joints_way"]
  oneway = not bwd
  if oneway:
    tg["oneway"] = "yes"
  tg["lanes"] = str(len(fwd) + len(bwd) + int(twltl))
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
    if all(v and v != "?" for v in tv):
      tg["turn:lanes" + key] = "|".join(tv)
    elif any(tv):
      iss.append(f"turn:lanes{key} partly known: " + "|".join(v if v and v != "?" else "?" for v in tv) +
                 (" (arrow painted, type unlisted)" if "?" in tv else ""))
    tg["width:lanes" + key] = "|".join(f"{S['width_lanes'][t]:.1f}" for t in grp)
    for t in grp:
      if np.median(tracks[t]["X"][:, 3]) < NARROW:
        iss.append(f"lane {'+'.join(map(str, tracks[t]['lanes']))} {np.median(tracks[t]['X'][:, 3]):.1f} m wide: narrower than any lane")
  if change:
    cf = [change[i] for i, (t, d) in enumerate(S["order"]) if d > 0]
    cb = [change[i] for i, (t, d) in enumerate(S["order"]) if d < 0][::-1]
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
for s, S in enumerate(sections):
  C = S["C"]
  key = [(bool(d), c) for d, c in zip(S["deck"], S["change"])]
  pieces = runs_of(key, MINLEN)
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
    if q == 0 and (s, 0) in ext:
      nd.insert(0, ext[(s, 0)])
    if last and (s, 1) in ext:
      nd.append(ext[(s, 1)])
    start_node = nd[-1]
    tg, iss = tags_for(S, s, change)
    if on_deck:
      tg["bridge"], tg["layer"] = "yes", "1"
    if S["width"]:
      tg["width"] = f"{S['width']:.1f}"
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

# --- crossings: a footway along each crossing strip, noded where it crosses a way on its deck
def node_xy(i):
  return np.array(nodes[i][:2])


for J in junctions:
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
        cn = new_node(*p, zc, {"highway": "crossing", "crossing": "marked"})
        w["nodes"].insert(q + 1, cn)
        cuts.append((foot.project(Point(p)), cn))
        break
    fn = []
    pts = [(foot.project(Point(p)), None, p) for p in A[:, :3]] + [(d, cn, None) for d, cn in cuts]
    for d, cn, p in sorted(pts, key=lambda v: v[0]):
      fn.append(cn if cn is not None else new_node(*p))
    ways.append({"nodes": fn, "tags": {"highway": "footway", "footway": "crossing", "crossing": "marked"}, "issues": [] if cuts else ["crosses no way"], "section": None})

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
print(len(tracks), "tracks;", len(sections), "sections;", len(ways), "ways;", len(nodes), "nodes ->", out_osm, out_geo)
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
