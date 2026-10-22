"""lanes.py x0 x1 y0 y1 out.json [--visible=visible.json]: lanes as the space between neighbouring lane lines (or a line and the road's edge).

Every lengthwise paint line (MINLEN m or longer, so hatching and arrows drop out) is sampled every STEP m; square to it
at each sample, the nearest boundary either side on its deck is found: another lengthwise line (a double line's two
halves count as one), or the road surface's edge (kerb, pavement, gutter or nothing) where that comes first. Neighbours
LANE_MIN to LANE_MAX m apart bound a lane. Lane samples are keyed by their two boundaries (line ids; "e" for the edge)
and chained along: a lane is one key's run, split at gaps over GAP m.
Out: [{"left": id, "right": id, "left_all"/"right_all": every boundary id along it (lanes joined across a break),
"pts": [[x, y, z, width], ...]}] with left/right as seen along pts."""
import json, math, multiprocessing, os, sys, collections
import numpy as np
from scipy import ndimage as ndi
import scan
import arrows

STEP, MINLEN, LANE_MIN, LANE_MAX, GAP = 1.0, 6.0, 1.5, 9.0, 3.0
PIECE_GAP = 0.5  # m between a line piece's end and the next piece's start
GUTTER_SHARE = 0.5  # of a lane's kerb-side samples with a gutter for the rest to take one
SHOULDER_MAX = 4.0  # m: line to road's edge, a shoulder below it, a lane from it (direction.py's SHOULDER)
# twin lines: nearest-other-line distances on the interchange peak at 0-0.3 m, dip at 0.3-0.4, lanes from ~4 m; and they
# must stay that close along most of where they see each other (lines meeting at a gore's nose only touch at the tip)
TWIN, TWIN_SHARE = 0.35, 0.8
x0, x1, y0, y1 = scan.x0, scan.x1, scan.y0, scan.y1
lines = {}
for r in scan.src.paint():
  P = np.array(r["pts"], float)
  if r["len"] < MINLEN or not ((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)).any():
    continue
  lines[r["id"]] = r
# lines the game files hold but the game never draws (visible.py, against the captured imagery) aren't lines; a pale
# deck's panel seams and gutter edges (visible.py's "edge") aren't paint either but are where its lanes divide
vis = next((a[10:] for a in sys.argv if a.startswith("--visible=")), None)
hidden, edge, V = set(), set(), {}
if vis:
  V = json.load(open(vis))
  edge = {int(k) for k, v in V.items() if v.get("edge")}
  hidden = {int(k) for k, v in V.items() if v["drawn"] is False} - edge
  lines = {k: v for k, v in lines.items() if k not in hidden}
  print(len(hidden), "lines not drawn in the game dropped")
# a piece shorter than MINLEN carrying on the end of a line (lines are stored in pieces: a 5.9 m piece bridges two
# motorway edge lines at a merge) is part of that line
ends = [(np.array(r["pts"][0][:2]), np.array(r["pts"][-1][:2])) for r in lines.values()]
for r in scan.src.paint():
  if r["len"] >= MINLEN or r["len"] < 1.0 or r["id"] in lines or r["id"] in hidden | edge:
    continue
  P = np.array(r["pts"], float)
  if not ((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)).any():
    continue
  if any(min(np.linalg.norm(P[k, :2] - e) for e in pair) < PIECE_GAP for pair in ends for k in (0, -1)):
    lines[r["id"]] = r
# a dashed line stored dash by dash (the motorway west of the interchange: 3.9 m dashes 8 m apart): DASH_MIN or more
# short pieces of one colour in line with each other, each within DASH_GAP m of the next, are one dashed line (named
# for its lowest piece id; the others are found as it), and two or more carrying on in line past a dashed line's end are part
# of it (the ramps west of the interchange: their long dashed lines stop 8-30 m short of the dashes). Arrows and text
# don't line up three times over
DASH_SAME = 0.1  # share a dash's length may differ from the others'
DASH_GAP, DASH_LAT, DASH_BEND, DASH_ANG, DASH_MIN = 12.0, 0.3, 0.1, 20.0, 3  # (DASH_BEND: m more sideways per m apart, round a ramp's curve)
short = []
for r in scan.src.paint():
  if r["len"] >= MINLEN or r["len"] < 1.0 or r["id"] in lines or r["id"] in hidden | edge:
    continue
  P = np.array(r["pts"], float)
  if ((P[:, 0] >= x0 - 20) & (P[:, 0] <= x1 + 20) & (P[:, 1] >= y0 - 20) & (P[:, 1] <= y1 + 20)).any():
    d = P[-1, :2] - P[0, :2]
    short.append((r, P, d / max(np.linalg.norm(d), 1e-9)))
nshort = len(short)
# (a dashed line's own ends, as pieces: dashes carrying on in line past where it stops are part of it)
for r in list(lines.values()):
  if r["style"] != "dashed" or r["id"] in edge:  # (an edge's pieces carry it on below)
    continue
  P = np.array(r["pts"], float)
  for E in (P[:2], P[-2:]):
    d = E[-1, :2] - E[0, :2]
    if np.linalg.norm(d) > 1e-6:
      short.append((r, E, d / np.linalg.norm(d)))
dpar = list(range(len(short)))


def droot(i):
  while dpar[i] != i:
    i = dpar[i]
  return i


for a in range(len(short)):
  ra, Pa, da = short[a]
  for b in range(a + 1, len(short)):
    rb, Pb, db = short[b]
    if rb["colour"] != ra["colour"] or abs(da @ db) < math.cos(math.radians(DASH_ANG)):
      continue
    g = min((Pb[j, :2] - Pa[i, :2] for i in (0, -1) for j in (0, -1)), key=np.linalg.norm)
    off = (Pb[:, :2].mean(0) - Pa[:, :2].mean(0)) @ np.array([da[1], -da[0]])
    if np.linalg.norm(g) <= DASH_GAP and abs(off) <= DASH_LAT + DASH_BEND * np.linalg.norm(Pb[:, :2].mean(0) - Pa[:, :2].mean(0)):
      dpar[droot(b)] = droot(a)
dash_of = {}  # piece id -> the dashed line it is part of
groups_ = collections.defaultdict(list)
for i in range(len(short)):
  groups_[droot(i)].append(i)
for mem in groups_.values():
  # a dashed line's dashes are one length (the motorway's 3.86-3.97 m, a ramp's 1.94-1.97): a piece of another length
  # in line with them is other paint (a solid guide line round an island, a crossing's piece)
  dl = [short[i][0]["len"] for i in mem if i < nshort]
  if dl:
    L0 = float(np.median(dl))
    mem = [i for i in mem if i >= nshort or abs(short[i][0]["len"] - L0) <= DASH_SAME * L0]
  if sum(i < nshort for i in mem) < 2:
    continue
  longs = {short[i][0]["id"] for i in mem if i >= nshort}
  if longs:  # dashes carrying on a dashed line (two lines bridged by dashes: the lower id's): it runs on through them
    lid_ = min(longs)
    L = np.array(lines[lid_]["pts"], float)
    ends_ = {0: [], -1: []}
    for i in mem:
      if i < nshort:
        dash_of[short[i][0]["id"]] = lid_
        c = short[i][1][:, :2].mean(0)
        ends_[min((0, -1), key=lambda k: np.linalg.norm(c - L[k, :2]))].append(i)
    for k, mem_ in ends_.items():
      mem_.sort(key=lambda i: np.linalg.norm(short[i][1][:, :2].mean(0) - L[k, :2]))
      for i in mem_:
        P = short[i][1]
        P = P if np.linalg.norm(P[0, :2] - L[k, :2]) <= np.linalg.norm(P[-1, :2] - L[k, :2]) else P[::-1]  # its near end first
        L = np.vstack([L, P]) if k == -1 else np.vstack([P[::-1], L])
    lines[lid_] = {**lines[lid_], "pts": L.tolist(), "len": float(np.linalg.norm(np.diff(L[:, :2], axis=0), axis=1).sum())}
    continue
  if len(mem) < DASH_MIN:
    continue
  d = short[mem[0]][2]
  mem.sort(key=lambda i: short[i][1][:, :2].mean(0) @ d)
  pts = np.vstack([short[i][1] if (short[i][1][-1, :2] - short[i][1][0, :2]) @ d >= 0 else short[i][1][::-1] for i in mem])
  rid = min(short[i][0]["id"] for i in mem)
  lines[rid] = {**short[mem[0]][0], "id": rid, "pts": pts.tolist(), "style": "dashed",
                "len": float(np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1).sum())}
  for i in mem:
    dash_of[short[i][0]["id"]] = rid
print(len(dash_of), "dashes joined into", len(set(dash_of.values())), "dashed lines (or carrying one on)")
# the surface raster holds a deck's panel seam in pieces (0.6-2 m every other panel joint, 12-13 m apart): past an edge
# line's end, the piece nearest the line carried straight on (from EDGE_BACK m back), up to EDGE_GAP m ahead and within
# EDGE_LAT m of it sideways, carries it on, across the gap too
EDGE_GAP, EDGE_LAT, EDGE_DZ, EDGE_BACK = 13.0, 0.5, 1.0, 10.0
pool = {}
for r in scan.src.paint():
  P = np.array(r["pts"], float)
  if (r["len"] < MINLEN and r["colour"] == "white" and r.get("src") == "surface" and r["id"] not in lines
      and r["id"] not in dash_of and not V.get(str(r["id"]), {}).get("drawn")
      and ((P[:, 0] >= x0 - 20) & (P[:, 0] <= x1 + 20) & (P[:, 1] >= y0 - 20) & (P[:, 1] <= y1 + 20)).any()):
    pool[r["id"]] = P
carried = 0
for lid in sorted(edge & set(lines)):
  L = np.array(lines[lid]["pts"], float)
  for k in (0, -1):
    while True:
      Lk = L if k == -1 else L[::-1]
      d = np.cumsum(np.linalg.norm(np.diff(Lk[::-1, :2], axis=0), axis=1))
      back = Lk[::-1][1:][min(np.searchsorted(d, EDGE_BACK), len(d) - 1)]
      e = Lk[-1, :2] - back[:2]
      run = max(np.linalg.norm(e), 1e-9)
      e /= run
      best = None
      for pid, P in pool.items():
        near0 = np.linalg.norm(P[0, :2] - Lk[-1, :2]) <= np.linalg.norm(P[-1, :2] - Lk[-1, :2])
        q = P[0] if near0 else P[-1]
        v = q[:2] - Lk[-1, :2]
        a, lat = v @ e, abs(v[0] * e[1] - v[1] * e[0])
        zq = Lk[-1, 2] + (Lk[-1, 2] - back[2]) / run * a
        if 0 < a <= EDGE_GAP and lat <= EDGE_LAT and abs(q[2] - zq) <= EDGE_DZ and (best is None or lat < best[0]):
          best = (lat, pid, P if near0 else P[::-1])
      if best is None:
        break
      scan.bridge(Lk[-1, :3], best[2][0, :3], best[1], dashed=lines[lid]["style"] == "dashed")  # (found as the line, its piece's id)
      L = np.vstack([L, best[2]]) if k == -1 else np.vstack([best[2][::-1], L])
      dash_of[best[1]] = lid
      del pool[best[1]]
      carried += 1
  lines[lid] = {**lines[lid], "pts": L.tolist(), "len": float(np.linalg.norm(np.diff(L[:, :2], axis=0), axis=1).sum())}
print(len(edge & set(lines)), "edge lines (panel seams, gutter edges),", carried, "pieces carrying them on")
keep = set(lines) | set(dash_of)


def resample(P, step):
  d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
  s = np.arange(step / 2, d[-1], step)
  return np.stack([np.interp(s, d, P[:, k]) for k in range(3)], 1), s


# two edge lines closer than a lane runs (a deck's gutter edge and a stripe on its panels a metre in) are one edge: the
# strip between them is no lane, so the lane beside them runs to the outer one. The inner one, with another edge line
# under LANE_MIN m to one side and a lane's width of road to the other over most of its length, is dropped
inner = set()
for lid in sorted(edge & set(lines)):
  Q, _ = resample(np.array(lines[lid]["pts"], float), STEP)
  hit = 0
  for k in range(1, len(Q) - 1):
    e = Q[k + 1, :2] - Q[k - 1, :2]
    e /= max(np.linalg.norm(e), 1e-9)
    n = np.array([e[1], -e[0]])
    xs = [(o, dash_of.get(i, i)) for o, i, *_ in scan.crossings(Q[k, :2], n, Q[k, 2], -LANE_MAX, LANE_MAX)]
    xs = [(o, i) for o, i in xs if i in keep and i != lid and abs(o) > TWIN]
    for side in (1, -1):
      near_ = [i for o, i in xs if 0 < side * o < LANE_MIN]
      far_ = [side * o for o, i in xs if side * o < 0]
      if near_ and near_[0] in edge and far_ and min(-np.array(far_)) >= LANE_MIN:
        hit += 1
        break
  if hit > 0.5 * max(len(Q) - 2, 1):
    inner.add(lid)
for lid in inner:
  del lines[lid]
  dash_of = {k: v for k, v in dash_of.items() if v != lid}
keep = set(lines) | set(dash_of)
print(len(inner), "edge lines inside another (one edge):", sorted(inner))


# pass 1: what each line sample sees square to it; lines running within TWIN m of each other are one marking (a double
# yellow, solid beside dashed), whether or not the extraction paired them
seen = []  # (lid, s, c, n, z, crossings)
near_any = []
parent = dict(dash_of)  # (a dash is found as the dashed line it is part of)
pair_seen, pair_close = collections.Counter(), collections.Counter()


def find(i):
  while parent.get(i, i) != i:
    i = parent[i]
  return i


for lid, r in lines.items():
  P = np.array(r["pts"], float)
  Q, S = resample(P, STEP)
  for k, (q, s) in enumerate(zip(Q, S)):
    if not (x0 - 5 <= q[0] <= x1 + 5 and y0 - 5 <= q[1] <= y1 + 5):
      continue
    k0, k1 = max(0, k - 1), min(len(Q) - 1, k + 1)
    e = Q[k1, :2] - Q[k0, :2]
    if np.linalg.norm(e) < 1e-6:
      continue
    e /= np.linalg.norm(e)
    n = np.array([e[1], -e[0]])
    c, z = q[:2], q[2]
    xs = [x for x in scan.crossings(c, n, z, -LANE_MAX - 0.5, LANE_MAX + 0.5) if x[1] in keep and x[1] != lid]
    seen.append((lid, s, c, n, z, xs))
    if xs:
      o = min(abs(x[0]) for x in xs)
      near_any.append(o)
    for x in xs:
      pair_seen[lid, x[1]] += 1
      if abs(x[0]) <= TWIN:
        pair_close[lid, x[1]] += 1
for (a, b), nc in pair_close.items():
  if nc >= 5 and nc >= TWIN_SHARE * pair_seen[a, b]:
    ra, rb = find(a), find(b)
    if ra != rb:
      parent[max(ra, rb)] = min(ra, rb)
# yellow lines stored in pieces: a piece carrying on from another's end (within PIECE_GAP, heading on the same way:
# not two lines leaving a point together, like a gore's sides at its tip) is the same line, so a median bounded by
# it is one (white lines' pieces are joined where lanes are: across breaks, and by connect.py's links)
ends_dir = []
for lid, r in lines.items():
  P = np.array(r["pts"], float)[:, :2]
  if len(P) < 2 or r["colour"] != "yellow":
    continue
  for p, q in ((P[0], P[1]), (P[-1], P[-2])):
    d = p - q
    ends_dir.append((lid, p, d / max(np.linalg.norm(d), 1e-9)))  # pointing out of the line at that end
for a in range(len(ends_dir)):
  for b in range(a + 1, len(ends_dir)):
    la, pa, da = ends_dir[a]
    lb, pb, db = ends_dir[b]
    if la != lb and np.linalg.norm(pa - pb) < PIECE_GAP and da @ db < -math.cos(math.radians(30)):
      ra, rb = find(la), find(lb)
      if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)
h, bins = np.histogram(near_any, bins=[0, .1, .2, .3, .4, .5, .6, .8, 1, 1.5, 2, 3, 4, 10])
print("nearest other line:", " ".join(f"{bins[i]:.1f}:{h[i]}" for i in range(len(h))))

# solid lines the line extraction missed (a gore's outline drawn as part of its chevron marking) are still painted in
# the surface raster: paint at the same offset all along +-RL_ALONG m, RL_MIN-RL_MAX m wide, painted over RL_SOLID of
# its length; chevrons move sideways along, arrows and text are masked out by the markings list and arrows the list
# misses by their shape (an arrow's shaft is a short solid lengthwise stroke)
RL_ALONG, RL_MIN, RL_MAX, RL_SOLID = 1.5, 0.08, 0.8, 0.9
marks = scan.src.features()
marks = [[f["x"], f["y"], f["z"], max(f.get("len", 0), f.get("wid", 0)) / 2 + 0.5] for f in marks
         if f["kind"] != "crossing" and scan.inbox(f["x"], f["y"], 20)]
code_, zz_ = arrows.load(x0 - 20, x1 + 20, y0 - 20, y1 + 20, 0.1)
zk_ = np.where(np.isnan(zz_), np.inf, zz_)
shape_, found = arrows.find((code_ >> 3) > 0, zz_, x0 - 20, y1 + 20, 0.1)


def arrow_at(X, Y, z):
  """per point: on an arrow found by its shape (its own paint grown 0.2 m), on the deck nearest z."""
  c = np.clip(((X - (x0 - 20)) / 0.1).astype(int), 0, shape_.shape[2] - 1)
  r = np.clip((((y1 + 20) - Y) / 0.1).astype(int), 0, shape_.shape[1] - 1)
  dz = np.abs(zk_[:, r, c] - z)
  l = dz.argmin(0)
  return shape_[l, r, c] & (dz[l, np.arange(len(l))] < 2)


marks = np.array(marks + [[x, y, z, r + 0.5] for x, y, z, r in found])
print(len(found), "arrows found by shape")
raster_found = collections.Counter()


def raster_line(c, n, z, side, lim):
  """offset (> 0, on `side`) of the nearest raster-only solid lengthwise line within lim m, or None."""
  if lim <= 0.4:
    return None
  t = np.arange(0.4, lim, 0.05)
  a = np.arange(-RL_ALONG, RL_ALONG + 1e-6, 0.1)
  T, A = np.meshgrid(t, a)
  e = np.array([-n[1], n[0]])
  X, Y = c[0] + side * T * n[0] + A * e[0], c[1] + side * T * n[1] + A * e[1]
  code, h = scan.T.sample(X.ravel(), Y.ravel(), z, zwin=1.0)
  paint = ((code >> 3) > 0).reshape(T.shape)
  if len(marks):
    near = marks[(np.abs(marks[:, 2] - z) < 2) & (np.linalg.norm(marks[:, :2] - c, axis=1) < lim + RL_ALONG + 5)]
    for mx, my, mz, r in near:
      paint &= np.hypot(X - mx, Y - my) > r
  col = paint.mean(0) >= RL_SOLID
  k = 0
  while k < len(t):
    if col[k]:
      j = k
      while j + 1 < len(t) and col[j + 1]:
        j += 1
      if RL_MIN <= (j - k + 1) * 0.05 <= RL_MAX:
        return float((t[k] + t[j]) / 2)
      k = j + 1
    else:
      k += 1
  return None


# pass 2: lanes between neighbouring markings (or a marking and the road's edge)
def look(item):
  """the lane samples beside one line sample: [(key, sample, raster line used: before the edge)]."""
  lid, s, c, n, z, xs = item
  out = []
  gutter = math.nan  # m of gutter between a lane and the kerb (0: none; nan: not a lane to the kerb)
  me = find(lid)
  # this line's own other half, if it is one half of a double line: a lane on that side starts beyond it
  twin_at = {side: max([side * x[0] for x in xs if find(x[1]) == me and 0 < side * x[0] <= TWIN], default=0.0)
             for side in (1, -1)}
  xs = [x for x in xs if find(x[1]) != me]
  for side in (1, -1):
    gutter = math.nan
    cand = sorted([x for x in xs if side * x[0] > 0], key=lambda x: side * x[0])
    edge, _ = scan.extent(c + n * side * 0.3, n, z, side)  # step off the line's own paint first
    edge += 0.3
    if cand and side * cand[0][0] < edge:
      o, other = side * cand[0][0], find(cand[0][1])
    else:
      # to the kerb a shoulder, gutter and all; a lane (as wide as a shoulder can't be, direction.py's SHOULDER)
      # stops at the gutter
      o, other = edge, "e"
      if edge >= SHOULDER_MAX:
        road, _ = scan.extent(c + n * side * 0.3, n, z, side, scan.ROAD)
        o = max(road + 0.3, SHOULDER_MAX) if road + 0.3 < edge else edge
        gutter = edge - o
    r = raster_line(c, n, z, side, o - TWIN)
    raster = None
    if r is not None:
      raster = other == "e"
      o, other = r, "r"
    # two yellow lines further apart than a double line's halves bound a painted median however narrow it is, and a
    # line and the road's edge a shoulder (driveable, not a lane) down to where they touch
    yellow_pair = lines[lid]["colour"] == "yellow" and other not in ("e", "r") and cand and cand[0][2] == "yellow"
    if not (LANE_MIN <= o <= LANE_MAX or (yellow_pair or other == "e") and TWIN < o < LANE_MIN):
      out.append((None, None, raster))
      continue
    o0 = twin_at[side]
    if o - o0 < TWIN:
      out.append((None, None, raster))
      continue
    m = c + n * side * (o0 + o) / 2
    o -= o0
    # key as seen travelling along this line's point order: left boundary first
    key = (me, other) if side == 1 else (other, me)
    out.append((key, (float(m[0]), float(m[1]), float(z), float(o), lid, float(s), gutter if other == "e" else math.nan), raster))
  return out


samples = collections.defaultdict(list)  # (a, b) with a left of b -> [(x, y, z, w, line id, s)]
with multiprocessing.get_context("fork").Pool(min(16, os.cpu_count())) as pool:  # (samples are independent)
  for got in pool.imap(look, seen, chunksize=64):
    for key, q, raster in got:
      if raster is not None:
        raster_found[raster] += 1
      if key is not None:
        samples[key].append(q)
print("raster-only lines used:", raster_found[False], "before a line,", raster_found[True], "before the edge")


EDGE_IN, EDGE_OUT, ALONG = 0.5, 1.3, 1.5  # outer bands of a lane, 0.5-1.3 m in from each boundary, +-1.5 m along


def hatch_share(m, e, z, w):
  """share of painted raster in the lane's two outer bands around m: chevrons reach a lane's sides, arrows and text
  sit in its middle."""
  n = np.array([e[1], -e[0]])
  if w / 2 - EDGE_OUT < 0.3:  # too narrow for an arrow: the whole space between the lines
    t = np.arange(-w / 2 + EDGE_IN, w / 2 - EDGE_IN, 0.1)
    if not len(t):
      return 0.0
  else:
    t = np.r_[np.arange(-w / 2 + EDGE_IN, -w / 2 + EDGE_OUT, 0.1), np.arange(w / 2 - EDGE_OUT, w / 2 - EDGE_IN, 0.1)]
  a = np.arange(-ALONG, ALONG + 1e-6, 0.1)
  T, A = np.meshgrid(t, a)
  X, Y = m[0] + T * n[0] + A * e[0], m[1] + T * n[1] + A * e[1]
  code, h = scan.T.sample(X.ravel(), Y.ravel(), z, zwin=1.0)
  return float(((code >> 3) > 0).mean())


# both lines of a lane see it: keep the line that saw more of it, its samples in order along it
best = {}
for key, sm in samples.items():
  # the same lane seen from both its lines; a generic boundary (edge, raster-only line) has no samples of its own, so
  # its lanes are only seen from the named line, in that line's order: keep their key's order (left and right differ)
  u = key if set(key) & {"e", "r"} else tuple(sorted(map(str, key)))
  byline = collections.defaultdict(list)
  for q in sm:
    byline[q[4]].append(q)
  lid, qs = max(byline.items(), key=lambda kv: len(kv[1]))
  if u not in best or len(qs) > len(best[u][1]):
    best[u] = (key, qs)
# the line that saw the most of a painted median may not see all of it (its far line stored in pieces that stop 7 m
# short of its nose): the other lines' samples carrying on past either end, in line with it, extend it (placed along
# it by how far past the end they lie, within CARRY m). Medians only: lanes keep the one line's view (extending them
# re-picks which of two overlapping runs survives on the motorway)
CARRY = 30.0
allq = collections.defaultdict(list)
for key, sm in samples.items():
  allq[key if set(key) & {"e", "r"} else tuple(sorted(map(str, key)))].extend(sm)
for u, (key, qs) in best.items():
  qs = sorted(qs, key=lambda q: q[5])
  if len(qs) < 2 or not all(k in lines and lines[k]["colour"] == "yellow" for k in key):
    continue
  P, S = np.array([q[:2] for q in qs]), [q[5] for q in qs]
  ends = []
  for p, prev, s0, sg in ((P[0], P[1], S[0], -1), (P[-1], P[-2], S[-1], 1)):
    e = p - prev
    ends.append((p, e / max(np.linalg.norm(e), 1e-9), s0, sg))
  extra = {}
  for q in allq[u]:
    if q[4] == qs[0][4]:
      continue
    for p, e, s0, sg in ends:
      v = np.array(q[:2]) - p
      past = v @ e
      if STEP / 2 < past <= CARRY and abs(v[0] * e[1] - v[1] * e[0]) < q[3] / 2:
        sn = s0 + sg * past
        extra.setdefault(round(sn / STEP), q[:5] + (sn,) + q[6:])
  best[u] = (key, qs + list(extra.values()))
# outer-band paint share: lanes with arrows on the interchange read at most 0.02, a gore's chevrons 0.06-0.19
HATCH = 0.04
shares = []
runs = []  # (key, samples x y z w)
for key, qs in best.values():
  qs.sort(key=lambda q: q[5])
  cur = [qs[0]]
  for q in qs[1:] + [None]:
    if q is None or q[5] - cur[-1][5] > GAP:
      if len(cur) >= 3:
        # samples missing within the run (a line crossed it there) filled in, so it stays one sample per STEP
        s = np.array([c[5] for c in cur])
        si = np.arange(s[0], s[-1] + 1e-6, STEP)
        A = np.array([c[:4] for c in cur])
        # a lane to the kerb with a gutter along most of it: where the surface raster has no gutter (it holds the
        # gutter in pieces along a kerb), the lane still stops a gutter's width short of the kerb
        g = np.array([c[6] for c in cur])
        if "e" in key and (g > 0).sum() >= GUTTER_SHARE * (~np.isnan(g)).sum() > 0:
          d = np.gradient(A[:, :2], axis=0)
          d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
          sg = 1 if key[1] == "e" else -1  # the kerb's side, right of the samples' order or left
          fix = g == 0
          gw = float(np.median(g[g > 0]))
          A[fix, :2] -= sg * np.stack([d[fix, 1], -d[fix, 0]], 1) * gw / 2
          A[fix, 3] -= gw
        runs.append([key, np.stack([np.interp(si, s, A[:, k]) for k in range(4)], 1), ({key[0]}, {key[1]})])
      cur = []
    if q is not None:
      cur.append(q)

# a lane carries on across a break in one of its lines where its other line runs on unbroken through the gap (a
# dashed line stored as two pieces); in a junction both lines stop, so nothing joins across one. JOIN is provisional:
# the one break measured on the interchange is 8.2 m; re-measure on a bigger area
JOIN = 12.0
JOIN_W = 0.5  # the narrower end at least this share of the wider
line_pts = collections.defaultdict(list)
for lid, s_, c, n, z, xs in seen:
  line_pts[find(lid)].append((c[0], c[1], z))
line_pts = {k: np.array(v) for k, v in line_pts.items()}


def heading(A):
  d = A[-1, :2] - A[0, :2]
  return d / max(np.linalg.norm(d), 1e-9)


joined = True
while joined:
  joined = False
  for i in range(len(runs)):
    for j in range(len(runs)):
      if i == j or runs[i] is None or runs[j] is None:
        continue
      (ka, A, ida), (kb, B, idb) = runs[i], runs[j]
      if heading(A) @ heading(B) < 0:
        B, idb = B[::-1], idb[::-1]
      shared = sorted((x for x in (ida[0] & idb[0]) | (ida[1] & idb[1]) if x != "e"), key=str)  # sorted: sets of str vary by run
      gap = B[0, :2] - A[-1, :2]
      along = gap @ heading(A)
      if not shared or not 0 < along <= JOIN or abs(gap @ np.array([heading(A)[1], -heading(A)[0]])) > A[-1, 3] / 4:
        continue
      # a break is in a line: where the other side goes from the road's edge to a line it's another space (the sliver
      # between a crossing's line and the kerb rounding its corner, then the crossing)
      if any(("e" in ida[s]) != ("e" in idb[s]) for s in (0, 1) if not (ida[s] & idb[s]) - {"e"}):
        continue
      if abs(A[-1, 2] - B[0, 2]) > 1:
        continue
      # (the same lane is about as wide either side of a break: not a lane and a sliver beyond it, such as the space
      # between its line and a traffic island's kerb)
      if min(A[-1, 3], B[0, 3]) < JOIN_W * max(A[-1, 3], B[0, 3]):
        continue
      mid = (A[-1, :2] + B[0, :2]) / 2
      S = line_pts.get(shared[0])
      if S is None or np.min(np.linalg.norm(S[:, :2] - mid, axis=1)) > A[-1, 3] / 2 + 1.0:
        continue
      k = max(1, int(round(np.linalg.norm(gap) / STEP)))
      fill = A[-1] + (B[0] - A[-1]) * (np.arange(1, k)[:, None] / k)
      runs[i] = [ka, np.vstack([A, fill, B]), (ida[0] | idb[0], ida[1] | idb[1])]
      runs[j] = None
      joined = True
runs = [r for r in runs if r is not None]

# a run lying almost wholly inside a longer run's band on its deck is a leftover of a break the longer one already
# spans (found again out to the road's edge where a line was missing): drop it
def inside(A, B):
  """share of A's samples within B's band (centre within half of B's width, same deck)."""
  d = np.linalg.norm(A[:, None, :2] - B[None, :, :2], axis=2)
  k = d.argmin(1)
  return float(np.mean((d[np.arange(len(A)), k] < B[k, 3] / 2) & (np.abs(A[:, 2] - B[k, 2]) < 1)))


keep_run = [True] * len(runs)
for i, (ki, A, _) in enumerate(runs):
  for j, (kj, B, _) in enumerate(runs):
    if i != j and keep_run[j] and len(B) > len(A) and inside(A, B) >= 0.8:
      keep_run[i] = False
      break
runs = [r for r, k in zip(runs, keep_run) if k]

# a lane widening on one side only while both its sides carry on, out past where one lane can be: a road widening
# ahead of a split whose dividing line isn't painted (GTA marks some with faint road texture and raised markers only),
# or narrowing after a merge (read against the traffic). The lane carries straight on at its usual width along its
# straight side (a line, or the road's edge: a ramp's deck with no paint beside its gutter), and a new lane opens in the
# space beside it out to the far line, its inner side "u" (an unpainted divide); connect.py links it and shapes its taper
SPLIT_WIDEN = 1.5  # m wider than its usual width at its end (the east road's: 6.6 -> 8.6 m over its last 21 m)
SPLIT_RUN = 15  # samples before it starts widening (within SPLIT_START m of its usual width) its edges' ways are read over
SPLIT_START = 0.2
SPLIT_EDGE = 0.5  # m: the most its straight line is from where it's expected, carrying on
# m more one edge moves out than the other (the east road's: 1.44 and 0.84 m, the road bending 2 deg as it widens)
SPLIT_SIDE = 0.5
SPLIT_FAR = 2 * LANE_MAX  # m: lines further apart aren't one road's
SPLIT_LEN = 200  # m at most carried on
SPLIT_JUMP = 1.0  # m the far line moves sideways at most from one metre to the next
SPLIT_MIN = 10  # m at least carried on: a taper is longer (the ramp deck's 57 m, the east road's 43)
EDGE_SMOOTH = 9  # samples a lane carried on along the road's edge is smoothed over
wide_line = {r["id"] for r in scan.src.paint() if r["width"] >= 0.4}  # stop lines and crossings' lines (0.43-0.52 m; lane lines 0.10-0.31)


def widen_split(B0, ids, others, pre=None):
  """for a run B0 (ids: its left and right boundaries along it; pre: the samples of the run it carries on from, if
  any, read for its usual width and way) widening at its end as above: B0's samples with its widened stretch narrowed
  back to its usual width along its straight side, the samples it carries on with, the new lane's samples beside both,
  and which side it opens on (+1 right); else None (others: the other runs' samples)."""
  B = np.vstack([pre, B0]) if pre is not None else B0
  n_pre = len(B) - len(B0)
  w0 = float(np.median(B[:, 3]))
  if len(B) < SPLIT_RUN + 2 or B[-1, 3] < w0 + SPLIT_WIDEN:
    return None
  # (from where it starts widening)
  k0 = max([k for k in range(len(B)) if B[k, 3] <= w0 + SPLIT_START] + [0])
  if k0 < SPLIT_RUN:
    return None
  g = np.gradient(B[:, :2], axis=0)
  g /= np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
  nk = np.stack([g[:, 1], -g[:, 0]], 1)
  move = {}
  for s in (1, -1):  # how far each edge has moved out from the line it ran along before (fitted there)
    E = B[:, :2] + s * nk * B[:, 3:4] / 2
    R = E[k0 - SPLIT_RUN:k0 + 1]
    e = np.linalg.svd(R - R.mean(0))[2][0]
    e = e if e @ (R[-1] - R[0]) > 0 else -e
    move[s] = float((E[-1] - R.mean(0)) @ np.array([e[1], -e[0]]) * s)
  painted = [bool(ids[q] - {"e", "r", "u"}) for q in (0, 1)]
  if painted[0] != painted[1]:
    # a line one side, the road's edge (kerb) the other: the lane against the kerb carries on, and the one opening (or,
    # against the traffic, ending) is the one along the line (the ramp deck's merge; which edge moves out more there
    # depends on how the road bends, not on which lane drops)
    if ids[0 if painted[1] else 1] != {"e"}:
      return None
    s = 1 if painted[1] else -1
  else:
    s = 1 if move[1] > move[-1] else -1  # the side it opens on
    if move[s] - move[-s] < SPLIT_SIDE or not painted[0]:
      return None
  straight = ids[0 if s > 0 else 1]
  by_edge = not (straight - {"e", "r", "u"})
  # its own widened stretch: back to its usual width along its straight side, the rest the new lane's start
  B1, tail = B0.copy(), []
  for k in range(max(k0 + 1, n_pre), len(B)):
    e = B[k, :2] - s * nk[k] * B[k, 3] / 2
    B1[k - n_pre] = [*(e + s * nk[k] * w0 / 2), B[k, 2], min(w0, B[k, 3])]
    if B[k, 3] - w0 > 0.1:
      tail.append([*(e + s * nk[k] * (w0 + B[k, 3]) / 2), B[k, 2], B[k, 3] - w0])
  edge = B[-SPLIT_RUN:, :2] - s * nk[-SPLIT_RUN:] * B[-SPLIT_RUN:, 3:4] / 2  # the straight side
  d = edge[-1] - edge[0]
  d /= max(np.linalg.norm(d), 1e-9)
  n = np.array([d[1], -d[0]])
  p, z = edge[-1], B[-1, 2]
  keep, new, prev_far = [], [], None
  for _ in range(int(SPLIT_LEN / STEP)):
    q = p + d * STEP
    if not scan.inbox(q[0], q[1], 5.0) or any(x[1] in wide_line for x in scan.crossings(p, d, z, 0.0, STEP)):
      break
    xs = [x for x in scan.crossings(q, n, z, -SPLIT_FAR, SPLIT_FAR) if x[1] not in wide_line and x[1] in lines]  # (drawn ones)
    if by_edge:  # the road's edge (to the gutter, as a lane's "e" side), found from a metre inside the lane
      inside = q + s * n * 1.0
      ext, _ = scan.extent(inside, n, z, -s, scan.ROAD)
      at = [x for x in [s * (1.0 - ext)] if abs(x) <= SPLIT_EDGE]
    else:
      at = [x[0] for x in xs if abs(x[0]) <= SPLIT_EDGE]  # its straight line, carrying on
    far = sorted(s * x[0] for x in xs if s * x[0] > w0 + LANE_MIN)
    if not at or not far:
      break
    farpt = q + s * n * far[0]  # (before q moves onto the straight side: far is measured from q)
    if prev_far is not None and abs((farpt - prev_far) @ n) > SPLIT_JUMP:
      break  # the far side found is another line: the opening's far line can't jump a metre sideways in one
    prev_far = farpt
    q = q + n * min(at, key=abs)
    d = q - p
    d /= max(np.linalg.norm(d), 1e-9)
    n = np.array([d[1], -d[0]])
    c = q + s * n * w0 / 2
    code, h = scan.T.sample(np.array([c[0]]), np.array([c[1]]), z, zwin=1.0)
    if np.isfinite(h[0]):
      z = float(h[0])
    m = q + s * n * (w0 + far[0]) / 2
    if any(np.any((np.linalg.norm(R[:, :2] - m, axis=1) < R[:, 3] / 2) & (np.abs(R[:, 2] - z) < 1.5)) for R in others):
      return None  # (the opening has a lane of its own further on: a bay, linked by connect.py)
    keep.append([*c, z, w0])
    new.append([*m, z, far[0] - w0])
    p = q
  if len(keep) < SPLIT_MIN:
    return None
  keep, new = np.array(keep), np.array(new)
  if by_edge:  # (the road's edge read from the raster wavers by a few tenths: the lanes along it follow it smoothed)
    side = new[:, :2] - keep[:, :2]
    side /= np.maximum(np.linalg.norm(side, axis=1, keepdims=True), 1e-9)
    q = keep[:, :2] - side * w0 / 2  # (the edge)
    qs = ndi.uniform_filter1d(q, EDGE_SMOOTH, axis=0, mode="nearest")
    far = new[:, :2] + side * new[:, 3:4] / 2
    keep[:, :2] = qs + side * w0 / 2
    new[:, :2] = (qs + side * w0 + far) / 2
    new[:, 3] = np.linalg.norm(far - (qs + side * w0), axis=1)
  return B1, keep, np.array(tail + new.tolist()).reshape(-1, 4), s


def carries_on_from(B, bids, r):
  """the samples, along B, of the run B carries on from (its end within GAP m of B's start, heading B's way, with a line
  of B's on the same side), or None."""
  best = None
  for o in runs:
    if o is r:
      continue
    for O, oids in ((o[1], o[2]), (o[1][::-1], (o[2][1], o[2][0]))):
      dd = np.linalg.norm(O[-1, :2] - B[0, :2])
      head = (O[-1, :2] - O[max(len(O) - 5, 0), :2]) @ (B[min(4, len(B) - 1), :2] - B[0, :2])
      if dd <= GAP and head > 0 and ((oids[0] & bids[0]) | (oids[1] & bids[1])) - {"e", "r", "u"} and (best is None or dd < best[0]):
        best = (dd, O)
  return best and best[1]


SPLIT_ADD = []
for r in runs:
  key, A, ids = r
  others = [o[1] for o in runs if o is not r]
  for at_end in (True, False):
    B, bids = (A, ids) if at_end else (A[::-1], (ids[1], ids[0]))
    got = widen_split(B, bids, others, carries_on_from(B, bids, r))
    if got is None:
      continue
    B1, K, N, s = got
    far_ids = bids[1] if s > 0 else bids[0]
    nkey = ("u", key[1] if at_end else key[0]) if s > 0 else (key[0] if at_end else key[1], "u")
    nids = ({"u"}, far_ids) if s > 0 else (far_ids, {"u"})
    if at_end:
      A = np.vstack([B1, K])
    else:
      A = np.vstack([B1, K])[::-1]
      N, nkey, nids = N[::-1], nkey[::-1], nids[::-1]
    SPLIT_ADD.append([nkey, N, nids])
    print(f"widening split: {key} carried on {len(K)} m, a lane opening on its {'right' if s > 0 else 'left'} ({nkey})")
  r[1] = A
runs += SPLIT_ADD

def snap(p, n, z):
  """the painted line's middle within +-0.4 m of p across n on this deck, or None (a double line counts as one:
  following one half alone jumps to a neighbouring line where the halves part)."""
  t = np.arange(-0.4, 0.401, 0.05)
  X, Y = p[0] + t * n[0], p[1] + t * n[1]
  code, h = scan.T.sample(X, Y, z, zwin=1.0)
  on = ((code >> 3) > 0) & ~arrow_at(X, Y, z)  # (an arrow beside a gore's tip isn't its outline)
  if not on.any():
    return None, z
  return p + n * float(t[on].mean()), float(np.nanmean(h[on]))


def edge_snap(p, out, z):
  """the road surface's edge within +-0.6 m of p, going outwards along out on this deck, or None."""
  t = np.arange(-0.6, 0.601, 0.05)
  code, h = scan.T.sample(p[0] + t * out[0], p[1] + t * out[1], z, zwin=1.0)
  road = np.isin(code & 7, scan.DRIVEABLE)
  k = np.flatnonzero(road[:-1] & ~road[1:])
  if not len(k):
    return None, z
  return p + out * float(t[k[0]] + 0.025), float(h[k[0]])


def snap_in(p, inward, z, reach):
  """snap, else the first paint inward of p within reach m (an outline closing in faster than snap's window)."""
  q, zq = snap(p, inward, z)
  if q is not None or reach <= 0.4:
    return q, zq
  t = np.arange(0.4, reach, 0.05)
  X, Y = p[0] + t * inward[0], p[1] + t * inward[1]
  code, h = scan.T.sample(X, Y, z, zwin=1.0)
  paint = ((code >> 3) > 0) & ~arrow_at(X, Y, z)
  on = np.flatnonzero(paint)
  if not len(on):
    return None, z
  k = on[0]
  while k + 1 < len(t) and paint[k + 1]:
    k += 1
  return p + inward * float((t[on[0]] + t[k]) / 2), float(h[on[0]])


def tip(B, edge=(False, False)):
  """samples carrying a narrowing band on past B's last sample to where its two boundaries meet, following each
  outline's paint (a gore's outline is in the raster only and slants in too steeply for raster_line's straight test
  near the tip), or the road's edge on a side that is the edge (a shoulder closing to where kerb and line touch);
  edge says which of B's sides, left and right along it, are the road's edge."""
  d = B[-1, :2] - B[max(len(B) - 4, 0), :2]
  d /= max(np.linalg.norm(d), 1e-9)
  n = np.array([d[1], -d[0]])
  z = B[-1, 2]
  sides = [B[-1, :2] - n * B[-1, 3] / 2, B[-1, :2] + n * B[-1, 3] / 2]
  w, out = B[-1, 3], []
  for _ in range(int(GORE_TIP / STEP)):
    snapped = [edge_snap(p + d * STEP, -n if k == 0 else n, z) if edge[k] else snap_in(p + d * STEP, n if k == 0 else -n, z, w)
               for k, p in enumerate(sides)]
    if any(p is None for p, _ in snapped):
      break
    sides = [p for p, _ in snapped]
    w2 = float((sides[1] - sides[0]) @ n)
    if w2 > w + 0.2:  # the boundaries part again: not a tip
      break
    z = snapped[0][1]
    if w2 <= TWIN:  # they meet, or run on together as one double line
      break
    out.append([*((sides[0] + sides[1]) / 2), z, w2])
    w = w2
  return np.array(out).reshape(-1, 4)


GORE_TIP = 30.0  # m at most traced past a gore's run end
TIP_BACK = 4  # samples back from a gore's run end its tip is traced from
GORE_W = 3.0  # m: unhatched space beside a gore narrower than this is the gore's unhatched end
ISLAND_LEN = 10  # m: a strip narrower than any lane, between white lines and shorter than this, is a painted island
NARROWING = 0.3  # m narrower over a run's last 4 m: its boundaries are closing in
GORE_MIN = 9  # m hatched at least for a lane-wide run (crossing strips and junction paint are shorter)
out = []
for key, A, ids in runs:
  d = np.gradient(A[:, :2], axis=0)
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  share = np.array([hatch_share(A[k, :2], d[k], A[k, 2], A[k, 3]) for k in range(len(A))])
  # chevrons repeat about every 5 m (measured on the interchange's gores): bridge the gaps between them, and drop
  # stretches shorter than a gore (an arrow's tip reaching a band)
  # (closing pads with the run's own end values: padding with "not hatched" erodes 3 samples off a gore at a run's
  # end, which dropped a 12 m gore under the 9 m minimum)
  hat = ndi.binary_closing(np.pad(share >= HATCH, 7, mode="edge"), np.ones(7, bool))[7:-7]
  hat = ndi.binary_opening(np.r_[False, hat, False], np.ones(GORE_MIN, bool))[1:-1]
  # a strip narrower than any lane between white lines, hatched anywhere, is a gore (a junction approach's small one is
  # hatched over only its wide 5 m; between yellow lines a strip is a median, and its paint reaches the bands)
  white = all(k == "r" or k in lines and lines[k]["colour"] == "white" for k in key)
  if all(k in lines and lines[k]["colour"] == "yellow" for k in key):
    hat[:] = False  # hatching between yellow lines is a painted median's (direction.py), not a gore
  if not hat.any() and white and np.median(A[:, 3]) < GORE_W and ((share >= HATCH).sum() >= 3 or len(A) < ISLAND_LEN):
    hat[:] = True  # (unhatched and short: a wedge where a solid line splits off beside a dashed one at a stop line)
  if hat.any():
    # the same space beside a gore but unhatched and narrower than any lane is the gore too (GTA doesn't hatch a gore
    # to its ends; lanes on the interchange are 4-7 m, shoulders, found separately at the road's edge, 1.5-3 m)
    lab, nl = ndi.label(~hat)
    for j in range(1, nl + 1):
      ix = np.flatnonzero(lab == j)
      if np.median(A[ix, 3]) < GORE_W:
        hat[ix] = True
    # a gore reaching a run's end carries on to its tip, if its outlines meet further on
    # (traced from TIP_BACK samples before the end: an outline slanting in to the tip is too steep for raster_line, which
    # then measures the run's last few metres to the next straight paint beyond it, such as an arrow's shaft)
    if hat[-1]:
      n = TIP_BACK if len(A) > 2 * TIP_BACK and hat[-TIP_BACK:].all() else 0
      T = tip(A[:len(A) - n])
      if len(T) < n:
        n, T = 0, tip(A)
      A, share, hat = np.vstack([A[:len(A) - n], T]), np.r_[share[:len(share) - n], np.zeros(len(T))], np.r_[hat[:len(hat) - n], np.ones(len(T), bool)]
    if hat[0]:
      n = TIP_BACK if len(A) > 2 * TIP_BACK and hat[:TIP_BACK].all() else 0
      T = tip(A[n:][::-1])[::-1]
      if len(T) < n:
        n, T = 0, tip(A[::-1])[::-1]
      A, share, hat = np.vstack([T, A[n:]]), np.r_[np.zeros(len(T)), share[n:]], np.r_[np.ones(len(T), bool), hat[n:]]
  # any other run narrowing at an end (a median closing, a lane merging away, the space beside a gore's tip) carries on
  # to where its two boundaries meet
  edge = (key[0] == "e", key[1] == "e")
  if len(A) >= 5 and not hat[-1] and A[-1, 3] < A[-5, 3] - NARROWING:
    T = tip(A, edge)
    A, share, hat = np.vstack([A, T]), np.r_[share, np.zeros(len(T))], np.r_[hat, np.zeros(len(T), bool)]
  if len(A) >= 5 and not hat[0] and A[0, 3] < A[4, 3] - NARROWING:
    T = tip(A[::-1], edge[::-1])[::-1]
    A, share, hat = np.vstack([T, A]), np.r_[np.zeros(len(T)), share], np.r_[np.zeros(len(T), bool), hat]
  shares.extend(share.tolist())
  # a gore's tapering tip is too narrow for the bands: a stretch of the same space beside a gore that narrows, away
  # from it, to under half the gore's width is its tip (a lane carrying on past a gore keeps its width)
  if hat.any():
    gw = np.median(A[hat, 3])
    lab, nl = ndi.label(~hat)
    for j in range(1, nl + 1):
      ix = np.flatnonzero(lab == j)
      after = ix[0] > 0 and hat[ix[0] - 1]  # the gore is before this stretch: its far end is its last sample
      before = ix[-1] < len(hat) - 1 and hat[ix[-1] + 1]
      far = A[ix[-1], 3] if after else A[ix[0], 3] if before else None
      if far is not None and far < gw / 2:
        hat[ix] = True
  # pieces: lane or gore, split where the hatching starts or stops
  k0 = 0
  for k in range(1, len(A) + 1):
    if k == len(A) or hat[k] != hat[k0]:
      if k - k0 >= 3:
        out.append({"kind": "gore" if hat[k0] else "lane", "left": key[0], "right": key[1],
                    "left_all": sorted(ids[0], key=str), "right_all": sorted(ids[1], key=str),
                    "pts": [[round(float(v), 2) for v in p] for p in A[k0:k]], "hatch": [round(float(v), 3) for v in share[k0:k]]})
      k0 = k
h, bins = np.histogram(shares, bins=[0, .01, .02, .04, .06, .08, .1, .15, .2, .3, .5, 1.01])
print("outer-band paint share:", " ".join(f"{bins[i]:.2f}:{h[i]}" for i in range(len(h))))
json.dump(out, open(sys.argv[5], "w"))
print(len(lines), "lines,", sum(o["kind"] == "lane" for o in out), "lane runs,", sum(o["kind"] == "gore" for o in out), "gore runs ->", sys.argv[5])
