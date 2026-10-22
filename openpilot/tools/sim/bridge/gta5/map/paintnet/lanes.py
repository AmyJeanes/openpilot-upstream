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
# lines the game files hold but the game never draws (visible.py, against the captured imagery) aren't lines
vis = next((a[10:] for a in sys.argv if a.startswith("--visible=")), None)
hidden = set()
if vis:
  hidden = {int(k) for k, v in json.load(open(vis)).items() if v["drawn"] is False}
  lines = {k: v for k, v in lines.items() if k not in hidden}
  print(len(hidden), "lines not drawn in the game dropped")
# a piece shorter than MINLEN carrying on the end of a line (lines are stored in pieces: a 5.9 m piece bridges two
# motorway edge lines at a merge) is part of that line
ends = [(np.array(r["pts"][0][:2]), np.array(r["pts"][-1][:2])) for r in lines.values()]
for r in scan.src.paint():
  if r["len"] >= MINLEN or r["len"] < 1.0 or r["id"] in lines or (vis and r["id"] in hidden):
    continue
  P = np.array(r["pts"], float)
  if not ((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)).any():
    continue
  if any(min(np.linalg.norm(P[k, :2] - e) for e in pair) < PIECE_GAP for pair in ends for k in (0, -1)):
    lines[r["id"]] = r
keep = set(lines)


def resample(P, step):
  d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
  s = np.arange(step / 2, d[-1], step)
  return np.stack([np.interp(s, d, P[:, k]) for k in range(3)], 1), s


# pass 1: what each line sample sees square to it; lines running within TWIN m of each other are one marking (a double
# yellow, solid beside dashed), whether or not the extraction paired them
seen = []  # (lid, s, c, n, z, crossings)
near_any = []
parent = {}
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
