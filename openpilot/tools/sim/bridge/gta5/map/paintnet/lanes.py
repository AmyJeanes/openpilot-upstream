"""lanes.py x0 x1 y0 y1 out.json: lanes as the space between neighbouring lane lines (or a line and the road's edge).

Every lengthwise paint line (MINLEN m or longer, so hatching and arrows drop out) is sampled every STEP m; square to it
at each sample, the nearest boundary either side on its deck is found: another lengthwise line (a double line's two
halves count as one), or the road surface's edge (kerb, pavement, gutter or nothing) where that comes first. Neighbours
LANE_MIN to LANE_MAX m apart bound a lane. Lane samples are keyed by their two boundaries (line ids; "e" for the edge)
and chained along: a lane is one key's run, split at gaps over GAP m.
Out: [{"left": id, "right": id, "left_all"/"right_all": every boundary id along it (lanes joined across a break),
"pts": [[x, y, z, width], ...]}] with left/right as seen along pts."""
import json, math, sys, collections
import numpy as np
from scipy import ndimage as ndi
import scan
import arrows

STEP, MINLEN, LANE_MIN, LANE_MAX, GAP = 1.0, 6.0, 1.5, 9.0, 3.0
# twin lines: nearest-other-line distances on the interchange peak at 0-0.3 m, dip at 0.3-0.4, lanes from ~4 m; and they
# must stay that close along most of where they see each other (lines meeting at a gore's nose only touch at the tip)
TWIN, TWIN_SHARE = 0.35, 0.8
x0, x1, y0, y1 = scan.x0, scan.x1, scan.y0, scan.y1
lines = {}
for l in open(scan.PAINT):
  r = json.loads(l)
  P = np.array(r["pts"], float)
  if r["len"] < MINLEN or not ((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)).any():
    continue
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
h, bins = np.histogram(near_any, bins=[0, .1, .2, .3, .4, .5, .6, .8, 1, 1.5, 2, 3, 4, 10])
print("nearest other line:", " ".join(f"{bins[i]:.1f}:{h[i]}" for i in range(len(h))))

# solid lines the line extraction missed (a gore's outline drawn as part of its chevron marking) are still painted in
# the surface raster: paint at the same offset all along +-RL_ALONG m, RL_MIN-RL_MAX m wide, painted over RL_SOLID of
# its length; chevrons move sideways along, arrows and text are masked out by the markings list and arrows the list
# misses by their shape (an arrow's shaft is a short solid lengthwise stroke)
RL_ALONG, RL_MIN, RL_MAX, RL_SOLID = 1.5, 0.08, 0.8, 0.9
marks = [json.loads(l) for l in open(scan.PAINT.replace("polylines", "features"))]
marks = [[f["x"], f["y"], f["z"], max(f.get("len", 0), f.get("wid", 0)) / 2 + 0.5] for f in marks
         if f["kind"] != "crossing" and scan.inbox(f["x"], f["y"], 20)]
code_, zz_ = arrows.load(x0 - 20, x1 + 20, y0 - 20, y1 + 20, 0.1)
found = arrows.find((code_ >> 3) > 0, zz_, x0 - 20, y1 + 20, 0.1)[1]
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
samples = collections.defaultdict(list)  # (a, b) with a left of b -> [(x, y, z, w, line id, s)]
for lid, s, c, n, z, xs in seen:
  me = find(lid)
  xs = [x for x in xs if find(x[1]) != me]
  for side in (1, -1):
    cand = sorted([x for x in xs if side * x[0] > 0], key=lambda x: side * x[0])
    edge, _ = scan.extent(c + n * side * 0.3, n, z, side)  # step off the line's own paint first
    edge += 0.3
    if cand and side * cand[0][0] < edge:
      o, other = side * cand[0][0], find(cand[0][1])
    else:
      o, other = edge, "e"
    r = raster_line(c, n, z, side, o - TWIN)
    if r is not None:
      raster_found[other == "e"] += 1
      o, other = r, "r"
    if not LANE_MIN <= o <= LANE_MAX:
      continue
    m = c + n * side * o / 2
    # key as seen travelling along this line's point order: left boundary first
    key = (me, other) if side == 1 else (other, me)
    samples[key].append((float(m[0]), float(m[1]), float(z), float(o), lid, float(s)))
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
        runs.append([key, np.array([c[:4] for c in cur]), ({key[0]}, {key[1]})])
      cur = []
    if q is not None:
      cur.append(q)

# a lane carries on across a break in one of its lines where its other line runs on unbroken through the gap (a
# dashed line stored as two pieces); in a junction both lines stop, so nothing joins across one. JOIN is provisional:
# the one break measured on the interchange is 8.2 m; re-measure on a bigger area
JOIN = 12.0
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
      shared = [x for x in (ida[0] & idb[0]) | (ida[1] & idb[1]) if x != "e"]
      gap = B[0, :2] - A[-1, :2]
      along = gap @ heading(A)
      if not shared or not 0 < along <= JOIN or abs(gap @ np.array([heading(A)[1], -heading(A)[0]])) > A[-1, 3] / 4:
        continue
      if abs(A[-1, 2] - B[0, 2]) > 1:
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

out = []
for key, A, ids in runs:
  d = np.gradient(A[:, :2], axis=0)
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  share = np.array([hatch_share(A[k, :2], d[k], A[k, 2], A[k, 3]) for k in range(len(A))])
  # chevrons repeat about every 5 m (measured on the interchange's gores): bridge the gaps between them, and drop
  # stretches shorter than a gore (an arrow's tip reaching a band)
  hat = ndi.binary_closing(np.r_[False, share >= HATCH, False], np.ones(7, bool))[1:-1]
  hat = ndi.binary_opening(np.r_[False, hat, False], np.ones(9, bool))[1:-1]
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
