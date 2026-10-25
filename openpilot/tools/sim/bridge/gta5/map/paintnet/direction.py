"""direction.py x0 x1 y0 y1 lanes.json out.json: each lane's travel direction from the paint, checked against arrows and GTA links.

Rule (US markings, which GTA follows): yellow paint is on a driver's left (centre lines, left edges), white marks lanes
going the same way and the right edge. A lane's direction is the one that puts its nearest yellow line, looking
across its own road surface (kerb to kerb) at each sample, on its left. Lanes with no yellow in reach get no direction from paint ("?").
Checks, independent of the rule: road arrows inside the lane (their heading), and GTA links on the same deck within
half a lane of its centre (one-way links' direction; two-way links can't say).
Where its road surface has yellow only at both edges (GTA paints some one-way ramps so: yellow-on-the-left then gives
its lanes opposite ways), a lane with no one-way GTA link within half its width takes the links across that road
surface, if they all run one way (a wide ramp's one link runs down its middle; groups.py puts links before yellow).
Traffic: a lane must run alongside a GTA vehicle link (any direction) on its deck within FLOW_NEAR m, within FLOW_ANG
deg, over at least half its samples; strips between a stop line and a crossing, or across a junction, run across
every link and are marked "not_traffic".
Crossings: a strip between a crossing's two border lines spans the road, so the surface ends just past both its ends
(within STRIP_END m; measured on the interchange: crossing strips 1.7-2.4 m, every lane 11 m or more at one end) and
is no longer than a road is wide (CROSSING_MAX);
carried on to that edge, its area cut to the road's surface so its ends follow the kerbs). What its lines bound
against the road's edge is a kerb corner, not a shoulder: dropped.
Zebra crossings are where the game's crossing decals are (a strip mostly inside one), bars and all, and take the
decals' outline; the strips between its bars sharing a decal are one crossing.
Centre spaces: yellow on both sides (each side's own line) is the space between opposing traffic, not a lane for either way (US markings),
unless the road ends within SHOULDER m past both lines (a one-way ramp GTA paints yellow at both edges: a lane):
a painted median (solid both sides, often opening into back-to-back left-turn bays) or, with a dashed yellow on a side,
a two-way left-turn lane. Marked "median" / "twltl", given no direction.
Medians, two-way turn lanes, gores and shoulders must run alongside a GTA link as lanes must (no-traffic strips, across
it, only near one): what doesn't is paint off the road network (car parks, loading yards), dropped.
Shoulders (a line and the road's edge less than SHOULDER m apart: measured line-to-edge widths on the interchange
fall in 1.5-3.0 m or 5 m and up, none in 3.5-4.5) are marked, not given lanes.
A gore, median, two-way turn lane or shoulder whose end a stop line or crossing's line crosses (within END_BACK m of
it, or END_AHEAD m ahead) ends at that line's near edge, cut along it ("stops": {"start"|"end": [x, y, ux, uy]}).
A raised island interrupting a painted median or two-way turn lane is marked "island" (not driveable)."""
import json, math, sys, collections
import numpy as np
import cv2
import scan

SHOULDER = 4.0
STRIP_END = 5.0
# m: a crossing spans its road, so it's no longer than a road is wide (crossing strips measured 4-30 m); a lane between
# a line and the kerb can end against kerbs at both ends too (a loop ramp's edge lane: 315 m; lanes under a deck: 55, 65)
CROSSING_MAX = 40.0
PAINT_PAST = 0.5  # m a crossing reaches past its own paint's end (GTA paints a crossing to the gutter)
# stop lines and crossings' lines are 0.43-0.52 m wide, lane lines 0.10-0.31; the east junction's gores end 0.05 and
# 0.66 m past the middle of their crossing's line
STOP_WIDE, END_BACK, END_AHEAD = 0.4, 3.0, 1.0
ISLAND_GAP = 60.0  # m at most between a median's stretches either side of an island (the interchange's: 33 m)
ISLAND_REACH = 5.0  # m: an island is narrower than this either way (the interchange's: 2.3 m kerb to kerb)
# (a shoulder lies outside the outermost lane: up to SHOULDER m further from the link; a strip between a stop line
# and its crossing beside a junction's arm is 11-18 m from the links down the arms' middles)
ACROSS_NEAR = 25.0
FLOW_NEAR, FLOW_ANG = 12.0, 30.0  # lane samples to the nearest parallel link on the interchange: 0-12 m, then a drop (flowdist.py)
lanes = json.load(open(sys.argv[5]))
paint, style = {}, {}
for r in scan.src.paint():
  paint[r["id"]], style[r["id"]] = r["colour"], r["style"]
arrows = [a for a in scan.src.features() if a["kind"] == "arrow" and scan.inbox(a["x"], a["y"], 10)]

# GTA link segments with travel sense: one-way (fwd only or back only)
links = []
raw = {}
for o in scan.src.paths():
  if o["t"] == "l":
    raw[((o["a"], o["i"]), (o["ta"], o["ti"]))] = o
def segs(pairs):
  """GTA link segments as arrays (start, along, length, mid height), those 1 m or longer."""
  A = np.array([a[:3] for a, _ in pairs], float).reshape(-1, 3)
  B = np.array([b[:3] for _, b in pairs], float).reshape(-1, 3)
  ab = B[:, :2] - A[:, :2]
  L = np.linalg.norm(ab, axis=1)
  k = L >= 1
  return A[k, :2], ab[k], L[k], ((A[:, 2] + B[:, 2]) / 2)[k]


def near_segs(G, p, z, near):
  """the segments of G on this deck (within 3 m) passing within near m of p, and their unit directions."""
  A, ab, L, zm = G
  t = np.clip(((p - A) * ab).sum(1) / L ** 2, 0, 1)
  hit = (np.abs(zm - z) <= 3) & (np.linalg.norm(A + t[:, None] * ab - p, axis=1) <= near)
  return ab[hit] / L[hit, None]


every = segs([(A, B) for (ka, kb, A, B) in scan.L])


def flows(p, d, z, near=FLOW_NEAR):
  """a GTA link on this deck runs past p within near m, along d either way within FLOW_ANG deg."""
  return bool((np.abs(near_segs(every, p, z, near) @ d) >= math.cos(math.radians(FLOW_ANG))).any())


for (ka, kb, A, B) in scan.L:
  l = raw.get((ka, kb)) or raw.get((kb, ka))
  f = (l["f"][2] >> 5) & 7, (l["f"][2] >> 2) & 7
  if raw.get((ka, kb)) is None:
    f = f[::-1]
  if f[0] and not f[1]:
    links.append((A, B))
  elif f[1] and not f[0]:
    links.append((B, A))
one_way = segs(links)


def link_votes(p, d, z, w):
  """+1 / -1 per one-way link on this deck passing within w/2 of p, running along d / against it."""
  c = near_segs(one_way, p, z, w / 2) @ d
  return int((c > 0.9).sum() - (c < -0.9).sum())


def road_votes(p, d, z):
  """link_votes across p's whole road surface (kerb to kerb, square to d), where every one-way link there runs the same
  way (a two-way road's links running both ways say nothing)."""
  n = np.array([d[1], -d[0]])
  lo, hi = scan.extent(p, n, z, -1)[0], scan.extent(p, n, z, 1)[0]
  c = near_segs(one_way, p + n * (hi - lo) / 2, z, (lo + hi) / 2) @ d
  c = c[np.abs(c) > 0.9]
  return int(np.sign(c[0]) * len(c)) if len(c) and (np.all(c > 0) or np.all(c < 0)) else 0


# zebra crossings: the game's crossing decals (their outlines); a strip lying mostly inside one is that crossing, its
# bars read as hatching or not (rail crossings are other decals)
zebras = [(np.array(f["poly"], np.float32), f["z"]) for f in scan.src.features()
          if f["kind"] == "crossing" and "road_crossing" in f["tex"] and len(f["poly"]) >= 3 and scan.inbox(f["x"], f["y"], 20)]
ZEBRA_SHARE = 0.5


def zebra(A):
  """the decals (indices in zebras) a strip lies mostly inside, or []."""
  A = np.asarray(A)
  inside, hit = np.zeros(len(A), bool), []
  for k, (poly, z) in enumerate(zebras):
    m = np.array([abs(p[2] - z) < 2 and cv2.pointPolygonTest(poly, (float(p[0]), float(p[1])), False) >= 0 for p in A])
    if m.any():
      hit.append(k)
    inside |= m
  return hit if inside.mean() >= ZEBRA_SHARE else []


def zebra_strip(A, polys):
  """the decals' own outline as samples every metre across the road (the way the road is narrowest, kerb to kerb,
  through the decals' middle: the strip found between two of a zebra's bars runs along it): the middle and width of
  the decals across."""
  A = np.asarray(A)
  V = np.vstack(polys).astype(float)
  m, z = V.mean(0), float(np.median(A[:, 2]))
  span = lambda u: scan.extent(m, u, z, 1)[0] + scan.extent(m, u, z, -1)[0]
  e = min((np.array([math.cos(a), math.sin(a)]) for a in np.radians(np.arange(0, 180, 5))), key=span)
  e /= max(np.linalg.norm(e), 1e-9)
  n = np.array([e[1], -e[0]])
  o = A[0, :2]
  out = []
  for t in np.arange(((V - o) @ e).min() + 0.5, ((V - o) @ e).max(), 1.0):
    across = []
    for P in polys:  # where the line square to the strip at t crosses each outline
      P = P.astype(float)
      a, b = (P - o) @ e - t, (P - o) @ n
      for i in range(len(P)):
        j = (i + 1) % len(P)
        if (a[i] <= 0) != (a[j] <= 0):
          across.append(b[i] + (b[j] - b[i]) * a[i] / (a[i] - a[j]))
    if len(across) >= 2:
      lo, hi = min(across), max(across)
      c = o + e * t + n * (lo + hi) / 2
      k = int(np.argmin(np.linalg.norm(A[:, :2] - c, axis=1)))
      out.append([round(float(c[0]), 2), round(float(c[1]), 2), float(A[k, 2]), round(float(hi - lo), 2)])
  return out



def axis(A):
  """a crossing's way along it, first end to last (its middle line drifts sideways where a zebra's decal tapers)."""
  P = np.asarray(A, float)[:, :2]
  e = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][0]
  return e if e @ (P[-1] - P[0]) >= 0 else -e


def end_reach(A):
  """how far the road surface carries on past each end of a strip, along it from its middle: (last end, first end)."""
  A = np.asarray(A, float)
  e = axis(A)
  return [scan.extent(p[:2], s * e, p[2], 1)[0] for p, s in ((A[-1], 1), (A[0], -1))]


def paint_reach(p, e, paint, w):
  """how far past p along e a crossing's own paint (points [x, y] within its width w and 1 m of its band) reaches, or
  None."""
  near = paint[np.abs((paint - p) @ np.array([e[1], -e[0]])) <= w / 2 + 1]
  return float(((near - p) @ e).max()) if len(near) else None


def to_kerb(A, paint):
  """a crossing carried on at its width to the road's edge at either end, from its middle (its lines stop short of a
  rounded kerb corner), but no further than PAINT_PAST past its own paint (the road carrying on into the next crossing)."""
  A = np.asarray(A, float)
  ext = []
  for (p, e), o in zip(((A[-1], axis(A)), (A[0], -axis(A))), end_reach(A)):
    reach = paint_reach(p[:2], e, paint, A[:, 3].max())
    o = min(o, STRIP_END, reach + PAINT_PAST if reach is not None else STRIP_END)
    t = np.r_[np.arange(1.0, o, 1.0), o] if o > 0.5 else np.zeros(0)
    ext.append([[*(p[:2] + e * v), p[2], p[3]] for v in t])
  return [[round(float(v), 2) for v in q] for q in ext[1][::-1] + A.tolist() + ext[0]]


OUTLINE_RES, OUTLINE_PAST = 0.1, 0.5  # m: raster step; carried past each end beyond half its width
OUTLINE_SLIVER = 9  # raster steps: narrower slivers dropped
line_xy = {r["id"]: np.array(r["pts"], float)[:, :2] for r in scan.src.paint() if scan.inbox(r["pts"][0][0], r["pts"][0][1], 60)}


def band(A, past=0.0):
  """a crossing's band as a ring of [x, y]: centre +- half its width, carried on along it past each end by past m plus
  half its widest (past None: not carried on)."""
  A = np.asarray(A, float)
  if past is None:
    xy, w = A[:, :2], A[:, 3]
  else:  # (by its widest half width: a zebra's tapering end is still as far from the kerb)
    e, h = axis(A), A[:, 3].max() / 2 + past
    xy = np.vstack([A[0, :2] - e * h, A[:, :2], A[-1, :2] + e * h])
    w = np.r_[A[0, 3], A[:, 3], A[-1, 3]]
  dd = np.gradient(xy, axis=0)
  dd /= np.maximum(np.linalg.norm(dd, axis=1, keepdims=True), 1e-9)
  n = np.stack([dd[:, 1], -dd[:, 0]], 1) * w[:, None] / 2
  return xy, np.vstack([xy - n, (xy + n)[::-1]])


def outline(A, paint, others=(), own=()):
  """a crossing's area as a ring of [x, y]: its band carried on past each end by half its width and OUTLINE_PAST (a
  kerb meeting its end at up to 45 deg, or rounding a corner, is reached on both sides), cut to the road's surface
  on its deck, the part joined to its middle: its ends follow the kerbs. Where the road carries on past an end (into
  the next crossing), it stops PAINT_PAST past its own paint (points [x, y]: its lines, or its decals' outline), and
  what it reaches past its own ground (its band, and own: rings, a zebra's decals) inside another crossing's (others:
  (ring, hard)) is that one's, and anything inside a hard one (a zebra's decal: its paint) is."""
  A = np.asarray(A, float)
  ax = axis(A)
  xy, ring = band(A, OUTLINE_PAST)
  zs = np.r_[A[0, 2], A[:, 2], A[-1, 2]]
  lo, hi = np.vstack([ring, *own]).min(0) - 1, np.vstack([ring, *own]).max(0) + 1
  W, H = (np.ceil((hi - lo) / OUTLINE_RES)).astype(int)
  px = lambda P: np.stack([(P[:, 0] - lo[0]) / OUTLINE_RES - 0.5, (hi[1] - P[:, 1]) / OUTLINE_RES - 0.5], 1)
  mask = np.zeros((H, W), np.uint8)
  for o in [ring, *own]:  # (a zebra's decals' slanted ends fall between its band's samples)
    cv2.fillPoly(mask, [np.round(px(o)).astype(np.int32)], 1)
  r, c = np.nonzero(mask)
  X, Y = lo[0] + (c + 0.5) * OUTLINE_RES, hi[1] - (r + 0.5) * OUTLINE_RES
  k = np.argmin((X[:, None] - xy[None, :, 0]) ** 2 + (Y[:, None] - xy[None, :, 1]) ** 2, axis=1)  # the deck's height there
  code, _ = scan.T.sample(X, Y, zs[k], zwin=1.0)
  ok = np.isin(code & 7, scan.DRIVEABLE)
  for p, e in ((A[-1, :2], ax), (A[0, :2], -ax)):
    reach = paint_reach(p, e, paint, A[:, 3].max())
    if reach is not None:
      ok &= (np.stack([X, Y], 1) - p) @ e <= reach + PAINT_PAST
  mine = np.zeros_like(mask)
  for o in [band(A, None)[1], *own]:  # (one at a time: one call fills where rings overlap by parity)
    cv2.fillPoly(mine, [np.round(px(o)).astype(np.int32)], 1)
  theirs = np.zeros_like(mask)
  for o, hard in others:
    cv2.fillPoly(theirs, [np.round(px(o)).astype(np.int32)], 2 if hard else 1)
  ok &= ~((theirs[r, c] == 2) | (theirs[r, c] == 1) & (mine[r, c] == 0))
  road = np.zeros_like(mask)
  road[r, c] = ok
  # (a sliver of road between its band's edge and the start of a kerb's rounding, along the kerb, isn't crossing)
  road = cv2.morphologyEx(road, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (OUTLINE_SLIVER,) * 2))
  _, lab = cv2.connectedComponents(road, connectivity=4)
  mc, mr = np.round(px(A[len(A) // 2, None, :2])[0]).astype(int)
  m = lab[mr, mc] if 0 <= mr < H and 0 <= mc < W else 0
  if m == 0:
    return None
  cs, _ = cv2.findContours((lab == m).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  C = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.0, True)[:, 0].astype(float)
  return [[round(float(lo[0] + (x + 0.5) * OUTLINE_RES), 2), round(float(hi[1] - (y + 0.5) * OUTLINE_RES), 2)] for x, y in C]


def near_link(p, z):
  """a GTA vehicle link on this deck within ACROSS_NEAR m of p, any way."""
  return len(near_segs(every, p, z, ACROSS_NEAR)) > 0


def on_network(A, along, near=FLOW_NEAR):
  """a strip lies on the road network: a GTA vehicle link runs alongside it within near m (along: within FLOW_ANG deg
  of it) or, for a strip across the road, within ACROSS_NEAR m, over half its samples."""
  A = np.asarray(A, float)
  d = np.gradient(A[:, :2], axis=0)
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  ks = range(0, len(A), 2)
  return np.mean([flows(A[k, :2], d[k], A[k, 2], near) if along else near_link(A[k, :2], A[k, 2]) for k in ks]) >= 0.5


# a zebra holds a strip between each pair of its bars: the strips sharing a decal are one crossing
hits = [zebra(r["pts"]) if r["kind"] in ("lane", "gore") else [] for r in lanes]
zp = list(range(len(zebras)))


def zroot(k):
  while zp[k] != k:
    zp[k] = zp[zp[k]]
    k = zp[k]
  return k


for h in hits:
  for k in h[1:]:
    zp[zroot(k)] = zroot(h[0])
zgroup = collections.defaultdict(list)  # a zebra's root decal -> its strips (lane indices)
for i, h in enumerate(hits):
  if h:
    zgroup[zroot(h[0])].append(i)

def edge_bound(A):
  """the road ends within SHOULDER m past both its lines over most of its length: yellow both sides is then a one-way
  road's edges (GTA paints some ramps' both edges yellow), not the space between opposing traffic."""
  A = np.asarray(A, float)
  d = np.gradient(A[:, :2], axis=0)
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  past = []
  for k in range(0, len(A), 3):
    n = np.array([d[k, 1], -d[k, 0]])
    past.append(max(scan.extent(A[k, :2], n * s, A[k, 2], 1)[0] for s in (1, -1)) - A[k, 3] / 2)
  return np.mean(np.array(past) < SHOULDER) > 0.5


out, stats = [], collections.Counter()
for i, r in enumerate(lanes):
  if hits[i]:
    grp = zgroup[zroot(hits[i][0])]
    if i != grp[0]:
      stats["zebra crossing's other strips"] += 1
      continue
    polys = [zebras[k][0] for k in sorted({k for g in grp for k in hits[g]})]
    pts = zebra_strip(r["pts"], polys)  # (the decals reach the kerbs)
    sides = {f"{s}_all": sorted({l for g in grp for l in lanes[g].get(f"{s}_all", [lanes[g][s]])}, key=str) for s in ("left", "right")}
    out.append({**r, **sides, "kind": "crossing", "zebra": True, "pts": pts, "paint": np.vstack(polys).astype(float), "decals": polys})
    stats["zebra crossing"] += 1
    continue
  if r["kind"] != "lane":  # gores pass through
    out.append(r)
    continue
  A = np.array(r["pts"])
  W = np.median(A[:, 3])
  sides = [r.get("left_all", [r["left"]]), r.get("right_all", [r["right"]])]
  # (each side's own line: a lane beside a gore or median touches its yellow paint in places)
  if all(paint.get(r[s]) == "yellow" for s in ("left", "right")) and not edge_bound(A):
    kind = "twltl" if any(paint.get(i) == "yellow" and style.get(i) == "dashed" for ids in sides for i in ids) else "median"
    out.append({**r, "kind": kind}); stats[kind] += 1
    continue
  if "e" in (r["left"], r["right"]) and W < SHOULDER:
    out.append({**r, "kind": "shoulder"}); stats["shoulder"] += 1
    continue
  d = np.gradient(A[:, :2], axis=0)
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  if len(A) >= 3 and max(end_reach(A)) <= STRIP_END and np.linalg.norm(np.diff(A[:, :2], axis=0), axis=1).sum() <= CROSSING_MAX:
    ids = [i for i in r.get("left_all", [r["left"]]) + r.get("right_all", [r["right"]]) if i in line_xy]
    paint_xy = np.vstack([line_xy[i] for i in ids])
    out.append({**r, "kind": "crossing", "pts": to_kerb(A, paint_xy), "paint": paint_xy}); stats["crossing"] += 1
    continue
  ks = range(0, len(A), 2)
  if np.mean([flows(A[k, :2], d[k], A[k, 2]) for k in ks]) < 0.5:
    out.append({**r, "kind": "not_traffic"}); stats["not_traffic"] += 1
    continue
  yl = 0  # +1: yellow on the left as seen along pts
  edges = []  # per sample: yellow only at both edges of its road surface (a one-way road GTA paints yellow both sides)
  for k in range(0, len(A), 3):
    n = np.array([d[k, 1], -d[k, 0]])
    lo, _ = scan.extent(A[k, :2], n, A[k, 2], -1)  # this road surface only: not across a median or kerb
    hi, _ = scan.extent(A[k, :2], n, A[k, 2], 1)
    xs = scan.crossings(A[k, :2], n, A[k, 2], -lo - 0.3, hi + 0.3)
    ys = [o for o, lid, col, sty, z in xs if col == "yellow"]
    edges.append(bool(ys) and all(o < -lo + SHOULDER or o > hi - SHOULDER for o in ys) and min(ys) < 0 < max(ys))
    if ys:
      o = min(ys, key=abs)
      yl += 1 if o < 0 else -1
  yellow_edges = bool(edges) and np.mean(edges) > 0.5
  sense = "+" if yl > 0 else "-" if yl < 0 else "?"
  # arrows inside the lane: heading (deg from east, counter-clockwise: the eastbound motorway's arrows read 0) vs the lane's direction
  av = 0
  for a in arrows:
    p = np.array([a["x"], a["y"]])
    k = int(np.argmin(np.linalg.norm(A[:, :2] - p, axis=1)))
    if np.linalg.norm(A[k, :2] - p) > A[k, 3] / 2 or abs(a["z"] - A[k, 2]) > 2:
      continue
    h = math.radians(a["heading"])
    ad = np.array([math.cos(h), math.sin(h)])
    c = ad @ d[k]
    if abs(c) > 0.7:
      av += 1 if c > 0 else -1
  lv = sum(link_votes(A[k, :2], d[k], A[k, 2], A[k, 3]) for k in range(0, len(A), 5))
  if not lv and yellow_edges:  # (a wide one-way ramp: its one link runs down the middle, not along each lane)
    lv = sum(road_votes(A[k, :2], d[k], A[k, 2]) for k in range(0, len(A), 5))
  check = lambda v: "agree" if v and (v > 0) == (sense == "+") else "disagree" if v else "none"
  stats[(sense, "arrows " + check(av), "links " + check(lv))] += 1
  out.append({**r, "sense": sense, "yellow_votes": yl, "arrow_votes": av, "link_votes": lv})
xs = [r for r in out if r["kind"] == "crossing"]
ground = {id(r): [(band(r["pts"], None)[1], False), *((d, True) for d in r.get("decals", []))] for r in xs}
for r in xs:
  others = [g for o in xs if o is not r and abs(np.median(np.array(o["pts"])[:, 2]) - np.median(np.array(r["pts"])[:, 2])) < 2 for g in ground[id(o)]]
  r["outline"] = outline(r["pts"], r.pop("paint"), others, r.get("decals", []))
for r in xs:
  r.pop("decals", None)
# a crossing's lines run across the road: what they bound to the road's edge is the kerb rounding a corner of it
xline = {i for r in out if r["kind"] == "crossing" for i in r.get("left_all", [r["left"]]) + r.get("right_all", [r["right"]])} - {"e"}
wide_ = {r["id"] for r in scan.src.paint() if r["width"] >= STOP_WIDE}


def corner(r):
  """a strip between a crossing's line and the road's edge; or a lane-wide wedge, shorter than it is wide, between a
  stop line or crossing's line (STOP_WIDE) and the edge, where a corner is wide enough to read as a lane."""
  if "e" not in (r["left"], r["right"]):
    return False
  other = {r["left"], r["right"]} - {"e"}
  if r["kind"] in ("shoulder", "not_traffic"):
    return other <= xline
  A = np.asarray(r["pts"], float)
  return r["kind"] == "lane" and other <= wide_ and np.linalg.norm(np.diff(A[:, :2], axis=0), axis=1).sum() < A[:, 3].max()


drop = [r for r in out if corner(r)]
stats["kerb corner at a crossing (dropped)"] = len(drop)
out = [r for r in out if not any(r is d for d in drop)]
# paint off the road network (a car park's or loading yard's bay lines, read as medians, turn lanes, gores): nothing
# along it that traffic drives; a strip between a stop line and a crossing runs across traffic, so only needs it near
off = [r for r in out if r["kind"] in ("median", "twltl", "gore", "shoulder", "not_traffic")
       and not on_network(r["pts"], r["kind"] != "not_traffic", FLOW_NEAR + SHOULDER if r["kind"] == "shoulder" else FLOW_NEAR)]
stats["off the road network (dropped)"] = len(off)
out = [r for r in out if not any(r is o for o in off)]
# a gore's or median's paint runs on into the crossing or stop line at its end: cut back to that line's near edge
width = {r["id"]: r["width"] for r in scan.src.paint()}
line_xy = {r["id"]: np.array(r["pts"], float)[:, :2] for r in scan.src.paint() if scan.inbox(r["pts"][0][0], r["pts"][0][1], 60)}
for r in out:
  if r["kind"] not in ("gore", "median", "twltl", "shoulder"):
    continue
  A, own, stops = np.array(r["pts"], float), set(r.get("left_all", [r["left"]]) + r.get("right_all", [r["right"]])), {}
  for which in ("end", "start"):
    B = A if which == "end" else A[::-1]
    if len(B) < 3:
      break
    e = B[-1, :2] - B[max(len(B) - 6, 0), :2]  # (over 5 m)
    e /= max(np.linalg.norm(e), 1e-9)
    xs = [x for x in scan.crossings(B[-1, :2], e, B[-1, 2], -END_BACK, END_AHEAD) if width.get(x[1], 0) >= STOP_WIDE and x[1] not in own and x[1] in line_xy]
    if not xs:
      continue
    o, lid = min(xs, key=lambda x: x[0])[:2]
    c = B[-1, :2] + e * o  # (where the line crosses the strip)
    L = line_xy[lid]
    k = int(np.argmin([np.linalg.norm(c - (L[m] + L[m + 1]) / 2) for m in range(len(L) - 1)]))
    u = (L[k + 1] - L[k]) / max(np.linalg.norm(L[k + 1] - L[k]), 1e-9)
    m = np.array([-u[1], u[0]]) * (-1.0 if np.array([-u[1], u[0]]) @ e > 0 else 1.0)  # square to the line, into the strip
    # the samples wholly before its near edge (the paint past its lines' ends drifts: a gore's centre at its tip), the
    # last carried on its own way to the edge
    keep = B[(B[:, :2] - c) @ m > width[lid] / 2 + 0.1]
    if len(keep) < 2:
      continue
    ek = keep[-1, :2] - keep[-2, :2]
    ek /= max(np.linalg.norm(ek), 1e-9)
    if ek @ m > -0.3:
      continue
    q = keep[-1, :2] + ek * ((width[lid] / 2 - (keep[-1, :2] - c) @ m) / (ek @ m))
    B = np.vstack([keep, [*q, keep[-1, 2], keep[-1, 3]]])
    A = B if which == "end" else B[::-1]
    stops[which] = [round(float(v), 3) for v in (*q, *u)]
  if stops:
    r["pts"], r["stops"] = np.round(A, 2).tolist(), stops
    stats["strip ends cut back to a stop line or crossing"] += len(stops)


def kerbed(c, n, z):
  """the surface that isn't driveable across c (along n) on this deck, round c: (offset of its middle, width), or
  None where c is on the road."""
  t = np.arange(-ISLAND_REACH, ISLAND_REACH + 1e-6, 0.05)
  code, _ = scan.T.sample(c[0] + t * n[0], c[1] + t * n[1], z, zwin=1.0)
  off = ~np.isin(code & 7, scan.DRIVEABLE)
  k = len(t) // 2
  if not off[k]:
    return None
  a, b = k, k
  while a > 0 and off[a - 1]:
    a -= 1
  while b < len(t) - 1 and off[b + 1]:
    b += 1
  return (t[a] + t[b]) / 2, t[b] - t[a] + 0.05


# a raised island in a painted median (kerbed, not driveable; the median's paint stops at its noses): between two
# stretches of one median or two-way turn lane in line, where the gap's middle isn't road, marked "island", as wide as
# the kerbed surface, so the divider runs on unbroken
cen = [r for r in out if r["kind"] in ("median", "twltl")]
for a in cen:
  for b in cen:
    if a is b or {a["left"], a["right"]} != {b["left"], b["right"]}:
      continue
    A, B = np.array(a["pts"], float), np.array(b["pts"], float)
    g = B[0, :2] - A[-1, :2]
    L = np.linalg.norm(g)
    da, db = A[-1, :2] - A[max(len(A) - 4, 0), :2], B[min(3, len(B) - 1), :2] - B[0, :2]
    if not 1.0 < L <= ISLAND_GAP or abs(B[0, 2] - A[-1, 2]) > 2 or min(da @ g / max(np.linalg.norm(da), 1e-9), db @ g / max(np.linalg.norm(db), 1e-9)) < L * math.cos(math.radians(15)):
      continue
    e = g / L
    n = np.array([e[1], -e[0]])
    pts, kerb = [], 0
    own = {a["left"], a["right"]}
    m = max(int(round(L)), 2)
    for s in np.arange(1, m) * L / m:  # (filling the gap: each sample stands for the metre round it)
      c, z = A[-1, :2] + e * s, A[-1, 2] + (B[0, 2] - A[-1, 2]) * s / L
      k = kerbed(c, n, z)
      kerb += k is not None and k[1] <= ISLAND_REACH
      # out to the median's lines where they run on along its sides (the gutter between is the island's, not a shoulder)
      side = [x[0] for x in scan.crossings(c, n, z, -ISLAND_REACH, ISLAND_REACH) if x[1] in own]
      w = A[-1, 3] + (B[0, 3] - A[-1, 3]) * s / L
      lo, hi = min([o for o in side if o < 0], default=-w / 2), max([o for o in side if o > 0], default=w / 2)
      pts.append([*np.round(c + n * (lo + hi) / 2, 2), round(float(z), 2), round(float(hi - lo), 2)])
    if len(pts) >= 3 and kerb >= 0.5 * len(pts):
      out.append({"kind": "island", "left": a["left"], "right": a["right"], "pts": pts})
      stats["island in a median"] += 1
      P = np.array(pts)
      gut = [r for r in out if r["kind"] == "shoulder" and "e" in (r["left"], r["right"]) and {r["left"], r["right"]} - {"e"} <= own
             and np.mean([np.min(np.linalg.norm(P[:, :2] - q[:2], axis=1)) < P[0, 3] for q in np.array(r["pts"])]) >= 0.5]
      out = [r for r in out if not any(r is g for g in gut)]
      stats["gutter beside an island (shoulder dropped)"] += len(gut)
json.dump(out, open(sys.argv[6], "w"))
for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
  print(v, k)
