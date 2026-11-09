"""nodes.py x0 x1 y0 y1 groups.json dir.json out.json: lanes from GTA's path links where the paint gives none.

The paint gives no lanes on roads GTA leaves unpainted (most residential streets, service roads, the port's yards,
which the surface raster holds as pavement) or paints too worn to read (rural centre lines in scraps). There a road's
lanes come from its links (those GTA's GPS doesn't route on only where surfaced or as dirt tracks: not the river bed).
A road is a chain of vehicle links through nodes joining just two, with the same lanes
each way. Each way of it, where no paint lane, gore, median or crossing lies within COVER m of the middle of that
way's lanes for RUN_MIN m or more, gets the link's lane count that way: across the road's surface where it is found
(kerb to kerb, split evenly between all its lanes, if each gets LANE_FIT m) and otherwise as GTA paints its links
(ynd_to_osm.layout's widths and median). A stretch between two painted ones under GAP_MAX m is a painted junction's
inside or a break in the paint connect.py bridges: left alone. Lanes stop short of a node where three or more links meet
(a junction: the widest other road there, half
across, plus CLEAR m); every node where a road's lanes end and others go on is a junction seed for junctions.py.
Out: groups.json with the node lanes added to its lanes ("source": "nodes", their own groups, running along their links,
"cut": [start, end] at a junction they stop short of: connect.py doesn't carry them on across it, "dead": [start, end]
at a dead end GTA's links make too: osm.py tags it noexit)
and "node_junctions": [{"node", "x", "y", "z", "r": m to the widest road's edge there plus CLEAR}]."""
import json, math, sys, collections
import numpy as np
from scipy.spatial import cKDTree
import scan

COVER = 1.0  # m beyond a painted strip's edge that still covers a way's lanes
RUN_MIN = 15.0  # m: shorter stretches without paint are left to connect.py and junctions.py
LINK_MIN = 6.0  # m: a road between two junctions is kept down to this (osm.py's MINLEN: shorter, the junctions are one)
GAP_MAX = 40.0  # m: unpainted between two painted stretches (a junction's box is 25-35 m across; connect.py joins 20 m)
STEP = 1.0  # m between samples
SMOOTH = 6.0  # m either side the links' line is averaged over (they zigzag across lanes; lanes don't)
WIDTH_EVERY = 4  # samples between surface measurements
WIDTH_REACH = 16.0  # m along the road a surface measurement stands for
SIDE_OUT = 1.5  # m a side's measured edge may stray from its median along the road
RUN_ON = 0.3  # of a step: a lane edge running on less than this along the road doubles back
FOLD = 0.3  # an edge's bend radius over its offset (1 + offset x curvature) below this folds over
LANE_FIT = (2.4, 7.5)  # m: a lane's width when the surface is split evenly between the lanes
CLEAR = 1.0  # m
PAINT_SEED = 2.0  # m: the reach of a junction seed where paint lanes take over from node lanes
MARGIN = 30.0  # m round the box links are read from
COVERING = ("lane", "gore", "median", "twltl", "island", "crossing", "not_traffic")
# ynd_to_osm.layout: lanes as GTA paints its links (normal, narrow, one-way freeway) and the painted median per offset step
PAINTED_LANE, PAINTED_NARROW, PAINTED_FREEWAY, MEDIAN_STEP = 5.5, 4.4, 6.1, 0.9
GPS_OFF_ROAD = 0.5  # of a road GTA's GPS doesn't route (not a dirt track) that must be surfaced for its lanes
FREEWAY, NO_GPS, OFFROAD = 64, 1, 8  # node flags 2, 2, 0 (ynd_to_osm's): an off-road link without GPS is a dirt track

x0, x1, y0, y1 = scan.x0, scan.x1, scan.y0, scan.y1
G = json.load(open(sys.argv[5]))
D = json.load(open(sys.argv[6]))
stats = collections.Counter()


def layout(lf, back, freeway):
  """ynd_to_osm.layout: (lane width, m right of the link to each way's first lane on a two-way link, to its lanes'
  middle on a one-way one)."""
  steps = ((lf[1] >> 4) & 7) * (-1 if lf[1] & 128 else 1)
  w = PAINTED_FREEWAY if freeway and not back else PAINTED_NARROW if lf[1] & 2 else PAINTED_LANE
  return w, steps * MEDIAN_STEP / 2 if back and steps > 0 else steps / 14 * w


def spans(fwd, back, lf, freeway):
  """each way's lanes across the road as GTA paints the link: (lo, hi) m right of it for the lanes along it and
  those against it (None where it has none)."""
  w, off = layout(lf, back if fwd else 0, freeway)
  if fwd and back:
    return (off, off + fwd * w), (-off - back * w, -off)
  if fwd:
    return (off - fwd * w / 2, off + fwd * w / 2), None
  return None, (-off - back * w / 2, -off + back * w / 2)


# what the paint covers: every painted strip's samples, half widths and heights
# (a paint lane with no direction is no lane to drive: groups.py's lanes with one, and dir.json's other strips)
C = [(p[0], p[1], p[2], p[3] / 2) for r in [r for r in G["lanes"] if r["dir"] != "?"] + [r for r in D if r["kind"] != "lane" and r["kind"] in COVERING]
     for p in r["pts"]]
C = np.array(C, float).reshape(-1, 4)
tree = cKDTree(C[:, :2]) if len(C) else None
reach = float(C[:, 3].max()) + COVER if len(C) else 0.0


def covered(P, z):
  """P (n, 2) each within COVER m of a painted strip on its deck."""
  out = np.zeros(len(P), bool)
  if tree is None:
    return out
  for i, ix in enumerate(tree.query_ball_point(P, reach)):
    if ix:
      c = C[ix]
      out[i] = bool(((np.linalg.norm(c[:, :2] - P[i], axis=1) <= c[:, 3] + COVER) & (np.abs(c[:, 2] - z[i]) <= 2.5)).any())
  return out


# the vehicle links near the box, each pair of nodes once: lanes each way and flags
inbox = lambda n: x0 - MARGIN <= n["x"] <= x1 + MARGIN and y0 - MARGIN <= n["y"] <= y1 + MARGIN
E, no_gps = {}, set()
for (ka, kb), l in scan.links.items():
  if ka not in scan.nodes or kb not in scan.nodes or not scan.ok(scan.nodes[ka]) or not scan.ok(scan.nodes[kb]):
    continue
  if not (inbox(scan.nodes[ka]) or inbox(scan.nodes[kb])):
    continue
  fwd, back = (l["f"][2] >> 5) & 7, (l["f"][2] >> 2) & 7
  if fwd + back == 0:
    continue
  fa, fb = scan.nodes[ka]["f"], scan.nodes[kb]["f"]
  if (fa[2] | fb[2]) & NO_GPS and not (fa[0] | fb[0]) & OFFROAD:
    no_gps.add((ka, kb) if ka < kb else (kb, ka))
  key = (ka, kb) if ka < kb else (kb, ka)
  if key not in E:
    E[key] = (fwd, back, l["f"]) if ka < kb else (back, fwd, l["f"])
nbr = collections.defaultdict(set)
for a, b in E:
  nbr[a].add(b)
  nbr[b].add(a)


def way(a, b):
  """lanes a -> b, lanes b -> a, flags."""
  f, k, lf = E[(a, b) if a < b else (b, a)]
  return (f, k, lf) if a < b else (k, f, lf)


def freeway(a, b):
  return bool(scan.nodes[a]["f"][2] & FREEWAY and scan.nodes[b]["f"][2] & FREEWAY)


def sig(a, b):
  f, k, lf = way(a, b)
  return f, k, spans(f, k, lf, freeway(a, b))


def half_width(a, b):
  f, k, lf = way(a, b)
  return max(abs(v) for s in spans(f, k, lf, freeway(a, b)) if s for v in s)


# roads: chains of links through nodes joining just two, the same lanes each way
chains, seen = [], set()
for a, b in sorted(E):
  if (a, b) in seen:
    continue
  ch = [a, b]
  seen.add((a, b))
  for end in (1, 0):  # grow forwards, then backwards
    while True:
      n, p = (ch[-1], ch[-2]) if end else (ch[0], ch[1])
      if len(nbr[n]) != 2:
        break
      q = next(iter(nbr[n] - {p}))
      key = (n, q) if n < q else (q, n)
      if key in seen or (sig(p, n) != sig(n, q) if end else sig(q, n) != sig(n, p)):
        break
      seen.add(key)
      if end:
        ch.append(q)
      else:
        ch.insert(0, q)
  chains.append(ch)


def smooth(P, s):
  """P averaged over SMOOTH m either side, the window shrinking to stay symmetric at the ends."""
  k = int(round(SMOOTH / STEP))
  out = P.copy()
  cs = np.vstack([np.zeros((1, P.shape[1])), np.cumsum(P, 0)])
  for i in range(len(P)):
    h = min(k, i, len(P) - 1 - i)
    out[i] = (cs[i + h + 1] - cs[i - h]) / (2 * h + 1)
  return out


seeds = {}  # where roads' lanes end at a node joining others: junctions.py's node junctions
out, group = [], max((r["group"] for r in G["lanes"]), default=-1) + 1
for ci, ch in enumerate(chains):
  N = np.array([[scan.nodes[n]["x"], scan.nodes[n]["y"], scan.nodes[n]["z"]] for n in ch], float)
  seg = np.linalg.norm(np.diff(N[:, :2], axis=0), axis=1)
  L = float(seg.sum())
  if L < RUN_MIN:
    continue
  cum = np.r_[0, np.cumsum(seg)]
  s = np.arange(0, L + 1e-6, STEP)
  P = np.stack([np.interp(s, cum, N[:, i]) for i in range(3)], 1)
  P[:, :2] = smooth(P[:, :2], s)
  d = np.gradient(P[:, :2], axis=0)
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  nrm = np.stack([d[:, 1], -d[:, 0]], 1)  # right of the chain's way
  fwd, back, sp = sig(ch[0], ch[1])
  nl = fwd + back
  # the surface across the road: its edges where found (the deck's height read at the middle)
  code, h = scan.T.sample(P[:, 0], P[:, 1], P[:, 2], zwin=2.0)
  on = np.isin(np.asarray(code) & 7, scan.DRIVEABLE) & np.isfinite(h)
  paved = ((np.asarray(code) & 7) > 0) & np.isfinite(h)  # (any surface: the port's yards are pavement; the river bed has none)
  if any(((a, b) if a < b else (b, a)) in no_gps for a, b in zip(ch, ch[1:])) and paved.mean() < GPS_OFF_ROAD:
    stats["roads GTA's GPS doesn't route, with no road surface (river bed, runways)"] += 1
    continue
  P[on, 2] = h[on]
  meas = []
  for i in range(0, len(P), WIDTH_EVERY):
    if not on[i] or not (x0 <= P[i, 0] <= x1 and y0 <= P[i, 1] <= y1):
      continue
    lw, le = scan.extent(P[i, :2], nrm[i], P[i, 2], -1)
    rw, re_ = scan.extent(P[i, :2], nrm[i], P[i, 2], 1)
    if "far" not in (le, re_) and LANE_FIT[0] * nl <= lw + rw <= LANE_FIT[1] * nl:
      meas.append((s[i], -lw, rw))
  M = np.array(meas, float).reshape(-1, 3)
  if len(M):  # a side opening onto a driveway or alley reads far wider than the road: off the road's own width, out
    M = M[(np.abs(M[:, 1] - np.median(M[:, 1])) <= SIDE_OUT) & (np.abs(M[:, 2] - np.median(M[:, 2])) <= SIDE_OUT)]
  lo_g = min(v for q in sp if q for v in q)
  hi_g = max(v for q in sp if q for v in q)
  lo, hi, surf = np.full(len(s), lo_g), np.full(len(s), hi_g), np.zeros(len(s), bool)
  for i in range(len(s)):
    k = np.abs(M[:, 0] - s[i]) <= WIDTH_REACH
    if k.sum() >= 2:
      lo[i], hi[i], surf[i] = np.median(M[k, 1]), np.median(M[k, 2]), True
  # a bend tighter than the road is wide folds its inner edge over: no lanes there (a cul-de-sac's turning loop)
  th = np.unwrap(np.arctan2(d[:, 1], d[:, 0]))
  kap = smooth(np.gradient(th, STEP)[:, None], s)[:, 0]  # left turns positive; n points right
  flat = (1 + lo * kap >= FOLD) & (1 + hi * kap >= FOLD)
  stats["samples dropped on bends tighter than the road"] += int((~flat).sum())
  wl = (hi - lo) / nl
  # each way's lanes: (lo, hi) across the road, from the surface or the link's layout
  ways = []
  if fwd:
    flo = np.where(surf, lo + back * wl, sp[0][0])
    fhi = np.where(surf, hi, sp[0][1])
    ways.append((1, fwd, flo, fhi))
  if back:
    blo = np.where(surf, lo, sp[1][0])
    bhi = np.where(surf, lo + back * wl, sp[1][1])
    ways.append((-1, back, blo, bhi))
  # stop short of junctions at either end
  keep = (P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1) & flat
  at_junction, cut, dead = [False, False], [False, False], [False, False]
  for end, idx, nxt in ((ch[0], 0, ch[1]), (ch[-1], -1, ch[-2])):
    if len(nbr[end]) < 2:
      dead[idx] = x0 <= scan.nodes[end]["x"] <= x1 and y0 <= scan.nodes[end]["y"] <= y1  # (a dead end in GTA too)
      continue
    at_junction[idx] = True
    if len(nbr[end]) >= 3:  # clear of the other roads there, half across the widest
      cut[idx] = True
      r = max(half_width(end, q) for q in nbr[end] if q != nxt) + CLEAR
      keep &= np.linalg.norm(P[:, :2] - N[idx, :2], axis=1) > r
      stats["lane ends cut back at a junction node"] += 1
    n_ = scan.nodes[end]
    if x0 <= n_["x"] <= x1 and y0 <= n_["y"] <= y1:
      seeds[end] = {"node": f"{end[0]}:{end[1]}", "x": round(n_["x"], 2), "y": round(n_["y"], 2), "z": round(float(P[idx, 2]), 2),
                    "r": round(max(half_width(end, q) for q in nbr[end]) + CLEAR, 2)}
  for sg, n, a, b in ways:
    a, b = smooth(a[:, None], s)[:, 0], smooth(b[:, None], s)[:, 0]  # (where measured and GTA's widths meet)
    # both edges must run on along the road: where one doubles back (a kink in the links, a width changing fast), none
    run_on = np.ones(len(s), bool)
    for e in (a, b):
      Q = P[:, :2] + nrm * e[:, None]
      ok = (np.diff(Q, axis=0) * d[:-1]).sum(1) >= RUN_ON * STEP
      run_on &= np.r_[ok, True] & np.r_[True, ok]
    stats["samples dropped where a lane edge doubles back"] += int((keep & ~run_on).sum())
    mid = P[:, :2] + nrm * ((a + b) / 2)[:, None]
    cov = covered(mid, P[:, 2])
    free = keep & ~cov & run_on
    k = 0
    while k < len(s):
      if not free[k]:
        k += 1
        continue
      j = k
      while j < len(s) and free[j]:
        j += 1
      run = (j - k) * STEP
      painted = (k > 0 and cov[k - 1]) and (j < len(s) and cov[j])
      # (a road all the way from one junction to the next, however short, joins them)
      whole = at_junction[0] and at_junction[-1] and not keep[:k].any() and not keep[j:].any()
      if run < (LINK_MIN if whole else RUN_MIN) or (painted and run < GAP_MAX):
        stats["unpainted stretches left (short, or between paint)"] += 1
        k = j
        continue
      w = (b[k:j] - a[k:j]) / n
      ends = [cut[0] and not keep[:k].any(), cut[-1] and not keep[j:].any()]  # (along the chain: at a junction it was cut back from)
      dead_ = [dead[0] and not keep[:k].any(), dead[-1] and not keep[j:].any()]
      # where the paint's lanes take over (a side road meeting a painted one away from GTA's junction node, paint
      # starting on a road): a junction seed there, for the lanes either side to meet at
      for q_, side in ((k - 1, "s"), (j, "e")):
        if 0 <= q_ < len(s) and cov[q_]:
          seeds[("p", ci, sg, side)] = {"node": f"paint {ci}{side}", "x": round(float(mid[q_, 0]), 2), "y": round(float(mid[q_, 1]), 2),
                                        "z": round(float(P[q_, 2]), 2), "r": PAINT_SEED}
      ids = [f"n{ci}{'f' if sg > 0 else 'b'}:{q}" for q in range(n + 1)]
      ids[n] = "e"
      if not (fwd and back):
        ids[0] = "e"
      else:
        ids[0] = f"n{ci}:c"
      for q in range(n):  # q = 0 beside the other way (or the left edge), n - 1 at the kerb
        off = a[k:j] + (q + 0.5) * w if sg > 0 else b[k:j] - (q + 0.5) * w
        A = np.column_stack([P[k:j, :2] + nrm[k:j] * off[:, None], P[k:j, 2], w])
        if sg < 0:
          A = A[::-1]
        out.append({"kind": "lane", "left": ids[q], "right": ids[q + 1], "left_all": [ids[q]], "right_all": [ids[q + 1]],
                    "pts": np.round(A, 2).tolist(), "hatch": [0.0] * len(A), "sense": "+", "yellow_votes": 0, "arrow_votes": 0,
                    "link_votes": len(A), "group": group, "dir": "pts", "source": "nodes",
                    "surface": round(float(surf[k:j].mean()), 2), "cut": ends if sg > 0 else ends[::-1],
                    "dead": dead_ if sg > 0 else dead_[::-1]})
        stats["node lanes"] += 1
        stats["node lane m"] += len(A) * STEP
      group += 1
      k = j

G["lanes"] += out
G["node_junctions"] = list(seeds.values())
json.dump(G, open(sys.argv[7], "w"))
print(len(E), "links in", len(chains), "roads;", stats["node lanes"], "node lanes,", round(stats["node lane m"] / 1000, 2), "km")
for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
  print(" ", round(v), k)
