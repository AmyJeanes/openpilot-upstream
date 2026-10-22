"""direction.py x0 x1 y0 y1 lanes.json out.json: each lane's travel direction from the paint, checked against arrows and GTA links.

Rule (US markings, which GTA follows): yellow paint is on a driver's left (centre lines, left edges), white marks lanes
going the same way and the right edge. A lane's direction is the one that puts its nearest yellow line, looking
across its own road surface (kerb to kerb) at each sample, on its left. Lanes with no yellow in reach get no direction from paint ("?").
Checks, independent of the rule: road arrows inside the lane (their heading), and GTA links on the same deck within
half a lane of its centre (one-way links' direction; two-way links can't say).
Traffic: a lane must run alongside a GTA vehicle link (any direction) on its deck within FLOW_NEAR m, within FLOW_ANG
deg, over at least half its samples; strips between a stop line and a crossing, or across a junction, run across
every link and are marked "not_traffic".
Crossings: a strip between a crossing's two border lines spans the road, so the surface ends just past both its ends
(within STRIP_END m; measured on the interchange: crossing strips 1.7-2.4 m, every lane 11 m or more at one end).
Centre spaces: yellow on both sides is the space between opposing traffic, not a lane for either way (US markings):
a painted median (solid both sides, often opening into back-to-back left-turn bays) or, with a dashed yellow on a side,
a two-way left-turn lane. Marked "median" / "twltl", given no direction.
Shoulders (a line and the road's edge less than SHOULDER m apart: measured line-to-edge widths on the interchange
fall in 1.5-3.0 m or 5 m and up, none in 3.5-4.5) are marked, not given lanes."""
import json, math, sys, collections
import numpy as np
import scan

SHOULDER = 4.0
STRIP_END = 5.0
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
every = [(A, B) for (ka, kb, A, B) in scan.L]


def flows(p, d, z):
  """a GTA link on this deck runs past p within FLOW_NEAR m, along d either way within FLOW_ANG deg."""
  for A, B in every:
    if abs((A[2] + B[2]) / 2 - z) > 3:
      continue
    ab = B[:2] - A[:2]
    L = np.linalg.norm(ab)
    if L < 1:
      continue
    t = np.clip((p - A[:2]) @ ab / L ** 2, 0, 1)
    if np.linalg.norm(A[:2] + t * ab - p) <= FLOW_NEAR and abs((ab / L) @ d) >= math.cos(math.radians(FLOW_ANG)):
      return True
  return False


for (ka, kb, A, B) in scan.L:
  l = raw.get((ka, kb)) or raw.get((kb, ka))
  f = (l["f"][2] >> 5) & 7, (l["f"][2] >> 2) & 7
  if raw.get((ka, kb)) is None:
    f = f[::-1]
  if f[0] and not f[1]:
    links.append((A, B))
  elif f[1] and not f[0]:
    links.append((B, A))


def link_votes(p, d, z, w):
  """+1 / -1 per one-way link on this deck passing within w/2 of p, running along d / against it."""
  v = 0
  for A, B in links:
    if abs((A[2] + B[2]) / 2 - z) > 3:
      continue
    ab = B[:2] - A[:2]
    L = np.linalg.norm(ab)
    if L < 1:
      continue
    t = np.clip((p - A[:2]) @ ab / L ** 2, 0, 1)
    if np.linalg.norm(A[:2] + t * ab - p) > w / 2:
      continue
    c = (ab / L) @ d
    if abs(c) > 0.9:
      v += 1 if c > 0 else -1
  return v


out, stats = [], collections.Counter()
for r in lanes:
  if r["kind"] != "lane":  # gores pass through
    out.append(r)
    continue
  A = np.array(r["pts"])
  W = np.median(A[:, 3])
  sides = [r.get("left_all", [r["left"]]), r.get("right_all", [r["right"]])]
  if all(any(paint.get(i) == "yellow" for i in ids) for ids in sides):
    kind = "twltl" if any(style.get(i) == "dashed" for ids in sides for i in ids) else "median"
    out.append({**r, "kind": kind}); stats[kind] += 1
    continue
  if "e" in (r["left"], r["right"]) and W < SHOULDER:
    out.append({**r, "kind": "shoulder"}); stats["shoulder"] += 1
    continue
  d = np.gradient(A[:, :2], axis=0)
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  ends = []
  for p, q in ((A[-1], A[-3]), (A[0], A[2])):
    e = (p[:2] - q[:2]) / max(np.linalg.norm(p[:2] - q[:2]), 1e-9)
    ends.append(scan.extent(p[:2], e, p[2], 1)[0])
  if len(A) >= 3 and max(ends) <= STRIP_END:
    out.append({**r, "kind": "crossing"}); stats["crossing"] += 1
    continue
  ks = range(0, len(A), 2)
  if np.mean([flows(A[k, :2], d[k], A[k, 2]) for k in ks]) < 0.5:
    out.append({**r, "kind": "not_traffic"}); stats["not_traffic"] += 1
    continue
  yl = 0  # +1: yellow on the left as seen along pts
  for k in range(0, len(A), 3):
    n = np.array([d[k, 1], -d[k, 0]])
    lo, _ = scan.extent(A[k, :2], n, A[k, 2], -1)  # this road surface only: not across a median or kerb
    hi, _ = scan.extent(A[k, :2], n, A[k, 2], 1)
    xs = scan.crossings(A[k, :2], n, A[k, 2], -lo - 0.3, hi + 0.3)
    ys = [o for o, lid, col, sty, z in xs if col == "yellow"]
    if ys:
      o = min(ys, key=abs)
      yl += 1 if o < 0 else -1
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
  check = lambda v: "agree" if v and (v > 0) == (sense == "+") else "disagree" if v else "none"
  stats[(sense, "arrows " + check(av), "links " + check(lv))] += 1
  out.append({**r, "sense": sense, "yellow_votes": yl, "arrow_votes": av, "link_votes": lv})
json.dump(out, open(sys.argv[6], "w"))
for k, v in sorted(stats.items(), key=lambda kv: -kv[1]):
  print(v, k)
