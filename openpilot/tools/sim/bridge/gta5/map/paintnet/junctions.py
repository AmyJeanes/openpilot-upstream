"""junctions.py x0 x1 y0 y1 groups.json links.json dir.json out.json: each junction's box, the road between where its
arms' lanes end, where lane paint stops meaning lanes.

A junction is the crossings and no-traffic strips (stop line to crossing) on one stretch of road on a deck that no other
lane, gore, median or shoulder covers (found from each strip outwards, the junction's own lanes left out of the walls as
its strips are found). Its arms' mouths are the lanes ending or starting within ATTACH m of that stretch of road, not carrying on into
another (a free turn lane round an island bypasses it): the line across a lane's end
(its stop line, or where it starts beyond the junction), so an arm with no crossing still bounds the box. The box is
the road round its strips on its deck that no other lane, gore, median or shoulder covers (they and the kerbs wall it
in: an arm's lanes end at its mouth, a slip road's lanes keep its own crossing out), within the hull of the strips and
the mouths. Its mouths must face two ways at least (lanes heading more than CROSS deg apart, either way along): a
crossing on a road with no other road meeting it is not a junction. Lanes lying mostly inside it are the junction's own (guide lines and turn paint through it, not lanes).
Each lane arriving at a mouth gets its turns from its arrows: the listed road arrow in its band, heading its way, nearest
the junction within ARROW_BACK m (along it and the lane it carries straight on from; not back across a bay's opening:
the lane it opens from has the through lane's arrows). GTA paints a left+right arrow as a left and a right decal on
one spot (6 on the map), so decals stacked within ARROW_STACK m pointing the same way are one arrow with all their
turns. An arrow found only by its shape (arrows.py: the list misses many) takes the type its paint matches (arrows.kind), else "?"; a lane with no arrow, None.
Paths across it: its mouths make arms (the ways its roads leave it, within CROSS deg); an arm's arriving lanes reach
another arm's leaving lanes by the turn between them (through within TURN_ANG deg of straight on, left or right
beyond, a U-turn past UTURN: none). Each turn's lanes pair off with the leaving lanes in order across the road, from
the inside of a left turn, from the kerb for through and right; a guide (lane paint carried through the box) running
from an arriving lane's end to a leaving lane's start pairs them and is the path's way. A lane with no arrow
(or "?") is taken as through, plus left as its arm's leftmost and right as its rightmost where no arrowed lane in its
arm turns that way and no free turn lane round the box does (a single lane: every turn); flagged "inferred". Flags:
"merge" (more lanes turning than leaving lanes: the outer ones share the last), "no exit" (an arrow's way has no
road leaving), and per junction "unreached" (leaving lanes no path or free turn lane reaches).
Out: [{"id", "z", "ring": [[x, y], ...], "strips": [dir.json index], "mouths": [[lane, "start"|"end", turns]],
"inside": [lane], "paths": [{"from", "to", "turn", "guide", "flags", "pts": [[x, y, z, w], ...]}], "unreached": [lane]}];
turns like "left", "through;right", "left;right" (OSM turn:lanes values), only at "end" mouths."""
import json, os, sys
import numpy as np
import cv2
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Point, Polygon, box as rect
from shapely.ops import unary_union
import scan
import arrows

REACH = 40.0  # m round a junction's strips its road is looked for (the interchange's junction boxes are 25-35 m across)
ATTACH = 8.0  # m from a lane's end to its junction's strips (0-1.8 m on the interchange; the nearest other ends 24 m)
CROSS = 30.0  # deg between two arms' headings (direction.py's FLOW_ANG: a lane runs along a link within it)
ARROW_ALIGN, ARROW_DZ, ARROW_SPILL = 30.0, 2.5, 0.5  # deg, m, m outside a lane: an arrow is in it (osm.py's)
ARROW_STACK = 1.5  # m between stacked decals of one arrow (twice-placed decals on the map: 0.0-0.6 m)
ARROW_BACK = 60.0  # m back from a lane's end its arrows are looked for (GTA paints them 5-40 m before a stop line)
INSIDE = 0.5  # share of a lane's samples inside a box for it to be the junction's own
EDGE_BOX = 6.0  # m: lane ends this near the pilot box's edge are cut by it, not a junction's (osm.py's)
GUIDE_MIN = 6.0  # m: shorter lanes inside a box (beside a corner's arc) guide nothing (export.py's)
RES = 0.1  # m: raster step of the cut to the road
SLIVER = 9  # raster steps: narrower slivers dropped (direction.py's OUTLINE_SLIVER)
SEAL = 0.3  # m the walls are grown by: neighbouring lanes' bands part by up to 0.3 m at a shared line (osm.py's ADJ)
TURN_ANG, UTURN = 45.0, 150.0  # deg off straight on: a turn beyond the first, a U-turn beyond the second (the interchange's: 0-6 and 69-106)
GUIDE_NEAR = 10.0  # m from a guide's end to the lane end it carries on (the interchange's: 0.5-7.3 m along, 0.5-2.0 m across)
x0, x1, y0, y1 = scan.x0, scan.x1, scan.y0, scan.y1
G = json.load(open(sys.argv[5]))
LK = json.load(open(sys.argv[6]))
for k, p in LK.get("pts", {}).items():  # openings reshaped by connect.py
  G["lanes"][int(k)]["pts"] = p
LA = [np.array(r["pts"], float) for r in G["lanes"]]
STOPS = {int(k): v for k, v in LK.get("stops", {}).items()}
D = json.load(open(sys.argv[7]))
at_edge = lambda p: min(p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1]) < EDGE_BOX


def frame(A):
  k = np.arange(len(A))
  d = A[np.minimum(k + 2, len(A) - 1), :2] - A[np.maximum(k - 2, 0), :2]
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  return d, np.stack([d[:, 1], -d[:, 0]], 1)


def meet(a, d, cut):
  """where the line through a along d meets cut ([x, y, ux, uy]: a stop line's point and way), or None (export.py's)."""
  q, u = np.array(cut[:2]), np.array(cut[2:])
  den = d[0] * -u[1] + u[0] * d[1]
  if abs(den) < 0.3:
    return None
  return a + d * (((q - a)[0] * -u[1] + u[0] * (q - a)[1]) / den)


def band(A, stop=None):
  """a run's area: its centre +- half its width, its ends cut along its stop lines (stop: {"start"|"end": cut})."""
  d, n = frame(A)
  h = A[:, 3:4] / 2
  if len(A) < 2:
    return Point(A[0, :2]).buffer(max(float(h[0, 0]), 0.3))
  left, right = A[:, :2] - n * h, A[:, :2] + n * h
  for k, which in ((0, "start"), (-1, "end")):
    cut = (stop or {}).get(which)
    ends = [meet(e[k], d[k], cut) for e in (left, right)] if cut else [None]
    if all(m is not None for m in ends) and all(np.linalg.norm(m - e[k]) < 3 for m, e in zip(ends, (left, right))):
      left[k], right[k] = ends
  return Polygon(np.vstack([left, right[::-1]])).buffer(0)


# (a no-traffic strip at the pilot's edge is a lane cut by it, not a junction's)
strips = [(k, np.array(r["pts"], float), Polygon(r["outline"]).buffer(0) if r.get("outline") else band(np.array(r["pts"], float)))
          for k, r in enumerate(D) if r["kind"] == "crossing" or r["kind"] == "not_traffic" and not (at_edge(r["pts"][0]) or at_edge(r["pts"][-1]))]
walls_dir = [(np.array(r["pts"], float), r.get("stops")) for r in D if r["kind"] in ("gore", "median", "twltl", "shoulder", "island")]


def cut(hull, pts, walls, seed):
  """the driveable surface within hull on the deck of pts (x, y, z: each place takes its nearest one's height), less
  walls (areas, grown SEAL m), the part overlapping seed most, as a ring."""
  lo, hi = np.array(hull.bounds[:2]) - 1, np.array(hull.bounds[2:]) + 1
  W, H = np.ceil((hi - lo) / RES).astype(int)
  px = lambda P: np.round(np.stack([(P[:, 0] - lo[0]) / RES - 0.5, (hi[1] - P[:, 1]) / RES - 0.5], 1)).astype(np.int32)
  mask = np.zeros((H, W), np.uint8)
  cv2.fillPoly(mask, [px(np.array(hull.exterior.coords))], 1)
  r, c = np.nonzero(mask)
  X, Y = lo[0] + (c + 0.5) * RES, hi[1] - (r + 0.5) * RES
  near = cKDTree(pts[:, :2]).query(np.stack([X, Y], 1))[1]
  code, _ = scan.T.sample(X, Y, pts[near, 2], zwin=1.0)
  road = np.zeros_like(mask)
  road[r, c] = np.isin(code & 7, scan.DRIVEABLE)
  wall = np.zeros_like(mask)
  for g in walls:
    g = g.buffer(SEAL)
    for poly in getattr(g, "geoms", [g]):
      cv2.fillPoly(wall, [px(np.array(poly.exterior.coords))], 1)
  road &= 1 - wall
  sd = np.zeros_like(mask)
  for poly in getattr(seed, "geoms", [seed]):
    cv2.fillPoly(sd, [px(np.array(poly.exterior.coords))], 1)
  n, lab = cv2.connectedComponents(road, connectivity=4)
  if n < 2:
    return None
  m = 1 + int(np.bincount(lab[(sd > 0) & (lab > 0)], minlength=n)[1:].argmax())
  # (a sliver between walls, such as where a painted median's tip stops short of the stop line, isn't the junction's)
  keep = cv2.morphologyEx((lab == m).astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SLIVER,) * 2))
  cs, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
  if not cs:
    return None
  C = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.0, True)[:, 0].astype(float)
  return [[round(float(lo[0] + (x + 0.5) * RES), 2), round(float(hi[1] - (y + 0.5) * RES), 2)] for x, y in C]


# road arrows: the listed ones, stacked decals merged, then those found only by their shape
listed = [a for a in scan.src.features() if a["kind"] == "arrow" and scan.inbox(a["x"], a["y"], 20)]
marks = []  # (x, y, z, heading deg or None, turns set or None)
for a in listed:
  h = np.radians(a["heading"])
  for m in marks:
    if m[3] is not None and np.hypot(m[0] - a["x"], m[1] - a["y"]) <= ARROW_STACK and abs(m[2] - a["z"]) < 1 and        abs((m[3] - a["heading"] + 180) % 360 - 180) < ARROW_ALIGN:
      m[4].update(a["arrow"].split(";"))
      break
  else:
    marks.append([a["x"], a["y"], a["z"], a["heading"], set(a["arrow"].split(";"))])
for x, y, z, _ in arrows.box(x0 - 20, x1 + 20, y0 - 20, y1 + 20, 0.1)[3]:
  if all(np.hypot(x - m[0], y - m[1]) > 3.0 for m in marks):
    marks.append([x, y, z, None, None])
TURN_ORDER = ["left", "through", "right"]
TMPL = arrows.templates(scan.T, scan.src.features(), os.path.expanduser("~/.cache/paintnet/arrow_templates.npz"))
pred, succ, feeds = {}, {}, {}  # feeds: what carries on or merges into a lane
for i, j, g, *br in LK["links"]:
  if not br:
    pred.setdefault(j, []).append(i)
    succ.setdefault(i, []).append(j)
  if br != ["branch"]:
    feeds.setdefault(j, []).append(i)
merging = {i for i, j, g, *br in LK["links"] if br == ["merge"]}


def turns_of(i):
  """lane i's turns where it arrives: the arrow nearest its end along it and what it carries straight on from."""
  chain, back = [LA[i]], [i]
  while len(pred.get(back[-1], [])) == 1 and pred[back[-1]][0] not in back:
    back.append(pred[back[-1]][0])
    chain.append(LA[back[-1]])
  X = np.vstack(chain[::-1])
  d, n = frame(X)
  togo = np.r_[np.cumsum(np.linalg.norm(np.diff(X[::-1, :2], axis=0), axis=1))[::-1], 0.0]  # m to the end
  best = None
  for x, y, z, h, tu in marks:
    k = int(np.argmin(np.hypot(X[:, 0] - x, X[:, 1] - y)))
    off = np.array([x, y]) - X[k, :2]
    if abs(off @ n[k]) > X[k, 3] / 2 + ARROW_SPILL or abs(off @ d[k]) > 1.0 or abs(X[k, 2] - z) > ARROW_DZ or togo[k] > ARROW_BACK:
      continue
    if h is not None and np.array([np.cos(np.radians(h)), np.sin(np.radians(h))]) @ d[k] < np.cos(np.radians(ARROW_ALIGN)):
      continue
    if best is None or togo[k] < best[0]:
      best = (togo[k], None, tu, (x, y, z), k)
  if best is None:
    return None
  if best[2] is None:  # found only by its shape: typed by its picture, along the lane
    return arrows.kind(scan.T, best[3][0], best[3][1], best[3][2], d[best[4]], TMPL) or "?"
  return ";".join(t for t in TURN_ORDER if t in best[2])


def deck_near(A, seed, z):
  """A runs on the junction's deck where it comes near it (its samples within REACH m of its strips' box: a long lane
  can climb onto another level further off, and a road on a deck over the junction is on another)."""
  b = rect(*seed.bounds).buffer(REACH, join_style=2).bounds
  m = (A[:, 0] >= b[0]) & (A[:, 0] <= b[2]) & (A[:, 1] >= b[1]) & (A[:, 1] <= b[3])
  return m.any() and abs(np.median(A[m, 2]) - z) < 2


def junction(mem):
  """the deck's height, its strips' union, and the lanes on its deck: (z, seed, on_deck, own) for strips mem."""
  z = float(np.median(np.vstack([strips[i][1] for i in mem])[:, 2]))
  seed = unary_union([strips[i][2] for i in mem])
  hull0 = seed.convex_hull
  on_deck = [i for i, A in enumerate(LA) if deck_near(A, seed, z)]
  own = {i for i in on_deck if np.mean([hull0.contains(Point(p)) for p in LA[i][:, :2]]) >= INSIDE}
  return z, seed, on_deck, own


def walls_of(z, on_deck, own):
  return [band(LA[i], STOPS.get(i)) for i in on_deck if i not in own] + [band(A, st) for A, st in walls_dir if abs(np.median(A[:, 2]) - z) < 2]


deg = lambda v: float(np.degrees(np.arctan2(v[1], v[0])))
wrap = lambda a: (a + 180) % 360 - 180
length = lambda A: float(np.linalg.norm(np.diff(A[:, :2], axis=0), axis=1).sum())


def curve(A, B, g=None):
  """samples a metre apart (x, y, z, width) from A's end to B's start: along guide g's centre, else a cubic from A's
  heading into B's."""
  if g is not None:
    C = LA[g][:, :4]
    P = np.vstack([A[-1:, :4], C if np.linalg.norm(C[0, :2] - A[-1, :2]) < np.linalg.norm(C[-1, :2] - A[-1, :2]) else C[::-1], B[:1, :4]])
  else:
    da, db = frame(A)[0][-1], frame(B)[0][0]
    c = np.linalg.norm(B[0, :2] - A[-1, :2])
    t = np.linspace(0, 1, 50)[:, None]
    P = A[-1, :4] + (B[0, :4] - A[-1, :4]) * t
    P[:, :2] = (2 * t ** 3 - 3 * t ** 2 + 1) * A[-1, :2] + (t ** 3 - 2 * t ** 2 + t) * c * da + (3 * t ** 2 - 2 * t ** 3) * B[0, :2] + (t ** 3 - t ** 2) * c * db
  s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
  q = np.linspace(0, s[-1], max(int(round(s[-1])), 1) + 1)
  return np.stack([np.interp(q, s, P[:, c]) for c in range(4)], 1)


def near_end(p, A, k, slack=0.0):
  """p lies within GUIDE_NEAR m along A's sample k and within half A's width (+ slack) across it."""
  d, n = frame(A)
  off = p - A[k, :2]
  return abs(off @ d[k]) <= GUIDE_NEAR and abs(off @ n[k]) <= A[k, 3] / 2 + slack


def paths(mouths, guides):
  """the paths across a junction from its mouths, and the leaving lanes none reaches (the module's docstring)."""
  M = []  # lane, "end"|"start", turns, the way its road leaves the box (deg)
  for i, which, tu in mouths:
    d = frame(LA[i])[0][-1 if which == "end" else 0]
    M.append((i, which, tu, deg(-d if which == "end" else d)))
  # arms: the mouths round the box by the way they leave it, split where that turns more than CROSS
  order = sorted(range(len(M)), key=lambda m: M[m][3])
  gaps = [(M[order[(k + 1) % len(order)]][3] - M[order[k]][3]) % 360 for k in range(len(order))]
  arms, cur = [], []
  s = next((k + 1 for k, g in enumerate(gaps) if g > CROSS), 0)
  for m, g in zip(order[s:] + order[:s], gaps[s:] + gaps[:s]):
    cur.append(m)
    if g > CROSS:
      arms.append(cur); cur = []
  if cur:
    arms.append(cur)
  out = [deg(np.sum([[np.cos(np.radians(M[m][3])), np.sin(np.radians(M[m][3]))] for m in a], 0)) for a in arms]

  def across(lanes, flow, which):
    """lanes in order across the road, left to right facing flow (deg)."""
    r = np.array([np.sin(np.radians(flow)), -np.cos(np.radians(flow))])
    return sorted(lanes, key=lambda i: LA[i][-1 if which == "end" else 0, :2] @ r)

  arriving = [across([M[m][0] for m in a if M[m][1] == "end"], o + 180, "end") for a, o in zip(arms, out)]
  leaving = [across([M[m][0] for m in a if M[m][1] == "start"], o, "start") for a, o in zip(arms, out)]
  turns = {M[m][0]: M[m][2] for m in range(len(M))}
  reached = set()
  bypass = set()  # (arm, turn) a free turn lane round the box takes: a lane carrying on into a leaving lane from that arm's road
  res = []
  for a in range(len(arms)):
    ways = {}
    for b in range(len(arms)):
      th = wrap(out[b] - out[a] - 180)
      t = "through" if abs(th) <= TURN_ANG else "left" if TURN_ANG < th <= UTURN else "right" if -UTURN <= th < -TURN_ANG else None
      if b != a and t and (t not in ways or abs(th - {"left": 90, "through": 0, "right": -90}[t]) < ways[t][1]):
        ways[t] = (b, abs(th - {"left": 90, "through": 0, "right": -90}[t]))
    for t, (b, _) in ways.items():
      for j in leaving[b]:
        for p in feeds.get(j, []):
          if p not in turns and abs(wrap(deg(frame(LA[p])[0][0]) - (out[a] + 180))) < CROSS:
            bypass.add((a, t)); reached.add(j)
    L = arriving[a]
    known = {i: set(turns[i].split(";")) for i in L if turns[i] and turns[i] != "?"}
    sets, guessed = {}, set()
    for k, i in enumerate(L):
      if i in known:
        sets[i] = known[i]
        continue
      guessed.add(i)
      sets[i] = {"through"} | {t for t, edge in (("left", k == 0), ("right", k == len(L) - 1))
                               if edge and (a, t) not in bypass and not any(t in v for v in known.values())}
    for t in ("left", "through", "right"):
      want = [i for i in L if t in sets[i]]
      if not want:
        continue
      if t not in ways or not leaving[ways[t][0]]:
        res += [{"from": i, "to": None, "turn": t, "guide": None, "flags": ["no exit"]} for i in want if i in known]
        continue
      T = leaving[ways[t][0]]
      fixed = {}
      for i in want:
        for g in guides:
          G0, G1 = LA[g][0, :2], LA[g][-1, :2]
          for e0, e1 in ((G0, G1), (G1, G0)):
            y = next((y for y in T if near_end(e1, LA[y], 0)), None)
            if near_end(e0, LA[i], -1) and y is not None and y not in [v[0] for v in fixed.values()]:
              fixed[i] = (y, g)
      rest = [i for i in want if i not in fixed]
      free = [y for y in T if y not in [v[0] for v in fixed.values()]] or T
      if t != "left":  # (paired from the kerb)
        rest, free = rest[::-1], free[::-1]
      pairs = list(fixed.items()) + [(i, (free[min(k, len(free) - 1)], None)) for k, i in enumerate(rest)]
      for k, (i, (y, g)) in enumerate(pairs):
        flags = (["inferred"] if i in guessed else []) + (["merge"] if [v[0] for _, v in pairs].count(y) > 1 else [])
        res.append({"from": i, "to": y, "turn": t, "guide": g, "flags": flags, "pts": curve(LA[i], LA[y], g).round(2).tolist()})
        reached.add(y)
  unreached = [j for lv in leaving for j in lv if j not in reached]
  return res, unreached, [[round(o, 1), arriving[k], leaving[k]] for k, o in enumerate(out)]


out, taken = [], set()
for s0 in range(len(strips)):
  if s0 in taken:
    continue
  mem = [s0]
  region = None
  for _ in range(4):  # strips on the same unwalled road join, their lanes stop walling it, until nothing changes
    z, seed, on_deck, own = junction(mem)
    area = rect(*seed.bounds).buffer(REACH, join_style=2)
    ring = cut(area, np.vstack([strips[i][1] for i in mem]), walls_of(z, on_deck, own), seed)
    if ring is None:
      break
    region = Polygon(ring).buffer(0)
    grown = [i for i, (_, A, p) in enumerate(strips) if (i in mem or i not in taken) and abs(np.median(A[:, 2]) - z) < 2
             and p.intersection(region).area >= 0.5 * p.area]
    if set(grown) <= set(mem):
      break
    mem = sorted(set(mem) | set(grown))
  taken |= set(mem)
  z, seed, on_deck, own = junction(mem)
  mem = [strips[i] for i in mem]
  mouths, corners, heads = [], [], []
  for i in on_deck:
    if i in own:
      continue
    A = LA[i]
    d, n = frame(A)
    for k, which in ((-1, "end"), (0, "start")):
      # (a lane leaving may also be carried on or merged into: a free turn lane round the box joining it)
      if at_edge(A[k]) or which == "end" and (succ.get(i) or i in merging) or (region if region is not None else seed).distance(Point(A[k, :2])) > ATTACH:
        continue
      mouths.append([i, which, turns_of(i) if which == "end" else None])
      heads.append(np.arctan2(d[k, 1], d[k, 0]) % np.pi)
      corners += [A[k, :2] - n[k] * A[k, 3] / 2, A[k, :2] + n[k] * A[k, 3] / 2]
  gap = lambda a, b: min(abs(a - b), np.pi - abs(a - b))
  if not any(gap(a, b) > np.radians(CROSS) for a in heads for b in heads):
    continue
  hull = unary_union([seed.convex_hull, MultiPoint(corners).convex_hull] if corners else [seed]).convex_hull
  pts = np.vstack([A for _, A, _ in mem] + [LA[i][[0, -1]] for i, _, _ in mouths])
  ring = cut(hull, pts, walls_of(z, on_deck, own), seed)
  if ring is None:
    continue
  box = Polygon(ring).buffer(0)
  inside = [i for i in on_deck if np.mean([box.contains(Point(p)) for p in LA[i][:, :2]]) >= INSIDE]
  ps, unreached, arms = paths(mouths, [i for i in inside if length(LA[i]) >= GUIDE_MIN])
  out.append({"id": len(out), "z": round(z, 2), "ring": ring, "strips": [k for k, _, _ in mem], "mouths": mouths, "inside": inside,
              "paths": ps, "unreached": unreached})
  print(f"junction {len(out) - 1}: {len(mem)} strips, {len(mouths)} mouths, {box.area:.0f} m2, lanes inside {inside}")
  for o, arr, lv in arms:
    print(f"  arm leaving {o:6.1f} deg: arriving {arr} ({', '.join(str(tu) for i in arr for j, w, tu in mouths if j == i and w == 'end')}), leaving {lv}")
  for p in ps:
    print(f"    {p['from']:3d} -> {p['to']}  {p['turn']:8s} {'guide ' + str(p['guide']) if p['guide'] is not None else ''} {' '.join(p['flags'])}")
  if unreached:
    print(f"  unreached: {unreached}")
json.dump(out, open(sys.argv[8], "w"))
print(len(out), "junctions ->", sys.argv[8])
