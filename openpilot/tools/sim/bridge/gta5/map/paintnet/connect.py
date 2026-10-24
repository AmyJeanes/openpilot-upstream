"""connect.py x0 x1 y0 y1 groups.json out.json [dir.json]: which lane carries on into which, where the paint between them breaks.

A lane's end links to a lane starting just ahead of it (0 < gap <= JOIN m along its travel direction, or up to SEAM m
before its end where the two are one lane side by side: cut back to it), same deck, within
HEAD deg) when one of their edges lines up (within EDGE m sideways): the lane carries on with a line starting or ending
beside it. A lane can have two successors (a turn bay opening beside it: its one edge lines up with each) or two
predecessors (a merge). A lane opening beside one that carries on links to it as a "branch" (see BRANCH_RUN). No link where paint crosses the gap (a stop line or crossing: lanes end at a junction's box).
Out: {"links": [[from lane index, to lane index, gap m(, "branch"|"merge")]], "pts": {lane index: new pts}, "stops": {lane
index: {"end"|"start": [x, y, ux, uy]}}}: pts replace a lane's samples where an opening was reshaped (see reshape) or
an end runs on to its stop line; stops are where a lane meets its stop line or crossing and that line's way, to cut
its end along."""
import json, math, sys, collections
import numpy as np
from scipy import ndimage as ndi
import scan

# JOIN: breaks measured at the interchange's openings (where the lane plus the opening is wider than a lane can be, so
# unsampled) 12.6, 13.9 and 16.2 m; a junction's break is never linked (paint crosses it)
JOIN, HEAD, EDGE, OVERLAP, WIDE_END = 20.0, 15.0, 0.5, 0.6, 1.2
# m: the same lane carries on where the next one starts this far before it ends, side by side with it (GTA's paint
# pieces overlap at a seam: 3.3 m on the ramp's deck); the earlier one is cut back to where the next starts
SEAM = 5.0
G = json.load(open(sys.argv[5]))
lanes = G["lanes"]


def frame(A, k):
  """unit direction at end sample k (along pts = travel direction), over the 10 m beside it (a lane's centre drifts
  sideways where a bay opens beside it), and the unit normal to its right."""
  d = A[min(k + 10, len(A) - 1), :2] - A[max(k - 10, 0), :2]
  d /= max(np.linalg.norm(d), 1e-9)
  return d, np.array([d[1], -d[0]])


def abreast(A, p):
  """A's sample level with p along A: the nearest one, moved on to the one p is least ahead of or behind (where A's
  centre drifts sideways as it widens, the nearest sample to a point beside it can be metres along)."""
  k = int(np.linalg.norm(A[:, :2] - p, axis=1).argmin())
  near = range(max(k - 5, 0), min(k + 6, len(A)))
  return min(near, key=lambda q: abs((p - A[q, :2]) @ frame(A, q)[0]))


def crossed(p, q, z):
  """paint crossing the segment p -> q square-ish (over 60 deg to it) on this deck: a stop line or crossing."""
  d = q - p
  L = np.linalg.norm(d)
  if L < 0.5:
    return False
  # scan.crossings takes paint meeting a section at over 40 deg; the section here is the gap itself
  return len(scan.crossings((p + q) / 2, d / L, z, -L / 2, L / 2)) > 0


# a break longer than JOIN (round a bend too) where the road is wider than a lane can be, so unsampled (a lane either
# side of it at lanes.py's widest, LANE_MAX: two lanes narrowing into one, the divide unpainted), and a line
# one side of both lanes runs on unbroken from the one's edge to the other's, links too (a ramp's two lanes merging
# over 31 m round a 20 deg bend, climbing 2.7 m: its height is the line's, not the lane's grade carried on)
LINE_JOIN = 60.0  # m along the line at most (EXIT_RUN: the longest taper traced)
LINE_TURN = 45.0  # deg the road turns across it at most (junctions.py's TURN_ANG: more is a junction's turn)
LINE_NEAR = 1.0  # m from a lane's edge to its line (EDGE_NEAR)
LANE_MAX, WIDE_TOL = 9.0, 0.5  # m: lanes.py's widest lane between two lines, and how near it a lane cut by it ends
line_pts = {r["id"]: np.array(r["pts"], float) for r in scan.src.paint_near(scan.x0 - 60, scan.x1 + 60, scan.y0 - 60, scan.y1 + 60)}


def on_line(L, p):
  """p's distance from polyline L (x, y, z), how far along L it lies (m), and L's height there."""
  a, b = L[:-1, :2], L[1:, :2]
  v = b - a
  t = np.clip(((p - a) * v).sum(1) / np.maximum((v * v).sum(1), 1e-12), 0, 1)
  d = np.linalg.norm(a + t[:, None] * v - p, axis=1)
  k = int(d.argmin())
  return float(d[k]), float(np.r_[0, np.cumsum(np.linalg.norm(v, axis=1))][k] + t[k] * np.linalg.norm(v[k])), float(L[k, 2] + t[k] * (L[k + 1, 2] - L[k, 2]))


def along(i, j, A, B, da, na, nb):
  """the break from A's end to B's start along a line on the same side of both, unbroken, on their decks: (m along it,
  the line, its side of the lanes (+1 right), where A's and B's edges lie along it), or None."""
  for key, sg in (("left_all", -1), ("right_all", 1)):
    for lid in set(lanes[i][key]) & set(lanes[j][key]):
      L = line_pts.get(lid)
      if L is None or len(L) < 2:
        continue
      (d0, s0, z0), (d1, s1, z1) = on_line(L, A[-1, :2] + sg * na * A[-1, 3] / 2), on_line(L, B[0, :2] + sg * nb * B[0, 3] / 2)
      k = min(int(np.linalg.norm(L[:, :2] - A[-1, :2], axis=1).argmin()), len(L) - 2)
      way = 1 if (L[k + 1, :2] - L[k, :2]) @ da > 0 else -1  # (the line stored either way along)
      if d0 <= LINE_NEAR and d1 <= LINE_NEAR and abs(z0 - A[-1, 2]) < 1.0 and abs(z1 - B[0, 2]) < 1.0 and 0 < (s1 - s0) * way <= LINE_JOIN:
        return (s1 - s0) * way, lid, sg, s0, s1
  return None


links, along_links = [], []
ok = [r["dir"] != "?" for r in lanes]
P = [np.array(r["pts"]) for r in lanes]
# across a level crossing (scan.past_gap) a lane carries on into the lane beyond, whatever paint the gap holds (the
# crossing's stop line) and wherever its lines go: the lanes ending at it and those starting beyond, each in order
# across the road, pair off from the kerb (their ends' headings wobble too much to read offsets across 20 m by); a lane
# whose way to it runs over another lane isn't at it
rail = collections.defaultdict(set)  # lane ending at a crossing -> lanes starting beyond it
for i, A in enumerate(P):
  if not ok[i]:
    continue
  da, na = frame(A, len(A) - 1)
  gap = scan.past_gap(A[-1, :2], da, A[-1, 2])
  if gap is None:
    continue
  near, far, da = gap
  na = np.array([da[1], -da[0]])
  run = A[-1, :2] + da * np.arange(1.0, max(near, 1.5), 1.0)[:, None]
  if any(q != i and np.any(np.linalg.norm(C[:, None, :2] - run[None], axis=2).min(0) < C[:, 3].max() / 2) for q, C in enumerate(P)
         if abs(C[0, 2] - A[-1, 2]) < 1.5 and np.linalg.norm(C[:, :2] - A[-1, :2], axis=1).min() < far):
    continue
  for j, B in enumerate(P):
    v = B[0, :2] - A[-1, :2]
    if j != i and ok[j] and far - 1.0 <= v @ da <= far + JOIN / 2 and abs(v @ na) <= A[-1, 3] + B[0, 3]        and da @ frame(B, 0)[0] >= math.cos(math.radians(HEAD)) and abs(A[-1, 2] - B[0, 2]) < 1.5:
      rail[i].add(j)
used_i, used_j = set(), set()
while rail:  # (each crossing's lanes: those reached from one another through shared successors)
  grp_i, grp_j = {next(iter(rail))}, set()
  while True:
    grp_j |= set().union(*(rail[i] for i in grp_i))
    more = {i for i in rail if rail[i] & grp_j} - grp_i
    if not more:
      break
    grp_i |= more
  m = np.sum([frame(P[i], len(P[i]) - 1)[0] for i in grp_i] + [frame(P[j], 0)[0] for j in grp_j], 0)
  n = np.array([m[1], -m[0]]) / max(np.linalg.norm(m), 1e-9)  # right of travel
  for i, j in zip(sorted(grp_i, key=lambda i: -P[i][-1, :2] @ n), sorted(grp_j, key=lambda j: -P[j][0, :2] @ n)):
    links.append([i, j, round(float(np.linalg.norm(P[j][0, :2] - P[i][-1, :2])), 2)])
    used_i.add(i); used_j.add(j)
  for i in grp_i:
    del rail[i]
for i, A in enumerate(P):
  if not ok[i]:
    continue
  da, na = frame(A, len(A) - 1)
  ea = A[-1, :2]
  la, ra = ea - na * A[-1, 3] / 2, ea + na * A[-1, 3] / 2  # left and right edge points at its end
  # its grade over its last 10 m, carried across the break (a ramp's deck climbing 12% rises 2 m over a 17 m break)
  k0 = max(len(A) - 11, 0)
  grade = (A[-1, 2] - A[k0, 2]) / max(np.linalg.norm(A[-1, :2] - A[k0, :2]), 1e-9)
  for j, B in enumerate(P):
    if j == i or not ok[j] or i in used_i or j in used_j:
      continue
    eb = B[0, :2]
    gap = (eb - ea) @ da
    db, nb = frame(B, 0)
    if gap > JOIN and da @ db >= math.cos(math.radians(LINE_TURN)) and gap <= LINE_JOIN:
      mid = (ea + eb) / 2
      run = along(i, j, A, B, da, na, nb)
      if run is not None and max(A[-1, 3], B[0, 3]) >= LANE_MAX - WIDE_TOL and not crossed(ea, eb, A[-1, 2]) and not any(
          q not in (i, j) and np.any((np.linalg.norm(C[:, :2] - mid, axis=1) < C[:, 3] / 2) & (np.abs(C[:, 2] - A[-1, 2]) < 2.0)) for q, C in enumerate(P)):
        along_links.append((i, j, *run))
      continue
    if abs(A[-1, 2] + grade * max(gap, 0.0) - B[0, 2]) > 1.5:
      continue
    if da @ db < math.cos(math.radians(HEAD)):
      continue
    if not -SEAM <= gap <= JOIN or abs(gap) < 1e-9:
      continue
    lb, rb = eb - nb * B[0, 3] / 2, eb + nb * B[0, 3] / 2
    # sideways offsets of the successor's edges from the predecessor's, square to the successor's direction (the
    # predecessor's end tilts where a bay opens beside it: 11 deg at the interchange, 2 m sideways over a 13 m gap)
    off_l, off_r = (lb - la) @ nb, (rb - ra) @ nb
    # the same lane carrying on: its two ends overlap mostly, side to side (a line curving and a kerb stepping across
    # a break under a bridge: 1.1 and 1.5 m off at both edges, 75% overlap)
    lap = min(0.0, off_l) - max(0.0, off_r) + min(A[-1, 3], B[0, 3])
    # (not a lane ending widened, where a bay opens beside the lane carrying on: 1.36-1.7x its usual width here)
    same = lap >= OVERLAP * min(A[-1, 3], B[0, 3]) and A[-1, 3] <= WIDE_END * np.median(A[:, 3])
    if gap < 0 and not same:
      continue
    if not same and min(abs(off_l), abs(off_r)) > EDGE:
      continue
    # the successor must lie within the predecessor's span (not beside it) for the edge that doesn't line up
    if not same and abs(off_l) > EDGE and not -EDGE <= off_l <= A[-1, 3] + EDGE:
      continue
    if not same and abs(off_r) > EDGE and not -A[-1, 3] - EDGE <= off_r <= EDGE:
      continue
    if crossed(ea, eb, A[-1, 2]):
      continue
    links.append([i, j, round(float(gap), 2)])
# of lanes carrying on into one along lines, the one about as wide as it does (the others merge into that one, below);
# its break filled along its line, so a lane merging has its samples to taper beside
filled = []
for j in sorted({l[1] for l in along_links}):
  i, _, run, lid, sg, s0, s1 = min((l for l in along_links if l[1] == j), key=lambda l: abs(P[l[0]][-1, 3] - P[j][0, 3]))
  L, A, B = line_pts[lid], P[i], P[j]
  if i in filled:  # (splitting into two: filled towards the first)
    links.append([i, j, round(float(run), 2)])
    continue
  arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(L[:, :2], axis=0), axis=1))]
  sq = np.arange(1, int(abs(s1 - s0))) * np.sign(s1 - s0) + s0
  E = np.stack([np.interp(sq, arc, L[:, c]) for c in range(3)], 1)  # the line's points along the break
  d = np.gradient(E[:, :2], axis=0) if len(E) > 1 else np.zeros((len(E), 2))
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  w = A[-1, 3] + (B[0, 3] - A[-1, 3]) * (sq - s0) / (s1 - s0)
  F = np.column_stack([E[:, :2] - sg * np.stack([d[:, 1], -d[:, 0]], 1) * w[:, None] / 2, E[:, 2], w])
  P[i] = np.vstack([A, F])
  filled.append(i)
  links.append([i, j, round(float(np.linalg.norm(B[0, :2] - P[i][-1, :2])), 2)])
seam_cut = []  # lanes cut back at a seam to where the lane they carry on into starts
for i, j, gap in links:
  if gap < 0:
    A, d = P[i], frame(P[i], len(P[i]) - 1)[0]
    keep = (A[:, :2] - P[j][0, :2]) @ d < 0
    if keep.sum() >= 2:
      P[i] = A[keep]
      seam_cut.append(i)
# a lane opening beside another (a turn lane splitting off), whether that one carries on or ends there: its band widens
# by about the new lane's width over the BRANCH_RUN m before the new lane's own line starts (the interchange's
# lower-deck left turn: 5.2 -> 8.8 m; the top street's left-turn bay: 5.2 -> 8.9 m), and the new lane starts touching
# that band (beside it, or up to JOIN m past its end)
STOP_AHEAD = 3.0  # m ahead of a lane's end looked at for a stop line or crossing
BRANCH_RUN, BRANCH_FROM, WIDEN = 15, 10, 0.3  # widened by at least WIDEN of the new lane's width: measured 0.40, 0.69, 0.75 (only part
# of the widening is sampled where lane plus opening pass the lane step's width limit)
BOX_CUT = 3.0  # m inside the box's edge a lane starting is cut by it (lanes run on a metre or so past it)
OPEN_TOL, STEP_M = 0.3, 1.0  # m wider than its usual width where an opening starts; sample spacing
# a lane splitting in two where a line starts inside it, widened on one side before that (a turn bay opening across
# the lane's edge, a gore's nose between the bay and the through lane): the successor on the side that ran on straight
# carries the lane on, the other opens from it as a branch
succ0 = {}
for l in links:
  succ0.setdefault(l[0], []).append(l)
for i, ls in succ0.items():
  A = P[i]
  if len(ls) != 2 or len(A) < BRANCH_FROM:
    continue
  if A[-BRANCH_RUN:, 3].max() - np.median(A[:, 3]) < WIDEN * min(P[l[1]][0, 3] for l in ls):
    continue
  k0 = max(len(A) - 1 - BRANCH_RUN, 0)
  _, n = frame(A, k0)
  # each edge's move sideways over the widening, square to the lane before it
  drift = {s: abs((A[-1, :2] + s * n * A[-1, 3] / 2 - A[k0, :2] - s * n * A[k0, 3] / 2) @ n) for s in (1, -1)}
  s = max(drift, key=drift.get)
  side = [np.sign((P[l[1]][0, :2] - A[-1, :2]) @ n) for l in ls]
  print(f" split {lanes[i]['left']}|{lanes[i]['right']}: edges moved {drift[1]:.2f} m right, {drift[-1]:.2f} m left")
  if side[0] != side[1] and drift[s] - drift[-s] > OPEN_TOL:
    ls[side.index(s)].append("branch")
has_pred = {j for _, j, *_ in links}
for j, B in enumerate(P):
  if j in has_pred or not ok[j] or not scan.inbox(B[0, 0], B[0, 1], -BOX_CUT):
    continue  # (a lane starting at the box's edge is cut there, not opening)
  db, nb = frame(B, 0)
  for i, A in enumerate(P):
    if i == j or not ok[i] or len(A) < BRANCH_FROM:  # a fragment (in a junction's mouth) has no usual width to widen from
      continue
    k = abreast(A, B[0, :2])
    at_end = k == len(A) - 1
    if k == 0 or abs(A[k, 2] - B[0, 2]) > 1.5:
      continue
    da, na = frame(A, k)
    if da @ db < math.cos(math.radians(HEAD)):
      continue
    off = B[0, :2] - A[k, :2]
    along = off @ da
    if not (abs(along) <= 1.5 or at_end and 0 < along <= JOIN):
      continue
    # sideways square to the new lane (the widening band's own direction tilts): the two bands touch or overlap
    c = off @ nb
    if min(A[k, 3] / 2, c + B[0, 3] / 2) - max(-A[k, 3] / 2, c - B[0, 3] / 2) < -EDGE:
      continue
    if A[max(k - BRANCH_RUN, 0):k + 1, 3].max() - np.median(A[:, 3]) < WIDEN * B[0, 3]:
      continue
    if at_end and crossed(A[k, :2], B[0, :2], A[k, 2]):
      continue
    links.append([i, j, round(float(max(along, 0)), 2), "branch"])
    break


def beside(A, p, d, w, mid_only=True):
  """where on lane A a lane whose end is at p (direction d, width w) touches it running alongside: A's sample index, or
  None. mid_only: A must have been running BRANCH_FROM m already and carry on as far past it (not lanes starting or
  ending together at a junction, a turn bay a couple of metres short of its neighbour at the stop line)."""
  k = abreast(A, p)  # (level with p: the nearest sample can be a lane piece's start a step sideways, metres along)
  if mid_only and not BRANCH_FROM <= k <= len(A) - BRANCH_FROM:
    return None
  da, na = frame(A, k)
  if da @ d < math.cos(math.radians(HEAD)):
    return None
  off = p - A[k, :2]
  c = off @ na
  if abs(off @ da) > 1.5 or min(A[k, 3] / 2, c + w / 2) - max(-A[k, 3] / 2, c - w / 2) < -EDGE:
    return None
  return k


# a lane carrying on across a pedestrian crossing with no stop line at it (a free turn lane round a traffic island): past
# the crossing its road has no lines until the lane it leads into starts, turned up to CROSS_TURN deg, wide where the
# turn lane's road joins another (a junction's exit lanes start at their usual width: the junction's own work); it
# merges into that lane where a smooth path from its end to that lane's start stays on the road and crosses no line
# but the crossing's own
CROSS_JOIN, CROSS_TURN = 25.0, 60.0
EDGE_NEAR = 1.0  # m from a lane's edge to the line along it (lanes.py's: a lane's sides are its lines' centres)
# lane lines are 0.10-0.15 m wide, stop lines and crossings' lines 0.43-0.52; the interchange's lane ends stop
# 0.1-3.1 m short of theirs
STOP_REACH, STOP_WIDE, STOP_BACK, STEP = 4.0, 0.3, 8, 1.0  # (STOP_BACK: samples; STEP: lanes.py's sample step)
width = {r["id"]: r["width"] for r in scan.src.paint()}
strips = [r for r in json.load(open(sys.argv[7])) if r["kind"] == "crossing"] if len(sys.argv) > 7 else []


def path(A, B):
  """a smooth path (cubic Hermite) from A's end to B's start along their directions there: [x, y, z, w] each metre."""
  p, q = A[-1, :2], B[0, :2]
  dp = A[-1, :2] - A[max(len(A) - 4, 0), :2]
  dq = B[min(3, len(B) - 1), :2] - B[0, :2]
  dp, dq = dp / max(np.linalg.norm(dp), 1e-9), dq / max(np.linalg.norm(dq), 1e-9)
  L = np.linalg.norm(q - p)
  t = np.linspace(0, 1, max(int(round(L)), 1) + 1)[:, None]
  xy = (2 * t ** 3 - 3 * t ** 2 + 1) * p + (t ** 3 - 2 * t ** 2 + t) * L * dp + (3 * t ** 2 - 2 * t ** 3) * q + (t ** 3 - t ** 2) * L * dq
  return np.hstack([xy, A[-1, 2:] + (B[0, 2:] - A[-1, 2:]) * t])


def back(B, s):
  """samples running lane B back from its start at its width to the near edge of the first wide line across it (a
  junction's crossing or stop line) within CROSS_JOIN m, in B's order, where a line runs along its edge on side s (+1 =
  right of travel) most of the way (within EDGE_NEAR m; lines are stored in pieces: not necessarily its own); none
  where none does (the lane really starts there)."""
  d, n = frame(B, 0)
  p = B[0]
  xs = [x for x in scan.crossings(p[:2], -d, p[2], 0.0, CROSS_JOIN) if x[0] > 0 and width.get(x[1], 0) >= STOP_WIDE]
  if not xs:
    return np.zeros((0, 4))
  edge = xs[0][0] - width[xs[0][1]] / 2 - STEP / 2  # (a sample stands for the metre round it)
  F = np.array([[*(p[:2] - d * v), p[2], p[3]] for v in np.r_[np.arange(STEP, edge, STEP), edge] if v > 0]).reshape(-1, 4)
  along = [any(abs(x[0] - s * f[3] / 2) <= EDGE_NEAR for x in scan.crossings(f[:2], n, f[2], -f[3] / 2 - EDGE_NEAR, f[3] / 2 + EDGE_NEAR)) for f in F]
  return F[::-1] if len(F) and np.mean(along) >= 0.5 else np.zeros((0, 4))


def cuts(p, n, line):
  """where the section through p along n crosses polyline line (Nx2): offsets along n."""
  out = []
  for a, b in zip(line[:-1], line[1:]):
    e = b - a
    den = n[0] * -e[1] + e[0] * n[1]
    if abs(den) < 1e-9:
      continue
    r = a - p
    t, u = (r[0] * -e[1] + e[0] * r[1]) / den, (n[0] * r[1] - n[1] * r[0]) / den
    if 0 <= u <= 1:
      out.append(t)
  return out


def kerb_fit(C, w0, w1, guide=None):
  """a lane's samples across unpainted road (from a lane end w0 wide to a lane start w1 wide) fitted to it: each side
  follows the nearest lane line or the road's edge (kerb, less a gutter) on it while that carries on from the side's
  last position (a line guiding a turn lane past where an island ends; not the next road's lines where it opens out),
  and runs smoothly on to the next lane's edge where neither does. guide (Nx2): an edge followed like a line (the side
  of the lane a turn lane merges into)."""
  d = np.gradient(C[:, :2], axis=0) if len(C) > 1 else np.array([[1.0, 0.0]])
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  N = np.stack([d[:, 1], -d[:, 0]], 1)
  side = np.zeros((len(C), 2))
  for k, s in enumerate((-1, 1)):
    kerb = np.array([scan.extent(C[q, :2], N[q], C[q, 2], s)[0] for q in range(len(C))])
    road = np.array([scan.extent(C[q, :2], N[q], C[q, 2], s, scan.ROAD)[0] for q in range(len(C))])
    # a lane stops a gutter's width short of a kerb with a gutter along at least half of it (lanes.py's GUTTER_SHARE:
    # the raster holds the gutter in pieces)
    g = (kerb - road)[kerb < scan.REACH]  # (where there is a kerb on this side at all)
    gw = float(np.median(g[g > 0])) if len(g) and (g > 0).sum() >= 0.5 * len(g) else 0.0
    v, prev = np.full(len(C), np.nan), w0 / 2
    for q in range(len(C)):
      h = C[q, 3] / 2
      xs = [s * x[0] for x in scan.crossings(C[q, :2], N[q], C[q, 2], -h - KERB_REACH, h + KERB_REACH) if s * x[0] > OPEN_TOL]
      if guide is not None:
        xs += [s * t for t in cuts(C[q, :2], N[q], guide) if OPEN_TOL < s * t <= h + KERB_REACH]
      e = min(xs + [kerb[q] - gw])
      if abs(e - prev) <= KERB_STEP:
        v[q] = prev = e
    t = np.arange(-1, len(C) + 1)
    known = np.r_[True, ~np.isnan(v), True]
    side[:, k] = np.interp(t, t[known], np.r_[w0 / 2, v, w1 / 2][known])[1:-1]
    side[:, k] = np.minimum(side[:, k], kerb - gw)
  side = ndi.median_filter(side, size=(5, 1), mode="nearest")
  out = C.copy()
  out[:, :2] = C[:, :2] + N * (side[:, 1:] - side[:, :1]) / 2
  out[:, 3] = side.sum(1)
  return out


pts = {i: np.round(P[i], 2).tolist() for i in seam_cut + filled}  # lane index -> reshaped samples
KERB_REACH = 3.0  # m: a side of a turn lane's unpainted stretch moves out or in to a line or the road's edge within this
KERB_STEP = 0.8  # m a side moves at most from one metre's sample to the next and still follows the same line or kerb
turn_fill = {}  # lane index -> samples carrying it on to the lane it turns into
has_pred, has_succ = {j for _, j, *_ in links}, {i for i, *_ in links}
for i, A in enumerate(P):
  if i in has_succ or not ok[i] or len(A) < 3:
    continue
  da, _ = frame(A, len(A) - 1)
  strip = next((r for r in strips if min(np.linalg.norm(np.array(r["pts"])[:, :2] - (A[-1, :2] + da * STOP_AHEAD), axis=1))
                < np.median(np.array(r["pts"])[:, 3]) / 2 + STOP_AHEAD and abs(r["pts"][0][2] - A[-1, 2]) < 1.5), None)
  if strip is None:
    continue
  own = {strip["left"], strip["right"]}
  if any(x[1] not in own for x in scan.crossings(A[-1, :2], da, A[-1, 2], -1.0, STOP_AHEAD)):  # a stop line: a junction
    continue
  best = None
  for j, B in enumerate(P):
    if j in has_pred or j == i or not ok[j] or len(B) < 3 or abs(B[0, 2] - A[-1, 2]) > 1.5:
      continue
    if B[0, 3] <= WIDE_END * np.median(B[:, 3]):
      continue
    gap = (B[0, :2] - A[-1, :2]) @ da
    if not 0 < gap <= CROSS_JOIN or frame(B, 0)[0] @ da < math.cos(math.radians(CROSS_TURN)):
      continue
    C = path(A, B)
    code, _ = scan.T.sample(C[:, 0], C[:, 1], C[:, 2], zwin=1.0)
    if not np.isin(code & 7, scan.DRIVEABLE).all():
      continue
    d = np.diff(C[:, :2], axis=0)
    ln = np.linalg.norm(d, axis=1)
    if any(x[1] not in own for (p, q), dd, l, z in zip(zip(C[:-1], C[1:]), d, ln, C[:-1, 2]) if l > 0.1
           for x in scan.crossings((p[:2] + q[:2]) / 2, dd / l, z, -l / 2, l / 2)):
      continue
    if best is None or ln.sum() < best[1]:
      best = (j, ln.sum(), gap)
  if best:
    j = best[0]
    links.append([i, j, round(float(best[2]), 2), "merge"])
    has_pred.add(j)
    # the lane it joins starts wide where the turn lane merges in beside it: that lane (its road's own, through the
    # junction behind) keeps its usual width along its far side and runs back along its line there to the junction;
    # the turn lane carries on into the rest, the taper, narrowing
    B = P[j]
    _, nb = frame(B, 0)
    side_in = 1.0 if (A[-1, :2] - B[0, :2]) @ nb > 0 else -1.0
    w0 = float(np.median(B[:, 3]))
    taper, q = [], 0
    while q < len(B) and B[q, 3] > w0 + OPEN_TOL:
      _, n = frame(B, q)
      far = B[q, :2] - side_in * n * B[q, 3] / 2
      taper.append([*(far + side_in * n * (w0 + B[q, 3]) / 2), B[q, 2], B[q, 3] - w0])
      B[q, :2], B[q, 3] = far + side_in * n * w0 / 2, w0
      q += 1
    T = np.array(taper).reshape(-1, 4) if taper else B[:1]
    P[j] = B = np.vstack([back(B, -side_in), B])
    pts[j] = np.round(B, 2).tolist()
    E = B[:60]  # (its start: where the turn lane comes in)
    nE = np.stack([np.gradient(E[:, 1]), -np.gradient(E[:, 0])], 1)
    nE /= np.maximum(np.linalg.norm(nE, axis=1, keepdims=True), 1e-9)
    near = E[:, :2] + side_in * nE * E[:, 3:4] / 2  # the side it merges on: the turn lane runs along it
    turn_fill[i] = np.vstack([kerb_fit(path(A, T)[1:-1], A[-1, 3], T[0, 3], near), T[:len(taper)]])

# an exit or added lane: a lane starting beside one that has been running a while and carries on, which doesn't widen
# for it (a motorway's edge line runs on straight; the exit's taper opens outside it)
has_pred = {j for _, j, *_ in links}
for j, B in enumerate(P):
  if j in has_pred or not ok[j] or len(B) < 3:
    continue
  db, _ = frame(B, 0)
  for i, A in enumerate(P):
    if i != j and ok[i] and len(A) >= BRANCH_FROM:
      k = beside(A, B[0, :2], db, B[0, 3])
      if k is not None and abs(A[k, 2] - B[0, 2]) <= 1.5:
        links.append([i, j, 0.0, "branch"])
        break
# a merge: a lane ending beside one that has been running a while and carries on (an on-ramp's lane tapering out; the
# lane beside may carry on into the next, its lines stored in pieces)
has_succ = {i for i, *_ in links}
on_to = {}  # a lane's plain successor nearest its end
for i, j, g, *br in links:
  if not br and (i not in on_to or np.linalg.norm(P[j][0, :2] - P[i][-1, :2]) < np.linalg.norm(P[on_to[i]][0, :2] - P[i][-1, :2])):
    on_to[i] = j
for i, A in enumerate(P):
  if i in has_succ or not ok[i] or len(A) < 3:
    continue
  da, _ = frame(A, len(A) - 1)
  for j, B in enumerate(P):
    if j != i and ok[j] and len(B) >= BRANCH_FROM:
      Bx = np.vstack([B, P[on_to[j]]]) if j in on_to else B
      k = beside(Bx, A[-1, :2], da, A[-1, 3])
      # (not a lane stopping at a stop line or crossing, the lane beside running on a little further)
      if k is not None and abs(Bx[k, 2] - A[-1, 2]) <= 1.5 and not crossed(A[-1, :2], A[-1, :2] + da * STOP_AHEAD, A[-1, 2]):
        links.append([i, j if k < len(B) else on_to[j], 0.0, "merge"])
        break


def side_taper(A, k, s, step):
  """samples filling the space beside lane A on side s (+1 = right of travel), from A's sample k + step on in steps of
  step (-1: back from an exit's start, +1: on from a merging lane's end), between A's edge and the nearest line or
  the road's edge beyond, while it narrows at TAPER_RATE or more (an exit's or merge's edge line closing in on A's;
  a shoulder beside it doesn't) and is wider than OPEN_TOL. Where the edge line stops short of A's (beyond it the
  space is a step wider), the taper closes at the rate it was closing. In A's order."""
  taper, w = [], []
  q = k + step
  while 0 <= q < len(A) and abs(q - k) <= EXIT_RUN:
    d, n = frame(A, q)
    inner = A[q, :2] + s * n * A[q, 3] / 2
    z = A[q, 2]
    edge, _ = scan.extent(inner + s * n * 0.3, n, z, s)  # step off A's own edge line first
    xs = [s * o for o, *_ in scan.crossings(inner, n, z, -OPEN_REACH, OPEN_REACH) if s * o > OPEN_TOL]
    wd = min(xs + [edge + 0.3])
    if wd > OPEN_REACH:  # (open road: no taper's outer line)
      break
    if len(w) >= 3 and (w[-3] - w[-1]) / 2 < TAPER_RATE:  # not closing in: a shoulder, not a taper
      taper, w = taper[:-2], w[:-2]
      break
    if wd <= OPEN_TOL:
      break
    if w and wd > w[-1] + OPEN_TOL:  # the edge line stopped short: close at the rate it was closing
      rate = (w[-3] - w[-1]) / 2 if len(w) >= 3 else TAPER_RATE
      wd = w[-1] - rate
      while wd > OPEN_TOL and 0 <= q < len(A):
        d, n = frame(A, q)
        inner = A[q, :2] + s * n * A[q, 3] / 2
        taper.append([*(inner + s * n * wd / 2), A[q, 2], wd])
        wd -= rate
        q += step
      break
    taper.append([*(inner + s * n * wd / 2), z, wd])
    w.append(wd)
    q += step
  return np.array(taper[::step]).reshape(-1, 4)


def exit_taper(A, B, k):
  """an exit's start, A not widening for it: the space beside A narrowing back from where the exit's own lines start."""
  db, nb = frame(B, 0)
  s = 1.0 if (B[0, :2] - A[k, :2]) @ nb > 0 else -1.0
  return np.vstack([side_taper(A, k, s, -1), B])


def merge_taper(A, B):
  """a merging lane A's end carried on beside the lane B it merges into, narrowing until its edge line meets B's (only
  narrowing: beyond a taper already closed lies the gutter)."""
  da, na = frame(A, len(A) - 1)
  k = int(np.linalg.norm(B[:, :2] - A[-1, :2], axis=1).argmin())
  s = 1.0 if (A[-1, :2] - B[k, :2]) @ frame(B, k)[1] > 0 else -1.0
  F = side_taper(B, k, s, 1)
  n, w = 0, A[-1, 3]
  while n < len(F) and F[n, 3] <= w:
    w = F[n, 3]
    n += 1
  return np.vstack([A, F[:n]])


def line_taper(i, A, B):
  """a lane A (index i) merging into B across a break B was filled along a line (two lanes narrowing into one, unpainted:
  its space closes over the whole break, faster towards its end than side_taper's TAPER_RATE allows at its start): A
  carried on along its own outer line, narrowing evenly to nothing level with B's end; None if that line isn't found."""
  k = int(np.linalg.norm(B[:, :2] - A[-1, :2], axis=1).argmin())
  s = 1.0 if (A[-1, :2] - B[k, :2]) @ frame(B, k)[1] > 0 else -1.0  # A's side of B, so its outer side
  da, na = frame(A, len(A) - 1)
  dB, nB = frame(B, len(B) - 1)
  for lid in lanes[i]["right_all" if s > 0 else "left_all"]:
    L = line_pts.get(lid)
    if L is None or len(L) < 2:
      continue
    (d0, s0, _), (d1, s1, _) = on_line(L, A[-1, :2] + s * na * A[-1, 3] / 2), on_line(L, B[-1, :2] + s * nB * B[-1, 3] / 2)
    if d0 > LINE_NEAR or d1 > LINE_NEAR + B[-1, 3] or abs(s1 - s0) < 2:
      continue
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(L[:, :2], axis=0), axis=1))]
    sq = np.arange(1, int(abs(s1 - s0))) * np.sign(s1 - s0) + s0
    E = np.stack([np.interp(sq, arc, L[:, c]) for c in range(3)], 1)
    d = np.gradient(E[:, :2], axis=0) if len(E) > 1 else np.tile(da, (len(E), 1))
    d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    d *= np.sign((d @ da).mean() or 1)  # (along A's travel)
    w = A[-1, 3] * (1 - (sq - s0) / (s1 - s0))
    keep = w > OPEN_TOL
    F = np.column_stack([E[:, :2] - s * np.stack([d[:, 1], -d[:, 0]], 1) * w[:, None] / 2, E[:, 2], w])[keep]
    return np.vstack([A, F])
  return None


def reshape(i, j):
  """an opening drawn as the lanes are: the lane it opens from keeps its usual width along its edge that runs on
  straight, and the widened part becomes the new lane's start, a taper growing out of its side (filled on across any
  gap to where the new lane's own lines start)."""
  A, B = P[i].copy(), P[j]
  w0 = float(np.median(A[:, 3]))
  k = int(np.linalg.norm(A[:, :2] - B[0, :2], axis=1).argmin())
  wide = A[:, 3] > w0 + OPEN_TOL
  # (widened up to where the new lane starts: a stretch a little wider further back, to a median's edge, which the
  # lane narrows from where the new lane takes the median's space, isn't the lane widening for it)
  if not wide[max(k - BRANCH_NEAR, 0):k + 1].any():
    return A, exit_taper(A, B, k)
  # the widened stretch: the last one starting before the new lane does, to where the lane is back to its width
  lo = max(k - BRANCH_RUN, 0)  # (a stretch already wide where the look-back starts counts from there)
  a = max(q for q in range(lo, k + 1) if wide[q] and (q == lo or not wide[q - 1]))
  b = a
  while b + 1 < len(A) and wide[b + 1]:
    b += 1
  d0 = A[a, :2] - A[max(a - 10, 0), :2] if a >= 3 else A[min(b + 10, len(A) - 1), :2] - A[b, :2]
  d0 /= max(np.linalg.norm(d0), 1e-9)
  n0 = np.array([d0[1], -d0[0]])
  s = 1.0 if (B[0, :2] - A[k, :2]) @ n0 > 0 else -1.0  # the side it opens on (+1 = right of travel)
  # the edge that runs on straight read from its own line's paint (each sample's centre and width were measured square
  # to a centre drifting as the lane widens: rebuilt from them, the edge wobbles); from where the lane starts widening
  while a > 0 and A[a - 1, 3] > w0 + STRAIGHT_TOL:
    a -= 1
  stable_ids = set(lanes[i]["left_all" if s > 0 else "right_all"]) - {"e", "r"}
  edge = []
  for q in range(a, b + 1):
    c, w = A[q, :2], A[q, 3]
    xs = [c + n0 * x[0] for x in scan.crossings(c, n0, A[q, 2], -w / 2 - 1, w / 2 + 1) if x[1] in stable_ids]
    edge.append(min(xs, key=lambda p: np.linalg.norm(p - (c - s * n0 * w / 2))) if xs else c - s * n0 * w / 2)
  edge = np.array(edge)
  de = np.gradient(edge, axis=0) if len(edge) > 1 else np.array([d0])
  de /= np.maximum(np.linalg.norm(de, axis=1, keepdims=True), 1e-9)
  ne = np.stack([de[:, 1], -de[:, 0]], 1)
  # the taper's inner edge: the lane's edge at its usual width, then on to the new lane's own inner edge where it starts
  inner, guess = [], []  # (x, y, z) and the width the samples alone give
  for q in range(a, b + 1):
    c, w = A[q, :2].copy(), A[q, 3]
    stable, nq = edge[q - a], ne[q - a]
    A[q, :2], A[q, 3] = stable + s * nq * w0 / 2, w0
    if w <= w0 + OPEN_TOL:  # only starting to widen: no taper yet
      continue
    if (c - B[0, :2]) @ d0 < -0.5:  # before the new lane's own start
      inner.append([*(stable + s * nq * w0), A[q, 2]])
      guess.append(w - w0)
  db, nb = frame(B, 0)
  end = [*(B[0, :2] - s * nb * B[0, 3] / 2), B[0, 2]]
  start = inner[-1] if inner else [*(A[b, :2] + s * ne[-1] * w0 / 2), A[b, 2]]
  g = np.linalg.norm(np.subtract(end[:2], start[:2]))
  for t in np.arange(1, int(g / STEP_M)) * STEP_M / max(g, 1e-9):
    inner.append(list(np.add(start, np.subtract(end, start) * t)))
    guess.append((guess[-1] if guess else 0) * (1 - t) + B[0, 3] * t)
  # its outer edge: the nearest line beyond the inner edge on the side it opens on (the median's edge curving away,
  # a yellow edge line): the widened stretch can pass the lane step's width limit, where samples were filled in
  # straight, and the gap to the new lane's own lines has no samples at all
  taper = []
  for (x, y, z), wg in zip(inner, guess):
    p = np.array([x, y])
    xs = [s * o for o, *_ in scan.crossings(p, n0, z, -OPEN_REACH, OPEN_REACH) if s * o > OPEN_TOL]
    wd = min(xs) if xs else wg
    if wd > OPEN_TOL:
      taper.append([*(p + s * n0 * wd / 2), z, wd])
  return A, np.vstack([np.array(taper).reshape(-1, 4), B])


BRANCH_NEAR = 3  # samples before the new lane's start the lane it opens from must be widened within
STRAIGHT_TOL = 0.1  # m wider than its usual width: the lane has started widening (its centre drifting)
OPEN_REACH = 9.0  # m: a taper's outer line looked for no further than the widest lane
EXIT_RUN = 60  # m at most an exit's taper is traced back
TAPER_RATE = 0.05  # m narrower per m at least: the interchange's exit and merge tapers close at 0.12-0.23 m/m
for i, F in turn_fill.items():  # (a turn lane's stretch across its crossing and round the curve is part of it)
  if len(F):
    P[i] = np.vstack([P[i], F])
    pts[i] = np.round(P[i], 2).tolist()
for i, j, g, *br in links:
  if br == ["merge"]:
    T = line_taper(i, P[i], P[j]) if j in filled else None
    P[i] = T if T is not None else merge_taper(P[i], P[j])
    pts[i] = np.round(P[i], 2).tolist()
  if br == ["branch"]:
    r = reshape(i, j)
    if r is not None:
      P[i] = r[0]  # a lane with two openings reshapes from the first's result
      pts[i] = np.round(r[0], 2).tolist()
      pts[j] = np.round(r[1], 2).tolist()
# a lane's lines stop short of the stop line or crossing it ends at (or starts beyond): an end carrying on into nothing
# runs on at its width to the near edge of the first wide line across it within STOP_REACH m
line_xy = {r["id"]: np.array(r["pts"], float)[:, :2] for r in scan.src.paint() if scan.inbox(r["pts"][0][0], r["pts"][0][1], 60)}
stops = {}  # lane -> {"end"|"start": [x, y, ux, uy]}: where it meets its stop line or crossing, and that line's way
# (a branch leaves from beside its lane; a lane merging joins another beside it)
linked_out, linked_in = {l[0] for l in links if l[3:] != ["branch"]}, {l[1] for l in links if l[3:] != ["merge"]}
for i, A in pts.items():  # (openings reshaped above: their bays' samples are in pts only)
  P[i] = np.array(A)
stopped, trimmed = 0, 0
for i in range(len(lanes)):
  A = P[i]
  if len(A) < 5:
    continue
  for at_end in (True, False):
    if (i in linked_out) if at_end else (i in linked_in):
      continue
    # a stop line set back from the crossing, with the lane lines painted on past it to the crossing: the lane ends at
    # the stop line (the interchange: crossed 1.3-6.3 m before its end; a short wide line mid-lane is something else)
    B = A if at_end else A[::-1]
    for k in range(len(B) - 1, max(len(B) - 1 - STOP_BACK, 0), -1):
      p, q = B[k - 1, :2], B[k, :2]
      L = max(np.linalg.norm(q - p), 1e-9)
      xs = [x for x in scan.crossings(p, (q - p) / L, B[k, 2], 0.0, L) if width.get(x[1], 0) >= STOP_WIDE and 0 <= x[0] < L]
      if xs:
        cut = (k - 1) + (xs[0][0] - width[xs[0][1]] / 2 - STEP / 2) / L  # sample position of the line's near edge
        keep = int(np.floor(cut)) + 1
        if keep >= 3:
          B = np.vstack([B[:keep], B[keep - 1] + (B[keep] - B[keep - 1]) * (cut - keep + 1)]) if cut > keep - 1 + 0.05 else B[:keep]
          P[i] = A = B if at_end else B[::-1]
          pts[i] = np.round(A, 2).tolist()
          trimmed += 1
        break
    p = A[-1] if at_end else A[0]
    if not scan.inbox(p[0], p[1], -6.0):  # (cut by the pilot's edge)
      continue
    e = (A[-1, :2] - A[-5, :2]) if at_end else (A[0, :2] - A[4, :2])
    e /= max(np.linalg.norm(e), 1e-9)
    xs = [x for x in scan.crossings(p[:2], e, p[2], 0.0, STOP_REACH + 0.5) if x[0] > 0 and width.get(x[1], 0) >= STOP_WIDE]
    if not xs:
      continue
    o, lid = xs[0][0] - width[xs[0][1]] / 2, xs[0][1]
    # the line's way where it crosses: the lane's end is cut along it (lanes often meet a stop line at a slant)
    L = line_xy[lid]
    q = p[:2] + e * o
    k = int(np.argmin([np.linalg.norm(q - (L[m] + L[m + 1]) / 2) for m in range(len(L) - 1)]))
    u = (L[k + 1] - L[k]) / max(np.linalg.norm(L[k + 1] - L[k]), 1e-9)
    stops.setdefault(i, {})["end" if at_end else "start"] = [round(float(v), 3) for v in (*q, *u)]
    edge = o - STEP / 2  # (a sample stands for the metre round it)
    if edge >= STEP / 2:
      t = np.r_[np.arange(STEP, edge, STEP), edge]
      F = np.array([[*(p[:2] + e * v), p[2], p[3]] for v in t])
      P[i] = A = np.vstack([A, F]) if at_end else np.vstack([F[::-1], A])
      pts[i] = np.round(A, 2).tolist()
    stopped += 1
print(trimmed, "lane ends cut back to a stop line across them;", stopped, "run on to their stop line or crossing")
json.dump({"links": links, "pts": pts, "stops": stops}, open(sys.argv[6], "w"))
succ, pred = {}, {}
for i, j, g, *_ in links:
  succ.setdefault(i, []).append(j)
  pred.setdefault(j, []).append(i)
split = {i for i, v in succ.items() if len(v) > 1} | {l[0] for l in links if l[3:] == ["branch"]}  # a branch's lane also carries on
merges = sum(len(v) > 1 for v in pred.values()) + sum(1 for l in links if l[3:] == ["merge"])
print(len(links), "links;", len(split), "splits;", merges, "merges")
for i, j, g, *kind in links:
  print(f" {lanes[i]['left']}|{lanes[i]['right']} -> {lanes[j]['left']}|{lanes[j]['right']}  gap {g} m", *kind)
