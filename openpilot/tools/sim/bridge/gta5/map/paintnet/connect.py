"""connect.py x0 x1 y0 y1 groups.json out.json: which lane carries on into which, where the paint between them breaks.

A lane's end links to a lane starting just ahead of it (0 < gap <= JOIN m along its travel direction, same deck, within
HEAD deg) when one of their edges lines up (within EDGE m sideways): the lane carries on with a line starting or ending
beside it. A lane can have two successors (a turn bay opening beside it: its one edge lines up with each) or two
predecessors (a merge). A lane opening beside one that carries on links to it as a "branch" (see BRANCH_RUN). No link where paint crosses the gap (a stop line or crossing: lanes end at a junction's box).
Out: {"links": [[from lane index, to lane index, gap m(, "branch")]], "pts": {lane index: new pts}}: pts replace a
lane's samples where an opening was reshaped (see reshape)."""
import json, math, sys
import numpy as np
import scan

# JOIN: breaks measured at the interchange's openings (where the lane plus the opening is wider than a lane can be, so
# unsampled) 12.6, 13.9 and 16.2 m; a junction's break is never linked (paint crosses it)
JOIN, HEAD, EDGE, OVERLAP, WIDE_END = 20.0, 15.0, 0.5, 0.6, 1.2
G = json.load(open(sys.argv[5]))
lanes = G["lanes"]


def frame(A, k):
  """unit direction at end sample k (along pts = travel direction), over the 10 m beside it (a lane's centre drifts
  sideways where a bay opens beside it), and the unit normal to its right."""
  d = A[min(k + 10, len(A) - 1), :2] - A[max(k - 10, 0), :2]
  d /= max(np.linalg.norm(d), 1e-9)
  return d, np.array([d[1], -d[0]])


def crossed(p, q, z):
  """paint crossing the segment p -> q square-ish (over 60 deg to it) on this deck: a stop line or crossing."""
  d = q - p
  L = np.linalg.norm(d)
  if L < 0.5:
    return False
  # scan.crossings takes paint meeting a section at over 40 deg; the section here is the gap itself
  return len(scan.crossings((p + q) / 2, d / L, z, -L / 2, L / 2)) > 0


links = []
ok = [r["dir"] != "?" for r in lanes]
P = [np.array(r["pts"]) for r in lanes]
for i, A in enumerate(P):
  if not ok[i]:
    continue
  da, na = frame(A, len(A) - 1)
  ea = A[-1, :2]
  la, ra = ea - na * A[-1, 3] / 2, ea + na * A[-1, 3] / 2  # left and right edge points at its end
  for j, B in enumerate(P):
    if j == i or not ok[j] or abs(A[-1, 2] - B[0, 2]) > 1.5:
      continue
    db, nb = frame(B, 0)
    if da @ db < math.cos(math.radians(HEAD)):
      continue
    eb = B[0, :2]
    gap = (eb - ea) @ da
    if not 0 < gap <= JOIN:
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
# a lane opening beside another (a turn lane splitting off), whether that one carries on or ends there: its band widens
# by about the new lane's width over the BRANCH_RUN m before the new lane's own line starts (the interchange's
# lower-deck left turn: 5.2 -> 8.8 m; the top street's left-turn bay: 5.2 -> 8.9 m), and the new lane starts touching
# that band (beside it, or up to JOIN m past its end)
STOP_AHEAD = 3.0  # m ahead of a lane's end looked at for a stop line or crossing
BRANCH_RUN, BRANCH_FROM, WIDEN = 15, 10, 0.3  # widened by at least WIDEN of the new lane's width: measured 0.40, 0.69, 0.75 (only part
# of the widening is sampled where lane plus opening pass the lane step's width limit)
has_pred = {j for _, j, *_ in links}
for j, B in enumerate(P):
  if j in has_pred or not ok[j]:
    continue
  db, nb = frame(B, 0)
  for i, A in enumerate(P):
    if i == j or not ok[i] or len(A) < BRANCH_FROM:  # a fragment (in a junction's mouth) has no usual width to widen from
      continue
    k = int(np.linalg.norm(A[:, :2] - B[0, :2], axis=1).argmin())
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
  None. mid_only: A must have been running BRANCH_FROM m already and carry on past it (not lanes starting or ending
  together at a junction)."""
  k = int(np.linalg.norm(A[:, :2] - p, axis=1).argmin())
  if mid_only and not BRANCH_FROM <= k <= len(A) - 3:
    return None
  da, na = frame(A, k)
  if da @ d < math.cos(math.radians(HEAD)):
    return None
  off = p - A[k, :2]
  c = off @ na
  if abs(off @ da) > 1.5 or min(A[k, 3] / 2, c + w / 2) - max(-A[k, 3] / 2, c - w / 2) < -EDGE:
    return None
  return k


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
# a merge: a lane ending beside one that has been running a while and carries on (an on-ramp's lane tapering out)
has_succ = {i for i, *_ in links}
for i, A in enumerate(P):
  if i in has_succ or not ok[i] or len(A) < 3:
    continue
  da, _ = frame(A, len(A) - 1)
  for j, B in enumerate(P):
    if j != i and ok[j] and len(B) >= BRANCH_FROM:
      k = beside(B, A[-1, :2], da, A[-1, 3])
      # (not a lane stopping at a stop line or crossing, the lane beside running on a little further)
      if k is not None and abs(B[k, 2] - A[-1, 2]) <= 1.5 and not crossed(A[-1, :2], A[-1, :2] + da * STOP_AHEAD, A[-1, 2]):
        links.append([i, j, 0.0, "merge"])
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
  """a merging lane A's end carried on beside the lane B it merges into, narrowing until its edge line meets B's."""
  da, na = frame(A, len(A) - 1)
  k = int(np.linalg.norm(B[:, :2] - A[-1, :2], axis=1).argmin())
  s = 1.0 if (A[-1, :2] - B[k, :2]) @ frame(B, k)[1] > 0 else -1.0
  return np.vstack([A, side_taper(B, k, s, 1)])


def reshape(i, j):
  """an opening drawn as the lanes are: the lane it opens from keeps its usual width along its edge that runs on
  straight, and the widened part becomes the new lane's start, a taper growing out of its side (filled on across any
  gap to where the new lane's own lines start)."""
  A, B = P[i].copy(), P[j]
  w0 = float(np.median(A[:, 3]))
  k = int(np.linalg.norm(A[:, :2] - B[0, :2], axis=1).argmin())
  wide = A[:, 3] > w0 + OPEN_TOL
  if not wide[max(k - BRANCH_RUN, 0):k + 1].any():
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


OPEN_TOL, STEP_M = 0.3, 1.0  # m wider than its usual width where an opening starts; sample spacing
STRAIGHT_TOL = 0.1  # m wider than its usual width: the lane has started widening (its centre drifting)
OPEN_REACH = 9.0  # m: a taper's outer line looked for no further than the widest lane
EXIT_RUN = 60  # m at most an exit's taper is traced back
TAPER_RATE = 0.05  # m narrower per m at least: the interchange's exit and merge tapers close at 0.12-0.23 m/m
pts = {}
for i, j, g, *br in links:
  if br == ["merge"]:
    P[i] = merge_taper(P[i], P[j])
    pts[i] = np.round(P[i], 2).tolist()
  if br == ["branch"]:
    r = reshape(i, j)
    if r is not None:
      P[i] = r[0]  # a lane with two openings reshapes from the first's result
      pts[i] = np.round(r[0], 2).tolist()
      pts[j] = np.round(r[1], 2).tolist()
json.dump({"links": links, "pts": pts}, open(sys.argv[6], "w"))
succ, pred = {}, {}
for i, j, g, *_ in links:
  succ.setdefault(i, []).append(j)
  pred.setdefault(j, []).append(i)
split = {i for i, v in succ.items() if len(v) > 1} | {l[0] for l in links if l[3:] == ["branch"]}  # a branch's lane also carries on
merges = sum(len(v) > 1 for v in pred.values()) + sum(1 for l in links if l[3:] == ["merge"])
print(len(links), "links;", len(split), "splits;", merges, "merges")
for i, j, g, *kind in links:
  print(f" {lanes[i]['left']}|{lanes[i]['right']} -> {lanes[j]['left']}|{lanes[j]['right']}  gap {g} m", *kind)
