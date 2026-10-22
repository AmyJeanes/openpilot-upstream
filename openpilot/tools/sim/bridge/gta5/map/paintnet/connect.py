"""connect.py x0 x1 y0 y1 groups.json out.json: which lane carries on into which, where the paint between them breaks.

A lane's end links to a lane starting just ahead of it (0 < gap <= JOIN m along its travel direction, same deck, within
HEAD deg) when one of their edges lines up (within EDGE m sideways): the lane carries on with a line starting or ending
beside it. A lane can have two successors (a turn bay opening beside it: its one edge lines up with each) or two
predecessors (a merge). A lane opening beside one that carries on links to it as a "branch" (see BRANCH_RUN). No link where paint crosses the gap (a stop line or crossing: lanes end at a junction's box).
JOIN is provisional (the one opening measured on the interchange is 13 m; re-measure on a bigger area).
Out: {"links": [[from lane index, to lane index, gap m(, "branch")]], "connectors": [{"from", "to", "ring": [[x, y], ...], "z"}],
"pts": {lane index: new pts}}: a connector is the successor's width carried back to its predecessor's end, filling the
break; pts replace a lane's samples where an opening was reshaped (see reshape)."""
import json, math, sys
import numpy as np
import scan

JOIN, HEAD, EDGE = 15.0, 15.0, 0.5
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


links, conns = [], []
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
    if min(abs(off_l), abs(off_r)) > EDGE:
      continue
    # the successor must lie within the predecessor's span (not beside it) for the edge that doesn't line up
    if abs(off_l) > EDGE and not -EDGE <= off_l <= A[-1, 3] + EDGE:
      continue
    if abs(off_r) > EDGE and not -A[-1, 3] - EDGE <= off_r <= EDGE:
      continue
    if crossed(ea, eb, A[-1, 2]):
      continue
    links.append([i, j, round(float(gap), 2)])
    # the successor's edges carried back to the predecessor's end: a strip across the break
    back = gap
    ring = [lb - da * back, rb - da * back, rb, lb]
    conns.append({"from": i, "to": j, "z": round(float((A[-1, 2] + B[0, 2]) / 2), 2),
                  "ring": [[round(float(x), 2), round(float(y), 2)] for x, y in ring]})
# a lane opening beside another (a turn lane splitting off), whether that one carries on or ends there: its band widens
# by about the new lane's width over the BRANCH_RUN m before the new lane's own line starts (the interchange's
# lower-deck left turn: 5.2 -> 8.8 m; the top street's left-turn bay: 5.2 -> 8.9 m), and the new lane starts touching
# that band (beside it, or up to JOIN m past its end)
BRANCH_RUN, WIDEN = 15, 0.5
has_pred = {j for _, j, *_ in links}
for j, B in enumerate(P):
  if j in has_pred or not ok[j]:
    continue
  db, nb = frame(B, 0)
  for i, A in enumerate(P):
    if i == j or not ok[i] or len(A) < 3:
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


def reshape(i, j):
  """an opening drawn as the lanes are: the lane it opens from keeps its usual width along its edge that runs on
  straight, and the widened part becomes the new lane's start, a taper growing out of its side (filled on across any
  gap to where the new lane's own lines start)."""
  A, B = P[i].copy(), P[j]
  w0 = float(np.median(A[:, 3]))
  k = int(np.linalg.norm(A[:, :2] - B[0, :2], axis=1).argmin())
  wide = A[:, 3] > w0 + OPEN_TOL
  if not wide[max(k - BRANCH_RUN, 0):k + 1].any():
    return None
  # the widened stretch: the last one starting before the new lane does, to where the lane is back to its width
  a = max(q for q in range(max(k - BRANCH_RUN, 0), k + 1) if wide[q] and (q == 0 or not wide[q - 1]))
  b = a
  while b + 1 < len(A) and wide[b + 1]:
    b += 1
  d0 = A[a, :2] - A[max(a - 10, 0), :2] if a >= 3 else A[min(b + 10, len(A) - 1), :2] - A[b, :2]
  d0 /= max(np.linalg.norm(d0), 1e-9)
  n0 = np.array([d0[1], -d0[0]])
  s = 1.0 if (B[0, :2] - A[k, :2]) @ n0 > 0 else -1.0  # the side it opens on (+1 = right of travel)
  taper = []
  for q in range(a, b + 1):
    c, w = A[q, :2].copy(), A[q, 3]
    A[q, :2], A[q, 3] = c - s * n0 * (w - w0) / 2, w0
    if (c - B[0, :2]) @ d0 < -0.5:  # before the new lane's own start
      taper.append([*(c + s * n0 * w0 / 2), A[q, 2], w - w0])
  taper = np.array(taper).reshape(-1, 4)
  if len(taper):
    g = np.linalg.norm(B[0, :2] - taper[-1, :2])
    t = np.arange(1, int(g / STEP_M)) * STEP_M / max(g, 1e-9)
    fill = taper[-1] + (B[0] - taper[-1]) * t[:, None]
    taper = np.vstack([taper, fill])
  return A, np.vstack([taper, B])


OPEN_TOL, STEP_M = 0.3, 1.0  # m wider than its usual width where an opening starts; sample spacing
pts = {}
for i, j, g, *br in links:
  if br:
    r = reshape(i, j)
    if r is not None:
      P[i] = r[0]  # a lane with two openings reshapes from the first's result
      pts[i] = np.round(r[0], 2).tolist()
      pts[j] = np.round(r[1], 2).tolist()
json.dump({"links": links, "connectors": conns, "pts": pts}, open(sys.argv[6], "w"))
succ, pred = {}, {}
for i, j, g, *_ in links:
  succ.setdefault(i, []).append(j)
  pred.setdefault(j, []).append(i)
split = {i for i, v in succ.items() if len(v) > 1} | {l[0] for l in links if l[3:]}  # a branch's lane also carries on
print(len(links), "links;", len(split), "splits;", sum(len(v) > 1 for v in pred.values()), "merges")
for i, j, g, *kind in links:
  print(f" {lanes[i]['left']}|{lanes[i]['right']} -> {lanes[j]['left']}|{lanes[j]['right']}  gap {g} m", *kind)
