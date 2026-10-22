"""connect.py x0 x1 y0 y1 groups.json out.json: which lane carries on into which, where the paint between them breaks.

A lane's end links to a lane starting just ahead of it (0 < gap <= JOIN m along its travel direction, same deck, within
HEAD deg) when one of their edges lines up (within EDGE m sideways): the lane carries on with a line starting or ending
beside it. A lane can have two successors (a turn bay opening beside it: its one edge lines up with each) or two
predecessors (a merge). No link where paint crosses the gap (a stop line or crossing: lanes end at a junction's box).
JOIN is provisional (the one opening measured on the interchange is 13 m; re-measure on a bigger area).
Out: {"links": [[from lane index, to lane index, gap m]], "connectors": [{"from", "to", "ring": [[x, y], ...], "z"}]}:
a connector is the successor's width carried back to its predecessor's end, filling the break."""
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
    # sideways offsets of the successor's edges from the predecessor's, square to its direction
    off_l, off_r = (lb - la) @ na, (rb - ra) @ na
    if min(abs(off_l), abs(off_r)) > EDGE:
      continue
    # the successor must lie within the predecessor's span (not beside it) for the edge that doesn't line up
    if abs(off_l) > EDGE and not -EDGE <= (lb - la) @ na <= A[-1, 3] + EDGE:
      continue
    if abs(off_r) > EDGE and not -A[-1, 3] - EDGE <= (rb - ra) @ na <= EDGE:
      continue
    if crossed(ea, eb, A[-1, 2]):
      continue
    links.append([i, j, round(float(gap), 2)])
    # the successor's edges carried back to the predecessor's end: a strip across the break
    back = gap
    ring = [lb - da * back, rb - da * back, rb, lb]
    conns.append({"from": i, "to": j, "z": round(float((A[-1, 2] + B[0, 2]) / 2), 2),
                  "ring": [[round(float(x), 2), round(float(y), 2)] for x, y in ring]})
json.dump({"links": links, "connectors": conns}, open(sys.argv[6], "w"))
succ, pred = {}, {}
for i, j, g in links:
  succ.setdefault(i, []).append(j)
  pred.setdefault(j, []).append(i)
print(len(links), "links;", sum(len(v) > 1 for v in succ.values()), "splits;", sum(len(v) > 1 for v in pred.values()), "merges")
for i, j, g in links:
  print(f" {lanes[i]['left']}|{lanes[i]['right']} -> {lanes[j]['left']}|{lanes[j]['right']}  gap {g} m")
