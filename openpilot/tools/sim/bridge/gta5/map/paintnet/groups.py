"""groups.py x0 x1 y0 y1 dir.json out.json: lanes grouped into direction groups and given one travel direction each.

Lanes sharing a white line run the same way (white separates lanes going the same way in every case measured; yellow
doesn't follow one rule: GTA paints both edges of some motorway carriageways yellow). A group is lanes joined by white
lines. Its direction is voted by road arrows in its lanes and GTA one-way links along them (direction.py's votes).
A yellow line with lanes both sides is a centre line only sometimes (GTA also paints left-turn bays yellow), so it
only gives a group with no votes of its own a direction: right-hand traffic puts a centre line on the driver's left;
failing that, direction.py's yellow-on-the-left reading of its lanes does.
Groups whose arrows and links disagree, or which have no evidence at all, are flagged rather than guessed."""
import json, sys, collections
import numpy as np
import scan

lanes = [r for r in json.load(open(sys.argv[5])) if r["kind"] == "lane"]
colour = {}
for r in scan.src.paint():
  colour[r["id"]] = r["colour"]


def dirs(A):
  d = np.gradient(A[:, :2], axis=0)
  return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)


R = [np.array(r["pts"]) for r in lanes]
D = [dirs(A) for A in R]


def touching(i, j):
  """lanes i and j lie side by side somewhere (a sample of each within their half widths plus 0.5 m), and how
  their directions relate there (+1 same way along pts, -1 opposite)."""
  A, B = R[i], R[j]
  if abs(np.median(A[:, 2]) - np.median(B[:, 2])) > 2:
    return 0
  dd = np.linalg.norm(A[:, None, :2] - B[None, :, :2], axis=2)
  lim = A[:, 3][:, None] / 2 + B[:, 3][None, :] / 2 + 0.5
  k = np.argwhere(dd <= lim)
  if not len(k):
    return 0
  a, b = k[len(k) // 2]
  return 1 if D[i][a] @ D[j][b] > 0 else -1


by_line = collections.defaultdict(list)
for i, r in enumerate(lanes):
  for side in ("left", "right"):
    for lid in r.get(side + "_all", [r[side]]):
      if lid not in ("e", "r"):  # the road's edge and raster-only lines have no identity to share
        by_line[lid].append((i, side))
same, centre = [], []  # (i, j, relation)
for lid, uses in by_line.items():
  for a in range(len(uses)):
    for b in range(a + 1, len(uses)):
      i, j = uses[a][0], uses[b][0]
      t = touching(i, j)
      if not t:
        continue
      (same if colour.get(lid) == "white" else centre).append((i, j, t, lid, uses[a][1], uses[b][1]))

# direction groups: union over white lines, keeping each lane's flip relative to its group's reference
parent, flip = list(range(len(lanes))), [1] * len(lanes)


def find(i):
  if parent[i] == i:
    return i, 1
  r, f = find(parent[i])
  parent[i], flip[i] = r, flip[i] * f
  return r, flip[i]


for i, j, t, *_ in same:
  ri, fi = find(i)
  rj, fj = find(j)
  if ri != rj:
    parent[rj], flip[rj] = ri, fi * fj * t
groups = collections.defaultdict(list)
for i in range(len(lanes)):
  g, f = find(i)
  groups[g].append((i, f))

# votes in each group's reference orientation
votes = collections.Counter()
for g, mem in groups.items():
  for i, f in mem:
    votes[g, "arrows"] += f * lanes[i].get("arrow_votes", 0)
    votes[g, "links"] += f * (1 if lanes[i].get("link_votes", 0) > 0 else -1 if lanes[i].get("link_votes", 0) < 0 else 0)
# centre lines: the line is on each side's left. Lane i has the line on its `side` (as seen along its pts), so i
# runs along its pts if side == "left": constraint sense(g_i) * f_i = +1 if left else -1
cons = []
for i, j, t, lid, si, sj in centre:
  gi, fi = find(i)
  gj, fj = find(j)
  cons.append((gi, fi * (1 if si == "left" else -1), gj, fj * (1 if sj == "left" else -1), lid))
sense = {}
for g in groups:
  v = votes[g, "arrows"] * 3 + votes[g, "links"]  # an arrow is a direct statement; a link only lies near a lane
  sense[g] = 1 if v > 0 else -1 if v < 0 else 0
# propagate centre-line constraints to groups with no votes of their own
for _ in range(len(groups)):
  for gi, si, gj, sj, lid in cons:
    if sense[gi] == 0 and gi != gj:
      sense[gi] = si
    if sense[gj] == 0 and gi != gj:
      sense[gj] = sj
# still none (a carriageway beside a painted median, which isn't a lane to pair with): direction.py's own reading,
# the nearest yellow line across the lane's road on its left
for g, mem in groups.items():
  if not sense[g]:
    y = sum(f * (1 if lanes[i].get("sense") == "+" else -1 if lanes[i].get("sense") == "-" else 0) * lanes[i].get("yellow_votes", 0) for i, f in mem)
    sense[g] = 1 if y > 0 else -1 if y < 0 else 0
flags = []
for g, mem in groups.items():
  a, l = votes[g, "arrows"], votes[g, "links"]
  why = []
  if a and l and (a > 0) != (l > 0):
    why.append("arrows and links disagree")
  if not sense[g]:
    why.append("no direction evidence")
  if why:
    flags.append({"group": g, "lanes": [i for i, _ in mem], "why": sorted(set(why))})
out = []
for g, mem in groups.items():
  for i, f in mem:
    s = sense[g] * f
    A = R[i] if s >= 0 else R[i][::-1]
    L, Rt = ("left", "right") if s >= 0 else ("right", "left")  # reversed, its sides swap
    out.append({**lanes[i], "group": g, "pts": A.tolist(), "dir": "pts" if s else "?",
                "left": lanes[i][L], "right": lanes[i][Rt], "left_all": lanes[i][L + "_all"], "right_all": lanes[i][Rt + "_all"]})
json.dump({"lanes": out, "flags": flags}, open(sys.argv[6], "w"))
print(len(lanes), "lanes in", len(groups), "direction groups;", len(same), "white joins,", len(centre), "yellow joins;", len(flags), "flagged")
for f in flags:
  print(" flag", f["group"], len(f["lanes"]), "lanes:", "; ".join(f["why"]))
