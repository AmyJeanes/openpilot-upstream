"""joingap.py x0 x1 y0 y1: for each lengthwise line's end, the nearest other line starting ahead of it in line (within
0.35 m sideways and 10 deg), as a histogram of the gap, to set lanes.py's JOIN from data."""
import json, math, sys
import numpy as np
import scan

L = {}
for l in open(scan.PAINT):
  r = json.loads(l)
  P = np.array(r["pts"], float)
  if r["len"] >= 6 and scan.inbox(P[:, 0].mean(), P[:, 1].mean(), 20):
    L[r["id"]] = (P, r["style"], r["colour"])
ends = []
for i, (P, st, co) in L.items():
  for P0, P1 in ((P[-1], P[-2]), (P[0], P[1])):  # each end, tangent pointing out of the line
    t = P0[:2] - P1[:2]; t /= np.linalg.norm(t)
    best = None
    for j, (Q, st2, co2) in L.items():
      if j == i:
        continue
      for Q0, Q1 in ((Q[0], Q[1]), (Q[-1], Q[-2])):  # the other line's end, tangent pointing into it
        u = Q1[:2] - Q0[:2]; u /= np.linalg.norm(u)
        d = Q0[:2] - P0[:2]
        ahead, side = d @ t, abs(d @ np.array([t[1], -t[0]]))
        if 0 < ahead < 40 and side < 1.5 and u @ t > math.cos(math.radians(10)) and abs(Q0[2] - P0[2]) < 1:
          if best is None or ahead < best[0]:
            best = (ahead, j, st, st2, round(side, 2))
    if best:
      ends.append(best)
g = np.array([e[0] for e in ends])
h, b = np.histogram(g, bins=[0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30, 40])
print(len(L), "lines,", len(ends), "ends with a line ahead in line")
print(" ".join(f"{b[i]:.1f}-{b[i+1]:.0f}:{h[i]}" for i in range(len(h))))
for e in sorted(ends)[:40]:
  print(round(e[0], 2), e[1:])
