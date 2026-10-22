"""flowdist.py x0 x1 y0 y1 lanes.json: distance from lane samples to the nearest GTA link on their deck running within
30 deg of the lane, as a histogram (to set direction.py's FLOW_NEAR from data)."""
import json, math, sys
import numpy as np
import scan

lanes = [r for r in json.load(open(sys.argv[5])) if r["kind"] == "lane"]
segs = [(A, B) for (ka, kb, A, B) in scan.L]
best = []
for r in lanes:
  A = np.array(r["pts"])
  d = np.gradient(A[:, :2], axis=0)
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  for k in range(0, len(A), 2):
    p, z, m = A[k, :2], A[k, 2], 99.0
    for P, Q in segs:
      if abs((P[2] + Q[2]) / 2 - z) > 3:
        continue
      ab = Q[:2] - P[:2]
      L = np.linalg.norm(ab)
      if L < 1 or abs((ab / L) @ d[k]) < math.cos(math.radians(30)):
        continue
      t = np.clip((p - P[:2]) @ ab / L ** 2, 0, 1)
      m = min(m, float(np.linalg.norm(P[:2] + t * ab - p)))
    best.append(m)
h, b = np.histogram(best, bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 100])
print(" ".join(f"{b[i]:.0f}-{b[i+1]:.0f}:{h[i]}" for i in range(len(h))))
