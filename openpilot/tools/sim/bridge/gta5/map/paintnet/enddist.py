"""enddist.py x0 x1 y0 y1 lanes.json: how far the road surface continues past each end of each lane run (along the
lane), to tell strips across a road (a crossing between its two border lines: kerb at both ends) from lanes."""
import json, sys
import numpy as np
import scan

lanes = [r for r in json.load(open(sys.argv[5])) if r["kind"] == "lane"]
rows = []
for r in lanes:
  A = np.array(r["pts"])
  if len(A) < 3:
    continue
  e1 = A[-1, :2] - A[-3, :2]; e1 /= np.linalg.norm(e1)
  e0 = A[0, :2] - A[2, :2]; e0 /= np.linalg.norm(e0)
  d1, k1 = scan.extent(A[-1, :2], e1, A[-1, 2], 1)
  d0, k0 = scan.extent(A[0, :2], e0, A[0, 2], 1)
  rows.append((min(d0, d1), max(d0, d1), len(A), r["left"], r["right"], A[len(A) // 2, :2].round(1).tolist(), k0, k1))
rows.sort(key=lambda r: (r[1], r[0]))
for row in rows[:25]:
  print(row)
mx = np.array([r[1] for r in rows])
h, b = np.histogram(mx, bins=[0, 1, 2, 3, 5, 8, 12, 20, 31])
print("further end:", " ".join(f"{b[i]:.0f}-{b[i+1]:.0f}:{h[i]}" for i in range(len(h))))
