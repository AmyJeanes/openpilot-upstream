"""plotspot.py x0 x1 y0 y1 out.png [lane ids...]: the surface raster's paint and kerbs, paint line ids, and lane bands."""
import json, sys
import numpy as np
import cv2
sys.path[:0] = ["/home/amy/git/openpilot-slowroads/openpilot/tools/sim/bridge/gta5/map/roadpaint"]
import arrows, src
x0, x1, y0, y1 = map(float, sys.argv[1:5])
res = 0.05
code, zz = arrows.load(x0, x1, y0, y1, res)
c = code[0]
img = np.zeros(c.shape + (3,), np.uint8)
img[(c & 7) == 1] = (60, 60, 60)
img[(c & 7) == 2] = (40, 40, 140)
img[(c & 7) == 3] = (90, 110, 90)
img[(c & 7) == 4] = (60, 60, 120)
img[(c >> 3) == 1] = (230, 230, 230)
img[(c >> 3) == 2] = (0, 200, 230)
s = 1 / res
P = lambda p: (int(round((p[0] - x0) * s)), int(round((y1 - p[1]) * s)))
O = "/mnt/e/gta5_audit/paintnet"
G = json.load(open(f"{O}/ic_groups.json")); L = json.load(open(f"{O}/ic_links.json"))
want = set(sys.argv[6:])
cols = [(0, 255, 0), (255, 0, 255), (0, 128, 255), (255, 255, 0), (255, 128, 0), (128, 255, 128)]
ci = 0
for k, r in enumerate(G["lanes"]):
  if want and str(k) not in want:
    continue
  A = np.array(L["pts"].get(str(k), r["pts"]))
  if not ((A[:, 0] > x0) & (A[:, 0] < x1) & (A[:, 1] > y0) & (A[:, 1] < y1)).any():
    continue
  d = np.gradient(A[:, :2], axis=0); d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  n = np.stack([d[:, 1], -d[:, 0]], 1)
  col = cols[ci % len(cols)]; ci += 1
  for side in (-1, 1):
    E = A[:, :2] + side * n * A[:, 3:4] / 2
    cv2.polylines(img, [np.array([P(p) for p in E], np.int32)], False, col, 2)
  cv2.polylines(img, [np.array([P(p) for p in A[:, :2]], np.int32)], False, col, 1)
  cv2.putText(img, str(k), P(A[len(A) // 2, :2]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
for r in src.paint():
  Q = np.array(r["pts"])
  m = (Q[:, 0] > x0) & (Q[:, 0] < x1) & (Q[:, 1] > y0) & (Q[:, 1] < y1)
  if m.any() and r["len"] > 3:
    q = Q[m][len(Q[m]) // 2]
    cv2.putText(img, str(r["id"]), P(q), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
cv2.imwrite(sys.argv[5], img)
