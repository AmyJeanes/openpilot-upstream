"""show_groups.py groups.json base.jpg x0 x1 y0 y1 out.jpg [--cm=10] [--z=lo,hi]: groups.py's lanes coloured by
direction group with arrowheads along their travel direction (grey where unknown), flagged groups circled red, group
ids at their middles. --z keeps one deck."""
import json, sys
import numpy as np
import cv2

gj, base, out = sys.argv[1], sys.argv[2], sys.argv[7]
x0, x1, y0, y1 = map(float, sys.argv[3:7])
s = 100.0 / next((float(a[5:]) for a in sys.argv if a.startswith("--cm=")), 10.0)
zr = next((tuple(map(float, a[4:].split(","))) for a in sys.argv if a.startswith("--z=")), (-1e9, 1e9))
img = cv2.imread(base)
img = (img[:, :int((x1 - x0) * s)] * 0.45).astype(np.uint8)
P = lambda p: (int(round((p[0] - x0) * s)), int(round((y1 - p[1]) * s)))
G = json.load(open(gj))
flagged = {f["group"] for f in G["flags"]}
rng = np.random.default_rng(7)
col = {}
for r in G["lanes"]:
  A = np.array(r["pts"])
  if not zr[0] <= np.median(A[:, 2]) <= zr[1]:
    continue
  c = col.setdefault(r["group"], tuple(int(v) for v in rng.integers(90, 256, 3)))
  if r["dir"] == "?":
    c = (128, 128, 128)
  q = np.array([P(p) for p in A], np.int32)
  cv2.polylines(img, [q], False, c, 2, cv2.LINE_AA)
  if r["dir"] != "?":
    for k in range(5, len(A) - 1, 12):
      cv2.arrowedLine(img, P(A[k - 3]), P(A[k]), c, 2, cv2.LINE_AA, tipLength=0.6)
  m = A[len(A) // 2]
  if r["group"] in flagged:
    cv2.circle(img, P(m), 14, (0, 0, 255), 2)
  cv2.putText(img, str(r["group"]), (P(m)[0] + 6, P(m)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
