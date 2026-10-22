"""show_trace.py trace.json base.jpg x0 x1 y0 y1 out.jpg [--cm=10]: trace.py's carriageways over an image of the area
(surface.py's top layer or overlay.py): each carriageway its own colour, centre line thick, edges thin, id at its start."""
import json, sys
import numpy as np
import cv2

tr, base, out = sys.argv[1], sys.argv[2], sys.argv[7]
x0, x1, y0, y1 = map(float, sys.argv[3:7])
s = 100.0 / next((float(a[5:]) for a in sys.argv if a.startswith("--cm=")), 10.0)
img = cv2.imread(base)
img = (img[:, :int((x1 - x0) * s)] * 0.5).astype(np.uint8)
P = lambda p: (int(round((p[0] - x0) * s)), int(round((y1 - p[1]) * s)))
rng = np.random.default_rng(3)
for c in json.load(open(tr)):
  col = tuple(int(v) for v in rng.integers(80, 256, 3))
  pts = np.array(c["pts"])
  xy = pts[:, :2]
  d = np.gradient(xy, axis=0)
  d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
  n = np.stack([d[:, 1], -d[:, 0]], 1)
  L = xy - n * pts[:, 3:4]
  R = xy + n * pts[:, 4:5]
  for q, th in ((xy, 3), (L, 1), (R, 1)):
    cv2.polylines(img, [np.array([P(p) for p in q], np.int32)], False, col, th, cv2.LINE_AA)
  cv2.putText(img, str(c["id"]), P(xy[len(xy) // 2]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
