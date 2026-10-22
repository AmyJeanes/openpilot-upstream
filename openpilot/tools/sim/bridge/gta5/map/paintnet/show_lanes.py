"""show_lanes.py lanes.json base.jpg x0 x1 y0 y1 out.jpg [--cm=10]: lanes.py's lane runs as centre lines over an image
of the area (surface.py's top layer): lanes between two paint lines green, a paint line and the road's edge orange,
each run's width as its line's thickness band (thin edges either side)."""
import json, sys
import numpy as np
import cv2

ln, base, out = sys.argv[1], sys.argv[2], sys.argv[7]
x0, x1, y0, y1 = map(float, sys.argv[3:7])
s = 100.0 / next((float(a[5:]) for a in sys.argv if a.startswith("--cm=")), 10.0)
img = cv2.imread(base)
img = (img[:, :int((x1 - x0) * s)] * 0.55).astype(np.uint8)
P = lambda p: (int(round((p[0] - x0) * s)), int(round((y1 - p[1]) * s)))
for r in json.load(open(ln)):
  A = np.array(r["pts"])
  col = (255, 0, 255) if r.get("kind") == "gore" else (0, 140, 255) if "e" in (r["left"], r["right"]) else (60, 220, 60)
  cv2.polylines(img, [np.array([P(p) for p in A], np.int32)], False, col, 2, cv2.LINE_AA)
cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
