"""show_sections.py sections.jsonl base.jpg x0 x1 y0 y1 out.jpg [--cm=10]: scan.py's sections drawn over an overlay
image of the same area (overlay.py or the left half of surface.py's): span green where both ends are kerbs/walk/gutter,
orange where one end is a step/none/far, crossings as dots (white/yellow)."""
import json, sys, collections
import numpy as np
import cv2

sec, base, out = sys.argv[1], sys.argv[2], sys.argv[7]
x0, x1, y0, y1 = map(float, sys.argv[3:7])
img = cv2.imread(base)
H, W = img.shape[:2]
s = 100.0 / next((float(a[5:]) for a in sys.argv if a.startswith("--cm=")), 10.0)
if img.shape[1] > (x1 - x0) * s + 4:  # surface.py's two halves: keep the top layer
  img = img[:, :int((x1 - x0) * s)]
img = (img * 0.6).astype(np.uint8)
P = lambda p: (int(round((p[0] - x0) * s)), int(round((y1 - p[1]) * s)))
ends = collections.Counter()
for line in open(sec):
  d = json.loads(line)
  if d.get("off_road"):
    cv2.circle(img, P(d["c"]), 2, (0, 0, 255), -1)
    continue
  c, n = np.array(d["c"][:2]), np.array(d["n"])
  hard = {"kerb", "walk", "gutter"}
  good = d["lend"] in hard and d["rend"] in hard
  ends[(d["lend"], d["rend"])] += 1
  cv2.line(img, P(c - n * d["left"]), P(c + n * d["right"]), (0, 200, 0) if good else (0, 140, 255), 1)
  for o, lid, colour, style, z in d["lines"]:
    cv2.circle(img, P(c + n * o), 2, (0, 230, 255) if colour == "yellow" else (255, 255, 255), -1)
cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(ends.most_common(12))
