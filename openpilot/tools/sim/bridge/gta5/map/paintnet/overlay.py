"""overlay.py x0 x1 y0 y1 out.jpg [--cm=12]: the satellite tiles over an area with game-file paint (v2) and path-node
links drawn on top, to see how paint, nodes and imagery line up."""
import json, os, sys
import numpy as np
import cv2

F = "/mnt/e/gta5_ortho_20261018"
PAINT = "/mnt/e/gta5_scratch/gamefiles_v2/rp_all/polylines.jsonl"
PATHS = os.path.expanduser("~/gta5map/paths.jsonl")
x0, x1, y0, y1 = map(float, sys.argv[1:5])
out = sys.argv[5]
cm = next((float(a[5:]) for a in sys.argv if a.startswith("--cm=")), 12.0)
s = 100.0 / cm  # px per m
W, H = int((x1 - x0) * s), int((y1 - y0) * s)
img = np.zeros((H, W, 3), np.uint8)
k = int(round(3 / cm * 2560)) / 2560  # tile scale

def tile(i, j):
  n = f"o{i:03d}_{j:03d}"
  for d in ("full_water/tiles", "full_filled", "full/tiles"):
    p = f"{F}/{d}/{n}.png"
    if os.path.exists(p):
      return cv2.imread(p)
  return None

for i in range(int((x0 + 4300) / 69.12) - 1, int((x1 + 4300) / 69.12) + 2):
  for j in range(int((y0 + 4300) / 38.88) - 1, int((y1 + 4300) / 38.88) + 2):
    t = tile(i, j)
    if t is None:
      continue
    t = t[72:72 + 1296, 128:128 + 2304]
    cx, cy = -4300 + 69.12 * i, -4300 + 38.88 * j
    tw, th = int(round(69.12 * s)), int(round(38.88 * s))
    t = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)
    c0, r0 = int(round((cx - 34.56 - x0) * s)), int(round((y1 - (cy + 19.44)) * s))
    a, b = max(0, -r0), max(0, -c0)
    r1, c1 = min(H, r0 + th), min(W, c0 + tw)
    if r1 > r0 + a and c1 > c0 + b:
      img[r0 + a:r1, c0 + b:c1] = t[a:r1 - r0, b:c1 - c0]

img = (img * 0.75).astype(np.uint8)
P = lambda x, y: (int(round((x - x0) * s)), int(round((y1 - y) * s)))
nodes, links = {}, []
for line in open(PATHS):
  r = json.loads(line)
  if r["t"] == "n":
    nodes[(r["a"], r["i"])] = r
  elif r["t"] == "l":
    links.append(r)
for l in links:
  a, b = nodes.get((l["a"], l["i"])), nodes.get((l["ta"], l["ti"]))
  if a and b and (x0 <= a["x"] <= x1 and y0 <= a["y"] <= y1 or x0 <= b["x"] <= x1 and y0 <= b["y"] <= y1):
    fw, bk = (l["f"][2] >> 5) & 7, (l["f"][2] >> 2) & 7
    cv2.line(img, P(a["x"], a["y"]), P(b["x"], b["y"]), (255, 0, 255) if fw and bk else (255, 160, 0), 1, cv2.LINE_AA)
for n in nodes.values():
  if x0 <= n["x"] <= x1 and y0 <= n["y"] <= y1:
    cv2.circle(img, P(n["x"], n["y"]), 2, (0, 0, 255) if n["f"][2] & 4 else (255, 255, 255), -1)
col = {"white": (255, 255, 255), "yellow": (0, 220, 255)}
for line in open(PAINT):
  r = json.loads(line)
  pts = np.array(r["pts"])[:, :2]
  if not ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)).any():
    continue
  c = col.get(r["colour"], (0, 255, 0))
  q = np.array([P(x, y) for x, y in pts], np.int32)
  if r["style"] == "dashed":
    c = tuple(int(v * 0.6) for v in c)
  cv2.polylines(img, [q], False, c, 1, cv2.LINE_AA)
cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 88])
print(out, W, H)
