"""rp_view.py <roadpaint out dir> <cx> <cy> <half m> <zref> <out.png> [--segs]: top-down view of the tiles (surface
classes and paint at the layer nearest zref), optionally with the exact paint segments drawn over (red/green)."""
import sys, pathlib
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import rp

d = pathlib.Path(sys.argv[1])
cx, cy, half, zref = map(float, sys.argv[2:6])
out = sys.argv[6]
res = rp.RES
n = int(2 * half / res)
xs = cx - half + (np.arange(n) + 0.5) * res
ys = cy + half - (np.arange(n) + 0.5) * res
X, Y = np.meshgrid(xs, ys)
T = rp.Tiles(d / 'tiles')
code, z = T.sample(X, Y, zref)
cls, paint = code & 7, code >> 3
img = np.full((n, n, 3), 20, np.uint8)
img[cls == rp.ROAD] = (60, 60, 60)
img[cls == rp.KERB] = (150, 60, 60)
img[cls == rp.WALK] = (90, 90, 120)
img[cls == rp.GUTTER] = (110, 90, 60)
img[paint == 1] = (255, 255, 255)
img[paint == 2] = (255, 200, 0)
im = Image.fromarray(img)
if '--segs' in sys.argv:
  S = rp.read_segments(d / 'segments.bin')
  m = (np.abs(S['x1'] - cx) < half) & (np.abs(S['y1'] - cy) < half) & (np.abs(S['z1'] - zref) < 3)
  dr = ImageDraw.Draw(im)
  for s in S[m]:
    p = ((s['x1'] - cx + half) / res, (cy + half - s['y1']) / res)
    q = ((s['x2'] - cx + half) / res, (cy + half - s['y2']) / res)
    dr.line([p, q], fill=(255, 0, 0) if s['decal'] else (0, 220, 0), width=1)
if '--poly' in sys.argv:
  import json, random
  dr = ImageDraw.Draw(im)
  for line in open(d / 'polylines.jsonl'):
    L = json.loads(line)
    P = np.array(L['pts'])
    if not ((np.abs(P[:, 0] - cx) < half) & (np.abs(P[:, 1] - cy) < half)).any() or np.abs(P[:, 2] - zref).min() > 3:
      continue
    random.seed(L['id'])
    col = (random.randint(0, 255), random.randint(60, 255), random.randint(0, 255)) if L['style'] == 'dashed' else (255, 0, 0)
    xy = [((x - cx + half) / res, (cy + half - y) / res) for x, y, _ in P]
    dr.line(xy, fill=col, width=3 if L['pair'] is not None else 1)
im.save(out)
