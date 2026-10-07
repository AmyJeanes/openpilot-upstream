"""at.py <extract dir> <x> <y>: geometries (entity, texture, shader, z, uv) whose triangles cover the point."""
import json, sys
import numpy as np

d, x, y = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
for line in open(d + '/geoms.jsonl'):
  g = json.loads(line)
  V = np.array(g['v'])
  if not ((abs(V[:, 0] - x) < 60) & (abs(V[:, 1] - y) < 60)).any():
    continue
  for t in np.array(g['i']).reshape(-1, 3):
    P = V[t]
    (x0, y0), (x1, y1), (x2, y2) = P[:, :2]
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(den) < 1e-9:
      continue
    l0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / den
    l1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / den
    l2 = 1 - l0 - l1
    if min(l0, l1, l2) >= 0:
      L = np.array([l0, l1, l2])
      print(g['ent'], g['tex'], g['sh'], 'z=%.2f' % (L @ P[:, 2]), 'uv=%.3f,%.3f' % (L @ P[:, 3], L @ P[:, 4]), 'a=%d' % (L @ P[:, 5]))
