"""painted_islands.py's painted islands from white outlines and roadpaint's tiles, made here, and kerbs kept off them
(junctions.off_islands). No pytest needed: `python test_painted_islands.py` runs them all."""
import json
import os
import struct
import tempfile
import zlib

import numpy as np

from openpilot.tools.sim.bridge.gta5.map import painted_islands
from openpilot.tools.sim.bridge.gta5.map.junctions import off_islands

# a triangle 12 m on a side (62 m^2) outlined by two polylines meeting end to end, in the first tile (0..64 m)
A, B, C = [20.0, 20.0, 0.0], [32.0, 20.0, 0.0], [26.0, 30.4, 0.0]


def write_polylines(path, rows):
  with open(path, 'w') as f:
    for n, pts in enumerate(rows):
      pts = np.array(pts, float)
      length = float(np.hypot(*np.diff(pts[:, :2], axis=0).T).sum())
      f.write(json.dumps({'id': n, 'colour': 'white', 'style': 'solid', 'width': 0.15, 'len': length, 'pts': pts.tolist()}) + '\n')


def write_tile(d, inside_code):
  """Tile 0_0 (roadpaint's format: 5 cm pixels, row 0 at its north edge), one surface at height 0: road, but
  `inside_code` within the triangle."""
  n = 1280
  xs = (np.arange(n) + 0.5) * 0.05
  x, y = np.meshgrid(xs, 64.0 - xs)
  inside = (y >= 20.0) & (y <= 20.0 + (x - 20.0) * (10.4 / 6.0)) & (y <= 20.0 + (32.0 - x) * (10.4 / 6.0))
  code = np.where(inside, inside_code, 1).astype(np.uint8)[None]
  z = np.zeros((1, n, n), np.float32)
  body = struct.pack('<iiiifi', 0, 0, 0, n, 0.05, 1)
  for arr in (code, z):
    c = zlib.compress(arr.tobytes())
    body += struct.pack('<i', len(c)) + c
  with open(os.path.join(d, '0_0.bin'), 'wb') as f:
    f.write(body)


def test_outline_of_two_polylines():
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'polylines.jsonl')
    write_polylines(path, [[A, B, C], [C, A], [[50.0, 50.0, 0.0], [55.0, 50.0, 0.0], [55.0, 51.0, 0.0]]])  # and a stray L
    rings = painted_islands.outlines(path)
  assert len(rings) == 1 and rings[0].colour == 'white'
  ring = rings[0].ring
  assert np.allclose(ring[0], ring[-1]) and abs(painted_islands.area(ring[:-1]) - 62.4) < 0.1


def test_painted_not_raised():
  island = painted_islands.Island(np.array([A, B, C, A]), 'white')
  with tempfile.TemporaryDirectory() as d:
    write_tile(d, 1)  # road inside: painted
    assert len(painted_islands.painted([island], d)) == 1
    write_tile(d, 3)  # pavement inside: a raised island
    assert painted_islands.painted([island], d) == []


def test_no_kerbs_round_a_painted_island():
  tri = np.array([A, B, C])[:, :2]
  kerb = np.array([[20.0, 19.5], [32.0, 19.5], [40.0, 19.5], [50.0, 19.5]])  # 0.5 m off its base, on to 50 m
  pieces = off_islands(kerb, [tri])
  assert len(pieces) == 1 and pieces[0][0, 0] >= 32.9 and abs(pieces[0][-1, 0] - 50.0) < 1e-6
  assert len(off_islands(kerb + [0.0, -5.0], [tri])) == 1 and np.allclose(off_islands(kerb + [0.0, -5.0], [tri])[0][[0, -1]],
                                                                         (kerb + [0.0, -5.0])[[0, -1]])


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
