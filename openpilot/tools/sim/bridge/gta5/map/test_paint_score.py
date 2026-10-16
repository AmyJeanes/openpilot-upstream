"""paint_score.py on made maps and paint. No pytest needed: `python test_paint_score.py`."""
import json
import os
import tempfile

import numpy as np

from openpilot.tools.sim.bridge.gta5.map import paint_score as ps
from openpilot.tools.sim.bridge.gta5.test_lane_slots import osm_map

ROAD = {'highway': 'primary', 'lanes': '2', 'oneway': 'yes', 'width': '11'}  # lanes at -2.75 and +2.75 m


def lanes_json(path, lines):
  """lines: [(kind, [(x, y), ...])] as lanes.json (heights 0)."""
  kinds = ['edge', 'dashed', 'solid', 'centre', 'centre_dashed']
  json.dump({'kinds': kinds, 'lines': [[kinds.index(k)] + [v for p in pts for v in p] for k, pts in lines],
             'heights': [[0] * len(pts) for _, pts in lines], 'layers': [0] * len(lines)}, open(path, 'w'))


def roads_json(path):
  json.dump({'classes': ['primary'], 'ways': [[0, 2, 1, 0.0, -10.0, 0.0, 210.0]], 'widths': [11.0]}, open(path, 'w'))


def test_static_distances():
  # the map's lane line on x = 0; the paint's 0.3 m and 1.4 m off it (white), an edge 3 m off (unmatched at 2.5 m)
  with tempfile.TemporaryDirectory() as d:
    lanes_json(os.path.join(d, 'lanes.json'), [('dashed', [(0.0, 0.0), (0.0, 200.0)])])
    roads_json(os.path.join(d, 'roads.json'))
    paint = os.path.join(d, 'paint.json')
    lanes_json(paint, [('dashed', [(0.3, 0.0), (0.3, 100.0)]), ('solid', [(1.4, 100.0), (1.4, 200.0)]),
                       ('edge', [(3.0, 0.0), (3.0, 200.0)])])
    rep, places, arr = ps.static(d, paint)
  assert abs(rep['matched_share'] - 0.5) < 0.02, rep
  assert abs(rep['all']['over_0.5'] - 0.5) < 0.02 and abs(rep['all']['over_1.0'] - 0.5) < 0.02, rep
  assert rep['by_family']['white']['matched_km'] == 0.2 and rep['by_family']['edge']['matched_km'] == 0
  assert places and all(p['mean_off_m'] > 1.0 for p in places)
  d = arr['d'][arr['matched']]
  assert np.allclose(np.sort(np.unique(np.round(d, 2))), [0.3, 1.4])


def test_paint_lane_centre():
  lines = ps.parse_lines('-8.30:4;-2.80:1;2.90:1;8.40:4')
  assert ps.bracket(lines, 0.0) == ((-2.8, 1), (2.9, 1))
  err = ps.paint_errors(np.array([0.5, 6.0, 20.0]), [lines] * 3, np.ones(3, bool))
  assert abs(err[0] - 0.45) < 1e-9 and abs(err[1] - (6.0 - 5.65)) < 1e-9 and np.isnan(err[2])  # out past the edge


def test_keys_ramp_each_change():
  s = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
  assert ps._keys(s, np.array([0, 0, 1, 1, 1])) == [(0.0, 0.0), (1.0, 0.0), (2.0, 1.0), (4.0, 1.0)]


def test_rebuilt_route_and_target():
  # two ways north along x = 0; the car 2.75 m right of the line in lane 1 of the first, lane 1 of the second, whose
  # cross-section is 1 m further left (placement:offset=-1): the static target steps 1 m left at the node
  nodes = {1: (0.0, 0.0), 2: (0.0, 100.0), 3: (0.0, 200.0), 4: (20.0, 100.0)}
  shifted = {**ROAD, 'placement:offset': '-1'}
  osm = osm_map(nodes, {10: (ROAD, [1, 2]), 11: (shifted, [2, 3]), 12: (ROAD, [2, 4])})
  n = 150
  y = np.linspace(0.5, 199.5, n)
  z = {'trip': np.array(['a'] * n), 'x': np.full(n, 2.75), 'y': y, 'z': np.full(n, np.nan), 'h': np.zeros(n),  # north
       'route_at': y.copy(), 'route_right': np.full(n, 2.75), 'target_lane': np.ones(n), 't': np.arange(n) * 0.05}
  pts = ps.rebuild_route(osm, z['x'], z['y'], z['z'], z['h'], z['route_at'], z['route_right'])
  assert pts is not None and np.allclose(pts[1], (0.0, 100.0)), pts
  targets, legs = ps.target_centres(osm, z)
  T = targets['T']
  first, second = y < 99.0, y > 101.0
  assert np.allclose(T[first], 0.0, atol=1e-6) and np.allclose(T[second], -1.0, atol=1e-6), T
  assert np.isfinite(targets['T_line']).mean() > 0.9


if __name__ == "__main__":
  for name, fn in sorted(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"ok {name}")
