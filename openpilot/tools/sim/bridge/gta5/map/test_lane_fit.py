"""lane_fit.py on made ways and paint. No pytest needed: `python test_lane_fit.py`."""
import json
import os
import tempfile

import numpy as np

from openpilot.tools.sim.bridge.gta5.map import lane_fit as fit
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import FORWARD, OsmLanes
from openpilot.tools.sim.bridge.gta5.map.osm_pbf import OsmData
from openpilot.tools.sim.bridge.gta5.map.paint_score import PaintIndex

TWO = {'highway': 'primary', 'lanes': '2', 'oneway': 'yes', 'width': '11'}  # lanes at -5.5..0 and 0..5.5
NODES = {1: (0.0, 0.0), 2: (0.0, 60.0), 3: (0.0, 120.0), 4: (0.0, 180.0)}


def made(ways: dict, nodes: dict = NODES) -> OsmLanes:
  ids = np.array(sorted(nodes), np.int64)
  tags = {i: {'gta:node': f'1:{i}', 'ele': '10'} for i in ids}
  data = OsmData(ids, np.array([nodes[i][1] for i in ids]), np.array([nodes[i][0] for i in ids]), tags, dict(ways), {})
  return OsmLanes(data, lambda lat, lon: (np.asarray(lon, float), np.asarray(lat, float)))


def files(d: str, lines_x: list[tuple[float, str]], stacked: bool = False):
  """Paint lines straight north at x (lanes.json format), and roads.json with the made road (and one over it)."""
  kinds = ['solid', 'dashed', 'centre', 'centre_dashed', 'edge']
  paint = os.path.join(d, 'paint.json')
  json.dump({'kinds': kinds, 'lines': [[kinds.index(k), x, -10.0, x, 190.0] for x, k in lines_x],
             'heights': [[10.0, 10.0] for _ in lines_x], 'layers': [0] * len(lines_x)}, open(paint, 'w'))
  roads = {'classes': ['primary'], 'ways': [[0, 2, 1, 0.0, -10.0, 0.0, 190.0]], 'widths': [11.0], 'heights': [[10, 10]]}
  if stacked:
    roads['ways'].append([0, 2, 1, -50.0, 90.0, 50.0, 90.0])
    roads['widths'].append(11.0)
    roads['heights'].append([18, 18])
  json.dump(roads, open(os.path.join(d, 'roads.json'), 'w'))
  return PaintIndex(paint), fit.Stacked(os.path.join(d, 'roads.json'))


def run(osm, way, lines_x, stacked=False):
  with tempfile.TemporaryDirectory() as d:
    paint, st = files(d, lines_x, stacked)
    return fit.fit_way(osm, way, paint, st)


def test_both_sides_set_centre_and_width():
  # the paint's lines 0.6 m right of the map's, its right lane 0.2 m wider: each edge onto its line, and the tags that
  # say so read back to the same lanes
  osm = made({10: (TWO, [1, 2]), 11: (TWO, [2, 3]), 12: (TWO, [3, 4])})
  f, why, info = run(osm, 11, [(-4.9, 'solid'), (0.6, 'dashed'), (6.3, 'solid')])
  assert why is None and f is not None, (why, info)
  assert np.allclose(f['after'], [-4.9, 0.6, 6.3], atol=0.01) and f['paint_side'].all()
  tags = fit.tags_for(osm, 11, f['after'])
  assert tags['width:lanes'] == '5.5|5.7', tags
  osm.applied = {'ways': {osm.way_key(11): {'tags': tags}}}
  osm._lanes.clear()
  sec = osm.lanes(11).section(FORWARD)
  assert np.allclose([sec[0].left, sec[0].right, sec[1].right], [-4.9, 0.6, 6.3], atol=0.02)
  assert osm.lanes(10).section(FORWARD)[0].left == -5.5  # the others as tagged


def test_one_side_keeps_the_width():
  # paint only between the lanes and on the right: the left lane is moved with its right edge, its width kept
  osm = made({10: (TWO, [1, 2]), 11: (TWO, [2, 3]), 12: (TWO, [3, 4])})
  f, why, _ = run(osm, 11, [(0.6, 'dashed'), (6.3, 'solid')])
  assert why is None and f is not None
  assert np.allclose(f['after'], [0.6 - 5.5, 0.6, 6.3], atol=0.01) and list(f['paint_side']) == [False, True, True]


def test_lane_under_4_m_left_alone():
  # a line 3 m from the next: a bay or a false line, not a lane: left alone and listed
  osm = made({10: (TWO, [1, 2]), 11: (TWO, [2, 3]), 12: (TWO, [3, 4])})
  f, why, _ = run(osm, 11, [(-5.5, 'solid'), (-2.5, 'dashed'), (5.5, 'solid')])
  assert f is None and why and ('under' in why or 'inside' in why), why


def test_paint_line_inside_a_lane_left_alone():
  # the paint has three lanes where the map (the game files) has two: their counts disagree, the map's are kept
  osm = made({10: (TWO, [1, 2]), 11: (TWO, [2, 3]), 12: (TWO, [3, 4])})
  f, why, _ = run(osm, 11, [(-5.5, 'solid'), (-0.4, 'dashed'), (0.0, 'dashed'), (2.75, 'dashed'), (5.5, 'solid')])
  assert f is None


def test_stacked_left_alone():
  osm = made({10: (TWO, [1, 2]), 11: (TWO, [2, 3]), 12: (TWO, [3, 4])})
  nodes = {**NODES, 5: (0.0, 80.0), 6: (0.0, 100.0)}
  osm = made({10: (TWO, [1, 2]), 11: (TWO, [2, 5]), 13: (TWO, [5, 6]), 14: (TWO, [6, 3]), 12: (TWO, [3, 4])}, nodes)
  f, why, _ = run(osm, 13, [(-4.9, 'solid'), (0.6, 'dashed'), (6.3, 'solid')], stacked=True)
  assert f is None and why and 'stacked' in why, why


def test_one_lane_among_many_is_ambiguous():
  # a one-lane link whose line runs on a lane line of a wide carriageway: either lane beside it fits as well
  one = {'highway': 'motorway_link', 'lanes': '1', 'oneway': 'yes', 'width': '6'}
  osm = made({10: (one, [1, 2]), 11: (one, [2, 3]), 12: (one, [3, 4])})
  f, why, info = run(osm, 11, [(-12.0, 'solid'), (-6.0, 'dashed'), (0.0, 'dashed'), (6.0, 'dashed'), (12.0, 'solid')])
  assert f is None and info.get('matched', 0) == 0, info


def test_applied_file_read_with_the_map():
  # OsmLanes.load reads GEOMETRY_FILE beside the pbf; way_key is the end nodes' GTA ids
  osm = made({10: (TWO, [1, 2])})
  assert osm.way_key(10) == '1:1>1:2'


if __name__ == "__main__":
  for name, fn in sorted(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"ok {name}")
