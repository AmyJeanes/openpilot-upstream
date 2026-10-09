"""lane_match.py's lane readings and junction areas on small maps made here. No pytest needed: `python
test_lane_match.py` runs them all (they need only numpy, as the bridge's environment)."""
import math
import os
import tempfile

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.junctions import Junctions
from openpilot.tools.sim.bridge.gta5.map.lane_match import MEDIAN, ONCOMING, OWN, WRONG_WAY, JunctionAreas, LaneMatcher
from openpilot.tools.sim.bridge.gta5.map.test_junctions import make

NORTH, SOUTH, EAST = math.pi / 2, -math.pi / 2, 0.0
# a divided road north (2 + 2 lanes 3.5 m wide, a 2 m median), the other direction's left turn bay laid in its median
# as its own one-way way, a one-way road north to its east and a road over the divided one, 10 m up
NODES = {1: (0.0, -200.0), 2: (0.0, 0.0), 3: (0.3, -40.0), 4: (0.3, -100.0), 5: (100.0, -200.0), 6: (100.0, 0.0),
         7: (6.0, -120.0), 8: (6.0, -180.0)}
DIVIDED = {'highway': 'primary', 'lanes': '4', 'lanes:forward': '2', 'lanes:backward': '2', 'width': '16',
           'width:lanes:forward': '3.5|3.5', 'width:lanes:backward': '3.5|3.5'}
WAYS = {
  1: (DIVIDED, [1, 2]),
  2: ({'highway': 'primary', 'oneway': 'yes', 'lanes': '1', 'width': '3.5', 'turn:lanes': 'left'}, [3, 4]),
  3: ({'highway': 'primary', 'oneway': 'yes', 'lanes': '2', 'width': '7'}, [5, 6]),
  4: ({'highway': 'primary', 'oneway': 'yes', 'lanes': '1', 'width': '3.5', 'bridge': 'yes'}, [7, 8]),
}
HEIGHTS = {n: {'ele': '10' if n in (7, 8) else '0'} for n in NODES}


def matcher() -> LaneMatcher:
  return LaneMatcher(make(NODES, WAYS, HEIGHTS))


def test_lanes_of_a_divided_road():
  m = matcher()
  r = m.match(6.0, -150.0, NORTH, 0.5)
  assert (r.way, r.lane, r.lanes, r.kind, r.oncoming) == (1, 1, 2, OWN, False)
  assert m.match(2.5, -150.0, NORTH, 0.5).lane == 0
  r = m.match(-3.0, -150.0, NORTH, 0.5)
  assert (r.lane, r.kind, r.oncoming, r.bay) == (-1, ONCOMING, True, False)
  assert m.match(0.0, -150.0, NORTH, 0.5).kind == MEDIAN
  r = m.match(-3.0, -150.0, SOUTH, 0.5)  # the other way, it's that direction's own lane
  assert (r.lane, r.kind) == (0, OWN)
  assert m.match(50.0, -150.0, NORTH, 0.5) is None  # off the roads
  assert m.match(6.0, -150.0, EAST, 0.5) is None  # across them


def test_lanes_beside():
  m = matcher()
  assert m.match(6.0, -150.0, NORTH, 0.5).beside == (OWN, None)  # the kerb to the right
  assert m.match(2.5, -150.0, NORTH, 0.5).beside == (ONCOMING, OWN)  # across the median
  assert m.match(-3.0, -150.0, NORTH, 0.5).beside == (ONCOMING, OWN)  # in the oncoming lanes: ours to the right
  assert m.match(98.25, -100.0, NORTH, 0.5).beside == (None, OWN)  # a one-way road: its left lane


def test_one_way_driven_against():
  r = matcher().match(101.0, -100.0, SOUTH, 0.5)
  assert (r.way, r.lane, r.kind, r.oncoming, r.bay) == (3, -1, WRONG_WAY, True, False)


def test_opposing_turn_bay_in_the_median():
  # northbound in the median, in the southbound left turn bay laid there
  r = matcher().match(0.6, -70.0, NORTH, 0.5)
  assert (r.way, r.kind, r.oncoming, r.bay) == (2, WRONG_WAY, True, True)
  r = matcher().match(0.6, -70.0, SOUTH, 0.5)  # southbound it's that direction's bay
  assert (r.way, r.kind, r.oncoming) == (2, OWN, False)


def test_road_overhead_is_not_the_one_driven():
  m = matcher()
  assert m.match(6.0, -150.0, NORTH, 0.5).way == 1  # under the road above, heading against it
  assert m.match(6.0, -150.0, NORTH, 10.5).kind == WRONG_WAY  # on it


def test_lane_centre():
  m = matcher()
  x, y, h = m.lane_centre(0.0, -150.0, NORTH, 9)  # clamped to the rightmost
  assert abs(x - 6.25) < 1e-6 and abs(y + 150.0) < 1e-6 and abs(h - NORTH) < 1e-6
  assert abs(m.lane_centre(0.0, -150.0, NORTH, 0)[0] - 2.75) < 1e-6
  x, y, h = m.lane_centre(0.0, -150.0, SOUTH, 9)
  assert abs(x + 6.25) < 1e-6 and abs(h - SOUTH) < 1e-6
  x, _, h = m.lane_centre(100.0, -150.0, NORTH, 0)  # a one-way's lanes are centred on its line
  assert abs(x - 98.25) < 1e-6 and abs(h - NORTH) < 1e-6
  assert m.lane_centre(50.0, -150.0, NORTH, 0) is None


def test_junction_areas():
  road = {'highway': 'residential', 'lanes': '2', 'width': '11'}
  osm = make({1: (0.0, 0.0), 2: (-100.0, 0.0), 3: (100.0, 0.0), 4: (0.0, 100.0), 5: (0.0, -100.0)},
             {10: (road, [2, 1]), 11: (road, [1, 3]), 12: (road, [4, 1]), 13: (road, [1, 5])},
             {n: {'ele': '20'} for n in range(1, 6)})
  areas = JunctionAreas.from_junctions(Junctions(osm))
  assert len(areas.centres) == 1 and abs(areas.heights[0] - 20.0) < 1e-6
  assert areas.inside(2.0, 2.0, 20.5) and areas.inside(2.0, 2.0)
  assert not areas.inside(0.0, 50.0, 20.5)
  assert not areas.inside(2.0, 2.0, 40.0)  # a road passing over it
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'areas.npz')
    areas.save(path)
    again = JunctionAreas.load(path)
  assert np.allclose(again.centres, areas.centres) and np.allclose(again.polygons[0], areas.polygons[0])
  assert again.inside(2.0, 2.0, 20.5) and not again.inside(0.0, 50.0, 20.5)


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
