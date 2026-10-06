"""osm_lanes.py's cross-sections, lines and lane geometry, on the fixtures and on tags; and ynd_to_osm.py's lane tags,
which osm_lanes must read back to GTA's own layout (paths.Link) for every kind of link. No pytest in the venv:
`python test_osm_lanes.py` runs them all."""
import os
import xml.etree.ElementTree as ET

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.lane_parity import compare, swapped
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, CENTRE, DIVIDER, EDGE, FORWARD, MEDIAN, UK, US, WayLanes, \
  lane_counts, offset_line
from openpilot.tools.sim.bridge.gta5.map.paths import Link

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


def fixture(name: str) -> dict[int, dict]:
  """The tags of each way in a fixture."""
  root = ET.parse(os.path.join(FIXTURES, name)).getroot()
  return {int(w.get('id')): {t.get('k'): t.get('v') for t in w.iter('tag')} for w in root.iter('way')}


def spans(road: WayLanes, direction: int = FORWARD):
  return [(s.heading, round(s.left, 3), round(s.right, 3)) for s in road.section(direction)]


def lines(road: WayLanes, direction: int = FORWARD):
  return [(line.kind, round(line.offset, 3), line.style) for line in road.lines(direction)]


def test_two_way():
  road = WayLanes.from_tags(fixture('two_way.osm')[1])
  assert spans(road) == [(-1, -5.5, 0.0), (1, 0.0, 5.5)]
  assert spans(road, BACKWARD) == [(-1, -5.5, 0.0), (1, 0.0, 5.5)]
  assert lines(road) == [(EDGE, -5.5, None), (CENTRE, 0.0, 'dashed'), (EDGE, 5.5, None)]


def test_single_track():
  road = WayLanes.from_tags(fixture('single_track.osm')[1])
  assert road.single_track and road.counts == (0, 0, 1)
  for d in (FORWARD, BACKWARD):
    assert [(s.left, s.right) for s in road.ours(d)] == [(-2.75, 2.75)]
    assert road.oncoming(d) == []
  assert lines(road) == [(EDGE, -2.75, None), (EDGE, 2.75, None)]


def test_median():
  ways = fixture('median.osm')
  road = WayLanes.from_tags(ways[1])
  assert spans(road) == [(-1, -12.0, -6.5), (-1, -6.5, -1.0), (1, 1.0, 6.5), (1, 6.5, 12.0)]
  assert road.medians() == [(-1.0, 1.0)]
  assert lines(road)[2:4] == [(MEDIAN, -1.0, 'double_solid'), (MEDIAN, 1.0, 'double_solid')]
  road = WayLanes.from_tags(ways[2])  # 2 + 1, the line in the median's middle
  assert spans(road) == [(-1, -6.5, -1.0), (1, 1.0, 6.5), (1, 6.5, 12.0)]
  assert spans(road, BACKWARD) == [(-1, -12.0, -6.5), (-1, -6.5, -1.0), (1, 1.0, 6.5)]


def test_unequal():
  road = WayLanes.from_tags(fixture('unequal.osm')[1])
  assert spans(road) == [(-1, -5.5, 0.0), (1, 0.0, 5.5), (1, 5.5, 11.0)]
  assert spans(road, BACKWARD) == [(-1, -11.0, -5.5), (-1, -5.5, 0.0), (1, 0.0, 5.5)]
  assert [k for k, *_ in lines(road)] == [EDGE, CENTRE, DIVIDER, EDGE]
  assert lines(road)[1][2] == 'double_solid'  # two lanes one way


def test_oneway():
  ways = fixture('oneway.osm')
  road = WayLanes.from_tags(ways[1])
  assert spans(road) == [(1, -8.25, -2.75), (1, -2.75, 2.75), (1, 2.75, 8.25)]
  assert road.section(BACKWARD)[0].heading == -1 and road.ours(BACKWARD) == []
  # lane 2 may change left but not right: dashed by lane 1, solid on its own side towards lane 3
  assert [(k, s) for k, _, s in lines(road)] == [(EDGE, None), (DIVIDER, 'dashed'), (DIVIDER, 'solid_dashed'), (EDGE, None)]
  road = WayLanes.from_tags(ways[2])
  assert spans(road) == [(1, -5.5, 0.0)]


def test_turn_bay():
  ways = fixture('turn_bay.osm')
  before, bay = (WayLanes.from_tags(ways[k], defaults=US) for k in (1, 2))
  assert spans(before) == [(-1, -3.5, 0.0), (1, 0.0, 3.5)]
  assert spans(bay) == [(-1, -3.5, 0.0), (1, 0.0, 3.5), (1, 3.5, 7.0)]
  assert [s.lane.turns for s in bay.ours()] == [frozenset({'left'}), frozenset({'through'})]
  assert [(k, s) for k, _, s in lines(bay)][2] == (DIVIDER, 'solid')  # change:lanes=no
  assert spans(bay, BACKWARD) == [(-1, -7.0, -3.5), (-1, -3.5, 0.0), (1, 0.0, 3.5)]


def test_freeway():
  road = WayLanes.from_tags(fixture('freeway.osm')[1])
  assert spans(road) == [(1, -5.55, -1.85), (1, -1.85, 1.85), (1, 1.85, 5.55), (1, 5.55, 9.25)]
  ours = road.ours()
  assert ours[3].lane.turns == {'slight_right'} and ours[3].lane.destination == 'Airport'
  assert [(k, s) for k, _, s in lines(road)][1:4] == [(DIVIDER, 'dashed'), (DIVIDER, 'dashed'), (DIVIDER, 'solid')]


def test_left_hand_traffic():
  tags = fixture('lht.osm')[1]
  road = WayLanes.from_tags(tags, drive_on_right=False)
  assert spans(road) == [(1, -7.0, -3.5), (1, -3.5, 0.0), (-1, 0.0, 3.5)]
  assert spans(road, BACKWARD) == [(1, -3.5, 0.0), (-1, 0.0, 3.5), (-1, 3.5, 7.0)]
  assert [s.lane.turns for s in road.ours()] == [frozenset({'through'}), frozenset({'right'})]
  # a plain two-way road mirrors
  rht, lht = (WayLanes.from_tags({'highway': 'residential', 'lanes': '2', 'width': '7'}, drive_on_right=r) for r in (True, False))
  assert [(s.left, s.right) for s in rht.ours()] == [(0.0, 3.5)] and [(s.left, s.right) for s in lht.ours()] == [(-3.5, 0.0)]


def test_counts_and_fallbacks():
  assert lane_counts({'highway': 'residential'}) == (1, 1, 0)
  assert lane_counts({'highway': 'track'}) == (0, 0, 1)
  assert lane_counts({'highway': 'motorway'}) == (2, 0, 0)  # one-way by default
  assert lane_counts({'highway': 'primary', 'lanes': '3'}) == (2, 1, 0)
  assert lane_counts({'highway': 'primary', 'lanes': '3', 'lanes:backward': '2'}) == (1, 2, 0)
  assert lane_counts({'highway': 'primary', 'lanes': '2', 'oneway': '-1'}) == (0, 2, 0)
  road = WayLanes.from_tags({'highway': 'residential'}, defaults=UK)
  assert spans(road) == [(-1, -3.3, 0.0), (1, 0.0, 3.3)] and lines(road) == [(EDGE, -3.3, None), (EDGE, 3.3, None)]
  road = WayLanes.from_tags({'highway': 'primary', 'lanes': '2', 'width': '9'}, defaults=UK)  # width shared out
  assert spans(road) == [(-1, -4.5, 0.0), (1, 0.0, 4.5)]
  road = WayLanes.from_tags({'highway': 'primary', 'lanes': '2', 'width': "24'"})
  assert abs(road.width - 7.3152) < 1e-9


def test_centre_turn_lane():
  road = WayLanes.from_tags({'highway': 'secondary', 'lanes': '3', 'lanes:both_ways': '1', 'width': '10.5',
                             'turn:lanes:both_ways': 'left'})
  assert spans(road) == [(-1, -5.25, -1.75), (0, -1.75, 1.75), (1, 1.75, 5.25)]
  assert [(k, s) for k, _, s in lines(road)][1:3] == [(CENTRE, 'solid_dashed'), (CENTRE, 'dashed_solid')]
  assert len(road.ours()) == 1 and len(road.oncoming()) == 1


def test_access_lanes():
  road = WayLanes.from_tags({'highway': 'primary', 'oneway': 'yes', 'lanes': '2', 'bus:lanes': 'designated|', 'width': '7'})
  assert [s.lane.access for s in road.ours()] == [False, True]


def test_offset_line():
  np.testing.assert_allclose(offset_line([(0, 0), (10, 0)], 2.0), [(0, -2), (10, -2)])  # right of travel east is south
  corner = offset_line([(0, 0), (10, 0), (10, 10)], 1.0)  # a left turn: the right side's corner is mitred outward
  np.testing.assert_allclose(corner, [(0, -1), (11, -1), (11, 10)], atol=1e-9)
  road = WayLanes.from_tags(fixture('unequal.osm')[1])
  ours = road.centres([(0, 0), (100, 0)])
  np.testing.assert_allclose(ours[0], [(0, -2.75), (100, -2.75)])
  np.testing.assert_allclose(road.centres([(0, 0), (100, 0)], BACKWARD)[0], [(100, 2.75), (0, 2.75)])


def flags(fwd: int, back: int, steps: int, narrow: bool) -> list[int]:
  """A GTA link record's flags: lanes each way, offset in fourteenths of a lane (paths.Link), narrow."""
  return [0, (2 if narrow else 0) | ((abs(steps) & 7) << 4) | (128 if steps < 0 else 0), (fwd << 5) | (back << 2), 0, 0]


def test_lane_tags_reproduce_gta_layout():
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import lane_tags
  mismatched = set()
  for fwd in range(1, 5):
    for back in range(4):
      for steps in range(-7, 8):
        for narrow in (False, True):
          if not back and steps not in (0, -7):  # the one-way offsets GTA uses
            continue
          f = flags(fwd, back, steps, narrow)
          road = WayLanes.from_tags({'highway': 'primary', **lane_tags(fwd, back, f)})
          for d, ab, ba in ((FORWARD, f, swapped(f)), (BACKWARD, swapped(f), f)):
            if Link(ab).lanes and compare(road, d, Link(ab), Link(ba)) is not None:
              mismatched.add((fwd, back, steps))
  # overlapping lanes OSM can't say: negative offsets on two-way links but a single track's
  assert mismatched == {(f, b, s) for f in range(1, 5) for b in range(1, 4) for s in range(-7, 0) if (f, b, s) != (1, 1, -7)}


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
