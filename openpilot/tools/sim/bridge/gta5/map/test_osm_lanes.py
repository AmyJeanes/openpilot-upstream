"""osm_lanes.py's cross-sections, lines and lane geometry, on the fixtures and on tags; and ynd_to_osm.py's lane tags,
which osm_lanes must read back to the layout of GTA's links as painted (ynd_to_osm.layout) for every kind of link. No
pytest in the venv: `python test_osm_lanes.py` runs them all."""
import os
import xml.etree.ElementTree as ET

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.lane_parity import compare, swapped
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, CENTRE, DIVIDER, EDGE, EDGE_LINE, FORWARD, MEDIAN, PARKING, UK, US, \
  Section, WayLanes, _opens_left, lane_counts, offset_line
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
          for freeway in (False, True):
            if not back and steps not in (0, -7):  # the one-way offsets GTA uses
              continue
            f = flags(fwd, back, steps, narrow)
            road = WayLanes.from_tags({'highway': 'primary', **lane_tags(fwd, back, f, freeway and fwd >= 2)})
            for d, ab, ba in ((FORWARD, f, swapped(f)), (BACKWARD, swapped(f), f)):
              if Link(ab).lanes and compare(road, d, ab, ba, freeway) is not None:
                mismatched.add((fwd, back, steps))
  # overlapping lanes OSM can't say: negative offsets on two-way links but a single track's
  assert mismatched == {(f, b, s) for f in range(1, 5) for b in range(1, 4) for s in range(-7, 0) if (f, b, s) != (1, 1, -7)}


def test_painted_layout():
  """The lanes as measured in the game (map audit, topshot): Vinewood Blvd's narrow 2 + 2 with 6 steps of median, its
  double yellows 2.7 m either side, lane lines 7.1, kerbs 11.3-11.6; Palomino Ave's normal 2 + 2 with 6 steps, 2.7, 7.9,
  13.4; a 3-lane freeway's edges -8.9 / +9.3."""
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import lane_tags
  for narrow, edges in ((True, [2.7, 7.1, 11.5]), (False, [2.7, 8.2, 13.7])):
    road = WayLanes.from_tags({'highway': 'primary', **lane_tags(2, 2, flags(2, 2, 6, narrow))})
    assert [round(v, 2) for s in road.ours() for v in (s.left, s.right)][::2] + [round(road.ours()[-1].right, 2)] == edges
  road = WayLanes.from_tags({'highway': 'motorway', **lane_tags(3, 0, flags(3, 0, 0, False), True)})
  assert np.allclose(road.edges(), (-9.15, 9.15))


def test_placement_offset():
  # placement:offset moves the whole cross-section (lanes, lines, kerbs) right of where placement puts it
  tags = {'highway': 'motorway', 'oneway': 'yes', 'lanes': '2', 'width': '12.2', 'width:lanes': '6.2|6', 'placement': 'right_of:1'}
  base, moved = WayLanes.from_tags(tags), WayLanes.from_tags({**tags, 'placement:offset': '-0.6'})
  assert spans(base) == [(1, -6.2, 0.0), (1, 0.0, 6.0)]
  assert spans(moved) == [(1, -6.8, -0.6), (1, -0.6, 5.4)] and np.allclose(moved.edges(), (-6.8, 5.4))
  assert np.allclose([o for _, o, _ in lines(moved)], [o - 0.6 for _, o, _ in lines(base)])
  assert spans(moved, BACKWARD) == [(-1, -5.4, 0.6), (-1, 0.6, 6.8)]  # to the left seen the other way
  # without placement, off the middle of the lanes; malformed, ignored
  plain = {k: v for k, v in tags.items() if k != 'placement'}
  assert spans(WayLanes.from_tags({**plain, 'placement:offset': '0.25'})) == [(1, -5.85, 0.35), (1, 0.35, 6.35)]
  assert spans(WayLanes.from_tags({**plain, 'placement:offset': 'left'})) == spans(WayLanes.from_tags(plain))
  # a two-way road placed by its centre, painted 1 m right of the line
  two = WayLanes.from_tags({'highway': 'primary', 'lanes': '3', 'lanes:forward': '2', 'lanes:backward': '1', 'width': '16.5',
                            'width:lanes:forward': '5.5|4.5', 'width:lanes:backward': '6.5', 'placement:forward': 'left_of:1',
                            'placement:backward': 'left_of:1', 'placement:offset': '1'})
  assert spans(two) == [(-1, -5.5, 1.0), (1, 1.0, 6.5), (1, 6.5, 11.0)]
  # ynd_to_osm writes paint_survey's offsets so that osm_lanes reads the lanes back where they're painted
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import lane_tags
  road = WayLanes.from_tags({'highway': 'primary', **lane_tags(2, 0, flags(2, 0, 0, True),
                                                               painted={'lanes': [5.4, 5.45], 'placement': 'right_of:1', 'offset': 0.35})})
  assert spans(road) == [(1, -5.05, 0.35), (1, 0.35, 5.8)]
  painted = {'forward': [5.5, 4.5], 'backward': [6.5], 'median': 0.0, 'middle': False, 'offset': 1.0}
  road = WayLanes.from_tags({'highway': 'primary', **lane_tags(2, 1, flags(2, 1, 0, False), painted=painted)})
  assert spans(road) == [(-1, -5.5, 1.0), (1, 1.0, 6.5), (1, 6.5, 11.0)]


def test_parking_lane():
  road = WayLanes.from_tags({'highway': 'residential', 'lanes': '2', 'width': '13.3', 'parking:right': 'lane',
                             'parking:right:width': '2.3'})
  assert spans(road) == [(-1, -5.5, 0.0), (1, 0.0, 5.5)]  # not a lane, and the line stays between the lanes
  assert np.allclose(road.edges(), (-5.5, 7.8)) and np.allclose(road.parking_lanes(), [(5.5, 7.8)])
  assert lines(road) == [(EDGE, -5.5, None), (CENTRE, 0.0, 'dashed'), (PARKING, 5.5, None), (EDGE, 7.8, None)]
  assert lines(road, BACKWARD) == [(EDGE, -7.8, None), (PARKING, -5.5, None), (CENTRE, 0.0, 'dashed'), (EDGE, 5.5, None)]
  road = WayLanes.from_tags({'highway': 'primary', 'lanes': '2', 'oneway': 'yes', 'parking:both': 'lane'})
  assert spans(road) == [(1, -5.5, 0.0), (1, 0.0, 5.5)] and np.allclose(road.edges(), (-7.8, 7.8))
  road = WayLanes.from_tags({'highway': 'primary', 'lanes': '2', 'parking:left': 'street_side'})  # bays off the carriageway
  assert road.parking_lanes() == [] and road.edges() == (-5.5, 5.5)


def test_shoulders():
  # the Great Ocean Hwy: 7 m lanes either side of a 5.9 m median, edge lines with 4.4 m and 3.2 m shoulders beyond
  road = WayLanes.from_tags({'highway': 'trunk', 'lanes': '2', 'width': '27.5', 'width:lanes:forward': '7',
                             'width:lanes:backward': '7', 'shoulder': 'both', 'shoulder:left:width': '4.4',
                             'shoulder:right:width': '3.2'})
  assert spans(road) == [(-1, -9.95, -2.95), (1, 2.95, 9.95)]  # the line stays the middle of the lanes and median
  assert np.allclose(road.edges(), (-14.35, 13.15)) and np.isclose(road.tagged[0], 19.9)
  assert lines(road) == [(EDGE, -14.35, None), (EDGE_LINE, -9.95, 'solid'), (MEDIAN, -2.95, 'solid'), (MEDIAN, 2.95, 'solid'),
                         (EDGE_LINE, 9.95, 'solid'), (EDGE, 13.15, None)]
  assert lines(road, BACKWARD)[:2] == [(EDGE, -13.15, None), (EDGE_LINE, -9.95, 'solid')]
  assert [ln.colour for ln in road.lines() if ln.kind == EDGE_LINE] == ['white', 'white']
  # a one-way road's left edge line is yellow; road surface beyond the lanes with no line painted
  road = WayLanes.from_tags({'highway': 'primary', 'lanes': '2', 'oneway': 'yes', 'width': '13', 'shoulder': 'both',
                             'shoulder:width': '1', 'shoulder:right:markings': 'no'})
  assert spans(road) == [(1, -5.5, 0.0), (1, 0.0, 5.5)] and np.allclose(road.edges(), (-6.5, 6.5))
  assert lines(road) == [(EDGE, -6.5, None), (EDGE_LINE, -5.5, 'solid'), (DIVIDER, 0.0, 'dashed'), (EDGE, 6.5, None)]
  assert [ln.colour for ln in road.lines() if ln.kind == EDGE_LINE] == ['yellow']
  road = WayLanes.from_tags({'highway': 'primary', 'lanes': '1', 'oneway': '-1', 'width': '7.5', 'shoulder': 'left'})
  assert lines(road) == [(EDGE, -4.75, None), (EDGE_LINE, -2.75, 'solid'), (EDGE, 2.75, None)]  # travelling backward, its right
  assert [ln.white for ln in road.lines() if ln.kind == EDGE_LINE] == [True]
  assert lines(road, BACKWARD) == [(EDGE, -2.75, None), (EDGE_LINE, 2.75, 'solid'), (EDGE, 4.75, None)]
  # the files' colour where it isn't the default; none on an unmarked road
  road = WayLanes.from_tags({'highway': 'unclassified', 'lanes': '2', 'width': '13', 'shoulder': 'right',
                             'shoulder:right:width': '2', 'shoulder:right:markings': 'yellow'})
  assert lines(road)[-2:] == [(EDGE_LINE, 5.5, 'solid'), (EDGE, 7.5, None)] and road.lines()[-2].colour == 'yellow'
  road = WayLanes.from_tags({'highway': 'residential', 'lanes': '2', 'width': '13', 'lane_markings': 'no', 'shoulder': 'right',
                             'shoulder:right:width': '2'})
  assert lines(road) == [(EDGE, -5.5, None), (EDGE, 7.5, None)]


def test_one_edge_line_per_edge():
  # the yellow line the files paint along a one-way way's edge (divider:left / :right) is that edge's shoulder edge line
  # too: one line, the divider's kind, yellow, whatever the shoulder's markings say
  def edge_lines(tags, direction=FORWARD):
    return [(round(ln.offset, 3), ln.style, ln.colour) for ln in WayLanes.from_tags(tags).lines(direction) if ln.kind == EDGE_LINE]
  tags = {'highway': 'primary', 'lanes': '2', 'oneway': 'yes', 'width': '13', 'shoulder': 'both', 'shoulder:width': '1'}
  assert edge_lines(tags) == [(-5.5, 'solid', 'yellow'), (5.5, 'solid', 'white')]
  assert edge_lines({**tags, 'divider:left': 'dashed_line'}) == [(-5.5, 'dashed', 'yellow'), (5.5, 'solid', 'white')]
  assert edge_lines({**tags, 'divider:left': 'solid_line', 'shoulder:left:markings': 'no'})[0] == (-5.5, 'solid', 'yellow')
  assert edge_lines({**tags, 'divider:right': 'solid_line', 'shoulder:right:markings': 'white'})[1] == (5.5, 'solid', 'yellow')
  assert edge_lines({**tags, 'shoulder:left:markings': 'white'})[0] == (-5.5, 'solid', 'white')
  assert edge_lines({**tags, 'shoulder:right:markings': 'no'}) == [(-5.5, 'solid', 'yellow')]
  assert edge_lines({**tags, 'divider:left': 'solid_line'}, BACKWARD) == [(-5.5, 'solid', 'white'), (5.5, 'solid', 'yellow')]
  # drawn against its traffic: the lanes' left (divider:left) is the way's right, the shoulder on the way's left white
  back = {'highway': 'primary', 'lanes': '2', 'oneway': '-1', 'width': '12', 'shoulder': 'left', 'shoulder:left:width': '1',
          'divider:left': 'solid_line'}
  assert edge_lines(back) == [(-5.5, 'solid', 'white'), (5.5, 'solid', 'yellow')]
  # no shoulder but room between the lanes and the kerbs: the line at the lanes' edge, the kerb beyond; none unmarked
  road = WayLanes.from_tags({'highway': 'primary', 'lanes': '2', 'oneway': 'yes', 'width': '13', 'width:lanes': '5.5|5.5',
                             'divider:left': 'solid_line'})
  assert lines(road)[:2] == [(EDGE, -6.5, None), (EDGE_LINE, -5.5, 'solid')]
  assert edge_lines({**tags, 'lane_markings': 'no', 'divider:left': 'solid_line'}) == []
  # an edge line's own colour, else white (a line between lanes one way is white too; the centre and median yellow)
  from openpilot.tools.sim.bridge.gta5.map.osm_lanes import Line
  assert Line(EDGE_LINE, 5.5, 'solid').white and not Line(EDGE_LINE, -5.5, 'solid', 'yellow').white
  # side_by_side's line at a kerb with an edge line on its side would be a second line: osm_to_roads leaves it out
  from openpilot.tools.sim.bridge.gta5.map.osm_to_roads import lined_kerbs
  assert lined_kerbs(iter([(EDGE, -6.5), (EDGE_LINE, -5.5), (DIVIDER, 0.0), (EDGE, 6.5)])) == {-6.5}  # (as a generator)
  assert lined_kerbs([(EDGE, -5.5), (EDGE_LINE, 5.5), (EDGE, 7.5)]) == {7.5}
  assert lined_kerbs([(EDGE, -5.5), (DIVIDER, 0.0), (EDGE, 5.5)]) == set()


def test_placed_lanes_then_road_edges_then_edge_lines():
  # ynd_to_osm's order on a surveyed one-way link: the paint places its line on the lane line (paint_survey.place,
  # right_of:1), road_edges then puts the kerbs out at the asphalt's edges with shoulders beyond the painted edge lines,
  # leaving every line where the placement put it, and edge_line_tags reads the yellow left edge at the lanes' edge
  from collections import Counter

  from openpilot.tools.sim.bridge.gta5.map.paint_survey import correct_oneway
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import edge_line_tags, lane_tags, road_edges
  marks = [(-6.7, 'yellow', 'solid'), (-0.05, 'white', 'dashed'), (6.2, 'white', 'solid')]
  # the asphalt ending at the yellow line (no shoulder: the kerb on the line), then 1.3 m beyond it
  for asphalt, shoulder, left in ((-6.8, None, -6.7), (-8.0, '1.3', -8.0)):
    files = game_files(marks, asphalt, 8.6)
    got, _ = correct_oneway(files, 2, (-6.4, 6.4))
    assert got == {'lanes': [6.7, 6.2], 'placement': 'right_of:1'}, got
    why = Counter()
    tags = road_edges({'highway': 'primary', **lane_tags(2, 0, flags(2, 0, 0, False), painted=got)}, files, why)
    assert tags.get('shoulder:left:width') == shoulder and tags['shoulder:right:width'] == '2.4'
    tags = edge_line_tags(tags, files, why)
    assert tags['divider:left'] == 'solid_line' and not any(k.endswith(':markings') for k in tags), tags
    road = WayLanes.from_tags(tags)
    assert spans(road) == [(1, -6.7, 0.0), (1, 0.0, 6.2)] and np.allclose(road.edges(), (left, 8.6))
    assert [(ln.kind, round(ln.offset, 2), ln.style, ln.white) for ln in road.lines()] ==       [(EDGE, left, None, False), (EDGE_LINE, -6.7, 'solid', False), (DIVIDER, 0.0, 'dashed', True), (EDGE_LINE, 6.2, 'solid', True),
       (EDGE, 8.6, None, False)]


def game_files(marks, left=None, right=None):
  """Three game-file sections along a link with these marks [(offset, colour, kind)] and asphalt's edges."""
  return [{'s': s, 'marks': [{'type': k, 'colour': c, 'offset': o, 'conf': 0.95} for o, c, k in marks],
           'kerbs': {'left': left, 'right': right}, 'src': 'gamefiles'} for s in (2, 5, 8)]


def test_road_edges_from_the_game_files():
  from collections import Counter

  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import road_edges
  why = Counter()
  # the Great Ocean Hwy: lanes out to the edge lines 1.25 m past the layout's kerbs, shoulders out to the asphalt
  goh = game_files([(-9.9, 'white', 'solid'), (-3.0, 'yellow', 'double_solid'), (2.9, 'yellow', 'solid'), (9.9, 'white', 'solid')],
                   -14.3, 13.1)
  tags = {'highway': 'trunk', 'lanes': '2', 'lanes:forward': '1', 'lanes:backward': '1', 'width': '17.3',
          'width:lanes:forward': '5.7', 'width:lanes:backward': '5.7'}
  road = WayLanes.from_tags(road_edges(tags, goh, why))
  assert spans(road) == [(-1, -9.9, -2.95), (1, 2.95, 9.9)] and np.allclose(road.edges(), (-14.3, 13.1))
  assert [ln.kind for ln in road.lines()] == [EDGE, EDGE_LINE, MEDIAN, MEDIAN, EDGE_LINE, EDGE]
  # with more lanes one way (placement on the centre) each side runs out to its own edge line
  tags = {'highway': 'trunk', 'lanes': '3', 'lanes:forward': '2', 'lanes:backward': '1', 'width': '16.5',
          'placement:forward': 'left_of:1', 'placement:backward': 'left_of:1'}
  marks = [(-6.9, 'white', 'solid'), (0.0, 'yellow', 'double_solid'), (5.5, 'white', 'solid'), (12.0, 'white', 'solid')]
  road = WayLanes.from_tags(road_edges(tags, game_files(marks, -11.0), why))
  assert spans(road) == [(-1, -6.9, 0.0), (1, 0.0, 5.5), (1, 5.5, 12.0)] and np.allclose(road.edges(), (-11.0, 12.0))
  # one edge line, the lanes kept centred: the other side moves out alike where its asphalt has room, its shoulder unpainted
  marks = [(0.0, 'yellow', 'double_solid'), (6.1, 'white', 'edge_line')]
  got = road_edges({'highway': 'residential', 'lanes': '2', 'width': '11'}, game_files(marks, -9.0, 6.4), why)
  road = WayLanes.from_tags(got)
  assert spans(road) == [(-1, -6.1, 0.0), (1, 0.0, 6.1)] and np.allclose(road.edges(), (-9.0, 6.4))
  assert got['shoulder:left:markings'] == 'no' and lines(road)[-2:] == [(EDGE_LINE, 6.1, 'solid'), (EDGE, 6.4, None)]
  tags = {'highway': 'residential', 'lanes': '2', 'width': '11'}
  got = road_edges(tags, game_files(marks, -5.6, 6.4), why)  # no room on the left: the lanes stay, the kerb moves out
  assert got == {**tags, 'width': '11.9', 'shoulder': 'right', 'shoulder:right:width': '0.9', 'shoulder:right:markings': 'no'}
  # kerbs out to the asphalt's edges where no line is painted, the lanes as they were
  got = road_edges({'highway': 'primary', 'lanes': '1', 'oneway': 'yes', 'width': '5.5'}, game_files([], -4.5, 4.6), why)
  assert np.allclose(WayLanes.from_tags(got).edges(), (-4.5, 4.6)) and spans(WayLanes.from_tags(got)) == [(1, -2.75, 2.75)]
  # but not over other lanes painted beyond
  beside = game_files([(2.75, 'white', 'dashed'), (8.25, 'white', 'solid')], None, 8.6)
  assert road_edges({'highway': 'motorway', 'lanes': '1', 'oneway': 'yes', 'width': '5.5'}, beside, why)['width'] == '5.5'


def test_single_track_painted_as_two_lanes():
  from openpilot.tools.sim.bridge.gta5.map.paint_survey import painted_centre
  assert painted_centre(game_files([(0.2, 'yellow', 'dashed'), (4.1, 'yellow', 'solid')]))  # the port's yard roads
  assert painted_centre(game_files([(-0.3, 'white', 'dashed')], -5.0, 5.2))
  assert not painted_centre(game_files([(0.3, 'yellow', 'solid')], -1.0, 0.5))  # too narrow for two lanes: a path's paint
  assert not painted_centre(game_files([(3.0, 'yellow', 'solid')]))
  assert not painted_centre(game_files([(0.0, 'yellow', 'dashed')])[:1])


def test_lane_tags_fold_a_bay_into_the_median():
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import lane_tags

  def spans_of(fwd, back, f, bays):
    return spans(WayLanes.from_tags({'highway': 'primary', **lane_tags(fwd, back, f, False, bays)}))
  # narrow 2 + 2 with 6 steps of median (2.7 m either side): the bay fills the median, the lanes stay where they were
  assert spans_of(3, 2, flags(2, 2, 6, True), (True, False)) == \
    [(-1, -11.5, -7.1), (-1, -7.1, -2.7), (1, -2.7, 2.7), (1, 2.7, 7.1), (1, 7.1, 11.5)]
  assert spans_of(3, 3, flags(2, 2, 6, True), (True, True))[2:4] == [(1, 0.0, 2.7), (-1, -2.7, 0.0)][::-1]
  assert spans_of(3, 0, flags(2, 0, 0, False), (True, False)) == [(1, -11.0, -5.5), (1, -5.5, 0.0), (1, 0.0, 5.5)]


def test_detached_bays():
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import detached_bays

  def node(x, y, slip=False):
    return {'x': x, 'y': y, 'z': 0.0, 'f': [0, 1 if slip else 0, 0, 0, 0]}

  def find(bay_x, widens=False):
    nodes = {'P': node(0, -100), 'S': node(0, -60), 'Q': node(0, -30), 'J': node(0, 0), 'B1': node(bay_x, -40, True),
             'B2': node(bay_x, -15, True), 'W': node(-50, 0), 'E': node(50, 0), 'N': node(0, 50)}
    road, near = flags(2, 2, 6, False), flags(3, 2, 6, False) if widens else flags(2, 2, 6, False)
    rows = [('P', 'S', 2, 2, road), ('S', 'Q', 2, 2, road), ('Q', 'J', 3 if widens else 2, 2, near),  # northbound lanes forward
            ('S', 'B1', 1, 0, flags(1, 0, 0, False)), ('B1', 'B2', 1, 0, flags(1, 0, 0, False)), ('B2', 'J', 1, 0, flags(1, 0, 0, False)),
            ('W', 'J', 1, 1, flags(1, 1, 0, False)), ('J', 'E', 1, 1, flags(1, 1, 0, False)), ('J', 'N', 2, 2, road)]
    return detached_bays(nodes, rows, lambda k: k == 'J')
  # in the median, just left of the line: folded, from where it splits off to the junction
  assert find(-0.3) == [([3, 4, 5], [(1, True), (2, True)], 'S', 'J')]
  assert find(-0.3, widens=True) == find(-0.3)  # also beside a road gaining a lane on the way
  assert find(-9.0) == []  # in the oncoming lanes
  assert find(6.0) == []  # on the right: a slip lane, not a bay


def test_lanes_open_beside_a_bay():
  """A road gaining a lane: on the left where it's a turn bay, on the right where a bay was there already."""
  def section(turns):
    return Section.of(WayLanes.from_tags({'highway': 'primary', 'oneway': 'yes', 'lanes': str(len(turns)), 'turn:lanes': '|'.join(turns)}))
  one, bay, wider = section(['']), section(['left', 'through']), section(['left', 'through', 'through;right'])
  assert _opens_left(one, bay) and not _opens_left(bay, wider) and not _opens_left(one, section(['', 'right']))


def test_split_tapers():
  from collections import defaultdict
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import TAPER_NODE_AREA, remap_restrictions, split_tapers

  def node(x, y):
    return {'x': x, 'y': y, 'z': 0.0, 'f': [0, 0, 0, 0, 0], 'st': 0}
  nodes = {'P': node(0, -100), 'S': node(0, -60), 'Q': node(0, -30), 'J': node(0, 0)}
  road = flags(2, 2, 6, False)
  info = [[1, 'P', 'S', 2, 2, 'primary', 40, None, road], [2, 'S', 'Q', 3, 2, 'primary', 40, None, road],
          [3, 'Q', 'J', 3, 2, 'primary', 40, None, road]]
  lane_links = {('S', 'Q'), ('Q', 'J')}
  arrows = {1: {'turn:lanes:forward': 'through|through;right'}, 2: {'turn:lanes:forward': 'left|through|through;right'},
            3: {'turn:lanes:forward': 'left|through|through;right'}}
  bay_to = defaultdict(set, {2: {'Q'}, 3: {'J'}})
  # the bay opens 20 m before S and is open 5 m after it
  chain = [('P', 'S'), ('S', 'Q'), ('Q', 'J')]
  rows, parent, widen, bays, _ = split_tapers(nodes, info, [(chain, [0.0, 40.0, 70.0, 100.0], 20.0, 45.0)], lane_links, arrows, bay_to)
  assert [(r[0], r[1], r[2], r[3]) for r in rows] == [(1, 'P', (TAPER_NODE_AREA, 1), 2), (4, (TAPER_NODE_AREA, 1), 'S', 3),
                                                       (2, 'S', (TAPER_NODE_AREA, 2), 3), (5, (TAPER_NODE_AREA, 2), 'Q', 3), (3, 'Q', 'J', 3)]
  assert parent == {4: 1, 5: 2} and nodes[(TAPER_NODE_AREA, 1)]['y'] == -80.0
  assert widen == {4: {'forward': (0.0, 0.8)}, 2: {'forward': (0.8, 1.0)}}
  # arrows into S, before the lane's junction J: the new lane goes on through there
  assert arrows[4]['turn:lanes:forward'] == 'through|through|through;right' and arrows[1]['turn:lanes:forward'] == 'through|through;right'
  assert bays[1] == (False, False) and bays[4] == (True, False)
  # restrictions on to the pieces: from the piece at the via node, every piece of a via way in order
  ends = {r[0]: (r[1], r[2]) for r in rows}
  got = remap_restrictions([('no_u_turn', 1, [('n', 'S')], 1), ('no_left_turn', 1, [('w', 2)], 3)], ends, {1: [1, 4], 2: [2, 5]})
  assert got == [('no_u_turn', 4, [('n', 'S')], 4), ('no_left_turn', 4, [('w', 2), ('w', 5)], 3)]


def test_bay_full_width_from_its_start():
  # a lane open full width from where its chain begins (start == end: a junction just before, the painted lanes before
  # it, carriageways joining) is full width at that node, not widening from nothing over the first way (Eclipse Blvd,
  # where it drew a bay opening just past each junction the painted lane carries on through)
  from collections import defaultdict
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import split_tapers

  def node(x, y):
    return {'x': x, 'y': y, 'z': 0.0, 'f': [0, 0, 0, 0, 0], 'st': 0}
  nodes = {'S': node(0, -60), 'Q': node(0, -30), 'J': node(0, 0)}
  road = flags(2, 2, 6, False)
  info = [[2, 'S', 'Q', 3, 2, 'primary', 40, None, road], [3, 'Q', 'J', 3, 2, 'primary', 40, None, road]]
  chain = [('S', 'Q'), ('Q', 'J')]
  rows, parent, widen, bays, _ = split_tapers(nodes, info, [(chain, [0.0, 30.0, 60.0], 0.0, 0.0)], set(chain), {},
                                              defaultdict(set, {2: {'Q'}, 3: {'J'}}))
  assert widen == {} and parent == {} and [r[3] for r in rows] == [3, 3]


def test_back_to_back_bays():
  # a median shared by both ways' bays (a diamond): the forward one opens 45-55 m along towards D, the backward one 60-70 m
  # from D towards A, each from its own end
  from collections import defaultdict
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import TAPER_NODE_AREA, split_tapers

  def node(x, y):
    return {'x': x, 'y': y, 'z': 0.0, 'f': [0, 0, 0, 0, 0], 'st': 0}
  nodes = {'A': node(0, -100), 'B': node(0, -60), 'C': node(0, -40), 'D': node(0, 0)}
  road = flags(2, 2, 6, False)
  info = [[1, 'A', 'B', 2, 3, 'primary', 40, None, road], [2, 'B', 'C', 2, 2, 'primary', 40, None, road],
          [3, 'C', 'D', 3, 2, 'primary', 40, None, road]]
  ahead = ([('A', 'B'), ('B', 'C'), ('C', 'D')], [0.0, 40.0, 60.0, 100.0], 45.0, 55.0)
  behind = ([('D', 'C'), ('C', 'B'), ('B', 'A')], [0.0, 40.0, 60.0, 100.0], 60.0, 70.0)
  bay_to = defaultdict(set, {1: {'A'}, 3: {'D'}})
  rows, parent, widen, bays, applied = split_tapers(nodes, info, [ahead, behind], {('C', 'D'), ('B', 'A')}, {}, bay_to)
  n = [(TAPER_NODE_AREA, k) for k in (1, 2, 3)]
  assert applied == 2 and [r[:5] for r in rows] == [[1, 'A', n[0], 2, 3], [4, n[0], 'B', 2, 3], [2, 'B', n[1], 2, 2],
                                                     [5, n[1], n[2], 3, 2], [6, n[2], 'C', 3, 2], [3, 'C', 'D', 3, 2]]
  assert nodes[n[0]]['y'] == -70.0 and widen == {4: {'backward': (1.0, 0.0)}, 5: {'forward': (0.0, 1.0)}}
  assert bays[4] == (False, True) and bays[5] == (True, False) and bays[2] == (False, False)


def test_unpainted_bay_opens_before_gta_lane():
  # no painted opening: the bay is full width where GTA's lane begins (S), widening over TAPER_DEFAULT before it; where
  # the road begins at its carriageways joining (P), open from there
  from collections import Counter
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import TAPER_DEFAULT, lane_tapers

  def node(x, y):
    return {'x': x, 'y': y, 'z': 0.0, 'f': [0, 0, 0, 0, 0], 'st': 0}
  nodes = {'P': node(0, -100), 'S': node(0, -60), 'Q': node(0, -30), 'J': node(0, 0), 'U': node(5, -150), 'V': node(-5, -150)}
  road = flags(2, 2, 6, False)
  info = [[1, 'P', 'S', 2, 2, 'primary', 40, None, road], [2, 'S', 'Q', 3, 2, 'primary', 40, None, road],
          [3, 'Q', 'J', 3, 2, 'primary', 40, None, road]]
  lane_links, why = {('S', 'Q'), ('Q', 'J')}, Counter()
  [(chain, starts, start, end)] = lane_tapers(nodes, info, lane_links, lambda k: k == 'J', {}, [], why)
  assert chain == [('P', 'S'), ('S', 'Q'), ('Q', 'J')] and list(starts) == [0.0, 40.0, 70.0, 100.0]
  assert (start, end) == (40.0 - TAPER_DEFAULT, 40.0)
  carriageways = info + [[4, 'U', 'P', 2, 0, 'primary', 40, None, road], [5, 'P', 'V', 2, 0, 'primary', 40, None, road]]
  [(chain, _, start, end)] = lane_tapers(nodes, carriageways, lane_links, lambda k: k == 'J', {}, [], why)
  assert chain[0] == ('P', 'S') and (start, end) == (0.0, 0.0)
  assert why == Counter({"no paint: full width where GTA's lane begins": 1, 'no paint: open where the carriageways join': 1})
  # the paint counts the lane on P-S already (its counts the paint's, one more than GTA's): full width from P
  painted = [[1, 'P', 'S', 3, 2, 'primary', 40, None, road]] + info[1:]
  [(chain, _, start, end)] = lane_tapers(nodes, painted, lane_links, lambda k: k == 'J', {}, [], why, {1: (2, 2)})
  assert chain[0] == ('P', 'S') and (start, end) == (0.0, 0.0)


def test_dead_end_lanes():
  # a two-way link running on from a one-way link with nothing else at the node takes its direction
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import dead_end_lanes
  info = [[1, 'A', 'B', 1, 1], [2, 'C', 'B', 1, 0], [3, 'B', 'D', 1, 1], [4, 'D', 'E', 1, 0], [5, 'D', 'F', 1, 1]]
  assert dead_end_lanes(info[:2]) == 1 and info[0] == [1, 'B', 'A', 1, 0]  # traffic leaves B towards A
  info = [[1, 'A', 'B', 1, 1], [2, 'B', 'C', 1, 0]]
  assert dead_end_lanes(info) == 1 and info[0] == [1, 'A', 'B', 1, 0]  # and arrives at B from A
  info = [[3, 'B', 'D', 1, 1], [4, 'D', 'E', 1, 0], [5, 'D', 'F', 1, 1]]
  assert dead_end_lanes(info) == 0  # a junction
  chain = [[k, str(k), str(k + 1), 1, 1] for k in range(1, 6)] + [[9, '6', '7', 1, 0]]
  assert dead_end_lanes(chain) == 3 and [r[4] for r in chain] == [1, 1, 0, 0, 0, 0]  # up to DEAD_END_LINKS back


def test_neighbours_paint():
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import neighbours_paint

  def node(x, y):
    return {'x': x, 'y': y}
  nodes = {'A': node(0, 0), 'B': node(0, 30), 'C': node(0, 36), 'D': node(30, 36)}
  painted = {1: {'forward': [5.0], 'backward': [6.0], 'median': 0.0, 'divider': 'double_solid_line', 'middle': True}}
  rows = [(1, 'A', 'B', 1, 1), (2, 'C', 'B', 1, 1), (3, 'C', 'D', 1, 1)]
  got = neighbours_paint(nodes, rows, painted, {2, 3})
  assert got == {2: {**painted[1], 'forward': [6.0], 'backward': [5.0]}}  # drawn the other way; 3 turns off
  # GTA's 1 + 1 between two links painted 2 + 1: painted so, at one end only it isn't
  wide = {'forward': [4.0, 4.0], 'backward': [5.0], 'median': 0.0, 'middle': True}
  nodes['E'] = node(0, 70)
  rows = [(1, 'A', 'B', 2, 1), (2, 'B', 'C', 1, 1), (4, 'C', 'E', 2, 1)]
  assert neighbours_paint(nodes, rows, {1: wide, 4: wide}, {2}) == {2: wide}
  assert neighbours_paint(nodes, rows, {1: wide}, {2}) == {}
  # a link with sections the survey couldn't read: only where the road is painted alike at both ends
  assert neighbours_paint(nodes, rows, {1: wide, 4: wide}, set(), {2}) == {2: wide}
  same = [(1, 'A', 'B', 1, 1), (2, 'B', 'C', 1, 1), (4, 'C', 'E', 2, 1)]
  assert neighbours_paint(nodes, same, painted, set(), {2}) == {}


def test_unmarked_roads():
  # a short way between two the game files show unpainted is unpainted too, not where the road beyond is painted
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import unmarked_roads

  def node(x):
    return {'x': x, 'y': 0.0}
  nodes = {k: node(x) for k, x in (('A', 0.0), ('B', 30.0), ('C', 36.0), ('D', 66.0))}
  edges = {'left': -5.0, 'right': 5.0}

  def bare(n):
    return [{'src': 'gamefiles', 'marks': [], 'kerbs': edges}] * n
  rows = [(1, 'A', 'B', True, 'residential', False, False), (2, 'B', 'C', True, 'residential', False, False),
          (3, 'C', 'D', True, 'residential', False, False)]
  samples = {1: bare(4), 2: bare(2), 3: bare(4)}
  assert unmarked_roads(nodes, rows, lambda w, a, b: samples[w]) == {1, 2, 3}
  samples[3] = [{'src': 'gamefiles', 'marks': [{'conf': 1.0, 'colour': 'yellow', 'type': 'double_solid', 'offset': 0.0}], 'kerbs': edges}] * 4
  assert unmarked_roads(nodes, rows, lambda w, a, b: samples[w]) == {1}
  # a dirt track (or an off-road unclassified road) bare in the files is unpainted though they draw no asphalt edges;
  # a residential street without edges read isn't (a gap in the files)
  dirt = {w: [{'src': 'gamefiles', 'marks': [], 'kerbs': {'left': None, 'right': None}}] * 4 for w in (1, 2, 3)}
  rows = [(1, 'A', 'B', True, 'track', True, True), (2, 'B', 'C', True, 'unclassified', False, True),
          (3, 'C', 'D', True, 'residential', False, False)]
  assert unmarked_roads(nodes, rows, lambda w, a, b: dirt[w]) == {1, 2}


def test_median_edge_kinds():
  base = {'highway': 'primary', 'lanes': '4', 'lanes:forward': '2', 'lanes:backward': '2', 'width': '27.4',
          'width:lanes:forward': '5.5|5.5', 'width:lanes:backward': '5.5|5.5'}

  def kinds(tags, direction=FORWARD):
    return [line.style for line in WayLanes.from_tags(tags).lines(direction) if line.kind == MEDIAN]
  assert kinds(base) == ['solid', 'solid']  # one line each by default
  assert kinds({**base, 'divider': 'double_solid_line'}) == ['double_solid', 'double_solid']
  sides = {**base, 'divider:forward': 'solid_line', 'divider:backward': 'double_solid_line'}
  assert kinds(sides) == ['double_solid', 'solid']  # left to right: beside the oncoming lanes, beside ours
  assert kinds(sides, BACKWARD) == ['solid', 'double_solid']


def test_line_colours():
  # a white dashed centre (the port's); lane lines one way are white
  two = {'highway': 'residential', 'lanes': '2', 'lanes:forward': '1', 'lanes:backward': '1', 'width': '11', 'divider': 'dashed_line'}
  centre = [line for line in WayLanes.from_tags({**two, 'divider:colour': 'white'}).lines() if line.kind == CENTRE]
  assert [(line.style, line.white) for line in centre] == [('dashed', True)]
  assert [line.white for line in WayLanes.from_tags(two).lines() if line.kind == CENTRE] == [False]
  oneway = WayLanes.from_tags({'highway': 'primary', 'lanes': '2', 'oneway': 'yes', 'width': '11'})
  assert [line.white for line in oneway.lines() if line.kind == DIVIDER] == [True]


def test_edge_lines():
  # a one-way carriageway's yellow left edge beside its median (divider:left), seen either way along it
  tags = {'highway': 'motorway', 'lanes': '2', 'oneway': 'yes', 'width': '12', 'divider:left': 'solid_line'}
  road = WayLanes.from_tags(tags)
  assert [(ln.kind, ln.offset, ln.style, ln.white) for ln in road.lines() if ln.kind in (EDGE, EDGE_LINE)] == \
    [(EDGE, -6.0, None, False), (EDGE_LINE, -6.0, 'solid', False), (EDGE, 6.0, None, False)]
  assert [ln.offset for ln in road.lines(BACKWARD) if ln.kind == EDGE_LINE] == [6.0]  # its lanes' left, on the right seen backward
  both = WayLanes.from_tags({**tags, 'divider:right': 'double_solid_line'})
  assert [(ln.offset, ln.style) for ln in both.lines() if ln.kind == EDGE_LINE] == [(-6.0, 'solid'), (6.0, 'double_solid')]
  assert not [ln for ln in WayLanes.from_tags({**tags, 'lane_markings': 'no'}).lines() if ln.kind == EDGE_LINE]
  # a two-way way has no such edges
  two = {'highway': 'residential', 'lanes': '2', 'width': '11', 'divider:left': 'solid_line'}
  assert not [ln for ln in WayLanes.from_tags(two).lines() if ln.kind == EDGE_LINE]


def test_street_centres():
  # East Galileo Ave: a link given the class default between one painted dashed and a junction; carried on past another
  # default up to the nearest painted way, the halves of a centre seen the other way swapped
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import street_centres

  def node(x, y=0.0):
    return {'x': x, 'y': y}
  nodes = {'A': node(0), 'B': node(30), 'C': node(40), 'D': node(70), 'E': node(40, 30)}
  two = {'highway': 'residential', 'lanes': '2', 'lanes:forward': '1', 'lanes:backward': '1', 'width': '11', 'name': 'East Galileo Ave'}
  ways = [(1, 'A', 'B', {**two, 'divider': 'dashed_line'}), (2, 'B', 'C', {**two, 'divider': 'double_solid_line'}),
          (3, 'C', 'D', {**two, 'divider': 'double_solid_line'})]
  assert street_centres(nodes, ways, {2, 3}) == {2: {'divider': 'dashed_line'}, 3: {'divider': 'dashed_line'}}
  # the nearer side; white with its colour, the halves swapped where the way is drawn the other way
  ways[2] = (3, 'D', 'C', {**two, 'divider': 'solid_line;dashed_line', 'divider:colour': 'white'})
  ways[0] = (1, 'A', 'B', {**two, 'divider': 'double_solid_line'})
  nodes['A'], nodes['Z'] = node(-200), node(-260)  # 230 m on to its painted side ahead: further than CENTRE_CARRY
  assert street_centres(nodes, [(0, 'Z', 'A', {**two, 'divider': 'solid_line'})] + ways, {1, 2}) == \
    {1: {'divider': 'solid_line'}, 2: {'divider': 'dashed_line;solid_line', 'divider:colour': 'white'}}
  # an unpainted road on: no centre; another street, or a road turning off, gives nothing
  ways[2] = (3, 'C', 'D', {**two, 'lane_markings': 'no'})
  assert street_centres(nodes, ways, {1, 2}) == {1: {'divider': 'no'}, 2: {'divider': 'no'}}
  # a way given the default then found unpainted itself keeps that, and is unpainted to the ways beside it
  ways[2] = (3, 'C', 'D', {**two, 'divider': 'dashed_line'})
  ways[1] = (2, 'B', 'C', {**two, 'lane_markings': 'no'})
  assert street_centres(nodes, ways, {1, 2}) == {1: {'divider': 'no'}}
  ways = [(2, 'B', 'C', {**two, 'divider': 'double_solid_line'}), (3, 'C', 'D', {**two, 'divider': 'dashed_line', 'name': 'Other St'}),
          (4, 'C', 'E', {**two, 'divider': 'dashed_line'})]
  assert street_centres(nodes, ways, {2}) == {}


def test_carried_edges():
  # a short link without sections between two whose yellow left edge the files show; not where only one side has it
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import carried_edges
  nodes = {k: {'x': x, 'y': 0.0} for k, x in (('A', 0.0), ('B', 40.0), ('C', 50.0), ('D', 90.0))}
  one = {'highway': 'motorway', 'lanes': '2', 'oneway': 'yes', 'width': '12'}
  ways = [(1, 'A', 'B', {**one, 'divider:left': 'solid_line', 'divider:right': 'solid_line'}), (2, 'B', 'C', one),
          (3, 'C', 'D', {**one, 'divider:left': 'solid_line'})]
  assert carried_edges(nodes, ways, {2}) == {2: {'divider:left': 'solid_line'}}
  assert carried_edges(nodes, ways[:2], {2}) == {}


def test_painted_along():
  # paint along a link's line (a dash or two), not paint across its ends (a stop line, a crossing's stripe)
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import painted_along
  nodes = {'A': {'x': 0.0, 'y': 0.0, 'z': 0.0}, 'B': {'x': 30.0, 'y': 0.0, 'z': 0.0}, 'C': {'x': 6.0, 'y': 0.0, 'z': 0.0}}

  class Paint:
    def __init__(self, spans):
      self.spans = spans

    def near(self, p, z):
      return any(a <= p[0] <= b and abs(p[1]) < 0.6 for a, b in self.spans)
  assert painted_along(Paint([(10.0, 14.0)]), nodes, 'A', 'B')
  assert not painted_along(Paint([(0.0, 3.0), (27.0, 30.0), (15.0, 15.4)]), nodes, 'A', 'B')
  assert painted_along(Paint([]), nodes, 'A', 'C')  # too short to tell


def test_median_turn_lane():
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import lane_turns
  xy = {'P': (0, -100), 'Q': (0, -60), 'R': (0, -30), 'J': (0, 0), 'W': (-50, 0), 'E': (50, 0), 'N': (0, 50)}
  nodes = {k: {'x': x, 'y': y} for k, (x, y) in xy.items()}
  ways = [(1, 'P', 'Q', True), (2, 'Q', 'R', True), (3, 'R', 'J', True), (4, 'W', 'J', True), (5, 'J', 'E', True), (6, 'J', 'N', True)]
  lanes_to = {e: 1 for _, a, b, _ in ways for e in ((a, b), (b, a))}

  def turns(medians):
    tags, _, _, opened = lane_turns(nodes, ways, lanes_to, lambda k: k == 'J', {}, set(), [], medians=medians)
    return tags.get(3), opened
  # a one-lane approach in a median: the median is its left-turn lane on its last 30 m
  assert turns({('P', 'Q'), ('Q', 'R'), ('R', 'J')}) == ({'turn:lanes:forward': 'left|through;right'}, {('R', 'J')})
  assert turns(set()) == (None, set())  # no median: one lane, no arrows


def test_painted_median_lane():
  # a left arrow painted in a median GTA has no lane for (a hatched median ended, the lane opening in its room): the
  # median is a lane from that link on to the junction ahead; not from an arrow in a lane, nor a through arrow
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import PaintedArrows, painted_median_lanes
  xy = {'P': (0, -100), 'Q': (0, -40), 'R': (0, -20), 'J': (0, 0), 'W': (-50, 0), 'E': (50, 0), 'N': (0, 50)}
  nodes = {k: {'x': x, 'y': y} for k, (x, y) in xy.items()}
  ways = [(1, 'P', 'Q', True), (2, 'Q', 'R', True), (3, 'R', 'J', True), (4, 'W', 'J', True), (5, 'J', 'E', True), (6, 'J', 'N', True)]
  lanes_to = {e: 1 for _, a, b, _ in ways for e in ((a, b), (b, a))}
  road = [(1, 'P', 'Q'), (2, 'Q', 'R'), (3, 'R', 'J')]
  medians = {e for _, a, b in road for e in ((a, b), (b, a))}
  spans = {e: [(2.7, 7.1)] for e in medians}  # each way's lane beyond a 5.4 m median

  def chains(mark):
    return painted_median_lanes(nodes, ways, lanes_to, medians, PaintedArrows([mark]), spans, lambda k: k == 'J')
  assert chains((0.3, -70.0, 0.0, 0.0, 'left')) == [[('P', 'Q'), ('Q', 'R'), ('R', 'J')]]
  assert chains((0.3, -30.0, 0.0, 0.0, 'left')) == [[('Q', 'R'), ('R', 'J')]]
  assert chains((4.5, -70.0, 0.0, 0.0, 'left')) == []  # in the lane
  assert chains((0.3, -70.0, 0.0, 0.0, 'through')) == []
  assert chains((-0.3, -70.0, 0.0, 180.0, 'left')) == [[('Q', 'P')]]  # the other way's, on to where its road ends


def test_painted_turn_lanes():
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import PaintedArrows, arrows, lane_turns, road_on, with_paint
  everything = {'left', 'through', 'right'}
  assert road_on([10.0, 90.0, -40.0]) == {0, 2}
  assert road_on([89.0, 51.0, -77.0, -126.0]) == {1}  # a skewed junction's road on
  assert road_on([60.0, -90.0]) == set()  # a T
  # lanes taken from the left by the moves' lanes out: through as many as it has, a turn one, the right turn sharing first
  assert arrows(3, {'left': 1, 'through': 2, 'right': 1}) == ['left', 'through', 'through;right']
  assert arrows(3, {'left': 1, 'through': 1, 'right': 1}) == ['left', 'through', 'right']
  assert arrows(2, {'left': 1, 'through': 2, 'right': 1}) == ['left;through', 'through;right']
  assert arrows(4, {'left': 2, 'through': 1, 'right': 1}) == ['left', 'left', 'through', 'right']  # a turn's spare lanes out
  assert arrows(2, {'through': 1, 'right': 2}) == ['through', 'right']
  assert arrows(3, {'left': 1, 'right': 2}) == ['left', 'right', 'right']
  assert arrows(2, {'left': 1, 'right': 1}) == ['left', 'right']
  assert arrows(2, {'right': 1}) == ['right', 'right']
  # a painted lane's own arrow, less moves the junction hasn't; the lanes between don't cross it
  assert with_paint(['left', 'through', 'through;right'], {1: 'right'}, everything) == ['left', 'right', 'right']
  assert with_paint(['left', 'through;right'], {0: 'left;through'}, {'through', 'right'}) == ['through', 'through;right']

  # a 3-lane one-way approach from the south into a crossroads
  xy = {'P': (0, -100), 'Q': (0, -30), 'J': (0, 0), 'W': (-50, 0), 'E': (50, 0), 'N': (0, 50)}
  nodes = {k: {'x': x, 'y': y} for k, (x, y) in xy.items()}
  ways = [(1, 'P', 'Q', False), (2, 'Q', 'J', False), (3, 'J', 'W', True), (4, 'J', 'E', True), (5, 'J', 'N', True)]
  lanes_to = {e: 1 for _, a, b, two_way in ways for e in ((a, b), (b, a)) if two_way}
  lanes_to.update({('P', 'Q'): 3, ('Q', 'J'): 3, ('Q', 'P'): 0, ('J', 'Q'): 0})
  spans = {e: [(-8.25, -2.75), (-2.75, 2.75), (2.75, 8.25)] for e in (('P', 'Q'), ('Q', 'J'))}

  def turns(marks, ahead=1):
    painted = PaintedArrows(marks) if marks is not None else None
    tags = lane_turns(nodes, ways, {**lanes_to, ('J', 'N'): ahead}, lambda k: k == 'J', {}, set(), [], painted=painted,
                      spans=spans)[0]
    return tags.get(2, {}).get('turn:lanes')
  assert turns(None) == 'left|through|right'  # one lane on ahead
  assert turns(None, ahead=2) == 'left|through|through;right'
  # painted through;right in the right lane 20 m out, through in the middle one 60 m out; one pointing the other way
  marks = [(5.5, -20.0, 0.0, 0.0, 'through;right'), (0.0, -60.0, 0.0, 0.0, 'through'), (-5.5, -20.0, 0.0, 180.0, 'right')]
  assert turns(marks) == 'left|through|through;right'
  assert turns([(-5.5, -20.0, 0.0, 0.0, 'left;through')]) == 'left;through|through|right'
  assert turns([(5.5, -95.0, 0.0, 0.0, 'through;right')]) == 'left|through|right'  # beyond ARROW_REACH


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
