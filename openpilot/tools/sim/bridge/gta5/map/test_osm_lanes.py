"""osm_lanes.py's cross-sections, lines and lane geometry, on the fixtures and on tags; and ynd_to_osm.py's lane tags,
which osm_lanes must read back to the layout of GTA's links as painted (ynd_to_osm.layout) for every kind of link. No
pytest in the venv: `python test_osm_lanes.py` runs them all."""
import os
import xml.etree.ElementTree as ET

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.lane_parity import compare, swapped
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, CENTRE, DIVIDER, EDGE, FORWARD, MEDIAN, PARKING, UK, US, \
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
