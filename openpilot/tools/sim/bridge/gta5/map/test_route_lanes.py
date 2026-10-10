"""Lanes along a route (osm_lanes.py RouteLanes, Section, turn_targets, the route-to-way matching), the PBF reader
(osm_pbf.py) and nav's target lanes from turn arrows, on small hand-made maps. No pytest needed: `python
test_route_lanes.py` runs them all (they need only numpy, as the bridge's environment)."""
import os
import struct
import tempfile
import zlib

import numpy as np

from openpilot.selfdrive.navd.planner import Through, Turn, aim, lane_plan, parse_arrows, throughs
from openpilot.tools.sim.bridge.gta5.map import osm_pbf
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes, RouteLanes, Section, WayLanes, _curve_jogs, _ease_jogs, _keyed, \
  continuing, \
  turn_targets, ways_from_nodes, ways_from_trace
from openpilot.tools.sim.bridge.gta5.map.osm_pbf import OsmData

# a junction at (0, 0): a one-way road north into it, two lanes, widening to three 30 m before it (a left turn bay);
# a one-way road west out of it, and two-way roads east and north
NODES = {1: (0.0, -100.0), 2: (0.0, -30.0), 3: (0.0, 0.0), 4: (-100.0, 0.0), 5: (100.0, 0.0), 6: (0.0, 100.0)}
WAYS = {
  10: ({'highway': 'primary', 'oneway': 'yes', 'lanes': '2', 'width': '7'}, [1, 2]),
  11: ({'highway': 'primary', 'oneway': 'yes', 'lanes': '3', 'width': '10.5', 'turn:lanes': 'left|through|through;right'}, [2, 3]),
  12: ({'highway': 'primary', 'oneway': 'yes', 'lanes': '2', 'width': '7'}, [3, 4]),
  13: ({'highway': 'primary', 'lanes': '2', 'width': '7'}, [3, 5]),
  14: ({'highway': 'primary', 'lanes': '2', 'width': '7'}, [3, 6]),
}


def junction_map() -> OsmLanes:
  ids = np.array(sorted(NODES), np.int64)
  data = OsmData(ids, np.array([NODES[i][1] for i in ids]), np.array([NODES[i][0] for i in ids]), {}, dict(WAYS), {})
  return OsmLanes(data, lambda lat, lon: (lon, lat))


def left_turn_route() -> np.ndarray:
  return np.array([NODES[1], NODES[2], NODES[3], NODES[4]])


def test_section():
  sec = Section.of(WayLanes.from_tags({'highway': 'primary', 'lanes': '3', 'lanes:forward': '2', 'lanes:backward': '1',
                                       'width': '12.5', 'width:lanes:forward': '3.5|3.5', 'width:lanes:backward': '3.5'}))
  assert (sec.lanes, sec.back, sec.lo, sec.hi, sec.two_way) == (2, 1, -1, 1, True)
  left = sec.ours[0].left  # a 2 m median between the directions
  assert abs(left - (sec.spans[0].right + 2.0)) < 1e-6
  assert abs(sec.offset(0) - (left + 1.75)) < 1e-6 and abs(sec.offset(0.5) - (left + 3.5)) < 1e-6
  assert abs(sec.frac(left + 1.75)) < 1e-6 and abs(sec.frac(left + 7.0) - 1.5) < 1e-6
  assert sec.lane(left + 4.0) == 1 and sec.lane(left - 3.0) == -1 and sec.lane(left + 50.0) == 1
  assert sec.offset(-1) == sec.spans[0].centre  # the oncoming lane's own centre
  single = Section.of(WayLanes.from_tags({'highway': 'service', 'lanes': '1', 'width': '4'}))
  assert (single.lanes, single.lo, single.hi, single.two_way) == (1, 0, 0, True)


def test_turn_targets():
  turns = [frozenset({'left'}), frozenset({'through'}), frozenset({'through', 'right'})]
  assert turn_targets(turns, 'left') == [0] and turn_targets(turns, 'right') == [2] and turn_targets(turns, 'through') == [1, 2]
  assert turn_targets(turns, 'left', fork=True) == []  # forks and exits take slight_* arrows
  assert turn_targets([frozenset({'slight_left'}), frozenset({'through'})], 'left', fork=True) == [0]
  assert turn_targets([frozenset(), frozenset()], 'left') == []  # no arrows: nothing said


def test_ways_from_nodes():
  osm = junction_map()
  assert osm.tagged
  assert ways_from_nodes(left_turn_route(), osm) == [(10, True, 0, 1), (11, True, 1, 2), (12, True, 2, 3)]
  # starting and ending part way along ways, against the north road's direction
  pts = np.array([(0.0, 60.0), NODES[3], (0.0, -10.0)])
  assert ways_from_nodes(pts, osm) == [(14, False, 0, 1), (11, False, 1, 2)]


def test_ways_from_trace():
  osm = junction_map()
  pts = np.array([(0.0, 90.0), (0.0, 40.0), NODES[3], (50.0, 0.0), NODES[5]])
  edges = [{'way_id': 14, 'begin_shape_index': 0, 'end_shape_index': 2}, {'way_id': 13, 'begin_shape_index': 2, 'end_shape_index': 4}]
  assert ways_from_trace(edges, pts, osm) == [(14, False, 0, 2), (13, True, 2, 4)]


def test_route_lanes():
  osm = junction_map()
  pts = left_turn_route()
  lanes = RouteLanes.from_osm(pts, ways_from_nodes(pts, osm), osm)
  assert [s.lanes for s in lanes.sections] == [2, 3, 2]
  assert lanes.arrows == [(100.0, lanes.sections[1].turns)] and lanes.arrows_near(100.0) == lanes.sections[1].turns
  assert list(lanes.junctions) == [100.0]
  assert len(lanes.corners) == 1 and abs(lanes.corners[0][0] - 100.0) < 2 and abs(lanes.corners[0][1] - 90.0) < 1
  # the bay widens from nothing on the left over TAPER_M from where the way's lane count rises
  assert 1 in lanes.tapers
  start = lanes.section_at(70.0, 1)
  assert start.lanes == 3 and start.ours[0].right - start.ours[0].left < 0.01
  assert abs(start.ours[1].left - lanes.sections[0].ours[0].left) < 0.01  # the old lanes go on where they were
  assert abs(lanes.section_at(85.0, 1).ours[0].right - lanes.section_at(85.0, 1).ours[0].left - 1.75) < 0.01
  assert lanes.section_at(10.0, 0) is lanes.sections[0]
  assert 3.3 < lanes.section_at(99.0, 1).ours[0].right - lanes.section_at(99.0, 1).ours[0].left < 3.5
  # the leftmost lane: into the bay as it opens, then through the corner on a fillet, into the left lane out
  line = lanes.lane_line(0.0, [(0.0, 0.0)])
  before = line[line[:, 1] < -45.0]
  assert np.allclose(before[:, 0], -1.75, atol=0.05)
  bay = line[(line[:, 1] > -28.0) & (line[:, 1] < -20.0)]
  assert np.all(np.diff(bay[:, 0]) <= 0.01) and bay[-1, 0] < -2.5  # drifting over with the bay
  after = line[line[:, 0] < -30.0]
  assert len(after) and np.allclose(after[:, 1], -1.75, atol=0.05)
  corner = line[(line[:, 1] > -25.0) & (line[:, 0] > -25.0)]
  assert np.hypot(*corner.T).min() > 2.5  # cuts the corner rather than running to the junction's node
  arc = line[line[:, 1] > -20.0]
  d = np.diff(arc, axis=0)
  turn = np.degrees(np.abs(np.diff(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))))
  assert turn.max() < 15.0  # smooth
  # from inside the corner, the rest of the fillet
  mid = lanes.lane_line(95.0, [(0.0, 0.0)])
  assert np.hypot(*(mid[0] - line[np.argmin(np.abs(np.hypot(*(line - mid[0]).T)))])) < 0.6


def test_fillets_near_the_route_ends():
  # routed from the car 12 m before the junction (Dutch London St), and ending 15 m past it: the corner has its fillet,
  # in what room there is, not the ways' sharp corner
  osm = junction_map()

  def smooth(line):
    d = np.diff(line, axis=0)
    d = d[np.hypot(*d.T) > 0.05]
    return float(np.degrees(np.abs(np.diff(np.unwrap(np.arctan2(d[:, 1], d[:, 0]))))).max())
  for pts in (np.array([(0.0, -12.0), NODES[3], NODES[4]]), np.array([NODES[1], NODES[2], NODES[3], (-15.0, 0.0)])):
    lanes = RouteLanes.from_osm(pts, ways_from_nodes(pts, osm), osm)
    assert len(lanes.corners) == 1
    line = lanes.lane_line(0.0, [(0.0, 0.0)])
    assert smooth(line) < 25.0 and np.hypot(*line.T).min() > 2.5, smooth(line)
  # the car past the corner's apex keeps the rest of its fillet
  pts = left_turn_route()
  lanes = RouteLanes.from_osm(pts, ways_from_nodes(pts, osm), osm)
  line = lanes.lane_line(0.0, [(0.0, 0.0)])
  past = lanes.lane_line(108.0, [(0.0, 0.0)])
  assert np.hypot(*(line - past[0]).T).min() < 0.3


def test_taper_lines():
  # the bay's way: its lines move with the lanes over TAPER_M from where it opens; the white line beside the opening
  # lane starts once it has opened (on the next way, here none)
  osm = junction_map()
  d, knots = osm.taper(11)
  assert d == 1 and [round(s) for s, _ in knots] == [0, 30]
  lines = osm.line_geometry(11)
  left = next(g for line, g in lines if line.kind == 'edge' and g[0, 0] < 0)
  assert np.allclose(left[0], (-3.5, -30.0), atol=0.01) and np.allclose(left[-1], (-5.25, 0.0), atol=0.01)
  assert len([line for line, _ in lines if line.kind == 'divider']) == 1  # between the old lanes only
  assert osm.taper(10) is None and osm.taper(13) is None


def test_explicit_taper():
  # a way whose bay widens along it (width:lanes:start / :end): no default taper, the bay opened at the way's end
  ways = {**WAYS, 11: ({**WAYS[11][0], 'width:lanes': '3.5|3.5|3.5', 'width:lanes:start': '0|3.5|3.5'}, [2, 3])}
  ids = np.array(sorted(NODES), np.int64)
  osm = OsmLanes(OsmData(ids, np.array([NODES[i][1] for i in ids]), np.array([NODES[i][0] for i in ids]), {}, ways, {}),
                 lambda lat, lon: (lon, lat))
  assert osm.has_ends(11) and osm.taper(10) is None
  assert [round(sec.spans[0].right - sec.spans[0].left, 2) for _, sec in osm.taper(11)[1]] == [0.0, 3.5]
  pts = left_turn_route()
  lanes = RouteLanes.from_osm(pts, ways_from_nodes(pts, osm), osm)
  assert 1 not in lanes.tapers and 1 in lanes.explicit
  mid = lanes.section_at(85.0, 1)
  assert abs(mid.ours[0].right - mid.ours[0].left - 1.75) < 0.01
  assert lanes.opening(1) == (100.0, 1, True)
  assert lanes.opened_at(85.0, 1).lanes == 2 and lanes.opened_at(100.0, 1).lanes == 3


def test_blend_where_a_road_carries_on():
  # a two-way road bending at node 2 from one way onto the next, measured wider there: its lanes move across over
  # BLEND_M either side of the node, its lines meet there mitred, and nav's lanes move with them
  nodes = {1: (0.0, -100.0), 2: (0.0, 0.0), 3: (30.0, 95.0)}
  ways = {20: ({'highway': 'primary', 'lanes': '2', 'width': '11'}, [1, 2]),
          21: ({'highway': 'primary', 'lanes': '2', 'width': '13'}, [3, 2])}  # drawn the other way
  ids = np.array(sorted(nodes), np.int64)
  osm = OsmLanes(OsmData(ids, np.array([nodes[i][1] for i in ids]), np.array([nodes[i][0] for i in ids]), {}, ways, {}),
                 lambda lat, lon: (lon, lat))
  assert osm.taper(20) is None and osm.blend(20)[1][0][0] == 0.0
  a = {line.offset: g for line, g in osm.line_geometry(20)}
  b = {line.offset: g for line, g in osm.line_geometry(21)}
  assert np.allclose(a[-5.5][-1], b[6.5][-1], atol=1e-6) and np.allclose(a[5.5][-1], b[-6.5][-1], atol=1e-6)
  assert np.allclose(a[0.0][-1], b[0.0][-1], atol=1e-6)
  assert 6.0 < np.hypot(*a[5.5][-1]) < 6.1  # halfway, mitred out a little at the bend
  assert np.allclose(a[-5.5][0], (-5.5, -100.0)) and np.allclose(a[-5.5][-5], (-5.5, -10.0))  # untouched 10 m back
  assert osm.edges_at(20, 85.0) == (-5.5, 5.5) and abs(osm.edges_at(20, 100.0)[1] - 6.0) < 1e-6
  pts = np.array([nodes[1], nodes[2], nodes[3]])
  lanes = RouteLanes.from_osm(pts, ways_from_nodes(pts, osm), osm)
  widths = [lanes.section_at(s).edges[1] for s in (85.0, 95.0, 100.0 - 1e-6)] + [lanes.section_at(100.0, 1).edges[1]]
  assert np.allclose(widths, [5.5, 5.5 + 0.15625, 6.0, 6.0], atol=1e-3), widths
  assert lanes.section_at(110.0 + 1e-3, 1).edges[1] == 6.5 and lanes.section_at(150.0, 1).lanes == 1


def test_placement_offset_runs_on():
  # a freeway's two links end to end, the second's lanes painted 0.6 m right of its line (placement:offset): no step at
  # the node; the lanes move across over BLEND_M either side of it, lines, kerbs and nav's lanes alike
  nodes = {1: (0.0, -100.0), 2: (0.0, 0.0), 3: (0.0, 100.0)}
  tags = {'highway': 'motorway', 'oneway': 'yes', 'lanes': '2', 'width': '12'}
  ways = {30: (tags, [1, 2]), 31: ({**tags, 'placement:offset': '0.6'}, [2, 3])}
  ids = np.array(sorted(nodes), np.int64)
  osm = OsmLanes(OsmData(ids, np.array([nodes[i][1] for i in ids]), np.array([nodes[i][0] for i in ids]), {}, ways, {}),
                 lambda lat, lon: (lon, lat))
  a = {round(line.offset, 2): g for line, g in osm.line_geometry(30)}
  b = {round(line.offset, 2): g for line, g in osm.line_geometry(31)}
  assert sorted(a) == [-6.0, 0.0, 6.0] and sorted(b) == [-5.4, 0.6, 6.6]
  for x, y in ((-6.0, -5.4), (0.0, 0.6), (6.0, 6.6)):
    assert np.allclose(a[x][-1], b[y][0], atol=1e-6) and abs(a[x][-1][0] - (x + 0.3)) < 1e-6  # met halfway at the node
    assert np.allclose(a[x][0], (x, -100.0)) and np.allclose(b[y][-1], (y, 100.0))  # each on its own paint beyond
  assert osm.edges_at(30, 89.0) == (-6.0, 6.0) and osm.edges_at(31, 11.0) == (-5.4, 6.6)
  pts = np.array([nodes[1], nodes[2], nodes[3]])
  lanes = RouteLanes.from_osm(pts, ways_from_nodes(pts, osm), osm)
  centre = [lanes.section_at(s, k).offset(0) for s, k in ((85.0, 0), (100.0 - 1e-6, 0), (100.0, 1), (115.0, 1))]
  assert np.allclose(centre, [-3.0, -2.7, -2.7, -2.4], atol=1e-3), centre

  def lines_with(extra):
    nodes4 = {**nodes, 4: (30.0, 50.0), 5: (-30.0, -50.0)}
    ids = np.array(sorted(nodes4), np.int64)
    osm = OsmLanes(OsmData(ids, np.array([nodes4[i][1] for i in ids]), np.array([nodes4[i][0] for i in ids]), {}, {**ways, **extra}, {}),
                   lambda lat, lon: (lon, lat))
    return {round(line.offset, 2): g[-1] for line, g in osm.line_geometry(30)}
  # a lane change leaving at the node (one-way, 31 degrees off): the freeway's links still run on into each other
  lane_change = {32: ({**tags, 'lanes': '1', 'width': '6'}, [2, 4])}
  assert all(np.allclose(lines_with(lane_change)[x], a[x][-1]) for x in a)
  # a two-way road joining there is a junction: no blend
  side = {33: ({'highway': 'primary', 'lanes': '2', 'width': '11'}, [5, 2])}
  assert np.allclose(lines_with(side)[-6.0], (-6.0, 0.0))


def test_route_lanes_bend_has_no_fillet():
  # a road bending through a node that isn't a junction keeps its own lane line
  osm = junction_map()
  pts = np.array([(0.0, -100.0), (0.0, -30.0), (0.0, 0.0), (-100.0, 0.0)])
  lanes = RouteLanes(pts, [Section.of(osm.lanes(10))] * 3)
  assert lanes.corners == []


def test_nav_targets_from_arrows():
  arrows = parse_arrows([[100.0, ['left', 'through', 'right;through']]])
  turn = Turn(110.0, 'left', 90.0)
  aim(turn, arrows)
  assert turn.targets == (0, 0, 3) and turn.lanes(3) == (0, 0)
  right = Turn(110.0, 'right', -90.0)
  aim(right, arrows)
  assert right.lanes(3) == (2, 2) and right.lanes(4) == (3, 3)  # counted from its side where the lanes differ
  none = Turn(200.0, 'left', 90.0)
  aim(none, arrows)
  assert none.targets is None and none.lanes(3) == (0, 0)
  route = np.array([(0.0, y) for y in np.arange(0.0, 300.0, 5.0)])
  ahead = throughs(route, parse_arrows([[100.0, ['left', 'through', 'through']]]), [])
  assert len(ahead) == 1 and isinstance(ahead[0], Through) and ahead[0].lanes(3) == (1, 2)
  keys = lane_plan(route, [], (0, 3), lambda d, after: 3, 10.0, arrows=[[100.0, ['left', 'through', 'through']]])
  assert keys[-1][1] == 1.0 and keys[-1][0] <= 100.0  # out of the left only lane before the junction
  assert lane_plan(route, [], (0, 3), lambda d, after: 3, 10.0) == [(0.0, 0.0)]


def lane_drop_map(arrows: str = 'left;through|through;right', out_tags: dict | None = None) -> OsmLanes:
  """A road north through a junction at (0, 0) with a road east: 2 lanes north and 1 south into it, and out of it 1
  north and 2 south (the southbound left turn lane), as wide: one direction's lane ends as the other's begins."""
  nodes = {1: (0.0, -100.0), 2: (0.0, 0.0), 3: (100.0, 0.0), 4: (0.0, 100.0)}
  split = {'highway': 'primary', 'lanes': '3', 'width': '16.5', 'placement:forward': 'left_of:1', 'placement:backward': 'left_of:1'}
  ways = {
    20: ({**split, 'lanes:forward': '2', 'lanes:backward': '1', 'turn:lanes:forward': arrows}, [1, 2]),
    21: ({**split, 'lanes:forward': '1', 'lanes:backward': '2'} if out_tags is None else out_tags, [2, 4]),
    22: ({'highway': 'primary', 'lanes': '2', 'width': '11'}, [2, 3]),
  }
  ids = np.array(sorted(nodes), np.int64)
  data = OsmData(ids, np.array([nodes[i][1] for i in ids]), np.array([nodes[i][0] for i in ids]), {}, ways, {})
  return OsmLanes(data, lambda lat, lon: (lon, lat))


def test_continuing_lanes():
  osm = lane_drop_map()
  into, out = Section.of(osm.lanes(20)), Section.of(osm.lanes(21))
  assert continuing(into, out) == [1]  # as wide: the kerbs carry on, so the right lane does
  narrower = Section.of(WayLanes.from_tags({'highway': 'primary', 'lanes': '2', 'width': '11'}))
  assert continuing(into, narrower) == [0]  # narrower: the line carries on, and the lane on the outside ends


def test_lane_drops():
  osm = lane_drop_map()
  pts = np.array([(0.0, -100.0), (0.0, 0.0), (0.0, 100.0)])
  assert RouteLanes.from_osm(pts, ways_from_nodes(pts, osm), osm).drops == [(100.0, (1, 1, 2))]
  turning = np.array([(0.0, -100.0), (0.0, 0.0), (100.0, 0.0)])  # the right turn onto the road east
  assert RouteLanes.from_osm(turning, ways_from_nodes(turning, osm), osm).drops == []
  only = lane_drop_map('left|through;right')  # a turn only lane: the arrows say where to go straight on
  assert RouteLanes.from_osm(pts, ways_from_nodes(pts, only), only).drops == []
  same = lane_drop_map(out_tags={'highway': 'primary', 'lanes': '3', 'lanes:forward': '2', 'lanes:backward': '1', 'width': '16.5',
                                 'placement:forward': 'left_of:1', 'placement:backward': 'left_of:1'})
  assert RouteLanes.from_osm(pts, ways_from_nodes(pts, same), same).drops == []  # no lane ends


def test_nav_aims_for_the_lanes_that_carry_on():
  route = np.array([(0.0, y) for y in np.arange(0.0, 300.0, 5.0)])
  ahead = throughs(route, [], [], [[100.0, 1, 1, 2]])
  assert len(ahead) == 1 and isinstance(ahead[0], Through) and ahead[0].lanes(2) == (1, 1)
  assert throughs(route, [], [Turn(110.0, 'left', 90.0)], [[100.0, 1, 1, 2]]) == []  # turning there
  # over to the lane that carries on, done by the last place a lane change for it starts, then the one lane past it
  keys = lane_plan(route, [], (0, 2), lambda d, after: 1 if after and d >= 100.0 else 2, 10.0, drops=[[100.0, 1, 1, 2]])
  assert keys == [(0.0, 0.0), (30.0, 0.0), (70.0, 1.0), (100.0, 1.0), (100.0, 0.0)]
  assert lane_plan(route, [], (0, 2), lambda d, after: 2, 10.0) == [(0.0, 0.0)]


def bay_map() -> OsmLanes:
  """A one-lane one-way road north whose left turn bay opens on its left from 100 m along over the next 30 m, into a
  junction with a road east at 200 m."""
  nodes = {1: (0.0, 0.0), 2: (0.0, 100.0), 3: (0.0, 130.0), 4: (0.0, 200.0), 5: (100.0, 200.0), 6: (0.0, 300.0)}
  one = {'highway': 'primary', 'oneway': 'yes', 'lanes': '1', 'width': '5.5'}
  two = {'highway': 'primary', 'oneway': 'yes', 'lanes': '2', 'width:lanes': '5.5|5.5', 'turn:lanes': 'left|through;right'}
  ways = {30: (one, [1, 2]), 31: ({**two, 'width:lanes:start': '0|5.5', 'width:lanes:end': '5.5|5.5'}, [2, 3]), 32: (two, [3, 4]),
          33: (one, [4, 5]), 34: (one, [4, 6])}
  ids = np.array(sorted(nodes), np.int64)
  data = OsmData(ids, np.array([nodes[i][1] for i in ids]), np.array([nodes[i][0] for i in ids]), {}, ways, {})
  return OsmLanes(data, lambda lat, lon: (lon, lat))


def test_lane_plan_keeps_its_lane_as_a_bay_opens():
  osm = bay_map()
  pts = np.array([(0.0, 0.0), (0.0, 100.0), (0.0, 130.0), (0.0, 200.0), (0.0, 300.0)])
  lanes = RouteLanes.from_osm(pts, ways_from_nodes(pts, osm), osm)
  assert lanes.openings == [(100.0, 1)]
  # the car's lane carries on as the right one: no change, and the line stays in it
  keys = lane_plan(pts, [], (0, 1), lambda d, after: 2 if d > 100.0 or (after and d >= 100.0) else 1, 10.0, opens=[[100.0, 1]])
  assert keys == [(0.0, 0.0), (100.0, 0.0), (100.0, 1.0)]
  line = lanes.lane_line(0.0, keys)
  # the way's line is the middle of the road as the bay opens on the left: the lane stays right of the bay, not in it
  at = [np.interp(y, line[:, 1], line[:, 0]) for y in (50.0, 115.0, 160.0)]
  assert np.allclose(at, [0.0, 1.375, 2.75], atol=0.05) and np.abs(np.diff(line[:, 0])).max() < 0.5
  # a fork the route takes right, past which the road carries on with all its lanes (GTA's bay link folded in): no renumbering
  assert lane_plan(pts, [[150.0, 'right', 1, 1, True, 1, False]], (1, 2), lambda d, after: 2, 10.0) ==     [(0.0, 1.0), (150.0, 1.0), (150.0, 1.0)]
  assert lane_plan(pts, [[150.0, 'right', 1, 1, True, 1, False]], (1, 2), lambda d, after: 1 if after else 2, 10.0)[-1] == (150.0, 0.0)


def test_keyed_lanes_step_where_keys_share_a_place():
  arriving, leaving = _keyed(np.array([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0]), np.array([0.0, 10.0, 10.0, 20.0]),
                             np.array([0.0, 1.0, 2.0, 0.0]))
  assert arriving.tolist() == [0.0, 0.0, 0.5, 1.0, 1.0, 0.0] and leaving.tolist() == [0.0, 0.0, 0.5, 2.0, 1.0, 0.0]


def divided_road_end() -> tuple[OsmLanes, np.ndarray]:
  """A divided road ending at node 3, (0, 0): the eastbound carriageway (two lanes, 7 m) along y = -3.5 whose last link
  angles across to the node, the westbound one leaving the node the same way back along y = 3.5, and the two-way road
  on east, two lanes each way, whose eastbound lanes sit where the carriageway's did (y -1.75 and -5.25)."""
  nodes = {1: (-80.0, -3.5), 2: (-15.0, -3.5), 3: (0.0, 0.0), 4: (100.0, 0.0), 5: (-15.0, 3.5), 6: (-80.0, 3.5)}
  one = {'highway': 'primary', 'oneway': 'yes', 'lanes': '2', 'width': '7'}
  ways = {1: (one, [1, 2]), 2: (one, [2, 3]), 3: ({'highway': 'primary', 'lanes': '4', 'width': '14'}, [3, 4]),
          4: (one, [3, 5]), 5: (one, [5, 6])}
  ids = np.array(sorted(nodes), np.int64)
  data = OsmData(ids, np.array([nodes[i][1] for i in ids]), np.array([nodes[i][0] for i in ids]), {}, ways, {})
  return OsmLanes(data, lambda lat, lon: (lon, lat)), np.array([nodes[1], nodes[2], nodes[3], nodes[4]])


def test_carriageway_lanes_run_on_into_the_road():
  # the carriageway's angled last link moves its lanes across to the road's (blend), so the lane line runs straight on
  # in the lane rather than following the link across and jogging back at the node
  osm, pts = divided_road_end()
  d, knots = osm.blend(2)
  at_node = knots[-1][1]
  cos = 15.0 / np.hypot(15.0, 3.5)
  assert d == 1 and np.allclose([sp.centre for sp in at_node.ours], [1.75 * cos, 5.25 * cos], atol=0.05)
  assert osm.blend(4) is not None  # the westbound carriageway's first link, parting from the road, likewise
  lanes = RouteLanes.from_osm(pts, ways_from_nodes(pts, osm), osm)
  line = lanes.lane_line(0.0, [(0.0, 1.0)])
  run = line[(line[:, 0] > -70.0) & (line[:, 0] < 60.0)]
  assert np.abs(run[:, 1] + 5.25).max() < 0.35, np.abs(run[:, 1] + 5.25).max()


def test_lane_line_curves_through_a_jog():
  # a jog the lanes can't blend across (the road's lanes elsewhere at a node of three roads): where the route kinks
  # within the move across, the lane line still turns smoothly
  s = np.arange(0.0, 81.0, 1.0)
  pts = np.column_stack([s, np.where(s < 40.0, 0.0, (s - 40.0) * 0.3)])  # the route bending 17 deg at 40 m
  line = np.column_stack([pts[:, 0], pts[:, 1] - np.where(s < 40.0, 0.0, 3.0)])
  out = _curve_jogs(line, s, [(30.0, 50.0, 0.0, 80.0)])
  d = np.diff(out, axis=0)
  turn = np.degrees(np.abs(np.diff(np.unwrap(np.arctan2(d[:, 1], d[:, 0])))))
  assert turn.max() < 3.0 and np.allclose(out[s <= 25.0], line[s <= 25.0]) and np.allclose(out[s >= 55.0], line[s >= 55.0])


def test_lane_line_eases_across_jogs():
  s = np.arange(0.0, 41.0, 1.0)
  offs = np.where(s < 20.0, 0.0, 5.0)
  eased = _ease_jogs(s, offs, [20.0])
  assert np.allclose(eased[s <= 10.0], 0.0) and np.allclose(eased[s >= 30.0], 5.0)
  assert np.allclose(eased[(s > 10.0) & (s < 30.0)], (s[(s > 10.0) & (s < 30.0)] - 10.0) / 4.0)


def pbf_bytes() -> bytes:
  """A tiny PBF, encoded by hand: three nodes (one tagged), a way and a relation."""
  def v(n):
    out = b''
    while True:
      out += bytes([(n & 0x7f) | (0x80 if n > 0x7f else 0)])
      n >>= 7
      if not n:
        return out

  def zz(n):
    return (n << 1) ^ (n >> 63)

  def ld(f, b):
    return v(f << 3 | 2) + v(len(b)) + b

  def vi(f, n):
    return v(f << 3) + v(n)

  def packed(xs):
    return b''.join(v(x) for x in xs)

  def deltas(xs):
    return packed(zz(b - a) for a, b in zip([0, *xs], xs, strict=False))

  strings = ['', 'highway', 'primary', 'ele', '12.5', 'type', 'connectivity', 'from', 'to']
  table = ld(1, b''.join(ld(1, s.encode()) for s in strings))
  ids, lats, lons = [5, 7, 1000000000000], [10, -20, 30], [-40, 50, 60]
  dense = ld(1, deltas(ids)) + ld(8, deltas(lats)) + ld(9, deltas(lons)) + ld(10, packed([0, 3, 4, 0, 0]))
  way = vi(1, 99) + ld(2, packed([1])) + ld(3, packed([2])) + ld(8, deltas([5, 7, 1000000000000]))
  rel = vi(1, 3) + ld(2, packed([5])) + ld(3, packed([6])) + ld(8, packed([7, 8])) + ld(9, deltas([99, 99])) + ld(10, packed([1, 1]))
  block = table + ld(2, ld(2, dense)) + ld(2, ld(3, way)) + ld(2, ld(4, rel))
  out = b''
  for kind, raw in ((b'OSMHeader', b''), (b'OSMData', block)):
    blob = vi(2, len(raw)) + ld(3, zlib.compress(raw))
    head = ld(1, kind) + vi(3, len(blob))
    out += struct.pack('>I', len(head)) + head + blob
  return out


def test_pbf():
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'tiny.osm.pbf')
    with open(path, 'wb') as f:
      f.write(pbf_bytes())
    data = osm_pbf.read(path)
    assert data.node_ids.tolist() == [5, 7, 1000000000000]
    assert np.allclose(data.lat, [1e-6, -2e-6, 3e-6]) and np.allclose(data.lon, [-4e-6, 5e-6, 6e-6])
    assert data.node_tags == {7: {'ele': '12.5'}}
    assert data.ways == {99: ({'highway': 'primary'}, [5, 7, 1000000000000])}
    assert data.relations == {3: ({'type': 'connectivity'}, [('w', 99, 'from'), ('w', 99, 'to')])}
    assert osm_pbf.read(path, relations=('restriction',)).relations == {}
    assert data.index([7, 8]).tolist() == [1, -1]


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
