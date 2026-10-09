"""junctions.py's junction geometry on the fixtures and on small maps made here: trims, areas, kerbs round the corners,
which roads make one arm or one junction, and stop lines. No pytest needed: `python test_junctions.py` runs them all."""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

from openpilot.tools.sim.bridge.gta5.map import through_paint
from openpilot.tools.sim.bridge.gta5.map.junctions import MERGE_GAP, STOP_SETBACK, Junctions, clip_outside, in_fan, lane_moves
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, FORWARD, OsmLanes
from openpilot.tools.sim.bridge.gta5.map.osm_pbf import OsmData
from openpilot.tools.sim.bridge.gta5.map.osm_to_roads import PaintAreas

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')
TWO_WAY = {'highway': 'residential', 'lanes': '2', 'width': '11'}
ONE_WAY = {'highway': 'primary', 'lanes': '2', 'oneway': 'yes', 'width': '8'}


def make(nodes: dict, ways: dict, node_tags: dict | None = None, relations: dict | None = None) -> OsmLanes:
  """A map in metres: nodes {id: (x, y)}, ways {id: (tags, [node ids])}, relations {id: (tags, [(type, ref, role)])}."""
  ids = np.array(sorted(nodes), np.int64)
  data = OsmData(ids, np.array([nodes[i][1] for i in ids], float), np.array([nodes[i][0] for i in ids], float),
                 node_tags or {}, dict(ways), relations or {})
  return OsmLanes(data, lambda lat, lon: (lon, lat))


def fixture(name: str) -> OsmLanes:
  root = ET.parse(os.path.join(FIXTURES, name)).getroot()
  scale = 111319.49  # m per degree at the equator, where the fixtures are
  nodes = {int(n.get('id')): (float(n.get('lon')) * scale, float(n.get('lat')) * scale) for n in root.iter('node')}
  ways = {int(w.get('id')): ({t.get('k'): t.get('v') for t in w.iter('tag')}, [int(nd.get('ref')) for nd in w.iter('nd')])
          for w in root.iter('way')}
  relations = {int(r.get('id')): ({t.get('k'): t.get('v') for t in r.iter('tag')},
                                  [(m.get('type')[0], int(m.get('ref')), m.get('role')) for m in r.iter('member')])
               for r in root.iter('relation')}
  return make(nodes, ways, relations=relations)


def only(js: Junctions):
  assert len(js.junctions) == 1, [j.nodes for j in js.junctions]
  return js.junctions[0]


def cross(arms: dict, into=(), relations: dict | None = None) -> OsmLanes:
  """A junction at node 1, (0, 0), with a way out to each (x, y): {way id: (tags, (x, y))}, those in `into` drawn
  towards it."""
  nodes = {1: (0.0, 0.0)}
  ways = {}
  for k, (wid, (tags, end)) in enumerate(arms.items()):
    nodes[10 + k] = end
    ways[wid] = (tags, [10 + k, 1] if wid in into else [1, 10 + k])
  return make(nodes, ways, relations=relations)


def test_fixture_crossroads():
  j = only(Junctions(fixture('turn_bay.osm')))
  assert j.nodes == [3] and len(j.arms) == 4
  assert in_fan([[0.0, 0.0]], j.centre, j.polygon).all()
  for arm in j.arms:
    assert 3.0 < arm.trim < 15.0, arm.trim
  for k, kerb in enumerate(j.kerbs):  # each corner's kerb runs from one arm's mouth to the next's, rounded
    assert np.allclose(kerb[0], j.arms[k].mouth()[1], atol=1e-6)
    assert np.allclose(kerb[-1], j.arms[(k + 1) % 4].mouth()[0], atol=1e-6)
    assert len(kerb) > 4
  # the corner kerbs keep out of the roads: every kerb point is at least a road's half width from both roads' lines
  for kerb in j.kerbs:
    assert (np.minimum(np.abs(kerb[:, 0]), np.abs(kerb[:, 1])) > 3.0).all()


def test_tee():
  osm = cross({1: (TWO_WAY, (-100.0, 0.0)), 2: (TWO_WAY, (100.0, 0.0)), 3: (TWO_WAY, (0.0, -100.0))})
  j = only(Junctions(osm))
  assert len(j.arms) == 3
  straight = [k for k in j.kerbs if len(k) == 2]
  assert len(straight) == 1 and np.allclose(straight[0][:, 1], 5.5)  # the far side's kerb runs straight across
  trims = {arm.members[0].ways[0][0]: arm.trim for arm in j.arms}
  assert trims[1] == trims[2] and trims[3] > 5.5


def test_narrow_fork():
  # a road forking 30 degrees apart: its branches are trimmed back further, to where their kerbs part
  a = math.radians(15)
  osm = cross({1: (TWO_WAY, (-100.0, 0.0)), 2: (TWO_WAY, (100 * math.cos(a), 100 * math.sin(a))),
               3: (TWO_WAY, (100 * math.cos(a), -100 * math.sin(a)))})
  j = only(Junctions(osm))
  trims = {arm.members[0].ways[0][0]: arm.trim for arm in j.arms}
  assert trims[2] > 20.0 and trims[3] > 20.0 and trims[1] < trims[2]


def test_turn_lane_beside_its_road():
  # a turn lane mapped as its own one-way way beside the road it leaves is one arm with it, its kerbs the outer ones
  nodes = {1: (0.0, 0.0), 2: (-100.0, 0.0), 3: (-100.0, 8.25), 4: (100.0, 0.0), 5: (0.0, 100.0), 6: (0.0, -100.0)}
  lane = {'highway': 'residential', 'lanes': '1', 'oneway': 'yes', 'width': '5.5', 'turn:lanes': 'left'}
  ways = {1: (TWO_WAY, [2, 1]), 2: (lane, [3, 1]), 3: (TWO_WAY, [1, 4]), 4: (TWO_WAY, [1, 5]), 5: (TWO_WAY, [1, 6])}
  j = only(Junctions(make(nodes, ways)))
  assert len(j.arms) == 4
  west = next(arm for arm in j.arms if len(arm.members) == 2)
  assert {m.ways[0][0] for m in west.members} == {1, 2}


def test_lanes_splitting_is_no_junction():
  # a two-way road whose directions part into one-way ways: three ways at a node, but only two roads
  nodes = {1: (0.0, 0.0), 2: (-100.0, 0.0), 3: (100.0, 3.0), 4: (100.0, -3.0)}
  ways = {1: (TWO_WAY, [2, 1]), 2: ({**ONE_WAY, 'lanes': '1', 'width': '5.5'}, [3, 1]),
          3: ({**ONE_WAY, 'lanes': '1', 'width': '5.5'}, [1, 4])}
  assert Junctions(make(nodes, ways)).junctions == []


def test_divided_road_is_one_junction():
  # a two-way road crossing a divided road's carriageways, 14 m apart: their trimmed ends overlap, so one junction
  # with the road between them inside it
  nodes = {1: (0.0, -7.0), 2: (0.0, 7.0), 3: (-100.0, -7.0), 4: (100.0, -7.0), 5: (100.0, 7.0), 6: (-100.0, 7.0),
           7: (0.0, -100.0), 8: (0.0, 100.0)}
  ways = {1: (ONE_WAY, [3, 1]), 2: (ONE_WAY, [1, 4]), 3: (ONE_WAY, [5, 2]), 4: (ONE_WAY, [2, 6]),
          5: (TWO_WAY, [7, 1]), 6: (TWO_WAY, [1, 2]), 7: (TWO_WAY, [2, 8])}
  j = only(Junctions(make(nodes, ways)))
  assert sorted(j.nodes) == [1, 2] and j.inside == {6} and len(j.arms) == 6
  assert in_fan([[0.0, 0.0]], j.centre, j.polygon).all()


def test_side_road_at_a_gap_in_the_median():
  # a side road from the north meeting a divided road whose carriageways are 20 m apart, a two-way road across the gap in
  # the median between them: one junction, its area square across both carriageways at the median's noses
  nodes = {1: (0.0, -10.0), 2: (0.0, 10.0), 3: (-100.0, -10.0), 4: (100.0, -10.0), 5: (100.0, 10.0), 6: (-100.0, 10.0),
           8: (0.0, 100.0)}
  ways = {1: (ONE_WAY, [3, 1]), 2: (ONE_WAY, [1, 4]), 3: (ONE_WAY, [5, 2]), 4: (ONE_WAY, [2, 6]),
          6: (TWO_WAY, [1, 2]), 7: (TWO_WAY, [2, 8])}
  js = Junctions(make(nodes, ways))
  j = only(js)
  assert sorted(j.nodes) == [1, 2] and j.inside == {6} and len(j.arms) == 5
  assert js.trims[(1, 1)] > 1.0 and abs(js.trims[(1, 1)] - js.trims[(4, 2)]) < 0.01
  assert js.trims[(2, 1)] > 1.0 and abs(js.trims[(2, 1)] - js.trims[(3, 2)]) < 0.01


def test_junctions_apart_are_trimmed_to_fit():
  # two crossroads of wide roads 22 m apart: kept apart, each trimmed back no further than leaves a gap between them
  wide = {'highway': 'primary', 'lanes': '4', 'width': '20'}
  nodes = {1: (0.0, 0.0), 2: (22.0, 0.0), 3: (-100.0, 0.0), 4: (122.0, 0.0), 5: (0.0, 100.0), 6: (0.0, -100.0),
           7: (22.0, 100.0), 8: (22.0, -100.0)}
  ways = {1: (wide, [3, 1]), 2: (wide, [1, 2]), 3: (wide, [2, 4]), 4: (wide, [1, 5]), 5: (wide, [1, 6]),
          6: (wide, [2, 7]), 7: (wide, [2, 8])}
  js = Junctions(make(nodes, ways))
  assert len(js.junctions) == 2
  assert js.trims[(2, 1)] + js.trims[(2, 2)] <= 22.0 - MERGE_GAP + 0.01


def test_stop_lines():
  # signals 20 m south of a crossroads, for traffic heading north (the way into it drawn towards it): across the lanes
  # on the right arriving, at the node; a crossing just before the junction moves it back behind the crossing
  nodes = {1: (0.0, 0.0), 2: (0.0, -20.0), 3: (0.0, -100.0), 4: (-100.0, 0.0), 5: (100.0, 0.0), 6: (0.0, 100.0),
           20: (-8.0, -18.0), 21: (8.0, -18.0)}
  ways = {1: (TWO_WAY, [3, 2]), 2: (TWO_WAY, [2, 1]), 3: (TWO_WAY, [1, 4]), 4: (TWO_WAY, [1, 5]), 5: (TWO_WAY, [1, 6]),
          9: ({'highway': 'footway', 'footway': 'crossing'}, [20, 21])}
  signals = {2: {'highway': 'traffic_signals', 'traffic_signals:direction': 'forward'}}
  j = only(Junctions(make(nodes, ways, signals)))
  assert len(j.stops) == 1
  stop = j.stops[0]
  assert stop.kind == 'stop' and np.allclose(sorted(stop.line[:, 0]), [0.0, 5.5])
  assert stop.along > 18.0 + 1.5 and stop.along >= stop.member.trim + STOP_SETBACK
  backward = {2: {'highway': 'traffic_signals', 'traffic_signals:direction': 'backward'}}  # for traffic leaving it
  assert only(Junctions(make(nodes, ways, backward))).stops == []
  on_node = {1: {'highway': 'traffic_signals'}}  # mapped on the junction's node: every way in, at its mouth
  j = only(Junctions(make(nodes, ways, on_node)))
  at_mouth = [abs(s.along - (s.member.trim + STOP_SETBACK)) < 1e-6 for s in j.stops]
  assert len(j.stops) == 4 and sum(at_mouth) == 3  # the south one behind its crossing


def test_stop_line_between_two_junctions():
  # a stop sign between two crossroads 60 m apart, for traffic heading north, at a node the ways either side are both
  # drawn out of: its direction (forward) faces both junctions by one way or the other; the line goes only on the
  # approach to the junction whose mouth is nearer (north), not across the southbound lanes to the south one too
  nodes = {1: (0.0, 0.0), 2: (0.0, -12.0), 3: (0.0, -60.0), 4: (-100.0, 0.0), 5: (100.0, 0.0), 6: (0.0, 100.0),
           7: (-100.0, -60.0), 8: (100.0, -60.0), 9: (0.0, -160.0)}
  ways = {1: (TWO_WAY, [2, 1]), 2: (TWO_WAY, [2, 3]), 3: (TWO_WAY, [1, 4]), 4: (TWO_WAY, [1, 5]), 5: (TWO_WAY, [1, 6]),
          6: (TWO_WAY, [3, 7]), 7: (TWO_WAY, [3, 8]), 8: (TWO_WAY, [3, 9])}
  js = Junctions(make(nodes, ways, {2: {'highway': 'stop', 'direction': 'forward'}}))
  stops = [(j.nodes, s) for j in js.junctions for s in j.stops]
  assert len(stops) == 1 and stops[0][0] == [1], [(n, s.line.tolist()) for n, s in stops]
  assert np.allclose(sorted(stops[0][1].line[:, 0]), [0.0, 5.5])  # the northbound lanes


def test_lane_change_cutting_across_is_no_junction():
  # two one-way carriageways side by side, and GTA's lane change from one to the other cutting across at 55 degrees:
  # the traffic all runs one way, so no junction where it leaves or joins (its own heading would spread the flows
  # past MERGE_FLOW)
  lane = {'highway': 'primary', 'lanes': '1', 'oneway': 'yes', 'width': '4'}
  q = (18.0 * math.cos(math.radians(55.0)), 18.0 * math.sin(math.radians(55.0)))
  nodes = {1: (-100.0, 0.0), 2: (0.0, 0.0), 3: (100.0, 0.0), 4: (-100.0, q[1]), 5: q, 6: (100.0, q[1])}
  ways = {1: (lane, [1, 2]), 2: (lane, [2, 3]), 3: (lane, [4, 5]), 4: (lane, [5, 6]), 5: (lane, [2, 5])}
  js = Junctions(make(nodes, ways))
  assert js.lane_changes == {5} and js.junctions == [], [j.nodes for j in js.junctions]


def test_slip_triangle_is_one_junction():
  # a side road from the east (node 3) meets a main road running north-west to south-east through nodes 1 and 2, 25 m
  # apart, by a one-way slip into it at 1 and one out of it at 2, as GTA lays a triangle where a side road joins: one
  # junction of the three nodes, the triangle's roads inside it, three arms (the main road either way, the side road)
  one = {'highway': 'residential', 'lanes': '1', 'oneway': 'yes', 'width': '5.5'}
  nodes = {1: (-23.0, 10.0), 2: (0.0, 0.0), 3: (9.0, 14.0), 4: (-100.0, 43.0), 5: (90.0, -38.0), 6: (100.0, 14.0)}
  ways = {1: (TWO_WAY, [4, 1]), 2: (TWO_WAY, [1, 2]), 3: (TWO_WAY, [2, 5]), 4: (one, [3, 1]), 5: (one, [2, 3]),
          6: (TWO_WAY, [3, 6])}
  j = only(Junctions(make(nodes, ways)))
  assert sorted(j.nodes) == [1, 2, 3] and len(j.arms) == 3 and j.inside == {2, 4, 5}, (j.nodes, j.inside)


def test_turn_bay_beside_a_two_way_road_is_no_junction():
  # an eastbound turn bay laid as its own one-way way: it leaves the two-way road's middle at node 2 at 30 degrees to
  # its right, runs beside it 7 m out, and a lane change from it rejoins the road at node 5 (as GTA lays them): lanes
  # parting and merging on the eastbound side, no traffic crossing, so no junctions along the approach
  one = {'highway': 'residential', 'lanes': '1', 'oneway': 'yes', 'width': '5.5'}
  road = {'highway': 'primary', 'lanes': '4', 'width': '18'}
  nodes = {1: (-100.0, 0.0), 2: (0.0, 0.0), 3: (12.0, -7.0), 4: (40.0, -7.0), 5: (50.0, 0.0), 6: (150.0, 0.0), 7: (90.0, -7.0)}
  ways = {1: (road, [1, 2]), 2: (road, [2, 5]), 3: (road, [5, 6]), 4: (one, [2, 3]), 5: (one, [3, 4]), 6: (one, [4, 5]),
          7: (one, [4, 7])}
  assert not Junctions(make(nodes, ways)).junctions
  # the bay leaving to the left instead, across the westbound lanes: a turn, a junction
  mirrored = {n: (x, -y) for n, (x, y) in nodes.items()}
  assert 2 in {n for j in Junctions(make(mirrored, ways)).junctions for n in j.nodes}


def test_driveway_off_a_main_road_is_minor():
  # a driveway (service road, 5.5 m) off a 20 m main road: its junction is minor, the main road's lines carried on
  # across it; a residential side road as wide is not
  road = {'highway': 'primary', 'lanes': '4', 'width': '20'}
  nodes = {1: (-100.0, 0.0), 2: (0.0, 0.0), 3: (100.0, 0.0), 4: (0.0, -60.0)}
  for side, minor in (({'highway': 'service', 'lanes': '1', 'width': '5.5'}, True), (TWO_WAY, False)):
    j = only(Junctions(make(nodes, {1: (road, [1, 2]), 2: (road, [2, 3]), 3: (side, [2, 4])})))
    assert j.minor == minor and (set(j.carried) >= {1, 2}) == minor, (side, j.minor, j.carried)


def test_surveyed_stop_lines():
  # a stop line surveyed where it's painted (source:position=survey), 8 m from the crossroads' node, nearer than the
  # road would be trimmed: the road ends there and the line is drawn at it, not moved out behind the crossing
  nodes = {1: (0.0, 0.0), 2: (0.0, -20.0), 3: (0.0, -100.0), 4: (-100.0, 0.0), 5: (100.0, 0.0), 6: (0.0, 100.0),
           7: (0.0, -8.0), 20: (-8.0, -6.0), 21: (8.0, -6.0)}
  ways = {1: (TWO_WAY, [3, 2]), 2: (TWO_WAY, [2, 7]), 6: (TWO_WAY, [7, 1]), 3: (TWO_WAY, [1, 4]), 4: (TWO_WAY, [1, 5]),
          5: (TWO_WAY, [1, 6]), 9: ({'highway': 'footway', 'footway': 'crossing'}, [20, 21])}
  tags = {'highway': 'traffic_signals', 'traffic_signals:direction': 'forward'}
  plain = only(Junctions(make(nodes, ways, {7: tags})))
  assert plain.stops[0].along > 8.0 + 1.0  # unsurveyed: no nearer than the mouth, behind the crossing
  j = only(Junctions(make(nodes, ways, {7: {**tags, 'source:position': 'survey'}})))
  stop = j.stops[0]
  assert len(j.stops) == 1 and abs(stop.along - 8.0) < 1e-6 and stop.member.trim <= 8.0 - STOP_SETBACK + 1e-6
  assert all(m.trim > 8.0 for arm in j.arms for m in arm.members if m is not stop.member)


def test_surveyed_stop_line_level_with_the_road_it_meets():
  # a skewed T whose side road's stop line is painted 4 m from the node, inside the main road's 5.5 m to its kerb: the
  # junction is shaped as without it (its corners round, no kerb runs out across the main road), and the line is drawn
  # where it's painted, along the main road's edge rather than square across the skewed side road
  side = (-math.sin(math.radians(20.0)), -math.cos(math.radians(20.0)))  # 20 degrees off square
  nodes = {1: (0.0, 0.0), 2: (-100.0, 0.0), 3: (100.0, 0.0), 4: (100.0 * side[0], 100.0 * side[1]), 7: (4.0 * side[0], 4.0 * side[1])}
  ways = {1: (TWO_WAY, [2, 1]), 2: (TWO_WAY, [1, 3]), 3: (TWO_WAY, [4, 7]), 4: (TWO_WAY, [7, 1])}
  plain = only(Junctions(make(nodes, ways)))
  j = only(Junctions(make(nodes, ways, {7: {'highway': 'stop', 'direction': 'forward', 'source:position': 'survey'}})))
  stop = j.stops[0]
  assert len(j.stops) == 1 and abs(stop.along - 4.0) < 1e-6
  assert sorted(arm.trim for arm in j.arms) == sorted(arm.trim for arm in plain.arms) and stop.member.trim > 5.5
  d = stop.line[1] - stop.line[0]
  assert abs(d[1]) < 0.02 * abs(d[0]), stop.line  # along the main road
  assert np.allclose(np.mean(stop.line, axis=0)[1], -4.0 * math.cos(math.radians(20.0)), atol=0.05)  # at the paint


def test_directed_signal_on_a_junction_node_is_one_approach_s():
  # GTA's light for the road from the south sits on the node where a slip parts from it, a junction of its own: that
  # junction's ways in aren't all stopped there (an undirected one on a junction's node stops them all)
  nodes = {1: (0.0, 0.0), 2: (0.0, -40.0), 3: (0.0, -100.0), 4: (-100.0, 0.0), 5: (100.0, 0.0), 6: (0.0, 100.0),
           7: (30.0, -10.0)}
  ways = {1: (TWO_WAY, [3, 2]), 2: (TWO_WAY, [2, 1]), 3: (TWO_WAY, [1, 4]), 4: (TWO_WAY, [1, 5]), 5: (TWO_WAY, [1, 6]),
          6: (TWO_WAY, [2, 7]), 7: (TWO_WAY, [7, 5])}
  signal = {'highway': 'traffic_signals', 'traffic_signals:direction': 'forward'}
  js = Junctions(make(nodes, ways, {2: signal}))
  at_2 = next(j for j in js.junctions if 2 in j.nodes)
  main = next(j for j in js.junctions if 1 in j.nodes)
  assert at_2.stops == [] and len(main.stops) == 1
  js = Junctions(make(nodes, ways, {2: {'highway': 'traffic_signals'}}))
  assert len(next(j for j in js.junctions if 2 in j.nodes).stops) == 3


def test_through_road_carried_across_a_slip_triangle():
  # a main road east-west through a slip triangle's two corners (one junction of three nodes), the side road from the
  # north meeting it by two one-way slips: the main road is carried on through along its way inside, where it's a
  # priority road
  side = {'highway': 'residential', 'lanes': '1', 'oneway': 'yes', 'width': '4'}
  nodes = {1: (-100.0, 0.0), 2: (-15.0, 0.0), 3: (15.0, 0.0), 4: (100.0, 0.0), 5: (0.0, 15.0), 6: (0.0, 100.0)}
  main = {**TWO_WAY, 'divider': 'double_solid_line', 'priority_road': 'yes_unposted'}
  ways = {1: (main, [1, 2]), 2: (main, [2, 3]), 3: (main, [3, 4]), 4: (TWO_WAY, [6, 5]), 5: (side, [5, 2]), 6: (side, [3, 5])}
  j = only(Junctions(make(nodes, ways)))
  assert 2 in j.inside and set(j.through) == {1, 2, 3} and set(j.carried) >= {1, 2, 3}


def test_corner_ends_at_a_painted_island_tip():
  # two roads parting at 20 degrees with a painted gore between them (traffic_calming=painted_island) from 6 m out:
  # the corner between them ends at the gore's tip, not 40 m back where their kerbs meet
  a = (-100.0 * math.cos(math.radians(80.0)), -100.0 * math.sin(math.radians(80.0)))
  b = (100.0 * math.cos(math.radians(80.0)), -100.0 * math.sin(math.radians(80.0)))
  nodes = {1: (0.0, 0.0), 2: (0.0, 100.0), 3: a, 4: b}
  ways = {1: (TWO_WAY, [2, 1]), 2: (TWO_WAY, [1, 3]), 3: (TWO_WAY, [1, 4])}
  plain = only(Junctions(make(nodes, ways)))
  tip = (0.0, -20.0)
  ring = {11: tip, 12: (-4.0, -40.0), 13: (4.0, -40.0)}
  island = {**nodes, **ring}
  with_island = {**ways, 9: ({'area': 'yes', 'traffic_calming': 'painted_island'}, [11, 12, 13, 11])}
  j = only(Junctions(make(island, with_island)))
  trims = sorted(arm.trim for arm in j.arms)
  assert max(arm.trim for arm in plain.arms) > 30.0 and 15.0 < trims[-1] < 25.0, (trims, [arm.trim for arm in plain.arms])


def test_crossing_lines_on_the_road():
  nodes = {1: (0.0, 0.0), 2: (0.0, -100.0), 3: (-100.0, 0.0), 4: (100.0, 0.0), 20: (-10.0, -20.0), 21: (10.0, -20.0)}
  ways = {1: (TWO_WAY, [2, 1]), 2: (TWO_WAY, [3, 1]), 3: (TWO_WAY, [1, 4]),
          9: ({'highway': 'footway', 'footway': 'crossing'}, [20, 21])}
  lines = Junctions(make(nodes, ways)).crossing_lines()
  assert len(lines) == 1 and np.allclose(sorted(lines[0][[0, -1], 0]), [-5.5, 5.5])


def moves(js: Junctions, j) -> dict[tuple[int, int], list]:
  """{(way in, way out): [(lane in, lane out, kind)]}"""
  out: dict[tuple[int, int], list] = {}
  for mv in js.movements(j):
    out.setdefault((mv.into.ways[0][0], mv.out.ways[0][0]), []).append((mv.lane_in, mv.lane_out, mv.kind))
  return out


def check_paths(js: Junctions, j):
  """Every move's path starts on its lane's centre arriving and ends on its lane's centre leaving, heading along them,
  and runs smoothly (no step turning more than 25 degrees), mostly inside the junction's area or its approaches."""
  for mv in js.movements(j):
    p = mv.path
    for m, lane, at, arriving in ((mv.into, mv.lane_in, p[0], True), (mv.out, mv.lane_out, p[-1], False)):
      w, fwd = m.ways[0]
      spans = js.osm.lanes(w).ours((BACKWARD if fwd else FORWARD) if arriving else (FORWARD if fwd else BACKWARD))
      s = m.line.project(at)
      u = m.line.tangent(s)
      left = np.array([-u[1], u[0]])
      off = float((at - m.line.at(s)) @ left)  # m left of the road looking out
      want = spans[lane].centre if arriving else -spans[lane].centre
      assert abs(off - want) < 0.05, (off, want)
    d = np.diff(p, axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    assert np.all(np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi) < np.radians(25)), mv.kind


def test_movements_turn_bay():
  # turn_bay.osm: from the west, lanes left|through; the bay is connected to the road north (connectivity 1:1), the
  # right turn south is banned (no_right_turn), and the through lane goes east
  js = Junctions(fixture('turn_bay.osm'))
  j = only(js)
  got = moves(js, j)
  assert got[(2, 4)] == [(0, 0, 'left')]  # the connectivity relation's
  assert got[(2, 3)] == [(1, 0, 'through')]
  assert (2, 5) not in got  # restricted
  assert all(a != b for a, b in got)  # no U-turns
  assert all(len(ms) == 1 for ms in got.values())  # one lane in on each move
  assert {w for w, _ in got if w != 2} == {3, 4, 5} and len(got) == 2 + 3 * 3
  check_paths(js, j)


def test_movements_untagged_multilane():
  # a one-way road north into a crossroads with three lanes and no arrows: the left lane turns left too, the right one
  # right, the middle one goes through; at a T with no way through, half go each way
  wide = {'highway': 'primary', 'lanes': '3', 'oneway': 'yes', 'width': '10.5'}
  osm = cross({1: (wide, (0.0, -100.0)), 2: (TWO_WAY, (-100.0, 0.0)), 3: (TWO_WAY, (100.0, 0.0)), 4: (TWO_WAY, (0.0, 100.0))},
              into={1})
  js = Junctions(osm)
  j = only(js)
  got = moves(js, j)
  assert got[(1, 2)] == [(0, 0, 'left')] and got[(1, 3)] == [(2, 0, 'right')]
  assert sorted(got[(1, 4)]) == [(0, 0, 'through'), (1, 0, 'through'), (2, 0, 'through')]
  check_paths(js, j)
  assert lane_moves([frozenset()] * 2, [(90.0, 1), (-90.0, 1)]) == [[(0, 0)], [(1, 0)]]


def test_movements_turn_lanes_and_restrictions():
  # arrows left|through|through;right into two-lane exits: the left lane to the left road's left lane, the through
  # lanes in order, the right lane also right; an only_straight_on restriction leaves only the through moves
  arrows = {**ONE_WAY, 'lanes': '3', 'width': '12', 'turn:lanes': 'left|through|through;right'}
  arms = {1: (arrows, (0.0, -100.0)), 2: (ONE_WAY, (-100.0, 0.0)), 3: (ONE_WAY, (100.0, 0.0)), 4: (ONE_WAY, (0.0, 100.0))}
  js = Junctions(cross(arms, into={1}))
  j = only(js)
  got = moves(js, j)
  assert got[(1, 2)] == [(0, 0, 'left')]
  assert got[(1, 4)] == [(1, 0, 'through'), (2, 1, 'through')]
  assert got[(1, 3)] == [(2, 1, 'right')]
  check_paths(js, j)
  only_on = {1: ({'type': 'restriction', 'restriction': 'only_straight_on'}, [('w', 1, 'from'), ('n', 1, 'via'), ('w', 4, 'to')])}
  js2 = Junctions(cross(arms, into={1}, relations=only_on))
  assert set(moves(js2, only(js2))) == {(1, 4)}


def test_movements_divided_road():
  # a junction of two nodes (the divided road test's): every move between its roads keeps to the one-way rules of the
  # road between its nodes, and none turns back into the carriageway it came from
  nodes = {1: (0.0, -7.0), 2: (0.0, 7.0), 3: (-100.0, -7.0), 4: (100.0, -7.0), 5: (100.0, 7.0), 6: (-100.0, 7.0),
           7: (0.0, -100.0), 8: (0.0, 100.0)}
  ways = {1: (ONE_WAY, [3, 1]), 2: (ONE_WAY, [1, 4]), 3: (ONE_WAY, [5, 2]), 4: (ONE_WAY, [2, 6]),
          5: (TWO_WAY, [7, 1]), 6: (TWO_WAY, [1, 2]), 7: (TWO_WAY, [2, 8])}
  js = Junctions(make(nodes, ways))
  j = only(js)
  got = moves(js, j)
  assert (1, 4) not in got and (3, 2) not in got  # U-turns
  assert (1, 2) in got and (1, 7) in got and (5, 7) in got and (5, 4) in got and (3, 4) in got
  check_paths(js, j)


def test_bridge_from_a_junction():
  # a bridge starting at a junction: its area, on the bridge's layer, cuts the lines of the ground roads meeting it too,
  # but not those of a road on the ground's layer passing under it
  bridge = {**TWO_WAY, 'bridge': 'yes', 'layer': '1'}
  nodes = {1: (0.0, 0.0), 2: (-100.0, 0.0), 3: (0.0, -100.0), 4: (100.0, 0.0), 5: (-50.0, 2.0), 6: (50.0, 2.0)}
  ways = {1: (TWO_WAY, [1, 2]), 2: (TWO_WAY, [1, 3]), 3: (bridge, [1, 4]), 9: (TWO_WAY, [5, 6])}
  js = Junctions(make(nodes, ways))
  j = only(js)
  paint = PaintAreas(js)
  assert paint.layer == [1]
  line = np.array([[0.0, 0.0], [-100.0, 0.0]])
  assert any(a is j.polygon for _, a in paint.near(line, 0, {1}))
  assert any(a is j.polygon for _, a in paint.near(line, 1, {7}))
  assert not any(a is j.polygon for _, a in paint.near(line, 0, {9}))


def test_slanted_roads_lines_end_in_the_area():
  # a divided road's carriageways meeting a wide road at a slant (Eclipse Blvd): the wide road is trimmed back little,
  # and the carriageways' lines' ends at the node, square across their own line, reach out past its mouth; they're cut
  # there (PaintAreas' node_ends), the wide road's own lines aren't
  wide = {'highway': 'primary', 'lanes': '5', 'lanes:forward': '3', 'lanes:backward': '2', 'width': '21.4'}
  half = {**ONE_WAY, 'width': '8.8'}
  nodes = {1: (0.0, 0.0), 2: (100.0, 0.0), 3: (-82.0, 57.0), 4: (-93.0, -36.0)}
  ways = {1: (wide, [1, 2]), 2: (half, [1, 3]), 3: (half, [4, 1])}
  osm = make(nodes, ways)
  js = Junctions(osm)
  only(js)
  paint = PaintAreas(js)
  for wid in (2, 3):
    for line, geom in osm.line_geometry(wid):
      for piece in clip_outside(geom, paint.near(geom, 0, {wid}, kerbs_only=line.kind == 'edge')):
        assert piece[:, 0].max() < 0.3, (wid, line.kind, piece)
  for _line, geom in osm.line_geometry(1):
    kept = clip_outside(geom, paint.near(geom, 0, {1}))
    assert kept and min(piece[:, 0].min() for piece in kept) < 1.0  # from its mouth on


SIDE = {'highway': 'service', 'lanes': '1', 'width': '5', 'lane_markings': 'no'}


PRIORITY = {**TWO_WAY, 'priority_road': 'yes_unposted'}


def test_carried_through_a_side_road():
  # a priority road carried straight on past a side road keeps its centre line across the junction (both ways,
  # whichever way they're drawn); its area still cuts the side road's lines and the road's kerbs
  arms = {1: (PRIORITY, (-100.0, 0.0)), 2: (PRIORITY, (100.0, 3.0)), 3: (SIDE, (0.0, 100.0))}
  js = Junctions(cross(arms, into={1}))
  j = only(js)
  assert j.through == j.carried == {1: {0.0}, 2: {0.0}}
  paint = PaintAreas(js)
  line = np.array([[0.0, 0.0], [-100.0, 0.0]])
  assert not any(a is j.polygon for _, a in paint.near(line, 0, {1}, offset=0.0))
  assert any(a is j.polygon for _, a in paint.near(line, 0, {1}, offset=2.75))  # not one of its lines
  assert any(a is j.polygon for _, a in paint.near(line, 0, {1}, kerbs_only=True))
  assert any(a is j.polygon for _, a in paint.near(line, 0, {3}, offset=0.0))
  plain = only(Junctions(cross({**arms, 2: (TWO_WAY, (100.0, 3.0))}, into={1})))  # a priority road one side only
  assert plain.through == {1: {0.0}, 2: {0.0}} and plain.carried == {}


def test_carried_on_past_a_short_way():
  # the road's first way out west is 3 m long and a side road meets at a skew, so the area reaches into the next way:
  # its lines are carried across too
  nodes = {1: (0.0, 0.0), 20: (-3.0, 0.0), 10: (-100.0, 0.0), 11: (100.0, 0.0), 12: (-60.0, 60.0)}
  ways = {1: (PRIORITY, [1, 20]), 4: (PRIORITY, [20, 10]), 2: (PRIORITY, [1, 11]), 3: (SIDE, [1, 12])}
  j = only(Junctions(make(nodes, ways)))
  assert j.through == {1: {0.0}, 2: {0.0}} and j.carried == {1: {0.0}, 2: {0.0}, 4: {0.0}}


def test_through_only_by_a_road_with_the_way():
  arms = {1: (PRIORITY, (-100.0, 0.0)), 2: (PRIORITY, (100.0, 0.0)), 3: (SIDE, (0.0, 100.0))}
  signals = cross(arms)
  signals.data.node_tags[1] = {'highway': 'traffic_signals'}
  assert only(Junctions(signals)).through == {}
  nodes = {1: (0.0, 0.0), 10: (-100.0, 0.0), 11: (100.0, 0.0), 12: (0.0, 100.0), 99: (5.0, 0.0)}
  ways = {1: (PRIORITY, [10, 1]), 2: (PRIORITY, [1, 99, 11]), 3: (SIDE, [1, 12])}
  assert only(Junctions(make(nodes, ways, {99: {'highway': 'stop'}}))).through == {}  # a stop sign on the road, 5 m out
  equals = {**arms, 3: (PRIORITY, (0.0, 100.0)), 4: (PRIORITY, (0.0, -100.0))}
  assert only(Junctions(cross(equals))).through == {}  # a crossroads of equals: neither has the way
  minor = {**equals, 3: (SIDE, (0.0, 100.0)), 4: ({**TWO_WAY, 'highway': 'service'}, (0.0, -100.0))}
  assert only(Junctions(cross(minor))).carried == {1: {0.0}, 2: {0.0}}
  layouts = {**arms, 2: ({**PRIORITY, 'lanes': '3', 'lanes:forward': '2', 'lanes:backward': '1', 'width': '16'}, (100.0, 0.0))}
  assert only(Junctions(cross(layouts))).through == {}  # its lines don't meet across the junction


def test_painted_across():
  # through_paint tags the road carried through as a priority road where the game paints its lines across the junction
  class Paint:
    def __init__(self, ys):
      self.ys = ys

    def near(self, p, z):
      return any(abs(p[1] - y) < 0.6 for y in self.ys)
  osm = cross({1: (TWO_WAY, (-100.0, 0.0)), 2: (TWO_WAY, (100.0, 0.0)), 3: (SIDE, (0.0, 100.0))})
  for n in (1, 10, 11, 12):
    osm.data.node_tags[n] = {'ele': '0'}
  js = Junctions(osm)
  assert through_paint.painted_across(osm, js, Paint([0.0]))[0] == {1, 2}
  assert through_paint.painted_across(osm, js, Paint([]))[0] == set()  # none painted across


def test_clip_outside():
  square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])
  pieces = clip_outside(np.array([[-3.0, 0.0], [3.0, 0.0]]), [(np.zeros(2), square)])
  assert len(pieces) == 2 and np.allclose(pieces[0], [[-3, 0], [-1, 0]]) and np.allclose(pieces[1], [[1, 0], [3, 0]])
  folded = np.array([[2.0, 0.0], [0.0, 2.0], [-2.0, 0.0], [-1.0, 0.0], [-2.0, 0.5]])  # folds back on itself
  assert in_fan([[-1.5, 0.1], [0.0, 1.0]], np.zeros(2), folded).all()


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
