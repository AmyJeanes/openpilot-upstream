"""junctions.py's junction geometry on the fixtures and on small maps made here: trims, areas, kerbs round the corners,
which roads make one arm or one junction, and stop lines. No pytest needed: `python test_junctions.py` runs them all."""
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.junctions import MERGE_GAP, STOP_SETBACK, Junctions, clip_outside, in_fan
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes
from openpilot.tools.sim.bridge.gta5.map.osm_pbf import OsmData

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')
TWO_WAY = {'highway': 'residential', 'lanes': '2', 'width': '11'}
ONE_WAY = {'highway': 'primary', 'lanes': '2', 'oneway': 'yes', 'width': '8'}


def make(nodes: dict, ways: dict, node_tags: dict | None = None) -> OsmLanes:
  """A map in metres: nodes {id: (x, y)}, ways {id: (tags, [node ids])}."""
  ids = np.array(sorted(nodes), np.int64)
  data = OsmData(ids, np.array([nodes[i][1] for i in ids], float), np.array([nodes[i][0] for i in ids], float),
                 node_tags or {}, dict(ways), {})
  return OsmLanes(data, lambda lat, lon: (lon, lat))


def fixture(name: str) -> OsmLanes:
  root = ET.parse(os.path.join(FIXTURES, name)).getroot()
  scale = 111319.49  # m per degree at the equator, where the fixtures are
  nodes = {int(n.get('id')): (float(n.get('lon')) * scale, float(n.get('lat')) * scale) for n in root.iter('node')}
  ways = {int(w.get('id')): ({t.get('k'): t.get('v') for t in w.iter('tag')}, [int(nd.get('ref')) for nd in w.iter('nd')])
          for w in root.iter('way')}
  return make(nodes, ways)


def only(js: Junctions):
  assert len(js.junctions) == 1, [j.nodes for j in js.junctions]
  return js.junctions[0]


def cross(arms: dict) -> OsmLanes:
  """A junction at (0, 0) with a way out to each (x, y): {way id: (tags, (x, y))}."""
  nodes = {1: (0.0, 0.0)}
  ways = {}
  for k, (wid, (tags, end)) in enumerate(arms.items()):
    nodes[10 + k] = end
    ways[wid] = (tags, [1, 10 + k])
  return make(nodes, ways)


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


def test_crossing_lines_on_the_road():
  nodes = {1: (0.0, 0.0), 2: (0.0, -100.0), 3: (-100.0, 0.0), 4: (100.0, 0.0), 20: (-10.0, -20.0), 21: (10.0, -20.0)}
  ways = {1: (TWO_WAY, [2, 1]), 2: (TWO_WAY, [3, 1]), 3: (TWO_WAY, [1, 4]),
          9: ({'highway': 'footway', 'footway': 'crossing'}, [20, 21])}
  lines = Junctions(make(nodes, ways)).crossing_lines()
  assert len(lines) == 1 and np.allclose(sorted(lines[0][[0, -1], 0]), [-5.5, 5.5])


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
