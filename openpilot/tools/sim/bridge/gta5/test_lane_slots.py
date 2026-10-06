"""Route input v2's lane slots (gta5_lane_slots.py) on the map fixtures and small hand-made maps. No pytest needed:
`python test_lane_slots.py` runs them all."""
import os
import time
import xml.etree.ElementTree as ET

import numpy as np

from openpilot.tools.sim.bridge.gta5 import gta5_lane_slots as ls
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes
from openpilot.tools.sim.bridge.gta5.map.osm_pbf import OsmData
from openpilot.tools.sim.bridge.gta5.map.router import Route

FIXTURES = os.path.join(os.path.dirname(__file__), 'map', 'fixtures')
M_PER_DEG = 111319.49
# Valhalla maneuver types
START, DEST, RIGHT, LEFT, EXIT_RIGHT = 1, 4, 10, 15, 20


def osm_map(nodes: dict[int, tuple[float, float]], ways: dict, drive_on_right: bool = True) -> OsmLanes:
  """A map of nodes at (x, y) m."""
  ids = np.array(sorted(nodes), np.int64)
  data = OsmData(ids, np.array([nodes[i][1] for i in ids]), np.array([nodes[i][0] for i in ids]), {}, dict(ways), {})
  return OsmLanes(data, lambda lat, lon: (np.asarray(lon, float), np.asarray(lat, float)), drive_on_right=drive_on_right)


def fixture(name: str, drive_on_right: bool = True) -> tuple[OsmLanes, dict[int, tuple[float, float]]]:
  root = ET.parse(os.path.join(FIXTURES, name)).getroot()
  nodes = {int(n.get('id')): (float(n.get('lon')) * M_PER_DEG, float(n.get('lat')) * M_PER_DEG) for n in root.iter('node')}
  ways = {int(w.get('id')): ({t.get('k'): t.get('v') for t in w.iter('tag')}, [int(nd.get('ref')) for nd in w.iter('nd')])
          for w in root.iter('way')}
  return osm_map(nodes, ways, drive_on_right), nodes


def route(osm: OsmLanes, nodes: dict, ids: list[int], turns: dict[int, int] | None = None) -> Route:
  """Through nodes `ids`, with Valhalla maneuvers {shape index: type} between its start and destination."""
  mans = [(0, START), *sorted((turns or {}).items()), (len(ids) - 1, DEST)]
  return Route(np.array([nodes[i] for i in ids], float), [{'type': t, 'begin_shape_index': k} for k, t in mans], osm=osm)


def rows(slots: ls.LaneSlots, s: float, v: float = 0.0) -> tuple[str, str, float | None]:
  """(here, out, m to be in the target by) as describe() writes them, kerb first."""
  text = ls.describe(slots.encode(s, v))
  here, rest = text.removeprefix('here ').split(' out ')
  out, _, by = rest.partition(' by ')
  return here, out, float(by.removesuffix(' m')) if by else None


def test_left_hand_traffic_turn():
  # east along a road with two lanes forward (through | right) left of one back, then right across traffic into a
  # road with one lane each way: kerb first is from the left
  osm, nodes = fixture('lht.osm', drive_on_right=False)
  slots = ls.LaneSlots(route(osm, nodes, [1, 2, 4], {1: RIGHT}), drive_on_right=False)
  assert rows(slots, 0.0) == ('aao.....', 'ao......', None)  # its window starts 60 m before the end at 70 m
  assert rows(slots, 20.0) == ('aTo.....', 'ao......', 50.0)  # the inner lane, the right turn's arrow
  assert rows(slots, 85.0) == ('aTo.....', 'ao......', -15.0)  # past the end, until the junction
  assert rows(slots, 105.0) == ('ao......', 'ao......', None)  # on the road out; one lane into it: no target
  assert rows(slots, 150.0)[1] == '........'  # no maneuver ahead
  vec = slots.encode(20.0)
  assert vec.shape == (ls.LANE_SLOTS_LEN,) and abs(vec[ls.TARGET_DIST] - 0.5) < 1e-4
  assert np.all(vec[ls.LANES_HERE].reshape(ls.SLOTS, 3).sum(axis=1) <= 1)  # one-hot or zero


def test_left_hand_traffic_through():
  # straight on past the right-only lane: the through lane is the target, though no maneuver is there
  osm, nodes = fixture('lht.osm', drive_on_right=False)
  slots = ls.LaneSlots(route(osm, nodes, [1, 2, 3]), drive_on_right=False)
  assert rows(slots, 30.0) == ('Tao.....', '........', 40.0)
  assert rows(slots, 120.0) == ('ao......', '........', None)


def test_mirror():
  # the same junction in both traffic sides, mirrored: the slots are the same, as they count from the kerb
  ways = {1: ({'highway': 'primary', 'lanes': '4', 'width': '14'}, [1, 2]), 2: ({'highway': 'primary', 'lanes': '2', 'width': '7'}, [2, 3])}
  out = []
  for right, sign in ((True, 1.0), (False, -1.0)):
    osm = osm_map({1: (0.0, -200.0), 2: (0.0, 0.0), 3: (sign * 200.0, 0.0)}, ways, right)
    slots = ls.LaneSlots(route(osm, {1: (0.0, -200.0), 2: (0.0, 0.0), 3: (sign * 200.0, 0.0)}, [1, 2, 3], {1: RIGHT if right else LEFT}),
                         drive_on_right=right)
    out.append([rows(slots, s) for s in (50.0, 150.0, 210.0)])
  assert out[0] == out[1], out
  # without turn arrows, the turn's outside lane: the kerb lane, from 30 m before it less the lane change's lead (60 m
  # at low speed); into the one lane of the road out, which needs no target
  assert out[0] == [('aaoo....', 'ao......', None), ('Taoo....', 'ao......', 20.0), ('ao......', 'ao......', None)]


def test_turn_bay_change_lanes():
  # a left turn bay opens 40 m before the junction, its lanes change:lanes=no: the car must be in the bay's lane by
  # where the bay opens, rather than 30 m before the junction; before it, the one lane forward needs no target
  osm, nodes = fixture('turn_bay.osm')
  slots = ls.LaneSlots(route(osm, nodes, [1, 2, 3, 5], {2: LEFT}))
  move = slots.moves[0]
  assert move.targets == {0} and len(move.bans) == 1 and abs(move.window(0.0)[1] - 60.0) < 0.01
  assert rows(slots, 40.0) == ('ao......', 'ao......', None)
  assert rows(slots, 80.0) == ('aTo.....', 'ao......', -20.0)


def test_freeway_exit():
  # the exit lane is the kerb lane (slot 0) by its slight_right arrow; change:lanes keeps the lanes beside it from
  # changing into it along the whole way, so the target applies from its start
  osm, nodes = fixture('freeway.osm')
  slots = ls.LaneSlots(route(osm, nodes, [1, 2, 4], {1: EXIT_RIGHT}))
  assert slots.moves[0].fork and slots.moves[0].targets == {3}
  assert rows(slots, 50.0, 25.0) == ('Taaa....', 'a.......', -50.0)
  assert rows(slots, 210.0) == ('a.......', 'a.......', None)
  # going on along the freeway, out of the exit-only lane
  slots = ls.LaneSlots(route(osm, nodes, [1, 2, 3]))
  assert rows(slots, 50.0) == ('aTTT....', '........', -50.0)


def test_lane_count_fork_and_window():
  # a two-lane one-way road, a one-lane branch off to the right with no arrows: by the branch's lane count, the kerb
  # lane; its window begins nav's lead before 40 m short of the fork, longer the faster the car goes
  nodes = {1: (0.0, -600.0), 2: (0.0, 0.0), 3: (0.0, 200.0), 4: (40.0, 150.0)}
  ways = {1: ({'highway': 'trunk', 'oneway': 'yes', 'lanes': '2', 'width': '7'}, [1, 2]),
          2: ({'highway': 'trunk', 'oneway': 'yes', 'lanes': '2', 'width': '7'}, [2, 3]),
          3: ({'highway': 'trunk_link', 'oneway': 'yes', 'lanes': '1', 'width': '4'}, [2, 4])}
  slots = ls.LaneSlots(route(osm_map(nodes, ways), nodes, [1, 2, 4], {1: EXIT_RIGHT}))
  assert slots.moves[0].fork and slots.moves[0].targets == {1}
  assert rows(slots, 470.0, 5.0) == ('aa......', 'a.......', None)
  assert rows(slots, 500.0, 5.0) == ('Ta......', 'a.......', 60.0)
  start, end = slots.moves[0].window(25.0)
  assert end == 600.0 - 50.0 and start == end - (8.0 + 4.0 + 8.0) * 25.0


def test_exit_narrowed_by_next_turn():
  # right from a one-lane road onto a three-lane one-way road, then left 70 m on: it turns into the right lane, but
  # the left turn needs the left lane from straight away
  nodes = {1: (0.0, -200.0), 2: (0.0, 0.0), 3: (70.0, 0.0), 4: (70.0, 200.0), 5: (300.0, 0.0)}
  ways = {1: ({'highway': 'residential', 'oneway': 'yes', 'lanes': '1', 'width': '4'}, [1, 2]),
          2: ({'highway': 'primary', 'oneway': 'yes', 'lanes': '3', 'width': '10.5'}, [2, 3, 5]),
          3: ({'highway': 'primary', 'oneway': 'yes', 'lanes': '2', 'width': '7'}, [3, 4])}
  slots = ls.LaneSlots(route(osm_map(nodes, ways), nodes, [1, 2, 3, 4], {1: RIGHT, 2: LEFT}))
  assert rows(slots, 150.0)[1] == 'aaT.....'  # the left lane of three, kerb first
  assert rows(slots, 215.0)[0] == 'aaa.....'  # nothing for nav's TURN_HOLDS m out of a turn
  assert rows(slots, 225.0) == ('aaT.....', 'aT......', 15.0)  # then here, and the next turn's road out


def test_no_lanes_and_preview():
  assert not ls.preview(None, 0.0, 0.0).any() and ls.preview(None, 0.0, 0.0).shape == (ls.PREVIEW_LEN,)
  osm, nodes = fixture('two_way.osm')
  ids = sorted(nodes)[:2]
  slots = ls.LaneSlots(route(osm, nodes, ids))
  vec = ls.preview(slots, 10.0, 0.0)
  assert vec[ls.SIDE] == 1.0 and ls.describe(vec[:ls.LANE_SLOTS_LEN]) == 'here ao...... out ........'
  untagged = Route(np.array([(0.0, 0.0), (0.0, 100.0)]), [])  # no lanes known at all
  assert not ls.LaneSlots(untagged).encode(10.0).any()
  here, out, dist = ls.decode(vec)
  assert list(here[:3]) == [ls.ALLOWED, ls.ONCOMING, -1] and (out == -1).all() and dist is None


def test_encode_is_cheap():
  osm, nodes = fixture('lht.osm', drive_on_right=False)
  slots = ls.LaneSlots(route(osm, nodes, [1, 2, 4], {1: RIGHT}), drive_on_right=False)
  t = time.perf_counter()
  for s in np.linspace(0.0, 200.0, 2000):
    slots.encode(float(s), 10.0)
  assert (time.perf_counter() - t) / 2000 < 0.5e-3


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
