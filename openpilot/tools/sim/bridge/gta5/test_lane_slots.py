"""Route input v2's lane slots (gta5_lane_slots.py) on the map fixtures and small hand-made maps. No pytest needed:
`python test_lane_slots.py` runs them all."""
import os
import time
import xml.etree.ElementTree as ET

import numpy as np

from openpilot.tools.sim.bridge.gta5 import gta5_lane_slots as ls
from openpilot.tools.sim.bridge.gta5 import gta5_nav
from openpilot.tools.sim.bridge.gta5 import gta5_route_input as ri
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import TAPER_M, OsmLanes
from openpilot.tools.sim.bridge.gta5.map.osm_pbf import OsmData
from openpilot.tools.sim.bridge.gta5.map.router import Route

FIXTURES = os.path.join(os.path.dirname(__file__), 'map', 'fixtures')
M_PER_DEG = 111319.49
# Valhalla maneuver types
START, DEST, RIGHT, LEFT, EXIT_RIGHT = 1, 4, 10, 15, 20
TWO_EACH_WAY = {'highway': 'primary', 'lanes': '4', 'lanes:forward': '2', 'lanes:backward': '2', 'width': '14'}


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


def rows(slots: ls.LaneSlots, s: float, v: float = 0.0) -> tuple[str, str, float | None, float | None]:
  """(here, out, m to where the car may start moving into the target, m to where it must be in it) as describe()
  writes them, kerb first."""
  text, _, when = ls.describe(slots.encode(s, v)).partition(' | ')
  here, out = text.removeprefix('here ').split(' out ')
  if not when:
    return here, out, None, None
  start, end = when.removeprefix('from ').removesuffix(' m').split(' by ')
  return here, out, float(start), float(end)


def bay(right: bool = True) -> tuple[ls.LaneSlots, Route]:
  """A turn across the oncoming lanes from a bay opening 60 m before the junction between our two lanes and two
  oncoming lanes; its taper ends 30 m on, 270 m along."""
  osm, nodes = fixture('bay_two_way.osm' if right else 'lht_bay.osm', right)
  r = route(osm, nodes, [1, 2, 3, 4], {2: LEFT if right else RIGHT})
  return ls.LaneSlots(r, right), r


def centre_lane(right: bool = True) -> tuple[ls.LaneSlots, Route]:
  """A turn across traffic from a shared centre turn lane, 300 m along a road with two lanes each way beside it."""
  if right:
    osm, nodes = fixture('centre_turn_lane.osm')
  else:  # mirrored
    nodes = {1: (0.0, -300.0), 2: (0.0, 0.0), 3: (200.0, 0.0), 4: (-200.0, 0.0)}
    centre = {'highway': 'secondary', 'lanes': '5', 'lanes:forward': '2', 'lanes:backward': '2', 'lanes:both_ways': '1',
              'turn:lanes:both_ways': 'right', 'width': '17.5'}
    osm = osm_map(nodes, {1: (centre, [1, 2]), 2: (TWO_EACH_WAY, [2, 3]), 3: (TWO_EACH_WAY, [2, 4])}, drive_on_right=False)
  r = route(osm, nodes, [1, 2, 3], {1: LEFT if right else RIGHT})
  return ls.LaneSlots(r, right), r


def kerb_first(slots: ls.LaneSlots, sec):
  return sec.spans[::-1] if slots.drive_on_right else list(sec.spans)


def check_targets(slots: ls.LaneSlots, r: Route, v: float):
  """Over the whole route every 1 m: a target slot is always one of our own lanes there at full width, or a centre
  turn lane with the turn's arrow within CENTRE_LANE_M of the turn; never an oncoming lane or a bay still opening."""
  for s in np.arange(0.0, r.length, 1.0):
    vec = slots.encode(float(s), v)
    here = slots.section_here(float(s))
    k = int(np.searchsorted(slots.along, s, side='right'))
    for block, sec in ((ls.LANES_HERE, here), (ls.LANES_EXIT, None)):
      states = vec[block].reshape(ls.SLOTS, 3)
      if sec is None:  # the road out of the next maneuver
        nxt = slots._next(float(s))
        sec = slots.section_here(slots.moves[nxt].along + ls.EXIT_PAST) if nxt is not None else None
      if sec is None:
        assert not states.any(), s
        continue
      spans = kerb_first(slots, sec)
      for slot in np.flatnonzero(states[:, ls.TARGET]):
        span = spans[slot]
        assert span.heading != -1, (s, slot)
        assert span.right - span.left > 3.4, (s, slot, span)  # a lane fully there (the fixtures' lanes are 3.5 m)
        if span.heading == 0:
          assert block == ls.LANES_HERE and slots.moves[k].along - s <= ls.CENTRE_LANE_M, s
      # the slots match the cross-section: oncoming lanes oncoming, padding past the road
      assert all(states[j, ls.ONCOMING] for j, sp in enumerate(spans[:ls.SLOTS]) if sp.heading == -1), s
      assert not states[len(spans):].any(), s


def test_left_hand_traffic_turn():
  # east along a road with two lanes forward (through | right) left of one back, then right across traffic into a
  # road with one lane each way: kerb first is from the left
  osm, nodes = fixture('lht.osm', drive_on_right=False)
  slots = ls.LaneSlots(route(osm, nodes, [1, 2, 4], {1: RIGHT}), drive_on_right=False)
  assert rows(slots, 0.0) == ('aTo.....', 'ao......', 10.0, 70.0)  # shown before the changes may start, 60 m before 70 m
  assert rows(slots, 20.0) == ('aTo.....', 'ao......', 0.0, 50.0)  # the inner lane, the right turn's arrow
  assert rows(slots, 85.0) == ('aTo.....', 'ao......', 0.0, 0.0)  # past the end, until the junction: be in it now
  assert rows(slots, 105.0) == ('ao......', 'ao......', None, None)  # on the road out; one lane into it: no target
  assert rows(slots, 150.0)[1] == '........'  # no maneuver ahead
  vec = slots.encode(20.0)
  assert vec.shape == (ls.LANE_SLOTS_LEN,) and abs(vec[ls.TARGET_END] - 0.5) < 1e-4 and vec[ls.TARGET_START] == 0.0
  assert np.all(vec[ls.LANES_HERE].reshape(ls.SLOTS, 3).sum(axis=1) <= 1)  # one-hot or zero


def test_left_hand_traffic_through():
  # straight on past the right-only lane: the through lane is the target, though no maneuver is there
  osm, nodes = fixture('lht.osm', drive_on_right=False)
  slots = ls.LaneSlots(route(osm, nodes, [1, 2, 3]), drive_on_right=False)
  assert rows(slots, 30.0) == ('Tao.....', '........', 0.0, 40.0)
  assert rows(slots, 120.0) == ('ao......', '........', None, None)


def test_mirror():
  # the same junction in both traffic sides, mirrored: the slots are the same, as they count from the kerb
  ways = {1: ({'highway': 'primary', 'lanes': '4', 'width': '14'}, [1, 2]), 2: ({'highway': 'primary', 'lanes': '2', 'width': '7'}, [2, 3])}
  out = []
  for right, sign in ((True, 1.0), (False, -1.0)):
    nodes = {1: (0.0, -200.0), 2: (0.0, 0.0), 3: (sign * 200.0, 0.0)}
    slots = ls.LaneSlots(route(osm_map(nodes, ways, right), nodes, [1, 2, 3], {1: RIGHT if right else LEFT}), drive_on_right=right)
    out.append([rows(slots, s) for s in (50.0, 150.0, 210.0)])
  assert out[0] == out[1], out
  # without turn arrows, the turn's outside lane: the kerb lane; the changes may start 60 m (at low speed) before 30 m
  # short of the turn, and it shows TARGET_SHOW m before that; into the one lane of the road out, which needs no target
  assert out[0] == [('Taoo....', 'ao......', 60.0, 120.0), ('Taoo....', 'ao......', 0.0, 20.0), ('ao......', 'ao......', None, None)]


def test_turn_bay_change_lanes():
  # a left turn bay opens 40 m before the junction from the one lane forward, change:lanes=no along it: it's the
  # target once its taper has ended, 30 m on, and due straight away; before, the one lane forward needs no target
  osm, nodes = fixture('turn_bay.osm')
  slots = ls.LaneSlots(route(osm, nodes, [1, 2, 3, 5], {2: LEFT}))
  move = slots.moves[0]
  assert move.targets == {0} and abs(move.opens - 90.0) < 0.01 and move.bans == [(move.opens, move.along)]
  assert abs(move.window(0.0)[2] - 90.0) < 0.01  # not in it before it's there
  for s in (40.0, 70.0, 89.0):
    assert rows(slots, s) == ('ao......', 'ao......', None, None), s
  assert rows(slots, 91.0) == ('aTo.....', 'ao......', 0.0, 0.0)


def test_bay_opens_after_its_taper():
  # a bay opening beside the oncoming lanes isn't a lane until its taper has ended: until then the target is the
  # through lane it opens from, next to the oncoming lanes, by the time the bay is there; then the bay itself
  for right in (True, False):
    slots, r = bay(right)
    opens = 240.0 + TAPER_M
    assert all(slots.section_here(s).lanes == 2 for s in np.arange(0.0, opens - 0.01, 1.0))
    assert slots.section_here(opens + 0.01).lanes == 3
    assert abs(slots.moves[0].opens - opens) < 0.01
    assert rows(slots, 20.0, 10.0) == ('aToo....', 'aToo....', 50.0, 250.0), right
    assert rows(slots, 250.0, 10.0) == ('aToo....', 'aToo....', 0.0, 20.0), right  # in the taper: not yet the bay
    assert rows(slots, 285.0, 10.0) == ('aaToo...', 'aToo....', 0.0, 0.0), right
    for v in (5.0, 15.0):
      check_targets(slots, r, v)


def test_centre_turn_lane_only_near_the_turn():
  # a shared centre lane is the turn's lane only within CENTRE_LANE_M of it; before, the lane beside it
  for right in (True, False):
    slots, r = centre_lane(right)
    assert slots.moves[0].centre
    assert rows(slots, 200.0, 10.0) == ('aTooo...', 'aToo....', 0.0, 40.0), right
    assert rows(slots, 239.0, 10.0)[0] == 'aTooo...'
    assert rows(slots, 250.0, 10.0) == ('aaToo...', 'aToo....', 0.0, 20.0), right
    for v in (5.0, 15.0):
      check_targets(slots, r, v)


def test_targets_never_oncoming():
  # every 1 m along the fixtures' approaches, at two speeds
  osm, nodes = fixture('lht.osm', drive_on_right=False)
  cases = [(ls.LaneSlots(route(osm, nodes, [1, 2, 4], {1: RIGHT}), False), route(osm, nodes, [1, 2, 4], {1: RIGHT}))]
  osm, nodes = fixture('turn_bay.osm')
  r = route(osm, nodes, [1, 2, 3, 5], {2: LEFT})
  cases.append((ls.LaneSlots(r), r))
  osm, nodes = fixture('freeway.osm')
  r = route(osm, nodes, [1, 2, 4], {1: EXIT_RIGHT})
  cases.append((ls.LaneSlots(r), r))
  for slots, r in cases:
    for v in (5.0, 15.0):
      check_targets(slots, r, v)


def test_nav_lane_reading_waits_for_the_bay():
  # nav's lane (Route.lane) counts a bay only once it's open, so nav never changes towards one before: in the taper
  # the car in the lane beside the oncoming ones is in the left turn's lanes already (lane 0 of 2), then beside the bay
  _, r = bay()
  arrows = gta5_nav.parse_arrows(r.lane_arrows(r.length, 0.0))
  for s, lanes_here, k, want in ((250.0, 2, 0, (0, 0)), (285.0, 3, 1, (0, 0))):
    r.at, r.seg, r.misaligned = s, r.lanes.segment(s), 0.0
    sec = r.lanes.opened_at(s, r.seg)
    r.right = sec.ours[k].centre
    turn = gta5_nav.Turn(300.0 - s, 'left', 90.0)
    gta5_nav.aim(turn, [(d - s, lanes) for d, lanes in arrows])
    assert r.lane() == [k, lanes_here] and turn.lanes(lanes_here) == want, (s, r.lane())
  r.at, r.seg = 250.0, r.lanes.segment(250.0)
  assert r.section(r.seg).lanes == 3  # the way's own lanes, which nav read before


def test_freeway_exit():
  # the exit lane is the kerb lane (slot 0) by its slight_right arrow; change:lanes keeps the lanes beside it from
  # changing into it along the whole way, so it is due from its start
  osm, nodes = fixture('freeway.osm')
  slots = ls.LaneSlots(route(osm, nodes, [1, 2, 4], {1: EXIT_RIGHT}))
  assert slots.moves[0].fork and slots.moves[0].targets == {3}
  assert rows(slots, 50.0, 25.0) == ('Taaa....', 'a.......', 0.0, 0.0)
  assert rows(slots, 210.0) == ('a.......', 'a.......', None, None)
  # going on along the freeway, out of the exit-only lane
  slots = ls.LaneSlots(route(osm, nodes, [1, 2, 3]))
  assert rows(slots, 50.0) == ('aTTT....', '........', 0.0, 0.0)


def test_ban_holds_the_start_back():
  # solid lines (change:lanes=no) for the first 200 m of a 400 m approach to a right turn: the target shows, but the
  # car may only start moving into it once they end
  nodes = {1: (0.0, -400.0), 2: (0.0, -200.0), 3: (0.0, 0.0), 4: (200.0, 0.0)}
  road = {'highway': 'primary', 'oneway': 'yes', 'lanes': '3', 'width': '10.5'}
  ways = {1: ({**road, 'change:lanes': 'no|no|no'}, [1, 2]), 2: (road, [2, 3]), 3: (road, [3, 4])}
  slots = ls.LaneSlots(route(osm_map(nodes, ways), nodes, [1, 2, 3, 4], {2: RIGHT}))
  assert slots.moves[0].bans == [(0.0, 200.0)]
  assert rows(slots, 100.0, 25.0) == ('Taa.....', 'Taa.....', 100.0, 270.0)
  assert rows(slots, 250.0, 25.0) == ('Taa.....', 'Taa.....', 0.0, 120.0)


def test_lane_count_fork_and_window():
  # a two-lane one-way road, a one-lane branch off to the right with no arrows: by the branch's lane count, the kerb
  # lane; the changes may start nav's lead before 40 m short of the fork, longer the faster the car goes
  nodes = {1: (0.0, -600.0), 2: (0.0, 0.0), 3: (0.0, 200.0), 4: (40.0, 150.0)}
  ways = {1: ({'highway': 'trunk', 'oneway': 'yes', 'lanes': '2', 'width': '7'}, [1, 2]),
          2: ({'highway': 'trunk', 'oneway': 'yes', 'lanes': '2', 'width': '7'}, [2, 3]),
          3: ({'highway': 'trunk_link', 'oneway': 'yes', 'lanes': '1', 'width': '4'}, [2, 4])}
  slots = ls.LaneSlots(route(osm_map(nodes, ways), nodes, [1, 2, 4], {1: EXIT_RIGHT}))
  assert slots.moves[0].fork and slots.moves[0].targets == {1}
  assert rows(slots, 300.0, 5.0) == ('aa......', 'a.......', None, None)  # more than TARGET_SHOW m before
  assert rows(slots, 470.0, 5.0) == ('Ta......', 'a.......', 30.0, 90.0)
  assert rows(slots, 500.0, 5.0) == ('Ta......', 'a.......', 0.0, 60.0)
  show, start, end = slots.moves[0].window(25.0)
  assert end == 600.0 - 50.0 and start == end - (8.0 + 4.0 + 8.0) * 25.0 and show == 0.0


def test_exit_narrowed_by_next_turn():
  # right from a one-lane road onto a three-lane one-way road, then left 70 m on: it turns into the right lane, but
  # the left turn needs the left lane from straight away
  nodes = {1: (0.0, -200.0), 2: (0.0, 0.0), 3: (70.0, 0.0), 4: (70.0, 200.0), 5: (300.0, 0.0), 6: (-100.0, 0.0)}
  ways = {1: ({'highway': 'residential', 'oneway': 'yes', 'lanes': '1', 'width': '4'}, [1, 2]),
          2: ({'highway': 'primary', 'oneway': 'yes', 'lanes': '3', 'width': '10.5'}, [2, 3, 5]),
          3: ({'highway': 'primary', 'oneway': 'yes', 'lanes': '2', 'width': '7'}, [3, 4]),
          4: ({'highway': 'primary', 'oneway': 'yes', 'lanes': '3', 'width': '10.5'}, [6, 2])}  # a junction, not a corner
  slots = ls.LaneSlots(route(osm_map(nodes, ways), nodes, [1, 2, 3, 4], {1: RIGHT, 2: LEFT}))
  assert rows(slots, 150.0)[1] == 'aaT.....'  # the left lane of three, kerb first
  assert rows(slots, 215.0)[0] == 'aaa.....'  # nothing for nav's TURN_HOLDS m out of a turn
  assert rows(slots, 225.0) == ('aaT.....', 'aT......', 0.0, 15.0)  # then here, and the next turn's road out


def test_no_lanes_and_preview():
  assert not ls.preview(None, 0.0, 0.0).any() and ls.preview(None, 0.0, 0.0).shape == (ls.PREVIEW_LEN,)
  osm, nodes = fixture('two_way.osm')
  ids = sorted(nodes)[:2]
  slots = ls.LaneSlots(route(osm, nodes, ids))
  vec = ls.preview(slots, 10.0, 0.0)
  assert vec[ls.SIDE] == 1.0 and ls.describe(vec[:ls.LANE_SLOTS_LEN]) == 'here ao...... out ........'
  untagged = Route(np.array([(0.0, 0.0), (0.0, 100.0)]), [])  # no lanes known at all
  assert not ls.LaneSlots(untagged).encode(10.0).any()
  here, out, start, end = ls.decode(vec)
  assert list(here[:3]) == [ls.ALLOWED, ls.ONCOMING, -1] and (out == -1).all() and start is None and end is None


def test_route_input_v2():
  """The model's route input is v1's 173 floats, unchanged, then the lane slots."""
  slots, r = bay()
  v2, v1 = ri.RouteInput(r), ri.RouteInput(r, lanes=False)
  assert ri.ROUTE_LEN == ri.V1_LEN + ls.LANE_SLOTS_LEN and ri.LANES.stop == ri.ROUTE_LEN
  for s in np.linspace(0.0, r.length, 60):
    vec, old = v2.encode(float(s), 0.0, 8.0), v1.encode(float(s), 0.0, 8.0)
    assert vec.shape == (ri.ROUTE_LEN,)
    np.testing.assert_array_equal(vec[:ri.V1_LEN], old[:ri.V1_LEN])
    np.testing.assert_array_equal(vec[ri.LANES], slots.encode(float(s), 8.0))
    assert not old[ri.LANES].any()
  assert any(ls.decode(v2.encode(float(s), 0.0, 8.0)[ri.LANES])[2] is not None for s in np.linspace(0.0, r.length, 60))
  assert ri.describe(v2.encode(150.0, 0.0, 8.0)).count('here') == 1


def test_encode_is_cheap():
  slots, r = bay()
  t = time.perf_counter()
  for s in np.linspace(0.0, r.length, 2000):
    slots.encode(float(s), 10.0)
  assert (time.perf_counter() - t) / 2000 < 0.5e-3


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
