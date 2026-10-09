"""The car's lane, and nav's lanes along a route, where the road's lanes change. No pytest needed:
`python test_lane_numbering.py` runs them all (the live map's cases only with ~/gta5map_lanes and its router)."""
import os
from types import SimpleNamespace

import numpy as np

from openpilot.tools.sim.bridge.gta5.test_lane_slots import START, DEST, osm_map
from openpilot.tools.sim.bridge.gta5.map.router import Route

# Vinewood Blvd as the map has it: three lanes our way (a left-turn lane, straight, straight or right) and two back,
# the way's line down the middle of the left-turn lane
VINEWOOD = {'highway': 'primary', 'lanes': '5', 'lanes:forward': '3', 'lanes:backward': '2', 'width': '26.5',
            'width:lanes:forward': '4.5|5.5|5.5', 'width:lanes:backward': '5.5|5.5', 'divider': 'double_solid_line',
            'placement:forward': 'middle_of:1', 'turn:lanes:forward': 'left|through|right;through'}


def straight(tags: dict, length: float = 300.0) -> Route:
  nodes = {1: (0.0, 0.0), 2: (0.0, length / 2), 3: (0.0, length)}
  osm = osm_map(nodes, {1: (tags, [1, 2]), 2: (tags, [2, 3])})
  return Route(np.array([nodes[1], nodes[2], nodes[3]]), [{'type': START, 'begin_shape_index': 0},
                                                         {'type': DEST, 'begin_shape_index': 2}], osm=osm)


def test_wide_road_outer_lane_is_on_the_road():
  # Amy's Vinewood Blvd stop at (556.99, 78.40), heading 67.5: centred in the rightmost lane, 10.9 m right of the
  # way's line; over ON_ROAD from it, but between its kerbs
  r = straight(VINEWOOD)
  r.locate(np.array([10.9, 100.0]), heading=0.0)
  assert r.off > 8.0 and r.on_road()
  assert r.lane() == [2, 3]
  r.locate(np.array([25.0, 100.0]), heading=0.0)  # past the kerb: another road beside it
  assert not r.on_road()
  r.locate(np.array([10.9, 100.0]), heading=180.0)  # the other way
  assert not r.on_road()


def world(tagged: bool = True):
  from openpilot.tools.sim.bridge.gta5.gta5_world import GTA5World
  w = GTA5World.__new__(GTA5World)
  w.navigator = SimpleNamespace(router=SimpleNamespace(osm=object() if tagged else None))
  w.good_lane = (None, 0.0)
  return w


def test_car_lane_never_from_a_gta_link():
  from openpilot.tools.sim.bridge.gta5 import gta5_world
  w = world()
  # the route's reading on its road; the plugin's 1-lane GTA link says [0, 1]
  assert w._car_lane([2, 3], 2.15, [0, 1], None, 10.0) == ([2, 3], 2.15)
  # no route reading (stopped, misaligned): the map's match on our side of the road, never the GTA link
  own = {"lane": 2, "lanes": 3, "kind": "own"}
  assert w._car_lane(None, None, [0, 1], own, 11.0) == ([2, 3], None)
  assert w._car_lane(None, None, [0, 1], {"lane": -1, "lanes": 2, "kind": "oncoming"}, 12.0) == ([2, 3], None)  # held
  assert w._car_lane(None, None, [0, 1], None, 11.0 + gta5_world.LANE_HOLD + 0.1) == (None, None)
  # the plugin reading another lane of as many: neither, unless the map's match sides with the route's
  assert w._car_lane([1, 3], 1.0, [0, 3], None, 20.0) == (None, None)
  assert w._car_lane([1, 3], 1.0, [0, 3], {"lane": 1, "lanes": 3, "kind": "own"}, 20.0) == ([1, 3], 1.0)
  # a map without lane tags has only the plugin's
  assert world(tagged=False)._car_lane(None, None, [0, 1], None, 30.0) == ([0, 1], None)


def one_way(lanes: int, place: str, turns: str | None = None) -> dict:
  tags = {'highway': 'primary', 'oneway': 'yes', 'lanes': str(lanes), 'width': str(3.5 * lanes), 'placement': place}
  if turns:
    tags['turn:lanes'] = turns
  return tags


def road(*parts: tuple[float, dict]) -> Route:
  """A straight road north along x = 0, of ways [(m long, tags)] one after another."""
  nodes, ways, y = {1: (0.0, 0.0)}, {}, 0.0
  for k, (length, tags) in enumerate(parts):
    y += length
    nodes[k + 2] = (0.0, y)
    ways[k + 1] = (tags, [k + 1, k + 2])
  osm = osm_map(nodes, ways)
  r = Route(np.array([nodes[i] for i in sorted(nodes)]), [{'type': START, 'begin_shape_index': 0},
                                                       {'type': DEST, 'begin_shape_index': len(nodes) - 1}], osm=osm)
  r.locate(np.array([0.0, 1.0]), heading=0.0)
  return r


def ribbon(r: Route, lane: int, n: int):
  """The lane line nav's plan draws from lane `lane` of n at the car: its offset (m right of the route) s m along."""
  from openpilot.selfdrive.navd.planner import lane_plan
  forks = [[f.along - r.at, f.side, f.lanes, f.lanes_in, f.keep, f.other, f.slip] for f in r.forks if f.along > r.at]
  line = r.lane_line(lane_plan(r.rest(), forks, [lane, n], r.lanes_at, 15.0, None, r.lane_arrows(r.length, 0.0),
                               r.lane_drops(r.length, 0.0), maps=r.lane_maps(r.length)))

  def at(s: float) -> float:
    p = np.array([np.interp(s, r.along, r.points[:, k]) for k in (0, 1)])
    q = np.array([np.interp(s + 1.0, r.along, r.points[:, k]) for k in (0, 1)])
    t = (q - p) / np.hypot(*(q - p))
    near = line[np.argmin(np.hypot(*(line - p).T))]
    return float((near - p) @ np.array([t[1], -t[0]]))
  return at


def maps(r: Route) -> list:
  return [(round(s), list(m), n) for s, m, n in r.lane_maps_along()]


def test_left_turn_bay_opening():
  # a left-turn bay opens beside our two lanes: the lanes we're in carry on, numbered one further right; also where
  # only the ways' lines (the right kerb kept) say so, without arrows
  for turns in ('left|through|through', None):
    r = road((150.0, one_way(2, 'right_of:2')), (150.0, one_way(3, 'right_of:3', turns)), (100.0, one_way(3, 'right_of:3')))
    assert maps(r) == [(150, [1, 2], 3)], turns
    for lane, x in ((0, -5.25), (1, -1.75)):
      at = ribbon(r, lane, 2)
      assert all(abs(at(y) - x) < 0.05 for y in (100.0, 190.0, 250.0, 380.0)), (turns, lane, [at(y) for y in (100.0, 190.0, 250.0)])


def test_right_turn_bay_opening():
  r = road((150.0, one_way(2, 'left_of:1')), (150.0, one_way(3, 'left_of:1', 'through|through|right')), (100.0, one_way(3, 'left_of:1')))
  assert maps(r) == [(150, [0, 1], 3)]  # numbered from the left, our lanes keep their numbers
  at = ribbon(r, 1, 2)
  assert all(abs(at(y) - 5.25) < 0.05 for y in (100.0, 190.0, 250.0, 380.0))


def test_lane_drops():
  # the right lane ends: the car in it merges into the one beside it, the others keep theirs
  r = road((150.0, one_way(3, 'left_of:1')), (250.0, one_way(2, 'left_of:1')))
  assert maps(r) == [(150, [0, 1, None], 2)]
  assert abs(ribbon(r, 2, 3)(300.0) - 5.25) < 0.05 and abs(ribbon(r, 0, 3)(300.0) - 1.75) < 0.05
  assert abs(ribbon(r, 2, 3)(120.0) - 5.25) < 0.05  # out of it before it ends, not across where it does
  # the left lane ends (the right kerb kept): the lanes right of it carry on, numbered one further left
  r = road((150.0, one_way(3, 'right_of:3')), (250.0, one_way(2, 'right_of:2')))
  assert maps(r) == [(150, [None, 0, 1], 2)]
  for lane, x in ((1, -5.25), (2, -1.75)):
    at = ribbon(r, lane, 3)
    assert abs(at(100.0) - x) < 0.05 and abs(at(300.0) - x) < 0.05
  assert abs(ribbon(r, 0, 3)(300.0) + 5.25) < 0.05  # the lane that ends, into the one beside it


def test_widening_and_back():
  # 2 lanes widen to 3 for 150 m and back, on the right and on the left: the car keeps to its lane throughout
  for place, n3, back in (('left_of:1', [0, 1], [0, 1, None]), ('right_of:', [1, 2], [None, 0, 1])):
    p2, p3 = (place, place) if place == 'left_of:1' else ('right_of:2', 'right_of:3')
    r = road((150.0, one_way(2, p2)), (150.0, one_way(3, p3)), (150.0, one_way(2, p2)))
    assert maps(r) == [(150, n3, 3), (300, back, 2)], place
    for lane in (0, 1):
      at = ribbon(r, lane, 2)
      x = at(100.0)
      assert all(abs(at(y) - x) < 0.05 for y in (200.0, 280.0, 400.0)), (place, lane, [at(y) for y in (200.0, 280.0, 400.0)])


def test_map_count_blip():
  # a way the map has one lane too many for 12 m, its line in the middle as the ways either side have theirs: the car
  # comes out of it in the lane it went in
  centred = {'highway': 'primary', 'oneway': 'yes', 'lanes': '2', 'width': '7'}
  blip = {'highway': 'primary', 'oneway': 'yes', 'lanes': '3', 'width': '10.5'}
  r = road((150.0, centred), (12.0, blip), (150.0, centred))
  for lane, x in ((0, -1.75), (1, 1.75)):
    at = ribbon(r, lane, 2)
    assert abs(at(100.0) - x) < 0.05 and abs(at(250.0) - x) < 0.05, (lane, at(100.0), at(250.0))


def test_map_split_takes_its_side():
  # a one-way road of two lanes splits, a one-lane link off to the right the route takes, the road going on left: the
  # link takes the right lane (where the two ways' lines lie at the node can't say), and the car in the left lane is
  # in it before the split
  nodes = {1: (0.0, 0.0), 2: (0.0, 300.0), 3: (0.0, 500.0), 4: (30.0, 480.0)}
  ways = {1: (one_way(2, 'left_of:1'), [1, 2]), 2: (one_way(2, 'left_of:1'), [2, 3]),
          3: ({'highway': 'primary_link', 'oneway': 'yes', 'lanes': '1', 'width': '4'}, [2, 4])}
  r = Route(np.array([nodes[1], nodes[2], nodes[4]]), [{'type': START, 'begin_shape_index': 0},
                                                      {'type': DEST, 'begin_shape_index': 2}], osm=osm_map(nodes, ways))
  r.locate(np.array([0.0, 1.0]), heading=0.0)
  assert maps(r) == [(300, [None, 0], 1)]
  at = ribbon(r, 0, 2)
  assert abs(at(50.0) - 1.75) < 0.05 and abs(at(275.0) - 5.25) < 0.05  # as nav changes for it: 12 s at 15 m/s before 270 m


def test_carriageway_joins_its_two_way_road():
  # one carriageway of a divided road (a one-lane one-way) becomes a two-way road with a left-turn bay where the median
  # was, the other carriageway leaving the node: the carriageway's lane carries on as the road's right lane, the
  # outside kerb kept, though the road's line runs down the bay (where the ways' lines lie says the bay)
  nodes = {1: (0.0, 0.0), 2: (0.0, 150.0), 3: (0.0, 350.0), 5: (-12.0, 0.0)}
  two_way = {'highway': 'residential', 'lanes': '3', 'lanes:forward': '1', 'lanes:backward': '2', 'width': '15.5',
             'width:lanes:forward': '5.5', 'width:lanes:backward': '4.5|5.5', 'placement:backward': 'middle_of:1'}
  for turns in (None, 'left|right;through'):
    tags = dict(two_way, **({'turn:lanes:backward': turns} if turns else {}))
    ways = {1: ({'highway': 'residential', 'oneway': 'yes', 'lanes': '1', 'width': '5.5'}, [1, 2]), 2: (tags, [3, 2]),
            3: ({'highway': 'residential', 'oneway': 'yes', 'lanes': '1', 'width': '5.5'}, [2, 5])}
    r = Route(np.array([nodes[1], nodes[2], nodes[3]]), [{'type': START, 'begin_shape_index': 0},
                                                        {'type': DEST, 'begin_shape_index': 2}], osm=osm_map(nodes, ways))
    r.locate(np.array([0.0, 1.0]), heading=0.0)
    assert maps(r) == [(150, [1], 2)], turns
    at = ribbon(r, 0, 1)
    assert abs(at(100.0)) < 0.05 and abs(at(250.0) - 5.0) < 0.05, (turns, at(100.0), at(250.0))


def test_jog_along_an_angled_link():
  # a carriageway's last link angles 5 m across to the line of the road it joins, the road's lanes 5 m right of its
  # line where the carriageway's were on it: the lanes run on straight along the kerb, and so does the lane line
  # (eased over 10 m either side of the node, it swung 2-3 m left along the link and back)
  from openpilot.tools.sim.bridge.gta5.map.osm_lanes import _ease_jogs
  points = np.array([(0.0, 0.0), (0.0, 100.0), (-5.0, 115.0), (-5.0, 300.0)])
  along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(points, axis=0).T))))
  sj = float(along[2])
  s = np.unique(np.concatenate((np.arange(0.0, along[-1], 2.0), along)))
  offs = np.where(s < sj, 0.0, 5.0)
  out = _ease_jogs(s, offs, [sj], points, along)
  xy = np.stack([np.interp(s, along, points[:, 0]), np.interp(s, along, points[:, 1])], axis=1)
  d = np.gradient(xy, axis=0)
  n = np.stack([d[:, 1], -d[:, 0]], axis=1) / np.hypot(d[:, 0], d[:, 1])[:, None]
  x = (xy + n * out[:, None])[:, 0]
  near = (s > 80.0) & (s < 140.0)
  assert np.abs(x[near]).max() < 0.6, x[near]
  assert np.allclose(_ease_jogs(s, offs, [sj]), np.interp(s, [sj - 10.0, sj + 10.0], [0.0, 5.0]))  # without the route: as before


def test_arrows_show_from_where_the_lanes_go_on_into_the_junction():
  # the junction's arrows are tagged on the last way into it alone; the UI's lanes show them from 250 m back while the
  # lanes carry on into it, also across a left-turn bay opening (before it opens, the two lanes it opens beside)
  from openpilot.selfdrive.navd import lane_slots as ls
  from openpilot.tools.sim.bridge.gta5 import gta5_nav_msgs as nm
  r = road((150.0, one_way(2, 'right_of:2')), (100.0, one_way(3, 'right_of:3')),
           (60.0, one_way(3, 'right_of:3', 'left|through|right')), (100.0, one_way(3, 'right_of:3')))
  lanes = r.lanes
  assert lanes.turns_at(200.0) == [frozenset({'left'}), frozenset({'through'}), frozenset({'right'})]
  assert lanes.turns_at(100.0) == [frozenset({'through'}), frozenset({'right'})]
  assert lanes.turns_at(350.0) == [frozenset()] * 3  # past the junction
  slots = ls.LaneSlots(r)
  assert [g['directions'] for g in nm.lane_guide(slots, 100.0, 10.0, None).lanes] == [['straight'], ['right']]
  assert [g['directions'] for g in nm.lane_guide(slots, 200.0, 10.0, None).lanes] == [['left'], ['straight'], ['right']]


def test_crossing_between_carriageways():
  # two one-way carriageways of two lanes heading east side by side, joined by a link angling across: with hatching
  # between them (their lanes 3 m apart) it's no way for a driver; lanes side by side (a dashed line), a lane change
  from openpilot.tools.sim.bridge.gta5.map.router import separated
  for apart, expect in ((10.0, True), (7.0, False)):
    nodes = {1: (0.0, 0.0), 2: (100.0, 0.0), 3: (200.0, 0.0), 4: (0.0, -apart), 5: (125.0, -apart), 6: (200.0, -apart)}
    tags = {'highway': 'motorway', 'oneway': 'yes', 'lanes': '2', 'width': '7'}
    ways = {1: (tags, [1, 2]), 2: (tags, [2, 3]), 3: (tags, [4, 5]), 4: (tags, [5, 6]),
            5: ({'highway': 'motorway', 'oneway': 'yes', 'lanes': '1', 'width': '3.5'}, [2, 5])}
    osm = osm_map(nodes, ways)
    assert separated(osm, np.array(nodes[2]), np.array(nodes[5])) is expect, apart


def test_turn_markers_on_the_lane_line():
  # the overlay's turn and signal markers go on nav's lane line, not the route's line down the road's middle
  from openpilot.tools.sim.bridge.gta5.gta5_world import on_path
  line = np.array([[4.0, 0.0], [4.0, 50.0], [20.0, 66.0]])
  assert np.allclose(on_path(np.array([0.0, 30.0]), line), [4.0, 30.0])
  assert np.allclose(on_path(np.array([0.0, 30.0]), line[:0]), [0.0, 30.0])
  assert np.allclose(on_path(np.array([-30.0, 30.0]), line), [-30.0, 30.0])  # too far from it: as it was


def test_live_lanes_follow_the_maps():
  # nav's live targets at a junction, followed back to the car's lanes across where they change on the way
  from openpilot.selfdrive.navd.planner import LaneMaps, Planner, Through, Turn
  p = Planner()
  p.lane = (1, 2)
  through = Through(100.0, (1, 1, 3))  # left | through | right at the junction, a right-turn bay opening on the way
  p.maps = LaneMaps([[50.0, [0, 1], 3]])
  assert p._lanes_for(through) == (1, 1)
  p.maps = None
  assert p._lanes_for(through) == (0, 0)  # counted from the right: into the left-turn lane
  # a left turn from a bay still to open: the lane it opens beside
  p.maps = LaneMaps([[50.0, [1, 2], 3]])
  turn = Turn(100.0, "left", 90.0)
  turn.targets = (0, 0, 3)
  assert p._lanes_for(turn) == (0, 0)
  p.lane = (1, 3)  # a reading from before the change: the maps don't start from it
  assert p._lanes_for(turn) == (0, 0)


def test_lane_slots_follow_the_maps():
  # straight on at a junction with left | through | right, its right-turn bay opening 100 m before beside a left-turn
  # lane already there: before the bay opens, the target is the right of our two lanes, which carries on as the
  # through lane (counted from the right, it would be the left-turn lane)
  from openpilot.selfdrive.navd import lane_slots as ls
  r = road((150.0, one_way(2, 'left_of:1', 'left|through')), (100.0, one_way(3, 'left_of:1', 'left|through|right')),
           (100.0, one_way(2, 'left_of:1')))
  assert maps(r) == [(150, [0, 1], 3), (250, [None, 0, 1], 2)]
  slots = ls.LaneSlots(r)
  move = next(m for m in slots.moves if abs(m.along - 250.0) < 1.0)
  here = slots.section_here(100.0)
  assert here.lanes == 2 and move.targets == {1} and slots.wanted(move, 100.0, here) == ({1}, False)
  slots.maps = []
  assert slots.wanted(move, 100.0, here) == ({0}, False)  # without the maps


LIVE_MAP = os.path.expanduser("~/gta5map_lanes")


def live_router():
  """The bridge's router on the live map and its Valhalla (map/README.md), None without them."""
  import urllib.request
  if not os.path.exists(os.path.join(LIVE_MAP, "gta5.osm.pbf")):
    return None
  try:
    urllib.request.urlopen("http://127.0.0.1:8002/status", timeout=2.0).read()
  except OSError:
    return None
  from openpilot.tools.sim.bridge.gta5.junction_plan import make_router
  router = make_router(LIVE_MAP, None, None)
  router.osm = router.roads
  return router


def live_route(router, x, y, z, heading, dest):
  r = router.route(np.array([x, y]), (-heading) % 360, np.array(dest), z)
  r.locate(np.array([x, y]), z, heading)
  return r


def test_live_vinewood_bridge_left_bay():
  # Amy's spot on the Vinewood Blvd bridge over the LS Freeway, heading 61, in the right of two lanes: a left-turn bay
  # opens, then at Elgin Ave the right lane only turns right, so nav moves into the through lane, which carries on as
  # the left of two past the junction. Counted from the left, it was taken for the right lane there and the ribbon
  # swung back into it.
  router = live_router()
  if router is None:
    print("skipped: no live map")
    return
  r = live_route(router, 773.2, -27.5, 81.6, 61.0, (248.4, 263.4))
  assert r.lane() == [1, 2]
  m = {round(s - r.at): (list(mp), n) for s, mp, n in r.lane_maps_along()}
  assert m[25] == ([1, 2], 3) and m[88] == ([None, 0, 1], 2)
  at = ribbon(r, 1, 2)
  assert abs(at(r.at) - 10.5) < 0.6  # the right lane, moving over for Elgin Ave from the car, as nav does
  assert abs(at(r.at + 70.0) - 5.0) < 0.3  # the through lane, for Elgin Ave
  assert abs(at(r.at + 110.0) - 5.0) < 0.3  # still it past the junction, now the left of two


def test_live_vinewood_stop_reads_its_lane():
  # Amy stopped at a light on Vinewood Blvd (556.99, 78.40), heading 67.5, centred in the rightmost of three lanes,
  # 10.9 m right of the way's line: on the road, in lane [2, 3], not the plugin's 1-lane link
  router = live_router()
  if router is None:
    print("skipped: no live map")
    return
  r = live_route(router, 556.99, 78.40, 95.37, 67.5, (-813.6, 179.5))
  assert r.on_road() and r.lane() == [2, 3]


def live_plan(router, x, y, z, heading, dest, lane=None):
  """A route on the live map from the car and nav's plan along it from its lane (else the route's reading): (route,
  keys, the plan's lane changes [(m from, m to, lane from, lane to)])."""
  from openpilot.selfdrive.navd.planner import lane_plan
  r = live_route(router, x, y, z, heading, dest)
  info = r.info(r.length)
  keys = lane_plan(r.rest(), info['forks'], lane or r.lane(), r.lanes_at, 20.0, None, info['laneArrows'], info['laneDrops'],
                   maps=r.lane_maps(r.length), turns=r.turns(r.length), crossings=[a - r.at for a in r.stops + r.junctions if a > r.at])
  ramps = [(a[0], b[0], a[1], b[1]) for a, b in zip(keys, keys[1:], strict=False) if b[0] > a[0] + 0.01 and abs(b[1] - a[1]) > 0.01]
  return r, keys, ramps


def test_live_freeway_merge_exit_and_splits():
  router = live_router()
  if router is None:
    print("skipped: no live map")
    return
  dest = (1727.2, 1403.26)
  # LS Freeway on-ramp (1252.9, 521.0): it joins the freeway on the right, into its right lane, not across to the left
  r, keys, ramps = live_plan(router, 1252.89, 520.99, 80.66, 332.0, dest)
  assert r.lane() == [0, 1] and [(round(s), list(m)) for s, m, _ in r.lane_maps_along()][:1] == [(39, [1])]
  assert not [c for c in ramps if c[0] < 150.0]  # no change straight after joining
  # LS Freeway (1666.3, 1243.2) in the right of two lanes, the route off at the exit just ahead: no change at all
  r, keys, ramps = live_plan(router, 1666.32, 1243.17, 84.92, 344.5, dest)
  assert r.lane() == [1, 2] and not ramps
  # Olympic Fwy split (-193.6, -1196.2) in the second lane of four: GTA's forks and the map's lanes agree on the left
  # branch's lanes, the right of its two being the one on; no change through the interchange
  dest = (-680.13, -2021.56)
  r, keys, ramps = live_plan(router, -193.62, -1196.23, 36.84, 96.3, dest)
  assert r.lane() == [1, 4] and not [c for c in ramps if c[0] < 440.0]
  # Dutch London St ramp split (-741.9, -1838.4), the route's branch the right one: GTA's link from the split straight
  # across the hatched gore to it is no way; the route takes the right branch's lane from the node at the car (its
  # link off at 64 deg, which a route can't set off along: from just behind the car), with no lane change
  r, keys, ramps = live_plan(router, -741.90, -1838.37, 26.97, 196.2, dest, lane=[0, 1])
  assert r.length < 300.0 and not router.crossings(r.points) and not ramps, (r.length, ramps)
  k = int(np.argmin(np.hypot(*(r.points - np.array([-752.0, -1851.0])).T)))
  assert np.hypot(*(r.points[k] - np.array([-752.0, -1851.0]))) < 1.0  # onto the right branch's own lane


def bridge_line(r: Route, lane) -> np.ndarray:
  """The lane line as the bridge draws it (GTA5World._lane_line), from the car's lane."""
  from openpilot.selfdrive.navd.planner import Planner
  from openpilot.tools.sim.bridge.gta5.gta5_world import GTA5World
  w = GTA5World.__new__(GTA5World)
  w.nav, w.route, w.lane_line = Planner(), r, (None, 0.0, [])
  return np.array(w._lane_line({"lane": lane}, 0.0))


def test_live_meteor_carriageway_into_its_road():
  # Meteor St, heading 158-163, in the right of the divided road's carriageways as they become one road with a left-turn
  # bay, the route turning right ahead: the carriageway's lane is the road's right lane, no change at all; and the lane
  # line, drawn as the bridge draws it, runs on into it without swinging towards the bay and back (GTA's carriageway
  # links jog left then right into the road's first node, which lies on the bay's line)
  router = live_router()
  if router is None:
    print("skipped: no live map")
    return
  r, keys, ramps = live_plan(router, 530.3, 149.1, 98.2, 163.0, (-813.6, 179.5))
  assert r.lane() == [0, 1]
  assert [(round(s - r.at), list(m)) for s, m, _ in r.lane_maps_along()][:1] == [(18, [1])]
  assert not [c for c in ramps if c[0] < 70.0]
  import math
  x, y, h = 531.3, 153.8, 158.0
  r = live_route(router, x, y, 99.0, h, (-813.6, 179.5))
  line = bridge_line(r, r.lane())
  fwd = np.array([-math.sin(math.radians(h)), math.cos(math.radians(h))])
  a, side = (line - np.array([x, y])) @ fwd, (line - np.array([x, y])) @ np.array([fwd[1], -fwd[0]])
  near = (a > 0.0) & (np.abs(side) < 25.0)
  d = [float(np.interp(v, a[near], side[near])) for v in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0)]
  assert min(d[1:6]) > d[-1] - 1.0, d  # no swing left past where it settles (was 1.9 m)
  assert max(d[1:6]) < d[0] + 1.0, d  # nor right


def sideways(r: Route, keys, x: float, y: float, heading: float, ahead: list[float]) -> list[float]:
  """The lane line's offset (m right) of a straight line from the car along its heading, at each distance ahead."""
  import math
  line = r.lane_line(keys)
  fwd = np.array([-math.sin(math.radians(heading)), math.cos(math.radians(heading))])
  a, side = (line - np.array([x, y])) @ fwd, (line - np.array([x, y])) @ np.array([fwd[1], -fwd[0]])
  near = (a > 0.0) & (np.abs(side) < 25.0)
  return [float(np.interp(d, a[near], side[near])) for d in ahead]


def test_live_eclipse_carriageways_join():
  # Eclipse Blvd where the grass median ends and the south carriageway joins the two-way road, a left-turn lane
  # beginning past the nose. GTA's carriageway link before the node angles 5 m across to the road's line, which runs
  # down the new left lane: the lane line moved across over 10 m either side of the node, following the link out and
  # back (a lane's swing left and back right, at 20 m). It now moves across along the link, as the link does.
  router = live_router()
  if router is None:
    print("skipped: no live map")
    return
  x, y, h = -459.7, 237.8, 261.8
  r, keys, ramps = live_plan(router, x, y, 83.1, h, (-813.6, 179.5))
  assert r.lane() == [1, 2] and not [c for c in ramps if c[0] < 60.0]
  d = sideways(r, keys, x, y, h, [1.0, 10.0, 15.0, 20.0, 30.0, 40.0])
  line = np.interp([10.0, 15.0, 20.0, 30.0], [1.0, 40.0], [d[0], d[-1]])  # the road runs straight, a little off the car's heading
  assert max(abs(a - b) for a, b in zip(d[1:5], line, strict=True)) < 1.3, d
  # further east, the same join (-133.6, 245.0), the route on east and turning left 170 m on: no move into the new
  # left lane as it begins (it's no turn bay: a junction is between), only before the turn
  x, y, h = -133.6, 245.0, 276.0
  r, keys, ramps = live_plan(router, x, y, 95.2, h, (120.0, 240.0))
  assert r.lane() == [1, 2] and not [c for c in ramps if c[0] < 50.0], ramps  # two changes, by 30 m before the turn
  d = sideways(r, keys, x, y, h, [1.0, 20.0, 30.0, 40.0])
  assert max(d) - min(d) < 3.0, d  # the road's own slight angle, no lane's swing


def test_live_fx2_right_carriageway_from_the_split():
  # Olympic Fwy (FX2: 17.0, -1234.8, heading 271, to the right exit at (913, -1215)): at x 461 the four lanes split into
  # two carriageways with hatching between, and GTA joins them by a shortcut link at x 774 (-> 798.5, -1219). The route
  # took the left carriageway and that link across the hatching; it now takes the right carriageway from the split,
  # and nav is in its lanes (the right two of four) before the split
  router = live_router()
  if router is None:
    print("skipped: no live map")
    return
  r, keys, ramps = live_plan(router, 17.0, -1234.8, 37.2, 271.0, (1038.5, -1369.0), lane=[0, 4])
  assert not router.crossings(r.points)
  for x in (480.0, 600.0, 760.0):  # on the right carriageway (its line at y -1230 to -1220), not the left (-1218 to -1203)
    k = int(np.argmin(np.abs(r.points[:, 0] - x)))
    assert r.points[k, 1] < -1219.0, (x, r.points[k])
  split = next(f.along for f in r.forks if 400.0 < f.along < 480.0)
  before = [c for c in ramps if c[0] < split]
  assert before and before[-1][1] <= split and before[-1][3] == 2.0, ramps  # into the third lane of four by the split


def test_live_hairpin_is_no_turn():
  # Mt Vinewood Dr (-896.8, 1732.3), heading 153, on D2b: a hairpin of the road itself 22 m on, no junction: no turn for
  # nav to signal, only the left at the junction 159 m on
  router = live_router()
  if router is None:
    print("skipped: no live map")
    return
  r = live_route(router, -896.8, 1732.3, 0.0 + float(router.paths.z[int(np.argmin(np.hypot(*(router.paths.xy - np.array([-896.8, 1732.3])).T)))]) + 1.0,
                 153.0, (-709.0, 1758.8))
  turns = r.turns(r.length)
  assert [(t[1], round(t[0] / 10.0)) for t in turns][:1] == [('left', 16)], turns


def test_live_alta_bay_from_its_opening():
  # Alta St (227.2, 276.2), left at the junction 85 m on, its left-turn bay opening 34 m on: into the bay from where it
  # opens, the change to the lane beside it first, not as late as the turn allows
  router = live_router()
  if router is None:
    print("skipped: no live map")
    return
  r, keys, ramps = live_plan(router, 227.24, 276.22, 105.19, 159.0, (1727.2, 1403.26), lane=[1, 2])
  opens = next(s for s, m, n in r.lane_maps_along() if n > len(m)) - r.at
  assert round(opens) == 34
  before = [c for c in ramps if c[1] <= opens + 0.1]
  into = [c for c in ramps if opens - 0.1 <= c[0] < 85.0]
  assert before == [(0.0, opens, 1.0, 0.0)] or before[-1][1:] == (opens, 1.0, 0.0)  # beside the bay as it opens
  assert len(into) == 1 and abs(into[0][0] - opens) < 0.1 and into[0][3] == 0.0  # and into it from there


if __name__ == "__main__":
  import sys
  tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
  only = sys.argv[1:]
  for name, fn in tests:
    if only and name not in only:
      continue
    fn()
    print(f"ok {name}")
