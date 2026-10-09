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
  # straight on at a junction with left | through | right, its right-turn bay opening 100 m before: before it opens,
  # the target is the right of our two lanes, which carries on as the through lane (counted from the right, the left)
  from openpilot.selfdrive.navd import lane_slots as ls
  r = road((150.0, one_way(2, 'left_of:1')), (100.0, one_way(3, 'left_of:1', 'left|through|right')),
           (100.0, one_way(2, 'left_of:1')))
  slots = ls.LaneSlots(r)
  here, target = slots.target(100.0, 10.0)
  assert here.lanes == 2 and target is not None and target.want == {1}
  assert slots.target(200.0, 10.0)[1].want == {1}


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
  assert abs(at(r.at + 20.0) - 10.5) < 0.3  # the right lane
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


if __name__ == "__main__":
  import sys
  tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
  only = sys.argv[1:]
  for name, fn in tests:
    if only and name not in only:
      continue
    fn()
    print(f"ok {name}")
