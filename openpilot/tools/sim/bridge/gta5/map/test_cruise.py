"""Cruising (cruise.py): the walk's way out of each junction and the rolling route, on small hand-made maps; and cruises
driven on the live map where it and the router are here."""
import math
import urllib.error
import urllib.request

import numpy as np

from openpilot.tools.sim.bridge.gta5.map import cruise as cr
from openpilot.tools.sim.bridge.gta5.map.cruise import MAX_THROUGH, Cruise, Restrictions, Walk, Walker, start_link
from openpilot.tools.sim.bridge.gta5.map.lane_match import LaneReading
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, FORWARD, OsmLanes, Section
from openpilot.tools.sim.bridge.gta5.map.osm_pbf import OsmData
from openpilot.tools.sim.bridge.gta5.map.router import Route

MAIN = {'highway': 'primary', 'lanes': '2', 'width': '12', 'name': 'Main St'}
SIDE = {'highway': 'primary', 'lanes': '2', 'width': '12', 'name': 'Side St'}
LANE = {'highway': 'residential', 'lanes': '2', 'width': '10'}
ARM = 400.0  # m: arms long enough to lead on (cruise.LEADS_ON)


def make_map(nodes: dict, ways: dict) -> OsmLanes:
  ids = np.array(sorted(nodes), np.int64)
  data = OsmData(ids, np.array([nodes[i][1] for i in ids], float), np.array([nodes[i][0] for i in ids], float), {}, dict(ways), {})
  return OsmLanes(data, lambda lat, lon: (lon, lat))


def arm(deg: float, length: float = ARM) -> tuple[float, float]:
  """A point `length` m from the origin at `deg` right of north."""
  return length * math.sin(math.radians(deg)), length * math.cos(math.radians(deg))


def junction(arms: dict, approach: dict = MAIN, restrictions: Restrictions | None = None):
  """A road from the south (node 1, way 10: `approach`'s tags) to a junction at the origin (node 2), with ways out
  {way id: (tags, deg right of north, length)} each to its own node (the way id's), where the walk from the south
  goes: (the way it took out, its walk)."""
  nodes = {1: (0.0, -ARM), 2: (0.0, 0.0)}
  ways = {10: (approach, [1, 2])}
  for wid, (tags, deg, *length) in arms.items():
    nodes[wid] = arm(deg, *length)
    ways[wid] = (tags, [2, wid])
  osm = make_map(nodes, ways)
  walk = Walk(Walker(osm, restrictions=restrictions), 1, 2)
  walk.extend(ARM + 50.0, MAX_THROUGH)
  return osm.pairs[(2, walk.nodes[2])][0] if len(walk.nodes) > 2 else None, walk


def test_straight_on_the_same_road():
  """At a crossroads it carries on along its road, and where its road bends off (within SAME_ROAD_TURN) it follows it
  rather than a lesser road straight on; bending further, the way out turning least."""
  took, walk = junction({11: (MAIN, 0.0), 12: (SIDE, -90.0), 13: (SIDE, 90.0)})
  assert took == 11 and walk.choices == [(1, 2)]
  took, _ = junction({11: (MAIN, 50.0), 12: (LANE, 0.0)})
  assert took == 11
  took, _ = junction({11: (MAIN, 80.0), 12: (LANE, 0.0)})
  assert took == 12
  took, _ = junction({11: (SIDE, 30.0), 12: (SIDE, -10.0)})  # neither the road it's on: the one turning least
  assert took == 12


def test_ways_it_never_takes():
  """Not a one-way the wrong way, nor a dead end or a minor road while there's another way on, nor a U-turn, nor a move
  the map's turn restrictions forbid."""
  oneway_in = {**MAIN, 'oneway': '-1'}  # drawn out of the junction, driven into it
  assert junction({11: (oneway_in, 0.0), 12: (SIDE, 90.0)})[0] == 12
  assert junction({11: (SIDE, 80.0, 100.0), 12: (SIDE, -90.0)})[0] == 12  # 11: a dead end 100 m on
  assert junction({11: ({**LANE, 'highway': 'service'}, 0.0), 12: (SIDE, 90.0)})[0] == 12
  assert junction({11: ({**LANE, 'gta:switched_off': 'yes'}, 0.0), 12: (SIDE, 90.0)})[0] == 12
  took, walk = junction({11: (MAIN, 170.0)})  # the only way on turns back
  assert took is None and walk.ended
  rules = Restrictions({1: ({'type': 'restriction', 'restriction': 'no_straight_on'}, [('w', 10, 'from'), ('n', 2, 'via'), ('w', 11, 'to')])})
  assert junction({11: (MAIN, 0.0), 12: (SIDE, 90.0), 13: (SIDE, -70.0)}, restrictions=rules)[0] == 13
  # a run of ways: from 10 along 11 on to 12 is forbidden (a U-turn through a median, say)
  run = Restrictions({1: ({'type': 'restriction', 'restriction': 'no_u_turn'}, [('w', 10, 'from'), ('w', 11, 'via'), ('w', 12, 'to')])})
  assert run.allows([10], 2, 11) and not run.allows([10, 11], 7, 12) and run.allows([9, 11], 7, 12)
  assert run.allows([10], 2, 10)  # on along the same way is no turn


def test_turn_only_lane():
  """The car in a turn-only lane (its turn:lanes arrows) turns that way at the first junction; in a lane straight on
  it carries on; past the first junction its lane says nothing."""
  approach = {'highway': 'primary', 'oneway': 'yes', 'lanes': '2', 'width': '8', 'turn:lanes': 'left|through', 'name': 'Main St'}
  nodes = {1: (0.0, -ARM), 2: (0.0, 0.0), 3: (0.0, 200.0), 11: arm(-90.0), 12: arm(90.0), 4: (0.0, 200.0 + ARM),
           13: (-ARM, 200.0)}
  ways = {10: (approach, [1, 2]), 11: (SIDE, [2, 11]), 12: (SIDE, [2, 12]), 14: (MAIN, [2, 3]), 15: (MAIN, [3, 4]),
          16: (SIDE, [3, 13])}
  osm = make_map(nodes, ways)
  walker = Walker(osm)
  sec = Section.of(osm.lanes(10), FORWARD)

  class Matcher:
    def __init__(self, lane, direction=FORWARD):
      self.lane, self.direction = lane, direction

    def match(self, x, y, heading, z=None):
      return LaneReading(10, self.direction, self.lane, 2, 'own', False, sec.offset(self.lane), 0.0)

  pos = np.array([0.0, -100.0])
  u, v, turn = start_link(osm, walker, pos, 0.0, None, Matcher(0))
  assert (u, v, turn) == (1, 2, 'left')
  walk = Walk(walker, u, v, turn, car=ARM - 100.0)
  walk.extend(ARM + 50.0, MAX_THROUGH)
  assert walk.nodes[:3] == [1, 2, 11]
  u, v, turn = start_link(osm, walker, pos, 0.0, None, Matcher(1))
  assert turn is None
  walk = Walk(walker, u, v, turn, car=ARM - 100.0)
  walk.extend(ARM + 300.0, MAX_THROUGH)
  assert walk.nodes[:4] == [1, 2, 3, 4]
  walk = Walk(walker, 1, 2, 'left', car=ARM - 200.0)  # the junction further on than LANE_REACH: the lane says nothing
  walk.extend(ARM + 50.0, MAX_THROUGH)
  assert walk.nodes[:3] == [1, 2, 3]
  walk = Walk(walker, 1, 2, 'left', car=ARM - 100.0)
  walk.extend(ARM + 300.0, MAX_THROUGH)
  assert walk.nodes[2] == 11 and walk.turn is None  # spent at the first junction
  assert start_link(osm, walker, pos, 180.0, None, Matcher(0, BACKWARD)) is None  # against the one-way


class FakeRouter:
  """Valhalla stood in for: a route straight from the car through the locations, each request kept."""
  def __init__(self, osm: OsmLanes):
    self.osm, self.roads, self.paths = osm, None, None
    self.requests: list[tuple] = []

  def via(self, start, through, end) -> Route:
    self.requests.append((start, list(through), end))
    return Route(np.vstack([start[0], *[p for p, _ in through], end[0]]), [])

  def crossings(self, points):
    return []


def test_rolling_route(monkeypatch):
  """The cruise route from the car: the first a short one (FIRST_AHEAD), each next walked on from the last's end with
  the roads already chosen kept, through no more than MAX_THROUGH junctions; at the road's end it ends, and walks no
  further."""
  # Main St north 5 km, with a side street off it every 50 m (a choice each: the side streets lead on) to 4.5 km
  nodes, ways = {}, {}
  for k in range(101):
    nodes[k + 1] = (0.0, 50.0 * k)
    if k:
      ways[k] = (MAIN, [k, k + 1])
    if 0 < k < 90:
      nodes[1000 + k] = (ARM, 50.0 * k)
      ways[1000 + k] = (SIDE, [k + 1, 1000 + k])
  osm = make_map(nodes, ways)
  monkeypatch.setattr(cr, "strays", lambda points, line, *a, **kw: None)  # the fake's straight lines aren't the roads
  c = Cruise(FakeRouter(osm))
  route, walk = c.make(np.array([0.0, 10.0]), 0.0, None)
  assert route.cruise and walk.nodes[:2] == [1, 2]
  assert cr.FIRST_AHEAD <= walk.along[-1] < cr.FIRST_AHEAD + 60.0
  assert all(abs(osm.node_xy(n)[0]) < 1e-6 for n in walk.nodes)  # straight on along Main St, past every side street
  first = list(walk.nodes)
  route2, walk2 = c.make(np.array([0.0, 300.0]), 0.0, None, walk)
  k = first.index(walk2.nodes[0])
  assert route2 is not None and 4 <= k <= 6 and walk2.nodes[:len(first) - k] == first[k:]  # trimmed to the car's link
  _, through, _ = c.router.requests[-1]
  assert len(through) == MAX_THROUGH and all(p[1] > 300.0 for p, _ in through)
  assert walk2.along[-1] < cr.AHEAD  # held to MAX_THROUGH junctions ahead
  assert c.make(np.array([0.0, 300.0]), 0.0, None, walk2)[0] is None  # no further than it has: none made
  far, walk3 = c.make(np.array([0.0, 4800.0]), 0.0, None)
  assert far is not None and walk3.ended and osm.node_xy(walk3.nodes[-1])[1] == 5000.0
  assert c.make(np.array([0.0, 4900.0]), 0.0, None, walk3)[0] is None


def test_cruise_updates(monkeypatch):
  """Cruise.update: no route until one is made on its thread; walked on as the car nears its end; walked afresh when
  the car leaves it; and none once reset."""
  nodes = {k + 1: (0.0, 100.0 * k) for k in range(51)} | {1000: (40.0, 0.0), 1001: (40.0, 5000.0)}  # and a road beside it
  osm = make_map(nodes, {k: (MAIN, [k, k + 1]) for k in range(1, 51)} | {1000: (SIDE, [1000, 1001])})
  monkeypatch.setattr(cr, "strays", lambda points, line, *a, **kw: None)
  c = Cruise(FakeRouter(osm))
  calls = []

  def start(pos, bearing, z, now, fresh):
    calls.append(fresh)
    route, walk = c.make(pos, bearing, z, None if fresh else c.walk)
    if route is not None:
      c.route, c.walk = route, walk
  monkeypatch.setattr(c, "_start", start)
  assert c.update(np.array([0.0, 10.0]), 0.0, 0.0) is None and calls == [True]
  r = c.update(np.array([0.0, 10.0]), 0.0, 0.1)  # the first, short, is walked on at once
  assert r is not None and r.cruise and calls == [True, False] and c.route.length > r.length
  r = c.route
  c.update(np.array([0.0, 10.0 + r.length - cr.EXTEND_AT + 5.0]), 0.0, 1.0)
  assert calls == [True, False, False] and c.route is not r
  off = c.route.points[0] + np.array([40.0, 50.0])  # on the road beside it
  c.update(off, 0.0, 2.0)
  c.update(off, 0.0, 2.0 + Cruise.OFF_FOR + 0.1)
  assert calls[-1] is True and c.route.points[-1][0] == 40.0
  c.reset()
  assert c.route is None and c.walk is None


def live():
  """The live map and a router answering, else None."""
  from openpilot.tools.sim.bridge.gta5 import mapdrive_sim as ms
  if ms.live_map() is None:
    return None
  try:
    urllib.request.urlopen("http://127.0.0.1:8002/status", timeout=2.0)
  except (OSError, urllib.error.URLError):
    return None
  return ms


# start points on the live map (where live cars stood: x, y, heading), and their lane from the left
LIVE_STARTS = [((-224.0, 79.0, 352.0), 0), ((-506.2, -273.8, 112.0), 0), ((776.8, -499.5, 27.0), 9),
               ((17.0, -1234.8, 271.0), 0), ((1811.5, 2113.8, 188.0), 0)]


def test_live_cruises():
  """Cruises on the live map, through Valhalla: 2 km from each start, the map driver on the lagged car, without it
  stopping or aborting, a U-turn, or the car in an oncoming lane or against a one-way outside a junction."""
  ms = live()
  if ms is None:
    print("live cruises: skipped (no live map or router)")
    return
  from openpilot.tools.sim.bridge.gta5.map.lane_match import JunctionAreas
  from openpilot.tools.sim.bridge.gta5.map.router import Router
  m = ms.live_map()
  router = Router("http://127.0.0.1:8002", timeout=10.0, paths=m["paths"], osm=m["osm"], roads=m["osm"])
  areas = JunctionAreas.cached(m["osm"], build=False)
  for pose, lane in LIVE_STARTS:
    ct = ms.cruise(pose, router, distance=2000.0, areas=areas, lane=lane)
    md = ct.trip.md
    assert md.finished is None and ct.driven >= 2000.0, (pose, md.finished, ct.driven)
    assert not ct.uturns and not ct.wrong_way, (pose, ct.uturns, ct.wrong_way)
    assert len(ct.routes) >= 2 and all(r.cruise for r in ct.routes)
  print(f"live cruises: ok ({len(LIVE_STARTS)} x 2 km)")
