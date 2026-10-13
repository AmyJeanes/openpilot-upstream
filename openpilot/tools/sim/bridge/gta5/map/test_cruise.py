"""Cruising (cruise.py): the walk's way out of each junction and the rolling route, on small hand-made maps; and cruises
driven on the live map where it and the router are here."""
import math
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import numpy as np

from openpilot.tools.sim.bridge.gta5.map import cruise as cr
from openpilot.tools.sim.bridge.gta5.map.cruise import MAX_THROUGH, Cruise, Exit, Restrictions, Roaming, Walk, Walker, start_link
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
    nodes[wid], nodes[wid + 100] = arm(deg, *length), arm(deg, 1.5 * (length[0] if length else ARM))  # and on a link
    ways[wid] = (tags, [2, wid, wid + 100])
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


def test_banned_inside_junction():
  """A banned link inside a junction (a short link across it, before the way out's own) rules out every way out across
  it, as it does the way out whose own link it is."""
  nodes = {1: (0.0, -ARM), 2: (0.0, 0.0), 3: (0.0, 8.0), 7: (0.0, 16.0), 4: arm(0.0), 5: arm(-60.0), 6: arm(90.0)}
  ways = {10: (MAIN, [1, 2]), 11: (MAIN, [2, 3, 7, 4]), 12: (SIDE, [7, 5]), 13: (SIDE, [2, 6])}
  osm = make_map(nodes, ways)
  for banned, took in ((set(), 4), ({(3, 7)}, 6)):
    walk = Walk(Walker(osm, banned), 1, 2)
    walk.extend(ARM + 50.0, MAX_THROUGH)
    assert took in walk.nodes and (7 in walk.nodes) == (not banned)



def test_pockets():
  """A dead end further on than the walk looks for one (LEADS_ON) is a pocket all the same: by the map's roaming links
  it carries on round the loop rather than straight on along its road into the cul-de-sac. Started in the pocket it
  drives out of it; heading into it, it drives to its end, where the road ends. A loop shorter than ROAM_MIN is a
  pocket too, as a car park's."""
  # a loop of Main St (a square, 600 m a side, each way round it a roaming set) with a 600 m cul-de-sac straight on
  # from its south-west corner's road
  nodes = {1: (0.0, -600.0), 2: (0.0, 0.0), 3: (600.0, 0.0), 4: (600.0, -600.0), 5: (0.0, 600.0), 6: (0.0, 300.0)}
  ways = {10: (MAIN, [1, 2]), 11: (MAIN, [2, 3]), 12: (MAIN, [3, 4]), 13: (MAIN, [4, 1]), 14: (MAIN, [2, 6, 5])}
  osm = make_map(nodes, ways)
  roaming = Roaming.find(osm)
  assert roaming.links == {(1, 2), (2, 3), (3, 4), (4, 1), (2, 1), (3, 2), (4, 3), (1, 4)}
  walk = Walk(Walker(osm), 1, 2)
  walk.extend(650.0, MAX_THROUGH)
  assert walk.nodes[2] == 6  # by LEADS_ON alone: straight on into it
  walk = Walk(Walker(osm, roaming=roaming), 1, 2)
  walk.extend(3000.0, MAX_THROUGH)
  assert walk.nodes[:6] == [1, 2, 3, 4, 1, 2] and not walk.ended
  small = make_map({k: (x / 2, y / 2) for k, (x, y) in nodes.items()}, ways)
  assert not Roaming.find(small).links
  walk = Walk(Walker(osm, roaming=roaming), 6, 2)  # in the pocket, heading out
  walk.extend(1000.0, MAX_THROUGH)
  assert walk.nodes[2] in (1, 3) and not walk.ended
  walk = Walk(Walker(osm, roaming=roaming), 2, 6)  # heading in: no way on, nor round
  walk.extend(1000.0, MAX_THROUGH)
  assert walk.nodes == [2, 6, 5] and walk.ended


def test_turn_only_lane():
  """The car in a turn-only lane (its turn:lanes arrows) turns that way at the first junction; in a lane straight on
  it carries on; past the first junction its lane says nothing."""
  approach = {'highway': 'primary', 'oneway': 'yes', 'lanes': '2', 'width': '8', 'turn:lanes': 'left|through', 'name': 'Main St'}
  nodes = {1: (0.0, -ARM), 2: (0.0, 0.0), 3: (0.0, 200.0), 11: arm(-90.0), 12: arm(90.0), 4: (0.0, 200.0 + ARM),
           13: (-ARM, 200.0), 111: arm(-90.0, 2 * ARM), 112: arm(90.0, 2 * ARM), 104: (0.0, 200.0 + 2 * ARM)}
  ways = {10: (approach, [1, 2]), 11: (SIDE, [2, 11, 111]), 12: (SIDE, [2, 12, 112]), 14: (MAIN, [2, 3]),
          15: (MAIN, [3, 4, 104]), 16: (SIDE, [3, 13])}
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
    if 0 < k < 90:  # each on to an avenue beside Main St
      nodes[1000 + k] = (ARM, 50.0 * k)
      ways[1000 + k] = (SIDE, [k + 1, 1000 + k])
      if k > 1:
        ways[2000 + k] = (LANE, [1000 + k - 1, 1000 + k])
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


def grid():
  """Main St north along x = 0 (way 2000 + 200 + row) between West Ave and East Ave (x = -400, 400), a street across
  every 300 m (Row 0 to Row 7, west and east halves) but for Row 2's west half: every road leads on round the grid, so
  none is a pocket. Node 100 * column + row."""
  nodes = {100 * c + r: (400.0 * (c - 2), 300.0 * r) for c in (1, 2, 3) for r in range(8)}
  names = {1: 'West Ave', 2: 'Main St', 3: 'East Ave'}
  ways = {2000 + 100 * c + r: ({**MAIN, 'name': names[c]}, [100 * c + r, 100 * c + r + 1]) for c in (1, 2, 3) for r in range(7)}
  for r in range(8):
    if r != 2:
      ways[1000 + 10 * r + 1] = ({**SIDE, 'name': f'Row {r}'}, [100 + r, 200 + r])
    ways[1000 + 10 * r + 2] = ({**SIDE, 'name': f'Row {r}'}, [200 + r, 300 + r])
  return make_map(nodes, ways)


def cruising(monkeypatch, osm, y: float) -> Cruise:
  """A cruise from Main St at y heading north, along it."""
  monkeypatch.setattr(cr, "strays", lambda points, line, *a, **kw: None)
  c = Cruise(FakeRouter(osm))
  c.route, c.walk = c.make(np.array([0.0, y]), 0.0, None)
  assert all(abs(osm.node_xy(n)[0]) < 1e-6 for n in c.walk.nodes)  # straight on along Main St
  return c


def routed(c: Cruise):
  """Waits for the route being made, and swaps it in as update would (a prompt's waits in pending for update)."""
  for _ in range(500):
    if not c.busy:
      if c.pending is not None:
        (c.route, c.walk), c.pending, c.prompted = c.pending, None, None
      return
    time.sleep(0.01)
  raise AssertionError("not routed")


def test_side_exit():
  """At a fork (every way out within FORK_TURN) an arrow keeps to its side; elsewhere it's the turn that way, if any."""
  w = Walker(make_map({1: (0.0, 0.0), 2: (0.0, 1.0)}, {}))
  fork = [Exit(15.0, [0, 1]), Exit(-20.0, [0, 2])]
  assert w.side_exit('left', fork, fork) == (fork[0], True) and w.side_exit('right', fork, fork) == (fork[1], True)
  tee = [Exit(-2.0, [0, 1]), Exit(-90.0, [0, 2])]
  assert w.side_exit('left', tee, tee) == (None, False) and w.side_exit('right', tee, tee) == (tee[1], False)


def test_prompt_turns(monkeypatch):
  """An arrow key turns at the next junction with a way out that side within PROMPT_SECS s (PROMPT_MIN m), routed
  again from there, not one nearer than the car can still turn at (prompt_need); the other arrow undoes it; a junction
  with no way out that side is passed by (a lane change instead), and one the turn restrictions forbid is said to be."""
  osm = grid()
  c = cruising(monkeypatch, osm, 150.0)
  pos = np.array([0.0, 150.0])
  assert c.prompt('left', pos, 0.0, None, 9.0) == ('none', '')  # Row 1 150 m on: beyond 120 m at 9 m/s
  assert c.prompt('left', pos, 0.0, None, 15.0) == ('turn', 'Row 1')
  routed(c)
  k = c.walk.nodes.index(201)
  assert c.walk.nodes[k + 1] == 101 and c.route.cruise and c.forced['side'] == 'left'
  assert c.prompt('left', pos, 0.0, None, 15.0) == ('turn', 'Row 1')  # already: nothing new
  assert not c.busy
  assert c.prompt('right', np.array([0.0, 285.0]), 0.0, None, 10.0) == ('late', '')  # too near to undo
  assert not c.busy and c.forced is not None
  assert c.prompt('right', pos, 0.0, None, 15.0) == ('cancel', 'Row 1')
  routed(c)
  k = c.walk.nodes.index(201)
  assert c.walk.nodes[k + 1] == 202 and c.forced is None
  assert c.prompt('right', pos, 0.0, None, 15.0) == ('turn', 'Row 1')
  routed(c)
  assert c.walk.nodes[c.walk.nodes.index(201) + 1] == 301
  c = cruising(monkeypatch, osm, 250.0)  # 50 m short of Row 1: too late at 15 m/s
  assert c.prompt('left', np.array([0.0, 250.0]), 0.0, None, 15.0) == ('late', '')
  c = cruising(monkeypatch, osm, 258.0)  # 42 m short: in time at 4 m/s (its lane unknown: one lane change allowed for)
  assert c.prompt('left', np.array([0.0, 258.0]), 0.0, None, 4.0) == ('turn', 'Row 1')
  c = cruising(monkeypatch, osm, 500.0)  # Row 2 has no west half: no left there
  assert c.prompt('left', np.array([0.0, 500.0]), 0.0, None, 15.0) == ('none', '')
  rules = Restrictions({1: ({'type': 'restriction', 'restriction': 'no_left_turn'},
                            [('w', 2200, 'from'), ('n', 201, 'via'), ('w', 1011, 'to')])})
  monkeypatch.setattr(Restrictions, "of", classmethod(lambda cls, osm: rules))
  c = cruising(monkeypatch, osm, 200.0)
  assert c.prompt('left', np.array([0.0, 200.0]), 0.0, None, 15.0) == ('forbidden', '')


def test_prompt_other_way_out(monkeypatch):
  """Where the router won't take the way out an arrow key chose, the junction's next way out that side is tried."""
  monkeypatch.setattr(cr, "strays", lambda points, line, *a, **kw: None)
  nodes = {1: (0.0, -ARM), 2: (0.0, 0.0)}
  ways = {10: (MAIN, [1, 2])}
  for wid, deg in ((11, 0.0), (12, 50.0), (13, 90.0)):  # on, a slip road right, the street right
    nodes[wid], nodes[wid + 100] = arm(deg), arm(deg, 1.5 * ARM)
    ways[wid] = ({**SIDE, 'name': f'Way {wid}'}, [2, wid, wid + 100])
  osm = make_map(nodes, ways)
  c = Cruise(FakeRouter(osm))
  c.route, c.walk = c.make(np.array([0.0, -100.0]), 0.0, None)
  route = c._route

  def refuse(walk, at, ahead, known, keep=None):
    if keep == (2, 12):
      c.banned.add(keep)
      return None, walk
    return route(walk, at, ahead, known, keep)
  monkeypatch.setattr(c, "_route", refuse)
  assert c.prompt('right', np.array([0.0, -100.0]), 0.0, None, 10.0) == ('turn', 'Way 12')
  routed(c)
  assert c.walk.has((2, 13)) and c.forced['link'] == (2, 13) and not c.take_notes()


def test_prompt_crossing(monkeypatch):
  """A way out by one of GTA's shortcut links across a painted separation (Router.crossings), which the route would ban,
  isn't one an arrow key takes: the next that side is, without asking the router; with none, it can't turn there."""
  monkeypatch.setattr(cr, "strays", lambda points, line, *a, **kw: None)
  nodes = {1: (0.0, -ARM), 2: (0.0, 0.0)}
  ways = {10: (MAIN, [1, 2])}
  for wid, deg in ((11, 0.0), (12, 50.0), (13, 90.0)):
    nodes[wid], nodes[wid + 100] = arm(deg), arm(deg, 1.5 * ARM)
    ways[wid] = ({**SIDE, 'name': f'Way {wid}'}, [2, wid, wid + 100])
  osm = make_map(nodes, ways)
  for across, want in (({12}, ('turn', 'Way 13')), ({12, 13}, ('forbidden', ''))):
    router = FakeRouter(osm)
    spots = [osm.node_xy(n) for n in across]
    router.crossings = lambda points, spots=spots: [p for p in spots if any(np.hypot(*(q - p)) < 1.0 for q in points)]
    c = Cruise(router)
    c.route, c.walk = c.make(np.array([0.0, -100.0]), 0.0, None)
    asked = len(router.requests)
    assert c.prompt('right', np.array([0.0, -100.0]), 0.0, None, 10.0) == want
    routed(c)
    assert not c.banned and not c.take_notes()
    if want[0] == 'turn':
      assert c.walk.has((2, 13)) and c.forced['link'] == (2, 13) and len(router.requests) == asked + 1
    else:
      assert len(router.requests) == asked


def test_prompt_too_late(monkeypatch):
  """An arrow key's route goes in with the next update while the car can still make the turn; one that comes after the
  car's nearer than that (routing can take seconds) is dropped, said to be too late."""
  osm = grid()
  for y, late in ((160.0, False), (290.0, True)):
    c = cruising(monkeypatch, osm, 150.0)
    assert c.prompt('left', np.array([0.0, 150.0]), 0.0, None, 15.0) == ('turn', 'Row 1')
    while c.busy:
      time.sleep(0.01)
    made = c.pending[0]
    c.update(np.array([0.0, y]), 0.0, -1.0)  # (before its next try: no route walked on meanwhile)
    if late:
      assert c.pending is None and c.route is not made and c.forced is None and c.take_notes() == [('late', 'left', '')]
    else:
      assert c.route is made and c.forced is not None and not c.take_notes()


def test_prompt_need():
  """How near a turn may be: by the lane changes into a lane for it (turn:lanes arrows, else the outermost lane that
  side; a pocket opening on that side counted), and by speed, braking for a turn but not a fork."""
  approach = {'highway': 'primary', 'oneway': 'yes', 'lanes': '3', 'width': '12', 'turn:lanes': 'left|through|through;right'}
  nodes = {1: (0.0, -ARM), 2: (0.0, 0.0), 11: arm(-90.0), 12: arm(90.0), 13: arm(0.0)}
  osm = make_map(nodes, {10: (approach, [1, 2]), 11: (SIDE, [2, 11]), 12: (SIDE, [2, 12]), 13: (SIDE, [2, 13])})
  walk = Walk(Walker(osm), 1, 2)
  assert [cr.lanes_over(walk, 1, 'left', [i, 3]) for i in range(3)] == [0, 1, 2]
  assert [cr.lanes_over(walk, 1, 'right', [i, 3]) for i in range(3)] == [2, 1, 0]
  assert cr.lanes_over(walk, 1, 'left', [0, 2]) == 1  # the left-turn pocket opens left of the car's lane
  assert cr.lanes_over(walk, 1, 'left', None) == 1
  plain = make_map(nodes, {10: ({**approach, 'turn:lanes': ''}, [1, 2]), 11: (SIDE, [2, 11]), 12: (SIDE, [2, 12])})
  walk = Walk(Walker(plain), 1, 2)
  assert cr.lanes_over(walk, 1, 'right', [0, 3]) == 2 and cr.lanes_over(walk, 1, 'left', [0, 3]) == 0
  assert cr.prompt_need(5.0, 0, False) < 30.0 < cr.prompt_need(10.0, 0, False) < cr.prompt_need(10.0, 1, False)
  assert cr.prompt_need(20.0, 0, True) < cr.prompt_need(20.0, 0, False)


def test_prompt_swap():
  """An arrow key's route waits only while the car's in a turn or in a junction (then dropped should it be too late,
  test_prompt_too_late); a route walked on waits for the car to be clear of any junction."""
  c = Cruise(FakeRouter(grid()))
  prompted, walked = object(), object()
  c.prompted = prompted
  ahead = SimpleNamespace(at=100.0, junctions=[125.0], lanes=None, turns=lambda d, behind=0.0: [])  # a junction 25 m on
  inside = SimpleNamespace(at=100.0, junctions=[105.0], lanes=None, turns=lambda d, behind=0.0: [])
  turning = SimpleNamespace(at=100.0, junctions=[105.0], lanes=None, turns=lambda d, behind=0.0: [[5.0, 'left', 0.0, 90.0]])
  assert not c._held(ahead, prompted) and c._held(inside, prompted) and c._held(turning, prompted)
  assert c._held(ahead, walked)


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
