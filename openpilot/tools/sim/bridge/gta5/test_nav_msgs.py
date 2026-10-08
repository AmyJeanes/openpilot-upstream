"""The nav messages (gta5_nav_msgs.py) on the map fixtures, and the UI's reading of them (selfdrive/ui/nav/nav_state.py).
No pytest needed: `python test_nav_msgs.py` runs them all."""
import time

import numpy as np

from openpilot.cereal import messaging
from openpilot.selfdrive.ui.nav.nav_state import NavState, format_distance
from openpilot.selfdrive.navd import lane_slots as ls
from openpilot.tools.sim.bridge.gta5 import gta5_nav_msgs as nm
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import TAPER_M
from openpilot.tools.sim.bridge.gta5.test_lane_slots import DEST, LEFT, START, TWO_EACH_WAY, fixture, osm_map, route


def bay_route():
  """bay_two_way.osm: a left turn from a bay opening between our two lanes and the two oncoming, 270 m along."""
  osm, nodes = fixture('bay_two_way.osm')
  r = route(osm, nodes, [1, 2, 3, 4], {2: LEFT})
  r.maneuvers[0].update(street_names=['Alta St'], time=20.0)
  r.maneuvers[1].update(street_names=['Bay Rd'], time=10.0)
  r.maneuvers[2].update(street_names=[], time=0.0)
  return osm, r, ls.LaneSlots(r)


class PM:
  def __init__(self):
    self.sent: list[tuple[str, object]] = []

  def send(self, service, msg):
    self.sent.append((service, messaging.log_from_bytes(msg) if isinstance(msg, bytes) else msg.as_reader()))

  def last(self, service):
    return next(getattr(m, service) for s, m in reversed(self.sent) if s == service)


def test_valhalla_maneuvers():
  assert nm.MANEUVERS[10] == ("turn", "right") and nm.MANEUVERS[21] == ("off ramp", "slight left")
  _, r, _ = bay_route()
  mans = nm.maneuvers(r)
  assert [(m.type, m.modifier, m.primary) for m in mans] == [("turn", "left", "Bay Rd"), ("arrive", "straight", "")]
  assert mans[0].along == r.along[2] and mans[1].along == r.length
  assert nm.destination_name(r) == "Bay Rd"


def test_time_remaining():
  _, r, _ = bay_route()
  turn = float(r.along[2])
  assert abs(nm.time_remaining(r, 0.0) - 30.0) < 1e-6
  assert abs(nm.time_remaining(r, turn / 2) - 20.0) < 1e-6  # half the first stretch's 20 s left, and the 10 s after
  assert nm.time_remaining(r, r.length) == 0.0
  r.maneuvers[0].pop('time')
  assert abs(nm.time_remaining(r, 0.0) - r.length / nm.DEFAULT_SPEED) < 1e-6


def test_instruction():
  _, r, slots = bay_route()
  msg = messaging.new_message('navInstruction')
  nm.fill_instruction(msg.navInstruction, r, 20.0, 10.0, lanes=nm.lane_guide(slots, 20.0, 10.0, [1, 2]))
  ni = msg.navInstruction
  assert ni.valid and ni.maneuverType == "turn" and ni.maneuverModifier == "left" and ni.maneuverPrimaryText == "Bay Rd"
  assert abs(ni.maneuverDistance - (r.along[2] - 20.0)) < 1e-3
  assert [(m.type, m.primaryText) for m in ni.allManeuvers] == [("turn", "Bay Rd"), ("arrive", "")]
  assert abs(ni.distanceRemaining - (r.length - 20.0)) < 1e-3 and ni.destinationName == "Bay Rd"
  x, y = to_game(ni.position.latitude, ni.position.longitude)
  assert np.hypot(x - nm.point_at(r, 20.0)[0], y - nm.point_at(r, 20.0)[1]) < 0.01
  d = r.points[1] - r.points[0]
  assert abs(ni.bearingDeg - np.degrees(np.arctan2(d[0], d[1])) % 360) < 0.01
  assert ni.showFull and len(ni.lanes) == 4


def test_lanes_before_the_bay_opens():
  # until the bay is there, the target is the inner through lane it opens from, by where it opens (rows' 250 m on)
  _, r, slots = bay_route()
  opens = 240.0 + TAPER_M
  g = nm.lane_guide(slots, 20.0, 10.0, [1, 2])  # in the kerb lane
  assert [lane["oncoming"] for lane in g.lanes] == [True, True, False, False]  # left to right
  assert [lane["active"] for lane in g.lanes] == [False, False, True, False]
  assert [lane["current"] for lane in g.lanes] == [False, False, False, True]
  assert g.show and abs(g.distance - (opens - 20.0)) < 0.01 and abs(g.opens - (opens - 20.0)) < 0.01
  assert g.lanes[2]["activeDirection"] in g.lanes[2]["directions"]
  g = nm.lane_guide(slots, 20.0, 10.0, [0, 2])  # already in it
  assert g.distance == 0.0 and g.lanes[2]["current"] and g.lanes[2]["active"]


def test_lanes_in_the_bay():
  _, r, slots = bay_route()
  g = nm.lane_guide(slots, 285.0, 10.0, [1, 3])
  assert len(g.lanes) == 5 and [lane["active"] for lane in g.lanes] == [False, False, True, False, False]
  assert g.lanes[2]["activeDirection"] == "left" and g.opens == 0.0 and g.distance == 0.0  # due now: no distance left
  assert nm.lane_guide(slots, 285.0, 10.0, [0, 3]).lanes[2]["current"]


def test_no_lanes_shown_before_the_window():
  osm, nodes = fixture('turn_bay.osm')
  r = route(osm, nodes, [1, 2, 3, 5], {2: LEFT})
  g = nm.lane_guide(ls.LaneSlots(r), 40.0, 0.0, [0, 1])
  assert g.lanes and not g.show and not any(lane["active"] for lane in g.lanes)
  assert nm.lane_guide(None, 40.0, 0.0, None) == nm.NO_LANES


def test_directions():
  assert nm.directions(frozenset()) == ["straight"]
  assert nm.directions(frozenset({"through", "left"})) == ["left", "straight"]
  assert nm.directions(frozenset({"slight_right", "sharp_right"})) == ["slightRight", "right"]


def test_simplify():
  line = np.array([[0.0, 0.0], [10.0, 0.1], [20.0, 0.0], [20.0, 30.0], [20.3, 40.0], [20.0, 50.0]])
  out = nm.simplify(line, 1.0)
  assert out.tolist() == [[0.0, 0.0], [20.0, 0.0], [20.0, 50.0]]
  assert len(nm.simplify(line, 0.01)) == len(line)
  dense = nm.densify(np.array([[0.0, 0.0], [100.0, 0.0]]), 25.0)
  assert len(dense) == 5 and np.all(np.diff(dense[:, 0]) <= 25.0)


def test_roads_near():
  # a road beside the route, one crossing it, and one far off; a car park's service road is left off
  nodes = {1: (0.0, 0.0), 2: (0.0, 1000.0), 3: (100.0, -500.0), 4: (100.0, 1500.0), 5: (-500.0, 500.0), 6: (500.0, 500.0),
           7: (2000.0, 0.0), 8: (2000.0, 1000.0), 9: (10.0, 10.0), 10: (20.0, 20.0)}
  service = {'highway': 'service'}
  osm = osm_map(nodes, {1: (TWO_EACH_WAY, [1, 2]), 2: (TWO_EACH_WAY, [3, 4]), 3: (TWO_EACH_WAY, [5, 6]),
                        4: (TWO_EACH_WAY, [7, 8]), 5: (service, [9, 10])})
  roads = nm.Roads(osm)
  assert len(roads.points) == 4
  near = roads.near(np.array([[0.0, 200.0], [0.0, 800.0]]), width=250.0)
  assert len(near) == 3 and all(abs(w - 14.0) < 0.01 for _, w in near)
  beside = [pts for pts, _ in near if np.allclose(pts[:, 0], 100.0)][0]
  assert beside[:, 1].min() < -40.0 and beside[:, 1].max() > 1040.0  # clipped to the corridor, a step past its edges
  assert len(roads.near(np.array([[0.0, 200.0], [0.0, 800.0]]), width=250.0, budget=4)) == 2  # nearest first


def test_route_message():
  osm, r, _ = bay_route()
  roads = nm.Roads(osm).near(r.points)
  msg = nm.route_message(r, roads)
  coords = msg.navRoute.coordinates
  xy = np.array([to_game(c.latitude, c.longitude) for c in coords])
  assert np.allclose(xy[0], r.points[0], atol=0.01) and np.allclose(xy[-1], r.points[-1], atol=0.01)
  assert len(msg.navRoute.roads) == len(roads) and msg.navRoute.roads[0].width == roads[0][1]
  assert len(nm.route_message(None, []).navRoute.coordinates) == 0


def test_publishing():
  osm, r, slots = bay_route()
  pm = PM()
  pub = nm.NavMessages(pm)
  pub.update(None, 0.0, osm, now=0.0)
  assert not pm.last('navInstruction').valid and len(pm.last('navRoute').coordinates) == 0
  r.at = 20.0
  pub.update(r, 10.0, osm, lambda route: slots, now=0.05)  # within INSTRUCTION_EVERY: no instruction yet
  assert not pm.last('navInstruction').valid
  assert len(pm.last('navRoute').coordinates) > 1 and len(pm.last('navRoute').roads) == 0  # the route at once
  for _ in range(100):
    if not pub.busy:
      break
    time.sleep(0.02)
  pub.update(r, 10.0, osm, lambda route: slots, now=0.2)
  assert pm.last('navInstruction').valid and len(pm.last('navRoute').roads) > 0  # then the roads, once gathered
  sent = len(pm.sent)
  pub.update(r, 10.0, osm, lambda route: slots, now=0.25)
  assert len(pm.sent) == sent  # nothing due
  pub.update(r, 10.0, osm, lambda route: slots, now=0.2 + nm.ROUTE_EVERY)
  assert [s for s, _ in pm.sent[sent:]] == ['navInstruction', 'navRoute']


def test_ui_reads_them():
  osm, r, slots = bay_route()
  pm = PM()
  pub = nm.NavMessages(pm)
  r.at = 20.0
  pub.update(r, 10.0, osm, lambda route: slots, now=1.0)
  nav = NavState()
  nav._read_route(pm.last('navRoute'))
  fix = nav.read_instruction(pm.last('navInstruction'))
  nav.pose.update(0.0, 0.0, 0.0, fix)
  g = nav.guidance
  assert g.maneuver.type == "turn" and g.show_lanes and abs(g.lane_open_distance - (240.0 + TAPER_M - 20.0)) < 0.01
  assert [lane.oncoming for lane in g.lanes] == [True, True, False, False]
  pos, bearing = nav.car()
  along, _ = nav.along_route(pos, bearing)
  assert abs(along - 20.0) < 0.5 and abs(nav.route_along[-1] - r.length) < 1.0
  turn = nav.route_point(along + g.maneuver.distance)
  assert np.hypot(*(turn - nav.projection.local(*np.array(to_lat_lon_of(r.points[2]))))) < 1.0
  version = nav.version
  nav._read_route(pm.last('navRoute'))
  assert nav.version == version  # the same route sent again isn't read again


def to_lat_lon_of(p):
  return nm.to_lat_lon(float(p[0]), float(p[1]))


def test_distances_read_as_a_driver_would():
  assert format_distance(183.0, True) == "180 m" and format_distance(1234.0, True) == "1.2 km"
  assert format_distance(60.0, False) == "200 ft" and format_distance(400.0, False) == "0.2 mi"


def test_valhalla_types_all_named():
  for kind, (t, mod) in nm.MANEUVERS.items():
    assert t and mod, kind
  assert START in nm.MANEUVERS and DEST in nm.MANEUVERS


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
