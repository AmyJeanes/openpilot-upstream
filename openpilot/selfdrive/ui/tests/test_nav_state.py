import math

import numpy as np

from openpilot.cereal import messaging
from openpilot.selfdrive.ui.nav.draw import maneuver_kind
from openpilot.selfdrive.ui.nav.nav_state import (M_PER_DEG, ROUTE_GONE_S, CardLane, CarLane, NavState, PoseTracker, Projection,
                                                  RouteStarts, card_lanes, format_duration, format_trip_distance, instruction_guidance,
                                                  maneuver_phase, model_lane, wrap)
from openpilot.selfdrive.ui.nav.text import maneuver_road


def route_msg(points_m: list[tuple[float, float]], lat0: float = 51.5):
  """A navRoute through points (m east, north) about lat0."""
  msg = messaging.new_message('navRoute')
  coords = msg.navRoute.init('coordinates', len(points_m))
  for c, (x, y) in zip(coords, points_m, strict=True):
    c.latitude = float(lat0 + y / M_PER_DEG)
    c.longitude = float(x / (M_PER_DEG * np.cos(np.radians(lat0))))
  return msg.navRoute


def test_projection_is_metres():
  p = Projection(51.5, -0.1)
  east, north = p.local(51.5 + 100 / M_PER_DEG, -0.1 + 50 / (M_PER_DEG * np.cos(np.radians(51.5))))
  assert abs(east - 50.0) < 1e-6 and abs(north - 100.0) < 1e-6


def test_along_route_takes_the_part_driven():
  # north 200 m, then a hairpin back south 20 m to the east: from beside the way back, heading north, the car is on
  # the way out
  nav = NavState()
  nav._read_route(route_msg([(0.0, 0.0), (0.0, 200.0), (20.0, 200.0), (20.0, 0.0)]))
  along, seg = nav.along_route(np.array([12.0, 100.0]), 0.0)
  assert seg == 0 and abs(along - 100.0) < 0.5
  along, seg = nav.along_route(np.array([12.0, 100.0]), 180.0)
  assert seg == 2 and abs(along - 320.0) < 0.5


def test_guidance_and_lanes():
  msg = messaging.new_message('navInstruction')
  ni = msg.navInstruction
  ni.valid, ni.maneuverType, ni.maneuverModifier, ni.maneuverPrimaryText, ni.maneuverDistance = True, "turn", "right", "Elm St", 180.0
  mans = ni.init('allManeuvers', 2)
  mans[0].distance, mans[0].type, mans[0].modifier = 180.0, "turn", "right"
  mans[1].distance, mans[1].type, mans[1].modifier = 580.0, "turn", "left"
  lanes = ni.init('lanes', 3)
  lanes[0].oncoming = True
  lanes[1].current, lanes[1].directions = True, ["straight"]  # the simulator's truth, which the card ignores
  lanes[2].active, lanes[2].directions, lanes[2].activeDirection = True, ["right"], "right"
  ni.showFull, ni.laneDistance = True, 60.0
  g = instruction_guidance(ni)
  assert g.show_lanes and maneuver_road(g) == "Elm St"
  # the oncoming lane left off; the car in the straight one by the model, the turn lit in the other
  ours = [CardLane("up", False, False), CardLane("right", False, True)]
  assert card_lanes(g) == (ours, None)
  assert card_lanes(g, CarLane(0, 2, True)) == (ours, 0)
  assert card_lanes(g, CarLane(1, 3, True)) == (ours, None)  # the model counts other lanes than nav's
  g.secondary = "Exit 3"
  assert maneuver_road(g) == "Exit 3 · Elm St"
  g.show_lanes = False
  assert card_lanes(g) == ([], None)


def test_lanes_always():
  msg = messaging.new_message('navInstruction')
  ni = msg.navInstruction
  ni.valid = True  # no maneuver near and no lane guidance: the road's lanes, none active
  lanes = ni.init('lanes', 4)
  lanes[0].oncoming = True
  for lane in lanes:
    lane.directions = ["straight"]
  g = instruction_guidance(ni)
  car = CarLane(2, 3, True)
  assert not g.show_lanes and card_lanes(g, car) == ([], None)
  assert card_lanes(g, car, always=True) == ([CardLane("up", False, False)] * 3, 2)
  lanes[2].oncoming = lanes[3].oncoming = True  # one lane our way: nothing to choose between
  assert card_lanes(instruction_guidance(ni), CarLane(0, 1, True), always=True) == ([], None)


def test_car_lane_from_the_lane_head():
  lh = messaging.new_message('modelV2').modelV2.laneHead
  assert model_lane(lh) is None  # a model without the head
  lh.laneIdx, lh.laneCount, lh.prob = 2, 3, 0.8
  assert model_lane(lh) == CarLane(2, 3, True)
  lh.prob = 0.3
  assert model_lane(lh) == CarLane(2, 3, False)


def test_shared_lane_lights_the_branch_taken():
  msg = messaging.new_message('navInstruction')
  ni = msg.navInstruction
  ni.valid, ni.maneuverType, ni.maneuverModifier, ni.maneuverDistance, ni.showFull = True, "turn", "left", 150.0, True
  lanes = ni.init('lanes', 3)
  lanes[0].active, lanes[0].directions, lanes[0].activeDirection = True, ["left"], "left"
  lanes[1].active, lanes[1].directions, lanes[1].activeDirection = True, ["left", "straight"], "left"
  lanes[2].directions = ["straight"]
  got, here = card_lanes(instruction_guidance(ni))
  assert [lane.arrow for lane in got] == ["left", "upleft", "up"] and here is None
  assert got[1] == CardLane("upleft", False, True) and got[2] == CardLane("up", False, False)
  lanes[1].activeDirection = "straight"  # the route goes on straight: that branch lit, not the turn
  assert card_lanes(instruction_guidance(ni))[0][1] == CardLane("upleft", True, False)


def test_phase_opens_ahead_and_closes_a_little_later():
  msg = messaging.new_message('navInstruction')
  ni = msg.navInstruction
  ni.valid, ni.maneuverType, ni.maneuverModifier = True, "turn", "right"
  g = instruction_guidance(ni)
  for dist, was_open, phase in ((1000.0, False, "cruise"), (190.0, False, "approach"), (220.0, True, "approach"),
                                (220.0, False, "cruise"), (15.0, True, "turn")):
    g.maneuver.distance = dist
    assert maneuver_phase(g, 10.0, was_open) == phase, (dist, was_open)
  g.maneuver.distance = 400.0
  assert maneuver_phase(g, 30.0, False) == "approach"  # 15 s ahead at speed


def test_maneuver_icons():
  assert maneuver_kind("turn", "left") == ("right", True)
  assert maneuver_kind("off ramp", "slight right") == ("exit", False)
  assert maneuver_kind("fork", "slight left") == ("slight", True)
  assert maneuver_kind("turn", "uturn") == ("uturn", False)
  assert maneuver_kind("arrive", "right") == ("arrive", False)
  assert maneuver_kind("continue", "straight") == ("straight", False)


def test_pose_follows_the_yaw_rate_between_fixes():
  # a quarter circle of radius 30 m at 6 m/s, fixes at 10 Hz and frames at 20 Hz: the heading moves every frame, by
  # about the yaw rate's step, never by a fix's jump
  t = PoseTracker()
  v, r = 6.0, 30.0
  w = v / r  # rad/s, turning right
  bearings, dt = [], 0.05
  for k in range(int((math.pi / 2) / w / dt) + 1):
    now = k * dt
    a = w * now
    fix = (np.array([r - r * math.cos(a), r * math.sin(a)]), math.degrees(a)) if k % 2 == 0 else None
    t.update(now, v, w, fix)
    bearings.append(t.bearing)
    assert abs(wrap(t.bearing - math.degrees(a))) < 1.0, k
  steps = np.diff(np.unwrap(np.radians(bearings)))
  assert np.all(steps > 0) and np.degrees(steps).max() < 1.5 * math.degrees(w * dt)


def test_pose_eases_towards_fixes_without_a_yaw_rate():
  # no yaw rate (no deviceMotion): a heading 30 deg off is taken in gradually, the short way across north
  t = PoseTracker()
  t.update(0.0, 0.0, 0.0, (np.zeros(2), 350.0))
  t.update(0.05, 0.0, 0.0, (np.zeros(2), 20.0))
  assert 350.0 < t.bearing < 352.0
  for k in range(2, 100):
    t.update(k * 0.05, 0.0, 0.0)
  assert abs(wrap(t.bearing - 20.0)) < 0.5
  t.update(5.0, 0.0, 0.0, (np.array([100.0, 0.0]), 90.0))  # far off: a new place, taken at once
  assert t.pos[0] == 100.0 and t.bearing == 90.0


def test_pose_keeps_its_place_across_a_new_route():
  t = PoseTracker()
  a, b = Projection(51.5, 0.0), Projection(51.501, 0.001)
  t.update(0.0, 0.0, 0.0, (np.array([10.0, 20.0]), 45.0))
  where = a.lat_lon(t.pos)
  t.shift(a, b)
  assert np.allclose(b.lat_lon(t.pos), where)


def test_durations():
  assert format_duration(0.0) == "0 min" and format_duration(20.0) == "1 min" and format_duration(9900.0) == "2h 45m"


def test_trip_distances():
  assert format_trip_distance(4100.0, True) == "4.1 km" and format_trip_distance(1609344.0, True) == "1,609 km"
  assert format_trip_distance(1609344.0, False) == "1,000 mi" and format_trip_distance(4023.0, False) == "2.5 mi"


def test_route_starts_only_with_a_new_destination():
  # a new route sets Navigate on openpilot from its setting; a reroute, the route sent again or a moment without
  # guidance don't, so the driver's choice for the route stays
  r, end = RouteStarts(), (34.02, -118.30)
  assert not r.update(True, None, 0.0)  # the route's end not known yet
  assert r.update(True, end, 0.1)
  assert not r.update(True, end, 0.2)
  assert not r.update(True, (end[0] + 20 / M_PER_DEG, end[1]), 0.3)  # rerouted to a snapped end 20 m on
  assert not r.update(False, None, 1.0) and not r.update(True, end, 3.0)  # guidance lost for 2 s
  assert r.update(True, (end[0] + 500 / M_PER_DEG, end[1]), 4.0)  # a new destination
  assert not r.update(False, None, 5.0) and not r.update(False, None, 5.0 + ROUTE_GONE_S + 1)
  assert r.update(True, (end[0] + 500 / M_PER_DEG, end[1]), 40.0)  # the same place again, long after: a new route
  r.forget()  # ended by the driver
  assert r.update(True, (end[0] + 500 / M_PER_DEG, end[1]), 41.0)


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
