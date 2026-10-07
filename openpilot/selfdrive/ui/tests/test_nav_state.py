import math

import numpy as np

from openpilot.cereal import messaging
from openpilot.selfdrive.ui.nav.nav_state import M_PER_DEG, NavState, PoseTracker, Projection, format_duration, instruction_guidance, wrap
from openpilot.selfdrive.ui.nav.text import lane_caption, maneuver_road, then_text


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


def test_guidance_and_captions():
  msg = messaging.new_message('navInstruction')
  ni = msg.navInstruction
  ni.valid, ni.maneuverType, ni.maneuverModifier, ni.maneuverPrimaryText, ni.maneuverDistance = True, "turn", "right", "Elm St", 180.0
  mans = ni.init('allManeuvers', 2)
  mans[0].distance, mans[0].type, mans[0].modifier = 180.0, "turn", "right"
  mans[1].distance, mans[1].type, mans[1].modifier = 580.0, "turn", "left"
  lanes = ni.init('lanes', 3)
  lanes[0].oncoming = True
  lanes[1].current, lanes[1].directions = True, ["straight"]
  lanes[2].active, lanes[2].directions, lanes[2].activeDirection = True, ["right"], "right"
  ni.showFull, ni.laneDistance = True, 60.0
  g = instruction_guidance(ni)
  assert g.show_lanes and maneuver_road(g) == "Elm St"
  assert lane_caption(g, True) == ("Keep right", "in lane by 60 m")
  assert then_text(g, True) == "Then left in 400 m"
  g.lane_open_distance = 300.0
  assert lane_caption(g, True) == ("Turn lane opens", "from 300 m · in by 60 m")


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
  assert format_duration(0.0) == "0 min" and format_duration(20.0) == "1 min" and format_duration(4000.0) == "1 h 7 min"


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
