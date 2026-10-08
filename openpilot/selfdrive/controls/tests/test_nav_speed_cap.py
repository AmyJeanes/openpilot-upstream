"""Navigation's speed cap (navSpeed) in the longitudinal planner: the cruise speed is the lower of it and the driver's set
speed, slowing follows the cap's ramp within the cruise limits, and the set speed returns once the cap is released."""
import math

import numpy as np

from opendbc.car.structs import car
from openpilot.cereal import messaging
from openpilot.common.constants import CV
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_planner import A_CRUISE_MIN, LongitudinalPlanner, get_nav_speed_cap
from openpilot.selfdrive.navd.planner import slow_for

SERVICES = ('carControl', 'carState', 'controlsState', 'vehicleParameters', 'radarState', 'modelV2', 'selfdriveState', 'navSpeed')
ACCEL_TAU = 0.3  # s, the car's response to the planned accel


class FakeSM:
  """The planner's SubMaster: one message each, read as builders, with their log times."""
  def __init__(self, nav: bool = True):
    services = SERVICES if nav else SERVICES[:-1]
    self.data = {s: getattr(messaging.new_message(s), s) for s in services}
    self.seen = dict.fromkeys(services, True)
    self.seen['navSpeed'] = False
    self.logMonoTime = dict.fromkeys(services, 0)
    self.data['controlsState'].longControlState = 'pid'
    self.data['selfdriveState'].enabled = True
    self.data['carState'].vCruise = 255.0

  def __getitem__(self, s):
    return self.data[s]


def car_params():
  CP = car.CarParams.new_message()
  CP.openpilotLongitudinalControl = True
  CP.longitudinalActuatorDelay = 0.5
  CP.steerRatio, CP.wheelbase = 12.0, 2.875
  return CP


class Drive:
  """The planner driving a simple car: the planned accel reaches it through a first-order lag."""
  def __init__(self, v: float, set_speed: float, nav: bool = True):
    self.sm = FakeSM(nav)
    self.planner = LongitudinalPlanner(car_params(), init_v=v)
    self.v, self.a, self.s, self.t = v, 0.0, 0.0, 0.0
    self.set_speed(set_speed)

  def set_speed(self, v: float):
    self.sm['carState'].vCruise = float(v * CV.MS_TO_KPH)

  def cap(self, cap: float | None, age: float = 0.0):
    """navSpeed's cap (m/s, 0 for none), sent age s before this step; None: none ever sent."""
    if cap is None:
      self.sm.seen['navSpeed'] = False
      return
    self.sm.seen['navSpeed'] = True
    self.sm["navSpeed"].speedCap = float(cap)
    self.sm.logMonoTime['navSpeed'] = int((self.t - age) * 1e9)

  def step(self) -> float:
    cs = self.sm['carState']
    cs.vEgo, cs.aEgo, cs.standstill = float(self.v), float(self.a), bool(self.v < 0.01)
    self.sm.logMonoTime['modelV2'] = int(self.t * 1e9)
    self.planner.update(self.sm)
    self.a += (self.planner.output_a_target - self.a) * DT_MDL / ACCEL_TAU
    self.v = max(self.v + self.a * DT_MDL, 0.0)
    self.s += self.v * DT_MDL
    self.t += DT_MDL
    return self.v


def test_cap_reading():
  sm = FakeSM()
  assert get_nav_speed_cap(sm) == math.inf  # nothing sent yet
  sm.seen['navSpeed'] = True
  sm['navSpeed'].speedCap = 7.0
  assert get_nav_speed_cap(sm) == 7.0
  sm['navSpeed'].speedCap = 0.0
  assert get_nav_speed_cap(sm) == math.inf  # 0: no cap
  sm['navSpeed'].speedCap = 7.0
  sm.logMonoTime['modelV2'] = int(1.5e9)
  assert get_nav_speed_cap(sm) == math.inf  # stale: navigation stopped sending
  assert get_nav_speed_cap(FakeSM(nav=False)) == math.inf  # a planner without the service


def test_never_above_set_speed():
  d = Drive(15.0, 15.0)
  speeds = []
  for _ in range(int(20 / DT_MDL)):
    d.cap(25.0)
    speeds.append(d.step())
  assert max(speeds) < 15.05
  assert abs(speeds[-1] - 15.0) < 0.1


def test_cap_lowers_speed_within_cruise_limits():
  d = Drive(15.0, 15.0)
  speeds, accels = [], []
  for _ in range(int(20 / DT_MDL)):
    d.cap(8.0)
    speeds.append(d.step())
    accels.append(d.planner.output_a_target)
  assert abs(speeds[-1] - 8.0) < 0.2
  assert min(accels) >= A_CRUISE_MIN - 1e-3
  assert np.max(np.abs(np.diff(accels))) / DT_MDL < 2.0  # jerk-limited, no step


def test_turn_approach_and_release():
  """navd's ramp for a turn 250 m ahead (4.5 m/s, 0.6 m/s^2 with a second's lag): smooth slowing to the turn's speed,
  never faster than the set speed, then back up to it once nav releases the cap."""
  turn_at, turn_speed = 250.0, 4.5
  d = Drive(15.0, 15.0)
  speeds, accels, at_turn = [], [], None
  while d.t < 60.0:
    left = turn_at - d.s
    d.cap(slow_for(turn_speed, left, d.v) if left > 0 else 0.0)
    if at_turn is None and left <= 0:
      at_turn = d.v
    speeds.append(d.step())
    accels.append(d.planner.output_a_target)
  assert at_turn is not None and at_turn < turn_speed + 0.5
  assert max(speeds) < 15.05
  before = np.array(speeds[:int(np.argmin(speeds))])
  assert np.all(np.diff(before) < 1e-3)  # slows without surging between
  assert min(accels) >= A_CRUISE_MIN - 1e-3
  assert speeds[-1] > 14.5  # released after the turn


def test_set_speed_below_cap_wins():
  d = Drive(10.0, 10.0)
  for _ in range(int(10 / DT_MDL)):
    d.cap(12.0)
    d.step()
  assert abs(d.v - 10.0) < 0.1
  d.set_speed(7.0)  # the driver lowers the set speed: below the cap, it rules
  for _ in range(int(15 / DT_MDL)):
    d.cap(12.0)
    d.step()
  assert abs(d.v - 7.0) < 0.2


def test_stale_cap_lapses():
  d = Drive(12.0, 12.0)
  for _ in range(int(10 / DT_MDL)):
    d.cap(6.0, age=2.0)  # last sent 2 s ago: navigation stopped
    d.step()
  assert abs(d.v - 12.0) < 0.1
