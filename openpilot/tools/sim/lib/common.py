import math
import multiprocessing
import numpy as np

from abc import ABC, abstractmethod
from collections import namedtuple

W, H = 1928, 1208


vec3 = namedtuple("vec3", ["x", "y", "z"])

class GPSState:
  def __init__(self):
    self.latitude = 0
    self.longitude = 0
    self.altitude = 0

  def from_xy(self, xy):
    """Simulates a lat/lon from an xy coordinate on a plane, for simple simulation. TODO: proper global projection?"""
    BASE_LAT = 32.75308505188913
    BASE_LON = -117.2095393365393
    DEG_TO_METERS = 100000

    self.latitude = float(BASE_LAT + xy[0] / DEG_TO_METERS)
    self.longitude = float(BASE_LON + xy[1] / DEG_TO_METERS)
    self.altitude = 0


class IMUState:
  def __init__(self):
    self.accelerometer: vec3 = vec3(0,0,0)
    self.gyroscope: vec3 = vec3(0,0,0)
    self.bearing: float = 0


class SimulatorState:
  def __init__(self):
    self.valid = False
    self.is_engaged = False
    self.ignition = True

    self.velocity = vec3(0, 0, 0)
    self.bearing: float = 0
    self.gps = GPSState()
    self.imu = IMUState()

    self.steering_angle: float = 0

    self.user_gas: float = 0
    self.user_brake: float = 0
    self.user_torque: float = 0

    self.cruise_button = 0

    self.left_blinker = False
    self.right_blinker = False
    self.left_blindspot = False  # a vehicle in the blind spot that side, as the car's blind-spot monitor reports it
    self.right_blindspot = False

    self.speed_limit: float = 0  # m/s, 0 if unknown
    self.speed_limit_follow = False  # whether the set speed follows the speed limit as it changes, as from a map
    self.cruise_cap: float = 0  # m/s, a lower cruise speed the car asks for, as for a turn on its route; 0 for none
    self.cap_set_speed = True  # the simulated car shows cruise_cap as its set speed; off, openpilot reads the cap itself

  @property
  def speed(self):
    return math.sqrt(self.velocity.x ** 2 + self.velocity.y ** 2 + self.velocity.z ** 2)


class World(ABC):
  sets_blinkers = False  # read_sensors sets the blinkers every step, rather than the bridge clearing them for key presses
  sets_torque = False  # read_sensors sets the driver's steering torque every step, rather than the bridge's keys
  publishes_gps = False  # the world publishes GNSS itself, as the device's GNSS daemon would, rather than the sensors' fixed fix

  def __init__(self, dual_camera):
    self.dual_camera = dual_camera

    self.image_lock = multiprocessing.Semaphore(value=0)
    self.road_image = np.zeros((H, W, 3), dtype=np.uint8)
    self.wide_road_image = np.zeros((H, W, 3), dtype=np.uint8)

    self.exit_event = multiprocessing.Event()

  @abstractmethod
  def apply_controls(self, steer_sim, throttle_out, brake_out, /):
    pass

  @abstractmethod
  def tick(self):
    pass

  @abstractmethod
  def read_state(self):
    pass

  @abstractmethod
  def read_sensors(self, simulator_state: SimulatorState, /):
    pass

  @abstractmethod
  def read_cameras(self):
    pass

  def camera_yuv(self, wide: bool) -> bytes | None:
    """Optional: a ready NV12 frame, for worlds that produce one more cheaply than converting road_image."""
    return None

  @abstractmethod
  def close(self, reason: str):
    pass

  @abstractmethod
  def reset(self):
    pass
