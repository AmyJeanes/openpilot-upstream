"""Simulated GNSS from the game car's position, published as openpilot's GNSS daemon publishes it on a real device, so
that navd localizes from what a real car gives it.

GTA5_GPS picks the profile, default qcom3x; off leaves the simulator's own gpsLocationExternal at (0, 0), as before:
- qcom3x: a comma 3X. qcomgpsd's gpsLocation at 1 Hz with its fields (no horizontalAccuracy or flags, as the modem's
  position report gives neither), 300 ms late, with a slowly wandering bias (Gauss-Markov, 3 m, 60 s) and 1 m white
  noise on the position, and course noise that grows at low speed.
- ublox: a comma three's u-blox: ubloxd's gpsLocationExternal at 10 Hz, 100 ms late, 1.5 m / 30 s bias and 0.3 m noise.
- perfect: qcom3x's message and rate without noise or delay.
The noise is seeded (GTA5_GPS_SEED), so repeated drives get the same. The figures are starting points from general
GNSS behaviour, not fitted to a device's logs yet.
"""
import math
import os
import time
from collections import deque
from typing import NamedTuple

import numpy as np

from openpilot.cereal import log, messaging
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon


class Profile(NamedTuple):
  service: str
  rate: float  # Hz
  latency: float  # s from the position to its message
  bias: float  # m, the position bias's sigma, per axis
  bias_tau: float  # s, its correlation time
  white: float  # m, per axis, per fix
  course: float  # deg, course noise's sigma at speed
  course_slow: float  # deg m/s: and this much / max(v, 1 m/s) more
  speed: float  # m/s, speed noise's sigma
  altitude: float  # m, the altitude's sigma


PROFILES = {
  "qcom3x": Profile("gpsLocation", 1.0, 0.3, 3.0, 60.0, 1.0, 1.0, 10.0, 0.15, 7.0),
  "ublox": Profile("gpsLocationExternal", 10.0, 0.1, 1.5, 30.0, 0.3, 0.5, 5.0, 0.1, 5.0),
  "perfect": Profile("gpsLocation", 1.0, 0.0, 0.0, 60.0, 0.0, 0.0, 0.0, 0.0, 0.0),
}


class Gnss:
  """update() each step with the car's true position and velocity; a fix goes out at the profile's rate."""

  def __init__(self, profile: str = "qcom3x", seed: int | None = None, pm=None):
    self.profile = PROFILES[profile]
    self.name = profile
    self.pm = pm if pm is not None else messaging.PubMaster([self.profile.service])
    self.rng = np.random.default_rng(seed)
    p = self.profile
    self.bias = self.rng.normal(0.0, 1.0, 3) * np.array([p.bias, p.bias, p.altitude])  # east, north, up
    self.history: deque = deque()  # (t, wall, x, y, z, ve, vn, vu)
    self.next_fix = 0.0
    self.last_fix: float | None = None

  def update(self, now: float, x: float, y: float, z: float, ve: float, vn: float, vu: float = 0.0):
    p = self.profile
    self.history.append((now, time.time(), x, y, z, ve, vn, vu))  # noqa: TID251  (a fix carries its wall clock time)
    while len(self.history) > 1 and self.history[1][0] <= now - p.latency:
      self.history.popleft()
    if now < self.next_fix or self.history[0][0] > now - p.latency:
      return None
    self.next_fix = max(self.next_fix + 1.0 / p.rate, now)
    msg = self.fix(*self.history[0][1:], now)
    self.pm.send(p.service, msg)
    return msg

  def fix(self, wall: float, x: float, y: float, z: float, ve: float, vn: float, vu: float, now: float):
    p, rng = self.profile, self.rng
    dt = 1.0 / p.rate if self.last_fix is None else now - self.last_fix
    self.last_fix = now
    a = math.exp(-dt / p.bias_tau)
    self.bias = self.bias * a + rng.normal(0.0, 1.0, 3) * np.array([p.bias, p.bias, p.altitude]) * math.sqrt(1.0 - a * a)
    e, n, u = np.array([x, y, z]) + self.bias + rng.normal(0.0, 1.0, 3) * np.array([p.white, p.white, 0.0])
    lat, lon = to_lat_lon(float(e), float(n))
    v = math.hypot(ve, vn)
    course_sigma = p.course + p.course_slow / max(v, 1.0)
    bearing = (math.degrees(math.atan2(ve, vn)) + rng.normal(0.0, 1.0) * course_sigma) % 360.0
    speed = max(v + rng.normal(0.0, 1.0) * p.speed, 0.0)
    scale = speed / v if v > 0.01 else 0.0
    v_ned = [vn * scale, ve * scale, -vu]
    msg = messaging.new_message(p.service, valid=True)
    g = getattr(msg, p.service)
    g.latitude, g.longitude, g.altitude = lat, lon, float(u)
    g.speed, g.bearingDeg, g.vNED = speed, bearing, v_ned
    g.unixTimestampMillis = int(wall * 1000)
    g.bearingAccuracyDeg = course_sigma if p.course_slow else 0.1
    g.speedAccuracy = max(p.speed, 0.1)
    g.verticalAccuracy = max(p.altitude, 1.0)
    if p.service == "gpsLocation":  # as qcomgpsd
      g.source = log.GpsLocationData.SensorSource.qcomdiag
      g.hasFix = True
    else:  # as ubloxd
      g.source = log.GpsLocationData.SensorSource.ublox
      g.flags = 1
      g.hasFix = True
      g.horizontalAccuracy = max(math.hypot(p.bias, p.white), 1.0)
    return msg


def from_env() -> Gnss | None:
  name = os.getenv("GTA5_GPS", "qcom3x")
  if name == "off":
    return None
  seed = os.getenv("GTA5_GPS_SEED")
  return Gnss(name, int(seed) if seed else None)
