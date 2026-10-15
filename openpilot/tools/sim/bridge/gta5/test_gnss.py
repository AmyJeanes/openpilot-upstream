"""The simulated GNSS (gta5_gnss.py)."""
import numpy as np
from cereal import log

from openpilot.tools.sim.bridge.gta5.gta5_gnss import Gnss
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game


class Clock:
  def __init__(self):
    self.t = 100.0

  def monotonic(self):
    return self.t

  def time(self):
    return 1.7e9 + self.t


class PubMaster:
  def __init__(self):
    self.sent = []

  def send(self, service, msg):
    self.sent.append((service, msg))


def test_gnss_as_a_3x():
  import openpilot.tools.sim.bridge.gta5.gta5_gnss as gnss_mod
  clock = Clock()
  gnss_mod.time = clock
  pm = PubMaster()
  g = Gnss("qcom3x", seed=1, pm=pm)
  errors = []
  for k in range(1200):  # 60 s at 20 Hz, heading east at 10 m/s
    t = k * 0.05
    clock.t = 100.0 + t
    g.update(clock.t, 10.0 * t, 50.0, 30.0, 10.0, 0.0)
  assert all(s == "gpsLocation" for s, _ in pm.sent) and 59 <= len(pm.sent) <= 61  # 1 Hz
  for _, msg in pm.sent:
    fix = msg.gpsLocation
    assert fix.source == log.GpsLocationData.SensorSource.qcomdiag and fix.hasFix and fix.horizontalAccuracy == 0.0
    x, y = to_game(fix.latitude, fix.longitude)
    true_x = 10.0 * ((fix.unixTimestampMillis / 1000.0 - 1.7e9) - 100.0)  # the position when it was taken, 300 ms back
    errors.append(np.hypot(x - true_x, y - 50.0))
  assert 0.3 < np.mean(errors) < 10.0  # noisy, but metres, not tens
  pm_still = PubMaster()
  g = Gnss("qcom3x", seed=2, pm=pm_still)
  for k in range(2400):  # 120 s standing still: the fixes drift, they don't hop about
    clock.t = 300.0 + k * 0.05
    g.update(clock.t, 0.0, 0.0, 30.0, 0.0, 0.0)
  still = np.array([to_game(m.gpsLocation.latitude, m.gpsLocation.longitude) for _, m in pm_still.sent])
  assert np.mean(np.hypot(*np.diff(still, axis=0).T)) < 0.3 and np.max(np.hypot(*still.T)) > 0.5
  pm2 = PubMaster()
  g = Gnss("perfect", pm=pm2)
  clock.t = 200.0
  g.update(clock.t, 1.0, 2.0, 3.0, 0.0, 5.0)
  fix = pm2.sent[0][1].gpsLocation
  assert np.allclose(to_game(fix.latitude, fix.longitude), (1.0, 2.0)) and abs(fix.bearingDeg) < 1e-6 and fix.speed == 5.0
