"""The bridge's side of the nav model: GTA5_NAV_MODEL's allManeuvers in gta5_nav_msgs, and gta5_mapsd's FrameSM. No
pytest needed: `python test_bridge_nav.py` runs them all."""
import math
import time
from types import SimpleNamespace

import numpy as np

from openpilot.selfdrive.navigation.helpers import latest_gps_source
from openpilot.tools.sim.bridge.gta5 import gta5_nav_msgs as nm
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon
from openpilot.tools.sim.bridge.gta5.navmodel.frame import lat_lon
from openpilot.tools.sim.bridge.gta5.navmodel.gta5_mapsd import FrameSM
from openpilot.tools.sim.bridge.gta5.navmodel.nav_input import ModelNav, navigation
from openpilot.tools.sim.bridge.gta5.navmodel.test_nav_input import point_at, route


class Out:
  def __init__(self):
    self.instructions, self.routes = [], []

  def nav_instruction(self, content):
    self.instructions.append(content)

  def nav_route(self, content):
    self.routes.append(content)


def test_nav_model_maneuvers_in_the_instruction():
  r = route("exits")
  r.at = 2000.0
  pose = (point_at(r, r.at), 0.0)
  old = nm.NAV_MODEL
  try:
    nm.NAV_MODEL = False
    ui = nm.NavMessages(Out()).instruction(r, 20.0, pose=pose)
    nm.NAV_MODEL = True
    msgs = nm.NavMessages(Out())
    for s in np.arange(0.0, r.at, 10.0):  # the tracker follows the car there, as it would at 10 Hz
      msgs.instruction(r, 20.0, pose=(point_at(r, s), 0.0))
    ni = msgs.instruction(r, 20.0, pose=pose)
  finally:
    nm.NAV_MODEL = old
  assert len(ui["allManeuvers"]) <= 5 and len(ni["allManeuvers"]) > 5  # all of them, not the UI's next five
  assert ni["maneuverType"] == ui["maneuverType"] and ni["maneuverDistance"] == ui["maneuverDistance"]  # the banner as before
  nav = ModelNav(r)
  for s in np.arange(0.0, r.at, 10.0):
    nav.all_maneuvers_at(point_at(r, s))
  assert ni["allManeuvers"] == nav.all_maneuvers_at(pose[0])
  v = navigation(ni["allManeuvers"])
  assert v[:46].sum() == 1 and 0 <= v[46] < 30
  assert msgs.model_nav[0] is r and msgs._model_nav(r) is msgs.model_nav[1]  # one per route


class SM:
  """A SubMaster as FrameSM reads it, with a route and a fix from gpsLocation."""
  def __init__(self, route_lat_lon, fix):
    self.msgs = {"navRoute": SimpleNamespace(coordinates=[SimpleNamespace(latitude=a, longitude=o) for a, o in route_lat_lon]),
                 "gpsLocation": SimpleNamespace(hasFix=True, speed=12.0, bearingDeg=90.0, latitude=fix[0], longitude=fix[1]),
                 "gpsLocationExternal": SimpleNamespace(hasFix=False, speed=0.0, bearingDeg=0.0, latitude=0.0, longitude=0.0)}
    now = time.monotonic()
    self.updated = {"navRoute": True}
    self.valid = dict.fromkeys(self.msgs, True)
    self.recv_time = dict.fromkeys(self.msgs, now)
    self.logMonoTime = {"navRoute": 1, "gpsLocation": 2, "gpsLocationExternal": 1}

  def __getitem__(self, name):
    return self.msgs[name]


def test_frame_sm_moves_route_and_fix():
  pts = [(100.0, 200.0), (150.0, 260.0), (3000.0, 7000.0)]
  bridge = [to_lat_lon(*p) for p in pts]
  sm = SM(bridge, to_lat_lon(120.0, 230.0))
  same = FrameSM(sm, 0.0)
  assert same["navRoute"] is sm["navRoute"] and same["gpsLocation"] is sm["gpsLocation"]
  view = FrameSM(sm, 34.0)
  for c, p in zip(view["navRoute"].coordinates, pts, strict=True):
    lat, lon = lat_lon(*p, 34.0)
    assert math.isclose(c.latitude, lat, abs_tol=1e-9) and math.isclose(c.longitude, lon, abs_tol=1e-9)
  assert view["navRoute"] is view["navRoute"]  # converted once per message
  fix = view["gpsLocation"]
  lat, lon = lat_lon(120.0, 230.0, 34.0)
  assert math.isclose(fix.latitude, lat, abs_tol=1e-9) and fix.speed == 12.0 and fix.hasFix
  assert latest_gps_source(view) == "gpsLocation"  # comma's own pick of the fix, through the view


if __name__ == "__main__":
  for name, f in list(globals().items()):
    if name.startswith("test_") and callable(f):
      f()
      print("ok", name)
