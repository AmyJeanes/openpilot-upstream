"""Drivers (gta5_drivers.py) and the brain's use of them: the engage key, the step, the hand-back and arrival."""
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.tools.sim.bridge.gta5 import gta5_drivers
from openpilot.tools.sim.bridge.gta5.gta5_drivers import INACTIVE, ROUTE_GAP, TOOK_OVER, MapDriver, OpenpilotDriver
from openpilot.tools.sim.bridge.gta5.gta5_brain import Brain
from openpilot.tools.sim.bridge.gta5.gta5_opio import NullIO

CONTROL = {"type": "control", "active": True, "curvature": 0.0, "accel": 0.5}


class FakeMapDrive:
  """Stands in for gta5_mapdrive.MapDrive: drives whatever route it's given, finishing when told."""
  def __init__(self, spec):
    self.spec, self.route, self.routes, self.finished, self.phase = spec, None, [], None, "drive"
    self.indicator, self.label = None, "none"

  def step(self, route, state, t, collisions=0):
    self.route = route
    self.routes.append(route)
    return CONTROL

  def take_events(self):
    return []


def state(**kw) -> dict:
  return {"t": 1.0, "pos": [0.0, 0.0, 0.0], "vEgo": 10.0, "heading": 0.0, **kw}


def test_map_driver(monkeypatch):
  """It needs a route to engage, drives on along its last route for ROUTE_GAP s without one, hands back at once on the
  driver's input, says when it arrived, and keeps the plugin's indicator as its map driver has it."""
  monkeypatch.setattr(gta5_drivers, "MapDrive", FakeMapDrive)
  sent = []
  p = MapDriver(sent.append)
  route = object()
  assert not p.engage(None) and not p.engaged and p.controls(state()) is None
  assert p.engage(route) and p.engaged and sent == [{"type": "ai", "on": 0, "indicator": "off"}]
  assert p.step(state(), route, 0.0) is None and p.controls(state()) == CONTROL
  t = 0.0
  while p.step(state(), None, t) is None:  # a new destination: the last route until the router has one
    assert p.md.routes[-1] is route
    t += 0.05
  assert abs(t - ROUTE_GAP) < 0.1 and p.step(state(), None, t) == "abort: no route"
  p.disengage("abort: no route")
  assert not p.engaged and p.controls(state()) is None and INACTIVE in sent[-2:]
  assert p.engage(route) and p.step(state(user={"brake": True}), route, 0.0) == TOOK_OVER
  assert p.engage(route)
  p.md.indicator = "left"
  p.step(state(), route, 0.0)
  assert sent[-1] == {"type": "ai", "indicator": "left"}
  p.step(state(indicator="left"), route, 1.0)
  assert sent[-1] == {"type": "ai", "indicator": "left"} and len([m for m in sent if m.get("indicator") == "left"]) == 1
  p.step(state(), route, 1.0 + gta5_drivers.INDICATOR_EVERY)  # the game dropped it: asked again
  assert len([m for m in sent if m.get("indicator") == "left"]) == 2
  p.md.finished = "arrived"
  assert p.step(state(), route, 5.0) == "arrived"
  waits = MapDriver(sent.append, wait_for_route=True)  # expert mode's: engaged before its route, waiting for it
  assert waits.engage(None) and waits.step(state(), None, 100.0) is None and waits.md.routes == [None]


def test_openpilot_driver():
  """The cruise buttons engage and disengage it, the driver's brake or steering cancels it (once per press), and it
  has controls only while engaged."""
  cmds = []
  io = SimpleNamespace(engaged=False, cruise=cmds.append, controls=lambda state: None)
  p = OpenpilotDriver(io)
  assert p.engage(None) and cmds == ["cruise_down"] and p.controls(state()) is None
  io.engaged = True
  p.step(state(user={"brake": True}), None, 0.0)
  p.step(state(user={"brake": True}), None, 0.01)
  assert cmds == ["cruise_down", "cruise_cancel"]
  p.disengage("key")
  assert cmds[-1] == "cruise_cancel"


def world(monkeypatch, driver: str):
  """A brain with just its drivers, expert mode off, openpilot's cruise buttons kept in cmds."""
  monkeypatch.setattr(gta5_drivers, "MapDrive", FakeMapDrive)
  monkeypatch.setattr(gta5_drivers, "DRIVER", driver)
  w = Brain.__new__(Brain)
  w.cmds, w.sent = [], []
  w.io = SimpleNamespace(engaged=False, drives=True, cruise=w.cmds.append, controls=lambda state: None)
  w.log, w._send = None, w.sent.append
  w.expert = SimpleNamespace(on=False, md_driving=False, update=lambda state, route, engaged: False)
  w.destination = SimpleNamespace(arrived=lambda: w.sent.append("arrived"))
  w._init_drivers()
  w.route = None
  return w


def test_engage_key_map(monkeypatch):
  """GTA5_DRIVER=map: the key engages the map driver along the route and disengages it; without a route it's openpilot's.
  The brain steps whoever drives and takes their controls; arriving hands back and ends the destination."""
  w = world(monkeypatch, "map")
  w._engage_key()  # no route
  assert w.cmds == ["cruise_down"] and not w.driver.engaged
  w.io.engaged = True
  w._engage_key()
  assert w.cmds == ["cruise_down", "cruise_cancel"]
  w.io.engaged = False
  w.route = np.zeros((2, 2))
  w._engage_key()
  assert w.driver.engaged and len(w.cmds) == 2
  assert w._step_drivers(state()) is w.driver and w.driver.controls(state()) == CONTROL
  w.driving = w.driver
  assert w._map_drive() is w.driver.md and w._driver_name() == "map"
  w._engage_key()
  assert not w.driver.engaged and w._step_drivers(state()) is None and w._map_drive() is None
  w._engage_key()
  w.driver.md.finished = "arrived"
  assert w._step_drivers(state()) is None and not w.driver.engaged and w.sent[-1] == "arrived"


def test_engage_key_openpilot(monkeypatch):
  """GTA5_DRIVER=openpilot (the default): the key is the cruise buttons, as it always was."""
  w = world(monkeypatch, "openpilot")
  assert w.driver is w.openpilot
  w._engage_key()
  assert w.cmds == ["cruise_down"]
  w.io.engaged = True
  assert w._step_drivers(state()) is w.openpilot
  w._engage_key()
  assert w.cmds == ["cruise_down", "cruise_cancel"]


def test_old_setting(monkeypatch):
  """GTA5_PILOT, the setting's old name, is refused rather than ignored."""
  monkeypatch.setenv("GTA5_PILOT", "map")
  monkeypatch.delenv("GTA5_DRIVER", raising=False)
  with pytest.raises(ValueError, match="GTA5_DRIVER"):
    gta5_drivers.driver_from_env(OpenpilotDriver(SimpleNamespace()), lambda m: None)


def test_no_openpilot(monkeypatch):
  """Without openpilot (NullIO, the host's --no-link) the engage key's driver is the map driver whatever GTA5_DRIVER
  says, and without a route the key does nothing rather than engage openpilot."""
  monkeypatch.setattr(gta5_drivers, "MapDrive", FakeMapDrive)
  monkeypatch.setattr(gta5_drivers, "DRIVER", "openpilot")
  w = Brain.__new__(Brain)
  w.sent, w.io, w.log, w.route = [], NullIO(), None, None
  w._send = w.sent.append
  w.expert = SimpleNamespace(on=False, md_driving=False, update=lambda state, route, engaged: False)
  w.destination = SimpleNamespace(arrived=lambda: None)
  w._init_drivers()
  assert isinstance(w.driver, MapDriver) and not w.openpilot.engaged
  w._engage_key()
  assert not w.driver.engaged and w.sent == []
  w.route = np.zeros((2, 2))
  w._engage_key()
  assert w.driver.engaged and w._step_drivers(state()) is w.driver and w.driver.controls(state()) == CONTROL
  w._engage_key()
  assert not w.driver.engaged


class FakeCruise:
  """Stands in for map/cruise.Cruise: its route whatever's set, and how often it was reset."""
  def __init__(self):
    self.route, self.resets, self.updates = None, 0, 0

  def update(self, pos, bearing, now, z=None):
    self.updates += 1
    return self.route

  def reset(self):
    self.resets += 1


def test_cruising_is_for_the_map_driver_alone(monkeypatch):
  """A brain with a router cruises only without openpilot (NullIO, the host's --no-link); with openpilot, in process
  or through the link, the engage key without a destination engages openpilot as before."""
  for io, cruises in ((NullIO(), True), (SimpleNamespace(drives=True), False)):
    w = Brain.__new__(Brain)
    w.io, w.navigator, w.lane_matcher = io, SimpleNamespace(router=SimpleNamespace(osm=None, roads=None)), None
    w._init_cruise()
    assert (w.cruise is not None) == cruises
    w.navigator = None
    w._init_cruise()
    assert w.cruise is None


def test_no_openpilot_cruises(monkeypatch):
  """Without openpilot, with no destination the engage key waits for the cruise route and engages the map driver on it
  once the car's on it. A destination's route takes over from it, and once that's gone the cruise route is back;
  stopped at a cruise route's end it hands back without ending a destination. A press while it waits cancels it, and
  with no cruise route in CRUISE_WAIT s it gives up."""
  from openpilot.tools.sim.bridge.gta5 import gta5_brain
  monkeypatch.setattr(gta5_drivers, "MapDrive", FakeMapDrive)
  clock = [100.0]
  monkeypatch.setattr(gta5_brain.time, "monotonic", lambda: clock[0])
  w = Brain.__new__(Brain)
  w.sent, w.io, w.log, w.route = [], NullIO(), None, None
  w._send = w.sent.append
  w.expert = SimpleNamespace(on=False, md_driving=False, update=lambda state, route, engaged: False)
  w.destination = SimpleNamespace(arrived=lambda: w.sent.append("arrived"))
  w._init_drivers()
  w.cruise = FakeCruise()
  pos = np.zeros(2)
  assert w._cruise_route(None, pos, 0.0, 0.0) is None and w.cruise.updates == 0  # not asked: no cruising
  w._engage_key()
  assert not w.driver.engaged and w.cruise_asked is not None
  assert w._cruise_route(None, pos, 0.0, 0.0) is None and not w.driver.engaged
  cruise = SimpleNamespace(cruise=True, on_road=lambda: False)
  w.cruise.route = cruise
  assert w._cruise_route(None, pos, 0.0, 0.0) is cruise and not w.driver.engaged  # the car isn't on it yet
  cruise.on_road = lambda: True
  w.route = w._cruise_route(None, pos, 0.0, 0.0)
  assert w.route is cruise and w.driver.engaged and w.cruise_asked is None
  assert w._step_drivers(state()) is w.driver and w.driver.md.route is cruise
  dest = SimpleNamespace(cruise=False)
  resets = w.cruise.resets
  assert w._cruise_route(dest, pos, 0.0, 0.0) is dest and w.cruise.resets == resets + 1  # a destination: its route
  w.route = dest
  w._step_drivers(state())
  assert w.driver.md.route is dest
  assert w._cruise_route(None, pos, 0.0, 0.0) is cruise  # the destination cleared: cruising again
  w.route = cruise
  w._step_drivers(state())
  w.driver.md.finished = "arrived"  # stopped at the end of the cruise route (a dead end)
  assert w._step_drivers(state()) is None and not w.driver.engaged and "arrived" not in w.sent
  w.route = w._cruise_route(None, pos, 0.0, 0.0)
  assert w.route is None
  w._engage_key()
  w._engage_key()  # cancelled while it waited
  assert w.cruise_asked is None and not w.driver.engaged
  w.cruise.route = None
  w._engage_key()
  clock[0] += gta5_brain.CRUISE_WAIT + 0.1
  w._cruise_route(None, pos, 0.0, 0.0)
  assert w.cruise_asked is None and not w.driver.engaged  # gave up
