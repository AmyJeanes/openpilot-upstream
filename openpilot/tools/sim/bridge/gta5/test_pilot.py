"""Pilots (gta5_pilot.py) and the world's use of them: the engage key, the step, the hand-back and arrival."""
from types import SimpleNamespace

import numpy as np

from openpilot.tools.sim.bridge.gta5 import gta5_pilot
from openpilot.tools.sim.bridge.gta5 import gta5_world as world_mod
from openpilot.tools.sim.bridge.gta5.gta5_pilot import INACTIVE, ROUTE_GAP, TOOK_OVER, MapPilot, OpenpilotPilot
from openpilot.tools.sim.bridge.gta5.gta5_world import GTA5World

CONTROL = {"type": "control", "active": True, "curvature": 0.0, "accel": 0.5}


class FakeMapDriver:
  """Stands in for gta5_mapdrive.MapDriver: drives whatever route it's given, finishing when told."""
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


def test_map_pilot(monkeypatch):
  """It needs a route to engage, drives on along its last route for ROUTE_GAP s without one, hands back at once on the
  driver's input, says when it arrived, and keeps the plugin's indicator as its map driver has it."""
  monkeypatch.setattr(gta5_pilot, "MapDriver", FakeMapDriver)
  sent = []
  p = MapPilot(sent.append)
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
  p.step(state(), route, 1.0 + gta5_pilot.INDICATOR_EVERY)  # the game dropped it: asked again
  assert len([m for m in sent if m.get("indicator") == "left"]) == 2
  p.md.finished = "arrived"
  assert p.step(state(), route, 5.0) == "arrived"
  waits = MapPilot(sent.append, wait_for_route=True)  # expert mode's: engaged before its route, waiting for it
  assert waits.engage(None) and waits.step(state(), None, 100.0) is None and waits.md.routes == [None]


def test_openpilot_pilot():
  """The cruise buttons engage and disengage it, the driver's brake or steering cancels it (once per press), and it
  has controls only while engaged."""
  cmds = []
  sim = SimpleNamespace(is_engaged=False)
  p = OpenpilotPilot(sim, cmds.append, sm=None, tesla=False)
  assert p.engage(None) and cmds == ["cruise_down"] and p.controls(state()) is None
  sim.is_engaged = True
  p.step(state(user={"brake": True}), None, 0.0)
  p.step(state(user={"brake": True}), None, 0.01)
  assert cmds == ["cruise_down", "cruise_cancel"]
  p.disengage("key")
  assert cmds[-1] == "cruise_cancel"


def world(monkeypatch, pilot: str):
  """A GTA5World with just its pilots, expert mode off."""
  monkeypatch.setattr(gta5_pilot, "MapDriver", FakeMapDriver)
  monkeypatch.setattr(gta5_pilot, "PILOT", pilot)
  w = GTA5World.__new__(GTA5World)
  w.simulator_state = SimpleNamespace(is_engaged=False)
  w.cmds, w.sent = [], []
  w.q = SimpleNamespace(put=lambda c: w.cmds.append(c))
  w.sm, w.tesla, w.log, w._send = None, False, None, w.sent.append
  w.expert = SimpleNamespace(on=False, md_driving=False, update=lambda state, route, engaged: False)
  w.destination = SimpleNamespace(arrived=lambda: w.sent.append("arrived"))
  monkeypatch.setattr(world_mod, "control_cmd_gen", lambda cmd: cmd)
  w._init_pilots()
  w.route = None
  return w


def test_engage_key_map(monkeypatch):
  """GTA5_PILOT=map: the key engages the map driver along the route and disengages it; without a route it's openpilot's.
  The world steps whoever drives and takes their controls; arriving hands back and ends the destination."""
  w = world(monkeypatch, "map")
  w._engage_key()  # no route
  assert w.cmds == ["cruise_down"] and not w.pilot.engaged
  w.simulator_state.is_engaged = True
  w._engage_key()
  assert w.cmds == ["cruise_down", "cruise_cancel"]
  w.simulator_state.is_engaged = False
  w.route = np.zeros((2, 2))
  w._engage_key()
  assert w.pilot.engaged and len(w.cmds) == 2
  assert w._step_pilots(state()) is w.pilot and w.pilot.controls(state()) == CONTROL
  w.driving = w.pilot
  assert w._map_driver() is w.pilot.md and w._driver_name() == "map"
  w._engage_key()
  assert not w.pilot.engaged and w._step_pilots(state()) is None and w._map_driver() is None
  w._engage_key()
  w.pilot.md.finished = "arrived"
  assert w._step_pilots(state()) is None and not w.pilot.engaged and w.sent[-1] == "arrived"


def test_engage_key_openpilot(monkeypatch):
  """GTA5_PILOT=openpilot (the default): the key is the cruise buttons, as it always was."""
  w = world(monkeypatch, "openpilot")
  assert w.pilot is w.openpilot
  w._engage_key()
  assert w.cmds == ["cruise_down"]
  w.simulator_state.is_engaged = True
  assert w._step_pilots(state()) is w.openpilot
  w._engage_key()
  assert w.cmds == ["cruise_down", "cruise_cancel"]
