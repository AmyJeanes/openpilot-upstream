"""The plugin's sound and HUD text as the car engages and hands back (gta5_events.py), through the brain's drivers."""
import time
from types import SimpleNamespace

import numpy as np

from openpilot.tools.sim.bridge.gta5 import gta5_drivers, gta5_events
from openpilot.tools.sim.bridge.gta5.gta5_brain import Brain
from openpilot.tools.sim.bridge.gta5.gta5_drivers import TOOK_OVER, took_over
from openpilot.tools.sim.bridge.gta5.gta5_events import REFUSE_WAIT, SECS, event, prompt
from openpilot.tools.sim.bridge.gta5.gta5_opio import NullIO
from openpilot.tools.sim.bridge.gta5.test_drivers import FakeMapDrive, state


def test_event():
  assert event("engage", "map") == ("engage", "Map driver engaged")
  assert event("engage", "map", cruise=True) == ("engage", "Cruising")
  assert event("engage", "openpilot") == ("engage", "openpilot engaged")
  assert event("handback", "map", TOOK_OVER) == ("disengage", "Disengaged")
  assert event("handback", "map", "abort: the driver pressed the engage key") == ("disengage", "Disengaged")
  assert event("handback", "openpilot", "driver input") == ("disengage", "Disengaged")
  assert event("handback", "map", "arrived") == ("disengage", "Arrived")
  assert event("handback", "openpilot", "") == ("warning", "openpilot disengaged")
  assert event("refuse", why="no route") == ("refuse", "Can't engage: no route")
  gave_up = {
    "abort: collision": "collision",
    "abort: 2.3 m off the path": "off the path",
    "abort: heading 50 deg off the path": "off the path",
    "abort: off the road (+3.1 m right of its line)": "off the road",
    "abort: lateral acceleration 5.2 m/s^2": "too fast for a bend",
    "abort: in the oncoming lanes": "in the oncoming lanes",
    "abort: no progress in 20 s": "stuck",
    "abort: held 30 s by a vehicle standing across our way (deadlock)": "deadlock",
    "abort: held 60 s by a vehicle in the way": "blocked by a car",
    "abort: held 20 s by a parked vehicle in the way": "blocked by a parked car",
    "abort: held 60 s by a pedestrian in the way": "blocked by a pedestrian",
    "abort: plan: no lanes on the route": "no lanes on the route",
    "abort: plan: the car is +2.1 m from its lane line": "too far from its lane",
    "abort: plan: lane 3 120 m along, where the road has 2": "no plan",
    "abort: no route": "no route",
    "the road ends": "the road ends",
  }
  for why, short in gave_up.items():
    assert event("handback", "map", why) == ("warning", f"Map driver gave up: {short}"), why

  assert prompt("turn", "left", "Row 1") == ("", "Next left: Row 1") and prompt("turn", "right") == ("", "Next right")
  assert prompt("keep", "left") == ("", "Keeping left") and prompt("lane", "right") == ("", "Changing lane right")
  assert prompt("late", "right") == ("refuse", "Too late for that right") and prompt("none", "left") == ("refuse", "No left here")


def brain(monkeypatch, driver: str, io=None) -> Brain:
  """A brain with just its drivers and its announcer, expert mode off; plugin commands in w.sent."""
  monkeypatch.setattr(gta5_drivers, "MapDrive", FakeMapDrive)
  monkeypatch.setattr(gta5_drivers, "DRIVER", driver)
  w = Brain.__new__(Brain)
  w.sent, w.cmds = [], []
  w.io = io or SimpleNamespace(engaged=False, drives=True, cruise=w.cmds.append, controls=lambda state: None)
  w.log, w._send, w.route = None, w.sent.append, None
  w.expert = SimpleNamespace(on=False, md_driving=False, update=lambda state, route, engaged: False)
  w.destination = SimpleNamespace(arrived=lambda: None)
  w._init_drivers()
  return w


def step(w: Brain, s: dict | None = None, now: float | None = None):
  """The drivers' part of Brain.step."""
  s = s or state()
  w.driving = w._step_drivers(s)
  w.announcer.step(time.monotonic() if now is None else now, w.driver, w.openpilot, took_over(s), w.expert.on, w.route)


def events(w: Brain) -> list[tuple]:
  out = [(m.get("sound"), m.get("text")) for m in w.sent if isinstance(m, dict) and m.get("type") == "event"]
  w.sent.clear()
  return out


def test_map_driver(monkeypatch):
  """Engaging, the player's hand-back, an abort (once, as it starts to stop; the hand-back after it is silent), arrival
  and a cruise route's engage."""
  w = brain(monkeypatch, "map", NullIO())
  w._engage_key()
  assert events(w) == [("refuse", "Can't engage: no route")]
  w.route = np.zeros((2, 2))
  w._engage_key()
  step(w)
  assert events(w) == [("engage", "Map driver engaged")]
  step(w, state(user={"brake": True}))
  assert not w.driver.engaged and events(w) == [("disengage", "Disengaged")]

  w._engage_key()
  step(w)
  events(w)
  w._engage_key()  # the key hands back
  step(w)
  assert events(w) == [("disengage", "Disengaged")]

  w._engage_key()
  step(w)
  events(w)
  md = w.driver.md
  md.phase, md.why = "abort", "collision"
  step(w)
  step(w)
  assert w.driver.engaged and events(w) == [("warning", "Map driver gave up: collision")]
  md.finished = "abort: collision"
  step(w)
  assert not w.driver.engaged and events(w) == []

  w._engage_key()
  step(w)
  events(w)
  w.driver.md.finished = "arrived"
  step(w)
  assert events(w) == [("disengage", "Arrived")]

  w.route = SimpleNamespace(cruise=True)
  w._engage_key()
  step(w)
  assert events(w) == [("engage", "Cruising")]
  monkeypatch.setattr(gta5_drivers, "ROUTE_GAP", 0.0)
  w.route = None  # no route for too long: gave up
  step(w)
  assert not w.driver.engaged and events(w) == [("warning", "Map driver gave up: no route")]


def test_arrow_keys(monkeypatch):
  """While the map driver drives, an arrow key cruising asks the cruise for its turn, and with none that way ahead (or
  with a destination) the map drive for a lane change; what they do, and what becomes of them later, on the HUD. Not
  while it's disengaged, nor in expert mode."""
  w = brain(monkeypatch, "map", NullIO())
  w.presses, w._track = {}, lambda s: None
  turns, notes = [("turn", "Row 1"), ("none", ""), ("none", "")], []
  asked = []
  w.cruise = SimpleNamespace(prompt=lambda side, pos, bearing, z, v: (asked.append(side), turns.pop(0))[1],
                             take_notes=lambda: [notes.pop() for _ in list(notes)], take_log=list)
  presses = {"leftPresses": 0, "rightPresses": 0}

  def press(key):
    presses[key] += 1
    w._update_buttons(state(**presses))

  w._update_buttons(state(**presses))  # the counts so far
  press("leftPresses")  # disengaged: the plugin's indicator
  assert asked == [] and events(w) == []
  w.route = SimpleNamespace(cruise=True)
  w._engage_key()
  step(w)
  events(w)
  md = w.driver.md
  press("leftPresses")
  assert asked == ["left"] and md.asks == [] and events(w) == [(None, "Next left: Row 1")]
  press("rightPresses")  # no right turn ahead: a lane change, made a step later
  assert asked == ["left", "right"] and md.asks == ["right"] and events(w) == []
  md.prompts.append(("lane", "right"))
  w._update_buttons(state(**presses))
  assert events(w) == [(None, "Changing lane right")]
  md.answer = "no_lane"
  press("rightPresses")
  assert events(w) == [("refuse", "No right here")]
  notes.append(("forbidden", "left", "Row 1"))  # a turn that couldn't be routed
  w._update_buttons(state(**presses))
  assert events(w) == [("refuse", "Can't turn left there")]
  w.route = SimpleNamespace(cruise=False)  # a destination: lane changes alone
  md.answer = "ok"
  press("leftPresses")
  assert len(asked) == 3 and md.asks == ["right", "right", "left"]
  w.expert.on = True
  press("leftPresses")
  assert md.asks == ["right", "right", "left"]


def test_openpilot(monkeypatch):
  """openpilot engaging, handed back by the player and disengaging by itself; the key asking it in vain is a refusal."""
  w = brain(monkeypatch, "openpilot")
  w._engage_key()
  w.io.engaged = True
  step(w)
  assert events(w) == [("engage", "openpilot engaged")]
  w._engage_key()
  w.io.engaged = False
  step(w)
  assert events(w) == [("disengage", "Disengaged")]
  w.io.engaged = True
  step(w)
  events(w)
  w.io.engaged = False
  step(w, state(), time.monotonic() + 10)  # long after the key, no input
  assert events(w) == [("warning", "openpilot disengaged")]
  w._engage_key()
  step(w)
  assert events(w) == []
  step(w, state(), time.monotonic() + REFUSE_WAIT + 0.5)
  assert events(w) == [("refuse", "Can't engage: openpilot didn't engage")]
  step(w, state(), time.monotonic() + 2 * REFUSE_WAIT + 1)
  assert events(w) == []


def test_combined(monkeypatch):
  """The map driver with openpilot: no route engages openpilot instead, and nothing is refused."""
  w = brain(monkeypatch, "map")
  w._engage_key()
  assert w.cmds == ["cruise_down"] and events(w) == []
  w.io.engaged = True
  step(w)
  assert events(w) == [("engage", "openpilot engaged")]


def test_quiet(monkeypatch):
  """GTA5_SOUNDS=0 leaves out the sound, GTA5_HUD_EVENTS=0 the text; expert mode's drives are silent."""
  w = brain(monkeypatch, "map", NullIO())
  w.route = np.zeros((2, 2))
  monkeypatch.setattr(gta5_events, "SOUNDS", False)
  w._engage_key()
  step(w)
  assert [m for m in w.sent if m.get("type") == "event"] == [{"type": "event", "text": "Map driver engaged", "secs": SECS}]
  w.sent.clear()
  monkeypatch.setattr(gta5_events, "SOUNDS", True)
  monkeypatch.setattr(gta5_events, "HUD_EVENTS", False)
  step(w, state(user={"gas": True}))
  assert [m for m in w.sent if m.get("type") == "event"] == [{"type": "event", "sound": "disengage"}]
  w.sent.clear()
  monkeypatch.setattr(gta5_events, "SOUNDS", False)
  w._engage_key()
  step(w)
  assert events(w) == []
  monkeypatch.setattr(gta5_events, "SOUNDS", True)
  monkeypatch.setattr(gta5_events, "HUD_EVENTS", True)
  w.expert.on = True
  step(w, state(user={"brake": True}))
  w._engage_key()
  assert not w.driver.engaged and events(w) == []


def test_cruise(monkeypatch):
  """The map driver alone with no destination: the engage key's wait for the road ahead, "Cruising" once on it, the
  road's end, a press cancelling the wait, and giving up on it after CRUISE_WAIT s."""
  from openpilot.tools.sim.bridge.gta5 import gta5_brain
  from openpilot.tools.sim.bridge.gta5.test_drivers import FakeCruise
  clock = [100.0]
  monkeypatch.setattr(gta5_brain.time, "monotonic", lambda: clock[0])
  w = brain(monkeypatch, "map", NullIO())
  w.cruise = FakeCruise()
  pos = np.zeros(2)
  w._engage_key()
  assert events(w) == [(None, "Looking for the road ahead")]
  w.cruise.route = SimpleNamespace(cruise=True, on_road=lambda: True)
  w.route = w._cruise_route(None, pos, 0.0, 0.0)
  step(w, now=clock[0])
  assert w.driver.engaged and events(w) == [("engage", "Cruising")]
  w.driver.md.finished = "arrived"  # stopped at the cruise route's end, a dead end
  step(w, now=clock[0] + 10)
  assert not w.driver.engaged and events(w) == [("warning", "Map driver gave up: the road ends")]
  w.cruise.route = None
  w.route = w._cruise_route(None, pos, 0.0, 0.0)
  w._engage_key()
  w._engage_key()
  assert events(w) == [(None, "Looking for the road ahead"), ("disengage", "Not cruising")]
  w._engage_key()
  clock[0] += gta5_brain.CRUISE_WAIT + 0.1
  w._cruise_route(None, pos, 0.0, 0.0)
  assert events(w) == [(None, "Looking for the road ahead"), ("refuse", "Can't engage: no road ahead")]
