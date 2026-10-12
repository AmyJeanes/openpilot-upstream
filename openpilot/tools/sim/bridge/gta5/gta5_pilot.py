"""Pilots: who drives the car while it's engaged.

The engage key engages the bridge's pilot (GTA5_PILOT: openpilot, the default, or map) and disengages it, and the HUD's
engaged state, the driver's hand-back and arrival go through it, whichever it is. Each bridge step the world steps the
pilot driving with the game state and the route, then sends the plugin its control message.
- OpenpilotPilot: openpilot's own controls (carControl), engaged with the simulated car's cruise buttons.
- MapPilot: our map driver (gta5_mapdrive.py) along the route. Without a route it can't engage, and the engage key
  engages openpilot instead. Expert mode (gta5_expert.py) drives with a MapPilot of its own.
A pilot hands the car back at once on the driver's brake, gas or steering."""
import math
import os

import numpy as np
from opendbc.car.tesla.values import CarControllerParams as TeslaParams
from opendbc.car.vehicle_model import VehicleModel

from openpilot.tools.sim.bridge.gta5.gta5_mapdrive import USER_STEER, MapDriver

PILOT = os.getenv("GTA5_PILOT", "openpilot")
PILOTS = ("openpilot", "map")
INACTIVE = {"type": "control", "active": False}
TOOK_OVER = "abort: driver input"  # why a pilot hands back to the driver, as the map driver's own abort has it
ROUTE_GAP = 5.0  # s an engaged map driver drives on along its last route while there's none (routing a new destination)
INDICATOR_EVERY = 2.0  # s between asking the plugin again for the indicator its state doesn't show


def took_over(state: dict) -> bool:
  """The driver's brake, gas or steering."""
  user = state.get("user") or {}
  return abs(float(user.get("steer") or 0.0)) > USER_STEER or bool(user.get("brake")) or bool(user.get("gas"))


class Pilot:
  name = ""

  @property
  def engaged(self) -> bool:
    raise NotImplementedError

  def engage(self, route) -> bool:
    """Whether it engaged (or asked to)."""
    raise NotImplementedError

  def disengage(self, why: str) -> None:
    raise NotImplementedError

  def step(self, state: dict, route, now: float) -> str | None:
    """A bridge step: why it hands the car back now (the world then disengages it; "arrived" ends the destination too),
    else None."""
    return None

  def controls(self, state: dict) -> dict | None:
    """The plugin's control message for this step; None leaves the car to the player."""
    return None


class OpenpilotPilot(Pilot):
  """openpilot drives: the cruise buttons engage it, and carControl's curvature and acceleration go to the car."""
  name = "openpilot"

  def __init__(self, simulator_state, cruise, sm, tesla: bool):
    self.simulator_state = simulator_state
    self.cruise = cruise  # sends the simulated car a cruise button (common.control_cmd_gen's names)
    self.sm = sm
    self.tesla = tesla
    self.VM: VehicleModel | None = None
    self.steering = False  # whether the driver was steering
    self.braking = False

  @property
  def engaged(self) -> bool:
    return bool(self.simulator_state.is_engaged)

  def engage(self, route) -> bool:
    self.cruise("cruise_down")
    return True

  def disengage(self, why: str) -> None:
    self.cruise("cruise_cancel")

  def step(self, state: dict, route, now: float) -> str | None:
    # the brake and steering disengage (gas overrides), since the game's steering has no torque for openpilot to blend
    # with, and fighting it soon trips the excessive actuation lockout; the simulated car's cruise stays on through a
    # brake press otherwise, and openpilot re-engages on its release
    user = state.get("user") or {}
    steering, braking = abs(user.get("steer", 0)) > USER_STEER, bool(user.get("brake"))
    if (steering and not self.steering or braking and not self.braking) and self.engaged:
      self.cruise("cruise_cancel")
    self.steering, self.braking = steering, braking
    return None

  def update_model(self):
    if self.VM is None and self.sm.seen['carParams']:
      self.VM = VehicleModel(self.sm['carParams'])

  def controls(self, state: dict) -> dict | None:
    if not self.engaged:
      return None
    actuators = self.sm['carControl'].actuators
    # openpilot's curvature is right-positive (controlsd negates the vehicle model's); the plugin takes left-positive
    curvature = -actuators.curvature
    accel = float(actuators.accel)
    if self.tesla and self.VM is not None and self.sm.seen['carOutput']:
      # What a Model 3 executes: the angle after the car controller's rate and lateral jerk limits, through the car's
      # vehicle model, and the acceleration within its limits
      angle = self.sm['carOutput'].actuatorsOutput.steeringAngleDeg
      curvature = self.VM.calc_curvature(math.radians(angle), max(abs(state['vEgo']), 1.0), state.get("bank", 0.0))
      accel = float(np.clip(accel, TeslaParams.ACCEL_MIN, TeslaParams.ACCEL_MAX))
    return {"type": "control", "active": True, "curvature": curvature, "accel": accel}


class MapPilot(Pilot):
  """Our map driver drives the route: a new MapDriver (its plan made where the car is) each time it engages.

  wait_for_route (expert mode's trips): it engages before the route comes, holding the car until it does, and drives on
  whatever happens to the route; else it needs one to engage, and hands back after ROUTE_GAP s without one."""
  name = "map"

  def __init__(self, send, spec: dict | None = None, event=None, wait_for_route: bool = False):
    self.send = send
    self.spec = dict(spec or {})
    self.event = event  # event(what, **detail): the map driver's events, for a log
    self.wait_for_route = wait_for_route
    self.md: MapDriver | None = None  # the last drive's, kept after it ends for what it did
    self.on = False
    self.msg: dict | None = None
    self.indicator: str | None = None
    self.label = "none"
    self.next_indicator = 0.0
    self.no_route_since: float | None = None

  @property
  def engaged(self) -> bool:
    return self.on

  def engage(self, route) -> bool:
    if route is None and not self.wait_for_route:
      return False
    self.md = MapDriver(dict(self.spec))
    self.on, self.msg, self.indicator, self.label, self.no_route_since = True, None, None, "none", None
    self.send({"type": "ai", "on": 0, "indicator": "off"})  # our controls drive, not the game's AI
    print("gta5: the map driver drives", flush=True)
    return True

  def disengage(self, why: str) -> None:
    if not self.on:
      return
    self.on, self.msg = False, None
    self.send(INACTIVE)
    self.send({"type": "ai", "indicator": "off"})
    self.indicator, self.label = None, "none"
    print(f"gta5: the map driver hands back ({why})", flush=True)

  def step(self, state: dict, route, now: float) -> str | None:
    md = self.md
    if not self.on or md is None:
      return None
    if took_over(state):
      return TOOK_OVER
    if not self.wait_for_route:
      self.no_route_since = None if route is not None else (self.no_route_since or now)
      if route is None:
        if md.route is None or now - self.no_route_since >= ROUTE_GAP:
          return "abort: no route"
        route = md.route
    self.msg = md.step(route, state, now, state.get("collisions", 0))
    for e in md.take_events():
      what = e.pop("event")
      if what != "mapdrive" or e.get("phase") in ("abort", "end") or "splice" in e:
        detail = e.get("phase") or e.get("kind") or ""
        print(f"gta5: mapdrive {what} {detail}" + (f" ({e['why']})" if e.get("why") else "") +
              (f" at {e['pos']}" if e.get("pos") else ""), flush=True)
      if self.event is not None:
        self.event(what, **e)
    self._indicator(md.indicator, md.label, state, now)
    return md.finished

  def _indicator(self, side: str | None, label: str, state: dict, now: float):
    self.label = label
    if side != self.indicator or (state.get("indicator") != side and now >= self.next_indicator):
      self.indicator, self.next_indicator = side, now + INDICATOR_EVERY
      self.send({"type": "ai", "indicator": side or "off"})

  def controls(self, state: dict) -> dict | None:
    return self.msg if self.on else None

  @property
  def drives_route(self) -> bool:
    """Driving the route itself, not holding for one, nor stopping after an abort."""
    return self.on and self.md is not None and self.md.phase not in ("abort", "wait", "")


def pilot_from_env(openpilot: OpenpilotPilot, send, event=None) -> Pilot:
  if PILOT not in PILOTS:
    raise ValueError(f"GTA5_PILOT={PILOT}: one of {', '.join(PILOTS)}")
  if PILOT == "map":
    print("gta5: the engage key drives with the map driver (GTA5_PILOT=map)", flush=True)
    return MapPilot(send, event=event)
  return openpilot
