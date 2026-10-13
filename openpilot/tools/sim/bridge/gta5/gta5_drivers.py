"""Drivers: who drives the car while it's engaged.

The engage key engages the bridge's driver (GTA5_DRIVER: openpilot, the default, or map) and disengages it, and the
HUD's engaged state, the hand-back to the player and arrival go through it, whichever it is. Each bridge step the brain
(gta5_brain.py) steps the driver driving with the game state and the route, then sends the plugin its control message.
- OpenpilotDriver: openpilot, through the OpenpilotIO (gta5_opio.py): engaged with the simulated car's cruise buttons,
  its controls carControl's.
- MapDriver: our map drive (gta5_mapdrive.py) along the route. Without a route it can't engage, and the engage key
  engages openpilot instead; without openpilot (gta5_host.py --no-link) the brain gives it a cruise route instead, on
  along the road (map/cruise.py). Expert mode (gta5_expert.py) drives with a MapDriver of its own.
While the map driver drives, the arrow keys are the player's prompts to it (the brain's), not the plugin's indicator.
A driver hands the car back at once on the player's brake, gas or steering. The simulated person in the seat who works
the stalk for openpilot is gta5_driver.HumanDriver, not one of these."""
import os

from openpilot.tools.sim.bridge.gta5.gta5_mapdrive import USER_STEER, MapDrive
from openpilot.tools.sim.bridge.gta5.gta5_opio import OpenpilotIO

DRIVER = os.getenv("GTA5_DRIVER", "openpilot")
DRIVERS = ("openpilot", "map")
INACTIVE = {"type": "control", "active": False}
TOOK_OVER = "abort: driver input"  # why a driver hands back to the player, as the map drive's own abort has it
ROUTE_GAP = 5.0  # s an engaged map driver drives on along its last route while there's none (routing a new destination)
ROAD_ENDS = "the road ends"  # why the map driver hands back at a cruise route's end, which only a road's end leaves short
INDICATOR_EVERY = 2.0  # s between asking the plugin again for the indicator its state doesn't show


def took_over(state: dict) -> bool:
  """The player's brake, gas or steering."""
  user = state.get("user") or {}
  return abs(float(user.get("steer") or 0.0)) > USER_STEER or bool(user.get("brake")) or bool(user.get("gas"))


class Driver:
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
    """A bridge step: why it hands the car back now (the brain then disengages it; "arrived" ends the destination too),
    else None."""
    return None

  def controls(self, state: dict) -> dict | None:
    """The plugin's control message for this step; None leaves the car to the player."""
    return None


class OpenpilotDriver(Driver):
  """openpilot drives: the cruise buttons engage it, and its controls (carControl's curvature and acceleration) go to
  the car."""
  name = "openpilot"

  def __init__(self, io: OpenpilotIO):
    self.io = io
    self.steering = False  # whether the player was steering
    self.braking = False

  @property
  def engaged(self) -> bool:
    return bool(self.io.engaged)

  def engage(self, route) -> bool:
    self.io.cruise("cruise_down")
    return True

  def disengage(self, why: str) -> None:
    self.io.cruise("cruise_cancel")

  def step(self, state: dict, route, now: float) -> str | None:
    # the brake and steering disengage (gas overrides), since the game's steering has no torque for openpilot to blend
    # with, and fighting it soon trips the excessive actuation lockout; the simulated car's cruise stays on through a
    # brake press otherwise, and openpilot re-engages on its release
    user = state.get("user") or {}
    steering, braking = abs(user.get("steer", 0)) > USER_STEER, bool(user.get("brake"))
    if (steering and not self.steering or braking and not self.braking) and self.engaged:
      self.io.cruise("cruise_cancel")
    self.steering, self.braking = steering, braking
    return None

  def controls(self, state: dict) -> dict | None:
    return self.io.controls(state)


class MapDriver(Driver):
  """Our map drive along the route: a new MapDrive (its plan made where the car is) each time it engages.

  wait_for_route (expert mode's trips): it engages before the route comes, holding the car until it does, and drives on
  whatever happens to the route; else it needs one to engage, and hands back after ROUTE_GAP s without one."""
  name = "map"

  def __init__(self, send, spec: dict | None = None, event=None, wait_for_route: bool = False):
    self.send = send
    self.spec = dict(spec or {})
    self.event = event  # event(what, **detail): the map drive's events, for a log
    self.wait_for_route = wait_for_route
    self.md: MapDrive | None = None  # the last drive's, kept after it ends for what it did
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
    self.md = MapDrive(dict(self.spec))
    self.on, self.msg, self.indicator, self.label, self.no_route_since = True, None, None, "none", None
    self.send({"type": "ai", "on": 0, "indicator": "off", "arrows": "bridge"})  # our controls drive, not the game's AI
    print("gta5: the map driver drives", flush=True)
    return True

  def disengage(self, why: str) -> None:
    if not self.on:
      return
    self.on, self.msg = False, None
    self.send(INACTIVE)
    self.send({"type": "ai", "indicator": "off", "arrows": "plugin"})
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
    if md.finished == "arrived" and getattr(md.route, "cruise", False):
      return ROAD_ENDS  # stopped at the end of a cruise route: no destination to end
    return md.finished

  def _indicator(self, side: str | None, label: str, state: dict, now: float):
    self.label = label
    if side != self.indicator or (state.get("indicator") != side and now >= self.next_indicator):
      self.indicator, self.next_indicator = side, now + INDICATOR_EVERY
      self.send({"type": "ai", "indicator": side or "off"})

  def controls(self, state: dict) -> dict | None:
    return self.msg if self.on else None

  def ask_lane(self, side: str, now: float) -> str:
    """An arrow key's lane change (MapDrive.ask_lane), "no_lane" while it isn't driving."""
    return self.md.ask_lane(side, now) if self.on and self.md is not None else "no_lane"

  def take_prompts(self) -> list[tuple[str, str]]:
    return self.md.take_prompts() if self.md is not None else []

  @property
  def drives_route(self) -> bool:
    """Driving the route itself, not holding for one, nor stopping after an abort."""
    return self.on and self.md is not None and self.md.phase not in ("abort", "wait", "")


def driver_from_env(openpilot: OpenpilotDriver, send, event=None, which: str | None = None) -> Driver:
  """The engage key's driver: `which` where given, else GTA5_DRIVER's."""
  if os.getenv("GTA5_PILOT") is not None and os.getenv("GTA5_DRIVER") is None:
    raise ValueError("GTA5_PILOT is GTA5_DRIVER now")
  which = which or DRIVER
  if which not in DRIVERS:
    raise ValueError(f"GTA5_DRIVER={which}: one of {', '.join(DRIVERS)}")
  if which == "map":
    print(f"gta5: the engage key drives with the map driver ({'GTA5_DRIVER=map' if DRIVER == 'map' else 'no openpilot'})",
          flush=True)
    return MapDriver(send, event=event)
  return openpilot
