"""The GTA layer's side of navd: its inputs from the game and the route, and the destination from the game.

nav_inputs: the planner's inputs (navd/inputs.py) from the world's state, which still carries the game's truth (pose,
heading, the lane from the route and the plugin) until navd localizes itself and estimates the lane.

Destination: the game map's waypoint (else the end of a mission's GPS route) or a pick on the map view, whichever was
set last, kept in game metres for the router and written to the NavDestination param as lat/lon, as a phone or the UI
sets it for navd; cleared when the car arrives (and then the game's waypoint too, never a mission's blip).
"""
import time

import numpy as np

from openpilot.selfdrive.navd.inputs import NavInputs
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon

CANCELLED_FROM = 100.0  # m: GTA clears the waypoint as the car nears it; farther off, the player cleared it
ENDED_CHECK = 0.5  # s between looks at NavDestination for a route the driver ended (the UI's slide to end removes it)
# m the game's target moves before it's a new destination: a mission's blip can be on a moving car or ped, and each new
# destination drops the route until the router has another
MOVED = {"waypoint": 1.0, "mission": 30.0}


def nav_inputs(state: dict, engaged: bool, indicator: str | None, desire: dict[str, float], now: float,
               drive: bool = True) -> NavInputs:
  """The world's state (after _map_route: the route's points and Route.info) as the planner's inputs."""
  return NavInputs(
    t=now, engaged=engaged, drive=drive, v=state.get("vEgo", 0.0), yaw_rate=state.get("yawRate", 0.0), blinker=indicator, desire=desire,
    route=state.get("route"), dest=state.get("waypoint"), route_end=state.get("routeEnd"), forks=state.get("forks"),
    stops=state.get("stops"), stop_kinds=state.get("stopKinds"), junctions=state.get("junctions"), limits=state.get("limits"),
    lane_arrows=state.get("laneArrows"), lane_drops=state.get("laneDrops"), lane_maps=state.get("laneMaps"), turns=state.get("turns"), two_way=state.get("twoWay"),
    road_classes=state.get("roadClasses"), model_lane=state.get("modelLane"),
    pos=state.get("pos", (0.0, 0.0)), heading=state.get("heading", 0.0), truth_lane=state.get("lane"),
    truth_lane_frac=state.get("laneFrac"), truth_lane_plugin=state.get("lanePlugin"), truth_lane_map=state.get("laneMap"))


class Destination:
  def __init__(self, params, send, ends: bool = True):
    self.params = params
    self.send = send  # to the plugin
    # honour the UI's slide to end; off, a NavDestination the manager clears (offroad, restart) can't end a test drive
    self.ends = ends
    self.dest: np.ndarray | None = None  # game metres
    self.from_game = False
    self.game_target: np.ndarray | None = None  # the game's target the destination last took, and its kind
    self.game_kind: str | None = None  # "waypoint" (the player's) or "mission" (a mission's GPS route)
    self.written: tuple | None = None  # what NavDestination holds
    self.param_ok = True
    self.ended_target: np.ndarray | None = None  # the game's target of a route the driver ended: ignored until it changes
    self.check_every = ENDED_CHECK
    self._checked = 0.0

  def update(self, waypoint, pos: np.ndarray, picked: tuple | None, mission=None) -> np.ndarray | None:
    """The destination, given the game's waypoint ([x, y], (0, 0) or None for none), the car's position, a pick on the
    map view (([x, y] or None,) or None for none) and the end of a mission's GPS route ([x, y] or None)."""
    target, kind = self._game_target(waypoint, mission)
    self._check_ended(target)
    if self.ended_target is not None:
      if target is not None and np.hypot(*(target - self.ended_target)) <= MOVED[kind]:
        target = None  # the ended route's: as if there were none
      else:
        self.ended_target = None  # a new target, or none: the next one counts again
    if target is not None and not (kind == "mission" and self._nearing_cleared_waypoint(pos)):
      if kind != self.game_kind or self.game_target is None or np.hypot(*(target - self.game_target)) > MOVED[kind]:
        self.dest, self.from_game = target, True
        self.game_target, self.game_kind = target, kind
    elif target is None:
      self.game_target = None  # the same target back is new again; its kind stays for arriving
      if self.from_game and self.dest is not None and np.hypot(*(self.dest - pos)) > CANCELLED_FROM:
        self.dest = None
    if picked is not None:
      self.dest, self.from_game = (np.array(picked[0], dtype=float) if picked[0] else None), False
    self._write()
    return self.dest

  @staticmethod
  def _game_target(waypoint, mission) -> tuple[np.ndarray | None, str | None]:
    """The player's waypoint, else the mission's route's end."""
    for value, kind in ((waypoint, "waypoint"), (mission, "mission")):
      if value is not None:
        value = np.array(value, dtype=float)
        if value.any():
          return value, kind
    return None, None

  def _nearing_cleared_waypoint(self, pos: np.ndarray) -> bool:
    """GTA cleared the player's waypoint as the car neared it: a mission's route waits until the car arrives."""
    return self.game_kind == "waypoint" and self.from_game and self.dest is not None and np.hypot(*(self.dest - pos)) <= CANCELLED_FROM

  def _check_ended(self, target: np.ndarray | None):
    """The driver ended the route (the UI removed NavDestination, which we wrote): the destination goes, and the game's
    waypoint or mission route is ignored until it changes (it isn't cleared in the game)."""
    now = time.monotonic()
    if not self.ends or self.written is None or not self.param_ok or now - self._checked < self.check_every:
      return
    self._checked = now
    try:
      ended = self.params.get("NavDestination") is None
    except Exception:
      return
    if ended:
      self.dest, self.written = None, None
      self.ended_target = target.copy() if target is not None and self.from_game else None
      self.from_game = False
      print("gta5: the driver ended the route")

  def arrived(self):
    """navd says the car has arrived: the destination is done, and the player's waypoint goes too (a mission's blip is
    the mission's to clear)."""
    if not (self.from_game and self.game_kind == "mission"):
      self.send({"type": "waypoint", "off": True})
    self.dest = None
    self._write()

  def _write(self):
    want = None if self.dest is None else tuple(float(v) for v in self.dest)
    if want == self.written or not self.param_ok:
      return
    self.written = want
    try:
      if want is None:
        self.params.remove("NavDestination")
      else:
        lat, lon = to_lat_lon(*want)
        self.params.put("NavDestination", {"latitude": lat, "longitude": lon})
    except Exception as e:  # params built before the key existed: navd's runtime doesn't read it yet
      print(f"gta5: no NavDestination param: {e}")
      self.param_ok = False
