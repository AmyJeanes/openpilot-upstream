"""The GTA layer's side of navd: its inputs from the game and the route, and the destination from the game.

nav_inputs: the planner's inputs (navd/inputs.py) from the world's state, which still carries the game's truth (pose,
heading, the lane from the route and the plugin) until navd localizes itself and estimates the lane.

Destination: the game map's waypoint or a pick on the map view, whichever was set last, kept in game metres for the
router and written to the NavDestination param as lat/lon, as a phone or the UI sets it for navd; cleared when the
car arrives (and then the game's waypoint too).
"""
import numpy as np

from openpilot.selfdrive.navd.inputs import NavInputs
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon

CANCELLED_FROM = 100.0  # m: GTA clears the waypoint as the car nears it; farther off, the player cleared it


def nav_inputs(state: dict, engaged: bool, indicator: str | None, desire: dict[str, float], now: float) -> NavInputs:
  """The world's state (after _map_route: the route's points and Route.info) as the planner's inputs."""
  return NavInputs(
    t=now, engaged=engaged, v=state.get("vEgo", 0.0), yaw_rate=state.get("yawRate", 0.0), blinker=indicator, desire=desire,
    route=state.get("route"), dest=state.get("waypoint"), route_end=state.get("routeEnd"), forks=state.get("forks"),
    stops=state.get("stops"), junctions=state.get("junctions"), limits=state.get("limits"), lane_arrows=state.get("laneArrows"),
    lane_drops=state.get("laneDrops"), two_way=state.get("twoWay"),
    pos=state.get("pos", (0.0, 0.0)), heading=state.get("heading", 0.0), truth_lane=state.get("lane"),
    truth_lane_frac=state.get("laneFrac"), truth_lane_plugin=state.get("lanePlugin"), truth_lane_map=state.get("laneMap"))


class Destination:
  def __init__(self, params, send):
    self.params = params
    self.send = send  # to the plugin
    self.dest: np.ndarray | None = None  # game metres
    self.from_game = False
    self.game_waypoint: np.ndarray | None = None
    self.written: tuple | None = None  # what NavDestination holds
    self.param_ok = True

  def update(self, waypoint, pos: np.ndarray, picked: tuple | None) -> np.ndarray | None:
    """The destination, given the game's waypoint ([x, y], (0, 0) or None for none), the car's position and a pick on
    the map view (([x, y] or None,) or None for none)."""
    waypoint = np.array(waypoint or (0.0, 0.0), dtype=float)
    if waypoint.any():
      if self.game_waypoint is None or np.hypot(*(waypoint - self.game_waypoint)) > 1.0:
        self.dest, self.from_game = waypoint, True
      self.game_waypoint = waypoint
    else:
      self.game_waypoint = None
      if self.from_game and self.dest is not None and np.hypot(*(self.dest - pos)) > CANCELLED_FROM:
        self.dest = None
    if picked is not None:
      self.dest, self.from_game = (np.array(picked[0], dtype=float) if picked[0] else None), False
    self._write()
    return self.dest

  def arrived(self):
    """navd says the car has arrived: the destination is done, and the game's waypoint goes too."""
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
