"""The navigation guidance the onroad UI shows, read from navInstruction and navRoute. Shared by every layout: plain
values, and the route and nearby roads in local metres, projected once per navRoute."""
import datetime
import math
import time
from dataclasses import dataclass, field

import numpy as np

from openpilot.common.constants import CV
from openpilot.system.ui.lib.multilang import tr

M_PER_DEG = 111319.49  # m per degree of latitude, as the map's projection
STALE_AFTER = 2.0  # s without a navInstruction: navigation has stopped
# The car's pose between navigation's updates (10 Hz): dead reckoned each frame from openpilot's own yaw rate and speed,
# and drawn towards each reported pose with these time constants, so the map neither snaps nor drifts
POSITION_TC = 0.5  # s
BEARING_TC = 0.8  # s
SNAP_DIST = 50.0  # m from the reported position: a new place, taken at once
DT_MAX = 1.0  # s of dead reckoning in one step: through a stalled frame, not a screen that was off
YAW_STALE = 0.5  # s without deviceMotion: no yaw rate
FEET_PER_M = 3.28084


@dataclass
class Lane:
  directions: list[str]
  active: bool
  active_direction: str
  oncoming: bool
  current: bool


@dataclass
class Maneuver:
  distance: float  # m from the car
  type: str
  modifier: str
  primary: str


@dataclass
class Road:
  points: np.ndarray  # (n, 2) local m, east and north
  width: float  # m


@dataclass
class Guidance:
  maneuver: Maneuver | None = None
  secondary: str = ""
  maneuvers: list[Maneuver] = field(default_factory=list)
  lanes: list[Lane] = field(default_factory=list)
  show_lanes: bool = False
  lane_distance: float = 0.0
  lane_open_distance: float = 0.0
  distance_remaining: float = 0.0
  time_remaining: float = 0.0
  destination: str = ""
  speed_limit: float = 0.0
  position: tuple[float, float] | None = None  # lat, lon
  bearing: float = 0.0


def instruction_guidance(ni) -> Guidance:
  g = Guidance()
  if ni.maneuverType:
    g.maneuver = Maneuver(ni.maneuverDistance, ni.maneuverType, ni.maneuverModifier, ni.maneuverPrimaryText)
  g.secondary = ni.maneuverSecondaryText
  g.maneuvers = [Maneuver(m.distance, m.type, m.modifier, m.primaryText) for m in ni.allManeuvers]
  g.lanes = [Lane([str(d) for d in lane.directions], lane.active, str(lane.activeDirection), lane.oncoming, lane.current)
             for lane in ni.lanes]
  g.show_lanes = ni.showFull and any(lane.active for lane in g.lanes)
  g.lane_distance, g.lane_open_distance = ni.laneDistance, ni.laneOpenDistance
  g.distance_remaining, g.time_remaining = ni.distanceRemaining, ni.timeRemaining
  g.destination, g.speed_limit = ni.destinationName, ni.speedLimit
  if ni.position.latitude or ni.position.longitude:
    g.position = (ni.position.latitude, ni.position.longitude)
  g.bearing = ni.bearingDeg
  return g


class Projection:
  """Equirectangular about an origin: fine for the few km a route map shows, anywhere but the poles."""
  def __init__(self, lat0: float, lon0: float):
    self.lat0, self.lon0 = lat0, lon0
    self.kx = M_PER_DEG * math.cos(math.radians(lat0))

  def local(self, lat, lon) -> np.ndarray:
    return np.stack([(np.asarray(lon, np.float64) - self.lon0) * self.kx, (np.asarray(lat, np.float64) - self.lat0) * M_PER_DEG], axis=-1)

  def lat_lon(self, xy: np.ndarray) -> tuple[float, float]:
    return float(self.lat0 + xy[1] / M_PER_DEG), float(self.lon0 + xy[0] / self.kx)


def wrap(deg: float) -> float:
  return (deg + 180.0) % 360.0 - 180.0


class PoseTracker:
  """The car's position (local m) and heading (deg clockwise from north) for drawing: moved on every frame by the car's
  speed and yaw rate, and pulled smoothly towards each pose navigation reports, moved on alike since it came."""
  def __init__(self):
    self.pos: np.ndarray | None = None
    self.bearing = 0.0
    self._fix: tuple[np.ndarray, float] | None = None
    self._t: float | None = None

  def reset(self) -> None:
    self.pos, self._fix, self._t = None, None, None

  def shift(self, old: 'Projection', new: 'Projection') -> None:
    """Keeps the pose where it is as the local frame moves to another origin (a new route)."""
    if self.pos is not None:
      self.pos = new.local(*old.lat_lon(self.pos))
    if self._fix is not None:
      self._fix = (new.local(*old.lat_lon(self._fix[0])), self._fix[1])

  def update(self, now: float, v: float, yaw_rate: float, fix: tuple[np.ndarray, float] | None = None) -> None:
    """yaw_rate: rad/s, clockwise (right) positive, as the bearing turns. fix: a newly reported (position, bearing)."""
    dt = min(max(now - self._t, 0.0), DT_MAX) if self._t is not None else 0.0
    self._t = now
    if fix is not None:
      if self.pos is None or float(np.hypot(*(fix[0] - self.pos))) > SNAP_DIST:
        self.pos, self.bearing = fix[0].copy(), fix[1]
      self._fix = (fix[0].copy(), fix[1])
    if self.pos is None or self._fix is None:
      return
    turn = math.degrees(yaw_rate) * dt
    self.pos, self.bearing = self._step(self.pos, self.bearing, v * dt, turn)
    self._fix = self._step(self._fix[0], self._fix[1], v * dt, turn)
    self.pos = self.pos + (self._fix[0] - self.pos) * (1.0 - math.exp(-dt / POSITION_TC))
    self.bearing = (self.bearing + wrap(self._fix[1] - self.bearing) * (1.0 - math.exp(-dt / BEARING_TC))) % 360.0

  @staticmethod
  def _step(pos: np.ndarray, bearing: float, dist: float, turn: float) -> tuple[np.ndarray, float]:
    mid = math.radians(bearing + turn / 2)
    return pos + dist * np.array([math.sin(mid), math.cos(mid)]), (bearing + turn) % 360.0


class NavState:
  """Reads the nav messages from a SubMaster that has them. `active` says whether to show the guidance."""
  def __init__(self):
    self.guidance = Guidance()
    self.active = False
    self.projection: Projection | None = None
    self.route = np.zeros((0, 2))  # local m
    self.route_along = np.zeros(0)
    self.roads: list[Road] = []
    self.version = 0  # counts navRoute updates, for drawing caches
    self.pose = PoseTracker()
    self._route_fingerprint: tuple = ()

  def update(self, sm) -> None:
    now = time.monotonic()
    if sm.updated["navRoute"]:
      self._read_route(sm["navRoute"])
    fix = self.read_instruction(sm["navInstruction"]) if sm.updated["navInstruction"] else None
    fresh = sm.recv_frame["navInstruction"] > 0 and now - sm.recv_time["navInstruction"] < STALE_AFTER
    self.active = fresh and sm["navInstruction"].valid
    if not self.active:
      self.pose.reset()
      return
    # the device's z axis points down, so its rate about z is the bearing's (clockwise)
    moving = sm.recv_frame["deviceMotion"] > 0 and now - sm.recv_time["deviceMotion"] < YAW_STALE
    yaw_rate = sm["deviceMotion"].angularVelocityDevice.z if moving else 0.0
    self.pose.update(now, sm["carState"].vEgo, yaw_rate, fix)

  def read_instruction(self, ni) -> tuple[np.ndarray, float] | None:
    """Takes a navInstruction's guidance; returns the pose it reports, in local m, if any."""
    self.guidance = instruction_guidance(ni) if ni.valid else Guidance()
    g = self.guidance
    if g.position is None or self.projection is None:
      return None
    return self.projection.local(*g.position), g.bearing

  @staticmethod
  def _fingerprint(nr) -> tuple:
    """Enough of a navRoute to tell a new one from the same one sent again, without reading it all."""
    def ends(coords):
      return (len(coords),) + ((coords[0].latitude, coords[0].longitude, coords[-1].latitude, coords[-1].longitude) if len(coords) else ())
    roads = nr.roads
    return ends(nr.coordinates) + (len(roads),) + (ends(roads[0].coordinates) + ends(roads[-1].coordinates) if len(roads) else ())

  def _read_route(self, nr) -> None:
    fingerprint = self._fingerprint(nr)
    if fingerprint == self._route_fingerprint:
      return
    self._route_fingerprint = fingerprint
    coords = nr.coordinates
    self.version += 1
    if len(coords) < 2:
      self.route, self.route_along, self.roads = np.zeros((0, 2)), np.zeros(0), []
      return
    lat = np.array([c.latitude for c in coords])
    lon = np.array([c.longitude for c in coords])
    old, self.projection = self.projection, Projection(float(lat[0]), float(lon[0]))
    if old is not None:
      self.pose.shift(old, self.projection)
    self.route = self.projection.local(lat, lon)
    self.route_along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(self.route, axis=0).T))))
    self.roads = [Road(self.projection.local([c.latitude for c in r.coordinates], [c.longitude for c in r.coordinates]), r.width)
                  for r in nr.roads if len(r.coordinates) >= 2]

  def car(self) -> tuple[np.ndarray, float] | None:
    """The car in local m and its heading (deg clockwise from north), as drawn this frame."""
    return (self.pose.pos, self.pose.bearing) if self.pose.pos is not None else None

  def along_route(self, pos: np.ndarray, bearing: float) -> tuple[float, int]:
    """m along the route to its point nearest pos, and the segment it's on, of those heading the car's way: where the
    route comes back past itself, the part the car is driving."""
    pts = self.route
    if len(pts) < 2:
      return 0.0, 0
    a, ab = pts[:-1], np.diff(pts, axis=0)
    t = np.clip(((pos - a) * ab).sum(axis=1) / np.maximum((ab * ab).sum(axis=1), 1e-9), 0.0, 1.0)
    d = np.hypot(*(a + ab * t[:, None] - pos).T)
    off = np.abs((np.degrees(np.arctan2(ab[:, 0], ab[:, 1])) - bearing + 180) % 360 - 180)
    if (off < 90).any():
      d = np.where(off < 90, d, np.inf)
    k = int(np.argmin(d))
    return float(self.route_along[k] + t[k] * math.hypot(*ab[k])), k

  def route_point(self, along: float) -> np.ndarray:
    return np.array([np.interp(along, self.route_along, self.route[:, 0]), np.interp(along, self.route_along, self.route[:, 1])])


def format_distance(m: float, metric: bool) -> str:
  """Rounded as a driver reads it: 10 m steps under a km, then 0.1 km; 50 ft steps under 0.1 mi, then 0.1 mi."""
  if metric:
    if m < 1000:
      return f"{max(int(round(m / 10.0)) * 10, 0)} m"
    return f"{m / 1000:.1f} km"
  miles = m / (CV.MPH_TO_KPH * 1000.0)
  if miles < 0.1:
    return f"{max(int(round(m * FEET_PER_M / 50.0)) * 50, 0)} ft"
  return f"{miles:.1f} mi"


def format_duration(s: float) -> str:
  minutes = max(math.ceil(s / 60.0), 1) if s > 0 else 0
  if minutes < 60:
    return tr("{} min").format(minutes)
  return tr("{} h {} min").format(minutes // 60, minutes % 60)


def format_arrival(s: float) -> str:
  return (datetime.datetime.now() + datetime.timedelta(seconds=s)).strftime("%H:%M")
