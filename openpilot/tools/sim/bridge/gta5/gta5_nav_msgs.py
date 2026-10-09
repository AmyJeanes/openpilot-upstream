"""openpilot's navInstruction and navRoute messages (cereal log.capnp) from the bridge's route, for the onroad UI's
navigation view. Nothing here is GTA's: it reads a router.Route over the map, its lane slots, the map's roads and the
car's place on the route, and writes the map's own lat/lon, as a navigation service on a real map would.

navInstruction: the next maneuver (OSRM's type and modifier words, as upstream navd sent), the lanes for it, the time
and distance left, and the car's position and heading. navRoute: the route, and the roads near the route ahead
simplified for a small map (fork fields), from a background thread since gathering them takes tens of ms."""
import threading
import time
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

from openpilot.cereal import log, messaging
from openpilot.selfdrive.navd.maneuvers import junction_maneuvers
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import FORWARD, LEFTS, RIGHTS

Direction = log.NavInstruction.Direction

INSTRUCTION_EVERY = 0.1  # s
ROUTE_EVERY = 2.0  # s, for a UI that starts after the route was sent
ROUTE_TOLERANCE = 1.0  # m the simplified route keeps to the route
ROAD_TOLERANCE = 2.0  # m, for the roads near it
CORRIDOR_BEHIND, CORRIDOR_AHEAD = 200.0, 2500.0  # m of route the roads are gathered along
CORRIDOR_WIDTH = 250.0  # m either side of it
CORRIDOR_STEP = 25.0  # m between the route's points the roads are measured from, and the most between a road's
CORRIDOR_AGAIN = 800.0  # m driven before gathering the roads again
MAX_ROAD_POINTS = 8000  # in one navRoute
DEFAULT_ROAD_WIDTH = 7.0  # m
MINOR = frozenset({"service", "track", "living_street"})  # left off the map: car parks and drives clutter it
DEFAULT_SPEED = 13.4  # m/s, for the time left where the route has no times
ON_ROUTE = 15.0  # m: off the route by more, and outside its road's kerbs, the car's lane isn't one of its road's
TARGET_DIRECTION = {"left": "left", "right": "right", "through": "straight"}
FORK_DIRECTION = {"left": "slightLeft", "right": "slightRight"}

# Valhalla maneuver types as OSRM's (type, modifier)
MANEUVERS = {
  1: ("depart", "straight"), 2: ("depart", "right"), 3: ("depart", "left"),
  4: ("arrive", "straight"), 5: ("arrive", "right"), 6: ("arrive", "left"),
  7: ("new name", "straight"), 8: ("continue", "straight"),
  9: ("turn", "slight right"), 10: ("turn", "right"), 11: ("turn", "sharp right"),
  12: ("turn", "uturn"), 13: ("turn", "uturn"),
  14: ("turn", "sharp left"), 15: ("turn", "left"), 16: ("turn", "slight left"),
  17: ("on ramp", "straight"), 18: ("on ramp", "right"), 19: ("on ramp", "left"),
  20: ("off ramp", "slight right"), 21: ("off ramp", "slight left"),
  22: ("fork", "straight"), 23: ("fork", "slight right"), 24: ("fork", "slight left"),
  25: ("merge", "straight"), 37: ("merge", "slight right"), 38: ("merge", "slight left"),
  26: ("roundabout", "straight"), 27: ("exit roundabout", "straight"),
}
SHOWN = {"turn", "on ramp", "off ramp", "fork", "merge", "roundabout", "exit roundabout", "arrive"}


class Maneuver(NamedTuple):
  along: float  # m along the route
  type: str
  modifier: str
  primary: str  # the road it leads onto
  secondary: str  # an exit number or sign, else ""


def street(m: dict) -> str:
  names = m.get("street_names") or m.get("begin_street_names") or []
  return names[0] if names else ""


def sign(m: dict) -> str:
  s = m.get("sign") or {}
  exits = [e.get("text", "") for e in s.get("exit_number_elements", [])]
  toward = [e.get("text", "") for e in s.get("exit_toward_elements", []) + s.get("exit_branch_elements", [])]
  parts = ([f"Exit {exits[0]}"] if exits and exits[0] else []) + toward[:1]
  return ", ".join(p for p in parts if p)


def maneuvers(route) -> list[Maneuver]:
  """The route's maneuvers worth showing, in order: turns, ramps, forks, merges and the arrival; turns through a
  junction's links as one (navd's junction_maneuvers)."""
  out = []
  for m in junction_maneuvers(route):
    kind = MANEUVERS.get(m.get("type"))
    i = m.get("begin_shape_index", 0)
    if kind is None or kind[0] not in SHOWN or i >= len(route.along):
      continue
    out.append(Maneuver(float(route.along[i]), kind[0], kind[1], street(m), sign(m)))
  return out


def destination_name(route) -> str:
  """The destination's street: the arrival's, else the last maneuver's that names one."""
  for m in reversed(route.maneuvers):
    if name := street(m):
      return name
  return ""


def time_remaining(route, at: float) -> float:
  """s to the destination from `at` m along: by the router's time for each maneuver's stretch, the part of it left."""
  mans = route.maneuvers
  if not mans or any("time" not in m for m in mans):
    return max(route.length - at, 0.0) / DEFAULT_SPEED
  starts = [float(route.along[min(m.get("begin_shape_index", 0), len(route.along) - 1)]) for m in mans]
  ends = starts[1:] + [route.length]
  total = 0.0
  for m, a, b in zip(mans, starts, ends, strict=True):
    if b <= at:
      continue
    total += float(m["time"]) * (1.0 if a >= at else (b - at) / max(b - a, 1e-6))
  return total


def heading_at(route, at: float) -> float:
  """The route's bearing at `at` m along, clockwise from north."""
  k = int(np.clip(np.searchsorted(route.along, at, side="right") - 1, 0, len(route.points) - 2))
  d = route.points[k + 1] - route.points[k]
  return float(np.degrees(np.arctan2(d[0], d[1])) % 360)


def point_at(route, at: float) -> np.ndarray:
  return np.array([np.interp(at, route.along, route.points[:, 0]), np.interp(at, route.along, route.points[:, 1])])


def directions(turns: frozenset[str]) -> list[str]:
  """A lane's arrows as NavInstruction directions, left to right; straight where none are marked."""
  out = []
  for t, d in (("sharp_left", "left"), ("left", "left"), ("slight_left", "slightLeft"), ("through", "straight"),
               ("slight_right", "slightRight"), ("right", "right"), ("sharp_right", "right")):
    if t in turns and d not in out:
      out.append(d)
  return out or ["straight"]


class LaneGuide(NamedTuple):
  lanes: list[dict]  # left to right: directions, active, activeDirection, oncoming, current
  show: bool
  distance: float  # m ahead by which to be in an active lane, 0 when in one
  opens: float  # m ahead to where it opens, 0 when open


NO_LANES = LaneGuide([], False, 0.0, 0.0)


def lane_guide(slots, s: float, v: float, car_lane: list[int] | None) -> LaneGuide:
  """The road's lanes at the car s m along, and the ones the route needs (navd lane_slots.LaneSlots.target).
  car_lane is router.Route.lane()'s [lane from the left of ours, of how many]."""
  if slots is None:
    return NO_LANES
  here, target = slots.target(s, v)
  if here is None:
    return NO_LANES
  want, centre = (target.want, target.centre) if target is not None else (set(), False)
  side = getattr(target.move.nav, "side", "through") if target is not None else "through"
  fork = target is not None and target.move.fork
  use = (FORK_DIRECTION if fork else TARGET_DIRECTION).get(side, "straight")
  current = here.first + car_lane[0] if car_lane else None
  lanes, inside = [], False
  for i, span in enumerate(here.spans):
    ours = here.first <= i < here.first + here.lanes
    shared = span.heading == 0 and not ours  # a centre turn lane
    dirs = directions(span.lane.turns)
    if shared:
      dirs = [d for d in dirs if d != "straight"] or ["left" if span.lane.turns & LEFTS else "right" if span.lane.turns & RIGHTS else "straight"]
    active = (ours and i - here.first in want) or (shared and centre)
    lanes.append({"directions": dirs, "active": active, "activeDirection": (use if use in dirs else dirs[0]) if active else "none",
                  "oncoming": span.heading == -1, "current": i == current})
    inside |= active and i == current
  if target is None:
    return LaneGuide(lanes, False, 0.0, 0.0)
  opens = max(target.opens - s, 0.0) if target.opens is not None else 0.0
  return LaneGuide(lanes, True, 0.0 if inside else max(target.end - s, 0.0), opens)


def fill_instruction(ni, route, at: float, v: float, project=to_lat_lon, lanes: LaneGuide = NO_LANES,
                     pose: tuple | None = None) -> None:
  """A navInstruction for the car `at` m along `route`. `pose` is the car's own position (map metres) and heading
  (deg clockwise from north), as the car's localizer has them; without it, its place on the route stands in."""
  ni.valid = True
  ahead = [m for m in maneuvers(route) if m.along > at]
  if ahead:
    m = ahead[0]
    ni.maneuverType, ni.maneuverModifier = m.type, m.modifier
    ni.maneuverPrimaryText = m.primary or (destination_name(route) if m.type == "arrive" else "")
    ni.maneuverSecondaryText = m.secondary
    ni.maneuverDistance = m.along - at
    ni.init("allManeuvers", min(len(ahead), 5))
    for out, a in zip(ni.allManeuvers, ahead, strict=False):
      out.distance, out.type, out.modifier, out.primaryText = a.along - at, a.type, a.modifier, a.primary
  ni.distanceRemaining = max(route.length - at, 0.0)
  ni.timeRemaining = ni.timeRemainingTypical = time_remaining(route, at)
  k = int(np.clip(np.searchsorted(route.along, at, side="right") - 1, 0, max(len(route.limit_list) - 1, 0)))
  ni.speedLimit = float(route.limit_list[k]) if route.limit_list else 0.0
  ni.speedLimitSign = "mutcd"
  ni.destinationName = destination_name(route)
  xy, bearing = (pose[0], pose[1]) if pose is not None else (point_at(route, at), heading_at(route, at))
  lat, lon = project(float(xy[0]), float(xy[1]))
  ni.position.latitude, ni.position.longitude = float(lat), float(lon)
  ni.bearingDeg = float(bearing) % 360
  ni.showFull = lanes.show
  ni.laneDistance, ni.laneOpenDistance = lanes.distance, lanes.opens
  ni.init("lanes", len(lanes.lanes))
  for out, lane in zip(ni.lanes, lanes.lanes, strict=True):
    out.directions = lane["directions"]
    out.active, out.activeDirection = lane["active"], lane["activeDirection"]
    out.oncoming, out.current = lane["oncoming"], lane["current"]


def simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
  """Douglas-Peucker: the points a polyline keeps within `tolerance` m."""
  n = len(points)
  if n < 3:
    return points
  keep = np.zeros(n, bool)
  keep[[0, -1]] = True
  stack = [(0, n - 1)]
  while stack:
    a, b = stack.pop()
    if b - a < 2:
      continue
    seg = points[b] - points[a]
    rel = points[a + 1:b] - points[a]
    length = np.hypot(*seg)
    d = np.abs(rel[:, 0] * seg[1] - rel[:, 1] * seg[0]) / length if length > 1e-9 else np.hypot(*rel.T)
    k = int(np.argmax(d))
    if d[k] > tolerance:
      keep[a + 1 + k] = True
      stack += [(a, a + 1 + k), (a + 1 + k, b)]
  return points[keep]


def densify(points: np.ndarray, step: float) -> np.ndarray:
  """Points no further than `step` m apart along the polyline."""
  d = np.hypot(*np.diff(points, axis=0).T)
  if len(points) < 2 or d.max() <= step:
    return points
  s = np.concatenate(([0.0], np.cumsum(d)))
  t = np.unique(np.concatenate((s, np.arange(0.0, s[-1], step))))
  return np.stack([np.interp(t, s, points[:, 0]), np.interp(t, s, points[:, 1])], axis=1)


class Roads:
  """The map's roads (an osm_lanes.OsmLanes), each way's points and width, with their bounding boxes for the roads
  near a stretch of route."""
  def __init__(self, osm):
    self.points: list[np.ndarray] = []
    self.widths: list[float] = []
    boxes = []
    for wid, (tags, _) in osm.ways.items():
      if tags.get("highway") in MINOR:
        continue
      pts = osm.way_points(wid)
      if len(pts) < 2:
        continue
      try:
        lo, hi = osm.lanes(wid).edges(FORWARD)
        width = float(hi - lo) if hi > lo else DEFAULT_ROAD_WIDTH
      except Exception:
        width = DEFAULT_ROAD_WIDTH
      self.points.append(pts)
      self.widths.append(width)
      boxes.append((*pts.min(axis=0), *pts.max(axis=0)))
    self.boxes = np.array(boxes, float).reshape(-1, 4)

  def near(self, line: np.ndarray, width: float = CORRIDOR_WIDTH, tolerance: float = ROAD_TOLERANCE,
           budget: int = MAX_ROAD_POINTS) -> list[tuple[np.ndarray, float]]:
    """The parts of the roads within `width` m of a polyline, simplified, nearest roads first within `budget` points."""
    line = densify(line, CORRIDOR_STEP)
    lo, hi = line.min(axis=0) - width, line.max(axis=0) + width
    b = self.boxes
    cand = np.flatnonzero((b[:, 0] <= hi[0]) & (b[:, 2] >= lo[0]) & (b[:, 1] <= hi[1]) & (b[:, 3] >= lo[1]))
    found = []
    for k in cand:
      pts = densify(self.points[k], CORRIDOR_STEP)
      d = np.full(len(pts), np.inf)
      for chunk in range(0, len(line), 256):
        sub = line[chunk:chunk + 256]
        d = np.minimum(d, np.hypot(pts[:, None, 0] - sub[None, :, 0], pts[:, None, 1] - sub[None, :, 1]).min(axis=1))
      inside = d <= width
      if not inside.any():
        continue
      # each run of points inside, with a point either side so the road reaches the edge
      edges = np.flatnonzero(np.diff(np.concatenate(([0], inside.astype(np.int8), [0]))))
      for a, z in zip(edges[::2], edges[1::2], strict=True):
        run = simplify(pts[max(a - 1, 0):min(z + 1, len(pts))], tolerance)
        if len(run) >= 2:
          found.append((float(d[a:z].min()), run, self.widths[k]))
    found.sort(key=lambda f: f[0])
    out, used = [], 0
    for _, run, w in found:
      if used + len(run) > budget:
        break
      out.append((run, w))
      used += len(run)
    return out


def route_message(route, roads: list[tuple[np.ndarray, float]], project: Callable = to_lat_lon):
  """A navRoute: the route simplified, and the roads near it; empty without a route."""
  msg = messaging.new_message("navRoute", valid=True)
  if route is None or len(route.points) < 2:
    return msg
  pts = simplify(np.asarray(route.points, float), ROUTE_TOLERANCE)
  _coordinates(msg.navRoute.init("coordinates", len(pts)), pts, project)
  for r, (line, width) in zip(msg.navRoute.init("roads", len(roads)), roads, strict=True):
    _coordinates(r.init("coordinates", len(line)), line, project)
    r.width = width
  return msg


def _coordinates(out, pts: np.ndarray, project: Callable):
  lat, lon = project(pts[:, 0], pts[:, 1])
  for c, a, o in zip(out, np.asarray(lat, float).tolist(), np.asarray(lon, float).tolist(), strict=True):
    c.latitude, c.longitude = a, o


class NavMessages:
  """Publishes navInstruction every INSTRUCTION_EVERY (valid=false without a route) and navRoute for the bridge's
  route. A new route goes out at once without its roads; a thread gathers them and builds the full navRoute, which is
  sent once ready and again every ROUTE_EVERY, so the bridge's loop never builds the big message itself.
  `slots_for(route)` gives the route's LaneSlots, or None."""
  def __init__(self, pm=None, project: Callable = to_lat_lon):
    self.pm = pm or messaging.PubMaster(["navInstruction", "navRoute"])
    self.project = project
    self.next_instruction = 0.0
    self.next_route = 0.0
    self.route = None  # the route of the navRoute last sent
    self.route_bytes = route_message(None, []).to_bytes()  # that navRoute
    self.ready: tuple | None = None  # from the thread: (route, m along it the roads are about, its navRoute or None once sent)
    self.road_index: Roads | None = None
    self.osm = None
    self.busy = False
    self.lock = threading.Lock()

  def update(self, route, v: float, osm=None, slots_for: Callable | None = None, now: float | None = None,
             pose: tuple | None = None) -> None:
    """`pose`: the car's position (map metres) and heading (deg clockwise from north), for fill_instruction."""
    now = time.monotonic() if now is None else now
    if now >= self.next_instruction:
      self.next_instruction = now + INSTRUCTION_EVERY
      self.pm.send("navInstruction", self.instruction(route, v, slots_for, pose))

    if route is not self.route:
      self.route_bytes = route_message(route, [], self.project).to_bytes()  # the route alone: a few hundred points
      self.next_route = 0.0
    with self.lock:
      ready = self.ready
      if ready is not None and ready[0] is route and ready[2] is not None:
        self.route_bytes, self.next_route = ready[2], 0.0
        self.ready = (ready[0], ready[1], None)
    if now >= self.next_route:
      self.route, self.next_route = route, now + ROUTE_EVERY
      self.pm.send("navRoute", self.route_bytes)
    if route is not None and osm is not None and not self.busy:
      if ready is None or ready[0] is not route or route.at - ready[1] >= CORRIDOR_AGAIN:
        self.busy = True
        threading.Thread(target=self._gather, args=(route, route.at, osm), daemon=True).start()

  def instruction(self, route, v: float, slots_for: Callable | None = None, pose: tuple | None = None):
    msg = messaging.new_message("navInstruction", valid=True)
    if route is None or len(route.points) < 2:
      return msg
    slots = None
    if slots_for is not None:
      try:
        slots = slots_for(route)
      except Exception as e:  # lanes on a map they can't read: the rest of the guidance carries on
        print(f"nav msgs: lane slots: {e!r}")
    try:
      guide = lane_guide(slots, route.at, v, route.lane() if route.on_road(ON_ROUTE) else None)
    except Exception as e:
      print(f"nav msgs: lanes: {e!r}")
      guide = NO_LANES
    fill_instruction(msg.navInstruction, route, route.at, v, self.project, guide, pose)
    return msg

  def _gather(self, route, at: float, osm):
    """On a thread: the roads near the route ahead of `at`, and the navRoute with them."""
    try:
      if self.road_index is None or self.osm is not osm:
        self.road_index, self.osm = Roads(osm), osm
      s = np.arange(max(at - CORRIDOR_BEHIND, 0.0), min(at + CORRIDOR_AHEAD, route.length), CORRIDOR_STEP)
      line = np.stack([np.interp(s, route.along, route.points[:, 0]), np.interp(s, route.along, route.points[:, 1])], axis=1)
      roads = self.road_index.near(line) if len(line) >= 2 else []
      out = route_message(route, roads, self.project).to_bytes()
    except Exception as e:  # only the UI's map: it mustn't stop the bridge
      print(f"nav msgs: roads: {e!r}")
      out = None
    with self.lock:
      self.ready = (route, at, out)
    self.busy = False
