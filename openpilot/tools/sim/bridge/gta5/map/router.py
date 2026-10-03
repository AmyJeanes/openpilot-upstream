"""Routes over a Valhalla server (map/README.md), as a car's navigation would, and follows the car along the route."""
import json
import threading
import urllib.request

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game, to_lat_lon

HEADING_TOLERANCE = 45.0  # deg: start on a road heading the car's way, not the opposite carriageway


def decode_polyline(encoded: str, precision: int = 6) -> list[tuple[float, float]]:
  """Google's encoded polyline, as Valhalla returns shapes: (lat, lon) pairs."""
  out, index, lat, lon = [], 0, 0, 0
  while index < len(encoded):
    for axis in range(2):
      shift = result = 0
      while True:
        b = ord(encoded[index]) - 63
        index += 1
        result |= (b & 0x1f) << shift
        shift += 5
        if b < 0x20:
          break
      delta = ~(result >> 1) if result & 1 else result >> 1
      if axis == 0:
        lat += delta
      else:
        lon += delta
    out.append((lat / 10 ** precision, lon / 10 ** precision))
  return out


class Route:
  """A route's points (game metres), and where along it the car is."""
  def __init__(self, points: np.ndarray, maneuvers: list[dict]):
    self.points = points
    self.along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(self.points, axis=0).T))))
    self.maneuvers = maneuvers
    self.at = 0.0  # m along the route to the car

  @property
  def length(self) -> float:
    return float(self.along[-1])

  def locate(self, pos: np.ndarray, search: float = 60.0) -> float:
    """Moves the car to its nearest point on the route from a little behind where it was, so where the route passes
    back near itself the car stays on the part it's driving; returns its distance off the route."""
    lo = max(0, int(np.searchsorted(self.along, self.at - 10.0)) - 1)
    hi = min(len(self.points) - 1, int(np.searchsorted(self.along, self.at + search)) + 1)
    a, b = self.points[lo:hi], self.points[lo + 1:hi + 1]
    if len(a) == 0:
      return float(np.hypot(*(pos - self.points[-1])))
    ab = b - a
    t = np.clip(np.einsum('ij,ij->i', pos - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-9), 0.0, 1.0)
    near = a + ab * t[:, None]
    d = np.hypot(*(near - pos).T)
    i = int(np.argmin(d))
    self.at = float(self.along[lo + i] + t[i] * np.hypot(*ab[i]))
    return float(d[i])

  def ahead(self, distance: float, step: float) -> np.ndarray:
    """Points every `step` m from the car for `distance` m, or to the route's end."""
    s = np.arange(self.at, min(self.at + distance, self.length) + 1e-6, step)
    return np.stack([np.interp(s, self.along, self.points[:, 0]), np.interp(s, self.along, self.points[:, 1])], axis=1)


class Router:
  def __init__(self, url: str, timeout: float = 2.0):
    self.url = url.rstrip('/')
    self.timeout = timeout

  def route(self, pos: np.ndarray, bearing: float, dest: np.ndarray) -> Route:
    """From the car, setting off the way it faces (bearing clockwise from north), to dest."""
    def location(p, **kw):
      lat, lon = to_lat_lon(float(p[0]), float(p[1]))
      return {'lat': lat, 'lon': lon, **kw}
    request = {
      'locations': [location(pos, heading=round(bearing) % 360, heading_tolerance=HEADING_TOLERANCE), location(dest)],
      'costing': 'auto',
      'directions_options': {'units': 'kilometers'},
    }
    req = urllib.request.Request(f"{self.url}/route", data=json.dumps(request).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=self.timeout) as r:
      trip = json.loads(r.read())['trip']
    points, maneuvers = [], []
    for leg in trip['legs']:
      base = len(points)
      points += [to_game(lat, lon) for lat, lon in decode_polyline(leg['shape'])]
      maneuvers += [{**m, 'begin_shape_index': m['begin_shape_index'] + base} for m in leg['maneuvers']]
    return Route(np.array(points, dtype=float), maneuvers)


class Navigator:
  """Keeps a route from the car to a destination, routing again when the destination moves or the car leaves the
  route. Routing runs on its own thread, so update() never waits on the server."""
  OFF_ROUTE = 15.0  # m from the route
  OFF_FOR = 1.5  # s
  RETRY = 3.0  # s between attempts

  def __init__(self, router: Router):
    self.router = router
    self.route: Route | None = None
    self.dest: np.ndarray | None = None
    self.off_since: float | None = None
    self.next_try = 0.0
    self.busy = False
    self.lock = threading.Lock()

  def update(self, pos: np.ndarray, bearing: float, dest: np.ndarray | None, now: float) -> Route | None:
    if dest is None:
      self.dest, self.route = None, None
      return None
    if self.dest is None or np.hypot(*(dest - self.dest)) > 1.0:
      self.dest, self.route, self.next_try = dest, None, 0.0
    with self.lock:
      route = self.route
    if route is not None:
      off = route.locate(pos)
      self.off_since = None if off < self.OFF_ROUTE else (self.off_since or now)
      if self.off_since is not None and now - self.off_since > self.OFF_FOR and now >= self.next_try:
        self._start(pos, bearing, dest, now)
    elif now >= self.next_try:
      self._start(pos, bearing, dest, now)
    return route

  def _start(self, pos: np.ndarray, bearing: float, dest: np.ndarray, now: float):
    if self.busy:
      return
    self.busy, self.next_try = True, now + self.RETRY
    threading.Thread(target=self._route, args=(pos.copy(), bearing, dest.copy()), daemon=True).start()

  def _route(self, pos: np.ndarray, bearing: float, dest: np.ndarray):
    try:
      route = self.router.route(pos, bearing, dest)
      route.locate(pos)
      with self.lock:
        if self.dest is not None and np.hypot(*(dest - self.dest)) <= 1.0:
          self.route, self.off_since = route, None
    except (OSError, ValueError, KeyError) as e:
      print(f"router: {e}")
    finally:
      self.busy = False
