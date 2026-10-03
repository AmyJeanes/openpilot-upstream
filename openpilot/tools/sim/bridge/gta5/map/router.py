"""Routes over a Valhalla server (map/README.md), as a car's navigation would, and follows the car along the route."""
import json
import math
import threading
import urllib.error
import urllib.request

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game, to_lat_lon
from openpilot.tools.sim.bridge.gta5.map.paths import CAR_HEIGHT, Link, Paths, wrap

SERVICE_PENALTY, SERVICE_FACTOR = 120, 4.0  # s onto a service road, and its cost over a road's
HEADING_TOLERANCE = 45.0  # deg: start on a road heading the car's way, not the opposite carriageway
SNAP_TOLERANCE = 60.0  # deg, the roads leaving the next node of the road found at the car's height and heading
FORK_SPREAD = 40.0  # deg either side of straight on, up to nav's turns: a road branching off within this is a fork
OTHER_LEVEL = 3.0  # m above or below the route: the car is on another road, passing over or under it
WRONG_WAY = 100.0  # deg from the route's direction: the car isn't driving that part of it
FORK_BEHIND = 50.0  # m: nav keeps to a fork's side a little past it


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


class Fork:
  def __init__(self, along: float, side: str, lanes: int, lanes_in: int, keep: bool):
    self.along = along  # m along the route
    self.side = side  # the branch the route takes, "left" or "right"
    self.lanes, self.lanes_in = lanes, lanes_in  # on that branch, and on the road before
    # whether it's a fork in the road, rather than GTA splitting a road's lanes before a junction, where the car just
    # keeps to its lane
    self.keep = keep


class Route:
  """A route's points (game metres), and where along it the car is. With GTA's road data, also the road's height and
  lanes along it, and the roads that fork off it; with the map's speed limits, those."""
  def __init__(self, points: np.ndarray, maneuvers: list[dict], paths: Paths | None = None, limits: np.ndarray | None = None):
    self.points = points
    self.along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(self.points, axis=0).T))))
    self.maneuvers = maneuvers
    self.at = 0.0  # m along the route to the car
    self.seg = 0  # the route segment the car is on
    self.right = 0.0  # m right of the route's line
    self.off = 0.0  # m off the route
    n = len(points)
    self.z = np.full(n, np.nan)
    self.links: list[Link | None] = [None] * max(n - 1, 0)
    self.limits = limits if limits is not None else np.zeros(max(n - 1, 0))  # m/s per segment, 0 unknown
    self.forks: list[Fork] = []
    if paths is not None and n >= 2:
      self._add_paths(paths)
    self.limit_list = [round(float(v), 2) for v in self.limits]
    self.lane_counts = [link.lanes if link is not None else 0 for link in self.links]

  def _add_paths(self, paths: Paths):
    pts = self.points
    nodes = paths.route_nodes(pts)
    # the route starts and ends part way along a link: the node before its start and after its end
    if nodes[0] is None and nodes[1] is not None:
      nodes[0] = self._link_end(paths, nodes[1], pts[1] - pts[0], before=True)
      start = nodes[0]
    else:
      start = None
    if nodes[-1] is None and nodes[-2] is not None:
      nodes[-1] = self._link_end(paths, nodes[-2], pts[-1] - pts[-2], before=False)
      end = nodes[-1]
    else:
      end = None
    for k, i in enumerate(nodes):
      if i is not None:
        self.z[k] = paths.z[i]
    for k in range(len(pts) - 1):
      a, b = nodes[k], nodes[k + 1]
      if a is not None and b is not None:
        self.links[k] = paths.links.get((a, b))
    for k, i in ((0, start), (len(pts) - 1, end)):
      if i is not None:  # the end's height along its link
        j = nodes[1] if k == 0 else nodes[-2]
        span = np.hypot(*(paths.xy[j] - paths.xy[i]))
        t = np.hypot(*(pts[k] - paths.xy[i])) / span if span > 0 else 0.0
        self.z[k] = paths.z[i] + (paths.z[j] - paths.z[i]) * t
    known = ~np.isnan(self.z)
    if known.any() and not known.all():
      self.z = np.interp(self.along, self.along[known], self.z[known])
    for k in range(1, len(pts) - 1):
      prev, i, nxt = nodes[k - 1], nodes[k], nodes[k + 1]
      if prev is None or i is None or nxt is None or prev == i or i == nxt:
        continue
      fork = self._fork(paths, prev, i, nxt, float(self.along[k]))
      if fork is not None:
        self.forks.append(fork)

  @staticmethod
  def _link_end(paths: Paths, i: int, direction: np.ndarray, before: bool) -> int | None:
    """The node linked to i along `direction` of travel: behind i, or ahead of it."""
    h = math.degrees(math.atan2(-direction[0], direction[1]))
    ends = paths.into.get(i, ()) if before else paths.out.get(i, ())
    best, best_off = None, 10.0
    for j in ends:
      off = abs(wrap((paths.link_heading(j, i) if before else paths.link_heading(i, j)) - h))
      if off < best_off:
        best, best_off = j, off
    return best

  @staticmethod
  def _fork(paths: Paths, prev: int, i: int, nxt: int, along: float) -> Fork | None:
    ours = wrap(paths.link_heading(i, nxt) - paths.link_heading(prev, i))
    if abs(ours) >= FORK_SPREAD:
      return None
    others = paths.forks(prev, i, nxt, FORK_SPREAD)
    if not others:
      return None
    if all(rel > ours for rel, _ in others):
      side = "right"
    elif all(rel < ours for rel, _ in others):
      side = "left"
    else:
      return None  # the middle of three: straight on
    link_in, link = paths.links.get((prev, i)), paths.links.get((i, nxt))
    if link_in is None or link is None:
      return None
    lane_split = link.no_nav and all(paths.links[(i, j)].no_nav for _, j in others)
    return Fork(along, side, link.lanes, link_in.lanes, bool(paths.flags[i][2] & 64) or not lane_split)

  @property
  def length(self) -> float:
    return float(self.along[-1])

  def locate(self, pos: np.ndarray, z: float | None = None, heading: float | None = None, search: float = 60.0) -> float:
    """Moves the car to its nearest point on the route from a little behind where it was, so where the route passes
    back near itself the car stays on the part it's driving; returns its distance off the route. With the car's height
    and heading (game degrees), parts of the route above or below it, or the other way, don't count."""
    lo = max(0, int(np.searchsorted(self.along, self.at - 10.0)) - 1)
    hi = min(len(self.points) - 1, int(np.searchsorted(self.along, self.at + search)) + 1)
    a, b = self.points[lo:hi], self.points[lo + 1:hi + 1]
    if len(a) == 0:
      self.off = float(np.hypot(*(pos - self.points[-1])))
      return self.off
    ab = b - a
    t = np.clip(np.einsum('ij,ij->i', pos - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-9), 0.0, 1.0)
    near = a + ab * t[:, None]
    d = np.hypot(*(near - pos).T)
    other = np.zeros(len(d), dtype=bool)
    if z is not None and not np.isnan(self.z[lo:hi + 1]).any():
      road_z = self.z[lo:hi] + (self.z[lo + 1:hi + 1] - self.z[lo:hi]) * t
      other |= np.abs(z - CAR_HEIGHT - road_z) > OTHER_LEVEL
    if heading is not None:
      seg_heading = np.degrees(np.arctan2(-ab[:, 0], ab[:, 1]))
      other |= np.abs((heading - seg_heading + 180) % 360 - 180) > WRONG_WAY
    if other.all():
      self.off = float(d.min()) + 100.0  # off the route, though it's near
      return self.off
    d = np.where(other, np.inf, d)
    i = int(np.argmin(d))
    self.seg = lo + i
    self.at = float(self.along[lo + i] + t[i] * np.hypot(*ab[i]))
    length = max(float(np.hypot(*ab[i])), 1e-6)
    self.right = float((pos[0] - a[i, 0]) * ab[i, 1] - (pos[1] - a[i, 1]) * ab[i, 0]) / length
    self.off = float(d[i])
    return self.off

  def ahead(self, distance: float, step: float) -> np.ndarray:
    """Points every `step` m from the car for `distance` m, or to the route's end."""
    s = np.arange(self.at, min(self.at + distance, self.length) + 1e-6, step)
    return np.stack([np.interp(s, self.along, self.points[:, 0]), np.interp(s, self.along, self.points[:, 1])], axis=1)

  def lane(self) -> list[int] | None:
    """The car's lane, as the plugin reports it: [i from the left, of n], i negative in the oncoming lanes."""
    link = self.links[self.seg] if self.seg < len(self.links) else None
    if link is None or not link.lanes:
      return None
    return [link.lane(self.right), link.lanes]

  def changes(self, values, distance: float) -> list[list[float]]:
    """[[m ahead, value], ...] where a per-segment value changes within `distance` m, from the car's segment on."""
    out: list[list[float]] = []
    last = None
    for k in range(self.seg, len(values)):
      d = max(0.0, float(self.along[k]) - self.at)
      if d > distance:
        break
      if values[k] != last:
        out.append([round(d, 1), values[k]])
        last = values[k]
    return out

  def info(self, distance: float) -> dict:
    """What nav uses of the route ahead, beyond its points, within `distance` m."""
    return {
      "routeEnd": round(self.length - self.at, 1),
      "forks": [[round(f.along - self.at, 1), f.side, f.lanes, f.lanes_in, f.keep] for f in self.forks
                if -FORK_BEHIND < f.along - self.at < distance],
      "limits": self.changes(self.limit_list, distance),
      "laneCounts": self.changes(self.lane_counts, distance),
    }


class Router:
  def __init__(self, url: str, timeout: float = 2.0, paths: Paths | None = None):
    self.url = url.rstrip('/')
    self.timeout = timeout
    self.paths = paths

  def _post(self, action: str, request: dict) -> dict:
    req = urllib.request.Request(f"{self.url}/{action}", data=json.dumps(request).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=self.timeout) as r:
      return json.loads(r.read())

  def route(self, pos: np.ndarray, bearing: float, dest: np.ndarray, z: float | None = None) -> Route:
    """From the car, setting off the way it faces (bearing clockwise from north), to dest. Given the car's height,
    from the road it's on, not one passing over or under it."""
    def location(p, **kw):
      lat, lon = to_lat_lon(float(p[0]), float(p[1]))
      return {'lat': lat, 'lon': lon, **kw}
    start = location(pos, heading=round(bearing) % 360, heading_tolerance=HEADING_TOLERANCE)
    snapped = self.paths.snap(pos, z, -bearing) if self.paths is not None and z is not None else None
    if snapped is not None:
      # from the next node of the road at the car's height, as the roads leaving a node are all at its height, where a
      # point on the road could find another passing over or under it nearer; the route then starts back at the car
      start = location(self.paths.xy[snapped[2]], heading=round(-snapped[1]) % 360, heading_tolerance=SNAP_TOLERANCE,
                       node_snap_tolerance=1.0)
    request = {
      'locations': [start, location(dest)],
      'costing': 'auto',
      # car parks, alleys and drives (GTA's nodes off for traffic or without GPS), which the model doesn't take for roads
      'costing_options': {'auto': {'service_penalty': SERVICE_PENALTY, 'service_factor': SERVICE_FACTOR}},
      'directions_options': {'units': 'kilometers'},
    }
    trip = self._post('route', request)['trip']
    points, maneuvers, limits = [], [], []
    for leg in trip['legs']:
      base = len(points)
      leg_points = [to_game(lat, lon) for lat, lon in decode_polyline(leg['shape'])]
      points += leg_points
      maneuvers += [{**m, 'begin_shape_index': m['begin_shape_index'] + base} for m in leg['maneuvers']]
      limits.append(self._limits(leg['shape'], len(leg_points)))
      if base:
        limits[-2] = np.append(limits[-2], 0.0)  # the segment joining the legs
    limits = np.concatenate(limits) if limits else np.zeros(0)
    if snapped is not None and points and np.hypot(*(np.array(points[0]) - self.paths.xy[snapped[2]])) < 1.5:
      points.insert(0, tuple(snapped[0]))
      maneuvers = [{**m, 'begin_shape_index': m['begin_shape_index'] + 1} for m in maneuvers]
      limits = np.concatenate((limits[:1], limits))
    return Route(np.array(points, dtype=float), maneuvers, self.paths, limits)

  def _limits(self, shape: str, n: int) -> np.ndarray:
    """The map's speed limit (m/s, 0 where it has none) along each segment of a route's shape."""
    out = np.zeros(max(n - 1, 0))
    try:
      trace = self._post('trace_attributes', {
        'encoded_polyline': shape, 'shape_match': 'walk_or_snap', 'costing': 'auto',
        'filters': {'attributes': ['edge.speed_limit', 'edge.begin_shape_index', 'edge.end_shape_index', 'matched.edge_index'],
                    'action': 'include'},
      })
    except urllib.error.HTTPError as e:
      print(f"router: no speed limits: {e}: {e.read()[:200]!r}")
      return out
    except (OSError, ValueError) as e:
      print(f"router: no speed limits: {e}")
      return out
    edges = trace.get('edges', [])
    limits = [e.get('speed_limit') or 0 for e in edges]
    limits = [v / 3.6 if 0 < v < 250 else 0.0 for v in limits]  # km/h; Valhalla says 255 for unlimited
    if 'matched_points' in trace:  # map matched: each point's edge
      for k, m in enumerate(trace['matched_points'][:len(out)]):
        e = m.get('edge_index')
        out[k] = limits[e] if e is not None and e < len(edges) else 0.0
    else:  # the route's own edges, along its shape
      for e, v in zip(edges, limits, strict=True):
        out[e.get('begin_shape_index', 0):e.get('end_shape_index', 0)] = v
    return out


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

  def update(self, pos: np.ndarray, bearing: float, dest: np.ndarray | None, now: float, z: float | None = None) -> Route | None:
    if dest is None:
      self.dest, self.route = None, None
      return None
    if self.dest is None or np.hypot(*(dest - self.dest)) > 1.0:
      self.dest, self.route, self.next_try = dest, None, 0.0
    with self.lock:
      route = self.route
    if route is not None:
      off = route.locate(pos, z, -bearing)
      self.off_since = None if off < self.OFF_ROUTE else (self.off_since or now)
      if self.off_since is not None and now - self.off_since > self.OFF_FOR and now >= self.next_try:
        self._start(pos, bearing, dest, now, z)
    elif now >= self.next_try:
      self._start(pos, bearing, dest, now, z)
    return route

  def _start(self, pos: np.ndarray, bearing: float, dest: np.ndarray, now: float, z: float | None):
    if self.busy:
      return
    self.busy, self.next_try = True, now + self.RETRY
    threading.Thread(target=self._route, args=(pos.copy(), bearing, dest.copy(), z), daemon=True).start()

  def _route(self, pos: np.ndarray, bearing: float, dest: np.ndarray, z: float | None):
    try:
      route = self.router.route(pos, bearing, dest, z)
      route.locate(pos, z, -bearing)
      with self.lock:
        if self.dest is not None and np.hypot(*(dest - self.dest)) <= 1.0:
          self.route, self.off_since = route, None
    except (OSError, ValueError, KeyError) as e:
      print(f"router: {e}")
    finally:
      self.busy = False
