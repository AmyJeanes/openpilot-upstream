"""Offline harness for the map driver (gta5_mapdrive.py): a kinematic car whose curvature and acceleration follow the
commands with the game car's lag (first order, 0.15-0.3 s, after a frame's delay), stepped in 20 Hz game frames; made
routes (straight, curved, a junction turn with its stop line, a lane change for a turn) and real routes rebuilt from
recordings on the live map (routes.json's points and maneuvers, ~/gta5map_lanes' lanes, paths and stop lines)."""
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

from openpilot.tools.sim.bridge.gta5.gta5_mapdrive import LANE_W, MapDriver
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, FORWARD, Lane, RouteLanes, Section, Span
from openpilot.tools.sim.bridge.gta5.map.router import Route

FRAME = 0.05  # s, the game's frames reaching the bridge
SUB = 0.01  # s, the car's integration step
LAT_KI = 1.0  # 1/s, the plugin's lat_ki
LIVE_MAP = os.path.expanduser(os.getenv("GTA5_MAP", "~/gta5map_lanes"))
RECORDINGS = os.getenv("GTA5_RECORDINGS", "/mnt/e/gta5rec/data")
START, DEST, RIGHT, LEFT = 1, 4, 10, 15  # Valhalla maneuver types


class LagCar:
  """The game car under the plugin's control: curvature and acceleration lag the commands (first order), the commands
  acting `delay` s after they're sent; a stop is held (the plugin's handbrake). Game heading: deg counterclockwise
  from north. Imperfections for robustness checks: `gain` scales the curvature the car takes (the plugin's adaptive
  steering gain off), `noise` the pose's (m and deg, normal) and the yaw rate's (rad/s) noise, seeded. Its position
  (x, y) is its middle, `wheel_base` / 2 ahead of the rear axle, which moves along its heading, as the game reports it."""
  def __init__(self, x: float, y: float, heading: float, v: float = 0.0, tau_k: float = 0.22, tau_a: float = 0.22,
               delay: float = 0.05, gain: float = 1.0, noise: float = 0.0, seed: int = 0, wheel_base: float = 2.8):
    self.x, self.y, self.h, self.v = x, y, heading, v
    self.wheel_base = wheel_base
    self.k = self.a = self.lat_i = 0.0
    self.tau_k, self.tau_a, self.delay = tau_k, tau_a, delay
    self.gain, self.noise = gain, noise
    self.rng = np.random.default_rng(seed)
    self.queue: list[tuple[float, float, float]] = []  # (when it acts, curvature, accel)
    self.cmd = (0.0, 0.0)
    self.t = 0.0
    self.collisions = 0
    self.user: dict = {}
    self.blocked = False  # held still, as against a wall

  def state(self) -> dict:
    n = self.rng.normal(0.0, self.noise, 4) if self.noise else np.zeros(4)
    return {"pos": [self.x + n[0], self.y + n[1], 0.0], "vEgo": self.v, "heading": self.h + 5 * n[2],
            "yawRate": self.k * self.v + 0.1 * n[3], "aMeas": self.a,
            "t": int(self.t), "wheelBase": self.wheel_base, "collisions": self.collisions, "user": dict(self.user), "engagePresses": 0, "inVehicle": True,
            "ai": {"on": False}}

  def control(self, msg: dict | None):
    if msg is not None and msg.get("type") == "control" and msg.get("active"):
      self.queue.append((self.t + self.delay, float(msg["curvature"]), float(msg["accel"])))

  def advance(self, secs: float = FRAME):
    end = self.t + secs - 1e-9
    while self.t < end:
      while self.queue and self.queue[0][0] <= self.t + 1e-9:
        _, k, a = self.queue.pop(0)
        self.cmd = (k, a)
      k_cmd, a_cmd = self.cmd
      # the plugin's integral on the yaw rate error (lat_ki 1/s), which takes out a steering gain error in about a second
      if self.v > 3.0:
        self.lat_i = float(np.clip(self.lat_i + LAT_KI * (k_cmd - self.k) * SUB, -0.1, 0.1))
      else:
        self.lat_i *= max(0.0, 1.0 - SUB)
      self.k += (self.gain * (k_cmd + self.lat_i) - self.k) * min(SUB / self.tau_k, 1.0)
      self.a += (a_cmd - self.a) * min(SUB / self.tau_a, 1.0)
      if self.blocked:
        self.v, self.a = 0.0, 0.0
      elif self.v < 0.3 and a_cmd <= 0:
        self.v = 0.0  # the handbrake hold
      else:
        self.v = max(self.v + self.a * SUB, 0.0)
      b = self.wheel_base / 2
      hr = math.radians(self.h)
      rx, ry = self.x + math.sin(hr) * b, self.y - math.cos(hr) * b  # the rear axle
      self.h += math.degrees(self.k * self.v * SUB)
      hr = math.radians(self.h)
      rx += -math.sin(hr) * self.v * SUB
      ry += math.cos(hr) * self.v * SUB
      self.x, self.y = rx - math.sin(hr) * b, ry + math.cos(hr) * b
      self.t += SUB


# *** made routes ***

def section(ours: int = 2, back: int = 2, w: float = LANE_W, median: float = 0.0) -> Section:
  """A road's cross-section, the route's line between the directions."""
  mine = [Span(Lane(FORWARD, w), median / 2 + k * w, median / 2 + (k + 1) * w, 1) for k in range(ours)]
  onc = [Span(Lane(BACKWARD, w), -(median / 2 + (k + 1) * w), -(median / 2 + k * w), -1) for k in reversed(range(back))]
  spans = onc + mine
  return Section(spans, (spans[0].left, spans[-1].right))


def made_route(points, sec: Section | list, junctions=(), turns: dict | None = None, stops=(), road_class: str = "primary",
               limit: float = 0.0) -> Route:
  """A Route on made lanes: `sec` for every segment (or one per segment), junction nodes at the given point indices,
  turn maneuvers {point index: type} and stop lines [(m along, kind)]."""
  pts = np.asarray(points, float)
  n = len(pts)
  mans = [{"type": START, "begin_shape_index": 0}] + [{"type": t, "begin_shape_index": i, "street_names": ["Out St"]}
                                                     for i, t in sorted((turns or {}).items())] + [{"type": DEST, "begin_shape_index": n - 1}]
  r = Route(pts, mans)
  secs = sec if isinstance(sec, list) else [sec] * (n - 1)
  r.osm_lanes = True
  r.junctions = [float(r.along[i]) for i in junctions]
  r._lanes = RouteLanes(pts, secs, junctions=r.junctions)
  r.lane_counts = [s.lanes for s in secs]
  r.classes = [road_class] * (n - 1)
  r.limits = np.full(n - 1, limit)
  r.limit_list = [limit] * (n - 1)
  for along, kind in stops:
    r.stops.append(float(along))
    r.stop_kinds.append(kind)
  return r


def line(*legs: tuple[float, float], start=(0.0, 0.0), heading: float = 0.0, step: float = 10.0) -> np.ndarray:
  """Points along legs [(m, deg turned left at its start)] from `start`, heading `heading` (game deg): straight legs,
  turned sharply at the nodes between."""
  pts = [np.array(start, float)]
  h = heading
  for length, turn in legs:
    h += turn
    d = np.array([-math.sin(math.radians(h)), math.cos(math.radians(h))])
    k = max(int(round(length / step)), 1)
    for _ in range(k):
      pts.append(pts[-1] + d * length / k)
  return np.array(pts)


def arc(radius: float, angle: float, lead: float = 150.0, tail: float = 200.0, step: float = 5.0) -> np.ndarray:
  """Straight north `lead` m, a circular bend of `angle` deg (left positive) at `radius` m, straight on `tail` m."""
  pts = [np.array([0.0, y]) for y in np.arange(0.0, lead, step)]
  cx = -radius if angle > 0 else radius
  n = max(int(abs(math.radians(angle)) * radius / step), 2)
  for th in np.linspace(0.0, math.radians(abs(angle)), n):
    pts.append(np.array([cx + (radius * math.cos(th) if angle > 0 else -radius * math.cos(th)), lead + radius * math.sin(th)]))
  h = math.radians(angle)
  d = np.array([-math.sin(h), math.cos(h)])
  for k in range(1, int(tail / step) + 1):
    pts.append(pts[-1] + d * step)
  return np.array(pts)


def straight(length: float = 600.0, ours: int = 2, back: int = 2, **kw) -> Route:
  return made_route(line((length, 0.0)), section(ours, back), **kw)


def curve(radius: float, angle: float, ours: int = 2, back: int = 2, **kw) -> Route:
  return made_route(arc(radius, angle), section(ours, back), **kw)


def junction_turn(side: str = "left", lead: float = 200.0, tail: float = 150.0, ours: int = 2, back: int = 2,
                  stop: str | None = "stop", stop_back: float = 12.0) -> Route:
  """North `lead` m to a junction node, then a square turn and `tail` m on; a stop line `stop_back` m before it."""
  pts = line((lead, 0.0), (tail, 90.0 if side == "left" else -90.0))
  corner = int(round(lead / 10.0))
  return made_route(pts, section(ours, back), junctions=[corner], turns={corner: LEFT if side == "left" else RIGHT},
                    stops=[(lead - stop_back, stop)] if stop else [])


def start_pose(route: Route, lane: float, along: float = 0.0) -> tuple[float, float, float]:
  """(x, y, heading) in a lane of the route's road `along` m on."""
  k = max(int(np.searchsorted(route.along, along, side="right")) - 1, 0)
  k = min(k, len(route.points) - 2)
  a, b = route.points[k], route.points[k + 1]
  d = (b - a) / max(np.hypot(*(b - a)), 1e-9)
  p = a + d * (along - route.along[k])
  sec = route.section(k)
  off = sec.offset(lane) if sec is not None and sec.lanes else 0.0
  p = p + np.array([d[1], -d[0]]) * off
  return float(p[0]), float(p[1]), math.degrees(math.atan2(-d[0], d[1]))


# *** real routes ***

_LIVE: dict = {}


def live_map():
  """The live map's lanes, GTA paths and stop lines (read only), loaded once; None without it."""
  if "osm" not in _LIVE:
    _LIVE["osm"] = None
    if os.path.exists(os.path.join(LIVE_MAP, "gta5.osm.pbf")):
      from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
      from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes
      from openpilot.tools.sim.bridge.gta5.map.paths import Paths
      from openpilot.tools.sim.bridge.gta5.map.stop_lines import StopLines
      osm = OsmLanes.load(os.path.join(LIVE_MAP, "gta5.osm.pbf"), to_game)
      paths = Paths(os.path.join(LIVE_MAP, "paths.jsonl"))
      paths.index()
      _LIVE.update(osm=osm, paths=paths, stops=StopLines.of(osm, build=False))
  return _LIVE if _LIVE["osm"] is not None else None


def recorded_route(segment: str, index: int = 0) -> tuple[Route, tuple[float, float, float], float] | None:
  """A recording's route rebuilt on the live map, the car's pose (x, y, heading) and speed where it began; None
  without the map or the recording."""
  m = live_map()
  path = os.path.join(RECORDINGS, segment)
  if m is None or not os.path.exists(path):
    return None
  r = json.load(open(os.path.join(path, "routes.json")))["routes"][index]
  z = np.load(os.path.join(path, "gta5.npz"))
  cols = list(z["frame_columns"])
  f = z["frame"][min(int(r["frame"]), len(z["frame"]) - 1)]
  route = Route(np.array(r["points"], float), r["maneuvers"], m["paths"], None, m["osm"], m["stops"])
  pose = (float(f[cols.index("x")]), float(f[cols.index("y")]), float(f[cols.index("heading")]))
  route.locate(np.array(pose[:2]), None, pose[2])
  return route, pose, float(f[cols.index("v_ego")])


TRIP_CACHE = os.getenv("GTA5_TRIP_ROUTES", "/mnt/e/gta5_audit/mapdrive_trials/routes_cache.json")


def trip_route(spec: str, router_url: str = "http://127.0.0.1:8002") -> tuple[Route, tuple[float, float, float]] | None:
  """An e2e trip ('x,y,z,heading,lane>dx,dy') routed on the live map as the bridge routes it (points, maneuvers and
  speed limits kept in TRIP_CACHE, so it's asked of the router once), and the car's pose in its start lane (9: the
  rightmost); None without the map, or the router for a trip not yet cached."""
  m = live_map()
  if m is None:
    return None
  start, dest = spec.split(">")
  x, y, z, heading, *lane = (float(v) for v in start.split(","))
  dx, dy = (float(v) for v in dest.split(","))
  cache = json.load(open(TRIP_CACHE)) if os.path.exists(TRIP_CACHE) else {}
  if spec not in cache:
    from openpilot.tools.sim.bridge.gta5.map.router import Router
    try:
      router = Router(router_url, timeout=10.0, paths=m["paths"], osm=m["osm"], roads=m["osm"])
      r = router.route(np.array([x, y]), (-heading) % 360, np.array([dx, dy]), z)
    except OSError:
      return None
    cache[spec] = {"points": r.points.round(3).tolist(), "maneuvers": r.maneuvers, "limits": np.asarray(r.limits).round(3).tolist()}
    os.makedirs(os.path.dirname(TRIP_CACHE), exist_ok=True)
    json.dump(cache, open(TRIP_CACHE, "w"))
  c = cache[spec]
  route = Route(np.array(c["points"], float), c["maneuvers"], m["paths"], np.array(c["limits"], float), m["osm"], m["stops"])
  # where setup puts the car: in its lane where the spec's point is on the route's road, past the route's first jog
  route.locate(np.array([x, y]), None, heading, search=200.0)
  at = max(route.at, min(10.0, route.length / 4))
  k = max(int(np.searchsorted(route.along, at, side="right")) - 1, 0)
  sec = route.section(k)
  want = (lane[0] if lane else 0.0)
  if sec is not None and sec.lanes:
    want = min(want, sec.lanes - 1)
  px, py, ph = start_pose(route, want, at)
  route.at, route.seg = 0.0, 0
  return route, (px, py, ph)


# *** a trip ***

@dataclass
class Trip:
  md: MapDriver
  car: LagCar
  rows: list = field(default_factory=list)  # per frame: t, x, y, heading, v, yaw rate, kappa cmd, accel cmd, dev, s, phase
  events: list = field(default_factory=list)
  msgs: int = 0
  step_ms: list = field(default_factory=list)
  plan_ms: float = 0.0

  def col(self, k: int) -> np.ndarray:
    return np.array([r[k] for r in self.rows], float)

  @property
  def finished(self) -> str | None:
    return self.md.finished


COLS = ("t", "x", "y", "heading", "v", "yaw", "kappa", "accel", "dev", "s", "lane_off", "indicator")


def drive(route: Route, cfg: dict | None = None, pose=None, lane: float | None = None, v0: float = 0.0, seconds: float = 120.0,
          tau: float = 0.22, delay: float = 0.05, faults: dict | None = None, until_done: bool = True, lane_map=None,
          gain: float = 1.0, noise: float = 0.0) -> Trip:
  """Drives a route with the map driver on the lagged car. faults: {"collision": t, "push": (t, m right),
  "steer": t, "block": t, "surge": (path m, m/s more over 0.4 s)}. lane_map(route, state) gives the state's laneMap (None: none)."""
  import time
  if pose is None:
    route.at, route.seg = 0.0, 0  # a route driven before starts again at its start
    pose = start_pose(route, lane if lane is not None else 0.0, 0.0)
  car = LagCar(*pose, v=v0, tau_k=tau, tau_a=tau, delay=delay, gain=gain, noise=noise)
  route.at, route.seg = 0.0, 0
  route.locate(np.array(pose[:2], float), None, pose[2], search=route.length)
  md = MapDriver({"seed": 1, **(cfg or {})})
  trip = Trip(md, car)
  faults = faults or {}
  pushed = False
  while car.t < seconds:
    if "collision" in faults and car.t >= faults["collision"]:
      car.collisions = 1
    if "steer" in faults and car.t >= faults["steer"]:
      car.user = {"steer": 0.3}
    if "block" in faults and car.t >= faults["block"]:
      car.blocked = True
    if "push" in faults and not pushed and car.t >= faults["push"][0]:
      hr = math.radians(car.h)
      car.x += math.cos(hr) * faults["push"][1]
      car.y += math.sin(hr) * faults["push"][1]
      pushed = True
    if "surge" in faults and faults["surge"][0] <= md.s < faults["surge"][0] + 0.4 * max(car.v, 1.0):
      car.v += faults["surge"][1] * FRAME / 0.4  # an uncommanded surge, as the game car over a bump on a steep rise
    st = car.state()
    route.locate(np.array(st["pos"][:2]), None, st["heading"])
    st["laneMap"] = lane_map(route, st) if lane_map is not None else None
    t0 = time.perf_counter()
    msg = md.step(route, st, car.t, car.collisions)
    ms = (time.perf_counter() - t0) * 1000
    if md.plans and not trip.plan_ms:
      trip.plan_ms = ms
    else:
      trip.step_ms.append(ms)
    trip.events += md.take_events()
    car.control(msg)
    trip.msgs += msg is not None
    sec = route.section(route.seg)
    trip.rows.append((car.t, car.x, car.y, car.h, car.v, car.k * car.v, md.kappa, md.a, md.dev, md.s, route.right,
                      {"left": -1, "right": 1}.get(md.indicator, 0)))
    if until_done and md.finished is not None and (md.finished != "arrived" or car.v < 0.05):
      break
    car.advance(FRAME)
  return trip
