"""Map matching: which road the car is on, and which way along it, from GNSS fixes and the car's own speed and yaw rate.

A fix alone puts the car a few metres off its road (a comma 3X's position wanders by about that much), which on a dual
carriageway is as near the other carriageway as its own, and at junctions and beside parallel roads is near several.
The matcher scores each road near a fix by
- distance from the fix,
- how well the road's direction of travel agrees with the car's heading (one-way roads only have their own way), the
  heading being the gyro's yaw rate integrated and pulled towards the GNSS course while the car moves, and
- whether the roads lead there from the last fix's candidates in about the distance the car's odometry says it drove,
a hidden Markov model decoded online with Viterbi's forward pass. Between fixes the match moves on along the roads by
odometry. A road the roads don't lead to from the last match is taken only on several fixes' evidence.

Positions are planar metres (x east, y north); headings are degrees counterclockwise from north (navd's convention),
GNSS bearings clockwise from north.
"""
import heapq
import math
from typing import NamedTuple

import numpy as np

ROADS = frozenset({'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'service',
                   'living_street', 'track', 'road', 'motorway_link', 'trunk_link', 'primary_link', 'secondary_link',
                   'tertiary_link'})
CLASS_PRIOR = {'service': -0.5, 'track': -1.0}  # log: drives, car parks and tracks are less likely than the road beside

SIGMA = 5.0  # m: a fix's error, per axis (a 3X's: a few metres of slowly wandering bias and about 1 m of noise)
RADIUS = 30.0  # m from a fix to the roads it may be on
CANDIDATES = 12  # roads kept per fix, nearest first
BETA = 8.0  # m: how far the distance along the roads between two fixes' candidates may differ from odometry's
UNREACHABLE = 12.0  # log: a candidate the roads don't lead to from any of the last fix's within reach
REACH = 60.0  # m beyond odometry's distance (and SIGMA's allowance) the roads are searched between fixes
ROAD_SHAPE = 20.0  # deg: how far a road's own line may turn from the car's heading along it (bends, junction mouths)
HEADING_CAP = 120.0  # deg: a road heading more than this from the car scores no worse (it may be reversing)
COURSE_SPEED = 3.0  # m/s: below, the GNSS course says little; the heading is mostly the gyro's
STOPPED = 1.0  # m/s: below, the course is meaningless (whatever accuracy the receiver claims)
REVERSING = 0.5  # m/s backwards
COURSE_SIGMA = 3.0  # deg: the GNSS course's error at speed, where the fix gives none
GYRO_DRIFT = 1.0  # deg/sqrt(s): how fast the integrated heading loses accuracy (a random walk)
SLOW_COURSE_SIGMA, SLOW_COURSE_MIN = 45.0, 10.0  # deg: a slow car's course error, unless the fix gives one; and at least
SURE_HEADING = 30.0  # deg: a heading known this well tells the road's direction
TURN_ONTO = 60.0  # deg: between fixes the match moves only onto a road this near the car's heading


def wrap(deg):
  return (deg + 180.0) % 360.0 - 180.0


def oneway_of(tags: dict) -> int:
  """1 along the way, -1 against it, 0 both ways (OSM's oneway, and what motorways and roundabouts imply)."""
  v = tags.get('oneway')
  if v in ('yes', 'true', '1'):
    return 1
  if v in ('-1', 'reverse'):
    return -1
  if v is None and (tags.get('highway') == 'motorway' or tags.get('junction') in ('roundabout', 'circular')):
    return 1
  return 0


class RoadGraph:
  """The map's roads as directed edges, one per way segment and direction the road may be driven."""
  CELL = 25.0  # m

  def __init__(self, xy: np.ndarray, ways: list[tuple[int, dict, list[int]]]):
    """xy: [N, 2] node positions. ways: (way id, tags, node indices into xy) for each road."""
    self.xy = np.asarray(xy, float)
    u, v, way, prior = [], [], [], []
    for wid, tags, refs in ways:
      if tags.get('highway') not in ROADS:
        continue
      d = oneway_of(tags)
      p = CLASS_PRIOR.get(tags.get('highway'), 0.0)
      for a, b in zip(refs, refs[1:], strict=False):
        if a == b:
          continue
        for s, e, ok in ((a, b, d >= 0), (b, a, d <= 0)):
          if ok:
            u.append(s)
            v.append(e)
            way.append(wid)
            prior.append(p)
    self.u, self.v = np.array(u, int), np.array(v, int)
    self.way, self.prior = np.array(way, np.int64), np.array(prior, float)
    self.a, self.b = self.xy[self.u], self.xy[self.v]
    ab = self.b - self.a
    self.length = np.maximum(np.hypot(ab[:, 0], ab[:, 1]), 1e-6)
    self.heading = np.degrees(np.arctan2(-ab[:, 0], ab[:, 1]))
    self.out: dict[int, list[int]] = {}
    for k, s in enumerate(self.u):
      self.out.setdefault(int(s), []).append(k)
    twin = {(int(s), int(e)): k for k, (s, e) in enumerate(zip(self.u, self.v, strict=True))}
    self.reverse = np.array([twin.get((int(e), int(s)), -1) for s, e in zip(self.u, self.v, strict=True)], int)
    self.cells: dict[tuple[int, int], list[int]] = {}
    lo = np.floor(np.minimum(self.a, self.b) / self.CELL).astype(int)
    hi = np.floor(np.maximum(self.a, self.b) / self.CELL).astype(int)
    for k in range(len(self.u)):
      for cx in range(lo[k, 0], hi[k, 0] + 1):
        for cy in range(lo[k, 1], hi[k, 1] + 1):
          self.cells.setdefault((cx, cy), []).append(k)

  @classmethod
  def from_osm(cls, osm) -> 'RoadGraph':
    """From a map with `xy` per node, `data.index(node ids)` and `ways` {id: (tags, node ids)} (an OsmLanes)."""
    ways = [(wid, tags, [int(i) for i in osm.data.index(refs)]) for wid, (tags, refs) in osm.ways.items()]
    return cls(osm.xy, [(w, t, r) for w, t, r in ways if min(r, default=-1) >= 0])

  def near(self, p, radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The edges within `radius` of p: (edges, t along each from 0 to 1, distance), nearest first."""
    c0 = np.floor((np.asarray(p, float) - radius) / self.CELL).astype(int)
    c1 = np.floor((np.asarray(p, float) + radius) / self.CELL).astype(int)
    found: set[int] = set()
    for cx in range(c0[0], c1[0] + 1):
      for cy in range(c0[1], c1[1] + 1):
        found.update(self.cells.get((cx, cy), ()))
    if not found:
      return np.zeros(0, int), np.zeros(0), np.zeros(0)
    e = np.fromiter(found, int, len(found))
    t, d = self.project(e, p)
    keep = d <= radius
    e, t, d = e[keep], t[keep], d[keep]
    order = np.argsort(d, kind='stable')
    return e[order], t[order], d[order]

  def project(self, e, p) -> tuple[np.ndarray, np.ndarray]:
    a, ab = self.a[e], self.b[e] - self.a[e]
    t = np.clip(((p[0] - a[..., 0]) * ab[..., 0] + (p[1] - a[..., 1]) * ab[..., 1]) / self.length[e] ** 2, 0.0, 1.0)
    d = np.hypot(a[..., 0] + ab[..., 0] * t - p[0], a[..., 1] + ab[..., 1] * t - p[1])
    return t, d

  def point(self, e: int, t: float) -> np.ndarray:
    return self.a[e] + (self.b[e] - self.a[e]) * t

  def distances(self, e: int, t: float, limit: float) -> dict[int, float]:
    """m along the roads from t along edge e to each node reachable within `limit`, not turning back onto e's twin."""
    start = int(self.v[e])
    best = {start: (1.0 - t) * float(self.length[e])}
    heap = [(best[start], start, int(self.reverse[e]))]
    while heap:
      d, n, banned = heapq.heappop(heap)
      if d > best.get(n, math.inf) or d > limit:
        continue
      for k in self.out.get(n, ()):
        if k == banned:
          continue
        m, dk = int(self.v[k]), d + float(self.length[k])
        if dk < best.get(m, math.inf) and dk <= limit:
          best[m] = dk
          heapq.heappush(heap, (dk, m, int(self.reverse[k])))
    return best


class Match(NamedTuple):
  edge: int  # RoadGraph edge
  t: float  # along it, 0 to 1
  point: np.ndarray  # [x, y] m on the road
  heading: float  # deg, the road's direction of travel there
  way: int  # the map's way id
  off: float  # m from the last fix to its candidate on this road
  confidence: float  # the match's share of the candidates' probability at the last fix
  sure: bool  # the car's heading was known, so the road's direction is too (not just its line)


class MapMatcher:
  """predict() every step with the car's speed and yaw rate; update() with each GNSS fix; `match` is the road now."""

  def __init__(self, graph: RoadGraph, sigma: float = SIGMA, radius: float = RADIUS):
    self.g = graph
    self.sigma, self.radius = sigma, radius
    self.heading: float | None = None  # deg, the car's, fused
    self.heading_var = 0.0  # deg^2
    self.yaw_rate = 0.0  # deg/s, the last step's
    self.v = 0.0
    self.odo = 0.0  # m driven since the last fix
    self.edges = np.zeros(0, int)  # the last fix's candidates
    self.ts = np.zeros(0)
    self.scores = np.zeros(0)  # log, the best path's to each
    self.match: Match | None = None

  def predict(self, dt: float, v: float, yaw_rate: float):
    """dt s since the last call; v m/s (carState.vEgo, negative reversing); yaw_rate rad/s, left positive."""
    self.v, self.yaw_rate = v, math.degrees(yaw_rate)
    if self.heading is not None:
      self.heading = (self.heading + self.yaw_rate * dt) % 360.0
      self.heading_var += GYRO_DRIFT ** 2 * dt
    step = abs(v) * dt
    self.odo += step
    if self.match is not None and step > 0.0:
      self.match = self._advance(self.match, step)

  def update(self, x: float, y: float, bearing: float | None = None, bearing_accuracy: float | None = None,
             speed: float | None = None, age: float = 0.0) -> Match | None:
    """A GNSS fix: position, course (deg clockwise from north) and its accuracy, speed, and its age (s since it was
    measured: the match comes forward from it by odometry)."""
    speed = self.v if speed is None else speed
    if bearing is not None:
      course = -bearing + self.yaw_rate * age + (180.0 if self.reversing else 0.0)  # the course is the way it moves
      if speed >= COURSE_SPEED:
        self._course(course, max(bearing_accuracy or COURSE_SIGMA, 1.0) ** 2)
      elif speed >= STOPPED and (bearing_accuracy is not None or self.heading is None):
        # a slow car's course counts for as little as the fix says (a receiver's course means nothing standing still)
        self._course(course, max(bearing_accuracy or SLOW_COURSE_SIGMA, SLOW_COURSE_MIN) ** 2)
    moved = self.odo
    p = np.array([x, y], float)
    e, t, d = self.g.near(p, self.radius)
    e, t, d = e[:CANDIDATES], t[:CANDIDATES], d[:CANDIDATES]
    if not len(e):
      self.edges, self.ts, self.scores, self.match = e, t, d, None
      self.odo = 0.0
      return None
    back = abs(self.v) * age  # where the car was at the fix, along its way
    emission = -0.5 * (d / self.sigma) ** 2 + self.g.prior[e]
    if self.heading is not None:
      then = self.travel - self.yaw_rate * age
      dh = np.minimum(np.abs(wrap(self.g.heading[e] - then)), HEADING_CAP)
      emission -= 0.5 * dh ** 2 / (self.heading_var + ROAD_SHAPE ** 2)
    if len(self.edges):
      scores = emission + self._transitions(e, t, max(moved - back, 0.0))
    else:
      scores = emission
    scores -= scores.max()
    self.edges, self.ts, self.scores = e, t, scores
    self.odo = back
    k = int(np.argmax(scores))
    w = np.exp(scores)
    m = Match(int(e[k]), float(t[k]), self.g.point(int(e[k]), float(t[k])), float(self.g.heading[e[k]]),
              int(self.g.way[e[k]]), float(d[k]), float(w[k] / w.sum()), self.heading_known)
    self.match = self._advance(m, back) if back > 0.0 else m
    return self.match

  @property
  def reversing(self) -> bool:
    return self.v < -REVERSING

  @property
  def travel(self) -> float:
    """deg, the way the car moves along the road: its heading, or behind it while reversing."""
    return (self.heading + (180.0 if self.reversing else 0.0)) % 360.0

  @property
  def heading_known(self) -> bool:
    return self.heading is not None and self.heading_var < SURE_HEADING ** 2

  def _course(self, heading: float, var: float):
    if self.heading is None:
      self.heading, self.heading_var = heading % 360.0, var
      return
    k = self.heading_var / (self.heading_var + var)
    self.heading = (self.heading + k * wrap(heading - self.heading)) % 360.0
    self.heading_var *= 1.0 - k

  def _transitions(self, e: np.ndarray, t: np.ndarray, odo: float) -> np.ndarray:
    """The best of the last fix's scores plus the log chance of getting from each to each new candidate."""
    limit = odo + REACH + 3.0 * self.sigma
    g = self.g
    best = np.full(len(e), -np.inf)
    for pe, pt, ps in zip(self.edges, self.ts, self.scores, strict=True):
      if ps < best.max() - UNREACHABLE:
        continue  # can't beat an unreachable jump from a better one
      reach = g.distances(int(pe), float(pt), limit)
      dn = np.full(len(e), np.inf)
      same = e == pe
      dn[same] = (t[same] - pt) * g.length[pe]
      for j in np.flatnonzero(~same):
        n = int(g.u[e[j]])
        if n in reach:
          dn[j] = reach[n] + t[j] * g.length[e[j]]
      cost = np.where(np.isfinite(dn), np.maximum(-np.abs(dn - odo) / BETA, -UNREACHABLE), -UNREACHABLE)
      best = np.maximum(best, ps + cost)
    return best

  def _advance(self, m: Match, step: float) -> Match:
    """The match moved `step` m on along the roads, at a node onto the road nearest the car's heading."""
    g = self.g
    e, t = m.edge, m.t + step / g.length[m.edge]
    while t > 1.0:
      left = (t - 1.0) * g.length[e]
      nxt = [k for k in g.out.get(int(g.v[e]), ()) if k != g.reverse[e]]
      if self.heading is not None:
        travel = self.travel
        nxt = [k for k in nxt if abs(wrap(g.heading[k] - travel)) < TURN_ONTO]
        nxt.sort(key=lambda k: abs(wrap(g.heading[k] - travel)))
      elif len(nxt) != 1:
        nxt = []
      if not nxt:
        t = 1.0
        break
      e, t = nxt[0], left / g.length[nxt[0]]
    return m._replace(edge=int(e), t=float(t), point=g.point(int(e), float(t)), heading=float(g.heading[e]),
                      way=int(g.way[e]))
