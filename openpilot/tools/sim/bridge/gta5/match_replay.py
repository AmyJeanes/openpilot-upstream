#!/usr/bin/env python3
"""Replays recorded drives through navd's map matching offline, as a 3X's GNSS would have seen them, and scores the
matches against the game's own pose.

  match_replay.py extract --out POSES.npz [--log bridge.jsonl]   the car's true pose at 10 Hz from the bridge's log
  match_replay.py run POSES.npz --map DIR --out RESULT.json [--profile qcom3x|stress] [--figs DIR]

Each drive (the log cut where the car stood still for a minute, jumped, paused or left the car) gets GNSS fixes from
gta5_gnss.Gnss (seeded per drive) and carState-like speed and yaw rate, and these go to three matchers:
- nearest: the road nearest the fix, either way on a two-way road by the GNSS course (no heading otherwise);
- n1: the nearest road heading within Route.locate's WRONG_WAY of the GNSS course: N1's heading gate, at every road;
- n2: navd's MapMatcher (selfdrive/navd/map_match.py).
`stress` adds a city's worse GNSS to qcom3x: more bias and noise, a course that means nothing when stopped, and now
and then a multipath jump of 10-25 m.

Ground truth: the true position and heading (and height, by the map's `ele`) snapped to the map's directed edges at
each fix: an edge within its road's half width (+2.5 m) heading within 50 deg of the car, at the car's height; where
none is (junction interiors, car parks, off road), the fix isn't scored. A match is
- right: the truth's edge, one along the roads within NEAR m of it, or a parallel one through the car's position;
- wrong way: heading over 90 deg from the truth's edge (the other carriageway, or the other way on a two-way road);
- wrong road: anything else.
Turns: where the truth's edge turns over 45 deg and holds; latency is from the first fix on the new road to the first
of two fixes in a row matched to it. Reroutes: the drive's own path (the truth's edges joined along the roads) is the
route, and Navigator's rule (over OFF_ROUTE m from it for over OFF_FOR s) runs on each matcher's pose, n1's being the
fix and its course as N1 would give Route.locate: every trigger where the car is truly on the route is false.
"""
import argparse
import heapq
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

from openpilot.selfdrive.navd.map_match import MapMatcher, RoadGraph, wrap
from openpilot.tools.sim.bridge.gta5.gta5_gnss import PROFILES, Gnss
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
from openpilot.tools.sim.bridge.gta5.map.paths import CAR_HEIGHT
from openpilot.tools.sim.bridge.gta5.map.router import WRONG_WAY, Navigator, Route

BRIDGE_LOG = os.path.expanduser("~/gta5test/bridge.jsonl")
STEP = 0.1  # s between poses, at least (the bridge logs at about 7 Hz)
STILL_SPLIT = 60.0  # s standing still that ends a drive
MIN_DRIVE = 30.0  # s moving, at least
TRUTH_RADIUS, TRUTH_SIDE, TRUTH_HEADING, TRUTH_HEIGHT = 12.0, 2.5, 50.0, 3.0
NEAR = 30.0  # m along the roads: a match this near the truth's edge is on the same road (one edge's lag at a node)
PARALLEL, PARALLEL_HEADING = 7.0, 30.0  # m, deg: a way this near the car, heading its way, is the same road (GTA splits
# a road's lane groups into ways side by side)
TURN, TURN_SETTLED, TURN_GAP, TURN_MAX = 45.0, 20.0, 8, 15  # deg, deg, fixes between turns, fixes to wait for a match
EPISODE_GAP = 2  # fixes: wrong runs this near are one episode
DEVIATE = 30.0  # deg: a road leaving the drive's path at least this far from the car's way, for a route the car leaves
DEVIATE_WAIT = 30.0  # s after leaving a route to wait for the reroute
DEVIATE_GONE = 80.0  # m the car must get from where it left the route within DEVIATE_WAIT, for the test to count
DUAL = 30.0  # m: a one-way road with another one-way road this near heading the other way is a dual carriageway
METHODS = ("nearest", "n1", "n2")


# *** extract ***

def extract(log: str, out: str):
  """The log's car poses at STEP s: mono, x, y, z, heading, v, yaw rate, paused, in vehicle, engaged."""
  cols: dict[str, list] = {k: [] for k in ("mono", "x", "y", "z", "heading", "v", "yaw", "paused", "inveh", "en")}
  size, last, engaged = os.path.getsize(log), -1e9, False
  with open(log, "rb") as f:
    while f.tell() < size and (line := f.readline()):
      if not line.startswith(b'{"mono": '):
        continue
      if b'"control"' in line[:60]:
        engaged = bool(json.loads(line)["control"].get("active"))
        continue
      m = float(line[9:line.index(b",")])
      if b'"state"' not in line[:60] or m - last < STEP:
        continue
      s = json.loads(line)["state"]
      if not s.get("pos"):
        continue
      last = m
      for k, v in (("mono", m), ("x", s["pos"][0]), ("y", s["pos"][1]), ("z", s["pos"][2]), ("heading", s.get("heading", 0.0)),
                   ("v", s.get("vEgo", 0.0)), ("yaw", s.get("yawRate", 0.0)), ("paused", bool(s.get("paused"))),
                   ("inveh", bool(s.get("inVehicle", True))), ("en", engaged)):
        cols[k].append(v)
  np.savez_compressed(out, **{k: np.array(v) for k, v in cols.items()})


def drives(d) -> list[slice]:
  """The poses cut into drives."""
  m, x, y, v = d["mono"], d["x"], d["y"], d["v"]
  cut = np.zeros(len(m), bool)
  cut[1:] = (np.diff(m) > 1.0) | (np.hypot(np.diff(x), np.diff(y)) > 30.0) | d["paused"][1:] | ~d["inveh"][1:]
  still = 0.0
  for k in range(1, len(m)):
    still = still + (m[k] - m[k - 1]) if abs(v[k]) < 0.3 else 0.0
    if still > STILL_SPLIT:
      cut[k], still = True, 0.0
  starts = np.flatnonzero(cut).tolist()
  out, dt = [], np.diff(m, append=m[-1])
  for a, b in zip([0] + starts, starts + [len(m)], strict=True):
    if dt[a:b - 1][np.abs(v[a:b - 1]) > 1.0].sum() >= MIN_DRIVE:
      out.append(slice(a, b))
  return out


# *** the map and truth ***

class Truth:
  def __init__(self, osm, graph: RoadGraph):
    self.g = graph
    ids = osm.ids
    tags = osm.data.node_tags
    self.ele = np.array([float(tags.get(int(i), {}).get("ele", "nan")) for i in ids])
    self.half = np.full(len(graph.u), 3.5)
    for k, w in enumerate(graph.way):
      t = osm.ways[int(w)][0]
      try:
        width = float(t.get("width", "7"))
      except ValueError:
        width = 7.0
      self.half[k] = width / 2.0
    self.into: dict[int, list[int]] = {}
    for k, n in enumerate(graph.v):
      self.into.setdefault(int(n), []).append(k)
    self._near: dict[int, set[int]] = {}
    self._dual: dict[int, bool] = {}

  def dual(self, e: int) -> bool:
    """Whether edge e is a carriageway of a dual carriageway: one-way, beside a one-way road heading the other way."""
    if e not in self._dual:
      g = self.g
      mid = (g.a[e] + g.b[e]) / 2
      near, _, _ = g.near(mid, DUAL)
      self._dual[e] = g.reverse[e] < 0 and any(g.reverse[k] < 0 and abs(wrap(g.heading[k] - g.heading[e])) > 150.0
                                               for k in near if k not in self.near(e))
    return self._dual[e]

  def snap(self, x: float, y: float, z: float, heading: float) -> int:
    """The edge the car truly is on, -1 for none."""
    g = self.g
    e, t, d = g.near((x, y), TRUTH_RADIUS)
    if not len(e):
      return -1
    dh = np.abs(wrap(g.heading[e] - heading))
    road_z = self.ele[g.u[e]] + (self.ele[g.v[e]] - self.ele[g.u[e]]) * t
    ok = (dh < TRUTH_HEADING) & (d <= self.half[e] + TRUTH_SIDE) & ~(np.abs(z - CAR_HEIGHT - road_z) > TRUTH_HEIGHT)
    if not ok.any():
      return -1
    score = np.where(ok, d + 0.05 * dh, np.inf)
    return int(e[int(np.argmin(score))])

  def near(self, e: int) -> set[int]:
    """The edges within NEAR m of edge e along the roads, ahead or behind."""
    if e not in self._near:
      g, out = self.g, {e}
      for start, ahead in ((int(g.v[e]), True), (int(g.u[e]), False)):
        frontier = [(start, 0.0)]
        while frontier:
          n, dist = frontier.pop()
          for k in (g.out.get(n, ()) if ahead else self.into.get(n, ())):
            if k not in out and dist < NEAR:
              out.add(k)
              frontier.append((int(g.v[k]) if ahead else int(g.u[k]), dist + float(g.length[k])))
      self._near[e] = out
    return self._near[e]

  def judge(self, m: int | None, truth: int, x: float, y: float) -> str:
    """right, wrong_way, wrong_road or none (no match)."""
    if m is None or m < 0:
      return "none"
    g = self.g
    dh = abs(wrap(g.heading[m] - g.heading[truth]))
    if dh > 90.0:
      return "wrong_way"
    if m in self.near(truth):
      return "right"
    _, d = g.project(np.array([m]), (x, y))
    return "right" if d[0] < PARALLEL and dh < PARALLEL_HEADING else "wrong_road"

  def on_road(self, m: int | None, truth: int) -> bool:
    """Strictly on the truth's road: near it along the roads, and heading its way."""
    g = self.g
    return m is not None and m >= 0 and m in self.near(truth) and abs(wrap(g.heading[m] - g.heading[truth])) < PARALLEL_HEADING


def path_chains(g: RoadGraph, edges: list[int], limit: float = 150.0) -> list[list[int]]:
  """The drive's path: its truth edges in order, joined along the roads; cut where they don't join."""
  out, chain = [], []
  for e in edges:
    if e < 0 or (chain and e == chain[-1]):
      continue
    if chain and int(g.v[chain[-1]]) != int(g.u[e]):
      between = _join(g, int(g.v[chain[-1]]), int(g.u[e]), limit)
      if between is None:
        out.append(chain)
        chain = []
      else:
        nodes = [int(g.v[chain[-1]])] + between
        for a, b in zip(nodes, nodes[1:], strict=False):
          chain.append(min((k for k in g.out[a] if int(g.v[k]) == b), key=lambda k: g.length[k]))
    chain.append(e)
  if chain:
    out.append(chain)
  return out


def chain_points(g: RoadGraph, chain: list[int]) -> np.ndarray:
  return np.vstack([g.a[chain[0]]] + [g.b[e] for e in chain])


def route_path(g: RoadGraph, edges: list[int]) -> list[np.ndarray]:
  """The drive's path as routes, one per chain of path_chains."""
  return [chain_points(g, c) for c in path_chains(g, edges)]


def straight_on(g: RoadGraph, e: int, length: float) -> list[int]:
  """Edges on from edge e, each the one nearest straight on, for `length` m."""
  out, run = [], 0.0
  while run < length:
    nxt = [k for k in g.out.get(int(g.v[e]), ()) if k != g.reverse[e] and k not in out]
    if not nxt:
      break
    e = min(nxt, key=lambda k: abs(wrap(g.heading[k] - g.heading[e])))
    out.append(e)
    run += float(g.length[e])
  return out


def deviations(g: RoadGraph, chain: list[int], before: float = 250.0, after: float = 300.0) -> list[tuple[int, int, np.ndarray]]:
  """Routes the drive leaves: its path to a node where the car took one road and the route another (DEVIATE deg or
  more from the car's), then straight on. Each: (index in chain of the route's first edge, of the edge the car took
  where it leaves the route, the route's points)."""
  out, along, last_at = [], np.cumsum([g.length[e] for e in chain]), -1e9
  for i in range(1, len(chain)):
    if along[i - 1] < before or along[i - 1] - last_at < 2 * after:
      continue
    took, came = chain[i], chain[i - 1]
    other = [k for k in g.out.get(int(g.u[took]), ()) if k not in (took, g.reverse[came])
             and abs(wrap(g.heading[k] - g.heading[took])) >= DEVIATE]
    if not other:
      continue
    alt = min(other, key=lambda k: abs(wrap(g.heading[k] - g.heading[came])))
    lo = int(np.searchsorted(along, along[i - 1] - before))
    route = chain[lo:i] + [alt] + straight_on(g, alt, after)
    out.append((lo, i, chain_points(g, route)))
    last_at = along[i - 1]
  return out


def _join(g: RoadGraph, start: int, goal: int, limit: float) -> list[int] | None:
  """The nodes after start up to goal along the shortest way within `limit` m, None if there's none."""
  best, prev, heap = {start: 0.0}, {}, [(0.0, start)]
  while heap:
    d, n = heapq.heappop(heap)
    if n == goal:
      nodes = []
      while n != start:
        nodes.append(n)
        n = prev[n]
      return nodes[::-1]
    if d > best.get(n, math.inf):
      continue
    for k in g.out.get(n, ()):
      m, dk = int(g.v[k]), d + float(g.length[k])
      if dk <= limit and dk < best.get(m, math.inf):
        best[m], prev[m] = dk, n
        heapq.heappush(heap, (dk, m))
  return None


# *** GNSS ***

class Stress:
  """qcom3x's fixes made worse as in a city: more bias and noise, a stopped car's course random, multipath jumps."""
  def __init__(self, seed: int):
    self.rng = np.random.default_rng(seed + 7919)
    self.bias = np.zeros(2)

  def __call__(self, xy: np.ndarray, bearing: float, accuracy: float, speed: float, dt: float) -> tuple[np.ndarray, float, float]:
    a = math.exp(-dt / 30.0)
    self.bias = self.bias * a + self.rng.normal(0.0, 4.0, 2) * math.sqrt(1.0 - a * a)
    xy = xy + self.bias + self.rng.normal(0.0, 1.5, 2)
    if self.rng.random() < 0.03:
      ang = self.rng.uniform(0.0, 2 * math.pi)
      xy = xy + self.rng.uniform(10.0, 25.0) * np.array([math.cos(ang), math.sin(ang)])
    if speed < 1.0:
      bearing, accuracy = self.rng.uniform(0.0, 360.0), 180.0
    return xy, bearing, accuracy


# *** the matchers ***

def nearest(g: RoadGraph, xy: np.ndarray, bearing: float | None, gate: float | None) -> int | None:
  e, _, d = g.near(xy, 30.0)
  if not len(e):
    return None
  heading = None if bearing is None else -bearing
  if gate is not None and heading is not None:
    ok = np.abs(wrap(g.heading[e] - heading)) <= gate
    e, d = e[ok], d[ok]
    if not len(e):
      return None
  k = int(e[0])
  twin = int(g.reverse[k])
  if heading is not None and twin >= 0 and twin in e and abs(wrap(g.heading[twin] - heading)) < abs(wrap(g.heading[k] - heading)):
    k = twin
  return k


class Rerouter:
  """Navigator's off-route rule on one matcher's pose, over the drive's own path."""
  def __init__(self, routes: list[np.ndarray]):
    self.routes = [Route(r, []) for r in routes]
    self.off_since: float | None = None
    self.next_try = 0.0
    self.triggers: list[float] = []
    self.false = 0  # triggers while the car was truly on the route
    self.false_moving = 0  # of them, while it moved

  def route_for(self, xy) -> Route | None:
    best, best_d = None, 25.0
    for r in self.routes:
      d = float(np.min(np.hypot(*(r.points - xy).T)))
      if d < best_d:
        best, best_d = r, d
    return best

  def update(self, now: float, route: Route | None, xy, heading: float | None, true_off: bool, v: float = 0.0):
    if route is None:
      self.off_since = None
      return
    off = route.locate(np.asarray(xy, float), None, heading)
    self.off_since = None if off < Navigator.OFF_ROUTE else (self.off_since if self.off_since is not None else now)
    if self.off_since is not None and now - self.off_since > Navigator.OFF_FOR and now >= self.next_try:
      self.triggers.append(now)
      if not true_off:
        self.false += 1
        self.false_moving += v > 1.0
      self.next_try = now + Navigator.RETRY
      self.off_since = None
      return True
    return False


def replay_drive(d, sl: slice, g: RoadGraph, truth: Truth, profile: str, seed: int) -> dict:
  m, x, y, z = d["mono"][sl], d["x"][sl], d["y"][sl], d["z"][sl]
  heading, v, yaw = d["heading"][sl], d["v"][sl], d["yaw"][sl]
  base = "qcom3x" if profile == "stress" else profile
  gnss = Gnss(base, seed, pm=SimpleNamespace(send=lambda *a: None))
  stress = Stress(seed) if profile == "stress" else None
  latency = PROFILES[base].latency
  mm = MapMatcher(g)
  rows = []  # per fix: t, truth edge, true x, y, v, matched edges by method, fix xy, n2 point, n2 heading
  last_fix_t = None
  for k in range(len(m)):
    dt = m[k] - m[k - 1] if k else 0.0
    mm.predict(dt, float(v[k]), float(yaw[k]))
    h = math.radians(heading[k])
    msg = gnss.update(float(m[k]), float(x[k]), float(y[k]), float(z[k]),
                      float(-v[k] * math.sin(h)), float(v[k] * math.cos(h)))
    if msg is None:
      continue
    fix = getattr(msg, PROFILES[base].service)
    fx, fy = to_game(fix.latitude, fix.longitude)
    xy, bearing, accuracy = np.array([fx, fy]), float(fix.bearingDeg), float(fix.bearingAccuracyDeg)
    if stress is not None:
      xy, bearing, accuracy = stress(xy, bearing, accuracy, float(fix.speed), 1.0 if last_fix_t is None else m[k] - last_fix_t)
    last_fix_t = m[k]
    match = mm.update(float(xy[0]), float(xy[1]), bearing, accuracy, float(fix.speed), latency)
    rows.append({
      "t": float(m[k]), "truth": truth.snap(float(x[k]), float(y[k]), float(z[k]), float(heading[k])),
      "x": float(x[k]), "y": float(y[k]), "v": float(v[k]), "fix": xy, "bearing": bearing,
      "nearest": nearest(g, xy, bearing, None), "n1": nearest(g, xy, bearing, WRONG_WAY),
      "n2": None if match is None else match.edge, "n2_point": None if match is None else match.point,
      "n2_heading": None if match is None or not match.sure else match.heading, "n2_sure": match is not None and match.sure,
    })
  return score(rows, g, truth)


def _pose(method: str, r: dict) -> tuple[np.ndarray, float | None]:
  """Where Navigator puts the car for a matcher: n1 and nearest the fix and its course, n2 its match (Navigator.update
  with a match: its road's way only when it's sure of the direction)."""
  if method == "n2" and r["n2_point"] is not None:
    return r["n2_point"], r["n2_heading"]
  return r["fix"], -r["bearing"]


def score(rows: list[dict], g: RoadGraph, truth: Truth) -> dict:
  out: dict = {"fixes": len(rows), "moving": 0, "scored": 0}
  moving = [r for r in rows if r["v"] > 1.0]  # forwards: the truth's edge is the way the car faces
  scored = [r for r in moving if r["truth"] >= 0]
  out["moving"], out["scored"] = len(moving), len(scored)
  # turns: the truth's edge turns and holds
  turns = []
  last = -10 ** 9
  for i, r in enumerate(rows):
    if r["truth"] < 0 or i - last < TURN_GAP or i + 2 >= len(rows):
      continue
    h = g.heading[r["truth"]]
    before = [rows[j]["truth"] for j in range(max(0, i - 6), i - 1) if rows[j]["truth"] >= 0]
    after = [rows[j]["truth"] for j in (i + 1, i + 2)]
    if not before or min(after) < 0 or abs(wrap(g.heading[before[0]] - h)) < TURN:
      continue
    if any(abs(wrap(g.heading[a] - h)) > TURN_SETTLED for a in after) or rows[i]["v"] < 1.0:
      continue
    if truth.on_road(before[0], r["truth"]):
      continue
    turns.append(i)
    last = i
  for method in METHODS:
    verdicts = [truth.judge(r[method], r["truth"], r["x"], r["y"]) if r["truth"] >= 0 and r["v"] > 1.0 else None
                for r in rows]
    n = max(len(scored), 1)
    res = {k: sum(v == k for v in verdicts) / n for k in ("right", "wrong_way", "wrong_road", "none")}
    dual = [vd for vd, r in zip(verdicts, rows, strict=True) if vd is not None and truth.dual(r["truth"])]
    res["dual_fixes"] = len(dual)
    res["dual_wrong_way"] = sum(vd == "wrong_way" for vd in dual)
    res["dual_wrong_road"] = sum(vd == "wrong_road" for vd in dual)
    if method == "n2":
      res["wrong_way_unsure"] = sum(vd == "wrong_way" and not r["n2_sure"] for vd, r in zip(verdicts, rows, strict=True)) / n
    for kind in ("wrong_way", "wrong_road"):
      eps, run_end = [], -10 ** 9
      for i, vd in enumerate(verdicts):
        if vd == kind:
          if i - run_end > EPISODE_GAP + 1:
            eps.append([i, i])
          else:
            eps[-1][1] = i
          run_end = i
      res[f"{kind}_episodes"] = len(eps)
      res[f"{kind}_episode_s"] = [round(rows[b]["t"] - rows[a]["t"] + 1.0, 1) for a, b in eps]
      res[f"{kind}_at"] = [[round(rows[a]["x"], 1), round(rows[a]["y"], 1), round(rows[a]["t"], 1)] for a, b in eps]
    lat = []
    for i in turns:
      found = None
      for j in range(i, min(i + TURN_MAX, len(rows) - 1)):
        if rows[j]["truth"] >= 0 and truth.on_road(rows[j][method], rows[j]["truth"]) and \
           (rows[j + 1]["truth"] < 0 or truth.on_road(rows[j + 1][method], rows[j + 1]["truth"])):
          found = rows[j]["t"] - rows[i]["t"]
          break
      lat.append(found)
    res["turn_latency"] = lat
    out[method] = res
  out["turns"] = len(turns)
  # reroutes on the drive's own path
  routes = route_path(g, [r["truth"] for r in rows])
  out["route_km"] = round(sum(float(np.hypot(*np.diff(r, axis=0).T).sum()) for r in routes) / 1000.0, 2)
  checkers = {k: Rerouter(routes) for k in METHODS + ("truth",)}
  for r in rows:
    true_xy = np.array([r["x"], r["y"]])
    tr = checkers["truth"]
    route = tr.route_for(true_xy)
    true_heading = float(g.heading[r["truth"]]) if r["truth"] >= 0 else None
    true_off = route is None or (route.locate(true_xy, None, true_heading) > 10.0)
    for k in METHODS:
      rr = checkers[k]
      rt = None if route is None else rr.routes[tr.routes.index(route)]
      if rr.update(r["t"], rt, *_pose(k, r), true_off, r["v"]) and rt is not None:
        rt.at = max(route.at - 10.0, 0.0)  # rerouted: the new route from the car
  # routes the car leaves: how soon each matcher's pose says so
  first: dict[int, int] = {}
  for j, r in enumerate(rows):
    first.setdefault(r["truth"], j)
  for k in METHODS:
    out[k]["deviation_delay"], out[k]["deviation_early"] = [], 0
  out["deviations"] = 0
  for chain in path_chains(g, [r["truth"] for r in rows]):
    for lo, i, pts in deviations(g, chain):
      j0 = next((first[e] for e in chain[i:] if e in first), None)
      js = next((first[e] for e in chain[lo:i] if e in first), None)
      if j0 is None or js is None or js >= j0:
        continue
      later = [r for r in rows[j0:] if r["t"] <= rows[j0]["t"] + DEVIATE_WAIT]
      if float(np.hypot(later[-1]["x"] - rows[j0]["x"], later[-1]["y"] - rows[j0]["y"])) < DEVIATE_GONE:
        continue  # the drive ends (or the car stops) before it's clearly off the route
      out["deviations"] += 1
      t_dev = rows[j0]["t"]
      for k in METHODS:
        rr, got = Rerouter([pts]), None
        for r in rows[js:]:
          if r["t"] > t_dev + DEVIATE_WAIT:
            break
          if rr.update(r["t"], rr.routes[0], *_pose(k, r), False, r["v"]):
            if r["t"] < t_dev - 2.0:
              out[k]["deviation_early"] += 1
            else:
              got = round(r["t"] - t_dev, 2)
              break
        out[k]["deviation_delay"].append(got)
  for k in METHODS:
    out[k]["reroutes"] = len(checkers[k].triggers)
    out[k]["false_reroutes"] = checkers[k].false
    out[k]["false_reroutes_moving"] = checkers[k].false_moving
    out[k]["reroute_at"] = checkers[k].triggers
  out["rows"] = rows
  return out


# *** run ***

_G: dict = {}


def _load(map_dir: str):
  if not _G:
    from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes
    osm = OsmLanes.load(os.path.join(map_dir, "gta5.osm.pbf"), to_game)
    g = RoadGraph.from_osm(osm)
    _G.update(osm=osm, g=g, truth=Truth(osm, g))
  return _G


def _work(args):
  poses, map_dir, profile, i, a, b = args
  G = _load(map_dir)
  if G.get("poses_path") != poses:
    G["poses"], G["poses_path"] = dict(np.load(poses)), poses
  d = G["poses"]
  r = replay_drive(d, slice(a, b), G["g"], G["truth"], profile, seed=1000 + i)
  r["drive"], r["start"] = i, float(d["mono"][a])
  return r


def summarise(results: list[dict]) -> dict:
  hours = sum(r["moving"] for r in results) / 3600.0
  scored = sum(r["scored"] for r in results)
  out = {"drives": len(results), "moving_h": round(hours, 2), "scored_share": round(scored / max(sum(r["moving"] for r in results), 1), 3),
         "route_km": round(sum(r["route_km"] for r in results), 1), "turns": sum(r["turns"] for r in results)}
  for k in METHODS:
    s = {}
    for key in ("right", "wrong_way", "wrong_road", "none"):
      s[key] = round(sum(r[k][key] * r["scored"] for r in results) / max(scored, 1) * 100, 2)  # % of scored moving time
    for kind in ("wrong_way", "wrong_road"):
      eps = [e for r in results for e in r[k][f"{kind}_episode_s"]]
      s[f"{kind}_episodes"] = len(eps)
      s[f"{kind}_episodes_per_h"] = round(len(eps) / max(hours, 1e-9), 2)
      s[f"{kind}_episode_median_s"] = float(np.median(eps)) if eps else 0.0
      s[f"{kind}_episodes_over_3s"] = sum(e > 3.0 for e in eps)
    lat = [x for r in results for x in r[k]["turn_latency"]]
    got = [x for x in lat if x is not None]
    s["turn_latency_median_s"] = float(np.median(got)) if got else None
    s["turn_latency_p90_s"] = float(np.percentile(got, 90)) if got else None
    s["turns_unmatched_15s"] = sum(x is None for x in lat)
    s["reroutes"] = sum(r[k]["reroutes"] for r in results)
    s["false_reroutes"] = sum(r[k]["false_reroutes"] for r in results)
    s["false_reroutes_per_h"] = round(s["false_reroutes"] / max(hours, 1e-9), 2)
    s["false_reroutes_moving"] = sum(r[k]["false_reroutes_moving"] for r in results)
    dual = sum(r[k]["dual_fixes"] for r in results)
    s["dual_fixes"] = dual
    s["dual_wrong_way"] = round(sum(r[k]["dual_wrong_way"] for r in results) / max(dual, 1) * 100, 2)
    s["dual_wrong_road"] = round(sum(r[k]["dual_wrong_road"] for r in results) / max(dual, 1) * 100, 2)
    if k == "n2":
      s["wrong_way_unsure"] = round(sum(r[k]["wrong_way_unsure"] * r["scored"] for r in results) / max(scored, 1) * 100, 2)
    dev = [x for r in results for x in r[k]["deviation_delay"]]
    got = [x for x in dev if x is not None]
    s["deviations"] = len(dev)
    s["deviation_delay_median_s"] = float(np.median(got)) if got else None
    s["deviation_delay_p90_s"] = float(np.percentile(got, 90)) if got else None
    s["deviations_missed_30s"] = sum(x is None for x in dev)
    s["deviation_early"] = sum(r[k]["deviation_early"] for r in results)
    out[k] = s
  return out


def figures(results: list[dict], out_dir: str, g: RoadGraph, per_kind: int = 8):
  """The longest wrong-way episodes of n1 and n2, and wrong-road ones of n2: the roads (grey arrows, their direction of
  travel), the true path, the fixes and each matcher's matched edges."""
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  os.makedirs(out_dir, exist_ok=True)
  for method, kind in (("n1", "wrong_way"), ("n2", "wrong_way"), ("n2", "wrong_road")):
    eps = [(s, r, t) for r in results for (_, _, t), s in zip(r[method][f"{kind}_at"], r[method][f"{kind}_episode_s"], strict=True)]
    eps.sort(key=lambda e: -e[0])
    for n, (s, r, t) in enumerate(eps[:per_kind]):
      rows = [row for row in r["rows"] if t - 15.0 <= row["t"] <= t + s + 10.0]
      xy = np.array([[row["x"], row["y"]] for row in rows])
      c = xy.mean(axis=0)
      half = max(60.0, float(np.abs(xy - c).max()) + 20.0)
      fig, ax = plt.subplots(figsize=(8, 8))
      e, _, _ = g.near(c, half * 1.5)
      for j in e:
        ax.annotate("", g.b[j], g.a[j], arrowprops={"arrowstyle": "-|>", "color": "0.7", "lw": 0.7})
      ax.plot(xy[:, 0], xy[:, 1], "k-", lw=2.5, label="true path")
      bad = [row for row in rows if t <= row["t"] <= t + s]
      if bad:
        ax.plot([row["x"] for row in bad], [row["y"] for row in bad], "y-", lw=6, alpha=0.5, label=f"{method} {kind}")
      fx = np.array([row["fix"] for row in rows])
      ax.plot(fx[:, 0], fx[:, 1], "c.", ms=5, label="GNSS fixes")
      for meth, col, off in (("n1", "r", -0.6), ("n2", "b", 0.6)):
        first = True
        for row in rows:
          k = row[meth]
          if k is not None:
            ax.plot([g.a[k][0] + off, g.b[k][0] + off], [g.a[k][1] + off, g.b[k][1] + off], col + "-", lw=1.5, alpha=0.6,
                    label=f"{meth} matched edges" if first else None)
            first = False
      ax.set_xlim(c[0] - half, c[0] + half)
      ax.set_ylim(c[1] - half, c[1] + half)
      ax.set_aspect("equal")
      ax.legend(loc="upper right", fontsize=7)
      ax.set_title(f"{method} {kind} {s:.0f} s: drive {r['drive']} at t {t - r['start']:.0f} s, ({c[0]:.0f}, {c[1]:.0f})", fontsize=9)
      fig.savefig(os.path.join(out_dir, f"{method}_{kind}_{n:02d}_d{r['drive']}.png"), dpi=80)
      plt.close(fig)


def run(poses: str, map_dir: str, out: str, profile: str, figs: str | None, workers: int, only: int | None):
  d = np.load(poses)
  sls = drives(d)
  if only:
    sls = sls[:only]
  print(f"{len(sls)} drives", file=sys.stderr, flush=True)
  jobs = [(poses, map_dir, profile, i, s.start, s.stop) for i, s in enumerate(sls)]
  if workers > 1:
    import multiprocessing as mp
    with mp.get_context("fork").Pool(workers) as pool:
      results = []
      for k, r in enumerate(pool.imap_unordered(_work, jobs)):
        results.append(r)
        if k % 20 == 0:
          print(f"{k + 1}/{len(jobs)}", file=sys.stderr, flush=True)
  else:
    results = [_work(j) for j in jobs]
  results.sort(key=lambda r: r["drive"])
  summary = summarise(results)
  if figs:
    figures(results, figs, _load(map_dir)["g"])
  for r in results:
    r.pop("rows")
  with open(out, "w") as f:
    json.dump({"profile": profile, "summary": summary, "drives": results}, f, default=float)
  print(json.dumps(summary, indent=1))


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  e = sub.add_parser("extract")
  e.add_argument("--log", default=BRIDGE_LOG)
  e.add_argument("--out", required=True)
  r = sub.add_parser("run")
  r.add_argument("poses")
  r.add_argument("--map", required=True)
  r.add_argument("--out", required=True)
  r.add_argument("--profile", default="qcom3x", choices=sorted(PROFILES) + ["stress"])
  r.add_argument("--figs")
  r.add_argument("--workers", type=int, default=1)
  r.add_argument("--only", type=int, help="the first N drives")
  args = ap.parse_args()
  if args.cmd == "extract":
    extract(args.log, args.out)
  else:
    run(args.poses, args.map, args.out, args.profile, args.figs, args.workers, args.only)


if __name__ == "__main__":
  main()
