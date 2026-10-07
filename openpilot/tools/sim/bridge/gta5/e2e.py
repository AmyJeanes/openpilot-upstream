#!/usr/bin/env python3
"""End-to-end drive tests: random trips on the map, driven by openpilot with nav, recorded for analysis.

Needs the services running (svc.sh: valhalla, openpilot, bridge with GTA5_MAP, GTA5_ROUTER, GTA5_LOG and GTA5_DEBUG) and
the game in a car. Each trip puts the car at a start, sets the map view's destination, engages and watches until the car
arrives, disengages, gets stuck, leaves the road, reroutes too often or runs out of time. A JSON line per trip goes to
--out; a 2 Hz trace of each to <out dir>/traces/<out name>/<id>.jsonl (its path in the record as 'trace').

  e2e.py run --mode city --n 20 --seed 1             random trips in Los Santos (or --mode map, the whole map)
  e2e.py run --trip "x,y,z,heading>dx,dy"            one trip; heading in game degrees (counterclockwise from north)
  e2e.py run --replay city1-03                       a trip again, by its id in the results
  e2e.py pick --mode city --n 5 --seed 1             only print trips
  e2e.py summary [results.jsonl ...]                 outcomes, maneuvers, failures across runs
  e2e.py run --trips f --tune t.json                 nav's turn parameters for every trip (see below)
  e2e.py sweep --variants v.json --trips f --reps 2  each trip with each variant of nav's turn parameters
  e2e.py sweepsum [results.jsonl ...]                turns per variant: made, lane, landed, oncoming, collisions, stops, speed
  e2e.py sweepsum --backfill [...]                   and turns' landing lanes from the traces of results from before them
  e2e.py shorten --trips f [...] > short.txt          short trips for those trips' turns (see below)
  e2e.py sweep --short --trips short.txt --reps 8 ... run them: missed turns end the trip, shorter timeout

A short trip starts SHORT_BEFORE m before a turn, at a game road node (so the game faces the car along the road), and
ends SHORT_AFTER m past it, built from the trip's route in earlier results (or, with --results, every turn on those
results' routes). sweepsum reports each variant's rates with 95% ranges, and each variant against the first (or
"base") trip for trip, run for run.

Nav's turn parameters (navd planner.Tune) come from the file the bridge was started with as GTA5_NAVTUNE (svc.sh:
BRIDGE_EXTRA="GTA5_NAVTUNE=$HOME/gta5test/navtune.json"); --tune and sweep write it before each trip (E2E_NAVTUNE, the
same path by default). A variants file is {"name": {param: value, ...}, ...}; {} is the defaults.

Every maneuver on the route is scored: done, or missed (a reroute near it), with the car's lane, speed, set speed, the
nav's signal and the model's desire probabilities over the approach. Gas presses are only the test driver's, after the
car has stood still for a while (the model won't pull away from a stop by itself).

Time in the oncoming lanes is read two ways: by the game's lane reading (the route's and the plugin's) and by the map's
lane tags alone outside junctions (map/lane_match.py: also one-ways driven the wrong way and the other direction's turn
bays). safe() counts either (oncoming_any_*); oncoming_s / oncoming_max are the game's alone, as before. With lane tags,
a trip starts in the middle of its start lane by the map.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
import re
import socket
import subprocess
import sys
import time
import types
import urllib.request
from collections import Counter, defaultdict, deque

# E2E_STACK=sunnypilot drives the sunnypilot port's stack (~/gta5test/spsvc.sh) instead; E2E_SVC, E2E_STATE_LOG,
# E2E_BRIDGE_LOG and E2E_MODEL_FILE override its parts one by one
STACK = os.getenv("E2E_STACK", "openpilot")
_SP = STACK == "sunnypilot"
os.environ.setdefault("OPENPILOT_PREFIX", "spgta5" if _SP else "gta5")

import numpy as np

from openpilot.tools.sim.bridge.gta5.gta5_rx import DEBUG_PORT
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game, to_lat_lon
from openpilot.tools.sim.bridge.gta5.map.router import HEADING_TOLERANCE, decode_polyline

HOME = os.path.expanduser("~")
MAP_DIR = os.getenv("GTA5_MAP", f"{HOME}/gta5map")
VALHALLA = os.getenv("GTA5_ROUTER", "http://localhost:8002")
MAP_VIEW = "http://localhost:8793"
_LOGS = f"{HOME}/gta5test/sp" if _SP else f"{HOME}/gta5test"
STATE_LOG = os.getenv("E2E_STATE_LOG", f"{_LOGS}/bridge.jsonl")
BRIDGE_LOG = os.getenv("E2E_BRIDGE_LOG", f"{_LOGS}/bridge.log")
SVC = os.getenv("E2E_SVC", f"{HOME}/gta5test/{'spsvc.sh' if _SP else 'svc.sh'}")
MODEL_FILE = os.getenv("E2E_MODEL_FILE", f"{_LOGS}/model")
ENGAGEABLE_WAIT = 30.0  # s for openpilot to allow engaging after a teleport (locationd settling)
OUT_DIR = f"{HOME}/gta5test/e2e"
NAVTUNE = os.getenv("E2E_NAVTUNE", f"{HOME}/gta5test/navtune.json")
TUNE_ACK = 4.0  # s for the bridge to log the parameters it read
MODEL3 = "0x0040B009"  # Amy's Model 3 add-on

CITY = (-1500, 1500, -2500, 1000)  # x0, x1, y0, y1: Los Santos
TRIP_MIN, TRIP_MAX = 800.0, 3000.0  # m of route
STRAIGHT_MIN, STRAIGHT_MAX = 500.0, 2500.0  # m start to destination
# Valhalla maneuver types
TYPES = {1: "start", 2: "start right", 3: "start left", 4: "destination", 5: "destination right", 6: "destination left",
         7: "becomes", 8: "continue", 9: "slight right", 10: "right", 11: "sharp right", 12: "u-turn right",
         13: "u-turn left", 14: "sharp left", 15: "left", 16: "slight left", 17: "ramp straight", 18: "ramp right",
         19: "ramp left", 20: "exit right", 21: "exit left", 22: "stay straight", 23: "stay right", 24: "stay left",
         25: "merge", 26: "roundabout enter", 27: "roundabout exit", 37: "merge right", 38: "merge left"}
REAL = set(range(9, 12)) | set(range(14, 25))  # turns, ramps, exits and forks: what the car has to do something for
RAMP = set(range(17, 25))
LEFT = {3, 6, 13, 14, 15, 16, 19, 21, 24, 38}
RIGHT = {2, 5, 9, 10, 11, 12, 18, 20, 23, 37}

TICK = 0.1  # s
TRACE_SLACK = 300.0  # s after a result's trip its old-style trace (named by trip id alone) may have last been written
TRACE_EVERY = 0.5  # s
STOPPED = 0.3  # m/s
NUDGE_AFTER = 10.0  # s stopped, short of the destination: the test driver presses the gas
NUDGES = 3
# s of full throttle: a Model 3 pulls about 9 m/s^2, so this gets it rolling and openpilot takes it from there
START_GAS, NUDGE_GAS = 0.25, 0.3
STUCK_AFTER = 15.0  # s stopped after the last nudge
PINNED = 5.0  # m moved over all the nudges, less than which the car is against a wall or kerb
FIRST_MANEUVER = 150.0  # m: picked trips have no maneuver nearer the start
ARRIVED_WITHIN = 40.0  # m of the destination, disengaged
MAX_REROUTES = 5
REPEAT_WITHIN = 20.0  # m from the last reroute: the same one again
OFF_ROAD = 15.0  # m from any road link
OFF_ROAD_FOR = 5.0  # s
COLLISION_GAP = 1.0  # s between contacts that count as separate collisions
CRASH_WITHIN, CRASH_STOP = 3.0, 5.0  # s: stopped this soon after a collision, and for this long, is a crash
FALL = 5.0  # m dropped within a second: off a bridge or ramp
MISSED_NEAR = 150.0  # m from the next maneuver when the car leaves the route: that maneuver was missed
REACHED = 35.0  # m: and the car came at least this near it
DONE_PAST = 25.0  # m along the route past a maneuver, where the car must get to
DONE_WITHIN = 15.0  # m of that point
ARC_FROM = 10.0  # deg: in a turn's arc, from its heading before and the one after
STALE = 5.0  # s without game state or openpilot messages: restart that service
# short trips (shorten, run --short): one turn, its approach and the way out of it, to run more of them
SHORT_BEFORE = 150.0  # m along the road from the start to the turn
SHORT_AFTER = 120.0  # m on to the destination (the bridge arrives 20-40 m short of it, past where landing lanes are read)
SHORT_SLACK = 40.0  # m either way the start or destination may move to find a node for it
SHORT_MIN = 60.0  # m: the least road before the turn a start can make do with
SHORT_TIMEOUT = 120.0  # s
MISS_GRACE = 15.0  # s a short trip goes on after a missed maneuver (to see what the car does straight after), then ends
EDGE_ANGLE = 35.0  # deg from a turn's bearing in or out, for the links into and out of it
WALK_TURN = 45.0  # deg: the most a road bends from one link to the next, walking along it
SPAWN_STRAIGHT = 20.0  # deg between a start node's links: the game faces the car along the road there
SPAWN_ALONE = 4.0  # m: no other node this near a start (setup puts the car at the node nearest it: an overpass's, say)


# *** the map and trips ***

def load_paths():
  try:
    import osmium  # noqa: F401
  except ImportError:
    sys.modules["osmium"] = types.ModuleType("osmium")  # ynd_to_osm only needs it to write maps
  from openpilot.tools.sim.bridge.gta5.map import ynd_to_osm as ynd
  nodes, links, streets = ynd.load(os.path.join(MAP_DIR, "paths.jsonl"))
  es = ynd.edges(nodes, links)
  cross = ynd.crossovers(nodes, es)
  return ynd, nodes, streets, es, cross


class Map:
  """The game's road links: trip ends, and how far the car is from any road."""
  CELL = 50.0

  def __init__(self):
    ynd, nodes, streets, es, cross = load_paths()
    self.starts, self.dests = [], []
    segs = []
    for (a, b), (fwd, back, _) in es.items():
      na, nb = nodes[a], nodes[b]
      segs.append((na['x'], na['y'], nb['x'], nb['y']))
      if (a, b) in cross:
        continue
      kind = ynd.highway(nodes, a, b, fwd, back) if fwd else ynd.highway(nodes, b, a, back, fwd)
      length = math.hypot(nb['x'] - na['x'], nb['y'] - na['y'])
      if kind in ('service', 'track') or length < 15 or na['st'] not in streets or na['st'] != nb['st']:
        continue
      if (na['f'][2] | nb['f'][2]) & 4:  # junction nodes
        continue
      mx, my, mz = (na['x'] + nb['x']) / 2, (na['y'] + nb['y']) / 2, (na['z'] + nb['z']) / 2
      self.dests.append((mx, my, mz))
      if kind in ('motorway', 'trunk'):
        continue  # don't start from a standstill on a freeway
      # at a path node, so setup's nearest node is this one and faces this way (not the other carriageway's)
      for lanes, (p, q) in ((fwd, (na, nb)), (back, (nb, na))):
        if lanes:
          self.starts.append((p['x'], p['y'], p['z'], math.degrees(math.atan2(-(q['x'] - p['x']), q['y'] - p['y'])) % 360))
    self.grid = defaultdict(list)
    for s in segs:
      x0, x1, y0, y1 = min(s[0], s[2]), max(s[0], s[2]), min(s[1], s[3]), max(s[1], s[3])
      for i in range(int(x0 // self.CELL), int(x1 // self.CELL) + 1):
        for j in range(int(y0 // self.CELL), int(y1 // self.CELL) + 1):
          self.grid[(i, j)].append(s)
    self.grid = {k: np.array(v) for k, v in self.grid.items()}
    # junction and stop-line nodes, to tell excursions in a junction from those after it
    self.junctions = defaultdict(list)
    for n in nodes.values():
      if n['f'][2] & 4 or (n['f'][1] >> 3) in (15, 16):
        self.junctions[(int(n['x'] // self.CELL), int(n['y'] // self.CELL))].append((n['x'], n['y']))
    self.junctions = {k: np.array(v) for k, v in self.junctions.items()}
    # the links each way they can be driven, to walk along roads (short trips)
    self.nodes, self.streets, self.ynd = nodes, streets, ynd
    self.out, self.into, self.lanes = defaultdict(list), defaultdict(list), {}
    for (a, b), (fwd, back, _) in es.items():
      if (a, b) in cross:
        continue
      for p, q, n in ((a, b, fwd), (b, a, back)):
        if n:
          self.out[p].append(q)
          self.into[q].append(p)
          self.lanes[(p, q)] = n
    self.node_cells = defaultdict(list)
    for k, n in nodes.items():
      self.node_cells[(int(n['x'] // self.CELL), int(n['y'] // self.CELL))].append(k)
    self._lane_map: LaneMap | None | bool = False  # not loaded yet

  @property
  def lane_map(self) -> 'LaneMap | None':
    """The map's lanes (gta5.osm.pbf's tags), None where it has none."""
    if self._lane_map is False:
      self._lane_map = LaneMap.load(MAP_DIR)
    return self._lane_map

  def in_junction(self, x: float, y: float) -> bool:
    i, j = int(x // self.CELL), int(y // self.CELL)
    cand = [self.junctions[k] for k in ((i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)) if k in self.junctions]
    return bool(cand) and float(np.min(np.hypot(*(np.concatenate(cand) - [x, y]).T))) < JUNCTION_NEAR

  def road_distance(self, x: float, y: float) -> float:
    i, j = int(x // self.CELL), int(y // self.CELL)
    cand = [self.grid[k] for k in ((i + di, j + dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)) if k in self.grid]
    if not cand:
      return 1e9
    s = np.concatenate(cand)
    a, b, p = s[:, :2], s[:, 2:], np.array([x, y])
    ab = b - a
    t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-9), 0, 1)
    return float(np.min(np.hypot(*(a + ab * t[:, None] - p).T)))

  def bearing(self, a, b) -> float:
    """Compass bearing (clockwise from north) of the link a->b."""
    na, nb = self.nodes[a], self.nodes[b]
    return math.degrees(math.atan2(nb['x'] - na['x'], nb['y'] - na['y'])) % 360

  def gap(self, a, b) -> float:
    na, nb = self.nodes[a], self.nodes[b]
    return math.hypot(nb['x'] - na['x'], nb['y'] - na['y'])

  def nodes_near(self, x: float, y: float, r: float) -> list:
    i, j, n = int(x // self.CELL), int(y // self.CELL), int(r // self.CELL) + 1
    return [k for di in range(-n, n + 1) for dj in range(-n, n + 1) for k in self.node_cells.get((i + di, j + dj), ())
            if math.hypot(self.nodes[k]['x'] - x, self.nodes[k]['y'] - y) < r]

  def link_at(self, x: float, y: float, bearing: float, into: bool, names=None):
    """The link running within EDGE_ANGLE of bearing that ends nearest (x, y) from behind it (into), or starts nearest
    it and goes on past it, on a street of `names` if it can (an overpass's links can run the same way): (a, b) or None."""
    ux, uy = math.sin(math.radians(bearing)), math.cos(math.radians(bearing))
    best = None
    for a in self.nodes_near(x, y, 60.0):
      for b in self.out[a]:
        ang = abs(angle_diff(self.bearing(a, b), bearing))
        end, other = (self.nodes[b], self.nodes[a]) if into else (self.nodes[a], self.nodes[b])
        if ang > EDGE_ANGLE or (((other['x'] - x) * ux + (other['y'] - y) * uy) > 0) == into:
          continue
        score = math.hypot(end['x'] - x, end['y'] - y) + 0.3 * ang
        if names and self.streets.get(self.nodes[a]['st']) not in names:
          score += 30
        if best is None or score < best[0]:
          best = (score, a, b)
    return best and best[1:]

  def walk(self, k, bearing: float, far: float, back: bool) -> list:
    """[(node, m)] along the road from k (whose way is bearing), against the way of travel (back) or with it, out to far
    m: at each node the link that bends least, staying on the street if it can."""
    path, d, seen = [(k, 0.0)], 0.0, {k}
    while d < far:
      best = None
      for n in (self.into[k] if back else self.out[k]):
        b = self.bearing(n, k) if back else self.bearing(k, n)
        bend = abs(angle_diff(b, bearing))
        if n in seen or bend > WALK_TURN:
          continue
        score = bend + (0 if self.nodes[n]['st'] == self.nodes[k]['st'] else 20)
        if best is None or score < best[0]:
          best = (score, n, b)
      if best is None:
        break
      _, n, bearing = best
      d += self.gap(k, n)
      k = n
      seen.add(k)
      path.append((k, d))
    return path

  def spawnable(self, prev, k, nxt) -> str | None:
    """Why the car can't start at node k, on the road prev -> k -> nxt (None: it can)."""
    n = self.nodes[k]
    if n['f'][2] & 4 or (n['f'][1] >> 3) in (15, 16):
      return 'junction'
    if prev is not None and abs(angle_diff(self.bearing(prev, k), self.bearing(k, nxt))) > SPAWN_STRAIGHT:
      return 'bend'
    if len(self.nodes_near(n['x'], n['y'], SPAWN_ALONE)) > 1:
      return 'another node near'
    return None

  def short_trip(self, x: float, y: float, b_in: float, b_out: float, before: float = SHORT_BEFORE,
                 after: float = SHORT_AFTER, names_in=None, names_out=None):
    """A trip from a node about `before` m up the road into a turn at (x, y) to one about `after` m on past it (compass
    bearings in and out of it): (spec, info) or (None, why not)."""
    link_in, link_out = self.link_at(x, y, b_in, True, names_in), self.link_at(x, y, b_out, False, names_out)
    if not link_in or not link_out:
      return None, f"no road {'into' if not link_in else 'out of'} the turn"
    u, v = link_in
    d0 = math.hypot(self.nodes[u]['x'] - x, self.nodes[u]['y'] - y)
    walked = self.walk(u, self.bearing(u, v), before + SHORT_SLACK, back=True)
    chain, dist = [v] + [k for k, _ in walked], [0.0] + [d0 + d for _, d in walked]
    why = {i: self.spawnable(chain[i + 1] if i + 1 < len(chain) else None, chain[i], chain[i - 1]) for i in range(1, len(chain))}
    near = [i for i in why if why[i] is None and abs(dist[i] - before) <= SHORT_SLACK]
    notes = []
    if near:
      i = min(near, key=lambda i: abs(dist[i] - before))
    else:
      ok = [i for i in why if why[i] is None and dist[i] >= SHORT_MIN]
      if not ok:
        return None, f"no node to start at {SHORT_MIN:.0f} m or more before it (road walked back {dist[-1]:.0f} m)"
      i = min(ok, key=lambda i: abs(dist[i] - before))
      notes.append(f"{'short' if dist[i] < before else 'long'} road in: no node to start at nearer {before:.0f} m")
    k, ahead = chain[i], chain[i - 1]
    off = angle_diff(self.bearing(k, ahead), b_in)
    if abs(off) > 30:
      notes.append(f"the road bends {off:+.0f} deg on the way in")
    a, b = link_out
    d1 = math.hypot(self.nodes[a]['x'] - x, self.nodes[a]['y'] - y) + self.gap(a, b)
    on = [(a, 0.0)] + [(n, d1 + d) for n, d in self.walk(b, self.bearing(a, b), after + SHORT_SLACK, back=False)]
    ends = [(n, d) for n, d in on if d >= after - SHORT_SLACK and not self.nodes[n]['f'][2] & 4]
    dn, dd = min(ends, key=lambda e: abs(e[1] - after)) if ends else on[-1]
    if dd < after - SHORT_SLACK:
      notes.append(f"short road out ({dd:.0f} m)")
    kind = self.ynd.highway(self.nodes, k, ahead, self.lanes.get((k, ahead), 0), self.lanes.get((ahead, k), 0))
    if kind in ('motorway', 'trunk'):
      notes.append(f"starts on a {kind}")
    reach = min(math.hypot(self.nodes[n]['x'] - x, self.nodes[n]['y'] - y)
                for n, _ in self.walk(k, self.bearing(k, ahead), dist[i] + 20, back=False))
    if reach > 25:
      notes.append(f"the road on from the start passes {reach:.0f} m from the turn")
    sx, sy, sz = (self.nodes[k][c] for c in 'xyz')
    spec = (round(sx, 1), round(sy, 1), round(sz, 1), round((-self.bearing(k, ahead)) % 360), round(self.nodes[dn]['x'], 1),
            round(self.nodes[dn]['y'], 1))
    return spec, {'before': round(dist[i]), 'after': round(dd), 'lanes': self.lanes.get((k, ahead)),
                  'street': self.streets.get(self.nodes[k]['st']), 'street_out': self.streets.get(self.nodes[dn]['st']),
                  'notes': notes}


class LaneMap:
  """The car's lane by the map's lane tags alone (lane_match.py), outside its junctions' areas: the oncoming check that
  doesn't rest on the game's lane reading, and the lane centres trips start at."""
  def __init__(self, osm, areas):
    from openpilot.tools.sim.bridge.gta5.map.lane_match import LaneMatcher
    self.matcher, self.areas = LaneMatcher(osm), areas

  @classmethod
  def load(cls, map_dir: str) -> 'LaneMap | None':
    from openpilot.tools.sim.bridge.gta5.map.lane_match import JunctionAreas
    from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes
    path = os.path.join(map_dir, "gta5.osm.pbf")
    if not os.path.exists(path):
      return None
    osm = OsmLanes.load(path, to_game)
    if not osm.tagged:
      return None
    areas = JunctionAreas.cached(osm, build=False)
    if areas is None:
      print("e2e: building the map's junction areas (once per map)...", flush=True)
      areas = JunctionAreas.cached(osm)
    return cls(osm, areas)

  def read(self, x: float, y: float, z: float, heading: float) -> list | None:
    """[lane, lanes, kind] at a point (game heading), kind 'bay' for the other direction's turn bay; None off the map's
    lanes or in a junction's area, where the car is on the moves through it."""
    if self.areas.inside(x, y, z):
      return None
    r = self.matcher.match(x, y, math.radians(heading + 90.0), z)
    return None if r is None else [r.lane, r.lanes, 'bay' if r.bay else r.kind]

  def lane_start(self, x: float, y: float, z: float, heading: float, lane: int) -> tuple[float, float, float] | None:
    """The middle of lane `lane` (from the left, clamped) of the road at a trip's start: (x, y, game heading)."""
    at = self.matcher.lane_centre(x, y, math.radians(heading + 90.0), lane, z)
    return None if at is None else (at[0], at[1], (math.degrees(at[2]) - 90.0) % 360)


MAP_ONCOMING = ('oncoming', 'wrong-way', 'bay')  # LaneMap.read's kinds that are in the oncoming lanes
MOVING = 1.0  # m/s: slower, time in the oncoming lanes doesn't count
ONCOMING_HOLD = 3.0  # s a stretch in the oncoming lanes holds over readings that can't say (the map in a junction)


def oncoming_readings(p: dict) -> tuple[bool | None, bool | None, bool | None]:
  """Whether a trace point is in the oncoming lanes by the game's lane reading (route and plugin), by the map's lanes
  (outside junctions) and by either: True, False, or None where it can't say."""
  game = p['lane'][0] < 0 if p.get('lane') else None
  ml = p.get('mlane')
  by_map = None if ml is None else ml[2] in MAP_ONCOMING
  either = True if game or by_map else (None if game is None and by_map is None else False)
  return game, by_map, either


class OncomingTime:
  """Time moving in the oncoming lanes by one reading, and its longest stretch. A point that can't say (None) breaks a
  stretch only after `hold` s of them."""
  def __init__(self, hold: float = 0.0):
    self.s = self.max = self.run = 0.0
    self.hold, self.unknown = hold, 0.0

  def add(self, prev: dict, prev_on: bool | None, on: bool | None, dt: float) -> bool:
    """The step from trace point prev (with its reading prev_on) to the next (reading on); whether the stretch grew."""
    if prev_on and prev['v'] > MOVING:
      self.s, self.run = self.s + dt, self.run + dt
      self.max = max(self.max, self.run)
      self.unknown = 0.0
      return True
    if on is None:
      self.unknown += dt
      if self.unknown <= self.hold:
        return False
    if not on:
      self.run, self.unknown = 0.0, 0.0
    return False


class OncomingTimes:
  """A trip's time in the oncoming lanes by the game's lane reading (as e2e has always counted it: no reading is not
  oncoming), by the map's lanes, and by either, which safe() goes by: the map can't say inside junctions, the game's
  reading misses one-ways driven the wrong way and the other direction's turn bays."""
  def __init__(self):
    self.game, self.map, self.any = OncomingTime(), OncomingTime(ONCOMING_HOLD), OncomingTime(ONCOMING_HOLD)

  def add(self, prev: dict, p: dict) -> bool:
    """The step between two trace points; whether the stretch by either reading grew."""
    dt = p['t'] - prev['t']
    (g0, m0, e0), (g1, m1, e1) = oncoming_readings(prev), oncoming_readings(p)
    self.game.add(prev, bool(g0), bool(g1), dt)
    self.map.add(prev, m0, m1, dt)
    return self.any.add(prev, e0, e1, dt)

  def result(self) -> dict:
    return {f"oncoming{key}_{k}": round(getattr(timer, k), 1) for key, timer in (('', self.game), ('_map', self.map), ('_any', self.any))
            for k in ('s', 'max')}


def oncoming_times(hist: list[dict], lane_map: LaneMap | None = None) -> dict:
  """The oncoming times of a trip's trace (Trip's results' oncoming_* keys); points without the map's reading get it
  from lane_map."""
  times, prev = OncomingTimes(), None
  for p in hist:
    if 'mlane' not in p and lane_map is not None:
      p['mlane'] = lane_map.read(p['x'], p['y'], p['z'], p['h'])
    if prev is not None:
      times.add(prev, p)
    prev = p
  return times.result()


def angle_diff(a, b):
  return (a - b + 180) % 360 - 180


def in_city(x, y):
  return CITY[0] < x < CITY[1] and CITY[2] < y < CITY[3]


def post_json(url: str, obj: dict, timeout: float = 5.0) -> dict:
  req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={'Content-Type': 'application/json'})
  with urllib.request.urlopen(req, timeout=timeout) as r:
    return json.loads(r.read() or b'{}')


def get_json(url: str, timeout: float = 2.0) -> dict:
  with urllib.request.urlopen(url, timeout=timeout) as r:
    return json.loads(r.read())


def plan(x, y, heading, dx, dy) -> dict:
  """The route the bridge's navigator would make (router.Router's request), with its maneuvers and road attributes.
  `heading` is the game's (counterclockwise from north)."""
  def loc(px, py, **kw):
    lat, lon = to_lat_lon(px, py)
    return {'lat': lat, 'lon': lon, **kw}
  bearing = (-heading) % 360
  trip = post_json(f"{VALHALLA}/route", {
    'locations': [loc(x, y, heading=round(bearing) % 360, heading_tolerance=HEADING_TOLERANCE), loc(dx, dy)],
    'costing': 'auto', 'directions_options': {'units': 'kilometers'}})['trip']
  leg = trip['legs'][0]  # one leg: two locations
  pts = np.array([to_game(*p) for p in decode_polyline(leg['shape'])])
  along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))))
  edges = []
  try:
    attrs = ['edge.road_class', 'edge.lane_count', 'edge.use', 'edge.begin_shape_index', 'edge.end_shape_index',
             'edge.names', 'edge.bridge', 'edge.tunnel', 'node.fork', 'node.type', 'node.traffic_signal']
    edges = post_json(f"{VALHALLA}/trace_attributes", {'encoded_polyline': leg['shape'], 'shape_match': 'edge_walk',
                      'costing': 'auto', 'filters': {'attributes': attrs, 'action': 'include'}})['edges']
  except (OSError, ValueError, KeyError):
    pass

  def edge_at(i, before):
    for e in edges:
      if (e['begin_shape_index'] < i <= e['end_shape_index']) if before else (e['begin_shape_index'] <= i < e['end_shape_index']):
        return e
    return None

  maneuvers = []
  for m in leg['maneuvers']:
    i = m['begin_shape_index']
    e_in, e_out = edge_at(i, True), edge_at(i, False)
    angle = None
    if 'bearing_before' in m and 'bearing_after' in m:
      angle = (m['bearing_after'] - m['bearing_before'] + 180) % 360 - 180  # clockwise, right-positive
    maneuvers.append({
      'type': m['type'], 'kind': TYPES.get(m['type'], str(m['type'])), 'instruction': m.get('instruction', ''),
      'bearing': m.get('bearing_after'),
      'angle': angle, 'x': round(float(pts[i][0]), 1), 'y': round(float(pts[i][1]), 1), 'along': round(float(along[i]), 1),
      'real': m['type'] in REAL,
      'in': e_in and {k: e_in.get(k) for k in ('road_class', 'lane_count', 'use', 'names', 'bridge', 'tunnel')},
      'out': e_out and {k: e_out.get(k) for k in ('road_class', 'lane_count', 'use', 'names', 'bridge', 'tunnel')},
      'fork': bool(e_in and (e_in.get('end_node') or {}).get('fork')),
      'signal': bool(e_in and (e_in.get('end_node') or {}).get('traffic_signal')),
    })
  return {'length': float(along[-1]), 'time': trip['summary']['time'], 'points': pts, 'along': along, 'maneuvers': maneuvers}


def spec_str(t) -> str:
  x, y, z, h, dx, dy = t
  return f"{x:.1f},{y:.1f},{z:.1f},{h:.0f}>{dx:.1f},{dy:.1f}"


def parse_spec(s: str):
  """'x,y,z,heading[,lane]>dx,dy': the trip, and its start lane or None."""
  a, b = s.split('>')
  start = [float(v) for v in a.split(',')]
  dx, dy = (float(v) for v in b.split(','))
  return (*start[:4], dx, dy), (int(start[4]) if len(start) > 4 else None)


def pick_trips(m: Map, mode: str, n: int, seed: int) -> list[tuple]:
  rng = random.Random(seed)
  starts = [s for s in m.starts if mode != 'city' or in_city(s[0], s[1])]
  dests = [d for d in m.dests if mode != 'city' or in_city(d[0], d[1])]
  dest_xy = np.array([(d[0], d[1]) for d in dests])
  trips = []
  for _ in range(n * 60):
    if len(trips) >= n:
      break
    x, y, z, bearing = rng.choice(starts)
    dist = np.hypot(dest_xy[:, 0] - x, dest_xy[:, 1] - y)
    near = np.nonzero((dist > STRAIGHT_MIN) & (dist < STRAIGHT_MAX))[0]
    if not len(near):
      continue
    dx, dy, _ = dests[int(rng.choice(near))]
    heading = (-bearing) % 360
    try:
      p = plan(x, y, heading, dx, dy)
    except (OSError, ValueError, KeyError):
      continue
    ms = p['maneuvers']
    if not TRIP_MIN < p['length'] < TRIP_MAX or p['length'] > 2.5 * math.hypot(dx - x, dy - y):
      continue
    if any(mm['type'] in (12, 13) for mm in ms) or not any(mm['real'] and mm['along'] > 30 for mm in ms):
      continue
    if any(mm['real'] and mm['along'] < FIRST_MANEUVER for mm in ms):
      continue  # nav needs room to get up to speed and into lane
    if ms[0].get('bearing') is not None and abs(angle_diff(ms[0]['bearing'], bearing)) > 60:
      continue  # the router starts it on the other carriageway
    trips.append((round(x, 1), round(y, 1), round(z, 1), round(heading), round(dx, 1), round(dy, 1)))
  return trips


# *** the game, the bridge and openpilot ***

class Tail:
  """Follows a growing file from its current end."""
  def __init__(self, path: str):
    self.path = path
    self.pos = os.path.getsize(path) if os.path.exists(path) else 0
    self.buf = b''

  def lines(self) -> list[str]:
    try:
      size = os.path.getsize(self.path)
      if size < self.pos:  # replaced
        self.pos, self.buf = 0, b''
      if size == self.pos:
        return []
      with open(self.path, 'rb') as f:
        f.seek(self.pos)
        data = f.read(min(size - self.pos, 4 << 20))
      self.pos += len(data)
    except OSError:
      return []
    data = self.buf + data
    *out, self.buf = data.split(b'\n')
    return [line.decode(errors='ignore').replace('\r', '') for line in out]


def cmd(type_: str, **kw):
  with socket.create_connection(("127.0.0.1", DEBUG_PORT), timeout=2) as s:
    s.sendall((json.dumps({"type": type_, **kw}) + "\n").encode())


def ai_off():
  """Hands the car back from the plugin's AI driver: one left on (an expert run) overrides openpilot for every trip."""
  from openpilot.tools.sim.bridge.gta5.gta5_expert import CONTROL
  try:
    if CONTROL.exists():
      tmp = CONTROL.with_suffix(".tmp")
      tmp.write_text(json.dumps({"on": False}) + "\n")
      tmp.replace(CONTROL)
    cmd("ai", on=0, indicator="off")
  except OSError as e:
    print(f"e2e: couldn't turn the AI driver off: {e}", flush=True)


def driving_model() -> str:
  """The driving model svc.sh last started openpilot with (MODEL=...)."""
  try:
    with open(MODEL_FILE) as f:
      return f.read().strip()
  except OSError:
    return 'picked' if _SP else 'big'


def svc(action: str, what: str):
  print(f"e2e: svc {action} {what}", flush=True)
  subprocess.run(["bash", SVC, action, what], check=False)


class Infra(Exception):
  pass


class Rig:
  """Game state from the bridge's log, openpilot's messages, the map view's state and the bridge's nav lines."""
  def __init__(self):
    from openpilot.cereal import messaging
    from openpilot.common.params import Params
    self.messaging = messaging
    self.params = Params()
    self.sm = messaging.SubMaster(['selfdriveState', 'carState', 'modelV2'])
    self.states = Tail(STATE_LOG)
    self.log = Tail(BRIDGE_LOG)
    self.state: dict = {}
    self.state_t = 0.0
    self.contacts = 0  # frames the plugin saw the car touch something, since last taken
    self.last_contacts: int | None = None
    self.view: dict = {}
    self.view_t = 0.0
    self.nav_lines: list[tuple[float, str]] = []

  def update(self):
    now = time.monotonic()
    for line in self.states.lines():
      if '"state"' not in line:
        continue
      try:
        s = json.loads(line)['state']
      except (ValueError, KeyError):
        continue
      if s.get('inVehicle'):
        self.state, self.state_t = s, now
        n = s.get('collisions')
        if n is not None:
          self.contacts += max(0, n - self.last_contacts) if self.last_contacts is not None else 0
          self.last_contacts = n
    for line in self.log.lines():
      if line.startswith(('nav:', 'router:')) or 'Traceback' in line or 'Error' in line:
        self.nav_lines.append((now, line.strip()))
    self.sm.update(0)
    if now - self.view_t > 0.3:
      try:
        self.view, self.view_t = get_json(f"{MAP_VIEW}/state", 1.0), now
      except (OSError, ValueError):
        pass

  def take_contacts(self) -> int:
    a, self.contacts = self.contacts, 0
    return a

  @property
  def engaged(self) -> bool:
    return bool(self.sm['selfdriveState'].enabled)

  def wait(self, secs: float, until=None) -> bool:
    end = time.monotonic() + secs
    while time.monotonic() < end:
      self.update()
      if until is not None and until():
        return True
      time.sleep(TICK)
    return False

  def health(self) -> str | None:
    """What's broken, if anything: the bridge (no game state), openpilot (no messages) or the game (nothing from it)."""
    now = time.monotonic()
    if now - self.state_t > STALE:
      return 'bridge'
    if now - max(self.sm.recv_time['selfdriveState'], 0) > STALE and self.sm.recv_frame['selfdriveState'] >= 0:
      return 'openpilot'
    # modeld can stop alone (as after a bridge restart) while openpilot stays engaged: the car then just stops
    if now - max(self.sm.recv_time['modelV2'], 0) > STALE and now - self.sm.recv_time['selfdriveState'] < 1.0:
      return 'modeld'
    return None

  def recover(self, what: str):
    if what in ('openpilot', 'modeld'):
      svc('restart', 'openpilot')
      time.sleep(20)
      svc('restart', 'bridge')
    elif what == 'valhalla':
      # other Valhalla instances share svc.sh's stop pattern: only wait for it, unless allowed to restart it
      if os.getenv("E2E_RESTART_VALHALLA") == "1":
        svc('restart', 'valhalla')
      time.sleep(10)
      return
    else:
      svc('restart', 'bridge')
    # the game reconnects to the bridge on its own; openpilot takes a while to come up
    self.sm = self.messaging.SubMaster(['selfdriveState', 'carState', 'modelV2'])
    ok = self.wait(120, lambda: time.monotonic() - self.state_t < 1.0 and self.sm.recv_frame['selfdriveState'] > 0
                   and time.monotonic() - self.sm.recv_time['selfdriveState'] < 1.0
                   and time.monotonic() - self.sm.recv_time['modelV2'] < 1.0)
    print(f"e2e: recovered {what}: {ok}", flush=True)
    if not ok:
      raise Infra(f"{what} didn't come back")

  def set_engaged(self, on: bool) -> bool:
    for _ in range(3):
      if self.engaged == on:
        return True
      cmd("engage")
      if self.wait(3, lambda: self.engaged == on):
        return True
    return self.engaged == on


# *** a trip ***

class Trip:
  def __init__(self, rig: Rig, roads: Map, trip_id: str, spec: tuple, args, trace_path: str, lane: int | None = None,
               tune: dict | None = None, variant: str | None = None):
    self.rig, self.roads, self.id, self.spec, self.args = rig, roads, trip_id, spec, args
    self.tune, self.variant = tune, variant
    self.lane = args.lane if lane is None else lane
    self.snap_base = trace_path.removesuffix('.jsonl')
    self.trace = open(trace_path, 'w', buffering=1)
    self.history: deque = deque(maxlen=int(60 / TICK))
    self.events: list[dict] = []
    self.maneuvers: list[dict] = []  # scored
    self.route: dict | None = None
    self.pending = 0  # index into the route's maneuvers of the next one
    self.routes_seen = 0
    self.reroutes: list[dict] = []
    self.nudges = 0
    self.distance = 0.0
    self.off_road_t: float | None = None
    self.stopped_t: float | None = None
    self.last_trace = 0.0
    self.collisions = 0
    self.oncoming = OncomingTimes()  # time moving in the oncoming lanes, and the longest stretch
    self.oncoming_snapped, self.oncoming_snaps = False, 0
    self.route_t = 0.0  # when the route being followed was made (trip time)
    self.alert = ''
    self.landing: list[dict] = []  # turns scored done whose landing lane is still to be read
    self.miss_t: float | None = None  # when a short trip missed a maneuver

  def write_tune(self) -> bool:
    """Writes nav's turn parameters for this trip; whether the bridge logged reading them."""
    rig = self.rig
    rig.update()
    seen = len(rig.nav_lines)
    with open(NAVTUNE + ".tmp", 'w') as f:
      json.dump(self.tune, f)
    os.replace(NAVTUNE + ".tmp", NAVTUNE)
    ok = rig.wait(TUNE_ACK, lambda: any(line.startswith('nav: tune') for _, line in rig.nav_lines[seen:]))
    if not ok:
      print(f"e2e: the bridge didn't read {NAVTUNE}: is it running with GTA5_NAVTUNE={NAVTUNE}?", flush=True)
    return ok

  def snap(self, name: str) -> str:
    """Saves the cameras' next frames as <trace>_<name>_road.png etc."""
    path = f"{self.snap_base}_{name}"
    try:
      cmd("snap", path=path)
    except OSError:
      pass
    return path

  def event(self, what: str, **kw):
    s = self.rig.state
    e = {'t': round(time.monotonic() - self.t0, 1), 'event': what, 'pos': [round(v, 1) for v in s.get('pos', [0, 0, 0])], **kw}
    self.events.append(e)
    print(f"  {e}", flush=True)

  def setup(self) -> str | None:
    rig, (x, y, z, h, dx, dy) = self.rig, self.spec
    if rig.engaged and not rig.set_engaged(False):
      return 'could not disengage'
    post_json(f"{MAP_VIEW}/destination", {})
    cmd("waypoint", off=1)
    cmd("world", hour=12, weather="EXTRASUNNY", freeze=1)
    cmd("traffic", on=int(self.args.traffic))
    cmd("lead", remove=1)
    kw = {"x": x, "y": y, "z": z, "heading": h, "lane": self.lane, "fix": 1}
    # at the middle of the start lane by the map: the game's own lanes from a road node can put the car on a centre line
    lane_map = self.roads.lane_map
    at = lane_map.lane_start(x, y, z, h, self.lane) if lane_map is not None else None
    if at is not None:
      kw.update({"laneX": round(at[0], 2), "laneY": round(at[1], 2), "laneZ": z, "laneHeading": round(at[2], 1)})
    self.lane_start = at and [round(v, 1) for v in at]
    if self.args.car and not getattr(self.args, 'car_spawned', False):
      kw["model"] = self.args.car
      self.args.car_spawned = True
    cmd("setup", **kw)
    t = time.monotonic()
    placed = rig.wait(20, lambda: time.monotonic() - t > 4.5 and rig.state_t > t + 4 and rig.state.get('vEgo', 9) < 0.5
                      and math.hypot(rig.state['pos'][0] - x, rig.state['pos'][1] - y) < 40)
    if not placed:
      return f"not placed (at {rig.state.get('pos')})"
    if abs(angle_diff(rig.state['heading'], h)) > 45:
      return f"placed facing {rig.state['heading']:.0f}, not {h:.0f}"
    rig.wait(1.0)
    # a teleport upsets locationd ("locationd Temporary Error"): wait until openpilot would engage
    t = time.monotonic()
    if not rig.wait(ENGAGEABLE_WAIT, lambda: rig.sm['selfdriveState'].engageable):
      ss = rig.sm['selfdriveState']
      return f"not engageable after {ENGAGEABLE_WAIT:.0f} s: {ss.alertText1} {ss.alertText2}".strip()
    self.engageable_after = round(time.monotonic() - t, 1)
    return None

  def run(self) -> dict:
    rig, (x, y, z, h, dx, dy) = self.rig, self.spec
    self.t0 = time.monotonic()
    rec = {'id': self.id, 'spec': spec_str(self.spec), 'mode': 'short' if getattr(self.args, 'short', False) else self.args.mode,
           'car': self.args.car or 'current',
           'traffic': int(self.args.traffic), 'lane': self.lane, 'model': driving_model(), 'stack': STACK, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    if self.tune is not None:
      rec.update({'variant': self.variant, 'tune': self.tune, 'tune_ack': self.write_tune()})
    problem = self.setup()
    if problem:
      return {**rec, 'outcome': 'setup', 'detail': problem}
    s = rig.state
    rec['engageable_after'] = getattr(self, 'engageable_after', None)
    rec['lane_start'] = getattr(self, 'lane_start', None)  # the map's lane centre it was placed at (x, y, heading)
    rec['start'] = {'pos': [round(v, 1) for v in s['pos']], 'heading': round(s['heading']), 'street': s.get('street'), 'lane': s.get('lane')}
    try:
      self.route = plan(s['pos'][0], s['pos'][1], s['heading'], dx, dy)
    except (OSError, ValueError, KeyError) as e:
      rig.recover('valhalla')
      return {**rec, 'outcome': 'infra', 'detail': f"route: {e}"}
    first = self.route['maneuvers'][0].get('bearing') if self.route['maneuvers'] else None
    if first is not None and abs(angle_diff(first, (-s['heading']) % 360)) > 60:
      return {**rec, 'outcome': 'setup', 'detail': f"the route sets off at {first}, the car faces {(-s['heading']) % 360:.0f}"}
    rec['route'] = {'length': round(self.route['length']), 'time': round(self.route['time']),
                    'maneuvers': [m for m in self.route['maneuvers']]}
    routes0 = (rig.view.get('nav') or {}).get('routes', 0)
    post_json(f"{MAP_VIEW}/destination", {"x": dx, "y": dy})
    if not rig.wait(10, lambda: (rig.view.get('nav') or {}).get('routes', 0) > routes0):
      return {**rec, 'outcome': 'infra', 'detail': 'the bridge made no route'}
    self.routes_seen = rig.view['nav']['routes']
    if not rig.set_engaged(True):
      ss = rig.sm['selfdriveState']
      return {**rec, 'outcome': 'no_engage', 'detail': f"{ss.alertText1} {ss.alertText2}".strip()}
    cmd("gas", secs=START_GAS)  # the model won't pull away from a stop
    self.t0 = time.monotonic()
    self.pending = 1 if self.route['maneuvers'] and self.route['maneuvers'][0]['type'] in (1, 2, 3) else 0
    timeout = SHORT_TIMEOUT if getattr(self.args, 'short', False) else max(180.0, 2.5 * self.route['time'] + 120)
    outcome, detail = None, ''
    rig.take_contacts()
    health0 = s.get('bodyHealth')
    last_contact = -1e9
    last_pos = np.array(s['pos'][:2])
    while outcome is None:
      time.sleep(TICK)
      rig.update()
      now = time.monotonic()
      broken = rig.health()
      if broken:
        rig.recover(broken)
        outcome, detail = 'infra', f"{broken} stopped"
        break
      s, cs, ss = rig.state, rig.sm['carState'], rig.sm['selfdriveState']
      pos = np.array(s['pos'][:2])
      self.distance += float(np.hypot(*(pos - last_pos)))
      last_pos = pos
      v = s.get('vEgo', 0.0)
      self._sample(now, s, cs, ss)
      self._follow(now, pos)
      left = float(np.hypot(*(pos - np.array([dx, dy]))))
      if ss.alertText1 != self.alert:
        self.alert = ss.alertText1
        if self.alert:
          self.event('alert', text=f"{ss.alertText1} {ss.alertText2}".strip(), v=round(v, 1), street=s.get('street'))
      ago = next((p for p in reversed(self.history) if p['t'] <= self.history[-1]['t'] - 1.0), None)
      if ago is not None and ago['z'] - s['pos'][2] > FALL:
        self.event('fell', dz=round(ago['z'] - s['pos'][2], 1))
        outcome, detail = 'fell', f"{ago['z'] - s['pos'][2]:.0f} m from ({ago['x']:.0f}, {ago['y']:.0f})"
        break
      if rig.take_contacts():
        if now - last_contact > COLLISION_GAP:
          self.collisions += 1
          self.event('collision', v=round(v, 1), lane=s.get('lane'), street=s.get('street'),
                     damage=None if health0 is None else round(health0 - s.get('bodyHealth', health0)))
        last_contact = now
      if not rig.engaged:
        if left < ARRIVED_WITHIN:
          outcome = 'arrived'
        else:
          outcome, detail = 'disengaged', f"{ss.alertText1} {ss.alertText2}".strip() or str(ss.alertType)
        break
      if sum(not r['repeat'] for r in self.reroutes) > MAX_REROUTES:
        outcome = 'reroutes'
        break
      if self.miss_t is not None and now - self.miss_t > MISS_GRACE:
        outcome, detail = 'missed', f"{left:.0f} m left"
        break
      if now - self.t0 > timeout:
        outcome, detail = 'timeout', f"{left:.0f} m left"
        break
      off = self.roads.road_distance(*pos) if int(now / TICK) % 5 == 0 else None
      if off is not None:
        self.off_road_t = None if off < OFF_ROAD else (self.off_road_t or now)
        if self.off_road_t and now - self.off_road_t > OFF_ROAD_FOR:
          outcome, detail = 'off_road', f"{off:.0f} m from a road"
          break
      if v < STOPPED and left > ARRIVED_WITHIN:
        self.stopped_t = self.stopped_t or now
        stood = now - self.stopped_t
        if stood > CRASH_STOP and self.stopped_t - last_contact < CRASH_WITHIN:
          outcome, detail = 'crash', f"stopped by a collision, {left:.0f} m left"
          break
        if self.nudges < NUDGES and stood > NUDGE_AFTER:
          self.nudges += 1
          if self.nudges == 1:
            self.nudged_from = pos.copy()
          self.event('nudge', n=self.nudges, street=s.get('street'))
          cmd("gas", secs=NUDGE_GAS)
          self.stopped_t = now
        elif self.nudges >= NUDGES and stood > STUCK_AFTER:
          # the gas presses barely moved it: it's up against something
          pinned = float(np.hypot(*(pos - self.nudged_from))) < PINNED
          outcome, detail = ('crash' if self.collisions or pinned else 'stuck'), f"{left:.0f} m left"
          break
      else:
        self.stopped_t = None
    if outcome == 'arrived' and self.pending < len(self.route['maneuvers']):
      for i in range(self.pending, len(self.route['maneuvers'])):  # the last ones, to the destination
        if self.route['maneuvers'][i]['type'] not in (4, 5, 6):
          self._score(i, 'done')
    self._land(final=True)
    if outcome not in ('arrived', 'infra'):
      rec_snap = self.snap('end')
      rig.wait(1.0)
    else:
      rec_snap = None
    if rig.engaged:
      rig.set_engaged(False)
    post_json(f"{MAP_VIEW}/destination", {})
    s = rig.state
    rec.update({
      'outcome': outcome, 'detail': detail, 'clean': outcome == 'arrived' and not self.reroutes and not self.nudges,
      **self.oncoming.result(),
      'duration': round(time.monotonic() - self.t0, 1), 'distance': round(self.distance), 'nudges': self.nudges,
      'collisions': self.collisions,
      'damage': None if health0 is None else round(health0 - s.get('bodyHealth', health0)), 'end': {'pos': [round(v, 1) for v in s.get('pos', [0, 0, 0])], 'street': s.get('street'),
                                             'left': round(float(np.hypot(s['pos'][0] - dx, s['pos'][1] - dy)))},
      'reroutes': self.reroutes, 'maneuvers': self.maneuvers, 'events': self.events, 'snap': rec_snap,
      # modeld's run time per frame (ms) and its dropped-frame percentage
      'model_ms': [round(float(np.mean([p['mt'] for p in self.history])), 1), max(p['mt'] for p in self.history)] if self.history else None,
      'frame_drop': max(p['drop'] for p in self.history) if self.history else None,
      'nav_log': [[round(t - self.t0, 1), line] for t, line in rig.nav_lines if t >= self.t0 - 10],
    })
    self.trace.close()
    return rec

  def _sample(self, now, s, cs, ss):
    md = self.rig.sm['modelV2'].meta
    ds = list(md.desireState)
    try:
      nav_desire = (self.rig.params.get("NavDesire") or b'')
      nav_desire = nav_desire.decode() if isinstance(nav_desire, bytes) else str(nav_desire)
    except Exception:
      nav_desire = ''
    view_nav = self.rig.view.get('nav') or {}
    p = {'t': round(now - self.t0, 2), 'x': round(s['pos'][0], 1), 'y': round(s['pos'][1], 1), 'z': round(s['pos'][2], 1),
         'h': round(s['heading']), 'v': round(s.get('vEgo', 0.0), 2), 'set': round(cs.cruiseState.speed, 2),
         'cap': view_nav.get('cap'), 'lane': s.get('lane'), 'ind': s.get('indicator'),
         'bl': 'L' if cs.leftBlinker else 'R' if cs.rightBlinker else '', 'desire': nav_desire, 'street': s.get('street'),
         'pL': round(ds[1], 2) if len(ds) > 6 else None, 'pR': round(ds[2], 2) if len(ds) > 6 else None,
         'kL': round(ds[5], 2) if len(ds) > 6 else None, 'kR': round(ds[6], 2) if len(ds) > 6 else None,
         'lc': str(md.laneChangeState), 'en': bool(ss.enabled), 'alert': ss.alertText1 or None,
         'mt': round(self.rig.sm['modelV2'].modelExecutionTime * 1000, 1), 'drop': round(self.rig.sm['modelV2'].frameDropPerc, 1)}
    p['junc'] = self.roads.in_junction(p['x'], p['y'])
    lane_map = self.roads.lane_map
    p['mlane'] = lane_map.read(p['x'], p['y'], p['z'], p['h']) if lane_map is not None else None
    prev = self.history[-1] if self.history else None
    self.history.append(p)
    self._land()
    if prev is not None and self.oncoming.add(prev, p):
      if self.oncoming.any.run >= ONCOMING_OK and not self.oncoming_snapped and self.oncoming_snaps < ONCOMING_SNAPS:
        # frames to check the lane readings by
        self.oncoming_snapped, self.oncoming_snaps = True, self.oncoming_snaps + 1
        self.event('oncoming', lane=prev['lane'], mlane=prev['mlane'], street=p['street'], v=p['v'],
                   snap=self.snap(f"onc{self.oncoming_snaps}"))
    elif not self.oncoming.any.run:
      self.oncoming_snapped = False
    if now - self.last_trace >= TRACE_EVERY:
      self.last_trace = now
      self.trace.write(json.dumps(p) + "\n")

  def _land(self, final: bool = False):
    """Reads the landing lane of each turn the car has got LAND_FAR m past (or LAND_WAIT s, or at the trip's end)."""
    p = self.history[-1] if self.history else None
    for m in list(self.landing):
      if not final and p and math.hypot(p['x'] - m['x'], p['y'] - m['y']) < LAND_FAR and p['t'] < m['t'] + LAND_WAIT:
        continue
      self.landing.remove(m)
      m['land'] = landing(list(self.history), m)

  def _follow(self, now, pos):
    """Scores the route's maneuvers as the car gets past them, and the one it missed when the bridge reroutes."""
    r = self.route
    ms = r['maneuvers']
    # done: the car got to a point past it on the route (a later one being done means the earlier ones were too)
    for i in range(self.pending, len(ms)):
      if ms[i]['type'] in (4, 5, 6):
        break
      past = min(ms[i]['along'] + DONE_PAST, r['along'][-1])
      px, py = np.interp(past, r['along'], r['points'][:, 0]), np.interp(past, r['along'], r['points'][:, 1])
      if math.hypot(pos[0] - px, pos[1] - py) < DONE_WITHIN:
        for k in range(self.pending, i + 1):
          self._score(k, 'done')
        self.pending = i + 1
        break
      if ms[i]['along'] - self._along(pos) > 400:
        break
    n = (self.rig.view.get('nav') or {}).get('routes', self.routes_seen)
    if n > self.routes_seen:
      self.routes_seen = n
      self._reroute(pos)

  def _along(self, pos) -> float:
    r = self.route
    d = np.hypot(r['points'][:, 0] - pos[0], r['points'][:, 1] - pos[1])
    return float(r['along'][int(np.argmin(d))])

  def _reroute(self, pos):
    s = self.rig.state
    ms = self.route['maneuvers']
    # where the car left the route: Navigator reroutes after 1.5 s more than 15 m off it
    left_at = next((p for p in reversed(self.history) if p['t'] < self.history[-1]['t'] - 1.5), self.history[-1])
    nxt = next((i for i in range(self.pending, len(ms)) if ms[i]['type'] not in (7, 8)), None)
    near = None if nxt is None else math.hypot(left_at['x'] - ms[nxt]['x'], left_at['y'] - ms[nxt]['y'])
    # a maneuver the car never came near on this route (as one round the block behind it) wasn't the one it missed
    reached = None if nxt is None else min((math.hypot(p['x'] - ms[nxt]['x'], p['y'] - ms[nxt]['y'])
                                            for p in self.history if p['t'] >= self.route_t), default=1e9)
    if reached is not None and reached > REACHED:
      near = None
    rr = {'t': round(time.monotonic() - self.t0, 1), 'pos': [round(v, 1) for v in s['pos']], 'heading': round(s['heading']),
          'street': s.get('street'), 'lane': s.get('lane'), 'v': round(s.get('vEgo', 0.0), 1),
          'maneuver': nxt if near is not None and near < MISSED_NEAR else None,
          'maneuver_dist': None if near is None else round(near)}
    # the navigator routes again every few seconds while the car stays off any route it makes, as when stopped off the
    # road: count only reroutes from somewhere new
    last = next((r for r in reversed(self.reroutes) if not r.get('repeat')), None)
    rr['repeat'] = last is not None and math.hypot(s['pos'][0] - last['pos'][0], s['pos'][1] - last['pos'][1]) < REPEAT_WITHIN
    if rr['repeat']:
      self.reroutes.append(rr)
      return
    rr['snap'] = self.snap(f"rr{len(self.reroutes)}")
    if rr['maneuver'] is not None:
      rr['kind'] = ms[nxt]['kind']
      self._score(nxt, 'missed')
      if getattr(self.args, 'short', False) and self.miss_t is None:
        self.miss_t = time.monotonic()
    self.reroutes.append(rr)
    self.event('reroute', **{k: rr[k] for k in ('street', 'lane', 'maneuver', 'maneuver_dist')}, kind=rr.get('kind'))
    try:
      self.route = plan(s['pos'][0], s['pos'][1], s['heading'], self.spec[4], self.spec[5])
      self.route_t = self.history[-1]['t']
      self.pending = 1 if self.route['maneuvers'] and self.route['maneuvers'][0]['type'] in (1, 2, 3) else 0
      rr['new_route'] = {'length': round(self.route['length']), 'maneuvers': [m['kind'] for m in self.route['maneuvers']]}
    except (OSError, ValueError, KeyError) as e:
      self.event('plan_failed', error=str(e))

  def _score(self, i: int, result: str):
    """The maneuver and how the car came up to it, from the history."""
    m = self.route['maneuvers'][i]
    hist = list(self.history)
    if not hist:
      return
    d = [math.hypot(p['x'] - m['x'], p['y'] - m['y']) for p in hist]
    k = int(np.argmin(d))
    t0 = hist[k]['t']

    def at(dt):
      j = min(range(len(hist)), key=lambda q: abs(hist[q]['t'] - (t0 + dt)))
      p = hist[j]
      return {key: p[key] for key in ('v', 'set', 'cap', 'lane', 'ind', 'bl', 'desire')}

    side = 'L' if m['type'] in LEFT else 'R' if m['type'] in RIGHT else None
    window = [p for p in hist if t0 - 12 <= p['t'] <= t0 + 5]
    prob_key = {'L': 'pL', 'R': 'pR'}.get(side)
    # when the indicator came on for it: the start of its last run on that side before the car got there
    signal_t = None
    for p in window:
      on = bool(side and p['ind'] and p['ind'][0].upper() == side)
      if on and signal_t is None:
        signal_t = p['t'] - t0
      elif not on and p['t'] <= t0:
        signal_t = None
    desires = sorted({p['desire'] for p in window if p['desire']})
    stopped = sum(TICK for p in hist if t0 - 20 <= p['t'] <= t0 and p['v'] < STOPPED)
    around = [p for p in hist if t0 - 2 <= p['t']]
    lanes_after = [p['lane'] for p in hist if p['t'] > t0 + 1 and p['lane']]
    nav_lines = [line for t, line in self.rig.nav_lines if self.t0 + t0 - 25 <= t <= self.t0 + t0 + 5]
    self.maneuvers.append({
      'i': i, 'result': result, 'route_n': len(self.reroutes), 't': t0,
      **{key: m[key] for key in ('kind', 'type', 'angle', 'x', 'y', 'along', 'real', 'in', 'out', 'fork', 'signal', 'instruction')},
      'bearing': m.get('bearing'),
      'closest': round(d[k], 1), 'z': hist[k]['z'], 'street': hist[k]['street'],
      'approach': {f"{dt}s": at(dt) for dt in (-10, -5, -2, 0)},
      'signal_t': None if signal_t is None else round(signal_t, 1),
      'max_turn_p': None if not prob_key else max((p[prob_key] or 0) for p in window) if window else None,
      'max_keep': max((max(p['kL'] or 0, p['kR'] or 0) for p in window), default=None),
      'desires': desires, 'stopped_before': round(stopped, 1), 'nav': nav_lines[-12:],
      'min_v': round(min((p['v'] for p in hist if t0 - 10 <= p['t'] <= t0 + 3), default=0.0), 2),
      'oncoming': any(p['lane'] and p['lane'][0] < 0 for p in around),
      'oncoming_map': any(oncoming_readings(p)[1] for p in around),
      'end_lane': lanes_after[-1] if lanes_after else None,
      'collision': any(e['event'] == 'collision' and t0 - 5 <= e['t'] for e in self.events),
      'speeds': turn_speeds(hist, t0),
    })
    if result == 'done' and turn_maneuver(self.maneuvers[-1]):
      self.landing.append(self.maneuvers[-1])
    tag = f"{m['kind']} {m['angle']}" + (f" -> {m['out'].get('names')}" if m.get('out') else '')
    print(f"  maneuver {i} {result}: {tag}, lane {at(-2)['lane']}, v {at(0)['v']}, signal {None if signal_t is None else round(signal_t, 1)}, stopped {stopped:.0f} s", flush=True)


# *** commands ***

def results_files(paths):
  return paths or sorted(glob.glob(os.path.join(OUT_DIR, "*.jsonl")))


ONCOMING_OK = 1.0  # s in the oncoming lanes a driver would let go (the lane reading flickers across junctions)
ONCOMING_SNAPS = 3  # frames saved per trip, as the car has been in the oncoming lanes that long
JUNCTION_NEAR = 15.0  # m from a junction or stop-line node: in the junction (recorded, as the lane reading there can follow GTA's diagonal links)
# m past a turn's point where the lane it lands in is read: out of the junction, and before nav's lane plan can move on
# for the next maneuver (navd planner.TURN_HOLDS, then a lane per LANE_LINE_CHANGE m)
LAND_FROM, LAND_TO = 10.0, 35.0
# m: where the road on is junction nodes all the way (they come close together), past the turn's diagonal links
LAND_JUNCTION_FROM, LAND_FAR = 20.0, 50.0
LAND_WAIT = 20.0  # s after the turn to stop waiting for the car to get past it


def landing(hist: list[dict], m: dict, in_junction=None) -> dict | None:
  """The lane the car lands in after a turn (the reading most often seen out of the junction LAND_FROM to LAND_TO m past
  its point, else LAND_JUNCTION_FROM to LAND_FAR in it), against the one nav's lane plan (navd planner.lane_plan) has it
  arrive in: the turn side's lane, 0 from the left for a left turn and the rightmost for a right. A history without
  junction flags takes them from in_junction(x, y)."""
  clear, junction = [], []
  for p in hist:
    if p['t'] <= m['t']:
      continue
    d = math.hypot(p['x'] - m['x'], p['y'] - m['y'])
    if d > LAND_FAR:
      break
    if not p['lane']:
      continue
    junc = p['junc'] if 'junc' in p else bool(in_junction and in_junction(p['x'], p['y']))
    if LAND_FROM <= d <= LAND_TO and not junc:
      clear.append(tuple(p['lane']))
    elif d >= LAND_JUNCTION_FROM and junc:
      junction.append(tuple(p['lane']))
  lanes = clear or junction
  if not lanes:
    return None
  lane = Counter(lanes).most_common(1)[0][0]
  planned = 0 if m['type'] in LEFT else lane[1] - 1
  return {'lane': list(lane), 'planned': planned, 'ok': lane[0] == planned, 'reads': len(lanes), 'junc': not clear}


def trace_of(r: dict) -> str | None:
  """A result's trace: its recorded path, or for results from before that, traces/<id>.jsonl unless a later run of the
  same id has written over it since (None)."""
  if r.get('trace'):
    return r['trace'] if os.path.exists(r['trace']) else None
  path = os.path.join(r.get('_dir', OUT_DIR), 'traces', f"{r['id']}.jsonl")
  if not os.path.exists(path):
    return None
  try:
    end = time.mktime(time.strptime(r['started'], '%Y-%m-%d %H:%M:%S')) + (r.get('duration') or 0) + TRACE_SLACK
  except (KeyError, ValueError):
    return path
  return path if os.path.getmtime(path) <= end else None


def backfill_landing(rs: list[dict]):
  """Reads the landing lane of done turns in results from before it was recorded, from their 2 Hz traces (the turn's
  time being the trace point nearest its point)."""
  roads = None
  for r in rs:
    turns = [m for m in r.get('maneuvers', []) if 'land' not in m and m['result'] == 'done' and turn_maneuver(m)]
    path = trace_of(r)
    if not turns or path is None:
      continue
    hist = [json.loads(line) for line in open(path)]
    if not hist:
      continue
    if roads is None and 'junc' not in hist[0]:
      roads = Map()
    for m in turns:
      t = m.get('t')
      if t is None:
        t = min(hist, key=lambda p, m=m: math.hypot(p['x'] - m['x'], p['y'] - m['y']))['t']
      m['land'] = landing(hist, {**m, 't': t}, roads.in_junction if roads else None)


def oncoming_max(r: dict) -> float | None:
  """A result's longest stretch in the oncoming lanes: by either the game's or the map's lane reading, or for results
  from before the map's, the game's alone; None from before either."""
  return r.get('oncoming_any_max', r.get('oncoming_max'))


def safe(r: dict) -> bool:
  """Arrived with nothing a driver would have taken over for: no collision, gas press or time in the oncoming lanes
  (reroutes are fine). Results from before oncoming_s count a turn that went into the oncoming lanes."""
  longest = oncoming_max(r)
  oncoming = longest >= ONCOMING_OK if longest is not None else any(m.get('oncoming') for m in r.get('maneuvers', []))
  return r['outcome'] == 'arrived' and not r.get('collisions') and not r.get('nudges') and not oncoming


def load_results(paths) -> list[dict]:
  out = []
  for p in results_files(paths):
    for line in open(p):
      try:
        out.append({**json.loads(line), '_dir': os.path.dirname(p), '_file': os.path.basename(p)})
      except ValueError:
        pass
  return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
  """95% range of a rate k/n (Wilson score interval)."""
  if not n:
    return 0.0, 1.0
  p = k / n
  c = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
  mid = (p + z * z / (2 * n)) / (1 + z * z / n)
  return max(0.0, mid - c), min(1.0, mid + c)


def rate(k: int, n: int) -> str:
  lo, hi = wilson(k, n)
  return f"{k}/{n} ({100 * lo:.0f}-{100 * hi:.0f}%)"


def mean_range(xs: list[float], n: int = 2000) -> str:
  """Mean and its 95% bootstrap range."""
  if not xs:
    return '-'
  rng = np.random.default_rng(0)
  means = rng.choice(np.array(xs), (n, len(xs))).mean(axis=1)
  return f"{np.mean(xs):.1f} ({np.percentile(means, 2.5):.1f}-{np.percentile(means, 97.5):.1f})"


def sign_p(wins: int, losses: int) -> float:
  """Two-sided sign test (exact): how likely a split this uneven is by chance."""
  n = wins + losses
  if not n:
    return 1.0
  k = min(wins, losses)
  return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def hint(p: float) -> str:
  return 'clear (p<0.05)' if p < 0.05 else 'suggestive (p<0.2)' if p < 0.2 else 'not clear'


def crashed(r: dict) -> bool:
  return r['outcome'] in ('crash', 'fell')


def run_key(r: dict) -> tuple | None:
  """(file, trip, run) of a sweep result, from its id (<trip>-<variant>-<run>[-tag]), to pair it with the other variants'."""
  mark = f"-{r['variant']}-"
  if mark not in r['id']:
    return None
  trip, rest = r['id'].rsplit(mark, 1)
  return r.get('_file'), trip, rest.split('-')[0]


def paired(by: dict, out=print):
  """Each variant against the reference one ('base', else the first), trip for trip and run for run."""
  ref = 'base' if 'base' in by else sorted(by)[0]
  ref_runs = {run_key(r): r for r in by[ref] if run_key(r)}
  for name, group in sorted(by.items()):
    if name == ref:
      continue
    pairs = [(ref_runs[run_key(r)], r) for r in group if run_key(r) in ref_runs]
    if not pairs:
      continue
    out(f"  {name} vs {ref}, {len(pairs)} paired runs:")
    for label, f, better in (('arrived', lambda r: r['outcome'] == 'arrived', True), ('safe', safe, True),
                             ('crash', crashed, False)):
      gain = sum(f(b) and not f(a) for a, b in pairs)
      loss = sum(f(a) and not f(b) for a, b in pairs)
      if not better:
        gain, loss = loss, gain
      out(f"    {label:9s} better in {gain}, worse in {loss}: {hint(sign_p(gain, loss))}")
    for label, key in (('oncoming', 'oncoming_any_max'), ("game's", 'oncoming_max')):
      diffs = [b[key] - a[key] for a, b in pairs if key in a and key in b]
      if diffs:
        less, more = sum(d < -0.5 for d in diffs), sum(d > 0.5 for d in diffs)
        out(f"    {label:9s} longest stretch {mean_range(diffs)} s on average; shorter in {less}, longer in {more}: "
            f"{hint(sign_p(less, more))}")


def landings(turns: list[dict]) -> str:
  """Done turns that landed in nav's planned lane, of those with a landing reading."""
  read = [m['land'] for m in turns if m.get('land')]
  return f"{sum(x['ok'] for x in read)}/{len(read)}"


def read_trips(args, roads) -> list[tuple[str, tuple, int | None]]:
  trips: list[tuple[str, tuple, int | None]] = []
  if args.trip:
    for t in args.trip:
      spec, lane = parse_spec(t)
      trips.append((f"trip-{hashlib.sha1(t.encode()).hexdigest()[:6]}", spec, lane))
  elif args.trips:
    for line in open(args.trips):
      line = line.split('#')[0].split()
      if len(line) == 2:  # id spec
        trips.append((line[0], *parse_spec(line[1])))
  elif args.replay:
    known = {r['id']: (r['spec'], r.get('lane')) for r in load_results([])}
    for rid in args.replay:
      spec, lane = parse_spec(known[rid][0])
      trips.append((f"{rid}-r{time.strftime('%H%M')}", spec, lane if lane is not None else known[rid][1]))
  else:
    picked = pick_trips(roads, args.mode, args.n, args.seed)
    trips = [(f"{args.mode}{args.seed}-{k:02d}", t, None) for k, t in enumerate(picked)]
  return trips


def cmd_run(args):
  roads = Map()
  trips = read_trips(args, roads)
  if args.reps > 1:
    trips = [(f"{tid}-{r}", spec, lane) for r in range(args.reps) for tid, spec, lane in trips]
  if args.tag:
    trips = [(f"{tid}-{args.tag}", spec, lane) for tid, spec, lane in trips]
  tune = None
  if getattr(args, 'tune', None):
    with open(args.tune) as f:
      tune = json.load(f)
  name = os.path.splitext(os.path.basename(args.tune))[0] if tune is not None else None
  drive(args, roads, [(tid, spec, lane, tune, name) for tid, spec, lane in trips])


def cmd_sweep(args):
  """Each trip with each variant of nav's turn parameters, alternating variants trip by trip so that drift in the game
  over the run falls on all of them alike."""
  roads = Map()
  with open(args.variants) as f:
    variants = json.load(f)
  base = read_trips(args, roads)
  tag = f"-{args.tag}" if args.tag else ''
  trips = [(f"{tid}-{name}-{r}{tag}", spec, lane, tune, name)
           for r in range(args.reps) for tid, spec, lane in base for name, tune in variants.items()]
  drive(args, roads, trips)


def new_trace(trace_dir: str, tid: str) -> str:
  """A trace path no earlier run has used (a trip id run again into the same results file gets -2, -3, ...)."""
  path, k = os.path.join(trace_dir, f"{tid}.jsonl"), 2
  while os.path.exists(path):
    path, k = os.path.join(trace_dir, f"{tid}-{k}.jsonl"), k + 1
  return path


TRIP_OVERHEAD = 11.0  # s per trip to place the car, settle and engage (median over the results so far)
TRIP_SPEED = 6.0  # m/s from a standstill over a trip's straight-line length, in town (with the time turns and stops take)


def trip_minutes(spec: tuple, args) -> float:
  """A rough time for a trip, for the sweep's estimate (short trips' failures end sooner)."""
  x, y, _, _, dx, dy = spec
  drive_s = 1.3 * math.hypot(dx - x, dy - y) / TRIP_SPEED
  return (TRIP_OVERHEAD + min(drive_s, SHORT_TIMEOUT if getattr(args, 'short', False) else 1e9)) / 60


def drive(args, roads, trips):
  out = args.out or os.path.join(OUT_DIR, f"{args.name or time.strftime('%m%d-%H%M')}.jsonl")
  trace_dir = os.path.join(os.path.dirname(out), "traces", os.path.splitext(os.path.basename(out))[0])
  os.makedirs(trace_dir, exist_ok=True)
  if roads.lane_map is None:
    print(f"e2e: {MAP_DIR} has no lane tags: oncoming time by the game's lane reading alone, starts at road nodes", flush=True)
  rig = Rig()
  rig.wait(2)
  ai_off()
  print(f"e2e: {len(trips)} trips -> {out}, about {sum(trip_minutes(spec, args) for _, spec, *_ in trips):.0f} min", flush=True)
  status_path = os.path.join(os.path.dirname(out), "status.json")
  counts: Counter = Counter()
  for k, (tid, spec, lane, tune, variant) in enumerate(trips):
    for attempt in range(2):
      print(f"e2e: trip {k + 1}/{len(trips)} {tid} {spec_str(spec)}", flush=True)
      with open(status_path, 'w') as f:
        json.dump({'out': out, 'trip': k + 1, 'of': len(trips), 'id': tid, 'counts': counts, 'at': time.strftime('%H:%M:%S')}, f)
      try:
        broken = rig.health()
        if broken:
          rig.recover(broken)
        trace = new_trace(trace_dir, tid)
        rec = Trip(rig, roads, tid, spec, args, trace, lane, tune, variant).run()
        rec['trace'] = trace
      except (OSError, Infra) as e:
        rec = {'id': tid, 'spec': spec_str(spec), 'mode': args.mode, 'outcome': 'infra', 'detail': str(e)}
        try:
          rig.recover('bridge')
        except Infra as e2:
          print(f"e2e: giving up: {e2}", flush=True)
          return
      print(f"e2e: {tid}: {rec['outcome']} {rec.get('detail', '')} {rec.get('duration', '')} s, "
            f"{sum(not rr.get('repeat') for rr in rec.get('reroutes', []))} reroutes, {rec.get('nudges', 0)} nudges", flush=True)
      if rec['outcome'] in ('infra', 'setup') and attempt == 0:
        continue
      with open(out, 'a') as f:
        f.write(json.dumps(rec) + "\n")
      counts[rec['outcome']] += 1
      break
  with open(status_path, 'w') as f:
    json.dump({'out': out, 'done': True, 'of': len(trips), 'counts': counts, 'at': time.strftime('%H:%M:%S')}, f)
  print(f"e2e: done {dict(counts)}", flush=True)


def cmd_pick(args):
  roads = Map()
  print(f"{len(roads.starts)} starts, {len(roads.dests)} destinations")
  for k, t in enumerate(pick_trips(roads, args.mode, args.n, args.seed)):
    p = plan(t[0], t[1], t[3], t[4], t[5])
    print(f"{args.mode}{args.seed}-{k:02d} {spec_str(t)}  {p['length']:.0f} m  " +
          ", ".join(m['kind'] for m in p['maneuvers'] if m['real']))


def cmd_summary(args):
  rs = load_results(args.files)
  if not rs:
    print("no results")
    return
  print(f"{len(rs)} trips")
  by_mode = defaultdict(Counter)
  for r in rs:
    by_mode[r.get('mode', '?')][r['outcome']] += 1
    by_mode[r.get('mode', '?')]['clean'] += bool(r.get('clean'))
    by_mode[r.get('mode', '?')]['safe'] += safe(r)
  for mode, c in by_mode.items():
    print(f"  {mode}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
    n = sum(v for k, v in c.items() if k not in ('clean', 'safe'))
    print(f"    arrived {rate(c['arrived'], n)}, safe {rate(c['safe'], n)}, crashed {rate(c['crash'] + c['fell'], n)} (95% ranges)")
  ms = [m for r in rs for m in r.get('maneuvers', []) if m.get('real')]
  kinds = defaultdict(Counter)
  for m in ms:
    kinds[m['kind']][m['result']] += 1
  if args.backfill:
    backfill_landing(rs)
  turns = [m for r in rs for m in r.get('maneuvers', []) if turn_maneuver(m)]
  left = [m for m in turns if m['type'] in LEFT]
  right = [m for m in turns if m['type'] not in LEFT]
  wide = [m for m in turns if m.get('land') and m['land']['lane'][1] > 1]
  print(f"turns landed in nav's lane: {landings(turns)} (left {landings(left)}, right {landings(right)}, "
        f"onto 2+ lanes {landings(wide)})")
  print("real maneuvers (done/missed):")
  for kind, c in sorted(kinds.items(), key=lambda kv: -sum(kv[1].values())):
    print(f"  {kind:14s} {c['done']:3d} / {c['missed']:3d}")
  missed = [(r, m) for r in rs for m in r.get('maneuvers', []) if m['result'] == 'missed']
  if missed:
    tags = Counter()
    for r, m in missed:
      for t in tag_miss(m):
        tags[t] += 1
    print("missed maneuver tags: " + ", ".join(f"{k} {v}" for k, v in tags.most_common()))
  elsewhere = [rr for r in rs for rr in r.get('reroutes', []) if rr.get('maneuver') is None and not rr.get('repeat')]
  print(f"reroutes not near a maneuver: {len(elsewhere)}")
  print("failures:")
  for r in rs:
    if r['outcome'] != 'arrived' or not r.get('clean') or not safe(r):
      miss = [f"{m['kind']}({','.join(tag_miss(m))})" for m in r.get('maneuvers', []) if m['result'] == 'missed']
      print(f"  {r['id']:16s} {r['outcome']:10s} {r.get('detail', '')[:40]:40s} rr={sum(not rr.get('repeat') for rr in r.get('reroutes', []))} "
            f"nudge={r.get('nudges', 0)} {' '.join(miss)}  [{r['spec']}]")


def turn_speeds(hist: list[dict], t0: float) -> dict | None:
  """The speed through a turn passed at t0: entering its arc (turned 10 deg from 4 s before), the slowest and the mean
  in it, leaving it (within 10 deg of the heading at the end of the history, up to 6 s after), and how far it turned."""
  near = [p for p in hist if t0 - 6 <= p['t'] <= t0 + 6]
  if len(near) < 3:
    return None
  h0 = min(near, key=lambda p: abs(p['t'] - (t0 - 4)))['h']
  turned = [angle_diff(p['h'], h0) for p in near]
  final = turned[-1]
  arc = [k for k, a in enumerate(turned) if abs(a) > ARC_FROM and abs(a - final) > ARC_FROM]
  if not arc:
    return None
  vs = [near[k]['v'] for k in arc]
  after = next((near[k]['v'] for k in range(arc[-1] + 1, len(near))), near[arc[-1]]['v'])
  return {'entry': near[arc[0]]['v'], 'arc_min': min(vs), 'arc_mean': round(float(np.mean(vs)), 2), 'exit': after,
          'turned': round(final)}


def turn_maneuver(m: dict) -> bool:
  return bool(m.get('real')) and m['type'] in (9, 10, 11, 14, 15, 16) and abs(m.get('angle') or 0) >= 45


def cmd_sweepsum(args):
  """Per variant: trips and outcomes, and over its turns (45 deg or more) how many were made, ended in a lane of the
  car's way, landed in nav's planned lane (of those with a landing reading; and of those onto 2+ lanes), went into the
  oncoming lanes, hit something, stopped before the turn, the slowest speed by it, the speeds into, through and out of
  the arc, and how far the car turned of the turn's angle."""
  rs = [r for r in load_results(args.files) if r.get('variant') is not None]
  if not rs:
    print("no sweep results (trips with a variant)")
    return
  if args.backfill:
    backfill_landing(rs)
  by = defaultdict(list)
  for r in rs:
    by[r['variant']].append(r)
  cols = [('variant', 16), ('trips', 5), ('arrived', 7), ('safe', 4), ('crash', 5), ('turns', 5), ('made', 5), ('lane ok', 7),
          ('landed', 6), ('land 2+', 7), ('oncoming', 8), ('hit', 4), ('stopped', 7), ('min v', 6), ('v in', 5), ('arc min', 7), ('arc mean', 8),
          ('v out', 5), ('turned', 6)]
  print(" ".join(f"{c:>{w}s}" if k else f"{c:{w}s}" for k, (c, w) in enumerate(cols)) + "  tune")
  for name, group in sorted(by.items()):
    turns = [m for r in group for m in r.get('maneuvers', []) if turn_maneuver(m)]
    n = len(turns) or 1

    def pct(k, n=n):
      return f"{100 * k / n:.0f}%"
    made = sum(m['result'] == 'done' for m in turns)
    lane_ok = sum(bool(m.get('end_lane')) and m['end_lane'][0] >= 0 for m in turns)
    land = [m['land'] for m in turns if m.get('land')]
    wide = [x for x in land if x['lane'][1] > 1]

    def landed(xs):
      return f"{100 * sum(x['ok'] for x in xs) / len(xs):.0f}%" if xs else '-'
    oncoming = sum(bool(m.get('oncoming') or m.get('oncoming_map')) for m in turns)
    hit =sum(bool(m.get('collision')) for m in turns)
    stopped = sum(m.get('stopped_before', 0) >= 2 for m in turns)
    speeds = [m['min_v'] for m in turns if m.get('min_v') is not None]
    min_v = float(np.mean(speeds)) if speeds else float('nan')
    outcomes = Counter(r['outcome'] for r in group)
    sp = [m['speeds'] for m in turns if m.get('speeds')]

    def mean(key, sp=sp):
      return f"{np.mean([x[key] for x in sp]):.1f}" if sp else '-'
    # how far the car turned, of the turn's angle: under 1 runs wide or misses, over 1 swings round
    ratio = [abs(m['speeds']['turned']) / abs(m['angle']) for m in turns if m.get('speeds') and m.get('angle')]
    row = [name, len(group), outcomes['arrived'], sum(safe(r) for r in group), outcomes['crash'] + outcomes['fell'], len(turns), pct(made), pct(lane_ok),
           landed(land), landed(wide), pct(oncoming), pct(hit), pct(stopped), f"{min_v:.1f}", mean('entry'), mean('arc_min'), mean('arc_mean'),
           mean('exit'), f"{np.mean(ratio):.2f}" if ratio else '-']
    print(" ".join(f"{str(v):>{w}s}" if k else f"{str(v):{w}s}" for k, (v, (_, w)) in enumerate(zip(row, cols, strict=True)))
          + "  " + json.dumps(group[0].get('tune')))
  print("95% ranges:")
  for name, group in sorted(by.items()):
    n = len(group)
    onc = [oncoming_max(r) for r in group if oncoming_max(r) is not None]
    game = [r['oncoming_max'] for r in group if 'oncoming_max' in r and 'oncoming_any_max' in r]
    print(f"  {name:16s} arrived {rate(sum(r['outcome'] == 'arrived' for r in group), n)}, safe {rate(sum(safe(r) for r in group), n)}, "
          f"crashed {rate(sum(crashed(r) for r in group), n)}, longest oncoming {mean_range(onc)} s"
          + (f" (by the game's reading {mean_range(game)} s)" if game else ""))
  if len(by) > 1:
    print("paired:")
    paired(by)


TURN_AT = re.compile(r"at \((-?[\d.]+),\s*(-?[\d.]+)\)")  # a trips file's note of where its turn is


def turn_of(maneuvers: list[dict], at: tuple | None = None) -> dict | None:
  """A trip's turn: the sharpest real maneuver within 30 m of `at` (the trips file's note), else its first real one of 45
  deg or more, else its first real one."""
  real = [m for m in maneuvers if m.get('real') and m.get('angle') is not None and m.get('bearing') is not None]
  if at is not None:
    near = [m for m in real if math.hypot(m['x'] - at[0], m['y'] - at[1]) < 30]
    return max(near, key=lambda m: abs(m['angle'])) if near else None
  return next((m for m in real if abs(m['angle']) >= 45), real[0] if real else None)


EXIT_NOTE = re.compile(r"exit heading (\d+)")  # overturn-turns.txt's notes: the game heading out of the turn


def noted_turn(maneuvers: list[dict], at: tuple, comment: str) -> dict | None:
  """The turn a trips file's note describes, from any maneuver in the results through it (its trip's own route may go
  round it): a real one within 15 m, on the noted side and out the noted way."""
  side = LEFT if 'left' in comment else RIGHT if 'right' in comment else None
  exit_note = EXIT_NOTE.search(comment)
  best = None
  for m in maneuvers:
    if m.get('bearing') is None and exit_note:
      m = {**m, 'bearing': (-float(exit_note[1])) % 360}
    d = math.hypot(m['x'] - at[0], m['y'] - at[1])
    if not m.get('real') or m.get('bearing') is None or m.get('angle') is None or d > 15:
      continue
    if side and m['type'] not in side:
      continue
    if exit_note and abs(angle_diff(m['bearing'], (-float(exit_note[1])) % 360)) > 30:
      continue
    if best is None or d < best[0]:
      best = (d, m)
  return best and best[1]


def cmd_shorten(args):
  """Prints short trips for the turns of trip files' trips (their routes from earlier results with the same spec), or
  for every turn on the routes of --results files; a line each, with what was found, for a trips file."""
  roads = Map()
  routes, known = {}, []
  for r in load_results([]):
    if (r.get('route') or {}).get('maneuvers'):
      routes[r['spec']] = r['route']['maneuvers']  # the last run's
      known += r['route']['maneuvers']
    known += r.get('maneuvers', [])
  turns = []  # (name, maneuver, lane, source)
  for path in args.trips or []:
    for line in open(path):
      parts = line.split('#')[0].split()
      if len(parts) != 2:
        continue
      spec, lane = parse_spec(parts[1])
      comment = line.partition('#')[2]
      note = TURN_AT.search(comment)
      at = note and (float(note[1]), float(note[2]))
      ms = routes.get(spec_str(spec))
      if ms is None:
        try:  # no run of it yet: the router's route, if it's up
          ms = plan(spec[0], spec[1], spec[3], spec[4], spec[5])['maneuvers']
        except (OSError, ValueError, KeyError):
          ms = []
      m = turn_of(ms, at)
      if m is None and at:
        m = noted_turn(known, at, comment)
      if m is None:
        print(f"# {parts[0]}: no route in the results with its turn ({parts[1]})", file=sys.stderr)
        continue
      turns.append((f"S{parts[0]}", m, lane, os.path.basename(path)))
  for r in load_results(args.results) if args.results else []:
    for k, m in enumerate((r.get('route') or {}).get('maneuvers', [])):
      if turn_maneuver(m) and m.get('bearing') is not None:
        turns.append((f"S{r['id']}-m{k}", m, None, r['id']))
  seen, specs = set(), {}
  print(f"# short trips (e2e.py shorten): start {SHORT_BEFORE:.0f} m before the turn at a road node, destination "
        f"{SHORT_AFTER:.0f} m past it; run with --short")
  for name, m, lane, source in turns:
    if name in seen:
      continue
    seen.add(name)
    b_out = m['bearing']
    spec, info = roads.short_trip(m['x'], m['y'], (b_out - m['angle']) % 360, b_out, names_in=(m.get('in') or {}).get('names'),
                                  names_out=(m.get('out') or {}).get('names'))
    if spec is None:
      print(f"# {name}: {info}", file=sys.stderr)
      continue
    if lane is None:
      lane = 0 if m['type'] in LEFT else 9
    if (spec, lane) in specs:
      print(f"# {name}: the same as {specs[(spec, lane)]}", file=sys.stderr)
      continue
    specs[(spec, lane)] = name
    x, y, z, h, dx, dy = spec
    notes = ''.join(f"; {n}" for n in info['notes'])
    print(f"{name} {x},{y},{z},{h},{lane}>{dx},{dy}    # {m['kind']} {m['angle']:+.0f} at ({m['x']:.0f},{m['y']:.0f}), "
          f"{info['street']} ({info['lanes']} lanes) -> {info['street_out']}, {info['before']} m before, {info['after']} m after, "
          f"from {source}{notes}")


def tag_miss(m: dict) -> list[str]:
  """Rough reasons a maneuver was missed, from the approach."""
  tags = []
  if m.get('route_n') and m.get('along', 1e9) < 100:
    tags.append('right_after_reroute')  # a fresh route that turns off before the car can
  side = 'L' if m['type'] in LEFT else 'R' if m['type'] in RIGHT else None
  a2 = m['approach'].get('-2s', {})
  lane = a2.get('lane')
  if m['type'] in RAMP or m.get('fork'):
    tags.append('fork/ramp')
  if lane and side and lane[1] > 1 and lane[0] != (0 if side == 'L' else lane[1] - 1):
    tags.append('wrong_lane')
  if lane and lane[0] < 0:
    tags.append('oncoming_lane')
  if m['approach'].get('0s', {}).get('v', 0) > 9 and abs(m.get('angle') or 0) > 45:
    tags.append('fast')
  if m.get('stopped_before', 0) >= 2:
    tags.append('stopped_before')
  if m.get('signal_t') is None and abs(m.get('angle') or 0) > 30:
    tags.append('no_signal')
  elif side and (m.get('max_turn_p') or 0) < 0.3:
    tags.append('model_ignored')
  if (m.get('in') or {}).get('bridge') or (m.get('out') or {}).get('bridge'):
    tags.append('bridge')
  return tags or ['?']


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest='command', required=True)
  r = sub.add_parser('run')
  r.add_argument('--mode', choices=['city', 'map'], default='city')
  r.add_argument('--n', type=int, default=10)
  r.add_argument('--seed', type=int, default=1)
  r.add_argument('--trip', action='append', help='x,y,z,heading>dx,dy')
  r.add_argument('--replay', action='append', help='a trip id from the results')
  r.add_argument('--trips', help="a file of trips, a line each: id x,y,z,heading[,lane]>dx,dy")
  r.add_argument('--reps', type=int, default=1, help='runs of each trip')
  r.add_argument('--tag', default='', help='appended to trip ids, e.g. the model')
  r.add_argument('--out', help='results file (default ~/gta5test/e2e/<name>.jsonl)')
  r.add_argument('--name')
  r.add_argument('--car', default=MODEL3, help="model to spawn for the first trip ('' keeps the current car)")
  r.add_argument('--lane', type=int, default=9, help='start lane from the left (clamped: 9 is the rightmost)')
  r.add_argument('--traffic', type=int, default=0)
  r.add_argument('--tune', help="nav's turn parameters for every trip, a JSON file ({} is the defaults)")
  r.add_argument('--short', action='store_true', help=f"short trips: a missed maneuver ends the trip, {SHORT_TIMEOUT:.0f} s timeout")
  sw = sub.add_parser('sweep')
  sw.add_argument('--variants', required=True, help='{"name": {param: value, ...}, ...}')
  for a, kw in (('--mode', {'choices': ['city', 'map'], 'default': 'city'}), ('--n', {'type': int, 'default': 10}),
                ('--seed', {'type': int, 'default': 1}), ('--trip', {'action': 'append'}), ('--replay', {'action': 'append'}),
                ('--trips', {}), ('--reps', {'type': int, 'default': 1}), ('--tag', {'default': ''}), ('--out', {}),
                ('--name', {}), ('--car', {'default': MODEL3}), ('--lane', {'type': int, 'default': 9}),
                ('--traffic', {'type': int, 'default': 0}), ('--short', {'action': 'store_true'})):
    sw.add_argument(a, **kw)
  ss = sub.add_parser('sweepsum')
  ss.add_argument('files', nargs='*')
  ss.add_argument('--backfill', action='store_true', help="read turns' landing lanes from the traces of older results")
  pk = sub.add_parser('pick')
  pk.add_argument('--mode', choices=['city', 'map'], default='city')
  pk.add_argument('--n', type=int, default=10)
  pk.add_argument('--seed', type=int, default=1)
  sh = sub.add_parser('shorten')
  sh.add_argument('--trips', action='append', help='a trips file (repeatable)')
  sh.add_argument('--results', action='append', help='a results file: every turn on its routes')
  sm = sub.add_parser('summary')
  sm.add_argument('files', nargs='*')
  sm.add_argument('--backfill', action='store_true', help="read turns' landing lanes from the traces of older results")
  args = p.parse_args()
  {'run': cmd_run, 'sweep': cmd_sweep, 'sweepsum': cmd_sweepsum, 'pick': cmd_pick, 'summary': cmd_summary,
   'shorten': cmd_shorten}[args.command](args)


if __name__ == '__main__':
  main()
