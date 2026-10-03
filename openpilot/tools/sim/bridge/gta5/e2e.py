#!/usr/bin/env python3
"""End-to-end drive tests: random trips on the map, driven by openpilot with nav, recorded for analysis.

Needs the services running (svc.sh: valhalla, openpilot, bridge with GTA5_MAP, GTA5_ROUTER, GTA5_LOG and GTA5_DEBUG) and
the game in a car. Each trip puts the car at a start, sets the map view's destination, engages and watches until the car
arrives, disengages, gets stuck, leaves the road, reroutes too often or runs out of time. A JSON line per trip goes to
--out; a 2 Hz trace of each to <out dir>/traces/<id>.jsonl.

  e2e.py run --mode city --n 20 --seed 1             random trips in Los Santos (or --mode map, the whole map)
  e2e.py run --trip "x,y,z,heading>dx,dy"            one trip; heading in game degrees (counterclockwise from north)
  e2e.py run --replay city1-03                       a trip again, by its id in the results
  e2e.py pick --mode city --n 5 --seed 1             only print trips
  e2e.py summary [results.jsonl ...]                 outcomes, maneuvers, failures across runs

Every maneuver on the route is scored: done, or missed (a reroute near it), with the car's lane, speed, set speed, the
nav's signal and the model's desire probabilities over the approach. Gas presses are only the test driver's, after the
car has stood still for a while (the model won't pull away from a stop by itself).
"""
import argparse
import glob
import hashlib
import json
import math
import os
import random
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
STALE = 5.0  # s without game state or openpilot messages: restart that service


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
    return None

  def recover(self, what: str):
    if what == 'openpilot':
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
                   and time.monotonic() - self.sm.recv_time['selfdriveState'] < 1.0)
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
  def __init__(self, rig: Rig, roads: Map, trip_id: str, spec: tuple, args, trace_path: str, lane: int | None = None):
    self.rig, self.roads, self.id, self.spec, self.args = rig, roads, trip_id, spec, args
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
    self.route_t = 0.0  # when the route being followed was made (trip time)
    self.alert = ''

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
    rec = {'id': self.id, 'spec': spec_str(self.spec), 'mode': self.args.mode, 'car': self.args.car or 'current',
           'traffic': int(self.args.traffic), 'lane': self.lane, 'model': driving_model(), 'stack': STACK, 'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    problem = self.setup()
    if problem:
      return {**rec, 'outcome': 'setup', 'detail': problem}
    s = rig.state
    rec['engageable_after'] = getattr(self, 'engageable_after', None)
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
    timeout = max(180.0, 2.5 * self.route['time'] + 120)
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
    self.history.append(p)
    if now - self.last_trace >= TRACE_EVERY:
      self.last_trace = now
      self.trace.write(json.dumps(p) + "\n")

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
    nav_lines = [line for t, line in self.rig.nav_lines if self.t0 + t0 - 25 <= t <= self.t0 + t0 + 5]
    self.maneuvers.append({
      'i': i, 'result': result, 'route_n': len(self.reroutes),
      **{key: m[key] for key in ('kind', 'type', 'angle', 'x', 'y', 'along', 'real', 'in', 'out', 'fork', 'signal', 'instruction')},
      'closest': round(d[k], 1), 'z': hist[k]['z'], 'street': hist[k]['street'],
      'approach': {f"{dt}s": at(dt) for dt in (-10, -5, -2, 0)},
      'signal_t': None if signal_t is None else round(signal_t, 1),
      'max_turn_p': None if not prob_key else max((p[prob_key] or 0) for p in window) if window else None,
      'max_keep': max((max(p['kL'] or 0, p['kR'] or 0) for p in window), default=None),
      'desires': desires, 'stopped_before': round(stopped, 1), 'nav': nav_lines[-12:],
    })
    tag = f"{m['kind']} {m['angle']}" + (f" -> {m['out'].get('names')}" if m.get('out') else '')
    print(f"  maneuver {i} {result}: {tag}, lane {at(-2)['lane']}, v {at(0)['v']}, signal {None if signal_t is None else round(signal_t, 1)}, stopped {stopped:.0f} s", flush=True)


# *** commands ***

def results_files(paths):
  return paths or sorted(glob.glob(os.path.join(OUT_DIR, "*.jsonl")))


def load_results(paths) -> list[dict]:
  out = []
  for p in results_files(paths):
    for line in open(p):
      try:
        out.append(json.loads(line))
      except ValueError:
        pass
  return out


def cmd_run(args):
  trips: list[tuple[str, tuple, int | None]] = []
  roads = Map()
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
  if args.reps > 1:
    trips = [(f"{tid}-{r}", spec, lane) for r in range(args.reps) for tid, spec, lane in trips]
  if args.tag:
    trips = [(f"{tid}-{args.tag}", spec, lane) for tid, spec, lane in trips]
  out = args.out or os.path.join(OUT_DIR, f"{args.name or time.strftime('%m%d-%H%M')}.jsonl")
  os.makedirs(os.path.join(os.path.dirname(out), "traces"), exist_ok=True)
  rig = Rig()
  rig.wait(2)
  print(f"e2e: {len(trips)} trips -> {out}", flush=True)
  status_path = os.path.join(os.path.dirname(out), "status.json")
  counts: Counter = Counter()
  for k, (tid, spec, lane) in enumerate(trips):
    for attempt in range(2):
      print(f"e2e: trip {k + 1}/{len(trips)} {tid} {spec_str(spec)}", flush=True)
      with open(status_path, 'w') as f:
        json.dump({'out': out, 'trip': k + 1, 'of': len(trips), 'id': tid, 'counts': counts, 'at': time.strftime('%H:%M:%S')}, f)
      try:
        broken = rig.health()
        if broken:
          rig.recover(broken)
        rec = Trip(rig, roads, tid, spec, args, os.path.join(os.path.dirname(out), "traces", f"{tid}.jsonl"), lane).run()
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
  for mode, c in by_mode.items():
    print(f"  {mode}: " + ", ".join(f"{k} {v}" for k, v in c.most_common()))
  ms = [m for r in rs for m in r.get('maneuvers', []) if m.get('real')]
  kinds = defaultdict(Counter)
  for m in ms:
    kinds[m['kind']][m['result']] += 1
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
    if r['outcome'] != 'arrived' or not r.get('clean'):
      miss = [f"{m['kind']}({','.join(tag_miss(m))})" for m in r.get('maneuvers', []) if m['result'] == 'missed']
      print(f"  {r['id']:16s} {r['outcome']:10s} {r.get('detail', '')[:40]:40s} rr={sum(not rr.get('repeat') for rr in r.get('reroutes', []))} "
            f"nudge={r.get('nudges', 0)} {' '.join(miss)}  [{r['spec']}]")


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
  pk = sub.add_parser('pick')
  pk.add_argument('--mode', choices=['city', 'map'], default='city')
  pk.add_argument('--n', type=int, default=10)
  pk.add_argument('--seed', type=int, default=1)
  sm = sub.add_parser('summary')
  sm.add_argument('files', nargs='*')
  args = p.parse_args()
  {'run': cmd_run, 'pick': cmd_pick, 'summary': cmd_summary}[args.command](args)


if __name__ == '__main__':
  main()
