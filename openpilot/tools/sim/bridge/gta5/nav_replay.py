#!/usr/bin/env python3
"""Replays recorded drives through the bridge's navigation, open loop, and writes a decision log per trip: each step's
game commands, NavDesire writes, cruise cap, the blinkers and steering torque openpilot gets, the bridge's cruise
commands, the route input and lane slots (hashed), the next turn's points and nav's printed lines. The world's
read_sensors runs as live, with the game, openpilot and the router stubbed and a fake clock; routing is synchronous,
and the router's answers are cached, so a replay is repeatable without a router. Two code versions given the same
inputs must write byte-identical logs (the nav layering split's gate: it must not change a decision).

  nav_replay.py extract RESULTS.jsonl [TRIP_ID ...] --out DIR   inputs from e2e results, their 2 Hz traces and bridge.jsonl
  nav_replay.py run DIR --logs OUT [--router URL] [--cache FILE] [--map DIR]
  nav_replay.py diff A B                                         first difference per trip between two `run` outputs

`run` imports the bridge from PYTHONPATH, so pointing that at another checkout replays its code with the same inputs
(the harness supports the bridge before and after the navd split). The inputs: the game's state at 20 Hz from the
bridge's GTA5_LOG, engaged from the controls logged there, and the model's desire probabilities and lane change state
from the e2e trace (2 Hz, held between points); the destination is the trip's, given as the e2e harness gives it.
"""
import argparse
import contextlib
import gzip
import hashlib
import io
import json
import os
import sys
import threading
import time as real_time
from types import SimpleNamespace

import numpy as np

BRIDGE_LOG = os.path.expanduser("~/gta5test/bridge.jsonl")
DROP = ("out", "steerBone", "rotVel", "mount", "world", "density", "vehicle", "ai", "collisions", "bodyHealth", "camHeight",
        "pitch", "roll", "wheelBase", "resets", "paused")  # state keys nav and the world's nav glue don't read
BEFORE, AFTER = 3.0, 1.0  # s of the drive replayed before the trip's start (as the destination is set) and after its end


# *** extract ***

def _mono(line: bytes) -> float | None:
  return float(line[9:line.index(b",")]) if line.startswith(b'{"mono": ') else None


def _seek(f, size: int, target: float) -> int:
  """The offset of a line at or a little before the first logged at `target` (the log's mono only grows)."""
  lo, hi = 0, size
  while hi - lo > 1 << 16:
    mid = (lo + hi) // 2
    f.seek(mid)
    f.readline()
    m = _mono(f.readline())
    if m is None or m >= target:
      hi = mid
    else:
      lo = mid
  f.seek(lo)
  if lo:
    f.readline()
  return f.tell()


def _window(path: str, start: float, end: float) -> tuple[list, list]:
  """The states [(mono, state)] and controls' engaged [(mono, active)] logged between start and end."""
  states, controls = [], []
  with open(path, "rb") as f:
    f.seek(_seek(f, os.path.getsize(path), start - 1.0))
    for line in f:
      m = _mono(line)
      if m is None or m < start:
        continue
      if m > end:
        break
      d = json.loads(line)
      if "state" in d:
        states.append((m, d["state"]))
      elif "control" in d and d["control"].get("type") == "control":
        controls.append((m, bool(d["control"].get("active"))))
  return states, controls


def _offset(states: list, trace: list) -> tuple[float | None, int]:
  """mono - trace t, and how many points agree: the trace's points are the bridge's states (positions to 0.1 m) that e2e
  read as it wrote them."""
  by_pos: dict[tuple, list[float]] = {}
  for m, s in states:
    if s.get("pos"):
      by_pos.setdefault((round(s["pos"][0], 1), round(s["pos"][1], 1)), []).append(m)
  found = []
  for p in trace:
    if p.get("v", 0.0) > 2.0:
      ms = by_pos.get((p["x"], p["y"]))
      if ms and len(ms) == 1:
        found.append(ms[0] - p["t"])
  if len(found) < 5:
    return None, len(found)
  off = float(np.median(found))
  return off, sum(abs(f - off) < 1.0 for f in found)


def _visits(path: str, x: float, y: float, start: float, end: float, step: float = 2.0) -> list[float]:
  """When the logged car was near (x, y) between start and end, sampled every `step` s: one time per visit."""
  out: list[float] = []
  size = os.path.getsize(path)
  with open(path, "rb") as f:
    m = start
    while m < end:
      f.seek(_seek(f, size, m))
      for _ in range(40):
        line = f.readline()
        if b'"state"' in line[:40]:
          p = json.loads(line)["state"].get("pos")
          t = _mono(line)
          if p and abs(p[0] - x) < 30.0 and abs(p[1] - y) < 30.0 and t is not None and (not out or t - out[-1] > 60.0):
            out.append(t)
          break
      m += step
  return out


def _find(log: str, trace: list, near: float, search: float) -> float | None:
  """The trip's mono - trace t: the visit to its start, within `search` s of `near`, whose states its trace matches
  best. The log's monotonic clock drifts from the wall clock the results are stamped with (8 % under WSL), and the
  same trip was driven many times, so the trace's exact positions pick the run."""
  best, best_n = None, 0
  for v in _visits(log, trace[0]["x"], trace[0]["y"], near - search, near + search):
    states, _ = _window(log, v - 60.0, v + 60.0 + trace[-1]["t"])
    off, n = _offset(states, trace)
    if off is not None and n > best_n:
      best, best_n = off, n
  return best


def _tune(nav_log) -> dict:
  for _, line in nav_log or []:
    if line.startswith("nav: tune "):
      return json.loads(line[len("nav: tune "):line.rindex(" from ")])
  return {}


def extract(results: str, ids: list[str], out: str, log: str):
  os.makedirs(out, exist_ok=True)
  wall_to_mono = real_time.time() - real_time.monotonic()  # noqa: TID251  (results are stamped with the wall clock)
  name = os.path.splitext(os.path.basename(results))[0]
  drift = None
  for line in open(results):
    r = json.loads(line)
    if "started" not in r or (ids and r["id"] not in ids):
      continue
    trace_path = r.get("trace") or os.path.join(os.path.dirname(results), "traces", name, f"{r['id']}.jsonl")
    if not os.path.exists(trace_path):
      print(f"{r['id']}: no trace {trace_path}")
      continue
    trace = [p for p in map(json.loads, open(trace_path)) if "x" in p]
    if not trace:
      continue
    began = real_time.mktime(real_time.strptime(r["started"], "%Y-%m-%d %H:%M:%S")) - wall_to_mono
    # the first trip is looked for widely, and the next near where the drift so far puts them
    off = _find(log, trace, began + (drift or 0.0), 6 * 3600.0 if drift is None else 900.0)
    if off is None:
      print(f"{r['id']}: its trace doesn't match the bridge's log around {r['started']}")
      continue
    drift = off - began
    t0, t1 = off - BEFORE, off + trace[-1]["t"] + AFTER
    states, controls = _window(log, t0 - 120.0, t1)
    frames, c, k = [], 0, 0
    engaged = False
    for m, s in states:
      if not t0 <= m <= t1:
        continue
      while c < len(controls) and controls[c][0] <= m:
        engaged = controls[c][1]
        c += 1
      while k + 1 < len(trace) and off + trace[k + 1]["t"] <= m:
        k += 1
      p = trace[k] if off + trace[k]["t"] <= m else {}
      meta = {"lc": p.get("lc", "off"), **{key: p.get(key, 0.0) or 0.0 for key in ("pL", "pR", "kL", "kR")}}
      frames.append({"mono": m, "en": engaged, "meta": meta, "state": {key: v for key, v in s.items() if key not in DROP}})
    dest = [float(v) for v in r["spec"].split(">")[1].split(",")[:2]]
    label = r["id"].split("-", 1)[1].rsplit("-", 1)[0] if r["id"].count("-") >= 2 else ""
    head = {"trip": r["id"], "results": results, "dest": dest, "tune": _tune(r.get("nav_log")), "refresh": "_dr" in label,
            "frames": len(frames), "start": off}
    path = os.path.join(out, f"{name}-{r['id']}.jsonl.gz")
    with gzip.open(path, "wt") as f:
      f.write(json.dumps(head) + "\n")
      for fr in frames:
        f.write(json.dumps(fr) + "\n")
    print(f"{r['id']}: {len(frames)} frames, refresh {head['refresh']}, tune {head['tune']} -> {path}")


# *** run ***

class Clock:
  def __init__(self):
    self.t = 0.0

  def monotonic(self) -> float:
    return self.t

  def time(self) -> float:
    return 1.7e9 + self.t

  def sleep(self, s: float):
    pass


CLOCK = Clock()


def _patch_clocks():
  """Every bridge and navd module's `time` becomes the fake clock."""
  for name, mod in list(sys.modules.items()):
    if name.startswith(("openpilot.tools.sim.bridge.gta5", "openpilot.selfdrive.navd")) and getattr(mod, "time", None) is real_time:
      mod.time = CLOCK


class Params:
  def __init__(self, refresh: bool, log: list):
    self.refresh, self.log, self.values = refresh, log, {}

  def get_bool(self, key: str, block: bool = False) -> bool:
    return self.refresh if key == "TurnDesireRefresh" else bool(self.values.get(key))

  def get(self, key: str, block: bool = False):
    return self.values.get(key)

  def put(self, key: str, value, block: bool = False):
    self.values[key] = value
    if key == "NavDesire":
      self.log.append(value)

  def remove(self, key: str):
    self.values.pop(key, None)
    if key == "NavDesire":
      self.log.append("")


class Writer:
  def __init__(self):
    self.last = ""

  def write(self, vec):
    self.last = hashlib.sha1(np.ascontiguousarray(vec, dtype=np.float32).tobytes()).hexdigest()[:16]


class MapView:
  def __init__(self, dest):
    self.dest = (dest,)

  def take_destination(self):
    d, self.dest = self.dest, None
    return d

  def update(self, state: dict):
    pass


class Model:
  """The SubMaster's modelV2 as the world reads it: meta.desireState and meta.laneChangeState."""
  def __init__(self):
    from openpilot.cereal import log
    self.log = log
    self.meta = SimpleNamespace(desireState=[0.0] * 7, laneChangeState=log.LaneChangeState.off)

  def set(self, meta: dict):
    d = self.log.Desire
    probs = [0.0] * 7
    probs[d.turnLeft], probs[d.turnRight], probs[d.keepLeft], probs[d.keepRight] = meta["pL"], meta["pR"], meta["kL"], meta["kR"]
    self.meta.desireState = probs
    self.meta.laneChangeState = getattr(self.log.LaneChangeState, meta["lc"], self.log.LaneChangeState.off)


class Map:
  """The map stack, loaded once for every trip: the router's paths and lanes, and the world's lane matcher."""
  def __init__(self, map_dir: str, router: str | None, cache: str):
    from openpilot.tools.sim.bridge.gta5 import gta5_world
    from openpilot.tools.sim.bridge.gta5.map.router import Router
    w = gta5_world.GTA5World.__new__(gta5_world.GTA5World)
    w.navigator = SimpleNamespace(router=Router(router or "http://127.0.0.1:1"))
    w.lane_matcher, w.junction_areas, w.gnss = None, None, None
    w._load_paths(os.path.join(map_dir, "paths.jsonl"))
    self.paths, self.osm = w.navigator.router.paths, w.navigator.router.osm
    self.lane_matcher, self.junction_areas = w.lane_matcher, w.junction_areas
    self.graph = None  # navd's road graph, for GTA5_NAV_MATCH
    self.url, self.cache_path = router, cache
    self.cache = json.load(open(cache)) if os.path.exists(cache) else {}
    self.misses = 0

  def router(self):
    from openpilot.tools.sim.bridge.gta5.map.router import Router
    r = Router(self.url or "http://127.0.0.1:1", timeout=30.0, paths=self.paths, osm=self.osm)
    post = r._post

    def cached(action: str, request: dict) -> dict:
      key = action + " " + json.dumps(request, sort_keys=True)
      if key not in self.cache:
        if not self.url:
          raise OSError(f"no cached answer for {key[:120]}")
        self.cache[key] = post(action, request)
        self.misses += 1
      return json.loads(json.dumps(self.cache[key]))
    r._post = cached
    return r

  def save(self):
    if self.misses:
      tmp = f"{self.cache_path}.tmp"
      with open(tmp, "w") as f:
        json.dump(self.cache, f)
      os.replace(tmp, self.cache_path)


def _sync(navigator):
  """Routes on the caller's thread: a route is ready the step it's asked for."""
  def start(pos, bearing, dest, now, z):
    if navigator.busy:
      return
    navigator.busy, navigator.next_try = True, now + navigator.RETRY
    navigator._route(pos.copy(), bearing, dest.copy(), z)
  navigator._start = start


def _world(head: dict, mp: Map, rec: dict):
  from openpilot.tools.sim.bridge.common import QueueMessage  # noqa: F401  (control_cmd_gen's)
  from openpilot.tools.sim.bridge.gta5 import gta5_world
  from openpilot.tools.sim.bridge.gta5.map.router import Navigator
  from openpilot.tools.sim.lib.common import SimulatorState
  w = gta5_world.GTA5World.__new__(gta5_world.GTA5World)
  w.simulator_state = SimulatorState()
  w.q = SimpleNamespace(put=lambda m: rec["q"].append(str(getattr(m, "data", m))))
  w.lock = threading.Lock()
  w.state, w.last_frame_time = None, 0.0
  w.model = Model()
  w.sm = {"modelV2": w.model}
  w.tesla, w.metric, w.VM, w.log, w.recorder = True, False, None, None, None
  w._send = lambda obj: rec["cmd"].append(obj)
  w.params = Params(head["refresh"], rec["nd"])
  w.presses, w.curvature, w.steering, w.gnss, w.nav_msgs = {}, 0.0, False, None, None
  navd = hasattr(w, "_init_nav")
  if navd:
    w._init_nav()
  else:  # the bridge before the split: GTA5World.__init__'s nav part
    from openpilot.tools.sim.bridge.gta5.gta5_nav import Nav, PullAway
    w.indicator, w.indicator_t, w.indicator_heading, w.lane_changing = None, 0.0, 0.0, False
    w.nav = Nav(w._send, w._set_nav_desire, refresh=w.params.get_bool("TurnDesireRefresh"))
    w.pull_away = PullAway(w._send)
    w.next_map, w.lane_line = 0.0, (None, 0.0, [])
    w.dest, w.dest_from_game, w.game_waypoint, w.route = None, False, None, None
    w.route_input, w.routes, w.cap, w.gps_route = None, 0, 0.0, []
  tune_cls = type(w.nav.tune)
  w.nav.tune = tune_cls("")
  w.nav.tune.values.update(head["tune"])
  w.expert = SimpleNamespace(update=lambda state, route, engaged: False)
  w.map_view = MapView(head["dest"])
  w.navigator = Navigator(mp.router())
  _sync(w.navigator)
  w.lane_matcher, w.junction_areas = mp.lane_matcher, mp.junction_areas
  if getattr(gta5_world, "NAV_MATCH", False):  # routing from navd's map match, on simulated 3X GNSS seeded alike each run
    from openpilot.selfdrive.navd.map_match import MapMatcher, RoadGraph
    from openpilot.tools.sim.bridge.gta5.gta5_gnss import Gnss
    if mp.graph is None:
      mp.graph = RoadGraph.from_osm(mp.osm)
    w.gnss = Gnss("qcom3x", seed=0, pm=SimpleNamespace(send=lambda *a: None))
    w.matcher, w.match_t = MapMatcher(mp.graph), None
  w.route_writer, w.lanes_writer = Writer(), Writer()
  w._overlay = lambda state, v: []
  return w, navd


def _turn_points(w, navd: bool):
  r = w.route
  if r is None or w.state is None:
    return None
  state = getattr(w, "_replay_state", None)
  if state is None or not state.get("route"):
    return None
  route = np.array(state["route"], dtype=float)
  if navd:
    tp = w.nav.turn_points(route, state.get("forks"), state.get("stops"), state.get("junctions"))
  else:
    tp = w.nav.turn_points(route, state)
  return None if tp is None else [None if p is None else np.asarray(p).tolist() for p in tp]


def replay(path: str, mp: Map, out: str):
  with gzip.open(path, "rt") as f:
    head = json.loads(f.readline())
    frames = [json.loads(line) for line in f]
  rec = {"cmd": [], "q": [], "nd": []}
  w, navd = _world(head, mp, rec)
  _patch_clocks()
  map_route = w._map_route  # the nav state dict, for the turn points: _map_route's result, as nav is given it

  def keep(state, bearing):
    w._replay_state = map_route(state, bearing)
    return w._replay_state
  w._map_route = keep
  lines = 0
  with gzip.open(out, "wt") as log:
    log.write(json.dumps({"trip": head["trip"], "navd": navd}) + "\n")
    for fr in frames:
      CLOCK.t = fr["mono"]
      w.state, w.last_frame_time = fr["state"], fr["mono"]
      w.simulator_state.is_engaged = fr["en"]
      w.model.set(fr["meta"])
      for k in rec:
        rec[k].clear()
      buf = io.StringIO()
      with contextlib.redirect_stdout(buf):
        w.read_sensors(w.simulator_state)
        tp = _turn_points(w, navd)
      s = w.simulator_state
      step = {"t": fr["mono"], "cmd": rec["cmd"], "q": rec["q"], "nd": rec["nd"], "cap": s.cruise_cap,
              "bl": [s.left_blinker, s.right_blinker], "tq": s.user_torque, "sl": s.speed_limit, "slf": s.speed_limit_follow,
              "ri": w.route_writer.last, "ls": w.lanes_writer.last, "routes": w.routes, "tp": tp,
              "out": buf.getvalue().splitlines()}
      log.write(json.dumps(step) + "\n")
      lines += 1
  return head["trip"], lines


def run(inputs: str, logs: str, router: str | None, cache: str, map_dir: str):
  os.makedirs(logs, exist_ok=True)
  mp = Map(map_dir, router, cache)
  files = sorted(p for p in os.listdir(inputs) if p.endswith(".jsonl.gz"))
  for p in files:
    t = real_time.monotonic()
    trip, n = replay(os.path.join(inputs, p), mp, os.path.join(logs, p.replace(".jsonl.gz", ".log.gz")))
    print(f"{trip}: {n} steps in {real_time.monotonic() - t:.1f} s", file=sys.stderr, flush=True)
    mp.save()


def diff(a: str, b: str) -> int:
  bad = 0
  for p in sorted(set(os.listdir(a)) | set(os.listdir(b))):
    pa, pb = os.path.join(a, p), os.path.join(b, p)
    if not (os.path.exists(pa) and os.path.exists(pb)):
      print(f"{p}: only in {'A' if os.path.exists(pa) else 'B'}")
      bad += 1
      continue
    la, lb = gzip.open(pa, "rt").read().splitlines(), gzip.open(pb, "rt").read().splitlines()
    first = next((k for k in range(1, max(len(la), len(lb))) if k >= len(la) or k >= len(lb) or la[k] != lb[k]), None)
    if first is None:
      print(f"{p}: identical, {len(la) - 1} steps")
      continue
    bad += 1
    print(f"{p}: differs at step {first}:\n  A {la[first] if first < len(la) else '(end)'}\n  B {lb[first] if first < len(lb) else '(end)'}")
  return bad


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  e = sub.add_parser("extract")
  e.add_argument("results")
  e.add_argument("ids", nargs="*")
  e.add_argument("--out", required=True)
  e.add_argument("--log", default=BRIDGE_LOG)
  r = sub.add_parser("run")
  r.add_argument("inputs")
  r.add_argument("--logs", required=True)
  r.add_argument("--router", help="a Valhalla server for answers not yet cached")
  r.add_argument("--cache", required=True, help="the router's answers, JSON")
  r.add_argument("--map", required=True, help="the map's folder: paths.jsonl and gta5.osm.pbf")
  d = sub.add_parser("diff")
  d.add_argument("a")
  d.add_argument("b")
  args = ap.parse_args()
  if args.cmd == "extract":
    extract(args.results, args.ids, args.out, args.log)
  elif args.cmd == "run":
    os.environ.setdefault("GTA5_DEBUG", "1")
    run(args.inputs, args.logs, args.router, args.cache, args.map)
  else:
    sys.exit(1 if diff(args.a, args.b) else 0)


if __name__ == "__main__":
  main()
