#!/usr/bin/env python3
"""Unattended recording: the game's AI expert driver (gta5_expert.py) drives random trips on our map for hours while the
bridge records them for training (gta5_record.py).

  record_run.py run --hours 8 [--seed 1] [--name night1]      drive and record (~/gta5test/record.sh starts it in tmux)
  record_run.py run --dry-run --n 30 [--seed 1]                only pick and print the trips (Valhalla queries alone)
  record_run.py summary ~/gta5test/recruns/night1.jsonl        a run's summary again, as it stands

Needs the services running (svc.sh) with the bridge started with BRIDGE_EXTRA (by default "GTA5_EXPERT=1
GTA5_RECORD=/mnt/e/gta5rec"); --fix-bridge restarts it with that if it runs without them. Each trip is put to the
expert with `gta5_cmd.py expert route <trip> <settings>`; the settings are the k=v options of --settings, else of
--settings-file, read again before each trip so a better set can be dropped in during a run, else SETTINGS. --randomise
runs `gta5_cmd.py randomise <seed>` (the scene, car and camera mount) before each trip, once gta5_cmd has it.

Trips are random, 0.5-3 km (--min-len, --max-len), mostly in Los Santos (--map-share of them anywhere on the map), and
each is the best of --candidates picked ones: the most junctions per km (--junction-weight), the maneuver classes
(square, soft and sharp turns each way, ramps, exits, forks) and areas the run has had least of, and the least road
driven before, in this run or the earlier runs in the runs folder.

A trip fails when the car isn't placed at its start, gets no route, makes no progress for --stuck-s, has more collisions
than --max-collision-frames or --max-collision-events, is in the oncoming lanes for --oncoming-s, is more than OFF_ROUTE
m off the route for --off-route-s, reroutes too often, or runs out of time (scaled by the route's length and time); then
the expert is stopped and the next trip's teleport starts afresh. A bridge that died, hung or stopped recording is
restarted (svc.sh restart bridge, with BRIDGE_EXTRA). A plugin that stays disconnected for --plugin-wait s, a driver who
takes over with the engage key, too many failures in a row or restarts, and low disk (under --min-free-gb on the
recordings' drive, or --min-free-logs-gb on WSL's disk or the Windows drive holding it) end the run. So do --hours, the stop file (touch <runs>/STOP) and
SIGINT (Ctrl-C), which end the trip being driven at once.

Into the runs folder (--runs, ~/gta5test/recruns): <name>.jsonl a JSON line per trip (route, settings, outcome, metrics,
segments), <name>.segments.jsonl each recorded segment and the trip it's from (by the middle of its frames, so filter
training data by trip outcome), <name>.summary.json and .txt at the end, <name>.log the console, status.json now."""
import argparse
import glob
import json
import math
import os
import random
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from openpilot.tools.sim.bridge.gta5 import e2e
from openpilot.tools.sim.bridge.gta5.gta5_expert import DEFAULTS as EXPERT_DEFAULTS
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game, to_lat_lon
from openpilot.tools.sim.bridge.gta5.map.router import HEADING_TOLERANCE, decode_polyline

HOME = os.path.expanduser("~")
T = f"{HOME}/gta5test"
REPO = os.path.dirname(os.path.abspath(__file__))
GTA5_CMD = os.path.join(REPO, "gta5_cmd.py")
SVC = f"{T}/svc.sh"
# the recordings go to a big Windows drive (drvfs: 145 MB/s written, against about 2 MB/s needed), the logs stay in WSL
BRIDGE_EXTRA = "GTA5_EXPERT=1 GTA5_RECORD=/mnt/e/gta5rec"
SETTINGS_FILE = f"{T}/record_settings.txt"
# the tuned set (notes/ai_tune_runs.txt, G1), when there's no settings file
SETTINGS = ("task=coord speed=10 ramp=1.0 lead=3 launch=1.5 decel=0.8 turn_speed=5 arrive=gentle targets=smooth ahead_min=100 " +
            "ahead_max=160 speed_step=0.2 style=1075845283 ability=1.0")
SMOOTH = f"{T}/notes/ai_smooth.py"
NOT_SETTINGS = {"on", "need_route", "dest"}  # gta5_cmd's route sets these

# picking trips
AREA_CELL = 500.0  # m: the city's areas, for variety
COVER_CELL = 30.0  # m: road driven before
COVER_OLD = 0.5  # weight of road driven in earlier runs, against this one's
FIRST_REAL = 120.0  # m: no maneuver nearer the start (the car has to get going and into lane)
DETOUR = 2.5  # route length / straight line, at most
STRAIGHTNESS = 1.35  # route length / straight line, typically: for picking a destination for a length
LEN_BINS = 4  # length bands between --min-len and --max-len, filled evenly
JUNCTION_STEP = 10.0  # m between route points looked at for junctions
MAX_JUNCTIONS_KM = 15.0
TOO_FAMILIAR = 0.7  # of a route's road driven before in this run: picked only if nothing else is
GEOM_STEP = 50.0  # m between the route points kept in the run log
# a rough time per trip: placing and settling, then the expert's average speed in town (short tuning drives: 3-5 m/s)
EST_OVERHEAD = 20.0  # s
EST_SPEED = 5.0  # m/s

# driving
TICK = 0.5  # s
PLACE_WAIT = 12.0  # s after the route command for the car to be at the trip's start
PLACED_NEAR = 40.0  # m
ROUTE_WAIT = 25.0  # s after the route command for the expert to drive our route
PROGRESS = 10.0  # m moved that counts as progress
OFF_ROUTE = 30.0  # m
REROUTE_JUMP = 50.0  # m the route's remaining length grows by: a new route from somewhere else
MAX_REROUTES = 4
COLLISION_GAP = 2.0  # s without contact between collisions
TIMEOUT_BASE = 90.0  # s
TIMEOUT_SPEED = 3.0  # m/s: the trip's timeout allows at least its length at this speed
TIMEOUT_ROUTE = 2.0  # x Valhalla's time for it, if longer
STALE = 10.0  # s without game state from the bridge
HUNG_WAIT = 60.0  # s more of no game state, with the bridge running and the game connected, before restarting the bridge
BRIDGE_UP_WAIT = 180.0  # s for a restarted bridge's game connection
RECORD_STALL = 300.0  # s driving without a segment recorded (they're 60 s)
ROUTER_WAIT = 600.0  # s of Valhalla failing before giving up
COOLDOWN = 2.0  # s between trips
FRAME_HZ = 20.0  # recorded frames per second
RECORDED = re.compile(r"gta5: recorded (\S+): (\d+) frames(?:, ended: (.*))?")


def say(*a):
  msg = time.strftime("%H:%M:%S ") + " ".join(str(v) for v in a)
  print(msg, flush=True)
  if LOG is not None:
    LOG.write(msg + "\n")
    LOG.flush()


LOG = None


def parse_extra(s: str) -> dict:
  return dict(tok.split("=", 1) for tok in shlex.split(s) if "=" in tok)


def read_settings(args) -> tuple[list[str], list[str]]:
  """The expert's k=v options, and those that aren't any: --settings, else --settings-file's (# comments), else
  SETTINGS."""
  text = args.settings
  if text is None:
    try:
      with open(args.settings_file) as f:
        text = " ".join(line.split("#")[0] for line in f)
    except OSError:
      text = SETTINGS
  good, bad = [], []
  for tok in text.split():
    key = tok.partition("=")[0]
    (bad if "=" not in tok or key not in EXPERT_DEFAULTS or key in NOT_SETTINGS else good).append(tok)
  return good, bad


def randomise_supported() -> bool:
  # gta5_cmd sends any other first word to the plugin as a command type, so look for the subcommand itself
  try:
    with open(GTA5_CMD) as f:
      return "randomise" in f.read()
  except OSError:
    return False


# *** trips ***

def maneuver_class(m: dict) -> str | None:
  """What the car does at a route maneuver: <side> square|soft|sharp for turns, else ramp, exit, fork, roundabout."""
  t, a = m["type"], m.get("angle")
  if t in (26, 27):
    return "roundabout"
  if t in (17, 18, 19):
    return "ramp"
  if t in (20, 21):
    return "exit"
  if t in (22, 23, 24):
    return "fork"
  if t not in e2e.REAL:
    return None
  side = "left" if t in e2e.LEFT else "right"
  if a is not None:
    a = abs(a)
    return f"{side} {'sharp' if a >= 120 else 'square' if a >= 55 else 'soft'}"
  return f"{side} {'sharp' if t in (11, 14) else 'soft' if t in (9, 16) else 'square'}"


def plan(x, y, heading, dx, dy) -> dict:
  """e2e.plan's route and maneuvers, without its second query for road attributes, which picking doesn't need."""
  def loc(px, py, **kw):
    lat, lon = to_lat_lon(px, py)
    return {"lat": lat, "lon": lon, **kw}
  trip = e2e.post_json(f"{e2e.VALHALLA}/route", {
    "locations": [loc(x, y, heading=round((-heading) % 360) % 360, heading_tolerance=HEADING_TOLERANCE), loc(dx, dy)],
    "costing": "auto", "directions_options": {"units": "kilometers"}})["trip"]
  leg = trip["legs"][0]
  pts = np.array([to_game(*p) for p in decode_polyline(leg["shape"])])
  along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))))
  maneuvers = []
  for m in leg["maneuvers"]:
    angle = None
    if "bearing_before" in m and "bearing_after" in m:
      angle = (m["bearing_after"] - m["bearing_before"] + 180) % 360 - 180  # clockwise, right-positive
    maneuvers.append({"type": m["type"], "kind": e2e.TYPES.get(m["type"], str(m["type"])), "bearing": m.get("bearing_after"),
                      "angle": angle, "along": float(along[m["begin_shape_index"]]), "real": m["type"] in e2e.REAL})
  return {"length": float(along[-1]), "time": trip["summary"]["time"], "points": pts, "along": along, "maneuvers": maneuvers}


class Picker:
  """Random trips, each the best of a few candidates for junctions and for what the run has had least of."""
  def __init__(self, roads, args, old_routes: list[list]):
    self.roads, self.args = roads, args
    self.rng = random.Random(args.seed)
    self.city_starts = [s for s in roads.starts if e2e.in_city(s[0], s[1])]
    self.map_starts = [s for s in roads.starts if not e2e.in_city(s[0], s[1])] or self.city_starts
    self.start_areas = {id(ss): [self._area(s[0], s[1]) for s in ss] for ss in (self.city_starts, self.map_starts)}
    self.city_dests = np.array([d for d in roads.dests if e2e.in_city(d[0], d[1])])
    self.all_dests = np.array(roads.dests)
    self.areas: Counter = Counter()
    self.classes: Counter = Counter()
    self.bins = [0] * LEN_BINS
    self.covered: dict[tuple, float] = {}
    self.starts_used: set = set()
    self.n = 0
    for pts in old_routes:
      for c in self._cells(np.asarray(pts, dtype=float)):
        self.covered[c] = max(self.covered.get(c, 0.0), COVER_OLD)

  def _cells(self, pts: np.ndarray) -> set:
    if len(pts) < 2:
      return set()
    along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))))
    s = np.arange(0.0, along[-1], COVER_CELL / 2)
    xs, ys = np.interp(s, along, pts[:, 0]), np.interp(s, along, pts[:, 1])
    return {(int(x // COVER_CELL), int(y // COVER_CELL)) for x, y in zip(xs, ys, strict=True)}

  def _area(self, x, y) -> tuple:
    return int(x // AREA_CELL), int(y // AREA_CELL)

  def _bin(self, length: float) -> int:
    a = self.args
    return min(LEN_BINS - 1, max(0, int((length - a.min_len) / (a.max_len - a.min_len) * LEN_BINS)))

  def candidate(self) -> dict | None:
    a, rng = self.args, self.rng
    anywhere = rng.random() < a.map_share
    starts = self.map_starts if anywhere else self.city_starts
    dests = self.all_dests if anywhere else self.city_dests
    # an area the run has had little of, then a length band it has had little of
    w = [1.0 / (1 + self.areas[k]) ** 2 for k in self.start_areas[id(starts)]]
    x, y, z, bearing = starts[int(rng.choices(range(len(starts)), weights=w)[0])]
    bw = [1.0 / (1 + n) ** 2 for n in self.bins]
    b = rng.choices(range(LEN_BINS), weights=bw)[0]
    lo = a.min_len + (a.max_len - a.min_len) * b / LEN_BINS
    want = rng.uniform(lo, lo + (a.max_len - a.min_len) / LEN_BINS) / STRAIGHTNESS
    dist = np.hypot(dests[:, 0] - x, dests[:, 1] - y)
    near = np.nonzero((dist > 0.75 * want) & (dist < 1.25 * want))[0]
    if not len(near):
      return None
    dx, dy, _ = dests[int(rng.choice(near))]
    heading = (-bearing) % 360
    try:
      p = plan(x, y, heading, dx, dy)
    except (OSError, ValueError, KeyError):
      return None
    ms = p["maneuvers"]
    if not a.min_len <= p["length"] <= a.max_len or p["length"] > DETOUR * math.hypot(dx - x, dy - y):
      return None
    if any(m["type"] in (12, 13) for m in ms) or not any(m["real"] for m in ms):
      return None
    if any(m["real"] and m["along"] < FIRST_REAL for m in ms):
      return None
    if ms[0].get("bearing") is not None and abs(e2e.angle_diff(ms[0]["bearing"], bearing)) > 60:
      return None  # the router starts it on the other carriageway
    classes = [c for c in (maneuver_class(m) for m in ms) if c]
    pts = p["points"]
    s = np.arange(0.0, p["length"], JUNCTION_STEP)
    inj = [self.roads.in_junction(float(np.interp(v, p["along"], pts[:, 0])), float(np.interp(v, p["along"], pts[:, 1]))) for v in s]
    junctions = int(sum(1 for i in range(1, len(inj)) if inj[i] and not inj[i - 1]))
    cells = self._cells(pts)
    familiar = sum(self.covered.get(c, 0.0) for c in cells) / max(len(cells), 1)
    first_left = next((m["type"] in e2e.LEFT for m in ms if m["real"]), False)
    lane = a.lane if a.lane != "auto" else (0 if first_left else 9)
    geom = [[round(float(np.interp(v, p["along"], pts[:, 0]))), round(float(np.interp(v, p["along"], pts[:, 1])))]
            for v in np.append(np.arange(0.0, p["length"], GEOM_STEP), p["length"])]
    return {"spec": f"{x:.1f},{y:.1f},{z:.1f},{heading:.0f},{lane}>{dx:.1f},{dy:.1f}", "start": (x, y), "dest": (dx, dy),
            "area": "map" if anywhere else "city", "length": round(p["length"]), "time": round(p["time"]),
            "maneuvers": [m["kind"] for m in ms], "classes": classes, "junctions": junctions, "familiar": round(familiar, 2),
            "cells": cells, "geom": geom}

  def score(self, c: dict) -> float:
    km = c["length"] / 1000.0
    jd = min(c["junctions"] / km, MAX_JUNCTIONS_KM) / MAX_JUNCTIONS_KM
    novelty = sum(1.0 / (1 + self.classes[k]) for k in c["classes"]) / max(len(c["classes"]), 1)
    variety = len(set(c["classes"])) / 4.0
    reuse = 1.0 if self._area(*c["start"][:2]) in self.starts_used else 0.0
    return self.args.junction_weight * jd + novelty + 0.5 * min(variety, 1.0) - 2.0 * c["familiar"] - 0.3 * reuse

  def next(self) -> dict | None:
    cands = []
    for _ in range(self.args.candidates * 15):
      c = self.candidate()
      if c is not None:
        cands.append(c)
        if len(cands) >= self.args.candidates:
          break
    if not cands:
      return None
    fresh = [c for c in cands if c["familiar"] < TOO_FAMILIAR] or cands
    best = max(fresh, key=self.score)
    self.n += 1
    best["id"] = f"{self.args.name}-{self.n:03d}"
    self.areas[self._area(*best["start"])] += 1
    self.starts_used.add(self._area(*best["start"]))
    self.classes.update(best["classes"])
    self.bins[self._bin(best["length"])] += 1
    for cell in best.pop("cells"):
      self.covered[cell] = 1.0
    return best


def est_seconds(c: dict) -> float:
  return EST_OVERHEAD + c["length"] / EST_SPEED


def old_routes(runs: str, exclude: str) -> list[list]:
  out = []
  for path in glob.glob(os.path.join(runs, "*.jsonl")):
    if path.endswith(".segments.jsonl") or os.path.abspath(path) == os.path.abspath(exclude):
      continue
    for r in read_jsonl(path):
      if (r.get("route") or {}).get("geom"):
        out.append(r["route"]["geom"])
  return out


def read_jsonl(path: str) -> list[dict]:
  out = []
  try:
    with open(path) as f:
      for line in f:
        try:
          out.append(json.loads(line))
        except ValueError:
          pass
  except OSError:
    pass
  return out


def plan_text(routes: list[dict]) -> str:
  if not routes:
    return "no trips"
  lens = np.array([r["length"] for r in routes])
  secs = np.array([est_seconds(r) for r in routes])
  classes = Counter(k for r in routes for k in r["classes"])
  areas = Counter((r["area"], int(r["start"][0] // AREA_CELL), int(r["start"][1] // AREA_CELL)) for r in routes)
  jkm = np.array([r["junctions"] / (r["length"] / 1000) for r in routes])
  return "\n".join([
    f"{len(routes)} trips: length {lens.min():.0f}-{lens.max():.0f} m (median {np.median(lens):.0f}), " +
    f"{np.mean([len(r['classes']) for r in routes]):.1f} maneuvers and {np.mean([r['junctions'] for r in routes]):.1f} " +
    f"junctions each ({np.median(jkm):.1f}/km median)",
    f"about {np.mean(secs) / 60:.1f} min each ({EST_SPEED:.0f} m/s + {EST_OVERHEAD:.0f} s): {3600 / np.mean(secs):.1f} trips/hour, " +
    f"{3600 / np.mean(secs) * np.mean(lens) / 1000:.1f} km/hour",
    f"start areas: {len(areas)} different {AREA_CELL:.0f} m cells ({sum(1 for k in areas if k[0] == 'map')} outside the city), " +
    f"busiest {areas.most_common(1)[0][1]} trips; road driven before: {np.mean([r['familiar'] for r in routes]):.2f} mean",
    "maneuvers: " + ", ".join(f"{k} {n}" for k, n in classes.most_common()),
  ])


# *** the bridge and the game ***

class Watch:
  """The bridge's game state (GTA5_LOG), expert mode's rows and the bridge's console, as they're written."""
  def __init__(self, state_log: str, expert_log: str, bridge_log: str):
    self.states, self.rows_tail, self.console = e2e.Tail(state_log), e2e.Tail(expert_log), e2e.Tail(bridge_log)
    self.state: dict = {}
    self.state_t = time.monotonic()  # grace until the first
    self.rows: list[dict] = []  # since last taken
    self.connected: bool | None = None
    self.disconnected_t: float | None = None
    self.arrived = False
    self.recorded: list[tuple] = []  # (wall time, name, frames, ended), since last taken
    self.record_error: str | None = None
    self.last_recorded = time.monotonic()

  def restart_logs(self):
    """After a bridge restart: its console is a new file."""
    self.console.pos, self.console.buf = 0, b""

  def update(self):
    now = time.monotonic()
    last = None
    for line in self.states.lines():
      if '"state"' in line:
        last = line
    if last is not None:
      try:
        s = json.loads(last)["state"]
        self.state, self.state_t = s, now
      except (ValueError, KeyError):
        pass
    for line in self.rows_tail.lines():
      try:
        r = json.loads(line)
      except ValueError:
        continue
      self.rows.append(r)
    for line in self.console.lines():
      if "gta5: game connected" in line:
        self.connected, self.disconnected_t = True, None
      elif "gta5: game disconnected" in line:
        self.connected, self.disconnected_t = False, now
      elif "gta5: expert arrived" in line:
        self.arrived = True
      elif (m := RECORDED.search(line)):
        self.recorded.append((time.monotonic(), m.group(1), int(m.group(2)), m.group(3)))
        self.last_recorded = time.monotonic()
      elif "gta5: recording stopped" in line or ("gta5: recording:" in line and "failed" in line):
        self.record_error = line.strip()

  def take_rows(self) -> list[dict]:
    r, self.rows = self.rows, []
    return r

  def stale(self) -> bool:
    return time.monotonic() - self.state_t > STALE


def bridge_pids() -> list[int]:
  r = subprocess.run(["pgrep", "-f", "[r]un_bridge.py"], capture_output=True, text=True, check=False)
  return [int(v) for v in r.stdout.split()]


def bridge_env() -> dict:
  for pid in bridge_pids():
    try:
      with open(f"/proc/{pid}/environ", "rb") as f:
        env = dict(kv.split("=", 1) for kv in f.read().decode(errors="ignore").split("\0") if "=" in kv)
      if "OPENPILOT_PREFIX" in env:
        return env
    except OSError:
      pass
  return {}


def disk_free(paths: list[str]) -> dict:
  """GB free on each path's filesystem (its nearest existing folder's, before the bridge makes it)."""
  out = {}
  for p in paths:
    q = p
    while q and not os.path.exists(q):
      q = os.path.dirname(q)
    out[p] = round(shutil.disk_usage(q or "/").free / 1e9, 1)
  return out


def dir_bytes(path: str) -> int:
  total = 0
  for root, _, files in os.walk(path):
    for f in files:
      try:
        total += os.path.getsize(os.path.join(root, f))
      except OSError:
        pass
  return total


def file_bytes(path: str) -> int:
  try:
    return os.path.getsize(path)
  except OSError:
    return 0


def smooth_metrics(rows: list[dict]) -> str | None:
  """ai_smooth.py's line for the drive, if it can be had."""
  try:
    sys.path.insert(0, os.path.dirname(SMOOTH))
    import ai_smooth  # type: ignore
  except Exception:
    return None
  finally:
    if sys.path and sys.path[0] == os.path.dirname(SMOOTH):
      sys.path.pop(0)
  try:
    ds = ai_smooth.drives(rows)
    return ai_smooth.summary(max(ds, key=len)) if ds else None
  except Exception as e:
    return f"error: {e}"


class Stop(Exception):
  """Ends the run."""


class Run:
  def __init__(self, args):
    self.args = args
    self.extra = parse_extra(args.bridge_extra)
    self.expert_env = {**os.environ, "GTA5_EXPERT": self.extra.get("GTA5_EXPERT", "1")}
    rec_dir = self.extra.get("GTA5_RECORD", f"{T}/rec")
    self.rec_dir = rec_dir
    self.out = os.path.join(args.runs, f"{args.name}.jsonl")
    self.seg_out = os.path.join(args.runs, f"{args.name}.segments.jsonl")
    self.status_path = os.path.join(args.runs, "status.json")
    self.state_log = self.extra.get("GTA5_LOG", f"{T}/bridge.jsonl")
    self.expert_log = self.extra.get("GTA5_EXPERT_LOG", os.path.join(os.path.dirname(self.state_log), "expert.jsonl"))
    self.watch = Watch(self.state_log, self.expert_log, args.bridge_log)
    self.stop_asked: str | None = None
    self.windows: list[tuple[float, str]] = []  # (wall time, trip id): when each trip started, for its segments
    self.recs: list[dict] = []
    self.restarts = 0
    self.fail_streak = 0
    self.warned_randomise = False
    self.segments: list[dict] = []
    # the recordings' drive, and for the logs WSL's own disk, a file growing on the Windows drive (--host-disk) that it
    # reports far more free space than
    self.limits = {rec_dir: args.min_free_gb, "/": args.min_free_logs_gb}
    if args.host_disk:
      self.limits[args.host_disk] = min(args.min_free_logs_gb, self.limits.get(args.host_disk, math.inf))
    self.disks = list(self.limits)

  # *** stopping ***

  def on_signal(self, signum, frame):
    if self.stop_asked:
      raise KeyboardInterrupt
    self.stop_asked = signal.Signals(signum).name
    say(f"record: {self.stop_asked}: stopping (again to stop at once)")

  def check_stop(self):
    if self.stop_asked is None and os.path.exists(self.args.stop_file):
      self.stop_asked = f"stop file {self.args.stop_file}"
      say(f"record: {self.stop_asked}")
    return self.stop_asked

  def check_disk(self) -> dict:
    free = disk_free(self.disks)
    low = {p: g for p, g in free.items() if g < self.limits[p]}
    if low:
      raise Stop(f"low disk: {low} GB free, under {[self.limits[p] for p in low]}")
    return free

  # *** the bridge ***

  def bridge_ok(self) -> str | None:
    """What's wrong with the bridge or the game, if anything."""
    w = self.watch
    w.update()
    if not bridge_pids():
      return "bridge died"
    if w.record_error:
      return f"recording failed: {w.record_error}"
    if w.connected is False:
      return "plugin disconnected"
    if w.stale():
      return "no game state"
    return None

  def restart_bridge(self, why: str):
    self.restarts += 1
    if self.restarts > self.args.max_restarts:
      raise Stop(f"{why}, and the bridge was restarted {self.args.max_restarts} times already")
    say(f"record: restarting the bridge ({why}), BRIDGE_EXTRA={self.args.bridge_extra!r}")
    self.expert("off", quiet=True)
    subprocess.run(["bash", SVC, "restart", "bridge"], env={**os.environ, "BRIDGE_EXTRA": self.args.bridge_extra}, check=False)
    time.sleep(2)
    w = self.watch
    w.restart_logs()
    w.connected, w.record_error, w.last_recorded = None, None, time.monotonic()
    end = time.monotonic() + BRIDGE_UP_WAIT
    while time.monotonic() < end:
      w.update()
      if w.connected and not w.stale():
        say("record: the bridge is back and the game connected")
        time.sleep(3)
        return
      time.sleep(1)
    self.wait_plugin("the restarted bridge's game connection")

  def wait_plugin(self, what: str):
    say(f"record: waiting up to {self.args.plugin_wait:.0f} s for {what}")
    w = self.watch
    end = time.monotonic() + self.args.plugin_wait
    while time.monotonic() < end:
      if self.check_stop():
        raise Stop(self.stop_asked)
      w.update()
      if w.connected is not False and not w.stale():
        say("record: the game is back")
        return
      time.sleep(2)
    # reloading the plugin (touching its DLL) isn't safe to do unattended: a person has to look
    raise Stop(f"{what}: none for {self.args.plugin_wait:.0f} s; reload the plugin by hand")

  def recover(self, problem: str):
    say(f"record: {problem}")
    if problem == "plugin disconnected":
      self.wait_plugin("the plugin to reconnect")
    elif problem == "no game state":
      # the bridge runs and the game's connected but nothing comes: a hang, or the game paused or loading
      end = time.monotonic() + HUNG_WAIT
      while time.monotonic() < end and self.watch.stale():
        self.watch.update()
        time.sleep(1)
      if self.watch.stale():
        self.restart_bridge("no game state")
    else:
      self.restart_bridge(problem)

  def ensure_bridge(self):
    env = bridge_env()
    want = {k: v for k, v in self.extra.items() if k in ("GTA5_EXPERT", "GTA5_RECORD")}
    wrong = {k: env.get(k) for k, v in want.items() if env.get(k) != v}
    if not wrong:
      return
    if not self.args.fix_bridge:
      raise Stop(f"the bridge runs without {want} (it has {wrong}): run with --fix-bridge, or " +
                 f"BRIDGE_EXTRA={shlex.quote(self.args.bridge_extra)} {SVC} restart bridge")
    self.restarts -= 1  # not a failure
    self.restart_bridge(f"it runs without {want}")

  def gta5_cmd(self, *argv, quiet: bool = False, timeout: float = 40.0) -> subprocess.CompletedProcess | None:
    try:
      r = subprocess.run([sys.executable, GTA5_CMD, *argv], env=self.expert_env, capture_output=True, text=True,
                         timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
      if not quiet:
        say(f"record: gta5_cmd {' '.join(argv[:2])} timed out")
      return None
    if r.returncode and not quiet:
      say(f"record: gta5_cmd {' '.join(argv[:2])}: {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
    return r

  def expert(self, *argv, quiet: bool = False) -> subprocess.CompletedProcess | None:
    return self.gta5_cmd("expert", *argv, quiet=quiet)

  # *** a trip ***

  def drive(self, c: dict, settings: list[str], world: dict) -> dict:
    a, w = self.args, self.watch
    rec = {"id": c["id"], "run": a.name, "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "spec": c["spec"], "area": c["area"],
           "route": {k: c[k] for k in ("length", "time", "maneuvers", "classes", "junctions", "familiar", "geom")},
           "settings": " ".join(settings), "traffic": a.traffic, "world": world or None}
    timeout = a.timeout_factor * (TIMEOUT_BASE + max(c["length"] / TIMEOUT_SPEED, TIMEOUT_ROUTE * c["time"]))
    rec["timeout"] = round(timeout)
    self.windows.append((time.monotonic(), c["id"]))

    def done(outcome: str, detail: str = "", **kw) -> dict:
      rec.update({"outcome": outcome, "detail": detail, **kw})
      return rec

    if a.randomise:
      if randomise_supported():
        seed = a.seed * 1000 + len(self.windows)  # each trip's own, repeatable from the run's
        r = self.gta5_cmd("randomise", str(seed), *a.randomise_args.split())
        rec["randomise_seed"] = seed
        rec["randomise"] = None if r is None else (r.stdout + r.stderr).strip()[-2000:]
      elif not self.warned_randomise:
        self.warned_randomise = True
        say("record: gta5_cmd.py has no randomise yet: trips go without it")
    try:
      if a.traffic != "keep":
        e2e.cmd("traffic", on=int(a.traffic == "on"))
      if c.get("traffic"):  # the trip's own, over randomise's and --traffic's (a wrong-way clip's light traffic)
        e2e.cmd("traffic", **c["traffic"])
      if world:
        e2e.cmd("world", **world, freeze=1)
    except OSError as e:
      return done("infra", f"the bridge's debug port: {e}")
    w.update()
    w.take_rows()
    w.arrived = False
    t_cmd = time.monotonic()
    r = self.expert("route", c["spec"], *settings, *c.get("expert_extra", []))
    if r is None or r.returncode:
      return done("setup", "the route command failed" + (f": {r.stderr.strip()[-200:]}" if r is not None else ""))
    t0 = time.monotonic()
    sx, sy = c["start"]
    placed = route_seen = False
    rows: list[dict] = []
    progress_t, anchor = t0, None
    distance, last_pos = 0.0, None
    off_t = onc_t = None
    onc_max = 0.0
    coll_events, coll_last, coll_prev = 0, -1e9, 0
    reroutes, last_end = 0, None
    oncoming_s = c.get("oncoming_s", a.oncoming_s)
    ww_events: list[dict] = []
    outcome = None
    while outcome is None:
      time.sleep(TICK)
      now = time.monotonic()
      w.update()
      new = [row for row in w.take_rows() if row.get("mono", 0) >= t_cmd - 1]
      rows += [row for row in new if "event" not in row]
      ww_events += [row for row in new if row.get("event") == "wrongway"]
      if self.check_stop():
        outcome = ("stopped", self.stop_asked)
        break
      problem = self.bridge_ok()
      if problem:
        outcome = ("infra", problem)
        break
      stops = [row.get("why", "") for row in new if row.get("event") == "stop"]
      if any("engage key" in why for why in stops):
        outcome = ("took_over", "the driver pressed the engage key")
        break
      if stops:
        outcome = ("expert_stopped", stops[-1])
        break
      pos = w.state.get("pos")
      if not placed:
        if pos and math.hypot(pos[0] - sx, pos[1] - sy) < PLACED_NEAR:
          placed, anchor, last_pos = True, pos[:2], pos[:2]
          progress_t = now
        elif now - t0 > PLACE_WAIT:
          outcome = ("teleport", f"at {[round(v) for v in pos[:2]] if pos else None}, not the start")
          break
        continue
      live = [row for row in new if "event" not in row and row.get("active")]
      if not route_seen:
        if any(row.get("routeAt") is not None for row in live):
          route_seen = True
          progress_t = now
        elif now - t0 > ROUTE_WAIT:
          outcome = ("no_route", "the expert didn't get our route")
          break
        continue
      if pos:
        step = math.hypot(pos[0] - last_pos[0], pos[1] - last_pos[1])
        if step < 20:  # not a teleport
          distance += step
        last_pos = pos[:2]
        if math.hypot(pos[0] - anchor[0], pos[1] - anchor[1]) > PROGRESS:
          anchor, progress_t = pos[:2], now
      if w.arrived or any(row.get("arrived") for row in live):
        outcome = ("arrived", "")
        break
      if now - t0 > timeout:
        outcome = ("timeout", f"{live[-1].get('routeEnd') if live else '?'} m left")
        break
      if now - progress_t > a.stuck_s:
        outcome = ("stuck", f"under {PROGRESS:.0f} m in {a.stuck_s:.0f} s")
        break
      if live:
        last = live[-1]
        coll = int(last.get("collisions") or 0)
        if coll > coll_prev:
          if now - coll_last > COLLISION_GAP:
            coll_events += 1
          coll_last = now
        coll_prev = coll
        if coll > a.max_collision_frames or coll_events > a.max_collision_events:
          outcome = ("collisions", f"{coll_events} collisions, {coll} frames in contact")
          break
        moving_onc = bool(last.get("oncoming")) and (last.get("vEgo") or 0) > 1.0
        onc_t = (onc_t or now) if moving_onc else None
        if onc_t:
          onc_max = max(onc_max, now - onc_t)
          if now - onc_t > oncoming_s:
            outcome = ("oncoming", f"{now - onc_t:.0f} s in the oncoming lanes on {last.get('street')}")
            break
        off = last.get("routeOff")
        lost = last.get("routeAt") is None
        off_t = (off_t or now) if lost or (off is not None and off > OFF_ROUTE) else None
        if off_t and now - off_t > a.off_route_s:
          outcome = ("off_route", "no route" if lost else f"{off:.0f} m off the route")
          break
        end = last.get("routeEnd")
        if end is not None and last_end is not None and end > last_end + REROUTE_JUMP:
          reroutes += 1
          if reroutes > MAX_REROUTES:
            outcome = ("reroutes", f"{reroutes} new routes")
            break
        if end is not None:
          last_end = end
      if time.monotonic() - w.last_recorded > RECORD_STALL:
        outcome = ("infra", f"no segment recorded for {RECORD_STALL:.0f} s")
        break
    self.expert("off", quiet=True)
    w.update()
    ww_events += [row for row in w.take_rows() if row.get("event") == "wrongway"]
    if ww_events:
      rec["wrongway_phases"] = ww_events
    act = [row for row in rows if row.get("active")]
    v = [row.get("vEgo") or 0.0 for row in act]
    ts = [row.get("t") or row["mono"] for row in act]
    return done(*outcome, duration=round(time.monotonic() - t0, 1), drive_s=round(ts[-1] - ts[0], 1) if len(ts) > 1 else 0.0,
                distance=round(distance), mean_v=round(float(np.mean(v)), 2) if v else None,
                collisions=max((int(row.get("collisions") or 0) for row in act), default=0), collision_events=coll_events,
                oncoming_frames=sum(bool(row.get("oncoming")) for row in act), oncoming_max=round(onc_max, 1),
                reroutes=reroutes, smooth=smooth_metrics(rows) if act else None)

  # *** segments ***

  def take_segments(self) -> list[str]:
    """Writes the segments recorded since last time, each with the trip whose time its frames' middle falls in."""
    out = []
    for wall, name, frames, ended in self.watch.recorded:
      mid = wall - 3.0 - frames / FRAME_HZ / 2  # the line comes a few s after its last frame
      trip = next((tid for t, tid in reversed(self.windows) if t <= mid), None)
      seg = {"segment": name, "trip": trip, "frames": frames, "ended": ended, "logged": time.strftime("%H:%M:%S")}
      self.segments.append(seg)
      with open(self.seg_out, "a") as f:
        f.write(json.dumps(seg) + "\n")
      out.append((trip, name))
    self.watch.recorded = []
    return [n for tid, n in out if self.recs and tid == self.recs[-1]["id"]] if out else []

  # *** the run ***

  def status(self, **kw):
    counts = Counter(r["outcome"] for r in self.recs)
    with open(self.status_path + ".tmp", "w") as f:
      json.dump({"name": self.args.name, "at": time.strftime("%H:%M:%S"), "trips": len(self.recs), "counts": counts,
                 "segments": len(self.segments), "restarts": self.restarts, **kw}, f)
    os.replace(self.status_path + ".tmp", self.status_path)

  def run(self, picker: Picker) -> str:
    """Drives picker's trips until the hours are up or something stops it (a Stop raised in picker.next too); why."""
    a = self.args
    started, start_mono = time.strftime("%Y-%m-%dT%H:%M:%S"), time.monotonic()
    end_mono = start_mono + a.hours * 3600
    free0 = disk_free(self.disks)
    rec0, logs0 = dir_bytes(self.rec_dir), file_bytes(self.state_log) + file_bytes(self.expert_log)
    say(f"record: run {a.name} (seed {a.seed}) for {a.hours:g} h -> {self.out}; free GB {free0}; settings from " +
        f"{'--settings' if a.settings is not None else a.settings_file}")
    why = "done"
    rng = random.Random(a.seed + 1)
    settings_text = None
    pool = ThreadPoolExecutor(1)
    upcoming = pool.submit(picker.next)
    try:
      self.check_disk()
      self.ensure_bridge()
      while True:
        if self.check_stop():
          why = self.stop_asked
          break
        if time.monotonic() >= end_mono:
          why = f"{a.hours:g} h up"
          break
        free = self.check_disk()
        if self.fail_streak >= a.max_fail_streak:
          raise Stop(f"{self.fail_streak} trips failed in a row")
        problem = self.bridge_ok()
        if problem:
          self.recover(problem)
          continue
        c, waited = upcoming.result(), 0.0
        upcoming = pool.submit(picker.next)  # picked while this one's driven, as picking can take a while (Valhalla)
        while c is None:
          if waited >= ROUTER_WAIT:
            raise Stop(f"no trips from Valhalla ({e2e.VALHALLA}) for {ROUTER_WAIT:.0f} s")
          say(f"record: no trip picked (is Valhalla up at {e2e.VALHALLA}?); retrying")
          time.sleep(30)
          waited += 30
          if self.check_stop():
            raise Stop(self.stop_asked)
          c = picker.next()
        settings, bad = read_settings(a)
        if " ".join(settings + bad) != settings_text:
          settings_text = " ".join(settings + bad)
          say(f"record: expert settings: {' '.join(settings) or '(defaults)'}" + (f"; not expert options, ignored: {bad}" if bad else ""))
        world = {}
        if a.hours_of_day:
          world["hour"] = int(rng.choice(a.hours_of_day.split(",")))
        if a.weathers:
          world["weather"] = rng.choice(a.weathers.split(","))
        say(f"record: trip {c['id']} {c['spec']} {c['area']} {c['length']} m, {c['junctions']} junctions: {', '.join(c['classes'])}")
        self.status(trip=c["id"], free_gb=free, elapsed_h=round((time.monotonic() - start_mono) / 3600, 2))
        rec = self.drive(c, settings, world)
        self.recs.append(rec)
        rec["segments"] = self.take_segments()
        rec["free_gb"] = disk_free(self.disks)
        with open(self.out, "a") as f:
          f.write(json.dumps(rec) + "\n")
        say(f"record: {rec['id']}: {rec['outcome']} {rec.get('detail', '')} in {rec.get('duration')} s, " +
            f"{rec.get('distance')} m, {rec.get('collisions')} contact frames, segments {rec['segments']}")
        if not rec.get("wrongway"):  # a wrong-way clip's outcome says nothing about the run's health
          self.fail_streak = 0 if rec["outcome"] == "arrived" else self.fail_streak + 1
        if rec["outcome"] == "took_over":
          raise Stop("the driver took over (engage key)")
        if rec["outcome"] == "stopped":
          why = self.stop_asked
          break
        if rec["outcome"] == "infra":
          problem = self.bridge_ok() or (rec["detail"] if "segment" in rec["detail"] else None)
          if problem:
            self.recover(problem)
        time.sleep(COOLDOWN)
    except Stop as e:
      why = str(e)
      say(f"record: stopping: {why}")
    except KeyboardInterrupt:
      why = "interrupted"
    finally:
      pool.shutdown(wait=False, cancel_futures=True)
      self.expert("off", quiet=True)
      self.gta5_cmd("ai", "off", quiet=True)
      time.sleep(3)
      self.watch.update()
      self.take_segments()
      extra = {"why": why, "hours": round((time.monotonic() - start_mono) / 3600, 2), "started": started,
               "free_gb_start": free0, "free_gb_end": disk_free(self.disks),
               "rec_gb": round((dir_bytes(self.rec_dir) - rec0) / 1e9, 2),
               "logs_gb": round((file_bytes(self.state_log) + file_bytes(self.expert_log) - logs0) / 1e9, 2),
               "restarts": self.restarts}
      s, text = summarize(self.recs, self.segments, extra)
      base = os.path.join(a.runs, a.name)
      with open(base + ".summary.json", "w") as f:
        json.dump(s, f, indent=1)
      with open(base + ".summary.txt", "w") as f:
        f.write(text + "\n")
      say("\n" + text)
      self.status(done=True, why=why)
    return why


def summarize(recs: list[dict], segs: list[dict], extra: dict | None = None) -> tuple[dict, str]:
  extra = extra or {}
  counts = Counter(r.get("outcome") for r in recs)
  ok = [r for r in recs if r.get("outcome") == "arrived"]
  by_trip = defaultdict(list)
  for sg in segs:
    by_trip[sg.get("trip")].append(sg)
  outcome = {r["id"]: r.get("outcome") for r in recs}
  seg_by_outcome = Counter()
  for tid, ss in by_trip.items():
    seg_by_outcome[outcome.get(tid, "between trips")] += sum(sg["frames"] for sg in ss)
  classes = Counter(k for r in ok for k in (r.get("route") or {}).get("classes", []))
  s = {
    "trips": len(recs), "outcomes": dict(counts), "arrived_rate": round(len(ok) / len(recs), 3) if recs else None,
    "drive_h": round(sum(r.get("drive_s") or 0 for r in recs) / 3600, 2), "km": round(sum(r.get("distance") or 0 for r in recs) / 1000, 1),
    "arrived_km": round(sum(r.get("distance") or 0 for r in ok) / 1000, 1),
    "segments": len(segs), "recorded_min": round(sum(sg["frames"] for sg in segs) / FRAME_HZ / 60, 1),
    "recorded_min_by_outcome": {k: round(v / FRAME_HZ / 60, 1) for k, v in seg_by_outcome.items()},
    "maneuvers_arrived": dict(classes.most_common()),
    "failures": [{"id": r["id"], "outcome": r.get("outcome"), "detail": r.get("detail")} for r in recs if r.get("outcome") != "arrived"],
    **extra,
  }
  lines = [f"run summary: {s['trips']} trips, {s['arrived_rate'] if recs else '-'} arrived; outcomes {s['outcomes']}",
           f"  {s['drive_h']} h driving, {s['km']} km ({s['arrived_km']} km on trips that arrived)",
           f"  {s['segments']} segments, {s['recorded_min']} min recorded; by trip outcome {s['recorded_min_by_outcome']}",
           f"  maneuvers on arrived trips: {s['maneuvers_arrived']}"]
  if extra:
    lines.append(f"  stopped: {extra.get('why')} after {extra.get('hours')} h; recordings +{extra.get('rec_gb')} GB, logs " +
                 f"+{extra.get('logs_gb')} GB; free GB {extra.get('free_gb_start')} -> {extra.get('free_gb_end')}; " +
                 f"bridge restarts {extra.get('restarts')}")
  for f in s["failures"][-15:]:
    lines.append(f"  {f['id']}: {f['outcome']} {f['detail']}")
  return s, "\n".join(lines)


# *** commands ***

def cmd_run(args):
  global LOG
  os.makedirs(args.runs, exist_ok=True)
  args.name = args.name or time.strftime("rec-%m%d-%H%M")
  if args.seed is None:
    args.seed = random.SystemRandom().randrange(100000)
  print(f"seed {args.seed}")
  if args.router:
    e2e.VALHALLA = args.router
  else:
    router = parse_extra(args.bridge_extra).get("GTA5_ROUTER")
    if router:
      e2e.VALHALLA = router  # the bridge's router, for routes it would make too
  roads = e2e.Map()
  out = os.path.join(args.runs, f"{args.name}.jsonl")
  picker = Picker(roads, args, [] if args.no_avoid else old_routes(args.runs, out))
  if args.dry_run:
    routes = []
    for _ in range(args.n):
      c = picker.next()
      if c is None:
        print(f"no trip picked: is Valhalla up at {e2e.VALHALLA}?")
        break
      routes.append(c)
      print(f"{c['id']} {c['spec']:48s} {c['area']:4s} {c['length']:5d} m {est_seconds(c) / 60:4.1f} min " +
            f"{c['junctions']:2d} junctions  {', '.join(c['classes'])}", flush=True)
    print(plan_text(routes))
    print(f"expert settings: {' '.join(read_settings(args)[0]) or '(defaults)'}")
    return
  lock = os.path.join(args.runs, "run.lock")
  try:
    with open(lock) as f:
      pid = int(f.read().strip() or 0)
    if pid and os.path.exists(f"/proc/{pid}"):
      sys.exit(f"another run is going (pid {pid}, {lock})")
  except (OSError, ValueError):
    pass
  with open(lock, "w") as f:
    f.write(str(os.getpid()))
  if os.path.exists(args.stop_file):
    print(f"removing an old stop file {args.stop_file}")
    os.remove(args.stop_file)
  LOG = open(os.path.join(args.runs, f"{args.name}.log"), "a")
  run = Run(args)
  signal.signal(signal.SIGINT, run.on_signal)
  signal.signal(signal.SIGTERM, run.on_signal)
  try:
    run.run(picker)
  finally:
    try:
      os.remove(lock)
    except OSError:
      pass


def cmd_summary(args):
  for path in args.files:
    recs = [r for r in read_jsonl(path) if "outcome" in r]
    segs = read_jsonl(path.removesuffix(".jsonl") + ".segments.jsonl")
    summary_json = path.removesuffix(".jsonl") + ".summary.json"
    extra = {}
    if os.path.exists(summary_json):
      with open(summary_json) as f:
        extra = {k: v for k, v in json.load(f).items() if k in ("why", "hours", "rec_gb", "logs_gb", "free_gb_start",
                                                                "free_gb_end", "restarts")}
    print(f"{path}:\n{summarize(recs, segs, extra)[1]}")


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest="command", required=True)
  r = sub.add_parser("run")
  r.add_argument("--dry-run", action="store_true", help="only pick and print --n trips")
  r.add_argument("--n", type=int, default=30, help="trips for --dry-run")
  r.add_argument("--lane", default="auto", help="start lane from the left (9: the rightmost); auto: leftmost when the " +
                 "first maneuver is a left, else rightmost")
  r.add_argument("--min-len", type=float, default=500.0, help="m of route")
  r.add_argument("--max-len", type=float, default=3000.0)
  r.add_argument("--map-share", type=float, default=0.15, help="of trips that start anywhere on the map, not in the city")
  r.add_argument("--candidates", type=int, default=6, help="trips picked for each one driven, the best of them kept")
  r.add_argument("--junction-weight", type=float, default=1.0, help="how much junctions per km count in picking")
  r.add_argument("--no-avoid", action="store_true", help="don't avoid road driven in earlier runs")
  run_arguments(r)
  sm = sub.add_parser("summary")
  sm.add_argument("files", nargs="+")
  args = p.parse_args()
  {"run": cmd_run, "summary": cmd_summary}[args.command](args)


def run_arguments(r: argparse.ArgumentParser):
  """A run's options for driving, recording and recovering (junction_run.py's too)."""
  r.add_argument("--hours", type=float, default=8.0)
  r.add_argument("--name", help="the run's name (default rec-MMDD-HHMM)")
  r.add_argument("--seed", type=int, help="for the trips (default a random one, logged)")
  r.add_argument("--settings", help="the expert's k=v options, e.g. 'task=coord speed=10' (else --settings-file's)")
  r.add_argument("--settings-file", default=SETTINGS_FILE, help="k=v options, # comments; read before each trip")
  r.add_argument("--randomise", action="store_true", help="gta5_cmd.py randomise <seed> before each trip (the scene, car and " +
                 "camera mount), if it has it")
  r.add_argument("--randomise-args", default="", help="for it, e.g. model_share=0.6")
  r.add_argument("--traffic", choices=["on", "off", "keep"], default="keep", help="the plugin's traffic, set each trip")
  r.add_argument("--hours-of-day", default="", help="e.g. 8,12,17: each trip's time of day, picked from these")
  r.add_argument("--weathers", default="", help="e.g. EXTRASUNNY,CLEAR,CLOUDS: each trip's weather, picked from these")
  r.add_argument("--stuck-s", type=float, default=120.0, help=f"s without moving {PROGRESS:.0f} m")
  r.add_argument("--oncoming-s", type=float, default=10.0, help="s moving in the oncoming lanes")
  r.add_argument("--off-route-s", type=float, default=15.0, help=f"s more than {OFF_ROUTE:.0f} m off the route")
  r.add_argument("--max-collision-frames", type=int, default=400, help="frames in contact in a trip")
  r.add_argument("--max-collision-events", type=int, default=6, help="separate collisions in a trip")
  r.add_argument("--timeout-factor", type=float, default=1.0, help="x each trip's timeout")
  r.add_argument("--max-fail-streak", type=int, default=8, help="trips failing in a row that end the run")
  r.add_argument("--max-restarts", type=int, default=5, help="bridge restarts that end the run")
  r.add_argument("--plugin-wait", type=float, default=600.0, help="s the game may stay disconnected")
  r.add_argument("--min-free-gb", type=float, default=50.0, help="on the recordings drive")
  r.add_argument("--min-free-logs-gb", type=float, default=20.0, help="on WSL's disk and --host-disk, for the logs")
  r.add_argument("--host-disk", default="/mnt/c", help="the Windows drive holding WSL's disk (ext4.vhdx)")
  r.add_argument("--bridge-extra", default=os.getenv("BRIDGE_EXTRA") or BRIDGE_EXTRA,
                 help="the bridge's extra environment (svc.sh's BRIDGE_EXTRA), for restarts and checks")
  r.add_argument("--fix-bridge", action="store_true", help="restart the bridge with --bridge-extra if it runs without it")
  r.add_argument("--bridge-log", default=f"{T}/bridge.log")
  r.add_argument("--router", default=os.getenv("GTA5_ROUTER"), help="Valhalla (default the bridge's GTA5_ROUTER, else :8002)")
  r.add_argument("--runs", default=f"{T}/recruns", help="the runs folder")
  r.add_argument("--stop-file", default=f"{T}/recruns/STOP")


if __name__ == "__main__":
  main()
