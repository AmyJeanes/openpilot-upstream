"""Expert mode: the game's own AI drives the car along our route, as expert driving to record for training the driving
model (~/gta5test/notes/ai_driver.md has the research).

Off unless the bridge runs with GTA5_EXPERT set: 1 for the control file at CONTROL, or the file's path. The bridge then
watches that file (gta5_cmd.py expert on|off|route writes it; changes from before the bridge started are ignored):
  {"on": true, "speed": 12, "style": 1076369579, "ability": 1, "aggr": 0, "task": "longrange", "limits": true,
   "need_route": false, "dest": null, "ahead_min": 60, "ahead_max": 120, "past": 25, "targets": "junction",
   "retarget_every": 0, "ramp": 0, "lead": 2, "launch": null, "decel": 0, "turn_speed": 6, "arrive": "task",
   "stop_before": 15, "speed_step": 0.5, "hold_after": 3}
While on, openpilot is kept disengaged and nav's cues wait. On our map's route (GTA5_ROUTER) the plugin's AI driver is
given a target 60-120 m ahead, just past the next junction or maneuver, so its own short pathfinding can only take the
route's way through it (targets=smooth: ahead_max on, moved only once ahead_min is left, at most every retarget_every
s); the indicators follow the route's turns, ramps and exits, and `label` holds the desire they stand for. Its speed cap
is `speed`, or the map's limit where lower; ramp (m/s^2) raises it from the car's speed (at most `lead` m/s above it,
`launch` from a standstill) rather than at once, decel (m/s^2) lowers it ahead of turns (to turn_speed) and lower
limits, and arrive=gentle brings it to a stop stop_before m short of the route's end and holds it there, rather than the
task's own arrival. A ramp wants task=coord, as the longrange task doesn't take a raised cap until it's re-tasked, and a
lead of about 3, as the AI settles 1-2 m/s under its cap. `dest` ignores a route that doesn't end near it, as the last
trip's. Off the map's routes the AI drives to the game's waypoint, or wanders. Each game state appends a JSON line to
the log (GTA5_EXPERT_LOG, else expert.jsonl beside GTA5_LOG, else /tmp/gta5_expert.jsonl): the AI's state, the target,
the label, the lane readings and the drive's collisions (frames in contact since it started), so samples can be
filtered later. speed_by_class ("residential:10,primary:14,motorway:18", classes as Valhalla names them, ramp for
ramps, default for the rest) caps the speed by the road's class, slowing ahead for a lower one, and decel_fast is the
slowing rate above 12 m/s. unstick: stopped unstick_after s not at a light, the AI drives with unstick_style
(steering round a parked car) until it has moved unstick_dist m or unstick_for s have passed. The engage key stops
the AI and expert mode; so does arriving, hold_after s after stopping, and the control file is set off. With expert
mode off, the bridge turns off a plugin AI driver left on (always without GTA5_EXPERT, else once the bridge starts or
while openpilot is engaged)."""
import json
import math
import os
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np

from openpilot.selfdrive.navd.maneuvers import Maneuver, maneuvers, route_point

CONTROL = Path("/tmp/gta5_expert.json")
POLL_EVERY = 0.5  # s
DEFAULTS = {"on": False, "speed": 12.0, "style": 1076369579, "ability": 1.0, "aggr": 0.0, "task": "longrange", "limits": True,
            "need_route": False, "dest": None, "ahead_min": 60.0, "ahead_max": 120.0, "past": 25.0, "targets": "junction",
            "retarget_every": 0.0, "ramp": 0.0, "lead": 2.0, "launch": None, "decel": 0.0, "turn_speed": 6.0,
            "arrive": "task", "stop_before": 15.0, "speed_step": 0.5, "hold_after": 3.0,
            "speed_by_class": None, "decel_fast": 0.0, "unstick": False, "unstick_after": 10.0,
            "unstick_style": 1076369579, "unstick_dist": 30.0, "unstick_for": 20.0}
STANDSTILL = 0.5  # m/s, below which a ramped cap starts from `launch`
DEST_NEAR = 50.0  # m from the destination asked for, the end of a route for it
LIMIT_LOOKAHEAD = 600.0  # m, lower speed limits ahead slowed for
FAST = 12.0  # m/s, above which decel_fast is the slowing rate
ROUTER = os.getenv("GTA5_ROUTER")  # Valhalla, for the road classes along a route
HOLD_SPEED = 0.01  # m/s: the AI pulls up and waits in its lane, as in a queue
ARRIVE_DECEL = 1.0  # m/s^2, arrive=gentle without decel
EVENT_MIN = 8.0  # m ahead: a junction or maneuver nearer than this is being driven through
BEFORE_EVENT = 10.0  # m: a straight target that would land in a junction stays this far short of it
RETARGET_NEAR = 30.0  # m from the target, whatever it's for
STOP_RANGE = 5.0  # m, for a target on the way
FINAL_STOP = 8.0  # m, the route's end
ARRIVED = 15.0  # m from the route's end, stopped
REQUEST_EVERY = 2.0  # s between asking the plugin again for what its state doesn't show
SPEED_STEP = 0.5  # m/s change worth sending
# desires (their maneuvers: navd/maneuvers.py)
SIGNAL_TIME, SIGNAL_DIST = 5.0, 50.0  # s, m: signal from the later of these before a maneuver
TURN_DONE_HEADING = 20.0  # deg from the way out
TURN_DONE_AFTER = 5.0  # m past it, at least
TURN_PAST = 40.0  # m past it, at the latest
KEEP_PAST = 30.0  # m
# Valhalla maneuver types
NOT_EVENTS = {0, 1, 2, 3, 4, 5, 6}  # start and destination


def control_path() -> Path | None:
  v = os.getenv("GTA5_EXPERT")
  if not v or v == "0":
    return None
  return CONTROL if v == "1" else Path(v)


def log_path() -> str:
  if os.getenv("GTA5_EXPERT_LOG"):
    return os.environ["GTA5_EXPERT_LOG"]
  if os.getenv("GTA5_LOG"):
    return str(Path(os.environ["GTA5_LOG"]).with_name("expert.jsonl"))
  return "/tmp/gta5_expert.jsonl"


def wrap(deg: float) -> float:
  return (deg + 180) % 360 - 180


class Expert:
  def __init__(self, send, cancel_engagement, clear_nav_desire):
    self.send = send
    self.cancel_engagement = cancel_engagement
    self.clear_nav_desire = clear_nav_desire
    self.path = control_path()
    self.started = time.time()  # noqa: TID251  # compared with the control file's mtime, a wall clock time
    self.mtime = 0.0
    self.next_poll = 0.0
    self.cfg = dict(DEFAULTS)
    self.on = False  # asked for and not stopped since
    self.active = False  # driving: the plugin was asked to
    self.last_t = None
    self.game_t = 0.0
    self.next_cancel = 0.0
    self.log = None
    self.warned = False
    self.seen_ai = False  # a state with the plugin's AI driver in it, since the bridge started
    self.next_guard = 0.0
    self._reset()
    if self.path is not None:
      print(f"gta5: expert mode watches {self.path}", flush=True)

  def _reset(self):
    self.route = None
    self.events: list[float] = []
    self.mans: list[Maneuver] = []
    self.done: set[int] = set()
    self.target: np.ndarray | None = None
    self.target_along = 0.0
    self.anchor: float | None = None  # the junction or maneuver the target is just past
    self.final = False
    self.sent_target: tuple | None = None
    self.sent_speed = -1.0
    self.label = "none"
    self.indicator: str | None = None
    self.next_request = 0.0
    self.next_indicator = 0.0
    self.aborts: int | None = None
    self.arrived = False
    self.arrived_t = 0.0
    self.classes: tuple = (None, [])  # the route they're for, and the road class of each of its segments
    self.stopped_since: float | None = None  # game time the car stopped, not at a light
    self.unstick_from: tuple | None = None  # (game time, m along the route) while steering round a blockage
    self.collisions0 = 0  # the plugin counts frames in contact since the car was entered
    self.cmd, self.cmd_t = 0.0, 0.0  # the ramped speed, and when it was worked out
    self.target_t = 0.0

  # *** control file ***

  def _poll(self, now: float):
    if self.path is None or now < self.next_poll:
      return
    self.next_poll = now + POLL_EVERY
    try:
      mtime = self.path.stat().st_mtime
      if mtime == self.mtime or mtime < self.started:
        return
      self.mtime = mtime
      cfg = {**DEFAULTS, **json.loads(self.path.read_text())}
    except (OSError, ValueError) as e:
      if not isinstance(e, FileNotFoundError):
        print(f"gta5: expert control file: {e}", flush=True)
      return
    was = self.on
    self.cfg, self.on = cfg, bool(cfg["on"])
    print(f"gta5: expert {'on' if self.on else 'off'} {json.dumps({k: v for k, v in cfg.items() if k != 'on'})}", flush=True)
    if self.on and not was:
      self._reset()
      if self.log is None:
        self.log = open(log_path(), "a", buffering=1)
    elif self.active:
      self._send_settings()  # settings changed while driving

  def _settings(self, speed: float | None = None) -> dict:
    c = self.cfg
    out = {"style": int(c["style"]), "ability": float(c["ability"]), "aggr": float(c["aggr"]), "task": str(c["task"])}
    if speed is not None:
      out["speed"] = round(speed, 2)
      self.sent_speed = speed
    return out

  def _send_settings(self):
    # a ramped speed goes on from where it is
    self.send({"type": "ai", **self._settings(None if float(self.cfg["ramp"]) > 0 else float(self.cfg["speed"]))})

  def _lead(self, v: float) -> float:
    """How far above the car's speed a ramped cap may be."""
    launch = self.cfg.get("launch")
    return float(launch) if launch is not None and v < STANDSTILL else float(self.cfg["lead"])

  def _start_speed(self, v: float) -> float:
    if float(self.cfg["ramp"]) > 0:
      self.cmd, self.cmd_t = max(v, 0.0) + self._lead(v), self.game_t
      return min(self.cmd, float(self.cfg["speed"]))
    return float(self.cfg["speed"])

  # *** each bridge step ***

  def update(self, state: dict, route, engaged: bool) -> bool:
    """Whether expert mode drives now; the bridge then leaves openpilot and nav out."""
    now = time.monotonic()
    self._poll(now)
    if not self.on:
      if self.active:
        self._stop("off")
      self._guard(state.get("ai") or {}, engaged, now)
      return False
    if "ai" not in state:
      if not self.warned:
        print("gta5: expert mode needs the plugin's ai command (an older gta5op_core.dll is loaded)", flush=True)
        self.warned = True
      return False
    if engaged and now >= self.next_cancel:
      self.next_cancel = now + 1.0
      self.cancel_engagement()
    # once per game frame: the bridge steps at 100 Hz
    if state.get("t") == self.last_t:
      return True
    self.last_t = state.get("t")
    self.game_t = float(state.get("t") or now)  # the ramp's clock: the game's, which stops while it does
    ai = state.get("ai") or {}
    if self.aborts is None:
      self.aborts = ai.get("aborts", 0)
    elif ai.get("aborts", 0) > self.aborts:
      self.on = False
      self._stop("the driver pressed the engage key", plugin=False)
      return False
    dest = self.cfg.get("dest")
    if route is not None and dest and np.hypot(*(route.points[-1] - np.asarray(dest[:2], dtype=float))) > DEST_NEAR:
      route = None  # the last trip's, still there until the router has the new one
    if route is None and self.cfg["need_route"] and not self.active:
      self._write(state, ai)
      return True  # waiting for the route
    v = state.get("vEgo", 0.0)
    if not self.active:
      self.active = True
      self.clear_nav_desire()
      self.collisions0 = state.get("collisions", 0)
      self.send({"type": "ai", "on": 1, "stop": STOP_RANGE, **self._settings(self._start_speed(v))})
      self.next_request = now + REQUEST_EVERY
    elif not ai.get("on") and now >= self.next_request:
      # the plugin dropped it (reloaded, or the player out of the driver's seat): ask again
      self.next_request = now + REQUEST_EVERY
      self.sent_target = None
      self.send({"type": "ai", "on": 1, "stop": STOP_RANGE, **self._settings(self._start_speed(v))})
    if route is not self.route:
      self._new_route(route)
    if self.arrived and self.game_t - self.arrived_t > float(self.cfg["hold_after"]):
      # the car is given back, so a left-on AI can't drive the next run
      self.on = False
      self._stop("arrived")
      self._write_control({"on": False})
      return False
    if route is not None:
      self._follow(route, state, now)
    self._write(state, ai)
    return True

  def _new_route(self, route):
    self.route, self.target, self.anchor, self.final, self.done = route, None, None, False, set()
    if route is not None and self._class_caps() and ROUTER:
      threading.Thread(target=self._fetch_classes, args=(route,), daemon=True).start()
    if route is None:
      self.mans, self.events = [], []
      if self.sent_target is not None:
        self.send({"type": "ai", "clear": 1})  # the waypoint, or wandering
        self.sent_target = None
      self._set_indicator(None, "none")
      return
    self.mans = maneuvers(route)
    events = set(route.junctions) | {m.along for m in self.mans}
    events |= {float(route.along[m["begin_shape_index"]]) for m in route.maneuvers
               if m.get("type") not in NOT_EVENTS and m.get("begin_shape_index", 0) < len(route.along)}
    self.events = sorted(events)

  def _pick(self, route) -> tuple[float, float | None, bool]:
    """The next target: m along the route, the junction or maneuver it's just past, and whether it's the route's end."""
    c, at = self.cfg, route.at
    lo, hi, past = at + float(c["ahead_min"]), at + float(c["ahead_max"]), float(c["past"])
    if route.length <= hi:
      return route.length, None, True
    ahead = [e for e in self.events if e > at + EVENT_MIN]
    if c["targets"] == "smooth":
      # ahead_max on, moved on only once ahead_min is left, and not inside a junction
      near = next((e for e in ahead if abs(e - hi) < past), None)
      if near is None:
        return hi, None, False
      return (near - BEFORE_EVENT if near - BEFORE_EVENT >= lo else near + past), None, False
    if not ahead or ahead[0] + past > hi:
      if ahead and ahead[0] < hi + BEFORE_EVENT:
        return max(lo, ahead[0] - BEFORE_EVENT), None, False  # short of a junction just beyond reach
      return hi, None, False
    target, anchor = ahead[0] + past, ahead[0]
    # short of ahead_min: past the next ones too, while they're in reach
    for e in ahead[1:]:
      if target >= lo or e + past > hi:
        break
      target, anchor = e + past, e
    # then on towards the next, so the AI isn't slowing to arrive as it comes out of the junction
    nxt = next((e for e in ahead if e > target), None)
    return max(target, min(hi, nxt - BEFORE_EVENT if nxt is not None else route.length)), anchor, False

  def _follow(self, route, state: dict, now: float):
    if self.cfg["unstick"]:
      self._unstick(route, state)
    at, c = route.at, self.cfg
    due = self.target is None or (not self.final and (self.target_along - at < RETARGET_NEAR or (
      now - self.target_t >= float(c["retarget_every"]) and
      ((self.anchor is not None and at > self.anchor + EVENT_MIN) or
       (self.anchor is None and self.target_along - at < float(c["ahead_min"]))))))
    if due:
      self.target_t = now
      self.target_along, self.anchor, self.final = self._pick(route)
      xy = route_point(route, self.target_along)
      z = float(np.interp(self.target_along, route.along, route.z)) if not np.isnan(route.z).any() else state["pos"][2]
      self.target = np.array([xy[0], xy[1], z])
    t = tuple(round(float(v), 1) for v in self.target)
    if t != self.sent_target:
      self.send({"type": "ai", "x": t[0], "y": t[1], "z": t[2], "stop": FINAL_STOP if self.final else STOP_RANGE})
      self.sent_target = t
    v = state.get("vEgo", 0.0)
    cap = self._speed(route, v, self.game_t)
    step = float(c["speed_step"])
    if abs(cap - self.sent_speed) >= step or (cap < step and self.sent_speed != cap):
      self.send({"type": "ai", "speed": round(cap, 2)})
      self.sent_speed = cap
    gentle = c["arrive"] == "gentle"
    left = route.length - at - (float(c["stop_before"]) if gentle else 0.0)
    if self.final and left < ARRIVED and v < 0.5 and not self.arrived:
      self.arrived, self.arrived_t = True, self.game_t
      print("gta5: expert arrived", flush=True)
    self._desire(route, state, now)

  def _unstick(self, route, state: dict):
    """Stopped a while, not at a light, with the route going on: steer round what's in the way (a parked car, which a
    style without SteerAroundStationaryCars waits behind for ever) with unstick_style, until the car has moved on
    unstick_dist m or unstick_for s have passed."""
    c, t, v = self.cfg, self.game_t, state.get("vEgo", 0.0)
    if self.unstick_from is not None:
      t0, at0 = self.unstick_from
      if route.at - at0 >= float(c["unstick_dist"]) or t - t0 >= float(c["unstick_for"]):
        self.unstick_from, self.stopped_since = None, None
        self.send({"type": "ai", "style": int(c["style"])})
        self._event("unstick", phase="end", moved=round(route.at - at0, 1), secs=round(t - t0, 1))
      return
    ai = state.get("ai") or {}
    going_on = route.length - route.at > float(c["stop_before"]) + ARRIVED
    blocked = v < STANDSTILL and not ai.get("stoppedAtLight") and not self.arrived and going_on
    if not blocked:
      self.stopped_since = None
      return
    if self.stopped_since is None:
      self.stopped_since = t
    elif t - self.stopped_since > float(c["unstick_after"]):
      self.unstick_from = (t, route.at)
      self.send({"type": "ai", "style": int(c["unstick_style"])})
      self.target = None  # a fresh target: the plugin tasks the AI again with the new style
      print(f"gta5: expert unstick: stopped {t - self.stopped_since:.0f} s, not at a light", flush=True)
      self._event("unstick", phase="start", stopped=round(t - self.stopped_since, 1), pos=state.get("pos"))

  def _event(self, what: str, **kw):
    if self.log is not None:
      self.log.write(json.dumps({"mono": round(time.monotonic(), 3), "event": what, **kw}) + "\n")

  def _class_caps(self) -> dict[str, float]:
    """speed_by_class as {road class: m/s}, from "residential:10,primary:14" or a JSON object."""
    spec = self.cfg.get("speed_by_class")
    if isinstance(spec, dict):
      return {str(k): float(v) for k, v in spec.items()}
    out = {}
    for part in str(spec or "").split(","):
      k, _, v = part.partition(":")
      try:
        out[k.strip()] = float(v)
      except ValueError:
        pass
    return out

  def _fetch_classes(self, route):
    """Valhalla's road class (use "ramp" as the class ramp) of each segment of the route's shape, into self.classes."""
    from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon
    classes: list[str | None] = [None] * max(len(route.points) - 1, 0)
    try:
      req = {"shape": [dict(zip(("lat", "lon"), to_lat_lon(float(x), float(y)), strict=True)) for x, y in route.points],
             "shape_match": "walk_or_snap", "costing": "auto",
             "filters": {"attributes": ["edge.road_class", "edge.use", "edge.begin_shape_index", "edge.end_shape_index",
                                        "matched.edge_index"], "action": "include"}}
      http = urllib.request.Request(f"{ROUTER}/trace_attributes", json.dumps(req).encode(), {"Content-Type": "application/json"})
      trace = json.loads(urllib.request.urlopen(http, timeout=5).read())
    except (OSError, ValueError) as e:
      print(f"gta5: expert: no road classes: {e}", flush=True)
      return
    edges = trace.get("edges", [])
    names = ["ramp" if e.get("use") == "ramp" else e.get("road_class") for e in edges]
    if trace.get("matched_points"):
      for k, m in enumerate(trace["matched_points"][:len(classes)]):
        i = m.get("edge_index")
        classes[k] = names[i] if i is not None and i < len(names) else None
    else:
      for e, name in zip(edges, names, strict=True):
        classes[e.get("begin_shape_index", 0):e.get("end_shape_index", 0)] = [name] * (e.get("end_shape_index", 0) - e.get("begin_shape_index", 0))
      classes = classes[:max(len(route.points) - 1, 0)]
    self.classes = (route, classes)

  def _class_cap(self, route, k: int, caps: dict[str, float]) -> float:
    """The speed_by_class cap on segment k: "default", else the lowest, where the class isn't known (yet)."""
    known = self.classes[1] if self.classes[0] is route else []
    name = known[k] if k < len(known) else None
    if name in caps:
      return caps[name]
    return caps.get("default", min(caps.values()))

  def _allowed(self, s: float, d: float, decel: float) -> float:
    """The speed from which to slow to s within d m: at decel, and at decel_fast above FAST."""
    fast = float(self.cfg["decel_fast"]) or decel
    d = max(d, 0.0)
    slow_part = max(FAST * FAST - s * s, 0.0) / (2 * decel)
    if d <= slow_part:
      return math.sqrt(s * s + 2 * decel * d)
    return math.sqrt(max(s, FAST) ** 2 + 2 * fast * (d - slow_part))

  def _speed(self, route, v: float, now: float) -> float:
    """The AI's speed cap: the set cap or the map's limit where lower; with decel, lower ahead of turns, lower limits and
    (arrive=gentle) a stop short of the route's end; with ramp, raised from the car's speed no faster than that."""
    c, at = self.cfg, route.at
    cap = float(c["speed"])
    limits = route.limits if c["limits"] else []
    if route.seg < len(limits) and limits[route.seg] > 0:
      cap = min(cap, float(limits[route.seg]))
    caps = self._class_caps()
    if caps:
      cap = min(cap, self._class_cap(route, route.seg, caps))
    decel, gentle = float(c["decel"]), c["arrive"] == "gentle"
    ahead: list[tuple[float, float]] = []  # (m ahead, m/s there)
    if decel > 0:
      ahead += [(m.along - at, float(c["turn_speed"])) for i, m in enumerate(self.mans) if m.turn and i not in self.done]
      for k in range(route.seg + 1, len(limits)):
        d = float(route.along[k]) - at
        if d > LIMIT_LOOKAHEAD:
          break
        if 0 < limits[k] < cap:
          ahead.append((d, float(limits[k])))
      for k in range(route.seg + 1, len(route.points) - 1) if caps else ():
        d = float(route.along[k]) - at
        if d > LIMIT_LOOKAHEAD:
          break
        if self._class_cap(route, k, caps) < cap:
          ahead.append((d, self._class_cap(route, k, caps)))
    if gentle:
      ahead.append((route.length - at - float(c["stop_before"]), HOLD_SPEED))
      decel = decel if decel > 0 else ARRIVE_DECEL
    for d, s in ahead:
      cap = min(cap, self._allowed(s, d, decel))
    ramp = float(c["ramp"])
    if ramp > 0:
      dt = min(max(now - self.cmd_t, 0.0), 0.5)
      self.cmd, self.cmd_t = min(cap, self.cmd + ramp * dt, max(v, 0.0) + self._lead(v)), now
      cap = self.cmd
    return max(cap, HOLD_SPEED)

  def _desire(self, route, state: dict, now: float):
    at, v, heading = route.at, state.get("vEgo", 0.0), state.get("heading", 0.0)
    current = None
    for i, m in enumerate(self.mans):
      if i in self.done:
        continue
      d = m.along - at
      if m.turn:
        over = d < -TURN_PAST or (d < -TURN_DONE_AFTER and abs(wrap(heading - m.exit_heading)) < TURN_DONE_HEADING)
      else:
        over = d < -KEEP_PAST
      if over:
        self.done.add(i)
        continue
      if d <= max(SIGNAL_DIST, v * SIGNAL_TIME):
        current = m
      break
    if current is None:
      self._set_indicator(None, "none")
    else:
      side = "left" if current.desire.endswith("Left") else "right"
      self._set_indicator(side if current.signal else None, current.desire)
    # the game's indicator differs (the bridge's turn cancel, or the arrow keys): set it again
    if state.get("indicator") != self.indicator and now >= self.next_indicator:
      self.next_indicator = now + REQUEST_EVERY
      self.send({"type": "ai", "indicator": self.indicator or "off"})

  def _set_indicator(self, side: str | None, label: str):
    self.label = label
    if side != self.indicator:
      self.indicator = side
      self.next_indicator = time.monotonic() + REQUEST_EVERY
      self.send({"type": "ai", "indicator": side or "off"})

  def _guard(self, ai: dict, engaged: bool, now: float):
    """With expert mode off, the plugin's AI driver mustn't drive: it outlives the bridge, and would override openpilot."""
    if "on" not in ai:
      return
    first, self.seen_ai = not self.seen_ai, True
    if not ai["on"] or now < self.next_guard:
      return
    if self.path is None:
      why = "expert mode isn't enabled (GTA5_EXPERT)"
    elif first:
      why = "left on from before the bridge started"
    elif engaged:
      why = "openpilot is engaged"
    else:
      return  # gta5_cmd.py ai on, for a manual test
    self.next_guard = now + REQUEST_EVERY
    print(f"gta5: WARNING: the plugin's AI driver is on with expert mode off ({why}): turning it off", flush=True)
    self.send({"type": "ai", "on": 0, "indicator": "off"})

  def _write_control(self, cfg: dict):
    if self.path is None:
      return
    try:
      tmp = self.path.with_suffix(".tmp")
      tmp.write_text(json.dumps(cfg) + "\n")
      tmp.replace(self.path)
    except OSError as e:
      print(f"gta5: expert control file: {e}", flush=True)

  def _stop(self, why: str, plugin: bool = True):
    if plugin:
      self.send({"type": "ai", "on": 0, "indicator": "off"})
    else:
      self.send({"type": "ai", "indicator": "off"})
    print(f"gta5: expert stopped ({why})", flush=True)
    self.active = False
    if self.log is not None:
      self.log.write(json.dumps({"mono": time.monotonic(), "event": "stop", "why": why}) + "\n")
    self._reset()

  def _write(self, state: dict, ai: dict):
    if self.log is None:
      return
    r = self.route
    lane, plugin = state.get("lane"), state.get("lanePlugin", state.get("lane"))
    self.log.write(json.dumps({
      "mono": round(time.monotonic(), 3), "t": state.get("t"), "active": self.active, "ai": ai,
      "target": None if self.target is None else [round(float(v), 1) for v in self.target],
      "targetAhead": None if r is None or self.target is None else round(self.target_along - r.at, 1), "final": self.final,
      "arrived": self.arrived, "desire": self.label, "indicator": self.indicator, "gameIndicator": state.get("indicator"),
      "pos": state.get("pos"), "heading": state.get("heading"), "vEgo": state.get("vEgo"), "speedCap": self.sent_speed,
      "routeAt": None if r is None else round(r.at, 1), "routeOff": None if r is None else round(r.off, 1),
      "routeEnd": None if r is None else round(r.length - r.at, 1), "lane": lane, "lanePlugin": plugin,
      "laneFrac": state.get("laneFrac"), "twoWay": state.get("twoWay"),
      "oncoming": any(bool(x) and x[0] < 0 for x in (lane, plugin)), "traffic": state.get("traffic"),
      "collisions": state.get("collisions", 0) - self.collisions0 if self.active else 0,
      "collisionsTotal": state.get("collisions"), "street": state.get("street"), "unstick": self.unstick_from is not None,
    }) + "\n")
