"""Expert mode: the game's own AI drives the car along our route, as expert driving to record for training the driving
model (~/gta5test/notes/ai_driver.md has the research).

Off unless the bridge runs with GTA5_EXPERT set: 1 for the control file at CONTROL, or the file's path. The bridge then
watches that file (gta5_cmd.py expert on|off|route writes it; changes from before the bridge started are ignored):
  {"on": true, "speed": 12, "style": 1076369579, "ability": 1, "aggr": 0, "task": "longrange", "limits": true,
   "need_route": false, "ahead_min": 60, "ahead_max": 120, "past": 25}
While on, openpilot is kept disengaged and nav's cues wait. On our map's route (GTA5_ROUTER) the plugin's AI driver is
given a target 60-120 m ahead, just past the next junction or maneuver, so its own short pathfinding can only take the
route's way through it; the indicators follow the route's turns, ramps and exits, and `label` holds the desire they stand
for. Off the map's routes the AI drives to the game's waypoint, or wanders. Each game state appends a JSON line to the
log (GTA5_EXPERT_LOG, else expert.jsonl beside GTA5_LOG, else /tmp/gta5_expert.jsonl): the AI's state, the target, the
label and the lane readings, so samples can be filtered later. The engage key stops the AI and expert mode."""
import json
import math
import os
import time
from pathlib import Path

import numpy as np

CONTROL = Path("/tmp/gta5_expert.json")
POLL_EVERY = 0.5  # s
DEFAULTS = {"on": False, "speed": 12.0, "style": 1076369579, "ability": 1.0, "aggr": 0.0, "task": "longrange", "limits": True,
            "need_route": False, "ahead_min": 60.0, "ahead_max": 120.0, "past": 25.0}
EVENT_MIN = 8.0  # m ahead: a junction or maneuver nearer than this is being driven through
BEFORE_EVENT = 10.0  # m: a straight target that would land in a junction stays this far short of it
RETARGET_NEAR = 30.0  # m from the target, whatever it's for
STOP_RANGE = 5.0  # m, for a target on the way
FINAL_STOP = 8.0  # m, the route's end
ARRIVED = 15.0  # m from the route's end, stopped
REQUEST_EVERY = 2.0  # s between asking the plugin again for what its state doesn't show
SPEED_STEP = 0.5  # m/s change worth sending
# desires
HEADING_SPAN = 20.0  # m before and after a maneuver for its headings
TURN_ANGLE = 45.0  # deg: a turn; less is a keep
BEND_ANGLE = 15.0  # deg: less than this is straight on
SIGNAL_TIME, SIGNAL_DIST = 5.0, 50.0  # s, m: signal from the later of these before a maneuver
TURN_DONE_HEADING = 20.0  # deg from the way out
TURN_DONE_AFTER = 5.0  # m past it, at least
TURN_PAST = 40.0  # m past it, at the latest
KEEP_PAST = 30.0  # m
# Valhalla maneuver types
TURNS = {9: "Right", 10: "Right", 11: "Right", 14: "Left", 15: "Left", 16: "Left"}
FORKS = {18: "Right", 19: "Left", 20: "Right", 21: "Left", 23: "Right", 24: "Left"}
SIGNALED_FORKS = {18, 19, 20, 21}  # ramps and exits
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


class Maneuver:
  """A maneuver on the route the car must do something for: its place (m along), desire and whether it's signaled."""
  def __init__(self, along: float, desire: str, signal: bool, exit_heading: float, turn: bool):
    self.along = along
    self.desire = desire
    self.signal = signal
    self.exit_heading = exit_heading
    self.turn = turn


def route_point(route, s: float) -> np.ndarray:
  return np.array([np.interp(s, route.along, route.points[:, 0]), np.interp(s, route.along, route.points[:, 1])])


def heading_between(a: np.ndarray, b: np.ndarray) -> float:
  """Game degrees, counterclockwise from north, as the plugin's heading."""
  return math.degrees(math.atan2(-(b[0] - a[0]), b[1] - a[1]))


def maneuvers(route) -> list[Maneuver]:
  out = []
  for m in route.maneuvers:
    kind, i = m.get("type"), m.get("begin_shape_index", 0)
    if kind not in TURNS and kind not in FORKS or i >= len(route.along):
      continue
    s = float(route.along[i])
    before, here, after = (route_point(route, v) for v in (s - HEADING_SPAN, s, s + HEADING_SPAN))
    exit_heading = heading_between(here, after)
    change = wrap(exit_heading - heading_between(before, here))
    if kind in TURNS:
      if abs(change) >= TURN_ANGLE:
        out.append(Maneuver(s, "turn" + TURNS[kind], True, exit_heading, True))
      elif abs(change) >= BEND_ANGLE:
        out.append(Maneuver(s, "keep" + TURNS[kind], False, exit_heading, False))
    else:
      out.append(Maneuver(s, "keep" + FORKS[kind], kind in SIGNALED_FORKS, exit_heading, False))
  return out


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
    self.next_cancel = 0.0
    self.log = None
    self.warned = False
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

  def _settings(self) -> dict:
    c = self.cfg
    return {"speed": float(c["speed"]), "style": int(c["style"]), "ability": float(c["ability"]), "aggr": float(c["aggr"]),
            "task": str(c["task"])}

  def _send_settings(self):
    self.send({"type": "ai", **self._settings()})
    self.sent_speed = float(self.cfg["speed"])

  # *** each bridge step ***

  def update(self, state: dict, route, engaged: bool) -> bool:
    """Whether expert mode drives now; the bridge then leaves openpilot and nav out."""
    now = time.monotonic()
    self._poll(now)
    if not self.on:
      if self.active:
        self._stop("off")
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
    ai = state.get("ai") or {}
    if self.aborts is None:
      self.aborts = ai.get("aborts", 0)
    elif ai.get("aborts", 0) > self.aborts:
      self.on = False
      self._stop("the driver pressed the engage key", plugin=False)
      return False
    if route is None and self.cfg["need_route"] and not self.active:
      self._write(state, ai)
      return True  # waiting for the route
    if not self.active:
      self.active = True
      self.clear_nav_desire()
      self.send({"type": "ai", "on": 1, "stop": STOP_RANGE, **self._settings()})
      self.sent_speed = float(self.cfg["speed"])
      self.next_request = now + REQUEST_EVERY
    elif not ai.get("on") and now >= self.next_request:
      # the plugin dropped it (reloaded, or the player out of the driver's seat): ask again
      self.next_request = now + REQUEST_EVERY
      self.sent_target = None
      self.send({"type": "ai", "on": 1, "stop": STOP_RANGE, **self._settings()})
    if route is not self.route:
      self._new_route(route)
    if route is not None:
      self._follow(route, state, now)
    self._write(state, ai)
    return True

  def _new_route(self, route):
    self.route, self.target, self.anchor, self.final, self.done = route, None, None, False, set()
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
    at = route.at
    due = self.target is None or (not self.final and (self.target_along - at < RETARGET_NEAR or
                                                       (self.anchor is not None and at > self.anchor + EVENT_MIN) or
                                                       (self.anchor is None and self.target_along - at < float(self.cfg["ahead_min"]))))
    if due:
      self.target_along, self.anchor, self.final = self._pick(route)
      xy = route_point(route, self.target_along)
      z = float(np.interp(self.target_along, route.along, route.z)) if not np.isnan(route.z).any() else state["pos"][2]
      self.target = np.array([xy[0], xy[1], z])
    t = tuple(round(float(v), 1) for v in self.target)
    if t != self.sent_target:
      self.send({"type": "ai", "x": t[0], "y": t[1], "z": t[2], "stop": FINAL_STOP if self.final else STOP_RANGE})
      self.sent_target = t
    # speed: the cap, or the map's limit where lower
    cap = float(self.cfg["speed"])
    limit = float(route.limits[route.seg]) if self.cfg["limits"] and route.seg < len(route.limits) else 0.0
    if limit > 0:
      cap = min(cap, limit)
    if abs(cap - self.sent_speed) >= SPEED_STEP:
      self.send({"type": "ai", "speed": round(cap, 2)})
      self.sent_speed = cap
    if self.final and route.length - at < ARRIVED and state.get("vEgo", 0) < 0.5 and not self.arrived:
      self.arrived = True
      print("gta5: expert arrived", flush=True)
    self._desire(route, state, now)

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
      "collisions": state.get("collisions"), "street": state.get("street"),
    }) + "\n")
