"""Wrong-way clips: expert mode drives the car out of its lane into the oncoming lanes of an undivided road, holds it
there, and brings it back, for training the driving model to tell it's in an oncoming lane and to steer back out.

A clip is an expert route trip (gta5_cmd.py expert route) whose control file has `wrongway`, a JSON object:
  {"clip": "W003p0", "lane": -1, "approach_m": 70, "drift_m": 60, "hold_s": 3, "speed": 10, "recover": "ai",
   "recover_m": 60, "after_m": 40}
The bridge (gta5_expert.py) drives it with a pure pursuit controller (the plugin's curvature and acceleration control,
the game's AI off) along a path planned once at the start on the route's lanes (router.Route.lane_line): in the start
lane for approach_m (`approach`), ramping into lane `lane` (-1 the nearest oncoming lane, -0.5 straddling the centre
line, -2 the far one) over drift_m (`drift`), there for hold_s at `speed` (`hold`), then back into the start lane:
recover "ai" hands the car to the game's AI, driving the trip's route on to its destination as expert mode always does
(`ai_recover` until the plugin's lane reading is one of ours again, then `done`); recover "path" has the controller
ramp back over recover_m (`recover`), drive after_m on in lane (`done`) and stop. A collision, an off-road or off-path
reading, losing every lane reading, an unsuitable road or the AI not recovering ends the clip (`abort`): the car is
braked to a stop and expert mode stops with "wrongway abort: <why>".

Phases are logged to the expert log (an event row at each change, and `ww` on every row) and recorded per frame in
gta5.npz (gta5_record.py: the `wrongway` array, only in segments with a clip), so labelling can keep the drift and
hold frames out of imitation and use them for the oncoming lane label."""
import math

import numpy as np

PHASES = ("", "approach", "drift", "hold", "recover", "ai_recover", "done", "abort")  # gta5.npz's codes: their index
DRIVEN = {"approach", "drift", "hold", "recover", "done", "abort"}  # the controller drives (not the game's AI)

ACCEL_MAX, DECEL_MAX, SPEED_GAIN = 1.5, 3.0, 0.8
ABORT_DECEL = 3.5  # m/s^2
LOOKAHEAD = (6.0, 0.8, 20.0)  # m: min, s of speed, max
PATH_DEV = 3.0  # m off the planned path, held PATH_DEV_S, ends the clip
PATH_DEV_S = 0.7
OFF_ROAD = 1.0  # m past the road's outer edge
NO_LANE_S = 2.0  # s without any lane reading (the plugin's, the route's, the map's)
STOPPED = 0.3  # m/s
RECOVERED_S = 1.0  # s back in one of our lanes (the plugin's reading) that ends ai_recover
AI_RECOVER_MAX = 15.0  # s for the AI to be back in our lanes
DONE_AI_S = 3.0  # s of `done` logged before the clip hands over to the plain expert trip
MAX_S = 120.0  # s for the whole clip
# a road to drive it on: our lanes and the oncoming ones 1 or 2 each, no gap (median) between them, no junction node
SUIT_LANES = (1, 2)
MEDIAN_GAP = 1.0  # m between the directions' lanes
JUNCTION_CLEAR = 15.0  # m: no junction node this near the drift, hold or recovery
MAX_BEND = 25.0  # deg of heading change over the drift, hold and recovery
LANE_SECONDS = 1.8  # s at least to cross a lane, drifting or recovering


def defaults(cfg: dict) -> dict:
  c = {"clip": "", "lane": -1.0, "from_lane": None, "approach_m": 60.0, "drift_m": 60.0, "hold_s": 3.0, "speed": 10.0,
       "recover": "ai", "recover_m": 60.0, "after_m": 40.0, **cfg}
  for k in ("lane", "approach_m", "drift_m", "hold_s", "speed", "recover_m", "after_m"):
    c[k] = float(c[k])
  return c


def keys(c: dict, from_lane: float) -> list[tuple[float, float]]:
  """The planned lanes as Route.lane_line's keys: (m on from the clip's start, lane from the left of ours)."""
  a = max(c["approach_m"], c["speed"] ** 2 / (2 * 1.2) + 25.0)  # time to get up to speed in lane first
  # no faster across than LANE_SECONDS a lane, as a car can swerve
  across = abs(from_lane - c["lane"]) * LANE_SECONDS * c["speed"]
  d, h, r = max(c["drift_m"], across), c["hold_s"] * c["speed"], max(c["recover_m"], across)
  return [(0.0, from_lane), (a, from_lane), (a + d, c["lane"]), (a + d + h, c["lane"]), (a + d + h + r, from_lane),
          (a + d + h + r + c["after_m"] + 60.0, from_lane)]


def phase_marks(k: list[tuple[float, float]]) -> list[tuple[float, str]]:
  """(m on from the start, the phase starting there) for the controller's path."""
  return [(k[0][0], "approach"), (k[1][0], "drift"), (k[2][0], "hold"), (k[3][0], "recover"), (k[4][0], "done")]


def suitable(route, lo: float, hi: float) -> str | None:
  """Why the route between lo and hi m along it (the drift, hold and recovery) isn't a road for a clip, or None."""
  if route is None or route.lanes is None:
    return "no lanes on the route"
  if hi > route.length - 5.0:
    return f"route too short ({route.length:.0f} m for {hi:.0f})"
  ks = range(max(int(np.searchsorted(route.along, lo, side='right')) - 1, 0),
             min(int(np.searchsorted(route.along, hi, side='right')), len(route.points) - 1))
  for k in ks:
    sec = route.section(k)
    if sec is None or not sec.lanes:
      return f"no lanes at {route.along[k]:.0f} m"
    if not sec.two_way or sec.lanes not in SUIT_LANES or sec.back not in SUIT_LANES or sec.back != sec.lanes:
      return f"{sec.lanes}+{sec.back} lanes at {route.along[k]:.0f} m"
    if sec.spans[sec.first].left - sec.spans[sec.first - 1].right > MEDIAN_GAP:
      return f"a median at {route.along[k]:.0f} m"
  near = [j for j in route.junctions if lo - JUNCTION_CLEAR <= j <= hi + JUNCTION_CLEAR]
  if near:
    return f"a junction at {near[0]:.0f} m"
  s = np.arange(lo, hi, 10.0)
  if len(s) >= 2:
    x, y = np.interp(s, route.along, route.points[:, 0]), np.interp(s, route.along, route.points[:, 1])
    h = np.unwrap(np.arctan2(np.diff(x), np.diff(y)))
    if np.degrees(h.max() - h.min()) > MAX_BEND:
      return f"bends {np.degrees(h.max() - h.min()):.0f} deg"
  return None


def pursuit(path: np.ndarray, s_path: np.ndarray, s_car: float, pos, heading_deg: float, v: float) -> float:
  """Pure pursuit's curvature (left-positive, as the plugin takes it) towards the path's point a lookahead on."""
  ld = min(max(LOOKAHEAD[0], LOOKAHEAD[1] * v + 4.0), LOOKAHEAD[2])
  s = min(s_car + ld, s_path[-1])
  tx, ty = np.interp(s, s_path, path[:, 0]), np.interp(s, s_path, path[:, 1])
  h = math.radians(heading_deg)
  dx, dy = tx - pos[0], ty - pos[1]
  fwd = dx * -math.sin(h) + dy * math.cos(h)
  left = dx * -math.cos(h) + dy * -math.sin(h)
  d2 = max(fwd * fwd + left * left, 1.0)
  return 2.0 * left / d2


def project(path: np.ndarray, s_path: np.ndarray, pos, s_prev: float, window: float = 40.0) -> tuple[float, float]:
  """The car's (m along the path, m off it), searched from a little behind s_prev on."""
  lo = max(int(np.searchsorted(s_path, s_prev - 5.0)) - 1, 0)
  hi = min(int(np.searchsorted(s_path, s_prev + window)) + 1, len(path) - 1)
  a, b = path[lo:hi], path[lo + 1:hi + 1]
  if not len(a):
    return s_prev, float(np.hypot(*(np.asarray(pos[:2]) - path[-1])))
  ab = b - a
  p = np.asarray(pos[:2], dtype=float)
  t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-9), 0.0, 1.0)
  near = a + ab * t[:, None]
  d = np.hypot(*(near - p).T)
  i = int(np.argmin(d))
  return float(s_path[lo + i] + t[i] * np.hypot(*ab[i])), float(d[i])


class WrongWay:
  """One clip's controller. step() each game frame returns the plugin message to send (None: send nothing, the AI
  drives); `phase`, `events` (since last taken) and `finished` (None, "done" or "abort: <why>") for expert mode."""
  def __init__(self, cfg: dict):
    self.c = defaults(cfg)
    self.clip = str(self.c["clip"])
    self.phase = ""
    self.events: list[dict] = []
    self.finished: str | None = None
    self.path: np.ndarray | None = None
    self.s_path: np.ndarray | None = None
    self.marks: list[tuple[float, str]] = []
    self.keys: list[tuple[float, float]] = []
    self.s0 = self.s = 0.0
    self.dev = 0.0
    self.target_lane: float | None = None
    self.target_right: float | None = None
    self.from_lane: float | None = None
    self.why = ""
    self.collisions0: int | None = None
    self.t0 = None
    self.dev_t = None
    self.no_lane_t = None
    self.ok_lane_t = None
    self.phase_t = 0.0
    self.ai_wanted = False  # the AI is to take over (ai_recover): expert mode starts it

  def info(self) -> dict:
    return {"clip": self.clip, "phase": self.phase, "lane": self.target_lane, "right": self.target_right,
            "dev": round(self.dev, 2), "at": round(self.s - self.s0, 1)}

  def plan_record(self) -> dict:
    """What gta5.json and the trip record keep of the clip: its settings, keys and planned path (every ~4 m)."""
    p = None if self.path is None else self.path[::2].round(2).tolist()
    return {"clip": self.clip, "cfg": self.c, "from_lane": self.from_lane, "keys": self.keys, "marks": self.marks,
            "s0": round(self.s0, 2), "path": p}

  def _set(self, phase: str, t: float, **kw):
    if phase == self.phase:
      return
    self.phase, self.phase_t = phase, t
    self.events.append({"event": "wrongway", "clip": self.clip, "phase": phase, "t": t, **kw})

  def take_events(self) -> list[dict]:
    e, self.events = self.events, []
    return e

  def _abort(self, why: str, t: float):
    self.why = why
    self._set("abort", t, why=why)

  def _plan(self, route, state: dict, t: float) -> bool:
    sec = route.section(route.seg)
    if sec is None or not sec.lanes:
      self._abort("unsuitable: no lanes where it starts", t)
      return False
    # the lane the car is in (setup may not place it in the one asked for), else the one asked for, else the rightmost
    here = sec.lane(route.right) if route.off < 10.0 else None
    fl = self.c.get("from_lane")
    fl = here if here is not None and 0 <= here < sec.lanes else fl if fl is not None else sec.lanes - 1
    self.from_lane = float(min(max(int(fl), 0), sec.lanes - 1))
    self.keys = keys(self.c, self.from_lane)
    lo, hi = route.at + self.keys[1][0], route.at + self.keys[4][0]
    why = suitable(route, lo, hi)
    if why is not None:
      self._abort(f"unsuitable: {why}", t)
      return False
    line = route.lane_line(self.keys)
    if line is None or len(line) < 2:
      self._abort("unsuitable: no lane line", t)
      return False
    self.path = np.asarray(line, dtype=float)[:, :2]
    self.s_path = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(self.path, axis=0).T))))
    self.s0, _ = project(self.path, self.s_path, state["pos"], 0.0, window=self.s_path[-1])
    self.s = self.s0
    self.marks = [(self.s0 + d, ph) for d, ph in phase_marks(self.keys)]
    return True

  def _lane_target(self, ds: float) -> float:
    ks = self.keys
    return float(np.interp(ds, [k[0] for k in ks], [k[1] for k in ks]))

  def step(self, route, state: dict, t: float, collisions: int) -> dict | None:
    """The plugin control message for this frame (t: the game's clock), or None while the AI drives."""
    if self.finished is not None:
      return None
    pos, v, heading = state.get("pos"), float(state.get("vEgo") or 0.0), float(state.get("heading") or 0.0)
    if self.t0 is None:
      self.t0, self.collisions0 = t, collisions
      if pos is None or route is None or not self._plan(route, state, t):
        if self.phase != "abort":
          self._abort("unsuitable: no route", t)
      else:
        self._set("approach", t, plan=self.plan_record())
    if self.phase == "abort":
      if v < STOPPED:
        self.finished = f"abort: {self.why}"
        self._set_done_event(t)
        return {"type": "control", "active": True, "curvature": 0.0, "accel": -1.0}
      return {"type": "control", "active": True, "curvature": 0.0, "accel": -ABORT_DECEL}

    # what ends a clip
    if collisions > (self.collisions0 or 0):
      self._abort("collision", t)
    elif t - self.t0 > MAX_S:
      self._abort("took too long", t)
    readings = [r for r in (state.get("lanePlugin", state.get("lane")), (state.get("laneMap") or {}).get("lane")) if r is not None]
    self.no_lane_t = None if readings else (self.no_lane_t if self.no_lane_t is not None else t)
    if self.phase != "abort" and self.no_lane_t is not None and t - self.no_lane_t > NO_LANE_S:
      self._abort("no lane reading", t)
    if self.phase != "abort" and route is not None and route.off < 30.0:
      sec = route.section(route.seg)
      if sec is not None and sec.lanes and not (sec.edges[0] - OFF_ROAD <= route.right <= sec.edges[1] + OFF_ROAD):
        self._abort(f"off the road ({route.right:+.1f} m right of its line)", t)
    if self.phase == "abort":
      return {"type": "control", "active": True, "curvature": 0.0, "accel": -ABORT_DECEL}

    if self.phase == "ai_recover":
      plugin = state.get("lanePlugin", state.get("lane"))
      ours = plugin is not None and plugin[0] >= 0
      self.ok_lane_t = (self.ok_lane_t if self.ok_lane_t is not None else t) if ours else None
      if self.ok_lane_t is not None and t - self.ok_lane_t >= RECOVERED_S:
        self._set("done", t, recovered_s=round(self.ok_lane_t - self.phase_t, 2))
      elif t - self.phase_t > AI_RECOVER_MAX:
        self.why = f"the AI not back in our lanes in {AI_RECOVER_MAX:.0f} s"
        self.ai_wanted = False
        self._set("abort", t, why=self.why)
        return {"type": "control", "active": True, "curvature": 0.0, "accel": -ABORT_DECEL}
      return None
    if self.phase == "done" and self.c["recover"] == "ai":
      if t - self.phase_t > DONE_AI_S:
        self.finished = "done"
        self._set_done_event(t)
      return None

    self.s, self.dev = project(self.path, self.s_path, pos, self.s)
    self.dev_t = (self.dev_t if self.dev_t is not None else t) if self.dev > PATH_DEV else None
    if self.dev_t is not None and t - self.dev_t > PATH_DEV_S:
      self._abort(f"{self.dev:.1f} m off the path", t)
      return {"type": "control", "active": True, "curvature": 0.0, "accel": -ABORT_DECEL}
    ds = self.s - self.s0
    phase = [ph for d, ph in self.marks if self.s >= d][-1] if self.s >= self.marks[0][0] else "approach"
    if phase == "recover" and self.c["recover"] == "ai":
      self.ai_wanted = True
      self.target_lane, self.target_right = self.from_lane, None
      self._set("ai_recover", t)
      return None
    self._set(phase, t)
    self.target_lane = self._lane_target(ds)
    sec = route.section(route.seg) if route is not None else None
    self.target_right = round(sec.offset(self.target_lane), 2) if sec is not None and sec.lanes else None
    target_v = self.c["speed"]
    if phase == "done" and ds > self.keys[4][0] + self.c["after_m"]:
      target_v = 0.0
      if v < STOPPED:
        self.finished = "done"
        self._set_done_event(t)
    accel = float(np.clip(SPEED_GAIN * (target_v - v), -DECEL_MAX, ACCEL_MAX))
    if target_v == 0.0:
      accel = min(accel, -1.5)
    return {"type": "control", "active": True, "curvature": round(pursuit(self.path, self.s_path, self.s, pos, heading, v), 5),
            "accel": round(accel, 3)}

  def _set_done_event(self, t: float):
    self.events.append({"event": "wrongway", "clip": self.clip, "phase": "end", "t": t, "finished": self.finished})
