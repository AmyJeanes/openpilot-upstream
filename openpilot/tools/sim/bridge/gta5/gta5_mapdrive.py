"""Map driver: expert mode driving our route itself along the map's lanes, rather than the game's AI (gta5_expert.py
with driver=map; design in the openpilot-gta5 task's MAPDRIVER_DESIGN.md). Traffic off for now: no leads, no gaps.

The plan is made once when a route arrives (and spliced on a reroute, past the lane change under way and 3 s on):
nav's lane plan (planner.lane_plan, as the ribbon draws it) with absolute along-route keys, each lane change re-timed
within its lane-slot window (by the trip's style, freeway ones within +-20 % of nav's schedule) and reshaped as a
quintic (minimum-jerk) move, then put on the lanes by RouteLanes.lane_line (its offsets, jogs and fillets). That line is
the intent, recorded as the path labels; the car aims for it plus a small, slow in-lane bias and wander (never periodic).
The plan is checked against the lane slots' targets (lane_slots.py): a lane the slots don't allow is a
plan_slot_mismatch event (an abort with plan_check=abort).

Each game frame: pure pursuit (lookahead clamp(5, 0.6 v + 3, 25) m) from the pose ~70 ms on, plus the path's curvature
0.3 s ahead as feedforward (the pursuit's own curve-cutting taken out), rate limited to the style's lateral jerk and
capped at 4.5 m/s^2; the speed follows a static profile (limit or road class x style, the path's curvature at the
style's lateral acceleration, nav's turn speeds) with stop signs (full stop and dwell), give way lines (slow, then on),
traffic lights (no state to read: driven through and marked light_unknown +-5 s) and a gentle arrival stop_before m
short of the end. Aborts (dev > 1.5 m for 1 s, heading off by 30 deg, a collision, off the road, no progress for 20 s,
lateral acceleration over 4.5, the driver's input, a failed plan) brake to a stop and end the trip (on_abort=ai hands it
to the game's AI). Map anomalies (kerb_contact, lane_disagree, tracking_saturated, gta_link_offset, stop_line_far,
plan_slot_mismatch) are logged with their place, for the map-fix work.

The control file's `mapdrive` object sets it up: {"seed": 7, "preset": "normal", "style": {"t_lc": 5.0}, "bias_max":
0.3, "wander": 0.1, "on_abort": "stop", "speed": null, "plan_check": "event"}; bias_max 0 and wander 0 drive exactly on
the line."""
import bisect
import hashlib
import math
import os
import random
import subprocess

import numpy as np

from openpilot.selfdrive.navd.maneuvers import maneuvers
from openpilot.selfdrive.navd.planner import FREEWAY, TUNE, FwyState, Turn, lane_plan
from openpilot.tools.sim.bridge.gta5.gta5_wrongway import project, pursuit

PHASES = ("", "wait", "drive", "stop", "give_way", "arrive", "done", "abort")  # gta5.npz mapx phase codes: the index
REASONS = ("", "limit", "curve", "turn", "stop", "give_way", "arrive", "start", "abort", "hold")
MAPX_COLUMNS = ("phase", "plan_s", "path_dev", "target_lane", "target_right", "v_prof", "a_cmd", "kappa_cmd", "speed_reason",
                "lc_tau", "lead_gap")
SOURCES = ("none", "ai", "map", "ai_fallback")  # gta5.npz expert_src codes

STEP = 1.0  # m between the path's points
LANE_W = 5.5  # m, GTA's lanes
FRONT = 2.4  # m from the car's origin to its front bumper
HALF_WIDTH = 1.0  # m
LATENCY = 0.07  # s from the game's frame to the plugin acting on the control
FF_PREVIEW = 0.3  # s ahead the path's curvature is fed forward (the plugin's yaw-rate loop lag)
LOOKAHEAD = (5.0, 0.6, 3.0, 25.0)  # m: min, s of speed, plus, max
KAPPA_MAX = 0.2  # 1/m
A_LAT_MAX = 4.5  # m/s^2 commanded at most, and an abort measured above it
CURV_SMOOTH = 7.0  # m: the path's curvature averaged over this
JERK_SHARE = 2.0  # x the style's lateral jerk the speed profile allows where the path's curvature changes
RATE_SHARE = 3.0  # x it the commanded curvature may change at: a safety limit above the profile's
COMPUTE_EVERY = 0.1  # s: a new frame is computed at once, else at least this often; between, the last control is resent
DECEL_MAX = 4.0  # m/s^2
ABORT_DECEL = 3.5
HOLD_ACCEL = -1.5  # m/s^2 holding a stop
STOPPED = 0.25  # m/s
STOP_NEAR = 2.0  # m from a stop target, stopped: at it
GIVE_WAY_SPEED = 3.0  # m/s at the line
LIGHT_MARK = 5.0  # s either side of a light passed without its state
SIGNAL_TIME, SIGNAL_DIST = 5.0, 50.0  # s, m before a turn (as the AI expert)
TURN_DONE_HEADING, TURN_DONE_AFTER, TURN_PAST, KEEP_PAST = 20.0, 5.0, 40.0, 30.0
MIN_LC_M = 30.0  # m a lane change takes at the least
JOIN_MAX = 8.0  # m from its lane line the car may start
BEND_DEG = 20.0  # deg a route point turns that makes its road beside it unclear
KAPPA_DRIVABLE = 0.1  # 1/m: a tighter bend in the lane line (a jog, a corner without a fillet) is eased out
LC_DONE = 0.8  # share of a lane change after which its signal goes off
CROSSING_CLEAR = 5.0  # m past a stop line or junction node a city lane change may start
TURN_SPAN = 10.0  # m either side of a turn's point held to its turn speed
# aborts
DEV_ABORT, DEV_ABORT_S = 1.5, 1.0
HEADING_ABORT, HEADING_ABORT_S = 30.0, 0.5
NO_PROGRESS_S = 20.0
A_LAT_ABORT_S = 0.5
OFF_ROAD = 1.0  # m past the kerb
USER_STEER = 0.02
# anomalies
KERB_MARGIN = 0.2  # m the car's side may come past the kerb before it counts
JUNCTION_CLEAR = 25.0  # m from a junction node, where cross-sections don't describe the road, kerbs aren't checked
LANE_DISAGREE_S = 2.0
SATURATED_S = 0.5
GTA_OFFSET = 1.5  # m between GTA's lane centre and the map's for the planned lane
STOP_LINE_FAR = 30.0  # m from a stop line on to its junction's first node
ANOMALY_SPACING = 20.0  # m between anomalies of one kind
# road class speeds (m/s) where the map has no limit: 65, 55, 40, 35, 30, 25, 15 mph
CLASS_SPEED = {"motorway": 29.1, "trunk": 24.6, "motorway_link": 17.9, "trunk_link": 17.9, "primary": 15.6, "secondary": 15.6,
               "tertiary": 13.4, "primary_link": 13.4, "secondary_link": 13.4, "tertiary_link": 11.2, "residential": 11.2,
               "unclassified": 11.2, "living_street": 6.7, "service": 6.7}
DEFAULT_SPEED = 13.4

# the trip's style: (low, high, which way "brisk" goes: +1 high, -1 low, 0 neither), drawn once per trip from its seed
STYLE_RANGES = {
  "speed": (0.85, 1.10, 1), "a_lat": (1.5, 2.8, 1), "j_lat": (1.2, 2.5, 1), "accel": (1.0, 2.0, 1), "decel": (1.0, 2.0, 1),
  "jerk": (1.0, 2.0, 1), "turn": (0.9, 1.15, 1), "t_lc": (4.0, 6.5, -1), "lc_timing": (0.1, 0.9, 0),
  "fwy_timing": (-0.2, 0.2, 0), "signal_lead": (1.5, 4.0, -1), "stop_margin": (0.5, 2.5, -1), "stop_dwell": (0.5, 2.0, -1),
  "reaction": (0.4, 1.5, -1), "bias": (-1.0, 1.0, 0),
}
PRESETS = {"calm": 0.2, "normal": 0.5, "brisk": 0.8}
DEFAULTS = {"seed": None, "preset": "normal", "style": {}, "bias_max": 0.3, "wander": 0.1, "wander_m": 300.0,
            "on_abort": "stop", "speed": None, "plan_check": "event", "stop_before": 15.0, "hold_after": 3.0, "retime": True,
            "drive_on_right": True}


def pick_style(seed: int, preset: str = "normal", over: dict | None = None) -> dict:
  """A trip's style from its seed: each value drawn about the preset's place in its range (brisk: the quick end)."""
  rng = random.Random(seed)
  centre = PRESETS.get(preset, 0.5)
  out = {}
  for k, (lo, hi, way) in STYLE_RANGES.items():
    u = rng.random() if way == 0 else min(max(rng.gauss(centre if way > 0 else 1.0 - centre, 0.15), 0.0), 1.0)
    out[k] = round(lo + (hi - lo) * u, 3)
  out.update(over or {})
  return out


_VERSION: dict | None = None


def map_version() -> dict:
  """The map's and the code's versions, for recordings: the sha1 of the map's gta5.osm.pbf (GTA5_MAP) and the git rev."""
  global _VERSION
  if _VERSION is None:
    _VERSION = {"pbf_sha1": None, "git": None}
    pbf = os.path.join(os.getenv("GTA5_MAP", os.path.expanduser("~/gta5map_lanes")), "gta5.osm.pbf")
    try:
      h = hashlib.sha1()
      with open(pbf, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
          h.update(block)
      _VERSION["pbf_sha1"] = h.hexdigest()[:16]
    except OSError:
      pass
    try:
      _VERSION["git"] = subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "--short=12", "HEAD"],
                                                text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
      pass
  return _VERSION


# *** the plan's keys ***

EPS = 1e-3


def lane_at(keys: list[tuple[float, float]], s: float) -> float:
  """The keys' lane s m along: the last key at or before s (the later of two at one place), towards the next."""
  i = bisect.bisect_right([k[0] for k in keys], s) - 1
  if i < 0:
    return keys[0][1]
  if i >= len(keys) - 1:
    return keys[-1][1]
  (s0, l0), (s1, l1) = keys[i], keys[i + 1]
  return l1 if s1 - s0 < EPS else l0 + (l1 - l0) * (s - s0) / (s1 - s0)


def _ramp(keys, j: int) -> bool:
  return keys[j + 1][0] - keys[j][0] > EPS and abs(keys[j + 1][1] - keys[j][1]) > EPS


def chains(keys: list[tuple[float, float]]) -> list[tuple[int, int]]:
  """Each lane change in the keys: (first key, last key), a run of ramps with any renumbering steps (two keys at one
  place, where the road's lanes change) inside it."""
  out, i, n = [], 0, len(keys)
  while i < n - 1:
    if not _ramp(keys, i):
      i += 1
      continue
    j = i
    while j < n - 1 and (_ramp(keys, j) or (keys[j + 1][0] - keys[j][0] <= EPS and j + 1 < n - 1 and _ramp(keys, j + 1))):
      j += 1
    out.append((i, j))
    i = j
  return out


def quintic(t):
  t = np.clip(t, 0.0, 1.0)
  return t * t * t * (10.0 + t * (-15.0 + 6.0 * t))


def shaped(keys: list[tuple[float, float]], step: float = 2.0) -> list[tuple[float, float]]:
  """The keys with each lane change a quintic (minimum-jerk) move across, sampled every `step` m, in place of a ramp;
  renumbering steps inside a change stay where they are, at their share of the move."""
  out: list[tuple[float, float]] = []
  last = 0
  for a, b in chains(keys):
    out += keys[last:a]
    s0, s1 = keys[a][0], keys[b][0]
    span = max(s1 - s0, EPS)
    for j in range(a, b):
      (sa, la), (sb, lb) = keys[j], keys[j + 1]
      if sb - sa <= EPS:
        continue
      d = (lb - la) / (sb - sa) * span  # the whole move's lanes in this numbering
      x = la - d * (sa - s0) / span
      ss = np.unique(np.concatenate((np.arange(sa, sb, step), [sb])))
      out += [(float(s), float(x + d * quintic((s - s0) / span))) for s in ss]
    last = b + 1
  out += keys[last:]
  dedup = [out[0]] if out else []
  for k in out[1:]:
    if abs(k[0] - dedup[-1][0]) > EPS or abs(k[1] - dedup[-1][1]) > EPS:
      dedup.append(k)
  return dedup


def lc_seconds(style: dict, lanes: float) -> float:
  """s a change across `lanes` lanes takes: the style's, and no quicker than its lateral jerk allows (a quintic's peak
  jerk is 60 D / T^3)."""
  d = LANE_W * max(abs(lanes), 1.0)
  return float(min(max(style["t_lc"] * math.sqrt(max(abs(lanes), 1.0)), (60.0 * d / style["j_lat"]) ** (1 / 3)), 9.0))


# *** the path ***

def resample(points: np.ndarray, step: float = STEP) -> tuple[np.ndarray, np.ndarray]:
  p = np.asarray(points, float)[:, :2]
  keep = np.concatenate(([True], np.hypot(*np.diff(p, axis=0).T) > 1e-6))
  p = p[keep]
  s = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))))
  if len(p) < 2:
    return p, s
  su = np.arange(0.0, s[-1], step)
  su = np.append(su, s[-1]) if s[-1] - su[-1] > 0.25 * step else su
  return np.stack([np.interp(su, s, p[:, 0]), np.interp(su, s, p[:, 1])], axis=1), su


def headings(p: np.ndarray) -> np.ndarray:
  """Radians counterclockwise from east, unwrapped, at each point."""
  d = np.gradient(p, axis=0)
  return np.unwrap(np.arctan2(d[:, 1], d[:, 0]))


def smooth(x: np.ndarray, n: int) -> np.ndarray:
  if n <= 1 or len(x) < n:
    return x
  pad = n // 2
  xp = np.concatenate((np.full(pad, x[0]), x, np.full(pad, x[-1])))
  return np.convolve(xp, np.ones(n) / n, mode="valid")[:len(x)]


def wander(s: np.ndarray, rng: random.Random, sigma: float, spacing: float) -> tuple[np.ndarray, list[float]]:
  """Slow smooth lateral wander: random knots every `spacing` m (normal, sigma m), eased between (C1, flat at each
  knot), so it never weaves at a fixed period."""
  if sigma <= 0 or not len(s):
    return np.zeros_like(s), []
  n = int(s[-1] // spacing) + 2
  knots = [rng.gauss(0.0, sigma) for _ in range(n)]
  k = np.minimum((s // spacing).astype(int), n - 2)
  t = s / spacing - k
  e = t * t * (3 - 2 * t)
  kn = np.array(knots)
  return kn[k] + (kn[k + 1] - kn[k]) * e, [round(v, 3) for v in knots]


def soften(p: np.ndarray, kmax: float, iters: int = 400) -> tuple[np.ndarray, list[tuple[int, float]]]:
  """The polyline (points STEP apart) with bends tighter than kmax eased out, as a driver cuts a corner the car can't
  take as drawn (a lane line's jog, a corner without a fillet): their points pulled towards their neighbours'
  middle until none is. Also each such bend as it was: (point index, its curvature)."""
  p = np.asarray(p, float).copy()

  def bends(q):
    s = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(q, axis=0).T))))
    k = smooth(np.gradient(headings(q), np.maximum(s, np.arange(len(s)) * 1e-6)), 3)
    bad = np.abs(k) > kmax
    bad[:2] = bad[-2:] = False
    return k, bad
  if len(p) < 5:
    return p, []
  k, bad = bends(p)
  e = np.flatnonzero(np.diff(np.concatenate(([0], bad.astype(np.int8), [0]))))
  spots = [(int((a + b) // 2), float(k[a:b][np.argmax(np.abs(k[a:b]))])) for a, b in zip(e[::2], e[1::2], strict=True)]
  for a, b in zip(e[::2], e[1::2], strict=True):
    # each bend in a window around it, the window's ends held
    lo, hi = max(int(a) - 40, 0), min(int(b) + 40, len(p))
    w = p[lo:hi]
    for _ in range(iters):
      _, bw = bends(w)
      bw[:3] = bw[-3:] = False
      if not bw.any():
        break
      m = np.convolve(bw.astype(float), np.ones(9), mode="same") > 0
      m[:2] = m[-2:] = False
      mid = w.copy()
      mid[1:-1] = 0.5 * (w[:-2] + w[2:])
      w[m] += 0.5 * (mid[m] - w[m])
    p[lo:hi] = w
  return p, spots


def hermite_join(path: np.ndarray, pos: np.ndarray, heading: float, v: float, tol: float = 8.0) -> np.ndarray:
  """The path's start replaced by a cubic from the car's pose (heading in game deg) into it, where the car heads more
  than `tol` deg off the path's way."""
  if len(path) < 8:
    return path
  j = int(min(max(12.0, 2.5 * v) / STEP, len(path) - 3))
  th = headings(path[:3])[0]
  near = float(np.min(np.hypot(*(path[:j] - np.asarray(pos, float)).T)))
  if abs(wrap(heading - game_heading(th))) <= tol and near < 0.75:
    return path
  p0, p1 = np.asarray(pos, float), path[j]
  d = float(np.hypot(*(p1 - p0)))
  h = math.radians(heading)
  t0 = np.array([-math.sin(h), math.cos(h)]) * d
  t1 = (path[j + 1] - path[j - 1]) / max(float(np.hypot(*(path[j + 1] - path[j - 1]))), 1e-9) * d
  u = np.linspace(0.0, 1.0, j + 1)[:, None]
  curve = (2 * u**3 - 3 * u**2 + 1) * p0 + (u**3 - 2 * u**2 + u) * t0 + (-2 * u**3 + 3 * u**2) * p1 + (u**3 - u**2) * t1
  return np.vstack([curve[:-1], path[j:]])


def along_route(route, pts: np.ndarray, start: float) -> tuple[np.ndarray, np.ndarray]:
  """Each point's (m along the route, m right of its line), searched on from `start` m along so where the route
  passes back near itself the points stay on their own part."""
  rp, ra = route.points[:, :2], route.along
  k = max(int(np.searchsorted(ra, start, side="right")) - 1, 0)
  out_s, out_r = np.empty(len(pts)), np.empty(len(pts))
  for n, p in enumerate(pts):
    lo, hi = max(k - 1, 0), min(k + 8, len(rp) - 1)
    a, b = rp[lo:hi], rp[lo + 1:hi + 1]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0.0, 1.0)
    near = a + ab * t[:, None]
    d = np.hypot(*(near - p).T)
    i = int(np.argmin(d))
    k = lo + i
    seg = max(float(np.hypot(*ab[i])), 1e-6)
    out_s[n] = ra[k] + t[i] * seg
    out_r[n] = float((p[0] - a[i, 0]) * ab[i, 1] - (p[1] - a[i, 1]) * ab[i, 0]) / seg
  return np.maximum.accumulate(out_s), out_r


def segment_of(route, s: float) -> int:
  return int(min(max(np.searchsorted(route.along, s, side="right") - 1, 0), max(len(route.points) - 2, 0)))


def road_speed(route, k: int) -> float:
  """The road's speed on route segment k: the map's limit, else by its class."""
  limits = getattr(route, "limits", None)
  if limits is not None and k < len(limits) and limits[k] > 0:
    return float(limits[k])
  classes = getattr(route, "classes", None) or []
  return CLASS_SPEED.get(classes[k] if k < len(classes) else "", DEFAULT_SPEED)


def wrap(deg: float) -> float:
  return (deg + 180) % 360 - 180


def game_heading(theta: float) -> float:
  """The game's heading (deg counterclockwise from north) of a direction theta radians from east."""
  return wrap(math.degrees(theta) - 90.0)


class MapDriver:
  """One trip's map driver. step() each bridge step returns the plugin control message (resent between frames);
  `phase`, `events` (since last taken), `indicator`/`label` (the desire it signals) and `finished` (None, "arrived" or
  "abort: <why>") for expert mode."""
  def __init__(self, cfg: dict | None = None, tune=None):
    self.c = {**DEFAULTS, **(cfg or {})}
    seed = self.c.get("seed")
    self.seed = int(seed) if seed is not None else random.randrange(1 << 31)
    self.style = pick_style(self.seed, str(self.c["preset"]), self.c.get("style") or {})
    self.rng = random.Random(self.seed + 1)
    self.tune = tune or TUNE
    self.phase = ""
    self.events: list[dict] = []
    self.finished: str | None = None
    self.why = ""
    self.route = None
    self.plans = 0
    self.splices: list[dict] = []
    self.anomalies: list[dict] = []
    self.aborts: list[dict] = []
    self.lights: list[dict] = []
    self.plan: dict | None = None  # plan_record of the plan driven
    self.indicator: str | None = None
    self.label = "none"
    self.reason = ""
    self.msg: dict | None = None
    self.last_key = None
    self.last_t: float | None = None
    self.t0: float | None = None
    self.collisions0: int | None = None
    self.kappa = 0.0
    self.a = 0.0
    self.s = 0.0
    self.dev = 0.0
    self.v_prof = math.nan
    self.lc_tau = math.nan
    self.target_lane = math.nan
    self.target_right = math.nan
    self.timers: dict[str, float | None] = {}
    self.progress = (0.0, 0.0)  # (s, t) of the last progress
    self.stopped_at: float | None = None
    self.arrived_t: float | None = None
    self.engage0: int | None = None
    self._last_anomaly: dict[str, float] = {}
    self.path = None
    self.join_m = 0.0  # m of path from the car into its lane line, where the road's own checks wait
    self.unclear = np.zeros(0)

  # *** what expert mode and recordings read ***

  def info(self) -> dict:
    return {"phase": self.phase, "s": round(self.s, 1), "dev": round(self.dev, 2), "lane": None if math.isnan(self.target_lane) else round(self.target_lane, 2),
            "vProf": None if math.isnan(self.v_prof) else round(self.v_prof, 2), "a": round(self.a, 2), "kappa": round(self.kappa, 5),
            "reason": self.reason, "lcTau": None if math.isnan(self.lc_tau) else round(self.lc_tau, 2), "label": self.label}

  def mapx_row(self) -> list[float]:
    return [PHASES.index(self.phase) if self.phase in PHASES else 0, self.s, self.dev, self.target_lane, self.target_right, self.v_prof,
            self.a, self.kappa, REASONS.index(self.reason) if self.reason in REASONS else 0, self.lc_tau, math.nan]

  def summary(self) -> dict:
    """gta5.json's mapx: the trip's style and seed, its plans' keys, splices, aborts, anomalies and lights passed."""
    return {"seed": self.seed, "style": self.style, "cfg": {k: v for k, v in self.c.items() if k != "style"}, "version": map_version(),
            "keys": self.plan["keys"] if self.plan else None, "plans": self.plans, "splices": self.splices, "aborts": self.aborts,
            "anomalies": self.anomalies, "lights": self.lights, "slot_check": self.plan["slot_check"] if self.plan else None}

  def intent_line(self, step: int = 2) -> list:
    """The intent (the planned path without the in-lane wander) from the car on, for the ribbon and recordings."""
    if self.path is None:
      return []
    i = max(int(self.s / STEP) - 2, 0)
    return self.intent[i::step].round(1).tolist()

  def take_events(self) -> list[dict]:
    e, self.events = self.events, []
    return e

  def _event(self, what: str, when: float, **kw):
    self.events.append({"event": what, "t": round(when, 3), **kw})

  def _set(self, phase: str, t: float, **kw):
    if phase != self.phase:
      self.phase = phase
      self._event("mapdrive", t, phase=phase, **kw)

  def _anomaly(self, kind: str, when: float, pos, at: float, **kw):
    last = self._last_anomaly.get(kind)
    if last is not None and abs(at - last) < ANOMALY_SPACING:
      return
    self._last_anomaly[kind] = at
    a = {**kw, "kind": kind, "t": round(when, 3), "pos": [round(float(pos[0]), 1), round(float(pos[1]), 1)], "at": round(at, 1)}
    self.anomalies.append(a)
    self._event("anomaly", when, **{k: v for k, v in a.items() if k != "t"})

  def _abort(self, why: str, t: float, pos=None):
    if self.phase == "abort":
      return
    self.why = why
    self.aborts.append({"why": why, "t": round(t, 3), "s": round(self.s, 1), "pos": None if pos is None else [round(float(pos[0]), 1), round(float(pos[1]), 1)]})
    self._set("abort", t, why=why)

  # *** planning ***

  def _speed_at(self, route, s: float) -> float:
    v = road_speed(route, segment_of(route, s)) * self.style["speed"]
    return min(v, float(self.c["speed"])) if self.c.get("speed") else v

  def nav_keys(self, route, lane) -> list[tuple[float, float]]:
    """nav's lane plan from the car's lane, as the ribbon has it, with absolute along-route keys."""
    info = route.info(route.length)
    v = self._speed_at(route, route.at)
    keys = lane_plan(route.rest(), info["forks"], lane, route.lanes_at, v, self.tune, info["laneArrows"], info["laneDrops"],
                     maps=route.lane_maps(route.length), turns=route.turns(route.length),
                     classes=route.changes(route.classes, route.length), limits=route.changes(route.limit_list, route.length),
                     fwy=FwyState(), crossings=[a - route.at for a in route.stops + route.junctions if a > route.at])
    return [(route.at + d, float(lane)) for d, lane in keys]

  def retime(self, route, keys: list[tuple[float, float]], slots) -> list[tuple[float, float]]:
    """Each single lane change over the style's time at the road's speed, its end placed by the style within its
    move's lane-slot window (freeway ones nav's schedule +-fwy_timing of a lane's time), clear of stop lines and junction
    nodes in the city."""
    keys = list(keys)
    crossings = sorted(route.stops + route.junctions)
    for a, b in chains(keys):
      if b != a + 1:
        continue  # renumbering inside it: left as nav has it
      (s0, la), (s1, lb) = keys[a], keys[b]
      v = self._speed_at(route, s0)
      length = max(lc_seconds(self.style, lb - la) * v, MIN_LC_M * max(abs(lb - la), 1.0))
      lower = keys[a - 1][0] if a > 0 else s0
      upper = keys[b + 1][0] if b + 1 < len(keys) else s1
      window = None
      if slots is not None:
        for m in slots.moves:
          if m.along >= s1 - 1.0 and m.need:
            _, w0, w1 = m.window(v)
            if w1 >= s1 - 1.0:
              window = (w0, w1)
            break
      k = segment_of(route, s0)
      fwy = (route.classes[k] if k < len(route.classes) else "") in FREEWAY
      hi = min(upper, window[1] if window is not None else s1)
      if fwy:
        e = s1 + self.style["fwy_timing"] * self.tune.fwy_lane_time * v
      elif window is not None:
        lo = max(window[0], lower) + length
        e = lo + self.style["lc_timing"] * max(hi - lo, 0.0)
      else:
        e = s1
      e = min(max(e, lower + length), hi)
      start = max(e - length, lower)
      if e - start < 0.6 * length:
        continue  # no room: as nav has it
      if not fwy:
        for c in crossings:
          if start - CROSSING_CLEAR < c < e:
            if c + CROSSING_CLEAR + length <= hi:
              start, e = c + CROSSING_CLEAR, c + CROSSING_CLEAR + length
            elif c - CROSSING_CLEAR - length >= lower:
              start, e = c - CROSSING_CLEAR - length, c - CROSSING_CLEAR
        if any(start - CROSSING_CLEAR < c < e for c in crossings) and not any(s0 - CROSSING_CLEAR < c < s1 for c in crossings):
          continue  # only nav's timing was clear of them
      keys[a], keys[b] = (float(start), la), (float(e), lb)
    return keys

  def slot_check(self, route, keys: list[tuple[float, float]], slots) -> dict:
    """The plan against the lane slots' targets: from each target's end to its move, the plan's lane (rounded) must be
    one of them."""
    out = {"checked": 0, "mismatches": []}
    if slots is None:
      return out
    for m in slots.moves:
      if not m.need or m.along <= route.at + 5.0:
        continue
      v = self._speed_at(route, m.along)
      _, _, end = m.window(v)
      bad = None
      for s in np.arange(max(end, route.at + 1.0), m.along - 2.0, 2.0):
        _, target = slots.target(float(s), v)
        if target is None or target.centre or not target.want or target.move is not m:
          continue
        out["checked"] += 1
        lane = lane_at(keys, float(s))
        if int(round(lane)) not in target.want and abs(lane - round(lane)) < 0.25:
          bad = {"s": round(float(s), 1), "move": round(m.along, 1), "plan": round(lane, 2), "want": sorted(target.want)}
          break
      if bad is not None:
        out["mismatches"].append(bad)
    return out

  def _make_plan(self, route, state: dict, t: float) -> bool:
    lanes = route.lanes
    if lanes is None:
      self._abort("plan: no lanes on the route", t, state.get("pos"))
      return False
    lane = route.lane() if route.on_road() else None
    if lane is None or lane[0] < 0:
      sec = route.section(route.seg)
      if sec is None or not sec.lanes:
        k = next((j for j in range(route.seg, min(route.seg + 6, len(lanes.sections))) if lanes.sections[j] is not None and lanes.sections[j].lanes), None)
        sec = lanes.sections[k] if k is not None else None
      if sec is None:
        self._abort("plan: no lanes where it starts", t, state.get("pos"))
        return False
      lane = [min(max(sec.lane(route.right), 0), sec.lanes - 1), sec.lanes]
    try:
      from openpilot.selfdrive.navd.lane_slots import LaneSlots
      slots = LaneSlots(route, bool(self.c["drive_on_right"]))
    except Exception as e:  # the slots on a map they can't read: plan without them
      self._event("mapdrive_warning", t, why=f"lane slots: {e!r}")
      slots = None
    nav = self.nav_keys(route, lane)
    keys = self.retime(route, nav, slots) if self.c["retime"] else nav
    check = self.slot_check(route, keys, slots)
    for bad in check["mismatches"]:
      p = route_point(route, bad["s"])
      self._anomaly("plan_slot_mismatch", t, p, bad["s"], **bad)
    if check["mismatches"] and self.c["plan_check"] == "abort":
      self._abort(f"plan: lane {check['mismatches'][0]['plan']} where the slots want {check['mismatches'][0]['want']}", t, state.get("pos"))
      return False
    dense = shaped(keys)
    at0 = route.at
    line = lanes.lane_line(at0, [(s - at0, lane_) for s, lane_ in dense if s >= at0 - 1e-6] or [(0.0, float(lane[0]))])
    if line is None or len(line) < 3:
      self._abort("plan: no lane line", t, state.get("pos"))
      return False
    intent, s_path = resample(np.asarray(line, float))
    if len(intent) < 3:
      self._abort("plan: lane line too short", t, state.get("pos"))
      return False
    intent, sharp = soften(intent, KAPPA_DRIVABLE)
    intent, s_path = resample(intent)
    bias = self.style["bias"] * float(self.c["bias_max"])
    off, knots = wander(s_path, self.rng, float(self.c["wander"]), float(self.c["wander_m"]) / 2)
    off = np.clip(bias + off, -0.45, 0.45) * np.clip(s_path / 20.0, 0.0, 1.0)
    # from where the car is, across to the line as a lane change would (setup places it in a lane, a reroute may not)
    e0 = self._offset_from(intent, np.asarray(state["pos"][:2], float))
    if abs(e0) > JOIN_MAX:
      self._abort(f"plan: the car is {e0:+.1f} m from its lane line", t, state.get("pos"))
      return False
    v0 = max(float(state.get("vEgo") or 0.0), 5.0)
    join = max(MIN_LC_M, lc_seconds(self.style, e0 / LANE_W) * v0) if abs(e0) > 0.5 else 15.0
    off = off + e0 * (1.0 - quintic(s_path / join))
    self.join_m = max(join, 30.0)
    self._derive(route, intent, off, t, start=(np.asarray(state["pos"][:2], float), float(state.get("heading") or 0.0), v0))
    for i, k in sharp:
      self._anomaly("sharp_corner", t, intent[min(i, len(intent) - 1)], i * STEP, kappa=round(k, 3))
    self.keys, self.nav_plan = keys, nav
    self.plans += 1
    self.plan = {"at0": round(at0, 2), "keys": [(round(s, 2), round(lane_, 3)) for s, lane_ in keys],
                 "nav_keys": [(round(s, 2), round(lane_, 3)) for s, lane_ in nav], "lane": list(lane), "bias": round(bias, 3),
                 "wander_knots": knots, "wander_m": float(self.c["wander_m"]) / 2, "slot_check": check,
                 "stops": [[round(x["along"], 1), x["kind"]] for x in self.stops], "path": self.intent[::2].round(2).tolist()}
    self.changes = self._changes(keys)  # each lane change on the path: (path m from, to, side)
    self._plan_anomalies(route, t)
    return True

  def _unclear(self, along: float) -> bool:
    """Whether a place (m along the route) is near a junction node or a sharp bend of the route's line."""
    u = self.unclear
    if not len(u):
      return False
    i = int(np.searchsorted(u, along))
    return min(abs(u[min(i, len(u) - 1)] - along), abs(u[max(i - 1, 0)] - along)) < JUNCTION_CLEAR

  @staticmethod
  def _offset_from(line: np.ndarray, p: np.ndarray) -> float:
    """m right of a polyline's start (its first 30 m) that p is."""
    n = min(len(line), 31)
    a, b = line[:n - 1], line[1:n]
    ab = b - a
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9), 0.0, 1.0)
    d = np.hypot(*(a + ab * t[:, None] - p).T)
    i = int(np.argmin(d))
    seg = max(float(np.hypot(*ab[i])), 1e-6)
    return float((p[0] - a[i, 0]) * ab[i, 1] - (p[1] - a[i, 1]) * ab[i, 0]) / seg

  def _derive(self, route, intent: np.ndarray, off: np.ndarray, t: float, start=None):
    """The driven path (intent moved `off` m right; from the car's pose `start` (pos, heading, speed) along a cubic
    where it heads another way) and what's read along it: curvature, the route's along and right, the stops and
    turns, the speed profile."""
    th = headings(intent)
    target = intent + np.stack([np.sin(th), -np.cos(th)], axis=1) * off[:, None]
    if start is not None:
      target = hermite_join(target, *start)
    self.intent, self.path = intent, target
    self.s_path = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(target, axis=0).T))))
    self.theta = headings(target)
    k_raw = np.gradient(self.theta, self.s_path)
    self.kappa_path = smooth(k_raw, max(int(CURV_SMOOTH / STEP) | 1, 1))
    self.route_s, self.route_r = along_route(route, target, route.at - 5.0)
    self.route = route
    # where the route's own line says little about the road beside it: junction nodes, and its points where it bends
    # sharply (a lane line or the car cuts inside such a corner, past the segment's own kerbs)
    d = np.diff(route.points[:, :2], axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    turn = np.degrees(np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi))
    self.unclear = np.sort(np.concatenate((np.asarray(route.junctions, float), route.along[1:-1][turn > BEND_DEG])))
    self.s = 0.0
    # stops ahead, as path m
    self.stops = []
    for along, kind in zip(route.stops, route.stop_kinds, strict=False):
      if along <= route.at + 1.0:
        continue
      s_line = float(np.interp(along, self.route_s, self.s_path))
      self.stops.append({"along": float(along), "kind": kind, "s": s_line, "done": False, "t_stop": None,
                         "target": s_line - FRONT - self.style["stop_margin"]})
    arrive_along = route.length - float(self.c["stop_before"])
    self.s_arrive = float(np.interp(max(arrive_along, route.at + 5.0), self.route_s, self.s_path))
    # the static speed profile: the road's speed, the curvature, nav's turn speeds, the arrival
    k = np.clip(np.searchsorted(route.along, self.route_s, side="right") - 1, 0, max(len(route.points) - 2, 0))
    lim = np.array([road_speed(route, int(j)) for j in k]) * self.style["speed"]
    if self.c.get("speed"):
      lim = np.minimum(lim, float(self.c["speed"]))
    curve = np.sqrt(self.style["a_lat"] / np.maximum(np.abs(self.kappa_path), 1e-4))
    # no faster than the lateral jerk allows where the curvature changes (jerk = v^3 dkappa/ds)
    dk = np.abs(np.gradient(self.kappa_path, self.s_path))
    curve = np.minimum(curve, np.cbrt(JERK_SHARE * self.style["j_lat"] / np.maximum(dk, 1e-6)))
    v = np.minimum(lim, curve)
    why = np.where(curve < lim, REASONS.index("curve"), REASONS.index("limit"))
    for d, side, exit_h, angle in route.turns(route.length):
      s_t = float(np.interp(route.at + d, self.route_s, self.s_path))
      cap = Turn(d, side, exit_h, abs(angle)).speed(self.tune) * self.style["turn"]
      m = (np.abs(self.s_path - s_t) < TURN_SPAN) & (v > cap)
      v[m], why[m] = cap, REASONS.index("turn")
    end = self.s_path >= self.s_arrive
    v[end], why[end] = 0.0, REASONS.index("arrive")
    b = self.style["decel"] * 0.8
    for i in range(len(v) - 2, -1, -1):
      ds = self.s_path[i + 1] - self.s_path[i]
      lim_b = math.sqrt(v[i + 1] ** 2 + 2 * b * ds)
      if lim_b < v[i]:
        v[i], why[i] = lim_b, why[i + 1]
    self.v_static, self.why_static = v, why
    self.mans = maneuvers(route)
    self.done_mans: set[int] = set()

  def _changes(self, keys):
    out = []
    for a, b in chains(keys):
      s0 = float(np.interp(keys[a][0], self.route_s, self.s_path))
      s1 = float(np.interp(keys[b][0], self.route_s, self.s_path))
      d = keys[b][1] - keys[a][1]
      # the lanes' numbering may step inside it: the side is the way across the road, by the path's right offset
      r0 = float(np.interp(s0, self.s_path, self.route_r))
      r1 = float(np.interp(s1, self.s_path, self.route_r))
      side = "right" if (r1 - r0 if abs(r1 - r0) > 1.0 else d) > 0 else "left"
      out.append((s0, s1, side))
    return out

  def _plan_anomalies(self, route, t: float):
    """Map anomalies the plan itself shows: the car's side over a kerb along it, GTA's lanes elsewhere than the map's,
    a stop line far from its junction."""
    lanes = route.lanes
    junctions = np.asarray(route.junctions, float)
    for i in range(0, len(self.path), 3):
      rs, rr = self.route_s[i], self.route_r[i]
      if self._unclear(rs) or self.s_path[i] < self.join_m:
        continue
      k = segment_of(route, rs)
      sec = lanes.section_at(rs, k)
      if sec is None or not sec.lanes:
        continue
      lo, hi = sec.edges
      if rr - HALF_WIDTH < lo - KERB_MARGIN or rr + HALF_WIDTH > hi + KERB_MARGIN:
        self._anomaly("kerb_contact", t, self.path[i], float(self.s_path[i]), planned=True, right=round(float(rr), 2),
                      edges=[round(lo, 2), round(hi, 2)])
      link = route.links[k] if k < len(route.links) else None
      if link is not None and link.lanes:
        lane = lane_at(self.keys, rs)
        if abs(lane - round(lane)) > 0.2:
          continue
        if link.lanes != sec.lanes or link.back != len([sp for sp in sec.spans if sp.heading == -1]):
          self._anomaly("gta_link_offset", t, self.path[i], float(self.s_path[i]), why="lane counts",
                        map=[sec.lanes, len([sp for sp in sec.spans if sp.heading == -1])], gta=[link.lanes, link.back])
          continue
        gta = link.inner + (round(lane) + 0.5) * link.width
        mine = sec.offset(round(lane))
        if abs(gta - mine) > GTA_OFFSET:
          self._anomaly("gta_link_offset", t, self.path[i], float(self.s_path[i]), why="lane centre", lane=int(round(lane)),
                        map=round(mine, 2), gta=round(gta, 2))
    for st in self.stops:
      after = junctions[junctions > st["along"] - 1.0] if len(junctions) else junctions
      gap = float(after[0] - st["along"]) if len(after) else math.inf
      if gap > STOP_LINE_FAR:
        self._anomaly("stop_line_far", t, route_point(route, st["along"]), st["s"], kind_line=st["kind"],
                      to_junction=None if math.isinf(gap) else round(gap, 1))

  def _splice(self, route, state: dict, t: float):
    """A new route under way: the old path kept past the lane change under way and 3 s on, then the new plan's."""
    old_path, old_s, s, v = self.path, self.s_path, self.s, float(state.get("vEgo") or 0.0)
    cut = s + max(3.0 * v, 10.0)
    for s0, s1, _ in self.changes:
      if s0 <= s <= s1:
        cut = max(cut, s1)
    part = (old_s >= s - 2.0) & (old_s <= cut)  # from the car on: the new route starts there
    keep, old_intent = old_path[part], self.intent[part]
    if not self._make_plan(route, state, t):
      return
    if len(keep) >= 2:
      j, d = project(self.path, self.s_path, keep[-1], 0.0, window=self.s_path[-1])
      tail = self.path[self.s_path > j + 2.0]
      tail_i = self.intent[self.s_path > j + 2.0]
      joined, _ = resample(np.vstack([keep, tail]))
      joined_i, _ = resample(np.vstack([old_intent, tail_i]))
      n = min(len(joined), len(joined_i))
      th = headings(joined_i[:n])
      off = ((joined[:n] - joined_i[:n]) * np.stack([np.sin(th), -np.cos(th)], axis=1)).sum(axis=1)
      self._derive(route, joined_i[:n], off, t)
      self.changes = self._changes(self.keys)
      self.splices.append({"t": round(t, 3), "cut": round(cut, 1), "gap": round(float(d), 2), "pos": [round(float(keep[-1][0]), 1), round(float(keep[-1][1]), 1)]})
      self._event("mapdrive", t, phase=self.phase, splice=self.splices[-1])

  # *** each step ***

  def step(self, route, state: dict, t: float, collisions: int = 0) -> dict | None:
    """The plugin control message for this bridge step (t: s, a monotonic clock); None until there's a route."""
    if self.finished is not None:
      return {"type": "control", "active": True, "curvature": 0.0, "accel": HOLD_ACCEL} if self.finished == "arrived" else None
    pos = state.get("pos")
    v = float(state.get("vEgo") or 0.0)
    if pos is None:
      return self.msg
    key = (round(pos[0], 3), round(pos[1], 3), round(float(state.get("heading") or 0.0), 3), round(v, 3))
    if key == self.last_key and self.msg is not None and self.last_t is not None and t - self.last_t < COMPUTE_EVERY:
      return self.msg  # the same game frame: the control again, as the plugin drops one older than 0.3 s
    dt = min(max(t - self.last_t, 0.02), 0.2) if self.last_t is not None else 0.05
    self.last_key, self.last_t = key, t
    if self.t0 is None:
      self.t0, self.collisions0, self.engage0 = t, collisions, state.get("engagePresses")
      self.progress = (0.0, t)
    if route is None:
      self._set("wait", t)
      self.msg = {"type": "control", "active": True, "curvature": 0.0, "accel": HOLD_ACCEL}
      return self.msg
    if route is not self.route and self.phase != "abort":
      if self.path is None:
        if not self._make_plan(route, state, t):
          return self._abort_msg(v, t)
        self._set("drive", t, plan=self.plan, style=self.style, seed=self.seed, version=map_version())
      else:
        self._splice(route, state, t)
    if self.phase == "abort":
      return self._abort_msg(v, t)
    self._checks(route, state, t, collisions, v)
    if self.phase == "abort":
      return self._abort_msg(v, t)
    accel = self._longitudinal(state, v, t, dt)
    curvature = self._lateral(state, v, t, dt)
    self._labels(route, state, v, t)
    self._watch(route, state, v, t)
    self.msg = {"type": "control", "active": True, "curvature": round(curvature, 5), "accel": round(accel, 3)}
    return self.msg

  def _abort_msg(self, v: float, t: float) -> dict:
    self.indicator, self.label = None, "none"
    self.a = HOLD_ACCEL if v < STOPPED else -ABORT_DECEL
    self.kappa = 0.0
    self.reason = "abort"
    if v < STOPPED and self.finished is None:
      self.finished = f"abort: {self.why}"
      self._event("mapdrive", t, phase="end", finished=self.finished)
    self.msg = {"type": "control", "active": True, "curvature": 0.0, "accel": self.a}
    return self.msg

  def _checks(self, route, state: dict, t: float, collisions: int, v: float):
    pos = state["pos"]
    self.s, self.dev = project(self.path, self.s_path, pos, self.s)

    def held(name: str, cond: bool, secs: float) -> bool:
      start = self.timers.get(name)
      start = (start if start is not None else t) if cond else None
      self.timers[name] = start
      return start is not None and t - start >= secs
    user = state.get("user") or {}
    engage = state.get("engagePresses")
    if collisions > (self.collisions0 or 0):
      self._abort("collision", t, pos)
    elif abs(float(user.get("steer") or 0.0)) > USER_STEER or user.get("brake") or user.get("gas"):
      self._abort("driver input", t, pos)
    elif engage is not None and self.engage0 is not None and engage > self.engage0:
      self._abort("the driver pressed the engage key", t, pos)
    elif held("dev", self.dev > DEV_ABORT, DEV_ABORT_S):
      self._abort(f"{self.dev:.1f} m off the path", t, pos)
    herr = abs(wrap(float(state.get("heading") or 0.0) - game_heading(float(np.interp(self.s, self.s_path, self.theta)))))
    if self.phase != "abort" and held("heading", herr > HEADING_ABORT and v > 2.0, HEADING_ABORT_S):
      self._abort(f"heading {herr:.0f} deg off the path", t, pos)
    a_lat = abs(v * float(state.get("yawRate") or 0.0))
    if self.phase != "abort" and held("a_lat", a_lat > A_LAT_MAX, A_LAT_ABORT_S):
      self._abort(f"lateral acceleration {a_lat:.1f} m/s^2", t, pos)
    if self.s > self.progress[0] + 1.0:
      self.progress = (self.s, t)
    waiting = self.phase in ("stop", "arrive") or self.stopped_at is not None
    if waiting:
      self.progress = (self.progress[0], t)
    if self.phase != "abort" and t - self.progress[1] > NO_PROGRESS_S:
      self._abort(f"no progress in {NO_PROGRESS_S:.0f} s", t, pos)
    if self.phase != "abort" and route.off < 30.0 and not route.elsewhere:
      sec = route.section(route.seg)
      near_j = self._unclear(route.at) or self.s < self.join_m
      if sec is not None and sec.lanes and not near_j and not (sec.edges[0] - OFF_ROAD <= route.right <= sec.edges[1] + OFF_ROAD):
        if held("off_road", True, 0.5):
          self._abort(f"off the road ({route.right:+.1f} m right of its line)", t, pos)
      else:
        self.timers["off_road"] = None

  def _longitudinal(self, state: dict, v: float, t: float, dt: float) -> float:
    st = self.style
    s = self.s
    sp = self.s_path
    look = max(2.0, 1.2 * v)
    vp_here = float(np.interp(s, sp, self.v_static))
    vp = float(np.interp(s + look, sp, self.v_static))
    reason = REASONS[int(self.why_static[min(int((s + look) / STEP), len(sp) - 1)])]
    phase = "drive"
    b = st["decel"]
    # the next stop sign or give way line not yet done
    for x in self.stops:
      if x["done"]:
        continue
      if x["kind"] == "lights":
        if s + FRONT >= x["s"]:
          x["done"] = True
          mark = {"t": round(t, 3), "s": round(x["s"], 1), "along": round(x["along"], 1), "from": round(t - LIGHT_MARK, 3), "to": round(t + LIGHT_MARK, 3)}
          self.lights.append(mark)
          self._event("light_unknown", t, **{k: v for k, v in mark.items() if k != "t"})
        continue
      d = x["target"] - s
      if x["kind"] == "give_way":
        if s + FRONT >= x["s"]:
          x["done"] = True
          continue
        cap = math.sqrt((GIVE_WAY_SPEED * st["speed"]) ** 2 + 2 * b * 0.8 * max(d, 0.0))
        if cap < vp:
          vp, reason = cap, "give_way"
          if d < 30.0:
            phase = "give_way"
        break
      # a stop sign: a full stop at its target, the dwell, then on
      if x["t_stop"] is not None:
        if t - x["t_stop"] >= st["stop_dwell"] + st["reaction"]:
          x["done"] = True
          self.stopped_at = None
          continue
        self.stopped_at = x["t_stop"]
        self.v_prof, self.reason = 0.0, "stop"
        self._set("stop", t)
        return self._out_accel(HOLD_ACCEL, dt, hard=True)
      if (d < STOP_NEAR and v < STOPPED) or d < -1.0:
        x["t_stop"] = t
        self.stopped_at = t
        self._event("stopped", t, kind="stop", miss=round(d, 2))
        self.v_prof, self.reason = 0.0, "stop"
        self._set("stop", t)
        return self._out_accel(HOLD_ACCEL, dt, hard=True)
      cap = math.sqrt(2 * b * 0.8 * max(d - 0.3, 0.0))
      if cap < vp:
        vp, reason = cap, "stop"
        if d < 40.0:
          phase = "stop"
      break
    # the arrival
    d_end = self.s_arrive - s
    if d_end < STOP_NEAR and v < STOPPED or d_end < -1.0:
      if self.arrived_t is None:
        self.arrived_t = t
        self._event("arrived", t, miss=round(d_end, 2))
      self._set("arrive", t)
      self.v_prof, self.reason = 0.0, "arrive"
      if t - self.arrived_t >= float(self.c["hold_after"]) and self.finished is None:
        self.finished = "arrived"
        self._event("mapdrive", t, phase="end", finished="arrived")
      return self._out_accel(HOLD_ACCEL, dt, hard=True)
    self._set(phase, t)
    self.v_prof, self.reason = min(vp_here, vp), reason
    if v < 0.5 and vp > 0.5 and reason not in ("stop", "arrive"):
      self.reason = "start"
    a = (vp * vp - v * v) / (2.0 * look)
    if vp < 0.3 and v < 1.0:
      a = min(a, -0.5) if (self.s_arrive - s) > 0 else HOLD_ACCEL
    a = float(np.clip(a, -DECEL_MAX, st["accel"]))
    return self._out_accel(a, dt, hard=a < -b)

  def _out_accel(self, a: float, dt: float, hard: bool = False) -> float:
    j = self.style["jerk"] * (3.0 if hard else 1.0)
    if a < self.a:
      a = max(a, self.a - 2.5 * j * dt) if not hard else max(a, self.a - 4.0 * j * dt)
    else:
      a = min(a, self.a + j * dt)
    self.a = float(a)
    return self.a

  def _lateral(self, state: dict, v: float, t: float, dt: float) -> float:
    pos = np.asarray(state["pos"][:2], float)
    h = float(state.get("heading") or 0.0)
    yaw = float(state.get("yawRate") or 0.0)
    # the pose when the control acts
    hm = math.radians(h + math.degrees(yaw * LATENCY) / 2)
    pred = pos + v * LATENCY * np.array([-math.sin(hm), math.cos(hm)])
    h_pred = h + math.degrees(yaw * LATENCY)
    s_pred, _ = project(self.path, self.s_path, pred, self.s, window=40.0)
    ld = min(max(LOOKAHEAD[0], LOOKAHEAD[1] * v + LOOKAHEAD[2]), LOOKAHEAD[3])
    sp = self.s_path
    k_ff = float(np.interp(s_pred + v * FF_PREVIEW, sp, self.kappa_path))
    k_pp = pursuit(self.path, sp, s_pred, pred, h_pred, v, ld)
    # the pursuit's own curvature from the path itself (its chord), taken out as the feedforward stands for it; the
    # heading interpolated, as a per-point one steps several degrees in a tight bend and makes the steering chatter
    ideal = np.array([np.interp(s_pred, sp, self.path[:, 0]), np.interp(s_pred, sp, self.path[:, 1])])
    k_pp0 = pursuit(self.path, sp, s_pred, ideal, game_heading(float(np.interp(s_pred, sp, self.theta))), v, ld)
    k = k_ff + (k_pp - k_pp0)
    vv = max(v, 1.0)
    cap = min(KAPPA_MAX, A_LAT_MAX / (vv * vv))
    rate = max(RATE_SHARE * self.style["j_lat"] / (vv * vv), 0.03)
    want = k
    k = float(np.clip(k, -cap, cap))
    k = float(np.clip(k, self.kappa - rate * dt, self.kappa + rate * dt))
    sat = abs(want - k) > max(0.15 * abs(want), 0.002) and v > 2.0
    start = self.timers.get("sat")
    self.timers["sat"] = (start if start is not None else t) if sat else None
    if self.timers["sat"] is not None and t - self.timers["sat"] >= SATURATED_S:
      self._anomaly("tracking_saturated", t, pos, self.s, want=round(want, 4), got=round(k, 4), v=round(v, 1))
    self.kappa = k
    return k

  def _labels(self, route, state: dict, v: float, t: float):
    """The desire the car signals: a lane change from signal_lead s before it until most of the way across, a turn or
    signalled keep as the AI expert signals them."""
    s = self.s
    rs = float(np.interp(s, self.s_path, self.route_s))
    self.target_lane = lane_at(self.keys, rs)
    sec = route.section(segment_of(route, rs))
    self.target_right = round(sec.offset(self.target_lane), 2) if sec is not None and sec.lanes else math.nan
    self.lc_tau = math.nan
    side = label = None
    for s0, s1, d in self.changes:
      if s0 - self.style["signal_lead"] * max(v, 3.0) <= s < s0 + LC_DONE * (s1 - s0):
        side, label = d, "laneChange" + d.capitalize()
      if s0 <= s <= s1:
        self.lc_tau = (s - s0) / max(s1 - s0, EPS)
    at, heading = route.at, float(state.get("heading") or 0.0)
    for i, m in enumerate(self.mans):
      if i in self.done_mans:
        continue
      d = m.along - at
      over = (d < -TURN_PAST or (d < -TURN_DONE_AFTER and abs(wrap(heading - m.exit_heading)) < TURN_DONE_HEADING)) if m.turn else d < -KEEP_PAST
      if over:
        self.done_mans.add(i)
        continue
      if d <= max(SIGNAL_DIST, v * SIGNAL_TIME):
        side = ("left" if m.desire.endswith("Left") else "right") if m.signal else side
        label = m.desire
      break
    self.indicator, self.label = side, label or "none"

  def _watch(self, route, state: dict, v: float, t: float):
    """Live map anomalies: the car's side over the kerb, the map's lane matching disagreeing with the plan."""
    pos = state["pos"]
    near_j = self._unclear(route.at)
    if route.off < 30.0 and not route.elsewhere and not near_j and self.s >= self.join_m:
      sec = route.section(route.seg)
      if sec is not None and sec.lanes and (route.right - HALF_WIDTH < sec.edges[0] - KERB_MARGIN or route.right + HALF_WIDTH > sec.edges[1] + KERB_MARGIN):
        self._anomaly("kerb_contact", t, pos, self.s, planned=False, right=round(route.right, 2), edges=[round(e, 2) for e in sec.edges])
    m = state.get("laneMap") or {}
    changing = not math.isnan(self.lc_tau)
    sec = route.section(route.seg)
    bad = (not near_j and not changing and m.get("kind") == "own" and m.get("lane") is not None and sec is not None and
           m.get("lanes") == sec.lanes and abs(self.target_lane - round(self.target_lane)) < 0.1 and m["lane"] != round(self.target_lane))
    start = self.timers.get("lane")
    self.timers["lane"] = (start if start is not None else t) if bad else None
    if self.timers["lane"] is not None and t - self.timers["lane"] >= LANE_DISAGREE_S:
      self._anomaly("lane_disagree", t, pos, self.s, plan=round(self.target_lane, 2), matched=[m.get("lane"), m.get("lanes")])


def route_point(route, s: float) -> np.ndarray:
  return np.array([np.interp(s, route.along, route.points[:, 0]), np.interp(s, route.along, route.points[:, 1])])
