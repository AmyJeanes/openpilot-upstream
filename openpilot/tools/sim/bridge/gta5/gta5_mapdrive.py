"""Map driver: expert mode driving our route itself along the map's lanes, rather than the game's AI (gta5_expert.py
with driver=map; design in the openpilot-gta5 task's MAPDRIVER_DESIGN.md). Traffic off for now: no gaps for lane changes,
but other vehicles (the plugin's nearby; traffic off still leaves some) are kept clear of along the path.

The plan is made once when a route arrives (and spliced on a reroute, past the lane change under way and 3 s on):
nav's lane plan (planner.lane_plan, as the ribbon draws it) with absolute along-route keys, each lane change placed
within its lane-slot window (by the trip's style, freeway ones within +-20 % of nav's schedule), then re-placed to take
its time (the style's t_lc, no quicker than its lateral acceleration allows) at the speed the car will drive it, its end
kept, moved off jogs in the route's line (route_jogs, where the lane line swerves; jog_change where there's no
room), and reshaped as a quintic (minimum-jerk) move; then put on the lanes by RouteLanes.lane_line (its offsets, jogs
and fillets), bends the car can't take as drawn eased out (sharp_corner), swinging wide rather than cutting in, and
kept on its lane (_clamp, plan_clamp): our body out of each turn's inside corner kerb (a far-side turn's where it's a
median), within its lane outside lane changes and turns' corners, and past a fork off its gore. That line is the
intent, recorded as the path labels; the car aims for it plus a small, slow in-lane bias and wander (never periodic,
faded out through turns, junctions and forks, and within its lane), joining it from where the car is. The plan is
checked against the lane slots' targets (lane_slots.py, compared by where the lanes are): where the slots want another
lane, plan_check=fix (the default) changes into it by the target's end, event only logs it, abort ends the trip; each
is a plan_slot_mismatch anomaly.

Each game frame: pure pursuit (lookahead clamp(5, 0.6 v + 3, 25) m) from the rear axle (the game's position is the
car's middle, which moves inwards of its heading in a bend), predicted `latency` (0.1) s on, plus the path's curvature
`ff_preview` (0.32) s ahead as feedforward (the pursuit's own curve-cutting taken out), rate limited to 3x the style's
lateral jerk (more once off the path) and capped at 4.5 m/s^2; the speed follows a static profile (limit or road class x style, the path's
curvature's envelope at the style's lateral acceleration and lateral jerk, nav's turn speeds) with stop signs (full stop and
dwell), give way lines (slow, then on), traffic lights (no state to read: driven through and marked light_unknown +-5 s)
and a gentle arrival stop_before m short of the end. Other vehicles: one whose body comes within LEAD_SIDE of ours along
the path ahead (in a turn too) is followed (the intelligent driver model's braking, by the style's headway) or stopped
behind, LEAD_STOP short; before a junction or turn the car waits while a moving one's way (straight on at its speed)
crosses ours within CROSS_GAP s of when we'd be there; and it's no faster than stops for one at the edge of what the
plugin reports (its nearby's "ahead", 15 m where it has none), but for range_share (0.75) of the speed. Aborts (dev >
1.5 m for 1 s, heading off by 30 deg, a collision, off the road, no progress for 20 s, held by a vehicle for WAIT_MAX s,
lateral acceleration over 4.5, the driver's input, a failed plan) brake to a stop and end the trip (on_abort=ai hands it
to the game's AI). Map anomalies are logged with their place, for the map-fix work:
kerb_contact (past a road's kerbs, or into a turn's corner kerb, from its legs' cross-sections), gta_edge (the
plan's lane near GTA's road edge where the map's road is wider), lane_disagree, tracking_saturated, gta_link_offset,
stop_line_far, plan_slot_mismatch, sharp_corner, jog_change, plan_clamp.

The control file's `mapdrive` object sets it up: {"seed": 7, "preset": "normal", "style": {"t_lc": 5.0}, "bias_max":
0.3, "wander": 0.1, "on_abort": "stop", "speed": null, "plan_check": "fix", "latency": 0.1, "ff_preview": 0.32,
"rear_axle": null, "range_share": 0.75}; bias_max 0 and wander 0 drive exactly on the line; rear_axle (m behind the car's
position, default half the state's wheelBase) 0 steers from the car's middle; range_share 1 drops the range cap."""
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

PHASES = ("", "wait", "drive", "stop", "give_way", "arrive", "done", "abort", "follow", "yield")  # gta5.npz mapx phase codes: the index
REASONS = ("", "limit", "curve", "turn", "stop", "give_way", "arrive", "start", "abort", "hold", "lead", "cross", "range")
# lc_tau: 0..1 through the lane change under way by path distance, NaN outside one; lc_dir: its side, -1 left / +1 right
MAPX_COLUMNS = ("phase", "plan_s", "path_dev", "target_lane", "target_right", "v_prof", "a_cmd", "kappa_cmd", "speed_reason",
                "lc_tau", "lead_gap", "lc_dir")
SOURCES = ("none", "ai", "map", "ai_fallback")  # gta5.npz expert_src codes

STEP = 1.0  # m between the path's points
LANE_W = 5.5  # m, GTA's lanes
FRONT = 2.4  # m from the car's origin to its front bumper (and its rear), where the plugin's nearby.dims has none
WHEEL_BASE = 2.8  # m, where the plugin's state has none: the rear axle half of it behind the car's origin
HALF_WIDTH = 1.0  # m, likewise
LATENCY = 0.1  # s from the game's frame to the plugin acting on the control (the pose is predicted this far)
FF_PREVIEW = 0.32  # s ahead the path's curvature is fed forward: the car's curvature lag (about 0.2 s in game) and the delay
LOOKAHEAD = (5.0, 0.6, 3.0, 25.0)  # m: min, s of speed, plus, max
KAPPA_MAX = 0.2  # 1/m
A_LAT_MAX = 4.5  # m/s^2 commanded at most, and an abort measured above it
CURV_SMOOTH = 7.0  # m: the path's curvature averaged over this
FADE_NEAR, FADE_OVER = 15.0, 25.0  # m: in-lane bias and wander gone this near a turn or junction node, back over this
SOFT_IN, SOFT_OUT = 0.5, 1.5  # m an eased corner may come inside the line drawn, and swing outside it before and after
JERK_SHARE = 2.0  # x the style's lateral jerk the speed profile allows where the path's curvature changes
RATE_OFF_PATH = 2.0  # x more of it, 1 m off the path
RATE_SHARE = 3.0  # x it the commanded curvature may change at: a safety limit above the profile's
COMPUTE_EVERY = 0.1  # s: a new frame is computed at once, else at least this often; between, the last control is resent
DECEL_MAX = 4.0  # m/s^2
ABORT_DECEL = 3.5
HOLD_ACCEL = -1.5  # m/s^2 holding a stop
STOPPED = 0.25  # m/s
STOP_NEAR = 2.0  # m from a stop target, stopped: at it
STOP_LAG = 0.3  # s the car carries on at its speed before braking takes hold
GIVE_WAY_SPEED = 3.0  # m/s at the line
LIGHT_MARK = 5.0  # s either side of a light passed without its state
SIGNAL_TIME, SIGNAL_DIST = 5.0, 50.0  # s, m before a turn (as the AI expert)
TURN_DONE_HEADING, TURN_DONE_AFTER, TURN_PAST, KEEP_PAST = 20.0, 5.0, 40.0, 30.0
JOG_LINK, JOG_TURN, JOG_BACK, JOG_PAD = 25.0, 10.0, 12.0, 10.0  # m, deg, deg, m (route_jogs)
MIN_LC_M = 20.0  # m a lane change takes at the least
LC_A_LAT = 0.6  # x the style's lateral acceleration, a lane change's peak at most
LC_MAX_S = 9.0  # s a change takes at the most
LC_SHORTEST = 12.0  # m a change re-timed to its time takes at the least, a lane (pulling away, slowly)
LC_FIT = 0.15  # a change timed more than this share off its time (by the speed it'll be driven at) is re-placed
LC_MIN_SHARE = 0.6  # of its time a change squeezed by what's before it keeps at least, else it's left as it was
JOIN_MAX = 8.0  # m from its lane line the car may start
BEND_DEG = 20.0  # deg a route point turns that makes its road beside it unclear
KAPPA_DRIVABLE = 0.1  # 1/m: a tighter bend in the lane line (a jog, a corner without a fillet) is eased out
LC_DONE = 0.8  # share of a lane change after which its signal goes off
CROSSING_CLEAR = 5.0  # m past a stop line or junction node a city lane change may start
TURN_SPAN = 10.0  # m either side of a turn's point held to its turn speed
# vehicles (the plugin's nearby: their bodies in our frame, read every 0.1 s)
LEAD_SIDE = 0.4  # m beside our body another's may come along the path before it's in the way
LEAD_AHEAD = 80.0  # m of path looked along
LEAD_OFF = 6.0  # m off the path at most a body's point is placed along it
LEAD_STOP = 4.0  # m from our front bumper to a stopped vehicle we stop at
LEAD_BESIDE = 1.0  # m behind our front bumper a body may start and still be ahead (not beside us)
NEARBY_AHEAD = 15.0  # m ahead of our origin the plugin reports vehicles, where its nearby has no "ahead"
NEARBY_SIDE = 15.0  # m either side, likewise ("side")
RANGE_LAG = 0.6  # s before braking takes hold, for the range cap (the command's delay, its jerk and the car's lag)
RANGE_STOP = 1.0  # m short of a vehicle at the range's end the range cap stops at
RANGE_BODY = 2.5  # m of its body before its middle (which the plugin's reach is to)
WAIT_MAX = 60.0  # s held by a vehicle in the way before the trip ends
CROSS_MOVING = 2.0  # m/s: a slower vehicle isn't crossing
CROSS_T = 6.0  # s ahead a crossing vehicle's way is predicted (straight on at its speed)
CROSS_GAP = 2.0  # s between it and us at the place our ways cross, at the least
CROSS_ANGLE = 25.0  # deg off our way at least, for a crossing
CROSS_NEAR = 25.0  # m from a junction node or turn the crossing may be, to be held for
CROSS_STOP = 2.0  # m short of its way we stop at
CROSS_KEEP = 0.5  # s a hold is kept after the crossing last looked likely
# aborts
DEV_ABORT, DEV_ABORT_S = 1.5, 1.0
HEADING_ABORT, HEADING_ABORT_S = 30.0, 0.5
NO_PROGRESS_S = 20.0
A_LAT_ABORT_S = 0.5
OFF_ROAD = 1.0  # m past the kerb
OFF_ROAD_DEV = 0.75  # m off the path as well, for an off-road abort
USER_STEER = 0.02
# anomalies
KERB_MARGIN = 0.2  # m the car's side may come past the kerb before it counts
CORNER_LEG = 25.0  # m before and after a turn's point its legs' kerbs are taken, outside the junction
CORNER_ANGLE = (30.0, 150.0)  # deg a turn checked against its corner kerb turns
CORNER_LEG_MIN = 6.0  # m: a corner nearer the next than twice this has no legs of its own to take
CORNER_SAME = 10.0  # m between a turn and a corner of the lane line that are one
MEDIAN_GAP = 1.0  # m between the two directions that's a median
MEDIAN_RADIUS = 2.0  # m a median's nose is taken as rounded
CORNER_NODE = 20.0  # m from the turn's point to a junction node, for a junction's corner
CORNER_NEAR = 20.0  # m from the turn's point the kerb lines' corner may be
CORNER_INSIDE = 0.5  # m the plan keeps inside each leg's kerb line, or the line isn't the kerb
CORNER_SPAN = 40.0  # m either side of a turn's point checked against its corner kerb
KERB_RADIUS = 6.0  # m: a junction's corner kerbs taken as rounded this much (the map has no corner geometry)
EDGE_WIDER = 0.75  # m the map's road edge may lie past GTA's before a lane near GTA's edge is suspect
EDGE_CLEAR = 0.8  # m from the car's side to GTA's road edge (its link's lanes) below which the plan is too near it
JUNCTION_CLEAR = 25.0  # m from a junction node, where cross-sections don't describe the road, kerbs aren't checked
LANE_DISAGREE_S = 2.0
SATURATED_S = 0.5
GTA_OFFSET = 1.5  # m between GTA's lane centre and the map's for the planned lane
STOP_LINE_FAR = 30.0  # m from a stop line on to its junction's first node
ANOMALY_SPACING = 20.0  # m between anomalies of one kind
# the plan kept on its lane (_clamp)
CLAMP_MARGIN = 0.3  # m our body keeps in from a corner's kerb or its lane's edges
CLAMP_TURN = 45.0  # deg a turn turns, about whose point the lane's edges are left to the corner kerbs
CORNER_ZONE = 15.0  # m either side of such a turn's point
CLAMP_EASE = 15.0  # m a push eases in and out over
LANE_GRID = 2.0  # m between the places the lane is read at
CLAMP_MAX = 2.5  # m a push is at most
FORK_BEFORE, FORK_AFTER = 20.0, 40.0  # m before and after a fork its gore side is kept to
FORK_ROOM = 0.3  # m from the lane's centre towards the gore at most
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
  "reaction": (0.4, 1.5, -1), "bias": (-1.0, 1.0, 0), "headway": (1.0, 2.0, -1),
}
PRESETS = {"calm": 0.2, "normal": 0.5, "brisk": 0.8}
DEFAULTS = {"seed": None, "preset": "normal", "style": {}, "bias_max": 0.3, "wander": 0.1, "wander_m": 300.0,
            "on_abort": "stop", "speed": None, "plan_check": "fix", "stop_before": 15.0, "hold_after": 3.0, "retime": True,
            "drive_on_right": True, "latency": LATENCY, "ff_preview": FF_PREVIEW, "fit_durations": True,
            "rear_axle": None, "range_share": 0.75}


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
  place, where the road's lanes change) inside it. Two ramps meeting in a lane's centre where it doesn't renumber
  (keys alike), or turning back the other way, are two changes, one after the other."""
  out, i, n = [], 0, len(keys)
  while i < n - 1:
    if not _ramp(keys, i):
      # a change can begin with a jump in its numbering where it starts (nav's ramp across a lane map at its start)
      if not (i + 2 < n and keys[i + 1][0] - keys[i][0] <= EPS and abs(keys[i + 1][1] - round(keys[i + 1][1])) > 0.05 and
              _ramp(keys, i + 1) and (keys[i + 2][1] - keys[i + 1][1]) * (keys[i + 1][1] - keys[i][1]) > 0):
        i += 1
        continue
    j = i
    way = keys[i + 1][1] - keys[i][1]

    def same_way(m: int) -> bool:
      return (keys[m + 1][1] - keys[m][1]) * way > 0
    def joined(m: int) -> bool:  # keys m and m + 1 at one place, inside a change: renumbered, or not at a lane's centre
      return keys[m + 1][0] - keys[m][0] <= EPS and (abs(keys[m + 1][1] - keys[m][1]) > EPS or
                                                     abs(keys[m][1] - round(keys[m][1])) > 0.05)
    while j < n - 1 and ((_ramp(keys, j) and same_way(j)) or
                         (joined(j) and j + 1 < n - 1 and _ramp(keys, j + 1) and same_way(j + 1))):
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
  """s a change across `lanes` lanes takes: the style's (more lanes by their square root), and no quicker than a peak
  lateral acceleration of LC_A_LAT x the style's allows (a quintic's peak is 5.77 D / T^2)."""
  n = max(abs(lanes), 1.0)
  a = LC_A_LAT * style["a_lat"]
  return float(min(max(style["t_lc"] * math.sqrt(n), math.sqrt(5.77 * LANE_W * n / a)), LC_MAX_S))


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
    lo, hi = max(int(a) - 60, 0), min(int(b) + 60, len(p))
    w = p[lo:hi].copy()
    orig = w.copy()
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
    # easing a corner cuts inside it, towards the kerb's corner: no more than SOFT_IN, the curve moved out instead
    # (up to SOFT_OUT, eased in and out over the window), as a driver swings wide for a corner too tight to take
    peak = float(k[a:b][np.argmax(np.abs(k[a:b]))])
    th = headings(orig)
    inward = np.sign(peak) * np.stack([-np.sin(th), np.cos(th)], axis=1)  # towards the corner's centre
    cut = np.einsum("ij,ij->i", w - orig, inward)
    excess = min(float(cut.max()) - SOFT_IN, SOFT_OUT)
    if excess > 0:
      n = len(w)
      x = np.arange(n) / max(n - 1, 1)
      centre = (int((a + b) // 2) - lo) / max(n - 1, 1)
      u = np.clip(np.where(x < centre, x / max(centre, 1e-6), (1 - x) / max(1 - centre, 1e-6)) * 1.6, 0.0, 1.0)
      w -= inward * (excess * u * u * (3 - 2 * u))[:, None]
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


def route_jogs(route) -> list[tuple[float, float]]:
  """Where the route's line jogs sideways and back (a short link turning JOG_TURN or more one way, then back within
  JOG_BACK of the way it was going, as across a junction where a road's line moves over to its other carriageway's):
  [(m along from, to)], padded JOG_PAD either side. The lane line swerves there, by the lanes' offsets from a diagonal."""
  p = route.points[:, :2]
  if len(p) < 4:
    return []
  d = np.diff(p, axis=0)
  h = np.degrees(np.arctan2(d[:, 1], d[:, 0]))
  seg = np.hypot(*d.T)
  out = []
  for k in range(1, len(d) - 1):
    if seg[k] > JOG_LINK or seg[k] < 1e-3:
      continue
    t_in, t_out = wrap(h[k] - h[k - 1]), wrap(h[k + 1] - h[k])
    if abs(t_in) >= JOG_TURN and abs(t_out) >= JOG_TURN and t_in * t_out < 0 and abs(t_in + t_out) < JOG_BACK:
      out.append((float(route.along[k]) - JOG_PAD, float(route.along[k + 1]) + JOG_PAD))
  return out


def along_route(route, pts: np.ndarray, start: float) -> tuple[np.ndarray, np.ndarray]:
  """Each point's (m along the route, m right of its line), searched on from `start` m along so where the route
  passes back near itself the points stay on their own part."""
  rp, ra = route.points[:, :2], route.along
  k = max(int(np.searchsorted(ra, start, side="right")) - 1, 0)
  out_s, out_r = np.empty(len(pts)), np.empty(len(pts))
  batch = 32
  for n0 in range(0, len(pts), batch):
    p = pts[n0:n0 + batch]
    # the segments from a little behind the last point's to as far on as the batch could reach
    lo = max(k - 1, 0)
    hi = min(max(int(np.searchsorted(ra, ra[k] + batch * STEP * 1.5 + 30.0)), lo + 2), len(rp) - 1)
    a, ab = rp[lo:hi], rp[lo + 1:hi + 1] - rp[lo:hi]
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9)
    rel = p[:, None, :] - a[None, :, :]
    t = np.clip(np.einsum("nij,ij->ni", rel, ab) / ab2, 0.0, 1.0)
    d = np.hypot(*(rel - ab[None] * t[..., None]).transpose(2, 0, 1))
    i = np.argmin(d, axis=1)
    n = np.arange(len(p))
    seg = np.sqrt(ab2[i])
    out_s[n0:n0 + batch] = ra[lo + i] + t[n, i] * seg
    out_r[n0:n0 + batch] = (rel[n, i, 0] * ab[i, 1] - rel[n, i, 1] * ab[i, 0]) / seg
    k = lo + int(i[-1])
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
    self.lc_dir = math.nan
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
    self.corners: list[dict] = []
    self.clamps: list[dict] = []  # where _clamp moved the plan back: {"i": path index, "push": m right, "why", "length"}
    self.jog_changes: list[tuple[float, float]] = []  # lane changes left across a jog, with no room off it
    self.unclear = np.zeros(0)
    self.front, self.rear, self.half_width = FRONT, FRONT, HALF_WIDTH  # our car's body from its origin (_body)
    self.cars: list[dict] | None = None  # the vehicles around, in the world (_vehicles); None: the plugin reports none
    self.cars_rows = None
    self.reach = (NEARBY_AHEAD, NEARBY_SIDE)  # m ahead of our origin and to either side the plugin reports them
    self.lead_gap = math.nan  # m from our front bumper to the vehicle in our way along the path
    self.held_since: float | None = None  # when a vehicle in the way began holding the car up
    self.cross_hold: tuple | None = None  # (path m we stop at, when, details) of the last crossing held for

  def _body(self, state: dict):
    """Our car's body from the plugin's nearby.dims (its model's bounds: min x, max x, min y, max y)."""
    dims = (state.get("nearby") or {}).get("dims")
    if dims and len(dims) == 4 and dims[3] > 0.5 and dims[2] < -0.5 and dims[1] - dims[0] > 0.5:
      self.front, self.rear, self.half_width = float(dims[3]), float(-dims[2]), float(max(-dims[0], dims[1]))

  @property
  def body(self) -> tuple[float, float, float]:
    return self.front, self.rear, self.half_width

  # *** what expert mode and recordings read ***

  def info(self) -> dict:
    return {"phase": self.phase, "s": round(self.s, 1), "dev": round(self.dev, 2), "lane": None if math.isnan(self.target_lane) else round(self.target_lane, 2),
            "vProf": None if math.isnan(self.v_prof) else round(self.v_prof, 2), "a": round(self.a, 2), "kappa": round(self.kappa, 5),
            "reason": self.reason, "lcTau": None if math.isnan(self.lc_tau) else round(self.lc_tau, 2), "label": self.label,
            "lead": None if math.isnan(self.lead_gap) else round(self.lead_gap, 1)}

  def mapx_row(self) -> list[float]:
    return [PHASES.index(self.phase) if self.phase in PHASES else 0, self.s, self.dev, self.target_lane, self.target_right, self.v_prof,
            self.a, self.kappa, REASONS.index(self.reason) if self.reason in REASONS else 0, self.lc_tau, self.lead_gap, self.lc_dir]

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
    if phase != self.phase and self.phase != "abort":  # an abort ends the trip: nothing leaves it
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
    out, last = [], -math.inf
    for d, lane in keys:  # in order along: nav's can step back a few cm
      last = max(last, route.at + d)
      out.append((last, float(lane)))
    return out

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
      j = a - 1  # back over holds in its lane to the change, step or start before it
      while j > 0 and abs(keys[j - 1][1] - keys[j][1]) < EPS and keys[j][0] - keys[j - 1][0] > EPS:
        j -= 1
      lower = keys[j][0] if a > 0 else s0
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
      if not (lower - EPS <= start < e <= upper + EPS):
        continue
      keys[a], keys[b] = (float(start), la), (float(e), lb)
    return keys

  def slot_check(self, route, keys: list[tuple[float, float]], slots) -> dict:
    """The plan against the lane slots' targets: from each target's end to its move, the plan's lane (rounded) must be
    one of them, compared by where the lanes are (the slots number the lanes as they are open, the plan as the road's
    lanes carry on)."""
    out = {"checked": 0, "mismatches": []}
    if slots is None:
      return out
    for m in slots.moves:
      if not m.need or m.along <= route.at + 5.0:
        continue
      v = self._speed_at(route, m.along)
      _, _, end = m.window(v)
      # where the lanes renumber between (a bay or lane opening, which nav moves into only once it's there), from a
      # lane change's length past it
      steps = [keys[i][0] for i in range(len(keys) - 1) if keys[i + 1][0] - keys[i][0] <= EPS and
               abs(keys[i + 1][1] - keys[i][1]) > EPS and end - EPS <= keys[i][0] < m.along - 1.0]
      first = max([end] + [x + MIN_LC_M for x in steps] + ([m.opens + MIN_LC_M] if m.opens is not None else []))
      bad = None
      for s in np.arange(max(first, route.at + 1.0), m.along - 2.0, 2.0):
        good, target = self._slot_lanes(route, slots, float(s), v)
        if not good or target.move is not m:
          continue
        out["checked"] += 1
        lane = lane_at(keys, float(s))
        if int(round(lane)) not in good and abs(lane - round(lane)) < 0.25:
          bad = {"s": round(float(s), 1), "move": round(m.along, 1), "end": round(float(end), 1), "plan": round(lane, 2),
                 "want": sorted(good), "slots": sorted(target.want)}
          break
      if bad is not None:
        out["mismatches"].append(bad)
    return out

  @staticmethod
  def _slot_lanes(route, slots, s: float, v: float):
    """The slots' target lanes s m along as the plan numbers them (RouteLanes.section_at's lanes whose centres lie in
    a target lane as the slots have them, opened_at's), and the target; (None, target) without lanes to aim for."""
    here, target = slots.target(s, v)
    if target is None or target.centre or not target.want or here is None:
      return None, target
    sec = route.lanes.section_at(s)
    if sec is None or not sec.lanes:
      return None, target
    spans = [here.ours[w] for w in target.want if w < len(here.ours)]
    good = {i for i in range(sec.lanes) if any(sp.left - 0.3 <= sec.offset(i) <= sp.right + 0.3 for sp in spans)}
    return good or None, target

  def follow_slots(self, route, keys: list[tuple[float, float]], mismatches: list[dict], slots) -> list[tuple[float, float]]:
    """The keys with a lane change into the slots' nearest target lane for each mismatch, done by its target's end
    (from a lane change's length before, not over the plan's change before), then in that lane to its move: the plan's
    own changes there dropped, its renumbering steps (lanes beginning or ending) kept."""
    keys = list(keys)
    for bad in sorted(mismatches, key=lambda b: b["s"]):
      end, move = min(bad["end"], bad["s"]), bad["move"]
      v = self._speed_at(route, end)
      good, _ = self._slot_lanes(route, slots, end + 0.1, v)
      want = sorted(good) if good else bad["want"]
      p = round(lane_at(keys, end + 0.1))
      w = min(want, key=lambda x: abs(x - p))
      delta = w - p
      if delta == 0:
        continue
      length = max(lc_seconds(self.style, delta) * v, MIN_LC_M * abs(delta))
      changes = [keys[b][0] for a, b in chains(keys) if keys[b][0] <= end]
      start = max(end - length, max(changes, default=keys[0][0]), keys[0][0])
      if end - start < 0.4 * length:
        continue  # no room: left as it is (still a mismatch)
      out = [k for k in keys if k[0] <= start] + [(start, lane_at(keys, start)), (end, float(w))]
      lane, i = float(w), 0
      inside = [k for k in keys if end < k[0] < move - EPS]
      while i < len(inside):
        s, a = inside[i]
        if i + 1 < len(inside) and inside[i + 1][0] - s <= EPS:  # a renumbering step: carried
          b = inside[i + 1][1]
          out += [(s, lane), (s, lane + (b - a))]
          lane += b - a
          i += 2
          continue
        i += 1  # a point of the plan's own change or hold: the lane stays
      # past the move, the plan on from the lane it's now in: its own changes towards that lane already made (the gap
      # taken up by them), renumbering steps carried
      rest = [k for k in keys if k[0] >= move - EPS]
      pv = lane_at(keys, move - 0.01)
      gap = lane - pv
      out.append((move, lane))
      for n, (s, val) in enumerate(rest):
        step = n > 0 and s - rest[n - 1][0] <= EPS and abs(val - rest[n - 1][1]) > EPS
        dv = val - pv
        if not step and abs(gap) > EPS and dv * gap > 0:
          gap -= math.copysign(min(abs(dv), abs(gap)), gap)
        out.append((s, val + gap))
        pv = val
      keys = out
    return keys

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
    fixed = []
    for _ in range(4 if self.c["plan_check"] == "fix" else 0):  # a fix can leave the next move wanting another lane
      if not check["mismatches"]:
        break
      fixed += check["mismatches"]
      keys = self.follow_slots(route, keys, check["mismatches"], slots)
      check = {**self.slot_check(route, keys, slots), "fixed": fixed}
    for bad in fixed:
      self._anomaly("plan_slot_mismatch", t, route_point(route, bad["s"]), bad["s"], fixed=True, **bad)
    for bad in check["mismatches"]:
      self._anomaly("plan_slot_mismatch", t, route_point(route, bad["s"]), bad["s"] + 0.5, fixed=False, **bad)
    if check["mismatches"] and self.c["plan_check"] == "abort":
      self._abort(f"plan: lane {check['mismatches'][0]['plan']} where the slots want {check['mismatches'][0]['want']}", t, state.get("pos"))
      return False
    at0 = route.at
    built = self._build(route, keys, lane, state, t)
    if built is None:
      return False
    fitted = self.fit_durations(route, keys, float(state.get("vEgo") or 0.0)) if self.c["fit_durations"] else keys
    fitted = self.off_jogs(route, fitted)
    if fitted != keys:
      keys = fitted
      built = self._build(route, keys, lane, state, t)
      if built is None:
        return False
    intent, sharp, knots, bias = built
    for i, k in sharp:
      self._anomaly("sharp_corner", t, intent[min(i, len(intent) - 1)], i * STEP, kappa=round(k, 3))
    for c in self.clamps:
      i = min(c["i"], len(intent) - 1)
      self._anomaly("plan_clamp", t, intent[i], i * STEP, why=c["why"], push=c["push"], length=round(c["length"], 1))
    self.keys, self.nav_plan = keys, nav
    self.plans += 1
    self.plan = {"at0": round(at0, 2), "keys": [(round(s, 2), round(lane_, 3)) for s, lane_ in keys],
                 "nav_keys": [(round(s, 2), round(lane_, 3)) for s, lane_ in nav], "lane": list(lane), "bias": round(bias, 3),
                 "wander_knots": knots, "wander_m": float(self.c["wander_m"]) / 2, "slot_check": check,
                 "stops": [[round(x["along"], 1), x["kind"]] for x in self.stops], "path": self.intent[::2].round(2).tolist()}
    self.changes = self._changes(keys)  # each lane change on the path: (path m from, to, side, m across)
    self._plan_anomalies(route, t)
    return True

  def _build(self, route, keys: list[tuple[float, float]], lane, state: dict, t: float):
    """The path for keys from the car (_derive's): (intent, sharp bends eased, wander knots, bias), None (an abort)
    where there's no line."""
    lanes = route.lanes
    dense = shaped(keys)
    at0 = route.at
    line = lanes.lane_line(at0, [(s - at0, lane_) for s, lane_ in dense if s >= at0 - 1e-6] or [(0.0, float(lane[0]))])
    if line is None or len(line) < 3:
      self._abort("plan: no lane line", t, state.get("pos"))
      return None
    intent, s_path = resample(np.asarray(line, float))
    if len(intent) < 3:
      self._abort("plan: lane line too short", t, state.get("pos"))
      return None
    sharp: list = []
    for n in range(4):  # a near reversal can take more than one pass
      intent, found = soften(intent, KAPPA_DRIVABLE)
      intent, s_path = resample(intent)
      sharp = found if n == 0 else sharp
      if not found:
        break
    intent, room_l, room_r = self._clamp(route, keys, intent)
    intent, s_path = resample(intent)
    bias = self.style["bias"] * float(self.c["bias_max"])
    off, knots = wander(s_path, random.Random(self.seed + 1), float(self.c["wander"]), float(self.c["wander_m"]) / 2)
    n = min(len(off), len(room_l))
    off[:n] = np.clip(bias + off[:n], -room_l[:n], room_r[:n])  # no further than its lane leaves room for
    off = np.clip(off, -0.45, 0.45) * np.clip(s_path / 20.0, 0.0, 1.0) * self._fade(route, intent)
    # from where the car is, across to the line as a lane change would (setup places it in a lane, a reroute may not)
    e0 = self._offset_from(intent, np.asarray(state["pos"][:2], float))
    if abs(e0) > JOIN_MAX:
      self._abort(f"plan: the car is {e0:+.1f} m from its lane line", t, state.get("pos"))
      return None
    v0 = max(float(state.get("vEgo") or 0.0), 5.0)
    join = max(MIN_LC_M, lc_seconds(self.style, e0 / LANE_W) * v0) if abs(e0) > 0.5 else 15.0
    off = off + e0 * (1.0 - quintic(s_path / join))
    self.join_m = max(join, 30.0)
    self._derive(route, intent, off, t, start=(np.asarray(state["pos"][:2], float), float(state.get("heading") or 0.0), v0))
    return intent, sharp, knots, bias

  def _fade(self, route, intent: np.ndarray) -> np.ndarray:
    """1 along the line, easing to 0 within FADE_NEAR m of the route's turns, junction nodes, forks and sharp bends,
    where the car keeps to the line itself."""
    along, _ = along_route(route, intent, route.at - 5.0)
    d = np.diff(route.points[:, :2], axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    bend = np.degrees(np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi))
    spots = np.concatenate((np.asarray(route.junctions, float), route.along[1:-1][bend > BEND_DEG],
                            [route.at + tr[0] for tr in route.turns(route.length)], [f.along for f in route.forks]))
    if not len(spots):
      return np.ones(len(intent))
    spots = np.sort(spots)
    i = np.clip(np.searchsorted(spots, along), 1, len(spots) - 1) if len(spots) > 1 else np.zeros(len(along), int)
    near = np.minimum(np.abs(spots[i] - along), np.abs(spots[np.maximum(i - 1, 0)] - along))
    u = np.clip((near - FADE_NEAR) / FADE_OVER, 0.0, 1.0)
    return u * u * (3 - 2 * u)

  def drive_speeds(self, v0: float) -> np.ndarray:
    """The speed the car will drive the path at (m/s, at least 1): the static profile, stopping at stop signs, and
    no quicker up to it than the style's acceleration from v0 (as from a standstill, or after a stop)."""
    v = self.v_static.copy()
    sp = self.s_path
    b, a = self.style["decel"] * 0.8, self.style["accel"] * 0.8
    for st in self.stops:
      if st["kind"] == "stop":
        i = int(np.clip(np.searchsorted(sp, st["target"]), 0, len(v) - 1))
        v[i] = 0.0
        for j in range(i - 1, -1, -1):
          lim = math.sqrt(v[j + 1] ** 2 + 2 * b * (sp[j + 1] - sp[j]))
          if lim >= v[j]:
            break
          v[j] = lim
    v[0] = min(v[0], max(v0, 0.0))
    for j in range(len(v) - 1):
      v[j + 1] = min(v[j + 1], math.sqrt(v[j] ** 2 + 2 * a * (sp[j + 1] - sp[j])))
    return np.maximum(v, 1.0)

  def fit_durations(self, route, keys: list[tuple[float, float]], v0: float) -> list[tuple[float, float]]:
    """The keys with each lane change re-placed to take its time (lc_seconds) at the speed the car will drive it
    (drive_speeds of the path just built), its end kept: started later where it'd be slow, earlier where quick, never
    before the key before it; city ones not across a stop line or junction node it didn't cross already. Where that
    leaves it short and it's the move into the plan's lanes from the car's (at the plan's start), it ends later
    instead, up to the key after. Renumbering steps inside a change stay at their places."""
    original, keys = keys, list(keys)
    v = self.drive_speeds(v0)
    sp = self.s_path
    tcum = np.concatenate(([0.0], np.cumsum(np.diff(sp) / np.maximum(0.5 * (v[1:] + v[:-1]), 1.0))))
    crossings = sorted(route.stops + route.junctions)

    def path_s(along):
      return float(np.interp(along, self.route_s, sp))

    def along_of(ps):
      return float(np.interp(ps, sp, self.route_s))

    def secs(s_a, s_b):
      return float(np.interp(path_s(s_b), sp, tcum) - np.interp(path_s(s_a), sp, tcum))

    def at_time(s_from, dt):
      return along_of(float(np.interp(np.interp(path_s(s_from), sp, tcum) + dt, tcum, sp)))

    for a, b in reversed(chains(keys)):  # from the last, so earlier indices hold as keys come and go
      (s0, la), (s1, _) = keys[a], keys[b]
      lanes = abs(self._chain_lanes(keys, a, b))
      want = lc_seconds(self.style, lanes)
      if abs(secs(s0, s1) - want) <= LC_FIT * want:
        continue
      j = a - 1  # back over holds in its lane to the change, step or start before it
      while j > 0 and abs(keys[j - 1][1] - keys[j][1]) < EPS and keys[j][0] - keys[j - 1][0] > EPS:
        j -= 1
      lower = keys[j][0] if a > 0 else s0
      upper = keys[b + 1][0] if b + 1 < len(keys) and abs(keys[b + 1][1] - keys[b][1]) < EPS else s1
      shortest = LC_SHORTEST * max(lanes, 1.0)
      s_new = min(max(at_time(s1, -want), lower), along_of(path_s(s1) - shortest))
      k = segment_of(route, s0)
      fwy = (route.classes[k] if k < len(route.classes) else "") in FREEWAY
      if not fwy and any(s_new - CROSSING_CLEAR < c < s1 for c in crossings) and not any(s0 - CROSSING_CLEAR < c < s1 for c in crossings):
        s_new = max(c for c in crossings if s_new - CROSSING_CLEAR < c < s1) + CROSSING_CLEAR
      e_new = s1
      if secs(s_new, s1) < (1 - LC_FIT) * want and s0 <= keys[0][0] + 1.0:
        e_new = max(s1, min(at_time(s_new, want), upper - 1.0))
      if abs(secs(s_new, e_new) - want) >= abs(secs(s0, s1) - want) or e_new - s_new < shortest - 1e-6:
        continue  # no better than it was
      first = a
      while first > 0 and keys[first - 1][0] > s_new + EPS:  # holds it now starts before
        first -= 1
      after = b + 1
      while after < len(keys) and keys[after][0] < e_new - EPS:  # holds it now ends after
        after += 1
      keys[first:after] = self._replace_chain(keys, a, b, s_new, e_new)
    if any(k2[0] < k1[0] - EPS for k1, k2 in zip(keys, keys[1:], strict=False)):
      return original  # a re-placing gone wrong: as it was
    return keys

  def off_jogs(self, route, keys: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The keys with each single lane change moved off the route's jogs (route_jogs), where the lane line already
    swerves: wholly before the jog where there's room back to the key before, else after it where there's room on to
    the key after; one with neither stays, logged as a jog_change anomaly."""
    spans = route_jogs(route)
    if not spans:
      return keys
    keys = list(keys)
    for a, b in reversed(chains(keys)):
      if b != a + 1:
        continue
      (s0, la), (s1, lb) = keys[a], keys[b]
      hit = next(((j0, j1) for j0, j1 in spans if j0 < s1 and s0 < j1), None)
      if hit is None:
        continue
      length = s1 - s0
      lower = keys[a - 1][0] if a > 0 else s0
      upper = keys[b + 1][0] if b + 1 < len(keys) else s1

      def clear(x0, x1, crossings=()):
        return not any(j0 < x1 and x0 < j1 for j0, j1 in spans) and not any(x0 - CROSSING_CLEAR < c < x1 for c in crossings)
      # before or after it; first where it crosses no stop line or junction node, as nav keeps changes off them
      options = [(hit[0] - length, hit[0]), (hit[1], hit[1] + length)]
      ok = [o for o in options if o[0] >= lower and o[1] <= upper - 1.0 and clear(*o)]
      best = [o for o in ok if clear(*o, sorted(route.stops + route.junctions))] or ok
      if best:
        keys[a], keys[b] = (best[0][0], la), (best[0][1], lb)
      else:
        self.jog_changes.append((s0, s1))
    return keys

  @staticmethod
  def _chain_lanes(keys, a: int, b: int) -> float:
    """Lanes a change crosses: its ramps' changes, without its renumbering steps."""
    return sum(keys[j + 1][1] - keys[j][1] for j in range(a, b) if keys[j + 1][0] - keys[j][0] > EPS)

  @staticmethod
  def _replace_chain(keys, a: int, b: int, s0: float, s1: float) -> list[tuple[float, float]]:
    """A change's keys (a to b) moved to run from s0 to s1, linearly across (shaped() makes it a quintic): its lanes
    moved, in the end's numbering, at the new share for each place, and its renumbering steps (two keys at one place)
    where they were, each lane before one in the numbering before it."""
    steps = [(keys[j][0], keys[j + 1][1] - keys[j][1]) for j in range(a, b)
             if keys[j + 1][0] - keys[j][0] <= EPS and abs(keys[j + 1][1] - keys[j][1]) > EPS]
    moved = sum(keys[j + 1][1] - keys[j][1] for j in range(a, b) if keys[j + 1][0] - keys[j][0] > EPS)
    end = keys[b][1]

    def lane(s: float, before: bool = False) -> float:
      """In the numbering at s (before a step there, with `before`)."""
      f = min(max((s - s0) / max(s1 - s0, EPS), 0.0), 1.0)
      later = sum(jump for pos, jump in steps if pos > s + EPS or (before and abs(pos - s) <= EPS))
      return end - moved * (1.0 - f) - later
    out = [(s0, lane(s0, before=True))]
    for pos, _ in steps:
      out += [(pos, lane(pos, before=True)), (pos, lane(pos))]
    out.append((s1, lane(s1)))
    return sorted(out, key=lambda k: k[0])

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
                         "target": s_line - self.front - self.style["stop_margin"]})
    arrive_along = route.length - float(self.c["stop_before"])
    self.s_arrive = float(np.interp(max(arrive_along, route.at + 5.0), self.route_s, self.s_path))
    # the static speed profile: the road's speed, the curvature, nav's turn speeds, the arrival
    k = np.clip(np.searchsorted(route.along, self.route_s, side="right") - 1, 0, max(len(route.points) - 2, 0))
    lim = np.array([road_speed(route, int(j)) for j in k]) * self.style["speed"]
    if self.c.get("speed"):
      lim = np.minimum(lim, float(self.c["speed"]))
    # by the curvature's envelope (its largest within CURV_SMOOTH either way): the average smooths a short S (a jog, a
    # lane change across a bend) down to less than the car must take
    n = max(int(CURV_SMOOTH / STEP), 1)
    k_abs = np.abs(smooth(k_raw, 3))
    env = k_abs.copy()
    for j in range(1, n + 1):
      env[j:] = np.maximum(env[j:], k_abs[:-j])
      env[:-j] = np.maximum(env[:-j], k_abs[j:])
    curve = np.sqrt(self.style["a_lat"] / np.maximum(np.maximum(env, np.abs(self.kappa_path)), 1e-4))
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
    self.corners = self._corners(route)
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
      out.append((s0, s1, side, round(r1 - r0, 2)))
    return out

  def _plan_anomalies(self, route, t: float):
    """Map anomalies the plan itself shows: the car's side over a kerb along it, GTA's lanes elsewhere than the map's,
    a stop line far from its junction."""
    lanes = route.lanes
    junctions = np.asarray(route.junctions, float)
    found: dict[tuple, list] = {}  # (kind, why): [(path index, details)], each run of them one anomaly
    every = 5
    for i in range(0, len(self.path), every):
      rs, rr = self.route_s[i], self.route_r[i]
      if self._unclear(rs) or self.s_path[i] < self.join_m:
        continue
      k = segment_of(route, rs)
      sec = lanes.section_at(rs, k)
      if sec is None or not sec.lanes:
        continue
      lo, hi = sec.edges
      if rr - self.half_width < lo - KERB_MARGIN or rr + self.half_width > hi + KERB_MARGIN:
        found.setdefault(("kerb_contact", ""), []).append((i, {"right": round(float(rr), 2), "edges": [round(lo, 2), round(hi, 2)]}))
      link = route.links[k] if k < len(route.links) else None
      if link is not None and link.lanes:
        lane = lane_at(self.keys, rs)
        if abs(lane - round(lane)) > 0.2:
          continue
        back = len([sp for sp in sec.spans if sp.heading == -1])
        if link.lanes != sec.lanes or link.back != back:
          found.setdefault(("gta_link_offset", "lane counts"), []).append((i, {"map": [sec.lanes, back], "gta": [link.lanes, link.back]}))
          continue
        gta_lo, gta_hi = link.inner - link.back * link.width, link.inner + link.lanes * link.width
        room = min(rr - gta_lo, gta_hi - rr) - self.half_width
        # the map's lanes wider than GTA's on the near side (a lane placed out towards GTA's barrier or kerb; a shoulder
        # beyond them is no lane)
        lanes_lo, lanes_hi = sec.spans[0].left, sec.spans[-1].right
        wider = (lanes_lo < gta_lo - EDGE_WIDER) if rr - gta_lo < gta_hi - rr else (lanes_hi > gta_hi + EDGE_WIDER)
        if room < EDGE_CLEAR and wider:
          found.setdefault(("gta_edge", ""), []).append((i, {"room": round(float(room), 2), "right": round(float(rr), 2),
                                                            "gta_edges": [round(gta_lo, 2), round(gta_hi, 2)],
                                                            "map_edges": [round(float(lo), 2), round(float(hi), 2)]}))
        gta = link.inner + (round(lane) + 0.5) * link.width
        mine = float(sec.offset(round(lane)))
        if abs(gta - mine) > GTA_OFFSET:
          found.setdefault(("gta_link_offset", "lane centre"), []).append((i, {"lane": int(round(lane)), "map": round(mine, 2), "gta": round(gta, 2)}))
    for (kind, why), hits in found.items():
      runs = [[hits[0]]]
      for h in hits[1:]:
        if h[0] - runs[-1][-1][0] <= 3 * every:
          runs[-1].append(h)
        else:
          runs.append([h])
      for run in runs:
        i, detail = run[len(run) // 2]
        extra = {"why": why} if why else {}
        self._anomaly(kind, t, self.path[i], float(self.s_path[i]), planned=True, length=round(float(self.s_path[run[-1][0]] - self.s_path[run[0][0]]) + every, 1),
                      **extra, **detail)
    for s0, s1 in self.jog_changes:
      self._anomaly("jog_change", t, route_point(route, s0), float(np.interp(s0, self.route_s, self.s_path)),
                    span=[round(s0, 1), round(s1, 1)])
    for c in self.corners:  # the plan cutting a junction's corner kerb
      hits = [i for i in range(len(self.path)) if abs(self.route_s[i] - c["along"]) < CORNER_SPAN and
              (self._corner_hit(c, self.path[i], float(self.theta[i]), self.body) or 0.0) > KERB_MARGIN]
      if hits:
        i = hits[len(hits) // 2]
        self._anomaly("kerb_contact", t, self.path[i], float(self.s_path[i]), planned=True, corner=c["side"],
                      depth=round(max(self._corner_hit(c, self.path[j], float(self.theta[j]), self.body) for j in hits), 2),
                      corner_at=[round(float(v), 1) for v in c["corner"]])
    for st in self.stops:
      after = junctions[junctions > st["along"] - 1.0] if len(junctions) else junctions
      gap = float(after[0] - st["along"]) if len(after) else math.inf
      if gap > STOP_LINE_FAR:
        self._anomaly("stop_line_far", t, route_point(route, st["along"]), st["s"], kind_line=st["kind"],
                      to_junction=None if math.isinf(gap) else round(gap, 1))

  def _corners(self, route, line: np.ndarray | None = None, line_s: np.ndarray | None = None,
               bends: list | None = None) -> list[dict]:
    """Each turn's corner kerb on its inside (the route's turns, and its corners where the lane line takes a fillet): the
    kerb lines of the road in and the road out (the map's cross-sections a leg from the turn's point, CORNER_LEG or half
    the way to the next corner, carried on straight), meeting at the corner. Near-side turns (right turns where traffic
    drives on the right) always; far-side ones where that side is a median (_kerb). Only where that's a junction's corner
    the line (the path, or `line` at line_s m along the route) clears: a turn of CORNER_ANGLE at a junction node, the
    corner near the turn's point, and the line inside both kerb lines on the legs. Turns whose kerb runs inside the
    block (the road bends there) are added to `bends`."""
    line = self.path if line is None else line
    line_s = self.route_s if line_s is None else line_s
    out = []
    lanes = route.lanes
    junctions = np.asarray(route.junctions, float)
    route.turns(0.0)
    sites = [(tr.dist, tr.side, abs(tr.angle)) for tr in getattr(route, "_turns", None) or []]
    sites += [(sc, "left" if turned > 0 else "right", abs(turned)) for sc, turned in lanes.corners
              if not any(abs(sc - s0) < CORNER_SAME for s0, _, _ in sites)]
    sites.sort()
    for n, (dist, side, angle) in enumerate(sites):
      room = min([abs(dist - s0) for k, (s0, _, _) in enumerate(sites) if k != n] + [2 * CORNER_LEG])
      leg = min(CORNER_LEG, room / 2)
      a, b = dist - leg, dist + leg
      if leg < CORNER_LEG_MIN or not CORNER_ANGLE[0] <= angle <= CORNER_ANGLE[1] or a < 10.0 or b > route.length - 10.0:
        continue
      if not len(junctions) or np.abs(junctions - dist).min() > CORNER_NODE:
        continue
      sa, sb = lanes.section_at(a), lanes.section_at(b)
      if sa is None or sb is None or not sa.lanes or not sb.lanes:
        continue
      kerb_a, kerb_b = self._kerb(sa, side), self._kerb(sb, side)
      if kerb_a is None or kerb_b is None:
        continue
      chord = min(10.0, leg)
      pa, pb = route_point(route, a), route_point(route, b)
      da, db = pa - route_point(route, a - chord), route_point(route, b + chord) - pb
      da, db = da / max(np.hypot(*da), 1e-6), db / max(np.hypot(*db), 1e-6)
      right = side == "right"
      # outward of the road, into the corner block
      na = np.array([da[1], -da[0]]) * (1 if right else -1)
      nb = np.array([db[1], -db[0]]) * (1 if right else -1)
      ka = pa + np.array([da[1], -da[0]]) * kerb_a[0]
      kb = pb + np.array([db[1], -db[0]]) * kerb_b[0]
      m = np.array([da, -db]).T
      if abs(np.linalg.det(m)) < 0.2:
        continue  # the legs near parallel: no corner to speak of
      u = np.linalg.solve(m, kb - ka)
      corner = ka + da * u[0]
      if np.hypot(*(corner - route_point(route, dist))) > CORNER_NEAR:
        continue
      if self._road_in_block(route, a, b, ka, na, kb, nb, side):
        if bends is not None:
          bends.append(float(dist))
        continue  # the map's own kerb runs inside the block: a bend, a slip or a wide corner, not a square one
      ia, ib = (int(np.argmin(np.abs(line_s - x))) for x in (a, b))
      if (line[ia] - ka) @ na > -CORNER_INSIDE or (line[ib] - kb) @ nb > -CORNER_INSIDE:
        continue  # the line outside a leg's kerb line: the line isn't the kerb
      out.append({"along": float(dist), "side": side, "ka": ka, "na": na, "kb": kb, "nb": nb, "corner": corner,
                  "radius": min(kerb_a[1], kerb_b[1]), "leg": leg})
    return out

  def _kerb(self, sec, side: str) -> tuple[float, float] | None:
    """m right of the line of a cross-section's kerb on `side`, and how rounded its corner is taken: the road's edge on
    the near side; on the far side our own carriageway's edge where it's a median's (nothing coming the other way, or
    MEDIAN_GAP between the directions), MEDIAN_RADIUS; None where the other way's lanes run beside ours."""
    near = side == ("right" if self.c["drive_on_right"] else "left")
    if near:
      return (sec.edges[1] if side == "right" else sec.edges[0]), KERB_RADIUS
    if not any(sp.heading == -1 for sp in sec.spans):
      return (sec.edges[1] if side == "right" else sec.edges[0]), MEDIAN_RADIUS
    ours = sec.ours
    if side == "left" and sec.first > 0 and ours[0].left - sec.spans[sec.first - 1].right >= MEDIAN_GAP:
      return ours[0].left, MEDIAN_RADIUS
    beyond = sec.first + sec.lanes
    if side == "right" and beyond < len(sec.spans) and sec.spans[beyond].left - ours[-1].right >= MEDIAN_GAP:
      return ours[-1].right, MEDIAN_RADIUS
    return None

  def _road_in_block(self, route, a: float, b: float, ka, na, kb, nb, side: str) -> bool:
    """Whether the map's kerb on `side` between a and b m along (the route's cross-sections there, every 2 m) reaches
    more than 1 m into the corner block the two kerb lines make."""
    lanes = route.lanes
    for s_ in np.arange(a, b, 2.0):
      k = segment_of(route, float(s_))
      sec = lanes.section_at(float(s_), k)
      if sec is None or not sec.lanes:
        continue
      kerb = self._kerb(sec, side)
      if kerb is None:
        continue
      p0, p1 = route.points[k], route.points[k + 1]
      d = (p1 - p0) / max(float(np.hypot(*(p1 - p0))), 1e-6)
      e = route_point(route, float(s_)) + np.array([d[1], -d[0]]) * kerb[0]
      if min(float((e - ka) @ na), float((e - kb) @ nb)) > 1.0:
        return True
    return False

  @staticmethod
  def _corner_hit(c: dict, pos, theta: float, body: tuple[float, float, float] = (FRONT, FRONT, HALF_WIDTH)) -> float | None:
    """How deep (m) the car (its origin at pos, heading theta radians from east; body: m to its front, rear and side)
    reaches into a corner's kerb block, the block's corner rounded by its radius; None where it doesn't."""
    f = np.array([math.cos(theta), math.sin(theta)])
    lat = np.array([-f[1], f[0]])
    r = c.get("radius", KERB_RADIUS)
    deepest = None
    for fx in (body[0], -body[1]):
      for lx in (body[2], -body[2]):
        q = np.asarray(pos, float) + f * fx + lat * lx
        d1, d2 = float((q - c["ka"]) @ c["na"]), float((q - c["kb"]) @ c["nb"])
        if d1 <= 0 or d2 <= 0:
          continue
        if d1 < r and d2 < r:
          depth = r - math.hypot(r - d1, r - d2)  # into the rounded corner
          if depth <= 0:
            continue
        else:
          depth = min(d1, d2)
        deepest = depth if deepest is None else max(deepest, depth)
    return deepest

  # *** the plan kept on its lane ***

  def _clamp(self, route, keys: list[tuple[float, float]], intent: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The intent moved back where it strays (the lane line's fillets and eased corners cut across what's beside the
    lane): our body out of each turn's inside corner kerb (_corners) by CLAMP_MARGIN; outside lane changes and the
    corners of turns of CLAMP_TURN or more (but bends), our body within its lane, CLAMP_MARGIN in from its edges, and
    near a fork within FORK_ROOM of the lane's centre on the gore's side. Each push eased in and out over CLAMP_EASE m,
    at most CLAMP_MAX. Also how far (m) the line may then move left and right in its lane (for the in-lane bias and
    wander): (intent, left, right). The pushes are kept in `clamps`, plan_clamp anomalies."""
    self.clamps = []
    along, _ = along_route(route, intent, route.at - 5.0)
    bends: list[float] = []
    self._corners(route, intent, along, bends)
    lanes_ = self._lane_room(route, keys, np.arange(along[0], along[-1] + LANE_GRID, LANE_GRID), bends)
    for n in range(4):  # a push turns the body, and moves it nearer another corner
      need, why, room_l, room_r = self._strays(route, intent, lanes_)
      if n == 3 or not np.any(need):
        break
      for a, b in _runs(np.abs(need) > 0.05):
        i = a + int(np.argmax(np.abs(need[a:b])))
        self.clamps.append({"i": i, "push": round(float(need[i]), 2), "why": why[i], "length": float(b - a) * STEP})
      th = headings(intent)
      push = eased(need, int(CLAMP_EASE / STEP))
      intent, _ = resample(intent + np.stack([np.sin(th), -np.cos(th)], axis=1) * push[:, None])
    return intent, room_l, room_r

  def _lane_room(self, route, keys, along: np.ndarray, bends: list[float]):
    """At each place along the route (along, every LANE_GRID m): our lane's centre (m right of the route's line; NaN
    where the lane is no guide: lane changes, jogs, tapers, and about turns of CLAMP_TURN or more but `bends`), how far
    left and right of it our middle may be (in by our half width and CLAMP_MARGIN; near a fork, at most FORK_ROOM
    towards its gore), and why."""
    n = len(along)
    centre, left, right, width = np.full(n, np.nan), np.zeros(n), np.zeros(n), np.full(n, np.nan)
    why = ["lane"] * n
    lanes = route.lanes
    turns = [tr.dist for tr in (getattr(route, "_turns", None) or []) if abs(tr.angle) >= CLAMP_TURN]
    turns += [sc for sc, turned in lanes.corners if abs(turned) >= CLAMP_TURN]
    turns = np.asarray([x for x in turns if not any(abs(x - y) < CORNER_SAME for y in bends)], float)
    jogs = route_jogs(route)
    forks = [(f.along, f.side) for f in route.forks if f.keep]
    ks = [k[0] for k in keys]
    room0 = self.half_width + CLAMP_MARGIN
    for i in range(n):
      rs = float(along[i])
      j = bisect.bisect_right(ks, rs) - 1
      lane = lane_at(keys[max(j, 0):j + 2], rs) if 0 <= j < len(keys) - 1 else keys[min(max(j, 0), len(keys) - 1)][1]
      if abs(lane - round(lane)) > 0.02:
        continue  # a lane change under way
      if len(turns) and np.abs(turns - rs).min() < CORNER_ZONE or any(j0 < rs < j1 for j0, j1 in jogs):
        continue
      sec = lanes.section_at(rs, segment_of(route, rs))
      if sec is None or not sec.lanes or not sec.lo <= round(lane) <= sec.hi:
        continue
      sp = sec.spans[sec.first + int(round(lane))]
      width[i] = sp.right - sp.left
      room = width[i] / 2 - room0
      if room < 0.1:
        continue  # narrower than the car: a lane closing or opening, which the lane line moves across
      centre[i], left[i], right[i] = sp.centre, room, room
      gore = next((side for fa, side in forks if -FORK_BEFORE < rs - fa < FORK_AFTER), None)
      if gore == "right" and room > FORK_ROOM:  # the branch taken is the right one: the gore on our left
        left[i], why[i] = FORK_ROOM, "fork"
      elif gore == "left" and room > FORK_ROOM:
        right[i], why[i] = FORK_ROOM, "fork"
    # where the lane's centre jumps (a jog the lane line eases across), or it runs out or in (a taper to narrower than
    # the car), it's no guide
    w = int(CORNER_ZONE / LANE_GRID)
    for j in np.flatnonzero((np.abs(np.diff(centre)) > 1.0) | (width[1:] < 2 * room0 + 0.2)):
      centre[max(j - w, 0):j + w] = np.nan
    return along, centre, left, right, why

  def _strays(self, route, line: np.ndarray, lanes_):
    """How far (m, right positive) each point of a line must move to keep our body where _clamp keeps it (lanes_: its
    _lane_room), why, and the room it has left and right in its lane after."""
    along, right = along_route(route, line, route.at - 5.0)
    grid, centre, left, right_, kind = lanes_
    g = np.clip(np.round((along - grid[0]) / LANE_GRID).astype(int), 0, len(grid) - 1)
    off = right - centre[g]
    ok = ~np.isnan(off)
    lim_l, lim_r = left[g], right_[g]
    # the moves right that keep it clear: at least lo, at most hi
    lo = np.where(ok & (off < -lim_l), np.minimum(-lim_l - np.nan_to_num(off), CLAMP_MAX), -np.inf)
    hi = np.where(ok & (off > lim_r), np.maximum(lim_r - np.nan_to_num(off), -CLAMP_MAX), np.inf)
    why = [kind[j] for j in g]
    room_l = np.where(ok, np.clip(lim_l + np.nan_to_num(off), 0.0, 0.45), 0.45)
    room_r = np.where(ok, np.clip(lim_r - np.nan_to_num(off), 0.0, 0.45), 0.45)
    th = headings(line)
    front, rear, hw = self.body
    m = CLAMP_MARGIN
    for c in self._corners(route, line, along):
      # within its legs: beyond them the road may bend away from their kerb lines
      for i in np.flatnonzero(np.abs(along - c["along"]) < c["leg"]):
        d = self._corner_hit(c, line[i], float(th[i]), (front + m, rear + m, hw + m))
        if d is not None and d > 0.05:
          d = min(d, CLAMP_MAX)
          if c["side"] == "right":
            hi[i] = min(hi[i], -d)
          else:
            lo[i] = max(lo[i], d)
          why[i] = "corner"
    both = (lo > 0) & (hi < 0)  # pushed both ways: halfway
    need = np.where(both, (np.where(both, lo, 0.0) + np.where(both, hi, 0.0)) / 2, np.where(lo > 0, lo, np.where(hi < 0, hi, 0.0)))
    need[: int(10.0 / STEP)] = 0.0  # where the car joins its line
    return need, why, room_l, room_r

  def _splice(self, route, state: dict, t: float):
    """A new route under way: the old path kept past the lane change under way and 3 s on, then the new plan's."""
    old_path, old_s, s, v = self.path, self.s_path, self.s, float(state.get("vEgo") or 0.0)
    cut = s + max(3.0 * v, 10.0)
    for s0, s1, *_ in self.changes:
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

  # *** vehicles ***

  def _vehicles(self, state: dict, t: float):
    """The vehicles around from the plugin's nearby, into the world by the pose of the state that brought them (it
    reports every 0.1 s, the state comes each frame, and its places are from the car where it was then)."""
    nb = state.get("nearby")
    if nb is None:
      self.cars = None
      return
    self.reach = (float(nb.get("ahead") or NEARBY_AHEAD), float(nb.get("side") or NEARBY_SIDE))
    rows = nb.get("v") or []
    if self.cars is not None and rows == self.cars_rows:
      return
    self.cars_rows = rows
    h = math.radians(float(state.get("heading") or 0.0))
    right, fwd = np.array([math.cos(h), math.sin(h)]), np.array([-math.sin(h), math.cos(h)])
    pos = np.asarray(state["pos"][:2], float)
    cars = []
    for v in rows:
      if len(v) < 9:
        continue
      x, y, rel, vx, vy, mnx, mxx, mny, mxy = (float(a) for a in v[:9])
      c, s = math.cos(math.radians(rel)), math.sin(math.radians(rel))
      local = np.array([(x + cx * c - cy * s, y + cx * s + cy * c) for cx, cy in ((mnx, mny), (mxx, mny), (mxx, mxy), (mnx, mxy))])
      cars.append({"c": pos + np.outer(local[:, 0], right) + np.outer(local[:, 1], fwd), "vel": right * vx + fwd * vy, "t": t})
    self.cars = cars

  def _place(self, pts: np.ndarray, ahead: float) -> tuple[np.ndarray, np.ndarray]:
    """Points' (m along the path, m right of it) on the path from the car to `ahead` m on; NaN beyond LEAD_OFF off it
    or off either end."""
    sp = self.s_path
    lo = max(int(np.searchsorted(sp, self.s - 5.0)) - 1, 0)
    hi = min(int(np.searchsorted(sp, self.s + ahead)) + 1, len(sp) - 1)
    a, ab = self.path[lo:hi], self.path[lo + 1:hi + 1] - self.path[lo:hi]
    if not len(a):
      return np.full(len(pts), np.nan), np.full(len(pts), np.nan)
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9)
    rel = pts[:, None, :] - a[None, :, :]
    tt = np.einsum("nij,ij->ni", rel, ab) / ab2
    tc = np.clip(tt, 0.0, 1.0)
    d = np.hypot(*(rel - ab[None] * tc[..., None]).transpose(2, 0, 1))
    i = np.argmin(d, axis=1)
    n = np.arange(len(pts))
    seg = np.sqrt(ab2[i])
    along = sp[lo + i] + tc[n, i] * seg
    right = (rel[n, i, 0] * ab[i, 1] - rel[n, i, 1] * ab[i, 0]) / seg
    off = (d[n, i] > LEAD_OFF) | ((i == 0) & (tt[n, i] < 0.0)) | ((i == len(a) - 1) & (tt[n, i] > 1.0))
    along[off], right[off] = np.nan, np.nan
    return along, right

  def _car_now(self, car: dict, t: float, ahead: float = 0.0) -> np.ndarray:
    """A vehicle's corners `ahead` s after now, straight on at its velocity from its reading."""
    return car["c"] + car["vel"] * (t - car["t"] + ahead)

  @staticmethod
  def _outline(c: np.ndarray) -> np.ndarray:
    """A body's corners and the middles of its sides and its middle: the points placed along the path."""
    return np.vstack([c, (c + np.roll(c, -1, axis=0)) / 2, c.mean(axis=0)[None]])

  def _lead(self, t: float) -> tuple[float, float, dict] | None:
    """The nearest vehicle in our way along the path ahead (its body within LEAD_SIDE of ours, or across the path):
    (m from our front bumper to it, its speed along the path, it); None."""
    if not self.cars:
      return None
    best = None
    far = self.s + self.front + LEAD_AHEAD
    for car in self.cars:
      pts = self._outline(self._car_now(car, t))
      along, right = self._place(pts, LEAD_AHEAD + self.front)
      ok = ~np.isnan(along)
      if not ok.any():
        continue
      near = ok & (np.abs(np.nan_to_num(right, nan=99.0)) < self.half_width + LEAD_SIDE)
      across = ok.sum() >= 2 and np.nanmin(right) < 0.0 < np.nanmax(right) and np.nanmax(along) - np.nanmin(along) < 12.0
      if not near.any() and not across:
        continue
      start = float(np.min(along[near])) if near.any() else float(np.nanmin(along))
      if start < self.s + self.front - LEAD_BESIDE or start > far:
        continue  # beside or behind us: no braking keeps clear of it
      gap = start - self.s - self.front
      j = int(np.clip(np.searchsorted(self.s_path, start), 0, len(self.s_path) - 1))
      th = self.theta[j]
      v_along = max(float(car["vel"] @ np.array([math.cos(th), math.sin(th)])), 0.0)
      if best is None or gap < best[0]:
        best = (gap, v_along, car)
    return best

  def _visible(self) -> float:
    """m of path ahead of our front bumper inside the box the plugin reports vehicles within (reach ahead of our
    origin, to either side)."""
    ahead, side = self.reach
    sp = self.s_path
    i0 = int(np.searchsorted(sp, self.s))
    i1 = int(np.searchsorted(sp, self.s + ahead + 1.0))
    if i1 <= i0:
      return 0.0
    p0 = np.array([np.interp(self.s, sp, self.path[:, 0]), np.interp(self.s, sp, self.path[:, 1])])
    th = float(np.interp(self.s, sp, self.theta))
    f, r = np.array([math.cos(th), math.sin(th)]), np.array([math.sin(th), -math.cos(th)])
    q = self.path[i0:i1] - p0
    out = ((q @ f) > ahead) | (np.abs(q @ r) > side)
    k = int(np.argmax(out)) if out.any() else len(q)
    end = sp[i0 + k - 1] if k > 0 else self.s
    return max(float(end) - self.s - self.front - RANGE_BODY, 0.0) if out.any() else max(ahead - self.front - RANGE_BODY, 0.0)

  def _crossing(self, t: float, v: float) -> tuple[float, dict] | None:
    """A moving vehicle whose way (straight on at its speed, CROSS_T s) crosses ours near a junction or turn ahead
    within CROSS_GAP s of when we'd be there: (m from our front bumper to where we'd stop for it, details); None."""
    if not self.cars:
      return None
    # s for the car to go on from here: speeding up from v at the style's accel, no faster than the profile
    sp = self.s_path
    i0 = int(np.searchsorted(sp, self.s))
    i1 = min(int(np.searchsorted(sp, self.s + LEAD_AHEAD + self.rear)) + 1, len(sp))
    ds = np.maximum(sp[i0:i1] - self.s, 0.0)
    speed = np.maximum(np.minimum(self.v_static[i0:i1], np.sqrt(v * v + 2 * max(self.style["accel"], 0.5) * ds)), 1.0)
    secs = np.concatenate(([0.0], np.cumsum(np.diff(ds) / (0.5 * (speed[1:] + speed[:-1])))))

    def when(d: float) -> float:
      return float(np.interp(d, ds, secs)) if len(ds) > 1 else math.inf
    best = None
    for car in self.cars:
      speed = float(np.hypot(*car["vel"]))
      if speed < CROSS_MOVING:
        continue
      times = np.arange(0.0, CROSS_T + 1e-6, 0.25)
      pts = self._outline(self._car_now(car, t))
      along, right = self._place(np.concatenate([pts + car["vel"] * dt for dt in times]), LEAD_AHEAD)
      m = (~np.isnan(along) & (np.abs(np.nan_to_num(right, nan=99.0)) < self.half_width + LEAD_SIDE)).reshape(len(times), -1)
      along = along.reshape(len(times), -1)
      hits = [(float(times[k]), float(np.min(along[k][m[k]])), float(np.max(along[k][m[k]]))) for k in range(len(times)) if m[k].any()]
      if not hits:
        continue
      first = min(h[1] for h in hits)
      if first < self.s + self.front - LEAD_BESIDE or not self._near_crossing(first):
        continue  # in our way already (a lead), or not where ways cross
      j = int(np.clip(np.searchsorted(self.s_path, first), 0, len(self.s_path) - 1))
      th = self.theta[j]
      angle = abs(wrap(math.degrees(math.atan2(car["vel"][1], car["vel"][0]) - th)))
      if angle < CROSS_ANGLE:
        continue  # going our way: a lead once it's in it
      t_in, t_out = min(h[0] for h in hits), max(h[0] for h in hits)
      last = max(h[2] for h in hits)
      us_in = when(first - self.s - self.front)
      us_out = when(last - self.s + self.rear)
      if us_in > t_out + CROSS_GAP or us_out < t_in - CROSS_GAP:
        continue  # it's gone before we're there, or we're through before it comes
      d = first - self.s - self.front - CROSS_STOP
      if d + CROSS_STOP - 0.5 < v * RANGE_LAG + v * v / (2 * DECEL_MAX):
        continue  # too late to stop short of it: on through
      if best is None or d < best[0]:
        best = (d, {"at": round(first, 1), "in": round(t_in, 1), "out": round(t_out, 1), "us": round(us_in, 1),
                    "angle": round(angle), "speed": round(speed, 1)})
    return best

  def _near_crossing(self, s: float) -> bool:
    """Whether a place on the path (m) is near a junction node or turn of the route."""
    along = float(np.interp(s, self.s_path, self.route_s))
    spots = list(self.route.junctions) + [tr.dist for tr in (getattr(self.route, "_turns", None) or [])]
    return any(abs(a - along) < CROSS_NEAR for a in spots)

  # *** each step ***

  def step(self, route, state: dict, t: float, collisions: int = 0) -> dict | None:
    """The plugin control message for this bridge step (t: s, a monotonic clock); None until there's a route."""
    if self.finished is not None:
      return {"type": "control", "active": True, "curvature": 0.0, "accel": HOLD_ACCEL} if self.finished == "arrived" else None
    pos = state.get("pos")
    v = float(state.get("vEgo") or 0.0)
    if pos is None:
      return self.msg
    self._body(state)
    key =(round(pos[0], 3), round(pos[1], 3), round(float(state.get("heading") or 0.0), 3), round(v, 3))
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
    self._vehicles(state, t)
    self._checks(route, state, t, collisions, v)
    if self.phase == "abort":
      return self._abort_msg(v, t)
    accel = self._longitudinal(state, v, t, dt)
    if self.phase == "abort":
      return self._abort_msg(v, t)
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
    waiting = self.phase in ("stop", "arrive", "follow", "yield") or self.stopped_at is not None
    if waiting:
      self.progress = (self.progress[0], t)
    if self.phase != "abort" and t - self.progress[1] > NO_PROGRESS_S:
      self._abort(f"no progress in {NO_PROGRESS_S:.0f} s", t, pos)
    if self.phase != "abort" and route.off < 30.0 and not route.elsewhere:
      sec = route.section(route.seg)
      near_j = self._unclear(route.at) or self.s < self.join_m
      # off the road and off the plan: on the plan, it's the plan past the map's kerbs (a kerb_contact anomaly), not
      # the driving
      if sec is not None and sec.lanes and not near_j and self.dev > OFF_ROAD_DEV and \
         not (sec.edges[0] - OFF_ROAD <= route.right <= sec.edges[1] + OFF_ROAD):
        if held("off_road", True, 0.5):
          self._abort(f"off the road ({route.right:+.1f} m right of its line)", t, pos)
      else:
        self.timers["off_road"] = None

  def _longitudinal(self, state: dict, v: float, t: float, dt: float) -> float:
    st = self.style
    s = self.s
    sp = self.s_path
    self.lead_gap = math.nan
    look = max(2.0, 1.2 * v)
    vp_here = float(np.interp(s, sp, self.v_static))
    vp = float(np.interp(s + look, sp, self.v_static))
    reason = REASONS[int(self.why_static[min(int((s + look) / STEP), len(sp) - 1)])]
    phase = "drive"
    b = st["decel"]
    stop_d = None  # m on to the stop the car is to come to next
    # the next stop sign or give way line not yet done
    for x in self.stops:
      if x["done"]:
        continue
      if x["kind"] == "lights":
        if s + self.front >= x["s"]:
          x["done"] = True
          mark = {"t": round(t, 3), "s": round(x["s"], 1), "along": round(x["along"], 1), "from": round(t - LIGHT_MARK, 3), "to": round(t + LIGHT_MARK, 3)}
          self.lights.append(mark)
          self._event("light_unknown", t, **{k: v for k, v in mark.items() if k != "t"})
        continue
      d = x["target"] - s
      if x["kind"] == "give_way":
        if s + self.front >= x["s"]:
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
      stop_d = d
      if cap < vp:
        vp, reason = cap, "stop"
        if d < 40.0:
          phase = "stop"
      break
    # the arrival
    d_end = self.s_arrive - s
    stop_d = min(stop_d, d_end) if stop_d is not None else d_end
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
    # vehicles: no faster than stops for one at the edge of what the plugin reports (but for range_share of the speed),
    # one about to cross our way, and the one in it
    detail: dict = {}
    need = math.inf  # the acceleration a vehicle holding the car up allows
    if self.cars is not None:
      vis = self._visible()
      k = DECEL_MAX * RANGE_LAG
      cap = max(-k + math.sqrt(k * k + 2 * DECEL_MAX * max(vis - RANGE_STOP, 0.0)), float(self.c["range_share"]) * vp)
      if cap < vp:
        vp, reason = cap, "range"
      cross = self._crossing(t, v)
      if cross is not None:
        self.cross_hold = (s + self.front + cross[0], t, cross[1])
      elif self.cross_hold is not None and t - self.cross_hold[1] < CROSS_KEEP and s + self.front < self.cross_hold[0]:
        cross = (self.cross_hold[0] - s - self.front, self.cross_hold[2])  # held on through a moment's doubt
      if cross is not None:
        d, info = cross
        cap = math.sqrt(2 * b * 0.8 * max(d - 0.3, 0.0))
        stop_d = min(stop_d, d)
        if cap < vp:
          vp, reason, phase, detail = cap, "cross", "yield", info
          if v > cap:  # as much as stops short of it, no harder where it was seen late
            need = -v * v / (2.0 * max(d - v * STOP_LAG, 0.15))
      lead = self._lead(t)
      if lead is not None:
        gap, vl, _ = lead
        self.lead_gap = gap
        a_lead = self._follow(gap, vl, v)
        if a_lead < min(need, (vp * vp - v * v) / (2.0 * look)):
          cap = math.sqrt(vl * vl + 2 * b * max(gap - LEAD_STOP - st["headway"] * vl, 0.0))  # what it holds us to, for the record
          vp, reason, phase, need = min(cap, vp), "lead", "follow", a_lead
          detail = {"gap": round(gap, 1), "speed": round(vl, 1)}
    held = phase in ("follow", "yield") and v < STOPPED
    self.held_since = (self.held_since if self.held_since is not None else t) if held else None
    if self.held_since is not None and t - self.held_since > WAIT_MAX:
      self._abort(f"held {WAIT_MAX:.0f} s by a vehicle in the way", t, state.get("pos"))
      return self.a
    self._set(phase, t, **detail)
    self.v_prof, self.reason = min(vp_here, vp), reason
    if v < 0.5 and vp > 0.5 and reason not in ("stop", "arrive"):
      self.reason = "start"
    a = (vp * vp - v * v) / (2.0 * look)
    if not math.isinf(need):
      a = need
    # the last metres to a stop: the deceleration that stops at it, from where the car is once the controls act
    d_eff = stop_d - 0.2 - v * STOP_LAG
    a_stop = -v * v / (2.0 * max(d_eff, 0.15))
    if a_stop < min(a, -0.2) and d_eff < 30.0:
      a = a_stop
    if vp < 0.3 and v < 1.0:
      a = min(a, -0.5) if (self.s_arrive - s) > 0 else HOLD_ACCEL
    a = float(np.clip(a, -DECEL_MAX, st["accel"]))
    return self._out_accel(a, dt, hard=a < -b)

  def _follow(self, gap: float, vl: float, v: float) -> float:
    """Our acceleration behind a vehicle gap m ahead going vl m/s along the path (the intelligent driver model's
    braking term, by the style's accel, decel and headway), no harder than the style's decel unless stopping LEAD_STOP
    short of it (as it slows to its speed) takes more: then just that."""
    st = self.style
    acc, b, hw = st["accel"], st["decel"], st["headway"]
    want = LEAD_STOP + v * hw + v * (v - vl) / (2.0 * math.sqrt(acc * b))
    a = acc * (1.0 - (max(want, 0.0) / max(gap, 0.1)) ** 2)
    if v <= vl:
      return a
    need = -(v * v - vl * vl) / (2.0 * max(gap - LEAD_STOP - v * STOP_LAG, 0.15))
    return need if need < -b else max(a, -b)

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
    # steered from the rear axle, which moves along the car's heading: the game's position is the car's middle, which in
    # a bend moves inwards of its heading (b x curvature), and steering from it cuts every corner by about b x
    # curvature x lookahead (0.7 m in a square turn)
    back = float(self.c["rear_axle"]) if self.c.get("rear_axle") is not None else float(state.get("wheelBase") or WHEEL_BASE) / 2
    hr = math.radians(h)
    pos = pos - back * np.array([-math.sin(hr), math.cos(hr)])
    # the pose when the control acts
    lat = float(self.c["latency"])
    hm = math.radians(h + math.degrees(yaw * lat) / 2)
    pred = pos + v * lat * np.array([-math.sin(hm), math.cos(hm)])
    h_pred = h + math.degrees(yaw * lat)
    s_pred, _ = project(self.path, self.s_path, pred, self.s, window=40.0)
    ld = min(max(LOOKAHEAD[0], LOOKAHEAD[1] * v + LOOKAHEAD[2]), LOOKAHEAD[3])
    sp = self.s_path
    k_ff = float(np.interp(s_pred + v * float(self.c["ff_preview"]), sp, self.kappa_path))
    k_pp = pursuit(self.path, sp, s_pred, pred, h_pred, v, ld)
    # the pursuit's own curvature from the path itself (its chord), taken out as the feedforward stands for it; the
    # heading interpolated, as a per-point one steps several degrees in a tight bend and makes the steering chatter
    ideal = np.array([np.interp(s_pred, sp, self.path[:, 0]), np.interp(s_pred, sp, self.path[:, 1])])
    k_pp0 = pursuit(self.path, sp, s_pred, ideal, game_heading(float(np.interp(s_pred, sp, self.theta))), v, ld)
    k = k_ff + (k_pp - k_pp0)
    vv = max(v, 1.0)
    cap = min(KAPPA_MAX, A_LAT_MAX / (vv * vv))
    rate = max(RATE_SHARE * self.style["j_lat"] / (vv * vv), 0.03)
    rate *= 1.0 + RATE_OFF_PATH * min(max((self.dev - 0.3) / 0.7, 0.0), 1.0)  # quicker back onto the path once off it
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
    self.lc_tau = self.lc_dir = math.nan
    side = label = None
    for s0, s1, d, _ in self.changes:
      if s0 - self.style["signal_lead"] * max(v, 3.0) <= s < s0 + LC_DONE * (s1 - s0):
        side, label = d, "laneChange" + d.capitalize()
      if s0 <= s <= s1:
        self.lc_tau = (s - s0) / max(s1 - s0, EPS)
        self.lc_dir = -1.0 if d == "left" else 1.0
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
    for c in self.corners:
      if abs(route.at - c["along"]) < CORNER_SPAN:
        depth = self._corner_hit(c, pos[:2], math.radians(float(state.get("heading") or 0.0) + 90.0), self.body)
        if depth is not None and depth > KERB_MARGIN:
          self._anomaly("kerb_contact", t, pos, self.s, planned=False, corner=c["side"], depth=round(depth, 2),
                        corner_at=[round(float(v), 1) for v in c["corner"]])
    near_j = self._unclear(route.at)
    if route.off < 30.0 and not route.elsewhere and not near_j and self.s >= self.join_m:
      sec = route.section(route.seg)
      hw = self.half_width
      if sec is not None and sec.lanes and (route.right - hw < sec.edges[0] - KERB_MARGIN or route.right + hw > sec.edges[1] + KERB_MARGIN):
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


def eased(need: np.ndarray, n: int) -> np.ndarray:
  """A push (right positive) each way reaching at least `need` wherever it's asked, eased in and out over n points
  either side (the largest within n/2 either way, then averaged over n)."""
  out = np.zeros(len(need))
  for sign in (1.0, -1.0):
    x = np.maximum(sign * need, 0.0)
    if not x.any():
      continue
    h = max(n // 2, 1)
    env = x.copy()
    for j in range(1, h + 1):
      env[j:] = np.maximum(env[j:], x[:-j])
      env[:-j] = np.maximum(env[:-j], x[j:])
    k = np.hanning(2 * h + 3)[1:-1]
    out += sign * np.convolve(np.pad(env, h, mode="edge"), k / k.sum(), mode="valid")[:len(need)]
  return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
  e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
  return list(zip(e[::2], e[1::2], strict=True))


def route_point(route, s: float) -> np.ndarray:
  return np.array([np.interp(s, route.along, route.points[:, 0]), np.interp(s, route.along, route.points[:, 1])])
