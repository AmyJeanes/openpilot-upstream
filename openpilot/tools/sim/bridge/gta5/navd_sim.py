#!/usr/bin/env python3
"""navd's planner closed loop on real routes, the driving model replaced by a stand-in: how often the car is in a lane
for each turn, fork and exit by the last place it can take it, and what the lane changes nav asks for cost.

The routes are mapdrive_sim's (made routes, and e2e trips routed by Valhalla on the live map, GTA5_MAP); the planner is
navd's own (Planner, fed by the bridge's nav_inputs from Route.info, as gta5_brain does). In between, standing in for
the game, openpilot and the driver:

- The car (Car) runs along the route in its lanes: lane keeping in a lane the road's lane maps carry it through
  (laneMaps: a lane that ends merges into the nearest), speed from the road's limit, its curvature, a turn's speed
  (Turn.speed) and its stop lines (lights cycling, stop signs), behind the traffic ahead, with the game car's
  acceleration lag. It takes each turn at the junction as n16 does with its nav input: from the lane it is in, into the
  road out's outside lane on that side (with `turn_needs_pulse`, only once nav's turn signal has pulsed the model).
- The driving model's lane change (ModelCfg): once the blinker is on with openpilot reading a lane change (NavDesire
  "laneChange", else above 20 mph) and the blind spot that way has been clear for 0.5 s, the driver's nudge (PLAN D
  stage 3: auto-approved, as in GTA) starts it; it moves across at a rate that falls with speed below `v_ok` and is
  nothing below `v_min` (the offline desire probe: n16's plan barely moves for a lane change pulse below ~10 m/s),
  taking lc_time s above v_ok. DesireHelper gives up after 10 s; a change not past half way then comes back.
- Traffic (Traffic): vehicles in the route's lanes at a density per km per lane, at the road's speed give or take,
  stopping at red lights and queueing; the blind spot (a vehicle alongside or just behind, in the lane beside) and the
  lead come from them.
- The lane nav plans from (LaneHead): the truth (the bridge's reading on the route, lane_source "map"), or a noisy
  lane head (modelV2.laneHead: [index, count, probability]) with a given strict accuracy, count accuracy, lag and
  dropouts, as lane_source "model" (no truth at all) or "fused".

Each maneuver (a turn, a fork or exit, a way straight on past lanes that only turn, a lane that ends) is judged at its
commit point (a turn's stop line or junction entry, a fork's gore, where a lane ends) by the planner's own lane logic
(lanes_for) on the truth: in a lane for it, or not (missed: the model can't make it from there, and a reroute
follows; the car carries on along the route, as after one). A lane that ends with the car still in it is a forced
merge, not a miss.

  python -m openpilot.tools.sim.bridge.gta5.navd_sim                     # the scenario trips and 40 real routes
  python -m openpilot.tools.sim.bridge.gta5.navd_sim --routes 10 --model n16 --lane truth,model:0.85 --out r.json
  python -m openpilot.tools.sim.bridge.gta5.navd_sim --nav map,mapbox --lane model:0.85   # full map vs Mapbox data

With `--nav mapbox`, nav is comma navd's step 1 lane guidance (selfdrive/navigation/lane_guidance.py) fed only what a real
car gets: the trip's Mapbox-compatible response from mapbox_server.Builder (in process; the same as the GTA server
answers), the car's progress along it, its speed, blinker and blind spots, openpilot's lane change state and the lane
head stand-in ("truth": a perfect lane head). The judge still uses the full map's lanes.

Needs the live map (GTA5_MAP) and, for trips not yet in GTA5_TRIP_ROUTES' cache, Valhalla on :8002."""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from openpilot.common.constants import CV
from openpilot.selfdrive.navd import planner as nav
from openpilot.selfdrive.navd.planner import Fork, LaneMaps, Planner, Through, Tune, Turn, Ends, aim, aim_fork, \
  junction_entry, lane_ends, lanes_for, parse_arrows, throughs
from openpilot.selfdrive.navigation import lane_guidance as lg
from openpilot.tools.sim.bridge.gta5.gta5_navd import nav_inputs
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import Section
from openpilot.tools.sim.bridge.gta5.map.router import Route

DT = 0.05  # s, the game's frames
NAV_EVERY = 2  # frames per planner step (10 Hz)
ROUTE_AHEAD, ROUTE_STEP = 3500.0, 5.0  # gta5_brain's
LANE_CHANGE_SPEED_MIN = 20 * CV.MPH_TO_MS  # desire_helper's
LANE_TURN_SPEED = 19 * CV.MPH_TO_MS
LANE_CHANGE_TIME_MAX = 10.0  # s, desire_helper's
BLINDSPOT_CLEAR_TIME = 0.5  # s
TURN_CANCEL_DEG, TURN_CANCEL_YAW = 60.0, 0.1  # the driver's stalk (gta5_driver)
PULSE_MEMORY = 6.6  # s the model keeps a desire pulse
# the car's speed: m/s by road class where the map has no limit; lateral and longitudinal comfort
CLASS_SPEED = {"motorway": 29.0, "trunk": 25.0, "motorway_link": 15.0, "trunk_link": 15.0, "primary": 17.8,
               "secondary": 15.6, "tertiary": 13.3, "residential": 11.0, "": 13.3}
A_LAT = 2.0  # m/s^2: a square turn (r ~10 m) at ~4.5 m/s, as Turn.speed has it
DECEL, ACCEL, BRAKE = 1.5, 1.5, 3.0  # m/s^2: planned slowing, pulling away, the most for a light or lead
TAU_A = 0.22  # s, the game car's acceleration lag (mapdrive_sim.LagCar)
STOP_SHORT = 1.0  # m before a stop line the car stops
STOP_SIGN_HOLD = 1.5  # s stopped at a stop sign
LIGHT_CYCLE, LIGHT_GREEN = 40.0, 20.0  # s
CAR_LEN, GAP_MIN, HEADWAY = 4.8, 3.0, 1.2  # m, m, s
BLIND_BEHIND, BLIND_AHEAD = 7.0, 2.0  # m of the lane beside, from the car's middle, the blind spot monitor covers
TURN_TAKE, TURN_TAKE_TIME = 80.0, 3.0  # m before the turn, s to move over a lane at 2 m/s or more
APPROACH = 400.0  # m before a maneuver's commit point the changes it needs are counted from
COMMIT_BEFORE = 2.0  # m before a fork's gore, or where lanes end, it's judged


@dataclass
class ModelCfg:
  """The driving model stand-in's lane change: lc_time s (uniform) at or above v_ok m/s, slower below, none at all below
  v_min. turn_needs_pulse: a turn is only taken with nav's turn signal pulsed within PULSE_MEMORY s (n16 has the nav
  vector, so it isn't by default)."""
  name: str = "n16"
  v_ok: float = 10.0
  v_min: float = 5.0
  lc_time: tuple = (4.0, 6.0)
  turn_needs_pulse: bool = False
  # a turn desire's pulse moves the car one lane towards the turn's side (where there's a lane and a gap) within
  # TURN_TAKE m of the turn, at any speed it's moving at: the probe's 2-12 m of plan movement the right way for a turn
  # pulse, even from a standstill (an assumption to check in game: it's what takes a car into a bay opening late)
  turn_takes_lane: bool = True

  def rate(self, v: float, base: float) -> float:
    """Lane change progress per s at speed v, for a change taking `base` s above v_ok: falling off steeply (squared)
    below it, as the probe found n16's plan barely moving (under 2 m in 6 s) for a lane change pulse below ~10 m/s."""
    f = min(max((v - self.v_min) / max(self.v_ok - self.v_min, 1e-6), 0.0), 1.0)
    return f * f / base


PRESETS = {"n16": ModelCfg("n16", 10.0, 5.0), "finetuned": ModelCfg("finetuned", 4.0, 2.0)}


@dataclass
class HeadCfg:
  """The lane nav plans from. source "truth": the bridge's reading (lane_source "map"); "model": the lane head alone, no
  truth passed to nav at all; "fused": both (planner's fused). The head: right `acc` of the time (strict), else one
  lane off; its count right `count_acc` of the time, else one off; `lag` s behind; out (None) `drop` of the time; each
  error lasting `dwell` s on average (count errors 3x as long, as a kerb lane misread holds along a road)."""
  source: str = "truth"
  acc: float = 0.85
  count_acc: float = 0.95
  lag: float = 0.5
  drop: float = 0.05
  dwell: float = 1.0

  @property
  def label(self) -> str:
    return "truth" if self.source == "truth" else f"{self.source}:{self.acc:g}"


@dataclass
class TrafficCfg:
  density: float = 10.0  # vehicles per km per lane
  speed: tuple = (0.8, 1.05)  # of the road's speed


class Markov:
  """A two-state process (on a share `p_on` of the time, each spell `dwell` s on average), stepped every dt."""
  def __init__(self, p_on: float, dwell: float, rng: np.random.Generator):
    self.p_on, self.dwell, self.rng = p_on, dwell, rng
    self.on = bool(rng.random() < p_on)

  def step(self, dt: float) -> bool:
    if self.p_on <= 0.0:
      self.on = False
    elif self.p_on >= 1.0:
      self.on = True
    else:
      # mean spell lengths: on `dwell`, off dwell * (1 - p) / p
      leave = dt / self.dwell if self.on else dt * self.p_on / (self.dwell * (1.0 - self.p_on))
      if self.rng.random() < leave:
        self.on = not self.on
    return self.on


class LaneHead:
  """A noisy modelV2.laneHead from the true lane: [index from the left of ours, count, probability]."""
  def __init__(self, cfg: HeadCfg, rng: np.random.Generator):
    self.cfg, self.rng = cfg, rng
    self.err = Markov(1.0 - cfg.acc, cfg.dwell, rng)
    self.err_side = 1
    self.count_err = Markov(1.0 - cfg.count_acc, 3 * cfg.dwell, rng)
    self.count_side = 1
    self.count_left = False
    self.out = Markov(cfg.drop, cfg.dwell, rng)
    self.history: list[tuple[float, list | None]] = []

  def step(self, t: float, truth: list | None, dt: float) -> list | None:
    self.history.append((t, truth))
    while len(self.history) > 1 and self.history[1][0] <= t - self.cfg.lag:
      self.history.pop(0)
    seen = self.history[0][1]
    was_err, was_count = self.err.on, self.count_err.on
    err, cerr, out = self.err.step(dt), self.count_err.step(dt), self.out.step(dt)
    if err and not was_err:
      self.err_side = 1 if self.rng.random() < 0.5 else -1
    if cerr and not was_count:
      self.count_side = 1 if self.rng.random() < 0.5 else -1
      self.count_left = bool(self.rng.random() < 0.5)
    if out or seen is None or seen[0] < 0:
      return None
    i, n = int(seen[0]), int(seen[1])
    if cerr:
      # a lane counted on top (a wide kerb lane read as two, a hatched area) or one missed (parked cars in the kerb
      # lane, a turn lane not yet painted), on the left or the right; the car's own lane is seen
      n_seen = max(1, n + self.count_side)
      if self.count_left and (self.count_side > 0 or i > 0):
        i += self.count_side
      i = min(max(i, 0), n_seen - 1)
      n = n_seen
    if err:
      j = i + self.err_side
      i = j if 0 <= j < n else i - self.err_side if 0 <= i - self.err_side < n else i
    prob = self.rng.uniform(0.55, 0.9) if err else self.rng.uniform(0.75, 0.98)
    return [i, n, round(float(prob), 3)]


# *** the route's lanes, the car's and the traffic's ***

def heading_of(d: np.ndarray) -> float:
  """Game heading (deg counterclockwise from north) of a direction."""
  return math.degrees(math.atan2(-d[0], d[1]))


class Road:
  """What the stand-ins read of a route, along it (absolute m): its points on a 1 m grid, the speed the model drives at
  (limits, curvature, turns), its lane maps, stop lines and turns."""
  def __init__(self, route: Route, model: ModelCfg):
    self.route = route
    self.s = np.arange(0.0, route.length + 1e-6, 1.0)
    self.xy = np.stack([np.interp(self.s, route.along, route.points[:, 0]), np.interp(self.s, route.along, route.points[:, 1])], axis=1)
    d = np.gradient(self.xy, axis=0)
    self.heading = np.degrees(np.unwrap(np.arctan2(-d[:, 0], d[:, 1])))
    k = np.gradient(np.radians(self.heading))
    w = 9
    k = np.convolve(k, np.ones(w) / w, mode="same")
    self.curv = k
    seg = np.clip(np.searchsorted(route.along, self.s, side="right") - 1, 0, max(len(route.points) - 2, 0))
    limit = np.array([route.limit_list[i] if i < len(route.limit_list) else 0.0 for i in seg])
    cls = [route.classes[i] if i < len(route.classes) else "" for i in seg]
    default = np.array([CLASS_SPEED.get(c, CLASS_SPEED[""]) for c in cls])
    v = np.where(limit > 0, limit, default)
    v = np.minimum(v, np.sqrt(A_LAT / np.maximum(np.abs(k), 1e-4)))
    route.at, route.seg = 0.0, 0
    self.turns = [Turn(float(d), side, float(eh), float(a)) for d, side, eh, a in route.turns(1e7)]
    for t in self.turns:
      near = (self.s > t.dist - 6.0) & (self.s < t.dist + 10.0)
      v[near] = np.minimum(v[near], t.speed())
    for i in range(len(v) - 2, -1, -1):  # slowing ahead of time
      v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * DECEL * 1.0))
    self.v = v
    self.maps = list(route.lane_maps_along())
    self.stops = list(zip(route.stops, route.stop_kinds, strict=True))

  def at(self, s: float) -> int:
    return int(min(max(round(s), 0), len(self.s) - 1))

  def section(self, s: float) -> Section | None:
    lanes = self.route.lanes
    if lanes is None:
      return None
    k = int(min(max(np.searchsorted(self.route.along, s, side="right") - 1, 0), len(lanes.sections) - 1))
    return lanes.section_at(s, k)

  def count(self, s: float) -> int:
    sec = self.section(s)
    return sec.lanes if sec is not None and sec.lanes else 0

  def pose(self, s: float, right: float) -> tuple[float, float, float]:
    i = self.at(s)
    j = min(i + 1, len(self.s) - 1)
    i0 = max(j - 1, 0)
    d = self.xy[j] - self.xy[i0]
    n = d / max(np.hypot(*d), 1e-9)
    p = self.xy[i] + np.array([n[1], -n[0]]) * right
    return float(p[0]), float(p[1]), float(self.heading[i])


def follow(v_lead: float, gap: float) -> float:
  """The fastest a vehicle may go `gap` m (bumper to bumper) behind one at v_lead and still stop GAP_MIN short of it
  braking at BRAKE after HEADWAY / 2 (Gipps)."""
  b, tr = BRAKE, HEADWAY / 2
  return max(0.0, -b * tr + math.sqrt((b * tr) ** 2 + v_lead ** 2 + 2 * b * max(gap - GAP_MIN, 0.0)))


def carry_lane(lane: float, mapping, n_after: int) -> float:
  """A lane (float, during a change) through a lane map: the whole lane it's in carried on, the fraction kept."""
  base = int(math.floor(lane + 0.5))
  frac = lane - base
  new = nav.carry(mapping, base)
  return float(min(max(new + frac, 0.0), max(n_after - 1, 0)))


@dataclass
class Vehicle:
  s: float
  lane: float
  v: float
  want: float  # share of the road's speed
  stopped_t: float = 0.0
  passed_stop: float = -1.0  # the stop line (m along) it last stopped at


class Traffic:
  """Other vehicles in the route's lanes, about the car (WINDOW): `density` per km per lane, kept up as the car moves on
  (a vehicle slower than the car comes into view ahead, a faster one from behind), following the road's speed, the
  lights and each other, kept to their lanes."""
  WINDOW = (-150.0, 450.0)

  def __init__(self, road: Road, cfg: TrafficCfg, rng: np.random.Generator, s0: float, lights: 'Lights'):
    self.road, self.cfg, self.rng, self.lights = road, cfg, rng, lights
    self.vs: list[Vehicle] = []
    lo, hi = max(s0 + self.WINDOW[0], 0.0), min(s0 + self.WINDOW[1], road.s[-1])
    if cfg.density > 0:
      for _ in range(int(rng.poisson(cfg.density * (hi - lo) / 1000.0 * self._lanes(lo, hi)))):
        s = float(rng.uniform(lo, hi))
        n = road.count(s)
        if n:
          self._add(s, int(rng.integers(n)))

  def _lanes(self, lo: float, hi: float) -> float:
    return float(np.mean([max(self.road.count(s), 0) for s in np.linspace(lo, hi, 7)]))

  def _add(self, s: float, lane: int, want: float | None = None, car: 'NavSim | None' = None) -> bool:
    if s < 0 or s > self.road.s[-1] - 5.0:
      return False
    if any(abs(o.s - s) < CAR_LEN + GAP_MIN + 3.0 and abs(o.lane - lane) < 0.5 for o in self.vs):
      return False
    if car is not None and abs(car.s - s) < 40.0:
      return False
    want = float(self.rng.uniform(*self.cfg.speed)) if want is None else want
    self.vs.append(Vehicle(s, float(lane), want * self.road.v[self.road.at(s)], want))
    return True

  def step(self, t: float, dt: float, car: 'NavSim', turn_reset: tuple | None):
    road = self.road
    car_s = car.s
    lo, hi = car_s + self.WINDOW[0], car_s + self.WINDOW[1]
    if self.cfg.density > 0:
      # new vehicles where the window's count runs short: ahead for one slower than the car, behind for one faster
      lo_c, hi_c = max(lo, 0.0), min(hi, road.s[-1])
      want_n = self.cfg.density * max(hi_c - lo_c, 0.0) / 1000.0 * self._lanes(lo_c, hi_c) if hi_c > lo_c else 0.0
      if len(self.vs) < want_n and self.rng.random() < dt * 2.0:
        speed = float(self.rng.uniform(*self.cfg.speed))
        faster = speed * road.v[road.at(car_s)] > car.v + 0.5
        s = lo_c + 5.0 if faster else hi_c - 5.0
        n = road.count(s)
        if n and not (faster and lo_c <= 0.0):
          self._add(s, int(self.rng.integers(n)), speed, car)
    keep = []
    order = sorted(self.vs, key=lambda o: -o.s)
    for k, o in enumerate(order):
      want = o.want * road.v[road.at(o.s)]
      lead = next((p for p in order[:k] if abs(p.lane - o.lane) < 0.5), None)
      if abs(car.lane - o.lane) < 0.5 and car.s > o.s and (lead is None or car.s < lead.s):
        lead = car  # the car, in its lane ahead
      if lead is not None:
        want = min(want, follow(lead.v, lead.s - o.s - CAR_LEN))
      stop = self.lights.stop_for(o.s, t, o)
      if stop is not None:
        want = min(want, math.sqrt(max(0.0, 2 * BRAKE * (stop - o.s - STOP_SHORT))))
      o.v = max(0.0, min(want, o.v + ACCEL * dt) if want > o.v else max(want, o.v - 2 * BRAKE * dt))
      s0, o.s = o.s, o.s + o.v * dt
      lane0 = o.lane
      for (sm, mp, n_after) in road.maps:
        if s0 < sm <= o.s and len(mp):
          i = int(round(o.lane))
          if not 0 <= i < len(mp) or mp[i] is None and self.rng.random() < 0.5:
            o.lane = -1  # off the route's road: a lane ending at a split, half of those that end merging
          else:
            o.lane = carry_lane(o.lane, mp, n_after)
      if turn_reset is not None and s0 < turn_reset[0] <= o.s:
        # through the route's turn: those in its lanes turn with the car, the rest go their own way
        o.lane = turn_reset[1](o.lane) if turn_reset[2](o.lane) else -1
      n = road.count(o.s)
      if n and o.lane >= n - 0.5:
        o.lane = float(n - 1)
      if o.lane != -1 and abs(o.lane - car.lane) < 0.8 and abs(o.s - car.s) < CAR_LEN + GAP_MIN + 2.0 and abs(o.lane - lane0) > 0.25:
        o.lane = -1  # merged across by the car: it waits its turn instead, out of the sim
      if o.lane >= 0 and lo - 50 < o.s < hi + 50 and o.s < road.s[-1] - 2.0:
        keep.append(o)
    self.vs = keep

  def beside(self, s: float, lane: int, side: int) -> bool:
    """Whether a vehicle is in the blind spot on that side of a car in `lane` at s."""
    return any(abs(o.lane - (lane + side)) < 0.5 and -BLIND_BEHIND <= o.s - s <= BLIND_AHEAD for o in self.vs)

  def lead(self, s: float, lane: float) -> Vehicle | None:
    ahead = [o for o in self.vs if abs(o.lane - lane) < 0.75 and o.s > s]
    return min(ahead, key=lambda o: o.s) if ahead else None

  def in_gap(self, s: float, v: float, lane: int) -> bool:
    """Whether a vehicle in `lane` is too near for the car to move in beside it at s: within a car length and the gap
    ahead, or behind closing faster than it could brake for."""
    for o in self.vs:
      if abs(o.lane - lane) >= 0.5:
        continue
      ds = o.s - s
      if -CAR_LEN - GAP_MIN <= ds <= CAR_LEN + GAP_MIN or (ds < 0 and follow(v, -ds - CAR_LEN) < o.v):
        return True
    return False

  def touching(self, s: float, lane: float) -> set[int]:
    return {id(o) for o in self.vs if abs(o.lane - lane) < 0.8 and abs(o.s - s) < CAR_LEN}

  def yield_to(self, s: float, lane: float):
    """The car merged into a lane (its own ended): a vehicle that close there lets it in, out of the sim."""
    self.vs = [o for o in self.vs if not (abs(o.lane - lane) < 0.8 and abs(o.s - s) < CAR_LEN + GAP_MIN + 2.0)]


class Lights:
  """The route's stop lines: lights cycling LIGHT_CYCLE s (LIGHT_GREEN of it green) from a random phase each, stop signs
  each vehicle stops at for a moment, give way lines it slows for."""
  def __init__(self, road: Road, rng: np.random.Generator):
    self.lines = [(float(s), k, float(rng.uniform(0, LIGHT_CYCLE))) for s, k in road.stops]

  def red(self, phase: float, t: float) -> bool:
    return (t + phase) % LIGHT_CYCLE >= LIGHT_GREEN

  def stop_for(self, s: float, t: float, who) -> float | None:
    """The stop line (m along) the vehicle at s must stop at, if any: the next red light within braking reach, or the
    next stop sign it hasn't stopped at yet."""
    for line, kind, phase in self.lines:
      if line < s - 0.5:
        continue
      if line - s > 120.0:
        return None
      if kind == "lights" and self.red(phase, t):
        # a light turning red as the car is nearly at it: on through
        if line - s < who.v ** 2 / (2 * BRAKE) - 2.0:
          continue
        return line
      if kind == "stop" and who.passed_stop != line:
        if line - s < STOP_SHORT + 0.5 and who.v < 0.3:
          who.stopped_t += DT
          if who.stopped_t >= STOP_SIGN_HOLD:
            who.passed_stop, who.stopped_t = line, 0.0
            continue
        return line
    return None


# *** the maneuvers, judged ***

@dataclass
class Maneuver:
  kind: str  # "turn", "fork", "through", "drop" (a lane that ends), "split" (lanes leaving at a fork)
  s: float  # m along to it
  commit: float  # m along to its commit point
  args: tuple  # to remake its planner object (Turn / Fork / Through / Ends) seen from the car
  xy: tuple = (0.0, 0.0)
  # judged at the commit point
  judged: bool = False
  lanes: tuple | None = None  # (lo, hi) of n, from the left, as nav numbers them there
  lane: list | None = None  # the car's [i, n]
  made: bool | None = None
  relevant: bool = False  # not every lane goes the route's way
  need: int | None = None  # the most lane changes it needed over its approach (APPROACH m before its commit point on)
  v: float = 0.0
  pulsed: bool = False  # nav's turn signal pulsed the model before it
  opens: float | None = None  # m before its commit point its nearest target lane begins (a bay, a lane added), None: further
  mapbox: str | None = None  # with --nav mapbox: "lanes" where Mapbox's data gives its lanes, "none" where it has none

  def move(self, dist: float):
    if self.kind == "turn":
      return Turn(dist, *self.args)
    if self.kind == "fork":
      return Fork(dist, *self.args)
    if self.kind == "through":
      return Through(dist, *self.args)
    return Ends(dist, *self.args)


def maneuvers(route: Route, road: Road) -> list[Maneuver]:
  """The route's maneuvers nav plans lanes for (as lane_plan finds them), from its start."""
  route.at, route.seg = 0.0, 0
  info = route.info(1e7)
  arrows = parse_arrows(info["laneArrows"])
  pts = route.ahead(1e7, ROUTE_STEP)
  turns = [Turn(float(d), side, float(eh), float(a)) for d, side, eh, a in info["turns"] if d > nav.MIN_AHEAD_MAP]
  forks = [Fork(float(d), side, *rest) for d, side, *rest in info["forks"] if d > 0]
  moves = turns + forks
  maps = LaneMaps(info["laneMaps"])
  for m in moves:
    aim(m, arrows)
    aim_fork(m, maps.maps)
  thr = throughs(pts, arrows, moves, None)
  ends = lane_ends(maps.maps, moves)
  out = []
  for t in turns:
    entry, _ = junction_entry(t.dist, route.stops, route.junctions)
    out.append(Maneuver("turn", t.dist, max(min(entry, t.dist - 5.0), t.dist - 50.0), (t.side, t.exit_heading, t.angle)))
  # the others just before the lanes change there (the lane maps at it carry the car's lane on)
  for f in forks:
    out.append(Maneuver("fork", f.dist, f.dist - COMMIT_BEFORE, (f.side, f.ours, f.lanes_in, f.keep, f.other, f.slip)))
  for th in thr:
    out.append(Maneuver("through", th.dist, th.dist - COMMIT_BEFORE, (th.targets,)))
  for e in ends:
    mapping = next((mp for d, mp, _ in maps.maps if abs(d - e.dist) < 1e-6), None)
    if mapping is None:
      continue
    split = any(abs(f.dist - e.dist) < nav.MERGE_FORK_NEAR for f in forks)
    out.append(Maneuver("split" if split else "drop", e.dist, e.dist - COMMIT_BEFORE, (mapping,)))
  for m in out:
    m.xy = tuple(round(float(c), 1) for c in road.xy[road.at(m.s)])
  return sorted(out, key=lambda m: m.commit)


def judge_lanes(m: Maneuver, s: float, info: dict, lane: list) -> tuple[int, int] | None:
  """The lanes (lo, hi of lane[1]) nav would have the car in for the maneuver, seen from s, by the planner's own logic."""
  if lane is None or lane[0] < 0:
    return None
  mv = m.move(m.s - s)
  maps = LaneMaps(info["laneMaps"]) if info.get("laneMaps") is not None else None
  if isinstance(mv, (Turn, Fork)):  # as the planner aims them; the others carry their own lanes
    aim(mv, parse_arrows(info.get("laneArrows")))
    aim_fork(mv, maps.maps if maps is not None else None)
  return lanes_for(mv, (int(lane[0]), int(lane[1])), maps)


# *** the trip ***

@dataclass
class Change:
  """A lane change nav asked for."""
  t: float
  s: float
  side: str
  v: float
  maneuver: int | None  # index of the maneuver ahead it's for (the next relevant one not yet judged)
  start_t: float | None = None
  start_v: float | None = None
  end_t: float | None = None
  outcome: str = ""  # "done", "gave_up" (blinker off before it started), "failed" (started, didn't get across)

  @property
  def hanging(self) -> float:
    end = self.start_t if self.start_t is not None else self.end_t
    return max(0.0, (end if end is not None else self.t) - self.t)


@dataclass
class TripResult:
  name: str
  model: str
  head: str
  length: float
  driven: float = 0.0
  secs: float = 0.0
  arrived: bool = False
  maneuvers: list = field(default_factory=list)
  changes: list = field(default_factory=list)
  collisions: int = 0
  skipped: int = 0  # turns and forks nav left to the route
  requests: list = field(default_factory=list)  # nav's requests to the driver, as logged once over (navRequest events)
  wall_s: float = 0.0
  error: str | None = None


class NavSim:
  def __init__(self, route: Route, lane: float, model: ModelCfg, head: HeadCfg, traffic: TrafficCfg | None = None,
               tune: dict | None = None, seed: int = 0, name: str = "", s0: float = 0.0, stage: int = 3, nav: str = "map"):
    self.route, self.model, self.head_cfg = route, model, head
    self.stage = stage  # PLAN D's: the simulated driver approves at any stage, as GTA's
    self.rng = np.random.default_rng(seed)
    self.road = Road(route, model)
    self.lights = Lights(self.road, self.rng)
    self.moves = maneuvers(route, self.road)
    self.s = float(s0)
    n0 = max(self.road.count(self.s + 0.5), 1)
    self.lane = float(min(max(lane, 0.0), n0 - 1))
    self.v = float(self.road.v[self.road.at(self.s)]) * 0.8
    self.a = 0.0
    self.t = 0.0
    self.traffic = Traffic(self.road, traffic or TrafficCfg(), self.rng, self.s, self.lights)
    self.traffic.vs = [o for o in self.traffic.vs if not (abs(o.s - self.s) < CAR_LEN + 2 and abs(o.lane - self.lane) < 0.5)]
    self.head = LaneHead(head, np.random.default_rng(seed + 7919))  # its own: the same traffic whatever the lane source
    t = Tune("")
    # the planner told the speed this model needs for a lane change, as the bridge would per model (unless the tune says)
    t.values.update({"lane_source": {"truth": "map", "model": "model", "fused": "fused"}[head.source],
                     "model_lc_speed": model.v_ok, **(tune or {})})
    self.mapbox: lg.MapboxRoute | None = None
    if nav == "mapbox":
      # comma navd's lane guidance on the trip's Mapbox-compatible response; the model's lane change speed as above
      from openpilot.tools.sim.bridge.gta5.mapbox_server import Builder
      from openpilot.tools.sim.bridge.gta5.mapdrive_sim import live_map
      m = live_map()
      gt = lg.Tune.of({"lc_min_speed": model.v_ok, **(tune or {})})
      self.mapbox = lg.MapboxRoute(Builder(m["osm"] if m else None).directions(route)["routes"][0], gt)
      self.mb_scale = self.mapbox.length / max(route.length, 1e-6)  # Mapbox's metres per the route's
      self.nav = lg.LaneGuidance(gt)
      self.nav.set_route(self.mapbox)
      for mv in self.moves:
        near = [x for x in self.mapbox.moves if abs(x.along - mv.s * self.mb_scale) < 40.0]
        mv.mapbox = "lanes" if any(x.lanes for x in near) else "none"
    else:
      self.nav = Planner(t)
      self.nav.was_engaged = True  # no tune line per trip
    # openpilot and the driver
    self.indicator: str | None = None
    self.lc_state = "off"  # desire_helper's: "off", "pre", "starting"
    self.lc_dir: str | None = None
    self.lc_timer = 0.0
    self.lc_from = 0.0  # lane the change started from
    self.lc_p = 0.0  # progress across
    self.lc_base = 5.0
    self.prev_one_blinker = False
    self.clear = {"left": BLINDSPOT_CLEAR_TIME, "right": BLINDSPOT_CLEAR_TIME}
    self.pulse_t = -1e9
    self.pulse_side: str | None = None
    self.ind_heading = 0.0
    self.map_k = sum(sm <= self.s for sm, _, _ in self.road.maps)  # the next lane map
    self.turn_moves = [m for m in self.moves if m.kind == "turn"]
    self.turn_k = sum(m.s + 10.0 <= self.s for m in self.turn_moves)  # the next turn
    self.moves = [m for m in self.moves if m.commit > self.s + 1.0]
    self.judge_k = 0
    self.nav_step = False
    self.stopped = Vehicle(0.0, 0.0, 0.0, 1.0)  # the car's own stop-sign state, as a Vehicle's
    self.changes: list[Change] = []
    self.open_change: Change | None = None
    self.result = TripResult(name, model.name, head.label, route.length)
    self.prev_heading = float(self.road.heading[self.road.at(self.s)])
    self.yaw = 0.0
    self.touching: set[int] = set()  # the vehicles the car is in contact with
    self.take: list | None = None  # a turn desire moving the car over: [from lane, to lane, progress]
    self.took = -1.0  # the pulse that last did

  # the car's lane on the road, as the bridge reads it
  def _truth(self) -> tuple[list | None, float | None, float]:
    sec = self.road.section(self.s)
    right = sec.offset(self.lane) if sec is not None and sec.lanes else 0.0
    r = self.route
    r.at = self.s
    r.seg = int(min(max(np.searchsorted(r.along, self.s, side="right") - 1, 0), len(r.points) - 2))
    r.right, r.off, r.misaligned, r.elsewhere = right, 0.0, 0.0, False
    return r.lane(), r.lane_frac(), right

  def _turn_lane(self, m: Maneuver) -> callable:
    """Where a lane before a turn comes out on the road after it: its side's outside lanes, in order."""
    n_out = max(self.road.count(m.s + nav.TURN_HOLDS), 1)
    lo, hi = (m.lanes[0], m.lanes[1]) if m.lanes is not None else (0, 0)
    side = m.args[0]

    def out(lane: float) -> float:
      i = int(round(lane))
      if side == "left":
        return float(min(max(i - lo, 0), n_out - 1))
      return float(min(max(n_out - 1 - (hi - i), 0), n_out - 1))
    return out

  def step_nav(self, truth: list | None, frac: float | None):
    r = self.route
    x, y, h = self.road.pose(self.s, r.right)
    head = self.head.step(self.t, truth, DT * NAV_EVERY)
    use_truth = self.head_cfg.source != "model"
    info = r.info(ROUTE_AHEAD)
    state = {**info, "route": r.ahead(ROUTE_AHEAD, ROUTE_STEP).round(1).tolist(), "vEgo": self.v, "yawRate": self.yaw,
             "pos": [x, y, 0.0], "heading": h, "lane": truth if use_truth else None, "laneFrac": frac if use_truth else None,
             "twoWay": r.two_way(), "modelLane": head if self.head_cfg.source != "truth" else None,
             "blindspot": (self.traffic.beside(self.s, int(round(self.lane)), -1), self.traffic.beside(self.s, int(round(self.lane)), 1)),
             "waypoint": [float(r.points[-1][0]), float(r.points[-1][1])]}
    desire = {self.pulse_side: 0.9} if self.pulse_side and self.t - self.pulse_t < PULSE_MEMORY else {}
    skipped = len(self.nav.skipped)
    if self.mapbox is not None:
      lane = head if self.head_cfg.source != "truth" else ([truth[0], truth[1], 1.0] if truth and truth[0] >= 0 else None)
      bs = state["blindspot"]
      out = self.nav.update(lg.GuidanceInputs(self.t, self.s * self.mb_scale, self.v, lane, self.indicator,
                                              self.lc_state == "starting", bs[0], bs[1], self.stage))
      out.arrived = False
    else:
      out = self.nav.update(nav_inputs(state, True, self.indicator, desire, self.t, True, self.stage))
    self.result.skipped += max(len(self.nav.skipped) - skipped, 0)
    for req in out.requests:
      if req == "cancelSignal":
        self._indicator(None)
      else:
        side = "left" if req.endswith("Left") else "right"
        self._indicator(side)
        if req.startswith("laneChange"):
          self._new_change(side)
    return out, info

  def _new_change(self, side: str):
    nxt = next((k for k in range(self.judge_k, len(self.moves)) if self.moves[k].commit > self.s), None)
    c = Change(self.t, self.s, side, self.v, nxt)
    self._close_change("gave_up")
    self.changes.append(c)
    self.open_change = c

  def _close_change(self, outcome: str):
    c = self.open_change
    if c is not None and not c.outcome:
      c.outcome, c.end_t = outcome, self.t
    self.open_change = None

  def _indicator(self, side: str | None):
    if side != self.indicator:
      self.ind_heading = float(self.road.heading[self.road.at(self.s)])
    if side is None and self.open_change is not None and self.lc_state != "starting":
      self._close_change("gave_up" if self.open_change.start_t is None else self.open_change.outcome or "failed")
    self.indicator = side

  def step_openpilot(self):
    """desire_helper's lane change states, the driver's nudge (stage 3: at once, the blind spot clear) and the turn
    desire's pulse."""
    nav_desire = self.nav.desire.partition("+")[0]
    one = self.indicator is not None
    side = self.indicator
    # desire_helper's nav_lane_change: "laneChange", or "laneChange:<side>" with the blinker that side
    nav_lc = nav_desire == "laneChange" or (one and nav_desire == f"laneChange:{side}")
    lane = int(round(self.lane))
    for sd, off in (("left", -1), ("right", 1)):
      self.clear[sd] = 0.0 if self.traffic.beside(self.s, lane, off) else self.clear[sd] + DT
    below = self.v < LANE_CHANGE_SPEED_MIN and not nav_lc
    rising = one and not self.prev_one_blinker
    if self.lc_state == "off" and rising and not below:
      self.lc_state, self.lc_timer, self.lc_dir = "pre", 0.0, side
    elif self.lc_state == "pre":
      self.lc_dir = side
      if not one or below:
        self.lc_state = "off"
      elif not self.nav.signaling and self.clear[side] >= BLINDSPOT_CLEAR_TIME - 1e-6:
        self.lc_state, self.lc_timer = "starting", 0.0
        self.lc_from, self.lc_p = self.lane, 0.0
        self.lc_base = float(self.rng.uniform(*self.model.lc_time))
        if self.open_change is not None and self.open_change.start_t is None and self.open_change.side == side:
          self.open_change.start_t, self.open_change.start_v = self.t, self.v
    elif self.lc_state == "starting":
      self.lc_timer += DT
      d = -1.0 if self.lc_dir == "left" else 1.0
      n = max(self.road.count(self.s), 1)
      target = self.lc_from + d
      if not 0 <= round(target) <= n - 1:
        self.lc_p = 0.0  # no lane that way: the model stays
      elif self.lc_p < 0.5 and self.traffic.in_gap(self.s, self.v, round(target)):
        pass  # the model waits short of the line for the gap in the lane it's moving into
      else:
        self.lc_p = min(1.0, self.lc_p + self.model.rate(self.v, self.lc_base) * DT)
      self.lane = self.lc_from + d * self.lc_p
      if self.lc_p >= 1.0:
        self.lane = float(round(target))
        self.lc_state = "off"
        if self.open_change is not None and self.open_change.side == self.lc_dir:
          self._close_change("done")
        if not self.nav.signaling:
          self._indicator(None)  # the driver's stalk, once the change is done
      elif self.lc_timer > LANE_CHANGE_TIME_MAX or not one:
        # given up (desire_helper's timeout, or the blinker off): back unless past half way
        self.lane = float(round(target)) if self.lc_p >= 0.5 else float(round(self.lc_from))
        self.lc_state = "off"
        if self.open_change is not None and self.open_change.side == self.lc_dir:
          self._close_change("done" if self.lc_p >= 0.5 else "failed")
    self.prev_one_blinker = one
    # the turn desire's pulse: a blinker below the lane change speed, NavDesire not a lane change
    if rising and self.v < LANE_TURN_SPEED and not nav_lc:
      self.pulse_t, self.pulse_side = self.t, side
    self._turn_take()
    # the driver cancels a turn's signal once the car has come round
    if one and self.nav.signaling:
      turned = abs((self.road.heading[self.road.at(self.s)] - self.ind_heading + 180) % 360 - 180)
      if turned > TURN_CANCEL_DEG and abs(self.yaw) < TURN_CANCEL_YAW:
        self._indicator(None)

  def _turn_take(self):
    """turn_takes_lane: a pulsed turn desire moving the car one lane towards its side, once per pulse."""
    if not self.model.turn_takes_lane or self.lc_state == "starting":
      self.take = None
      return
    live = self.pulse_side is not None and self.t - self.pulse_t < PULSE_MEMORY and self.nav.signaling
    nxt = self.turn_moves[self.turn_k] if self.turn_k < len(self.turn_moves) else None
    if self.take is None:
      if not live or self.took == self.pulse_t or nxt is None or nxt.args[0] != self.pulse_side or nxt.s - self.s > TURN_TAKE:
        return
      d = -1 if self.pulse_side == "left" else 1
      here = int(round(self.lane))
      if not 0 <= here + d < self.road.count(self.s):
        return
      self.take, self.took = [float(here), float(here + d), 0.0], self.pulse_t
    a, b, p = self.take
    if p < 0.5 and self.traffic.in_gap(self.s, self.v, int(b)):
      return
    p = min(1.0, p + min(self.v / 2.0, 1.0) / TURN_TAKE_TIME * DT)
    self.take[2] = p
    self.lane = a + (b - a) * p
    if p >= 1.0:
      self.lane, self.take = b, None

  def step_car(self):
    road = self.road
    i = road.at(self.s)
    want = float(road.v[i])
    stop = self.lights.stop_for(self.s, self.t, self.stopped)
    if stop is not None:
      want = min(want, math.sqrt(max(0.0, 2 * BRAKE * (stop - self.s - STOP_SHORT))))
    lead = self.traffic.lead(self.s, self.lane)
    if lead is not None:
      want = min(want, follow(lead.v, lead.s - self.s - CAR_LEN))
    if self.s >= road.s[-1] - 10.0:
      want = 0.0
    cmd = min(max((want - self.v) / 0.5, -2 * BRAKE), ACCEL)
    self.a += (cmd - self.a) * min(DT / TAU_A, 1.0)
    s0 = self.s
    self.v = max(0.0, self.v + self.a * DT)
    self.s = min(self.s + self.v * DT, road.s[-1])
    h = float(road.heading[road.at(self.s)])
    self.yaw = math.radians(h - self.prev_heading) / DT
    self.prev_heading = h
    # through the road's lane maps, and the turns
    turn_reset = None
    while self.map_k < len(road.maps) and road.maps[self.map_k][0] <= self.s:
      sm, mp, n_after = road.maps[self.map_k]
      if sm > s0 - 1e-9 and len(mp):
        if self.lc_state == "starting":
          self.lc_from = carry_lane(self.lc_from, mp, n_after)
        if self.take is not None:
          self.lane, self.take = self.take[1], None
        before = self.lane
        self.lane = carry_lane(self.lane, mp, n_after)
        i = int(math.floor(before + 0.5))
        if 0 <= i < len(mp) and mp[i] is None:
          self.traffic.yield_to(self.s, self.lane)
      self.map_k += 1
    turns = self.turn_moves
    while self.turn_k < len(turns) and turns[self.turn_k].s + 10.0 <= self.s:
      m = turns[self.turn_k]
      out = self._turn_lane(m)
      self.take = None
      self.lane = out(self.lane)
      self.lc_state = "off" if self.lc_state == "starting" else self.lc_state
      lo, hi = (m.lanes[0], m.lanes[1]) if m.lanes is not None else (0, 0)
      turn_reset = (m.s + 10.0, out, lambda lane, lo=lo, hi=hi: lo - 0.5 <= lane <= hi + 0.5)
      self.turn_k += 1
    n = road.count(self.s)
    if n > 0 and self.lc_state != "starting" and self.take is None:
      # lane keeping: into the nearest lane there is
      self.lane = float(min(max(self.lane, 0.0), n - 1))
    self.traffic.step(self.t, DT, self, turn_reset)
    now = self.traffic.touching(self.s, self.lane)
    self.result.collisions += len(now - self.touching)
    self.touching = now

  def judge(self, info: dict, truth: list | None):
    if self.nav_step:
      for m in self.moves[self.judge_k:]:
        if m.commit - APPROACH > self.s:
          break
        lanes = judge_lanes(m, self.s, info, truth) if truth is not None else None
        if lanes is not None:
          lo, hi = lanes
          need = lo - truth[0] if truth[0] < lo else truth[0] - hi if truth[0] > hi else 0
          m.need = max(m.need or 0, need)
    while self.judge_k < len(self.moves) and self.moves[self.judge_k].commit <= self.s:
      m = self.moves[self.judge_k]
      m.judged, m.v, m.lane = True, round(self.v, 1), truth
      m.pulsed = m.kind == "turn" and self.pulse_side == m.args[0] and self.t - self.pulse_t < PULSE_MEMORY + 20.0
      lanes = judge_lanes(m, self.s, info, truth) if truth is not None else None
      if lanes is None:
        i = int(round(self.lane))
        n = max(self.road.count(self.s), 1)
        lanes = lanes_for(m.move(m.s - self.s), (i, n), None)
        m.lane = [i, n]
      m.lanes = (int(lanes[0]), int(lanes[1]), int(m.lane[1]))
      m.opens = self._opens(m, lanes)
      m.relevant = (lanes[0], lanes[1]) != (0, m.lane[1] - 1)
      m.made = lanes[0] <= m.lane[0] <= lanes[1]
      if m.kind == "turn" and self.model.turn_needs_pulse and not m.pulsed:
        m.made = False
      self.judge_k += 1

  def _opens(self, m: Maneuver, lanes: tuple[int, int]) -> float | None:
    """m before the commit point where the target lane nearest the car's begins, within APPROACH m; None if before.
    Lanes are numbered as the road's sections at the commit point (the car's own count there)."""
    before = [(s, mp, n) for s, mp, n in self.road.maps if m.commit - APPROACH < s <= m.commit + 0.5]
    if not before:
      return None
    i = int(round(self.lane))
    want = lanes[0] if i < lanes[0] else lanes[1] if i > lanes[1] else i
    at = nav.bay_opens(before, want)
    return None if at is None else round(m.commit - at, 1)

  def run(self, seconds: float | None = None) -> TripResult:
    t0 = time.perf_counter()
    limit = seconds if seconds is not None else self.route.length / 3.0 + 180.0
    frame = 0
    info = None
    truth = None
    try:
      while self.t < limit:
        truth, frac, _ = self._truth()
        self.nav_step = frame % NAV_EVERY == 0
        if self.nav_step:
          out, info = self.step_nav(truth, frac)
          if out.arrived:
            self.result.arrived = True
            break
        self.judge(info, truth)
        self.step_openpilot()
        self.step_car()
        self.t += DT
        frame += 1
        if self.s >= self.road.s[-1] - 10.5 and self.v < 0.3:
          self.result.arrived = True
          break
    except Exception as e:  # a trip the harness can't drive is reported, not fatal to the run
      import traceback
      self.result.error = f"{type(e).__name__}: {e} @ {traceback.format_exc().splitlines()[-3].strip()}"
    self._close_change("gave_up")
    if self.nav.request is not None:
      self.nav._close_request("trip over")
    self.result.requests = list(self.nav.request_log)
    self.result.driven, self.result.secs = round(self.s, 1), round(self.t, 1)
    self.result.maneuvers = [asdict(m) for m in self.moves if m.judged]
    self.result.changes = [{**asdict(c), "hanging": round(c.hanging, 2)} for c in self.changes]
    self.result.wall_s = round(time.perf_counter() - t0, 2)
    return self.result


# *** trips ***

# the e2e trips and scenarios in the repo's tests and tools (test_mapdrive's real trips, bsm_trial's FX1,
# mapdrive_trials' example): 'x,y,z,heading,lane>dx,dy'
SCENARIOS = {
  "MD012": "883.2,-2077.8,29.5,96,0>-250.5,-1067.2",
  "MD014": "670.0,-2893.8,5.2,0,9>1358.8,-1110.4",
  "MD029": "-236.0,-597.2,33.2,20,9>-743.5,-168.0",
  "LANE_END": "123.0,-517.0,42.2,198,9>-418.0,-1444.2",
  "FX1": "799.0,-1752.5,28.9,264,1>814.2,41.7",
  "MDT": "-357.8,-1658.9,18.3,331,0>1468.2,-1050.9",
}
REAL_BOX = (-1500.0, 1500.0, -2500.0, 1000.0)  # e2e.CITY: Los Santos
TRIPS_FILE = os.path.join(os.path.dirname(__file__), "navd_sim_trips.json")
# the trips' routes as Valhalla gave them (mapdrive_sim.trip_route's cache), so a run needs no router
ROUTES_FILE = os.path.join(os.path.dirname(__file__), "navd_sim_routes.json.gz")


def use_routes():
  """trip_route's cache, seeded from ROUTES_FILE (unless GTA5_TRIP_ROUTES names one)."""
  import gzip
  from openpilot.tools.sim.bridge.gta5 import mapdrive_sim as ms
  if os.getenv("GTA5_TRIP_ROUTES"):
    return
  path = os.path.expanduser("~/.cache/gta5/navd_sim_routes.json")
  if not os.path.exists(path) and os.path.exists(ROUTES_FILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(ROUTES_FILE, "rt") as f, open(path, "w") as g:
      g.write(f.read())
  ms.TRIP_CACHE = path


def save_routes(specs: list[str]):
  """The trips' routes from trip_route's cache into ROUTES_FILE."""
  import gzip
  from openpilot.tools.sim.bridge.gta5 import mapdrive_sim as ms
  cache = json.load(open(ms.TRIP_CACHE)) if os.path.exists(ms.TRIP_CACHE) else {}
  keep = {k: cache[k] for k in specs if k in cache}
  old = {}
  if os.path.exists(ROUTES_FILE):
    with gzip.open(ROUTES_FILE, "rt") as f:
      old = json.load(f)
  if keep and any(k not in old for k in keep):
    with gzip.open(ROUTES_FILE, "wt") as f:
      json.dump({**old, **keep}, f)


def random_specs(n: int, seed: int = 1, box=REAL_BOX, min_dist: float = 1500.0, max_dist: float = 3500.0) -> list[tuple[str, str]]:
  """n trips between random points on GTA's road links in `box` (two lanes or more at the start), each in a random lane
  ('x,y,z,heading,lane>dx,dy'), as e2e's random trips are picked."""
  from openpilot.tools.sim.bridge.gta5.mapdrive_sim import live_map
  m = live_map()
  paths = m["paths"]
  rng = np.random.default_rng(seed)
  x0, x1, y0, y1 = box
  keys = [k for k, link in paths.links.items() if link.lanes >= 2 and not link.no_nav and not link.shortcut]
  keys = [k for k in keys if all(x0 <= paths.xy[i][0] <= x1 and y0 <= paths.xy[i][1] <= y1 for i in k)
          and np.hypot(*(paths.xy[k[1]] - paths.xy[k[0]])) > 15.0]
  dests = [k for k, link in paths.links.items() if link.lanes >= 1 and not link.no_nav
           and all(x0 <= paths.xy[i][0] <= x1 and y0 <= paths.xy[i][1] <= y1 for i in k)]
  out = []
  while len(out) < n:
    a, b = keys[int(rng.integers(len(keys)))]
    p = (paths.xy[a] + paths.xy[b]) / 2
    z = (paths.z[a] + paths.z[b]) / 2
    h = paths.link_heading(a, b)
    for _ in range(50):
      c, d = dests[int(rng.integers(len(dests)))]
      q = (paths.xy[c] + paths.xy[d]) / 2
      if min_dist <= np.hypot(*(q - p)) <= max_dist:
        break
    else:
      continue
    lane = int(rng.integers(0, paths.links[(a, b)].lanes))
    out.append((f"R{len(out) + 1:02d}", f"{p[0]:.1f},{p[1]:.1f},{z:.1f},{h % 360:.0f},{lane}>{q[0]:.1f},{q[1]:.1f}"))
  return out


def load_trip(spec: str):
  """(route, start lane, m along it the car starts) for a trip spec on the live map, as setup places the car; None where
  it can't be routed."""
  from openpilot.tools.sim.bridge.gta5.mapdrive_sim import trip_route
  got = trip_route(spec)
  if got is None:
    return None
  route, pose = got
  route.locate(np.array(pose[:2]), None, pose[2], search=200.0)
  lane, s0 = route.lane(), route.at
  route.at, route.seg = 0.0, 0
  return route, (lane[0] if lane and lane[0] >= 0 else 0), s0


def trips(n_real: int, seed: int = 1) -> list[tuple[str, str]]:
  """The scenario trips and n_real random ones; the random specs are kept in TRIPS_FILE, so a run picks the same ones."""
  saved = json.load(open(TRIPS_FILE)) if os.path.exists(TRIPS_FILE) else {}
  key = f"{seed}"
  real = [tuple(x) for x in saved.get(key, [])]
  if len(real) < n_real:
    real = random_specs(max(n_real, len(real)), seed)
    saved[key] = real
    with open(TRIPS_FILE, "w") as f:
      json.dump(saved, f, indent=0)
  return list(SCENARIOS.items()) + real[:n_real]


# *** metrics ***

def summarise(results: list[dict]) -> dict:
  """Per run (model x lane source): over the judged maneuvers that need a lane (relevant), those in a lane for it at the
  commit point; the lane changes asked for and how they went; misses (each a reroute) and the time a change hung."""
  ms = []
  for r in results:  # a split at a fork is the fork's gore seen as lanes ending: counted once, as the fork
    forks = [m["s"] for m in r["maneuvers"] if m["kind"] == "fork"]
    ms += [m for m in r["maneuvers"] if m["kind"] != "split" or not any(abs(f - m["s"]) < nav.MERGE_FORK_NEAR for f in forks)]
  rel = [m for m in ms if m["relevant"] and m["kind"] != "drop"]
  made = [m for m in rel if m["made"]]
  needing = [m for m in rel if (m["need"] or 0) > 0]
  cs = [c for r in results for c in r["changes"]]
  done = [c for c in cs if c["outcome"] == "done"]
  v_ok = {r["model"] for r in results}
  late_v = PRESETS[next(iter(v_ok))].v_ok if len(v_ok) == 1 and next(iter(v_ok)) in PRESETS else 10.0
  by_kind = {}
  for k in ("turn", "fork", "through", "split"):
    rk = [m for m in rel if m["kind"] == k]
    by_kind[k] = [sum(m["made"] for m in rk), len(rk)]
  drops = [m for m in ms if m["kind"] == "drop" and m["relevant"]]
  return {
    "trips": len(results), "errors": sum(r["error"] is not None for r in results),
    "arrived": sum(r["arrived"] for r in results), "km": round(sum(r["driven"] for r in results) / 1000.0, 1),
    "maneuvers": len(ms), "relevant": len(rel), "in_lane": len(made),
    "in_lane_pct": round(100.0 * len(made) / max(len(rel), 1), 1),
    "needing_change": len(needing), "needing_made": sum(m["made"] for m in needing),
    "needing_pct": round(100.0 * sum(m["made"] for m in needing) / max(len(needing), 1), 1),
    "by_kind": by_kind,
    "missed": len(rel) - len(made), "forced_merges": sum(not m["made"] for m in drops),
    "nav_skipped": sum(r["skipped"] for r in results),
    "lc_requested": len(cs), "lc_done": len(done), "lc_gave_up": sum(c["outcome"] == "gave_up" for c in cs),
    "lc_failed": sum(c["outcome"] == "failed" for c in cs),
    "lc_late": sum(c["v"] < late_v for c in cs),
    "lc_started_slow": sum(c["start_v"] is not None and c["start_v"] < late_v for c in cs),
    "hanging_s": round(sum(c["hanging"] for c in cs), 1),
    "hanging_per_lc": round(sum(c["hanging"] for c in cs) / max(len(cs), 1), 2),
    "collisions": sum(r["collisions"] for r in results),
    "requests": request_outcomes([q for r in results for q in r.get("requests", [])]),
    "missed_by": missed_by(rel),
  }


LATE_LANE = 80.0  # m: a target lane beginning nearer its commit point than this opened late (a bay, a lane added)


def missed_by(rel: list[dict]) -> dict:
  """The misses by what nav had to go on: the target lane opening late (LATE_LANE m or less before the commit point:
  where Mapbox, giving no opening points, can't time a change), and with --nav mapbox, no lanes from Mapbox there."""
  out = {"late_lane": 0, "no_mapbox_lanes": 0, "other": 0}
  for m in rel:
    if m["made"]:
      continue
    if m.get("opens") is not None and m["opens"] <= LATE_LANE:
      out["late_lane"] += 1
    elif m.get("mapbox") == "none":
      out["no_mapbox_lanes"] += 1
    else:
      out["other"] += 1
  return out


def request_outcomes(reqs: list[dict]) -> dict:
  """nav's logged requests (navRequest events) by kind and outcome, and the lane changes' by reason."""
  out: dict = {}
  for q in reqs:
    k = out.setdefault(q["kind"], {})
    k[q["outcome"]] = k.get(q["outcome"], 0) + 1
    if q["kind"] == "laneChange":
      r = out.setdefault("byReason", {}).setdefault(q["reason"], {})
      r[q["outcome"]] = r.get(q["outcome"], 0) + 1
  return out


COLUMNS = [("in lane", lambda s: f"{s['in_lane']}/{s['relevant']} ({s['in_lane_pct']:.0f}%)"),
           ("needing a change", lambda s: f"{s['needing_made']}/{s['needing_change']} ({s['needing_pct']:.0f}%)"),
           ("missed", lambda s: str(s["missed"])),
           ("LC req/done/gave up/failed", lambda s: f"{s['lc_requested']}/{s['lc_done']}/{s['lc_gave_up']}/{s['lc_failed']}"),
           ("LC late (asked < v_ok)", lambda s: str(s["lc_late"])),
           ("hanging s (per LC)", lambda s: f"{s['hanging_s']:.0f} ({s['hanging_per_lc']:.1f})"),
           ("forced merges", lambda s: str(s["forced_merges"])),
           ("contacts (sim)", lambda s: str(s["collisions"])),
           ("arrived", lambda s: f"{s['arrived']}/{s['trips']}")]


def kinds(rows: list[tuple[str, dict]]) -> str:
  """In a lane at the commit point, by kind of maneuver."""
  out = ["| run | turns | forks / exits | straight on | lanes leaving at a split |", "|---|---|---|---|---|"]
  for name, s in rows:
    out.append(f"| {name} | " + " | ".join(f"{a}/{b}" for a, b in (s["by_kind"][k] for k in ("turn", "fork", "through", "split"))) + " |")
  return "\n".join(out)


def table(rows: list[tuple[str, dict]]) -> str:
  head = "| run | " + " | ".join(c for c, _ in COLUMNS) + " |"
  out = [head, "|" + "---|" * (len(COLUMNS) + 1)]
  for name, s in rows:
    out.append(f"| {name} | " + " | ".join(f(s) for _, f in COLUMNS) + " |")
  return "\n".join(out)


def parse_head(spec: str) -> HeadCfg:
  """'truth', or 'model:0.85' / 'fused:0.9' (strict accuracy), optionally ':lag:drop:count_acc'."""
  parts = spec.split(":")
  if parts[0] == "truth":
    return HeadCfg("truth")
  vals = [float(p) for p in parts[1:]]
  cfg = HeadCfg(parts[0])
  for k, v in zip(("acc", "lag", "drop", "count_acc"), vals, strict=False):
    setattr(cfg, k, v)
  return cfg


_LOADED: list = []  # (name, spec, route, lane, s0), for the workers forked from run()
_JOB: dict = {}


def _one(k: int) -> dict:
  name, _spec, route, lane, s0 = _LOADED[k]
  j = _JOB
  model = PRESETS[j["model"]]
  if j.get("no_turn_take"):
    model = ModelCfg(**{**asdict(model), "turn_takes_lane": False})
  sim = NavSim(route, lane, model, parse_head(j["head"]), TrafficCfg(density=j["density"]), j["tune"],
               seed=j["seed"] * 1000 + k, name=name, s0=s0, stage=j.get("stage", 3), nav=j.get("nav", "map"))
  return asdict(sim.run())


def run(names_specs, models: list[str], heads: list[str], tune: dict | None = None, density: float = 10.0, seed: int = 0,
        quiet: bool = False, jobs: int = 1, no_turn_take: bool = False, stage: int = 3, navs: tuple = ("map",)) -> dict:
  """Each nav x model x lane source over the trips: {"[mapbox / ]model / head": {"summary": ..., "trips": [...]}}."""
  import multiprocessing
  _LOADED.clear()
  for name, spec in names_specs:
    got = load_trip(spec)
    if got is None:
      if not quiet:
        print(f"{name}: no route", file=sys.stderr)
      continue
    _LOADED.append((name, spec, *got))
  out = {}
  for nav_kind, model, head in [(a, b, c) for a in navs for b in models for c in heads]:
    key = ("mapbox / " if nav_kind == "mapbox" else "") + f"{model} / {head}"
    _JOB.update(model=model, head=head, density=density, tune=tune, seed=seed, no_turn_take=no_turn_take, stage=stage,
                nav=nav_kind)
    if jobs > 1:
      with multiprocessing.get_context("fork").Pool(jobs) as pool:
        res = pool.map(_one, range(len(_LOADED)), chunksize=1)
    else:
      res = [_one(k) for k in range(len(_LOADED))]
    for r in res:
      if not quiet:
        rel = [m for m in r["maneuvers"] if m["relevant"] and m["kind"] != "drop"]
        made, done = sum(m["made"] for m in rel), sum(c["outcome"] == "done" for c in r["changes"])
        err = "  " + r["error"] if r["error"] else ""
        line = f"{key:24s} {r['name']:9s} {r['driven']:6.0f}/{r['length']:6.0f} m {r['secs']:5.0f} s  in lane {made}/{len(rel)}"
        print(f"{line}  LC {done}/{len(r['changes'])}  ({r['wall_s']:.1f} s){err}", file=sys.stderr, flush=True)
    out[key] = {"summary": summarise(res), "trips": res}
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--routes", type=int, default=40, help="random real routes on top of the scenario trips")
  ap.add_argument("--seed", type=int, default=1, help="the random routes' seed")
  ap.add_argument("--sim-seed", type=int, default=0, help="traffic, lights and lane head noise")
  ap.add_argument("--model", default="n16,finetuned", help=f"stand-in settings: {','.join(PRESETS)}")
  ap.add_argument("--lane", default="truth", help="lane sources: truth, model:ACC[:LAG:DROP:COUNT_ACC], fused:ACC...")
  ap.add_argument("--density", type=float, default=10.0, help="traffic, vehicles per km per lane")
  ap.add_argument("--tune", default="{}", help="planner Tune overrides, JSON")
  ap.add_argument("--only", default="", help="comma-separated trip names")
  ap.add_argument("--out", default="", help="the full results, JSON")
  ap.add_argument("--requests", default="", help="every request nav made (navRequest events, with the run and trip), JSON lines")
  ap.add_argument("--stage", type=int, default=3, help="PLAN D's stage nav's requests are made at (the driver approves any)")
  ap.add_argument("--no-turn-take", action="store_true", help="the stand-in's turn desire doesn't take it a lane over")
  ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="trips run in parallel")
  navs = "map (navd's planner on the full map, as the bridge runs it), mapbox (comma navd's lane guidance on Mapbox data)"
  ap.add_argument("--nav", default="map", help=f"{navs}, or both: map,mapbox")
  args = ap.parse_args()
  use_routes()
  todo = trips(args.routes, args.seed)
  if args.only:
    keep = set(args.only.split(","))
    todo = [t for t in todo if t[0] in keep]
  res = run(todo, args.model.split(","), args.lane.split(","), json.loads(args.tune), args.density, args.sim_seed,
            jobs=args.jobs, no_turn_take=args.no_turn_take, stage=args.stage, navs=tuple(args.nav.split(",")))
  save_routes([spec for _, spec in todo])
  print(table([(k, v["summary"]) for k, v in res.items()]))
  print()
  print(kinds([(k, v["summary"]) for k, v in res.items()]))
  if args.out:
    with open(args.out, "w") as f:
      json.dump(res, f)
  if args.requests:
    with open(args.requests, "w") as f:
      for key, v in res.items():
        for trip in v["trips"]:
          for q in trip["requests"]:
            f.write(json.dumps({"run": key, "trip": trip["name"], **q}) + "\n")


if __name__ == "__main__":
  main()
