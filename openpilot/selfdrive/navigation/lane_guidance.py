"""Step 1 lane guidance: lane changes for the route's maneuvers, from what a real car has. navd's Mapbox route (steps,
maneuvers, each intersection's approach lanes, maxspeed), the driving model's current-lane head (modelV2.laneHead:
lane from the left, lane count, probability) and carState; nothing else: no OSM lane tags, bay openings, lane widths
or truth lane. It asks the driver for lane changes (navInstruction.request, NavDesire sided by the approval stage),
signals turns, and shows the lanes under active guidance (navInstruction.lanes), as nav apps do.

- Targets: each maneuver's intersection's valid lanes (Mapbox's `lanes`, left to right); where the next maneuver follows
  within `pair_gap` m, the first's targets narrowed to those nearest the second's side, so both can be made. At a
  junction the route goes straight on through, the valid lanes if not all are. No lanes from Mapbox at a turn or exit:
  the outermost lane on its side.
- The car's lane: the lane head's, held through flicker and dropouts (LaneReader), counted from the maneuver's side.
  Where the model counts fewer lanes than Mapbox lists at the junction (a turn bay not open yet), the missing lanes are
  taken to be on the turn's side: counted from there, the target is the outermost lane, and the car is in place in it;
  the turn's signal takes it into the bay as it opens (the turn desire, `turn_into`).
- Timing: a change is asked for once time left <= lanes away x lc_time + margin, time left being the distance to the
  last place it must be done (`done_before` m before the maneuver) over the planning speed max(v, maxspeed_share x
  maxspeed), so slowing traffic doesn't put the ask off; at the latest `min_dist` m per lane away; with lc_time_fwy at
  freeway speeds. The last place moves to `signal_room` m before the last traffic signal on the way (Mapbox's
  traffic_signal: a queue there, where the model can't change lanes). A maneuver's remaining changes follow
  `change_gap` after each. Below `lc_min_speed` (the model's lane change speed: n16 ~10 m/s) a change not yet started
  is withdrawn and left to the turn desire.
- Request and approve (PLAN D): stage 1 the driver's blinker on the request's side approves (NavDesire
  "laneChange:<side>"), 3 auto-approved (GTA, NavDesire "laneChange"). `requests` holds what the driver (or GTA's
  simulated one) does with the stalk: laneChangeLeft / signalTurnRight / cancelSignal.

Game-agnostic: imports nothing from tools/sim. navd.py runs it behind the NavLaneGuidance param (or NAVD_LANE_GUIDANCE=1).
"""
import json
import os
from dataclasses import dataclass, field, fields

import numpy as np

from openpilot.selfdrive.navigation.route import cumulative_distance, local_meters

LANE_TURN_SPEED = 19 * 0.44704  # m/s: below it a blinker is a turn, unless NavDesire makes it a lane change (desire_helper)
PARAM_LEAD = 0.4  # s: openpilot reads NavDesire every 0.2 s, so the blinker comes on after it's set
PARAM_HOLD = 0.5  # s NavDesire stays a lane change after the change
BLINKER_GAP = 0.3  # s the blinker stays off between one request's and the next's
FREEWAY_SPEED = 26.0  # m/s (~60 mph) maxspeed: a freeway's, wherever Mapbox doesn't say motorway
APPROACH = 60.0  # m: lanes on intersections this near before a maneuver are its approach's
TURN_KINDS = ("turn", "end of road", "continue", "new name")
FORK_KINDS = ("fork", "off ramp")  # on ramps and merges join a road: only their lanes, where given, say anything
ROUNDABOUTS = ("roundabout", "rotary", "roundabout turn", "exit roundabout", "exit rotary")
REASONS = {"turn": "turn", "fork": "fork", "exit": "exit", "through": "straightOn", "roundabout": "turn"}
# Mapbox lane indications -> navInstruction.Lane.Direction
DIRECTIONS = {"left": "left", "sharp left": "left", "slight left": "slightLeft", "straight": "straight",
              "slight right": "slightRight", "right": "right", "sharp right": "right", "uturn": "left", "none": "none"}


@dataclass
class Tune:
  lc_time: float = 6.0  # s a lane change takes, with the gap before the next
  lc_time_fwy: float = 8.0  # s at freeway speeds
  fwy_speed: float = 22.0  # m/s planning speed from which lc_time_fwy applies
  margin: float = 4.0  # s more
  maxspeed_share: float = 0.8  # of the road's maxspeed: the planning speed at least
  min_dist: float = 60.0  # m per lane away: asked by this distance before the last place, whatever the speed
  done_before_turn: float = 35.0  # m before a turn's (or roundabout's) maneuver point the changes must be done
  done_before_fork: float = 60.0  # m before a fork's or exit's
  done_before_through: float = 30.0  # m before a junction the route goes straight on through
  signal_room: float = 30.0  # m before the last traffic signal on the way to a move: a queue there, where the model can't change
  signal_window: float = 150.0  # m before a move, the signals that count
  lc_min_speed: float = 10.0  # m/s: slower, a change not yet started is withdrawn (the model's lane change speed)
  pair_gap: float = 100.0  # m: a maneuver this soon after one is planned with it
  max_ahead: float = 3500.0  # m: no change asked for further out
  fallback: bool = True  # no lanes from Mapbox at a turn or exit: the outermost lane on its side
  ui_town: float = 402.0  # m before a maneuver its lanes show, without a banner saying
  ui_fwy: float = 1609.0
  # the car's lane from the lane head
  lane_prob: float = 0.7  # a new lane is taken once read this sure
  lane_hold: float = 1.0  # s, held
  lane_keep: float = 0.4  # the lane held stays while read at least this sure
  lane_stale: float = 3.0  # s without such a reading: no lane
  # turns
  signal_time: float = 5.0  # s before a turn its signal comes on, below the lane turn speed
  signal_max: float = 50.0  # m
  signal_min: float = 28.0  # m: signalled by here whatever the speed
  signal_last: float = 26.0  # m: a change towards the turn still under way gives way to its signal by here
  turn_into: float = 80.0  # m: slower than lc_min_speed, one lane beside the turn's (or before its bay opens), its signal takes the car over
  turn_done: float = 20.0  # m past the turn: its signal off
  # requests
  change_timeout: float = 10.0  # s a lane change may hang before it's withdrawn (desire_helper's)
  change_gap: float = 2.0  # s between lane changes
  change_tries: int = 3  # for one maneuver without progress

  @classmethod
  def of(cls, overrides: dict | None = None) -> "Tune":
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in (overrides or {}).items() if k in names})


# *** the route ***

def decode_polyline(encoded: str, precision: int = 6) -> list[tuple[float, float]]:
  """Google's encoded polyline: (lat, lon) pairs."""
  out, index, lat, lon = [], 0, 0, 0
  while index < len(encoded):
    vals = []
    for _ in range(2):
      shift = result = 0
      while True:
        b = ord(encoded[index]) - 63
        index += 1
        result |= (b & 0x1f) << shift
        shift += 5
        if b < 0x20:
          break
      vals.append(~(result >> 1) if result & 1 else result >> 1)
    lat, lon = lat + vals[0], lon + vals[1]
    out.append((lat / 10 ** precision, lon / 10 ** precision))
  return out


def coordinates(geometry, precision: int = 6) -> np.ndarray:
  """A Mapbox geometry (geojson or an encoded polyline) as (lon, lat) rows."""
  if isinstance(geometry, dict):
    return np.asarray(geometry["coordinates"], dtype=np.float64).reshape((-1, 2))
  return np.asarray([(lo, la) for la, lo in decode_polyline(geometry, precision)], dtype=np.float64).reshape((-1, 2))


def speed_of(m: dict) -> float | None:
  """A maxspeed annotation in m/s; None unknown or none."""
  if "speed" not in m:
    return None
  return float(m["speed"]) * (0.44704 if m.get("unit") == "mph" else 1 / 3.6)


@dataclass
class Move:
  """A place on the route nav plans lanes for: a maneuver, or a junction the route goes straight on through where not
  every lane does."""
  along: float  # m along the route
  kind: str  # "turn", "fork", "exit", "roundabout", "through"
  side: str | None  # "left" / "right": the side its lanes are counted from
  lanes: list[dict] | None  # Mapbox's lanes at it, left to right (cycle lanes dropped), None without
  freeway: bool = False
  ui_from: float = 0.0  # m before it its lanes show
  name: str = ""
  step: int = -1  # its step, for a maneuver
  signal: bool = False  # nav signals it (a turn left or right)

  @property
  def valid(self) -> list[int]:
    return [i for i, lane in enumerate(self.lanes or []) if lane.get("valid")]


def car_lanes(lanes: list[dict]) -> list[dict]:
  """Mapbox's lanes less those only for cycles (bus lanes are lanes the model sees)."""
  return [lane for lane in lanes if lane.get("access", {}).get("designated") != ["bicycle"]]


def motorway(it: dict) -> bool:
  """An intersection on a motorway (Mapbox calls the UK's A-road trunks "trunk": not one)."""
  return "motorway" in it.get("classes", []) or it.get("mapbox_streets_v8", {}).get("class") in ("motorway", "motorway_link")


def side_of(modifier: str) -> str | None:
  return "left" if "left" in modifier else "right" if "right" in modifier else None


def edge_side(valid: list[int], n: int) -> str | None:
  """For a way straight on: the edge its valid lanes touch (the nearer one if neither), None if all lanes are valid."""
  if not valid or len(valid) == n:
    return None
  if valid[0] == 0:
    return "left"
  if valid[-1] == n - 1:
    return "right"
  return "left" if valid[0] <= n - 1 - valid[-1] else "right"


class MapboxRoute:
  """A Mapbox route (routes[0] of a Directions response): its length, the moves nav plans lanes for along it, and its
  maxspeed by distance along."""
  def __init__(self, route: dict, tune: Tune | None = None):
    t = tune or Tune()
    steps = [s for leg in route["legs"] for s in leg["steps"]]
    self.steps = steps
    starts = np.r_[0.0, np.cumsum([s["distance"] for s in steps])]
    self.length = float(starts[-1])
    # the route's geometry, for the intersections' places and maxspeed (overview=full), else the steps'
    if route.get("geometry") is not None:
      geo = coordinates(route["geometry"])
    else:
      geo = np.concatenate([coordinates(s["geometry"]) for s in steps]) if steps else np.zeros((0, 2))
    along = cumulative_distance(geo) if len(geo) > 1 else np.zeros(len(geo))
    scale = self.length / along[-1] if len(along) and along[-1] > 0 else 1.0
    self.geo_along = along * scale
    speeds = [speed_of(m) for leg in route["legs"] for m in leg.get("annotation", {}).get("maxspeed", [])]
    self.maxspeeds = speeds if len(speeds) == max(len(geo) - 1, 0) else []

    def at(it: dict, k: int) -> float:
      gi = it.get("geometry_index")
      if gi is not None and route.get("geometry") is not None and gi < len(self.geo_along):
        return float(self.geo_along[gi])
      pts = coordinates(steps[k]["geometry"])
      if len(pts) < 2:
        return float(starts[k])
      p = local_meters(it["location"], pts[0])
      seg = local_meters(pts, pts[0])
      d = np.linalg.norm(seg - p, axis=1)
      return float(starts[k] + cumulative_distance(pts)[int(d.argmin())])

    self.moves: list[Move] = []
    self.signals: list[float] = []  # m along to the intersections with traffic signals (Mapbox's traffic_signal)
    runs: list[tuple[float, list[dict], int, bool]] = []  # each run of an intersection's lanes: (last along, lanes, step)
    for k, s in enumerate(steps):
      for it in s.get("intersections") or []:
        lanes = car_lanes(it["lanes"]) if it.get("lanes") else None
        a = at(it, k)
        if it.get("traffic_signal"):
          self.signals.append(a)
        fwy = motorway(it) or (self.maxspeed(a) or 0.0) >= FREEWAY_SPEED
        if lanes and runs and runs[-1][1] == lanes and runs[-1][2] == k:
          runs[-1] = (a, lanes, k, fwy)
        elif lanes:
          runs.append((a, lanes, k, fwy))
    maneuver_at = []
    for k, s in enumerate(steps):
      if k == 0:
        continue
      mv = s["maneuver"]
      typ, mod = mv.get("type", ""), mv.get("modifier", "")
      if typ == "arrive":
        continue
      a = float(starts[k])
      int0 = s["intersections"][0] if s.get("intersections") else {}
      lanes = car_lanes(int0["lanes"]) if int0.get("lanes") else None
      if lanes is None:  # the approach's, where the maneuver's own intersection has none (not another maneuver's)
        near = [r for r in runs if r[2] == k - 1 and 0.0 <= a - r[0] <= APPROACH and r[0] > float(starts[k - 1]) + 5.0]
        lanes = near[-1][1] if near else None
      fwy = motorway(int0) or (self.maxspeed(a - 50.0) or 0.0) >= FREEWAY_SPEED
      side = side_of(mod)
      if typ in ROUNDABOUTS:
        kind = "roundabout"
      elif typ in FORK_KINDS:
        kind = "exit" if typ == "off ramp" or (fwy and typ == "fork") else "fork"
      elif typ in TURN_KINDS and mod in ("left", "right", "sharp left", "sharp right", "slight left", "slight right"):
        kind = "turn"
      else:
        kind = "through"
      if kind == "through":
        side = edge_side([i for i, x in enumerate(lanes or []) if x.get("valid")], len(lanes or []))
        if side is None:
          continue
      if mod == "uturn" or side is None:
        continue
      ui = self._banner_lanes_from(steps[k - 1]) or (t.ui_fwy if fwy else t.ui_town)
      self.moves.append(Move(a, kind, side, lanes, fwy, ui, s.get("name", ""), k,
                             signal=kind == "turn" and mod in ("left", "right", "sharp left", "sharp right")))
      maneuver_at.append(a)
    # junctions straight on through, where not every lane goes on
    for a, lanes, _, fwy in runs:
      if any(-5.0 <= m - a <= APPROACH for m in maneuver_at):
        continue  # a maneuver's own approach (Mapbox repeats its lanes on the way into it)
      valid = [i for i, x in enumerate(lanes) if x.get("valid")]
      side = edge_side(valid, len(lanes))
      if side is None or not valid:
        continue
      self.moves.append(Move(a, "through", side, lanes, fwy, t.ui_fwy if fwy else t.ui_town))
    self.moves.sort(key=lambda m: m.along)

  @staticmethod
  def _banner_lanes_from(step: dict) -> float | None:
    for b in step.get("bannerInstructions", []):
      if any(c.get("type") == "lane" for c in (b.get("sub") or {}).get("components", [])):
        return float(b["distanceAlongGeometry"])
    return None

  def maxspeed(self, along: float) -> float | None:
    if not self.maxspeeds:
      return None
    k = int(np.clip(np.searchsorted(self.geo_along, along, side="right") - 1, 0, len(self.maxspeeds) - 1))
    return self.maxspeeds[k]


# *** the car's lane ***

class LaneReader:
  """The lane head's lane, held: a new lane taken once read lane_prob sure for lane_hold s, the lane held kept while read
  lane_keep sure and through unsure readings and dropouts for lane_stale s; nav's own lane change moves it over."""
  def __init__(self, tune: Tune):
    self.t = tune
    self.lane: tuple[int, int] | None = None
    self.seen: tuple[tuple[int, int] | None, float] = (None, 0.0)
    self.kept = -1e9  # when the lane held was last read

  def update(self, head: list | None, now: float) -> tuple[int, int] | None:
    t = self.t
    if head is not None and head[0] is not None and head[0] >= 0 and head[1] > 0:
      read = (int(head[0]), int(head[1]))
      p = float(head[2]) if len(head) > 2 else 1.0
      if self.lane is not None and read == self.lane and p >= t.lane_keep:
        self.kept = now
      if p >= t.lane_prob:
        if self.seen[0] != read:
          self.seen = (read, now)
        if read != self.lane and now - self.seen[1] >= t.lane_hold - 1e-6:
          self.lane, self.kept = read, now
      else:
        self.seen = (None, now)
    if self.lane is not None and now - self.kept > t.lane_stale:
      self.lane = None
    return self.lane

  def moved(self, side: str, now: float):
    """nav's lane change done: the lane beside, until the head says otherwise."""
    if self.lane is not None:
      i, n = self.lane
      self.lane, self.kept = (min(max(i + (-1 if side == "left" else 1), 0), n - 1), n), now
      self.seen = (None, now)


def from_side(lane: tuple[int, int], side: str) -> int:
  i, n = lane
  return i if side == "left" else n - 1 - i


def targets(move: Move, n_car: int, fallback: bool = True) -> list[int] | None:
  """The lanes for a move counted from its side (0 = outermost there), in the car's count n_car; None: any lane."""
  side = move.side
  if not move.lanes:
    return [0] if fallback and move.kind in ("turn", "fork", "exit") else None
  n = len(move.lanes)
  valid = move.valid
  if not valid or len(valid) == n:
    return None
  t = sorted({j if side == "left" else n - 1 - j for j in valid})
  if n_car < n and move.kind != "through":
    # the model counts fewer: the lanes not yet there (a bay) are on the move's side, the outermost lane stands for them
    d = n - n_car
    t = sorted({max(x - d, 0) for x in t})
  return [x for x in t if x < max(n_car, 1)] or [min(t[0], max(n_car - 1, 0))]


def narrowed(first: list[int], second_side: str | None, first_side: str) -> list[int]:
  """The first move's targets nearest the second's side, for both."""
  if second_side is None or len(first) < 2:
    return first
  return [min(first)] if second_side == first_side else [max(first)]


# *** guidance ***

@dataclass
class GuidanceInputs:
  t: float  # s, monotonic
  along: float  # m along the route (navd's progress)
  v: float  # m/s
  lane_head: list | None = None  # modelV2.laneHead: [lane from the left, lanes, probability]; None without a fresh one
  blinker: str | None = None  # "left" / "right" (carState)
  changing: bool = False  # openpilot's lane change under way (laneChangeState starting / finishing)
  left_blindspot: bool = False
  right_blindspot: bool = False
  stage: int = 1  # NavLaneChangeStage


@dataclass
class GuidanceOutputs:
  requests: list[str]  # to the driver, in order: laneChangeLeft/Right, signalTurnLeft/Right, cancelSignal
  desires: list[str]  # NavDesire's values set this step
  request: dict | None  # navInstruction.request now: kind, side, reason, distance, laneChanges, stage, id, approved
  lanes: list[dict] | None  # navInstruction.lanes under active guidance: directions, active, activeDirection, current
  lane_distance: float = 0.0  # m by which to be in an active lane, 0 when in one (navInstruction.laneDistance)
  events: list[dict] = field(default_factory=list)  # each request once over ({"event": "navRequest", ...})


class LaneGuidance:
  """Step 1 lane guidance over a Mapbox route: update() each step (10 Hz) with the car's progress, speed and lane head."""
  def __init__(self, tune: Tune | dict | None = None):
    self.tune = tune if isinstance(tune, Tune) else Tune.of(tune)
    self.reader = LaneReader(self.tune)
    self.route: MapboxRoute | None = None
    self.lane: tuple[int, int] | None = None
    self.desire = ""  # NavDesire as set
    self.request: dict | None = None
    self.request_meta: dict = {}
    self.request_id = 0
    self.request_log: list[dict] = []  # every request once over, as its navRequest event
    self.changing: str | None = None  # the side of the change asked for, until it's over
    self.change_side: str | None = None  # the last change's side, for NavDesire's hold
    self.change_t = -1e9  # when it was asked for, or the last ended
    self.change_send_at = 0.0  # when its blinker comes on (once openpilot has read NavDesire); 0 once on
    self.change_seen = False  # the blinker came on its side
    self.change_started = False  # openpilot began it
    self.change_hold_until = 0.0
    self.tries: dict[float, int] = {}  # changes asked for each move (by where it is) without one done
    self.series: float | None = None  # the move whose changes are under way: the next follows change_gap after one
    self.signaled: Move | None = None  # the turn signalled
    self.signal_at = 0.0  # its signal not before (after a change's blinker went off)
    self.was_changing = False
    self.now = 0.0
    self.along = 0.0
    self.v = 0.0
    self.stage = 1
    self.requests: list[str] = []
    self.desires: list[str] = []
    self.events: list[dict] = []
    self.skipped: list = []  # turns left to the route (none: the model tries from where it is)

  def set_route(self, route: dict | MapboxRoute | None):
    """A new route (routes[0] of navd's Mapbox response, or a MapboxRoute), or None to clear it."""
    if self.request is not None:
      self._close_request("replaced")
    self.route = None  # a route that fails to read leaves none, never the last one
    self.route = route if isinstance(route, MapboxRoute) or route is None else MapboxRoute(route, self.tune)
    self.tries.clear()
    self.changing, self.signaled, self.change_send_at, self.series = None, None, 0.0, None

  @property
  def signaling(self) -> bool:
    return self.signaled is not None

  # *** the step ***

  def update(self, inp: GuidanceInputs) -> GuidanceOutputs:
    self.requests, self.desires, self.events = [], [], []
    self.now, self.along, self.v, self.stage = inp.t, inp.along, inp.v, inp.stage
    self.lane = self.reader.update(inp.lane_head, inp.t)
    lanes_ui, lane_distance = None, 0.0
    if self.route is not None:
      self._watch(inp)
      ahead = [m for m in self.route.moves if m.along - inp.along > -self._passed(m) and m.along - inp.along < self.tune.max_ahead]
      self._turn(inp, ahead)
      needs = self._needs(ahead)
      self._change(inp, needs)
      shown = next((m for m in ahead if m.lanes and m.along - inp.along > 0.0), None)
      if shown is not None:
        lanes_ui, lane_distance = self._lanes_ui(shown, next((nd for m, nd in needs if m is shown), None))
    self._set_desire()
    if self.request is not None:
      self.request["approved"] = inp.blinker == self.request["side"]
      if self.request["approved"] and self.request_meta.get("approved_t") is None:
        self.request_meta["approved_t"] = inp.t
    return GuidanceOutputs(self.requests, self.desires, self._request_now(), lanes_ui, lane_distance, self.events)

  def _passed(self, m: Move) -> float:
    return self.tune.turn_done if m.signal else 2.0

  def _done_before(self, m: Move) -> float:
    """m before the move its changes must be done by: done_before for its kind, or signal_room before the last traffic
    signal within signal_window on the way to it, whichever is further."""
    t = self.tune
    d = {"fork": t.done_before_fork, "exit": t.done_before_fork, "through": t.done_before_through}.get(m.kind, t.done_before_turn)
    lights = [a for a in (self.route.signals if self.route is not None else []) if 0.0 < m.along - a <= t.signal_window]
    return max(d, m.along - max(lights) + t.signal_room) if lights and t.signal_room > 0 else d

  def _targets(self, m: Move, ahead: list[Move], n_car: int) -> list[int] | None:
    """The move's target lanes from its side in the car's count; narrowed for a maneuver close after it."""
    tg = targets(m, n_car, self.tune.fallback)
    nxt = next((x for x in ahead if x.along > m.along + 1.0), None)
    if tg is not None and nxt is not None and nxt.along - m.along <= self.tune.pair_gap and nxt.kind != "through":
      tg = narrowed(tg, nxt.side, m.side)
    return tg

  def _need(self, m: Move, ahead: list[Move]) -> dict | None:
    """What the car needs for the move: "changes" (lanes away), "side" (of the first change), "deadline" (m to the last
    place they're done), "bay" (in place before its bay opens), "lanes" (the targets as lanes from the left in the car's
    count, None: any); None where the car's lane isn't known."""
    if self.lane is None:
      return None
    i, n = self.lane
    deadline = m.along - self.along - self._done_before(m)
    tg = self._targets(m, ahead, n)
    if tg is None:
      return {"changes": 0, "side": m.side, "deadline": deadline, "bay": False, "lanes": None}
    k = from_side(self.lane, m.side)
    best = min(tg, key=lambda x: abs(x - k))
    other = "right" if m.side == "left" else "left"
    bay = bool(m.lanes) and n < len(m.lanes) and m.kind != "through" and best == 0 and k == 0
    lanes = {x if m.side == "left" else n - 1 - x for x in tg}
    return {"changes": abs(k - best), "side": m.side if k > best else other, "deadline": deadline, "bay": bay, "lanes": lanes}

  def _needs(self, ahead: list[Move]) -> list[tuple[Move, dict]]:
    out = []
    for m in ahead:
      nd = self._need(m, ahead[ahead.index(m):])
      if nd is not None:
        out.append((m, nd))
    return out

  def _plan_speed(self) -> float:
    vmax = self.route.maxspeed(self.along) if self.route is not None else None
    return max(self.v, self.tune.maxspeed_share * vmax if vmax else 0.0)

  def _due(self, need: dict) -> bool:
    t = self.tune
    left = need["deadline"]
    if need["changes"] <= 0:
      return False
    vp = max(self._plan_speed(), 1.0)
    lc = t.lc_time_fwy if vp >= t.fwy_speed else t.lc_time
    return left / vp <= need["changes"] * lc + t.margin or left <= need["changes"] * t.min_dist

  # *** lane changes ***

  def _watch(self, inp: GuidanceInputs):
    """The change asked for: its blinker on once NavDesire is read, then started, done, declined, timed out or
    withdrawn."""
    if self.changing is None:
      self.was_changing = inp.changing
      return
    if inp.changing:
      self.change_started = True
    self.change_seen |= inp.blinker == self.changing
    if self.change_started and not inp.changing and self.was_changing:
      self.reader.moved(self.changing, inp.t)
      self.lane = self.reader.lane
      self._end_change(inp.blinker, "done")
    elif self.change_seen and not self.change_started and inp.blinker != self.changing:
      self._end_change(inp.blinker, "declined")  # the blinker off before openpilot could start it
    elif not self.change_started and inp.t - self.change_t > self.tune.change_timeout:
      self._end_change(inp.blinker, "timeout")
    elif not self.change_started and inp.v < self.tune.lc_min_speed:
      self._end_change(inp.blinker, "gave_up")  # too slow for the model: left to the turn desire
    elif self.change_send_at and inp.t >= self.change_send_at:
      if inp.blinker is not None and inp.blinker != self.changing:
        # a blinker still on (another side): off first, as openpilot starts a change on the blinker's edge
        self.requests.append("cancelSignal")
        self.change_send_at = inp.t + BLINKER_GAP
      else:
        self.change_send_at = 0.0
        self.requests.append("laneChange" + self.changing.capitalize())
    self.was_changing = inp.changing

  def _change(self, inp: GuidanceInputs, needs: list[tuple[Move, dict]]):
    """Into the lanes for the moves ahead (nearest first): the first the car isn't in the lanes for, once due, by a change
    that doesn't leave the lanes of those before it."""
    t = self.tune
    if self.lane is None or self.changing is not None or self.signaled is not None or inp.changing:
      return
    if inp.t - self.change_t < t.change_gap or inp.v < t.lc_min_speed:
      return
    i, n = self.lane
    for m, nd in needs:
      if nd["changes"] <= 0:
        continue
      if m.signal and self._turn_into(m, nd):
        return
      key = round(m.along, 1)
      if not (self._due(nd) or self.series == key) or self.tries.get(key, 0) >= t.change_tries:
        continue
      to = i + (-1 if nd["side"] == "left" else 1)
      before = [(x, y) for x, y in needs if x.along < m.along and y["lanes"]]
      if any(min(abs(to - j) for j in y["lanes"]) > min(abs(i - j) for j in y["lanes"]) for _, y in before):
        return  # not away from the lanes for a move before it
      # a move before it wanting the same way: the change is the earlier's (its reason, deadline and count)
      m, nd = next(((x, y) for x, y in before if y["changes"] > 0 and y["side"] == nd["side"]), (m, nd))
      key = round(m.along, 1)
      self.tries[key] = self.tries.get(key, 0) + 1
      self._open_request("laneChange", nd["side"], REASONS.get(m.kind, "other"), max(nd["deadline"], 0.0), nd["changes"],
                         m.along - self.along, f"{nd['changes']} lane(s) {nd['side']} for the {m.kind} in {m.along - self.along:.0f} m")
      self.changing, self.change_side, self.change_t = nd["side"], nd["side"], inp.t
      self.change_started, self.change_seen = False, False
      self.change_send_at = inp.t + PARAM_LEAD
      self.was_changing = inp.changing
      return

  def _end_change(self, blinker: str | None, outcome: str):
    if self.changing is None:
      return
    if self.request is not None and self.request["kind"] == "laneChange":
      if outcome == "done":
        self.tries.pop(self.request_meta.get("move"), None)
        self.series = self.request_meta.get("move")
      self._close_request(outcome)
    if blinker == self.changing:
      self.requests.append("cancelSignal")  # the driver's stalk off once it's over, done or not
    self.changing, self.change_t, self.change_send_at, self.change_started = None, self.now, 0.0, False
    self.change_hold_until = self.now + PARAM_HOLD

  # *** turns ***

  def _turn_into(self, m: Move, need: dict) -> bool:
    """Slower than the model changes lanes at and near the turn: one lane beside its lanes on its side, or in place
    before its bay opens, so the turn's signal takes the car over (the turn desire), not a lane change."""
    t = self.tune
    if t.turn_into <= 0 or m.along - self.along > t.turn_into or self.v >= min(t.lc_min_speed, LANE_TURN_SPEED):
      return False
    return need["bay"] or (need["changes"] == 1 and need["side"] == m.side)

  def _turn(self, inp: GuidanceInputs, ahead: list[Move]):
    """The turn ahead's signal: on near it once slow (the lane turn speed), or by signal_min whatever the speed, or
    earlier where its signal is to take the car over (turn_into); off once past it."""
    t = self.tune
    if self.signaled is not None:
      if self.along > self.signaled.along + t.turn_done:
        if self.request is not None and self.request["kind"] == "turn":
          self._close_request("done")
        if inp.blinker == self.signaled.side:
          self.requests.append("cancelSignal")
        self.signaled = None
      return
    m = next((x for x in ahead if x.signal and x.along - self.along > -2.0), None)
    if m is None:
      return
    dist = m.along - self.along
    need = self._need(m, ahead[ahead.index(m):])
    into = need is not None and self._turn_into(m, need)
    window = max(t.signal_min, min(t.signal_max, inp.v * t.signal_time))
    if not (dist < t.signal_min or (dist < window and inp.v < LANE_TURN_SPEED) or into):
      return
    if self.changing is not None:
      if not into and dist > t.signal_last:
        return  # the change towards it may still finish
      self._end_change(inp.blinker, "turn")
      self.signal_at = self.now + PARAM_HOLD + PARAM_LEAD  # the blinker off, NavDesire cleared, then the turn's
      return
    if self.now < max(self.signal_at, self.change_hold_until + PARAM_LEAD):
      return
    self.signaled = m
    self._open_request("turn", m.side, "turn", max(dist, 0.0), 0, dist, f"signal the {m.side} turn in {dist:.0f} m")
    self.requests.append("signalTurn" + m.side.capitalize())

  # *** NavDesire, the request, the lanes ***

  def _set_desire(self):
    """A lane change while one is asked for (and PARAM_HOLD after): sided at stages 1-2, so only the driver's blinker
    that side is one; bare at stage 3."""
    side = self.changing or (self.change_side if self.now < self.change_hold_until else None)
    want = "" if side is None else ("laneChange" if self.stage >= 3 else f"laneChange:{side}")
    if want != self.desire:
      self.desire = want
      self.desires.append(want)

  def _lanes_ui(self, m: Move, need: dict | None) -> tuple[list[dict] | None, float]:
    """navInstruction.lanes for the next move with lanes while its guidance is active (within its lanes' banner
    distance, or a request open), the car's lane marked where the model counts as many; and how far ahead to be in an
    active lane by (0 when in one)."""
    lane_distance = max(need["deadline"], 0.0) if need is not None and need["changes"] > 0 else 0.0
    if not (self.request is not None or m.along - self.along <= m.ui_from):
      return None, lane_distance
    cur = None
    if self.lane is not None and self.lane[1] == len(m.lanes):
      cur = self.lane[0]
    out = []
    for j, lane in enumerate(m.lanes):
      dirs = [DIRECTIONS.get(x, "none") for x in lane.get("indications", [])] or ["none"]
      out.append({"directions": dirs, "active": bool(lane.get("valid")),
                  "activeDirection": DIRECTIONS.get(lane.get("valid_indication", ""), "none") if lane.get("valid") else "none",
                  "current": j == cur})
    return out, lane_distance

  def _open_request(self, kind: str, side: str, reason: str, distance: float, changes: int, target: float, why: str):
    if self.request is not None:
      self._close_request("replaced")
    self.request_id += 1
    self.request = {"kind": kind, "side": side, "reason": reason, "distance": round(max(distance, 0.0), 1),
                    "laneChanges": int(changes), "stage": int(self.stage), "id": self.request_id, "approved": False}
    self.request_meta = {"t": round(self.now, 2), "along": self.along, "lane": list(self.lane) if self.lane else None,
                         "v": round(self.v, 1), "target": round(target, 1), "why": why, "approved_t": None,
                         "move": round(self.along + target, 1)}

  def _request_now(self) -> dict | None:
    """The request as it stands: its distance counted down as the car drives on."""
    if self.request is None:
      return None
    left = self.request["distance"] - (self.along - self.request_meta["along"])
    return {**self.request, "distance": round(max(left, 0.0), 1)}

  def _close_request(self, outcome: str):
    """The request over, logged as planner.py logs its own (navRequest): done, declined, gave_up (withdrawn, too slow),
    timeout, turn (the turn's signal took over), replaced."""
    r, m = self.request, self.request_meta
    if r is None:
      return
    approved = m.get("approved_t")
    event = {"event": "navRequest", **{k: r[k] for k in ("id", "kind", "side", "reason", "laneChanges", "stage")},
             "t": m["t"], "lane": m["lane"], "lane_src": "model", "v": m["v"], "distance": r["distance"],
             "target": m["target"], "why": m["why"], "approved_after": None if approved is None else round(approved - m["t"], 2),
             "driven": round(self.along - m["along"], 1), "secs": round(self.now - m["t"], 2), "outcome": outcome,
             "lane_after": list(self.lane) if self.lane else None}
    self.events.append(event)
    self.request_log.append(event)
    self.request, self.request_meta = None, {}


def enabled(params=None) -> bool:
  """The switch: NAVD_LANE_GUIDANCE=1, else the NavLaneGuidance param."""
  env = os.getenv("NAVD_LANE_GUIDANCE")
  if env is not None:
    return env == "1"
  try:
    return bool(params is not None and params.get_bool("NavLaneGuidance"))
  except Exception:
    return False


def tune_from_env() -> Tune:
  """NAVD_LANE_TUNE: a JSON object of Tune overrides."""
  try:
    return Tune.of(json.loads(os.getenv("NAVD_LANE_TUNE", "{}")))
  except (ValueError, TypeError):
    return Tune()
