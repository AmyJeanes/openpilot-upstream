#!/usr/bin/env python3
"""In-game trials of nav's lane changes against the blind-spot monitor: each trip (an e2e trip name, or
'x,y,z,heading[,lane]>dx,dy') is placed in traffic, given its destination, engaged and driven by openpilot with nav until
it arrives, disengages, gets stuck, jams (stopped behind a stopped vehicle, where the test driver doesn't press the gas)
or times out. Every lane change nav asks for is measured, one JSON line each ("change"), then one for the trip ("trip":
contacts are separate touches after engaging, contact_frames the game's frames in contact).

  python -m openpilot.tools.sim.bridge.gta5.bsm_trial FX1 "799.0,-1752.5,28.9,264,1>814.2,41.7" --out bsm.jsonl \\
    [--traffic vehicles:1.5,parked:1,peds:1] [--timeout 420] [--reps 2]
  python -m openpilot.tools.sim.bridge.gta5.bsm_trial --summary bsm.jsonl [...]
  python -m openpilot.tools.sim.bridge.gta5.bsm_trial --recheck bsm.jsonl --out bsm_rechecked.jsonl
                                                       the changes measured again from the bridge's state log

Needs the bridge (GTA5_DEBUG, for nav's lines in its log), openpilot and the game in a car, as e2e.py, but never restarts
any of them: a stalled service ends the run. openpilot's side is its own messages (OPENPILOT_PREFIX): carState's blinkers
and blind-spot flags, modelV2's laneChangeState. The game's side is the bridge's state log: the car's pose, the vehicles
around it ("nearby") and its contact frames; the car's lane is read from the map's lanes (the bridge's GTA5_MAP: the
svc.sh map dir, ~/gta5test/map_dir), which also says whether a change moved the car over at all.

Each change, from nav's "lane change <side>, <why>" line:
- what it's for (lane ending, a turn, fork, way straight on, oncoming, bay) and how far; warmup when asked within WARMUP
  s of engaging, as the car still pulls away from where it was placed (a freeway at a standstill isn't driving);
- as the blinker came on: the blind-spot flag that way (flag_at_blinker) and a vehicle truly alongside (beside_at_blinker:
  its body overlapping ours along, within BESIDE_OUT m out from our side);
- wait_s from the blinker to openpilot's laneChangeStarting, held_s of it with the flag set;
- whether it started into an occupied side: the flag set at the start (started_flagged, the failure the monitor exists to
  prevent), a vehicle truly alongside (started_beside), and the vehicle closing from behind in that lane (rear_at_start:
  its gap, closing speed and time to us, also past the monitor's closing reach);
- from the start to LAND s after laneChangeStarting ends: the least sideways clearance alongside (min_side_gap), the least
  gap and time to a vehicle closing from behind in the lane (min_rear_gap, min_ttc), the hardest a vehicle there braked
  (max_brake), and contacts (separate touches, CONTACT_GAP apart); squeezed if any passes SQUEEZE_*;
- the outcome: done (the map's lanes have the car a lane over that way while the road keeps its lanes; landed: the
  kind of lane it ends up in, own or oncoming: the map's lanes counted from that side), partial (MOVED_M over without reaching the next lane),
  not_taken (laneChangeStarting came and went without the car moving over: this fork's desire helper has no
  laneChangeFinishing, so it leaves laneChangeStarting the same way whether the model changed lanes or not, and the driver
  then cancels the blinker), given_up (nav's reason), ended (the blinker off unstarted), superseded, or open at the
  trip's end; nav's lowest speed cap while it waited (merge slowing).
"""
import argparse
import datetime
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict, deque

os.environ.setdefault("OPENPILOT_PREFIX", "gta5")

from openpilot.tools.sim.bridge.gta5.e2e import BRIDGE_LOG, MAP_VIEW, STATE_LOG, Tail, ai_off, cmd, get_json, post_json
from openpilot.tools.sim.bridge.gta5.gta5_blindspot import footprint
from openpilot.tools.sim.bridge.gta5.gta5_cmd import find_trip

ENGAGEABLE_WAIT = 30.0  # s
STALE = 5.0  # s without game state or openpilot messages: the run ends
STOPPED = 0.3  # m/s
GAS_AFTER = 15.0  # s stopped: the test driver presses the gas (the model won't pull away from a stop)
GAS_CLEAR = 15.0  # m: not with a vehicle this near ahead, which the car would be pushed into (a queue)
GAS_TRIES = 3
STUCK_AFTER = 60.0  # s stopped after the last press, or behind a stopped vehicle (jam)
ARRIVED_WITHIN = 40.0  # m of the destination, disengaged
WARMUP = 15.0  # s after engaging
CHANGE_MAX = 25.0  # s a change is followed after it starts
LAND = 3.0  # s after laneChangeStarting ends that the lane it landed in is read, and contacts still count
STATES_KEPT = 120.0  # s of game states kept for measuring changes
BESIDE_OUT = 6.0  # m out from our side: the lane beside (GTA's are 5.5 m), not the one past it
BESIDE_ALONG = (1.0, 0.5)  # m behind our rear, ahead of our front, that a body overlapping counts as alongside
REAR_WITHIN = 30.0  # m behind our rear bumper, in the lane, for the rear gap
CLOSER_TTC = 3.0  # s: a vehicle closing from behind this soon at the start is reported (the monitor's closing_time)
SQUEEZE_GAP = 0.5  # m sideways clearance alongside
SQUEEZE_TTC = 1.5  # s to a vehicle closing from behind in the lane
SQUEEZE_BRAKE = 3.0  # m/s^2 a vehicle there braked
BRAKE_WINDOW = 0.5  # s
TRACK_NEAR = 2.0  # m from where a vehicle was predicted to be, for following it
TRACK_JUMP = 5.0  # m/s change between two readings: not the same vehicle
CONTACT_GAP = 1.0  # s between contact frames that count as separate touches
MOVED_LANES = 1  # lanes over that way on the map
MOVED_M = 2.5  # m over that way across the road (the map's offset from its line) without changing lanes: partial
CHANGE_LINE = re.compile(r"nav: lane change (left|right), (.*)")
GIVEN_UP = re.compile(r"nav: lane change (left|right) given up, (.*)")
HELD = re.compile(r"nav: lane change (left|right) held by the blind spot")
FOR_WHAT = re.compile(r"for the (.+?) in (-?\d+) m")
OFF, PRE, STARTING = 0, 1, 2  # log.LaneChangeState; this fork's desire helper never sets laneChangeFinishing


def parse_spec(spec: str) -> tuple[list[float], int | None, tuple[float, float]]:
  a, b = spec.split(">")
  start = [float(v) for v in a.split(",")]
  dx, dy = (float(v) for v in b.split(","))
  return start[:4], (int(start[4]) if len(start) > 4 else None), (dx, dy)


def kind_of(why: str) -> tuple[str, float | None]:
  m = FOR_WHAT.search(why)
  if m:
    return m.group(1), float(m.group(2))
  return ("oncoming" if "oncoming" in why else "bay" if "bay" in why else why), None


def side_vehicles(nearby: dict, side: str) -> list[tuple[float, float, list]]:
  """The vehicles on `side` going our way: (sideways gap from our side, m; along gap, m, negative behind our rear,
  positive ahead of our front, 0 overlapping; the row)."""
  mnx, mxx, mny, mxy = nearby.get("dims") or (-1.0, 1.0, -2.4, 2.4)
  out = []
  for v in nearby.get("v") or []:
    if abs(v[2]) > 60.0:
      continue  # oncoming or crossing
    x0, x1, y0, y1 = footprint(v)
    gap = mnx - x1 if side == "left" else x0 - mxx
    if gap > BESIDE_OUT or gap < -(mxx - mnx):
      continue
    along = 0.0 if y1 >= mny and y0 <= mxy else (y1 - mny if y1 < mny else y0 - mxy)
    out.append((gap, along, v))
  return out


def beside(nearby: dict | None, side: str) -> bool:
  return bool(nearby) and any(-BESIDE_ALONG[0] <= along <= BESIDE_ALONG[1] and gap > -1.0 for gap, along, _ in side_vehicles(nearby, side))


def rear_closer(nearby: dict | None, side: str, v_ego: float) -> dict | None:
  """The vehicle behind in the lane that way due soonest: its gap (m), closing speed (m/s) and time to us (s)."""
  best = None
  for gap, along, v in side_vehicles(nearby or {}, side):
    closing = v[4] - v_ego
    if along < 0 and gap > -1.0 and closing > 0.1:
      ttc = -along / closing
      if best is None or ttc < best["ttc"]:
        best = {"gap": round(-along, 1), "closing": round(closing, 1), "ttc": round(ttc, 2)}
  return best


class Tracks:
  """Vehicles around the car followed from reading to reading by where they are in the world, predicted on by their
  velocity, for their braking. A reading repeated (the plugin reports them every 0.1 s, the state comes at 20 Hz) is
  skipped, as its positions are relative to where the car was."""
  def __init__(self):
    self.tracks: list[dict] = []  # {"p": (x, y), "vel": (vx, vy), "t", "hist": deque[(t, speed)]}
    self.last = None

  def update(self, t: float, pos, heading: float, nearby: dict | None) -> list[tuple[dict, list]]:
    rows = (nearby or {}).get("v") or []
    if rows == self.last:
      return []
    self.last = rows
    h = math.radians(heading)
    right, fwd = (math.cos(h), math.sin(h)), (-math.sin(h), math.cos(h))
    seen: list[tuple[dict, list]] = []
    for v in rows:
      p = (pos[0] + v[0] * right[0] + v[1] * fwd[0], pos[1] + v[0] * right[1] + v[1] * fwd[1])
      vel = (v[3] * right[0] + v[4] * fwd[0], v[3] * right[1] + v[4] * fwd[1])
      speed = math.hypot(v[3], v[4])

      def miss(tr, p=p):
        dt = t - tr["t"]
        return math.dist((tr["p"][0] + tr["vel"][0] * dt, tr["p"][1] + tr["vel"][1] * dt), p)
      free = [tr for tr in self.tracks if not any(tr is s for s, _ in seen)]
      best = min(free, key=miss, default=None)
      if best is None or miss(best) > TRACK_NEAR or abs(best["hist"][-1][1] - speed) > TRACK_JUMP:
        best = {"hist": deque(maxlen=40)}
        self.tracks.append(best)
      best.update(p=p, vel=vel, t=t)
      best["hist"].append((t, speed))
      seen.append((best, v))
    self.tracks = [tr for tr, _ in seen]
    return seen

  @staticmethod
  def braking(tr: dict) -> float:
    """m/s^2 it slowed over the last BRAKE_WINDOW s (0 if not, or not followed that long)."""
    hist = tr["hist"]
    t1, v1 = hist[-1]
    old = [(t, v) for t, v in hist if t1 - t >= BRAKE_WINDOW]
    if not old:
      return 0.0
    t0, v0 = old[-1]
    return max(0.0, (v0 - v1) / (t1 - t0))


def map_dir() -> str:
  """The map the bridge plans on: GTA5_MAP, else svc.sh's."""
  if os.getenv("GTA5_MAP"):
    return os.environ["GTA5_MAP"]
  try:
    with open(os.path.expanduser("~/gta5test/map_dir")) as f:
      return f.read().strip()
  except OSError:
    return os.path.expanduser("~/gta5map_lanes")


class Lanes:
  """The car's lane by the map's lanes (lane_match.py): (lane from the left, lanes, kind, m right of the road's line),
  None off them."""
  def __init__(self):
    from openpilot.tools.sim.bridge.gta5.e2e import LaneMap
    self.map = LaneMap.load(map_dir())
    if self.map is None:
      print(f"bsm: no lane-tagged map in {map_dir()}: changes can't be told done from not taken", flush=True)

  def read(self, s: dict) -> tuple[int, int, str, float] | None:
    if self.map is None or "pos" not in s:
      return None
    x, y, z = s["pos"][:3]
    r = self.map.matcher.match(x, y, math.radians(s["heading"] + 90.0), z)
    return None if r is None else (r.lane, r.lanes, r.kind, r.right)


def contact_events(states: list[tuple[float, dict]]) -> tuple[int, int]:
  """Separate touches (contact frames CONTACT_GAP apart) and contact frames, over the states."""
  events, frames, last, prev = 0, 0, -math.inf, None
  for m, s in states:
    c = s.get("collisions")
    if c is None:
      continue
    if prev is not None and c > prev:
      frames += c - prev
      if m - last > CONTACT_GAP:
        events += 1
      last = m
    prev = c
  return events, frames


def assess(r: dict, states: list[tuple[float, dict]], lanes: Lanes | None):
  """The game's side of a change (module docstring), from the states around it and its times (monotonic): mono_blinker,
  mono_start and mono_end (when laneChangeStarting ended), as recorded live or found again by --recheck."""
  side, tb, ts, te = r["side"], r.get("mono_blinker"), r.get("mono_start"), r.get("mono_end")

  def at(m):
    near = [ms for ms in states if abs(ms[0] - m) < 0.3]
    return min(near, key=lambda ms: abs(ms[0] - m))[1] if near else None
  if tb is not None and (s := at(tb)) is not None:
    r["beside_at_blinker"] = beside(s.get("nearby"), side)
  if ts is None:
    return
  s = at(ts)
  if s is not None:
    r["started_beside"] = beside(s.get("nearby"), side)
    r["rear_at_start"] = rear_closer(s.get("nearby"), side, s.get("vEgo", 0.0))
    r["start_v"] = round(s.get("vEgo", 0.0), 1)
  end = (te if te is not None else ts + CHANGE_MAX) + LAND
  during = [(m, s) for m, s in states if ts <= m <= end]
  tracks = Tracks()
  for m, s in during:
    nearby, v_ego = s.get("nearby") or {}, s.get("vEgo", 0.0)
    seen = tracks.update(m, s["pos"], s["heading"], nearby)
    for gap, along, v in side_vehicles(nearby, side):
      if along == 0.0:
        r["min_side_gap"] = round(gap if r.get("min_side_gap") is None else min(gap, r["min_side_gap"]), 2)
      elif along < 0 and -along <= REAR_WITHIN and gap > -1.0:
        r["min_rear_gap"] = round(-along if r.get("min_rear_gap") is None else min(-along, r["min_rear_gap"]), 1)
        closing = v[4] - v_ego
        if closing > 0.1:
          r["min_ttc"] = round(-along / closing if r.get("min_ttc") is None else min(-along / closing, r["min_ttc"]), 2)
    lane = {id(v) for _, along, v in side_vehicles(nearby, side) if -REAR_WITHIN <= along <= 0.0}
    for tr, v in seen:
      if id(v) in lane:
        r["max_brake"] = round(max(r.get("max_brake") or 0.0, Tracks.braking(tr)), 1)
  r["contacts"], r["contact_frames"] = contact_events([(m, s) for m, s in during if m >= ts])
  r["squeezed"] = bool((r.get("min_side_gap") is not None and r["min_side_gap"] < SQUEEZE_GAP)
                       or (r.get("min_ttc") is not None and r["min_ttc"] < SQUEEZE_TTC)
                       or (r.get("max_brake") or 0.0) >= SQUEEZE_BRAKE or r["contacts"] > 0)
  if lanes is None or lanes.map is None or te is None:
    r["outcome"] = r["outcome"] or "unknown"
    return
  before = [x for m, s in states if ts - 2.0 <= m <= ts and (x := lanes.read(s)) and x[1] > 0]
  through = [x for m, s in states if ts <= m <= te + LAND and (x := lanes.read(s)) and x[1] > 0]
  if not before or not through:
    r["outcome"] = r["outcome"] or "unknown"
    r["detail"] = r.get("detail") or "no map lane before or after"
    return
  (l0, n0, _, x0), (l1, n1, k1, _) = before[-1], through[-1]
  sign = 1 if side == "right" else -1
  # lanes counted from the side it changes to, which lanes ending on the other side (or the branch not taken at a
  # split) don't renumber
  def from_side(x):
    return x[0] if side == "left" else x[1] - 1 - x[0]
  # by where it ends up (the last second's readings, as one stray reading across a ramp's link can say anything)
  last = sorted(from_side(x) for m, s in states if te + LAND - 1.0 <= m <= te + LAND and (x := lanes.read(s)) and x[1] > 0)
  over = from_side(before[-1]) - (last[len(last) // 2] if last else from_side(through[-1]))
  shift = max((sign * (x[3] - x0) for x in through if x[1] == n0), default=0.0)  # while the road's line stays put
  r["lane_before"], r["lane_after"], r["lanes_over"], r["shift_m"], r["landed"] = [l0, n0], [l1, n1], over, round(shift, 1), k1
  if r.get("outcome") in (None, "done", "partial", "not_taken", "unknown"):
    r["outcome"] = "done" if over >= MOVED_LANES else "partial" if shift >= MOVED_M else "not_taken"


class Change:
  """openpilot's side of a change, followed live: its times, the flags and the wait."""
  def __init__(self, trip: str, t: float, mono: float, side: str, why: str, state: dict):
    self.side = side
    kind, dist = kind_of(why)
    self.rec = {"kind": "change", "trip": trip, "side": side, "why": why, "what": kind, "dist": dist,
                "t": round(t, 1), "warmup": t < WARMUP, "mono_ask": round(mono, 2),
                "pos": [round(v, 1) for v in state.get("pos", [0, 0])[:2]], "v_ask": round(state.get("vEgo", 0.0), 1),
                "street": state.get("street"), "blinker_after": None, "mono_blinker": None, "flag_at_blinker": None,
                "beside_at_blinker": None, "wait_s": None, "held_s": 0.0, "held_logged": False, "started": False,
                "mono_start": None, "mono_end": None, "starting_s": None, "started_flagged": None, "flag_before_start": None,
                "started_beside": None, "rear_at_start": None, "start_v": None, "min_side_gap": None, "min_rear_gap": None,
                "min_ttc": None, "max_brake": 0.0, "contacts": 0, "contact_frames": 0, "squeezed": False,
                "lane_before": None, "lane_after": None, "lanes_over": None, "shift_m": None, "landed": None, "outcome": None, "detail": None,
                "min_cap": None, "cap_reason": None, "v_min_waiting": None}
    self.prev_state = OFF
    self.prev_flag = False
    self.last = mono
    self.closed_at: float | None = None  # mono when it closed; assessed once LAND s on

  def step(self, mono: float, cs, lc: int, state: dict, cap: tuple[float, str] | None):
    r, side = self.rec, self.side
    dt, self.last = mono - self.last, mono
    blinker = cs.leftBlinker if side == "left" else cs.rightBlinker
    flag = bool(cs.leftBlindspot if side == "left" else cs.rightBlindspot)
    if r["mono_blinker"] is None and blinker:
      r["mono_blinker"] = round(mono, 2)
      r["blinker_after"] = round(mono - r["mono_ask"], 2)
      r["flag_at_blinker"] = flag
    if r["mono_start"] is None:
      if cap is not None and cap[0] > 0 and (r["min_cap"] is None or cap[0] < r["min_cap"]):
        r["min_cap"], r["cap_reason"] = round(cap[0], 2), cap[1]
      if r["mono_blinker"] is not None:
        v = state.get("vEgo", 0.0)
        r["v_min_waiting"] = round(v if r["v_min_waiting"] is None else min(v, r["v_min_waiting"]), 2)
        if flag and lc == PRE:
          r["held_s"] = round(r["held_s"] + dt, 2)
      if lc == STARTING and self.prev_state == PRE:
        r["started"], r["mono_start"] = True, round(mono, 2)
        r["wait_s"] = round(mono - (r["mono_blinker"] or mono), 2)
        r["started_flagged"], r["flag_before_start"] = flag, self.prev_flag
    elif r["mono_end"] is None and (lc != STARTING or mono - r["mono_start"] > CHANGE_MAX):
      r["mono_end"] = round(mono, 2)
      r["starting_s"] = round(mono - r["mono_start"], 2)
      self.close(mono)
    if r["mono_blinker"] is not None and not blinker and r["mono_start"] is None:
      self.close(mono, "ended", "the blinker went off before it started")
    self.prev_state, self.prev_flag = lc, flag

  def close(self, mono: float, outcome: str | None = None, detail: str | None = None):
    if self.closed_at is not None:
      return
    self.closed_at = mono
    r = self.rec
    r["outcome"], r["detail"] = r["outcome"] or outcome, r["detail"] or detail
    if r["wait_s"] is None and r["mono_blinker"] is not None:
      r["wait_s"] = round(mono - r["mono_blinker"], 2)


class Run:
  def __init__(self, out_path: str):
    from openpilot.cereal import messaging
    self.sm = messaging.SubMaster(["carState", "modelV2", "selfdriveState", "navSpeed"], poll="modelV2")
    self.states_tail, self.log = Tail(STATE_LOG), Tail(BRIDGE_LOG)
    self.states: deque[tuple[float, dict]] = deque()
    self.state: dict = {}
    self.state_t = 0.0
    self.nav_lines: list[str] = []
    self.contacts, self.contact_frames = 0, 0  # since reset, as contact_events counts them
    self.last_collisions: int | None = None
    self.last_contact = -math.inf
    self.lanes = Lanes()
    self.out = open(out_path, "a", buffering=1)

  def update(self, wait_ms: int = 100):
    self.sm.update(wait_ms)
    now = time.monotonic()
    for line in self.states_tail.lines():
      if '"state"' not in line:
        continue
      try:
        d = json.loads(line)
        s, m = d["state"], d["mono"]
      except (ValueError, KeyError):
        continue
      if s.get("inVehicle"):
        self.state, self.state_t = s, now
        self.states.append((m, s))
        c = s.get("collisions")
        if c is not None and self.last_collisions is not None and c > self.last_collisions:
          self.contact_frames += c - self.last_collisions
          self.contacts += m - self.last_contact > CONTACT_GAP
          self.last_contact = m
        self.last_collisions = c if c is not None else self.last_collisions
    while self.states and self.states[0][0] < now - STATES_KEPT:
      self.states.popleft()
    self.nav_lines += [ln.strip() for ln in self.log.lines() if ln.startswith("nav:")]

  def take(self) -> list[str]:
    lines, self.nav_lines = self.nav_lines, []
    return lines

  @property
  def engaged(self) -> bool:
    return bool(self.sm["selfdriveState"].enabled)

  def wait(self, secs: float, until=None) -> bool:
    end = time.monotonic() + secs
    while time.monotonic() < end:
      self.update()
      if until is not None and until():
        return True
    return False

  def set_engaged(self, on: bool) -> bool:
    for _ in range(3):
      if self.engaged == on:
        return True
      cmd("engage")
      if self.wait(3, lambda: self.engaged == on):
        return True
    return self.engaged == on

  def stale(self) -> str | None:
    now = time.monotonic()
    if now - self.state_t > STALE:
      return "no game state from the bridge"
    if now - max(self.sm.recv_time["modelV2"], 0) > STALE:
      return "no modelV2 from openpilot"
    return None

  def setup(self, spec: str, traffic: dict) -> str | None:
    (x, y, z, h), lane, (dx, dy) = parse_spec(spec)
    if self.engaged and not self.set_engaged(False):
      return "could not disengage"
    ai_off()
    post_json(f"{MAP_VIEW}/destination", {})
    cmd("waypoint", off=1)
    cmd("world", hour=12, weather="EXTRASUNNY", rain=-1, freeze=1)
    cmd("traffic", on=1, **traffic)
    cmd("lead", remove=1)
    kw = {"x": x, "y": y, "z": z, "heading": h, "fix": 1}
    if lane is not None:
      kw["lane"] = lane
    cmd("setup", **kw)
    t = time.monotonic()
    placed = self.wait(20, lambda: time.monotonic() - t > 4.5 and self.state_t > t + 4 and self.state.get("vEgo", 9) < 0.5
                       and math.hypot(self.state["pos"][0] - x, self.state["pos"][1] - y) < 40)
    if not placed:
      return f"not placed (at {self.state.get('pos')})"
    if not self.wait(ENGAGEABLE_WAIT, lambda: self.sm["selfdriveState"].engageable):
      ss = self.sm["selfdriveState"]
      return f"not engageable: {ss.alertText1} {ss.alertText2}".strip()
    routes0 = ((get_json(f"{MAP_VIEW}/state", 1.0).get("nav") or {}).get("routes", 0))
    post_json(f"{MAP_VIEW}/destination", {"x": dx, "y": dy})

    def routed():
      try:
        return (get_json(f"{MAP_VIEW}/state", 1.0).get("nav") or {}).get("routes", 0) > routes0
      except (OSError, ValueError):
        return False
    if not self.wait(10, routed):
      return "the bridge made no route"
    if not self.set_engaged(True):
      ss = self.sm["selfdriveState"]
      return f"not engaged: {ss.alertText1} {ss.alertText2}".strip()
    return None

  def trip(self, name: str, spec: str, traffic: dict, timeout: float) -> dict:
    rec = {"kind": "trip", "trip": name, "spec": spec, "traffic": traffic, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    problem = self.setup(spec, traffic)
    if problem:
      return {**rec, "outcome": "setup", "detail": problem}
    _, _, (dx, dy) = parse_spec(spec)
    cmd("gas", secs=0.5)  # the model won't pull away from a stop
    self.take()
    t0 = time.monotonic()
    self.contacts, self.contact_frames = 0, 0  # not those of placing the car
    rec["mono_engaged"] = round(t0, 2)
    changes: list[Change] = []
    open_: Change | None = None
    stopped_t, gas, gas_t = None, 0, 0.0
    outcome, detail, max_v = None, "", 0.0
    while outcome is None:
      self.update()
      now = time.monotonic()
      t = now - t0
      bad = self.stale()
      if bad:
        outcome, detail = "infra", bad
        break
      s, cs = self.state, self.sm["carState"]
      lc = self.sm["modelV2"].meta.laneChangeState.raw
      ns = self.sm["navSpeed"] if self.sm.seen["navSpeed"] else None
      cap = (ns.speedCap, str(ns.reason)) if ns is not None and now - self.sm.recv_time["navSpeed"] < 1.0 else None
      max_v = max(max_v, s.get("vEgo", 0.0))
      for line in self.take():
        if m := GIVEN_UP.match(line):
          if open_ is not None and open_.side == m.group(1):
            open_.close(now, "given_up", m.group(2))
        elif m := HELD.match(line):
          if open_ is not None:
            open_.rec["held_logged"] = True
        elif m := CHANGE_LINE.match(line):
          if open_ is not None:
            open_.close(now, "superseded", "nav asked for another")
          open_ = Change(name, t, now, m.group(1), m.group(2), s)
          changes.append(open_)
          print(f"bsm: {name} t={t:.0f} lane change {m.group(1)}, {m.group(2)}", flush=True)
      if open_ is not None and open_.closed_at is None:
        open_.step(now, cs, lc, s, cap)
      self._emit_due(changes, now)
      v = s.get("vEgo", 0.0)
      left = math.hypot(s["pos"][0] - dx, s["pos"][1] - dy)
      if not self.engaged:
        outcome = "arrived" if left < ARRIVED_WITHIN else "disengaged"
        detail = "" if outcome == "arrived" else f"{self.sm['selfdriveState'].alertText1} ({left:.0f} m left)"
        break
      if t > timeout:
        outcome, detail = "timeout", f"{left:.0f} m left"
        break
      ahead = s.get("vehicleAhead", 0.0)
      if v < STOPPED:
        stopped_t = stopped_t or now
        blocked = 0.0 < ahead < GAS_CLEAR
        if blocked and now - stopped_t > STUCK_AFTER:
          outcome, detail = "jam", f"stopped {now - stopped_t:.0f} s behind a vehicle {ahead:.0f} m ahead, {left:.0f} m left"
          break
        if not blocked and gas < GAS_TRIES and now - stopped_t > GAS_AFTER and now - gas_t > GAS_AFTER:
          cmd("gas", secs=0.5)
          gas, gas_t = gas + 1, now
        if gas >= GAS_TRIES and now - gas_t > STUCK_AFTER:
          outcome, detail = "stuck", f"{left:.0f} m left"
          break
      else:
        stopped_t = None
    end = time.monotonic()
    for c in changes:
      c.close(end, "open", f"trip {outcome}")
    self.wait(LAND if outcome != "infra" else 0.0)
    self._emit_due(changes, math.inf)
    self.set_engaged(False)
    rs = [c.rec for c in changes]
    return {**rec, "outcome": outcome, "detail": detail, "secs": round(end - t0), "max_v": round(max_v, 1),
            "contacts": self.contacts, "contact_frames": self.contact_frames, "gas": gas, **trip_counts(rs)}

  def _emit_due(self, changes: list[Change], now: float):
    for c in changes:
      if c.closed_at is not None and not c.rec.get("_emitted") and now - c.closed_at >= LAND:
        assess(c.rec, list(self.states), self.lanes)
        c.rec["_emitted"] = True
        emit(self.out, c.rec)


def trip_counts(rs: list[dict]) -> dict:
  return {"changes": len(rs), "changes_started": sum(r["started"] for r in rs), "done": sum(r["outcome"] == "done" for r in rs),
          "not_taken": sum(r["outcome"] in ("not_taken", "partial") for r in rs), "started_flagged": sum(bool(r["started_flagged"]) for r in rs),
          "started_beside": sum(bool(r["started_beside"]) for r in rs), "given_up": sum(r["outcome"] == "given_up" for r in rs),
          "squeezed": sum(bool(r["squeezed"]) for r in rs), "held": sum(r["held_s"] > 0 for r in rs)}


def emit(out, rec: dict):
  r = {k: v for k, v in rec.items() if not k.startswith("_")}
  out.write(json.dumps(r) + "\n")
  flag = " STARTED INTO A FLAGGED SIDE" if r["started_flagged"] else ""
  print(f"bsm: {r['trip']} {r['side']} {r['what']}: {r['outcome']} wait {r['wait_s']} held {r['held_s']} " +
        f"lanes {r['lane_before']}->{r['lane_after']} squeezed {r['squeezed']}{flag}", flush=True)


def read_rows(paths: list[str]) -> list[dict]:
  rows = []
  for p in paths:
    with open(p) as f:
      rows += [json.loads(line) for line in f if line.strip()]
  return rows


def states_between(a: float, b: float) -> list[tuple[float, dict]]:
  """The bridge's states from monotonic a to b, found in its (long) log by bisecting on its times."""
  size = os.path.getsize(STATE_LOG)
  out = []
  with open(STATE_LOG, "rb") as f:
    def mono_at(pos):
      f.seek(pos)
      f.readline()
      for _ in range(100):
        line = f.readline()
        if not line:
          return None
        try:
          return json.loads(line)["mono"]
        except (ValueError, KeyError):
          continue
      return None
    lo, hi = 0, size
    while hi - lo > 1 << 16:
      mid = (lo + hi) // 2
      m = mono_at(mid)
      if m is None or m > a:
        hi = mid
      else:
        lo = mid
    f.seek(lo)
    f.readline()
    for line in f:
      try:
        d = json.loads(line)
        m = d["mono"]
      except (ValueError, KeyError):
        continue
      if m > b:
        break
      if m >= a and "state" in d and d["state"].get("inVehicle"):
        out.append((m, d["state"]))
  return out


def recheck(paths: list[str], out_path: str):
  """The changes of earlier results measured again from the bridge's state log (which must still hold them). Results
  from before the changes carried their times are placed by the blinker: the trip's indicator coming on in the log,
  matched to the changes' blinkers in order; a change's laneChangeStarting is then taken to end when the plugin's
  indicator goes off (the driver cancels it then)."""
  lanes = Lanes()
  wall_to_mono = time.time() - time.monotonic()  # noqa: TID251  the results date their trips by the wall clock
  trips, pending = [], []  # each trip's changes come before its own line
  for r in read_rows(paths):
    if r["kind"] == "change":
      pending.append(r)
    else:
      trips.append((r, pending))
      pending = []
  with open(out_path, "w", buffering=1) as out:
    for tr, ch in trips:
      a = datetime.datetime.strptime(tr["started"], "%Y-%m-%d %H:%M:%S").timestamp() - wall_to_mono
      states = states_between(a, a + 60.0 + tr.get("secs", 0) + LAND + 30.0)
      edges, prev = [], None  # (mono, side, on)
      for m, s in states:
        ind = s.get("indicator")
        if ind != prev:
          edges.append((m, ind))
        prev = ind
      t0 = tr.get("mono_engaged")
      first = next((c for c in ch if c.get("blinker_after") is not None), None)
      if t0 is None and first is not None:
        on = next((m for m, ind in edges if ind == first["side"]), None)
        t0 = None if on is None else on - first["t"] - first["blinker_after"]
      for c in ch:
        c = dict(c)
        if c.get("mono_blinker") is None and t0 is not None and c.get("blinker_after") is not None:
          want = t0 + c["t"] + c["blinker_after"]
          on = min((m for m, ind in edges if ind == c["side"] and abs(m - want) < 1.5), key=lambda m: abs(m - want), default=None)
          if on is not None:
            c["mono_blinker"] = round(on, 2)
            if c.get("started") and c.get("wait_s") is not None:
              c["mono_start"] = round(on + c["wait_s"], 2)
              off = next((m for m, ind in edges if m > c["mono_start"] and ind != c["side"]), None)
              c["mono_end"] = None if off is None else round(off, 2)
              c["starting_s"] = None if off is None else round(off - c["mono_start"], 2)
        c["warmup"] = c["t"] < WARMUP
        for k in ("min_side_gap", "min_rear_gap", "min_ttc", "lane_before", "lane_after", "lanes_over", "shift_m", "landed"):
          c[k] = None
        c["max_brake"], c["contacts"], c["contact_frames"] = 0.0, 0, 0
        if c.get("outcome") in ("aborted", "done", "unfinished"):
          c["outcome"], c["detail"] = None, None
        assess(c, states, lanes)
        c["rechecked"] = True
        emit(out, c)
      e0 = t0 if t0 is not None else a
      contacts, frames = contact_events([(m, s) for m, s in states if e0 <= m <= e0 + tr.get("secs", 0)])
      fixed = {**tr, "contacts": contacts, "contact_frames": frames, "rechecked": True}
      out.write(json.dumps(fixed) + "\n")


def summary(paths: list[str]):
  rows = read_rows(paths)
  trips = [r for r in rows if r["kind"] == "trip"]
  ch = [r for r in rows if r["kind"] == "change"]
  print(f"{len(trips)} trips: {dict(Counter(r['outcome'] for r in trips))}; contacts (separate touches) " +
        f"{sum(r.get('contacts', 0) for r in trips)}")
  for r in trips:
    if r["outcome"] not in ("arrived",):
      print(f"  {r['trip']}: {r['outcome']} {r.get('detail', '')}")
  warm = [r for r in ch if r.get("warmup")]
  ch = [r for r in ch if not r.get("warmup")]
  print(f"{len(ch) + len(warm)} lane changes asked for, {len(warm)} of them in the warmup (counted apart, listed below)")
  by = defaultdict(list)
  for r in ch:
    by[r["what"]].append(r)
  for what, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    started = [r for r in rs if r["started"]]
    waits = sorted(r["wait_s"] for r in started if r["wait_s"] is not None)
    print(f"  {what}: {len(rs)} asked, {len(started)} started; flagged at the blinker {sum(bool(r['flag_at_blinker']) for r in rs)}, " +
          f"beside {sum(bool(r['beside_at_blinker']) for r in rs)}; held {sum(r['held_s'] > 0 for r in rs)} " +
          f"(max {max((r['held_s'] for r in rs), default=0):.1f} s); wait median {waits[len(waits) // 2] if waits else None} " +
          f"max {waits[-1] if waits else None} s")
    print(f"    STARTED INTO A FLAGGED SIDE {sum(bool(r['started_flagged']) for r in rs)}, beside at start " +
          f"{sum(bool(r['started_beside']) for r in rs)}; squeezed {sum(bool(r['squeezed']) for r in rs)}, contacts " +
          f"{sum(r['contacts'] for r in rs)}; outcomes {dict(Counter(r['outcome'] for r in rs))}")
  for r in ch + warm:
    tag = "warmup " if r.get("warmup") else ""
    note = []
    if r["outcome"] in ("given_up", "not_taken", "partial", "ended", "unknown") or r.get("landed") not in (None, "own"):
      note.append(f"{r['outcome']} ({r.get('detail') or ''}) lanes {r.get('lane_before')}->{r.get('lane_after')} landed {r.get('landed')}")
    closer = r.get("rear_at_start")
    if closer and closer["ttc"] < CLOSER_TTC:
      note.append(f"closer behind at the start {closer}")
    if r["started_flagged"] or r["started_beside"] or r["squeezed"]:
      note.append(f"flagged {r['started_flagged']} beside {r['started_beside']} side_gap {r['min_side_gap']} rear_gap {r['min_rear_gap']} " +
                  f"ttc {r['min_ttc']} brake {r['max_brake']} contacts {r['contacts']}")
    if note:
      print(f"    {tag}{r['trip']} t={r['t']} {r['side']} {r['what']} at {r['pos']} v {r.get('start_v') or r['v_ask']}: " + "; ".join(note))


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("trips", nargs="*")
  ap.add_argument("--out", default=os.path.expanduser("~/gta5test/bsm_trial.jsonl"))
  ap.add_argument("--traffic", default="vehicles:1.5,parked:1,peds:1", help="density multipliers, key:value,...")
  ap.add_argument("--timeout", type=float, default=420.0, help="s per trip")
  ap.add_argument("--reps", type=int, default=1)
  ap.add_argument("--summary", nargs="+", help="results files to sum up instead")
  ap.add_argument("--recheck", nargs="+", help="results files to measure again from the state log, to --out")
  # a spec starting with a minus would be taken for an option: parsed as a stand-in, put back in its place
  argv = [f"@spec{i}" if ">" in x and x.startswith("-") else x for i, x in enumerate(sys.argv[1:])]
  args = ap.parse_intermixed_args(argv)
  args.trips = [sys.argv[1:][int(x[5:])] if x.startswith("@spec") else x for x in args.trips]
  if args.summary:
    summary(args.summary)
    return
  if args.recheck:
    recheck(args.recheck, args.out)
    return
  traffic = {k: float(v) for k, _, v in (p.partition(":") for p in args.traffic.split(",") if p)}
  run = Run(args.out)
  for rep in range(args.reps):
    for name in args.trips:
      spec = find_trip(name)
      rec = run.trip(name, spec, traffic, args.timeout)
      rec["rep"] = rep
      run.out.write(json.dumps(rec) + "\n")
      print(f"bsm: trip {json.dumps(rec)}", flush=True)
      if rec["outcome"] == "infra":
        print("bsm: a service stalled: stopping (nothing is restarted)", flush=True)
        return


if __name__ == "__main__":
  main()
