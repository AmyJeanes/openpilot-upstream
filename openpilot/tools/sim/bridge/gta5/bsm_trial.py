#!/usr/bin/env python3
"""In-game trials of nav's lane changes against the blind-spot monitor: each trip (an e2e trip name, or
'x,y,z,heading[,lane]>dx,dy') is placed in traffic, given its destination, engaged and driven by openpilot with nav until
it arrives, disengages, gets stuck or times out. Every lane change nav asks for is measured, one JSON line each ("change"),
then one for the trip ("trip").

  python -m openpilot.tools.sim.bridge.gta5.bsm_trial FX1 "799.0,-1752.5,28.9,264,1>814.2,41.7" --out bsm.jsonl \\
    [--traffic vehicles:1.5,parked:1,peds:1] [--timeout 420] [--reps 2]
  python -m openpilot.tools.sim.bridge.gta5.bsm_trial --summary bsm.jsonl [...]

Needs the bridge (GTA5_DEBUG, for nav's lines in its log; GTA5_MAP), openpilot and the game in a car, as e2e.py, but
never restarts any of them: a stalled service ends the run. openpilot's side is its own messages (OPENPILOT_PREFIX):
carState's blinkers and blind-spot flags, modelV2's laneChangeState. The game's side is the bridge's state log: the
vehicles around the car ("nearby") and its contacts.

Each change, from nav's "lane change <side>, <why>" line: its kind (lane ending, turn, fork, way straight on, ...),
whether the blind spot that way was set (flag_at_blinker) and a vehicle truly alongside (beside_at_blinker: its body
overlapping ours along, within BESIDE_OUT m out from our side) as the blinker came on; how long openpilot waited in
preLaneChange (wait_s, held_s of it with the flag set); whether the change started, and whether it started into an
occupied side: the flag set as openpilot went to laneChangeStarting (started_flagged, the failure the monitor exists to
prevent) or a vehicle truly alongside then (started_beside). Through the change: the least sideways clearance to a
vehicle overlapping ours along on that side (min_side_gap), the least gap and time to a vehicle closing from behind in
that lane (min_rear_gap, min_ttc), the hardest a vehicle there braked (max_brake), contacts, and squeezed (any of those
past SQUEEZE_*). Its outcome: done, given_up (nav's reason), ended (the blinker off unstarted), aborted (started, then
back) or open at the trip's end; nav's lowest speed cap meanwhile (merge slowing).
"""
import argparse
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
GAS_TRIES = 3
STUCK_AFTER = 60.0  # s stopped after the last press
ARRIVED_WITHIN = 40.0  # m of the destination, disengaged
CHANGE_MAX = 25.0  # s a change is followed after it starts
BESIDE_OUT = 6.0  # m out from our side: the lane beside (GTA's are 5.5 m), not the one past it
BESIDE_ALONG = (1.0, 0.5)  # m behind our rear, ahead of our front, that a body overlapping counts as alongside
REAR_WITHIN = 30.0  # m behind our rear bumper, in the lane, for the rear gap
SQUEEZE_GAP = 0.5  # m sideways clearance alongside
SQUEEZE_TTC = 1.5  # s to a vehicle closing from behind in the lane
SQUEEZE_BRAKE = 3.0  # m/s^2 a vehicle there braked
TRACK_NEAR = 3.0  # m a vehicle moves between two states, for following it
CHANGE_LINE = re.compile(r"nav: lane change (left|right), (.*)")
GIVEN_UP = re.compile(r"nav: lane change (left|right) given up, (.*)")
HELD = re.compile(r"nav: lane change (left|right) held by the blind spot")
FOR_WHAT = re.compile(r"for the (.+?) in (-?\d+) m")
OFF, PRE, STARTING, FINISHING = 0, 1, 2, 3  # log.LaneChangeState


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


def side_vehicles(nearby: dict, side: str) -> list[tuple[float, float, float, list]]:
  """The vehicles on `side` going our way: (sideways gap from our side, m; along gap, m, negative behind our rear,
  positive ahead of our front, 0 overlapping; their footprint's rear in our frame, the row)."""
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
    out.append((gap, along, y0, v))
  return out


def beside(nearby: dict | None, side: str) -> bool:
  if not nearby:
    return False
  return any(-BESIDE_ALONG[0] <= along <= BESIDE_ALONG[1] and gap > -1.0 for gap, along, _, _ in side_vehicles(nearby, side))


class Tracks:
  """Vehicles around the car followed from state to state by where they are in the world, for their braking."""
  def __init__(self):
    self.tracks: list[dict] = []  # {"p": (x, y), "hist": deque[(t, speed)]}

  def update(self, t: float, pos, heading: float, nearby: dict | None) -> list[tuple[dict, list]]:
    h = math.radians(heading)
    right, fwd = (math.cos(h), math.sin(h)), (-math.sin(h), math.cos(h))
    seen = []
    for v in (nearby or {}).get("v") or []:
      p = (pos[0] + v[0] * right[0] + v[1] * fwd[0], pos[1] + v[0] * right[1] + v[1] * fwd[1])
      best = min(self.tracks, key=lambda tr: math.dist(tr["p"], p), default=None)
      if best is None or math.dist(best["p"], p) > TRACK_NEAR or any(best is s for s, _ in seen):
        best = {"p": p, "hist": deque(maxlen=40)}
        self.tracks.append(best)
      best["p"] = p
      best["hist"].append((t, math.hypot(v[3], v[4])))
      seen.append((best, v))
    self.tracks = [tr for tr, _ in seen]
    return seen

  @staticmethod
  def braking(tr: dict, window: float = 0.5) -> float:
    """m/s^2 it slowed over the last `window` s (0 if not)."""
    hist = tr["hist"]
    if len(hist) < 2:
      return 0.0
    t1, v1 = hist[-1]
    old = [(t, v) for t, v in hist if t1 - t >= window]
    if not old:
      return 0.0
    t0, v0 = old[-1]
    return max(0.0, (v0 - v1) / (t1 - t0))


class Change:
  def __init__(self, trip: str, t: float, side: str, why: str, state: dict):
    self.side, self.t_ask = side, t
    kind, dist = kind_of(why)
    self.rec = {"kind": "change", "trip": trip, "side": side, "why": why, "what": kind, "dist": dist,
                "t": round(t, 1), "pos": [round(v, 1) for v in state.get("pos", [0, 0])[:2]], "v_ask": round(state.get("vEgo", 0.0), 1),
                "street": state.get("street"), "blinker_after": None, "flag_at_blinker": None, "beside_at_blinker": None,
                "wait_s": None, "held_s": 0.0, "held_logged": False, "started": False, "started_flagged": None,
                "flag_before_start": None, "started_beside": None, "start_v": None, "nearest_at_start": None,
                "min_side_gap": None, "min_rear_gap": None, "min_ttc": None, "max_brake": 0.0, "contacts": 0,
                "squeezed": False, "outcome": None, "detail": None, "min_cap": None, "cap_reason": None, "v_min_waiting": None}
    self.blinker_t: float | None = None
    self.start_t: float | None = None
    self.prev_state = OFF
    self.prev_flag = False
    self.lane0 = state.get("lane")
    self.last_t = t
    self.done = False

  def step(self, t: float, cs, lc: int, state: dict, tracks: list, cap: tuple[float, str] | None, contacts: int):
    r, side = self.rec, self.side
    dt, self.last_t = t - self.last_t, t
    blinker = cs.leftBlinker if side == "left" else cs.rightBlinker
    flag = bool(cs.leftBlindspot if side == "left" else cs.rightBlindspot)
    nearby = state.get("nearby")
    if self.blinker_t is None and blinker:
      self.blinker_t = t
      r["blinker_after"] = round(t - self.t_ask, 2)
      r["flag_at_blinker"], r["beside_at_blinker"] = flag, beside(nearby, side)
    if self.start_t is None:
      if cap is not None and cap[0] > 0 and (r["min_cap"] is None or cap[0] < r["min_cap"]):
        r["min_cap"], r["cap_reason"] = round(cap[0], 2), cap[1]
      if self.blinker_t is not None:
        v = state.get("vEgo", 0.0)
        r["v_min_waiting"] = round(v if r["v_min_waiting"] is None else min(v, r["v_min_waiting"]), 2)
        if flag and lc == PRE:
          r["held_s"] = round(r["held_s"] + dt, 2)
      if lc == STARTING and self.prev_state == PRE:
        self.start_t = t
        r["started"] = True
        r["wait_s"] = round(t - (self.blinker_t or t), 2)
        r["started_flagged"], r["flag_before_start"] = flag, self.prev_flag
        r["started_beside"] = beside(nearby, side)
        r["start_v"] = round(state.get("vEgo", 0.0), 1)
        near = sorted(side_vehicles(nearby or {}, side), key=lambda g: abs(g[1]) + abs(g[0]))
        if near:
          gap, along, _, v = near[0]
          r["nearest_at_start"] = {"side_gap": round(gap, 1), "along": round(along, 1), "speed": round(math.hypot(v[3], v[4]), 1)}
    if self.start_t is not None and t - self.start_t <= CHANGE_MAX and lc in (STARTING, FINISHING):
      v_ego = state.get("vEgo", 0.0)
      for gap, along, _, v in side_vehicles(nearby or {}, side):
        if along == 0.0:
          r["min_side_gap"] = round(gap if r["min_side_gap"] is None else min(gap, r["min_side_gap"]), 2)
        elif along < 0 and -along <= REAR_WITHIN and gap > -1.0:
          r["min_rear_gap"] = round(-along if r["min_rear_gap"] is None else min(-along, r["min_rear_gap"]), 1)
          closing = v[4] - v_ego
          if closing > 0.1:
            ttc = -along / closing
            r["min_ttc"] = round(ttc if r["min_ttc"] is None else min(ttc, r["min_ttc"]), 2)
      lane = {id(v) for _, along, _, v in side_vehicles(nearby or {}, side) if -REAR_WITHIN <= along <= 0.0}
      for tr, v in tracks:
        if id(v) in lane:
          r["max_brake"] = round(max(r["max_brake"], Tracks.braking(tr)), 1)
      r["contacts"] += contacts
    if self.start_t is not None and lc == FINISHING:
      r["outcome"] = "done"
    if self.start_t is not None and r["outcome"] is None and lc == PRE and self.prev_state == STARTING:
      r["outcome"], r["detail"] = "aborted", "back to preLaneChange"
    ended = lc == OFF or lc == PRE and self.prev_state in (STARTING, FINISHING)
    if self.blinker_t is not None and not blinker and r["outcome"] is None and self.start_t is None:
      self.close(t, "ended", "the blinker went off before it started")
    elif self.start_t is not None and (ended or t - self.start_t > CHANGE_MAX):
      self.close(t, r["outcome"] or "unfinished", r["detail"])
    self.prev_state, self.prev_flag = lc, flag

  def close(self, t: float, outcome: str, detail: str | None = None):
    r = self.rec
    if self.done:
      return
    self.done = True
    r["outcome"], r["detail"] = r["outcome"] or outcome, r["detail"] or detail
    if r["wait_s"] is None and self.blinker_t is not None:
      r["wait_s"] = round(t - self.blinker_t, 2)
    s = r
    s["squeezed"] = bool((s["min_side_gap"] is not None and s["min_side_gap"] < SQUEEZE_GAP)
                         or (s["min_ttc"] is not None and s["min_ttc"] < SQUEEZE_TTC) or s["max_brake"] >= SQUEEZE_BRAKE
                         or s["contacts"] > 0)


class Run:
  def __init__(self, out_path: str):
    from openpilot.cereal import messaging
    self.messaging = messaging
    self.sm = messaging.SubMaster(["carState", "modelV2", "selfdriveState", "navSpeed"], poll="modelV2")
    self.states, self.log = Tail(STATE_LOG), Tail(BRIDGE_LOG)
    self.state: dict = {}
    self.state_t = 0.0
    self.contacts = 0
    self.last_contacts: int | None = None
    self.nav_lines: list[str] = []
    self.out = open(out_path, "a", buffering=1)

  def update(self, wait_ms: int = 100):
    self.sm.update(wait_ms)
    now = time.monotonic()
    for line in self.states.lines():
      if '"state"' not in line:
        continue
      try:
        s = json.loads(line)["state"]
      except (ValueError, KeyError):
        continue
      if s.get("inVehicle"):
        self.state, self.state_t = s, now
        n = s.get("collisions")
        if n is not None:
          self.contacts += max(0, n - self.last_contacts) if self.last_contacts is not None else 0
          self.last_contacts = n
    self.nav_lines += [ln.strip() for ln in self.log.lines() if ln.startswith("nav:")]

  def take(self) -> tuple[list[str], int]:
    lines, self.nav_lines = self.nav_lines, []
    c, self.contacts = self.contacts, 0
    return lines, c

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
    changes: list[Change] = []
    open_: Change | None = None
    tracks = Tracks()
    stopped_t, gas, gas_t, last_state = None, 0, 0.0, None
    outcome, detail, contacts_total, max_v = None, "", 0, 0.0
    while outcome is None:
      self.update()
      now = time.monotonic()
      t = now - t0
      lines, contacts = self.take()
      contacts_total += contacts
      bad = self.stale()
      if bad:
        outcome, detail = "infra", bad
        break
      s, cs = self.state, self.sm["carState"]
      lc = self.sm["modelV2"].meta.laneChangeState.raw
      ns = self.sm["navSpeed"] if self.sm.seen["navSpeed"] else None
      cap = (ns.speedCap, str(ns.reason)) if ns is not None and now - self.sm.recv_time["navSpeed"] < 1.0 else None
      seen = tracks.update(t, s.get("pos", (0, 0)), s.get("heading", 0.0), s.get("nearby")) if s is not last_state else []
      last_state = s
      max_v = max(max_v, s.get("vEgo", 0.0))
      for line in lines:
        if m := GIVEN_UP.match(line):
          if open_ is not None and open_.side == m.group(1):
            open_.close(t, "given_up", m.group(2))
        elif m := HELD.match(line):
          if open_ is not None:
            open_.rec["held_logged"] = True
        elif m := CHANGE_LINE.match(line):
          if open_ is not None and not open_.done:
            open_.close(t, "superseded", "nav asked for another")
          open_ = Change(name, t, m.group(1), m.group(2), s)
          changes.append(open_)
          print(f"bsm: {name} t={t:.0f} lane change {m.group(1)}, {m.group(2)}", flush=True)
      if open_ is not None and not open_.done:
        open_.step(t, cs, lc, s, seen, cap, contacts)
      for c in changes:
        if c.done:
          self._emit(c)
      v = s.get("vEgo", 0.0)
      left = math.hypot(s["pos"][0] - dx, s["pos"][1] - dy)
      if not self.engaged:
        outcome = "arrived" if left < ARRIVED_WITHIN else "disengaged"
        detail = "" if outcome == "arrived" else f"{self.sm['selfdriveState'].alertText1} ({left:.0f} m left)"
        break
      if t > timeout:
        outcome, detail = "timeout", f"{left:.0f} m left"
        break
      if v < STOPPED:
        stopped_t = stopped_t or now
        if gas < GAS_TRIES and now - stopped_t > GAS_AFTER and now - gas_t > GAS_AFTER:
          cmd("gas", secs=0.5)
          gas, gas_t = gas + 1, now
        if gas >= GAS_TRIES and now - gas_t > STUCK_AFTER:
          outcome, detail = "stuck", f"{left:.0f} m left"
          break
      else:
        stopped_t = None
    for c in changes:
      if not c.done:
        c.close(time.monotonic() - t0, "open", f"trip {outcome}")
        self._emit(c)
    self.set_engaged(False)
    rs = [c.rec for c in changes]
    return {**rec, "outcome": outcome, "detail": detail, "secs": round(time.monotonic() - t0), "max_v": round(max_v, 1),
            "contacts": contacts_total, "changes": len(rs), "changes_started": sum(r["started"] for r in rs),
            "started_flagged": sum(bool(r["started_flagged"]) for r in rs), "started_beside": sum(bool(r["started_beside"]) for r in rs),
            "given_up": sum(r["outcome"] == "given_up" for r in rs), "squeezed": sum(r["squeezed"] for r in rs),
            "held": sum(r["held_s"] > 0 for r in rs), "gas": gas}

  def _emit(self, c: Change):
    if c.rec.get("_emitted"):
      return
    c.rec["_emitted"] = True
    r = {k: v for k, v in c.rec.items() if k != "_emitted"}
    self.out.write(json.dumps(r) + "\n")
    flag = " STARTED INTO A FLAGGED SIDE" if r["started_flagged"] else ""
    print(f"bsm: {r['trip']} {r['side']} {r['what']}: {r['outcome']} wait {r['wait_s']} held {r['held_s']}" +
          f" squeezed {r['squeezed']}{flag}", flush=True)


def summary(paths: list[str]):
  rows = []
  for p in paths:
    with open(p) as f:
      rows += [json.loads(line) for line in f if line.strip()]
  trips = [r for r in rows if r["kind"] == "trip"]
  ch = [r for r in rows if r["kind"] == "change"]
  print(f"{len(trips)} trips: {dict(Counter(r['outcome'] for r in trips))}; contacts {sum(r.get('contacts', 0) for r in trips)}")
  by = defaultdict(list)
  for r in ch:
    by[r["what"]].append(r)
  print(f"{len(ch)} lane changes asked for")
  for what, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
    started = [r for r in rs if r["started"]]
    waits = sorted(r["wait_s"] for r in started if r["wait_s"] is not None)
    print(f"  {what}: {len(rs)} asked, {len(started)} started; flagged at the blinker {sum(bool(r['flag_at_blinker']) for r in rs)}, " +
          f"beside {sum(bool(r['beside_at_blinker']) for r in rs)}; held {sum(r['held_s'] > 0 for r in rs)} " +
          f"(max {max((r['held_s'] for r in rs), default=0):.1f} s); wait median {waits[len(waits) // 2] if waits else None} " +
          f"max {waits[-1] if waits else None} s")
    print(f"    STARTED INTO A FLAGGED SIDE {sum(bool(r['started_flagged']) for r in rs)}, beside at start " +
          f"{sum(bool(r['started_beside']) for r in rs)}; squeezed {sum(r['squeezed'] for r in rs)}, contacts " +
          f"{sum(r['contacts'] for r in rs)}; outcomes {dict(Counter(r['outcome'] for r in rs))}")
    for r in rs:
      if r["outcome"] == "given_up":
        print(f"    given up: {r['trip']} at {r['pos']}: {r['detail']}")
      if r["started_flagged"] or r["started_beside"] or r["squeezed"]:
        print(f"    !! {r['trip']} {r['side']} at {r['pos']} t={r['t']}: flagged {r['started_flagged']} beside {r['started_beside']} " +
              f"nearest {r['nearest_at_start']} side_gap {r['min_side_gap']} rear_gap {r['min_rear_gap']} ttc {r['min_ttc']} " +
              f"brake {r['max_brake']} contacts {r['contacts']}")


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("trips", nargs="*")
  ap.add_argument("--out", default=os.path.expanduser("~/gta5test/bsm_trial.jsonl"))
  ap.add_argument("--traffic", default="vehicles:1.5,parked:1,peds:1", help="density multipliers, key:value,...")
  ap.add_argument("--timeout", type=float, default=420.0, help="s per trip")
  ap.add_argument("--reps", type=int, default=1)
  ap.add_argument("--summary", nargs="+", help="results files to sum up instead")
  # a spec starting with a minus would be taken for an option: parsed as a stand-in, put back in its place
  argv = [f"@spec{i}" if ">" in x and x.startswith("-") else x for i, x in enumerate(sys.argv[1:])]
  args = ap.parse_intermixed_args(argv)
  args.trips = [sys.argv[1:][int(x[5:])] if x.startswith("@spec") else x for x in args.trips]
  if args.summary:
    summary(args.summary)
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
