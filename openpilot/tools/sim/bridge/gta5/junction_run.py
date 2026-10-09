#!/usr/bin/env python3
"""Junction recording run: the AI expert drives a junction_plan.py plan (each approach out every legal way, pass by
pass) while the bridge records, with record_run.py's Run (placing, recording, recovery, the stop file).

  junction_run.py run --plan plan.json --name jr1 [--until 2026-10-09T08:00] [record_run.py's run options]
  junction_run.py run --plan plan.json --name jr1 --dry-run [--valhalla valhalla.json]   the trips it would drive, checked
  junction_run.py manifest jr1 [--plan plan.json]                                         rewrite the manifest

It resumes: a run of the same --name (its <name>.jsonl in --runs) carries on where it stopped. A trip is done once it
arrives. A way out whose trips the expert failed (EXPERT_FAULTS) --max-tries times is dropped, and an approach after
--approach-fails of them, as places the expert can't drive; a trip that failed for anything else (the bridge, the game,
placing the car) goes again next. Before each trip its route is checked on the bridge's router (the live map): if it no
longer goes in by the approach and out by the planned way the trip is skipped (plan_mismatch). Once every trip has been
tried, those not done go once more, then the run ends ("plan done").

Exit status: 0 finished (plan done, --hours or --until up, the stop file, Ctrl-C, the driver took over); 3 stopped by
something a restart may get past (the game gone for --plugin-wait, failures in a row, bridge restarts, no router);
4 needs a person (low disk, the bridge's environment). junction_rec.sh runs it in tmux and restarts it on 3 once the
game sends state again.

Into --runs, beside record_run's files for <name>: <name>.manifest.json, rewritten after every trip: the plan, each
trip's outcome and segments, and per way out the trips that arrived. gta5-train's scripts/juncrec_labels.sh reads it."""
import argparse
import datetime
import json
import os
import random
import signal
import sys
import threading
import time
from collections import Counter, defaultdict

from openpilot.tools.sim.bridge.gta5 import e2e
from openpilot.tools.sim.bridge.gta5 import record_run as rr

EXPERT_FAULTS = {"timeout", "stuck", "collisions", "oncoming", "off_route", "reroutes", "expert_stopped"}
AGAIN = {"infra", "setup", "teleport", "no_route"}  # not the junction's doing: the trip goes again next
EXIT_MOVED = 25.0  # m: the live route's way out this far from the plan's is another way
FINISHED = 0
RESUMABLE = 3
NEEDS_PERSON = 4


def load_plan(path: str) -> dict:
  with open(path) as f:
    return json.load(f)


class Checker:
  """Routes a planned trip on the bridge's router (junction_plan.route_check) and says if it no longer drives the way
  out planned. Loading the map's paths and roads takes ~10 s; done once, on the first check."""
  def __init__(self, map_dir: str, url: str | None, valhalla: str | None):
    self.map_dir, self.url, self.valhalla = map_dir, url, valhalla
    self.router = None

  def __call__(self, a: dict, e: dict) -> str | None:
    from openpilot.tools.sim.bridge.gta5 import junction_plan as jp
    if self.router is None:
      self.router = jp.make_router(self.map_dir, self.url, self.valhalla)
    r = jp.route_check(self.router, a["start"], e["dest"], a["via_in"], e["via_out"], max(a["before"], jp.BEFORE), e["after"])
    if not r["ok"]:
      if r["why"].startswith("no route: URLError") or r["why"].startswith("no route: ConnectionRefused"):
        return None  # the router isn't up: the bridge can't route it either, and the trip goes again
      return r["why"]
    moved = ((r["exit_point"][0] - e["exit_point"][0]) ** 2 + (r["exit_point"][1] - e["exit_point"][1]) ** 2) ** 0.5
    return f"way out moved {moved:.0f} m" if moved > EXIT_MOVED else None


class PlanPicker:
  """record_run.Run's picker over a plan: next() the next trip to drive (from Run's look-ahead thread), record() each
  outcome; raises Stop("plan done") when nothing is left."""
  def __init__(self, plan: dict, args, done: list[dict], checker, write_rec):
    self.plan, self.args, self.checker, self.write_rec = plan, args, checker, write_rec
    self.approaches = {a["id"]: a for a in plan["approaches"]}
    self.exits = {e["id"]: e for a in plan["approaches"] for e in a["exits"]}
    # wrong-way trips (junction_plan.py wrongway) go only with --wrongway; without it the plan is driven as without them
    self.trips = [t for t in plan["trips"] if getattr(args, "wrongway", False) or not t.get("wrongway")]
    self.by_id = {t["id"]: t for t in self.trips}
    self.cv = threading.Condition()
    self.arrived, self.tried = Counter(), Counter()
    self.exit_fails, self.approach_fails = Counter(), Counter()
    self.mismatch: dict[str, str] = {}
    self.again: list[str] = []
    self.out: set[str] = set()  # handed out, outcome not yet recorded
    self.cursor, self.sweep = 0, False
    self.seq = len(done)
    for r in done:
      self._count(r)

  def _count(self, rec: dict):
    t = self.by_id.get(rec.get("plan_trip"))
    if t is None:
      return
    o = rec.get("outcome")
    if o == "arrived":
      self.arrived[t["id"]] += 1
    elif o == "plan_mismatch":
      self.mismatch[t["id"]] = rec.get("detail", "")
    if o not in AGAIN and o != "stopped":
      self.tried[t["id"]] += 1
    if o in EXPERT_FAULTS:
      self.exit_fails[t["exit"]] += 1
      self.approach_fails[t["approach"]] += 1

  def dropped(self, t: dict) -> str | None:
    if t["id"] in self.mismatch:
      return f"plan mismatch: {self.mismatch[t['id']]}"
    if self.exit_fails[t["exit"]] >= self.args.max_tries:
      return "the expert failed this way out"
    if self.approach_fails[t["approach"]] >= self.args.approach_fails:
      return "the expert failed this approach"
    return None

  def pending(self, t: dict) -> bool:
    return not self.arrived[t["id"]] and self.dropped(t) is None and t["id"] not in self.out

  def _pick(self) -> dict | None:
    while self.again:
      t = self.by_id[self.again.pop(0)]
      if self.pending(t):
        return t
    while True:
      while self.cursor < len(self.trips):
        t = self.trips[self.cursor]
        self.cursor += 1
        if self.pending(t) and (self.sweep or not self.tried[t["id"]]):
          return t
      if self.sweep:
        return None
      self.sweep, self.cursor = True, 0  # every trip tried: those not done, once more

  def next(self) -> dict:
    while True:
      with self.cv:
        t = self._pick()
        while t is None:
          if not self.out:
            raise rr.Stop("plan done")
          self.cv.wait(5.0)  # the trip being driven may have to go again
          t = self._pick()
        self.out.add(t["id"])
      why = None
      if t.get("wrongway"):
        break  # no junction to check; the bridge checks the road as the clip starts (gta5_wrongway.suitable)
      a, e = self.approaches[t["approach"]], self.exits[t["exit"]]
      if self.checker is not None:
        try:
          why = self.checker(a, e)
        except Exception as ex:  # a broken check must not stop the night
          rr.say(f"junction: checking {t['id']} failed: {type(ex).__name__}: {ex}")
      if why is None:
        break
      rr.say(f"junction: {t['id']} skipped, its route on the live map: {why}")
      rec = {"id": f"{self.args.name}-m{len(self.mismatch):03d}", "run": self.args.name, "plan_trip": t["id"],
             "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "spec": t["spec"], "outcome": "plan_mismatch", "detail": why}
      self.write_rec(rec)
      with self.cv:
        self.out.discard(t["id"])
        self._count(rec)
    self.seq += 1
    if t.get("wrongway"):
      return wrongway_trip(t, f"{self.args.name}-{self.seq:04d}", getattr(self.args, "ww_recover", "plan"))
    return {"id": f"{self.args.name}-{self.seq:04d}", "plan_trip": t["id"], "spec": t["spec"],
            "start": (a["start"]["x"], a["start"]["y"]), "dest": tuple(e["dest"]), "area": t["split"],
            "length": e["length"], "time": e["time"] or round(e["length"] / 8.0), "maneuvers": e["maneuvers"],
            "classes": [e["kind"]], "junctions": 1, "familiar": 0.0, "geom": e["geom"][::2],
            "world": t.get("world") or {},
            "plan": {"approach": t["approach"], "exit": t["exit"], "pass": t["pass"], "split": t["split"], "lane": t["lane"],
                     "kind": e["kind"], "junction": a["junction"], "extra": t["extra"], "freeway": bool(a.get("freeway"))}}

  def record(self, rec: dict):
    with self.cv:
      self._count(rec)
      self.out.discard(rec.get("plan_trip"))
      if rec.get("outcome") in AGAIN and rec.get("plan_trip"):
        self.again.append(rec["plan_trip"])
      self.cv.notify_all()

  def progress(self) -> dict:
    with self.cv:
      done = sum(1 for t in self.trips if self.arrived[t["id"]])
      dropped = sum(1 for t in self.trips if not self.arrived[t["id"]] and self.dropped(t))
      return {"plan_trips": len(self.trips), "arrived": done, "dropped": dropped, "sweep": self.sweep}


class JunctionRun(rr.Run):
  def __init__(self, args, plan_path: str):
    super().__init__(args)
    self.plan_path = plan_path
    self.picker: PlanPicker | None = None

  def ensure_bridge(self):
    """record_run's check over every variable of --bridge-extra (junction_rec.sh's turns the map overlay off, whose
    thread stalls the bridge's frames), not only expert mode and recording."""
    env = rr.bridge_env()
    wrong = {k: env.get(k) for k, v in self.extra.items() if env.get(k) != v}
    if not wrong:
      return
    if not self.args.fix_bridge:
      raise rr.Stop(f"the bridge runs without {self.extra} (it has {wrong}): run with --fix-bridge")
    self.restarts -= 1  # not a failure
    self.restart_bridge(f"it runs with {wrong}")

  def drive(self, c: dict, settings: list[str], world: dict) -> dict:
    write_manifest(self.args.runs, self.args.name, self.plan_path)
    # the plan's time of day and weather for this trip (set after randomise's), unless --hours-of-day / --weathers
    rec = super().drive(c, settings, world or c.get("world") or {})
    rec["plan_trip"], rec["plan"] = c["plan_trip"], c["plan"]
    if c.get("wrongway"):
      rec["wrongway"] = c["wrongway"]
      ends = [e for e in rec.get("wrongway_phases", []) if e.get("phase") == "end"]
      rec["wrongway_end"] = ends[-1].get("finished") if ends else None
      if rec["outcome"] == "expert_stopped" and "wrongway abort" in rec.get("detail", ""):
        rec["outcome"] = "ww_abort"
    self.picker.record(rec)
    rr.say(f"junction: {rec['plan_trip']} ({c['plan']['split']}, {c['plan']['kind']}): {rec['outcome']}; plan {self.picker.progress()}")
    return rec


def wrongway_trip(t: dict, run_id: str, recover: str = "plan") -> dict:
  """A wrong-way clip's trip for record_run.Run.drive: its route, the clip for expert mode, its traffic, and the
  oncoming time its controller allows."""
  w = t["wrongway"]
  clip = {**w["clip"], "clip": t["id"], **({"recover": recover} if recover != "plan" else {})}
  hold_onc = (w["clip"]["drift_m"] + w["clip"]["recover_m"]) / max(w["clip"]["speed"], 3.0) + w["clip"]["hold_s"]
  return {"id": run_id, "plan_trip": t["id"], "spec": t["spec"], "start": tuple(w["start"]), "dest": tuple(w["dest"]),
          "area": "wrongway", "length": w["length"], "time": round(w["length"] / 6.0), "maneuvers": [], "classes": ["wrongway"],
          "junctions": 0, "familiar": 0.0, "geom": w["geom"], "world": t.get("world") or {},
          "traffic": w.get("traffic"), "wrongway": clip, "oncoming_s": hold_onc + 25.0,
          "expert_extra": ["wrongway=" + json.dumps(clip, separators=(",", ":"))],
          "plan": {"approach": t["approach"], "exit": t["exit"], "pass": t["pass"], "split": "wrongway", "lane": t["lane"],
                   "kind": "wrongway", "junction": None, "extra": False, "freeway": False}}


def write_rec_line(path: str, rec: dict):
  with open(path, "a") as f:
    f.write(json.dumps(rec) + "\n")


def scene(rec: dict) -> dict | None:
  """What gta5_cmd randomise picked for a trip (its `randomise {json}` line): car, mount, traffic; the world the plan's."""
  text = rec.get("randomise") or ""
  k = text.find("randomise {")
  try:
    s = json.JSONDecoder().raw_decode(text[k + len("randomise "):])[0] if k >= 0 else None
  except ValueError:
    return None
  return {key: s[key] for key in ("vehicle", "mount", "traffic") if key in s} if isinstance(s, dict) else None


def manifest(runs: str, name: str, plan_path: str | None) -> dict:
  """The run as it stands: trips, segments, and per way out and approach what arrived."""
  recs = [r for r in rr.read_jsonl(os.path.join(runs, f"{name}.jsonl")) if "outcome" in r]
  segs = rr.read_jsonl(os.path.join(runs, f"{name}.segments.jsonl"))
  if plan_path is None:  # the one the run's manifest names
    try:
      with open(os.path.join(runs, f"{name}.manifest.json")) as f:
        plan_path = json.load(f).get("plan")
    except (OSError, ValueError):
      pass
  plan = load_plan(plan_path) if plan_path and os.path.exists(plan_path) else {"approaches": [], "trips": []}
  run_to_plan = {r["id"]: r.get("plan_trip") for r in recs}
  trips = {t["id"]: t for t in plan["trips"]}
  arrived = defaultdict(list)
  for r in recs:
    if r["outcome"] == "arrived" and r.get("plan_trip") in trips:
      arrived[trips[r["plan_trip"]]["exit"]].append(r["id"])
  ways, approaches = [], []
  for a in plan["approaches"]:
    got = [e["id"] for e in a["exits"] if arrived.get(e["id"])]
    approaches.append({"id": a["id"], "split": a["split"], "name": a.get("name"), "junction": a["junction"],
                       "ways": len(a["exits"]), "ways_arrived": len(got), "pairs": len(got) >= 2})
    for e in a["exits"]:
      ways.append({"id": e["id"], "approach": a["id"], "split": a["split"], "kind": e["kind"], "dest": e["dest"],
                   "arrived": arrived.get(e["id"], [])})
  by_split = {}
  for split in ("train", "eval"):
    ap = [x for x in approaches if x["split"] == split]
    by_split[split] = {"approaches": len(ap), "with_pairs": sum(x["pairs"] for x in ap),
                       "ways": sum(x["ways"] for x in ap), "ways_arrived": sum(x["ways_arrived"] for x in ap),
                       "trips_arrived": sum(len(w["arrived"]) for w in ways if w["split"] == split)}
  return {"run": name, "written": time.strftime("%Y-%m-%dT%H:%M:%S"), "plan": plan_path, "plan_made": plan.get("made"),
          "map": plan.get("map"), "outcomes": dict(Counter(r["outcome"] for r in recs)), "by_split": by_split,
          "trips": [{**{k: r.get(k) for k in ("id", "plan_trip", "plan", "started", "outcome", "detail", "duration", "distance",
                                              "segments", "world", "randomise_seed")}, "scene": scene(r)} for r in recs],
          "segments": [{**s, "plan_trip": run_to_plan.get(s.get("trip"))} for s in segs],
          "approaches": approaches, "ways": ways}


def write_manifest(runs: str, name: str, plan_path: str | None) -> dict:
  m = manifest(runs, name, plan_path)
  path = os.path.join(runs, f"{name}.manifest.json")
  with open(path + ".tmp", "w") as f:
    json.dump(m, f, indent=1)
  os.replace(path + ".tmp", path)
  return m


def exit_code(why: str) -> int:
  if why.startswith(("low disk", "the bridge runs without")):
    return NEEDS_PERSON
  if why in ("plan done", "interrupted", "SIGINT", "SIGTERM") or why.endswith(" h up") or why.startswith("stop file") or \
     "took over" in why:
    return FINISHED
  return RESUMABLE


def cmd_run(args) -> int:
  os.makedirs(args.runs, exist_ok=True)
  if args.seed is None:
    args.seed = random.SystemRandom().randrange(100000)
  if args.until:
    left = (datetime.datetime.fromisoformat(args.until) - datetime.datetime.now()).total_seconds() / 3600
    if left <= 0:
      print(f"--until {args.until} has passed")
      return FINISHED
    args.hours = min(args.hours, left)
  if args.router:
    e2e.VALHALLA = args.router
  else:
    router = rr.parse_extra(args.bridge_extra).get("GTA5_ROUTER")
    if router:
      e2e.VALHALLA = router
  plan = load_plan(args.plan)
  out = os.path.join(args.runs, f"{args.name}.jsonl")
  done = [r for r in rr.read_jsonl(out) if "outcome" in r]
  if done:
    print(f"resuming {args.name}: {len(done)} trips in {out}")
  checker = None if args.no_check else Checker(args.map or plan["map"], None if args.valhalla else e2e.VALHALLA, args.valhalla)

  if args.dry_run:
    picker = PlanPicker(plan, args, done, checker, lambda rec: None)
    n, kinds, splits = 0, Counter(), Counter()
    secs = 0.0
    try:
      while n < args.n:
        c = picker.next()
        n += 1
        t = picker.by_id[c["plan_trip"]]
        secs += t["est_s"]
        kinds[c["plan"]["kind"]] += 1
        splits[c["plan"]["split"]] += 1
        print(f"{c['id']} {t['id']:12s} {c['plan']['split']:5s} pass {t['pass']} lane {t['lane']} {c['plan']['kind']:13s} " +
              f"{c['length']:4d} m ~{t['est_s']} s  {c['spec']}", flush=True)
        picker.record({"plan_trip": t["id"], "outcome": "arrived"})
    except rr.Stop as e:
      print(f"stops: {e}")
    print(f"{n} trips, {secs / 3600:.1f} h estimated; {dict(splits)}; ways out {dict(kinds.most_common())}; " +
          f"skipped as no longer routed so: {len(picker.mismatch)} {sorted(set(picker.mismatch.values()))[:8]}")
    return FINISHED

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
  if args.clear_stop and os.path.exists(args.stop_file):
    print(f"removing an old stop file {args.stop_file}")
    os.remove(args.stop_file)
  rr.LOG = open(os.path.join(args.runs, f"{args.name}.log"), "a")
  run = JunctionRun(args, os.path.abspath(args.plan))
  run.picker = PlanPicker(plan, args, done, checker, lambda rec: write_rec_line(out, rec))
  signal.signal(signal.SIGINT, run.on_signal)
  signal.signal(signal.SIGTERM, run.on_signal)
  rr.say(f"junction: plan {args.plan} ({plan.get('made')}, map {plan.get('map')}): {len(plan['trips'])} trips; " +
         f"{run.picker.progress()}")
  try:
    why = run.run(run.picker)
  finally:
    try:
      os.remove(lock)
    except OSError:
      pass
    m = write_manifest(args.runs, args.name, os.path.abspath(args.plan))
    rr.say(f"junction: manifest {args.name}.manifest.json: {m['by_split']}")
  code = exit_code(why)
  rr.say(f"junction: stopped ({why}), exit {code}")
  return code


def main() -> int:
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest="command", required=True)
  r = sub.add_parser("run")
  r.add_argument("--plan", required=True, help="junction_plan.py's plan")
  r.add_argument("--until", help="ISO local time the run ends by (each resumed run takes what's left)")
  r.add_argument("--max-tries", type=int, default=2, help="expert failures that drop a way out")
  r.add_argument("--approach-fails", type=int, default=4, help="expert failures that drop an approach")
  r.add_argument("--no-check", action="store_true", help="don't route each trip on the live router first")
  r.add_argument("--map", help="the map folder for the check (default the plan's)")
  r.add_argument("--valhalla", help="check on a valhalla.json in process (--dry-run on a copy of the map)")
  r.add_argument("--dry-run", action="store_true", help="only list --n trips in order, each checked, as if all arrived")
  r.add_argument("--n", type=int, default=10000)
  r.add_argument("--wrongway", action="store_true", help="drive the plan's wrong-way trips too (junction_plan.py " +
                 "wrongway); without it they're left out, as in a plan without them")
  r.add_argument("--ww-recover", choices=["plan", "ai", "path"], default="plan",
                 help="who brings a wrong-way clip's car back: as the plan says, the game's AI, or the map path controller")
  r.add_argument("--clear-stop", action="store_true", help="remove an old stop file first (a fresh start, not a resume)")
  rr.run_arguments(r)
  r.set_defaults(name="jr1")
  ma = sub.add_parser("manifest")
  ma.add_argument("name")
  ma.add_argument("--plan")
  ma.add_argument("--runs", default=f"{rr.T}/recruns")
  args = p.parse_args()
  if args.command == "manifest":
    m = write_manifest(args.runs, args.name, args.plan)
    print(json.dumps({k: m[k] for k in ("run", "plan", "outcomes", "by_split")}, indent=1))
    return 0
  return cmd_run(args)


if __name__ == "__main__":
  sys.exit(main())
