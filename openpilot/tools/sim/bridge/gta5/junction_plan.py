#!/usr/bin/env python3
"""Junction recording plan: junctions the AI expert drives through again and again, each time out a different legal way,
so training has the same approach with different routes and the real path each one led to (junction_run.py drives it).

  junction_plan.py plan --out plan.json [--hours 8] [--map ~/gta5map_lanes] [--router URL | --valhalla valhalla.json]
  junction_plan.py show plan.json [--hours 6] [--fig plan.png]

Junctions are GTA's junction nodes, clustered (CLUSTER_LINK, CLUSTER_NEAR). An approach is a road into one: the car
starts at a road node about BEFORE m up it (SHORT_BEFORE where the router would leave that road for another street from
further back; e2e.Map's walk and spawnable rules, as e2e's short trips), so the model's history is warm well before the
junction, and each way out ends AFTER m down its road. Every approach and way out is routed as the bridge would route it
(router.Router with the map's paths and roads; --valhalla runs Valhalla in process on a copy of the map, else
--router's service) and kept only where the route runs in along the approach's link, out along that way's link, with no
turn, ramp or roundabout before the junction, no U-turn and no detour: so the ways out are the legal ones. An approach
needs two.

Held out for eval, whole junctions: those of --eval-trips (the in-game A/B set, so it stays a test), driven from each
trip's own start; no training junction lies within EVAL_BUFFER m of one. --bad-labels (a labels root) leaves out
junctions where the expert collided, left the route or drove in the oncoming lanes before (the labeller's rejections).
Training approaches are picked by kind (kind_score: those nav and the model get wrong first), spread over the city
(AREA_CELL), until --hours is filled at --train-reps passes; eval junctions are driven --eval-reps times, --fail-reps
for the --fail-trips ones.

The trips go pass by pass (each pass every approach and way out once, the training ones in a nearest-neighbour tour and
then the eval ones, so the recordings' segments split cleanly), an approach's ways out one after another from the same
start, in an order rotating by pass. On approaches with two or more lanes the start lane alternates leftmost /
rightmost, so the expert has to pick its lane for the way out. --extra-passes more training passes follow, for a night
that goes faster than the estimate (EST_OVERHEAD + length / EST_SPEED a trip).
"""
import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HOME = os.path.expanduser("~")
DEFAULT_MAP = f"{HOME}/gta5map_lanes"

BEFORE, BEFORE_SLACK, MIN_BEFORE = 250.0, 40.0, 180.0  # m of road from the start to the junction
SHORT_BEFORE, SHORT_SLACK, SHORT_MIN = 190.0, 30.0, 150.0  # where the router leaves the road in from further back
AFTER, AFTER_SLACK, MIN_AFTER = 180.0, 40.0, 120.0  # m on from it to the destination
CLUSTER_LINK = 35.0  # m: junction nodes linked this near are one junction
CLUSTER_NEAR = 20.0  # m: and any this near
ARM_DEG = 30.0  # links in or out within this of each other's bearing are one road
UTURN_DEG = 30.0  # a way out within this of the way back
PASS_NEAR = 10.0  # m: a route runs along a link this near, its way (PASS_DEG)
PASS_DEG = 50.0
NOT_BEFORE = {10, 11, 14, 15, 17, 18, 19, 20, 21, 26, 27}  # Valhalla turns, ramps, exits, roundabouts: none on the way in
EXIT_KEEP = 15.0  # m: ways out whose routes pass the same point this near are one
TURN_NEAR = 40.0  # m along from the junction: a maneuver there is the junction's
NO_TURN_BEFORE = 40.0  # m before the junction: no NOT_BEFORE maneuver on the way in
BEND_FROM = 100.0  # m before the junction: the road's bend from there in (SO05, SR1: 30-50 deg)
OUT_WITHIN = 120.0  # m from in to out along the route: further is a detour
DETOUR = 150.0  # m a route may be longer than before + after
EVAL_BUFFER = 300.0  # m between an eval junction and any training one
BAD_NEAR = 40.0  # m: a rejected stretch this near a junction counts against it
AREA_CELL = 500.0  # m: spreading the picks
TOO_CLOSE = 60.0  # m: an approach starting this near a picked one's start is skipped
GEOM_STEP = 25.0  # m between the route points kept in the plan
EST_OVERHEAD = 23.0  # s a trip: randomise, setup and placing (~12 s between trips), launch, arrival hold
EST_SPEED = 5.0  # m/s on the road, light waits included (overnight1: 5.7 m/s on 0.5-3 km trips)


def wrap(d: float) -> float:
  return (d + 180.0) % 360.0 - 180.0


# *** routing ***

def make_router(map_dir: str, url: str | None, config: str | None):
  """router.Router as the bridge has it (the map's paths for heights and nodes, its roads for destinations), over
  --router's service or, with --valhalla, Valhalla in process; no speed limits (one query less a route)."""
  from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
  from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes
  from openpilot.tools.sim.bridge.gta5.map.paths import Paths
  from openpilot.tools.sim.bridge.gta5.map.router import Router

  actor = None
  if config:
    from valhalla import Actor
    cfg = json.loads(Path(config).read_text())
    cfg["mjolnir"].pop("tile_extract", None)  # read the tile folder
    actor = Actor(cfg)

  class PlanRouter(Router):
    def _post(self, action, request):
      if actor is None:
        return super()._post(action, request)
      r = getattr(actor, action)(json.dumps(request))
      return json.loads(r) if isinstance(r, str) else r

    def _limits(self, shape, n):
      return np.zeros(max(n - 1, 0))

  paths = Paths(os.path.join(map_dir, "paths.jsonl"))
  paths.index()
  pbf = os.path.join(map_dir, "gta5.osm.pbf")
  roads = OsmLanes.load(pbf, to_game) if os.path.exists(pbf) else None
  return PlanRouter(url or "http://127.0.0.1:8002", timeout=10.0, paths=paths, osm=None, roads=roads)


def wrap_rad(r: float) -> float:
  return (r + math.pi) % (2 * math.pi) - math.pi


def passes(pts: np.ndarray, seg, lo: int = 0, into: bool = True) -> int | None:
  """The route point from lo on nearest the link seg ([[x, y], [x, y]]) where the route runs along it its way, or None:
  into a junction, coming along it (a slip road can leave at its first node); out of one, going on along it."""
  a, b = np.asarray(seg[0], float), np.asarray(seg[1], float)
  ab = b - a
  t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-9), 0.0, 1.0)
  d = np.hypot(*(a + t[:, None] * ab - pts).T)
  h_seg = math.atan2(ab[0], ab[1])
  best = None
  for k in np.flatnonzero(d <= PASS_NEAR):
    if k < lo:
      continue
    if into and k == 0 or not into and k + 1 >= len(pts):
      continue
    v = pts[k] - pts[k - 1] if into else pts[k + 1] - pts[k]
    if np.hypot(*v) < 1e-6 or abs(math.degrees(wrap_rad(math.atan2(v[0], v[1]) - h_seg))) > PASS_DEG:
      continue
    if best is None or d[k] < d[best]:
      best = int(k)
  return best


def route_check(router, start: dict, dest: list, via_in: list, via_out: list, before: float, after: float) -> dict:
  """Routes start -> dest and whether it goes in along via_in (the approach's link into the junction) and out along
  via_out (the way out's link from it): {"ok", "why", and the route's length, time, kind of way out, ...}."""
  from openpilot.tools.sim.bridge.gta5 import e2e
  from openpilot.tools.sim.bridge.gta5.record_run import maneuver_class
  bearing = (-start["heading"]) % 360
  try:
    r = router.route(np.array([start["x"], start["y"]]), bearing, np.array(dest, float), start["z"])
  except Exception as e:  # no route at all is an answer here
    return {"ok": False, "why": f"no route: {type(e).__name__}"}
  pts, along = r.points, r.along
  k_in = passes(pts, via_in)
  if k_in is None:
    return {"ok": False, "why": "not in by the approach"}
  k_out = passes(pts, via_out, k_in, into=False)
  if k_out is None:
    return {"ok": False, "why": "not out that way"}
  s_in, s_out = float(along[k_in]), float(along[k_out])
  if s_out - s_in > OUT_WITHIN:
    return {"ok": False, "why": "out that way only after a detour"}
  if along[-1] > before + after + DETOUR:
    return {"ok": False, "why": f"detour ({along[-1]:.0f} m)"}
  mans = []
  for m in r.maneuvers:
    s = float(along[min(m["begin_shape_index"], len(along) - 1)])
    angle = wrap(m["bearing_after"] - m["bearing_before"]) if "bearing_before" in m and "bearing_after" in m else None
    mans.append({"type": m["type"], "s": s, "angle": angle, "real": m["type"] in e2e.REAL})
  if any(m["type"] in (12, 13) for m in mans):
    return {"ok": False, "why": "U-turn"}
  if any(m["type"] in NOT_BEFORE and m["s"] < s_in - NO_TURN_BEFORE for m in mans):
    return {"ok": False, "why": "a turn on the way in"}
  at = [m for m in mans if m["real"] and s_in - NO_TURN_BEFORE <= m["s"] <= s_out + TURN_NEAR]
  kind = maneuver_class(at[0]) if at else "straight"

  def point(s):
    return np.array([np.interp(s, along, pts[:, 0]), np.interp(s, along, pts[:, 1])])

  def heading(s0, s1):
    d = point(s1) - point(s0)
    return math.degrees(math.atan2(d[0], d[1])) % 360  # compass
  change = wrap(heading(s_out + 10, s_out + 30) - heading(s_in - 20, s_in - 5))
  bend = wrap(heading(s_in - 20, s_in - 5) - heading(max(s_in - BEND_FROM, 0.0), max(s_in - BEND_FROM + 20, 5.0)))
  geom =[[round(float(v), 1) for v in point(s)] for s in np.append(np.arange(0.0, along[-1], GEOM_STEP), along[-1])]
  return {"ok": True, "length": round(float(along[-1])), "time": round(float(sum(m.get("time", 0) for m in r.maneuvers))),
          "s_in": round(s_in, 1), "s_out": round(s_out, 1), "kind": kind,
          "angle": round(at[0]["angle"]) if at and at[0]["angle"] is not None else None, "change": round(change),
          "bend": round(bend), "maneuvers": [e2e.TYPES.get(m["type"], str(m["type"])) for m in mans],
          "exit_point": [round(float(v), 1) for v in point(s_out + 30)], "geom": geom}


# *** junctions on GTA's links ***

def clusters(m) -> list[dict]:
  jn = [k for k, n in m.nodes.items() if n["f"][2] & 4 and (m.out.get(k) or m.into.get(k))]
  parent = {k: k for k in jn}

  def find(k):
    while parent[k] != k:
      parent[k] = parent[parent[k]]
      k = parent[k]
    return k
  jset = set(jn)
  for k in jn:
    n = m.nodes[k]
    for o in list(m.out.get(k, ())) + list(m.into.get(k, ())):
      if o in jset and m.gap(k, o) < CLUSTER_LINK:
        parent[find(o)] = find(k)
    for o in m.nodes_near(n["x"], n["y"], CLUSTER_NEAR):
      if o in jset:
        parent[find(o)] = find(k)
  groups = defaultdict(set)
  for k in jn:
    groups[find(k)].add(k)
  out = []
  for nodes in groups.values():
    pts = np.array([[m.nodes[k]["x"], m.nodes[k]["y"]] for k in nodes])
    c = pts.mean(axis=0)
    out.append({"nodes": nodes, "x": float(c[0]), "y": float(c[1]), "size": float(np.hypot(*(pts - c).T).max())})
  return out


def arms(m, links, bearing_of) -> list[tuple]:
  """Links grouped by bearing (ARM_DEG): one representative each, the one with the most lanes."""
  found = []
  for ln in sorted(links, key=lambda ln: -m.lanes.get(ln, 0)):
    b = bearing_of(ln)
    if all(abs(wrap(b - fb)) > ARM_DEG for fb, _ in found):
      found.append((b, ln))
  return [ln for _, ln in found]


def xy(m, k) -> list[float]:
  return [m.nodes[k]["x"], m.nodes[k]["y"]]


def junction_arms(m, cl: dict) -> tuple[list, list]:
  nodes = cl["nodes"]
  ins = [(u, v) for v in nodes for u in m.into.get(v, ()) if u not in nodes]
  outs = [(a, b) for a in nodes for b in m.out.get(a, ()) if b not in nodes]
  return arms(m, ins, lambda ln: m.bearing(*ln)), arms(m, outs, lambda ln: m.bearing(*ln))


def approach_start(m, u, v, before: float, slack: float, least: float):
  """The node to start at about `before` m up the road into u -> v: (node, the node after it, m to v, the nodes
  passed) or None."""
  walked = m.walk(u, m.bearing(u, v), before + slack, back=True)
  d0 = m.gap(u, v)
  chain, dist = [v] + [k for k, _ in walked], [0.0] + [d0 + d for _, d in walked]
  ok = {}
  for i in range(1, len(chain)):
    if m.spawnable(chain[i + 1] if i + 1 < len(chain) else None, chain[i], chain[i - 1]) is None:
      ok[i] = dist[i]
  near = [i for i, d in ok.items() if abs(d - before) <= slack] or [i for i, d in ok.items() if d >= least]
  if not near:
    return None
  i = min(near, key=lambda i: abs(dist[i] - before))
  return chain[i], chain[i - 1], dist[i], [m.nodes[k] for k in chain[1:i]]


def exit_dest(m, a, b):
  """The node to end at about AFTER m down the road out by a -> b: (node, m from a) or None."""
  d1 = m.gap(a, b)
  on = [(b, d1)] + [(n, d1 + d) for n, d in m.walk(b, m.bearing(a, b), AFTER + AFTER_SLACK, back=False)[1:]]
  ends = [(n, d) for n, d in on if d >= MIN_AFTER and not m.nodes[n]["f"][2] & 4]
  if not ends:
    return None
  return min(ends, key=lambda e: abs(e[1] - AFTER))


def ways_out(m, out_arms, b_in: float) -> list[dict]:
  exits = []
  for a, b in out_arms:
    if abs(wrap(m.bearing(a, b) - b_in - 180.0)) <= UTURN_DEG:
      continue
    d = exit_dest(m, a, b)
    if d is None:
      continue
    dn, dd = d
    exits.append({"via_out": [xy(m, a), xy(m, b)], "dest": [round(m.nodes[dn]["x"], 1), round(m.nodes[dn]["y"], 1)],
                  "after": round(dd), "bearing": round(m.bearing(a, b)), "street": m.streets.get(m.nodes[b]["st"])})
  return exits


def approach(m, u, v, exits: list[dict], start: dict, before: float, lanes_start: int, bend: float, passed: list) -> dict:
  lanes_in = m.lanes.get((u, v), 1)
  return {"start": start, "via_in": [xy(m, u), xy(m, v)], "before": round(before), "bearing_in": round(m.bearing(u, v)),
          "bend_in": round(bend), "lanes_start": lanes_start, "lanes_in": lanes_in, "bay": lanes_in > lanes_start,
          "street": m.streets.get(m.nodes[u]["st"]), "junctions_on_way": sum(1 for n in passed if n["f"][2] & 4),
          "exits": [dict(e) for e in exits]}


def candidates(m, cl: dict, before: float = BEFORE, slack: float = BEFORE_SLACK, least: float = MIN_BEFORE) -> list[dict]:
  """Every approach of junction cl with a start node and two or more ways out on GTA's links (not routed yet)."""
  in_arms, out_arms = junction_arms(m, cl)
  found = []
  for u, v in in_arms:
    b_in = m.bearing(u, v)
    exits = ways_out(m, out_arms, b_in)
    if len(exits) < 2:
      continue
    st = approach_start(m, u, v, before, slack, least)
    if st is None:
      continue
    k, ahead, dist, passed = st
    kind = m.ynd.highway(m.nodes, k, ahead, m.lanes.get((k, ahead), 0), m.lanes.get((ahead, k), 0))
    if kind in ("motorway", "trunk", "service", "track"):
      continue  # no standing start on a freeway; car parks and alleys aren't the roads to learn
    sn = m.nodes[k]
    start = {"x": round(sn["x"], 1), "y": round(sn["y"], 1), "z": round(sn["z"], 1), "heading": round((-m.bearing(k, ahead)) % 360)}
    found.append(approach(m, u, v, exits, start, dist, m.lanes.get((k, ahead), 1), wrap(m.bearing(k, ahead) - b_in), passed))
  return found


def eval_approach(m, router, cl: dict, spec: str) -> dict | None:
  """The approach an e2e trip drives into junction cl on, from that trip's own start (the A/B's placing), with every
  way out of the junction (not routed yet)."""
  from openpilot.tools.sim.bridge.gta5 import e2e
  (sx, sy, sz, sh, dx, dy), _ = e2e.parse_spec(spec)
  start = {"x": sx, "y": sy, "z": sz, "heading": sh}
  try:
    r = router.route(np.array([sx, sy]), (-sh) % 360, np.array([dx, dy]), sz)
  except Exception:
    return None
  in_arms, out_arms = junction_arms(m, cl)
  best = None
  for u, v in in_arms:
    k = passes(r.points, [xy(m, u), xy(m, v)])
    if k is not None and (best is None or r.along[k] < best[0]):
      best = (float(r.along[k]), u, v)
  if best is None:
    return None
  s_in, u, v = best
  exits = ways_out(m, out_arms, m.bearing(u, v))
  bearing = (-sh) % 360
  link = m.link_at(sx, sy, bearing, False)
  lanes_start = m.lanes.get(tuple(link), 1) if link else 1
  return approach(m, u, v, exits, start, s_in, lanes_start, wrap(bearing - m.bearing(u, v)), [])


def kind_score(a: dict) -> tuple[float, list[str]]:
  """How much an approach is worth recording: the kinds nav and the model get wrong (TASK.md's always-failing A/B
  trips: SO05 / SR1 bend on the way in, SL7 / SO06 / SO10 / SR6 two lanes in, SL7 / SO05 / SO06 / SR4 skewed, SO06
  sharp, SR1 onto a freeway, SR6 onto an unnamed road; most are lefts)."""
  tags = []
  ex = a["exits"]
  turns = [e for e in ex if e["kind"] != "straight"]
  if a["lanes_in"] >= 2:
    tags.append("lanes")
  if a["bay"]:
    tags.append("bay")
  if abs(a["bend_in"]) >= 25:
    tags.append("bend_in")
  angles = [abs(e["angle"]) for e in turns if e.get("angle") is not None]
  if any(55 <= x <= 78 or 102 <= x <= 135 for x in angles):
    tags.append("skewed")
  if any(x >= 120 for x in angles) or any("sharp" in e["kind"] for e in turns):
    tags.append("sharp")
  if len(ex) >= 3:
    tags.append("three_ways")
  if any(e["kind"].split()[0] in ("fork", "ramp", "exit") for e in ex):
    tags.append("fork_ramp")
  if any(e.get("street") is None for e in ex):
    tags.append("unnamed_out")
  if any(e["kind"].startswith("left") for e in ex):
    tags.append("left")
  w = {"lanes": 1.0, "bay": 0.7, "bend_in": 1.0, "skewed": 0.8, "sharp": 0.5, "three_ways": 0.5, "fork_ramp": 0.5,
       "unnamed_out": 0.4, "left": 0.3}
  return 1.0 + sum(w[t] for t in tags), tags


# *** held out and left out ***

def eval_points(path: str) -> list[dict]:
  """The junctions of an e2e trips file: (name, turn point, spec) from lines 'NAME spec # ... at (x,y) ...'."""
  out = []
  for line in open(os.path.expanduser(path)):
    body, _, comment = line.partition("#")
    parts = body.split()
    at = re.search(r"at \((-?[\d.]+),\s*(-?[\d.]+)\)", comment)
    if len(parts) >= 2 and ">" in parts[1] and at:
      out.append({"name": parts[0], "spec": parts[1], "turn": [float(at.group(1)), float(at.group(2))]})
  return out


def bad_spots(labels_root: str) -> list[tuple[float, float, str]]:
  """Where the expert collided, left the route, swerved or drove in the oncoming lanes before: one point per stretch
  the labeller rejected for it (labels.npz reject bits, the labels root manifest's reason order)."""
  root = Path(os.path.expanduser(labels_root))
  man = json.loads((root / "manifest.json").read_text())
  bits = {r: 1 << i for i, r in enumerate(man["summary"].get("reason_order") or [])}
  out = []
  for e in man["segments"]:
    p = root / e["segment"] / "labels.npz"
    if not p.exists():
      continue
    lab = np.load(p)
    rej, cam = lab["reject"], lab["camera_xy"]
    for reason in ("collision", "off_route", "oncoming", "swerve"):
      if reason not in bits:
        continue
      on = (rej & bits[reason]) != 0
      for k in np.flatnonzero(on & ~np.concatenate(([False], on[:-1]))):
        if np.all(np.isfinite(cam[k])):
          out.append((float(cam[k][0]), float(cam[k][1]), reason))
  return out


# *** the plan ***

def est_seconds(length: float) -> float:
  return EST_OVERHEAD + length / EST_SPEED


def tour(items: list[dict], start=(0.0, 0.0)) -> list[dict]:
  left, out, p = list(items), [], np.asarray(start, float)
  while left:
    k = min(range(len(left)), key=lambda i: math.hypot(left[i]["start"]["x"] - p[0], left[i]["start"]["y"] - p[1]))
    a = left.pop(k)
    out.append(a)
    p = np.array([a["start"]["x"], a["start"]["y"]])
  return out


def verify(router, a: dict) -> dict | None:
  """The approach with its routed ways out (kind, length, geometry), or None with fewer than two."""
  kept = []
  for e in a["exits"]:
    r = route_check(router, a["start"], e["dest"], a["via_in"], e["via_out"], max(a["before"], BEFORE), e["after"])
    e["check"] = r.get("why", "ok")
    if not r["ok"]:
      continue
    if any(math.hypot(r["exit_point"][0] - k["exit_point"][0], r["exit_point"][1] - k["exit_point"][1]) < EXIT_KEEP for k in kept):
      continue
    kept.append({**e, **{k: r[k] for k in ("length", "time", "s_in", "s_out", "kind", "angle", "change", "bend", "maneuvers",
                                             "exit_point", "geom")}})
  if len(kept) < 2:
    return None
  # the road's bend on the way in from its routes (GTA's links bend more over the 250 m from the start)
  return {**a, "bend_in": int(np.median([e.pop("bend") for e in kept])), "exits": sorted(kept, key=lambda e: e["change"])}


def make_plan(args) -> dict:
  from openpilot.tools.sim.bridge.gta5 import e2e
  e2e.MAP_DIR = args.map
  t0 = time.monotonic()
  m = e2e.Map()
  router = make_router(args.map, args.router, args.valhalla)
  print(f"map {args.map} loaded in {time.monotonic() - t0:.0f} s; routing on {args.valhalla or args.router or 'localhost:8002'}", flush=True)
  rng = random.Random(args.seed)
  cls = clusters(m)
  print(f"{len(cls)} junctions on GTA's links", flush=True)

  # eval: the A/B trips' junctions, from each trip's own start
  fail = set(args.fail_trips.split(",")) if args.fail_trips else set()
  evals, eval_ids, eval_missing = [], set(), {}
  for ep in (eval_points(args.eval_trips) if args.eval_trips else []):
    c = min(cls, key=lambda c: math.hypot(c["x"] - ep["turn"][0], c["y"] - ep["turn"][1]))
    if math.hypot(c["x"] - ep["turn"][0], c["y"] - ep["turn"][1]) > 30.0 + c["size"]:
      eval_missing[ep["name"]] = "no junction at its turn"
      continue
    eval_ids.add(id(c))  # held out even when it can't be driven
    a = eval_approach(m, router, c, ep["spec"])
    v = verify(router, a) if a is not None else None
    if v is None:
      # its route on this map misses the junction (or leaves under two ways out from its start): the road it was
      # made for, from a start of our own
      (sx, sy, *_), _ = e2e.parse_spec(ep["spec"])
      b_spec = math.degrees(math.atan2(ep["turn"][0] - sx, ep["turn"][1] - sy)) % 360
      alts = [x for x in candidates(m, c) + candidates(m, c, SHORT_BEFORE, SHORT_SLACK, SHORT_MIN)
              if abs(wrap(x["bearing_in"] - b_spec)) < 45]
      for x in sorted(alts, key=lambda x: abs(wrap(x["bearing_in"] - b_spec))):
        v = verify(router, x)
        if v is not None:
          v["own_start"] = "its route misses the junction" if a is None else "under two ways out from its start"
          break
    if v is None:
      eval_missing[ep["name"]] = ("its route misses the junction" if a is None else "under two legal ways out from its start (" +
                                  ", ".join(e.get("check", "-") for e in a["exits"]) + ")") + "; none from a start of our own"
      continue
    v.update(split="eval", name=ep["name"], reps=args.fail_reps if ep["name"] in fail else args.eval_reps,
             junction=[round(c["x"], 1), round(c["y"], 1)], jsize=round(c["size"], 1))
    evals.append(v)
  for name, why in eval_missing.items():
    print(f"eval {name}: {why}", flush=True)
  held = [c for c in cls if id(c) in eval_ids]
  eval_xy = np.array([[c["x"], c["y"]] for c in held]) if held else np.zeros((0, 2))

  bad = bad_spots(args.bad_labels) if args.bad_labels and os.path.exists(os.path.expanduser(args.bad_labels)) else []
  bad_xy = np.array([[x, y] for x, y, _ in bad]) if bad else np.zeros((0, 2))
  bad_kind = [k for _, _, k in bad]

  def badness(c) -> str | None:
    if not len(bad_xy):
      return None
    near = np.flatnonzero(np.hypot(*(bad_xy - [c["x"], c["y"]]).T) < BAD_NEAR + c["size"])
    kinds = Counter(bad_kind[i] for i in near)
    if kinds["collision"] or kinds["off_route"] or kinds["oncoming"] + kinds["swerve"] >= 2:
      return ", ".join(f"{k} {n}" for k, n in kinds.items())
    return None

  skipped = Counter()
  cands = []
  for c in cls:
    if id(c) in eval_ids:
      continue
    if len(eval_xy) and np.hypot(*(eval_xy - [c["x"], c["y"]]).T).min() < EVAL_BUFFER:
      skipped["near an eval junction"] += 1
      continue
    if not e2e.in_city(c["x"], c["y"]) and rng.random() > args.map_share:
      skipped["outside the city"] += 1
      continue
    if badness(c):
      skipped["the expert had trouble there"] += 1
      continue
    long = {a["bearing_in"]: a for a in candidates(m, c)}
    short = {a["bearing_in"]: a for a in candidates(m, c, SHORT_BEFORE, SHORT_SLACK, SHORT_MIN)}
    for b_in, a in (long | {b: s for b, s in short.items() if b not in long}).items():
      alt = short.get(b_in)
      for x in (a, alt):
        if x is not None:
          x["junction"], x["jsize"] = [round(c["x"], 1), round(c["y"], 1)], round(c["size"], 1)
      # a rough kind from GTA's links alone, to route the likeliest first
      pre = 1.0 * (a["lanes_in"] >= 2) + 0.7 * a["bay"] + 0.5 * (abs(a["bend_in"]) >= 25) + 0.5 * (len(a["exits"]) >= 3)
      cands.append((pre + rng.random() * 0.8, a, alt if alt is not None and alt["start"] != a["start"] else None))
  cands.sort(key=lambda t: -t[0])
  print(f"{len(cands)} candidate approaches; left out: {dict(skipped)}", flush=True)
  per_trip = est_seconds(BEFORE + AFTER + 20)
  eval_secs = sum(a["reps"] * sum(est_seconds(e["length"]) for e in a["exits"]) for a in evals)
  train_ways_wanted = max(0.0, (args.hours * 3600 - eval_secs) / per_trip / args.train_reps)
  pool, checks = [], Counter()
  t1 = time.monotonic()
  for _, a, alt in cands:
    if sum(len(p["exits"]) for p in pool) >= args.pool_factor * train_ways_wanted or time.monotonic() - t1 > args.max_route_s:
      break
    v = verify(router, a)
    if v is None and alt is not None:
      v = verify(router, alt)
      if v is not None:
        checks["from the nearer start"] += 1
    for e in a["exits"] + (alt["exits"] if alt else []):
      checks[e.get("check", "-")] += 1
    if v is None:
      skipped["under two legal ways out"] += 1
      continue
    v["score"], v["tags"] = kind_score(v)
    pool.append(v)
  print(f"routed {len(pool)} approaches with two or more ways out in {time.monotonic() - t1:.0f} s; way-out checks {dict(checks)}", flush=True)
  for a in evals:
    a["score"], a["tags"] = kind_score(a)

  # pick: kind first, spread over the city, until the hours are filled
  chosen, cells, per_junction, starts = [], Counter(), Counter(), []
  budget = args.hours * 3600 - eval_secs
  left = list(pool)

  def cell(a):
    return int(a["junction"][0] // AREA_CELL), int(a["junction"][1] // AREA_CELL)

  def worth(a):
    return a["score"] / (1.0 + 0.7 * cells[cell(a)]) / (1.0 + 1.5 * per_junction[tuple(a["junction"])])
  while left and budget > 0:
    a = max(left, key=worth)
    left.remove(a)
    if any(math.hypot(a["start"]["x"] - s[0], a["start"]["y"] - s[1]) < TOO_CLOSE for s in starts):
      continue
    a.update(split="train", reps=args.train_reps)
    chosen.append(a)
    starts.append((a["start"]["x"], a["start"]["y"]))
    cells[cell(a)] += 1
    per_junction[tuple(a["junction"])] += 1
    budget -= args.train_reps * sum(est_seconds(e["length"]) for e in a["exits"])
  spare = sorted(left, key=lambda a: -a["score"])[:args.spares]
  for a in spare:
    a.update(split="spare", reps=0)

  approaches = tour(chosen) + tour(evals)
  for i, a in enumerate(approaches):
    a["id"] = f"{'E' if a['split'] == 'eval' else 'J'}{i:03d}"
    for j, e in enumerate(a["exits"]):
      e["id"] = f"{a['id']}x{j}"
  for i, a in enumerate(spare):
    a["id"] = f"S{i:03d}"
    for j, e in enumerate(a["exits"]):
      e["id"] = f"{a['id']}x{j}"

  trips = []
  passes_n = max([a["reps"] for a in approaches] + [0]) + args.extra_passes
  for p in range(passes_n):
    for a in approaches:
      reps = a["reps"] + (args.extra_passes if a["split"] == "train" else 0)
      if p >= reps:
        continue
      n = len(a["exits"])
      for j in range(n):
        k = (j + p) % n
        e = a["exits"][k]
        lane = (0 if (p + k) % 2 == 0 else 9) if a["lanes_start"] >= 2 else 9
        s = a["start"]
        trips.append({"id": f"{e['id']}p{p}", "approach": a["id"], "exit": e["id"], "pass": p, "split": a["split"],
                      "extra": p >= a["reps"], "lane": lane,
                      "spec": f"{s['x']:.1f},{s['y']:.1f},{s['z']:.1f},{s['heading']:.0f},{lane}>{e['dest'][0]:.1f},{e['dest'][1]:.1f}",
                      "est_s": round(est_seconds(e["length"]))})
  return {"made": time.strftime("%Y-%m-%dT%H:%M:%S"), "map": args.map,
          "pbf_bytes": os.path.getsize(os.path.join(args.map, "gta5.osm.pbf")), "router": args.valhalla or args.router, "seed": args.seed,
          "settings": {k: getattr(args, k) for k in ("hours", "train_reps", "eval_reps", "fail_reps", "extra_passes",
                                                     "eval_trips", "fail_trips", "bad_labels", "map_share")},
          "constants": {"before": BEFORE, "short_before": SHORT_BEFORE, "after": AFTER, "est_overhead": EST_OVERHEAD,
                        "est_speed": EST_SPEED, "eval_buffer": EVAL_BUFFER},
          "bad_spots": len(bad), "left_out": dict(skipped), "eval_missing": eval_missing,
          "held_out_junctions": [[round(c["x"], 1), round(c["y"], 1)] for c in held],
          "approaches": approaches, "spares": spare, "trips": trips}


def summary(plan: dict, hours: float | None = None) -> str:
  trips = plan["trips"]
  hours = hours or plan["settings"]["hours"]
  lines = []
  for split in ("train", "eval"):
    a_s = [a for a in plan["approaches"] if a["split"] == split]
    if not a_s:
      continue
    ways = sum(len(a["exits"]) for a in a_s)
    kinds = Counter(e["kind"] for a in a_s for e in a["exits"])
    tags = Counter(t for a in a_s for t in a.get("tags", []))
    lens = [e["length"] for a in a_s for e in a["exits"]]
    lines += [f"{split}: {len(a_s)} approaches at {len({tuple(a['junction']) for a in a_s})} junctions, {ways} ways out " +
              f"({ways / len(a_s):.1f} each); lanes in >= 2 on {sum(a['lanes_in'] >= 2 for a in a_s)}, start lanes >= 2 on " +
              f"{sum(a['lanes_start'] >= 2 for a in a_s)}",
              f"  ways out: {', '.join(f'{k} {n}' for k, n in kinds.most_common())}",
              f"  kinds: {', '.join(f'{k} {n}' for k, n in tags.most_common())}",
              f"  trip length {min(lens)}-{max(lens)} m (median {np.median(lens):.0f}); approach {np.median([a['before'] for a in a_s]):.0f} m " +
              f"median (min {min(a['before'] for a in a_s)})"]
  secs = np.array([t["est_s"] for t in trips])
  fit = int(np.searchsorted(np.cumsum(secs), hours * 3600))
  done = trips[:fit]
  lines.append(f"trips: {len(trips)} planned ({sum(not t['extra'] for t in trips)} + {sum(t['extra'] for t in trips)} in extra passes), " +
               f"{np.mean(secs):.0f} s each estimated: {3600 / np.mean(secs):.1f} trips/h, {fit} in {hours:g} h")
  for split in ("train", "eval"):
    d = [t for t in done if t["split"] == split]
    ways = Counter(t["exit"] for t in d)
    a_s = [a for a in plan["approaches"] if a["split"] == split]
    full = sum(all(ways[e["id"]] >= 1 for e in a["exits"]) for a in a_s)
    lines.append(f"  in {hours:g} h, {split}: {len(d)} trips, {len(ways)} ways out driven ({sum(ways.values())} passes), " +
                 f"{full}/{len(a_s)} approaches with every way out; ways out by passes {dict(sorted(Counter(ways.values()).items()))}")
  lines.append(f"  at 75% arrived (overnight1: 74% on 0.5-3 km trips): ~{int(0.75 * fit)} junction passes, " +
               f"~{int(0.75 * sum(t['split'] == 'train' for t in done))} for training")
  lines.append(f"left out: {plan.get('left_out')}; expert trouble spots from labels: {plan.get('bad_spots')}")
  if plan.get("eval_missing"):
    lines.append(f"eval junctions held out but not driven: {plan['eval_missing']}")
  ev = [a for a in plan["approaches"] if a["split"] == "eval"]
  if ev:
    lines.append("eval: " + "; ".join(f"{a['name']} x{a['reps']} {'/'.join(e['kind'] for e in a['exits'])}" for a in ev))
  return "\n".join(lines)


def figure(plan: dict, path: str):
  import matplotlib
  matplotlib.use("Agg")
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(figsize=(10, 12))
  colours = {"train": "#1f77b4", "eval": "#d62728", "spare": "#bbbbbb"}
  for a in plan.get("spares", []) + plan["approaches"]:
    for e in a["exits"]:
      g = np.array(e["geom"])
      ax.plot(g[:, 0], g[:, 1], lw=0.8, color=colours[a["split"]], alpha=0.8)
    ax.plot(*a["junction"], "o", ms=3, color=colours[a["split"]])
    if a["split"] == "eval":
      ax.annotate(a.get("name", a["id"]), a["junction"], fontsize=7, color=colours["eval"])
  for p in plan.get("held_out_junctions", []):
    ax.plot(*p, "x", ms=5, color=colours["eval"])
  ax.set_aspect("equal")
  ax.set_title("junction recording plan: train (blue), eval (red), spare (grey)")
  fig.savefig(path, dpi=130, bbox_inches="tight")


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest="command", required=True)
  pl = sub.add_parser("plan")
  pl.add_argument("--out", required=True)
  pl.add_argument("--map", default=DEFAULT_MAP, help="the map folder (paths.jsonl, gta5.osm.pbf): the live one or a copy")
  pl.add_argument("--router", help="Valhalla's URL (default http://localhost:8002, the bridge's)")
  pl.add_argument("--valhalla", help="a valhalla.json to route in process instead (a copy of the map's, its tile folder)")
  pl.add_argument("--hours", type=float, default=8.0)
  pl.add_argument("--train-reps", type=int, default=2, help="passes over the training approaches")
  pl.add_argument("--eval-reps", type=int, default=1)
  pl.add_argument("--fail-reps", type=int, default=2, help="for --fail-trips' junctions")
  pl.add_argument("--extra-passes", type=int, default=1, help="more training passes after those, for a fast night")
  pl.add_argument("--eval-trips", default=f"{HOME}/gta5test/e2e/short-ab.txt", help="e2e trips whose junctions are held out")
  pl.add_argument("--fail-trips", default="SL7,SO05,SO06,SO10,SR1,SR4,SR6", help="those failing in every arm of the A/Bs")
  pl.add_argument("--bad-labels", default=f"{HOME}/git/gta5-train/out/labels_overnight1",
                  help="a labels root: junctions where the expert was rejected for collisions, off route, oncoming are left out")
  pl.add_argument("--map-share", type=float, default=0.15, help="of junctions outside the city considered")
  pl.add_argument("--pool-factor", type=float, default=2.5, help="routed ways out, against those wanted")
  pl.add_argument("--max-route-s", type=float, default=1800.0, help="s of routing candidates at most")
  pl.add_argument("--spares", type=int, default=30, help="routed approaches kept unpicked, for swapping in")
  pl.add_argument("--seed", type=int, default=1)
  pl.add_argument("--fig", help="a map of the plan (png)")
  sh = sub.add_parser("show")
  sh.add_argument("plan")
  sh.add_argument("--hours", type=float)
  sh.add_argument("--fig")
  a = p.parse_args()
  if a.command == "plan":
    plan = make_plan(a)
    Path(a.out).write_text(json.dumps(plan, indent=1))
    print(f"wrote {a.out}")
  else:
    plan = json.loads(Path(a.plan).read_text())
  print(summary(plan, getattr(a, "hours", None) if a.command == "show" else None))
  if a.fig:
    figure(plan, a.fig)


if __name__ == "__main__":
  sys.exit(main())
