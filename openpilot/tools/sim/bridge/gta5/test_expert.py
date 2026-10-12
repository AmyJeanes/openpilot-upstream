#!/usr/bin/env python3
"""Offline checks of expert mode's targets around forks and GTA's shortcut links (gta5_expert.py), the router's U-turn
checks (map/router.py) and junction_plan.py recheck, on made-up routes: run as a script."""
import io
import json
import math
from types import SimpleNamespace

import numpy as np

from openpilot.tools.sim.bridge.gta5 import gta5_expert
from openpilot.tools.sim.bridge.gta5.gta5_expert import FORK_PAST, USE_SHORTCUTS, Expert
from openpilot.tools.sim.bridge.gta5.map.router import Fork, Route, Router, sharp_turns

STYLE = 1075845291
CFG = {"on": True, "task": "coord", "speed": 18, "ramp": 1.0, "lead": 3, "launch": 1.5, "decel": 0.8, "turn_speed": 4,
       "arrive": "gentle", "targets": "smooth", "ahead_min": 100, "ahead_max": 160, "speed_step": 0.2, "style": STYLE,
       "unstick": 1}


def straight(length: float = 1200.0, step: float = 12.0) -> Route:
  """A road east from the origin, a node every `step` m."""
  xs = np.arange(0.0, length + 1e-6, step)
  return Route(np.stack([xs, np.zeros_like(xs)], axis=1), [{"type": 1, "begin_shape_index": 0},
                                                          {"type": 4, "begin_shape_index": len(xs) - 1}])


def with_shortcut(route: Route, k: int) -> Route:
  route.links = [None] * (len(route.points) - 1)
  route.links[k] = SimpleNamespace(shortcut=True, lanes=0)
  return route


def drive(route: Route, cfg: dict | None = None, speed: float = 15.0):
  """Expert mode along the route at a steady speed: what it sent, with route.at when it sent it."""
  sent = []
  e = Expert(lambda m: sent.append((m, route.at)), lambda: None, lambda: None)
  e.log = io.StringIO()
  e.cfg = {**e.cfg, **CFG, **(cfg or {})}
  e.on = True
  clock = [0.0]
  real = gta5_expert.time.monotonic
  gta5_expert.time.monotonic = lambda: clock[0]
  try:
    x, t = 0.0, 0
    while x < route.length - 30:
      route.locate(np.array([x, 0.0]))
      state = {"t": t, "pos": [x, 0.0, 0.0], "vEgo": speed, "heading": 270.0, "ai": {"on": True, "aborts": 0},
               "collisions": 0}
      e.update(state, route, False)
      x, t, clock[0] = x + speed * 0.5, t + 1, clock[0] + 0.5
  finally:
    gta5_expert.time.monotonic = real
  return e, sent


def targets(sent) -> list[tuple[float, float, dict]]:
  """(m along the route of each target sent, route.at then, the message)"""
  return [(m["x"], at, m) for m, at in sent if m.get("type") == "ai" and "x" in m]


def test_plain_route_unchanged():
  """No fork or shortcut link: targets as before, and never the shortcut flag."""
  route = straight()
  e, sent = drive(route)
  assert all(not (m.get("style", 0) & USE_SHORTCUTS) for m, _ in sent), [m for m, _ in sent if "style" in m]
  assert all("style" not in m for _, _, m in targets(sent))
  ref = Expert(lambda m: None, lambda: None, lambda: None)
  ref.cfg = {**ref.cfg, **CFG}
  for x, at, _ in targets(sent):
    probe = straight()
    probe.at = at
    ref.events = []
    want = ref._pick_at(probe)[0]
    assert abs(x - want) < 0.2, (x, want, at)
  print(f"plain route: ok, {len(targets(sent))} targets as before")


def test_shortcut_link():
  """A shortcut link on the route: UseShortCutLinks with the target that first lies past it, a target past it by
  FORK_PAST rather than in or short of it, and the flag off again once past."""
  route = with_shortcut(straight(), 50)  # 600-612 m
  s0, s1 = float(route.along[50]), float(route.along[51])
  e, sent = drive(route)
  styles = [(m["style"], at) for m, at in sent if "style" in m and "x" in m]
  assert styles and styles[0][0] == STYLE | USE_SHORTCUTS, styles
  on_at = styles[0][1]
  assert s0 - 160 - FORK_PAST - 1 <= on_at < s0 - 100, (on_at, s0)
  assert styles[-1][0] == STYLE and styles[-1][1] > s1 + 20, styles
  for x, _, _ in targets(sent):
    assert not (s0 - 25 <= x < s1 + FORK_PAST), f"a target at {x:.0f} m, at the shortcut ({s0:.0f}-{s1:.0f})"
  assert any(abs(x - (s1 + FORK_PAST)) < 1 for x, _, _ in targets(sent))
  rows = [json.loads(r) for r in e.log.getvalue().splitlines()]
  assert [r["on"] for r in rows if r.get("event") == "shortcuts"] == [True, False]
  assert any(r.get("shortcuts") for r in rows if "event" not in r)
  e2, sent2 = drive(with_shortcut(straight(), 50), {"shortcuts": "off"})
  assert all(not (m.get("style", 0) & USE_SHORTCUTS) for m, _ in sent2)
  print(f"shortcut link: ok, flag on from {on_at:.0f} m to {styles[-1][1]:.0f} m (link {s0:.0f}-{s1:.0f} m)")


def test_fork():
  """A fork the route keeps left at: the target lands past it where the branches have parted; UseShortCutLinks only
  with shortcuts=forks and on a freeway."""
  route = straight()
  route.forks = [Fork(500.0, "left", 2, 2, True)]
  e, sent = drive(route)
  xs = [x for x, _, _ in targets(sent)]
  assert not any(475 <= x < 500 + FORK_PAST for x in xs), xs
  assert any(abs(x - (500 + FORK_PAST)) < 1 for x in xs), xs
  assert all("style" not in m for _, _, m in targets(sent))
  route = straight()
  route.forks = [Fork(500.0, "left", 2, 2, True)]
  route.classes = ["motorway"] * (len(route.points) - 1)
  _, sent = drive(route, {"shortcuts": "forks"})
  styles = [(m["style"], at) for m, at in sent if "style" in m and "x" in m]
  assert [s for s, _ in styles] == [STYLE | USE_SHORTCUTS, STYLE], styles
  route.classes = ["primary"] * (len(route.points) - 1)
  _, sent = drive(route, {"shortcuts": "forks"})
  assert not [m for m, _ in sent if "style" in m and "x" in m]
  route = straight()
  route.forks = [Fork(500.0, "right", 1, 1, False)]  # GTA splitting a road's lanes before a junction: the car keeps lane
  e, _ = drive(route)
  assert e.parts == [], e.parts
  print("fork: ok")


def test_unstick_keeps_shortcuts():
  e = Expert(lambda m: None, lambda: None, lambda: None)
  e.cfg = {**e.cfg, **CFG, "unstick_style": 7}
  e.shortcuts_on = True
  assert e._style() == STYLE | USE_SHORTCUTS
  e.unstick_from = (0.0, 0.0)
  assert e._style() == 7 | USE_SHORTCUTS
  print("unstick style: ok")


def turn_route(turn_deg: float, via_link: bool = False) -> Route:
  """100 m east, then a turn of turn_deg (left positive) onto a road 100 m long; via_link: as a left onto an unnamed
  median link, then another left 12 m on, the two making the turn."""
  pts = [(x, 0.0) for x in np.arange(0.0, 100.1, 10.0)]
  h = math.radians(-90.0 + turn_deg)  # game heading -90 is east; left turns add
  def go(p, heading, d):
    return (p[0] - math.sin(heading) * d, p[1] + math.cos(heading) * d)
  mans = [{"type": 1, "begin_shape_index": 0}]
  if via_link:
    mid = go(pts[-1], math.radians(0.0), 12.0)  # north across the median
    mans.append({"type": 15, "begin_shape_index": len(pts) - 1})
    pts.append(mid)
    mans.append({"type": 15, "begin_shape_index": len(pts) - 1, "street_names": ["Back St"]})
    p = mid
  else:
    mans.append({"type": 15 if turn_deg > 0 else 10, "begin_shape_index": len(pts) - 1})
    p = pts[-1]
  for _ in range(10):
    p = go(p, h, 10.0)
    pts.append(p)
  mans.append({"type": 4, "begin_shape_index": len(pts) - 1})
  return Route(np.array(pts), mans)


def test_sharp_turns():
  assert sharp_turns(turn_route(90.0)) == []
  assert sharp_turns(turn_route(-120.0)) == []
  assert len(sharp_turns(turn_route(160.0))) == 1
  assert len(sharp_turns(turn_route(180.0, via_link=True))) == 1, "two lefts across a median"
  r = turn_route(90.0)
  r.maneuvers[1]["type"] = 13
  assert len(sharp_turns(r)) == 1
  print("sharp turns: ok")


def test_router_avoids_uturns():
  """A route with a U-turn is swapped for an alternative without, and marked so."""
  from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon

  def encode(points):
    out, last = [], (0, 0)
    for x, y in points:
      lat, lon = to_lat_lon(x, y)
      cur = (round(lat * 1e6), round(lon * 1e6))
      for v in (cur[0] - last[0], cur[1] - last[1]):
        v = ~(v << 1) if v < 0 else v << 1
        while v >= 0x20:
          out.append(chr((0x20 | (v & 0x1f)) + 63))
          v >>= 5
        out.append(chr(v + 63))
      last = cur
    return "".join(out)

  def trip(route):
    return {"legs": [{"shape": encode(route.points), "maneuvers": route.maneuvers}], "summary": {"time": 60}}
  bad, good = turn_route(180.0, via_link=True), turn_route(90.0)
  calls = []

  class Fake(Router):
    def _post(self, action, request):
      calls.append(request)
      if action == "trace_attributes":
        return {"edges": []}
      if "alternates" in request:
        return {"trip": trip(bad), "alternates": [{"trip": trip(bad)}, {"trip": trip(good)}]}
      return {"trip": trip(bad)}

  r = Fake("http://none").route(np.array([0.0, 0.0]), 90.0, np.array([100.0, 100.0]))
  assert r.uturns_avoided and not sharp_turns(r), sharp_turns(r)
  assert np.allclose(r.points[-1], good.points[-1], atol=0.5)
  assert any("alternates" in c for c in calls)

  class Plain(Router):
    def _post(self, action, request):
      return {"edges": []} if action == "trace_attributes" else {"trip": trip(good)}
  r = Plain("http://none").route(np.array([0.0, 0.0]), 90.0, np.array([100.0, 100.0]))
  assert not r.uturns_avoided and len(r.points) == len(good.points)
  print("router U-turns: ok")


def test_recheck():
  """recheck: a way out failing the U-turn check is dropped; a training approach left with one is swapped for a spare,
  its trips where the dropped approach's were; every other trip as it was."""
  from openpilot.tools.sim.bridge.gta5 import junction_plan as jp

  def approach(aid, x, n=2, split="train"):
    return {"id": aid, "split": split, "reps": 2, "start": {"x": x, "y": 0.0, "z": 0.0, "heading": 270}, "via_in": [[0, 0], [1, 0]],
            "before": 250, "lanes_start": 2, "lanes_in": 2, "bay": False, "bend_in": 0, "junction": [x + 250, 0.0],
            "exits": [{"id": f"{aid}x{j}", "dest": [x + 400, j * 10.0], "via_out": [[0, 0], [1, 0]], "after": 180,
                       "length": 430, "kind": "straight", "angle": 0, "street": "A"} for j in range(n)]}
  plan = {"settings": {"train_reps": 2, "extra_passes": 1},
          "approaches": [approach("J000", 0.0), approach("J001", 1000.0, 3), approach("E002", 2000.0, 2, "eval")],
          "spares": [approach("S000", 1010.0), approach("S001", 5000.0)]}
  plan["trips"] = []
  for p in range(3):
    for a in plan["approaches"]:
      if a["split"] == "eval" and p > 0:
        continue
      plan["trips"] += jp.pass_trips(plan, a, p, 0)
    plan["trips"].append({"id": f"W000p{p}", "approach": "W000", "exit": "W000x0", "pass": p, "split": "wrongway"})
  bad = {"J000x1", "J001x2", "E002x0"}

  def fake_check(router, start, dest, via_in, via_out, before, after):
    hit = [e["id"] for a in plan["approaches"] + plan["spares"] for e in a["exits"] if e["dest"] == dest and a["start"] == start]
    return {"ok": False, "why": "U-turn"} if hit and hit[0] in bad else {"ok": True, "shortcuts": 0}

  def fake_verify(router, a):
    return {**a, "exits": a["exits"]}
  real = jp.route_check, jp.verify
  jp.route_check, jp.verify = fake_check, fake_verify
  try:
    new, rep = jp.recheck(plan, None)
  finally:
    jp.route_check, jp.verify = real
  assert rep["dropped_ways"].keys() == bad, rep
  assert rep["dropped_approaches"] == ["J000"] and rep["replaced"] == {"J000": "S001"}, rep  # S000 starts too near J001
  ids = [t["id"] for t in new["trips"]]
  old = [t["id"] for t in plan["trips"] if t["exit"] not in bad and t["approach"] != "J000"]
  assert [i for i in ids if not i.startswith("S")] == old
  assert ids[:2] == ["S001x0p0", "S001x1p0"], ids[:4]
  assert sum(i.startswith("S001") for i in ids) == 6 and "E002x1p0" in ids
  assert [a["id"] for a in new["approaches"]] == ["J001", "E002", "S001"] and [s["id"] for s in new["spares"]] == ["S000"]
  assert rep["trips_dropped"] == 6 + 3 + 1 and rep["trips_added"] == 6
  print(f"recheck: ok, {rep['trips_before']} -> {rep['trips_after']} trips")


if __name__ == "__main__":
  test_plain_route_unchanged()
  test_shortcut_link()
  test_fork()
  test_unstick_keeps_shortcuts()
  test_sharp_turns()
  test_router_avoids_uturns()
  test_recheck()
