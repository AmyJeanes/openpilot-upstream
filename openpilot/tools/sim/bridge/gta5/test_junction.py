from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.tools.sim.bridge.gta5 import record_run as rr
from openpilot.tools.sim.bridge.gta5.junction_plan import passes
from openpilot.tools.sim.bridge.gta5.junction_run import PlanPicker, exit_code


def plan(n_approaches=2, n_exits=2, passes_n=2) -> dict:
  approaches, trips = [], []
  for i in range(n_approaches):
    a = {"id": f"J{i}", "split": "train", "junction": [i * 500.0, 0.0], "start": {"x": i * 500.0, "y": -250.0, "z": 0.0, "heading": 0},
         "exits": [{"id": f"J{i}x{j}", "kind": ["left square", "straight", "right square"][j], "dest": [i * 500.0 + j, 180.0],
                    "length": 430, "time": 60, "maneuvers": [], "geom": [[0, 0], [0, 1]]} for j in range(n_exits)]}
    approaches.append(a)
  for p in range(passes_n):
    for a in approaches:
      for j in range(n_exits):
        e = a["exits"][(j + p) % n_exits]
        trips.append({"id": f"{e['id']}p{p}", "approach": a["id"], "exit": e["id"], "pass": p, "split": "train",
                      "extra": False, "lane": 9, "spec": "0,0,0,0,9>0,180", "est_s": 100})
  return {"approaches": approaches, "trips": trips}


def args(**kw):
  return SimpleNamespace(name="t", max_tries=2, approach_fails=3, **kw)


def drive_all(picker, outcome=lambda t: "arrived", limit=100) -> list[str]:
  order = []
  with pytest.raises(rr.Stop, match="plan done"):
    for _ in range(limit):
      c = picker.next()
      order.append(c["plan_trip"])
      picker.record({"plan_trip": c["plan_trip"], "outcome": outcome(c["plan_trip"])})
  return order


def test_plan_order_and_done():
  p = plan()
  picker = PlanPicker(p, args(), [], None, lambda rec: None)
  assert drive_all(picker) == [t["id"] for t in p["trips"]]
  assert picker.progress()["arrived"] == len(p["trips"])


def test_resume_skips_done_and_retries_failed_last():
  p = plan()
  done = [{"plan_trip": "J0x0p0", "outcome": "arrived"}, {"plan_trip": "J0x1p0", "outcome": "stuck"}]
  picker = PlanPicker(p, args(), done, None, lambda rec: None)
  order = drive_all(picker)
  assert "J0x0p0" not in order
  assert order[-1] == "J0x1p0"  # tried once already: after every untried trip
  assert picker.seq == 2 + len(order)


def test_failed_way_out_dropped_and_infra_again_next():
  p = plan()
  infra_once = {"J1x0p0"}

  def outcome(t):
    if t.startswith("J0x1"):
      return "collisions"
    if t in infra_once:
      infra_once.discard(t)
      return "infra"
    return "arrived"
  picker = PlanPicker(p, args(), [], None, lambda rec: None)
  order = drive_all(picker, outcome)
  assert sum(t.startswith("J0x1") for t in order) == 2  # --max-tries, then the way out is dropped
  k = order.index("J1x0p0")
  assert order[k + 1] == "J1x0p0"  # not its fault: again at once
  assert picker.dropped(p["trips"][1]) == "the expert failed this way out"


def test_mismatch_skipped():
  p = plan(n_approaches=1)
  recs = []
  picker = PlanPicker(p, args(), [], lambda a, e: "not in by the approach" if e["id"] == "J0x1" else None, recs.append)
  order = drive_all(picker)
  assert all(not t.startswith("J0x1") for t in order)
  assert {r["plan_trip"] for r in recs} == {"J0x1p0", "J0x1p1"} and all(r["outcome"] == "plan_mismatch" for r in recs)


def test_exit_codes():
  assert exit_code("plan done") == 0 and exit_code("8 h up") == 0 and exit_code("stop file /x") == 0
  assert exit_code("the driver took over (engage key)") == 0
  assert exit_code("8 trips failed in a row") == 3 and exit_code("x: none for 600 s; reload the plugin by hand") == 3
  assert exit_code("low disk: {}") == 4


def test_passes_by_direction():
  # a road north into a junction at (0, 0), turning off east at its first node by a slip road
  route = np.array([[0.0, -50.0], [0.0, -20.0], [0.0, -10.0], [10.0, -2.0], [40.0, 0.0]])
  assert passes(route, [[0.0, -10.0], [0.0, 0.0]]) == 2  # in along the link, leaving at its first node
  assert passes(route, [[0.0, 10.0], [0.0, 0.0]]) is None  # the other way's link into the junction
  assert passes(route, [[5.0, 0.0], [30.0, 0.0]], 2, into=False) == 3
  assert passes(route, [[-5.0, 0.0], [-30.0, 0.0]], 2, into=False) is None
