"""navd_sim, the planner's closed-loop harness, on made routes (no live map): its stand-ins for the model's lane change
(speed-dependent), the lane head's noise, the blind spot from traffic, and the judging of maneuvers."""
import numpy as np

from openpilot.tools.sim.bridge.gta5 import mapdrive_sim as ms
from openpilot.tools.sim.bridge.gta5 import navd_sim as ns
from openpilot.tools.sim.bridge.gta5.mapdrive_sim import LEFT, RIGHT


def slow_turn(side: str = "right", lead: float = 400.0, limit: float = 7.0, ours: int = 2):
  """A two-way road of `ours` lanes each way, north `lead` m at `limit` m/s, then a turn at a junction with lights."""
  pts = ms.line((lead, 0.0), (200.0, 90.0 if side == "left" else -90.0))
  corner = int(round(lead / 10.0))
  return ms.made_route(pts, ms.section(ours, 2), junctions=[corner], turns={corner: LEFT if side == "left" else RIGHT},
                       stops=[(lead - 12.0, "lights")], limit=limit)


def drive(route, lane: int, model: str = "n16", head: str = "truth", density: float = 0.0, seed: int = 1, **kw):
  sim = ns.NavSim(route, lane, ns.PRESETS[model], ns.parse_head(head), ns.TrafficCfg(density=density), seed=seed, **kw)
  return sim, sim.run()


def test_rate_by_speed():
  m = ns.PRESETS["n16"]
  assert m.rate(15.0, 5.0) == 1 / 5.0 and m.rate(10.0, 5.0) == 1 / 5.0
  assert m.rate(5.0, 5.0) == 0.0 and m.rate(3.0, 5.0) == 0.0
  assert 0.0 < m.rate(7.5, 5.0) < 0.3 / 5.0
  f = ns.PRESETS["finetuned"]
  assert f.rate(5.0, 5.0) == 1 / 5.0 and f.rate(2.0, 5.0) == 0.0


def test_city_turn_made_from_afar():
  """Two changes over for a turn 400 m on at city speed: both done well before the slowing, the turn made."""
  for side, lane in (("right", 0), ("left", 2)):
    r = ms.junction_turn(side, lead=400.0, tail=200.0, ours=3, back=2, stop="lights")
    for model in ("n16", "finetuned"):
      sim, res = drive(r, lane, model)
      assert res.error is None and res.arrived
      turn = [m for m in res.maneuvers if m["kind"] == "turn"]
      assert len(turn) == 1 and turn[0]["made"] and turn[0]["relevant"] and turn[0]["need"] == 2, turn
      assert [c["outcome"] for c in res.changes] == ["done", "done"]
      assert all(c["start_v"] > 10.0 for c in res.changes)


def test_slow_road_needs_the_finetune():
  """At 7 m/s n16's lane change doesn't get across in desire_helper's 10 s; the finetune's does, and the turn is made."""
  r = slow_turn("right", limit=7.0)
  _, n16 = drive(r, 0, "n16")
  _, fine = drive(r, 0, "finetuned")
  t16 = [m for m in n16.maneuvers if m["kind"] == "turn"][0]
  tf = [m for m in fine.maneuvers if m["kind"] == "turn"][0]
  assert not t16["made"] and tf["made"]
  assert n16.changes and all(c["outcome"] != "done" for c in n16.changes)
  assert fine.changes[0]["outcome"] == "done"


def test_turn_lane_out():
  """Out of a right turn in the road out's right lane, a left turn's left lane."""
  for side, want in (("right", 1), ("left", 0)):
    r = ms.junction_turn(side, lead=300.0, tail=200.0, ours=2, back=2, stop=None)
    sim, res = drive(r, 1 if side == "right" else 0, "n16")
    assert res.maneuvers[0]["made"]
    assert int(round(sim.lane)) == want


def test_blind_spot_holds_the_change():
  """A vehicle held alongside in the lane the change is into: the blinker stays on, the change waits (hanging), and
  starts once it has gone by."""
  r = ms.junction_turn("right", lead=600.0, tail=200.0, ours=2, back=2, stop=None)
  sim = ns.NavSim(r, 0, ns.PRESETS["n16"], ns.HeadCfg("truth"), ns.TrafficCfg(density=0.0), seed=1)
  # alongside a little behind, as fast as the car for a while, then dropping back
  sim.traffic.vs.append(ns.Vehicle(sim.s - 3.0, 1.0, sim.v, 1.0))
  orig = sim.traffic.step

  def step(t, dt, car, reset):
    for o in sim.traffic.vs:
      o.want = 1.0 if t < 26.0 else 0.4
    orig(t, dt, car, reset)
  sim.traffic.step = step
  res = sim.run()
  c = res.changes[0]
  assert c["hanging"] > 3.0 and c["outcome"] == "done", res.changes
  assert res.maneuvers[0]["made"] and res.collisions == 0


def test_lane_head_noise():
  rng = np.random.default_rng(0)
  head = ns.LaneHead(ns.HeadCfg("model", acc=0.8, count_acc=1.0, lag=0.5, drop=0.1, dwell=1.0), rng)
  out = [head.step(k * 0.1, [1, 3], 0.1) for k in range(20000)]
  seen = [o for o in out if o is not None]
  assert 0.85 < len(seen) / len(out) < 0.95
  right = np.mean([o[0] == 1 for o in seen])
  assert 0.75 < right < 0.85
  assert all(o[1] == 3 and 0 <= o[0] < 3 for o in seen)
  # lagged: a change of lane shows lag s later
  head = ns.LaneHead(ns.HeadCfg("model", acc=1.0, count_acc=1.0, lag=0.5, drop=0.0), rng)
  got = [head.step(k * 0.1, [0, 2] if k < 10 else [1, 2], 0.1) for k in range(20)]
  assert [g[0] for g in got].index(1) == 15


def test_model_lane_source_without_truth():
  """With a perfect lane head and no truth at all, nav plans the same changes and makes the turn."""
  r = ms.junction_turn("right", lead=400.0, tail=200.0, ours=3, back=2, stop="lights")
  _, res = drive(r, 0, "n16", head="model:1.0:0.3:0.0:1.0")
  assert res.maneuvers[0]["made"] and [c["outcome"] for c in res.changes] == ["done", "done"]


def test_summary():
  r = ms.junction_turn("right", lead=400.0, tail=200.0, ours=3, back=2, stop="lights")
  _, a = drive(r, 0, "n16")
  _, b = drive(slow_turn("right", limit=7.0), 0, "n16")
  from dataclasses import asdict
  s = ns.summarise([asdict(a), asdict(b)])
  assert s["relevant"] == 2 and s["in_lane"] == 1 and s["missed"] == 1
  assert s["lc_done"] == 2 and s["lc_requested"] >= 3 and s["lc_late"] >= 1
  assert "in lane" in ns.table([("x", s)])
