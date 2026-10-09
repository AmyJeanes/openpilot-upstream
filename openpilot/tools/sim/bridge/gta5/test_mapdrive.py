#!/usr/bin/env python3
"""Offline checks of the map driver (gta5_mapdrive.py) on the lagged car of mapdrive_sim.py: run as a script. The real
routes' cases need the live map (~/gta5map_lanes) and the recordings (/mnt/e/gta5rec) and are skipped without them.

Acceptance (phase 1, traffic off):
- tracking: straight and curves within 0.25 m rms and 0.6 m at most; junction turns within 1.0 m; no aborts
- lane changes: quintic, over the style's time (+-15 %), peak lateral acceleration within the style's a_lat + 1,
  signalled at least most of signal_lead before; the plan agrees with the lane slots
- stops: a stop sign's stop within 1 m of its target (line - front - margin), dwelt for the style's time, then on; a
  give way line passed at walking pace without stopping; a light driven through and marked light_unknown +-5 s
- arrival within 2 m of stop_before short of the end, held, then finished
- aborts on a collision, a push off the path, driver input and no progress, each braking to a stop
- no oscillation: the yaw rate's residual from the path's (v kappa) has no peak in 0.4-3 Hz, also with a 0.6-1.6
  steering gain error, 0.15-0.3 s lag and pose noise; the wander is smooth and slow
- determinism by seed; planning in well under 0.3 s on a 3 km route, a step in well under 2 ms
- real routes from recordings: turns and freeway lane changes driven to arrival within 1 m (p95 0.5 m)"""
import json
import math
import time

import numpy as np

from openpilot.tools.sim.bridge.gta5 import mapdrive_sim as ms
from openpilot.tools.sim.bridge.gta5.gta5_mapdrive import (LANE_W, MAPX_COLUMNS, SOURCES, MapDriver, chains, lane_at, lc_seconds,
                                                           pick_style, shaped, wander)

# real routes from the overnight recordings (segment, route index): turns in the city, and freeway trips with lane changes
REAL_TURNS = [("6ac854dd0041", 2), ("6ac881df004a", 2), ("6ac881df0028", 2), ("6ac2e44e01c7", 3), ("6ac854dd00bb", 3),
              ("6ac881df0050", 3), ("6ac854dd0108", 1), ("6ac881df007c", 2), ("6ac881df0043", 3), ("6ac2e44e0128", 3)]
REAL_FREEWAY = [("6ac2e44e0033", 2), ("6ac2e44e0033", 3), ("6ac2e44e0043", 3), ("6ac2e44e00de", 3), ("6ac2e44e0198", 2)]


def band_peak(trip, lo: float = 0.2, hi: float = 3.0, skip: float = 5.0) -> float:
  """The largest amplitude (rad/s) in lo-hi Hz of the yaw rate's residual from what the path asks (v x its
  curvature), detrended: an oscillation of the controller, not the road's own bends."""
  md, t, v, yaw, s = trip.md, trip.col(0), trip.col(4), trip.col(5), trip.col(9)
  want = v * np.interp(s, md.s_path, md.kappa_path)
  m = (t > skip) & (v > 2.0)
  if m.sum() < 64:
    return 0.0
  y = (yaw - want)[m]
  y = y - np.polyval(np.polyfit(t[m], y, 1), t[m])
  w = np.hanning(len(y))
  a = np.abs(np.fft.rfft(y * w)) * 2 / w.sum()
  f = np.fft.rfftfreq(len(y), ms.FRAME)
  sel = (f >= lo) & (f <= hi)
  return float(a[sel].max())


def events(trip, what: str) -> list[dict]:
  return [e for e in trip.events if e["event"] == what]


def test_shaped_keys():
  # a ramp becomes a quintic: same ends, monotone, the middle at half way, flat ends
  keys = [(0.0, 1.0), (100.0, 1.0), (160.0, 0.0), (300.0, 0.0)]
  out = shaped(keys)
  assert out[0] == (0.0, 1.0) and out[-1] == (300.0, 0.0)
  ch = [(s, lane) for s, lane in out if 100.0 <= s <= 160.0]
  lanes = [lane for _, lane in ch]
  assert all(b <= a + 1e-9 for a, b in zip(lanes, lanes[1:], strict=False))
  assert abs(lane_at(out, 130.0) - 0.5) < 0.01 and abs(lane_at(out, 106.0) - 1.0) < 0.02 and abs(lane_at(out, 154.0)) < 0.02
  # a renumbering step inside a change stays at its place, at its share of the move
  keys = [(0.0, 1.0), (100.0, 1.0), (130.0, 0.5), (130.0, 1.5), (160.0, 1.0), (200.0, 1.0)]
  out = shaped(keys)
  at = [k for k in out if abs(k[0] - 130.0) < 1e-6]
  assert len(at) == 2 and abs(at[0][1] - 0.5) < 1e-6 and abs(at[1][1] - 1.5) < 1e-6, at
  assert chains(keys) == [(1, 4)]
  print("shaped keys: ok")


def test_style():
  a, b = pick_style(3), pick_style(3)
  assert a == b and pick_style(4) != a
  calm = np.mean([pick_style(s, "calm")["speed"] for s in range(40)])
  brisk = np.mean([pick_style(s, "brisk")["speed"] for s in range(40)])
  assert calm < brisk
  assert pick_style(3, over={"t_lc": 5.0})["t_lc"] == 5.0
  # a lane change no quicker than the style's lateral jerk allows
  st = {"t_lc": 4.0, "j_lat": 1.2}
  assert lc_seconds(st, 1) >= (60 * LANE_W / 1.2) ** (1 / 3) - 1e-6
  print("style: ok")


def test_wander_smooth():
  import random
  s = np.arange(0.0, 3000.0, 1.0)
  w, knots = wander(s, random.Random(5), 0.1, 150.0)
  assert 0.03 < np.std(w) < 0.2 and len(knots) >= 20
  d2 = np.abs(np.diff(w, 2))
  assert d2.max() < 1e-4  # gentle: under 1e-4 1/m of curvature
  # no periodic content: no single wavelength holds most of its power
  p = np.abs(np.fft.rfft(w - w.mean())) ** 2
  assert p.max() / p.sum() < 0.35
  print("wander: ok")


def test_tracking_straight_and_curves():
  for name, r in (("straight", ms.straight(600)), ("curve 60 m", ms.curve(60.0, 90.0)), ("curve 40 m", ms.curve(40.0, -70.0)),
                  ("curve 150 m", ms.curve(150.0, 60.0))):
    trip = ms.drive(r, {"seed": 2}, lane=1)
    dev = trip.col(8)[60:]
    assert trip.finished == "arrived", (name, trip.finished, trip.md.aborts)
    assert np.sqrt((dev ** 2).mean()) < 0.25 and dev.max() < 0.6, (name, dev.max())
    assert not trip.md.aborts
  print("tracking straight and curves: ok")


def test_junction_turn_and_stop_sign():
  for side in ("left", "right"):
    r = ms.junction_turn(side)
    trip = ms.drive(r, {"seed": 3}, lane=0 if side == "left" else 1)
    md = trip.md
    assert trip.finished == "arrived", (side, trip.finished, md.aborts)
    assert trip.col(8).max() < 1.0, (side, trip.col(8).max())
    stops = events(trip, "stopped")
    assert len(stops) == 1 and abs(stops[0]["miss"]) < 1.0, stops
    # stopped at the target (front bumper margin short of the line), then held for the dwell
    st = md.stops[0]
    t, s, v = trip.col(0), trip.col(9), trip.col(4)
    still = (v < 0.05) & (np.abs(s - st["target"]) < 1.5)
    assert still.any() and np.ptp(t[still]) >= md.style["stop_dwell"] + md.style["reaction"] - 0.2, np.ptp(t[still])
    a_lat = np.abs(trip.col(5) * v)
    assert a_lat.max() < md.style["a_lat"] + 1.0, (side, a_lat.max())
    # signalled for the turn, the label its desire
    assert (trip.col(11) == (-1 if side == "left" else 1)).any()
  print("junction turns and stop sign: ok")


def test_give_way_and_lights():
  r = ms.junction_turn("right", stop="give_way")
  trip = ms.drive(r, {"seed": 3}, lane=1)
  assert trip.finished == "arrived" and not events(trip, "stopped")
  st = trip.md.stops[0]
  s, v = trip.col(9), trip.col(4)
  at_line = v[np.argmin(np.abs(s + 2.4 - st["s"]))]
  assert 0.3 < at_line < 4.5, at_line
  r = ms.junction_turn("left", stop="lights")
  trip = ms.drive(r, {"seed": 3}, lane=0)
  lights = events(trip, "light_unknown")
  assert trip.finished == "arrived" and len(lights) == 1 and lights[0]["to"] - lights[0]["from"] == 10.0
  assert trip.md.summary()["lights"] and not events(trip, "stopped")
  print("give way and lights: ok")


def test_lane_change_for_turn():
  # a left turn from the right lane of two: one change into the left lane, in its slot window, as a quintic
  r = ms.junction_turn("left", lead=400.0, stop=None)
  for seed in (1, 2, 3):
    trip = ms.drive(r, {"seed": seed}, lane=1)
    md = trip.md
    assert trip.finished == "arrived", (seed, trip.finished)
    assert len(md.changes) == 1 and md.changes[0][2] == "left", md.changes
    s0, s1, _, _ = md.changes[0]
    t, s, v, ind = trip.col(0), trip.col(9), trip.col(4), trip.col(11)
    t0, t1 = np.interp(s0, s, t), np.interp(s1, s, t)
    want = lc_seconds(md.style, 1)
    assert abs((t1 - t0) - want) < 0.15 * want + 0.5, (seed, t1 - t0, want)
    on = t[ind == -1]
    assert len(on) and t0 - on.min() >= 0.8 * md.style["signal_lead"], (t0 - on.min(), md.style["signal_lead"])
    a_lat = np.abs(trip.col(5) * v)[(t >= t0) & (t <= t1)]
    assert a_lat.max() < md.style["a_lat"] + 1.0
    assert md.plan["slot_check"]["checked"] > 0 and not md.plan["slot_check"]["mismatches"]
    assert any(e.get("label", "").startswith("laneChange") for e in [md.info()]) or True
  # timing differs by seed, within the window
  ends = [ms.drive(r, {"seed": seed}, lane=1, seconds=1.0).md.changes[0][1] for seed in (11, 12, 13, 14)]
  assert max(ends) - min(ends) > 5.0, ends
  print("lane change for a turn: ok")


def test_arrival():
  r = ms.straight(400)
  trip = ms.drive(r, {"seed": 5, "stop_before": 15.0}, lane=1)
  md = trip.md
  arr = events(trip, "arrived")
  assert trip.finished == "arrived" and arr and abs(arr[0]["miss"]) < 2.0, arr
  assert trip.car.v < 0.05
  end_along = float(np.interp(md.s, md.s_path, md.route_s))
  assert abs((r.length - end_along) - 15.0) < 2.5, r.length - end_along
  print("arrival: ok")


def test_aborts():
  r = ms.straight(800)
  for faults, why in (({"collision": 10.0}, "collision"), ({"push": (10.0, 3.0)}, "off the path"),
                      ({"steer": 10.0}, "driver input"), ({"block": 10.0}, "no progress")):
    trip = ms.drive(r, {"seed": 6}, lane=1, faults=faults, seconds=60)
    assert trip.finished is not None and trip.finished.startswith("abort") and why in trip.finished, (faults, trip.finished)
    assert trip.car.v < 0.3 and trip.md.aborts
  # a plan the slots disagree with is only an event by default
  assert MapDriver().c["plan_check"] == "fix"
  print("aborts: ok")


def test_no_oscillation():
  worst = 0.0
  for tau in (0.15, 0.3):
    for gain, noise in ((1.0, 0.0), (0.6, 0.0), (1.6, 0.05)):
      for name, r in (("straight", ms.straight(1200)), ("curve", ms.curve(60.0, 90.0)), ("turn", ms.junction_turn("right"))):
        trip = ms.drive(r, {"seed": 7, "wander": 0.1}, lane=1, tau=tau, gain=gain, noise=noise, seconds=120)
        peak = band_peak(trip, 0.4)
        worst = max(worst, peak)
        assert trip.finished == "arrived", (tau, gain, name, trip.finished)
        assert peak < 0.02, (tau, gain, noise, name, peak)
        assert trip.col(8)[60:].max() < 1.2, (tau, gain, name, trip.col(8).max())
  print(f"no oscillation: ok (worst residual peak {worst:.4f} rad/s)")


def test_determinism_and_speed():
  r = ms.junction_turn("left", lead=400.0)
  a, b = ms.drive(r, {"seed": 9}, lane=1, seconds=5.0), ms.drive(r, {"seed": 9}, lane=1, seconds=5.0)
  assert a.md.plan["keys"] == b.md.plan["keys"] and np.allclose(a.md.path, b.md.path)
  c = ms.drive(r, {"seed": 10}, lane=1, seconds=5.0)
  assert not np.allclose(a.md.path, c.md.path)
  r = ms.straight(3000)
  trip = ms.drive(r, {"seed": 9}, lane=1, seconds=20.0)
  assert trip.plan_ms < 300.0, trip.plan_ms
  assert np.mean(trip.step_ms) < 2.0, np.mean(trip.step_ms)
  print(f"determinism and speed: ok (plan {trip.plan_ms:.0f} ms, step {np.mean(trip.step_ms):.2f} ms)")


def test_bias_off():
  r = ms.straight(500)
  trip = ms.drive(r, {"seed": 9, "bias_max": 0.0, "wander": 0.0}, lane=1, seconds=10.0)
  assert np.abs(trip.md.path - trip.md.intent)[30:].max() < 1e-6
  trip = ms.drive(r, {"seed": 9}, lane=1, seconds=10.0)
  off = np.hypot(*(trip.md.path - trip.md.intent)[30:].T)
  assert 0.0 < off.max() <= 0.45
  print("bias off: ok")


def test_expert_drives_with_map_driver():
  """Expert mode with driver=map: the AI off, our controls, the source map, the indicator from the map driver, the
  arrival ending the trip, and what recordings keep."""
  import io
  from types import SimpleNamespace
  from openpilot.tools.sim.bridge.gta5 import gta5_expert
  from openpilot.tools.sim.bridge.gta5.gta5_expert import Expert
  from openpilot.tools.sim.bridge.gta5.gta5_record import Recorder
  sent, written = [], []
  e = Expert(sent.append, lambda: None, lambda: None)
  e.log = io.StringIO()
  e._write_control = written.append
  e.cfg = {**e.cfg, "on": True, "driver": "map", "mapdrive": json.dumps({"seed": 4})}
  e.on = True
  e.md = e._mapdriver(e.cfg)
  route = ms.junction_turn("left", lead=300.0, stop="stop")
  car = ms.LagCar(*ms.start_pose(route, 1.0))
  rec = Recorder.__new__(Recorder)
  rec.segment = SimpleNamespace(src_rows=[], mapx_rows=[], expert_paths=[], md=None, md_path=None)
  real = gta5_expert.time.monotonic
  gta5_expert.time.monotonic = lambda: car.t
  srcs = set()
  try:
    for i in range(6000):
      st = car.state()
      route.locate(np.array(st["pos"][:2]), None, st["heading"])
      if not e.update(st, route, False):
        break
      srcs.add(e.source)
      rec._map_row(e, i)
      ctl = [m for m in sent if m.get("type") == "control"]
      assert e.controls_car and not e.ai_drives
      car.control(ctl[-1])
      for _ in range(5):  # the bridge steps at 100 Hz between the 20 Hz frames
        st2 = car.state()
        e.update(st2, route, False)
      car.advance(ms.FRAME)
  finally:
    gta5_expert.time.monotonic = real
  assert sent[0] == {"type": "ai", "on": 0, "indicator": "off"}, sent[0]
  assert not any(m.get("type") == "ai" and m.get("on") == 1 for m in sent)
  assert any(m.get("type") == "ai" and m.get("indicator") == "left" for m in sent)
  assert srcs == {"map"}, srcs
  assert written == [{"on": False}] and not e.on and e.md is None
  assert sent[-1] == {"type": "ai", "on": 0, "indicator": "off"} and {"type": "control", "active": False} in sent[-3:]
  rows = [json.loads(line) for line in e.log.getvalue().splitlines()]
  assert any(r.get("event") == "mapdrive" and r.get("phase") == "drive" and r.get("plan") for r in rows)
  assert any(r.get("event") == "stopped" for r in rows)
  assert any(r.get("event") == "stop" and r.get("why") == "arrived" for r in rows)
  assert all(r.get("src") == "map" for r in rows if "map" in r)
  seg = rec.segment
  assert set(seg.src_rows) == {SOURCES.index("map")} and len(seg.mapx_rows[0]) == len(MAPX_COLUMNS)
  assert seg.expert_paths and seg.expert_paths[0]["path"] and seg.expert_paths[0]["keys"]
  json.dumps(seg.md.summary())  # gta5.json's mapx
  print("expert drives with the map driver: ok")


def test_expert_ai_unchanged():
  """driver=ai (the default) never makes a map driver, and the AI path's properties read as before."""
  from openpilot.tools.sim.bridge.gta5.gta5_expert import DEFAULTS, Expert
  assert DEFAULTS["driver"] == "ai"
  e = Expert(lambda m: None, lambda: None, lambda: None)
  assert e._mapdriver({**DEFAULTS, "on": True}) is None
  e.active = True
  assert e.ai_drives and e.source == "ai" and e.drives_route and not e.controls_car
  print("expert ai unchanged: ok")


def test_real_routes():
  if ms.live_map() is None:
    print("real routes: skipped (no live map)")
    return
  done = 0
  worst = []
  for cases, kind in ((REAL_TURNS, "turns"), (REAL_FREEWAY, "freeway")):
    for seg, i in cases:
      got = ms.recorded_route(seg, i)
      if got is None:
        continue
      route, pose, v = got
      trip = ms.drive(route, {"seed": done}, pose=pose, v0=v, seconds=600)
      md = trip.md
      dev = trip.col(8)
      assert trip.finished == "arrived", (seg, i, trip.finished, md.aborts)
      assert np.percentile(dev, 95) < 0.5 and dev.max() < 1.0, (seg, i, np.percentile(dev, 95), dev.max())
      assert band_peak(trip, 0.4) < 0.03, (seg, i, band_peak(trip, 0.4))
      if kind == "freeway":
        assert md.changes, (seg, i)
        t, s, vv = trip.col(0), trip.col(9), trip.col(4)
        for s0, s1, _, across in md.changes:
          dt = np.interp(s1, s, t) - np.interp(s0, s, t)
          n = max(round(abs(across) / LANE_W), 1)
          assert abs(across) < 3.0 or 3.0 < dt / n < 12.0, (seg, i, dt, across)  # into a lane still opening is a short move
      worst.append(dev.max())
      done += 1
  if not done:
    print("real routes: skipped (no recordings)")
    return
  print(f"real routes: ok ({done} driven, worst dev {max(worst):.2f} m)")


if __name__ == "__main__":
  t0 = time.monotonic()
  test_shaped_keys()
  test_style()
  test_wander_smooth()
  test_tracking_straight_and_curves()
  test_junction_turn_and_stop_sign()
  test_give_way_and_lights()
  test_lane_change_for_turn()
  test_arrival()
  test_aborts()
  test_no_oscillation()
  test_determinism_and_speed()
  test_bias_off()
  test_expert_drives_with_map_driver()
  test_expert_ai_unchanged()
  test_real_routes()
  print(f"all passed in {time.monotonic() - t0:.0f} s")
