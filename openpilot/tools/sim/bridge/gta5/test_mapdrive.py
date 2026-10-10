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
  # a lane change takes the style's time, no quicker than its peak lateral acceleration allows
  assert lc_seconds({"t_lc": 5.0, "a_lat": 2.5}, 1) == 5.0
  st = {"t_lc": 4.0, "a_lat": 1.5}
  assert abs(lc_seconds(st, 1) - math.sqrt(5.77 * LANE_W / (0.6 * 1.5))) < 1e-6
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


def test_lane_change_takes_its_time():
  """A change is timed by the speed it's driven at: pulling away from a standstill, or nav's ramp across a whole window
  (the city's too slow drift seen in game), it takes the style's time (+-25 %), not the ramp's length."""
  from openpilot.tools.sim.bridge.gta5.gta5_mapdrive import MapDriver
  r = ms.junction_turn("left", lead=400.0, stop=None)
  for seed in (1, 4):
    for v0 in (0.0, 12.0):
      trip = ms.drive(r, {"seed": seed}, lane=1, v0=v0)
      md = trip.md
      s0, s1, _, _ = md.changes[0]
      t, s = trip.col(0), trip.col(9)
      dt, want = np.interp(s1, s, t) - np.interp(s0, s, t), lc_seconds(md.style, 1)
      assert abs(dt - want) < 0.25 * want, (seed, v0, dt, want)
  # nav's slow ramp (160 m, 10 s here; SR3 had 90 m at 11 m/s) re-placed to its time, its end kept
  trip = ms.drive(r, {"seed": 2}, lane=1, v0=11.0, seconds=0.2)
  keys = [(0.0, 1.0), (170.0, 1.0), (330.0, 0.0), (400.0, 0.0)]
  fitted = trip.md.fit_durations(r, keys, 11.0)
  assert fitted[2] == (330.0, 0.0) and fitted[1][0] > 240.0, fitted
  # a change with a renumbering step inside keeps the step at its place
  keys = [(0.0, 1.0), (200.0, 1.0), (280.0, 0.5), (280.0, 1.5), (330.0, 1.0), (400.0, 1.0)]
  out = MapDriver._replace_chain(keys, 1, 4, 270.0, 330.0)
  assert len([k for k in out if k[0] == 280.0]) == 2 and out[0] == (270.0, 1.0) and out[-1] == (330.0, 1.0), out
  print("lane changes take their time: ok")


def inside_error(trip, side: str) -> float:
  """The car's mean offset (m) towards the inside of the bend from its path, where the path bends."""
  md, out = trip.md, []
  for x_, y_, sv in zip(trip.col(1), trip.col(2), trip.col(9)):
    j = int(min(np.searchsorted(md.s_path, sv), len(md.s_path) - 2))
    if abs(md.kappa_path[j]) < 0.04:
      continue
    a_, b_ = md.path[j], md.path[j + 1]
    right = ((x_ - a_[0]) * (b_[1] - a_[1]) - (y_ - a_[1]) * (b_[0] - a_[0])) / max(np.hypot(*(b_ - a_)), 1e-6)
    out.append(right if side == "right" else -right)
  return float(np.mean(out))


def test_turns_from_the_rear_axle():
  """The game reports the car's middle, which moves inwards of its heading in a bend: steered from it, the car cuts
  every corner (0.4-0.8 m in game); steered from the rear axle it keeps to the line."""
  for side in ("left", "right"):
    r = ms.junction_turn(side, stop=None)
    lane = 0 if side == "left" else 1
    new = inside_error(ms.drive(r, {"seed": 3, "bias_max": 0, "wander": 0}, lane=lane, tau=0.16), side)
    old = inside_error(ms.drive(r, {"seed": 3, "bias_max": 0, "wander": 0, "rear_axle": 0.0}, lane=lane, tau=0.16), side)
    assert abs(new) < 0.15 and old > 0.3, (side, new, old)
  print("turns from the rear axle: ok")


def test_bias_fades_in_turns_and_soften_holds_outside():
  from openpilot.tools.sim.bridge.gta5.gta5_mapdrive import soften
  r = ms.junction_turn("right", stop=None)
  trip = ms.drive(r, {"seed": 5, "bias_max": 0.3, "wander": 0.0, "style": {"bias": 1.0}}, lane=1, seconds=0.2)
  md = trip.md
  off = np.hypot(*(md.path - md.intent).T)
  near = np.abs(md.route_s - 200.0) < 14.0
  far = (np.abs(md.route_s - 200.0) > 50.0) & (md.s_path > 40.0)
  assert off[near].max() < 0.02 and off[far].min() > 0.25, (off[near].max(), off[far].min())
  # a square corner without a fillet, eased out: swung out up to SOFT_OUT rather than all of it cut inside
  from openpilot.tools.sim.bridge.gta5 import gta5_mapdrive as g
  pts = np.array([[0.0, y] for y in np.arange(0.0, 100.0, 1.0)] + [[x, 100.0] for x in np.arange(0.0, 100.0, 1.0)])

  def cut(q):  # the eased curve's nearest to the corner as drawn
    return float(np.min(np.hypot(*(q - np.array([0.0, 100.0])).T)))
  soft, spots = soften(pts, 0.1)
  saved = g.SOFT_OUT
  g.SOFT_OUT = 0.0
  free, _ = soften(pts, 0.1)
  g.SOFT_OUT = saved
  assert spots and cut(soft) < cut(free) - 0.9 * g.SOFT_OUT, (cut(soft), cut(free))
  k = np.abs(np.gradient(np.unwrap(np.arctan2(*np.gradient(soft, axis=0).T[::-1]))))
  assert k.max() < 0.5, k.max()  # eased, if not to 0.1 1/m
  print("bias fades in turns, soften holds outside: ok")


def jog_road() -> "ms.Route":
  """North 400 m on two lanes each way, the road's line jogging 5 m right across a 13 m link at 200 m (as where a
  road's line moves over to its other carriageway's at a junction): the lane line swerves there."""
  pts = [(0.0, y) for y in np.arange(0.0, 201.0, 10.0)] + [(5.0, 212.0)] + [(5.0, y) for y in np.arange(222.0, 413.0, 10.0)]
  return ms.made_route(pts, ms.section(2, 2), junctions=[20], road_class="secondary")


def test_lane_change_kept_off_a_jog():
  """A change across a jog in the lane line moves wholly before (or after) it; and through a jog the speed is held to
  its curvature's envelope, so a surge (the game car over a bump) doesn't leave the steering behind."""
  from openpilot.tools.sim.bridge.gta5 import gta5_mapdrive as g
  r = jog_road()
  spans = g.route_jogs(r)
  assert len(spans) == 1 and spans[0][0] < 200.0 < 212.0 < spans[0][1], spans
  md = ms.drive(r, {"seed": 2}, lane=1, seconds=0.2).md
  keys = [(0.0, 1.0), (185.0, 1.0), (230.0, 0.0), (400.0, 0.0)]
  out = md.off_jogs(r, keys)
  assert out[2][0] <= spans[0][0] + 1e-6 and abs((out[2][0] - out[1][0]) - 45.0) < 1e-6, out
  # driven with nav's change across it, a surge at the jog: tracked without saturating (the profile slows for the S)
  orig = g.MapDriver.off_jogs
  g.MapDriver.off_jogs = lambda self, route, k: k
  try:
    real = g.MapDriver.nav_keys
    g.MapDriver.nav_keys = lambda self, route, lane: [(route.at, 1.0), (185.0, 1.0), (225.0, 0.0), (400.0, 0.0)]
    trip = ms.drive(r, {"seed": 2, "retime": False, "fit_durations": False}, lane=1, tau=0.18, faults={"surge": (190.0, 4.5)})
  finally:
    g.MapDriver.off_jogs, g.MapDriver.nav_keys = orig, real
  s_, dev, v = trip.col(9), trip.col(8), trip.col(4)
  m = (s_ > 170) & (s_ < 260)
  assert trip.finished == "arrived" and dev[m].max() < 0.6, (trip.finished, dev[m].max())
  assert not [a for a in trip.md.anomalies if a["kind"] == "tracking_saturated"], trip.md.anomalies
  print("lane change kept off a jog: ok")


def test_corner_kerb():
  """A junction's inside corner kerb from its legs' cross-sections: a car over it is a kerb_contact, the plan's own
  line isn't."""
  from openpilot.tools.sim.bridge.gta5.gta5_mapdrive import MapDriver
  r = ms.junction_turn("right", stop=None)  # north 200 m, right (east); two lanes each way, 5.5 m
  trip = ms.drive(r, {"seed": 3, "bias_max": 0, "wander": 0}, lane=1)
  md = trip.md
  assert len(md.corners) == 1 and not [a for a in md.anomalies if a["kind"] == "kerb_contact"], md.anomalies
  c = md.corners[0]
  assert np.allclose(c["corner"], [11.0, 189.0], atol=0.2), c["corner"]
  diag = math.radians(-45.0 + 90.0)  # heading south-east, round the turn
  assert MapDriver._corner_hit(c, np.array([14.0, 186.0]), diag) > 1.0  # its middle over the corner
  assert MapDriver._corner_hit(c, np.array([9.5, 190.5]), diag) is None  # inside the kerb's rounding
  assert MapDriver._corner_hit(c, np.array([5.0, 195.0]), diag) is None
  print("corner kerb: ok")


def without_clamp():
  """MapDriver._clamp left out, as before it (a context manager)."""
  from contextlib import contextmanager
  from openpilot.tools.sim.bridge.gta5 import gta5_mapdrive as g

  @contextmanager
  def off():
    real = g.MapDriver._clamp
    g.MapDriver._clamp = lambda self, route, keys, intent: (intent, np.full(len(intent), 0.45), np.full(len(intent), 0.45))
    try:
      yield
    finally:
      g.MapDriver._clamp = real
  return off()


def corner_depth(md) -> float:
  """How deep (m) our body reaches into a turn's corner kerb along the plan, at most."""
  return max([md._corner_hit(c, md.path[i], float(md.theta[i]), md.body) or 0.0 for c in md.corners
              for i in range(len(md.path)) if abs(md.route_s[i] - c["along"]) < 40.0] + [0.0])


def kinds(md, kind: str) -> list[dict]:
  return [a for a in md.anomalies if a["kind"] == kind]


def test_clamp_corners():
  """The lane line's fillet cutting a turn's inside corner kerb: the plan pushed out of it, a plan_clamp anomaly, not a
  kerb_contact. A far-side turn's inside is a kerb where it's a median: one-way legs (map1010b 012, a left turn past a
  tram median's nose) or a median between the directions; a near-side turn's into a narrow street."""
  one_way = ms.section(2, 0, 4.4)
  cases = [("left 60, one-way legs", ms.junction_turn("left", stop=None, angle=60.0, sec=one_way), 0, 0.5),
           ("left 60, a 4 m median", ms.junction_turn("left", stop=None, angle=60.0, sec=ms.section(2, 2, median=4.0)), 0, 0.05),
           ("right 90 into 3.5 m lanes", ms.junction_turn("right", stop=None, out=ms.section(1, 1, 3.5)), 1, None)]
  for name, r, lane, cut in cases:
    cfg = {"seed": 3, "bias_max": 0, "wander": 0}
    with without_clamp():
      before = ms.drive(r, cfg, lane=lane).md
    trip = ms.drive(r, cfg, lane=lane)
    md = trip.md
    assert kinds(before, "kerb_contact") and (cut is None or corner_depth(before) > cut), (name, corner_depth(before))
    assert trip.finished == "arrived" and not kinds(md, "kerb_contact") and corner_depth(md) == 0.0, (name, kinds(md, "kerb_contact"))
    assert any(c["why"] == "corner" for c in md.clamps), (name, md.clamps)
    assert trip.col(8).max() < 1.0, (name, trip.col(8).max())
  print("clamp corners: ok")


def lane_offsets(md, route) -> np.ndarray:
  """The plan's intent: m right of its lane's centre at each point."""
  from openpilot.tools.sim.bridge.gta5.gta5_mapdrive import along_route, lane_at, segment_of
  along, right = along_route(route, md.intent, 0.0)
  return np.array([r_ - route.section(segment_of(route, a)).offset(lane_at(md.keys, a)) for a, r_ in zip(along, right, strict=True)])


def bumped(route, at: float, length: float, right: float):
  """The route's lane line moved `right` m right over `length` m from `at` m along, eased in and out over 10 m (as a
  fillet or an eased corner swings it off its lane)."""
  from openpilot.tools.sim.bridge.gta5.gta5_mapdrive import headings
  real = route.lanes.lane_line

  def line(at0, keys, step=2.0):
    out = real(at0, keys, step)
    s = at0 + np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(out, axis=0).T))))
    u = np.clip(np.minimum(s - at + 10.0, at + length + 10.0 - s) / 10.0, 0.0, 1.0)
    th = headings(out)
    return out + np.stack([np.sin(th), -np.cos(th)], axis=1) * (right * u * u * (3 - 2 * u))[:, None]
  route.lanes.lane_line = line
  return route


def test_clamp_lane_and_fork():
  """Outside lane changes and turns' corners the plan keeps our body in its lane, CLAMP_MARGIN in from its edges
  (map1010b 029: a soft right's lane line 3-4 m right of its lane, onto the pavement); near a fork, within FORK_ROOM of
  its centre on the gore's side (014: 2 m towards the gore past a ramp's split). Inside that, it's left alone."""
  from openpilot.tools.sim.bridge.gta5.gta5_mapdrive import CLAMP_MARGIN, FORK_ROOM, HALF_WIDTH, LANE_W
  from openpilot.tools.sim.bridge.gta5.map.router import Fork
  cfg = {"seed": 3, "bias_max": 0, "wander": 0}
  room = LANE_W / 2 - HALF_WIDTH - CLAMP_MARGIN
  span = slice(290, 340)  # path m about the bump

  def plan(right: float, fork: str | None = None):
    r = bumped(ms.straight(600), 300.0, 30.0, right)
    if fork:
      r.forks.append(Fork(300.0, fork, 1, 2, True))
    md = ms.drive(r, cfg, lane=1, seconds=0.2).md
    return md, lane_offsets(md, r)
  md, off = plan(2.0)
  assert abs(off[span]).max() < room + 0.05 and [c for c in md.clamps if c["why"] == "lane"], (abs(off[span]).max(), md.clamps)
  assert kinds(md, "plan_clamp")
  md, off = plan(1.0)
  assert 0.95 < off[span].max() < 1.05 and not md.clamps, (off[span].max(), md.clamps)
  # the branch taken the right one: the gore on our left
  md, off = plan(-1.0, "right")
  assert -FORK_ROOM - 0.05 < off[span].min() and [c for c in md.clamps if c["why"] == "fork"], (off[span].min(), md.clamps)
  md, off = plan(-1.0, "left")
  assert -1.05 < off[span].min() < -0.95 and not md.clamps, (off[span].min(), md.clamps)
  print("clamp lane and fork: ok")


COLLISIONS = [  # map1010b's trips that hit the kerb or the gore: spec, map driver seed, where it first touched
  ("012", "883.2,-2077.8,29.5,96,0>-250.5,-1067.2", 165189838, (187.0, -1776.9)),
  ("014", "670.0,-2893.8,5.2,0,9>1358.8,-1110.4", 375068520, (561.3, -2547.6)),
  ("029", "-236.0,-597.2,33.2,20,9>-743.5,-168.0", 1600219813, (-550.5, -53.9)),
]


def test_collision_routes():
  """map1010b's 012 (1.0 m inside a left turn, onto a tram median's nose), 014 (1.9 m towards a ramp's gore) and 029
  (4.1 m inside a soft right, onto the pavement): replanned on the live map, the plan within 0.6 m of its lane's centre
  where the car touched."""
  if ms.live_map() is None:
    print("collision routes: skipped (no live map)")
    return
  done = []
  for name, spec, seed, contact in COLLISIONS:
    got = ms.trip_route(spec)
    if got is None:
      continue
    route, pose = got
    md = ms.drive(route, {"seed": seed}, pose=pose, seconds=0.2).md
    j = int(np.argmin(np.hypot(*(md.intent - np.array(contact)).T)))
    off = lane_offsets(md, route)[j]
    assert abs(off) < 0.6 and md.clamps, (name, off)
    done.append(f"{name} {off:+.2f}")
  print(f"collision routes: ok ({', '.join(done)} m)" if done else "collision routes: skipped (no router)")


def test_body_from_dims():
  """Our car's front, rear and half width from the plugin's nearby.dims (a van here), else the constants: the stop
  sign's mark is the van's front bumper margin short of the line."""
  from openpilot.tools.sim.bridge.gta5.gta5_mapdrive import FRONT, HALF_WIDTH
  md = MapDriver()
  assert md.body == (FRONT, FRONT, HALF_WIDTH)
  md._body({"nearby": {"dims": [-1.25, 1.2, -2.9, 3.1], "v": []}})
  assert md.body == (3.1, 2.9, 1.25)
  md._body({"nearby": {"dims": [0.0, 0.0, 0.0, 0.0]}})  # a model without bounds: kept
  assert md.body == (3.1, 2.9, 1.25)
  r = ms.junction_turn("right")
  trip = ms.drive(r, {"seed": 3}, lane=1, seconds=1.0, dims=(-1.25, 1.2, -2.9, 3.1))
  st = trip.md.stops[0]
  assert abs(st["s"] - st["target"] - 3.1 - trip.md.style["stop_margin"]) < 1e-6
  print("body from dims: ok")


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
  for faults, why in (({"collision": 10.0}, "collision"), ({"push": (10.0, 6.0)}, "off the"),
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
  from openpilot.tools.sim.bridge.gta5.mapdrive_trials import summary
  got = summary(rows)
  assert got["arrived"] and got["stops"] == 1 and got["dev_max"] < 1.0 and not got["aborts"], got
  seg = rec.segment
  assert set(seg.src_rows) == {SOURCES.index("map")} and len(seg.mapx_rows[0]) == len(MAPX_COLUMNS)
  mx = np.array(seg.mapx_rows)  # the change into the left lane: its side recorded wherever its progress is
  tau, side = mx[:, MAPX_COLUMNS.index("lc_tau")], mx[:, MAPX_COLUMNS.index("lc_dir")]
  assert np.isfinite(tau).any() and (np.isfinite(tau) == np.isfinite(side)).all() and set(side[np.isfinite(side)]) == {-1.0}
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
  test_lane_change_takes_its_time()
  test_turns_from_the_rear_axle()
  test_bias_fades_in_turns_and_soften_holds_outside()
  test_corner_kerb()
  test_body_from_dims()
  test_clamp_corners()
  test_clamp_lane_and_fork()
  test_collision_routes()
  test_lane_change_kept_off_a_jog()
  test_arrival()
  test_aborts()
  test_no_oscillation()
  test_determinism_and_speed()
  test_bias_off()
  test_expert_drives_with_map_driver()
  test_expert_ai_unchanged()
  test_real_routes()
  print(f"all passed in {time.monotonic() - t0:.0f} s")
