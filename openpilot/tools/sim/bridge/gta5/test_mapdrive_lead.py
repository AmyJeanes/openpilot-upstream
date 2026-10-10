#!/usr/bin/env python3
"""Offline checks of the map driver's handling of other vehicles (gta5_mapdrive.py, the plugin's nearby) on the lagged
car of mapdrive_sim.py: run as a script.

- a stopped vehicle in our lane ahead, on a straight and through a turn: stopped behind it, LEAD_STOP from it, without a
  touch, braking no harder than the style's decel where it was seen in time; lead_gap recorded
- a vehicle parked partly in our lane: stopped behind it; one parked clear of the lane, or in the oncoming lane through a
  turn, driven past
- held by a vehicle that stays in the way: the trip ends after WAIT_MAX s (PARKED_WAIT s for a parked one), and nothing
  leaves the abort; behind a queue that moves on now and then, however long it all takes, it's driven to arrival
- static things the plugin's probes hit on the path (nearby.obst): stopped short of, LEAD_STOP from it, and the trip ends
  after OBST_WAIT s; walls along the road's edges are no obstacles
- a slower vehicle ahead: followed at its speed, LEAD_STOP + the style's headway behind, without hunting; one braking to
  a stop: stopped behind it
- a parked vehicle where the plan's lane room is no guide: at the end of a lane change into its lane, passed within the
  change's lanes; past a junction whose sections are no guide, within our lane's edges carried across it
- a driven vehicle standing across our way (side on or head on): no lead to queue behind; passed in our lane where it
  pokes in, else stopped for, waited for while it moves, and a deadlock ends the trip after DEADLOCK_WAIT s
- a vehicle crossing the junction ahead as we'd reach it: held for, then on; one that's through first, or comes long
  after, not; held at the junction's line, not in the lanes short of the crosser's; stopped in a junction, held where we
  are (no creeping on) while a crosser, or one slowed or standing beside our way, is about
- pedestrians (the plugin's nearby "p"): one walking across the road we turn into as we'd reach it is waited for (mdlead
  traffic trip 1 hit one there at 5 m/s); without the list, braked for from the plugin's count of those just ahead, too
  late to miss it in the turn, but slower;
  one standing in our lane is stopped behind until it walks off; one on the pavement, walking along it, driven past
- what the plugin reports reaches only so far: without vehicles in view, no faster than stops for one at its edge (but
  for range_share of the speed); none of it where the plugin reports nothing"""
import math
import time

import numpy as np

from openpilot.tools.sim.bridge.gta5 import mapdrive_sim as ms
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, FORWARD, Lane, Section, Span
from openpilot.tools.sim.bridge.gta5.gta5_mapdrive import (CLAMP_MARGIN, DEADLOCK_WAIT, LANE_W, LEAD_STOP, MAPX_COLUMNS, NUDGE_CLEAR,
                                                           NUDGE_MARGIN, NUDGE_NODE, NUDGE_WANT, OBST_WAIT, PARKED_WAIT, WAIT_MAX, MapDriver)

REACH = {"ahead": 120.0, "side": 40.0}  # the plugin's reach (core.cpp NEARBY_AHEAD, NEARBY_SIDE_AHEAD)


def phases(trip, what: str) -> list[dict]:
  return [e for e in trip.events if e["event"] == "mapdrive" and e.get("phase") == what]


def nudges(trip) -> list[dict]:
  return [e for e in trip.events if e["event"] == "nudge"]


def parked(route, lane: float, along: float, right: float = 0.0, turned: float = 0.0, **kw) -> ms.Vehicle:
  x, y, h = ms.start_pose(route, lane, along)
  hr = math.radians(h)
  return ms.Vehicle(x + math.cos(hr) * right, y + math.sin(hr) * right, h + turned, **kw)


def stopped_behind(trip, decel: float | None = None, clear_min: float = LEAD_STOP - 1.0):
  v, clear, gap = trip.col(4), trip.col(13), trip.col(12)
  assert trip.car.collisions == 0 and clear.min() > clear_min, (trip.finished, clear.min())
  assert v[-1] < 0.1 and abs(gap[-1] - LEAD_STOP) < 1.0, (v[-1], gap[-1])
  assert phases(trip, "follow"), trip.events[:5]
  if decel is not None:
    assert trip.col(7).min() > -decel - 0.3, (trip.col(7).min(), decel)


def test_stopped_ahead():
  r = ms.straight(600)
  trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[parked(r, 1, 250.0)], reach=REACH, seconds=45)
  stopped_behind(trip)
  # seen at the plugin's reach, from the city's speed: stopped within DECEL_MAX; from a standstill, within the style's
  trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[parked(r, 1, 50.0)], reach=REACH, seconds=30)
  stopped_behind(trip, decel=trip.md.style["decel"])
  row = trip.md.mapx_row()
  assert abs(row[MAPX_COLUMNS.index("lead_gap")] - LEAD_STOP) < 1.0, row
  print("stopped ahead: ok")


def test_stopped_in_a_turn():
  for side, lane in (("right", 1), ("left", 0)):
    r = ms.junction_turn(side, stop=None)
    trip = ms.drive(r, {"seed": 3}, lane=lane, vehicles=[parked(r, lane, 212.0)], reach=REACH, seconds=60)
    stopped_behind(trip, decel=trip.md.style["decel"], clear_min=1.5)  # bodies nearer at the corners, part way round
    # stopped part way round the corner: its lead was along the path, not straight ahead
    assert abs(trip.col(3)[-1] - trip.col(3)[0]) > 15.0, trip.col(3)[-1]
    # parked in the oncoming lane of the road out: not in our way
    trip = ms.drive(r, {"seed": 3}, lane=lane, vehicles=[parked(r, -1, 212.0, turned=180.0)], reach=REACH, seconds=60)
    assert trip.finished == "arrived" and not phases(trip, "follow") and trip.car.collisions == 0, (side, trip.finished)
  print("stopped in a turn: ok")


def test_parked_partly_in_lane():
  r = ms.straight(600)
  # its near side 1.75 m into our 5.5 m lane, within LEAD_SIDE of our body (map1010b 016: a classic at the kerb, our right
  # front into it): without the nudge, stopped behind
  trip = ms.drive(r, {"seed": 2, "nudge": False}, lane=1, vehicles=[parked(r, 1, 200.0, right=2.0, driven=False)], reach=REACH, seconds=40)
  stopped_behind(trip)
  # 1 m into it (NUDGE_CLEAR from our body, by the trip's bias: shifted to NUDGE_WANT), or clear of it: driven past
  for right in (2.75, 4.4):
    trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[parked(r, 1, 200.0, right=right, driven=False)], reach=REACH, seconds=60)
    assert trip.finished == "arrived" and not phases(trip, "follow") and trip.col(13).min() > 0.3, (right, trip.finished, trip.col(13).min())
    assert all(e["ok"] for e in nudges(trip)) and (right < 4.0 or not nudges(trip)), nudges(trip)
  # in the next lane: not ours
  trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[parked(r, 0, 200.0)], reach=REACH, seconds=60)
  assert trip.finished == "arrived" and not phases(trip, "follow"), trip.finished
  print("parked partly in lane: ok")


def ended(trip, why: str):
  """The trip ended on one abort, for `why`, and nothing came after it but its end (an abort is terminal)."""
  assert trip.finished == f"abort: {why}" and len(trip.md.aborts) == 1, (trip.finished, trip.md.aborts[:3])
  after = [e.get("phase") for e in trip.events if e["event"] == "mapdrive"]
  after = after[after.index("abort") + 1:]
  assert after == ["end"], after[:6]
  assert trip.car.v < 0.1 and trip.car.collisions == 0, (trip.car.v, trip.car.collisions)


def kerbs(route, right: float) -> list[np.ndarray]:
  """Walls along the route's line `right` m either side of it (kerbs, parapets)."""
  pts = route.points[:, :2]
  d = np.gradient(pts, axis=0)
  d = d / np.hypot(*d.T)[:, None]
  normal = np.stack([d[:, 1], -d[:, 0]], axis=1)
  return [pts + normal * right, pts - normal * right]


def test_static_obstacles():
  """Static things the plugin's probes hit ahead (nearby.obst; map1010c 005, 015 and 017 hit a wall, a parapet's end and
  a lamp post the plan led into, at 3-6 m/s, which nothing in the driver could see): one on the path is stopped short of
  as a stopped vehicle is, LEAD_STOP from it, and the trip ends OBST_WAIT s on; without the probes, it's hit. Walls along
  the road's edges, on a straight and round bends, are no obstacles."""
  r = ms.straight(600)
  x, y, _ = ms.start_pose(r, 1, 150.0)
  post = [np.array([[x - 0.3, y], [x + 0.3, y]])]  # in the middle of our lane, as 017's lamp post
  cfg = {"seed": 2, "speed": 6.0}  # the probes reach 15 m: a stop from city turning speeds, as there
  trip = ms.drive(r, cfg, lane=1, walls=post, reach=REACH, seconds=60)
  ended(trip, f"held {OBST_WAIT:.0f} s by a static obstacle in the way")
  assert abs(y - trip.car.y - ms.CAR_DIMS[3] - LEAD_STOP) < 1.0, y - trip.car.y
  assert phases(trip, "follow")[0].get("obst"), phases(trip, "follow")[:1]
  trip = ms.drive(r, cfg, lane=1, walls=post, probe=False, reach=REACH, seconds=60)
  assert trip.finished == "abort: collision", trip.finished
  # the plan off its lane past the clamp's reach (005: 3 m right of it), into the wall along the road's edge
  from openpilot.tools.sim.bridge.gta5.test_mapdrive import bumped
  r = bumped(ms.straight(600), 120.0, 40.0, 5.0)
  for probe in (True, False):
    trip = ms.drive(r, cfg, lane=1, walls=kerbs(r, 11.0), probe=probe, reach=REACH, seconds=60)
    if probe:
      ended(trip, f"held {OBST_WAIT:.0f} s by a static obstacle in the way")
    else:
      assert trip.finished == "abort: collision", trip.finished
  for name, route, edge in (("straight", ms.straight(600), 11.0), ("curve 60 m", ms.curve(60.0, 90.0), 11.0),
                            ("curve 40 m, a lane each way", ms.curve(40.0, -70.0, ours=1, back=1), 5.5)):
    trip = ms.drive(route, {"seed": 2}, lane=1 if edge > 6 else 0, walls=kerbs(route, edge), reach=REACH)
    assert trip.finished == "arrived" and not phases(trip, "follow") and trip.car.collisions == 0, (name, trip.finished)
  print("static obstacles: ok")


def test_held_ends_the_trip():
  # a driven vehicle stopped in our lane for good: WAIT_MAX s behind it, then the trip ends (mdlead traffic trips 2 and 3
  # left the abort for follow again each step, for ever)
  r = ms.straight(600)
  trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[parked(r, 1, 120.0)], reach=REACH, seconds=150)
  ended(trip, f"held {WAIT_MAX:.0f} s by a vehicle in the way")
  assert trip.events[-1]["phase"] == "end" and trip.md.info()["leadDriven"] is True, trip.events[-1]
  # parked (no one at its wheel, mdlead trips 2 and 3 by the camera): it won't move, so PARKED_WAIT s
  trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[parked(r, 1, 120.0, driven=False)], reach=REACH, seconds=150)
  ended(trip, f"held {PARKED_WAIT:.0f} s by a parked vehicle in the way")
  t, v = trip.col(0), trip.col(4)
  stopped = t[(t > 5.0) & (v < 0.25)][0]
  assert PARKED_WAIT < trip.md.aborts[0]["t"] - stopped < PARKED_WAIT + 3.0, (trip.md.aborts[0]["t"], stopped)
  follow = phases(trip, "follow")[0]
  assert follow["driven"] is False and abs(follow["right"]) < 0.1, follow
  assert [e["why"] for e in nudges(trip)] == ["no room"], nudges(trip)  # in the middle of our lane: no passing it in the lane
  print("held ends the trip: ok")


def passed(trip, clear_min: float | None = None):
  """Arrived, the parked vehicles passed with the clearance each shift was made for (less the car's tracking), never
  held behind one, the path back on the plan after."""
  ok = [e for e in nudges(trip) if e["ok"]]
  planned = min(e["clear"] for e in ok)
  assert trip.finished == "arrived" and trip.car.collisions == 0 and not phases(trip, "follow"), (trip.finished, phases(trip, "follow")[:2])
  assert ok and planned >= NUDGE_CLEAR - 1e-6 and all(e["ok"] for e in nudges(trip)), nudges(trip)
  assert trip.col(13).min() > (clear_min if clear_min is not None else planned - 0.1), (trip.col(13).min(), planned)
  assert abs(np.interp(trip.md.s, trip.md.s_path, trip.md.nudge_off)) < 1e-3 and trip.md.info()["nudge"] is None
  return ok


def test_nudge_past_a_parked_car():
  r = ms.straight(600)
  free = ms.drive(r, {"seed": 2}, lane=1, reach=REACH, seconds=60)
  # its near side 1.75 m into our 5.5 m lane, about 1 m from our path's middle (the leadRight of the live17 trips' parked
  # cars), then 2.55 m in (0.2 m from it): shifted left within the lane, the tighter pass the slower
  for right, want in ((2.0, NUDGE_WANT), (1.2, 0.6)):
    trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[parked(r, 1, 200.0, right=right, driven=False)], reach=REACH, seconds=60)
    e = passed(trip)[0]
    assert e["clear"] > want - 1e-6 and e["shift"] < -0.5 and e["v"] <= 12.0 + 1e-6, e
    # our body CLAMP_MARGIN in from the lane's left edge (the lane line to the other lane, 5.5 m right of the route's line)
    left = trip.col(10) - trip.md.half_width
    assert left.min() > LANE_W + CLAMP_MARGIN - 0.1, left.min()
    t, x, v = trip.col(0), trip.col(1), trip.col(4)
    beside = np.abs(trip.col(2) - 200.0) < 3.0
    assert v[beside].max() < e["v"] + 0.5 and x[beside].max() < x[0] - 0.5, (v[beside].max(), x[beside].max())
    assert t[-1] < free.col(0)[-1] + 5.0, (t[-1], free.col(0)[-1])
  # under way: the expert row's map dict carries it
  md = ms.drive(r, {"seed": 2}, lane=1, vehicles=[parked(r, 1, 200.0, right=2.0, driven=False)], reach=REACH, seconds=20).md
  assert md.info()["nudge"]["off"] < -0.5 and md.info()["nudge"]["clear"] >= NUDGE_WANT - 1e-6, md.info()["nudge"]
  assert md.summary()["nudges"][0]["ok"]
  # on the left of the left lane, beside the centre line: shifted right
  e = passed(ms.drive(r, {"seed": 2}, lane=0, vehicles=[parked(r, 0, 200.0, right=-2.0, driven=False)], reach=REACH, seconds=60))[0]
  assert e["shift"] > 0.5, e
  # driven (traffic): never passed, followed and held for
  trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[parked(r, 1, 200.0, right=2.0)], reach=REACH, seconds=40)
  assert not nudges(trip) and phases(trip, "follow") and trip.car.collisions == 0, nudges(trip)
  print("nudge past a parked car: ok")


def test_nudge_no_room():
  # its near side 1.75 m into a 3.5 m lane: no room in the lane, stopped behind and the trip ends as before
  r = ms.made_route(ms.line((600.0, 0.0)), ms.section(1, 1, w=3.5))
  trip = ms.drive(r, {"seed": 2}, lane=0, vehicles=[parked(r, 0, 200.0, right=1.75, driven=False)], reach=REACH, seconds=80)
  ended(trip, f"held {PARKED_WAIT:.0f} s by a parked vehicle in the way")
  assert [e["why"] for e in nudges(trip)] == ["no room"] and not trip.md.nudge_off.any(), nudges(trip)
  # 2.85 m into a 5.5 m lane beside the oncoming one: passing it would take us over the centre line, so neither
  r = ms.made_route(ms.line((600.0, 0.0)), ms.section(1, 1))
  trip = ms.drive(r, {"seed": 2}, lane=0, vehicles=[parked(r, 0, 200.0, right=0.9, driven=False)], reach=REACH, seconds=80)
  ended(trip, f"held {PARKED_WAIT:.0f} s by a parked vehicle in the way")
  assert [e["why"] for e in nudges(trip)] == ["no room"] and not trip.md.nudge_off.any(), nudges(trip)
  assert (trip.col(10) - trip.md.half_width).min() > 0.0  # our left side never past the centre line
  # nor into a lane beside going our way (no borrowing it)
  r = ms.straight(600)
  trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[parked(r, 1, 200.0, right=0.9, driven=False)], reach=REACH, seconds=80)
  ended(trip, f"held {PARKED_WAIT:.0f} s by a parked vehicle in the way")
  print("nudge no room: ok")


def test_nudge_two_in_a_row():
  # parked along the kerb 7 m apart, the second further in, a third 20 m on: one shift held past them all, not back in between
  r = ms.straight(600)
  cars = [parked(r, 1, 200.0, right=2.0, driven=False), parked(r, 1, 212.0, right=1.6, driven=False),
          parked(r, 1, 245.0, right=2.2, driven=False)]
  trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=cars, reach=REACH, seconds=60)
  passed(trip)
  y, x = trip.col(2), trip.col(1)
  between = (y > 205.0) & (y < 240.0)
  assert x[between].max() < x[0] - 0.5, x[between].max() - x[0]
  assert len(nudges(trip)) == 3 and band_ok(trip), nudges(trip)
  print("nudge two in a row: ok")


def band_ok(trip) -> bool:
  """No weave: the commanded curvature changes sign only a few times along the pass."""
  k = trip.col(6)[trip.col(4) > 2.0]
  return int(np.sum(np.diff(np.sign(k[np.abs(k) > 2e-4])) != 0)) <= 6


def test_nudge_on_a_bend():
  r = ms.curve(60.0, 90.0)
  for right in (2.0, -2.0):  # on the bend's outside, then its inside
    trip = ms.drive(r, {"seed": 3}, lane=1 if right > 0 else 0, vehicles=[parked(r, 1 if right > 0 else 0, 200.0, right=right, driven=False)],
                    reach=REACH, seconds=80)
    passed(trip)
    assert trip.col(8).max() < 0.35, trip.col(8).max()
  print("nudge on a bend: ok")


def test_nudge_before_a_turn():
  for side, lane in (("right", 1), ("left", 0)):
    r = ms.junction_turn(side, stop=None)
    right = 2.0 if side == "right" else -2.0
    # 35 m before the turn: eased out before its corner (NUDGE_NODE short of its point, where the map's sections stop
    # guiding it), then the turn as ever
    trip = ms.drive(r, {"seed": 3}, lane=lane, vehicles=[parked(r, lane, 165.0, right=right, driven=False)], reach=REACH, seconds=80)
    e = passed(trip)[0]
    out = e["at"][1] + trip.md.rear + NUDGE_MARGIN + e["ease"][1]
    assert out < 200.0 - NUDGE_NODE + 0.5 and trip.col(8).max() < 0.35, (e, out, trip.col(8).max())
    # 10 m before it, where the lane gives way to the turn: no shift, stopped behind it as before
    trip = ms.drive(r, {"seed": 3}, lane=lane, vehicles=[parked(r, lane, 190.0, right=right, driven=False)], reach=REACH, seconds=80)
    ended(trip, f"held {PARKED_WAIT:.0f} s by a parked vehicle in the way")
    assert [e["why"] for e in nudges(trip)] == ["no lane"], nudges(trip)
  print("nudge before a turn: ok")


def test_nudge_where_the_lane_is_no_guide():
  # mdnudge traffic trip 2: a car parked at the kerb of the lane a change ends in (refused "no lane", then held 10 s):
  # passed within both lanes of the change, ours either way
  r = ms.junction_turn("right", lead=400.0, stop=None)
  s0, s1, _, _ = ms.drive(r, {"seed": 3}, lane=0, seconds=0.2).md.changes[0]
  trip = ms.drive(r, {"seed": 3}, lane=0, vehicles=[parked(r, 1, s1 + 4.0, right=2.0, driven=False)], reach=REACH, seconds=90)
  e = passed(trip)[0]
  assert e["shift"] < -0.5 and (trip.col(10) - trip.md.half_width).min() > CLAMP_MARGIN - 0.1, e  # never over the centre line
  # trip 3: one just past a junction whose sections (its middle, narrower than the car) leave the lane no guide: our
  # lane's edges carried across it on the straight
  pts = ms.line((600.0, 0.0), step=2.0)
  narrow = Section([Span(Lane(BACKWARD, 5.5), -5.5, 0.0, -1), Span(Lane(FORWARD, 2.6), 1.45, 4.05, 1)], (-5.5, 5.5))
  r = ms.made_route(pts, [narrow if 148 <= k < 152 else ms.section(1, 1) for k in range(len(pts) - 1)], junctions=[150])
  trip = ms.drive(r, {"seed": 2}, lane=0, vehicles=[parked(r, 0, 312.0, right=2.0, driven=False)], reach=REACH, seconds=80)
  passed(trip)
  assert (trip.col(10) - trip.md.half_width).min() > CLAMP_MARGIN - 0.1
  print("nudge where the lane is no guide: ok")


def test_standing_across():
  # mdnudge traffic trips 1 and 4: a driven vehicle standing across our way (a truck swung wide into our lane, a crossing
  # car stopped at our nose) is no lead to queue behind: stopped for, waited for while it moves, and a deadlock (it may be
  # waiting for us, and we can't back off) ends the trip after DEADLOCK_WAIT s, not WAIT_MAX
  r = ms.made_route(ms.line((600.0, 0.0)), ms.section(2, 2), junctions=[30])
  x, y, h = ms.start_pose(r, 1, 300.0)
  for turned in (90.0, 180.0):  # side on, head on
    trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[ms.Vehicle(x - 1.0, y, h + turned)], reach=REACH, seconds=120)
    ended(trip, f"held {DEADLOCK_WAIT:.0f} s by a vehicle standing across our way (deadlock)")
    assert not phases(trip, "follow") and phases(trip, "yield")[0].get("across"), trip.events[:4]
  # it drives off after 40 s: on to arrival
  trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[ms.Vehicle(x - 1.0, y, h + 90.0, later=((40.0, 2.0, 8.0),))], reach=REACH, seconds=150)
  assert trip.finished == "arrived" and trip.car.collisions == 0 and not trip.md.aborts, trip.finished
  # its nose 1.6 m into our lane from a driveway, seen far enough ahead: passed in our lane, as a parked one
  r = ms.straight(600)
  x, y, h = ms.start_pose(r, 1, 250.0)
  trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[ms.Vehicle(x + LANE_W / 2 + 2.4 - 1.6, y, h + 90.0)], reach=REACH, seconds=80)
  assert passed(trip)[0]["driven"]
  print("standing across: ok")


def test_queue_that_moves():
  # the map driver can't see lights: behind a queue at one, it waits as long as the queue moves on now and then, each time
  # less than WAIT_MAX s still but longer in all; the lead creeps on 0.3 m (too little for us to move), then 4 m, then goes
  r = ms.straight(600)
  lead = parked(r, 1, 120.0, start=45.0, accel=1.0, until=0.7,
                later=((45.7, -2.0, 0.0), (95.0, 1.5, 2.5), (97.0, -2.0, 0.0), (140.0, 1.5, 12.0)))
  trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[lead], reach=REACH, seconds=240)
  t, v = trip.col(0), trip.col(4)
  assert trip.finished == "arrived" and trip.car.collisions == 0 and not trip.md.aborts, (trip.finished, trip.md.aborts)
  assert (v < 0.25)[(t > 20.0) & (t < 140.0)].mean() > 0.9, "held most of 2 min"
  print("queue that moves: ok")


def test_following():
  r = ms.straight(1500)
  trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[parked(r, 1, 80.0, speed=8.0)], reach=REACH, seconds=90)
  t, v, gap = trip.col(0), trip.col(4), trip.col(12)
  m = t > 50.0
  want = LEAD_STOP + trip.md.style["headway"] * 8.0
  assert trip.car.collisions == 0 and np.ptp(v[m]) < 0.3 and abs(v[m].mean() - 8.0) < 0.2, (np.ptp(v[m]), v[m].mean())
  assert abs(np.nanmean(gap[m]) - want) < 2.0, (np.nanmean(gap[m]), want)
  # it brakes to a stop from 12 m/s at 3 m/s^2: stopped behind it
  trip = ms.drive(r, {"seed": 2}, lane=1, v0=12.0, vehicles=[parked(r, 1, 120.0, speed=12.0, accel=-3.0, until=0.0, start=8.0)],
                  reach=REACH, seconds=40)
  stopped_behind(trip)
  print(f"following: ok (gap {np.nanmean(gap[m]):.1f} m at 8 m/s, headway {trip.md.style['headway']:.2f} s)")


def test_crossing():
  r = ms.made_route(ms.line((600.0, 0.0)), ms.section(2, 2), junctions=[30])  # north, a junction node at 300 m
  base = ms.drive(r, {"seed": 2}, lane=1, seconds=60)
  t_at = float(np.interp(302.75, base.col(2), base.col(0)))  # when we'd cross the westbound lane nearer the middle
  x_us = ms.start_pose(r, 1, 300.0)[0]

  def crosser(meet: float) -> ms.Vehicle:  # from the east along that lane at 10 m/s, at our lane at t_at + meet
    return ms.Vehicle(x_us + 10.0 * (t_at + meet), 302.75, 90.0, speed=10.0)
  trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[crosser(0.0)], reach=REACH, seconds=90)
  assert trip.finished == "arrived" and trip.car.collisions == 0 and phases(trip, "yield"), (trip.finished, trip.car.collisions)
  assert trip.col(13).min() > 2.0, trip.col(13).min()
  real = MapDriver._crossing
  MapDriver._crossing = lambda self, t, v: None
  try:  # without the hold, it hits us
    assert ms.drive(r, {"seed": 2}, lane=1, vehicles=[crosser(0.0)], reach=REACH, seconds=90).car.collisions
  finally:
    MapDriver._crossing = real
  for meet in (-5.0, 5.0, 8.0):  # through first, or after
    trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[crosser(meet)], reach=REACH, seconds=90)
    assert trip.finished == "arrived" and not phases(trip, "yield") and trip.car.collisions == 0, (meet, trip.finished)
  # turning left across an oncoming car going straight on at 12 m/s (map1010b 030 was cross traffic through a red light)
  r = ms.junction_turn("left", stop=None)
  base = ms.drive(r, {"seed": 3}, lane=0, seconds=60)
  k = int(np.argmax(base.col(1) < -2.75))  # where our middle reaches its lane
  for meet in (-1.0, 0.0, 1.0):
    o = ms.Vehicle(-2.75, base.col(2)[k] + 12.0 * (base.col(0)[k] + meet), 180.0, speed=12.0)
    trip = ms.drive(r, {"seed": 3}, lane=0, vehicles=[o], reach=REACH, seconds=90)
    assert trip.finished == "arrived" and trip.car.collisions == 0 and phases(trip, "yield"), (meet, trip.finished)
  print("crossing: ok")


def test_crossing_at_the_line():
  # mdnudge traffic trip 5: yields held 2 m short of the crosser's lane, past the junction's line; with a stream of cross
  # traffic seen in time, now held at the line
  r = ms.made_route(ms.line((600.0, 0.0)), ms.section(2, 2), junctions=[30], stops=[(288.0, "lights")])
  base = ms.drive(r, {"seed": 2}, lane=1, seconds=60)
  t_at = float(np.interp(297.25, base.col(2), base.col(0)))
  x_us = ms.start_pose(r, 1, 300.0)[0]
  for meet in (-1.0, 0.0):
    stream = [ms.Vehicle(x_us - 10.0 * (t_at + meet + 1.5 * k), 297.25, 270.0, speed=10.0) for k in range(4)]
    trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=stream, reach=REACH, seconds=90)
    t, y, v = trip.col(0), trip.col(2), trip.col(4)
    m = (t > 5.0) & (y < 400.0)
    k = np.flatnonzero(m)[np.argmin(v[m])]
    assert trip.finished == "arrived" and trip.car.collisions == 0 and v[k] < 0.5, (meet, trip.finished, v[k])
    assert y[k] + trip.md.front < 288.0 and any(e.get("line") for e in phases(trip, "yield")), (meet, y[k] + trip.md.front)
  print("crossing at the line: ok")


def test_no_creep_in_a_junction():
  # trip 5 again: stopped in the junction, released into a creep on each lapse of a yield; a crosser GTA's AI slowed for
  # our nose dropped from the crossing check (under CROSS_MOVING) and squeezed it. Here, 4 m past the line, a stream on
  # the far lane holds us while a car on the near lane slows and stands 0.75 m off our nose's corner for 7 s, then goes on
  # across: held where we are until it's gone
  r = ms.made_route(ms.line((600.0, 0.0)), ms.section(2, 2), junctions=[30], stops=[(288.0, "lights")])
  x0, y0, h0 = ms.start_pose(r, 1, 292.0)
  ystop, xs = y0 + 2.4 + 2.5, x0 - 1.0 - 0.75 - 2.4
  slow = ms.Vehicle(x0 - 30.0, ystop, -90.0, speed=6.0, accel=-6.0 ** 2 / (2 * (xs - x0 + 30.0)), until=0.0, later=((16.0, 2.0, 10.0),))
  far = [ms.Vehicle(x0 + 10.0 * (1.0 + 2.0 * k), ystop + 5.5, 90.0, speed=10.0) for k in range(5)]
  trip = ms.drive(r, {"seed": 2}, pose=(x0, y0, h0), lane=1, vehicles=[slow] + far, reach=REACH, seconds=60)
  t, y = trip.col(0), trip.col(2)
  moved = t[np.argmax(y > y0 + 0.3)]
  assert trip.finished == "arrived" and trip.car.collisions == 0 and moved > 18.0 and trip.col(13).min() > 1.0, (moved, trip.col(13).min())
  assert any(e.get("slowed") for e in phases(trip, "yield"))
  print("no creep in a junction: ok")


def test_pedestrians():
  # mdlead traffic trip 1: turning left at 5 m/s, a pedestrian crossing the road out (walking across it, 12 m past the
  # node) was hit; the driver had no pedestrians. Here one walks north across the road out at x -12, reaching our lane as
  # we would
  r = ms.junction_turn("left", stop=None)
  base = ms.drive(r, {"seed": 3}, lane=0, seconds=60)
  k = int(np.argmax(base.col(1) < -12.0))
  t_at, y_at = base.col(0)[k], base.col(2)[k]

  def walker(meet: float = 0.0) -> ms.Vehicle:
    return ms.pedestrian(-12.0, y_at - 1.4 * (t_at + meet), 0.0)
  real = ms.PED_BOX
  ms.PED_BOX = (0.0, 0.0, 0.0)
  try:  # neither listed nor counted (the plugin before either): it's hit, as in the game
    blind = ms.drive(r, {"seed": 3}, lane=0, peds=[walker()], ped_list=False, reach=REACH, seconds=60)
  finally:
    ms.PED_BOX = real
  assert blind.car.collisions and blind.finished == "abort: collision", blind.finished
  for meet in (-1.0, 0.0, 1.0):
    trip = ms.drive(r, {"seed": 3}, lane=0, peds=[walker(meet)], reach=REACH, seconds=90)
    assert trip.finished == "arrived" and trip.car.collisions == 0, (meet, trip.finished)
    held = [e for e in phases(trip, "yield") + phases(trip, "follow") if e.get("ped")]
    assert held and trip.col(13).min() > 0.8, (meet, held[:2], trip.col(13).min())
  # a plugin without the list counts only those within 2.5 m of our heading, 1-12 m ahead (traffic.peds): in a turn the
  # walker comes into that 2 m from us, too late to stop, but braking hard for it takes the edge off
  late = ms.drive(r, {"seed": 3}, lane=0, peds=[walker()], ped_list=False, reach=REACH, seconds=90)

  def impact(trip) -> float:
    return float(trip.col(4)[np.argmax(trip.col(13) <= 0.0)]) if trip.car.collisions else 0.0
  assert phases(late, "yield") and impact(late) < impact(blind) - 0.5, (impact(late), impact(blind))
  # standing in our lane on a straight, walking off east to the pavement at 40 s: stopped behind, then on
  r = ms.straight(600)
  x, y, _ = ms.start_pose(r, 1, 150.0)
  trip = ms.drive(r, {"seed": 2}, lane=1, peds=[ms.pedestrian(x, y, 270.0, 0.0, start=40.0, accel=2.0, until=1.4)], reach=REACH, seconds=120)
  t, v = trip.col(0), trip.col(4)
  assert trip.finished == "arrived" and trip.car.collisions == 0 and not trip.md.aborts, trip.finished
  assert (v[(t > 25.0) & (t < 40.0)] < 0.1).all() and phases(trip, "follow")[0].get("ped"), "held behind it"
  assert phases(trip, "follow")[0]["driven"] is None  # a pedestrian: neither driven nor parked (held WAIT_MAX)
  # on the pavement beside our lane, walking along it either way: driven past at the speed without it
  free = ms.drive(r, {"seed": 2}, lane=1, reach=REACH, seconds=120)
  for heading in (0.0, 180.0):
    walk = ms.pedestrian(x + 5.5 / 2 + 1.5, y, heading)
    trip = ms.drive(r, {"seed": 2}, lane=1, peds=[walk], reach=REACH, seconds=120)
    assert trip.finished == "arrived" and not phases(trip, "yield") and not phases(trip, "follow"), (heading, trip.events[:4])
    assert abs(trip.col(0)[-1] - free.col(0)[-1]) < 0.5, (trip.col(0)[-1], free.col(0)[-1])
  print("pedestrians: ok")


def test_pedestrian_at_the_kerb():
  # traffic2 trip 6: a man waiting at the right kerb of a junction's crosswalk 45 m ahead, at 15.5 m/s towards a light we
  # can't see, stepped out at 0.9 m/s and was hit at 4.25 m/s; the game gave -3.3 m/s^2 for the -4.0 asked. Here he steps
  # out when we'd be 2.5 s from his line at speed, on a car that delivers no more than 3.3 m/s^2 either
  r = ms.made_route(ms.line((600.0, 0.0)), ms.section(2, 2), junctions=[30])
  cfg = {"seed": 2, "speed": 15.5}
  base = ms.drive(r, cfg, lane=1, v0=15.5, reach=REACH, seconds=60, brake=3.3)
  t_at = float(np.interp(302.0, base.col(2) + base.md.front, base.col(0)))

  def walker(early: float) -> ms.Vehicle:
    return ms.pedestrian(11.5, 302.0, 90.0, 0.0, start=t_at - early, accel=2.0, until=0.9)
  real = MapDriver._ped_kerb
  MapDriver._ped_kerb = lambda self, t, v, look: None
  try:  # standing at the kerb he counts for nothing until he steps out: too late to stop then, as in the game
    blind = ms.drive(r, cfg, lane=1, v0=15.5, peds=[walker(2.5)], reach=REACH, seconds=60, brake=3.3)
  finally:
    MapDriver._ped_kerb = real
  hit = float(blind.col(0)[np.argmax(blind.col(13) <= 0.0)]) - (t_at - 2.5)
  assert blind.finished == "abort: collision" and 2.8 < hit < 3.8, (blind.finished, hit)
  for early in (1.8, 2.5, 3.3, 6.0):
    trip = ms.drive(r, cfg, lane=1, v0=15.5, peds=[walker(early)], reach=REACH, seconds=90, brake=3.3)
    assert trip.finished == "arrived" and trip.car.collisions == 0 and trip.col(13).min() > 1.5, (early, trip.finished, trip.col(13).min())
    kerb = [e for e in trip.events if e["event"] == "ped_kerb"]
    assert kerb and trip.col(7).min() > -3.3 - 0.1, (early, kerb[:1], trip.col(7).min())  # no harder than the game gives
  # a busy pavement away from junctions, standing and walking along it: no slower
  r = ms.straight(700)
  free = ms.drive(r, {"seed": 2}, lane=1, reach=REACH, seconds=90)
  peds = [ms.pedestrian(12.0 + (k % 3) * 0.7, float(y), (0.0, 180.0, 0.0)[k % 3], (0.0, 1.4, 1.2)[k % 3])
          for k, y in enumerate(np.arange(100.0, 560.0, 12.0))]
  trip = ms.drive(r, {"seed": 2}, lane=1, peds=peds, reach=REACH, seconds=90)
  assert trip.finished == "arrived" and not [e for e in trip.events if e["event"] == "ped_kerb"], trip.finished
  assert abs(trip.col(0)[-1] - free.col(0)[-1]) < 0.5, (trip.col(0)[-1], free.col(0)[-1])
  # one there walking towards the road: ready for him
  t0 = float(np.interp(195.0, free.col(2), free.col(0)))  # setting off from the pavement as we are 55 m from him
  trip = ms.drive(r, {"seed": 2}, lane=1, peds=[ms.pedestrian(13.5, 250.0, 90.0, 0.0, start=t0, accel=2.0, until=0.8)], reach=REACH, seconds=90)
  assert trip.finished == "arrived" and trip.car.collisions == 0 and [e for e in trip.events if e["event"] == "ped_kerb"]
  print("pedestrian at the kerb: ok")


def test_range():
  r = ms.straight(800)
  free = ms.drive(r, {"seed": 2}, lane=1, seconds=40).col(4).max()
  # the plugin as built (15 m ahead, no "ahead" in nearby): no faster than stops at its edge, but range_share of the speed
  slow = ms.drive(r, {"seed": 2}, lane=1, vehicles=[], seconds=40).col(4).max()
  assert abs(slow - 0.75 * free) < 0.3, (slow, free)
  strict = ms.drive(r, {"seed": 2, "range_share": 0.0}, lane=1, vehicles=[], seconds=40).col(4).max()
  assert 5.0 < strict < 7.5, strict
  # reaching 120 m: the city's speeds as they were
  far = ms.drive(r, {"seed": 2}, lane=1, vehicles=[], reach=REACH, seconds=40).col(4).max()
  assert abs(far - free) < 0.1, (far, free)
  print(f"range: ok ({free:.1f} m/s free, {strict:.1f} stopping within 15 m, {slow:.1f} at range_share)")


if __name__ == "__main__":
  t0 = time.monotonic()
  test_stopped_ahead()
  test_stopped_in_a_turn()
  test_parked_partly_in_lane()
  test_held_ends_the_trip()
  test_static_obstacles()
  test_queue_that_moves()
  test_nudge_past_a_parked_car()
  test_nudge_no_room()
  test_nudge_two_in_a_row()
  test_nudge_on_a_bend()
  test_nudge_before_a_turn()
  test_nudge_where_the_lane_is_no_guide()
  test_standing_across()
  test_following()
  test_crossing()
  test_crossing_at_the_line()
  test_no_creep_in_a_junction()
  test_pedestrians()
  test_pedestrian_at_the_kerb()
  test_range()
  print(f"all passed in {time.monotonic() - t0:.0f} s")
