#!/usr/bin/env python3
"""Offline checks of the map driver's handling of other vehicles (gta5_mapdrive.py, the plugin's nearby) on the lagged
car of mapdrive_sim.py: run as a script.

- a stopped vehicle in our lane ahead, on a straight and through a turn: stopped behind it, LEAD_STOP from it, without a
  touch, braking no harder than the style's decel where it was seen in time; lead_gap recorded
- a vehicle parked partly in our lane: stopped behind it; one parked clear of the lane, or in the oncoming lane through a
  turn, driven past
- held by a vehicle that stays in the way: the trip ends after WAIT_MAX s (PARKED_WAIT s for a parked one), and nothing
  leaves the abort; behind a queue that moves on now and then, however long it all takes, it's driven to arrival
- a slower vehicle ahead: followed at its speed, LEAD_STOP + the style's headway behind, without hunting; one braking to
  a stop: stopped behind it
- a vehicle crossing the junction ahead as we'd reach it: held for, then on; one that's through first, or comes long
  after, not
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
from openpilot.tools.sim.bridge.gta5.gta5_mapdrive import LEAD_STOP, MAPX_COLUMNS, PARKED_WAIT, WAIT_MAX, MapDriver

REACH = {"ahead": 120.0, "side": 40.0}  # the plugin's reach (core.cpp NEARBY_AHEAD, NEARBY_SIDE_AHEAD)


def phases(trip, what: str) -> list[dict]:
  return [e for e in trip.events if e["event"] == "mapdrive" and e.get("phase") == what]


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
  # front into it)
  trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[parked(r, 1, 200.0, right=2.0, driven=False)], reach=REACH, seconds=40)
  stopped_behind(trip)
  # 1 m into it, or clear of it: driven past
  for right in (2.75, 4.4):
    trip = ms.drive(r, {"seed": 2}, lane=1, vehicles=[parked(r, 1, 200.0, right=right, driven=False)], reach=REACH, seconds=60)
    assert trip.finished == "arrived" and not phases(trip, "follow") and trip.col(13).min() > 0.3, (right, trip.finished, trip.col(13).min())
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
  print("held ends the trip: ok")


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
  # on the pavement beside our lane, walking along it either way: driven past at the speed without it
  free = ms.drive(r, {"seed": 2}, lane=1, reach=REACH, seconds=120)
  for heading in (0.0, 180.0):
    walk = ms.pedestrian(x + 5.5 / 2 + 1.5, y, heading)
    trip = ms.drive(r, {"seed": 2}, lane=1, peds=[walk], reach=REACH, seconds=120)
    assert trip.finished == "arrived" and not phases(trip, "yield") and not phases(trip, "follow"), (heading, trip.events[:4])
    assert abs(trip.col(0)[-1] - free.col(0)[-1]) < 0.5, (trip.col(0)[-1], free.col(0)[-1])
  print("pedestrians: ok")


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
  test_queue_that_moves()
  test_following()
  test_crossing()
  test_pedestrians()
  test_range()
  print(f"all passed in {time.monotonic() - t0:.0f} s")
