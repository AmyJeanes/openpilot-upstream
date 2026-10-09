"""Blind-spot monitoring: the detector, its way to openpilot's carState through the simulated Model 3's CAN, openpilot's
lane change waiting on it, and nav and the driver waiting with the blinker on."""
import contextlib
import io
import json
import os
import tempfile

import numpy as np

import openpilot.selfdrive.navd.planner as nav_mod
import openpilot.tools.sim.bridge.gta5.gta5_driver as driver_mod
import openpilot.tools.sim.bridge.gta5.gta5_overlay as ov
import openpilot.tools.sim.lib.simulated_tesla as tesla_mod
from opendbc.car.can_definitions import CanData
from opendbc.car.tesla.interface import CarInterface
from openpilot.cereal import log
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.tools.sim.bridge.gta5.gta5_blindspot import BlindSpot, BlindSpotTune, footprint
from openpilot.tools.sim.bridge.gta5.gta5_driver import NUDGE_TIMEOUT, NUDGE_TORQUE, Driver
from openpilot.tools.sim.bridge.gta5.test_nav import Clock, Drive, route_to_turn
from openpilot.tools.sim.lib.common import SimulatorState, vec3

LaneChangeState = log.LaneChangeState
DIMS = [-1.0, 1.0, -2.4, 2.4]  # ours and the other cars': 2 m wide, 4.8 m long
LANE = 5.5  # m, GTA's
BESIDE = [1.5 * LANE, 1.5 * LANE]  # from our middle to the far edge of the lane beside, centred in ours


def car(x, y, heading=0.0, vy=20.0, vx=0.0, dims=DIMS, driven=1):
  return [x, y, heading, vx, vy, *dims, driven]


def state(vehicles, beside=BESIDE, v=20.0, **extra):
  return {"vEgo": v, "pos": [100.0, 200.0, 30.6], "heading": 0.0, "nearby": {"dims": DIMS, "v": vehicles}, "beside": beside, **extra}


def flags(vehicles, **kw):
  return BlindSpot(BlindSpotTune()).update(state(vehicles, **kw), 10.0)


def test_car_beside_in_the_next_lane():
  assert flags([car(-LANE, 0.0)]) == (True, False)
  assert flags([car(LANE, -3.0)]) == (False, True)
  assert flags([car(LANE * 1.4, 0.0)]) == (False, True)  # at the far side of a wide lane
  assert flags([car(-2 * LANE, 0.0), car(2 * LANE, 0.0)]) == (False, False)  # two lanes over
  assert flags([car(0.0, -12.0), car(0.6, 10.0)]) == (False, False)  # ours, behind and ahead


def test_a_gap_between_two_cars_is_clear():
  ahead = car(-LANE, 2.4 + 1.0 + 2.4)  # its rear a metre past our front bumper
  behind = car(-LANE, -2.4 - 9.0 - 2.4)  # its front 9 m behind our rear bumper, at our speed
  assert flags([ahead, behind]) == (False, False)
  assert flags([car(-LANE, -2.4 - 6.0 - 2.4)]) == (True, False)  # 6 m behind


def test_closing_from_behind():
  gap = 12.0  # m from its front to our rear bumper
  y = -2.4 - gap - 2.4
  assert flags([car(-LANE, y, vy=26.0)]) == (True, False)  # 6 m/s faster: there in 2 s
  assert flags([car(-LANE, y, vy=22.0)]) == (False, False)  # 2 m/s: barely gaining
  assert flags([car(-LANE, -2.4 - 25.0 - 2.4, vy=25.0)]) == (False, False)  # 5 s away
  assert flags([car(-LANE, -2.4 - 35.0 - 2.4, vy=40.0)]) == (False, False)  # beyond 30 m


def test_bodies_not_middles():
  bus = car(-LANE, -15.0, dims=[-1.3, 1.3, -6.0, 6.0])  # its middle 15 m back, its front 9 m back: in the zone
  assert flags([bus]) == (True, False)
  # turned across the lane line: its middle beyond the zone's reach, its nose in it
  assert footprint(car(-9.0, 0.0, heading=-30.0))[1] > -9.0 + 1.0
  assert flags([car(-9.0, 0.0, heading=-30.0)]) == (True, False)


def test_oncoming_crossing_and_parked_cars_dont_count():
  assert flags([car(-LANE, 0.0, heading=180.0, vy=-20.0)]) == (False, False)
  assert flags([car(-LANE, 0.0, heading=90.0, vx=-10.0, vy=0.0)]) == (False, False)
  assert flags([car(LANE, 0.0, vy=0.0, driven=0)], v=0.0) == (False, False)  # parked
  assert flags([car(LANE, 0.0, vy=0.0, driven=1)], v=0.0) == (False, True)  # waiting in traffic beside us


def test_no_lane_our_way_no_zone():
  assert flags([car(-LANE, 0.0), car(LANE, 0.0)], beside=[BESIDE[0], None]) == (True, False)  # the kerb on the right
  # without the route's lanes: 4.5 m out, and the map's lane match for which sides have lanes
  assert flags([car(-LANE, 0.0), car(LANE, 0.0)], beside=None) == (True, True)
  matched = {"kind": "own", "lane": 0, "lanes": 2}
  assert flags([car(-LANE, 0.0), car(LANE, 0.0)], beside=None, laneMap=matched) == (False, True)


def test_set_at_once_cleared_after_a_moment():
  bs = BlindSpot(BlindSpotTune())
  assert bs.update(state([car(-LANE, 0.0)]), 10.0) == (True, False)
  assert bs.update(state([]), 10.1) == (True, False)
  assert bs.update(state([]), 10.45) == (True, False)
  assert bs.update(state([]), 10.55) == (False, False)
  assert bs.update(state([car(LANE, 0.0)]), 10.6) == (False, True)


def test_tune_file():
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "bs.json")
    with open(path, "w") as f:
      json.dump({"behind": 3.0}, f)
    bs = BlindSpot(BlindSpotTune(path))
    assert bs.update(state([car(-LANE, -2.4 - 6.0 - 2.4)]), 10.0) == (False, False)


def test_overlay_zones():
  bs = BlindSpot(BlindSpotTune())
  st = {**state([car(LANE, 0.0)]), "heading": 90.0}  # facing west: right is north
  bs.update(st, 10.0)
  items = bs.overlay(st)
  assert [k for k, _ in items] == ["z", "z", "Z"]
  left, right = items[0][1], items[1][1]
  assert np.all(left[:, 1] < 200.0) and np.all(right[:, 1] > 200.0)
  assert np.allclose(left[:, 2], 30.0)
  ahead = bs.overlay(st, lead=0.5)[0][1]
  assert np.allclose(ahead[:, 0], left[:, 0] - 10.0)  # 20 m/s for 0.5 s westwards


def test_overlay_sends_the_zones_between_full_updates():
  bs = BlindSpot(BlindSpotTune())
  st = {**state([car(LANE, 0.0)]), "debug": {"on": True, "layers": "edz"}}
  bs.update(st, 10.0)
  o, snaps = ov.Overlay(background=False), []
  o._hand = snaps.append
  for _ in range(3):  # no route: only the zones keep it going between full updates
    o.next_route = 0.0
    o.update(st, None, None, list, lambda: None, False, None, lambda: bs.overlay(st))
  assert len(snaps) == 3 and [s["full"] for s in snaps] == [True, False, False]
  msg = o.make(snaps[0])
  kinds = [part[0] for part in msg["g"].split(";")]
  assert kinds.count("z") == 2 and kinds.count("Z") == 1
  assert o.make(snaps[1])["g"].count("z") == 2


def tesla_can(sim_state: SimulatorState) -> list[CanData]:
  """The simulated Model 3's CAN for a step, as CanData."""
  sent = []
  t = tesla_mod.SimulatedTesla.__new__(tesla_mod.SimulatedTesla)
  t.metric, t.cruise_enabled, t.set_speed, t.limit, t.presses = False, False, 0.0, 0.0, tesla_mod.deque()

  class SM:
    def update(self, _):
      pass

    def __getitem__(self, _):
      class C:
        cruiseControl = type("CC", (), {"cancel": False})
      return C

  class PM:
    def send(self, _, msgs):
      sent.extend(msgs)

  t.sm, t.pm = SM(), PM()
  can_list = tesla_mod.can_list_to_can_capnp
  tesla_mod.can_list_to_can_capnp = lambda msgs: msgs
  try:
    t.send_can_messages(sim_state)
  finally:
    tesla_mod.can_list_to_can_capnp = can_list
  return [CanData(*m) for m in sent]


class Car:
  """openpilot's view of the simulated Model 3: its CAN through opendbc's Tesla CarState."""
  def __init__(self):
    self.ci = CarInterface(CarInterface.get_non_essential_params("TESLA_MODEL_3"))
    self.state = SimulatorState()
    self.state.valid = True
    self.state.velocity = vec3(0.0, 25.0, 0.0)
    self.t = 0

  def step(self):
    self.t += int(DT_MDL * 1e9)
    return self.ci.update([(self.t, tesla_can(self.state))])


def test_tesla_can_carries_the_blind_spot():
  c = Car()
  cs = c.step()
  assert not cs.leftBlindspot and not cs.rightBlindspot
  c.state.left_blindspot = True
  cs = c.step()
  assert cs.leftBlindspot and not cs.rightBlindspot
  c.state.left_blindspot, c.state.right_blindspot = False, True
  cs = c.step()
  assert not cs.leftBlindspot and cs.rightBlindspot


def test_openpilot_lane_change_waits_for_the_blind_spot():
  # the driver's stalk and nudge, the blind spot and the car's CAN through openpilot's CarState into its DesireHelper
  # (modeld's lane change state), as selfdrived alerts on it
  clock = Clock()
  driver_mod.time = clock
  c, dh, drv = Car(), DesireHelper(), Driver(lambda m: None, lambda: None)
  c.state.left_blinker = True
  states, blocked_alert = [], []
  for k in range(200):  # 10 s
    clock.t += DT_MDL
    c.state.left_blindspot = k < 120  # occupied for 6 s, longer than the driver's nudge lasts
    c.state.user_torque = drv.stalk("left", 0.0, 0.0, dh.lane_change_state, False, c.state.left_blindspot)
    cs = c.step()
    dh.update(cs, True, 1.0)
    states.append((k, dh.lane_change_state, dh.desire, c.state.user_torque))
    blocked_alert.append(dh.lane_change_state == LaneChangeState.preLaneChange and cs.leftBlindspot)
  waiting = [s for s in states if s[0] < 120]
  assert all(st == LaneChangeState.preLaneChange and desire == log.Desire.none for _, st, desire, _ in waiting[5:])
  assert all(tq == 0 for *_, tq in waiting)  # no nudge while it's occupied, as sunnypilot's nudgeless change
  assert all(blocked_alert[5:120])  # "Car Detected in Blindspot"
  started = next(k for k, st, *_ in states if st == LaneChangeState.laneChangeStarting)
  assert 120 <= started <= 130  # within half a second of clearing
  assert states[started + 1][2] == log.Desire.laneChangeLeft


def test_driver_nudges_once_clear_for_its_timeout():
  clock = Clock()
  driver_mod.time = clock
  d = Driver(lambda m: None, lambda: None)
  assert d.stalk("right", 0.0, 0.0, LaneChangeState.preLaneChange, False, True) == 0
  clock.t += 20.0
  assert d.stalk("right", 0.0, 0.0, LaneChangeState.preLaneChange, False, True) == 0
  clock.t += 0.01
  assert d.stalk("right", 0.0, 0.0, LaneChangeState.preLaneChange, False, False) == -NUDGE_TORQUE
  clock.t += NUDGE_TIMEOUT
  assert d.stalk("right", 0.0, 0.0, LaneChangeState.preLaneChange, False, False) == 0


def test_nav_signals_and_waits_then_changes_once_clear():
  d, route = Drive((0, 2), v=10.0), route_to_turn(600.0)
  d.nav.tune.values["lane_change_early"] = 20.0  # room to wait, as freeway changes start early
  y, started, cleared_at = 0.0, None, None
  while y < 550.0:
    occupied = started is None or d.clock.t - started < 15.0  # held past the lane change's 10 s timeout
    d.step(route, y, {"blindspot": [False, occupied]})
    if d.nav.changing and started is None:
      started = d.clock.t
    if started is not None and not occupied and cleared_at is None:
      cleared_at = d.clock.t
    if cleared_at is not None and d.clock.t - cleared_at > 3.0 and d.indicator:
      d.indicator, d.lane = None, (1, 2)  # openpilot changed lanes once clear, and the stalk cancelled
    if started is not None and occupied and d.clock.t - started > 1.0:
      assert d.indicator == "right" and d.nav.changing == "right" and d.nav.desire == "laneChange"
    y += d.v * 0.05
  assert started is not None and cleared_at is not None
  assert [m["type"] for m in d.sent] == ["setIndicator"]  # on once, never cancelled while it waited
  assert d.nav.changing is None and d.lane == (1, 2)


def test_nav_gives_up_a_change_still_blocked_at_its_last_place():
  # a fork's lane change, held all the way: given up (the blinker off) at the last place it may start, and the fork
  # left to the route, rather than changing into the car
  d = Drive((0, 2), v=15.0)
  route = np.array([(0.0, y) for y in np.arange(0.0, 800.0, 5.0)])
  fork_at, y, gave_up = 400.0, 0.0, None
  while y < 395.0:
    d.step(route, y, {"blindspot": [False, True], "laneFrac": 0.0, "routeEnd": 790.0 - y,
                      "forks": [[fork_at - y, "right", 1, 2, True, 0, False]]})
    if gave_up is None and any(m["type"] == "indicatorOff" for m in d.sent):
      gave_up = fork_at - y
    y += d.v * 0.05
  last = max(40.0, 2.0 * d.v)  # FORK_LAST_DIST, FORK_LAST
  assert gave_up is not None and last - 2.0 < gave_up <= last
  assert [m["type"] for m in d.sent] == ["setIndicator", "indicatorOff"]
  assert len(d.nav.skipped) == 1  # left to the route


def test_nav_gives_up_a_waiting_change_no_longer_needed():
  # held for the blind spot, then the route no longer wants it (rerouted on past the fork): the blinker goes off
  # rather than waiting on to change lanes for nothing
  d = Drive((0, 2), v=15.0)
  route = np.array([(0.0, y) for y in np.arange(0.0, 800.0, 5.0)])
  y, started, gave_up = 0.0, None, None
  while y < 300.0:
    rerouted = started is not None and d.clock.t - started > 3.0
    forks = [] if rerouted else [[400.0 - y, "right", 1, 2, True, 0, False]]
    d.step(route, y, {"blindspot": [False, True], "routeEnd": 790.0 - y, "forks": forks})
    if d.nav.changing and started is None:
      started = d.clock.t
    if gave_up is None and any(m["type"] == "indicatorOff" for m in d.sent):
      gave_up = d.clock.t
    y += d.v * 0.05
  assert started is not None and gave_up is not None and 3.0 < gave_up - started < 3.2
  assert [m["type"] for m in d.sent] == ["setIndicator", "indicatorOff"] and d.nav.changing is None


def merge(car_beside, until: float, end_at: float = 400.0, forks=None, v0: float = 15.0):
  """The car in the right lane of two, the right lane ending at end_at m, driving north from 0 towards it with its speed
  following nav's cap (openpilot braking at up to 3 m/s^2); car_beside(t, y) whether the left blind spot is occupied at
  time t s with the car at y m. Once clear with the blinker on for 1 s, openpilot changes lanes. Returns the Drive and
  [(t, y, v, cap, occupied, indicator)] each step until y passes `until` or the car changes lanes."""
  d = Drive((1, 2), v=v0)
  route = np.array([(0.0, y) for y in np.arange(0.0, end_at + 300.0, 5.0)])
  y, t, clear_for, trace = 0.0, 0.0, 0.0, []
  while y < until and d.lane == (1, 2):
    occupied = car_beside(t, y)
    cap, _ = d.step(route, y, {"blindspot": [occupied, False], "routeEnd": end_at + 290.0 - y, "forks": forks(y) if forks else [],
                               "laneMaps": [[end_at - y, [0, None], 1]] if end_at > y else []})
    trace.append((t, y, d.v, cap, occupied, d.indicator))
    clear_for = clear_for + 0.05 if d.indicator == "left" and not occupied else 0.0
    if clear_for >= 1.0:
      d.lane, d.indicator = (0, 2), None  # openpilot changed lanes once clear, and the stalk cancelled
    want = min(v0, cap) if cap > 0 else v0
    d.v = max(want, d.v - 3.0 * 0.05) if want < d.v else min(want, d.v + 1.0 * 0.05)
    y += d.v * 0.05
    t += 0.05
  return d, trace


def test_nav_falls_in_behind_the_car_alongside_out_of_an_ending_lane():
  # a car alongside at our speed all the way: the lane ends, so the change isn't given up at its last place; the car
  # slows, the other gets ahead, and the change goes once it's clear, short of the lane's end
  d, trace = merge(lambda t, y: abs(15.0 * t - y) < 12.0, until=400.0)
  assert d.lane == (0, 2)
  assert [m["type"] for m in d.sent] == ["setIndicator"]  # on once, never given up
  t, y, v, *_ = trace[-1]
  assert y < 400.0 and 15.0 * t - y > 12.0  # changed behind it, before the end
  assert min(s[2] for s in trace) < 13.0  # slowed to let it by
  held = [s for s in trace if s[4] and s[5] == "left"]
  assert held and held[0][1] < 400.0 - 30.0  # held before its last place to start (lane_change_last), and past it


def test_nav_crawls_to_the_lanes_end_while_the_blind_spot_holds():
  d, trace = merge(lambda t, y: True, until=399.0)
  assert d.lane == (1, 2) and d.nav.changing == "left" and d.indicator == "left"  # still waiting at the end, not given up
  assert [m["type"] for m in d.sent] == ["setIndicator"]
  crawl = [s for s in trace if s[1] > 400.0 - nav_mod.MERGE_STOP_BEFORE]
  assert crawl and max(s[2] for s in crawl) < 1.0 and crawl[-1][3] == nav_mod.MERGE_CRAWL  # over its last m
  assert d.reason == "laneChange"


def test_nav_merge_starts_late_at_any_speed():
  # a lane ending the car only finds itself in past the last place to start (the lane reading late): it still changes
  d = Drive((1, 2), v=1.0)
  route = np.array([(0.0, y) for y in np.arange(0.0, 300.0, 5.0)])
  caps = []
  for _ in range(60):
    waiting = d.nav.changing is None
    cap, _ = d.step(route, 200.0, {"blindspot": [False, False], "routeEnd": 900.0, "laneMaps": [[12.0, [0, None], 1]]})
    if waiting and d.reason == "laneChange":
      caps.append(cap)
  assert d.nav.changing == "left" and d.indicator == "left"
  assert caps and max(caps) < 2.0  # slowing for the end until it goes


def test_nav_gives_up_a_change_out_of_lanes_ending_at_a_split():
  # lanes ending where the route leaves by a fork are the other branch's: a reroute will do, so it's given up as ever
  d, trace = merge(lambda t, y: True, until=395.0, forks=lambda y: [[400.0 - y, "left", 1, 2, True, 1, False]])
  assert [m["type"] for m in d.sent] == ["setIndicator", "indicatorOff"] and d.nav.changing is None
  assert min(s[2] for s in trace) > 9.0  # no slowing to merge


def test_nav_merge_doesnt_start_fresh_in_the_crawl():
  # a lane reading that first puts the car in the ending lane within the crawl's reach of its end: no change from it
  d = Drive((1, 2), v=1.0)
  route = np.array([(0.0, y) for y in np.arange(0.0, 300.0, 5.0)])
  for _ in range(60):
    d.step(route, 200.0, {"blindspot": [False, False], "routeEnd": 900.0, "laneMaps": [[nav_mod.MERGE_STOP_BEFORE - 2.0, [0, None], 1]]})
  assert d.nav.changing is None and d.indicator is None


def map_read(lane, lanes, beside, kind="own"):
  return {"lane": lane, "lanes": lanes, "kind": kind, "bay": False, "oncoming": kind == "oncoming", "beside": beside, "areas": True}


def test_nav_never_changes_into_a_lane_the_map_reads_as_oncoming():
  # nav reads the car in the right lane of two, with its lane ending (or a left turn ahead), but the map has it in the
  # left lane, the oncoming lanes beside it: no change, mandatory or not, and no slowing to merge
  route = np.array([(0.0, y) for y in np.arange(0.0, 800.0, 5.0)])
  for extra in ({"laneMaps": [[120.0, [0, None], 1]]}, {"forks": [[120.0, "left", 1, 2, True, 1, False]]}):
    d = Drive((1, 2), v=10.0)
    y = 0.0
    while y < 115.0:
      d.step(route - [0.0, 0.0], y, {"blindspot": [False, False], "routeEnd": 790.0 - y,
                                    "laneMap": map_read(0, 2, ["oncoming", "own"]),
                                    **{k: [[v[0][0] - y, *v[0][1:]]] for k, v in extra.items()}})
      assert d.nav.changing is None and d.indicator is None
      y += d.v * 0.05


def test_nav_gives_up_a_change_once_the_map_reads_oncoming_that_way():
  d = Drive((1, 2), v=10.0)
  route = np.array([(0.0, y) for y in np.arange(0.0, 800.0, 5.0)])
  y, sent = 0.0, None
  while y < 200.0:
    beside = ["oncoming", "own"] if d.indicator == "left" else ["own", "own"]  # the map turns once the blinker is on
    d.step(route, y, {"blindspot": [False, False], "routeEnd": 790.0 - y, "laneMaps": [[300.0 - y, [0, None], 1]],
                      "laneMap": map_read(1, 2, beside)})
    if sent is None and d.indicator == "left":
      sent = y
    if sent is not None and d.nav.changing is None:
      break
    y += d.v * 0.05
  assert sent is not None and d.nav.changing is None
  assert [m["type"] for m in d.sent][:2] == ["setIndicator", "indicatorOff"]


def test_nav_still_changes_back_out_of_the_oncoming_lanes():
  d = Drive((-1, 2), v=10.0)
  route = np.array([(0.0, y) for y in np.arange(0.0, 800.0, 5.0)])
  for k in range(80):
    d.step(route, k * 0.5, {"blindspot": [False, False], "routeEnd": 700.0,
                            "laneMap": map_read(-1, 2, ["oncoming", "own"], kind="oncoming")})
  assert d.nav.changing == "right" and d.indicator == "right"


def test_nav_merge_waits_for_its_lane_and_the_maps_to_agree():
  # nav reads the car in the right lane of three, which ends; the map has it a lane further left, which carries on:
  # nothing until they agree, then the merge goes
  d = Drive((2, 3), v=10.0)
  route = np.array([(0.0, y) for y in np.arange(0.0, 800.0, 5.0)])
  y, agree_from = 0.0, 120.0
  while y < 250.0 and d.nav.changing is None:
    agree = y >= agree_from
    cap, _ = d.step(route, y, {"blindspot": [False, False], "routeEnd": 790.0 - y, "laneMaps": [[260.0 - y, [0, 1, None], 2]],
                               "laneMap": map_read(2 if agree else 1, 3, ["own", None if agree else "own"])})
    if not agree:
      assert d.nav.changing is None and not (d.reason == "laneChange" and cap < 10.0)
    y += d.v * 0.05
  assert d.nav.changing == "left" and y >= agree_from






def test_flag_held_through_its_zone_flickering():
  # the map's lanes beside dropping out for a reading (as where lanes merge) don't drop the flag at once
  bs = BlindSpot(BlindSpotTune())
  assert bs.update(state([car(LANE, 0.0)]), 10.0) == (False, True)
  assert bs.update(state([car(LANE, 0.0)], beside=[BESIDE[0], None]), 10.05) == (False, True)
  assert bs.update(state([car(LANE, 0.0)]), 10.1) == (False, True)
  assert bs.update(state([], beside=[BESIDE[0], None]), 10.7) == (False, False)  # a lane that has truly gone


def test_openpilot_lane_change_waits_out_a_moments_gap():
  # a car passing ahead out of the zone as the next, closing from behind, comes into it: the flag drops for a frame
  # between them (bsm_trial 20261009b, FX1 and FX2); with the nudge held on, no lane change starts in that gap
  c, dh = Car(), DesireHelper()
  c.state.left_blinker = True
  started = []
  for k in range(120):
    c.state.left_blindspot = k not in (40, 41) and k < 80
    c.state.user_torque = NUDGE_TORQUE
    dh.update(c.step(), True, 1.0)
    started += [k] if dh.lane_change_state == LaneChangeState.laneChangeStarting else []
  assert started and started[0] >= 80 + round(0.5 / DT_MDL) - 1  # only once clear for half a second


def test_nav_merge_held_past_where_it_first_put_the_lanes_end():
  # the map's end of the lane further on than first read (the route's distances shift as the car nears it): a merge
  # held by the blind spot waits on to the end the map now gives, not given up where the first reading put it
  d = Drive((1, 2), v=8.0)
  route = np.array([(0.0, y) for y in np.arange(0.0, 900.0, 5.0)])
  y = 0.0
  while y < 420.0:
    end = 400.0 + max(0.0, y - 300.0) * 0.5
    cap, _ = d.step(route, y, {"blindspot": [True, False], "routeEnd": 890.0 - y, "laneMaps": [[end - y, [0, None], 1]]})
    d.v = min(8.0, cap) if cap > 0 else 8.0
    y += max(d.v, 0.5) * 0.05
  assert d.nav.change_end is not None and d.nav.driven > d.nav.change_end  # past where it first had the end
  assert d.nav.changing == "left" and [m["type"] for m in d.sent] == ["setIndicator"]


def test_nav_merge_held_through_the_lane_reading_dropping_out():
  # the lane reading lost while a merge waits on the blind spot (lanes merging read badly): it waits on, still slowing
  d = Drive((1, 2), v=6.0)
  route = np.array([(0.0, y) for y in np.arange(0.0, 700.0, 5.0)])
  y, caps_lost = 0.0, []
  while y < 395.0:
    lost = d.nav.changing is not None and y > 330.0
    cap, _ = d.step(route, y, {"blindspot": [True, False], "routeEnd": 690.0 - y, "laneMaps": [[400.0 - y, [0, None], 1]],
                               **({"lane": None} if lost else {})})
    if lost:
      caps_lost.append(cap)
    d.v = min(6.0, cap) if cap > 0 else 6.0
    y += d.v * 0.05
  assert d.nav.changing == "left" and [m["type"] for m in d.sent] == ["setIndicator"]
  assert caps_lost and caps_lost[-1] <= nav_mod.MERGE_CRAWL + 0.5


def test_nav_merge_held_keeps_slowing_when_the_readings_disagree():
  # signalled and held, then nav's lane and the map's stop agreeing: it still slows for the lane's end
  d2 = Drive((1, 2), v=10.0)
  route = np.array([(0.0, y) for y in np.arange(0.0, 700.0, 5.0)])
  y, caps = 0.0, []
  while y < 390.0:
    disagree = d2.nav.changing is not None
    cap, _ = d2.step(route, y, {"blindspot": [True, False], "routeEnd": 690.0 - y, "laneMaps": [[400.0 - y, [0, None], 1]],
                                "laneMap": map_read(0 if disagree else 1, 2, [None, "own"] if disagree else ["own", None])})
    caps.append(cap)
    d2.v = min(d2.v, cap) if cap > 0 else d2.v
    y += max(d2.v, 0.5) * 0.05
  assert d2.nav.changing == "left" and min(caps) <= nav_mod.MERGE_CRAWL + 0.5


def test_nav_no_merge_out_of_lanes_at_a_fork_left_to_the_route():
  # a fork's change held by the blind spot and given up, the fork left to the route: the lanes the plan had ending at
  # its split carry on as the branch the car now takes, so no merge out of them follows (bsm_trial 20261009b: a crawl
  # on the freeway, and late changes into a closing car)
  d = Drive((0, 2), v=15.0)
  route = np.array([(0.0, y) for y in np.arange(0.0, 800.0, 5.0)])
  fork_at, y, caps, left = 400.0, 0.0, [], False
  while y < 398.0:
    cap, _ = d.step(route, y, {"blindspot": [False, True], "laneFrac": 0.0, "routeEnd": 790.0 - y,
                               "forks": [[fork_at - y, "right", 1, 2, True, 0, False]],
                               "laneMaps": [[fork_at - 5.0 - y, [None, 0], 1]] if y < fork_at - 5.0 else []})
    caps.append((d.nav.changing, cap))
    left |= bool(d.nav.skipped)
    y += d.v * 0.05
  assert left and d.nav.changing is None  # the fork left to the route
  assert [m["type"] for m in d.sent] == ["setIndicator", "indicatorOff"]  # the change, given up; no merge after
  given_up = max(k for k, (changing, _) in enumerate(caps) if changing)
  assert all(c == 0 or c > 9.0 for _, c in caps[given_up + 1:])  # nor slowing for one


def test_nav_reports_a_bay_the_map_reads_as_oncoming():
  # the bay test's drive, but the map reads the lane left of the car as oncoming: no change into it, and the bay is
  # reported once as a map error, with the map's reading
  route = route_to_turn(200.0, "left")
  d = Drive((0, 2), v=6.0)
  y, lines = 0.0, io.StringIO()
  debug, nav_mod.DEBUG = nav_mod.DEBUG, True
  try:
    with contextlib.redirect_stdout(lines):
      while y < 195.0:
        bay_at = 170.0 - y
        forks = [[bay_at, "right", 2, 2, False, 0, True]] if bay_at > 0 else []
        read = {**map_read(0, 2, ["oncoming", "own"]), "way": 123, "right": 2.5}
        d.step(route, y, {"forks": forks, "routeEnd": 300.0 - y, "laneMap": read})
        y += d.v * 0.05
  finally:
    nav_mod.DEBUG = debug
  bay = [e for e in d.events if e.get("kind") == "bay_reads_oncoming"]
  assert len(bay) == 1 and bay[0]["event"] == "anomaly" and bay[0]["side"] == "left"
  assert bay[0]["map"]["way"] == 123 and bay[0]["map"]["beside"] == ["oncoming", "own"]
  assert abs(bay[0]["pos"][1] + bay[0]["turn_dist"] - 200.0) < 1.0  # the left turn's bay, reported as it opens
  assert lines.getvalue().count("map error: the turn bay left") == 1
  assert d.nav.bay_to == 0.0  # never changed into it


def test_nav_merge_still_held_after_a_moments_gap():
  # the blind spot clear for a single step while a merge waits: openpilot doesn't start on it, so nav still holds the
  # change and keeps slowing for the lane's end rather than taking it as gone
  d, trace = merge(lambda t, y: not 20.0 <= t < 20.06, until=399.0)
  assert d.nav.changing == "left" and not d.nav.change_went and [m["type"] for m in d.sent] == ["setIndicator"]
  assert max(s[2] for s in trace if s[1] > 400.0 - nav_mod.MERGE_STOP_BEFORE) < 1.0
