#!/usr/bin/env python3
"""The comma four (mici) navigation UI on the desktop: scripted scenarios of navigation (navInstruction, navRoute) and
openpilot state played to the real mici UI on a test prefix. Run from openpilot/ with the repo root on PYTHONPATH.

Render: the UI runs in this process on a virtual clock, with touches injected through its own input path. Each scenario
saves <scenario>__<moment>.png at the moments it names, checks its state (exit status 1 on a failure) and, with
--video, writes <scenario>.mp4 with captions. Offscreen (Xvfb) at the lowest CPU priority, mici's 536x240 drawn at
SCALE (2 for viewing; 1 for the native raster):

  OPENPILOT_PREFIX=micinav SCALE=2 xvfb-run -a -s "-screen 0 1280x720x24" nice -n 19 \\
    python selfdrive/ui/tests/mici_nav_player.py --out ~/navdemo/mici/impl [--video] [scenario ...]

Live: publishes a scenario's messages in real time (looping with --loop) for the UI run on its own on the same prefix,
a window to click and drag in (touches in the script are skipped):

  OPENPILOT_PREFIX=micinav python selfdrive/ui/tests/mici_nav_player.py --live drive --loop
  OPENPILOT_PREFIX=micinav SCALE=2 python selfdrive/ui/ui.py

Watch: the mici UI in a window on a prefix as it is, publishing nothing, so it can sit beside a running openpilot (the
GTA sim's gta5 prefix) without taking its UI's sockets away. Touches act for real (Navigate on openpilot, slide to end):

  OPENPILOT_PREFIX=gta5 SCALE=2 python selfdrive/ui/tests/mici_nav_player.py --watch

Scenarios: see SCENARIOS (python mici_nav_player.py --list)."""
import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field

import numpy as np

PREFIX = os.environ.get("OPENPILOT_PREFIX", "")
if __name__ == "__main__" and (not PREFIX or (PREFIX == "gta5" and "--watch" not in sys.argv)):
  sys.exit("set OPENPILOT_PREFIX to a test prefix of its own (not the live gta5 one, except to --watch it)")

from openpilot.cereal import log, messaging
from openpilot.selfdrive.ui.tests.nav_fake import FakeCamera, to_lat_lon

FPS = int(os.getenv("FPS", "30"))  # frames a virtual second: the device draws at 60, half is plenty for frames and video
STEP = 1.0 / FPS
PREROLL = 1.5  # s for the UI to come up onroad before a scenario starts (not saved)
TURN_AT, AFTER = 2614.0, 1600.0  # m along the route to the turn, then on to the destination
MILE = 1609.344
DEFAULT_CAMERA = os.path.expanduser("~/navdemo/camera.png")


# *** the world: a route north, a turn, on to the destination ***

@dataclass
class World:
  s: float | None = None  # m along the route; None: no route
  v: float = 20.0  # m/s
  drive: bool = False  # the car moves along the route at the speed profile's speed
  side: str = "right"  # the turn's side
  road: str = "Vespucci Blvd"
  destination: str = "Del Perro Pier"
  lanes: bool = True  # lanes for the turn within 400 m
  lht: bool = False  # left-hand traffic: our lanes on the road's left
  car_lane: int | None = 1  # the model's lane head, from the left of our lanes; None: none
  lane_prob: float = 0.9
  send_route: bool = True  # navRoute (the map)
  position: bool = True  # navInstruction's pose
  trip: bool = True  # time and distance left
  maneuver: bool = True
  engaged: bool = True
  experimental: bool = True
  rhd: bool = False
  alert: tuple[str, str, int, str] | None = None  # text1, text2, size, alertType
  signal: str = ""  # the turn signal on: left, right

  def end(self) -> float:
    return TURN_AT + AFTER


def speed_profile(s: float) -> float:
  """A city drive: 20 m/s, slowing to 7 for the turn, 14 after it, slowing for the destination."""
  if s < TURN_AT:
    return float(np.interp(TURN_AT - s, [0.0, 150.0], [7.0, 20.0]))
  if s < TURN_AT + AFTER - 150:
    return float(np.interp(s - TURN_AT, [0.0, 100.0], [7.0, 14.0]))
  return float(np.interp(TURN_AT + AFTER - s, [0.0, 150.0], [3.0, 14.0]))


def drive_time(s0: float, s1: float) -> float:
  """s of virtual time the car takes from s0 to s1 m along the route at the speed profile, frame by frame as it moves."""
  t, s = 0.0, s0
  while s < s1:
    s += speed_profile(s) * STEP
    t += STEP
  return t


def route_points(w: World) -> np.ndarray:
  sign = 1.0 if w.side == "right" else -1.0
  return np.array([(0.0, 0.0), (0.0, TURN_AT), (sign * AFTER, TURN_AT)])


def pose_at(w: World, s: float) -> tuple[tuple[float, float], float]:
  sign = 1.0 if w.side == "right" else -1.0
  if s < TURN_AT:
    return (0.0, s), 0.0
  return (sign * (s - TURN_AT), TURN_AT), 90.0 if sign > 0 else 270.0


def nearby_roads() -> list[tuple[np.ndarray, float]]:
  """A street grid around both turns' routes, for the map."""
  roads = []
  for x in np.arange(-AFTER - 750.0, AFTER + 751.0, 250.0):
    roads.append((np.array([(x, -400.0), (x, TURN_AT + 750.0)]), 14.0 if x == 0 else 9.0))
  for y in np.arange(-250.0, TURN_AT + 751.0, 250.0):
    roads.append((np.array([(-AFTER - 750.0, y), (AFTER + 750.0, y)]), 14.0 if abs(y - TURN_AT) < 1 else 9.0))
  return roads


def route_message(w: World):
  msg = messaging.new_message("navRoute", valid=True)
  pts = route_points(w)
  coords = msg.navRoute.init("coordinates", len(pts))
  for c, (x, y) in zip(coords, pts, strict=True):
    c.latitude, c.longitude = to_lat_lon(x, y)
  roads = nearby_roads()
  out = msg.navRoute.init("roads", len(roads))
  for o, (line, width) in zip(out, roads, strict=True):
    cs = o.init("coordinates", len(line))
    for c, (x, y) in zip(cs, line, strict=True):
      c.latitude, c.longitude = to_lat_lon(x, y)
    o.width = width
  return msg


def lane_spec(w: World) -> list[tuple[list[str], bool, str, bool]]:
  """Across the whole road, left to right: (directions, active, activeDirection, oncoming); four our way (the two
  nearest the turn take it, the inner one shared with straight on) and two oncoming."""
  t = w.side
  ours = [(["straight"], False, "none", False), (["straight"], False, "none", False),
          (["straight", t], True, t, False), ([t], True, t, False)]
  if t == "left":
    ours = list(reversed(ours))
  onc = [(["straight"], False, "none", True)] * 2
  return ours + onc if w.lht else onc + ours


def instruction_message(w: World):
  msg = messaging.new_message("navInstruction", valid=True)
  ni = msg.navInstruction
  if w.s is None or w.s >= w.end():
    return msg  # no route: valid false
  s = w.s
  ni.valid = True
  ni.destinationName = w.destination
  ahead = [(TURN_AT, "turn", w.side, w.road), (w.end(), "arrive", "straight", "")]
  ahead = [m for m in ahead if m[0] > s] if w.maneuver else []
  if ahead:
    at, kind, modifier, primary = ahead[0]
    ni.maneuverType, ni.maneuverModifier, ni.maneuverPrimaryText, ni.maneuverDistance = kind, modifier, primary, at - s
    mans = ni.init("allManeuvers", len(ahead))
    for out, (a, k, mod, p) in zip(mans, ahead, strict=True):
      out.distance, out.type, out.modifier, out.primaryText = a - s, k, mod, p
  if w.trip:
    ni.distanceRemaining = w.end() - s
    ni.timeRemaining = ni.timeRemainingTypical = ni.distanceRemaining / 9.0
  if w.position:
    (x, y), bearing = pose_at(w, s)
    ni.position.latitude, ni.position.longitude = to_lat_lon(x, y)
    ni.bearingDeg = bearing
  if w.lanes and s < TURN_AT and TURN_AT - s < 400.0:
    ni.showFull = True
    spec = lane_spec(w)
    out_lanes = ni.init("lanes", len(spec))
    for o, (dirs, active, use, oncoming) in zip(out_lanes, spec, strict=True):
      o.directions, o.active, o.activeDirection, o.oncoming = dirs, active, use, oncoming
    ni.laneDistance = max(TURN_AT - s - 60.0, 0.0)
  return msg


class Publisher:
  """navInstruction every frame, navRoute every 2 s, and the openpilot state the onroad view reads."""
  SERVICES = ["deviceState", "pandaStates", "selfdriveState", "carState", "controlsState", "carParams", "deviceMotion",
              "modelV2", "extrinsicsCalibration", "driverMonitoringState", "driverStateV2", "navInstruction", "navRoute"]

  def __init__(self, camera: str | None):
    self.pm = messaging.PubMaster(self.SERVICES)
    self.cam = FakeCamera(camera) if camera else None
    self._route_t = -1e9
    self._route_key: tuple = ()

  def send(self, w: World, now: float):
    key = (w.side, w.send_route, w.s is None)
    if w.s is not None and w.send_route and (now - self._route_t > 2.0 or key != self._route_key):
      self.pm.send("navRoute", route_message(w))
      self._route_t = now
    self._route_key = key
    self.pm.send("navInstruction", instruction_message(w))
    self._send_onroad(w)
    if self.cam is not None:
      self.cam.send()

  def _send_onroad(self, w: World):
    msgs = {s: messaging.new_message(s, valid=True) for s in self.SERVICES[:11] if s != "pandaStates"}
    ds = msgs["deviceState"].deviceState
    ds.started, ds.deviceType = True, "mici"
    ps = messaging.new_message("pandaStates", 1, valid=True)
    ps.pandaStates[0].ignitionLine = True
    ps.pandaStates[0].pandaType = log.PandaState.PandaType.tres
    msgs["pandaStates"] = ps
    ss = msgs["selfdriveState"].selfdriveState
    ss.enabled = ss.active = w.engaged
    ss.state = log.SelfdriveState.OpenpilotState.enabled if w.engaged else log.SelfdriveState.OpenpilotState.disabled
    ss.experimentalMode = w.experimental
    if w.alert:
      ss.alertText1, ss.alertText2, ss.alertSize, ss.alertType = w.alert
      ss.alertStatus = log.SelfdriveState.AlertStatus.normal
    cs = msgs["carState"].carState
    cs.vEgo = cs.vEgoCluster = w.v
    cs.vCruiseCluster, cs.cruiseState.enabled = 72.0, True
    cs.leftBlinker, cs.rightBlinker = w.signal == "left", w.signal == "right"
    msgs["controlsState"].controlsState.deprecated.vCruise = 72.0
    msgs["carParams"].carParams.openpilotLongitudinalControl = True
    dm = msgs["driverMonitoringState"].driverMonitoringState
    dm.activePolicy = log.DriverMonitoringState.MonitoringPolicy.vision
    dm.isRHD = w.rhd
    dm.visionPolicyState.faceDetected = True
    dm.visionPolicyState.awarenessPercent = 100
    self._model(msgs["modelV2"].modelV2, w)
    for s, m in msgs.items():
      self.pm.send(s, m)

  @staticmethod
  def _model(md, w: World):
    """A straight road (bending into the turn in the junction), its lanes 3.6 m wide, and the lane head."""
    x = np.linspace(0.0, float(np.clip(w.v * 5.5, 30.0, 110.0)), 33)  # the path reaches ~5.5 s ahead, as the model's
    curv = 0.0
    if w.s is not None and -15.0 < TURN_AT - w.s < 25.0:
      curv = 0.012 if w.side == "right" else -0.012
    bend = 0.5 * curv * x ** 2
    md.position.x, md.position.y, md.position.z = x.tolist(), bend.tolist(), [0.0] * 33
    lines = md.init("laneLines", 4)
    for line, y0 in zip(lines, (-5.4, -1.8, 1.8, 5.4), strict=True):
      line.x, line.y, line.z = x.tolist(), (y0 + bend).tolist(), [1.22] * 33
    edges = md.init("roadEdges", 2)
    for edge, y0 in zip(edges, (-7.2, 7.2), strict=True):
      edge.x, edge.y, edge.z = x.tolist(), (y0 + bend).tolist(), [1.22] * 33
    md.laneLineProbs = [0.5, 0.9, 0.9, 0.5]
    dp = md.meta.disengagePredictions  # a confident model: the confidence ball high and green
    dp.brakeDisengageProbs, dp.steerOverrideProbs = [0.02] * 6, [0.02] * 6
    md.roadEdgeStds = [0.4, 0.4]
    near = w.s is not None and 0.0 < TURN_AT - w.s < 150.0
    md.acceleration.x = [-1.0 if near else 0.0] * 33
    if w.car_lane is not None:
      md.laneHead.laneIdx, md.laneHead.laneCount, md.laneHead.prob = w.car_lane, 4, w.lane_prob


# *** scenarios ***

@dataclass
class Event:
  t: float
  fn: object
  shot: bool = False  # runs after the frame at t is drawn (a shot or a check); else before it
  caption: str = ""


@dataclass
class Scenario:
  doc: str
  events: list[Event] = field(default_factory=list)
  until: float = 0.0


def scenarios(c: "Context") -> dict[str, Scenario]:
  """The moments of the approved concept A, and the touches, as scripts of timed events."""
  far, mid, near, arrive = TURN_AT - 1.5 * MILE, TURN_AT - 0.4 * MILE, TURN_AT - 330.0, TURN_AT + AFTER - 110.0
  S = Scenario
  return {
    "no_route": S("No route: the stock driving view, no map page; a tap goes home.", [
      c.cap(0.0, "no route: stock mici onroad"), c.shot(1.0, "driving"),
      c.check(1.0, "no map page in the pager", lambda: not c.map_page.is_visible),
      c.tap(1.1, "camera"), c.check(2.5, "a tap goes home", lambda: c.page() == "home"),
    ], 2.6),
    "drive": S("A drive from 1.5 mi before a right turn, through it, to the next maneuver.", [
      c.world(0.0, s=far, drive=True, car_lane=2), c.cap(0.0, "1.5 mi: a chip, icon + distance"), c.shot(1.5, "far_1.5mi"),
      c.world(2.5, s=mid), c.cap(2.5, "0.4 mi: still a chip; no street name"), c.shot(4.0, "mid_0.4mi"),
      c.check(4.0, "a chip far off", lambda: c.card.mode == "chip"),
      c.world(5.0, s=near), c.cap(5.0, "~300 m: the card grows the lanes, the car under our lane"),
      c.shot(5.0 + drive_time(near, TURN_AT - 280.0), "card_300m"),
      c.check(5.0 + drive_time(near, TURN_AT - 280.0), "the card near the turn", lambda: c.card.mode == "card"),
      c.shot(5.0 + drive_time(near, TURN_AT - 100.0), "card_100m"),
      c.cap(5.0 + drive_time(near, TURN_AT - 20.0), "in the junction: Now"),
      c.shot(5.0 + drive_time(near, TURN_AT - 8.0), "junction_now"),
      c.check(5.0 + drive_time(near, TURN_AT - 8.0), "Now in the junction", lambda: c.card.mode == "now"),
      c.cap(5.0 + drive_time(near, TURN_AT + 10.0), "after the turn: the next maneuver (the destination)"),
      c.shot(5.0 + drive_time(near, TURN_AT + 60.0), "after_turn"),
    ], 5.5 + drive_time(near, TURN_AT + 60.0)),
    "lane_change": S("Navigation asks for a lane change: mici's own alert, the lanes-only card under it.", [
      c.world(0.0, s=TURN_AT - 190.0, v=12.0, car_lane=0), c.cap(0.0, "near the turn, the car in a lane that doesn't take it"),
      c.shot(0.3, "before_request"), c.cap(0.7, "move right / signal to confirm, the lanes under it"),
      c.shot(2.0, "request"), c.check(2.0, "the request shows, the card as lanes", lambda: c.card.mode == "lanes" and c.request() == "right"),
      c.world(3.0, alert=("Steer Right", "Confirm Lane Change", 2, "preLaneChangeRight/warning"), signal="right"),
      c.cap(3.0, "signalled: openpilot's own prompt takes over"), c.shot(4.0, "signalled"),
      c.world(5.0, alert=("Changing Lanes", "", 1, "laneChange/warning"), car_lane=1),
      c.world(6.5, car_lane=2, alert=None, signal=""), c.cap(6.5, "in a turn lane: the request clears"),
      c.shot(8.5, "in_turn_lane"), c.check(8.5, "no request in a turn lane", lambda: c.request() is None and c.card.mode == "card"),
    ], 9.0),
    "arrival": S("The destination: a pin chip, then the route ends and the card goes.", [
      c.world(0.0, s=arrive, drive=True), c.cap(0.0, "arriving: a pin chip"), c.shot(1.0, "arrive"),
      c.cap(drive_time(arrive, TURN_AT + AFTER), "arrived: navigation drops the route"),
      c.shot(drive_time(arrive, TURN_AT + AFTER) + 1.5, "arrived"),
      c.check(drive_time(arrive, TURN_AT + AFTER) + 1.5, "no card, no map page after arriving",
              lambda: c.card.mode == "hidden" and not c.map_page.is_visible),
    ], drive_time(arrive, TURN_AT + AFTER) + 2.0),
    "noo": S("Navigate on openpilot from the map page: tap the road button; blue lanes on, white off.", [
      c.world(0.0, s=TURN_AT - 180.0, v=0.0, car_lane=2), c.cap(0.0, "guidance only: white lanes"), c.shot(0.8, "drive_noo_off"),
      c.cap(1.0, "tap the driving view: the map page"), c.tap(1.0, "camera"), c.shot(2.2, "map_noo_off"),
      c.check(2.2, "a tap opens the map page", lambda: c.page() == "map"),
      c.cap(2.5, "tap the road button: Navigate on openpilot on"), c.tap(2.5, "button"), c.shot(2.75, "map_noo_turning_on"),
      c.shot(3.5, "map_noo_on"), c.check(3.5, "NoO on", lambda: c.noo()),
      c.cap(4.0, "swipe back to driving: blue lanes"), c.swipe(4.0, -1), c.shot(5.5, "drive_noo_on"),
      c.check(5.5, "back on the driving view", lambda: c.page() == "onroad"),
      c.cap(6.0, "disengaged: white (openpilot won't make the move)"), c.world(6.0, engaged=False), c.shot(7.0, "drive_disengaged"),
      c.world(7.5, engaged=True), c.tap(7.5, "camera"), c.cap(7.5, "tap the road button again: off"), c.tap(8.7, "button"),
      c.shot(9.5, "map_noo_off_again"), c.check(9.5, "NoO off", lambda: not c.noo()),
    ], 10.0),
    "slide": S("Slide to end: the button grows, the track and label appear, the X by 75%; early springs back.", [
      c.world(0.0, s=TURN_AT - 180.0, v=0.0, car_lane=2), c.tap(0.2, "camera"), c.cap(0.2, "the map page; the road button at rest"),
      c.shot(1.3, "map_rest"), c.cap(1.5, "slide a little: it grows, the track appears"), c.slide(1.5, 140, release=False),
      c.shot(2.0, "slide_partial"), c.release(2.2), c.cap(2.2, "let go early: it springs back"), c.shot(2.3, "springing_back"),
      c.check(2.9, "still routing after an early release", lambda: c.session.route_on and c.page() == "map"),
      c.cap(3.0, "slide past the end point: the red X"), c.slide(3.0, 330, release=False), c.shot(3.5, "slide_past_end"),
      c.release(3.7), c.cap(3.7, "let go: it finishes, the route ends, back to driving"), c.shot(3.85, "finishing"),
      c.check(5.5, "route ended, map page gone, driving view", lambda: not c.session.route_on and c.page() == "onroad"),
      c.shot(5.5, "after_end"), c.check(5.5, "no turn card", lambda: c.card.mode == "hidden"),
    ], 6.0),
    "long_names": S("A long street: two lines and cut short on the map page; the driving view has none.", [
      c.world(0.0, s=TURN_AT - 180.0, v=0.0, car_lane=2, road="Hillcrest Ridge Access Road Northbound Service Lane",
              destination="Pacific Bluffs Country Club"),
      c.shot(0.8, "drive"), c.tap(1.0, "camera"), c.shot(2.2, "map"),
      c.world(2.5, s=TURN_AT + AFTER - 250.0), c.shot(3.5, "map_arrive"),
    ], 3.6),
    "metric": S("Metric units: m and km on the chip, the card and the map page.", [
      c.metric(0.0), c.world(0.0, s=TURN_AT - 1800.0, v=20.0), c.shot(1.0, "chip_km"),
      c.world(1.5, s=TURN_AT - 180.0, car_lane=2), c.shot(2.5, "card_m"), c.tap(2.6, "camera"), c.shot(3.8, "map"),
    ], 4.0),
    "lht": S("A left turn with our lanes on the road's left (left-hand traffic, RHD driver monitoring).", [
      c.world(0.0, s=TURN_AT - 200.0, v=15.0, side="left", lht=True, car_lane=1, rhd=True), c.shot(1.0, "drive"),
      c.check(1.0, "the card near the turn", lambda: c.card.mode == "card"),
      c.tap(1.2, "camera"), c.shot(2.4, "map"),
    ], 2.5),
    "partial": S("Partial data: no map or pose, no lanes, an unknown lane, no maneuver; each degrades on its own.", [
      c.world(0.0, s=TURN_AT - 200.0, v=15.0, car_lane=2, send_route=False, position=False), c.cap(0.0, "no navRoute, no pose: no map"),
      c.shot(1.0, "no_map_drive"), c.tap(1.1, "camera"), c.shot(2.3, "no_map_page"),
      c.swipe(2.5, -1), c.world(2.5, send_route=True, position=True, lanes=False), c.cap(2.5, "no lanes: the card without them"),
      c.shot(4.0, "no_lanes"), c.check(4.0, "a card without lanes", lambda: c.card.mode == "card" and not c.card.lanes()[0]),
      c.world(4.5, lanes=True, car_lane=None), c.cap(4.5, "no lane head: lanes without the car"), c.shot(5.5, "no_car_lane"),
      c.world(6.0, car_lane=0, lane_prob=0.3), c.cap(6.0, "the model unsure: the car faint, no request"), c.shot(7.0, "unsure_lane"),
      c.world(7.5, maneuver=False, car_lane=2, lane_prob=0.9), c.cap(7.5, "no maneuver: no card"), c.shot(8.5, "no_maneuver"),
      c.check(8.5, "no card without a maneuver", lambda: c.card.mode == "hidden"),
      c.world(9.0, maneuver=True, trip=False), c.tap(9.0, "camera"), c.cap(9.0, "no trip figures: no trip row"),
      c.shot(10.2, "no_trip_page"),
    ], 10.3),
    "pages": S("Swipes between the pages, and the map page's 15 s auto return (home keeps stock 5 s).", [
      c.world(0.0, s=TURN_AT - 900.0, v=20.0, drive=True), c.swipe(0.5, 1), c.cap(0.5, "swipe right: the map page"),
      c.check(1.8, "swipe right from driving: map page", lambda: c.page() == "map"),
      c.swipe(2.0, 1), c.check(3.3, "again: home", lambda: c.page() == "home"),
      c.cap(3.4, "home returns to driving after the stock 5 s"),
      c.check(6.8, "home still 4.5 s after the swipe", lambda: c.page() == "home"),
      c.check(8.8, "home went back to driving by 6.5 s", lambda: c.page() == "onroad"),
      c.tap(10.0, "camera"), c.cap(10.0, "the map page waits 15 s untouched"),
      c.check(24.0, "map page still at 14 s", lambda: c.page() == "map"),
      c.check(26.5, "map page back to driving by 16.5 s", lambda: c.page() == "onroad"),
    ], 27.0),
  }


class Context:
  def __init__(self, camera: str | None, out: str, name: str):
    from openpilot.common.params import Params
    from openpilot.common.version import terms_version, training_version
    self.params = Params()
    self.params.put("HasAcceptedTerms", terms_version)
    self.params.put("CompletedTrainingVersion", training_version)
    self.params.put_bool("IsMetric", False)
    self.params.put_bool("NavigateOnOpenpilot", False)
    self.params.put_bool("NavigateOnOpenpilotDefault", False)
    self.params.put_bool("NavShowLanesAlways", False)
    self.w = World()
    self.pub = Publisher(camera)
    self.out, self.name = out, name
    self.clock = 0.0
    self.t0 = PREROLL  # the scenario's time 0 on the clock
    self.frames: list[list] = []
    self.failed = False
    self.layout = None
    self.held: tuple[float, float] | None = None  # where a press is being held

  # the UI under test
  @property
  def map_page(self):
    return self.layout._nav_page

  @property
  def card(self):
    return self.layout._car_onroad_layout._nav_card

  @property
  def session(self):
    return self.layout._nav_session

  def page(self) -> str:
    lay = self.layout
    names = {id(lay._alerts_layout): "alerts", id(lay._home_layout): "home", id(lay._nav_page): "map",
             id(lay._car_onroad_layout): "onroad", id(lay._body_onroad_layout): "body"}
    shown = [w for w in lay._scroller.items if w.is_visible]
    return names[id(min(shown, key=lambda w: abs(w.rect.x - lay.rect.x)))]

  def noo(self) -> bool:
    return self.session.noo and self.params.get_bool("NavigateOnOpenpilot")

  def request(self) -> str | None:
    a = self.card.request_alert()
    return None if a is None else ("right" if a.alert_type.startswith("navLaneChangeRight") else "left")

  # events
  def world(self, t: float, **kw) -> Event:
    def f():
      for k, v in kw.items():
        setattr(self.w, k, v)
    return Event(t, f)

  def metric(self, t: float) -> Event:
    def f():
      from openpilot.selfdrive.ui.ui_state import ui_state
      self.params.put_bool("IsMetric", True)
      ui_state.is_metric = True
    return Event(t, f)

  def cap(self, t: float, text: str) -> Event:
    return Event(t, lambda: None, caption=text)

  def shot(self, t: float, name: str) -> Event:
    def f():
      import pyray as rl
      img = frame_image()
      if img is None:
        return
      rl.image_format(img, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8)
      rl.export_image(img, os.path.join(self.out, f"{self.name}__{name}.png"))
      rl.unload_image(img)
      c = self.card
      from openpilot.selfdrive.ui.ui_state import ui_state
      state = f"started {ui_state.started} page {self.page()} card {c.mode} phase {c.phase} route {self.session.route_on} noo {self.session.noo}"
      print(f"{self.clock - self.t0:6.2f} s {name}: {state} slide {self.map_page.button.slide.offset:.0f}", flush=True)
    return Event(t, f, shot=True)

  def check(self, t: float, what: str, ok) -> Event:
    def f():
      good = bool(ok())
      self.failed |= not good
      print(f"{self.clock - self.t0:6.2f} s {'PASS' if good else 'FAIL'}: {what}", flush=True)
    return Event(t, f, shot=True)

  def _point(self, what: str) -> tuple[float, float]:
    if what == "button":
      return self.map_page.button.centre
    return 200.0, 130.0  # the camera, left of the card and below driver monitoring

  def _ev(self, x: float, y: float, pressed: bool = False, released: bool = False):
    from openpilot.system.ui.lib.application import MouseEvent, MousePos
    return MouseEvent(MousePos(x, y), 0, pressed, released, not released, self.clock)

  def tap(self, t: float, what: str) -> Event:
    def f():
      x, y = self._point(what)
      self.frames += [[self._ev(x, y, pressed=True)], [self._ev(x, y)], [self._ev(x, y, released=True)]]
    return Event(t, f)

  def swipe(self, t: float, direction: int) -> Event:
    """A quick swipe across the screen: direction 1 drags right (the page to the left comes in), -1 left."""
    def f():
      x0, y, dx, n = 268.0 - direction * 150, 120.0, direction * 300.0, 8
      self.frames += [[self._ev(x0, y, pressed=True)]] + [[self._ev(x0 + dx * i / n, y)] for i in range(1, n + 1)]
      self.frames.append([self._ev(x0 + dx, y, released=True)])
    return Event(t, f)

  def slide(self, t: float, px: float, release: bool) -> Event:
    """Presses the road button and slides it px left over a quarter second; held there unless release."""
    def f():
      x0, y = self._point("button")
      n = max(2, int(0.25 * FPS))
      self.frames += [[self._ev(x0, y, pressed=True)]] + [[self._ev(x0 - px * i / n, y)] for i in range(1, n + 1)]
      self.held = (x0 - px, y)
      if release:
        self.frames.append([self._ev(x0 - px, y, released=True)])
        self.held = None
    return Event(t, f)

  def release(self, t: float) -> Event:
    def f():
      if self.held is not None:
        self.frames.append([self._ev(*self.held, released=True)])
        self.held = None
    return Event(t, f)

  def next_events(self) -> list:
    """This frame's touches: scripted ones, else the finger held where a slide left it."""
    if self.frames:
      return self.frames.pop(0)
    if self.held is not None:
      return [self._ev(*self.held)]
    return []

  def step_world(self):
    w = self.w
    if w.drive and w.s is not None:
      w.v = speed_profile(w.s)
      w.s += w.v * STEP


def timers(ctx: Context) -> dict[str, list[float]]:
  """Times (real, ms) the whole UI's frame, the driving view's card and the map page, as each is drawn."""
  perf: dict[str, list[float]] = {"ui frame": [], "turn card": [], "map page": []}

  def timed(obj, attr, key):
    fn = getattr(obj, attr)

    def wrapper(*a, **kw):
      t0 = time.perf_counter()
      try:
        return fn(*a, **kw)
      finally:
        perf[key].append((time.perf_counter() - t0) * 1000)
    setattr(obj, attr, wrapper)
  timed(ctx.layout, "render", "ui frame")
  timed(ctx.card, "render", "turn card")
  timed(ctx.map_page, "_render", "map page")
  return perf


def frame_image():
  """The frame drawn so far, as an image the way it shows (call between the UI's draw and the end of the frame)."""
  import pyray as rl
  from openpilot.system.ui.lib.application import gui_app
  rl.rl_draw_render_batch_active()
  rt = gui_app._render_texture
  if rt is None:
    return rl.load_image_from_screen()
  img = rl.load_image_from_texture(rt.texture)
  rl.image_flip_vertical(img)
  return img


def srt_time(t: float) -> str:
  ms = int(round(max(t, 0.0) * 1000))
  return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def write_srt(path: str, scen: Scenario) -> bool:
  caps = sorted((e.t, e.caption) for e in scen.events if e.caption)
  if not caps:
    return False
  with open(path, "w") as f:
    for i, (t, text) in enumerate(caps):
      end = caps[i + 1][0] if i + 1 < len(caps) else scen.until
      f.write(f"{i + 1}\n{srt_time(t)} --> {srt_time(end)}\n{text}\n\n")
  return True


def start_video(path: str, w: int, h: int, srt: str | None) -> subprocess.Popen:
  pad = 22 * max(1, round(h / 240))
  vf = "vflip,format=yuv420p"
  if srt:  # a caption strip under the screen, so captions never cover the UI
    style = f"FontName=Inter,FontSize={11},PrimaryColour=&H00FFFFFF,BorderStyle=3,OutlineColour=&H00000000,MarginV=2,Alignment=2"
    vf = f"vflip,pad=iw:ih+{pad}:0:0:black,subtitles={srt}:original_size={w}x{h + pad}:force_style='{style}',format=yuv420p"
  args = ["ffmpeg", "-v", "error", "-nostats", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}", "-r", str(FPS),
          "-i", "pipe:0", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-y", path]
  return subprocess.Popen(args, stdin=subprocess.PIPE)


def run(name: str, args) -> bool:
  real_monotonic = time.monotonic
  ctx = Context(args.camera, args.out, name)
  time.monotonic = lambda: ctx.clock
  import pyray as rl
  real_get_time, real_frame_time = rl.get_time, rl.get_frame_time
  rl.get_time = lambda: ctx.clock
  rl.get_frame_time = lambda: STEP
  video = None
  try:
    import openpilot.selfdrive.ui.mici.layouts.main as mici_main
    from openpilot.selfdrive.ui.ui_state import ui_state
    from openpilot.system.ui.lib.application import gui_app
    mici_main.ONROAD_DELAY = 0.1  # onroad within the preroll (from the first frame its pages aren't laid out yet)
    scen = scenarios(ctx)[name]
    events = sorted(scen.events, key=lambda e: (e.t, e.shot))
    gui_app.init_window(f"mici nav {name}", fps=FPS)
    ctx.layout = mici_main.MiciMainLayout()
    perf = timers(ctx)
    gui_app._mouse.get_events = ctx.next_events
    ctx.pub.send(ctx.w, ctx.clock)
    ui_state.update()
    for _ in gui_app.render():
      t = ctx.clock - ctx.t0
      if t < 0:
        for v in perf.values():
          v.clear()
      while events and events[0].shot and events[0].t <= t + 1e-6:
        events.pop(0).fn()
      if args.video and t >= -1e-6:
        if video is None:
          srt = os.path.join(args.out, f"{name}.srt")
          rt = gui_app._render_texture
          w, h = (rt.texture.width, rt.texture.height) if rt is not None else (gui_app.width, gui_app.height)
          video = start_video(os.path.join(args.out, f"{name}.mp4"), w, h, srt if write_srt(srt, scen) else None)
        frame_video(video)
      if t > scen.until:
        break
      ctx.clock = round(ctx.clock + STEP, 6)
      t = ctx.clock - ctx.t0
      while events and not events[0].shot and events[0].t <= t + 1e-6:
        events.pop(0).fn()
      ctx.step_world()
      ctx.pub.send(ctx.w, ctx.clock)
      ui_state.update()
    gui_app.close()
    for k, v in perf.items():
      if v:
        print(f"perf {k}: mean {np.mean(v):.2f} ms, p90 {np.percentile(v, 90):.2f}, max {np.max(v):.2f} over {len(v)} frames", flush=True)
  finally:
    time.monotonic = real_monotonic
    rl.get_time, rl.get_frame_time = real_get_time, real_frame_time
    if video is not None:
      video.stdin.close()
      video.wait(timeout=120)
  return not ctx.failed


def frame_video(proc: subprocess.Popen):
  """Hands the frame drawn so far to ffmpeg, as it comes off the render texture (flipped there by the filter)."""
  import pyray as rl
  from openpilot.system.ui.lib.application import gui_app
  rl.rl_draw_render_batch_active()
  rt = gui_app._render_texture
  if rt is None:
    img = rl.load_image_from_screen()
    rl.image_flip_vertical(img)  # undone by the filter's vflip
  else:
    img = rl.load_image_from_texture(rt.texture)
  rl.image_format(img, rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8)
  proc.stdin.write(bytes(rl.ffi.buffer(img.data, img.width * img.height * 4)))
  rl.unload_image(img)


def live(name: str, args):
  """Plays a scenario's world events in real time for a UI running on its own; touches and checks are left out."""
  ctx = Context(args.camera, "", name)
  ctx.t0 = 0.0
  scen = scenarios(ctx)[name]
  print(f"{name}: {scen.doc}", flush=True)
  while True:
    ctx.w = World()
    events = sorted((e for e in scen.events if not e.shot and e.fn.__qualname__.startswith("Context.world")), key=lambda e: e.t)
    start = time.monotonic()
    while True:
      ctx.clock = time.monotonic() - start
      while events and events[0].t <= ctx.clock:
        events.pop(0).fn()
      ctx.step_world()
      ctx.pub.send(ctx.w, ctx.clock)
      if ctx.clock > scen.until:
        break
      time.sleep(STEP)
    if not args.loop:
      break


def watch():
  """The mici UI on the prefix as it is: reads it, and publishes nothing (msgq hands a topic to its newest publisher)."""
  import openpilot.selfdrive.ui.mici.layouts.main as mici_main
  from openpilot.selfdrive.ui.ui_state import ui_state
  from openpilot.system.ui.lib.application import gui_app

  class NoPublish:
    def __init__(self, *args, **kwargs):
      pass

    def send(self, *args, **kwargs):
      pass
  messaging.PubMaster = NoPublish
  gui_app.init_window(f"mici UI ({PREFIX})")
  mici_main.MiciMainLayout()
  for _ in gui_app.render():
    ui_state.update()


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--out", default=os.path.expanduser("~/navdemo/mici/impl"))
  ap.add_argument("--camera", default=DEFAULT_CAMERA if os.path.exists(DEFAULT_CAMERA) else None, help="a still road frame (png)")
  ap.add_argument("--video", action="store_true", help="also an MP4 of each scenario, with captions")
  ap.add_argument("--live", action="store_true", help="publish one scenario in real time for a UI run separately")
  ap.add_argument("--loop", action="store_true", help="with --live, play it over and over")
  ap.add_argument("--watch", action="store_true", help="only the UI, on the prefix as it is; publishes nothing")
  ap.add_argument("--list", action="store_true")
  ap.add_argument("names", nargs="*")
  args = ap.parse_args()
  if args.watch:
    watch()
    return
  os.makedirs(f"/dev/shm/msgq_{PREFIX}", exist_ok=True)
  if args.list:
    for n, s in scenarios(Context(None, "", "")).items():
      print(f"{n:12s} {s.doc}")
    return
  if args.live:
    live(args.names[0] if args.names else "drive", args)
    return
  os.makedirs(args.out, exist_ok=True)
  names = args.names or list(scenarios(Context(None, "", "")))
  if len(names) > 1:  # each afresh, in its own process
    failed = []
    for n in names:
      print(f"== {n}", flush=True)
      cmd = [sys.executable, __file__, "--out", args.out] + (["--camera", args.camera] if args.camera else [])
      if subprocess.run(cmd + (["--video"] if args.video else []) + [n], check=False).returncode != 0:
        failed.append(n)
    print(f"failed: {' '.join(failed)}" if failed else "all passed", flush=True)
    sys.exit(1 if failed else 0)
  sys.exit(0 if run(names[0], args) else 1)


if __name__ == "__main__":
  main()
