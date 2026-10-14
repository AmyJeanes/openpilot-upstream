"""The bridge's brain: what decides how the car is driven and what's shown, with none of openpilot's machinery, so the
same brain runs in the bridge in WSL (gta5_world.py, openpilot in process) and on Windows (gta5_host.py, openpilot behind
the link). Each step takes the newest game state: the destination and our route to it, the car's lane by the map, nav's
planner and the simulated person at the wheel for openpilot (gta5_driver.HumanDriver), the drivers (gta5_drivers.py)
and expert mode, the debug overlay, the map view and the mark key's bookmarks. openpilot is an OpenpilotIO
(gta5_opio.py): its state in; cruise buttons, the human driver's inputs, nav's desire and speed cap, the route input and
lane slots out. Plugin commands go to send()."""
import json
import math
import os
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from openpilot.selfdrive.navd.lane_slots import preview
from openpilot.selfdrive.navd.map_match import MapMatcher, RoadGraph
from openpilot.selfdrive.navd.planner import Planner, Tune, lane_plan
from openpilot.selfdrive.navd.route_input import ROUTE_LEN, RouteInput
from openpilot.tools.sim.bridge.gta5.gta5_blindspot import BlindSpot
from openpilot.tools.sim.bridge.gta5.gta5_driver import HumanDriver
from openpilot.tools.sim.bridge.gta5.gta5_expert import Expert
from openpilot.tools.sim.bridge.gta5.gta5_nav_msgs import NavMessages
from openpilot.tools.sim.bridge.gta5.gta5_navd import Destination, nav_inputs
from openpilot.tools.sim.bridge.gta5.gta5_opio import OpenpilotIO
from openpilot.tools.sim.bridge.gta5.gta5_overlay import PROCESS as OVERLAY_PROCESS, ROUTE_LEAD as OVERLAY_LEAD, GpsRoute, Overlay, \
  OverlayProcess
from openpilot.tools.sim.bridge.gta5.gta5_drivers import INACTIVE, Driver, MapDriver, OpenpilotDriver, driver_from_env, took_over
from openpilot.tools.sim.bridge.gta5.gta5_events import Announcer
from openpilot.tools.sim.bridge.gta5.map.cruise import Cruise, Restrictions, Roaming
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
from openpilot.tools.sim.bridge.gta5.map.lane_match import JunctionAreas, LaneMatcher
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes
from openpilot.tools.sim.bridge.gta5.map.paths import CAR_HEIGHT, Paths
from openpilot.tools.sim.bridge.gta5.map.router import Navigator, Route, Router
from openpilot.tools.sim.bridge.gta5.map.stop_lines import StopLines

MAP = os.getenv("GTA5_MAP")  # the map's folder (map/README.md): serves the map view
MAP_EVERY = 0.1  # s
LANE_LINE_EVERY = 0.5  # s
ON_PATH = 15.0  # m: the overlay's turn markers go on nav's lane line where it passes this near their route points
ROUTER = os.getenv("GTA5_ROUTER")  # a Valhalla server on the map (map/README.md) routes, rather than the game's GPS
# m: the route nav gets, in the form of the plugin's GTA route (500 m), as far as a freeway move's lane changes may start
ROUTE_AHEAD, ROUTE_STEP = 3500.0, 5.0
# s the car's last lane reading by the map's lanes stands while there's none: the plugin's own reading counts GTA's
# link, one of several side by side on a wide road (a 1-lane link reads [0, 1] in any lane), so it's used only on a
# map without lane tags
LANE_HOLD = 3.0
# the route input of a route-conditioned driving model (navd/route_input.py); modeld reads it only if its model has one
ROUTE_INPUT = os.getenv("GTA5_ROUTE_INPUT", "1") != "0"  # 0: no route for the model, to A/B one model with and without
OFF_ROUTE_INPUT = 15.0  # m off the route: no route input, as gta5-train's labels
# route input v2's lane slots (navd/lane_slots.py; also in the model's route input), to their own file for watch_route.py
LANE_SLOTS = os.getenv("GTA5_LANE_SLOTS", "1") != "0"
MARKS = os.path.expanduser(os.getenv("GTA5_MARKS", "~/gta5test/marks.jsonl"))  # the mark key's bookmarks, a JSON line each
MARK_BEFORE, MARK_AFTER = 30.0, 5.0  # s of the car's track a mark keeps before and after the press (a press comes late)
TRACK_EVERY = 0.2  # s between the track's points
PLAN_BEHIND, PLAN_AHEAD = 300.0, 80.0  # m of the plan about the car a mark keeps
# the car's set speed follows the map's speed limits along the route (GTA5_FOLLOW_LIMIT=1, as before navSpeed, for A/B runs);
# off, the set speed is the driver's alone and nav caps the speed at the limit (navSpeed, reason speedLimit)
FOLLOW_LIMIT = os.getenv("GTA5_FOLLOW_LIMIT") == "1"
# Navigate on openpilot: "param" (the default) follows the NavigateOnOpenpilot param the UI's nav button sets (off:
# guidance only); "on" nav drives regardless, as test runs need; "off" guidance only
NOO = os.getenv("GTA5_NOO", "param")
NOO_CHECK = 0.5  # s between reads of the param
# how nav's speed cap reaches openpilot: "plan" (the default) publishes it as navSpeed for openpilot's longitudinal
# planner, below the driver's set speed, which stays the car's; "can" lowers the simulated car's set speed to it instead
NAV_SPEED = os.getenv("GTA5_NAV_SPEED", "plan")
NAV_SPEED_EVERY = 0.05  # s
# routing places the car by navd's map match of the simulated GNSS (selfdrive/navd/map_match.py) rather than the game's
# pose; needs GTA5_GPS on and the map's gta5.osm.pbf. Off: the match puts the car on its road's line, so the route's
# lane for the car is wrong (replays ask for needless lane changes) until the lane comes from elsewhere.
NAV_MATCH = os.getenv("GTA5_NAV_MATCH") == "1"
# openpilot's navInstruction and navRoute for the onroad UI's navigation view (gta5_nav_msgs.py), on our router's route
NAV_MSGS = os.getenv("GTA5_NAV_MSGS") == "1"
# GTA has no speed limits; a guess from the street's name, mph: freeways, highways and routes, then anything else
SPEED_LIMITS = ((('Fwy', 'Freeway'), 65), (('Hwy', 'Highway', 'Route'), 55))
CITY_SPEED_LIMIT = 35
HUD_EVERY = 0.2  # s
CRUISE_WAIT = 10.0  # s the engage key waits for a cruise route before giving up


def speed_limit(street: str) -> float:
  if not street:
    return 0.0
  mph = next((limit for words, limit in SPEED_LIMITS if any(w in street for w in words)), CITY_SPEED_LIMIT)
  return mph * 0.44704


def on_path(p, line: np.ndarray, near: float = ON_PATH):
  """The point of a polyline nearest p, where it passes within `near` m; else p."""
  if p is None or len(line) < 2:
    return p
  p = np.asarray(p, float)
  a, ab = line[:-1], np.diff(line, axis=0)
  t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-9), 0.0, 1.0)
  foot = a + ab * t[:, None]
  d = np.hypot(*(foot - p).T)
  k = int(np.argmin(d))
  return foot[k] if d[k] < near else p


class Brain:
  def __init__(self, io: OpenpilotIO, send, log=None, match: bool = False):
    """send(cmd) sends the plugin a command; log is a file for nav's and the drivers' events (GTA5_LOG); match: routing
    from navd's map match of the GNSS (NAV_MATCH), from the fixes openpilot gets (io.take_fix)."""
    self.io = io
    self._send = send
    self.log = log
    self.match = match
    self.recorder = None  # gta5_record.Recorder, with GTA5_RECORD; marks and the overlay read its state
    self.map_view = None  # map_view.MapView, made by whoever runs the brain
    self.nav_msgs = NavMessages(io) if NAV_MSGS else None  # openpilot's navInstruction and navRoute, through io
    self.metric = bool(io.params.get_bool("IsMetric"))
    self.next_hud = 0.0
    self.presses: dict[str, int] = {}
    self.track: deque = deque(maxlen=int(MARK_BEFORE / TRACK_EVERY) + 10)  # [mono, x, y, v], for marks
    self.lane_lines: deque = deque(maxlen=int(MARK_BEFORE / 2) + 2)  # (mono, the line ahead then), every 2 s
    self.marks_due: list[tuple[float, dict]] = []  # marks waiting for MARK_AFTER s more of the track
    self._init_nav()
    self.blindspot = BlindSpot()
    self.expert = Expert(self._send, lambda: self.io.cruise("cruise_cancel"), lambda: self._set_nav_desire(""),
                         wrote=self.io.expert_control)
    self._init_drivers()
    self.navigator = Navigator(Router(ROUTER)) if ROUTER else None
    self._init_cruise()
    self.lane_matcher: LaneMatcher | None = None  # the map's lanes, for the car's lane by them (with its lane tags)
    self.junction_areas: JunctionAreas | None = None
    if self.navigator is not None and MAP and os.path.exists(os.path.join(MAP, "paths.jsonl")):
      threading.Thread(target=self._load_paths, args=(os.path.join(MAP, "paths.jsonl"),), daemon=True).start()
    self.overlay = OverlayProcess() if OVERLAY_PROCESS else Overlay()  # the plugin map debug overlay, while it asks for it
    self.gps = GpsRoute()  # our route on the game map, while the plugin asks for it

  def _init_nav(self):
    """navd's planner, and the GTA layer's parts around it: the simulated person at the wheel and the destination."""
    self.io.params.remove("NavDesire")  # a killed bridge can leave one
    self.nav = Planner(Tune(os.getenv("GTA5_NAVTUNE")), refresh=self.io.params.get_bool("TurnDesireRefresh"))
    self.human = HumanDriver(self._send, lambda: self.io.cruise("cruise_cancel"))
    self.destination = Destination(self.io.params, self._send, ends=NOO != "on")
    self.next_map = 0.0
    self.lane_line: tuple = (None, 0.0, [])  # the route it's for, until when, and the line
    self.route: Route | None = None
    self.route_input: tuple | None = None  # (the Route it encodes, its RouteInput, which has its LaneSlots)
    self.routes = 0  # routes the navigator has made, counting reroutes, for tests to follow
    self.cap = 0.0
    self.cap_reason = ""
    self.next_nav_speed = 0.0
    self.gps_route: list = []
    self.noo, self.noo_t = NOO != "off", 0.0
    self.matcher: MapMatcher | None = None  # with NAV_MATCH, once the map's roads are loaded
    self.match_t: float | None = None  # its last step
    self.good_lane: tuple[list[int] | None, float] = (None, 0.0)  # the last lane read by the map's lanes, and when
    self.picked: tuple | None = None  # a destination from WSL's map view port, through the link (io.take_commands)

  def _init_drivers(self):
    """Who drives while engaged (gta5_drivers.py): openpilot, and the engage key's driver (GTA5_DRIVER; the map driver
    without openpilot)."""
    self.openpilot = OpenpilotDriver(self.io)
    self.driver = driver_from_env(self.openpilot, self._send, self._log_event, None if self.io.drives else "map")
    self.driving: Driver | Expert | None = None  # whose controls the car takes, as of the last step
    self.announcer = Announcer(self._send)  # the plugin's sound and HUD text as a driver engages and hands back
    # the map driver's route without a destination, where it's the only driver (no openpilot): map/cruise.py
    self.cruise: Cruise | None = None
    self.cruise_asked: float | None = None  # when the engage key asked to cruise, until its route comes

  def _init_cruise(self):
    """Cruising (map/cruise.py) needs the router, and is only for the map driver driving alone: with openpilot (the
    link, or in process) the engage key without a destination engages openpilot instead."""
    self.cruise = Cruise(self.navigator.router, lambda: self.lane_matcher) if self.navigator is not None and not self.io.drives else None

  # *** each step ***

  def step(self, state: dict):
    """A bridge step, the player in the car and the game's state fresh: nav, the drivers, the overlay, the map view."""
    v = state["vEgo"]
    bearing = (-state["heading"]) % 360  # the game's heading is counterclockwise from north; a bearing is clockwise
    model = self.io.model() or {}
    for cmd in self.io.take_commands():
      self._command(cmd)
    self._match_step(state)
    inputs: dict = {"speed_limit": speed_limit(state.get("street", ""))}
    self.gps_route = state.get("route") or []
    if self.navigator is not None:
      state = self._map_route(state, bearing)
      limits = state.get("limits")
      known = bool(limits) and limits[0][0] == 0 and limits[0][1] > 0
      if known:
        inputs["speed_limit"] = limits[0][1]
      inputs["speed_limit_follow"] = known and FOLLOW_LIMIT
    # the car's blind-spot monitor (gta5_blindspot.py), for openpilot (the Model 3's own signals), nav and the driver
    left, right = self.blindspot.update(state, time.monotonic())
    inputs.update(left_blindspot=left, right_blindspot=right)
    state = {**state, "blindspot": [left, right]}
    indicator = state.get("indicator")
    inputs["user_torque"] = self.human.stalk(indicator, state["heading"], state["yawRate"], model.get("lane_change", 0),
                                              self.nav.signaling, left if indicator == "left" else right if indicator == "right" else False)
    self.io.set_inputs(**inputs)
    if self.nav_msgs is not None:
      # the game's pose stands in for the car's localizer (GPS and odometry)
      self.nav_msgs.update(self.route, v, self.navigator.router.osm if self.navigator is not None else None, self._lane_slots,
                           pose=(np.array(state["pos"][:2], dtype=float), bearing))
    for msg in self._overlay(state, v):
      self._send(msg)
    self.driving = self._step_drivers(state)
    self.announcer.step(time.monotonic(), self.driver, self.openpilot, took_over(state), self.expert.on, self.route)
    if self.driving is not None and self.driving is not self.openpilot:
      # expert mode or another driver drives: openpilot stays disengaged, and nav and pull-away wait
      self._set_cap(0.0, "")
      self._set_blinkers()
      self._update_buttons(state)
      if self._map_drive() is not None:
        self._lane_line(state, v)  # the map driver's path, for recordings and the overlay, whether drawn or not
      self._update_map(state, bearing, v)
      return
    state = {**state, "modelLane": self._model_lane()}
    drives = self._nav_drives()
    out = self.nav.update(nav_inputs(state, self.openpilot.engaged, state.get("indicator"), model.get("turns") or {},
                                     time.monotonic(), drives))
    self.human.act(out)
    for e in out.events:  # map errors nav found, in the state log as the map driver's anomalies are in its own
      if self.log:
        self.log.write(json.dumps({"mono": round(time.monotonic(), 3), **e}) + "\n")
    for d in out.desires:
      self._set_nav_desire(d)
    self._set_cap(out.cap, out.cap_reason)
    self._set_blinkers()
    if out.arrived:
      self.destination.arrived()
    self.human.pull_away.update(state, self.openpilot.engaged)
    self._update_buttons(state)
    self._update_map(state, bearing, v)

  def controls(self, state: dict) -> dict:
    """The plugin's control message for this step: whoever drives' (openpilot's through the OpenpilotIO), else none."""
    msg = self.driving.controls(state) if self.driving is not None else None
    return msg if msg is not None else INACTIVE

  def hud(self):
    """The game's HUD: the set speed as openpilot's UI shows it (km/h; 0 when not set), and whether a driver is engaged."""
    now = time.monotonic()
    if now < self.next_hud:
      return
    self.next_hud = now + HUD_EVERY
    set_kph = self.io.set_speed_kph
    self._send({"type": "hud", "setSpeed": set_kph if 0 < set_kph < 255 else 0, "metric": self.metric,
                "engaged": self.driver.engaged or self.openpilot.engaged})

  def close(self):
    self._flush_marks()
    self.overlay.close()

  # *** openpilot ***

  def _model_lane(self) -> list | None:
    """The driving model's lane for nav (modelV2.laneHead): [index from the left of ours, count, probability], None
    without a fresh one or from a model without the head."""
    return (self.io.model() or {}).get("lane_head")

  def _set_cap(self, cap: float, reason: str):
    """nav's speed cap (m/s, 0 for none) to openpilot: as navSpeed for its planner, or as the car's set speed (NAV_SPEED)."""
    self.cap = cap
    by_set_speed = NAV_SPEED == "can"
    self.io.set_inputs(cruise_cap=cap, cap_set_speed=by_set_speed)
    now = time.monotonic()
    if by_set_speed or (now < self.next_nav_speed and reason == self.cap_reason):
      self.cap_reason = reason
      return
    self.cap_reason, self.next_nav_speed = reason, now + NAV_SPEED_EVERY
    self.io.nav_speed(cap, reason or "none")

  def _set_blinkers(self):
    # set once per step, from the indicator and nav's repeat gap: the car thread could read any value set in between
    left, right = self.human.blinkers(self.nav.blinker_gap(time.monotonic()))
    self.io.set_inputs(left_blinker=left, right_blinker=right)

  def _set_nav_desire(self, desire: str):
    self.io.nav_desire(desire)

  def _write_route_input(self, state: dict):
    """The driving model's route input for the route and the car's place on it; zero off it."""
    if not ROUTE_INPUT:
      return
    if self.route is None or self.route.off > OFF_ROUTE_INPUT or not self._nav_drives():  # guiding only: no route for the model
      self.io.route_input(np.zeros(ROUTE_LEN, np.float32))
      return
    enc = self._route_encoder(self.route)
    try:
      vec = enc.encode(self.route.at, state["heading"], state["vEgo"])
    except Exception as e:  # the lane slots on a map they can't read: the rest of the input carries on without them
      print(f"gta5: lane slots: {e!r}")
      enc = self._route_encoder(self.route, lanes=False)
      vec = enc.encode(self.route.at, state["heading"], state["vEgo"])
    self.io.route_input(vec)

  def _write_lane_slots(self, state: dict, lane: list[int] | None = None):
    """Route input v2's lane slots for the preview, zero off the route; with the car's lane, for the model's guess at it."""
    if not LANE_SLOTS:
      return
    route = self.route if self.route is not None and self.route.off <= OFF_ROUTE_INPUT else None
    try:
      vec = preview(self._route_encoder(route).slots if route is not None else None, route.at if route is not None else 0.0,
                    state["vEgo"], lane)
    except Exception as e:  # only a preview: it mustn't stop the bridge
      print(f"gta5: lane slots: {e!r}")
      vec = preview(None, 0.0, 0.0)
      if route is not None:
        self._route_encoder(route, lanes=False)
    self.io.lane_slots(vec)

  # *** nav, the map and the drivers ***

  def _nav_drives(self) -> bool:
    """Navigate on openpilot: nav drives, rather than only guiding."""
    if NOO != "param":
      return NOO != "off"
    now = time.monotonic()
    if now - self.noo_t > NOO_CHECK:
      self.noo_t = now
      try:
        self.noo = self.io.params.get_bool("NavigateOnOpenpilot")
      except Exception:  # params built before the key existed
        self.noo = True
    return self.noo

  def _step_drivers(self, state: dict) -> Driver | Expert | None:
    """Steps whoever drives (expert mode first, while it's on), handing the car back where the driver says so: who
    drives now, None for the player."""
    now = time.monotonic()
    self.openpilot.step(state, self.route, now)
    p = self.driver
    if self.expert.update(state, self.route, self.openpilot.engaged):
      if p is not self.openpilot and p.engaged:
        p.disengage("expert mode drives")
      return self.expert
    if p is not self.openpilot and p.engaged:
      why = p.step(state, self.route, now)
      if why is not None:
        self.announcer.handback(why)
        p.disengage(why)
        if why == "arrived":
          self.destination.arrived()
      return p if p.engaged else None
    return self.openpilot if self.openpilot.engaged else None

  def _map_drive(self):
    """The MapDrive of the map driver driving now (expert mode's, or the engage key's), else None."""
    if self.expert.md_driving:
      return self.expert.md
    return self.driver.md if isinstance(self.driver, MapDriver) and self.driver.engaged else None

  def _driver_name(self) -> str:
    """Who drives now, for marks: expert mode's source (ai, map, ...), the driver's name, or user."""
    if self.driving is self.expert:
      return self.expert.source
    return self.driving.name if self.driving is not None else "user"

  def _log_event(self, what: str, **kw):
    if self.log:
      self.log.write(json.dumps({"mono": round(time.monotonic(), 3), "event": what, **kw}) + "\n")

  def _command(self, cmd: dict):
    """One from WSL, through the link (gta5_opio.COMMANDS)."""
    if cmd.get("type") == "expert":
      if self.expert.path is None:  # as the bridge leaves its control file alone without GTA5_EXPERT
        print("gta5: expert mode isn't enabled here (GTA5_EXPERT): WSL's control file is left alone", flush=True)
      else:
        self.expert.control(cmd.get("cfg") or {})
    elif cmd.get("type") == "destination":
      self.picked = (cmd.get("xy"),)

  def _match_step(self, state: dict):
    """navd's map match, moved on each step by the car's speed and yaw rate and corrected by each GNSS fix openpilot
    gets (the link's, from the host)."""
    if self.matcher is None:
      return
    now = time.monotonic()
    self.matcher.predict(0.0 if self.match_t is None else now - self.match_t, state["vEgo"], state["yawRate"])
    self.match_t = now
    fix = self.io.take_fix()
    if fix is not None:
      x, y = to_game(fix["lat"], fix["lon"])
      self.matcher.update(x, y, fix["bearing"], fix["bearing_accuracy"], fix["speed"], fix["latency"])

  def _load_paths(self, path: str):
    try:
      paths = Paths(path)
      paths.index()
      self.navigator.router.paths = paths
    except (OSError, ValueError, KeyError) as e:
      print(f"gta5: no road heights or lanes for routes: {e}")
      return
    osm = os.path.join(os.path.dirname(path), "gta5.osm.pbf")
    try:
      lanes = OsmLanes.load(osm, to_game) if os.path.exists(osm) else None
    except (OSError, ValueError, KeyError) as e:
      print(f"gta5: no lanes from {osm}: {e}")
      lanes = None
    self.navigator.router.roads = lanes
    if self.match and lanes is not None:
      self.matcher = MapMatcher(RoadGraph.from_osm(lanes))
      print("gta5: routing from navd's map match of the GNSS")
    if lanes is not None and lanes.tagged:
      self.junction_areas = self._junction_areas(lanes)
      StopLines.of(lanes, build=False)  # read before routes are made on the map, which take their stop lines from it
      self.navigator.router.osm = lanes
      print(f"gta5: lanes from the map's tags ({osm})")
      self.lane_matcher = LaneMatcher(lanes)
    else:
      print("gta5: the map has no lane tags: lanes from GTA's links")
    if getattr(self, "cruise", None) is not None and lanes is not None:  # nav_replay loads the map on a bare brain
      Restrictions.of(lanes)  # read now (a few s each), not when the engage key first asks to cruise
      Roaming.of(lanes)
    self.navigator.refresh()

  def _map_route(self, state: dict, bearing: float) -> dict:
    """The state with our route to the destination, in the form of the plugin's GTA route, and what nav uses of the
    map along it. The destination is whichever was set last of the game map's waypoint (else a mission's GPS route's end)
    and the map view's."""
    pos = np.array(state["pos"][:2], dtype=float)
    picked = self.map_view.take_destination() if self.map_view is not None else None
    if self.picked is not None:
      picked, self.picked = self.picked, None
    dest = self.destination.update(state.get("waypoint"), pos, picked, (state.get("mission") or {}).get("dest"))
    match = self.matcher.match if self.matcher is not None else None
    md = self.expert.md if self.expert.md_driving else getattr(self.driver, "md", None) if self.driver.engaged else None
    avoid = md.take_avoid() if md is not None else None
    if avoid and dest is not None:
      self._reroute(pos, bearing, dest, state["pos"][2], avoid)
    route = self.navigator.update(pos, bearing, dest, time.monotonic(), state["pos"][2], match=match)
    if self.cruise is not None:
      route = self._cruise_route(route, pos, bearing, state["pos"][2])
    self.routes += route is not None and route is not self.route
    self.route = route
    self._write_route_input(state)
    state = {**state, "waypoint": dest.tolist() if dest is not None else None, "route": [], "laneMap": self._map_lane(state)}
    plugin = state.get("lane")
    if self.route is None:
      lane, _ = self._car_lane(None, None, plugin, state["laneMap"], time.monotonic())
      self._write_lane_slots(state, lane)
      return {**state, "lane": lane, "lanePlugin": plugin}
    on = self.route.on_road()
    lane, frac = self._car_lane(self.route.lane() if on else None, self.route.lane_frac() if on else None, plugin,
                                state["laneMap"], time.monotonic())
    self._write_lane_slots(state, lane)
    return {**state, **self.route.info(ROUTE_AHEAD), "route": self.route.ahead(ROUTE_AHEAD, ROUTE_STEP).round(1).tolist(),
            "lane": lane, "lanePlugin": plugin, "laneFrac": frac, "twoWay": self.route.two_way() if on else None,
            "beside": self.route.beside() if on else None, "besideLanes": self.route.beside_lanes() if on else None}

  def _reroute(self, pos: np.ndarray, bearing: float, dest: np.ndarray, z: float, avoid: list):
    """A route to the destination keeping off `avoid`'s links (the map driver giving up a fork's branch it couldn't
    change into in time), on its own thread as the navigator's are; it becomes the navigator's route when it comes."""
    nav = self.navigator

    def run():
      try:
        route = nav.router.route(pos.copy(), bearing, dest.copy(), z, avoid=avoid)
        route.locate(pos, z, -bearing)
        with nav.lock:
          if nav.dest is not None and np.hypot(*(dest - nav.dest)) <= 1.0:
            nav.route, nav.off_since = route, None
        print(f"gta5: routed again keeping off {len(avoid)} link(s): the other branch", flush=True)
      except (OSError, ValueError, KeyError) as e:
        print(f"gta5: routing again off a branch failed: {e}", flush=True)
    threading.Thread(target=run, daemon=True).start()

  def _cruise_route(self, route: Route | None, pos: np.ndarray, bearing: float, z: float) -> Route | None:
    """The route the map driver drives where it's the only driver: the destination's, else (while it drives, or the
    engage key waits for one) a cruise route, which also stands in while a new destination is routed. The engage key's
    wait ends here, engaging it once the car is on its cruise route."""
    now = time.monotonic()
    if route is not None or not (self.driver.engaged or self.cruise_asked is not None):
      self.cruise.reset()
      return route
    cruise = self.cruise.update(pos, bearing, now, z)
    if self.cruise_asked is not None and not self.driver.engaged:
      if cruise is not None and cruise.on_road():
        self.cruise_asked = None
        if self.driver.engage(cruise):
          print("gta5: the map driver cruises: no destination, on along the road", flush=True)
      elif now - self.cruise_asked > CRUISE_WAIT:
        self.cruise_asked = None
        self.announcer.refused("no road ahead")
        print("gta5: the engage key: no road to cruise on", flush=True)
    return cruise

  def _car_lane(self, lane: list[int] | None, frac: float | None, plugin: list[int] | None, map_lane: dict | None,
                now: float) -> tuple[list[int] | None, float | None]:
    """The car's lane [i from the left, of n] and where it is between lanes, for nav, its lane line and recordings: the
    route's (`lane`, on its road), else the map's lane match on our side of the road, else the last of those for
    LANE_HOLD s. The plugin's own reading stands only on a map without lane tags; where it reads as many lanes as the
    route's but another one, and the map's match doesn't side with the route's, there's no lane."""
    matched = [map_lane["lane"], map_lane["lanes"]] if map_lane and map_lane.get("kind") == "own" and map_lane.get("lanes") else None
    tagged = self.navigator is not None and self.navigator.router.osm is not None
    if lane and plugin and plugin[1] == lane[1] and plugin[0] != lane[0] and matched != lane:
      return None, None  # they disagree: no lane changes on either
    if lane:
      self.good_lane = (lane, now)
      return lane, frac
    if matched:
      self.good_lane = (matched, now)
      return matched, None
    if not tagged and plugin:
      return plugin, None
    held, t = self.good_lane
    return (held, None) if held is not None and now - t < LANE_HOLD else (None, None)

  @staticmethod
  def _junction_areas(osm: OsmLanes) -> JunctionAreas | None:
    """The map's junction areas from their cache, built with its stop lines (stop_lines.py) by a separate process the
    first time (about half a minute)."""
    areas = JunctionAreas.cached(osm, build=False)
    if areas is not None and StopLines.cached(osm, build=False) is not None:
      return areas
    print("gta5: building the map's junction areas and stop lines in the background", flush=True)
    try:
      nice = ["nice", "-n", "10"] if os.name == "posix" else []
      subprocess.run([*nice, sys.executable, "-m", "openpilot.tools.sim.bridge.gta5.map.lane_match", osm.path],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, check=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
      print(f"gta5: no junction areas: {e}")
      return None
    return JunctionAreas.cached(osm, build=False)

  def _map_lane(self, state: dict) -> dict | None:
    """The car's lane by the map's lanes alone (lane_match.py), for nav's way back out of the oncoming lanes; None in a
    junction's area, where the car is on the moves through it. "areas" says whether those were looked at yet."""
    m, areas = self.lane_matcher, self.junction_areas
    if m is None:
      return None
    x, y, z = state["pos"]
    if areas is not None and areas.inside(x, y, z):
      return None
    r = m.match(x, y, math.radians(state["heading"] + 90.0), z)
    return None if r is None else {"lane": r.lane, "lanes": r.lanes, "kind": r.kind, "bay": r.bay, "oncoming": r.oncoming,
                                   "beside": list(r.beside), "areas": areas is not None, "way": r.way, "right": r.right}

  def _route_encoder(self, route: Route, lanes: bool = True) -> RouteInput:
    """The route's RouteInput, made once per route (a few ms); without its lane slots where they fail."""
    if self.route_input is None or self.route_input[0] is not route or (not lanes and self.route_input[1].slots is not None):
      osm = self.navigator.router.osm
      side = osm.drive_on_right if osm is not None else True
      try:
        enc = RouteInput(route, side, lanes)
      except Exception as e:
        if not lanes:
          raise
        print(f"gta5: lane slots: {e!r}")
        enc = RouteInput(route, side, False)
      self.route_input = (route, enc)
    return self.route_input[1]

  def _lane_slots(self, route: Route):
    """The route's lane slots, from its RouteInput (made once per route), for the nav messages' lane guidance."""
    return self._route_encoder(route).slots

  def _overlay(self, state: dict, v: float) -> list[dict]:
    paths = self.navigator.router.paths if self.navigator is not None else None
    osm = self.navigator.router.osm if self.navigator is not None else None
    out = self.overlay.update(state, self.route, paths, lambda: self._lane_line(state, v), lambda: self._turn_points(state),
                              self.recorder is not None and self.recorder.recording, osm, lambda: self.blindspot.overlay(state, OVERLAY_LEAD))
    return out + self.gps.update(state, self.route)

  def _turn_points(self, state: dict):
    """The next turn navd signals and where its signal comes on, for the overlay: on nav's lane line, as the route's
    points are its road's line (a two-way road's middle, a junction's centre). Once signalled, where the car was."""
    points = self.nav.turn_points(np.array(state["route"], dtype=float), state.get("forks"), state.get("stops"),
                                  state.get("junctions"), state.get("turns"))
    if points is None:
      return None
    line = np.asarray(self._lane_line(state, state.get("vEgo", 0.0)), float).reshape(-1, 2)
    turn, signal = points
    return on_path(turn, line), signal if signal is None or self.nav.signaling else on_path(signal, line)

  def _update_map(self, state: dict, bearing: float, v: float):
    now = time.monotonic()
    if self.map_view is None or now < self.next_map:
      return
    self.next_map = now + MAP_EVERY
    waypoint = state.get("waypoint")
    speed = f"{v * 3.6:.0f} km/h" if self.metric else f"{v / 0.44704:.0f} mph"
    cruising = self.route is not None and self.route.cruise
    self.map_view.update({
      "t": now,
      "car": {"x": state["pos"][0], "y": state["pos"][1], "z": round(state["pos"][2] - CAR_HEIGHT, 1), "bearing": bearing},
      "routes": {"gps": self.gps_route, "nav": [] if self.route is None else self.route.rest().round(1).tolist(),
                 "lanes": self._lane_line(state, v)},
      "waypoint": waypoint if waypoint and any(waypoint) else None,
      "text": f"{state.get('street', '')}  {speed}{'  engaged' if self.driver.engaged or self.openpilot.engaged else ''}" +
              ("  cruising (no destination)" if cruising else ""),
      "nav": {"routes": self.routes, "cruise": cruising, "length": None if self.route is None else round(self.route.length, 1),
              "at": None if self.route is None else round(self.route.at, 1), "cap": round(self.cap, 2)},
    })

  def _lane_line(self, state: dict, v: float) -> list:
    """The route in the lanes nav aims for, for the map; while the map driver drives, the path it drives (its intent,
    without the in-lane wander), so the ribbon is what's driven."""
    r, now = self.route, time.monotonic()
    md = self._map_drive()
    if md is not None and md.path is not None:
      if r is not self.lane_line[0] or now >= self.lane_line[1]:
        self.lane_line = (r, now + LANE_LINE_EVERY, md.intent_line())
      return self.lane_line[2]
    if r is None:
      return []
    if r is self.lane_line[0] and now < self.lane_line[1]:
      return self.lane_line[2]  # it can take several ms on a long route; the view trims it to the car
    forks = [[f.along - r.at, f.side, f.lanes, f.lanes_in, f.keep, f.other, f.slip] for f in r.forks if f.along > r.at]
    lane = list(self.nav.lane) if self.nav.lane is not None else state.get("lane")  # the lane nav plans from
    line = r.lane_line(lane_plan(r.rest(), forks, lane, r.lanes_at, v, self.nav.tune, r.lane_arrows(r.length, 0.0),
                                 r.lane_drops(r.length, 0.0), maps=r.lane_maps(r.length), turns=r.turns(r.length),
                                 classes=r.changes(r.classes, r.length), limits=r.changes(r.limit_list, r.length),
                                 fwy=self.nav.fwy_state(), crossings=[a - r.at for a in r.stops if a > r.at]))
    self.lane_line = (r, now + LANE_LINE_EVERY, [] if line is None else line.round(1).tolist())
    return self.lane_line[2]

  def _update_buttons(self, state: dict):
    """The plugin's keys stand in for the cruise buttons: engage sets when disengaged and cancels when engaged, and the
    speed keys are resume/accel and set/decel, which step the set speed while engaged (to the next multiple of 5 with
    shift). The arrow keys prompt the map driver while it drives (_arrow)."""
    commands = {"speedUpPresses": "cruise_up", "speedDownPresses": "cruise_down", "speedUp5Presses": "cruise_up5",
                "speedDown5Presses": "cruise_down5"}
    self._track(state)
    marks = state.get("markPresses")
    if marks is not None:
      new = marks - self.presses.get("markPresses", marks)
      self.presses["markPresses"] = marks
      if new > 0:  # a reloaded core counts from 0 again
        self._mark(state)
    for key in ("engagePresses", *commands):
      presses = state.get(key)
      if presses is None:
        continue
      new = presses - self.presses.get(key, presses)
      self.presses[key] = presses
      for _ in range(max(0, min(new, 5))):
        if key == "engagePresses":
          self._engage_key()
        else:
          self.io.cruise(commands[key])
    for key, side in (("leftPresses", "left"), ("rightPresses", "right")):
      presses = state.get(key)
      if presses is None:
        continue
      new = presses - self.presses.get(key, presses)
      self.presses[key] = presses
      if new > 0:  # one prompt however many presses came in a step
        self._arrow(side, state)
    self._prompt_news()

  def _arrow(self, side: str, state: dict):
    """An arrow key while the map driver drives (the plugin's indicator otherwise): cruising, the turn that way at the
    next junction or fork within a few seconds (Cruise.prompt); else, or with none ahead, a change into the lane beside
    (MapDrive.ask_lane); on the HUD, what it does or why it can't."""
    d = self.driver
    if self.expert.on or not isinstance(d, MapDriver) or not d.engaged:
      return
    kind, road = "none", ""
    if self.cruise is not None and self.route is not None and self.route.cruise:
      kind, road = self.cruise.prompt(side, np.array(state["pos"][:2], float), (-state["heading"]) % 360, state["pos"][2],
                                      float(state.get("vEgo") or 0.0))
    if kind == "none":
      asked = d.ask_lane(side, time.monotonic())
      kind = {"ok": None, "cancel": "lane_cancel"}.get(asked, asked)
    if kind is not None:
      self.announcer.prompt(kind, side, road)
    print(f"gta5: the {side} arrow: {kind or 'a lane change'}" + (f" ({road})" if road else ""), flush=True)

  def _prompt_news(self):
    """What became of the arrow keys' prompts since: a turn that couldn't be routed, a lane change made or given up."""
    if self.cruise is not None:
      for kind, side, road in self.cruise.take_notes():
        self.announcer.prompt(kind, side, road)
      for e in self.cruise.take_log():
        what = e.pop("event")
        self._log_event(what, **e)
        print(f"gta5: cruise {what} {json.dumps(e)}", flush=True)
    if isinstance(self.driver, MapDriver):
      for kind, side in self.driver.take_prompts():
        self.announcer.prompt(kind, side)

  def _engage_key(self):
    """Disengages whichever driver is engaged, else engages the driver, or openpilot where it can't (the map driver
    without a route). While expert mode is on the key is openpilot's alone. Without openpilot it's the map driver's
    alone: without a destination it cruises (_cruise_route) where there's a router, once its route comes, and a press
    while it waits for that cancels it; with no router it does nothing."""
    drivers = [self.openpilot] if self.expert.on else [self.driver, self.openpilot]
    drivers = list(dict.fromkeys(p for p in drivers if p is not self.openpilot or self.io.drives))
    self.announcer.pressed(time.monotonic())
    if self.cruise_asked is not None:
      self.cruise_asked = None
      self.announcer.status("cancel")
      print("gta5: the engage key: not cruising after all", flush=True)
      return
    for p in drivers:
      if p.engaged:
        p.disengage("the driver pressed the engage key")
        return
    for p in drivers:
      if p.engage(self.route):
        if p is self.openpilot:
          self.announcer.asked_openpilot(time.monotonic())
        return
    if drivers and self.cruise is not None:
      self.cruise_asked = time.monotonic()
      self.announcer.status("wait")
      print("gta5: the engage key: no destination for the map driver: it cruises once it has the road ahead", flush=True)
      return
    if drivers:
      self.announcer.refused("no route")
    print("gta5: the engage key: no destination for the map driver" if drivers else "gta5: the engage key: expert mode drives",
          flush=True)

  def _track(self, state: dict):
    """The car's recent track and the line it was to follow, for marks; marks whose MARK_AFTER has passed are written."""
    now = time.monotonic()
    pos = state.get("pos")
    if pos and (not self.track or now - self.track[-1][0] >= TRACK_EVERY):
      self.track.append([round(now, 2), round(float(pos[0]), 2), round(float(pos[1]), 2), round(float(state.get("vEgo") or 0.0), 2)])
      line = getattr(self, "lane_line", None)
      if line and line[2] and (not self.lane_lines or now - self.lane_lines[-1][0] >= 2.0):
        self.lane_lines.append((now, line[2][:200]))
    while self.marks_due and now >= self.marks_due[0][0]:
      self._write_mark(self.marks_due.pop(0)[1])

  def _mark(self, state: dict):
    """A bookmark (the plugin's mark key): where and when, what drove and what it was doing, the recording's segment and
    frame, the line it was following (the map driver's plan, nav's lane line from MARK_BEFORE on) and the route about
    it; written to MARKS MARK_AFTER s later with the car's track from MARK_BEFORE s before to then. openpilot's
    userBookmark goes now, so loggerd keeps the segment."""
    now = time.monotonic()
    r, md, rec = self.route, self._map_drive(), self.recorder
    seg = rec.segment if rec is not None else None
    plan = None
    if md is not None and md.path is not None:
      k = (md.s_path >= md.s - PLAN_BEHIND) & (md.s_path <= md.s + PLAN_AHEAD)
      plan = md.path[k][::2].round(2).tolist()
    route_line = None
    if r is not None:
      k = (r.along >= r.at - PLAN_BEHIND) & (r.along <= r.at + PLAN_AHEAD)
      route_line = r.points[k][:, :2].round(2).tolist()
    early = next((ln for t, ln in self.lane_lines if t >= now - MARK_BEFORE), None)
    mark = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "mono": round(now, 3), "t_game": state.get("t"),
            "pos": [round(float(c), 2) for c in state.get("pos") or []], "heading": state.get("heading"), "vEgo": state.get("vEgo"),
            "street": state.get("street"), "lane": state.get("lane"), "indicator": state.get("indicator"),
            "driver": self._driver_name(), "map": md.info() if md is not None else None,
            "route": None if r is None else {"at": round(float(r.at), 1), "length": round(float(r.length), 1),
                                             "dest": [round(float(c), 1) for c in r.points[-1][:2]]},
            "segment": seg.path.name if seg is not None else None, "frame": len(seg.rows) if seg is not None else None,
            "recording": rec.route_name if rec is not None else None,
            "plan": plan, "route_line": route_line, "lane_line_before": early,
            "lane_line": (getattr(self, "lane_line", None) or (None, None, None))[2]}
    self.io.bookmark()
    print(f"gta5: mark at {mark['pos'][:2]} {mark['street']!r} ({mark['driver']}, segment {mark['segment']} f{mark['frame']})", flush=True)
    self.marks_due.append((now + MARK_AFTER, mark))

  def _flush_marks(self):
    for _, mark in self.marks_due:  # with the track as far as it got
      self._write_mark(mark)
    self.marks_due = []

  def _write_mark(self, mark: dict):
    mark["track"] = [p for p in self.track if mark["mono"] - MARK_BEFORE <= p[0]]  # [mono, x, y, v], to MARK_AFTER on
    try:
      Path(MARKS).parent.mkdir(parents=True, exist_ok=True)
      with open(MARKS, "a") as f:
        f.write(json.dumps(mark) + "\n")
    except OSError as e:
      print(f"gta5: mark not saved: {e}", flush=True)
