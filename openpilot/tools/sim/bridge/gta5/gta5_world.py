import gc
import json
import math
import multiprocessing
import os
import subprocess
import sys
import threading
import time
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import numpy as np
from opendbc.car.vehicle_model import VehicleModel
from openpilot.cereal import log, messaging
from opendbc.car.tesla.values import CarControllerParams as TeslaParams
from openpilot.common.params import Params
from openpilot.selfdrive.modeld.route_input import RouteInputWriter
from openpilot.selfdrive.navd.lane_slots import PREVIEW_LEN, lane_slots_path, preview
from openpilot.selfdrive.navd.map_match import MapMatcher, RoadGraph
from openpilot.selfdrive.navd.planner import Planner, Tune, lane_plan
from openpilot.selfdrive.navd.route_input import ROUTE_LEN, RouteInput
from openpilot.tools.sim.lib.simulated_tesla import is_tesla
from openpilot.tools.sim.bridge.common import control_cmd_gen
from openpilot.tools.sim.bridge.gta5 import gta5_gnss
from openpilot.tools.sim.bridge.gta5.gta5_cmd import display_from_env
from openpilot.tools.sim.bridge.gta5.gta5_driver import Driver
from openpilot.tools.sim.bridge.gta5.gta5_expert import Expert
from openpilot.tools.sim.bridge.gta5.gta5_nav_msgs import NavMessages
from openpilot.tools.sim.bridge.gta5.gta5_navd import Destination, nav_inputs
from openpilot.tools.sim.bridge.gta5.gta5_overlay import GpsRoute, Overlay
from openpilot.tools.sim.bridge.gta5.gta5_record import RECORD, Recorder
from openpilot.tools.sim.bridge.gta5.gta5_rx import NV12_SIZE, SLOTS, VIEWS, rx_main
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
from openpilot.tools.sim.bridge.gta5.map.lane_match import JunctionAreas, LaneMatcher
from openpilot.tools.sim.bridge.gta5.map.map_view import MapView
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes
from openpilot.tools.sim.bridge.gta5.map.paths import CAR_HEIGHT, Paths
from openpilot.tools.sim.bridge.gta5.map.router import Navigator, Route, Router
from openpilot.tools.sim.lib.common import SimulatorState, World, vec3

OP_WHEELBASE = 2.7
OP_STEER_RATIO = 15.38
MIN_FRAME_SPACING = 0.040  # s
DEBUG = bool(os.getenv("GTA5_DEBUG"))  # print commanded vs measured motion each second
LOG = os.getenv("GTA5_LOG")  # a file to record the game state and the controls sent, a JSON line each, for analysis
CAMLOG = os.getenv("GTA5_CAMLOG")  # a file to record each camera frame handed to openpilot: its seq and rx's newest
MAP = os.getenv("GTA5_MAP")  # the map's folder (map/README.md): serves the map view
MAP_PORT = int(os.getenv("GTA5_MAP_PORT", "8793"))
MAP_EVERY = 0.1  # s
LANE_LINE_EVERY = 0.5  # s
ROUTER = os.getenv("GTA5_ROUTER")  # a Valhalla server on the map (map/README.md) routes, rather than the game's GPS
ROUTE_AHEAD, ROUTE_STEP = 1000.0, 5.0  # m: the route nav gets, in the form of the plugin's GTA route (500 m)
ON_ROUTE = 8.0  # m: the car's lane from the route's road, rather than the plugin's guess at the road it's on
# the route input of a route-conditioned driving model (navd/route_input.py); modeld reads it only if its model has one
ROUTE_INPUT = os.getenv("GTA5_ROUTE_INPUT", "1") != "0"  # 0: no route for the model, to A/B one model with and without
OFF_ROUTE_INPUT = 15.0  # m off the route: no route input, as gta5-train's labels
# route input v2's lane slots (navd/lane_slots.py; also in the model's route input), to their own file for watch_route.py
LANE_SLOTS = os.getenv("GTA5_LANE_SLOTS", "1") != "0"
# the car's set speed follows the map's speed limits along the route (GTA5_FOLLOW_LIMIT=1, as before navSpeed, for A/B runs);
# off, the set speed is the driver's alone and nav caps the speed at the limit (navSpeed, reason speedLimit)
FOLLOW_LIMIT = os.getenv("GTA5_FOLLOW_LIMIT") == "1"
# openpilot's navInstruction and navRoute for the onroad UI's navigation view (gta5_nav_msgs.py), on our router's route
NAV_MSGS = os.getenv("GTA5_NAV_MSGS") == "1"
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
# GTA has no speed limits; a guess from the street's name, mph: freeways, highways and routes, then anything else
SPEED_LIMITS = ((('Fwy', 'Freeway'), 65), (('Hwy', 'Highway', 'Route'), 55))
CITY_SPEED_LIMIT = 35
# the plugin reads the bridge's address from here when its gta5op.ini doesn't set one
BRIDGE_FILE = Path(os.getenv("GTA5_BRIDGE_FILE", "/mnt/c/Users/Public/gta5op-bridge.txt"))
PIN_UI = Path(__file__).parent / "pin_ui.ps1"
# Python's full garbage collections walk every object the bridge holds (over a million, mostly the maps) with the GIL
# held for 0.4-0.8 s, stalling the camera and sensor threads; the bridge makes next to no cyclic garbage, so they run
# only once it has been idle (no game state, or on foot) this long, at most every FULL_GC_EVERY s
FULL_GC_IDLE = 2.0  # s
FULL_GC_EVERY = 60.0  # s


def hold_full_collections():
  """Automatic garbage collection only of the young generations, which take a few ms (GTA5World._idle_gc does the rest)."""
  gen0, gen1, _ = gc.get_threshold()
  gc.set_threshold(gen0, gen1, 1 << 30)


def pin_ui() -> subprocess.Popen | None:
  """Keeps openpilot's UI window above the game's, from Windows (GTA5_PIN_UI=0 turns it off)."""
  if os.getenv("GTA5_PIN_UI", "1") == "0":
    return None
  try:
    script = subprocess.check_output(["wslpath", "-w", str(PIN_UI)], text=True).strip()
    return subprocess.Popen(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script, "-Loop"],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  except (OSError, subprocess.CalledProcessError) as e:
    print(f"gta5: couldn't keep the UI on top: {e}")
    return None


def own_ip() -> str:
  """The address the game can reach this machine at; under WSL that's the distro's IP, as localhost forwarding goes stale."""
  host = os.getenv("GTA5_BRIDGE_HOST")
  if host:
    return host
  try:
    return subprocess.check_output(["hostname", "-I"], text=True).split()[0]
  except (OSError, subprocess.CalledProcessError, IndexError):
    return "127.0.0.1"


def speed_limit(street: str) -> float:
  if not street:
    return 0.0
  mph = next((limit for words, limit in SPEED_LIMITS if any(w in street for w in words)), CITY_SPEED_LIMIT)
  return mph * 0.44704


class GTA5World(World):
  sets_blinkers = True
  sets_torque = True
  idle_since: float | None = None  # when the bridge last stopped driving (_idle_gc)
  next_full_gc = 0.0

  def __init__(self, simulator_state: SimulatorState, q: Queue, port: int):
    super().__init__(dual_camera=True)
    hold_full_collections()
    self.simulator_state = simulator_state
    self.q = q

    self.lock = threading.Lock()
    self.new_frame = threading.Condition(self.lock)
    self.state: dict | None = None
    self.slots: dict[str, tuple] = {}  # by view, the newest frame released to the camera thread: (slot, seq, state, arrived)
    self.pair: dict[str, tuple] = {}  # the frame being handed to openpilot, road and wide from the same one
    self.handed = -1  # its seq
    self.last_frame_time = 0.0
    self.sm = messaging.SubMaster(['carControl', 'carParams', 'vehicleParameters', 'modelV2', 'selfdriveState', 'carState', 'controlsState',
                                   'carOutput'])
    self.tesla = is_tesla()
    self.metric = Params().get_bool("IsMetric")
    self.next_hud = 0.0
    self.VM: VehicleModel | None = None
    self.last_status = 0.0
    self.presses: dict[str, int] = {}
    self.curvature = 0.0  # what the steering is set for
    self.steering = False  # whether the driver was steering
    self.pinner: subprocess.Popen | None = None
    self.log = open(LOG, "a", buffering=1) if LOG else None
    self.params = Params()
    self._init_nav()
    self.gnss = gta5_gnss.from_env()
    self.publishes_gps = self.gnss is not None
    self.expert = Expert(self._send, lambda: self.q.put(control_cmd_gen("cruise_cancel")), lambda: self._set_nav_desire(""))
    self.map_view = MapView(os.path.join(MAP, "roads.json"), MAP_PORT) if MAP else None
    self.navigator = Navigator(Router(ROUTER)) if ROUTER else None
    self.lane_matcher: LaneMatcher | None = None  # the map's lanes, for the car's lane by them (with its lane tags)
    self.junction_areas: JunctionAreas | None = None
    if self.navigator is not None and MAP and os.path.exists(os.path.join(MAP, "paths.jsonl")):
      threading.Thread(target=self._load_paths, args=(os.path.join(MAP, "paths.jsonl"),), daemon=True).start()
    self.route_writer = RouteInputWriter(ROUTE_LEN) if ROUTE_INPUT else None
    self.nav_msgs = NavMessages() if NAV_MSGS else None
    self.lanes_writer = RouteInputWriter(PREVIEW_LEN, lane_slots_path()) if LANE_SLOTS else None
    self.recorder = Recorder(RECORD, self) if RECORD else None
    self.overlay = Overlay()  # the plugin map debug overlay, while it asks for it
    self.gps = GpsRoute()  # our route on the game map, while the plugin asks for it
    if self.map_view:
      print(f"gta5: map view on http://localhost:{MAP_PORT}/")

    self.shm = {name: SharedMemory(create=True, size=NV12_SIZE * SLOTS) for name in VIEWS}
    self.rx_latest = multiprocessing.Value('q', -1, lock=False)  # the newest frame's seq that rx has written
    self.camlog = open(CAMLOG, "a", buffering=1) if CAMLOG else None
    frames_recv, frames_send = multiprocessing.Pipe(duplex=False)
    controls_recv, self.controls = multiprocessing.Pipe(duplex=False)
    ready_recv, ready_send = multiprocessing.Pipe(duplex=False)
    self.rx = multiprocessing.Process(name="gta5 rx", daemon=True, target=rx_main,
                                      args=(port, frames_send, controls_recv, ready_send, {n: m.name for n, m in self.shm.items()},
                                            self.rx_latest, display_from_env()))
    self.rx.start()
    error = ready_recv.recv() if ready_recv.poll(10) else "timed out"
    if error is not None:
      self.close("rx failed")
      raise RuntimeError(f"could not listen for the game on port {port}: {error}")
    self.frames = frames_recv
    threading.Thread(target=self._frame_reader, daemon=True).start()
    self.pinner = pin_ui()

    addr = f"{own_ip()}:{port}"
    try:
      BRIDGE_FILE.write_text(addr + "\n")
      print(f"gta5: waiting for the game at {addr} (written to {BRIDGE_FILE})")
    except OSError as e:
      print(f"gta5: waiting for the game at {addr}; set bridge={addr} in gta5op.ini ({BRIDGE_FILE}: {e})")

  def _init_nav(self):
    """navd's planner, and the GTA layer's parts around it: the simulated driver and the destination."""
    self.params.remove("NavDesire")  # a killed bridge can leave one
    self.nav = Planner(Tune(os.getenv("GTA5_NAVTUNE")), refresh=self.params.get_bool("TurnDesireRefresh"))
    self.driver = Driver(self._send, lambda: self.q.put(control_cmd_gen("cruise_cancel")))
    self.destination = Destination(self.params, self._send, ends=NOO != "on")
    self.next_map = 0.0
    self.lane_line: tuple = (None, 0.0, [])  # the route it's for, until when, and the line
    self.route: Route | None = None
    self.route_input: tuple | None = None  # (the Route it encodes, its RouteInput, which has its LaneSlots)
    self.routes = 0  # routes the navigator has made, counting reroutes, for tests to follow
    self.cap = 0.0
    self.cap_reason = ""
    self.nav_speed_pm = messaging.PubMaster(["navSpeed"]) if NAV_SPEED != "can" else None
    self.next_nav_speed = 0.0
    self.gps_route: list = []
    self.noo, self.noo_t = NOO != "off", 0.0
    self.matcher: MapMatcher | None = None  # with NAV_MATCH, once the map's roads are loaded
    self.match_t: float | None = None

  def _nav_drives(self) -> bool:
    """Navigate on openpilot: nav drives, rather than only guiding."""
    if NOO != "param":
      return NOO != "off"
    now = time.monotonic()
    if now - self.noo_t > NOO_CHECK:
      self.noo_t = now
      try:
        self.noo = self.params.get_bool("NavigateOnOpenpilot")
      except Exception:  # params built before the key existed
        self.noo = True
    return self.noo

  def _frame_reader(self):
    last_release = 0.0
    while True:
      try:
        slot, views, state, seq = self.frames.recv()
      except (EOFError, OSError):
        return
      arrived = time.monotonic()
      with self.lock:
        self.state = state
        self.last_frame_time = arrived
      if self.log:
        self.log.write(json.dumps({"mono": self.last_frame_time, "state": state}) + "\n")
      # modeld discards a road frame arriving within 25 ms of the last, and counts it as a drop that invalidates camera odometry;
      # after a late frame the camera thread would otherwise catch up by sending the next one immediately
      wait = MIN_FRAME_SPACING - (time.monotonic() - last_release)
      if wait > 0:
        time.sleep(wait)
      with self.new_frame:
        for name in views:
          self.slots[name] = (slot, seq, state, arrived)
        self.new_frame.notify_all()
      # signal at most one pending frame, for the same reason
      if self.image_lock.get_value() == 0:
        self.image_lock.release()
        last_release = time.monotonic()

  def _send(self, obj: dict):
    if self.log and obj.get("type") == "control":
      self.log.write(json.dumps({"mono": time.monotonic(), "control": obj}) + "\n")
    try:
      self.controls.send(obj)
    except (BrokenPipeError, OSError):
      pass

  # *** World interface ***

  def _send_hud(self):
    """The set speed for the game's HUD, as openpilot's UI shows it (km/h; 0 when not set)."""
    now = time.monotonic()
    if now < self.next_hud:
      return
    self.next_hud = now + 0.2
    cluster = self.sm['carState'].vCruiseCluster
    set_kph = self.sm['controlsState'].deprecated.vCruise if cluster == 0.0 else cluster
    self._send({"type": "hud", "setSpeed": set_kph if 0 < set_kph < 255 else 0, "metric": self.metric,
                "engaged": self.simulator_state.is_engaged})

  def apply_controls(self, steer_angle, throttle_out, brake_out):
    """Sends carControl's desired curvature and acceleration rather than the bridge's steer_angle/throttle/brake:
    those are shaped for the simulated Honda, not for the game car."""
    self.sm.update(0)
    if self.VM is None and self.sm.seen['carParams']:
      self.VM = VehicleModel(self.sm['carParams'])
    with self.lock:
      state = self.state
    if state is None or not state.get("inVehicle"):
      return
    self._send_hud()
    if not self.simulator_state.is_engaged:
      self._send({"type": "control", "active": False})
      return
    actuators = self.sm['carControl'].actuators
    # openpilot's curvature is right-positive (controlsd negates the vehicle model's); the plugin takes left-positive
    curvature = -actuators.curvature
    accel = float(actuators.accel)
    if self.tesla and self.VM is not None and self.sm.seen['carOutput']:
      # What a Model 3 executes: the angle after the car controller's rate and lateral jerk limits, through the car's
      # vehicle model, and the acceleration within its limits
      angle = self.sm['carOutput'].actuatorsOutput.steeringAngleDeg
      curvature = self.VM.calc_curvature(math.radians(angle), max(abs(state['vEgo']), 1.0), state.get("bank", 0.0))
      accel = float(np.clip(accel, TeslaParams.ACCEL_MIN, TeslaParams.ACCEL_MAX))
    self._send({"type": "control", "active": True, "curvature": curvature, "accel": accel})

    now = time.monotonic()
    if DEBUG and now - self.last_status > 1.0:
      self.last_status = now
      out = state.get("out", {})
      print(f"gta5: v={state['vEgo']:5.2f} curvature cmd={curvature:+.4f} meas={self.curvature:+.4f} " +
            f"accel cmd={actuators.accel:+.2f} meas={state['aMeas']:+.2f} | steer={out.get('steer', 0):+.2f} " +
            f"thr={out.get('throttle', 0):.2f} brk={out.get('brake', 0):.2f} latI={out.get('latI', 0):+.3f} lonI={out.get('lonI', 0):+.2f}",
            flush=True)

  def read_state(self):
    pass

  def read_sensors(self, simulator_state: SimulatorState):
    with self.lock:
      state = self.state
      fresh = time.monotonic() - self.last_frame_time < 0.5
    if state is None or not fresh or not state.get("inVehicle"):
      simulator_state.valid = False
      simulator_state.user_torque = 0
      self._idle_gc()
      return
    self.idle_since = None

    v = state["vEgo"]
    yaw_rate = state["yawRate"]  # left-positive
    # the game's heading is counterclockwise from north; openpilot's bearing is clockwise
    bearing = (-state["heading"]) % 360
    simulator_state.velocity = vec3(v * math.sin(math.radians(bearing)), v * math.cos(math.radians(bearing)), 0)
    simulator_state.bearing = bearing
    simulator_state.imu.bearing = bearing

    # the plugin reports the curvature the car follows, standing in for a steering angle sensor
    self.curvature = state.get("steerCurvature", yaw_rate / v if v > 2.0 else 0.0)
    grade, bank = state.get("grade", 0.0), state.get("bank", 0.0)  # radians: nose up, right side down (NED roll)
    if self.VM is not None:
      # The angle for the curvature the car follows, through the car's own fixed vehicle model (carParams', with no
      # offset), as a real sensor measures a physical angle: paramsd then learns the car's values, as on a real Model 3.
      # An angle through paramsd's learned model would confirm whatever it learned, letting its estimates wander.
      steer = self.VM.get_steer_from_curvature(self.curvature, max(abs(v), 1.0), bank)
      simulator_state.steering_angle = math.degrees(steer)
    else:
      simulator_state.steering_angle = math.degrees(self.curvature * OP_WHEELBASE * OP_STEER_RATIO)

    # IMU in the raw sensor frame locationd expects (it reads device = [-v2, -v1, -v0]; device is x fwd, y right, z down).
    # locationd rejects a gyro that disagrees with camera odometry's yaw rate, so a zero IMU fails on winding roads.
    # The accelerometer measures gravity too, from which locationd gets the pitch and roll that paramsd and the planner use.
    g = 9.81
    lat_accel = -v * yaw_rate - g * math.sin(bank) * math.cos(grade)  # device y (right) points away from a left turn's center
    fwd_accel = state["aMeas"] + g * math.sin(grade)
    simulator_state.imu.accelerometer = vec3(g * math.cos(grade) * math.cos(bank), -lat_accel, -fwd_accel)
    simulator_state.imu.gyroscope = vec3(yaw_rate, 0, 0)
    if self.gnss is not None:
      p = state["pos"]
      fix = self.gnss.update(time.monotonic(), p[0], p[1], p[2], simulator_state.velocity.x, simulator_state.velocity.y)
      if self.matcher is not None:
        self._match(fix, v, yaw_rate)

    user = state.get("user") or {}
    # driver input while engaged: gas overrides; the brake and steering disengage, since the game's steering has no torque
    # for openpilot to blend with, and fighting it soon trips the excessive actuation lockout
    simulator_state.user_gas = 1.0 if user.get("gas") else 0.0
    simulator_state.user_brake = 1.0 if user.get("brake") else 0.0
    steering = abs(user.get("steer", 0)) > 0.02
    if steering and not self.steering and self.simulator_state.is_engaged:
      self.q.put(control_cmd_gen("cruise_cancel"))
    self.steering = steering
    simulator_state.speed_limit = speed_limit(state.get("street", ""))
    # set once per step, held until the next: the car thread could read any value set in between
    simulator_state.user_torque = self.driver.stalk(state.get("indicator"), state["heading"], state["yawRate"],
                                                    self.sm['modelV2'].meta.laneChangeState, self.nav.signaling)
    desire = self.sm['modelV2'].meta.desireState
    turns = {"left": desire[log.Desire.turnLeft], "right": desire[log.Desire.turnRight], "keepLeft": desire[log.Desire.keepLeft],
             "keepRight": desire[log.Desire.keepRight]} if len(desire) > log.Desire.keepRight else {}
    self.gps_route = state.get("route") or []
    if self.navigator is not None:
      state = self._map_route(state, bearing)
      limits = state.get("limits")
      known = bool(limits) and limits[0][0] == 0 and limits[0][1] > 0
      if known:
        simulator_state.speed_limit = limits[0][1]
      simulator_state.speed_limit_follow = known and FOLLOW_LIMIT
    if self.nav_msgs is not None:
      # the game's pose stands in for the car's localizer (GPS and odometry)
      self.nav_msgs.update(self.route, v, self.navigator.router.osm if self.navigator is not None else None, self._lane_slots,
                           pose=(np.array(state["pos"][:2], dtype=float), bearing))
    for msg in self._overlay(state, v):
      self._send(msg)
    if self.expert.update(state, self.route, self.simulator_state.is_engaged):
      # the game's AI drives (gta5_expert.py): openpilot stays disengaged, and nav and pull-away wait
      self._set_cap(simulator_state, 0.0, "")
      self._set_blinkers(simulator_state)
      self._update_buttons(state)
      self._update_map(state, bearing, v)
      simulator_state.valid = True
      return
    drives = self._nav_drives()
    out = self.nav.update(nav_inputs(state, self.simulator_state.is_engaged, state.get("indicator"), turns, time.monotonic(), drives))
    self.driver.act(out)
    for d in out.desires:
      self._set_nav_desire(d)
    self._set_cap(simulator_state, out.cap, out.cap_reason)
    self._set_blinkers(simulator_state)
    if out.arrived:
      self.destination.arrived()
    self.driver.pull_away.update(state, self.simulator_state.is_engaged)
    self._update_buttons(state)
    self._update_map(state, bearing, v)
    simulator_state.valid = True

  def _idle_gc(self):
    """The full garbage collection hold_full_collections leaves out, while nothing is driving."""
    now = time.monotonic()
    if self.idle_since is None:
      self.idle_since = now
    elif now - self.idle_since > FULL_GC_IDLE and now >= self.next_full_gc:
      self.next_full_gc = now + FULL_GC_EVERY
      gc.collect()

  def _set_cap(self, simulator_state: SimulatorState, cap: float, reason: str):
    """nav's speed cap (m/s, 0 for none) to openpilot: as navSpeed for its planner, or as the car's set speed (NAV_SPEED)."""
    simulator_state.cruise_cap = self.cap = cap
    simulator_state.cap_set_speed = self.nav_speed_pm is None
    now = time.monotonic()
    if self.nav_speed_pm is None or (now < self.next_nav_speed and reason == self.cap_reason):
      self.cap_reason = reason
      return
    self.cap_reason, self.next_nav_speed = reason, now + NAV_SPEED_EVERY
    msg = messaging.new_message("navSpeed", valid=True)
    msg.navSpeed.speedCap = cap
    msg.navSpeed.reason = reason or "none"
    self.nav_speed_pm.send("navSpeed", msg)

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
    if NAV_MATCH and lanes is not None and self.gnss is not None:
      self.matcher = MapMatcher(RoadGraph.from_osm(lanes))
      print("gta5: routing from navd's map match of the GNSS")
    if lanes is not None and lanes.tagged:
      self.navigator.router.osm = lanes
      print(f"gta5: lanes from the map's tags ({osm})")
      self.lane_matcher = LaneMatcher(lanes)
      self.junction_areas = self._junction_areas(lanes)
    else:
      print("gta5: the map has no lane tags: lanes from GTA's links")

  def _match(self, fix, v: float, yaw_rate: float):
    """navd's map match, moved on each step by the car's speed and yaw rate and corrected by each GNSS fix."""
    now = time.monotonic()
    self.matcher.predict(0.0 if self.match_t is None else now - self.match_t, v, yaw_rate)
    self.match_t = now
    if fix is not None:
      g = getattr(fix, self.gnss.profile.service)
      x, y = to_game(g.latitude, g.longitude)
      self.matcher.update(x, y, g.bearingDeg, g.bearingAccuracyDeg, g.speed, self.gnss.profile.latency)

  def _map_route(self, state: dict, bearing: float) -> dict:
    """The state with our route to the destination, in the form of the plugin's GTA route, and what nav uses of the
    map along it. The destination is whichever was set last of the game map's waypoint (else a mission's GPS route's end)
    and the map view's."""
    pos = np.array(state["pos"][:2], dtype=float)
    picked = self.map_view.take_destination() if self.map_view is not None else None
    dest = self.destination.update(state.get("waypoint"), pos, picked, (state.get("mission") or {}).get("dest"))
    match = self.matcher.match if self.matcher is not None else None
    route = self.navigator.update(pos, bearing, dest, time.monotonic(), state["pos"][2], match=match)
    self.routes += route is not None and route is not self.route
    self.route = route
    self._write_route_input(state)
    self._write_lane_slots(state)
    state = {**state, "waypoint": dest.tolist() if dest is not None else None, "route": [], "laneMap": self._map_lane(state)}
    if self.route is None:
      return state
    on = self.route.off < ON_ROUTE
    lane, plugin = self.route.lane() if on else None, state.get("lane")
    frac = self.route.lane_frac() if lane else None
    if lane and plugin and lane[0] != plugin[0]:
      lane, frac = None, None  # they disagree: no lane changes on either
    elif not lane:
      lane = plugin
    return {**state, **self.route.info(ROUTE_AHEAD), "route": self.route.ahead(ROUTE_AHEAD, ROUTE_STEP).round(1).tolist(),
            "lane": lane, "lanePlugin": plugin, "laneFrac": frac, "twoWay": self.route.two_way() if on else None}

  @staticmethod
  def _junction_areas(osm: OsmLanes) -> JunctionAreas | None:
    """The map's junction areas from their cache, built by a separate process the first time (about half a minute)."""
    areas = JunctionAreas.cached(osm, build=False)
    if areas is not None:
      return areas
    print("gta5: building the map's junction areas in the background", flush=True)
    try:
      subprocess.run(["nice", "-n", "10", sys.executable, "-m", "openpilot.tools.sim.bridge.gta5.map.lane_match", osm.path],
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
                                   "areas": areas is not None}

  def _write_route_input(self, state: dict):
    """The driving model's route input for the route and the car's place on it; zero off it."""
    if self.route_writer is None:
      return
    if self.route is None or self.route.off > OFF_ROUTE_INPUT or not self._nav_drives():  # guiding only: no route for the model
      self.route_writer.write(np.zeros(ROUTE_LEN, np.float32))
      return
    enc = self._route_encoder(self.route)
    try:
      vec = enc.encode(self.route.at, state["heading"], state["vEgo"])
    except Exception as e:  # the lane slots on a map they can't read: the rest of the input carries on without them
      print(f"gta5: lane slots: {e!r}")
      enc = self._route_encoder(self.route, lanes=False)
      vec = enc.encode(self.route.at, state["heading"], state["vEgo"])
    self.route_writer.write(vec)

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

  def _write_lane_slots(self, state: dict):
    """Route input v2's lane slots for the preview; zero off the route."""
    if self.lanes_writer is None:
      return
    route = self.route if self.route is not None and self.route.off <= OFF_ROUTE_INPUT else None
    try:
      vec = preview(self._route_encoder(route).slots if route is not None else None, route.at if route is not None else 0.0,
                    state["vEgo"])
    except Exception as e:  # only a preview: it mustn't stop the bridge
      print(f"gta5: lane slots: {e!r}")
      vec = preview(None, 0.0, 0.0)
      if route is not None:
        self._route_encoder(route, lanes=False)
    self.lanes_writer.write(vec)

  def _overlay(self, state: dict, v: float) -> list[dict]:
    paths = self.navigator.router.paths if self.navigator is not None else None
    osm = self.navigator.router.osm if self.navigator is not None else None
    out = self.overlay.update(state, self.route, paths, lambda: self._lane_line(state, v), lambda: self._turn_points(state),
                              self.recorder is not None, osm)
    return out + self.gps.update(state, self.route)

  def _turn_points(self, state: dict):
    """The next turn navd signals and where its signal comes on, for the overlay."""
    return self.nav.turn_points(np.array(state["route"], dtype=float), state.get("forks"), state.get("stops"), state.get("junctions"))

  def _update_map(self, state: dict, bearing: float, v: float):
    now = time.monotonic()
    if self.map_view is None or now < self.next_map:
      return
    self.next_map = now + MAP_EVERY
    waypoint = state.get("waypoint")
    speed = f"{v * 3.6:.0f} km/h" if self.metric else f"{v / 0.44704:.0f} mph"
    self.map_view.update({
      "t": now,
      "car": {"x": state["pos"][0], "y": state["pos"][1], "z": round(state["pos"][2] - CAR_HEIGHT, 1), "bearing": bearing},
      "routes": {"gps": self.gps_route, "nav": [] if self.route is None else self.route.rest().round(1).tolist(),
                 "lanes": self._lane_line(state, v)},
      "waypoint": waypoint if waypoint and any(waypoint) else None,
      "text": f"{state.get('street', '')}  {speed}{'  engaged' if self.simulator_state.is_engaged else ''}",
      "nav": {"routes": self.routes, "length": None if self.route is None else round(self.route.length, 1),
              "at": None if self.route is None else round(self.route.at, 1), "cap": round(self.cap, 2)},
    })

  def _lane_line(self, state: dict, v: float) -> list:
    """The route in the lanes nav aims for, for the map."""
    r, now = self.route, time.monotonic()
    if r is None:
      return []
    if r is self.lane_line[0] and now < self.lane_line[1]:
      return self.lane_line[2]  # it can take several ms on a long route; the view trims it to the car
    forks = [[f.along - r.at, f.side, f.lanes, f.lanes_in, f.keep, f.other, f.slip] for f in r.forks if f.along > r.at]
    line = r.lane_line(lane_plan(r.rest(), forks, state.get("lane"), r.lanes_at, v, self.nav.tune, r.lane_arrows(r.length, 0.0),
                                 r.lane_drops(r.length, 0.0), r.lane_opens(r.length, 0.0)))
    self.lane_line = (r, now + LANE_LINE_EVERY, [] if line is None else line.round(1).tolist())
    return self.lane_line[2]

  def _set_nav_desire(self, desire: str):
    if desire:
      self.params.put("NavDesire", desire)
    else:
      self.params.remove("NavDesire")

  def _update_buttons(self, state: dict):
    """The plugin's keys stand in for the cruise buttons: engage sets when disengaged and cancels when engaged, and the
    speed keys are resume/accel and set/decel, which step the set speed while engaged (to the next multiple of 5 with
    shift)."""
    commands = {"speedUpPresses": "cruise_up", "speedDownPresses": "cruise_down", "speedUp5Presses": "cruise_up5",
                "speedDown5Presses": "cruise_down5"}
    for key in ("engagePresses", *commands):
      presses = state.get(key)
      if presses is None:
        continue
      new = presses - self.presses.get(key, presses)
      self.presses[key] = presses
      for _ in range(max(0, min(new, 5))):
        if key == "engagePresses":
          cmd = "cruise_cancel" if self.simulator_state.is_engaged else "cruise_down"
        else:
          cmd = commands[key]
        self.q.put(control_cmd_gen(cmd))

  def _set_blinkers(self, simulator_state: SimulatorState):
    # set once per step, from the indicator and nav's repeat gap: the car thread could read any value set in between
    simulator_state.left_blinker, simulator_state.right_blinker = self.driver.blinkers(self.nav.blinker_gap(time.monotonic()))

  def read_cameras(self):
    pass

  def camera_yuv(self, wide: bool) -> bytes | None:
    name = "wide" if wide else "road"
    with self.new_frame:
      if not wide:
        # each game frame once: the camera thread's wake-ups and the frames' arrival drift apart, which would hand some frames
        # twice and skip others; and the wide view comes from the same frame as the road view, even if another arrives between
        if self.slots.get("road", (None, -1))[1] == self.handed:
          self.new_frame.wait(0.1)
        self.pair = dict(self.slots)
        self.handed = self.pair.get("road", (None, -1))[1]
      slot, seq, state, arrived = self.pair.get(name, (None, -1, self.state, self.last_frame_time))
    frame = None
    if slot is not None:
      buf = self.shm[name].buf
      assert buf is not None
      frame = bytes(buf[slot * NV12_SIZE:(slot + 1) * NV12_SIZE])
      if self.camlog is not None:
        # rx reuses a slot SLOTS frames on: newest - seq >= SLOTS means this frame was written over before it was read
        self.camlog.write(f"{time.monotonic():.4f} {name} {seq} {self.rx_latest.value}\n")
    if self.recorder is not None:
      self.recorder.add(wide, frame, state, arrived)
    return frame

  def tick(self):
    pass

  def reset(self):
    self._send({"type": "control", "active": False})

  def close(self, reason: str):
    self._send({"type": "control", "active": False})
    if self.pinner is not None:
      self.pinner.terminate()
    if self.map_view is not None:
      self.map_view.close()
    self.exit_event.set()
    # the camera thread waits for each game frame, and none come once the game connection is gone; wake it so the bridge
    # process can exit, handing it a blank frame rather than the shared memory freed below
    with self.lock:
      self.slots.clear()
    self.image_lock.release()
    if self.rx.is_alive():
      self.rx.terminate()
      self.rx.join(2)
    for shm in self.shm.values():
      try:
        shm.close()
        shm.unlink()
      except FileNotFoundError:
        pass
    if self.recorder is not None:
      self.recorder.close()
    print(f"gta5: closing ({reason})")
