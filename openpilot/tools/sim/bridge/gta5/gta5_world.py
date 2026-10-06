import json
import math
import multiprocessing
import os
import subprocess
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
from openpilot.tools.sim.lib.simulated_tesla import is_tesla
from openpilot.tools.sim.bridge.common import control_cmd_gen
from openpilot.tools.sim.bridge.gta5.gta5_expert import Expert
from openpilot.tools.sim.bridge.gta5.gta5_nav import Nav, PullAway, lane_plan
from openpilot.tools.sim.bridge.gta5.gta5_overlay import GpsRoute, Overlay
from openpilot.tools.sim.bridge.gta5.gta5_record import RECORD, Recorder
from openpilot.tools.sim.bridge.gta5.gta5_route_input import ROUTE_LEN, RouteInput
from openpilot.tools.sim.bridge.gta5.gta5_rx import NV12_SIZE, SLOTS, VIEWS, rx_main
from openpilot.tools.sim.bridge.gta5.map.map_view import MapView
from openpilot.tools.sim.bridge.gta5.map.paths import Paths
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
# the route input of a route-conditioned driving model (gta5_route_input.py); modeld reads it only if its model has one
ROUTE_INPUT = os.getenv("GTA5_ROUTE_INPUT", "1") != "0"  # 0: no route for the model, to A/B one model with and without
OFF_ROUTE_INPUT = 15.0  # m off the route: no route input, as gta5-train's labels
FOLLOW_LIMIT = os.getenv("GTA5_FOLLOW_LIMIT", "1") != "0"  # the set speed follows the map's speed limits along the route
CANCELLED_FROM = 100.0  # m: GTA clears the waypoint as the car nears it; farther off, the player cleared it
# openpilot starts a signaled lane change on a steering nudge towards it; give that nudge for the driver.
# Positive is left, and it must exceed the simulated Honda's steeringPressed threshold.
NUDGE_TORQUE = 2000
NUDGE_TIMEOUT = 3.0  # s after the indicator comes on
TURN_CANCEL_DEG = 60.0  # heading change since the indicator came on that counts as a turn taken
TURN_CANCEL_YAW_RATE = 0.1  # rad/s, straightened out
# GTA has no speed limits; a guess from the street's name, mph: freeways, highways and routes, then anything else
SPEED_LIMITS = ((('Fwy', 'Freeway'), 65), (('Hwy', 'Highway', 'Route'), 55))
CITY_SPEED_LIMIT = 35
# the plugin reads the bridge's address from here when its gta5op.ini doesn't set one
BRIDGE_FILE = Path(os.getenv("GTA5_BRIDGE_FILE", "/mnt/c/Users/Public/gta5op-bridge.txt"))
PIN_UI = Path(__file__).parent / "pin_ui.ps1"
LaneChangeState = log.LaneChangeState


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

  def __init__(self, simulator_state: SimulatorState, q: Queue, port: int):
    super().__init__(dual_camera=True)
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
    self.indicator: str | None = None
    self.indicator_t = 0.0
    self.indicator_heading = 0.0
    self.lane_changing = False
    self.presses: dict[str, int] = {}
    self.curvature = 0.0  # what the steering is set for
    self.steering = False  # whether the driver was steering
    self.pinner: subprocess.Popen | None = None
    self.log = open(LOG, "a", buffering=1) if LOG else None
    self.params = Params()
    self.params.remove("NavDesire")  # a killed bridge can leave one
    self.nav = Nav(self._send, self._set_nav_desire)
    self.pull_away = PullAway(self._send)
    self.expert = Expert(self._send, lambda: self.q.put(control_cmd_gen("cruise_cancel")), lambda: self._set_nav_desire(""))
    self.map_view = MapView(os.path.join(MAP, "roads.json"), MAP_PORT) if MAP else None
    self.next_map = 0.0
    self.lane_line: tuple = (None, 0.0, [])  # the route it's for, until when, and the line
    self.navigator = Navigator(Router(ROUTER)) if ROUTER else None
    if self.navigator is not None and MAP and os.path.exists(os.path.join(MAP, "paths.jsonl")):
      threading.Thread(target=self._load_paths, args=(os.path.join(MAP, "paths.jsonl"),), daemon=True).start()
    self.dest: np.ndarray | None = None
    self.dest_from_game = False
    self.game_waypoint: np.ndarray | None = None
    self.route: Route | None = None
    self.route_writer = RouteInputWriter(ROUTE_LEN) if ROUTE_INPUT else None
    self.route_input: tuple | None = None  # (the Route it encodes, its RouteInput)
    self.routes = 0  # routes the navigator has made, counting reroutes, for tests to follow
    self.cap = 0.0
    self.gps_route: list = []
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
                                            self.rx_latest))
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
      return

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

    user = state.get("user") or {}
    # driver input while engaged: gas overrides; the brake and steering disengage, since the game's steering has no torque
    # for openpilot to blend with, and fighting it soon trips the excessive actuation lockout
    simulator_state.user_gas = 1.0 if user.get("gas") else 0.0
    simulator_state.user_brake = 1.0 if user.get("brake") else 0.0
    steering = abs(user.get("steer", 0)) > 0.02
    if steering and not self.steering and self.simulator_state.is_engaged:
      self.q.put(control_cmd_gen("cruise_cancel"))
    self.steering = steering
    simulator_state.user_torque = 0  # but for the lane change nudge
    simulator_state.speed_limit = speed_limit(state.get("street", ""))
    self._update_indicator(simulator_state, state.get("indicator"), state["heading"], state["yawRate"])
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
    for msg in self._overlay(state, v):
      self._send(msg)
    if self.expert.update(state, self.route, self.simulator_state.is_engaged):
      # the game's AI drives (gta5_expert.py): openpilot stays disengaged, and nav and pull-away wait
      simulator_state.cruise_cap = self.cap = 0.0
      self._set_blinkers(simulator_state)
      self._update_buttons(state)
      self._update_map(state, bearing, v)
      simulator_state.valid = True
      return
    simulator_state.cruise_cap, arrived = self.nav.update(state, self.simulator_state.is_engaged, state.get("indicator"), turns)
    self.cap = simulator_state.cruise_cap
    self._set_blinkers(simulator_state)
    if arrived:
      self.q.put(control_cmd_gen("cruise_cancel"))
      self.dest = None
    self.pull_away.update(state, self.simulator_state.is_engaged)
    self._update_buttons(state)
    self._update_map(state, bearing, v)
    simulator_state.valid = True

  def _load_paths(self, path: str):
    try:
      paths = Paths(path)
      paths.index()
      self.navigator.router.paths = paths
    except (OSError, ValueError, KeyError) as e:
      print(f"gta5: no road heights or lanes for routes: {e}")

  def _map_route(self, state: dict, bearing: float) -> dict:
    """The state with our route to the destination, in the form of the plugin's GTA route, and what nav uses of the
    map along it. The destination is whichever was set last of the game map's waypoint and the map view's."""
    pos = np.array(state["pos"][:2], dtype=float)
    waypoint = np.array(state.get("waypoint") or (0.0, 0.0), dtype=float)
    if waypoint.any():
      if self.game_waypoint is None or np.hypot(*(waypoint - self.game_waypoint)) > 1.0:
        self.dest, self.dest_from_game = waypoint, True
      self.game_waypoint = waypoint
    else:
      self.game_waypoint = None
      if self.dest_from_game and self.dest is not None and np.hypot(*(self.dest - pos)) > CANCELLED_FROM:
        self.dest = None
    if self.map_view is not None:
      picked = self.map_view.take_destination()
      if picked is not None:
        self.dest, self.dest_from_game = (np.array(picked[0], dtype=float) if picked[0] else None), False
    route = self.navigator.update(pos, bearing, self.dest, time.monotonic(), state["pos"][2])
    self.routes += route is not None and route is not self.route
    self.route = route
    self._write_route_input(state)
    state = {**state, "waypoint": self.dest.tolist() if self.dest is not None else None, "route": []}
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

  def _write_route_input(self, state: dict):
    """The driving model's route input for the route and the car's place on it; zero off it."""
    if self.route_writer is None:
      return
    if self.route is None or self.route.off > OFF_ROUTE_INPUT:
      self.route_writer.write(np.zeros(ROUTE_LEN, np.float32))
      return
    if self.route_input is None or self.route_input[0] is not self.route:
      self.route_input = (self.route, RouteInput(self.route))  # once per route, a few ms
    self.route_writer.write(self.route_input[1].encode(self.route.at, state["heading"]))

  def _overlay(self, state: dict, v: float) -> list[dict]:
    paths = self.navigator.router.paths if self.navigator is not None else None
    out = self.overlay.update(state, self.route, paths, lambda: self._lane_line(state, v), self.nav, self.recorder is not None)
    return out + self.gps.update(state, self.route)

  def _update_map(self, state: dict, bearing: float, v: float):
    now = time.monotonic()
    if self.map_view is None or now < self.next_map:
      return
    self.next_map = now + MAP_EVERY
    waypoint = state.get("waypoint")
    speed = f"{v * 3.6:.0f} km/h" if self.metric else f"{v / 0.44704:.0f} mph"
    self.map_view.update({
      "t": now,
      "car": {"x": state["pos"][0], "y": state["pos"][1], "bearing": bearing},
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
    line = r.lane_line(lane_plan(r.rest(), forks, state.get("lane"), r.lanes_at, v, self.nav.tune))
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
    gap = self.nav.blinker_gap
    simulator_state.left_blinker = self.indicator == "left" and not gap
    simulator_state.right_blinker = self.indicator == "right" and not gap

  def _update_indicator(self, simulator_state: SimulatorState, indicator: str | None, heading: float, yaw_rate: float):
    """The plugin's indicator is the blinker stalk: nudge the wheel to start the lane change, and cancel the indicator
    once it is done, as a car's stalk would."""
    now = time.monotonic()
    if indicator != self.indicator:
      self.indicator, self.indicator_t, self.lane_changing = indicator, now, False
      self.indicator_heading = heading
    if indicator is None:
      return
    # a turn: cancel once the car has come round and straightened out
    turned = abs((heading - self.indicator_heading + 180) % 360 - 180)
    if turned > TURN_CANCEL_DEG and abs(yaw_rate) < TURN_CANCEL_YAW_RATE:
      self._send({"type": "indicatorOff"})
      self.indicator_heading = heading  # once, until the plugin's indicator goes off
      return
    lane_change = self.sm['modelV2'].meta.laneChangeState
    if lane_change == LaneChangeState.laneChangeStarting:
      self.lane_changing = True
    elif self.lane_changing:
      self.lane_changing = False
      if not self.nav.signaling:  # a lane change still going as the blinker became the turn's
        self._send({"type": "indicatorOff"})
    elif (lane_change == LaneChangeState.preLaneChange and now - self.indicator_t < NUDGE_TIMEOUT and simulator_state.user_torque == 0
          and not self.nav.signaling):  # a turn on the route, not a lane change
      simulator_state.user_torque = NUDGE_TORQUE if indicator == "left" else -NUDGE_TORQUE

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
