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
from openpilot.tools.sim.lib.simulated_tesla import is_tesla
from openpilot.tools.sim.bridge.common import control_cmd_gen
from openpilot.tools.sim.bridge.gta5.gta5_nav import Nav, PullAway
from openpilot.tools.sim.bridge.gta5.gta5_rx import NV12_SIZE, SLOTS, VIEWS, rx_main
from openpilot.tools.sim.lib.common import SimulatorState, World, vec3

OP_WHEELBASE = 2.7
OP_STEER_RATIO = 15.38
MIN_FRAME_SPACING = 0.040  # s
DEBUG = bool(os.getenv("GTA5_DEBUG"))  # print commanded vs measured motion each second
LOG = os.getenv("GTA5_LOG")  # a file to record the game state and the controls sent, a JSON line each, for analysis
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
  def __init__(self, simulator_state: SimulatorState, q: Queue, port: int):
    super().__init__(dual_camera=True)
    self.simulator_state = simulator_state
    self.q = q

    self.lock = threading.Lock()
    self.state: dict | None = None
    self.slots: dict[str, int] = {}
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

    self.shm = {name: SharedMemory(create=True, size=NV12_SIZE * SLOTS) for name in VIEWS}
    frames_recv, frames_send = multiprocessing.Pipe(duplex=False)
    controls_recv, self.controls = multiprocessing.Pipe(duplex=False)
    ready_recv, ready_send = multiprocessing.Pipe(duplex=False)
    self.rx = multiprocessing.Process(name="gta5 rx", daemon=True, target=rx_main,
                                      args=(port, frames_send, controls_recv, ready_send, {n: m.name for n, m in self.shm.items()}))
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
        slot, views, state = self.frames.recv()
      except (EOFError, OSError):
        return
      with self.lock:
        for name in views:
          self.slots[name] = slot
        self.state = state
        self.last_frame_time = time.monotonic()
      if self.log:
        self.log.write(json.dumps({"mono": self.last_frame_time, "state": state}) + "\n")
      # modeld discards a road frame arriving within 25 ms of the last, and counts it as a drop that invalidates camera odometry;
      # after a late frame the 20 Hz camera thread would otherwise catch up by sending the next one immediately
      wait = MIN_FRAME_SPACING - (time.monotonic() - last_release)
      if wait > 0:
        time.sleep(wait)
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
    turns = {"left": desire[log.Desire.turnLeft], "right": desire[log.Desire.turnRight]} if len(desire) > log.Desire.turnRight else {}
    simulator_state.cruise_cap, arrived = self.nav.update(state, self.simulator_state.is_engaged, state.get("indicator"), turns)
    if self.nav.blinker_gap:
      simulator_state.left_blinker = simulator_state.right_blinker = False
    if arrived:
      self.q.put(control_cmd_gen("cruise_cancel"))
    self.pull_away.update(state, self.simulator_state.is_engaged)
    self._update_buttons(state)
    simulator_state.valid = True

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

  def _update_indicator(self, simulator_state: SimulatorState, indicator: str | None, heading: float, yaw_rate: float):
    """The plugin's indicator is the blinker stalk: nudge the wheel to start the lane change, and cancel the indicator
    once it is done, as a car's stalk would."""
    now = time.monotonic()
    if indicator != self.indicator:
      self.indicator, self.indicator_t, self.lane_changing = indicator, now, False
      self.indicator_heading = heading
    simulator_state.left_blinker = indicator == "left"
    simulator_state.right_blinker = indicator == "right"
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
      self._send({"type": "indicatorOff"})
    elif (lane_change == LaneChangeState.preLaneChange and now - self.indicator_t < NUDGE_TIMEOUT and simulator_state.user_torque == 0
          and not self.nav.signaling):  # a turn on the route, not a lane change
      simulator_state.user_torque = NUDGE_TORQUE if indicator == "left" else -NUDGE_TORQUE

  def read_cameras(self):
    pass

  def camera_yuv(self, wide: bool) -> bytes | None:
    name = "wide" if wide else "road"
    with self.lock:
      slot = self.slots.get(name)
    if slot is None:
      return None
    buf = self.shm[name].buf
    assert buf is not None
    return bytes(buf[slot * NV12_SIZE:(slot + 1) * NV12_SIZE])

  def tick(self):
    pass

  def reset(self):
    self._send({"type": "control", "active": False})

  def close(self, reason: str):
    self._send({"type": "control", "active": False})
    if self.pinner is not None:
      self.pinner.terminate()
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
    print(f"gta5: closing ({reason})")
