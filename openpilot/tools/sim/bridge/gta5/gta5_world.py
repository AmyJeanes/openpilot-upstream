import math
import multiprocessing
import os
import subprocess
import threading
import time
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

from opendbc.car.vehicle_model import VehicleModel
from openpilot.cereal import log, messaging
from openpilot.tools.sim.bridge.common import control_cmd_gen
from openpilot.tools.sim.bridge.gta5.gta5_rx import NV12_SIZE, SLOTS, VIEWS, rx_main
from openpilot.tools.sim.lib.common import SimulatorState, World, vec3

OP_WHEELBASE = 2.7
OP_STEER_RATIO = 15.38
MIN_FRAME_SPACING = 0.040  # s
DEBUG = bool(os.getenv("GTA5_DEBUG"))  # print commanded vs measured motion each second
# openpilot starts a signaled lane change on a steering nudge towards it; give that nudge for the driver.
# Positive is left, and it must exceed the simulated Honda's steeringPressed threshold.
NUDGE_TORQUE = 2000
NUDGE_TIMEOUT = 3.0  # s after the indicator comes on
# the plugin reads the bridge's address from here when its gta5op.ini doesn't set one
BRIDGE_FILE = Path(os.getenv("GTA5_BRIDGE_FILE", "/mnt/c/Users/Public/gta5op-bridge.txt"))
LaneChangeState = log.LaneChangeState


def own_ip() -> str:
  """The address the game can reach this machine at; under WSL that's the distro's IP, as localhost forwarding goes stale."""
  host = os.getenv("GTA5_BRIDGE_HOST")
  if host:
    return host
  try:
    return subprocess.check_output(["hostname", "-I"], text=True).split()[0]
  except (OSError, subprocess.CalledProcessError, IndexError):
    return "127.0.0.1"


class GTA5World(World):
  def __init__(self, simulator_state: SimulatorState, q: Queue, port: int):
    super().__init__(dual_camera=True)
    self.simulator_state = simulator_state
    self.q = q

    self.lock = threading.Lock()
    self.state: dict | None = None
    self.slots: dict[str, int] = {}
    self.last_frame_time = 0.0
    self.sm = messaging.SubMaster(['carControl', 'carParams', 'vehicleParameters', 'modelV2', 'selfdriveState'])
    self.VM: VehicleModel | None = None
    self.last_status = 0.0
    self.indicator: str | None = None
    self.indicator_t = 0.0
    self.lane_changing = False
    self.presses: dict[str, int] = {}
    self.curvature = 0.0  # what the steering is set for

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
    try:
      self.controls.send(obj)
    except (BrokenPipeError, OSError):
      pass

  # *** World interface ***

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
    if not self.simulator_state.is_engaged:
      self._send({"type": "control", "active": False})
      return
    actuators = self.sm['carControl'].actuators
    # openpilot's curvature is right-positive (controlsd negates the vehicle model's); the plugin takes left-positive
    curvature = -actuators.curvature
    self._send({"type": "control", "active": True, "curvature": curvature, "accel": float(actuators.accel)})

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

    # the plugin reports the curvature its steering is set for, standing in for a steering angle sensor
    self.curvature = state.get("steerCurvature", yaw_rate / v if v > 2.0 else 0.0)
    if self.VM is not None:
      # invert controlsd's measured curvature using the same learned params, so the reported angle matches its target
      lp = self.sm['vehicleParameters']
      roll = 0.0
      offset = 0.0
      if self.sm.seen['vehicleParameters']:
        self.VM.update_params(max(lp.stiffnessFactor, 0.1), max(lp.steerRatio, 0.1))
        roll, offset = lp.roll, lp.angleOffsetDeg
      simulator_state.steering_angle = math.degrees(self.VM.get_steer_from_curvature(self.curvature, max(abs(v), 1.0), roll)) + offset
    else:
      simulator_state.steering_angle = math.degrees(self.curvature * OP_WHEELBASE * OP_STEER_RATIO)

    # IMU in the raw sensor frame locationd expects (it reads device = [-v2, -v1, -v0]; device is x fwd, y right, z down).
    # locationd rejects a gyro that disagrees with camera odometry's yaw rate, so a zero IMU fails on winding roads.
    lat_accel = -v * yaw_rate  # device y (right) points away from a left turn's center
    simulator_state.imu.accelerometer = vec3(9.81, -lat_accel, -state["aMeas"])
    simulator_state.imu.gyroscope = vec3(yaw_rate, 0, 0)

    user = state.get("user") or {}
    # driver input while engaged: the brake disengages, gas and steering override, as in a real car
    simulator_state.user_gas = 1.0 if user.get("gas") else 0.0
    simulator_state.user_brake = 1.0 if user.get("brake") else 0.0
    simulator_state.user_torque = -math.copysign(10000, user["steer"]) if abs(user.get("steer", 0)) > 0.02 else 0
    self._update_indicator(simulator_state, state.get("indicator"))
    self._update_buttons(state)
    simulator_state.valid = True

  def _update_buttons(self, state: dict):
    """The plugin's keys stand in for the cruise buttons: engage sets when disengaged and cancels when engaged, and the
    speed keys are resume/accel and set/decel, which step the set speed while engaged."""
    for key in ("engagePresses", "speedUpPresses", "speedDownPresses"):
      presses = state.get(key)
      if presses is None:
        continue
      new = presses - self.presses.get(key, presses)
      self.presses[key] = presses
      for _ in range(max(0, min(new, 5))):
        if key == "engagePresses":
          cmd = "cruise_cancel" if self.simulator_state.is_engaged else "cruise_down"
        else:
          cmd = "cruise_up" if key == "speedUpPresses" else "cruise_down"
        self.q.put(control_cmd_gen(cmd))

  def _update_indicator(self, simulator_state: SimulatorState, indicator: str | None):
    """The plugin's indicator is the blinker stalk: nudge the wheel to start the lane change, and cancel the indicator
    once it is done, as a car's stalk would."""
    now = time.monotonic()
    if indicator != self.indicator:
      self.indicator, self.indicator_t, self.lane_changing = indicator, now, False
    simulator_state.left_blinker = indicator == "left"
    simulator_state.right_blinker = indicator == "right"
    if indicator is None:
      return
    lane_change = self.sm['modelV2'].meta.laneChangeState
    if lane_change == LaneChangeState.laneChangeStarting:
      self.lane_changing = True
    elif self.lane_changing:
      self.lane_changing = False
      self._send({"type": "indicatorOff"})
    elif lane_change == LaneChangeState.preLaneChange and now - self.indicator_t < NUDGE_TIMEOUT and simulator_state.user_torque == 0:
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
