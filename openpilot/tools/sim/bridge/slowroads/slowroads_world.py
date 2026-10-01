import math
import multiprocessing
import os
import threading
import time
from multiprocessing import Queue
from multiprocessing.shared_memory import SharedMemory

from opendbc.car.vehicle_model import VehicleModel
from openpilot.cereal import log, messaging
from openpilot.tools.sim.bridge.common import control_cmd_gen
from openpilot.tools.sim.lib.common import SimulatorState, World, vec3
from openpilot.tools.sim.bridge.slowroads.slowroads_inject import GAME_HOST, install
from openpilot.tools.sim.bridge.slowroads.slowroads_rx import NV12_SIZE, SLOTS, VIEWS, rx_main

# Steering is exchanged with the game as path curvature, so any in-game vehicle follows openpilot's plan
# regardless of its geometry; reported angles go through the simulated car's vehicle model to stay consistent.
OP_WHEELBASE = 2.7
OP_STEER_RATIO = 15.38
MIN_FRAME_SPACING = 0.040  # s
DEBUG = bool(os.getenv("SLOWROADS_DEBUG"))  # print commanded vs measured motion each second
# openpilot starts a signaled lane change on a steering nudge towards it; give that nudge for the driver.
# Positive is left, and it must exceed the simulated Honda's steeringPressed threshold.
NUDGE_TORQUE = 2000
NUDGE_TIMEOUT = 3.0  # s after the indicator comes on
LaneChangeState = log.LaneChangeState
AUTODRIVE_SETTLE_TIME = 2.0  # s
RESUME_WINDOW = 3.0  # s after the game resets the car
RESUME_INTERVAL = 0.5  # s


class SlowRoadsWorld(World):
  def __init__(self, simulator_state: SimulatorState, q: Queue, port: int, debug_port: int, inject: bool):
    super().__init__(dual_camera=True)
    self.simulator_state = simulator_state
    self.q = q
    self.port = port
    self.debug_port = debug_port

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
    self.autodrive: bool | None = None  # as last reported by the game
    self.autodrive_target: bool | None = None  # what the game was last asked for, by the driver or by us
    self.autodrive_target_t = 0.0
    self.driver_toggle_t = -math.inf
    self.resets: int | None = None
    self.reset_t = -math.inf
    self.resume_t = -math.inf

    self.shm = {name: SharedMemory(create=True, size=NV12_SIZE * SLOTS) for name in VIEWS}
    frames_recv, frames_send = multiprocessing.Pipe(duplex=False)
    controls_recv, self.controls = multiprocessing.Pipe(duplex=False)
    ready_recv, ready_send = multiprocessing.Pipe(duplex=False)
    self.rx = multiprocessing.Process(name="slowroads rx", daemon=True, target=rx_main,
                                      args=(port, frames_send, controls_recv, ready_send, {n: m.name for n, m in self.shm.items()}))
    self.rx.start()
    error = ready_recv.recv() if ready_recv.poll(10) else "timed out"
    if error is not None:
      self.close("rx failed")
      raise RuntimeError(f"could not start the game WebSocket server on port {port}: {error}")
    self.frames = frames_recv
    threading.Thread(target=self._frame_reader, daemon=True).start()

    if inject:
      self.inject()

  # *** game connection ***

  def inject(self):
    print(f"slowroads: {install(self.port, GAME_HOST, self.debug_port)}")

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
    those are shaped for the simulated Honda (its vehicle model and pedal scaling), not for the game car."""
    self.sm.update(0)
    if self.VM is None and self.sm.seen['carParams']:
      self.VM = VehicleModel(self.sm['carParams'])
    with self.lock:
      state = self.state
    if state is None:
      return
    if not self.simulator_state.is_engaged:
      self._send({"type": "control", "active": False})
      return
    actuators = self.sm['carControl'].actuators
    # openpilot curvature is right-positive (the vehicle model's is left-positive, see controlsd); the game steers left-positive
    curvature = -actuators.curvature
    game_steer = math.atan(curvature * state["wheelBase"])
    self._send({"type": "control", "active": True, "steer": game_steer, "accelCmd": float(actuators.accel)})

    now = time.monotonic()
    if DEBUG and now - self.last_status > 1.0:
      self.last_status = now
      v, game_curv = state["vEgo"], math.tan(state["steer"]) / state["wheelBase"]
      yaw_curv = state["yawRate"] / max(v, 1.0)
      print(f"slowroads: v={v:5.2f} curvature cmd={curvature:+.4f} game={game_curv:+.4f} yaw={yaw_curv:+.4f} " +
            f"accel cmd={actuators.accel:+.2f} meas={state['aMeas']:+.2f}", flush=True)

  def read_state(self):
    pass

  def read_sensors(self, simulator_state: SimulatorState):
    with self.lock:
      state = self.state
      fresh = time.monotonic() - self.last_frame_time < 0.5
    if state is None or not fresh:
      simulator_state.valid = False
      return

    heading = state["heading"]
    v = state["vEgo"]
    # heading is the game's yaw about +Y (left positive); x/y here are a flat east/north-style plane for GPS only
    simulator_state.velocity = vec3(v * math.cos(heading), v * math.sin(heading), 0)
    simulator_state.bearing = math.degrees(heading)
    simulator_state.imu.bearing = math.degrees(heading)
    curvature = math.tan(state["steer"]) / state["wheelBase"]
    if self.VM is not None:
      # invert controlsd's measured curvature using the same learned params, so the reported angle matches its target
      lp = self.sm['vehicleParameters']
      roll = 0.0
      offset = 0.0
      if self.sm.seen['vehicleParameters']:
        self.VM.update_params(max(lp.stiffnessFactor, 0.1), max(lp.steerRatio, 0.1))
        roll, offset = lp.roll, lp.angleOffsetDeg
      simulator_state.steering_angle = math.degrees(self.VM.get_steer_from_curvature(curvature, max(abs(v), 1.0), roll)) + offset
    else:
      simulator_state.steering_angle = math.degrees(curvature * OP_WHEELBASE * OP_STEER_RATIO)

    # IMU in the raw sensor frame locationd expects (it reads device = [-v2, -v1, -v0]; device is x fwd, y right, z down).
    # locationd rejects a gyro that disagrees with camera odometry's yaw rate, so a zero IMU fails on winding roads.
    yaw_rate = state["yawRate"]  # left-positive
    lat_accel = -v * yaw_rate  # device y (right) points away from a left turn's center
    simulator_state.imu.accelerometer = vec3(9.81, -lat_accel, -state["aMeas"])
    simulator_state.imu.gyroscope = vec3(yaw_rate, 0, 0)

    user = state.get("user") or {}
    # driver input in game while engaged disengages openpilot like a real car would
    simulator_state.user_gas = 1.0 if user.get("accel") else 0.0
    simulator_state.user_brake = 1.0 if user.get("brake") else 0.0
    simulator_state.user_torque = -math.copysign(10000, user["steer"]) if abs(user.get("steer", 0)) > 0.02 else 0
    self._update_indicator(simulator_state, state.get("indicator"))
    self._update_autodrive(state.get("autodrive"))
    self._resume_after_reset(state.get("resets"), v)
    simulator_state.valid = True

  def _resume_after_reset(self, resets: int | None, v_ego: float):
    """The game puts a stopped or strayed car back on the road at a standstill, where openpilot waits for resume like after
    any stop; without it the game sees the car stopped and resets it again."""
    now = time.monotonic()
    if resets is not None and self.resets is not None and resets != self.resets:
      self.reset_t = now
    self.resets = resets
    if (self.simulator_state.is_engaged and abs(v_ego) < 0.5 and now - self.reset_t < RESUME_WINDOW
        and now - self.resume_t > RESUME_INTERVAL):
      self.q.put(control_cmd_gen("cruise_up"))
      self.resume_t = now

  def _update_autodrive(self, autodrive: bool | None):
    """The game's autodrive toggle is openpilot's engage button, and shows whether openpilot is engaged."""
    if autodrive is None:
      return
    now = time.monotonic()
    if self.autodrive is not None and autodrive != self.autodrive and autodrive != self.autodrive_target:
      # the driver toggled it, or the game dropped it on a pedal or steering input
      self.q.put(control_cmd_gen("cruise_down" if autodrive else "cruise_cancel"))
      self.autodrive_target, self.autodrive_target_t, self.driver_toggle_t = autodrive, now, now
    self.autodrive = autodrive

    engaged = self.simulator_state.is_engaged
    # leave openpilot time to act on the driver's toggle before showing its state, unless it can't engage at all right now
    # (e.g. while calibrating), when the toggle goes straight back off; openpilot's own alert says why
    cannot_engage = autodrive and not engaged and self.sm.seen['selfdriveState'] and not self.sm['selfdriveState'].engageable
    settled = now - self.driver_toggle_t > AUTODRIVE_SETTLE_TIME or cannot_engage
    if settled and autodrive != engaged and (self.autodrive_target != engaged or now - self.autodrive_target_t > 1.0):
      self._send({"type": "autodrive", "on": engaged})
      self.autodrive_target, self.autodrive_target_t = engaged, now

  def _update_indicator(self, simulator_state: SimulatorState, indicator: str | None):
    """The game's indicator is the blinker stalk: nudge the wheel to start the lane change, and cancel the indicator
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
    # process can exit (else Ctrl-C needs pressing twice), handing it a blank frame rather than the shared memory freed below
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
    print(f"slowroads: closing ({reason})")
