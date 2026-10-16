"""The GTA V bridge's world (openpilot's tools/sim World): the game's frames to openpilot's cameras, its state to the
simulated car's sensors, and the plugin's controls from whoever drives.

The decisions (nav, the drivers, expert mode, the overlay, the map view, marks) are the brain's (gta5_brain.py), which
talks to openpilot through LocalIO here. Run with GTA5_LINK=1 the world is the link instead: no brain, openpilot's side
of the Windows host (gta5_host.py), which runs the brain and reaches this LocalIO through gta5_link.LinkServer."""
import gc
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
from opendbc.car.tesla.values import CarControllerParams as TeslaParams
from opendbc.car.vehicle_model import VehicleModel
from openpilot.cereal import log, messaging
from openpilot.common.params import Params
from openpilot.selfdrive.modeld.route_input import RouteInputWriter
from openpilot.selfdrive.navd.lane_slots import PREVIEW_LEN, lane_slots_path
from openpilot.selfdrive.navd.route_input import ROUTE_LEN
from openpilot.tools.sim.lib.simulated_tesla import is_tesla
from openpilot.tools.sim.bridge.common import control_cmd_gen
from openpilot.tools.sim.bridge.gta5 import gta5_gnss, gta5_stacks
from openpilot.tools.sim.bridge.gta5.gta5_brain import MAP, NAV_MATCH, Brain, speed_limit
from openpilot.tools.sim.bridge.gta5.gta5_cmd import display_from_env
from openpilot.tools.sim.bridge.gta5.gta5_opio import INPUTS, PARAMS, OpenpilotIO
from openpilot.tools.sim.bridge.gta5.gta5_drivers import INACTIVE
from openpilot.tools.sim.bridge.gta5.gta5_expert import CONTROL, control_path
from openpilot.tools.sim.bridge.gta5.gta5_record import RECORD, Recorder
from openpilot.tools.sim.bridge.gta5.gta5_rx import NV12_SIZE, SLOTS, VIEWS, rx_main
from openpilot.tools.sim.bridge.gta5.map.map_view import MapView
from openpilot.tools.sim.lib.common import SimulatorState, World, vec3

OP_WHEELBASE = 2.7
OP_STEER_RATIO = 15.38
MIN_FRAME_SPACING = 0.040  # s
DEBUG = bool(os.getenv("GTA5_DEBUG"))  # print commanded vs measured motion each second
LOG = os.getenv("GTA5_LOG")  # a file to record the game state and the controls sent, a JSON line each, for analysis
CAMLOG = os.getenv("GTA5_CAMLOG")  # a file to record each camera frame handed to openpilot: its seq and rx's newest
MAP_PORT = int(os.getenv("GTA5_MAP_PORT", "8793"))
MODEL_STALE = 0.5  # s: the model's lane (modelV2.laneHead) for nav from a message no older
LINK = os.getenv("GTA5_LINK") == "1"  # the link: no brain here, the Windows host's (gta5_host.py)
# the plugin reads the bridge's address from here when its gta5op.ini doesn't set one
BRIDGE_FILE = Path(os.getenv("GTA5_BRIDGE_FILE", "/mnt/c/Users/Public/gta5op-bridge.txt"))
PIN_UI = Path(__file__).parent / "pin_ui.ps1"
# Python's full garbage collections walk every object the bridge holds (over a million, mostly the maps) with the GIL
# held for 0.4-0.8 s, stalling the camera and sensor threads; the bridge makes next to no cyclic garbage, so they run
# only once it has been idle (no game state, or on foot) this long, at most every FULL_GC_EVERY s
FULL_GC_IDLE = 2.0  # s
FULL_GC_EVERY = 60.0  # s
# the simulated car's inputs from the brain before it sets any, and after the link loses the host
DEFAULT_INPUTS = {"user_torque": 0, "left_blinker": False, "right_blinker": False, "left_blindspot": False,
                  "right_blindspot": False}


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


class LocalIO(OpenpilotIO):
  """openpilot in this process (gta5_opio.OpenpilotIO): the simulated car's state and buttons, its messages and params."""
  fix: dict | None = None  # the newest GNSS fix published (take_fix's), counted by fix_seq, at fix_t
  fix_seq = 0
  fix_t = 0.0
  fix_taken = 0
  def __init__(self, simulator_state: SimulatorState, q: Queue, tesla: bool):
    self.simulator_state = simulator_state
    self.q = q
    self.tesla = tesla
    self.sm = messaging.SubMaster(['carControl', 'carParams', 'vehicleParameters', 'modelV2', 'selfdriveState', 'carState',
                                   'controlsState', 'carOutput'])
    self.params = Params()
    self.VM: VehicleModel | None = None
    self.inputs = dict(DEFAULT_INPUTS)
    self.bookmark_pm = None  # made on the first bookmark
    self.route_writer: RouteInputWriter | None = None  # made on the first vector
    self.lanes_writer: RouteInputWriter | None = None
    self.nav_pub = None  # gta5_nav_pub.NavPublisher, made on the first nav message

  @property
  def engaged(self) -> bool:
    return bool(self.simulator_state.is_engaged)

  def update(self):
    self.sm.update(0)
    if self.VM is None and self.sm.seen['carParams']:
      self.VM = VehicleModel(self.sm['carParams'])
    cluster = self.sm['carState'].vCruiseCluster
    self.set_speed_kph = self.sm['controlsState'].deprecated.vCruise if cluster == 0.0 else cluster

  def model(self) -> dict:
    meta = self.sm['modelV2'].meta
    desire = meta.desireState
    turns = {"left": desire[log.Desire.turnLeft], "right": desire[log.Desire.turnRight], "keepLeft": desire[log.Desire.keepLeft],
             "keepRight": desire[log.Desire.keepRight]} if len(desire) > log.Desire.keepRight else {}
    lane_change = meta.laneChangeState
    lane_head = None
    if self.sm.recv_frame['modelV2'] > 0 and time.monotonic() - self.sm.recv_time['modelV2'] <= MODEL_STALE:
      lh = self.sm['modelV2'].laneHead
      lane_head = [int(lh.laneIdx), int(lh.laneCount), round(float(lh.prob), 3)] if 0 <= lh.laneIdx < lh.laneCount else None
    return {"turns": turns, "lane_change": int(getattr(lane_change, "raw", lane_change)), "lane_head": lane_head}

  def controls(self, state: dict) -> dict | None:
    if not self.engaged:
      return None
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
    return {"type": "control", "active": True, "curvature": curvature, "accel": accel}

  def cruise(self, cmd: str):
    self.q.put(control_cmd_gen(cmd))

  def set_inputs(self, **inputs):
    self.inputs.update((k, v) for k, v in inputs.items() if k in INPUTS)

  def apply(self, simulator_state: SimulatorState):
    """The brain's inputs onto the simulated car, once per step: the car thread could read any value set in between."""
    for k, v in self.inputs.items():
      setattr(simulator_state, k, v)

  def reset(self):
    """The brain's inputs back to their defaults, and nav's desire gone (the link lost the host)."""
    self.inputs = dict(DEFAULT_INPUTS)
    self.nav_desire("")

  def nav_desire(self, desire: str):
    if desire:
      self.params.put("NavDesire", desire)
    else:
      self.params.remove("NavDesire")

  def route_input(self, vec):
    if self.route_writer is None:
      self.route_writer = RouteInputWriter(ROUTE_LEN)
    self.route_writer.write(np.asarray(vec, np.float32))

  def lane_slots(self, vec):
    if self.lanes_writer is None:
      self.lanes_writer = RouteInputWriter(PREVIEW_LEN, lane_slots_path())
    self.lanes_writer.write(np.asarray(vec, np.float32))

  def bookmark(self):
    if self.bookmark_pm is None:
      self.bookmark_pm = messaging.PubMaster(["userBookmark"])
    # new messages default to invalid, and selfdrived alerts on an invalid userBookmark until a valid one comes
    self.bookmark_pm.send("userBookmark", messaging.new_message("userBookmark", valid=True))

  def gnss_fix(self, msg, profile):
    """A GNSS fix gta5_gnss published, for navd's map match: the brain's here, or the host's through the status."""
    g = getattr(msg, profile.service)
    self.fix = {"lat": g.latitude, "lon": g.longitude, "bearing": g.bearingDeg, "bearing_accuracy": g.bearingAccuracyDeg,
                "speed": g.speed, "latency": profile.latency}
    self.fix_seq, self.fix_t = self.fix_seq + 1, time.monotonic()

  def take_fix(self) -> dict | None:
    if self.fix is None or self.fix_taken == self.fix_seq:
      return None
    self.fix_taken = self.fix_seq
    return self.fix

  def nav_instruction(self, content: dict | None):
    self._nav_pub().nav_instruction(content)

  def nav_route(self, content: dict | None):
    self._nav_pub().nav_route(content)

  def _nav_pub(self):
    if self.nav_pub is None:
      from openpilot.tools.sim.bridge.gta5.gta5_nav_pub import NavPublisher
      self.nav_pub = NavPublisher()
    return self.nav_pub

  def status(self) -> dict:
    """What the link sends the host (gta5_opio.RemoteIO) of openpilot's state."""
    params = {}
    for key in PARAMS:
      try:
        value = self.params.get(key)
      except Exception:  # params built before the key existed
        value = None
      params[key] = value.decode() if isinstance(value, bytes) else value
    for key in ("IsMetric", "NavigateOnOpenpilot", "TurnDesireRefresh"):
      try:
        params[key] = self.params.get_bool(key)
      except Exception:
        params[key] = False
    gnss = None if self.fix is None else {**self.fix, "seq": self.fix_seq, "age": time.monotonic() - self.fix_t}
    return {"type": "status", "engaged": self.engaged, "set_speed_kph": float(self.set_speed_kph), "model": self.model(),
            "params": params, "gnss": gnss}


class GTA5World(World):
  sets_blinkers = True
  sets_torque = True
  idle_since: float | None = None  # when the bridge last stopped driving (_idle_gc)
  next_full_gc = 0.0
  stall: gta5_stacks.Stall | None = None  # the bridge loop's heartbeat (gta5_stacks.py)
  brain: Brain | None = None
  link = None  # gta5_link.LinkServer, as the link
  recorder: Recorder | None = None

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
    self.last_status = 0.0
    self.curvature = 0.0  # what the steering is set for
    self.pinner: subprocess.Popen | None = None
    self.log = open(LOG, "a", buffering=1) if LOG else None
    self.io = LocalIO(simulator_state, q, is_tesla())
    self.gnss = gta5_gnss.from_env()
    self.publishes_gps = self.gnss is not None
    if LINK:
      from openpilot.tools.sim.bridge.gta5.gta5_link import LinkServer
      self.link = LinkServer(self.io, control=control_path() or CONTROL)
    else:
      self.brain = Brain(self.io, self._send, self.log, match=NAV_MATCH and self.gnss is not None)
      if RECORD:
        self.recorder = self.brain.recorder = Recorder(RECORD, self.brain)

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
    stacks = gta5_stacks.watch()
    self.stall = gta5_stacks.Stall(stacks) if stacks is not None else None
    if MAP and self.brain is not None:
      # made once rx has forked: a socket it inherited would outlive this process
      self.brain.map_view = MapView(os.path.join(MAP, "roads.json"), MAP_PORT, stacks=stacks)
      print(f"gta5: map view on http://localhost:{MAP_PORT}/")
    elif self.link is not None:
      self.link.view = MapView(os.path.join(MAP or "", "roads.json"), MAP_PORT, stacks=stacks)
      print(f"gta5: destinations for the host on http://localhost:{MAP_PORT}/destination")
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

  def apply_controls(self, steer_angle, throttle_out, brake_out):
    """Sends the controls of whoever drives (the brain's drivers; openpilot's are carControl's curvature and acceleration),
    rather than the bridge's steer_angle/throttle/brake: those are shaped for the simulated Honda, not for the game car.
    As the link, openpilot's, which the host's relay passes on only while openpilot drives."""
    self.io.update()
    with self.lock:
      state = self.state
    if state is None or not state.get("inVehicle"):
      return
    if self.brain is not None:
      self.brain.hud()
      msg = self.brain.controls(state)
    else:
      msg = self.io.controls(state) or INACTIVE
    self._send(msg)

    now = time.monotonic()
    if DEBUG and msg.get("active") and now - self.last_status > 1.0:
      self.last_status = now
      out = state.get("out", {})
      print(f"gta5: v={state['vEgo']:5.2f} curvature cmd={msg['curvature']:+.4f} meas={self.curvature:+.4f} " +
            f"accel cmd={msg['accel']:+.2f} meas={state['aMeas']:+.2f} | steer={out.get('steer', 0):+.2f} " +
            f"thr={out.get('throttle', 0):.2f} brk={out.get('brake', 0):.2f} latI={out.get('latI', 0):+.3f} lonI={out.get('lonI', 0):+.2f}",
            flush=True)

  def read_state(self):
    pass

  def read_sensors(self, simulator_state: SimulatorState):
    if self.stall is not None:
      self.stall.beat(time.monotonic())
    if self.link is not None:
      self.link.step()  # the host's calls, and openpilot's state back to it
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
    vm = self.io.VM
    if vm is not None:
      # The angle for the curvature the car follows, through the car's own fixed vehicle model (carParams', with no
      # offset), as a real sensor measures a physical angle: paramsd then learns the car's values, as on a real Model 3.
      # An angle through paramsd's learned model would confirm whatever it learned, letting its estimates wander.
      steer = vm.get_steer_from_curvature(self.curvature, max(abs(v), 1.0), bank)
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
      if fix is not None:
        self.io.gnss_fix(fix, self.gnss.profile)

    user = state.get("user") or {}
    simulator_state.user_gas = 1.0 if user.get("gas") else 0.0
    simulator_state.user_brake = 1.0 if user.get("brake") else 0.0
    simulator_state.speed_limit = speed_limit(state.get("street", ""))
    if self.brain is not None:
      self.brain.step(state)
    self.io.apply(simulator_state)
    simulator_state.valid = True

  def _idle_gc(self):
    """The full garbage collection hold_full_collections leaves out, while nothing is driving."""
    now = time.monotonic()
    if self.idle_since is None:
      self.idle_since = now
    elif now - self.idle_since > FULL_GC_IDLE and now >= self.next_full_gc:
      self.next_full_gc = now + FULL_GC_EVERY
      gc.collect()

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
    if self.link is not None and not wide:
      self.link.camera(state.get("hostSeq") if slot is not None and state else None)
    return frame

  def tick(self):
    pass

  def reset(self):
    self._send({"type": "control", "active": False})

  def close(self, reason: str):
    if self.brain is not None:
      self.brain.close()
      if self.brain.map_view is not None:
        self.brain.map_view.close()
    if self.link is not None:
      self.link.close()
    if self.stall is not None:
      self.stall.stop()
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
    if self.recorder is not None:
      self.recorder.close()
    print(f"gta5: closing ({reason})")
