"""Records driving in comma's comma1M segment layout, for training the driving model on the game (GTA5_RECORD=<folder>).

Each minute of driving becomes <folder>/data/<hex name>/ with:
- fcamera.hevc, ecamera.hevc: the road and wide frames openpilot's camerad got, 1928x1208 at 20 Hz, I and P frames only;
- frame_info.safetensors: device_type tici, and per camera the HEVC frame index, frame times, size and codec, as
  openpilot.distill's localizer writes it;
- localizer.safetensors: the game's own pose at each road frame (frame_t, the 43-wide frame_states, rpy and
  wide_from_device_euler), with the map placed at ANCHOR on the earth;
- gta5.npz: everything raw, per road frame: the game state, the driver's, nav's and the game AI's (expert mode) inputs,
  and modeld's outputs for that
  frame with the desire input it was given (rebuilt as modeld's DesireHelper builds it), plus its raw output vector
  when modeld runs with SEND_RAW_PRED=1. localizer and frame_info are made from it, so `finalize` can remake them;
- routes.json: nav's routes and the lane line it aims for, when on a map route;
- gta5.json: the segment's settings and counts, the camera's mount (from the plugin's state when it reports one, else
  GTA5_RECORD_MOUNT) and the scene at its start: the car, weather, time and traffic density.

A segment ends early at a gap in the frames, a teleport, leaving the car or the camera's mount changing. Encoding is
libx265 on the CPU, niced: about 1.3 cores per camera."""
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

RECORD = os.getenv("GTA5_RECORD")
ENCODER = os.getenv("GTA5_RECORD_ENCODER", "libx265")  # or hevc_nvenc
# the system's: openpilot's venv puts an ffmpeg without libx265 first on PATH
FFMPEG = os.getenv("GTA5_RECORD_FFMPEG", "/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "ffmpeg")
CRF = int(os.getenv("GTA5_RECORD_CRF", "21"))
# the camera's mount on the car, m forward of and above the car's origin, which the game's pose is for, when the plugin's
# state has none
MOUNT = tuple(float(v) for v in os.getenv("GTA5_RECORD_MOUNT", "1.0,0.6").split(","))
SCENE_KEYS = ("vehicle", "world", "density")  # the plugin's state kept in gta5.json as the segment's scene
ANCHOR = (34.0522, -118.2437, 0.0)  # Los Angeles: the game's x east, y north and z up, from here
SEGMENT_FRAMES = 1200  # 60 s at 20 Hz, as comma's segments
GAP = 0.5  # s between game frames that ends a segment
QUEUE_FRAMES = 40  # per camera; a full queue (the encoder behind) ends the segment
MODEL_WAIT = 2.0  # s a finished segment waits for modeld's outputs on its last frames
KEYINT = 20
DEVICE_TYPE_TICI = 4  # cereal InitData.DeviceType
DESIRE_LEN = 8
DESIRE_PRED_LEN = 32
FRAME_INFO_METADATA = {
  "encoded_numpy_dtypes": json.dumps({f"{camera}/{field}": {"dtype": dtype, "shape": [1]} for camera in ("fcamera", "ecamera")
                                      for field, dtype in (("codec_name", "<U4"), ("global_prefix", "|S88"))}, separators=(",", ":")),
  "schema_version": "1",
}

# per road frame, in gta5.npz's "frame" array
FRAME_COLUMNS = ("frame_id", "t", "x", "y", "z", "heading", "pitch", "roll", "grade", "bank", "v_ego", "a_meas", "yaw_rate",
                 "rot_vel_x", "rot_vel_y", "rot_vel_z", "steer_curvature", "cam_height", "wheel_base", "resets", "collisions",
                 "indicator", "user_steer", "user_gas", "user_brake", "engaged", "blinker_gap", "cruise_cap", "lane", "lanes",
                 "route", "route_at", "route_off", "route_right", "expert", "expert_label")
# per road frame, NaN where modeld's output for it never came
MODEL_COLUMNS = ("time", "frame_id_extra", "execution_time", "lane_change_state", "lane_change_direction", "action_curvature",
                 "action_accel", "left_blinker", "right_blinker", "v_ego", "lat_active", "nav")
INDICATORS = {None: 0, "left": 1, "right": 2}


def state_mount(state: dict) -> tuple | None:
  """The camera's mount the plugin reports: m forward, up and right of the car's origin, and deg pitch up and yaw left."""
  m = state.get("mount")
  if not isinstance(m, dict) or "y" not in m or "z" not in m:
    return None
  return tuple(round(float(m.get(k, 0.0)), 4) for k in ("y", "z", "x", "pitch", "yaw"))


def full_mount(info: dict) -> tuple:
  """A segment's mount as forward, up, right, pitch, yaw: gta5.json's mount, with mount_detail's right and angles."""
  fwd, up = tuple(info.get("mount", MOUNT))[:2]
  d = info.get("mount_detail") or {}
  return (fwd, up, d.get("x", 0.0), d.get("pitch", 0.0), d.get("yaw", 0.0))


def save_safetensors(path: Path, tensors: dict[str, np.ndarray], metadata: dict[str, str] | None = None) -> None:
  """safetensors.numpy.save_file, which the bridge's environment doesn't have."""
  dtypes = {np.dtype(k): v for k, v in (("float64", "F64"), ("float32", "F32"), ("float16", "F16"), ("int64", "I64"), ("int32", "I32"),
                                         ("int16", "I16"), ("int8", "I8"), ("uint32", "U32"), ("uint8", "U8"), ("bool", "BOOL"))}
  header: dict = {"__metadata__": metadata} if metadata else {}
  blobs, offset = [], 0
  for name in sorted(tensors):
    a = np.ascontiguousarray(tensors[name])
    a = a.astype(a.dtype.newbyteorder("<"), copy=False)
    blobs.append(a.tobytes())
    header[name] = {"dtype": dtypes[a.dtype], "shape": list(a.shape), "data_offsets": [offset, offset + len(blobs[-1])]}
    offset += len(blobs[-1])
  head = json.dumps(header, separators=(",", ":")).encode()
  head += b" " * (-len(head) % 8)
  with open(path, "wb") as f:
    f.write(len(head).to_bytes(8, "little"))
    f.write(head)
    for blob in blobs:
      f.write(blob)


def encoder_args(path: Path, width: int, height: int, stride: int, y_height: int) -> list[str]:
  """ffmpeg reading padded NV12 frames from stdin, writing raw HEVC without B-frames, as comma's encoder does: the
  training loaders count decoded frames against the frame index, in decode order."""
  codec = (["-c:v", "hevc_nvenc", "-preset", "p1", "-tune", "ll", "-rc", "constqp", "-qp", str(CRF), "-bf", "0", "-g", str(KEYINT)]
           if ENCODER == "hevc_nvenc" else
           ["-c:v", "libx265", "-preset", "ultrafast", "-crf", str(CRF), "-x265-params",
            f"bframes=0:keyint={KEYINT}:min-keyint={KEYINT}:scenecut=0:pools=4:info=0:log-level=error"])
  return ["nice", "-n", "10", FFMPEG,"-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "nv12",
          "-s", f"{stride}x{y_height}", "-r", "20", "-i", "pipe:0", "-vf", f"crop={width}:{height}:0:0", "-fps_mode", "passthrough",
          *codec, "-f", "hevc", str(path)]


class Video:
  """One camera's encoder, fed from a queue by its own thread, so a slow encoder never holds up the camera thread."""
  def __init__(self, path: Path, width: int, height: int, stride: int, y_height: int, uv_height: int):
    self.path = path
    self.frame_bytes = stride * (y_height + uv_height)
    self.proc = subprocess.Popen(encoder_args(path, width, height, stride, y_height), stdin=subprocess.PIPE,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
      import fcntl
      fcntl.fcntl(self.proc.stdin.fileno(), 1031, 4 << 20)  # F_SETPIPE_SZ: fewer, larger writes
    except OSError:
      pass
    self.queue: queue.Queue = queue.Queue(QUEUE_FRAMES)
    self.frames = 0
    self.error = ""
    self.thread = threading.Thread(target=self._write, daemon=True)
    self.thread.start()

  def add(self, frame: bytes) -> bool:
    if not self.thread.is_alive():
      return False
    try:
      self.queue.put_nowait(frame)
    except queue.Full:
      return False
    self.frames += 1
    return True

  def _write(self):
    stdin = self.proc.stdin
    assert stdin is not None
    while (frame := self.queue.get()) is not None:
      try:
        stdin.write(memoryview(frame)[:self.frame_bytes])
      except (BrokenPipeError, OSError, ValueError) as e:
        self.error = str(e)
        break
    try:
      stdin.close()
    except (OSError, ValueError):
      pass
    # communicate() would flush the stdin closed above
    err = self.proc.stderr.read() if self.proc.stderr else b""
    self.proc.wait()
    if self.proc.returncode:
      self.error = f"ffmpeg exited {self.proc.returncode}: {err.decode(errors='replace').strip()[-300:]}"
    if self.error:
      print(f"gta5: recording: {self.path} failed: {self.error}", flush=True)

  def finish(self):
    """Never blocks the camera thread: with the queue full, its newest frames go, which keeps the video's frames the
    segment's first ones."""
    with self.queue.mutex:
      if len(self.queue.queue) >= QUEUE_FRAMES:
        self.queue.queue.pop()
        self.frames -= 1
      self.queue.queue.append(None)
      self.queue.not_empty.notify()

  def wait(self):
    self.thread.join()


class Segment:
  def __init__(self, path: Path, info: dict, size: tuple):
    path.mkdir(parents=True, exist_ok=True)
    self.path = path
    self.info = info
    self.road = Video(path / "fcamera.hevc", *size)
    self.wide = Video(path / "ecamera.hevc", *size)
    self.rows: list[list[float]] = []
    self.routes: list[dict] = []
    self.lane_lines: list[dict] = []
    self.expert_labels: list[str] = []
    self.mount = None  # the plugin's report of it, which a change of ends the segment
    self.broken = ""
    self.done_at = 0.0  # when it stopped taking frames


class Recorder:
  def __init__(self, root: str, world=None, synthetic: bool = False):
    from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
    from openpilot.tools.sim.lib.camerad import W, H
    stride, y_height, uv_height, _ = get_nv12_info(W, H)
    self.size = (W, H, stride, y_height, uv_height)
    self.data = Path(root) / "data"
    self.world = world
    self.synthetic = synthetic
    self.started = datetime.now()
    self.route_name = self.started.strftime("%Y-%m-%d--%H-%M-%S")
    self.segment_count = 0
    self.segment: Segment | None = None
    self.wide_for: Segment | None = None  # the segment the next wide frame belongs to
    self.done: list[Segment] = []
    self.road_id = self.wide_id = 0  # camerad's frame ids: it counts every frame the camera thread sends
    self.last_t = 0.0
    self.last_resets = None
    self.last_route = None
    self.last_lane_line = None
    self.lock = threading.Lock()
    self.model: dict[int, dict] = {}  # road frame id -> what modeld made of it
    self.stopped = False
    self.failed = False
    if world is not None:
      threading.Thread(target=self._model_loop, daemon=True).start()
    print(f"gta5: recording to {self.data}", flush=True)

  # *** frames, from the camera thread ***

  def add(self, wide: bool, frame: bytes | None, state: dict | None, t: float) -> None:
    """A frame as it went to camerad (None for a blank one), with the game state that came with it and when it came."""
    if wide:
      self.wide_id += 1
      seg, self.wide_for = self.wide_for, None
      if seg is not None and (frame is None or not seg.wide.add(frame)):
        seg.broken = "no wide frame" if frame is None else "wide encoder behind"
      return
    frame_id = self.road_id
    self.road_id += 1
    self._wrap_up()
    if self.failed:
      self._end()
      return
    ok = frame is not None and state is not None and state.get("inVehicle") and not state.get("paused") and "pos" in state
    ok = ok and not (state.get("mount") or {}).get("pending")  # the dashcam mount is found a moment after a new car
    seg = self.segment
    if seg is not None and (not ok or seg.broken or len(seg.rows) >= SEGMENT_FRAMES or t - self.last_t > GAP or
                            state.get("resets") != self.last_resets or state_mount(state) != seg.mount):
      self._end()
      seg = None
    if not ok or (seg is not None and t == self.last_t):  # camerad sometimes sends a game frame twice; the loaders need new times
      return
    if seg is None:
      seg = self.segment = self._start(frame_id, state)
    if not seg.road.add(frame):
      seg.broken = "road encoder behind"
      if not seg.road.thread.is_alive():  # it failed: don't start another each frame
        self.failed = True
        print("gta5: recording stopped: the encoder failed", flush=True)
      return
    self.wide_for = seg
    self.last_t, self.last_resets = t, state.get("resets")
    seg.rows.append(self._row(frame_id, t, state, len(seg.rows)))

  def _start(self, frame_id: int, state: dict) -> Segment:
    name = f"{int(self.started.timestamp()):08x}{self.segment_count:04x}"
    mount = state_mount(state)
    info = {"route": self.route_name, "segment": self.segment_count, "first_frame_id": frame_id, "synthetic": self.synthetic,
            "anchor": ANCHOR, "mount": mount[:2] if mount else MOUNT, "encoder": ENCODER, "crf": CRF,
            "started": datetime.now().isoformat(timespec="seconds")}
    if mount:
      info["mount_detail"] = state["mount"]
    scene = {k: state[k] for k in SCENE_KEYS if k in state}
    if scene:
      info["scene"] = scene
    self.segment_count += 1
    self.last_route = self.last_lane_line = None
    seg = Segment(self.data / name, info, self.size)
    seg.mount = mount
    return seg

  def _row(self, frame_id: int, t: float, state: dict, index: int) -> list[float]:
    w = self.world
    pos, rot_vel, user = state["pos"], state.get("rotVel") or (0, 0, 0), state.get("user") or {}
    lane = state.get("lane") or (-1, 0)
    route, route_idx = getattr(w, "route", None), -1
    if route is not None:
      if route is not self.last_route:
        self.segment.routes.append({"frame": index, "points": np.round(route.points, 2).tolist(), "maneuvers": route.maneuvers})
        self.last_route = route
      route_idx = len(self.segment.routes) - 1
    # the game's AI driving (gta5_expert.py), and the desire its indicators stand for
    expert = getattr(w, "expert", None)
    expert_on = bool(getattr(expert, "active", False))
    label = str(getattr(expert, "label", "")) if expert_on else ""
    if label not in self.segment.expert_labels:
      self.segment.expert_labels.append(label)
    lane_line = w.lane_line[2] if w is not None else None
    if lane_line and lane_line is not self.last_lane_line:
      self.segment.lane_lines.append({"frame": index, "line": lane_line})
      self.last_lane_line = lane_line
    return [frame_id, t, *pos, state["heading"], state.get("pitch", 0), state.get("roll", 0), state.get("grade", 0), state.get("bank", 0),
            state["vEgo"], state.get("aMeas", 0), state.get("yawRate", 0), *rot_vel, state.get("steerCurvature", 0),
            state.get("camHeight", 0), state.get("wheelBase", 0), state.get("resets", 0), state.get("collisions", 0),
            INDICATORS.get(state.get("indicator"), 0), user.get("steer", 0), user.get("gas", False), user.get("brake", False),
            w.simulator_state.is_engaged if w is not None else 0, w.nav.blinker_gap if w is not None else 0,
            getattr(w, "cap", 0), lane[0], lane[1], route_idx,
            *((route.at, route.off, route.right) if route is not None else (np.nan,) * 3),
            expert_on, self.segment.expert_labels.index(label)]

  def _end(self):
    seg, self.segment, self.wide_for = self.segment, None, None
    if seg is None:
      return
    seg.road.finish()
    seg.wide.finish()
    seg.done_at = time.monotonic()
    self.done.append(seg)

  def _wrap_up(self, wait: bool = False):
    """Saves finished segments once modeld's outputs for their last frames are in."""
    now = time.monotonic()
    with self.lock:  # the camera thread and close() both get here
      ready = [s for s in self.done if wait or now - s.done_at > MODEL_WAIT]
      self.done = [s for s in self.done if s not in ready]
    for seg in ready:
      t = threading.Thread(target=self._save, args=(seg,), daemon=not wait)
      t.start()
      if wait:
        t.join()

  def _save(self, seg: Segment):
    seg.road.wait()
    seg.wide.wait()
    rows = np.array(seg.rows, dtype=np.float64).reshape(-1, len(FRAME_COLUMNS))
    out = {"frame": rows, "frame_columns": np.array(FRAME_COLUMNS), "expert_label_values": np.array(seg.expert_labels or [""])}
    with self.lock:
      model = [self.model.pop(int(i), None) for i in rows[:, 0]]
      for i in [k for k in self.model if rows.size and k < rows[-1, 0]]:
        del self.model[i]
    if any(m is not None for m in model):
      out.update(stack_model(model))
    seg.info.update({"frames": len(rows), "road_frames": seg.road.frames, "wide_frames": seg.wide.frames, "ended": seg.broken or None,
                     "errors": [e for e in (seg.road.error, seg.wide.error) if e] or None,
                     "model_frames": sum(m is not None for m in model)})
    np.savez(seg.path / "gta5.npz", **out)
    print(f"gta5: recorded {seg.path.name}: {len(rows)} frames" + (f", ended: {seg.broken}" if seg.broken else ""), flush=True)
    (seg.path / "routes.json").write_text(json.dumps({"routes": seg.routes, "lane_lines": seg.lane_lines}, default=json_default))
    (seg.path / "gta5.json").write_text(json.dumps(seg.info, indent=1))
    # the frame index means reading the whole video back: done apart from the bridge's process
    with open(seg.path / "finalize.log", "w") as log:
      subprocess.Popen(["nice", "-n", "10", sys.executable, "-m", "openpilot.tools.sim.bridge.gta5.gta5_record", "finalize", str(seg.path)],
                       stdin=subprocess.DEVNULL, stdout=log, stderr=log)

  def close(self):
    self.stopped = True
    self._end()
    self._wrap_up(wait=True)

  # *** modeld's outputs ***

  def _model_loop(self):
    from openpilot.cereal import log, messaging
    from openpilot.common.params import Params
    from openpilot.selfdrive.controls.lib.desire_helper import NAV_KEEP, lane_turn_desire, nav_split
    sock = messaging.sub_sock("modelV2", conflate=False, timeout=100)
    sm = messaging.SubMaster(["carState", "carControl", "extrinsicsCalibration"])
    params = Params()
    desire = prev_desire = None
    LCS, LCD, D = log.LaneChangeState, log.LaneChangeDirection, log.Desire
    while not self.stopped:
      msgs = messaging.drain_sock(sock, wait_for_one=True)
      if not msgs:
        continue
      sm.update(0)
      cs, cal = sm["carState"], sm["extrinsicsCalibration"]
      for msg in msgs:
        m = msg.modelV2
        meta = m.meta
        nav_raw = params.get("NavDesire") or ""
        # the desire modeld gave this frame was built after its previous frame, from that frame's lane change state
        rec = {"time": msg.logMonoTime * 1e-9, "frame_id_extra": m.frameIdExtra, "execution_time": m.modelExecutionTime,
               "lane_change_state": meta.laneChangeState.raw, "lane_change_direction": meta.laneChangeDirection.raw,
               "action_curvature": m.action.desiredCurvature, "action_accel": m.action.desiredAcceleration,
               "left_blinker": cs.leftBlinker, "right_blinker": cs.rightBlinker, "v_ego": cs.vEgo,
               "lat_active": sm["carControl"].latActive, "nav": nav_raw,
               "desire_state": list(meta.desireState), "desire_pred": list(meta.desirePrediction),
               "calib": list(cal.rpyCalib), "wide_from_device_euler": list(cal.wideFromDeviceEuler)}
        if desire is not None:
          rec["desire_input"] = desire
          rec["desire_pulse"] = np.where(desire - prev_desire > .99, desire, 0) if prev_desire is not None else None
        if len(m.rawPredictions):
          rec["raw"] = np.frombuffer(m.rawPredictions, dtype=np.float32).astype(np.float16)
        prev_desire = desire
        # DesireHelper.update after this frame, as modeld does, for the next frame's input
        nav, stacked = nav_split(nav_raw)
        d = lane_turn_desire(cs, nav)
        if d == D.none and meta.laneChangeState.raw == LCS.laneChangeStarting:
          d = {LCD.left: D.laneChangeLeft, LCD.right: D.laneChangeRight}.get(meta.laneChangeDirection.raw, D.none)
        if d == D.none:
          d = NAV_KEEP.get(nav, stacked)
        stacked = stacked if stacked != d else D.none
        desire = np.zeros(DESIRE_LEN, dtype=np.float32)
        if 0 <= d < DESIRE_LEN:
          desire[d] = 1
        if 0 < stacked < DESIRE_LEN:
          desire[stacked] = 1
        desire[0] = 0
        with self.lock:
          self.model[m.frameId] = rec
          if len(self.model) > 3 * SEGMENT_FRAMES:  # not recording, or nothing saved: keep what a segment could still want
            for k in [k for k in self.model if k < self.road_id - 2 * SEGMENT_FRAMES]:
              del self.model[k]


def stack_model(model: list[dict | None]) -> dict[str, np.ndarray]:
  """modeld's per-frame records as arrays, NaN where there's none."""
  n = len(model)
  out = {"model": np.full((n, len(MODEL_COLUMNS)), np.nan), "model_columns": np.array(MODEL_COLUMNS)}
  nav = sorted({m["nav"] for m in model if m is not None})
  out["model_nav_values"] = np.array(nav or [""])
  vectors = {"desire_state": DESIRE_LEN, "desire_pred": DESIRE_PRED_LEN, "desire_input": DESIRE_LEN, "desire_pulse": DESIRE_LEN,
             "calib": 3, "wide_from_device_euler": 3}
  for k, size in vectors.items():
    out[f"model_{k}"] = np.full((n, size), np.nan, dtype=np.float32)
  raw_len = next((len(m["raw"]) for m in model if m is not None and "raw" in m), 0)
  if raw_len:
    out["model_raw"] = np.full((n, raw_len), np.nan, dtype=np.float16)
  for i, m in enumerate(model):
    if m is None:
      continue
    out["model"][i] = [nav.index(m[c]) if c == "nav" else float(m[c]) for c in MODEL_COLUMNS]
    for k, size in vectors.items():
      v = m.get(k)
      if v is not None and len(v):
        out[f"model_{k}"][i, :min(size, len(v))] = v[:size]
    if raw_len and "raw" in m and len(m["raw"]) == raw_len:
      out["model_raw"][i] = m["raw"]
  return out


def json_default(o):
  if isinstance(o, np.ndarray):
    return o.tolist()
  if isinstance(o, np.generic):
    return o.item()
  raise TypeError(f"{type(o).__name__} is not JSON serializable")


# *** segment files from gta5.npz ***

def columns(npz, name: str = "frame") -> dict[str, np.ndarray]:
  return {c: npz[name][:, i] for i, c in enumerate(npz[f"{name}_columns"])}


def device_poses(f: dict[str, np.ndarray], mount: tuple = MOUNT) -> tuple[np.ndarray, np.ndarray]:
  """The camera's position (NED, m from the map's origin) and ned_from_device rotations, from the game's car poses and
  the mount: m forward, up, and optionally right, deg pitch up and yaw left. Device axes: x forward, y right, z down.
  The game's heading is counterclockwise from north, its grade nose up and its bank right side down."""
  from openpilot.common.transformations.orientation import rot_from_euler
  fwd, up, right, pitch, yaw = (tuple(mount) + (0.0, 0.0, 0.0))[:5]
  grade, bank = f["grade"], f["bank"]
  roll = np.arcsin(np.clip(np.sin(bank) / np.cos(grade), -1, 1))  # bank is the right axis' drop, roll the angle about x
  ned_from_car = rot_from_euler(np.stack([roll, grade, -np.radians(f["heading"])], axis=1))
  car_ned = np.stack([f["y"], f["x"], -f["z"]], axis=1)
  ned_from_device = ned_from_car @ rot_from_euler([0.0, np.radians(pitch), -np.radians(yaw)])
  return car_ned + ned_from_car @ np.array([fwd, right, -up]), ned_from_device


def localizer(npz, info: dict) -> dict[str, np.ndarray]:
  """localizer.safetensors as openpilot.distill's localizer.localize writes it, from the game's ground truth."""
  from openpilot.common.transformations.coordinates import LocalCoord
  from openpilot.common.transformations.orientation import quat_from_rot
  f = columns(npz)
  local = LocalCoord.from_geodetic(list(info.get("anchor", ANCHOR)))
  ecef_from_ned = local.ned2ecef_matrix
  pos_ned, ned_from_device = device_poses(f, full_mount(info))
  ecef_from_device = ecef_from_ned @ ned_from_device
  # the game's rotation velocity is in the car's axes (x right, y forward, z up)
  omega = np.stack([f["rot_vel_y"], f["rot_vel_x"], -f["rot_vel_z"]], axis=1)
  v = f["v_ego"]
  velocity_device = np.stack([v, np.zeros_like(v), np.zeros_like(v)], axis=1)  # the game gives no sideslip
  accel_device = np.stack([f["a_meas"], v * omega[:, 2], np.zeros_like(v)], axis=1)

  states = np.zeros((len(v), 43), dtype=np.float64)
  states[:, 0:3] = pos_ned @ ecef_from_ned.T + local.init_ecef
  states[:, 3:7] = quat_from_rot(ecef_from_device)
  states[:, 7:10] = np.einsum("nij,nj->ni", ecef_from_device, velocity_device)
  states[:, 10:13] = omega
  states[:, 19:22] = accel_device
  states[:, [18, 22, 29]] = 1.0
  rpy, wide = np.zeros(3), np.zeros(3)
  if "model_calib" in npz:
    seen = ~np.isnan(npz["model_calib"][:, 1]) & (np.abs(npz["model_calib"]).sum(axis=1) > 0)
    if seen.any():  # what modeld warped its frames by, which the model's plan is relative to
      rpy = np.median(npz["model_calib"][seen], axis=0).astype(np.float64)
      wide = np.median(npz["model_wide_from_device_euler"][seen], axis=0).astype(np.float64)
  rpy[0] = wide[0] = 0.0
  states[:, 33:36] = wide
  states[:, 36:43] = states[:, :7]
  return {"frame_t": f["t"].copy(), "frame_states": states, "rpy": rpy, "wide_from_device_euler": wide}


def frame_info(seg: Path, frame_t: np.ndarray, width: int, height: int) -> tuple[dict[str, np.ndarray], int]:
  """frame_info.safetensors as openpilot.distill's localizer.logs writes it, and the frames both videos have."""
  from openpilot.tools.lib.vidindex import hevc_index
  out: dict[str, np.ndarray] = {"device_type": np.asarray(DEVICE_TYPE_TICI, dtype=np.int64)}
  indexes = {}
  for camera in ("fcamera", "ecamera"):
    frame_types, size, prefix = hevc_index(str(seg / f"{camera}.hevc"))
    indexes[camera] = (np.array(frame_types + [(0xFFFFFFFF, size)], dtype=np.uint32), prefix)
  n = min(len(frame_t), *(len(index) - 1 for index, _ in indexes.values()))
  for camera, (index, prefix) in indexes.items():
    if len(index) - 1 > n:
      index = np.concatenate([index[:n], [[0xFFFFFFFF, index[n, 1]]]]).astype(np.uint32)
    out.update({
      f"{camera}/codec_name": np.asarray(["hevc"], dtype="<U4").view(np.uint8).reshape(1, -1).copy(),
      f"{camera}/frame_count": np.asarray([n], dtype=np.int64),
      f"{camera}/global_prefix": np.frombuffer(prefix, dtype=np.uint8)[None].copy(),
      f"{camera}/height": np.asarray([height], dtype=np.int64),
      f"{camera}/index": index,
      f"{camera}/t": frame_t[:n].copy(),
      f"{camera}/width": np.asarray([width], dtype=np.int64),
    })
  return out, n


def finalize(seg: Path) -> None:
  from openpilot.tools.sim.lib.camerad import W, H
  info = json.loads((seg / "gta5.json").read_text())
  npz = dict(np.load(seg / "gta5.npz"))
  loc = localizer(npz, info)
  fi, n = frame_info(seg, loc["frame_t"], W, H)
  if n < len(loc["frame_t"]):
    loc = {k: v[:n] if k in ("frame_t", "frame_states") else v for k, v in loc.items()}
  save_safetensors(seg / "localizer.safetensors", loc, {"schema_version": "1"})
  save_safetensors(seg / "frame_info.safetensors", fi, FRAME_INFO_METADATA)
  info["video_frames"] = n
  (seg / "gta5.json").write_text(json.dumps(info, indent=1))
  print(f"{seg}: {n} frames")


# *** offline ***

def replay(log_path: str, root: str, start: float, seconds: float, frames_path: str | None) -> None:
  """A synthetic segment from a GTA5_LOG: its real game states at their real times, with frames from a raw padded NV12
  file (looped), or grey. For testing the format and loaders only: the frames don't show the drive."""
  rec = Recorder(root, synthetic=True)
  size = rec.size[2] * (rec.size[3] + rec.size[4])
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
  nv12_size = get_nv12_info(rec.size[0], rec.size[1])[3]
  source = open(frames_path, "rb") if frames_path else None
  grey = bytes([128]) * nv12_size

  def next_frame() -> bytes:
    if source is None:
      return grey
    data = source.read(size)
    if len(data) < size:
      source.seek(0)
      data = source.read(size)
    return data + bytes(nv12_size - size)

  with open(log_path) as f:
    for line in f:
      if '"state"' not in line:
        continue
      d = json.loads(line)
      t = d["mono"]
      if t < start:
        continue
      if t > start + seconds:
        break
      seg = rec.segment
      while seg is not None and max(seg.road.queue.qsize(), seg.wide.queue.qsize()) > QUEUE_FRAMES // 2:
        time.sleep(0.005)  # offline, wait for the encoders rather than end the segment
      frame = next_frame()
      rec.add(False, frame, d["state"], t)
      rec.add(True, frame, d["state"], t)
  rec.close()
  for seg in sorted(rec.data.glob(f"{int(rec.started.timestamp()):08x}*")):
    deadline = time.monotonic() + 120
    while not (seg / "frame_info.safetensors").exists() and time.monotonic() < deadline:
      time.sleep(0.2)
    print(seg, (seg / "gta5.json").read_text())


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  sub = parser.add_subparsers(dest="cmd", required=True)
  p = sub.add_parser("finalize", help="(re)make a segment's frame_info and localizer from its gta5.npz and videos")
  p.add_argument("segments", nargs="+", type=Path)
  p = sub.add_parser("replay", help="a synthetic segment from a GTA5_LOG, for testing")
  p.add_argument("log")
  p.add_argument("root")
  p.add_argument("--start", type=float, required=True, help="the log's mono time to start at")
  p.add_argument("--seconds", type=float, default=60.0)
  p.add_argument("--frames", help="raw NV12 frames padded as camerad's buffers, looped")
  args = parser.parse_args()
  if args.cmd == "finalize":
    for s in args.segments:
      finalize(s)
  else:
    replay(args.log, args.root, args.start, args.seconds, args.frames)
