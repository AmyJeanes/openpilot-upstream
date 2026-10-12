"""The recorder's segments as frames come (gta5_record.py), with stand-in encoders; and the same recording made in the
bridge and in the Windows host, from the same synthetic drive, with the real encoder where there's one."""
import json
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.tools.sim.bridge.gta5 import gta5_record, gta5_relay
from openpilot.tools.sim.bridge.gta5.gta5_record import FrameWorker, Recorder, model_record_from_json, model_record_to_json, stack_model


class Encoder:
  def __init__(self):
    self.ok = True
    self.thread = SimpleNamespace(is_alive=lambda: self.ok)
    self.frames: list[bytes] = []

  def add(self, frame) -> bool:
    self.frames.append(bytes(frame))
    return self.ok

  def finish(self):
    pass


class Segment:
  def __init__(self, path, info, size, video=None):
    self.road, self.wide = Encoder(), Encoder()
    self.info = info
    self.rows: list = []
    self.seqs: list = []
    self.md, self.mount, self.broken, self.done_at = None, None, "", 0.0
    self.routes, self.lane_lines, self.expert_labels, self.ww_rows, self.src_rows, self.mapx_rows = [], [], [], [], [], []


def test_recording_only_while_a_segment_takes_frames(tmp_path, monkeypatch):
  monkeypatch.setattr(gta5_record, "Segment", Segment)
  monkeypatch.setattr(gta5_record, "MODEL_WAIT", 1e9)  # nothing saved
  rec = Recorder(str(tmp_path))
  rec._row = lambda *args: []
  car = {"inVehicle": True, "pos": [0.0, 0.0, 0.0], "resets": 0}
  t = iter(range(100))

  def frame(**state):
    rec.add(False, b"frame", {**car, **state}, next(t) * 0.05)
    return rec.recording

  assert not rec.recording  # idle
  assert frame()
  assert not frame(inVehicle=False) and rec.segment is None  # on foot
  assert frame()
  assert not frame(paused=True)
  n = rec.segment_count
  car["resets"] = 1
  assert frame() and rec.segment_count == n + 1  # a teleport: the next segment starts at once
  rec.segment.road.ok = False
  assert not frame() and rec.failed  # the encoder failed
  assert not frame() and rec.segment is None


def model_record(i: int) -> dict:
  """A record as gta5_record.ModelTap makes them."""
  desire = np.eye(8, dtype=np.float32)[i % 8]
  rec = {"time": 100.0 + i * 0.05, "frame_id_extra": i, "execution_time": 0.012, "lane_change_state": i % 3, "lane_change_direction": 0,
         "action_curvature": 0.001 * i, "action_accel": -0.25, "left_blinker": False, "right_blinker": i % 2 == 0, "v_ego": 10.0 + i,
         "lat_active": True, "nav": "turnLeft" if i > 10 else "", "desire_state": [0.125] * 8, "desire_pred": [i / 64] * 32,
         "calib": [0.0, 0.01, -0.02], "wide_from_device_euler": [0.0, 0.003, 0.001],
         "raw": np.linspace(-1, 1, 40).astype(np.float16) * i}
  if i:
    rec["desire_input"] = desire
    rec["desire_pulse"] = np.where(desire > 0.5, desire, 0) if i > 1 else None
  return rec


def test_model_record_json():
  """modeld's records through the link's JSON lines stack into the same arrays."""
  recs = [model_record(i) for i in range(5)] + [None]
  back = [model_record_from_json(json.loads(json.dumps(model_record_to_json(r)))) if r is not None else None for r in recs]
  a, b = stack_model(recs), stack_model(back)
  assert a.keys() == b.keys() and all(a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f") for k in a)


def car_state(i: int) -> dict:
  return {"t": 50.0 + i * 0.05, "inVehicle": True, "pos": [100.0, 200.0 + i * 0.5, 30.0], "heading": 0.0, "pitch": 0.01, "vEgo": 10.0,
          "aMeas": 0.0, "yawRate": 0.0, "rotVel": [0.0, 0.0, 0.0], "resets": 0, "collisions": 0, "user": {}, "lane": [1, 3]}


def nv12_frames(i: int) -> tuple[bytes, bytes]:
  """A frame as the plugin sends it (unpadded) and as camerad has it (padded): a gradient that moves on each frame."""
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
  from openpilot.tools.sim.lib.common import W, H
  y = ((np.arange(W)[None, :] + np.arange(H)[:, None] * 2 + i * 8) % 220 + 16).astype(np.uint8)
  uv = np.full((H // 2, W), 128, np.uint8)
  uv[:, ::2] = (np.arange(H // 2)[:, None] + i * 4) % 64 + 96
  stride, y_height, uv_height, size = get_nv12_info(W, H)
  padded = np.zeros(size, np.uint8)
  planes = padded[:stride * (y_height + uv_height)].reshape(-1, stride)
  planes[:H, :W], planes[y_height:y_height + H // 2, :W] = y, uv
  return y.tobytes() + uv.tobytes(), padded.tobytes()


def safetensors(path: Path) -> dict[str, np.ndarray]:
  data = path.read_bytes()
  n = int.from_bytes(data[:8], "little")
  head = json.loads(data[8:8 + n])
  dtypes = {"F64": "<f8", "F32": "<f4", "I64": "<i8", "U32": "<u4", "U8": "u1"}
  return {k: np.frombuffer(data[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]], dtype=dtypes[v["dtype"]]).reshape(v["shape"])
          for k, v in head.items() if k != "__metadata__"}


def finished(root: Path) -> Path:
  segs = sorted((root / "data").iterdir())
  assert len(segs) == 1
  deadline = time.monotonic() + 120
  while True:  # finalize writes gta5.json last, with video_frames
    try:
      if "video_frames" in json.loads((segs[0] / "gta5.json").read_text()):
        return segs[0]
    except (OSError, ValueError):
      pass
    assert time.monotonic() < deadline, (segs[0] / "finalize.log").read_text()
    time.sleep(0.2)


@pytest.mark.skipif(shutil.which(gta5_record.FFMPEG) is None, reason="no ffmpeg")
def test_host_recording_matches_the_bridges(tmp_path):
  """The same drive recorded in the bridge (camerad's padded frames, modeld's records by frame id) and in the host (the
  plugin's frames by the relay's numbers, the link's camera ids and records through JSON): the same files, the host's
  npz with host_seq besides. With no link, the host's has no model columns and no frame ids."""
  n = 30
  bridge, host, alone = (Recorder(str(tmp_path / name), host=name != "bridge") for name in ("bridge", "host", "alone"))
  for i in range(n):
    plugin, padded = nv12_frames(i)
    state, t, seq = car_state(i), 10.0 + i * 0.05, 7000 + i
    bridge.add(False, padded, state, t)
    bridge.add(True, padded, state, t)
    bridge.on_model(i, model_record(i))
    host.on_link({"type": "camera", "seq": seq, "frame_id": i})
    host.on_link(json.loads(json.dumps({"type": "model", "frame_id": i, "rec": model_record_to_json(model_record(i))})))
    for rec in (host, alone):
      rec.add(False, memoryview(plugin), state, t, seq)
      rec.add(True, memoryview(plugin), state, t)
  for rec in (bridge, host, alone):
    rec.close()
  a, b, c = finished(tmp_path / "bridge"), finished(tmp_path / "host"), finished(tmp_path / "alone")
  files = {p.name for p in a.iterdir()}
  assert files == {p.name for p in b.iterdir()} == {p.name for p in c.iterdir()} and "gta5.npz" in files
  for name in ("fcamera.hevc", "ecamera.hevc", "routes.json"):
    assert (a / name).read_bytes() == (b / name).read_bytes() == (c / name).read_bytes(), name
  for name in ("frame_info.safetensors", "localizer.safetensors"):
    fa, fb = safetensors(a / name), safetensors(b / name)
    assert fa.keys() == fb.keys() and all(np.array_equal(fa[k], fb[k]) for k in fa), name
  ia, ib, ic = (json.loads((s / "gta5.json").read_text()) for s in (a, b, c))
  assert ia.keys() == ib.keys() == ic.keys() and ia["frames"] == ia["video_frames"] == ia["model_frames"] == n
  same = lambda d: {k: v for k, v in d.items() if k not in ("route", "started")}  # noqa: E731
  assert same(ia) == same(ib) and ic["model_frames"] == 0 and ic["first_frame_id"] is None
  na, nb, nc = (dict(np.load(s / "gta5.npz")) for s in (a, b, c))
  assert nb.keys() == na.keys() | {"host_seq"} and list(nb["host_seq"]) == list(range(7000, 7000 + n))
  assert all(na[k].dtype == nb[k].dtype and np.array_equal(na[k], nb[k], equal_nan=na[k].dtype.kind == "f") for k in na)
  assert nc.keys() == {k for k in nb if not k.startswith("model")} and np.isnan(nc["frame"][:, 0]).all()
  assert np.array_equal(nc["frame"][:, 1:], na["frame"][:, 1:], equal_nan=True)


def free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


# a stand-in plugin and bridge, each in its own process, as the game and the link are: a busy test process would hold them
# up as it holds up anything in it
PLUGIN = r"""
import json, os, socket, struct, sys, time
port, spec, go = int(sys.argv[1]), json.load(open(sys.argv[2])), sys.argv[3]
fb = spec["frame_bytes"]

def msg(state, road, wide):
  view = lambda v: os.urandom(fb) if v is None else bytes([v]) * fb
  head = json.dumps({"type": "frame", "width": spec["width"], "height": spec["height"], "views": ["road", "wide"],
                     "state": state}).encode()
  body = struct.pack("<I", len(head)) + head + view(road) + view(wide)
  return struct.pack("<I", len(body)) + body

end = time.monotonic() + 30
while True:
  try:
    s = socket.create_connection(("127.0.0.1", port), timeout=1.0)
    break
  except OSError:
    if time.monotonic() > end:
      sys.exit("no relay")
    time.sleep(0.05)
s.settimeout(None)
warm = msg(spec["warmup"], 1, 1)
while not os.path.exists(go):  # until the recorder's frame worker is up
  s.sendall(warm)
  time.sleep(0.05)
for state, road, wide in spec["frames"]:
  s.sendall(msg(state, road, wide))
  time.sleep(0.05)
time.sleep(2.0)
"""
BRIDGE = r"""
import socket, sys
conn, _ = socket.create_server(("127.0.0.1", int(sys.argv[1]))).accept()
while conn.recv(1 << 20):
  pass
"""


@pytest.fixture
def relay(tmp_path):
  """A relay and a stand-in bridge draining it, in processes of their own and ended with the test: (the relay's plugin
  port, its frames port)."""
  bridge_port, ports = free_port(), (free_port(), free_port())
  bridge = subprocess.Popen([sys.executable, "-c", BRIDGE, str(bridge_port)])
  (tmp_path / "link.txt").write_text(f"127.0.0.1:{bridge_port}\n")
  proc = subprocess.Popen([sys.executable, gta5_relay.__file__, "--port", str(ports[0]), "--frames-port", str(ports[1]),
                           "--brain-port", str(free_port()), "--plugin-file", str(tmp_path / "plugin.txt"), "--link-file",
                           str(tmp_path / "link.txt")], stdout=subprocess.DEVNULL)
  try:
    yield ports
  finally:
    for p in (proc, bridge):
      p.kill()
      p.wait()


def padded(frame: bytes) -> bytes:
  """A frame as the plugin sends it, padded as camerad's buffers are."""
  from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
  from openpilot.tools.sim.lib.common import W, H
  stride, y_height, uv_height, size = get_nv12_info(W, H)
  out = np.zeros(size, np.uint8)
  planes = out[:stride * (y_height + uv_height)].reshape(-1, stride)
  src = np.frombuffer(frame, np.uint8)
  planes[:H, :W] = src[:W * H].reshape(H, W)
  planes[y_height:y_height + H // 2, :W] = src[W * H:].reshape(H // 2, W)
  return out.tobytes()


@pytest.mark.skipif(shutil.which(gta5_record.FFMPEG) is None, reason="no ffmpeg")
def test_frame_worker(tmp_path, relay):
  """The host's recorder through its frame worker process while this process is kept busy, as the brain keeps the
  host's (a busy process took frames 0.5-1.5 s apart and broke recordings into segments a frame long): one segment of
  every frame whose views match, timed and numbered by the worker, a mismatched pair dropped, and the same video as the
  bridge's recorder makes of the same frames."""
  from openpilot.tools.sim.bridge.gta5.gta5_rx import FRAME_BYTES
  from openpilot.tools.sim.lib.common import W, H
  relay_port, frames_port = relay
  rec = Recorder(str(tmp_path / "host"), host=True)
  worker = FrameWorker(frames_port, rec)
  values = [40 + 4 * i for i in range(40)]  # flat views always match
  frames = [[car_state(i), v, v] for i, v in enumerate(values)]
  frames.insert(10, [car_state(10), None, None])  # noise: a mismatched pair
  spec = {"frame_bytes": FRAME_BYTES, "width": W, "height": H, "warmup": {**car_state(0), "inVehicle": False}, "frames": frames}
  (tmp_path / "spec.json").write_text(json.dumps(spec))
  go = tmp_path / "go"
  plugin = subprocess.Popen([sys.executable, "-c", PLUGIN, str(relay_port), str(tmp_path / "spec.json"), str(go)])
  stop = threading.Event()

  def busy():
    while not stop.is_set():
      x = 0
      for i in range(1000):
        x += i * i

  try:
    end = time.monotonic() + 60
    while not worker.seen:  # the worker's process starting, out of the car: no segment yet
      assert time.monotonic() < end and plugin.poll() is None, "the frame worker never sent a frame"
      time.sleep(0.05)
    threading.Thread(target=busy, daemon=True).start()
    go.write_text("go")
    bridge_rec = Recorder(str(tmp_path / "bridge"))
    for i, v in enumerate(values):
      f = padded(bytes([v]) * FRAME_BYTES)
      bridge_rec.add(False, f, car_state(i), i * 0.05)
      bridge_rec.add(True, f, car_state(i), i * 0.05)
    end = time.monotonic() + 30
    while rec.segment is None or len(rec.segment.rows) < len(values):
      assert time.monotonic() < end, (rec.segment_count, rec.segment and len(rec.segment.rows))
      time.sleep(0.05)
  finally:
    stop.set()
    plugin.kill()
    plugin.wait()
  rec.close()
  bridge_rec.close()
  a, b = finished(tmp_path / "bridge"), finished(tmp_path / "host")
  info = json.loads((b / "gta5.json").read_text())
  assert rec.segment_count == 1 and info["frames"] == len(values) and info["end"] == "the recording stopped" and not info["ended"]
  npz = np.load(b / "gta5.npz")
  assert len(set(npz["host_seq"])) == len(values) and np.all(np.diff(npz["frame"][:, 1]) < gta5_record.GAP)
  for name in ("fcamera.hevc", "ecamera.hevc"):
    assert (a / name).read_bytes() == (b / name).read_bytes(), name
  assert worker.proc.poll() is not None  # closed with the recorder


@pytest.mark.skipif(shutil.which(gta5_record.FFMPEG) is None, reason="no ffmpeg")
def test_frame_worker_outlives_no_host(tmp_path):
  """A host that dies with a segment open: its frame worker, unable to tell it anything more, still finishes the videos
  and ends, leaving no encoder behind."""
  proc = subprocess.Popen([sys.executable, "-m", "openpilot.tools.sim.bridge.gta5.gta5_record", "frames", "--port", str(free_port())],
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE)
  proc.stdin.write((json.dumps({"op": "open", "id": "v", "path": str(tmp_path / "fcamera.hevc")}) + "\n").encode())
  proc.stdin.flush()
  time.sleep(2.0)  # the encoder starts
  proc.stdout.close()  # as a dead host's end of the pipe
  proc.stdin.close()
  assert proc.wait(30) == 0
  assert (tmp_path / "fcamera.hevc").exists()
