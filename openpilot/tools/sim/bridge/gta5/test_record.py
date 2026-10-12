"""The recorder's segments as frames come (gta5_record.py), with stand-in encoders; and the same recording made in the
bridge and in the Windows host, from the same synthetic drive, with the real encoder where there's one."""
import json
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.tools.sim.bridge.gta5 import gta5_record, gta5_relay
from openpilot.tools.sim.bridge.gta5.gta5_record import Recorder, RelayFrames, model_record_from_json, model_record_to_json, stack_model


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
  def __init__(self, path, info, size):
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


def plugin_message(i: int, road: bytes, wide: bytes) -> bytes:
  from openpilot.tools.sim.lib.common import W, H
  header = json.dumps({"type": "frame", "t": i, "width": W, "height": H, "views": ["road", "wide"], "state": car_state(i)}).encode()
  body = struct.pack("<I", len(header)) + header + road + wide
  return struct.pack("<I", len(body)) + body


def free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


@pytest.fixture
def relay(tmp_path):
  """A relay in its own process, ended with the test (a relay keeps trying to reach its bridge), and its stand-in bridge:
  (its plugin port, its frames port, the bridge's listening socket)."""
  bridge_srv = socket.create_server(("127.0.0.1", 0))
  link_file = tmp_path / "link.txt"
  link_file.write_text(f"127.0.0.1:{bridge_srv.getsockname()[1]}\n")
  ports = free_port(), free_port()
  proc = subprocess.Popen([sys.executable, gta5_relay.__file__, "--port", str(ports[0]), "--frames-port", str(ports[1]),
                           "--brain-port", str(free_port()), "--plugin-file", str(tmp_path / "plugin.txt"), "--link-file", str(link_file)],
                          stdout=subprocess.DEVNULL)
  try:
    yield ports[0], ports[1], bridge_srv
  finally:
    proc.kill()
    proc.wait()
    bridge_srv.close()


def test_relay_frames(tmp_path, monkeypatch, relay):
  """The host's recorder fed from the relay's frames port: each frame with the relay's number for it, as the bridge got
  it; a pair whose views don't match dropped, as the link's rx drops it."""
  from openpilot.tools.sim.bridge.gta5.gta5_rx import FRAME_BYTES
  monkeypatch.setattr(gta5_record, "RETRY_EVERY", 0.05)
  monkeypatch.setattr(gta5_record, "Segment", Segment)
  monkeypatch.setattr(gta5_record, "MODEL_WAIT", 1e9)
  relay_port, frames_port, bridge_srv = relay
  rec = Recorder(str(tmp_path), host=True)
  feed = RelayFrames(frames_port, rec)
  bridge, _ = bridge_srv.accept()
  up = bytearray()

  def read_up():
    try:
      while data := bridge.recv(1 << 20):
        up.extend(data)
    except OSError:  # closed as the test ends
      pass

  threading.Thread(target=read_up, daemon=True).start()
  end = time.monotonic() + 5
  while True:
    try:
      game = socket.create_connection(("127.0.0.1", relay_port), timeout=1.0)
      break
    except OSError:
      assert time.monotonic() < end
      time.sleep(0.05)
  time.sleep(0.5)  # the recorder's connection is the relay's
  rng = np.random.default_rng(1)
  frames = [(bytes([60 + i]) * FRAME_BYTES, bytes([60 + i]) * FRAME_BYTES) for i in range(4)]  # flat views match
  frames.insert(2, (rng.integers(0, 255, FRAME_BYTES, np.uint8).tobytes(), rng.integers(0, 255, FRAME_BYTES, np.uint8).tobytes()))
  for i, (road, wide) in enumerate(frames):
    game.sendall(plugin_message(i, road, wide))
    time.sleep(0.2)  # the recorder keeps up, so no frame is replaced by a newer one; the dropped pair's gap stays under GAP

  def bridge_seqs() -> list[int]:
    """The relay's numbers in the whole messages the bridge has had."""
    seqs, at, got = [], 0, bytes(up)
    while at + 8 <= len(got) and at + 4 + struct.unpack_from("<I", got, at)[0] <= len(got):
      head_len = struct.unpack_from("<I", got, at + 4)[0]
      seqs.append(json.loads(got[at + 8:at + 8 + head_len])["hostSeq"])
      at += 4 + struct.unpack_from("<I", got, at)[0]
    return seqs

  end = time.monotonic() + 10
  while len(bridge_seqs()) < 5 or rec.segment is None or len(rec.segment.seqs) < 4:
    assert time.monotonic() < end, (len(up), feed.mismatched)
    time.sleep(0.05)
  seqs = bridge_seqs()
  assert len(seqs) == 5 and seqs == list(range(seqs[0], seqs[0] + 5)) and feed.mismatched == 1
  kept = frames[:2] + frames[3:]
  assert rec.segment.seqs == seqs[:2] + seqs[3:]
  assert rec.segment.road.frames == [r for r, _ in kept] and rec.segment.wide.frames == [w for _, w in kept]
  rec.stopped = True
  for s in (game, bridge):
    s.close()
