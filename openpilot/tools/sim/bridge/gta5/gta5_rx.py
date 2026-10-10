"""Game connection for the GTA V bridge, run in its own process.

The plugin sends raw NV12 frames for both cameras (about 140 MB/s), which would compete for the GIL with the bridge's
100 Hz threads. This process owns the TCP connection, copies frames into shared memory and hands the bridge only small
messages: (slot, views, state). It also listens on localhost for debug commands (gta5_cmd.py), forwarded to the plugin,
and sets the plugin's map debug overlay and GPS route again each time the game connects."""
import json
import os
import socket
import struct
import threading
import time
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.tools.sim.bridge.gta5 import gta5_stacks
from openpilot.tools.sim.lib.camerad import W, H

VIEWS = ("road", "wide")
SLOTS = 8  # the bridge may lag a few frames behind when its threads are busy; a slot written over while it is read mixes frames
NV12_SIZE = get_nv12_info(W, H)[3]
FRAME_BYTES = W * H * 3 // 2  # one view as the plugin sends it: Y rows, then interleaved UV rows, unpadded
DEBUG_PORT = 8792
# Plugin settings that reset when its core reloads (a DLL swap, a GTA restart), which reconnects it: the receiver keeps
# the last sent and sets them again on each connection. Each command type's key in the plugin's state, and the settings
# the state reports as the command takes them.
DISPLAY = {"debug": ("debug", ("layers", "force", "ground", "thin", "width", "dist", "grow", "sides", "flip")),
           "gpsroute": ("gpsRoute", ("max",))}
# The game sometimes renders an openpilot frame with the previous openpilot frame's camera, so a road frame can hold the
# wide render or the other way round. A pair matches when the wide frame's center is the road frame shrunk by the
# lenses' focal length ratio; mismatched pairs (correlation under the threshold) are dropped.
WIDE_FROM_ROAD_SCALE = 597.732 / 2600.85
MATCH_THRESHOLD = 0.4
MATCH_SHIFT = 4  # px at quarter size, searched each way


def nv12_planes(buf) -> tuple[np.ndarray, np.ndarray]:
  """Y and UV (interleaved, W bytes per row) views into a padded NV12 buffer for a W x H frame."""
  stride, y_height, uv_height, size = get_nv12_info(W, H)
  planes = np.frombuffer(buf, dtype=np.uint8, count=size)[:stride * (y_height + uv_height)].reshape(-1, stride)
  return planes[:H, :W], planes[y_height:y_height + H // 2, :W]


def views_match(road_y: np.ndarray, wide_y: np.ndarray) -> float:
  """Normalized correlation of the road frame's center, shrunk to the wide lens's scale, with the wide frame's center."""
  from PIL import Image
  r, w = road_y[::4, ::4].astype(np.float32), wide_y[::4, ::4].astype(np.float32)
  h, wd = r.shape
  ch, cw = int(h * 0.6), int(wd * 0.6)
  crop = r[(h - ch) // 2:(h + ch) // 2, (wd - cw) // 2:(wd + cw) // 2]
  sh, sw = round(ch * WIDE_FROM_ROAD_SCALE), round(cw * WIDE_FROM_ROAD_SCALE)
  small = np.asarray(Image.fromarray(crop).resize((sw, sh), Image.BILINEAR), dtype=np.float32)
  a = small - small.mean()
  best = -1.0
  # the wide frame comes a little after the road one, so in a turn the view has moved on a few pixels
  for dy in range(-MATCH_SHIFT, MATCH_SHIFT + 1):
    for dx in range(-MATCH_SHIFT, MATCH_SHIFT + 1):
      y0, x0 = (h - sh) // 2 + dy, (wd - sw) // 2 + dx
      center = w[y0:y0 + sh, x0:x0 + sw]
      b = center - center.mean()
      denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
      best = max(best, float((a * b).sum()) / denom if denom > 1e-3 else 1.0)
  return best


def nv12_to_rgb(frame: np.ndarray) -> np.ndarray:
  """Unpadded NV12 (as the plugin sends it) -> RGB, BT.601 limited range, for snapshots."""
  y = frame[:W * H].reshape(H, W).astype(np.float32) - 16
  uv = frame[W * H:].reshape(H // 2, W // 2, 2).astype(np.float32) - 128
  uv = uv.repeat(2, axis=0).repeat(2, axis=1)
  u, v = uv[..., 0], uv[..., 1]
  rgb = np.stack([1.164 * y + 1.596 * v, 1.164 * y - 0.392 * u - 0.813 * v, 1.164 * y + 2.017 * u], axis=-1)
  return np.clip(rgb, 0, 255).astype(np.uint8)


def recv_exact(sock: socket.socket, view: memoryview) -> bool:
  while len(view):
    n = sock.recv_into(view)
    if n == 0:
      return False
    view = view[n:]
  return True


class Receiver:
  def __init__(self, frames: Connection, controls: Connection, shm_names: dict[str, str], latest=None,
               display: list[dict] | None = None, debug_port: int = DEBUG_PORT):
    self.latest = latest  # the newest frame's seq, for the bridge to tell a slot written over since it was announced
    self.port = 0
    self.debug_port = debug_port
    self.display = {cmd["type"]: dict(cmd) for cmd in display or []}  # DISPLAY commands, each type's merged
    self.display_unconfirmed: set[str] = set()  # types sent that the plugin's state doesn't show yet
    self.display_lock = threading.Lock()
    self.frames = frames
    self.controls = controls
    self.shm = {name: SharedMemory(name=shm_names[name]) for name in VIEWS}
    self.lock = threading.Lock()
    self.conn: socket.socket | None = None
    self.seq = 0
    self.snap_path: str | None = None
    self.state_path: str | None = None
    self.mismatched = 0
    self.bad_headers = 0
    self.burst_path = ""
    self.burst_left = 0

  def send_plugin(self, obj: dict) -> None:
    data = (json.dumps(obj) + "\n").encode()
    with self.lock:
      if self.conn is not None:
        try:
          self.conn.sendall(data)
        except OSError:
          pass

  def control_loop(self) -> None:
    while True:
      try:
        self.send_plugin(self.controls.recv())
      except (EOFError, OSError):
        return

  def debug_loop(self, srv: socket.socket) -> None:
    while True:
      conn, _ = srv.accept()
      with conn, conn.makefile("r") as f:
        for line in f:
          try:
            cmd = json.loads(line)
          except json.JSONDecodeError:
            continue
          self.command(cmd)

  def command(self, cmd: dict) -> None:
    if cmd.get("type") == "snap":
      self.snap_path = cmd.get("path", "/tmp/gta5")
    elif cmd.get("type") == "state":
      self.state_path = cmd.get("path", "/tmp/gta5state.json")
    elif cmd.get("type") == "burst":
      self.burst_path, self.burst_left = cmd.get("path", "/tmp/gta5burst"), int(cmd.get("count", 40))
    elif cmd.get("type") in DISPLAY:
      with self.display_lock:
        # the plugin keeps a setting a command leaves out, so the one to set again is all of them merged
        self.display[cmd["type"]] = {**self.display.get(cmd["type"], {}), **cmd}
        self.display_unconfirmed.add(cmd["type"])
        self.send_plugin(cmd)
    else:
      self.send_plugin(cmd)

  def connected(self, conn: socket.socket | None) -> None:
    with self.lock:
      self.conn = conn
    if conn is None:
      return
    with self.display_lock:
      for cmd in self.display.values():
        self.send_plugin(cmd)
      self.display_unconfirmed = set(self.display)
      if self.display:
        print("gta5: set " + ", ".join(f"{t} {'on' if c.get('on') else 'off'}" for t, c in self.display.items()) + " again",
              flush=True)

  def follow_display(self, state: dict) -> None:
    """Takes up the settings the plugin had before this bridge started, and changes it made itself (F7 toggles the
    overlay), once its state shows what was sent last."""
    with self.display_lock:
      for t, (key, reported) in DISPLAY.items():
        got = state.get(key)
        if not isinstance(got, dict) or "on" not in got:
          continue
        on, want = int(bool(got["on"])), self.display.get(t)
        if want is None and not on:
          self.display_unconfirmed.discard(t)
          continue
        if want is None:
          want = self.display[t] = {"type": t, "on": on}
        elif t in self.display_unconfirmed:
          if int(bool(want.get("on"))) != on:
            continue
        elif int(bool(want.get("on"))) != on:
          want["on"] = on
        else:
          continue
        self.display_unconfirmed.discard(t)
        for k in reported:
          if k in got:
            want.setdefault(k, got[k])

  def on_frame(self, msg: memoryview) -> None:
    head_len = struct.unpack_from("<I", msg, 0)[0]
    try:
      head = json.loads(bytes(msg[4:4 + head_len]))
    except ValueError as e:  # a plugin bug in one header shouldn't take the camera down
      self.bad_headers += 1
      if self.bad_headers % 100 == 1:
        at = getattr(e, "pos", 0)
        near = bytes(msg[4 + max(at - 120, 0):4 + min(at + 60, head_len)])
        print(f"gta5: skipped a frame with a bad header ({self.bad_headers} so far): {e}: {near!r}", flush=True)
      return
    if self.state_path:  # the next frame's state alone, without its pictures as snap saves them
      with open(self.state_path, "w") as f:
        json.dump(head["state"], f, indent=1)
      self.state_path = None
    self.follow_display(head["state"])
    if (head["width"], head["height"]) != (W, H):
      raise ValueError(f"unsupported frame size {head['width']}x{head['height']}")
    slot = self.seq % SLOTS
    off = 4 + head_len
    if tuple(head["views"]) == VIEWS:
      road_y = np.frombuffer(msg[off:off + W * H], dtype=np.uint8).reshape(H, W)
      wide_y = np.frombuffer(msg[off + FRAME_BYTES:off + FRAME_BYTES + W * H], dtype=np.uint8).reshape(H, W)
      if views_match(road_y, wide_y) < MATCH_THRESHOLD:
        self.mismatched += 1
        if self.mismatched % 20 == 1:
          print(f"gta5: dropped a mismatched road/wide pair ({self.mismatched} so far)", flush=True)
        return
    for name in head["views"]:
      frame = np.frombuffer(msg[off:off + FRAME_BYTES], dtype=np.uint8)
      off += FRAME_BYTES
      buf = self.shm[name].buf
      assert buf is not None
      y, uv = nv12_planes(buf[slot * NV12_SIZE:(slot + 1) * NV12_SIZE])
      y[:] = frame[:W * H].reshape(H, W)
      uv[:] = frame[W * H:].reshape(H // 2, W)
      if self.snap_path:
        from PIL import Image
        Image.fromarray(nv12_to_rgb(frame)).save(f"{self.snap_path}_{name}.png")
      if self.burst_left:
        from PIL import Image
        # quarter-size luma, cheap enough to keep up with consecutive frames
        Image.fromarray(np.ascontiguousarray(y[::4, ::4])).save(f"{self.burst_path}_{self.seq:06d}_{name}.jpg", quality=80)
    if self.snap_path:
      with open(f"{self.snap_path}_state.json", "w") as f:
        json.dump(head, f, indent=1)
      print(f"gta5: saved {self.snap_path}_*.png", flush=True)
      self.snap_path = None
    if self.burst_left:
      self.burst_left -= 1
    self.seq += 1
    if self.latest is not None:
      self.latest.value = self.seq - 1
    self.frames.send((slot, list(head["views"]), head["state"], self.seq - 1))

  def serve(self, port: int, ready: Connection) -> None:
    srv = socket.create_server(("0.0.0.0", port), reuse_port=True)
    debug_srv = socket.create_server(("127.0.0.1", self.debug_port), reuse_port=True)
    self.port, self.debug_port = srv.getsockname()[1], debug_srv.getsockname()[1]
    threading.Thread(target=self.control_loop, daemon=True).start()
    threading.Thread(target=self.debug_loop, args=(debug_srv,), daemon=True).start()
    ready.send(None)
    buf = bytearray(64 << 20)
    while True:
      conn, addr = srv.accept()
      conn.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 16 << 20)
      conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
      print(f"gta5: game connected from {addr[0]}", flush=True)
      self.connected(conn)
      try:
        size = memoryview(bytearray(4))
        while recv_exact(conn, size):
          total = struct.unpack("<I", size)[0]
          if total > len(buf):
            buf = bytearray(total)
          msg = memoryview(buf)[:total]
          if not recv_exact(conn, msg):
            break
          self.on_frame(msg)
      except OSError:
        pass
      finally:
        self.connected(None)
        conn.close()
        print("gta5: game disconnected", flush=True)


def exit_with_parent(parent: int, every: float = 1.0) -> None:
  """Ends this process once the bridge's is gone: killed by a signal, it would leave this one serving the game and the
  debug port, so the bridge looks alive."""
  while os.getppid() == parent:
    time.sleep(every)
  print("gta5 rx: the bridge process is gone; exiting", flush=True)
  os._exit(1)


def rx_main(port: int, frames: Connection, controls: Connection, ready: Connection, shm_names: dict[str, str], latest=None,
            display: list[dict] | None = None) -> None:
  gta5_stacks.watch("rx")
  threading.Thread(target=exit_with_parent, args=(os.getppid(),), daemon=True).start()
  try:
    Receiver(frames, controls, shm_names, latest, display).serve(port, ready)
  except Exception as e:
    ready.send(f"{type(e).__name__}: {e}")
    raise
