"""The link: the bridge in WSL as openpilot's side of the Windows host (gta5_host.py), run with GTA5_LINK=1.

The game's frames and state reach the link's rx through the host's relay (gta5_relay.py), as they reach the bridge
directly, and the link sends openpilot's controls back the same way: the relay passes them to the plugin only while
openpilot drives. The decisions are the host's brain's (gta5_brain.py): its calls on openpilot (gta5_opio.RemoteIO)
arrive here as JSON lines on LINK_PORT, and LinkServer applies them to the world's LocalIO on the bridge's own thread,
once per step, and sends back openpilot's state (LocalIO.status) every STATUS_EVERY s. One host at a time; when it goes,
the simulated car's inputs go back to their defaults.

It also stands in for the host in WSL, where gta5_cmd.py and the harnesses drive expert mode: the expert control file's
changes while a host is connected go to it (gta5_opio.COMMANDS), and what the host's expert mode writes to its control
file is written to this one, so a harness waiting for a trip's end sees it; a destination posted to its map view
(gta5_cmd.py expert route, the map view on WSL's GTA5_MAP_PORT, which takes destinations alone here) goes to the host
too. A host can't listen where WSL reaches it: Windows' firewall blocks connections from WSL.

For the host's recorder the link is a model tap, while the host asks for it (op model_tap): modeld's record for each
frame (gta5_record.ModelTap) as {"type": "model", "frame_id", "rec"}, and for each road frame handed to camerad its
frame id and the relay's number for the frame, {"type": "camera", "seq", "frame_id"}."""
import json
import os
import queue
import socket
import threading
import time
from pathlib import Path

from openpilot.tools.sim.bridge.gta5.gta5_opio import LINK_PORT, OPS, STATUS_EVERY, close
from openpilot.tools.sim.bridge.gta5.gta5_record import ModelTap, model_record_to_json

CONTROL_EVERY = 0.5  # s between looks at the expert control file
TAP_BACKLOG = 400  # lines waiting for the host past which the model tap's are dropped (about 10 s of them)


class LinkServer:
  def __init__(self, io, port: int = LINK_PORT, control: Path | None = None, model_tap=ModelTap):
    """io: the world's LocalIO; control: the expert control file to pass on; model_tap(on_record) starts modeld's
    records, the first time the host asks for them."""
    self.io = io
    self.calls: queue.SimpleQueue = queue.SimpleQueue()  # (op, args) from the host, None when it went
    self.lines: queue.SimpleQueue = queue.SimpleQueue()  # messages for the host, sent in order on their own thread
    self.lock = threading.Lock()
    self.conn: socket.socket | None = None
    self.next_status = 0.0
    self.control = control
    self.control_mtime: float | None = None  # the file's as last passed on or written; None: to be read afresh
    self.next_control = 0.0
    self.view = None  # map_view.MapView, for the destinations posted to it
    self.out: list[dict] = []  # COMMANDS for the host, sent with the next status
    self.make_tap = model_tap
    self.tap = None
    self.tap_on = False  # the host's recorder wants the model tap
    self.road_id = 0  # camerad's id for the next road frame: one per camera_yuv call, as camerad counts them
    self.srv = socket.create_server(("0.0.0.0", port), reuse_port=True)
    threading.Thread(target=self._serve, daemon=True).start()
    threading.Thread(target=self._sender, daemon=True).start()
    print(f"gta5: the link waits for the host on port {port}", flush=True)

  def _serve(self):
    while True:
      try:
        conn, addr = self.srv.accept()
      except OSError:
        return
      conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
      with self.lock:
        old, self.conn = self.conn, conn
        self.control_mtime = None  # its changes from now on
      if old is not None:
        close(old)  # a new host replaces the last
      print(f"gta5: the host connected from {addr[0]}", flush=True)
      try:
        for line in conn.makefile("r", encoding="utf-8"):
          msg = json.loads(line)
          if msg.get("op") in OPS:
            self.calls.put((msg["op"], msg.get("args") or []))
      except (OSError, ValueError):
        pass
      with self.lock:
        if self.conn is conn:
          self.conn = None
          self.calls.put(None)
          print("gta5: the host went", flush=True)
      close(conn)

  def _sender(self):
    while True:
      msg = self.lines.get()
      with self.lock:
        conn = self.conn
      if conn is None:
        continue
      try:
        conn.sendall((json.dumps(msg) + "\n").encode())
      except OSError:
        pass  # the reader sees it go

  def step(self):
    """The host's calls since the last step, then openpilot's state to it when due."""
    while True:
      try:
        call = self.calls.get_nowait()
      except queue.Empty:
        break
      if call is None:
        self.io.reset()
        self.tap_on = False
        continue
      op, args = call
      try:
        self._apply(op, args)
      except (TypeError, ValueError) as e:
        print(f"gta5: link: bad {op} {args!r}: {e}", flush=True)
    now = time.monotonic()
    with self.lock:
      conn = self.conn
    picked = self.view.take_destination() if self.view is not None else None
    if conn is None:
      return
    if picked is not None:
      self.out.append({"type": "destination", "xy": picked[0], **({"add": True} if picked[1:] == ("add",) else {})})
    self._poll_control(now)
    if now < self.next_status:
      return
    self.next_status = now + STATUS_EVERY
    out, self.out = self.out, []
    for msg in [*out, self.io.status()]:
      self.lines.put(msg)

  def _poll_control(self, now: float):
    if self.control is None or now < self.next_control:
      return
    self.next_control = now + CONTROL_EVERY
    try:
      mtime = self.control.stat().st_mtime
    except OSError:
      mtime = 0.0
    with self.lock:
      last, self.control_mtime = self.control_mtime, mtime
    if last is None or mtime == last or not mtime:
      return
    try:
      self.out.append({"type": "expert", "cfg": json.loads(self.control.read_text())})
    except (OSError, ValueError) as e:
      print(f"gta5: link: expert control file: {e}", flush=True)

  def _write_control(self, cfg: dict):
    """The host's expert mode's write to its control file, to this one; not passed back."""
    if self.control is None:
      return
    tmp = self.control.with_suffix(".tmp")
    tmp.write_text(json.dumps(cfg) + "\n")
    os.replace(tmp, self.control)
    self.control_mtime = self.control.stat().st_mtime

  def _apply(self, op: str, args: list):
    if op == "expert_control":
      try:
        self._write_control(args[0])
      except OSError as e:
        print(f"gta5: link: expert control file: {e}", flush=True)
    elif op == "set_inputs":
      self.io.set_inputs(**args[0])
    elif op == "param_put":
      self.io.params.put(args[0], args[1])
    elif op == "param_remove":
      self.io.params.remove(args[0])
    elif op == "model_tap":
      self.tap_on = bool(args[0])
      if self.tap_on and self.tap is None:
        self.tap = self.make_tap(self._model)
    else:
      getattr(self.io, op)(*args)

  # *** the model tap ***

  def camera(self, seq: int | None):
    """A road frame handed to camerad, from the camera thread: seq the relay's number for it (None for a blank one)."""
    frame_id, self.road_id = self.road_id, self.road_id + 1
    if self.tap_on and seq is not None:
      self._tap({"type": "camera", "seq": seq, "frame_id": frame_id})

  def _model(self, frame_id: int, rec: dict):
    if self.tap_on:
      self._tap({"type": "model", "frame_id": frame_id, "rec": model_record_to_json(rec)})

  def _tap(self, msg: dict):
    if self.lines.qsize() < TAP_BACKLOG:  # a host that stops reading mustn't grow the queue without end
      self.lines.put(msg)

  def close(self):
    self.srv.close()
    if self.view is not None:
      self.view.close()
    with self.lock:
      if self.conn is not None:
        close(self.conn)
