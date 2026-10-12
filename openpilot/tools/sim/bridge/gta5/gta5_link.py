"""The link: the bridge in WSL as openpilot's side of the Windows host (gta5_host.py), run with GTA5_LINK=1.

The game's frames and state reach the link's rx through the host's relay (gta5_relay.py), as they reach the bridge
directly, and the link sends openpilot's controls back the same way: the relay passes them to the plugin only while
openpilot drives. The decisions are the host's brain's (gta5_brain.py): its calls on openpilot (gta5_opio.RemoteIO)
arrive here as JSON lines on LINK_PORT, and LinkServer applies them to the world's LocalIO on the bridge's own thread,
once per step, and sends back openpilot's state (LocalIO.status) every STATUS_EVERY s. One host at a time; when it goes,
the simulated car's inputs go back to their defaults."""
import json
import queue
import socket
import threading
import time

from openpilot.tools.sim.bridge.gta5.gta5_opio import LINK_PORT, OPS, STATUS_EVERY, close


class LinkServer:
  def __init__(self, io, port: int = LINK_PORT):
    """io: the world's LocalIO."""
    self.io = io
    self.calls: queue.SimpleQueue = queue.SimpleQueue()  # (op, args) from the host, None when it went
    self.lock = threading.Lock()
    self.conn: socket.socket | None = None
    self.next_status = 0.0
    self.srv = socket.create_server(("0.0.0.0", port), reuse_port=True)
    threading.Thread(target=self._serve, daemon=True).start()
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

  def step(self):
    """The host's calls since the last step, then openpilot's state to it when due."""
    while True:
      try:
        call = self.calls.get_nowait()
      except queue.Empty:
        break
      if call is None:
        self.io.reset()
        continue
      op, args = call
      try:
        self._apply(op, args)
      except (TypeError, ValueError) as e:
        print(f"gta5: link: bad {op} {args!r}: {e}", flush=True)
    now = time.monotonic()
    with self.lock:
      conn = self.conn
    if conn is None or now < self.next_status:
      return
    self.next_status = now + STATUS_EVERY
    try:
      conn.sendall((json.dumps(self.io.status()) + "\n").encode())
    except OSError:
      pass  # the reader sees it go

  def _apply(self, op: str, args: list):
    if op == "set_inputs":
      self.io.set_inputs(**args[0])
    elif op == "param_put":
      self.io.params.put(args[0], args[1])
    elif op == "param_remove":
      self.io.params.remove(args[0])
    else:
      getattr(self.io, op)(*args)

  def close(self):
    self.srv.close()
    with self.lock:
      if self.conn is not None:
        close(self.conn)
