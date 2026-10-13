#!/usr/bin/env python3
"""The Windows host: the bridge's brain (gta5_brain.py) on Windows, the game through the relay (gta5_relay.py, its
brain port) and openpilot through the link (the bridge in WSL run with GTA5_LINK=1, gta5_link.py).

  python -m openpilot.tools.sim.bridge.gta5.gta5_host [--brain-port 8784] [--link-file C:\\Users\\Public\\gta5op-link.txt]
                                                      [--no-link]

Each step (STEP s) it takes the newest game state from the relay and steps the brain with it, which drives the car
through the relay itself when its own driver drives (the map driver, expert mode), and otherwise leaves the car's
controls to the link's openpilot: it tells the relay whose they are. It waits for the link's first state before it
starts, as the brain reads openpilot's params once. The brain's settings are the bridge's environment variables
(GTA5_MAP, GTA5_ROUTER, GTA5_DRIVER, GTA5_LOG...), as paths on Windows; GTA5_MAP_PORT is 8783 here by default, as WSL's
localhost forwarding holds 8793 while the link's world could serve it.

--no-link runs it without openpilot: the game, the relay and the host alone, no WSL. openpilot is then gta5_opio.NullIO,
never engaged, and the engage key drives with the map driver alone (whatever GTA5_DRIVER says); without a destination
it cruises on along the road (map/cruise.py, with GTA5_ROUTER), and does nothing without a router. gta5_cmd.py's debug
port (gta5_ports.DEBUG_PORT) is then the host's, which passes its commands to the plugin as they are.

GTA5_RECORD=<folder> records here (gta5_record.py): the frames from the relay's frames port (--frames-port), modeld's
outputs from the link's model tap (none with --no-link), the encoder GTA5_RECORD_FFMPEG's (else ffmpeg on the PATH)."""
import argparse
import gc
import json
import os
import socket
import threading
import time

from openpilot.tools.sim.bridge.gta5.gta5_brain import MAP, NAV_MATCH, Brain
from openpilot.tools.sim.bridge.gta5.gta5_opio import NullIO, RemoteIO, link_address
from openpilot.tools.sim.bridge.gta5.gta5_ports import DEBUG_PORT
from openpilot.tools.sim.bridge.gta5.gta5_record import RECORD, FrameWorker, Recorder
from openpilot.tools.sim.bridge.gta5.gta5_relay import BRAIN_PORT, FRAMES_PORT, LINK_FILE
from openpilot.tools.sim.bridge.gta5.map.map_view import MapView

STEP = 0.01  # s: the bridge's 100 Hz
FRESH = 0.5  # s: a game state older than this isn't driven on
MAP_PORT = int(os.getenv("GTA5_MAP_PORT", "8783"))
LOG = os.getenv("GTA5_LOG")
LINK_WAIT = 60.0  # s for the link's first state before starting without openpilot's params
# as the bridge's world: full collections walk the maps' million objects for most of a second, so they run only while
# the car's out of reach (no fresh state, or on foot) this long, at most every FULL_GC_EVERY s
FULL_GC_IDLE, FULL_GC_EVERY = 2.0, 60.0  # s


class Relay:
  """The relay's brain port: the newest game state in, plugin commands out."""
  def __init__(self, port: int):
    self.port = port
    self.lock = threading.Lock()
    self.sock: socket.socket | None = None
    self.state: dict | None = None
    self.state_t = 0.0
    threading.Thread(target=self._run, daemon=True).start()

  def send(self, cmd: dict):
    with self.lock:
      sock = self.sock
    if sock is None:
      return
    try:
      sock.sendall((json.dumps(cmd) + "\n").encode())
    except OSError:
      pass  # the reader sees it go

  def serve_debug(self, port: int):
    """gta5_cmd.py's commands straight to the plugin (the link's own debug port is in WSL, if it runs at all)."""
    try:
      srv = socket.create_server(("127.0.0.1", port))
    except OSError as e:
      print(f"gta5: no debug commands on 127.0.0.1:{port} ({e})", flush=True)
      return
    threading.Thread(target=self._debug_loop, args=(srv,), daemon=True).start()

  def _debug_loop(self, srv: socket.socket):
    while True:
      conn, _ = srv.accept()
      with conn, conn.makefile("r") as f:
        for line in f:
          try:
            self.send(json.loads(line))
          except ValueError:
            continue

  def newest(self) -> tuple[dict | None, float]:
    with self.lock:
      return self.state, self.state_t

  def _run(self):
    last = None
    while True:
      try:
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=1.0)
      except OSError as e:
        if str(e) != last:
          print(f"gta5: waiting for the relay on 127.0.0.1:{self.port} ({e})", flush=True)
          last = str(e)
        time.sleep(1.0)
        continue
      sock.settimeout(None)
      sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
      print(f"gta5: connected to the relay on 127.0.0.1:{self.port}", flush=True)
      last = None
      with self.lock:
        self.sock = sock
      try:
        for line in sock.makefile("rb"):
          try:
            head = json.loads(line)
          except ValueError:
            continue
          with self.lock:
            self.state, self.state_t = head.get("state"), time.monotonic()
      except OSError:
        pass
      with self.lock:
        self.sock = None
      sock.close()
      print("gta5: the relay went", flush=True)


def main(argv: list[str] | None = None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--brain-port", type=int, default=BRAIN_PORT)
  ap.add_argument("--frames-port", type=int, default=FRAMES_PORT)
  ap.add_argument("--link-file", default=LINK_FILE)
  ap.add_argument("--no-link", action="store_true", help="no openpilot: the game, the relay and the host alone")
  args = ap.parse_args(argv)
  io = NullIO() if args.no_link else RemoteIO(link_address(args.link_file))
  relay = Relay(args.brain_port)
  if args.no_link:
    relay.serve_debug(DEBUG_PORT)
  if isinstance(io, RemoteIO):
    end = time.monotonic() + LINK_WAIT
    while not io.connected and time.monotonic() < end:
      time.sleep(0.1)
    io.update()
    if not io.connected:
      print("gta5: no state from the link yet: starting with openpilot's params unknown", flush=True)
  else:
    print("gta5: no link: no openpilot", flush=True)
  log = open(LOG, "a", buffering=1) if LOG else None
  brain = Brain(io, relay.send, log, match=NAV_MATCH and not args.no_link)  # the fixes are the link's
  if RECORD:
    brain.recorder = Recorder(RECORD, brain, host=True)
    io.model_tap(brain.recorder.on_link)
    FrameWorker(args.frames_port, brain.recorder)
  if MAP:
    brain.map_view = MapView(os.path.join(MAP, "roads.json"), MAP_PORT, host="127.0.0.1")  # 0.0.0.0 asks the firewall
    print(f"gta5: map view on http://localhost:{MAP_PORT}/", flush=True)
  gen0, gen1, _ = gc.get_threshold()
  gc.set_threshold(gen0, gen1, 1 << 30)
  idle_since, next_gc = None, 0.0
  owner = None
  next_t = time.monotonic()
  try:
    while True:
      now = time.monotonic()
      io.update()
      state, t = relay.newest()
      if state is not None and now - t < FRESH and state.get("inVehicle"):
        idle_since = None
        brain.step(state)
        brain.hud()
        mine = brain.driving is not None and brain.driving is not brain.openpilot
        if ("brain" if mine else "link") != owner:
          owner = "brain" if mine else "link"
          relay.send({"type": "owner", "who": owner})
        if mine:
          relay.send(brain.controls(state))
      elif idle_since is None:
        idle_since = now
      elif now - idle_since > FULL_GC_IDLE and now >= next_gc:
        next_gc = now + FULL_GC_EVERY
        gc.collect()
      next_t += STEP
      time.sleep(max(0.0, next_t - time.monotonic()))
      if time.monotonic() - next_t > 1.0:
        next_t = time.monotonic()  # fell behind (a pause): don't race to catch up
  except KeyboardInterrupt:
    pass
  finally:
    relay.send({"type": "owner", "who": "link"})
    brain.close()
    if brain.recorder is not None:
      brain.recorder.close()
    if brain.map_view is not None:
      brain.map_view.close()


if __name__ == "__main__":
  main()
