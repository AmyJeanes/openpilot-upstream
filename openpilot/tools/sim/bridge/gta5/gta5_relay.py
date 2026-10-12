#!/usr/bin/env python3
"""The game's connection on Windows: the plugin connects to this relay on localhost, and the relay pipes it to the
bridge in WSL, so the game never needs WSL's address. Standard library only: it runs on Windows' own Python.

  py gta5_relay.py [--port 8781] [--brain-port 8784] [--plugin-file C:\\Users\\Public\\gta5op-bridge.txt]
                   [--link-file C:\\Users\\Public\\gta5op-link.txt]

It writes its own address to the plugin's bridge file; the bridge must then write its address elsewhere, so start it
with GTA5_BRIDGE_FILE=/mnt/c/Users/Public/gta5op-link.txt. A bridge started without that writes its address over the
relay's and the plugin connects to it directly again.

The relay keeps a connection to the bridge whenever it can, and listens for the plugin only while there's a bridge or a
brain to take the game, as the plugin switches to openpilot's camera as soon as it connects: with neither, the plugin's
connects are refused and the game stays as it is. The bridge takes the relay's connection for the game's, so it reports
the game connected while the relay waits for it; what the bridge sends meanwhile (the debug overlay's settings) waits
for the plugin. A plugin that leaves (the game closed, its core reloaded) ends the bridge's connection too, so the bridge
sees the game go and come back as before.

The brain (gta5_host.py, the bridge's brain on Windows, the bridge in WSL then being the link) connects on the brain
port: it gets each frame's header (the game state) as a JSON line, the newest only if it falls behind, and its lines go
to the plugin. Who drives decides whose controls the car takes: the brain says {"type": "owner", "who": "brain"} while
its own driver drives, and the bridge's control messages (openpilot's) are dropped until it says "link" again, or goes.
While a brain is connected the bridge's HUD messages are dropped too: the brain sends the HUD.

The game and the brain don't need the bridge: with a brain connected the plugin stays while the bridge is gone (the
link restarting, or a host run without openpilot), its frames dropped, and a bridge that comes back gets them from the
next message's start. With no brain, the bridge going ends the plugin's connection, as without a relay.
Every STATS_EVERY s it logs the frames piped up, MB/s and the longest gap between frames."""
import argparse
import json
import select
import socket
import struct
import sys
import threading
import time
from collections import deque

PORT = 8781
BRAIN_PORT = 8784
PLUGIN_FILE = r"C:\Users\Public\gta5op-bridge.txt"
LINK_FILE = r"C:\Users\Public\gta5op-link.txt"
CONNECT_TIMEOUT = 1.0  # s
RETRY_EVERY = 1.0  # s
CHUNK = 4 << 20
SOCK_BUF = 16 << 20
WAITING_MAX = 1 << 20  # bytes from the bridge kept for a plugin yet to connect
STATS_EVERY = 10.0  # s
CHECK_EVERY = 0.2  # s between looks at whether the game still has a bridge or a brain


def log(msg: str):
  print(f"{time.strftime('%H:%M:%S')} relay: {msg}", flush=True)


def tune(sock: socket.socket):
  sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, SOCK_BUF)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, SOCK_BUF)


def bridge_address(link_file: str) -> tuple[str, int] | None:
  try:
    host, _, port = open(link_file).read().strip().rpartition(":")
    return (host, int(port)) if host else None
  except (OSError, ValueError):
    return None


class Frames:
  """The plugin's messages as they're piped (each a 4-byte little-endian size, then that many bytes: a 4-byte header
  size, the header JSON, the frames): counted, and each one's header handed to on_header."""
  def __init__(self, on_header=None):
    self.on_header = on_header
    self.need, self.head = 0, b""  # bytes left of the message, else the size bytes read so far
    self.body = bytearray()  # the message's first bytes, while its header is wanted
    self.want = 0  # bytes of the message wanted for its header (4 + its size once known), 0 once it's had
    self.count = self.bytes = 0
    self.last: float | None = None
    self.gap = 0.0

  def feed(self, data: memoryview, now: float) -> list[int]:
    """Where in data messages start."""
    self.bytes += len(data)
    starts = []
    i = 0
    while i < len(data):
      if self.need:
        n = min(self.need, len(data) - i)
        if self.want:
          n = min(n, self.want - len(self.body))
          self.body += data[i:i + n]
          if self.want == 4 and len(self.body) == 4:
            self.want = 4 + struct.unpack("<I", self.body)[0]
          if len(self.body) == self.want:
            if self.on_header is not None:
              self.on_header(bytes(self.body[4:]))
            self.want = 0
        self.need -= n
        i += n
        continue
      if not self.head:
        starts.append(i)
      take = min(4 - len(self.head), len(data) - i)
      self.head += bytes(data[i:i + take])
      i += take
      if len(self.head) == 4:
        self.need, self.head = struct.unpack("<I", self.head)[0], b""
        self.body, self.want = bytearray(), 4 if self.need >= 4 else 0
        self.count += 1
        if self.last is not None:
          self.gap = max(self.gap, now - self.last)
        self.last = now
    return starts

  def take(self) -> tuple[int, int, float]:
    out = (self.count, self.bytes, self.gap)
    self.count = self.bytes = 0
    self.gap = 0.0
    return out


class Game:
  """The plugin's connection as the relay writes to it, from the bridge's pipe and the brain's alike, a line at a time,
  with who the car's controls belong to."""
  def __init__(self):
    self.lock = threading.Lock()
    self.sock: socket.socket | None = None
    self.owner = "link"  # whose control messages the car takes: the bridge's (openpilot's), or the brain's
    self.brain = False  # a brain is connected: the HUD is its

  def send(self, data: bytes):
    with self.lock:
      if self.sock is None:
        return
      try:
        self.sock.sendall(data)
      except OSError:
        pass  # the pipe sees it go

  def from_link(self, line: bytes):
    """One of the bridge's lines for the plugin: its controls only while the car is the link's, its HUD only without a
    brain."""
    if b'"control"' in line or b'"hud"' in line:
      try:
        kind = json.loads(line).get("type")
      except ValueError:
        kind = None
      if (kind == "control" and self.owner != "link") or (kind == "hud" and self.brain):
        return
    self.send(line)

  def from_brain(self, line: bytes):
    try:
      msg = json.loads(line)
    except ValueError:
      return
    if msg.get("type") == "owner":
      who = "brain" if msg.get("who") == "brain" else "link"
      if who != self.owner:
        self.owner = who
        log(f"the car's controls are the {who}'s")
      return
    if msg.get("type") == "control" and self.owner != "brain":
      return
    self.send(line if line.endswith(b"\n") else line + b"\n")


class Brain:
  """The brain's connection on the brain port: the newest header goes to it on its own thread, so a slow brain can't
  hold up the frames, and its lines go to the game."""
  def __init__(self, game: Game, port: int):
    self.game = game
    self.cv = threading.Condition()
    self.header: bytes | None = None
    self.sock: socket.socket | None = None
    self.srv = socket.create_server(("127.0.0.1", port))
    threading.Thread(target=self._serve, daemon=True).start()
    threading.Thread(target=self._sender, daemon=True).start()

  def on_header(self, header: bytes):
    with self.cv:
      if self.sock is not None:
        self.header = header
        self.cv.notify()

  def _sender(self):
    while True:
      with self.cv:
        while self.header is None or self.sock is None:
          self.cv.wait()
        header, sock, self.header = self.header, self.sock, None
      try:
        sock.sendall(header + b"\n")
      except OSError:
        pass  # its reader sees it go

  def _serve(self):
    while True:
      conn, _ = self.srv.accept()
      conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
      with self.cv:
        old, self.sock = self.sock, conn
      if old is not None:
        finish(threading.Event(), old)  # a new brain replaces the last
        old.close()
      self.game.brain = True
      log("the brain connected")
      try:
        for line in conn.makefile("rb"):
          self.game.from_brain(line)
      except OSError:
        pass
      with self.cv:
        if self.sock is conn:
          self.sock = None
          self.game.brain, self.game.owner = False, "link"
          log("the brain went: the car's controls are the link's")
      conn.close()


class Bridge:
  """The relay's connection to the bridge, made again whenever it goes: its lines go to the game, or wait for the
  plugin (the last WAITING_MAX bytes of them), and the plugin's frames go to it."""
  def __init__(self, game: Game, link_file: str):
    self.game = game
    self.link_file = link_file
    self.lock = threading.Lock()
    self.sock: socket.socket | None = None
    self.gen = 0  # counts connections, for the frames to start each one at a message's start
    self.waiting: deque[bytes] = deque()
    self.waiting_bytes = 0
    self.playing = False  # the plugin is connected: the lines go straight to it
    threading.Thread(target=self._run, daemon=True).start()

  @property
  def up(self) -> bool:
    return self.sock is not None

  def current(self) -> tuple[socket.socket, int] | None:
    with self.lock:
      return (self.sock, self.gen) if self.sock is not None else None

  def drop(self, sock: socket.socket | None = None):
    """Ends the connection (that one, if given); the bridge sees the game go, and the relay connects again."""
    with self.lock:
      sock = sock or self.sock
    if sock is not None:
      finish(threading.Event(), sock)

  def game_on(self, on: bool):
    """The plugin came (what the bridge sent meanwhile goes to it first) or went."""
    with self.lock:
      self.playing = on
      lines, self.waiting, self.waiting_bytes = self.waiting, deque(), 0
      if on:
        for line in lines:
          self.game.from_link(line)

  def _connect(self) -> socket.socket:
    last = None
    while True:
      addr = bridge_address(self.link_file)
      if addr is not None:
        try:
          sock = socket.create_connection(addr, timeout=CONNECT_TIMEOUT)
          sock.settimeout(None)
          tune(sock)
          log(f"connected to the bridge at {addr[0]}:{addr[1]}")
          return sock
        except OSError as e:
          why = f"{addr[0]}:{addr[1]}: {e}"
      else:
        why = f"no bridge address in {self.link_file}"
      if why != last:
        log(f"waiting for the bridge ({why})")
        last = why
      time.sleep(RETRY_EVERY)

  def _run(self):
    while True:
      sock = self._connect()
      with self.lock:
        self.sock, self.gen, self.waiting, self.waiting_bytes = sock, self.gen + 1, deque(), 0
      try:
        for line in sock.makefile("rb"):
          with self.lock:
            playing = self.playing
            if not playing:
              self.waiting.append(line)
              self.waiting_bytes += len(line)
              while self.waiting_bytes > WAITING_MAX:
                self.waiting_bytes -= len(self.waiting.popleft())
          if playing:
            self.game.from_link(line)
      except OSError:  # reset: a bridge shutting down as the relay connected
        pass
      with self.lock:
        self.sock = None
      sock.close()
      log("the bridge went")


def pipe_up(game: socket.socket, bridge: Bridge, done: threading.Event, frames: Frames):
  """The plugin's frames to the bridge, as they come; dropped while there's none, and to a new one from the next
  message's start."""
  buf = bytearray(CHUNK)
  view = memoryview(buf)
  gen = None  # the bridge connection the frames go to
  try:
    while not done.is_set():
      n = game.recv_into(buf)
      if n == 0:
        break
      starts = frames.feed(view[:n], time.monotonic())
      to = bridge.current()
      if to is None:
        gen = None
        continue
      sock, g = to
      start = 0
      if g != gen:
        if not starts:
          continue
        start, gen = starts[0], g
      try:
        sock.sendall(view[start:n])
      except OSError:
        bridge.drop(sock)
        gen = None
  except OSError:
    pass
  finally:
    finish(done, game)


def finish(done: threading.Event, *socks: socket.socket):
  done.set()
  for s in socks:
    try:
      s.shutdown(socket.SHUT_RDWR)
    except OSError:
      pass


def wanted(game: Game, bridge: Bridge) -> bool:
  """Whether there's a bridge or a brain to take the game."""
  return bridge.up or game.brain


def wait_for_plugin(port: int, game: Game, bridge: Bridge) -> socket.socket | None:
  """The plugin's connection; None once there's neither a bridge nor a brain to take it."""
  srv = socket.create_server(("127.0.0.1", port))
  try:
    while wanted(game, bridge):
      ready, _, _ = select.select([srv], [], [], CHECK_EVERY)
      if ready:
        conn, addr = srv.accept()
        tune(conn)
        log(f"game connected from {addr[0]}")
        return conn
    return None
  finally:
    srv.close()  # one game at a time; it's open again once this one leaves


def play(sock: socket.socket, game: Game, brain: Brain, bridge: Bridge):
  """The plugin's connection, until it leaves or there's neither a bridge nor a brain."""
  done, frames = threading.Event(), Frames(brain.on_header)
  with game.lock:
    game.sock = sock
  bridge.game_on(True)
  try:
    up = threading.Thread(target=pipe_up, args=(sock, bridge, done, frames), daemon=True)
    stats = threading.Thread(target=stats_loop, args=(frames, done), daemon=True)
    up.start()
    stats.start()
    while up.is_alive():
      up.join(CHECK_EVERY)
      if not wanted(game, bridge) and not done.is_set():
        log("no bridge and no brain: letting the game go")
        finish(done, sock)
  finally:
    done.set()
    bridge.game_on(False)
    with game.lock:
      game.sock = None
    sock.close()
    bridge.drop()


def stats_loop(frames: Frames, done: threading.Event):
  while not done.wait(STATS_EVERY):
    n, b, gap = frames.take()
    log(f"{n} frames, {b / STATS_EVERY / 1e6:.1f} MB/s up, longest gap {gap * 1000:.0f} ms")


def main(argv: list[str] | None = None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--port", type=int, default=PORT)
  ap.add_argument("--brain-port", type=int, default=BRAIN_PORT)
  ap.add_argument("--plugin-file", default=PLUGIN_FILE)
  ap.add_argument("--link-file", default=LINK_FILE)
  args = ap.parse_args(argv)
  with open(args.plugin_file, "w") as f:
    f.write(f"127.0.0.1:{args.port}\n")
  log(f"the plugin connects to 127.0.0.1:{args.port} ({args.plugin_file}); the bridge's address from {args.link_file}")
  log(f"the brain connects to 127.0.0.1:{args.brain_port}")
  game = Game()
  brain = Brain(game, args.brain_port)
  bridge = Bridge(game, args.link_file)
  while True:
    if not wanted(game, bridge):
      time.sleep(CHECK_EVERY)
      continue
    try:
      sock = wait_for_plugin(args.port, game, bridge)
    except OSError as e:  # the relay must outlive anything one connection does
      log(f"waiting for the game: {e}")
      time.sleep(RETRY_EVERY)
      continue
    if sock is None:
      continue
    try:
      play(sock, game, brain, bridge)
    except OSError as e:
      log(f"pipe: {e}")
    log("the game left; starting over")


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    sys.exit(0)
