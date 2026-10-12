#!/usr/bin/env python3
"""The game's connection on Windows: the plugin connects to this relay on localhost, and the relay pipes it to the
bridge in WSL, so the game never needs WSL's address. Standard library only: it runs on Windows' own Python.

  py gta5_relay.py [--port 8781] [--brain-port 8784] [--frames-port 8785] [--plugin-file C:\\Users\\Public\\gta5op-bridge.txt]
                   [--link-file C:\\Users\\Public\\gta5op-link.txt]

It writes its own address to the plugin's bridge file; the bridge must then write its address elsewhere, so start it
with GTA5_BRIDGE_FILE=/mnt/c/Users/Public/gta5op-link.txt. A bridge started without that writes its address over the
relay's and the plugin connects to it directly again.

The relay keeps a connection to the bridge whenever it can, and listens for the plugin only while there's a bridge or a
brain to take the game: with neither, the plugin's connects are refused and the game stays as it is. The plugin renders
and sends openpilot's camera only while something takes its frames, a bridge or the frames port's recorder: the relay
tells it {"type": "frames", "on": 1|0} as it connects and whenever that changes, off only once nothing has taken them for
FRAMES_HOLD s, so a bridge or recorder restarting doesn't switch the game's camera back and forth. With frames off the
plugin keeps the game's own camera and sends each state alone, a header without frames, which goes only to the brain. A
plugin connected straight to a bridge gets no such command and sends frames as before. The bridge takes the relay's
connection for the game's, so it reports
the game connected while the relay waits for it; what the bridge sends meanwhile (the debug overlay's settings) waits
for the plugin. A plugin that leaves (the game closed, its core reloaded) ends the bridge's connection too, so the bridge
sees the game go and come back as before.

The brain (gta5_host.py, the bridge's brain on Windows, the bridge in WSL then being the link) connects on the brain
port: it gets each frame's header (the game state) as a JSON line, the newest only if it falls behind, and its lines go
to the plugin. Who drives decides whose controls the car takes: the brain says {"type": "owner", "who": "brain"} while
its own driver drives, and the bridge's control messages (openpilot's) are dropped until it says "link" again, or goes.
While a brain is connected the bridge's HUD messages are dropped too: the brain sends the HUD.

The game and the brain don't need the bridge: with a brain connected the plugin stays while the bridge is gone (the
link restarting, or a host run without openpilot), sending its state alone once the hold is over, and a bridge that
comes back gets the frames from the next message's start. With no brain, the bridge going ends the plugin's connection,
as without a relay.

Each frame's header goes up with the relay's number for the frame first ("hostSeq", counting on across connections), by
which the link's model tap and the host's recorder find the same frame; a state alone gets none. The host's recorder
(gta5_record.FrameWorker) connects on the frames port and gets the plugin's frames as the bridge does, bridge or not, the
newest only if it falls behind (FrameTap). Every STATS_EVERY s it logs the frames piped up, MB/s, the longest gap
between frames and the states sent alone."""
import argparse
import itertools
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
FRAMES_PORT = 8785
PLUGIN_FILE = r"C:\Users\Public\gta5op-bridge.txt"
LINK_FILE = r"C:\Users\Public\gta5op-link.txt"
CONNECT_TIMEOUT = 1.0  # s
RETRY_EVERY = 1.0  # s
CHUNK = 4 << 20
SOCK_BUF = 16 << 20
WAITING_MAX = 1 << 20  # bytes from the bridge kept for a plugin yet to connect
STATS_EVERY = 10.0  # s
CHECK_EVERY = 0.2  # s between looks at whether the game still has a bridge or a brain
TAP_BUFFERS = 3  # the frames port's message buffers kept for reuse: one filling, one waiting, one sending
FRAMES_HOLD = 2.0  # s the plugin's frames stay on after the last bridge or recorder goes


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


def stamp(header: bytes, seq: int) -> bytes:
  """A frame's header JSON with the relay's number for the frame first; one that isn't a JSON object goes unchanged."""
  if header[:1] != b"{":
    return header
  rest = header[1:].lstrip()
  return b'{"hostSeq":%d%s' % (seq, b"" if rest[:1] == b"}" else b",") + rest


def recv_exact(sock: socket.socket, n: int) -> bytes | None:
  """n bytes, None if the connection ends first."""
  buf = bytearray(n)
  view = memoryview(buf)
  while len(view):
    k = sock.recv_into(view)
    if k == 0:
      return None
    view = view[k:]
  return bytes(buf)


class Frames:
  """The plugin's messages as they're piped: counted, with their bytes and the longest gap between them, and the states
  it sent alone."""
  def __init__(self):
    self.count = self.bytes = self.states = 0
    self.last: float | None = None
    self.gap = 0.0

  def add(self, size: int, now: float):
    self.count += 1
    self.bytes += size
    if self.last is not None:
      self.gap = max(self.gap, now - self.last)
    self.last = now

  def take(self) -> tuple[int, int, float, int]:
    out = (self.count, self.bytes, self.gap, self.states)
    self.count = self.bytes = self.states = 0
    self.gap = 0.0
    return out


def frames_line(on: bool) -> bytes:
  return b'{"type": "frames", "on": %d}\n' % on


class FramesSwitch:
  """Whether the plugin's frames are wanted: while a bridge or a recorder takes them, and FRAMES_HOLD s after."""
  def __init__(self):
    self.until = float("-inf")

  def on(self, taken: bool, now: float) -> bool:
    if taken:
      self.until = now + FRAMES_HOLD
    return now < self.until


class Game:
  """The plugin's connection as the relay writes to it, from the bridge's pipe and the brain's alike, a line at a time,
  with who the car's controls belong to."""
  def __init__(self):
    self.lock = threading.Lock()
    self.sock: socket.socket | None = None
    self.owner = "link"  # whose control messages the car takes: the bridge's (openpilot's), or the brain's
    self.brain = False  # a brain is connected: the HUD is its

  def attach(self, sock: socket.socket | None, first: bytes = b""):
    """The plugin's connection, or None once it's gone; first goes to it ahead of anything else."""
    with self.lock:
      self.sock = sock
      if sock is not None and first:
        try:
          sock.sendall(first)
        except OSError:
          pass  # the pipe sees it go

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


class Copy:
  """One message as it's piped, for the frames port."""
  __slots__ = ("buf", "size", "n")

  def __init__(self, buf: bytearray, size: int):
    self.buf, self.size, self.n = buf, size, 0

  def write(self, data):
    k = len(data)
    self.buf[self.n:self.n + k] = data
    self.n += k


class FrameTap:
  """The frames port, for the host's recorder (gta5_record.FrameWorker): the plugin's messages as the bridge gets them.
  Each is copied as it's piped and handed over whole by the tap's own thread, the newest only if the recorder falls
  behind, so it never holds up the bridge's stream; nothing is copied while no recorder is connected. One recorder at a
  time, a new one replacing the last."""
  def __init__(self, port: int):
    self.cv = threading.Condition()
    self.sock: socket.socket | None = None
    self.ready: Copy | None = None  # the newest whole message, waiting to go
    self.free: list[bytearray] = []
    self.srv = socket.create_server(("127.0.0.1", port))
    threading.Thread(target=self._serve, daemon=True).start()
    threading.Thread(target=self._sender, daemon=True).start()

  @property
  def taken(self) -> bool:
    return self.sock is not None

  def begin(self, size: int) -> Copy | None:
    with self.cv:
      if self.sock is None:
        return None
      buf = next((b for b in self.free if len(b) >= size), None)
      if buf is not None:
        self.free.remove(buf)
    return Copy(buf if buf is not None else bytearray(size), size)

  def end(self, copy: Copy):
    with self.cv:
      if copy.n != copy.size or self.sock is None:
        self._reuse(copy.buf)
        return
      if self.ready is not None:
        self._reuse(self.ready.buf)
      self.ready = copy
      self.cv.notify()

  def _reuse(self, buf: bytearray):
    if len(self.free) < TAP_BUFFERS:
      self.free.append(buf)

  def _sender(self):
    while True:
      with self.cv:
        while self.ready is None or self.sock is None:
          self.cv.wait()
        copy, sock, self.ready = self.ready, self.sock, None
      try:
        sock.sendall(memoryview(copy.buf)[:copy.n])
      except OSError:
        pass  # its reader sees it go
      with self.cv:
        self._reuse(copy.buf)

  def _serve(self):
    while True:
      conn, _ = self.srv.accept()
      tune(conn)
      with self.cv:
        old, self.sock = self.sock, conn
      if old is not None:
        finish(threading.Event(), old)
      log("the recorder connected for the frames")
      try:
        while conn.recv(4096):  # it sends nothing: this only waits for it to go
          pass
      except OSError:
        pass
      with self.cv:
        if self.sock is conn:
          self.sock, self.ready = None, None
          log("the recorder went")
      conn.close()


class Bridge:
  """The relay's connection to the bridge, made again whenever it goes: its lines go to the game, or wait for the
  plugin (the last WAITING_MAX bytes of them), and the plugin's frames go to it."""
  def __init__(self, game: Game, link_file: str):
    self.game = game
    self.link_file = link_file
    self.lock = threading.Lock()
    self.sock: socket.socket | None = None
    self.waiting: deque[bytes] = deque()
    self.waiting_bytes = 0
    self.playing = False  # the plugin is connected: the lines go straight to it
    threading.Thread(target=self._run, daemon=True).start()

  @property
  def up(self) -> bool:
    return self.sock is not None

  def current(self) -> socket.socket | None:
    with self.lock:
      return self.sock

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
        self.sock, self.waiting, self.waiting_bytes = sock, deque(), 0
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


def pipe_up(game: socket.socket, bridge: Bridge, done: threading.Event, frames: Frames, on_header=None,
            tap: FrameTap | None = None, seqs=None):
  """The plugin's frames to the bridge: each message's header stamped with the next of seqs (stamp), the rest as it
  comes. A message goes whole to the bridge connection there was at its start, so a new one gets them from the next
  message's start, and none while there's none. Each header goes to on_header, and the messages to the frames port's
  tap, bridge or not. A message without a frame's header goes unchanged; a state alone (a header with nothing after it)
  goes only to on_header, unstamped."""
  buf = bytearray(CHUNK)
  view = memoryview(buf)
  seqs = seqs if seqs is not None else itertools.count()

  def send(sock: socket.socket | None, data) -> socket.socket | None:
    if sock is None:
      return None
    try:
      sock.sendall(data)
      return sock
    except OSError:
      bridge.drop(sock)
      return None

  try:
    while not done.is_set():
      size = recv_exact(game, 4)
      if size is None:
        break
      rest = struct.unpack("<I", size)[0]
      head = size
      if rest >= 4:
        head_len = recv_exact(game, 4)
        if head_len is None:
          break
        n = struct.unpack("<I", head_len)[0]
        rest -= 4
        head += head_len
        if n <= rest:
          header = recv_exact(game, n)
          if header is None:
            break
          rest -= n
          if not rest and header[:1] == b"{":
            frames.states += 1
            if on_header is not None:
              on_header(header)
            continue
          header = stamp(header, next(seqs))
          head = struct.pack("<II", 4 + len(header) + rest, len(header)) + header
          if on_header is not None:
            on_header(header)
      to = send(bridge.current(), head)
      frames.add(len(head) + rest, time.monotonic())
      copy = tap.begin(len(head) + rest) if tap is not None else None
      if copy is not None:
        copy.write(head)
      while rest:
        k = game.recv_into(view[:min(CHUNK, rest)])
        if k == 0:
          return
        to = send(to, view[:k])
        if copy is not None:
          copy.write(view[:k])
        rest -= k
      if copy is not None:
        tap.end(copy)
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


def frames_taker(bridge: Bridge, tap: FrameTap | None) -> str | None:
  """What takes the plugin's frames: the bridge, the recorder, or nothing (None)."""
  return "a bridge" if bridge.up else "the recorder" if tap is not None and tap.taken else None


def play(sock: socket.socket, game: Game, brain: Brain, bridge: Bridge, tap: FrameTap | None = None, seqs=None,
         switch: FramesSwitch | None = None):
  """The plugin's connection, until it leaves or there's neither a bridge nor a brain; its frames on while switch says."""
  done, frames = threading.Event(), Frames()
  switch = switch if switch is not None else FramesSwitch()

  def frames_on() -> tuple[bool, str]:
    taker = frames_taker(bridge, tap)
    on = switch.on(taker is not None, time.monotonic())
    return on, (f"{taker} takes them" if taker else "nothing takes them, held a moment" if on else
                "nothing takes them: the state goes alone")

  on, why = frames_on()
  game.attach(sock, frames_line(on))
  log(f"frames {'on' if on else 'off'}: {why}")
  bridge.game_on(True)
  try:
    up = threading.Thread(target=pipe_up, args=(sock, bridge, done, frames, brain.on_header, tap, seqs), daemon=True)
    stats = threading.Thread(target=stats_loop, args=(frames, done), daemon=True)
    up.start()
    stats.start()
    while up.is_alive():
      up.join(CHECK_EVERY)
      if not wanted(game, bridge) and not done.is_set():
        log("no bridge and no brain: letting the game go")
        finish(done, sock)
      now_on, why = frames_on()
      if now_on != on:
        on = now_on
        game.send(frames_line(on))
        log(f"frames {'on' if on else 'off'}: {why}")
  finally:
    done.set()
    bridge.game_on(False)
    game.attach(None)
    sock.close()
    bridge.drop()


def stats_loop(frames: Frames, done: threading.Event):
  while not done.wait(STATS_EVERY):
    n, b, gap, states = frames.take()
    log(f"{n} frames, {b / STATS_EVERY / 1e6:.1f} MB/s up, longest gap {gap * 1000:.0f} ms" +
        (f", {states} states alone" if states else ""))


def main(argv: list[str] | None = None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--port", type=int, default=PORT)
  ap.add_argument("--brain-port", type=int, default=BRAIN_PORT)
  ap.add_argument("--frames-port", type=int, default=FRAMES_PORT)
  ap.add_argument("--plugin-file", default=PLUGIN_FILE)
  ap.add_argument("--link-file", default=LINK_FILE)
  args = ap.parse_args(argv)
  with open(args.plugin_file, "w") as f:
    f.write(f"127.0.0.1:{args.port}\n")
  log(f"the plugin connects to 127.0.0.1:{args.port} ({args.plugin_file}); the bridge's address from {args.link_file}")
  log(f"the brain connects to 127.0.0.1:{args.brain_port}, its recorder to 127.0.0.1:{args.frames_port}")
  game = Game()
  brain = Brain(game, args.brain_port)
  try:
    tap = FrameTap(args.frames_port)
  except OSError as e:  # the game and the link matter more than recording
    log(f"no frames port for a recorder: {e}")
    tap = None
  # numbered on from the wall clock's second, so a restarted relay's numbers don't meet its last ones'
  seqs = itertools.count(int(time.time()) * 1000)  # noqa: TID251
  bridge = Bridge(game, args.link_file)
  switch = FramesSwitch()
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
      play(sock, game, brain, bridge, tap, seqs, switch)
    except OSError as e:
      log(f"pipe: {e}")
    log("the game left; starting over")


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    sys.exit(0)
