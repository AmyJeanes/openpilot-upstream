#!/usr/bin/env python3
"""The game's connection on Windows: the plugin connects to this relay on localhost, and the relay pipes it to the
bridge in WSL, so the game never needs WSL's address. Standard library only: it runs on Windows' own Python.

  py gta5_relay.py [--port 8781] [--plugin-file C:\\Users\\Public\\gta5op-bridge.txt] [--link-file C:\\Users\\Public\\gta5op-link.txt]

It writes its own address to the plugin's bridge file; the bridge must then write its address elsewhere, so start it
with GTA5_BRIDGE_FILE=/mnt/c/Users/Public/gta5op-link.txt. A bridge started without that writes its address over the
relay's and the plugin connects to it directly again.

The relay connects to the bridge first and only then listens for the plugin, as the plugin switches to openpilot's
camera as soon as it connects: with no bridge, the plugin's connects are refused and the game stays as it is. The
bridge takes the relay's connection for the game's, so it reports the game connected while the relay waits for it;
what the bridge sends meanwhile (the debug overlay's settings) waits for the plugin. A plugin that leaves (the game
closed, its core reloaded) ends the bridge's connection too, so the bridge sees the game go and come back as before.
Every STATS_EVERY s it logs the frames piped up, MB/s and the longest gap between frames."""
import argparse
import select
import socket
import struct
import sys
import threading
import time

PORT = 8781
PLUGIN_FILE = r"C:\Users\Public\gta5op-bridge.txt"
LINK_FILE = r"C:\Users\Public\gta5op-link.txt"
CONNECT_TIMEOUT = 1.0  # s
RETRY_EVERY = 1.0  # s
CHUNK = 4 << 20
SOCK_BUF = 16 << 20
WAITING_MAX = 1 << 20  # bytes from the bridge kept for a plugin yet to connect
STATS_EVERY = 10.0  # s


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
  """Counts the plugin's messages (each a 4-byte little-endian size, then that many bytes) as they're piped."""
  def __init__(self):
    self.need, self.head = 0, b""  # bytes left of the message, else the size bytes read so far
    self.count = self.bytes = 0
    self.last: float | None = None
    self.gap = 0.0

  def feed(self, data: memoryview, now: float):
    self.bytes += len(data)
    i = 0
    while i < len(data):
      if self.need:
        step = min(self.need, len(data) - i)
        self.need -= step
        i += step
        continue
      take = min(4 - len(self.head), len(data) - i)
      self.head += bytes(data[i:i + take])
      i += take
      if len(self.head) == 4:
        self.need, self.head = struct.unpack("<I", self.head)[0], b""
        self.count += 1
        if self.last is not None:
          self.gap = max(self.gap, now - self.last)
        self.last = now

  def take(self) -> tuple[int, int, float]:
    out = (self.count, self.bytes, self.gap)
    self.count = self.bytes = 0
    self.gap = 0.0
    return out


def pipe(src: socket.socket, dst: socket.socket, done: threading.Event, frames: Frames | None = None):
  buf = bytearray(CHUNK)
  view = memoryview(buf)
  try:
    while not done.is_set():
      n = src.recv_into(buf)
      if n == 0:
        break
      if frames is not None:
        frames.feed(view[:n], time.monotonic())
      dst.sendall(view[:n])
  except OSError:
    pass
  finally:
    done.set()
    for s in (src, dst):
      try:
        s.shutdown(socket.SHUT_RDWR)
      except OSError:
        pass


def connect_bridge(link_file: str) -> socket.socket:
  last = None
  while True:
    addr = bridge_address(link_file)
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
      why = f"no bridge address in {link_file}"
    if why != last:
      log(f"waiting for the bridge ({why})")
      last = why
    time.sleep(RETRY_EVERY)


def wait_for_plugin(bridge: socket.socket, port: int) -> tuple[socket.socket | None, bytes]:
  """The plugin's connection, and what the bridge sent before it came; None if the bridge went first."""
  srv = socket.create_server(("127.0.0.1", port))
  waiting = bytearray()
  try:
    while True:
      ready, _, _ = select.select([srv, bridge], [], [], 1.0)
      if bridge in ready:
        data = bridge.recv(65536)
        if not data:
          log("the bridge closed before the game connected")
          return None, b""
        waiting += data
        del waiting[:max(0, len(waiting) - WAITING_MAX)]
      if srv in ready:
        conn, addr = srv.accept()
        tune(conn)
        log(f"game connected from {addr[0]}")
        return conn, bytes(waiting)
  finally:
    srv.close()  # one game at a time; it's open again once this one leaves


def stats_loop(frames: Frames, done: threading.Event):
  while not done.wait(STATS_EVERY):
    n, b, gap = frames.take()
    log(f"{n} frames, {b / STATS_EVERY / 1e6:.1f} MB/s up, longest gap {gap * 1000:.0f} ms")


def main(argv: list[str] | None = None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--port", type=int, default=PORT)
  ap.add_argument("--plugin-file", default=PLUGIN_FILE)
  ap.add_argument("--link-file", default=LINK_FILE)
  args = ap.parse_args(argv)
  with open(args.plugin_file, "w") as f:
    f.write(f"127.0.0.1:{args.port}\n")
  log(f"the plugin connects to 127.0.0.1:{args.port} ({args.plugin_file}); the bridge's address from {args.link_file}")
  while True:
    bridge = connect_bridge(args.link_file)
    game, waiting = wait_for_plugin(bridge, args.port)
    if game is None:
      bridge.close()
      continue
    done, frames = threading.Event(), Frames()
    try:
      if waiting:
        game.sendall(waiting)
      up = threading.Thread(target=pipe, args=(game, bridge, done, frames), daemon=True)
      down = threading.Thread(target=pipe, args=(bridge, game, done), daemon=True)
      stats = threading.Thread(target=stats_loop, args=(frames, done), daemon=True)
      for t in (up, down, stats):
        t.start()
      up.join()
      down.join()
    except OSError as e:
      log(f"pipe: {e}")
    finally:
      done.set()
      game.close()
      bridge.close()
    log("the game or the bridge left; starting over")


if __name__ == "__main__":
  try:
    main()
  except KeyboardInterrupt:
    sys.exit(0)
