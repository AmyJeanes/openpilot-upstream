"""gta5_relay.py between a stand-in plugin and bridge on localhost."""
import socket
import struct
import threading
import time

from openpilot.tools.sim.bridge.gta5 import gta5_relay


def free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def recv_n(sock: socket.socket, n: int) -> bytes:
  out = bytearray()
  while len(out) < n:
    chunk = sock.recv(n - len(out))
    assert chunk, "closed early"
    out += chunk
  return bytes(out)


def plugin_connect(port: int, timeout: float = 5.0) -> socket.socket:
  end = time.monotonic() + timeout
  while True:
    try:
      return socket.create_connection(("127.0.0.1", port), timeout=1.0)
    except OSError:
      assert time.monotonic() < end, "the relay never listened"
      time.sleep(0.05)


def test_relay(tmp_path, monkeypatch):
  """The relay listens only once the bridge is up; what the bridge sent before the plugin came reaches it first; frames
  go up and commands down unchanged; the plugin leaving ends the bridge's connection, and the relay starts over."""
  monkeypatch.setattr(gta5_relay, "RETRY_EVERY", 0.05)
  relay_port = free_port()
  plugin_file, link_file = tmp_path / "plugin.txt", tmp_path / "link.txt"
  threading.Thread(target=gta5_relay.main, args=(["--port", str(relay_port), "--plugin-file", str(plugin_file),
                                                  "--link-file", str(link_file), "--brain-port", str(free_port())],), daemon=True).start()
  time.sleep(0.3)
  assert plugin_file.read_text().strip() == f"127.0.0.1:{relay_port}"
  try:
    socket.create_connection(("127.0.0.1", relay_port), timeout=0.5).close()
    raise AssertionError("listening without a bridge")
  except OSError:
    pass

  bridge_srv = socket.create_server(("127.0.0.1", 0))
  link_file.write_text(f"127.0.0.1:{bridge_srv.getsockname()[1]}\n")
  bridge, _ = bridge_srv.accept()
  bridge.sendall(b"settings\n")  # as rx sends the overlay's settings on each connection
  game = plugin_connect(relay_port)
  assert recv_n(game, 9) == b"settings\n"
  frames = [bytes([i]) * (3 << 20) for i in range(3)]
  for f in frames:
    game.sendall(struct.pack("<I", len(f)) + f)
  for f in frames:
    assert struct.unpack("<I", recv_n(bridge, 4))[0] == len(f) and recv_n(bridge, len(f)) == f
  bridge.sendall(b"command\n")
  assert recv_n(game, 8) == b"command\n"

  game.close()
  bridge.settimeout(5.0)
  assert bridge.recv(1) == b""  # the game left: so did the relay's connection to the bridge
  bridge.close()
  again, _ = bridge_srv.accept()  # and it's back, waiting for the game
  game2 = plugin_connect(relay_port)
  game2.sendall(struct.pack("<I", 1) + b"x")
  assert recv_n(again, 5) == struct.pack("<I", 1) + b"x"
  game2.close()
  again.close()
  bridge_srv.close()


def test_relay_bridge_reset(tmp_path, monkeypatch):
  """A bridge that resets the relay's connection before the game comes (one shutting down as the relay connected) has
  the relay start over, not stop."""
  monkeypatch.setattr(gta5_relay, "RETRY_EVERY", 0.05)
  relay_port = free_port()
  bridge_srv = socket.create_server(("127.0.0.1", 0))
  link_file = tmp_path / "link.txt"
  link_file.write_text(f"127.0.0.1:{bridge_srv.getsockname()[1]}\n")
  threading.Thread(target=gta5_relay.main, args=(["--port", str(relay_port), "--plugin-file", str(tmp_path / "plugin.txt"),
                                                  "--link-file", str(link_file), "--brain-port", str(free_port())],), daemon=True).start()
  first, _ = bridge_srv.accept()
  first.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
  first.close()  # a reset, not a clean close
  second, _ = bridge_srv.accept()
  game = plugin_connect(relay_port)
  second.sendall(b"hi\n")  # the bridge's commands are JSON lines
  assert recv_n(game, 3) == b"hi\n"
  game.close()
  second.close()
  bridge_srv.close()


def read_lines(sock: socket.socket, until: bytes) -> list[bytes]:
  """Lines from sock up to and including the one holding `until`."""
  out, buf = [], b""
  while True:
    chunk = sock.recv(65536)
    assert chunk, "closed early"
    buf += chunk
    while b"\n" in buf:
      line, buf = buf.split(b"\n", 1)
      out.append(line)
      if until in line:
        return out


def test_relay_brain(tmp_path, monkeypatch):
  """The brain gets each frame's header; the car takes the bridge's controls until the brain says they're its own, then
  the brain's; with a brain the bridge's HUD goes; the brain going gives the controls back to the bridge."""
  monkeypatch.setattr(gta5_relay, "RETRY_EVERY", 0.05)
  relay_port, brain_port = free_port(), free_port()
  bridge_srv = socket.create_server(("127.0.0.1", 0))
  link_file = tmp_path / "link.txt"
  link_file.write_text(f"127.0.0.1:{bridge_srv.getsockname()[1]}\n")
  threading.Thread(target=gta5_relay.main, args=(["--port", str(relay_port), "--brain-port", str(brain_port), "--plugin-file",
                                                  str(tmp_path / "plugin.txt"), "--link-file", str(link_file)],), daemon=True).start()
  brain = plugin_connect(brain_port)
  bridge, _ = bridge_srv.accept()
  game = plugin_connect(relay_port)
  header = b'{"views": ["road"], "state": {"t": 1.5, "inVehicle": true}}'
  body = struct.pack("<I", len(header)) + header + b"\0" * (1 << 20)
  game.sendall(struct.pack("<I", len(body)) + body)
  assert read_lines(brain, b"inVehicle") == [header]
  assert recv_n(bridge, 4 + len(body))[4 + 4:4 + 4 + len(header)] == header

  bridge.sendall(b'{"type": "control", "active": true, "curvature": 0.2}\n{"type": "hud", "setSpeed": 30}\n{"type": "mark1"}\n')
  got = read_lines(game, b"mark1")
  assert got == [b'{"type": "control", "active": true, "curvature": 0.2}', b'{"type": "mark1"}']  # a brain: no bridge HUD
  brain.sendall(b'{"type": "owner", "who": "brain"}\n{"type": "control", "active": true, "curvature": 0.1}\n')
  time.sleep(0.2)
  bridge.sendall(b'{"type": "control", "active": false}\n{"type": "mark2"}\n')
  got = read_lines(game, b"mark2")
  assert got == [b'{"type": "control", "active": true, "curvature": 0.1}', b'{"type": "mark2"}']

  brain.close()
  time.sleep(0.3)
  bridge.sendall(b'{"type": "control", "active": false}\n{"type": "hud", "setSpeed": 30}\n{"type": "mark3"}\n')
  got = read_lines(game, b"mark3")
  assert got == [b'{"type": "control", "active": false}', b'{"type": "hud", "setSpeed": 30}', b'{"type": "mark3"}']
  game.close()
  bridge.close()
  bridge_srv.close()


def test_frames_count():
  """Message counting and each message's header, across chunks split anywhere, the size bytes included."""
  heads = []
  f = gta5_relay.Frames(heads.append)
  msgs = [struct.pack("<I", 2) + b"{}" + b"frame", struct.pack("<I", 0), struct.pack("<I", 4) + b"head"]
  data = b"".join(struct.pack("<I", len(m)) + m for m in msgs)
  for i in range(0, len(data), 3):
    f.feed(memoryview(data[i:i + 3]), float(i))
  assert f.take()[:2] == (3, len(data)) and heads == [b"{}", b"", b"head"]
