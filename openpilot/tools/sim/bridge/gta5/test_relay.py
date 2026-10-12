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
                                                  "--link-file", str(link_file)],), daemon=True).start()
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
                                                  "--link-file", str(link_file)],), daemon=True).start()
  first, _ = bridge_srv.accept()
  first.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
  first.close()  # a reset, not a clean close
  second, _ = bridge_srv.accept()
  game = plugin_connect(relay_port)
  second.sendall(b"hi")
  assert recv_n(game, 2) == b"hi"
  game.close()
  second.close()
  bridge_srv.close()


def test_frames_count():
  """Message counting across chunks split anywhere, the size bytes included."""
  f = gta5_relay.Frames()
  data = b"".join(struct.pack("<I", n) + b"a" * n for n in (5, 0, 7))
  for i in range(0, len(data), 3):
    f.feed(memoryview(data[i:i + 3]), float(i))
  assert f.take()[:2] == (3, len(data))
