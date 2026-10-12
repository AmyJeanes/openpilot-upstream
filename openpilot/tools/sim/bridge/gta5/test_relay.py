"""gta5_relay.py between a stand-in plugin and bridge on localhost."""
import json
import socket
import struct
import threading
import time

from openpilot.tools.sim.bridge.gta5 import gta5_relay

FRAMES_ON, FRAMES_OFF = b'{"type": "frames", "on": 1}', b'{"type": "frames", "on": 0}'


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
                                                  "--link-file", str(link_file), "--brain-port", str(free_port()),
                                                  "--frames-port", str(free_port())],), daemon=True).start()
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
  assert read_lines(game, b"settings") == [FRAMES_ON, b"settings"]  # a bridge takes the frames
  frames =[bytes([i]) * (3 << 20) for i in range(3)]
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
                                                  "--link-file", str(link_file), "--brain-port", str(free_port()),
                                                  "--frames-port", str(free_port())],), daemon=True).start()
  first, _ = bridge_srv.accept()
  first.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
  first.close()  # a reset, not a clean close
  second, _ = bridge_srv.accept()
  game = plugin_connect(relay_port)
  second.sendall(b"hi\n")  # the bridge's commands are JSON lines
  assert read_lines(game, b"hi") == [FRAMES_ON, b"hi"]
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
  threading.Thread(target=gta5_relay.main, args=(["--port", str(relay_port), "--brain-port", str(brain_port), "--frames-port",
                                                  str(free_port()), "--plugin-file", str(tmp_path / "plugin.txt"), "--link-file",
                                                  str(link_file)],), daemon=True).start()
  brain = plugin_connect(brain_port)
  bridge, _ = bridge_srv.accept()
  game = plugin_connect(relay_port)
  header = b'{"views": ["road"], "state": {"t": 1.5, "inVehicle": true}}'
  body = struct.pack("<I", len(header)) + header + b"\0" * (1 << 20)
  game.sendall(struct.pack("<I", len(body)) + body)
  stamped = read_lines(brain, b"inVehicle")
  assert len(stamped) == 1 and json.loads(stamped[0]) == {"hostSeq": json.loads(stamped[0])["hostSeq"], **json.loads(header)}
  up = recv_n(bridge, 4 + 4 + len(stamped[0]) + (1 << 20))
  assert up == struct.pack("<II", 4 + len(stamped[0]) + (1 << 20), len(stamped[0])) + stamped[0] + b"\0" * (1 << 20)

  bridge.sendall(b'{"type": "control", "active": true, "curvature": 0.2}\n{"type": "hud", "setSpeed": 30}\n{"type": "mark1"}\n')
  got = read_lines(game, b"mark1")
  assert got == [FRAMES_ON, b'{"type": "control", "active": true, "curvature": 0.2}', b'{"type": "mark1"}']  # a brain: no bridge HUD
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


def test_stamp():
  """The relay's number goes first in a header that's a JSON object; anything else goes unchanged."""
  assert json.loads(gta5_relay.stamp(b'{"views": [], "state": {}}', 7)) == {"hostSeq": 7, "views": [], "state": {}}
  assert gta5_relay.stamp(b"{}", 3) == b'{"hostSeq":3}' and gta5_relay.stamp(b"", 1) == b"" and gta5_relay.stamp(b"x", 1) == b"x"


class Stream:
  """A socket's stand-in for pipe_up's game side: recv_into hands out data in pieces of at most `piece` bytes."""
  def __init__(self, data: bytes, piece: int):
    self.data, self.piece = memoryview(data), piece

  def recv_into(self, view) -> int:
    n = min(len(view), self.piece, len(self.data))
    view[:n] = self.data[:n]
    self.data = self.data[n:]
    return n

  def shutdown(self, how):
    pass


class Sink:
  """A Bridge's stand-in for pipe_up, its own connection, which keeps what's sent."""
  def __init__(self):
    self.data = bytearray()

  def current(self):
    return self

  def drop(self, sock=None):
    pass

  def sendall(self, data):
    self.data += data


def test_pipe_up_pieces():
  """Messages across pieces split anywhere: headers stamped in order and handed on, the rest unchanged, and messages
  without a frame's header (too short, a header past the end) passed as they are."""
  msgs = [struct.pack("<I", 2) + b"{}" + b"frame", struct.pack("<I", 0), b"ab", struct.pack("<I", 99) + b"head",
          struct.pack("<I", 8) + b'{"a": 1}' + b"\1" * 5000]
  data = b"".join(struct.pack("<I", len(m)) + m for m in msgs)
  for piece in (1, 3, 4096):
    heads, bridge, frames = [], Sink(), gta5_relay.Frames()
    gta5_relay.pipe_up(Stream(data, piece), bridge, threading.Event(), frames, heads.append, seqs=iter(range(10, 20)))
    want = [b'{"hostSeq":10}', b"", b'{"hostSeq":12,"a": 1}']
    assert heads == want
    out = [struct.pack("<I", len(want[0])) + want[0] + b"frame", struct.pack("<I", 0), b"ab", struct.pack("<I", 99) + b"head",
           struct.pack("<I", len(want[2])) + want[2] + b"\1" * 5000]
    assert bytes(bridge.data) == b"".join(struct.pack("<I", len(m)) + m for m in out)
    assert frames.take()[0] == len(msgs)


def test_frames_port(tmp_path, monkeypatch):
  """The frames port gets the messages as the bridge does, whole; a recorder that doesn't read doesn't hold up the
  bridge, and gets the newest message once it reads again."""
  monkeypatch.setattr(gta5_relay, "RETRY_EVERY", 0.05)
  relay_port, frames_port = free_port(), free_port()
  bridge_srv = socket.create_server(("127.0.0.1", 0))
  link_file = tmp_path / "link.txt"
  link_file.write_text(f"127.0.0.1:{bridge_srv.getsockname()[1]}\n")
  threading.Thread(target=gta5_relay.main, args=(["--port", str(relay_port), "--brain-port", str(free_port()), "--frames-port",
                                                  str(frames_port), "--plugin-file", str(tmp_path / "plugin.txt"), "--link-file",
                                                  str(link_file)],), daemon=True).start()
  recorder = plugin_connect(frames_port)
  bridge, _ = bridge_srv.accept()
  game = plugin_connect(relay_port)
  time.sleep(0.2)  # the relay takes the recorder's connection
  size = 7 << 20
  count = 30

  def message(i: int) -> bytes:
    header = json.dumps({"views": ["road", "wide"], "state": {"i": i}}).encode()
    body = struct.pack("<I", len(header)) + header + bytes([i]) * size
    return struct.pack("<I", len(body)) + body

  threading.Thread(target=lambda: [game.sendall(message(i)) for i in range(count)], daemon=True).start()
  bridge.settimeout(20.0)
  got = []
  for _ in range(count):
    total = struct.unpack("<I", recv_n(bridge, 4))[0]
    msg = recv_n(bridge, total)
    head_len = struct.unpack("<I", msg[:4])[0]
    got.append((json.loads(msg[4:4 + head_len]), msg))
  seqs = [h["hostSeq"] for h, _ in got]
  assert [h["state"]["i"] for h, _ in got] == list(range(count)) and seqs == list(range(seqs[0], seqs[0] + count))
  # all of it reached the bridge while the recorder read nothing; now it reads what's waiting for it
  recorder.settimeout(5.0)
  seen = []
  while not seen or seen[-1] != seqs[-1]:
    total = struct.unpack("<I", recv_n(recorder, 4))[0]
    msg = recv_n(recorder, total)
    head = json.loads(msg[4:4 + struct.unpack("<I", msg[:4])[0]])
    assert msg == got[head["state"]["i"]][1]
    seen.append(head["hostSeq"])
  assert len(seen) < count and seen == sorted(seen)
  for s in (game, bridge, recorder, bridge_srv):
    s.close()


def frame_msg(t: float, size: int = 1 << 16) -> bytes:
  """A plugin message: its header with the game state, then stand-in frames."""
  header = b'{"views": ["road"], "state": {"t": %.2f, "inVehicle": true}}' % t
  body = struct.pack("<I", len(header)) + header + b"\0" * size
  return struct.pack("<I", len(body)) + body


def recv_msg(sock: socket.socket) -> tuple[dict, bytes]:
  """A message as the relay sends it on: its header, and what follows it."""
  msg = recv_n(sock, struct.unpack("<I", recv_n(sock, 4))[0])
  n = struct.unpack("<I", msg[:4])[0]
  return json.loads(msg[4:4 + n]), msg[4 + n:]


def test_relay_no_bridge(tmp_path, monkeypatch):
  """With no bridge the relay takes the game for a brain: the brain gets the state and the car its controls, the frames
  dropped but for the recorder's frames port. A bridge that comes gets them from the next message's start; one that goes
  leaves the game to the brain; the brain going too lets the game go."""
  monkeypatch.setattr(gta5_relay, "RETRY_EVERY", 0.05)
  relay_port, brain_port, frames_port = free_port(), free_port(), free_port()
  link_file = tmp_path / "link.txt"  # none yet: no bridge
  threading.Thread(target=gta5_relay.main, args=(["--port", str(relay_port), "--brain-port", str(brain_port), "--frames-port",
                                                  str(frames_port), "--plugin-file", str(tmp_path / "plugin.txt"), "--link-file",
                                                  str(link_file)],), daemon=True).start()
  time.sleep(0.3)
  try:
    socket.create_connection(("127.0.0.1", relay_port), timeout=0.5).close()
    raise AssertionError("listening with neither a bridge nor a brain")
  except OSError:
    pass
  brain = plugin_connect(brain_port)
  recorder = plugin_connect(frames_port)
  game = plugin_connect(relay_port)
  time.sleep(0.2)  # the relay takes the recorder's connection
  game.sendall(frame_msg(1.0))
  head = json.loads(read_lines(brain, b'"t": 1.00')[-1])
  assert head["views"] == ["road"] and "hostSeq" in head
  assert recv_msg(recorder) == (head, b"\0" * (1 << 16))  # with no bridge, the recorder still gets the frames
  brain.sendall(b'{"type": "owner", "who": "brain"}\n{"type": "control", "active": true, "curvature": 0.1}\n')
  assert read_lines(game, b"curvature") == [FRAMES_ON, b'{"type": "control", "active": true, "curvature": 0.1}']  # the recorder's

  half = frame_msg(2.0)
  game.sendall(half[:len(half) // 2])  # a message begun before the bridge comes
  bridge_srv = socket.create_server(("127.0.0.1", 0))
  link_file.write_text(f"127.0.0.1:{bridge_srv.getsockname()[1]}\n")
  bridge, _ = bridge_srv.accept()
  bridge.settimeout(5.0)
  bridge.sendall(b"settings\n")
  assert recv_n(game, 9) == b"settings\n"
  time.sleep(0.2)
  game.sendall(half[len(half) // 2:] + frame_msg(3.0))
  got, frames = recv_msg(bridge)  # from the next message on
  assert got["state"]["t"] == 3.0 and got["hostSeq"] == head["hostSeq"] + 2 and frames == b"\0" * (1 << 16)

  bridge.close()
  bridge_srv.close()
  time.sleep(0.3)
  game.sendall(frame_msg(4.0))
  assert read_lines(brain, b'"t": 4.00')
  brain.sendall(b'{"type": "mark"}\n')
  assert read_lines(game, b"mark") == [b'{"type": "mark"}']  # the game stayed

  brain.close()
  game.settimeout(5.0)
  assert game.recv(1) == b""  # neither a bridge nor a brain: the game goes
  game.close()
  recorder.close()


def test_frames_switch(monkeypatch):
  """Frames on at once while something takes them, off only once nothing has for FRAMES_HOLD s."""
  monkeypatch.setattr(gta5_relay, "FRAMES_HOLD", 2.0)
  s = gta5_relay.FramesSwitch()
  assert not s.on(False, 0.0)
  assert s.on(True, 1.0) and s.on(False, 2.9) and not s.on(False, 3.0)
  assert s.on(True, 4.0) and s.on(False, 5.0) and s.on(True, 5.5) and s.on(False, 7.4) and not s.on(False, 7.5)


def state_msg(t: float) -> bytes:
  """A plugin message with the state alone: a header and nothing after it."""
  header = b'{"type": "state", "t": %.2f, "state": {"t": %.2f, "inVehicle": true}}' % (t, t)
  body = struct.pack("<I", len(header)) + header
  return struct.pack("<I", len(body)) + body


def test_pipe_up_state_alone():
  """A state alone goes only to on_header, unstamped and taking no number; the frames around it go on as before."""
  state = state_msg(1.0)
  data = struct.pack("<I", 8) + struct.pack("<I", 2) + b"{}f1" + state + struct.pack("<I", 8) + struct.pack("<I", 2) + b"{}f2"
  for piece in (1, 4096):
    heads, bridge, frames = [], Sink(), gta5_relay.Frames()
    gta5_relay.pipe_up(Stream(data, piece), bridge, threading.Event(), frames, heads.append, seqs=iter(range(10, 20)))
    assert heads == [b'{"hostSeq":10}', state[8:], b'{"hostSeq":11}']
    out = [struct.pack("<I", len(h)) + h + f for h, f in ((heads[0], b"f1"), (heads[2], b"f2"))]
    assert bytes(bridge.data) == b"".join(struct.pack("<I", len(m)) + m for m in out)
    n, _, _, states = frames.take()
    assert (n, states) == (2, 1)


class Lines:
  """A socket's lines one at a time, what's come past the last kept for the next."""
  def __init__(self, sock: socket.socket):
    self.sock, self.buf = sock, b""

  def next(self, timeout: float = 5.0) -> bytes:
    self.sock.settimeout(timeout)
    while b"\n" not in self.buf:
      chunk = self.sock.recv(65536)
      assert chunk, "closed early"
      self.buf += chunk
    line, self.buf = self.buf.split(b"\n", 1)
    return line

  def none(self, wait: float):
    """Nothing comes for wait s."""
    try:
      raise AssertionError(f"got {self.next(wait)!r}")
    except TimeoutError:
      pass


def test_relay_frames_on_off(tmp_path, monkeypatch):
  """With only a brain the plugin's frames are off and its states alone reach the brain, and nothing else; a recorder or
  a bridge coming turns them on, and they go off FRAMES_HOLD s after the last goes, not while one comes straight back."""
  monkeypatch.setattr(gta5_relay, "RETRY_EVERY", 0.05)
  monkeypatch.setattr(gta5_relay, "FRAMES_HOLD", 0.6)
  relay_port, brain_port, frames_port = free_port(), free_port(), free_port()
  link_file = tmp_path / "link.txt"  # none yet: no bridge
  threading.Thread(target=gta5_relay.main, args=(["--port", str(relay_port), "--brain-port", str(brain_port), "--frames-port",
                                                  str(frames_port), "--plugin-file", str(tmp_path / "plugin.txt"), "--link-file",
                                                  str(link_file)],), daemon=True).start()
  brain = plugin_connect(brain_port)
  game = plugin_connect(relay_port)
  lines = Lines(game)
  assert lines.next() == FRAMES_OFF  # only a brain: map driving
  game.sendall(state_msg(1.0))
  head = json.loads(read_lines(brain, b'"t": 1.00')[-1])
  assert head == {"type": "state", "t": 1.0, "state": {"t": 1.0, "inVehicle": True}}  # no relay number

  recorder = plugin_connect(frames_port)
  assert lines.next() == FRAMES_ON
  game.sendall(state_msg(2.0) + frame_msg(3.0))
  assert read_lines(brain, b'"t": 3.00')  # the brain gets the newest header
  recorder.settimeout(5.0)
  got, frames = recv_msg(recorder)  # the frame, not the state before it
  assert got["state"]["t"] == 3.0 and "hostSeq" in got and frames == b"\0" * (1 << 16)

  recorder.close()
  t0 = time.monotonic()
  assert lines.next() == FRAMES_OFF and time.monotonic() - t0 > 0.5
  game.sendall(state_msg(4.0))
  assert read_lines(brain, b'"t": 4.00')  # the state still reaches the brain

  bridge_srv = socket.create_server(("127.0.0.1", 0))
  link_file.write_text(f"127.0.0.1:{bridge_srv.getsockname()[1]}\n")
  bridge, _ = bridge_srv.accept()
  bridge.settimeout(5.0)
  assert lines.next() == FRAMES_ON
  game.sendall(state_msg(5.0) + frame_msg(6.0))
  assert recv_msg(bridge)[0]["state"]["t"] == 6.0

  bridge.close()  # a bridge restarting: the relay connects again at once
  again, _ = bridge_srv.accept()
  lines.none(1.0)  # longer than the hold: no off and on again
  link_file.unlink()
  again.close()
  bridge_srv.close()
  t0 = time.monotonic()
  assert lines.next() == FRAMES_OFF and time.monotonic() - t0 > 0.5
  game.sendall(state_msg(7.0))
  assert read_lines(brain, b'"t": 7.00')

  brain.close()
  game.settimeout(5.0)
  assert game.recv(1) == b""  # neither a bridge nor a brain: the game goes
  game.close()


def test_relay_old_plugin(tmp_path, monkeypatch):
  """A plugin that doesn't know the frames command sends frames anyway: their headers still reach the brain."""
  monkeypatch.setattr(gta5_relay, "RETRY_EVERY", 0.05)
  relay_port, brain_port = free_port(), free_port()
  threading.Thread(target=gta5_relay.main, args=(["--port", str(relay_port), "--brain-port", str(brain_port), "--frames-port",
                                                  str(free_port()), "--plugin-file", str(tmp_path / "plugin.txt"), "--link-file",
                                                  str(tmp_path / "link.txt")],), daemon=True).start()
  brain = plugin_connect(brain_port)
  game = plugin_connect(relay_port)
  assert Lines(game).next() == FRAMES_OFF
  heads = []
  for t in (1.0, 2.0):
    game.sendall(frame_msg(t))
    heads.append(json.loads(read_lines(brain, b'"t": %.2f' % t)[-1]))
  assert heads[1]["hostSeq"] == heads[0]["hostSeq"] + 1
  brain.close()
  game.close()
