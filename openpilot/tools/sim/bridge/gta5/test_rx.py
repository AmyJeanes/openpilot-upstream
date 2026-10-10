import json
import multiprocessing
import os
import signal
import socket
import struct
import threading
import time
from multiprocessing.shared_memory import SharedMemory

import pytest

from openpilot.tools.sim.bridge.gta5 import gta5_overlay
from openpilot.tools.sim.bridge.gta5.gta5_cmd import debug_cmd, display_from_env
from openpilot.tools.sim.bridge.gta5.gta5_rx import VIEWS, Receiver, exit_with_parent
from openpilot.tools.sim.lib.camerad import W, H


def test_display_from_env():
  assert display_from_env({}) == []
  assert display_from_env({"GTA5_DEBUG_OVERLAY": " ", "GTA5_GPSROUTE": ""}) == []
  got = display_from_env({"GTA5_DEBUG_OVERLAY": "on layers=edges,arrows width=2", "GTA5_GPSROUTE": "off"})
  assert got == [{"type": "debug", "on": 1, "layers": "ea", "width": 2.0}, {"type": "gpsroute", "on": 0}]
  all_layers = display_from_env({"GTA5_DEBUG_OVERLAY": "on layers=all"})[0]["layers"]
  assert set(all_layers) == set(gta5_overlay.LAYERS.values())
  assert display_from_env({"GTA5_DEBUG_OVERLAY": "yes", "GTA5_GPSROUTE": "on"}) == [{"type": "gpsroute", "on": 1}]


class FakePlugin:
  """The plugin's side of the game connection: commands in, frames with only a state out."""
  def __init__(self, port: int):
    self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
    self.lines = self.sock.makefile("r")

  def read(self, n: int) -> list[dict]:
    return [json.loads(self.lines.readline()) for _ in range(n)]

  def frame(self, state: dict) -> None:
    head = json.dumps({"width": W, "height": H, "views": [], "state": state}).encode()
    msg = struct.pack("<I", len(head)) + head
    self.sock.sendall(struct.pack("<I", len(msg)) + msg)

  def close(self) -> None:
    self.lines.close()
    self.sock.close()


@pytest.fixture
def receiver():
  shm = [SharedMemory(create=True, size=1) for _ in VIEWS]

  def start(display=None) -> Receiver:
    frames_recv, frames_send = multiprocessing.Pipe(duplex=False)
    controls_recv, controls = multiprocessing.Pipe(duplex=False)
    ready_recv, ready_send = multiprocessing.Pipe(duplex=False)
    rx = Receiver(frames_send, controls_recv, {n: m.name for n, m in zip(VIEWS, shm, strict=True)}, display=display, debug_port=0)
    threading.Thread(target=rx.serve, args=(0, ready_send), daemon=True).start()
    assert ready_recv.poll(5) and ready_recv.recv() is None
    rx.frames_out, rx.controls_in = frames_recv, controls  # for the test to read, and to keep open
    return rx

  yield start
  for m in shm:
    m.close()
    m.unlink()


def gta5_cmd(rx: Receiver, cmd: dict) -> None:
  with socket.create_connection(("127.0.0.1", rx.debug_port), timeout=5) as s:
    s.sendall((json.dumps(cmd) + "\n").encode())


def wait_frame(rx: Receiver) -> None:
  assert rx.frames_out.poll(5)
  rx.frames_out.recv()


def test_display_set_again_on_reconnect(receiver):
  rx = receiver(display_from_env({"GTA5_DEBUG_OVERLAY": "on layers=all", "GTA5_GPSROUTE": "on"}))
  plugin = FakePlugin(rx.port)
  debug, gps = plugin.read(2)
  assert debug["type"] == "debug" and debug["on"] == 1 and set(debug["layers"]) == set(gta5_overlay.LAYERS.values())
  assert gps == {"type": "gpsroute", "on": 1}

  # options sent through gta5_cmd go to the plugin at once and are kept, merged with the earlier ones
  gta5_cmd(rx, debug_cmd(["on", "width=3", "dist=200"]))
  assert plugin.read(1) == [{"type": "debug", "on": 1, "width": 3.0, "dist": 200.0}]
  gta5_cmd(rx, {"type": "gpsroute", "on": 0})
  assert plugin.read(1) == [{"type": "gpsroute", "on": 0}]

  # a core reload: the plugin reconnects with everything off, and gets it all back
  plugin.close()
  plugin = FakePlugin(rx.port)
  debug, gps = plugin.read(2)
  assert debug == {"type": "debug", "on": 1, "layers": debug["layers"], "width": 3.0, "dist": 200.0}
  assert gps == {"type": "gpsroute", "on": 0}

  # its frames from before it took them show the reloaded core's defaults, which aren't taken for a change of its own
  plugin.frame({"debug": {"on": False}, "gpsRoute": {"on": False}})
  wait_frame(rx)
  plugin.frame({"debug": {"on": True, "width": 3.0}, "gpsRoute": {"on": False}})
  wait_frame(rx)
  assert rx.display["debug"]["on"] == 1
  # then F7 turns the overlay off, which a reload keeps
  plugin.frame({"debug": {"on": False, "width": 3.0}, "gpsRoute": {"on": False}})
  wait_frame(rx)
  plugin.close()
  plugin = FakePlugin(rx.port)
  debug, gps = plugin.read(2)
  assert debug["on"] == 0 and debug["width"] == 3.0
  plugin.close()


def test_nothing_set_unless_asked(receiver):
  rx = receiver()
  plugin = FakePlugin(rx.port)
  plugin.frame({"debug": {"on": False}, "gpsRoute": {"on": False}})
  wait_frame(rx)
  gta5_cmd(rx, {"type": "ai", "on": 0})  # other commands pass through, and aren't kept
  assert plugin.read(1) == [{"type": "ai", "on": 0}]
  plugin.close()
  plugin = FakePlugin(rx.port)
  plugin.frame({})
  wait_frame(rx)  # connected
  gta5_cmd(rx, {"type": "reset"})
  assert plugin.read(1) == [{"type": "reset"}]
  assert rx.display == {}
  plugin.close()


def test_takes_up_settings_from_before_the_bridge(receiver):
  # a bridge restart: the plugin kept its overlay, set by the bridge before; this one learns it from the state
  rx = receiver()
  plugin = FakePlugin(rx.port)
  plugin.frame({"debug": {"on": True, "layers": "ed", "width": 2.5, "force": False, "lines": 40},
                "gpsRoute": {"on": True, "max": 80, "points": 12}})
  wait_frame(rx)
  plugin.close()
  plugin = FakePlugin(rx.port)
  debug, gps = plugin.read(2)
  assert debug == {"type": "debug", "on": 1, "layers": "ed", "width": 2.5, "force": False}
  assert gps == {"type": "gpsroute", "on": 1, "max": 80}
  plugin.close()


def _bridge_killed(pid_out):
  rx = multiprocessing.Process(target=exit_with_parent, args=(os.getpid(), 0.05))
  rx.start()
  pid_out.send(rx.pid)
  os.kill(os.getpid(), signal.SIGKILL)


def test_rx_ends_with_the_bridge_process():
  # killed by a signal, the bridge's process would leave rx serving its ports, so the bridge looked alive
  pid_in, pid_out = multiprocessing.Pipe(duplex=False)
  bridge = multiprocessing.Process(target=_bridge_killed, args=(pid_out,))
  bridge.start()
  assert pid_in.poll(5)
  rx = pid_in.recv()
  bridge.join(5)
  assert bridge.exitcode == -signal.SIGKILL
  deadline = time.monotonic() + 5
  while time.monotonic() < deadline:
    try:
      with open(f"/proc/{rx}/stat") as f:
        if f.read().rsplit(")", 1)[1].split()[0] == "Z":  # exited, not yet reaped by whoever took it on
          return
    except FileNotFoundError:
      return
    time.sleep(0.05)
  raise AssertionError("rx outlived the bridge process")
