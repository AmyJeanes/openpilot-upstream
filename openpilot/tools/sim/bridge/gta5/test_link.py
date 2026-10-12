"""The brain's openpilot over the wire: gta5_opio.RemoteIO on the host to gta5_link.LinkServer in the link, on localhost."""
import json
import socket
import time
from types import SimpleNamespace

from openpilot.tools.sim.bridge.gta5 import gta5_link, gta5_opio
from openpilot.tools.sim.bridge.gta5.gta5_link import LinkServer
from openpilot.tools.sim.bridge.gta5.gta5_opio import RemoteIO


class FakeLocalIO:
  """The link's LocalIO: keeps the calls; its state is what the test sets."""
  def __init__(self):
    self.calls: list[tuple] = []
    self.inputs: dict = {}
    self.values: dict = {"IsMetric": True, "NavDestination": None}
    self.engaged = False
    self.gnss: dict | None = None
    self.params = self

  def status(self) -> dict:
    return {"type": "status", "engaged": self.engaged, "set_speed_kph": 40.0, "params": dict(self.values),
            "model": {"turns": {}, "lane_change": 2, "lane_head": [0, 2, 0.9]}, "gnss": self.gnss}

  def set_inputs(self, **inputs):
    self.inputs.update(inputs)

  def reset(self):
    self.calls.append(("reset",))
    self.inputs = {}

  def put(self, key, value):
    self.values[key] = value
    self.calls.append(("put", key, value))

  def remove(self, key):
    self.values[key] = None
    self.calls.append(("remove", key))

  def __getattr__(self, op):
    return lambda *args: self.calls.append((op, *args))


def free_port() -> int:
  with socket.socket() as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def until(cond, link: LinkServer, timeout: float = 5.0):
  end = time.monotonic() + timeout
  while not cond():
    assert time.monotonic() < end, "timed out"
    link.step()
    time.sleep(0.01)


def test_link_stands_in_for_the_host_in_wsl(tmp_path, monkeypatch):
  """The WSL expert control file's changes while the host is connected go to it, not those from before; what the
  host's expert mode writes goes to the file, and isn't passed back; a destination posted to the link's map view goes
  to the host."""
  monkeypatch.setattr(gta5_opio, "RETRY_EVERY", 0.05)
  monkeypatch.setattr(gta5_link, "CONTROL_EVERY", 0.0)
  control = tmp_path / "gta5_expert.json"
  control.write_text('{"on": false, "from": "before"}\n')
  port = free_port()
  link = LinkServer(FakeLocalIO(), port, control=control)
  picks: list = []
  link.view = SimpleNamespace(take_destination=lambda: picks.pop() if picks else None, close=lambda: None)
  remote = RemoteIO(lambda: ("127.0.0.1", port))
  until(lambda: remote.connected, link)
  picks.append(([10.0, -20.0],))
  got: list[dict] = []
  until(lambda: got.extend(remote.take_commands()) or any(c["type"] == "destination" for c in got), link)
  assert got == [{"type": "destination", "xy": [10.0, -20.0]}]  # and not the file from before
  time.sleep(0.01)
  control.write_text('{"on": true, "speed": 9}\n')
  until(lambda: got.extend(remote.take_commands()) or len(got) == 2, link)
  assert got[1] == {"type": "expert", "cfg": {"on": True, "speed": 9}}
  remote.expert_control({"on": False})
  until(lambda: json.loads(control.read_text()) == {"on": False}, link)
  for _ in range(20):
    link.step()
    got.extend(remote.take_commands())
    time.sleep(0.01)
  assert len(got) == 2  # its own write isn't passed back
  link.close()


def test_link_round_trip(monkeypatch):
  """The host's calls arrive and are applied on the link's step; openpilot's state comes back, with each GNSS fix once;
  vectors go at most every VECTOR_EVERY s, the newest; a param the brain set holds against a state from before the link
  had it; the nav messages' navRoute goes whole only when it changed; the host going resets the link's inputs."""
  monkeypatch.setattr(gta5_opio, "RETRY_EVERY", 0.05)
  port = free_port()
  local = FakeLocalIO()
  link = LinkServer(local, port)
  remote = RemoteIO(lambda: ("127.0.0.1", port))
  until(lambda: remote.connected, link)
  remote.update()
  assert remote.params.get_bool("IsMetric") and remote.set_speed_kph == 40.0 and not remote.engaged
  assert remote.model() == {"turns": {}, "lane_change": 2, "lane_head": [0, 2, 0.9]}

  remote.cruise("cruise_down")
  remote.set_inputs(user_torque=0, left_blinker=True)
  remote.set_inputs(user_torque=0, left_blinker=True)  # unchanged: not sent again
  remote.nav_desire("turnLeft")
  remote.nav_speed(6.0, "turnLeft")
  remote.bookmark()
  until(lambda: ("bookmark",) in local.calls, link)
  assert ("cruise", "cruise_down") in local.calls and ("nav_desire", "turnLeft") in local.calls
  assert ("nav_speed", 6.0, "turnLeft") in local.calls and local.inputs == {"user_torque": 0, "left_blinker": True}

  remote.route_input([1.0, 2.0])
  remote.route_input([1.0, 3.0])  # replaces the first before it's sent
  remote.update()
  until(lambda: any(c[0] == "route_input" for c in local.calls), link)
  sent = [c[1] for c in local.calls if c[0] == "route_input"]
  assert sent == [[1.0, 3.0]]

  local.engaged = True
  until(lambda: remote.update() or remote.engaged, link)
  remote.params.put("NavDestination", {"latitude": 1.0, "longitude": 2.0})
  remote.update()  # a state from before the put still says None: the brain's value holds
  assert remote.params.get("NavDestination") == {"latitude": 1.0, "longitude": 2.0}
  until(lambda: ("put", "NavDestination", {"latitude": 1.0, "longitude": 2.0}) in local.calls, link)

  assert remote.take_fix() is None
  local.gnss = {"lat": 34.0, "lon": -118.0, "bearing": 90.0, "bearing_accuracy": 2.0, "speed": 10.0, "latency": 0.3,
                "seq": 1, "age": 0.1}
  fix = []
  until(lambda: fix.append(remote.take_fix()) or fix[-1] is not None, link)
  assert fix[-1]["lat"] == 34.0 and fix[-1]["latency"] >= 0.4  # its age in the link added
  assert remote.take_fix() is None  # each fix once

  route = {"coordinates": [[34.0, -118.0], [34.1, -118.1]], "roads": []}
  remote.nav_instruction({"distanceRemaining": 120.0})
  remote.nav_route(route)
  remote.nav_route(route)  # again, unchanged: the link sends the one it has
  until(lambda: len([c for c in local.calls if c[0] == "nav_route"]) == 2, link)
  nav = [c for c in local.calls if c[0] in ("nav_instruction", "nav_route")]
  assert nav == [("nav_instruction", {"distanceRemaining": 120.0}), ("nav_route", route), ("nav_route", None)]

  remote._drop(remote.sock)
  until(lambda: ("reset",) in local.calls, link)
  assert local.inputs == {}
  until(lambda: remote.connected, link)
  remote.nav_route(route)  # a new connection: the link may be new too, so it's sent whole
  until(lambda: len([c for c in local.calls if c[0] == "nav_route"]) == 3, link)
  assert local.calls[-1] == ("nav_route", route)
  link.close()


def test_model_tap(monkeypatch):
  """Once the host asks, the link sends camerad's frame id for each of the relay's frames it hands on and modeld's records,
  their arrays exact; nothing before it asks, or for a blank frame; the host gets {"type": "link"} on each connection and
  asks again, and the link stops when the host goes."""
  import numpy as np
  from openpilot.tools.sim.bridge.gta5.gta5_record import model_record_from_json
  monkeypatch.setattr(gta5_opio, "RETRY_EVERY", 0.05)
  port = free_port()
  taps = []
  local = FakeLocalIO()
  link = LinkServer(local, port, model_tap=lambda on_record: taps.append(on_record) or object())
  remote = RemoteIO(lambda: ("127.0.0.1", port))
  until(lambda: remote.connected, link)
  link.camera(100)  # before the host asks: counted, not sent
  got = []
  remote.model_tap(got.append)
  until(lambda: link.tap_on, link)
  assert len(taps) == 1
  link.camera(101)
  link.camera(None)  # a blank frame: counted, not sent
  link.camera(102)
  rec = {"time": 1.5, "nav": "turnLeft", "left_blinker": True, "desire_pulse": None,
         "desire_input": np.eye(8, dtype=np.float32)[3], "raw": np.linspace(-2, 2, 50).astype(np.float16)}
  taps[0](1, rec)
  until(lambda: len(got) >= 3, link)
  assert got[:2] == [{"type": "camera", "seq": 101, "frame_id": 1}, {"type": "camera", "seq": 102, "frame_id": 3}]
  assert got[2]["type"] == "model" and got[2]["frame_id"] == 1
  back = model_record_from_json(got[2]["rec"])
  assert back.keys() == rec.keys() and all(np.array_equal(back[k], v) and np.asarray(back[k]).dtype == np.asarray(v).dtype
                                           for k, v in rec.items() if isinstance(v, np.ndarray))
  assert {k: v for k, v in back.items() if not isinstance(v, np.ndarray)} == {k: v for k, v in rec.items() if not isinstance(v, np.ndarray)}

  remote._drop(remote.sock)  # the host reconnects: told so, and asks again
  until(lambda: ("reset",) in local.calls and {"type": "link"} in got and link.tap_on, link)
  assert len(taps) == 1  # the tap is made once
  link.close()
