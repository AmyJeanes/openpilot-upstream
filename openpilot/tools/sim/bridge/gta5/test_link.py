"""The brain's openpilot over the wire: gta5_opio.RemoteIO on the host to gta5_link.LinkServer in the link, on localhost."""
import socket
import time

from openpilot.tools.sim.bridge.gta5 import gta5_opio
from openpilot.tools.sim.bridge.gta5.gta5_link import LinkServer
from openpilot.tools.sim.bridge.gta5.gta5_opio import RemoteIO


class FakeLocalIO:
  """The link's LocalIO: keeps the calls; its state is what the test sets."""
  def __init__(self):
    self.calls: list[tuple] = []
    self.inputs: dict = {}
    self.values: dict = {"IsMetric": True, "NavDestination": None}
    self.engaged = False
    self.params = self

  def status(self) -> dict:
    return {"type": "status", "engaged": self.engaged, "set_speed_kph": 40.0, "params": dict(self.values),
            "model": {"turns": {}, "lane_change": 2, "lane_head": [0, 2, 0.9]}}

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


def test_link_round_trip(monkeypatch):
  """The host's calls arrive and are applied on the link's step; openpilot's state comes back; vectors go at most every
  VECTOR_EVERY s, the newest; a param the brain set holds against a state from before the link had it; the host going
  resets the link's inputs."""
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
