"""openpilot as the bridge's brain (gta5_brain.py) sees it, and the wire that carries it from Windows to WSL.

OpenpilotIO is what the brain reads of openpilot (engaged, the set speed, the model's desires, lane change and lane,
its params) and what it tells openpilot (cruise buttons, the simulated driver's inputs, nav's desire and speed cap, the
route input and lane slots, a bookmark), and openpilot's control message for the plugin. LocalIO (gta5_world.py) is
openpilot in the same process: the bridge in WSL. RemoteIO here makes the same calls from the Windows host
(gta5_host.py) to the link (the bridge in WSL run with GTA5_LINK=1, gta5_link.LinkServer), which applies them to its
LocalIO and sends its state back: JSON lines over TCP, the host connecting out (Windows blocks connections from WSL).
NullIO is no openpilot at all: the host run with --no-link.

Standard library only: it runs on Windows' own Python."""
import json
import socket
import threading
import time
from collections import deque
from collections.abc import Callable

LINK_PORT = 8796  # the link's control port, on the address the link writes to its link file
# the SimulatorState fields the brain sets each step (the link sets the rest from the game state itself)
INPUTS = ("user_torque", "left_blinker", "right_blinker", "left_blindspot", "right_blindspot", "speed_limit",
          "speed_limit_follow", "cruise_cap", "cap_set_speed")
PARAMS = ("IsMetric", "NavigateOnOpenpilot", "TurnDesireRefresh", "NavDestination")  # the params the brain reads
OPS = ("cruise", "set_inputs", "nav_desire", "nav_speed", "route_input", "lane_slots", "bookmark", "param_put", "param_remove",
       "nav_instruction", "nav_route", "expert_control")
COMMANDS = ("expert", "destination")  # what the link passes on to the brain from WSL (take_commands)
STATUS_EVERY = 0.05  # s between the link's states
VECTOR_EVERY = 0.05  # s between route input or lane slot vectors sent, the newest when it changed
CONNECT_TIMEOUT = 1.0  # s
RETRY_EVERY = 1.0  # s
PARAM_HOLD = 2.0  # s a param the brain set keeps its value against the link's states, which may predate the change


class OpenpilotIO:
  """What the brain needs of openpilot. engaged and set_speed_kph are as of the last update()."""
  drives = True  # openpilot is there to drive: else only the brain's own drivers can
  engaged = False
  set_speed_kph = 0.0  # the HUD's set speed, km/h; 0 when not set
  params = None  # get(key), get_bool(key), put(key, value), remove(key)

  def update(self) -> None:
    """Once per bridge step, before the rest."""

  def model(self) -> dict | None:
    """The newest modelV2 as nav uses it: {"turns": desire probabilities by turnLeft/turnRight/keepLeft/keepRight ({}
    without them), "lane_change": laneChangeState as an int, "lane_head": [lane index from the left, lane count,
    probability], None without a fresh one or from a model without the head}; None before any."""
    return None

  def controls(self, state: dict) -> dict | None:
    """openpilot's control message for the plugin while it's engaged, else None; None too where openpilot sends its own
    (the link, behind RemoteIO)."""
    return None

  def cruise(self, cmd: str) -> None:
    """A cruise button on the simulated car (common.control_cmd_gen's names: cruise_down sets, cruise_cancel...)."""

  def set_inputs(self, **inputs) -> None:
    """INPUTS fields of the simulated car, held until set again."""

  def nav_desire(self, desire: str) -> None:
    """The NavDesire param ("" removes it)."""

  def nav_speed(self, cap: float, reason: str) -> None:
    """navSpeed for openpilot's longitudinal planner (m/s, 0 for none)."""

  def route_input(self, vec) -> None:
    """The driving model's route input (navd/route_input.py)."""

  def lane_slots(self, vec) -> None:
    """Route input v2's lane slots preview (navd/lane_slots.py)."""

  def bookmark(self) -> None:
    """userBookmark, so loggerd keeps the segment."""

  def nav_instruction(self, content: dict | None) -> None:
    """navInstruction for the onroad UI, from gta5_nav_msgs' content (None: no route)."""

  def nav_route(self, content: dict) -> None:
    """navRoute for the onroad UI, from gta5_nav_msgs' content; the same object again when it's to be sent again."""

  def take_fix(self) -> dict | None:
    """The newest GNSS fix openpilot got (gta5_gnss.py) since the last call, for navd's map match: {"lat", "lon",
    "bearing", "bearing_accuracy", "speed", "latency": s from the position to now}; None without a new one."""
    return None

  def take_commands(self) -> list[dict]:
    """What came from WSL for the brain since the last call (the link's COMMANDS): {"type": "expert", "cfg"}, the WSL
    expert control file's new settings; {"type": "destination", "xy": [x, y] or None}, a destination posted to WSL's
    map view port (gta5_cmd.py expert route)."""
    return []

  def expert_control(self, cfg: dict) -> None:
    """What expert mode writes to its control file, for the WSL one (the link's)."""


class MemoryParams:
  """Params kept here, for no openpilot: a key never put reads as None (False)."""
  def __init__(self):
    self.values: dict = {}

  def get(self, key: str):
    return self.values.get(key)

  def get_bool(self, key: str) -> bool:
    return bool(self.values.get(key))

  def put(self, key: str, value) -> None:
    self.values[key] = value

  def remove(self, key: str) -> None:
    self.values.pop(key, None)


class NullIO(OpenpilotIO):
  """No openpilot (gta5_host.py --no-link): never engaged, no model, its params at their defaults, and what's told it
  goes nowhere."""
  drives = False
  connected = False

  def __init__(self):
    self.params = MemoryParams()


class RemoteParams:
  """The link's PARAMS as of its last state; puts and removes go to it."""
  def __init__(self, io: "RemoteIO"):
    self.io = io
    self.values: dict = {}
    self.set_at: dict[str, float] = {}  # when the brain last set each key

  def get(self, key: str):
    return self.values.get(key)

  def get_bool(self, key: str) -> bool:
    return bool(self.values.get(key))

  def put(self, key: str, value) -> None:
    self.values[key], self.set_at[key] = value, time.monotonic()
    self.io.send("param_put", key, value)

  def remove(self, key: str) -> None:
    self.values[key], self.set_at[key] = None, time.monotonic()
    self.io.send("param_remove", key)

  def update(self, values: dict) -> None:
    now = time.monotonic()
    for key, value in values.items():
      if now - self.set_at.get(key, -PARAM_HOLD) >= PARAM_HOLD:
        self.values[key] = value


class RemoteIO(OpenpilotIO):
  """OpenpilotIO over TCP to the link at address() (host, port), connecting again whenever the link goes: while it's
  down openpilot reads as disengaged, with no model, and calls are dropped."""
  def __init__(self, address: Callable[[], tuple[str, int] | None]):
    self.address = address
    self.params = RemoteParams(self)
    self.sock: socket.socket | None = None
    self.lock = threading.Lock()
    self.status: dict = {}
    self.status_t = 0.0
    self.sent_inputs: dict = {}
    self.vectors: dict[str, tuple] = {}  # op: (the newest vector, sent yet, when one was last sent)
    self.nav_route_sent: dict | None = None  # the navRoute content the link has
    self.fix_seq: int | None = None  # the link's last GNSS fix taken
    self.commands: deque[dict] = deque()
    self.connected = False
    threading.Thread(target=self._run, daemon=True).start()

  # *** openpilot's state ***

  def update(self) -> None:
    s = self.status if self.connected else {}
    self.engaged = bool(s.get("engaged"))
    self.set_speed_kph = float(s.get("set_speed_kph") or 0.0)
    self.params.update(s.get("params") or {})
    now = time.monotonic()
    for op, (vec, sent, t) in list(self.vectors.items()):
      if not sent and now - t >= VECTOR_EVERY and self.send(op, vec):
        self.vectors[op] = (vec, True, now)

  def model(self) -> dict | None:
    return self.status.get("model") if self.connected else None

  def take_fix(self) -> dict | None:
    status, status_t = self.status, self.status_t
    fix = status.get("gnss") if self.connected else None
    if not fix or fix["seq"] == self.fix_seq:
      return None
    self.fix_seq = fix["seq"]
    # its age in the link when sent, then here since: each measured on its own side's clock
    return {**fix, "latency": fix["latency"] + fix["age"] + (time.monotonic() - status_t)}

  def take_commands(self) -> list[dict]:
    out = []
    while self.commands:
      out.append(self.commands.popleft())
    return out

  # *** to openpilot ***

  def cruise(self, cmd: str) -> None:
    self.send("cruise", cmd)

  def set_inputs(self, **inputs) -> None:
    changed = {k: v for k, v in inputs.items() if self.sent_inputs.get(k, ...) != v}
    if changed and self.send("set_inputs", changed):
      self.sent_inputs.update(changed)

  def nav_desire(self, desire: str) -> None:
    self.send("nav_desire", desire)

  def nav_speed(self, cap: float, reason: str) -> None:
    self.send("nav_speed", cap, reason)

  def route_input(self, vec) -> None:
    self._vector("route_input", vec)

  def lane_slots(self, vec) -> None:
    self._vector("lane_slots", vec)

  def bookmark(self) -> None:
    self.send("bookmark")

  def expert_control(self, cfg: dict) -> None:
    self.send("expert_control", cfg)

  def nav_instruction(self, content: dict | None) -> None:
    self.send("nav_instruction", content)

  def nav_route(self, content: dict) -> None:
    # thousands of points: the link keeps the last, and None has it send that again
    if self.send("nav_route", None if content is self.nav_route_sent else content):
      self.nav_route_sent = content

  def _vector(self, op: str, vec):
    vec = [round(float(v), 5) for v in vec]
    old = self.vectors.get(op)
    if old is None or old[0] != vec:
      self.vectors[op] = (vec, False, old[2] if old is not None else 0.0)

  # *** the connection ***

  def send(self, op: str, *args) -> bool:
    with self.lock:
      sock = self.sock
    if sock is None:
      return False
    try:
      sock.sendall((json.dumps({"op": op, "args": args}) + "\n").encode())
      return True
    except OSError:
      self._drop(sock)
      return False

  def _drop(self, sock: socket.socket):
    with self.lock:
      if self.sock is sock:
        self.sock, self.connected = None, False
        self.sent_inputs = {}  # a new link starts from its defaults
        self.nav_route_sent = None
        self.fix_seq = None
        self.vectors = {op: (vec, False, t) for op, (vec, _, t) in self.vectors.items()}
    close(sock)

  def _run(self):
    last = None
    while True:
      addr = self.address()
      try:
        if addr is None:
          raise OSError("no link address")
        sock = socket.create_connection(addr, timeout=CONNECT_TIMEOUT)
      except OSError as e:
        why = f"{addr}: {e}"
        if why != last:
          print(f"gta5: waiting for the link ({why})", flush=True)
          last = why
        time.sleep(RETRY_EVERY)
        continue
      sock.settimeout(None)
      sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
      print(f"gta5: connected to the link at {addr[0]}:{addr[1]}", flush=True)
      last = None
      with self.lock:
        self.sock = sock
      try:
        for line in sock.makefile("r", encoding="utf-8"):
          msg = json.loads(line)
          if msg.get("type") == "status":
            self.status, self.status_t, self.connected = msg, time.monotonic(), True
          elif msg.get("type") in COMMANDS:
            self.commands.append(msg)
      except (OSError, ValueError):
        pass
      self._drop(sock)
      print("gta5: the link went", flush=True)


def close(sock: socket.socket):
  """Closes a socket another thread reads through a file object, which otherwise keeps it open."""
  try:
    sock.shutdown(socket.SHUT_RDWR)
  except OSError:
    pass
  sock.close()


def link_address(link_file: str, port: int = LINK_PORT) -> Callable[[], tuple[str, int] | None]:
  """The link's control address: the host in the link file the bridge writes (GTA5_BRIDGE_FILE), on port."""
  def read() -> tuple[str, int] | None:
    try:
      host = open(link_file).read().strip().rpartition(":")[0]
      return (host, port) if host else None
    except OSError:
      return None
  return read
