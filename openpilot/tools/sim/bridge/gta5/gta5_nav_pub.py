"""openpilot's navInstruction and navRoute (cereal log.capnp) built from gta5_nav_msgs' content and published, where
openpilot is: the bridge in WSL, for its own brain or, as the link, for the Windows host's."""
import threading

from openpilot.cereal import messaging


def fill_instruction(ni, content: dict) -> None:
  """A navInstruction's fields from gta5_nav_msgs.instruction's content."""
  ni.valid = True
  for key, value in content.items():
    if key == "allManeuvers":
      for out, m in zip(ni.init("allManeuvers", len(value)), value, strict=True):
        out.distance, out.type, out.modifier, out.primaryText = m["distance"], m["type"], m["modifier"], m["primaryText"]
    elif key == "lanes":
      for out, lane in zip(ni.init("lanes", len(value)), value, strict=True):
        out.directions = lane["directions"]
        out.active, out.activeDirection = lane["active"], lane["activeDirection"]
        out.oncoming, out.current = lane["oncoming"], lane["current"]
    elif key == "request":  # navd's NavOutputs.request
      r = ni.request
      r.kind, r.side, r.reason = value["kind"], value["side"], value.get("reason", "other")
      r.distance, r.laneChanges = float(value.get("distance", 0.0)), int(value.get("laneChanges", 0))
      r.stage, r.id, r.approved = int(value.get("stage", 0)), int(value.get("id", 0)), bool(value.get("approved", False))
    elif key == "position":
      ni.position.latitude, ni.position.longitude = value["latitude"], value["longitude"]
    else:
      setattr(ni, key, value)


def route_message(content: dict):
  """A navRoute from gta5_nav_msgs.route_content's content."""
  msg = messaging.new_message("navRoute", valid=True)
  if not content.get("coordinates"):
    return msg
  _coordinates(msg.navRoute.init("coordinates", len(content["coordinates"])), content["coordinates"])
  for r, road in zip(msg.navRoute.init("roads", len(content["roads"])), content["roads"], strict=True):
    _coordinates(r.init("coordinates", len(road["coordinates"])), road["coordinates"])
    r.width = road["width"]
  return msg


def _coordinates(out, points: list):
  for c, (lat, lon) in zip(out, points, strict=True):
    c.latitude, c.longitude = lat, lon


class NavPublisher:
  """Publishes the content it's given, with OpenpilotIO's nav_instruction and nav_route (so it's NavMessages' `out`
  too). A navRoute with the roads near the route is thousands of points, built on a thread of its own and sent from it
  once ready, so the bridge's loop never builds it; the route alone is built at once. nav_route(None), or the same
  content again, sends the last navRoute again."""
  def __init__(self, pm=None):
    self.pm = pm or messaging.PubMaster(["navInstruction", "navRoute"])
    self.lock = threading.Lock()  # the navRoute's content and message, and sending it, from either thread
    self.content: dict | None = None
    self.data: bytes | None = None  # its navRoute, once built

  def nav_instruction(self, content: dict | None) -> None:
    """A navInstruction; valid=false without content (no route)."""
    msg = messaging.new_message("navInstruction", valid=True)
    if content is not None:
      fill_instruction(msg.navInstruction, content)
    self.pm.send("navInstruction", msg)

  def nav_route(self, content: dict | None) -> None:
    with self.lock:
      if content is None or content is self.content:
        if self.data is not None:
          self.pm.send("navRoute", self.data)
        return
      self.content, self.data = content, None
    if content.get("roads"):
      threading.Thread(target=self._build, args=(content,), daemon=True).start()
    else:
      self._build(content)

  def _build(self, content: dict):
    data = route_message(content).to_bytes()
    with self.lock:
      if self.content is content:  # else a newer one came while this was built
        self.data = data
        self.pm.send("navRoute", data)
