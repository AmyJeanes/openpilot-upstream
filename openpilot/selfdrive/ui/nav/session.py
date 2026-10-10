"""The route as the driver has it, shared by the nav views: whether there's a route to show (navigation has one and the
driver hasn't ended it here), and Navigate on openpilot, which each new destination takes from its setting."""
import math
import time

from openpilot.common.params import Params
from openpilot.selfdrive.ui.nav.nav_state import M_PER_DEG, NavState, RouteStarts
from openpilot.selfdrive.ui.ui_state import ui_state

NOO_HOLD_S = 2.0  # a tapped Navigate on openpilot shows as set while the param catches up
END_MOVED = 50.0  # m the destination moves for a route that was ended to count as a new one


class NavSession:
  def __init__(self, nav: NavState, params: Params | None = None):
    self.nav = nav
    self._params = params or Params()
    self.route_on = False
    self.noo = False  # Navigate on openpilot, as shown
    self._noo_set_t = -1e9
    self._ended: tuple | None = None  # the route the driver ended, until nav drops it or has a new destination
    self._starts = RouteStarts()

  def _route_key(self) -> tuple:
    return self.nav.guidance.destination, self.nav.route_end

  def _new_destination(self, ended: tuple) -> bool:
    (name, end), (name0, end0) = self._route_key(), ended
    if name != name0 or (end is None) != (end0 is None):
      return True
    if end is None or end0 is None:
      return False
    dy, dx = (end[0] - end0[0]) * M_PER_DEG, (end[1] - end0[1]) * M_PER_DEG * math.cos(math.radians(end[0]))
    return math.hypot(dx, dy) > END_MOVED

  def update(self, now: float) -> None:
    if self._ended is not None and (not self.nav.active or self._new_destination(self._ended)):
      self._ended = None
    self.route_on = self.nav.active and self._ended is None
    if self._starts.update(self.route_on, self.nav.route_end, now):  # a new destination, not a reroute
      self.set_noo(self._params.get_bool("NavigateOnOpenpilotDefault"))
    if now - self._noo_set_t > NOO_HOLD_S:
      self.noo = ui_state.navigate_on_openpilot

  def set_noo(self, on: bool) -> None:
    self.noo, self._noo_set_t = on, time.monotonic()
    self._params.put_bool("NavigateOnOpenpilot", on)

  def end(self) -> None:
    """The driver ended the route: hidden until navigation drops it or sets a new destination."""
    self._starts.forget()
    self._ended = self._route_key()
    self.route_on = False
    self._params.remove("NavDestination")  # as a phone or the map would: navd drops the destination
