import numpy as np

from openpilot.selfdrive.navd.map_match import Match
from openpilot.tools.sim.bridge.gta5.map.router import Navigator, Route


def navigator_on(route: Route) -> tuple[Navigator, list]:
  nav = Navigator(router=None)
  starts: list = []
  nav._start = lambda pos, bearing, dest, now, z: starts.append((pos.copy(), bearing))
  nav.dest, nav.route = np.array([0.0, 500.0]), route
  return nav, starts


def match(x: float, y: float, heading: float, sure: bool) -> Match:
  return Match(0, 0.5, np.array([x, y]), heading, 1, 2.0, 0.9, sure)


def test_navigator_off_route_by_the_match():
  """A northbound route; the match puts the car on the southbound carriageway 12 m west. Sure of the direction, that
  is off the route (and routing again starts from the match, the way its road goes); unsure, only the distance counts."""
  route = Route(np.array([(0.0, 0.0), (0.0, 500.0)]), [])
  nav, starts = navigator_on(route)
  dest = np.array([0.0, 500.0])
  for t in (10.0, 11.0, 12.0):
    nav.update(np.array([0.0, 100.0]), 0.0, dest, t, match=match(-12.0, 100.0, 180.0, sure=True))
  assert route.off > Navigator.OFF_ROUTE and len(starts) == 1
  assert np.allclose(starts[0][0], (-12.0, 100.0)) and starts[0][1] == -180.0
  route2 = Route(np.array([(0.0, 0.0), (0.0, 500.0)]), [])
  nav2, starts2 = navigator_on(route2)
  for t in (10.0, 11.0, 12.0):
    nav2.update(np.array([0.0, 100.0]), 0.0, dest, t, match=match(-12.0, 100.0, 180.0, sure=False))
  assert route2.off < Navigator.OFF_ROUTE and not starts2


def test_navigator_without_a_match_is_unchanged():
  route = Route(np.array([(0.0, 0.0), (0.0, 500.0)]), [])
  nav, starts = navigator_on(route)
  nav.update(np.array([3.0, 100.0]), 0.0, np.array([0.0, 500.0]), 0.0)
  assert abs(route.off - 3.0) < 1e-6 and abs(route.at - 100.0) < 1e-6 and not starts
