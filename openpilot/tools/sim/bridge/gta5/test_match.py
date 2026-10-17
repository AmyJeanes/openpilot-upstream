import numpy as np

from openpilot.tools.sim.bridge.gta5.navd.map_match import Match
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


def test_navigator_stops():
  """A route through two stops: each is passed as the car goes by where the route passes it, dropping it keeps the route,
  and any other change to the stops routes again (through them, to Valhalla)."""
  calls: list = []
  route = Route(np.array([(0.0, 0.0), (0.0, 300.0), (300.0, 300.0)]), [])

  class FakeRouter:
    def route(self, pos, bearing, dest, z, through=()):
      calls.append([list(p) for p in through])
      return route

  nav = Navigator(FakeRouter())
  dest, stops = np.array([300.0, 300.0]), [np.array([5.0, 100.0]), np.array([150.0, 310.0])]
  nav._route(np.array([0.0, 0.0]), 0.0, dest, None, 0, stops)  # what _start's thread does
  assert nav.route is None  # the route came for a destination no longer wanted
  nav.dest, nav.stops = dest, list(stops)
  nav._route(np.array([0.0, 0.0]), 0.0, dest, None, 0, stops)
  assert calls[-1] == [[5.0, 100.0], [150.0, 310.0]] and np.allclose(nav.stop_at, (100.0, 450.0))
  nav._start = lambda *a: calls.append("again")
  assert nav.update(np.array([0.0, 50.0]), 0.0, dest, 0.0, stops=stops) is route and nav.passed() == 0
  nav.update(np.array([0.0, 97.0]), 0.0, dest, 1.0, stops=stops)
  assert nav.passed() == 1
  assert nav.update(np.array([0.0, 120.0]), 0.0, dest, 2.0, stops=stops[1:]) is route and nav.passed() == 0
  assert np.allclose(nav.stop_at, (450.0,)) and calls[-1] != "again"
  assert nav.update(np.array([0.0, 130.0]), 0.0, dest, 3.0, stops=[np.array([200.0, 290.0])]) is None
  assert calls[-1] == "again"


def test_navigator_routes_again_with_more_map():
  """A route made before the map's paths and lanes loaded is routed again once they have, on the route meanwhile."""
  route = Route(np.array([(0.0, 0.0), (0.0, 500.0)]), [])
  nav, starts = navigator_on(route)
  dest = np.array([0.0, 500.0])
  nav.update(np.array([0.0, 100.0]), 0.0, dest, 0.0)
  nav.refresh()
  assert nav.update(np.array([0.0, 101.0]), 0.0, dest, 1.0) is route and len(starts) == 1
  nav.route_version = nav.map_version  # the new route came
  nav.update(np.array([0.0, 102.0]), 0.0, dest, 5.0)
  assert len(starts) == 1
