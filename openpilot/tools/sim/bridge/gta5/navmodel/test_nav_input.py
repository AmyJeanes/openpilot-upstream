"""nav_input.py on GTA routes from the bridge's Valhalla (fixtures/routes.json: city turns, a freeway trip, a trip with
exits). No pytest needed: `python test_nav_input.py` runs them all."""
import json
from pathlib import Path

import numpy as np

from openpilot.selfdrive.navigation.helpers import VALHALLA_COMMAND_COUNT, valhalla_type
from openpilot.selfdrive.navigation.route import RouteTracker
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon
from openpilot.tools.sim.bridge.gta5.map.router import Route
from openpilot.tools.sim.bridge.gta5.navmodel.nav_input import AS_NAVD, KEEPS, MERGES, ModelNav, navigation, words

ROUTES = json.loads((Path(__file__).parent / "fixtures" / "routes.json").read_text())
NAVD_TYPES = {8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28}  # all valhalla_type can give > 6


def route(name: str) -> Route:
  r = ROUTES[name]
  return Route(np.array(r["points"], dtype=float), [dict(m) for m in r["maneuvers"]])


def lon_lat(xy) -> tuple[float, float]:
  lat, lon = to_lat_lon(float(xy[0]), float(xy[1]))
  return lon, lat


def point_at(r: Route, s: float) -> np.ndarray:
  return np.array([np.interp(s, r.along, r.points[:, 0]), np.interp(s, r.along, r.points[:, 1])])


def test_words_read_back_as_the_type():
  for t in range(VALHALLA_COMMAND_COUNT):
    w = words(t)
    if t in MERGES:
      assert valhalla_type(*w) is None, t  # skipped, as navd's merge steps are
    elif w is not None:
      assert valhalla_type(*w) == AS_NAVD.get(t, t), (t, w)
  given = {valhalla_type(*words(t)) for t in range(VALHALLA_COMMAND_COUNT) if words(t) is not None} - {None}
  assert given - set(range(7)) == NAVD_TYPES - {28}  # every type navd gives but ferries'


def test_distances_as_navd_measures_them():
  """Each maneuver where navd puts a Mapbox step's: its steps' geometries end to end (each from its maneuver to the
  next, sharing that point), the tracker's distance at each step's first point."""
  for name in ROUTES:
    r = route(name)
    nav = ModelNav(r, junctions=False, keep_run=None)
    coords = nav.coordinates
    starts = [m["begin_shape_index"] for m in r.maneuvers]
    steps = [coords[a:b + 1] for a, b in zip(starts, starts[1:] + [len(coords) - 1], strict=True)]
    tracker = RouteTracker(np.concatenate(steps))
    navd = tracker.distance[np.cumsum([0] + [len(s) for s in steps[:-1]])]
    ours = [d for d, *_ in nav.maneuvers]
    np.testing.assert_allclose(ours, navd, atol=1e-6, err_msg=name)
    game = r.along[starts]  # and they're game metres, to the mean Earth radius navd measures by (0.11 % short)
    np.testing.assert_allclose(ours, game * 6371008.8 / 6378137.0, rtol=1e-4, atol=0.01, err_msg=name)


def test_vector_along_a_city_route():
  r = route("city")
  nav = ModelNav(r)
  turns = [(d, t) for d, *_, t in nav.maneuvers if t > 6]
  first_d, first_t = turns[0]
  assert first_t == 10  # Turn right onto Elgin Ave.
  v = navigation(nav.all_maneuvers_at(r.points[0]))
  assert v.shape == (47,) and v.dtype == np.float32
  assert v[10] == 1 and v[:46].sum() == 1
  assert abs(v[46] - first_d / 100) < 1e-4
  # drive it, 3 m right of the line as a GNSS fix might be: one-hot on the next turn, its distance falling to 0
  seen = []
  for s in np.arange(0, r.length, 5.0):
    p = point_at(r, s)
    heading = point_at(r, min(s + 1, r.length)) - p
    p = p + 3.0 * np.array([heading[1], -heading[0]]) / max(np.hypot(*heading), 1e-9)
    v = navigation(nav.all_maneuvers_at(p))
    if v[:46].any():
      t = int(np.argmax(v[:46]))
      if not seen or seen[-1] != t:
        seen.append(t)
      assert 0 <= v[46] < 40
  expected = [t for k, (_, t) in enumerate(turns) if k == 0 or turns[k - 1][1] != t]  # two lefts in turn read as one
  assert seen == expected, (seen, expected)
  assert not navigation(nav.all_maneuvers_at(r.points[-1]))[:46].any()  # past the last turn: none, though arriving


def test_traveled_moves_on_to_the_next():
  """modeld's dead reckoning between navInstructions (v_ego * age): past a maneuver, the one after it."""
  r = route("city")
  nav = ModelNav(r)
  turns = [(d, t) for d, *_, t in nav.maneuvers if t > 6]
  mans = nav.all_maneuvers_at(r.points[0])
  v = navigation(mans, traveled=turns[0][0] - 10)
  assert v[turns[0][1]] == 1 and abs(v[46] - 0.1) < 1e-4
  v = navigation(mans, traveled=turns[0][0] + 1)
  assert v[turns[1][1]] == 1 and abs(v[46] - (turns[1][0] - turns[0][0] - 1) / 100) < 1e-4


def test_passed_maneuvers_drop_out_and_progress_holds():
  r = route("exits")
  nav = ModelNav(r)
  s = 3000.0
  mans = nav.all_maneuvers_at(point_at(r, s))
  assert all(m["distance"] >= 0 for m in mans)
  assert len(mans) == sum(d >= s * 0.9989 - 0.5 for d, *_ in nav.maneuvers)
  back = nav.all_maneuvers_at(point_at(r, s - 20))  # a fix behind: the tracker never goes back
  assert [m["distance"] for m in back] == [m["distance"] for m in mans]


def test_keeps_of_lane_splits_fold():
  """La Puerta Fwy: keep left 620, 632, 645 m along (two "at the fork", then "to take La Puerta Fwy") is one keep left
  at 620 m, onto La Puerta Fwy."""
  r = route("freeway")
  raw = ModelNav(r, keep_run=None)
  nav = ModelNav(r)
  keeps = [(round(d), t, road) for d, _, _, road, t in nav.maneuvers if t in KEEPS]
  raw_keeps = [(round(d), t) for d, *_, t in raw.maneuvers if t in KEEPS]
  assert len(keeps) < len(raw_keeps)
  at = [k for k in keeps if 600 < k[0] < 660]
  assert len(at) == 1 and at[0][1] == 24 and at[0][2] == "La Puerta Fwy", keeps
  assert sum(600 < d < 660 for d, _ in raw_keeps) == 3
  for (d0, *_), (d1, *_) in zip(keeps, keeps[1:], strict=False):
    assert d1 - d0 > 50 * 0.998


def test_turns_through_a_junctions_links_are_one():
  """Power St to Vespucci Blvd: "Bear left." (an unnamed link) then "Bear right onto Vespucci Blvd" 1 m on is one
  maneuver; the jog comes back nearly straight."""
  r = route("city")
  raw = [m for m in r.maneuvers if 99 <= m["begin_shape_index"] <= 101]
  assert [m["type"] for m in raw] == [16, 9]
  nav = ModelNav(r)
  near = [(t, road) for d, _, _, road, t in nav.maneuvers if abs(d - r.along[100]) < 5]
  assert len(near) == 1 and near[0][1] == "Vespucci Blvd", near


if __name__ == "__main__":
  for name, f in list(globals().items()):
    if name.startswith("test_") and callable(f):
      f()
      print("ok", name)
