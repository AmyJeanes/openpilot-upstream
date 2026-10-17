"""comma's nav model's `navigation` input from the bridge's routes: a router.Route's Valhalla maneuvers as
navInstruction.allManeuvers, measured along the route the way comma's navd measures its Mapbox steps, so that modeld's
navigation_input (selfdrive/navigation/helpers.py, comma nav-model-integration afbf2e5) gives the vector the model
expects: one-hot of the next maneuver's Valhalla type (> 6) and its distance / 100.

The router is Valhalla, the model's own types, so they go in as they are rather than through Mapbox. allManeuvers'
fields are Mapbox's words (type, modifier: cereal NavInstruction.Maneuver), which modeld turns back into a Valhalla type
with valhalla_type(); each type goes as the words valhalla_type reads back as that same type (WORDS). The types navd
never produces go as navd would give them: 7 (becomes) as 8 (continue, Mapbox's "new name"), 12 (U-turn right) as 13,
merges (25, 37, 38) as Mapbox's "merge", which the model skips for the maneuver after it.

Distances, as navd's (selfdrive/navigation/navd.py): a RouteTracker over the route's (lon, lat), each maneuver at its
begin shape point, less the car's progress (the tracker's projection of the car's position, never going back); the
maneuvers passed drop out.

GTA's roads give the router maneuvers a real map's wouldn't, and they're tidied first (both as options):
- a turn through a junction's links (two lefts across a median) is one, as navd's junction_maneuvers makes it;
- a run of keeps where GTA splits a road's lanes into groups (keep left, keep left, keep left 12 m apart, "at the
  fork" with no road named) is one keep, at the first, on the side of the last (the branch the route ends on),
  onto the last's road: each keep within KEEP_RUN m of the one before joins the run.
"""
from types import SimpleNamespace

import numpy as np

from openpilot.tools.sim.bridge.gta5.navd.maneuvers import junction_maneuvers
from openpilot.selfdrive.navigation.helpers import navigation_input
from openpilot.selfdrive.navigation.route import RouteTracker
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon

# Valhalla maneuver types as the Mapbox words comma's valhalla_type() reads back as the same type
WORDS = {
  1: ("depart", "straight"), 2: ("depart", "right"), 3: ("depart", "left"),
  4: ("arrive", "straight"), 5: ("arrive", "right"), 6: ("arrive", "left"),
  8: ("continue", "straight"),
  9: ("turn", "slight right"), 10: ("turn", "right"), 11: ("turn", "sharp right"), 13: ("turn", "uturn"),
  14: ("turn", "sharp left"), 15: ("turn", "left"), 16: ("turn", "slight left"),
  17: ("on ramp", "straight"), 18: ("on ramp", "right"), 19: ("on ramp", "left"),
  20: ("off ramp", "right"), 21: ("off ramp", "left"),
  22: ("fork", "straight"), 23: ("fork", "right"), 24: ("fork", "left"),
  26: ("roundabout", "straight"), 27: ("exit roundabout", "straight"),
}
# types navd can't give, as navd gives them: becomes as continue, U-turn right as the one U-turn type
AS_NAVD = {7: 8, 12: 13}
# merges: Mapbox's merge step, which valhalla_type gives no type (the model never saw one)
MERGES = {25: ("merge", "straight"), 37: ("merge", "slight right"), 38: ("merge", "slight left")}
KEEPS = {22, 23, 24}
KEEP_RUN = 50.0  # m between keeps of one run of GTA's lane splits


def words(valhalla_type: int) -> tuple[str, str] | None:
  """The Mapbox (type, modifier) for a Valhalla maneuver type; None for one with none (ferries, transit, elevators)."""
  t = AS_NAVD.get(valhalla_type, valhalla_type)
  return WORDS.get(t) or MERGES.get(t)


def street(m: dict) -> str:
  names = m.get("street_names") or m.get("begin_street_names") or []
  return names[0] if names else ""


def fold_keeps(maneuvers: list[dict], along, run: float = KEEP_RUN) -> list[dict]:
  """Runs of keeps each within `run` m of the one before as one: at the first's place, the last's type (its side) and
  road. `along(m)` is m along the route to a maneuver."""
  out: list[dict] = []
  last_keep = None  # (m along, index in out) of the run's latest keep
  for m in maneuvers:
    if m.get("type") in KEEPS:
      s = along(m)
      if last_keep is not None and s - last_keep[0] <= run:
        first = out[last_keep[1]]
        out[last_keep[1]] = {**m, "begin_shape_index": first["begin_shape_index"],
                             "exit_shape_index": m.get("begin_shape_index", 0), "folded": first.get("folded", 1) + 1}
        last_keep = (s, last_keep[1])
        continue
      out.append(m)
      last_keep = (s, len(out) - 1)
    else:
      out.append(m)
      last_keep = None
  return out


class ModelNav:
  """The model's maneuvers along one route, and allManeuvers / the navigation vector for the car's place on it.

  `route` is a router.Route (points in game metres, Valhalla maneuvers with begin_shape_index); `project` gives
  (lat, lon) for game metres, the bridge's map frame by default."""
  def __init__(self, route, project=to_lat_lon, junctions: bool = True, keep_run: float | None = KEEP_RUN):
    pts = np.asarray(route.points, dtype=np.float64)
    lat, lon = project(pts[:, 0], pts[:, 1])
    self.coordinates = np.column_stack((np.asarray(lon, np.float64), np.asarray(lat, np.float64)))  # navRoute's (lon, lat)
    self.tracker = RouteTracker(self.coordinates)
    n = len(pts)
    mans = junction_maneuvers(route) if junctions else list(route.maneuvers)
    if keep_run is not None:
      mans = fold_keeps(mans, lambda m: float(route.along[min(m.get("begin_shape_index", 0), n - 1)]), keep_run)
    self.maneuvers: list[tuple[float, str, str, str, int]] = []  # (m along, type, modifier, road, Valhalla type)
    for m in mans:
      w = words(m.get("type", 0))
      if w is None:
        continue
      i = min(m.get("begin_shape_index", 0), n - 1)
      self.maneuvers.append((float(self.tracker.distance[i]), w[0], w[1], street(m), int(m.get("type", 0))))

  def all_maneuvers(self, longitude: float, latitude: float) -> list[dict]:
    """navInstruction.allManeuvers for the car at (longitude, latitude), as navd sends them: those not yet passed."""
    progress = self.tracker.project(longitude, latitude)
    return [{"distance": float(d - progress), "type": t, "modifier": mod, "primaryText": road}
            for d, t, mod, road, _ in self.maneuvers if d >= progress]

  def all_maneuvers_at(self, xy, project=to_lat_lon) -> list[dict]:
    """all_maneuvers for the car at game metres xy."""
    lat, lon = project(float(xy[0]), float(xy[1]))
    return self.all_maneuvers(lon, lat)


def navigation(all_maneuvers: list[dict], traveled: float = 0.0) -> np.ndarray:
  """The model's navigation vector from allManeuvers' content, by comma's own navigation_input."""
  return navigation_input([SimpleNamespace(**m) for m in all_maneuvers], traveled)
