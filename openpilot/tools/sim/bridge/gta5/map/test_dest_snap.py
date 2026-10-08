"""Destinations on the road they face (dest_snap.py, and the router's use of it), on small hand-made maps. No pytest
needed: `python test_dest_snap.py` runs them all."""
import urllib.error

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.dest_snap import DestinationSnapper, _cross
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes
from openpilot.tools.sim.bridge.gta5.map.osm_pbf import OsmData
from openpilot.tools.sim.bridge.gta5.map.router import SIDE_DETOUR, Router

STREET = {'highway': 'residential', 'lanes': '2', 'width': '10'}
SERVICE = {'highway': 'service', 'lanes': '1', 'width': '4'}

# a two-way street north-south along x = 0; on its east side a drive to a dead end and a car park (a loop of aisles)
# off it, on its west an alley behind the buildings
NODES = {1: (0.0, -100.0), 2: (0.0, 0.0), 3: (0.0, 60.0), 4: (0.0, 120.0), 5: (-30.0, -100.0), 6: (-30.0, 120.0),
         21: (25.0, 0.0), 31: (30.0, 60.0), 32: (60.0, 60.0), 33: (60.0, 30.0), 34: (30.0, 30.0)}
WAYS = {
  10: (STREET, [1, 2]), 11: (STREET, [2, 3]), 12: (STREET, [3, 4]),
  20: ({**SERVICE, 'service': 'driveway'}, [2, 21]),
  30: ({'highway': 'service', 'service': 'parking_aisle', 'width': '6'}, [3, 31, 32, 33, 34, 31]),
  40: ({**SERVICE, 'service': 'alley'}, [5, 6]), 41: (STREET, [1, 5]), 42: (STREET, [4, 6]),
}


def make_map(nodes, ways, drive_on_right=True) -> OsmLanes:
  ids = np.array(sorted(nodes), np.int64)
  data = OsmData(ids, np.array([nodes[i][1] for i in ids]), np.array([nodes[i][0] for i in ids]), {}, dict(ways), {})
  return OsmLanes(data, lambda lat, lon: (lon, lat), drive_on_right=drive_on_right)


def street_map() -> DestinationSnapper:
  return DestinationSnapper(make_map(NODES, WAYS))


def test_driveway():
  # a building at the end of a drive faces the street: arrive northbound, the building on the right
  s = street_map().snap((32.0, 2.0))
  assert s.way in (10, 11) and s.highway == 'residential'
  assert abs(s.point[0]) < 1e-6 and abs(s.point[1]) <= 3.0
  assert s.heading == 0.0 and s.kerb_side and not s.as_is
  assert street_map().stub(int(np.flatnonzero(street_map().way == 20)[0]))


def test_alley():
  # nearer the alley behind than the street in front: the street, southbound to have the building on the right
  s = street_map().snap((-18.0, -50.0))
  assert s.way == 10 and s.heading == 180.0 and s.kerb_side


def test_on_a_car_park_aisle():
  # clearly in the car park, far from the street: the aisle it's on
  s = street_map().snap((45.0, 60.5))
  assert s.way == 30 and s.heading is None and s.as_is


def test_beside_a_car_park():
  # off the aisles, nearer the street than the aisles allow for: the street
  s = street_map().snap((14.0, 45.0))
  assert s.way == 11 and s.heading == 0.0


def test_no_side_in_a_car_park_or_past_a_dead_end():
  # beside an aisle, far from the street: the aisle, either way
  s = street_map().snap((45.0, 67.0))
  assert s.way == 30 and s.heading is None and not s.kerb_side
  # past the end of a long cul-de-sac (test_cul_de_sac): no side to arrive on
  nodes = {1: (0.0, -100.0), 2: (0.0, 0.0), 3: (0.0, 100.0), 4: (75.0, 0.0), 5: (150.0, 0.0)}
  s = DestinationSnapper(make_map(nodes, {1: (STREET, [1, 2, 3]), 2: (STREET, [2, 4, 5])})).snap((158.0, 3.0))
  assert s.way == 2 and s.heading is None


def test_named_service_road():
  # a street the map has as a service road (its name, no kind of service): nearer than a street, it wins
  nodes = {1: (0.0, -100.0), 2: (0.0, 100.0), 3: (40.0, -100.0), 4: (40.0, 100.0)}
  ways = {1: ({**SERVICE, 'name': 'Amarillo Vista'}, [1, 2]), 2: (STREET, [3, 4])}
  assert DestinationSnapper(make_map(nodes, ways)).snap((12.0, 0.0)).way == 1
  ways[1] = ({**SERVICE, 'name': 'Amarillo Vista', 'service': 'alley'}, [1, 2])
  assert DestinationSnapper(make_map(nodes, ways)).snap((12.0, 0.0)).way == 2


def test_on_the_street():
  s = street_map().snap((1.0, -50.0))
  assert s.way == 10 and s.heading is None and s.as_is and not s.kerb_side


def test_drive_on_left():
  snapper = DestinationSnapper(make_map(NODES, WAYS, drive_on_right=False))
  assert snapper.snap((32.0, 2.0)).heading == 180.0


def test_divided_road():
  # two one-way carriageways 10 m apart: the one on the waypoint's side, its way
  nodes = {1: (5.0, -100.0), 2: (5.0, 100.0), 3: (-5.0, 100.0), 4: (-5.0, -100.0)}
  oneway = {'highway': 'primary', 'oneway': 'yes', 'lanes': '2', 'width': '8'}
  snapper = DestinationSnapper(make_map(nodes, {1: (oneway, [1, 2]), 2: (oneway, [3, 4])}))
  east, west = snapper.snap((25.0, 0.0)), snapper.snap((-25.0, 0.0))
  assert east.way == 1 and east.heading == 0.0 and not east.kerb_side
  assert west.way == 2 and west.heading == 180.0


def test_motorway_beside():
  # a freeway nearer than the street: nobody stops on it
  nodes = {1: (0.0, -100.0), 2: (0.0, 100.0), 3: (40.0, -100.0), 4: (40.0, 100.0)}
  ways = {1: ({'highway': 'motorway', 'lanes': '2', 'width': '11'}, [1, 2]), 2: (STREET, [3, 4])}
  s = DestinationSnapper(make_map(nodes, ways)).snap((14.0, 0.0))
  assert s.way == 2 and s.heading == 180.0  # west of a northbound street: arrive southbound


def test_cul_de_sac():
  # a house past the end of a long cul-de-sac: the cul-de-sac, not the far street it leads off
  nodes = {1: (0.0, -100.0), 2: (0.0, 0.0), 3: (0.0, 100.0), 4: (75.0, 0.0), 5: (150.0, 0.0)}
  ways = {1: (STREET, [1, 2, 3]), 2: (STREET, [2, 4, 5])}
  snapper = DestinationSnapper(make_map(nodes, ways))
  s = snapper.snap((158.0, 0.0))
  assert s.way == 2 and s.point[0] > 140.0
  assert not snapper.stub(int(np.flatnonzero(snapper.way == 2)[-1]))


def test_road_behind():
  # a short dead-end street between the waypoint and a through road: the one it faces, not the one behind it
  nodes = {1: (0.0, -30.0), 2: (0.0, 30.0), 3: (-20.0, -100.0), 4: (-20.0, -30.0), 5: (-20.0, 100.0)}
  ways = {1: (STREET, [1, 2]), 2: (STREET, [4, 1]), 3: (STREET, [3, 4, 5])}
  snapper = DestinationSnapper(make_map(nodes, ways))
  assert snapper.stub(int(np.flatnonzero(snapper.way == 1)[0]))
  assert snapper.snap((12.0, 0.0)).way == 1
  assert _cross(np.array([12.0, 0.0]), np.array([-20.0, 0.0]), np.array([0.0, -30.0]), np.array([0.0, 30.0]))
  assert not _cross(np.array([12.0, 0.0]), np.array([-20.0, 0.0]), np.array([0.0, 5.0]), np.array([0.0, 30.0]))


def test_nothing_near():
  assert street_map().snap((500.0, 500.0)) is None


def encode(points, precision=6) -> str:
  """Google's encoded polyline of game points, as Valhalla returns."""
  out, last = [], (0, 0)
  for x, y in points:
    lat, lon = to_lat_lon(x, y)
    cur = (round(lat * 10 ** precision), round(lon * 10 ** precision))
    for v in (cur[0] - last[0], cur[1] - last[1]):
      v = ~(v << 1) if v < 0 else v << 1
      while v >= 0x20:
        out.append(chr((0x20 | (v & 0x1f)) + 63))
        v >>= 5
      out.append(chr(v + 63))
    last = cur
  return ''.join(out)


class FakeRouter(Router):
  """Answers routes from canned trips by the destination asked for, and keeps the requests."""
  def __init__(self, roads, answer):
    super().__init__('http://unused', roads=roads)
    self.answer, self.requests = answer, []

  def _post(self, action, request):
    if action != 'route':
      return {}
    self.requests.append(request['locations'][-1])
    found = self.answer(request['locations'][-1])
    if found is None:
      raise urllib.error.HTTPError('fake', 400, 'no path', None, None)
    time, end = found
    return {'trip': {'summary': {'time': time}, 'legs': [{'shape': encode([(0.0, -300.0), end]), 'maneuvers': []}]}}


def test_router_arrives_kerb_side():
  roads = make_map(NODES, WAYS)
  r = FakeRouter(roads, lambda loc: (100.0, (0.0, 1.0)))
  route = r.route(np.array([0.0, -300.0]), 0.0, np.array([32.0, 2.0]))
  first = r.requests[0]
  assert first['heading'] == 0 and first['node_snap_tolerance'] == 0
  assert 'heading' not in r.requests[1]  # either way, to compare
  assert len(r.requests) == 2 and np.allclose(route.points[-1], (0.0, 1.0), atol=0.1)


def test_router_takes_either_way_round_a_long_detour():
  roads = make_map(NODES, WAYS)
  r = FakeRouter(roads, lambda loc: (100.0 + SIDE_DETOUR + 1, (0.0, 3.0)) if 'heading' in loc else (100.0, (0.0, -3.0)))
  route = r.route(np.array([0.0, -300.0]), 0.0, np.array([32.0, 2.0]))
  assert np.allclose(route.points[-1], (0.0, -3.0), atol=0.1)


def test_router_falls_back():
  # no route arriving that way (as past a one-way's wrong end), then none to the road at all: the nearest road
  roads = make_map(NODES, WAYS)
  r = FakeRouter(roads, lambda loc: None if 'node_snap_tolerance' in loc else (100.0, (25.0, 0.0)))
  route = r.route(np.array([0.0, -300.0]), 0.0, np.array([32.0, 2.0]))
  assert len(r.requests) == 3 and 'heading' in r.requests[0] and 'heading' not in r.requests[1]
  assert 'node_snap_tolerance' not in r.requests[2] and np.allclose(route.points[-1], (25.0, 0.0), atol=0.1)


def test_router_on_road_as_before():
  roads = make_map(NODES, WAYS)
  r = FakeRouter(roads, lambda loc: (100.0, (1.0, -50.0)))
  r.route(np.array([0.0, -300.0]), 0.0, np.array([1.0, -50.0]))
  lat, lon = to_lat_lon(1.0, -50.0)
  assert r.requests == [{'lat': lat, 'lon': lon}]


def test_router_without_roads():
  r = FakeRouter(None, lambda loc: (100.0, (25.0, 0.0)))
  r.route(np.array([0.0, -300.0]), 0.0, np.array([32.0, 2.0]))
  assert len(r.requests) == 1 and 'heading' not in r.requests[0]


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
