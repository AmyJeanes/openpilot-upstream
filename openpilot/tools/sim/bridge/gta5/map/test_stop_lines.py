"""Nav's stop lines from the map's (stop_lines.py, router.Route.stops): on small maps made here, and on the GTA map where
it's built (GTA5_MAP, default ~/gta5map_lanes; skipped without it). No pytest needed: `python test_stop_lines.py` runs
them all."""
import heapq
import json
import os
import tempfile

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.junctions import Junctions
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes, oneway_of
from openpilot.tools.sim.bridge.gta5.map.osm_pbf import OsmData
from openpilot.tools.sim.bridge.gta5.map.paths import Paths
from openpilot.tools.sim.bridge.gta5.map.router import Route
from openpilot.tools.sim.bridge.gta5.map.stop_lines import StopLines

TWO_WAY = {'highway': 'residential', 'lanes': '2', 'width': '11'}
LANE = {'highway': 'residential', 'lanes': '1', 'oneway': 'yes', 'width': '4'}
LIVE_MAP = os.path.expanduser(os.path.join(os.getenv("GTA5_MAP", "~/gta5map_lanes"), "gta5.osm.pbf"))


def make(nodes: dict, ways: dict, node_tags: dict | None = None) -> OsmLanes:
  """A map in metres: nodes {id: (x, y)}, ways {id: (tags, [node ids])}."""
  ids = np.array(sorted(nodes), np.int64)
  data = OsmData(ids, np.array([nodes[i][1] for i in ids], float), np.array([nodes[i][0] for i in ids], float),
                 node_tags or {}, dict(ways), {})
  return OsmLanes(data, lambda lat, lon: (lon, lat))


# a crossroads at node 1, (0, 0), with traffic lights 20 m south of it for traffic heading north
NODES = {1: (0.0, 0.0), 2: (0.0, -20.0), 3: (0.0, -100.0), 4: (-100.0, 0.0), 5: (100.0, 0.0), 6: (0.0, 100.0)}
WAYS = {1: (TWO_WAY, [3, 2]), 2: (TWO_WAY, [2, 1]), 3: (TWO_WAY, [1, 4]), 4: (TWO_WAY, [1, 5]), 5: (TWO_WAY, [1, 6])}
SIGNALS = {2: {'highway': 'traffic_signals', 'traffic_signals:direction': 'forward'}}


def route(osm: OsmLanes, points) -> Route:
  return Route(np.array(points, float), [], osm=osm)


def stop_along(osm: OsmLanes) -> float:
  """m out from the junction's node to its one stop line, as junctions.py draws it."""
  (stop,) = [s for j in Junctions(osm).junctions for s in j.stops]
  return stop.along


def test_stop_line_its_own_way_only():
  osm = make(NODES, WAYS, SIGNALS)
  along = stop_along(osm)
  north = route(osm, [NODES[i] for i in (3, 2, 1, 6)])
  assert np.allclose(north.stops, [100.0 - along], atol=0.05) and north.stop_kinds == ['lights'], (north.stops, along)
  turn = route(osm, [NODES[i] for i in (3, 2, 1, 5)])  # turning off right from the same approach
  assert np.allclose(turn.stops, [100.0 - along], atol=0.05)
  for ids in ((6, 1, 2, 3), (5, 1, 2, 3), (5, 1, 4), (4, 1, 6)):  # leaving past it, or never on its approach
    assert route(osm, [NODES[i] for i in ids]).stops == [], ids


def test_info_carries_kinds():
  osm = make(NODES, WAYS, {2: {'highway': 'stop', 'direction': 'forward'}})
  r = route(osm, [NODES[i] for i in (3, 2, 1, 6)])
  r.locate(np.array([0.0, -60.0]))
  info = r.info(300.0)
  assert len(info['stops']) == 1 and info['stopKinds'] == ['stop'] and abs(info['stops'][0] - (60.0 - stop_along(osm))) < 0.1


def test_route_starts_or_ends_beside_the_line():
  # the lights' node 3 m from the junction's node, inside its area: the line goes out at its mouth, part way along the
  # link in from node 3
  nodes = {**NODES, 2: (0.0, -3.0)}
  osm = make(nodes, WAYS, SIGNALS)
  along = stop_along(osm)
  assert 3.0 < along < 100.0, along
  line = -along
  before, past = line - 5.0, line + 1.0
  r = route(osm, [(0.0, before), nodes[2], nodes[1], nodes[6]])  # the car short of it
  assert np.allclose(r.stops, [5.0], atol=0.05), r.stops
  assert route(osm, [(0.0, past), nodes[2], nodes[1], nodes[6]]).stops == []  # the car past it
  r = route(osm, [nodes[3], (0.0, past)])  # ending just past it
  assert np.allclose(r.stops, [100.0 + line], atol=0.05), r.stops
  assert route(osm, [nodes[3], (0.0, before)]).stops == []  # ending short of it


def lanes_map(stop_nodes: dict) -> OsmLanes:
  """Two one-way lanes side by side, 4 m apart, into a junction at node 1 (2, 0), each its own link as GTA lays some
  approaches, with roads on west, east and north; stop_nodes {node: tags} on the lanes' nodes 11 (0, -20) and 21
  (4, y)."""
  y21 = -27.0 if 21 in stop_nodes else -20.0
  nodes = {1: (2.0, 0.0), 10: (0.0, -100.0), 11: (0.0, -20.0), 20: (4.0, -100.0), 21: (4.0, y21), 4: (-100.0, 0.0),
           5: (100.0, 0.0), 6: (2.0, 100.0)}
  ways = {1: (LANE, [10, 11, 1]), 2: (LANE, [20, 21, 1]), 3: (TWO_WAY, [1, 4]), 4: (TWO_WAY, [1, 5]), 5: (TWO_WAY, [1, 6])}
  return make(nodes, ways, stop_nodes)


def line_y(osm: OsmLanes, node: int) -> float:
  (stop,) = [s for j in Junctions(osm).junctions for s in j.stops if s.node == node]
  return float(stop.line[:, 1].mean())


def test_lane_beside_the_lines_lane():
  # the line drawn across the left lane only reaches the right lane, which has none of its own: a route there stops at it
  tags = {'highway': 'traffic_signals', 'traffic_signals:direction': 'forward'}
  osm = lanes_map({11: tags})
  y = line_y(osm, 11)
  for start, via in (((0.0, -100.0), (0.0, -20.0)), ((4.0, -100.0), (4.0, -20.0))):
    r = route(osm, [start, via, (2.0, 0.0), (2.0, 100.0)])
    assert len(r.stops) == 1 and abs(r.stops[0] - (y + 100.0)) < 0.3, (start, r.stops, y)


def test_staggered_lines_per_lane():
  # each lane its own line, 7 m apart: a route takes its own lane's, not the other's reaching across it
  tags = {'highway': 'traffic_signals', 'traffic_signals:direction': 'forward'}
  osm = lanes_map({11: tags, 21: tags})
  for start, via, node in (((0.0, -100.0), (0.0, -20.0), 11), ((4.0, -100.0), (4.0, -27.0), 21)):
    r = route(osm, [start, via, (2.0, 0.0), (2.0, 100.0)])
    assert len(r.stops) == 1 and abs(r.stops[0] - (line_y(osm, node) + 100.0)) < 0.3, (node, r.stops)


def gta_paths(path: str, nodes: dict, links: list, stops: dict, junctions=()):
  """A paths.jsonl of GTA nodes {id: (x, y)} and one-lane links [(a, b)] each way, stop line nodes {id: special}."""
  with open(path, 'w') as f:
    for i, (x, y) in nodes.items():
      f.write(json.dumps({'t': 'n', 'a': 0, 'i': i, 'x': x, 'y': y, 'z': 0.0, 'st': '',
                          'f': [0, stops.get(i, 0) << 3, 4 if i in junctions else 0]}) + '\n')
    for a, b in links:
      for p, q in ((a, b), (b, a)):
        f.write(json.dumps({'t': 'l', 'a': 0, 'i': p, 'ta': 0, 'ti': q, 'f': [0, 0, 1 << 5 | 1 << 2]}) + '\n')


def test_gta_stop_nodes_only_without_lane_tags():
  # GTA's stop line nodes: its lights' at node 2 and a stop junction's at node 7, 20 m north of the junction, facing
  # nothing on the way north (its node doesn't say which way it faces)
  nodes = {**NODES, 7: (0.0, 20.0)}
  with tempfile.TemporaryDirectory() as d:
    gta_paths(os.path.join(d, 'paths.jsonl'), nodes, [(3, 2), (2, 1), (1, 4), (1, 5), (1, 7), (7, 6)], {2: 15, 7: 16}, {1})
    paths = Paths(os.path.join(d, 'paths.jsonl'))
  points = np.array([nodes[i] for i in (3, 2, 1, 7, 6)], float)
  gta = Route(points, [], paths)
  assert gta.stops == [80.0, 120.0] and gta.stop_kinds == ['lights', 'stop']
  ways = {1: (TWO_WAY, [3, 2]), 2: (TWO_WAY, [2, 1]), 3: (TWO_WAY, [1, 4]), 4: (TWO_WAY, [1, 5]), 5: (TWO_WAY, [1, 7, 6])}
  tagged = Route(points, [], paths, osm=make(nodes, ways, SIGNALS))
  assert len(tagged.stops) == 1 and tagged.stop_kinds == ['lights'] and tagged.stops[0] < 80.0 + 1e-6


def test_cache_round_trip():
  osm = make(NODES, WAYS, SIGNALS)
  lines = StopLines.cached(osm)
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'stop_lines.npz')
    lines.save(path)
    back = StopLines.load(path)
  pts = [NODES[i] for i in (3, 2, 1, 6)]
  along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(np.array(pts), axis=0).T))))
  assert back.on_route(pts, along, osm) == lines.on_route(pts, along, osm) and len(back) == len(lines)


# *** on the GTA map ***

_live: list = []


def live() -> OsmLanes | None:
  if not _live:
    from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
    _live.append(OsmLanes.load(LIVE_MAP, to_game) if os.path.exists(LIVE_MAP) else None)
  return _live[0]


def drive(osm: OsmLanes, start, end) -> np.ndarray:
  """The shortest way along the map's links, driven only their legal way, from the node nearest start to the node
  nearest end: its nodes' places, as a route's shape points."""
  def nearest(p):
    return int(osm.ids[int(np.argmin(np.hypot(*(osm.xy - np.asarray(p, float)).T)))])
  a, b = nearest(start), nearest(end)
  dist, prev, heap = {a: 0.0}, {}, [(0.0, a)]
  while heap:
    d, i = heapq.heappop(heap)
    if i == b:
      break
    if d > dist[i]:
      continue
    for j in osm.links.get(i, ()):
      w, fwd = osm.pairs[(i, j)]
      one = oneway_of(osm.ways[w][0])
      if one and (one == 1) != fwd:
        continue
      nd = d + float(np.hypot(*(osm.node_xy(j) - osm.node_xy(i))))
      if nd < dist.get(j, np.inf):
        dist[j], prev[j] = nd, i
        heapq.heappush(heap, (nd, j))
  path = [b]
  while path[-1] != a:
    path.append(prev[path[-1]])
  return np.array([osm.node_xy(i) for i in path[::-1]])


def stops_on(osm: OsmLanes, start, end) -> list[tuple[np.ndarray, str]]:
  r = Route(drive(osm, start, end), [], osm=osm)
  return [(np.array([np.interp(s, r.along, r.points[:, 0]), np.interp(s, r.along, r.points[:, 1])]), k)
          for s, k in zip(r.stops, r.stop_kinds, strict=True)]


def test_tongva_dr():
  # GTA's stop line node at (-1361, 2149) faced the junction 40 m east; the paint is across the westbound lanes at the
  # junction west's mouth, where the map's line is: none eastbound, one westbound on the paint
  osm = live()
  if osm is None:
    print('test_tongva_dr skipped: no GTA map')
    return
  east = stops_on(osm, (-1388.0, 2147.5), (-1337.2, 2150.7))
  assert east == [], east
  west = stops_on(osm, (-1337.2, 2150.7), (-1388.0, 2147.5))
  assert len(west) == 1 and west[0][1] == 'stop' and abs(west[0][0][0] + 1362.3) < 1.0, west


def test_zancudo_rd():
  # a T junction at (-1678, 2435) with a stop line on Zancudo Rd's northbound approach from the south: none for a
  # route turning south out of the junction past it (GTA's node counted both ways), one northbound
  osm = live()
  if osm is None:
    print('test_zancudo_rd skipped: no GTA map')
    return
  south = stops_on(osm, (-1696.5, 2437.3), (-1678.5, 2410.0))
  assert south == [], south
  north = stops_on(osm, (-1678.5, 2410.0), (-1659.3, 2431.5))
  assert len(north) == 1 and north[0][1] == 'stop' and abs(north[0][0][1] - 2424.6) < 1.0, north


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
