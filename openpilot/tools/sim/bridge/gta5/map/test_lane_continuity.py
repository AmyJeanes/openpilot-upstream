"""Lanes carry on without a step where a road's lane count changes at a node (osm_lanes taper_u): the lanes move across
over a taper centred on the node, the new lane widening from nothing on the side it opens (the lane ending narrowing to
nothing), the lanes carrying on counted from the other side; nav's cross-sections (RouteLanes.section_at), the lines
drawn (OsmLanes.line_geometry) and the map driver's target agree. No pytest needed: `python test_lane_continuity.py`
(the live map's cases only with ~/gta5map_lanes)."""
import os

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.osm_lanes import DROP_AFTER, DROP_BEFORE, TAPER_AFTER, TAPER_BEFORE, TAPER_M, OsmLanes, \
  RouteLanes, ways_from_nodes
from openpilot.tools.sim.bridge.gta5.map.osm_pbf import OsmData

NODES = {1: (0.0, 0.0), 2: (0.0, 100.0), 3: (0.0, 200.0), 4: (0.0, 300.0)}
TWO = {'highway': 'primary', 'lanes': '2', 'oneway': 'yes', 'width': '11'}  # GTA lays each link's lanes about its line
THREE = {'highway': 'primary', 'lanes': '3', 'oneway': 'yes', 'width': '16.5'}
LEFT_BAY = {**THREE, 'turn:lanes': 'left|through|through'}  # the lane it gains only turns left: it opens on the left


def osm_of(ways: dict) -> OsmLanes:
  ids = np.array(sorted(NODES), np.int64)
  data = OsmData(ids, np.array([NODES[i][1] for i in ids]), np.array([NODES[i][0] for i in ids]), {}, dict(ways), {})
  return OsmLanes(data, lambda lat, lon: (np.asarray(lon, float), np.asarray(lat, float)))


def route_lanes(osm: OsmLanes, ids=(1, 2, 3, 4)) -> RouteLanes:
  pts = np.array([NODES[i] for i in ids], float)
  return RouteLanes.from_osm(pts, ways_from_nodes(pts, osm), osm)


def lane_centres(lanes: RouteLanes, s: np.ndarray, node: float) -> list[np.ndarray]:
  """The centres of the lanes the fewer and the more share, along s (m right of the line), each as the route numbers
  it: of the fewer lanes' count before the node and the more's after (section_at at each s)."""
  out = []
  for v in s:
    k = lanes.segment(v)
    sec = lanes.section_at(float(v), k)
    out.append([sp.centre for sp in sec.ours])
  return out


def carried(lanes: RouteLanes, s: np.ndarray, before: int, extra_left: bool, opening: bool) -> np.ndarray:
  """[len(s), lanes carried on] centres: in the section with more lanes, those but the extra one(s)."""
  rows = []
  for c in lane_centres(lanes, s, 0.0):
    if len(c) == before:
      rows.append(c)
    else:
      extra = len(c) - before
      rows.append(c[extra:] if extra_left else c[:before])
  return np.array(rows)


def check(osm: OsmLanes, n_few: int, extra_left: bool, opening: bool):
  lanes = route_lanes(osm)
  node = 100.0  # where the lane count changes
  s = np.arange(1.0, 299.0, 0.25)
  c = carried(lanes, s, n_few, extra_left, opening)
  # continuous: no step anywhere (a lane moving 5.5 m / 2 across 12 m moves under 0.15 m in 0.25 m)
  assert np.abs(np.diff(c, axis=0)).max() < 0.15, np.abs(np.diff(c, axis=0)).max()
  before, after = (TAPER_BEFORE, TAPER_AFTER) if opening else (DROP_BEFORE, DROP_AFTER)
  moving = np.abs(np.diff(c, axis=0)).max(axis=1) > 1e-6
  where = s[1:][moving]
  assert not len(where) or (where.min() >= node - before - 0.3 and where.max() <= node + after + 0.3), where
  return lanes, s, c


def test_lane_opens_on_the_left():
  # 2 lanes, then 3 with a left-turn bay on the left: the two through lanes carry on as lanes 1 and 2, moving across
  # from where the 2-lane way has them to where the 3-lane way has them over the taper; the bay widens from nothing
  osm = osm_of({10: (TWO, [1, 2]), 11: (LEFT_BAY, [2, 3]), 12: (LEFT_BAY, [3, 4])})
  lanes, s, c = check(osm, 2, True, True)
  assert np.allclose(c[s < 90.0][0], [-2.75, 2.75]) and np.allclose(c[s > 110.0][-1], [0.0, 5.5])
  bay = [lanes.section_at(v, lanes.segment(v)).ours for v in (100.0 - TAPER_BEFORE + 0.5, 100.0, 100.0 + TAPER_AFTER)]
  assert len(bay[0]) == 2 and abs(bay[1][0].right - bay[1][0].left - 2.75) < 0.01 and abs(bay[2][0].right - bay[2][0].left - 5.5) < 0.01
  assert lanes.opening(1) == (100.0 + TAPER_M, 1, True)


def test_lane_opens_on_the_right():
  # no arrows: the lane it gains opens on the outside, the right; the two lanes carry on as lanes 0 and 1
  osm = osm_of({10: (TWO, [1, 2]), 11: (THREE, [2, 3]), 12: (THREE, [3, 4])})
  lanes, s, c = check(osm, 2, False, True)
  assert np.allclose(c[s < 90.0][0], [-2.75, 2.75]) and np.allclose(c[s > 110.0][-1], [-5.5, 0.0])
  assert lanes.opening(1) == (100.0 + TAPER_M, 1, False)


def test_lane_ends_on_the_right():
  # 3 lanes, then 2: the kerb lane narrows to nothing from DROP_BEFORE before the node to DROP_AFTER past it, the two
  # lanes left moving across to where the 2-lane way has them
  osm = osm_of({10: (THREE, [1, 2]), 11: (TWO, [2, 3]), 12: (TWO, [3, 4])})
  lanes, s, c = check(osm, 2, False, False)
  assert np.allclose(c[s < 90.0][0], [-5.5, 0.0]) and np.allclose(c[s > 110.0][-1], [-2.75, 2.75])
  end = lanes.section_at(100.0 + DROP_AFTER - 0.01, 0)
  assert end.lanes == 3 and end.ours[-1].right - end.ours[-1].left < 0.01


def test_two_way_road_gains_a_lane_our_way():
  # one lane each way, then two our way and one back (a lane opening on the right of ours): the oncoming lane stays
  # put, ours carries on as lane 0
  one_one = {'highway': 'primary', 'lanes': '2', 'lanes:forward': '1', 'lanes:backward': '1', 'width': '11'}
  two_one = {'highway': 'primary', 'lanes': '3', 'lanes:forward': '2', 'lanes:backward': '1', 'width': '16.5',
             'placement:forward': 'left_of:1', 'placement:backward': 'left_of:1'}
  osm = osm_of({10: (one_one, [1, 2]), 11: (two_one, [2, 3]), 12: (two_one, [3, 4])})
  lanes, s, c = check(osm, 1, False, True)
  assert np.allclose(c[s < 90.0][0], [2.75]) and np.allclose(c[s > 110.0][-1], [2.75])
  onc = [lanes.section_at(v, lanes.segment(v)).spans[0].centre for v in s]
  assert np.allclose(onc, -2.75)


def test_lines_meet_at_the_node():
  # lanes.json's lines: the way before and after a lane opening each take their half of the taper, so every line the
  # two share meets at the node, and the kerb on the side that opens moves out across it
  osm = osm_of({10: (TWO, [1, 2]), 11: (THREE, [2, 3]), 12: (THREE, [3, 4])})
  a = {(line.kind, round(line.offset, 2)): g for line, g in osm.line_geometry(10)}
  b = {(line.kind, round(line.offset, 2)): g for line, g in osm.line_geometry(11)}
  a_end = sorted(g[-1, 0] for g in a.values())
  b_start = sorted(g[0, 0] for g in b.values())
  for x in a_end:  # each of the way before's lines ends where one of the way after's starts
    assert min(abs(x - y) for y in b_start) < 0.01, (x, b_start)
  edges = sorted(g[:, 0].max() for (kind, _), g in a.items() if kind == 'edge')
  assert abs(edges[-1] - (5.5 + 8.25) / 2) < 0.01  # the right kerb half way out at the node


def test_target_right_follows_section_at():
  # MapDrive's target_right is the target lane's centre where the plan is, as section_at has it: no step at the node
  from openpilot.tools.sim.bridge.gta5.gta5_mapdrive import target_centre
  from openpilot.tools.sim.bridge.gta5.map.router import Route
  from openpilot.tools.sim.bridge.gta5.mapdrive_sim import DEST, START
  osm = osm_of({10: (TWO, [1, 2]), 11: (THREE, [2, 3]), 12: (THREE, [3, 4])})
  pts = np.array([NODES[i] for i in (1, 2, 3, 4)], float)
  r = Route(pts, [{'type': START, 'begin_shape_index': 0}, {'type': DEST, 'begin_shape_index': 3}], osm=osm)
  s = np.arange(80.0, 120.0, 0.5)
  rights = [target_centre(r, v, 0.0) for v in s]  # lane 0 is the same lane both sides: it opens on the right
  assert np.abs(np.diff(rights)).max() < 0.3 and rights[0] == -2.75 and rights[-1] == -5.5, rights


# *** freeway exits, on-ramps and lane chains (RouteLanes._diverges) ***

MOTORWAY3 = {'highway': 'motorway', 'lanes': '3', 'oneway': 'yes', 'width': '18'}  # lanes at -6, 0, 6
RAMP = {'highway': 'motorway_link', 'lanes': '1', 'oneway': 'yes', 'width': '6'}


def freeway(ways: dict, nodes: dict, path: str | None = None) -> OsmLanes:
  ids = np.array(sorted(nodes), np.int64)
  data = OsmData(ids, np.array([nodes[i][1] for i in ids]), np.array([nodes[i][0] for i in ids]), {}, dict(ways), {})
  osm = OsmLanes(data, lambda lat, lon: (np.asarray(lon, float), np.asarray(lat, float)))
  osm.path = path
  return osm


EXIT_NODES = {1: (0.0, 0.0), 2: (0.0, 200.0), 3: (0.0, 400.0), 4: (8.0, 240.0), 5: (30.0, 300.0), 6: (60.0, 380.0)}
EXIT_WAYS = {10: (MOTORWAY3, [1, 2]), 11: (MOTORWAY3, [2, 3]), 12: (RAMP, [2, 4]), 13: (RAMP, [4, 5]), 14: (RAMP, [5, 6])}


def exit_lane(osm, ids=(1, 2, 4, 5, 6)):
  pts = np.array([EXIT_NODES[i] for i in ids], float)
  lanes = RouteLanes.from_osm(pts, ways_from_nodes(pts, osm), osm)
  out = []
  for v in np.arange(200.5, lanes.along[-1] - 0.5, 0.5):
    k = lanes.segment(v)
    sec = lanes.section_at(float(v), k)
    p = np.array([np.interp(v, lanes.along, pts[:, 0]), np.interp(v, lanes.along, pts[:, 1])])
    d = (pts[k + 1] - pts[k]) / np.hypot(*(pts[k + 1] - pts[k]))
    out.append(p + np.array([d[1], -d[0]]) * sec.ours[0].centre)
  return lanes, np.array(out)


def test_exit_leaves_from_the_kerb_lane():
  # GTA's ramp link starts at the motorway's middle node and angles out across its lanes: the exit lane runs on as the
  # kerb lane (6 m right of the line) until the ramp's own line comes out past it, then follows the ramp, never
  # swinging in towards the motorway's middle
  osm = freeway(EXIT_WAYS, EXIT_NODES)
  lanes, line = exit_lane(osm)
  assert lanes.explicit, 'no diverge'
  assert line[:, 0].min() > 5.9, line[:, 0].min()  # never inside the kerb lane's middle
  assert abs(line[0, 0] - 6.0) < 0.05, line[0]  # from the kerb lane at the node
  assert np.all(np.diff(line[line[:, 1] < 300.0, 0]) > -0.01)  # and only ever out from it
  far = line[-1]
  assert np.hypot(*(far - EXIT_NODES[6])) < 1.0  # on the ramp's own lane by its end
  steps = np.hypot(*np.diff(line, axis=0).T)
  assert steps.max() < 1.0, steps.max()  # no step


def test_exit_kept_off_a_hatched_gore():
  # a hatched gore painted where the moved lane would run (lanes.json beside the map): the branch is left as it was
  import json
  import tempfile
  with tempfile.TemporaryDirectory() as d:
    ring = [[4.0, 210.0], [9.0, 210.0], [9.0, 260.0], [4.0, 260.0], [4.0, 210.0]]
    json.dump({'kinds': ['hatched'], 'lines': [[0] + [v for p in ring for v in p]], 'heights': [[0.0] * len(ring)],
               'layers': [0]}, open(os.path.join(d, 'lanes.json'), 'w'))
    from openpilot.tools.sim.bridge.gta5.map.router import LaneLines
    LaneLines._loaded.clear()
    osm = freeway(EXIT_WAYS, EXIT_NODES, os.path.join(d, 'gta5.osm.pbf'))
    lanes, _ = exit_lane(osm)
    LaneLines._loaded.clear()
  assert not lanes.explicit


def test_on_ramp_joins_as_the_kerb_lane():
  # the mirror: a ramp link ending at the motorway's middle node; its lane comes in to the kerb lane and runs on in it
  nodes = {1: (60.0, 20.0), 2: (30.0, 100.0), 3: (8.0, 160.0), 4: (0.0, 200.0), 5: (0.0, 400.0), 6: (0.0, 0.0)}
  ways = {10: (RAMP, [1, 2]), 11: (RAMP, [2, 3]), 12: (RAMP, [3, 4]), 13: (MOTORWAY3, [6, 4]), 14: (MOTORWAY3, [4, 5])}
  osm = freeway(ways, nodes)
  pts = np.array([nodes[i] for i in (1, 2, 3, 4, 5)], float)
  lanes = RouteLanes.from_osm(pts, ways_from_nodes(pts, osm), osm)
  assert lanes.explicit
  xs = []
  for v in np.arange(0.5, lanes.along[-1] - 0.5, 0.5):
    k = lanes.segment(v)
    sec = lanes.section_at(float(v), k)
    p = np.array([np.interp(v, lanes.along, pts[:, 0]), np.interp(v, lanes.along, pts[:, 1])])
    d = (pts[k + 1] - pts[k]) / np.hypot(*(pts[k + 1] - pts[k]))
    q = p + np.array([d[1], -d[0]]) * (sec.ours[-1].centre if sec.lanes > 1 else sec.ours[0].centre)
    xs.append(q)
  xs = np.array(xs)
  assert xs[:, 0].min() > 5.9, xs[:, 0].min()  # it never cuts in past the kerb lane's middle towards the middle node
  assert np.hypot(*np.diff(xs, axis=0).T).max() < 1.0


def test_chain_split_runs_straight_on():
  # a 3-lane link splitting at its middle node into a 2-lane link and GTA's 1-lane chain alongside on the right (as
  # 402:719 -> 738 into 738 -> 762 and 738 -> 763): the chain's lane runs straight on as the right lane, from where
  # it is, not from the middle node
  nodes = {1: (0.0, 0.0), 2: (0.0, 100.0), 3: (-3.0, 160.0), 4: (2.0, 140.0), 5: (4.0, 200.0), 6: (5.0, 300.0)}
  two = {'highway': 'motorway', 'lanes': '2', 'oneway': 'yes', 'width': '12'}
  ways = {10: (MOTORWAY3, [1, 2]), 11: (two, [2, 3]), 12: (RAMP, [2, 4]), 13: (RAMP, [4, 5]), 14: (RAMP, [5, 6])}
  osm = freeway(ways, nodes)
  pts = np.array([nodes[i] for i in (1, 2, 4, 5, 6)], float)
  lanes = RouteLanes.from_osm(pts, ways_from_nodes(pts, osm), osm)
  xs = []
  for v in np.arange(80.0, lanes.along[-1] - 0.5, 0.5):
    k = lanes.segment(v)
    sec = lanes.section_at(float(v), k)
    p = np.array([np.interp(v, lanes.along, pts[:, 0]), np.interp(v, lanes.along, pts[:, 1])])
    d = (pts[k + 1] - pts[k]) / np.hypot(*(pts[k + 1] - pts[k]))
    xs.append(p + np.array([d[1], -d[0]]) * sec.ours[-1].centre)
  xs = np.array(xs)
  first = xs[xs[:, 1] < 140.0]  # (to the chain's last link within DIVERGE_M, over which it eases back onto its own)
  assert np.abs(first[:, 0] - 6.0).max() < 0.3, first[:, 0]  # the right lane, straight on through the split
  assert np.hypot(*np.diff(xs, axis=0).T).max() < 1.0


LIVE_MAP = os.path.expanduser("~/gta5map_lanes")
# real lane openings and drops on the live map (route points: its nodes), each a place the 16 Oct drives passed
LIVE_PLACES = {
  'open_2_3_left_-460_-221': [(-494.25, -171.0), (-488.25, -181.25), (-482.25, -191.75), (-476.25, -202.25), (-470.25, -212.5),
                              (-462.0, -221.5), (-457.0, -230.25), (-447.25, -247.75), (-440.25, -264.5), (-433.26, -274.5)],
  'open_1_2_right_644_-1244': [(575.76, -1227.75), (590.75, -1231.51), (601.75, -1234.0), (614.49, -1236.75), (626.25, -1239.75),
                               (633.5, -1241.5), (641.5, -1243.75), (652.0, -1246.26), (669.0, -1246.5), (682.25, -1245.75)],
  'drop_2_1_1183_434': [(1156.25, 400.51), (1164.25, 409.25), (1172.49, 418.25), (1180.75, 427.75), (1184.5, 432.5),
                        (1192.25, 442.75), (1196.25, 447.5), (1199.5, 452.5), (1202.75, 457.24), (1206.0, 461.25)],
  'open_1_2_left_-74_-1552': [(-34.25, -1585.25), (-40.5, -1579.5), (-47.0, -1573.5), (-55.75, -1566.0), (-64.75, -1558.75),
                              (-74.75, -1552.75), (-83.75, -1545.25), (-92.75, -1537.75), (-103.0, -1529.0)],
}


def test_live_tapers():
  if not os.path.exists(os.path.join(LIVE_MAP, "gta5.osm.pbf")) or not LIVE_PLACES:
    print("skipped: no live map")
    return
  from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
  osm = OsmLanes.load(os.path.join(LIVE_MAP, "gta5.osm.pbf"), to_game)
  for name, pts in LIVE_PLACES.items():
    pts = np.asarray(pts, float)
    lanes = RouteLanes.from_osm(pts, ways_from_nodes(pts, osm), osm)
    assert lanes.tapers, name
    for k, (a, b) in lanes.tapers.items():
      node = float(lanes.along[k])
      few, opening = min(a.lanes, b.lanes), b.lanes > a.lanes
      s = np.arange(max(node - 20.0, 0.5), min(node + 20.0, lanes.along[-1] - 0.5), 0.25)
      rows = []
      for v in s:
        sec = lanes.section_at(float(v), lanes.segment(v))
        c = [sp.centre for sp in sec.ours]
        rows.append(c if len(c) == few else None)
      steps = [abs(np.subtract(p, q)).max() for p, q in zip(rows, rows[1:], strict=False) if p is not None and q is not None]
      assert max(steps, default=0.0) < 0.3, (name, max(steps))


if __name__ == "__main__":
  for name, fn in sorted(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"ok {name}")
