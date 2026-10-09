"""stop_paint.py's stop lines from painted lines, on small maps made here. No pytest needed: `python test_stop_paint.py`
runs them all (the rewrite test needs pyosmium, as ynd_to_osm does)."""
import os
import tempfile

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.junctions import Junctions
from openpilot.tools.sim.bridge.gta5.map.stop_paint import Line, StopPaint, _interpolate
from openpilot.tools.sim.bridge.gta5.map.test_junctions import TWO_WAY, make

# a crossroads at node 1, (0, 0); the south road is two ways, from node 3 to node 2 (20 m south) and on to the junction,
# both drawn towards it, so its northbound lanes are on the right, x 0 to 5.5
NODES = {1: (0.0, 0.0), 2: (0.0, -20.0), 3: (0.0, -100.0), 4: (-100.0, 0.0), 5: (100.0, 0.0), 6: (0.0, 100.0)}
WAYS = {1: (TWO_WAY, [3, 2]), 2: (TWO_WAY, [2, 1]), 3: (TWO_WAY, [1, 4]), 4: (TWO_WAY, [1, 5]), 5: (TWO_WAY, [1, 6])}
SIGNALS = {2: {'highway': 'traffic_signals', 'traffic_signals:direction': 'forward'}}


def line(n: int, x0: float, x1: float, y: float) -> Line:
  return Line(n, np.array([[x0, y], [x1, y]]), 0.0)


def place(lines, tags=None, crossings=None):
  return StopPaint(Junctions(make(NODES, WAYS, tags)), lines, crossings).place()


def test_moved_to_the_paint():
  placed, counts = place([line(7, 0.3, 5.2, -15.0)], SIGNALS)
  assert counts.get('moved to the paint') == 1 and len(placed) == 1
  p = placed[0]
  assert p.way == 2 and abs(p.t - 0.25) < 1e-6 and p.node is None and p.moved_from == [2] and p.line == 7
  assert p.tags == {'highway': 'traffic_signals', 'traffic_signals:direction': 'forward', 'source:position': 'survey'}
  assert np.allclose(p.xy, [0.0, -15.0])


def test_paint_for_the_other_way():
  # a line across the southbound lanes only (leaving the junction) isn't this approach's
  placed, counts = place([line(7, -5.2, -0.3, -15.0)], SIGNALS)
  assert placed == [] and counts.get('no paint') == 1


def test_crossing_edges():
  # a crossing's two edges across the whole road: the stop line is its far edge, out from the junction
  placed, _ = place([line(7, -5.4, 5.4, -12.0), line(8, -5.4, 5.4, -15.5)], SIGNALS)
  assert len(placed) == 1 and placed[0].line == 8 and np.allclose(placed[0].xy, [0.0, -15.5])


def test_wide_crossing():
  # lines 7 m apart across the lanes in: a crossing's edges only where it's painted between them
  lines = [line(7, 0.3, 5.4, -12.0), line(8, 0.3, 5.4, -19.0)]
  placed, _ = place(lines, SIGNALS)
  assert len(placed) == 1 and placed[0].line == 7
  zebra = (np.array([[-5.5, -18.5], [5.5, -18.5], [5.5, -12.5], [-5.5, -12.5]]), 0.0)
  placed, _ = place(lines, SIGNALS, [zebra])
  assert len(placed) == 1 and placed[0].line == 8
  # or both across the whole road: its edges painted without stripes
  placed, _ = place([line(7, -5.4, 5.4, -12.0), line(8, -5.4, 5.4, -19.0)], SIGNALS)
  assert len(placed) == 1 and placed[0].line == 8


def test_line_in_pieces():
  # a line laid as two decals end to end covers the lanes as one
  placed, _ = place([line(7, 0.3, 2.5, -15.0), line(8, 2.9, 5.3, -15.1)])
  assert len(placed) == 1 and abs(placed[0].t - 0.25) < 0.01


def test_too_far_to_move():
  # paint 17 m out from the map's stop line is another line's (MAX_MOVE): the stop line stays
  placed, counts = place([line(7, 0.3, 5.2, -37.0)], SIGNALS)
  assert placed == [] and counts.get('paint too far from the stop line') == 1


def test_line_of_the_junction_behind():
  # crossroads 36 m apart: the line across the whole road at the south one's mouth is its own, not a stop line for
  # traffic heading north out of it to the north one
  nodes = {1: (0.0, 0.0), 2: (0.0, -20.0), 7: (0.0, -36.0), 3: (0.0, -100.0), 4: (-100.0, 0.0), 5: (100.0, 0.0),
           6: (0.0, 100.0), 8: (-100.0, -36.0), 9: (100.0, -36.0)}
  ways = {1: (TWO_WAY, [3, 7]), 10: (TWO_WAY, [7, 2]), 2: (TWO_WAY, [2, 1]), 3: (TWO_WAY, [1, 4]), 4: (TWO_WAY, [1, 5]),
          5: (TWO_WAY, [1, 6]), 11: (TWO_WAY, [7, 8]), 12: (TWO_WAY, [7, 9])}
  js = Junctions(make(nodes, ways, SIGNALS))
  south = next(j for j in js.junctions if 7 in j.nodes)
  mouth = -36.0 + max(m.trim for arm in south.arms for m in arm.members if m.nodes[1] == 2)
  placed, counts = StopPaint(js, [line(7, -5.4, 5.4, mouth + 1.0)]).place()
  assert counts.get("lines at another junction's mouth left out") == 1 and not any(p.moved_from == [2] for p in placed)


def test_new_stop_lines():
  # no stop line in the map: one added where the paint covers only the lanes into the junction
  placed, counts = place([line(7, 0.3, 5.2, -15.0)])
  assert counts.get('new (stop)') == 1 and len(placed) == 1 and placed[0].moved_from == []
  assert placed[0].tags == {'highway': 'stop', 'direction': 'forward', 'source:position': 'survey'}
  # not from a line across the whole road (a crossing's edge), nor from one with a painted crossing just ahead of it
  _, counts = place([line(7, -5.2, 5.2, -15.0)])
  assert counts.get('new: line across the whole road') == 1 and 'new (stop)' not in counts
  zebra = (np.array([[0.0, -14.0], [5.5, -14.0], [5.5, -11.0], [0.0, -11.0]]), 0.0)
  _, counts = place([line(7, 0.3, 5.2, -15.0)], crossings=[zebra])
  assert counts.get('new: a crossing ahead') == 1


def test_stop_line_for_the_other_way():
  # GTA's stop node 15 m south of a crossroads on the south road, its direction for traffic heading south (to the
  # junction 45 m on): the paint beside it is across the northbound lanes, into the crossroads, and none is on
  # its own approach; its sign goes to the paint, the way it faces
  nodes = {1: (0.0, 0.0), 2: (0.0, -15.0), 3: (0.0, -60.0), 4: (-100.0, 0.0), 5: (100.0, 0.0), 6: (0.0, 100.0),
           7: (-100.0, -60.0), 8: (100.0, -60.0), 9: (0.0, -160.0)}
  ways = {1: (TWO_WAY, [3, 2]), 2: (TWO_WAY, [2, 1]), 3: (TWO_WAY, [1, 4]), 4: (TWO_WAY, [1, 5]), 5: (TWO_WAY, [1, 6]),
          6: (TWO_WAY, [3, 7]), 7: (TWO_WAY, [3, 8]), 8: (TWO_WAY, [3, 9])}
  js = Junctions(make(nodes, ways, {2: {'highway': 'stop', 'direction': 'backward'}}))
  assert [(j.nodes, len(j.stops)) for j in js.junctions if j.stops] == [([3], 1)]  # the south junction's
  placed, counts = StopPaint(js, [line(7, 0.3, 5.2, -13.0)]).place()
  assert counts.get('moved to the paint the other way') == 1 and len(placed) == 1
  p = placed[0]
  assert p.moved_from == [2] and p.tags == {'highway': 'stop', 'direction': 'forward', 'source:position': 'survey'}
  assert np.allclose(p.xy, [0.0, -13.0])


def test_widths_at_the_cut():
  tags = {'width:lanes:forward:start': '0|3.5', 'width:lanes:forward:end': '3|3.5', 'lanes': '2'}
  first, second = _interpolate(tags, 0.5)
  assert first['width:lanes:forward:end'] == '1.5|3.5' and second['width:lanes:forward:start'] == '1.5|3.5'
  assert first['width:lanes:forward:start'] == '0|3.5' and second['width:lanes:forward:end'] == '3|3.5'


def test_rewrite():
  try:
    import osmium
  except ImportError:
    print('test_rewrite skipped: no pyosmium')
    return
  from openpilot.tools.sim.bridge.gta5.map import osm_pbf, stop_paint
  from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon
  from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import remap_restrictions
  placed, _ = place([line(7, 0.3, 5.2, -15.0)], SIGNALS)
  with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, 'map.osm.pbf')
    w = osmium.SimpleWriter(path)
    for n, (x, y) in sorted(NODES.items()):
      lat, lon = to_lat_lon(x, y)
      w.add_node(osmium.osm.mutable.Node(id=n, version=1, location=(lon, lat), tags=SIGNALS.get(n, {})))
    for wid, (tags, refs) in sorted(WAYS.items()):
      w.add_way(osmium.osm.mutable.Way(id=wid, version=1, nodes=refs, tags=tags))
    w.add_relation(osmium.osm.mutable.Relation(id=1, version=1, tags={'type': 'restriction', 'restriction': 'no_left_turn'},
                                               members=[('w', 2, 'from'), ('n', 1, 'via'), ('w', 3, 'to')]))
    w.close()
    new_nodes, pieces = stop_paint.rewrite(path, placed, lambda: 1000, to_lat_lon, remap_restrictions)
    data = osm_pbf.read(path)
    assert list(new_nodes) == [1000] and pieces == {2: [(2, 2, 1000), (6, 1000, 1)]}
    assert data.ways[2][1] == [2, 1000] and data.ways[6][1] == [1000, 1] and data.ways[6][0] == TWO_WAY
    assert 'highway' not in data.node_tags.get(2, {}) and data.node_tags[1000]['highway'] == 'traffic_signals'
    rels = {tuple(tuple(m) for m in members) for _, members in data.relations.values()}
    assert (('w', 6, 'from'), ('n', 1, 'via'), ('w', 3, 'to')) in rels  # the piece at the junction now
    assert (('w', 2, 'from'), ('n', 1000, 'via'), ('w', 2, 'to')) in rels  # no turning back part way along


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
