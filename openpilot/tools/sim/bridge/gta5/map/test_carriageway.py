"""Carriageways made of one-way ways side by side, as GTA draws its freeways: paint_survey.correct_carriageway,
.outer_lines and correct_oneway on made-up samples, ynd_to_osm.lane_changes and .split_shared on made-up links, and
side_by_side.py's kerbs and Junctions.merges on small maps made here. No pytest needed: `python test_carriageway.py`."""
import numpy as np

from openpilot.tools.sim.bridge.gta5.map.junctions import Junctions
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import FORWARD, offset_line
from openpilot.tools.sim.bridge.gta5.map.paint_survey import correct_carriageway, correct_oneway, outer_lines
from openpilot.tools.sim.bridge.gta5.map.side_by_side import SideBySide
from openpilot.tools.sim.bridge.gta5.map.test_junctions import make
from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import SPLIT_NODE_AREA, lane_changes, split_shared


def mark(offset, colour='white', kind='dashed'):
  return {'type': kind, 'colour': colour, 'offset': offset, 'conf': 1.0, 'pair': None}


def sample(marks, s):
  return {'a': '1:0', 'b': '1:1', 's': s, 'dir': 'ab', 'marks': marks, 'junction': False, 'bay': False, 'src': 'gamefiles'}


# five 6 m lanes across a freeway carriageway, seen from a link of two of them: a yellow left edge, raised markers beside
# a dashed line (GTA lays both), a solid line between this link's lanes and the next link's, a white edge on the right
CARRIAGEWAY = [mark(-12.9, 'yellow', 'solid'), mark(-8.2, kind='markers'), mark(-6.8), mark(-0.6), mark(5.4, kind='solid'),
               mark(11.4), mark(17.4, kind='edge_line')]


def test_lanes_in_their_carriageway():
  samples = [sample(CARRIAGEWAY, s) for s in (2.0, 5.0, 8.0)]
  assert correct_oneway(samples, 2, (-6.1, 6.1)) == (None, 'one-way, off the line')
  # the two lanes about the line, which goes on the line between them (0.6 m off): only that line moves
  got, why = correct_carriageway(samples, 2)
  assert why is None and got == {'lanes': [6.8, 5.4], 'placement': 'right_of:1', 'change': ['yes', 'not_right']}, got
  # a link of one lane about the line: on its middle (0.2 m off), the lane moved with it
  shifted = [sample([{**m, 'offset': m['offset'] - 2.2} for m in CARRIAGEWAY], s) for s in (2.0, 5.0)]
  got, why = correct_carriageway(shifted, 1)
  assert why is None and got['lanes'] == [6.0] and got['placement'] == 'middle_of:1', got
  # nothing painted near enough the line to place it
  assert correct_carriageway([sample([mark(-12.3, 'yellow', 'solid'), mark(-6.2)], s) for s in (2.0, 5.0)], 2)[0] is None
  assert correct_carriageway(samples[:1], 2) == (None, 'carriageway, no game files')


FREEWAY = {'highway': 'motorway', 'oneway': 'yes', 'lanes': '1', 'width': '6'}


def test_no_kerbs_between_ways_side_by_side():
  # eastbound: 1a, 1b along y = 0 and 2a, 2b along y = -6 (6 m wide each, edge to edge at y = -3), lane change 3 from 1's
  # node at x = 50 to 2's at x = 70; westbound 4 along y = 6, its kerb on 1's left kerb
  nodes = {1: (0.0, 0.0), 2: (50.0, 0.0), 3: (100.0, 0.0), 4: (0.0, -6.0), 5: (70.0, -6.0), 6: (100.0, -6.0), 7: (100.0, 6.0),
           8: (0.0, 6.0)}
  ways = {11: (FREEWAY, [1, 2]), 12: (FREEWAY, [2, 3]), 21: (FREEWAY, [4, 5]), 22: (FREEWAY, [5, 6]), 3: (FREEWAY, [2, 5]),
          4: (FREEWAY, [7, 8])}
  osm = make(nodes, ways)
  side = SideBySide(osm, list(ways), lambda w: 0)

  def kerbs(wid, right):
    pts = osm.way_points(wid)
    lo, hi = osm.lanes(wid).edges(FORWARD)
    return side.kerb(offset_line(pts, hi if right else lo), None, 0, {wid}, right)

  def length(pieces):
    return sum(float(np.hypot(*np.diff(p, axis=0).T).sum()) for p in pieces)
  kept, between = kerbs(12, True)  # on 2's left edge: the lane line between them, dashed
  assert not kept and {style for _, style in between} == {'dashed'} and abs(length([p for p, _ in between]) - 50.0) < 1.0
  kept, between = kerbs(21, False)  # 2's left kerb: under 1, drawn by 1 as the lane line
  assert not kept and not between
  for right in (False, True):  # the lane change is on the road all the way
    assert kerbs(3, right) == ([], [])
  assert side.crosses(3) and not any(side.crosses(w) for w in (11, 12, 21, 22, 4))
  kept, between = kerbs(11, False)  # 1's left kerb: the oncoming way beside it doesn't run the same way
  assert abs(length(kept) - 50.0) < 1.0 and not between
  kept, _ = kerbs(4, False)
  assert abs(length(kept) - 100.0) < 1.0
  kept, _ = kerbs(22, True)  # the carriageway's outer kerb
  assert abs(length(kept) - 30.0) < 1.0
  # ways it wasn't given as one-way ways (two-way roads) keep their kerbs
  line = offset_line(osm.way_points(12), 3.0)
  kept, between = SideBySide(osm, [], lambda w: 0).kerb(line, None, 0, {12}, True)
  assert len(kept) == 1 and np.array_equal(kept[0], line) and not between


def test_lane_change_across_a_gore():
  # eastbound carriageways 1 (y = 0) and 2 (y = -14), 6 m wide each with a painted gore 8 m wide between their kerbs,
  # and GTA's lane changes across it in an X: 3 from 1's node at x = 50 to 2's at x = 70, 4 from 2's at x = 40 to 1's at
  # x = 60 (Dutch London St). They draw no kerbs and cut none: the gore's edges are the carriageways' kerbs
  nodes = {1: (0.0, 0.0), 2: (50.0, 0.0), 3: (60.0, 0.0), 4: (100.0, 0.0), 5: (0.0, -14.0), 6: (40.0, -14.0), 7: (70.0, -14.0),
           8: (100.0, -14.0)}
  ways = {11: (FREEWAY, [1, 2]), 12: (FREEWAY, [2, 3]), 13: (FREEWAY, [3, 4]), 21: (FREEWAY, [5, 6]), 22: (FREEWAY, [6, 7]),
          23: (FREEWAY, [7, 8]), 3: (FREEWAY, [2, 7]), 4: (FREEWAY, [6, 3])}
  osm = make(nodes, ways)
  side = SideBySide(osm, list(ways), lambda w: 0)
  assert side.across == {3, 4}

  def kept(wid, right):
    pts = osm.way_points(wid)
    lo, hi = osm.lanes(wid).edges(FORWARD)
    return side.kerb(offset_line(pts, hi if right else lo), None, 0, {wid}, right)

  for wid in (3, 4):
    assert kept(wid, False) == ([], []) and kept(wid, True) == ([], [])
  for wid, right in ((11, True), (12, True), (13, True), (21, False), (22, False), (23, False)):
    pieces = kept(wid, right)[0]
    length = float(np.hypot(*(osm.way_points(wid)[-1] - osm.way_points(wid)[0])))
    assert len(pieces) == 1 and abs(float(np.hypot(*np.diff(pieces[0], axis=0).T).sum()) - length) < 0.5, (wid, pieces)
  # side by side, edge to edge, they're lane changes over the lanes as before (test_no_kerbs_between_ways_side_by_side)
  near = {k: (x, y if y == 0.0 else -6.0) for k, (x, y) in nodes.items()}
  assert not SideBySide(make(near, ways), list(ways), lambda w: 0).across


def test_no_kerbs_across_a_diverge():
  # a carriageway of four 6 m lanes along y = 0 parts at x = 0 into two of two lanes, both starting from its middle (as
  # GTA lays them), one bearing left to (100, 20), one right to (100, -20); and they merge again from x = 200 to 300
  wide, half = {**FREEWAY, 'lanes': '4', 'width': '24'}, {**FREEWAY, 'lanes': '2', 'width': '12'}
  nodes = {1: (-100.0, 0.0), 2: (0.0, 0.0), 3: (100.0, 20.0), 4: (100.0, -20.0), 5: (200.0, 20.0), 6: (200.0, -20.0),
           7: (300.0, 0.0), 8: (400.0, 0.0)}
  ways = {1: (wide, [1, 2]), 2: (half, [2, 3]), 3: (half, [2, 4]), 4: (half, [3, 5]), 5: (half, [4, 6]), 6: (half, [5, 7]),
          7: (half, [6, 7]), 8: (wide, [7, 8])}
  osm = make(nodes, ways)
  side = SideBySide(osm, list(ways), lambda w: 0)

  def kept(wid, right):
    pts = osm.way_points(wid)
    lo, hi = osm.lanes(wid).edges(FORWARD)
    return side.kerb(offset_line(pts, hi if right else lo), None, 0, {wid}, right)[0]

  def xs(pieces):
    return [(round(float(p[0, 0])), round(float(p[-1, 0]))) for p in pieces]
  # the left branch's outer kerb from where it nears the wide carriageway's edge (y = 12 - SHARED, x ~ 23), not across
  # its lanes; its inner kerb (the gore's edge) from GORE m on, where the branches have parted, and once they're more
  # than GORE_SHARED apart (before that, the edge painted on the gore)
  outer, inner = xs(kept(2, False)), xs(kept(2, True))
  assert len(outer) == 1 and 20 <= outer[0][0] <= 26 and outer[0][1] == 99, outer
  assert len(inner) == 1 and 40 <= inner[0][0] <= 50 and inner[0][1] == 101, inner
  assert xs(kept(1, False)) == [(-100, 0)] and xs(kept(1, True)) == [(-100, 0)]  # the wide carriageway's own
  # merging: the mirror image, back from the wide carriageway's start
  outer = xs(kept(6, False))
  assert len(outer) == 1 and outer[0][0] == 201 and 274 <= outer[0][1] <= 280, outer
  assert xs(kept(8, True)) == [(300, 400)]


def test_no_kerbs_on_a_two_way_road():
  # a one-way turn bay leaving a two-way road (18 m wide along y = 0) from its middle at x = 0, out to y = -12 by x = 20:
  # its kerbs are left out where they lie on the road's carriageway (out to y = -9 - SHARED), not across its lanes
  road = {'highway': 'primary', 'lanes': '4', 'width': '18'}
  bay = {'highway': 'residential', 'lanes': '1', 'oneway': 'yes', 'width': '5.5'}
  nodes = {1: (-100.0, 0.0), 2: (0.0, 0.0), 3: (100.0, 0.0), 4: (20.0, -12.0), 5: (100.0, -12.0)}
  ways = {1: (road, [1, 2]), 2: (road, [2, 3]), 3: (bay, [2, 4]), 4: (bay, [4, 5])}
  osm = make(nodes, ways)
  side = SideBySide(osm, list(ways), lambda w: 0)
  for right in (False, True):
    pts = osm.way_points(3)
    lo, hi = osm.lanes(3).edges(FORWARD)
    kept, _ = side.kerb(offset_line(pts, hi if right else lo), None, 0, {3}, right)
    assert all(np.all(p[:, 1] < -9.0) for p in kept), (right, [p.round(1).tolist() for p in kept])
  pts = osm.way_points(4)  # its own carriageway beside the road keeps its outer kerb
  lo, hi = osm.lanes(4).edges(FORWARD)
  kept, _ = side.kerb(offset_line(pts, hi), None, 0, {4}, True)
  assert sum(float(np.hypot(*np.diff(p, axis=0).T).sum()) for p in kept) > 75.0


def test_freeway_kerbs_by_a_gore_are_painted_edges():
  # a motorway of two lanes along y = 0 (12 m wide) with an on-ramp lane beside it 3 m out (its middle at y = -11.75)
  # across a flush gore, and the other carriageway beyond a 4 m median (y = 16, westbound): the kerbs facing the gore are
  # the solid lines painted there; those at the median, beside traffic the other way, stay kerbs
  fwy = {**FREEWAY, 'lanes': '2', 'width': '12'}
  ramp = {'highway': 'motorway_link', 'oneway': 'yes', 'lanes': '1', 'width': '5.5'}
  nodes = {1: (0.0, 0.0), 2: (100.0, 0.0), 3: (0.0, -11.75), 4: (100.0, -11.75), 5: (100.0, 16.0), 6: (0.0, 16.0)}
  osm = make(nodes, {1: (fwy, [1, 2]), 2: (ramp, [3, 4]), 3: (fwy, [5, 6])})
  side = SideBySide(osm, [1, 2, 3], lambda w: 0)

  def kerb(wid, right):
    lo, hi = osm.lanes(wid).edges(FORWARD)
    return side.kerb(offset_line(osm.way_points(wid), hi if right else lo), None, 0, {wid}, right)
  for wid, right in ((1, True), (2, False)):
    kept, lines = kerb(wid, right)
    assert not kept and [style for _, style in lines] == ['solid'], (wid, kept, lines)
  for wid, right in ((1, False), (3, False), (2, True)):
    kept, lines = kerb(wid, right)
    assert len(kept) == 1 and not lines, (wid, right)


def test_two_way_road_beside_a_turn_bay_has_a_lane_line_not_a_kerb():
  # a two-way road 14 m wide along y = 0, eastbound on the right (south); a one-way eastbound turn bay laid as its own
  # way beside it, its middle 9.5 m south (edges at y = -7.25 .. -11.75): the road's south kerb (y = -7) lies on the
  # bay's left edge, the solid line between them; its north kerb stays
  road = {'highway': 'primary', 'lanes': '4', 'width': '14'}
  bay = {'highway': 'residential', 'lanes': '1', 'oneway': 'yes', 'width': '4.5'}
  nodes = {1: (0.0, 0.0), 2: (100.0, 0.0), 3: (20.0, -9.5), 4: (80.0, -9.5)}
  osm = make(nodes, {1: (road, [1, 2]), 2: (bay, [3, 4])})
  side = SideBySide(osm, [1, 2], lambda w: 0)
  lo, hi = osm.lanes(1).edges(FORWARD)
  kept, lines = side.kerb(offset_line(osm.way_points(1), hi), None, 0, {1}, True)  # south, eastbound traffic's side
  assert {st for _, st in lines} == {'solid'} and sum(_len(p) for p, _ in lines) > 55.0
  assert all(p[:, 0].max() <= 21.0 or p[:, 0].min() >= 79.0 for p in kept), [p.round(1).tolist() for p in kept]
  kept, lines = side.kerb(offset_line(osm.way_points(1), lo), None, 0, {1}, False)
  assert len(kept) == 1 and not lines and _len(kept[0]) > 99.0


def _len(p):
  return float(np.hypot(*np.diff(p, axis=0).T).sum())


def test_outer_lines_and_missing_lane_lines():
  # a one-way link's lanes from -6 to 6: a solid line at its left edge, dashed at its right, the next links' lines
  samples = [sample([mark(-6.1, kind='solid'), mark(5.9), mark(12.0)], s) for s in (2.0, 5.0, 8.0)]
  assert outer_lines(samples, (-6.0, 6.0)) == (False, True)
  assert outer_lines([sample([mark(-12.0)], s) for s in (2.0, 5.0)], (-6.0, 6.0)) == (None, None)
  # two lanes between painted edges with the lane line between missing from the files: GTA's two, evenly
  edges = [sample([mark(-6.4, 'yellow', 'solid'), mark(6.5, 'yellow', 'solid')], s) for s in (2.0, 5.0, 8.0)]
  got, why = correct_oneway(edges, 2, (-6.1, 6.1))
  assert why is None and got['lanes'] == [6.45, 6.45], got


def node(x, y):
  return {'x': x, 'y': y}


def test_two_way_lane_changes_run_with_the_traffic():
  # two northbound one-way links side by side (x = 0 and x = 6), two-way links crossing between them in an X
  nodes = {1: node(0, 0), 2: node(0, 20), 3: node(0, 40), 4: node(6, 0), 5: node(6, 20), 6: node(6, 40), 7: node(40, 20),
           8: node(6, 60), 9: node(30, 60)}
  es = {(1, 2): (1, 0, None), (2, 3): (1, 0, None), (4, 5): (1, 0, None), (5, 6): (1, 0, None), (6, 8): (1, 0, None),
        (2, 6): (1, 1, None), (3, 5): (1, 1, None), (8, 9): (1, 1, None)}
  assert lane_changes(nodes, es) == {(2, 6): True, (3, 5): False}
  # a two-way street off to the east at one's end: that one stays two-way
  es[(5, 7)] = (1, 1, None)
  assert lane_changes(nodes, es) == {(2, 6): True}


def test_no_junction_where_lanes_only_merge():
  oneway = {'highway': 'primary', 'oneway': 'yes', 'lanes': '1'}
  # a lane change leaving a one-way street at 20 degrees: lanes parting, no junction
  nodes = {1: (0.0, 0.0), 2: (50.0, 0.0), 3: (100.0, 0.0), 4: (100.0, 18.0)}
  ways = {1: (oneway, [1, 2]), 2: (oneway, [2, 3]), 3: (oneway, [2, 4])}
  assert not Junctions(make(nodes, ways)).junctions
  # a freeway's exit lane leaving it, with GTA's lane changes between them as links at ~60 degrees, the lane classed
  # as a minor road: lanes parting, no junction areas across the carriageway
  fwy, link, lane = ({'highway': h, 'oneway': 'yes', 'lanes': '1'} for h in ('motorway', 'motorway_link', 'residential'))
  lattice = {1: (0.0, 0.0), 2: (50.0, 0.0), 3: (60.0, 0.0), 4: (150.0, 0.0), 5: (53.0, -6.0), 6: (63.0, -6.0), 7: (150.0, -6.0),
             8: (0.0, -6.0)}
  assert not Junctions(make(lattice, {1: (fwy, [1, 2]), 2: (fwy, [2, 3]), 3: (fwy, [3, 4]), 4: (link, [2, 5]), 5: (lane, [8, 5]),
                                      6: (lane, [5, 6]), 7: (link, [3, 6]), 8: (lane, [6, 7])})).junctions
  # a one-way side street joining at 80 degrees: a junction
  nodes[4] = (60.0, 57.0)
  ways[3] = (oneway, [4, 2])
  assert len(Junctions(make(nodes, ways)).junctions) == 1


def test_carriageways_through_one_node_are_parted():
  def gta(x, y):
    return {'a': 1, 'i': 0, 'x': x, 'y': y, 'z': 0.0, 'f': [0, 0, 4, 0, 0], 'st': 0, 'sp': 1}
  one, two = [0, 0, 2 << 5], [0, 0, (1 << 5) | (1 << 2)]
  # a divided road's carriageways 16 m apart (westbound north), both bent in to one node in the median, where a side
  # road from the north meets it, and a turn lane in the median
  nodes = {'W1': gta(60, 8), 'W2': gta(-60, 8), 'E1': gta(-60, -8), 'E2': gta(60, -8), 'N': gta(0, 0), 'S': gta(0, 40),
           'T': gta(30, 2)}
  info = [[1, 'W1', 'N', 2, 0, 'primary', 40, None, one], [2, 'N', 'W2', 2, 0, 'primary', 40, None, one],
          [3, 'E1', 'N', 2, 0, 'primary', 40, None, one], [4, 'N', 'E2', 2, 0, 'primary', 40, None, one],
          [5, 'S', 'N', 1, 1, 'residential', 25, None, two], [6, 'T', 'N', 1, 0, 'primary', 40, None, one]]
  assert split_shared(nodes, info) == 1
  w, e = (SPLIT_NODE_AREA, 1), (SPLIT_NODE_AREA, 2)
  assert (nodes[w]['x'], nodes[w]['y'], nodes[e]['x'], nodes[e]['y']) == (0, 8, 0, -8)
  assert [r[1:3] for r in info] == [['W1', w], [w, 'W2'], ['E1', e], [e, 'E2'], ['S', w], ['T', 'N'], [w, 'N'], ['N', e]]
  assert [r[0] for r in info[-2:]] == [7, 8] and info[-1][3:5] == [1, 1]
  # without the turn lane, a link straight across
  nodes = {k: v for k, v in nodes.items() if k[0] != SPLIT_NODE_AREA}
  info = [r for r in info if r[0] < 6]
  for r, (a, b) in zip(info, [('W1', 'N'), ('N', 'W2'), ('E1', 'N'), ('N', 'E2'), ('S', 'N')], strict=True):
    r[1], r[2] = a, b
  assert split_shared(nodes, info) == 1 and [r[1:3] for r in info[5:]] == [[w, e]]
  # a second divided road crossing it through the same node: left as GTA has it
  nodes |= {'S1': gta(-8, 60), 'S2': gta(-8, -60), 'N1': gta(8, -60), 'N2': gta(8, 60)}
  info = info[:4] + [[5, 'S1', 'N', 2, 0, 'primary', 40, None, one], [6, 'N', 'S2', 2, 0, 'primary', 40, None, one],
                                  [7, 'N1', 'N', 2, 0, 'primary', 40, None, one], [8, 'N', 'N2', 2, 0, 'primary', 40, None, one]]
  for r, (a, b) in zip(info, [('W1', 'N'), ('N', 'W2'), ('E1', 'N'), ('N', 'E2')], strict=False):
    r[1], r[2] = a, b
  assert split_shared(nodes, info) == 0
  # nor where the node is on a carriageway's line, as where two-way traffic meets a one-way pair
  info = info[:4]
  nodes['N'] = gta(0, 8)
  assert split_shared(nodes, info) == 0


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
