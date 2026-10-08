"""Carriageways made of one-way ways side by side, as GTA draws its freeways: paint_survey.correct_carriageway on made-up
samples, and side_by_side.py's kerbs on a small map made here. No pytest needed: `python test_carriageway.py`."""
import numpy as np

from openpilot.tools.sim.bridge.gta5.map.osm_lanes import FORWARD, offset_line
from openpilot.tools.sim.bridge.gta5.map.paint_survey import correct_carriageway, correct_oneway
from openpilot.tools.sim.bridge.gta5.map.side_by_side import SideBySide
from openpilot.tools.sim.bridge.gta5.map.test_junctions import make


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


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
