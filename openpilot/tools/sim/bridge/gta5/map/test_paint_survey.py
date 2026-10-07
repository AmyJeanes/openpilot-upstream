"""paint_survey.py on made-up survey samples. No pytest in the venv: `python test_paint_survey.py` runs them all."""
import json
import os
import tempfile

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.paint_survey import _clean, along, arrows, centre_kind, correct, correct_oneway, disagree, \
  line_kinds, load, median_edges, opening_taper, sources, strips, swing_taper


def mark(offset, colour='white', kind='dashed', conf=1.0, pair=None):
  return {'type': kind, 'colour': colour, 'offset': offset, 'conf': conf, 'pair': pair}


def sample(marks, s=5.0, **kw):
  return {'a': '1:0', 'b': '1:1', 's': s, 'dir': 'ab', 'marks': marks, 'junction': False, 'bay': False, **kw}


def test_median_and_lanes():
  # a 2 + 2 road with a 5.4 m painted median and 4.4 m lanes, kerbs read as white double lines, which aren't lanes
  marks = [mark(-2.7, 'yellow', 'double_solid', pair=[-2.8, -2.6]), mark(2.7, 'yellow', 'double_solid', pair=[2.6, 2.8]),
           mark(-7.1), mark(7.1), mark(11.5, kind='double_solid'), mark(-11.5, kind='edge_line')]
  got, why = correct([sample(marks, s) for s in (2, 5, 8)], 2, 2, (-11.5, 11.5))
  assert why is None and got == {'middle': True, 'forward': [4.4, 4.4], 'backward': [4.4, 4.4], 'median': 5.4}


def test_centre_off_the_line():
  # a 1 + 1 road whose centre is 0.3 m right of the link: the lanes' widths say it, the kerbs stay
  got, _ = correct([sample([mark(0.3, 'yellow', 'double_solid')], s) for s in (2, 8)], 1, 1, (-5.5, 5.5))
  assert got == {'middle': True, 'forward': [5.2], 'backward': [5.8], 'median': 0.0}
  # with more lanes one way the line is the centre's, so it must be on it
  got, why = correct([sample([mark(1.0, 'yellow'), mark(6.5), mark(-11.0, kind='edge_line')], s) for s in (2, 8)], 2, 1, (-5.5, 11.0))
  assert got is None and why == 'centre off the line'


def test_disagreements_change_nothing():
  centre = mark(0.0, 'yellow', 'double_solid')
  assert correct([sample([centre])], 1, 1, (-5.5, 5.5)) == (None, 'no centre')  # one sample only
  assert correct([sample([centre, mark(5.5)], s) for s in (2, 8)], 2, 2, (-11, 11)) == (None, '1+2 lanes painted, GTA has 2+2')
  assert correct([sample([centre], s) for s in (2, 8)], 1, 0, (-2.75, 2.75)) == (None, 'one-way')


def test_load_both_ways():
  with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False) as f:
    f.write(json.dumps(sample([mark(1.0)], dir='ba')) + '\n' + json.dumps(sample([mark(2.0)], junction=True)) + '\n')
  try:
    samples = load([f.name])
  finally:
    os.unlink(f.name)
  assert list(samples) == [((1, 1), (1, 0))]  # seen travelling b to a; the junction section left out
  assert along(samples, (1, 0), (1, 1))[0]['marks'][0]['offset'] == -1.0


def test_game_files_first_with_their_kerbs():
  centre = mark(0.0, 'yellow', 'double_solid')
  camera = [sample([centre, mark(5.0), mark(-5.0)], s) for s in (2, 8)]
  files = [sample([centre, mark(5.4), mark(-5.4)], s, src='gamefiles', kerbs={'left': -10.7, 'right': 10.9, 'conf': 0.8})
           for s in (0, 3, 6, 9)]
  use, other = sources(camera + files)
  assert use == files and other == camera
  assert sources(camera) == (camera, []) and sources(camera, camera_corrects=False) == ([], camera)
  got, _ = correct(use, 2, 2, (-11.0, 11.0))
  assert got == {'middle': True, 'forward': [5.4, 5.4], 'backward': [5.4, 5.4], 'median': 0.0, 'divider': 'double_solid_line'}  # to the asphalt edges
  assert disagree(got, correct(other, 2, 2, (-11.0, 11.0))[0])
  lopsided = [{**d, 'kerbs': {'left': -7.6, 'right': 13.8}} for d in files]  # can't be the middle of the road: the layout's
  assert correct(lopsided, 2, 2, (-11.0, 11.0))[0]['forward'] == [5.4, 5.6]


def test_painted_arrows():
  def arrow(offset, kind, d='ab'):
    return {'offset': offset, 'kind': kind, 'dir': d, 'conf': 0.95}
  files = [sample([], s, src='gamefiles', arrows=arrs) for s, arrs in
           ((0, [arrow(2.4, 'left'), arrow(7.9, 'through'), arrow(13.1, 'through;right')]), (3, [arrow(2.5, 'left')]),
            (6, [arrow(-5.0, 'through;left', 'oncoming'), arrow(-10.5, 'right', 'oncoming')]))]
  assert arrows(files) == {'forward': ['left', 'through', 'through;right'], 'backward': ['left;through', 'right']}
  assert arrows([{**d, 'src': None} for d in files]) == {}  # the camera's arrows aren't read
  flipped = along({((1, 1), (1, 0)): files}, (1, 0), (1, 1))
  assert arrows(flipped) == {'forward': ['left;through', 'right'], 'backward': ['left', 'through', 'through;right']}


def test_lines_that_cant_be_crossed():
  # 3 lanes our way from the game files: solid between the left two, dashed on the left half / solid right between
  # the right two (as seen travelling a -> b); oncoming 1 lane
  marks = [mark(0.0, 'yellow', 'double_solid'), mark(5.5, kind='solid'), mark(11.0, kind='dashed_solid')]
  files = [sample(marks, s, src='gamefiles') for s in (0, 3, 6)]
  got, _ = correct(files, 3, 1, (-5.5, 16.5))
  assert got['change:forward'] == ['not_right', 'not_left', 'not_left'] and 'change:backward' not in got
  # seen the other way, the same line's halves swap
  got, _ = correct(along({((1, 1), (1, 0)): files}, (1, 0), (1, 1)), 1, 3, (-16.5, 5.5))
  assert got['change:backward'] == ['not_right', 'not_left', 'not_left']
  camera = [{**d, 'src': None} for d in files]
  assert 'change:forward' not in (correct(camera, 3, 1, (-5.5, 16.5))[0] or {})  # the camera's kinds aren't trusted


def test_one_way_from_the_game_files():
  # a 3-lane freeway: yellow left edge, lane lines 6.1 m apart (a solid one), white edge line
  marks = [mark(-9.15, 'yellow', 'solid'), mark(-3.05), mark(3.05, kind='solid'), mark(9.2, kind='edge_line')]
  files = [sample(marks, s, src='gamefiles') for s in (0, 3, 6)]
  got, _ = correct_oneway(files, 3, (-9.15, 9.15))
  assert got == {'lanes': [6.1, 6.1, 6.15], 'change': ['yes', 'not_right', 'not_left']}
  assert correct_oneway(files, 2, (-6.1, 6.1))[0] is None  # GTA says 2 lanes: the paint isn't read
  assert correct_oneway([{**d, 'src': None} for d in files], 3, (-9.15, 9.15))[1] == 'one-way, no game files'
  bare = [sample([mark(0.1)], s, src='gamefiles', kerbs={'left': -5.4, 'right': 5.6}) for s in (0, 3)]  # no edge lines
  assert correct_oneway(bare, 2, (-5.5, 5.5))[0] == {'lanes': [5.5, 5.5]}


def test_centre_kind_from_the_game_files():
  files = [sample([mark(0.1, 'yellow', 'solid_dashed', pair=[0.0, 0.2])], s, src='gamefiles') for s in (0, 3)]
  assert correct(files, 1, 1, (-5.5, 5.5))[0]['divider'] == 'solid_line;dashed_line'
  # seen the other way along the link, the halves swap
  assert correct(along({((1, 1), (1, 0)): files}, (1, 0), (1, 1)), 1, 1, (-5.5, 5.5))[0]['divider'] == 'dashed_line;solid_line'
  assert 'divider' not in correct([{**d, 'src': None} for d in files], 1, 1, (-5.5, 5.5))[0]


def test_parking_strips_and_painted_counts():
  # Eclipse Blvd as painted: 2 lanes one way, 3 the other, the centre 1.9 m left of GTA's 2 + 2 link; asphalt to
  # +-10.8, a paver strip to the kerb's face 2.3 m beyond on the right
  marks = [mark(-1.9, 'yellow', 'double_solid'), mark(-6.35), mark(2.4), mark(6.6)]
  files = [sample(marks, s, src='gamefiles', kerbs={'left': -10.8, 'right': 10.8}, kerb_step={'left': -11.4, 'right': 13.1})
           for s in (0, 3, 6)]
  assert correct(files, 2, 2, (-11.0, 11.0))[1] == '2+3 lanes painted, GTA has 2+2'
  got, _ = correct(files, 2, 2, (-11.0, 11.0), counts_from_paint=True)
  assert got['forward'] == [4.3, 4.2, 4.2] and got['backward'] == [4.45, 4.45] and got['parking'] == (0.0, 2.3) and got['middle']
  assert strips([{**d, 'kerb_step': {'left': -11.4, 'right': 11.3}} for d in files]) == (0.0, 0.0)  # a gutter, not a strip


def test_cleaning_the_game_files_lines():
  # a solid line read dashed in a worn section; raised markers that are the gutter's edge; a white dashed centre
  polylines = [{'id': 1, 'style': 'dashed', 'painted': 0.76, 'dashes': [[0, 61.5], [67.9, 103.5]]},
               {'id': 2, 'style': 'dashed', 'painted': 0.38, 'dashes': [[0, 0.4], [2.7, 3.1], [5.3, 5.7], [8.0, 8.4]]},
               {'id': 3, 'style': 'dashed', 'painted': 0.29, 'dashes': [[0, 1.0], [6.0, 7.0], [12.0, 13.0]]}]
  with tempfile.NamedTemporaryFile('w', suffix='.jsonl', delete=False) as f:
    f.write('\n'.join(json.dumps(p) for p in polylines))
  try:
    kinds = line_kinds(f.name)
  finally:
    os.unlink(f.name)
  assert kinds == {1: 'solid', 2: 'solid', 3: 'dashed'}  # worn, tiled decals, real dashes
  d = sample([{**mark(2.0, kind='dashed'), 'line': 1}, {**mark(0.1, kind='dashed'), 'line': 3}, mark(5.3, kind='markers')],
             src='gamefiles', kerbs={'left': -5.5, 'right': 5.5})
  assert [(m['offset'], m['type']) for m in _clean(d, kinds)['marks']] == [(2.0, 'solid'), (0.1, 'dashed')]
  centre = sample([{**mark(0.1, kind='dashed'), 'line': 3}, mark(5.3, kind='edge_line')], src='gamefiles')
  assert centre_kind([_clean(centre, kinds)] * 2, 0.0, 0.4) == 'dashed_line'  # no yellow: the white centre


def test_opening_taper():
  # a 5.4 m median's right edge at +2.7 swinging across to -2.7 between 40 and 60 m along, a stray line at the end
  def edge(d):
    return 2.7 - 5.4 * min(max((d - 40.0) / 20.0, 0.0), 1.0)
  sections = [(d, sample([mark(edge(d), 'yellow', 'double_solid')], src='gamefiles')) for d in range(1, 100, 3)]
  sections.append((99.0, sample([mark(2.5, 'yellow')], src='gamefiles')))
  start, end = opening_taper(sections, 5.4)
  assert abs(start - 41.6) < 1.0 and abs(end - 58.4) < 1.0
  assert opening_taper([(d, sample([mark(2.7, 'yellow')], src='gamefiles')) for d in range(1, 100, 3)], 5.4) is None
  assert opening_taper([(d, {**s, 'src': None}) for d, s in sections], 5.4) is None  # the game files' only


def test_swing_taper():
  # a road along +y with a 5.4 m median; our edge at +2.7 swings across to -2.7 between 20 and 35 m along, steeper than
  # sections read, and runs on beside the oncoming edge
  road = np.array([[0.0, 0.0], [0.0, 30.0], [0.0, 60.0]])
  ours = (1, np.array([[2.7, 0.0], [2.7, 20.0], [-2.7, 35.0], [-2.7, 60.0]]))
  oncoming = (2, np.array([[-2.7, 0.0], [-2.7, 60.0]]))
  start, end = swing_taper([ours, oncoming], road, 2.7)
  assert abs(start - 21.2) < 0.3 and abs(end - 33.3) < 0.3  # 8% across, and within 0.6 m of the far edge
  # the other way's edge crossing back to back with ours (a diamond): it reads as a swing here too, but our edge runs on
  # at +2.7 past it, so it's theirs
  theirs = (3, np.array([[2.7, 5.0], [-2.7, 18.0]]))
  edge = (4, np.array([[2.7, 0.0], [2.7, 60.0]]))
  assert swing_taper([theirs, edge, oncoming], road, 2.7) is None
  assert swing_taper([oncoming], road, 2.7) is None


def test_median_edges():
  # double solid beside the oncoming lanes, one solid line beside ours; a lone centre line says nothing
  marks = [mark(-2.7, 'yellow', 'double_solid', pair=[-2.8, -2.6]), mark(2.6, 'yellow', 'solid'), mark(7.0)]
  files = [sample(marks, s, src='gamefiles') for s in (1.0, 4.0, 7.0)]
  assert median_edges(files, 5.4) == ('double_solid_line', 'solid_line')
  assert median_edges([sample([mark(0.1, 'yellow', 'double_solid')], src='gamefiles')] * 3, 5.4) == (None, None)


def test_arrows_from_features():
  feature = {'kind': 'left', 'offset': 2.0, 'dir': 'ab', 'conf': 0.95, 's': 1.0}
  files = [sample([], 0, src='gamefiles', features=[feature, {**feature, 'kind': 'through;right', 'offset': 7.5}, {'kind': 'stop', 'offset': 5.0}])]
  assert arrows(files) == {'forward': ['left', 'through;right']}


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
