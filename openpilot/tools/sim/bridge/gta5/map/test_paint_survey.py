"""paint_survey.py on made-up survey samples. No pytest in the venv: `python test_paint_survey.py` runs them all."""
import json
import os
import tempfile

from openpilot.tools.sim.bridge.gta5.map.paint_survey import along, arrows, correct, disagree, load, sources


def mark(offset, colour='white', kind='dashed', conf=1.0, pair=None):
  return {'type': kind, 'colour': colour, 'offset': offset, 'conf': conf, 'pair': pair}


def sample(marks, s=5.0, **kw):
  return {'a': '1:0', 'b': '1:1', 's': s, 'dir': 'ab', 'marks': marks, 'junction': False, 'bay': False, **kw}


def test_median_and_lanes():
  # a 2 + 2 road with a 5.4 m painted median and 4.4 m lanes, kerbs read as white double lines, which aren't lanes
  marks = [mark(-2.7, 'yellow', 'double_solid', pair=[-2.8, -2.6]), mark(2.7, 'yellow', 'double_solid', pair=[2.6, 2.8]),
           mark(-7.1), mark(7.1), mark(11.5, kind='double_solid'), mark(-11.5, kind='edge_line')]
  got, why = correct([sample(marks, s) for s in (2, 5, 8)], 2, 2, (-11.5, 11.5))
  assert why is None and got == {'forward': [4.4, 4.4], 'backward': [4.4, 4.4], 'median': 5.4}


def test_centre_off_the_line():
  # a 1 + 1 road whose centre is 0.3 m right of the link: the lanes' widths say it, the kerbs stay
  got, _ = correct([sample([mark(0.3, 'yellow', 'double_solid')], s) for s in (2, 8)], 1, 1, (-5.5, 5.5))
  assert got == {'forward': [5.2], 'backward': [5.8], 'median': 0.0}
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
  got, _ = correct(use, 2, 2, (-11.0, 11.0))
  assert got == {'forward': [5.4, 5.4], 'backward': [5.4, 5.4], 'median': 0.0}  # out to the asphalt's edges, 10.8 m
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


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
