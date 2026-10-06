"""validate_lanes.py on the fixtures, which are clean, and on broken copies of them. No pytest in the venv:
`python test_validate_lanes.py` runs them all."""
import os
import tempfile

from openpilot.tools.sim.bridge.gta5.map.validate_lanes import check_tags, validate

FIXTURES = os.path.join(os.path.dirname(__file__), 'fixtures')


def validate_text(name: str, old: str, new: str, drive_on_right: bool = True):
  """The issues in a fixture with `old` replaced by `new`."""
  with open(os.path.join(FIXTURES, name)) as f:
    text = f.read()
  assert old in text
  with tempfile.NamedTemporaryFile('w', suffix='.osm', delete=False) as f:
    f.write(text.replace(old, new))
  try:
    return validate(f.name, drive_on_right)
  finally:
    os.unlink(f.name)


def checks(tags: dict, length: float | None = None) -> set[str]:
  return {check for check, _ in check_tags(tags, length)}


def test_fixtures_are_clean():
  for name in sorted(os.listdir(FIXTURES)):
    issues = validate(os.path.join(FIXTURES, name), drive_on_right=name != 'lht.osm')
    assert issues == [], (name, issues)


def test_left_hand_placements_read_right_hand():
  issues = validate(os.path.join(FIXTURES, 'lht.osm'))
  assert [i.check for i in issues] == ['placement']


def test_lane_counts():
  road = {'highway': 'primary', 'lanes': '3', 'lanes:forward': '2', 'lanes:backward': '1'}
  assert checks(road) == set()
  assert checks({**road, 'lanes': '4'}) == {'lanes'}
  assert checks({**road, 'lanes': 'two'}) == {'lanes'}
  assert checks({'highway': 'primary', 'oneway': 'yes', 'lanes': '2', 'lanes:backward': '1'}) == {'lanes'}
  assert checks({**road, 'turn:lanes:forward': 'left|through|through'}) == {'count'}
  assert checks({**road, 'turn:lanes:backward': 'left|through'}) == {'count'}
  assert checks({**road, 'width:lanes': '3|3|3'}) == set()  # all the lanes of a two-way way
  assert checks({**road, 'destination:lanes:forward': 'A'}) == {'count'}


def test_values():
  road = {'highway': 'primary', 'oneway': 'yes', 'lanes': '2'}
  assert checks({**road, 'turn:lanes': 'left;through|straight'}) == {'turn'}
  assert checks({**road, 'turn:lanes': 'left||'}) == {'count'}
  assert checks({**road, 'turn:lanes': 'left|'}) == set()  # an empty lane has no arrow
  assert checks({**road, 'change:lanes': 'yes|maybe'}) == {'change'}
  assert checks({**road, 'divider': 'painted'}) == {'divider'}
  assert checks({**road, 'divider': 'barrier'}) == set()
  assert checks({**road, 'width': 'wide'}) == {'width'}


def test_widths():
  road = {'highway': 'primary', 'lanes': '2', 'width': '10'}
  assert checks({**road, 'width:lanes:forward': '5.5', 'width:lanes:backward': '5.5'}) == {'width'}
  assert checks({**road, 'width:lanes:forward': '5.5', 'width:lanes:backward': '4'}) == set()  # a 0.5 m median
  assert checks({**road, 'width:lanes:forward': '10'}) == {'width'}  # none left for the other lane


def test_placement():
  road = {'highway': 'primary', 'lanes': '3', 'lanes:forward': '2', 'lanes:backward': '1', 'width': '10.5'}
  assert checks({**road, 'placement:forward': 'left_of:1', 'placement:backward': 'left_of:1'}) == set()
  assert checks({**road, 'placement:forward': 'left_of:3'}) == {'placement'}
  assert checks({**road, 'placement:forward': 'left:1'}) == {'placement'}
  assert checks({**road, 'placement:forward': 'left_of:1', 'placement:backward': 'right_of:1'}) == {'placement'}
  assert checks({**road, 'placement': 'transition'}, length=25) == set()
  assert checks({**road, 'placement': 'transition'}, length=60) == {'transition'}


def test_arrow_without_its_way_out():
  # the right turn is banned by the fixture's restriction
  issues = validate_text('turn_bay.osm', "v=\"left|through\"", "v=\"left|through;right\"")
  assert [(i.check, i.osm) for i in issues] == [('exit', 'w2')] and 'lane 2 right' in issues[0].detail
  issues = validate_text('freeway.osm', 'through|through|through|slight_right', 'slight_left|through|through|slight_right')
  assert [i.detail for i in issues] == ['turn:lanes lane 1 slight_left: no allowed way out that way']


def test_connectivity():
  assert [i.check for i in validate_text('turn_bay.osm', "v=\"1:1\"", "v=\"1:2\"")] == ['connectivity']  # one lane out
  assert [i.check for i in validate_text('turn_bay.osm', "v=\"1:1\"", "v=\"3:1\"")] == ['connectivity']  # two lanes in
  assert [i.check for i in validate_text('turn_bay.osm', "v=\"1:1\"", "v=\"1:(1)|2:1,bw\"")] == []
  assert [i.check for i in validate_text('turn_bay.osm', "v=\"1:1\"", "v=\"1-1\"")] == ['connectivity']
  issues = validate_text('turn_bay.osm', "<member type='node' ref='3' role='via'/>", '')
  assert [i.check for i in issues] == ['connectivity']


if __name__ == '__main__':
  for name, test in list(globals().items()):
    if name.startswith('test_'):
      test()
      print(f'{name} ok')
