"""ynd_to_osm.py's U-turn bans and traps.py's check that restrictions cut no road off, on made-up networks. No pytest in
the venv: `python test_traps.py` runs them all."""
from openpilot.tools.sim.bridge.gta5.map.traps import cut_off, free_traps
from openpilot.tools.sim.bridge.gta5.map.ynd_to_osm import no_u_turns


def nodes_at(**xy):
  return {k: {'x': x, 'y': y} for k, (x, y) in xy.items()}


def test_hairpin_is_the_road():
  # a two-way road bending back round a hairpin over 28 m: no node on it has another way on
  nodes = nodes_at(A=(0, -50), B=(0, 20), C=(10, 30), D=(20, 20), E=(20, -50))
  ways = [(1, 'A', 'B', True), (2, 'B', 'C', True), (3, 'C', 'D', True), (4, 'D', 'E', True)]
  got = no_u_turns(nodes, ways)
  assert got and all(wi == wo for wi, via, wo in got)  # back along the same way only


def test_median_gap_is_a_u_turn():
  # a divided road's carriageways joined by a 10 m gap: through it from one to the other is a U-turn
  nodes = nodes_at(N1=(0, -50), N2=(0, 0), N3=(0, 50), S3=(10, 50), S2=(10, 0), S1=(10, -50))
  ways = [(1, 'N1', 'N2', False), (2, 'N2', 'N3', False), (3, 'S3', 'S2', False), (4, 'S2', 'S1', False),
          (5, 'N2', 'S2', True)]
  assert (1, [('w', 5)], 4) in no_u_turns(nodes, ways)


def test_acute_junction_only_way_on():
  # a one-way approach whose only way on leaves at 150 degrees, and at a node with a way on straight ahead
  nodes = nodes_at(P=(0, -40), J=(0, 0), Q=(20, -34.64), R=(-30, 10))
  ways = [(1, 'P', 'J', False), (2, 'J', 'Q', False), (3, 'R', 'J', False)]
  assert not [r for r in no_u_turns(nodes, ways) if r[0] != r[2]]
  ways.append((4, 'J', 'R', False))
  assert (1, [('n', 'J')], 2) in no_u_turns(nodes, ways)


def ring():
  nodes = nodes_at(a=(0, 0), b=(100, 0), c=(100, 100), d=(0, 100), Y=(200, 0), Z=(200, 100), W=(160, -30))
  ways = [(1, 'a', 'b', True), (2, 'b', 'c', True), (3, 'c', 'd', True), (4, 'd', 'a', True),
          (10, 'b', 'Y', False), (11, 'Y', 'Z', False), (12, 'Z', 'c', False), (13, 'Y', 'W', False), (14, 'W', 'b', False),
          (16, 'c', 'Y', False)]
  return nodes, ways


def test_cut_off():
  nodes, ways = ring()
  assert cut_off(ways, []) == (set(), set())
  # GTA's flag and a U-turn ban leave the way to Y no way on: trapped
  flag, ban = ('no_left_turn', 10, [('n', 'Y')], 11), ('no_u_turn', 10, [('n', 'Y')], 13)
  assert cut_off(ways, [flag, ban]) == ({(10, 1)}, set())
  # a ban through ways forbids only the whole move, but the only way on from 3 towards d is along 4: stuck on 3 that
  # way; nothing leads on to 1 from a, but a car gets to it the other way
  assert cut_off(ways, [('no_u_turn', 3, [('w', 4)], 1)]) == ({(3, 1)}, set())
  # nothing leads on to 13 and 14 but the way to Y
  assert cut_off(ways, [ban, ('no_straight_on', 16, [('n', 'Y')], 13)])[1] == {(13, 1), (14, 1)}


def test_free_traps():
  nodes, ways = ring()
  flag, ban = ('no_left_turn', 10, [('n', 'Y')], 11), ('no_u_turn', 10, [('n', 'Y')], 13)
  keep = ('no_right_turn', 16, [('n', 'Y')], 13)
  # the U-turn ban before GTA's flag, left out as the only way on to 13 too
  assert free_traps(nodes, ways, [flag, ban, keep]) == ([flag, keep], [ban], [])
  # out of the trap on 3 towards d, the ban is from the ways on to 3 instead: a car on 3 may leave, none goes in to
  # turn back
  through = ('no_u_turn', 3, [('w', 4)], 1)
  earlier = [('no_u_turn', 2, [('w', 3), ('w', 4)], 1), ('no_u_turn', 12, [('w', 3), ('w', 4)], 1)]
  assert free_traps(nodes, ways, [through, keep]) == ([keep, *earlier], [through], earlier)
  assert cut_off(ways, [keep, *earlier]) == (set(), set())


if __name__ == '__main__':
  for name, f in list(globals().items()):
    if name.startswith('test_'):
      f()
      print(name, 'ok')
