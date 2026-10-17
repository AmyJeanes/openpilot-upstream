import math

import numpy as np

from openpilot.selfdrive.navd.map_match import MapMatcher, RoadGraph, oneway_of, wrap


def dual_carriageway(gap: float = 12.0, length: float = 600.0, service: bool = False) -> RoadGraph:
  """Two one-way carriageways along y, northbound at x = gap / 2 and southbound at -gap / 2, nodes every 50 m, joined
  at both ends (a U-turn) so each leads to the other only far away. With `service`, also a two-way service road
  beside the northbound one, 10 m east of it, joined to it only at the ends."""
  xs = np.arange(0.0, length + 1.0, 50.0)
  xy, ways = [], []

  def line(x):
    start = len(xy)
    xy.extend((x, y) for y in xs)
    return list(range(start, len(xy)))
  north, south = line(gap / 2), line(-gap / 2)
  ways.append((1, {'highway': 'primary', 'oneway': 'yes'}, north))
  ways.append((2, {'highway': 'primary', 'oneway': 'yes'}, south[::-1]))
  ways.append((3, {'highway': 'primary', 'oneway': 'yes'}, [north[-1], south[-1]]))
  ways.append((4, {'highway': 'primary', 'oneway': 'yes'}, [south[0], north[0]]))
  if service:
    side = line(gap / 2 + 10.0)
    ways.append((5, {'highway': 'service'}, side))
    ways.append((6, {'highway': 'service'}, [north[0], side[0]]))
    ways.append((7, {'highway': 'service'}, [north[-1], side[-1]]))
  return RoadGraph(np.array(xy), ways)


def drive(mm: MapMatcher, path, v: float, offset=(0.0, 0.0), course: bool = True, jumps=None, rate: int = 10):
  """Drives a car along a polyline at v m/s: predict() every step, a fix (1 Hz) offset from the true position by
  `offset` (and by jumps[k] on the kth fix), with its course. Returns the match after each fix."""
  path = np.asarray(path, float)
  seg = np.diff(path, axis=0)
  along = np.concatenate(([0.0], np.cumsum(np.hypot(*seg.T))))
  heading = np.degrees(np.arctan2(-seg[:, 0], seg[:, 1]))
  out, dt, s, k, last = [], 1.0 / rate, 0.0, 0, None
  while s < along[-1]:
    i = min(int(np.searchsorted(along, s, side='right')) - 1, len(seg) - 1)
    h = heading[i]
    yaw = 0.0 if last is None else math.radians(wrap(h - last)) / dt
    last = h
    mm.predict(dt, v, yaw)
    if k % rate == 0:
      p = path[i] + seg[i] * (s - along[i]) / max(along[i + 1] - along[i], 1e-9) + np.asarray(offset)
      if jumps and k // rate in jumps:
        p = p + np.asarray(jumps[k // rate])
      out.append(mm.update(p[0], p[1], -h if course else None, 2.0, v))
    s += v * dt
    k += 1
  return out


def test_oneway_tags():
  assert oneway_of({'highway': 'primary', 'oneway': 'yes'}) == 1
  assert oneway_of({'highway': 'primary', 'oneway': '-1'}) == -1
  assert oneway_of({'highway': 'motorway'}) == 1
  assert oneway_of({'highway': 'primary', 'junction': 'roundabout'}) == 1
  assert oneway_of({'highway': 'residential'}) == 0


def test_graph_edges_only_the_legal_way():
  xy = np.array([(0.0, 0.0), (0.0, 50.0), (0.0, 100.0)])
  g = RoadGraph(xy, [(1, {'highway': 'primary', 'oneway': 'yes'}, [0, 1]), (2, {'highway': 'primary', 'oneway': '-1'}, [1, 2]),
                     (3, {'highway': 'residential'}, [0, 2]), (4, {'highway': 'footway'}, [0, 1])])
  edges = {(int(u), int(v)): k for k, (u, v) in enumerate(zip(g.u, g.v, strict=True))}
  assert sorted(edges) == [(0, 1), (0, 2), (2, 0), (2, 1)]  # one-ways one way, the two-way road both, no footway
  assert g.reverse[edges[(0, 2)]] == edges[(2, 0)] and g.reverse[edges[(0, 1)]] == -1


def test_dual_carriageway_fix_nearer_the_other_side():
  """A northbound car in the nearside lane (x = 8), its fixes 9 m west: nearer the southbound carriageway's line
  than its own. Nearest road would take the southbound one; the matcher keeps the car's own way throughout."""
  g = dual_carriageway()
  mm = MapMatcher(g)
  matches = drive(mm, [(8.0, 20.0), (8.0, 580.0)], 15.0, offset=(-9.0, 0.0))
  e, _, _ = g.near((-1.0, 300.0), 30.0)
  assert abs(wrap(g.heading[e[0]] - 180.0)) < 1.0  # the fix's nearest road heads south
  assert all(m is not None and abs(wrap(m.heading)) < 1.0 for m in matches)
  assert matches[-1].sure


def test_dual_carriageway_slow_and_stopped():
  """Slowing to a crawl and stopping (the course means nothing then) with the fixes drifting across the median:
  the gyro's heading keeps the match on the car's own carriageway."""
  g = dual_carriageway()
  mm = MapMatcher(g)
  drive(mm, [(8.0, 20.0), (8.0, 200.0)], 12.0, offset=(-3.0, 0.0))
  slow = drive(mm, [(8.0, 200.0), (8.0, 230.0)], 1.5, offset=(-10.0, 0.0))
  stopped = [mm.update(-2.0 + dx, 230.0, 180.0, 5.0, 0.0) for dx in (-1.0, 0.0, 1.0)]  # a stopped car's course is junk
  assert all(abs(wrap(m.heading)) < 1.0 for m in slow + stopped)


def test_unknown_heading_isnt_sure():
  g = dual_carriageway()
  mm = MapMatcher(g)
  m = mm.update(1.0, 300.0)  # no course: a fix between the carriageways
  assert m is not None and not m.sure


def test_multipath_jump_doesnt_switch_roads():
  """One fix 12 m off towards a parallel service road the main road doesn't lead to there: no switch."""
  g = dual_carriageway(service=True)
  mm = MapMatcher(g)
  matches = drive(mm, [(8.0, 20.0), (8.0, 580.0)], 15.0, jumps={15: (12.0, 0.0), 16: (12.0, 0.0)})
  assert all(int(m.way) == 1 for m in matches[2:])


def test_follows_the_car_onto_the_other_road():
  """The car truly turns onto the service road at the north end and drives it back south (a two-way road): the
  matcher follows it there within two fixes, and the right way along it."""
  g = dual_carriageway(service=True)
  mm = MapMatcher(g)
  path = [(6.0, 20.0), (6.0, 600.0), (16.0, 600.0), (16.0, 300.0)]
  matches = drive(mm, path, 12.0)
  tail = matches[-15:]
  assert all(int(m.way) == 5 and abs(wrap(m.heading - 180.0)) < 1.0 for m in tail)


def test_turn_at_a_junction():
  """A right turn at a crossroads of two-way roads: the match is on the new road within two fixes of the turn."""
  xy = np.array([(0.0, -200.0), (0.0, 0.0), (0.0, 200.0), (-200.0, 0.0), (200.0, 0.0)])
  g = RoadGraph(xy, [(1, {'highway': 'residential'}, [0, 1, 2]), (2, {'highway': 'residential'}, [3, 1, 4])])
  mm = MapMatcher(g)
  matches = drive(mm, [(2.0, -190.0), (2.0, -2.0), (190.0, -2.0)], 10.0, offset=(1.5, -2.0))
  turned = next(k for k, m in enumerate(matches) if abs(wrap(m.heading + 90.0)) < 1.0)  # east is -90 deg
  assert turned <= 21  # the turn is at 19-20 s
  assert all(abs(wrap(m.heading + 90.0)) < 1.0 for m in matches[turned:])
  assert all(abs(wrap(m.heading)) < 1.0 for m in matches[1:18])


def test_between_fixes_the_match_moves_on():
  g = dual_carriageway()
  mm = MapMatcher(g)
  drive(mm, [(6.0, 20.0), (6.0, 100.0)], 10.0)
  y0 = mm.match.point[1]
  for _ in range(5):
    mm.predict(0.1, 10.0, 0.0)
  assert abs(mm.match.point[1] - y0 - 5.0) < 0.01
