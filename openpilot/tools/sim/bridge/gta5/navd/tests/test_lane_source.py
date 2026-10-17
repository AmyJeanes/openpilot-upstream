"""The model's lane (modelV2.laneHead) as nav's lane source (lane_source "model" / "fused", model_lane_robust): hysteresis
and confidence, dropouts, the map's and the model's counts disagreeing, and the lane nav's own change was into."""
from openpilot.tools.sim.bridge.gta5.navd.inputs import NavInputs
from openpilot.tools.sim.bridge.gta5.navd.planner import Fork, Planner, Through, Tune, Turn


def planner(source: str = "model", **tune) -> Planner:
  p = Planner(Tune(""))
  p.tune.values.update(lane_source=source, **tune)
  return p


def feed(p: Planner, t0: float, secs: float, reading, truth=None, dt: float = 0.1) -> float:
  """laneHead readings ([i, n, prob] or None) every dt s for secs s from t0; returns the time after."""
  t = t0
  while t < t0 + secs - 1e-9:
    p.now = t
    p._read_lane(truth, t, reading)
    t += dt
  return t


def test_flicker_doesnt_move_it():
  p = planner()
  t = feed(p, 0.0, 1.5, [1, 3, 0.9])
  assert p.lane == (1, 3) and p.lane_src == "model"
  t = feed(p, t, 0.6, [2, 3, 0.8])  # a misread, shorter than model_lane_hold
  assert p.lane == (1, 3)
  t = feed(p, t, 1.2, [2, 3, 0.8])  # held: taken
  assert p.lane == (2, 3)


def test_hysteresis_on_the_probability():
  p = planner()
  t = feed(p, 0.0, 1.5, [1, 3, 0.9])
  t = feed(p, t, 5.0, [1, 3, 0.5])  # below model_lane_prob, above model_lane_keep: still read, kept
  assert p.lane == (1, 3)
  # an unsure other lane doesn't take it, nor does it count as reading the one held
  t = feed(p, t, 2.0, [2, 3, 0.5])
  assert p.lane == (1, 3)
  t = feed(p, t, 1.5, [2, 3, 0.5])
  assert p.lane is None  # model_lane_stale since the last sure reading of it
  # today's: unsure for model_lane_hold, and it's gone
  q = planner(model_lane_robust=False)
  t = feed(q, 0.0, 1.5, [1, 3, 0.9])
  feed(q, t, 1.2, [1, 3, 0.5])
  assert q.lane is None


def test_dropouts():
  p = planner()
  t = feed(p, 0.0, 1.5, [0, 2, 0.9])
  t = feed(p, t, 2.5, None)
  assert p.lane == (0, 2)
  feed(p, t, 1.0, None)
  assert p.lane is None


def test_counted_from_the_side_that_matters():
  """The map has three lanes, the model sees two (a wide kerb lane, or one the map adds): its right lane is the map's
  right one for a right turn ahead, its left the map's left for a left turn."""
  for side, want in (("right", (2, 3)), ("left", (1, 3))):
    p = planner()
    p.map_count = 3
    p.count_side = p._side_that_matters([Turn(150.0, side, 0.0)])
    feed(p, 0.0, 1.5, [1, 2, 0.9])
    assert p.lane == want, side
  # no count disagreement: as read
  p = planner()
  p.map_count, p.count_side = 2, "right"
  feed(p, 0.0, 1.5, [0, 2, 0.9])
  assert p.lane == (0, 2)
  # off: from the left, as the model counts
  p = planner(model_count_side=False)
  p.map_count, p.count_side = 3, "right"
  feed(p, 0.0, 1.5, [1, 2, 0.9])
  assert p.lane == (1, 3)


def test_side_that_matters():
  p = planner()
  p.map_count = 3
  assert p._side_that_matters([Fork(100.0, "right", 1, 3, True), Turn(300.0, "left", 0.0)]) == "right"
  assert p._side_that_matters([Through(80.0, (0, 1, 3))]) == "left"
  assert p._side_that_matters([Through(80.0, (1, 2, 3))]) == "right"
  p.count_side = "right"
  assert p._side_that_matters([]) == "right"  # nothing ahead needs a side: as before


def test_map_count():
  assert Planner._map_count(NavInputs(lane_counts=[[0.0, 3], [200.0, 2]])) == 3
  assert Planner._map_count(NavInputs(lane_maps=[[0.0, [1, 2], 3]], lane_counts=[[0.0, 3]])) == 2  # a lane still opening
  assert Planner._map_count(NavInputs()) is None


def test_the_lane_a_change_was_into():
  """nav's change right ends (the driver's stalk off): the model's reading of the lane it was into is taken at once
  (model_lane_after_change), and with no reading the lane held moves over with the car."""
  for reading, secs, want in (([2, 3, 0.8], 0.4, (2, 3)), (None, 0.4, (2, 3)), ([1, 3, 0.9], 0.4, (1, 3))):
    p = planner()
    t = feed(p, 0.0, 1.5, [1, 3, 0.9])
    p.model_expect = ((2, 3), t)
    feed(p, t, secs, reading)
    assert p.lane == want, reading


def test_fused():
  """The model's lane where it counts as many lanes as the map's reading, else the map's; the model's where the map
  has none (a junction's area). In GTA the map's reading is the truth, so where they disagree it wins."""
  p = planner("fused")
  p.map_count = 3
  feed(p, 0.0, 1.5, [1, 2, 0.9], truth=[1, 3])
  assert p.lane == (1, 3) and p.lane_src == "map"
  p = planner("fused")
  p.map_count = 3
  feed(p, 0.0, 1.5, [2, 3, 0.9], truth=[1, 3])
  assert p.lane == (2, 3) and p.lane_src == "model"
  t = feed(p, 2.0, 1.5, [2, 3, 0.9], truth=None)
  feed(p, t, 4.0, [2, 3, 0.9], truth=None)  # past LANE_STALE without the map's
  assert p.lane == (2, 3) and p.lane_src == "model"


def test_sure_wrong_without_truth():
  """Surely out of lanes 1-1 by the model alone: its lane held model_sure_wrong s, and read sure in the last second."""
  p = planner()
  t = feed(p, 0.0, 1.5, [0, 2, 0.9])
  assert not p._sure_wrong(1, 1)  # taken only just now: may be a misread
  t = feed(p, t, 3.0, [0, 2, 0.9])
  assert p._sure_wrong(1, 1) and not p._sure_wrong(0, 1)
  feed(p, t, 1.5, [0, 2, 0.5])  # read, but not surely, for over a second
  assert not p._sure_wrong(1, 1)
