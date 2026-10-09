"""The map debug overlay's encoder and the GPS route (gta5_overlay.py). The map tests use GTA's roads from
GTA5_MAP/paths.jsonl (default ~/gta5map), and the lines of a lane-tagged map from GTA5_LANES_MAP/gta5.osm.pbf (default
~/gta5map_lanes), and are skipped without them; `python test_overlay.py` prints the overlay's size and time per update
at test places, and `python test_overlay.py startup` how the bridge's frame loop fares while the overlay first gets its
roads: building the lane tags' lines in its thread (as before the cache), in the background, and from the cache."""
import os
import sys
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.tools.sim.bridge.gta5 import gta5_overlay as ov
from openpilot.selfdrive.navd.planner import Planner
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes
from openpilot.tools.sim.bridge.gta5.map.paths import Paths, wrap
from openpilot.tools.sim.bridge.gta5.map.router import Route

PATHS = os.path.join(os.path.expanduser(os.getenv("GTA5_MAP", "~/gta5map")), "paths.jsonl")
LANES = os.path.join(os.path.expanduser(os.getenv("GTA5_LANES_MAP", "~/gta5map_lanes")), "gta5.osm.pbf")
# x, y, z, heading: e2e trips' starts (~/gta5test/e2e): L7 before its X-shaped dual carriageway junction, X1 on a freeway
PLACES = {"L7": (-512.5, -914.1, 24.5, 152.0), "X1": (-379.7, -650.6, 36.2, 0.0)}
SAME_LEVEL = 3.0  # m; GTA's stacked interchange ramps are ~5 m apart


def decode(msg: dict) -> list[tuple[str, np.ndarray]]:
  """debugGeo's g as the plugin parses it (core.cpp ParseDebugGeo)."""
  origin = np.array([msg["ox"], msg["oy"], msg["oz"]])
  out = []
  for part in msg["g"].split(";") if msg["g"] else []:
    q = np.array([int(v) for v in part[1:].split(",")], np.int64).reshape(-1, 3)
    out.append((part[0], origin + np.cumsum(q, axis=0) / 10.0))
  return out


def test_encode_round_trip():
  rng = np.random.default_rng(0)
  items = [("e", rng.uniform(-150, 150, (7, 3))), ("m", np.array([[3.21, -4.56, 1.0]]))]
  origin = np.array([100.0, -200.0, 30.0])
  items = [(k, line + origin) for k, line in items]
  msg = {"ox": 100.0, "oy": -200.0, "oz": 30.0, "g": ov.encode(origin, items)}
  for (k, line), (k2, got) in zip(items, decode(msg), strict=True):
    assert k == k2
    np.testing.assert_allclose(got, line, atol=0.051)


def test_chain_joins_through_shared_points():
  pts = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]], float)
  segs = np.stack([pts[[1, 2]], pts[[0, 1]], pts[[2, 3]]])
  lines = ov.chain(segs)
  assert len(lines) == 1
  np.testing.assert_allclose(lines[0], pts)
  # a fork: three segments meeting at a point don't join through it
  fork = np.concatenate([segs, [[[20, 0, 0], [20, 10, 0]]]])
  assert len(ov.chain(fork)) == 3


def test_simplify_and_hull():
  line = np.column_stack([np.linspace(0, 100, 51), np.zeros(51), np.zeros(51)])
  assert len(ov.simplify(line, 0.15)) == 2
  hull = ov.convex_hull(np.array([[0, 0], [2, 0], [2, 2], [0, 2], [1, 1]], float))
  assert len(hull) == 4 and not any((hull == [1, 1]).all(1))


def test_ribbon_line_drops_jogs_keeps_turns():
  pts = np.array([[0, 0, 0], [0, 20, 0], [0, 20.5, 0], [2.5, 21, 0], [2.5, 40, 0], [2.5, 60, 0], [30, 60, 0]], float)
  out = ov.ribbon_line(pts)
  assert out[:, :2].tolist() == [[0, 0], [2.5, 21], [2.5, 40], [2.5, 60], [30, 60]]


def straight_route() -> Route:
  return Route(np.column_stack([np.zeros(101), np.arange(101) * 10.0]), [])


def lane_from(route: Route, at: float, x: float) -> np.ndarray:
  """Nav's lane plan line as gta5_world caches it: planned with the car at m along the route, x m right of it."""
  s = np.arange(at, route.along[-1], 2.0)
  return np.column_stack([np.full(len(s), x), s])


def ribbon_snap(route: Route, lane: np.ndarray, v: float = 20.0, **kw) -> dict:
  return {"pos": [0.5, route.at, ov.CAR_HEIGHT], "layers": "rm", "route": route, "v": v, "lane_line": lane, **kw}


def test_ribbon_starts_under_the_car_from_a_stale_plan():
  # nav's plan from 10 m back, down the middle of the car's lane, the car 1.25 m left of it: the ribbon starts where
  # the car will be, under it, and is on the plan's line RIBBON_EASE m on (a placed car saw it hook in from beside it)
  route = straight_route()
  route.at = 100.0
  items = ov.Ribbon().items(ribbon_snap(route, lane_from(route, 90.0, 1.75)))
  r = [line for k, line in items if k == "r"]
  assert len(r) == 1
  start = 100.0 + 20.0 * ov.ROUTE_LEAD
  np.testing.assert_allclose(r[0][0], [0.5, start, ov.RIBBON_LIFT], atol=1e-6)
  on = r[0][r[0][:, 1] >= start + ov.RIBBON_EASE]
  assert len(on) and np.all(np.abs(on[:, 0] - 1.75) < 0.01)
  assert np.all(np.diff(r[0][:, 0]) >= -1e-6)  # easing over, no hook


def test_ribbon_behind_is_where_it_was_drawn():
  route = straight_route()
  ribbon = ov.Ribbon()
  for y in np.arange(100.0, 141.0, 2.0):  # nav's plan moves a lane left at 120 m, and the car with it
    route.at = y
    x = 1.75 if y < 120 else -1.75
    items = ribbon.items({**ribbon_snap(route, lane_from(route, y - 3.0, x)), "pos": [x, y, ov.CAR_HEIGHT]})
  r = [line for k, line in items if k == "r"]
  b = [line for k, line in items if k == "b"]
  assert len(r) == len(b) == 1
  r, b = r[0], b[0]
  np.testing.assert_allclose(b[-1], r[0])  # they meet
  assert b[0, 1] >= r[0, 1] - ov.BEHIND - 0.5
  # in the lanes the ribbon was in as the car passed, not between them (the route's carriageway line, x 0 here)
  before, after = b[b[:, 1] < 118, 0], b[b[:, 1] > 122, 0]
  assert len(before) and len(after) and np.all(np.abs(before - 1.75) < 0.05) and np.all(np.abs(after + 1.75) < 0.05)
  route.at = 400.0  # a jump: it starts again
  items = ribbon.items(ribbon_snap(route, lane_from(route, 400.0, 1.75), v=0.0))
  assert not [line for k, line in items if k == "b"]


def test_ribbon_behind_only_where_the_car_went():
  # the car placed on a road, stopped: nav's first plan, before the car's lane is known, ramps in from another lane
  # past the car; the next is from the car's lane. Neither leaves a route behind the car, which hasn't moved
  route = straight_route()
  route.at = 100.0
  ramp = np.column_stack([np.interp(np.arange(100.0, 300.0, 2.0), [100.0, 106.0], [-6.0, 0.5]), np.arange(100.0, 300.0, 2.0)])
  ribbon = ov.Ribbon()
  for lane in (ramp, ramp, lane_from(route, 100.0, 0.5), lane_from(route, 100.0, 0.5)):
    items = ribbon.items(ribbon_snap(route, lane, v=0.0))
    assert not [line for k, line in items if k == "b"]
  for y in np.arange(100.0, 121.0, 2.0):  # then it drives on: the route behind is where it went
    route.at = y
    items = ribbon.items(ribbon_snap(route, lane_from(route, 100.0, 0.5), v=0.0))
  b = [line for k, line in items if k == "b"]
  assert len(b) == 1 and np.all(np.abs(b[0][:, 0] - 0.5) < 0.05) and b[0][0, 1] >= 99.0


def test_ribbon_behind_starts_again_for_a_route_the_other_way():
  # the car turning round in a junction: the route behind it was north along x = 1.75; the new route from where it
  # stands heads back south. The old line isn't kept behind the car, crossing the new one
  route = straight_route()
  ribbon = ov.Ribbon()
  for y in np.arange(100.0, 131.0, 2.0):
    route.at = y
    ribbon.items(ribbon_snap(route, lane_from(route, y - 3.0, 1.75)))
  assert len(ribbon.trail) >= 2
  back = Route(np.column_stack([np.zeros(31), 130.0 - np.arange(31) * 10.0]), [])
  south = np.column_stack([np.full(60, -1.75), 132.0 - np.arange(60) * 2.0])
  items = ribbon.items({**ribbon_snap(back, south, v=0.0), "pos": [0.0, 130.0, ov.CAR_HEIGHT]})
  b = [line for k, line in items if k == "b"]
  assert not b or all(np.all(line[:, 1] >= 129.0) for line in b), b
  assert ov.trail_turn(np.array([[0.0, 0.0], [0.0, 5.0]]), south, ov.arc(south), 10.0) > 170.0


def test_route_goes_between_full_updates_with_the_rest_as_sent():
  route = straight_route()
  route.at = 100.0
  overlay = ov.Overlay(background=False)
  full = overlay.make(ribbon_snap(route, lane_from(route, 100.0, 1.75), full=True, turn=np.array([0.0, 300.0])))
  route.at = 102.0
  part = overlay.make(ribbon_snap(route, lane_from(route, 100.0, 1.75), full=False))
  assert (part["ox"], part["oy"], part["oz"]) == (full["ox"], full["oy"], full["oz"])
  a, b = decode(full), decode(part)
  assert [line.tolist() for k, line in a if k == "m"] == [line.tolist() for k, line in b if k == "m"] == [[[0.0, 300.0, 0.0]]]
  ra, rb = (next(line for k, line in items if k == "r") for items in (a, b))
  assert abs(rb[0, 1] - ra[0, 1] - 2.0) < 0.11
  assert part["n"] == sum(len(line) for _, line in b) <= ov.MAX_POINTS
  assert overlay.stats["route_ms"] < 50


def test_update_sends_the_route_more_often(monkeypatch):
  now = [0.0]
  monkeypatch.setattr(ov, "time", SimpleNamespace(monotonic=lambda: now[0]))
  route = straight_route()
  overlay = ov.Overlay(background=False)
  overlay.thread = object()  # no worker: the snapshots are taken here
  state = {"pos": [0.0, 0.0, 0.6], "vEgo": 10.0, "debug": {"on": True, "layers": "r"}}
  fulls = []
  for ms in range(0, 1000, 10):
    now[0] = ms / 1000
    overlay.update(state, route, None, lambda: None, None, False)
    if overlay.snap is not None:
      fulls.append(overlay.snap["full"])
      overlay.snap = None
  assert fulls.count(True) == 2 and abs(len(fulls) - 1.0 / ov.ROUTE_EVERY) <= 1
  now[0] = 1.0
  overlay.update(state, route, None, lambda: None, None, False)
  now[0] = 1.0 + ov.ROUTE_EVERY
  overlay.update(state, route, None, lambda: None, None, False)
  assert overlay.snap["full"]  # one the worker hasn't got to yet stays full


def drive(overlay: ov.Overlay, route, monkeypatch, steps: int = 150, paths=None, osm=None, pos=None) -> list[dict]:
  """Overlay.update every 10 ms as the bridge calls it, the car driving along a straight route (or standing at pos) and
  nav's lane plan line planned again every 0.5 s: the messages it hands on. The worker is kept in step: a thread's
  snapshots are made here, the process waited for."""
  now = [0.0]
  monkeypatch.setattr(ov, "time", SimpleNamespace(monotonic=lambda: now[0]))
  inline = not isinstance(overlay, ov.OverlayProcess)
  if inline:
    overlay.thread = object()
  start, lane, sent = route.at, None, []
  for n in range(steps):
    now[0] = n * 0.01
    if pos is None:
      route.at = start + n * 0.2
    if n % 50 == 0:
      lane = lane_from(route, route.at, 1.75 if n < 100 else -1.75).tolist() if pos is None else route.rest()
    state = {"pos": [0.5, route.at, ov.CAR_HEIGHT] if pos is None else pos, "vEgo": 20.0, "route": [[0.0, 0.0]],
             "debug": {"on": True, "layers": ov.DEFAULT_LAYERS + "f"}}
    sent += overlay.update(state, route, paths, lambda lane=lane: lane, lambda: (np.array([0.0, 300.0]), np.array([0.0, 250.0])),
                           False, osm)
    if inline:
      if overlay.snap is not None:
        snap, overlay.snap = overlay.snap, None
        overlay.outbox = overlay.make(snap)
    else:
      overlay.wait()
  return sent


def test_process_makes_what_the_thread_makes(monkeypatch):
  thread = drive(ov.Overlay(background=False), straight_route(), monkeypatch)
  process = ov.OverlayProcess(wait_maps=True)
  try:
    assert drive(process, straight_route(), monkeypatch) == thread
    assert not process.failed and process.proc is not None
  finally:
    process.close()
  assert len(thread) > 10 and any(k == "m" for msg in thread for k, _ in decode(msg))


def test_maps_not_read_from_their_files_stay_in_a_thread(monkeypatch):
  osm = SimpleNamespace(path=None, project=to_game, defaults=None, drive_on_right=True, tagged=False)
  assert ov.from_files(None, None) and not ov.from_files(None, osm)
  overlay = ov.OverlayProcess()
  drive(overlay, straight_route(), monkeypatch, steps=60, osm=osm)
  assert overlay.failed and overlay.proc is None and overlay.thread is not None


def test_within_splits_at_the_radius():
  line = np.column_stack([np.linspace(-300, 300, 61), np.zeros(61), np.zeros(61)])
  runs = ov.within(line, np.zeros(2), 150.0)
  assert len(runs) == 1 and runs[0][0, 0] == -160 and runs[0][-1, 0] == 160


def test_clip_outside_cuts_out_areas():
  square = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], float)  # counterclockwise
  segs = np.array([[[-5, 5, 1], [15, 5, 3]], [[-5, 20, 0], [15, 20, 0]], [[2, 2, 0], [8, 8, 0]]], float)
  out = ov.clip_outside(segs, [square])
  assert len(out) == 3  # the crossing one in two pieces, the one outside whole, the one inside gone
  np.testing.assert_allclose(out[1:], [[[-5, 5, 1], [0, 5, 1.5]], [[10, 5, 2.5], [15, 5, 3]]])


@pytest.fixture(autouse=True, scope="module")
def cache_dir():
  """The overlay's cache in a folder of the tests' own, not the bridge's."""
  old = ov.CACHE_DIR
  with tempfile.TemporaryDirectory() as d:
    ov.CACHE_DIR = d
    yield d
  ov.CACHE_DIR = old


def test_marks_key_follows_the_map_files_and_settings(tmp_path):
  from openpilot.tools.sim.bridge.gta5.map.osm_lanes import GTA
  (tmp_path / "paths.jsonl").write_text("a")
  (tmp_path / "gta5.osm.pbf").write_bytes(b"b")
  paths = SimpleNamespace(path=str(tmp_path / "paths.jsonl"))
  osm = SimpleNamespace(path=str(tmp_path / "gta5.osm.pbf"), project=to_game, defaults=GTA, drive_on_right=True)
  key = ov.marks_key(paths, osm)
  assert key is not None and key == ov.marks_key(paths, osm)
  (tmp_path / "paths.jsonl").write_text("a2")
  assert ov.marks_key(paths, osm) not in (None, key)
  key = ov.marks_key(paths, osm)
  osm.drive_on_right = False
  assert ov.marks_key(paths, osm) not in (None, key)
  osm.project = lambda lat, lon: (lon, lat)  # read some other way than the background build reads them
  assert ov.marks_key(paths, osm) is None
  assert ov.marks_key(SimpleNamespace(), osm) is None


def test_saved_marks_load_exactly():
  rng = np.random.default_rng(1)
  marks = {"segs": rng.normal(size=(5, 2, 3)) * 1000, "kinds": np.array(list("edwcy"), "<U1"),
           "nodes": rng.integers(0, 10 ** 6, (5, 2))}
  ov.save_marks("t" * 32, marks)
  got = ov.load_marks("t" * 32)
  assert got is not None
  for k, v in marks.items():
    assert got[k].dtype == v.dtype and np.array_equal(got[k], v)
  assert ov.load_marks("u" * 32) is None
  with open(ov.marks_file("u" * 32), "wb") as f:
    f.write(b"torn")
  assert ov.load_marks("u" * 32) is None


def test_median_edges_take_their_own_kind():
  from openpilot.tools.sim.bridge.gta5.map.osm_lanes import CENTRE, MEDIAN, Line
  assert ov.marking_kinds(Line(MEDIAN, 3.0, "solid")) == [("c", 0.0)]
  assert ov.marking_kinds(Line(MEDIAN, 3.0, "double_solid")) == [("c", -ov.DOUBLE), ("c", ov.DOUBLE)]
  assert ov.marking_kinds(Line(CENTRE, 3.0, "double_solid")) == [("c", -ov.DOUBLE), ("c", ov.DOUBLE)]


def test_z_along_a_clipped_piece():
  from openpilot.tools.sim.bridge.gta5.map.osm_to_roads import z_along
  line = np.array([[0, 0], [10, 0], [10, 10]], float)
  z = np.array([0.0, 10.0, 20.0])
  np.testing.assert_allclose(z_along(np.array([[2.5, 0], [10, 5]]), line, z), [2.5, 15.0])
  np.testing.assert_allclose(z_along(np.column_stack([np.linspace(0, 10, 200), np.zeros(200)]), line, z)[[0, -1]], [0.0, 10.0])


def test_gps_route_sends_when_it_changes_or_the_plugin_lost_it():
  pts = np.column_stack([np.zeros(200), np.arange(200) * 5.0])
  route = Route(pts, [])
  gps = ov.GpsRoute()
  assert gps.update({"gpsRoute": {"on": False}}, route) == []
  sent = gps.update({"gpsRoute": {"on": True, "points": 0}}, route)
  assert sent and sent[0]["type"] == "gpsPoints"
  assert len(sent[0]["p"].split(";")) == 2  # a straight line decimates to its ends
  assert gps.update({"gpsRoute": {"on": True, "points": 2}}, route) == []
  gps.sent_t = -1e9
  assert gps.update({"gpsRoute": {"on": True, "points": 0}}, route)  # a reloaded plugin gets it again
  assert gps.update({"gpsRoute": {"on": True, "points": 2}}, None) == [{"type": "gpsPoints", "p": ""}]


def test_gps_route_capped():
  zigzag = np.column_stack([(np.arange(400) % 2) * 20.0, np.arange(400) * 20.0])
  p, reach = ov.gps_points(Route(zigzag, []), max_points=50)
  assert len(p.split(";")) == 50 and reach < 8000


@pytest.fixture(scope="module")
def paths():
  if not os.path.exists(PATHS):
    pytest.skip(f"no {PATHS}")
  p = Paths(PATHS)
  p.index()
  return p


@pytest.fixture(scope="module")
def osm():
  if not os.path.exists(LANES):
    pytest.skip(f"no {LANES}")
  return OsmLanes.load(LANES, to_game)


def drive_route(paths: Paths, place: str, length: float = 700.0, turn_after: float = 80.0, osm=None) -> Route:
  """A route from a place along GTA's roads, straight on but for a left turn at the first junction past turn_after m."""
  x, y, z, heading = PLACES[place]
  snapped = paths.snap(np.array([x, y]), z, heading)
  assert snapped is not None
  start, h, node = snapped
  nodes, done, turned = [node], 0.0, False
  pts = [start, paths.xy[node]]
  while done < length:
    i = nodes[-1]
    outs = [j for j in paths.out.get(i, ()) if j not in nodes and not paths.links[(i, j)].shortcut]
    if not outs:
      break
    h_in = paths.link_heading(nodes[-2], i) if len(nodes) > 1 else h
    rel = {j: wrap(paths.link_heading(i, j) - h_in) for j in outs}
    lefts = [j for j in outs if 45 < rel[j] < 135]
    if not turned and done > turn_after and paths.junction(i) and lefts:
      j, turned = lefts[0], True
    else:
      j = min(outs, key=lambda j: abs(rel[j]))
    done += float(np.hypot(*(paths.xy[j] - paths.xy[i])))
    nodes.append(j)
    pts.append(paths.xy[j])
  route = Route(np.array(pts, float), [], paths, osm=osm)
  route.locate(np.array([x, y]), z, heading)
  return route


def overlay_update(paths: Paths, place: str, overlay: ov.Overlay | None = None, osm=None, full: bool = True,
                   lane: bool = False) -> tuple[dict, dict]:
  """A full update, or (full False) the route's alone; with lane, the route's own line stands in for nav's lane plan."""
  overlay = overlay or ov.Overlay()
  route = drive_route(paths, place, osm=osm)
  x, y, z, heading = PLACES[place]
  state = {"pos": [x, y, z + ov.CAR_HEIGHT], "heading": heading, "vEgo": 10.0, **route.info(1000.0),
           "route": route.ahead(1000.0, 5.0).round(1).tolist()}
  nav = Planner()
  nav.v = 10.0
  snap = {"pos": state["pos"], "layers": ov.DEFAULT_LAYERS + "f", "route": route, "paths": paths, "osm": osm,
          "recording": False, "lane_line": route.rest() if lane else None, "v": 10.0, "full": full}
  points = nav.turn_points(np.array(state["route"]), state.get("forks"), state.get("stops"), state.get("junctions"))
  if points is not None:
    snap["turn"], snap["signal"] = points
  msg = overlay.make(snap)
  return msg, dict(overlay.stats)


def check_overlay(paths, place, osm=None, carried=()) -> set:
  """`carried`: the centres of the junctions a road's lines are carried across (Junction.carried)."""
  overlay = ov.Overlay(background=False)
  overlay_update(paths, place, overlay, osm)  # builds the road geometry
  msg, stats = overlay_update(paths, place, overlay, osm)
  items = decode(msg)
  kinds = {k for k, _ in items}
  assert {"e", "r"} <= kinds and kinds & set("dwcy"), kinds
  if place == "L7":  # the route turns left at its junction
    assert {"j", "m", "g"} <= kinds, kinds
  # no lane edges or dividers inside junction areas on their own level (bridges and stacked ramps pass over them), but
  # the lines of a road carried on through one
  areas = [line[:-1] for k, line in items if k == "j"]
  for k, line in items:
    if k in "edwcy":
      mids = (line[1:] + line[:-1]) / 2
      for area in areas:
        poly = area[:, :2]
        if k != "e" and any(((poly.min(0) <= c) & (c <= poly.max(0))).all() for c in carried):
          continue
        e = np.roll(poly, -1, axis=0) - poly
        rel = mids[:, None, :2] - poly[None]
        # well inside every edge: a kerb on the area's outline lies within the outline's simplification of it
        depth = (e[None, :, 0] * rel[..., 1] - e[None, :, 1] * rel[..., 0]) / np.hypot(*e.T)[None]
        inside = (depth > ov.AREA_SIMPLIFY + 0.2).all(1) & (np.abs(mids[:, 2] - area[:, 2].mean()) < SAME_LEVEL)
        assert not inside.any(), (k, mids[inside][:3])
  assert len(msg["g"]) <= ov.MAX_CHARS and msg["n"] <= ov.MAX_POINTS
  assert stats["ms"] < 250
  x, y, z, _ = PLACES[place]
  near = np.concatenate([line for _, line in items])
  assert np.hypot(near[:, 0] - x, near[:, 1] - y).min() < 15  # the car's own road
  assert np.abs(near[:, 2] - z).max() < ov.LEVEL + 1
  return kinds


@pytest.mark.parametrize("place", sorted(PLACES))
def test_overlay_at_places(paths, place):
  check_overlay(paths, place)


@pytest.mark.parametrize("place", sorted(PLACES))
def test_overlay_lines_from_lane_tags(paths, osm, carried, place):
  kinds = check_overlay(paths, place, osm, carried)
  assert "d" in kinds and kinds & set("cy" if place == "L7" else "dw"), kinds  # L7: a two-way road, X1 a freeway


@pytest.fixture(scope="module")
def carried(osm):
  from openpilot.tools.sim.bridge.gta5.map.junctions import Junctions
  from openpilot.tools.sim.bridge.gta5.map.osm_to_roads import drawn
  return [j.centre for j in Junctions(osm, drawn).junctions if j.carried]


@pytest.fixture(scope="module")
def fresh_marks(paths, osm):
  return ov.road_marks(paths, osm)


def test_route_heights_from_beside_the_route_not_its_length():
  # a route climbing 10%; a lane line 3 m beside it whose length has drifted 10 m from the route's
  route = SimpleNamespace(points=np.column_stack([np.arange(0.0, 210.0, 10.0), np.zeros(21)]))
  route.along = route.points[:, 0].copy()
  route.z = 0.1 * route.along
  x = np.arange(5.0, 195.0, 1.0)
  lane = np.column_stack([x, np.full(len(x), 3.0)])
  z = ov.route_heights(route, lane, x + 10.0, fallback=-99.0)
  np.testing.assert_allclose(z, 0.1 * x, atol=1e-6)
  assert np.abs(ov.route_z(route, x + 10.0, -99.0) - 0.1 * x).max() > 0.9  # what reading by length gave
  # no route place within the window of its guess: by length, as before
  far = ov.route_heights(route, lane[:1], np.array([150.0]), fallback=-99.0, window=5.0)
  np.testing.assert_allclose(far, [15.0])


def test_z_near_reads_the_nearest_road():
  # two roads out of a junction at (0, 0, 10): one climbing east 10%, one level north
  roads = np.array([[[0, 0, 10], [50, 0, 15]], [[0, 0, 10], [0, 50, 10]]], float)
  z = ov.z_near(np.array([[20.0, 3.0], [-2.0, 30.0], [40.0, -6.0]]), roads, fallback=-99.0)
  np.testing.assert_allclose(z, [12.0, 10.0, 14.0])
  np.testing.assert_allclose(ov.z_near(np.zeros((2, 2)), np.zeros((0, 2, 3)), fallback=7.0), [7.0, 7.0])


VINEWOOD_T = np.array([909.5, 527.8])  # Fenwell Pl meets Vinewood Park Dr, which falls ~2.4 m to the T's east mouth


def test_junction_lines_take_their_roads_heights(paths, osm, fresh_marks):
  """A junction on a hill: its outline, kerbs and arrows at its roads' heights where they are, not its node's height
  across it (at the Vinewood T the east mouth was drawn 2.4 m up, out of the plugin's reach of the ground)."""
  shapes = np.split(fresh_marks["shape_pts"], np.cumsum(fresh_marks["shape_len"])[:-1])
  outline = [s for s, k in zip(shapes, fresh_marks["shape_kind"], strict=True)
             if k == "j" and np.all(s[:, :2].min(0) < VINEWOOD_T) and np.all(s[:, :2].max(0) > VINEWOOD_T)]
  assert len(outline) == 1
  near = [(i, j) for (i, j) in paths.links if np.hypot(*(paths.xy[i] - VINEWOOD_T)) < 40.0]
  roads = np.array([[[*paths.xy[i], paths.z[i]], [*paths.xy[j], paths.z[j]]] for i, j in near])
  east = outline[0][outline[0][:, 0] > outline[0][:, 0].max() - 1.0]  # across the east mouth, below the node's height
  assert len(east) >= 3 and np.all(np.abs(east[:, 2] - ov.z_near(east, roads, 0.0)) < 0.6), east
  # every point of the outline within 0.8 m of GTA's road nearest it in plan (links of the T's roads; the side road's
  # corner on the main road's kerb lies between the two roads' heights, its own road's 0.7 m below the main road's)
  np.testing.assert_allclose(outline[0][:, 2], ov.z_near(outline[0], roads, 0.0), atol=0.8)
  segs, nodes = fresh_marks["segs"], fresh_marks["nodes"]
  kerbs = segs[(fresh_marks["kinds"] == "e") & (nodes[:, 0] == nodes[:, 1]) &
               (np.hypot(*(segs[:, 0, :2] - VINEWOOD_T).T) < 16.0)].reshape(-1, 3)
  assert len(kerbs) and np.abs(kerbs[:, 2] - ov.z_near(kerbs, roads, 0.0)).max() < 0.8  # (the same corner)


def test_arrow_strokes():
  strokes = ov.arrow_strokes(frozenset({"left", "through"}))
  assert [k for k, _ in strokes] == ["T", "T", "T", "L", "L"]  # the shared shaft, then each turn's branch and head
  tips = {k: path[-1] for k, path in strokes[1::2]}
  assert tips["L"][0] < -1.0 and abs(tips["T"][0]) < 1e-9 and tips["T"][1] > 2.0
  assert [k for k, _ in ov.arrow_strokes(frozenset({"right"}))] == ["R"] * 3
  assert ov.arrow_strokes(frozenset({"none"})) == [] and ov.arrow_strokes(frozenset()) == []


def test_lane_tags_marks_have_the_new_kinds(fresh_marks):
  kinds = set(fresh_marks["kinds"].tolist()) | set(fresh_marks["shape_kind"].tolist())
  # arrows of every turn, tapers, crossings, stop lines at lights, junctions
  assert set("LTRtxlj") <= kinds, kinds
  arrows = np.isin(fresh_marks["kinds"], list("LTR"))
  print(f"{arrows.sum()} arrow segments, {(fresh_marks['kinds'] == 't').sum()} taper segments, "
        f"{(fresh_marks['shape_kind'] == 'q').sum()} lane count flags, {(fresh_marks['shape_kind'] == 'k').sum()} give way lines")


def test_cached_lines_equal_a_fresh_build(paths, osm, fresh_marks):
  key = ov.marks_key(paths, osm)
  assert key is not None
  ov.save_marks(key, fresh_marks)
  cached = ov.load_marks(key)
  assert cached is not None
  for k, v in fresh_marks.items():
    assert cached[k].dtype == v.dtype and np.array_equal(cached[k], v), k
  fresh, from_cache = ov.RoadGeometry(paths, osm, fresh_marks), ov.RoadGeometry(paths, osm, cached)
  for x, y, z, _ in PLACES.values():
    a, b = fresh.near((x, y), z, ov.RADIUS, ov.DEFAULT_LAYERS), from_cache.near((x, y), z, ov.RADIUS, ov.DEFAULT_LAYERS)
    assert len(a) == len(b) > 0
    assert all(ka == kb and np.array_equal(la, lb) for (ka, la), (kb, lb) in zip(a, b, strict=True))


@pytest.mark.parametrize("place", sorted(PLACES))
def test_process_makes_what_the_thread_makes_on_the_map(paths, osm, fresh_marks, place, monkeypatch):
  ov.save_marks(ov.marks_key(paths, osm), fresh_marks)
  x, y, z, _ = PLACES[place]
  pos = [x, y, z + ov.CAR_HEIGHT]
  thread = drive(ov.Overlay(background=False), drive_route(paths, place, osm=osm), monkeypatch, 60, paths, osm, pos)
  process = ov.OverlayProcess(wait_maps=True)  # reads the maps from their files, the lines from the cache
  try:
    got = drive(process, drive_route(paths, place, osm=osm), monkeypatch, 60, paths, osm, pos)
  finally:
    process.close()
  assert got == thread and any(k == "e" for msg in got for k, _ in decode(msg))


def test_lines_build_in_the_background_then_load_from_the_cache(paths, osm, fresh_marks, tmp_path, monkeypatch):
  monkeypatch.setattr(ov, "CACHE_DIR", str(tmp_path))
  x, y, z, _ = PLACES["L7"]
  snap = {"pos": [x, y, z + ov.CAR_HEIGHT], "layers": ov.DEFAULT_LAYERS, "route": None, "paths": paths, "osm": osm}
  overlay = ov.Overlay()
  waits, t0 = [], time.monotonic()
  while overlay.geometry is None:
    assert time.monotonic() - t0 < 300, "the background build didn't finish"
    t = time.monotonic()
    msg = overlay.make(dict(snap))
    waits.append(time.monotonic() - t)
    assert msg["type"] == "debugGeo"  # the overlay goes on meanwhile
    time.sleep(0.2)
  assert overlay.stats["geometry"] == "background" and len(waits) > 3
  assert max(waits) < 1.0, waits  # never a whole build's stall in the overlay's thread
  key = ov.marks_key(paths, osm)
  cached = ov.load_marks(key)
  assert cached is not None and all(np.array_equal(cached[k], v) for k, v in fresh_marks.items())
  # the next start: straight from the cache
  again = ov.Overlay()
  t = time.monotonic()
  again.make(dict(snap))
  assert again.geometry is not None and again.stats["geometry"] == "cache"
  assert time.monotonic() - t < 2.0


def frame_loop(paths, osm, background: bool, seconds: float, period: float = 0.05) -> dict:
  """A stand-in for the bridge's frame loop in this thread while the overlay's worker first builds its roads: every
  period s it hands the overlay a snapshot (Overlay.update, as the bridge does) and does some I/O that lets go of the
  GIL, as the bridge's sockets and pipes do, and each wake-up is timed against its deadline."""
  x, y, z, _ = PLACES["L7"]
  state = {"pos": [x, y, z + ov.CAR_HEIGHT], "debug": {"on": True, "layers": ov.DEFAULT_LAYERS}}
  overlay = ov.Overlay(background=background)
  r, w = os.pipe()
  late, ready, t0 = [], None, time.monotonic()
  deadline = t0
  while time.monotonic() - t0 < seconds:
    overlay.update(state, None, paths, lambda: None, None, False, osm)
    if ready is None and overlay.geometry is not None:
      ready = time.monotonic() - t0
    for _ in range(20):
      os.write(w, b"x")
      os.read(r, 1)
    np.hypot(np.arange(1000.0), 1.0).sum()
    deadline += period
    time.sleep(max(deadline - time.monotonic(), 0.0))
    late.append(time.monotonic() - deadline)
    deadline = max(deadline, time.monotonic() - period)  # a missed frame is gone, not owed
  os.close(r)
  os.close(w)
  late_a = np.array(late) * 1000
  return {"ready_s": None if ready is None else round(ready, 1), "ticks": len(late_a),
          "late>25ms": round(float(np.mean(late_a > 25)), 3), "p99_ms": round(float(np.percentile(late_a, 99)), 1),
          "max_ms": round(float(late_a.max()), 1), "geometry": overlay.stats.get("geometry")}


def startup(seconds: float = 25.0):
  p = Paths(PATHS)
  p.index()
  osm = OsmLanes.load(LANES, to_game)
  with tempfile.TemporaryDirectory() as d:
    ov.CACHE_DIR = d
    print("in its thread (before the cache):", frame_loop(p, osm, False, seconds), flush=True)
    for f in os.listdir(d):
      os.remove(os.path.join(d, f))
    print("first start, in the background:  ", frame_loop(p, osm, True, seconds), flush=True)
    print("next start, from the cache:      ", frame_loop(p, osm, True, seconds), flush=True)


if __name__ == "__main__" and sys.argv[1:2] == ["startup"]:
  startup(*map(float, sys.argv[2:3]))
elif __name__ == "__main__":
  p = Paths(PATHS)
  p.index()
  overlay = ov.Overlay(background=False)
  for place in sorted(PLACES):
    for n in range(3):
      t = time.monotonic()
      msg, stats = overlay_update(p, place, overlay)
      kinds: dict = {}
      for k, line in decode(msg):
        kinds[k] = kinds.get(k, 0) + len(line)
      wall = 1000 * (time.monotonic() - t)
      print(f"{place} #{n}: {len(msg['g'])} chars, {msg['n']} points, {stats} (wall {wall:.0f} ms); points by kind {kinds}")
    for lane in (False, True):
      msg, stats = overlay_update(p, place, overlay, full=False, lane=lane)
      name = f"{place} route alone{' (lane plan)' if lane else ''}"
      print(f"{name}: {len(msg['g'])} chars, {msg['n']} points, route_ms {stats['route_ms']}")
