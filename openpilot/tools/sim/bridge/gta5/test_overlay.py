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
from openpilot.tools.sim.bridge.gta5.gta5_nav import Nav
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes
from openpilot.tools.sim.bridge.gta5.map.paths import Paths, wrap
from openpilot.tools.sim.bridge.gta5.map.router import Route

PATHS = os.path.join(os.path.expanduser(os.getenv("GTA5_MAP", "~/gta5map")), "paths.jsonl")
LANES = os.path.join(os.path.expanduser(os.getenv("GTA5_LANES_MAP", "~/gta5map_lanes")), "gta5.osm.pbf")
# x, y, z, heading: e2e trips' starts (~/gta5test/e2e): L7 before its X-shaped dual carriageway junction, X1 on a freeway
PLACES = {"L7": (-512.5, -914.1, 24.5, 152.0), "X1": (-379.7, -650.6, 36.2, 0.0)}


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


def overlay_update(paths: Paths, place: str, overlay: ov.Overlay | None = None, osm=None) -> tuple[dict, dict]:
  overlay = overlay or ov.Overlay()
  route = drive_route(paths, place, osm=osm)
  x, y, z, heading = PLACES[place]
  state = {"pos": [x, y, z + ov.CAR_HEIGHT], "heading": heading, "vEgo": 10.0, **route.info(1000.0),
           "route": route.ahead(1000.0, 5.0).round(1).tolist()}
  nav = Nav(lambda m: None, lambda d: None)
  nav.v = 10.0
  snap = {"pos": state["pos"], "layers": ov.DEFAULT_LAYERS + "f", "route": route, "paths": paths, "osm": osm,
          "recording": False, "lane_line": None}
  points = nav.turn_points(np.array(state["route"]), state)
  if points is not None:
    snap["turn"], snap["signal"] = points
  msg = overlay.make(snap)
  return msg, dict(overlay.stats)


def check_overlay(paths, place, osm=None) -> set:
  overlay = ov.Overlay(background=False)
  overlay_update(paths, place, overlay, osm)  # builds the road geometry
  msg, stats = overlay_update(paths, place, overlay, osm)
  items = decode(msg)
  kinds = {k for k, _ in items}
  assert {"e", "r"} <= kinds and kinds & set("dwcy"), kinds
  if place == "L7":  # the route turns left at its junction
    assert {"j", "m", "g"} <= kinds, kinds
  # no lane edges or dividers inside junction areas
  areas = [line[:-1, :2] for k, line in items if k == "j"]
  for k, line in items:
    if k in "edwcy":
      mids = (line[1:, :2] + line[:-1, :2]) / 2
      for poly in areas:
        e = np.roll(poly, -1, axis=0) - poly
        rel = mids[:, None] - poly[None]
        inside = ((e[None, :, 0] * rel[..., 1] - e[None, :, 1] * rel[..., 0]) > 0.2).all(1)
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
def test_overlay_lines_from_lane_tags(paths, osm, place):
  kinds = check_overlay(paths, place, osm)
  assert "d" in kinds and kinds & set("cy" if place == "L7" else "dw"), kinds  # L7: a two-way road, X1 a freeway


@pytest.fixture(scope="module")
def fresh_marks(paths, osm):
  return ov.road_marks(paths, osm)


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


if __name__ == "__main__" and sys.argv[1:] == ["startup"]:
  startup()
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
