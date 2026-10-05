"""The map debug overlay's encoder and the GPS route (gta5_overlay.py). The map tests use GTA's roads from
GTA5_MAP/paths.jsonl (default ~/gta5map) and are skipped without them; `python test_overlay.py` prints the overlay's
size and time per update at test places."""
import os
import time

import numpy as np
import pytest

from openpilot.tools.sim.bridge.gta5 import gta5_overlay as ov
from openpilot.tools.sim.bridge.gta5.gta5_nav import Nav
from openpilot.tools.sim.bridge.gta5.map.paths import Paths, wrap
from openpilot.tools.sim.bridge.gta5.map.router import Route

PATHS = os.path.join(os.path.expanduser(os.getenv("GTA5_MAP", "~/gta5map")), "paths.jsonl")
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


def drive_route(paths: Paths, place: str, length: float = 700.0, turn_after: float = 80.0) -> Route:
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
  route = Route(np.array(pts, float), [], paths)
  route.locate(np.array([x, y]), z, heading)
  return route


def overlay_update(paths: Paths, place: str, overlay: ov.Overlay | None = None) -> tuple[dict, dict]:
  overlay = overlay or ov.Overlay()
  route = drive_route(paths, place)
  x, y, z, heading = PLACES[place]
  state = {"pos": [x, y, z + ov.CAR_HEIGHT], "heading": heading, "vEgo": 10.0, **route.info(1000.0),
           "route": route.ahead(1000.0, 5.0).round(1).tolist()}
  nav = Nav(lambda m: None, lambda d: None)
  nav.v = 10.0
  snap = {"pos": state["pos"], "layers": ov.DEFAULT_LAYERS + "f", "route": route, "paths": paths, "recording": False,
          "lane_line": None}
  points = nav.turn_points(np.array(state["route"]), state)
  if points is not None:
    snap["turn"], snap["signal"] = points
  msg = overlay.make(snap)
  return msg, dict(overlay.stats)


@pytest.mark.parametrize("place", sorted(PLACES))
def test_overlay_at_places(paths, place):
  overlay = ov.Overlay()
  overlay_update(paths, place, overlay)  # builds the road geometry
  msg, stats = overlay_update(paths, place, overlay)
  items = decode(msg)
  kinds = {k for k, _ in items}
  assert {"e", "d", "r"} <= kinds, kinds
  if place == "L7":  # the route turns left at its junction
    assert {"j", "m", "g"} <= kinds, kinds
  # no lane edges or dividers inside junction areas
  areas = [line[:-1, :2] for k, line in items if k == "j"]
  for k, line in items:
    if k in "ed":
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


if __name__ == "__main__":
  p = Paths(PATHS)
  p.index()
  overlay = ov.Overlay()
  for place in sorted(PLACES):
    for n in range(3):
      t = time.monotonic()
      msg, stats = overlay_update(p, place, overlay)
      kinds: dict = {}
      for k, line in decode(msg):
        kinds[k] = kinds.get(k, 0) + len(line)
      wall = 1000 * (time.monotonic() - t)
      print(f"{place} #{n}: {len(msg['g'])} chars, {msg['n']} points, {stats} (wall {wall:.0f} ms); points by kind {kinds}")
