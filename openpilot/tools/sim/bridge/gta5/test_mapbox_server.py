"""The GTA Mapbox Directions emulator (mapbox_server.py) against real Mapbox responses (selfdrive/navigation/tests/
fixtures/mapbox): the same schema, the same sparseness. On the live map's navd_sim trips (GTA5_MAP; their routes are
cached, so no router is needed)."""
import json
import sys
import threading
import types
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from openpilot.tools.sim.bridge.gta5.mapbox_server import Builder, Server, encode_polyline, junction_lanes
from openpilot.tools.sim.bridge.gta5.map.router import decode_polyline

FIXTURES = Path(__file__).parents[4] / "selfdrive" / "navigation" / "tests" / "fixtures" / "mapbox"
LANE_KEYS = {"indications", "valid", "active", "valid_indication"}


@pytest.fixture(scope="module")
def trips():
  from openpilot.tools.sim.bridge.gta5 import navd_sim as ns
  from openpilot.tools.sim.bridge.gta5.mapdrive_sim import live_map
  m = live_map()
  if m is None:
    pytest.skip("no live map (GTA5_MAP)")
  ns.use_routes()
  builder = Builder(m["osm"])
  out = {}
  for name, spec in ns.trips(40):
    got = ns.load_trip(spec)
    if got is not None:
      out[name] = (got[0], builder.directions(got[0], annotations=("maxspeed", "congestion", "distance", "duration")))
  return out


def kinds(resp: dict) -> dict[str, list[dict]]:
  """The response's objects by kind."""
  out: dict[str, list[dict]] = {k: [] for k in ("response", "route", "leg", "step", "maneuver", "intersection", "lane",
                                                "banner", "primary", "component", "annotation", "waypoint")}
  out["response"].append(resp)
  out["waypoint"] += resp.get("waypoints", [])
  for r in resp["routes"]:
    out["route"].append(r)
    for leg in r["legs"]:
      out["leg"].append(leg)
      if "annotation" in leg:
        out["annotation"].append(leg["annotation"])
      for s in leg["steps"]:
        out["step"].append(s)
        out["maneuver"].append(s["maneuver"])
        for it in s["intersections"]:
          out["intersection"].append(it)
          out["lane"] += it.get("lanes", [])
        for b in s["bannerInstructions"]:
          out["banner"].append(b)
          out["primary"].append(b["primary"])
          out["component"] += b["primary"].get("components", []) + (b.get("sub") or {}).get("components", [])
  return out


def jtype(v) -> str:
  return "number" if isinstance(v, (int, float)) and not isinstance(v, bool) else type(v).__name__


def schema(objs: list[dict]) -> tuple[dict[str, set[str]], set[str]]:
  """key -> JSON types seen, and the keys every object has."""
  types_: dict[str, set[str]] = {}
  for o in objs:
    for k, v in o.items():
      types_.setdefault(k, set()).add(jtype(v))
  common = set.intersection(*(set(o) for o in objs)) if objs else set()
  return types_, common


@pytest.fixture(scope="module")
def real():
  out: dict[str, list[dict]] = {}
  for p in sorted(FIXTURES.glob("*.json")):
    for k, objs in kinds(json.loads(p.read_text())["response"]).items():
      out.setdefault(k, []).extend(objs)
  return out


# keys a real response has on every object that the emulator doesn't fill (Mapbox internals nav doesn't read)
NOT_EMULATED = {"step": {"pronunciation"}, "intersection": {"duration", "weight", "turn_weight"}}


def test_schema_parity(trips, real):
  """Every key the emulator gives is one Mapbox gives, with the same JSON type; every key Mapbox always gives, it gives."""
  for name, (_, resp) in trips.items():
    for kind, objs in kinds(resp).items():
      if not objs:
        continue
      want, always = schema(real[kind])
      got, _ = schema(objs)
      extra = set(got) - set(want)
      assert not extra, f"{name}: {kind} has keys Mapbox lacks: {extra}"
      for k, t in got.items():
        assert t <= want[k], f"{name}: {kind}.{k} is {t}, Mapbox's {want[k]}"
      for o in objs:
        missing = always - set(o) - NOT_EMULATED.get(kind, set()) - {"modifier"}
        assert not missing, f"{name}: a {kind} lacks {missing}"


def test_geometry_and_distances(trips):
  for route, resp in trips.values():
    r = resp["routes"][0]
    pts = decode_polyline(r["geometry"])
    assert len(pts) == len(route.points)
    leg = r["legs"][0]
    assert abs(sum(s["distance"] for s in leg["steps"]) - r["distance"]) < 1.0
    assert len(leg["annotation"]["maxspeed"]) == len(pts) - 1
    gis = [it["geometry_index"] for s in leg["steps"] for it in s["intersections"]]
    assert gis == sorted(gis) and len(set(gis)) == len(gis) and gis[-1] < len(pts)
    for s in leg["steps"]:
      assert s["intersections"][0]["location"] == s["maneuver"]["location"]


def test_lanes_only_at_intersections_and_nothing_more(trips):
  """Lanes come only on intersections, with Mapbox's lane keys and no more: no openings, widths or geometry."""
  n = 0
  for _, resp in trips.values():
    for s in resp["routes"][0]["legs"][0]["steps"]:
      assert "lanes" not in s and "lanes" not in s["maneuver"]
      for it in s["intersections"]:
        for lane in it.get("lanes", []):
          n += 1
          assert set(lane) <= LANE_KEYS
          assert ("valid_indication" in lane) == lane["valid"]
  assert n > 1000


def test_bay_only_an_extra_lane_at_its_junction(trips):
  """A junction whose approach gains a lane (a bay) within its last 100 m: its intersection lists the bay among its
  lanes, as many as the map's turn:lanes there, while nothing says where the bay begins; intersections further back on
  another way carry that way's own lanes."""
  found = 0
  for route, resp in trips.values():
    info = route.info(1e9)
    counts = info["laneCounts"]

    def count(s, counts=counts):
      return next((c for d, c in reversed(counts) if d <= s), 0)
    pts = decode_polyline(resp["routes"][0]["geometry"])
    ints = {it["geometry_index"]: it for s in resp["routes"][0]["legs"][0]["steps"] for it in s["intersections"]}
    for e, arrows, *_ in info["laneArrows"]:
      if not (count(e - 2.0) > count(e - 100.0) > 0 and len(arrows) == count(e - 2.0)):
        continue
      k = int(np.argmin(np.abs(route.along - e)))
      it = ints.get(k)
      assert it is not None and len(it["lanes"]) == len(arrows) > count(e - 100.0)
      found += 1
    assert len(pts) == len(route.points)
  assert found >= 5


def test_close_maneuvers_each_have_their_lanes(trips):
  """Two maneuvers within 100 m: the first's banner shows the second's lanes, each intersection its own move's."""
  found = 0
  for _, resp in trips.values():
    steps = resp["routes"][0]["legs"][0]["steps"]
    for k in range(1, len(steps) - 1):
      a, b = steps[k], steps[k + 1]
      la, lb = a["intersections"][0].get("lanes"), b["intersections"][0].get("lanes")
      if a["distance"] >= 100 or not la or not lb or b["maneuver"]["type"] == "arrive":
        continue
      found += 1
      sub = [c for c in a["bannerInstructions"][-1].get("sub", {}).get("components", []) if c["type"] == "lane"]
      assert [c["directions"] for c in sub] == [x["indications"] for x in lb]
      assert all(x["valid"] for x in la if x.get("valid_indication")) and any(x["valid"] for x in la)
  assert found >= 3


def test_fork_and_exit_lanes(trips):
  """At a fork or exit with lanes, the valid lanes are those whose arrows go the route's way, on its side."""
  found = 0
  for _, resp in trips.values():
    for s in resp["routes"][0]["legs"][0]["steps"]:
      mv = s["maneuver"]
      lanes = s["intersections"][0].get("lanes")
      if mv["type"] not in ("fork", "off ramp") or not lanes or "modifier" not in mv or not any(x["valid"] for x in lanes):
        continue
      found += 1
      valid = [i for i, x in enumerate(lanes) if x["valid"]]
      assert valid == list(range(valid[0], valid[-1] + 1))  # side by side
      side = "left" if "left" in mv["modifier"] else "right" if "right" in mv["modifier"] else None
      if side == "left":
        assert valid[0] <= len(lanes) - 1 - valid[-1]
      elif side == "right":
        assert len(lanes) - 1 - valid[-1] <= valid[0]
  assert found >= 3


def test_junction_lanes():
  lanes = junction_lanes(["left", "left;through", "through", "right"], "left")
  assert [x["valid"] for x in lanes] == [True, True, False, False]
  assert [x.get("valid_indication") for x in lanes] == ["left", "left", None, None]
  assert lanes[1]["indications"] == ["left", "straight"]
  assert [x["valid"] for x in junction_lanes(["slight_left", "through", "through"], "left", fork=True)] == [True, False, False]
  assert [x["valid"] for x in junction_lanes(["", ""], "through")] == [True, True]
  assert junction_lanes(["left", "right"], None)[0]["valid"] is False


def test_polyline_round_trip():
  pts = [(0.0123456, -0.0234567), (0.0, 0.0), (-0.01, 0.02)]
  assert np.allclose(decode_polyline(encode_polyline(pts)), pts, atol=1e-6)


def test_server_answers_as_mapbox(trips):
  """The HTTP server: the path and query navd sends, with or without a token, geojson as navd asks for it."""
  name, (route, _) = next(iter(trips.items()))
  server = Server.__new__(Server)
  server.router = MagicMock()
  server.router.route.return_value = route
  server.builder = Builder(None)
  server.lock = threading.Lock()
  httpd = server.serve(0)
  t = threading.Thread(target=httpd.serve_forever, daemon=True)
  t.start()
  try:
    port = httpd.server_address[1]
    query = "access_token=x&geometries=geojson&overview=full&steps=true&banner_instructions=true&bearings=90,90;"
    url = f"http://127.0.0.1:{port}/directions/v5/mapbox/driving-traffic/0.001,-0.002;0.003,0.004?{query}"
    body = json.loads(urllib.request.urlopen(url, timeout=5).read())
    assert body["code"] == "Ok" and len(body["routes"]) == 1
    step = body["routes"][0]["legs"][0]["steps"][0]
    assert step["geometry"]["type"] == "LineString" and len(step["geometry"]["coordinates"][0]) == 2
    args = server.router.route.call_args.args
    assert args[1] == 90.0
    with pytest.raises(urllib.error.HTTPError):
      urllib.request.urlopen(f"http://127.0.0.1:{port}/directions/v5/mapbox/driving/0,0", timeout=5)  # one coordinate: 422
  finally:
    httpd.shutdown()


def test_navd_uses_the_host(trips, monkeypatch):
  """comma's navd routes through the emulator by MAPBOX_HOST alone, with no token, and steps through its response."""
  try:
    import openpilot.common.params  # noqa: F401  (native: fails where it isn't built)
  except OSError:
    stub = types.ModuleType("openpilot.common.params")
    stub.Params = MagicMock
    monkeypatch.setitem(sys.modules, "openpilot.common.params", stub)
  from openpilot.selfdrive.navigation import navd
  name, (route, _) = next(iter(trips.items()))
  server = Server.__new__(Server)
  server.router = MagicMock()
  server.router.route.return_value = route
  server.builder = Builder(None)
  server.lock = threading.Lock()
  httpd = server.serve(0)
  threading.Thread(target=httpd.serve_forever, daemon=True).start()
  try:
    monkeypatch.setenv("MAPBOX_HOST", f"http://127.0.0.1:{httpd.server_address[1]}")
    monkeypatch.delenv("MAPBOX_TOKEN", raising=False)
    params = MagicMock()
    params.get.side_effect = {"LanguageSetting": "main_en"}.get
    with patch.object(navd, "Api") as api:
      engine = navd.RouteEngine(MagicMock(), MagicMock(), params=params)
      api.assert_not_called()
    engine.last_position, engine.last_bearing = (0.001, -0.002), 90.0
    engine.calculate_route((0.003, 0.004))
    assert engine.route is not None and len(engine.route) == len(_steps(route, server))
    assert engine.route_geometry[0].shape[1] == 2
  finally:
    httpd.shutdown()


def _steps(route, server):
  return server.builder.directions(route)["routes"][0]["legs"][0]["steps"]
