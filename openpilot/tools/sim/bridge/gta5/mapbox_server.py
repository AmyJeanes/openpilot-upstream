#!/usr/bin/env python3
"""A Mapbox-compatible Directions API for GTA, from Valhalla routes on our map: what navd asks api.mapbox.com for, with
the same schema and the same sparseness, so nav runs on GTA exactly as on a real car's Mapbox data.

  python -m openpilot.tools.sim.bridge.gta5.mapbox_server [--port 8003] [--router http://127.0.0.1:8002]
  MAPBOX_HOST=http://127.0.0.1:8003 ... navd                      # navd asks it instead of Mapbox

GET /directions/v5/mapbox/<profile>/<lon,lat;lon,lat...> answers as Mapbox does (any profile; `bearings`,
`geometries` polyline6 / polyline / geojson, `steps`, `banner_instructions`, `annotations`, `overview`; any
access_token is ignored), in the map's lat/lon (map/gta5_map.py: GTA metres <-> lat/lon).

What the response holds, as in the real responses (selfdrive/navigation/tests/fixtures/mapbox/README.md):
- one route, one leg per pair of waypoints given; steps from the route's Valhalla maneuvers (junction_maneuvers and
  fold_keeps, as navmodel/nav_input.py tidies them), their type and modifier as Mapbox words (nav_input.words);
- intersections at every node of the route where ways meet, and at the map's stop lines (traffic_signal, stop_sign,
  yield_sign), each with bearings, entry, in / out and geometry_index;
- `lanes` where the map has turn:lanes (Route.info's laneArrows): the lanes into the junction as they are at its stop
  line (bays included), left to right, on every intersection of the way that carries them up to that junction, as
  Mapbox repeats a way's lanes; valid / active / valid_indication for the route's move at that junction
  (osm_lanes.turn_targets). A junction with no turn:lanes has none, as on Mapbox;
- bannerInstructions for each step's next maneuver, with lane components (sub) from that maneuver's intersection, from
  402 m before it in town and 1 or 2 miles on motorways (what the real ones do);
- annotation per geometry segment: maxspeed from the map's limits (else by road class), congestion "unknown",
  distance and duration.

Nothing Mapbox lacks: no lane openings, widths or geometry, no lanes between junctions, no truth lane.
"""
import argparse
import json
import math
import os
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

from openpilot.selfdrive.navd.maneuvers import junction_maneuvers
from openpilot.selfdrive.navigation.route import cumulative_distance
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game, to_lat_lon
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import route_nodes, turn_targets, ways_from_nodes
from openpilot.tools.sim.bridge.gta5.navmodel.nav_input import fold_keeps, street, words

PORT = 8003
BANNER_TOWN = 402.336  # m (a quarter mile): banner lanes this far before a maneuver in town
BANNER_MOTORWAY = (3218.69, 1609.34)  # m: 2 miles, else 1 mile, on motorways (shorter steps: the whole step)
FREEWAY = frozenset({"motorway", "motorway_link", "trunk", "trunk_link"})
MPH = 0.44704  # m/s
# by road class where the map has no limit: mph, as US-style signs give them
CLASS_MPH = {"motorway": 65, "motorway_link": 35, "trunk": 55, "trunk_link": 35, "primary": 40, "primary_link": 30,
             "secondary": 35, "secondary_link": 30, "tertiary": 30, "tertiary_link": 25, "residential": 25,
             "unclassified": 25, "living_street": 15, "service": 15, "": 30}
# OSM highway -> Mapbox streets v8 class
STREET_CLASS = {"motorway": "motorway", "motorway_link": "motorway_link", "trunk": "trunk", "trunk_link": "trunk_link",
                "primary": "primary", "primary_link": "primary_link", "secondary": "secondary",
                "secondary_link": "secondary_link", "tertiary": "tertiary", "tertiary_link": "tertiary_link",
                "residential": "street", "unclassified": "street", "living_street": "street", "service": "service"}
# OSM turn:lanes values -> Mapbox lane indications, in Mapbox's order
INDICATION = {"reverse": "uturn", "sharp_left": "sharp left", "left": "left", "slight_left": "slight left",
              "merge_to_right": "slight right", "through": "straight", "slight_right": "slight right", "right": "right",
              "sharp_right": "sharp right", "merge_to_left": "slight left", "none": "none", "": "none"}
ORDER = ["uturn", "sharp left", "left", "slight left", "straight", "slight right", "right", "sharp right", "none"]
SIGNALS = {"lights": "traffic_signal", "stop": "stop_sign", "give_way": "yield_sign"}
STOP_AT = 10.0  # m: a stop line this near a route point is at it (its signal node), else not given
LANE_JUNCTION = 1.0  # m between where laneArrows end and the junction's route point
SNAP_BEHIND, SNAP_AHEAD = 5.0, 30.0  # m: a maneuver is at the junction with lanes this near it


# *** polylines ***

def encode_polyline(lat_lon, precision: int = 6) -> str:
  """Google's encoded polyline of (lat, lon) pairs."""
  out, last = [], (0, 0)
  for lat, lon in lat_lon:
    cur = (int(round(lat * 10 ** precision)), int(round(lon * 10 ** precision)))
    for v in (cur[0] - last[0], cur[1] - last[1]):
      v = ~(v << 1) if v < 0 else v << 1
      while v >= 0x20:
        out.append(chr((0x20 | (v & 0x1f)) + 63))
        v >>= 5
      out.append(chr(v + 63))
    last = cur
  return "".join(out)


def geometry(lon_lat: np.ndarray, kind: str):
  if kind == "geojson":
    return {"type": "LineString", "coordinates": [[round(float(lo), 6), round(float(la), 6)] for lo, la in lon_lat]}
  return encode_polyline([(la, lo) for lo, la in lon_lat], 5 if kind == "polyline" else 6)


def bearing(d) -> int:
  """Mapbox's bearing (deg clockwise from north) of a game-metres direction."""
  return int(round(math.degrees(math.atan2(float(d[0]), float(d[1]))))) % 360


# *** lanes ***

def indications(arrows: str) -> list[str]:
  got = {INDICATION.get(a, "none") for a in arrows.split(";")} if arrows else {"none"}
  return sorted(got, key=ORDER.index)


MOVE_INDICATIONS = {"left": ("left", "slight left", "sharp left", "uturn"), "right": ("right", "slight right", "sharp right"),
                    "through": ("straight", "none", "slight left", "slight right")}


def junction_lanes(arrows: list[str], move: str | None, fork: bool = False) -> list[dict]:
  """A junction's approach lanes, left to right, as Mapbox gives them: indications, and for the route's move through
  it ('left', 'through', 'right'; None: unknown, no lane valid), valid / active / valid_indication."""
  turns = [frozenset(a.split(";")) - {""} if a else frozenset() for a in arrows]
  ok: list[int] = []
  if move is not None:
    ok = turn_targets(turns, move, fork) if fork else []
    if not ok:
      ok = turn_targets(turns, move)
    if not ok and move == "through":  # lanes without arrows go straight on (OSRM's "none")
      ok = [i for i, t in enumerate(turns) if not t or "through" in t]
  out = []
  for i, a in enumerate(arrows):
    ind = indications(a)
    lane = {"indications": ind, "valid": i in ok, "active": i in ok}
    if i in ok:
      lane["valid_indication"] = next((x for x in MOVE_INDICATIONS.get(move, ()) if x in ind), ind[0])
    out.append(lane)
  return out


def banner_lanes(lanes: list[dict]) -> list[dict]:
  out = []
  for lane in lanes:
    c = {"text": "", "type": "lane", "active": bool(lane["active"]), "directions": list(lane["indications"])}
    if lane["active"]:
      c["active_direction"] = lane.get("valid_indication", lane["indications"][0])
    out.append(c)
  return out


# *** the response ***

class Builder:
  """A Mapbox Directions response for a router.Route on our map. `osm` (OsmLanes) gives the junctions' other roads;
  without it the intersections hold the route's own two."""
  def __init__(self, osm=None):
    self.osm = osm

  def _nodes(self, route):
    if self.osm is None:
      return [None] * len(route.points), []
    nodes = route_nodes(route.points, self.osm)
    return nodes, ways_from_nodes(route.points, self.osm)

  def _ways_at(self, node) -> list[tuple[np.ndarray, bool]]:
    """The roads at a map node: (direction, drivable away from it) of each link."""
    from openpilot.tools.sim.bridge.gta5.map.osm_lanes import oneway_of
    out = []
    for m in dict.fromkeys(self.osm.links.get(node, ())):
      w, fwd = self.osm.pairs[(node, m)]
      one = oneway_of(self.osm.ways[w][0])
      out.append((self.osm.node_xy(m) - self.osm.node_xy(node), one == 0 or (one == 1) == fwd))
    return out

  def directions(self, route, waypoints: list | None = None, geometries: str = "polyline6", steps: bool = True,
                 banners: bool = True, overview: str = "full", annotations: tuple = ("maxspeed", "congestion")) -> dict:
    pts = np.asarray(route.points, float)[:, :2]
    n = len(pts)
    lat, lon = to_lat_lon(pts[:, 0], pts[:, 1])
    lon_lat = np.column_stack((np.asarray(lon, float), np.asarray(lat, float)))
    along = cumulative_distance(lon_lat)  # as Mapbox measures its geometry
    info = route.at, route.seg
    route.at, route.seg = 0.0, 0
    try:
      ri = route.info(1e9)
    finally:
      route.at, route.seg = info
    nodes, ways = self._nodes(route)
    classes = list(getattr(route, "classes", [""] * (n - 1)))
    limits = np.asarray(route.limits, float) if len(route.limits) == n - 1 else np.zeros(n - 1)
    mph = [round(v / MPH) if v > 0 else CLASS_MPH.get(c, CLASS_MPH[""]) for v, c in zip(limits, classes, strict=True)]
    seg_len = np.diff(along)
    seg_t = seg_len / np.maximum(np.array(mph, float) * MPH, 1.0)

    def idx_at(s_game: float) -> int:
      return int(np.argmin(np.abs(route.along - s_game)))

    # intersections: route points where ways meet, and the stop lines
    ints: dict[float, dict] = {}  # by geometry position (index + fraction), to keep them in order

    def heading_at(k: int, back: bool) -> np.ndarray:
      if back:
        return pts[max(k - 1, 0)] - pts[k] if k > 0 else -(pts[min(k + 1, n - 1)] - pts[k])
      return pts[min(k + 1, n - 1)] - pts[k] if k < n - 1 else -(pts[k - 1] - pts[k])

    for k in range(n):
      node = nodes[k]
      roads = self._ways_at(node) if node is not None else []
      if k not in (0, n - 1) and len(roads) < 3:
        continue
      ints[float(k)] = self._intersection(k, roads, pts, heading_at, k == 0, k == n - 1)
    for s, kind in zip(route.stops, route.stop_kinds, strict=True):
      k = idx_at(s)  # on the route's nearest point, as Mapbox's intersections are all on its geometry's
      if abs(route.along[k] - s) > STOP_AT or not SIGNALS.get(kind):
        continue
      it = ints.get(float(k)) or self._intersection(k, [], pts, heading_at, k == 0, k == n - 1)
      it[SIGNALS[kind]] = True
      ints[float(k)] = it

    # lanes: each junction's from its turn:lanes, on the intersections of the way that carries them there
    forks = {round(f.along, 1): f for f in route.forks}
    turns = ri["turns"]
    for entry in ri["laneArrows"]:
      e, arrows = float(entry[0]), entry[1]
      move = entry[2] if len(entry) > 2 else None
      kj = idx_at(e)
      if abs(route.along[kj] - e) > LANE_JUNCTION + 5.0:
        continue
      fork = next((f for a, f in forks.items() if abs(a - e) < 5.0), None)
      if move is None:  # as the route turns there, else straight on
        t = next((t for t in turns if abs(t[0] - e) < 25.0), None)
        move = t[1] if t is not None else (fork.side if fork is not None else "through")
      lanes = junction_lanes(arrows, move, fork is not None and move != "through")
      i0 = next((i0 for _, _, i0, i1 in ways if i0 < kj <= i1), max(kj - 1, 0))
      for pos, it in ints.items():
        if i0 <= pos <= kj and not (pos == 0.0 and kj > 0):
          it["lanes"] = [dict(x) for x in lanes]
      if float(kj) not in ints:  # the junction itself, though the map's nodes don't meet there
        ints[float(kj)] = {**self._intersection(kj, [], pts, heading_at, False, False), "lanes": lanes}

    # steps, one per maneuver
    mans = fold_keeps(junction_maneuvers(route), lambda m: float(route.along[min(m.get("begin_shape_index", 0), n - 1)]))
    lane_ends = sorted(idx_at(float(entry[0])) for entry in ri["laneArrows"])
    starts, kept = [], []
    for m in mans:
      if words(m.get("type", 0)) is None:
        continue
      b = min(int(m.get("begin_shape_index", 0)), n - 1) if starts else 0  # the first step from the route's start
      if m.get("type") not in (1, 2, 3, 4, 5, 6):
        # at the junction whose lanes it's taken from (Mapbox's maneuver is at the junction's node; GTA's link into
        # it starts at the stop line)
        b = next((k for k in lane_ends if -SNAP_BEHIND <= route.along[k] - route.along[b] <= SNAP_AHEAD), b)
      if starts and b <= starts[-1] and m.get("type") not in (4, 5, 6):
        continue
      if starts and b < starts[-1]:
        b = starts[-1]
      starts.append(b)
      kept.append(m)
    if not kept or kept[0].get("type") not in (1, 2, 3):
      kept.insert(0, {"type": 1, "begin_shape_index": 0})
      starts.insert(0, 0)
    if kept[-1].get("type") not in (4, 5, 6):
      kept.append({"type": 4, "begin_shape_index": n - 1})
      starts.append(n - 1)
    order = sorted(ints)
    step_list = []
    for j, (m, b) in enumerate(zip(kept, starts, strict=True)):
      e = starts[j + 1] if j + 1 < len(starts) else n - 1
      mt, mod = words(m.get("type", 0))
      d = float(along[e] - along[b]) if e > b else 0.0
      t = float(seg_t[b:e].sum())
      step_ints = []
      if float(b) not in ints:
        ints[float(b)] = self._intersection(b, [], pts, heading_at, b == 0, b == n - 1)
        order = sorted(ints)
      for pos in order:
        if b <= pos < e or (pos == b == e):
          it = {k: v for k, v in ints[pos].items() if not k.startswith("_")}
          it["geometry_index"] = int(pos)
          step_ints.append(it)
      h_in, h_out = heading_at(b, True), heading_at(b, False)
      name = street(m)
      step = {
        "geometry": geometry(lon_lat[b:e + 1] if e > b else lon_lat[[b, b]], geometries),
        "maneuver": {"location": [round(float(lon_lat[b, 0]), 6), round(float(lon_lat[b, 1]), 6)],
                     "bearing_before": bearing(-h_in) if b > 0 else 0, "bearing_after": bearing(h_out) if e > b else 0,
                     "type": mt, **({"modifier": mod} if mt not in ("depart", "arrive") or mod != "straight" else {}),
                     "instruction": m.get("instruction", "")},
        "name": name, "mode": "driving", "driving_side": "right", "weight": round(t, 3), "duration": round(t, 3),
        "duration_typical": round(t, 3), "weight_typical": round(t, 3), "distance": round(d, 3),
        "speedLimitSign": "mutcd", "speedLimitUnit": "mph", "intersections": step_ints, "bannerInstructions": [],
      }
      step_list.append(step)
    if banners:
      for j in range(len(step_list) - 1):
        self._banners(step_list[j], step_list[j + 1], classes[min(starts[j + 1], n - 2)] if n > 1 else "")

    legs_out = [{"steps": step_list if steps else [], "summary": ", ".join(dict.fromkeys(s["name"] for s in step_list if s["name"]))[:80],
                 "distance": round(float(along[-1]), 3), "duration": round(float(seg_t.sum()), 3),
                 "weight": round(float(seg_t.sum()), 3), "duration_typical": round(float(seg_t.sum()), 3),
                 "weight_typical": round(float(seg_t.sum()), 3), "admins": [{"iso_3166_1_alpha3": "USA", "iso_3166_1": "US"}],
                 "via_waypoints": []}]
    ann = {}
    if "maxspeed" in annotations:
      ann["maxspeed"] = [{"speed": int(v), "unit": "mph"} for v in mph]
    if "congestion" in annotations:
      ann["congestion"] = ["unknown"] * (n - 1)
    if "distance" in annotations:
      ann["distance"] = [round(float(x), 1) for x in seg_len]
    if "duration" in annotations:
      ann["duration"] = [round(float(x), 3) for x in seg_t]
    if ann:
      legs_out[0]["annotation"] = ann
    r = {"weight_name": "auto", "weight": legs_out[0]["weight"], "duration": legs_out[0]["duration"],
         "duration_typical": legs_out[0]["duration"], "weight_typical": legs_out[0]["weight"],
         "distance": legs_out[0]["distance"], "legs": legs_out}
    if overview != "false":
      r["geometry"] = geometry(lon_lat, geometries)
    wps = waypoints or [pts[0], pts[-1]]
    out_wps = []
    for p in (wps[0], wps[-1]):
      la, lo = to_lat_lon(float(p[0]), float(p[1]))
      out_wps.append({"distance": 0.0, "name": "", "location": [round(float(lo), 6), round(float(la), 6)],
                      "time_zone": {"offset": "-07:00", "identifier": "America/Los_Angeles", "abbreviation": "PDT"}})
    return {"routes": [r], "waypoints": out_wps, "code": "Ok", "uuid": "gta5-" + format(abs(hash(pts.tobytes())) % 16 ** 12, "012x")}

  def _intersection(self, k, roads, pts, heading_at, first, last) -> dict:
    la, lo = to_lat_lon(float(pts[k][0]), float(pts[k][1]))
    back, fwd = heading_at(k, True), heading_at(k, False)
    bs = {bearing(back): False, bearing(fwd): True}
    for d, drivable in roads:
      b = bearing(d)
      if b not in bs and min(abs((b - x + 180) % 360 - 180) for x in bs) > 5:
        bs[b] = drivable
    if first:
      bs = {bearing(fwd): True}
    elif last:
      bs = {bearing(back): False}
    keys = sorted(bs)
    it = {"location": [round(float(lo), 6), round(float(la), 6)], "bearings": keys, "entry": [bs[b] for b in keys],
          "admin_index": 0, "is_urban": True, "mapbox_streets_v8": {"class": "street"}}
    if not first:
      it["in"] = keys.index(bearing(back))
    if not last:
      it["out"] = keys.index(bearing(fwd))
    return it

  @staticmethod
  def _banners(step: dict, nxt: dict, cls: str):
    """bannerInstructions on step for the maneuver that ends it (nxt's), lanes from nxt's own intersection."""
    mv = nxt["maneuver"]
    text = nxt["name"] or ("Your destination" if mv["type"] == "arrive" else "")
    primary = {"text": text, "type": mv["type"], "components": [{"text": text, "type": "text"}]}
    if "modifier" in mv:
      primary["modifier"] = mv["modifier"]
    d = step["distance"]
    lanes = next((it.get("lanes") for it in nxt["intersections"][:1] if it.get("lanes")), None)
    out = [{"distanceAlongGeometry": d, "primary": primary}]
    if lanes and mv["type"] != "arrive":
      at = next((x for x in BANNER_MOTORWAY if d > x), d) if cls in FREEWAY else min(d, BANNER_TOWN)
      sub = {"text": "", "components": banner_lanes(lanes)}
      if at >= d:
        out = [{"distanceAlongGeometry": d, "primary": primary, "sub": sub}]
      else:
        out.append({"distanceAlongGeometry": round(at, 3), "primary": primary, "sub": sub})
    step["bannerInstructions"] = out


# *** the server ***

def parse_coords(text: str) -> list[tuple[float, float]]:
  return [tuple(float(v) for v in c.split(",")[:2]) for c in text.split(";") if c]


class Server:
  """Routes on our map with Valhalla (map/router.py), answered as Mapbox Directions."""
  def __init__(self, router_url: str = "http://127.0.0.1:8002", map_data=None):
    from openpilot.tools.sim.bridge.gta5.map.router import Router
    from openpilot.tools.sim.bridge.gta5.mapdrive_sim import live_map
    m = map_data or live_map()
    if m is None:
      raise RuntimeError("no live map (GTA5_MAP)")
    self.router = Router(router_url, timeout=10.0, paths=m["paths"], osm=m["osm"], roads=m["osm"])
    self.builder = Builder(m["osm"])
    self.lock = threading.Lock()

  def answer(self, path: str, query: dict) -> tuple[int, dict]:
    parts = path.strip("/").split("/")
    if len(parts) != 5 or parts[:3] != ["directions", "v5", "mapbox"]:
      return 404, {"code": "NotFound", "message": "Not Found"}
    coords = parse_coords(urllib.parse.unquote(parts[4]))
    if len(coords) < 2:
      return 422, {"code": "InvalidInput", "message": "at least two coordinates"}
    game = [np.array(to_game(lat, lon)) for lon, lat in coords]
    first = (query.get("bearings", [""])[0].split(";")[0] or "").split(",")[0]
    heading = float(first) if first else 0.0
    q = {k: v[0] for k, v in query.items()}
    with self.lock:
      try:
        route = self.router.route(game[0], heading, game[-1], through=game[1:-1])
      except Exception as e:  # Valhalla's errors: no route
        return 200, {"code": "NoRoute", "message": str(e), "routes": []}
      body = self.builder.directions(route, waypoints=[game[0], game[-1]], geometries=q.get("geometries", "polyline"),
                                     steps=q.get("steps", "false") == "true", banners=q.get("banner_instructions", "false") == "true",
                                     overview=q.get("overview", "simplified"),
                                     annotations=tuple(q.get("annotations", "").split(",")))
    return 200, body

  def serve(self, port: int = PORT, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    server = self

    class Handler(BaseHTTPRequestHandler):
      def do_GET(self):
        u = urllib.parse.urlsplit(self.path)  # not urlparse: it takes the path's ';' for params
        code, body = server.answer(u.path, urllib.parse.parse_qs(u.query))
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

      def log_message(self, fmt, *args):  # the query holds no token of ours, but keep the log quiet
        if os.getenv("GTA5_MAPBOX_LOG"):
          print("mapbox_server:", self.path.split("?")[0], *args[1:])

    httpd = ThreadingHTTPServer((host, port), Handler)
    return httpd


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--port", type=int, default=PORT)
  ap.add_argument("--host", default="127.0.0.1")
  ap.add_argument("--router", default=os.getenv("GTA5_ROUTER", "http://127.0.0.1:8002"))
  args = ap.parse_args()
  httpd = Server(args.router).serve(args.port, args.host)
  print(f"mapbox_server: http://{args.host}:{args.port}/directions/v5/mapbox/driving-traffic/... (router {args.router})")
  httpd.serve_forever()


if __name__ == "__main__":
  main()
