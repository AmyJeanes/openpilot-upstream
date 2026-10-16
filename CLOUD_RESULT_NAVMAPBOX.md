# Cloud result: navigation on Mapbox data, a Mapbox-compatible GTA server, step 1 lane guidance

Brief: `CLOUD_TASK_NAVMAPBOX.md`. Branch `navmapbox-cloud`, one commit per item.

## Preflight

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Repo | PASS | `navmapbox-cloud` = `7e1700c` (the brief), parent `laneguide-merge` = `913fb0f7c`; cloned `--depth 1 --single-branch` |
| 2 | Data | PASS | `gta5-distill` `cloud-data` = `1c75209`; sha256 and size match `cloud/MANIFEST.md` for `live18/map1/{paths.jsonl,nodes.npz}*` and all 5 `live21b/*`. Map assembled in `/home/user/map`: live21b's `gta5.osm.pbf`, `lanes.json` (69727 lines, 1451 stop lines), `roads.json` + live18's `paths.jsonl`, `nodes.npz`; `GTA5_MAP=/home/user/map`, also linked as `~/gta5map_lanes` and `~/gta5map` |
| 3 | Dependencies | PASS (native params missing) | Python 3.12 venv (uv), PyPI only: numpy, osmium, pyvalhalla 3.9.1, shapely, scipy, pytest 9.1, pycapnp 2.1.0, pyzmq, opendbc 0.3.1 (cereal's `car.capnp`; 0.2.1 breaks log.capnp), zstandard, setproctitle, requests. openpilot's `msgq` built from its public source (`commaai/msgq`, `pip install .`, outside the repo), so `cereal.messaging` works with real sockets, no shim. Not available: native `params` (`libparams_c.so` needs capnp's generated C++ headers), so comma's `selfdrive/navigation/tests/test_navigation.py` (imports `OpenpilotPrefix` -> `Params`) and anything constructing `Params()` can't run; raylib (UI) neither |
| 4 | Valhalla | PASS | `valhalla.get_config()` -> `valhalla.json`; `valhalla_build_tiles` on live21b's pbf: 11 s, 17 MB; `valhalla_service valhalla.json 2` on :8002, `/status` = 3.9.1 |
| 5 | Mapbox | PASS | one Directions request (driving-traffic, Southampton) with `MAPBOX_TOKEN` from the environment: status 200, 1 route |
| 6 | Baseline tests | 145 pass, 1 fail | `test_nav.py`, `test_nav_msgs.py`, `test_lane_slots.py`, `test_lane_numbering.py`, `test_navd_sim.py`, `selfdrive/navd/tests`, `test_nav_approval.py`, `test_nav_request.py`: 145 passed, 1 failed in 261 s. The failure is the known `test_lane_numbering.py::test_live_freeway_merge_exit_and_splits` (also on Amy's PC). The overlay cache test isn't in this set |
| 6b | navd_sim baseline | running | `navd_sim --model n16,finetuned --lane truth,model:0.9,model:0.85,model:0.8,model:0.85:1.0:0.15:0.7` (the laneguide baseline on live21b, plus the other heads item 4 compares); numbers in item 4 |
| 7 | Push access | PASS | this file, committed and pushed to `navmapbox-cloud` |

Test command (repo root, venv active, Valhalla up):

```
export PYTHONPATH=$PWD GTA5_MAP=/home/user/map
G=openpilot/tools/sim/bridge/gta5
python -m pytest -q $G/test_nav.py $G/test_nav_msgs.py $G/test_lane_slots.py $G/test_lane_numbering.py $G/test_navd_sim.py \
  openpilot/selfdrive/navd/tests openpilot/selfdrive/controls/tests/test_nav_approval.py openpilot/selfdrive/ui/tests/test_nav_request.py
```

## Item 1: what Mapbox really returns

20 Directions requests (navd's `driving-traffic`, `steps`, `banner_instructions`, `annotations=maxspeed,congestion`,
`overview=full`, `polyline6`) over Southampton, Eastleigh, the M27/M3/M275, Portsmouth, Winchester, the New Forest and
London; 12 kept as fixtures in `openpilot/selfdrive/navigation/tests/fixtures/mapbox/` (token stripped, 680 KB), each
named with its place and what it shows; the README there has the full write-up, `fetch.py` refetches them, and
`selfdrive/navigation/tests/test_mapbox_fixtures.py` (39 tests) pins the facts below.

| feature asked for | fixture: where |
|---|---|
| right turn into a bay that opens shortly before | `soton_uni_to_general`: Winchester Rd -> Dale Rd (2 lanes 250 m out, 3 in the last 33 m) |
| a lane that has always been there | `soton_bargate_to_portswood`: Dorset St (A33) into Onslow Rd, 3-4 lanes for 260 m |
| a lane added mid-block | `soton_central_to_ocean_village`: West Quay Rd; `london_vauxhall_gyratory`: 4, 3, 4, 2, 5 lanes along one step |
| motorway exit with a lane drop | `m27_j5_west_exit_j3`, `eastleigh_m3_m27_west`: M27 J3 (3 lanes, 4 at the gore, the exit lane leftmost) |
| lane-gain on-ramp | the same two: M27 2 -> 4 lanes past J4/J5 |
| two maneuvers within 100 m | `portsmouth_m275_south` (10 m), `soton_shirley_to_west_quay` (80 m), `soton_central_to_ocean_village` (11 m) |
| roundabout | `winchester_m3_j9_city` (entries gaining a lane), `london_old_street_roundabout`, rotaries on the M27 slips |
| no lane data | `newforest_lyndhurst_brockenhurst` (6 km, none at all); most residential turns elsewhere |

What `lanes` holds:
- **Only where OSM has `turn:lanes`**, then on every intersection of that way (signals, crossings, side roads), repeated,
  up to the junction it ends at: 403 of 2257 intersections (18%); 38 of 169 maneuvers (22%) have lanes at their own
  intersection. Left to right, `indications`, `valid`, `active` (= `valid` in all 20 responses: no lane anticipation
  seen, even for the 6 close maneuver pairs with lanes), `valid_indication` on valid lanes, and sometimes `access`
  (bus and cycle lanes are listed as lanes).
- **Valid for the move at the way's own junction**, not the next maneuver: 250 m before Dale Rd the way says `S* R`
  (straight on there), only the last way before the turn `S R R*`.
- **The maneuver's intersection** has the junction's lanes, but in 5 of 24 cases a different list from 13-49 m before:
  the M27 J3 exit lane exists only at the gore node (`S* S S` -> `l* S S S`), a roundabout entry gains its third lane,
  and Dale Rd's node shows 2 lanes where the approach shows 3. **Counted from the maneuver's side the target agrees in
  all five**, which is what the guidance counts by.
- **Bays and added lanes are not flagged**: a bay is one more lane at the intersections nearest the junction (12
  maneuvers have more lanes there than 100-400 m back); where it begins isn't given, nor widths or geometry.
- **Banner lanes** (`sub.components` of type `lane`) equal the next maneuver's intersection lanes (26 of 26), from 402 m
  before it in town (or the whole step when shorter), 1609 m or 3219 m on motorways.
- **maxspeed**: per geometry segment, `{"speed": 48, "unit": "km/h"}` (UK mph as km/h: 32, 48, 64, 80, 97, 113...) or
  `{"unknown": true}` (5% of segments); congestion `low` / `moderate` / `unknown`.

## Item 2: a Mapbox-compatible Directions server for GTA

`openpilot/tools/sim/bridge/gta5/mapbox_server.py`: `Builder.directions(route)` makes the response from a
`router.Route` on our map (in process: navd_sim uses it so), and `Server` answers
`GET /directions/v5/mapbox/<profile>/<lon,lat;...>` over HTTP from a Valhalla route (`map/router.py`), in the map's
lat/lon (`map/gta5_map.py`). Run: `python -m openpilot.tools.sim.bridge.gta5.mapbox_server --port 8003`.

How it matches item 1:
- **Steps** from the route's Valhalla maneuvers, tidied as `navmodel/nav_input.py` does (`junction_maneuvers`,
  `fold_keeps`), typed with its Mapbox words (`words`); each turn's step starts at the junction with lanes within 30 m
  ahead of Valhalla's point (GTA's link into a junction starts at its stop line; Mapbox's maneuver is at the junction
  node). Distances as Mapbox measures its geometry (`navigation/route.cumulative_distance`); `polyline6`, `polyline` or
  `geojson` as asked.
- **Intersections** at every route node where three or more of the map's ways meet (bearings of all of them, `entry` by
  one-way, `in`/`out`), at the ends, and at stop lines on a route node (`traffic_signal`, `stop_sign`, `yield_sign`).
- **Lanes** only where the map has turn:lanes (`Route.info`'s laneArrows): the junction's lanes as at its stop line,
  bays included, on every intersection of the way that carries them into the junction (as Mapbox repeats a way's
  lanes), `valid`/`active`/`valid_indication` for the route's move there (`osm_lanes.turn_targets`, slight arrows for a
  fork, arrow-less lanes valid straight on). `active` = `valid`, as in every real response.
- **Banners** for each step's next maneuver, lane components from its intersection, from 402 m in town (the whole step
  if shorter) and 2 / 1 mile on motorways.
- **Annotations**: maxspeed per segment from the map's limits in mph (US-style signs; `speedLimitUnit` mph), else by
  road class; congestion `unknown`; distance, duration.
- **Nothing more**: no opening points, widths, lane geometry, lanes between junctions or truth lane (tested: lane
  objects hold only Mapbox's keys).

On the 46 navd_sim trips: 805 of 3104 intersections (26%) and 156 of 437 maneuvers (36%) have lanes, against 18% and
22% in the UK responses: our map tags turn:lanes at more junctions than UK OSM does, so GTA, if anything, still gives
nav a little more than a real car gets. 118 maneuver pairs within 100 m (30 with lanes at both), 11 forks/exits with
lanes.

**navd**: `navd.py` takes `MAPBOX_HOST` (env) or the `NavMapboxHost` param (new key, persistent string) as the
Mapbox host, with `MAPBOX_TOKEN` if set and otherwise no token (no comma API token is fetched); unchanged otherwise.
`MAPBOX_HOST=http://127.0.0.1:8003` routes navd on GTA. A live request through Valhalla checked by hand (MDT's trip,
14 steps, 4.2 km).

Tests: `tools/sim/bridge/gta5/test_mapbox_server.py` (10): schema parity with the fixtures (every key the emulator
gives is one Mapbox gives, same JSON type; every key Mapbox always gives is there, but for Mapbox-internal weights),
geometry/distances/geometry indexes, lanes only on intersections with Mapbox's keys only, a bay only as an extra
lane at its junction (more lanes there than 100 m back, nothing saying where it opens), two close maneuvers (the first's
banner shows the second's lanes), forks and exits (valid lanes side by side, on the branch's side), the lane builder,
the polyline round trip, the HTTP server as navd calls it, and comma's `RouteEngine` routing through it by
`MAPBOX_HOST` alone with no token.
