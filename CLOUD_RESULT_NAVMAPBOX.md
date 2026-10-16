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
| 6b | navd_sim baseline | PASS | `navd_sim --model n16,finetuned --lane truth,model:0.9,model:0.85,model:0.8,model:0.85:1.0:0.15:0.7` on live21b, 46 trips, 302 maneuvers that need a lane: n16 / truth 247 (82%), n16 / model:0.85 250, finetuned / truth 262 (87%), finetuned / model:0.85 255 (laneguide's on live19b: 247 / 250 / 261 / 254); the full table in item 4 |
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
  node). Valhalla's keeps as OSRM/Mapbox has them (found in item 4): a `fork` only where the map's lanes split, a
  `merge` where the road joins one with as many lanes or more (on-ramps), no maneuver where the map has no fork; and a
  `fork` step at a fork in the road Valhalla names no maneuver for. Distances as Mapbox measures its geometry (`navigation/route.cumulative_distance`); `polyline6`, `polyline` or
  `geojson` as asked.
- **Intersections** at every route node where three or more of the map's ways meet (bearings of all of them, `entry` by
  one-way, `in`/`out`), at the ends, and at stop lines on a route node (`traffic_signal`, `stop_sign`, `yield_sign`).
- **Lanes** only where the map has turn:lanes (`Route.info`'s laneArrows): the junction's lanes as at its stop line,
  bays included, on every intersection of the way that carries them into the junction (as Mapbox repeats a way's
  lanes), `valid`/`active`/`valid_indication` for the route's move there (`osm_lanes.turn_targets`, slight arrows for a
  fork, arrow-less lanes valid straight on). `active` = `valid`, as in every real response. At a fork whose arrows
  don't single out the route's branch (all valid, or none there), the lanes from the map's lane counts (`Fork.lanes`
  of `lanes_in`, the branch's slight left/right), as a tagged diverge shows them: 16 of the 19 UK forks and exits in
  item 1 have lanes. `GTA5_MAPBOX_FORK_LANES=0` turns this off (item 4 measures both). Each intersection carries its
  road's `mapbox_streets_v8` class, and `classes: [motorway]` on motorways.
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

## Item 3: step 1 lane guidance in comma's navd

`openpilot/selfdrive/navigation/lane_guidance.py` (imports nothing from `tools/sim`), run by comma's
`selfdrive/navigation/navd.py` in its own process.

What it does, as agreed ("Step 1 lane guidance"):
- **Route** (`MapboxRoute`): from navd's Mapbox response (`routes[0]`, geojson or polyline6), the moves to plan lanes
  for: each maneuver at its step's start (turn / end of road left-right-slight, fork, off ramp = exit, roundabout /
  rotary; side from the modifier; on ramps and merges only by their lanes) with its intersection's lanes (else the
  approach's last within 60 m, never another maneuver's), and each
  junction the route goes straight on through where not all lanes are valid (the last intersection of each run of
  repeated lanes, side = the edge the valid lanes touch); cycle-only lanes dropped. `maxspeed` by distance along;
  freeway = Mapbox's `motorway` class or a 60 mph limit (UK A-road "trunk"s aren't).
- **Targets** (`targets`): the valid lanes, counted from the move's side; where the next maneuver follows within
  `pair_gap` (100 m), narrowed to those nearest its side (`narrowed`). No lanes at a turn, fork or exit: the outermost
  lane on its side (`fallback`).
- **The car's lane** (`LaneReader`): `modelV2.laneHead` taken once 0.7 sure for 1 s, kept while 0.4 sure and through
  dropouts for 3 s, moved over by nav's own completed change; counted from the move's side. Where the model counts fewer
  lanes than Mapbox lists (a bay not open), the missing ones are taken to be on the turn's side: the target is the
  outermost lane there and the car is in place in it (`bay`).
- **Timing** (`_due`): asked once `deadline / v_plan <= changes x lc_time + margin` or `deadline <= changes x
  min_dist`, deadline = m to `done_before` before the maneuver (turns 35 m, forks/exits 60 m, straight on 30 m), or to
  `signal_room` (30 m) before the last traffic signal within 150 m on the way to it (Mapbox's `traffic_signal`: a queue
  there, where n16 can't change lanes), `v_plan = max(v, maxspeed_share x maxspeed)`; `lc_time_fwy` from 22 m/s. The
  rest of a maneuver's changes follow `change_gap` after each. A change not started below `lc_min_speed` (the model's:
  n16 10 m/s) is withdrawn (`gave_up`); none is asked below it. Changes for a later move never take the car away from
  an earlier move's lanes, and one an earlier move wants the same way is asked for as the earlier's.
- **Turns**: the turn's signal (request kind `turn`) once below the lane turn speed within `signal_time` x v (28-50 m), or
  at `signal_min` whatever the speed, or within `turn_into` (80 m) when slower than `lc_min_speed` and either one lane
  beside its lanes on its side or in place before its bay: the turn desire takes the car over (laneguide's
  `turn_into_lane` / `BAY_SIGNAL`). A change towards it gives way to it by `signal_last`; the blinker goes off and on
  again so the model sees the turn's pulse.
- **Request and approve**: `navInstruction.request` (kind, side, reason turn/fork/exit/straightOn, distance, laneChanges,
  stage, id, approved = the blinker on its side) and `NavDesire` `laneChange:<side>` at stages 1-2, `laneChange` at 3,
  held 0.5 s after; the stalk actions in `requests` (the blinker 0.4 s after NavDesire, so openpilot has read it; a
  blinker still on the other side goes off first, as openpilot starts a change on the blinker's edge; off again once
  the change is over, done or not).
  Outcomes logged as the planner's (`navRequest`: done, declined, timeout, gave_up, turn, replaced).
- **Lane UI**: `navInstruction.lanes` (directions, active, activeDirection, current where the model counts as many) for
  the next move with lanes, only within its banner's lane distance (402 m town, 1-2 miles motorway) or with a request
  open; `laneDistance` = m to the deadline while out of the lanes.

**navd** (`navd.py`): with the `NavLaneGuidance` param (new, persistent bool) or `NAVD_LANE_GUIDANCE=1`, navd also
subscribes `carState` and `modelV2`, runs at 10 Hz (route requests, rerouting and navRoute still at 1 Hz), asks Mapbox
for `annotations=maxspeed,congestion`, steps the guidance from its progress along the route, fills `request`, `lanes`
and `laneDistance` on `navInstruction` and writes `NavDesire`. `NAVD_LANE_TUNE` takes Tune overrides (JSON). Off (the
default), navd is exactly as before.

**GTA**: the existing bridge-driven planner path is unchanged and stays the default (`GTA5_NAV_PATH=bridge`). With
`GTA5_NAV_PATH=navd` the bridge leaves nav to comma's navd: its planner no longer sets NavDesire or drives the stalk,
its own navInstruction/navRoute (`GTA5_NAV_MSGS`) are off, and its simulated driver works the stalk from navd's
`navInstruction.request` (`gta5_navd.NavdRequests`: a new lane change or turn puts the indicator on its side, its end
takes it off). The bridge still writes `NavDestination` and the simulated GNSS navd localizes from.

Also: `turn_targets` / `LEFTS` / `RIGHTS` moved to `selfdrive/navd/lane_targets.py` (re-exported by `osm_lanes`), so
`planner.py` no longer imports `tools/sim`.

Tests: `selfdrive/navigation/tests/test_lane_guidance.py` (16): timing by speed and lanes away, slowing traffic not
putting the ask off, a freeway exit three lanes over (lc_time_fwy), counting from the turn's side with a bay not open
(in place, then the turn's signal takes it over; one lane out, one change), two close maneuvers narrowed together, the
no-lanes fallback (and off), withdrawal when too slow, the request and NavDesire by stage (approval, the blinker after
NavDesire, done, the next change after change_gap), the lanes only under guidance, the lane reader, the real Dale Rd
bay and M27 J3 exit counted right, comma's `RouteEngine` running it end to end (request, lanes, laneDistance on
navInstruction, NavDesire written), and (from item 4) lights bringing the deadline forward, the blinker off before
the other side's, and approach lanes never another maneuver's. `tools/sim/bridge/gta5/test_navd_path.py` (1): the bridge's driver from navd's
requests.

## Item 4: measure

`navd_sim --nav mapbox`: nav is comma navd's lane guidance (item 3) fed only the trip's emulated Mapbox response
(`mapbox_server.Builder`, in process: what the GTA server answers), the car's progress along it, its speed, blinker and
blind spots, openpilot's lane change state and the lane head stand-in (`truth` = a perfect lane head: the road's lane
and count at the car). Nothing of the full map reaches nav. The judge is unchanged: laneguide's, on the full map's lanes
at each commit point. Same 46 trips (163 km), same traffic and light seeds, the lane head on its own noise stream, so
the two navs differ only by what nav knows and does. `lc_min_speed` is the stand-in's `v_ok`, as the planner gets
`model_lc_speed`.

Reproduce (venv, `GTA5_MAP` set, no router needed, about 70 min on 4 cores for both navs and all heads):

```
python -m openpilot.tools.sim.bridge.gta5.navd_sim --nav map,mapbox --model n16,finetuned \
  --lane truth,model:0.9,model:0.85,model:0.8,model:0.85:1.0:0.15:0.7 [--sim-seed 1] [--tune '{"lc_time": 6}'] --out r.json
```

(`GTA5_MAPBOX_FORK_LANES=0` for the emulator without fork lanes; `--tune` keys go to whichever nav runs: the
guidance's `Tune` in `lane_guidance.py`.)

### Mapbox data alone against laneguide's full-map baseline (seed 0, final code)

In a target lane at the commit point, of 302 maneuvers that need one (107 turns, 108 forks/exits, 87 straight on):

| model / lane head | in lane: full map -> Mapbox | turns | forks / exits | straight on | LC req / done / failed (map -> Mapbox) | LC asked < v_ok | hanging s | forced merges | misses: late lane / no Mapbox lanes / other (Mapbox) |
|---|---|---|---|---|---|---|---|---|---|
| n16 / truth | 247 -> **247** (+0) | 68 -> 77 | 95 -> 91 | 84 -> 79 | 181/120/53 -> 159/106/46 | 85 -> 0 | 27 -> 50 | 7 -> 20 | 28 / 4 / 23 (map: 37 / - / 18) |
| n16 / model:0.9 | 250 -> **244** (-6) | 71 -> 74 | 97 -> 91 | 82 -> 79 | 187/122/58 -> 164/103/51 | 77 -> 0 | 39 -> 61 | 6 -> 24 | 29 / 4 / 25 (map: 34 / - / 18) |
| n16 / model:0.85 | 250 -> **244** (-6) | 72 -> 74 | 97 -> 91 | 81 -> 79 | 181/124/52 -> 162/102/51 | 78 -> 0 | 35 -> 58 | 6 -> 22 | 29 / 5 / 24 (map: 34 / - / 18) |
| n16 / model:0.8 | 248 -> **244** (-4) | 71 -> 74 | 95 -> 91 | 82 -> 79 | 180/119/55 -> 163/104/50 | 75 -> 0 | 47 -> 56 | 7 -> 21 | 29 / 5 / 24 (map: 35 / - / 19) |
| n16 / hard | 242 -> **245** (+3) | 67 -> 76 | 95 -> 90 | 80 -> 79 | 210/117/88 -> 190/109/73 | 81 -> 0 | 37 -> 60 | 8 -> 22 | 28 / 4 / 25 (map: 36 / - / 24) |
| finetuned / truth | 262 -> **254** (-8) | 80 -> 76 | 99 -> 98 | 83 -> 80 | 219/148/63 -> 232/154/70 | 8 -> 0 | 53 -> 75 | 4 -> 22 | 28 / 2 / 18 (map: 27 / - / 13) |
| finetuned / model:0.9 | 257 -> **250** (-7) | 78 -> 72 | 99 -> 97 | 80 -> 81 | 218/146/67 -> 235/144/80 | 5 -> 0 | 79 -> 106 | 8 -> 23 | 29 / 2 / 21 (map: 26 / - / 19) |
| finetuned / model:0.85 | 255 -> **252** (-3) | 74 -> 75 | 99 -> 96 | 82 -> 81 | 209/139/66 -> 238/146/82 | 5 -> 0 | 51 -> 96 | 8 -> 22 | 28 / 2 / 20 (map: 28 / - / 19) |
| finetuned / model:0.8 | 256 -> **253** (-3) | 76 -> 74 | 99 -> 97 | 81 -> 82 | 211/138/67 -> 242/151/82 | 7 -> 0 | 79 -> 90 | 7 -> 22 | 29 / 2 / 18 (map: 27 / - / 19) |
| finetuned / hard | 244 -> **248** (+4) | 69 -> 71 | 96 -> 95 | 79 -> 82 | 249/132/109 -> 268/146/113 | 11 -> 0 | 68 -> 124 | 8 -> 21 | 30 / 3 / 21 (map: 32 / - / 26) |

("hard" = `model:0.85:1.0:0.15:0.7`: 1 s lag, 15% dropouts, counts right 70%. "late lane": the target lane begins 80 m
or less before the commit point, a bay or a lane added; "no Mapbox lanes": none at that maneuver in the emulated data.)

Seed 1 (traffic, lights and head noise redrawn), the same comparison:

| model / lane head | full map | Mapbox | turns | forks / exits | straight on |
|---|---|---|---|---|---|
| n16 / truth | 245 | **247** (+2) | 71 -> 80 | 91 -> 88 | 83 -> 79 |
| n16 / model:0.85 | 242 | **244** (+2) | 69 -> 77 | 91 -> 88 | 82 -> 79 |
| finetuned / truth | 255 | **256** (+1) | 76 -> 83 | 97 -> 94 | 82 -> 79 |
| finetuned / model:0.85 | 254 | **254** (+0) | 76 -> 79 | 97 -> 95 | 81 -> 80 |

Over both seeds the Mapbox path is level with the full-map planner within the noise (about ±3 maneuvers between seeds):
n16 +1 (truth) and -2 (0.85), finetuned -3.5 (truth) and -1.5 (0.85), and ahead with the hard head.

Reading it:
- **Turns gain** (+3 to +9 for n16): with the truth head, 14 of n16's turns with a late target lane (a bay) are made
  on Mapbox data and not on the full map, by the bay rule (the model counting fewer lanes than Mapbox lists, the car is in place in the
  outermost lane) and the turn's signal taking it in, against 4 the other way. The full-map planner times changes into
  the bay as it opens, which n16 can't make at the speed it's doing by then.
- **Forks / exits lose** 3 to 6, **straight on** 2 to 5: where the map's lanes say more than Mapbox's (a fork's lanes
  splitting a little before the gore, two arrows at a junction the route goes straight through, a junction whose
  approach lanes Mapbox shows only at its last way).
- **Lanes ending** are Mapbox's real blind spot: forced merges (the car still in a lane when it ends) 20-24 against
  6-8: no lane-end or lane-drop information, so nav never moves the car out early; the model merges at the end. They
  aren't in the in-lane count, but in game they are late merges into traffic.
- **The lane head's noise matters less** on Mapbox data (n16 244-247 across all heads, finetuned 248-254): targets
  are counted from the maneuver's side, so a miscount of the far lanes doesn't move them, and n16 is asked for fewer
  changes (159-190 against 180-210; the finetune, with its lower lane change speed, for more). With the hard head it
  does better than the full map.
- **No change asked below the model's lane change speed** (0 against 75-88 for n16): withdrawn instead and left to the
  turn desire. Changes hang longer (blinker on, not started: 50-60 s against 27-47 for n16 over the 46 trips; not
  broken down by cause here: the blind spot, or a gap the stand-in waits for).
- **What the misses come from** (n16 / model:0.85, Mapbox): 29 a late lane (Mapbox gives no opening point; the full
  map misses 34 of these), 5 no lanes in Mapbox's data at that maneuver (the fallback lane was wrong), 24 other
  (changes that failed: started too slow or blocked, the model's count off, map lanes Mapbox doesn't show).

The emulator's fork lanes (lanes at a fork from its lane counts where turn:lanes don't single out the branch, as 16 of
19 UK forks and exits have) are worth about 22 forks: without them (`GTA5_MAPBOX_FORK_LANES=0`), n16 / model:0.85
makes 222 (forks/exits 70) and finetuned 229 (73), against 244 (91) and 252 (96). Where a real diverge has no
turn:lanes, nav falls back to the outermost lane on the branch's side, which is wrong at a lane split.

### Fixes found while measuring (v1 -> final, n16 / model:0.85: 230 -> 244)

- Guidance: an exit with no lanes of its own took the lanes of the exit 51 m before it as its approach's (sent the
  car three lanes the wrong way); a new request switching the blinker straight from one side to the other gave
  openpilot no edge to start on (hung 10 s); a change openpilot gave up on left its blinker on; changes ran into the
  queue at lights (now done `signal_room` before the last signal); on ramps and merges were planned as forks.
- Emulator: Valhalla's keeps where an on-ramp joins (1 lane into 4) or where the road just carries on were Mapbox
  "forks", and their fallback pulled the car to the outermost lane; now merges / no step, as OSRM gives them.

### Tuning (one at a time, n16 and finetuned, model:0.85, seed 0; defaults in the first row)

| setting | n16: in lane (turns / forks / straight) LC req / done / failed | finetuned |
|---|---|---|
| defaults: lc_time 6, margin 4, maxspeed_share 0.8, min_dist 60, signal_room 30, lc_time_fwy 8 | 244 (74/91/79) 162/102/51 | 252 (75/96/81) 238/146/82 |
| lc_time 4 | 245 (74/92/79) 158/102/48 | 251 (74/97/80) 234/142/83 |
| lc_time 8 | 245 (74/92/79) 168/104/53 | 251 (75/97/79) 239/151/80 |
| margin 2 | 244 (73/92/79) 156/102/46 | 253 (75/97/81) 234/140/84 |
| margin 7 | 246 (74/93/79) 167/102/53 | 250 (74/96/80) 242/145/87 |
| maxspeed_share 0.6 | 240 (72/89/79) 161/98/54 | 249 (73/95/81) 236/134/90 |
| maxspeed_share 1.0 | 245 (74/92/79) 169/108/49 | 251 (75/96/80) 245/153/82 |
| min_dist 30 / 100 | 244 / 244 (no change) | 252 / 252 |
| signal_room 0 | 242 (74/89/79) 161/99/54 | 250 (74/95/81) 238/144/88 |
| signal_room 50 | 244 (74/91/79) 164/102/52 | 251 (75/96/80) 240/146/85 |
| lc_time_fwy 6 | 242 (73/90/79) 160/99/51 | 253 (75/96/82) 235/144/82 |
| lc_time_fwy 11 | 245 (74/92/79) 161/102/49 | 252 (75/96/81) 241/148/83 |

What changed and why: nothing beyond the noise (±2) except a low maxspeed share (0.6: -4 / -3, the ask comes too late
once traffic slows the car, which is what the share is for), dropping the signal room (-2 / -2: changes asked into the
queue) and a short freeway lc_time (-2 for n16). `min_dist` never binds here (the planning speed is at least 0.8 x
maxspeed, so time decides first; it's for roads with no maxspeed). So the defaults stay: the timing isn't what limits
the Mapbox path. What does: lanes ending unannounced, forks where Mapbox's lanes are thin, and n16's own lane change
at low speed, which the finetune's numbers show (+8 over n16).

## Item 5: hand-over

### In short

- **What Mapbox's lanes are** (item 1, 20 real UK responses, 12 kept as fixtures): lanes only where OSM has turn:lanes,
  repeated on every intersection of that way and valid for the move at that way's junction; a bay is just one more
  lane at the intersections nearest the junction, never flagged or located; the maneuver's own intersection sometimes
  lists other lanes than its approach, but counted from the maneuver's side the target agrees; banner lanes = the
  maneuver's intersection lanes from 402 m (town) or 1-2 miles (motorways); maxspeed per segment; 84% of forks/exits
  and 22% of maneuvers overall have lanes.
- **The GTA server matches it** (item 2): `mapbox_server.py` answers navd's Directions request from Valhalla on our
  map with the same schema (tested key by key against the fixtures) and the same sparseness: lanes only where our map
  has turn:lanes (plus diverges' lanes from their counts, as tagged UK diverges have), no opening points, widths,
  lane geometry or truth. Our map tags more junctions than UK OSM (36% of maneuvers with lanes against 22%).
- **Step 1 guidance in comma's navd** (item 3): `selfdrive/navigation/lane_guidance.py`, behind `NavLaneGuidance`.
- **Measured** (item 4): on Mapbox data and the lane head alone, level with laneguide's full-map planner within the
  noise (n16 247 vs 247 with a perfect head, 244 vs 250 at 0.85, 245 vs 242 with the hard head; seed 1: +2, +2);
  turns better (the bay rule), forks/straight on a few worse, forced merges 3x (lane ends unknown); timing constants
  don't matter within sensible ranges.

### How to run GTA on the new path

Everything as today (Valhalla on :8002 on the live map, the bridge with `GTA5_ROUTER`/`GTA5_MAP`), plus:

```
# 1. the Mapbox-compatible server on our map (needs Valhalla and GTA5_MAP)
python -m openpilot.tools.sim.bridge.gta5.mapbox_server --port 8003

# 2. openpilot: navd and mapsd run (NavigationEnabled), navd routes through the server and runs the lane guidance
#    (params, or the env vars for the manager's processes: MAPBOX_HOST=http://127.0.0.1:8003 NAVD_LANE_GUIDANCE=1)
NavigationEnabled=1  NavLaneGuidance=1  NavMapboxHost=http://127.0.0.1:8003  NavLaneChangeStage=3
#    the finetuned model's lower lane change speed: NAVD_LANE_TUNE='{"lc_min_speed": 4}' (n16's 10 is the default)

# 3. the bridge leaves nav to navd: no NavDesire, navInstruction or navRoute of its own; its simulated driver works
#    the stalk from navd's navInstruction.request
GTA5_NAV_PATH=navd   (with GTA5_NAV_MSGS unset or 0; GTA5_GPS default qcom3x: navd localizes from the simulated GNSS)
```

The bridge still writes `NavDestination` (from the game's waypoint) and its own Valhalla route for the overlay and map
driver. Leaving `GTA5_NAV_PATH` unset (`bridge`) runs today's bridge planner unchanged. Navd's events go to its cloudlog
as `navd.request` (the same fields as the planner's `navRequest`).

### In-game checks (n16 unless said; what to watch)

1. **The server's route vs the map**: any trip, `curl 'http://127.0.0.1:8003/directions/v5/mapbox/driving-traffic/<lon,lat;lon,lat>?steps=true&banner_instructions=true&overview=full&geometries=geojson'`
   (lon = x / 111319.49, lat = y / 111319.49): steps at the junctions the overlay shows, lanes only on junction
   approaches with turn:lanes, `valid` the lanes the route can use. And navd picks it up: navInstruction valid,
   maneuverDistance counting down.
2. **A bay that opens late, the bay rule** (the Mapbox path's main gain): R02 `808.6,-2076.0,28.4,265,1>-981.5,158.6`,
   its left turn at (602.9, -372.0) (also R06, R40): Mapbox lists the bay (2 lanes) while the model counts 1; nav asks
   nothing, then signals the turn below 10 m/s within 80 m (request kind turn). Does n16's turn desire take it into the
   bay as it opens? If not, set `NAVD_LANE_TUNE='{"turn_into": 0}'` and compare. Same at (230.4, -222.3) (R11
   `-244.4,-1050.5,26.2,159,1>434.6,882.6`, the lane opens 18 m before).
3. **Counting from the turn's side with a miscounting model**: MD029 `-236.0,-597.2,33.2,20,9>-743.5,-168.0` and LANE_END
   `123.0,-517.0,42.2,198,9>-418.0,-1444.2` (wide kerb lanes, parked cars): the lane nav asks from should follow the
   maneuver's side; navInstruction.request.laneChanges should match the lanes the overlay shows.
4. **Freeway exit across 3 lanes**: R02, the exit at (791.7, -633.2): requests come (3 x 8 + 4) s x speed before the
   last place (~800 m at 29 m/s), one after another 2 s apart; watch they all happen and blind-spot holds behave.
5. **On-ramp merges then an exit**: R01 `-1369.8,-554.1,29.2,125,1>586.6,-302.4`, the slip joining the Del Perro Fwy
   at (-936.5, -564.7) and (-881.2, -546.5), then the exit right at (-694.0, -527.8): no lane change asked for the
   merges (Valhalla's "keep left" there was a Mapbox fork before item 4's fix), then the exit from the right lane.
6. **Two close maneuvers**: R10 `-78.0,-2058.9,21.8,109,0>639.4,-358.9`, the turns at (602.9, -372.2) and (662.8,
   -403.2), 68 m apart: the first turn's lane is the one nearest the second's side.
7. **No lanes from Mapbox**: MD029's turn at (-224.9, -393.2) and R34 `-753.2,-836.4,21.5,269,1>620.4,31.8` at (261.5,
   -365.6): nav falls back to the outermost lane on the turn's side; misses here mean the map's turn:lanes are missing
   or wrong there.
8. **A lane ending**: R10 at (-253.0, -548.8), MDT `-357.8,-1658.9,18.3,331,0>1468.2,-1050.9` at (121.4, -1261.8),
   FX1/R15 at (1107.7, -1644.4): Mapbox says nothing, so nav doesn't move the car out; watch how n16 merges at the end
   (the sim's forced merges, 3x the full map's).
9. **Withdrawal when slow**: R40 `1073.5,-1537.2,27.0,0,1>38.6,4.5`, stopped at the red light at (602.9, -372.1) in the
   wrong lane: no lane change asked below 10 m/s (a request made before is withdrawn: `gave_up`), the turn signalled.
10. **Stage 1 and the lanes UI**: `NavLaneChangeStage=1`, any city trip: the mici alert "Change lanes right for the
    turn / signal to confirm" only while a request is open, NavDesire `laneChange:right`, only the right blinker
    changes lanes below 20 mph; the lane strip only within 402 m of the maneuver (or with a request), its active
    lanes Mapbox's valid ones.

### What's still GTA-specific under `selfdrive/` (the debt)

- `selfdrive/navd/lane_slots.py` imports `Section`, `LEFTS`, `RIGHTS` from `tools/sim/bridge/gta5/map/osm_lanes.py`
  (route input v2's lane slots, on our map's sections). The only code import of `tools/sim` left under `selfdrive/`.
  `planner.py` no longer has one (`turn_targets` moved to `selfdrive/navd/lane_targets.py`).
- `selfdrive/navd/` as a whole (the planner, `inputs.py`'s truth fields, `map_match.py`, `route_input.py`) still plans on
  the full map's lanes (`Route.info`), which only GTA has; it runs inside the bridge. The new path doesn't use it.
- Mentions only (comments/docstrings): `navigation/navd.py` (the server as an example of a custom host),
  `lane_guidance.py`, `modeld/route_input.py`, a few UI files and tests.

### Open questions

1. Does n16's turn desire take the car into a bay as it opens (check 2)? The Mapbox path's turn gains rest on it, as
   laneguide's `turn_into_lane` did.
2. Lane ends: Mapbox has none. Should nav react to the lane head's count dropping on the car's side (a lane ending
   ahead is visible to the model before the end), or leave merges to the model? 3x the forced merges is the cost now.
3. Diverges without turn:lanes in real Mapbox data (3 of 19 UK forks/exits here): the fallback picks the outermost lane
   on the branch's side, wrong at a multi-lane split. Is the lane head good enough to count the branch's lanes, or
   should nav prefer "any lane on that side" there?
4. Bus and cycle lanes are in Mapbox's lane lists (`access`); cycle lanes are dropped here, bus lanes kept. Does the
   lane head count a bus lane? Real-drive logs (Amy's UK drives) would say.
5. Mapbox's maneuver intersection sometimes lists fewer lanes than its approach (Dale Rd); the guidance trusts the
   maneuver's and counts from the side. Worth checking on real drives where the approach has the bay and the node
   doesn't.
6. The emulator's fork lanes are worth ~22 forks in navd_sim; they assume GTA's diverges would be tagged as UK ones
   are. Should the GTA map get explicit turn:lanes at diverges (ynd_to_osm) instead, so the server needn't make them?
7. comma's `test_navigation.py` couldn't run here (native params); its RouteEngine tests should be run on Amy's PC
   with the navd change (`update(full=...)`, the guidance off by default).

### Environment notes

- Final suite (the preflight's set plus the new tests): 212 passed, 1 failed (the known
  `test_live_freeway_merge_exit_and_splits`), 168 s. New tests: `test_mapbox_fixtures.py` 39, `test_mapbox_server.py`
  10, `test_lane_guidance.py` 16, `test_navd_path.py` 1, `test_navd_sim.py` +1.
- Not runnable here: comma's `selfdrive/navigation/tests/test_navigation.py` and anything constructing `Params()`
  (native params need capnp's generated C++ headers); the new navd tests stub `Params` only where it's missing, as
  laneguide's approval tests do. `msgq` was built from its public source, so cereal messaging is real.
- Lint (ruff) on the new and changed files: clean, but for findings in `osm_lanes.py` that predate this work.
- Mapbox: 20 requests in all (1 preflight + 20 fetches, 12 kept), the token read from the environment only; no
  file, log or commit holds it (checked by value over the branch's files and history).
