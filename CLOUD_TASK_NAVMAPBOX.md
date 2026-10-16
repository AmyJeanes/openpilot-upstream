# Cloud task: navigation on Mapbox data, with a Mapbox-compatible GTA server (step 1 lane guidance)

You are a Claude Code cloud session. You see only GitHub and the internet allowed by the environment, not Amy's PC
(no GTA, no GPU, no live services).

Repos:
- `AmyJeanes/openpilot-upstream`: read, and push branch `navmapbox-cloud` only. Clone it shallow and single-branch:
  `git clone --depth 1 --branch navmapbox-cloud --single-branch`. The full repo is very large.
- `AmyJeanes/gta5-distill` (private): read only, branch `cloud-data`, data under `cloud/` (see `cloud/MANIFEST.md`).

The session may start on another branch. Work on `navmapbox-cloud`, which is `laneguide-merge` plus this brief.

The environment provides **`MAPBOX_TOKEN`** (Amy's Mapbox public token) and access to `api.mapbox.com`. Read it from
the environment only. Never print, log, echo or commit it; strip `access_token` from any URL you save or print; keep
real responses few (test fixtures). Stay well inside the free tier (a few hundred requests at most).

## Direction (Amy, 16 Oct late)

Navigation must work purely from data a real car has, and build on comma's nav (`openpilot/selfdrive/navigation/`:
`navd.py` routes with Mapbox Directions through comma's proxy, or `api.mapbox.com` with `MAPBOX_TOKEN`; `mapsd.py`
renders the nav model's map image). Amy: **"realistically for this project to work, we need to rely on mapboxes data
alone and our GTA implementation should feed equivalent data."** And: "We'll keep the full data for our map driver
for training data to the model to help it enter turn lanes."

So:
- nav sees only what Mapbox gives: route geometry, steps and maneuvers, each intersection's approach lanes
  (`indications`, `valid`, `active`, `valid_indication`), banner lane components, `maxspeed` (and congestion), plus GPS,
  carState and modelV2 (`laneHead`: lane index from the left, lane count, probabilities);
- in GTA, the bridge serves a **Mapbox-compatible Directions API** built from our map, with the same sparseness: lanes
  only at intersections, no lane-opening points, no lane widths or geometry, no truth lane;
- nothing GTA-specific under `openpilot/selfdrive/` (GTA code lives in `openpilot/tools/sim/bridge/gta5/`);
- the full map (OSM lane tags, bays, openings) stays for the map driver, training labels and the GTA server only.

Full context and the agreed plan: read `NAV_MAPBOX_BRIEF.md` below (copied here) before starting.

### Step 1 lane guidance, as agreed

- Target lanes: the maneuver intersection's valid lanes; when the next maneuver follows within about 100 m, plan both.
- The car's lane: `laneHead`, counted from the maneuver's side (laneguide's `model_count_side` / `_side_that_matters`).
  Before a turn bay opens, the model counts fewer lanes than Mapbox lists; counted from the turn's side, the target is
  the outermost lane on that side, and the car is in place there. The turn desire takes it into the bay as it opens
  (laneguide's `turn_into_lane`, navd's `BAY_SIGNAL`).
- Timing: ask for the changes when time left <= lanes away x `lc_time` + margin. Time left = (distance to the last
  place the change must be done, a few tens of m before the stop line) / planning speed, with planning speed =
  max(current speed, share x `maxspeed`), so slowing traffic doesn't delay the ask. A minimum distance for slow
  approaches; a longer `lc_time` at freeway speed. A change the car is too slow to make (n16 below ~10 m/s) is
  withdrawn and left to the turn desire.
- Request and approve (laneguide's `navInstruction.request`, `NavLaneChangeStage`, sided `NavDesire`): stage 1 the
  driver's blinker confirms (real car); stage 3 auto-approved in GTA.
- Lane UI (`navInstruction.lanes`) only under active guidance, from Mapbox's lanes, like nav apps.
- No lanes from Mapbox at a junction: fall back to the outermost lane on the turn's side.

## What exists (read it first)

- `openpilot/selfdrive/navigation/navd.py`, `mapsd.py`, `helpers.py` (comma's; `helpers.py` has
  nav-model-integration's `valhalla_type` / `navigation_input`): the Mapbox request (`mapbox_host`,
  `driving-traffic`, `steps`, `banner_instructions`...), `navInstruction`/`navRoute`.
- `openpilot/selfdrive/navd/` (ours, a library the GTA bridge drives in process): `planner.py` (lane plans, timing,
  requests, `Tune`; `lane_source` defaults to "model" now, the bridge asks for "fused" via
  `tools/sim/bridge/gta5/gta5_navd.bridge_tune`), `inputs.py`, `lane_slots.py`, `maneuvers.py`, `map_match.py`,
  `route_input.py`. Known debt: `planner.py` and `lane_slots.py` import `tools/sim/bridge/gta5/map/osm_lanes.py`.
- `openpilot/tools/sim/bridge/gta5/`: `gta5_nav_msgs.py` / `gta5_nav_pub.py` (the bridge publishes navInstruction and
  navRoute itself today), `navmodel/nav_input.py` (Valhalla maneuvers to Mapbox words for n16), `navmodel/gta5_mapsd.py`,
  `map/router.py` (Valhalla routing on our map, `Route.info`), `map/osm_lanes.py` (`RouteLanes`, `turn_targets`),
  `map/gta5_map.py` (`to_lat_lon` / `to_game`: GTA metres <-> the map's lat/lon), `navd_sim.py` (closed-loop planner
  harness with a model stand-in, laneguide's), `test_*.py`.
- `CLOUD_RESULT_LANEGUIDE.md` (navd_sim baseline, request/approve) and `CLOUD_RESULT_LANEGEOM.md` (map geometry, live21).

## Step 0: preflight (stop on failure)

Record pass/fail with one line of evidence in `CLOUD_RESULT_NAVMAPBOX.md` under "Preflight":
1. Repo: branch `navmapbox-cloud`, parent `laneguide-merge` (913fb0f7c). Note both hashes.
2. Data: `cloud-data`. Check sha256 against the manifest for `live18/map1/{paths.jsonl,nodes.npz}*` and `live21b/*`,
   then assemble the map: live21b's `gta5.osm.pbf`, `lanes.json`, `roads.json` with live18's `paths.jsonl` and
   `nodes.npz`. Point `GTA5_MAP` at it (and `~/gta5map_lanes`, which some tests read).
3. Dependencies from PyPI only (numpy, pyosmium, pyvalhalla, shapely, scipy, pytest, pycapnp...). openpilot's native
   builds (msgq, params, raylib) may be missing: note which tests need them.
4. Valhalla: tiles from the pbf with pyvalhalla, `valhalla_service` on port 8002.
5. Mapbox reachable: one Directions request with `MAPBOX_TOKEN` returns 200 (print the status and route count only).
6. Baseline: `test_nav.py`, `test_nav_msgs.py`, `test_lane_slots.py`, `test_lane_numbering.py`, `test_navd_sim.py`,
   `selfdrive/navd/tests`, `selfdrive/controls/tests/test_nav_approval.py`, `selfdrive/ui/tests/test_nav_request.py`,
   and `navd_sim --model n16,finetuned --lane truth,model:0.85` (the laneguide baseline on live21b). Known failures on
   Amy's PC too: `test_live_freeway_merge_exit_and_splits`, the overlay cache test.
7. Push access: commit only `CLOUD_RESULT_NAVMAPBOX.md` and push.

## Work (in order; commit, push and update the result file after each)

### 1. What Mapbox really returns

Fetch Directions responses (`steps=true`, `banner_instructions=true`, `annotations=maxspeed,congestion`,
`overview=full`, `geometries=polyline6`, the profile navd uses) for about 12 UK routes that pass: a right turn into a
turn bay that opens shortly before the junction, a lane that has always been there, a lane added mid-block, a
motorway exit with a lane drop, a lane-gain on-ramp, two maneuvers within 100 m, a roundabout, and a junction with no
lane data. Pick real places (Hampshire/Southampton/Portsmouth/London are fine; name each). Save them as fixtures under
`openpilot/selfdrive/navigation/tests/fixtures/mapbox/` with the token stripped, and document what `lanes` holds:
present at which intersections, how bays and added lanes look, how far before a maneuver the banner lanes start
(`distanceAlongGeometry`), what `maxspeed` looks like. This is the ground truth for items 2 and 3.

### 2. A Mapbox-compatible Directions server for GTA

`openpilot/tools/sim/bridge/gta5/mapbox_server.py`: an HTTP server answering
`/directions/v5/mapbox/<profile>/<lon,lat;lon,lat...>` with the same JSON schema as item 1's fixtures, from a
Valhalla route on our map (`map/router.py`), in the map's lat/lon (`map/gta5_map.py`):
- legs, steps, maneuvers (type/modifier as Mapbox's: reuse `navmodel/nav_input.py`'s mapping), `bannerInstructions`
  with lane components, distances, durations, geometry (polyline6);
- `intersections` at each junction node passed, with `bearings`, `entry`, `in`/`out` and `lanes`: the approach lanes
  as they are at the junction's stop line (bays included), from the map's `turn:lanes` / lane counts (`RouteLanes`,
  `turn_targets`), `valid`/`active`/`valid_indication` for the route's move;
- `annotation.maxspeed` from the map's `maxspeed` (else by road class), congestion "unknown";
- nothing Mapbox lacks: no opening points, widths, lane geometry, truth lane.

navd reaches it by its host setting (`navd.py` already switches `mapbox_host` on `MAPBOX_TOKEN`; add a `MAPBOX_HOST`
env/param override, so navd itself stays game-agnostic). Tests: schema parity with the fixtures (same keys and types
where present), a bay appearing only as an extra lane at its junction, two close maneuvers, a fork/exit.

### 3. Step 1 lane guidance in comma's navd

A lane guidance module under `openpilot/selfdrive/navigation/` (or our planner's pieces moved there) that runs in
comma's navd process from navd's Mapbox route plus `modelV2.laneHead` and `carState`, and emits laneguide's
`navInstruction.request`, `NavDesire` (sided by stage) and `navInstruction.lanes` under active guidance, following
"Step 1 lane guidance" above. It must import nothing from `tools/sim` (move generic helpers such as `turn_targets`
into navd if needed). Keep the existing bridge-driven planner path working (GTA runs today), with the new path behind a
switch, documented. Tests: timing by speed and lanes away (including freeway exits), counting from the turn's side
with a bay not yet open, two close maneuvers, no-lanes fallback, withdrawal when too slow, the request and the sided
desire, the lanes shown only under guidance.

### 4. Measure

`navd_sim` with nav fed only the emulated Mapbox data (item 2's server or the same builder in process) and the lane
head stand-in (`model:0.9`, `model:0.85`, `model:0.8`, the "hard" head), against laneguide's full-map baseline on the
same trips and seeds: in a target lane at commit points (turns, forks/exits, straight on), lane changes requested /
done / failed / late, hanging time, and which misses come from information Mapbox doesn't give (bays opening late,
lanes added mid-block). Then tune `lc_time`, the margin, the maxspeed share and the minimum distance, and report what
changed and why.

### 5. Hand-over

`CLOUD_RESULT_NAVMAPBOX.md`:
- what Mapbox's lanes really contain (item 1), and how the GTA server matches it;
- the metrics table (item 4) against the baseline;
- how to run GTA on the new path (server, navd host, switches), and 8-10 in-game checks with what to watch for;
- what's still GTA-specific under `selfdrive/` (the remaining debt), and open questions.

## Done means

- Branch `navmapbox-cloud` pushed, one commit per item, with the rationale in each message.
- No new test failures against the baseline; new tests for items 2 and 3.
- `selfdrive/navigation/` and the new guidance code import nothing from `tools/sim`; the token appears nowhere in the
  repo, logs or result file.
- navd_sim's comparison reproducible with one documented command.

---

## NAV_MAPBOX_BRIEF.md (the agreed plan, for reference)

# Brief: navigation on Mapbox-equivalent data (step 1 lane guidance)

Agreed with Amy, 16 Oct 2026 (late). Supersedes PLAN §E's N2-N6 rows for the nav side; the map work and the map driver
are unchanged.

## Principles

- **Nav runs on what a real car gets from Mapbox, nothing more.** Route geometry, steps and maneuvers, each
  intersection's approach lanes (arrows, valid, active), `maxspeed` (and congestion where present), plus GPS, carState and
  modelV2 (including `laneHead`). No OSM lane tags, bay openings, lane widths or truth lanes in nav.
- **GTA feeds equivalent data.** The bridge emulates a Mapbox Directions response from our map, with the same sparseness:
  lanes only at intersections, no opening points, no lane geometry. GTA must not flatter nav.
- **The full map stays for GTA's own use:** the map driver (expert), training labels (lane count, current lane, bay
  entry) and the emulator itself. Its main job now is teaching the model to enter turn lanes and know its lane.
- **Nothing GTA-specific under `selfdrive/`** (PLAN §E). Map fixes live in the OSM itself (lanegeom baked into the pbf).
- **Build on comma's nav** (`selfdrive/navigation/navd.py`, `mapsd.py`): one nav process, swappable router.

## What Mapbox gives (and doesn't)

From the Directions API (`steps=true`, `banner_instructions=true`, `annotations=maxspeed,congestion`):
- `steps[].intersections[].lanes[]`, left to right whatever the driving side: `indications`, `valid`, `active`,
  `valid_indication`. Present for every intersection on the route from the moment it is computed.
- Banner `sub` components of type `lane` for the next maneuver (the UI's lane strip).
- Not given: where a lane begins or ends (no bay/turn-lane-opened flag), widths, lanes between intersections, which
  lane carries on as which. Richer lane data (Electronic Horizon/ADAS) is SDK-only and separately licensed.
- To verify with real responses (task 4) before relying on details.

## Tasks

1. **Router interface in navd** (`selfdrive/navigation/`): `route(dest, pos, heading) -> Route` returning a
   Mapbox-shaped structure (geometry, steps, maneuvers with type/modifier/distances, intersections with lanes,
   annotations). Backends: `mapbox` (comma's proxy or `MAPBOX_TOKEN`, as navd does now) and `gta` (the emulator, over a
   socket/HTTP from the bridge, so navd stays game-agnostic). Tile source for mapsd is a separate setting (comma's hosted,
   self-hosted PMTiles, local extract; GTA = our `gta5_roads.pmtiles`).
2. **GTA Mapbox emulator** (bridge side, `tools/sim/bridge/gta5/`): from the Valhalla route on our map, emit the
   Mapbox-shaped response: steps/maneuvers as Mapbox types (reuse `navmodel/nav_input.py`'s Valhalla->Mapbox mapping),
   intersections at every junction node passed with approach lanes from `turn:lanes`/lane counts at the stop line (the
   lanes as they are at the junction, bays included), valid/active for the route's move, `maxspeed`. Tests: shape matches
   Mapbox's schema; a turn bay shows only as an extra lane at its junction.
3. **Step 1 lane guidance** (navd, reusing laneguide's planner pieces: request/approve, robust model lane source,
   request log, UI prompt):
   - Target lanes: the maneuver intersection's valid lanes (and the next maneuver's when it follows within ~100 m: plan
     both together).
   - Car's lane: `laneHead` (index, count), counted from the maneuver's side (laneguide's `model_count_side`); before a
     bay opens the target is "the outermost lane on the turn's side", and the car counts as in place there.
   - Timing: ask when time left <= lanes away x `lc_time` + margin, where time left = (distance to the last place the
     change must be done, a few tens of m before the stop line) / planning speed, planning speed = max(current speed,
     share x `maxspeed`) so slowing traffic doesn't delay the ask; a minimum distance for slow approaches; freeway exits
     with a longer `lc_time` at speed. Withdraw a request the car is too slow to carry out (n16 < ~10 m/s) and leave it
     to the turn desire.
   - Late bays: the turn desire as the bay opens (laneguide's `turn_into_lane`, navd's BAY_SIGNAL cue).
   - Request and approve: stage 1 suggest (the driver's blinker confirms) on a real car; stage 3 auto in GTA.
   - Lane UI only under active guidance (within the maneuver's guidance distance), like nav apps.
   - No Mapbox lanes at a junction: fall back to the outermost lane on the turn's side.
4. **Real Mapbox check:** pull Directions responses for ~10 UK junctions Amy knows (turn bays, lanes that open, exits,
   double turns) with a personal token (free tier); record what `lanes` holds and where; compare with the emulator's
   output for similar GTA places. Amy's token is provided in the cloud environment as `MAPBOX_TOKEN` (api.mapbox.com
   reachable): read it from the environment only; never print, log or commit it, and strip `access_token` from any
   saved URL. Keep the saved responses few (test fixtures), in the private repo.
5. **Measure:** `navd_sim` with nav seeing only the emulated Mapbox data and the lane head stand-in (truth off), against
   laneguide's numbers on the full map; then in-game runs with `NAVD_LANE_SOURCE=model`.
6. **Lane head training** (gta5-train `n16` and successors): lane count and current lane from GTA ground truth (lane
   head from the kerb, `lane_head_kerb` -> cereal `laneHead` from the left), bay entry from the map driver's
   recordings; later real-drive labels (pseudo-labels, Amy's UK drives). This is what makes step 1 work without truth.

## Out of scope for step 1

Lane planning from OSM on the real car, route-to-OSM matching, sunnypilot's mapd (speed limits/road names only; no lane
tags, junctions or route), multi-source map versioning (one source: Mapbox; GTA: one generated pbf).

## Acceptance

- `selfdrive/` imports nothing from `tools/sim`; navd runs with either router backend unchanged.
- The emulator's response validates against Mapbox's schema and carries no lane data Mapbox lacks.
- navd_sim and an in-game set with nav on emulated Mapbox data + model lane: in-lane rate at commit points reported
  against laneguide's full-map baseline; failed/late lane changes reported.
