# Cloud task: lane guidance (navd planner) for comma's nav models

You are a Claude Code cloud session. You see only GitHub, not Amy's PC (no GTA, no GPU, no live services).

Repos:
- `AmyJeanes/openpilot-upstream`: read, and push branch `laneguide-cloud` only. Clone it shallow and single-branch:
  `git clone --depth 1 --branch laneguide-cloud --single-branch`. The full repo is very large.
- `AmyJeanes/gta5-distill` (private): read only, branch `cloud-data`, data under `cloud/` (see `cloud/MANIFEST.md`).

The session may start on another branch. Work on `laneguide-cloud`, which is `gta5` plus this brief.

## Direction (Amy, 16 Oct)

comma now ships nav-capable driving models. `n16` (on HF `commaai/openpilot_driving_models`) takes a 47-float nav
vector (next maneuver only) and a map image, and drives turns and speed itself, with experimental mode on. Our project
no longer rebuilds turning or speed control. It focuses on what comma's models lack: **lane guidance**, meaning being
in the right turn lane, exit lane or fork lane in time to complete the route. The long-term goal (PLAN "D") is that the
model requests lane changes and the driver approves them, auto-approved in GTA. This task builds the baseline that D is
measured against: **nav knows the target lane and requests the changes**.

## What exists (read it first)

- `openpilot/selfdrive/navd/planner.py` (about 1900 lines): navd's planner. It already plans lane changes for turns,
  forks, turn bays, freeway moves and merges, with tries, gaps, keep desires and signal timing. See its constants
  (`LANE_CHANGE_*`, `FORK_*`, `BAY_*`, `SIGNAL_*`, `TURN_NEAR`).
- `inputs.py`: `NavInputs` / `NavOutputs`; the planner asks the driver through `requests`.
- The lane source: game truth, the model's lane head (`model_lane`, `modelV2.laneHead`), or "fused" (planner.py
  around 241 and 1335).
- `lane_slots.py`, `route_input.py`, `maneuvers.py`: the route's target lanes, from the map's turn lanes
  (`osm_lanes.turn_targets`) and lane arrows.
- The bridge side: `openpilot/tools/sim/bridge/gta5/gta5_navd.py` (`nav_inputs`) and `gta5_brain.py`. `GTA5_NOO`:
  "param" follows the UI's Navigate-on-openpilot button; "off" means guidance only, with no requests.
- Tests: `tools/sim/bridge/gta5/test_nav.py`, `test_nav_msgs.py`, `test_lane_slots.py`, `test_lane_numbering.py`,
  `selfdrive/navd/tests/`. The scenario trips are under `tools/sim/bridge/gta5/scenarios/`.
- Offline car and traffic simulation: `tools/sim/bridge/gta5/mapdrive_sim.py`, with a kinematic car carrying the game
  car's lag, `Vehicle` traffic and pedestrians, and real routes via Valhalla. It drives the rule-based map driver, not
  the model.

### Measured on Amy's PC, 16 Oct (offline desire probe on recorded GTA junction approaches)

- n16 takes a **turn** desire pulse well: desire_state goes to 1.00, and the 6 s plan moves 2–12 m the right way. It
  works from a standstill.
- A **lane-change** pulse barely moves n16's plan below about 10 m/s (under 2 m), although the model registers it
  (desire_state about 1.0). City turn lanes are exactly that speed range, so a lane change requested late, once the car
  has slowed for the turn, may not happen at all.
- The turn desire and the nav vector add up. When they agree, the turn is strongest.
- A finetune of n16 that adds a current-lane head and city-speed lane changes is being written in another cloud
  session (`AmyJeanes/gta5-train`, branch `n16`). Don't depend on it, but design for it: the lane head arrives as
  `modelV2.laneHead` (laneIdx, laneCount, prob, idxProbs, countProbs).

### Another session is changing map geometry

Branch `lanegeom-cloud` changes lane-centre geometry. **Don't edit** `map/osm_lanes.py`, `map/ynd_to_osm.py` or
`map/osm_to_roads.py` geometry code. If you find a target-lane (turn lanes / lane arrows) fault in the map, list it
in your result with coordinates instead.

## Step 0: preflight (stop on failure)

Record pass/fail with one line of evidence in `CLOUD_RESULT_LANEGUIDE.md` under "Preflight":

1. Repo: branch `laneguide-cloud` (parent `gta5`). Note both hashes.
2. Data: `cloud-data`. Check sha256 against the manifest for `live18/map1/*` and `live19b/*`, then assemble the live map
   (live18's pbf, `roads.json`, `paths.jsonl`, `nodes.npz`, plus live19b's `lanes.json`). Point `GTA5_MAP` at it.
3. Dependencies from PyPI only (numpy, pyosmium, pyvalhalla, shapely, pytest...). openpilot's own native builds may be
   missing: note which tests need them.
4. Valhalla: build tiles from the pbf with pyvalhalla (`valhalla_build_config` is the venv's console script; `valhalla_build_tiles` and `valhalla_service` are under `site-packages/valhalla/bin`) and run it on port 8002, so `mapdrive_sim.trip_route` and the router tests work. If it's impossible, use made routes and say so.
5. Baseline: the tests listed above. Record the results.
6. Push access: commit only `CLOUD_RESULT_LANEGUIDE.md` and push.

## Work (in order; commit, push and update the result file after each)

### 1. A closed-loop planner harness with a model stand-in

`tools/sim/bridge/gta5/navd_sim.py`: run navd's planner closed loop on real routes, reusing `mapdrive_sim`'s car,
traffic and routes. The driving model is replaced by a parametrised stand-in:
- Lane keeping, slowing for turns and taking them as n16 does: speed from the route's curvature and stop lines. A turn
  pulse is followed by the turn at the junction.
- A lane change on approval, with **success that depends on speed**: completes in about 4–6 s above `v_ok`, takes
  longer or fails below it. Parameters from the probe: default `v_ok` = 10 m/s, plus a "finetuned" setting where it
  works down to about 4 m/s.
- The blind spot and gaps from the simulated traffic.
- The lane source as a setting: truth, or a noisy lane head with a given accuracy and lag.

Metrics per maneuver on the scenario trips and a few dozen real routes:
- in a target lane at the commit point (the last chance to make the turn or exit);
- lane changes requested, done, failed and late;
- maneuvers missed and reroutes;
- time with a requested change hanging.

Report the baseline with today's planner, under both stand-in settings.

### 2. Lane-change timing that fits the model

Using the harness, change the planner so lane changes for a maneuver are done while the car is still fast enough for the
model to make them:
- start earlier on approach, ahead of the slowing for the turn;
- sequence multi-lane moves with gaps;
- know the speed of the road and of the traffic.

Where a change can't be made in time (no gap, too slow), stop asking and leave it to a reroute (or to driving on to
the next junction), rather than forcing it at the junction. Keep openpilot's rules (blind spot, nudge) and comma's
safety intent.

Report the before and after metrics for both stand-in settings, and say which constants changed and why.

### 3. The model's lane as the lane source

Make the "model" / "fused" lane source robust enough to use without game truth:
- hysteresis and a minimum confidence;
- the map's and the model's lane counts disagreeing (the map is sometimes wrong: wide kerb lanes with parked cars in GTA,
  lanes added or dropped);
- lane head dropouts;
- counting from the side that matters for the maneuver (from the right for a right exit).

Test it in the harness with a noisy lane head, at the accuracies a finetune would plausibly reach. Strict and ±1 values
are in `gta5-train` README "G1c" if you can read it; otherwise assume 80–90% strict.

### 4. Request and approve

PLAN D stages: (1) suggest only, (2) auto blinker with driver confirmation on supported cars, (3) auto-approve (GTA).
- A cereal message or field for nav's pending request (kind: lane change or turn; side; why; distance by which it's
  needed).
- A UI prompt in openpilot's UI (`selfdrive/ui`), e.g. "Change lanes right for exit — signal to confirm".
- Approval by the driver's blinker firing the desire through `DesireHelper`, with nudge and blind-spot rules as now.
- A param for the stage. GTA runs stage 3, where the bridge's simulated driver approves.

Every request is logged (time, maneuver, lane, distance, outcome), so model requests can later be scored against
nav's. Add tests for the message, the UI state logic (no rendering test needed) and the approval path.

### 5. Hand-over

`CLOUD_RESULT_LANEGUIDE.md`:
- the metrics table per step;
- what Amy should check in game, with 8–10 routes or places and what to watch for;
- the map target-lane faults you listed;
- open questions.

## Done means

- Branch `laneguide-cloud` pushed, one commit per item, with the rationale in each message.
- Tests: no new failures against the baseline, plus new tests for items 1–4.
- The harness and its metrics are reproducible with one command, documented in the result file.
