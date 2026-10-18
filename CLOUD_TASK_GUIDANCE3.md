# Cloud task: lane guidance follow-ups 3 (a buffer at the lane change speed, the turn gap, wrong-side lane-end asks, chores)

You are a Claude Code cloud session. You see only GitHub and the internet allowed by the environment, not Amy's PC
(no GTA, no GPU, no live services).

Repos:
- `AmyJeanes/openpilot-upstream`: read, and push branch `guidance3-cloud` only. Clone it shallow and single-branch:
  `git clone --depth 1 --branch guidance3-cloud --single-branch`. The full repo is very large.
- `AmyJeanes/gta5-distill` (private): read only, branch `cloud-data`, data under `cloud/` (see `cloud/MANIFEST.md`).

The session may start on another branch. Work on `guidance3-cloud`, which is `gta5` (2658f3197) plus this brief.
No Mapbox token is needed (the fixtures are invented responses in Mapbox's schema; navd_sim builds Mapbox responses
from our map in process). If `MAPBOX_TOKEN` is set in the environment, unset it for every run (item 4 makes the test
suite do that itself).

## Background

Read `CLOUD_RESULT_GUIDANCE2.md` (repo root) first: its item 4 (no lane change asked below `model_lc_speed` on the
bridge planner) and its open questions are where this starts. Since then:
- `CLOUD_RESULT_NAVDMOVE.md`: the bridge's planner moved from `openpilot/selfdrive/navd/` to
  `openpilot/tools/sim/bridge/gta5/navd/` (module names unchanged); the route input's layouts are in
  `selfdrive/modeld/route_layout.py`.
- Amy's local merge (2658f3197) found four tests in `tools/sim/bridge/gta5/test_blindspot.py` still pinning lane
  change asks at 1-8 m/s; the cloud couldn't import that file (opendbc's `car.capnp` beside `openpilot.cereal`:
  capnp "Duplicate ID"). They now run their mechanics with `model_lc_speed` 0, the late merge also checks nothing is
  asked at 1 m/s by default, and the bay check reports a bay the map reads as oncoming before the speed gate. Try to
  make `test_blindspot.py`, `test_driver.py` and `test_world.py` importable in your environment (say how); if you
  can't, list every behaviour change that could touch them so Amy's local session runs them.

Amy's answers to guidance2's open questions: keep option B (it costs nothing); the lane ends behind the forced merges
are for the model later. The rest is below.

## Items

1. **A buffer at the lane change speed** (`tools/sim/bridge/gta5/navd/planner.py` `_lc_too_slow`, and
   `selfdrive/navigation/lane_guidance.py`'s `lc_min_speed` withdrawal): a car hovering at the model's lane change
   speed (10 m/s for n16) can see a change asked, withdrawn, asked again. Ask only at or above the speed; withdraw a
   change not started only once the car is a margin below it (a tune value, e.g. 1 m/s). First measure how often it
   happens in navd_sim (requests withdrawn as too slow and asked again for the same move within, say, 10 s), per run,
   before and after. Report in-lane and LC requested / done / gave up / failed against the baseline.
2. **The turn gap between ~8.5 and 10 m/s** (both planners): below `model_lc_speed` no lane change is asked, and the
   turn signal that takes the car into a turn's lane (`turn_into` in the planner, `turn_into` in lane_guidance) only
   happens below `min(model_lc_speed, 19 mph)`. In between, a turn needing a lane gets neither. Note openpilot's
   DesireHelper turns a blinker above its lane change speed minimum (about 20 mph) into a lane change desire, not a
   turn, so signalling there is the hazard item 4 of guidance2 removed. First measure: how often a move's last chance
   falls in the band in navd_sim, and what happens (made late, skipped to the route, missed). Then the smallest fix
   that keeps the move alive until the car is slow enough for the turn signal or fast enough for a change (for
   instance, not giving the move up to the route while the car is in the band and the move is still reachable), with
   tests. Don't signal a turn above DesireHelper's lane change minimum.
3. **The planner's own lane-ending asks on the wrong side**: guidance2's lane-end check (navd_sim's `lane_end_ask`,
   the full-map planner's asks, not option B's) found 1-8 per run `other_side` (another lane ends there, not the
   car's), more with the noisy lane head. Find why (the car's lane reading, the lane maps, the side picked), fix what's
   clearly a bug, and report the counts before and after. Leave the judge as it is.
4. **Chores**, each its own commit:
   - `selfdrive/navigation/tests/test_navigation.py::test_request_and_instruction` fails whenever `MAPBOX_TOKEN` is
     set (navd then uses api.mapbox.com): make the test clear it (monkeypatch), not the code.
   - A lint-only commit for the pre-existing ruff findings in the files `CLOUD_RESULT_NAVDMOVE.md` lists (B023
     closures over loop variables, F841, E501, B905, ISC002, PLR1704, B007), with ruff from `uv.lock`. B023 and B905
     fixes can change behaviour: bind the loop variable as a default argument (or restructure) so behaviour stays the
     same, and use `strict=` only where the lengths must match; show the test suite and a navd_sim run identical
     before and after (request log byte-identical, as navdmove did).

## Known map issue (don't fix here)

Amy's local session is fixing live21b's lanes at GTA's chain joins and some lane fits (branch `laneslide`, not in
`gta5`). Compare your runs with the baseline on the same map (live21b, as in the data), and note any result that
hinges on such a spot.

## Step 0: preflight (stop on failure)

Record pass/fail with one line of evidence each in `CLOUD_RESULT_GUIDANCE3.md` under "Preflight": the repo hashes;
the data (sha256 against `cloud/MANIFEST.md` for `live18/map1/{paths.jsonl,nodes.npz}*` and `live21b/*`, the map
assembled as before: live21b's `gta5.osm.pbf`, `lanes.json`, `roads.json` + live18's `paths.jsonl`, `nodes.npz`,
`GTA5_MAP` and `~/gta5map_lanes` pointing at it); dependencies from PyPI as in guidance2's preflight (msgq from its
public source; native params if you can, as guidance2 did, so `test_navigation.py` runs); Valhalla on :8002 from the
pbf; the baseline suite (guidance2's set with the moved path: `openpilot/selfdrive/navigation/tests`,
`tools/sim/bridge/gta5/navd/tests`, `test_mapbox_server.py`, `test_navd_path.py`, `test_nav.py`, `test_nav_msgs.py`,
`test_lane_slots.py`, `test_lane_numbering.py`, `test_navd_sim.py`, `selfdrive/controls/tests/test_nav_approval.py`,
`selfdrive/ui/tests/test_nav_request.py`, `selfdrive/modeld/test_route_input.py`, plus `test_blindspot.py` if you
get it importing); and the navd_sim baseline (`--nav map,mapbox --model n16,finetuned --lane truth,model:0.85`,
seeds 0 and 1) at 2658f3197, kept for the comparisons. Commit and push the preflight before item 1.

## Rules

- One commit per item, each with its tests; the result file updated as you go; push after each.
- Commit messages: what and why, ending with the line `Assisted-by: Claude Code`. No other trailers.
- Nothing GTA-specific under `openpilot/selfdrive/navigation/` (it imports nothing from `tools/sim`).
- Don't change the map data, the Mapbox emulator's output or navd_sim's judge; don't retune unrelated defaults.
- Stay inside the brief; anything else you find goes under "Open questions" in the result.

## Hand-over (`CLOUD_RESULT_GUIDANCE3.md`)

Preflight; per item what changed, the measurements (tables against the baseline, both seeds) and the tests; open
questions; and **in-game checks for Amy** (place, trip spec as `x,y,z,heading,lane>dx,dy` from navd_sim's trips, what
to watch), at least: a car hovering at about 10 m/s before a lane change (one ask, no on-off), a turn whose last
chance falls between 8.5 and 10 m/s, and one of item 3's fixed lane-end asks.
