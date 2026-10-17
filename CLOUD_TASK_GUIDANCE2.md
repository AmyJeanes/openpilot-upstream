# Cloud task: lane guidance follow-ups (lanes that end, bus lanes, forks without arrows, no slow lane changes)

You are a Claude Code cloud session. You see only GitHub and the internet allowed by the environment, not Amy's PC
(no GTA, no GPU, no live services).

Repos:
- `AmyJeanes/openpilot-upstream`: read, and push branch `guidance2-cloud` only. Clone it shallow and single-branch:
  `git clone --depth 1 --branch guidance2-cloud --single-branch`. The full repo is very large.
- `AmyJeanes/gta5-distill` (private): read only, branch `cloud-data`, data under `cloud/` (see `cloud/MANIFEST.md`).

The session may start on another branch. Work on `guidance2-cloud`, which is `gta5` (e55cb3b45) plus this brief.
No Mapbox token is needed: the fixtures in `selfdrive/navigation/tests/fixtures/mapbox/` are invented responses in
Mapbox's schema (`make_fixtures.py`), and navd_sim builds Mapbox responses from our map in process.

Another cloud session (`navdmove-cloud`) moves `openpilot/selfdrive/navd/` into the GTA bridge at the same time. Keep
your edits to `selfdrive/navd/planner.py` small and self-contained (item 4); Amy's local session merges this branch first.

## Background

`CLOUD_RESULT_NAVMAPBOX.md` (repo root) is the last session's result: step 1 lane guidance in comma's navd
(`openpilot/selfdrive/navigation/lane_guidance.py`, behind `NavLaneGuidance`), fed only Mapbox data and the model's
lane head; the GTA Mapbox-compatible server (`tools/sim/bridge/gta5/mapbox_server.py`); `navd_sim --nav map,mapbox`.
Read it first, especially item 4 (the measurements) and the open questions. Amy's answers to those questions:

1. **Lanes that end** (Mapbox has no lane-end data; the Mapbox path has 3x the full map's forced merges): build
   option B below now and see how well it works and what false positives it has. Longer term the model should learn
   to leave an ending lane itself (not this task).
2. **Forks without arrows**: keep the fallback (the outermost lane on the branch's side: never the wrong branch,
   sometimes one change more than needed). Separately, GTA's map will get per-lane arrows at diverges (not this task).
3. **Bus lanes**: Mapbox marks them (`access: {designated: [bus]}`). Never send the car into one. Whether the lane head
   counts a bus lane is for Amy's real drives later; keep them in the lane count for now.

## Items

1. **Option B for lanes that end** (`lane_guidance.py`): when an intersection ahead on the route lists fewer of our
   lanes than the car's road has now and the car is in a lane that drops out there, move it over in good time, with
   the same timing and request machinery as a maneuver's changes (a new request reason, e.g. `laneEnds`). Mapbox
   doesn't say which lane ends: infer the side from the lane lists (align the two lists by their arrows from the left
   and from the right; the side whose alignment holds keeps its lanes), and do nothing when it's ambiguous. Count the
   car's lane from that side with the lane head, as the maneuvers do. Never ask for it against a maneuver's targets.
   Measure in navd_sim against `gta5`'s Mapbox path (same seeds, trips and heads as `CLOUD_RESULT_NAVMAPBOX.md`):
   forced merges, in-lane at the commit points, LC requested / done / failed, and **false positives**: requests where
   the full map says no lane of the car's ended. Break the false positives down by cause, at least: turn:lanes on only
   part of a road (fewer lanes listed but none ended), a maneuver's node listing fewer lanes than its approach (the
   bay case, `turn_bay_right`), a lane ending on the other side of the road, lane gains and on-ramps. Tune or gate the
   rule from what you find; say what you tried.
2. **Bus lanes never a target** (`lane_guidance.py`): never a target lane, never a lane a change goes into; still
   counted. Test with the `lanes_vary_bus` fixture and a made route.
3. **Forks without arrows**: a test pinning the fallback (outermost lane on the branch's side) at a fork and an exit
   with no lanes from Mapbox.
4. **No lane change asked below the model's lane-change speed on the bridge planner** (`selfdrive/navd/planner.py`,
   the path GTA drives by default, `GTA5_NAV_PATH=bridge`): Amy saw n16 take a lane-change desire at about 8 m/s like
   a turn and hit a wall. The lane guidance already withdraws a change below `lc_min_speed` and leaves it to the turn
   desire; do the same in the planner against the model's lane-change speed (`model_lc_speed`, what navd_sim's `v_ok`
   stands in for). navd_sim's "LC late (asked < v_ok)" for the map nav should go to 0; report what it does to in-lane.

## Known map issue (don't fix here)

live21b's lane geometry has a bug Amy found in game: where two of GTA's parallel chains join or split, `RouteLanes`
slides the lanes up to 5 m sideways over ~80 m (the route line stays put), and the full-map planner's lanes jump there.
It's being fixed locally in the map data. Compare your runs with the baseline on the same map, and note any result
that hinges on such a spot.

## Step 0: preflight (stop on failure)

Record pass/fail with one line of evidence each in `CLOUD_RESULT_GUIDANCE2.md` under "Preflight": the repo hashes;
the data (sha256 against `cloud/MANIFEST.md` for `live18/map1/{paths.jsonl,nodes.npz}*` and `live21b/*`, the map
assembled as before: live21b's `gta5.osm.pbf`, `lanes.json`, `roads.json` + live18's `paths.jsonl`, `nodes.npz`,
`GTA5_MAP` and `~/gta5map_lanes` pointing at it); dependencies from PyPI (as `CLOUD_RESULT_NAVMAPBOX.md`'s preflight:
msgq built from its public source works); Valhalla on :8002 from the pbf; the baseline suite
(`openpilot/selfdrive/navigation/tests`, `tools/sim/bridge/gta5/test_mapbox_server.py`, `test_navd_path.py`,
`test_nav.py`, `test_nav_msgs.py`, `test_lane_slots.py`, `test_lane_numbering.py`, `test_navd_sim.py`,
`selfdrive/navd/tests`, `selfdrive/controls/tests/test_nav_approval.py`, `selfdrive/ui/tests/test_nav_request.py`;
all pass on Amy's PC) and the navd_sim baseline (`--nav map,mapbox --model n16,finetuned --lane truth,model:0.85`,
seed 0: Amy's PC gives n16/truth map 247, mapbox 247; n16/0.85 250 / 244; finetuned/truth 262 / 254; finetuned/0.85
255 / 252). Commit and push the preflight before item 1.

## Rules

- One commit per item, each with its tests; the result file updated as you go; push after each.
- Commit messages: what and why, ending with the line `Assisted-by: Claude Code`. No other trailers.
- Nothing GTA-specific under `openpilot/selfdrive/navigation/` (it imports nothing from `tools/sim`).
- Don't change the map data, the Mapbox emulator's output or navd_sim's judge, except to add the false-positive
  breakdown; don't relabel or retune unrelated defaults.
- Stay inside the brief; anything else you find goes under "Open questions" in the result.

## Hand-over (`CLOUD_RESULT_GUIDANCE2.md`)

Preflight; per item what changed, the measurements (tables against the baseline, both seeds where it matters) and the
tests; the false-positive breakdown for item 1; open questions; and **in-game checks for Amy** (place, trip spec as
`x,y,z,heading,lane>dx,dy` from navd_sim's trips, what to watch), at least: a lane ending before a junction, a lane
ending on the far side, a bay turn (no false lane-end ask), a road with bus lanes if GTA has one, a slow approach
(no lane change below the model's speed).
