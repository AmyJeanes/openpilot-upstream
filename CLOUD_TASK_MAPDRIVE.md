# Cloud task: map driver lane changes (gores, stuck gap waits) and red-light reading

You are a Claude Code cloud session. You see only GitHub, not Amy's PC (no GTA, no GPU, no live services).

Repos: `AmyJeanes/openpilot-upstream` (read and push, branch `mapdrive-cloud` only) and `AmyJeanes/gta5-distill`
(private, read only, branch `cloud-data`). The session may start on another branch of openpilot-upstream: work on
`mapdrive-cloud`, which is `gta5` plus this brief.

## Background

Amy drives openpilot in GTA V through a bridge: `openpilot/tools/sim/bridge/gta5/` on branch `gta5` of
`AmyJeanes/openpilot-upstream` (a fork of `commaai/openpilot`; the repo root holds an `openpilot/` folder). A rule-based
"map driver" (`gta5_mapdrive.py`, class `MapDrive`) drives GTA from our own map (an OSM pbf in a fake lat/lon frame,
`map/`), routing with Valhalla (`map/router.py`). Its offline harness is `mapdrive_sim.py` (a kinematic car with the game
car's lag, made routes and real routes on the live map, and simulated vehicles and pedestrians: `Vehicle`,
`pedestrian`, `drive`). Tests: `test_mapdrive.py`, `test_mapdrive_lead.py`, `test_blindspot.py`, `test_lane_slots.py`.

The live map is now **live19b**: live18's pbf, roads, paths and nodes, with a `lanes.json` that adds **hatched areas**
(kind `hatched` in `lanes.json`'s `kinds`; each line is `[kind, x0, y0, x1, y1, ...]`, a ring in game metres): gores,
painted medians, ghost islands and parking-bay blocks that no car should cross, which Amy approved one by one. The
router already won't route a shortcut link through one. The overlay and map view draw them (`gta5_overlay.hatch_lines`,
`hatched_areas`). The map driver doesn't know about them yet.

## Step 0: preflight (stop on failure)

Record pass/fail with one line of evidence in `CLOUD_RESULT_MAPDRIVE.md` under "Preflight":

1. Repo: `AmyJeanes/openpilot-upstream` branch `mapdrive-cloud`, shallow and single branch (`git clone --depth 1
   --branch mapdrive-cloud --single-branch`, or fetch just that branch; the full repo is very large). Its parent commit
   is `gta5` at `647143d89`; note both hashes.
2. Data: `AmyJeanes/gta5-distill` (private) branch `cloud-data`. Check sha256 against `cloud/MANIFEST.md` for
   `cloud/live18/map1/*` and `cloud/live19b/*`. Assemble the live map in one folder: live18's `gta5.osm.pbf`,
   `roads.json`, `paths.jsonl`, `nodes.npz` (gunzip the `.gz` ones) plus **live19b's `lanes.json`**; point `GTA5_MAP`
   at it.
3. Dependencies from PyPI only (numpy, pyosmium, pyvalhalla, shapely, pytest...). If an install is blocked, record it
   and design around it rather than working around the block.
4. Valhalla: build tiles from the pbf with pyvalhalla (`valhalla_build_config` is the venv's console script;
   `valhalla_build_tiles` / `valhalla_service` are under `site-packages/valhalla/bin`) and run it on port 8002, so
   `mapdrive_sim.trip_route` and the router tests work.
5. Baseline: `pytest -q test_mapdrive.py test_mapdrive_lead.py test_blindspot.py test_lane_slots.py` on the
   untouched branch, with GTA5_MAP set. On Amy's PC today: test_mapdrive + test_mapdrive_lead = 68 passed, 1 failed
   (`test_collision_routes`, known). Record yours; that's the bar for "no regressions".
6. Push access: on `mapdrive-cloud` (it already exists, with this brief), commit only `CLOUD_RESULT_MAPDRIVE.md` and
   push.

If 1, 2 or 6 fails, stop and push (or report) the preflight. If 4 fails, use made routes only and say so.

## Files generated from Amy's local data (in `cloud-data`, don't modify)

- `cloud/live19b/map1/lanes.json.gz`: built by `osm_to_roads.py --hatched` (`cloud/live19b/build19b.sh`).
- `cloud/live19b/hatched19b.json`: the hatched areas selected from Amy's review decisions (`select19b.py`). The
  decisions are hers: never add, drop or move an area to make a test pass. If an area looks wrong, list it.
- live18 files: as in the manifest.

## Work (in order; commit, push and update the result file after each)

### 1. Lane changes that don't cross hatched areas

A lane change (`MapDrive.changes`: `(s0, s1, side, shift)` on the path, planned from nav's lane keys) must not sweep
the car's body across a hatched area: in real life you don't cross a gore's chevrons to take or leave an exit, or a
painted median. Make the map driver read the hatched areas (from `lanes.json` beside the map's pbf, as the overlay does)
and plan changes so the car's swept body (`half_width`, the quintic move across) stays out of them, by starting the
change earlier (before the gore's nose), or later (after a painted island ends), keeping the existing least-time and
room rules (`_lc_least`, `_deferred_keys`). Where no legal window exists before the route needs the lane (the map or
route leaves none), keep the change and flag it with an anomaly event rather than failing the trip, so Amy can review
the spot. Also: the arrow-key lane changes Amy asks for (`ask_lane` / `_ask_check`) should refuse with a new reason
when the move would cross one (`gta5_events.py` has the HUD words; add one, e.g. "No {side} here: hatched").
Tests: made routes with a gore (a diverge with a hatched V between the through lanes and the exit), checking the
change starts before the nose and the swept body never overlaps the ring; and 3-5 real routes on live19b past motorway
gores, e.g. near (-705,-1988), (-754,-2147), (-624,-2102), and the run (724,-585), (623,-534), (568,-326), (327,-483).
Report per route: changes moved, how far, any flagged.

### 2. Waits for a lane gap that ignore clear gaps

Amy, watching in game today: the car keeps "waiting for a gap" with a clear gap of 15 s beside it and never moves
over. The brain log has waits of 121 s (38 deferrals) and 137 s (4 deferrals); first reasons `behind`, `closing`,
`ahead`, `beside`. How it works now: `_lc_gate` checks the next change from `LC_GATE_T` s before it; while
`_lane_busy` (the blind-spot zone, `gta5_blindspot.lane_occupied`) or `_lane_window` (predicted motion,
`_predict(..., speed_up=True)`, over the change and `LC_AFTER_S` after) says busy, the change is put off
`LC_DEFER_S` (2 s) on and re-checked as it comes up again. Suspects, to confirm or rule out with the simulator and
unit tests rather than guess: the speed-up prediction making a far car behind look like it closes; cars two lanes over,
on the other carriageway or across a median (`off_max`) counted; parked or static vehicles counted (`driven` unknown);
the `lw["why"]` and `clear` bookkeeping; the deferral moving the change ahead faster than the gate can see a gap.
Reproduce with `mapdrive_sim` traffic: a clear adjacent lane, with traffic only in lanes that shouldn't matter; a gap
of 3, 6 and 15 s in dense adjacent traffic; a car far behind closing slowly. The car must take a clear gap within about
a second of it opening (after `LC_CLEAR_S`), and still refuse real conflicts (keep `test_blindspot.py` and the existing
lane-gap tests passing). Report the root cause(s) with evidence.

### 3. Red lights: a held red released by a car turning on red

The plugin can't read the lamps, so `MapDrive._light` infers the light from traffic: red from a car of ours waiting at
the line or cross traffic going straight through; green from a car of ours crossing the line faster than
`LIGHT_THROUGH`, which wins over red. Where the route goes straight on, a car in any lane of ours counts
(`own_green = straight or ...`). In game today: `light_stop` (held, a car waiting), 7 s later `light_go` "through", and
4 s after that the car crossed the line with the light seen red: most likely a car in the lane beside turning right on
red, or one running the light. Change the rules so: only a car going straight through (heading kept, not yawing into a
turn: the plugin's `yaw`, `LIGHT_STRAIGHT`) counts as green; once held at a red, green needs a car in our own lane, or
two cars, or evidence lasting about a second, not one glimpse; and leave the late-red rule (a red first seen within
comfortable stopping distance is driven through) as it is. Unit tests on synthetic `cars` lists around a light: a
right-on-red turner beside us (stays red), our lane's queue moving off (goes), cross traffic stopping then ours moving
(goes), a single red-runner in the next lane (stays red).

## Constraints

- Keep the GTA work inside `tools/sim/bridge/gta5/`. Don't change behaviour outside these three topics; Amy reviews
  the diff before it goes near `gta5`.
- No regressions against your preflight baseline; run the four test files at the end and list the results.
- Comments: one line or none, the why not the what, matching the file's style (its docstrings are dense prose).
- Branch **`mapdrive-cloud`** only. Never push to `gta5` or `master`.
- Commit messages end with `Assisted-by: Claude Code`; no Co-Authored-By, session-link or Signed-off-by trailers.
- Don't modify the data pack. No secrets in commits.

## Outputs

- Code and tests on `mapdrive-cloud`.
- `CLOUD_RESULT_MAPDRIVE.md`: a short plain-English summary for Amy at the top, then preflight, findings per item
  (root causes with evidence, what changed, test results per route), deviations, and **an in-game check list** for a
  local session: the exact spots and what to watch (gores at the coordinates above, a lane change into a gap with
  traffic, a light with a right-turn lane).

## Stopping rule

Work through the items in order; after each, commit, push `mapdrive-cloud` and update the result file. If one is
blocked, record why, push, and continue with the next. Do not repeat a failed approach more than twice.
