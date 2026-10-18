# Cloud result: lane guidance follow-ups 3 (a buffer at the lane change speed, the turn gap, wrong-side lane-end asks, chores)

Brief: `CLOUD_TASK_GUIDANCE3.md`. Branch `guidance3-cloud`, one commit per item.

## Preflight

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Repo | PASS | `guidance3-cloud` = `079f1f1` (the brief), parent `2658f3197` (`gta5`); shallow (depth 1), single branch. `gta5-distill` `cloud-data` = `1c75209` |
| 2 | Data | PASS | sha256 and size match `cloud/MANIFEST.md` for `live18/map1/paths.jsonl.gz` (`1e5a61ab…`), `live18/map1/nodes.npz` (`0b71e974…`) and all 5 `live21b/*` (pbf `ef13e870…`, lanes `ca32feb3…`, roads `46697b4f…`, lanegeom `ec5c3a79…`, build.log `02e654f4…`). Map assembled in `/home/user/map`: live21b's `gta5.osm.pbf`, `lanes.json` (41376903 B), `roads.json` (14889877 B) + live18's `paths.jsonl` (18855947 B), `nodes.npz`; `GTA5_MAP=/home/user/map`, also linked as `~/gta5map_lanes` and `~/gta5map` |
| 3 | Dependencies | PASS | Python 3.12 venv (uv), PyPI: numpy, osmium, pyvalhalla 3.9.1, shapely, scipy, pytest, pycapnp 2.1.0, pyzmq, zstandard, setproctitle, requests, pyjwt, pillow, tqdm, pycryptodome, ruff 0.16.7 (`uv.lock`'s). `msgq` built from its public source at the submodule's commit (`commaai/msgq` `0e266c1`, `scons --minimal`, `pip install .`): a `carState` round-trips through PubMaster/SubMaster. Native params as guidance2 did (apt `capnproto` 1.0.1, `libcapnp-dev`, `libzmq3-dev`; `capnpc` into the git-ignored `cereal/gen/cpp`; json11 `2df9473`; `g++ -shared` of `params_c.cc params.cc swaglog.cc util.cc` + the schemas into the git-ignored `common/libparams_c.so`): `Params().put_bool/get_bool` round-trips, so `test_navigation.py` runs (20 pass). **opendbc from the submodule's commit** (`commaai/opendbc` `b3bd79a`, `pip install` of its source) instead of PyPI's 0.3.1: see "Importing test_blindspot.py" below |
| 4 | Valhalla | PASS | `valhalla_build_config` + `valhalla_build_tiles` on live21b's pbf: 10 s, 17 MB; `valhalla_service valhalla.json 2` on :8002, `/status` = 3.9.1 |
| 5 | Baseline tests | PASS: 314 passed, 0 failed (295 s) | guidance2's set with the moved path, plus `test_route_input.py` and the three files that now import: `test_navigation.py` 20, `test_lane_guidance.py` 28, `test_mapbox_fixtures.py` 39, `navd/tests` 18, `test_mapbox_server.py` 10, `test_navd_path.py` 1, `test_nav.py` 47, `test_nav_msgs.py` 17, `test_lane_slots.py` 20, `test_lane_numbering.py` 30, `test_navd_sim.py` 9, `test_nav_approval.py` 5, `test_nav_request.py` 3, `test_route_input.py` 9, **`test_blindspot.py` 35, `test_driver.py` 15, `test_world.py` 8**. `MAPBOX_TOKEN` is set in this cloud environment; every run here has it unset (with it set, `test_request_and_instruction` fails: `'https://api.mapbox.com/directions/v5/…' != 'https://api.commadotai.com/v1/nav/directions/v5/…'`, item 4) |
| 6 | navd_sim baseline | PASS (= guidance2's results at item 4) | `--nav map,mapbox --model n16,finetuned --lane truth,model:0.85` at `079f1f1` (= `2658f3197`'s code), `--sim-seed 0` and `1`, 46 trips, 302 maneuvers that need a lane, 0 trip errors; 13.5 / 13.3 min on 3 jobs. Seed 0 map 242 / 245 / 264 / 255, Mapbox 247 / 244 / 254 / 252; seed 1 map 248 / 243 / 255 / 254, Mapbox 247 / 244 / 256 / 254 (n16 truth / 0.85, finetuned truth / 0.85): every column of both tables below is the same as `CLOUD_RESULT_GUIDANCE2.md`'s "after" (item 4) and Mapbox baseline |
| 7 | Push access | PASS | this file, committed and pushed to `guidance3-cloud` |

Test command (repo root, venv active, Valhalla up, `MAPBOX_TOKEN` unset):

```
unset MAPBOX_TOKEN
export PYTHONPATH=$PWD GTA5_MAP=/home/user/map
G=openpilot/tools/sim/bridge/gta5
python -m pytest -q openpilot/selfdrive/navigation/tests $G/navd/tests $G/test_mapbox_server.py $G/test_navd_path.py \
  $G/test_nav.py $G/test_nav_msgs.py $G/test_lane_slots.py $G/test_lane_numbering.py $G/test_navd_sim.py \
  openpilot/selfdrive/controls/tests/test_nav_approval.py openpilot/selfdrive/ui/tests/test_nav_request.py \
  openpilot/selfdrive/modeld/test_route_input.py $G/test_blindspot.py $G/test_driver.py $G/test_world.py
```

### Importing test_blindspot.py, test_driver.py and test_world.py

The abort ("Duplicate ID @0x8e2af1e708af8b8d" for `car.capnp`, or for `include/c++.capnp` the other way round) is
PyPI's opendbc 0.3.1, not the repo: its `opendbc/car/structs.py` loads `car.capnp` with no import path (after trying
`from cereal import car`, which this tree doesn't have: it's `openpilot.cereal`), while `openpilot/cereal/__init__.py`
loads `log.capnp` with opendbc's `car/` as an import path, so pycapnp parses the same file under two names. The
submodule's own opendbc (`opendbc_repo` at `b3bd79a`, what Amy's tree builds) loads it as
`capnp.load(car.capnp, imports=[BASEDIR])`, the same way `log.capnp` reaches it, and both import orders then work.
With opendbc installed from `commaai/opendbc` at `b3bd79a` (`git clone`, `git checkout b3bd79a`, `pip install .`; its
DBCs ship in the source), the three files import and pass: 35 + 15 + 8. Nothing in the repo changed for it.

### The navd_sim baseline and how it's measured

Every navd_sim run here goes through `probe_sim.py` (in the session, not committed): navd_sim itself, with
measurement-only wrappers that add fields to each navRequest event (`end_v`, `end_t`, `end_why`: the planner's reason
for a withdrawal, `move_at`: where along the trip the move is) and per trip a `probe` list (each move's last chance for a
lane change, skips to the route, turn signals). They change nothing: on FX1 and MD029 the requests (less the added
fields), maneuvers and changes compare equal to plain navd_sim's, and the full baseline reproduces guidance2's numbers
in every column. Runs are from git worktrees of each item's commit, so later edits don't leak into a running sim.

| run | seed | in lane | turns | forks / exits | straight on | LC req / done / gave up / failed | forced merges | lane-end asks (right / other_side) |
|---|---|---|---|---|---|---|---|---|
| n16 / truth | 0 | 242 | 67 | 92 | 83 | 162/106/7/49 | 7 | 40 (36 / 4) |
| n16 / model:0.85 | 0 | 245 | 71 | 94 | 80 | 160/109/4/47 | 6 | 38 (30 / 8) |
| finetuned / truth | 0 | 264 | 81 | 99 | 84 | 214/147/8/59 | 4 | 41 (34 / 7) |
| finetuned / model:0.85 | 0 | 255 | 74 | 99 | 82 | 208/139/4/65 | 8 | 38 (30 / 8) |
| mapbox / n16 / truth | 0 | 247 | 77 | 91 | 79 | 159/106/7/46 | 20 | 0 |
| mapbox / n16 / model:0.85 | 0 | 244 | 74 | 91 | 79 | 162/102/9/51 | 22 | 0 |
| mapbox / finetuned / truth | 0 | 254 | 76 | 98 | 80 | 232/154/8/70 | 22 | 0 |
| mapbox / finetuned / model:0.85 | 0 | 252 | 75 | 96 | 81 | 238/146/10/82 | 22 | 0 |
| n16 / truth | 1 | 248 | 75 | 91 | 82 | 156/115/6/35 | 7 | 34 (33 / 1) |
| n16 / model:0.85 | 1 | 243 | 67 | 94 | 82 | 159/112/5/42 | 7 | 32 (27 / 5) |
| finetuned / truth | 1 | 255 | 76 | 96 | 83 | 222/145/21/56 | 5 | 44 (37 / 7) |
| finetuned / model:0.85 | 1 | 254 | 76 | 97 | 81 | 217/143/8/66 | 6 | 39 (33 / 6) |
| mapbox / n16 / truth | 1 | 247 | 80 | 88 | 79 | 160/102/2/56 | 18 | 0 |
| mapbox / n16 / model:0.85 | 1 | 244 | 77 | 88 | 79 | 169/108/2/59 | 19 | 0 |
| mapbox / finetuned / truth | 1 | 256 | 83 | 94 | 79 | 228/147/1/80 | 21 | 0 |
| mapbox / finetuned / model:0.85 | 1 | 254 | 79 | 95 | 80 | 243/151/2/90 | 20 | 0 |

Reproduce (repo root, venv, `GTA5_MAP` set, `MAPBOX_TOKEN` unset):

```
python -m openpilot.tools.sim.bridge.gta5.navd_sim --nav map,mapbox --model n16,finetuned --lane truth,model:0.85 \
  --sim-seed 0 --jobs 3 --out r.json --requests req.jsonl
```

## Item 1: a buffer at the lane change speed

**What changed**: a change is still asked for only at or above the model's lane change speed, but one asked and not
started yet is withdrawn only once the car is a margin below it.
- Bridge planner (`tools/sim/bridge/gta5/navd/planner.py`): Tune `model_lc_margin` (1.0 m/s). `_lc_too_slow(v, margin)`;
  every ask still uses it without a margin (`_change_for`'s `ready`, `_enter_bay`, `_keep_right`), the withdrawal of a
  change not started (`change_went` false) uses `model_lc_margin`. lc_wait's own withdrawal (below 0.8 x 10 m/s, a
  queue) is unchanged.
- comma navd's lane guidance (`selfdrive/navigation/lane_guidance.py`): Tune `lc_speed_margin` (1.0 m/s); `_change`
  asks at `v >= lc_min_speed` as before, `_watch` withdraws (`gave_up`) below `lc_min_speed - lc_speed_margin`.

For n16 (10 m/s) a change asked at 10 m/s or more stays asked down to 9 m/s; for the finetuned model (4 m/s), down to 3.

**Tests**: `test_nav.py::test_hovering_at_the_lane_change_speed_asks_once` (asked at 10.4 m/s and held by the blind spot,
the car hovering 9.3-10.4 m/s for 8 s: one request, nothing withdrawn; at 8.9 it's withdrawn; `model_lc_margin` 0: asked,
withdrawn, asked again) and `test_lane_guidance.py::test_hovering_at_the_lane_change_speed_asks_once` (never asked at 9.9;
asked at 10.4, hovering 9.3-10.4 for 8 s: one request; withdrawn at 8.9; `lc_speed_margin` 0: two withdrawals or more).
The guidance2 tests (`test_change_not_started_withdrawn_when_slowing` at 8 m/s, `test_withdrawn_when_too_slow` at 7) pass
unchanged.

**Measured**: "too slow / again" counts the requests withdrawn as too slow (the planner's own reason; on the Mapbox path
every `gave_up`) and of those, the ones asked again for the same move (same side, within 25 m) within 10 s.

| run | seed | too slow / again: baseline -> item 1 | in lane | LC req / done / gave up / failed | LC late (blinker < v_ok) | forced merges |
|---|---|---|---|---|---|---|
| n16 / truth | 0 | 4 / 1 -> 1 / 0 | 242 -> 242 | 162/106/7/49 -> 163/107/7/49 | 0 -> 2 | 7 -> 7 |
| n16 / model:0.85 | 0 | 3 / 1 -> 0 / 0 | 245 -> 245 | 160/109/4/47 -> 160/110/4/46 | 0 -> 1 | 6 -> 6 |
| finetuned / truth | 0 | 2 / 1 -> 0 / 0 | 264 -> 264 | 214/147/8/59 -> 215/147/8/60 | 0 -> 2 | 4 -> 4 |
| finetuned / model:0.85 | 0 | 0 / 0 -> 0 / 0 | 255 -> 255 | 208/139/4/65 -> 208/139/4/65 | 0 -> 0 | 8 -> 8 |
| mapbox / n16 / truth | 0 | 9 / 0 -> 5 / 0 | 247 -> 248 | 159/106/7/46 -> 163/107/7/49 | 0 -> 4 | 20 -> 20 |
| mapbox / n16 / model:0.85 | 0 | 12 / 0 -> 7 / 0 | 244 -> 244 | 162/102/9/51 -> 168/104/10/54 | 0 -> 5 | 22 -> 22 |
| mapbox / finetuned / truth | 0 | 3 / 1 -> 3 / 1 | 254 -> 254 | 232/154/8/70 -> same | 0 -> 0 | 22 -> 22 |
| mapbox / finetuned / model:0.85 | 0 | 3 / 1 -> 3 / 1 | 252 -> 252 | 238/146/10/82 -> same | 0 -> 0 | 22 -> 22 |
| n16 / truth | 1 | 5 / 2 -> 3 / 1 | 248 -> 248 | 156/115/6/35 -> same | 0 -> 1 | 7 -> 7 |
| n16 / model:0.85 | 1 | 6 / 1 -> 2 / 0 | 243 -> 242 | 159/112/5/42 -> 161/112/5/44 | 0 -> 3 | 7 -> 7 |
| finetuned / truth | 1 | 4 / 2 -> 0 / 0 | 255 -> 255 | 222/145/21/56 -> 224/145/21/58 | 0 -> 4 | 5 -> 5 |
| finetuned / model:0.85 | 1 | 1 / 1 -> 0 / 0 | 254 -> 254 | 217/143/8/66 -> same | 0 -> 1 | 6 -> 6 |
| mapbox / n16 / truth | 1 | 4 / 1 -> 1 / 0 | 247 -> 247 | 160/102/2/56 -> 161/102/1/58 | 0 -> 2 | 18 -> 18 |
| mapbox / n16 / model:0.85 | 1 | 5 / 1 -> 2 / 0 | 244 -> 244 | 169/108/2/59 -> 170/108/1/61 | 0 -> 2 | 19 -> 19 |
| mapbox / finetuned / truth | 1 | 1 / 0 -> 0 / 0 | 256 -> 256 | 228/147/1/80 -> same | 0 -> 0 | 21 -> 21 |
| mapbox / finetuned / model:0.85 | 1 | 0 / 0 -> 0 / 0 | 254 -> 254 | 243/151/2/90 -> same | 0 -> 0 | 20 -> 20 |

(turns, forks / exits and straight on are within one of the baseline in every run, and the same in most.)

- **The on-off is gone**: asked again within 10 s, 13 -> 3 over both seeds and all runs. All 13 in the baseline were
  withdrawn within 0.1 m/s of the threshold (9.96-9.99 m/s for n16, 3.91-3.95 for the finetuned model): FX1's fork at
  (870.0, -1764.6) on every n16 map run (asked at 10.0 m/s, withdrawn at 9.96 0.6 s later, asked again), R06's exit,
  R34's turn on the Mapbox path, R23 and R36 for the finetuned model. The 3 left are real slowdowns and recoveries:
  R06's exit on seed 1 (withdrawn at 8.9 m/s after 9 s slowing from 19.5 in traffic, asked again at 10.9) and the
  finetuned model's R23 turn on the Mapbox path (down to 2.9 m/s, asked again at 4.0).
- **What it costs**: changes asked at 10 m/s or more whose blinker comes on (PARAM_LEAD, 0.4 s on) or which start at
  9-10 m/s are no longer withdrawn: navd_sim's "LC late", counted at the blinker, goes 0 -> 1-5 a run, every one asked at
  10.0-11.1 m/s and started at 8.9-9.9 m/s (3.7-3.8 for the finetuned model). The stand-in moves slower there and several
  don't get across (LC failed +0-3 a run). In-lane is the same within one in every run (n16 +1 on the Mapbox path seed 0,
  -1 with the noisy head seed 1). Whether n16 changes lanes properly at 9-10 m/s in game is check 1 below; if not, a
  smaller margin (0.5) halves the window.

## Item 2: the turn gap between ~8.5 and 10 m/s

**Measured first** (the baseline, n16; the finetuned model has no gap: its lane change speed, 4 m/s, is below the
turn signal's). A move's "last chance" is the first step a lane change for it can't be asked any more: on the bridge
planner past its fast point (`_fast_room`) and slower than `model_lc_speed`, or past its last place; on the Mapbox path
past its deadline and slower than `lc_min_speed`, or its turn signal. "In the band": the car was then between the turn
signal's speed (19 mph, 8.49 m/s) and 10 m/s, outside the turn's lanes, with no change under way. What came of each,
by the probe's later events and the judge:

| run | seed | turns in the band | signalled from beside (turn_into) | skipped to the route | signalled from the wrong lane | nothing from nav | signalled once in its lanes | made |
|---|---|---|---|---|---|---|---|---|
| n16 / truth | 0 | 6 | 1 | 4 | 0 | 0 | 1 | 1 |
| n16 / model:0.85 | 0 | 6 | 2 | 0 | 4 | 0 | 0 | 1 |
| mapbox / n16 / truth | 0 | 27 | 4 | - | 3 | 14 | 6 | 10 |
| mapbox / n16 / model:0.85 | 0 | 31 | 6 | - | 6 | 15 | 4 | 13 |
| n16 / truth | 1 | 4 | 1 | 3 | 0 | 0 | 0 | 1 |
| n16 / model:0.85 | 1 | 6 | 2 | 1 | 3 | 0 | 0 | 2 |
| mapbox / n16 / truth | 1 | 23 | 3 | - | 1 | 13 | 6 | 13 |
| mapbox / n16 / model:0.85 | 1 | 16 | 2 | - | 1 | 10 | 3 | 8 |

(On the Mapbox path "nothing from nav" is mostly made anyway, 10-11 a run, by the car being in a lane for it at the
commit point after all, or a Mapbox turn the judge has no turn for; "made" counts the judged ones.)

Why the bridge planner's band cases fail, from the planner's own state at each skip (MDT, R07, R10, R21 on seed 0):
- **R21's left turn at (76.8, -155.7)** is the gap as the brief has it: one lane beside the turn's lane, still at 9.35
  m/s when it reaches the turn's signal distance (27 m out), so neither a change (too slow) nor the turn's signal (too
  fast): left to the route. A moment later it was below 8.49 and `turn_into_lane` would have signalled it.
- **MDT (93.0, -1357.1) and R07 (793.7, -1146.7)**: left to the route at 6.4 and 6.9 m/s, one lane beside, well inside
  80 m: slow enough for `turn_into_lane`, which refused because `change_went` was still true. `change_went` ("openpilot
  could start it") is set for the change under way and never cleared when it ends; `_turn_into` reads it as "a change is
  under way", so after the first lane change of a trip that got going, every later turn from beside was left to the
  route. That's not only the band: it's every turn_into_lane case after any started change.
- **R10 (603.8, -372.7)**: a change towards the turn started late (8.8 m/s), still under way 25 m out, given up there.
- The Mapbox path has no skip; in the band it signals the turn at `signal_min` (28 m) whatever the speed, so from beside
  the blinker comes on at 8.4-9.97 m/s (R29, R33, R40, R01...): no turn desire below 20 mph and above 19, a lane change
  desire above 20 mph (desire_helper), the hazard guidance2's item 4 removed.

**What changed**:
- Bridge planner: `_turn_into` refuses only while a change is under way (`changing is not None and change_went`), not
  after one has ended. New `_turn_gap` (Tune `turn_gap`, on): one lane beside the turn's lanes on its side, between
  `min(model_lc_speed, 19 mph)` and `model_lc_speed`, no change open, more than SIGNAL_LAST (8 m) from the turn: where
  `_signal_due` would leave the turn to the route, it keeps it (returns False, nothing signalled), so `turn_into_lane`
  signals it once the car has slowed below 19 mph.
- Lane guidance: `_turn_gap` (Tune `turn_gap`, on): where `turn_into` would take the car over (one lane beside on the
  turn's side, or in place before its bay opens, within `turn_into` m) but the car is between the lane turn speed and
  `lc_min_speed`, with no change open, the turn's signal waits until it's slow enough for `turn_into`; no blinker comes
  on in the band from there (none at all if it never slows: the car goes on and the route recalculates).
- Neither signals a turn above 19 mph that it didn't before; the in-lane signal at `signal_min` is unchanged.

**Tests**: `test_nav.py::test_turn_gap_keeps_the_turn_for_its_signal` (one lane beside a right turn at 9.3 m/s past its
last place for a change: no change, no blinker, not left to the route; at 7 m/s its signal; `turn_gap` off: left to
the route), `test_nav.py::test_turn_into_after_a_change_that_came_back` (a change towards the turn started, then the
stalk off with the car still beside: slower, the turn's signal from beside, not the route; fails without the
`_turn_into` fix), `test_lane_guidance.py::test_turn_gap_no_blinker_from_beside_until_slow` (no request at 9.2 m/s from
22 m out, the turn's signal at 7; `turn_gap` off: signalled at 9.2).

**Measured** (item 1 + item 2 against item 1):

| run | seed | in lane | turns | forks / exits | straight on | LC req / done / gave up / failed | forced merges | left to the route | contacts |
|---|---|---|---|---|---|---|---|---|---|
| n16 / truth | 0 | 242 -> 257 | 67 -> 83 | 92 -> 92 | 83 -> 82 | 163/107/7/49 -> 162/108/10/44 | 7 -> 8 | 53 -> 29 | 92 -> 86 |
| n16 / model:0.85 | 0 | 245 -> 255 | 71 -> 80 | 94 -> 94 | 80 -> 81 | 160/110/4/46 -> 163/111/7/45 | 6 -> 7 | 30 -> 13 | 99 -> 86 |
| finetuned / truth | 0 | 264 -> 269 | 81 -> 86 | 99 -> 99 | 84 -> 84 | 215/147/8/60 -> 215/148/8/59 | 4 -> 4 | 32 -> 27 | 93 -> 103 |
| finetuned / model:0.85 | 0 | 255 -> 258 | 74 -> 77 | 99 -> 99 | 82 -> 82 | 208/139/4/65 -> 207/140/4/63 | 8 -> 8 | 29 -> 21 | 69 -> 74 |
| mapbox / n16 / truth | 0 | 248 -> 247 | 78 -> 77 | 91 -> 91 | 79 -> 79 | 163/107/7/49 -> 164/107/7/50 | 20 -> 20 | - | |
| mapbox / n16 / model:0.85 | 0 | 244 -> 245 | 74 -> 74 | 91 -> 92 | 79 -> 79 | 168/104/10/54 -> 169/106/10/53 | 22 -> 22 | - | |
| mapbox / finetuned (both) | 0 | 254, 252: same | same | same | same | same | same | - | |
| n16 / truth | 1 | 248 -> 256 | 75 -> 83 | 91 -> 91 | 82 -> 82 | 156/115/6/35 -> 156/115/8/33 | 7 -> 8 | 42 -> 24 | 106 -> 81 |
| n16 / model:0.85 | 1 | 242 -> 248 | 66 -> 74 | 94 -> 92 | 82 -> 82 | 161/112/5/44 -> 164/115/7/42 | 7 -> 8 | 34 -> 13 | 59 -> 61 |
| finetuned / truth | 1 | 255 -> 258 | 76 -> 79 | 96 -> 96 | 83 -> 83 | 224/145/21/58 -> 218/144/17/57 | 5 -> 4 | 39 -> 35 | 96 -> 92 |
| finetuned / model:0.85 | 1 | 254 -> 255 | 76 -> 77 | 97 -> 97 | 81 -> 81 | 217/143/8/66 -> 216/142/8/66 | 6 -> 6 | 27 -> 23 | 68 -> 59 |
| mapbox / n16 / truth | 1 | 247 -> 247 | 80 -> 80 | 88 -> 88 | 79 -> 79 | 161/102/1/58 -> 161/101/1/59 | 18 -> 18 | - | |
| mapbox / n16 / model:0.85 | 1 | 244 -> 244 | 77 -> 77 | 88 -> 88 | 79 -> 79 | 170/108/1/61 -> 170/107/1/62 | 19 -> 19 | - | |
| mapbox / finetuned (both) | 1 | 256, 254: same | same | same | same | same | same | - | |

- **The bridge planner**: n16 +15 / +10 (seed 0) and +8 / +6 (seed 1) in lane, all of it turns; the finetuned model
  +5 / +3 and +3 / +1. Turns left to the route fall by a third to a half. On seed 0, n16 / truth makes 17 turns it
  missed and misses 1 it made (R40's right turn at (361.1, -271.4)); the 17 are turns signalled from one lane beside at
  1-8 m/s (MD012's at (-268.0, -1430.1), MDT's at (67.1, -1265.7), R02/R06/R40's at (603.0, -372.0), R11's at (230.4,
  -222.3), R17, R19 x2, R20, R23, R24, R25, R27 x2...), nearly all after an earlier change on the trip had set
  `change_went`: that fix, not the band, carries most of it (the split is below, "turn_gap off"). The stand-in's
  turn desire then moves the car a lane over (TURN_TAKE, 3 s): the sim credits what guidance2's probe saw n16 do (2-12 m
  of plan movement the right way); whether n16 does it in game is check 2.
- Contacts in the sim: n16 down (92 -> 86, 99 -> 86, 106 -> 81, 59 -> 61), the finetuned model mixed (93 -> 103, 69 ->
  74, 96 -> 92, 68 -> 59): the turn desire's move across goes into the gap the stand-in waits for, as a lane change does.
- Forced merges +1 in three n16 runs (7 -> 8, 6 -> 7): a turn taken changes which lane the car is in after it.
- **The Mapbox path**: within one (248 -> 247, 244 -> 245, the rest the same). In the band the blinker from beside is
  gone (seed 0: 3 -> 1 and 6 -> 1, the one left R29's at (288.5, -856.1), two lanes off; seed 1: 1 -> 0 and 1 -> 0);
  those turns are now signalled from beside once below 19 mph (seed 0: 4 -> 7 and 6 -> 13), but mostly too late for the
  judge's commit point. No turn is signalled from outside its lanes at 20 mph or more (0 / 0 on seed 0, n16).
- Band cases still left to the route on the bridge planner: MDT's left turn at (93.0, -1357.1) (the car on the wrong
  side of its lanes for the turn signal to help: lane 1 of 2 needed from lane 0 for a left turn) and R10's (a change
  under way not done 25 m out).
