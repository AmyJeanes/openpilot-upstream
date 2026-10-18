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
