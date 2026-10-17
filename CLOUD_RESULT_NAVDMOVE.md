# Cloud result: the GTA bridge's nav planner moves out of `selfdrive/` (no behaviour change)

Brief: `CLOUD_TASK_NAVDMOVE.md`. Branch `navdmove-cloud` (= `gta5` e55cb3b45 + the brief 8d4b72cf9).

## Preflight

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Repo | PASS | `navdmove-cloud` = `8d4b72cf9` (the brief), parent `gta5` = `e55cb3b45`; `gta5-distill` `cloud-data` = `1c75209` |
| 2 | Data | PASS | sha256 and size of the stored files match `cloud/MANIFEST.md` for `live18/map1/paths.jsonl.gz`, `live18/map1/nodes.npz` and all 5 `live21b/*` (pbf `ef13e870…`, lanes `ca32feb3…`, roads `46697b4f…`, lanegeom `ec5c3a79…`, build.log `02e654f4…`) |
| 3 | Map | PASS | `/home/user/map`: live21b's `gta5.osm.pbf`, `lanes.json` (41376903 B), `roads.json` (14889877 B) + live18's `paths.jsonl` (18855947 B), `nodes.npz`; `GTA5_MAP=/home/user/map`, also linked as `~/gta5map_lanes` and `~/gta5map` |
| 4 | Dependencies | PASS (native params missing, as before) | Python 3.12 venv (uv), PyPI only: numpy, osmium, pyvalhalla 3.9.1, shapely, scipy, pytest, pycapnp 2.1.0, pyzmq, opendbc 0.3.1, zstandard, setproctitle, requests, pillow, pyjwt, ruff 0.16.7 (`uv.lock`'s). `msgq` built from its public source (`commaai/msgq` 326a9f5, `pip install .`, outside the repo). Not available: native `params` (`libparams_c.so`), so comma's `selfdrive/navigation/tests/test_navigation.py` can't be collected (left out of every run); importing `opendbc.car` beside `openpilot.cereal` aborts (capnp "Duplicate ID" for `car.capnp`), so `test_blindspot.py`, `test_driver.py`, `test_world.py` and `gta5_world.py` can't import here, on `gta5` as on the branch |
| 5 | Valhalla | PASS | `valhalla_build_config` + `valhalla_build_tiles` on live21b's pbf: 13 s, 17 MB; `valhalla_service valhalla.json 2` on :8002, `/status` = 3.9.1 |
| 6 | Baseline tests | 502 pass, 1 fail | On the unmoved tree: `selfdrive/navigation/tests` (without `test_navigation.py`), `selfdrive/navd/tests`, `test_nav.py`, `test_nav_msgs.py`, `test_lane_slots.py`, `test_lane_numbering.py`, `test_navd_sim.py`, `test_mapbox_server.py`, `test_navd_path.py`, `test_nav_approval.py`, `test_nav_request.py`, `tools/sim/bridge/gta5/map`: 500 passed, 3 failed in 533 s; 2 of the 3 (`test_lane_guidance.py::test_navd_runs_it`, `test_mapbox_server.py::test_navd_uses_the_host`) only lacked `jwt`, and pass with PyJWT installed. The one left is `map/test_cruise.py::test_live_cruises` (a wrong-way at (-224, 79) on live21b), a known-red baseline the move must leave as it is |
| 7 | navd_sim baseline | PASS | `navd_sim --nav map,mapbox --model n16 --lane truth,model:0.85` (seed 0, `--sim-seed 0`, 46 trips, 302 maneuvers that need a lane), unmoved tree: map n16 / truth 247, map n16 / model:0.85 250, mapbox n16 / truth 247, mapbox n16 / model:0.85 244, as on Amy's PC; 10 min on 3 jobs. The result JSON (1682608 B) and every nav request (`--requests`, 593651 B) kept for the comparison |
| 8 | Push access | PASS | this file, committed and pushed to `navdmove-cloud` |

Test command (repo root, venv active, Valhalla up; after the move `selfdrive/navd/tests` is `tools/sim/bridge/gta5/navd/tests`):

```
export PYTHONPATH=$PWD GTA5_MAP=/home/user/map
G=openpilot/tools/sim/bridge/gta5
python -m pytest -q openpilot/selfdrive/navigation/tests --ignore=openpilot/selfdrive/navigation/tests/test_navigation.py \
  openpilot/selfdrive/navd/tests $G/test_nav.py $G/test_nav_msgs.py $G/test_lane_slots.py $G/test_lane_numbering.py \
  $G/test_navd_sim.py $G/test_mapbox_server.py $G/test_navd_path.py openpilot/selfdrive/controls/tests/test_nav_approval.py \
  openpilot/selfdrive/ui/tests/test_nav_request.py $G/map
python -m openpilot.tools.sim.bridge.gta5.navd_sim --nav map,mapbox --model n16 --lane truth,model:0.85 --sim-seed 0 \
  --jobs 3 --out sim.json --requests req.jsonl
```
