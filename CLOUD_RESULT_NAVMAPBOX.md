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
