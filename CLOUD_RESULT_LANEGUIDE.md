# Cloud result: lane guidance (navd planner) for comma's nav models

Brief: `CLOUD_TASK_LANEGUIDE.md`. Branch `laneguide-cloud`, one commit per step.

## Preflight

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Repo | PASS | `laneguide-cloud` = `885f569` (the brief), parent `gta5` = `84739d1` ("cereal: a backlog read in one update has no rate yet"); fetched `--depth 2`, single branch |
| 2 | Data | PASS | `gta5-distill` `cloud-data` = `891a36f`; sha256 and size match `cloud/MANIFEST.md` for all 11 `live18/map1/*` and `live19b/*` files. Live map assembled in `/home/user/map`: live18's `gta5.osm.pbf`, `roads.json`, `paths.jsonl`, `nodes.npz` + live19b's `lanes.json` (69666 lines, 1451 stop lines); `GTA5_MAP=/home/user/map` (also symlinked as `~/gta5map_lanes`, `~/gta5map`) |
| 3 | Dependencies | PASS (with a shim) | Python 3.12 venv (uv), from PyPI: numpy 2.5, osmium 4.3.1, pyvalhalla 3.9.1, shapely, pytest, pycapnp 2.1.0, pyzmq, scipy, comma's `opendbc` 0.3.1 (for cereal's `car.capnp`). Not available: openpilot's native `msgq` (the PyPI `msgq` is an unrelated SysV package, removed) and `params` (C++). `cereal.messaging` imports msgq, so `test_nav_msgs.py` (and anything importing `openpilot.cereal.messaging`) needs it; a socket-free stand-in outside the repo (`/home/user/shim/msgq`, on `PYTHONPATH`) lets the capnp-only paths (`new_message`, `log_from_bytes`) run. `DesireHelper` needs `Params` (native) and isn't importable here |
| 4 | Valhalla | PASS | `valhalla_build_config ... > valhalla.json`; `valhalla_build_tiles` (site-packages/valhalla/bin) on the pbf: 10 s, 17 MB of tiles; `valhalla_service valhalla.json 2` on :8002, `/status` answers (3.9.1) |
| 5 | Baseline tests | 112 pass, 1 fail | `test_nav.py`, `test_lane_slots.py`, `test_lane_numbering.py`, `selfdrive/navd/tests`: 96 passed, 1 failed in 263 s. `test_nav_msgs.py` (with the msgq shim): 16 passed. The failure: `test_lane_numbering.py::test_live_freeway_merge_exit_and_splits`, the Dutch London St ramp split (-741.9, -1838.4) routes 3063 m instead of < 300 m (Valhalla on these tiles takes a long way round from that pose); pre-existing, not touched by this work |
| 6 | Push access | PASS | this file, committed and pushed to `laneguide-cloud` |

Test command (from the repo root, venv active, Valhalla up):

```
export PYTHONPATH=$PWD:/home/user/shim GTA5_MAP=/home/user/map
G=openpilot/tools/sim/bridge/gta5
python -m pytest -q $G/test_nav.py $G/test_nav_msgs.py $G/test_lane_slots.py $G/test_lane_numbering.py openpilot/selfdrive/navd/tests
```
