# Cloud result: lane guidance follow-ups (lanes that end, bus lanes, forks without arrows, no slow lane changes)

Brief: `CLOUD_TASK_GUIDANCE2.md`. Branch `guidance2-cloud`, one commit per item.

## Preflight

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Repo | PASS | `guidance2-cloud` = `39aa520` (the brief), parent `e55cb3b` (`gta5`); shallow, single branch |
| 2 | Data | PASS | `gta5-distill` `cloud-data` = `1c75209`; sha256 and size match `cloud/MANIFEST.md` for `live18/map1/{paths.jsonl,nodes.npz}*` and all 5 `live21b/*`. Map assembled in `/home/user/map`: live21b's `gta5.osm.pbf`, `lanes.json`, `roads.json` + live18's `paths.jsonl`, `nodes.npz`; `GTA5_MAP=/home/user/map`, also linked as `~/gta5map_lanes` and `~/gta5map` |
| 3 | Dependencies | PASS | Python 3.12 venv (uv), PyPI: numpy, osmium, pyvalhalla 3.9.1, shapely, scipy, pytest 9.1, pycapnp 2.1.0, pyzmq, opendbc 0.3.1, zstandard, setproctitle, requests, pyjwt. `msgq` built from its public source at the submodule's commit (`commaai/msgq` `0e266c1`, `scons --minimal`, `pip install .`): `cereal.messaging` round-trips a real message. New this time: native `params` built too (apt `capnproto`/`libcapnp-dev`/`libzmq3-dev`, cereal's C++ headers generated with `capnpc` into the git-ignored `cereal/gen/cpp`, json11 from its public repo, `libparams_c.so` linked by hand), so comma's `selfdrive/navigation/tests/test_navigation.py` runs here (20 pass) |
| 4 | Valhalla | PASS | `valhalla_build_config` + `valhalla_build_tiles` on live21b's pbf: 9 s, 17 MB; `valhalla_service valhalla.json 2` on :8002, `/status` = 3.9.1 |
| 5 | Baseline tests | PASS: 233 passed, 0 failed (270 s) | the brief's set: `selfdrive/navigation/tests` (with `test_navigation.py`), `test_mapbox_server.py`, `test_navd_path.py`, `test_nav.py`, `test_nav_msgs.py`, `test_lane_slots.py`, `test_lane_numbering.py`, `test_navd_sim.py`, `selfdrive/navd/tests`, `test_nav_approval.py`, `test_nav_request.py`. `MAPBOX_TOKEN` is set in this cloud environment and makes navd pick `api.mapbox.com` (one `test_navigation.py` test then fails on the host); the brief needs no token, so every run here has it unset |
| 6 | navd_sim baseline | PASS (= Amy's PC) | `--nav map,mapbox --model n16,finetuned --lane truth,model:0.85`, seed 0, 46 trips, 302 maneuvers that need a lane: n16/truth map 247, mapbox 247; n16/0.85 250 / 244; finetuned/truth 262 / 254; finetuned/0.85 255 / 252. 12.7 min on 3 workers |
| 7 | Push access | PASS | this file, committed and pushed to `guidance2-cloud` |

Test command (repo root, venv active, Valhalla up, `MAPBOX_TOKEN` unset):

```
export PYTHONPATH=$PWD GTA5_MAP=/home/user/map
G=openpilot/tools/sim/bridge/gta5
python -m pytest -q openpilot/selfdrive/navigation/tests $G/test_mapbox_server.py $G/test_navd_path.py $G/test_nav.py \
  $G/test_nav_msgs.py $G/test_lane_slots.py $G/test_lane_numbering.py $G/test_navd_sim.py openpilot/selfdrive/navd/tests \
  openpilot/selfdrive/controls/tests/test_nav_approval.py openpilot/selfdrive/ui/tests/test_nav_request.py
```

The navd_sim baseline (seed 0), for the items below:

| run | in lane | turns | forks / exits | straight on | LC req / done / gave up / failed | LC late (asked < v_ok) | hanging s | forced merges |
|---|---|---|---|---|---|---|---|---|
| n16 / truth | 247 | 68 | 95 | 84 | 181/120/8/53 | 85 | 27 | 7 |
| n16 / model:0.85 | 250 | 72 | 97 | 81 | 181/124/5/52 | 78 | 35 | 6 |
| finetuned / truth | 262 | 80 | 99 | 83 | 219/148/8/63 | 8 | 53 | 4 |
| finetuned / model:0.85 | 255 | 74 | 99 | 82 | 209/139/4/66 | 5 | 51 | 8 |
| mapbox / n16 / truth | 247 | 77 | 91 | 79 | 159/106/7/46 | 0 | 50 | 20 |
| mapbox / n16 / model:0.85 | 244 | 74 | 91 | 79 | 162/102/9/51 | 0 | 58 | 22 |
| mapbox / finetuned / truth | 254 | 76 | 98 | 80 | 232/154/8/70 | 0 | 75 | 22 |
| mapbox / finetuned / model:0.85 | 252 | 75 | 96 | 81 | 238/146/10/82 | 0 | 96 | 22 |
