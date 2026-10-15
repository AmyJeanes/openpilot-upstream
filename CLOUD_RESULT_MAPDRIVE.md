# Cloud result: map driver lane changes, gap waits and red lights

## Summary for Amy

Preflight done; work on the three items follows below as it lands.

## Preflight

1. Repo: PASS. Shallow single-branch clone of `mapdrive-cloud` at `081e37263ec0`; its parents are `020b96f2f19e` (the
   brief) and `647143d89ce8` (`gta5`, matching `git ls-remote origin gta5`).
2. Data: PASS. `cloud-data` at `031aeacd62ff`; sha256 of all 11 files under `cloud/live18/map1/` and `cloud/live19b/`
   match `cloud/MANIFEST.md`. Live map assembled in one folder (live18's pbf, nodes.npz, gunzipped roads.json and
   paths.jsonl, live19b's gunzipped lanes.json: 69666 lines, 406 of kind `hatched`), `GTA5_MAP` set to it.
3. Dependencies: PASS with one gap. From PyPI into a Python 3.12 venv: numpy, osmium, pyvalhalla 3.9.1, shapely,
   pytest, scipy, pycapnp, opendbc 0.2.1, msgq 0.1.3. PyPI's opendbc is too old for this tree's `cereal/log.capnp`
   (`'Car.RadarData' has no member named 'ErrorDEPRECATED'`, capnp aborts the process), and the submodules aren't
   reachable here. So `test_blindspot.py` can't import `openpilot.cereal.log` / the simulated Tesla. Designed around:
   it runs with a scratch pytest plugin (not committed) that stubs `cereal.log`'s two enums and the Tesla/CAN imports;
   the 3 tests that need openpilot's real CarState/DesireHelper (`test_tesla_can_carries_the_blind_spot`,
   `test_openpilot_lane_change_waits_for_the_blind_spot`, `test_openpilot_lane_change_waits_out_a_moments_gap`) can't
   run here. None of them touch the map driver.
4. Valhalla: PASS. Tiles built from the pbf with pyvalhalla's `valhalla_build_tiles` (9 s), `valhalla_service` on
   :8002 answering `/status`; `mapdrive_sim.trip_route` routes on it (cache in a scratch file via
   `GTA5_TRIP_ROUTES`).
5. Baseline on the untouched branch: `test_mapdrive.py` + `test_mapdrive_lead.py` 68 passed, 1 failed
   (`test_collision_routes`: route 029's plan 1.12 m off its lane, the known one), as on Amy's PC;
   `test_lane_slots.py` 20 passed; `test_blindspot.py` 32 passed, 3 not runnable (above). This is the bar.
6. Push access: PASS (this file, pushed to `mapdrive-cloud`).
