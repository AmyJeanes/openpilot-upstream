# Cloud result: map lane geometry from the paint (drift fixes)

Cloud session, 16 Oct 2026. Only GitHub, no game. Work on `lanegeom-cloud`.

## Preflight

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Repo | PASS | `lanegeom-cloud` = `cf6ffcf3b33e` (the brief), parent `84739d1a9a5e` = `gta5` head (`git ls-remote origin gta5`). |
| 2 | Data | PASS | `cloud-data` @ `891a36f171ab`: 80/80 manifest rows under `live18/map1`, `live19b`, `mapfuse_20261014`, `gamefiles`, `drift_20261016` match sha256; no unlisted or missing files in those dirs. Live map assembled in `/home/user/live/live19b` (live18 pbf, roads.json, paths.jsonl, nodes.npz + live19b lanes.json). |
| 3 | Dependencies | PASS | `uv venv -p 3.12` + PyPI: numpy 2.x, shapely 2.2.0, osmium (pyosmium), pyvalhalla 3.9.1 (valhalla_build_tiles / valhalla_service in `site-packages/valhalla/bin`), scipy, pytest 9.1.1, matplotlib. Nothing blocked. |
| 4 | Rebuild | PASS | `osm_to_roads.py live18.pbf roads_check.json --lanes lanes.json --frame gta5 --hatched live19b/hatched19b.json` (build19b.sh's flags), 8 min: lanes.json md5 `0f0d14ad3d7c…` and roads.json md5 `dc87168dd6ec…` are byte-identical to live19b's and live18's. Log: "331 hatched areas (50 listed, 8 held, 0 of those not found), 75 given as rings / 69666 lines". |
| 5 | Baseline scores | pending | |
| 6 | Push access | pending | this commit |
