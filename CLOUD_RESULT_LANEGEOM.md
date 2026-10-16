# Cloud result: map lane geometry from the paint (drift fixes)

Cloud session, 16 Oct 2026. Only GitHub, no game. Work on `lanegeom-cloud`.

## Preflight

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Repo | PASS | `lanegeom-cloud` = `cf6ffcf3b33e` (the brief), parent `84739d1a9a5e` = `gta5` head (`git ls-remote origin gta5`). |
| 2 | Data | PASS | `cloud-data` @ `891a36f171ab`: 80/80 manifest rows under `live18/map1`, `live19b`, `mapfuse_20261014`, `gamefiles`, `drift_20261016` match sha256; no unlisted or missing files in those dirs. Live map assembled in `/home/user/live/live19b` (live18 pbf, roads.json, paths.jsonl, nodes.npz + live19b lanes.json). |
| 3 | Dependencies | PASS | `uv venv -p 3.12` + PyPI: numpy 2.x, shapely 2.2.0, osmium (pyosmium), pyvalhalla 3.9.1 (valhalla_build_tiles / valhalla_service in `site-packages/valhalla/bin`), scipy, pytest 9.1.1, matplotlib. Nothing blocked. |
| 4 | Rebuild | PASS | `osm_to_roads.py live18.pbf roads_check.json --lanes lanes.json --frame gta5 --hatched live19b/hatched19b.json` (build19b.sh's flags), 8 min: lanes.json md5 `0f0d14ad3d7c…` and roads.json md5 `dc87168dd6ec…` are byte-identical to live19b's and live18's. Log: "331 hatched areas (50 listed, 8 held, 0 of those not found), 75 given as rings / 69666 lines". |
| 5 | Baseline scores | PASS | Below. Scorer: `map/paint_score.py` (step 1), map = rebuilt live19b. |
| 6 | Push access | PASS | `81c99c4` pushed to `lanegeom-cloud` (preflight 1-4); this commit too. |

### Baseline scores (live19b, current `gta5` code)

**Map vs paint** (confident paint, 1,277k samples at 1 m; matched = a map line of the same family within 2.5 m, at its
height, within 25 deg: 73% of samples, 936 km). Share of matched length with the map's nearest line more than 0.5 / 1.0 m away:

| class | matched km | > 0.5 m | > 1.0 m | median |
|---|---|---|---|---|
| all | 935.6 | 25.3% | 8.7% | 0.11 m |
| motorway | 234.4 | 18.8% | 7.7% | 0.09 |
| trunk | 55.1 | 11.1% | 5.5% | 0.07 |
| primary | 194.9 | 34.7% | 9.9% | 0.17 |
| unclassified | 31.8 | 36.5% | 20.5% | 0.24 |
| residential | 399.9 | 24.0% | 7.0% | 0.11 |
| service | 18.5 | 55.2% | 34.8% | 0.61 |

By family: white lane lines 9.6% / 5.5%, yellow centre lines 6.9% / 4.1%, edges 53.1% / 15.1% (the paint's `edge` is
kerbs and painted edges alike; the map's is the asphalt edge, so edges are the noisiest family). The brief's "about 9%
over 1 m, about 12% on primary" reproduces as 8.7% and 9.9%.

**Drift re-score** (frames.npz, 97 trips; the routes rebuilt from the frames, 93 of 97 legs; frames moving, map driver,
own lane, no lane change within -2/+3 s, paint lane 4-7.5 m: 159,577 frames, 2.2 h). The rebuilt static target centre
matches the recorded `target_right - route_right` within 0.1 m on 97.3% of frames, 0.5 m on 99.7% (p99 0.21 m): the
scorer reproduces what the car was given. Share of those frames with the map's target centre more than 0.5 / 0.75 / 1.0 m
off the paint lane's centre:

| target | frames | > 0.5 m | > 0.75 m | > 1.0 m | > 1.0 m, > 20 m from junctions |
|---|---|---|---|---|---|
| recorded (target_right) | all | 11.3% | 6.3% | 3.8% | 2.5% |
| rebuilt static (`Section.offset`, what target_right is) | all | 10.4% | 5.9% | 3.6% | 2.5% |
| rebuilt `section_at` (tapers, hops, widening along a way) | all | 10.2% | 5.8% | 3.6% | 2.5% |
| rebuilt planned line (`lane_line`: eased jogs, fillets) | all | 11.5% | 6.9% | 4.5% | 2.6% |

(The brief: 2.5-3.8% of driven time more than 1 m off.)

**Target steps** (> 0.7 m in one frame, same target lane, no lane change):

| target | per km | away from junctions (> 20 m) | per km | > 2 m | paint-checked | spurious |
|---|---|---|---|---|---|---|
| recorded | 4.93 | 142 | 0.88 | 47 | 68 | 24 |
| rebuilt static | 5.55 | 162 | 1.01 | 66 | 77 | 23 |
| rebuilt `section_at` | 3.27 | 58 | 0.36 | 42 | 30 | 2 |

The recorded counts reproduce the brief exactly (792 steps, 4.9/km, 142 away, 47 over 2 m; 24 of 68 spurious against
the brief's 25 of 59, a slightly different paint check window). The rebuilt static target has a few more steps than the
recorded one because the rebuild changes segment with the car's place rather than the plan's (one frame apart; one-frame
glitches are taken out). The planner's own cross-section (`section_at`, with the hop logic of 14 Oct, after these
recordings) already removes most spurious steps (2 of 30 left), so most of cause 2 sits in the **static** target: the
labels (`target_right`), lanes.json, and the map driver's lane numbering read it.

Per-event status (24 j20 events, 5 recurring places, other named places): `cloud_lanegeom/baseline/events.csv`.

**Tests** (live map = rebuilt live19b in `~/gta5map_lanes`, Valhalla 3.9.1 tiles built from the live pbf on port 8002):

| test | baseline |
|---|---|
| test_lane_numbering.py (pytest) | 29 passed, 1 failed: `test_live_freeway_merge_exit_and_splits` (the router finds a 3,063 m route where the test wants < 300 m: this cloud Valhalla build routes differently; not a lane fault) |
| test_lane_slots.py | 20 passed |
| test_freeway_lanes.py | 7 passed |
| test_mapdrive.py (pytest) | 37 passed, 1 failed: `test_collision_routes` ('029' off 1.12 m), the known failure on Amy's PC. Recordings (/mnt/e/gta5rec) absent: their cases skip. |
| validate_lanes.py | 0 issues |
| lane_parity.py | 94,748 directed ways checked, 4,979 differ from GTA's links' layout (exit 1 when any differ) |

(`python test_lane_numbering.py` as a script stops at the first test needing pytest fixtures, so pytest is used.)
