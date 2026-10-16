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

## Step 1: the geometry scorer (`map/paint_score.py`)

`paint_score.py MAP_DIR --paint paint_lanes_confident.json --drift drift_20261016/ --out OUT [--base BASE_OUT]`
(`--legs` reuses the routes rebuilt from the frames; `--roads` gives the road classes when MAP_DIR has no roads.json).
It writes `report.json`, `report.md` (every number beside the baseline's with `--base`), `places_static.csv` (15 m cells
by matched paint length more than 1 m from the map), `places_drift.csv` (15 m cells by driven time with the target more
than 1 m off the paint), `events.csv` (the 24 j20 events, the 5 recurring places and the other places the brief names,
each with its largest |target - paint centre| for the three targets, before and after), and `steps_*.csv`.

- **Map vs paint**: the paint's lines (solid/dashed white, yellow centre lines, edges) sampled every 1 m; for each
  sample the nearest map line of its family within 2.5 m, within 3 m of height and 25 deg of heading (`Segments`,
  vectorised: about a minute for the whole map). With `--base`, the same samples are compared one for one (km moved
  more than 0.5 m nearer or further).
- **Drift re-score**: each trip's route is rebuilt from its frames (the route's line is the car less its recorded
  route_right across its heading; the map's nodes along that line, at the car's height, chained through the map's ways),
  so any map with the recorded map's ways re-scores the same frames. Three targets: the segment's own cross-section
  (`Section.offset`, what `target_right` is), the cross-section where the car is (`RouteLanes.section_at`), and the line
  the map driver plans through the recorded target lanes (`RouteLanes.lane_line`). On live19b the rebuilt static target
  is within 0.1 m of the recorded one on 97.3% of frames.
- `test_paint_score.py` (4 cases on made maps and paint) passes.

Baseline numbers are under Preflight; `cloud_lanegeom/baseline/` holds the baseline's report and lists. A rerun of the
baseline against itself gives identical numbers (deterministic).

## Step 2: lanes carry on across nodes (cause 2)

**What the paint says.** On the drift frames, at the 49 places the routes pass where a way's lane count rises or falls
at a node it just carries on through (the map's `tapers`), the paint lane the target lane is in was compared, every 3 m
from 24 m before the node to 24 m past it, with where the way before and the way after each put that lane (counted from
the side the lane opens or closes on):

| m from node | -24..-12 | -12..-6 | -6..0 | 0..+6 | +6..+12 | +12..+24 |
|---|---|---|---|---|---|---|
| lane added: mean error, way before / way after | 0.5 / 1.5 | 0.4-0.6 / 1.3 | 0.7-0.9 / 0.9-1.2 | 1.3-1.5 / 0.5 | 1.6 / 0.4 | 1.6-1.8 / 0.5-0.7 |
| lane ending: way before / way after | 0.5-0.7 / 1.5-1.7 | 0.5 / 1.3 | 0.5-0.7 / 1.3 | 0.7-1.1 / 0.7-1.1 | 1.2 / 0.6 | 1.1-1.2 / 0.6 |

So GTA's lanes really do move across at these nodes (each link's lanes sit about its own line), and the paint makes
the move within a few metres of the node: a lane added, from about 6 m before to 6 m past; a lane ending, from about
9 m before to 3 m past. The map instead kept the way before's lanes for **30 m past the node** (`TAPER_M`, a taper on
the way with more lanes only): inside those tapers the planner's cross-section was more than 1 m off the paint on 52%
of frames (lane added) and 49% (lane ending), against 19% and 29% for each way's own lanes. The paint shows the new
lane's own line appearing over the first 10-15 m past the node (a paint line at its place on 38-42% of frames by the
node, 61% by 10-15 m, 68-71% at the plateau from 15 m).

**Change** (`osm_lanes.py`):
- The taper is centred on the node: `TAPER_BEFORE`/`TAPER_AFTER` = 6/6 m where a lane begins, `DROP_BEFORE`/`DROP_AFTER`
  = 9/3 m where one ends (`taper_u`). The new lane widens from nothing on the side it opens (the lane ending narrows to
  nothing), and the lanes carrying on are counted from the other side, as before (`_blend`, `_opens_left`). The way with
  fewer lanes now takes its part of the taper too: its lanes move across to where they carry on (`_toward_many`), in
  `RouteLanes.section_at` (the planner, nav, labels) and `OsmLanes.taper` (lanes.json's lines and kerbs), so the two
  ways meet at the node with no step.
- `TAPER_M` (where a new lane counts as open, its line painted: nav's `opening`/`opened_at`, lane slots) is 15 m past
  the node, from the paint, instead of 30 m.
- `gta5_mapdrive.target_centre`: the map driver's `target_right` is the target lane's centre where the plan is, as
  `section_at` has it (tapers, hops, blends), not the segment's own cross-section, which stepped at every node where
  the next way's lanes sit elsewhere (most of the brief's 4.9 steps per km). The planned line was never stepped (it used
  `section_at`), only the logged target was.
- `_ease_jogs` is unchanged: with the tapers continuous it no longer fires at them; it remains for the real jogs
  (junction-node lane count changes, a divided road's carriageway onto the two-way road), where JOG_REACH 10 m matches
  the paint's few-metre move. Per-node paint taper lengths are not read (the measured window is the same at every node).

**Scores** (`cloud_lanegeom/step2/report.md`, before = baseline):

| | before | after |
|---|---|---|
| map vs paint > 1 m, all (same samples) | 8.7% | 8.6% |
| primary | 9.8% | 9.3% |
| trunk | 5.5% | 5.3% |
| motorway | 7.7% | 7.6% |
| paint length moved > 0.5 m nearer / further | | 2,471 m / 781 m |
| drift, `section_at` target > 1 m (all / j20) | 3.61% / 2.51% | 3.41% / 2.35% |
| drift, planned line > 1 m (all / j20) | 4.46% / 2.57% | 4.34% / 2.47% |
| inside the old tapers, `section_at` > 1 m: lane added / lane ending | 52% / 49% | 22% / 39% |
| inside the old tapers, planned line > 1 m: lane added / lane ending | 40% / 51% | 30% / 39% |
| target steps (target_right as now logged = `section_at`), per km | 4.93 (recorded) | 3.27 |
| ... away from junctions | 142 (0.88/km), 47 > 2 m | 57 (0.36/km), 43 > 2 m |
| ... spurious (paint lane did not move) | 24 of 68 | 1 of 29 |

The rebuilt static target (the old target_right) is unchanged by construction; its steps are the ones the planner never
had. Events: the 2-to-1 drop at (1190.9, 441.6) improves (planned line 3.36 -> 2.79 m), the rest are unchanged (their
errors are offsets, exits and plan logic: steps 3-5).

**Tests** (live map = this build): lane_numbering 28 passed + `test_live_freeway_merge_exit_and_splits` (baseline's
cloud-router failure); lane_slots 20; freeway_lanes 7; mapdrive 37 + `test_collision_routes` (baseline's); validate_lanes
0 issues; lane_parity 4,979 of 94,748 (unchanged); map tests (route_lanes, osm_lanes, lane_continuity, paint_score,
junctions, lane_match, carriageway, validate_lanes, stop_lines) 149 passed. Updated to the new taper: `test_route_lanes`
(`test_route_lanes`, `test_taper_lines`), `test_lane_slots` (`test_turn_bay_change_lanes`,
`test_bay_opens_after_its_taper`: the bay opens TAPER_M past its node), `test_lane_numbering`
(`test_right_turn_bay_beside_an_open_left_lane`). New: `map/test_lane_continuity.py` (2 -> 3 opening left by its arrows,
opening right, 3 -> 2 dropping, a two-way 1+1 -> 2+1, lines meeting at the node, target_right without a step, and four
live places: lane added at (-460,-221), (644,-1244), (-74,-1552), lane dropped at (1183,434)). `test_nav_msgs.py` also
uses TAPER_M but needs pycapnp/cereal and doesn't run here.

lanes.json: 69,714 lines (69,666 before); roads.json changes too (32,224 drawn ways, 26,443 junction areas, against
32,201 and 26,420): the ways beside a taper are drawn as their own strips now, and turn bays opening less than 30 m
before a junction are fully open at it, so those junctions' outlines follow.
