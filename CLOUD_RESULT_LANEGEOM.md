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

## Step 3: exits, on-ramps and lane chains anchored where they part (cause 2)

**Change** (`osm_lanes.py`, `RouteLanes._diverges`): where a route leaves a freeway (motorway/trunk and their links) at
a node onto a freeway way with fewer lanes that strays to one side, or joins one from such a way, the branch's lanes
run on as the road's lanes on that side (counted from that side), measured from the road's line run on straight past the
node, until the branch's own lanes come out past them (within 0.3 m: where GTA's link has crossed the road's lanes, at
the gore), and follow the branch from there. GTA starts every exit and chain split at the road's middle node, so before
the branch's lanes swung in to the middle and out. The branch may continue onto ways of any class (the Strawberry ramp
is `residential` after its first link); at most 100 m (then it eases onto its own lanes over its last link). Guards:
- **Never further into a hatched area**: the moved lane centres are checked against lanes.json's hatched rings
  (`router.LaneLines.inside`, at the route's height); a branch whose lanes would run further inside one (0.5 m) is left
  as it was. (A first version without this ran a lane diagonally across the gore at (1005, -935).)
- Not where the road itself bends tighter than 200 m radius over the 30 m before the node (a straight run-on strays).
- **The paint**: `diverge_report.py --keep` writes the places where the moved lanes sit further from the confident
  paint (0.1 m on average, paint on 30%+ of the stretch) into `lanegeom.json` beside the map as diverges to keep
  (`osm_lanes.GEOMETRY_FILE`, read by `OsmLanes.load`, keyed by GTA node ids `from>to`, with the evidence). A kept
  diverge, and any `_diverges` leaves be, keeps the lane-hop handling it had before (`_hops` single-ended).
- `_hops` now takes only links between two roads (both ends); one-ended hops are diverges/merges. `DIVERGES = False`
  restores the step-2 behaviour exactly (the report uses it for "before").

No nodes move and the pbf is unchanged: this is route geometry (what the map driver plans, nav's lanes, target_right).
lanes.json's lines are unchanged by step 3 (the ramp links are still drawn where GTA has them; see open questions).

**Map-wide** (`diverge_report.py`, `cloud_lanegeom/step3/diverges.csv`, `summary.json`): 583 freeway diverges and
merges; **109 change** (68 diverges, 41 merges), 97 by more than 1 m, 78 by more than 3 m. On the 107 with paint, the
branch lanes' mean distance from the paint lane's centre goes **0.50 -> 0.39 m**, the share over 1 m **17.8% -> 12.3%**;
42 places nearer the paint by 0.1 m+, 2 further. Lane-centre metres inside hatched areas at the changed places:
4.4 -> 0.0. Largest moves:

| key (GTA from>to) | node | kind | moved m | paint err before | after | paint on |
|---|---|---|---|---|---|---|
| 462:804>462:766 | (-646.0, -527.8) | diverge | 12.19 | 0.1 | 0.1 | 0.68 |
| 464:986>464:957 | (37.0, -524.2) | diverge | 12.15 | 0.26 | 0.26 | 0.55 |
| 465:713>465:774 | (637.2, -558.5) | diverge | 11.81 | 1.43 | 1.45 | 0.31 |
| 465:565>465:618 | (755.7, -614.5) | diverge | 11.66 | 1.01 | 1.01 | 0.04 |
| 465:578>465:644 | (743.7, -608.8) | diverge | 11.17 | 0.81 | 0.86 | 0.08 |
| 497:8>496:64 | (515.0, -506.5) | diverge | 10.92 | 0.65 | 0.7 | 0.37 |
| 497:5>496:57 | (527.8, -509.5) | diverge | 10.41 | 0.9 | 0.7 | 0.28 |
| 465:673>465:731 | (660.0, -571.2) | diverge | 10.07 | 1.52 | 1.52 | 0.33 |
| 495:41>495:63 | (-136.3, -497.8) | diverge | 9.96 | 0.86 | 0.82 | 0.48 |
| 464:979>464:974 | (1.0, -524.2) | diverge | 9.85 | 0.54 | 0.4 | 0.55 |

Kept as they were by the paint (`lanegeom.json` `diverges`, 14 places):

| key | node | kind | why |
|---|---|---|---|
| 399:205>399:214 | (-463.0, -1890.0) | merge | paint lane centre 0.29 m off as the map has it, 0.46 m moved (over 1 m 0.087 -> 0.212, paint on 0.95 of it) |
| 399:266>399:254 | (-447.7, -1859.7) | diverge | paint lane centre 0.1 m off as the map has it, 0.37 m moved (over 1 m 0.0 -> 0.143, paint on 0.76 of it) |
| 399:757>399:778 | (-422.7, -1591.2) | diverge | paint lane centre 0.1 m off as the map has it, 0.3 m moved (over 1 m 0.0 -> 0.132, paint on 0.43 of it) |
| 431:705>431:647 | (-126.7, -1235.0) | diverge | paint lane centre 0.38 m off as the map has it, 1.32 m moved (over 1 m 0.089 -> 0.558, paint on 0.39 of it) |
| 434:43>434:13 | (1044.5, -1503.0) | diverge | paint lane centre 0.57 m off as the map has it, 0.76 m moved (over 1 m 0.262 -> 0.328, paint on 0.95 of it) |
| 462:701>462:702 | (-936.5, -564.7) | merge | paint lane centre 0.12 m off as the map has it, 0.69 m moved (over 1 m 0.034 -> 0.294, paint on 0.47 of it) |
| 462:770>462:721 | (-991.7, -557.8) | merge | paint lane centre 0.39 m off as the map has it, 0.84 m moved (over 1 m 0.133 -> 0.354, paint on 0.45 of it) |
| 462:777>462:774 | (-947.0, -538.3) | diverge | paint lane centre 0.37 m off as the map has it, 1.8 m moved (over 1 m 0.297 -> 1.0, paint on 0.77 of it) |
| 494:70>494:79 | (-634.0, -473.3) | merge | paint lane centre 0.19 m off as the map has it, 0.35 m moved (over 1 m 0.009 -> 0.1, paint on 0.6 of it) |
| 496:89>496:111 | (380.7, -471.7) | merge | paint lane centre 0.6 m off as the map has it, 0.83 m moved (over 1 m 0.164 -> 0.344, paint on 0.42 of it) |
| 565:33>564:97 | (2559.2, 788.5) | merge | paint lane centre 0.4 m off as the map has it, 0.65 m moved (over 1 m 0.1 -> 0.236, paint on 0.77 of it) |
| 596:243>596:233 | (2105.3, 1370.0) | merge | paint lane centre 0.08 m off as the map has it, 0.32 m moved (over 1 m 0.0 -> 0.033, paint on 0.64 of it) |
| 846:344>846:353 | (-848.5, 5442.0) | diverge | paint lane centre 0.26 m off as the map has it, 1.76 m moved (over 1 m 0.071 -> 0.756, paint on 0.63 of it) |
| 912:192>912:178 | (105.0, 6483.0) | merge | paint lane centre 0.18 m off as the map has it, 0.67 m moved (over 1 m 0.0 -> 0.282, paint on 0.74 of it) |

Still further from the paint after the change, with too little paint to decide (under 30% of the stretch): 465:521>465:543 at (791.7, -633.3): 0.65 -> 0.97 m, paint on 16%; 399:790>399:824 at (-424.2, -1575.7): 0.22 -> 0.48 m, paint on 29%.

The brief's places: the **Strawberry -> Olympic exit** (432:955 > 432:982) moves 4.7 m: the exit lane now runs on in the
kerb lane, between the paint's lines at y -1170 and -1176.3 (centre -1173.2), and leaves along the ramp past x 250,
instead of dipping to -1177 towards the motorway's middle node; along the branch, the share of it more than 1 m off the
paint goes 24% -> 6% (`diverge_report`). On the drift frames it is unmeasurable afterwards (the frames there have the
car following the old dip, so the paint lane the new target sits in isn't bracketed from the car's pose). The chain split
**(1079.5, -1546.2)** was already right in the planner (14 Oct hop logic: target 1.37 m / line 1.10 m from the paint,
against 2.91 m for the recorded target); its 2-lane side (402:738 > 402:762) moves 1.6 m nearer the paint (0.55 -> 0.47 m).
**(-378, -2062)** is not a diverge in the map's terms: it is GTA's parallel one-lane chains, one surveyed link with its
lane 2.7 m off its line between unsurveyed links centred on theirs: step 4 (lane offsets from the paint).

**Drift re-score** (`cloud_lanegeom/step3/report.md`, before = step 2): `section_at` target over 1 m 3.41% -> 3.39%
(j20 2.35% -> 2.32%); planned line 4.34% -> 4.34%; target steps away from junctions 57 -> 54. Events: (-127.7, -1182.5)
planned line 3.51 -> 2.49 m, (-388.5, -975.8) 2.84 -> 2.40 m, (530.7, -498.4) 3.03 -> 1.81 m; the rest unchanged. (The
97 trips pass few exits; the map-wide report is the measure here.)

**Tests**: as the baseline (lane_numbering 29 + the cloud-router case; lane_slots 20; freeway_lanes 7; mapdrive 37 +
`test_collision_routes`; validate_lanes 0 issues; lane_parity unchanged); map tests 153 passed, with new cases in
`test_lane_continuity.py`: an exit leaving from the middle node runs on from the kerb lane and only ever outward, an
on-ramp ending at the middle node comes in to the kerb lane, a 3-lane link splitting into a 2-lane link and a 1-lane chain
keeps the chain's lane straight on as the right lane, and a hatched gore where the moved lane would run leaves the exit
as it was (the old code fails the first three).

## Step 4: lane centres from the paint (cause 1)

**Where the change lives, and why.** `lanegeom.json` beside the map's pbf (`osm_lanes.GEOMETRY_FILE`): its `ways`
section gives, per way, the OSM tags that place its lanes on the paint (`width:lanes` or `width:lanes:forward` /
`:backward`, `width`, `placement:offset`; null removes a tag), keyed by the way's end nodes' GTA ids (`a>b`), with the
evidence (edges before and after, which edges sit on paint, the median shift, stations). `OsmLanes.way_tags` lays them
over the pbf's tags when it reads a way's lanes (`OsmLanes.load` and `osm_to_roads.py` read the file;
`validate_lanes.py` checks the result; `lane_match.map_hash` includes the file, so the bridge's junction and stop-line
caches follow it). That is the smallest change that moves everything that reads lane geometry (lanes.json and
roads.json via osm_to_roads, nav and the map driver via RouteLanes, lane_match, the labels) while the pbf, its ways,
nodes, access and turn rules, and so Valhalla's routing, stay byte for byte; and it is one reviewable file, as the
phase-1 rule asks. The same file carries step 3's kept diverges.

**The fit** (`map/lane_fit.py`): the paint is the 2+-tile set only (`survey_paint.jsonl` lines with conf.mean >= 0.75,
>= 2 m, drawn again by a second tile on at least half the length two tiles covered: 1,137 km). Each way is read with
its strand (the ways it carries straight on into, 24 m past each end: GTA's links are mostly about 12 m), every 2 m,
clear of junction nodes and stacked roads. Painted lines only, as the brief says (a kerb isn't a lane's edge); both
sides of a lane on paint: its centre and width from the paint; one side: shifted to it, width kept (2.7-6.5 m).

**Three variants were built and scored**; the third is kept:

| variant | ways changed | student paint > 1 m (same samples) | drift planned line > 0.5 m | game-file paint > 1 m (out of sample) | game-file nearer / further by 0.5 m |
|---|---|---|---|---|---|
| A: kerbs as lane edges too | 7,527 | 8.6% -> 8.3% (40.6 km nearer / 3.5 km further) | 11.33% -> 10.14% | 7.05% -> 7.09% | 4.8 km / 4.4 km |
| B: painted lines only | 2,652 | 8.6% -> 8.7% | 11.33% -> 11.10% | 7.08% -> 7.08% | 4.8 km / 3.7 km |
| **C: B, and not where the build measured the lanes from the game files** | **1,383** | **8.6% -> 8.6% (2.0 km nearer / 1.0 km further)** | **11.33% -> 11.06%** | **7.11% -> 6.99%** | **2.9 km / 1.2 km** |

A and B moved residential and motorway lines away from the game files' own paint where the map already sat on it
(those ways' lanes were placed from the game files' survey, `source:width=survey`: median 0.05 m off the game-file
paint before, 0.75 m after). The game files' paint (`gamefiles/polylines.jsonl`) is independent of the student, so
C keeps the game files there (Amy's rule: prefer the game files where they and the paint disagree) and lists those
ways instead: **1,308 surveyed ways where the paint seen from above disagrees with the game files by 0.25 m or more**
(`cloud_lanegeom/step4/listed.csv`, "game files measured").

**Result (C)**: 1383 ways changed (primary 641, residential 536, motorway 95, unclassified 61, motorway_link 28,
trunk 20, trunk_link 2), 51 by more than 1 m, 10 by more than 2 m; 8,200 more fit the paint already within 0.25 m.
Left alone and listed: stacked 2838; lane over 6.5 m 1401; game files measured it (paint disagrees) 1308; lane counts disagree 451; paint disagrees along it 402; lane under 4.0 m 374; no unique key 185; lanes widen or move along it 174; tags cannot place it 2. Largest changes:

| key | class | where | moved m | edges before | after | on paint |
|---|---|---|---|---|---|---|
| 496:839>496:861 | residential | (96.2, -108.0) | 3.14 | -5.50|0.00|5.50 | -2.85|3.14|8.64 | PP. |
| 431:51>431:68 | motorway | (-378.0, -1489.8) | 3.04 | -2.75|2.75 | 0.29|5.40 | PP |
| 431:68>431:89 | motorway | (-376.0, -1478.7) | 3.01 | -2.75|2.75 | 0.26|5.42 | PP |
| 462:730>462:729 | motorway | (-679.2, -556.0) | 2.93 | -2.75|2.75 | 0.18|5.48 | PP |
| 653:114>653:117 | residential | (-1349.0, 2444.0) | 2.5 | -8.20|-2.70|2.70|8.20 | -10.54|-5.04|0.20|5.94 | ..PP |
| 561:103>561:90 | unclassified | (630.5, 634.8) | 2.4 | -2.20|2.20 | -0.40|4.60 | PP |
| 491:43>492:162 | primary | (-2040.8, -425.3) | 2.33 | -5.50|0.00|5.50 | -7.82|-2.33|3.20 | PPP |
| 561:116>561:103 | unclassified | (634.8, 646.0) | 2.32 | -2.75|2.75 | -0.43|4.55 | PP |
| 561:120>561:121 | unclassified | (650.5, 656.8) | 2.18 | -2.20|2.20 | -0.02|4.09 | PP |
| 561:92>561:98 | unclassified | (659.2, 641.5) | 2.1 | -2.20|2.20 | -4.28|0.10 | PP |
| 561:46>561:49 | unclassified | (643.5, 599.7) | 1.9 | -2.20|2.20 | -4.10|0.31 | PP |
| 399:481>399:504 | primary | (-398.5, -1756.0) | 1.67 | -5.50|0.00|5.50 | -3.83|0.95|6.45 | PP. |

By class (same samples before and after; student paint before = step 3, game files before = the live19b baseline):

| class | student paint > 1 m | nearer / further 0.5 m (m) | game files > 0.5 m | game files > 1 m | game files nearer / further (m) |
|---|---|---|---|---|---|
| all | 8.6% -> 8.6% | 2001 / 981 | 12.1% -> 11.9% | 7.1% -> 7.0% | 2895 / 1227 |
| motorway | 7.7% -> 7.7% | 483 / 313 | 12.6% -> 12.5% | 8.6% -> 8.5% | 615 / 259 |
| trunk | 5.4% -> 5.3% | 8 / 2 | 12.6% -> 12.3% | 9.6% -> 9.4% | 157 / 3 |
| primary | 9.5% -> 9.5% | 980 / 375 | 13.4% -> 12.2% | 7.9% -> 7.4% | 1598 / 401 |
| residential | 6.9% -> 6.9% | 486 / 231 | 10.2% -> 10.2% | 4.2% -> 4.3% | 477 / 484 |
| unclassified | 20.4% -> 20.5% | 33 / 52 | 13.2% -> 13.5% | 8.7% -> 8.7% | 30 / 64 |

Drift re-score (before = step 3): planned line > 0.5 m 11.3% -> 11.1%,
> 1 m 4.3% -> 4.2%; `section_at` > 0.5 m
10.0% -> 9.8%, > 1 m 3.4% -> 3.4%.
The brief's offset places are all left alone, each for a listed reason: (901.4, -695.3) the lane fits 6.9 m wide (a line
missed); (530.7, -498.4), (134.0, -492.0) and the Strawberry ramp are stacked interchanges; (-13.9, 274.6) a lane fits
3.8 m wide (a bay or a false line); (-378, -2062) the paint disagrees along the chain links (shifts of +1.6 and -1.1 m,
25% agree) and shows a line inside a lane (counts disagree); (125.6, -1367.1) has too little 2+-tile paint near it.
These need Amy's labels or a drive.

**Tests**: as the baseline (lane_numbering 29 + the cloud-router case, lane_slots 20, freeway_lanes 7, mapdrive 37 +
`test_collision_routes`; lane_parity unchanged: it checks the pbf); validate_lanes **0 issues with the applied tags**
(it now reads the file); map tests 153 passed; new `map/test_lane_fit.py` (7 cases: both sides set centre and width
and the tags read back; one side keeps the width; a lane under 4 m and a line inside a lane left alone; a stacked way
left alone; a one-lane link on a lane line of a wide carriageway ambiguous; the key).

Amy's own labels (150 reviewed tiles) are not used: they're in tile pixels and the projection to the world isn't in the
cloud data's tools; see open questions.

## Step 5: turn fillets (cause 3)

**Change**: `FILLET_RADIUS['left']` 20 -> 15 m (`map/osm_lanes.py`). Nothing else in the fillets or `_clamp` changed;
the map build (lanes.json, roads.json) doesn't use fillets, so this is driver code only. New tool `plan_term.py`
drives `mapdrive_sim.py` trips routed by Valhalla (pyvalhalla 3.9.1 on port 8002, tiles from the live pbf), records the
plan term per frame (MapDrive's intent before its bias and wander, less the map's target centre) plus the map term
and the car's distance from its paint lane centre, and sums them 30 m before (entry) and after (exit) each turn of
45 deg or more. Trips: `cloud_lanegeom/step5/plan_specs.json` (11 trips from the drift sessions: the 3 cause-3
places, the Strawberry ramp, (-378,-2062) and the (68.7,-1146) step). Results in `cloud_lanegeom/step5/`.

**Why the left radius.** In the drift frames (97 legs, all turns of 45 deg or more), the planned line was further
from the paint than the lane centre from about 15 m before a corner to 15 m after. On right turns this is mostly
inside +/-10 m, where the line has to cut the corner. On left turns, the 20 m arc is still curving 10-15 m out of
the corner: an arc of radius 20 ending 20 m on is about 1.4 m off its straight line 7.5 m before it ends. At 15 m:

| drift, planned line vs paint | step 4 | left radius 15 |
|---|---|---|
| all frames > 0.5 m | 11.06% | 10.86% |
| all frames > 1 m | 4.25% | 4.18% |
| > 1 m away from junctions (j20) | 2.49% | 2.46% |
| left turns, -15..-10 m, median | 0.61 m | 0.43 m |
| left turns, -10..-5 m, median | 0.82 m | 0.66 m |
| left turns, +10..+15 m, median | 0.49 m | 0.45 m |
| right turns | unchanged | unchanged |

The 90 deg left with a 4 m median (test_clamp_corners' road): the unclamped line reaches 2.32 m into the corner
kerb, was 3.79 m. At 60 deg it no longer reaches the kerb (was 0.30 m).

**Plan term on the mapdrive_sim trips (before -> after)**, frames over 2 m/s:

| | plan > 0.5 m | plan p90 | plan max | car > 0.5 m off its paint lane centre |
|---|---|---|---|---|
| all turn entries | 59.2% -> 56.4% | 2.80 -> 2.52 m | 8.63 -> 8.63 | 68.4% -> 65.7% |
| all turn exits | 64.1% -> 61.9% | 2.62 -> 2.47 m | 8.82 -> 8.82 | 47.8% -> 47.5% |
| elsewhere | 12.9% -> 12.9% | 0.67 -> 0.67 m | 5.39 -> 5.39 | 33.8% -> 33.8% |
| (867.3,-1745.8) trip 018 (left then right), entries | 66.9% -> 54.4% | 3.24 -> 2.09 m | 5.87 -> 4.44 | 55.2% -> 41.8% |
| (867.3,-1745.8) trip 018, exits | 68.5% -> 60.5% | 3.38 -> 2.55 m | 5.99 -> 4.58 | 41.0% -> 39.0% |

The other trips have only right turns, or no turns, and are the same frame for frame (per trip:
`plan_term_compare.md`). All 11 arrive before and after. A large plan term at a turn is mostly the target's fault,
not the plan's: the map's target centre follows the ways to the junction node and turns there, which no car drives
(at (-1521.4,-243.0) the target's corner, at the junction node, sits about 3 m outside the curve the car drives and
the paint lane's: plotted from `plan_term.py`'s per-frame output, which it saves as one .npy per trip). So the plan term at entries and exits is
an upper bound; the car-vs-paint column is the better guide.

**Tried and not kept** (measured, but each needs MapDrive retuned with it and an in-game check):
- **Shorter fillets, `FILLET_LEG` 12 m** (with `FILLET_REACH` kept at 20 m as the corner-to-junction distance).
  Drift planned line > 1 m: 4.25% -> 3.88% (15 m: 4.00%, 10 m: 3.92%). Left -15..-10 m: 0.61 -> 0.28; +10..+15:
  0.49 -> 0.20; right -10..-5: 1.10 -> 0.54. On the trips, plan > 0.5 m on exits went from 64.1% to 43.2%. But
  turn entries got worse: car > 0.5 m off its lane went from 68.4% to 80.9%. At (-1521.4,-243.0) the line swings
  about 1 m out before the right turn. The cause: a fillet 12 m long starts where the lane line is already smoothed
  round the corner (offset, unfold, `_smooth_line`), so it starts with a kink. Four mapdrive tests fail with it:
  `test_clamp_corners` and `test_far_corner_beside_the_flank` (their unclamped line no longer reaches the kerb they
  test against), `test_live_starts` (SL2b no longer has a `sharp_corner`), and `test_no_oscillation` (turn, tau
  0.15, gain 1.6: peak 0.0227 rad/s, limit 0.02). 16 m and 15 m fail 3 and 5 tests; 20/12, 12/20 and 20/15
  (entry/exit) fail 3 to 5.
- **Clamp settling earlier, `CORNER_ZONE` 15 -> 12 / 10**: fails 2 / 4 tests (`test_no_oscillation`,
  `test_far_corner_beside_the_flank`; at 10 also `test_clamp_corners` and `test_give_way_and_lights`).
- **Left radius 12**: drift no better than 15 (4.23%), and 3 tests fail.

**Tests**: lane_numbering 29 + the cloud-router case (as baseline), lane_slots 20, freeway_lanes 7, **mapdrive 38/38**
(the known `test_collision_routes` failure now passes: 029's plan is within 0.6 m of its lane's centre where map1010b
touched the pavement), validate_lanes 0 issues, lane_parity unchanged, map tests 160 passed. One test changed:
`test_clamp_corners`' median case is now a 75 deg left instead of 60 deg. At 60 deg the 15 m fillet no longer reaches a 4 m
median's kerb, so the clamp had nothing to push there; at 75 deg it reaches 0.79 m in, and the case still tests the
push. Amy, please check this change: it alters a test's scenario, not what the test checks.

