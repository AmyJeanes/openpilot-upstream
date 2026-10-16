# Cloud task: map lane geometry from the paint (drift fixes)

You are a Claude Code cloud session. You see only GitHub, not Amy's PC (no GTA, no GPU, no live services).

Repos:
- `AmyJeanes/openpilot-upstream`: read, and push branch `lanegeom-cloud` only. Clone it shallow and single-branch:
  `git clone --depth 1 --branch lanegeom-cloud --single-branch`. The full repo is very large.
- `AmyJeanes/gta5-distill` (private): read only, branch `cloud-data`, data under `cloud/` (see `cloud/MANIFEST.md`).

The session may start on another branch. Work on `lanegeom-cloud`, which is `gta5` plus this brief.

## Background

Amy drives GTA V through a bridge: `openpilot/tools/sim/bridge/gta5/`. Our map is an OSM pbf in a fake lat/lon frame,
built from GTA's path nodes (YND): `map/ynd_to_osm.py` → pbf, then `map/osm_to_roads.py` / `map/osm_lanes.py` →
`roads.json` and `lanes.json` (lane lines, per-lane geometry, hatched areas). Valhalla routes on the pbf (`map/router.py`).
A rule-based "map driver" (`gta5_mapdrive.py`, `MapDrive`) drives from that geometry. Its offline harness is
`mapdrive_sim.py`. Training labels for the driving model come from the same geometry, so geometry errors become label
errors.

**The problem:** in game, the map driver sometimes drifts into the neighbouring lane while it is meant to stay in its
lane. An investigation on 16 Oct (`cloud/drift_20261016/` on `cloud-data`) used 97 map-driver trips (161 km, live18
pbf). At each frame it split the car's offset from the **painted** lane centre into three parts: control (car vs its
planned path), plan (planned path vs the map's target lane centre) and map (map target centre vs paint centre). The
paint comes from the student segmentation of top-down captures (`paint_lanes_confident.json`, 93% precision; format in
`cloud/mapfuse_20261014/FORMAT.md`).

| Time more than 1 m off the paint lane centre, outside lane changes | map | plan | control |
|---|---|---|---|
| junctions included (5.9% of time) | 57% | 40% | 3% |
| more than 20 m from junction marks (3.0%) | 72% | 24% | 4% |

Control is ruled out (`path_dev` p99 under 0.35 m). GTA's city lanes are 4.9–6.3 m wide (median 5.7 m), so only the
larger map errors show up as drift.

### Cause 1: lane centres offset from the paint (the largest)

- The map's target centre is more than 1 m off the paint centre for 2.5–3.8% of driven time.
- On matched paint lines, the map's nearest line is more than 1 m away on about 9% of their length (about 12% on
  primary roads).
- Examples: (901.4,-695.3) 2.0 m; (530.7,-498.4) 2.9 m; (125.6,-1367.1) 2.7 m; (-13.9,274.6) 1.9 m.

### Cause 2: the target line steps sideways where links or lane counts change

- 4.9 steps per km with the same target lane and no lane change. 142 of them are away from junctions (0.9 per km), and
  47 of those are bigger than 2 m.
- Where paint covers both sides of a step, 25 of 59 are spurious: the paint lane did not move.
- `osm_lanes.py` `_ease_jogs` (around line 2013; `JOG_MIN`, `BLEND_M`, `TAPER_M`) eases each step into an S-weave across
  a dashed line.
- Confirmed cases:
  - (1079.5,-1546.2): link 402:719→738 has 3 lanes and splits at node 402:738 into GTA's parallel 1-lane chains. The
    route's chain 402:738→763→434:9 runs diagonally, so the car swings about 6 m over 60 m.
  - The Strawberry→Olympic exit at (256.7,-1176), link 432:982, on 3 trips: the ramp's link starts at the motorway's
    centre node, so the target swings about 9 m and the car crosses the dashed line diagonally.
  - (-378,-2062), the diverge 367:468→399:5, on 2 trips: the path crosses the solid gore line with no lane change.
  - (1190.9,441.6): a 2→1 lane drop at 530:200.
  - (68.7,-1146), on 3 trips: an 8 m step between link 432:1433 (one lane each way) and the 2-lane chain 432:1031.

### Cause 3: plan logic at turn entries and exits

- The planned path is more than 1 m off the map target 2.4% of the time overall, but 0.8% away from junctions. Median
  heading change 22°, about 6 m/s.
- Code: `osm_lanes.py` `_fillets` (around line 1870, `FILLET_REACH` 20 m), `gta5_mapdrive.py` `_clamp` (around line
  1813), and the plan's lane-number steps.
- Examples: (867.3,-1745.8) and (-1521.4,-243.0), each on 2 trips; (1135.2,-3114.7).

Event list: `events_ctx_1.0_j20.csv` (24 events away from junctions), `events_ctx_1.0.csv` (all 100),
`jumps.csv` (792 target steps), `plots_j20/`. Per-frame data: `frames.npz` / `ilat.npz` / `derived_j20.npz` and the
scripts that made them (`drift_frames.py`, `attribute.py`, `jumps.py`; paths inside are Amy's, adapt them).

Known caveats: the paint layer misreads stacked roads (e.g. (478.5,-482.4) is an interchange artefact), and very
narrow "lanes" under 4 m are turn bays or false lines. Don't fit geometry to paint there.

## Amy's rules (binding)

- Map correctness comes first. Generator fixes that make the map more accurate need no A/B, but every change must be
  measured, and nothing goes live without Amy.
- Hatched areas, island and gore decisions and review approvals are hers. Never add, drop or move a hatched area. If one
  looks wrong, list it.
- Only paint seen by 2+ tiles (or Amy's own labels, which override the student) may move geometry. Where paint and the
  game files disagree, prefer the game files' lane counts and flag the place.
- Don't change routing (the pbf's ways, access or turn rules) in this task. Geometry only, except where an exit or hop
  anchor needs a node moved; if so, list every such node.

## Step 0: preflight (stop on failure)

Record pass/fail with one line of evidence in `CLOUD_RESULT_LANEGEOM.md` under "Preflight":

1. Repo: branch `lanegeom-cloud` (parent `gta5`). Note both hashes.
2. Data: `cloud-data`. Check sha256 against `cloud/MANIFEST.md` for `live18/map1/*`, `live19b/*`,
   `mapfuse_20261014/*`, `gamefiles/*` and `drift_20261016/*`. Assemble the live map: live18's pbf, `roads.json`,
   `paths.jsonl` and `nodes.npz`, plus live19b's `lanes.json`.
3. Dependencies from PyPI only (numpy, shapely, pyosmium, pyvalhalla, pytest...). If an install is blocked, record it
   and design around it.
4. Rebuild check: rebuild `lanes.json` from the live pbf with the repo's own build (see `map/README.md`, and
   `cloud/live19b/build19b.sh` for the exact flags), and diff it against live19b's. It must be identical, or explain why
   not. This is your baseline build.
5. Baseline scores, recorded:
   - Map vs paint: share of matched paint length with the map's nearest line more than 0.5 / 1.0 m away, by road class.
   - Target steps away from junctions, per km and the spurious share.
   - The drift re-score: per frame, the map target centre vs the paint centre over `frames.npz`'s poses (the "map" term).
     Re-derive it from `frames.npz` so the same scorer can run on your rebuilt map.
   - Tests: `test_lane_numbering.py`, `test_lane_slots.py`, `test_freeway_lanes.py`, `test_mapdrive.py`,
     `map/validate_lanes.py`, `map/lane_parity.py`. On Amy's PC, test_mapdrive has 1 known failure,
     `test_collision_routes`.
6. Push access: commit only `CLOUD_RESULT_LANEGEOM.md` and push.

## Work (in order; commit, push and update the result file after each)

### 1. A geometry scorer

`map/paint_score.py`: the step 0.5 scores as a reusable tool (a map folder plus paint → a report and a per-place list).
Every later step reports these numbers before and after. Include per-event status for the 24 j20 events and the 5
recurring places.

### 2. Continuous lane positions across nodes (cause 2)

Make each lane's lateral position continuous across link boundaries. A lane that is added or dropped is counted from the
side it opens or closes on, not by re-centring the cross-section. Where GTA's lane counts on consecutive links make a
jump that the paint doesn't show (the 25 spurious of 59), the geometry must not step. Where the paint does move, the
step should follow the paint, over the paint's own taper length.

Keep `_ease_jogs` only for real jogs, and make its taper follow the paint where there is paint. Tests: synthetic
2→3 / 3→2 / 1+1→2 links (lane opening left, opening right, dropping), plus the real places above. Report steps per km,
the spurious share, and the per-event status.

### 3. Exits and lane-chain hops anchored where the paint diverges (cause 2: Strawberry, (-378,-2062), (1079,-1546))

- An exit lane should run as the kerb lane and leave it over the real gore. That means starting where the paint (or the
  game files' gore and hatched area) diverges, not at the motorway's centre node.
- Lane-chain hops (a multi-lane link splitting into GTA's parallel 1-lane chains) should continue each lane straight on
  from its own position, rather than starting from the centre node.

This touches every motorway exit and every chain split, so report how many places change, the largest moves, and before
and after scores. Never cross a hatched area (`lanes.json` kind `hatched`) with a lane centre.

### 4. Lane centres from paint (cause 1)

Where confident paint lines bound a lane on both sides (2+ tiles, or Amy's labels), set the lane centre and width from
them. In OSM terms that is the way's `placement` / `width:lanes`, or the lanes.json section offsets, whichever the
build reads. Choose the smallest change in the build that does it, and say why.

- Where only one side has paint, shift the lane to match it, and keep the width unless it falls outside 2.7–6.5 m.
- Where paint is missing or ambiguous (stacked roads, lanes under 4 m), leave it alone and list it.
- The phase-1 rule that image paint goes through an explicit "applied changes" file rather than `paint_survey` (see
  `cloud/phase1/PLAN.md` summary) applies here too. Write the applied offsets to one reviewable file with stable keys,
  and have the build read it.

Report the before and after scores by road class and the largest changes.

### 5. Turn fillets that settle in the target lane (cause 3)

Shorter or lane-aware fillets (`_fillets`, `FILLET_REACH`), and a `_clamp` that settles into the target lane before the
paint's lane line starts. Report the plan term at turn entries and exits before and after, on the 3 example places and
on `mapdrive_sim.py` routes through them. That needs Valhalla: build tiles from the pbf with pyvalhalla (`valhalla_build_config` is the venv's console script; `valhalla_build_tiles` and `valhalla_service` are under `site-packages/valhalla/bin`) and run it on port 8002, so `mapdrive_sim.trip_route` and the router tests work. If Valhalla is impossible, use made routes and say so.

### 6. Candidate build and hand-over

Build a candidate map folder (`live21` naming: `lanes.json`, plus `roads.json` if it changed, and the pbf only if step 3
moved nodes). Put it under `candidates/live21/` on the branch if it's under 50 MB compressed. Otherwise give the exact
build command Amy runs.

Write `candidates/live21/README.md` with:
- what changed, with numbers;
- the 10 places Amy should drive in game, with coordinates and what to watch for, including the Strawberry ramp,
  (-378,-2062), (1079,-1546), (68.7,-1146) and (530.7,-498.4);
- anything you flagged for her (hatched oddities, paint vs game-file disagreements, moved nodes).

## Done means

- Branch `lanegeom-cloud` pushed, one commit per step, with the rationale in each message.
- `CLOUD_RESULT_LANEGEOM.md` holds the preflight, per-step before and after scores, the test results against the
  baseline (no new failures), and open questions.
- No regressions: lane-numbering, lane-slot, freeway-lane and map-driver tests no worse than the baseline.
  `validate_lanes.py` clean, or new findings explained.
