# live21 candidate: lane geometry from the paint

Built in the cloud from `lanegeom-cloud` (steps 2-5 of `CLOUD_TASK_LANEGEOM.md`; details and every number in
`CLOUD_RESULT_LANEGEOM.md` at the repo root). **Not live.** Nothing here has been driven in game.

## Files

| file | sha256 | what |
|---|---|---|
| `lanes.json` | `cb7d0e13dcd8fd5e…` | rebuilt (69,727 lines; live19b 69,666): 8,732 lines differ from live19b's |
| `roads.json` | `c3f309f2274c580c…` | rebuilt (32,401 ways, 26,620 junction areas) |
| `lanegeom.json` | `f34f741c884caa8b…` | **new, goes beside the pbf**: 1,383 way tag overrides (step 4) and 14 kept diverges (step 3), each with its evidence |
| `gta5.osm.pbf` | not included | **unchanged**: live19b's, sha256 `65e320c418bfd881…`. No nodes moved, no ways, access or turn rules changed, so Valhalla's tiles stay as they are |

Install: copy the three files into the live map folder (beside `gta5.osm.pbf`, `paths.jsonl`, `nodes.npz`). The bridge
needs this branch's code too: `OsmLanes.load` reads `lanegeom.json` (without it the route geometry ignores step 4's
tags and step 3's kept diverges, and lanes.json/roads.json would disagree with the planner). `lane_match.map_hash`
includes the file, so the junction and stop-line caches rebuild themselves.

Rebuild (8 min; gives these files byte for byte, checked in the cloud from the branch head):

```
python openpilot/tools/sim/bridge/gta5/map/osm_to_roads.py MAP/gta5.osm.pbf MAP/roads.json --lanes MAP/lanes.json \
  --frame gta5 --hatched gta5-distill/cloud/live19b/hatched19b.json     # MAP holds live19b's pbf + this lanegeom.json
```

## What changed (live19b -> live21)

Scored with `map/paint_score.py` against the 2-tile confident paint, the drift frames of 16 Oct (97 trips, 159,577
frames), and the game files' own paint (independent of the student). Before = live19b on the current `gta5` code.

| measure | live19b | live21 |
|---|---|---|
| map lines vs paint, > 1 m (same samples, all classes) | 8.6% | 8.5% (4.4 km of paint nearer by 0.5 m+, 1.7 km further) |
| ... primary | 9.7% | 9.2% |
| ... trunk | 5.5% | 5.3% |
| map lines vs game-file paint, > 1 m (out of sample) | 7.1% | 7.0% (2.9 km nearer, 1.2 km further; primary 7.9% -> 7.4%) |
| drift: lane centre where the car is (`section_at`, what target_right now logs) > 1 m | 3.61% | 3.36% |
| ... > 1 m, more than 20 m from junctions | 2.51% | 2.30% |
| drift: the planned line > 0.5 m / > 1 m | 11.45% / 4.46% | 10.86% / 4.18% |
| target steps away from junctions (`section_at`) | 58 (0.36/km), 2 spurious | 52 (0.32/km), 1 spurious |
| freeway diverges/merges with paint: branch lanes' mean distance from the paint lane centre | 0.50 m | 0.39 m (lanes over 1 m off: 17.8% -> 12.3%) |
| lane-centre metres inside hatched areas at the changed diverges | 4.4 | 0.0 |

By step:
- **Step 2, lanes carry on across nodes**: a lane that begins or ends at a node tapers across it (6 m either side
  where a lane opens, 9 m before / 3 m after where one ends), the paint's own window, instead of stepping or holding
  the old layout 30 m past the node. Inside the old tapers the lane centre is more than 1 m off the paint 22% of the
  time where a lane opens (was 52%) and 39% where one ends (was 49%).
- **Step 3, exits and chains part where the paint parts**: 109 of 583 freeway diverges and merges move (97 by more
  than 1 m). The branch's lane runs on in the road's kerb lane until GTA's link has crossed out past it, instead of
  swinging in to the road's middle node. 14 places are kept as they were because the paint prefers the old line
  (`lanegeom.json` `diverges`). Route geometry only: lanes.json's lines don't change in this step.
- **Step 4, lane centres and widths from the paint**: 1,383 ways get `width:lanes` / `width` / `placement:offset` tags in
  `lanegeom.json`: centre and width where the paint shows both sides of a lane, a shift with the width kept where
  it shows one. Primary 641, residential 536, motorway 95, unclassified 61, motorway_link 28, trunk 20, trunk_link 2.
  51 moved more than 1 m, 10 more than 2 m.
- **Step 5, left-turn fillets**: radius 20 -> 15 m (driver code only). Left-turn entries, 15-10 m before the corner:
  median planned line vs paint 0.61 -> 0.43 m. Right turns are unchanged.

The 24 drift events and the brief's places (`cloud_lanegeom/live21/events.csv`): 5 better, none worse, the rest the
same. Better: (530.7,-498.4) 3.03 -> 1.81 m, (1190.9,441.6) 3.36 -> 2.79, (-127.7,-1182.5) 3.51 -> 2.49,
(-388.5,-975.8) 2.84 -> 2.40, all by the planned line. The Strawberry ramp can't be scored from these frames
afterwards: the cars there followed the old dip, so the new target's paint lane isn't bracketed from their poses.

## 10 places to drive

| # | where | what changed | watch for |
|---|---|---|---|
| 1 | **Strawberry -> Olympic exit ramp**, (256.7, -1176) heading east, take the exit | step 3: the exit lane (432:955 > 432:982) runs on in the kerb lane, between the paint lines at y -1170 and -1176.3, and leaves along the ramp past x 250. It moved 4.7 m and no longer dips to y -1177 towards the motorway's middle node. Along the branch, more than 1 m off the paint: 24% -> 6% | the car stays in the kerb lane up to the gore and doesn't nudge left at the node; labels' target_right without a jump |
| 2 | (-378, -2062), the parallel one-lane chains (trips 068/082) | **not fixed**: the paint disagrees along the chain links (shift +2.1 m, 20% agree), and the surveyed links are kept as the game files have them | the 2.5 m target offset is still there: a candidate for Amy's labels |
| 3 | (1079.5, -1546.2), chain split, keep the 2-lane side | step 3: the 2-lane side (402:738 > 402:762) is 1.6 m nearer the paint (0.55 -> 0.47 m). The planner was already right here (planned line 1.10 m, recorded target 2.91 m) | straight on through the split, no hop |
| 4 | (68.7, -1146), link 432:1433 onto the 2-lane chain 432:1031 | **not fixed**: the 8 m step (static 3.46 m, planned line 2.66 m). The ways here are stacked, surveyed or ambiguous, and one link (432:1067>432:1065) moved 0.67 m | whether the step is still felt in game; label it if so |
| 5 | (530.7, -498.4), stacked interchange exit | step 3: diverges 497:5>496:57 / 497:8>496:64 moved 10-11 m. Planned line 3.03 -> 1.81 m off the paint | the exit lane leaves from the kerb lane; check the lower and upper decks both |
| 6 | (867.3, -1745.8), trip 018: left then right through the junction | step 5: left-turn fillet radius 15. In mapdrive_sim, the turn entry's plan term over 0.5 m went from 67% to 54% and the car off its paint lane from 55% to 42%. Step 4 moved 401:446>401:489 by 0.71 m | the left turn clears the median nose and settles in the lane out sooner; no swing wide |
| 7 | (-378.0, -1489.8), motorway 431:51>431:68>431:89 | step 4: the 2-lane way moved 3.0 m right onto both its paint lines (edges -2.75/2.75 -> 0.29/5.40) | centred between the painted lines; no step where the moved ways meet their neighbours |
| 8 | (96.2, -108.0), residential 496:839>496:861 | step 4: the largest move, 3.1 m. Two-way, 1+1 lanes, the lane line onto the paint (edges -5.5/0/5.5 -> -2.85/3.14/8.64) | that the road really is offset here (the kerb side has no paint: a wrong match would put the right lane on the pavement) |
| 9 | (791.7, -633.3), diverge 465:521>465:543 | step 3 moved it, but **further from the paint**: 0.65 -> 0.97 m, with paint on only 16% of the stretch (too little to decide) | which line the car takes; if worse, add it to `lanegeom.json` `diverges` to keep it as it was |
| 10 | (1190.9, 441.6), 2 lanes to 1 (trip 056) | step 2: the lane ending tapers 9 m before to 3 m past its node. Planned line 3.36 -> 2.79 m | a smooth merge, no step at the node |

## Flags

- **Hatched areas**: none added, dropped or moved: the same 331 from `hatched19b.json`, with the same build flags
  ("50 listed, 8 held, 0 of those not found, 75 given as rings"). Step 3's moved lanes are checked against them and
  never run further into one. One oddity: at **(1005, -935)** the exit's straight run-on would cross the hatched gore
  diagonally (a first version of step 3 did that), so it is left as it was.
- **Paint vs game files disagree** (game files kept, as the rule says): **1,308 ways** whose lanes the build measured
  from the game files (`source:width=survey`) where the 2-tile paint has them 0.25 m or more off. 451 more ways have
  lane counts that disagree (the paint shows a line inside a lane). 402 have paint that disagrees along their length. All listed with their
  numbers in `cloud_lanegeom/step4/listed.csv`, with the rest of the 7,135 ways left alone: 2,838 stacked, 1,401
  with a lane over 6.5 m (a line missed), 374 with a lane under 4 m, 185 without a unique key, 174 where the lanes widen
  or move along the way, 2 the tags can't place.
- **Moved nodes**: none. The pbf is byte-identical to live19b's, so routing is unchanged.
- **Diverges kept by the paint** (14) and the two that got further from the paint with little paint
  (465:521>465:543 at (791.7,-633.3), place 9; 399:790>399:824 at (-424.2,-1575.7): 0.22 -> 0.48 m, paint on 29%).
- **The brief's offset places all left alone**: (901.4,-695.3) has a lane 6.9 m wide; (530.7,-498.4), (134.0,-492.0)
  and the Strawberry ramp are stacked; (-13.9,274.6) has a lane 3.8 m wide; (-378,-2062) see place 2;
  (125.6,-1367.1) has too little 2-tile paint.

## Open questions for Amy

1. **Amy's labels** (150 reviewed tiles) are not used: they're in tile pixels, and the projection to the world isn't
   in the cloud data's tools. With it, places 2 and 4 and the stacked interchanges could be fitted.
2. **Shorter fillets** (`FILLET_LEG` 12 m) and an earlier clamp settle (`CORNER_ZONE` 12 m) are better against the
   paint (planned line > 1 m 4.25% -> 3.88%; turn exits in mapdrive_sim, plan > 0.5 m 64% -> 43%) but break mapdrive
   tests, `test_no_oscillation` among them, and swing the line out before the right turn at (-1521.4,-243.0). Worth
   retuning together with the lane line's own corner smoothing, in game. Not in this candidate.
3. Step 5 changed one test scenario: `test_clamp_corners`' median case is now a 75 deg left (at 60 deg the 15 m
   fillet no longer reaches the median, so the clamp had nothing to test). Please check this change.
4. lanes.json still draws the exit links where GTA has them (from the road's middle node). Step 3 moves the route
   geometry the car follows, not the drawn lines. Redrawing them would need the same rule in `osm_to_roads`.
5. `test_lane_numbering::test_live_freeway_merge_exit_and_splits` fails in the cloud (the cloud Valhalla build routes
   3 km where the test wants under 300 m) before and after. Please run it on the PC. `test_nav_msgs.py` (needs
   pycapnp) wasn't run.
