# live21d candidate: place 7's exit and the lane fit rules

live21c (live21b with the 7 lane fits rejected on the Lane Fit Review page put back) plus this branch's code. **Not
live.** Built and checked offline; nothing here has been driven in game yet.

## Files

Built under `/mnt/e/gta5map_candidates/live21d/` (not in the repo: lanes.json and roads.json rebuild from the pbf).

| file | sha256 | what |
|---|---|---|
| `gta5.osm.pbf` | `85959dad08c66afa…` | live19b's pbf baked (`lane_geometry.py`) with `lanegeom.review.json` |
| `lanegeom.review.json` | `64b808f24ccbb918…` | **in this folder**: live21's lane geometry less 33 fits (below); the 14 kept diverges unchanged |
| `roads.json` | `6e1ba290b2a82477…` | `osm_to_roads.py` on this branch, `--hatched` live19b's hatched19b.json |
| `lanes.json` | `47ff609f6803ae87…` | the same run |

No nodes, ways, access or turn rules change: the 348 scan trips route identically on live21d's Valhalla tiles and the
live ones, so the live tiles and nav map tiles stay.

## What changed

**Place 7 (code, `osm_lanes.merge_side`).** GTA splits the road at (-450, -1619) into two overlapping 2-lane chains.
Where the right chain parts at 399:757, its left lane goes on to the main road (a 1-lane link 50 deg off the route's
heading) and its right lane to the exit. The split was outside `MERGE_SPREAD` (40 deg), so lanes were matched by
place, and the exit way's line (on the chain's line, between its lanes, with a kept diverge) took the left lane: the
map driver moved left into the main lanes and cut back across the solid line into the exit. A one-way road that
splits into two one-way roads whose lanes add up (2 = 1 + 1), with no other road at the node, is now a fork at any
angle: the branch on the right takes the right lanes. In the offline drive (mapdrive_sim) the car now keeps the right
lane into the exit. It still clips the first metre or two of the solid line where it begins: it holds the chain's
right-lane centre rather than drifting right as the exit lane opens (the full fix is the road gaining a tapered exit
lane in the map). Scan: the lane maps change at this node only (4 of 348 trips).

**Lane fit rules (code, `lane_fit.py`).**
- A fit needs at least `JUNCTION_CLEAR` (2) matched stations outside junctions' areas (roads.json's areas holding a
  node where 3 or more ways meet); a way fitted from a junction's paint alone is listed instead.
- No lanes across the centre line: a two-way way's centre line may not move more than half the lane beside it, and a
  one-way way's lanes may not all move onto the oncoming side of its line (left of it driving on the right: the
  car-park aisles mapped one way whose paint is a two-way aisle's).

A stricter junction rule (no stations in junction areas at all) was tried first and rejected: it dropped 135 fits,
and on the crops about half of those it dropped by more than 0.75 m were right (lane paint runs on into the junction
polygons on approaches). The junction stations' shifts agree with the clear stations' on most ways.

**The 33 fits dropped** (live21d keeps 1,350 of live21's 1,383; the rules were judged by running `lane_fit.py` with and without
them on live19b's map with the local 2-tile paint, `paint_lanes_confident.json`; live21's own tags are kept for every
fit that stays):
- the 7 live21c rejected: by rule 496:839>496:861 (centre line 3.1 m), 561:92>561:98 and 561:46>561:49 (oncoming
  side), 562:112>562:118 (junction paint alone); by Amy's review answers 653:114>653:117 (junction), 491:43>492:162 and
  398:570>398:566 (unsure, not applied). 653:114's error is the size of the shift, not junction paint: its stations
  outside the junction say the same.
- 26 fitted from junction paint alone, none moved more than 0.72 m. The rule dropped 30; the 16 that moved 0.4 m or
  more went on the live21d Map Review page, where Amy confirmed 12 and kept 4 (464:339>464:338, 493:275>493:218,
  494:315>494:306, 494:832>494:830). Nothing in the stations, shift, paint or road class separates those 4 from
  neighbouring ways she dropped with the same evidence, so they're kept by name (`AMY_KEEP` in the build's filter),
  as live21c's rejects were. The first build without them (37 dropped) is kept beside as `*.v1`.

Scan (348 trips, live21c vs live21d, same code): lane centres move at most 0.46 m, all at dropped fits, and no lane
map changes. With the code change too, only place 7's node changes lane maps (4 trips).

## Rebuild

```
lane_fit.py MAP --paint paint_lanes_confident.json --out OUT          # MAP: live19b's pbf + roads.json; with and without the rules
lane_geometry.py live19b/gta5.osm.pbf lanegeom.review.json --out live21d/gta5.osm.pbf
osm_to_roads.py live21d/gta5.osm.pbf live21d/roads.json --lanes live21d/lanes.json --frame gta5 --hatched hatched19b.json
```

## Check in game

1. **Place 7**: `-475.0,-1648.0,20.0,317,1` to `-372.8,-1430.5`. The car should stay in the right lane and take the exit
   without moving left first; watch the start of the solid line.
2. **Alta St** (496:839>496:861, (96, -108)) and **place 9's start**: unchanged from live21c (Alta's fit stays dropped).
3. A junction mouth with a dropped fit Amy confirmed, e.g. 495:282>495:284, and one she kept, e.g. 494:315>494:306.
