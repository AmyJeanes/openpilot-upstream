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

## Item 1: lanes that end, option B (`b22ec06`)

**What changed** (`selfdrive/navigation/lane_guidance.py`): `MapboxRoute` compares each run of an intersection's lanes
with the run before it on the same road (no turning maneuver between; a maneuver's own node lists the road into it).
Where the later lists fewer lanes, `ends_side` aligns the two lists lane by lane from the left and from the right by
their arrows (indication sets, bus lanes as bus lanes); the side whose alignment holds keeps its lanes, the other's
lanes end; both or neither: ambiguous, nothing. That makes a move of kind `ends` (side = the side that keeps, `before`
= lanes before) at the first intersection listing fewer (`ends_at` 1.0 of the way there from the last listing more).
Its targets are the lanes that carry on, counted from that side, only while the model counts more lanes than the list
after and as many as the list before. It's timed and asked as a maneuver's changes are (same `_due`, request and
NavDesire machinery), `done_before_ends` 30 m before it, request reason `laneEnds`. Never against a maneuver: not away
from the lanes of a move before it (as for any move), not out of a lane a maneuver within `ends_guard` (150 m) past the
end wants, and an `ends` move never takes over a maneuver's change as "the earlier's" (the reason stays the
maneuver's). Not in the lanes UI. Gates (each a `Tune` switch, found below): `ends_count_match`, `ends_at_maneuver`
(off: not where the shorter list is a maneuver's own node or approach), `ends_valid_only` (on: not where only lanes
not valid at the junction before drop out).

**navd_sim** (`tools/sim/bridge/gta5/navd_sim.py`, judge untouched): every request with reason `laneEnds` (either nav)
is checked on the full map when it's made: the car's lane (truth) followed through the lane maps to the request's target
+ 50 m. Right if it ends; otherwise a false positive by cause, first match of: `bay` (with --nav mapbox: the end within
60 m of a Mapbox turn/fork/exit), `other_side` (a lane ends there, not the car's), `gain` (lanes gained from 200 m
before the car to there: an on-ramp, a lane added), `partial` (none of these: Mapbox's lists differ where the road's
lanes don't, as with turn:lanes on only part of a road). Printed as a third table; `lane_end_asks` per trip in --out.

**Tests** (`test_lane_guidance.py`, +8): `ends_side` (left, right, ambiguous both ways, a bus lane not lining up); the
move (place, `ends_at`, off, ambiguous lists); asked in good time from the lane that ends (one change, laneEnds, the
timing formula) and nothing from a lane that carries on; the count gate; never against a right turn 100 m past the end
whose lane the car is in (and asked when the turn is 1 km on); no lane end at Marsh Lane's node (the bay fixture
`turn_bay_right`; there with `ends_at_maneuver`); `lanes_vary_bus` (4 -> 2 lanes, the bus lane's side losing them; 4 ->
3 losing only the bus lane: no move; gains aren't ends); `lanes_added_midblock` (a right-turn lane at the junction
before: no move).

### Measurements (navd_sim, Mapbox path, live21b, the 46 trips; same seeds, trips and heads as NAVMAPBOX)

Final rule against the baseline (seed 0; seed 1 below):

| run | in lane: baseline -> option B | turns | forks / exits | straight on | LC req / done / failed | forced merges | lane-end asks (right / false) |
|---|---|---|---|---|---|---|---|
| mapbox / n16 / truth | 247 -> 247 | 77 -> 77 | 91 -> 91 | 79 -> 79 | 159/106/46 -> 159/106/46 | 20 -> 20 | 0 (0 / 0) |
| mapbox / n16 / model:0.85 | 244 -> 244 | 74 -> 74 | 91 -> 91 | 79 -> 79 | 162/102/51 -> 162/102/51 | 22 -> 22 | 0 |
| mapbox / finetuned / truth | 254 -> 254 | 76 -> 76 | 98 -> 98 | 80 -> 80 | 232/154/70 -> 232/154/70 | 22 -> 22 | 0 |
| mapbox / finetuned / model:0.85 | 252 -> 252 | 75 -> 75 | 96 -> 96 | 81 -> 81 | 238/146/82 -> 238/146/82 | 22 -> 22 | 0 |

In-lane at the commit points is unchanged and so are the forced merges: on the GTA trips the final rule makes no `ends`
move at all, so the Mapbox path drives exactly as before (identical numbers). With the gates off it makes 10 and asks
1-2 times per run, every one false. What I tried (n16 and finetuned, truth and model:0.85, seed 0):

| variant | `ends` moves on the 46 trips | asks per run (n16 truth / 0.85; finetuned truth / 0.85) | right | false by cause | in lane, forced merges |
|---|---|---|---|---|---|
| first cut: count gate on, no maneuver or valid-lane gates (and before a fix: lists across a turn compared) | 18 (10 after the fix) | 2 / 0 (n16 only run) | 0 | 2 `bay` | 248 / 244, 20 / 22 |
| count gate off | 10 | 2 / 1 (n16) | 0 | 2 / 1 `bay` | 248 / 244, 20 / 22 |
| all gates off (count, maneuver, valid-lane) | 10 | 2 / 1 / 2 / 1 | 0 | all `bay` | 248 / 244 / 254 / 252, 20 / 22 / 22 / 22 |
| all gates off, `ends_at` 0.5 (half way back to the last longer list) | 10 | 3 / 2 / 3 / 2 | 0 | `gain` 2 / 1 / 2 / 1, `other_side` 1 each | same |
| **final** (count gate, no maneuver node/approach, a valid lane must drop out) | **0** | 0 | 0 | 0 | 247 / 244 / 254 / 252, 20 / 22 / 22 / 22 |

The false-positive breakdown, by what the 10 ungated candidates are (the static list, `MapboxRoute` on each trip's
emulated response, against the full map's lane maps within 250 m before and 60 m after):

| cause | candidates | example | removed by |
|---|---|---|---|
| a lane only for a turn at the junction before (fewer listed past it, none of the car's ended between: turn:lanes covering a road's lanes only up to the junction) | 7 (3 places) | (202.6, 194.9): `le st* st/ri*` into the junction, `st* st/ri*` after; the map's left-turn lane ends at the junction (`[None, 0, 1]`), the shorter list starts about 160 m on, at (53, 258) | the count gate in the drive (the car already counts 2); `ends_valid_only` statically |
| a maneuver's approach listing fewer than further back (the bay case, here a left-turn lane `le*` alone on the last 60 m) | 3 (1 place: R06, R18, R38's left turn at (377, -2179)) | the 1-3 asks per run above, all at (287, -2131) | `ends_at_maneuver` off |
| the maneuver's own node listing fewer (`turn_bay_right`'s Marsh Lane) | 0 in GTA (the fixture: 1) | | `ends_at_maneuver` off |
| a lane ending on the other side, lanes gained / on-ramps | 0 as candidates; they show up as causes only when the end is placed earlier (`ends_at` 0.5 at (287, -2131): `gain` 2, `other_side` 1), as the window then takes in the map's drop-add-drop there | | `ends_at` 1.0 |
| the car's lane really ending | 0 | | |

Why option B can't see GTA's lane ends: the emulator lists lanes only where our map has turn:lanes, on the
intersections of the way into a junction (as Mapbox does), so the road past a lane drop almost never has a list of its
own to compare. The 20 forced merges on the Mapbox path (n16 / truth, seed 0), each with Mapbox's lists from 250 m
before to 60 m after it: 3 have no lanes from Mapbox anywhere near (MD029 at (-195.5, -489.4), R10 (-253.0, -548.8),
R31 (-1418.2, -584.4)); 3 have lists before it with the same count and none after (R08/R30 at (-1788.5, 81.5), R39
(-154.9, -804.7)); 6 are lanes the map ends at a junction whose own list still has them (MD014 (796.4, -1442.4)
`le/st* st*`, R04, R05/R16 (-995.9, -207.8) `le st* st*` where the map keeps only the right lane, R23 (512.0, 83.5),
R28; in R23 and R28 Mapbox marks the lane turn-only and the straight-on move is there, the change just didn't happen);
8 are at or next to maneuvers (MD029 (-752.1, -40.9), FX1/R15 (1107.7, -1644.4) 84 m past a turn, MDT (121.4,
-1261.8), R24, R27 x2, R34). None has a shorter list after the drop on the same road. On the real-schema fixtures the
rule fires where it should: `lanes_vary_bus` (4 -> 2 between junctions, the lanes on the bus lane's side), and with
the gates off on `lanes_added_midblock` (a turn lane at the junction before: false, `partial`) and `turn_bay_right`
(the node: false, `bay`).

Hinging on a known map issue: the one place the ungated rule asks, (287, -2131) before the left turn at (377, -2179)
on R06/R18/R38, is where the map's lanes go 2 -> 1 -> 2 -> 1 within 50 m (drops at 174 and 225 m, a lane added at 199
m on R06): it looks like one of live21b's chain joins; worth a look in game (check 3 below).

Seed 1 (final rule; the Mapbox path again identical to its baseline):

| run (seed 1) | baseline (= NAVMAPBOX's seed 1) | option B, final | all gates off | forced merges (all three) | asks, all gates off (all `bay`, at (287, -2131)) |
|---|---|---|---|---|---|
| mapbox / n16 / truth | 247 | 247 | 246 | 18 | 2 |
| mapbox / n16 / model:0.85 | 244 | 244 | 244 | 19 | 1 |
| mapbox / finetuned / truth | 256 | 256 | 255 | 21 | 2 |
| mapbox / finetuned / model:0.85 | 254 | 254 | 253 | 20 | 1 |

**So**: option B as specified is built, tested and on by default, but on GTA's emulated data it never fires, and the
3x forced merges stay (20-22 against the full map's 4-8). Without the gates it fires only at one maneuver's approach
and is wrong every time (and costs about a maneuver a run). On real Mapbox data it has something to work with only
where a road's turn:lanes run on between junctions (the `lanes_vary_bus` kind); how often that is on Amy's drives is
open (question 1). The lane ends behind the forced merges need either the lane head (the model's count dropping on the
car's side ahead, last session's question 2) or the model leaving an ending lane itself, as Amy plans.

## Item 2: bus lanes never a target (`2dcbd17`)

**What changed** (`lane_guidance.py`): `targets` leaves bus lanes (`access: {designated: [bus]}`) out of a move's
valid lanes (a move whose valid lanes are all bus lanes has no target: nothing asked); they stay in the lane count, so
the car's lane is counted as before. An `ends` move keeps only the non-bus lanes that carry on. A lane change is never
asked into a bus lane on the road the car is on (`MapboxRoute.bus_lanes`: that road's next lane list ahead, else its
last behind, each bus lane counted from the edge it's nearer in its list, so a model counting fewer or more lanes still
finds the kerb one); the change for that move isn't asked (the car stays beside the bus lane).

**Tests** (`test_lane_guidance.py`, +3): a made left turn whose valid lanes include a bus lane at the kerb (target the
lane beside it; from the bus lane, asked out; all-bus: no target); a made fork with no lanes of its own (the fallback:
the outermost lane) along a road whose lists show a bus lane at the left: one change to the lane beside it, then
nothing more; the `lanes_vary_bus` fixture (Canal Road: no target is a bus lane, `bus_lanes` finds it, and from the lane
beside it no change left is ever asked along the road and for the right turn).

**GTA**: our map has no bus lanes (no `bus`/`psv` key or value on any way of live21b's pbf) and the emulator gives no
`access`, so navd_sim can't change: the Mapbox path at the final commit (`c369422`, seed 0, both models and heads) is
identical to the baseline and to item 1's (247 / 244 / 254 / 252, every column).

## Item 3: forks without arrows (`9bc9f06`)

**Test** (`test_lane_guidance.py`, +1, no code change): a town fork (slight left) and a motorway exit (off ramp slight
right) with no lanes from Mapbox at the node or on the approach: kind fork / exit, target the outermost lane on the
branch's side counted from that side (`[0]`, whatever the model counts), asked from 2 lanes away (reason fork / exit,
2 changes), nothing asked from that lane; a fork on a motorway is an exit the same way; with `fallback` off, nothing
asked.

## Item 4: no lane change asked below the model's lane change speed on the bridge planner (`90e7d82`)

**What changed** (`selfdrive/navd/planner.py`, small and self-contained: one helper, four conditions): `_lc_too_slow(v)`
= `model_lc_speed > 0 and v < model_lc_speed`. Below it no lane change is asked for at all: planned changes for turns,
forks, straight on and lanes ending (`_change_for`'s `ready`, which before held only city changes below `lc_wait` x
`model_lc_speed` = 8 m/s and let merges and the rest through above 2 m/s), into a bay (`_enter_bay`) and out of an
oncoming lane (`_keep_right`). A change asked for but not started yet (`change_went` false) is withdrawn (`gave_up`)
once the car is slower, before its blinker is sent, as `lane_guidance` does below `lc_min_speed`. The turn is then
left to what the planner already does there: `turn_into_lane` signals it from beside its lanes (its desire takes the car
over), else the route. `model_lc_speed` 0 keeps the old behaviour. On GTA the bridge's default is n16's 10 m/s; for the
finetuned model set `model_lc_speed` 4 in the bridge's tune file, as navd_sim does per model.

**Tests** (`tools/sim/bridge/gta5/test_nav.py`): three tests pinned asks below 10 m/s and now run their mechanics with
`model_lc_speed` 0 or above it (the bay at 6 m/s, the city timing at 10.5 m/s instead of 9, a change at 11 m/s);
new: the slow bay (6 m/s, n16: no change, the turn signalled), no change at 9 m/s (between `lc_wait` and the model's
speed), and a change asked at 11 m/s withdrawn when the car slows to 8 before it starts (`gave_up`, no blinker). Suite:
247 passed.

**Measured** (navd_sim, `--nav map`, the bridge planner on the full map; same trips, heads and seeds as the baseline):

| run | seed | in lane: before -> after | turns | forks / exits | straight on | LC req / done / gave up / failed | LC late (asked < v_ok) | hanging s | forced merges |
|---|---|---|---|---|---|---|---|---|---|
| n16 / truth | 0 | 247 -> 242 | 68 -> 67 | 95 -> 92 | 84 -> 83 | 181/120/8/53 -> 162/106/7/49 | **85 -> 0** | 27 -> 32 | 7 -> 7 |
| n16 / model:0.85 | 0 | 250 -> 245 | 72 -> 71 | 97 -> 94 | 81 -> 80 | 181/124/5/52 -> 160/109/4/47 | **78 -> 0** | 35 -> 27 | 6 -> 6 |
| finetuned / truth | 0 | 262 -> 264 | 80 -> 81 | 99 -> 99 | 83 -> 84 | 219/148/8/63 -> 214/147/8/59 | **8 -> 0** | 53 -> 40 | 4 -> 4 |
| finetuned / model:0.85 | 0 | 255 -> 255 | 74 -> 74 | 99 -> 99 | 82 -> 82 | 209/139/4/66 -> 208/139/4/65 | **5 -> 0** | 51 -> 50 | 8 -> 8 |
| n16 / truth | 1 | 245 -> 248 | 71 -> 75 | 91 -> 91 | 83 -> 82 | 184/122/8/54 -> 156/115/6/35 | **88 -> 0** | 50 -> 40 | 8 -> 7 |
| n16 / model:0.85 | 1 | 242 -> 243 | 69 -> 67 | 91 -> 94 | 82 -> 82 | 179/118/5/56 -> 159/112/5/42 | **74 -> 0** | 24 -> 19 | 8 -> 7 |
| finetuned / truth | 1 | 255 -> 255 | 76 -> 76 | 97 -> 96 | 82 -> 83 | 224/146/21/57 -> 222/145/21/56 | **11 -> 0** | 87 -> 72 | 4 -> 5 |
| finetuned / model:0.85 | 1 | 254 -> 254 | 76 -> 76 | 97 -> 97 | 81 -> 81 | 217/142/8/67 -> 217/143/8/66 | **6 -> 0** | 36 -> 36 | 7 -> 6 |

"LC late (asked < v_ok)" for the map nav is 0 in every run. In-lane over both seeds: n16 -1 (truth: -5, +3) and -2
(0.85: -5, +1), finetuned +1 and 0: within the seed-to-seed noise (about ±3), the finetune barely touched (its
`v_ok` is 4 m/s, so few of its asks were slow). n16 is asked for 19-28 fewer changes a run and 4-19 fewer fail (started
too slow to get across). What n16 loses (seed 0): forks and lanes leaving at a split whose change used to be asked at
5-9 m/s after a stop and now comes too late (R34's fork at (261.5, -365.6), R01/R31's split at (-1158.9, -683.8),
LANE_END's straight on at (85.3, -1050.9)), and two turns with the car at walking pace; what it gains: slow turns
now signalled rather than asked for as a change (bays opening 13-18 m before the turn at R21/R11's (230.3, -222.0) and
R28's (303.8, -472.4); R18's turn at (-768.0, -1710.1)). In game those slow asks
are the n16 hazard Amy saw; the stand-in still moves over (slowly) for a lane change desire at 5-10 m/s, so the sim
credits them, which is the optimistic side.

The planner's own lane-ending asks, through item 1's check (a side result): 36 of 40 right on seed 0 (n16 / truth),
the rest `other_side`; before item 4 on seed 1, 43 of 45 (finetuned 37 of 44); after, 33 of 34. The `other_side` ones
are 1-8 per run (more with the noisy head); not looked into (question 4).

## Hand-over

### In short

- **Item 1, option B** (`laneEnds` from shrinking lane lists): built, gated, tested, on by default. On GTA's emulated
  Mapbox data it never fires: the road past a drop has no lane list of its own to compare (lanes only on turn:lanes
  approaches, as on Mapbox), so the Mapbox path's in-lane, changes and forced merges are exactly the baseline's (20-22
  forced merges against the full map's 4-8, both seeds). Ungated it fires 1-2 times a run, always wrong (a maneuver's
  approach listing fewer), and placed earlier it adds `gain` / `other_side` false positives. navd_sim now checks every
  lane-end ask on the full map and breaks the false ones down by cause.
- **Item 2, bus lanes**: never a target, never changed into, still counted (no bus lanes on GTA's map).
- **Item 3**: the outermost-lane fallback at forks and exits without lanes is pinned by a test.
- **Item 4, no slow lane changes on the bridge planner**: "LC late (asked < v_ok)" 85/78/8/5 -> 0 (seed 0) and
  88/74/11/6 -> 0 (seed 1); in-lane within the noise (n16 -1 / -2, finetuned +1 / 0 over both seeds).
- Suite: 247 passed, 0 failed (the preflight's set plus the new tests), with `MAPBOX_TOKEN` unset.

Reproduce (repo root, venv, `GTA5_MAP` set, `MAPBOX_TOKEN` unset; ~13 min per seed on 4 cores):

```
python -m openpilot.tools.sim.bridge.gta5.navd_sim --nav map,mapbox --model n16,finetuned --lane truth,model:0.85 \
  [--sim-seed 1] [--tune '{"ends_count_match": false, "ends_at_maneuver": true, "ends_valid_only": false}'] --out r.json
```

### In-game checks for Amy

Trip specs are `x,y,z,heading,lane>dx,dy` (navd_sim's). "Mapbox path" = `GTA5_NAV_PATH=navd` with the Mapbox server
and `NavLaneGuidance` as in `CLOUD_RESULT_NAVMAPBOX.md`; "bridge" = the default path.

1. **A lane ending before a junction, unannounced**: FX1 `799.0,-1752.5,28.9,264,1>814.2,41.7`, the left lane ending at
   (1107.7, -1644.4) 84 m past the left turn (also R08 `-1269.8,-710.5,10.1,111,0>176.1,816.2` at (-1788.5, 81.5): two
   lanes listed up to 33 m before, none after). Mapbox path: no `laneEnds` request at all (navInstruction.request's
   reason); watch how n16 merges at the end (the sim's forced merge). Bridge: the planner's lane-ending change is
   asked only at 10 m/s or more.
2. **A lane ending on the far side**: R19 `694.0,-851.2,42.3,90,0>-1075.9,391.6` through the junction at (202.8, 194.7),
   where the map's left-turn lane ends (`le st* st/ri*` into it), then on past (53, 258), where Mapbox's lists go from 3
   lanes to 2. From the right lanes: no `laneEnds` request on either path, and none past the junction from any lane.
3. **A bay turn, no false lane-end ask**: R06 `90.8,-2046.6,17.3,269,1>-394.4,407.1` (also R38, R18 later on their
   trips) along (287, -2131) to the left turn at (377, -2179): Mapbox lists 2 lanes, then the left-turn lane alone on
   the last 60 m. Mapbox path: only the turn's request (kind turn or a laneChange with reason turn), never `laneEnds`.
   Also look at the lanes on the ground there: the map has them 2 -> 1 -> 2 -> 1 within 50 m (drops at about 174 and 225
   m into R06, a lane added at 199 m), which looks like one of live21b's chain joins.
4. **Bus lanes**: GTA's map has none tagged; nothing to check in game (Amy's real drives: does the lane head count them?).
5. **A slow approach, no lane change below the model's speed** (bridge, n16, `model_lc_speed` 10): MDT
   `-357.8,-1658.9,18.3,331,0>1468.2,-1050.9` at the lane ending by (-248.6, -1441.2), where the old planner asked at
   5 m/s and from a standstill; MD029 `-236.0,-597.2,33.2,20,9>-743.5,-168.0` approaching the left turn at (-193.7,
   -487.3) (asked at 9.8 m/s before); FX1's left fork, asked at 8 m/s at (826.5, -1754.6). Watch: no NavDesire
   `laneChange` and no lane change blinker from nav below 10 m/s; a request opened above it and the car slowing below
   before the change starts is withdrawn (navRequest outcome `gave_up`, the blinker off); near a turn, its signal comes
   instead (turn_into_lane), never a lane change desire n16 could take as a turn.

### Open questions

1. Option B only has food where turn:lanes run on between junctions with fewer lanes on the later way (the
   `lanes_vary_bus` kind). How common is that on Amy's real drives? If rare, keep it (it costs nothing here) or drop
   it; the lane ends behind the forced merges need the lane head or the model itself.
2. (287, -2131) on R06/R18/R38: the map's lanes 2 -> 1 -> 2 -> 1 within 50 m; a chain join (the known live21b issue)?
   The only place the ungated option B asks hinges on it.
3. Item 4 also stops merges out of an ending lane below 10 m/s (a queue): the sim's forced merges didn't rise (7/6/4/8
   -> 7/6/4/8 on seed 0, 8/8/4/7 -> 7/7/5/6 on seed 1), but in game n16 then has to merge on its own at the end.
   There's no hysteresis around `model_lc_speed`: a car hovering at 10 m/s could see ask / withdraw cycles (not seen
   as a problem in the sim: fewer asks overall).
4. The full-map planner's own lane-ending asks are 1-8 per run `other_side` by the new check (the car's lane doesn't
   end within the ask's target + 50 m, another does), more with the noisy head; not investigated.
5. comma's `test_navigation.py::test_request_and_instruction` fails whenever `MAPBOX_TOKEN` is set in the environment
   (navd then uses api.mapbox.com); it should clear it. Not touched.
6. Pre-existing ruff findings in `test_nav.py` (B007, B905) are left as they were.

### Environment notes

- Native params were built here for the first time (apt `capnproto`, `libcapnp-dev`, `libzmq3-dev`; `capnpc` into the
  git-ignored `cereal/gen/cpp`; json11 from its public repo; `g++ -shared` of `params_c.cc params.cc swaglog.cc util.cc`
  with the generated schemas), so `test_navigation.py` ran (20 pass). Nothing of it is committed.
- Measurements ran from git worktrees of the item commits (`b22ec06` for item 1, `90e7d82` for item 4), so later edits
  didn't leak into a running sim.
