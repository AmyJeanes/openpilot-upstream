# Cloud result: lane guidance (navd planner) for comma's nav models

Brief: `CLOUD_TASK_LANEGUIDE.md`. Branch `laneguide-cloud`, one commit per step.

## Preflight

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Repo | PASS | `laneguide-cloud` = `885f569` (the brief), parent `gta5` = `84739d1` ("cereal: a backlog read in one update has no rate yet"); fetched `--depth 2`, single branch |
| 2 | Data | PASS | `gta5-distill` `cloud-data` = `891a36f`; sha256 and size match `cloud/MANIFEST.md` for all 11 `live18/map1/*` and `live19b/*` files. Live map assembled in `/home/user/map`: live18's `gta5.osm.pbf`, `roads.json`, `paths.jsonl`, `nodes.npz` + live19b's `lanes.json` (69666 lines, 1451 stop lines); `GTA5_MAP=/home/user/map` (also symlinked as `~/gta5map_lanes`, `~/gta5map`) |
| 3 | Dependencies | PASS (with a shim) | Python 3.12 venv (uv), from PyPI: numpy 2.5, osmium 4.3.1, pyvalhalla 3.9.1, shapely, pytest, pycapnp 2.1.0, pyzmq, scipy, comma's `opendbc` 0.3.1 (for cereal's `car.capnp`). Not available: openpilot's native `msgq` (the PyPI `msgq` is an unrelated SysV package, removed) and `params` (C++). `cereal.messaging` imports msgq, so `test_nav_msgs.py` (and anything importing `openpilot.cereal.messaging`) needs it; a socket-free stand-in outside the repo (`/home/user/shim/msgq`, on `PYTHONPATH`) lets the capnp-only paths (`new_message`, `log_from_bytes`) run. `DesireHelper` needs `Params` (native) and isn't importable here |
| 4 | Valhalla | PASS | `valhalla_build_config ... > valhalla.json`; `valhalla_build_tiles` (site-packages/valhalla/bin) on the pbf: 10 s, 17 MB of tiles; `valhalla_service valhalla.json 2` on :8002, `/status` answers (3.9.1) |
| 5 | Baseline tests | 112 pass, 1 fail | `test_nav.py`, `test_lane_slots.py`, `test_lane_numbering.py`, `selfdrive/navd/tests`: 96 passed, 1 failed in 263 s. `test_nav_msgs.py` (with the msgq shim): 16 passed. The failure: `test_lane_numbering.py::test_live_freeway_merge_exit_and_splits`, the Dutch London St ramp split (-741.9, -1838.4) routes 3063 m instead of < 300 m (Valhalla on these tiles takes a long way round from that pose); pre-existing, not touched by this work |
| 6 | Push access | PASS | this file, committed and pushed to `laneguide-cloud` |

Test command (from the repo root, venv active, Valhalla up):

```
export PYTHONPATH=$PWD:/home/user/shim GTA5_MAP=/home/user/map
G=openpilot/tools/sim/bridge/gta5
python -m pytest -q $G/test_nav.py $G/test_nav_msgs.py $G/test_lane_slots.py $G/test_lane_numbering.py openpilot/selfdrive/navd/tests
```

## Step 1: the closed-loop harness (`tools/sim/bridge/gta5/navd_sim.py`)

navd's `Planner` runs closed loop on real routes, fed by the bridge's own `nav_inputs` from `Route.info` (as
`gta5_brain` does, at 10 Hz), with stand-ins for the game, openpilot, the driver and the driving model:

- **Routes**: mapdrive_sim's: its made routes (tests) and `trip_route` on the live map. The scenario trips are the six
  real e2e trips in the repo (`test_mapdrive`'s 012/014/029, its lane-end trip, bsm_trial's FX1, mapdrive_trials'
  example); the scenario lists under `/mnt/e` aren't in the cloud data. Plus 40 random Los Santos trips (e2e's city
  box, 1.5-3.5 km apart, random start lane; 2-8.5 km routes, about half with a freeway leg), seed 1, kept in
  `navd_sim_trips.json`; their Valhalla routes are committed (`navd_sim_routes.json.gz`), so a run needs no router.
- **Car** (frenet, on the route's own lane sections): lane keeping in a lane the route's `laneMaps` carry it through
  (an ending lane merges into the nearest); speed from the limit (or road class), curvature (2 m/s² lateral), each
  turn's `Turn.speed` and the stop lines (lights on a 40 s cycle from a random phase, stop signs), slowing at
  1.5 m/s², behind the lead (Gipps); the game car's 0.22 s acceleration lag. Turns are taken from whatever lane the car
  is in (n16 has the nav vector), into the road out's outside lane on that side.
- **openpilot + driver (PLAN D stage 3)**: nav's requests work the blinker; desire_helper's rules: a lane change below
  20 mph only with NavDesire `laneChange`, the blind spot clear 0.5 s, the driver's nudge at once, 10 s timeout; the
  driver cancels a turn's blinker once round, a lane change's once done.
- **Model lane change** (`ModelCfg`): 4-6 s at or above `v_ok`, the rate falling off as the square of
  (v - v_min)/(v_ok - v_min) below it, none below `v_min`; the model waits short of the line for a gap in the lane it
  moves into; given up (desire_helper's 10 s) short of half way, it comes back. `n16`: v_ok 10, v_min 5 m/s (the probe:
  under 2 m of plan movement below ~10 m/s). `finetuned`: v_ok 4, v_min 2 m/s.
- **Traffic**: 10 vehicles per km per lane about the car (-150..450 m), 80-105% of the road's speed, stopping at red
  lights (queues), following each other and the car; they give the blind spot (alongside, -7..+2 m) and the lead.
- **Lane source** (`HeadCfg`): `truth` (the bridge's reading, lane_source "map"), or a noisy `modelV2.laneHead`
  (strict accuracy, count accuracy, lag, dropouts, Markov errors) as lane_source "model" (no truth at all) or "fused".
- **Judge**: each maneuver at its commit point (a turn's stop line / junction entry, 2 m before a fork's gore or where
  lanes end), by the planner's own `lanes_for` (lifted out of `Planner._lanes_for`, unchanged) on the true lane: in a
  lane for it or not. Not = missed (the model can't make it from there: a reroute); the car carries on along the route
  as after one. "Relevant" maneuvers are those where not every lane goes the route's way; "needing a change" those
  where the car was out of its lanes somewhere in the last 400 m. A split at a fork counts once, as the fork.

Metrics: in a target lane at the commit point; lane changes requested / done / given up (blinker off before it
started) / failed (started, didn't get across); late = asked below the model's v_ok; time a requested change hung
(blinker on, not started: blind spot); forced merges (an ending lane); contacts with traffic (a sanity check of the
traffic model: they come from its lane renumbering at lane maps and turns, ~1-2 per trip, not from nav).

Reproduce (venv, `GTA5_MAP` set; about 3 min on 4 cores for both settings):

```
python -m openpilot.tools.sim.bridge.gta5.navd_sim --model n16,finetuned --lane truth --out base.json
```

Tests: `test_navd_sim.py` (8, made routes only: the rate by speed, a two-change turn made from afar, a slow road only
the finetune manages, the lanes out of a turn, a blind spot holding a change, the lane head's noise and lag, the model
lane source without truth, the summary).

### Baseline: today's planner, truth lane, 46 trips (163 km, 881 maneuvers judged)

| run | in lane | needing a change | missed | LC req/done/gave up/failed | LC late (asked < v_ok) | hanging s (per LC) | forced merges | contacts (sim) | arrived |
|---|---|---|---|---|---|---|---|---|---|
| n16 / truth | 248/302 (82%) | 95/148 (64%) | 54 | 197/115/7/75 | 98 | 112 (0.6) | 8 | 81 | 46/46 |
| finetuned / truth | 257/302 (85%) | 106/150 (71%) | 45 | 200/139/7/54 | 8 | 128 (0.6) | 5 | 76 | 46/46 |

| run | turns | forks / exits | straight on |
|---|---|---|---|
| n16 / truth | 68/107 | 97/108 | 83/87 |
| finetuned / truth | 79/107 | 96/108 | 82/87 |

What the misses are (n16; finetuned in brackets):

- Turns whose change failed slow, 19 (13): asked 30-60 m before the turn, once the car is slowing for it or for a red
  light. Half of n16's change requests (98 of 197) come below 10 m/s. Many of these are into a turn bay that opens
  only 45-60 m before the junction (nav changes into it as it opens: `_enter_bay`), or a lane the map only starts there.
- Turns with no change asked, 10 (8): the lanes for a second turn right after the first (out of a left turn in the
  left lane, a right turn 40 m on), and turns nav left to the route after a fork it gave up.
- Forks / exits with no change asked, 9 (10): mostly map lanes: the route's lane begins on the far side only 20 m
  before the gore (listed under map faults below), or GTA's fork counts disagree with the lane maps.
- Fast failures, 12 (8): changes started above v_ok but too near the junction to finish before the car slowed.

So the finetune helps (+11 turns), but timing is half the problem: changes asked too late even for a model that can
change at 4 m/s.

## Step 2: lane changes timed for the model

### What the baseline says the problem is

Turns needing a change, split by where their target lane begins (`Maneuver.opens`, from the route's lane maps):

| turns needing a change | n16 | finetuned |
|---|---|---|
| target lane there more than 80 m before the commit point | 30/35 | 32/36 |
| target lane begins less than 80 m before (a bay, a lane the map adds there) | 5/38 | 14/37 |

So where the lane exists early, today's planner already gets n16 there (86%). The misses are lanes that open late,
usually with a red light or a queue ahead: the car is slow by the time the lane exists, and n16's lane change doesn't
move below ~10 m/s. Timing can only partly help there; the finetune and the turn desire are the levers.

### Changes (`planner.py`, all `Tune` keys; `model_lc_speed: 0` gives today's planner exactly, checked: identical runs)

- `model_lc_speed` (10 m/s, n16's): the speed the model needs for a lane change. With it, a city move's changes
  (`_change_for`, not freeway moves, which `fwy_` already times far ahead) must end before the car slows below it:
  before a turn's slowing (`slow_decel` 1.5 m/s² from it down to `Turn.speed`, by the junction's entry) and
  `queue_room` (30 m) before the last lights before the move (`_fast_room`). They start `city_lead` before that, timed
  at the road's limit, not the car's speed (`_plan_speed`), so slowing traffic doesn't put them off.
- `lc_wait` (0.8 of `model_lc_speed`): before that point, a change waits while the car is slower than that (a queue,
  a light); one asked for that hasn't started when the car slows is withdrawn, and doesn't count as a try that got
  nowhere (LANE_CHANGE_TRIES). Lower waits gain nothing in lanes and ask for more futile changes (wait 0: 211 requests,
  116 late; 0.8: 183, 89).
- Past that point and slower than `model_lc_speed`: not asked for, rather than forced at the junction. A turn is then
  left to the route at its last place (`_signal_due`'s existing rule), a fork at once (where surely out of its lanes).
- `turn_into_lane` (80 m): slower than the model changes lanes at (and the lane turn speed), one lane beside a turn's
  lanes on its side, nav signals the turn rather than asking for a lane change: the probe found n16 takes a turn
  pulse well even from a standstill (2-12 m the right way), unlike a lane change pulse. This is what takes a car into a
  bay that opens late beside it; the stand-in models it (`turn_takes_lane`), which is an assumption to check in game.
- Unchanged: the blind spot and nudge rules (desire_helper, `_watch_change`), merges out of an ending lane (`must`),
  freeway timing, bays at speed (`_enter_bay` runs as before unless `turn_into_lane` applies).

### Before / after (46 trips, truth lane, two traffic/light seeds)

| run | in lane | needing a change | missed | LC req/done/gave up/failed | LC late (asked < v_ok) | hanging s |
|---|---|---|---|---|---|---|
| seed 0, before, n16 | 248/302 (82%) | 95/148 (64%) | 54 | 200/114/6/80 | 98 | 111 |
| seed 0, after, n16 | 249/302 (82%) | 90/142 (63%) | 53 | 183/115/9/59 | 89 | 138 |
| seed 1, before, n16 | 244/302 (81%) | 95/153 (62%) | 58 | 198/121/12/65 | 98 | 41 |
| seed 1, after, n16 | 245/302 (81%) | 91/148 (62%) | 57 | 187/123/9/55 | 96 | 51 |
| seed 0, before, finetuned | 257/302 (85%) | 107/151 (71%) | 45 | 204/140/8/56 | 8 | 130 |
| seed 0, after, finetuned | 260/302 (86%) | 106/147 (72%) | 42 | 218/146/8/64 | 11 | 128 |
| seed 1, before, finetuned | 258/302 (85%) | 112/156 (72%) | 44 | 199/145/12/42 | 7 | 55 |
| seed 1, after, finetuned | 261/302 (86%) | 109/150 (73%) | 41 | 222/149/16/57 | 8 | 62 |

| run | turns | forks / exits | straight on |
|---|---|---|---|
| seed 0 before / after, n16 | 68 → 72 /107 | 97 → 94 /108 | 83 → 83 /87 |
| seed 1 before / after, n16 | 67 → 71 /107 | 95 → 94 /108 | 82 → 80 /87 |
| seed 0 before / after, finetuned | 79 → 82 /107 | 96 → 97 /108 | 82 → 81 /87 |
| seed 1 before / after, finetuned | 81 → 83 /107 | 95 → 97 /108 | 82 → 81 /87 |

Ablations (seed 0): without `turn_into_lane`, n16 246 (turns 69), finetuned 259 (81). With the stand-in's turn desire
not taking it a lane over (`--no-turn-take`): before 248 / 257, after 244 / 259 (n16 turns 68 → 67).

Reading it: the noise between seeds is about ±2 maneuvers. The gain is real but small: +3-4 turns for each setting,
most of it from `turn_into_lane`, which depends on the turn desire doing what the probe suggests; n16 loses a few forks
it used to make with changes started at 6-8 m/s that came good as the car sped up. What the timing does reliably is
cut the changes n16 can't make: failed changes down 20-25% (80 → 59, 65 → 55), late requests down a little. Without
the turn-desire assumption the change is neutral for n16 (within noise) and slightly positive for the finetune.

The constants that matter, and why: `model_lc_speed` 10 (the probe's n16 threshold; set it to the finetune's 4 when
that model drives, as the harness does per stand-in), `slow_decel` 1.5 (the stand-in's and roughly the game's comfort
braking for a turn), `queue_room` 30 m (a few cars queued at a light), `lc_wait` 0.8, `turn_into_lane` 80 m (a bay's
typical length in the map). `LANE_CHANGE_*`, `FORK_*`, `BAY_*`, `SIGNAL_*` are unchanged.

Tests: `test_nav.py` +4 (a change ends before the slowing for lights and starts further out; a slow change waits for the
traffic; too slow near the turn isn't forced and the turn is left; slow beside a turn's lane signals the turn). Seven
existing tests that exercise low-speed mechanics (5-8 m/s on made routes) pin `model_lc_speed: 0` or
`turn_into_lane: 0`, with a comment: they test today's rules, which stay one setting away.

## Step 3: the model's lane as the lane source

### Changes (`planner.py`, `Tune`; `model_lane_robust: false` gives today's code exactly)

- Hysteresis and confidence (`_read_model_lane_robust`): a new lane is taken once read at least `model_lane_prob`
  (0.7) for `model_lane_hold` (1 s); the lane held stays while read at least `model_lane_keep` (0.4) sure, and through
  unsure readings and dropouts for `model_lane_stale` (3 s), then None (today: None after 1 s unsure).
- nav's own change: when the driver's stalk goes off after a change openpilot made, the lane it was into is expected:
  a reading of it is taken after `model_lane_after_change` (0.3 s), and with no reading the lane held moves over.
- Counts disagreeing (`_model_in_map`, `model_count_side`): nav plans in the map's lanes, so the model's lane is
  numbered in the map's count at the car (`NavInputs.lane_counts`, Route.info's laneCounts, which the bridge now
  passes, or the first lane map's), counting from the side the next maneuver that needs lanes has them on
  (`_side_that_matters`: a right turn or exit from the right, a left from the left, a way straight on from the edge its
  lanes touch). The edge the car is beside is what the model sees surely; a wide kerb lane with parked cars, or a lane
  the map adds or drops on the far side, then doesn't move it.
- Fused: as today (the model's where it counts as many lanes as the map's reading), now with the model's own count
  kept for that test, and the model's where the map has no reading. Preferring the model over the map's reading cost
  2-6 maneuvers here, because the map's reading in GTA is the truth.
- Sure-wrong without truth (`_sure_wrong`): today it needs the bridge's `lane_frac`, so with the model alone nav never
  leaves a turn to the route and would signal it from a wrong lane. Now: the model's lane held `model_sure_wrong` (3 s)
  and read sure within the last second. Held 0 s (just taken) it cost 6 maneuvers on misreads; 3 s costs none.

### Results (46 trips; lane heads with their own noise stream, so runs differ only by the lane source)

The lane head stand-in: strict accuracy as given (the gta5-train README wasn't readable from here, so 80-90% per the
brief), one lane off otherwise, 0.5 s lag, 5% dropouts, counts right 95% of the time (the miscounted lane on either
side), errors lasting ~1 s (counts ~3 s). "hard": 85%, 1 s lag, 15% dropouts, counts right 70%.

| run | truth | model 0.9 | model 0.85 | model 0.8 | fused 0.85 | hard (model) |
|---|---|---|---|---|---|---|
| n16, today's lane source | 247 (82%) | 249 | 250 | 246 | 243 | 237 (78%) |
| n16, robust | 247 (82%) | 250 | 250 | 248 | 248 | 241 (80%) |
| finetuned, today's | 261 (86%) | 254 | 255 | 252 | 258 | 239 (79%) |
| finetuned, robust | 261 (86%) | 256 | 254 | 256 | 251 | 242 (80%) |

(in a target lane at the commit point, of 302 maneuvers that need one; noise between seeds is about ±2)

Reading it: at 80-90% strict accuracy with these error dynamics, the model's lane costs n16 nothing measurable (its
misses are speed, not lane), and the finetune, which makes more changes, 5-7 maneuvers (2 points). The robust lane
source is level with today's there and gains 3-4 where the head is worse (lag, dropouts, miscounts), which is where it's
meant to help; most of its value is what the harness can't score: no truth needed at all (sure-wrong, dropouts, the
map's count). Fused is worth keeping as today's rule while the map's reading is the game's truth.

Tests: `selfdrive/navd/tests/test_lane_source.py` (9: a flicker, hysteresis, dropouts, counting from the side that
matters, the side itself, the map's count, the lane a change was into, fused, sure-wrong without truth).
