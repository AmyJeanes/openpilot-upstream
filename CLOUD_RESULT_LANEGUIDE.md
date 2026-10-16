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
