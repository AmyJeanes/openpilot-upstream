# Cloud result: map driver lane changes, gap waits and red lights

## Summary for Amy

Preflight done; item 1 (lane changes kept off hatched areas) is in. Items 2 and 3 follow below as they land.

## Preflight

1. Repo: PASS. Shallow single-branch clone of `mapdrive-cloud` at `081e37263ec0`; its parents are `020b96f2f19e` (the
   brief) and `647143d89ce8` (`gta5`, matching `git ls-remote origin gta5`).
2. Data: PASS. `cloud-data` at `031aeacd62ff`; sha256 of all 11 files under `cloud/live18/map1/` and `cloud/live19b/`
   match `cloud/MANIFEST.md`. Live map assembled in one folder (live18's pbf, nodes.npz, gunzipped roads.json and
   paths.jsonl, live19b's gunzipped lanes.json: 69666 lines, 406 of kind `hatched`), `GTA5_MAP` set to it.
3. Dependencies: PASS with one gap. From PyPI into a Python 3.12 venv: numpy, osmium, pyvalhalla 3.9.1, shapely,
   pytest, scipy, pycapnp, opendbc 0.2.1, msgq 0.1.3. PyPI's opendbc is too old for this tree's `cereal/log.capnp`
   (`'Car.RadarData' has no member named 'ErrorDEPRECATED'`, capnp aborts the process), and the submodules aren't
   reachable here. So `test_blindspot.py` can't import `openpilot.cereal.log` / the simulated Tesla. Designed around:
   it runs with a scratch pytest plugin (not committed) that stubs `cereal.log`'s two enums and the Tesla/CAN imports;
   the 3 tests that need openpilot's real CarState/DesireHelper (`test_tesla_can_carries_the_blind_spot`,
   `test_openpilot_lane_change_waits_for_the_blind_spot`, `test_openpilot_lane_change_waits_out_a_moments_gap`) can't
   run here. None of them touch the map driver.
4. Valhalla: PASS. Tiles built from the pbf with pyvalhalla's `valhalla_build_tiles` (9 s), `valhalla_service` on
   :8002 answering `/status`; `mapdrive_sim.trip_route` routes on it (cache in a scratch file via
   `GTA5_TRIP_ROUTES`).
5. Baseline on the untouched branch: `test_mapdrive.py` + `test_mapdrive_lead.py` 68 passed, 1 failed
   (`test_collision_routes`: route 029's plan 1.12 m off its lane, the known one), as on Amy's PC;
   `test_lane_slots.py` 20 passed; `test_blindspot.py` 32 passed, 3 not runnable (above). This is the bar.
6. Push access: PASS (this file, pushed to `mapdrive-cloud`).

## 1. Lane changes that don't cross hatched areas

**What changed**
- `map/router.py`: `hatched_near(points, osm)`: the hatched rings (`LaneLines.rings`, from the lanes.json beside the
  pbf) within 15 m of a route's line and on its level (within 3 m of the route's node heights), each with the m along
  the route where it's beside it. `Route` puts them on `RouteLanes.hatched` (`map/osm_lanes.py`; empty on a map
  without them, and on made routes unless `mapdrive_sim.made_route(..., hatched=[...])`).
- `gta5_mapdrive.py`:
  - `_hatch_hit(route, keys, a, b)`: where a change's body (`half_width`, front, rear, plus `HATCH_CLEAR` 0.2 m) would
    be in an area, modelled as the lane line is drawn (each place's cross-section, the quintic move across, the
    heading turned with the move), every 1 m where an area is beside the route. A place where our body in the lane it
    leaves or the lane it enters would be in the area too doesn't count (the map's lane running into paint, which no
    change mends).
  - `off_hatched` in `_make_plan` after `off_jogs`: a change that hits is moved, whichever moves it less, to end
    `HATCH_PAD` (5 m) short of where it would first meet the area (before a gore's nose) or to start 5 m past where
    it last would (after a painted island), stepping on past any area the new place meets (up to 4 tries each way);
    length kept, or shortened where room runs out to no less than `_lc_least`; within the holds in its lanes
    (`_change_room`: not back across the move before, not past a fork it leads to or the next lane event), onto no
    stop line (nor, off a freeway, a junction node) it was clear of. No place: kept where it was and flagged as a
    `hatched_change` anomaly (span, hit range, area centre). Each move is a `hatched_move` event and in
    `summary()["hatched"]`.
  - `_deferred_off_hatched` in `_lc_gate`: a change put off for a gap onto an area it wasn't on is put off past it,
    and where it can't be, it isn't put off: the existing "late" stop or, on a motorway before a fork, the fork's other
    branch, rather than across the chevrons. (A change already flagged is put off as before.)
  - `_ask_check`: an arrow key's change whose move (placed as `_asked_keys` would) hits an area is refused,
    `"hatched"`; `gta5_events.PROMPTS["hatched"]` = "No {side} here: hatched".
  - `"hatched": false` in the mapdrive config turns it all off.

**Tests** (new)
- `test_mapdrive.py::test_lane_change_off_hatched` (made routes): a change across a gore's V (nose at 500 m) moved
  from 530-585 to 445-500, the built path's body never in the V; a change across a painted island from inside its span
  (no room before) moved to start at its end; a change from the car short of the nose with the V on to the next key
  kept and flagged; `hatched: false` leaves it and its body is in the V. Driven: nav's change for a left turn
  (399-474 m) across a painted V from 440 m is made before it, the car's body never touching it.
- `test_mapdrive.py::test_asked_lane_change_past_hatching`: arrow key right beside a long painted median: `"hatched"`
  (HUD "No right here: hatched"); past its end: made.
- `test_mapdrive_lead.py::test_lane_gap_wait_stays_off_hatched`: the fork test's setup with a gore V between the lanes
  from 450 m: alongside until 300 m, the change is made before the nose; alongside until 430 m, the fork's other branch
  is taken (with `hatched: false` the wait put the change into the chevrons); never our body in the V, no touch.
- `test_mapdrive.py::test_hatched_live_routes` (live map + router, skipped without): 6 trips past the gores below,
  every change clear as planned and as put off every 20 m up to 200 m.

**Live routes (live19b)**

| Trip | Gore | Changes | Moved | Flagged | Deferrals checked |
|---|---|---|---|---|---|
| `-756,-1841,25,-161,0>-715,-1861` | (-705,-1988) | 4 | 0 | 0 | 11 |
| `-830,-2278,12,-28,0>-838,-2233` | (-754,-2147) | 1 | 0 | 0 | 2 |
| `955,-773,35,65,0>607,-517` | (724,-585) | 2 | 0 | 0 | 2 |
| `907,-719,35,54,0>504,-483` | (623,-534) | 3 | 0 | 0 | 2 |
| `730,-58,35,142,0>474,-422` | (568,-326) | 1 | 0 | 0 | 6 |
| `482,-502,33,83,0>199,-473` | (327,-483), under the overpass areas | 7 | 0 | 0 | 33 |

A wider scan (scratch scripts, not committed): 224 trips from the roads approaching each of the 7 gores to the roads
leaving them, from the leftmost and rightmost lane: no change moved or flagged; pushing every change near an area on
in 10 m steps up to 400 m (as a gap wait would) never swept one; forcing each change to run through the next area along
its route (76 cases) never swept one either, by the model and by the built path alike. Why: on live19b the gores lie
between separate ways past their split (or before their merge) node, so where the paint is, the route's cross-section
only has its own road's lanes and a plan can't move across it; the router already keeps routes off links through
paint. The guard bites where a road's own lanes run beside paint (ghost islands and painted medians within a way, a
gore's tip painted before its node), for gap waits pushed towards those, and for the arrow keys. Before the level
check, 5 of the 224 trips looked like they crossed areas 68281 and 68290: those are on the overpass at z 42 above the
motorway at z 33, and are now ignored.

Planning time with the guard on long live routes: +0 to +60 ms (e.g. 6.7 km: 600 vs 572 ms; 2.4 km with a 296 m
renumbering ramp beside 9 areas: 237 vs 181 ms).

**Areas that look wrong** (not changed)
- lanes.json line 68450 (a parking-bay block, centre about (-212,-1237), z 36): its east end has a zero-width spike
  from (-192.4,-1234.9) up to (-192.4,-1231.1) and back, 3.8 m into the lane beside it, and a tail out to
  (-235.4,-1237.5).
- Lines 68431 and 68557 are two overlapping rings for the same gore at (-705,-1988); harmless, but one may be a
  duplicate.

**Test results after item 1**: test_mapdrive + test_mapdrive_lead + test_lane_slots 92 passed, 1 failed
(`test_collision_routes`, the same 029 failure as the baseline); test_blindspot 32 passed (3 not runnable here).
