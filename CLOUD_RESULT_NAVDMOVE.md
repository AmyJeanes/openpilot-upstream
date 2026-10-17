# Cloud result: the GTA bridge's nav planner moves out of `selfdrive/` (no behaviour change)

Brief: `CLOUD_TASK_NAVDMOVE.md`. Branch `navdmove-cloud` (= `gta5` e55cb3b45 + the brief 8d4b72cf9).

## Preflight

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Repo | PASS | `navdmove-cloud` = `8d4b72cf9` (the brief), parent `gta5` = `e55cb3b45`; `gta5-distill` `cloud-data` = `1c75209` |
| 2 | Data | PASS | sha256 and size of the stored files match `cloud/MANIFEST.md` for `live18/map1/paths.jsonl.gz`, `live18/map1/nodes.npz` and all 5 `live21b/*` (pbf `ef13e870…`, lanes `ca32feb3…`, roads `46697b4f…`, lanegeom `ec5c3a79…`, build.log `02e654f4…`) |
| 3 | Map | PASS | `/home/user/map`: live21b's `gta5.osm.pbf`, `lanes.json` (41376903 B), `roads.json` (14889877 B) + live18's `paths.jsonl` (18855947 B), `nodes.npz`; `GTA5_MAP=/home/user/map`, also linked as `~/gta5map_lanes` and `~/gta5map` |
| 4 | Dependencies | PASS (native params missing, as before) | Python 3.12 venv (uv), PyPI only: numpy, osmium, pyvalhalla 3.9.1, shapely, scipy, pytest, pycapnp 2.1.0, pyzmq, opendbc 0.3.1, zstandard, setproctitle, requests, pillow, pyjwt, ruff 0.16.7 (`uv.lock`'s). `msgq` built from its public source (`commaai/msgq` 326a9f5, `pip install .`, outside the repo). Not available: native `params` (`libparams_c.so`), so comma's `selfdrive/navigation/tests/test_navigation.py` can't be collected (left out of every run); importing `opendbc.car` beside `openpilot.cereal` aborts (capnp "Duplicate ID" for `car.capnp`), so `test_blindspot.py`, `test_driver.py`, `test_world.py` and `gta5_world.py` can't import here, on `gta5` as on the branch |
| 5 | Valhalla | PASS | `valhalla_build_config` + `valhalla_build_tiles` on live21b's pbf: 13 s, 17 MB; `valhalla_service valhalla.json 2` on :8002, `/status` = 3.9.1 |
| 6 | Baseline tests | 502 pass, 1 fail | On the unmoved tree: `selfdrive/navigation/tests` (without `test_navigation.py`), `selfdrive/navd/tests`, `test_nav.py`, `test_nav_msgs.py`, `test_lane_slots.py`, `test_lane_numbering.py`, `test_navd_sim.py`, `test_mapbox_server.py`, `test_navd_path.py`, `test_nav_approval.py`, `test_nav_request.py`, `tools/sim/bridge/gta5/map`: 500 passed, 3 failed in 533 s; 2 of the 3 (`test_lane_guidance.py::test_navd_runs_it`, `test_mapbox_server.py::test_navd_uses_the_host`) only lacked `jwt`, and pass with PyJWT installed. The one left is `map/test_cruise.py::test_live_cruises` (a wrong-way at (-224, 79) on live21b), a known-red baseline the move must leave as it is |
| 7 | navd_sim baseline | PASS | `navd_sim --nav map,mapbox --model n16 --lane truth,model:0.85` (seed 0, `--sim-seed 0`, 46 trips, 302 maneuvers that need a lane), unmoved tree: map n16 / truth 247, map n16 / model:0.85 250, mapbox n16 / truth 247, mapbox n16 / model:0.85 244, as on Amy's PC; 10 min on 3 jobs. The result JSON (1682608 B) and every nav request (`--requests`, 593651 B) kept for the comparison |
| 8 | Push access | PASS | this file, committed and pushed to `navdmove-cloud` |

Test command (repo root, venv active, Valhalla up; after the move `selfdrive/navd/tests` is `tools/sim/bridge/gta5/navd/tests`):

```
export PYTHONPATH=$PWD GTA5_MAP=/home/user/map
G=openpilot/tools/sim/bridge/gta5
python -m pytest -q openpilot/selfdrive/navigation/tests --ignore=openpilot/selfdrive/navigation/tests/test_navigation.py \
  openpilot/selfdrive/navd/tests $G/test_nav.py $G/test_nav_msgs.py $G/test_lane_slots.py $G/test_lane_numbering.py \
  $G/test_navd_sim.py $G/test_mapbox_server.py $G/test_navd_path.py openpilot/selfdrive/controls/tests/test_nav_approval.py \
  openpilot/selfdrive/ui/tests/test_nav_request.py $G/map
python -m openpilot.tools.sim.bridge.gta5.navd_sim --nav map,mapbox --model n16 --lane truth,model:0.85 --sim-seed 0 \
  --jobs 3 --out sim.json --requests req.jsonl
```

## Commits

| commit | what |
|---|---|
| `4feb8f8` | this file's preflight |
| `d2c98a3` | the move: `git mv openpilot/selfdrive/navd openpilot/tools/sim/bridge/gta5/navd`, nothing else (11 files, all R100) |
| `d2f665a` | the imports (`openpilot.selfdrive.navd` -> `openpilot.tools.sim.bridge.gta5.navd`) in 39 Python files, and `selfdrive/navd/` -> `navd/` in the bridge README and two comments |
| `da10acc` | the route input's layouts into `selfdrive/modeld/route_layout.py`, so modeld's reader no longer imports the GTA encoder (below) |

## The new layout

```
openpilot/tools/sim/bridge/gta5/
  gta5_navd.py          the GTA layer's side of navd (unchanged): its inputs from the game, the destination
  navd/                 was openpilot/selfdrive/navd/
    __init__.py
    inputs.py           NavInputs / NavOutputs
    planner.py          the full-map planner
    lane_targets.py     turn_targets, LEFTS, RIGHTS (map/osm_lanes.py re-exports them, as before)
    lane_slots.py       route input v2's lane slots (imports map/osm_lanes.py)
    maneuvers.py
    map_match.py
    route_input.py      the route input encoder (RouteInput, decode, describe)
    tests/
      test_lane_source.py
      test_map_match.py
openpilot/selfdrive/modeld/
  route_input.py        the shared-memory writer/reader (unchanged but for its import and a docstring line)
  route_layout.py       NEW: the route input's layouts (Layout, LAYOUTS, LATEST, layout, convert and their index slices)
```

`navd/` keeps the package's name, so it reads as "gta5_navd.py drives navd/" and every import changes only its prefix.

## Old -> new import paths (for gta5-train)

Every module keeps its name; only the package prefix changes:

| old | new |
|---|---|
| `openpilot.selfdrive.navd` | `openpilot.tools.sim.bridge.gta5.navd` |
| `openpilot.selfdrive.navd.planner` | `openpilot.tools.sim.bridge.gta5.navd.planner` |
| `openpilot.selfdrive.navd.inputs` | `openpilot.tools.sim.bridge.gta5.navd.inputs` |
| `openpilot.selfdrive.navd.lane_slots` | `openpilot.tools.sim.bridge.gta5.navd.lane_slots` |
| `openpilot.selfdrive.navd.lane_targets` | `openpilot.tools.sim.bridge.gta5.navd.lane_targets` |
| `openpilot.selfdrive.navd.maneuvers` | `openpilot.tools.sim.bridge.gta5.navd.maneuvers` |
| `openpilot.selfdrive.navd.map_match` | `openpilot.tools.sim.bridge.gta5.navd.map_match` |
| `openpilot.selfdrive.navd.route_input` | `openpilot.tools.sim.bridge.gta5.navd.route_input` |
| `openpilot.selfdrive.navd.tests.test_lane_source` | `openpilot.tools.sim.bridge.gta5.navd.tests.test_lane_source` |
| `openpilot.selfdrive.navd.tests.test_map_match` | `openpilot.tools.sim.bridge.gta5.navd.tests.test_map_match` |
| file paths `openpilot/selfdrive/navd/<f>.py` | `openpilot/tools/sim/bridge/gta5/navd/<f>.py` |

New, optional: `openpilot.selfdrive.modeld.route_layout` holds `Layout`, `LAYOUTS`, `LATEST`, `layout`, `convert`,
`HEADING`, `NEXT`, `STOP`, `V1_LEN`, `LANES`, `ROUTE_LEN`, `NEXT_LANES` and `LANE_SLOTS_LEN`. The encoder imports them
all back, so `route_input.layout`, `.LAYOUTS`, `.convert`, `.ROUTE_LEN`, `.V1_LEN` etc. still resolve from the new
`...gta5.navd.route_input`; a gta5-train module that only wants the layouts (an export or a loader) can import
`route_layout` instead, which needs nothing but numpy. Names that stayed in the encoder: `RouteInput`, `NAV_BINS`,
`BIN_M`, `BIN_ZERO`, `NAV_LEN`, `PRESENT`, `HEADING_AHEAD`, `HEADING_EXT_*`, `HEADING_UNIT`, `DIST_UNIT`, `NEXT_*`,
`STOP_*`, `TANGENT_M`, `LANES_UNIT`, `LEFT`, `RIGHT`, `direction`, `wrap`, `Next`, `Decoded`, `decode`, `describe`.
`lane_slots.py` still imports `Paths` lazily, so the encoder still imports without openpilot's runtime
(`test_portable.py` passes).

A quick check of gta5-train: `git grep -n "selfdrive.navd\|selfdrive/navd"` there, then
`sed -i 's/openpilot\.selfdrive\.navd\b/openpilot.tools.sim.bridge.gta5.navd/g; s#selfdrive/navd/#tools/sim/bridge/gta5/navd/#g'`.

## What stayed in `selfdrive/` and why

- `selfdrive/navigation/` (comma's navd, `lane_guidance.py`, renderer, pmtiles, tests): the real-car path. It imported
  nothing from `selfdrive/navd/` on `gta5`, so nothing moved into it; `lane_targets.py` (`turn_targets`) is only used by
  the planner and `map/osm_lanes.py` (and through it `mapbox_server.py`), so it moved with the planner.
- `selfdrive/modeld/route_input.py`: the generic shared-memory writer and reader modeld runs (`modeld.py` imports
  `RouteInputReader`, `model_layout`, `write_model_layout`). It stays. On `gta5` it imported `Layout` and `layout` from the
  encoder, and the encoder imports the planner, `maneuvers` and `lane_slots` (and through it `map/osm_lanes.py`): after
  the move that's an import of `tools/sim` from modeld. The layouts are the writer's and reader's shared contract and
  game-agnostic (which floats of the 223 each model version reads), so commit `da10acc` moves their code verbatim into
  `selfdrive/modeld/route_layout.py` (numpy only, `LANE_SLOTS_LEN = 50` written out where the encoder took it from
  `lane_slots.py`); the encoder imports them back and asserts its lane slots are still 50 floats. Every constant, the
  three layouts (version, `take`, parts, width) and every conversion between them compared equal to `gta5`'s. modeld's
  reader and its test (`selfdrive/modeld/test_route_input.py`) now use `route_layout`, so that test no longer imports
  the encoder either.
- `selfdrive/controls/tests/test_nav_approval.py`, `selfdrive/ui/tests/test_nav_request.py`: neither imports navd or
  the bridge (desire_helper and the UI's nav_state on cereal messages only). Unchanged, and passing.
- params and cereal: no keys or messages are the moved planner's alone. The `Nav*` params (`NavDesire`, `NavDestination`,
  `NavLaneGuidance`, `NavLaneChangeStage`, ...) and `navInstruction` / `navRoute` are the interfaces comma's navd,
  controls, the UI and the bridge share; nothing in them names the moved modules. Unchanged.

## The grep

`git grep -n "tools.sim\|tools/sim" -- openpilot/selfdrive` on `da10acc`:

```
openpilot/selfdrive/navigation/lane_guidance.py:26:Game-agnostic: imports nothing from tools/sim. navd.py runs it behind ...
openpilot/selfdrive/navigation/navd.py:59:    # simulator's (GTA's tools/sim/bridge/gta5/mapbox_server.py); it gets MAPBOX_TOKEN if set, else no token
openpilot/selfdrive/navigation/tests/fixtures/mapbox/README.md:3:What navd gets from Mapbox, as the ground truth for the GTA emulator (`tools/sim/bridge/gta5/mapbox_server.py`) ...
```

No imports: these three are comments and docs in comma's navigation code (none from this branch), left as they are
(`lane_guidance.py` is guidance2-cloud's to edit). On `gta5` the same grep also had `selfdrive/navd/lane_slots.py`'s
import of `map/osm_lanes.py` and `lane_targets.py`'s docstring. Every `openpilot.tools` import under `selfdrive/` is a
`tools.lib` one (logreader, framereader, filereader, url_file, openpilotci, the CI containers, log_time_series, github_utils:
the process replay, locationd and onroad tests, `ui/tests/profile_onroad.py`). `git grep -n "selfdrive.navd\|selfdrive/navd"`
finds only the earlier `CLOUD_*.md` write-ups.

## Tests

Same command as the preflight (the moved tests at `tools/sim/bridge/gta5/navd/tests`), PyJWT installed for both:

| | `gta5` (unmoved) | `navdmove-cloud` |
|---|---|---|
| the brief's suites | 502 passed, 1 failed (503) | 502 passed, 1 failed (503), 352 s |
| failing | `map/test_cruise.py::test_live_cruises` | the same |

Compared per test id (junit XML, with the moved tests' package renamed): the same 503 tests, the same outcome for
each. On `gta5` the first run had 2 more failures for want of `jwt` (`test_navd_runs_it`, `test_navd_uses_the_host`);
both pass there with PyJWT, as on the branch.

The other tests that import the moved code, file by file, `gta5` / branch:

| file | `gta5` | branch |
|---|---|---|
| `selfdrive/modeld/test_route_input.py` | 9 passed | 9 passed (now on `route_layout`) |
| `test_portable.py` | 1 passed | 1 passed |
| `test_freeway_lanes.py` | 7 passed | 7 passed |
| `test_match.py` | 4 passed | 4 passed |
| `test_mapdrive.py` | 38 passed | 38 passed |
| `test_overlay.py` | 32 passed, 3 failed | 31 passed, 4 failed |
| `test_blindspot.py`, `test_driver.py`, `test_world.py` | abort on import (capnp, preflight 4) | the same |

`test_overlay.py` takes 34 min here and ran beside a navd_sim on both trees. Its 3 failures on both are
`test_overlay_lines_from_lane_tags[L7]`, `[X1]` and `test_lines_build_in_the_background_then_load_from_the_cache`; the
branch's 4th, `test_overlay_at_places[X1]`, passes on both trees when run alone (8.4 s / 8.8 s): it asserts the
overlay's build takes under 250 ms, a timing check under load, not the move.

ruff 0.16.7 (`uv.lock`'s): the moved files and `route_layout.py` pass (`All checks passed!`). The 46 changed Python
files together have 36 findings, the same 36 as the same files on `gta5` (compared by file and rule): B023 22, E501 6,
F841 3, B905 2, ISC002, PLR1704, B007 1 each, all in `map/osm_lanes.py`, `gta5_mapdrive.py`, `test_mapdrive.py`,
`test_nav.py`, `test_overlay.py` and `nav_replay.py`, on lines the branch doesn't touch (see open questions).

## navd_sim: before and after

`navd_sim --nav map,mapbox --model n16 --lane truth,model:0.85 --sim-seed 0 --jobs 3`, routes seed 1 (default), on
`gta5` and on `da10acc`:

| run | `gta5` | branch | Amy's PC (`gta5`) |
|---|---|---|---|
| map / n16 / truth | 247/302 | 247/302 | 247 |
| map / n16 / model:0.85 | 250/302 | 250/302 | 250 |
| mapbox / n16 / truth | 247/302 | 247/302 | 247 |
| mapbox / n16 / model:0.85 | 244/302 | 244/302 | 244 |

- `--requests` (every nav request, 593651 B): byte-identical, sha256 `13b01db7…` both.
- `--out` result JSON: identical but for each trip's `wall_s` (its wall-clock run time, `time.perf_counter`, 182 of
  184 trips differ, by up to 2.7 s); with `wall_s` taken out, the two are equal (`==` on the parsed JSON, the same
  sha256 of it sorted). Every count, change, contact and decision is the same.

## Merge with guidance2-cloud

`guidance2-cloud` has only its brief so far, so I tried it with a stand-in: a commit on `gta5` that edits
`selfdrive/navd/planner.py` (a line near the top and one at the end), merged into this branch. Git took the rename and
auto-merged the edit into `tools/sim/bridge/gta5/navd/planner.py`, no conflict. `planner.py` changes here only in its
two `import` lines (15-16), so an edit elsewhere in it carries across the same way; a new file guidance2-cloud adds
under `selfdrive/navd/` should follow the directory rename, but a new `from openpilot.selfdrive.navd...` import in
its code will need the new prefix after the merge (`git grep -n "selfdrive.navd"` finds it).

## Open questions

1. **`selfdrive/modeld/route_layout.py`** (commit `da10acc`) goes a step past a pure move: the brief asked to keep
   modeld's reader in `selfdrive/` and to say what couples it, and the acceptance grep wants no `tools/sim` import
   there, so I moved the layouts (verbatim) rather than leave modeld importing the GTA encoder. If you'd rather keep
   them in the encoder, revert `da10acc` alone: the branch then works the same, with `selfdrive/modeld/route_input.py`
   (and its test) importing `openpilot.tools.sim.bridge.gta5.navd.route_input` as the one coupling left. A smaller
   point: `LANE_SLOTS_LEN = 50` is now written in `route_layout.py` as well as derived in `lane_slots.py`; the encoder
   asserts they agree at import, so a change to the slots fails loudly until the layouts get a new version.
2. **ruff on the changed files**: the 36 findings above are on `gta5` too, on lines this branch doesn't touch; fixing
   them (closures over loop variables, unused variables, long lines, `zip` without `strict`) is behaviour-adjacent, so
   not here. Proposal: a lint-only commit on `gta5` of its own, with the tests above.
3. **`selfdrive/navigation/` comments naming `tools/sim`** (the three grep lines): docs, not imports; I left them for
   guidance2-cloud's files. If you want the grep empty, `navd.py:59` and the fixtures README could say "the GTA
   bridge's mapbox_server.py" instead.
4. **gta5-train**: not visible from here; the table above is every module path that changed. If it imports
   `selfdrive.modeld.route_input` (the writer and reader), that path didn't change.
5. The README section "navd and the GTA layer" now says navd lives beside the bridge because it plans on our map's
   lanes; the rest of it (the plan to make navd plan from a real car's inputs) is as it was.
