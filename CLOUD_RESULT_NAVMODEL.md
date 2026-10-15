# Cloud result: comma's nav model in the GTA V bridge

Brief: `CLOUD_TASK_NAVMODEL.md`. Status: preflight done, work items in progress.

## Preflight

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Repos | PASS | this repo `gta5` @ `175c09aa69eb` (`navmodel-cloud` = `gta5` + the brief, `87c80a8`); `commaai/openpilot` `nav-model-integration` @ `afbf2e547e93` ("n16 nav-wm model"), on `4c362ca` nav alpha (#39066) as the brief says |
| 2 | Data | PASS | `gta5-distill` `cloud-data` @ `17c5a5bf741b`: all six files of `cloud/live18/map1/` match `cloud/MANIFEST.md` (pbf `65e320c4…9d42`, roads `182d3007…6329`, lanes `79d5cd76…ecf2`, paths `1e5a61ab…52c2`, nodes `0b71e974…5a67`, log `f41c8370…2d77`) |
| 3 | PyPI deps | PASS | venv, Python 3.13: numpy 2.5.4, osmium 4.3.1, pyvalhalla 3.9.1, pmtiles 3.8.1, mapbox-vector-tile 2.2.0, pillow 12.3.0 |
| 4 | Valhalla | PASS | `valhalla_build_tiles` on `gta5.osm.pbf` in 11 s; bridge `Router.route` (215,-810) → (-1035,-2730): 296 points, 3186 m, 14 maneuvers (types 2,10,15,15,9,16,9,15,24,10,15,24,23,5) |
| 5 | Push | PASS | this commit on `navmodel-cloud` (the branch already held `gta5` + the brief; added this file only) |

Note: `valhalla_build_config` isn't in `site-packages/valhalla/bin` in pyvalhalla 3.9.1; it's the venv's console script
(`<venv>/bin/valhalla_build_config`). `valhalla_build_tiles` / `valhalla_service` are under `site-packages/valhalla/bin`.

## 1. The model's contract (comma `nav-model-integration` @ `afbf2e5`)

Read from `selfdrive/modeld/modeld.py` (hunk of `8b8ce13` "modeld wired"), `selfdrive/navigation/{navd,mapsd,renderer,
route,helpers,vector_tile,pmtiles,tiles}.py`, `tests/test_navigation.py`, `system/manager/process_config.py`, cereal.

**Processes.** `navd` (1 Hz) and `mapsd` (20 Hz) run when `NavigationEnabled` is set, onroad. navd routes with Mapbox
(`driving-traffic`, through comma's API or `MAPBOX_TOKEN`) from the latest GPS fix to the `NavDestination` param, and
publishes `navRoute` (the whole route's (lon, lat), resent every 10 s) and `navInstruction` (banner fields plus
`allManeuvers`). mapsd renders a frame from `navRoute` + the GPS fix and serves it on VisionIPC
(`"mapsd"`, `VISION_STREAM_MAP`, 4 buffers of 256x256 one-channel).

**Inputs, as modeld feeds them** (only to a model whose `input_specs` has the name; others ignore them):

| input | dtype | shape | from | timeout | when missing |
|---|---|---|---|---|---|
| `nav_map` | uint8 | 256x256 frame (the model's exact spec shape is in the pkl; modeld assigns the 2-D frame into it, so it broadcasts) | `vipc_client_map.recv(0)`, non-blocking, last frame kept | 0.5 s since the last frame | 0 (an all-black map) |
| `navigation` | float32 | 47 | `navigation_input(navInstruction.allManeuvers, v_ego * age)` | 2 s since the message, and the message `valid` | 0 |

modeld's packing was made dtype-aware for this (`nav_map` is the first non-float32 host input).

**`navigation` (47 floats).** The first maneuver in `allManeuvers` whose Valhalla type is > 6 and which is still ahead
after subtracting `v_ego * (now - recv_time)` (navd's 1 Hz is dead-reckoned between messages): one-hot at index =
type (0..45), and `[46]` = metres to it / 100, unclipped (2 km gives 20.0). The type comes from Mapbox words through
`valhalla_type(type, modifier)`: depart/arrive -> 1..6 (never used); turns by modifier -> 8,9,10,11,13,14,15,16; on ramp
17/18/19; off ramp 20 (straight and right), 21; fork 22/23/24; roundabout/rotary 26; exit roundabout 27; notification
28; merge -> none (skipped; the next one shows). So navd can produce only {8..11, 13..24, 26, 27, 28}: never Valhalla's
7 (becomes), 12 (U-turn right), 25/37/38 (merges) or ferries/transit. After the last turn the vector is all zero, even
approaching the destination (arrive is <= 6).

**How navd measures distance.** `RouteTracker` over the concatenated step geometries: equirectangular metres (mean
Earth radius), each maneuver at its step's first point; progress = projection of the GPS fix onto the route, searched
-30..+100 m around the last progress and never decreasing. `allManeuvers[i].distance` = maneuver's distance along -
progress, for those with distance >= progress (the passed ones drop out).

**`nav_map` (256x256 uint8)**, "as in training": the frame is **not at the GPS fix**. mapsd projects the fix onto the
route (same `RouteTracker`), adds `gps.speed * (now - fix receive time)`, and renders at that point of the route,
headed along the route's chord from 50 m behind to 50 m ahead. 1 m per pixel, ego at (128, 192) (3/4 down), heading up.
Roads white (255), strokes from the Protomaps class (below), drawn in levels: tunnels, ground, bridges, each upper level
first cutting a black band (stroke + 3 px) through those below, open at way ends; the route on top in grey (128, 5 px
wide); a white ego arrow with a black outline last. Road widths follow a latitude-dependent style zoom
(`log2(cos(lat) * C / 256) - 1`). The frame is black with timestamp 0 when there is no route, no fresh fix (< 2 s),
a tile is missing, or no road centreline lies within 100 m of the ego (`valid` false); a failed fetch retries after 10 s.
The renderer's styles by Protomaps `kind_detail` (= OSM highway, or the service type):
MAJOR motorway/trunk/primary (and `kind=highway` without detail), MEDIUM secondary/tertiary, STREET residential,
unclassified, living streets and the major links (motorway_link, trunk_link, primary_link), MINOR service, driveway,
parking aisle, alley and secondary/tertiary links, TRACK track; platform/pier/raceway hidden. Only layer `roads`, line
features with `kind` in highway/major_road/minor_road/other; `is_bridge` / `is_tunnel` booleans. Tiles z15.

**The two models.** `a52113e` "nav driving model", then `3ec09ae` "fix overflow inputs in map model", `a03bb1d` "rl nav
driving model (80eedcdd/12864)" and `afbf2e5` "n16 nav-wm model". All are the **big (chestnut) model**:
`big_driving_tinygrad.pkl`, precompiled for `USB+AMD:LLVM` (comma's USB eGPU). The ONNX
(`big_driving_supercombo.onnx`, 815 MB) is in the tree from `a03bb1d` (the rl model) and was not updated by `afbf2e5`,
which changed only the pkl. modeld feeds both through the same code, by input name, so by the code their inputs are
the same names; their exact `input_specs` (shapes, and whether n16 has more inputs) are only in the pkl/ONNX.

**What this means for GTA.** Amy's PC has no chestnut; the fork already runs a big model compiled locally
(`MODELD_DEV`, `MODELD_BIG_ONNX` -> `local_big_driving_tinygrad.pkl`, `MODELD_BIG=1`). The **rl** nav model can go that
way from its ONNX. The **n16** model has no ONNX on the branch, so it can't be compiled for her GPU; it needs comma's
ONNX (or a chestnut). The fork's modeld lacks comma's dtype-aware packing and the two nav inputs, so the `8b8ce13`
modeld hunk has to come over for a local run (it's upstream code; sunnypilot gets it with the sync).

**GTA's fake lat/lon** (`gta5_map.to_lat_lon`, equatorial radius about (0, 0)) puts GTA on the equator: 1 game metre is
1 px in the renderer (mercator at lat ~0), and `RouteTracker`'s mean-radius metres are 0.11 % short of game metres
(1 m per km; negligible). But the latitude-dependent style draws roads wider at the equator than at the latitudes comma
drives (MAJOR..TRACK stroke px: lat 0 = 19.0/15.5/9.9/6.7/7.0; lat 34 = 17.3/14.0/8.8/5.9/6.3; lat 47.6 = 15.6/12.6/
7.7/5.1/5.5): about 10 % wider than Los Angeles. See item 3 for a frame option.

**Unclear (needs the pkl/ONNX, which this session can't fetch):**
- the exact `input_specs` of `nav_map` / `navigation` and whether n16 differs (e.g. a map history); TODO local: print
  `jits['input_specs']` / the ONNX inputs.
- whether the model clips `navigation[46]` (the "overflow inputs" fix suggests something was clipped or rescaled).
- whether training saw Valhalla 7/25/37/38 (the helpers' comment says merges never); we follow what navd produces.
- the training renderer itself (the code says it is "styled after" it; we take renderer.py as the reference).

Blocked here: `huggingface.co` (comma's LFS host) is denied by this environment's network policy, so no model file
could be fetched and the CPU smoke test is a local TODO.
