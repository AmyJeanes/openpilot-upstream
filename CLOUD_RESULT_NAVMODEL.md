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

## 2. The navigation input from our routes

`openpilot/tools/sim/bridge/gta5/navmodel/nav_input.py` (`ModelNav`), tests `navmodel/test_nav_input.py` (7, pass:
`PYTHONPATH=. python openpilot/tools/sim/bridge/gta5/navmodel/test_nav_input.py`) on three GTA routes recorded from
this session's Valhalla (`navmodel/fixtures/routes.json`: city turns 3.2 km, a freeway trip 5.1 km with a left exit,
a 10.3 km trip with exits both sides). comma's `selfdrive/navigation/{__init__,helpers,route}.py` are copied unchanged
from `afbf2e5` (the module uses `navigation_input` and `RouteTracker` as they are).

**How it fits navd's schema.** `navInstruction.allManeuvers` is `List(Maneuver{distance, type :Text, modifier :Text})`
(the fork adds `primaryText`), and modeld reads `type`/`modifier` back through `valhalla_type()`. So the Valhalla type
goes in as the Mapbox words that `valhalla_type()` maps back to exactly that type (`WORDS`), no Mapbox round trip and no
schema or selfdrive change. Lossless for every type navd can produce (test: all 46 types read back). The types navd
never produces go as navd would give them: 7 becomes -> 8, U-turn right 12 -> 13, merges 25/37/38 -> Mapbox "merge"
(skipped by the model, as on the car). Ferries/transit have no words and are left out (GTA has none).

**Distances, as navd measures them.** `RouteTracker` over the route's (lon, lat) (the same points as navRoute), each
maneuver at its `begin_shape_index`; progress = projection of the car's position, monotone, ±window as navd. Test:
identical (1e-6 m) to navd's own formula over Mapbox-style step geometries built from the same route, and equal to game
metres along x 6371008.8/6378137 (the mean-radius 0.11 %). Passed maneuvers drop out; a fix behind doesn't move progress
back; modeld's `v_ego * age` dead reckoning moves to the next maneuver once one is passed (tested).

**GTA's maneuvers vs a real map's.** Over 31 random GTA trips (234 km) Valhalla gave 340 maneuvers > 6, 77 of them
within 30 m of the one before: GTA lays freeways as lane groups that split and rejoin ("Keep left at the fork" x3 within
40 m) and junctions as links across medians ("Bear left." onto an unnamed link, then "Bear right onto Vespucci Blvd"
1 m on). Two tidy-ups, both options of `ModelNav`:
- turns through a junction's links are one (`selfdrive/navd/maneuvers.junction_maneuvers`, the bridge's own): 340 -> 317;
- a run of keeps each within 50 m of the one before is one keep at the first, on the last's side and road (`fold_keeps`):
  -> 281 maneuvers, 23 within 30 m.
The bridge's `Route.forks` (`paths.jsonl`) didn't tell these lane-group keeps from real forks (it finds forks only at a
few of them), so the fold is by distance. Whether the model wants the raw keeps, the folded ones, or none of the
lane-group ones is a GTA test (item 6).

Other findings:
- No on-ramp types (17-19) in any of the 31 trips: entering a freeway from GTA's streets comes as a turn or "Take the
  exit" (20/21) or a keep, because of the map's guessed road classes. The model will see fewer ramp types than on a
  real map.
- The map has no `junction=roundabout`, so Valhalla never gives 26/27; GTA's roundabouts route as turns.
- The fork's existing `navInstruction.allManeuvers` (`gta5_nav_msgs.instruction`, for the UI) already goes through
  `valhalla_type` correctly (its words are close to ours), but leaves out continues (8), keeps only 5, folds no keeps, and
  measures from the bridge's own route position. For the model, item 4 swaps in `ModelNav`'s list.

## 3. The map image for GTA

**Tiles.** `navmodel/build_pmtiles.py gta5.osm.pbf gta5_roads.pmtiles` (pyosmium, shapely, mapbox-vector-tile,
pmtiles; 17 s): 72 559 ways -> 53 z15 tiles, 581 KB, a gzip PMTiles v3 archive of gzip MVT, layer `roads`, extent 4096
with a 64-unit buffer, as Protomaps': `kind` highway (motorway, motorway_link) / major_road (trunk, primary, secondary,
tertiary and links) / minor_road (residential, unclassified, road, service, track) / other (living_street, pedestrian...)
/ path (footway etc., not drawn); `kind_detail` = the OSM highway (service roads: their `service=*`; GTA's have none);
`is_bridge` + `level=1` on the 3 297 `bridge=yes` ways (`is_tunnel` where tagged: GTA's map tags none); `name`,
`oneway`. Read by comma's **unmodified** `pmtiles.py` / `vector_tile.py` / `renderer.py` (copied from `afbf2e5`).

**Local tile source.** `selfdrive/navigation/local_tiles.py` (`LocalTiles`: `size`, `read`, `prefetch`, `close`, the
interface `PMTiles` takes) and 4 lines in `mapsd.py`: `NAV_TILES=/path/gta5_roads.pmtiles` uses it in place of
`RemoteTiles`. This is the only change to comma's navigation files.

**Frames.** `navmodel/render_frames.py tiles out --route fixtures/routes.json:<name>` renders as mapsd does (tracker
following the car in 10 m fixes; it can't jump: the search window is -30..+100 m), one PNG per frame plus a contact
sheet. Committed in `navmodel/samples/` (a `.gitattributes` there keeps these small PNGs out of LFS, whose host is
HuggingFace): `contact_city.png`, `contact_freeway.png`, `contact_exits.png` (every 400/450 m of the three fixture
routes; all frames valid), `crossings_cut.png`, `crossings_bridge_on_bridge.png`, `crossings_level.png`,
`contact_lat0_vs_lat34.png`, and three single frames.

**Road classes -> widths.** Test `navmodel/test_map_frames.py` (6, pass; the whole-map one with `GTA5_MAP`) checks each
GTA class gets the renderer's style: motorway/trunk/primary MAJOR (19.0 px at GTA's equator), secondary/tertiary
MEDIUM (15.5; GTA's map has none), motorway_link/trunk_link/residential/unclassified STREET (9.9), service MINOR (6.7),
track TRACK (7.0). Whether Protomaps files tracks under `minor_road` (drawn) or `path` (not drawn) isn't clear from
the renderer (it has a track style, so we draw them); `--tracks path` gives the other.

**Bridges.** `navmodel/check_crossings.py` finds every crossing of two drawn roads with no shared node and compares
heights (`ele`): of 2 454, 702 are at grade (GTA's links crossing, drawn like a junction), and of 1 752 overpasses
(> 4 m apart) **1 208 cut correctly** (the upper is a bridge, the lower is cut under it; test `test_a_bridge_cuts_...`
checks the pixels as comma's own test does), **542 are bridge over bridge** (482 of them car-park decks, 46
major-over-major stacked interchanges): the renderer has one bridge level, so these cross without a cut, as on a real
map; **2 untagged** (no cut) and **0 inverted**. Layers aren't read by the renderer.

**Deviations from a real map's look** (for the GTA test to judge):
- GTA lays a freeway carriageway as several side-by-side lane-group ways where OSM has one. Each is a 19 px MAJOR
  stroke, so the band under the route is wider: median 36 px on the freeway trip (p25 19, p75 61; 58 % of frames over
  30 px), 27 px on the city trip. A fix would be carriageway centrelines in the builder (group one-way ways side by
  side on one level, as `map/side_by_side.py` finds them, and emit their middle line); not done here, a TODO.
- Roads are drawn about 10 % wider than at comma's latitudes, because GTA sits on the equator.
  `build_pmtiles.py --lat0 34` plus `frame.py` put GTA about 34 N (Web Mercator, 1 game metre = 1 px still; tested to
  0.05 % at the origin, 0.2 % across the map). The route and fix must then be in the same frame (item 4).
  `contact_lat0_vs_lat34.png` shows the same frames both ways (road pixels about 8 % fewer at 34 N).
- Car parks: 19 232 service ways (GTA's car-park aisles and drives) draw as MINOR roads, as a real map's would.
