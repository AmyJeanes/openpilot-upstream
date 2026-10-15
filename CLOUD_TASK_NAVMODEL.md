# Cloud task: comma's nav model in the GTA V bridge (feasibility, design, prototype)

You are a Claude Code cloud session. You see only GitHub, not Amy's PC (no GTA, no GPU, no live services).

## Background

Amy drives openpilot in GTA V through a bridge: `openpilot/tools/sim/bridge/gta5/` on branch `gta5` of
`AmyJeanes/openpilot-upstream` (this repo, a fork of `commaai/openpilot`; the repo root holds an `openpilot/` folder).
The bridge has its own map stack: a GTA road map as an OSM pbf in a fake lat/lon frame (`map/gta5_map.py`, `to_game`
/ the inverse), a Valhalla router (`map/router.py`, `Router.route`), and nav messages of its own (`gta5_nav_pub.py`,
`gta5_nav_msgs.py`, `gta5_brain.py`, the NavDesire work, tests `test_nav*.py`). A rule-based "map driver" (expert)
drives GTA to record training data; openpilot itself also drives in GTA.

comma has just published, on `commaai/openpilot` branch **`nav-model-integration`** (base: master's nav alpha #39066,
commit 4c362ca32), a driving model that takes navigation inputs:

- `nav_map`: a 256x256 uint8 image from `selfdrive/navigation/mapsd.py` (VisionIPC stream `VISION_STREAM_MAP`),
  rendered by `navigation/renderer.py` from Protomaps roads in PMTiles (1 m/px, ego 3/4 down, heading along the route,
  route grey, roads white, bridges cut over roads), centred on the route and moved ahead by the distance driven since
  the GPS fix.
- `navigation`: a 47-float vector (`navigation/helpers.py navigation_input`): one-hot of the next maneuver's
  **Valhalla maneuver type** (46 types; only types > 6) plus its distance / 100, from `navInstruction.allManeuvers`
  (navd converts Mapbox steps into Valhalla types, because the model was trained on Valhalla's).
- `modeld.py` reads both with timeouts (map 0.5 s, nav 2 s). Two models on the branch: "rl nav driving model",
  "n16 nav-wm model".

Our router already is Valhalla, so the navigation vector can come straight from our routes. The map image needs our
GTA roads in the PMTiles/Protomaps form the renderer reads (it fetches hosted basemap byte ranges via a signed URL in
`navigation/tiles.py`; for GTA it must read a local file).

## Goal

Work out whether and how openpilot running comma's nav model can drive GTA following our routes, design it, and build
and unit-test the parts that don't need GTA. Deliver a design Amy can read and a prototype a local session can try in
GTA.

## Step 0: preflight (stop on failure)

Record pass/fail with one line of evidence in `CLOUD_RESULT_NAVMODEL.md` under "Preflight":

1. Repos: this repo's `gta5` (shallow, single branch is fine: `git clone --depth 1 --branch gta5 --single-branch`;
   the full repo is very large) and `commaai/openpilot` branch `nav-model-integration` (shallow). Note both hashes.
2. Data: `AmyJeanes/gta5-distill` (private) branch `cloud-data` holds `cloud/live18/map1/` with the GTA map
   (`gta5.osm.pbf`, `roads.json`, `lanes.json`, `paths.jsonl`); verify its sha256 against `cloud/MANIFEST.md`.
3. Dependencies from PyPI only (light: numpy, pyosmium, pyvalhalla, pmtiles, mapbox-vector-tile or similar). If an
   install is blocked, record it and design around it rather than working around the block.
4. Valhalla: build tiles from the GTA pbf with pyvalhalla (`valhalla_build_config` / `valhalla_build_tiles`, call the
   binaries under `site-packages/valhalla/bin` directly) and route one GTA trip with `Router` or the Actor API.
5. Push access: create branch `navmodel-cloud` in this repo (from `gta5`) with a commit adding only
   `CLOUD_RESULT_NAVMODEL.md`, and push it.

If 1, 2 or 5 fails, stop and push (or report) the preflight. 3 and 4 failing narrows the prototype; carry on with design.

## Work

1. **Read and explain the model's contract** from the branch (modeld, navd, mapsd, renderer, helpers, tests): input
   shapes, dtypes, timing, what happens without a map or route, what "as in training" implies (route-centred pose,
   ahead-projection), and whether the two models differ in inputs. Note anything still unclear.
2. **Navigation input from our routes:** map our Valhalla route (maneuvers with type, length, begin shape index) to
   `navInstruction.allManeuvers` the way navd measures them along the route, so `navigation_input` gives the vector
   the model expects. Prefer feeding Valhalla types directly over a Mapbox round trip, and say how that fits navd's
   message schema. Unit-test it on GTA routes from preflight 4.
3. **Map image for GTA:** build a local PMTiles file of GTA roads in the Protomaps schema the renderer decodes
   (`vector_tile.py`: layer names, `kind` / `kind_detail`, bridge and level fields, zoom 15), from the GTA pbf.
   Make the tile source take a local file. Render sample frames along a GTA route with the unmodified `renderer.py`
   and commit a few PNGs plus a contact sheet. Check road classes map to the right widths and bridges cut correctly
   (GTA has many overpasses).
4. **Bridge integration design:** how the bridge publishes `navRoute` / `navInstruction` and runs (or replaces)
   mapsd from GTA's pose; which existing bridge nav code it replaces or sits beside (NavDesire, `gta5_nav_pub`);
   recording the two inputs alongside our training data; how the map driver's routes become the model's routes.
5. **Fit with the plan:** the work will soon be ported onto sunnypilot, which syncs comma's master; upstream removed
   experimental mode ("A New Dawn", f50108b4a). Keep the GTA parts in the bridge folder and avoid edits to selfdrive
   beyond a local-file tile source, so the port stays easy.
6. **Local test plan:** exact steps for a local session with GTA: which models, which processes, the trips to drive
   (motorway exits, city turns, a roundabout), what to measure (turns taken correctly, missed exits, interventions)
   and the comparison against our current default model.

## What cannot be done here

GTA, GPU model runs on camera frames, the live bridge. A CPU smoke test that loads a nav model and runs one frame with
synthetic inputs is welcome if feasible; leave anything else as a checkable TODO in `CLOUD_RESULT_NAVMODEL.md`.

## Outputs

- Branch **`navmodel-cloud`** in `AmyJeanes/openpilot-upstream` (from `gta5`). Never push to `gta5` or `master`.
  Bring comma's nav files over only as needed, with their source commit noted.
- `CLOUD_RESULT_NAVMODEL.md`: preflight, findings per work item, design, deviations, TODOs for local, and a short
  plain-English summary at the top for Amy.
- Commit messages end with `Assisted-by: Claude Code`; no Co-Authored-By, session-link or Signed-off-by trailers.
- Do not modify the data pack. No secrets in commits.

## Stopping rule

Work through the items in order; after each, commit, push `navmodel-cloud` and update the result file. If one is
blocked, record why, push, and continue with the next. Do not repeat a failed approach more than twice.
