# Cloud task: move the GTA bridge's nav planner out of `selfdrive/` (no behaviour change)

You are a Claude Code cloud session. You see only GitHub and the internet allowed by the environment, not Amy's PC
(no GTA, no GPU, no live services).

Repos:
- `AmyJeanes/openpilot-upstream`: read, and push branch `navdmove-cloud` only. Clone it shallow and single-branch:
  `git clone --depth 1 --branch navdmove-cloud --single-branch`. The full repo is very large.
- `AmyJeanes/gta5-distill` (private): read only, branch `cloud-data`, data under `cloud/` (see `cloud/MANIFEST.md`).

The session may start on another branch. Work on `navdmove-cloud`, which is `gta5` (e55cb3b45) plus this brief.

Another cloud session (`guidance2-cloud`) edits `selfdrive/navd/planner.py` (a small low-speed rule) and
`selfdrive/navigation/lane_guidance.py` at the same time. Move files with `git mv` in a commit of their own, before
any edit to them, so git's rename detection carries the other branch's edits across; Amy's local session merges
`guidance2-cloud` first and this branch after.

## Why

Amy's rule: nothing GTA-specific under `openpilot/selfdrive/`; GTA code lives in `openpilot/tools/sim/bridge/gta5/`.
`openpilot/selfdrive/navd/` is our full-map nav planner: a library the GTA bridge drives in process, planning on our
map's lanes (`Route.info`, OSM lane tags), which only GTA has. It's the last GTA code under `selfdrive/`:
`selfdrive/navd/lane_slots.py` still imports `tools/sim/bridge/gta5/map/osm_lanes.py`. comma's navd
(`selfdrive/navigation/navd.py`) with the Mapbox-only lane guidance (`selfdrive/navigation/lane_guidance.py`) is the
real-car path and stays where it is. Amy agreed: move it now with no behaviour change; delete the route-input lane
slots later, once the n16 finetune replaces the models that read them.

## The task

1. Move `openpilot/selfdrive/navd/` (the planner, `inputs.py`, `lane_slots.py`, `maneuvers.py`, `map_match.py`,
   `route_input.py`, their tests, and whatever else in it only the bridge uses) under `openpilot/tools/sim/bridge/gta5/`
   (e.g. `.../gta5/navd/`; pick a name that reads well beside `gta5_navd.py`) and update every import: the bridge,
   its tools and tests, `navd_sim.py`, scripts, docs and READMEs that name the old paths.
2. Anything game-agnostic that comma's navd or the lane guidance uses stays under `selfdrive/` (e.g.
   `selfdrive/navd/lane_targets.py`'s `turn_targets` if `selfdrive/navigation/` imports it: move it into
   `selfdrive/navigation/`). After the move, `selfdrive/navigation/` must import nothing from `tools/sim`, and nothing
   under `selfdrive/` may import from `tools/sim` (grep it; the `tools.lib` log readers are fine).
3. What else under `selfdrive/` reads the moved code: `selfdrive/modeld/route_input.py` (a generic reader of a
   shared-memory vector: keep it in `selfdrive/` if it doesn't import the planner; else say what couples it),
   `selfdrive/controls/tests/test_nav_approval.py`, `selfdrive/ui/tests/test_nav_request.py`, params, cereal. Keep
   the generic parts in `selfdrive/`; list any coupling you can't remove cleanly under "Open questions", with a proposal.
4. Amy's training repo (`gta5-train`, not available to you) imports openpilot modules by path, possibly some of
   these. List every old -> new import path in the result so she can update it. No compatibility shims under
   `selfdrive/`.

## Acceptance

- `git grep -n "tools.sim\|tools/sim" -- openpilot/selfdrive` shows only the log-reader imports from `tools.lib`.
- The same test suites pass as on `gta5` (`openpilot/selfdrive/navigation/tests`, the moved planner tests,
  `tools/sim/bridge/gta5/test_nav.py`, `test_nav_msgs.py`, `test_lane_slots.py`, `test_lane_numbering.py`,
  `test_navd_sim.py`, `test_mapbox_server.py`, `test_navd_path.py`, `selfdrive/controls/tests/test_nav_approval.py`,
  `selfdrive/ui/tests/test_nav_request.py`, and the map tests under `tools/sim/bridge/gta5/map`).
- No behaviour change: navd_sim before and after (`--nav map,mapbox --model n16 --lane truth,model:0.85`, seed 0) give
  identical result JSON (compare it; Amy's PC on `gta5`: n16/truth 247 both navs, n16/0.85 map 250, mapbox 244).
- ruff clean on the moved and changed files.

## Step 0: preflight (stop on failure)

In `CLOUD_RESULT_NAVDMOVE.md` under "Preflight", pass/fail with one line of evidence each: the repo hashes; the data
(sha256 against `cloud/MANIFEST.md` for `live18/map1/{paths.jsonl,nodes.npz}*` and `live21b/*`; the map assembled as
in `CLOUD_RESULT_NAVMAPBOX.md`'s preflight: live21b's `gta5.osm.pbf`, `lanes.json`, `roads.json` + live18's
`paths.jsonl`, `nodes.npz`; `GTA5_MAP` and `~/gta5map_lanes` pointing at it); dependencies from PyPI (msgq built from
its public source works; native params won't, as before); Valhalla on :8002 from the pbf; the suites above and the
navd_sim baseline JSON on the unmoved tree. Commit and push the preflight first.

## Rules

- Commits: the move (`git mv` only) first, then the import fixes, then anything else, each with what and why, ending
  with the line `Assisted-by: Claude Code`. No other trailers. Push after each.
- No behaviour change, no renames beyond the move, no retuning; stay inside the brief, with anything else under
  "Open questions".

## Hand-over (`CLOUD_RESULT_NAVDMOVE.md`)

Preflight; the new layout; every old -> new import path (for gta5-train); what stayed in `selfdrive/` and why; the
grep and the test results; the navd_sim comparison; open questions.
