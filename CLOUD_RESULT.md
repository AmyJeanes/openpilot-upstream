# Cloud result: map phase 1 (openpilot side)

Branch `phase1-cloud` from `gta5` (54f983874b52). The full record is `CLOUD_RESULT.md` on branch `phase1-cloud` of
`AmyJeanes/gta5-distill`; this file lists only what changed here.

## Preflight

- Shallow single-branch clone of `gta5` at 54f983874b52; the map code imports with numpy, scipy, shapely, pyosmium,
  pillow from PyPI.
- `openpilot/tools/sim/bridge/gta5/test_lane_numbering.py`: 30 passed; `map/test_*.py`: 266 passed (no Valhalla, no GTA).
- Push access: this commit.

## Steps K, A, G1, R

No map or router code changes were needed here. G0 is already on `gta5`: the `hatched` kind, `crosses_hatched`, and
the test_lane_numbering cases for a hatched ring.

The distill side (`gores.py`, `themes.py`) calls this branch's code read-only, to give the router's verdicts offline:
`router.LaneLines`, `hairpin`, `separated`, `oncoming`, `Router._shortcut`, `OsmLanes`, `Paths` and `osm_pbf`.

One count to know: on live18, `Router.crossings`' rule bans 361 directed drivable shortcut links, i.e. 173 node pairs
counted in their first direction. NOTES' 163 predates the oncoming-carriageway rule (11 pairs).
