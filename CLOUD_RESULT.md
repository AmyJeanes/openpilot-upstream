# Cloud result: map phase 1 (openpilot side)

Branch `phase1-cloud` from `gta5` (54f983874b52). The full record is `CLOUD_RESULT.md` on branch `phase1-cloud` of
`AmyJeanes/gta5-distill`; this file lists only what changed here.

## Preflight

- Shallow single-branch clone of `gta5` at 54f983874b52; the map code imports with numpy, scipy, shapely, pyosmium,
  pillow from PyPI.
- `openpilot/tools/sim/bridge/gta5/test_lane_numbering.py`: 30 passed; `map/test_*.py`: 266 passed (no Valhalla, no GTA).
- Push access: this commit.
