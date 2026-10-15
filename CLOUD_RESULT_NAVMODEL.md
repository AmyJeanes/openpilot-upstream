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
