# comma's nav model in GTA

comma's nav driving models (commaai/openpilot `nav-model-integration`, afbf2e5) take two inputs besides the camera:
`nav_map`, a 256x256 map frame mapsd renders on VisionIPC, and `navigation`, the next maneuver's Valhalla type and
distance from `navInstruction.allManeuvers`. Here the bridge gives them both from its own map and router. The design,
findings and the GTA test plan are in `CLOUD_RESULT_NAVMODEL.md` at the repository root.

| file | what |
|---|---|
| `nav_input.py` | `ModelNav`: a router.Route's Valhalla maneuvers as allManeuvers, measured as comma's navd measures them |
| `build_pmtiles.py` | the map's roads as a local PMTiles file in the Protomaps schema comma's renderer reads |
| `frame.py` | the map frame: the bridge's own (lat0 0), or about another latitude for the renderer's road widths |
| `render_frames.py` | map frames along a route as mapsd renders them, and a contact sheet |
| `check_crossings.py` | how the map's overpasses draw (cut, bridge over bridge, untagged) |
| `gta5_mapsd.py` | comma's mapsd over the bridge's navRoute and GNSS fix, from the local tiles, in either frame |
| `modeld_nav_inputs.patch` | comma's modeld hunk for the two inputs (8b8ce13), for the fork's modeld |
| `samples/` | frames and contact sheets (kept out of LFS by its `.gitattributes`) |

```bash
# once per map (pyosmium, shapely, mapbox-vector-tile, pmtiles)
python build_pmtiles.py ~/gta5map/gta5.osm.pbf ~/gta5map/gta5_roads.pmtiles
# terminal 3, beside openpilot and the bridge: the map frames
python -m openpilot.tools.sim.bridge.gta5.navmodel.gta5_mapsd --tiles ~/gta5map/gta5_roads.pmtiles
# the bridge publishes navRoute and navInstruction, with the model's maneuvers
GTA5_NAV_MSGS=1 GTA5_NAV_MODEL=1 GTA5_MAP=~/gta5map GTA5_ROUTER=http://localhost:8002 ./openpilot/tools/sim/run_bridge.py --simulator gta5
```

Tests: `python test_nav_input.py`, `python test_map_frames.py` (with `GTA5_MAP` also the whole map),
`python test_bridge_nav.py`, from the repository root with `PYTHONPATH=.`.
