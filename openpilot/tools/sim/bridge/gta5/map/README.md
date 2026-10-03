GTA V road map
==============

An OpenStreetMap file of Los Santos and Blaine County, made from the game's own road network (its vehicle path nodes),
so nav can route as a car's navigation would, with a standard router, rather than follow the game's GPS. The map view
shows the roads, the car and its route in a browser.

The map is the game's data, so it isn't in the repository; build it from your copy of the game.

## Building the map
1. Dump the path nodes and street names from the game's archives with `ynddump` (Windows, .NET 8 SDK). It builds on
   [CodeWalker](https://github.com/dexyfex/CodeWalker)'s library, which reads the Enhanced build from source, and only
   reads the game's files:
   ```powershell
   git clone https://github.com/dexyfex/CodeWalker C:\src\CodeWalker
   dotnet run -c Release --project ynddump -p:CodeWalker=C:\src\CodeWalker -- "<game folder>" paths.jsonl
   ```
2. Convert it, in a Python environment with `osmium` and `pyvalhalla` (`uv venv ~/gta5map/.venv && uv pip install osmium pyvalhalla numpy`):
   ```bash
   python ynd_to_osm.py paths.jsonl ~/gta5map/gta5.osm.pbf             # the map
   python osm_to_roads.py ~/gta5map/gta5.osm.pbf ~/gta5map/roads.json  # the map view's roads
   valhalla_build_config --mjolnir-tile-dir ~/gta5map/tiles --mjolnir-tile-extract ~/gta5map/tiles.tar \
     --mjolnir-timezone '' --mjolnir-admin '' > ~/gta5map/valhalla.json
   valhalla_build_tiles -c ~/gta5map/valhalla.json ~/gta5map/gta5.osm.pbf
   ```
   (run from the repository root with `PYTHONPATH=.`, as `python openpilot/tools/sim/bridge/gta5/map/ynd_to_osm.py ...`).

## Using it
```bash
valhalla_service ~/gta5map/valhalla.json 2   # the router, on port 8002
# terminal 2, the bridge: the map view on http://localhost:8793/, and nav on our routes
GTA5_MAP=~/gta5map GTA5_ROUTER=http://localhost:8002 ./openpilot/tools/sim/run_bridge.py --simulator gta5
```
Without `GTA5_ROUTER`, nav follows the game's GPS as before. A waypoint set on the game's map, or a tap on the map view
("Navigate here"), sets the destination; the later one wins, and the map view's isn't passed to the game's GPS.
The route sets off the way the car faces, and when the car leaves it nav routes again from where the car is.
`map_view.py roads.json --state <file>` shows the map alone, with a state from a file.

## The map
- Ways carry the lanes each way (`lanes:forward` / `lanes:backward`, `oneway`) and street names. Road classes are
  guessed (GTA has none): motorway for its highway nodes, primary with two lanes or more one way, service for car parks
  and alleys (nodes switched off for traffic, or without GPS), track off-road.
- Traffic lights are `highway=traffic_signals` on the stop line node. Flags with no OSM equivalent keep a `gta:` prefix:
  junction, no left / right turn, slip lane, keep left / right, left turn only lane.
- openpilot's driving model can't turn back on itself, so every move that turns back more than 135 degrees is a
  `no_u_turn` restriction: at a node, and through up to four short links (40 m) as through a median gap or a turning
  loop. GTA's nodes allow them everywhere. The short two-way links joining a divided road's carriageways away from
  junctions are left out. A trip from a dead end then has no route.
- Ped nodes and boat nodes are left out. The links GTA marks "don't use for navigation" stay in (`gta:no_nav`): without
  them most of the map is cut off from the rest.
- Game coordinates (metres) map to degrees about (0, 0), so the map sits on the equator: `gta5_map.py`.
