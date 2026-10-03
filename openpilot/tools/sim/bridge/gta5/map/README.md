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
   python ynd_to_osm.py paths.jsonl ~/gta5map/gta5.osm.pbf --sidecar ~/gta5map/nodes.npz  # the map
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
- Speed limits (`maxspeed`, mph) come from the road class and the speed class GTA gives each node (slow, normal, fast,
  faster), by the table in `ynd_to_osm.py`'s `LIMITS`. Nearly all the city is "normal", so the road class sets it there:
  30 with one lane each way, 40 with two or more (and on country roads), 20 in car parks and alleys, 15 where it's slow
  (docks, runways). "Fast" is the open country road (Route 68, Joshua Rd: 50) and the freeway in places (55); "faster"
  is the freeway (65). The city's slower freeway stretches are 50, its one-lane ramps 40. A street gets its most common
  limit along its length, and a stretch shorter than 60 m on the way through a junction takes the limit on both sides
  of it, so the limit doesn't change at every junction (GTA's junction links belong to the crossing street, and lane
  counts change at them).
- Nodes carry their height, `ele` (game z, metres). `--sidecar` also writes the roads as arrays for map matching:
  `x`, `y`, `z`, `node_id` per node; per link (one per OSM way; `way_id` is the way's, as in Valhalla's
  `trace_attributes`) `a`, `b` (node indices, in its direction of travel), `lanes_fwd`, `lanes_back`, `heading` (deg
  clockwise from north, a to b), `length`, `road_class` (into `classes`), `maxspeed_mph`, `name` (into `names`, -1 for
  none); and where links cross more than 4 m apart in height, as at overpasses, `overpass_links`, `overpass_xy` and
  `overpass_z` (each link's height there).
- Traffic lights are `highway=traffic_signals` on the stop line node. Flags with no OSM equivalent keep a `gta:` prefix:
  junction, no left / right turn, slip lane, keep left / right, left turn only lane.
- openpilot's driving model can't turn back on itself, so every move that turns back more than 135 degrees is a
  `no_u_turn` restriction: at a node, and through up to four short links (40 m) as through a median gap or a turning
  loop. GTA's nodes allow them everywhere. The short two-way links joining a divided road's carriageways away from
  junctions are left out. A trip from a dead end then has no route.
- Ped nodes and boat nodes are left out. The links GTA marks "don't use for navigation" stay in (`gta:no_nav`): without
  them most of the map is cut off from the rest.
- Game coordinates (metres) map to degrees about (0, 0), so the map sits on the equator: `gta5_map.py`.
