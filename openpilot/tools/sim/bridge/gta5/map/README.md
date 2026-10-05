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
   dotnet run -c Release --project ynddump -p:CodeWalker=C:\src\CodeWalker -- "<game folder>" paths.jsonl minimap.jsonl
   ```
   The optional `minimap.jsonl` is the road art of the game's radar and pause map (its road layer's triangles, about
   2 MB), for `--minimap`.
2. Convert it, in a Python environment with `osmium` and `pyvalhalla` (`uv venv ~/gta5map/.venv && uv pip install osmium pyvalhalla numpy`):
   ```bash
   python ynd_to_osm.py paths.jsonl ~/gta5map/gta5.osm.pbf --sidecar ~/gta5map/nodes.npz --minimap minimap.jsonl  # the map
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
With ynddump's `paths.jsonl` in `GTA5_MAP` too, the bridge reads GTA's own roads (`paths.py`): a route starts from the
road at the car's height and heading (from that road's next node, so not on a road passing over or under it), the car
counts as off the route on another level or heading the other way, and nav gets the lanes along the route (as
CodeWalker lays them out: 5.5 m wide, 4 m on narrow roads, out from the link by its offset, or centred on a one-way
link) and the forks in it.
`map_view.py roads.json --state <file>` shows the map alone, with a state from a file.

## The map
- Ways carry the lanes each way (`lanes:forward` / `lanes:backward`, `oneway`) and street names. Road classes are
  guessed (GTA has none): motorway for its highway nodes, primary with two lanes or more one way, service for car parks
  and alleys (nodes switched off for traffic, or without GPS), track off-road. GTA splits a road's lanes into separate
  links before a junction; its turn lanes and bays (one-way links at its slip lane and left turn only nodes) take the
  class (up to primary) and name of the road they split off, so the router doesn't avoid them as minor roads.
- GTA also switches off many real roads for traffic, or marks them off-road: the port's streets, quarry and oil-field
  roads, country roads. With `--minimap`, a minor link GTA's GPS may use is `unclassified` (keeping its limit) where the
  game's map draws it as an ordinary road: most of its length more than 15 m from a major road is within about 2 m of a
  drawn road, or, close to major roads all along, most of it is drawn and it joins such a link, or it's a link up to
  40 m long between two. Off-road ones also get `surface=unpaved`. Links without GPS (runways, the golf course, the
  prison, Fort Zancudo) stay service, and the ones the map draws as dirt tracks stay service or track.
- Ramps are `motorway_link` (`trunk_link` off a highway), so the router gives exits ("Take the exit on the right toward
  Popular St"). GTA doesn't mark them: a ramp is a run of one-way links that branches off a freeway and, within 1.5 km,
  reaches an ordinary road or joins another freeway. Runs that come back into the road they left (within 20 m of it and
  at its height) are its lanes: GTA draws a freeway's lanes as links side by side, with lane changes between them. Ramps
  carry no name (named for their freeway they'd read as staying on it) and, where it's known, `destination`: the street
  an off-ramp reaches or the freeway a connector joins. Where two freeways split, both stay freeways (keep left/right).
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
- Traffic lights are `highway=traffic_signals` on the stop line node, stop junctions `highway=stop`. GTA's don't say
  which way they face: a stop line is for the junction ahead of it within 30 m (GTA's are 12-24 m before it), or 50 m
  if that's nearer than the one behind; where a road has junctions about as near both ways, the one with more stop
  lines. Junctions are GTA's junction nodes where links cross (it also flags the nodes where a road's lanes split). A
  stop line facing its junction gets `traffic_signals:direction` / `direction=forward`, with the two-way ways at it
  drawn towards the junction; one with no junction ahead, as on a one-way link leaving one, is left out. Valhalla 3.9
  reads the direction only at a node inside a way, so with a way per link it still counts them both ways; nav's own
  stop list follows it (`paths.py`, `GTA5_STOP_DIRECTION=1`).
- Flags with no OSM equivalent keep a `gta:` prefix: junction, no left / right turn, slip lane, keep left / right, left
  turn only lane on nodes; switched off, no GPS and off-road on the ways at such nodes.
- GTA's no left / no right turn flags, and its one-lane left turn only lanes, become `no_left_turn` / `no_right_turn` /
  `no_straight_on` restrictions at the junction ahead of the node (as found for stop lines), to each way out turning
  more than 45 degrees that way, measured across the junction's node and up to 20 m on (a divided road's far side), and
  at the node itself. Where the flags leave an approach no way out they're ignored. Restrictions through ways go only
  as far as needed: Valhalla misses better routes past them.
- openpilot's driving model can't turn back on itself, so every move that turns back more than 135 degrees is a
  `no_u_turn` restriction: at a node, and through up to four short links (40 m) as through a median gap or a turning
  loop. GTA's nodes allow them everywhere. The short two-way links joining a divided road's carriageways away from
  junctions are left out. A trip from a dead end then has no route.
- Ped nodes and boat nodes are left out. The links GTA marks "don't use for navigation" stay in (`gta:no_nav`): without
  them most of the map is cut off from the rest.
- Game coordinates (metres) map to degrees about (0, 0), so the map sits on the equator: `gta5_map.py`.
