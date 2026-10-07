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
   python osm_to_roads.py ~/gta5map/gta5.osm.pbf ~/gta5map/roads.json --lanes ~/gta5map/lanes.json  # the map view's roads
   valhalla_build_config --mjolnir-tile-dir ~/gta5map/tiles --mjolnir-tile-extract ~/gta5map/tiles.tar \
     --mjolnir-timezone '' --mjolnir-admin '' > ~/gta5map/valhalla.json
   valhalla_build_tiles -c ~/gta5map/valhalla.json ~/gta5map/gta5.osm.pbf
   ```
   (run from the repository root with `PYTHONPATH=.`, as `python openpilot/tools/sim/bridge/gta5/map/ynd_to_osm.py ...`).
3. Check its lanes: `validate_lanes.py ~/gta5map/gta5.osm.pbf` checks the lane tags (also on a real OSM extract, with
   `--left` where traffic drives on the left), and `lane_parity.py paths.jsonl ~/gta5map/gta5.osm.pbf` that
   `osm_lanes.py` reads them back to the layout of GTA's links as painted (`ynd_to_osm.layout`). `test_osm_lanes.py` and `test_validate_lanes.py` run on the
   hand-written `fixtures/` (`python test_osm_lanes.py`); `test_route_lanes.py` on small maps made in the test, in the
   bridge's own environment.

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
counts as off the route on another level or heading the other way, and nav gets the forks along the route, its stop
lines and junctions.
The lanes along a route come from the map's lane tags, where `GTA5_MAP/gta5.osm.pbf` has them (`osm_lanes.py`,
`RouteLanes`, read with `osm_pbf.py`, which needs no pyosmium): the route's ways (each shape point is a map node; a real
map's would come from Valhalla's `trace_attributes`, `ways_from_trace`), each one's cross-section in the route's
direction, a lane widening from nothing over 30 m where a way's lane count rises (falls) at a node no other road
joins, the turn arrows into each junction, from which nav takes the lanes for each turn, and the line through the lanes
nav plans, on fillets from the lane in to the lane out through turns at junctions. A map without lane tags (only lane
counts, as before) gives them from GTA's own links instead (`paths.Link`: as CodeWalker lays them out, 5.5 m wide, 4 m
on narrow links, out from the link by its offset, or centred on a one-way link), so the bridge runs on either map. The
map view draws the roads at their width and, from `lanes.json` (`osm_to_roads.py --lanes`) zoomed in, their lines:
edges, white dashed lines between lanes one way (solid where `change:lanes` forbids crossing), the yellow line between
the directions; and the plugin's debug overlay the same lines, from the map's tags. Zoomed in, junctions are drawn as
real maps draw them (`junctions.py`, after osm2streets): each road trimmed back flat where its kerbs meet its
neighbours', the junction's area between, kerbs carried round its corners, no lane lines inside it, stop lines across
the lanes into it at its signals and stop signs (and behind its crossings), the lines on its approaches ending there,
and `footway=crossing` crossings striped across the road. Roads are drawn a layer at a time (`layer`, `bridge`,
`tunnel`): bridges over what they cross, with a dark casing, tunnels faded. A median between the directions is a
solid yellow line along each edge. "Turn paths" shows each lane's moves through the junctions (`Junctions.movements`:
`turn:lanes` arrows matched to the ways out by angle, `type=connectivity` relations, turn restrictions, no U-turns;
without arrows the outer lanes also turn): dashed blue left, grey through, orange right. Where the map has heights
(ynd_to_osm's nodes' `ele`), roads.json and lanes.json give each road, line and junction its height, and zoomed in
the view draws what's on another level from the car darker, under the car's: within about 4 m of its height, and 6 cm
more for each metre away from it (a road climbs), so a road crossing over or under it is dimmed but a hillside isn't.
The view keeps what it has drawn in tiles of screen pixels while the map pans, drawing them again when the zoom, what's
shown or a level changes.
`map_view.py roads.json --state <file>` shows the map alone, with a state from a file.

## The map
- Street names are the nodes' street, on links whose two nodes share it. GTA's links between two streets' nodes, as
  through a junction, have none; a router charges for every change of name, so it would favour streets with no names.
  A run of such links up to 100 m long going straight on (within 30 degrees) from the same street at both ends takes
  its name, except the links at a stop line (named on into the junction, Valhalla charges about 40 s more for going
  straight through it) and freeway links (named, Valhalla sends city trips round slower freeway detours: with a lane
  per link, our city streets carry far more junction and turn costs than real ones).
- Ways carry their lanes in standard OSM tags, as a real road's would be mapped, and `osm_lanes.py` reads them (any
  OSM map, not only ours) into each way's cross-section, its painted lines and its lanes' centre lines:
  - `lanes`, `lanes:forward` / `lanes:backward`, `oneway`, and `width` (kerb to kerb), with lanes as GTA paints them,
    measured in the game: 5.5 m, 4.4 m on narrow links, 6.1 m on one-way freeway links of two lanes or more.
    CodeWalker's 4 m narrow lanes (`paths.Link`) are where the game's cars drive, not the paint. The way's line is the
    boundary between the directions, the middle of the road unless the counts differ (`placement:forward` /
    `placement:backward=left_of:1`); a one-way link's lanes are centred on it.
  - GTA's offset between the directions of a two-way link is a painted median, 0.9 m a step whatever the lanes'
    width (5.2-5.5 m at 6 steps on narrow and normal links): `width` includes it, `width:lanes:forward` / `:backward`
    give the lanes, and what's left is centred between them (`divider=double_solid_line`).
  - A two-lane two-way street's centre is `divider=double_solid_line`, as GTA paints most of them (OSM's default
    reading is a dashed line); not on service roads and tracks.
  - GTA lays a left-turn bay as a one-way link of its own (through its slip lane or left turn only nodes) from where
    it splits off its road to the junction ahead, over a two-way road's median or beside a one-way road's left lane.
    Real maps make it a lane of the road (osm_lanes.md P3), and so does ours (`detached_bays`): the bay's links are
    dropped, the road's links from the split to the junction get one more lane that way (filling the median, as GTA
    paints it: the median tapers away where the bay opens) with `turn:lanes` such as `left|through|through;right` from
    there, and the left turn GTA forbade from the road's own lanes is allowed from it. osm_lanes widens the new lane
    from nothing over 30 m where the lane count rises. Every other link stays as GTA has it.
  - Where a two-way road's median (2.5 m or more) runs in to a junction it may turn left at and GTA has no bay link,
    the game paints the median as a left-turn lane on the approach (seen on 4 of 4 such approaches): its links within
    the median over the last 30 m (15 m at least) get that lane, `turn:lanes` `left|...`, filling the median.
  - Parking lanes on the carriageway (`parking:left|right|both=lane`, `parking:<side>:width`, OSM's street parking
    scheme) are part of `width` but not lanes: `osm_lanes.py` puts the kerb beyond them and the way's line in the
    middle of the lanes between them, and the map view draws them as faint strips. GTA's path data has no field for
    them and no measured road showed one, so our map has none; real maps do.
  - Links whose two directions share one lane (most car parks, alleys and tracks) are single-track roads: `lanes=1`,
    no direction counts, `lane_markings=no`, as real single-track lanes are mapped.
  - `turn:lanes` (`:forward` / `:backward`) on the lanes into a junction where roads cross, from the ways out of it
    less those the restrictions forbid: every lane the same way at a forced turn, else the outer lanes also turn (GTA's
    cars turn from the outermost) and the others go through, or half each way at a T. On every way from 30 m before
    the junction (Valhalla reads them from the way into it), not on one-lane approaches other than GTA's left turn
    only lanes, and not where the road bends into the junction, which leaves which way is through moot.
  - One link (offset -2/14 lane) has lanes overlapping that OSM can't describe: its kerbs are kept.
- Road classes are guessed (GTA has none): motorway for its highway nodes, primary with two lanes or more one way, service for car parks
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
- Where a link crosses over another (more than 4 m above it, with no node in common), it and the links beside it at
  its height within 15 m (GTA draws a bridge's lanes as links side by side) are `bridge=yes` with a `layer` one above
  what they cross: freeway interchanges stack up to `layer=5`. Without the ground's height, roads under others stay on
  the ground (no `tunnel`).
- GTA's pedestrian crossings are links of their own between its crossing nodes, joined to no road: they're
  `highway=footway` + `footway=crossing` + `crossing=marked`, and drawn as zebra crossings.
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
