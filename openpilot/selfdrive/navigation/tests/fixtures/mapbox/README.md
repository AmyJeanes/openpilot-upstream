# Mapbox Directions response fixtures

What navd gets from Mapbox, as the ground truth for the GTA emulator (`tools/sim/bridge/gta5/mapbox_server.py`) and the
step 1 lane guidance (`selfdrive/navigation/lane_guidance.py`). The routes are invented (roads near 0°N 0°E, made-up
names) and written by `make_fixtures.py`, in the schema and with the lane features of real UK responses: the findings
below come from 20 real responses (16 Oct 2026), which are not kept. Each file is `{name, features, made, request:
{path, params}, response}`.

Request: `/directions/v5/mapbox/driving-traffic/<lon,lat;lon,lat>` with `steps=true`, `banner_instructions=true`,
`annotations=maxspeed,congestion,distance,duration`, `overview=full`, `geometries=polyline6`, `alternatives=false`,
`language=en` (navd's own request asks `geometries=geojson` and, with the lane guidance, `maxspeed,congestion`).

| fixture | what it shows |
|---|---|
| `turn_bay_right` | a right-turn bay: 2 lanes (`S* R`) 250 m before Marsh Lane, 3 (`S R R*`) in the last 33 m, the turn's own intersection listing 2 (`S R*`); lanes valid for each way's own junction |
| `motorway_exit_lane_drop` | an exit with a lane drop: 3 lanes (`S* S S`) 43-270 m before the diverge, 4 at the gore (`l* S S S`); a lane gain 2 -> 4; banner lanes from 2 miles out |
| `motorway_exit_shared_lane` | an exit with a shared lane (`l* l/S* S S`); a rotary onto the slip road |
| `two_left_turns_close` | two left turns 80 m apart, lanes for the first; a stop sign |
| `close_pair_then_fork` | two maneuvers 10 m apart, a fork 259 m on (`l* S S`); lane gains and drops along a trunk road |
| `roundabout_entry_gains_lane` | a roundabout entry gaining a lane at the give way (`S* S*` -> `S* S* S`); a B-road with no lanes |
| `rural_no_lanes` | no lane data on 6 km of rural road; 40-60 mph limits, some unknown; a level crossing |
| `right_lanes_cycle` | 2 right lanes (`L R R*`) and a cycle lane (`access`); a left lane added 21 m before the next turn (`S* S*` -> `L* S S`) |
| `lanes_vary_bus` | 2 to 5 lanes along one step, added and dropped between junctions; two right lanes; bus lanes |
| `lanes_added_midblock` | a right lane at some junctions of one road only; two maneuvers 11 m apart; a roundabout |
| `lanes_always_there` | 3-4 lanes for 260 m into a slight right; a rotary |
| `lanes_last_5m` | the junction's 4 lanes only in the last 5 m (`L R*` -> `L L R* R`) |

## What `lanes` holds (the 20 real responses: 2257 intersections, 169 maneuvers)

- **Where**: only where OSM has `turn:lanes` on the way. Then on **every intersection of that way** (crossings, signal
  nodes, side roads; 403 of 2257 intersections, 18%), repeated identically, up to and including the junction where the
  way ends. 38 of 169 maneuvers (22%) have lanes at their own intersection; residential turns, rural roads and most
  roundabout exits have none.
- **Order**: left to right, whatever the driving side. Each lane: `indications` (`left`, `slight left`, `straight`,
  `slight right`, `right`, `sharp ...`, `uturn`, `none`), `valid`, `active`, `valid_indication` (only on valid lanes),
  and on some `access: {designated: [bus | bicycle]}` (bus and cycle lanes are listed as lanes).
- **Valid for which move**: the route's move at the junction where *that way's* lanes end, not the next maneuver's.
  250 m before the Dale Rd turn, Winchester Rd's earlier way says `S* R` (straight on at its own junction); only the
  last way before the turn says `S R R*`. `active` = `valid` in every case here (no anticipation seen for the close
  maneuver pairs, 6 with lanes).
- **The maneuver's intersection** (`steps[k].intersections[0]`, at the maneuver's `location`) carries the junction's
  lanes; 5 of 24 maneuvers with approach lanes show a different list there than 13-49 m before it: the M27 J3 exit
  lane appears only at the gore node (`S* S S` -> `l* S S S`), Winchester's roundabout entry gains a third lane
  (`S* S*` -> `S* S* S`), the A40 fork gains its left lane (`S* S` -> `L* S S`), but Dale Rd's node lists 2 lanes
  (`S R*`) where the 13-33 m before it list 3 (`S R R*`). **Counted from the maneuver's side, the target lanes agree in
  all five**: the exit/turn lane is the outermost on its side either way.
- **Bays and added lanes**: never flagged. A bay is just one more lane at the intersections nearest the junction (12
  maneuvers have more lanes there than 100-400 m back); where a lane begins is not given (no openings, widths or
  geometry), and an intersection-free stretch says nothing at all. A lane added mid-block shows the same way (Vauxhall:
  4, 3, 4, 2, 5 lanes along one step).
- **Banner lanes**: `bannerInstructions[].sub.components` of type `lane` (`directions`, `active`, `active_direction`)
  for the step's next maneuver, equal to that maneuver's intersection lanes (26 of 26). They start at the last banner's
  `distanceAlongGeometry`: 402 m (a quarter mile) in town, or the whole step when shorter; 1609 m or 3219 m (1 or 2
  miles) on motorways, else the whole step. The first banner of a long step has no lanes.

## The rest of the schema

- `annotation.maxspeed` per geometry segment: `{"speed": 48, "unit": "km/h"}` (UK mph limits as km/h: 8, 16, 24, 32,
  48, 64, 80, 97, 113 for 5-70 mph), or `{"unknown": true}` (5% of segments). `congestion`: `low`, `moderate`, `unknown` here
  (Mapbox also has `heavy` and `severe`). Steps also give `speedLimitSign` (`vienna`) and `speedLimitUnit` (`mph`).
- Steps: `maneuver` {`type`, `modifier`, `location`, `bearing_before`, `bearing_after`, `instruction`}, `name`, `ref`,
  `distance`, `duration`, `geometry`, `driving_side` (`left`), `rotary_name` / `exits` at roundabouts, `destinations`.
  Types seen: depart, turn, end of road, fork, off ramp, roundabout, rotary, new name, continue, arrive.
- Intersections: `location`, `geometry_index` (into the route's geometry), `bearings`, `entry`, `in`, `out`, `duration`,
  `weight`, `turn_weight`, `mapbox_streets_v8.class`, `is_urban`, `admin_index`, `classes` (motorway, tunnel...),
  `traffic_signal`, `stop_sign`, `yield_sign`, `railway_crossing`, `lanes`; `classes` seen: motorway, restricted, tunnel.
