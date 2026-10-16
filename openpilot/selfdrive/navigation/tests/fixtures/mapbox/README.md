# Real Mapbox Directions responses (UK)

What navd gets from Mapbox, kept as the ground truth for the GTA emulator (`tools/sim/bridge/gta5/mapbox_server.py`)
and the step 1 lane guidance (`selfdrive/navigation/lane_guidance.py`). Fetched 16 Oct 2026 with a personal token
(free tier, 20 requests in all; 12 kept), the token stripped: each file is `{name, place, features, fetched, request:
{path, params}, response}`, the response exactly as returned.

Request: `/directions/v5/mapbox/driving-traffic/<lon,lat;lon,lat>` with `steps=true`, `banner_instructions=true`,
`annotations=maxspeed,congestion,distance,duration`, `overview=full`, `geometries=polyline6`, `alternatives=false`,
`language=en` (navd's own request asks `geometries=geojson` and no annotations; the rest is the same).

| fixture | place | what it shows |
|---|---|---|
| `soton_uni_to_general` | Southampton: University Rd, Burgess Rd (A35), Winchester Rd, Dale Rd | a right-turn bay: Winchester Rd has 2 lanes (`S R`) 250 m before Dale Rd and 3 (`S R R*`) in the last 33 m; lanes valid for each way's own junction |
| `soton_central_to_ocean_village` | Southampton: Commercial Rd, West Quay Rd (A33), Town Quay | a right lane at some junctions of West Quay Rd only (added mid-block); two maneuvers 11 m apart; a roundabout |
| `soton_bargate_to_portswood` | Southampton: Civic Centre Rd, Dorset St (A33), Onslow Rd | 3-4 lanes for 260 m into the slight right onto Onslow Rd (lanes that have always been there); a rotary |
| `soton_shirley_to_west_quay` | Southampton: Shirley High St (A3057), West Quay Rd, Harbour Parade | two left turns 80 m apart, lanes for the first |
| `m27_j5_west_exit_j3` | M27 westbound from J5, off at J3 | the exit with a lane drop: 3 lanes (`S* S S`) 43-270 m before the diverge, 4 at it (`l* S S S`, the exit lane appearing at the gore node); a lane gain (2 -> 4) past J4; banner lanes from 2 miles out |
| `eastleigh_m3_m27_west` | Eastleigh, M27 J5 -> J3, M271, Nursling | the J3 off ramp with a shared lane (`l* l/S* S S`); lane gains on the M27; rotary on-slips |
| `portsmouth_m275_south` | Cosham, A3, M275 south, Commercial Rd | two maneuvers 10 m apart (left onto Western Rd, right onto the A3), the fork onto the M275 259 m on (`l* S S`); lane gains and drops along the M275 |
| `winchester_m3_j9_city` | Winchester: A31, the M3 J9/J10 roundabouts, Bar End Rd | roundabout entries gaining a lane at the give way (`S* S*` -> `S* S* S`); B-road steps with no lanes |
| `newforest_lyndhurst_brockenhurst` | New Forest: Lyndhurst -> Brockenhurst (A337) | no lane data anywhere on a 6 km rural A-road; 40-60 mph limits |
| `london_elephant_castle` | London: Blackfriars Rd (A201), Elephant and Castle, Walworth Rd | 2 right lanes (`L R R*`); a left lane added 21 m before Walworth Rd (`S* S*` -> `L* S S`); cycle lanes (`access`) |
| `london_vauxhall_gyratory` | London: Vauxhall gyratory, Bessborough Gardens (A202) | 2 to 5 lanes along one step, added and dropped between junctions; two right lanes; bus lanes |
| `london_old_street_roundabout` | London: City Rd (A501), Old Street roundabout | the junction's 4 lanes only in the last 5 m (`L R*` -> `L L R* R`) |

## What `lanes` holds (all 20 responses: 2257 intersections, 169 maneuvers)

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
