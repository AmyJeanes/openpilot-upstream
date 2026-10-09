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
   `osm_lanes.py` reads them back to the layout of GTA's links as painted (`ynd_to_osm.layout`). `test_osm_lanes.py`,
   `test_validate_lanes.py` and `test_paint_survey.py` run on the hand-written `fixtures/` and made-up samples (`python
   test_osm_lanes.py`); `test_route_lanes.py` and `test_lane_match.py` on small maps made in the test, in the bridge's
   own environment.

## Using it
```bash
valhalla_service ~/gta5map/valhalla.json 2   # the router, on port 8002
# terminal 2, the bridge: the map view on http://localhost:8793/, and nav on our routes
GTA5_MAP=~/gta5map GTA5_ROUTER=http://localhost:8002 ./openpilot/tools/sim/run_bridge.py --simulator gta5
```
Without `GTA5_ROUTER`, nav follows the game's GPS as before. A waypoint set on the game's map, or a tap on the map view
("Navigate here"), sets the destination; the later one wins, and the map view's isn't passed to the game's GPS.
The route sets off the way the car faces, and when the car leaves it nav routes again from where the car is.
With the map's `gta5.osm.pbf` in `GTA5_MAP`, the route's end is chosen from the map rather than the router's nearest
road, which is often a drive, car park aisle or alley beside the place meant (`dest_snap.py`; `GTA5_DEST_SNAP=0` turns
it off). A destination on a property with a drive (a driveway, car park, forecourt or private road) ends on the street
just short of where its drive joins, coming the way that turns in from the kerb side; any other parks at the kerb of
the road it faces, with the destination on the kerb side. Either gives way to the other direction where it's over
30 s longer. A destination on its nearest road with no drive is routed as before.
With ynddump's `paths.jsonl` in `GTA5_MAP` too, the bridge reads GTA's own roads (`paths.py`): a route starts from the
road at the car's height and heading (from that road's next node, so not on a road passing over or under it), the car
counts as off the route on another level or heading the other way, and nav gets the forks along the route and its
junctions. Nav's stop lines along a route are the map's (`stop_lines.py`), as the map view and the overlay draw them:
those the route crosses towards their junction, its own way only, each a stop sign's, lights' or give way's. A route
crosses a line where it drives a link the line lies on or reaches across (GTA lays some approaches as a link per lane,
the line on one of them; a lane with its own line keeps to it). They're worked out with the junction areas (about half
a minute, in the background the first time, kept in `~/.cache/gta5_lanes` by the map file's contents; `python -m
openpilot.tools.sim.bridge.gta5.map.lane_match gta5.osm.pbf` builds both), and the bridge routes on the map's lanes once
they're read. On a map without lane tags they're GTA's stop line nodes, as before (`GTA5_STOP_DIRECTION=1`: only
towards the junction each faces, by `paths.py`'s guess).
The lanes along a route come from the map's lane tags, where `GTA5_MAP/gta5.osm.pbf` has them (`osm_lanes.py`,
`RouteLanes`, read with `osm_pbf.py`, which needs no pyosmium): the route's ways (each shape point is a map node; a real
map's would come from Valhalla's `trace_attributes`, `ways_from_trace`), each one's cross-section in the route's
direction, a lane widening from nothing over 30 m where a way's lane count rises (falls) at a node no other road
joins, the turn arrows into each junction, from which nav takes the lanes for each turn, the junctions it goes straight
on through onto fewer lanes, with the lanes that carry on (`continuing`: kerb to kerb where the road is as wide on both
sides, else by its line), and the line through the lanes
nav plans, on fillets from the lane in to the lane out through turns at junctions (in what room there is near the route's
start or end, as a route from the car with a turn just ahead; kept until the car is past the fillet), moving across evenly over 10 m either
side of a node where the ways' lanes jog sideways (or between route points within 30 m that leave it straightest), on a
smooth curve where the route kinks within the move (`_curve_jogs`), and keeping to its lane where a turn bay opens on its
left. A map without lane tags (only lane
counts, as before) gives them from GTA's own links instead (`paths.Link`: as CodeWalker lays them out, 5.5 m wide, 4 m
on narrow links, out from the link by its offset, or centred on a one-way link), so the bridge runs on either map.
`lane_match.py` reads the car's lane from the map's lane tags alone, route or not: the way at its height running its
way whose lanes it is in (a turn bay laid as its own way in a road's median before the road), and whether that's an
oncoming lane, a one-way driven the wrong way or the other direction's turn bay; nav moves back over on it and e2e
counts the time. Inside junctions' areas (`junctions.py`) the reading means nothing; they take about half a minute to
work out, so they're kept in `~/.cache/gta5_lanes` (`GTA5_LANE_CACHE`) by the map file's contents, built by the bridge in
the background (`python -m openpilot.tools.sim.bridge.gta5.map.lane_match gta5.osm.pbf`) and by e2e when it starts. The
map view draws the roads at their width and, from `lanes.json` (`osm_to_roads.py --lanes`) zoomed in, their lines:
edges, white dashed lines between lanes one way (solid where `change:lanes` forbids crossing), the yellow line between
the directions; and the plugin's debug overlay the same lines, from the map's tags. Kerbs are drawn only where the road
surface ends: not between one-way ways side by side running the same way, as GTA's freeway links and the lane changes
cutting across between them, where the line between two such ways is a lane line, solid where either's `change:lanes`
says so (`side_by_side.py`). GTA's lane changes across the painted gore where two such carriageways part or meet have
no kerbs and cut none (Dutch London St): the gore's edges are the carriageways' own kerbs. A turn bay GTA starts from
a two-way road's middle, where the road carries on narrower by the bay's lane, has the wider road's kerb carried on
straight along its outside to the bay's end, where its outer edge meets it (Vinewood Blvd before Meteor St). Where every road at a node
is one-way and all run about one way (lanes merging, parting or changing across a carriageway) there is no junction
area; GTA's lane changes (`junctions.lane_changes`), cutting across at up to 60 degrees, don't count against that
where two carriageways are left to say it (the hatched slip island by Vinewood Blvd's stop line). Nor round painted
islands (`traffic_calming=painted_island`), road surface. Zoomed in, junctions are drawn as
real maps draw them (`junctions.py`, after osm2streets): each road trimmed back flat where its kerbs meet its
neighbours', the junction's area between, kerbs carried round its corners, no lane lines inside it (but the lines of a
road carried straight on past side roads, as the main road's centre line runs on across them) nor where the roads
start at a junction node (a road meeting it at a slant ends its lines square across itself at the node, reaching past
the mouth of a road beside it trimmed back little; `osm_to_roads.node_ends`), stop lines across
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

## Checking the map against the game from above
`imgcheck/` finds where the map disagrees with the game's own paint and kerbs, from top-down shots of the game, with no
driving: `python -m openpilot.tools.sim.bridge.gta5.map.imgcheck.run plan|capture|analyse|report RUN` (its docstring
has the options).
- `plan` lays tiles along every road (city, a sample of districts, or the whole map), the image's long side along the
  road: at 45 m up with a 50 degree field each 2560x1440 tile covers about 75 m by 42 m at 2.9 cm a pixel. They're
  ordered to keep hops short, as the game streams the world around the camera.
- `capture` shoots them with the plugin's `topcam` (fixed x/y, the map's height as `z`) and `grab` (the player's frame
  from the present hook, written as a BMP: no desktop capture, so nothing on the screen gets in the way). It needs the
  bridge connected and the player in a car. It sets noon, EXTRASUNNY, a frozen clock and no traffic first, checks them
  in the state before every shot, waits longer after long hops, shoots blurry (unstreamed) frames again, resumes where
  it stopped, rides out a bridge restart, and puts the topcam, overlay, time, weather and traffic back at the end.
- `analyse` draws the overlay's own road marks (its cache, `gta5_overlay.road_marks`) into each tile by the topcam's
  projection (`imgcheck/frame.py`) and compares them with the paint found in the image (top-hat brightness, colour by
  warmth over the road's, `imgcheck/paint.py`): paint with no map line within 0.4 m, map lines with no paint, the wrong
  colour, kerbs over even road surface (painted medians, kerbs across lanes), stop lines off the painted bar, and
  line-like paint inside junction areas (`imgcheck/compare.py`). Tiles whose ground under the camera is off the map's
  road height (something over the road) are left out; service roads count a quarter. Where the map's roads are
  stacked (a road surface more than 4 m over another, by roads.json's heights), a tile judges its road only where it
  is the top layer: under a deck it's hidden, not missing. `plan --under` lays a second pass along the roads under
  decks, the camera below the deck (topcam `abs=1`, 5 m above the road, a 90 degree field, a tile about every 10 m).
- `trial.py` shoots chosen tiles under several lighting settings (hour, weather, the plugin's `shadows` command for
  the cascade shadow natives) and compares how much of the road is in shadow and how many of the map's lines have
  their paint seen.
- `report` gathers the issues into spots across overlapping tiles and writes `RUN/report/index.html` (ranked, with
  crops: the game, our map over it, the paint found), `spots.json` and city heat maps.
`run.py import-survey RUN` takes the map audit's survey shots (`/mnt/e/gta5_audit/map_audit/survey_trips_*`, with their
topcam poses) as tiles, so the check runs on them without the game.

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
    give the lanes, and what's left is centred between them. Its edges are the kinds the game files paint there
    (`paint_survey.median_edges`): `divider=solid_line` / `double_solid_line` / `dashed_line` for both, else
    `divider:forward` / `divider:backward` for the edge beside that direction's lanes (OSM's direction suffix: no
    tag says a median's two edges apart). The game paints single and double edges about equally often; unsurveyed
    medians have no `divider`, and osm_lanes draws them one solid line each.
  - A two-lane two-way street's centre is `divider=double_solid_line`, as GTA paints most of them (OSM's default
    reading is a dashed line); not on service roads and tracks.
  - GTA lays a left-turn bay as a one-way link of its own (through its slip lane or left turn only nodes) from where
    it splits off its road to the junction ahead, over a two-way road's median or beside a one-way road's left lane.
    Real maps make it a lane of the road (osm_lanes.md P3), and so does ours (`detached_bays`): the bay's links are
    dropped, the road's links from the split to the junction get one more lane that way (filling the median, as GTA
    paints it: the median tapers away where the bay opens) with `turn:lanes` such as `left|through|through;right` from
    there, and the left turn GTA forbade from the road's own lanes is allowed from it. Every other link stays as GTA has
    it.
  - Where the game files paint a bay opening (`--survey-lines`: a yellow polyline swinging from the median's right
    edge across to its left edge, `paint_survey.swing_taper`, read off the whole polyline since sections miss steep
    ones; `--survey` sections alone: `opening_taper`), the road's links are split where the taper starts and ends
    (nodes added along the link, ids from `TAPER_NODE_AREA`; the router and lane parity skip them as GTA's), the lane
    count changes there in whole steps instead of at GTA's split, and the links in between widen the new lane from
    nothing with `width:lanes:forward:start` / `:end` (`:backward`; the widths at the way's first and last node). On
    links whose lane counts come from the paint, the lane is there where the paint counts one more than GTA. osm_lanes
    draws lines and lane centres between those sections, so the median's edge kinks across, and nav's lane slots open
    the lane where it's wide. The opening is looked for back along the carriageways GTA lays as one-way links side by
    side before they join into the road (read along the line midway between them; Meteor St): one opening there is
    already partly or fully open where the road begins. Where the paint shows no opening (GTA's split alone, AI data),
    the lane is full width where GTA's lane begins and widens over the 10 m before it (`TAPER_DEFAULT`) where the road
    before has a median to open in; on the 58 surveyed bays the painted opening is a median 9 m long and ends 5 m before
    GTA's split, so a lane widening from the split on (osm_lanes' default) opened about 35 m late. The lane is open
    from where the road begins at its carriageways joining (a grass median ending; Eclipse Blvd), from a junction less
    than 15 m before GTA's split, and from where the paint's lane counts already have it on the links before (full
    width at that node, not widening over the first link: Eclipse Blvd's left lane runs on through its junctions from
    the median's end). Elsewhere
    osm_lanes tapers a lane that appears or ends over 30 m (`TAPER_M`).
  - Where a road carries on from one way to the next (two ways meeting end to end, no junction) with the same lanes
    sitting elsewhere (other widths or line, as where the survey measured one link and not the next), osm_lanes moves
    them across on a smoothstep over up to 10 m either side of the node (`BLEND_M`, half of a shorter way; a tapered
    way keeps its taper and the next way takes it all), and mitres each way's lines into the next's: kerbs, lines,
    junctions' kerbs, nav's lane line and its lane slots run on without a step. The map view draws those ways' asphalt
    between their kerbs (roads.json's `strips`), and lanes.json's pieces that meet end to end are one line, so dashes
    run on. A lane appearing or ending at a node with no taper still steps there.
  - A divided road's carriageway joins (or parts from) the road it becomes at a node with the other carriageway and the
    two-way road going on (side roads aside); GTA lays its last link angled across to the road's line, so its lanes,
    laid along the link, stepped across to the road's at the node, as kerbs and as nav's lane line, where the lanes
    painted run on (Eclipse Blvd, Meteor St). The carriageway's lanes move across on a smoothstep over up to 30 m of
    its end way (`CARRIAGEWAY_BLEND`) to where the road's lanes are (ours, right-aligned: the road's extra lanes are the
    median's), the road's staying put. On the 183 divided-road ends, nav's lane line near the node is 0.39 m from the
    painted lane's middle on average (0.62 before; 0.27 m 25-40 m away from it).
  - Both ways' bays often share one median back to back (a diamond, or one line handing the median from one way's bay
    to the other's). A line crossing the median looks the same from either way: it is the bay of the way after which
    nothing runs on at the median's right edge. Each way's bay opens from its own end, both widths on the same ways
    where they overlap (each its share of the median, halved where both are open at once). On a widening way with
    more lanes one way, the line is placed midway between the lanes either side of the median and its bays
    (`placement:forward` / `:backward=left_of:N`), which keeps it on GTA's link as the bays widen.
  - Where a two-way road's median (2.5 m or more) runs in to a junction it may turn left at and GTA has no bay link,
    the game paints the median as a left-turn lane on the approach (seen on 4 of 4 such approaches): its links within
    the median over the last 30 m (15 m at least) get that lane, `turn:lanes` `left|...`, filling the median. Not
    where the game files paint the median's edge on our side on into the junction with no arrow of ours in it
    (`paint_survey.median_runs_in`): there the median stays a median, often with the other way's bay opening in it.
    Wherever else the game files paint a left arrow wholly inside a median (between the two directions' lanes), the
    median is a lane GTA has no link for (`painted_median_lanes`): from that link on towards the next junction, while
    the road runs on alone, one more lane `left`, opening as above (Vinewood Blvd before Meteor St, where a hatched
    median ends and the left-turn lane opens in its room; 44 roads).
  - Where GTA runs both carriageways of a divided road through one node in its median (a side road meeting it at a gap
    in the median; 84 junctions, each carriageway bent about 7 m across the median into it), each carriageway gets a
    node of its own on its line between its neighbours, joined by a two-way link across the median of the road's class
    (`split_shared`; node ids from `SPLIT_NODE_AREA`, tagged with GTA's `gta:node`). The other links go to the
    carriageway on their side; a turn lane in the median keeps GTA's node, which the link across runs through, so GTA's
    turn flags read as before. junctions.py makes the carriageways' nodes one junction, square across the median's
    noses. Left as GTA has them: two divided roads crossing at the node, lanes going on through between the
    carriageways, U-turn-only gaps, and carriageways more than 20 m apart (`JUNCTION_SPAN`).
  - These widths are class rules, right on average. `--survey <file.jsonl> ...` corrects them where the game's paint
    was surveyed (format and rules in `paint_survey.py`): read from the game files' road meshes and marking decals
    (`"src": "gamefiles"`, exact to a few cm, the primary source); the map audit's top-down camera survey only
    cross-checks them, and corrects links itself only when no game-file survey is given. On a two-way link whose samples agree with each other and with GTA's lane counts, the
    lanes take the measured widths out to the kerbs (the game files' asphalt edges where they're symmetric, else the
    layout's), so a centre line off GTA's link line is said by the lanes' widths (the way's line stays on GTA's nodes,
    the middle of the road); `source:width=survey` marks them. From the game files also: a two-way link's lane counts
    where its paint has other counts than GTA's (Eclipse Blvd's 3 + 2; arrows are then laid out for the painted
    lanes; there the asphalt's edges may be up to 0.8 m off the line's middle, `COUNT_KERB_TOL`), its centre line's
    kind (`divider`), `change:lanes` from the lines' kinds, paver strips 1.8-5.5 m wide
    between the asphalt's edge and the kerb's face as parking lanes (`parking:<side>=lane`, `:width`), one-way links'
    lanes between their painted edges (GTA's lanes evenly between them where the files miss every lane line). GTA's lane
    counts stay on one-way links. A freeway GTA draws as parallel links is painted as one carriageway: each link takes
    the painted lanes about its line (`paint_survey.correct_carriageway`), placed by `placement` on the nearest lane
    edge or middle (the lines move up to 1 m to fit), and a one-way link's outer lanes take `change:lanes` from the
    white line painted at their outer edge, the line between it and the next link (`paint_survey.outer_lines`). With
    `--survey-lines polylines.jsonl` (the game files' lines whole) a section's line kind is the whole line's (worn and
    tiled solid lines read as dashed in sections), and raised markers at the asphalt's edge or the kerb are dropped (the
    gutter's edge, checked in the game). A two-way link the files have no sections of (GTA's short links in and next to
    junctions, whose sections the survey leaves out) takes the painted lanes of the link its road runs on to with the
    same lane counts (`neighbours_paint`), so its lines run on at the paint's place and kind rather than GTA's class
    layout (a median where the game paints a double line; Mt Haan Rd). Build: `ynd_to_osm.py ... --survey
    rp_all/survey_gf.jsonl --survey-lines rp_all/polylines.jsonl`.
  - Where a link's lanes stay the class layout (the survey didn't correct them, turn bays folded in included), its
    lines still come from the game files' (`paint_survey.lane_lines`): each line between two lanes one way is the
    white line painted nearest it, less than halfway across the lanes beside it, in half the sections, and a solid one
    (or solid on one side) is `change:lanes` (the line beside a left-turn bay, approaches' solid lines). Where the files
    show no line between two lanes the line stays dashed: they miss thin dashed lane lines the game paints (Vinewood
    Blvd's, seen from above in the game), so no direction is left unmarked from them. Residential roads, service roads
    and tracks of two lanes or more the files show unpainted in 90% of 3 sections or more (the asphalt's edges read,
    so not a gap in the files; lines within 1 m of an edge are its edge lines) are `lane_markings=no` (the Vinewood
    Hills' streets, car parks), and so are the links along such a road between unpainted ones too short to show it
    themselves (fewer sections, all bare; `unmarked_roads`), so its lines don't stop and start (Fenwell Pl). Major and
    unclassified roads keep their lines there, being more likely gaps in the files (the Great Ocean Hwy, some freeways,
    Blaine roads), but for unclassified roads GTA's traffic doesn't use (switched off, back roads the minimap draws) or
    marks off-road, unpainted where their sections are bare even with no asphalt edges read (Baytree Canyon Rd); so are
    tracks and other off-road links (dirt, no asphalt to read edges of; the track off Senora Rd).
  - Parking lanes on the carriageway (`parking:left|right|both=lane`, `parking:<side>:width`, OSM's street parking
    scheme) are part of `width` but not lanes: `osm_lanes.py` puts the kerb beyond them and the way's line in the
    middle of the lanes between them, and the map view draws them as faint strips. GTA's path data has no field for
    them and no measured road showed one, so our map has none; real maps do.
  - Links whose two directions share one lane (most car parks, alleys and tracks) are single-track roads: `lanes=1`,
    no direction counts, `lane_markings=no`, as real single-track lanes are mapped.
  - A two-way link running on from a one-way link with nothing else at their node (45, mostly service roads and island
    side roads such as Route 68's Fort Zancudo turn) has a direction that goes nowhere there, a lane GTA's AI never
    drives: it is one-way as the one-way link (`dead_end_lanes`), rather than a lane ending at a point.
  - `turn:lanes` (`:forward` / `:backward`) on the lanes into a junction where roads cross, from the ways out of it
    less those the restrictions forbid. Each lane takes the arrow the game files paint in it up to 90 m before the
    junction (`--survey-features rp_all/features.jsonl`, placed in the lanes by position), less any move the junction
    has no way out for (through also where the road goes on skewed up to 55°). Lanes without paint follow the
    junction's ways out and their lane counts (GTA's paths connect a link, not a lane, to each way out): taken from
    the left, through gets as many lanes as it has lanes out, a turn one, spare lanes turn while a turn has lanes out
    to spare, and with too few the right turn shares the right through lane, then the left turn the left one. 3 lanes
    into 2 straight on are `left|through|through;right`, into 1 `left|through|right`. This agrees with the paint on
    95 of 149 approaches painted in every lane (a fixed left-only / through-and-right pattern: 69). They never turn
    across a painted lane. GTA lays many approaches as a link per lane, each with its own turn flags: each is its own
    approach, so a lane's arrows are the moves its own link has. On every way from 30 m before the junction, or from
    the furthest painted arrow (Valhalla reads them from the way into it); not on one-lane approaches other than GTA's
    left turn only lanes and those whose painted arrow covers every move they have, and not where the road bends into
    the junction, which leaves which way is through moot.
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
  reads the direction only at a node inside a way, so with a way per link it still counts them both ways; nav reads
  the map's lines, each its own way only (`stop_lines.py`).
- GTA's stop line nodes are 12-24 m before their junction's node, often metres off the painted line. With
  `--survey-lines` (and `--survey-features rp_all/features.jsonl` for the painted crossings) the map's stop lines go where the game files
  paint them (`stop_paint.py`): on each approach to a junction (junctions.py's), the thick white line across its
  lanes towards the junction (the far edge of a crossing painted there, which GTA paints as the stop line), as a node
  splitting the way there (ids as for tapers) that takes the signal or sign and its direction from GTA's node. An
  approach with a painted stop line and none in the map gets one (`traffic_signals` at a junction with signals, else
  `stop`) where the line doesn't reach on across the other direction's lanes (a crossing's edge; past the kerb is the
  class layout's lanes narrower than the road) and no crossing is painted just ahead of it. A GTA stop line with no
  paint on its own approach but a new one from the paint on the road the other way within 15 m goes there instead
  (GTA's node faced the wrong junction). GTA's node stays the stop line where nothing is painted. The decals' measured
  widths of stop lines run 0.16-0.3 m (`MIN_WIDTH` 0.15). The polylines run some stop lines on round the corner into
  the edge line they meet (an L): a bent polyline gives its straight end pieces. roadpaint reads an atlas decal's bands
  only at their own place in the texture, so a decal laid with its UVs a whole tile over misses them (many STOP
  decals' stop lines, as on Mt Haan Dr's T): those are read from `decals.jsonl` and `textures.tsv` beside
  features.jsonl. Of the 9724 approaches, those with a stop line painted across their own lanes and none in the map
  went from 126 to 28 (the rest are crossings' edges); 48 more have a painted STOP word and no line (the port's grid).
  A stop line painted level with the road it meets, nearer the junction's node than the class layout's kerbs meet,
  is drawn where it's painted, along that road's edge (between where the kerbs meet it: square across GTA's link it
  lay askew at a skewed T, Vinewood Park Dr), inside the junction's area, which is shaped as without it: cut back to
  it, the road's kerbs ran out across the other road's lanes (junctions.py).
- Painted islands and flush edges (`painted_islands.py`), from roadpaint's height-layered tiles beside its polylines
  (road, kerb, pavement or gutter at each height; a raised island reads pavement or kerb). Painted islands, flat
  painted areas with road all round them: areas outlined in white or yellow (the painted triangle where Mt Haan Dr's
  side road parts; a hatched area between the edge lines its strokes run between, its tips, Vinewood Blvd's hatched
  medians), and road surface between carriageways that no way's lanes cover (the chevron-hatched gores
  between GTA's links on Meteor St; GTA's lane changes across them don't count), each a closed way `area=yes` +
  `traffic_calming=painted_island` with the `colour` of the paint round it (435, 216 outlined). Flush edges: where a
  way's edge faces another carriageway within 8 m with road on beyond it (GTA's links side by side round paint, a
  slip parting from Route 68, the class layout's edges inside the asphalt), a strip of `area:highway=<its class>`
  (road surface) along it (264 km of edge). GTA lays its links round paint as round raised islands; no kerb is drawn
  within 1 m of a painted island or inside road surface (`junctions.off_islands`, the map view and the overlay), the
  map view and the overlay draw an island's outline in its colour, and a junction's corner ends at a painted
  island's tip, square, rather than trimming back over a painted gore (Vinewood Blvd's hatched median beside the
  left-turn lane, hidden under a junction area trimmed 40 m back).
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
  junctions are left out, and the two-way links GTA lays as lane changes between one-way links all running the same
  way (crossing in an X, as on the Elysian Fields Fwy) are one-way, the way the traffic runs (`lane_changes`). A trip
  from a dead end then has no route. A move with no other way on at any node along it
  is the road, not a U-turn: a hairpin bend, or the only way out of an acute junction.
- No restriction may cut road off that GTA's links join to the rest of the map (`traps.py`: a car on a way that way
  could not be routed out, or a way could not be routed to either way). Where restrictions do, the generator changes
  the least turning forbidden move out of or into each such area, U-turn bans before GTA's flags, until none is left:
  out of a trap the move is forbidden from the ways on to the trap's way instead (a car there may leave, but no route
  goes in to turn back); in to cut-off road it is allowed. It stops if the restrictions it writes cut any road off.
- Ped nodes and boat nodes are left out. The links GTA marks "don't use for navigation" stay in (`gta:no_nav`): without
  them most of the map is cut off from the rest.
- Game coordinates (metres) map to degrees about (0, 0), so the map sits on the equator: `gta5_map.py`.
