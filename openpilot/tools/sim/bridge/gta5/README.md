GTA V bridge
============

Drives Grand Theft Auto V Enhanced (story mode) with openpilot. A ScriptHookV plugin renders the game from a comma 3X-like
mount on the player's car, resamples the frame into openpilot's road and wide cameras, streams them and the car's state to
the bridge, and applies openpilot's commands to the car.

It's developed with the Steam version on Windows and openpilot in WSL.

## Setup
- openpilot built with a GPU for the driving model, as for the [Slow Roads bridge](../slowroads/README.md#setup).
- [ScriptHookV](http://www.dev-c.com/gtav/scripthookv/) for the installed game build, in the game folder. It disables
  BattlEye, so the game runs story mode only while it's installed.
- The plugin: `plugin/build.ps1 -Install` (Visual Studio with the C++ workload) puts `gta5op.asi` and `gta5op\` in the
  game folder. `gta5op.asi` loads at game start; `gta5op\gta5op_core.dll` reloads whenever it changes, so later builds
  apply to a running game. `gta5op\gta5op.ini` holds the settings, and `gta5op\gta5op.log` the plugin's log.
- In the game's settings: Pause Game On Focus Loss off (openpilot's UI takes focus), and a windowed or borderless display
  mode. The plugin captures the game's window, so keep it uncovered and not minimized.

## Running
```bash
# in both terminals: a simulated Tesla Model 3 (angle steering suits the game's steering; the sim's default is a Honda)
export OPENPILOT_PREFIX=gta5 MODELD_DEV=CUDA FINGERPRINT=TESLA_MODEL_3
GALLIUM_DRIVER=d3d12 BLOCK=soundd ./openpilot/tools/sim/launch_openpilot.sh   # terminal 1
./openpilot/tools/sim/run_bridge.py --simulator gta5            # terminal 2
```
Get into a car in the game. Once the bridge is running, the plugin connects to it and switches to the openpilot camera; it
switches back when the bridge stops. The bridge writes its address to `C:\Users\Public\gta5op-bridge.txt` for the plugin;
set `bridge=` in `gta5op.ini` if that path isn't reachable from WSL.

`gta5_cmd.py` sends debug commands through the running bridge: `snap` saves the next frames as PNGs, `burst` the next N
frames' luma, `reset part=hook|capture|camera` restarts that part of the camera pipeline, `setup` spawns a car
and puts it on the road nearest a point (`heading=` the way nearer that, `lane=` in that lane from the left), `camera` rotates the camera on its mount, `steertest` holds a steer bias and
throttle and logs the motion, `latlog` logs the steering loop each frame, `interleave` switches interleaving and the
present hook, `camera forward=` moves the mount for the current car model (saved in `gta5op.ini`), `paint` and `trim`
recolour the car (where its model allows), and `engage`, `indicator` and `cruise dir=down five=1` press those keys. For testing,
`traffic on=0` clears and stops traffic (see [Varied scenes](#varied-scenes) for the scene commands), `lead dist=30 speed=0` places a car about that far ahead in the lane
(`leadspeed v=` changes its speed, `remove=1` deletes it, `clear=<model>` any left behind) and the state then reports
its true range, `gas secs=1` presses the gas as the driver would, the state's `vehicleAhead` is the range to the first
vehicle straight ahead, and `GTA5_LOG=<file>` on terminal 2 records the game state and the controls sent, a JSON line
each. `watch_views.py` shows the road and wide camera streams
side by side, as openpilot gets them, outlining the part the driving model takes in. `GTA5_DEBUG=1` on terminal 2 prints the commanded and measured motion each second. The bridge keeps
openpilot's UI window above the game's (`pin_ui.ps1`; `GTA5_PIN_UI=0` turns that off).

## Controls
| Key | openpilot |
|---|---|
| F6 | Engage / disengage |
| Brake (S, left trigger), steering | Disengage |
| Gas | Override, without disengaging |
| Up / down arrow | Cruise speed up / down (hold to repeat); the phone is blocked while connected |
| Shift + up / down arrow | Cruise speed to the next multiple of 5 up / down |
| Left / right arrow | Left / right blinker: a lane change at 20 mph or more, below that the next turn (it cancels once taken) |
| F7 | Map debug overlay on / off (`key_debug` in gta5op.ini) |

The bridge's keys also work in terminal 2: `1` resume/accel, `2` set/decel, `3` cancel, `q` quit.

## Navigation
Set a waypoint on the game's map while engaged and the car follows GTA's GPS route to it: the plugin sends the route (a
point every 5 m for 500 m) and navd's planner finds the turns in it, caps the speed to 12 mph for each (16 mph for
a gentler one), and signals
it, which below 19 mph asks the driving model for the turn. The model takes that request as a pulse when the blinker
comes on, and forgets it after several seconds and at a stop, so the bridge drops the blinker briefly for openpilot
(not the car's lights) every 2.5 s until the turn starts, when the car pulls away again, or when the model stops
expecting the turn. It slows gently (0.6 m/s^2, done 25 m before the turn), and for bends and ramps on the route to
2 m/s^2 sideways (the Tesla's steering is limited to about 3). A turn is signalled 5 s before it (28-50 m): further out
the model stops short of it. A turn straight after another is signalled as soon as the first is done.
Near the waypoint it slows to a stop there and disengages.

Before a turn the car changes into the lanes the map's turn arrows (`turn:lanes`) allow it from, else the leftmost lane
for a left turn or the rightmost for a right; going straight on through a junction it moves out of lanes whose arrows
only turn, or that end there (the road out has fewer lanes its way: the lanes that carry on are those in line with its
lanes, kerb to kerb where it is as wide, else by its line); and it moves back over if it drifts into the oncoming lanes,
by its lane reading or by the map's lanes alone (`map/lane_match.py`, outside junctions' areas: an oncoming lane, a
one-way the wrong way, the other direction's turn bay), with keepRight. Lane changes are planned back from the last place one may start (30 m before
a turn), about 8 s each, and when there isn't room left the set speed comes down for it; a turn the car still isn't in
the lane for is left for the route to come round again, as the model won't take it from the wrong lane anyway. The car's
lane comes from the road's lanes along our route (map/README.md: the map's lane tags, or GTA's links on a map without
them), else from the plugin's guess at the road it's on, and the
bridge asks for the lane change through openpilot's `NavDesire` param, which makes the blinker mean a lane change at
any speed rather than the turn it means below 19 mph: it is set 0.4 s before the blinker comes on and kept 0.5 s after
it goes off, as openpilot reads it every 0.2 s (and afresh as a blinker comes on); no lane change starts while the car
is still turning. On our routes nav also knows the forks: where a road splits, as at a freeway exit or where GTA splits
a road's lanes before a junction, it moves into the lanes of the route's branch (planned the same way, and never out of
the lanes for a fork or turn before it), and for a fork in the road, holds the model's keep desire towards that branch
from 4 s before it to 40 m past, unless a turn the other way follows, or the route stays on the larger branch. There
is no keepLeft on a two-way road, as it takes the car into the oncoming lanes. A turn ends at its way out (the blinker
off, no more pulses once begun), with keepRight for a moment if the car is still swinging round a left turn
(`GTA5_EXIT_CUE=0` turns that off); a turn is left to the route only when the car is surely out of its lane. Where a centre turn bay or slip lane opens for
a turn (GTA's bays are short slip-lane links into the median), the car slows for the turn from there, changes into it
from the lane beside it, and then signals the turn (`GTA5_BAY=0` signals from the lane beside it instead). The car's
lane from the route counts only while the car heads along the route's link and agrees with the plugin's, if it has
one. With the map's speed limits, engaging sets the limit where the car is; after that the set speed is the driver's
alone (the arrow keys), and nav caps the speed at the limit where the car is and slows for a lower one ahead
(`GTA5_FOLLOW_LIMIT=1` has the set speed follow the limit instead, as before, for A/B runs). Routes avoid service
roads (car parks, alleys, drives), which the model doesn't see as roads.
GTA's route sometimes turns back on itself, after a missed turn or around roads its GPS avoids; the model can't make a
U-turn, so nav drives on until GTA routes round instead. Nor has it a desire for straight on, and it sometimes turns
where the route doesn't, as from a lane that becomes a turn lane; when its expectation of a turn the route doesn't take
rises, nav asks for the keep desire away from it (keepRight against a left turn), meant for forks, which holds it on
the road; not for 60 m after a turn, as the model's expectation of the turn it took fades, nor on a one-lane road.
Like a turn, the model forgets it at a stop, so it's asked again as the car pulls away.

Stopped at a red light, the car pulls away by itself once it turns green, with a short press of the gas as a driver
would give. The game's AI drivers know the lights: those waiting with the car at a red light (or queued behind one) all
clear at once when it turns green, about 2 s before they move. With no AI traffic around, as with `traffic on=0`, the car
can't tell, and waits for the gas, as on the real car; nor does it go while traffic crosses ahead or someone is in front.

Nav's turn parameters (speeds by turn angle, where slowing starts and ends, when to signal, re-pulsing, lane change
lead) can come from a JSON file, `GTA5_NAVTUNE`, read again whenever it changes, for sweeps with `e2e.py sweep`; keys
left out keep the defaults (`selfdrive/navd/planner.py` Tune). With `"signal_mode": "entry"` a turn is signalled at
`signal_entry_offset` m past its junction's entry (the map's stop line, else GTA's junction nodes), as a driver signals on
entering the junction, rather than 5 s before it. How far the model turns depends on its speed in the turn, so the
turn's speed holds through the arc until the car heads its way out and is straight (`turn_release`, `turn_release_m`)
and lifts at `release_accel`; nav logs the speeds into, through and out of each turn.

The model chooses where to turn and doesn't always: after a stop it can carry straight on, and with a turn asked for
and no turning to take it stops, so a turn is signalled only within 50 m of one. `GTA5_DEBUG=1` prints nav's decisions.
`gta5_cmd.py waypoint x= y=` sets a waypoint (`off=1` clears it).

With a map of the game's roads built ([map/README.md](map/README.md)), nav can route over it with a standard router
instead (`GTA5_ROUTER`), which never asks for a U-turn, and `GTA5_MAP` serves a map view of the car and its route.
On that route the bridge also writes a route input for route-conditioned driving models (`selfdrive/navd/route_input.py`, trained
by gta5-train `--route`) to shared memory, which modeld feeds only to a model with a `route` input; it is zero off the
route, and `GTA5_ROUTE_INPUT=0` turns it off to A/B one model with and without it.
`OPENPILOT_PREFIX=gta5 GALLIUM_DRIVER=d3d12 python openpilot/tools/sim/bridge/gta5/watch_route.py` (in the venv) shows that input live.
Below it, it shows route input v2's lane slots (`selfdrive/navd/lane_slots.py`: the road here and the road out of the next
maneuver, counted from the kerb, with the lanes the route needs), which the bridge also writes to a shared-memory file of
their own with the car's lane (`GTA5_LANE_SLOTS=0` turns that off). The same slots are the route input's last 50 floats
(route input v2); a model trained on v1's 173 reads only the start of it. On the road here it marks the model's lane
(`modelV2.laneHead`, outlined, its P(index) as faint bars) against the car's true lane (a green mark).
`GTA5_NAV_MSGS=1` also publishes openpilot's `navInstruction` and `navRoute` for that route (`gta5_nav_msgs.py`: the
next maneuver, its lanes, the time and distance left, the route and the map's roads near it), which the onroad UI shows
beside the camera view while a route is active. `nav_ui_demo.py` shows that view offline, with screenshots of each scene,
from a route it finds over the map itself (its docstring says how).
nav's speed cap reaches openpilot as `navSpeed` (cereal's custom.capnp: the cap and why, a turn either way, a bend, a
lower limit ahead, a lane change, a turn bay or the arrival), and openpilot's longitudinal planner drives the lower of it
and the set speed, as sunnypilot's map curve speed control does: the set speed stays the car's (the arrow keys), and the
onroad UI shows the cap under the MAX box while it is lower. `GTA5_NAV_SPEED=can` instead lowers the simulated car's set
speed to the cap (the car reports it on CAN, as before), for A/B runs.
`GTA5_NOO` is Navigate on openpilot: `param` (the default) follows the UI's nav button (the `NavigateOnOpenpilot`
param); `on` drives regardless, which test runs (e2e, A/Bs) must set; `off` only guides. Guiding only, nav takes no
actions (no turn signals or desires, lane changes, slowing for turns, arrival or route input for the model) but still
caps the speed for the road's bends up to the next turn and for its speed limits, and holds keepRight out of the
oncoming lanes. With `param` or `off`, the UI's slide to end (it removes `NavDestination`) ends the route, and the game's waypoint is ignored until it
changes.

### Blind-spot monitoring
The car has a blind-spot monitor, as a Tesla's (`gta5_blindspot.py`): the plugin reports the vehicles around the car
(`nearby`: each one's place, heading, velocity and model bounds), and the bridge flags a side while a vehicle going our
way overlaps the lane beside us from 8 m behind our rear bumper to our front bumper, or is closing on it from up to 30 m
behind fast enough to be there within 3 s; by the vehicles' bodies, out to the far edge of the lane beside by the map's
lanes (4.5 m without them), never for oncoming, crossing or parked vehicles or a side with no lane our way, and held
0.5 s once clear. The flags reach openpilot as the Model 3's own blind-spot signals (`DAS_status`, read into
`carState.leftBlindspot`/`rightBlindspot`), so openpilot's lane change waits in preLaneChange with "Car Detected in
Blindspot", and below 19 mph its lane turn desire waits too. A lane change nav asks for works like sunnypilot's
nudgeless one: the blinker comes on when nav decides, stays on while the blind spot that way is occupied (the driver
holds off the nudge, and openpilot would block it anyway), and the change starts as soon as it clears; nav's lane plan
still shows it. Still held at the last place it may start (for a fork, its last start point; for a turn 30 m before it),
nav gives it up and the blinker goes off, leaving the exit or turn to a reroute rather than changing into a car; and
likewise once the lanes ahead no longer need it (a reroute, or the car's lane has moved).
`GTA5_BLINDSPOT=0` turns it off; `GTA5_BLINDSPOT_TUNE` is a JSON file of its settings (the module's docstring says
what each is and why), read again whenever it changes. Needs the plugin's `nearby` (no flags with an older plugin).

### navd and the GTA layer
Navigation is being moved out of the bridge into a game-agnostic navd (`selfdrive/navd/`), which is to plan from only
what a real car and device give it (GNSS, the car's state, the model's outputs, an OSM map and router), so it ports to
the road. So far it holds the planner (`planner.py`, which takes `NavInputs` and returns a cap, NavDesire changes and
requests to the driver, `inputs.py`), the route input encoder and lane slots; it still runs inside the bridge, and its
inputs still carry the game's true pose and lane (`truth_*`). The bridge's side of it:
- `gta5_navd.py`: the planner's inputs from the game's state and our route, and the destination: the game's waypoint or
  a map view pick, written to the `NavDestination` param as lat/lon.
- `gta5_driver.py`: the simulated driver, who approves every request: works the stalk (the game's indicator) for turns
  and lane changes, nudges the wheel to start a lane change, cancels the indicator after a turn, cancels cruise on
  arriving and pulls away at green lights.
- `gta5_gnss.py`: GNSS from the car's position as a comma 3X publishes it (`gpsLocation`, 1 Hz, with noise and delay;
  `GTA5_GPS=qcom3x|ublox|perfect|off`, `GTA5_GPS_SEED`).
- `nav_replay.py`: replays e2e trips (bridge log and traces) through the bridge's nav into decision logs, to check that
  a refactor leaves every decision as it was.
- `match_replay.py`: replays the bridge log's drives through navd's map matching (`selfdrive/navd/map_match.py`) on
  simulated 3X GNSS, scored against the game's own pose snapped to the map: wrong road and wrong way shares, episodes,
  latency after turns, false and missed reroutes. `Navigator.update(..., match=...)` takes its matches, and
  `GTA5_NAV_MATCH=1` has the bridge route from them (off by default: the match puts the car on its road's line, so the
  route's lane for the car is wrong). `nav_replay.py run` follows the same variable.

### Map debug overlay and GPS route
`gta5_cmd.py debug on` (or F7) draws the map around the car into the world, from the player's camera, to spot map
problems while driving (`gta5_overlay.py`, drawn by `core.cpp` DrawDebug). Each kind is a strip of its own width lying
on the game's ground, colours as the map view's where it has them:

| Layer (`layers=`) | Kinds |
|---|---|
| edges `e` | kerbs (the road's edge, kerbed or not), green, 0.35 m |
| dividers `d` | lane lines white, centre lines and median edges yellow, 0.2 m, edged dark so they show over the game's own paint (`casing=0` drops it); dashed ones in 3 m dashes, 6 m gaps (the map has no dash phase, so they needn't line up with the paint's); a double line is two 0.3 m apart |
| parking `p` | where a parking lane meets the lanes, purple, dashed |
| stops `s` | stop lines 0.6 m, red at lights, orange at stop signs; give way lines orange, dashed |
| crossings `x` | zebra stripes, 3 m |
| junctions `j` | junction areas' outlines, blue (`fill` `f`: the areas, translucent) |
| arrows `a` | each lane's turn:lanes arrows, painted 4 and 20 m out from its stop line: blue left (and U-turns), white through, orange right |
| tapers `t` | the middle of a lane opening or closing along a taper or bay, translucent cyan, 1.2 m |
| flags `q` | a magenta post where a road's lane count changes at a node with no taper (suspect map data) |
| route `r`, nav `n`, points `m` | the route a 1.75 m translucent red band (darker behind the car, where it was drawn as the car passed), nav's lane plan a dashed white line over it, the next turn a white cylinder and where its signal comes on an amber cone |
| blind spot `z` | each side's blind-spot zone outlined faintly in amber, filled red down its middle while occupied (sent with the route, every 0.1 s) |

Strips widen and lift with distance beyond `grow=` m (40; up to 4x) so a 0.2 m line stays a few pixels wide; they're
drawn to `dist=` m (120; the bridge sends 150). `width=` scales them all, `widths=e:2,d:1.5` some layers; `ground=0`
draws them at the map's heights instead, which shows height errors in the map; `thin=1` gives back the 1-px lines.
The ground under each point comes from GET_GROUND_Z_FOR_3D_COORD, up to `probes=` (100) new points a frame, kept by
place, and the points a line on the ground doesn't need are then dropped; the status line shows the points, draw calls
and the share on the ground. At a busy freeway interchange that's ~4000 draw calls and 0.7 ms of the script thread a
frame, as much as `thin=1`. Lane edges and dividers stop at junctions (from the stop line in, and inside junction
areas, but for the lines of a road carried straight on past side roads). The bridge sends them every 0.5 s, and the route alone
every 0.1 s between, from where the car will be as it is drawn, while the plugin's debug is on (`GTA5_OVERLAY=0` stops
it); on a lane-tagged map they're all from its tags (built once and cached, `GTA5_OVERLAY_CACHE`), else from GTA's
lane bands. The plugin draws them only on the player's frames, never on the openpilot camera's (they're the ones with
the capture marker), and not at all while the bridge records unless `force=1`.

`gta5_cmd.py gpsroute on` (or `gps_route=1` in gta5op.ini) shows our route on the minimap and pause map as GTA's own
purple GPS line (a custom GPS route, `colour=` a HUD colour, `max=` points, 100 by default as GTA's limit isn't
documented; a longer route is sent in parts as the car goes). It works without the overlay. GTA's own route line to the
map's waypoint can't be hidden in Enhanced (SET_BLIP_ROUTE on the waypoint doesn't), so while ours shows the plugin
takes the waypoint off the map and holds it, marked by a purple blip of its own and still reported as the waypoint (the
bridge, nav and the AI driver go on using it); a new waypoint set on the map replaces it, `gta5_cmd.py waypoint off=1`
clears it, it's dropped within 20 m as GTA drops its own, and `gpsroute off` gives it back. `take=0` leaves the waypoint
alone (both lines show; `colour=` tells them apart). `gta5_cmd.py gtadirs x y z` prints GTA's own GPS directions from the car to a
point (its next turn and the distance to it), and `compare_dirs.py <trips>` compares them with our router's at e2e trips'
starts (it places the car, so run it only with nothing else driving).

Both reset when the plugin's core reloads (a DLL swap, a GTA restart), so the bridge keeps them: it merges every
`debug` and `gpsroute` command sent through it, takes up what the plugin's state shows (settings from before the bridge
started, F7), and sets them again each time the game connects, which a reload does. `GTA5_DEBUG_OVERLAY` and
`GTA5_GPSROUTE`, written as `gta5_cmd.py`'s arguments after `debug` or `gpsroute` (`GTA5_DEBUG_OVERLAY="on layers=all"`,
`GTA5_GPSROUTE=on`), set them from the bridge's start; unset, the bridge sets nothing until told or the state shows one on.

Checking that none of it reaches openpilot: with interleaving on, turn the overlay on with `force=1` and gpsroute on,
check the lines sit on the road from the player's camera at a few known places (the L7 X junction, a freeway: X1),
then `gta5_cmd.py snap /tmp/dbg` and record a minute (`GTA5_RECORD`) and look through the road and wide frames: no
coloured lines, markers, minimap or GPS line may show in any of them.

## AI expert driver
The game's own traffic AI can drive the car, as expert driving to record. `gta5_cmd.py ai on` tasks the player as the
driver of its car (to the map's waypoint, else wandering; `x= y= z=` for a point, `speed=` m/s cap, `style=` driving style
bits, `ability=`, `aggr=`), ignores openpilot's controls and the player's driving keys, and `ai off` (or the engage key)
gives the car back. Expert mode (`gta5_expert.py`, on with `GTA5_EXPERT=1` on terminal 2) follows our map's route:
`gta5_cmd.py expert route <trip>` places the car at an e2e trip's start, sets its destination and drives it, keeping the
AI's target 60-120 m ahead just past the next junction, with the indicators from the route's turns, openpilot disengaged
and a JSON line per frame in `expert.jsonl` beside `GTA5_LOG`; `expert off` stops it.

## Recording
`GTA5_RECORD=<folder>` on terminal 2 records the drive for training the driving model, in comma's comma1M segment layout
(`gta5_record.py`): each minute in the car becomes `<folder>/data/<hex>/` with the road and wide frames camerad got as
HEVC (`fcamera.hevc`, `ecamera.hevc`), `frame_info.safetensors` and `localizer.safetensors` (the game's own poses, with
the map placed at Los Angeles), which the commaai/torchtitan and openpilot.distill loaders read; and `gta5.npz` with the
game state, nav's and the AI driver's inputs, and modeld's outputs per frame, with the desire input it was given.
`SEND_RAW_PRED=1` on terminal 1 adds modeld's raw output vector, its vision features included. libx265 encodes on the
CPU (about 1.2 cores per camera, with the system ffmpeg); the video is about 75 MB a minute per camera. `GTA5_RECORD_MOUNT=<forward>,<up>` is the
camera's offset from the car's origin, which the poses are moved to (1,0.6 by default, the plugin's mount on the test
car). `python -m openpilot.tools.sim.bridge.gta5.gta5_record finalize <segment>` remakes a segment's safetensors from its
`gta5.npz`, and `replay` makes a synthetic segment from a `GTA5_LOG`, for testing loaders.

The camera's mount comes from the plugin's state when it reports one (`GTA5_RECORD_MOUNT` is then unused): `gta5.json`
has it as `mount` (forward, up) and `mount_detail` (with x right, pitch and yaw, which the poses include), and the scene
at the segment's start as `scene` (the car, weather and time, traffic density). A change of mount ends the segment.

### Varied scenes
For training data that isn't all one car on a sunny afternoon, `gta5_cmd.py` sets the scene (each is also a plugin command
of the same name, and the state reports it):
- `world hour= minute= weather= [transition=<s>] [rain=0-1|-1] [freeze=1] [clear=1]`: the time and the weather (EXTRASUNNY,
  CLEAR, CLOUDS, OVERCAST, SMOG, FOGGY, CLEARING, RAIN, THUNDER), at once or over a transition, the rain and puddles
  (-1 the weather's own), the clock stopped; `clear=1` lets the game's weather cycle again. State: `world`.
- `traffic vehicles= peds= parked= [random= scenario=] [reset=1]`: density multipliers (0-3), held each frame until
  changed; `on=0` still clears and stops traffic. State: `density`.
- `vehicle <car> [primary= secondary= pearl= wheel= dirt=] [colours=keep] [force=1]`: swaps the player's car for a new
  one where it is, at its speed, deleting the old one (the same model is only recoloured). `vehicle list` shows the
  curated cars (`gta5_scene.py`: about 20 common sedans, hatchbacks, SUVs, vans and a pickup), `model3` is the add-on
  Model 3, and any model name or 0x hash works; `random` picks one. Colours are palette indices, random unless given.
  State: `vehicle` (model, name, colours, dirt, swaps, error).
- `mount dashcam|comma [drop= back=] [jitter=1|0] [dx= dy= dz= pitch= yaw=]`: `comma` (the default) is a comma device's
  place, at the height openpilot's calibration expects; `dashcam` probes the car's own collision down its centre line
  for the roof over the driver and the windscreen ahead, and puts the camera `drop` m (0.08) below the roof and `back` m
  (0.06) behind the glass, ahead of the driver's seat (from the windscreen bone and the model's height when the probes
  miss; the comma place if the result is outside the cabin). `jitter=1` adds up to 2 cm and 0.5 deg at random. State:
  `mount` (mode, source: bone, dims or ini for comma, probe or bones for dashcam; x, y, z, pitch, yaw on the car).
- `randomise [seed] [model_share=0.6] [mount=dashcam|comma|keep] [jitter=0] [vehicle=0] [world=0] [traffic=0] [dry=1]`
  picks all of these for a drive, repeatably for a seed, prints them as one `randomise {json}` line, sends them, and
  waits 3 s for the car swap so a following `setup` places the new car. 60% Model 3, else the curated cars; half the
  time the model's own colours, else real-world shares (white, black, grey, silver, then blue, red...); time 75% day
  (07-19), 10% dawn or dusk, 15% night; weather mostly clear or cloudy, 5% fog, 5% smog, 9% rain or thunder; vehicle
  density 0.3-1.5 (mostly 1), pedestrians 0.3-1.2, both halved at night; the dashcam mount 5-10 cm below the roof with
  jitter. `record_run.py --randomise` runs it before each trip; leave its `--weathers`, `--hours-of-day` and `--traffic`
  unset, as they'd override it.

Other cars are for recording the AI expert: the bridge still models a Model 3 (its steering angle comes from the
curvature through the Model 3's vehicle model), the plugin's throttle table, coasting and brake gain were measured on a
sedan, its steer bias limit is a fraction of each car's own lock (the curvature gain is relearned per car), and the
driving model judges speed from the camera's height, so a dashcam mount or a tall car makes modeld's recorded outputs
and calibration unlike a comma device's. Keep openpilot runs on the Model 3 with the comma mount. On the Model 3 at the
comma mount (1.22 m above the ground), the bonnet isn't in the road camera, and is the bottom eighth or so of the wide
camera's picture, with the wipers below it.

### Junction recordings
For training the route input, the same approach driven out of a junction each legal way: `junction_plan.py plan --out
plan.json` picks junctions on our map (by kind, spread over the city, the in-game A/B trips' junctions held out for
eval) and routes each approach's ways out as the bridge would, `junction_run.py run --plan plan.json --name <run>` drives
them with the AI expert while the bridge records (record_run.py's Run; it resumes a stopped run and writes
`<run>.manifest.json` beside the run's log), and `junction_rec.sh <plan> <run> --until <time>` runs that in tmux,
resuming after the game comes back. gta5-train's `scripts/juncrec_labels.sh` makes the recordings training data.

## How it works
- The plugin attaches a scripted camera to the car where a comma device mounts, just behind the windscreen (from the
  car's windscreen bone, and at least 0.29 m below the roof, a Model 3's device height; else 1.22 m above the ground),
  level, hides the HUD,
  and captures the game window with Windows.Graphics.Capture.
- With `interleave=1`, the game renders the openpilot camera only on one frame per 20 Hz capture, and the player's own
  camera on the rest. Those frames carry a small coloured marker at the top-left, so timing between the script and the
  renderer doesn't matter. A hook on the game's D3D12 present checks for it on the GPU: it copies a marked frame out
  for openpilot and presents the player's previous frame in its place, so the openpilot view never shows
  (`present_hook=0` captures the window instead, where it flickers). Anything that blends frames together mixes the two
  views: turn off TAA, DLAA, DLSS and FSR upscaling, ray tracing's temporal denoising, and motion blur. DLSS frame
  generation on its own looked clean. Supersampling (2x) anti-aliases the magnified road view; it doesn't add pixels,
  as the frames are taken after the game scales them down. A 90 FPS cap leaves the player 70 and the GPU room for the
  model. The openpilot frames hide the help text and hold sounds where the player's camera hears them; the capture blacks
  out the radar's corner, as hiding the radar on them moves the notification feed and makes it flicker. The game's
  scripts see a script camera rendering on the frame after each, and the interaction menu (M) closes itself then, so
  the plugin swaps `IS_GAMEPLAY_CAM_RENDERING` in that script's native table for one that answers yes while connected
  (`script_hook.cpp`, with YimMenuV2's patterns for the Enhanced build; if they stop matching, the log says so and the
  menu just closes).
- One 76 degree render covers both openpilot cameras, magnified about 2.8x for the road camera and with the wide
  camera's outer field black. A pixel shader resamples it through the comma 3X lenses into NV12, as the Slow Roads
  bridge does (`lens=0` for openpilot's plain pinhole cameras). `split_views=1` renders each camera separately (30
  degree road, 116 degree wide, marked cyan and yellow) at 40 hidden frames a second, but the game sometimes renders a
  frame with the previous frame's camera, whether the field of view changes or the active camera does, even when set
  frames ahead; the bridge drops pairs whose wide center doesn't match the road view, which leaves gaps in turns.
- Frames go uncompressed over TCP (about 140 MB/s at 20 Hz), which the WSL network carries easily; `gta5_rx.py` copies them
  into shared memory in its own process, so the bridge's 100 Hz threads keep the GIL. For the same reason the debug
  overlay is made in a process of its own (`GTA5_OVERLAY_PROCESS=0` puts it back in a thread), and the bridge leaves
  Python's full garbage collections, which walk the maps' million or so objects for 0.4-0.8 s with the GIL held, until
  it has been idle for 2 s (no game frames, or on foot). Either stall stopped the camera and sensor threads long enough
  for modelV2 gaps of 400 ms and locationd errors. `stall_bench.py realtime <trips>` replays recorded trips (nav_replay's
  inputs) through the bridge's threads on the real clock and reports the main loop's, camera's and car thread's gaps.
- Control uses carControl's `curvature` and `accel`. The plugin steers with the game's steer bias, which sets a wheel
  angle and so a path curvature proportional to it. The game turns the wheels less as the speed rises (on the Model 3,
  3.4 1/m per unit bias up to 5 m/s, 2.2 at 11 m/s, 1.3 at 20 m/s, measured with `steertest`), and the low-speed gain
  depends on the car: the plugin scales a measured table of that fall by a low-speed gain it learns while driving
  (`curv_gain` is where it starts; collisions and the steering lock stay out of the fit), feeds forward the bias for
  the curvature, and integrates the yaw rate error. `gta5_cmd.py steergain schedule=0` goes back to one gain fitted at
  all speeds, for comparison. Throttle comes from a measured table of the acceleration it adds over coasting, which in the game is a hard
  -3 m/s^2 or so, and the brake covers anything beyond that; a stop is held with the handbrake, as the game's brake
  reverses a stopped car.
- The steering angle openpilot sees is the yaw rate's curvature through the car's own fixed vehicle model (carParams',
  no offset), and the Model 3's commanded angle goes back through the same model, so paramsd learns the car's values as
  on a real one. The IMU has the physics' yaw rate, and gravity from the car's grade and bank, which gives locationd its
  pitch and roll. The throttle adds what gravity takes on a grade, as a real car's drive unit delivers its acceleration.
- The plugin sets the maximum wanted level to zero while connected.
- openpilot and the bridge each restart on their own: modeld takes the restarted bridge's camera streams.
