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
`traffic on=0` clears and stops traffic, `lead dist=30 speed=0` places a car about that far ahead in the lane
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

The bridge's keys also work in terminal 2: `1` resume/accel, `2` set/decel, `3` cancel, `q` quit.

## Navigation
Set a waypoint on the game's map while engaged and the car follows GTA's GPS route to it: the plugin sends the route (a
point every 5 m for 500 m) and the bridge (`gta5_nav.py`) finds the turns in it, lowers the set speed to 12 mph for
each (16 mph for a gentler one; the car reports a lower cruise speed, as a car's own navigation would), and signals
it, which below 19 mph asks the driving model for the turn. The model takes that request as a pulse when the blinker
comes on, and forgets it after several seconds and at a stop, so the bridge drops the blinker briefly for openpilot
(not the car's lights) every 2.5 s until the turn starts, when the car pulls away again, or when the model stops
expecting the turn. It slows gently (0.6 m/s^2, done 25 m before the turn), and for bends and ramps on the route to
2 m/s^2 sideways (the Tesla's steering is limited to about 3). A turn is signalled 5 s before it (28-50 m): further out
the model stops short of it. A turn straight after another is signalled as soon as the first is done.
Near the waypoint it slows to a stop there and disengages.

Before a turn the car changes into the leftmost lane for a left turn or the rightmost for a right, and it moves back
over if it drifts into the oncoming lanes. Lane changes are planned back from the last place one may start (30 m before
a turn), about 8 s each, and when there isn't room left the set speed comes down for it; a turn the car still isn't in
the lane for is left for the route to come round again, as the model won't take it from the wrong lane anyway. The car's
lane comes from GTA's roads along our route (map/README.md), else from the plugin's guess at the road it's on, and the
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
one. With the map's speed limits, engaging sets the limit where the car is, the set speed
follows it as it changes along the route (`GTA5_FOLLOW_LIMIT=0` leaves the set speed alone), and a lower limit ahead
slows the car before it. Routes avoid service roads (car parks, alleys, drives), which the model doesn't see as roads.
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
left out keep the defaults (`gta5_nav.Tune`). With `"signal_mode": "entry"` a turn is signalled at
`signal_entry_offset` m past its junction's entry (GTA's stop line, else its junction nodes), as a driver signals on
entering the junction, rather than 5 s before it. How far the model turns depends on its speed in the turn, so the
turn's speed holds through the arc until the car heads its way out and is straight (`turn_release`, `turn_release_m`)
and lifts at `release_accel`; nav logs the speeds into, through and out of each turn.

The model chooses where to turn and doesn't always: after a stop it can carry straight on, and with a turn asked for
and no turning to take it stops, so a turn is signalled only within 50 m of one. `GTA5_DEBUG=1` prints nav's decisions.
`gta5_cmd.py waypoint x= y=` sets a waypoint (`off=1` clears it).

With a map of the game's roads built ([map/README.md](map/README.md)), nav can route over it with a standard router
instead (`GTA5_ROUTER`), which never asks for a U-turn, and `GTA5_MAP` serves a map view of the car and its route.

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
CPU (about 1.3 cores per camera); the video is about 30 MB a minute per camera. `GTA5_RECORD_MOUNT=<forward>,<up>` is the
camera's offset from the car's origin, which the poses are moved to (1,0.6 by default, the plugin's mount on the test
car). `python -m openpilot.tools.sim.bridge.gta5.gta5_record finalize <segment>` remakes a segment's safetensors from its
`gta5.npz`, and `replay` makes a synthetic segment from a `GTA5_LOG`, for testing loaders.

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
  model. The openpilot frames hide only the radar, and hold sounds where the player's camera hears them. The game's
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
  into shared memory in its own process, so the bridge's 100 Hz threads keep the GIL.
- Control uses carControl's `curvature` and `accel`. The plugin steers with the game's steer bias, which sets a wheel
  angle and so a path curvature roughly proportional to it, by an amount that depends on the car: it learns that gain
  while driving (`curv_gain` is where it starts), feeds forward the bias for the curvature, and integrates the yaw rate
  error. Throttle comes from a measured table of the acceleration it adds over coasting, which in the game is a hard
  -3 m/s^2 or so, and the brake covers anything beyond that; a stop is held with the handbrake, as the game's brake
  reverses a stopped car.
- The steering angle openpilot sees is the yaw rate's curvature through the car's own fixed vehicle model (carParams',
  no offset), and the Model 3's commanded angle goes back through the same model, so paramsd learns the car's values as
  on a real one. The IMU has the physics' yaw rate, and gravity from the car's grade and bank, which gives locationd its
  pitch and roll. The throttle adds what gravity takes on a grade, as a real car's drive unit delivers its acceleration.
- The plugin sets the maximum wanted level to zero while connected.
- openpilot and the bridge each restart on their own: modeld takes the restarted bridge's camera streams.
