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
and puts it on the road nearest a point, `camera` rotates the camera on its mount, `steertest` holds a steer bias and
throttle and logs the motion, `latlog` logs the steering loop each frame, `interleave` switches interleaving and the
present hook, `camera forward=` moves the mount for the current car model (saved in `gta5op.ini`), `paint` and `trim`
recolour the car (where its model allows), and `engage` and `indicator` press those keys. For testing,
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
  model.
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
