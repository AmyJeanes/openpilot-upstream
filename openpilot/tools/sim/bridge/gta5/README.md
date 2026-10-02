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
export OPENPILOT_PREFIX=gta5 MODELD_DEV=CUDA
# once after each WSL restart, which clears /dev/shm
python3 -c "from openpilot.common.prefix import OpenpilotPrefix; OpenpilotPrefix('gta5').create_dirs()"
GALLIUM_DRIVER=d3d12 BLOCK=soundd ./openpilot/tools/sim/launch_openpilot.sh   # terminal 1
./openpilot/tools/sim/run_bridge.py --simulator gta5            # terminal 2
```
Get into a car in the game. Once the bridge is running, the plugin connects to it and switches to the openpilot camera; it
switches back when the bridge stops. The bridge writes its address to `C:\Users\Public\gta5op-bridge.txt` for the plugin;
set `bridge=` in `gta5op.ini` if that path isn't reachable from WSL.

`gta5_cmd.py` sends debug commands through the running bridge: `snap` saves the next frames as PNGs, `setup` spawns a car
and puts it on the road nearest a point, `camera` rotates the camera on its mount, `steertest` holds a steer bias and logs the motion, and `engage` presses the
engage key. `../slowroads/view_cameras.py` shows the full camera frames openpilot gets. `GTA5_DEBUG=1` on terminal 2
prints the commanded and measured motion each second.

## Controls
| Key | openpilot |
|---|---|
| F6 | Engage / disengage |
| Brake (S, left trigger) | Disengage |
| Gas, steering | Override, without disengaging |
| `=` / `-` | Cruise speed up / down (hold to repeat) |
| `,` / `.` | Left / right blinker: starts a lane change at 20 mph or more |

The bridge's keys also work in terminal 2: `1` resume/accel, `2` set/decel, `3` cancel, `q` quit.

## How it works
- The plugin attaches a scripted camera to the car at the comma mount (1.22 m above the ground, level), hides the HUD,
  and captures the game window with Windows.Graphics.Capture.
- With `interleave=1`, the game renders the openpilot camera only on one frame per 20 Hz capture, and the player's own
  camera on the rest. Those frames carry a small magenta marker at the top-left that the capture picks them out by, so
  timing between the game and the capture doesn't matter. They still show on screen, as flicker. Anything that blends
  frames together mixes the two views: turn off TAA, DLSS and FSR (including frame generation), and motion blur.
- One game camera covers both openpilot cameras. The driving model only samples about 30 degrees either side of center,
  so a pinhole render with a 76 degree vertical field of view has enough coverage and resolution for both; the wide
  camera's outer field is black. A pixel shader resamples it through the comma 3X lenses into NV12, as the Slow Roads
  bridge does (`lens=0` for openpilot's plain pinhole cameras).
- Frames go uncompressed over TCP (about 140 MB/s at 20 Hz), which the WSL network carries easily; `gta5_rx.py` copies them
  into shared memory in its own process, so the bridge's 100 Hz threads keep the GIL.
- Control uses carControl's `curvature` and `accel`. The plugin steers with the game's steer bias, which turns the car at
  a yaw rate roughly proportional to it (`yaw_gain`; `gta5_cmd.py steertest` measures it): it feeds forward the yaw rate
  the curvature needs and integrates the error. Throttle and brake follow the requested acceleration plus an integral on
  the measured one, and a stop is held with the handbrake, as the game's brake reverses a stopped car.
- The steering angle openpilot sees is the yaw rate's curvature through openpilot's learned vehicle model, and the IMU
  uses the physics' yaw rate.
- The plugin sets the maximum wanted level to zero while connected.
