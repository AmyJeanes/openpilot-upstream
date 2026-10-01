Slow Roads bridge
=================

Drives [Slow Roads](https://slowroads.io) with openpilot. The bridge injects a script into the running game that renders
comma 3X-like road and wide cameras from the car, streams them and the car's state to openpilot, and applies openpilot's
commands to the car.

It's developed with the Steam version on Windows and openpilot in WSL. Other setups should work but are untested:
openpilot on native Linux (with the game under Proton, or on another machine), and the web version in Chrome started with
`--remote-debugging-port`.

## Setup
- openpilot built (`scons`), with a GPU for the driving model: the CPU takes ~350 ms per frame. Build with
  `MODELD_DEV=CUDA scons -j$(nproc)` (or another tinygrad device; only CUDA is tested). The device is part of the build,
  so keep `MODELD_DEV` set when rebuilding.
- The `tools` extra, which openpilot's setup installs (`uv sync --all-extras`); it includes the bridge's `websockets`
  and `av`.
- Steam launch option for Slow Roads: `--remote-debugging-port=9339`. The bridge uses that port to install its script.
  The port only listens on the game machine's localhost. For a game on another machine, forward it (e.g.
  `ssh -L 9339:localhost:9339 <game machine>`), or set `SLOWROADS_GAME_HOST` if it's reachable some other way. The game
  connects back to the bridge on port 8790.
- In the game: Drive lane = right (openpilot reads oncoming cars in the other lane as leads), and be driving, not paused.

## Running
Use an `OPENPILOT_PREFIX` so params and messaging stay separate from any other openpilot checkout.
```bash
export OPENPILOT_PREFIX=slowroads MODELD_DEV=CUDA
BLOCK=soundd BIG=1 ./openpilot/tools/sim/launch_openpilot.sh   # terminal 1 (GALLIUM_DRIVER=d3d12 for a GPU-rendered UI)
./openpilot/tools/sim/run_bridge.py --simulator slowroads       # terminal 2
./openpilot/tools/sim/bridge/slowroads/view_cameras.py          # optional: the full camera frames; the UI shows a crop
```
`probe_slowroads.py [seconds]` runs both headless and prints engagement health.

Options on terminal 1:
- `MODELD_BIG=1` runs the big driving model. Download `big_driving_supercombo.onnx` from
  [commaai/openpilot_driving_models](https://huggingface.co/commaai/openpilot_driving_models) and build with
  `MODELD_BIG_ONNX=<path> MODELD_DEV=CUDA scons`. The shipped `big_driving_tinygrad.pkl` is compiled for comma's AMD
  GPU and can't run elsewhere.
- `MAX_LAT_ACCEL=<m/s^2>` raises openpilot's 3.0 m/s^2 lateral acceleration limit.

`SLOWROADS_DEBUG=1` on terminal 2 prints the commanded and measured motion each second.

## Controls
The game's own controls stand in for the car's:

| Game control | openpilot |
|---|---|
| Autodrive toggle | Engage / cancel. It turns itself back off when openpilot can't engage, and openpilot's alert says why. |
| Brake | Disengage |
| Gas, steering | Override, without disengaging |
| Indicators (arrow keys) | Blinker: starts a lane change at 20 mph or more, and is canceled when it completes |

The bridge's keys also work in terminal 2: `1` resume/accel, `2` set/decel, `3` cancel, `q` quit.

## How it works
- `slowroads_inject.py` reaches the game's module-scoped objects through never-pausing conditional breakpoints on the
  DevTools port (`renderLive` for the game, `handleInput` for the driver's controls), then evaluates `game/*.js` in the
  page. A game update that renames these methods breaks injection. Run it directly to reinstall the page script after
  editing it. WSL can't reach a Windows game's DevTools port, so there it runs the same steps in
  `game/inject.ps1` through Windows PowerShell.
- `game/sr-page.js` renders each camera from a fixed mount (1.22 m high, level, at the cockpit) into its own render
  target after every game frame, at 20 Hz.
  - `lens.js` reproduces the comma 3X lenses (fleet-median calibrations): the narrow camera is a pinhole with barrel
    distortion, and the wide one is a fisheye. Each is rendered as a wider pinhole view and resampled through the lens,
    with the wide one limited to 65 degrees off-axis for frame time. Set `lens: false` in the page config for
    openpilot's plain pinhole cameras.
  - `scene.js` redoes what the game sets up for its own camera each frame: the sky and haze quad, clouds and stars,
    the falling snow box and flake size. It also hides the car body, which the wide camera would otherwise see through.
- Frames are hardware-encoded to H.264 with WebCodecs and sent over a WebSocket to the bridge machine's IP (under WSL,
  its own IP: `localhost` forwarding goes stale when the distro restarts). Keyframes are sent only on connect or on request; their
  size makes them arrive late. `slowroads_rx.py` decodes in its own process, into shared memory, so it doesn't compete
  with the bridge's 100 Hz threads.
- `slowroads_world.py` hands frames to openpilot at least 40 ms apart. modeld drops road frames that arrive closer
  together, which invalidates camera odometry. It also reports the car's state:
  - the IMU from the game's physics;
  - the steering angle through openpilot's learned vehicle model.
- Control uses carControl's `curvature` and `accel`, not the bridge's Honda-shaped steering angle and pedals, so any
  game vehicle follows the plan.
  - `controls.js` maps acceleration through the car's tuning, plus a bounded integral.
  - It holds the car at a stop, as a car's brake hold would.
  - It brakes only below -0.3 m/s^2, so the brake lights don't flicker.
  - It adds the driver's steering to openpilot's.
- The game puts a stopped or strayed car back on the road. The bridge then presses resume, or the car sits still and
  the game resets it again.
- The bridge clears `Offroad_ExcessiveActuation` on start. Game collisions and resets can trip it, and it latches.

## Limits
- `locationdTemporaryError` alerts mean the driving model is too slow for the GPU. When one frame takes longer than the
  50 ms between frames, modeld skips the next one and marks camera odometry invalid for that frame. The game runs
  uncapped (several hundred fps) and competes for the GPU. Set a max framerate in the game's Graphics settings, lower
  its render scale, or run the small model (without `MODELD_BIG`). `modelV2.modelExecutionTime` shows the model's
  time per frame.
- After a full stop in experimental mode, openpilot waits for resume (`1`) or gas.
- Calibration is learned as on a car: a fresh params directory calibrates once. Its result matches the mount.
- Both cameras are perfectly aligned. On a real 3X the wide one is about 1 degree off, which openpilot calibrates out.
- `cruiseMismatch` and `noGps` warnings are expected: the simulated car is a Honda with fixed cruise messages, and has
  no GPS.
- If WSL has been idle, interop can be lost (`Exec format error` from `powershell.exe`). The bridge prints the command
  to re-register it.
