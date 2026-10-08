#!/usr/bin/env python3
"""Sends a debug command to the GTA V plugin through the running bridge, e.g.
  gta5_cmd.py snap /tmp/gta5                    save the next road/wide frames as PNGs, and their state
  gta5_cmd.py burst /tmp/burst 40               save the next 40 frames' luma, quarter size, as JPEGs
  gta5_cmd.py setup x=2420 y=3000 z=46 model=sultan speed=20 hour=12 weather=EXTRASUNNY
                                                spawn a car and/or put it on the road nearest a point
  gta5_cmd.py world hour=12 weather=EXTRASUNNY freeze=1
                                                set the time and weather, and stop the clock (minute=; transition=30 changes
                                                the weather over 30 s; rain=0-1 the rain and puddles, -1 the weather's own;
                                                clear=1 lets the game's weather cycle again)
  gta5_cmd.py traffic vehicles=0.6 peds=0.8 parked=1
                                                traffic and pedestrian density, held until changed (reset=1; on=0 none)
  gta5_cmd.py vehicle premier [colours=random|keep] [primary=111 secondary=0 pearl=5 wheel=156 dirt=2] [force=1]
                                                swap the player's car for a new one where it is (model3, a car of
                                                `vehicle list`, any model name or 0x hash; random picks one), coloured
  gta5_cmd.py mount dashcam [drop=0.08 back=0.06] [jitter=1] [dx= dy= dz= pitch= yaw=]
                                                the camera at the top of the windscreen (comma: a comma device's place, the
                                                default); jitter=1 moves it a little at random, jitter=0 not at all
  gta5_cmd.py randomise [seed] [model_share=0.6] [mount=mixed|dashcam|comma|keep] [comma_share=0.5] [jitter=0] [vehicle=0] [world=0] [traffic=0]
             [dry=1]                            pick a drive's weather, time, traffic, car, colours and mount (gta5_scene.py),
                                                print them as a JSON line and set them; waits for the car swap
  gta5_cmd.py camera yaw=5 pitch=0              rotate the camera on its mount (degrees)
  gta5_cmd.py ai on speed=12 style=1076369579   the game's AI drives the car: to the map's waypoint, else wandering
  gta5_cmd.py ai x=100 y=-200 z=30              ... to a point; ai off gives the car back
  gta5_cmd.py expert on speed=12                expert mode (gta5_expert.py; the bridge runs with GTA5_EXPERT=1): the AI
                                                drives our route to the destination set; off stops it
  gta5_cmd.py expert route L1 [speed=12 ...]    put the car at an e2e trip's start (a name in ~/gta5test/e2e/*.txt, or
                                                'x,y,z,heading[,lane]>dx,dy'), set its destination and drive it
  gta5_cmd.py expert status                     the control file
  gta5_cmd.py debug on [layers=edges,dividers,stops,junctions,arrows,crossings,parking,tapers,flags,route,nav,points,fill|all]
             [width=1] [widths=e:2,d:1.5] [dist=120] [grow=40] [ground=1] [lift=0.05] [casing=1] [thin=0] [sides=1] [force=1]
                                                the map debug overlay in the world, on the player's frames only (never in
                                                openpilot's; gta5_overlay.py, core.cpp DrawDebug; F7 toggles it): strips of
                                                each kind's own width on the game's ground (ground=0: at the map's heights),
                                                width/widths scale them (all, or by layer letter), dist m drawn, grow m from
                                                the camera they start widening to stay visible (0 off), thin=1 the old 1-px
                                                lines; off while recording unless force=1; debug off
  gta5_cmd.py gpsroute on [colour=21 max=100 radar=16 map=16 take=1]
                                                our route on the minimap and map as a custom GPS route, the map's waypoint held
                                                off it meanwhile (take=0 leaves it); gpsroute off
                                                the bridge keeps the last debug and gpsroute settings and sets them again each
                                                time the game connects (a core reload, a GTA restart); GTA5_DEBUG_OVERLAY and
                                                GTA5_GPSROUTE (e.g. "on layers=all") give them from the bridge's start
  gta5_cmd.py gtadirs <x> <y> <z>               GTA's own GPS directions from the car to a point (its next turn and the
                                                distance to it), printed from the state; gtadirs off stops asking
  gta5_cmd.py state [/tmp/gta5state.json]       save and print the plugin's next state
  expert options (gta5_expert.py): speed style ability aggr task ramp lead launch decel turn_speed arrive=gentle stop_before
  targets=smooth ahead_min ahead_max past retarget_every limits=0"""
import glob
import json
import os
import random
import socket
import sys
import time
import urllib.request
from pathlib import Path

from openpilot.tools.sim.bridge.gta5 import gta5_overlay, gta5_scene
from openpilot.tools.sim.bridge.gta5.gta5_expert import CONTROL, control_path
from openpilot.tools.sim.bridge.gta5.gta5_rx import DEBUG_PORT

TRIPS = os.getenv("GTA5_TRIPS", os.path.expanduser("~/gta5test/e2e/*.txt"))
MAP_VIEW = f"http://localhost:{os.getenv('GTA5_MAP_PORT', '8793')}"
SETUP_WAIT = 6.0  # s: the plugin places the car about 3.5 s after the setup command
SWAP_WAIT = 3.0  # s: a car swap waits for its model to load, well under a second for a car already streamed
STATE_FILE = "/tmp/gta5state.json"
# GENERATE_DIRECTIONS_TO_COORD's direction (alloc8or's native DB)
GTA_DIRECTIONS = {0: "announce", 1: "calculating", 2: "proceed", 3: "left", 4: "right", 5: "straight", 6: "sharp left",
                  7: "sharp right", 8: "recalculating"}


def parse(value: str):
  try:
    return float(value)
  except ValueError:
    return value


def options(args: list[str]) -> dict:
  out = {}
  for arg in args:
    key, _, value = arg.partition("=")
    out[key] = parse(value)
  return out


def send(*cmds: dict) -> None:
  with socket.create_connection(("127.0.0.1", DEBUG_PORT)) as s:
    s.sendall("".join(json.dumps(cmd) + "\n" for cmd in cmds).encode())


def find_trip(name: str) -> str:
  if ">" in name:
    return name
  for path in sorted(glob.glob(TRIPS)):
    with open(path) as f:
      for line in f:
        parts = line.split("#")[0].split()
        if len(parts) >= 2 and parts[0] == name and ">" in parts[1]:
          return parts[1]
  sys.exit(f"no trip {name} in {TRIPS}")


def write_control(cfg: dict) -> Path:
  path = control_path() or CONTROL
  tmp = path.with_suffix(".tmp")
  tmp.write_text(json.dumps(cfg) + "\n")
  tmp.replace(path)  # whole, as the bridge may read it at any moment
  return path


def expert(argv: list[str]) -> None:
  sub = argv[0] if argv else "status"
  if sub == "status":
    path = control_path() or CONTROL
    print(f"{path}: {path.read_text().strip() if path.exists() else 'none'}")
  elif sub == "off":
    write_control({"on": False})
    send({"type": "ai", "on": 0, "indicator": "off"})  # at once, and even when the bridge doesn't watch the file
  elif sub == "on":
    print(f"wrote {write_control({'on': True, **options(argv[1:])})}")
  elif sub == "route" and len(argv) > 1:
    spec = find_trip(argv[1])
    start, dest = spec.split(">")
    x, y, z, heading, *lane = (float(v) for v in start.split(","))
    dx, dy = (float(v) for v in dest.split(","))
    write_control({"on": False})
    send({"type": "ai", "on": 0, "indicator": "off"})
    setup = {"type": "setup", "x": x, "y": y, "z": z, "heading": heading, "fix": 1}
    if lane:
      setup["lane"] = lane[0]
    send(setup)
    print(f"placing the car at {x:.0f},{y:.0f}, heading {heading:.0f}; destination {dx:.0f},{dy:.0f}")
    time.sleep(SETUP_WAIT)
    req = urllib.request.Request(f"{MAP_VIEW}/destination", json.dumps({"x": dx, "y": dy}).encode(),
                                 {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=5).read()
    print(f"wrote {write_control({'on': True, 'need_route': True, 'dest': [dx, dy], **options(argv[2:])})}")
  else:
    sys.exit(__doc__)


def vehicle(argv: list[str]) -> None:
  if not argv or argv[0] == "list":
    print(f"model3 {gta5_scene.MODEL_3} (add-on, the bridge's car)")
    for c in gta5_scene.CARS:
      print(f"{c.name} {c.hash} {c.kind}")
    return
  opts = options(argv[1:])
  seed = opts.pop("seed", None)
  rng = random.Random(None if seed is None else int(seed))
  car = gta5_scene.pick_car(rng, float(opts.pop("model_share", 0))) if argv[0] == "random" else argv[0]
  cmd = {"type": "vehicle", "model": gta5_scene.model_for(car)}
  if opts.pop("colours", "random") == "random":
    cmd.update({k: v for k, v in gta5_scene.pick_colours(rng, car).items() if k != "group"})
  cmd.update(opts)
  print(json.dumps(cmd))
  send(cmd)


def mount(argv: list[str]) -> None:
  if not argv or argv[0] not in ("comma", "dashcam"):
    sys.exit(__doc__)
  opts = options(argv[1:])
  seed = opts.pop("seed", None)
  cmd = {"type": "mount", "mode": argv[0]}
  if "jitter" in opts:
    picked = gta5_scene.pick_mount(random.Random(None if seed is None else int(seed)), argv[0], bool(opts.pop("jitter")))
    cmd.update({k: v for k, v in picked.items() if k not in ("mode", "drop")})
  cmd.update(opts)
  print(json.dumps(cmd))
  send(cmd)


def randomise(argv: list[str]) -> None:
  seeded = bool(argv) and "=" not in argv[0]
  seed = int(argv[0]) if seeded else random.randrange(1 << 31)
  opts = options(argv[1:] if seeded else argv)
  choice = gta5_scene.pick(seed, model_share=float(opts.get("model_share", 0.6)), mount=str(opts.get("mount", "mixed")),
                           jitter=bool(opts.get("jitter", 1)), vehicle=bool(opts.get("vehicle", 1)), world=bool(opts.get("world", 1)),
                           traffic=bool(opts.get("traffic", 1)), comma_share=float(opts.get("comma_share", 0.5)))
  print("randomise " + json.dumps(choice), flush=True)
  if opts.get("dry"):
    return
  send(*gta5_scene.commands(choice))
  if "vehicle" in choice:
    time.sleep(float(opts.get("wait", SWAP_WAIT)))  # a setup sent during the swap would place the old car


def on_off(argv: list[str]) -> int:
  if not argv or argv[0] not in ("on", "off"):
    raise ValueError(f"expected on or off, not {' '.join(argv)!r}")
  return int(argv[0] == "on")


def debug_cmd(argv: list[str]) -> dict:
  cmd = {"type": "debug", "on": on_off(argv), **options(argv[1:])}
  if "layers" in cmd:
    names = str(cmd["layers"])
    letters = set(gta5_overlay.LAYERS.values())
    cmd["layers"] = "".join(letters) if names == "all" else \
      "".join(gta5_overlay.LAYERS.get(n, n if n in letters else "") for n in names.split(","))
  return cmd


def gpsroute_cmd(argv: list[str]) -> dict:
  return {"type": "gpsroute", "on": on_off(argv), **options(argv[1:])}


DISPLAY_ENV = {"GTA5_DEBUG_OVERLAY": debug_cmd, "GTA5_GPSROUTE": gpsroute_cmd}


def display_from_env(env=os.environ) -> list[dict]:
  """The debug overlay and GPS route the bridge sets whenever the game connects, from GTA5_DEBUG_OVERLAY and GTA5_GPSROUTE,
  each written as this script's arguments after debug or gpsroute ("on layers=all width=2", "off")."""
  out = []
  for name, build in DISPLAY_ENV.items():
    if env.get(name, "").strip():
      try:
        out.append(build(env[name].split()))
      except ValueError as e:
        print(f"gta5: ignored {name}: {e}", flush=True)
  return out


def display(build, argv: list[str]) -> None:
  try:
    cmd = build(argv)
  except ValueError:
    sys.exit(__doc__)
  print(json.dumps(cmd))
  send(cmd)


def debug(argv: list[str]) -> None:
  display(debug_cmd, argv)


def gpsroute(argv: list[str]) -> None:
  display(gpsroute_cmd, argv)


def read_state(path: str = STATE_FILE, wait: float = 2.0) -> dict | None:
  """The plugin's state with its next frame, through the bridge's receiver."""
  try:
    os.remove(path)
  except FileNotFoundError:
    pass
  send({"type": "state", "path": path})
  end = time.monotonic() + wait
  while time.monotonic() < end:
    if os.path.exists(path):
      time.sleep(0.05)  # written whole by then
      with open(path) as f:
        return json.load(f)
    time.sleep(0.05)
  return None


def state(argv: list[str]) -> None:
  got = read_state(argv[0] if argv else STATE_FILE)
  print(json.dumps(got, indent=1) if got is not None else "no state: is the bridge running and the game connected?")


def gtadirs(argv: list[str]) -> None:
  if argv and argv[0] == "off":
    send({"type": "gtadirs", "on": 0})
    return
  if len(argv) < 3:
    sys.exit(__doc__)
  x, y, z = (float(v) for v in argv[:3])
  send({"type": "gtadirs", "on": 1, "x": x, "y": y, "z": z})
  print(json.dumps(wait_directions(x, y)))


def wait_directions(x: float, y: float, wait: float = 5.0) -> dict | None:
  """GTA's directions to (x, y) from the state, once its route is worked out (directions 0 and 1 mean it isn't yet)."""
  end, got = time.monotonic() + wait, None
  time.sleep(0.5)
  while time.monotonic() < end:
    s = read_state() or {}
    d = s.get("gtaDirs")
    if d and abs(d["to"][0] - x) < 0.5 and abs(d["to"][1] - y) < 0.5:
      got = {**d, "name": GTA_DIRECTIONS.get(d.get("direction"), "?")}
      if d.get("direction") not in (0, 1):
        return got
    time.sleep(0.25)
  return got


def main(argv: list[str]) -> None:
  if not argv:
    print(__doc__)
    sys.exit(1)
  sub = {"expert": expert, "vehicle": vehicle, "mount": mount, "randomise": randomise, "debug": debug, "gpsroute": gpsroute,
         "gtadirs": gtadirs, "state": state}.get(argv[0])
  if sub is not None:
    sub(argv[1:])
    return
  cmd: dict = {"type": argv[0]}
  if argv[0] == "snap":
    cmd["path"] = argv[1] if len(argv) > 1 else "/tmp/gta5"
  elif argv[0] == "burst":
    cmd["path"] = argv[1] if len(argv) > 1 else "/tmp/gta5burst"
    cmd["count"] = int(argv[2]) if len(argv) > 2 else 40
  elif argv[0] == "ai" and len(argv) > 1 and argv[1] in ("on", "off"):
    cmd.update({"on": int(argv[1] == "on"), **options(argv[2:])})
  else:
    cmd.update(options(argv[1:]))
  send(cmd)


if __name__ == "__main__":
  main(sys.argv[1:])
