#!/usr/bin/env python3
"""Sends a debug command to the GTA V plugin through the running bridge, e.g.
  gta5_cmd.py snap /tmp/gta5                    save the next road/wide frames as PNGs, and their state
  gta5_cmd.py burst /tmp/burst 40               save the next 40 frames' luma, quarter size, as JPEGs
  gta5_cmd.py setup x=2420 y=3000 z=46 model=sultan speed=20 hour=12 weather=EXTRASUNNY
                                                spawn a car and/or put it on the road nearest a point
  gta5_cmd.py world hour=12 weather=EXTRASUNNY freeze=1
                                                set the time and weather, and stop the clock
  gta5_cmd.py camera yaw=5 pitch=0              rotate the camera on its mount (degrees)
  gta5_cmd.py ai on speed=12 style=1076369579   the game's AI drives the car: to the map's waypoint, else wandering
  gta5_cmd.py ai x=100 y=-200 z=30              ... to a point; ai off gives the car back
  gta5_cmd.py expert on speed=12                expert mode (gta5_expert.py; the bridge runs with GTA5_EXPERT=1): the AI
                                                drives our route to the destination set; off stops it
  gta5_cmd.py expert route L1 [speed=12 ...]    put the car at an e2e trip's start (a name in ~/gta5test/e2e/*.txt, or
                                                'x,y,z,heading[,lane]>dx,dy'), set its destination and drive it
  gta5_cmd.py expert status                     the control file
  expert options (gta5_expert.py): speed style ability aggr task ramp lead launch decel turn_speed arrive=gentle stop_before
  targets=smooth ahead_min ahead_max past retarget_every limits=0"""
import glob
import json
import os
import socket
import sys
import time
import urllib.request
from pathlib import Path

from openpilot.tools.sim.bridge.gta5.gta5_expert import CONTROL, control_path
from openpilot.tools.sim.bridge.gta5.gta5_rx import DEBUG_PORT

TRIPS = os.getenv("GTA5_TRIPS", os.path.expanduser("~/gta5test/e2e/*.txt"))
MAP_VIEW = f"http://localhost:{os.getenv('GTA5_MAP_PORT', '8793')}"
SETUP_WAIT = 6.0  # s: the plugin places the car about 3.5 s after the setup command


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


def send(cmd: dict) -> None:
  with socket.create_connection(("127.0.0.1", DEBUG_PORT)) as s:
    s.sendall((json.dumps(cmd) + "\n").encode())


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


def main(argv: list[str]) -> None:
  if not argv:
    print(__doc__)
    sys.exit(1)
  if argv[0] == "expert":
    expert(argv[1:])
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
