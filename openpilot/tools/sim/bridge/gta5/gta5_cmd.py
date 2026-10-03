#!/usr/bin/env python3
"""Sends a debug command to the GTA V plugin through the running bridge, e.g.
  gta5_cmd.py snap /tmp/gta5                    save the next road/wide frames as PNGs, and their state
  gta5_cmd.py burst /tmp/burst 40               save the next 40 frames' luma, quarter size, as JPEGs
  gta5_cmd.py setup x=2420 y=3000 z=46 model=sultan speed=20 hour=12 weather=EXTRASUNNY
                                                spawn a car and/or put it on the road nearest a point
  gta5_cmd.py world hour=12 weather=EXTRASUNNY freeze=1
                                                set the time and weather, and stop the clock
  gta5_cmd.py camera yaw=5 pitch=0              rotate the camera on its mount (degrees)"""
import json
import socket
import sys

from openpilot.tools.sim.bridge.gta5.gta5_rx import DEBUG_PORT


def parse(value: str):
  try:
    return float(value)
  except ValueError:
    return value


def main(argv: list[str]) -> None:
  if not argv:
    print(__doc__)
    sys.exit(1)
  cmd: dict = {"type": argv[0]}
  if argv[0] == "snap":
    cmd["path"] = argv[1] if len(argv) > 1 else "/tmp/gta5"
  elif argv[0] == "burst":
    cmd["path"] = argv[1] if len(argv) > 1 else "/tmp/gta5burst"
    cmd["count"] = int(argv[2]) if len(argv) > 2 else 40
  else:
    for arg in argv[1:]:
      key, _, value = arg.partition("=")
      cmd[key] = parse(value)
  with socket.create_connection(("127.0.0.1", DEBUG_PORT)) as s:
    s.sendall((json.dumps(cmd) + "\n").encode())


if __name__ == "__main__":
  main(sys.argv[1:])
