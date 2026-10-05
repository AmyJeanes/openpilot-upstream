#!/usr/bin/env python3
"""Compares GTA's own GPS directions (GENERATE_DIRECTIONS_TO_COORD, as `gta5_cmd.py gtadirs` asks) with our router's at
e2e trips' starts (~/gta5test/e2e/*.txt): GTA's next direction and distance to it, Valhalla's first maneuver past the
start, and the first turn nav finds on the route.

Live only: for each trip it places the car at the start (the setup command), so run it with the game, the bridge
(GTA5_MAP and GTA5_ROUTER set) and Valhalla up, and nothing else driving (expert off, openpilot disengaged):
  python compare_dirs.py L7 R2 X1          # or --all
--dry lists the trips and the commands it would send, and touches nothing."""
import argparse
import glob
import json
import os
import time

import numpy as np

from openpilot.tools.sim.bridge.gta5 import gta5_cmd
from openpilot.tools.sim.bridge.gta5.gta5_nav import MIN_AHEAD_MAP, find_turn

# Valhalla's maneuver types: side and name, for the turns
VALHALLA = {9: ("right", "slight right"), 10: ("right", "right"), 11: ("right", "sharp right"), 14: ("left", "sharp left"),
            15: ("left", "left"), 16: ("left", "slight left"), 18: ("right", "ramp right"), 19: ("left", "ramp left"),
            20: ("right", "exit right"), 21: ("left", "exit left"), 22: ("straight", "stay straight"), 23: ("right", "stay right"),
            24: ("left", "stay left"), 12: ("right", "u-turn right"), 13: ("left", "u-turn left"), 17: ("straight", "ramp straight")}
GTA_SIDE = {3: "left", 4: "right", 5: "straight", 6: "left", 7: "right"}
AGREE_M = 30.0  # m apart, and the same side


def trips(names: list[str], every: bool) -> list[tuple[str, str]]:
  found = {}
  for path in sorted(glob.glob(gta5_cmd.TRIPS)):
    with open(path) as f:
      for line in f:
        parts = line.split("#")[0].split()
        if len(parts) >= 2 and ">" in parts[1]:
          found.setdefault(parts[0], parts[1])
  return [(n, found[n]) for n in (sorted(found) if every else names) if n in found]


def parse(spec: str):
  start, dest = spec.split(">")
  x, y, z, heading, *lane = (float(v) for v in start.split(","))
  dx, dy = (float(v) for v in dest.split(","))
  return (x, y, z, heading, lane[0] if lane else None), (dx, dy)


def valhalla(router, paths, start, dest) -> dict:
  x, y, z, heading, _ = start
  route = router.route(np.array([x, y]), (-heading) % 360, np.array(dest), z + 0.6)
  first = None
  for m in route.maneuvers:
    if m.get("type") in VALHALLA:
      side, name = VALHALLA[m["type"]]
      first = {"side": side, "type": name, "dist": round(float(route.along[m["begin_shape_index"]]), 1),
               "instruction": m.get("instruction", "")}
      break
  turn = find_turn(route.ahead(1000.0, 5.0), MIN_AHEAD_MAP)
  return {"maneuver": first, "nav_turn": None if turn is None else {"side": turn.side, "dist": round(turn.dist, 1)}}


def dest_z(paths, dest, fallback: float) -> float:
  if paths is None:
    return fallback
  i = int(np.argmin(np.hypot(*(paths.xy - np.array(dest)).T)))
  return float(paths.z[i])


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("trips", nargs="*")
  ap.add_argument("--all", action="store_true")
  ap.add_argument("--dry", action="store_true")
  ap.add_argument("--settle", type=float, default=gta5_cmd.SETUP_WAIT, help="s after placing the car")
  args = ap.parse_args()
  chosen = trips(args.trips, args.all)
  if not chosen:
    raise SystemExit(f"no such trips in {gta5_cmd.TRIPS}")
  map_dir = os.path.expanduser(os.getenv("GTA5_MAP", "~/gta5map"))
  router_url = os.getenv("GTA5_ROUTER", "http://localhost:8002")
  paths = router = None
  if not args.dry:
    from openpilot.tools.sim.bridge.gta5.map.paths import Paths
    from openpilot.tools.sim.bridge.gta5.map.router import Router
    if os.path.exists(os.path.join(map_dir, "paths.jsonl")):
      paths = Paths(os.path.join(map_dir, "paths.jsonl"))
      paths.index()
    router = Router(router_url, paths=paths)
  rows = []
  for name, spec in chosen:
    start, dest = parse(spec)
    x, y, z, heading, lane = start
    setup = {"type": "setup", "x": x, "y": y, "z": z, "heading": heading, "fix": 1, **({"lane": lane} if lane is not None else {})}
    dz = dest_z(paths, dest, z)
    if args.dry:
      print(f"{name}: send {json.dumps(setup)}, wait {args.settle:.0f} s, gtadirs {dest[0]:.1f} {dest[1]:.1f} <z>; route on {router_url}")
      continue
    gta5_cmd.send({"type": "ai", "on": 0}, setup)
    time.sleep(args.settle)
    gta5_cmd.send({"type": "gtadirs", "on": 1, "x": dest[0], "y": dest[1], "z": dz})
    gta = gta5_cmd.wait_directions(dest[0], dest[1], wait=8.0)
    try:
      ours = valhalla(router, paths, start, dest)
    except (OSError, ValueError, KeyError) as e:
      ours = {"error": str(e)}
    m = ours.get("maneuver")
    g_side = GTA_SIDE.get(gta.get("direction")) if gta else None
    agree = bool(m and g_side == m["side"] and abs(gta["dist"] - m["dist"]) < AGREE_M)
    row = {"trip": name, "gta": gta, **ours, "agree": agree}
    rows.append(row)
    gtxt = f"{gta['name']} in {gta['dist']:.0f} m" if gta else "none"
    mtxt = f"{m['type']} in {m['dist']:.0f} m ({m['instruction']})" if m else ours.get("error", "none")
    print(f"{name}: GTA {gtxt} | valhalla {mtxt} | nav {ours.get('nav_turn')} | {'agree' if agree else 'DIFFER'}", flush=True)
  if rows:
    gta5_cmd.send({"type": "gtadirs", "on": 0})
    print(f"{sum(r['agree'] for r in rows)}/{len(rows)} agree")
    out = "/tmp/gta5_compare_dirs.json"
    with open(out, "w") as f:
      json.dump(rows, f, indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
  main()
