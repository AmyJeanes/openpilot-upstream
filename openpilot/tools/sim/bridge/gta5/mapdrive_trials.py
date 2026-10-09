#!/usr/bin/env python3
"""In-game trials of the map driver (gta5_mapdrive.py) through a running bridge with GTA5_EXPERT=1: each trip (an e2e
trip name, or 'x,y,z,heading[,lane]>dx,dy') is driven with gta5_cmd.py expert route ... driver=map, waited on until
expert mode stops (arrived or aborted) or the timeout, and summed up from the expert log: one JSON line per trip.

  python -m openpilot.tools.sim.bridge.gta5.mapdrive_trials SL1 SR3 FX1 --out /tmp/mapdrive_trials.jsonl \\
    [--log ~/gta5test/expert.jsonl] [--timeout 240] [--seed 1] [--mapdrive '{"preset":"normal"}']"""
import argparse
import json
import os
import time

from openpilot.tools.sim.bridge.gta5 import gta5_cmd
from openpilot.tools.sim.bridge.gta5.gta5_expert import CONTROL, control_path


def summary(lines: list[dict]) -> dict:
  """A trip's outcome from its expert log rows."""
  ev = [r for r in lines if "event" in r]
  rows = [r for r in lines if "map" in r]
  phases = [r.get("phase") for r in ev if r["event"] == "mapdrive" and r.get("phase")]
  stop = next((r.get("why") for r in ev if r["event"] == "stop"), None)
  anomalies: dict[str, int] = {}
  for r in ev:
    if r["event"] == "anomaly":
      anomalies[r["kind"]] = anomalies.get(r["kind"], 0) + 1
  devs = [r["map"]["dev"] for r in rows if r["map"].get("dev") is not None and r["map"].get("phase") not in ("abort", "", "wait")]
  plan = next((r.get("plan") for r in ev if r["event"] == "mapdrive" and r.get("plan")), None) or {}
  return {"stop": stop, "phases": phases, "arrived": stop == "arrived",
          "aborts": [r.get("why") for r in ev if r["event"] == "mapdrive" and r.get("phase") == "abort"],
          "dev_max": max(devs, default=None), "dev_p95": sorted(devs)[int(0.95 * (len(devs) - 1))] if devs else None,
          "collisions": max((r.get("collisions") or 0 for r in rows), default=0), "anomalies": anomalies,
          "anomaly_places": [[r["kind"], r.get("pos")] for r in ev if r["event"] == "anomaly"][:40],
          "lights": sum(r["event"] == "light_unknown" for r in ev), "stops": sum(r["event"] == "stopped" for r in ev),
          "slot_check": plan.get("slot_check"), "keys": plan.get("keys"), "rows": len(rows)}


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("trips", nargs="+")
  ap.add_argument("--out", default="/tmp/mapdrive_trials.jsonl")
  ap.add_argument("--log", default=os.path.expanduser("~/gta5test/expert.jsonl"), help="the bridge's expert log")
  ap.add_argument("--timeout", type=float, default=240.0, help="s per trip")
  ap.add_argument("--seed", type=int, default=1, help="the first trip's style seed (one more each trip)")
  ap.add_argument("--mapdrive", default="{}", help="more of the map driver's settings, JSON")
  ap.add_argument("--driver", default="map", choices=["map", "map+ai", "ai"])
  ap.add_argument("--settings", default="", help="more expert k=v settings, e.g. 'speed=14'")
  args = ap.parse_args()
  control = control_path() or CONTROL
  for n, trip in enumerate(args.trips):
    cfg = {"seed": args.seed + n, **json.loads(args.mapdrive)}
    start = os.path.getsize(args.log) if os.path.exists(args.log) else 0
    t0 = time.monotonic()
    gta5_cmd.expert(["route", trip, f"driver={args.driver}", "mapdrive=" + json.dumps(cfg), *args.settings.split()])
    time.sleep(2.0)
    timed_out = True
    while time.monotonic() - t0 < args.timeout:
      try:
        if not json.loads(control.read_text()).get("on"):
          timed_out = False
          break
      except (OSError, ValueError):
        pass
      time.sleep(1.0)
    if timed_out:
      gta5_cmd.expert(["off"])
      time.sleep(2.0)
    lines = []
    if os.path.exists(args.log):
      with open(args.log) as f:
        f.seek(start)
        for line in f:
          try:
            lines.append(json.loads(line))
          except ValueError:
            pass
    out = {"trip": trip, "seed": cfg["seed"], "driver": args.driver, "secs": round(time.monotonic() - t0, 1), "timed_out": timed_out,
           **summary(lines)}
    print(json.dumps({k: v for k, v in out.items() if k not in ("keys", "anomaly_places")}), flush=True)
    with open(args.out, "a") as f:
      f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
  main()
