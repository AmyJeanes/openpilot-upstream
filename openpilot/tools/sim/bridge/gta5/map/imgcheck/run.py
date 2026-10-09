#!/usr/bin/env python3
"""Image-based check of the lane map against the game's own paint and kerbs, from top-down tiles of the game.

  run.py plan RUN [--area city|sample|all] [--classes roads|main] [--limit N] [--under]
                                                         tiles along the roads (RUN/plan.json); --under: under bridge
                                                         decks, the camera below the deck (a second pass)
  run.py capture RUN [--minutes M] [--desktop] [--png]    shoot the plan's tiles in the game (needs the bridge running
                                                         and the player in a car; resumable)
  run.py import-survey RUN [GLOB]                        use the map audit's survey shots (topcam poses in their
                                                         .shot.json) as tiles
  run.py analyse RUN [--procs 8] [--force]               check each tile against the map (RUN/analysis/)
  run.py report RUN [--top 150]                          spots across tiles, RUN/report/index.html, spots.json, heat maps
  run.py all RUN ...                                     analyse then report

RUN is a folder, by default /mnt/e/gta5_audit/imgcheck_<date>. The map is GTA5_MAP (~/gta5map_lanes), drawn from the
overlay's cached road marks for it. Run from the repository root with PYTHONPATH=. in the bridge's environment."""
import argparse
import glob
import json
import os
import sys
import time

from openpilot.tools.sim.bridge.gta5.map.imgcheck import report, tiles
from openpilot.tools.sim.bridge.gta5.map.imgcheck.frame import MapData

DEFAULT_RUN = f"/mnt/e/gta5_audit/imgcheck_{time.strftime('%Y%m%d')}"
SURVEY = "/mnt/e/gta5_audit/map_audit/survey_trips_*.shot.json"


def import_survey(run_dir: str, pattern: str) -> int:
  os.makedirs(os.path.join(run_dir, "tiles"), exist_ok=True)
  n = 0
  for f in sorted(glob.glob(pattern)):
    with open(f) as fh:
      d = json.load(fh)
    base = f[:-len(".shot.json")]
    image = next((base + e for e in (".jpg", ".png") if os.path.exists(base + e)), None)
    if image is None or not d.get("topcam"):
      continue
    name = os.path.basename(base)
    side = {"name": name, "source": "survey", "image": image, "topcam": d["topcam"], "zmap": d.get("z"), "size": [2560, 1440],
            "target": {"x": d.get("x"), "y": d.get("y"), "z": d.get("z"), "heading": d.get("heading")}}
    with open(os.path.join(run_dir, "tiles", name + ".json"), "w") as fh:
      json.dump(side, fh)
    n += 1
  return n


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("command", choices=["plan", "capture", "import-survey", "analyse", "report", "all"])
  ap.add_argument("run", nargs="?", default=DEFAULT_RUN)
  ap.add_argument("glob", nargs="?", default=SURVEY)
  ap.add_argument("--area", default="city", choices=["city", "sample", "all"])
  ap.add_argument("--classes", default="roads", choices=list(tiles.CLASSES))
  ap.add_argument("--step", type=float, default=tiles.STEP)
  ap.add_argument("--limit", type=int)
  ap.add_argument("--under", action="store_true", help="plan: tiles under bridge decks (the second pass)")
  ap.add_argument("--minutes", type=float)
  ap.add_argument("--desktop", action="store_true", help="screen captures instead of the plugin's grab")
  ap.add_argument("--png", action="store_true", help="keep tiles as PNG (default JPEG, quality 94)")
  ap.add_argument("--procs", type=int, default=8)
  ap.add_argument("--force", action="store_true")
  ap.add_argument("--top", type=int, default=report.TOP)
  a = ap.parse_args(argv)
  os.makedirs(a.run, exist_ok=True)
  if a.command == "plan":
    md = MapData()
    plan = tiles.plan(md, a.area, a.classes, a.step, a.limit, a.under)
    est = tiles.estimate(plan)
    tiles.save(plan, os.path.join(a.run, "plan.json"), {"area": a.area, "classes": a.classes, "step": a.step, "under": a.under, "height": tiles.HEIGHT,
                                                       "fov": tiles.FOV, "map_hash": md.map_hash, "estimate": est})
    print(json.dumps(est))
  elif a.command == "capture":
    from openpilot.tools.sim.bridge.gta5.map.imgcheck import capture
    plan, meta = tiles.load(os.path.join(a.run, "plan.json"))
    capture.run(a.run, plan, a.minutes, a.desktop, None if a.png else 94)
  elif a.command == "import-survey":
    print(f"{import_survey(a.run, a.glob)} survey shots as tiles in {a.run}/tiles")
  if a.command in ("analyse", "all"):
    report.analyse(a.run, a.procs, a.force)
  if a.command in ("report", "all"):
    report.write(a.run, a.procs, a.top)


if __name__ == "__main__":
  sys.exit(main())
