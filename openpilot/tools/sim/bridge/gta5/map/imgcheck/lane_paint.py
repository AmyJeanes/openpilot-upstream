#!/usr/bin/env python3
"""Whether captured game images show a line on the lane boundaries the game files paint none on.

paint_survey.bare_lane_lines finds boundaries between lanes one way that the files show bare (GTA's two lanes in one
painted lane, Senora Fwy's), but the files miss some freeways' lane lines too (the Olympic Fwy's), so ynd_to_osm draws
no line only where captured top-down images (imgcheck runs' tiles) agree: along each candidate, the share of the road
judged (paint masks within HALF m either side) that shows paint, over every tile within NEAR m. Writes JSON
{"bare": {"<a>|<b>": [m right of a -> b]}, "painted": {...}, "rows": [...]}, for ynd_to_osm --lines-bare.

  python -m openpilot.tools.sim.bridge.gta5.map.imgcheck.lane_paint map/gta5.osm.pbf out.json run_dir [run_dir...] \\
    --survey survey_gf.jsonl --survey-lines polylines.jsonl [--procs 3]
GTA5_MAP / GTA5_OVERLAY_CACHE name the map the runs' tiles are judged against, as for imgcheck.run.
"""
import argparse
import glob
import json
import os
from collections import Counter
from multiprocessing import Pool

import numpy as np

from openpilot.tools.sim.bridge.gta5.map import paint_survey
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import FORWARD, OsmLanes, WayLanes, offset_line

NEAR = 45.0  # m from a tile's camera a boundary is looked for in its image
STEP = 0.25  # m between samples along a boundary
HALF = 0.35  # m either side of a boundary paint counts as its line
JUDGED_MIN = 8.0  # m of a boundary the images must judge to say anything
PAINTED = 0.12  # share of the judged boundary with paint on it: a dashed line's is 0.25 or more


def key(n):
  return ((n - 1) // 65536, (n - 1) % 65536)


def candidates(pbf, survey):
  """The bare boundaries in the files: [{id, a, b, offset (m right of a -> b), name, hw, poly}]."""
  osm, out = OsmLanes.load(pbf, to_game), []
  for wid, (tags, refs) in osm.ways.items():
    if "highway" not in tags or tags.get("lane_markings") == "no":
      continue
    sec = WayLanes.from_tags(tags).section(FORWARD)
    a, b = key(refs[0]), key(refs[-1])
    found = paint_survey.bare_lane_lines(paint_survey.along(survey, a, b) or [], [(s.left, s.right, s.heading) for s in sec])
    pts = osm.way_points(wid)
    for k, heading in (("forward", 1), ("backward", -1)):
      if k not in found:
        continue
      bounds = [r for _, r in sorted((s.left, s.right) for s in sec if s.heading == heading)[:-1]]
      for v, bare in zip(bounds, found[k] if heading == 1 else found[k][::-1], strict=True):
        if bare:
          out.append({"id": wid, "a": a, "b": b, "offset": round(v, 2), "name": tags.get("name"), "hw": tags["highway"],
                      "poly": offset_line(pts, v).tolist()})
  return out


G = {}


def init():
  from openpilot.tools.sim.bridge.gta5.map.imgcheck import report
  from openpilot.tools.sim.bridge.gta5.map.imgcheck.frame import MapData
  report._md = MapData()
  G["report"] = report


def judge(job):
  """[(row, tile, m judged, m painted)] for the boundaries in one tile."""
  from openpilot.tools.sim.bridge.gta5.map.imgcheck import compare
  from openpilot.tools.sim.bridge.gta5.map.imgcheck.frame import load_image
  from openpilot.tools.sim.bridge.gta5.map.imgcheck.paint import max_filter
  report = G["report"]
  run, tile, polys = job
  side = json.load(open(os.path.join(run, "tiles", tile + ".json")))
  img = load_image(report.image_path(run, side))
  tc = compare.check(img, report.tile_camera({**side, "size": [img.shape[1], img.shape[0]]}), report._md)
  paint = tc.paint.white | tc.paint.unsure | tc.paint.yellow
  near = max_filter(paint.astype(np.float32), tc.px(2 * HALF) | 1) > 0
  out = []
  for i, poly in polys:
    p = np.array(poly, float)
    _, _, _, iu, iv, judged = tc.samples(np.column_stack([p, np.full(len(p), tc.map.level)]), STEP)
    judged &= tc.road[iv, iu]
    out.append((i, tile, float(judged.sum() * STEP), float((judged & near[iv, iu]).sum() * STEP)))
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("pbf")
  ap.add_argument("out")
  ap.add_argument("runs", nargs="+", help="imgcheck run dirs (analysed: their analysis/ says which tiles loaded)")
  ap.add_argument("--survey", nargs="+", required=True)
  ap.add_argument("--survey-lines")
  ap.add_argument("--procs", type=int, default=3)
  args = ap.parse_args()
  lines = paint_survey.line_kinds(args.survey_lines) if args.survey_lines else None
  rows = candidates(args.pbf, paint_survey.load(args.survey, lines))
  cams = []
  for run in args.runs:
    for f in sorted(glob.glob(f"{run}/analysis/*.json")):
      a = json.load(open(f))
      if not (a["summary"].get("unloaded") or a["summary"].get("covered")):
        cams.append((a["camera"]["x"], a["camera"]["y"], run, a["name"]))
  xy = np.array([c[:2] for c in cams]).reshape(-1, 2)
  jobs = {}
  for i, r in enumerate(rows):
    p = np.array(r["poly"])
    for k in np.nonzero(np.hypot(*(xy[:, None] - p[None]).transpose(2, 0, 1)).min(1) <= NEAR)[0]:
      jobs.setdefault(cams[k][2:], []).append((i, r["poly"]))
  print(f"lane_paint: {len(rows)} boundaries the files paint no line on, {len(jobs)} tiles", flush=True)
  with Pool(args.procs, initializer=init) as pool:
    for res in pool.imap_unordered(judge, [(run, tile, polys) for (run, tile), polys in jobs.items()], chunksize=4):
      for i, tile, judged, painted in res:
        r = rows[i]
        r["judged"], r["painted"] = r.get("judged", 0.0) + judged, r.get("painted", 0.0) + painted
        if judged:
          r.setdefault("tiles", []).append([tile, round(judged, 1), round(painted, 1)])
  out = {"bare": {}, "painted": {}}
  for r in rows:
    judged = r.get("judged") or 0.0
    r["share"] = round(r.get("painted", 0.0) / judged, 3) if judged else None
    r["verdict"] = "unseen" if judged < JUDGED_MIN else "painted" if r["share"] >= PAINTED else "bare"
    if r["verdict"] != "unseen":
      out[r["verdict"]].setdefault(f"{r['a'][0]}:{r['a'][1]}|{r['b'][0]}:{r['b'][1]}", []).append(r["offset"])
    del r["poly"]
  json.dump({**out, "rows": rows}, open(args.out, "w"), indent=0)
  print("lane_paint:", dict(Counter(r["verdict"] for r in rows)))


if __name__ == "__main__":
  main()
