#!/usr/bin/env python3
"""Whether captured game images show a line on the lane boundaries the game files paint none on.

paint_survey.bare_lane_lines finds boundaries between lanes one way that the files show bare (GTA's two lanes in one
painted lane, Senora Fwy's), but the files miss some freeways' lane lines too (the Olympic Fwy's), so ynd_to_osm draws
no line only where captured top-down images (imgcheck runs' tiles) agree: along each candidate, the share of the road
judged (paint masks within HALF m either side) that shows paint, over every tile within NEAR m.

A bare stretch runs on past the ways the files read bare (some read neither edge, some carry decals the game doesn't
show), so from each end of a bare boundary the images follow it onto the ways joined end to end there with as many
lanes that way: one joins the run where no image shows a line (paint DASH m or more along it) from that end for CARRY
of its length (judging SEEN of it), and the run follows on past it where they show none on it at all, so the first
painted dash stops it.

Writes JSON {"bare": {"<a>|<b>": [m right of a -> b]}, "painted": {...}, "carried": {...}, "rows": [...]}, for
ynd_to_osm --lines-bare ("carried": the boundaries a bare run carries on onto).

  python -m openpilot.tools.sim.bridge.gta5.map.imgcheck.lane_paint map/gta5.osm.pbf out.json run_dir [run_dir...] \\
    --survey survey_gf.jsonl --survey-lines polylines.jsonl [--procs 3]
GTA5_MAP / GTA5_OVERLAY_CACHE name the map the runs' tiles are judged against, as for imgcheck.run.
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict
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
CARRY = 0.5  # share of a joined way, from the bare run's end, the images must show bare for the run to carry onto it
SEEN = 0.6  # share of that way the images must judge
DASH = 1.5  # m of paint along a boundary that is a line on it, not a seam or patch edge crossing it


def key(n):
  return ((n - 1) // 65536, (n - 1) % 65536)


def lane_bounds(tags) -> dict[int, list[float]]:
  """{1 forward / -1 backward: the boundaries between that direction's lanes, m right of the way's line, left to right
  as seen travelling}, directions with two lanes or more."""
  sec = WayLanes.from_tags(tags).section(FORWARD)
  out = {}
  for heading in (1, -1):
    bounds = [r for _, r in sorted((s.left, s.right) for s in sec if s.heading == heading)[:-1]]
    if bounds:
      out[heading] = bounds if heading == 1 else bounds[::-1]
  return out


def candidates(osm, survey):
  """The bare boundaries in the files: [{id, a, b, offset (m right of a -> b), heading, k (left to right as seen
  travelling), name, hw, poly}]."""
  out = []
  for wid, (tags, refs) in osm.ways.items():
    if "highway" not in tags or tags.get("lane_markings") == "no":
      continue
    sec = WayLanes.from_tags(tags).section(FORWARD)
    a, b = key(refs[0]), key(refs[-1])
    found = paint_survey.bare_lane_lines(paint_survey.along(survey, a, b) or [], [(s.left, s.right, s.heading) for s in sec])
    pts = osm.way_points(wid)
    for name, heading in (("forward", 1), ("backward", -1)):
      for k, (v, bare) in enumerate(zip(lane_bounds(tags).get(heading, []), found.get(name, []), strict=False)):
        if bare:
          out.append({"id": wid, "a": a, "b": b, "offset": round(v, 2), "heading": heading, "k": k, "name": tags.get("name"),
                      "hw": tags["highway"], "poly": offset_line(pts, v).tolist()})
  return out


def joined(osm, ends, wid, heading, k):
  """The boundaries carrying on boundary k of `heading` on way wid past each of its ends: [(way, heading, k, the node
  it carries on from)] on the ways joined end to end there with as many lanes that way."""
  tags, refs = osm.ways[wid]
  count = len(lane_bounds(tags)[heading])
  out = []
  for node, arriving in ((refs[-1], heading == 1), (refs[0], heading == -1)):
    for o in ends[node]:
      orefs = osm.ways[o][1]
      if o == wid or orefs[0] == orefs[-1]:
        continue
      leaving = 1 if orefs[0] == node else -1  # o's direction away from the node
      h = leaving if arriving else -leaving
      if len(lane_bounds(osm.ways[o][0]).get(h, [])) == count:
        out.append((o, h, k, node))
  return out


def lines_along(painted: np.ndarray) -> np.ndarray:
  """The samples in runs of paint DASH m or longer."""
  out = np.zeros(len(painted), bool)
  edges = np.flatnonzero(np.diff(np.concatenate(([0], painted.astype(np.int8), [0]))))
  for a, b in zip(edges[::2], edges[1::2], strict=True):
    if (b - a) * STEP >= DASH:
      out[a:b] = True
  return out


def carries(judged: np.ndarray, painted: np.ndarray) -> tuple[bool, bool]:
  """(whether a bare run carries onto a boundary, whether on past it) from its samples from the run's end: judged by
  the images, and with a line along them (lines_along, in any one image: tiles' seams don't quite register)."""
  n = len(judged)
  first = int(np.argmax(painted)) if painted.any() else n
  ok = n > 0 and first >= CARRY * n and judged[:first].sum() >= SEEN * n
  return bool(ok), bool(ok and first == n)


G = {}


def init():
  from openpilot.tools.sim.bridge.gta5.map.imgcheck import report
  from openpilot.tools.sim.bridge.gta5.map.imgcheck.frame import MapData
  report._md = MapData()
  G["report"] = report


def judge(job):
  """[(row, tile, judged, painted)] for the boundaries in one tile: the samples the image judges and those of them with
  paint within HALF m."""
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
    out.append((i, tile, judged, judged & near[iv, iu]))
  return out


def look(pool, cams, xy, polys):
  """{i: [(tile, judged, painted)]} for polylines {i: poly}, over every tile within NEAR m."""
  jobs = {}
  for i, poly in polys.items():
    p = np.array(poly)
    for k in np.nonzero(np.hypot(*(xy[:, None] - p[None]).transpose(2, 0, 1)).min(1) <= NEAR)[0]:
      jobs.setdefault(cams[k][2:], []).append((i, poly))
  out = defaultdict(list)
  for res in pool.imap_unordered(judge, [(run, tile, ps) for (run, tile), ps in jobs.items()], chunksize=4):
    for i, tile, judged, painted in res:
      out[i].append((tile, judged, painted))
  return out, len(jobs)


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
  osm = OsmLanes.load(args.pbf, to_game)
  rows = candidates(osm, paint_survey.load(args.survey, lines))
  cams = []
  for run in args.runs:
    for f in sorted(glob.glob(f"{run}/analysis/*.json")):
      a = json.load(open(f))
      if not (a["summary"].get("unloaded") or a["summary"].get("covered")):
        cams.append((a["camera"]["x"], a["camera"]["y"], run, a["name"]))
  xy = np.array([c[:2] for c in cams]).reshape(-1, 2)
  out = {"bare": {}, "painted": {}, "carried": {}}

  def put(kind, r):
    out[kind].setdefault(f"{r['a'][0]}:{r['a'][1]}|{r['b'][0]}:{r['b'][1]}", []).append(r["offset"])

  with Pool(args.procs, initializer=init) as pool:
    seen, tiles = look(pool, cams, xy, {i: r["poly"] for i, r in enumerate(rows)})
    print(f"lane_paint: {len(rows)} boundaries the files paint no line on, {tiles} tiles", flush=True)
    for i, r in enumerate(rows):
      r["judged"] = sum(float(j.sum() * STEP) for _, j, _ in seen[i])
      r["painted"] = sum(float(p.sum() * STEP) for _, _, p in seen[i])
      r["tiles"] = [[t, round(float(j.sum() * STEP), 1), round(float(p.sum() * STEP), 1)] for t, j, p in seen[i] if j.any()]
      r["share"] = round(r["painted"] / r["judged"], 3) if r["judged"] else None
      r["verdict"] = "unseen" if r["judged"] < JUDGED_MIN else "painted" if r["share"] >= PAINTED else "bare"
      if r["verdict"] != "unseen":
        put(r["verdict"], r)
    ends = defaultdict(list)
    for wid, (_, refs) in osm.ways.items():
      ends[refs[0]].append(wid)
      ends[refs[-1]].append(wid)
    done = {(r["id"], r["heading"], r["k"]) for r in rows if r["verdict"] == "bare"}
    front = sorted(done)
    while front:
      nxt = {}
      for w, h, k in front:
        for o, oh, ok, node in joined(osm, ends, w, h, k):
          if (o, oh, ok) not in done:
            nxt.setdefault((o, oh, ok), (node, w))
      done |= set(nxt)
      polys, info = {}, {}
      for i, ((o, h, k), (node, w)) in enumerate(sorted(nxt.items())):
        tags, refs = osm.ways[o]
        v = lane_bounds(tags)[h][k]
        polys[i] = offset_line(osm.way_points(o), v).tolist()
        info[i] = {"id": o, "a": key(refs[0]), "b": key(refs[-1]), "offset": round(v, 2), "heading": h, "k": k,
                   "name": tags.get("name"), "hw": tags.get("highway"), "from": w, "flip": node == refs[-1]}
      seen, _ = look(pool, cams, xy, polys) if polys else ({}, 0)
      front = []
      for i, r in info.items():
        n = max((len(j) for _, j, _ in seen.get(i, [])), default=0)
        judged, painted = np.zeros(n, bool), np.zeros(n, bool)
        for _, j, p in seen.get(i, []):
          judged |= j
          painted |= lines_along(p)
        if r.pop("flip"):
          judged, painted = judged[::-1], painted[::-1]
        ok, on = carries(judged, painted)
        r.update({"length": round(n * STEP, 1), "seen_m": round(float(judged.sum() * STEP), 1),
                  "bare_m": round((int(np.argmax(painted)) if painted.any() else n) * STEP, 1),
                  "painted_m": round(float(painted.sum() * STEP), 1), "verdict": "carried" if ok else "stops"})
        rows.append(r)
        if ok:
          put("carried", r)
        if on:
          front.append((r["id"], r["heading"], r["k"]))
      print(f"lane_paint: {len(info)} joined boundaries, {sum(r['verdict'] == 'carried' for r in info.values())} carried on onto",
            flush=True)
  for r in rows:
    r.pop("poly", None)
  json.dump({**out, "rows": rows}, open(args.out, "w"), indent=0)
  print("lane_paint:", dict(Counter(r["verdict"] for r in rows)))


if __name__ == "__main__":
  main()
