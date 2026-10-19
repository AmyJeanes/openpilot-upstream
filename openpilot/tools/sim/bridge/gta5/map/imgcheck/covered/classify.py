"""classify.py <dir>: the missing covered nodes (covered.json) by what's over them in the albedo (1 m around): foliage
(green) or not; groups of linked non-foliage nodes (path links) -> groups.json, biggest first."""
import collections, json, os, sys
import numpy as np
from PIL import Image

D, F = sys.argv[1], "/mnt/e/gta5_ortho_20261018/full/tiles"
cov = [c for c in json.load(open(f"{D}/covered.json")) if not c["under_shot"]]
by_cell = collections.defaultdict(list)
for c in cov:
  by_cell[tuple(c["cell"])].append(c)
for (i, j), cs in by_cell.items():
  s = json.load(open(f"{F}/o{i:03d}_{j:03d}.json"))
  im = np.asarray(Image.open(f"{F}/o{i:03d}_{j:03d}.png").convert("RGB"), dtype=np.float32)
  (a, _, c0), (_, e, f) = s["world_to_px"]
  for c in cs:
    u, v = int(a * c["x"] + c0), int(e * c["y"] + f)
    w = im[max(v - 15, 0):v + 16, max(u - 15, 0):u + 16].reshape(-1, 3)
    r, g, b = w[:, 0], w[:, 1], w[:, 2]
    c["green"] = round(float(np.mean((g > r * 1.05) & (g > b * 1.1) & (w.max(1) > 25))), 2)
    c["rgb"] = [int(x) for x in w.mean(0)]
key = lambda c: (c["a"], c["i"])
solid = {key(c): c for c in cov if c["green"] < 0.5}
links = collections.defaultdict(set)
for l in open(os.path.expanduser("~/gta5map_lanes/paths.jsonl")):
  if '"t":"l"' not in l:
    continue
  r = json.loads(l)
  p, q = (r["a"], r["i"]), (r["ta"], r["ti"])
  if p in solid and q in solid:
    links[p].add(q), links[q].add(p)
seen, groups = set(), []
for k in solid:
  if k in seen:
    continue
  stack, g = [k], []
  seen.add(k)
  while stack:
    n = stack.pop(); g.append(solid[n])
    for m in links[n]:
      if m not in seen:
        seen.add(m), stack.append(m)
  xs, ys = [c["x"] for c in g], [c["y"] for c in g]
  groups.append({"nodes": len(g), "x": round(float(np.mean(xs)), 1), "y": round(float(np.mean(ys)), 1),
                 "span_m": round(float(np.hypot(max(xs) - min(xs), max(ys) - min(ys))), 1),
                 "cover_med": round(float(np.median([c["top"] - c["z"] for c in g])), 1), "z": round(float(np.median([c["z"] for c in g])), 1),
                 "cells": sorted({tuple(c["cell"]) for c in g}), "pts": [[c["x"], c["y"], c["z"], c["top"]] for c in g]})
groups.sort(key=lambda g: -g["nodes"])
json.dump(cov, open(f"{D}/covered_cls.json", "w"))
json.dump(groups, open(f"{D}/groups.json", "w"))
print(f"{len(cov)} missing nodes: {sum(c['green'] >= 0.5 for c in cov)} under foliage, {len(solid)} under something else in {len(groups)} groups")
print("groups by size:", collections.Counter(min(g['nodes'], 10) for g in groups).most_common())
for g in groups[:30]:
  print(g["nodes"], "nodes", g["span_m"], "m span, cover", g["cover_med"], "m, at", g["x"], g["y"], g["z"], "cells", len(g["cells"]))
