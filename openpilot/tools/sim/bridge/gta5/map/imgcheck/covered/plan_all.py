"""plan_all.py <dir>: an under-deck plan (under/plan.json format) for every covered road node no under-deck shot has:
not under foliage, not boat nodes (z 0), not multi-storey car parks (a group of 10+ nodes within 8 m spanning 3+ m of
height: their levels are interiors the camera never draws). Per cell, nodes binned to 3 m levels -> covered_plan.json."""
import collections, json, os, sys
import numpy as np

D = sys.argv[1]
cov = [c for c in json.load(open(f"{D}/covered_cls.json")) if c["green"] < 0.5 and c["z"] != 0.0]
park = set()
for g in json.load(open(f"{D}/groups.json")):
  zs = [p[2] for p in g["pts"]]
  if g["nodes"] >= 10 and g["span_m"] < 8 and max(zs) - min(zs) >= 3:
    park |= {(round(p[0], 2), round(p[1], 2), round(p[2], 2)) for p in g["pts"]}
cov = [c for c in cov if (round(c["x"], 2), round(c["y"], 2), round(c["z"], 2)) not in park]
keyed = {(c["a"], c["i"]): c for c in cov}
links = []
for l in open(os.path.expanduser("~/gta5map_lanes/paths.jsonl")):
  if '"t":"l"' in l:
    r = json.loads(l)
    p, q = (r["a"], r["i"]), (r["ta"], r["ti"])
    if p in keyed and q in keyed:
      links.append((p, q))
by_cell = collections.defaultdict(list)
for k, c in keyed.items():
  by_cell[tuple(c["cell"])].append(k)
tiles = []
for (i, j), ks in sorted(by_cell.items()):
  ks.sort(key=lambda k: keyed[k]["z"])
  levels, cur = [], []
  for k in ks:
    if cur and keyed[k]["z"] > keyed[cur[0]]["z"] + 3.0:
      levels.append(cur); cur = []
    cur.append(k)
  levels.append(cur)
  for lv, lk in enumerate(levels, 1):
    s = set(lk)
    segs = [[keyed[a]["x"], keyed[a]["y"], keyed[a]["z"], keyed[b]["x"], keyed[b]["y"], keyed[b]["z"], 6.0] for a, b in links if a in s and b in s]
    c0 = keyed[lk[0]]
    if not segs:
      segs = [[c0["x"] - 2, c0["y"], c0["z"], c0["x"] + 2, c0["y"], c0["z"], 6.0]]
    tiles.append({"name": f"o{i:03d}_{j:03d}_T{lv}", "i": i, "j": j, "x": round(-4300 + 69.12 * i, 2), "y": round(-4300 + 38.88 * j, 2),
                  "level": lv, "z_lo": min(keyed[k]["z"] for k in lk), "z_hi": max(keyed[k]["z"] for k in lk),
                  "crossings": [[keyed[k]["x"], keyed[k]["y"], keyed[k]["z"], keyed[k]["top"]] for k in lk], "segs": [[round(v, 2) for v in q] for q in segs]})
# roads first: cells with more nodes first
tiles.sort(key=lambda t: -len(t["crossings"]))
json.dump({"cm_per_px": 3.0, "near": 0.5, "far": 120.0, "dz": 4.0, "band": 3.0, "crossings": 0, "cells": len(by_cell), "tiles": tiles},
          open(f"{D}/covered_plan.json", "w"))
print(f"{len(keyed)} nodes ({len(park)} car-park nodes left out), {len(by_cell)} cells, {len(tiles)} cell-levels")
