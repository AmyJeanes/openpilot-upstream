"""find.py: road path nodes the height tiles show covered (top surface > COVER m above the node), split into those an
under-deck shot already covers and the missing ones; writes covered.json (per node) and cells.json (missing, per cell)."""
import collections, json, math, os, sys
import numpy as np
from openpilot.tools.sim.bridge.gta5.map.imgcheck.ortho_depth import decode

F = "/mnt/e/gta5_ortho_20261018"
OUT = sys.argv[1]
COVER = 4.0  # m of something over the road
nodes = [json.loads(l) for l in open(os.path.expanduser("~/gta5map_lanes/paths.jsonl")) if '"t":"n"' in l]
by_cell = collections.defaultdict(list)
for n in nodes:
  i, j = round((n["x"] + 4300) / 69.12), round((n["y"] + 4300) / 38.88)
  by_cell[(i, j)].append(n)
under = collections.defaultdict(list)
for t in json.load(open(f"{F}/under/plan.json"))["tiles"]:
  under[(t["i"], t["j"])].append((t["z_lo"], t["z_hi"]))
covered, notile = [], 0
for (i, j), ns in by_cell.items():
  p = f"{F}/depth/tiles/o{i:03d}_{j:03d}.json"
  if not os.path.exists(p):
    notile += len(ns)
    continue
  s = json.load(open(p))
  h = decode(f"{F}/depth/tiles/{s['height']['png']}", s["height"])
  (a, _, c), (_, e, f) = s["world_to_px"]
  for n in ns:
    u, v = int(a * n["x"] + c), int(e * n["y"] + f)
    w = h[max(v - 15, 0):v + 16, max(u - 15, 0):u + 16]  # 1 m around
    if w.size == 0 or not np.isfinite(w).any():
      continue
    top = float(np.nanmedian(w))
    if top - n["z"] > COVER:
      shot = any(lo - 1.0 <= n["z"] <= hi + 3.0 for lo, hi in under.get((i, j), []))
      covered.append({"a": n["a"], "i": n["i"], "x": n["x"], "y": n["y"], "z": n["z"], "top": round(top, 2), "cell": [i, j], "under_shot": shot})
miss = collections.defaultdict(list)
for c in covered:
  if not c["under_shot"]:
    miss[tuple(c["cell"])].append(c)
cells = [{"i": i, "j": j, "x": round(-4300 + 69.12 * i, 2), "y": round(-4300 + 38.88 * j, 2), "nodes": len(v),
          "z_lo": min(c["z"] for c in v), "z_hi": max(c["z"] for c in v), "cover_med": round(float(np.median([c["top"] - c["z"] for c in v])), 1)}
         for (i, j), v in sorted(miss.items())]
json.dump(covered, open(f"{OUT}/covered.json", "w"))
json.dump(cells, open(f"{OUT}/cells.json", "w"), indent=1)
print(f"{len(nodes)} nodes, {notile} without a tile, {len(covered)} covered, {sum(c['under_shot'] for c in covered)} under-deck shot, "
      f"{len(covered) - sum(c['under_shot'] for c in covered)} missing in {len(cells)} cells")
