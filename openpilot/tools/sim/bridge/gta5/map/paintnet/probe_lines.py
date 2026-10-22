"""probe_lines.py x0 x1 y0 y1: paint polylines touching a box: id, colour, style, length, ends and height."""
import json, sys
import numpy as np

PAINT = "/mnt/e/gta5_scratch/gamefiles_v2/rp_all/polylines.jsonl"
x0, x1, y0, y1 = map(float, sys.argv[1:5])
for line in open(PAINT):
  r = json.loads(line)
  P = np.array(r["pts"], float)
  if not ((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)).any():
    continue
  a, b = P[0].round(1).tolist(), P[-1].round(1).tolist()
  print(r["id"], r["colour"], r["style"], r.get("width"), round(r["len"], 1), r.get("painted"), r.get("src"), a, b,
        len(r.get("dashes") or []))
