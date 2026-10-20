"""ortho_edges.py <out json> [--min-px N]: uncaptured cells that still hold land. Every height tile is shot 128/72 px wider
than its cell on each side, so its margins already show the neighbouring cells: a margin (or corner) over an
uncaptured neighbour with more than N px above the local water surface + 0.5 m means that neighbour has land,
a pier or a structure the map is missing. Writes a plan in ortho's format plus per-cell evidence."""
import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

F = "/mnt/e/gta5_ortho_20261018"
D = f"{F}/depth/tiles"
CW, CH, CX0, CY0 = 2304, 1296, 128, 72
# neighbour offset -> (rows, cols) of the margin over it; row 0 is north, so j+1 is the top margin
SIDES = {(-1, 0): (slice(CY0, CY0 + CH), slice(0, CX0)), (1, 0): (slice(CY0, CY0 + CH), slice(CX0 + CW, None)),
         (0, 1): (slice(0, CY0), slice(CX0, CX0 + CW)), (0, -1): (slice(CY0 + CH, None), slice(CX0, CX0 + CW)),
         (-1, 1): (slice(0, CY0), slice(0, CX0)), (1, 1): (slice(0, CY0), slice(CX0 + CW, None)),
         (-1, -1): (slice(CY0 + CH, None), slice(0, CX0)), (1, -1): (slice(CY0 + CH, None), slice(CX0 + CW, None))}
captured = set()
RISE = 0.5  # m above the water surface that counts
SEABED = 0.0  # also seabed shallower than this (m) under the water (the 1 m water probes; the height tiles stop at the surface)


def scan(cell):
  i, j = cell
  want = [(di, dj) for di, dj in SIDES if (i + di, j + dj) not in captured]
  if not want:
    return []
  s = json.load(open(f"{D}/o{i:03d}_{j:03d}.json"))
  hs = s["height"]
  v = np.array(Image.open(f"{D}/{hs['png']}"))
  ok = v != hs["nodata"]
  h = hs["base_m"] + v.astype(np.float32) * hs["scale_m"]
  surf = 0.0
  w = np.load(f"{D}/o{i:03d}_{j:03d}_w.npz")
  if (w["c"] == 1).any():
    surf = float(np.nanmax(np.where(w["c"] == 1, w["h"], -1e9)))
  land = ok & (h > surf + RISE)
  shallow = {}
  if SEABED > 0:
    xc, yc = -4300 + 69.12 * i, -4300 + 38.88 * j
    wet = (w["c"] == 1) & (w["h"] - w["bed"] < SEABED)
    rows, cols = np.nonzero(wet)
    dx, dy = float(w["x0"]) + cols * float(w["step"]) - xc, float(w["y0"]) + rows * float(w["step"]) - yc
    for di, dj in want:
      inx = dx < -34.56 if di < 0 else dx > 34.56 if di > 0 else np.abs(dx) <= 34.56
      iny = dy < -19.44 if dj < 0 else dy > 19.44 if dj > 0 else np.abs(dy) <= 19.44
      shallow[(di, dj)] = int((inx & iny).sum()) * 1111  # 1 m2 points, counted in 3 cm px
  out = []
  for di, dj in want:
    r, c = SIDES[(di, dj)]
    n = int(land[r, c].sum()) + shallow.get((di, dj), 0)
    if n:
      out.append(((i + di, j + dj), n, (i, j), round(surf, 2)))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("out")
  ap.add_argument("--min-px", type=int, default=2000)
  ap.add_argument("--seabed", type=float, default=0.0, help="also seabed shallower than this many m, which shows through the water")
  a = ap.parse_args()
  global captured
  captured = {(int(f[1:4]), int(f[5:8])) for f in os.listdir(f"{F}/full/tiles") if f.endswith(".png") and len(f) == 12}
  cells = sorted((int(f[1:4]), int(f[5:8])) for f in os.listdir(D) if f.endswith("_h.png"))
  hits = {}
  with ProcessPoolExecutor(6, initializer=globals().update, initargs=({"captured": captured, "SEABED": a.seabed},)) as ex:
    for res in ex.map(scan, cells, chunksize=64):
      for nb, n, src, surf in res:
        e = hits.setdefault(nb, {"px": 0, "from": [], "surf": surf})
        e["px"] = max(e["px"], n)
        e["from"].append(list(src))
  need = sorted(c for c, e in hits.items() if e["px"] >= a.min_px)
  tiles = [{"name": f"o{i:03d}_{j:03d}", "i": i, "j": j, "x": round(-4300 + 69.12 * i, 2), "y": round(-4300 + 38.88 * j, 2)} for i, j in need]
  json.dump({"cm_per_px": 3.0, "w_m": 76.8, "h_m": 43.2, "step": [69.12, 38.88], "origin": [-4300.0, -4300.0], "tiles": tiles,
             "evidence": {f"{i},{j}": hits[(i, j)] for i, j in need}}, open(a.out, "w"))
  px = sorted(e["px"] for e in hits.values())
  print(f"{len(captured)} captured, {len(cells)} height tiles; {len(hits)} uncaptured neighbours with any land, {len(need)} with >= {a.min_px} px")
  for t in (500, 1000, 2000, 5000, 20000):
    print(f"  >= {t} px: {sum(p >= t for p in px)}")


if __name__ == "__main__":
  main()
