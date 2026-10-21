"""fill_all.py <tiles> <out> [--procs N]: the albedo map with water in one colour. Empty albedo pixels are water once the
gtrace reshoot has removed the holes in land; any empty patch still textured in the lit frame is reported and filled
from the lit frame instead. Writes <out>/<name>.png + its sidecar only for tiles that change (readers fall back to the original tile),
skipping tiles already done; prints a summary."""
import argparse
import glob
import json
import os
import re
import shutil
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy import ndimage

from waterfill import fill

WATER = np.array([38, 60, 70], np.uint8)  # Amy: deep teal
WATER_XML = "/mnt/e/gta5_scratch/gamefiles/water/common.rpf_data_levels_gta5_water.xml"
# the game's water.xml quads as rectangles (its triangle types are treated as their bounding box)
QUADS = [tuple(float(v) for v in m.groups()) for m in re.finditer(
  r'<minX value="([-\d.]+)" /><maxX value="([-\d.]+)" /><minY value="([-\d.]+)" /><maxY value="([-\d.]+)"', open(WATER_XML).read())]
PX_PER_M = 1 / 0.03


def in_water(cx, cy, shape):
  h, w = shape[:2]
  x = cx + (np.arange(w) + 0.5 - w / 2) / PX_PER_M
  y = cy - (np.arange(h) + 0.5 - h / 2) / PX_PER_M
  out = np.zeros((h, w), bool)
  for x0, x1, y0, y1 in QUADS:
    cols = (x >= x0) & (x <= x1)
    rows = (y >= y0) & (y <= y1)
    if cols.any() and rows.any():
      out[np.ix_(rows, cols)] = True
  return out


def one(job):
  src, out = job
  name = os.path.basename(src)[:-4]
  dst = os.path.join(out, name + ".png")
  if os.path.exists(dst):
    return name, None
  a = np.asarray(Image.open(src).convert("RGB"))
  empty = a.sum(2) == 0
  if not empty.any():  # unchanged: readers fall back to the original tile
    return name, {"water": 0.0, "textured": 0.0, "unchanged": True}
  im = a.copy()
  side = json.load(open(src[:-4] + ".json"))
  lit = np.asarray(Image.open(src[:-4] + "_lit.png").convert("RGB"))
  r, g_, b = (lit[..., c].astype(np.int32) for c in range(3))
  sky = empty & (b > 170) & (b >= r + 10) & (g_ > 150)  # nothing drawn there: the lit frame shows the sky
  wet_colour = (b >= r) & (r + g_ + b < 360)
  inquad = in_water(side["x"], side["y"], a.shape)
  lab, n = ndimage.label(empty & ~sky)
  ids = np.arange(1, n + 1)
  g = lit.mean(2).astype(np.float32)
  m = ndimage.uniform_filter(g, 11)
  std = np.sqrt(np.maximum(ndimage.uniform_filter(g * g, 11) - m * m, 0))
  flat = ndimage.median(std, lab, ids) < 3
  watery = (ndimage.mean(inquad, lab, ids) > 0.3) | (ndimage.mean(wet_colour, lab, ids) > 0.5)
  dark = ndimage.median(g, lab, ids) < 40  # deep or shaded water: near black in the lit frame (glass, foam are bright)
  water = np.isin(lab, ids[(flat | dark) & watery])
  other = empty & ~sky & ~water
  im[water] = WATER
  if other.any():  # glass, smoke, foam: from the lit frame, colour-matched
    f, _ = fill(a, lit)
    im[other] = f[other]
  if sky.any():  # gaps where nothing is drawn: the nearest real albedo
    _, (iy, ix) = ndimage.distance_transform_edt(empty, return_indices=True)
    im[sky] = a[iy[sky], ix[sky]]
  res = {"water": round(float(water.mean()), 4), "textured": round(float(other.mean()), 5), "sky": round(float(sky.mean()), 5),
         "water_outside_quads": round(float((water & ~inquad).mean()), 5)}
  tmp = dst + ".part"
  Image.fromarray(im).save(tmp, "PNG", compress_level=2)
  os.replace(tmp, dst)
  shutil.copy2(src[:-4] + ".json", os.path.join(out, name + ".json"))
  return name, res


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("tiles")
  ap.add_argument("out")
  ap.add_argument("--procs", type=int, default=6)
  a = ap.parse_args()
  os.makedirs(a.out, exist_ok=True)
  srcs = sorted(p for p in glob.glob(os.path.join(a.tiles, "o*.png")) if not p.endswith("_lit.png"))
  stats = {}
  with Pool(a.procs) as pool:
    for k, (name, res) in enumerate(pool.imap_unordered(one, [(s, a.out) for s in srcs], chunksize=8)):
      if res is not None:
        stats[name] = res
      if k % 1000 == 0:
        print(f"{k} of {len(srcs)}", flush=True)
  tex = {k: v["textured"] for k, v in stats.items() if v["textured"] > 0}
  out_q = {k: v["water_outside_quads"] for k, v in stats.items() if v.get("water_outside_quads", 0) > 0}
  summary = {"tiles": len(srcs), "done_now": len(stats), "written": sum(1 for v in stats.values() if not v.get("unchanged")),
             "with_water": sum(1 for v in stats.values() if v["water"] > 0),
             "with_lit_fill": len(tex), "worst_lit_fill": sorted(tex.items(), key=lambda kv: -kv[1])[:20],
             "with_sky_gaps": sum(1 for v in stats.values() if v.get("sky", 0) > 0),
             "water_outside_quads": len(out_q), "worst_water_outside_quads": sorted(out_q.items(), key=lambda kv: -kv[1])[:30]}
  json.dump({"summary": summary, "tiles": stats}, open(os.path.join(a.out, "fill_summary.json"), "w"), indent=1)
  print(json.dumps(summary))


if __name__ == "__main__":
  main()
