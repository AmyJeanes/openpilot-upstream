"""compose_full.py <out dir> [names... | --all] [--jobs N]: water drawn into the albedo tiles, per body (params.json,
bodies.npz from bodies.py), from each height tile's own water probe (<cell>_w.npz).

Water = inside the game's water (bilinear 0.5 contour of the 1 m probe) where the height tile shows the surface
(|H - W| <= 3 cm), the ground under it (the sea's plane isn't in the depth) or nothing. Opacity = a0 + (1 - a0) * fast
ramp of depth / dcut over drawn bed, solid where the albedo is empty; near empty water it rises to solid over
edge_fade_m (the game stops drawing the bed short of dcut in deep bodies); feathered inside the water. Body colour and
opacity are blended over body_blend_m where bodies meet. Empty albedo outside the water takes the lit frame.
Rivers (the river bodies, and water of river_step_bodies, or sea by a river, whose height steps by over river_step_m
within river_step_win_m) are flat
water.xml steps down a sloping bed (the game draws no river water from above): there the water also covers bed up to
river_rise_m above the step, empty albedo and dry islands under river_hole_m2, at least river_alpha_min opaque.
Writes <out>/tiles/<name>.png for tiles with water, stats.jsonl."""
import json, os, sys
from multiprocessing import Pool
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

S = os.path.dirname(os.path.abspath(__file__))
F = "/mnt/e/gta5_ortho_20261018"
T = f"{F}/depth/tiles"
TOL = 0.03
P = json.load(open(f"{S}/params.json"))
G = None  # bodies grid, loaded per worker


def nearest_fill(v, ok):
  if ok.all() or not ok.any():
    return v
  idx = ndi.distance_transform_edt(~ok, return_distances=False, return_indices=True)
  return v[tuple(idx)]


def load_globals(path):
  global G
  z = np.load(path)
  G = {"body": z["body"], "x0": int(z["x0"]), "y0": int(z["y0"])}


def body_fields(x0, y0, h, w, names):
  """teal r,g,b, a0, dcut on the probe's 1 m grid: per body, blended where bodies meet, nearest body off the water."""
  j0, i0 = y0 - G["y0"], x0 - G["x0"]
  pad = int(P["body_blend_m"] * 3)
  b = G["body"][j0 - pad:j0 + h + pad, i0 - pad:i0 + w + pad]
  ok = b > 0
  if not ok.any():
    b = np.full(b.shape, names.index("sea"), np.uint8); ok = np.ones(b.shape, bool)
  b = nearest_fill(b, ok)
  table = np.zeros((len(names), 5), np.float32)
  for k, n in enumerate(names):
    if n in P["bodies"]:
      q = P["bodies"][n]
      table[k] = [int(q["teal"][0:2], 16), int(q["teal"][2:4], 16), int(q["teal"][4:6], 16), q["a0"], q["dcut"]]
  f = table[b]
  f = np.stack([ndi.gaussian_filter(f[..., c], P["body_blend_m"]) for c in range(5)], -1)
  return f[pad:pad + h, pad:pad + w]


def compose(name, out):
  side = json.load(open(f"{T}/{name}.json"))
  if not side.get("water", {}).get("water_pts"):
    return None
  z = np.load(f"{T}/{side['water']['npz']}")
  h, c, bed, bc = z["h"], z["c"], z["bed"], z["bcode"]
  wat = c == 1
  x0, y0, step = int(z["x0"]), int(z["y0"]), float(z["step"])
  Wg = nearest_fill(np.where(wat, h, 0), wat)
  good = wat & (bc == 1) & (h - bed >= 0)
  Bg = nearest_fill(np.where(good, bed, 0), good) if good.any() else Wg.copy()
  names = json.load(open(os.path.join(os.path.dirname(BODIES), "bodies.json")))["bodies"]
  fld = body_fields(x0, y0, *wat.shape, names)
  M = json.load(open(f"{F}/full/tiles/{name}.json"))["world_to_px"]
  u, v = np.arange(2560) + 0.5, np.arange(1440) + 0.5
  X, Y = (u - M[0][2]) / M[0][0], (v - M[1][2]) / M[1][1]
  fi = np.broadcast_to(((X - x0) / step)[None, :], (1440, 2560))
  fj = np.broadcast_to(((Y - y0) / step)[:, None], (1440, 2560))
  smp = lambda g, o: ndi.map_coordinates(g, [fj, fi], order=o, mode="nearest").astype(np.float32)
  inside = smp(wat.astype(np.float32), 1) >= 0.5
  if not inside.any():
    return None
  W, B = smp(Wg, 1), smp(Bg, 1)
  hm = side["height"]
  hv = np.array(Image.open(f"{T}/{hm['png']}"))
  H = np.where(hv == hm["nodata"], np.nan, hm["base_m"] + hv.astype(np.float32) * hm["scale_m"])
  A = np.array(Image.open(f"{F}/full/tiles/{name}.png").convert("RGB"))
  ff = f"{F}/full_filled/{name}.png"  # only tiles that had gaps to fill are there
  cur = np.array(Image.open(ff).convert("RGB")) if os.path.exists(ff) else A
  empty = A.sum(2) == 0
  nan = np.isnan(H)
  below = inside & (H < W - TOL)
  water = inside & ((np.abs(H - W) <= TOL) | nan | below)
  bw = G["body"][y0 - G["y0"]:y0 - G["y0"] + wat.shape[0], x0 - G["x0"]:x0 - G["x0"] + wat.shape[1]]
  rv = np.isin(bw, [names.index(n) for n in P["rivers"]])
  # river water the bodies pass gave a lake's or the sea's name: lakes and the calm sea are one flat height, rivers
  # step down (pools and tanks also step, between basins: left alone; the sea only where a river runs into it)
  k = 2 * P["river_step_win_m"] + 1
  span = ndi.maximum_filter(np.where(wat, h, -1e9), k) - ndi.minimum_filter(np.where(wat, h, 1e9), k)
  ok = np.isin(bw, [names.index(n) for n in P["river_step_bodies"]])
  ok |= (bw == names.index("sea")) & ndi.binary_dilation(rv, iterations=30)
  rv |= wat & ok & (span > P["river_step_m"])
  river = inside & (smp(ndi.binary_dilation(rv, iterations=2).astype(np.float32), 1) >= 0.5)
  if river.any():
    water |= river & ((H <= W + P["river_rise_m"]) | empty)
    holes = ndi.binary_fill_holes(water) & ~water
    lab, n = ndi.label(holes)
    if n:
      small = np.flatnonzero(ndi.sum(np.ones_like(lab), lab, np.arange(1, n + 1)) * 0.03 ** 2 < P["river_hole_m2"]) + 1
      water |= np.isin(lab, small) & ndi.binary_dilation(river, iterations=30)
  captured = water & ~empty
  depth = np.clip(np.where(below, W - H, np.where(H > W + TOL, 0, W - B)), 0, None)
  teal = np.stack([smp(fld[..., k], 1) for k in range(3)], -1)
  a0, dcut = smp(fld[..., 3], 1), np.maximum(smp(fld[..., 4], 1), 0.1)
  s = np.clip(depth / dcut, 0, 1)
  alpha = np.where(captured, a0 + (1 - a0) * (1 - (1 - s) ** 2), 1.0).astype(np.float32)
  if river.any():  # the step heights say little about real depth there: one even tint, eased in from the banks
    t = np.clip(ndi.distance_transform_edt(water) * 0.03 / P["river_edge_m"], 0, 1)
    alpha = np.where(river, np.maximum(alpha, P["river_alpha_min"] * t * t * (3 - 2 * t)), alpha)
  if (water & empty).any() and captured.any():
    d = ndi.distance_transform_edt(~(water & empty)) * 0.03  # px are 3 cm
    t = np.clip(1 - d / P["edge_fade_m"], 0, 1)
    alpha = np.maximum(alpha, t * t * (3 - 2 * t))
  wm = water.astype(np.float32)
  num, den = ndi.gaussian_filter(alpha * wm, P["feather_px"]), ndi.gaussian_filter(wm, P["feather_px"])
  alpha = np.where(water, num / np.maximum(den, 1e-6), 0).astype(np.float32)
  rest = cur
  if (empty & ~water).any():  # the old fill painted these teal or lit; they aren't water: the lit frame
    lit = np.array(Image.open(f"{F}/full/tiles/{name}_lit.png").convert("RGB"))
    rest = np.where((empty & ~water)[..., None], lit, cur)
  base = A.astype(np.float32)
  img = np.where(water[..., None], base * (1 - alpha[..., None]) + teal * alpha[..., None], rest).astype(np.uint8)
  Image.fromarray(img).save(f"{out}/tiles/{name}.png", compress_level=3)
  bodies_here = np.unique(G["body"][y0 - G["y0"]:y0 - G["y0"] + wat.shape[0], x0 - G["x0"]:x0 - G["x0"] + wat.shape[1]])
  return {"name": name, "water": round(float(water.mean()), 4), "captured": round(float(captured.mean()), 4),
          "empty_water": round(float((water & empty).mean()), 4), "empty_not_water": round(float((empty & ~water).mean()), 4),
          "bodies": [names[b] for b in bodies_here if b]}


def work(name):
  try:
    return compose(name, OUT)
  except Exception as e:  # one bad tile shouldn't stop the run
    return {"name": name, "error": repr(e)}


BODIES = os.environ.get("BODIES", "/mnt/e/gta5_audit/water_full_20261019/bodies.npz")
OUT = sys.argv[1]
if __name__ == "__main__":
  os.makedirs(f"{OUT}/tiles", exist_ok=True)
  args = [a for a in sys.argv[2:] if not a.startswith("--")]
  jobs = next((int(a[7:]) for a in sys.argv if a.startswith("--jobs=")), 4)
  if "--all" in sys.argv:
    args = sorted(f[:-5] for f in os.listdir(T) if f.endswith(".json") and not os.path.exists(f"{OUT}/tiles/{f[:-5]}.png"))
  done = 0
  with Pool(jobs, initializer=load_globals, initargs=(BODIES,)) as pool, open(f"{OUT}/stats.jsonl", "a") as st:
    for r in pool.imap_unordered(work, args, chunksize=4):
      done += 1
      if r:
        st.write(json.dumps(r) + "\n"); st.flush()
        if "error" in r or len(args) < 50:
          print(json.dumps(r), flush=True)
      if done % 500 == 0:
        print(f"{done} of {len(args)}", flush=True)
  print("done", len(args))
