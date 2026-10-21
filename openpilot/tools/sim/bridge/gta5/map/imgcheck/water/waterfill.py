"""waterfill.py out_dir tile.png...: fill albedo's empty pixels (exactly 0: water, glass, some decals) from the lit frame,
colour-matched per tile by a per-channel linear fit of lit -> albedo over the pixels both have."""
import os
import sys

import numpy as np
from PIL import Image


def fill(albedo, lit):
  a = albedo.astype(np.float32)
  b = lit.astype(np.float32)
  empty = albedo.sum(2) == 0
  have = ~empty
  out = a.copy()
  if empty.any():
    idx = np.flatnonzero(have)
    if len(idx) > 20000:
      idx = np.random.default_rng(0).choice(idx, 20000, replace=False)
    for c in range(3):
      x, y = b[..., c].ravel()[idx], a[..., c].ravel()[idx]
      if len(idx) > 500:
        k, m = np.polyfit(x, y, 1)
      else:  # almost all water: no shared ground, use the full map's typical gain
        k, m = 0.55, 0.0
      out[..., c][empty] = b[..., c][empty] * k + m
  return np.clip(out, 0, 255).astype(np.uint8), empty


def main():
  out_dir = sys.argv[1]
  os.makedirs(out_dir, exist_ok=True)
  for p in sys.argv[2:]:
    a = np.asarray(Image.open(p).convert("RGB"))
    b = np.asarray(Image.open(p[:-4] + "_lit.png").convert("RGB"))
    f, empty = fill(a, b)
    name = os.path.basename(p)[:-4]
    w = a.shape[1] // 2
    side = np.concatenate([a[:, ::2][::2], f[:, ::2][::2]], 1)  # half-size: albedo | filled
    Image.fromarray(side).save(os.path.join(out_dir, f"{name}_side.jpg"), quality=90)
    print(name, f"empty {empty.mean():.1%}", "->", os.path.join(out_dir, f"{name}_side.jpg"), w)


if __name__ == "__main__":
  main()
