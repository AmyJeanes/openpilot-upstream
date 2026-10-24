"""arrows.py: arrows found by their shape in the surface raster's paint. The markings list misses many arrows (8 of the
interchange's 12 westbound motorway ones), and an arrow's shaft reads as a short lengthwise line.
blobs.py on the interchange: arrows 3.4-5.7 m long, 1.6-2.9 m wide, 34-47% filled; other paint pieces that size are
under 0.9 m wide or under 13% filled."""
import functools, math, sys
import numpy as np
import cv2
from scipy import ndimage as ndi
sys.path.insert(0, "/home/amy/git/openpilot-slowroads/openpilot/tools/sim/bridge/gta5/map/roadpaint")
import rp

TILES = "/mnt/e/gta5_scratch/gamefiles_v2/rp_all/tiles"
ARROW_LEN, ARROW_W, ARROW_FILL = (3.0, 6.5), 1.4, 0.3


def load(x0, x1, y0, y1, res, layers=3):
  """the surface raster over a box at res m: per layer, the code (class | paint << 3) and height; row 0 is north."""
  W, H = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
  xs = x0 + (np.arange(W) + 0.5) * res
  ys = y1 - (np.arange(H) + 0.5) * res
  code = np.zeros((layers, H, W), np.uint8)
  zz = np.full((layers, H, W), np.nan, np.float32)
  for tx in range(int(math.floor(x0 / rp.TILE)), int(math.floor(x1 / rp.TILE)) + 1):
    for ty in range(int(math.floor(y0 / rp.TILE)), int(math.floor(y1 / rp.TILE)) + 1):
      try:
        c, z = rp.read_tile(f"{TILES}/{tx}_{ty}.bin")
      except FileNotFoundError:
        continue
      mc = (xs >= tx * rp.TILE) & (xs < tx * rp.TILE + rp.TILE)
      mr = (ys >= ty * rp.TILE) & (ys < ty * rp.TILE + rp.TILE)
      col = ((xs[mc] - tx * rp.TILE) / rp.RES).astype(int)
      row = ((ty * rp.TILE + rp.TILE - ys[mr]) / rp.RES).astype(int)
      for l in range(c.shape[0]):
        code[l][np.ix_(mr, mc)] = c[l][np.ix_(row, col)]
        zz[l][np.ix_(mr, mc)] = z[l][np.ix_(row, col)]
  return code, zz


def find(paint, zz, x0, y1, res):
  """arrow-shaped paint pieces: a per-layer mask of them (grown 0.2 m), and each as (x, y, z, radius m)."""
  L = paint.shape[0]
  mask = np.zeros(paint.shape, bool)
  out = []
  for l in range(L):
    n, cl, st, _ = cv2.connectedComponentsWithStats(paint[l].astype(np.uint8), connectivity=8)
    for k in range(1, n):
      x, y, w, h, a = st[k]
      if a < 1.0 / (res * res) or max(w, h) * res > ARROW_LEN[1] * 1.5:
        continue
      m = cl[y:y + h, x:x + w] == k
      rr, cc = np.nonzero(m)
      P = np.stack([cc, rr], 1).astype(float)
      ctr = P.mean(0)
      _, ev = np.linalg.eigh(np.cov((P - ctr).T))
      ln, wd = np.ptp((P - ctr) @ ev[:, 1]) * res, np.ptp((P - ctr) @ ev[:, 0]) * res
      if ARROW_LEN[0] <= ln <= ARROW_LEN[1] and wd >= ARROW_W and a * res * res / (ln * wd) >= ARROW_FILL:
        mask[l, y:y + h, x:x + w] |= ndi.binary_dilation(m, iterations=max(1, int(round(0.2 / res))))
        out.append((x0 + (x + ctr[0] + 0.5) * res, y1 - (y + ctr[1] + 0.5) * res, float(np.nanmean(zz[l, y:y + h, x:x + w][m])), ln / 2))
  return mask, out


@functools.cache
def box(x0, x1, y0, y1, res):
  """load and find over a box, once per process (build.py does its box's before forking the steps): the codes, heights,
  arrow mask and arrows, read-only (shared)."""
  code, zz = load(x0, x1, y0, y1, res)
  mask, found = find((code >> 3) > 0, zz, x0, y1, res)
  for a in (code, zz, mask):
    a.setflags(write=False)
  return code, zz, mask, tuple(found)


# an arrow found by its shape has no type: it takes the type whose average picture (from the listed arrows of the
# common road-marking atlases, drawn in each one's own frame) its paint matches best, if well enough. Measured on 142
# held-out listed arrows: 137 right, all at least 0.03 ahead of the next type; the 5 wrong matched at 0.06-0.39 (through
# and through;right look alike: a right match can be only 0.08 ahead). A left+right
# arrow is a left and a right decal on one spot (features.py lists them as two), so its picture is theirs overlaid
TMPL_RES, TMPL_ALONG, TMPL_ACROSS, TMPL_N = 0.1, 3.2, 1.8, 120
TMPL_MIN, TMPL_MARGIN = 0.45, 0.03
TMPL_ATLAS = ("im_roadmarkings02", "dc_roadmarkings01", "cls_cs3")
_tmpl = None


def _crop(T, x, y, z, e, du=0.0, dv=0.0):
  """paint (0/1) round (x, y) on the deck at z in the frame of unit direction e (rows along it, columns across, left
  first), shifted du along and dv across."""
  ua = np.arange(-TMPL_ALONG, TMPL_ALONG + 1e-6, TMPL_RES) + du
  va = np.arange(-TMPL_ACROSS, TMPL_ACROSS + 1e-6, TMPL_RES) + dv
  U, V = np.meshgrid(ua, va, indexing="ij")
  n = np.array([-e[1], e[0]])
  code, _ = T.sample((x + U * e[0] + V * n[0]).ravel(), (y + U * e[1] + V * n[1]).ravel(), z, zwin=1.5)
  return ((code >> 3) > 0).reshape(U.shape).astype(np.float32)


def templates(T, feats, cache):
  """the average picture of each arrow type (cached in cache, an .npz path, keyed by the caller)."""
  global _tmpl
  if _tmpl is None:
    import os
    if os.path.exists(cache):
      _tmpl = {k.replace("_", ";"): v for k, v in np.load(cache).items()}
    else:
      by = {}
      for f in feats:
        if f["kind"] == "arrow" and f["tex"].startswith(TMPL_ATLAS):
          by.setdefault(f["arrow"], []).append(f)
      _tmpl = {}
      for k, L in by.items():
        L = sorted(L, key=lambda f: (f["x"], f["y"]))[::max(len(L) // TMPL_N, 1)][:TMPL_N]  # spread over the map
        h = [np.radians(f["heading"]) for f in L]
        C = [_crop(T, f["x"], f["y"], f["z"], np.array([math.cos(a), math.sin(a)])) for f, a in zip(L, h)]
        _tmpl[k] = np.mean([c for c in C if c.mean() > 0.05], 0)
      _tmpl["left;right"] = np.maximum(_tmpl["left"], _tmpl["right"])
      _tmpl["left;through;right"] = np.maximum(_tmpl["through;left"], _tmpl["through;right"])
      tmp = cache[:-4] + ".tmp.npz"
      np.savez(tmp, **{k.replace(";", "_"): v for k, v in _tmpl.items()})
      os.replace(tmp, cache)
  return _tmpl


def kind(T, x, y, z, e, tmpl):
  """the type of the arrow painted round (x, y) heading along e ("through", "through;left", ...; listed as osm orders
  them: left, through, right), or None if no picture matches well enough (searched over small shifts: a shape's centre
  is not its decal's)."""
  best = {}
  for du in np.arange(-0.6, 0.61, 0.2):
    for dv in np.arange(-0.3, 0.31, 0.15):
      c = _crop(T, x, y, z, e, du, dv)
      a = c - c.mean()
      for k, P in tmpl.items():
        b = P - P.mean()
        s = float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-9))
        best[k] = max(best.get(k, -1), s)
  r = sorted(best.items(), key=lambda kv: -kv[1])
  if r[0][1] < TMPL_MIN or r[0][1] - r[1][1] < TMPL_MARGIN:
    return None
  order = ["left", "through", "right"]
  return ";".join(t for t in order if t in r[0][0].split(";"))
