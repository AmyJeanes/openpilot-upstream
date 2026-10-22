"""visible.py x0 x1 y0 y1 out.json: whether each paint line in the game files shows in the captured imagery. Some lines
in the files are never drawn in the game (the top street's wide westbound dashes: no paint in the imagery), and they
split real lanes.

Per line: along its painted stretches (where the surface raster has paint under it; all along if hardly any), the imagery's brightness on the
line against the plain road beside it (the brighter side where both are road, just clear of the line). Lines under a deck read the under-deck imagery;
where no shot covers them they are unknown (see drawn). Out: {id: {"contrast": median of line minus beside, "on": the
line's own median (0-255 grey), "n": samples, "src": "sat"/"under", "colour", "drawn": true/false/null (unknown)}}."""
import functools, json, math, sys
import numpy as np
import cv2
from pmtiles.reader import MmapSource
from pmtiles.tile import deserialize_header, deserialize_directory, find_tile, zxy_to_tileid
sys.path.insert(0, "/home/amy/git/op-ortho/openpilot/tools/sim/bridge/gta5/map/webmap")
import gframe
import scan

Z = 21


class Reader:
  """pmtiles tiles by z x y, its directories read once (the library's reader re-reads them for every tile)."""

  def __init__(self, path):
    self.get_bytes = MmapSource(open(path, "rb"))
    self.h = deserialize_header(self.get_bytes(0, 127))
    self.dir = functools.lru_cache(maxsize=None)(lambda o, n: deserialize_directory(self.get_bytes(o, n)))

  def get(self, z, x, y):
    tid, o, n = zxy_to_tileid(z, x, y), self.h["root_offset"], self.h["root_length"]
    for _ in range(4):
      r = find_tile(self.dir(o, n), tid)
      if not r:
        return None
      if r.run_length:
        return self.get_bytes(self.h["tile_data_offset"] + r.offset, r.length)
      o, n = self.h["leaf_directory_offset"] + r.offset, r.length


SRC = {"sat": Reader("/mnt/e/gta5_webmap/sat.pmtiles"), "under": Reader("/mnt/e/gta5_webmap/under.pmtiles")}
cache = {}


def tile(src, tx, ty):
  k = (src, tx, ty)
  if k not in cache:
    data = SRC[src].get(Z, tx, ty)
    cache[k] = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED) if data else None
  return cache[k]


def grey(src, x, y):
  """the imagery's grey value at game points (nan where there is none, or it's transparent)."""
  lat, lon = gframe.lat_lon(np.asarray(x, float), np.asarray(y, float))
  n = 2 ** Z
  fx = (lon + 180) / 360 * n
  fy = (1 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2 * n
  out = np.full(len(fx), np.nan)
  tx, ty = fx.astype(int), fy.astype(int)
  for key in set(zip(tx.tolist(), ty.tolist())):
    im = tile(src, *key)
    if im is None:
      continue
    m = np.flatnonzero((tx == key[0]) & (ty == key[1]))
    s = im.shape[0]
    px = im[np.minimum(((fy[m] % 1) * s).astype(int), s - 1), np.minimum(((fx[m] % 1) * s).astype(int), s - 1)]
    ok = px[:, 3] >= 128 if px.shape[1] == 4 else np.ones(len(m), bool)
    out[m[ok]] = px[ok, :3].mean(1)
  return out


# drawn white paint is white: on the interchange's imagery, lines of 6 m or more the game doesn't draw read 104-118 grey
# (dashes over concrete, stains, a kerb's edge: a light-to-dark edge can pass a contrast test), drawn ones 148 and up
# (dashed lines measured on their dashes; scraps under 6 m read 97-135 and aren't used as lane lines). Yellow paint isn't bright, so a
# yellow line must stand out from the road (real ones +64..+70). Far darker than the road, a line is under something
# the surface raster doesn't hold (a rail bridge: -54): unknown. On pale concrete (a ramp's deck: 185-200 grey bare) a
# white line from the surface raster must also stand out from the road beside it: the deck's panel seams and kerb edges
# it holds as paint read +20..+27 there, drawn lines +42 and up (a decal is a painted texture: bright paint on bright
# concrete reads +33 and is drawn)
WHITE_MIN, YELLOW_CONTRAST, HIDDEN, SURFACE_CONTRAST = 135.0, 10.0, -15.0, 35.0


def drawn(v, colour):
  if v["contrast"] is None or v["contrast"] <= HIDDEN:
    return None
  if colour != "white":
    return v["contrast"] >= YELLOW_CONTRAST
  return v["on"] >= WHITE_MIN and (v.get("paint") != "surface" or v["contrast"] >= SURFACE_CONTRAST)


def covered(S):
  """per point: another surface more than 2.5 m above it (under a deck)."""
  cl, zl = scan.layers(S[:, 0], S[:, 1])
  return ((cl > 0) & (zl > S[:, 2] + 2.5)).any(0)


x0, x1, y0, y1 = map(float, sys.argv[1:5])
res = {}
for r in scan.src.paint():
  P = np.array(r["pts"], float)
  if not ((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1)).any():
    continue
  d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
  t = np.arange(0.1, d[-1], 0.25)
  if not len(t):
    continue
  S = np.stack([np.interp(t, d, P[:, k]) for k in range(3)], 1)
  e = np.gradient(S[:, :2], axis=0) if len(S) > 1 else np.diff(P[:2, :2], axis=0)
  e /= np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
  nrm = np.stack([e[:, 1], -e[:, 0]], 1)
  # painted stretches only: the raster's paint at the line's height
  painted = (scan.T.sample(S[:, 0], S[:, 1], S[:, 2], zwin=0.6)[0] >> 3) > 0
  if painted.sum() >= 4:  # (a line barely in the raster is measured all along: it may be one the game doesn't draw)
    S, nrm = S[painted], nrm[painted]
  off = r["width"] / 2 + 0.35
  cov = covered(S)
  # in the open the satellite imagery shows it; under a deck only the under-deck shots can (the satellite shows the
  # deck), and with none there it is unknown
  src = "under" if cov.mean() > 0.5 else "sat"
  on = grey(src, S[:, 0], S[:, 1])
  # beside it: plain road only (not another line, such as a double line's other half, and not a kerb), the brighter
  # side where both are: a line must stand out from the road on both sides, and road stains are dark
  sides = []
  for k in (1, -1):
    X, Y = S[:, 0] + nrm[:, 0] * off * k, S[:, 1] + nrm[:, 1] * off * k
    g = grey(src, X, Y)
    plain = scan.T.sample(X, Y, S[:, 2], zwin=0.6)[0] == 1
    sides.append(np.where(plain, g, np.nan))
  side = np.nanmax(np.stack(sides), 0)
  ok = ~np.isnan(on) & ~np.isnan(side)
  if ok.sum() >= 4:
    res[r["id"]] = {"contrast": round(float(np.median(on[ok] - side[ok])), 1), "on": round(float(np.median(on[ok])), 1),
                    "n": int(ok.sum()), "src": src, "colour": r["colour"], "paint": r["src"]}
  else:
    res[r["id"]] = {"contrast": None, "n": int(ok.sum()), "src": src, "colour": r["colour"], "paint": r["src"]}
for v in res.values():
  v["drawn"] = drawn(v, v["colour"])
json.dump(res, open(sys.argv[5], "w"))
c = np.array([v["contrast"] for v in res.values() if v["contrast"] is not None])
print(len(res), "lines,", len(c), "seen; contrast:", np.histogram(c, [-300, -10, 0, 5, 10, 15, 20, 30, 50, 80, 300])[0].tolist(),
      "(bins -10 0 5 10 15 20 30 50 80)")
print("not drawn:", sorted(k for k, v in res.items() if v["drawn"] is False))
