"""visible.py x0 x1 y0 y1 out.json: whether each paint line in the game files shows in the captured imagery. Some lines
in the files are never drawn in the game (the top street's wide westbound dashes: no paint in the imagery), and they
split real lanes.

Per line: along its painted stretches (where the surface raster has paint under it; all along if hardly any), the imagery's brightness on the
line against the plain road beside it (the brighter side where both are road, just clear of the line). Lines under a deck read the under-deck imagery;
where no shot covers them they are unknown (see drawn). Out: {id: {"contrast": median of line minus beside, "on": the
line's own median (0-255 grey), "n": samples, "src": "sat"/"under", "colour", "drawn": true/false/null (unknown)}}."""
import json, math, sys
import numpy as np
import cv2
from pmtiles.reader import Reader, MmapSource
sys.path.insert(0, "/home/amy/git/op-ortho/openpilot/tools/sim/bridge/gta5/map/webmap")
import gframe
import scan

Z = 21
SRC = {"sat": Reader(MmapSource(open("/mnt/e/gta5_webmap/sat.pmtiles", "rb"))),
       "under": Reader(MmapSource(open("/mnt/e/gta5_webmap/under.pmtiles", "rb")))}
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
  for i, (a, b) in enumerate(zip(fx, fy)):
    im = tile(src, int(a), int(b))
    if im is None:
      continue
    s = im.shape[0]
    px = im[min(int((b % 1) * s), s - 1), min(int((a % 1) * s), s - 1)]
    if px.shape[0] == 4 and px[3] < 128:
      continue
    out[i] = float(px[:3].mean())
  return out


# drawn white paint is white: on the interchange's imagery, lines of 6 m or more the game doesn't draw read 104-118 grey
# (dashes over concrete, stains, a kerb's edge: a light-to-dark edge can pass a contrast test), drawn ones 148 and up
# (dashed lines measured on their dashes; scraps under 6 m read 97-135 and aren't used as lane lines). Yellow paint isn't bright, so a
# yellow line must stand out from the road (real ones +64..+70). Far darker than the road, a line is under something
# the surface raster doesn't hold (a rail bridge: -54): unknown
WHITE_MIN, YELLOW_CONTRAST, HIDDEN = 135.0, 10.0, -15.0


def drawn(v, colour):
  if v["contrast"] is None or v["contrast"] <= HIDDEN:
    return None
  return v["on"] >= WHITE_MIN if colour == "white" else v["contrast"] >= YELLOW_CONTRAST


def covered(S):
  """per point: another surface more than 2.5 m above it (under a deck)."""
  out = np.zeros(len(S), bool)
  for k, (x, y, z) in enumerate(S):
    tx, ty = int(np.floor(x / scan.rp.TILE)), int(np.floor(y / scan.rp.TILE))
    t = scan.T._get(tx, ty)
    if t is None:
      continue
    c, zz = t
    col = min(int((x - tx * scan.rp.TILE) / scan.rp.RES), scan.rp.N - 1)
    row = min(int((ty * scan.rp.TILE + scan.rp.TILE - y) / scan.rp.RES), scan.rp.N - 1)
    out[k] = bool(((c[:, row, col] > 0) & (zz[:, row, col] > z + 2.5)).any())
  return out


x0, x1, y0, y1 = map(float, sys.argv[1:5])
res = {}
for l in open(scan.PAINT):
  r = json.loads(l)
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
  painted = np.array([(int(scan.T.sample(np.array([x]), np.array([y]), z, zwin=0.6)[0][0]) >> 3) > 0 for x, y, z in S])
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
    plain = np.array([int(scan.T.sample(np.array([x]), np.array([y]), z, zwin=0.6)[0][0]) == 1 for x, y, z in zip(X, Y, S[:, 2])])
    sides.append(np.where(plain, g, np.nan))
  side = np.nanmax(np.stack(sides), 0)
  ok = ~np.isnan(on) & ~np.isnan(side)
  if ok.sum() >= 4:
    res[r["id"]] = {"contrast": round(float(np.median(on[ok] - side[ok])), 1), "on": round(float(np.median(on[ok])), 1),
                    "n": int(ok.sum()), "src": src, "colour": r["colour"]}
  else:
    res[r["id"]] = {"contrast": None, "n": int(ok.sum()), "src": src, "colour": r["colour"]}
for v in res.values():
  v["drawn"] = drawn(v, v["colour"])
json.dump(res, open(sys.argv[5], "w"))
c = np.array([v["contrast"] for v in res.values() if v["contrast"] is not None])
print(len(res), "lines,", len(c), "seen; contrast:", np.histogram(c, [-300, -10, 0, 5, 10, 15, 20, 30, 50, 80, 300])[0].tolist(),
      "(bins -10 0 5 10 15 20 30 50 80)")
print("not drawn:", sorted(k for k, v in res.items() if v["drawn"] is False))
