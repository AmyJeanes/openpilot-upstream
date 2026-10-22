"""satcrop.py x0 x1 y0 y1 out.png [zoom]: the web map's satellite imagery over a game-coordinate box, at RES 0.1 m
per pixel (north up, like regions.py's rasters) so it lines up with them."""
import math, sys
import numpy as np
import cv2
from pmtiles.reader import Reader, MmapSource
sys.path.insert(0, "/home/amy/git/op-ortho/openpilot/tools/sim/bridge/gta5/map/webmap")
import gframe

RES = 0.1
x0, x1, y0, y1 = map(float, sys.argv[1:5])
Z = int(sys.argv[6]) if len(sys.argv) > 6 else 21
f = open("/mnt/e/gta5_webmap/sat.pmtiles", "rb")
rd = Reader(MmapSource(f))


def tile_xy(lat, lon):
  n = 2 ** Z
  tx = (lon + 180) / 360 * n
  ty = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n
  return tx, ty


W, H = int(round((x1 - x0) / RES)), int(round((y1 - y0) / RES))
gx = x0 + (np.arange(W) + 0.5) * RES
gy = y1 - (np.arange(H) + 0.5) * RES
X, Y = np.meshgrid(gx, gy)
lat, lon = gframe.lat_lon(X.ravel(), Y.ravel())
T = np.array([tile_xy(a, o) for a, o in zip(lat, lon)])
tx, ty = T[:, 0], T[:, 1]
out = np.zeros((H * W, 3), np.uint8)
cache = {}
for k in set(zip(np.floor(tx).astype(int), np.floor(ty).astype(int))):
  data = rd.get(Z, k[0], k[1])
  if data:
    cache[k] = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
ix, iy = np.floor(tx).astype(int), np.floor(ty).astype(int)
for k, im in cache.items():
  m = (ix == k[0]) & (iy == k[1])
  s = im.shape[0]
  px = np.clip(((tx[m] - k[0]) * s).astype(int), 0, s - 1)
  py = np.clip(((ty[m] - k[1]) * s).astype(int), 0, s - 1)
  out[m] = im[py, px]
cv2.imwrite(sys.argv[5], out.reshape(H, W, 3))
