"""surface.py x0 x1 y0 y1 out.jpg: the game files' road-surface raster (roadpaint v2 tiles) over an area, top layer
left, the layer below it right: road grey, kerb red, pavement tan, gutter orange, paint white/yellow."""
import sys
import numpy as np
import cv2
sys.path.insert(0, "/home/amy/git/openpilot-slowroads/openpilot/tools/sim/bridge/gta5/map/roadpaint")
import rp

D = "/mnt/e/gta5_scratch/gamefiles_v2/rp_all/tiles"
x0, x1, y0, y1 = map(float, sys.argv[1:5])
step = 0.1
xs = np.arange(x0, x1, step) + step / 2
ys = np.arange(y1, y0, -step) - step / 2
H, W = len(ys), len(xs)
code = np.zeros((3, H, W), np.uint8)
z = np.full((3, H, W), np.nan, np.float32)
for tx in range(int(np.floor(x0 / 64)), int(np.floor(x1 / 64)) + 1):
  for ty in range(int(np.floor(y0 / 64)), int(np.floor(y1 / 64)) + 1):
    try:
      c, zz = rp.read_tile(f"{D}/{tx}_{ty}.bin")
    except FileNotFoundError:
      continue
    mc = (xs >= tx * 64) & (xs < tx * 64 + 64)
    mr = (ys >= ty * 64) & (ys < ty * 64 + 64)
    col = ((xs[mc] - tx * 64) / rp.RES).astype(int)
    row = ((ty * 64 + 64 - ys[mr]) / rp.RES).astype(int)
    for l in range(c.shape[0]):
      code[l][np.ix_(mr, mc)] = c[l][np.ix_(row, col)]
      z[l][np.ix_(mr, mc)] = zz[l][np.ix_(row, col)]
# order layers top first per pixel
zk = np.where(code > 0, z, -np.inf)
order = np.argsort(-zk, axis=0)
code = np.take_along_axis(code, order, 0)
pal = np.array([[0, 0, 0], [90, 90, 90], [40, 40, 220], [150, 190, 210], [0, 140, 255]], np.uint8)
def render(c):
  im = pal[c & 7]
  p = c >> 3
  im[p == 1] = (255, 255, 255)
  im[p == 2] = (0, 220, 255)
  return im
im = np.hstack([render(code[0]), np.full((H, 8, 3), 255, np.uint8), render(code[1])])
cv2.imwrite(sys.argv[5], im, [cv2.IMWRITE_JPEG_QUALITY, 90])
print(sys.argv[5], im.shape, "layers used:", int((code[1] > 0).mean() * 1000) / 10, "% px have a 2nd layer")
