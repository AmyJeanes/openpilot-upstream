"""show_regions.py regions.npz out.jpg: regions.py's regions, each its own colour, walls white/yellow; left the top
surface at each pixel, right the one below it."""
import sys
import numpy as np
import cv2

R = np.load(sys.argv[1])
lab, z, wall = R["lab"], R["z"], R["wall"]
L, H, W = lab.shape
rng = np.random.default_rng(1)
pal = rng.integers(60, 230, (lab.max() + 1, 3)).astype(np.uint8)
pal[0] = 0
zk = np.where(np.isnan(z), -np.inf, z)
order = np.argsort(-zk, axis=0)
ims = []
for rank in (0, 1):
  li = order[rank]
  pick = lambda a: np.take_along_axis(a, li[None], 0)[0]
  lb, wl, valid = pick(lab), pick(wall), pick(zk) > -np.inf
  im = pal[lb]
  im[wl == 1] = (255, 255, 255)
  im[wl == 2] = (0, 220, 255)
  im[~valid] = 0
  ims.append(im)
cv2.imwrite(sys.argv[2], np.hstack([ims[0], np.full((H, 8, 3), 255, np.uint8), ims[1]]), [cv2.IMWRITE_JPEG_QUALITY, 90])
