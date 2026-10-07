"""Convert extracted .rgba textures (CodeWalker's BGRA order) to PNG, plus a labelled montage over grey for viewing."""
import re, sys, pathlib
import numpy as np
from PIL import Image, ImageDraw

def load(f):
  m = re.search(r'_(\d+)x(\d+)_', f.name)
  w, h = int(m[1]), int(m[2])
  return np.frombuffer(f.read_bytes(), np.uint8)[:w * h * 4].reshape(h, w, 4)[:, :, [2, 1, 0, 3]]

if __name__ == '__main__':
  out, pat, dirs = sys.argv[1], re.compile(sys.argv[2], re.I), sys.argv[3:]
  tiles = {}
  for d in dirs:
    for f in sorted(pathlib.Path(d).glob('*.rgba')):
      if pat.search(f.name) and f.name not in tiles:
        tiles[f.name] = load(f)
  S = 256
  cols = 4
  rows = (len(tiles) + cols - 1) // cols
  M = Image.new('RGB', (cols * S, rows * (S + 14)), (90, 90, 90))
  dr = ImageDraw.Draw(M)
  for k, (n, a) in enumerate(tiles.items()):
    im = Image.fromarray(a, 'RGBA').resize((S, S))
    bg = Image.new('RGBA', (S, S), (60, 60, 60, 255))
    bg.alpha_composite(im)
    x, y = (k % cols) * S, (k // cols) * (S + 14)
    M.paste(bg.convert('RGB'), (x, y + 14))
    dr.text((x + 2, y + 1), n[:40], fill=(255, 255, 0))
  M.save(out)
