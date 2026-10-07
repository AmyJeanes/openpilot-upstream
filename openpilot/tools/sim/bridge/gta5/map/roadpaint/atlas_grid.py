"""atlas_grid.py <out.png> <tex.rgba>...: textures over grey with a 0.1 u/v grid (u across, v down) for cell tables."""
import sys, pathlib
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from tex2png import load

S = 400
fs = sys.argv[2:]
cols = 3
M = Image.new('RGB', (cols * (S + 30), ((len(fs) + cols - 1) // cols) * (S + 40)), (30, 30, 30))
dr = ImageDraw.Draw(M)
for i, f in enumerate(fs):
  a = load(pathlib.Path(f))
  im = Image.fromarray(a, 'RGBA').resize((S, S))
  bg = Image.new('RGBA', (S, S), (90, 90, 90, 255)); bg.alpha_composite(im)
  x, y = (i % cols) * (S + 30) + 25, (i // cols) * (S + 40) + 30
  M.paste(bg.convert('RGB'), (x, y))
  for k in range(11):
    c = (255, 0, 0) if k % 5 == 0 else (0, 160, 255)
    dr.line([(x + k * S / 10, y), (x + k * S / 10, y + S)], fill=c)
    dr.line([(x, y + k * S / 10), (x + S, y + k * S / 10)], fill=c)
    dr.text((x + k * S / 10 - 4, y - 12), f'{k}', fill=(255, 255, 0))
    dr.text((x - 18, y + k * S / 10 - 5), f'{k}', fill=(255, 255, 0))
  dr.text((x, y + S + 2), pathlib.Path(f).name[:60], fill=(255, 255, 255))
M.save(sys.argv[1])
