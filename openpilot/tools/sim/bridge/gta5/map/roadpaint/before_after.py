"""before_after.py <old rp dir> <new rp dir> <out dir>: before/after figures of the re-extraction at the checked places."""
import sys, json, pathlib, collections
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import rp

old, new, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
SP = pathlib.Path('/mnt/e/gta5_audit/map_check_20261008/spots')
SPOTS = [  # name, centre, deck height, half size m, in-game shot
  ('fig1_great_ocean_w', -3080, 766, 19.4, 35, SP / 'G1a_great_ocean_w_top_off.jpg', 'G1a Great Ocean Hwy (-3080,766)'),
  ('fig2_great_ocean_n', 1200, 6486, 19.9, 30, SP / 'G1b_great_ocean_n_top_off.jpg', 'G1b Great Ocean Hwy north (1200,6486)'),
  ('fig3_vinewood', 258, 172, 103.8, 40, pathlib.Path('/mnt/e/gta5_audit/map_audit/survey_trips_39254_68.jpg'), 'Vinewood Blvd west of Alta (258,172)'),
  ('fig4_del_perro', -999, -558, 17.4, 35, SP / 'G2b_delperro_999_top_off.jpg', 'G2b Del Perro Fwy (-999,-558)'),
]
try:
  FONT = ImageFont.truetype('DejaVuSans.ttf', 22)
except OSError:
  FONT = ImageFont.load_default()


def polylines(d, cx, cy, half, zref):
  res = []
  for line in open(d / 'polylines.jsonl'):
    L = json.loads(line)
    P = np.array(L['pts'])
    if ((np.abs(P[:, 0] - cx) < half) & (np.abs(P[:, 1] - cy) < half)).any() and np.abs(P[:, 2] - zref).min() < 3:
      res.append(L)
  return res


def render(d, cx, cy, zref, half, res=0.05):
  n = int(2 * half / res)
  xs = cx - half + (np.arange(n) + 0.5) * res
  ys = cy + half - (np.arange(n) + 0.5) * res
  X, Y = np.meshgrid(xs, ys)
  code, _ = rp.Tiles(d / 'tiles').sample(X, Y, zref)
  cls = code & 7
  img = np.full((n, n, 3), 25, np.uint8)
  img[cls == rp.ROAD] = (70, 70, 70); img[cls == rp.KERB] = (140, 70, 70); img[cls == rp.WALK] = (95, 95, 120); img[cls == rp.GUTTER] = (110, 90, 60)
  im = Image.fromarray(img); dr = ImageDraw.Draw(im)
  for L in polylines(d, cx, cy, half, zref):
    col = (255, 205, 0) if L['colour'] == 'yellow' else (255, 255, 255)
    P = L['pts']
    xy = [((x - cx + half) / res, (cy + half - y) / res) for x, y, _ in P]
    if L['style'] == 'dashed' and L.get('dashes'):
      cum = np.r_[0, np.cumsum(np.hypot(np.diff([p[0] for p in P]), np.diff([p[1] for p in P])))]
      for a, b in L['dashes']:
        seg = [xy[i] for i in range(len(P)) if a <= cum[i] <= b]
        ia, ib = np.interp(a, cum, np.arange(len(P))), np.interp(b, cum, np.arange(len(P)))
        pa = tuple(np.interp(ia, np.arange(len(P)), np.array(xy)[:, k]) for k in (0, 1))
        pb = tuple(np.interp(ib, np.arange(len(P)), np.array(xy)[:, k]) for k in (0, 1))
        dr.line([pa] + seg + [pb], fill=col, width=5)
    else:
      dr.line(xy, fill=col, width=5)
  return im


def label(im, text):
  dr = ImageDraw.Draw(im)
  dr.rectangle([0, 0, im.width, 34], fill=(0, 0, 0))
  dr.text((8, 5), text, fill=(255, 255, 0), font=FONT)
  return im


out.mkdir(parents=True, exist_ok=True)
S = 720
for name, cx, cy, z, half, shot, title in SPOTS:
  panels = []
  if shot.exists():
    g = Image.open(shot).convert('RGB'); g.thumbnail((S * 16 // 9, S))
    panels.append(label(g, 'in game (top-down camera; not north-up)'))
  for tag, d in (('before (gamefiles)', old), ('after (gamefiles_v2)', new)):
    panels.append(label(render(d, cx, cy, z, half).resize((S, S)), f'{tag}: lines from polylines.jsonl'))
  W = sum(p.width for p in panels) + 10 * (len(panels) - 1)
  M = Image.new('RGB', (W, S + 40), (15, 15, 15))
  x = 0
  for p in panels:
    M.paste(p, (x, 40)); x += p.width + 10
  ImageDraw.Draw(M).text((8, 8), f'{title}, {2 * half:.0f} m square, north up; white/yellow = extracted paint lines', fill=(255, 255, 255), font=FONT)
  M.save(out / f'{name}.png')
  print(name)
