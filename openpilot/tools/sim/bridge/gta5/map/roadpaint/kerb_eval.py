"""kerb_eval.py: the kerb candidates at the hand-measured validation links, side by side.

For each link: the asphalt edge (road texture ends; survey 'kerbs'), the visual-mesh kerb step ('kerb_step'), the
collision edge (where TARMAC ends in the static collision, from ybnprobe's ybn/<site>.jsonl), the camera survey's
kerbs (val45_v2) and the hand measurements (measurements.md), all in metres right of the link a->b at its middle.
"""
import json, collections, sys
import numpy as np

SITES = {  # link: (ybn file, hand measurement: travel from, to, kerb right, kerb left (of that travel))
  ('494:142', '494:169'): ('494_142', (-633.0, -427.0), (-632.0, -408.5), 13.4, -13.8),
  ('465:225', '465:226'): ('465_225', (788.0, -851.2), (800.0, -851.2), 11.1, -11.1),
  ('528:350', '528:357'): ('528_350', (459.5, 282.3), (445.7, 287.5), 5.55, -5.55),
  ('463:986', '463:1015'): ('463_986', (-91.0, -630.7), (-87.3, -619.5), 13.65, -13.6),
  ('527:314', '527:316'): ('527_314', (-94.0, 255.5), (-81.5, 256.7), 10.7, -11.0),
  ('528:209', '528:215'): ('528_209', None, None, None, None),
  ('464:443', '464:468'): ('464_443', None, None, None, None),
  ('686:122', '686:124'): ('686_122', None, None, None, None),
  ('626:276', '626:288'): ('626_276', None, None, None, None),
}
GF = (sys.argv[1].rstrip('/') + '/') if len(sys.argv) > 1 else './'  # the folder with val_rp.jsonl and ybn/


def collision_edges(tris, a, d, right, s, zref):
  P = np.array([t['p'] for t in tris]).reshape(-1, 3, 3)
  M = np.array([t['m'] for t in tris])
  res = []
  for o in np.arange(-16, 16.001, 0.05):
    x, y = a + d * s + right * o
    x0, y0, x1, y1, x2, y2 = P[:, 0, 0], P[:, 0, 1], P[:, 1, 0], P[:, 1, 1], P[:, 2, 0], P[:, 2, 1]
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    with np.errstate(all='ignore'):
      l0 = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / den
      l1 = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / den
    l2 = 1 - l0 - l1
    hit = np.where((l0 >= 0) & (l1 >= 0) & (l2 >= 0) & (np.abs(den) > 1e-9))[0]
    best = None
    for i in hit:
      z = l0[i] * P[i, 0, 2] + l1[i] * P[i, 1, 2] + l2[i] * P[i, 2, 2]
      if abs(z - zref) < 3 and (best is None or z > best[0]):
        best = (z, M[i])
    res.append((o, best[1] if best else None))
  mid = len(res) // 2
  road = lambda m: m is not None and ('TARMAC' in m or 'MANHOLE' in m or 'ROAD' in m)  # manhole covers and road plates sit in the tarmac
  tar = [i for i, (o, m) in enumerate(res) if road(m)]
  if not tar:
    return None, None
  c0 = min(tar, key=lambda i: abs(i - mid))
  left = next((res[i][0] for i in range(c0, -1, -1) if not road(res[i][1])), None)
  right_ = next((res[i][0] for i in range(c0, len(res)) if not road(res[i][1])), None)
  return left, right_


def main():
  gf = collections.defaultdict(list)
  for l in open(GF + 'val_rp.jsonl'):
    o = json.loads(l); gf[(o['a'], o['b'])].append(o)
  cam = collections.defaultdict(list)
  for l in open('/home/amy/gta5test/map_audit/survey/val45_v2.jsonl'):
    o = json.loads(l); cam[(o['a'], o['b'])].append(o)
  nodes = {}
  for l in open('/home/amy/gta5map_lanes/paths.jsonl'):
    o = json.loads(l)
    if o['t'] == 'n':
      nodes[f"{o['a']}:{o['i']}"] = np.array([o['x'], o['y'], o['z']])
  rows = []
  for key, (yb, f, t, kr, kl) in SITES.items():
    A, B = nodes[key[0]], nodes[key[1]]
    d = (B[:2] - A[:2]); L = np.linalg.norm(d); d /= L
    right = np.array([d[1], -d[0]])
    secs = gf[key]
    g = min(secs, key=lambda o: abs(o['s'] - L / 2))
    tris = [json.loads(l) for l in open(f'{GF}ybn/{yb}.jsonl')]
    cl, cr = collision_edges(tris, A[:2], d, right, g['s'], g['z'])
    hand = (None, None)
    if f is not None:
      tv = np.array(t) - np.array(f)
      hand = (kl, kr) if tv @ d > 0 else (-kr, -kl)
    cm = [c['kerbs'] for c in cam[key]] or [c['kerbs'] for c in cam[(key[1], key[0])]]
    flip = 1 if cam[key] else -1
    cams = [(None if k['left'] is None else k['left'] * flip, None if k['right'] is None else k['right'] * flip) for k in cm]
    if flip == -1:
      cams = [(r, l) for l, r in cams]
    camk = tuple(np.nanmedian([np.nan if c[i] is None else c[i] for c in cams]) if cams else np.nan for i in (0, 1))
    rows.append((key, g['kerbs']['left'], g['kerbs']['right'], g['kerb_step']['left'], g['kerb_step']['right'], cl, cr, camk, hand))
  f = lambda v: '   -  ' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:+6.2f}'
  print('link                  asphalt edge L/R   mesh step L/R     collision L/R     camera L/R        hand L/R')
  for key, al, ar, sl, sr, cl, cr, ck, hd in rows:
    print(f'{key[0]}-{key[1]:9s} {f(al)} {f(ar)}    {f(sl)} {f(sr)}    {f(cl)} {f(cr)}    {f(ck[0])} {f(ck[1])}    {f(hd[0])} {f(hd[1])}')
  # agreement with the hand measurements
  for name, idx in (('asphalt edge', (1, 2)), ('mesh step', (3, 4)), ('collision', (5, 6)), ('camera', None)):
    dd = []
    for r in rows:
      hd = r[8]
      vals = (r[7][0], r[7][1]) if idx is None else (r[idx[0]], r[idx[1]])
      for v, h in zip(vals, hd):
        if v is not None and h is not None and not (isinstance(v, float) and np.isnan(v)):
          dd.append(v - h)
    dd = np.array(dd)
    if len(dd):
      print(f'{name:13s} vs hand: n={len(dd)} mean {dd.mean():+.2f} median |d| {np.median(np.abs(dd)):.2f} max |d| {np.abs(dd).max():.2f}')


if __name__ == '__main__':
  main()
