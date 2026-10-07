"""compare.py <gamefiles survey.jsonl> <camera survey.jsonl>: per cross-section of the camera survey, the nearest
game-file section on the same link (in the camera's direction), its marks, and matched-line offset differences."""
import json, sys, collections
import numpy as np

gf = collections.defaultdict(list)
for l in open(sys.argv[1]):
  o = json.loads(l)
  gf[(o['a'], o['b'])].append(o)

diffs = collections.defaultdict(list)
for l in open(sys.argv[2]):
  c = json.loads(l)
  key, flip = (c['a'], c['b']), 1
  if key not in gf:
    key, flip = (c['b'], c['a']), -1
  if key not in gf:
    continue
  s = c['s'] if flip == 1 else gf[key][0]['len'] - c['s']
  g = min(gf[key], key=lambda o: abs(o['s'] - s))
  gm = sorted(((flip * m['offset'], m['colour'], m['type']) for m in g['marks'] if m['cover'] >= 0.2), key=lambda t: t[0])
  cm = sorted(((m['offset'], m['colour'], m['type']) for m in c['marks'] if m['type'] != 'kerb_edge' and m['conf'] >= 0.5), key=lambda t: t[0])
  gk = [None if v is None else flip * v for v in (g['kerbs']['left'], g['kerbs']['right'])][::flip]
  gs = [None if v is None else flip * v for v in (g['kerb_step']['left'], g['kerb_step']['right'])][::flip]
  print(f"{c['a']}-{c['b']} s={c['s']:.1f} (gf s={g['s']:.1f})")
  print('   camera:', ' '.join(f'{o:+.2f}{col[0]}{t[0]}' for o, col, t in cm), '| kerbs', c['kerbs']['left'], c['kerbs']['right'])
  print('   files :', ' '.join(f'{o:+.2f}{col[0]}{t[0]}' for o, col, t in gm), '| kerbs', gk)
  for o, col, t in cm:
    best = min((x for x in gm if x[1] == col), key=lambda x: abs(x[0] - o), default=None)
    if best and abs(best[0] - o) < 0.6:
      diffs[key].append(best[0] - o)
  for side, ck, k, ks in (('L', c['kerbs']['left'], gk[0], gs[0]), ('R', c['kerbs']['right'], gk[1], gs[1])):
    if ck is not None and ks is not None and abs(ck - ks) < 1.5:
      diffs[('step',) + key].append(ks - ck)
    if ck is not None and k is not None and abs(ck - k) < 1.5:
      diffs[('kerb',) + key].append(k - ck)
print()
for k, v in diffs.items():
  v = np.array(v)
  print(k, f'n={len(v)} mean {v.mean():+.2f} median |d| {np.median(abs(v)):.2f} max |d| {abs(v).max():.2f}')
allv = np.concatenate([np.array(v) for k, v in diffs.items() if k[0] not in ('kerb', 'step')])
kv = np.concatenate([np.array(v) for k, v in diffs.items() if k[0] == 'kerb'])
sv = np.concatenate([np.array(v) for k, v in diffs.items() if k[0] == 'step'])
print(f'kerb steps: n={len(sv)} median |d| {np.median(abs(sv)):.2f} mean {sv.mean():+.2f}')
print(f'lines: n={len(allv)} median |d| {np.median(abs(allv)):.2f} p90 {np.percentile(abs(allv), 90):.2f}; kerbs: n={len(kv)} median |d| {np.median(abs(kv)):.2f} p90 {np.percentile(abs(kv), 90):.2f}')

# recall/precision within the files' road (between its asphalt edges), lines only
tot = hit = ftot = fhit = 0
miss, extra = collections.Counter(), collections.Counter()
for l in open(sys.argv[2]):
  c = json.loads(l)
  key, flip = (c['a'], c['b']), 1
  if key not in gf:
    key, flip = (c['b'], c['a']), -1
  if key not in gf:
    continue
  s = c['s'] if flip == 1 else gf[key][0]['len'] - c['s']
  g = min(gf[key], key=lambda o: abs(o['s'] - s))
  lo, hi = sorted(flip * v for v in (g['kerbs']['left'] or -99, g['kerbs']['right'] or 99))
  gm = [(flip * m['offset'], m['colour']) for m in g['marks'] if m['cover'] >= 0.2]
  cm = [(m['offset'], m['colour']) for m in c['marks'] if m['type'] != 'kerb_edge' and m['conf'] >= 0.5 and lo + 0.3 < m['offset'] < hi - 0.3]
  for o, col in cm:
    tot += 1
    if any(cc == col and abs(x - o) < 0.3 for x, cc in gm): hit += 1
    else: miss[key] += 1
  for o, col in gm:
    if lo + 0.3 < o < hi - 0.3:
      ftot += 1
      if any(cc == col and abs(x - o) < 0.3 for x, cc in cm): fhit += 1
      else: extra[key] += 1
print(f'camera lines found in files: {hit}/{tot}; file lines found by camera: {fhit}/{ftot}')
print('missed by files:', dict(miss)); print('not seen by camera:', dict(extra))
