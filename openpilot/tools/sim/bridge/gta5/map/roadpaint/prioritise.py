"""prioritise.py <disagreements.jsonl> <features.jsonl> <out top.jsonl>: the disagreements worth checking in game.

Drops camera-only lines lying on a typed marking (an arrow's or text's edges read as a line), keeps issues seen at
two or more camera sections of the same link at a similar offset (one-off detections are mostly cars, shadows and
glare), and writes one record per (link, issue, offset cluster) with how many sections show it, ranked by count.
Kerb disagreements are summarised separately (they are mostly a definition question, see the report).
"""
import json, sys, collections, os, pathlib
import numpy as np

D = [json.loads(l) for l in open(sys.argv[1])]
F = [json.loads(l) for l in open(sys.argv[2])]
FX = np.array([[f['x'], f['y'], f['z']] for f in F if f['kind'] != 'crossing']).reshape(-1, 3)
paths = pathlib.Path(os.environ.get('GTA5MAP', '~/gta5map')).expanduser() / 'paths.jsonl'
nodes = {}
for l in open(paths):
  o = json.loads(l)
  if o['t'] == 'n':
    nodes[f"{o['a']}:{o['i']}"] = np.array([o['x'], o['y']])


def world(d, off):
  A, B = nodes[d['a']], nodes[d['b']]
  v = (B - A) / max(np.linalg.norm(B - A), 1e-6)
  if d['dir'] != 'ab':
    v = -v
  right = np.array([v[1], -v[0]])
  s = d['s'] if d['dir'] == 'ab' else d['s']
  base = A + (B - A) / max(np.linalg.norm(B - A), 1e-6) * d['s']
  return base + right * off


keep = []
dropped = collections.Counter()
for d in D:
  if d['issue'] == 'camera_only' and len(FX):
    p = world(d, d['camera']['offset'])
    if (np.linalg.norm(FX[:, :2] - p, axis=1) < 3.5).any():
      dropped['on_marking'] += 1
      continue
  keep.append(d)
groups = collections.defaultdict(list)
for d in keep:
  if d['issue'] == 'kerb':
    continue
  side = d['camera'] or d['files']
  off = side.get('offset', 0)
  key = (d['a'], d['b'], d['dir'], d['issue'], (side.get('colour') or ''), round(off / 0.6))
  groups[key].append(d)
top = []
for key, g in groups.items():
  if len(g) < 2:
    dropped['single'] += 1
    continue
  g.sort(key=lambda d: d['s'])
  rep = dict(g[len(g) // 2])
  rep['sections'] = len(g)
  rep['s_range'] = [g[0]['s'], g[-1]['s']]
  top.append(rep)
top.sort(key=lambda d: -d['sections'])
with open(sys.argv[3], 'w') as f:
  for d in top:
    f.write(json.dumps(d) + '\n')
kerb = [d for d in keep if d['issue'] == 'kerb']
kd = np.array([d['files']['offset'] - d['camera']['offset'] for d in kerb]) if kerb else np.zeros(1)
print(json.dumps({'in': len(D), 'kept': len(keep), 'dropped': dropped, 'top': len(top),
                  'top_by_issue': collections.Counter(d['issue'] for d in top),
                  'kerb_disagreements': len(kerb), 'kerb_files_minus_camera_median': round(float(np.median(kd)), 2)}, default=dict))
