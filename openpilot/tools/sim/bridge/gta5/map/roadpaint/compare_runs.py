"""compare_runs.py <old rp dir> <new rp dir> <camera trips.jsonl> <out.json>: before/after numbers for a re-extraction.

Painted line km by kind (polylines), share of survey sections with any paint (map-wide and at named spots), the
Vinewood Blvd lane dashes (|offset| 6.6-7.8 m, x 225..290), and recall/precision against the camera survey: a confident
camera line counts as found when the files have a line of its colour within 0.6 m of it on the matched section.
"""
import sys, json, collections, pathlib
import numpy as np

SPOTS = {'G1a Great Ocean Hwy (-3080,766)': (-3080, 766), 'G1b Great Ocean Hwy N (1200,6486)': (1200, 6486),
         'G2b Del Perro Fwy (-999,-558)': (-999, -558), 'Vinewood Blvd (258,170)': (258, 170)}


def poly_km(d):
  km = collections.Counter()
  for l in open(d / 'polylines.jsonl'):
    L = json.loads(l)
    kind = f"{L['colour']} {'double ' if L['pair'] is not None else ''}{L['style']}"
    km[kind] += L['len'] * (L['painted'] if L['style'] == 'dashed' else 1) / 1e3
    km['_line km (dashed counted end to end)'] += L['len'] / 1e3
  return dict(sorted((k, round(v, 1)) for k, v in km.items()))


def load_survey(d):
  S = collections.defaultdict(list)
  for l in open(d / 'survey_gf.jsonl'):
    o = json.loads(l)
    S[(o['a'], o['b'])].append(o)
  return S


def sections_stats(S):
  n = sum(len(v) for v in S.values())
  painted = sum(1 for v in S.values() for o in v if o['marks'])
  spots = {}
  for name, (x, y) in SPOTS.items():
    sec = [o for v in S.values() for o in v if abs(o['x'] - x) < 25 and abs(o['y'] - y) < 25]
    spots[name] = {'sections': len(sec), 'with paint': sum(1 for o in sec if o['marks']),
                   'marks per section': round(np.mean([len(o['marks']) for o in sec]), 2) if sec else 0}
  vin = [o for v in S.values() for o in v if 225 <= o['x'] <= 290 and 150 <= o['y'] <= 200 and abs(o['z'] - 104) < 3]
  dash = [o for o in vin if any(6.6 <= abs(m['offset']) <= 7.8 and m['colour'] == 'white' for m in o['marks'])]
  return {'sections': n, 'with paint %': round(100 * painted / n, 1), 'spots': spots,
          'vinewood x225-290 sections': len(vin), 'vinewood with a white line at |6.6-7.8| m': len(dash)}


def camera(S, cam_path, boxes):
  tot = collections.Counter()
  for l in open(cam_path):
    c = json.loads(l)
    key, flip = (c['a'], c['b']), 1
    if key not in S:
      key, flip = (c['b'], c['a']), -1
    if key not in S:
      continue
    if c.get('dir', 'ab') != 'ab':
      flip = -flip
    secs = S[key]
    s = c['s'] if flip == 1 else secs[0]['len'] - c['s']
    g = min(secs, key=lambda o: abs(o['s'] - s))
    if abs(g['s'] - s) > 3:
      continue
    gm = [dict(m, offset=flip * m['offset']) for m in g['marks']]
    cm = [m for m in c['marks'] if m['type'] not in ('kerb_edge',) and m.get('conf', 1) >= 0.75 and abs(m['offset']) < 15]
    tags = ['all'] + [b for b, (x0, y0, x1, y1) in boxes.items() if x0 <= g['x'] <= x1 and y0 <= g['y'] <= y1]
    for m in cm:
      hit = any(x['colour'] == m['colour'] and abs(x['offset'] - m['offset']) < 0.6 for x in gm)
      for t in tags + [f"camera {m['colour']} {m['type']}"] * (m['type'] in ('dashed', 'solid', 'edge_line', 'double_solid')):
        tot[(t, 'cam')] += 1; tot[(t, 'found')] += hit
    for x in gm:
      if abs(x['offset']) < 15 and x.get('cover', 1) >= 0.2:
        conf = any(m['colour'] == x['colour'] and abs(x['offset'] - m['offset']) < 0.6 for m in c['marks'])
        for t in tags:
          tot[(t, 'files')] += 1; tot[(t, 'confirmed')] += conf
  out = {}
  for t in ['all'] + list(boxes) + sorted({k[0] for k in tot if k[0].startswith('camera ')}):
    if tot[(t, 'cam')]:
      out[t] = {'camera lines': tot[(t, 'cam')], 'recall %': round(100 * tot[(t, 'found')] / tot[(t, 'cam')], 1),
                'file lines': tot[(t, 'files')], 'camera-confirmed %': round(100 * tot[(t, 'confirmed')] / max(1, tot[(t, 'files')]), 1)}
  return out


if __name__ == '__main__':
  old, new, cam, outp = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
  boxes = {'vinewood': (180, 120, 340, 230), 'downtown': (-800, -1200, 500, -300), 'great ocean': (-3300, 0, -1500, 2500)}
  res = {}
  for tag, d in (('before', old), ('after', new)):
    S = load_survey(d)
    res[tag] = {'polyline km': poly_km(d), 'survey': sections_stats(S), 'camera': camera(S, cam, boxes)}
  json.dump(res, open(outp, 'w'), indent=1)
  print(json.dumps(res, indent=1))
