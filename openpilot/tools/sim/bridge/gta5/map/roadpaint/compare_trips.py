"""compare_trips.py <gamefiles survey.jsonl> <camera survey.jsonl> <out disagreements.jsonl> [--img dir --tiles dir]

Matches each camera cross-section to the game-file section nearest it on the same link (either direction), and lists
what differs, one JSON line per disagreement:
  issue: offset (same line, > 0.25 m apart) | type (solid/dashed/double differ) | camera_only (a confident camera
         line the files don't have, inside the files' asphalt) | files_only (a files line the camera doesn't report) |
         arrow (kinds differ) | kerb (> 0.5 m)
  with both sides' values, the camera's image and, with --img, a top-down render of the files at the spot.
Prints a summary (match rate, offset statistics) to stderr.
"""
import argparse, json, sys, pathlib, collections
import numpy as np


def norm_type(t):
  return {'edge_line': 'solid', 'kerb_edge': None}.get(t, t)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('gf'); ap.add_argument('cam'); ap.add_argument('out')
  ap.add_argument('--img'); ap.add_argument('--tiles')
  a = ap.parse_args()
  gf = collections.defaultdict(list)
  for l in open(a.gf):
    o = json.loads(l)
    gf[(o['a'], o['b'])].append(o)
  out = open(a.out, 'w')
  stats = collections.Counter(); diffs = []
  render = None
  if a.img:
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    import rp
    from PIL import Image, ImageDraw
    T = rp.Tiles(a.tiles, cache=16)
    pathlib.Path(a.img).mkdir(parents=True, exist_ok=True)
    def render(x, y, z, heading, name):
      half, res = 18.0, 0.05
      n = int(2 * half / res)
      xs = x - half + (np.arange(n) + 0.5) * res; ys = y + half - (np.arange(n) + 0.5) * res
      X, Y = np.meshgrid(xs, ys)
      code, _ = T.sample(X, Y, z)
      cls, paint = code & 7, code >> 3
      img = np.full((n, n, 3), 20, np.uint8)
      img[cls == 1] = (60, 60, 60); img[cls == 2] = (150, 60, 60); img[cls == 3] = (90, 90, 120); img[cls == 4] = (110, 90, 60)
      img[paint == 1] = 255; img[paint == 2] = (255, 200, 0)
      im = Image.fromarray(img); dr = ImageDraw.Draw(im)
      c = n / 2
      dx, dy = np.cos(heading), np.sin(heading)
      dr.line([(c - dy * 16 / res, c - dx * 16 / res), (c + dy * 16 / res, c + dx * 16 / res)], fill=(0, 255, 255))  # the section
      dr.line([(c, c), (c + dx * 4 / res, c - dy * 4 / res)], fill=(255, 0, 0), width=3)  # travel direction
      p = pathlib.Path(a.img) / name
      im.save(p)
      return str(p)
  for l in open(a.cam):
    c = json.loads(l)
    key, flip = (c['a'], c['b']), 1
    if key not in gf:
      key, flip = (c['b'], c['a']), -1
    if key not in gf:
      stats['no_link'] += 1
      continue
    if c.get('dir', 'ab') != 'ab':
      flip = -flip  # camera looked along b->a of its own record
    secs = gf[key]
    L = secs[0]['len']
    s = c['s'] if flip == 1 else L - c['s']
    g = min(secs, key=lambda o: abs(o['s'] - s))
    if abs(g['s'] - s) > 3:
      stats['far'] += 1
      continue
    stats['matched'] += 1
    # game-file marks in the camera's frame
    gm = [dict(m, offset=flip * m['offset']) for m in g['marks']]
    lo, hi = sorted(v * flip for v in (g['kerbs']['left'] if g['kerbs']['left'] is not None else -16, g['kerbs']['right'] if g['kerbs']['right'] is not None else 16))
    cm = [m for m in c['marks'] if norm_type(m['type']) and m.get('conf', 1) >= 0.5]
    base = {'a': c['a'], 'b': c['b'], 's': c['s'], 'dir': c.get('dir', 'ab'), 'x': c.get('x'), 'y': c.get('y'),
            'files_s': g['s'], 'img': c.get('img'), 'shot': c.get('shot')}
    found = []
    def emit(issue, cam, files):
      found.append(dict(base, issue=issue, camera=cam, files=files))
    used = set()
    for m in cm:
      best = min((x for x in gm if x['colour'] == m['colour']), key=lambda x: abs(x['offset'] - m['offset']), default=None)
      if best is not None and abs(best['offset'] - m['offset']) < 0.6:
        used.add(id(best))
        d = best['offset'] - m['offset']
        diffs.append(d)
        if abs(d) > 0.25:
          emit('offset', m, best)
        ct, ft = norm_type(m['type']), norm_type(best['type'])
        if (ct in ('dashed', 'solid') and ft in ('dashed', 'solid') and ct != ft and m.get('conf', 1) >= 0.75) or (('double' in ct) != ('double' in ft) and m.get('conf', 1) >= 0.75):
          emit('type', m, best)
      elif lo + 0.3 < m['offset'] < hi - 0.3 and m.get('conf', 1) >= 0.75:
        emit('camera_only', m, None)
    for x in gm:
      if id(x) not in used and lo + 0.3 < x['offset'] < hi - 0.3 and x.get('cover', 1) >= 0.2:
        emit('files_only', None, x)
    ca = sorted((ar['offset'], ar['kind']) for ar in c.get('arrows', []) if ar.get('dir', 'ab') == 'ab')
    fa = sorted((flip * ar['offset'], ar['kind']) for ar in g.get('arrows', []) if (ar.get('dir') == 'ab') == (flip == 1))
    for o, k in ca:
      m = [fk for fo, fk in fa if abs(fo - o) < 2.5]
      if m and k not in m and not (k.replace('through;', '') in [x.replace('through;', '') for x in m]):
        emit('arrow', {'offset': o, 'kind': k}, {'kinds': m})
    for side, ck in (('left', c['kerbs'].get('left')), ('right', c['kerbs'].get('right'))):
      fk = (g['kerbs'][side] if flip == 1 else g['kerbs']['right' if side == 'left' else 'left'])
      if ck is not None and fk is not None and abs(flip * fk - ck) > 0.5:
        emit('kerb', {'side': side, 'offset': ck}, {'offset': flip * fk, 'kerb_step': g['kerb_step']})
    for f in found:
      stats[f['issue']] += 1
      if render is not None:
        name = f"{f['a'].replace(':', '_')}-{f['b'].replace(':', '_')}_{f['s']:.1f}_{f['dir']}.png"
        f['files_img'] = str(pathlib.Path(a.img) / name)
        if not pathlib.Path(f['files_img']).exists():
          # heading of the camera's travel: from the files section neighbours
          k = secs.index(g)
          nb = secs[min(k + 1, len(secs) - 1)] if len(secs) > 1 else g
          pv = secs[max(k - 1, 0)]
          hd = np.arctan2(nb['y'] - pv['y'], nb['x'] - pv['x']) if nb is not pv else 0.0
          if flip == -1:
            hd += np.pi
          render(g['x'], g['y'], g['z'], hd, name)
      out.write(json.dumps(f) + '\n')
  d = np.abs(np.array(diffs)) if diffs else np.zeros(1)
  print(json.dumps({**stats, 'lines_matched': len(diffs), 'median_abs_diff': round(float(np.median(d)), 3), 'p90_abs_diff': round(float(np.percentile(d, 90)), 3)}), file=sys.stderr)


if __name__ == '__main__':
  main()
