"""survey_rp.py <roadpaint out dir> <out.jsonl> [--step m] [--links a-b,...] [--box x0,y0,x1,y1] [--workers n]

Survey-format cross-sections of every GTA vehicle link (or the given ones) from the game-file extraction: the exact
paint polylines (polylines.jsonl), typed markings (features.jsonl) and the height-layered tiles (for kerbs). One JSON
line per section, offsets in metres right of travel a->b, in the camera survey's schema plus:
  "src": "gamefiles", "z": link height used to pick the deck,
  marks[].{width, cover (painted share over 12 m), line (polyline id), pair (double line partner's offset)},
  marks[].type also "double_dashed" / "solid_dashed" (the two lines of a double differ; first is the left one),
  "kerbs": asphalt edge (where the road surface texture ends), "kerb_step": visual-mesh kerb (8 cm rise),
  "hatched": [[o0, o1]] spans of chevron/diagonal hatching, "stop_lines": [{s, from, to}],
  "features": [{kind, s, offset, heading_rel}] crossings / text / parking markers near the section.
"""
import argparse, json, math, sys, pathlib, collections
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import rp

PED_SPECIALS = {10, 14, 15, 16}  # as ynd_to_osm.node_ok
GRID = 32.0


def load_links(paths):
  nodes, last = {}, {}
  for line in open(paths):
    o = json.loads(line)
    if o['t'] == 'n':
      nodes[(o['a'], o['i'])] = o
    elif o['t'] == 'l':
      last[((o['a'], o['i']), (o['ta'], o['ti']))] = o
  ok = lambda n: (n['f'][1] >> 3) not in PED_SPECIALS and not n['f'][2] & 32
  out = {}
  for (ka, kb), l in last.items():
    if ka not in nodes or kb not in nodes or not ok(nodes[ka]) or not ok(nodes[kb]):
      continue
    fwd, back = (l['f'][2] >> 5) & 7, (l['f'][2] >> 2) & 7
    if fwd + back == 0:
      continue
    key = (ka, kb) if ka < kb else (kb, ka)
    out.setdefault(key, True)
  xyz = {k: (n['x'], n['y'], n['z']) for k, n in nodes.items()}
  return [(a, b) for a, b in out], xyz


class Lines:
  def __init__(self, path):
    self.L = [json.loads(l) for l in open(path)]
    segs = []
    for L in self.L:
      P = np.array(L['pts'], float)
      cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
      for k in range(len(P) - 1):
        segs.append((P[k, 0], P[k, 1], P[k, 2], P[k + 1, 0], P[k + 1, 1], P[k + 1, 2], L['id'], cum[k], cum[k + 1]))
    self.S = np.array(segs, float).reshape(-1, 9)
    self.grid = collections.defaultdict(list)
    for i, s in enumerate(self.S):
      for gx in range(int(math.floor(min(s[0], s[3]) / GRID)), int(math.floor(max(s[0], s[3]) / GRID)) + 1):
        for gy in range(int(math.floor(min(s[1], s[4]) / GRID)), int(math.floor(max(s[1], s[4]) / GRID)) + 1):
          self.grid[(gx, gy)].append(i)
    self.byid = {L['id']: L for L in self.L}

  def near(self, x0, y0, x1, y1):
    ids = set()
    for gx in range(int(math.floor(x0 / GRID)), int(math.floor(x1 / GRID)) + 1):
      for gy in range(int(math.floor(y0 / GRID)), int(math.floor(y1 / GRID)) + 1):
        ids.update(self.grid.get((gx, gy), ()))
    return np.array(sorted(ids), int)


def painted_at(L, s):
  if L['style'] != 'dashed':
    return True
  return any(a - 0.05 <= s <= b + 0.05 for a, b in L['dashes'])


def coverage(L, s, half=6.0):
  """painted share of the line within +-half m of distance s along it"""
  # a short line on its own (one dash whose neighbours weren't joined to it) is mostly gap over the window
  dashes = L['dashes'] if L['style'] == 'dashed' else [[0, L['len']]]
  lo, hi = s - half, s + half
  if L['len'] > 2 * half:  # a long line's own ends aren't gaps
    lo, hi = max(0, lo), min(L['len'], hi)
  return sum(max(0, min(b, hi) - max(a, lo)) for a, b in dashes) / (hi - lo)


G = {}


def init(d):
  G['lines'] = Lines(d / 'polylines.jsonl')
  G['tiles'] = rp.Tiles(d / 'tiles', cache=24)
  F = [json.loads(l) for l in open(d / 'features.jsonl')] if (d / 'features.jsonl').exists() else []
  G['feat'] = F
  G['fxy'] = np.array([[f['x'], f['y'], f['z']] for f in F]).reshape(-1, 3)


def section(A, B, s, step, zl):
  lines, S = G['lines'], G['lines'].S
  d = (B[:2] - A[:2]); L = np.linalg.norm(d); d = d / L
  right = np.array([d[1], -d[0]])
  p = A[:2] + d * s
  q0, q1 = p - right * 16, p + right * 16
  cand = lines.near(min(q0[0], q1[0]) - 1, min(q0[1], q1[1]) - 1, max(q0[0], q1[0]) + 1, max(q0[1], q1[1]) + 1)
  hits = []
  hatch = []
  if len(cand):
    C = S[cand]
    a, b = C[:, 0:2], C[:, 3:5]
    e = b - a
    r = q1 - q0
    den = e[:, 0] * r[1] - e[:, 1] * r[0]
    with np.errstate(all='ignore'):
      t = ((q0[0] - a[:, 0]) * r[1] - (q0[1] - a[:, 1]) * r[0]) / den      # along the segment
      u = ((q0[0] - a[:, 0]) * e[:, 1] - (q0[1] - a[:, 1]) * e[:, 0]) / den  # along the section, 0..1
    z = C[:, 2] + t * (C[:, 5] - C[:, 2])
    m = (np.abs(den) > 1e-9) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1) & (np.abs(z - zl) < 2.0)
    for i in np.where(m)[0]:
      lid = int(C[i, 6]); Lr = lines.byid[lid]
      sl = C[i, 7] + t[i] * (C[i, 8] - C[i, 7])
      el = e[i] / max(np.linalg.norm(e[i]), 1e-9)
      cosang = abs(float(el @ d))
      off = -16 + 32 * float(u[i])
      if cosang < 0.95 and cosang > 0.3:
        hatch.append(off)  # chevrons and diagonals cross the section at an angle
        continue
      if cosang <= 0.3:
        continue
      cov = coverage(Lr, sl)  # solid vs dashed here, not over the whole line (a bay line can start dashed)
      hits.append({'offset': round(off, 2), 'colour': Lr['colour'], 'style': 'solid' if cov >= 0.9 else 'dashed', 'width': Lr['width'],
                   'cover': round(cov, 2), 'line': lid, 'pair': Lr['pair'], 'painted_here': painted_at(Lr, sl)})
  hits.sort(key=lambda h: h['offset'])
  # one hit per line (a polyline can cross twice where it bends); then pair double lines
  seen, uniq = set(), []
  for h in hits:
    if h['line'] in seen:
      continue
    seen.add(h['line']); uniq.append(h)
  marks, used = [], set()
  for i, h in enumerate(uniq):
    if i in used:
      continue
    partner = None
    for j in range(i + 1, len(uniq)):
      g = uniq[j]
      if j in used or g['colour'] != h['colour']:
        continue
      if 0.08 < g['offset'] - h['offset'] < 0.45:
        partner = j
        break
      if g['offset'] - h['offset'] >= 0.45:
        break
    if partner is not None:
      g = uniq[partner]; used.add(partner)
      st = (h['style'], g['style'])
      kind = 'double_solid' if st == ('solid', 'solid') else 'double_dashed' if st == ('dashed', 'dashed') else 'solid_dashed' if st == ('solid', 'dashed') else 'dashed_solid'
      marks.append({'type': kind, 'colour': h['colour'], 'offset': round((h['offset'] + g['offset']) / 2, 2), 'width': round(g['offset'] - h['offset'] + (h['width'] + g['width']) / 2, 2),
                    'cover': min(h['cover'], g['cover']), 'line': [h['line'], g['line']], 'conf': 0.95})
    else:
      kind = 'solid' if h['style'] == 'solid' else 'dashed'
      if h['style'] == 'dashed' and h['cover'] < 0.2:
        kind = 'markers'
      marks.append({'type': kind, 'colour': h['colour'], 'offset': h['offset'], 'width': round(h['width'], 2), 'cover': h['cover'], 'line': h['line'], 'conf': 0.95})
  # kerbs from the tiles along the section at the link's height
  o = np.arange(-16, 16.001, 0.05)
  P = p + o[:, None] * right
  code, zz = G['tiles'].sample(P[:, 0], P[:, 1], zl)
  cls = code & 7
  mid = len(o) // 2
  # start from the nearest road pixel to the link (the link can run over a painted median or island)
  roadi = np.where(cls == rp.ROAD)[0]
  kerbs, steps = [None, None], [None, None]
  if len(roadi):
    c0 = roadi[np.argmin(np.abs(roadi - mid))]
    for side, rng in ((0, range(c0, -1, -1)), (1, range(c0, len(o)))):
      edge = None
      for i in rng:
        if cls[i] != rp.ROAD:  # gutter, kerb, pavement, or no surface at this height
          edge = i; break
      if edge is not None:
        kerbs[side] = round(float(o[edge] + (0.025 if side == 0 else -0.025)), 2)
      sg = -1 if side == 0 else 1
      for i in rng:
        j = i + sg * 6
        if j < 0 or j >= len(o):
          break
        if not np.isnan(zz[i]) and not np.isnan(zz[j]) and zz[j] - zz[i] > 0.08:
          k = max(range(6), key=lambda k: (zz[i + sg * (k + 1)] - zz[i + sg * k]) if not (np.isnan(zz[i + sg * (k + 1)]) or np.isnan(zz[i + sg * k])) else -1)
          steps[side] = round(float(o[i + sg * k] + sg * 0.025), 2)
          break
  # edge lines: the outermost solid white single line within 1.5 m inside the asphalt edge
  for side in (0, 1):
    if kerbs[side] is None:
      continue
    cand = [m for m in marks if m['type'] == 'solid' and m['colour'] == 'white' and 0 <= (m['offset'] - kerbs[0] if side == 0 else kerbs[1] - m['offset']) <= 1.5]
    if cand:
      (min if side == 0 else max)(cand, key=lambda m: m['offset'])['type'] = 'edge_line'
  hat = []
  for x in sorted(hatch):
    if hat and x - hat[-1][1] < 2.5:
      hat[-1][1] = x
    else:
      hat.append([x, x])
  hat = [[round(a, 2), round(b, 2)] for a, b in hat if len([h for h in hatch if a <= h <= b]) >= 2 or b - a > 0]
  return p, marks, kerbs, steps, hat


def feats(A, B, s, step, zl):
  d = (B[:2] - A[:2]); L = np.linalg.norm(d); d = d / L
  right = np.array([d[1], -d[0]])
  p = A[:2] + d * s
  F, X = G['feat'], G['fxy']
  arrows, other = [], []
  if not len(X):
    return arrows, other
  rel = X[:, :2] - p
  along, off = rel @ d, rel @ right
  m = (np.abs(along) <= step / 2) & (np.abs(off) < 16) & (np.abs(X[:, 2] - zl) < 2.5)
  hd = math.degrees(math.atan2(d[1], d[0]))
  for i in np.where(m)[0]:
    f = F[i]
    rec = {'kind': f.get('arrow', f['kind']), 'offset': round(float(off[i]), 2), 's': round(float(s + along[i]), 2)}
    if 'heading' in f:
      dh = (f['heading'] - hd + 180) % 360 - 180
      rec['dir'] = 'ab' if abs(dh) < 60 else 'oncoming' if abs(dh) > 120 else 'cross'
    if f['kind'] == 'arrow':
      rec['conf'] = 0.95
      arrows.append(rec)
    else:
      other.append(rec)
  return arrows, other


def stop_lines(A, B, zl):
  """white solid lines across the link (within 30 deg of perpendicular), wider than 0.25 m, on the link's span"""
  lines, S = G['lines'], G['lines'].S
  d = (B[:2] - A[:2]); L = np.linalg.norm(d); d = d / L
  right = np.array([d[1], -d[0]])
  cand = lines.near(min(A[0], B[0]) - 16, min(A[1], B[1]) - 16, max(A[0], B[0]) + 16, max(A[1], B[1]) + 16)
  out = {}
  for i in cand:
    s0 = S[i]
    Lr = lines.byid[int(s0[6])]
    if Lr['colour'] != 'white' or Lr['style'] != 'solid' or Lr['width'] < 0.25:
      continue
    e = s0[3:5] - s0[0:2]; le = np.linalg.norm(e)
    if le < 1e-3 or abs((e / le) @ d) > 0.5 or abs((s0[2] + s0[5]) / 2 - zl) > 2:
      continue
    for q in (s0[0:2], s0[3:5]):
      rel = q - A[:2]
      a, o = float(rel @ d), float(rel @ right)
      if -1 <= a <= L + 1 and abs(o) < 16:
        r = out.setdefault(int(s0[6]), [a, o, o])
        r[0] = (r[0] + a) / 2; r[1] = min(r[1], o); r[2] = max(r[2], o)
  return [{'s': round(a, 2), 'from': round(o0, 2), 'to': round(o1, 2), 'line': k} for k, (a, o0, o1) in out.items() if o1 - o0 > 1.5]


def do_link(args):
  (ka, kb), A, B, step = args
  A, B = np.array(A, float), np.array(B, float)
  L = float(np.linalg.norm(B[:2] - A[:2]))
  if L < 0.5:
    return []
  n = max(1, int(round(L / step)))
  out = []
  stops = stop_lines(A, B, (A[2] + B[2]) / 2)
  for k in range(n):
    s = (k + 0.5) * L / n
    zl = A[2] + (B[2] - A[2]) * s / L
    p, marks, kerbs, steps, hat = section(A, B, s, L / n, zl)
    arrows, other = feats(A, B, s, L / n, zl)
    rec = {'a': f'{ka[0]}:{ka[1]}', 'b': f'{kb[0]}:{kb[1]}', 's': round(s, 2), 'len': round(L, 2), 'dir': 'ab',
           'x': round(float(p[0]), 2), 'y': round(float(p[1]), 2), 'z': round(float(zl), 2), 'marks': marks, 'arrows': arrows,
           'kerbs': {'left': kerbs[0], 'right': kerbs[1], 'conf': 0.8}, 'kerb_step': {'left': steps[0], 'right': steps[1]},
           'hatched': hat, 'stop_lines': [x for x in stops if abs(x['s'] - s) <= L / n / 2 + 1e-6], 'features': other,
           'bay': False, 'img': None, 't': None, 'src': 'gamefiles'}
    out.append(json.dumps(rec))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('dir'); ap.add_argument('out')
  ap.add_argument('--step', type=float, default=3.0)
  ap.add_argument('--links'); ap.add_argument('--box'); ap.add_argument('--workers', type=int, default=4)
  a = ap.parse_args()
  d = pathlib.Path(a.dir)
  links, xyz = load_links(rp.PATHS)
  if a.links:
    want = set()
    for s in a.links.split(','):
      x, y = s.split('-')
      ka, kb = tuple(map(int, x.split(':'))), tuple(map(int, y.split(':')))
      want.add((ka, kb))
    jobs = [(k, xyz[k[0]], xyz[k[1]], a.step) for k in want]
  else:
    jobs = [(k, xyz[k[0]], xyz[k[1]], a.step) for k in links]
  if a.box:
    x0, y0, x1, y1 = map(float, a.box.split(','))
    jobs = [j for j in jobs if x0 <= (j[1][0] + j[2][0]) / 2 <= x1 and y0 <= (j[1][1] + j[2][1]) / 2 <= y1]
  # neighbouring links together, so the tile cache works
  jobs.sort(key=lambda j: (int(j[1][0] // 256), int(j[1][1] // 256)))
  print(f'{len(jobs)} links', file=sys.stderr)
  with open(a.out, 'w') as f, Pool(a.workers, initializer=init, initargs=(d,)) as pool:
    for i, recs in enumerate(pool.imap(do_link, jobs, chunksize=64)):
      for r in recs:
        f.write(r + '\n')
      if i % 5000 == 0:
        print(f'{i}/{len(jobs)}', file=sys.stderr, flush=True)


if __name__ == '__main__':
  main()
