"""features.py <roadpaint out dir> <atlas_cells.json>: typed road markings from the atlas decals and crossings.

Each atlas decal quad group that maps one feature cell becomes one feature: its kind from the cell (arrows resolved
to through / left / right / through;left / through;right by the handedness of the decal's UV mapping in the world),
its centre, heading (degrees anticlockwise from east, the way the marking faces) and size. Crossing textures become
crossing polygons. Writes <dir>/features.jsonl.
"""
import json, math, sys, pathlib
import numpy as np


def atlas_key(name, keys):
  best = None
  for k in keys:
    if name.startswith(k) and (best is None or len(k) > len(best)):
      best = k
  return best


def groups(I, sel):
  """Triangles (indices into I rows) in sel, grouped by shared vertices."""
  parent = {}
  def find(a):
    while parent.setdefault(a, a) != a:
      parent[a] = parent[parent[a]]
      a = parent[a]
    return a
  for t in sel:
    a, b, c = I[t]
    for x, y in ((a, b), (b, c)):
      ra, rb = find(x), find(y)
      if ra != rb:
        parent[ra] = rb
  out = {}
  for t in sel:
    out.setdefault(find(I[t][0]), []).append(t)
  return list(out.values())


def hull(P):
  P = sorted(set(map(tuple, P)))
  if len(P) < 3:
    return P
  def half(pts):
    h = []
    for p in pts:
      while len(h) >= 2 and (h[-1][0] - h[-2][0]) * (p[1] - h[-2][1]) - (h[-1][1] - h[-2][1]) * (p[0] - h[-2][0]) <= 0:
        h.pop()
      h.append(p)
    return h
  lo, up = half(P), half(P[::-1])
  return lo[:-1] + up[:-1]


AXIS = {'+u': (0, 1), '-u': (0, -1), '+v': (1, 1), '-v': (1, -1)}


def main():
  d = pathlib.Path(sys.argv[1])
  table = {k: v for k, v in json.load(open(sys.argv[2])).items() if not k.startswith('_')}
  keys = list(table)
  out = open(d / 'features.jsonl', 'w')
  n = {}
  for line in open(d / 'decals.jsonl'):
    g = json.loads(line)
    V = np.array(g['v'], float)
    I = np.array(g['i'], int).reshape(-1, 3)
    name = g['tex']
    if 'crossing' in name:
      for grp in groups(I, range(len(I))):
        idx = np.unique(I[grp].ravel())
        h = hull(np.round(V[idx, :2], 2).tolist())
        c = V[idx, :3].mean(0)
        rec = {'kind': 'crossing', 'x': round(c[0], 2), 'y': round(c[1], 2), 'z': round(c[2], 2), 'tex': name, 'poly': h}
        out.write(json.dumps(rec) + '\n'); n['crossing'] = n.get('crossing', 0) + 1
      continue
    key = atlas_key(name, keys)
    cells = table.get(key, {}).get('features', []) if key else []
    for cell in cells:
      u0, u1, v0, v1 = cell['u0'], cell['u1'], cell['v0'], cell['v1']
      sel = [t for t in range(len(I)) if (V[I[t], 3] >= u0 - 0.01).all() and (V[I[t], 3] <= u1 + 0.01).all()
             and (V[I[t], 4] >= v0 - 0.01).all() and (V[I[t], 4] <= v1 + 0.01).all()]
      for grp in groups(I, sel):
        idx = np.unique(I[grp].ravel())
        if len(idx) < 3:
          continue
        A = np.c_[V[idx, 3:5], np.ones(len(idx))]
        M, res, rank, _ = np.linalg.lstsq(A, V[idx, :3], rcond=None)  # world xyz = [u v 1] @ M
        if rank < 3:
          continue
        du, dv = M[0, :2], M[1, :2]
        ai, sg = AXIS[cell.get('dir', '-v')]
        f = (du if ai == 0 else dv) * sg
        cen = np.array([(u0 + u1) / 2, (v0 + v1) / 2, 1]) @ M
        # the marking's extent: corners of the cell's UV range actually used
        uvmin, uvmax = V[idx, 3:5].min(0), V[idx, 3:5].max(0)
        ext_u = np.linalg.norm(du) * (uvmax[0] - uvmin[0]); ext_v = np.linalg.norm(dv) * (uvmax[1] - uvmin[1])
        length, width = (ext_u, ext_v) if ai == 0 else (ext_v, ext_u)
        kind = cell['kind']
        rec = {'kind': kind}
        if kind == 'arrow':
          turn = cell.get('turn')
          if turn:
            ti, ts = AXIS[turn]
            tv = (du if ti == 0 else dv) * ts
            right = f[0] * tv[1] - f[1] * tv[0] < 0
            side = 'right' if right else 'left'
            rec['arrow'] = f'through;{side}' if cell.get('through') else side
          else:
            rec['arrow'] = 'through'
        rec.update({'x': round(float(cen[0]), 2), 'y': round(float(cen[1]), 2), 'z': round(float(cen[2]), 2),
                    'heading': round(math.degrees(math.atan2(f[1], f[0])), 1), 'len': round(float(length), 2),
                    'wid': round(float(width), 2), 'tex': name})
        out.write(json.dumps(rec) + '\n')
        k = rec.get('arrow', kind)
        n[k] = n.get(k, 0) + 1
  print(json.dumps(n, sort_keys=True))


if __name__ == '__main__':
  main()
