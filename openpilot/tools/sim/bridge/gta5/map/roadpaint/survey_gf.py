"""Survey-format cross-sections of road paint taken from the game files instead of the in-game camera.

survey_gf.py <extract dir> <out.jsonl> <step m> <area:index>-<area:index> ...

For each link a->b (GTA path node ids), renders the extracted road geometry around it and writes one JSON line per
cross-section every <step> m in the survey's format (offsets right of travel a->b), with "src": "gamefiles".
Arrows come from the marking-atlas cells their decal quads use, so their kind is exact.
"""
import json, math, sys, pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import render as R

# arrow cells of the marking atlases: (u0, u1, v0, v1, kind); arrows point along +u in texture space.
# 'turn+' turns toward +v, 'through_turn-' is straight on with a branch toward -v.
ARROW_CELLS = {
  'im_roadmarkings02im_roadmarkings02_a': [(0.5, 1.0, 0.0, 0.3, 'through'), (0.5, 1.0, 0.28, 0.635, 'turn+'), (0.5, 1.0, 0.63, 1.0, 'through_turn-')],
}


def arrows(d):
  out = []
  for line in open(d / 'geoms.jsonl'):
    g = json.loads(line)
    cells = ARROW_CELLS.get(g['tex'].lower())
    if not cells:
      continue
    V = np.array(g['v'])
    I = np.array(g['i']).reshape(-1, 3)
    for u0, u1, v0, v1, kind in cells:
      tris = [t for t in I if (V[t, 3] >= u0 - 0.01).all() and (V[t, 3] <= u1 + 0.01).all() and (V[t, 4] >= v0 - 0.01).all() and (V[t, 4] <= v1 + 0.01).all()]
      # group into arrows: triangles within 4 m of each other
      groups = []
      for t in tris:
        c = V[t, :2].mean(0)
        for gr in groups:
          if np.linalg.norm(gr['c'] - c) < 4:
            gr['t'].append(t); break
        else:
          groups.append({'c': c, 't': [t]})
      for gr in groups:
        idx = np.unique(np.concatenate(gr['t']))
        uv1 = np.c_[V[idx, 3:5], np.ones(len(idx))]
        M, *_ = np.linalg.lstsq(uv1, V[idx, :2], rcond=None)  # world xy = [u v 1] @ M
        du, dv = M[0], M[1]
        right_of = (du[0] * dv[1] - du[1] * dv[0]) < 0  # +v is to the right of the arrow's direction
        k = kind
        if kind == 'turn+':
          k = 'right' if right_of else 'left'
        elif kind == 'through_turn-':
          k = 'through;left' if right_of else 'through;right'
        cen = np.r_[(u0 + u1) / 2, (v0 + v1) / 2, 1] @ M
        out.append((float(cen[0]), float(cen[1]), math.atan2(du[1], du[0]), k))
  return out


def node_xyz():
  n = {}
  for line in open(R.PATHS):
    o = json.loads(line)
    if o['t'] == 'n':
      n[f"{o['a']}:{o['i']}"] = (o['x'], o['y'], o['z'])
  return n


def main():
  d, out, step = pathlib.Path(sys.argv[1]), sys.argv[2], float(sys.argv[3])
  nodes = node_xyz()
  arr = arrows(d)
  geoms = R.load_geoms(d)
  with open(out, 'w') as f:
    for spec in sys.argv[4:]:
      ia, ib = spec.split('-')
      A, B = np.array(nodes[ia]), np.array(nodes[ib])
      L = float(np.linalg.norm(B[:2] - A[:2]))
      c = (A[:2] + B[:2]) / 2
      half = L / 2 + 20
      r = R.Raster(c[0], c[1], half, 0.025, (A[2] + B[2]) / 2)
      R.rasterise(r, geoms)
      pm = R.paint_mask(r)
      dvec = (B[:2] - A[:2]) / L
      right = np.array([dvec[1], -dvec[0]])
      for s in np.arange(0, L + 1e-6, step):
        # lines over 3 m (exact offsets, keeps double lines and tapers apart), plus those only found over 12 m (a
        # dashed line whose gap spans the 3 m)
        lines, kerbs = R.profile(r, pm, A[:2], B[:2], s, along=1.5)
        far, _ = R.profile(r, pm, A[:2], B[:2], s, along=6.0)
        lines += [l for l in far if not any(abs(l[0] - k[0]) < 0.4 for k in lines)]
        lines.sort()
        # solid vs dashed over 12 m (a dash cycle is ~9 m): the share of rows with this colour within 0.3 m of the
        # line's offset, so a line slightly askew to the link still counts as continuous
        o_ = np.arange(-16, 16.01, 0.02)
        P = np.stack([R.sample(r, pm, A[:2], dvec, right, t, o_) for t in np.arange(s - 6, s + 6, r.res)])
        marks = []
        for o, w, colour, kind, cov in lines:
          code = 1 if colour == 'white' else 2
          band = np.abs(o_ - o) < 0.3
          cov9 = float((P[:, band] == code).any(1).mean())
          if kind != 'double_solid':
            kind = 'solid' if cov9 > 0.85 else 'dashed' if cov9 > 0.2 else 'markers'
          if w > 0.5 and kind != 'double_solid':
            continue  # chevrons/hatching and diagonals crossing the section, not a line along it
          marks.append({'type': kind, 'colour': colour, 'offset': o, 'width': w, 'cover': round(cov9, 2), 'conf': round(min(1.0, 0.5 + cov9 / 2), 2)})
        edge = [k[0] for k in kerbs]  # where the asphalt ends (gutter, kerb or pavement starts)
        p = A[:2] + dvec * s
        near = []
        for x, y, h, k in arr:
          rel = np.array([x, y]) - p
          along, off = float(rel @ dvec), float(rel @ right)
          cosh = math.cos(h - math.atan2(dvec[1], dvec[0]))
          if abs(along) <= step / 2 and abs(off) < 16 and abs(cosh) > 0.7:
            near.append({'offset': round(off, 2), 'kind': k, 's': round(float(s) + along, 2), 'dir': 'ab' if cosh > 0 else 'oncoming', 'conf': 0.95})
        rec = {'a': ia, 'b': ib, 's': round(float(s), 2), 'dir': 'ab', 'marks': marks, 'arrows': near,
               'len': round(L, 2), 'x': round(float(p[0]), 2), 'y': round(float(p[1]), 2),
               'kerbs': {'left': edge[0], 'right': edge[1], 'conf': 0.8},
               'kerb_step': {'left': kerbs[0][1], 'right': kerbs[1][1]},
               'bay': False, 'img': None, 't': None, 'src': 'gamefiles'}
        f.write(json.dumps(rec) + '\n')
        print(json.dumps(rec))


if __name__ == '__main__':
  main()
