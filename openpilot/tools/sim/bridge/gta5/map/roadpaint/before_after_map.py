"""before_after_map.py <old rp dir> <new rp dir> <out.png>: painted line km by kind before/after, and where paint was added."""
import sys, json, pathlib, collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

old, new, outp = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
CELL = 250.0


def load(d):
  km = collections.Counter(); grid = collections.Counter()
  for l in open(d / 'polylines.jsonl'):
    L = json.loads(l)
    f = L['painted'] if L['style'] == 'dashed' else 1.0
    kind = f"{L['colour']} {'double ' if L['pair'] is not None else ''}{L['style']}"
    km[kind] += L['len'] * f / 1e3
    P = np.array(L['pts'])
    seg = np.hypot(*np.diff(P[:, :2], axis=0).T) * f
    mid = (P[1:, :2] + P[:-1, :2]) / 2
    for (cx, cy), s in zip(np.floor(mid / CELL).astype(int), seg):
      grid[(cx, cy)] += s / 1e3
  return km, grid


ko, go = load(old); kn, gn = load(new)
kinds = sorted(set(ko) | set(kn), key=lambda k: -kn.get(k, 0))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1.1]})
y = np.arange(len(kinds))
GREY, BLUE = '#9aa0a6', '#2a6fdb'
a1.barh(y + 0.2, [ko.get(k, 0) for k in kinds], 0.38, color=GREY, label='before (gamefiles)')
a1.barh(y - 0.2, [kn.get(k, 0) for k in kinds], 0.38, color=BLUE, label='after (gamefiles_v2)')
for i, k in enumerate(kinds):
  a1.text(kn.get(k, 0) + 2, i - 0.2, f"{kn.get(k, 0):.0f} km ({kn.get(k, 0) - ko.get(k, 0):+.0f})", va='center', fontsize=10, color='#222')
a1.set_yticks(y, kinds); a1.invert_yaxis(); a1.set_xlabel('painted km (dashed lines: paint only)')
a1.set_title(f"Painted line km by kind: {sum(ko.values()):.0f} -> {sum(kn.values()):.0f} km")
a1.legend(loc='lower right', frameon=False); a1.spines[['top', 'right']].set_visible(False); a1.grid(axis='x', color='#eee')
cells = set(go) | set(gn)
xs = np.array([c[0] for c in cells]) * CELL + CELL / 2; ys = np.array([c[1] for c in cells]) * CELL + CELL / 2
dv = np.array([gn.get(c, 0) - go.get(c, 0) for c in cells])
tot = np.array([gn.get(c, 0) for c in cells])
a2.scatter(xs, ys, s=4, c='#dddddd', marker='s', linewidths=0)
m = dv > 0.05
sc = a2.scatter(xs[m], ys[m], s=10, c=dv[m], cmap='Blues', vmin=0, vmax=np.percentile(dv[m], 95), marker='s', linewidths=0)
a2.set_aspect('equal'); a2.set_title(f'Paint added per {CELL:.0f} m cell (km); grey = cells with paint')
plt.colorbar(sc, ax=a2, shrink=0.7, label='km added')
for name, (x, yv) in {'Great Ocean Hwy W': (-3080, 766), 'Great Ocean Hwy N': (1200, 6486), 'Vinewood': (258, 170), 'Del Perro Fwy': (-999, -558)}.items():
  a2.annotate(name, (x, yv), (x + 600, yv + 300), fontsize=9, arrowprops=dict(arrowstyle='-', color='#555'))
a2.set_xlabel('x (m)'); a2.set_ylabel('y (m)')
plt.tight_layout(); plt.savefig(outp, dpi=110)
print(outp)
