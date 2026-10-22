"""scan.py x0 x1 y0 y1 out.jsonl [--step=2]: kerb-to-kerb cross-sections of the road surface along every GTA vehicle
link in an area, square to the paint (the links zigzag across lanes; the paint doesn't).

Per section: link (a, b as "area:id"), s along it, centre x y z, unit normal pointing right of a->b, the road surface's
extent left and right of the centre on the link's deck (followed outwards at 5 cm while it stays road or gutter within 0.3 m
of height; "end" says what stopped it: kerb, walk, none, step = a height jump, far = still road at REACH), and the
paint lines crossed: offset right of centre, polyline id, colour, style, height."""
import json, math, os, sys, collections
import numpy as np
sys.path.insert(0, "/home/amy/git/openpilot-slowroads/openpilot/tools/sim/bridge/gta5/map/roadpaint")
import rp

TILES = "/mnt/e/gta5_scratch/gamefiles_v2/rp_all/tiles"
PAINT = "/mnt/e/gta5_scratch/gamefiles_v2/rp_all/polylines.jsonl"
PATHS = os.path.expanduser("~/gta5map/paths.jsonl")
PED_SPECIALS = {10, 14, 15, 16}
REACH, DS, ZSTEP = 30.0, 0.05, 0.3
END = {0: "none", 2: "kerb", 3: "walk", 4: "gutter"}
DRIVEABLE = (1, 4)  # road and gutter (flush beside a kerb): the road's edge is where a car would hit something

x0, x1, y0, y1 = map(float, sys.argv[1:5])
out = sys.argv[5]
step = next((float(a[7:]) for a in sys.argv if a.startswith("--step=")), 2.0)
inbox = lambda x, y, m=0: x0 - m <= x <= x1 + m and y0 - m <= y <= y1 + m

nodes, links = {}, {}
for line in open(PATHS):
  o = json.loads(line)
  if o["t"] == "n":
    nodes[(o["a"], o["i"])] = o
  elif o["t"] == "l":
    links[((o["a"], o["i"]), (o["ta"], o["ti"]))] = o
ok = lambda n: (n["f"][1] >> 3) not in PED_SPECIALS and not n["f"][2] & 32
L = []
for (ka, kb), l in links.items():
  if ka > kb or ka not in nodes or kb not in nodes or not ok(nodes[ka]) or not ok(nodes[kb]):
    continue
  a, b = nodes[ka], nodes[kb]
  if not (inbox(a["x"], a["y"]) or inbox(b["x"], b["y"])):
    continue
  if ((l["f"][2] >> 5) & 7) + ((l["f"][2] >> 2) & 7) == 0:
    continue
  L.append((ka, kb, np.array([a["x"], a["y"], a["z"]]), np.array([b["x"], b["y"], b["z"]])))

# paint segments in the area, gridded
segs = []
for line in open(PAINT):
  r = json.loads(line)
  P = np.array(r["pts"], float)
  if not ((P[:, 0] >= x0 - REACH) & (P[:, 0] <= x1 + REACH) & (P[:, 1] >= y0 - REACH) & (P[:, 1] <= y1 + REACH)).any():
    continue
  for k in range(len(P) - 1):
    segs.append((*P[k], *P[k + 1], r["id"], r["colour"] == "yellow", r["style"] == "dashed"))
S = np.array(segs, float)
G = 16.0
grid = collections.defaultdict(list)
for i, s in enumerate(S):
  for gx in range(int(math.floor(min(s[0], s[3]) / G)), int(math.floor(max(s[0], s[3]) / G)) + 1):
    for gy in range(int(math.floor(min(s[1], s[4]) / G)), int(math.floor(max(s[1], s[4]) / G)) + 1):
      grid[(gx, gy)].append(i)

def near(x, y, r):
  ids = set()
  for gx in range(int(math.floor((x - r) / G)), int(math.floor((x + r) / G)) + 1):
    for gy in range(int(math.floor((y - r) / G)), int(math.floor((y + r) / G)) + 1):
      ids.update(grid.get((gx, gy), ()))
  return np.fromiter(ids, int) if ids else np.zeros(0, int)

def paint_dir(c, d, z):
  """mean paint direction within 4 m of c on this deck, within 35 deg of d (doubled angles: lines have no sign)."""
  ix = near(c[0], c[1], 4.0)
  if not len(ix):
    return d
  s = S[ix]
  m = (s[:, :2] + s[:, 3:5]) / 2
  v = s[:, 3:5] - s[:, :2]
  ln = np.linalg.norm(v, axis=1)
  keep = (np.linalg.norm(m - c, axis=1) < 4.0) & (np.abs((s[:, 2] + s[:, 5]) / 2 - z) < 1.0) & (ln > 0.05)
  v, ln = v[keep], ln[keep]
  if not len(v):
    return d
  u = v / ln[:, None]
  cos = np.abs(u @ d)
  keep = cos > math.cos(math.radians(35))
  if keep.sum() == 0:
    return d
  u, w = u[keep], ln[keep]
  ang = np.arctan2(u[:, 1], u[:, 0]) * 2
  a = math.atan2((np.sin(ang) * w).sum(), (np.cos(ang) * w).sum()) / 2
  e = np.array([math.cos(a), math.sin(a)])
  return e if e @ d >= 0 else -e

T = rp.Tiles(TILES, cache=256)

def extent(c, n, z, sign):
  """distance from c along sign*n the surface stays road on the deck at z, and what ends it."""
  zc, last = z, 0.0
  for k in range(1, int(REACH / DS) + 1):
    p = c + sign * n * k * DS
    code, h = T.sample(np.array([p[0]]), np.array([p[1]]), zc, zwin=1.0)
    cl = int(code[0]) & 7
    if cl not in DRIVEABLE:
      return (k - 1) * DS, END.get(cl, str(cl))
    if abs(h[0] - zc) > ZSTEP:
      return (k - 1) * DS, "step"
    zc = float(h[0])
  return REACH, "far"

def crossings(c, n, z, lo, hi):
  """paint lines crossing the section from c + lo*n to c + hi*n on this deck, as (offset, id, yellow, dashed, z)."""
  a, b = c + lo * n, c + hi * n
  ix = near((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (hi - lo) / 2 + 1)
  out = []
  if not len(ix):
    return out
  s = S[ix]
  p, r = s[:, :2], s[:, 3:5] - s[:, :2]
  q, t = a, b - a
  den = r[:, 0] * t[1] - r[:, 1] * t[0]
  with np.errstate(divide="ignore", invalid="ignore"):
    u = ((q[0] - p[:, 0]) * t[1] - (q[1] - p[:, 1]) * t[0]) / den  # along the paint segment
    v = ((q[0] - p[:, 0]) * r[:, 1] - (q[1] - p[:, 1]) * r[:, 0]) / den  # along the section
  hit = (np.abs(den) > 1e-9) & (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
  for k in np.flatnonzero(hit):
    zz = s[k, 2] + u[k] * (s[k, 5] - s[k, 2])
    if abs(zz - z) > 1.2:
      continue
    ang = abs((r[k] / np.linalg.norm(r[k])) @ n)  # paint running across the section only
    if ang > math.sin(math.radians(50)):
      continue
    out.append((round(lo + v[k] * (hi - lo), 3), int(s[k, 6]), "yellow" if s[k, 7] else "white", "dashed" if s[k, 8] else "solid", round(float(zz), 2)))
  out.sort()
  return out

if __name__ == "__main__":
  nsec = 0
  with open(out, "w") as f:
    for ka, kb, A, B in L:
      d3 = B - A
      ln = float(np.linalg.norm(d3[:2]))
      if ln < 0.5:
        continue
      d = d3[:2] / ln
      for s in np.arange(step / 2, ln, step):
        c3 = A + d3 * (s / ln)
        c, z = c3[:2], c3[2]
        if not inbox(*c):
          continue
        # the deck's real height at the centre (node heights are approximate)
        code, h = T.sample(np.array([c[0]]), np.array([c[1]]), z, zwin=2.0)
        if int(code[0]) & 7 != 1:
          f.write(json.dumps({"a": f"{ka[0]}:{ka[1]}", "b": f"{kb[0]}:{kb[1]}", "s": round(float(s), 2), "c": [round(float(c[0]), 2), round(float(c[1]), 2), round(float(z), 2)], "off_road": True}) + "\n")
          continue
        z = float(h[0])
        e = paint_dir(c, d, z)
        n = np.array([e[1], -e[0]])  # right of travel
        left, lend = extent(c, n, z, -1)
        right, rend = extent(c, n, z, 1)
        xs = crossings(c, n, z, -left - 0.3, right + 0.3)
        f.write(json.dumps({"a": f"{ka[0]}:{ka[1]}", "b": f"{kb[0]}:{kb[1]}", "s": round(float(s), 2),
                            "c": [round(float(c[0]), 2), round(float(c[1]), 2), round(z, 2)], "n": [round(float(n[0]), 4), round(float(n[1]), 4)],
                            "left": round(left, 2), "lend": lend, "right": round(right, 2), "rend": rend, "lines": xs}) + "\n")
        nsec += 1
  print(len(L), "links", nsec, "sections ->", out)
