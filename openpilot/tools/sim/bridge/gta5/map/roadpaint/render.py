"""Rasterise the extracted road geometry (geoms.jsonl + tex/ from roadpaint's extract) into a world-aligned top-down
image: road surfaces, kerbs and pavements opaque (top surface within a height window of zref), decals blended on top.
Each texel's paint (none/white/yellow) is decided in texture space first: GTA bakes most lane lines into its
road-surface textures as bright bands along v, and draws the rest (centre lines, bays, arrows, stop lines) as decals
from marking atlases. Writes <prefix>_tex.png (the render) and <prefix>_paint.png (paint and surface classes, with our
map's lanes.json lines and GTA's links over it), and prints paint/kerb cross-sections along given links.

render.py <extract dir> <cx> <cy> <half size m> <m per px> <zref> <out prefix> [ax,ay,bx,by[@s1,s2..] ...]
"""
import json, os, re, sys, pathlib, math
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from tex2png import load

ROAD = re.compile(r'road|marking|blend|crossing|tarmac|asphalt|carpark|keep', re.I)
KERB = re.compile(r'kerb|curb', re.I)
GUTTER = re.compile(r'road_edge|gutter|edgedecal', re.I)
WALK = re.compile(r'sidewalk|pave|concrete|ground', re.I)
MAP = pathlib.Path(os.environ.get('GTA5MAP', '~/gta5map')).expanduser()  # ynddump's paths.jsonl, osm_to_roads' lanes.json
LANES = MAP / 'lanes.json'
PATHS = MAP / 'paths.jsonl'
NONE, ROADC, KERBC, WALKC, GUTC = 0, 1, 2, 3, 4


class Raster:
  def __init__(self, cx, cy, half, res, zref, zwin=4.0):
    self.x0, self.y1, self.res, self.zref, self.zwin = cx - half, cy + half, res, zref, zwin
    self.n = int(round(2 * half / res))
    self.rgb = np.zeros((self.n, self.n, 3), np.float32)
    self.z = np.full((self.n, self.n), -1e9, np.float32)
    self.cls = np.zeros((self.n, self.n), np.uint8)
    self.paint = np.zeros((self.n, self.n), np.float32)  # decal coverage
    self.pcode = np.zeros((self.n, self.n), np.uint8)  # paint of the visible texel: 0 none, 1 white, 2 yellow

  def px(self, x, y):  # world -> pixel (col, row), north up
    return (x - self.x0) / self.res, (self.y1 - y) / self.res

  def tri(self, P, UV, A, tex, decal, cls):
    if abs(P[:, 2].mean() - self.zref) > self.zwin:
      return
    c, r = self.px(P[:, 0], P[:, 1])
    c0, c1 = max(int(math.floor(c.min())), 0), min(int(math.ceil(c.max())), self.n - 1)
    r0, r1 = max(int(math.floor(r.min())), 0), min(int(math.ceil(r.max())), self.n - 1)
    if c0 > c1 or r0 > r1:
      return
    cc, rr = np.meshgrid(np.arange(c0, c1 + 1) + 0.5, np.arange(r0, r1 + 1) + 0.5)
    d = (r[1] - r[2]) * (c[0] - c[2]) + (c[2] - c[1]) * (r[0] - r[2])
    if abs(d) < 1e-9:
      return
    l0 = ((r[1] - r[2]) * (cc - c[2]) + (c[2] - c[1]) * (rr - r[2])) / d
    l1 = ((r[2] - r[0]) * (cc - c[2]) + (c[0] - c[2]) * (rr - r[2])) / d
    l2 = 1 - l0 - l1
    m = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
    if not m.any():
      return
    u = l0 * UV[0, 0] + l1 * UV[1, 0] + l2 * UV[2, 0]
    v = l0 * UV[0, 1] + l1 * UV[1, 1] + l2 * UV[2, 1]
    z = l0 * P[0, 2] + l1 * P[1, 2] + l2 * P[2, 2]
    tex, pmask = tex
    h, w = tex.shape[:2]
    ti, tj = np.floor(v * h).astype(int) % h, np.floor(u * w).astype(int) % w
    s = tex[ti, tj].astype(np.float32) / 255
    pc = pmask[ti, tj]
    sl = (slice(r0, r1 + 1), slice(c0, c1 + 1))
    if decal:  # lies on whatever surface is below, a few cm under it
      va = (l0 * A[0] + l1 * A[1] + l2 * A[2]) / 255
      al = s[..., 3] * va * m * (z > self.z[sl] - 0.35)
      self.rgb[sl] = self.rgb[sl] * (1 - al[..., None]) + s[..., :3] * al[..., None]
      self.paint[sl] = np.maximum(self.paint[sl], al)
      cov = ((al > 0.3) & (pc > 0)) | (al > 0.85)  # dirt and wear decals over a line leave it painted
      self.pcode[sl][cov] = pc[cov]
    else:
      top = m & (z > self.z[sl])
      self.rgb[sl][top] = s[..., :3][top]
      self.z[sl][top] = z[top]
      self.cls[sl][top] = cls
      self.pcode[sl][top] = pc[top]


def load_geoms(d):
  texs = {}
  for f in (d / 'tex').glob('*.rgba'):
    texs[re.sub(r'_\d+x\d+_.*', '', f.stem)] = f
  opaque, decals = [], []
  for line in open(d / 'geoms.jsonl'):
    g = json.loads(line)
    t = g['tex'].lower()
    if t not in texs:
      continue
    decal = 'decal' in g['sh'].lower()
    cls = GUTC if GUTTER.search(t) else KERBC if KERB.search(t) else ROADC if ROAD.search(t) else WALKC if WALK.search(t) else NONE
    if not decal and cls == NONE:
      continue
    (decals if decal else opaque).append((g, texs[t], decal, cls))
  return opaque + decals


def texture_paint(tex, decal):
  """Paint texels of one texture: 0 none, 1 white, 2 yellow. Road textures bake their lines in, so a texel is paint
  when it is much brighter than the texture's own asphalt (adaptive, so worn lines count); a decal's opaque bright
  texels are paint."""
  from PIL import ImageFilter
  rgb = tex[..., :3].astype(np.float32) / 255
  R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
  yellowish = (R - B > 0.2) & (B < 0.65 * R) & (G > 0.55 * R)
  if decal:
    cand = (tex[..., 3] > 128) & (rgb.max(-1) > 0.4)
    M = Image.fromarray((cand * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))
    cand = np.asarray(M) > 0
    out = cand.astype(np.uint8)
    out[cand & yellowish] = 2
    return out
  # road textures run their lines along v: find bright column bands (worn lines included), then the painted rows in
  # each band (dashes), and colour each band by its painted texels
  E = rgb.max(-1) - np.median(rgb.max(-1))
  h, w = E.shape
  top = np.sort(E, axis=0)[int(h * 0.7):].mean(0)  # mean of each column's brightest 30%: a 30%+ dash still shows
  base = np.median(top)
  band = top > base + 0.07
  out = np.zeros((h, w), np.uint8)
  j = 0
  while j < w:
    if not band[j]:
      j += 1
      continue
    k = j
    while k < w and band[k]:
      k += 1
    if k - j >= 2 and (k - j) < w * 0.25:
      rows = E[:, j:k].mean(1)
      kern = max(h // 24, 3)
      rows = np.convolve(rows, np.ones(kern) / kern, mode='same')
      painted = rows > 0.3 * (top[j:k].max() - base) + base
      if painted.mean() > 0.7:  # a solid line, worn in places
        painted[:] = True
      elif painted.mean() < 0.08:  # a speck, not a line
        painted[:] = False
      sel = np.zeros((h, w), bool)
      sel[painted, j:k] = True
      out[sel] = 2 if yellowish[sel].mean() > 0.5 else 1
    j = k
  return out


def rasterise(r, geoms, cache={}):
  for g, f, decal, cls in geoms:
    key = (f, decal)
    if key not in cache:
      t = load(f)
      cache[key] = (t, texture_paint(t, decal) if (decal or cls == ROADC) else np.zeros(t.shape[:2], np.uint8))
    V = np.array(g['v'], np.float64)
    I = np.array(g['i']).reshape(-1, 3)
    for t in I:
      r.tri(V[t, :3], V[t, 3:5], V[t, 5], cache[key], decal, cls)


def paint_mask(r):
  """0 none, 1 white, 2 yellow (per pixel), only on road surface."""
  out = r.pcode.copy()
  out[r.cls != ROADC] = 0
  return out


def links_near(cx, cy, half):
  nodes, links = {}, []
  for line in open(PATHS):
    o = json.loads(line)
    if o['t'] == 'n':
      nodes[(o['a'], o['i'])] = (o['x'], o['y'], o['z'])
    elif o['t'] == 'l':
      links.append(((o['a'], o['i']), (o['ta'], o['ti'])))
  out = []
  for a, b in links:
    if a in nodes and b in nodes:
      p, q = nodes[a], nodes[b]
      if min(p[0], q[0]) < cx + half and max(p[0], q[0]) > cx - half and min(p[1], q[1]) < cy + half and max(p[1], q[1]) > cy - half:
        out.append((a, b, p, q))
  return out


def lane_lines(cx, cy, half):
  L = json.load(open(LANES))
  out = []
  for ln in L['lines']:
    xy = np.array(ln[1:], float).reshape(-1, 2)
    if ((abs(xy[:, 0] - cx) < half) & (abs(xy[:, 1] - cy) < half)).any():
      out.append((L['kinds'][ln[0]], xy))
  return out


def sample(r, arr, a, d, right, s, o):
  pts = a + d * s + o[:, None] * right
  c, rr = r.px(pts[:, 0], pts[:, 1])
  ok = (c >= 0) & (c < r.n) & (rr >= 0) & (rr < r.n)
  ci, ri = np.clip(c.astype(int), 0, r.n - 1), np.clip(rr.astype(int), 0, r.n - 1)
  return np.where(ok, arr[ri, ci], 0)


def profile(r, pm, a, b, s, span=16.0, step=0.02, along=1.5):
  """Across the link a->b at s m along it (offsets right of travel), over a strip +-along m long: lines
  [(centre, width, colour, kind solid|dashed, coverage)] (two parallel lines < 0.4 m apart are one double line) and
  the kerb faces (first kerb/pavement pixel outward from the road on each side, at s)."""
  a, b = np.array(a, float), np.array(b, float)
  d = (b - a) / np.linalg.norm(b - a)
  right = np.array([d[1], -d[0]])
  o = np.arange(-span, span + step / 2, step)
  ss = np.arange(s - along, s + along + 1e-6, r.res)
  P = np.stack([sample(r, pm, a, d, right, t, o) for t in ss])  # rows along, cols across
  runs = []
  for colour, code in (('white', 1), ('yellow', 2)):
    cov = (P == code).mean(0)
    on = cov > 0.08
    k = 0
    while k < len(o):
      if on[k]:
        j = k
        while j < len(o) and on[j]:
          j += 1
        w = o[j - 1] - o[k] + step
        if 0.06 <= w <= 1.0:
          c = float((cov[k:j] * o[k:j]).sum() / cov[k:j].sum())
          runs.append([round(c, 2), round(w, 2), colour, float(cov[k:j].max())])
        k = j
      else:
        k += 1
  runs.sort()
  lines = []
  for c, w, colour, cv in runs:
    if lines and lines[-1][2] == colour and c - lines[-1][0] < 0.4 and lines[-1][3] != 'double':
      prev = lines.pop()
      lines.append([round((prev[0] + c) / 2, 2), round(c - prev[0] + w, 2), colour, 'double', round(max(cv, prev[4]), 2)])
    else:
      lines.append([c, w, colour, 'solid' if cv > 0.85 else 'dashed', round(cv, 2)])
  for ln in lines:
    if ln[3] == 'double':
      ln[3] = 'double_solid'  # TODO: per-line dash check
  cl = sample(r, r.cls, a, d, right, s, o)
  mid = len(o) // 2
  z = sample(r, r.z, a, d, right, s, o)
  kerbs = []  # per side: (asphalt ends / gutter starts, kerb face: first upward step of >= 8 cm within 30 cm)
  w = int(0.3 / step)
  for sgn, rng in ((-1, range(mid, w - 1, -1)), (1, range(mid, len(o) - w))):
    g = next((o[i] for i in rng if cl[i] in (GUTC, KERBC, WALKC)), None)
    k = next((o[i] for i in rng if z[i] > -1e8 and z[i + sgn * w] > -1e8 and z[i + sgn * w] - z[i] > 0.08), None)
    if k is not None:  # refine to the steepest point of the step
      i0 = int(round((k + span) / step))
      seg = [z[i0 + sgn * (j + 1)] - z[i0 + sgn * j] for j in range(w)]
      k = o[i0 + sgn * (int(np.argmax(seg)) + 1)] - sgn * step / 2
    kerbs.append(tuple(None if v is None else round(float(v), 2) for v in (g, k)))
  return lines, kerbs


def render(d, cx, cy, half, res, zref, prefix, specs):
  r = Raster(cx, cy, half, res, zref)
  rasterise(r, load_geoms(d))
  pm = paint_mask(r)
  Image.fromarray((r.rgb * 255).clip(0, 255).astype(np.uint8)).save(prefix + '_tex.png')
  img = np.full((r.n, r.n, 3), 30, np.uint8)
  img[r.cls == ROADC] = (60, 60, 60)
  img[r.cls == KERBC] = (150, 60, 60)
  img[r.cls == WALKC] = (90, 90, 120)
  img[r.cls == GUTC] = (110, 90, 60)
  img[pm == 1] = (255, 255, 255)
  img[pm == 2] = (255, 200, 0)
  im = Image.fromarray(img)
  dr = ImageDraw.Draw(im)
  col = {'edge': (0, 200, 255), 'dashed': (0, 255, 0), 'solid': (0, 160, 0), 'centre': (255, 0, 255), 'centre_dashed': (255, 0, 255)}
  for kind, xy in lane_lines(cx, cy, half):
    c, rr = r.px(xy[:, 0], xy[:, 1])
    dr.line(list(zip(c, rr)), fill=col.get(kind, (255, 128, 0)), width=1)
  for a, b, p, q in links_near(cx, cy, half):
    (c0, r0), (c1, r1) = r.px(p[0], p[1]), r.px(q[0], q[1])
    dr.line([(c0, r0), (c1, r1)], fill=(255, 0, 0), width=1)
  results = []
  for spec in specs:
    pts, _, ss = spec.partition('@')
    ax, ay, bx, by = map(float, pts.split(','))
    L = math.hypot(bx - ax, by - ay)
    for s in ([float(v) for v in ss.split(',')] if ss else np.linspace(0, L, 5)):
      runs, kerbs = profile(r, pm, (ax, ay), (bx, by), s)
      results.append(((ax, ay, bx, by), s, runs, kerbs))
      print(f'{ax},{ay}->{bx},{by} s={s:+.1f} kerbs {kerbs[0]} / {kerbs[1]}: ' + '  '.join(f'{o:+.2f}({w:.2f} {c[0]} {k} {cv})' for o, w, c, k, cv in runs))
      dx, dy = (bx - ax) / L, (by - ay) / L
      p0 = (ax + dx * s + dy * 16, ay + dy * s - dx * 16)
      p1 = (ax + dx * s - dy * 16, ay + dy * s + dx * 16)
      dr.line([r.px(*p0), r.px(*p1)], fill=(255, 255, 0), width=1)
  im.save(prefix + '_paint.png')
  return r, pm, results


if __name__ == '__main__':
  a = sys.argv
  render(pathlib.Path(a[1]), *map(float, a[2:7]), a[7], a[8:])
