"""trace.py x0 x1 y0 y1 out.json: carriageways traced along the road surface itself, seeded from scan.py's sections.

From a seed the tracer steps STEP m along the paint at the middle of the kerb-to-kerb span, re-measures the span square
to the paint there, and recentres. It stops where the span jumps (a junction or a gore nose: width changes by more than
JUMP m in one step, or an end reaches REACH), leaves the road, or reaches a stretch already traced on this deck. Each
carriageway: points [x, y, z, left_w, right_w] (the span either side of the point) and per step the paint crossed.
Spans count as one where their endpoints match, so the several GTA links of one carriageway trace it once."""
import json, math, sys
import numpy as np
import scan  # reuses its tiles, paint index and section helpers (scan.py's own run is skipped on import)

STEP, JUMP, MAXW, MINW = 1.0, 1.5, 40.0, 2.0
x0, x1, y0, y1 = scan.x0, scan.x1, scan.y0, scan.y1
OCC_CM = 0.5  # occupancy grid, m per cell
occ = {}  # (gx, gy, round(z / 3)) -> carriageway id


def measure(c, e, z):
  e = scan.paint_dir(c, e, z)
  n = np.array([e[1], -e[0]])
  l, le = scan.extent(c, n, z, -1)
  r, re_ = scan.extent(c, n, z, 1)
  return e, n, l, le, r, re_


def occ_keys(c, n, l, r, z):
  ks = set()
  for t in np.arange(-l, r + 1e-6, OCC_CM):
    p = c + n * t
    ks.add((int(math.floor(p[0] / OCC_CM)), int(math.floor(p[1] / OCC_CM)), int(round(z / 3))))
  return ks


def trace_from(c, e, z, cid):
  pts, lines = [], []
  for sign in (1, -1):  # forwards, then backwards from the seed
    p, d, zc = c.copy(), e * sign, z
    prev_w = None
    seq = []
    for k in range(5000):
      code, h = scan.T.sample(np.array([p[0]]), np.array([p[1]]), zc, zwin=1.0)
      if int(code[0]) & 7 != 1 or abs(h[0] - zc) > 0.5:
        break
      zc = float(h[0])
      d, n, l, le, r, re_ = measure(p, d, zc)
      w = l + r
      if le == "far" or re_ == "far" or w > MAXW or w < MINW:
        break
      if prev_w is not None and abs(w - prev_w) > JUMP:
        break
      m = p + n * (r - l) / 2  # recentre
      keys = occ_keys(m, n, w / 2, w / 2, zc)
      taken = [occ[k] for k in keys if k in occ and occ[k] != cid]
      if len(taken) > len(keys) * 0.5:
        break
      for kk in keys:
        occ.setdefault(kk, cid)
      xs = scan.crossings(m, n, zc, -w / 2 - 0.3, w / 2 + 0.3)
      seq.append(([round(float(m[0]), 2), round(float(m[1]), 2), round(zc, 2), round(w / 2, 2), round(w / 2, 2)], le, re_, xs))
      prev_w = w
      p = m + d * STEP
      if not (x0 - 20 <= p[0] <= x1 + 20 and y0 - 20 <= p[1] <= y1 + 20):
        break
    if sign == 1:
      fwd = seq
    else:
      back = seq[1:][::-1]
  seq = back + fwd
  return [s[0] for s in seq], [[s[1], s[2]] for s in seq], [s[3] for s in seq]


seeds = [json.loads(l) for l in open(sys.argv[6])] if len(sys.argv) > 6 and sys.argv[6].endswith(".jsonl") else None
if seeds is None:
  raise SystemExit("usage: trace.py x0 x1 y0 y1 out.json sections.jsonl")
out = []
seeds = [d for d in seeds if not d.get("off_road") and d["lend"] != "far" and d["rend"] != "far"]
seeds.sort(key=lambda d: -(d["left"] + d["right"]))  # widest first: whole carriageways before their slivers
for d in seeds:
  c, n, z = np.array(d["c"][:2]), np.array(d["n"]), d["c"][2]
  m = c + n * (d["right"] - d["left"]) / 2
  w = d["left"] + d["right"]
  keys = occ_keys(m, n, w / 2, w / 2, z)
  if sum(k in occ for k in keys) > len(keys) * 0.3:
    continue
  e = np.array([-n[1], n[0]])
  cid = len(out)
  pts, ends, xs = trace_from(m, e, z, cid)
  if len(pts) < 3:
    continue
  out.append({"id": cid, "seed": [d["a"], d["b"]], "pts": pts, "ends": ends, "lines": xs})
json.dump(out, open(sys.argv[5], "w"))
print(len(out), "carriageways,", sum(len(c["pts"]) for c in out), "steps ->", sys.argv[5])
