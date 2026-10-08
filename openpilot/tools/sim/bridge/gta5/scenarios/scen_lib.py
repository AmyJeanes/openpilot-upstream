"""Shared helpers for the scenario search: the live map's OSM ways/nodes (data/map.pkl from extract.py), GTA's links
(e2e.Map), the live router (junction_plan.make_router), and the junction recording plan's training geometry to keep
clear of."""
import json
import math
import os
import pickle
import re
from collections import defaultdict
import numpy as np

HOME = os.path.expanduser("~")
LIVE = f"{HOME}/gta5map_lanes"
D = "/mnt/e/gta5_audit/scenarios_20261008"
PLAN = "/mnt/e/juncrec/plan_jr1.json"
CLEAR = 300.0  # m from any training approach


def wrap(d):
  return (d + 180.0) % 360.0 - 180.0


class Osm:
  CELL = 40.0

  def __init__(self):
    d = pickle.load(open(f"{D}/data/map.pkl", "rb"))
    self.ways, self.nodes = d["ways"], d["nodes"]
    self.grid = defaultdict(list)
    for wi, w in enumerate(self.ways):
      p = w["pts"]
      for i in range(len(p) - 1):
        mx, my = (p[i][0] + p[i + 1][0]) / 2, (p[i][1] + p[i + 1][1]) / 2
        self.grid[(int(mx // self.CELL), int(my // self.CELL))].append((wi, i))
    self.ncell = defaultdict(list)
    for nid, (x, y, _z, _t) in self.nodes.items():
      self.ncell[(int(x // self.CELL), int(y // self.CELL))].append(nid)

  def segs_near(self, x, y, r):
    n = int(r // self.CELL) + 1
    i, j = int(x // self.CELL), int(y // self.CELL)
    for di in range(-n, n + 1):
      for dj in range(-n, n + 1):
        yield from self.grid.get((i + di, j + dj), ())

  def nodes_near(self, x, y, r, pred=None):
    n = int(r // self.CELL) + 1
    i, j = int(x // self.CELL), int(y // self.CELL)
    out = []
    for di in range(-n, n + 1):
      for dj in range(-n, n + 1):
        for nid in self.ncell.get((i + di, j + dj), ()):
          nx, ny, nz, t = self.nodes[nid]
          if math.hypot(nx - x, ny - y) <= r and (pred is None or pred(t)):
            out.append((nid, math.hypot(nx - x, ny - y), t))
    return out

  def way_along(self, a, b, near=4.0):
    """The OSM way under the link a -> b ((x, y) pairs): (way, forward?) or None."""
    mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    bl = math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))
    best = None
    for wi, i in self.segs_near(mx, my, near + 20):
      p, q = self.ways[wi]["pts"][i], self.ways[wi]["pts"][i + 1]
      ab = np.array([q[0] - p[0], q[1] - p[1]])
      L = float(ab @ ab)
      if L < 1e-6:
        continue
      t = np.clip(((mx - p[0]) * ab[0] + (my - p[1]) * ab[1]) / L, 0, 1)
      d = math.hypot(p[0] + t * ab[0] - mx, p[1] + t * ab[1] - my)
      if d > near:
        continue
      bw = math.degrees(math.atan2(ab[0], ab[1]))
      dd = abs(wrap(bw - bl))
      fwd = dd < 30
      if not fwd and dd < 150:
        continue
      if best is None or d < best[0]:
        best = (d, self.ways[wi], fwd)
    return best and best[1:]


def dir_tag(w, fwd, key):
  t = w["t"]
  if t.get("oneway") == "yes":
    return t.get(key)
  return t.get(f"{key}:{'forward' if fwd else 'backward'}")


def training_points():
  """Every training approach's route (start, way in, ways out) from the plan, densified to ~5 m."""
  path = next(p for p in (PLAN, PLAN.replace(".json", "_v3.json"), PLAN.replace(".json", "_v2.json")) if os.path.exists(p))
  if path != PLAN:
    print(f"scen_lib: {PLAN} missing (being regenerated?), using {path}", flush=True)
  plan = json.load(open(path))
  pts = []
  for a in plan["approaches"]:
    if a.get("split") != "train":
      continue
    for e in a["exits"]:
      g = np.asarray(e.get("geom") or [], float)
      for i in range(len(g) - 1):
        n = max(1, int(math.hypot(*(g[i + 1] - g[i])) // 5))
        for k in range(n):
          pts.append(g[i] + (g[i + 1] - g[i]) * k / n)
      if len(g):
        pts.append(g[-1])
    pts.append([a["start"]["x"], a["start"]["y"]])
  return np.array(pts), plan


def ab_points():
  out = []
  for path in (f"{HOME}/gta5test/e2e/short-ab-v2.txt", f"{HOME}/gta5test/e2e/short-ab.txt"):
    for ln in open(path):
      body, _, c = ln.partition("#")
      at = re.search(r"at \((-?[\d.]+),\s*(-?[\d.]+)\)", c)
      if len(body.split()) == 2 and at:
        out.append((body.split()[0], float(at[1]), float(at[2])))
  return out


def min_dist(pts, x, y):
  return float(np.min(np.hypot(pts[:, 0] - x, pts[:, 1] - y))) if len(pts) else 1e9


def route_info(router, spec):
  """Route an e2e spec as the bridge would: points, along, maneuvers [{type, kind, s, xy, street, angle}]."""
  from openpilot.tools.sim.bridge.gta5 import e2e
  (sx, sy, sz, sh, dx, dy), lane = e2e.parse_spec(spec)
  r = router.route(np.array([sx, sy]), (-sh) % 360, np.array([dx, dy]), sz)
  pts, along = r.points, r.along
  mans = []
  for m in r.maneuvers:
    k = min(m["begin_shape_index"], len(pts) - 1)
    ang = wrap(m["bearing_after"] - m["bearing_before"]) if "bearing_before" in m and "bearing_after" in m else None
    mans.append({"type": m["type"], "kind": e2e.TYPES.get(m["type"], str(m["type"])), "s": round(float(along[k]), 1),
                 "xy": [round(float(v), 1) for v in pts[k]], "street": (m.get("street_names") or [None])[0],
                 "angle": None if ang is None else round(ang), "real": m["type"] in e2e.REAL,
                 "instr": m.get("instruction")})
  zs = None
  if getattr(r, "z", None) is not None and np.isfinite(r.z).any():
    ss = np.arange(0.0, float(along[-1]), 10.0)
    z = np.interp(ss, along, np.where(np.isfinite(r.z), r.z, np.nanmean(r.z)))
    zs = [[round(float(a), 1), round(float(b), 1)] for a, b in zip(ss, z, strict=True)]
  return {"z": zs, "points": np.round(pts, 1).tolist(), "length": round(float(along[-1])), "maneuvers": mans,
          "stops": [round(float(s), 1) for s in getattr(r, "stops", [])], "time": round(sum(m.get("time", 0) for m in r.maneuvers))}
