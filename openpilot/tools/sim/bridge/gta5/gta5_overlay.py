"""The map debug overlay and the GPS route on the game's own map, both drawn by the plugin from what the bridge sends.

Overlay: while the plugin's map debug is on (`gta5_cmd.py debug on`, or its key), every EVERY s the map around the car,
in world coordinates with the road's height, for the plugin to draw into the world from the player's camera: lane edges,
lane dividers, stop lines (lights, signs), junction areas, the route along the lanes it takes, nav's lane plan, and the
next turn with where its signal comes on. The road pieces follow gta5_train's maprender (its lane bands, dividers left
out at junctions, junction areas as hulls at the junction nodes where roads cross, stop lines across the lanes into
their junction, the route as its carriageway_line), drawn at their own heights rather than filtered to the car's level.
Lane edges and dividers are also cut out of junction areas, and edges left out from a stop line in to its junction.

GPS route: while the plugin's gpsroute is on, our route ahead, decimated to the points GTA's custom GPS route takes,
whenever the route changes or the car nears the end of what was sent.

Message formats (the plugin parses flat JSON only, so the geometry is one string):
- debugGeo: ox, oy, oz (m, the origin), rec (recording), g: polylines separated by ';', each a kind letter then
  comma-separated decimetres from the origin, the first point x,y,z and the rest the change from the point before.
  Kinds: e lane edge, d divider, l stop line (light), s stop line (sign), j junction outline (closed), r route ahead,
  b route behind, n nav's lane plan, m the next turn, g where its signal comes on.
- gpsPoints: p, "x,y,z;x,y,z;..." in metres (empty clears)."""
import os
import threading
import time
from collections import defaultdict

import numpy as np

EVERY = 0.5  # s between overlay updates
RADIUS = 150.0  # m around the car
LEVEL = 40.0  # m above or below the car: tunnels and bridges further off are left out
BEHIND = 30.0  # m of route behind the car
MAX_POINTS = 2500  # vertices per update, nearest first by layer priority
MAX_CHARS = 60000
SIMPLIFY = 0.15  # m a dropped vertex may be off the line
RIBBON_GAP = 1.0  # m between the route's points, for the plugin's ribbon
RIBBON_TURN = 60.0  # deg: sharper corners in the route's line are cut where a side is shorter than RIBBON_JOG
RIBBON_JOG = 5.0  # m
CELL = 50.0  # m: node index cells
LIGHT = 15  # node special: a traffic light's stop line
CAR_HEIGHT = 0.6  # paths.CAR_HEIGHT
# drawn first when over budget
PRIORITY = "mgnrbljsde"
LAYER_OF = {"e": "e", "d": "d", "l": "s", "s": "s", "j": "j", "r": "r", "b": "r", "n": "n", "m": "m", "g": "m"}
LAYERS = {"edges": "e", "dividers": "d", "stops": "s", "junctions": "j", "route": "r", "nav": "n", "points": "m", "fill": "f"}
DEFAULT_LAYERS = "edsjrnm"
GPS_MAX = 100  # points: GTA's custom GPS route limit isn't documented; the plugin clamps to its own max too
GPS_SIMPLIFY = 3.0  # m
GPS_RESEND_BEFORE = 300.0  # m before the end of a capped route sent, the next part goes
ENABLED = os.getenv("GTA5_OVERLAY", "1") != "0"


def smooth(x: np.ndarray, sigma: float) -> np.ndarray:
  """scipy's gaussian_filter1d(x, sigma, mode="nearest"), which openpilot's environment doesn't have."""
  if sigma <= 0 or len(x) < 2:
    return x
  r = int(4.0 * sigma + 0.5)
  k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
  return np.convolve(np.pad(x, r, mode="edge"), k / k.sum(), mode="valid")


def offset_line(pts: np.ndarray, offs: np.ndarray) -> np.ndarray:
  """A polyline moved right by offs per point, further at corners (maprender.offset_line)."""
  d = np.diff(pts, axis=0)
  normals = np.stack([d[:, 1], -d[:, 0]], axis=1) / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)[:, None]
  vn = np.concatenate((normals[:1], normals[:-1] + normals[1:], normals[-1:]))
  vn /= np.maximum(np.hypot(vn[:, 0], vn[:, 1]), 1e-6)[:, None]
  vn /= np.maximum(np.einsum("ij,ij->i", vn, np.concatenate((normals[:1], normals))), 0.5)[:, None]
  return pts + vn * offs[:, None]


def carriageway_line(route, step: float = 2.0, smooth_m: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
  """The route moved into the middle of the lanes it takes (maprender.carriageway_line): (points [M, 2], m along)."""
  pts = np.asarray(route.points, np.float64)
  along = np.asarray(route.along, np.float64)
  if len(pts) < 2:
    return pts, along
  offs = np.full(len(pts), np.nan)
  for k, link in enumerate(route.links):
    if link is None or not link.lanes:
      continue
    off = link.inner + link.lanes * link.width / 2
    for v in (k, k + 1):
      offs[v] = off if np.isnan(offs[v]) else (offs[v] + off) / 2
  known = ~np.isnan(offs)
  grid = np.arange(0.0, along[-1], step)
  grid = grid[np.abs(grid[:, None] - along[None]).min(axis=1) > step / 4] if len(grid) else grid
  v = np.sort(np.concatenate((grid, along)))
  base = np.stack([np.interp(v, along, pts[:, 0]), np.interp(v, along, pts[:, 1])], axis=1)
  if not known.any():
    return base, v
  return offset_line(base, smooth(np.interp(v, along[known], offs[known]), smooth_m / step)), v


def convex_hull(pts: np.ndarray) -> np.ndarray:
  """Andrew's monotone chain, counterclockwise, for the few corners at a node."""
  p = np.unique(np.round(pts, 3), axis=0)
  if len(p) < 3:
    return p
  p = p[np.lexsort((p[:, 1], p[:, 0]))]

  def half(points):
    out: list = []
    for q in points:
      while len(out) >= 2 and (out[-1][0] - out[-2][0]) * (q[1] - out[-2][1]) - (out[-1][1] - out[-2][1]) * (q[0] - out[-2][0]) <= 0:
        out.pop()
      out.append(q)
    return out
  lower, upper = half(p), half(p[::-1])
  return np.array(lower[:-1] + upper[:-1])


def simplify(pts: np.ndarray, tol: float) -> np.ndarray:
  return pts if len(pts) <= 2 else pts[simplify_mask(pts, tol)]


def simplify_mask(pts: np.ndarray, tol: float) -> np.ndarray:
  """Douglas-Peucker on [N, 2 or 3] points (distance in all their dimensions): which to keep."""
  keep = np.zeros(len(pts), bool)
  keep[[0, -1]] = True
  stack = [(0, len(pts) - 1)]
  while stack:
    i, j = stack.pop()
    if j <= i + 1:
      continue
    a, b = pts[i], pts[j]
    ab = b - a
    seg = pts[i + 1:j] - a
    n2 = float(ab @ ab)
    t = np.clip(seg @ ab / n2, 0.0, 1.0) if n2 > 1e-12 else np.zeros(len(seg))
    d = np.linalg.norm(seg - t[:, None] * ab, axis=1)
    k = int(np.argmax(d))
    if d[k] > tol:
      keep[i + 1 + k] = True
      stack += [(i, i + 1 + k), (i + 1 + k, j)]
  return keep


def chain(segs: np.ndarray) -> list[np.ndarray]:
  """Segments [M, 2, 3] joined into polylines through points that only two of them share (0.25 m)."""
  if not len(segs):
    return []
  # 0.25 m across, 2 m in height, so roads passing over each other don't join
  q = np.concatenate([np.round(segs[:, :, :2] * 4), np.round(segs[:, :, 2:] / 2)], axis=2).astype(np.int64)
  keys = [tuple(q[m, e]) for m in range(len(segs)) for e in (0, 1)]
  at = defaultdict(list)
  for n, k in enumerate(keys):
    at[k].append(n)
  used = np.zeros(len(segs), bool)
  out = []

  def extend(line: list, end: int):  # from segment end index `end` (2 * seg + side) onwards
    while True:
      ends = at[keys[end]]
      if len(ends) != 2:
        return
      nxt = ends[0] if ends[1] == end else ends[1]
      s = nxt // 2
      if used[s]:
        return
      used[s] = True
      other = nxt ^ 1
      line.append(segs[s, other & 1])
      end = other

  for s in range(len(segs)):
    if used[s]:
      continue
    used[s] = True
    fwd = [segs[s, 0], segs[s, 1]]
    extend(fwd, 2 * s + 1)
    back = [segs[s, 0]]
    extend(back, 2 * s)
    out.append(np.array(back[::-1][:-1] + fwd))
  return out


def disk(centre: np.ndarray, radius: float, n: int = 16) -> np.ndarray:
  """A circle as a counterclockwise polygon."""
  a = np.arange(n) * (2 * np.pi / n)
  return centre + radius * np.column_stack([np.cos(a), np.sin(a)])


def clip_outside(segs: np.ndarray, areas: list[np.ndarray], min_len: float = 0.3, only: list | None = None) -> np.ndarray:
  """Segments [M, 2, 3] with their parts inside any of the convex counterclockwise polygons [K, 2] cut out (in plan);
  with `only`, each polygon cuts just the segments its mask [M] picks."""
  if not len(segs) or not areas:
    return segs
  a, d = segs[:, 0], segs[:, 1] - segs[:, 0]
  lo_xy, hi_xy = np.minimum(segs[:, 0, :2], segs[:, 1, :2]), np.maximum(segs[:, 0, :2], segs[:, 1, :2])
  cuts: dict[int, list] = defaultdict(list)
  for n, poly in enumerate(areas):
    hit = (hi_xy >= poly.min(0)).all(1) & (lo_xy <= poly.max(0)).all(1)
    m = np.nonzero(hit & only[n] if only is not None else hit)[0]
    if not len(m):
      continue
    e = np.roll(poly, -1, axis=0) - poly
    rel = a[m, None, :2] - poly[None]
    num = e[None, :, 0] * rel[..., 1] - e[None, :, 1] * rel[..., 0]  # >= 0 inside each edge (left of it)
    den = e[None, :, 0] * d[m, None, 1] - e[None, :, 1] * d[m, None, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
      t = -num / den
    enter = np.maximum(np.where(den > 1e-12, t, -np.inf).max(1), 0.0)
    leave = np.minimum(np.where(den < -1e-12, t, np.inf).min(1), 1.0)
    never = ((np.abs(den) <= 1e-12) & (num < 0)).any(1)
    for k, t0, t1, no in zip(m, enter, leave, never, strict=True):
      if not no and t1 > t0:
        cuts[int(k)].append((t0, t1))
  if not cuts:
    return segs
  out = [segs[k] for k in range(len(segs)) if k not in cuts]
  for k, spans in cuts.items():
    length = float(np.linalg.norm(d[k, :2]))
    t = 0.0
    for t0, t1 in sorted(spans) + [(1.0, 1.0)]:
      if (t0 - t) * length >= min_len:
        out.append(np.array([a[k] + d[k] * t, a[k] + d[k] * t0]))
      t = max(t, t1)
  return np.array(out).reshape(-1, 2, 3)


def within(line: np.ndarray, pos: np.ndarray, radius: float) -> list[np.ndarray]:
  """The runs of a polyline [N, 3] with points within radius of pos (2D), each with one point beyond at its ends."""
  if len(line) < 2:
    return [line] if len(line) and np.hypot(*(line[0, :2] - pos)) <= radius else []
  near = np.hypot(*(line[:, :2] - pos).T) <= radius
  if near.all():
    return [line]
  keep = near.copy()
  keep[1:] |= near[:-1]
  keep[:-1] |= near[1:]
  out, start = [], None
  for n, k in enumerate(np.append(keep, False)):
    if k and start is None:
      start = n
    elif not k and start is not None:
      if n - start >= 2:
        out.append(line[start:n])
      start = None
  return out


class RoadGeometry:
  """GTA's roads as lane bands, built once from a paths.Paths; `near` gives the overlay's pieces around a point."""

  def __init__(self, paths):
    from openpilot.tools.sim.bridge.gta5.map.paths import heading, roads_cross
    self.paths = paths
    xy = paths.xy
    a, b, inner, width, lanes = [], [], [], [], []
    for (i, j), link in paths.links.items():
      if link.lanes and not link.shortcut:
        a.append(i)
        b.append(j)
        inner.append(link.inner)
        width.append(link.width)
        lanes.append(link.lanes)
    A, B = np.array(a, np.int64), np.array(b, np.int64)
    d = xy[B] - xy[A]
    length = np.hypot(d[:, 0], d[:, 1])
    ok = length >= 0.3
    self.A, self.B, d, length = A[ok], B[ok], d[ok], length[ok]
    fwd = d / length[:, None]
    self.right = np.stack([fwd[:, 1], -fwd[:, 0]], axis=1)
    self.lo = np.array(inner)[ok]
    self.width = np.array(width)[ok]
    self.lanes = np.array(lanes, np.int64)[ok]
    self.hi = self.lo + self.lanes * self.width
    self.band = {(int(i), int(j)): k for k, (i, j) in enumerate(zip(self.A, self.B, strict=True))}
    self.node_bands = defaultdict(list)
    for k, (i, j) in enumerate(zip(self.A.tolist(), self.B.tolist(), strict=True)):
      self.node_bands[i].append(k)
      self.node_bands[j].append(k)
    self.stop_node = np.array([paths.stop_line(i) for i in range(len(xy))], bool)
    self.junction = np.zeros(len(xy), bool)
    for i, ks in self.node_bands.items():
      if paths.junction(i):
        nb = {int(self.B[k]) if self.A[k] == i else int(self.A[k]) for k in ks}
        self.junction[i] = roads_cross([heading(*(xy[j] - xy[i])) for j in nb])
    cell = np.floor(xy / CELL).astype(np.int64)
    order = np.lexsort((cell[:, 1], cell[:, 0]))
    keys, starts = np.unique(cell[order], axis=0, return_index=True)
    self.cells = {(int(cx), int(cy)): order[s:e] for (cx, cy), s, e in zip(keys, starts, list(starts[1:]) + [len(order)], strict=True)}
    self.stop_cache: dict[int, list] = {}
    self.hull_cache: dict[int, np.ndarray | None] = {}

  def _nodes_near(self, pos: np.ndarray, z: float, radius: float) -> np.ndarray:
    (x0, y0), (x1, y1) = np.floor((pos - radius) / CELL).astype(int), np.floor((pos + radius) / CELL).astype(int)
    found = [self.cells[(cx, cy)] for cx in range(x0, x1 + 1) for cy in range(y0, y1 + 1) if (cx, cy) in self.cells]
    if not found:
      return np.zeros(0, np.int64)
    n = np.concatenate(found)
    p = self.paths
    return n[(np.hypot(*(p.xy[n] - pos).T) <= radius) & (np.abs(p.z[n] - z) <= LEVEL)]

  def _stops(self, i: int) -> list:
    if i not in self.stop_cache:
      p, out = self.paths, []
      if p.stop_line(i):
        light = (p.flags[i][1] >> 3) == LIGHT
        for q in p.into.get(i, ()):
          k = self.band.get((q, i))
          if k is None or not any(j != q and p.stop_for(i, j) for j in p.out.get(i, ())):
            continue
          r = self.right[k]
          seg = np.array([[*(p.xy[i] + r * self.lo[k]), p.z[i]], [*(p.xy[i] + r * self.hi[k]), p.z[i]]])
          out.append(("l" if light else "s", seg))
      self.stop_cache[i] = out
    return self.stop_cache[i]

  def _hull(self, i: int) -> np.ndarray | None:
    if i not in self.hull_cache:
      p, corners = self.paths, []
      for k in self.node_bands.get(i, ()):
        base = p.xy[self.A[k]] if self.A[k] == i else p.xy[self.B[k]]
        corners += [base + self.right[k] * self.lo[k], base + self.right[k] * self.hi[k]]
      hull = convex_hull(np.array(corners)) if len(corners) >= 3 else np.zeros((0, 2))
      self.hull_cache[i] = np.column_stack([np.vstack([hull, hull[:1]]), np.full(len(hull) + 1, p.z[i])]) if len(hull) >= 3 else None
    return self.hull_cache[i]

  def near(self, pos, z: float, radius: float, layers: str) -> list[tuple[str, np.ndarray]]:
    """[(kind, polyline [N, 3])] of the road pieces within radius of pos (game x, y) and LEVEL of the road height z."""
    pos = np.asarray(pos, np.float64)[:2]
    p = self.paths
    nodes = self._nodes_near(pos, z, radius + 30.0)
    out: list[tuple[str, np.ndarray]] = []
    if not len(nodes):
      return out
    mask = np.zeros(len(p.xy), bool)
    mask[nodes] = True
    ks = np.nonzero(mask[self.A] | mask[self.B])[0]
    A, B, R = self.A[ks], self.B[ks], self.right[ks]
    P = np.column_stack([p.xy[A], p.z[A]])
    Q = np.column_stack([p.xy[B], p.z[B]])
    jA, jB = self.junction[A], self.junction[B]

    def side(off, sel):  # segments [M, 2, 3] off m right of each selected band's line
      r3 = np.column_stack([R[sel], np.zeros(sel.sum())]) * off[:, None]
      return np.stack([P[sel] + r3, Q[sel] + r3], axis=1)

    hulls = [(i, h) for i in nodes[self.junction[nodes]].tolist() if (h := self._hull(i)) is not None]
    # edges and dividers are cut out of junction areas and the circles round them: the links out of a junction node, as
    # GTA's X-shaped junctions' diagonals, cross the junction beyond its node's own hull
    masks = [h[:-1, :2] for _, h in hulls]
    masks += [disk(p.xy[i], float(np.hypot(*(h[:-1, :2] - p.xy[i]).T).max())) for i, h in hulls]
    if "e" in layers:
      # not from a stop line in to its junction, nor across one: they'd criss-cross its area
      inA, inB = jA | self.stop_node[A], jB | self.stop_node[B]
      sel = ~(inA & inB & (jA | jB))
      segs = np.concatenate([side(self.lo[ks][sel], sel), side(self.hi[ks][sel], sel)])
      segs = clip_outside(segs, masks)
      ends = np.round(segs * 4).astype(np.int64).reshape(-1, 6)
      diff = ends[:, :3] - ends[:, 3:]
      flip = diff[np.arange(len(diff)), np.argmax(diff != 0, axis=1)] > 0  # one key for both directions of a shared edge
      ends[flip] = np.concatenate([ends[flip, 3:], ends[flip, :3]], axis=1)
      _, first = np.unique(ends, axis=0, return_index=True)
      out += [("e", line) for line in chain(segs[np.sort(first)])]
    if "d" in layers:
      sel = ~(jA | jB)  # maprender: no dividers inside junctions
      n = self.lanes[ks]
      rows, offs = [], []
      for k in range(1, int(n.max()) if len(n) else 1):
        s = sel & (n > k)
        rows.append(s)
        offs.append(self.lo[ks][s] + k * self.width[ks][s])
      segs = [side(o, s) for s, o in zip(rows, offs, strict=True) if s.any()]
      if segs:
        out += [("d", line) for line in chain(clip_outside(np.concatenate(segs), masks))]
    if "s" in layers:
      for i in nodes.tolist():
        out += self._stops(i)
    if "j" in layers or "f" in layers:
      out += [("j", h) for _, h in hulls]
    return out


def route_z(route, along: np.ndarray, fallback: float) -> np.ndarray:
  z = getattr(route, "z", None)
  if z is None or not len(z) or np.isnan(z).any():
    return np.full(len(along), fallback)
  return np.interp(along, route.along, z)


def ribbon_line(pts: np.ndarray, gap: float = RIBBON_GAP, max_turn: float = RIBBON_TURN) -> np.ndarray:
  """The route's line [N, 3] for the plugin's ribbon: points at least gap m apart, and no corner sharper than max_turn
  deg (the corner's point is dropped while it is), so GTA's sideways jogs between lanes don't make it jagged."""
  if len(pts) <= 2:
    return pts
  keep = [pts[0]]
  for q in pts[1:-1]:
    if np.hypot(*(q[:2] - keep[-1][:2])) >= gap:
      keep.append(q)
  if len(keep) > 1 and np.hypot(*(pts[-1, :2] - keep[-1][:2])) < gap:
    keep.pop()
  keep.append(pts[-1])
  out = np.array(keep)
  while len(out) > 2:
    d = np.diff(out[:, :2], axis=0)
    h = np.arctan2(d[:, 1], d[:, 0])
    turn = np.degrees(np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi))
    seg = np.hypot(d[:, 0], d[:, 1])
    turn[np.minimum(seg[:-1], seg[1:]) >= RIBBON_JOG] = 0.0  # a real turn's corner, between longer stretches, stays
    k = int(np.argmax(turn))
    if turn[k] <= max_turn:
      break
    out = np.delete(out, k + 1, axis=0)
  return out


def encode(origin: np.ndarray, items: list[tuple[str, np.ndarray]]) -> str:
  parts = []
  for kind, line in items:
    q = np.round((line - origin) * 10).astype(np.int64)
    q[1:] = np.diff(q, axis=0)
    parts.append(kind + ",".join(map(str, q.ravel().tolist())))
  return ";".join(parts)


def build(snap: dict, geometry: RoadGeometry | None) -> dict:
  """The debugGeo message for a snapshot of the car, route and nav (Overlay._snapshot)."""
  pos3 = np.asarray(snap["pos"], np.float64)
  pos, road_z = pos3[:2], float(pos3[2]) - CAR_HEIGHT
  layers = snap["layers"]
  items: list[tuple[str, np.ndarray]] = []
  if geometry is not None:
    items += geometry.near(pos, road_z, RADIUS, layers)
  route = snap.get("route")
  if route is not None and "r" in layers and len(route.points) >= 2:
    line, along = snap["carriageway"]
    for kind, lo, hi in (("b", route.at - BEHIND, route.at), ("r", route.at, route.at + 2 * RADIUS + 100.0)):
      # from exactly the car's place on the route, so the two parts meet there
      lo, hi = max(lo, float(along[0])), min(hi, float(along[-1]))
      v = np.concatenate(([lo], along[(along > lo + RIBBON_GAP) & (along < hi - RIBBON_GAP)], [hi]))
      if hi - lo > RIBBON_GAP:
        xy = np.column_stack([np.interp(v, along, line[:, 0]), np.interp(v, along, line[:, 1])])
        pts = ribbon_line(np.column_stack([xy, route_z(route, v, road_z)]))
        items += [(kind, run) for run in within(pts, pos, RADIUS)]
  lane = snap.get("lane_line")
  if lane is not None and len(lane) >= 2 and "n" in layers:
    lane = np.asarray(lane, np.float64)
    s = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(lane, axis=0).T))))
    z = route_z(route, (route.at if route is not None else 0.0) + s, road_z) if route is not None else np.full(len(s), road_z)
    items += [("n", run) for run in within(np.column_stack([lane, z]), pos, RADIUS)]
  if "m" in layers:
    for kind, xy in (("m", snap.get("turn")), ("g", snap.get("signal"))):
      if xy is not None:
        z = road_z
        if route is not None:
          k = int(np.argmin(np.hypot(*(np.asarray(route.points) - xy).T)))
          z = float(route_z(route, np.array([route.along[k]]), road_z)[0])
        items.append((kind, np.array([[xy[0], xy[1], z]])))
  items = [(k, simplify(line, SIMPLIFY) if len(line) > 2 else line) for k, line in items]
  # nearest first within each layer, layers by priority, to the budget
  items.sort(key=lambda it: (PRIORITY.index(it[0]), float(np.hypot(*(it[1][:, :2] - pos).T).min())))
  kept, points = [], 0
  for kind, line in items:
    if points + len(line) > MAX_POINTS:
      continue
    kept.append((kind, line))
    points += len(line)
  origin = np.round(pos3)
  g = encode(origin, kept)
  while len(g) > MAX_CHARS and kept:
    kept = kept[:int(len(kept) * 0.8)]
    g = encode(origin, kept)
  return {"type": "debugGeo", "ox": float(origin[0]), "oy": float(origin[1]), "oz": float(origin[2]),
          "rec": int(bool(snap.get("recording"))), "n": points, "g": g}


class Overlay:
  """Runs the overlay off the bridge's loop: update() hands a snapshot to a worker thread every EVERY s while the
  plugin's map debug is on, and returns the message it built last for the caller to send."""

  def __init__(self):
    self.next = 0.0
    self.lock = threading.Lock()
    self.wake = threading.Event()
    self.snap: dict | None = None
    self.outbox: dict | None = None
    self.thread: threading.Thread | None = None
    self.geometry: RoadGeometry | None = None
    self.geometry_for = None
    self.carriageway: tuple | None = None  # (route, (points, along))
    self.stats: dict = {}

  def update(self, state: dict, route, paths, lane_line, nav, recording: bool) -> list[dict]:
    out = []
    with self.lock:
      if self.outbox is not None:
        out.append(self.outbox)
        self.outbox = None
    debug = state.get("debug") or {}
    now = time.monotonic()
    if not ENABLED or not debug.get("on") or now < self.next:
      return out
    self.next = now + EVERY
    layers = str(debug.get("layers") or DEFAULT_LAYERS)
    snap = {"pos": state["pos"], "layers": layers, "route": route, "paths": paths, "recording": recording}
    if "n" in layers:
      snap["lane_line"] = lane_line()
    if "m" in layers and state.get("route") and nav is not None:
      points = nav.turn_points(np.array(state["route"], dtype=float), state)
      if points is not None:
        snap["turn"], snap["signal"] = points
    with self.lock:
      self.snap = snap
    if self.thread is None:
      self.thread = threading.Thread(target=self._run, name="gta5 overlay", daemon=True)
      self.thread.start()
    self.wake.set()
    return out

  def _run(self):
    while True:
      self.wake.wait()
      self.wake.clear()
      with self.lock:
        snap, self.snap = self.snap, None
      if snap is None:
        continue
      try:
        msg = self.make(snap)
      except Exception as e:  # a debug aid mustn't take the bridge down
        print(f"gta5 overlay: {type(e).__name__}: {e}", flush=True)
        continue
      with self.lock:
        self.outbox = msg

  def make(self, snap: dict) -> dict:
    t0 = time.monotonic()
    paths = snap.get("paths")
    if paths is not None and self.geometry_for is not paths:
      self.geometry, self.geometry_for = RoadGeometry(paths), paths
      self.stats["geometry_s"] = round(time.monotonic() - t0, 2)
      t0 = time.monotonic()
    route = snap.get("route")
    if route is not None:
      if self.carriageway is None or self.carriageway[0] is not route:
        self.carriageway = (route, carriageway_line(route))
      snap["carriageway"] = self.carriageway[1]
    msg = build(snap, self.geometry)
    self.stats.update(ms=round((time.monotonic() - t0) * 1000, 1), chars=len(msg["g"]), points=msg["n"])
    return msg


def gps_points(route, max_points: int = GPS_MAX) -> tuple[str, float]:
  """Our route ahead for GTA's custom GPS route, and the m along the route it reaches: rest() decimated, at most
  max_points (a capped route goes as far as they reach, and is sent again as the car nears its end)."""
  pts = route.rest()
  along = route.at + np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))))
  z = route_z(route, along, 0.0)
  line = np.column_stack([pts, z, along])
  if len(line) > 2:
    line = line[simplify_mask(line[:, :2], GPS_SIMPLIFY)]
  line = line[:max_points]
  reach = float(line[-1, 3]) if len(line) else route.at
  return ";".join(f"{x:.1f},{y:.1f},{zz:.1f}" for x, y, zz, _ in line), reach


class GpsRoute:
  """Keeps the plugin's custom GPS route on our route while its gpsroute is on (state "gpsRoute": on, points)."""

  RESEND_EVERY = 1.0  # s: the plugin's state shows new points a frame or two after they're sent

  def __init__(self):
    self.sent_for = None
    self.sent_t = -1e9
    self.reach = 0.0
    self.capped = False

  def update(self, state: dict, route) -> list[dict]:
    gps = state.get("gpsRoute") or {}
    if not gps.get("on"):
      self.sent_for = None
      return []
    if route is None:
      if self.sent_for is not None or gps.get("points"):
        self.sent_for = None
        return [{"type": "gpsPoints", "p": ""}]
      return []
    near_end = self.capped and route.at > self.reach - GPS_RESEND_BEFORE
    now = time.monotonic()
    if route is self.sent_for and (gps.get("points") and not near_end or now - self.sent_t < self.RESEND_EVERY):
      return []
    self.sent_t = now
    max_points = int(gps.get("max") or GPS_MAX)
    p, self.reach = gps_points(route, max_points)
    self.capped = self.reach < route.length - 1.0
    self.sent_for = route
    return [{"type": "gpsPoints", "p": p}]
