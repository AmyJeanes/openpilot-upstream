#!/usr/bin/env python3
"""Scores a map's lane geometry against the paint seen from above (the student's confident paint lines,
mapfuse `paint_lanes_confident.json`, in lanes.json's format), and re-scores the map driver's drift recordings on it.

Two parts, each a report and a per-place list:

- **Map vs paint** (`static`): each confident paint line sampled every `STEP` m, and at each sample the map's nearest
  line of the same family (white lane lines, yellow centre lines, edges) running its way at its height. A sample with
  one within `MATCH` m is matched; the report gives, by road class (roads.json's nearest way), the share of matched
  length with the map's line more than 0.5 / 1.0 m away, and the places (15 m cells) with the most length over 1 m.
- **Drift re-score** (`drift`): the map term of the 16 Oct drift investigation (drift_20261016, `frames.npz`): at each
  frame the map's target lane centre against the centre of the paint lane it falls in (the frame's paint `lines`), for
  three targets: the route segment's own cross-section (`Section.offset`, what MapDrive._labels writes as target_right),
  the cross-section where the car is as the route's lanes move along it (`RouteLanes.section_at`: tapers, hops, lanes
  widening along a way), and the line the map driver plans through the recorded target lanes (`RouteLanes.lane_line`).
  The route each trip drove is rebuilt from its frames (the route's line at each frame, the car's place less its
  route_right, walked through the map's nodes), so the same frames score on any map whose ways are the recorded map's;
  on the map the trips drove, the rebuilt target centre is checked against the recorded one (`check`). Also the target
  line's steps (the target moving sideways in one frame, same target lane, no lane change), per km, away from junctions
  and the share the paint says are spurious (the paint lane did not move); the 24 events away from junctions, the
  recurring and named places of the brief, and 15 m cells by time over 1 m, before and after.

With `--base` (an earlier run's --out), report.md puts each number beside the baseline's, and the static samples are
compared one for one.

usage: paint_score.py MAP_DIR --paint paint_lanes_confident.json --drift DRIFT_DIR --out OUT_DIR [--base OUT_DIR_OF_BASELINE]
(MAP_DIR holds gta5.osm.pbf, lanes.json and roads.json; DRIFT_DIR frames.npz, ilat.npz, derived_j20.npz and
events_ctx_1.0_j20.csv.)
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np
from scipy.spatial import cKDTree

STEP = 1.0  # m between a paint line's samples
MATCH = 2.5  # m: a paint sample with a map line of its family this near is matched
ALIGN = 25.0  # deg between a paint line and the map's line matched to it
LEVEL = 3.0  # m of height between them at most
SEG_MAX = 2.0  # m: the map's lines are split into pieces at most this long for the search
PLACE = 15.0  # m cells for the per-place list
FAMILY = {'solid': 'white', 'dashed': 'white', 'centre': 'yellow', 'centre_dashed': 'yellow', 'edge': 'edge'}
THRESHOLDS = (0.5, 1.0)
# drift_20261016's selections (drift_events.py, attribute.py, jumps.py)
PHASE_OK = (2.0, 8.0, 9.0)  # drive, follow, yield
SEP = {0, 1, 2, 3}  # the paint kinds in a frame's lines that are lane lines (4 is an edge)
LANE_MIN, LANE_MAX = 2.4, 7.5  # m: a paint lane this wide
JUNCTION_M = 20.0  # m from junction marks: away from junctions
STEP_M = 0.7  # m: a target-line step
GLITCH_M = 0.5  # m: a rebuilt target off both frames beside it by this is a frame out of step with its segment
EVENT_S = 1.5  # s either side of an event's peak that it is scored over
PLACE_R = 15.0  # m round a recurring place
# the recurring places of the brief (CLOUD_TASK_LANEGEOM.md): (name, x, y)
RECURRING = [('strawberry_exit', 256.7, -1176.0), ('diverge_-378_-2062', -378.3, -2061.8), ('step_68_-1146', 68.7, -1145.9),
             ('plan_867_-1746', 867.3, -1745.8), ('plan_-1521_-243', -1521.4, -243.0)]
# other places the brief names
NAMED = [('chain_1079_-1546', 1079.5, -1546.2), ('drop_1191_442', 1190.9, 441.6), ('offset_901_-695', 901.4, -695.3),
         ('offset_531_-498', 530.7, -498.4), ('offset_126_-1367', 125.6, -1367.1), ('offset_-14_275', -13.9, 274.6),
         ('plan_1135_-3115', 1135.2, -3114.7), ('stacked_478_-482', 478.5, -482.4)]


# *** map vs paint ***

def _polylines(path: str, kinds_wanted=None):
  """lanes.json-format lines as [(kind name, points [N, 2], heights [N])]."""
  j = json.load(open(path))
  out = []
  for line, h in zip(j['lines'], j['heights'], strict=True):
    kind = j['kinds'][line[0]]
    if kinds_wanted is not None and kind not in kinds_wanted:
      continue
    p = np.asarray(line[1:], float).reshape(-1, 2)
    hz = np.asarray(h if isinstance(h, list) else [h], float)
    if len(hz) != len(p):
      hz = np.full(len(p), float(hz.mean()) if len(hz) else 0.0)
    if len(p) >= 2:
      out.append((kind, p, hz))
  return out


class Segments:
  """A family's map lines as short pieces, for the nearest one to a point at its height and heading."""
  def __init__(self, lines):
    a, b, za, zb = [], [], [], []
    for _, p, h in lines:
      for i in range(len(p) - 1):
        d = float(np.hypot(*(p[i + 1] - p[i])))
        if d < 1e-6:
          continue
        n = max(int(math.ceil(d / SEG_MAX)), 1)
        t = np.linspace(0.0, 1.0, n + 1)
        q = p[i] + (p[i + 1] - p[i]) * t[:, None]
        z = h[i] + (h[i + 1] - h[i]) * t
        a.append(q[:-1]); b.append(q[1:]); za.append(z[:-1]); zb.append(z[1:])
    self.a = np.concatenate(a) if a else np.zeros((0, 2))
    self.b = np.concatenate(b) if b else np.zeros((0, 2))
    self.z = (np.concatenate(za) + np.concatenate(zb)) / 2 if a else np.zeros(0)
    d = self.b - self.a
    self.u = d / np.maximum(np.hypot(*d.T), 1e-9)[:, None]
    self.tree = cKDTree((self.a + self.b) / 2) if len(self.a) else None

  def nearest(self, p: np.ndarray, z: np.ndarray, u: np.ndarray, reach: float) -> np.ndarray:
    """m from each point to the nearest piece within reach at its height, running its way (either sense); inf none."""
    out = np.full(len(p), np.inf)
    if self.tree is None:
      return out
    cos = math.cos(math.radians(ALIGN))
    K = 24
    n = len(self.a)
    for c0 in range(0, len(p), 200000):
      pp, zz, uu = p[c0:c0 + 200000], z[c0:c0 + 200000], u[c0:c0 + 200000]
      _, idx = self.tree.query(pp, k=K, distance_upper_bound=reach + SEG_MAX / 2)
      valid = idx < n
      idx = np.where(valid, idx, 0)
      a, ab = self.a[idx], self.b[idx] - self.a[idx]  # [M, K, 2]
      t = np.clip(np.einsum('mkj,mkj->mk', pp[:, None] - a, ab) / np.maximum(np.einsum('mkj,mkj->mk', ab, ab), 1e-12), 0.0, 1.0)
      d = np.hypot(*(a + ab * t[..., None] - pp[:, None]).transpose(2, 0, 1))
      keep = valid & (np.abs(self.z[idx] - zz[:, None]) < LEVEL) & (np.abs(np.einsum('mkj,mj->mk', self.u[idx], uu)) > cos)
      out[c0:c0 + 200000] = np.where(keep, d, np.inf).min(1)
    return out


def _samples(lines, step: float = STEP):
  """Points every `step` m along each line: (points, heights, unit directions, line index)."""
  P, Z, U, L = [], [], [], []
  for k, (_, p, h) in enumerate(lines):
    seg = np.hypot(*np.diff(p, axis=0).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    if s[-1] < step:
      continue
    at = np.arange(step / 2, s[-1], step)
    x, y, z = np.interp(at, s, p[:, 0]), np.interp(at, s, p[:, 1]), np.interp(at, s, h)
    i = np.clip(np.searchsorted(s, at, side='right') - 1, 0, len(seg) - 1)
    d = (p[i + 1] - p[i]) / np.maximum(seg[i], 1e-9)[:, None]
    P.append(np.stack([x, y], 1)); Z.append(z); U.append(d); L.append(np.full(len(at), k))
  return np.concatenate(P), np.concatenate(Z), np.concatenate(U), np.concatenate(L)


def road_classes(roads_path: str, p: np.ndarray, z: np.ndarray | None = None) -> list[str]:
  """Each point's road class: roads.json's nearest way (its middle) within its half width and 3 m."""
  r = json.load(open(roads_path))
  classes = r['classes']
  a, b, c, w = [], [], [], []
  for way, width in zip(r['ways'], r['widths'], strict=True):
    q = np.asarray(way[3:], float).reshape(-1, 2)
    for i in range(len(q) - 1):
      d = float(np.hypot(*(q[i + 1] - q[i])))
      n = max(int(math.ceil(d / 5.0)), 1)
      t = np.linspace(0, 1, n + 1)
      qq = q[i] + (q[i + 1] - q[i]) * t[:, None]
      a.append(qq[:-1]); b.append(qq[1:]); c += [way[0]] * n; w += [width] * n
  a, b, c, w = np.concatenate(a), np.concatenate(b), np.asarray(c), np.asarray(w)
  tree = cKDTree((a + b) / 2)
  names = np.array(classes + ['none'])
  out = np.full(len(p), len(classes))
  for c0 in range(0, len(p), 200000):
    pp = p[c0:c0 + 200000]
    _, idx = tree.query(pp, k=16, distance_upper_bound=25.0)
    valid = idx < len(a)
    idx = np.where(valid, idx, 0)
    aa, ab = a[idx], b[idx] - a[idx]
    t = np.clip(np.einsum('mkj,mkj->mk', pp[:, None] - aa, ab) / np.maximum(np.einsum('mkj,mkj->mk', ab, ab), 1e-12), 0, 1)
    d = np.where(valid, np.hypot(*(aa + ab * t[..., None] - pp[:, None]).transpose(2, 0, 1)) - w[idx] / 2, np.inf)
    j = d.argmin(1)
    best = d[np.arange(len(pp)), j]
    out[c0:c0 + 200000] = np.where(best < 3.0, c[idx[np.arange(len(pp)), j]], len(classes))
  return list(names[out])


def static(map_dir: str, paint_path: str, cache: dict | None = None) -> tuple[dict, list[dict], dict]:
  """The map's lines against the paint: the report, the per-place list, and per-sample arrays (for a before/after)."""
  cache = cache if cache is not None else {}
  if 'paint' not in cache:
    plines = _polylines(paint_path, set(FAMILY))
    P, Z, U, L = _samples(plines)
    fam = np.array([FAMILY[plines[k][0]] for k in L])
    cache['paint'] = (P, Z, U, L, fam)
  P, Z, U, L, fam = cache['paint']
  if 'classes' not in cache:
    cache['classes'] = np.array(road_classes(os.path.join(map_dir, 'roads.json'), P))
  cls = cache['classes']
  mlines = _polylines(os.path.join(map_dir, 'lanes.json'), set(FAMILY))
  d = np.full(len(P), np.inf)
  for f in set(FAMILY.values()):
    m = fam == f
    seg = Segments([ln for ln in mlines if FAMILY[ln[0]] == f])
    d[m] = seg.nearest(P[m], Z[m], U[m], MATCH)
  matched = d <= MATCH
  rep = {'samples': int(len(P)), 'step_m': STEP, 'match_m': MATCH, 'matched_share': round(float(matched.mean()), 4)}

  def shares(mask):
    n = int((mask & matched).sum())
    return {'matched_km': round(n * STEP / 1000, 2), **{f'over_{t}': round(float((d[mask & matched] > t).mean()), 4) if n else None
                                                        for t in THRESHOLDS},
            'median_m': round(float(np.median(d[mask & matched])), 3) if n else None}
  rep['all'] = shares(np.ones(len(P), bool))
  rep['by_class'] = {c: shares(cls == c) for c in sorted(set(cls)) if (cls == c).sum() * STEP > 500}
  rep['by_family'] = {f: shares(fam == f) for f in sorted(set(fam))}
  # places: 15 m cells by length over 1 m
  cells = defaultdict(list)
  cell_of = [(int(cx), int(cy)) for cx, cy in np.floor(P / PLACE)]
  n_matched = defaultdict(int)
  for i in np.flatnonzero(matched):
    n_matched[cell_of[i]] += 1
  for i in np.flatnonzero(matched & (d > 1.0)):
    cells[cell_of[i]].append(i)
  places = []
  for (cx, cy), idx in cells.items():
    idx = np.asarray(idx)
    places.append({'x': round((cx + 0.5) * PLACE, 1), 'y': round((cy + 0.5) * PLACE, 1), 'over_1m_m': round(len(idx) * STEP, 1),
                   'matched_m': round(n_matched[(cx, cy)] * STEP, 1), 'mean_off_m': round(float(d[idx].mean()), 2),
                   'class': max(set(cls[idx]), key=list(cls[idx]).count), 'families': sorted(set(fam[idx]))})
  places.sort(key=lambda r: -r['over_1m_m'])
  return rep, places, {'d': d, 'matched': matched}


# *** drift re-score ***

def _ele(osm, node: int) -> float:
  v = osm.data.node_tags.get(node, {}).get('ele')
  try:
    return float(v)
  except (TypeError, ValueError):
    return math.nan


def _right(h):
  hr = np.radians(h)
  return np.stack([np.cos(hr), np.sin(hr)], -1)


def rebuild_route(osm, x, y, z, h, route_at, route_right, tol: float = 1.2):
  """A recorded route's points from its frames: the route's line at each frame (the car less route_right across its
  heading), walked through the map's nodes it passes (joined by its ways, at the car's height), from the first frame's
  place to the last's. None where fewer than two nodes are found."""
  q = np.stack([x, y], 1) - route_right[:, None] * _right(h)
  fwd = np.stack([-np.sin(np.radians(h)), np.cos(np.radians(h))], 1)
  seen: dict[int, list[float]] = {}
  for i in range(0, len(q)):
    for n in osm.nodes_at(q[i], tol):
      e = _ele(osm, n)
      if math.isfinite(e) and math.isfinite(z[i]) and abs(e - z[i]) > 4.0:
        continue
      seen.setdefault(n, []).append(float(route_at[i] + (osm.node_xy(n) - q[i]) @ fwd[i]))
  cands = sorted((float(np.median(v)), n) for n, v in seen.items())
  if len(cands) < 2:
    return None
  # the longest chain of candidates in order along the route, each joined to the one before by a way (or two)
  m = len(cands)
  best = [1] * m
  prev = [-1] * m
  WINDOW = 8

  def joined(a, b):
    if (a, b) in osm.pairs:
      return True
    return any((a, c) in osm.pairs and (c, b) in osm.pairs for c in osm.links.get(a, ()))
  for j in range(m):
    for i in range(max(0, j - WINDOW), j):
      if cands[j][0] - cands[i][0] > 0.05 and joined(cands[i][1], cands[j][1]) and best[i] + 1 > best[j]:
        best[j], prev[j] = best[i] + 1, i
  j = int(np.argmax(best))
  chain = []
  while j >= 0:
    chain.append(cands[j][1])
    j = prev[j]
  chain = chain[::-1]
  nodes = [chain[0]]
  for b in chain[1:]:
    a = nodes[-1]
    if (a, b) not in osm.pairs:
      mid = next((c for c in osm.links.get(a, ()) if (c, b) in osm.pairs), None)
      if mid is not None:
        nodes.append(mid)
    nodes.append(b)
  pts = [osm.node_xy(n) for n in nodes]
  if np.hypot(*(q[0] - pts[0])) > 0.5:
    pts.insert(0, q[0])
  if np.hypot(*(q[-1] - pts[-1])) > 0.5:
    pts.append(q[-1])
  return np.asarray(pts, float)


def locate(points, along, p, at_guess, reach: float = 40.0):
  """The segment and m right of the route's line of each point, searching the segments within reach of its guess."""
  a, ab = points[:-1], np.diff(points, axis=0)
  L = np.maximum(np.hypot(*ab.T), 1e-9)
  seg = np.full(len(p), -1)
  right = np.full(len(p), np.nan)
  for i in range(len(p)):
    lo = max(int(np.searchsorted(along, at_guess[i] - reach)) - 1, 0)
    hi = min(int(np.searchsorted(along, at_guess[i] + reach)) + 1, len(a))
    if hi <= lo:
      continue
    aa, bb = a[lo:hi], ab[lo:hi]
    t = np.clip(np.einsum('ij,ij->i', p[i] - aa, bb) / (L[lo:hi] ** 2), 0, 1)
    d = np.hypot(*(aa + bb * t[:, None] - p[i]).T)
    k = int(np.argmin(d))
    if d[k] > 12.0:
      continue
    seg[i] = lo + k
    u = bb[k] / L[lo + k]
    right[i] = float((p[i] - aa[k]) @ np.array([u[1], -u[0]]))
  return seg, right


def parse_lines(s: str):
  if not s:
    return []
  return [(float(a), int(b)) for a, b in (q.split(':') for q in s.split(';'))]


def bracket(lines, lat):
  left = [ln for ln in lines if ln[0] < lat]
  right = [ln for ln in lines if ln[0] >= lat]
  return (left[-1] if left else None), (right[0] if right else None)


def lane_ok(left, right):
  return left is not None and right is not None and LANE_MIN <= right[0] - left[0] <= LANE_MAX


def _keys(s, lane):
  """lane_line's keys ([(m on from s[0], lane)]) for the recorded target lanes at route s: each change of lane ramped
  between the frames either side of it."""
  keys = [(0.0, float(lane[0]))]
  for i in range(1, len(s)):
    if lane[i] != lane[i - 1]:
      keys += [(float(s[i - 1] - s[0]), float(lane[i - 1])), (float(s[i] - s[0]), float(lane[i]))]
  keys.append((float(s[-1] - s[0]), float(lane[-1])))
  out = []
  for k in keys:
    if not out or k[0] > out[-1][0] + 1e-3:
      out.append(k)
    else:
      out[-1] = k
  return out


def target_centres(osm, z, legs=None, log=print):
  """The map's target lane centre (m right of the car) at every frame, on the rebuilt routes; NaN where unknown:
  `T` the segment's own cross-section's (Section.offset, as MapDrive._labels writes target_right), `T_at` the
  cross-section there as the route's lanes move along it (RouteLanes.section_at: tapers, lanes widening along a way,
  hops, blends), and `T_line` the line the map driver plans through the recorded target lanes (RouteLanes.lane_line:
  with its eased jogs and fillets). Also the legs (frame ranges with their route points), to reuse on another map with
  the same ways."""
  from openpilot.tools.sim.bridge.gta5.map.osm_lanes import RouteLanes, ways_from_nodes
  n = len(z['t'])
  trip, ra, rr = z['trip'], z['route_at'], z['route_right']
  T = np.full(n, np.nan)
  T_at = np.full(n, np.nan)
  T_line = np.full(n, np.nan)
  if legs is None:
    legs = []
    starts = np.flatnonzero(np.r_[True, trip[1:] != trip[:-1]])
    ends = np.r_[starts[1:], n]
    for a, b in zip(starts, ends, strict=True):
      cut = np.flatnonzero((np.diff(ra[a:b]) < -5.0) | (np.diff(ra[a:b]) > 30.0)) + a + 1
      for a0, b0 in zip(np.r_[a, cut], np.r_[cut, b], strict=True):
        if b0 - a0 < 20:
          continue
        pts = rebuild_route(osm, z['x'][a0:b0], z['y'][a0:b0], z['z'][a0:b0], z['h'][a0:b0], ra[a0:b0], rr[a0:b0])
        legs.append((int(a0), int(b0), pts))
  for a0, b0, pts in legs:
    if pts is None or len(pts) < 2:
      continue
    lanes = RouteLanes.from_osm(pts, ways_from_nodes(pts, osm), osm)
    along = lanes.along
    # the frames' along by the recorded one, less the rebuilt route's start's offset from the recorded start
    p = np.stack([z['x'][a0:b0], z['y'][a0:b0]], 1)
    seg0, _ = locate(pts, along, p[:1], np.array([0.0]), reach=1e6)
    shift = 0.0
    if seg0[0] >= 0:
      k = seg0[0]
      u = (pts[k + 1] - pts[k]) / max(float(np.hypot(*(pts[k + 1] - pts[k]))), 1e-9)
      shift = float(along[k] + (p[0] - pts[k]) @ u) - float(ra[a0])
    near, _ = locate(pts, along, p, ra[a0:b0] + shift)
    tl = z['target_lane'][a0:b0]
    L = np.maximum(np.hypot(*np.diff(pts, axis=0).T), 1e-9)
    # each frame's place along the route: its projection on its nearest segment, never back along the route (as the
    # plan's s, which the recorder took the segment by, only moves on), its segment by that, and the car's place
    # across that segment
    s_f = np.full(b0 - a0, np.nan)
    for i in np.flatnonzero(near >= 0):
      k = near[i]
      s_f[i] = float(along[k] + np.clip((p[i] - pts[k]) @ ((pts[k + 1] - pts[k]) / L[k]), 0.0, L[k]))
    s_f = np.fmax.accumulate(np.where(np.isfinite(s_f), s_f, -np.inf))
    seg = np.where((near >= 0) & np.isfinite(s_f), np.clip(np.searchsorted(along, s_f, side='right') - 1, 0, len(pts) - 2), -1)
    right = np.full(b0 - a0, np.nan)
    for i in np.flatnonzero(seg >= 0):
      k = seg[i]
      u = (pts[k + 1] - pts[k]) / L[k]
      right[i] = float((p[i] - pts[k]) @ np.array([u[1], -u[0]]))
    for i in range(b0 - a0):
      k = seg[i]
      if k < 0 or not np.isfinite(tl[i]):
        continue
      sec = lanes.sections[k]
      if sec is None or not sec.lanes:
        continue
      T[a0 + i] = sec.offset(tl[i]) - right[i]
      at = lanes.section_at(float(s_f[i]), k)
      if at is not None and at.lanes:
        T_at[a0 + i] = at.offset(tl[i]) - right[i]
    # the frame where the segment changes can take the target lane in the other segment's numbering (the recorder's
    # segment came from the plan's s, a frame apart from the car's): a value off both its neighbours by GLITCH_M is
    # the next frame's
    for arr in (T[a0:b0], T_at[a0:b0]):
      lo, hi = np.fmin(arr[:-2], arr[2:]), np.fmax(arr[:-2], arr[2:])
      bad = np.flatnonzero((arr[1:-1] < lo - GLITCH_M) | (arr[1:-1] > hi + GLITCH_M)) + 1
      arr[bad] = arr[bad + 1]
    # the planned line through the recorded lanes
    good = np.flatnonzero((seg >= 0) & np.isfinite(tl))
    if len(good) < 2:
      continue
    s_f = np.where(seg >= 0, s_f, np.nan)
    order = good[np.argsort(s_f[good], kind='stable')]
    s0 = max(float(s_f[order[0]]), lanes.start)
    try:
      line = lanes.lane_line(s0, _keys(s_f[order] - s_f[order[0]], tl[order]))
    except Exception as e:  # noqa: BLE001
      log(f'lane_line failed on leg {a0}-{b0}: {e!r}')
      line = None
    if line is None or len(line) < 2:
      continue
    lalong = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(line, axis=0).T))))
    lseg, lright = locate(line, lalong, p, np.where(np.isfinite(s_f), s_f - s0, 0.0), reach=60.0)
    ok = (lseg >= 0) & np.isfinite(s_f)
    T_line[a0:b0][ok] = -lright[ok]
  return {'T': T, 'T_at': T_at, 'T_line': T_line}, legs


VARIANTS = (('', 'T'), ('_at', 'T_at'), ('_line', 'T_line'))  # suffix, target array


def paint_errors(T, lines, mask):
  """Each frame's target centre (m right of the car) less the centre of the paint lane it falls in."""
  out = np.full(len(T), np.nan)
  for i in np.flatnonzero(mask & np.isfinite(T)):
    lt, rt = bracket(lines[i], T[i])
    if lane_ok(lt, rt):
      out[i] = T[i] - (lt[0] + rt[0]) / 2
  return out


def term_stats(err, m):
  e = np.abs(err[m])
  known = np.isfinite(e)
  return {'frames': int(m.sum()), 'paint_known': round(float(known.mean()), 4),
          **{f'over_{t}': round(float((e[known] > t).sum() / max(m.sum(), 1)), 4) for t in (0.5, 0.75, 1.0)},
          'p50': round(float(np.median(e[known])), 3) if known.any() else None,
          'p90': round(float(np.percentile(e[known], 90)), 3) if known.any() else None}


def target_steps(z, T, map_err, jd):
  """The target line's steps: |change| over STEP_M m in one frame, same target lane, no lane change; each with
  whether the paint lane moved with it (spurious where it didn't, drift_20261016's jumps.py)."""
  trip, t, x, y, v = z['trip'], z['t'], z['x'], z['y'], z['v']
  tl, lc = z['target_lane'], np.isfinite(z['lc_tau'])
  e_map = -T
  de = np.r_[0.0, np.diff(e_map)]
  same = np.r_[False, (trip[1:] == trip[:-1]) & (tl[1:] == tl[:-1])] & ~lc & np.isin(z['phase'], PHASE_OK) & (v > 3.0)
  idx = np.flatnonzero(same & np.isfinite(de) & (np.abs(de) > STEP_M))
  starts = {tr: i for i, tr in reversed(list(enumerate(trip)))}
  steps = []
  for i in idx:
    lo = starts[trip[i]]
    w = np.arange(max(lo, i - 60), min(i + 60, len(t)))
    w = w[trip[w] == trip[i]]
    w0 = w[(t[w] > t[i] - 1.5) & (t[w] < t[i] - 0.3)]
    w1 = w[(t[w] > t[i] + 0.3) & (t[w] < t[i] + 1.5)]
    me0 = np.nanmedian(map_err[w0]) if len(w0) and np.isfinite(map_err[w0]).any() else np.nan
    me1 = np.nanmedian(map_err[w1]) if len(w1) and np.isfinite(map_err[w1]).any() else np.nan
    spur = None if np.isnan(me0) or np.isnan(me1) else bool(abs((me1 - me0) - (-de[i])) < 0.5 * abs(de[i]) and abs(me1 - me0) > 0.5)
    steps.append({'trip': str(trip[i]), 'i': int(i), 'x': round(float(x[i]), 1), 'y': round(float(y[i]), 1), 'step': round(float(-de[i]), 2),
                  'junction_m': round(float(jd[i]), 1), 'spurious': spur})
  return steps


def step_stats(steps, km):
  far = [s for s in steps if s['junction_m'] > JUNCTION_M]
  known = [s for s in far if s['spurious'] is not None]
  return {'km': round(km, 1), 'all': len(steps), 'per_km': round(len(steps) / km, 2), 'away': len(far),
          'away_per_km': round(len(far) / km, 3), 'away_over_2m': sum(abs(s['step']) > 2 for s in far),
          'away_paint_checked': len(known), 'away_spurious': sum(s['spurious'] for s in known)}


def drift(osm, frames_dir: str, legs=None, log=print) -> tuple[dict, dict, list, dict]:
  """The drift re-score on a map (its OsmLanes): the report, per-frame arrays, the legs rebuilt and the steps."""
  z = np.load(os.path.join(frames_dir, 'frames.npz'))
  der = np.load(os.path.join(frames_dir, 'derived_j20.npz'))
  ilat = np.load(os.path.join(frames_dir, 'ilat.npz'))['ilat']
  n = len(z['t'])
  targets, legs = target_centres(osm, z, legs, log)
  T_rec = z['target_right'] - z['route_right']
  lines = [parse_lines(s) for s in z['lines']]
  base0 = np.isin(z['phase'], PHASE_OK) & (z['v'] > 4.0) & (z['kind'] == 'ours') & ~der['mask_lc']
  jd = der['jd']
  W_car = np.full(n, np.nan)
  for i in np.flatnonzero(base0):
    left, right = bracket(lines[i], 0.0) if lines[i] else (None, None)
    if lane_ok(left, right):
      W_car[i] = right[0] - left[0]
  dt = 0.05
  ok = base0 & np.isfinite(W_car) & (W_car >= 4.0) & np.isfinite(ilat)
  rep = {'frames': n, 'ok_frames': int(ok.sum()), 'ok_hours': round(ok.sum() * dt / 3600, 3),
         'legs': len(legs), 'legs_rebuilt': sum(p is not None for _, _, p in legs)}
  arrays = {'ok': ok, 'jd': jd, 'T_rec': T_rec, 'W_car': W_car}
  x, y, trip = z['x'], z['y'], z['trip']
  km = float(np.nansum(np.where(np.r_[False, trip[1:] == trip[:-1]], np.hypot(np.r_[0, np.diff(x)], np.r_[0, np.diff(y)]), 0)) / 1000)
  steps = {}
  for suf, key in VARIANTS:
    T = targets[key]
    err = paint_errors(T, lines, ok)
    arrays[key], arrays[f'map_err{suf}'] = T, err
    rep[f'target{suf}_known'] = round(float(np.isfinite(T[ok]).mean()), 4)
    rep[f'map_term{suf}_all'] = term_stats(err, ok)
    rep[f'map_term{suf}_j20'] = term_stats(err, ok & (jd > JUNCTION_M))
    if key != 'T_line':  # the planned line moves across smoothly: no steps to count
      steps[key] = target_steps(z, T, err, jd)
      rep[f'steps{suf}'] = step_stats(steps[key], km)
  # the rebuilt static target centre against the recorded one (meaningful only on the map the trips drove)
  T = targets['T']
  both = ok & np.isfinite(T) & np.isfinite(T_rec)
  diff = np.abs(T[both] - T_rec[both])
  rep['check'] = {'frames': int(both.sum()), 'within_0.1': round(float((diff < 0.1).mean()), 4),
                  'within_0.5': round(float((diff < 0.5).mean()), 4), 'p99': round(float(np.percentile(diff, 99)), 3)}
  rec_err = paint_errors(T_rec, lines, ok)
  rep['recorded_map_term_all'] = term_stats(rec_err, ok)
  rep['recorded_map_term_j20'] = term_stats(rec_err, ok & (jd > JUNCTION_M))
  arrays['map_err_rec'] = rec_err
  steps['T_rec'] = target_steps(z, T_rec, rec_err, jd)
  rep['steps_recorded'] = step_stats(steps['T_rec'], km)
  return rep, arrays, legs, steps


def event_status(frames_dir: str, arrays: dict, events_csv: str, base: dict | None = None) -> list[dict]:
  """Each event (drift_20261016's j20 list), recurring and named place: the largest |map term| there, its share over
  1 m and the largest target step, for each target (VARIANTS), and against a baseline's arrays whether it got better
  (by the planned line's term)."""
  z = np.load(os.path.join(frames_dir, 'frames.npz'), allow_pickle=False)
  trip, t, x, y = z['trip'], z['t'], z['x'], z['y']
  out = []

  def score(mask, arr, suf, key):
    e = np.abs(arr[f'map_err{suf}'][mask])
    tt = arr[key][mask]
    steps = np.abs(np.diff(tt))
    fin = np.isfinite(e)
    return {f'max{suf}': round(float(np.nanmax(e)), 2) if fin.any() else None,
            f'over1{suf}': round(float((e[fin] > 1.0).mean()), 3) if fin.any() else None,
            f'step{suf}': round(float(np.nanmax(steps)), 2) if np.isfinite(steps).any() else None}

  def row(r, m):
    for suf, key in VARIANTS:
      r.update(score(m, arrays, suf, key))
      if base is not None:
        r.update({f'base_{k}': v for k, v in score(m, base, suf, key).items()})
    return r
  for e in csv.DictReader(open(events_csv)):
    tr = trip == e['trip']
    if not tr.any():
      continue
    i = np.flatnonzero(tr)[int(np.argmin(np.hypot(x[tr] - float(e['x']), y[tr] - float(e['y']))))]
    m = tr & (np.abs(t - t[i]) <= EVENT_S) & arrays['ok']
    out.append(row({'kind': 'event', 'name': f"{e['trip']}@{e['x']},{e['y']}", 'cause': e['cause'], 'x': float(e['x']), 'y': float(e['y'])}, m))
  for kind, places in (('recurring', RECURRING), ('named', NAMED)):
    for name, px, py in places:
      m = (np.hypot(x - px, y - py) < PLACE_R) & arrays['ok']
      out.append(row({'kind': kind, 'name': name, 'cause': '', 'x': px, 'y': py, 'frames': int(m.sum()), 'trips': len(set(trip[m]))}, m))
  for r in out:
    a, b = r.get('max_line'), r.get('base_max_line')
    if a is not None and b is not None:
      r['status'] = 'fixed' if a < 1.0 <= b else 'better' if a - b < -0.25 else 'worse' if a - b > 0.25 else 'same'
  return out


def drift_places(frames_dir: str, arrays: dict, base: dict | None = None, key: str = 'map_err_at') -> list[dict]:
  """15 m cells by driven time with the target centre more than 1 m off the paint (and the baseline's there)."""
  z = np.load(os.path.join(frames_dir, 'frames.npz'))
  x, y, trip = z['x'], z['y'], z['trip']
  ok = arrays['ok']
  cells: dict = defaultdict(lambda: [0, 0, set(), 0])
  for i in np.flatnonzero(ok):
    c = cells[(int(x[i] // PLACE), int(y[i] // PLACE))]
    c[0] += 1
    e = arrays[key][i]
    if np.isfinite(e) and abs(e) > 1.0:
      c[1] += 1
      c[2].add(str(trip[i]))
    if base is not None and np.isfinite(base[key][i]) and abs(base[key][i]) > 1.0:
      c[3] += 1
  out = []
  for (cx, cy), (n, over, trips, b) in cells.items():
    if over or b:
      out.append({'x': round((cx + 0.5) * PLACE, 1), 'y': round((cy + 0.5) * PLACE, 1), 'over_1m_s': round(over * 0.05, 2),
                  'frames_s': round(n * 0.05, 2), 'trips': len(trips), **({'base_over_1m_s': round(b * 0.05, 2)} if base is not None else {})})
  out.sort(key=lambda r: -max(r['over_1m_s'], r.get('base_over_1m_s', 0)))
  return out


def static_compare(cur: dict, base: dict, classes: np.ndarray) -> dict:
  """Before/after on the same paint samples: by class, matched in both, the share over 1 m each, and the samples
  that moved more than 0.5 m nearer the paint or further from it."""
  out = {}
  both = cur['matched'] & base['matched']
  for c in ['all'] + sorted(set(classes)):
    m = both & ((classes == c) if c != 'all' else True)
    if m.sum() * STEP < 500:
      continue
    d0, d1 = base['d'][m], cur['d'][m]
    out[c] = {'km': round(m.sum() * STEP / 1000, 1), 'base_over_1.0': round(float((d0 > 1).mean()), 4), 'over_1.0': round(float((d1 > 1).mean()), 4),
              'base_over_0.5': round(float((d0 > 0.5).mean()), 4), 'over_0.5': round(float((d1 > 0.5).mean()), 4),
              'nearer_0.5_m': int(((d0 - d1) > 0.5).sum() * STEP), 'further_0.5_m': int(((d1 - d0) > 0.5).sum() * STEP)}
  return out


def markdown(report: dict) -> str:
  """report.json as a short page: the numbers each step reports before and after."""
  L = [f"# Paint score: {report['map_dir']}", '']
  st, bs = report['static'], report.get('base')
  L += ['## Map vs paint (matched paint length with the map\'s nearest line over 0.5 / 1.0 m)', '',
        '| class | km | > 0.5 m | > 1.0 m | base > 0.5 m | base > 1.0 m |', '|---|---|---|---|---|---|']
  for c, v in [('all', st['all'])] + list(st['by_class'].items()):
    b = (bs or {}).get('by_class', {}).get(c) if c != 'all' else (bs or {}).get('all')
    pct = lambda q: '' if q is None else f'{100 * q:.1f}%'  # noqa: E731
    L.append(f"| {c} | {v['matched_km']} | {pct(v['over_0.5'])} | {pct(v['over_1.0'])} | {pct((b or {}).get('over_0.5'))} | {pct((b or {}).get('over_1.0'))} |")
  if 'compare' in report:
    L += ['', 'Same samples before and after (matched in both):', '', '| class | km | base > 1 m | now > 1 m | m nearer by 0.5 | m further by 0.5 |', '|---|---|---|---|---|---|']
    for c, v in report['compare'].items():
      L.append(f"| {c} | {v['km']} | {100 * v['base_over_1.0']:.1f}% | {100 * v['over_1.0']:.1f}% | {v['nearer_0.5_m']} | {v['further_0.5_m']} |")
  d, bd = report.get('drift'), report.get('base_drift')
  if d:
    L += ['', '## Drift re-score: target centre vs paint lane centre, share of frames', '',
          '| target | > 0.5 m | > 1.0 m | > 1.0 m j20 | base > 0.5 m | base > 1.0 m | base > 1.0 m j20 |', '|---|---|---|---|---|---|---|']
    for suf, name in (('', 'static'), ('_at', 'section_at'), ('_line', 'planned line')):
      a, j = d[f'map_term{suf}_all'], d[f'map_term{suf}_j20']
      ba, bj = (bd or {}).get(f'map_term{suf}_all', {}), (bd or {}).get(f'map_term{suf}_j20', {})
      f = lambda q: '' if q is None else f'{100 * q:.2f}%'  # noqa: E731
      L.append(f"| {name} | {f(a['over_0.5'])} | {f(a['over_1.0'])} | {f(j['over_1.0'])} | {f(ba.get('over_0.5'))} | {f(ba.get('over_1.0'))} | {f(bj.get('over_1.0'))} |")
    L += ['', '| steps | per km | away | away per km | away > 2 m | paint-checked | spurious |', '|---|---|---|---|---|---|---|']
    for suf, name in (('', 'static'), ('_at', 'section_at')):
      for tag, src in (('', d), ('base ', bd or {})):
        s = src.get(f'steps{suf}')
        if s:
          L.append(f"| {tag}{name} | {s['per_km']} | {s['away']} | {s['away_per_km']} | {s['away_over_2m']} | {s['away_paint_checked']} | {s['away_spurious']} |")
    L += ['', f"Rebuilt vs recorded static target (meaningful on the recorded map only): {d['check']}"]
  ev = report.get('events')
  if ev:
    L += ['', '## Events and places: largest |target - paint centre| (m), static / section_at / planned line', '',
          '| kind | name | cause | static | section_at | line | base line | status |', '|---|---|---|---|---|---|---|---|']
    for r in ev:
      L.append(f"| {r['kind']} | {r['name']} | {r.get('cause', '')} | {r.get('max')} | {r.get('max_at')} | {r.get('max_line')} | "
               f"{r.get('base_max_line', '')} | {r.get('status', '')} |")
  return '\n'.join(L) + '\n'


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('map_dir')
  p.add_argument('--paint', required=True)
  p.add_argument('--drift', help='drift_20261016 folder (frames.npz, ilat.npz, derived_j20.npz, events_ctx_1.0_j20.csv)')
  p.add_argument('--out', required=True)
  p.add_argument('--base', help="a baseline's --out, for before/after")
  p.add_argument('--legs', help='rebuilt routes to reuse (legs.npz from an earlier run on a map with the same ways)')
  p.add_argument('--roads', help="roads.json for the road classes (default: the map's, else the base's map's)")
  args = p.parse_args()
  os.makedirs(args.out, exist_ok=True)
  report = {'map_dir': os.path.abspath(args.map_dir), 'paint': os.path.abspath(args.paint)}
  base_rep = json.load(open(os.path.join(args.base, 'report.json'))) if args.base and os.path.exists(os.path.join(args.base, 'report.json')) else None
  roads = args.roads or os.path.join(args.map_dir, 'roads.json')
  if not os.path.exists(roads) and base_rep:
    roads = os.path.join(base_rep['map_dir'], 'roads.json')
  cache: dict = {}
  plines = _polylines(args.paint, set(FAMILY))
  P, Z, U, Ls = _samples(plines)
  cache['paint'] = (P, Z, U, Ls, np.array([FAMILY[plines[k][0]] for k in Ls]))
  cache['classes'] = np.array(road_classes(roads, P))
  rep, places, arr = static(args.map_dir, args.paint, cache)
  report['static'] = rep
  np.savez_compressed(os.path.join(args.out, 'static.npz'), **arr, classes=cache['classes'])
  if args.base and os.path.exists(os.path.join(args.base, 'static.npz')):
    report['compare'] = static_compare(arr, dict(np.load(os.path.join(args.base, 'static.npz'))), cache['classes'])
  with open(os.path.join(args.out, 'places_static.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(places[0].keys()) if places else ['x'])
    w.writeheader()
    for r in places[:500]:
      w.writerow({**r, 'families': '+'.join(r['families'])})
  if args.drift:
    from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
    from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes
    osm = OsmLanes.load(os.path.join(args.map_dir, 'gta5.osm.pbf'), to_game)
    legs = None
    if args.legs:
      lz = np.load(args.legs, allow_pickle=True)
      legs = [tuple(v) for v in lz['legs']]
    drep, darr, legs, steps = drift(osm, args.drift, legs)
    report['drift'] = drep
    np.savez_compressed(os.path.join(args.out, 'drift.npz'), **darr)
    np.savez_compressed(os.path.join(args.out, 'legs.npz'), legs=np.array(legs, dtype=object))
    for key, rows in steps.items():
      with open(os.path.join(args.out, f'steps_{key}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['x'])
        w.writeheader()
        [w.writerow(s) for s in rows]
    base = None
    if args.base and os.path.exists(os.path.join(args.base, 'drift.npz')):
      base = dict(np.load(os.path.join(args.base, 'drift.npz')))
    ev = event_status(args.drift, darr, os.path.join(args.drift, 'events_ctx_1.0_j20.csv'), base)
    report['events'] = ev
    with open(os.path.join(args.out, 'events.csv'), 'w', newline='') as f:
      keys = list(dict.fromkeys(k for r in ev for k in r))
      w = csv.DictWriter(f, fieldnames=keys)
      w.writeheader()
      [w.writerow(r) for r in ev]
    dp = drift_places(args.drift, darr, base)
    with open(os.path.join(args.out, 'places_drift.csv'), 'w', newline='') as f:
      w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in dp for k in r)) or ['x'])
      w.writeheader()
      [w.writerow(r) for r in dp[:500]]
  if base_rep:
    report['base'] = base_rep.get('static')
    report['base_drift'] = base_rep.get('drift')
  json.dump(report, open(os.path.join(args.out, 'report.json'), 'w'), indent=1)
  open(os.path.join(args.out, 'report.md'), 'w').write(markdown(report))
  print(markdown(report))


if __name__ == '__main__':
  main()
