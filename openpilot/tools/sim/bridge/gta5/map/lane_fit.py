#!/usr/bin/env python3
"""Fits each way's lanes to the paint seen from above: where confident paint lines (drawn by 2 or more tiles) bound its
lanes, the lanes' edges are put on them, written as the OSM tags that place them (width:lanes[:forward|:backward],
width, placement:offset) into the map's lane geometry file (osm_lanes.GEOMETRY_FILE, `ways`, keyed by the way's end
nodes' GTA ids), which the build bakes into the pbf as those ways' tags (lane_geometry.py): lanes.json, nav, the map
driver and its labels all follow, and the pbf's ways, nodes and routing stay as they are. Fit a map built without it.

Each way is read with its strand (the ways the road carries straight on into with the same lane counts, WINDOW_M m
past each end; GTA's links are mostly about 12 m long), every STATION_M m, clear of junction nodes and of other roads
over or under it (stacked roads, whose paint the layer misreads; a way mostly under or over another is left alone). At
each station the paint's lines across it (paint_score.PaintIndex) are matched to the way's lane edges as the map has
them (match(): one edge on a line, the others on the nearest lines within a tolerance growing with the lanes between,
painted lines only, a centre line only on a yellow line, a line between lanes on a white one, at least two edges; a station where
a shift a lane away fits as well is ambiguous and dropped). Over the way: the median shift, if the stations agree (MAD
under AGREE_MAD, AGREE_SHARE of them within AGREE_TOL); each edge on its paint line where most stations found one (both
sides of a lane: its centre and width from the paint); an edge without, beside a lane with paint on its other side,
keeps that lane's width (one side: the lane shifted to its paint); else moved by the shift. Left alone, and listed: a
stacked way, stations disagreeing, a lane fitted under LANE_MIN m (a turn bay or a false line) or over LANE_MAX m, paint
lines inside a lane (the paint and the game files' lane counts disagree: the game files' counts are kept), a way whose
lanes the build measured from the game files' own paint (source:width=survey: where the paint seen from above has
them elsewhere, the game files are kept and the way is listed), a way whose
lanes widen or move along it (:start / :end tags), no unique key, or tags that can't say the fit. Too little paint
(MIN_STATIONS, MIN_SHARE): nothing to say.

usage: lane_fit.py MAP_DIR --paint paint_2tiles.json --out OUT_DIR [--write]
Writes OUT_DIR/fits.csv (every way fitted, moved or not), OUT_DIR/listed.csv (left alone, why) and summary.json; with
--write, the fits that move a lane edge MOVE_MIN m or more go into MAP_DIR's GEOMETRY_FILE `ways` (replacing the
section; the rest of the file is kept).
"""
import argparse
import csv
import json
import math
import os
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

from openpilot.tools.sim.bridge.gta5.map import osm_lanes
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, BOTH_WAYS, FORWARD, GEOMETRY_FILE, OsmLanes, WayLanes, oneway_of, \
  signed_metres

STATION_M = 2.0  # m between stations along a way
WINDOW_M = 24.0  # m past each end of a way, along the ways the road carries on into, its paint is also read over
END_SKIP = 12.0  # m from an end node where 3 or more ways meet that no station is taken (the junction's paint)
STACK_REACH, STACK_DZ = 10.0, 4.0  # m: a road this near across with this much height between is another level's
MAX_SHIFT = 3.5  # m a way's lanes may be found off the paint
MATCH_TOL, MATCH_GROW, MATCH_MAX = 0.4, 0.5, 1.2  # m from an edge, shifted, to a paint line it's on (see match())
AMBIGUOUS_M, AMBIGUOUS_COST = 1.0, 0.5  # m between two shifts fitting about as well (cost in tolerances): ambiguous
MIN_MATCHED = 2  # lane edges on paint lines at a station, at least
MIN_STATIONS, MIN_SHARE = 5, 0.3  # stations with a match: at least so many, and this share of the way's clear ones
AGREE_MAD, AGREE_TOL, AGREE_SHARE = 0.3, 0.4, 0.7  # m, m, share: the stations' shifts agree
EDGE_SHARE = 0.5  # of the matched stations an edge is on a paint line at for its place to be the paint's
LANE_MIN, LANE_MAX = 4.0, 6.5  # m: a lane fitted narrower is a bay or a false line, wider a missed line
ONE_SIDE_MIN, ONE_SIDE_MAX = 2.7, 6.5  # m: a lane with paint on one side keeps its width unless it falls outside these
INSIDE_SHARE = 0.5  # of the matched stations with a paint line inside a lane (a metre or more from its edges): counts disagree
MOVE_MIN = 0.25  # m an edge moves at least for the fit to be applied
VERIFY_TOL = 0.05  # m between the fitted edges and those the written tags give
ROADS = {'motorway', 'motorway_link', 'trunk', 'trunk_link', 'primary', 'primary_link', 'secondary', 'secondary_link',
         'tertiary', 'tertiary_link', 'unclassified', 'residential', 'living_street', 'road'}  # not service or track
WHITE, YELLOW, EDGE = {0, 1}, {2, 3}, {4}  # paint_score.PAINT_KINDS


def edges_of(road: WayLanes) -> tuple[np.ndarray, list[str]]:
  """The way's lane edges, m right of its line seen FORWARD, left to right, and what each must lie on: 'outer' (the
  lanes' outer edges: any line or kerb), 'centre' (between the directions: a yellow line), 'lane' (between lanes one way:
  white, or an edge where a lane meets a painted area). A median's two edges are each 'centre'."""
  sec = road.section(FORWARD)
  out, kinds = [], []
  for i, sp in enumerate(sec):
    if i == 0:
      out.append(sp.left)
      kinds.append('outer')
    else:
      prev = sec[i - 1]
      kind = 'lane' if prev.heading == sp.heading and sp.heading != 0 else 'centre'
      if sp.left - prev.right > 0.01:  # a median: both its edges
        out[-1:] = [prev.right]
        kinds[-1:] = ['centre']
        out.append(sp.left)
        kinds.append('centre')
      else:
        kinds[-1] = kind
    out.append(sp.right)
    kinds.append('outer')
  return np.array(out, float), kinds


def ok_kind(edge: str, k: int) -> bool:
  """Whether a paint line of kind k can be the lane edge: painted lines only (a kerb isn't a lane's edge), a centre
  line yellow, a line between lanes one way white."""
  if k in EDGE:
    return False
  return edge == 'outer' or (edge == 'centre' and k in YELLOW) or (edge == 'lane' and k in WHITE)


def match(edges: np.ndarray, kinds: list[str], paint: list[tuple[float, int]]):
  """The paint's lines the way's lane edges lie on at a station: one edge on a line (the anchor, within MAX_SHIFT of
  it), the others each on the line nearest where the anchor's shift puts it, within MATCH_TOL growing by MATCH_GROW an
  edge away from the anchor (lanes' widths add up), at most MATCH_MAX; in order, a line to each; painted lines only
  (ok_kind). The anchor and lines matching the most edges, the nearest in the tolerance, the smallest shift: (shift, each
  edge's line or NaN), None where fewer than MIN_MATCHED edges lie on lines, or another shift more than AMBIGUOUS_M away
  matches as many about as near (AMBIGUOUS_COST)."""
  if len(paint) < MIN_MATCHED:
    return None
  lat = np.array([q for q, _ in paint])
  pk = [k for _, k in paint]
  found = []
  for a, (ea, ka) in enumerate(zip(edges, kinds, strict=True)):
    for q0, k0 in paint:
      d = q0 - ea
      if abs(d) > MAX_SHIFT or not ok_kind(ka, k0):
        continue
      got = np.full(len(edges), np.nan)
      cost, last = 0.0, -np.inf
      for j, (ej, kj) in enumerate(zip(edges, kinds, strict=True)):
        tol = min(MATCH_TOL + MATCH_GROW * abs(j - a), MATCH_MAX)
        near = np.abs(lat - (ej + d))
        cand = [i for i in np.flatnonzero(near < tol) if ok_kind(kj, pk[i]) and lat[i] > last + 0.5]
        if not cand:
          continue
        i = min(cand, key=lambda i: near[i])
        got[j], last = lat[i], lat[i]
        cost += near[i] / tol
      n = int(np.isfinite(got).sum())
      if n < MIN_MATCHED:
        continue
      m = np.isfinite(got)
      found.append((n, cost, float(np.median(got[m] - edges[m])), got))
  if not found:
    return None
  found.sort(key=lambda f: (-f[0], round(f[1], 1), abs(f[2])))
  n, cost, d, got = found[0]
  # another shift a lane or so away fitting about as well: which lane is which can't be told here
  if any(f[0] == n and abs(f[2] - d) > AMBIGUOUS_M and f[1] <= cost + AMBIGUOUS_COST for f in found[1:]):
    return None
  return d, got


def strand(osm: OsmLanes, way: int, reach: float = WINDOW_M) -> list[np.ndarray]:
  """The way's line, run on back and on along the ways the road carries on into at nodes where only two meet with the
  same lane counts (OsmLanes._other), up to `reach` m past each end: [points before, the way's points, points after],
  each seen along the way."""
  counts = osm.lanes(way).counts
  out = []
  for at_end in (False, True):
    refs = osm.ways[way][1]
    node, w, dist, pts = (refs[-1] if at_end else refs[0]), way, 0.0, []
    for _ in range(32):
      found = osm._other(w, node)
      if found is None:
        break
      nb, _ = found
      if osm.lanes(nb).counts not in (counts, counts[1::-1] + counts[2:]):
        break
      q = osm.way_points(nb)
      nrefs = osm.ways[nb][1]
      q = q if (nrefs[0] == node) == at_end else q[::-1]  # running away from the way
      pts += list(q[1:])
      dist += float(np.hypot(*np.diff(q, axis=0).T).sum())
      node = nrefs[-1] if nrefs[0] == node else nrefs[0]
      w = nb
      if dist >= reach:
        break
    out.append(np.array(pts[::-1] if not at_end else pts, float).reshape(-1, 2))
  return [out[0], osm.way_points(way), out[1]]


def stations(osm: OsmLanes, way: int):
  """Points along the way and its strand within WINDOW_M of its ends, every STATION_M m, clear of nodes where 3 or more
  ways meet (END_SKIP): (point, unit heading along the way, height, on the way itself)."""
  before, own, after = strand(osm, way)
  pts = np.vstack([p for p in (before, own, after) if len(p)])
  keep = np.concatenate(([True], np.hypot(*np.diff(pts, axis=0).T) > 1e-6))
  pts = pts[keep]
  if len(pts) < 2:
    return []
  seg = np.hypot(*np.diff(pts, axis=0).T)
  along = np.concatenate(([0.0], np.cumsum(seg)))
  s0 = float(np.hypot(*np.diff(before, axis=0).T).sum()) + (float(np.hypot(*(own[0] - before[-1]))) if len(before) else 0.0)
  s1 = s0 + float(np.hypot(*np.diff(own, axis=0).T).sum())
  junctions = [float(along[i]) for i, q in enumerate(pts) if (n := osm.nodes_at(q, 0.3)) and osm.degree.get(n[0], 0) >= 3]
  refs = osm.ways[way][1]
  ele = [float(osm.data.node_tags.get(n, {}).get('ele', 'nan')) for n in (refs[0], refs[-1])]
  z = float(np.mean(ele)) if np.isfinite(ele).all() else math.nan
  out = []
  for v in np.arange(max(s0 - WINDOW_M, STATION_M / 2), min(s1 + WINDOW_M, along[-1]), STATION_M):
    if junctions and min(abs(v - j) for j in junctions) < END_SKIP:
      continue
    k = int(min(np.searchsorted(along, v, side='right') - 1, len(seg) - 1))
    u = (pts[k + 1] - pts[k]) / max(seg[k], 1e-9)
    out.append((pts[k] + u * (v - along[k]), u, z, s0 <= v <= s1))
  return out


class Stacked:
  """Whether another road passes over or under a point: roads.json's middles with their heights."""
  def __init__(self, roads_path: str):
    r = json.load(open(roads_path))
    pts, zs = [], []
    for way, h in zip(r['ways'], r['heights'], strict=True):
      q = np.asarray(way[3:], float).reshape(-1, 2)
      hz = np.asarray(h if isinstance(h, list) else [h] * len(q), float)
      if len(hz) != len(q):
        continue
      for i in range(len(q) - 1):
        n = max(int(math.ceil(np.hypot(*(q[i + 1] - q[i])) / 4.0)), 1)
        t = np.linspace(0, 1, n, endpoint=False)
        pts.append(q[i] + (q[i + 1] - q[i]) * t[:, None])
        zs.append(hz[i] + (hz[i + 1] - hz[i]) * t)
    self.p, self.z = np.concatenate(pts), np.concatenate(zs)
    self.tree = cKDTree(self.p)

  def at(self, p, z: float) -> bool:
    if not math.isfinite(z):
      return False
    idx = self.tree.query_ball_point(p, STACK_REACH)
    return bool(idx) and bool((np.abs(self.z[idx] - z) > STACK_DZ).any())


def fit_way(osm: OsmLanes, way: int, paint, stacked: Stacked) -> tuple[dict | None, str | None, dict]:
  """(the fit: edges before and after, how they were found; None where none), why it's left alone (None where it
  isn't), and counts."""
  tags = osm.ways[way][0]
  info: dict = {}
  if tags.get('highway') not in ROADS:
    return None, None, info
  road = osm.lanes(way)
  if not road.lanes or not road.markings:
    return None, None, info
  if osm.has_ends(way) or osm.moves_across(way):
    return None, 'lanes widen or move along the way (tagged :start / :end)', info
  edges, kinds = edges_of(road)
  st = stations(osm, way)
  if not st:
    return None, None, info
  clear, shifts, found, inside = 0, [], [], 0
  own = sum(1 for *_, mine in st if mine)
  own_clear = 0
  for p, u, z, mine in st:
    if stacked.at(p, z):
      continue
    clear += 1
    own_clear += mine
    paint_lines = paint.beside(p, u, z)
    m = match(edges, kinds, paint_lines)
    if m is None:
      continue
    d, got = m
    shifts.append(d)
    found.append(got)
    # paint lines inside a lane, a metre or more from its edges as fitted here
    fitted = np.where(np.isfinite(got), got, edges + d)
    lat = np.array([q for q, _ in paint_lines])
    for a, b in zip(fitted, fitted[1:], strict=False):
      if b - a > 2.0 and ((lat > a + 1.0) & (lat < b - 1.0)).any():
        inside += 1
        break
  info = {'stations': len(st), 'own': own, 'clear': clear, 'matched': len(shifts)}
  if (own and own_clear < 0.5 * own) or clear < 0.5 * len(st):  # the paint misreads stacked roads: not fitted there
    return None, 'stacked: another road over or under it', info
  if clear < MIN_STATIONS:
    return None, None, info
  if len(shifts) < MIN_STATIONS or len(shifts) < MIN_SHARE * clear:
    return None, None, info  # too little paint: nothing to say
  shifts = np.array(shifts)
  med = float(np.median(shifts))
  mad = float(np.median(np.abs(shifts - med)))
  agree = float(np.mean(np.abs(shifts - med) < AGREE_TOL))
  info.update(shift=round(med, 2), mad=round(mad, 2), agree=round(agree, 2))
  if mad > AGREE_MAD or agree < AGREE_SHARE:
    return None, f'the paint disagrees along it (shift {med:+.2f} m, MAD {mad:.2f}, {agree:.0%} agree)', info
  if inside >= INSIDE_SHARE * len(found):
    return None, f'paint lines inside its lanes on {inside} of {len(found)} stations: the paint and the game files\' lane counts disagree', info
  found = np.array(found)
  on = np.isfinite(found)
  share = on.mean(axis=0)
  after = edges + med
  paint_side = share >= EDGE_SHARE
  for j in range(len(edges)):
    if paint_side[j]:
      after[j] = float(np.median(found[on[:, j], j]))
  # an edge with no paint of its own beside a lane with paint on its other side: that lane keeps its width (in range),
  # shifted to its paint (where lanes either side both say, half way between)
  for j in range(len(edges)):
    if paint_side[j]:
      continue
    say = []
    if j > 0 and paint_side[j - 1] and ONE_SIDE_MIN <= edges[j] - edges[j - 1] <= ONE_SIDE_MAX and not kinds[j - 1] == kinds[j] == 'centre':
      say.append(after[j - 1] + edges[j] - edges[j - 1])
    if j + 1 < len(edges) and paint_side[j + 1] and ONE_SIDE_MIN <= edges[j + 1] - edges[j] <= ONE_SIDE_MAX and not kinds[j] == kinds[j + 1] == 'centre':
      say.append(after[j + 1] - (edges[j + 1] - edges[j]))
    if say:
      after[j] = float(np.mean(say))
  widths = np.diff(after)
  lane_w = [w for w, a, b in zip(widths, kinds, kinds[1:], strict=False) if not (a == 'centre' and b == 'centre')]  # not a median
  info.update(edges_on_paint=int(paint_side.sum()), edges=len(edges))
  if min(lane_w) < LANE_MIN:
    return None, f'a lane fitted {min(lane_w):.1f} m wide (under {LANE_MIN} m: a bay or a false line)', info
  if max(lane_w) > LANE_MAX:
    return None, f'a lane fitted {max(lane_w):.1f} m wide (over {LANE_MAX} m: a line missed)', info
  return {'before': edges, 'after': after, 'kinds': kinds, 'paint_side': paint_side}, None, info


def tags_for(osm: OsmLanes, way: int, after: np.ndarray) -> dict | None:
  """The tags placing the way's lanes on the fitted edges (m right of its line seen FORWARD): width:lanes (one-way) or
  width:lanes:forward / :backward, width (kerb to kerb, its parking, shoulders and margins kept) and placement:offset,
  as {tag: value, or None where the tag goes}; None where they can't place them (checked by reading them back)."""
  tags = dict(osm.ways[way][0])
  sec = WayLanes.from_tags(tags, osm.drive_on_right, osm.defaults).section(FORWARD)
  if any(sp.lane.direction == BOTH_WAYS for sp in sec):
    return None
  spans_e, j = [], 0  # each lane's fitted edges, in the section's order
  for i, sp in enumerate(sec):
    if i and sp.left - sec[i - 1].right > 0.01:
      j += 1
    spans_e.append((float(after[j]), float(after[j + 1])))
    j += 1
  widths = [b - a for a, b in spans_e]
  gap = lambda spans: sum(max(spans[i + 1][0] - spans[i][1], 0.0) for i in range(len(spans) - 1))  # noqa: E731
  new: dict = {}
  oneway = oneway_of(tags)
  if oneway:
    new['width:lanes'] = '|'.join(metres_str(w) for w in (widths if oneway == 1 else widths[::-1]))
  else:
    new['width:lanes:forward'] = '|'.join(metres_str(w) for w, sp in zip(widths, sec, strict=True) if sp.lane.direction == FORWARD)
    new['width:lanes:backward'] = '|'.join(metres_str(w) for w, sp in zip(widths[::-1], sec[::-1], strict=True) if sp.lane.direction == BACKWARD)
    new['width:lanes'] = None
  if (w_tag := osm_lanes.metres(tags.get('width'))) is not None:
    old_lanes = sum(sp.right - sp.left for sp in sec)
    new['width'] = metres_str(w_tag - old_lanes - gap([(sp.left, sp.right) for sp in sec]) + sum(widths) + gap(spans_e))

  def read(offset: float):
    t = {**tags, **new}
    t['placement:offset'] = metres_str(offset) if abs(offset) > 1e-3 else None
    return WayLanes.from_tags({k: v for k, v in t.items() if v is not None}, osm.drive_on_right, osm.defaults).section(FORWARD)
  base = signed_metres(tags.get('placement:offset')) or 0.0
  got = read(base)
  if len(got) != len(sec):
    return None
  off = base + spans_e[0][0] - got[0].left
  got = read(off)
  if max(max(abs(g.left - a), abs(g.right - b)) for g, (a, b) in zip(got, spans_e, strict=True)) > VERIFY_TOL:
    return None
  new['placement:offset'] = metres_str(off) if abs(off) > 1e-3 else None
  return {k: v for k, v in new.items() if v != tags.get(k)}


def why_kind(why: str) -> str:
  """A listed way's reason, without its numbers."""
  for key, kind in (('stacked', 'stacked'), ('disagrees along', 'paint disagrees along it'), ('lane counts', 'lane counts disagree'),
                    (f'under {LANE_MIN}', f'lane under {LANE_MIN} m'), (f'over {LANE_MAX}', f'lane over {LANE_MAX} m'),
                    (':start', 'lanes widen or move along it'), ('unique key', 'no unique key'), ('tags cannot', 'tags cannot place it'),
                    ('game files measured', 'game files measured it (paint disagrees)')):
    if key in why:
      return kind
  return why


def metres_str(m: float) -> str:
  return f'{m:.2f}'.rstrip('0').rstrip('.') if abs(m) >= 0.005 else '0'


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('map_dir')
  p.add_argument('--paint', required=True, help='the confident paint drawn by 2+ tiles, lanes.json format')
  p.add_argument('--out', required=True)
  p.add_argument('--write', action='store_true', help="write the fits into MAP_DIR's lane geometry file")
  args = p.parse_args()
  os.makedirs(args.out, exist_ok=True)
  from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
  from openpilot.tools.sim.bridge.gta5.map.paint_score import PaintIndex
  osm = OsmLanes.load(os.path.join(args.map_dir, 'gta5.osm.pbf'), to_game)
  paint = PaintIndex(args.paint)
  stacked = Stacked(os.path.join(args.map_dir, 'roads.json'))
  keys = Counter(osm.way_key(w) for w in osm.ways)
  fits, listed, written = [], [], {}
  for n, way in enumerate(osm.ways):
    if n % 5000 == 0:
      print(n, len(osm.ways), len(fits), len(listed), flush=True)
    fit, why, info = fit_way(osm, way, paint, stacked)
    key = osm.way_key(way)
    tags = osm.ways[way][0]
    pts = osm.way_points(way)
    mid = pts[len(pts) // 2]
    row = {'key': key, 'way': way, 'class': tags.get('highway'), 'x': round(float(mid[0]), 1), 'y': round(float(mid[1]), 1),
           'lanes': tags.get('lanes'), **info}
    if why:
      listed.append({**row, 'why': why})
      continue
    if fit is None:
      continue
    moved = float(np.abs(fit['after'] - fit['before']).max())
    row.update(moved_m=round(moved, 2), before='|'.join(f'{v:.2f}' for v in fit['before']), after='|'.join(f'{v:.2f}' for v in fit['after']),
               on_paint=''.join('P' if s else '.' for s in fit['paint_side']))
    if moved < MOVE_MIN:
      fits.append({**row, 'applied': False})
      continue
    if tags.get('source:width') == 'survey':  # the build measured these lanes from the game files' own paint: kept
      listed.append({**row, 'why': f'game files measured its lanes (source:width=survey); the paint has them up to {moved:.2f} m off'})
      continue
    if key is None or keys[key] > 1:
      listed.append({**row, 'why': 'no unique key (its end nodes have no GTA ids, or another way has them)'})
      continue
    new = tags_for(osm, way, fit['after'])
    if new is None:
      listed.append({**row, 'why': 'its tags cannot place the fitted lanes'})
      continue
    fits.append({**row, 'applied': True})
    written[key] = {'tags': new, 'class': tags.get('highway'), 'x': row['x'], 'y': row['y'],
                    'moved_m': row['moved_m'], 'before': row['before'], 'after': row['after'], 'on_paint': row['on_paint'],
                    'shift_m': info.get('shift'), 'stations': info.get('matched')}
  for name, rows in (('fits', fits), ('listed', listed)):
    with open(os.path.join(args.out, f'{name}.csv'), 'w', newline='') as f:
      keys_ = list(dict.fromkeys(k for r in rows for k in r))
      w = csv.DictWriter(f, fieldnames=keys_ or ['key'], lineterminator='\n')
      w.writeheader()
      [w.writerow(r) for r in rows]
  applied = [r for r in fits if r['applied']]
  summary = {'ways_fitted': len(fits), 'applied': len(applied), 'listed': len(listed),
             'listed_why': Counter(why_kind(r['why']) for r in listed).most_common(),
             'applied_by_class': Counter(r['class'] for r in applied).most_common(),
             'moved_over_1m': sum(r['moved_m'] > 1.0 for r in applied), 'moved_over_2m': sum(r['moved_m'] > 2.0 for r in applied),
             'largest': sorted(applied, key=lambda r: -r['moved_m'])[:20]}
  json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1)
  if args.write:
    path = os.path.join(args.map_dir, GEOMETRY_FILE)
    doc = json.load(open(path)) if os.path.exists(path) else {'version': 1}
    doc['ways'] = written
    doc.setdefault('notes', {})['ways'] = ('lanes fitted to the confident paint drawn by 2+ tiles (lane_fit.py): each way '
                                           '(a>b, its end nodes\' GTA ids) gets these tags over the pbf\'s (null: the tag goes); '
                                           'before/after are its lane edges m right of its line seen along it, on_paint P where the '
                                           'edge is on a paint line')
    json.dump(doc, open(path, 'w'), indent=1, sort_keys=True)
  print(json.dumps({k: v for k, v in summary.items() if k != 'largest'}, indent=1))


if __name__ == '__main__':
  main()
