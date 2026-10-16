#!/usr/bin/env python3
"""Every freeway diverge and merge on a map (exits, on-ramps, GTA's lane chains splitting off and joining), with
osm_lanes' RouteLanes._diverges on and off (DIVERGES): how far the branch's lanes move, how far they sit from the
paint lane's centre each way (the confident paint seen from above, paint_score.PaintIndex), and how many m of their
centres run inside a hatched area (lanes.json's, router.LaneLines.inside) each way.

usage: diverge_report.py MAP_DIR --paint paint_lanes_confident.json --out OUT_DIR [--keep]
Writes OUT_DIR/diverges.csv (a row per place where a lane moves) and summary.json. With --keep, the places where the
paint has the branch's lanes nearer where they were (by KEEP_WORSE m on average, the paint covering KEEP_KNOWN of the
stretch) go into MAP_DIR's lanegeom.json as diverges to keep (osm_lanes.GEOMETRY_FILE), with the evidence; the build
bakes them into the pbf (lane_geometry.py: osm_lanes.RUN_ON_TAG), and a run on that map reports it as it will be.
"""
import argparse
import csv
import json
import math
import os

import numpy as np

from openpilot.tools.sim.bridge.gta5.map import osm_lanes
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import HOP_ROADS, HOP_TURN, RouteLanes, oneway_of, ways_from_nodes

BACK_M, ON_M = 60.0, 140.0  # m of road before the node and of branch past it in each route
STEP = 1.0  # m between the samples along the branch
MOVED = 0.1  # m: a lane moved less than this is unchanged
KEEP_WORSE, KEEP_KNOWN = 0.1, 0.3  # m further from the paint lane's centre on average; share of the stretch with paint


def drivable(osm, a: int, b: int) -> bool:
  w, fwd = osm.pairs[(a, b)]
  one = oneway_of(osm.ways[w][0])
  return one == 0 or (one == 1) == fwd


def heading(osm, a: int, b: int) -> np.ndarray:
  v = osm.node_xy(b) - osm.node_xy(a)
  return v / max(float(np.hypot(*v)), 1e-9)


def walk(osm, a: int, b: int, length: float, forward: bool) -> list[int]:
  """From the link a -> b (driven that way), on (or back, with forward False) along the straightest drivable way at
  each node for `length` m: the nodes, a and b first."""
  nodes, dist = [a, b], float(np.hypot(*(osm.node_xy(b) - osm.node_xy(a))))
  while dist < length and len(nodes) < 200:
    p, q = nodes[-2], nodes[-1]
    u = heading(osm, p, q)
    best, best_cos = None, math.cos(math.radians(HOP_TURN))
    for r in osm.links.get(q, ()):
      if r in nodes or not (drivable(osm, q, r) if forward else drivable(osm, r, q)):
        continue
      c = float(u @ heading(osm, q, r))
      if c > best_cos:
        best, best_cos = r, c
    if best is None:
      break
    dist += float(np.hypot(*(osm.node_xy(best) - osm.node_xy(q))))
    nodes.append(best)
  return nodes


def places(osm):
  """(node, the route's nodes through it, 'diverge' | 'merge'): each freeway node where a way with fewer lanes leaves
  (joins) a one-way freeway way with more, within HOP_TURN of it."""
  out = []
  for n, deg in osm.degree.items():
    if deg < 3:
      continue
    near = osm.links.get(n, ())
    ins = [m for m in near if drivable(osm, m, n)]
    outs = [m for m in near if drivable(osm, n, m)]
    for a in ins:
      wa = osm.pairs[(a, n)][0]
      for b in outs:
        if b == a:
          continue
        wb = osm.pairs[(n, b)][0]
        ta, tb = osm.ways[wa][0], osm.ways[wb][0]
        if ta.get('highway') not in HOP_ROADS or tb.get('highway') not in HOP_ROADS:
          continue
        if float(heading(osm, a, n) @ heading(osm, n, b)) < math.cos(math.radians(HOP_TURN)):
          continue
        la = osm.lanes(wa).counts[0 if osm.pairs[(a, n)][1] else 1]
        lb = osm.lanes(wb).counts[0 if osm.pairs[(n, b)][1] else 1]
        if la == lb:
          continue
        back = walk(osm, n, a, BACK_M if la > lb else ON_M, forward=False)[::-1]  # ..., a, n
        on = walk(osm, n, b, ON_M if la > lb else BACK_M, forward=True)  # n, b, ...
        out.append((n, back[:-1] + on, 'diverge' if la > lb else 'merge'))
  return out


def lane_lines(lanes: RouteLanes, pts: np.ndarray, s0: float, s1: float):
  """Each of our lanes' centres along s0..s1 (where the lane count stays that of s0's): [lane][(point, heading)]."""
  out = None
  for v in np.arange(s0 + STEP / 2, s1, STEP):
    k = lanes.segment(v)
    sec = lanes.section_at(float(v), k)
    if sec is None or not sec.lanes:
      continue
    d = pts[k + 1] - pts[k]
    d = d / max(float(np.hypot(*d)), 1e-9)
    p = np.array([np.interp(v, lanes.along, pts[:, 0]), np.interp(v, lanes.along, pts[:, 1])])
    if out is None:
      out = [[] for _ in sec.ours]
    if len(sec.ours) != len(out):
      break
    for i, sp in enumerate(sec.ours):
      out[i].append((p + np.array([d[1], -d[0]]) * sp.centre, d, v))
  return out or []


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('map_dir')
  p.add_argument('--paint', required=True)
  p.add_argument('--out', required=True)
  p.add_argument('--keep', action='store_true', help="write the diverges the paint has better as they were to the map's lanegeom.json")
  args = p.parse_args()
  os.makedirs(args.out, exist_ok=True)
  from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
  from openpilot.tools.sim.bridge.gta5.map.paint_score import PaintIndex
  from openpilot.tools.sim.bridge.gta5.map.router import LaneLines
  osm = osm_lanes.OsmLanes.load(os.path.join(args.map_dir, 'gta5.osm.pbf'), to_game)
  hatched = LaneLines.of(osm)
  paint = PaintIndex(args.paint)
  rows = []
  found = places(osm)
  print(len(found), 'diverges and merges', flush=True)
  for n, nodes, kind in found:
    pts = osm.xy[osm.data.index(nodes)]
    ele = [float(osm.data.node_tags.get(m, {}).get('ele', 'nan')) for m in nodes]
    k = nodes.index(n)
    wa = ways_from_nodes(pts, osm)
    lines = {}
    for on in (False, True):
      osm_lanes.DIVERGES = on
      lanes = RouteLanes.from_osm(pts, wa, osm)
      a, b = (float(lanes.along[k]), float(lanes.along[-1])) if kind == 'diverge' else (float(lanes.along[0]), float(lanes.along[k]))
      lines[on] = lane_lines(lanes, pts, a, b)
    osm_lanes.DIVERGES = True
    old, new = lines[False], lines[True]
    if not old or len(old) != len(new) or len(old[0]) != len(new[0]):
      continue
    moved = max(float(np.hypot(*(np.array([q for q, _, _ in lo]) - np.array([q for q, _, _ in ln])).T).max()) for lo, ln in zip(old, new, strict=True))
    if moved < MOVED:
      continue
    z = float(np.nanmean(ele)) if np.isfinite(ele).any() else math.nan
    stats = {}
    for tag, ls in (('before', old), ('after', new)):
      errs, inside = [], 0.0
      for lane in ls:
        for (q, u, _), (q2, _, _) in zip(lane, lane[1:], strict=False):
          errs.append(paint.lane_error(q, u, z))
          if hatched is not None:
            inside += hatched.inside(q, q2, z, z)
      e = np.abs(np.array(errs, float))
      known = np.isfinite(e)
      stats[tag] = (round(float(np.mean(e[known])), 2) if known.any() else None, round(float(np.mean(e[known] > 1.0)), 3) if known.any() else None,
                    round(float(known.mean()), 2) if len(e) else None, round(inside, 1))
    xy = osm.node_xy(n)
    a_, b_ = (nodes[k], nodes[k + 1]) if kind == 'diverge' else (nodes[k - 1], nodes[k])
    rows.append({'key': f"{osm.gta(a_)}>{osm.gta(b_)}", 'node': osm.data.node_tags.get(n, {}).get('gta:node', n), 'x': round(float(xy[0]), 1), 'y': round(float(xy[1]), 1),
                 'kind': kind, 'lanes': len(new), 'moved_m': round(moved, 2),
                 'paint_err_before': stats['before'][0], 'paint_err_after': stats['after'][0],
                 'over1_before': stats['before'][1], 'over1_after': stats['after'][1], 'paint_known': stats['after'][2],
                 'hatched_m_before': stats['before'][3], 'hatched_m_after': stats['after'][3]})
  rows.sort(key=lambda r: -r['moved_m'])
  with open(os.path.join(args.out, 'diverges.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['node'], lineterminator='\n')
    w.writeheader()
    [w.writerow(r) for r in rows]

  def mean(key, rs):
    v = [r[key] for r in rs if r[key] is not None]
    return round(float(np.mean(v)), 3) if v else None
  known = [r for r in rows if r['paint_err_before'] is not None and r['paint_err_after'] is not None]
  summary = {'places': len(found), 'changed': len(rows), 'diverges': sum(r['kind'] == 'diverge' for r in rows),
             'merges': sum(r['kind'] == 'merge' for r in rows), 'moved_over_1m': sum(r['moved_m'] > 1 for r in rows),
             'moved_over_3m': sum(r['moved_m'] > 3 for r in rows), 'largest': rows[:10],
             'with_paint': len(known), 'paint_err_before': mean('paint_err_before', known), 'paint_err_after': mean('paint_err_after', known),
             'over1_before': mean('over1_before', known), 'over1_after': mean('over1_after', known),
             'paint_better': sum(r['paint_err_after'] < r['paint_err_before'] - 0.1 for r in known),
             'paint_worse': sum(r['paint_err_after'] > r['paint_err_before'] + 0.1 for r in known),
             'hatched_m_before': round(sum(r['hatched_m_before'] for r in rows), 1), 'hatched_m_after': round(sum(r['hatched_m_after'] for r in rows), 1),
             'into_hatched': [r for r in rows if r['hatched_m_after'] > r['hatched_m_before'] + 0.5]}
  summary['kept_by_paint_already'] = sorted({osm.way_key(w) for w, v in osm.ways.items()
                                              if v[0].get(osm_lanes.RUN_ON_TAG) == osm_lanes.RUN_ON_NO} - {None})
  if args.keep:
    keep = {r['key']: {'keep': True, 'x': r['x'], 'y': r['y'], 'kind': r['kind'],
                       'why': f"paint lane centre {r['paint_err_before']} m off as the map has it, {r['paint_err_after']} m moved "
                              f"(over 1 m {r['over1_before']} -> {r['over1_after']}, paint on {r['paint_known']} of it)"}
            for r in known if r['paint_err_after'] > r['paint_err_before'] + KEEP_WORSE and (r['paint_known'] or 0) >= KEEP_KNOWN}
    path = os.path.join(args.map_dir, osm_lanes.GEOMETRY_FILE)
    doc = json.load(open(path)) if os.path.exists(path) else {'version': 1}
    doc['diverges'] = {**doc.get('diverges', {}), **keep}
    doc.setdefault('notes', {})['diverges'] = ('freeway diverges and merges whose branch lanes osm_lanes._diverges would move further '
                                               'from the confident paint (diverge_report.py --keep): kept as the map has them')
    json.dump(doc, open(path, 'w'), indent=1, sort_keys=True)
    summary['kept_by_paint_written'] = sorted(keep)
  json.dump(summary, open(os.path.join(args.out, 'summary.json'), 'w'), indent=1)
  print(json.dumps({k: v for k, v in summary.items() if k not in ('largest',)}, indent=1))


if __name__ == '__main__':
  main()
