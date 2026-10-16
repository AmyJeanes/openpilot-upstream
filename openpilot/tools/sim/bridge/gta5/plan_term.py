#!/usr/bin/env python3
"""The map driver's plan against the map and the paint at turns, on routes through given places (mapdrive_sim.py on
the lagged car): each trip routed by Valhalla as the bridge routes it (mapdrive_sim.trip_route) and driven by MapDrive.
Per frame: the plan term (MapDrive's intent, the planned line before its bias and wander, less the map's target lane
centre where the car is: target_right), the map term (that target centre less the centre of the paint lane it is in,
paint_score.PaintIndex) and the car's own distance from its paint lane's centre; summed up at turn entries and exits
(TURN_ZONE m either side of a turn's corner, RouteLanes.corners of CLAMP_TURN deg or more) and elsewhere.

usage: plan_term.py SPECS.json --paint paint_lanes_confident.json --out OUT_DIR [--router URL]
(the live map from GTA5_MAP or ~/gta5map_lanes; SPECS.json: [{"name", "spec": "x,y,z,heading,lane>dx,dy"}];
GTA5_TRIP_ROUTES: where the routes are kept, so a later run drives the same ones.)
"""
import argparse
import json
import math
import os

import numpy as np

TURN_ZONE = 30.0  # m before and after a turn's corner its entry and exit are read over
SECONDS = 90.0  # s a trip is driven at most


def lateral_to(line: np.ndarray, p: np.ndarray, h: float) -> float:
  """m right of the car (at p, heading h, game degrees) that a polyline is where it crosses the car's lateral axis."""
  if line is None or len(line) < 2:
    return math.nan
  a, b = line[:-1], line[1:]
  ab = b - a
  t = np.clip(np.einsum('ij,ij->i', p - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-9), 0.0, 1.0)
  near = a + ab * t[:, None]
  d = np.hypot(*(near - p).T)
  i = int(np.argmin(d))
  if d[i] > 8.0:
    return math.nan
  hr = math.radians(h)
  return float((near[i] - p) @ np.array([math.cos(hr), math.sin(hr)]))


def drive_spec(spec: str, paint, router_url: str) -> dict | None:
  from openpilot.tools.sim.bridge.gta5 import mapdrive_sim as sim
  from openpilot.tools.sim.bridge.gta5.gta5_mapdrive import CLAMP_TURN, MapDrive
  from openpilot.tools.sim.bridge.gta5.map.paint_score import bracket, lane_ok
  got = sim.trip_route(spec, router_url)
  if got is None:
    return None
  route, pose = got
  rows = []

  class Recorded(MapDrive):
    def step(self, route_, state, t, collisions=0):
      out = super().step(route_, state, t, collisions)
      p = np.array(state['pos'][:2], float)
      h = float(state['heading'])
      T = (self.target_right - route_.right) if math.isfinite(self.target_right) else math.nan
      ilat = lateral_to(getattr(self, 'intent', None), p, h)
      hr = math.radians(h)
      u = np.array([-math.sin(hr), math.cos(hr)])
      lines = paint.beside(p, u)
      mp = math.nan
      if math.isfinite(T):
        lt, rt = bracket(lines, T)
        mp = T - (lt[0] + rt[0]) / 2 if lane_ok(lt, rt) else math.nan
      lc, rc = bracket(lines, 0.0)
      car = -(lc[0] + rc[0]) / 2 if lane_ok(lc, rc) else math.nan
      rows.append((t, p[0], p[1], h, float(state['vEgo']), float(route_.at), T, ilat, ilat - T if math.isfinite(T) else math.nan, mp, car))
      return out
  real = sim.MapDrive
  sim.MapDrive = Recorded
  try:
    trip = sim.drive(route, pose=pose, seconds=SECONDS)
  finally:
    sim.MapDrive = real
  r = np.array(rows, float).reshape(-1, 11)
  corners = [sc for sc, turned in route.lanes.corners if abs(turned) >= CLAMP_TURN] if route.lanes is not None else []
  zone = np.zeros(len(r), int)  # 0 elsewhere, -1 turn entry, 1 turn exit
  for sc in corners:
    rel = r[:, 5] - sc
    zone[(rel >= -TURN_ZONE) & (rel < 0)] = -1
    zone[(rel >= 0) & (rel <= TURN_ZONE)] = 1
  return {'rows': r, 'zone': zone, 'corners': corners, 'finished': trip.finished, 'length': float(route.length)}


def stats(r: np.ndarray, m: np.ndarray) -> dict:
  def one(col):
    e = np.abs(r[m, col])
    k = np.isfinite(e)
    return {'frames': int(m.sum()), 'known': int(k.sum()), 'over_0.5': round(float((e[k] > 0.5).mean()), 3) if k.any() else None,
            'over_1.0': round(float((e[k] > 1.0).mean()), 3) if k.any() else None, 'p90': round(float(np.percentile(e[k], 90)), 2) if k.any() else None,
            'max': round(float(e[k].max()), 2) if k.any() else None}
  return {'plan': one(8), 'map': one(9), 'car': one(10)}


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('specs')
  p.add_argument('--paint', required=True)
  p.add_argument('--out', required=True)
  p.add_argument('--router', default='http://127.0.0.1:8002')
  args = p.parse_args()
  os.makedirs(args.out, exist_ok=True)
  from openpilot.tools.sim.bridge.gta5.map.paint_score import PaintIndex
  paint = PaintIndex(args.paint)
  report = {}
  allrows, allzone = [], []
  for item in json.load(open(args.specs)):
    got = drive_spec(item['spec'], paint, args.router)
    if got is None:
      report[item['name']] = None
      continue
    r, zone = got['rows'], got['zone']
    np.save(os.path.join(args.out, f"{item['name']}.npy"), np.column_stack([r, zone]))
    report[item['name']] = {'spec': item['spec'], 'finished': got['finished'], 'corners': [round(c, 1) for c in got['corners']],
                            'entry': stats(r, (zone == -1) & (r[:, 4] > 2.0)), 'exit': stats(r, (zone == 1) & (r[:, 4] > 2.0)),
                            'elsewhere': stats(r, (zone == 0) & (r[:, 4] > 2.0))}
    allrows.append(r)
    allzone.append(zone)
  if allrows:
    r, zone = np.vstack(allrows), np.concatenate(allzone)
    report['all'] = {'entry': stats(r, (zone == -1) & (r[:, 4] > 2.0)), 'exit': stats(r, (zone == 1) & (r[:, 4] > 2.0)),
                     'elsewhere': stats(r, (zone == 0) & (r[:, 4] > 2.0))}
  json.dump(report, open(os.path.join(args.out, 'report.json'), 'w'), indent=1)
  print(json.dumps(report.get('all'), indent=1))


if __name__ == '__main__':
  main()
