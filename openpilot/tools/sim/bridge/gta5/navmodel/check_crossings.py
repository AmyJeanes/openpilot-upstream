#!/usr/bin/env python3
"""How the nav map draws GTA's overpasses: every place two drawn roads of the pbf cross with no node in common, by
their heights there (the nodes' ele) and what the renderer does with them. It has three levels (tunnels, ground,
bridges) and cuts a road under one drawn above it; it reads only is_bridge / is_tunnel (build_pmtiles.py), not layer.

  python check_crossings.py ~/gta5map/gta5.osm.pbf [--examples out.json] [--frames gta5_roads.pmtiles out_dir]

--frames renders a few of each kind, centred on the crossing and headed along the lower road, the route along it
(crossings_<kind>.png).

Kinds of crossing, by the upper road (more than OVER m above the other) and the lower:
- cut: the upper is a bridge, the lower isn't: drawn as on a real map, the lower cut under the upper;
- bridge_on_bridge: both are bridges (GTA's stacked interchanges): one level, no cut between them, as on a real map;
- untagged: the upper isn't a bridge: no cut, the two drawn alike, crossing as if at a junction;
- inverted: the lower is a bridge and the upper isn't: the lower drawn over the upper, cutting it;
- level: within OVER m of each other: a crossing at grade with no node, where GTA's links cross.
"""
import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
from openpilot.tools.sim.bridge.gta5.navmodel.build_pmtiles import road_properties

OVER = 4.0  # m between two roads' heights where they cross: one passes over the other
CELL = 50.0  # m squares the segments are found by


def read(pbf: str):
  import osmium

  ele: dict[int, float] = {}
  ways = []
  for o in osmium.FileProcessor(pbf, osmium.osm.NODE | osmium.osm.WAY).with_locations():
    if o.is_node():
      if "ele" in o.tags:
        ele[o.id] = float(o.tags["ele"])
      continue
    props = road_properties(dict(o.tags))
    if props is None or props["kind"] == "path":
      continue
    nodes = [n for n in o.nodes if n.location.valid()]
    if len(nodes) < 2:
      continue
    x, y = to_game(np.array([n.lat for n in nodes]), np.array([n.lon for n in nodes]))
    ways.append((o.id, [n.ref for n in nodes], np.column_stack((x, y)), props))
  return ele, ways


def crossings(ele: dict[int, float], ways) -> list[dict]:
  grid = defaultdict(list)
  segs = []  # (way index, k)
  for w, (_, _, xy, _) in enumerate(ways):
    for k in range(len(xy) - 1):
      a, b = xy[k], xy[k + 1]
      lo, hi = np.floor(np.minimum(a, b) / CELL).astype(int), np.floor(np.maximum(a, b) / CELL).astype(int)
      for cx in range(lo[0], hi[0] + 1):
        for cy in range(lo[1], hi[1] + 1):
          grid[(cx, cy)].append(len(segs))
      segs.append((w, k))
  found = {}
  for cell in grid.values():
    for i in range(len(cell)):
      wi, ki = segs[cell[i]]
      _, ni, pi, _ = ways[wi]
      for j in range(i + 1, len(cell)):
        wj, kj = segs[cell[j]]
        if wi == wj or (min(wi, wj), max(wi, wj)) in found:
          continue
        _, nj, pj, _ = ways[wj]
        if {ni[ki], ni[ki + 1]} & {nj[kj], nj[kj + 1]}:
          continue
        hit = intersect(pi[ki], pi[ki + 1], pj[kj], pj[kj + 1])
        if hit is None:
          continue
        t, u, p = hit
        zi = height(ele, ni[ki], ni[ki + 1], t)
        zj = height(ele, nj[kj], nj[kj + 1], u)
        if zi is None or zj is None:
          continue
        found[(min(wi, wj), max(wi, wj))] = (wi, wj, zi, zj, p)
  out = []
  for wi, wj, zi, zj, p in found.values():
    upper, lower = (wi, wj) if zi >= zj else (wj, wi)
    dz = abs(zi - zj)
    bu, bl = bool(ways[upper][3].get("is_bridge")), bool(ways[lower][3].get("is_bridge"))
    kind = "level" if dz <= OVER else "bridge_on_bridge" if bu and bl else "cut" if bu else "inverted" if bl else "untagged"
    out.append({"kind": kind, "at": [round(float(p[0]), 1), round(float(p[1]), 1)], "dz": round(dz, 1),
                "upper": ways[upper][3]["kind_detail"], "lower": ways[lower][3]["kind_detail"],
                "upper_way": ways[upper][0], "lower_way": ways[lower][0]})
  return out


def intersect(a, b, c, d):
  r, s = b - a, d - c
  den = r[0] * s[1] - r[1] * s[0]
  if abs(den) < 1e-9:
    return None
  t = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / den
  u = ((c[0] - a[0]) * r[1] - (c[1] - a[1]) * r[0]) / den
  if 0 < t < 1 and 0 < u < 1:
    return t, u, a + t * r
  return None


def height(ele, n0: int, n1: int, t: float) -> float | None:
  if n0 not in ele or n1 not in ele:
    return None
  return ele[n0] + (ele[n1] - ele[n0]) * t


def frames(tiles: str, out: str, found: list[dict], ways, per_kind: int = 8, seed: int = 0):
  from pathlib import Path

  from openpilot.selfdrive.navigation.local_tiles import LocalTiles
  from openpilot.selfdrive.navigation.pmtiles import PMTiles
  from openpilot.selfdrive.navigation.renderer import MapRenderer
  from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon
  from openpilot.tools.sim.bridge.gta5.navmodel.render_frames import contact_sheet

  renderer = MapRenderer(PMTiles(LocalTiles(tiles)))
  xy_of = {w[0]: w[2] for w in ways}
  rng = np.random.default_rng(seed)
  Path(out).mkdir(parents=True, exist_ok=True)
  for kind in ("cut", "bridge_on_bridge", "level", "untagged", "inverted"):
    some = [c for c in found if c["kind"] == kind]
    if not some:
      continue
    sheet = []
    for c in rng.permutation(len(some))[:per_kind]:
      c = some[c]
      p, xy = np.array(c["at"]), xy_of[c["lower_way"]]
      k = int(np.argmin(np.hypot(*(xy[:-1] + xy[1:] - 2 * p).T)))  # the lower way's segment at the crossing
      d = (xy[k + 1] - xy[k]) / max(np.hypot(*(xy[k + 1] - xy[k])), 1e-9)
      route = np.array([to_lat_lon(*(p + t * d))[::-1] for t in (-60.0, 0.0, 30.0)])
      lat, lon = to_lat_lon(*p)
      image, _ = renderer.render(lon, lat, float(np.degrees(np.arctan2(d[0], d[1])) % 360), route)
      sheet.append((f"{c['upper']} over {c['lower']} {c['dz']} m", image))
    contact_sheet(sheet).save(Path(out) / f"crossings_{kind}.png")


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
  ap.add_argument("pbf")
  ap.add_argument("--examples", help="write the crossings (json) here")
  ap.add_argument("--frames", nargs=2, metavar=("TILES", "OUT"), help="render examples of each kind")
  args = ap.parse_args()
  ele, ways = read(args.pbf)
  found = crossings(ele, ways)
  kinds = Counter(c["kind"] for c in found)
  print("crossings without a shared node:", len(found), dict(kinds))
  by_class = Counter((c["kind"], c["upper"], c["lower"]) for c in found if c["kind"] in ("untagged", "inverted"))
  print("untagged / inverted, by (upper, lower) class:", by_class.most_common(12))
  if args.examples:
    with open(args.examples, "w") as f:
      json.dump(found, f)
  if args.frames:
    frames(*args.frames, found, ways)


if __name__ == "__main__":
  main()
