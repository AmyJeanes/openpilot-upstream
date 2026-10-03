#!/usr/bin/env python3
"""Converts GTA V's vehicle path nodes (ynddump's JSON lines) into an OpenStreetMap file a standard router can use.

Game coordinates (metres, x east, y north) map to degrees about (0, 0) on a sphere, see gta5_map.to_lat_lon. Each node
link becomes a way with OSM lane tags; GTA flags with no OSM equivalent keep a gta: prefix.
"""
import argparse
import json
import math
from collections import defaultdict

import osmium

from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon

U_TURN = 135.0  # deg: openpilot's driving model can't turn back on itself, so the map forbids it
GAP = 40.0  # m: links this short between two roads, as through a median or a junction, are part of a turn
GAP_LINKS = 4
PED_SPECIALS = {14, 18}  # 10 marks a vehicle node at a pedestrian crossing
PED_CROSSING, TRAFFIC_LIGHT, STOP_JUNCTION = 10, 15, 16
# posted limits (mph) by road class and the nodes' speed class (slow, normal, fast, faster), see README.md
LIMITS = {
  'motorway': (25, 40, 55, 65),
  'trunk': (25, 45, 50, 55),
  'primary': (25, 40, 45, 50),
  'residential': (20, 30, 50, 50),
  'service': (15, 20, 35, 35),
  'track': (15, 25, 35, 35),
}
CITY_Y = 1300.0  # m: Los Santos is south of this
BLIP = 60.0  # m: a change of limit shorter than this along a road is dropped
FOLLOWS = 60.0  # deg: a link follows on from another if it turns less than this
RAMP = 1500.0  # m: a ramp or connector is no longer than this
LANES_APART = 20.0  # m: GTA draws a freeway's lanes as links side by side, joined by lane changes
OVERPASS_DZ = 4.0  # m: roads crossing with this much height between them are on different levels


def node_ok(n):
  return (n['f'][1] >> 3) not in PED_SPECIALS and not n['f'][2] & 32  # water


def load(path):
  nodes, links, streets = {}, defaultdict(list), {}
  for line in open(path):
    d = json.loads(line)
    if d['t'] == 'n':
      nodes[(d['a'], d['i'])] = d
    elif d['t'] == 'l':
      links[(d['a'], d['i'])].append(d)
    else:
      streets[d['h']] = d['name']
  return nodes, links, streets


def edges(nodes, links):
  """Undirected edges {(a, b): (lanes a->b, lanes b->a, link flags)} between vehicle nodes."""
  out = {}
  for ka, ls in links.items():
    if ka not in nodes or not node_ok(nodes[ka]):
      continue
    for l in ls:
      kb = (l['ta'], l['ti'])
      if kb not in nodes or not node_ok(nodes[kb]):
        continue
      fwd, back = (l['f'][2] >> 5) & 7, (l['f'][2] >> 2) & 7
      if fwd + back == 0:
        continue
      key = (ka, kb) if ka < kb else (kb, ka)
      if key not in out:
        out[key] = (fwd, back, l['f']) if ka < kb else (back, fwd, l['f'])
  return out


def direction(nodes, a, b):
  dx, dy = nodes[b]['x'] - nodes[a]['x'], nodes[b]['y'] - nodes[a]['y']
  d = math.hypot(dx, dy) or 1.0
  return dx / d, dy / d


def crossovers(nodes, es):
  """Two-way links joining the two carriageways of a divided road: a router would U-turn through them."""
  oneway_dirs = defaultdict(list)  # node -> directions of travel of its one-way edges
  for (a, b), (fwd, back, _) in es.items():
    if fwd and not back:
      oneway_dirs[a].append(direction(nodes, a, b))
      oneway_dirs[b].append(direction(nodes, a, b))
    elif back and not fwd:
      oneway_dirs[a].append(direction(nodes, b, a))
      oneway_dirs[b].append(direction(nodes, b, a))
  out = set()
  for (a, b), (fwd, back, _) in es.items():
    if not (fwd and back) or nodes[a]['f'][2] & 4 or nodes[b]['f'][2] & 4:
      continue
    if any(da[0] * db[0] + da[1] * db[1] < -0.8 for da in oneway_dirs[a] for db in oneway_dirs[b]):
      out.add((a, b))
  return out


def highway(nodes, a, b, fwd, back):
  fa, fb = nodes[a]['f'], nodes[b]['f']
  if (fa[0] | fb[0]) & 8:
    return 'track'
  if (fa[2] | fb[2]) & 128 or (fa[2] | fb[2]) & 1:  # switched off for traffic or no GPS: car parks, alleys
    return 'service'
  if fa[2] & fb[2] & 64:
    return 'motorway' if not (fwd and back) else 'trunk'
  if max(fwd, back) >= 2:
    return 'primary'
  return 'residential'


def speed_class(nodes, a, b):
  """The link's speed class, and whether it's out in the country."""
  return max(nodes[a].get('sp', 1), nodes[b].get('sp', 1)), min(nodes[a]['y'], nodes[b]['y']) > CITY_Y


def speed_limit(nodes, a, b, cls, fwd, back):
  sp, country = speed_class(nodes, a, b)
  if sp == 1 and cls == 'motorway' and max(fwd, back) >= 2:
    return 50  # the slower stretches of freeway in the city; one lane is a ramp
  if sp == 1 and cls == 'residential' and country:
    return 40
  return LIMITS[cls][sp]


def street_limits(ways, length):
  """Each street gets its most common limit (by length) for each speed class, so a limit doesn't change at every
  junction where GTA splits a road into one-way pairs; a link between two streets gets the lower of theirs.
  `ways` is [(street at a, street at b, group, limit)]; links off the named streets keep theirs."""
  totals = defaultdict(lambda: defaultdict(float))
  for i, (sa, sb, group, limit) in enumerate(ways):
    if sa and sa == sb:
      totals[(sa, group)][limit] += length[i]
  street = {k: max(v.items(), key=lambda kv: kv[1])[0] for k, v in totals.items()}
  out = []
  for sa, sb, group, limit in ways:
    named = [street[(s, group)] for s in {sa, sb} if (s, group) in street]
    out.append(min(named) if sa and sb and named else limit)
  return out


def blip_limits(nodes, ways, limits, length):
  """Drops a change of limit shorter than BLIP along the way through a junction, where GTA's links switch between the
  crossing streets and lane counts: such a stretch takes the limit of the roads on both sides of it, if they agree.
  `ways` is [(a, b, two_way)]; returns new limits."""
  def heading(p, q):
    return math.degrees(math.atan2(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y']))
  moves = []  # directed (way index, from, to, heading)
  for i, (a, b, two_way) in enumerate(ways):
    moves.append((i, a, b, heading(a, b)))
    if two_way:
      moves.append((i, b, a, heading(b, a)))
  leave, arrive = defaultdict(list), defaultdict(list)
  for m in moves:
    leave[m[1]].append(m)
    arrive[m[2]].append(m)

  def on(m, n):  # n follows on from m without turning
    return n[0] != m[0] and abs((n[3] - m[3] + 180) % 360 - 180) < FOLLOWS

  def ends(m, forward, limit, out):
    """The limits next to the stretch of `limit` from m on (or back), or None if the stretch is BLIP or longer."""
    found, seen, stack = set(), {m[0]}, [(m, 0.0)]
    while stack:
      n, dist = stack.pop()
      for k in (leave[n[2]] if forward else arrive[n[1]]):
        if k[0] in seen or not (on(n, k) if forward else on(k, n)):
          continue
        if out[k[0]] != limit:
          found.add(out[k[0]])
        elif dist + length[k[0]] >= BLIP:
          return None
        else:
          seen.add(k[0])
          stack.append((k, dist + length[k[0]]))
    return found

  out = list(limits)
  short = sorted((m for m in moves if length[m[0]] < BLIP), key=lambda m: length[m[0]])
  for _ in range(5):  # until settled: one change can leave the next link a blip
    changed = 0
    for m in short:
      before, after = ends(m, False, out[m[0]], out), ends(m, True, out[m[0]], out)
      if before and after and len(before & after) == 1:
        out[m[0]] = (before & after).pop()
        changed += 1
    if not changed:
      break
  return out


def overpasses(xyz, segs):
  """Where two links cross in x, y more than OVERPASS_DZ apart in height: [(i, j, x, y, z_i, z_j)] by link index.
  `xyz` is an (n, 3) array, `segs` (m, 2) node indices."""
  import numpy as np
  cell = 64.0
  p0, p1 = xyz[segs[:, 0]], xyz[segs[:, 1]]
  lo, hi = np.floor(np.minimum(p0, p1)[:, :2] / cell).astype(int), np.floor(np.maximum(p0, p1)[:, :2] / cell).astype(int)
  grid = defaultdict(list)
  for k in range(len(segs)):
    for cx in range(lo[k, 0], hi[k, 0] + 1):
      for cy in range(lo[k, 1], hi[k, 1] + 1):
        grid[(cx, cy)].append(k)
  found = set()
  for ks in grid.values():
    if len(ks) < 2:
      continue
    ks = np.array(ks)
    i, j = np.triu_indices(len(ks), 1)
    i, j = ks[i], ks[j]
    shared = (segs[i, :, None] == segs[j, None, :]).any(axis=(1, 2))
    i, j = i[~shared], j[~shared]
    a, b, c, d = p0[i], p1[i], p0[j], p1[j]
    r, s = b - a, d - c
    den = r[:, 0] * s[:, 1] - r[:, 1] * s[:, 0]
    ok = np.abs(den) > 1e-9
    den = np.where(ok, den, 1.0)
    qp = c - a
    t = (qp[:, 0] * s[:, 1] - qp[:, 1] * s[:, 0]) / den
    u = (qp[:, 0] * r[:, 1] - qp[:, 1] * r[:, 0]) / den
    zi, zj = a[:, 2] + t * r[:, 2], c[:, 2] + u * s[:, 2]
    hit = ok & (t > 0) & (t < 1) & (u > 0) & (u < 1) & (np.abs(zi - zj) > OVERPASS_DZ)
    for k in np.nonzero(hit)[0]:
      x, y = a[k, 0] + t[k] * r[k, 0], a[k, 1] + t[k] * r[k, 1]
      found.add((int(min(i[k], j[k])), int(max(i[k], j[k])), round(float(x), 1), round(float(y), 1),
                 round(float(zi[k] if i[k] < j[k] else zj[k]), 1), round(float(zj[k] if i[k] < j[k] else zi[k]), 1)))
  return sorted(found)


def ramps(nodes, ways, length, streets):
  """Freeway ramps and connectors, which GTA doesn't mark: one-way links that branch off a freeway (or a highway) and
  reach an ordinary road, or join a freeway other than the one they left, within RAMP. Links that branch off and come
  back into the road they left are lanes of it. `ways` is [(a, b, two_way, class, name)], `streets` the street names
  by street hash; returns {way index: (highway tag, destination)}."""
  def heading(p, q):
    return math.degrees(math.atan2(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y']))
  def turn(h0, h1):
    return abs((h1 - h0 + 180) % 360 - 180)
  def main(i):
    return ways[i][3] in ('motorway', 'trunk')
  moves = []  # directed (way index, from, to, heading)
  for i, (a, b, two_way, _, _) in enumerate(ways):
    moves.append((i, a, b, heading(a, b)))
    if two_way:
      moves.append((i, b, a, heading(b, a)))
  leave, arrive = defaultdict(list), defaultdict(list)
  for m in moves:
    leave[m[1]].append(m)
    arrive[m[2]].append(m)
  def on(m, forward):  # the moves on from m (or back)
    return leave[m[2]] if forward else arrive[m[1]]
  def end(m, forward):
    return m[2] if forward else m[1]

  def road(m, forward):
    """The ways along the road from m, going straight on, for 2 RAMP."""
    out, total = set(), 0.0
    while m and total < 2 * RAMP:
      out.add(m[0])
      total += length[m[0]]
      m = min(((turn(m[3], n[3]), n) for n in on(m, forward) if main(n[0]) and n[0] != m[0] and n[0] not in out), default=(180, None))
      m = m[1] if m[0] < 60 else None
    return out

  def beside(i, along):
    """Whether way i is part of the road `along` (ways), a lane alongside it or a lane change."""
    def near(k):
      p = nodes[k]
      for j in along:
        c, d = nodes[ways[j][0]], nodes[ways[j][1]]
        dx, dy = d['x'] - c['x'], d['y'] - c['y']
        t = min(max(((p['x'] - c['x']) * dx + (p['y'] - c['y']) * dy) / max(dx * dx + dy * dy, 1e-9), 0.0), 1.0)
        if math.hypot(c['x'] + t * dx - p['x'], c['y'] + t * dy - p['y']) < LANES_APART and \
           abs(c['z'] + t * (d['z'] - c['z']) - p['z']) < OVERPASS_DZ:
          return True
      return False
    return i in along or (near(ways[i][0]) and near(ways[i][1]))

  def walk(m0, forward):
    """Follows one-way links from m0: (its links, the ordinary roads reached, the main roads joined), or None if longer
    than RAMP."""
    chain, roads, joins, stack, total = {m0[0]}, set(), set(), [m0], length[m0[0]]
    while stack:
      m = stack.pop()
      k = end(m, forward)
      if not nodes[k]['f'][2] & 64:  # off the highway
        roads.update(streets.get(nodes[q]['st'], '') for n in leave[k] + arrive[k] if n[0] not in chain
                     for q in n[1:3] if not nodes[q]['f'][2] & 64)
        continue
      if any(n[0] not in chain and main(n[0]) for n in (arrive[k] if forward else leave[k])):
        joins.update(n[0] for n in on(m, forward) if n[0] not in chain)
        continue
      for n in on(m, forward):
        if n[0] in chain:
          continue
        if ways[n[0]][2]:  # a two-way road
          (joins if main(n[0]) else roads).add(n[0] if main(n[0]) else ways[n[0]][4] or '')
          continue
        chain.add(n[0])
        total += length[n[0]]
        if total > RAMP:
          return None
        stack.append(n)
    return chain, roads, joins

  out = {}
  for k in list(leave):
    for forward in (True, False):
      ins, outs = (arrive[k], leave[k]) if forward else (leave[k], arrive[k])
      for m_in in (m for m in ins if main(m[0])):
        ahead = [m for m in outs if m[0] != m_in[0] and turn(m_in[3], m[3]) < 90]
        straight = min((m for m in ahead if main(m[0])), key=lambda m: turn(m_in[3], m[3]), default=None)
        branches = [m for m in ahead if m is not straight and not ways[m[0]][2]]
        if not straight or not branches:
          continue
        along = road(straight, forward)
        for m in branches:
          w = walk(m, forward)
          if not w:
            continue
          chain, roads, joins = w
          other = {i for i in joins if not beside(i, along)}
          if not roads and not other:
            continue  # a lane of the road it left
          left = {ways[i][4] for i in along}
          dest = sorted(r for r in roads if r and r not in left) or sorted({ways[i][4] for i in other if ways[i][4]} - left)
          for i in chain:
            if i not in out or forward:
              out[i] = (ways[m_in[0]][3] + '_link', '; '.join(dest[:2]) if forward else '')
  return out


def write_sidecar(path, nodes, used, ways):
  """The roads as arrays for the bridge's map matching (game metres): nodes x, y, z and the links between them.
  `ways` is [(way id, a, b, fwd lanes, back lanes, class, limit, name)]."""
  import numpy as np
  index = {k: i for i, k in enumerate(used)}
  xyz = np.array([(nodes[k]['x'], nodes[k]['y'], nodes[k]['z']) for k in used], dtype=np.float32)
  segs = np.array([(index[a], index[b]) for _, a, b, *_ in ways], dtype=np.int32)
  d = xyz[segs[:, 1]] - xyz[segs[:, 0]]
  classes = list(LIMITS) + ['motorway_link', 'trunk_link']
  names = sorted({w[7] for w in ways if w[7]})
  name_index = {n: i for i, n in enumerate(names)}
  cross = overpasses(xyz, segs)
  np.savez_compressed(
    path, x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], node_id=np.array([node_id(k) for k in used], dtype=np.int64),
    way_id=np.array([w[0] for w in ways], dtype=np.int32), a=segs[:, 0], b=segs[:, 1],
    lanes_fwd=np.array([w[3] for w in ways], dtype=np.uint8), lanes_back=np.array([w[4] for w in ways], dtype=np.uint8),
    heading=np.degrees(np.arctan2(d[:, 0], d[:, 1])).astype(np.float32), length=np.hypot(d[:, 0], d[:, 1]).astype(np.float32),
    road_class=np.array([classes.index(w[5]) for w in ways], dtype=np.uint8), classes=np.array(classes),
    maxspeed_mph=np.array([w[6] for w in ways], dtype=np.uint8),
    name=np.array([name_index.get(w[7], -1) for w in ways], dtype=np.int16), names=np.array(names),
    overpass_links=np.array([c[:2] for c in cross], dtype=np.int32).reshape(-1, 2),
    overpass_xy=np.array([c[2:4] for c in cross], dtype=np.float32).reshape(-1, 2),
    overpass_z=np.array([c[4:] for c in cross], dtype=np.float32).reshape(-1, 2))
  return len(cross)


def node_tags(n):
  f0, f1, f2, f4 = n['f'][0], n['f'][1], n['f'][2], n['f'][4]
  tags = {}
  special = f1 >> 3
  if special == TRAFFIC_LIGHT:
    tags['highway'] = 'traffic_signals'
  elif special == STOP_JUNCTION:
    tags['highway'] = 'stop'
  elif special == PED_CROSSING:
    tags['highway'] = 'crossing'
  for bit, flags, tag in ((4, f2, 'gta:junction'), (128, f0, 'gta:no_left'), (64, f0, 'gta:no_right'),
                          (1, f1, 'gta:slip_lane'), (2, f1, 'gta:keep_left'), (4, f1, 'gta:keep_right'),
                          (128, f4, 'gta:left_turn_only')):
    if flags & bit:
      tags[tag] = 'yes'
  tags['gta:node'] = f"{n['a']}:{n['i']}"
  return tags


def no_u_turns(nodes, ways):
  """OSM no_u_turn restrictions for every move at a node that turns back by more than U_TURN: GTA's nodes allow them.
  `ways` is [(way id, a, b, two_way)] with a -> b the way's direction."""
  arrive, leave = defaultdict(list), defaultdict(list)  # node -> [(way id, heading)]
  for wid, a, b, two_way in ways:
    h = math.degrees(math.atan2(nodes[b]['x'] - nodes[a]['x'], nodes[b]['y'] - nodes[a]['y']))
    arrive[b].append((wid, h))
    leave[a].append((wid, h))
    if two_way:
      arrive[a].append((wid, h + 180))
      leave[b].append((wid, h + 180))
  def turn(h0, h1):
    return (h1 - h0 + 180) % 360 - 180

  out = []
  for n, ins in arrive.items():
    for wi, hi in ins:
      for wo, ho in leave[n]:
        if abs(turn(hi, ho)) > U_TURN:
          out.append((wi, [('n', n)], wo))
  # and through short links, as through a median gap or a junction: in, along them, out, turning back
  length = {wid: math.hypot(nodes[b]['x'] - nodes[a]['x'], nodes[b]['y'] - nodes[a]['y']) for wid, a, b, _ in ways}
  short_from = defaultdict(list)  # node -> [(way id, far node, heading)] for short links leaving it
  for wid, a, b, two_way in ways:
    if length[wid] <= GAP:
      h = math.degrees(math.atan2(nodes[b]['x'] - nodes[a]['x'], nodes[b]['y'] - nodes[a]['y']))
      short_from[a].append((wid, b, h))
      if two_way:
        short_from[b].append((wid, a, h + 180))
  for n, ins in arrive.items():
    for wi, hi in ins:
      stack = [(n, [], 0.0, hi, 0.0)]  # summing the turns, as a loop can come round past 180 deg
      while stack:
        p, via, dist, h, turned = stack.pop()
        for wv, q, hv in short_from[p]:
          if wv == wi or wv in via or dist + length[wv] > GAP:
            continue
          chain, turned_v = via + [wv], turned + turn(h, hv)
          for wo, ho in leave[q]:
            if wo != wi and wo not in chain and abs(turned_v + turn(hv, ho)) > U_TURN:
              out.append((wi, [('w', v) for v in chain], wo))
          if len(chain) < GAP_LINKS:
            stack.append((q, chain, dist + length[wv], hv, turned_v))
  return out


def node_id(k):
  return k[0] * 65536 + k[1] + 1


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('dump', help="ynddump's paths.jsonl")
  p.add_argument('out', help='.osm or .osm.pbf')
  p.add_argument('--sidecar', help="the roads as arrays for the bridge's map matching (.npz)")
  args = p.parse_args()

  nodes, links, streets = load(args.dump)
  es = edges(nodes, links)
  cross = crossovers(nodes, es)
  es = {k: v for k, v in es.items() if k not in cross}
  used = sorted({k for e in es for k in e})
  print(f"{len(used)} nodes, {len(es)} ways ({len(cross)} carriageway crossovers dropped)")

  w = osmium.SimpleWriter(args.out, overwrite=True)
  for k in used:
    n = nodes[k]
    lat, lon = to_lat_lon(n['x'], n['y'])
    w.add_node(osmium.osm.mutable.Node(id=node_id(k), version=1, location=(lon, lat), tags={**node_tags(n), 'ele': f"{n['z']:.1f}"}))
  info = []  # (way id, a, b, fwd, back, class, limit, name, link flags)
  for i, ((a, b), (fwd, back, lf)) in enumerate(sorted(es.items())):
    if not fwd:  # one-way the other way: draw it in its direction of travel
      a, b, fwd, back = b, a, back, fwd
    cls = highway(nodes, a, b, fwd, back)
    st = nodes[a]['st'] if nodes[a]['st'] == nodes[b]['st'] else 0
    info.append([i + 1, a, b, fwd, back, cls, speed_limit(nodes, a, b, cls, fwd, back), streets.get(st), lf])
  length = [math.hypot(nodes[b]['x'] - nodes[a]['x'], nodes[b]['y'] - nodes[a]['y']) for _, a, b, *_ in info]
  groups = [(streets.get(nodes[a]['st']), streets.get(nodes[b]['st']), speed_class(nodes, a, b), limit)
            for _, a, b, _, _, _, limit, _, _ in info]
  links = [(a, b, bool(back)) for _, a, b, _, back, *_ in info]
  limits = blip_limits(nodes, links, street_limits(groups, length), length)
  for row, limit in zip(info, limits, strict=True):
    row[6] = limit
  destination = {}
  for i, (tag, dest) in ramps(nodes, [(a, b, bool(back), cls, name) for _, a, b, _, back, cls, _, name, _ in info], length,
                              streets).items():
    info[i][5] = tag
    if dest:
      destination[info[i][0]] = dest
  ways = []
  for wid, a, b, fwd, back, cls, limit, name, lf in info:
    ways.append((wid, a, b, bool(back)))
    tags = {'highway': cls, 'lanes': str(fwd + back), 'maxspeed': f'{limit} mph'}
    if back:
      tags['lanes:forward'], tags['lanes:backward'] = str(fwd), str(back)
    else:
      tags['oneway'] = 'yes'
    if name and not cls.endswith("_link"):  # a ramp named for its freeway reads as staying on it
      tags['name'] = name
    if wid in destination:
      tags['destination'] = destination[wid]
    if lf[1] & 2:
      tags['gta:narrow'] = 'yes'
    if lf[2] & 1:
      tags['gta:no_nav'] = 'yes'
    w.add_way(osmium.osm.mutable.Way(id=wid, version=1, nodes=[node_id(a), node_id(b)], tags=tags))
  restrictions = no_u_turns(nodes, ways)
  for i, (wi, via, wo) in enumerate(restrictions):
    via = [(kind, node_id(ref) if kind == 'n' else ref, 'via') for kind, ref in via]
    w.add_relation(osmium.osm.mutable.Relation(id=i + 1, version=1, tags={'type': 'restriction', 'restriction': 'no_u_turn'},
                                               members=[('w', wi, 'from'), *via, ('w', wo, 'to')]))
  w.close()
  print(f"{len(restrictions)} U-turns forbidden")
  if args.sidecar:
    n = write_sidecar(args.sidecar, nodes, used, [r[:8] for r in info])
    print(f"{n} links crossing on different levels -> {args.sidecar}")


if __name__ == '__main__':
  main()
