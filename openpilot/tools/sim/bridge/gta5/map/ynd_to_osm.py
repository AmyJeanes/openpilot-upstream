#!/usr/bin/env python3
"""Converts GTA V's vehicle path nodes (ynddump's JSON lines) into an OpenStreetMap file a standard router can use.

Game coordinates (metres, x east, y north) map to degrees about (0, 0) on a sphere, see gta5_map.to_lat_lon. Each node
link becomes a way with OSM lane tags; GTA flags with no OSM equivalent keep a gta: prefix.
"""
import argparse
import json
import math
from collections import Counter, defaultdict

import osmium

from openpilot.tools.sim.bridge.gta5.map import paint_survey
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon
from openpilot.tools.sim.bridge.gta5.map.paths import heading as game_heading, junction_scores, roads_cross, toward_junction, \
  wrap

U_TURN = 135.0  # deg: openpilot's driving model can't turn back on itself, so the map forbids it
TURN_FLAG = 45.0  # deg: a move turning more than this is a left or right turn, for GTA's no left / no right flags
JUNCTION_SPAN = 20.0  # m across a junction from its node, to the ways out of it
MAX_VIA = 5  # ways in a turn restriction
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
STRAIGHT = 30.0  # deg: a link goes straight on from another if it turns less than this
NAME_RUN = 100.0  # m: unnamed links up to this long between two stretches of a street are that street
RAMP = 1500.0  # m: a ramp or connector is no longer than this
LANES_APART = 20.0  # m: GTA draws a freeway's lanes as links side by side, joined by lane changes
OVERPASS_DZ = 4.0  # m: roads crossing with this much height between them are on different levels
MAX_LAYER = 5  # OSM's highest layer
BRIDGE_REACH = 15.0  # m round where roads cross over each other: links this near at either's height are part of it
MINIMAP_ROAD = 180  # grey of the ordinary roads in GTA's minimap art (99 is dirt tracks and alleys)
MINIMAP_CELL = 2.0  # m
DRAWN = 0.5  # of a minor link's length away from major roads, on a road the minimap draws: it's a road
MAJOR_NEAR = 15.0  # m: the minimap's road this near a major link is that road
MIN_AWAY = 10.0  # m of a link away from major roads to judge it by; shorter, it's a road if it joins one
SWITCHED_OFF, NO_GPS, OFFROAD = 128, 1, 8  # GTA's node flags (f2, f2, f0)
APPROACH = 30.0  # m before a junction that its lanes' turn arrows are marked (GTA's stop lines are 12-24 m before it)
APPROACH_HEADING = 15.0  # m: a junction's ways out turn from the road's heading over this far before it
RESTRICTION_REACH = 60.0  # m back from an approach that a restriction on it can start (from a stop line or turn flag)
APPROACH_BEND = 20.0  # deg: where the way into a junction turns more than this from that, which way is through is moot
# m: lanes as painted, measured in the game (CodeWalker's 5.5 / 4.0 m are the AI's lanes, not the paint: narrow lanes are
# painted 4.2-4.5 m, freeway lanes 5.9-6.5 m), and the painted median per step of a two-way link's offset (5.2-5.5 m
# at 6 steps on narrow and normal links alike)
PAINTED_LANE, PAINTED_NARROW, PAINTED_FREEWAY = 5.5, 4.4, 6.1
MEDIAN_STEP = 0.9
MEDIAN_LANE_MIN = 15.0  # m: a median runs in to a junction at least this far to be painted as a turn lane
FREEWAY = 64  # node flags 2: a freeway's
MARKED = {'trunk', 'primary', 'residential'}  # classes of GTA's streets with painted centre lines
BAY_MIN = 2.5  # m: a two-way road's median narrower than this has no room for a turn bay


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
  out, last = {}, {}
  for ka, ls in links.items():
    for l in ls:  # where GTA lists a link twice, its last record, as paths.py reads it
      last[(ka, (l['ta'], l['ti']))] = l
  for (ka, kb), l in last.items():
    if ka not in nodes or not node_ok(nodes[ka]) or kb not in nodes or not node_ok(nodes[kb]):
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


def levels(nodes, links) -> dict[int, int]:
  """OSM layers from GTA's heights: a link crossing over another without a node in common (overpasses) is a layer
  above it, a level up for each road it's stacked over, as a freeway interchange's; the rest stay on the ground. `links`
  is [(a, b)]; returns {link index: layer} for those above ground."""
  import numpy as np
  used = sorted({k for link in links for k in link})
  index = {k: i for i, k in enumerate(used)}
  xyz = np.array([(nodes[k]['x'], nodes[k]['y'], nodes[k]['z']) for k in used], dtype=np.float64).reshape(-1, 3)
  segs = np.array([(index[a], index[b]) for a, b in links], dtype=np.int64).reshape(-1, 2)
  cross = overpasses(xyz, segs)
  p0, p1 = xyz[segs[:, 0]], xyz[segs[:, 1]]

  def deck(x, y, z):  # the links at a crossing's upper height near it: the bridge's other lanes, as GTA draws them
    a, d = p0[:, :2], (p1 - p0)[:, :2]
    t = np.clip(np.einsum('ij,ij->i', np.array([x, y]) - a, d) / np.maximum(np.einsum('ij,ij->i', d, d), 1e-9), 0, 1)
    near = np.hypot(*(a + d * t[:, None] - [x, y]).T) < BRIDGE_REACH
    return np.flatnonzero(near & (np.abs(p0[:, 2] + (p1 - p0)[:, 2] * t - z) < OVERPASS_DZ / 2))
  above = []  # (upper, lower)
  for i, j, x, y, zi, zj in cross:
    up, low, z_up, z_low = (i, j, zi, zj) if zi > zj else (j, i, zj, zi)
    lows = deck(x, y, z_low)
    above += [(u, lw) for u in {up, *deck(x, y, z_up).tolist()} for lw in {low, *lows.tolist()}]
  layer: dict[int, int] = {}
  for _ in range(MAX_LAYER):
    changed = False
    for up, low in above:
      if layer.get(up, 0) < min(layer.get(low, 0) + 1, MAX_LAYER):
        layer[up] = min(layer.get(low, 0) + 1, MAX_LAYER)
        changed = True
    if not changed:
      break
  return layer


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


def grow(mask, diagonal=False):
  """The cells set in `mask` and their neighbours, also the diagonal ones with `diagonal`."""
  out = mask.copy()
  out[1:] |= mask[:-1]
  out[:-1] |= mask[1:]
  out[:, 1:] |= mask[:, :-1]
  out[:, :-1] |= mask[:, 1:]
  if diagonal:
    out[1:, 1:] |= mask[:-1, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
  return out


def minimap_roads(path):
  """The ordinary roads GTA's minimap draws (ynddump's minimap.jsonl: triangles in game metres) as a raster of
  MINIMAP_CELL cells: (cells within about a cell of a road, x0, y1), the cell at row (y1 - y) // MINIMAP_CELL and
  column (x - x0) // MINIMAP_CELL."""
  import numpy as np
  with open(path) as f:
    tris = np.concatenate([np.array(d['xy'], dtype=np.float64).reshape(-1, 3, 2) for d in map(json.loads, f)
                           if d['grey'] == MINIMAP_ROAD])
  c = MINIMAP_CELL
  x0, y1 = tris[..., 0].min() - 2 * c, tris[..., 1].max() + 2 * c
  mask = np.zeros((int((y1 - tris[..., 1].min()) / c) + 3, int((tris[..., 0].max() - x0) / c) + 3), dtype=bool)
  for t in tris:
    edge = np.roll(t, -1, axis=0) - t
    area, length = edge[0, 0] * edge[1, 1] - edge[0, 1] * edge[1, 0], np.hypot(edge[:, 0], edge[:, 1])
    if abs(area) < 1e-6:
      continue
    c0, c1 = int((t[:, 0].min() - x0) / c), int((t[:, 0].max() - x0) / c)
    r0, r1 = int((y1 - t[:, 1].max()) / c), int((y1 - t[:, 1].min()) / c)
    x = (x0 + (np.arange(c0, c1 + 1) + 0.5) * c)[None, :, None]
    y = (y1 - (np.arange(r0, r1 + 1) + 0.5) * c)[:, None, None]
    inside = (edge[:, 0] * (y - t[:, 1]) - edge[:, 1] * (x - t[:, 0])) / length * np.sign(area)  # m in from each edge
    mask[r0:r1 + 1, c0:c1 + 1] |= (inside >= -c / 2).all(axis=2)  # cells the triangle touches
  return grow(mask), x0, y1


def minimap_classes(nodes, rows, minimap):
  """The minor links GTA lets its GPS use (switched off for traffic, or off-road) that its minimap draws as ordinary
  roads: the port's streets, quarry and oil-field roads, country roads. Most of each one's length away from major roads
  (where the drawn road is theirs) lies on a drawn road, or, close by major roads all along, most of it does and it joins
  such a link; and links up to GAP long between two of them. Links without GPS (runways, the golf course, the prison,
  Fort Zancudo) and those drawn as dirt tracks stay minor. `rows` is [(a, b, class)], `minimap` minimap_roads'; returns
  the indices of the rows that are roads."""
  import numpy as np
  mask, x0, y1 = minimap
  c = MINIMAP_CELL

  def samples(sel, step):  # points about every step along the rows, which row each is on, the rows' lengths
    p0 = np.array([(nodes[rows[r][0]]['x'], nodes[rows[r][0]]['y']) for r in sel]).reshape(-1, 2)
    p1 = np.array([(nodes[rows[r][1]]['x'], nodes[rows[r][1]]['y']) for r in sel]).reshape(-1, 2)
    length = np.hypot(*(p1 - p0).T)
    n = np.maximum(1, np.ceil(length / step)).astype(int)
    i = np.repeat(np.arange(len(sel)), n)
    t = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n) + 0.5) / n[i]
    return p0[i] + (p1[i] - p0[i]) * t[:, None], i, length

  def cells(p):
    return (np.clip(((y1 - p[:, 1]) / c).astype(int), 0, mask.shape[0] - 1),
            np.clip(((p[:, 0] - x0) / c).astype(int), 0, mask.shape[1] - 1))

  near = np.zeros_like(mask)
  near[cells(samples([r for r, (_, _, cls) in enumerate(rows) if cls not in ('service', 'track')], c / 2)[0])] = True
  for k in range(round(MAJOR_NEAR / c)):  # about a disc: straight and diagonal steps in turn
    near = grow(near, diagonal=k % 2 == 1)
  minor = [r for r, (a, b, cls) in enumerate(rows) if cls in ('service', 'track') and
           not (nodes[a]['f'][2] | nodes[b]['f'][2]) & NO_GPS]
  pts, i, length = samples(minor, c)
  drawn, away = mask[cells(pts)], ~near[cells(pts)]
  n = np.bincount(i, minlength=len(minor))
  n_away, n_drawn, n_drawn_away = (np.bincount(i, v, minlength=len(minor)) for v in (away, drawn, drawn & away))
  roads, by_major = set(), set()
  for k, r in enumerate(minor):
    if n_away[k] / n[k] * length[k] >= MIN_AWAY:
      if n_drawn_away[k] > DRAWN * n_away[k]:
        roads.add(r)
    elif n_drawn[k] > DRAWN * n[k]:
      by_major.add(r)
  at = defaultdict(set)  # node -> its rows that are roads
  for r in roads:
    at[rows[r][0]].add(r)
    at[rows[r][1]].add(r)
  gaps = [r for k, r in enumerate(minor) if length[k] <= GAP]
  joined = True
  while joined:  # where a road leaves a major one, as along a chain of short links, and short gaps between roads
    joined = False
    for r in [*by_major, *gaps]:
      ends = at[rows[r][0]], at[rows[r][1]]
      if r not in roads and ((ends[0] or ends[1]) if r in by_major else (ends[0] and ends[1])):
        roads.add(r)
        at[rows[r][0]].add(r)
        at[rows[r][1]].add(r)
        joined = True
  return roads


def street_runs(nodes, rows, length):
  """Names for the unnamed links on a street's way through a junction: GTA's links between two streets' nodes have
  neither name, and a router charges for every change of name, so it would favour streets with no names at all. A run
  of unnamed links up to NAME_RUN long going straight on (within STRAIGHT) from the same street at both ends takes its
  name, except for:
  - links at a stop line: named on into the junction, Valhalla 3.9 charges about 40 s more for going straight through
    it (Palomino Ave at South Rockford Dr);
  - freeway links: named, freeways lose the name changes that offset the turn and junction costs Valhalla charges on
    our lane-level city streets, and it sends city trips round slower freeway detours (up to 2x as long).
  `rows` is [(a, b, class, name)]; ramps keep no name. Returns {row index: name}."""
  def keeps_none(r):
    a, b, cls, _ = rows[r]
    return cls in ('motorway', 'trunk') or any(nodes[k]['f'][1] >> 3 in (TRAFFIC_LIGHT, STOP_JUNCTION) for k in (a, b))

  at = defaultdict(list)
  for r, (a, b, cls, _) in enumerate(rows):
    if not cls.endswith('_link'):
      at[a].append(r)
      at[b].append(r)

  def far(r, n):
    return rows[r][1] if rows[r][0] == n else rows[r][0]

  def heading(p, q):
    return game_heading(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y'])

  def ahead(r, n):  # the row straight on from row r through its node n
    h = heading(far(r, n), n)
    turn, q = min(((abs(wrap(heading(n, far(q, n)) - h)), q) for q in at[n] if q != r), default=(180.0, None))
    return q if turn < STRAIGHT else None

  out = {}
  for r, (a, b, cls, name) in enumerate(rows):
    if name or cls.endswith('_link') or r in out:
      continue
    run, total, ends = {r}, length[r], []
    for n in (a, b):
      q, m, end = r, n, None
      while total <= NAME_RUN:
        nxt = ahead(q, m)
        if nxt is None or nxt in run:
          break
        if rows[nxt][3]:
          end = rows[nxt][3]
          break
        run.add(nxt)
        total += length[nxt]
        q, m = nxt, far(nxt, m)
      ends.append(end)
    if total <= NAME_RUN and ends[0] and ends[0] == ends[1]:
      out.update({q: ends[0] for q in run if not keeps_none(q)})
  return out


def write_sidecar(path, nodes, used, ways):
  """The roads as arrays for the bridge's map matching (game metres): nodes x, y, z and the links between them.
  `ways` is [(way id, a, b, fwd lanes, back lanes, class, limit, name)]."""
  import numpy as np
  index = {k: i for i, k in enumerate(used)}
  xyz = np.array([(nodes[k]['x'], nodes[k]['y'], nodes[k]['z']) for k in used], dtype=np.float32)
  segs = np.array([(index[a], index[b]) for _, a, b, *_ in ways], dtype=np.int32)
  d = xyz[segs[:, 1]] - xyz[segs[:, 0]]
  classes = list(LIMITS) + ['motorway_link', 'trunk_link', 'unclassified']
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


def node_tags(n, stop='both'):
  """`stop`, for a stop line: 'forward' (its ways are drawn towards its junction), 'both' or 'none' (no junction ahead:
  left out)."""
  f0, f1, f2, f4 = n['f'][0], n['f'][1], n['f'][2], n['f'][4]
  tags = {}
  special = f1 >> 3
  if special in (TRAFFIC_LIGHT, STOP_JUNCTION) and stop == 'none':
    pass
  elif special == TRAFFIC_LIGHT:
    tags['highway'] = 'traffic_signals'
    if stop == 'forward':
      tags['traffic_signals:direction'] = 'forward'
  elif special == STOP_JUNCTION:
    tags['highway'] = 'stop'
    if stop == 'forward':
      tags['direction'] = 'forward'
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


def lane_node(n):
  return bool(n['f'][1] & 1 or n['f'][4] & 128)  # GTA's slip lane and left turn only flags: turn lanes and bays


def turn_lanes(nodes, rows, streets, junction):
  """Turn lanes and bays (one-way links at GTA's slip lane or left turn only nodes) as the road they split off, rather
  than as minor roads: its class (up to primary) and, along to the junction, its name. `rows` is [(a, b, two_way,
  class, name)], junction(node) where roads meet; returns {row index: (class, name)}."""
  rank = {'service': 0, 'residential': 1, 'primary': 2}
  arrive = defaultdict(list)  # node -> [(row, from node)]
  for r, (a, b, two_way, _, _) in enumerate(rows):
    arrive[b].append((r, a))
    if two_way:
      arrive[a].append((r, b))

  def lane(r):
    a, b, two_way = rows[r][:3]
    return not two_way and (lane_node(nodes[a]) or lane_node(nodes[b]))

  def heading(p, q):
    return game_heading(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y'])

  def behind(s, nxt, seen, lanes):  # the row arriving at s that runs on into s -> nxt, and where it comes from
    cands = [(abs(wrap(heading(p, s) - heading(s, nxt))), q, p) for q, p in arrive[s] if q not in seen and lane(q) == lanes]
    best = min(cands, default=None)
    return best[1:] if best and best[0] < FOLLOWS else None

  out = {}
  for r, (a, b, _, cls, name) in enumerate(rows):
    if not lane(r) or cls not in rank:
      continue
    s, nxt, seen = a, b, {r}
    while lane_node(nodes[s]):  # back along the lane to where it splits off the road
      step = behind(s, nxt, seen, True)
      if step is None:
        break
      seen.add(step[0])
      s, nxt = step[1], s
    parent = behind(s, nxt, seen, False) if not junction(s) else None
    if parent is None:
      continue
    road_cls, road_name = rows[parent[0]][3], rows[parent[0]][4]
    q, p = parent
    for _ in range(3):  # links touching junctions have no name: the road's further back
      if road_name:
        break
      step = behind(p, rows[q][1] if rows[q][0] == p else rows[q][0], {q}, False)
      if step is None:
        break
      q, p = step
      road_name = rows[q][4]
    if road_cls not in ('primary', 'residential') or rank[road_cls] <= rank[cls]:
      road_cls = cls
    end = nodes[b]
    if not (junction(b) or lane_node(end) or streets.get(end['st']) == road_name):
      road_name = None  # it turns off the road (a slip road)
    if (road_cls, name or road_name) != (cls, name):
      out[r] = (road_cls, name or road_name)
  return out


def detached_bays(nodes, rows, junction):
  """GTA's left-turn bays laid as links of their own, which OSM maps as lanes of their road (a lane count change and
  turn:lanes from where the bay opens): a one-way, one-lane chain through slip lane or left turn only nodes from the node
  where it splits off a road to the junction that road reaches next, on the road's left all along, over a two-way
  road's median or beside a one-way road's left lane. `rows` is [(a, b, fwd lanes, back lanes, link flags)]; returns
  [(the bay's rows, [(the road's row, along it)], its first node, the junction)]."""
  def heading(p, q):
    return game_heading(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y'])

  def length(p, q):
    return math.hypot(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y'])

  out, into = defaultdict(list), defaultdict(list)  # node -> [(next node, row, along it)]
  for r, (a, b, fwd, back, _) in enumerate(rows):
    if fwd:
      out[a].append((b, r, True))
      into[b].append((a, r, True))
    if back:
      out[b].append((a, r, False))
      into[a].append((b, r, False))

  def lanes(r, along):
    return rows[r][2] if along else rows[r][3]

  def one_lane(r, along):
    return lanes(r, along) == 1 and not lanes(r, not along)

  def right_of(p, q, k):  # m right of the line p -> q
    dx, dy = nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y']
    return ((nodes[k]['x'] - nodes[p]['x']) * dy - (nodes[k]['y'] - nodes[p]['y']) * dx) / max(math.hypot(dx, dy), 1e-9)

  def straight(p, q, n):
    return abs(wrap(heading(q, n) - heading(p, q)))

  found, taken, bays_at = [], set(), set()
  for s in sorted(out):
    for b1, r1, along1 in out[s]:
      if not one_lane(r1, along1):
        continue
      bay, bay_nodes, q, dist = [r1], [b1], b1, length(s, b1)
      while not junction(q) and len(out[q]) == 1 and len(into[q]) == 1 and dist < RAMP and one_lane(*out[q][0][1:]):
        q, r, _ = out[q][0]
        bay.append(r)
        bay_nodes.append(q)
        dist += length(bay_nodes[-2], q)
      j = bay_nodes.pop()
      if not junction(j) or not any(lane_node(nodes[k]) for k in bay_nodes):
        continue
      # the road from s, straight on beside the bay, to the same junction
      ahead = [m for m in out[s] if m[0] != b1 and abs(wrap(heading(s, m[0]) - heading(s, b1))) < STRAIGHT]
      if not ahead:
        continue
      m = min(ahead, key=lambda m: abs(wrap(heading(s, m[0]) - heading(s, b1))))
      road, path, total = [m[1:]], [s, m[0]], length(s, m[0])
      while not junction(path[-1]) and total < dist + GAP:
        q = path[-1]
        nxt = [n for n in out[q] if n[0] != path[-2] and straight(path[-2], q, n[0]) < STRAIGHT]
        if not nxt:
          break
        n = min(nxt, key=lambda n: straight(path[-2], q, n[0]))
        road.append(n[1:])
        path.append(n[0])
        total += length(q, n[0])
      if path[-1] != j or len({bool(lanes(r, not a)) for r, a in road}) != 1:
        continue
      r0, a0 = road[0]
      n, back, lf = lanes(r0, a0), lanes(r0, not a0), rows[r0][4]
      w, offset = layout(lf, back)
      if back:  # the median can change width along the road
        medians = [2 * layout(rows[r][4], back)[1] for r, _ in road]
        if min(medians) < BAY_MIN:
          continue  # no room to lie in
        offset = max(medians) / 2
      lo, hi = (-offset - 1.0, offset + w / 2) if back else (-n * w / 2 - 1.5 * w, -n * w / 2 + 0.5)

      def beside(k):  # the bay node's offset from the road link it lies along
        i = min(range(len(path) - 1), key=lambda i: length(path[i], k) + length(k, path[i + 1]) - length(path[i], path[i + 1]))
        return right_of(path[i], path[i + 1], k)
      # a road's other direction can have a bay of its own
      if taken & set(road) or ({r for r, _ in road} | set(bay)) & bays_at or set(bay) & {r for r, _ in taken} or \
         not all(lo <= beside(k) <= hi for k in bay_nodes):
        continue
      taken.update(road)
      bays_at.update(bay)
      found.append((bay, road, s, j))
  return found


def stop_directions(nodes, rows, toward):
  """Which way each stop line faces: tagged direction=forward, every two-way way at it is drawn towards its junction,
  so this turns ways round. `rows` is [(a, b, two_way)], `toward` {stop line node: nodes on from it towards its
  junction}; returns the stop lines it faces (where two need a way drawn both ways, the second keeps no direction)
  and the rows to turn round."""
  at = defaultdict(list)
  for r, (a, b, two_way) in enumerate(rows):
    if two_way:
      at[a].append(r)
      at[b].append(r)
  drawn, faced = {}, set()
  for i in sorted(toward):
    want = {}
    for r in at[i]:
      a, b, _ = rows[r]
      other = b if a == i else a
      want[r] = (i, other) if other in toward[i] else (other, i)
    if all(drawn.get(r, w) == w for r, w in want.items()):
      drawn.update(want)
      faced.add(i)
  return faced, {r for r, ab in drawn.items() if rows[r][:2] != ab}


def graph(ways):
  """{(node, next node): way id} and the nodes a car can drive on to / come from at each node, from [(way id, a, b,
  two_way)]."""
  way_of, out, into = {}, defaultdict(list), defaultdict(list)
  for wid, a, b, two_way in ways:
    for p, q in ((a, b), (b, a)) if two_way else ((a, b),):
      way_of[(p, q)] = wid
      out[p].append(q)
      into[q].append(p)
  return way_of, out, into


def junction_exits(nodes, junction, h_in, h_road, seen, out, way_of):
  """The ways out of a junction arrived at heading h_in (or h_road, the road's way in, as the link on to the
  junction can jog across lanes: a turn by either), going on across its straight short links, as to a divided road's
  far side: [(turn, the ways across, the way out, its (node, next node))], turns in degrees left positive, U-turns left
  out. `seen` (the nodes on the way in) is added to."""
  def length(p, q):
    return math.hypot(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y'])

  def heading(p, q):
    return game_heading(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y'])

  stack, exits = [(junction, [], 0.0)], []
  while stack:
    n, inner, dist = stack.pop()
    for m in out[n]:
      if m in seen:
        continue
      seen.add(m)
      w = way_of[(n, m)]
      turn = wrap(heading(n, m) - h_in)
      if abs(turn) <= TURN_FLAG and dist + length(n, m) <= JUNCTION_SPAN and set(out[m]) - {n}:
        stack.append((m, inner + [w], dist + length(n, m)))
      elif abs(turn) <= U_TURN:
        by_road = wrap(heading(n, m) - h_road)
        exits.append((turn if abs(turn) > TURN_FLAG or abs(by_road) > U_TURN else by_road, inner, w, (n, m)))
  return exits


def turn_restrictions(nodes, ways, toward, flags):
  """GTA's no left / no right turn flags, and its left turn only lanes, as OSM restrictions at the junction ahead of
  the node: from the way into the junction's node (from the node, along the ways to it, where lanes join on the way),
  through the junction's own nodes, to each way out turning that way by more than TURN_FLAG; and at the node itself.
  `ways` is [(way id, a, b, two_way)], `toward` {node: {next node: nodes on to the junction}}, `flags` {node: (no
  left, no right, a left turn only lane)}; returns [(restriction, from way, via, to way)], via [('n', node)] or
  [('w', way id), ...]; how many were left out for going through more than MAX_VIA ways; and how many approaches were
  left as they are, where GTA's flags forbid every way out."""
  way_of, out, into = graph(ways)

  def heading(p, q):
    return game_heading(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y'])

  def kind_of(turn, no_left, no_right, left_only):
    if turn > TURN_FLAG:
      return 'no_left_turn' if no_left else None
    if turn < -TURN_FLAG:
      return 'no_right_turn' if no_right or left_only else None
    return 'no_straight_on' if left_only else None

  found, skipped, dead_ends = set(), 0, 0
  for i in sorted(flags):
    if not toward.get(i):
      continue
    for p in set(into[i]) - set(toward[i]):  # and the ways off at the node itself, as a slip road from it
      for m in out[i]:
        turn = wrap(heading(i, m) - heading(p, i))
        kind = kind_of(turn, *flags[i])
        if kind and m not in toward[i] and m != p and abs(turn) <= U_TURN:
          found.add((kind, way_of[(p, i)], (('n', i),), way_of[(i, m)]))
    for path in toward[i].values():
      junction, h_in = path[-1], heading(path[0], path[-1])
      # and the road's way in, as the link on to the junction can jog across lanes: a turn by either
      h_road = min((heading(p, i) for p in set(into[i]) - set(toward[i])), default=h_in, key=lambda h: abs(wrap(h - h_in)))
      joined = any(set(into[n]) - {path[k + 1]} != {path[k - 1]} for k, n in enumerate(path[1:-1], 1))
      lead = [way_of[(p, q)] for p, q in zip(path[:-1], path[1:], strict=True)] if joined else [way_of[(path[-2], junction)]]
      exits = junction_exits(nodes, junction, h_in, h_road, set(path), out, way_of)
      kinds = [kind_of(turn, *flags[i]) for turn, *_ in exits]
      if all(kinds):
        dead_ends += 1  # GTA's flags leave no way out: forbid none
        continue
      moves = [(tuple(inner) + (w,), kind) for (_, inner, w, _), kind in zip(exits, kinds, strict=True)]
      for move, kind in moves:
        if kind is None:
          continue
        # forbid it from the first way it doesn't share with an allowed move: Valhalla misses routes past restrictions
        # through ways
        n = next(n for n in range(1, len(move) + 1) if all(k for m, k in moves if m[:n] == move[:n]))
        via = lead[1:] + list(move[:n - 1])
        if len(via) > MAX_VIA:
          skipped += 1
          continue
        found.add((kind, lead[0], tuple(('w', v) for v in via) or (('n', junction),), move[n - 1]))
  return [(kind, wf, list(via), wt) for kind, wf, via, wt in sorted(found)], skipped, dead_ends


def metres(m):
  return f'{m:.2f}'.rstrip('0').rstrip('.')


def placement(lanes_left):
  """placement=* for a line `lanes_left` lane widths from the left edge of the lanes; None off the half lanes it can say."""
  k = round(lanes_left * 2)
  if abs(k - lanes_left * 2) > 1e-6 or k < 0:
    return None
  return ('left_of:1' if k == 0 else f'right_of:{k // 2}') if k % 2 == 0 else f'middle_of:{(k + 1) // 2}'


def layout(lf, back, freeway=False):
  """How a link's lanes are painted: (lane width, m right of the link to each direction's first lane on a two-way link,
  to its lanes' middle on a one-way one). GTA's offset field (`steps`, -7..7) between a two-way link's directions is its
  painted median, MEDIAN_STEP m a step whatever its lanes' width; CodeWalker's reading, a fourteenth of a lane, is kept
  where it's negative (lanes overlapping on the line) and on one-way links, which no measurement covers."""
  steps = ((lf[1] >> 4) & 7) * (-1 if lf[1] & 128 else 1)
  w = PAINTED_FREEWAY if freeway and not back else PAINTED_NARROW if lf[1] & 2 else PAINTED_LANE
  return w, steps * MEDIAN_STEP / 2 if back and steps > 0 else steps / 14 * w


def lane_tags(fwd, back, lf, freeway=False, bays=(False, False), painted=None):
  """The lanes of a link as GTA paints them (layout), in standard tags that osm_lanes.py reads back to the same layout:
  each direction's lanes starting `offset` right of the link, or a one-way link's centred on it and moved by the offset.
  - Offset 0: the line is the boundary between the directions, the middle of the road unless the counts differ
    (placement:forward/backward=left_of:1 then).
  - A gap between the directions (offset > 0) is a median: `width` is kerb to kerb, its lanes' widths are given, and
    what's left is the median, centred between the directions.
  - Both directions sharing one lane on the line (1 + 1 lanes, offset -0.5 lane) is a single-track road: lanes=1,
    unmarked, as real single-track lanes are mapped. Other overlaps can't be said in OSM: the kerbs are kept.
  - A turn bay folded into the road (`bays`, forward and backward: detached_bays) is that direction's leftmost lane. On
    a two-way road it fills the median, as GTA paints it (the median tapers away where the bay opens), so the line runs
    down its middle; bays both ways share the median, the line between them. Beside a one-way road it is a lane wide.
  - Where the paint was surveyed (`painted`: paint_survey.correct's widths each way and median), its lanes take the
    measured widths, kerbs where the layout has them; the line is the middle of the road between them, or the centre's
    with more lanes one way (`source:width=survey`). A surveyed one-way link's painted lanes are centred on it.
  `lf` is the link's flags and `freeway` whether its nodes are a freeway's (a one-way freeway's lanes are wider);
  returns the tags."""
  w, offset = layout(lf, back, freeway)
  steps = ((lf[1] >> 4) & 7) * (-1 if lf[1] & 128 else 1)
  bf, bb = (int(v) for v in bays)
  if painted and not back:  # centred on the link
    tags = {'lanes': str(fwd), 'oneway': 'yes', 'width': metres(sum(painted['lanes'])),
            'width:lanes': '|'.join(map(metres, painted['lanes'])), 'source:width': 'survey'}
    if 'change' in painted:
      tags['change:lanes'] = '|'.join(painted['change'])
    return tags
  if not back:
    tags = {'lanes': str(fwd), 'oneway': 'yes', 'width': metres(fwd * w)}
    if (steps or bf) and (where := placement(bf + (fwd - bf) / 2 - offset / w)):
      tags['placement'] = where
    return tags
  if steps == -7 and fwd == back == 1:
    return {'lanes': '1', 'width': metres(w), 'lane_markings': 'no'}
  if painted:  # the lanes as painted, their counts too
    wf, wb = painted['forward'], painted['backward']
    left, right = painted.get('parking', (0.0, 0.0))
    tags = {'lanes': str(len(wf) + len(wb)), 'lanes:forward': str(len(wf)), 'lanes:backward': str(len(wb)),
            'width': metres(sum(wf) + sum(wb) + painted['median'] + left + right), 'width:lanes:forward': '|'.join(map(metres, wf)),
            'width:lanes:backward': '|'.join(map(metres, wb)), 'divider': painted.get('divider', 'double_solid_line'),
            'source:width': 'survey'}
    for d in ('forward', 'backward'):
      if f'change:{d}' in painted:
        tags[f'change:lanes:{d}'] = '|'.join(painted[f'change:{d}'])
    for side, strip in (('left', left), ('right', right)):
      if strip:
        tags[f'parking:{side}'], tags[f'parking:{side}:width'] = 'lane', metres(strip)
    if len(wf) != len(wb) and not painted.get('middle'):
      tags['placement:forward'] = tags['placement:backward'] = 'left_of:1'
    return tags
  bay = 2 * offset / (bf + bb) if bf + bb else 0.0
  tags = {'lanes': str(fwd + back), 'lanes:forward': str(fwd), 'lanes:backward': str(back),
          'width': metres((fwd + back - bf - bb) * w + 2 * offset)}
  if offset > 0:
    tags['width:lanes:forward'] = '|'.join([metres(bay)] * bf + [metres(w)] * (fwd - bf))
    tags['width:lanes:backward'] = '|'.join([metres(bay)] * bb + [metres(w)] * (back - bb))
    tags['divider'] = 'double_solid_line'
  if bf and bb:
    tags['placement:forward'] = tags['placement:backward'] = 'left_of:1'
  elif bf or bb:
    tags['placement:forward' if bf else 'placement:backward'] = 'middle_of:1'
  elif fwd != back:
    tags['placement:forward'] = tags['placement:backward'] = 'left_of:1'
  return tags


def arrows(n, kinds, fewer_left=False):
  """The turn arrows of n lanes into a junction whose ways out turn `kinds` ways (left, through, right): every lane
  the same way at a forced turn; else the outer lanes also turn, from the outermost as GTA's cars do, and the others
  go through; with no way through, half turn each way, the fewer on the side with fewer lanes out."""
  order = [k for k in ('left', 'through', 'right') if k in kinds]
  if n == 1 or len(order) == 1:
    return [';'.join(order)] * n
  if 'through' in order:
    out = ['through'] * n
    if 'left' in order:
      out[0] = 'left;through'
    if 'right' in order:
      out[-1] = 'through;right'
    return out
  left = n // 2 if fewer_left else n - n // 2
  return ['left'] * left + ['right'] * (n - left)


def lane_turns(nodes, ways, lanes_to, junction, toward, left_only, restrictions, left_bays=frozenset(), medians=frozenset(),
               painted=None):
  """turn:lanes on the lanes into GTA's junctions where roads cross (see arrows), from the ways out of each, a move
  turning more than TURN_FLAG being a left or right turn, as for GTA's turn flags, less those the restrictions forbid.
  Marked on every way along the approach from APPROACH m before the junction while the road runs on with the same
  lanes (Valhalla reads them from the way into the junction). None where every lane only goes through, or where the
  road bends into the junction (APPROACH_BEND), which leaves which way is through moot. A one-lane approach gets none,
  as real mappers leave them out, unless it's GTA's left turn only lane; on a wider road that lane is the left one. A
  turn bay folded into its road (`left_bays`, its road's links) is its left lane, marked from where it opens, also
  where the road gains lanes on the way (`left|through` before that). On a two-way road with a median (`medians`, its links (node, next node) where the median has room for a lane)
  running in to a junction it may turn left at, GTA paints the median as a left-turn lane without a link of its own
  (measured on 4 of 4 such approaches): one more lane, `left`, on the approach's links within the median, if they are
  at least MEDIAN_LANE_MIN long. Arrows painted on the approach (`painted`, by link (node, next node):
  paint_survey.arrows) replace these where there are as many as lanes and the junction has their ways out. `ways` is
  [(way id, a, b, two_way)], `left_only` GTA's left turn only lane nodes, `restrictions` [(kind, from way, via, to way)]
  as written; returns {way id: {tag: value}}, how many approaches got
  arrows, how many were left out for bending, and the links (node, next node) given a median lane."""
  way_of, out, into = graph(ways)
  drawn = {wid: (a, two_way) for wid, a, _, two_way in ways}

  def length(p, q):
    return math.hypot(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y'])

  def heading(p, q):
    return game_heading(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y'])

  banned = defaultdict(list)  # from way -> [(via ways, to way)]
  for _, wf, via, wt in restrictions:
    banned[wf].append((tuple(ref for t, ref in via if t == 'w'), wt))

  def forbidden(seq):
    return any(tuple(seq[k + 1:k + 1 + len(via)]) == via and seq[k + 1 + len(via):k + 2 + len(via)] == [wt]
               for k, w in enumerate(seq) for via, wt in banned.get(w, ()))

  only_left = {(path[-2], path[-1]) for i in left_only for path in toward.get(i, {}).values()} | set(left_bays)
  tags, approaches, bent, opened = defaultdict(dict), 0, 0, set()
  for j in sorted(into):
    if not junction(j):
      continue
    for p in into[j]:
      n, only = lanes_to[(p, j)], (p, j) in only_left
      # not beside a bay GTA lays as its own link (one not folded in): that's the turn lane
      median = (p, j) in medians and not only and not any(
        q != p and lanes_to[(q, j)] == 1 and not lanes_to[(j, q)] and 'f' in nodes[q] and lane_node(nodes[q]) and
        abs(wrap(heading(q, j) - heading(p, j))) < STRAIGHT for q in into[j])
      if junction(p) or (n < 2 and not only and not median):
        continue
      chain, q, nxt, dist = [(p, j)], p, j, length(p, j)
      start = p if dist >= APPROACH_HEADING else None
      while dist < (math.inf if (p, j) in left_bays else APPROACH):
        prev = [r for r in into[q] if r != nxt]
        if junction(q) or len(prev) != 1 or set(out[q]) - {prev[0]} != {nxt} or \
           lanes_to[(prev[0], q)] != n and (prev[0], q) not in left_bays:
          break
        q, nxt = prev[0], q
        chain.append((q, nxt))
        dist += length(q, nxt)
        if start is None and dist >= APPROACH_HEADING:
          start = q
      h_in = heading(start or chain[-1][0], j)
      if abs(wrap(heading(p, j) - h_in)) > APPROACH_BEND:
        bent += 1
        continue
      lead, back = [way_of[e] for e in chain[::-1]], 0.0  # and the road further back, where restrictions can start
      while back < RESTRICTION_REACH and len(prev := [r for r in into[q] if r != nxt]) == 1:
        q, nxt = prev[0], q
        lead.insert(0, way_of[(q, nxt)])
        back += length(q, nxt)
      exits = [('left' if t > TURN_FLAG else 'right' if t < -TURN_FLAG else 'through', lanes_to[e])
               for t, inner, w, e in junction_exits(nodes, j, h_in, h_in, {k for e in chain for k in e}, out, way_of)
               if not forbidden(lead + inner + [w])]
      allowed = {k for k, _ in exits}
      if not allowed:
        continue
      fewer_left = max((m for k, m in exits if k == 'left'), default=0) < max((m for k, m in exits if k == 'right'), default=0)
      lanes = arrows(n, allowed, fewer_left)
      if only and n > 1 and 'left' in allowed and allowed - {'left'}:
        lanes = ['left', *arrows(n - 1, allowed - {'left'}, fewer_left)]
      inside = next((k for k, e in enumerate(chain) if e not in medians), len(chain))  # the median runs in to the junction
      if median and 'left' in allowed and allowed - {'left'} and sum(length(*e) for e in chain[:inside]) >= MEDIAN_LANE_MIN:
        chain = chain[:inside]
        lanes = ['left', *arrows(n, allowed - {'left'}, fewer_left)]
        opened.update(chain)
      elif n < 2 and not only:
        continue
      seen = next((painted[e] for e in chain if e in (painted or {})), None)
      if seen and len(seen) == len(lanes) and all(set(k.split(';')) <= allowed for k in seen):
        lanes = seen
      if all(lane == 'through' for lane in lanes):
        continue
      approaches += 1
      for e in chain:
        wid = way_of[e]
        a, two_way = drawn[wid]
        key = 'turn:lanes' if not two_way else 'turn:lanes:forward' if e[0] == a else 'turn:lanes:backward'
        # before the road widens beside a bay: the bay and lanes going on
        tags[wid].setdefault(key, '|'.join(lanes if lanes_to[e] == n else ['left'] + ['through'] * (lanes_to[e] - 1)))
  return tags, approaches, bent, opened


def node_id(k):
  return k[0] * 65536 + k[1] + 1


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('dump', help="ynddump's paths.jsonl")
  p.add_argument('out', help='.osm or .osm.pbf')
  p.add_argument('--sidecar', help="the roads as arrays for the bridge's map matching (.npz)")
  p.add_argument('--minimap', help="ynddump's minimap.jsonl: minor links GTA's minimap draws as roads become roads")
  p.add_argument('--survey', nargs='*', default=[], help="surveys of the game's road paint (paint_survey.py): the lane "
                 "widths where they were measured")
  args = p.parse_args()

  nodes, links, streets = load(args.dump)
  es = edges(nodes, links)
  cross = crossovers(nodes, es)
  es = {k: v for k, v in es.items() if k not in cross}
  used = sorted({k for e in es for k in e})
  print(f"{len(used)} nodes, {len(es)} ways ({len(cross)} carriageway crossovers dropped)")

  info = []  # (way id, a, b, fwd, back, class, limit, name, link flags)
  for i, ((a, b), (fwd, back, lf)) in enumerate(sorted(es.items())):
    if not fwd:  # one-way the other way: draw it in its direction of travel
      a, b, fwd, back = b, a, back, fwd
    cls = highway(nodes, a, b, fwd, back)
    st = nodes[a]['st'] if nodes[a]['st'] == nodes[b]['st'] else 0
    info.append([i + 1, a, b, fwd, back, cls, speed_limit(nodes, a, b, cls, fwd, back), streets.get(st), lf])
  out, into = defaultdict(list), defaultdict(list)
  for _, a, b, _, back, *_ in info:
    out[a].append(b)
    into[b].append(a)
    if back:
      out[b].append(a)
      into[a].append(b)

  def moves_out(k):
    return out.get(k, ())

  def moves_into(k):
    return into.get(k, ())

  def link_length(p, q):
    return math.hypot(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y'])

  def link_heading(p, q):
    return game_heading(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y'])

  def junction(k):  # where roads meet
    return bool(nodes[k]['f'][2] & 4) and roads_cross([link_heading(k, m) for m in {*out.get(k, ()), *into.get(k, ())}])
  bays = detached_bays(nodes, [(a, b, fwd, back, lf) for _, a, b, fwd, back, *_, lf in info], junction)
  bay_to, left_bays, gone = defaultdict(set), set(), set()  # way id -> the nodes its bay lanes head to
  for bay, road, _, j in bays:
    gone.update(bay)
    for k, (r, along) in enumerate(road):
      row = info[r]
      row[3 if along else 4] += 1
      a, b = (row[1], row[2]) if along else (row[2], row[1])
      bay_to[row[0]].add(b)
      left_bays.add((a, b))
  info = [row for r, row in enumerate(info) if r not in gone]
  used = sorted({k for _, a, b, *_ in info for k in (a, b)})
  out.clear()
  into.clear()
  for _, a, b, _, back, *_ in info:
    out[a].append(b)
    into[b].append(a)
    if back:
      out[b].append(a)
      into[a].append(b)
  print(f"{len(bays)} turn bays laid as their own links as lanes of their roads ({len(gone)} links dropped)")
  lanes = turn_lanes(nodes, [(a, b, bool(back), cls, name) for _, a, b, _, back, cls, _, name, _ in info], streets, junction)
  for r, (cls, name) in lanes.items():  # keeping their limits, which come from the street's
    info[r][5], info[r][7] = cls, name
  print(f"{len(lanes)} turn lane links as the road they split off")
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
  drawn = set()  # way ids
  if args.minimap:  # keeping their limits
    roads = minimap_classes(nodes, [(a, b, cls) for _, a, b, _, _, cls, *_ in info], minimap_roads(args.minimap))
    for r in roads:
      info[r][5] = 'unclassified'
      drawn.add(info[r][0])
    print(f"{len(roads)} minor links the minimap draws as roads ({sum(length[r] for r in roads) / 1000:.1f} km) as unclassified")
  names = street_runs(nodes, [(a, b, cls, name) for _, a, b, _, _, cls, _, name, _ in info], length)
  for r, name in names.items():
    info[r][7] = name
  print(f"{len(names)} unnamed links through junctions named for their street")

  # stop lines and GTA's turn flags are for the junction ahead of them
  stops = [k for k in used if nodes[k]['f'][1] >> 3 in (TRAFFIC_LIGHT, STOP_JUNCTION) and not junction(k)]
  flags = {k: (bool(nodes[k]['f'][0] & 128), bool(nodes[k]['f'][0] & 64), bool(nodes[k]['f'][4] & 128)) for k in used
           if (nodes[k]['f'][0] & 192 or nodes[k]['f'][4] & 128) and not junction(k)}
  score = junction_scores(stops, moves_out, link_length, junction)
  toward = {k: toward_junction(k, moves_out, moves_into, link_length, link_heading, junction, score) for k in {*stops, *flags}}
  lanes_to = {}
  for _, a, b, fwd, back, *_ in info:
    lanes_to[(a, b)], lanes_to[(b, a)] = fwd, back
  left_lanes = {k for k, (_, _, only) in flags.items() if only}  # on any road, for its turn arrows
  for k, (no_left, no_right, left_only) in flags.items():  # on a wider road, left turn only is its left lane's
    flags[k] = (no_left, no_right, left_only and all(lanes_to[(k, j)] == 1 for j in toward[k]))
  faced, flip = stop_directions(nodes, [(a, b, bool(back)) for _, a, b, _, back, *_ in info],
                                {k: set(toward[k]) for k in stops if toward[k]})
  for r in flip:
    row = info[r]
    row[1], row[2], row[3], row[4] = row[2], row[1], row[4], row[3]
  stop = {k: 'forward' if k in faced else 'both' if toward[k] else 'none' for k in stops}
  either, none = sum(v == 'both' for v in stop.values()), sum(v == 'none' for v in stop.values())
  print(f"{len(stops)} stop lines: {len(faced)} facing their junction ({len(flip)} ways turned round), {either} either way, " +
        f"{none} with no junction ahead left out")

  # GTA's pedestrian crossings are links of their own between crossing nodes, joined to no road
  crossings = {wid for wid, a, b, *_ in info if nodes[a]['f'][1] >> 3 == nodes[b]['f'][1] >> 3 == PED_CROSSING}
  ways = [(wid, a, b, bool(back)) for wid, a, b, _, back, *_ in info if wid not in crossings]
  restrictions = [('no_u_turn', *r) for r in no_u_turns(nodes, ways)]
  u_turns = len(restrictions)
  turns, skipped, dead_ends = turn_restrictions(nodes, ways, toward, flags)
  # the road a turn bay was folded into may turn left from it now: the moves are its lanes' and its bay's together
  bay_roads = {wid for wid, ends in bay_to.items() if ends}
  folded = [r for r in turns if r[0] == 'no_left_turn' and ({r[1]} | {ref for t, ref in r[2] if t == 'w'}) & bay_roads]
  restrictions += [r for r in turns if r not in folded]
  print(f"{len(folded)} no-left turns from roads now with their bay as a lane dropped")
  medians = set()  # two-way links (node, next node) with room for a turn lane in their median, and no turn bay that way
  for wid, a, b, fwd, back, *_, lf in info:
    if back and 2 * layout(lf, back)[1] >= BAY_MIN:
      medians.update(e for e in ((a, b), (b, a)) if e[1] not in bay_to[wid])
  survey = paint_survey.load(args.survey) if args.survey else {}
  painted_arrows = {}  # (node, next node) -> the lanes' painted turn arrows that way, left to right
  for _, a, b, *_ in info:
    if (samples := paint_survey.along(survey, a, b)) is not None:
      for key, e in paint_survey.arrows(samples).items():
        painted_arrows[(a, b) if key == 'forward' else (b, a)] = e
  row_of = {row[0]: row for row in info}
  painted, why, skipped, centre_kinds = {}, Counter(), 0, {}
  # where the game files' paint covers the map, the camera's survey only checks it
  from_files = any(d.get('src') == paint_survey.GAMEFILES for ds in survey.values() for d in ds)
  for wid, a, b, fwd, back, *_, lf in info:
    if (samples := paint_survey.along(survey, a, b)) is None:
      continue
    w, offset = layout(lf, back, bool(nodes[a]['f'][2] & nodes[b]['f'][2] & FREEWAY) and fwd >= 2)
    if (back and offset < 0) or (not back and offset) or bay_to[wid]:  # bays fill medians: not the road's paint there
      why['turn bay' if bay_to[wid] else 'lanes overlap' if back else 'one-way, off-centre'] += 1
      skipped += 1
      continue
    if not back:
      painted[wid], reason = paint_survey.correct_oneway(samples, fwd, (-fwd * w / 2, fwd * w / 2))
      why[reason or 'measured one-way (game files)'] += 1
      continue
    use, other = paint_survey.sources(samples, camera_corrects=not from_files)
    if not use:
      why['no game files (camera a check only)'] += 1
      skipped += 1
      continue
    kerbs = (-(offset + back * w), offset + fwd * w)
    files = use[0].get('src') == paint_survey.GAMEFILES
    painted[wid], reason = paint_survey.correct(use, fwd, back, kerbs, counts_from_paint=files)
    if got := painted[wid]:
      medians.discard((a, b))  # the paint says where the turn lanes are
      medians.discard((b, a))
      if (len(got['forward']), len(got['backward'])) != (fwd, back):  # as many lanes as painted, arrows and all
        why[f"lanes from the paint: {len(got['backward'])}+{len(got['forward'])} where GTA has {back}+{fwd}"] += 1
        row = row_of[wid]
        row[3], row[4] = len(got['forward']), len(got['backward'])
        lanes_to[(a, b)], lanes_to[(b, a)] = row[3], row[4]
    why[reason or ('measured (game files)' if files else 'measured (camera)')] += 1
    if painted[wid] and len(other) >= paint_survey.MIN_SAMPLES and (check := paint_survey.correct(other, fwd, back, kerbs)[0]):
      why['sources disagree' if paint_survey.disagree(painted[wid], check) else 'sources agree'] += 1
    if not painted[wid] and offset <= 0 and (kind := paint_survey.centre_kind(samples, 0.0, paint_survey.CENTRE_TOL)):
      centre_kinds[wid] = kind  # the centre line's kind still shows where the lanes don't add up
  print(f"{len(centre_kinds)} more two-way links' centre lines of the kind the game files paint")
  arrows_at, approaches, bent, opened = lane_turns(nodes, ways, lanes_to, junction, toward, left_lanes, restrictions,
                                                   left_bays, medians, painted_arrows)
  way_of = graph(ways)[0]
  for p, q in opened:
    row = row_of[way_of[(p, q)]]
    row[3 if row[2] == q else 4] += 1
    bay_to[row[0]].add(q)
  print(f"{len(opened)} links into junctions with their median painted as a left-turn lane")
  if survey:
    print(f"paint survey of {len(painted) + skipped} links: " +
          ', '.join(f'{n} {k}' for k, n in why.most_common()))
  print(f"{approaches} approaches to junctions with turn arrows, on {len(arrows_at)} ways ({bent} bending into theirs left out)")
  layer_of = {ways[i][0]: v for i, v in levels(nodes, [(a, b) for _, a, b, _ in ways]).items()}
  print(f"{len(layer_of)} ways over others as bridges (up to layer {max(layer_of.values(), default=0)})")

  w = osmium.SimpleWriter(args.out, overwrite=True)
  for k in used:
    n = nodes[k]
    lat, lon = to_lat_lon(n['x'], n['y'])
    w.add_node(osmium.osm.mutable.Node(id=node_id(k), version=1, location=(lon, lat),
                                       tags={**node_tags(n, stop.get(k, 'both')), 'ele': f"{n['z']:.1f}"}))
  for wid, a, b, fwd, back, cls, limit, name, lf in info:
    if wid in crossings:
      w.add_way(osmium.osm.mutable.Way(id=wid, version=1, nodes=[node_id(a), node_id(b)],
                                       tags={'highway': 'footway', 'footway': 'crossing', 'crossing': 'marked'}))
      continue
    freeway = bool(nodes[a]['f'][2] & nodes[b]['f'][2] & FREEWAY) and fwd >= 2
    tags = {'highway': cls, **lane_tags(fwd, back, lf, freeway, (b in bay_to[wid], a in bay_to[wid]), painted.get(wid)),
            **arrows_at.get(wid, {}), 'maxspeed': f'{limit} mph'}
    if fwd == back == 1 and 'lane_markings' not in tags and cls in MARKED:
      tags.setdefault('divider', 'double_solid_line')  # how GTA paints most two-lane roads' centre (OSM reads dashed)
    if wid in centre_kinds and 'lane_markings' not in tags:
      tags['divider'] = centre_kinds[wid]
    if name and not cls.endswith("_link"):  # a ramp named for its freeway reads as staying on it
      tags['name'] = name
    if wid in destination:
      tags['destination'] = destination[wid]
    if wid in layer_of:
      tags['bridge'], tags['layer'] = 'yes', str(layer_of[wid])
    fa, fb = nodes[a]['f'], nodes[b]['f']
    for k, bit, tag in ((2, SWITCHED_OFF, 'gta:switched_off'), (2, NO_GPS, 'gta:no_gps'), (0, OFFROAD, 'gta:offroad')):
      if (fa[k] | fb[k]) & bit:
        tags[tag] = 'yes'
    if wid in drawn and 'gta:offroad' in tags:
      tags['surface'] = 'unpaved'
    if lf[2] & 1:
      tags['gta:no_nav'] = 'yes'
    w.add_way(osmium.osm.mutable.Way(id=wid, version=1, nodes=[node_id(a), node_id(b)], tags=tags))
  for i, (kind, wi, via, wo) in enumerate(restrictions):
    via = [(t, node_id(ref) if t == 'n' else ref, 'via') for t, ref in via]
    w.add_relation(osmium.osm.mutable.Relation(id=i + 1, version=1, tags={'type': 'restriction', 'restriction': kind},
                                               members=[('w', wi, 'from'), *via, ('w', wo, 'to')]))
  w.close()
  kinds = ', '.join(f'{sum(t[0] == k for t in turns)} {k}' for k in ('no_left_turn', 'no_right_turn', 'no_straight_on'))
  print(f"{u_turns} U-turns forbidden; GTA's turn flags: {len(turns)} turns forbidden ({kinds}), {skipped} through too many ways and {dead_ends} " +
        "approaches GTA leaves no way out of left out")
  if args.sidecar:
    n = write_sidecar(args.sidecar, nodes, used, [r[:8] for r in info])
    print(f"{n} links crossing on different levels -> {args.sidecar}")


if __name__ == '__main__':
  main()
