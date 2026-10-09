#!/usr/bin/env python3
"""Converts GTA V's vehicle path nodes (ynddump's JSON lines) into an OpenStreetMap file a standard router can use.

Game coordinates (metres, x east, y north) map to degrees about (0, 0) on a sphere, see gta5_map.to_lat_lon. Each node
link becomes a way with OSM lane tags; GTA flags with no OSM equivalent keep a gta: prefix.
"""
import argparse
import json
import math
import os
from collections import Counter, defaultdict

import osmium

from openpilot.tools.sim.bridge.gta5.map import paint_survey, traps
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_lat_lon
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, FORWARD, WayLanes
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
ARROW_REACH = 90.0  # m before a junction that the arrows painted in its approach's lanes are theirs (9 in 10 are within 80 m)
ARROW_ALIGN, ARROW_DZ, ARROW_SPILL = 30.0, 2.5, 0.5  # deg, m above or below, m outside a lane: a painted arrow is in it
THROUGH_SKEW = 55.0  # deg: a road going on this skewed is painted through (validate_lanes' through reaches 60)
# m: lanes as painted, measured in the game (CodeWalker's 5.5 / 4.0 m are the AI's lanes, not the paint: narrow lanes are
# painted 4.2-4.5 m, freeway lanes 5.9-6.5 m), and the painted median per step of a two-way link's offset (5.2-5.5 m
# at 6 steps on narrow and normal links alike)
PAINTED_LANE, PAINTED_NARROW, PAINTED_FREEWAY = 5.5, 4.4, 6.1
MEDIAN_STEP = 0.9
MEDIAN_LANE_MIN = 15.0  # m: a median runs in to a junction at least this far to be painted as a turn lane
FREEWAY = 64  # node flags 2: a freeway's
MARKED = {'trunk', 'primary', 'residential'}  # classes of GTA's streets with painted centre lines
BAY_MIN = 2.5  # m: a two-way road's median narrower than this has no room for a turn bay
EDGE_GAP = 0.5  # m: a painted median narrower than this is the double centre line's gap


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


SPLIT_NODE_AREA = 4097  # the nodes added where a divided road's carriageways are parted (split_shared) are (this, n)
SHARED_BEND = 75.0  # deg: a carriageway bending less than this at a node goes on through it
SHARED_AWAY = 2.0  # m: a node this far left of both its carriageways' own lines is in the median between them


def split_shared(nodes, info):
  """Parts the nodes where GTA runs both carriageways of a divided road through one node in the median between them, as
  where a side road meets it at a gap in the median: each carriageway gets a node of its own on its line between its
  nodes either side, and a two-way link across the median joins them, as GTA lays most such junctions itself (where
  they're no more than JUNCTION_SPAN apart, as far as GTA's turn flags are read across a junction). The node's other
  links go to the carriageway on their side; those whose far end is in the median too, as a turn lane, keep the node,
  and the link across runs through it. A node two divided roads cross at is left alone, and so is one
  with lanes going on through it between the carriageways: those are one road's lanes, laid as links side by side.
  `info` rows are [way id, a, b, fwd, back, class, limit, name, link flags] with a -> b one way's direction: changed in
  place, the links across added; returns how many nodes were parted."""
  ins, outs, touching = defaultdict(list), defaultdict(list), defaultdict(list)
  for r, (_, a, b, _, back, *_) in enumerate(info):
    touching[a].append(r)
    touching[b].append(r)
    if not back:
      outs[a].append(r)
      ins[b].append(r)

  def xy(k):
    return nodes[k]['x'], nodes[k]['y']

  def unit(p, q):
    dx, dy = q[0] - p[0], q[1] - p[1]
    d = math.hypot(dx, dy) or 1.0
    return dx / d, dy / d

  def left_of(p, q, c):  # m left of the line p -> q
    ux, uy = unit(p, q)
    return ux * (c[1] - p[1]) - uy * (c[0] - p[0])

  def angle(u, v):
    return math.degrees(math.acos(max(-1.0, min(1.0, u[0] * v[0] + u[1] * v[1]))))

  next_id, made, parted = max(r[0] for r in info) + 1, 0, 0
  for n in sorted(ins):
    if n not in outs or nodes[n]['f'][2] & FREEWAY or nodes[n]['f'][1] >> 3 == PED_CROSSING:
      continue
    pn = xy(n)
    pairs = [(ri, ro, info[ri][1], info[ro][2]) for ri in ins[n] for ro in outs[n]  # carriageways on through n
             if info[ri][1] != info[ro][2] and angle(unit(xy(info[ri][1]), pn), unit(pn, xy(info[ro][2]))) < SHARED_BEND]
    found = []
    for k, p in enumerate(pairs):
      for q in pairs[k + 1:]:
        if len({*p, *q}) < 8 or angle(unit(xy(p[2]), xy(p[3])), unit(xy(q[3]), xy(q[2]))) > 35.0:
          continue
        away = min(left_of(xy(p[2]), xy(p[3]), pn), left_of(xy(q[2]), xy(q[3]), pn))
        if away >= SHARED_AWAY:
          found.append((away, p, q))
    if not found:
      continue
    _, p, q = max(found)
    if any(not {*f[1][:2], *f[2][:2]} & {*p[:2], *q[:2]} for f in found):
      continue  # two divided roads crossing
    if len(touching[n]) == 4:
      continue  # a gap in the median for U-turns only: no junction to lay out
    at = []  # where n is along each carriageway's line
    for _, _, i, o in (p, q):
      dx, dy = nodes[o]['x'] - nodes[i]['x'], nodes[o]['y'] - nodes[i]['y']
      at.append(((pn[0] - nodes[i]['x']) * dx + (pn[1] - nodes[i]['y']) * dy) / max(dx * dx + dy * dy, 1e-9))
    if not all(0.1 < t < 0.9 for t in at):
      continue
    placed = [(nodes[i]['x'] + (nodes[o]['x'] - nodes[i]['x']) * t, nodes[i]['y'] + (nodes[o]['y'] - nodes[i]['y']) * t)
              for (_, _, i, o), t in zip((p, q), at, strict=True)]
    if math.dist(*placed) > JUNCTION_SPAN:
      continue  # GTA's turn flags are read across a junction no further than this
    middle = [r for r in touching[n] if r not in (p[0], p[1], q[0], q[1]) and  # the links between the carriageways
              all(left_of(xy(i), xy(o), xy(info[r][2] if info[r][1] == n else info[r][1])) >= 0.0 for _, _, i, o in (p, q))]
    if any(info[ri][2] == n and info[ro][1] == n and info[ri][1] != info[ro][2] and
           angle(unit(xy(info[ri][1]), pn), unit(pn, xy(info[ro][2]))) < SHARED_BEND for ri in middle for ro in middle):
      continue  # lanes going on through between them: one road's, not a median
    ends = []  # (node, the carriageway's far nodes)
    for (ri, ro, i, o), t in zip((p, q), at, strict=True):
      a, b = nodes[i], nodes[o]
      made += 1
      k = (SPLIT_NODE_AREA, made)
      nodes[k] = {**nodes[n], 'x': a['x'] + (b['x'] - a['x']) * t, 'y': a['y'] + (b['y'] - a['y']) * t,
                  'z': a['z'] + (b['z'] - a['z']) * t}
      info[ri][2], info[ro][1] = k, k
      ends.append((k, i, o))
    for r in touching[n]:
      if r in (p[0], p[1], q[0], q[1]) or r in middle:
        continue
      row = info[r]
      beyond = [k for k, i, o in ends if left_of(xy(i), xy(o), xy(row[2] if row[1] == n else row[1])) < 0.0]
      row[1 if row[1] == n else 2] = beyond[0]  # past that carriageway, on its side
    lf = list(info[p[0]][8])
    lf[1] &= ~0xF0  # no offset between its directions
    lf[2] = (lf[2] & ~0xFC) | (1 << 5) | (1 << 2)  # a lane each way
    cls = info[p[0]][5]  # its road's class: as a minor road, routers leave it out of long routes' turns across
    for a, b in [(ends[0][0], n), (n, ends[1][0])] if middle else [(ends[0][0], ends[1][0])]:
      info.append([next_id, a, b, 1, 1, cls, info[p[0]][6], None, lf])
      next_id += 1
    parted += 1
  return parted


DEAD_END_LINKS = 3  # two-way links in a row at most that a direction going nowhere is taken off


def dead_end_lanes(info):
  """GTA's two-way links running on from a one-way link at a node no other link meets: the direction the one-way
  link doesn't carry goes nowhere there (or comes from nowhere), a lane GTA's AI never drives. Each takes the one-way
  link's direction, and so on along the two-way links before it up to DEAD_END_LINKS (`info` rows [way id, a, b,
  fwd, back, ...] changed in place, drawn in their new direction of travel); returns how many."""
  changed = 0
  for _ in range(DEAD_END_LINKS):
    at = defaultdict(list)
    for row in info:
      at[row[1]].append(row)
      at[row[2]].append(row)
    keep = defaultdict(set)  # id(row) -> its directions kept: a -> b (True) or b -> a
    rows_of = {}
    for n, rows in at.items():
      if len(rows) != 2 or rows[0] is rows[1]:
        continue
      two, one = (rows[0], rows[1]) if rows[0][4] else (rows[1], rows[0])
      if not two[4] or one[4] or not two[3]:
        continue
      into = one[2] == n  # traffic arrives at n along the one-way link: it leaves along the two-way one
      keep[id(two)].add((two[1] == n) == into)
      rows_of[id(two)] = two
    found = 0
    for k, ways in keep.items():
      if len(ways) != 1:  # going nowhere both ways
        continue
      two = rows_of[k]
      if not ways.pop():
        two[1], two[2], two[3] = two[2], two[1], two[4]
      two[4] = 0
      found += 1
    changed += found
    if not found:
      break
  return changed


LANE_CHANGE_FLOW = 30.0  # deg: one-way links within this of their mean heading all run the same way
LANE_CHANGE_AXIS = 50.0  # deg: a link within this of that heading crosses between them, as a lane change


def lane_changes(nodes, es):
  """Two-way links between one-way links that all run the same way, as GTA lays some lane changes across a freeway or a
  one-way street (chains of them crossing in an X): {edge: whether it runs a -> b}, the way the traffic beside it runs.
  Every other two-way link at their ends must be one too, so that roads of their own stay two-way."""
  flow = defaultdict(list)  # node -> directions of travel of its one-way edges
  twos = defaultdict(set)
  for (a, b), (fwd, back, _) in es.items():
    if fwd and back:
      twos[a].add((a, b))
      twos[b].add((a, b))
    elif fwd or back:
      d = direction(nodes, a, b) if fwd else direction(nodes, b, a)
      flow[a].append(d)
      flow[b].append(d)
  out = {}
  for (a, b), (fwd, back, _) in es.items():
    if not (fwd and back) or not flow[a] or not flow[b]:
      continue
    ds = flow[a] + flow[b]
    mx, my = sum(d[0] for d in ds), sum(d[1] for d in ds)
    n = math.hypot(mx, my) or 1.0
    mx, my = mx / n, my / n
    ux, uy = direction(nodes, a, b)
    along = ux * mx + uy * my
    if min(d[0] * mx + d[1] * my for d in ds) > math.cos(math.radians(LANE_CHANGE_FLOW)) and \
        abs(along) > math.cos(math.radians(LANE_CHANGE_AXIS)):
      out[(a, b)] = along > 0
  while bad := {e for e in out if any(o not in out for n in e for o in twos[n])}:
    for e in bad:
      del out[e]
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
  Turning back on to another way is a U-turn only where the move leaves some other way on: where no node along it has
  one, it is the road itself, bending back round a hairpin or out of an acute junction.
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

  def choice(n, w_in, w_on):  # a way on from n other than w_on, arriving on w_in (and not back along it)
    return any(w not in (w_in, w_on) for w, _ in leave[n])

  out = []
  for n, ins in arrive.items():
    for wi, hi in ins:
      for wo, ho in leave[n]:
        if abs(turn(hi, ho)) > U_TURN and (wo == wi or choice(n, wi, wo)):
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
      stack = [(n, [], 0.0, hi, 0.0, False)]  # summing the turns, as a loop can come round past 180 deg
      while stack:
        p, via, dist, h, turned, left = stack.pop()
        for wv, q, hv in short_from[p]:
          if wv == wi or wv in via or dist + length[wv] > GAP:
            continue
          chain, turned_v = via + [wv], turned + turn(h, hv)
          left_v = left or choice(p, via[-1] if via else wi, wv)
          for wo, ho in leave[q]:
            if wo != wi and wo not in chain and abs(turned_v + turn(hv, ho)) > U_TURN and (left_v or choice(q, wv, wo)):
              out.append((wi, [('w', v) for v in chain], wo))
          if len(chain) < GAP_LINKS:
            stack.append((q, chain, dist + length[wv], hv, turned_v, left_v))
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


MINOR = {'service', 'track', 'unclassified'}  # roads too minor to end the road a turn lane opens along
TAPER_BACK = 100.0  # m back from where a turn lane's link began that its taper is looked for
TAPER_SNAP = 1.0  # m: a taper starting or ending this near a node starts or ends there, no node added
KEYS = ('forward', 'backward')
LINE_CELL = 25.0  # m: the squares the game files' polylines are looked up by
TAPER_NODE_AREA = 4096  # GTA's areas are below this: the nodes added where tapers start and end are (this, n)
TAPER_DEFAULT = 10.0  # m: a turn lane with no painted opening widens over this, to full width where GTA's lane begins
CARRIAGEWAY_TURN = 40.0  # deg: a carriageway joining a two-way road bends up to this into it
CARRIAGEWAY_APART = (3.0, 25.0)  # m between the two carriageways of a road GTA lays as one-way links side by side


def lane_tapers(nodes, info, lane_links, junction, survey, lines=None, why=None, recounted=None, ends=frozenset()):
  """Where the turn lanes folded into roads (detached_bays) or painted in medians (lane_turns) open, from the game
  files' paint: the median's right edge swings across over the taper, read off their yellow polylines (,
  paint_survey.swing_taper) where given, else off their sections (paint_survey.opening_taper). `lane_links` are the
  links (node, next node) carrying such a lane, `info` the rows [way id, a, b, fwd, back, ...]; returns [(the links
  along the road to the junction [(a, b)], m along to each one's start, the taper's start and end m along)], the road
  reaching back up to TAPER_BACK before the lane's links, across minor roads (service roads and tracks) meeting it and
  the other way's turn lanes (back to back bays share the median, each opening from its own end).
  The paint is looked for further back too, along the carriageways GTA lays as one-way links side by side before they
  join into the road, read along the line midway between them: a lane opening there is already open where the road
  begins (its start is before the road's, negative). Where the paint shows no opening, the lane widens over
  TAPER_DEFAULT to full width where its links begin, as the game paints most bays (on the surveyed ones a median 9 m
  long, ending 5 m before GTA's lane begins), on the road before where it has a median to open in, else from where they
  begin; where the paint counts the lane on the links before its own (`recounted`: {way id: GTA's lanes forward and
  backward} of the links whose counts are the paint's), to full width where that begins. `why` counts each lane's
  case. `ends`: links a lane ends on that aren't into a junction (painted_median_lanes')."""
  import numpy as np
  why = why if why is not None else Counter()
  recounted = recounted or {}

  def painted_on(a, b):  # the paint counts one more lane than GTA on the link a -> b
    row = rows[(a, b)]
    return row[0] in recounted and row[3 if row[1] == a else 4] > recounted[row[0]][0 if row[1] == a else 1]
  one_ways = defaultdict(list)  # one-way rows by the LINE_CELL squares their nodes are in
  for row in info:
    if row[3] and not row[4]:
      for k in (row[1], row[2]):
        one_ways[(int(nodes[k]['x'] // LINE_CELL), int(nodes[k]['y'] // LINE_CELL))].append(row)

  def xy(k):
    return np.array([nodes[k]['x'], nodes[k]['y']])

  def apart(q, s0):  # m left of the one-way link q -> s0 to the middle between it and its road's other carriageway
    p0, p1 = xy(q), xy(s0)
    d = (p1 - p0) / max(float(np.hypot(*(p1 - p0))), 1e-9)
    left, mid = np.array([-d[1], d[0]]), (p0 + p1) / 2
    cx, cy = int(mid[0] // LINE_CELL), int(mid[1] // LINE_CELL)
    best = None
    near = {id(r): r for i in (-1, 0, 1) for k in (-1, 0, 1) for r in one_ways.get((cx + i, cy + k), ())}
    for row in near.values():
      a, e = xy(row[1]), xy(row[2]) - xy(row[1])
      size = float(np.hypot(*e))
      if size < 1.0 or (e / size) @ d > -math.cos(math.radians(STRAIGHT)):  # not running the other way
        continue
      foot = a + e * np.clip((mid - a) @ e / size ** 2, 0.0, 1.0)
      lat = float((foot - mid) @ left)
      if CARRIAGEWAY_APART[0] < lat < CARRIAGEWAY_APART[1] and abs(float((foot - mid) @ d)) < LINE_CELL / 2 and \
         (best is None or lat < best):
        best = lat
    return None if best is None else best / 2
  grid = defaultdict(list)  # the polylines by LINE_CELL squares they pass through
  for k, (_, pts) in enumerate(lines or []):
    for cell in {(int(x // LINE_CELL), int(y // LINE_CELL)) for x, y in pts}:
      grid[cell].append(k)
  rows = {}
  at = defaultdict(list)
  out_of = defaultdict(list)  # node -> [(next node, lanes that way, two-way)]
  for row in info:
    _, a, b, fwd, back = row[:5]
    rows[(a, b)] = rows[(b, a)] = row
    at[a].append(row)
    at[b].append(row)
    if fwd:
      out_of[a].append(b)
    if back:
      out_of[b].append(a)
  into = defaultdict(list)
  for a, nxt in out_of.items():
    for b in nxt:
      into[b].append(a)

  def length(p, q):
    return math.hypot(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y'])

  def heading(p, q):
    return game_heading(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y'])

  found = []
  for p, j in sorted(lane_links):
    if not junction(j) and (p, j) not in ends:
      continue
    chain = [(p, j)]
    while (prev := [q for q in into[chain[0][0]] if (q, chain[0][0]) in lane_links and q != chain[0][1] and (q, chain[0][0]) not in chain]):
      chain.insert(0, (prev[0], chain[0][0]))
    if not all(rows[e][4] and rows[e][3] for e in chain):  # in a two-way road's median
      continue
    back = 0.0
    def minor(n, s1):  # where only minor roads meet the road
      return all(r[5] in MINOR for r in at[n] if s1 not in (r[1], r[2]) and rows.get((n, s1)) is not r and
                 not (abs(wrap(heading(r[1] if r[2] == n else r[2], n) - heading(n, s1))) < STRAIGHT))

    while back < TAPER_BACK and (len(at[chain[0][0]]) == 2 or minor(*chain[0])):  # the road just carrying on
      s0, s1 = chain[0]
      prev = [q for q in into[s0] if q != s1 and abs(wrap(heading(q, s0) - heading(s0, s1))) < STRAIGHT and rows[(q, s0)][3] and
              rows[(q, s0)][4] and 2 * layout(rows[(q, s0)][8], rows[(q, s0)][4])[1] >= BAY_MIN]  # with a median to open in
      if len(prev) != 1 or (prev[0], s0) in chain:
        break
      chain.insert(0, (prev[0], s0))
      back += length(prev[0], s0)
    starts = np.concatenate(([0.0], np.cumsum([length(a, b) for a, b in chain])))
    row = rows[chain[-1]]
    median = 2 * layout(row[8], row[4])[1]
    if median < BAY_MIN:
      continue
    begins = back  # m along where the lane's links begin, or before them the links the paint counts it on
    while (k := int(np.searchsorted(starts, begins - 1e-6))) > 0 and painted_on(*chain[k - 1]):
      begins = float(starts[k - 1])
    beyond = []
    if lines is not None:
      beyond, ahead, run = [], chain[0], 0.0  # the line midway between the carriageways before the road, back to front
      while back + run < TAPER_BACK:
        s0, s1 = ahead
        h = heading(s0, s1)
        prev = [q for q in into[s0] if q != s1 and not rows[(q, s0)][4] and abs(wrap(heading(q, s0) - h)) < CARRIAGEWAY_TURN]
        if len(prev) != 1:
          break

        def other_way(r, s0=s0, h=h):  # a minor road, or the other carriageway leaving s0
          return r[5] in MINOR or (not r[4] and r[1] == s0 and abs(wrap(heading(s0, r[2]) - h - 180.0)) < 60.0)
        if not all(other_way(r) for r in at[s0] if r is not rows[(s0, s1)] and r is not rows[(prev[0], s0)]) or \
           (off := apart(prev[0], s0)) is None:
          break
        p0, p1 = xy(prev[0]), xy(s0)
        d = (p1 - p0) / max(float(np.hypot(*(p1 - p0))), 1e-9)
        beyond.insert(0, p0 + np.array([-d[1], d[0]]) * off)
        run += float(np.hypot(*(p1 - p0)))
        ahead = (prev[0], s0)
      road = np.array(beyond + [xy(a) for a, _ in chain] + [xy(chain[-1][1])])
      extra = float(np.hypot(*np.diff(road[:len(beyond) + 1], axis=0).T).sum())
      lo, hi = road.min(axis=0) - median, road.max(axis=0) + median
      ks = {k for cx in range(int(lo[0] // LINE_CELL), int(hi[0] // LINE_CELL) + 1)
            for cy in range(int(lo[1] // LINE_CELL), int(hi[1] // LINE_CELL) + 1) for k in grid.get((cx, cy), ())}
      taper = paint_survey.swing_taper([lines[k] for k in sorted(ks)], road, median / 2)
      if taper is not None:
        taper = None if taper[0] < 0.5 else (taper[0] - extra, max(taper[1] - extra, 0.0))  # (seen opening)
    else:
      sections = []
      for (a, b), d0 in zip(chain, starts, strict=False):
        for sample in paint_survey.along(survey, a, b) or []:
          if 's' in sample:
            sections.append((d0 + sample['s'], sample))
      taper = paint_survey.opening_taper(sections, median)
      if taper is not None and taper[0] < 0.5:
        taper = None
    if taper is not None and taper[1] > starts[-1] - 5.0:  # opening into the junction
      taper = None
    if taper is not None:
      why['painted' if taper[0] >= 0.0 else 'painted, opening before the road begins'] += 1
    elif beyond:  # a median between carriageways ends where the road begins, the lane in its room from there
      taper = (0.0, 0.0)
      why['no paint: open where the carriageways join'] += 1
    elif begins < TAPER_DEFAULT + 5.0 and len(at[chain[0][0]]) > 2:  # a junction just before: open from there
      taper = (0.0, 0.0)
      why['no paint: full width from the junction just before it'] += 1
    elif begins >= 3.0:  # widening on the road before, to full width where its links begin
      taper = (max(begins - TAPER_DEFAULT, 0.0), begins)
      why["no paint: full width where GTA's lane begins"] += 1
    elif begins < back:  # the paint counts it all along the road before
      taper = (0.0, 0.0)
      why['no paint: full width where the painted lanes begin'] += 1
    elif len(at[chain[0][0]]) > 2:  # beginning at a junction (or where other roads join): full width from there
      why['no paint: full width from the junction it begins at'] += 1
      continue
    elif starts[-1] - 5.0 - begins >= 3.0:
      taper = (begins, min(begins + TAPER_DEFAULT, starts[-1] - 5.0))
      why['no paint, no road before: widening from where it begins'] += 1
    else:
      why['too short to widen'] += 1
      continue
    found.append((chain, starts, *taper))
  return found


def split_tapers(nodes, info, tapers, lane_links, arrows_at, bay_to, recounted=None):
  """The roads' rows with the turn lanes of `tapers` (lane_tapers) from where they open: a lane added to the links of
  the road before the lane's links where it opens earlier, taken off those where it opens later, links split where it
  starts and ends opening (a node added there, along the link, unless one is within TAPER_SNAP), and the links where it
  opens widening it from nothing (width:lanes...:start / :end). Returns the new rows, {new way id: the way it was split
  from}, {way id: {'forward' | 'backward': (lane width share at its first node, at its last)}} for the widening, and
  {way id: (a bay forward, a bay backward)} for lane_tags of the rows it changes, and how many tapers were applied: one
  a way and direction. A link carries the lane where it's one of `lane_links`, or where its lane counts are the
  paint's (`recounted`: {way id: GTA's lanes forward and backward}) and count one more than GTA that way. Each
  way's bay opens from its own end of a median they share (back to back bays). The rows' arrows (arrows_at) follow
  their lanes."""
  rows = {row[0]: row for row in info}
  head = {}  # (a, b) -> way id
  for row in info:
    head[(row[1], row[2])] = head[(row[2], row[1])] = row[0]
  cuts, effects = defaultdict(set), defaultdict(list)
  taken, applied = set(), 0  # (way id, forward)
  recounted = recounted or {}

  def carrying(a, b):
    wid = head[(a, b)]
    if wid in recounted:
      fwd = rows[wid][1] == a
      return rows[wid][3 if fwd else 4] > recounted[wid][0 if fwd else 1]
    return (a, b) in lane_links

  for chain, starts, start, end in tapers:
    start, end = (next((float(d) for d in starts if abs(d - at) <= TAPER_SNAP), at) for at in (start, end))
    # one taper a way and direction; no lanes taken off down to none
    sides = [(head[(a, b)], rows[head[(a, b)]][1] == a, carrying(a, b)) for a, b in chain]
    if any((wid, fwd) in taken or (carried and rows[wid][3 if fwd else 4] < 2) for wid, fwd, carried in sides):
      continue
    taken.update((wid, fwd) for wid, fwd, *_ in sides)
    applied += 1
    template = None
    for a, b in chain:
      wid = head[(a, b)]
      row = rows[wid]
      key = 'forward' if row[1] == a else 'backward'
      tag = arrows_at.get(wid, {}).get(f'turn:lanes:{key}')
      if template is None and (a, b) in lane_links and tag:
        template = tag
    for k, (a, b) in enumerate(chain):
      wid = head[(a, b)]
      row = rows[wid]
      forward = row[1] == a
      d0, d1 = starts[k], starts[k + 1]
      for at in (start, end):
        if d0 + TAPER_SNAP < at < d1 - TAPER_SNAP:
          cuts[wid].add(round((at - d0) / (d1 - d0) if forward else (d1 - at) / (d1 - d0), 4))
      # m along the road at the row's first and last node
      effects[wid].append((forward, d0 if forward else d1, d1 if forward else d0, start, end, carrying(a, b), template,
                           b == chain[-1][1]))
  out, parent, widen, bays = [], {}, {}, {}
  next_id = max(rows) + 1
  synthetic = 0
  for row in info:
    wid = row[0]
    if wid not in effects:
      out.append(row)
      continue
    ts = [0.0, *sorted(cuts[wid]), 1.0]
    original = dict(arrows_at.get(wid, {}))
    ends = [row[1]]
    for t in ts[1:-1]:
      a, b = nodes[row[1]], nodes[row[2]]
      synthetic += 1
      k = (TAPER_NODE_AREA, synthetic)
      nodes[k] = {'a': k[0], 'i': k[1], 'x': a['x'] + (b['x'] - a['x']) * t, 'y': a['y'] + (b['y'] - a['y']) * t,
                  'z': a['z'] + (b['z'] - a['z']) * t, 'f': [0, 0, 0, 0, 0], 'st': a['st'], 'sp': a.get('sp', 1)}
      ends.append(k)
    ends.append(row[2])
    for n, (t0, t1) in enumerate(zip(ts, ts[1:], strict=False)):
      pid = wid if n == 0 else next_id
      if n:
        next_id += 1
        parent[pid] = wid
      piece = [pid, ends[n], ends[n + 1], *row[3:]]
      tags = dict(original)
      fb = [row[2] in bay_to[wid], row[1] in bay_to[wid]]  # bays this way that open elsewhere
      for forward, da, db, start, end, carried, template, last in effects[wid]:
        key = 'forward' if forward else 'backward'
        p0, p1 = da + (db - da) * t0, da + (db - da) * t1  # m along the road at the piece's ends
        carries = (p0 + p1) / 2 > start
        lanes = piece[3 if forward else 4] - carried + carries
        piece[3 if forward else 4] = lanes
        arrows = tags.get(f'turn:lanes:{key}')
        if carries and not carried and arrows:  # (arrows only on the ways into a junction, as lane_turns)
          # into the lane's junction its arrows there; into one before it, on through
          if not last:
            arrows = 'through|' + arrows
          else:
            arrows = template if template and len(template.split('|')) == lanes else 'left|' + arrows
        elif carried and not carries and arrows:
          arrows = '|'.join(arrows.split('|')[1:]) or None
        if arrows and len(arrows.split('|')) == lanes:
          tags[f'turn:lanes:{key}'] = arrows
        else:
          tags.pop(f'turn:lanes:{key}', None)
        fb[0 if forward else 1] = carries
        if carries:
          # (a lane full width from where it opens, start == end: full width at its first node too)
          share = [1.0 if end - start < 1e-6 else min(max((p - start) / (end - start), 0.0), 1.0) for p in (p0, p1)]
          if min(share) < 1.0 - 1e-3:
            widen.setdefault(pid, {})[key] = tuple(share)
      if len(ts) > 2 or pid in widen or piece[3:5] != row[3:5]:
        arrows_at[pid] = tags
        bays[pid] = tuple(fb)
      out.append(piece)
  return out, parent, widen, bays, applied


def remap_restrictions(restrictions, ends_of, pieces):
  """Restrictions on ways split into pieces (`pieces`: way id -> its pieces' ids, first node to last) on to the pieces
  they run along: the from way's piece at the via, the to way's, and every piece of a via way. `ends_of`: way id ->
  (first node, last node), pieces included."""
  def touching(wid, nodes):
    return next((p for p in pieces[wid] if set(ends_of[p]) & nodes), pieces[wid][0])

  def along(wid, from_nodes):  # a via way's pieces in the order travelled, from the node it is entered at
    ps = pieces[wid]
    return ps if ends_of[ps[0]][0] in from_nodes else ps[::-1]

  out = []
  for kind, wf, via, wt in restrictions:
    if wf not in pieces and wt not in pieces and not any(t == 'w' and ref in pieces for t, ref in via):
      out.append((kind, wf, via, wt))
      continue
    vias = [ref for t, ref in via if t == 'w']
    if not vias:
      n = {via[0][1]}
      out.append((kind, touching(wf, n) if wf in pieces else wf, via, touching(wt, n) if wt in pieces else wt))
      continue
    shared_in = set(ends_of[pieces[wf][0] if wf in pieces else wf]) | set(ends_of[pieces[wf][-1] if wf in pieces else wf])
    new_via, at = [], shared_in
    for v in vias:
      ps = along(v, at) if v in pieces else [v]
      new_via += ps
      at = {ends_of[ps[-1]][1], ends_of[ps[-1]][0]}
    fn = set(ends_of[new_via[0]])
    tn = set(ends_of[new_via[-1]])
    out.append((kind, touching(wf, fn) if wf in pieces else wf, [('w', v) for v in new_via], touching(wt, tn) if wt in pieces else wt))
  return out


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


def road_on(turns):
  """Which of a junction's ways out (deg turned, left positive) go on through it: those turning up to TURN_FLAG, else
  the nearest straight up to THROUGH_SKEW, a skewed junction's road on, which GTA paints through and its turn flags
  leave open."""
  on = {k for k, t in enumerate(turns) if abs(t) <= TURN_FLAG}
  if not on and turns and abs(turns[k := min(range(len(turns)), key=lambda k: abs(turns[k]))]) <= THROUGH_SKEW:
    on = {k}
  return on


def turn_restrictions(nodes, ways, toward, flags):
  """GTA's no left / no right turn flags, and its left turn only lanes, as OSM restrictions at the junction ahead of
  the node: from the way into the junction's node (from the node, along the ways to it, where lanes join on the way),
  through the junction's own nodes, to each way out turning that way by more than TURN_FLAG but the road on (road_on);
  and at the node itself.
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
      on = road_on([turn for turn, *_ in exits])
      kinds = [kind_of(0.0 if k in on else turn, *flags[i]) for k, (turn, *_) in enumerate(exits)]
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
    with more lanes one way (`source:width=survey`). A surveyed one-way link's painted lanes are centred on it, or
    placed by `placement` where it is one of several links side by side making up a carriageway.
  `lf` is the link's flags and `freeway` whether its nodes are a freeway's (a one-way freeway's lanes are wider);
  returns the tags."""
  w, offset = layout(lf, back, freeway)
  steps = ((lf[1] >> 4) & 7) * (-1 if lf[1] & 128 else 1)
  bf, bb = (int(v) for v in bays)
  if painted and not back:  # centred on the link, or placed in its carriageway
    tags = {'lanes': str(fwd), 'oneway': 'yes', 'width': metres(sum(painted['lanes'])),
            'width:lanes': '|'.join(map(metres, painted['lanes'])), 'source:width': 'survey'}
    if 'placement' in painted:
      tags['placement'] = painted['placement']
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
            'width:lanes:backward': '|'.join(map(metres, wb)), 'source:width': 'survey'}
    if (divider := painted.get('divider', 'double_solid_line' if painted['median'] < EDGE_GAP else None)):
      tags['divider'] = divider  # a median's edges are their own (median_edges)
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
    if bf or bb:  # the line between a bay filling the median and the oncoming lanes; a median's edges are their own
      tags['divider'] = 'double_solid_line'
  if bf and bb:
    tags['placement:forward'] = tags['placement:backward'] = 'left_of:1'
  elif bf or bb:
    tags['placement:forward' if bf else 'placement:backward'] = 'middle_of:1'
  elif fwd != back:
    tags['placement:forward'] = tags['placement:backward'] = 'left_of:1'
  return tags


# minor roads the game may leave unpainted (not unclassified: Blaine's country roads, where the files may miss paint)
UNPAINTED_CLASSES = {'residential', 'service', 'track'}


def painted_lines(tags, cls, two_way, samples, why, unmarked=False):
  """A way's tags with the game files' lane lines where its lanes are the class layout's (paint_survey.lane_lines,
  .unpainted): `lane_markings=no` (and no divider) on a minor road they show unpainted, else `change:lanes`
  (`:forward` / `:backward`) where a painted line between its lanes is solid (or solid on one side). Major roads keep
  their lines where the files show none: those are more likely gaps in the files (the Great Ocean Hwy, some freeways)
  than unpainted. `unmarked`: the road is unpainted here (unmarked_roads), the way too short to show it itself."""
  road = WayLanes.from_tags(tags)
  if len(road.lanes) < 2:  # no lines to draw
    return tags
  if unmarked or (cls in UNPAINTED_CLASSES and paint_survey.unpainted(samples)):
    why['unpainted'] += 1
    return {**{k: v for k, v in tags.items() if not k.startswith('divider')}, 'lane_markings': 'no'}
  out = dict(tags)
  if not any(k.startswith('change:lanes') for k in tags):
    for key, change in paint_survey.lane_lines(samples, [(s.left, s.right, s.heading) for s in road.section()]).items():
      out[f'change:lanes:{key}' if two_way else 'change:lanes'] = '|'.join(change)
      why['lines not to cross'] += 1
  return out


def neighbours_paint(nodes, rows, painted, unsurveyed, unread=None):
  """The painted cross-sections (paint_survey.correct's) for two-way ways the game files have no sections of (`unsurveyed`
  way ids: as GTA's short links in and next to junctions, whose sections the survey leaves out), from the way the road
  runs on to at either end (within STRAIGHT) with the same lane counts each way and a painted cross-section, the longer
  one where both have; and for those and the ways whose few sections the survey couldn't read (`unread`), where the
  road is painted alike at both ends, with its counts (a link between two the paint recounts): so a road's lines run on
  at the paint's place and kind rather than GTA's class layout (a median where the game paints a double line). `rows`
  are [(way id, a, b, fwd, back)]; returns {way id: cross-section}."""
  unread = unsurveyed | (unread or set())
  at = defaultdict(list)
  for r in rows:
    if r[4]:
      at[r[1]].append(r)
      at[r[2]].append(r)

  def heading(p, q):
    return game_heading(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y'])

  def length(r):
    return math.hypot(nodes[r[2]]['x'] - nodes[r[1]]['x'], nodes[r[2]]['y'] - nodes[r[1]]['y'])
  out = {}
  for r in rows:
    wid, a, b, fwd, back = r
    if wid not in unread or not back:
      continue
    found = []
    for n, far in ((a, b), (b, a)):
      h = heading(far, n)  # along the way into n
      for o in at[n]:
        if o is r or not painted.get(o[0]):
          continue
        beyond = o[2] if o[1] == n else o[1]
        if abs(wrap(heading(n, beyond) - h)) >= STRAIGHT:
          continue
        same = (o[1] == n) == (n == b)  # o runs the way r does
        got = dict(painted[o[0]])
        if not same:  # seen the other way: the directions swap, and the sides
          got['forward'], got['backward'] = got['backward'], got['forward']
          for k in ('change:forward', 'change:backward'):
            got.pop(k, None)
          if 'parking' in got:
            got['parking'] = got['parking'][::-1]
        found.append((length(o), n, got))
    counts = [f for f in found if (len(f[2]['forward']), len(f[2]['backward'])) == (fwd, back)] if wid in unsurveyed else []
    # or the road painted alike at both ends (a link between two the paint recounts, or whose own few sections the
    # survey couldn't read)
    other = {n: (len(got['forward']), len(got['backward'])) for _, n, got in found}
    if not counts and len(other) == 2 and len(set(other.values())) == 1:
      counts = found
    if counts:
      out[wid] = max(counts, key=lambda f: f[0])[2]
  return out


UNMARKED_REACH = 100.0  # m along a road its unpainted stretches are looked for either side of a way too short to tell


def unmarked_roads(nodes, rows, samples_of):
  """The minor roads to leave unpainted (`lane_markings=no`): the two-way ways the game files show unpainted
  (paint_survey.unpainted), and the ways along the road between them too short to show it themselves (fewer sections,
  or none, all of them bare), where the road runs on unpainted to both sides (or ends) within UNMARKED_REACH, so a
  road's lines don't stop and start. Minor roads are UNPAINTED_CLASSES, and the unclassified roads GTA's traffic
  doesn't use (switched off: back roads the minimap draws) or marks off-road, unpainted there even where the files draw
  no asphalt edges (a road blended into the ground), as are tracks and other off-road ways (dirt, with no asphalt to
  edge). `rows` are [(way id, a, b, two-way, class, switched off, off-road)], `samples_of(way id, a, b)` the way's
  samples; returns the way ids."""
  minor = {r[0]: r for r in rows if r[3] and (r[4] in UNPAINTED_CLASSES or (r[4] == 'unclassified' and (r[5] or r[6])))}
  at = defaultdict(list)
  for r in minor.values():
    at[r[1]].append(r)
    at[r[2]].append(r)
  state = {}
  for wid, a, b, *_ in minor.values():
    cls, offroad = minor[wid][4], minor[wid][6]
    samples, edges = samples_of(wid, a, b), cls in UNPAINTED_CLASSES and cls != 'track' and not offroad
    state[wid] = 'bare' if paint_survey.unpainted(samples, edges_seen=edges) else \
      'short' if not samples or paint_survey.unpainted(samples, 1, edges) else 'painted'

  def heading(p, q):
    return game_heading(nodes[q]['x'] - nodes[p]['x'], nodes[q]['y'] - nodes[p]['y'])

  def on(r, n):  # the way the road runs on to from way r past its node n, and that way's far node
    h = heading(r[2] if n == r[1] else r[1], n)
    nxt = [(abs(wrap(heading(n, s[2] if n == s[1] else s[1]) - h)), s) for s in at[n] if s is not r]
    turn, s = min(nxt, key=lambda v: v[0], default=(None, None))
    return (s, s[2] if n == s[1] else s[1]) if s is not None and turn < STRAIGHT else (None, None)

  def length(r):
    return math.hypot(nodes[r[2]]['x'] - nodes[r[1]]['x'], nodes[r[2]]['y'] - nodes[r[1]]['y'])

  def reach(r, n):  # what the road is past way r's node n: 'bare', 'painted' or 'end'
    dist = 0.0
    while dist < UNMARKED_REACH:
      r, n = on(r, n)
      if r is None:
        return 'end'
      if state[r[0]] != 'short':
        return state[r[0]]
      dist += length(r)
    return 'painted'
  out = {wid for wid, v in state.items() if v == 'bare'}
  for wid, v in state.items():
    if v == 'short':
      r = minor[wid]
      sides = {reach(r, r[1]), reach(r, r[2])}
      if 'bare' in sides and sides <= {'bare', 'end'}:
        out.add(wid)
  return out


def outer_changes(tags, samples, why):
  """A one-way way's tags with change:lanes on its outer lanes' outer sides from the game files' white lines there
  (paint_survey.outer_lines): the lines between it and the links GTA lays beside it, on a freeway."""
  road = WayLanes.from_tags(tags)
  left, right = paint_survey.outer_lines(samples, road.edges(FORWARD))
  if left is None and right is None:
    return tags
  ways = {v: k for k, v in paint_survey.CHANGE.items()}
  change = [list(ways[v]) for v in tags.get('change:lanes', '|'.join(['yes'] * len(road.lanes))).split('|')]
  if len(change) != len(road.lanes):
    return tags
  if left is not None:
    change[0][0] = left
  if right is not None:
    change[-1][1] = right
  values = [paint_survey.CHANGE[tuple(c)] for c in change]
  out = {k: v for k, v in tags.items() if k != 'change:lanes'}
  if set(values) != {'yes'}:
    out['change:lanes'] = '|'.join(values)
    why['outer lines not to cross'] += 1
  return out


EDGE_LANE_MAX = 7.6  # m: an outer lane widened to the painted edge line is no wider (GTA paints lanes up to 7.5 m)
EDGE_SYMMETRY = 0.6  # m the two edge lines' distances from the way's line may differ for the lanes to stay centred on it
EDGE_ON = 0.3  # m between the lanes' edge and a painted edge line that is it
ASPHALT_SLACK = 0.2  # m the lanes may reach past the asphalt's edge
SHOULDER_MIN, SHOULDER_MAX = 0.3, 5.5  # m of road surface between the lanes' edge and the asphalt's: a shoulder


def road_edges(tags, samples, why):
  """A way's tags with its lanes running out to the edge lines the game files paint and its kerbs where their asphalt
  ends (paint_survey.road_edges): GTA paints many roads' edges as a solid line with a shoulder of road beyond it (the
  Great Ocean Hwy's 4 m), and the class layout's (or the survey's) kerbs fell inside the asphalt, on or short of the
  line. The outer lanes widen (or narrow) to the edge lines, leaving every other line where it was: each to its own
  where placement keeps the line on the lanes in between, else alike (the line the middle of the lanes) to both where
  they're about as far either side of it, or to the one painted where the other side's asphalt has room. The road
  surface between the lanes' edge and the asphalt's, with nothing else painted on it, is a shoulder
  (`shoulder:<side>:width`), its edge line painted where the lanes run to one (else `shoulder:<side>:markings=no`), in
  the files' colour (`shoulder:<side>:markings=white|yellow` where it isn't osm_lanes' default). Not on single tracks,
  centre turn lanes, tapers, parking lanes or lanes with room between them and their kerbs."""
  if any(k.endswith((':start', ':end')) or k.startswith(('parking', 'shoulder')) for k in tags):
    return tags
  road = WayLanes.from_tags(tags)
  sec = road.section(FORWARD)
  if not sec or road.margin > 0.01 or any(s.heading == 0 for s in sec):
    return tags
  one_way = all(s.heading == 1 for s in sec)
  edges = [sec[0].left, sec[-1].right]
  found = paint_survey.road_edges(samples, tuple(edges), (sec[0].right, sec[-1].left))
  lines, colours, asphalt, clear = ([f[k] for f in found] for k in range(4))
  sign = (-1.0, 1.0)
  out = dict(tags)

  def laid_out(to):  # the tags with the outer lanes out to `to` (left, right), if that leaves every other line in place
    move = [(edges[0] - to[0]), (to[1] - edges[1])]
    widths = [s.right - s.left for s in sec]
    widths[0] += move[0]
    widths[-1] += move[1]
    if not all(paint_survey.LANE_MIN <= w <= EDGE_LANE_MAX for w in (widths[0], widths[-1])) or \
        any(a is not None and (a - t) * s < -ASPHALT_SLACK for a, t, s in zip(asphalt, to, sign, strict=True)):
      return None
    new = dict(tags, width=metres(road.width + sum(move)))
    if one_way:
      new['width:lanes'] = '|'.join(metres(w) for w in widths)
    else:
      new.pop('width:lanes', None)
      new['width:lanes:forward'] = '|'.join(metres(w) for w, s in zip(widths, sec, strict=True) if s.heading == 1)
      new['width:lanes:backward'] = '|'.join([metres(w) for w, s in zip(widths, sec, strict=True) if s.heading == -1][::-1])
    got = WayLanes.from_tags(new).section(FORWARD)
    before = [v for s in sec for v in (s.left, s.right)][1:-1]
    after = [v for s in got for v in (s.left, s.right)][1:-1]
    moved = [abs(got[0].left - to[0]), abs(got[-1].right - to[1]), *(abs(a - b) for a, b in zip(before, after, strict=False))]
    return new if len(got) == len(sec) and max(moved) <= 0.02 else None
  tries = []
  if any(v is not None for v in lines):
    tries.append([v if v is not None else e for v, e in zip(lines, edges, strict=True)])  # each to its own
    if None not in lines and abs(lines[0] + lines[1]) <= EDGE_SYMMETRY:
      tries.append([(lines[0] - lines[1]) / 2, (lines[1] - lines[0]) / 2])
    elif None in lines:
      half = next(abs(v) for v in lines if v is not None)
      tries.append([-half, half])
  for to in tries:
    if max(abs(t - e) for t, e in zip(to, edges, strict=True)) <= 0.05:
      break
    if (new := laid_out(to)) is not None:
      out, edges = new, to
      why['lanes out to the painted edge lines'] += 1
      break
  shoulders, base = [0.0, 0.0], WayLanes.from_tags(out).width
  for i, side in enumerate(('left', 'right')):
    if asphalt[i] is not None and clear[i] and SHOULDER_MIN <= (wide := (asphalt[i] - edges[i]) * sign[i]) <= SHOULDER_MAX:
      shoulders[i] = wide
      out[f'shoulder:{side}:width'] = metres(wide)
      default = 'yellow' if one_way and i == 0 else 'white'
      if lines[i] is None or abs(lines[i] - edges[i]) > EDGE_ON:
        out[f'shoulder:{side}:markings'] = 'no'
      elif colours[i] != default:
        out[f'shoulder:{side}:markings'] = colours[i]
      why['shoulders' + (' (no edge line)' if out.get(f'shoulder:{side}:markings') == 'no' else '')] += 1
  if any(shoulders):
    out['shoulder'] = 'both' if all(shoulders) else 'left' if shoulders[0] else 'right'
    out['width'] = metres(base + sum(shoulders))
  return out


def arrows(n, out):
  """The turn arrows of n lanes into a junction from its ways out, `out` {move: lanes out that way (left, through,
  right)}: the lanes taken in order from the left, each move as many as it has lanes out to go on in. Through takes
  as many lanes as it has out (up to n), each turn one; lanes left over turn, left before right, while a turn has lanes
  out to spare, else go through (with no way through, turn to the side with more lanes out). Lanes too few: the right
  turn shares the right through lane, then the left turn the left one. So 3 lanes into 2 straight on: left | through
  | through;right; into 1: left | through | right. Every lane the same way at a forced turn."""
  order = [k for k in ('left', 'through', 'right') if k in out]
  if n == 1 or len(order) == 1:
    return [';'.join(order)] * n
  count = {k: min(out[k], n) if k == 'through' else 1 for k in order}
  spare = n - sum(count.values())
  while spare > 0:
    k = next((k for k in ('left', 'right') if k in out and count[k] < out[k]), None)
    if k is None:
      k = 'through' if 'through' in out else max(('left', 'right'), key=lambda k: (out[k], k == 'left'))
    count[k] += 1
    spare -= 1
  short = -spare
  share_right = min(short, 1) if 'right' in out and 'through' in out else 0
  share_left = short - share_right
  lanes = [set() for _ in range(n)]
  start = 0
  for k, shared in (('left', share_left), ('through', share_right), ('right', 0)):
    if k not in out:
      continue
    for i in range(start, start + count[k]):
      lanes[i].add(k)
    start += count[k] - (shared if k != 'right' else 0)
  return [';'.join(t for t in ('left', 'through', 'right') if t in kinds) for kinds in lanes]


def with_paint(lanes, seen, allowed):
  """Lanes' turn arrows (`lanes`, turn:lanes values) with those painted on them (`seen`, {lane: value}) where the
  junction has a way out they point along (`allowed`), less any move that isn't. A lane without paint keeps those of
  its own that don't cross a painted lane's (none further left than a painted lane on its left, none further right than
  one on its right), else through or the junction's other moves between theirs."""
  order = {'left': 0, 'through': 1, 'right': 2}
  painted = [set(seen[k].split(';')) & allowed if k in seen else set() for k in range(len(lanes))]
  out = list(painted)
  for k, lane in enumerate(lanes):
    if out[k]:
      continue
    lo = max((order[m] for i in range(k) for m in painted[i]), default=0)
    hi = min((order[m] for i in range(k + 1, len(lanes)) for m in painted[i]), default=2)
    own, between = set(lane.split(';')), {m for m in allowed if lo <= order[m] <= hi}
    out[k] = {m for m in own if lo <= order[m] <= hi} or between & {'through'} or between or own
  return [';'.join(t for t in ('left', 'through', 'right') if t in kinds) for kinds in out]


class PaintedArrows:
  """The game files' painted turn arrows (paint_survey.arrow_marks) by where they are."""
  CELL = 25.0  # m

  def __init__(self, marks):
    self.grid = defaultdict(list)
    for m in marks:
      self.grid[(int(m[0] // self.CELL), int(m[1] // self.CELL))].append(m)

  def on(self, a, b, spans):
    """The arrows painted in the lanes of a link from node a to node b pointing along it, `spans` its lanes' (left,
    right) m right of it, left to right: [(lane, m before b, turn:lanes value)]."""
    dx, dy = b['x'] - a['x'], b['y'] - a['y']
    length = math.hypot(dx, dy)
    if length < 1e-6 or not spans:
      return []
    ux, uy, h = dx / length, dy / length, game_heading(dx, dy)
    steps = int(length // self.CELL) + 1
    cells = {(int((a['x'] + dx * t / steps) // self.CELL) + i, int((a['y'] + dy * t / steps) // self.CELL) + k)
             for t in range(steps + 1) for i in (-1, 0, 1) for k in (-1, 0, 1)}
    out = []
    for c in cells:
      for x, y, z, heading, kind in self.grid.get(c, ()):
        s, right = (x - a['x']) * ux + (y - a['y']) * uy, (x - a['x']) * uy - (y - a['y']) * ux
        if not 0.0 <= s < length or abs(wrap(heading - h)) > ARROW_ALIGN or \
           abs(z - (a.get('z', z) + (b.get('z', z) - a.get('z', z)) * s / length)) > ARROW_DZ:
          continue
        lane = min(range(len(spans)), key=lambda k: abs(right - (spans[k][0] + spans[k][1]) / 2))
        if spans[lane][0] - ARROW_SPILL <= right <= spans[lane][1] + ARROW_SPILL:
          out.append((lane, length - s, kind))
    return out


def lane_turns(nodes, ways, lanes_to, junction, toward, left_only, restrictions, left_bays=frozenset(), medians=frozenset(),
               painted=None, spans=None):
  """turn:lanes on the lanes into GTA's junctions where roads cross (see arrows), from the ways out of each, a move
  turning more than TURN_FLAG but the road on (road_on) being a left or right turn, as for GTA's turn flags, less
  those the restrictions forbid.
  Marked on every way along the approach from APPROACH m before the junction, or from the furthest arrow painted
  there, while the road runs on with the same lanes (Valhalla reads them from the way into the junction). None where every lane only goes through, or where the
  road bends into the junction (APPROACH_BEND), which leaves which way is through moot. A one-lane approach gets none,
  as real mappers leave them out, unless it's GTA's left turn only lane or its painted arrow says every move it has;
  on a wider road that lane is the left one. A
  turn bay folded into its road (`left_bays`, its road's links) is its left lane, marked from where it opens, also
  where the road gains lanes on the way (`left|through` before that). On a two-way road with a median (`medians`, its links (node, next node) where the median has room for a lane)
  running in to a junction it may turn left at, GTA paints the median as a left-turn lane without a link of its own
  (measured on 4 of 4 such approaches; not where the game files paint the median on into the junction, left out of
  `medians`): one more lane, `left`, on the approach's links within the median, if they are at least MEDIAN_LANE_MIN
  long. Each lane takes the arrow painted on it up to ARROW_REACH m before the junction
  (`painted`: PaintedArrows, found in the lanes `spans` gives each link (node, next node): their (left, right) m right
  of it) where the junction has a way out it points along (with_paint), a skewed road on painted through or as a
  turn that way. `ways` is
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

  def painted_on(chain, n):
    """The arrow painted in each lane of an approach (its links back from the junction, `chain`, and the road's on
    back with its n lanes), {lane: turn:lanes value}: the one most seen there up to ARROW_REACH m before the junction;
    and the links from the junction back to the furthest painted."""
    far, total, (q, nxt) = list(chain), sum(length(*e) for e in chain), chain[-1]
    while total < ARROW_REACH and not junction(q) and len(prev := [r for r in into[q] if r != nxt]) == 1 and \
          set(out[q]) - {prev[0]} == {nxt} and lanes_to[(prev[0], q)] == n:
      q, nxt = prev[0], q
      far.append((q, nxt))
      total += length(q, nxt)
    votes, before, last = defaultdict(Counter), 0.0, 0
    for k, e in enumerate(far):
      if lanes_to[e] == n and len(spans.get(e, ())) == n:
        for lane, m, kind in painted.on(nodes[e[0]], nodes[e[1]], spans[e]):
          if m + before <= ARROW_REACH:
            votes[lane][kind] += 1
            last = k
      before += length(*e)
    return {lane: v.most_common(1)[0][0] for lane, v in votes.items()}, far[:last + 1]

  only_left ={(path[-2], path[-1]) for i in left_only for path in toward.get(i, {}).values()} | set(left_bays)
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
      if junction(p) or (n < 2 and not only and not median and painted is None):
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
        bent += n > 1 or only or median
        continue
      lead, back = [way_of[e] for e in chain[::-1]], 0.0  # and the road further back, where restrictions can start
      while back < RESTRICTION_REACH and len(prev := [r for r in into[q] if r != nxt]) == 1:
        q, nxt = prev[0], q
        lead.insert(0, way_of[(q, nxt)])
        back += length(q, nxt)
      found = junction_exits(nodes, j, h_in, h_in, {k for e in chain for k in e}, out, way_of)
      on = road_on([t for t, *_ in found])
      kept = [k for k, (_, inner, w, _) in enumerate(found) if not forbidden(lead + inner + [w])]
      exits = [('through' if k in on else 'left' if found[k][0] > 0 else 'right', lanes_to[found[k][3]]) for k in kept]
      allowed = {k for k, _ in exits}
      if not allowed:
        continue
      # GTA paints a skewed road on as through or as a turn that way
      paintable = allowed | {'left' if found[k][0] > 0 else 'right' for k in kept if k in on and abs(found[k][0]) > TURN_FLAG}
      lanes_out = {k: sum(m for kk, m in exits if kk == k) for k in allowed}
      but_left = {k: m for k, m in lanes_out.items() if k != 'left'}
      lanes = arrows(n, lanes_out)
      if only and n > 1 and 'left' in allowed and allowed - {'left'}:
        lanes = ['left', *arrows(n - 1, but_left)]
      seen, reach = painted_on(chain, n) if painted is not None else ({}, chain)
      inside = next((k for k, e in enumerate(chain) if e not in medians), len(chain))  # the median runs in to the junction
      shift = 0
      if median and 'left' in allowed and allowed - {'left'} and sum(length(*e) for e in chain[:inside]) >= MEDIAN_LANE_MIN:
        chain = chain[:inside]
        lanes, shift = ['left', *arrows(n, but_left)], 1
        opened.update(chain)
      if painted is not None:
        seen = {lane + shift: kind for lane, kind in seen.items()}
        lanes = with_paint(lanes, seen, paintable)
        if n < 2 and not only and not shift and (not any(set(k.split(';')) & paintable for k in seen.values()) or
                                                 set(lanes[0].split(';')) != allowed):
          continue  # a one-lane approach: only where its paint says all it has
        if not shift and any(set(k.split(';')) & paintable for k in seen.values()):
          chain += reach[len(chain):]  # and back to where they're painted
      elif n < 2 and not only and not shift:
        continue
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


def painted_median_lanes(nodes, ways, lanes_to, medians, painted, spans, junction):
  """Medians the game paints a left-turn arrow in: a lane GTA has no link for, as where a hatched median ends and the
  lane opens in its room (Vinewood Blvd before Meteor St), or a median the class layout puts where the paint has the
  lane next to the centre line. `medians` are links (node, next node) with room for a lane in their median (BAY_MIN),
  `spans` each link's lanes (left, right) m right of it, `painted` PaintedArrows. A link with a painted arrow turning
  left wholly inside its median (between the two directions' inner lane edges) and the links on from it towards the
  road's next junction, while the road runs on alone with its median and lanes: [chains of links, along the road]."""
  _, out, into = graph(ways)
  chains, used = [], set()
  for e in sorted(medians):
    p, q = e
    ours, theirs = spans.get(e), spans.get((q, p))
    if e in used or not ours or not theirs:
      continue
    lo, hi = -theirs[0][0], ours[0][0]  # the median, m right of the link
    if hi - lo < BAY_MIN or not any('left' in kind for _, _, kind in
                                    painted.on(nodes[p], nodes[q], [(lo + ARROW_SPILL, hi - ARROW_SPILL)])):
      continue
    chain = [e]
    while not junction(chain[-1][1]):
      p, q = chain[-1]
      nxt = [r for r in out[q] if r != p]
      if len(nxt) != 1 or len([r for r in into[q] if r != nxt[0]]) != 1 or (q, nxt[0]) not in medians or \
         lanes_to.get((q, nxt[0])) != lanes_to[e] or (q, nxt[0]) in chain:
        break
      chain.append((q, nxt[0]))
    used.update(chain)
    chains.append(chain)
  return chains


def node_id(k):
  return k[0] * 65536 + k[1] + 1


def painted_stop_lines(path, lines_path, features_path, nodes, info):
  """The map at `path` written again with its stop lines where the game files paint them (stop_paint.py); returns the
  rows and nodes used with the ways split for them."""
  from openpilot.tools.sim.bridge.gta5.map import stop_paint
  from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
  placed, counts = stop_paint.stop_lines(path, lines_path, features_path, to_game)
  synthetic = [max((k[1] for k in nodes if k[0] == TAPER_NODE_AREA), default=0)]

  def new_node_id():
    synthetic[0] += 1
    return node_id((TAPER_NODE_AREA, synthetic[0]))
  new_nodes, pieces = stop_paint.rewrite(path, placed, new_node_id, to_lat_lon, remap_restrictions)
  key_of = {node_id(k): k for k in nodes}
  for nid, (x, y, z) in new_nodes.items():
    k = ((nid - 1) // 65536, (nid - 1) % 65536)
    nodes[k] = {'a': k[0], 'i': k[1], 'x': x, 'y': y, 'z': z, 'f': [0, 0, 0, 0, 0], 'st': 0, 'sp': 1}
    key_of[nid] = k
  out = []
  for row in info:
    if row[0] not in pieces:
      out.append(row)
      continue
    out += [[pid, key_of[a], key_of[b], *row[3:]] for pid, a, b in pieces[row[0]]]
  out.sort(key=lambda r: r[0])
  told = ', '.join(f'{n} {k}' for k, n in sorted(counts.items(), key=lambda c: -c[1]))
  print(f"stop lines from the game files' paint: {told}; {len(new_nodes)} nodes added, {len(pieces)} ways split")
  return out, sorted({k for _, a, b, *_ in out for k in (a, b)})


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('dump', help="ynddump's paths.jsonl")
  p.add_argument('out', help='.osm or .osm.pbf')
  p.add_argument('--sidecar', help="the roads as arrays for the bridge's map matching (.npz)")
  p.add_argument('--minimap', help="ynddump's minimap.jsonl: minor links GTA's minimap draws as roads become roads")
  p.add_argument('--survey', nargs='*', default=[], help="surveys of the game's road paint (paint_survey.py): the lane "
                 "widths where they were measured")
  p.add_argument('--survey-lines', help="the game files' paint as polylines (polylines.jsonl): each line's kind by its "
                 "whole length, for the survey's sections, the stop lines where painted, and roads painted across junctions")
  p.add_argument('--survey-features', help="the game files' painted features (features.jsonl): crossings, for the stop "
                 "lines added where the map has none, and turn arrows, for the lanes' turn:lanes")
  args = p.parse_args()

  nodes, links, streets = load(args.dump)
  es = edges(nodes, links)
  cross = crossovers(nodes, es)
  es = {k: v for k, v in es.items() if k not in cross}
  changes = lane_changes(nodes, es)
  for k, ab in changes.items():
    fwd, back, lf = es[k]
    es[k] = (max(fwd, back), 0, lf) if ab else (0, max(fwd, back), lf)
  used = sorted({k for e in es for k in e})
  print(f"{len(used)} nodes, {len(es)} ways ({len(cross)} carriageway crossovers dropped, {len(changes)} lane changes one-way)")

  info = []  # (way id, a, b, fwd, back, class, limit, name, link flags)
  for i, ((a, b), (fwd, back, lf)) in enumerate(sorted(es.items())):
    # a lane change keeps its two-way link's class: as a freeway's, routers announce each one taken as a fork
    cls = highway(nodes, a, b, 1, 1) if (a, b) in changes else None
    if not fwd:  # one-way the other way: draw it in its direction of travel
      a, b, fwd, back = b, a, back, fwd
    cls = cls or highway(nodes, a, b, fwd, back)
    st = nodes[a]['st'] if nodes[a]['st'] == nodes[b]['st'] else 0
    info.append([i + 1, a, b, fwd, back, cls, speed_limit(nodes, a, b, cls, fwd, back), streets.get(st), lf])
  parted = split_shared(nodes, info)
  print(f"{parted} nodes with both carriageways of a divided road through them parted, a link across the median between")
  print(f"{dead_end_lanes(info)} two-way links one-way as the one-way link they run on from (a direction going nowhere)")
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
  restrictions, freed, earlier = traps.free_traps(nodes, ways, restrictions, MAX_VIA)
  print(f"{len(freed)} restrictions that cut road off left out (" +
        ', '.join(f'{n} {k}' for k, n in Counter(r[0] for r in freed).most_common()) +
        f"), {len(earlier)} from the ways on to a trap instead")
  medians = set()  # two-way links (node, next node) with room for a turn lane in their median, and no turn bay that way
  for wid, a, b, fwd, back, *_, lf in info:
    if back and 2 * layout(lf, back)[1] >= BAY_MIN:
      medians.update(e for e in ((a, b), (b, a)) if e[1] not in bay_to[wid])
  lines = paint_survey.line_kinds(args.survey_lines) if args.survey_lines else None
  survey = paint_survey.load(args.survey, lines) if args.survey else {}
  tracks = 0
  for row in info:  # GTA's single tracks the game files paint a centre line on are two-lane roads (the port's grid)
    lf = row[8]
    steps_7 = lf[1] & 0xF0 == 0xF0  # offset -7: both directions on one lane on the line
    if row[3] == row[4] == 1 and steps_7 and paint_survey.painted_centre(paint_survey.along(survey, row[1], row[2]) or []):
      row[8] = [lf[0], lf[1] & 0x0F, *lf[2:]]
      tracks += 1
  if survey:
    print(f"{tracks} single tracks with a painted centre line as two-lane roads")
  row_of = {row[0]: row for row in info}
  painted, why, left_out, centre_kinds, recounted, median_kinds = {}, Counter(), 0, {}, {}, {}
  # where the game files' paint covers the map, the camera's survey only checks it
  from_files = any(d.get('src') == paint_survey.GAMEFILES for ds in survey.values() for d in ds)
  for wid, a, b, fwd, back, *_, lf in info:
    if (samples := paint_survey.along(survey, a, b)) is None:
      continue
    w, offset = layout(lf, back, bool(nodes[a]['f'][2] & nodes[b]['f'][2] & FREEWAY) and fwd >= 2)
    if (back and offset < 0) or (not back and offset) or bay_to[wid]:  # bays fill medians: not the road's paint there
      why['turn bay' if bay_to[wid] else 'lanes overlap' if back else 'one-way, off-centre'] += 1
      left_out += 1
      continue
    if not back:
      painted[wid], reason = paint_survey.correct_oneway(samples, fwd, (-fwd * w / 2, fwd * w / 2))
      if not painted[wid] and reason != 'one-way, no game files':  # one of several links side by side, as on freeways
        painted[wid], reason = paint_survey.correct_carriageway(samples, fwd)
        reason = reason or 'measured one-way in its carriageway (game files)'
      why[reason or 'measured one-way (game files)'] += 1
      continue
    use, other = paint_survey.sources(samples, camera_corrects=not from_files)
    if not use:
      why['no game files (camera a check only)'] += 1
      left_out += 1
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
        recounted[wid] = (row[3], row[4])
        row[3], row[4] = len(got['forward']), len(got['backward'])
        lanes_to[(a, b)], lanes_to[(b, a)] = row[3], row[4]
    why[reason or ('measured (game files)' if files else 'measured (camera)')] += 1
    if (median := got['median'] if got and got.get('median', 0.0) > EDGE_GAP else 2 * offset) > EDGE_GAP:
      median_kinds[wid] = paint_survey.median_edges(samples, median)  # the painted median's, else GTA's
    if painted[wid] and len(other) >= paint_survey.MIN_SAMPLES and (check := paint_survey.correct(other, fwd, back, kerbs)[0]):
      why['sources disagree' if paint_survey.disagree(painted[wid], check) else 'sources agree'] += 1
    if not painted[wid] and offset <= 0 and (kind := paint_survey.centre_kind(samples, 0.0, paint_survey.CENTRE_TOL)):
      centre_kinds[wid] = kind  # the centre line's kind still shows where the lanes don't add up
  print(f"{len(centre_kinds)} more two-way links' centre lines of the kind the game files paint")
  unread = {wid for wid, _, _, _, back, *_ in info if back and not bay_to[wid] and not painted.get(wid)}
  unsurveyed = {wid for wid, a, b, *_ in info if wid in unread and
                not any(d.get('src') == paint_survey.GAMEFILES for d in paint_survey.along(survey, a, b) or [])}
  inherited = neighbours_paint(nodes, [(wid, a, b, fwd, back) for wid, a, b, fwd, back, *_ in info], painted, unsurveyed,
                               unread) if from_files else {}
  for wid, got in inherited.items():
    painted[wid] = got
    row = row_of[wid]
    medians.discard((row[1], row[2]))
    medians.discard((row[2], row[1]))
    if (len(got['forward']), len(got['backward'])) != (row[3], row[4]):  # the counts the paint has either side
      recounted[wid] = (row[3], row[4])
      row[3], row[4] = len(got['forward']), len(got['backward'])
      lanes_to[(row[1], row[2])], lanes_to[(row[2], row[1])] = row[3], row[4]
  print(f"{len(inherited)} two-way links without sections of their own painted as the road they run on from")
  shut = 0
  for _, a, b, _, back, *_, lf in info:
    for p, q in ((a, b), (b, a)):
      if (p, q) in medians and junction(q) and \
         paint_survey.median_runs_in(paint_survey.along(survey, p, q) or [], layout(lf, back)[1], link_length(p, q)):
        medians.discard((p, q))  # the median painted on into the junction: no turn lane in it
        shut += 1
  print(f"{shut} links into junctions with the median painted on into them, no left-turn lane in it")
  painted_arrows, spans = None, {}  # (node, next node) -> its lanes' (left, right), m right of it, left to right
  if args.survey_features:
    painted_arrows = PaintedArrows(paint_survey.arrow_marks(args.survey_features))
    for wid, a, b, fwd, back, *_, lf in info:
      freeway = bool(nodes[a]['f'][2] & nodes[b]['f'][2] & FREEWAY) and fwd >= 2
      road = WayLanes.from_tags(lane_tags(fwd, back, lf, freeway, (b in bay_to[wid], a in bay_to[wid]), painted.get(wid)))
      if road.single_track:  # one lane both ways has no turn:lanes each way
        continue
      for e, d in (((a, b), FORWARD), ((b, a), BACKWARD)):
        spans[e] = [(s.left, s.right) for s in road.ours(d)]
  arrows_at, approaches, bent, opened = lane_turns(nodes, ways, lanes_to, junction, toward, left_lanes, restrictions,
                                                   left_bays, medians, painted_arrows, spans)
  way_of = graph(ways)[0]
  painted_medians = painted_median_lanes(nodes, ways, lanes_to, medians - opened, painted_arrows, spans, junction) \
    if painted_arrows is not None else []
  drawn_from = {wid: a for wid, a, _, _ in ways}

  def turn_key(e):
    wid = way_of[e]
    return wid, 'turn:lanes:forward' if e[0] == drawn_from[wid] else 'turn:lanes:backward'

  def left_lane_already(e):  # GTA's left lane, its arrow painted beside the median: that lane's
    wid, key = turn_key(e)
    have = arrows_at.get(wid, {}).get(key) or arrows_at.get(wid, {}).get('turn:lanes')
    return bool(have) and have.split('|')[0] == 'left' and lanes_to[e] > 1
  painted_medians = [c for c in painted_medians if not any(left_lane_already(e) for e in c)]
  for chain in painted_medians:
    for e in chain:
      wid = way_of[e]
      a, two_way = next((a, t) for w, a, _, t in ways if w == wid)
      key = 'turn:lanes' if not two_way else 'turn:lanes:forward' if e[0] == a else 'turn:lanes:backward'
      if not (have := arrows_at.get(wid, {}).get(key)):
        continue  # no arrows into where the way ends (the left turn is further on): the lane alone
      lanes = have.split('|')
      if any('left' in k.split(';') for k in lanes):  # the left turn is the new lane's now
        lanes = [';'.join(t for t in k.split(';') if t != 'left') or 'none' for k in lanes]
        arrows_at[wid][key] = '|'.join(['left', *lanes])
      else:  # no left turn where the way ends: its arrow is for one further on
        arrows_at[wid][key] = '|'.join(['none', *lanes])
    opened.update(chain)
  for p, q in opened:
    row = row_of[way_of[(p, q)]]
    row[3 if row[2] == q else 4] += 1
    bay_to[row[0]].add(q)
  print(f"{len(opened)} links into junctions with their median painted as a left-turn lane (" +
        f"{sum(len(c) for c in painted_medians)} on {len(painted_medians)} roads from a left arrow painted in it)")
  if survey:
    print(f"paint survey of {len(painted) + left_out} links: " +
          ', '.join(f'{n} {k}' for k, n in why.most_common()))
  print(f"{approaches} approaches to junctions with turn arrows, on {len(arrows_at)} ways ({bent} bending into theirs left out)")
  layer_of = {ways[i][0]: v for i, v in levels(nodes, [(a, b) for _, a, b, _ in ways]).items()}
  print(f"{len(layer_of)} ways over others as bridges (up to layer {max(layer_of.values(), default=0)})")

  yellow = paint_survey.yellow_lines(args.survey_lines) if args.survey_lines else None
  taper_why = Counter()
  tapers = lane_tapers(nodes, info, set(left_bays) | opened, junction, survey, yellow, taper_why, recounted,
                       {c[-1] for c in painted_medians}) if survey else []
  link_ends = {r[0]: (r[1], r[2]) for r in info}  # GTA's links, before any are split
  info, parent, widen, piece_bays, applied = split_tapers(nodes, info, tapers, set(left_bays) | opened, arrows_at, bay_to,
                                                               recounted)

  def link_samples(wid, a, b):  # the survey's samples on a way, a piece of a split link only those along it
    a0, b0 = link_ends[parent.get(wid, wid)]
    samples = paint_survey.along(survey, a0, b0) or []
    if (a, b) == (a0, b0):
      return samples
    pa, pb = nodes[a], nodes[b]
    dx, dy = pb['x'] - pa['x'], pb['y'] - pa['y']
    return [d for d in samples if 'x' in d and
            0.0 <= ((d['x'] - pa['x']) * dx + (d['y'] - pa['y']) * dy) / max(dx * dx + dy * dy, 1e-9) <= 1.0]
  if parent:
    for pid, wid in parent.items():  # a piece is its way's but for its lanes
      for d in (layer_of, destination):
        if wid in d:
          d[pid] = d[wid]
      if wid in drawn:
        drawn.add(pid)
      if wid in centre_kinds:
        centre_kinds[pid] = centre_kinds[wid]
    pieces = defaultdict(list)
    for wid in [r[0] for r in info]:
      pieces[parent.get(wid, wid)].append(wid)
    pieces = {wid: ps for wid, ps in pieces.items() if len(ps) > 1}
    ends_of = {r[0]: (r[1], r[2]) for r in info}
    row_of_piece = {r[0]: r for r in info}
    restrictions = remap_restrictions(restrictions, ends_of, pieces)
    for ps in pieces.values():  # no turning back at the nodes added along a two-way link, which had none
      for p in ps:
        a, b = ends_of[p]
        row = row_of_piece[p]
        for n in (a, b):
          if n[0] == TAPER_NODE_AREA and row[3] and row[4]:
            restrictions.append(('no_u_turn', p, [('n', n)], p))
    used = sorted({k for _, a, b, *_ in info for k in (a, b)})
    info.sort(key=lambda r: r[0])  # osmium readers want ways in order of their ids
  final = [(wid, a, b, bool(back)) for wid, a, b, _, back, *_ in info if wid not in crossings]
  cut = [x - y for x, y in zip(traps.cut_off(final, restrictions), traps.cut_off(final, []), strict=True)]
  if any(cut):
    raise SystemExit(f"restrictions cut road off: {len(cut[0])} directed ways trapped, {len(cut[1])} unreachable, " +
                     f"e.g. {sorted(cut[0] | cut[1])[:5]}")
  for wid in widen:
    painted.pop(wid, None)
  print(f"{len(tapers)} turn lanes opening ({', '.join(f'{n} {k}' for k, n in taper_why.most_common())}), {applied} widening " +
        f"from there: {len(parent)} links split")
  lines_why, edges_why = Counter(), Counter()
  unmarked = unmarked_roads(nodes, [(wid, a, b, bool(back), cls, bool((nodes[a]['f'][2] | nodes[b]['f'][2]) & SWITCHED_OFF),
                                     bool((nodes[a]['f'][0] | nodes[b]['f'][0]) & OFFROAD))
                                    for wid, a, b, _, back, cls, *_ in info if wid not in crossings], link_samples) if survey else set()
  w = osmium.SimpleWriter(args.out, overwrite=True)
  for k in used:
    n = nodes[k]
    lat, lon = to_lat_lon(n['x'], n['y'])
    tags = {} if k[0] == TAPER_NODE_AREA else node_tags(n, stop.get(k, 'both'))
    w.add_node(osmium.osm.mutable.Node(id=node_id(k), version=1, location=(lon, lat), tags={**tags, 'ele': f"{n['z']:.1f}"}))
  for wid, a, b, fwd, back, cls, limit, name, lf in info:
    if wid in crossings:
      w.add_way(osmium.osm.mutable.Way(id=wid, version=1, nodes=[node_id(a), node_id(b)],
                                       tags={'highway': 'footway', 'footway': 'crossing', 'crossing': 'marked'}))
      continue
    freeway = bool(nodes[a]['f'][2] & nodes[b]['f'][2] & FREEWAY) and fwd >= 2
    bays = piece_bays.get(wid, (b in bay_to[parent.get(wid, wid)], a in bay_to[parent.get(wid, wid)]))
    tags = {'highway': cls, **lane_tags(fwd, back, lf, freeway, bays, None if wid in piece_bays else painted.get(wid)),
            **arrows_at.get(wid, {}), 'maxspeed': f'{limit} mph'}
    if wid in widen or (wid in piece_bays and all(bays)):
      # the turn lanes in the median, widening from nothing where they open: each its share of the median at each end,
      # halved where both ways' are open at once
      share = {key: widen.get(wid, {}).get(key, (1.0, 1.0) if bay else (0.0, 0.0)) for key, bay in zip(KEYS, bays, strict=True)}
      scale = [1.0 / max(share['forward'][e] + share['backward'][e], 1.0) for e in (0, 1)]
      room = 2 * layout(lf, back, freeway)[1]
      fuller = max((0, 1), key=lambda e: (share['forward'][e] + share['backward'][e]) * scale[e])
      full = dict(tags)  # as lane_tags lays the bays out, at their full widths
      for key, bay in zip(KEYS, bays, strict=True):
        if not bay or f'width:lanes:{key}' not in tags:
          continue
        widths = tags[f'width:lanes:{key}'].split('|')
        at = [room * share[key][e] * scale[e] for e in (0, 1)]
        tags[f'width:lanes:{key}'] = '|'.join([metres(at[fuller]), *widths[1:]])
        if abs(at[0] - at[1]) > 1e-3:
          tags[f'width:lanes:{key}:start'] = '|'.join([metres(at[0]), *widths[1:]])
          tags[f'width:lanes:{key}:end'] = '|'.join([metres(at[1]), *widths[1:]])
      if any(k.endswith(':start') for k in tags):
        # placement by a lane that widens along the way would move the line with it: the middle of the road where the
        # bays' full widths put it there, else midway between the lanes either side of the median and its bays
        plain = {k: v for k, v in tags.items() if not k.startswith('placement')}
        if abs(WayLanes.from_tags(plain).line - WayLanes.from_tags(full).line) < 0.05:
          tags = plain
        else:
          tags = {**plain, 'placement:forward': f'left_of:{1 + int(bays[0])}', 'placement:backward': f'left_of:{1 + int(bays[1])}'}
    median = any(WayLanes.from_tags(tags, at=end).gaps for end in ('', 'start', 'end'))
    if fwd == back == 1 and 'lane_markings' not in tags and cls in MARKED and not median:
      tags.setdefault('divider', 'double_solid_line')  # how GTA paints most two-lane roads' centre (OSM reads dashed)
    if wid in centre_kinds and 'lane_markings' not in tags:
      tags['divider'] = centre_kinds[wid]
    if median and (kinds := median_kinds.get(parent.get(wid, wid))) and 'lane_markings' not in tags:
      left, right = kinds  # seen along the way: beside the backward lanes, beside the forward ones
      if left == right and left:
        tags['divider'] = left
      elif left != right:
        tags.update({k: v for k, v in (('divider:forward', right), ('divider:backward', left)) if v})
    if survey and 'lane_markings' not in tags and tags.get('source:width') != 'survey' and \
        not any(k.endswith((':start', ':end')) for k in tags) and (wid in unmarked or (samples := link_samples(wid, a, b))):
      tags = painted_lines(tags, cls, bool(back), [] if wid in unmarked else samples, lines_why, wid in unmarked)
    if survey and not back and 'lane_markings' not in tags and (samples := link_samples(wid, a, b)):
      tags = outer_changes(tags, samples, lines_why)
    if survey and (samples := link_samples(wid, a, b)):
      tags = road_edges(tags, samples, edges_why)
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
  if survey:
    print("lane lines from the game files' paint: " + ', '.join(f'{n} {k}' for k, n in lines_why.most_common()))
    print("road edges from the game files: " + ', '.join(f'{n} {k}' for k, n in edges_why.most_common()))
  for i, (kind, wi, via, wo) in enumerate(restrictions):
    via = [(t, node_id(ref) if t == 'n' else ref, 'via') for t, ref in via]
    w.add_relation(osmium.osm.mutable.Relation(id=i + 1, version=1, tags={'type': 'restriction', 'restriction': kind},
                                               members=[('w', wi, 'from'), *via, ('w', wo, 'to')]))
  w.close()
  if args.survey_lines:
    info, used = painted_stop_lines(args.out, args.survey_lines, args.survey_features, nodes, info)
    from openpilot.tools.sim.bridge.gta5.map import through_paint
    from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
    counts = through_paint.priority_roads(args.out, args.survey_lines, to_game)
    print("roads carried through junctions: " + ', '.join(f'{n} {k}' for k, n in counts.items()))
    tiles = os.path.join(os.path.dirname(args.survey_lines), 'tiles')  # roadpaint's, beside its polylines
    if os.path.isdir(tiles):
      from openpilot.tools.sim.bridge.gta5.map import painted_islands
      from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game
      from openpilot.tools.sim.bridge.gta5.map.osm_lanes import OsmLanes
      osm = OsmLanes.load(args.out, to_game)
      found = painted_islands.islands(args.survey_lines, tiles, osm)
      strips = painted_islands.flush_strips(osm, tiles)
      painted_islands.add(args.out, found, to_lat_lon, strips)
      print(f"{len(found)} painted islands (traffic_calming=painted_island), {len(strips)} flush edges' road surface (area:highway)")
  kinds = ', '.join(f'{sum(t[0] == k for t in turns)} {k}' for k in ('no_left_turn', 'no_right_turn', 'no_straight_on'))
  print(f"{u_turns} U-turns forbidden; GTA's turn flags: {len(turns)} turns forbidden ({kinds}), {skipped} through too many ways and {dead_ends} " +
        "approaches GTA leaves no way out of left out")
  if args.sidecar:
    n = write_sidecar(args.sidecar, nodes, used, [r[:8] for r in info])
    print(f"{n} links crossing on different levels -> {args.sidecar}")


if __name__ == '__main__':
  main()
