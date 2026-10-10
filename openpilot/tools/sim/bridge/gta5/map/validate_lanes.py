#!/usr/bin/env python3
"""Checks the lane tags of an OSM file (osm_lanes.py's tags), ours or a real extract:
- lanes: counts are counts, and lanes = lanes:forward + lanes:backward + lanes:both_ways;
- count: every *:lanes value (turn, width, change, destination, access, ...) has as many lanes as the road that way;
- turn / change / width / divider / lane_markings: values OSM knows; our edge_line:<side> a colour, its offset metres;
- width: width:lanes fit in width (less its parking lanes);
- parking: parking:left|right|both values OSM knows, their widths metres;
- placement: well formed, its lane is on the road, and given both ways the two agree but for the median; our
  placement:offset metres;
- transition: placement=transition only on ways up to TRANSITION_MAX long;
- connectivity: type=connectivity relations have from, via and to, and their lanes are on those roads;
- exit: every turn arrow has a way out that way at the junction ahead, that no restriction forbids.

  python validate_lanes.py gta5.osm.pbf [--show 20]
"""
import argparse
import math
import re
from collections import Counter, defaultdict
from typing import NamedTuple

import osmium

from openpilot.tools.sim.bridge.gta5.map.osm_lanes import CHANGES, DIVIDERS, PER_LANE, PLACEMENT, SUFFIXES, TRANSITION_MAX, \
  TURNS, WayLanes, count, lane_counts, metres, oneway_of, signed_metres

ROADS = {'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'service', 'track',
         'living_street', 'road', 'busway', 'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link'}
PHYSICAL_DIVIDERS = {'barrier', 'kerb', 'grass_verge', 'rumble_strip', 'flexible_posts', 'hatched'}
PARKING = {'lane', 'street_side', 'on_kerb', 'half_on_kerb', 'shoulder', 'no', 'separate'}  # parking:<side>=*, OSM Street parking
METRES_PER_DEGREE = 6378137.0 * math.pi / 180
REACH = 100.0  # m on along the road from a way with arrows to the junction they're for
JUNCTION_SPAN = 20.0  # m across a junction's straight short ways, as to a divided road's far side
APPROACH_SPAN = 15.0  # m: turns are also measured from the road's heading over this far before the junction
STRAIGHT, U_TURN = 45.0, 135.0  # deg
# which way out each arrow needs, by its turn (deg, left positive)
NEEDS = {
  'left': lambda t: t > 30, 'sharp_left': lambda t: t > 30, 'slight_left': lambda t: t > 5,
  'right': lambda t: t < -30, 'sharp_right': lambda t: t < -30, 'slight_right': lambda t: t < -5,
  'through': lambda t: abs(t) < 60, 'reverse': lambda t: abs(t) > U_TURN,
}
CONNECTIVITY_LANE = re.compile(r'\(?(\d+|bw)\)?')


class Issue(NamedTuple):
  check: str
  osm: str  # w123, r45
  detail: str


def wrap(deg: float) -> float:
  return (deg + 180) % 360 - 180


def heading(p, q) -> float:
  """Counterclockwise from north, as paths.heading."""
  return math.degrees(math.atan2(-(q[0] - p[0]), q[1] - p[1]))


def arrows(tags: dict, d: int) -> str | None:
  """A way's turn:lanes travelled d (1 along it, -1 against)."""
  oneway = oneway_of(tags)
  return tags.get('turn:lanes:forward' if d == 1 else 'turn:lanes:backward') or (tags.get('turn:lanes') if oneway == d else None)


def check_tags(tags: dict, length: float | None = None, drive_on_right: bool = True) -> list[tuple[str, str]]:
  """The tag checks of one way: [(check, detail)]."""
  out = []
  oneway = oneway_of(tags)
  for k in ('lanes', 'lanes:forward', 'lanes:backward', 'lanes:both_ways'):
    if k in tags and count(tags[k]) is None:
      out.append(('lanes', f'{k}={tags[k]} is not a count'))
  lanes, fwd, back, both = (count(tags.get(k)) for k in ('lanes', 'lanes:forward', 'lanes:backward', 'lanes:both_ways'))
  if oneway and ((back if oneway == 1 else fwd) or both):
    out.append(('lanes', 'lanes against the way, or both ways, on a one-way road'))
  if not oneway and None not in (lanes, fwd, back) and lanes != fwd + back + (both or 0):
    out.append(('lanes', f'lanes={lanes} but forward {fwd} + backward {back} + both ways {both or 0}'))
  f, b, w = lane_counts(tags)
  want = {'': f if oneway == 1 else b if oneway == -1 else f + b + w, ':forward': f, ':backward': b, ':both_ways': w}
  # and the widths at the way's ends where its lanes widen or narrow along it
  keys = [(k, s, '') for k in PER_LANE for s in SUFFIXES] + [('width', s, e) for s in SUFFIXES for e in (':start', ':end')]
  for key, suffix, end in keys:
    tag = f'{key}:lanes{suffix}{end}'
    if tag not in tags:
      continue
    items = tags[tag].split('|')
    if len(items) != want[suffix]:
      out.append(('count', f'{tag} has {len(items)} lanes, the road {want[suffix]}'))
    if key == 'turn':
      unknown = {t for item in items for t in item.split(';') if t and t not in TURNS}
      if unknown:
        out.append(('turn', f'{tag}: {", ".join(sorted(unknown))}'))
    elif key == 'change' and (unknown := {v for v in items if v and v not in CHANGES}):
      out.append(('change', f'{tag}: {", ".join(sorted(unknown))}'))
    elif key == 'width' and any(v and metres(v) is None for v in items):
      out.append(('width', f'{tag}={tags[tag]}'))
  if 'width' in tags and not metres(tags['width']):
    out.append(('width', f"width={tags['width']}"))
  # a median's two edges where they differ; a one-way way's lines along its lanes' edges
  for key in ('divider', 'divider:forward', 'divider:backward', 'divider:left', 'divider:right'):
    if key in tags and tags[key] not in DIVIDERS and not set(tags[key].split(';')) <= PHYSICAL_DIVIDERS:
      out.append(('divider', f"{key}={tags[key]}"))
  if tags.get('divider:colour', 'yellow') not in ('white', 'yellow'):
    out.append(('divider', f"divider:colour={tags['divider:colour']}"))
  for side in ('left', 'right'):  # our edge lines no shoulder draws
    if tags.get(f'edge_line:{side}', 'white') not in ('white', 'yellow'):
      out.append(('divider', f"edge_line:{side}={tags[f'edge_line:{side}']}"))
    if f'edge_line:{side}:offset' in tags and signed_metres(tags[f'edge_line:{side}:offset']) is None:
      out.append(('divider', f"edge_line:{side}:offset={tags[f'edge_line:{side}:offset']}"))
  if tags.get('lane_markings', 'yes') not in ('yes', 'no'):
    out.append(('lane_markings', f"lane_markings={tags['lane_markings']}"))
  for side in ('left', 'right', 'both'):
    if tags.get(f'parking:{side}', 'no') not in PARKING:
      out.append(('parking', f"parking:{side}={tags[f'parking:{side}']}"))
    if f'parking:{side}:width' in tags and not metres(tags[f'parking:{side}:width']):
      out.append(('parking', f"parking:{side}:width={tags[f'parking:{side}:width']}"))

  road = WayLanes.from_tags(tags, drive_on_right)
  width, known, unknown = road.tagged
  if width and not unknown and known > width + 0.05:
    out.append(('width', f'width:lanes add up to {known:.2f} m, more than width={width:g}'))
  elif width and unknown and known > width - 0.05:
    out.append(('width', f'width:lanes leave no room in width={width:g} for {unknown} more lanes'))
  plain = f if oneway == 1 else b if oneway == -1 else f + b + w
  for key, n_lanes in (('placement', plain), ('placement:forward', f), ('placement:backward', b)):
    v = tags.get(key)
    if v is None:
      continue
    if v == 'transition':
      if length is not None and length > TRANSITION_MAX:
        out.append(('transition', f'placement=transition on a way {length:.0f} m long'))
    elif not (m := PLACEMENT.fullmatch(v)):
      out.append(('placement', f'{key}={v}'))
    elif int(m.group(2)) > n_lanes:
      out.append(('placement', f'{key}={v} but {n_lanes} lanes that way'))
  if 'placement:offset' in tags and signed_metres(tags['placement:offset']) is None:
    out.append(('placement', f"placement:offset={tags['placement:offset']}"))
  if len(road.placed) > 1:
    # apart by no more than the median, and where lanes widen along the way, the lanes either side of it (turn bays
    # opening in it: placed beyond them, the line doesn't move with them)
    lo, hi = min(road.placed.values()), max(road.placed.values())
    widening = any(k.startswith('width:lanes') and k.endswith((':start', ':end')) for k in tags)
    inner = {k for i in range(1, len(road.lanes)) if road.lanes[i - 1].direction != road.lanes[i].direction for k in (i - 1, i)} \
      if widening else set()
    between = sum(b - a for a, b in road.gaps) + sum(road.lanes[i].width for i in inner
                                                     if road.x[i] >= lo - 0.05 and road.x[i] + road.lanes[i].width <= hi + 0.05)
    if hi - lo > between + 0.05:
      out.append(('placement', f'placements {", ".join(road.placed)} disagree by {hi - lo - between:.2f} m'))
  return out


class Network:
  """The roads of an OSM file as moves between the nodes where ways meet."""
  def __init__(self, ways: dict, restrictions: list):
    self.ways = ways  # id -> (node refs, [(x, y)], tags)
    self.restrictions = defaultdict(list)  # from way -> [(kind, (via ways), to way)]
    for kind, wf, via, wt in restrictions:
      self.restrictions[wf].append((kind, via, wt))
    uses = Counter(n for refs, _, _ in ways.values() for n in refs)
    self.shared = {n for n, c in uses.items() if c > 1} | {refs[k] for refs, _, _ in ways.values() for k in (0, -1)}
    self.at = defaultdict(list)  # node -> [(way, index)]
    for wid, (refs, _, _) in ways.items():
      for i, n in enumerate(refs):
        self.at[n].append((wid, i))

  def moves(self, node):
    """The moves out of a node: (way, direction, end node, heading out, length, its points)."""
    out = []
    for wid, i in self.at[node]:
      refs, xy, tags = self.ways[wid]
      oneway = oneway_of(tags)
      for d in (1, -1):
        if (oneway and d != oneway) or not 0 <= i + d < len(refs):
          continue
        j, dist = i + d, 0.0
        while True:
          dist += math.dist(xy[j - d], xy[j])
          if refs[j] in self.shared or not 0 <= j + d < len(refs):
            break
          j += d
        out.append((wid, d, refs[j], heading(xy[i], xy[i + d]), dist, xy[i:j + 1] if d == 1 else xy[j:i + 1][::-1]))
    return out

  def forbidden(self, seq: list[int]) -> bool:
    """Whether restrictions forbid driving along the ways `seq`."""
    for k, w in enumerate(seq):
      for kind, via, wt in self.restrictions.get(w, ()):
        if tuple(seq[k + 1:k + 1 + len(via)]) != via or k + 1 + len(via) >= len(seq):
          continue
        nxt = seq[k + 1 + len(via)]
        if (kind.startswith('no_') and nxt == wt) or (kind.startswith('only_') and nxt != wt):
          return True
    return False

  def exits(self, wid: int, d: int) -> list[list[tuple[float, bool]]] | None:
    """The ways out of the junction at the end of the run of ways with arrows from way `wid` travelled `d` (mapped up
    to the junction, their arrows changing where the road gains lanes): [(turn, allowed)], measured once from the last bit of road in and once from the road
    over the last APPROACH_SPAN m, as it can bend or jog across lanes into the junction; None at a dead end or
    after REACH."""
    refs, xy, tags = self.ways[wid]
    marked = arrows(tags, d)
    trail = list(xy if d == 1 else xy[::-1])
    first, run = refs[0] if d == 1 else refs[-1], {wid}
    while math.dist(trail[0], trail[-1]) < APPROACH_SPAN:  # the arrows' run back, for the road's heading
      into = [(w, i, dd) for w, i in self.at[first] for dd in (1, -1) if w not in run and 0 <= i - dd < len(self.ways[w][0])
              and arrows(self.ways[w][2], dd) == marked and oneway_of(self.ways[w][2]) in (0, dd)]
      if len(into) != 1:
        break
      w, i, dd = into[0]
      pts = self.ways[w][1][:i + 1] if dd == 1 else self.ways[w][1][i:][::-1]
      trail[:0] = pts[:-1]
      first = self.ways[w][0][0] if dd == 1 else self.ways[w][0][-1]
      run.add(w)
    seq, dist, node = [wid], 0.0, refs[-1] if d == 1 else refs[0]
    while True:
      h_in = heading(trail[-2], trail[-1])
      ahead = [m for m in self.moves(node) if not (m[0] == seq[-1] and m[1] != d) and abs(wrap(m[3] - h_in)) <= U_TURN]
      # on while the road goes on marked, its arrows changing where it gains or loses lanes on the way in
      if len(ahead) != 1 or arrows(self.ways[ahead[0][0]][2], ahead[0][1]) is None:
        break
      m = ahead[0]
      dist += m[4]
      if dist > REACH or m[0] in seq:
        return None
      seq.append(m[0])
      trail += m[5][1:]
      node, d = m[2], m[1]
    if not ahead:
      return None
    back, k = 0.0, len(trail) - 1
    while k > 0 and back < APPROACH_SPAN:
      back += math.dist(trail[k - 1], trail[k])
      k -= 1
    out = []
    for h in (h_in, heading(trail[k], trail[-1])):
      found, seen, stack = [], {node}, [(node, [], 0.0)]
      while stack:
        n, inner, dist = stack.pop()
        for m in self.moves(n):
          turn = wrap(m[3] - h)
          if m[2] in seen or (m[0] == (inner or seq)[-1] and abs(turn) > U_TURN):  # not back the way it came
            continue
          seen.add(m[2])
          if abs(turn) <= STRAIGHT and dist + m[4] <= JUNCTION_SPAN and any(k[2] != n for k in self.moves(m[2])):
            stack.append((m[2], inner + [m[0]], dist + m[4]))
          else:
            found.append((turn, not self.forbidden(seq + inner + [m[0]])))
      out.append(found)
    return out


def validate(path: str, drive_on_right: bool = True) -> list[Issue]:
  ways, restrictions, connectivity = {}, [], []
  for o in osmium.FileProcessor(path).with_locations():
    if o.is_way() and o.tags.get('highway') in ROADS:
      xy = [(n.location.lon * METRES_PER_DEGREE * math.cos(math.radians(n.location.lat)), n.location.lat * METRES_PER_DEGREE)
            for n in o.nodes]
      ways[o.id] = ([n.ref for n in o.nodes], xy, dict(o.tags))
    elif o.is_relation() and o.tags.get('type') in ('restriction', 'connectivity'):
      members = [(m.type, m.ref, m.role) for m in o.members]
      if o.tags['type'] == 'connectivity':
        connectivity.append((o.id, dict(o.tags), members))
      elif (kind := o.tags.get('restriction', '')).startswith(('no_', 'only_')):
        wf = [r for t, r, role in members if t == 'w' and role == 'from']
        wt = [r for t, r, role in members if t == 'w' and role == 'to']
        if len(wf) == 1 and len(wt) == 1:
          restrictions.append((kind, wf[0], tuple(r for t, r, role in members if t == 'w' and role == 'via'), wt[0]))

  issues = []
  for wid, (_, xy, tags) in ways.items():
    length = sum(math.dist(p, q) for p, q in zip(xy, xy[1:], strict=False))
    issues += [Issue(check, f'w{wid}', detail) for check, detail in check_tags(tags, length, drive_on_right)]
  issues += check_connectivity(ways, connectivity)

  net = Network(ways, restrictions)
  for wid, (_, _, tags) in ways.items():
    oneway = oneway_of(tags)
    for d, key in ((1, 'turn:lanes:forward'), (-1, 'turn:lanes:backward')):
      keys = [k for k in (key, 'turn:lanes' if oneway == d else None) if k in tags]
      if not keys or (oneway and oneway != d):
        continue
      exits = net.exits(wid, d)
      if exits is None:
        issues.append(Issue('exit', f'w{wid}', f"{keys[0]} runs into a dead end, or on for over {REACH:.0f} m"))
        continue
      for k, span in enumerate(WayLanes.from_tags(tags, drive_on_right).ours(d), 1):
        for turn in sorted(span.lane.turns):
          need = NEEDS.get(turn)
          if need and not any(ok and need(t) for found in exits for t, ok in found):
            issues.append(Issue('exit', f'w{wid}', f'{keys[0]} lane {k} {turn}: no allowed way out that way'))
  return issues


def check_connectivity(ways: dict, relations: list) -> list[Issue]:
  out = []
  for rid, tags, members in relations:
    def bad(detail, rid=rid):
      out.append(Issue('connectivity', f'r{rid}', detail))
    roles = Counter(role for _, _, role in members)
    wf = [r for t, r, role in members if role == 'from' and t == 'w']
    wt = [r for t, r, role in members if role == 'to' and t == 'w']
    via = [(t, r) for t, r, role in members if role == 'via']
    if roles['from'] != 1 or roles['to'] != 1 or not via or len(wf) != 1 or len(wt) != 1 or set(roles) - {'from', 'via', 'to'}:
      bad('needs one from way, one to way and a via node or ways')
      continue
    if wf[0] not in ways or wt[0] not in ways or any(t == 'w' and r not in ways for t, r in via):
      continue  # cut off by the extract
    ends = [ways[r][0] for t, r in via if t == 'w']
    first = {via[0][1]} if via[0][0] == 'n' else {ends[0][0], ends[0][-1]}
    last = {via[-1][1]} if via[-1][0] == 'n' else {ends[-1][0], ends[-1][-1]}

    def lanes_into(wid, nodes, arriving):
      refs, _, t = ways[wid]
      f, b, w = lane_counts(t)
      at_end = refs[-1] in nodes if arriving else refs[0] in nodes
      at_start = refs[0] in nodes if arriving else refs[-1] in nodes
      if at_end:
        return f or w
      return (b or w) if at_start else None

    n_from, n_to = lanes_into(wf[0], first, True), lanes_into(wt[0], last, False)
    if n_from is None or n_to is None:
      bad('from or to way does not meet the via')
      continue
    value = tags.get('connectivity', '')
    for group in value.split('|'):
      lane_from, _, lanes_to = group.partition(':')
      tokens = [lane_from, *lanes_to.split(',')]
      if not lanes_to or not all(CONNECTIVITY_LANE.fullmatch(x) for x in tokens):
        bad(f'connectivity={value}: {group!r} is not lane:lanes')
        break
      nums = [CONNECTIVITY_LANE.fullmatch(x).group(1) for x in tokens]
      if (nums[0] != 'bw' and int(nums[0]) > n_from) or any(x != 'bw' and int(x) > n_to for x in nums[1:]):
        bad(f'connectivity={value}: lanes beyond the {n_from} in / {n_to} out')
        break
  return out


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('osm', help='.osm or .osm.pbf')
  p.add_argument('--show', type=int, default=10, help='issues to show of each check')
  p.add_argument("--left", action="store_true", help="traffic drives on the left (UK)")
  args = p.parse_args()
  issues = validate(args.osm, not args.left)
  by_check = defaultdict(list)
  for issue in issues:
    by_check[issue.check].append(issue)
  print(f'{len(issues)} issues' + (':' if issues else ''))
  for check, found in sorted(by_check.items(), key=lambda kv: -len(kv[1])):
    print(f'  {check}: {len(found)}')
    for issue in found[:args.show]:
      print(f'    {issue.osm} {issue.detail}')
  return 1 if issues else 0


if __name__ == '__main__':
  raise SystemExit(main())
