"""Where a destination meets the road: the road a map waypoint faces, and the way to arrive along it.

A waypoint is usually put on a building, a car park or a petrol station beside its street, so the nearest road is often
a car park aisle, a driveway or an alley next to the street meant. This picks the road by how far the waypoint is from
its kerb, plus a penalty for roads one doesn't arrive by:
- minor roads (`highway=service` with any `service=*`, `track`, private access): a driveway, car park aisle or alley,
  unless the waypoint is on one and no ordinary road is about as near (half the penalty for a named service road with
  no `service=*`, often a street);
- stubs: a road that runs from a junction to a dead end within STUB, as a drive into a lot does;
- motorways and ramps, where nobody stops, and bridges and tunnels, which pass over or under what the waypoint is beside;
- a road behind another: the line from the waypoint to it crosses an ordinary road nearer the waypoint.
Each penalty only reorders roads near the waypoint: one far nearer than any other still wins.

On the road found, the car arrives along a one-way road's direction; on a two-way road, with the waypoint on its kerb
side (the right where traffic drives on the right), unless the waypoint is on the road itself or past its dead end, or
the road is a minor one (either side of a car park aisle will do). On a divided road the carriageway nearer the
waypoint is the one on its side. The router may still arrive the other way where the kerb side is a long way round.
It reads only standard OSM tags (osm_lanes.OsmLanes), so it works the same on a real map."""
import math
from dataclasses import dataclass

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.osm_lanes import FORWARD, OsmLanes, oneway_of

SEARCH = 60.0  # m from the waypoint to look for roads
ON_ROAD = 1.0  # m beyond a road's kerb that still counts as on it
MINOR = frozenset({'service', 'track'})
MINOR_PENALTY = 25.0  # m: a driveway, car park aisle or alley loses to an ordinary road up to this much farther away ...
MINOR_ON = 6.0  # m: ... or, with the waypoint on it, to one whose kerb is within this
NO_STOP = frozenset({'motorway', 'motorway_link', 'trunk_link'})
NO_STOP_PENALTY = 40.0  # m
STUB = 80.0  # m from a junction to a dead end: a drive into a lot, not a street
STUB_PENALTY = 25.0  # m
LEVEL_PENALTY = 10.0  # m: a bridge or tunnel
BEHIND_PENALTY = 30.0  # m: an ordinary road lies between the waypoint and this one
END_GAP = 3.0  # m from a segment's ends, so the router can't take the destination for a road meeting there


@dataclass
class Snap:
  point: np.ndarray  # on the road's line, game metres
  way: int
  heading: float | None  # deg clockwise from north (a router's bearing) to arrive along, None either way
  kerb_side: bool  # the heading puts the waypoint on the kerb side of a two-way road (another way would do)
  highway: str
  off: float  # m from the waypoint to the road's kerb
  score: float  # m, the kerb distance plus penalties
  as_is: bool  # the waypoint is on its nearest road: a router's nearest road is this one


def bearing(d) -> float:
  return math.degrees(math.atan2(d[0], d[1])) % 360


class DestinationSnapper:
  def __init__(self, osm: OsmLanes):
    self.osm = osm
    a, b, way = [], [], []
    for wid, (_, refs) in osm.ways.items():
      for n0, n1 in zip(refs, refs[1:], strict=False):
        a.append(n0)
        b.append(n1)
        way.append(wid)
    self.na, self.nb, self.way = np.array(a, np.int64), np.array(b, np.int64), np.array(way, np.int64)
    self.a = osm.xy[osm.data.index(self.na)] if len(a) else np.zeros((0, 2))
    self.b = osm.xy[osm.data.index(self.nb)] if len(b) else np.zeros((0, 2))
    self._stub: dict[int, bool] = {}

  def neighbours(self, node: int) -> set[int]:
    return set(self.osm.links.get(node, ()))

  def _reach(self, node: int, prev: int, limit: float) -> tuple[float, bool]:
    """From node, coming from prev, along the road while it just carries on (two neighbours): how far until it doesn't,
    and whether it's a dead end there; not one once past `limit`."""
    dist, seen = 0.0, {prev, node}
    while True:
      nbrs = self.neighbours(node)
      if len(nbrs) <= 1:
        return dist, True
      if len(nbrs) != 2:
        return dist, False
      nxt = next(iter(nbrs - {prev}), None)
      if nxt is None or nxt in seen:
        return dist, False  # a loop
      dist += float(np.hypot(*(self.osm.node_xy(nxt) - self.osm.node_xy(node))))
      if dist > limit:
        return dist, False
      seen.add(nxt)
      prev, node = node, nxt

  def stub(self, k: int) -> bool:
    """Whether segment k lies on a road that runs from a junction to a dead end within STUB."""
    if k not in self._stub:
      length = float(np.hypot(*(self.b[k] - self.a[k])))
      back, dead_back = self._reach(int(self.na[k]), int(self.nb[k]), STUB)
      on, dead_on = self._reach(int(self.nb[k]), int(self.na[k]), STUB)
      self._stub[k] = (dead_back or dead_on) and back + length + on <= STUB
    return self._stub[k]

  def candidates(self, p: np.ndarray, search: float = SEARCH) -> list[dict]:
    """The road segments within `search` m of p: each one's nearest point to p and how far p is from its kerb."""
    ab = self.b - self.a
    length2 = np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-9)
    t = np.clip(np.einsum('ij,ij->i', p - self.a, ab) / length2, 0.0, 1.0)
    foot = self.a + ab * t[:, None]
    d = np.hypot(*(foot - p).T)
    out = []
    for k in np.flatnonzero(d < search):
      k = int(k)
      wid = int(self.way[k])
      tags = self.osm.ways[wid][0]
      lo, hi = self.osm.lanes(wid).edges(FORWARD)  # the kerbs, m right of the way's line
      seg = math.sqrt(length2[k])
      right = float((p[0] - self.a[k, 0]) * ab[k, 1] - (p[1] - self.a[k, 1]) * ab[k, 0]) / seg
      inside = 0.0 < t[k] < 1.0
      off = max(lo - right, right - hi, 0.0) if inside else max(float(d[k]) - (hi - lo) / 2, 0.0)
      out.append({'k': k, 'way': wid, 'tags': tags, 't': float(t[k]), 'foot': foot[k], 'd': float(d[k]), 'off': off,
                  'right': right, 'kerbs': (lo, hi), 'length': seg})
    return out

  @staticmethod
  def minor(tags: dict) -> bool:
    return tags.get('highway') in MINOR or 'service' in tags or \
      any(tags.get(k) in ('private', 'no') for k in ('access', 'vehicle', 'motor_vehicle', 'motorcar'))

  @staticmethod
  def street(tags: dict) -> bool:
    """A named service road with no kind of service given: as likely a street mapped as minor as a car park."""
    return tags.get('highway') == 'service' and 'name' in tags and 'service' not in tags

  @staticmethod
  def level(tags: dict) -> bool:
    return tags.get('bridge', 'no') != 'no' or tags.get('tunnel', 'no') != 'no' or tags.get('layer', '0') != '0'

  def _behind(self, p: np.ndarray, c: dict, others: list[dict]) -> bool:
    """Whether the line from p to c's foot crosses an ordinary road (at p's level) on the way, other than c's own."""
    q = c['foot']
    k = c['k']
    ends = {int(self.na[k]), int(self.nb[k])}
    for o in others:
      j = o['k']
      if o['way'] == c['way'] or self.minor(o['tags']) or self.level(o['tags']) or \
         int(self.na[j]) in ends or int(self.nb[j]) in ends:
        continue
      if _cross(p, q, self.a[j], self.b[j]):
        return True
    return False

  def score(self, p: np.ndarray, c: dict, cands: list[dict], behind: bool = True) -> float:
    """How far p is from c's kerb, plus its penalties; without the road-behind check (the slow one) unless `behind`."""
    tags, off = c['tags'], c['off']
    on = off <= ON_ROAD
    s = off
    if self.minor(tags):
      s += MINOR_ON if on else MINOR_PENALTY / 2 if self.street(tags) else MINOR_PENALTY
    if tags.get('highway') in NO_STOP and not on:
      s += NO_STOP_PENALTY
    if self.level(tags) and not on:
      s += LEVEL_PENALTY
    if not on and self.stub(c['k']):
      s += STUB_PENALTY
    if behind and not on and self._behind(p, c, cands):
      s += BEHIND_PENALTY
    return s

  def best(self, p: np.ndarray, cands: list[dict]) -> tuple[dict, float]:
    """The candidate with the lowest score (nearest first among equals), and its score."""
    base = sorted(((self.score(p, c, cands, behind=False), c['d'], n) for n, c in enumerate(cands)))
    best, best_key = None, (math.inf, math.inf)
    for s, d, n in base:
      if s > best_key[0]:
        break  # the road-behind penalty only adds
      key = (self.score(p, cands[n], cands), d)
      if key < best_key:
        best, best_key = cands[n], key
    assert best is not None
    return best, best_key[0]

  def snap(self, dest, search: float = SEARCH) -> Snap | None:
    """The road the destination faces, where on it to arrive and which way; None with no road within `search` m."""
    p = np.asarray(dest, float)[:2]
    cands = self.candidates(p, search)
    if not cands:
      return None
    best, score = self.best(p, cands)
    k, tags = best['k'], best['tags']
    on = best['off'] <= ON_ROAD
    nearest = best['d'] <= min(c['d'] for c in cands) + 0.01
    gap = 0.0 if on else min(END_GAP / max(best['length'], 1e-6), 0.5)
    t = min(max(best['t'], gap), 1.0 - gap)
    point = self.a[k] + (self.b[k] - self.a[k]) * t
    along = bearing(self.b[k] - self.a[k])
    oneway = oneway_of(tags)
    end = int(self.na[k]) if best['t'] <= 0.0 else int(self.nb[k]) if best['t'] >= 1.0 else None
    past_end = end is not None and len(self.neighbours(end)) <= 1
    # no side on the road itself, past its dead end, or in a car park or drive, where one stops either side
    kerb_side = not oneway and not on and not past_end and not self.minor(tags)
    if oneway:
      heading = along if oneway == 1 else (along + 180) % 360
    elif kerb_side:
      lo, hi = best['kerbs']
      heading = along if (best['right'] > (lo + hi) / 2) == self.osm.drive_on_right else (along + 180) % 360
    else:
      heading = None
    return Snap(point, best['way'], heading, kerb_side, tags.get('highway', ''), best['off'], score, on and nearest)


def _cross(p, q, a, b) -> bool:
  """Whether segments p-q and a-b cross (touching at an end doesn't count)."""
  def side(u, v, w):
    return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
  d1, d2, d3, d4 = side(a, b, p), side(a, b, q), side(p, q, a), side(p, q, b)
  return d1 * d2 < 0 and d3 * d4 < 0
