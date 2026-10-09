"""Routes over a Valhalla server (map/README.md), as a car's navigation would, and follows the car along the route."""
import json
import math
import os
import threading
import urllib.error
import urllib.request

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.dest_snap import DestinationSnapper, Snap
from openpilot.tools.sim.bridge.gta5.map.gta5_map import to_game, to_lat_lon
from openpilot.tools.sim.bridge.gta5.map.osm_lanes import BACKWARD, FORWARD, Lane, OsmLanes, RouteLanes, Section, Span, \
  ways_from_nodes
from openpilot.tools.sim.bridge.gta5.map.paths import CAR_HEIGHT, SLIP_LANE, Link, Paths, wrap
from openpilot.tools.sim.bridge.gta5.map.stop_lines import StopLines

SERVICE_PENALTY, SERVICE_FACTOR = 120, 4.0  # s onto a service road, and its cost over a road's
HEADING_TOLERANCE = 45.0  # deg: start on a road heading the car's way, not the opposite carriageway
SNAP_TOLERANCE = 60.0  # deg, the roads leaving the next node of the road found at the car's height and heading
FORK_SPREAD = 40.0  # deg either side of straight on, up to nav's turns: a road branching off within this is a fork
OTHER_LEVEL = 3.0  # m above or below the route: the car is on another road, passing over or under it
WRONG_WAY = 100.0  # deg from the route's direction: the car isn't driving that part of it
FORK_BEHIND = 50.0  # m: nav keeps to a fork's side a little past it
LANE_ALIGN = 10.0  # deg
LANES_NEAR = 60.0  # m from a point to look for a link with its lanes
ON_ROAD = 8.0  # m from the route's line: on its road, wherever its lanes are
KERB_MARGIN = 1.0  # m outside the kerbs of the route's road still on it
FORK_AT = 2.0  # m between a fork in GTA's roads and where the map's lanes change for it
TURN_STEP = 20.0  # m: the route's turns are found on its own points, with more where they're further apart
JUNCTION_BEHIND = 30.0  # m: nav times a turn from its junction's entry, which the car may be past
# the destination on the road it faces, rather than the router's nearest road, often a drive or car park (dest_snap.py)
DEST_SNAP = os.getenv("GTA5_DEST_SNAP", "1") != "0"
DEST_HEADING_TOLERANCE = 45.0  # deg
SIDE_DETOUR = 30.0  # s: arriving with the destination on the kerb side may take this much longer than across the road
# on a map without lane tags, where nav's stop lines are GTA's nodes: a stop line counts only on the way towards its
# junction, not for a route leaving the junction past it (off: both)
STOP_DIRECTION = os.getenv("GTA5_STOP_DIRECTION", "0") == "1"


def decode_polyline(encoded: str, precision: int = 6) -> list[tuple[float, float]]:
  """Google's encoded polyline, as Valhalla returns shapes: (lat, lon) pairs."""
  out, index, lat, lon = [], 0, 0, 0
  while index < len(encoded):
    for axis in range(2):
      shift = result = 0
      while True:
        b = ord(encoded[index]) - 63
        index += 1
        result |= (b & 0x1f) << shift
        shift += 5
        if b < 0x20:
          break
      delta = ~(result >> 1) if result & 1 else result >> 1
      if axis == 0:
        lat += delta
      else:
        lon += delta
    out.append((lat / 10 ** precision, lon / 10 ** precision))
  return out


def link_section(link: Link) -> Section:
  """A GTA link's lanes as a cross-section (its two directions laid out alike), for a map without lane tags."""
  w = link.width
  ours = [Span(Lane(FORWARD, w), link.inner + k * w, link.inner + (k + 1) * w, 1) for k in range(link.lanes)]
  oncoming = [Span(Lane(BACKWARD, w), -(link.inner + (k + 1) * w), -(link.inner + k * w), -1) for k in reversed(range(link.back))]
  spans = oncoming + ours
  return Section(spans, (min(s.left for s in spans), max(s.right for s in spans)))


class Fork:
  def __init__(self, along: float, side: str, lanes: int, lanes_in: int, keep: bool, other: int = 0, slip: bool = False):
    self.along = along  # m along the route
    self.side = side  # the branch the route takes, "left" or "right"
    self.lanes, self.lanes_in = lanes, lanes_in  # on that branch, and on the road before
    self.other = other  # lanes on the other branches
    # whether it's a fork in the road, rather than GTA splitting a road's lanes before a junction, where the car just
    # keeps to its lane
    self.keep = keep
    self.slip = slip  # the other branch only opens a slip lane or turn bay, as for a turn soon after


class Route:
  """A route's points (game metres), and where along it the car is. With GTA's road data, also the road's height and
  the roads that fork off it; with the map's speed limits, those. Its lanes come from the map's lane tags (osm_lanes.py),
  or where the map has none, from GTA's own layout of each link (paths.Link); its stop lines from the map's
  (stop_lines.py, as the map view and the overlay draw them: those it crosses into a junction, its own way), or on a map
  without lane tags, GTA's stop line nodes."""
  def __init__(self, points: np.ndarray, maneuvers: list[dict], paths: Paths | None = None, limits: np.ndarray | None = None,
               osm: OsmLanes | None = None, stop_lines: StopLines | None = None):
    self.points = points
    self.along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(self.points, axis=0).T))))
    self.maneuvers = maneuvers
    self.at = 0.0  # m along the route to the car
    self.seg = 0  # the route segment the car is on
    self.right = 0.0  # m right of the route's line
    self.off = 0.0  # m off the route
    self.misaligned = 0.0  # deg between the car's heading and the route's there
    self.elsewhere = False  # on another level or heading the other way, where right and seg are from before
    n = len(points)
    self.z = np.full(n, np.nan)
    self.links: list[Link | None] = [None] * max(n - 1, 0)
    self.limits = limits if limits is not None else np.zeros(max(n - 1, 0))  # m/s per segment, 0 unknown
    self.forks: list[Fork] = []
    self.stops: list[float] = []  # m along the route to stop lines, in order
    self.stop_kinds: list[str] = []  # each one's: stop_lines.KINDS, "stop" (a stop sign), "lights" or "give_way"
    self.junctions: list[float] = []  # and to junction nodes
    self.osm_lanes = osm is not None and osm.tagged and n >= 2
    if paths is not None and n >= 2:
      self._add_paths(paths, stops=not self.osm_lanes)
    if self.osm_lanes:
      lines = stop_lines if stop_lines is not None else StopLines.of(osm)
      for at, kind in lines.on_route(points, self.along, osm) if lines is not None else []:
        self.stops.append(at)
        self.stop_kinds.append(kind)
    self.limit_list = [round(float(v), 2) for v in self.limits]
    self._lanes: RouteLanes | None = None
    self._lanes_of: list | None = None  # the links a RouteLanes from GTA's layout was made from
    if self.osm_lanes:
      self._lanes = RouteLanes.from_osm(points, ways_from_nodes(points, osm), osm)
    self._maps: list | None = None  # lane_maps along the whole route
    self._turns: list | None = None  # turns along the whole route
    self.lane_counts = [sec.lanes if (sec := self.section(k)) is not None else 0 for k in range(max(n - 1, 0))]

  @property
  def lanes(self) -> RouteLanes | None:
    """The lanes along it: from the map's tags, else (GTA only, until every map has them) from GTA's links."""
    if not self.osm_lanes and self._lanes_of is not self.links and len(self.points) >= 2:
      self._lanes = RouteLanes(self.points, [link_section(link) if link is not None and link.lanes else None
                                             for link in self.links], junctions=self.junctions)
      self._lanes_of = self.links
    return self._lanes

  def section(self, k: int) -> Section | None:
    """The road's cross-section along segment k, None where it's unknown."""
    lanes = self.lanes
    return lanes.sections[k] if lanes is not None and 0 <= k < len(lanes.sections) else None

  def _add_paths(self, paths: Paths, stops: bool = True):
    pts = self.points
    nodes = paths.route_nodes(pts)
    for k, i in enumerate(nodes):
      if i is not None and 0 < k < len(pts) - 1:
        if stops and paths.stop_line(i) and (not STOP_DIRECTION or self._stops_here(paths, nodes, k)):
          self.stops.append(float(self.along[k]))
          self.stop_kinds.append(paths.stop_kind(i))
        if paths.junction(i):
          self.junctions.append(float(self.along[k]))
    # the route starts and ends part way along a link: the node before its start and after its end
    if nodes[0] is None and nodes[1] is not None:
      nodes[0] = self._link_end(paths, nodes[1], pts[1] - pts[0], before=True)
      start = nodes[0]
    else:
      start = None
    if nodes[-1] is None and nodes[-2] is not None:
      nodes[-1] = self._link_end(paths, nodes[-2], pts[-1] - pts[-2], before=False)
      end = nodes[-1]
    else:
      end = None
    for k, i in enumerate(nodes):
      if i is not None:
        self.z[k] = paths.z[i]
    for k in range(len(pts) - 1):
      a, b = nodes[k], nodes[k + 1]
      if a is not None and b is not None:
        self.links[k] = paths.links.get((a, b))
    # the map's own nodes along GTA's links (where a lane's taper starts or ends): the link they're on
    gta = [k for k, i in enumerate(nodes) if i is not None]
    for k0, k1 in zip(gta, gta[1:], strict=False):
      if k1 > k0 + 1 and (link := paths.links.get((nodes[k0], nodes[k1]))) is not None:
        for k in range(k0, k1):
          self.links[k] = link
    for k, i in ((0, start), (len(pts) - 1, end)):
      if i is not None:  # the end's height along its link
        j = nodes[1] if k == 0 else nodes[-2]
        span = np.hypot(*(paths.xy[j] - paths.xy[i]))
        t = np.hypot(*(pts[k] - paths.xy[i])) / span if span > 0 else 0.0
        self.z[k] = paths.z[i] + (paths.z[j] - paths.z[i]) * t
    known = ~np.isnan(self.z)
    if known.any() and not known.all():
      self.z = np.interp(self.along, self.along[known], self.z[known])
    for n, k in enumerate(gta[1:-1], 1):  # GTA's nodes in turn, past any of the map's own between them
      prev, i, nxt = nodes[gta[n - 1]], nodes[k], nodes[gta[n + 1]]
      if prev == i or i == nxt or (k - gta[n - 1] > 1 and (prev, i) not in paths.links) or \
         (gta[n + 1] - k > 1 and (i, nxt) not in paths.links):
        continue
      fork = self._fork(paths, prev, i, nxt, float(self.along[k]))
      if fork is not None:
        self.forks.append(fork)

  @staticmethod
  def _stops_here(paths: Paths, nodes: list[int | None], k: int) -> bool:
    nxt = next((j for j in nodes[k + 1:] if j is not None and j != nodes[k]), None)
    return nxt is None or paths.stop_for(nodes[k], nxt)

  @staticmethod
  def _link_end(paths: Paths, i: int, direction: np.ndarray, before: bool) -> int | None:
    """The node linked to i along `direction` of travel: behind i, or ahead of it."""
    h = math.degrees(math.atan2(-direction[0], direction[1]))
    ends = paths.into.get(i, ()) if before else paths.out.get(i, ())
    best, best_off = None, 10.0
    for j in ends:
      off = abs(wrap((paths.link_heading(j, i) if before else paths.link_heading(i, j)) - h))
      if off < best_off:
        best, best_off = j, off
    return best

  @staticmethod
  def _fork(paths: Paths, prev: int, i: int, nxt: int, along: float) -> Fork | None:
    ours = wrap(paths.link_heading(i, nxt) - paths.link_heading(prev, i))
    if abs(ours) >= FORK_SPREAD:
      return None
    others = paths.forks(prev, i, nxt, FORK_SPREAD)
    if not others:
      return None
    if all(rel > ours for rel, _ in others):
      side = "right"
    elif all(rel < ours for rel, _ in others):
      side = "left"
    else:
      return None  # the middle of three: straight on
    link_in, link = paths.links.get((prev, i)), paths.links.get((i, nxt))
    if link_in is None or link is None:
      return None
    roads = [j for _, j in others if not paths.flags[j][1] & SLIP_LANE]
    if not roads:
      return Fork(along, side, link_in.lanes, link_in.lanes, False, slip=True)
    lane_split = link.no_nav and all(paths.links[(i, j)].no_nav for j in roads)
    return Fork(along, side, link.lanes, link_in.lanes, bool(paths.flags[i][2] & 64) or not lane_split,
                sum(paths.links[(i, j)].lanes for j in roads))

  @property
  def length(self) -> float:
    return float(self.along[-1])

  def locate(self, pos: np.ndarray, z: float | None = None, heading: float | None = None, search: float = 60.0) -> float:
    """Moves the car to its nearest point on the route from a little behind where it was, so where the route passes
    back near itself the car stays on the part it's driving; returns its distance off the route. With the car's height
    and heading (game degrees), parts of the route above or below it, or the other way, don't count."""
    lo = max(0, int(np.searchsorted(self.along, self.at - 10.0)) - 1)
    hi = min(len(self.points) - 1, int(np.searchsorted(self.along, self.at + search)) + 1)
    a, b = self.points[lo:hi], self.points[lo + 1:hi + 1]
    if len(a) == 0:
      self.off, self.elsewhere = float(np.hypot(*(pos - self.points[-1]))), True
      return self.off
    ab = b - a
    t = np.clip(np.einsum('ij,ij->i', pos - a, ab) / np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-9), 0.0, 1.0)
    near = a + ab * t[:, None]
    d = np.hypot(*(near - pos).T)
    other = np.zeros(len(d), dtype=bool)
    if z is not None and not np.isnan(self.z[lo:hi + 1]).any():
      road_z = self.z[lo:hi] + (self.z[lo + 1:hi + 1] - self.z[lo:hi]) * t
      other |= np.abs(z - CAR_HEIGHT - road_z) > OTHER_LEVEL
    if heading is not None:
      seg_heading = np.degrees(np.arctan2(-ab[:, 0], ab[:, 1]))
      other |= np.abs((heading - seg_heading + 180) % 360 - 180) > WRONG_WAY
    self.elsewhere = bool(other.all())
    if self.elsewhere:
      self.off = float(d.min()) + 100.0  # off the route, though it's near
      return self.off
    d = np.where(other, np.inf, d)
    i = int(np.argmin(d))
    self.seg = lo + i
    self.at = float(self.along[lo + i] + t[i] * np.hypot(*ab[i]))
    length = max(float(np.hypot(*ab[i])), 1e-6)
    self.right = float((pos[0] - a[i, 0]) * ab[i, 1] - (pos[1] - a[i, 1]) * ab[i, 0]) / length
    self.off = float(d[i])
    self.misaligned = 0.0 if heading is None else abs((heading - float(np.degrees(np.arctan2(-ab[i, 0], ab[i, 1]))) + 180) % 360 - 180)
    return self.off

  def on_road(self, near: float = ON_ROAD) -> bool:
    """Whether the car is on the route's road where it is: within `near` m of the route's line, or between the kerbs
    of the road's lanes there, as on a wide road whose line runs along its far side (a turn lane's middle)."""
    if self.elsewhere:
      return False
    if self.off < near:
      return True
    lanes = self.lanes
    sec = lanes.section_at(self.at, self.seg) if lanes is not None and 0 <= self.seg < len(lanes.sections) else None
    return sec is not None and bool(sec.lanes) and sec.edges[0] - KERB_MARGIN <= self.right <= sec.edges[1] + KERB_MARGIN

  def ahead(self, distance: float, step: float) -> np.ndarray:
    """Points every `step` m from the car for `distance` m, or to the route's end."""
    s = np.arange(self.at, min(self.at + distance, self.length) + 1e-6, step)
    return np.stack([np.interp(s, self.along, self.points[:, 0]), np.interp(s, self.along, self.points[:, 1])], axis=1)

  def rest(self) -> np.ndarray:
    """The route on from the car: its own points, so the shape holds still as the car moves (unlike ahead()'s)."""
    here = [np.interp(self.at, self.along, self.points[:, 0]), np.interp(self.at, self.along, self.points[:, 1])]
    return np.vstack([here, self.points[self.along > self.at]])

  def lanes_at(self, ahead: float, after: bool = False) -> int:
    """The lanes the car's way just before (after: just past) a point `ahead` m on, from the nearest link with them."""
    k = int(np.searchsorted(self.along, self.at + ahead + (1.0 if after else -1.0), side='right')) - 1
    ks = range(max(k, 0), len(self.links)) if after else range(min(k, len(self.links) - 1), -1, -1)
    s = self.at + ahead
    for j in ks:
      if max(self.along[j] - s, s - self.along[j + 1], 0.0) > LANES_NEAR:  # the segment, however long, from the point
        break
      sec = self.section(j)
      if sec is not None and sec.lanes:
        return sec.lanes
    return 0

  def lane_line(self, keys: list[tuple[float, float]]) -> np.ndarray | None:
    """The route on from the car moved into the lanes keys gives ([(m on, lane from the left)], a ramp between each
    two), by each segment's own lanes (RouteLanes.lane_line: evenly across segments without, fillets through corners)."""
    lanes = self.lanes
    return lanes.lane_line(self.at, keys) if lanes is not None else None

  def _section_here(self) -> Section | None:
    """The lanes at the car, as they are there: a turn bay only once open, so nav never changes towards one before."""
    lanes = self.lanes
    sec = lanes.opened_at(self.at, self.seg) if lanes is not None and 0 <= self.seg < len(lanes.sections) else None
    return sec if sec is not None and sec.lanes else None

  def lane(self) -> list[int] | None:
    """The car's lane, as the plugin reports it: [i from the left, of n], i negative in the oncoming lanes. None where
    the car isn't heading along the route's way, as where GTA's links jog sideways between roads' lines."""
    sec = self._section_here()
    if sec is None or self.misaligned > LANE_ALIGN:
      return None
    return [sec.lane(self.right), sec.lanes]

  def lane_frac(self) -> float | None:
    """The car's lane as lane() gives it, but between lanes as it moves across: 0.0 the middle of the leftmost."""
    sec = self._section_here()
    if sec is None or self.misaligned > LANE_ALIGN:
      return None
    return round(sec.frac(self.right), 2)

  def two_way(self) -> bool | None:
    sec = self.section(self.seg)
    return None if sec is None else sec.two_way

  def changes(self, values, distance: float) -> list[list[float]]:
    """[[m ahead, value], ...] where a per-segment value changes within `distance` m, from the car's segment on."""
    out: list[list[float]] = []
    last = None
    for k in range(self.seg, len(values)):
      d = max(0.0, float(self.along[k]) - self.at)
      if d > distance:
        break
      if values[k] != last:
        out.append([round(d, 1), values[k]])
        last = values[k]
    return out

  def info(self, distance: float) -> dict:
    """What nav uses of the route ahead, beyond its points, within `distance` m."""
    return {
      "routeEnd": round(self.length - self.at, 1),
      "forks": [[round(f.along - self.at, 1), f.side, f.lanes, f.lanes_in, f.keep, f.other, f.slip] for f in self.forks
                if -FORK_BEHIND < f.along - self.at < distance],
      "limits": self.changes(self.limit_list, distance),
      "laneCounts": self.changes(self.lane_counts, distance),
      "stops": [round(a - self.at, 1) for a in self.stops if -JUNCTION_BEHIND < a - self.at < distance],
      "stopKinds": [k for a, k in zip(self.stops, self.stop_kinds, strict=True) if -JUNCTION_BEHIND < a - self.at < distance],
      "junctions": [round(a - self.at, 1) for a in self.junctions if -JUNCTION_BEHIND < a - self.at < distance],
      "laneArrows": self.lane_arrows(distance),
      "laneDrops": self.lane_drops(distance),
      "laneMaps": self.lane_maps(distance),
      "turns": self.turns(distance),
    }

  def lane_arrows(self, distance: float, behind: float = JUNCTION_BEHIND) -> list:
    """The turn arrows of the lanes into each junction within `distance` m: [[m ahead to where they end, each lane's
    from the left, ';'-separated[, the route's move through the junction as they call it, where known]]]."""
    arrows = self.lanes.arrows_moves if self.lanes is not None else []
    return [[round(a[0] - self.at, 1), [";".join(sorted(t)) for t in a[1]], *a[2:]] for a in arrows if -behind < a[0] - self.at < distance]

  def lane_opens(self, distance: float, behind: float = JUNCTION_BEHIND) -> list:
    """Where lanes begin on the left of ours within `distance` m (RouteLanes.openings): [[m ahead, how many]]."""
    opens = self.lanes.openings if self.lanes is not None else []
    return [[float(s - self.at), n] for s, n in opens if -behind < s - self.at < distance]

  def turns(self, distance: float) -> list:
    """The route's turns within `distance` m (navd planner.find_turn along its own shape, once, so each holds still as
    the car drives on): [[m ahead, side, exit heading (game deg), deg turned]]."""
    if self._turns is None:
      from openpilot.selfdrive.navd.planner import MIN_AHEAD_MAP, TURN_HOLDS, find_turn
      self._turns = []
      # its own points, a long segment split so that find_turn's window always holds the next
      seg = np.diff(self.along)
      s = np.concatenate([[0.0]] + [a + np.linspace(0.0, d, int(np.ceil(d / TURN_STEP)) + 1)[1:]
                                    for a, d in zip(self.along[:-1], seg, strict=True) if d > 1e-6])
      pts = np.stack([np.interp(s, self.along, self.points[:, 0]), np.interp(s, self.along, self.points[:, 1])], axis=1)
      turn = find_turn(pts, MIN_AHEAD_MAP) if len(pts) >= 2 else None
      while turn is not None:
        self._turns.append(turn)
        turn = find_turn(pts, turn.dist + TURN_HOLDS)
    return [[round(t.dist - self.at, 1), t.side, round(float(t.exit_heading), 1), round(float(t.angle), 1)] for t in self._turns
            if 0.0 < t.dist - self.at < distance]

  def lane_maps(self, distance: float, here: bool = True) -> list:
    """Where our lanes change within `distance` m (RouteLanes.lane_maps), the lane each before carries on as: [[m ahead,
    [for each lane before (from the left), the lane after, None where it ends], how many after]]. At a fork in the
    road the route's branch takes the lanes on its side, as many as it has (its ways start at the fork's node, so where
    their lanes lie says nothing). With `here`, first (at 0 m) the map from the car's lanes as lane() numbers them to
    the road's at the car, where a lane still opening there isn't one of lane()'s."""
    out = [[float(s - self.at), list(m), n] for s, m, n in self.lane_maps_along() if 0.0 < s - self.at < distance]
    conv = self._lane_here() if here else None
    return ([[0.0, list(conv[0]), conv[1]]] if conv is not None else []) + out

  def lane_maps_along(self) -> list[tuple[float, tuple, int]]:
    """lane_maps along the whole route: [(m along, the lane each lane before carries on as, how many after)]."""
    if self._maps is None:
      self._maps = []
      for s, m, n in self.lanes.lane_maps if self.lanes is not None else []:
        fork = next((f for f in self.forks if abs(f.along - s) < FORK_AT), None)
        if fork is not None and n <= len(m):
          shift = len(m) - n if fork.side == "right" else 0
          m = tuple(i - shift if 0 <= i - shift < n else None for i in range(len(m)))
        self._maps.append((s, m, n))
    return self._maps

  def _lane_here(self) -> tuple[tuple[int, ...], int] | None:
    """The map from lane()'s lanes to the road's at the car, where a lane still opening isn't one of lane()'s."""
    lanes = self.lanes
    if lanes is None or not 0 <= self.seg < len(lanes.sections):
      return None
    opening = lanes.opening(self.seg)
    sec = lanes.section_at(self.at, self.seg)
    if opening is None or self.at >= opening[0] or sec is None:
      return None
    _, extra, left = opening
    return tuple(i + extra if left else i for i in range(sec.lanes - extra)), sec.lanes

  def lane_drops(self, distance: float, behind: float = JUNCTION_BEHIND) -> list:
    """The junctions within `distance` m the route goes straight on through onto fewer lanes (RouteLanes.drops):
    [[m ahead, first and last lane into it that carry on, from the left, of how many]]."""
    drops = self.lanes.drops if self.lanes is not None else []
    return [[round(s - self.at, 1), lo, hi, n] for s, (lo, hi, n) in drops if -behind < s - self.at < distance]


class Router:
  def __init__(self, url: str, timeout: float = 2.0, paths: Paths | None = None, osm: OsmLanes | None = None,
               roads: OsmLanes | None = None):
    self.url = url.rstrip('/')
    self.timeout = timeout
    self.paths = paths
    self.osm = osm  # the map's lanes, where it tags them
    self.roads = roads  # the map's roads, tagged with lanes or not, for destinations (else osm's)
    self._snapper: DestinationSnapper | None = None
    self._snapped: tuple[tuple[float, float], Snap | None] | None = None

  def _post(self, action: str, request: dict) -> dict:
    req = urllib.request.Request(f"{self.url}/{action}", data=json.dumps(request).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=self.timeout) as r:
      return json.loads(r.read())

  def snap(self, dest: np.ndarray) -> Snap | None:
    """Where on the map's roads to arrive for dest (dest_snap.py); None without the map's roads."""
    roads = self.roads if self.roads is not None else self.osm
    if not DEST_SNAP or roads is None:
      return None
    key = (float(dest[0]), float(dest[1]))
    if self._snapper is None or self._snapper.osm is not roads:
      self._snapper, self._snapped = DestinationSnapper(roads), None
    if self._snapped is None or self._snapped[0] != key:
      self._snapped = (key, self._snapper.snap(dest))
    return self._snapped[1]

  def route(self, pos: np.ndarray, bearing: float, dest: np.ndarray, z: float | None = None) -> Route:
    """From the car, setting off the way it faces (bearing clockwise from north), to dest. Given the car's height,
    from the road it's on, not one passing over or under it."""
    def location(p, **kw):
      lat, lon = to_lat_lon(float(p[0]), float(p[1]))
      return {'lat': lat, 'lon': lon, **kw}
    start = location(pos, heading=round(bearing) % 360, heading_tolerance=HEADING_TOLERANCE)
    snapped = self.paths.snap(pos, z, -bearing) if self.paths is not None and z is not None else None
    if snapped is not None:
      # from the next node of the road at the car's height, as the roads leaving a node are all at its height, where a
      # point on the road could find another passing over or under it nearer; the route then starts back at the car
      start = location(self.paths.xy[snapped[2]], heading=round(-snapped[1]) % 360, heading_tolerance=SNAP_TOLERANCE,
                       node_snap_tolerance=1.0)
    request = {
      'costing': 'auto',
      # car parks, alleys and drives (GTA's nodes off for traffic or without GPS), which the model doesn't take for roads
      'costing_options': {'auto': {'service_penalty': SERVICE_PENALTY, 'service_factor': SERVICE_FACTOR}},
      'directions_options': {'units': 'kilometers'},
    }
    # the road the destination faces, arriving its way, else on that road either way, else the router's nearest road
    snap = self.snap(dest)
    ends = [location(dest)]
    if snap is not None and not snap.as_is:
      # on that road's line, not at a node where the router could take another road meeting there
      on_road = location(snap.point, node_snap_tolerance=0)
      ends = [on_road, *ends]
      if snap.heading is not None:
        ends.insert(0, {**on_road, 'heading': round(snap.heading) % 360, 'heading_tolerance': DEST_HEADING_TOLERANCE})
    trip = None
    for n, end in enumerate(ends):
      try:
        trip = self._post('route', {**request, 'locations': [start, end]})['trip']
        break
      except urllib.error.HTTPError:
        if n == len(ends) - 1:
          raise
    assert trip is not None
    if snap is not None and not snap.as_is and snap.kerb_side and n == 0:
      try:  # either way, if the kerb side means going a long way round
        either = self._post('route', {**request, 'locations': [start, ends[1]]})['trip']
        if either['summary']['time'] + SIDE_DETOUR < trip['summary']['time']:
          trip = either
      except urllib.error.HTTPError:
        pass
    points, maneuvers, limits = [], [], []
    for leg in trip['legs']:
      base = len(points)
      leg_points = [to_game(lat, lon) for lat, lon in decode_polyline(leg['shape'])]
      points += leg_points
      maneuvers += [{**m, 'begin_shape_index': m['begin_shape_index'] + base} for m in leg['maneuvers']]
      limits.append(self._limits(leg['shape'], len(leg_points)))
      if base:
        limits[-2] = np.append(limits[-2], 0.0)  # the segment joining the legs
    limits = np.concatenate(limits) if limits else np.zeros(0)
    if snapped is not None and points and np.hypot(*(np.array(points[0]) - self.paths.xy[snapped[2]])) < 1.5:
      points.insert(0, tuple(snapped[0]))
      maneuvers = [{**m, 'begin_shape_index': m['begin_shape_index'] + 1} for m in maneuvers]
      limits = np.concatenate((limits[:1], limits))
    return Route(np.array(points, dtype=float), maneuvers, self.paths, limits, self.osm)

  def _limits(self, shape: str, n: int) -> np.ndarray:
    """The map's speed limit (m/s, 0 where it has none) along each segment of a route's shape."""
    out = np.zeros(max(n - 1, 0))
    try:
      trace = self._post('trace_attributes', {
        'encoded_polyline': shape, 'shape_match': 'walk_or_snap', 'costing': 'auto',
        'filters': {'attributes': ['edge.speed_limit', 'edge.begin_shape_index', 'edge.end_shape_index', 'matched.edge_index'],
                    'action': 'include'},
      })
    except urllib.error.HTTPError as e:
      print(f"router: no speed limits: {e}: {e.read()[:200]!r}")
      return out
    except (OSError, ValueError) as e:
      print(f"router: no speed limits: {e}")
      return out
    edges = trace.get('edges', [])
    limits = [e.get('speed_limit') or 0 for e in edges]
    limits = [v / 3.6 if 0 < v < 250 else 0.0 for v in limits]  # km/h; Valhalla says 255 for unlimited
    if 'matched_points' in trace:  # map matched: each point's edge
      for k, m in enumerate(trace['matched_points'][:len(out)]):
        e = m.get('edge_index')
        out[k] = limits[e] if e is not None and e < len(edges) else 0.0
    else:  # the route's own edges, along its shape
      for e, v in zip(edges, limits, strict=True):
        out[e.get('begin_shape_index', 0):e.get('end_shape_index', 0)] = v
    return out


class Navigator:
  """Keeps a route from the car to a destination, routing again when the destination moves or the car leaves the
  route. Routing runs on its own thread, so update() never waits on the server."""
  OFF_ROUTE = 15.0  # m from the route
  OFF_FOR = 1.5  # s
  RETRY = 3.0  # s between attempts

  def __init__(self, router: Router):
    self.router = router
    self.route: Route | None = None
    self.dest: np.ndarray | None = None
    self.off_since: float | None = None
    self.next_try = 0.0
    self.busy = False
    self.lock = threading.Lock()

  def update(self, pos: np.ndarray, bearing: float, dest: np.ndarray | None, now: float, z: float | None = None,
             match=None) -> Route | None:
    """pos and bearing (clockwise from north) are the car's; with a map match (navd's map_match.Match, from GNSS), the
    car is where the match puts it on its road instead, heading that road's way if the match is sure of the direction
    (else only its distance from the route counts), and without a height."""
    if dest is None:
      self.dest, self.route = None, None
      return None
    if self.dest is None or np.hypot(*(dest - self.dest)) > 1.0:
      self.dest, self.route, self.next_try = dest, None, 0.0
    heading: float | None = -bearing
    if match is not None:
      pos, z, heading = np.asarray(match.point, float), None, match.heading if match.sure else None
      bearing = bearing if heading is None else -heading
    with self.lock:
      route = self.route
    if route is not None:
      off = route.locate(pos, z, heading)
      self.off_since = None if off < self.OFF_ROUTE or route.on_road() else (self.off_since or now)
      if self.off_since is not None and now - self.off_since > self.OFF_FOR and now >= self.next_try:
        self._start(pos, bearing, dest, now, z)
    elif now >= self.next_try:
      self._start(pos, bearing, dest, now, z)
    return route

  def _start(self, pos: np.ndarray, bearing: float, dest: np.ndarray, now: float, z: float | None):
    if self.busy:
      return
    self.busy, self.next_try = True, now + self.RETRY
    threading.Thread(target=self._route, args=(pos.copy(), bearing, dest.copy(), z), daemon=True).start()

  def _route(self, pos: np.ndarray, bearing: float, dest: np.ndarray, z: float | None):
    try:
      route = self.router.route(pos, bearing, dest, z)
      route.locate(pos, z, -bearing)
      with self.lock:
        if self.dest is not None and np.hypot(*(dest - self.dest)) <= 1.0:
          self.route, self.off_since = route, None
    except (OSError, ValueError, KeyError) as e:
      print(f"router: {e}")
    finally:
      self.busy = False
