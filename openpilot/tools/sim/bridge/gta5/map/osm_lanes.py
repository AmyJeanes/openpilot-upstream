"""Lanes from OpenStreetMap tags as real mappers tag them: each way's cross-section (its lanes left to right, how wide
they are and where the way's line runs among them), the lines painted between them, and their centre lines. It reads
only standard tags, so it works the same on our GTA V map (ynd_to_osm.py writes them) as on real OSM.

The tags (OSM wiki: Lanes, Key:turn, Key:width:lanes, Key:change, Key:divider, Proposed features/placement):
- `lanes`, `lanes:forward`, `lanes:backward`, `lanes:both_ways`: car lanes, turn lanes included. A two-way way with
  `lanes=1` is a single-track road, its one lane used both ways; `lanes:both_ways=1` on a wider one is a centre turn lane.
- `turn:lanes`, `width:lanes`, `change:lanes`, `destination:lanes`, `access:lanes`, ...: a value per lane, separated by
  `|`, left to right in the way's direction. `:forward` / `:backward` / `:both_ways` give one direction's lanes, the
  backward ones left to right as seen travelling backward; without a suffix on a two-way way, all its lanes.
- `placement`, `placement:forward`, `placement:backward` = `left_of:N` / `middle_of:N` / `right_of:N` / `transition`:
  where the way's line runs, by that direction's lanes numbered from the left (backward ones as seen travelling
  backward). Without it the line is the middle of the road. Given both ways, it's midway between the two: they differ by
  the median between the directions.
- `width`: kerb to kerb, metres, parking lanes on the carriageway included. Without `width:lanes` it's shared out
  evenly; on a two-way way, what's left after `width:lanes` is a median centred between the directions.
- `parking:left|right|both=lane` (with `parking:<side>:width`, else by `:orientation`): a parking lane on the
  carriageway between that kerb and the lanes; it isn't a lane. Without placement the line is the middle of the lanes
  (and median) between the parking lanes.
- `divider`: the marking between the directions, or both edges of a median between them (each one solid line by
  default). `divider:forward` / `divider:backward`: a median's edge beside that direction's lanes, where its two edges
  differ. `lane_markings=no`: no lines between lanes at all.
- Missing tags fall back to OSM's defaults, then to `Defaults` by road class: lanes 1 each way (2 on a one-way motorway
  or trunk), a single track on tracks, the line in the middle.

Geometry is in metres in a frame with y 90 degrees left of x (x east, y north): "right" is right of the direction of
travel. Traffic drives on the right unless `drive_on_right=False`, which mirrors where each direction's lanes are.
"""
import bisect
import math
import re
from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np

FORWARD, BACKWARD, BOTH_WAYS = 1, -1, 0
TURNS = frozenset({'left', 'slight_left', 'sharp_left', 'through', 'right', 'slight_right', 'sharp_right', 'reverse',
                   'merge_to_left', 'merge_to_right', 'none'})
# change:lanes: whether a lane's traffic may change to the lane on its left, on its right
CHANGES = {'yes': (True, True), 'no': (False, False), 'not_left': (False, True), 'not_right': (True, False),
           'only_left': (True, False), 'only_right': (False, True)}
DIVIDERS = {'no': None, 'dashed_line': 'dashed', 'solid_line': 'solid', 'double_solid_line': 'double_solid',
            'solid_line;dashed_line': 'solid_dashed', 'dashed_line;solid_line': 'dashed_solid'}
PER_LANE = ('turn', 'width', 'change', 'destination', 'destination:ref', 'access', 'motor_vehicle', 'vehicle', 'bus', 'psv')
SUFFIXES = ('', ':forward', ':backward', ':both_ways')
PLACEMENT = re.compile(r'(left_of|middle_of|right_of):([1-9][0-9]*)')
TRANSITION_MAX = 40.0  # m: placement=transition is for the short way where a line jumps across lanes
UNMARKED = {'residential', 'unclassified', 'service', 'track', 'living_street'}  # without lanes=*, no lines painted
EPS = 0.01  # m

# line kinds: a road's edge (kerb), between lanes one way, between the directions, a median's edge, and where a parking
# lane on the carriageway meets the lanes
EDGE, DIVIDER, CENTRE, MEDIAN, PARKING = 'edge', 'divider', 'centre', 'median', 'parking'
PARKING_LANE = {'parallel': 2.3, 'diagonal': 4.5, 'perpendicular': 5.0}  # m wide by orientation, where no width is mapped


class Defaults:
  """Lane widths (m) for ways with neither width:lanes nor width, by highway class."""
  def __init__(self, lane: float, by_class: dict[str, float] | None = None):
    self.lane, self.by_class = lane, by_class or {}

  def lane_width(self, highway: str) -> float:
    return self.by_class.get(highway.removesuffix('_link'), self.lane)


GTA = Defaults(5.5)  # CodeWalker's lane width, where the game's cars drive (ynd_to_osm.py writes width everywhere)
UK = Defaults(3.3, {'motorway': 3.65, 'trunk': 3.65, 'primary': 3.65})
US = Defaults(3.6)


@dataclass(frozen=True)
class Lane:
  direction: int  # FORWARD / BACKWARD along the way; BOTH_WAYS for a centre turn lane or a single track's one lane
  width: float  # m
  turns: frozenset[str] = frozenset()  # turn:lanes indications; empty where none are marked
  change_left: bool = True  # may change to the lane on its left, in its own direction of travel
  change_right: bool = True
  access: bool = True  # open to cars (not a bus lane or access:lanes=no)
  destination: str | None = None


class Span(NamedTuple):
  lane: Lane
  left: float  # m right of the way's line, seen in the direction of travel
  right: float
  heading: int  # 1 travelling our way, -1 oncoming, 0 used both ways

  @property
  def centre(self) -> float:
    return (self.left + self.right) / 2


class Line(NamedTuple):
  kind: str  # EDGE, DIVIDER, CENTRE or MEDIAN
  offset: float  # m right of the way's line, seen in the direction of travel
  style: str | None  # 'dashed', 'solid', 'double_solid', or 'dashed_solid' / 'solid_dashed' by halves from the left; None for an edge


def count(value: str | None) -> int | None:
  return int(value) if value is not None and value.strip().isdigit() else None


def metres(value: str | None) -> float | None:
  """A width: metres by default, also in m, km, ft, or feet and inches (12'6")."""
  if value is None:
    return None
  v = value.strip()
  if m := re.fullmatch(r"(\d+(?:\.\d+)?)\s*(m|km|ft|mi)?", v):
    scale = {'m': 1.0, None: 1.0, 'km': 1000.0, 'ft': 0.3048, 'mi': 1609.344}[m.group(2)]
    return float(m.group(1)) * scale
  if m := re.fullmatch(r"(\d+)'(?:\s*(\d+(?:\.\d+)?)\")?", v):
    return int(m.group(1)) * 0.3048 + float(m.group(2) or 0) * 0.0254
  return None


def oneway_of(tags: dict) -> int:
  """1 along the way, -1 against it, 0 both ways."""
  v = tags.get('oneway')
  if v in ('yes', 'true', '1'):
    return 1
  if v in ('-1', 'reverse'):
    return -1
  if v is None and (tags.get('highway') == 'motorway' or tags.get('junction') in ('roundabout', 'circular')):
    return 1
  return 0


def parking_lane(tags: dict, side: str) -> float:
  """How wide the parking lane on the carriageway on one side of the way ('left' or 'right', seen along it) is:
  `parking:<side>=lane` (or `parking:both`), `parking:<side>:width` wide, else by its orientation; 0 for none, or for
  parking off the carriageway (street_side bays, on the kerb, a shoulder)."""
  def tag(key):
    return tags.get(f'parking:{side}{key}', tags.get(f'parking:both{key}'))
  if tag('') != 'lane':
    return 0.0
  return metres(tag(':width')) or PARKING_LANE.get(tag(':orientation') or 'parallel', PARKING_LANE['parallel'])


def lane_counts(tags: dict) -> tuple[int, int, int]:
  """Lanes (forward, backward, both ways). A single track road is (0, 0, 1)."""
  highway, oneway = tags.get('highway', ''), oneway_of(tags)
  lanes, fwd, back, both = (count(tags.get(k)) for k in ('lanes', 'lanes:forward', 'lanes:backward', 'lanes:both_ways'))
  if lanes == 0:
    lanes = None
  if oneway:
    n = lanes or (fwd if oneway == 1 else back) or (2 if highway in ('motorway', 'trunk') else 1)
    return (n, 0, 0) if oneway == 1 else (0, n, 0)
  both = both or 0
  if fwd is None and back is None:
    if lanes is None:
      return (0, 0, 1) if highway == 'track' else (1, 1, both)
    rest = lanes - both
    if rest <= 1 and not both:
      return 0, 0, 1
    back = rest // 2
    return rest - back, back, both
  if fwd is None:
    fwd = max(lanes - back - both, 0) if lanes is not None else 1
  if back is None:
    back = max(lanes - fwd - both, 0) if lanes is not None else 1
  if fwd == back == 0 and not both:
    both = 1
  return fwd, back, both


class WayLanes:
  """A way's lanes, left to right as seen along it."""
  def __init__(self, lanes: list[Lane], width: float, median: float, margin: float, markings: bool, divider: str | None,
               parking: tuple[float, float] = (0.0, 0.0)):
    self.lanes = lanes
    self.width = width  # m, kerb to kerb
    self.parking = parking  # m of parking lane on the carriageway between the left and the right kerb and the lanes
    self.line = parking[0] + (width - parking[0] - parking[1]) / 2  # m from the left kerb to the way's line
    self.margin = margin  # m between each kerb (or parking lane) and its outer lane
    self.markings, self.divider = markings, divider
    self.median_edges: tuple[str | None, str | None] = (None, None)  # divider:forward, divider:backward
    self.placed: dict[str, float] = {}  # where each placement tag puts the line, m from the left kerb
    self.tagged: tuple[float | None, float, int] = (None, 0.0, 0)  # width=*, the lanes' width:lanes total, lanes without
    self.single_track = all(lane.direction == BOTH_WAYS for lane in lanes)
    turns = [i for i in range(1, len(lanes)) if lanes[i - 1].direction != lanes[i].direction]
    x, self.x, self.gaps = parking[0] + margin, [], []  # each lane's left edge and the median (either side of any centre lanes)
    for i, lane in enumerate(lanes):
      if median and i in turns:
        self.gaps.append((x, x + median / len(turns)))
        x += median / len(turns)
      self.x.append(x)
      x += lane.width

  @classmethod
  def from_tags(cls, tags: dict, drive_on_right: bool = True, defaults: Defaults = GTA, at: str = '') -> 'WayLanes':
    """`at` 'start' or 'end': the lanes at the way's first or last node, where width:lanes(:forward|:backward):start
    / :end (and width:start / :end) say they widen or narrow along it, as a lane opening in a taper."""
    highway, oneway = tags.get('highway', ''), oneway_of(tags)
    fwd, back, both = lane_counts(tags)
    backs = [(BACKWARD, k) for k in reversed(range(back))]  # seen along the way, backward lanes run from the far one
    order = [*backs, *((BOTH_WAYS, k) for k in range(both)), *((FORWARD, k) for k in range(fwd))] if drive_on_right else \
      [*((FORWARD, k) for k in range(fwd)), *((BOTH_WAYS, k) for k in range(both)), *backs]
    own = {d: {k: i for i, (dd, k) in enumerate(order) if dd == d} for d in (FORWARD, BACKWARD, BOTH_WAYS)}

    def get(key):  # a width at the way's end where it says one
      return tags.get(f'{key}:{at}', tags.get(key)) if at else tags.get(key)

    def per_lane(key):
      out: list[str | None] = [None] * len(order)
      plain = get(f'{key}:lanes') if key == 'width' else tags.get(f'{key}:lanes')
      if plain is not None:
        items = plain.split('|')
        if oneway:
          pos = own[FORWARD if oneway == 1 else BACKWARD]
          if len(items) == len(pos):
            for k, v in enumerate(items):
              out[pos[k]] = v
        elif len(items) == len(order):
          out = list(items)
      for suffix, d in ((':forward', FORWARD), (':backward', BACKWARD), (':both_ways', BOTH_WAYS)):
        v = get(f'{key}:lanes{suffix}') if key == 'width' else tags.get(f'{key}:lanes{suffix}')
        if v is not None and len(items := v.split('|')) == len(own[d]):
          for k, item in enumerate(items):
            out[own[d][k]] = item
      return out

    widths = [metres(v) for v in per_lane('width')]
    parking = (parking_lane(tags, 'left'), parking_lane(tags, 'right'))
    width = metres(get('width'))
    if width:
      width = max(width - sum(parking), 0.0)  # the carriageway's width includes its parking lanes
    known, unknown = sum(w for w in widths if w is not None), sum(w is None for w in widths)  # a lane opening from 0 m
    if unknown:
      share = (width - known) / unknown if width and width - known > EPS * unknown else defaults.lane_width(highway)
      widths = [share if w is None else w for w in widths]
    lanes_w = sum(widths)
    spare = width - lanes_w if width and width - lanes_w > EPS else 0.0
    width = lanes_w + spare + sum(parking)
    two_way = bool(fwd or both) and bool(back or both) and not (fwd == back == 0)
    turns, changes, destinations = per_lane('turn'), per_lane('change'), per_lane('destination')
    closed = [any(v in ('no', 'private') for v in vs) for vs in zip(*(per_lane(k) for k in ('access', 'motor_vehicle', 'vehicle')), strict=True)]
    bus = [any(v == 'designated' for v in vs) for vs in zip(per_lane('bus'), per_lane('psv'), strict=True)]
    lanes = []
    for i, (d, _) in enumerate(order):
      left, right = CHANGES.get(changes[i] or 'yes', (True, True))
      lanes.append(Lane(d, widths[i], frozenset(t for t in (turns[i] or '').split(';') if t in TURNS and t != 'none'),
                        left, right, not (closed[i] or bus[i]), destinations[i] or None))
    markings = tags.get('lane_markings') != 'no' and not (tags.get('lanes') is None and highway in UNMARKED)
    road = cls(lanes, width, spare if two_way else 0.0, 0.0 if two_way else spare / 2, markings, tags.get('divider'), parking)
    road.median_edges = (tags.get('divider:forward'), tags.get('divider:backward'))

    def place(value, d):  # m from the left kerb
      m = PLACEMENT.fullmatch(value or '')
      if not m:
        return None
      pos = own[d] if d is not None else {k: k for k in range(len(order))}
      n = int(m.group(2)) - 1
      if n not in pos:
        return None
      i = pos[n]
      left, right = road.x[i], road.x[i] + widths[i]
      if d == BACKWARD:
        left, right = right, left
      return {'left_of': left, 'middle_of': (left + right) / 2, 'right_of': right}[m.group(1)]

    plain_dir = (FORWARD if oneway == 1 else BACKWARD) if oneway else None
    for key, d in (('placement', plain_dir), ('placement:forward', FORWARD), ('placement:backward', BACKWARD)):
      if (at := place(tags.get(key), d)) is not None:
        road.placed[key] = at
    if road.placed:
      road.line = sum(road.placed.values()) / len(road.placed)
    tagged = metres(tags.get('width'))
    road.tagged = (tagged - sum(parking) if tagged else None, known, unknown)  # width=* less its parking lanes
    return road

  @property
  def counts(self) -> tuple[int, int, int]:
    return tuple(sum(lane.direction == d for lane in self.lanes) for d in (FORWARD, BACKWARD, BOTH_WAYS))

  def section(self, direction: int = FORWARD) -> list[Span]:
    """The whole road left to right, seen travelling `direction` along the way, oncoming lanes included."""
    out = []
    for lane, x in zip(self.lanes, self.x, strict=True):
      heading = 0 if lane.direction == BOTH_WAYS else (1 if lane.direction == direction else -1)
      left, right = x - self.line, x + lane.width - self.line
      out.append(Span(lane, left, right, heading) if direction == FORWARD else Span(lane, -right, -left, heading))
    return out if direction == FORWARD else out[::-1]

  def ours(self, direction: int = FORWARD) -> list[Span]:
    """The lanes travelling `direction`, left to right; a single track's one lane both ways."""
    return [s for s in self.section(direction) if s.heading == 1 or (self.single_track and s.heading == 0)]

  def oncoming(self, direction: int = FORWARD) -> list[Span]:
    return [s for s in self.section(direction) if s.heading == -1]

  def edges(self, direction: int = FORWARD) -> tuple[float, float]:
    """The kerbs, m right of the line."""
    return (-self.line, self.width - self.line) if direction == FORWARD else (self.line - self.width, self.line)

  def parking_lanes(self, direction: int = FORWARD) -> list[tuple[float, float]]:
    """The parking lanes on the carriageway, m right of the line, left to right seen travelling `direction`."""
    lo, hi = self.edges(direction)
    left, right = self.parking if direction == FORWARD else self.parking[::-1]
    return [s for s in ((lo, lo + left), (hi - right, hi)) if s[1] - s[0] > EPS]

  def medians(self, direction: int = FORWARD) -> list[tuple[float, float]]:
    """The median between the directions, m right of the line: in two halves either side of a centre turn lane."""
    out = [(a - self.line, b - self.line) for a, b in self.gaps]
    return out if direction == FORWARD else [(-b, -a) for a, b in out[::-1]]

  def lines(self, direction: int = FORWARD) -> list[Line]:
    """The lines on the road left to right: its edges, white lines between lanes one way (solid where change:lanes
    forbids crossing), and the centre line between the directions (divider=*; by default dashed with one lane each way,
    else double solid) or the edges of a median (divider=*, divider:forward / :backward; one solid line by default), and
    where a parking lane meets the lanes."""
    return [line for _, line in self.keyed_lines(direction)]

  def keyed_lines(self, direction: int = FORWARD, gaps=frozenset()) -> list[tuple[tuple, Line]]:
    """lines(), each with where it runs on the section (line_offset): ('edge', 0 | 1), ('parking', 0 | 1), ('lane', i)
    on the right of span i, ('median', i, 0 | 1) the edges of a median between spans i and i + 1. `gaps`: the spans i
    to give median edges where there's no median, as on a road whose median opens or closes along it (a taper)."""
    sec = self.section(direction)
    lo, hi = self.edges(direction)
    parking = self.parking_lanes(direction)
    out = [(('edge', 0), Line(EDGE, lo, None))] + [(('parking', 0), Line(PARKING, b, None)) for a, b in parking if a == lo]
    if self.markings:
      wide = max(self.counts[:2]) >= 2
      centre = DIVIDERS.get(self.divider, 'solid') if self.divider else ('double_solid' if wide else 'dashed')
      # a median's edges, beside the oncoming lanes and beside ours
      theirs, ours = self.median_edges if direction == BACKWARD else self.median_edges[::-1]
      edges = [DIVIDERS.get(v, 'solid') if v else 'solid' for v in (theirs or self.divider, ours or self.divider)]
      for i, (a, b) in enumerate(zip(sec, sec[1:], strict=False)):
        if a.heading == b.heading != 0:
          a_may = a.lane.change_right if a.heading == 1 else a.lane.change_left
          b_may = b.lane.change_left if b.heading == 1 else b.lane.change_right
          style = {(True, True): 'dashed', (False, False): 'solid', (True, False): 'dashed_solid', (False, True): 'solid_dashed'}[(a_may, b_may)]
          out.append((('lane', i), Line(DIVIDER, a.right, style)))
        elif 0 in (a.heading, b.heading):  # a centre turn lane's edge: dashed on its side
          out.append((('lane', i), Line(CENTRE, a.right, 'dashed_solid' if a.heading == 0 else 'solid_dashed')))
        elif centre is None:
          continue
        elif b.left - a.right > EPS or i in gaps:
          out += [(('median', i, 0), Line(MEDIAN, a.right, edges[0])), (('median', i, 1), Line(MEDIAN, b.left, edges[1]))]
        else:
          out.append((('lane', i), Line(CENTRE, a.right, centre)))
    out += [(('parking', 1), Line(PARKING, a, None)) for a, b in parking if b == hi]
    out.append((('edge', 1), Line(EDGE, hi, None)))
    return out

  def centres(self, points, direction: int = FORWARD) -> list[np.ndarray]:
    """The centre lines of the lanes travelling `direction` (ours), left to right, from the way's node positions."""
    pts = np.asarray(points, dtype=float)
    pts = pts if direction == FORWARD else pts[::-1]
    return [offset_line(pts, s.centre) for s in self.ours(direction)]

  def line_geometry(self, points, direction: int = FORWARD) -> list[tuple[Line, np.ndarray]]:
    pts = np.asarray(points, dtype=float)
    pts = pts if direction == FORWARD else pts[::-1]
    return [(line, offset_line(pts, line.offset)) for line in self.lines(direction)]


MITER_LIMIT = 4.0  # a corner's offset point goes no further than this many times the offset


def offset_line(points, right) -> np.ndarray:
  """A polyline (n, 2) moved `right` m to its right (one offset, or one per point), mitred at the corners; repeated
  points are dropped."""
  p = np.asarray(points, dtype=float)[:, :2]
  keep = np.concatenate(([True], np.hypot(*np.diff(p, axis=0).T) > 1e-6))
  p = p[keep]
  right = np.asarray(right, dtype=float)
  if right.ndim:
    right = right[keep]
  if len(p) < 2 or not np.any(right):
    return p.copy()
  d = np.diff(p, axis=0)
  d /= np.hypot(d[:, 0], d[:, 1])[:, None]
  n = np.stack([d[:, 1], -d[:, 0]], axis=1)  # right of each segment
  normal = np.concatenate([n[:1], n[:-1] + n[1:], n[-1:]])
  normal /= np.maximum(np.hypot(normal[:, 0], normal[:, 1]), 1e-9)[:, None]
  cos = np.concatenate([[1.0], np.einsum('ij,ij->i', normal[1:-1], n[1:]), [1.0]])
  scale = 1.0 / np.maximum(cos, 1.0 / MITER_LIMIT)
  return p + normal * (right * scale)[:, None]


def mitred(points, right, before=None, after=None) -> np.ndarray:
  """offset_line, its end corners mitred as if the polyline went on to the point `before` its start and `after` its
  end (None: square ends), so that it meets the offset line of the polyline it carries on into."""
  p = np.asarray(points, float)[:, :2]
  if len(p) < 2 or (before is None and after is None):
    return offset_line(p, right)
  r = np.broadcast_to(np.asarray(right, float), (len(p),))
  head, tail = ([before], [r[0]]) if before is not None else ([], []), ([after], [r[-1]]) if after is not None else ([], [])
  out = offset_line(np.vstack([*head[0], p, *tail[0]]), np.concatenate([head[1], r, tail[1]]))
  return out[len(head[0]):len(out) - len(tail[0])]


# *** along a route ***

LEFTS = frozenset({'left', 'slight_left', 'sharp_left'})
RIGHTS = frozenset({'right', 'slight_right', 'sharp_right'})
ROADS = frozenset({'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'service',
                   'living_street', 'track', 'road', 'motorway_link', 'trunk_link', 'primary_link', 'secondary_link',
                   'tertiary_link'})
# tags that place lanes, beyond their counts: a map without any has no lane geometry to read
GEOMETRY = ('width', 'placement', 'placement:forward', 'placement:backward')
TAPER_M = 30.0  # m: a lane begins (or ends) where a way's lane count changes, widening from (narrowing to) nothing over this
CORNER_TURN = 30.0  # deg: a route turning this much within CORNER_CHORD m either side of a point has a corner there
CORNER_CHORD = 10.0  # m
FILLET_REACH = 20.0  # m either side of a corner the lane line is replaced by a fillet
FILLET_MIN = 1.0  # m: a corner nearer the route's start or end than this has no room for a fillet
CORNER_SHARP, SHARP_SHARE = 10.0, 0.6  # a turn through a junction turns this share of its angle within this many m
FILLET_RADIUS = {'left': 20.0, 'right': 12.0}  # m at most, about as wide as GTA's AI drives its turns
MATCH_TOL = 0.5  # m between a route's shape point and a map node it is at
DROP_SPAN = 25.0  # m: junction nodes this near each other along a route are one junction, for the lanes through it
DROP_TURN = 30.0  # deg at most a route turns going straight on through a junction (over DROP_REACH m either side)
DROP_REACH = 15.0  # m
SAME_WIDTH = 0.5  # m: roads either side of a junction this near as wide are one road carrying on, kerb to kerb
# The route's move through a junction, as its turn:lanes call it (junction_move): a way out turning up to MOVE_THROUGH
# goes through, else the nearest one up to MOVE_SKEW where none is within it (a skewed junction's road on, painted
# through); the rest turn. Ways out are looked for across the junction's short straight links (a divided road's far side).
MOVE_THROUGH, MOVE_SKEW, MOVE_U_TURN = 45.0, 55.0, 135.0  # deg
MOVE_SPAN = 20.0  # m across the junction from its node
MOVE_HEADING = 15.0  # m before the junction that the road's heading into it is taken over


class Section:
  """A road's cross-section where a route runs along it, seen in the route's direction of travel: its lanes left to
  right (ours, oncoming and centre lanes), m right of the route's line. Lanes are numbered from the left of ours: 0 to
  lanes - 1 are ours, negative ones left of them (the oncoming lanes where traffic drives on the right). Not changed
  once made (its key is kept)."""
  def __init__(self, spans: list[Span], edges: tuple[float, float]):
    self.spans = spans
    ours = [i for i, s in enumerate(spans) if s.heading == 1] or [i for i, s in enumerate(spans) if s.heading == 0]
    self.first = ours[0] if ours else 0
    self.lanes = len(ours)
    self.back = self.first  # lanes left of ours
    self.lo, self.hi = -self.first, len(spans) - self.first - 1
    self.two_way = any(s.heading == -1 for s in spans) or bool(spans) and all(s.heading == 0 for s in spans)
    self.edges = edges
    self.centres = [s.centre for s in spans]

  @classmethod
  def of(cls, road: WayLanes, direction: int = FORWARD) -> 'Section':
    return cls(road.section(direction), road.edges(direction))

  @property
  def ours(self) -> list[Span]:
    return self.spans[self.first:self.first + self.lanes]

  @property
  def turns(self) -> list[frozenset[str]]:
    return [s.lane.turns for s in self.ours]

  @cached_property
  def key(self) -> tuple:
    """The layout, to tell one road's lanes from another's."""
    return (self.first, *((round(s.left, 2), round(s.right, 2)) for s in self.spans))

  def offset(self, lane: float) -> float:
    """m right of the line to lane `lane`'s centre, between two lanes' centres for a fraction."""
    lane = min(max(float(lane), self.lo), self.hi)
    k = int(np.floor(lane))
    c0 = self.centres[self.first + k]
    c1 = self.centres[min(self.first + k + 1, len(self.centres) - 1)]
    return c0 + (c1 - c0) * (lane - k)

  def frac(self, right: float) -> float:
    """The lane a point `right` m right of the line is in, between lanes as it moves across (0.0 the middle of our
    leftmost lane); beyond ours, by their outer lanes' widths."""
    ours = self.ours
    first, last = ours[0], ours[-1]
    if right < first.left:
      return (right - first.left) / max(first.right - first.left, EPS) - 0.5
    for i, s in enumerate(ours):
      if right <= s.right:
        return i + (right - s.left) / max(s.right - s.left, EPS) - 0.5
    return len(ours) - 0.5 + (right - last.right) / max(last.right - last.left, EPS)

  def lane(self, right: float) -> int:
    return int(min(max(np.floor(self.frac(right) + 0.5), self.lo), self.hi))


def continuing(into: Section, out: Section) -> list[int]:
  """Which of our lanes into a junction (from the left) carry on into the road out of it straight across: those whose
  middle falls in one of the road out's lanes our way. Where the road is as wide on both sides its kerbs carry on, and
  its line jogs if the split between the directions moves (one direction's lane ending as the other's begins);
  otherwise its line carries on."""
  (lo_in, hi_in), (lo_out, hi_out) = into.edges, out.edges
  same = abs((hi_in - lo_in) - (hi_out - lo_out)) < SAME_WIDTH
  shift = (lo_in - lo_out + hi_in - hi_out) / 2 if same else 0.0
  return [i for i, s in enumerate(into.ours) if any(o.left + shift <= s.centre <= o.right + shift for o in out.ours)]


def turn_targets(turns: list[frozenset[str]], move: str, fork: bool = False) -> list[int]:
  """The lanes (of ours, from the left) whose turn:lanes arrows allow a move: 'left' or 'right' (a turn that way, or
  with `fork`, keeping to that side at a fork or exit, which mappers mark slight_left / slight_right), or 'through'. []
  where no lane's arrows say."""
  if not any(turns):
    return []
  if move == 'through':
    return [i for i, t in enumerate(turns) if 'through' in t or not t]
  want = {f'slight_{move}'} if fork else LEFTS if move == 'left' else RIGHTS
  return [i for i, t in enumerate(turns) if t & want]


class OsmLanes:
  """A map's roads for routes over it: each way's nodes, in metres by `project` ((lat, lon) arrays to (x, y)), and its
  lanes (WayLanes, read once). `tagged` says whether the map places its lanes at all (width or placement on any way, or
  any *:lanes tag), rather than only counting them."""
  CELL = 2.0  # m

  def __init__(self, data, project, drive_on_right: bool = True, defaults: Defaults = GTA):
    x, y = project(np.asarray(data.lat), np.asarray(data.lon))
    self.data, self.drive_on_right, self.defaults = data, drive_on_right, defaults
    self.project = project
    self.path: str | None = None  # the file load() read it from, for caches built from it
    self.ids = data.node_ids
    self.xy = np.stack([np.asarray(x, float), np.asarray(y, float)], axis=1) if len(self.ids) else np.zeros((0, 2))
    self.ways = {w: v for w, v in data.ways.items() if v[0].get('highway') in ROADS}
    self.pairs: dict[tuple[int, int], tuple[int, bool]] = {}  # (node, next node) -> (way, along its direction)
    self.links: dict[int, list[int]] = {}  # node -> the nodes a way joins it to
    self.degree: dict[int, int] = {}  # node -> the ways at it
    self.tagged = False
    for wid, (tags, refs) in self.ways.items():
      for a, b in zip(refs, refs[1:], strict=False):
        self.pairs[(a, b)], self.pairs[(b, a)] = (wid, True), (wid, False)
        self.links.setdefault(a, []).append(b)
        self.links.setdefault(b, []).append(a)
      for n in set(refs):
        self.degree[n] = self.degree.get(n, 0) + 1
      self.tagged = self.tagged or any(k in tags for k in GEOMETRY) or any(':lanes' in k for k in tags)
    self.cells: dict[tuple[int, int], list[int]] = {}
    for k, (px, py) in enumerate(self.xy):
      self.cells.setdefault((int(px // self.CELL), int(py // self.CELL)), []).append(k)
    self._lanes: dict[int, WayLanes] = {}
    self._ends: dict[tuple[int, str], WayLanes] = {}
    self._at: dict[int, list[int]] | None = None  # node -> the ways at it
    self._tapers: dict[int, tuple[int, list[tuple[float, Section]]] | None] = {}
    self._blends: dict[int, tuple[int, list[tuple[float, Section]]] | None] = {}
    self._along: dict[int, np.ndarray] = {}
    self._carried: dict[tuple[int, int], int] = {}  # (way, node) -> the way a road carried across a junction there runs on to

  def carry_across(self, pairs) -> None:
    """Roads whose lines are painted on across a junction (junctions.py's Junction.carried): [(way, way)] meeting end to
    end at a junction node. Their lines and kerbs run on from one to the other as where a road just carries on (blend,
    offset_way), rather than stepping at the node."""
    for a, b in pairs:
      common = {self.ways[a][1][0], self.ways[a][1][-1]} & {self.ways[b][1][0], self.ways[b][1][-1]}
      if len(common) == 1 and a != b:
        node = common.pop()
        self._carried[(a, node)], self._carried[(b, node)] = b, a
    self._blends.clear()

  def _onto(self, way: int, node: int) -> tuple[int, int] | None:
    """_other's way, or at a junction the way the road carried across it runs on to (carry_across)."""
    found = self._other(way, node)
    if found is None and (other := self._carried.get((way, node))) is not None:
      found = other, FORWARD if self.ways[other][1][-1] == node else BACKWARD
    return found

  @classmethod
  def load(cls, path: str, project, **kw) -> 'OsmLanes':
    from openpilot.tools.sim.bridge.gta5.map import osm_pbf
    lanes = cls(osm_pbf.read(path, relations=('connectivity',)), project, **kw)
    lanes.path = path
    return lanes

  def lanes(self, way: int) -> WayLanes:
    if way not in self._lanes:
      self._lanes[way] = WayLanes.from_tags(self.ways[way][0], self.drive_on_right, self.defaults)
    return self._lanes[way]

  def node_xy(self, node: int) -> np.ndarray:
    return self.xy[int(self.data.index([node])[0])]

  def has_ends(self, way: int) -> bool:
    """Whether the way's lanes widen or narrow along it (width:lanes...:start / :end), as in a taper."""
    return any(k.startswith('width') and k.endswith((':start', ':end')) for k in self.ways[way][0])

  def lanes_at(self, way: int, end: str) -> WayLanes:
    """The way's lanes at its first node (`end` 'start') or last ('end')."""
    if not self.has_ends(way):
      return self.lanes(way)
    if (way, end) not in self._ends:
      self._ends[(way, end)] = WayLanes.from_tags(self.ways[way][0], self.drive_on_right, self.defaults, at=end)
    return self._ends[(way, end)]

  def length(self, way: int) -> float:
    return float(np.hypot(*np.diff(self.way_points(way), axis=0).T).sum())

  def _other(self, way: int, node: int) -> tuple[int, int] | None:
    """The one other way at a node where a road just carries on (two ways), and its direction away from the node
    reversed: the direction travelling it into the node (FORWARD where it ends there)."""
    if self._at is None:
      self._at = {}
      for w, (_, refs) in self.ways.items():
        for n in {refs[0], refs[-1]}:
          self._at.setdefault(n, []).append(w)
    if self.degree.get(node) != 2:
      return None
    others = [w for w in self._at.get(node, ()) if w != way]
    if len(others) != 1:
      return None
    return others[0], FORWARD if self.ways[others[0]][1][-1] == node else BACKWARD

  def _boundary(self, way: int, d: int, ahead: bool) -> tuple[float, 'Section'] | None:
    """Back (or on) from a way travelled d, while the road carries on with its layout: how far to where its lanes
    change, and the cross-section beyond (seen travelling d); None past TAPER_M, a junction, or a tapered way."""
    key = Section.of(self.lanes(way), d).key
    w, wd, dist = way, d, 0.0
    for _ in range(64):
      refs = self.ways[w][1]
      node = (refs[-1] if wd == FORWARD else refs[0]) if ahead else (refs[0] if wd == FORWARD else refs[-1])
      found = self._other(w, node)
      if found is None:
        return None
      nb, nd = found
      if ahead:
        nd = -nd  # travelling away from the node
      if self.has_ends(nb):
        return None
      sec = Section.of(self.lanes(nb), nd)
      if sec.key != key:
        return dist, sec
      dist += self.length(nb)
      if dist >= TAPER_M:
        return None
      w, wd = nb, nd
    return None

  def taper(self, way: int) -> tuple[int, list[tuple[float, 'Section']]] | None:
    """Where the way's lanes change along it: (the direction it's seen in, [(m along it that way, its cross-section
    there)], the cross-section varying linearly between). From its width:lanes:start / :end tags; else, a lane a road
    gains (or loses) at a node where it just carries on widens from (narrows to) nothing over TAPER_M from there, also
    across the ways after it with the same lanes. None where its lanes don't change."""
    if way in self._tapers:
      return self._tapers[way]
    out = None
    length = self.length(way)
    if self.has_ends(way):
      out = FORWARD, [(0.0, Section.of(self.lanes_at(way, 'start'))), (length, Section.of(self.lanes_at(way, 'end')))]
    else:
      for d in (FORWARD, BACKWARD):
        many = Section.of(self.lanes(way), d)
        for ahead in (False, True):
          found = self._boundary(way, d, ahead)
          if found is None or not (_taperable(found[1], many) and found[1].lanes < many.lanes):
            continue
          dist, few = found

          def state(along, dist=dist, few=few, many=many, ahead=ahead):
            return _blend(few, many, (dist + (length - along if ahead else along)) / TAPER_M)
          reach = TAPER_M - dist  # m of the way the taper runs into
          knots = [0.0, min(reach, length)] if not ahead else [max(length - reach, 0.0), length]
          knots = sorted({0.0, length, *knots})
          out = d, [(v, state(v)) for v in knots]
          break
        if out:
          break
    self._tapers[way] = out
    return out

  def line_geometry(self, way: int) -> list[tuple['Line', np.ndarray]]:
    """The lines painted along a way (WayLanes.lines), where they run: through a taper (taper()) the lines move with
    the lanes, the median's edge swinging across as a lane opens in it, and the line between an opening lane and the
    next one starts where it has opened. Where the road carries on onto one other way, each line meets that way's:
    mitred with it, and moved across to it over blend()'s span where the two ways' lanes sit apart."""
    road, pts = self.lanes(way), self.way_points(way)
    found = self.taper(way)
    tapered = found is not None
    if found is None:
      found = self.blend(way)
    if found is None:
      return [(line, self.offset_way(way, pts, line.offset)) for line in road.lines(FORWARD)]
    d, knots = found
    p = pts if d == FORWARD else pts[::-1]
    along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))))
    ks = np.array([v for v, _ in knots])
    extra = [v for v in ks if np.abs(along - v).min() > 0.05]
    s2 = np.sort(np.concatenate((along, extra)))
    p2 = np.stack([np.interp(s2, along, p[:, 0]), np.interp(s2, along, p[:, 1])], axis=1)
    secs = [sec for _, sec in knots]
    gaps = {i for sec in secs for i in range(len(sec.spans) - 1) if sec.spans[i + 1].left - sec.spans[i].right > EPS}
    opening = {i for sec in secs for i, sp in enumerate(sec.spans) if sp.right - sp.left < OPENED} if tapered else set()
    out = []
    for key, line in road.keyed_lines(d, gaps):
      if key[0] == 'lane' and line.kind == DIVIDER and {key[1], key[1] + 1} & opening:
        continue
      if key[0] == 'parking':
        offs = np.full(len(s2), line.offset)
      else:
        offs = np.interp(s2, ks, [line_offset(key, sec) for sec in secs])
      out.append((line, self.offset_way(way, p2, offs, d)))
    return out

  # *** where a road carries on from one way to the next ***

  def end_section(self, way: int, node: int, d: int) -> Section:
    """The way's cross-section at its end `node`, seen travelling d along it (a taper's there)."""
    found = self.taper(way)
    if found is None:
      return Section.of(self.lanes(way), d)
    td, knots = found
    sec = knots[0][1] if (self.ways[way][1][0] == node) == (td == FORWARD) else knots[-1][1]
    return sec if td == d else mirrored(sec)

  def _join(self, way: int, node: int, at_end: bool, own: Section) -> tuple[float, float, Section] | None:
    """Where the road carries on at a free (untapered) way's end node onto one other way with the same lanes sitting
    elsewhere: (m of this way the change is spread over, m of the other, its cross-section there seen travelling this
    way FORWARD). None where nothing moves; nothing of a tapered way, whose lines stay as its taper has them."""
    found = self._onto(way, node)
    if found is None or found[0] == way:
      return None
    nb, nd = found  # nd: travelling nb into the node
    other = self.end_section(nb, node, -nd if at_end else nd)
    if not _same_lanes(own, other) or _apart(own, other) < EPS:
      return None
    lb = 0.0 if self.taper(nb) is not None else min(BLEND_M, self.length(nb) / 2)
    return min(BLEND_M, self.length(way) / 2), lb, other

  def blend(self, way: int) -> tuple[int, list[tuple[float, Section]]] | None:
    """Where the road carries on from this way onto another with the same lanes in other places (other widths or
    line, as where the paint survey measured one way and not the next): the lanes move across from one to the other on
    a smoothstep, over up to BLEND_M either side of the node (half of a shorter way), so that the lines and kerbs run on
    without a step. As taper() gives a taper (seen FORWARD); None where neither end moves, and on a tapered way (the
    way beyond takes it all)."""
    if way in self._blends:
      return self._blends[way]
    out = None
    if self.taper(way) is None:
      refs, length = self.ways[way][1], self.length(way)
      own = Section.of(self.lanes(way), FORWARD)
      knots = []
      for node, at_end in ((refs[0], False), (refs[-1], True)):
        found = self._join(way, node, at_end, own)
        if found is None:
          continue
        la, lb, other = found
        for x in np.linspace(0.0, la, BLEND_KNOTS + 1):  # m from the node
          u = (la - x) / (la + lb) if at_end else (x + lb) / (la + lb)  # 0-1 across the whole change, this way first
          knots.append((length - x if at_end else x, _toward(own, other, _smooth(u) if at_end else 1.0 - _smooth(u))))
      if knots:
        ends = [(s, own) for s in (0.0, length) if all(abs(s - v) > EPS for v, _ in knots)]
        out = FORWARD, sorted(knots + ends, key=lambda k: k[0])
    self._blends[way] = out
    return out

  def _beyond(self, way: int, node: int, p: np.ndarray) -> np.ndarray | None:
    """Where the road goes on past the way's end node (p runs from it into the way): the next point of the one other
    way there, None where there's none or it turns back on itself."""
    found = self._onto(way, node)
    if found is None or found[0] == way:
      return None
    refs = self.ways[found[0]][1]
    here = self.node_xy(node)
    away = next((q for q in p[1:] if np.hypot(*(q - p[0])) > 1e-6), None)
    if away is None:
      return None
    for n in (refs[::-1] if refs[-1] == node else refs)[1:]:
      q = self.node_xy(n)
      u, v = here - q, away - here
      if (lu := float(np.hypot(*u))) > 1e-6:
        return q if float(u @ v) / (lu * float(np.hypot(*v))) > -0.5 else None
    return None

  def offset_way(self, way: int, p, right, d: int = FORWARD) -> np.ndarray:
    """offset_line of points p running along a way (in direction d), mitred at its ends with the way the road carries
    on onto there, so the two ways' lines meet."""
    p = np.asarray(p, float)
    refs = self.ways[way][1]
    first, last = (refs[0], refs[-1]) if d == FORWARD else (refs[-1], refs[0])
    return mitred(p, right, self._beyond(way, first, p), self._beyond(way, last, p[::-1]))

  def offset_nodes(self, nodes: list[int], right: float) -> np.ndarray:
    """offset_line of a run of ways' nodes, mitred at its ends with the ways the road carries on onto there."""
    p = self.xy[self.data.index(nodes)]
    if len(nodes) < 2 or nodes[0] == nodes[-1]:
      return offset_line(p, right)
    first, last = self.pairs[(nodes[0], nodes[1])][0], self.pairs[(nodes[-2], nodes[-1])][0]
    return mitred(p, right, self._beyond(first, nodes[0], p), self._beyond(last, nodes[-1], p[::-1]))

  def way_along(self, way: int) -> np.ndarray:
    if way not in self._along:
      self._along[way] = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(self.way_points(way), axis=0).T))))
    return self._along[way]

  def edges_at(self, way: int, s: float, direction: int = FORWARD) -> tuple[float, float]:
    """The kerbs s m along a way from its first node, m right of its line seen travelling `direction`, where
    line_geometry draws them (through tapers and blends)."""
    found = self.taper(way) or self.blend(way)
    if found is None:
      return self.lanes(way).edges(direction)
    d, knots = found
    v = s if d == FORWARD else self.length(way) - s
    ks = [k for k, _ in knots]
    lo, hi = (float(np.interp(v, ks, [sec.edges[i] for _, sec in knots])) for i in (0, 1))
    return (lo, hi) if d == direction else (-hi, -lo)

  def kerb_line(self, steps: list[tuple[int, bool]], nodes: list[int], side: int) -> np.ndarray:
    """A road's kerb (side 0 its left, 1 its right) along segments [(way, along it)] from node to node, as
    line_geometry draws each way's (through tapers and blends), stepping only where two ways' kerbs don't meet."""
    runs, pts, offs = [], [], []
    for k, (w, fwd) in enumerate(steps):
      d = FORWARD if fwd else BACKWARD
      refs, along, line = self.ways[w][1], self.way_along(w), self.way_points(w)
      a, b = nodes[k], nodes[k + 1]
      i = next(i for i in range(len(refs) - 1) if {refs[i], refs[i + 1]} == {a, b})
      sa, sb = (along[i], along[i + 1]) if refs[i] == a else (along[i + 1], along[i])
      found = self.taper(w) or self.blend(w)
      inner = [] if found is None else [v if found[0] == FORWARD else self.length(w) - v for v, _ in found[1]]
      inner = sorted((v for v in inner if min(sa, sb) + 0.05 < v < max(sa, sb) - 0.05), reverse=sb < sa)
      for n, s in enumerate([sa, *inner, sb]):
        off = self.edges_at(w, s, d)[side]
        if n == 0 and pts:
          if abs(off - offs[-1]) < 0.01:
            continue
          runs.append((pts, offs))
          pts, offs = [], []
        pts.append([np.interp(s, along, line[:, 0]), np.interp(s, along, line[:, 1])])
        offs.append(off)
    runs.append((pts, offs))
    return np.vstack([offset_line(np.array(p), np.array(o)) for p, o in runs])

  def way_points(self, way: int) -> np.ndarray:
    return self.xy[self.data.index(self.ways[way][1])]

  def nodes_at(self, p, tol: float = MATCH_TOL) -> list[int]:
    """The road nodes within tol of a point, nearest first: more than one where roads pass over each other."""
    cx, cy = int(p[0] // self.CELL), int(p[1] // self.CELL)
    found = []
    for dx in (-1, 0, 1):
      for dy in (-1, 0, 1):
        for k in self.cells.get((cx + dx, cy + dy), ()):
          d = float(np.hypot(*(self.xy[k] - p[:2])))
          if d < tol and int(self.ids[k]) in self.links:
            found.append((d, int(self.ids[k])))
    return [n for _, n in sorted(found)]


def runs(ways: list) -> list[tuple[int, bool, int, int]]:
  """Segment ways [(way, along it) or None per segment] as [(way, along it, first shape index, last)]."""
  out: list[tuple[int, bool, int, int]] = []
  for k, w in enumerate(ways):
    if w is None:
      continue
    if out and out[-1][:2] == w and out[-1][3] == k:
      out[-1] = (*w, out[-1][2], k + 1)
    else:
      out.append((*w, k, k + 1))
  return out


def route_nodes(points, osm: OsmLanes, tol: float = MATCH_TOL) -> list[int | None]:
  """The map node at each of a route's shape points (None where there is none): among nodes at the same place (over and
  under each other), those joined along the route."""
  found = [osm.nodes_at(p, tol) for p in np.asarray(points, float)[:, :2]]
  nodes: list[int | None] = []
  for k, cands in enumerate(found):
    prev = nodes[-1] if nodes else None
    nxt = found[k + 1] if k + 1 < len(found) else []
    joined = [c for c in cands if (prev is not None and (prev, c) in osm.pairs) or any((c, n) in osm.pairs for n in nxt)]
    nodes.append((joined or cands or [None])[0])
  return nodes


def junction_move(osm: OsmLanes, nodes: list[int | None], points, along, k: int) -> str | None:
  """'left', 'through' or 'right': the route's move through the junction at its shape point k, as the junction's
  turn:lanes call it (MOVE_THROUGH, MOVE_SKEW), rather than by how far the route turns; None where the route's way out
  isn't found among the junction's (nodes: route_nodes)."""
  j = nodes[k] if 0 < k < len(nodes) - 1 else None
  if j is None or nodes[k - 1] is None:
    return None
  pts = np.asarray(points, float)[:, :2]
  s = max(float(along[k]) - MOVE_HEADING, 0.0)
  back = np.array([np.interp(s, along, pts[:, 0]), np.interp(s, along, pts[:, 1])])
  h_in = heading_of(pts[k] - back)
  taken = {(a, b) for a, b in zip(nodes[k:], nodes[k + 1:], strict=False) if a is not None and b is not None}
  seen = {n for n in nodes[:k + 1] if n is not None}

  def drivable(a, b):
    w, fwd = osm.pairs[(a, b)]
    one = oneway_of(osm.ways[w][0])
    return one == 0 or (one == 1) == fwd

  exits = []  # (deg turned, left positive; whether it's the route's)
  stack = [(j, 0.0)]
  while stack:
    n, dist = stack.pop()
    for m in osm.links.get(n, ()):
      if m in seen or not drivable(n, m):
        continue
      seen.add(m)
      d = osm.node_xy(m) - osm.node_xy(n)
      turned = float(np.degrees((heading_of(d) - h_in + np.pi) % (2 * np.pi) - np.pi))
      length = float(np.hypot(*d))
      if abs(turned) <= MOVE_THROUGH and dist + length <= MOVE_SPAN and any(q != n and drivable(m, q) for q in osm.links.get(m, ())):
        stack.append((m, dist + length))
      elif abs(turned) <= MOVE_U_TURN:
        exits.append((turned, (n, m) in taken))
  ours = [e for e in exits if e[1]]
  if len(ours) != 1:
    return None
  turned = ours[0][0]
  nearest = min(abs(t) for t, _ in exits)
  if abs(turned) <= MOVE_THROUGH or (abs(turned) == nearest and nearest <= MOVE_SKEW):
    return 'through'
  return 'left' if turned > 0 else 'right'


MERGE_SPREAD = 40.0  # deg: a road joining the route's this near its heading on merges into it


def merge_side(osm: OsmLanes, nodes: list[int | None], k: int, split: bool = False) -> bool | None:
  """Whether another road joins the route's at its shape point k from behind (driven into the node, heading within
  MERGE_SPREAD of the route's way on) on the left of the route's road in (True) or the right (False); with `split`,
  leaves it ahead (driven away from the node, within MERGE_SPREAD of the route's way in) on the left of the route's
  road out. None where none does, or on both sides (nodes: route_nodes)."""
  if not 0 < k < len(nodes) - 1 or None in (nodes[k - 1], nodes[k], nodes[k + 1]):
    return None
  prev, j, nxt = nodes[k - 1], nodes[k], nodes[k + 1]
  here = osm.node_xy(j)
  ours = osm.node_xy(nxt if split else prev) - here  # the route's road on the side the other road is
  along = heading_of(here - osm.node_xy(prev)) if split else heading_of(osm.node_xy(nxt) - here)
  sides = set()
  for m in osm.links.get(j, ()):
    if m in (prev, nxt):
      continue
    w, fwd = osm.pairs[(j, m) if split else (m, j)]
    one = oneway_of(osm.ways[w][0])
    if one != 0 and (one == 1) != fwd:
      continue
    v = osm.node_xy(m) - here
    if abs(math.degrees((heading_of(v if split else -v) - along + math.pi) % (2 * math.pi) - math.pi)) > MERGE_SPREAD:
      continue
    cross = ours[0] * v[1] - ours[1] * v[0]
    sides.add(bool(cross > 0) if split else bool(cross < 0))
  return sides.pop() if len(sides) == 1 else None


def ways_from_nodes(points, osm: OsmLanes, tol: float = MATCH_TOL) -> list[tuple[int, bool, int, int]]:
  """The ways along a route whose shape points are the map's nodes, as on our GTA map, where every route point is one
  (but the ends, part way along a way): [(way id, along the way's direction, first shape index, last shape index)] for
  each run of segments on one way. Among nodes at the same place (over and under each other), those joined along the
  route."""
  pts = np.asarray(points, float)[:, :2]
  nodes = route_nodes(pts, osm, tol)
  ways: list[tuple[int, bool] | None] = [osm.pairs.get((a, b)) if a is not None and b is not None else None
                                         for a, b in zip(nodes, nodes[1:], strict=False)]

  def along(k, i, before):  # the way of end segment k, from node i back to (on from) the node it heads from (to)
    d = pts[k + 1] - pts[k]
    best, best_cos = None, np.cos(np.radians(10.0))
    for j in osm.links.get(i, ()):
      e = osm.node_xy(i) - osm.node_xy(j) if before else osm.node_xy(j) - osm.node_xy(i)
      c = float(d @ e) / max(float(np.hypot(*d) * np.hypot(*e)), 1e-9)
      if c > best_cos:
        best, best_cos = osm.pairs[(j, i) if before else (i, j)], c
    return best

  if len(ways) >= 2:
    if ways[0] is None and nodes[1] is not None:
      ways[0] = along(0, nodes[1], True)
    if ways[-1] is None and nodes[-2] is not None:
      ways[-1] = along(len(ways) - 1, nodes[-2], False)
  return runs(ways)


def ways_from_trace(edges: list[dict], points, osm: OsmLanes) -> list[tuple[int, bool, int, int]]:
  """The ways along a route from Valhalla's trace_attributes of its shape (shape_match edge_walk, edge.way_id,
  edge.begin_shape_index, edge.end_shape_index), as for a real map: each edge's direction along its way from where its
  first and last shape points fall on the way."""
  pts = np.asarray(points, float)[:, :2]
  ways: list[tuple[int, bool] | None] = [None] * max(len(pts) - 1, 0)
  for e in edges:
    w, i0, i1 = e.get('way_id'), e.get('begin_shape_index', 0), e.get('end_shape_index', 0)
    if w not in osm.ways or i1 <= i0:
      continue
    line = osm.way_points(w)
    t0, t1 = (_param(line, pts[i]) for i in (i0, i1))
    for k in range(i0, min(i1, len(ways))):
      ways[k] = (w, t1 >= t0)
  return runs(ways)


def _param(line: np.ndarray, p: np.ndarray) -> float:
  """How far along a polyline (m) its nearest point to p is."""
  a, ab = line[:-1], np.diff(line, axis=0)
  ab2 = np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-9)
  t = np.clip(np.einsum('ij,ij->i', p - a, ab) / ab2, 0.0, 1.0)
  i = int(np.argmin(np.hypot(*(a + ab * t[:, None] - p).T)))
  return float(np.concatenate(([0.0], np.cumsum(np.sqrt(ab2))))[i] + t[i] * np.sqrt(ab2[i]))


def fillet(p_in: np.ndarray, u_in: np.ndarray, p_out: np.ndarray, u_out: np.ndarray, radius_left: float,
           radius_right: float, step: float = 0.5) -> np.ndarray | None:
  """A path from p_in heading u_in to p_out heading u_out (unit vectors): straight on, the widest circular arc that fits
  (up to radius_left / radius_right for a left / right turn), straight on. None where the lines in and out don't meet
  ahead of p_in and behind p_out."""
  cross = u_in[0] * u_out[1] - u_in[1] * u_out[0]
  theta = float(np.arctan2(cross, u_in @ u_out))  # left positive
  if abs(cross) < 1e-3:
    return None
  a, b = np.linalg.solve(np.array([u_in, u_out]).T, p_out - p_in)  # p_in + a u_in = corner = p_out - b u_out
  if a <= 0 or b <= 0:
    return None
  half = np.tan(abs(theta) / 2)
  tangent = min(a, b, (radius_left if theta > 0 else radius_right) * half)
  radius = tangent / half
  corner = p_in + a * u_in
  start, end = corner - tangent * u_in, corner + tangent * u_out
  centre = start + radius * np.array([-u_in[1], u_in[0]]) * np.sign(theta)
  sweep = np.linspace(0, abs(theta), max(int(radius * abs(theta) / step), 2))
  ang = np.arctan2(start[1] - centre[1], start[0] - centre[0]) + np.sign(theta) * sweep
  arc = centre + radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)
  lead = p_in + np.outer(np.arange(0, a - tangent, step), u_in)
  tail = end + np.outer(np.arange(step, b - tangent, step), u_out)
  return np.concatenate((lead, arc, tail, p_out[None]))


def heading_of(d) -> np.ndarray:
  """Radians counterclockwise from east."""
  d = np.asarray(d, float)
  return np.arctan2(d[..., 1], d[..., 0])


def corners(points: np.ndarray, along: np.ndarray, turn: float = CORNER_TURN, chord: float = CORNER_CHORD,
            step: float = 1.0) -> list[tuple[float, float]]:
  """Where a route turns at least `turn` deg (and less than a U-turn) between its chords either side: [(m along,
  deg turned, left positive)]. A sideways jog that heads back the way it was going isn't a corner."""
  if len(points) < 2 or along[-1] < 2 * chord + step:
    return []
  s = np.arange(chord, along[-1] - chord, step)

  def at(v):
    return np.stack([np.interp(v, along, points[:, 0]), np.interp(v, along, points[:, 1])], axis=-1)
  here = at(s)
  bend = np.degrees((heading_of(at(s + chord) - here) - heading_of(here - at(s - chord)) + np.pi) % (2 * np.pi) - np.pi)
  out = []
  for a, b in _true_runs(np.abs(bend) >= turn / 2):
    sa, sb = s[a], s[b - 1]
    net = float(np.degrees((heading_of(at(sb + chord) - at(sb)) - heading_of(at(sa) - at(sa - chord)) + np.pi) % (2 * np.pi) - np.pi))
    if turn <= abs(net) <= 170.0:
      out.append((float(s[a + int(np.argmax(np.abs(bend[a:b])))]), net))
  return out


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
  edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
  return list(zip(edges[::2], edges[1::2], strict=True))


class RouteLanes:
  """A route's lanes (points [N, 2] in m): each segment's cross-section (Section, None where unknown), the turn arrows
  on the way into each junction, and the line through the lanes a plan takes, with fillets through its corners."""
  def __init__(self, points, sections: list[Section | None], arrows: list | None = None, tapers: dict | None = None,
               junctions=None, explicit: dict | None = None, moves: list | None = None, blended: dict | None = None,
               merges: dict | None = None, splits: dict | None = None):
    self.points = np.asarray(points, float)[:, :2]
    self.along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(self.points, axis=0).T))))
    self.sections = sections
    self.arrows: list[tuple[float, list[frozenset[str]]]] = arrows or []  # (m along where they end, each lane's)
    # the route's move through each of those junctions as the arrows call it (junction_move), None where unknown
    self.moves: list[str | None] = moves if moves is not None else [None] * len(self.arrows)
    self.tapers: dict[int, tuple[Section, Section]] = tapers or {}  # first segment of a way: (lanes before, after)
    # segments of a way whose lanes widen or narrow along it (width:lanes...:start / :end): its cross-sections where the
    # route enters and leaves it, and m along there
    self.explicit: dict[int, tuple[Section, Section, float, float]] = explicit or {}
    # segments of a way whose lanes move across to the next way's (OsmLanes.blend): its knots, whether the route runs
    # along it, and m along the way at the segment's ends
    self.blended: dict[int, tuple[list[tuple[float, Section]], bool, float, float]] = blended or {}
    # shape points where another road joins the route's from behind, as onto a freeway from its on-ramp: whether it
    # joins on the left of ours (merge_side)
    self.merges: dict[int, bool] = merges or {}
    # and where another road leaves it ahead, as an exit from a freeway: whether it leaves on the left of ours
    self.splits: dict[int, bool] = splits or {}
    # m along to the nodes where roads meet: a corner near one is a turn through a junction, the rest are bends
    self.junctions = np.sort(np.asarray(junctions if junctions is not None else [], float))
    self._corners: list[tuple[float, float]] | None = None
    self._drops: list[tuple[float, tuple[int, int, int]]] | None = None
    self._openings: list[tuple[float, int]] | None = None
    self._maps: list[tuple[float, tuple[int | None, ...], int]] | None = None
    self._runs: dict[int, tuple[int, int]] = {}

  @classmethod
  def from_osm(cls, points, ways_along: list[tuple[int, bool, int, int]], osm: OsmLanes) -> 'RouteLanes':
    """From the ways along it (ways_from_nodes, ways_from_trace)."""
    pts = np.asarray(points, float)[:, :2]
    ways: list[tuple[int, int] | None] = [None] * max(len(pts) - 1, 0)
    for w, fwd, i0, i1 in ways_along:
      for k in range(max(i0, 0), min(i1, len(ways))):
        ways[k] = (w, FORWARD if fwd else BACKWARD)
    cache: dict[tuple[int, int], Section] = {}
    sections: list[Section | None] = []
    for wd in ways:
      if wd is not None and wd not in cache:
        cache[wd] = Section.of(osm.lanes(wd[0]), wd[1])
      sections.append(cache[wd] if wd is not None else None)
    along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))))
    arrows: list[tuple[float, list[frozenset[str]]]] = []
    ends: list[int] = []  # the shape point each arrows' junction is at
    tapers: dict[int, tuple[Section, Section]] = {}
    degree = [osm.degree.get(n[0], 0) if (n := osm.nodes_at(p)) else 0 for p in pts]
    explicit: dict[int, tuple[Section, Section, float, float]] = {}
    for w, fwd, i0, i1 in ways_along:
      if osm.has_ends(w):
        d = FORWARD if fwd else BACKWARD
        first, last = (Section.of(osm.lanes_at(w, e), d) for e in (('start', 'end') if fwd else ('end', 'start')))
        k0, k1 = max(i0, 0), min(i1, len(ways))
        for k in range(k0, k1):
          explicit[k] = (first, last, float(along[k0]), float(along[k1]))
    blended = cls._blended(pts, ways_along, osm, len(ways))
    for k, sec in enumerate(sections):
      nxt = sections[k + 1] if k + 1 < len(sections) else None
      if sec is not None and any(sec.turns) and not (nxt is not None and any(nxt.turns) and nxt.lanes == sec.lanes):
        arrows.append((float(along[k + 1]), sec.turns))  # the end of a run of ways with arrows: the junction
        ends.append(k + 1)
      prev = sections[k - 1] if k else None
      if prev is not None and sec is not None and ways[k - 1] != ways[k] and degree[k] == 2 and _taperable(prev, sec) and \
         k not in explicit and k - 1 not in explicit:
        tapers[k] = (prev, sec)
    junctions = [float(along[k]) for k in range(1, len(pts) - 1) if degree[k] >= 3]
    nodes = route_nodes(pts, osm) if ends or any(d >= 3 for d in degree) else []
    moves = [junction_move(osm, nodes, pts, along, k) for k in ends]
    merges = {k: side for k in range(1, len(pts) - 1) if degree[k] >= 3 and (side := merge_side(osm, nodes, k)) is not None}
    splits = {k: side for k in range(1, len(pts) - 1) if degree[k] >= 3 and (side := merge_side(osm, nodes, k, True)) is not None}
    return cls(pts, sections, arrows, tapers, junctions, explicit, moves, blended, merges, splits)

  @staticmethod
  def _blended(pts: np.ndarray, ways_along: list[tuple[int, bool, int, int]], osm: OsmLanes, n: int) -> dict:
    """RouteLanes.blended: each segment on a way whose lanes move across to the next way's (OsmLanes.blend)."""
    out = {}
    for w, fwd, i0, i1 in ways_along:
      if osm.has_ends(w) or (found := osm.blend(w)) is None:
        continue
      line = osm.way_points(w)
      for k in range(max(i0, 0), min(i1, n)):
        out[k] = (found[1], fwd, _param(line, pts[k]), _param(line, pts[k + 1]))
    return out

  @property
  def corners(self) -> list[tuple[float, float]]:
    """The route's turns through junctions: its corners (corners()) with a junction within FILLET_REACH that turn
    mostly within CORNER_SHARP m, as at a junction's node, not along a road bending through it."""
    if self._corners is None:
      near = self.junctions
      self._corners = []
      for sc, turned in corners(self.points, self.along):
        if not len(near) or np.abs(near - sc).min() > FILLET_REACH:
          continue
        p = [np.array([np.interp(v, self.along, self.points[:, 0]), np.interp(v, self.along, self.points[:, 1])])
             for v in (sc - CORNER_SHARP, sc - CORNER_SHARP / 2, sc + CORNER_SHARP / 2, sc + CORNER_SHARP)]
        inner = np.degrees((heading_of(p[3] - p[2]) - heading_of(p[1] - p[0]) + np.pi) % (2 * np.pi) - np.pi)
        if abs(inner) >= SHARP_SHARE * abs(turned):
          self._corners.append((sc, turned))
    return self._corners

  @property
  def drops(self) -> list[tuple[float, tuple[int, int, int]]]:
    """Where the route goes straight on through a junction onto fewer lanes its way, all the lanes into it going
    straight on (by their arrows, or none): [(m along to the junction's first node, (the first and last of our lanes
    into it that carry on through it (continuing), from the left, of how many))]."""
    if self._drops is None:
      self._drops = []
      groups: list[list[float]] = []
      for s in self.junctions:
        if groups and s - groups[-1][-1] < DROP_SPAN:
          groups[-1].append(float(s))
        else:
          groups.append([float(s)])
      for g in groups:
        s0, s1 = g[0], g[-1]
        if s0 - DROP_REACH < 0 or s1 + DROP_REACH > self.along[-1]:
          continue
        into, out = self.sections[self.segment(s0 - 1e-3)], self.sections[self.segment(s1 + 1e-3)]
        if into is None or out is None or not 0 < out.lanes < into.lanes or not all(not t or 'through' in t for t in into.turns):
          continue
        p = [np.array([np.interp(v, self.along, self.points[:, 0]), np.interp(v, self.along, self.points[:, 1])])
             for v in (s0 - DROP_REACH, s0, s1, s1 + DROP_REACH)]
        turned = np.degrees((heading_of(p[3] - p[2]) - heading_of(p[1] - p[0]) + np.pi) % (2 * np.pi) - np.pi)
        on = continuing(into, out)
        if abs(turned) <= DROP_TURN and on and len(on) < into.lanes:
          self._drops.append((s0, (min(on), max(on), into.lanes)))
    return self._drops

  @property
  def openings(self) -> list[tuple[float, int]]:
    """Where lanes begin on the left of ours as the route's road widens, as a turn bay opens (opening): [(m along to
    where its lane count rises, how many)]."""
    if self._openings is None:
      self._openings = []
      for k in range(1, len(self.sections)):
        few, many = self.sections[k - 1], self.sections[k]
        if few is not None and many is not None and 0 < few.lanes < many.lanes and (o := self.opening(k)) is not None and o[2]:
          self._openings.append((float(self.along[k]), many.lanes - few.lanes))
    return self._openings

  @property
  def lane_maps(self) -> list[tuple[float, tuple[int | None, ...], int]]:
    """Where our lanes change along the route (in number, or where they are), the lane each one before carries on as
    (lanes numbered from the left, as everywhere): [(m along, (for each lane before, the lane after, None where it
    ends), how many after)]. Matched by where the lanes are (match_lanes) either side of each change: at a node as
    the cross-sections there have them (a taper's, or a widening way's, lane opening from nothing), across a junction
    or a stretch without lanes by where the lanes in run on to across the road out. Through a run of stretches
    shorter than BLIP_M (a junction's own links, a lane count the map has wrong for a way) a lane comes out as the road
    before goes on directly as the road after, so it doesn't move across for them. None through a junction the route
    turns at, where nav picks the lane out."""
    if self._maps is None:
      self._maps = self._lane_maps()
    return self._maps

  def _lane_maps(self) -> list[tuple[float, tuple[int | None, ...], int]]:
    secs = self.sections
    runs: list[list[int]] = []  # [first segment, last] of each run of segments with one layout of our lanes
    for k, sec in enumerate(secs):
      if sec is None or not sec.lanes:
        continue
      if runs and runs[-1][1] == k - 1 and _same_ours(secs[k - 1], sec, JOG_MIN):
        runs[-1][1] = k
      else:
        runs.append([k, k])
    turns = [sc for sc, turned in self.corners if abs(turned) >= MAP_TURN]
    anchors = [i for i, (k0, k1) in enumerate(runs)
               if i in (0, len(runs) - 1) or self.along[k1 + 1] - self.along[k0] >= BLIP_M]
    out = []
    for a, b in zip(anchors, anchors[1:], strict=False):
      chain = runs[a:b + 1]
      s0, s1 = float(self.along[chain[0][1] + 1]), float(self.along[chain[-1][0]])
      if any(s0 - FILLET_REACH <= sc <= s1 + FILLET_REACH for sc in turns):
        continue
      steps = [[float(self.along[q[0]]), self._match(r, q), secs[q[0]].lanes] for r, q in zip(chain, chain[1:], strict=False)]
      # out of short stretches between two of the same lanes as the road before goes on as the road after
      if len(chain) > 2 and s1 - s0 <= 2 * BLIP_M and secs[chain[0][1]].lanes == secs[chain[-1][0]].lanes:
        direct = self._match(chain[0], chain[-1])
        through = list(range(secs[chain[0][1]].lanes))
        for _, m, _ in steps[:-1]:
          through = [None if i is None else m[i] for i in through]
        last = list(steps[-1][1])
        for j in range(len(last)):
          came = [i for i, t in enumerate(through) if t == j]
          if came:
            last[j] = direct[came[0]]
        kept = [j for j in last if j is not None]
        if all(a < b for a, b in zip(kept, kept[1:], strict=False)):  # still one lane each, in order
          steps[-1][1] = last
      out += [(s, tuple(m), n) for s, m, n in steps if len(m) != n or any(j != i for i, j in enumerate(m))]
    return out

  def _match(self, r: list[int], q: list[int]) -> list[int | None]:
    """match_lanes from the end of run r of segments to the start of run q, where they meet or across what's between
    (the lanes before carried on straight across the road's line after)."""
    sa, sb = float(self.along[r[1] + 1]), float(self.along[q[0]])
    a, b = self.section_at(sa, r[1]), self.section_at(sb, q[0])
    ours_a = [(s.left, s.right) for s in a.ours]
    if sb - sa > EPS:
      pa, pb = self.points[r[1] + 1], self.points[q[0]]
      na, nb = (_right_normal(self.points[k + 1] - self.points[k]) for k in (r[1], q[0]))
      ours_a = [(float((pa + na * x - pb) @ nb), float((pa + na * y - pb) @ nb)) for x, y in ours_a]
    side = _turn_side(self._arrowed(r, sa, -1) or a, self._arrowed(q, sb, 1) or b)
    if side is None:
      side = _median_side(a, b)
    if side is None and sb - sa <= EPS and b.lanes > a.lanes and q[0] in self.merges:
      side = self.merges[q[0]]  # another road's lanes join ours on its side: two ways' lines meet at the node anyhow
    if side is None and sb - sa <= EPS and b.lanes < a.lanes and q[0] in self.splits:
      side = self.splits[q[0]]  # and leave on its side
    if side is not None:  # the turn lane's own side, wherever the lanes lie
      shift = (b.lanes - a.lanes) if side else 0
      return [i + shift if 0 <= i + shift < b.lanes else None for i in range(a.lanes)]
    return match_lanes(ours_a, [(s.left, s.right) for s in b.ours])

  def turns_at(self, s: float, opened: bool = False) -> list[frozenset[str]]:
    """Our lanes' turn arrows s m along: their own, else those of the lanes they carry on as (lane_maps) into the next
    junction with arrows within ARROWS_BACK m, the route not turning on the way (mappers, and our map, tag them on the
    way into the junction alone). With `opened`, of the lanes there as opened_at has them."""
    k = self.segment(s)
    sec = self.section_at(s, k) if 0 <= k < len(self.sections) else None
    if sec is None or not sec.lanes:
      return []
    turns = sec.turns
    if not any(turns):
      ahead = next((j for j in range(k + 1, len(self.sections)) if self.sections[j] is not None and any(self.sections[j].turns)
                    and self.along[j] - s <= ARROWS_BACK), None)
      if ahead is not None and not any(s < sc < self.along[ahead] + FILLET_REACH for sc, _ in self.corners):
        maps = [mp for d, mp, _ in self.lane_maps if s < d <= self.along[ahead] + EPS]
        if (len(maps[0]) if maps else self.sections[ahead].lanes) == sec.lanes:
          far = self.sections[ahead].turns
          turns = []
          for i in range(sec.lanes):
            j = i
            for mp in maps:
              j = mp[j] if j is not None else None
            turns.append(far[j] if j is not None and j < len(far) else frozenset())
    opening = self.opening(k) if opened else None
    if opening is not None and s < opening[0]:
      _, extra, left = opening
      turns = list(turns[extra:]) if left else list(turns[:len(turns) - extra])
    return list(turns)

  def _arrowed(self, run: list[int], s: float, step: int) -> Section | None:
    """The first cross-section with turn arrows in a run of segments of one layout, from its end at s (step -1: back
    from its last segment; 1: on from its first) within ARROWS_REACH m; None for none."""
    ks = range(run[1], run[0] - 1, -1) if step < 0 else range(run[0], run[1] + 1)
    for k in ks:
      if (s - self.along[k + 1] if step < 0 else self.along[k] - s) > ARROWS_REACH:
        break
      sec = self.sections[k]
      if sec is not None and any(sec.turns):
        return sec
    return None

  def segment(self, s: float) -> int:
    return int(min(max(np.searchsorted(self.along, s, side='right') - 1, 0), max(len(self.sections) - 1, 0)))

  def _run(self, k: int) -> tuple[int, int]:
    """The segments either side of k on the same road layout (Section.key), the road carrying on across ways."""
    if k in self._runs:
      return self._runs[k]
    key = self.sections[k].key
    k0, k1 = k, k
    while k0 > 0 and (p := self.sections[k0 - 1]) is not None and k0 - 1 not in self.explicit and p.key == key:
      k0 -= 1
    while k1 + 1 < len(self.sections) and (n := self.sections[k1 + 1]) is not None and k1 + 1 not in self.explicit and n.key == key:
      k1 += 1
    self._runs[k] = k0, k1
    return k0, k1

  def section_at(self, s: float, k: int | None = None) -> Section | None:
    """The cross-section s m along (on segment k, where s is at a point segments share): along a way whose lanes widen
    or narrow (width:lanes...:start / :end) as they do along it; near a node where the road carries on onto a way with
    the same lanes elsewhere, moving across to them as the lines do (OsmLanes.blend); else a lane widening from nothing
    over TAPER_M where the road's lane count has risen, or narrowing to nothing before it falls."""
    k = self.segment(s) if k is None else k
    sec = self.sections[k] if 0 <= k < len(self.sections) else None
    if sec is None:
      return sec
    if k in self.explicit:
      first, last, s0, s1 = self.explicit[k]
      return _lerp(first, last, (s - s0) / max(s1 - s0, EPS))
    if k in self.blended:
      knots, fwd, v0, v1 = self.blended[k]
      t = (s - self.along[k]) / max(self.along[k + 1] - self.along[k], EPS)
      here = at_knots(knots, v0 + (v1 - v0) * min(max(t, 0.0), 1.0))
      return here if fwd else mirrored(here)
    if not self.tapers:
      return sec
    k0, k1 = self._run(k)
    if k0 in self.tapers and s - self.along[k0] < TAPER_M and self.tapers[k0][0].lanes < sec.lanes:
      return _blend(self.tapers[k0][0], sec, (s - self.along[k0]) / TAPER_M)
    if k1 + 1 in self.tapers and self.along[k1 + 1] - s < TAPER_M and self.tapers[k1 + 1][1].lanes < sec.lanes:
      return _blend(self.tapers[k1 + 1][1], sec, (self.along[k1 + 1] - s) / TAPER_M)
    return sec

  def opening(self, k: int) -> tuple[float, int, bool] | None:
    """Where lanes begin on the way segment k is on, as its lane count rises: (m along where they are fully there, how
    many, whether on the left of ours); None where none begin. Fully there at the end of a way they widen along
    (width:lanes...:start / :end), else of their taper, or half way along a road shorter than one."""
    sec = self.sections[k] if 0 <= k < len(self.sections) else None
    if sec is None:
      return None
    if k in self.explicit:
      first, _, _, s1 = self.explicit[k]
      ours = first.ours
      shut = [i for i, sp in enumerate(ours) if sp.right - sp.left < OPENED]
      if not shut or len(shut) == len(ours):
        return None
      return s1, len(shut), shut[0] == 0
    if not self.tapers:
      return None
    k0, k1 = self._run(k)
    if k0 not in self.tapers or self.tapers[k0][0].lanes >= sec.lanes:
      return None
    run = self.along[k1 + 1] - self.along[k0]
    return float(self.along[k0] + (TAPER_M if run > TAPER_M else run / 2)), sec.lanes - self.tapers[k0][0].lanes, _opens_left(self.tapers[k0][0], sec)

  def opened_at(self, s: float, k: int | None = None) -> Section | None:
    """section_at's cross-section with only the lanes fully there: those still widening from nothing are left out, so a
    turn bay isn't a lane until it has opened."""
    k = self.segment(s) if k is None else k
    sec = self.section_at(s, k)
    opening = self.opening(k)
    if sec is None or opening is None or s >= opening[0]:
      return sec
    _, extra, left = opening
    lo = sec.first if left else sec.first + sec.lanes - extra
    return Section(sec.spans[:lo] + sec.spans[lo + extra:], sec.edges)

  @property
  def arrows_moves(self) -> list[tuple]:
    """arrows, each with the route's move through its junction where known: (m along, each lane's[, move])."""
    return [(e, turns) if move is None else (e, turns, move) for (e, turns), move in zip(self.arrows, self.moves, strict=True)]

  def arrows_near(self, s: float, before: float = 30.0, after: float = 5.0) -> list[frozenset[str]] | None:
    """The turn arrows of the lanes into the junction at a maneuver s m along: those ending nearest it, from `before`
    m before it to `after` past; None for none."""
    near = [(abs(e - s), turns) for e, turns in self.arrows if s - before <= e <= s + after]
    return min(near, key=lambda n: n[0])[1] if near else None

  def lane_line(self, at: float, keys: list[tuple[float, float]], step: float = 2.0) -> np.ndarray | None:
    """The route on from `at` m along, moved into the lanes keys gives ([(m on from at, lane)], ramping between each
    two), each segment by its own cross-section; across segments without one the offset runs evenly between the known
    ones. Through each corner of the route, a fillet from the lane in to the lane out replaces it (FILLET_REACH either
    side), as a car drives through a junction rather than along its ways to the node in the middle."""
    if not keys or at >= self.along[-1] - 1e-6:
      return None
    back = max(at - 2 * FILLET_REACH - CORNER_CHORD, 0.0)  # from behind, so a corner the car is in keeps its fillet
    kx = np.array([at + d for d, _ in keys], dtype=float)
    ky = np.array([lane for _, lane in keys], dtype=float)
    near = np.clip(np.searchsorted(self.along, kx), 1, len(self.along) - 1)
    near = np.where(np.abs(self.along[near - 1] - kx) < np.abs(self.along[near] - kx), near - 1, near)
    kx = np.where(np.abs(self.along[near] - kx) < 1e-3, self.along[near], kx)  # a step at a node falls between its segments
    inside = self.along[(self.along > back) & (self.along < self.along[-1])]
    grid = np.arange(back, self.along[-1], step)
    s2 = np.unique(np.concatenate(([back], inside, grid, kx[(kx > back) & (kx < self.along[-1])], [self.along[-1]])))
    s2 = s2[np.concatenate(([True], np.diff(s2) > 1e-3))]
    xy = np.stack([np.interp(s2, self.along, self.points[:, 0]), np.interp(s2, self.along, self.points[:, 1])], axis=1)
    lane_in, lane_out = _keyed(s2, kx, ky)
    seg = np.clip(np.searchsorted(self.along, (s2[:-1] + s2[1:]) / 2, side='right') - 1, 0, len(self.sections) - 1)
    # in Python floats: a few thousand points every half second on the bridge's main thread
    sv, lin, lout = s2.tolist(), lane_in.tolist(), lane_out.tolist()
    offl = [math.nan] * len(sv)
    jogs = []  # shared points where the lane in and the lane out don't line up
    last: tuple = (-1, -1, None)  # the cross-section at a point inside a segment, for the next piece's start
    for i, k in enumerate(seg.tolist()):
      for v, lane in ((i, lout), (i + 1, lin)):  # the piece's ends, averaged with the next's at a shared point
        if last[0] == v and last[1] == k:
          sec = last[2]
        else:
          sec = self.section_at(sv[v], k)
          last = (v, k, sec)
        if sec is None or not sec.lanes:
          continue
        off = sec.offset(lane[v])
        if not math.isnan(offl[v]) and abs(off - offl[v]) > JOG_MIN:
          jogs.append(v)
        offl[v] = off if math.isnan(offl[v]) else (offl[v] + off) / 2
    offs = np.array(offl)
    known = ~np.isnan(offs)
    if not known.any():
      return None
    offs = np.interp(s2, s2[known], offs[known])
    offs = _ease_jogs(s2, offs, [float(s2[v]) for v in jogs], self.points, self.along)
    line = offset_line(xy, offs)
    if len(line) != len(s2):
      return None
    line = _smooth_line(_unfold(line, xy), s2)
    line, s2 = self._fillets(line, s2)
    k = int(np.searchsorted(s2, at, side='right'))
    if k >= len(s2):
      return None
    start = np.array([np.interp(at, s2, line[:, 0]), np.interp(at, s2, line[:, 1])])
    return np.vstack([start, line[k:]])

  def _fillets(self, line: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cs = [c for c in self.corners if s[0] < c[0] < s[-1]]
    out_xy, out_s, last = [], [], 0
    for n, (sc, _) in enumerate(cs):
      lo = cs[n - 1][0] if n else -np.inf
      hi = cs[n + 1][0] if n + 1 < len(cs) else np.inf
      # a corner near the route's start (where it was routed from the car) or its end has less room either side
      start = s[0] <= self.along[0] + EPS
      a = sc - min(FILLET_REACH, (sc - lo) / 2, sc - s[0] - FILLET_MIN if start else np.inf)
      b = sc + min(FILLET_REACH, (hi - sc) / 2, s[-1] - sc - FILLET_MIN)
      if min(sc - a, b - sc) < FILLET_MIN or (not start and a - CORNER_CHORD < s[0]):
        continue

      def at(v, pts=line):
        return np.array([np.interp(v, s, pts[:, 0]), np.interp(v, s, pts[:, 1])])

      def route(v):
        return np.array([np.interp(v, self.along, self.points[:, 0]), np.interp(v, self.along, self.points[:, 1])])
      u_in = route(a) - route(max(a - CORNER_CHORD, self.along[0]))
      u_out = route(min(b + CORNER_CHORD, self.along[-1])) - route(b)
      u_in, u_out = u_in / max(np.hypot(*u_in), 1e-6), u_out / max(np.hypot(*u_out), 1e-6)
      path = fillet(at(a), u_in, at(b), u_out, FILLET_RADIUS['left'], FILLET_RADIUS['right'])
      i0, i1 = int(np.searchsorted(s, a)), int(np.searchsorted(s, b, side='right'))
      if path is None or i0 < last:
        continue
      out_xy += [line[last:i0], path]
      out_s += [s[last:i0], np.linspace(a, b, len(path))]
      last = i1
    if not out_xy:
      return line, s
    out_xy.append(line[last:])
    out_s.append(s[last:])
    xy, sv = np.concatenate(out_xy), np.concatenate(out_s)
    keep = np.concatenate(([True], (np.diff(sv) > 1e-6) & (np.hypot(*np.diff(xy, axis=0).T) > 1e-3)))
    return xy[keep], sv[keep]


MAP_TURN = 45.0  # deg: lanes aren't carried across a junction the route turns this much at (RouteLanes.lane_maps)
BLIP_M = 20.0  # m: a stretch of one layout of lanes shorter than this is a junction's link or a count the map has wrong
SAME_PLACE = 0.05  # m


def _same_ours(a: Section, b: Section, tol: float = SAME_PLACE) -> bool:
  return a.lanes == b.lanes and all(abs(x.left - y.left) < tol and abs(x.right - y.right) < tol
                                    for x, y in zip(a.ours, b.ours, strict=True))


def _right_normal(d) -> np.ndarray:
  n = np.array([d[1], -d[0]], float)
  return n / max(float(np.hypot(*n)), 1e-9)


JOG_MIN = 0.5  # m between the lane in and the lane out at a node that is a jog sideways, not a road's lanes carrying on
JOG_REACH = 10.0  # m either side of such a node the lane line moves across over
JOG_SPAN = 30.0  # m either side of it to the route's points the move across may start or end at instead
JOG_BETTER = 0.5  # m the lane line must come nearer straight for that
JOG_STRAIGHT = 20.0  # deg at most the route turns from before such a stretch to after it


def _unfold(line: np.ndarray, xy: np.ndarray) -> np.ndarray:
  """An offset line (one point per route point xy) without the loops it makes inside a corner of the route sharper
  than its offset rounds (each point no further back along the route than the one before): held where it would go
  back."""
  t = np.gradient(xy, axis=0)
  back = np.einsum('ij,ij->i', np.diff(line, axis=0), t[1:]) < 0
  if not back.any():
    return line
  out = line.copy()
  for i in range(1, len(out)):
    if float((out[i] - out[i - 1]) @ t[i]) < 0:
      out[i] = out[i - 1]
  return out


SMOOTH_M = 5.0  # m either side over which the lane line's points are averaged: the route's kinks at its nodes rounded


def _smooth_line(line: np.ndarray, s: np.ndarray) -> np.ndarray:
  """The lane line (points at s m along) each point the mean of those within SMOOTH_M along, its ends kept: GTA's
  links kink by 5-20 deg at nodes (a ramp joining a freeway), which a car's path rounds."""
  if len(line) < 3:
    return line
  c = np.concatenate(([[0.0, 0.0]], np.cumsum(line, axis=0)))
  lo = np.searchsorted(s, s - SMOOTH_M, side='left')
  hi = np.searchsorted(s, s + SMOOTH_M, side='right')
  out = (c[hi] - c[lo]) / (hi - lo)[:, None]
  out[0], out[-1] = line[0], line[-1]
  return out


def _keyed(s: np.ndarray, kx: np.ndarray, ky: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """The lane keys [(kx m along, ky lane)] give at each s, ramping between two keys, and stepping where keys share a
  place: (the lane arriving at s, the lane leaving it), which differ only at a step."""
  ux, first = np.unique(kx, return_index=True)
  last = len(kx) - 1 - np.unique(kx[::-1], return_index=True)[1]
  lo, hi = ky[first], ky[last]  # each place's first and last key
  j = np.clip(np.searchsorted(ux, s, side='right') - 1, 0, len(ux) - 1)
  nxt = np.minimum(j + 1, len(ux) - 1)
  span = ux[nxt] - ux[j]
  t = np.clip(np.divide(s - ux[j], span, out=np.zeros_like(s), where=span > 0), 0.0, 1.0)
  leaving = np.where(s < ux[0], lo[0], hi[j] + (lo[nxt] - hi[j]) * t)
  arriving = np.where(s == ux[j], lo[j], leaving)
  return arriving, leaving


def _ease_jogs(s: np.ndarray, offs: np.ndarray, jogs: list[float], points: np.ndarray | None = None,
               along: np.ndarray | None = None) -> np.ndarray:
  """The lane line's offsets (m right of the route at s m along) moving across evenly through each jog, as where one
  carriageway of a divided road joins the middle of the road it becomes: a car keeps to its lane across it rather than
  following the ways' sideways step. Over JOG_REACH either side of it; with the route's points (and m along to them),
  instead between whichever two of them within JOG_SPAN leave the lane line straightest (by JOG_BETTER m), on a road
  heading on within JOG_STRAIGHT: as where a carriageway's last link angles across to the line of the road it
  becomes, the lane line then moves across along that link and runs on straight."""
  out = offs.copy()
  xy = normals = None
  for n, sj in enumerate(jogs):
    lo_lim = max(sj - ((sj - jogs[n - 1]) / 2 if n else math.inf), float(s[0]))
    hi_lim = min(sj + ((jogs[n + 1] - sj) / 2 if n + 1 < len(jogs) else math.inf), float(s[-1]))
    before, after = s < sj - 1e-6, s > sj + 1e-6
    if not before.any() or not after.any():
      continue
    o_in, o_out = float(offs[before][-1]), float(offs[after][0])  # the lane's offsets either side of the step

    def ends(a, b):  # the offsets at a and b, either side of the jog (or at it)
      return (o_in if a >= sj - 1e-6 else float(np.interp(a, s, offs)), o_out if b <= sj + 1e-6 else float(np.interp(b, s, offs)))
    lo, hi = max(sj - JOG_REACH, lo_lim), min(sj + JOG_REACH, hi_lim)
    if points is not None and along is not None:
      def at(v):
        return np.array([np.interp(v, along, points[:, 0]), np.interp(v, along, points[:, 1])])
      if normals is None:
        xy = np.stack([np.interp(s, along, points[:, 0]), np.interp(s, along, points[:, 1])], axis=1)
        d = np.gradient(xy, axis=0)
        normals = np.stack([d[:, 1], -d[:, 0]], axis=1) / np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-9)[:, None]

      f0, f1 = max(sj - JOG_SPAN, lo_lim), min(sj + JOG_SPAN, hi_lim)
      u_in, u_out = at(f0 + JOG_REACH) - at(f0), at(f1) - at(f1 - JOG_REACH)
      span = (s >= f0) & (s <= f1)
      straight = min(np.hypot(*u_in), np.hypot(*u_out)) > 1e-6 and span.sum() >= 3 and abs(math.degrees(
        (heading_of(u_out) - heading_of(u_in) + math.pi) % (2 * math.pi) - math.pi)) <= JOG_STRAIGHT

      def bend(a, b):  # how far the lane line strays from straight over the stretch, moving across from a to b
        o = offs[span].copy()
        inside = (s[span] >= a) & (s[span] <= b)
        o[inside] = np.interp(s[span][inside], [a, b], ends(a, b))
        if b <= sj + 1e-6:
          o[s[span] > sj - 1e-6] = np.where(s[span][s[span] > sj - 1e-6] > b, o[s[span] > sj - 1e-6], o_out)
        p = xy[span] + normals[span] * o[:, None]
        chord = p[-1] - p[0]
        length = float(np.hypot(*chord))
        return float(np.abs((p - p[0]) @ np.array([chord[1], -chord[0]]) / length).max()) if length > 1e-6 else math.inf
      if straight and (best := bend(lo, hi)) > JOG_BETTER:  # nothing to gain where the lane line already runs on
        near = [float(v) for v in along if f0 <= v <= f1]
        for a in [lo] + [v for v in near if v <= sj + 1e-6]:
          for b in [hi] + [v for v in near if v >= sj - 1e-6]:
            if b - a >= 1.0 and (cost := bend(a, b)) < best - JOG_BETTER:
              best, lo, hi = cost, a, b
    inside = (s >= lo) & (s <= hi)
    if hi > lo and inside.any():
      out[inside] = np.interp(s[inside], [lo, hi], ends(lo, hi))
      if hi <= sj + 1e-6:
        out[(s >= sj - 1e-6) & (s <= sj + 1e-6)] = o_out
  return out


OPENED = 0.5  # m: a lane narrower than this is still opening, the line beside it not yet painted
DOUBLE_LINE = 0.15  # m from a double line's middle to each line: a closing median's edges come no nearer each other


def line_offset(key: tuple, sec: Section) -> float:
  """m right of the line of a keyed line (WayLanes.keyed_lines) on a cross-section."""
  kind = key[0]
  if kind == 'edge':
    return sec.edges[key[1]]
  if kind == 'lane':
    return sec.spans[key[1]].right
  a, b = sec.spans[key[1]].right, sec.spans[key[1] + 1].left
  mid = (a + b) / 2
  return min(a, mid - DOUBLE_LINE) if key[2] == 0 else max(b, mid + DOUBLE_LINE)


def _lerp(a: Section, b: Section, t: float) -> Section:
  """A cross-section t (0-1) of the way from a to b, two of one road's with the same lanes."""
  if len(a.spans) != len(b.spans):
    return b if t >= 0.5 else a
  t = min(max(t, 0.0), 1.0)
  spans = [Span(m.lane, o.left + (m.left - o.left) * t, o.right + (m.right - o.right) * t, m.heading)
           for o, m in zip(a.spans, b.spans, strict=True)]
  return Section(spans, tuple(o + (m - o) * t for o, m in zip(a.edges, b.edges, strict=True)))


def _taperable(a: Section, b: Section) -> bool:
  """Whether one road's lanes become the other's by lanes beginning or ending on the outside of ours alone."""
  return a.back == b.back and a.lanes != b.lanes and min(a.lanes, b.lanes) > 0


def _left_only(sec: Section) -> bool:
  return bool(sec.turns) and bool(sec.turns[0]) and sec.turns[0] <= LEFTS


def _right_only(sec: Section) -> bool:
  return bool(sec.turns) and bool(sec.turns[-1]) and sec.turns[-1] <= RIGHTS


def _opens_left(few: Section, many: Section) -> bool:
  """Whether the lanes a road gains begin on the left of ours: by the turn arrows where they say (_turn_side), else
  where our lanes before line up with the right of the lanes after better than with their left by SIDE_MARGIN a lane
  (the two ways' lines in place), else not (the road widens on the outside)."""
  side = _turn_side(few, many)
  if side is not None:
    return side
  extra = many.lanes - few.lanes
  if extra <= 0 or not few.lanes:
    return False
  fc, mc = [s.centre for s in few.ours], [s.centre for s in many.ours]
  left = sum(abs(a - b) for a, b in zip(fc, mc[extra:], strict=True))
  right = sum(abs(a - b) for a, b in zip(fc, mc[:few.lanes], strict=True))
  return right - left > SIDE_MARGIN * few.lanes


SIDE_MARGIN = 0.5  # m a lane


def _turn_side(a: Section, b: Section) -> bool | None:
  """Where the road's lanes change from a to b in number, the side the lanes begin or end on by their turn arrows:
  True on the left (a lane that only turns left is new in b, or a's ends), False on the right (one that only turns
  right), the left where both are (a left bay opening as the kerb lane becomes a right-turn lane); None where the
  arrows don't say. A bay opens where its road widens the other side as often as on its own (the lanes moving across
  over its taper), so where the lanes lie can't tell."""
  few, many = (a, b) if a.lanes < b.lanes else (b, a)
  if few.lanes == many.lanes or not few.lanes:
    return None
  if _left_only(many) and not _left_only(few):
    return True
  return False if _right_only(many) and not _right_only(few) else None


ARROWS_BACK = 250.0  # m before a junction its lanes' arrows show from, while the lanes carry on unchanged (turns_at)
ARROWS_REACH = 60.0  # m past a change of our lanes that the arrows of the lanes there say which side they changed on


def _median_side(a: Section, b: Section) -> bool | None:
  """Where one direction's carriageway of a divided road and the two-way road it becomes meet (a one-way's lanes and
  a road's with oncoming lanes, as many or not), the side our lanes begin or end on: the side the other direction is
  on, where the median was (True: on the left). The outside kerb carries on, whatever the ways' lines say. None
  otherwise."""
  if a.lanes == b.lanes or not a.lanes or not b.lanes:
    return None
  one, two = (a, b) if not a.two_way else (b, a)
  if one.two_way or not two.two_way:
    return None
  left = any(sp.heading == -1 for sp in two.spans[:two.first])
  right = any(sp.heading == -1 for sp in two.spans[two.first + two.lanes:])
  return None if left == right else left


LANE_GAP = 0.5  # of a lane's width: the cost of a lane carrying on as none, matching lanes across a change by place
GAP_BIAS = 1e-3  # m: between matches that cost the same, lanes begin or end on the right
SHIFT_COST = 0.1  # of each metre a's lanes are moved across to match b's (match_lanes)


def match_lanes(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[int | None]:
  """Which of lanes b each of lanes a carries on as (None: it ends), both [(left, right) m right of one line] left to
  right: the order-keeping match by where their middles are that costs least, a lane left without a counterpart
  (beginning or ending, a lane opening from nothing costing nothing) LANE_GAP of its width. The line may move under the
  lanes (two ways' lines placed apart at a node, or a junction's roads in and out): a's lanes are tried where they lie,
  and moved across by as much as either edge of them moves, or both, SHIFT_COST a metre moved. Where matches cost the
  same, lanes begin and end on the right."""
  if not a or not b:
    return [None] * len(a)
  dl, dr = b[0][0] - a[0][0], b[-1][1] - a[-1][1]
  best = None
  for shift in sorted({0.0, dl, dr, (dl + dr) / 2}, key=abs):
    cost, out = _align([(x + shift, y + shift) for x, y in a], b)
    cost += SHIFT_COST * abs(shift)
    if best is None or cost < best[0] - EPS:
      best = (cost, out)
  return best[1]


def _align(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> tuple[float, list[int | None]]:
  na, nb = len(a), len(b)
  ca, cb = [(x + y) / 2 for x, y in a], [(x + y) / 2 for x, y in b]

  def gap(lanes, p):
    return LANE_GAP * max(lanes[p][1] - lanes[p][0], 0.0) + GAP_BIAS * (len(lanes) - p) / len(lanes)
  inf = math.inf
  cost = [[inf] * (nb + 1) for _ in range(na + 1)]
  step = [[0] * (nb + 1) for _ in range(na + 1)]
  cost[0][0] = 0.0
  for i in range(na + 1):
    for j in range(nb + 1):
      if i and j and cost[i - 1][j - 1] + abs(ca[i - 1] - cb[j - 1]) < cost[i][j]:
        cost[i][j], step[i][j] = cost[i - 1][j - 1] + abs(ca[i - 1] - cb[j - 1]), 0
      if i and cost[i - 1][j] + gap(a, i - 1) < cost[i][j]:
        cost[i][j], step[i][j] = cost[i - 1][j] + gap(a, i - 1), 1
      if j and cost[i][j - 1] + gap(b, j - 1) < cost[i][j]:
        cost[i][j], step[i][j] = cost[i][j - 1] + gap(b, j - 1), 2
  out: list[int | None] = [None] * na
  i, j = na, nb
  while i or j:
    k = step[i][j]
    if k == 0:
      out[i - 1] = j - 1
      i, j = i - 1, j - 1
    elif k == 1:
      i -= 1
    else:
      j -= 1
  return cost[na][nb], out


def _blend(few: Section, many: Section, t: float) -> Section:
  """`many`'s lanes, t (0-1) of the way from `few`'s: its extra lanes of ours (on the side its arrows turn to, else the
  outer side, right of ours) at no width at t = 0."""
  t = min(max(t, 0.0), 1.0)
  extra = many.lanes - few.lanes
  left = _opens_left(few, many)
  fo, mo = few.ours, many.ours
  edge = fo[0].left if left else fo[-1].right
  zero = [Span(s.lane, edge, edge, 1) for s in (mo[:extra] if left else mo[-extra:])]
  old = few.spans[:few.first] + (zero + fo if left else fo + zero) + few.spans[few.first + few.lanes:]
  if len(old) != len(many.spans):
    return many
  spans = [Span(m.lane, o.left + (m.left - o.left) * t, o.right + (m.right - o.right) * t, m.heading)
           for o, m in zip(old, many.spans, strict=True)]
  edges = tuple(o + (m - o) * t for o, m in zip(few.edges, many.edges, strict=True))
  return Section(spans, edges)


BLEND_M = 10.0  # m either side of a node the lanes move across over where a road carries on from one way to the next
BLEND_KNOTS = 4  # straight pieces the smoothstep is drawn in on each side


def _same_lanes(a: Section, b: Section) -> bool:
  """Whether two cross-sections have the same lanes in the same order, whatever their widths and where the line runs."""
  return len(a.spans) == len(b.spans) and a.first == b.first and all(x.heading == y.heading for x, y in zip(a.spans, b.spans, strict=True))


def _apart(a: Section, b: Section) -> float:
  """m the furthest of two cross-sections' (the same lanes') lane edges and kerbs are apart."""
  return max([abs(x - y) for x, y in zip(a.edges, b.edges, strict=True)] +
             [max(abs(x.left - y.left), abs(x.right - y.right)) for x, y in zip(a.spans, b.spans, strict=True)])


def _toward(own: Section, other: Section, w: float) -> Section:
  """own's lanes, moved w (0-1) of the way across to where other's (the same lanes) are."""
  spans = [Span(o.lane, o.left + (m.left - o.left) * w, o.right + (m.right - o.right) * w, o.heading)
           for o, m in zip(own.spans, other.spans, strict=True)]
  return Section(spans, tuple(o + (m - o) * w for o, m in zip(own.edges, other.edges, strict=True)))


def _smooth(u: float) -> float:
  u = min(max(u, 0.0), 1.0)
  return u * u * (3.0 - 2.0 * u)


def mirrored(sec: Section) -> Section:
  """A cross-section seen travelling the other way."""
  return Section([Span(s.lane, -s.right, -s.left, -s.heading) for s in sec.spans[::-1]], (-sec.edges[1], -sec.edges[0]))


def at_knots(knots: list[tuple[float, Section]], v: float) -> Section:
  """The cross-section v m along a way from its knots [(m, Section)] (taper(), blend()), linear between them."""
  ks = [k for k, _ in knots]
  i = bisect.bisect_right(ks, v) - 1
  if i < 0:
    return knots[0][1]
  if i >= len(knots) - 1:
    return knots[-1][1]
  return _lerp(knots[i][1], knots[i + 1][1], (v - ks[i]) / max(ks[i + 1] - ks[i], EPS))
