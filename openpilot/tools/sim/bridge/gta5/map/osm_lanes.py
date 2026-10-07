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
- `divider`: the marking between the directions. `lane_markings=no`: no lines between lanes at all.
- Missing tags fall back to OSM's defaults, then to `Defaults` by road class: lanes 1 each way (2 on a one-way motorway
  or trunk), a single track on tracks, the line in the middle.

Geometry is in metres in a frame with y 90 degrees left of x (x east, y north): "right" is right of the direction of
travel. Traffic drives on the right unless `drive_on_right=False`, which mirrors where each direction's lanes are.
"""
import re
from dataclasses import dataclass
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
  def from_tags(cls, tags: dict, drive_on_right: bool = True, defaults: Defaults = GTA) -> 'WayLanes':
    highway, oneway = tags.get('highway', ''), oneway_of(tags)
    fwd, back, both = lane_counts(tags)
    backs = [(BACKWARD, k) for k in reversed(range(back))]  # seen along the way, backward lanes run from the far one
    order = [*backs, *((BOTH_WAYS, k) for k in range(both)), *((FORWARD, k) for k in range(fwd))] if drive_on_right else \
      [*((FORWARD, k) for k in range(fwd)), *((BOTH_WAYS, k) for k in range(both)), *backs]
    own = {d: {k: i for i, (dd, k) in enumerate(order) if dd == d} for d in (FORWARD, BACKWARD, BOTH_WAYS)}

    def per_lane(key):
      out: list[str | None] = [None] * len(order)
      plain = tags.get(f'{key}:lanes')
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
        v = tags.get(f'{key}:lanes{suffix}')
        if v is not None and len(items := v.split('|')) == len(own[d]):
          for k, item in enumerate(items):
            out[own[d][k]] = item
      return out

    widths = [metres(v) for v in per_lane('width')]
    parking = (parking_lane(tags, 'left'), parking_lane(tags, 'right'))
    width = metres(tags.get('width'))
    if width:
      width = max(width - sum(parking), 0.0)  # the carriageway's width includes its parking lanes
    known, unknown = sum(w for w in widths if w), sum(not w for w in widths)
    if unknown:
      share = (width - known) / unknown if width and width - known > EPS * unknown else defaults.lane_width(highway)
      widths = [w or share for w in widths]
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
    forbids crossing), and the centre line between the directions or the edges of a median (divider=*; by default
    dashed with one lane each way, else double solid), and where a parking lane meets the lanes."""
    sec = self.section(direction)
    lo, hi = self.edges(direction)
    parking = self.parking_lanes(direction)
    out = [Line(EDGE, lo, None)] + [Line(PARKING, b, None) for a, b in parking if a == lo]
    if self.markings:
      wide = max(self.counts[:2]) >= 2
      centre = DIVIDERS.get(self.divider, 'solid') if self.divider else ('double_solid' if wide else 'dashed')
      for a, b in zip(sec, sec[1:], strict=False):
        if a.heading == b.heading != 0:
          a_may = a.lane.change_right if a.heading == 1 else a.lane.change_left
          b_may = b.lane.change_left if b.heading == 1 else b.lane.change_right
          style = {(True, True): 'dashed', (False, False): 'solid', (True, False): 'dashed_solid', (False, True): 'solid_dashed'}[(a_may, b_may)]
          out.append(Line(DIVIDER, a.right, style))
        elif 0 in (a.heading, b.heading):  # a centre turn lane's edge: dashed on its side
          out.append(Line(CENTRE, a.right, 'dashed_solid' if a.heading == 0 else 'solid_dashed'))
        elif centre is None:
          continue
        elif b.left - a.right > EPS:
          out += [Line(MEDIAN, a.right, centre), Line(MEDIAN, b.left, centre)]
        else:
          out.append(Line(CENTRE, a.right, centre))
    out += [Line(PARKING, a, None) for a, b in parking if b == hi]
    out.append(Line(EDGE, hi, None))
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
CORNER_SHARP, SHARP_SHARE = 10.0, 0.6  # a turn through a junction turns this share of its angle within this many m
FILLET_RADIUS = {'left': 20.0, 'right': 12.0}  # m at most, about as wide as GTA's AI drives its turns
MATCH_TOL = 0.5  # m between a route's shape point and a map node it is at
DROP_SPAN = 25.0  # m: junction nodes this near each other along a route are one junction, for the lanes through it
DROP_TURN = 30.0  # deg at most a route turns going straight on through a junction (over DROP_REACH m either side)
DROP_REACH = 15.0  # m
SAME_WIDTH = 0.5  # m: roads either side of a junction this near as wide are one road carrying on, kerb to kerb


class Section:
  """A road's cross-section where a route runs along it, seen in the route's direction of travel: its lanes left to
  right (ours, oncoming and centre lanes), m right of the route's line. Lanes are numbered from the left of ours: 0 to
  lanes - 1 are ours, negative ones left of them (the oncoming lanes where traffic drives on the right)."""
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

  @property
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


def ways_from_nodes(points, osm: OsmLanes, tol: float = MATCH_TOL) -> list[tuple[int, bool, int, int]]:
  """The ways along a route whose shape points are the map's nodes, as on our GTA map, where every route point is one
  (but the ends, part way along a way): [(way id, along the way's direction, first shape index, last shape index)] for
  each run of segments on one way. Among nodes at the same place (over and under each other), those joined along the
  route."""
  pts = np.asarray(points, float)[:, :2]
  found = [osm.nodes_at(p, tol) for p in pts]
  nodes: list[int | None] = []
  for k, cands in enumerate(found):
    prev = nodes[-1] if nodes else None
    nxt = found[k + 1] if k + 1 < len(found) else []
    joined = [c for c in cands if (prev is not None and (prev, c) in osm.pairs) or any((c, n) in osm.pairs for n in nxt)]
    nodes.append((joined or cands or [None])[0])
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
               junctions=None):
    self.points = np.asarray(points, float)[:, :2]
    self.along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(self.points, axis=0).T))))
    self.sections = sections
    self.arrows: list[tuple[float, list[frozenset[str]]]] = arrows or []  # (m along where they end, each lane's)
    self.tapers: dict[int, tuple[Section, Section]] = tapers or {}  # first segment of a way: (lanes before, after)
    # m along to the nodes where roads meet: a corner near one is a turn through a junction, the rest are bends
    self.junctions = np.sort(np.asarray(junctions if junctions is not None else [], float))
    self._corners: list[tuple[float, float]] | None = None
    self._drops: list[tuple[float, tuple[int, int, int]]] | None = None

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
    tapers: dict[int, tuple[Section, Section]] = {}
    degree = [osm.degree.get(n[0], 0) if (n := osm.nodes_at(p)) else 0 for p in pts]
    for k, sec in enumerate(sections):
      nxt = sections[k + 1] if k + 1 < len(sections) else None
      if sec is not None and any(sec.turns) and not (nxt is not None and any(nxt.turns) and nxt.lanes == sec.lanes):
        arrows.append((float(along[k + 1]), sec.turns))  # the end of a run of ways with arrows: the junction
      prev = sections[k - 1] if k else None
      if prev is not None and sec is not None and ways[k - 1] != ways[k] and degree[k] == 2 and _taperable(prev, sec):
        tapers[k] = (prev, sec)
    junctions = [float(along[k]) for k in range(1, len(pts) - 1) if degree[k] >= 3]
    return cls(pts, sections, arrows, tapers, junctions)

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

  def segment(self, s: float) -> int:
    return int(min(max(np.searchsorted(self.along, s, side='right') - 1, 0), max(len(self.sections) - 1, 0)))

  def section_at(self, s: float, k: int | None = None) -> Section | None:
    """The cross-section s m along (on segment k, where s is at a point segments share), a lane widening from nothing
    over TAPER_M where a way's lane count has risen, or narrowing to nothing before it falls."""
    k = self.segment(s) if k is None else k
    sec = self.sections[k] if 0 <= k < len(self.sections) else None
    if sec is None or not self.tapers:
      return sec
    k0 = k
    while k0 > 0 and self.sections[k0 - 1] is sec:
      k0 -= 1
    k1 = k
    while k1 + 1 < len(self.sections) and self.sections[k1 + 1] is sec:
      k1 += 1
    if k0 in self.tapers and s - self.along[k0] < TAPER_M and self.tapers[k0][0].lanes < sec.lanes:
      return _blend(self.tapers[k0][0], sec, (s - self.along[k0]) / TAPER_M)
    if k1 + 1 in self.tapers and self.along[k1 + 1] - s < TAPER_M and self.tapers[k1 + 1][1].lanes < sec.lanes:
      return _blend(self.tapers[k1 + 1][1], sec, (self.along[k1 + 1] - s) / TAPER_M)
    return sec

  def opening(self, k: int) -> tuple[float, int, bool] | None:
    """Where lanes begin on the way segment k is on, as its lane count rises: (m along where they are fully there, how
    many, whether on the left of ours); None where none begin. Fully there at the end of their taper, or half way
    along a way shorter than one."""
    sec = self.sections[k] if 0 <= k < len(self.sections) else None
    if sec is None or not self.tapers:
      return None
    k0, k1 = k, k
    while k0 > 0 and self.sections[k0 - 1] is sec:
      k0 -= 1
    while k1 + 1 < len(self.sections) and self.sections[k1 + 1] is sec:
      k1 += 1
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
    back = max(at - FILLET_REACH - CORNER_CHORD, 0.0)  # from a little behind, so a corner the car is in still has its fillet
    kx = np.array([at + d for d, _ in keys], dtype=float) + np.arange(len(keys)) * 1e-3  # a step where two share a place
    ky = np.array([lane for _, lane in keys], dtype=float)
    inside = self.along[(self.along > back) & (self.along < self.along[-1])]
    grid = np.arange(back, self.along[-1], step)
    s2 = np.unique(np.concatenate(([back], inside, grid, kx[(kx > back) & (kx < self.along[-1])], [self.along[-1]])))
    s2 = s2[np.concatenate(([True], np.diff(s2) > 1e-3))]
    xy = np.stack([np.interp(s2, self.along, self.points[:, 0]), np.interp(s2, self.along, self.points[:, 1])], axis=1)
    lane = np.interp(s2, kx, ky)
    seg = np.clip(np.searchsorted(self.along, (s2[:-1] + s2[1:]) / 2, side='right') - 1, 0, len(self.sections) - 1)
    offs = np.full(len(s2), np.nan)
    for i, k in enumerate(seg):
      for v in (i, i + 1):  # the piece's ends, averaged with the next's at a shared point
        sec = self.section_at(s2[v], k)
        if sec is None or not sec.lanes:
          continue
        off = sec.offset(lane[v])
        offs[v] = off if np.isnan(offs[v]) else (offs[v] + off) / 2
    known = ~np.isnan(offs)
    if not known.any():
      return None
    offs = np.interp(s2, s2[known], offs[known])
    line = offset_line(xy, offs)
    if len(line) != len(s2):
      return None
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
      a = sc - min(FILLET_REACH, (sc - lo) / 2)
      b = sc + min(FILLET_REACH, (hi - sc) / 2)
      if a - CORNER_CHORD < s[0] or b + CORNER_CHORD > s[-1]:
        continue

      def at(v, pts=line):
        return np.array([np.interp(v, s, pts[:, 0]), np.interp(v, s, pts[:, 1])])

      def route(v):
        return np.array([np.interp(v, self.along, self.points[:, 0]), np.interp(v, self.along, self.points[:, 1])])
      u_in, u_out = route(a) - route(a - CORNER_CHORD), route(b + CORNER_CHORD) - route(b)
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


def _taperable(a: Section, b: Section) -> bool:
  """Whether one road's lanes become the other's by lanes beginning or ending on the outside of ours alone."""
  return a.back == b.back and a.lanes != b.lanes and min(a.lanes, b.lanes) > 0


def _opens_left(few: Section, many: Section) -> bool:
  """Whether the lanes a road gains begin on the left of ours: where its leftmost lane's arrows only turn left, and
  the road's before didn't (a turn bay there already, the road widens on the outside)."""
  def left_only(sec):
    return bool(sec.turns) and bool(sec.turns[0]) and sec.turns[0] <= LEFTS
  return left_only(many) and not left_only(few)


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
