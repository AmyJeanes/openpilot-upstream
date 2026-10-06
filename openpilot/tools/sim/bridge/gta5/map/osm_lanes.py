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
- `width`: kerb to kerb, metres. Without `width:lanes` it's shared out evenly; on a two-way way, what's left after
  `width:lanes` is a median centred between the directions.
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

# line kinds: a road's edge (kerb), between lanes one way, between the directions, a median's edge
EDGE, DIVIDER, CENTRE, MEDIAN = 'edge', 'divider', 'centre', 'median'


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
  def __init__(self, lanes: list[Lane], width: float, median: float, margin: float, markings: bool, divider: str | None):
    self.lanes = lanes
    self.width = width  # m, kerb to kerb
    self.line = width / 2  # m from the left kerb to the way's line
    self.margin = margin  # m between each kerb and its outer lane
    self.markings, self.divider = markings, divider
    self.placed: dict[str, float] = {}  # where each placement tag puts the line, m from the left kerb
    self.tagged: tuple[float | None, float, int] = (None, 0.0, 0)  # width=*, the lanes' width:lanes total, lanes without
    self.single_track = all(lane.direction == BOTH_WAYS for lane in lanes)
    turns = [i for i in range(1, len(lanes)) if lanes[i - 1].direction != lanes[i].direction]
    x, self.x, self.gaps = margin, [], []  # each lane's left edge and the median (either side of any centre lanes)
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
    width = metres(tags.get('width'))
    known, unknown = sum(w for w in widths if w), sum(not w for w in widths)
    if unknown:
      share = (width - known) / unknown if width and width - known > EPS * unknown else defaults.lane_width(highway)
      widths = [w or share for w in widths]
    lanes_w = sum(widths)
    spare = width - lanes_w if width and width - lanes_w > EPS else 0.0
    width = lanes_w + spare
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
    road = cls(lanes, width, spare if two_way else 0.0, 0.0 if two_way else spare / 2, markings, tags.get('divider'))

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
    road.tagged = (metres(tags.get('width')), known, unknown)
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

  def medians(self, direction: int = FORWARD) -> list[tuple[float, float]]:
    """The median between the directions, m right of the line: in two halves either side of a centre turn lane."""
    out = [(a - self.line, b - self.line) for a, b in self.gaps]
    return out if direction == FORWARD else [(-b, -a) for a, b in out[::-1]]

  def lines(self, direction: int = FORWARD) -> list[Line]:
    """The lines on the road left to right: its edges, white lines between lanes one way (solid where change:lanes
    forbids crossing), and the centre line between the directions or the edges of a median (divider=*; by default
    dashed with one lane each way, else double solid)."""
    sec = self.section(direction)
    lo, hi = self.edges(direction)
    out = [Line(EDGE, lo, None)]
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


def offset_line(points, right: float) -> np.ndarray:
  """A polyline (n, 2) moved `right` m to its right, mitred at the corners."""
  p = np.asarray(points, dtype=float)[:, :2]
  keep = np.concatenate(([True], np.hypot(*np.diff(p, axis=0).T) > 1e-6))
  p = p[keep]
  if len(p) < 2 or right == 0:
    return p.copy()
  d = np.diff(p, axis=0)
  d /= np.hypot(d[:, 0], d[:, 1])[:, None]
  n = np.stack([d[:, 1], -d[:, 0]], axis=1)  # right of each segment
  normal = np.concatenate([n[:1], n[:-1] + n[1:], n[-1:]])
  normal /= np.maximum(np.hypot(normal[:, 0], normal[:, 1]), 1e-9)[:, None]
  cos = np.concatenate([[1.0], np.einsum('ij,ij->i', normal[1:-1], n[1:]), [1.0]])
  scale = 1.0 / np.maximum(cos, 1.0 / MITER_LIMIT)
  return p + normal * (right * scale)[:, None]
