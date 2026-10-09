"""Surveys of the paint on GTA's roads as per-link corrections to the lanes ynd_to_osm.py writes, where its class rules
(layout) are only right on average. Two sources write the same format:
- "src": "gamefiles": the paint read from the game's own road meshes, textures and marking decals (roadpaint/), exact to
  a few cm, with the asphalt's edges and the painted arrows' kinds. The primary source.
- the map audit's top-down camera over the running game ("src" absent): looser (kerbs, cracks and junction guide lines
  also show as white lines, a double line's kind is often misread, arrows are unreliable). Used where the game files
  don't cover a link, and as a cross-check where both do.

A survey file has a JSON object per line, one per cross-section read across a GTA link:
  {"a": "528:241", "b": "528:243",  the link by its nodes' ids (area:index)
   "s": 7.5, "dir": "ab",           where along it, and which way the offsets are seen ("ab": travelling a to b)
   "marks": [{"type": "dashed", "colour": "white", "offset": 5.3, "conf": 0.9, "pair": [5.2, 5.4] (double lines)}, ...],
   "arrows": [{"offset": 4.9, "kind": "left|through|right|through;left|...", "dir": "ab|ahead|oncoming", "conf": 0.95}],
   "kerbs": {"left": -11.3, "right": 11.4, "conf": 0.8}, "junction": false, "bay": false, "src": "gamefiles"}
offsets in m right of the link's line, seen in the "dir" direction.

One-way links are corrected from the game files alone (correct_oneway: their painted edges and lane lines; where they
are one of several links side by side making up a carriageway, correct_carriageway: their lanes in it). Two-way
links are corrected from either source, only where the samples agree with each other and with GTA's lane counts:
- a yellow centre (one line, or a median's two edges more than MEDIAN_MIN apart) in at least MIN_SAMPLES samples and
  half of them, within CENTRE_REACH of the link's line;
- in each direction exactly its lanes less one white lane lines (dashed, solid or raised markers), seen as often,
  agreeing within AGREE;
- every lane LANE_MIN to LANE_MAX wide, its outer lanes running out to the kerbs: the game files' asphalt edges where
  they are seen on both sides about as far from the line (KERB_TOL), else where the class layout has them.
A way's line can't leave GTA's nodes, so the line stays put and the paint moves round it: with as many lanes each way,
or kerbs as far either side of it, the line is the middle of the road between the kerbs (OSM's default reading), so a
centre off the line is said by the lanes' widths alone; else (more lanes one way) the line is the centre's
(placement), which must then be on the line (within CENTRE_TOL). Lane counts never change here; disagreements are only
counted.
The game files' kinds of the lines between lanes (solid, solid on one half) give the lanes their change:lanes. Where
the lanes stay the class layout, lane_lines still reads the kinds of the lines between them, and
unpainted which roads have no lines at all. Other fields (z, a mark's width / cover / line id, kerb_step, hatched spans, stop lines, other
features) are read past.
"""
import json
import math
from collections import Counter, defaultdict

import numpy as np

CONF = 0.7  # a mark's detector confidence
MIN_SAMPLES = 2
AGREE = 0.4  # m between one line's offsets in different samples
MEDIAN_MIN = 1.0  # m between two yellow lines bounding a painted median, rather than one double line
CENTRE_REACH = 3.0  # m from the link's line to its centre
CENTRE_TOL = 0.4  # m, where the line must be the centre's
KERB_TOL = 0.4  # m between the two kerbs' distances from the line, to take them as the road's
COUNT_KERB_TOL = 0.8  # m, as KERB_TOL, where the painted lanes are more or fewer than GTA's and run out to the edges
KERB_REACH = 2.5  # m between a seen kerb and where the class layout has it
LANE_MIN, LANE_MAX = 3.0, 7.0  # m
LANE_LINES = ('dashed', 'solid', 'markers')  # white lane lines; the camera's double and edge lines are mostly kerbs
# the game files' white lines between lanes, by kind: (may cross from its left, from its right), halves given left to
# right as seen travelling a -> b
CROSSING = {'dashed': (True, True), 'markers': (True, True), 'double_dashed': (True, True), 'solid': (False, False),
            'double_solid': (False, False), 'solid_dashed': (False, True), 'dashed_solid': (True, False)}
# OSM's divider=* for a centre line's kind (halves left to right along the way; OSM has no double dashed line)
DIVIDER = {'double_solid': 'double_solid_line', 'solid': 'solid_line', 'dashed': 'dashed_line', 'double_dashed': 'dashed_line',
           'solid_dashed': 'solid_line;dashed_line', 'dashed_solid': 'dashed_line;solid_line'}
HALVES_SWAPPED = {'solid_dashed': 'dashed_solid', 'dashed_solid': 'solid_dashed'}  # a line seen the other way
CHANGE = {(True, True): 'yes', (False, False): 'no', (False, True): 'not_left', (True, False): 'not_right'}
GAMEFILES = 'gamefiles'


def node_key(s: str) -> tuple[int, int]:
  a, i = s.split(':')
  return int(a), int(i)


SOLID_PAINTED = 0.7  # of a line's length painted: solid with worn or hidden stretches, not dashed
DASH_PERIOD = 4.0  # m: a "dashed" line repeating faster than this is a solid one laid as tiled decals
MARKER_KERB = 1.0  # m: raised markers this near the asphalt's edge or the kerb are the gutter's edge, not paint
DASH_PIECE = 6.0  # m: a solid polyline no longer than this with another in line a gap ahead or behind is one dash
DASH_GAP = 14.0  # m at most between two such dashes' ends
IN_LINE = 0.3  # m between one dash's middle and the other's line, and IN_LINE_BEND more a metre ahead (on a bend)
IN_LINE_BEND = 0.15
DASH_TURN = 20.0  # deg at most between two dashes' headings
LOCAL = 12.0  # m either side of a section that a line's dashes are read over, for its kind there
SOLID_STRETCH = 20.0  # m: a dash this long is a solid stretch of the line
DASH_APART = 1.0  # m at least between dashes
DASH_EVEN = 1.0  # m at most between the longest and shortest of evenly laid dashes, and of their gaps


class LineKinds(dict):
  """{polyline id: its kind by its whole length}, and where the dashes of lines with dashes are (local), so a section
  reads the kind a line has where it crosses it (kind_at): a dashed line painted solid for a stretch, or the other way."""
  def __init__(self):
    super().__init__()
    self.local: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}  # id -> (points [N, 2], m along each, dashes [D, 2])

  def kind_at(self, i, x: float | None = None, y: float | None = None) -> str | None:
    kind = self.get(i)
    if kind is None or i not in self.local or x is None or y is None:
      return kind
    pts, arc, dashes = self.local[i]
    s = float(_project(pts, np.array([[x, y]], dtype=float))[0][0])
    s *= float(arc[-1]) / max(float(np.hypot(*np.diff(pts, axis=0).T).sum()), 1e-9)  # along the line's own (3D) length
    inside = dashes[(dashes[:, 0] <= s) & (dashes[:, 1] >= s)]
    if len(inside) and float((inside[:, 1] - inside[:, 0]).max()) >= SOLID_STRETCH:
      return 'solid'
    near = dashes[(dashes[:, 1] > s - LOCAL) & (dashes[:, 0] < s + LOCAL)]
    if len(near) < 2:
      return kind
    painted = float((np.minimum(near[:, 1], s + LOCAL) - np.maximum(near[:, 0], s - LOCAL)).sum()) / (2 * LOCAL)
    lengths, gaps = near[:, 1] - near[:, 0], near[1:, 0] - near[:-1, 1]
    # dashes: short, apart, and mostly gap or evenly laid (a worn solid line's breaks aren't)
    regular = len(near) >= 3 and np.ptp(lengths) <= DASH_EVEN and np.ptp(gaps) <= DASH_EVEN
    if float(lengths.max()) <= DASH_PIECE and float(gaps.min()) >= DASH_APART and float(np.diff(near[:, 0]).min()) >= DASH_PERIOD \
       and (painted < 0.5 or regular):
      return 'dashed'
    return kind


def line_kinds(path) -> LineKinds:
  """Whether each of the game files' polylines (polylines.jsonl: id, style, painted, dashes) is solid or dashed, by its
  whole length: a section's reading can call a solid line dashed where it's worn, interrupted or tiled. Where its
  dashes (and pts) show a stretch painted otherwise, a section there reads that (LineKinds.kind_at: Senora Rd's
  centre, solid for 900 m, then dashed). A line laid a decal a dash is short solid polylines in line: each is dashed."""
  out, pieces = LineKinds(), []
  with open(path) as f:
    for line in f:
      try:
        p = json.loads(line)
      except ValueError:
        continue
      starts = [d[0] for d in p.get('dashes') or []]
      period = float(np.median(np.diff(starts))) if len(starts) >= 3 else None
      solid = p['style'] == 'solid' or p.get('painted', 0) >= SOLID_PAINTED or (period is not None and period < DASH_PERIOD)
      out[p['id']] = 'solid' if solid else p['style']
      pts = np.array(p.get('pts') or [], dtype=float)
      if len(pts) < 2:
        continue
      if p['style'] == 'solid' and p.get('len', math.inf) <= DASH_PIECE:
        pieces.append((p['id'], p.get('colour'), pts[0, :2], pts[-1, :2]))
      elif len(starts) >= 2:
        arc = np.concatenate(([0.0], np.cumsum(np.sqrt((np.diff(pts, axis=0) ** 2).sum(1)))))
        out.local[p['id']] = (pts[:, :2], arc, np.array(p['dashes'], dtype=float))
  cells = defaultdict(list)
  for k, (_, _, a, b) in enumerate(pieces):
    cells[tuple(np.floor((a + b) / 2 / DASH_GAP).astype(int))].append(k)
  for i, colour, a, b in pieces:
    length = float(np.hypot(*(b - a)))
    u, mid = (b - a) / max(length, 1e-9), (a + b) / 2
    cx, cy = np.floor(mid / DASH_GAP).astype(int)
    for n in (n for dx in (-1, 0, 1) for dy in (-1, 0, 1) for n in cells.get((cx + dx, cy + dy), ())):
      j, other, c, d = pieces[n]
      v, m = d - c, (c + d) / 2
      ahead = abs(float((m - mid) @ u))
      gap = ahead - (length + float(np.hypot(*v))) / 2
      if j != i and other == colour and abs(float(v @ u)) >= math.cos(math.radians(DASH_TURN)) * float(np.hypot(*v)) and \
         abs(float((m - mid) @ np.array([-u[1], u[0]]))) <= IN_LINE + IN_LINE_BEND * ahead and DASH_PERIOD / 2 <= gap <= DASH_GAP:
        out[i] = 'dashed'
        break
  return out


HALVES = {('solid', 'solid'): 'double_solid', ('dashed', 'dashed'): 'double_dashed', ('solid', 'dashed'): 'solid_dashed',
          ('dashed', 'solid'): 'dashed_solid'}


def _clean(d: dict, lines: dict[int, str]) -> dict:
  """A game-file sample with its lines' kinds read off their whole polylines, and raised markers at the kerbs dropped."""
  kerbs = [v for k in ('kerbs', 'kerb_step') for v in (d.get(k) or {}).values() if isinstance(v, (int, float))]
  marks = []
  for m in d['marks']:
    if m['type'] == 'markers' and any(abs(m['offset'] - v) <= MARKER_KERB for v in kerbs):
      continue
    ids = m.get('line') if isinstance(m.get('line'), list) else [m.get('line')]
    kinds = [lines.kind_at(i, d.get('x'), d.get('y')) if isinstance(lines, LineKinds) else lines.get(i) for i in ids]
    if None not in kinds and m['type'] in ('dashed', 'solid') and len(kinds) == 1:
      m = {**m, 'type': kinds[0]}
    elif None not in kinds and len(kinds) == 2 and kinds[0] == kinds[1] and m['type'] in HALVES.values():
      m = {**m, 'type': HALVES[tuple(kinds)]}  # (which polyline is the left half isn't said: only where both agree)
    marks.append(m)
  return {**d, 'marks': marks}


def load(paths, lines: dict[int, str] | None = None) -> dict[tuple, list[dict]]:
  """{(a, b): [sample]} by link, each sample's offsets seen travelling a -> b; junction and bay sections left out, and
  lines that aren't whole JSON (a survey still being written). With the game files' `lines` (line_kinds), their
  samples' marks are cleaned (_clean)."""
  out = defaultdict(list)
  for path in paths:
    with open(path) as f:
      for line in f:
        try:
          d = json.loads(line)
        except ValueError:
          continue
        if d.get('junction') or d.get('bay'):
          continue
        if lines is not None and d.get('src') == GAMEFILES:
          d = _clean(d, lines)
        a, b = node_key(d['a']), node_key(d['b'])
        out[(b, a) if d.get('dir', 'ab') == 'ba' else (a, b)].append(d)
  return out


def _flip(d: dict) -> dict:
  kerbs = d.get('kerbs') or {}
  return {**d, **({'s': d['len'] - d['s']} if 'len' in d and 's' in d else {}),
          'marks': [{**m, 'offset': -m['offset'], 'pair': [-v for v in m['pair']][::-1] if m.get('pair') else None,
                     'type': HALVES_SWAPPED.get(m['type'], m['type'])} for m in d['marks']],
          **{key: [{**a, 'offset': -a['offset'], 'dir': 'oncoming' if a.get('dir') in ('ab', 'ahead') else 'ab'} if 'offset' in a else a
                   for a in d.get(key) or []] for key in ('arrows', 'features')},
          'kerbs': {**kerbs, 'left': -kerbs['right'] if kerbs.get('right') is not None else None,
                    'right': -kerbs['left'] if kerbs.get('left') is not None else None}}


def along(samples: dict, a, b) -> list[dict] | None:
  """A link's samples seen travelling a -> b, also from those seen the other way."""
  out = list(samples.get((a, b), [])) + [_flip(d) for d in samples.get((b, a), [])]
  return out or None


def rebase(samples: list[dict], p0, p1, q0, q1) -> list[dict]:
  """Samples seen along the line p0 -> p1 (each at its x, y on it) as seen along q0 -> q1, a link moved from where the
  survey read it: each mark, kerb and asphalt edge where it is in the world, measured right of the new line; those
  whose place falls off the new link's length are left out, and so are arrows and features."""
  p0, p1, q0, q1 = (np.asarray(v, float) for v in (p0, p1, q0, q1))
  u, v = (p1 - p0) / max(np.hypot(*(p1 - p0)), 1e-9), (q1 - q0) / max(np.hypot(*(q1 - q0)), 1e-9)
  r, rv, length = np.array([u[1], -u[0]]), np.array([v[1], -v[0]]), float(np.hypot(*(q1 - q0)))
  out = []
  for d in samples:
    if 'x' not in d:
      continue
    at = np.array([d['x'], d['y']])
    s = float((at - q0) @ v)
    if not 0.0 <= s <= length:
      continue

    def moved(o, at=at):
      return None if o is None else round(float((at + r * o - q0) @ rv), 3)
    marks = [{**m, 'offset': moved(m['offset']), 'pair': [moved(o) for o in m['pair']] if m.get('pair') else m.get('pair')} for m in d['marks']]
    sides = {key: {k: moved(o) if k in ('left', 'right') else o for k, o in (d.get(key) or {}).items()} for key in ('kerbs', 'kerb_step')}
    out.append({**d, 's': s, 'len': length, 'x': float(q0[0] + v[0] * s), 'y': float(q0[1] + v[1] * s), 'marks': marks,
                **sides, 'arrows': [], 'features': []})
  return out


TAPER_EDGE = 0.08  # of the way across: a median's right edge this near where it was (or ends up) is out of the taper
TAPER_LENGTH = (3.0, 80.0)  # m: a taper shorter or longer than these is misread
MEDIAN_SEEN = 2.0  # m at least the median's right edge swings across
ARRIVED = 0.6  # m: the edge this near the median's left edge has arrived, the two lines a double yellow


def opening_taper(sections: list[tuple[float, dict]], median: float) -> tuple[float, float] | None:
  """Where a turn lane opens in a two-way road's median, from the game files' sections along the road ([(m along it,
  the section seen travelling it)]): the median's right edge swings across to its left edge, the oncoming lanes', the
  lane opening behind it, as GTA paints its bays. (m along where the edge leaves its place, where it arrives),
  interpolated between the sections either side; None where the sections don't show one."""
  rows = []  # (m along, the rightmost yellow line: the median's right edge, or once across its left edge, its polyline)
  seen = []  # each row's yellow polylines
  for d, sample in sorted(sections, key=lambda r: r[0]):
    if sample.get('src') != GAMEFILES:
      continue
    ys = [(sum(m['pair']) / 2 if m.get('pair') else m['offset'], m.get('line')) for m in sample['marks']
          if m['colour'] == 'yellow' and m['conf'] >= CONF and abs(m['offset']) <= median / 2 + 1.5]
    if ys:
      y, line = max(ys, key=lambda v: v[0])
      rows.append((d, y, line if isinstance(line, int) else None))
      seen.append({i for _, v in ys for i in (v if isinstance(v, list) else [v]) if isinstance(i, int)})
  if len(rows) < 4:
    return None
  y0, y1 = float(np.median([y for _, y, _ in rows[:3]])), float(np.median([y for _, y, _ in rows[-3:]]))
  if y0 - y1 < MEDIAN_SEEN:  # no edge swinging across
    return None
  pts = [(d, float(np.clip((y0 - y) / (y0 - y1), 0.0, 1.0))) for d, y, _ in rows]
  # the swinging edge is often dashed: sections through its gaps, between where it's seen, read it as across already
  for line in {line for (_, p), (_, _, line) in zip(pts, rows, strict=True) if line is not None and TAPER_EDGE < p < 1 - TAPER_EDGE}:
    on = [k for k, s in enumerate(seen) if line in s]
    gaps = {k for k in range(on[0], on[-1]) if line not in seen[k]}
    pts, seen = [pt for k, pt in enumerate(pts) if k not in gaps], [s for k, s in enumerate(seen) if k not in gaps]
  # where the edge leaves its place and where it arrives, between the sections either side
  across = 1 - max(TAPER_EDGE, ARRIVED / (y0 - y1))
  arrive = next((k for k, (_, p) in enumerate(pts) if p >= across), None)
  if not arrive:
    return None
  leave = max((k for k in range(arrive) if pts[k][1] <= TAPER_EDGE), default=None)
  if leave is None:
    return None

  def cross(k, level):  # m along where the edge passes `level` between sections k and k + 1
    (d0, p0), (d1, p1) = pts[k], pts[k + 1]
    return d0 + (d1 - d0) * min(max((level - p0) / (p1 - p0), 0.0), 1.0) if p1 != p0 else d0
  start = cross(leave, TAPER_EDGE)
  end = cross(arrive - 1, across)
  # it opens once: shut before, open after (but for a stray line or two, as at a crossing)
  shut, open_ = [p for d, p in pts if d < start - 3.0], [p for d, p in pts if d > end + 3.0]
  if sum(p > 0.5 for p in shut) > 0.1 * len(shut) or sum(p < 0.5 for p in open_) > 0.1 * len(open_):
    return None
  return (float(start), float(end)) if TAPER_LENGTH[0] <= end - start <= TAPER_LENGTH[1] else None


SWING_STEP = 0.5  # m between the points a polyline is read at
SWING_BACK = 0.3  # m a swinging edge may wander back on its way across
RUNS_ON = 10.0  # m past a swing looked along for the median's right edge running on


def yellow_lines(path) -> list[tuple[int, np.ndarray]]:
  """The game files' yellow polylines (polylines.jsonl): [(id, points [N, 2], game x y)]."""
  out = []
  with open(path) as f:
    for line in f:
      try:
        p = json.loads(line)
      except ValueError:
        continue
      if p.get('colour') == 'yellow' and len(p.get('pts') or []) >= 2:
        out.append((p['id'], np.array(p['pts'], dtype=float)[:, :2]))
  return out


def _project(road: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Points as (m along the road polyline, m right of it), by the nearest of its segments."""
  a, d = road[:-1], np.diff(road, axis=0)
  length = np.hypot(d[:, 0], d[:, 1])
  starts = np.concatenate(([0.0], np.cumsum(length)))
  rel = pts[:, None, :] - a[None, :, :]
  t = np.clip((rel * d[None]).sum(-1) / np.maximum(length ** 2, 1e-9)[None], 0.0, 1.0)
  off = rel - t[..., None] * d[None]
  dist = np.hypot(off[..., 0], off[..., 1])
  k = np.argmin(dist, axis=1)
  i = np.arange(len(pts))
  right = (off[i, k, 0] * d[k, 1] - off[i, k, 1] * d[k, 0]) / np.maximum(length[k], 1e-9)
  return starts[k] + t[i, k] * length[k], right


def _cross(d: np.ndarray, off: np.ndarray, k: int, level: float) -> float:
  """m along where a line passes `level` m right of the road between its points k and k + 1."""
  d0, d1, o0, o1 = d[k], d[k + 1], off[k], off[k + 1]
  return float(d0 + (d1 - d0) * (o0 - level) / (o0 - o1)) if o0 != o1 else float(d0)


def swing_taper(lines: list[tuple[int, np.ndarray]], road: np.ndarray, half: float) -> tuple[float, float] | None:
  """Where a turn lane opens in a two-way road's median, from the game files' yellow polylines near the road (`road`:
  its line's points in the direction travelled, a median `half` m either side of it): a polyline swinging from the
  median's right edge across to its left edge, the oncoming lanes', the lane opening behind it, as GTA paints its bays.
  Read off the whole polyline, it is seen however steeply it crosses (sections only read lines along the road). (m
  along the road where it leaves its place, where it arrives) for the last one along the road; None where none does.
  A line crossing the median reads the same travelling either way: it is this way's bay where no line runs on at the
  median's right edge after it, and the other way's where one does (back to back bays, the median between them a
  diamond). Where the two bays meet at the one line (a centre line jogging across, the median handed from one way's
  bay to the other's) it is both ways'."""
  hi, lo = half - TAPER_EDGE * 2 * half, -half + max(TAPER_EDGE * 2 * half, ARRIVED)
  total = float(np.hypot(*np.diff(road, axis=0).T).sum())
  found, seen = [], []
  for _, pts in lines:
    seg = np.diff(pts, axis=0)
    n = np.maximum(np.ceil(np.hypot(seg[:, 0], seg[:, 1]) / SWING_STEP), 1).astype(int)
    dense = np.vstack([pts[:-1][k] + seg[k] * (np.arange(n[k])[:, None] / n[k]) for k in range(len(seg))] + [pts[-1:]])
    d, off = _project(road, dense)
    keep = (d > 0.0) & (d < total) & (np.abs(off) <= half + 1.5)
    if keep.sum() < 3:
      continue
    order = np.argsort(d[keep])
    d, off = d[keep][order], off[keep][order]
    seen.append((d, off))
    arrive = next((k for k in range(len(off)) if off[k] <= lo and (off[:k] >= hi).any()), None)
    if arrive is None:
      continue
    leave = int(np.nonzero(off[:arrive] >= hi)[0][-1])
    span = off[leave:arrive + 1]
    if (np.diff(span) > 0).any() and (np.maximum.accumulate(span[::-1])[::-1] - span).max() > SWING_BACK:
      continue  # not one crossing
    start, end = _cross(d, off, leave, hi), _cross(d, off, arrive - 1, lo)
    if TAPER_LENGTH[0] <= end - start <= TAPER_LENGTH[1]:
      # the median's right edge where the line left it: the paint's, which GTA's median width often isn't
      before = off[:leave + 1]
      found.append((start, end, float(np.median(before[before >= before.max() - SWING_BACK]))))

  def runs(a, b, at):  # a line along the median's edge at `at` between a and b m along
    return any(((d > a) & (d < b) & (np.abs(off - at) < ARRIVED)).any() for d, off in seen)
  found = [(s, e) for s, e, edge in found if not runs(e + 1.0, e + 1.0 + RUNS_ON, edge)]  # nothing left at its right edge after it
  if not found:
    return None
  last = max(e for _, e in found)  # a double line's two polylines: the same swing
  same = [(s, e) for s, e in found if e > last - 3.0]
  return float(np.median([s for s, _ in same])), float(np.median([e for _, e in same]))


MEDIAN_TOL = 1.0  # m between two yellow lines' spacing and the median's width, to take them as its edges
RUNS_IN = 8.0  # m before a link's end looked at for its median's right edge running on into the junction
RUNS_IN_TOL = (1.0, 1.5)  # m nearer and further than GTA's median edge a yellow line may be painted as that edge


def median_runs_in(samples: list[dict], half: float, length: float) -> bool:
  """Whether the game files paint a two-way link's median (`half` m either side of its line, by GTA's offset) as a
  median up to its end (`length` m along), seen travelling it: a yellow line at its right edge in every section over the
  last RUNS_IN m and no arrow of ours painted in it, so no left-turn lane opens in it there. Where GTA paints its median
  as a turn lane, the edge has swung across to the oncoming lanes' side before the junction."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  if any(a.get('dir') in ('ab', 'ahead') and abs(a.get('offset', math.inf)) < half for d in files for a in d.get('arrows') or []):
    return False
  near = [d for d in files if 's' in d and d['s'] >= length - RUNS_IN]

  def edge(d):
    return any(m['colour'] == 'yellow' and m['conf'] >= CONF and
               half - RUNS_IN_TOL[0] <= (sum(m['pair']) / 2 if m.get('pair') else m['offset']) <= half + RUNS_IN_TOL[1]
               for m in d['marks'])
  return bool(near) and all(edge(d) for d in near)


def median_edges(samples: list[dict], median: float) -> tuple[str | None, str | None]:
  """divider=* for a two-way link's median edges, seen travelling a -> b (left, the oncoming lanes', and right): the
  kinds of the two yellow lines the game files show about `median` m apart, in half their samples or more."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  seen = (Counter(), Counter())
  for d in files:
    ys = sorted(((sum(m['pair']) / 2 if m.get('pair') else m['offset']), m['type']) for m in d['marks']
                if m['colour'] == 'yellow' and m['conf'] >= CONF)
    pairs = [(abs(b[0] - a[0] - median), a, b) for i, a in enumerate(ys) for b in ys[i + 1:] if abs(b[0] - a[0] - median) <= MEDIAN_TOL]
    if pairs:
      _, a, b = min(pairs)
      seen[0][a[1]] += 1
      seen[1][b[1]] += 1
  out = []
  for side in seen:
    kind, n = side.most_common(1)[0] if side else (None, 0)
    out.append(DIVIDER.get(kind) if n >= MIN_SAMPLES and n * 2 >= len(files) else None)
  return out[0], out[1]


def sources(samples: list[dict], camera_corrects: bool = True) -> tuple[list[dict], list[dict]]:
  """(the samples to use, the others to cross-check them): the game files' where there are enough, else the camera's
  if `camera_corrects` (none where the game files cover the map: the camera is then only a check)."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  camera = [d for d in samples if d.get('src') != GAMEFILES]
  if len(files) >= MIN_SAMPLES:
    return files, camera
  return (camera, files) if camera_corrects else ([], camera)


def clusters(values: list[tuple[int, float]], tol: float = AGREE) -> list[tuple[float, int]]:
  """Offsets seen in several samples: [(median offset, in how many samples)] from [(sample, offset)], taking the best
  supported first."""
  left = sorted(values, key=lambda v: v[1])
  out = []
  while left:
    group = max(([v for v in left if abs(v[1] - c[1]) <= tol] for c in left), key=lambda g: len({s for s, _ in g}))
    out.append((float(np.median([v for _, v in group])), len({s for s, _ in group})))
    left = [v for v in left if v not in group]
  return sorted(out)


def measure(samples: list[dict]):
  """The yellow and white lines along a link: ([(offset, samples)], [(offset, samples)]), and each white mark
  [(offset, kind)]."""
  yellow, white, kinds = [], [], []
  for k, d in enumerate(samples):
    files = d.get('src') == GAMEFILES
    for m in d['marks']:
      if m['conf'] < CONF:
        continue
      if m['colour'] == 'yellow' and abs(m['offset']) <= CENTRE_REACH + 3.5:
        yellow.append((k, sum(m['pair']) / 2 if m.get('pair') else m['offset']))
      elif m['colour'] == 'white' and (m['type'] in CROSSING if files else m['type'] in LANE_LINES):
        offset = sum(m['pair']) / 2 if m.get('pair') else m['offset']
        white.append((k, offset))
        kinds.append((offset, m['type'] if files else None))
  return clusters(yellow), clusters(white), kinds


def changes(lines: list[float], kinds: list[tuple[float, str | None]], flip: bool) -> list[str] | None:
  """change:lanes for a direction's lanes from the kinds of the lines between them (`lines`, left to right as seen
  travelling it; `flip` where that's b -> a, which swaps a line's halves): None where a kind isn't known."""
  crossing = []
  for v in lines:
    seen = Counter(kind for o, kind in kinds if abs((-o if flip else o) - v) <= AGREE)
    kind = seen.most_common(1)[0][0] if seen else None
    if kind not in CROSSING:
      return None
    crossing.append(CROSSING[kind][::-1] if flip else CROSSING[kind])
  left = [True] + [c[1] for c in crossing]  # each lane may cross the line on its left (none: the centre or kerb)
  right = [c[0] for c in crossing] + [True]
  return [CHANGE[(a, b)] for a, b in zip(left, right, strict=True)]


def road_kerbs(samples: list[dict], kerbs: tuple[float, float], tol: float = KERB_TOL, one: bool = False) -> tuple[float, float]:
  """The kerbs to run the outer lanes out to: the game files' asphalt edges, where both are seen in half the samples,
  near where the class layout has them and about as far either side of the line (`tol`); else the layout's. With
  `one`, an edge seen on one side only stands for both (the line the middle of the road)."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  lefts = [d['kerbs']['left'] for d in files if (d.get('kerbs') or {}).get('left') is not None]
  rights = [d['kerbs']['right'] for d in files if (d.get('kerbs') or {}).get('right') is not None]
  if one and max(len(lefts), len(rights)) * 2 >= max(len(samples), 1) and min(len(lefts), len(rights)) * 2 < max(len(samples), 1):
    lefts, rights = (lefts, [-v for v in lefts]) if len(lefts) > len(rights) else ([-v for v in rights], rights)
  if min(len(lefts), len(rights)) * 2 < max(len(samples), 1):
    return kerbs
  left, right = float(np.median(lefts)), float(np.median(rights))
  if abs(left - kerbs[0]) > KERB_REACH or abs(right - kerbs[1]) > KERB_REACH or abs(right + left) > tol:
    return kerbs
  half = (right - left) / 2  # the line stays the middle of the road
  return -half, half


PARKING_MIN, PARKING_MAX = 1.8, 5.5  # m of paved strip between the asphalt's edge and the kerb's face: a parking lane
STRIP_AGREE = 0.6  # m between a strip's widths along a link


def strips(samples: list[dict]) -> tuple[float, float]:
  """The paved strips (paver or parking) left and right between the asphalt's edge and the kerb's face, from the game
  files where half their samples or more agree on one: their widths, 0 for none."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  out = []
  for side, sign in (('left', -1), ('right', 1)):
    gaps = [(d['kerb_step'][side] - d['kerbs'][side]) * sign for d in files
            if (d.get('kerbs') or {}).get(side) is not None and (d.get('kerb_step') or {}).get(side) is not None]
    gaps = [g for g in gaps if PARKING_MIN <= g <= PARKING_MAX]
    ok = files and len(gaps) * 2 >= len(files) and len(gaps) >= MIN_SAMPLES and max(gaps) - min(gaps) <= STRIP_AGREE
    out.append(round(float(np.median(gaps)), 2) if ok else 0.0)
  return out[0], out[1]


def correct(samples: list[dict], fwd: int, back: int, kerbs: tuple[float, float], counts_from_paint: bool = False):
  """The painted cross-section of a two-way link with fwd and back lanes and its kerbs where the class layout has them
  (m left and right of its line), from its samples seen along it (one source's: sources()): ({'forward': [widths],
  'backward': [widths] (each direction's lanes left to right as seen travelling it), 'median': m, and from the game
  files' line kinds 'change:forward' / 'change:backward': [change:lanes values] where a line can't be crossed, the
  'divider' its centre line's kind, 'parking' (left, right) strips beyond the asphalt's edges}, None), or (None, why
  not). With `counts_from_paint` (the game files only), the painted lanes may be more or fewer than GTA's: the count
  then comes from the lane lines, each lane still LANE_MIN to LANE_MAX wide."""
  if not (fwd and back):
    return None, 'one-way'
  yellow, white, kinds = measure(samples)
  need = max(MIN_SAMPLES, (len(samples) + 1) // 2)
  yellow = [v for v, c in yellow if c >= need]
  if len(yellow) == 1:
    lo = hi = yellow[0]
  elif len(yellow) == 2 and yellow[1] - yellow[0] > MEDIAN_MIN:
    lo, hi = yellow
  else:
    return None, 'no centre' if not yellow else 'centre unclear'
  centre = (lo + hi) / 2
  layout = kerbs
  if fwd == back or counts_from_paint:
    kerbs = road_kerbs(samples, kerbs)
  # kerbs alike either side: the line is the middle of the road, and the lanes' widths say where the centre is
  if abs(centre) > (CENTRE_REACH if abs(kerbs[0] + kerbs[1]) < 1e-6 else CENTRE_TOL):
    return None, 'centre off the line'
  lines = [v for v, c in white if c >= need]

  def between(kerbs):
    return [v for v in lines if hi + LANE_MIN * 0.8 < v < kerbs[1] - LANE_MIN * 0.8], \
      sorted(-v for v in lines if kerbs[0] + LANE_MIN * 0.8 < v < lo - LANE_MIN * 0.8)
  right, left = between(kerbs)
  if (len(right) != fwd - 1 or len(left) != back - 1) and counts_from_paint and kerbs == layout:
    # other counts than GTA's want the asphalt's edges to lay them out to, a little off the middle of the line or one side seen
    kerbs = road_kerbs(samples, layout, COUNT_KERB_TOL, one=True)
    right, left = between(kerbs)
  if (len(right) != fwd - 1 or len(left) != back - 1) and not (counts_from_paint and kerbs != layout):
    return None, f'{len(left) + 1}+{len(right) + 1} lanes painted, GTA has {back}+{fwd}'
  f = [round(float(x), 2) for x in np.diff([hi, *right, kerbs[1]])]
  b = [round(float(x), 2) for x in np.diff([-lo, *left, -kerbs[0]])]
  if not all(LANE_MIN <= x <= LANE_MAX for x in f + b):
    return None, 'lane widths'
  out = {'forward': f, 'backward': b, 'median': round(hi - lo, 2)}
  for key, between, flip in (('change:forward', right, False), ('change:backward', left, True)):
    if (got := changes(between, kinds, flip)) and set(got) != {'yes'}:
      out[key] = got
  if lo == hi and (divider := centre_kind(samples, lo)):
    out['divider'] = divider
  if kerbs != layout and any(parking := strips(samples)):  # beyond the asphalt's edges the lanes run out to
    out['parking'] = parking
  out['middle'] = abs(kerbs[0] + kerbs[1]) < 1e-6  # the line is the middle of the road between the kerbs
  return out, None


def centre_kind(samples: list[dict], at: float = 0.0, tol: float = AGREE) -> str | None:
  """divider=* for the one centre line the game files show near `at` m right of the line (centre_line); None where they
  show none, or a median."""
  found = centre_line(samples, at, tol)
  return found[0] if found else None


WHITE_CENTRE_REACH = 2.0  # m: white lines further from a two-way link's line are lane or edge lines, not its centre
# m: a thinner white centre is the port's thin double yellow line read off the road's texture (Buccaneer Way)
WHITE_CENTRE_WIDTH = 0.1
DASHES_SEEN = 0.2  # of the sections at least that cross one of a dashed centre's dashes, the rest its gaps


def centre_line(samples: list[dict], at: float = 0.0, tol: float = AGREE) -> tuple[str, str] | None:
  """(divider=*, colour) of the one centre line the game files show near `at` m right of the line in half their samples
  or more: yellow, or white where a section shows no yellow (Greenwich Pkwy's and the port's dashed ones). A dashed
  line laid a decal a dash shows only in the sections that cross a dash: in DASHES_SEEN of them, the others showing no
  line at all, it's dashed. None where they show none, or a median."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  seen, gaps = Counter(), 0
  def near(lines):
    return [m['type'] for m in lines if abs((sum(m['pair']) / 2 if m.get('pair') else m['offset']) - at) <= tol]
  for d in files:
    marks = [m for m in d['marks'] if m['conf'] >= CONF]
    yellow = [m for m in marks if m['colour'] == 'yellow' and abs(m['offset']) <= CENTRE_REACH + 3.5]
    white = [m for m in marks if m['colour'] == 'white' and m['type'] != 'edge_line' and abs(m['offset']) <= WHITE_CENTRE_REACH
             and m.get('width', WHITE_CENTRE_WIDTH) >= WHITE_CENTRE_WIDTH]
    # the yellow line, else a white one where no yellow is near (yellow lines further out being edge lines)
    lines, colour = (yellow, 'yellow') if near(yellow) or not near(white) else (white, 'white')
    if len(near(lines)) == 1 and len(lines) == 1:
      seen[(near(lines)[0], colour)] += 1
    gaps += not yellow and not white
  (kind, colour), n = seen.most_common(1)[0] if seen else ((None, None), 0)
  if kind in DIVIDER and (n >= MIN_SAMPLES and n * 2 >= len(files) or
                          kind == 'dashed' and len(seen) == 1 and n >= DASHES_SEEN * len(files) and (n + gaps) * 2 >= len(files)):
    return DIVIDER[kind], colour
  return None


def centre_bare(samples: list[dict]) -> bool:
  """Whether the game files show a two-way link with no centre line: MIN_SAMPLES or more sections, the asphalt's edges
  read in half of them (the road is in the files, not a gap in them), and no line near the link in any (no yellow
  within its centre's reach, a metre more for a median's edges, no white but edge lines within WHITE_CENTRE_REACH). A
  dashed line laid a decal a dash can fall between sections: check the line along the link itself too (ynd_to_osm)."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  if len(files) < MIN_SAMPLES:
    return False
  edges = sum((d.get('kerbs') or {}).get('left') is not None and (d.get('kerbs') or {}).get('right') is not None for d in files)
  return edges * 2 >= len(files) and not any(
    m['conf'] >= CONF and (m['colour'] == 'yellow' and abs(m['offset']) <= CENTRE_REACH + 1.0 or
                           m['colour'] == 'white' and m['type'] != 'edge_line' and abs(m['offset']) <= WHITE_CENTRE_REACH)
    for d in files for m in d['marks'])


EDGE_REACH = 2.5  # m between a one-way road's painted edge and where the class layout has its kerb
EDGE_WEIGHT = 0.5  # how much an asphalt edge's place counts against a painted line's, in placing a link's line
PLACE_PREFER = 0.0025  # m² of moved lines a placement tag must save over the middle of the lanes


def place(edges: list[float], weights: list[float], ok=None) -> tuple[str | None, list[float]] | None:
  """Where a one-way link's line goes among its lanes' edges (m right of the line, left to right; `weights` how much
  each one's place counts). A way's line stays on GTA's nodes and placement=* only puts it on a lane's edge or middle
  (none: the middle of the lanes), so only the edges that say where the line is move: one onto it (left_of / right_of,
  the lanes either side taking up the difference), a lane's two edges centred on it (middle_of), or the outer two
  (none); the others stay where painted. Of the options moving no edge more than PLACE_TOL whose widths are `ok`, the
  one moving them least (weighted squares; a tag must save PLACE_PREFER): (placement or None, the edges as drawn), else
  None."""
  n = len(edges) - 1
  c = (edges[0] + edges[-1]) / 2
  options = [((weights[0] + weights[-1]) * c * c, None, [edges[0] - c, *edges[1:-1], edges[-1] - c])]
  for j, e in enumerate(edges):
    options.append((weights[j] * e * e + PLACE_PREFER, 'left_of:1' if j == 0 else f'right_of:{j}', [*edges[:j], 0.0, *edges[j + 1:]]))
  for k in range(1, n + 1) if n > 1 else ():
    m = (edges[k - 1] + edges[k]) / 2
    options.append(((weights[k - 1] + weights[k]) * m * m + PLACE_PREFER, f'middle_of:{k}',
                    [*edges[:k - 1], edges[k - 1] - m, edges[k] - m, *edges[k + 1:]]))
  for _, where, drawn in sorted(options, key=lambda o: o[0]):
    widths = [b - a for a, b in zip(drawn, drawn[1:], strict=False)]
    if max(abs(a - b) for a, b in zip(drawn, edges, strict=True)) <= PLACE_TOL and (ok is None or ok(widths)):
      return where, drawn
  return None


def dashes(white: list[tuple[int, float]], need: int, taken: list[float], lo: float, hi: float) -> list[float]:
  """Lane lines between lo and hi that the sections cross only now and then: white lines seen in fewer than `need`
  samples (one at least), clear of the lines `taken`. A line crossing under half the sections across a link isn't
  solid there, whatever its pieces' kind: dashes with long gaps (GTA's freeway dashes are 4 m in 12)."""
  return [v for v, c in clusters(white) if c < need and lo + LANE_MIN * 0.8 < v < hi - LANE_MIN * 0.8 and
          all(abs(v - t) >= LINE_MERGE for t in taken)]


def correct_oneway(samples: list[dict], n: int, kerbs: tuple[float, float]):
  """The painted lanes of a one-way link with n lanes and its kerbs where the class layout has them, from the game
  files' samples alone (the camera's lines are too loose for roads without a yellow centre to anchor them): its edges
  are the yellow or white lines nearest those kerbs (within EDGE_REACH), else the asphalt's edges, with n - 1 white lane
  lines between (dashes the sections cross only now and then too: dashes(); or none, where the files miss them: n even
  lanes), each lane LANE_MIN to LANE_MAX wide, the link's line placed among them (place()): ({'lanes': [widths left to
  right], 'placement': placement=* where the line isn't the middle of the lanes, 'change': [...] where a line can't be
  crossed}, None), or (None, why not)."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  if len(files) < MIN_SAMPLES:
    return None, 'one-way, no game files'
  need = max(MIN_SAMPLES, (len(files) + 1) // 2)
  lines, kinds, white = [], [], []
  for k, d in enumerate(files):
    for m in d['marks']:
      if m['conf'] >= CONF and (m['colour'] == 'yellow' or m['type'] in CROSSING or m['type'] == 'edge_line'):
        offset = sum(m['pair']) / 2 if m.get('pair') else m['offset']
        if kerbs[0] - EDGE_REACH <= offset <= kerbs[1] + EDGE_REACH:
          lines.append((k, offset))
          kinds.append((offset, m['type'] if m['colour'] == 'white' else None))
          if m['colour'] == 'white' and m['type'] in CROSSING:
            white.append((k, offset))
  seen = [v for v, c in clusters(lines) if c >= need]
  painted = {}

  def edge(side, i):  # the painted edge nearest the layout's kerb, else the asphalt's
    near = [v for v in seen if abs(v - kerbs[i]) <= EDGE_REACH]
    if near:
      painted[i] = True
      return min(near, key=lambda v: abs(v - kerbs[i]))
    asphalt = [d['kerbs'][side] for d in files if (d.get('kerbs') or {}).get(side) is not None]
    return float(np.median(asphalt)) if len(asphalt) * 2 >= len(files) and abs(np.median(asphalt) - kerbs[i]) <= EDGE_REACH else None
  lo, hi = edge('left', 0), edge('right', 1)
  if lo is None or hi is None:
    return None, 'one-way, no edges'
  between = [v for v in seen if lo + LANE_MIN * 0.8 < v < hi - LANE_MIN * 0.8]
  if len(between) < n - 1 and len(between) + len(extra := dashes(white, need, [lo, *between, hi], lo, hi)) == n - 1:
    between = sorted(between + extra)
    kinds = [(o, t) for o, t in kinds if all(abs(o - v) > AGREE for v in extra)] + [(v, 'dashed') for v in extra]
  if not between and n > 1 and LANE_MIN <= (hi - lo) / n <= LANE_MAX:  # the files miss the lane lines: GTA's lanes, evenly
    if abs((lo + hi) / 2) > CENTRE_TOL:
      return None, 'one-way, off the line'
    return {'lanes': [round(float(hi - lo) / n, 2)] * n}, None
  if len(between) != n - 1:
    return None, f'one-way, {len(between) + 1} lanes painted, GTA has {n}'
  edges = [lo, *between, hi]
  found = place(edges, [1.0 if painted.get(0) else EDGE_WEIGHT, *[1.0] * len(between), 1.0 if painted.get(1) else EDGE_WEIGHT],
                lambda ws: all(LANE_MIN <= round(w, 2) <= LANE_MAX for w in ws))
  if found is None:
    return None, 'one-way, off the line' if place(edges, [1.0] * len(edges)) is None else 'lane widths'
  where, drawn = found
  out = {'lanes': [round(float(x), 2) for x in np.diff(drawn)]}
  if where:
    out['placement'] = where
  if (got := changes(between, kinds, False)) and set(got) != {'yes'}:
    out['change'] = got
  return out, None


LINE_TOL = 0.3  # m short of halfway across the narrower lane beside a boundary: the painted white line nearer it is it
INSIDE = 1.0  # m inside a direction's lanes' outer edges: white lines nearer them are its centre or edge lines
UNPAINTED = 0.9  # of the samples showing no centre or lane line at all: the road is unpainted
UNPAINTED_SAMPLES = 3


def lane_lines(samples: list[dict], section: list[tuple[float, float, int]]) -> dict[str, list[str]]:
  """change:lanes for each direction's lanes from the game files' white lines between them, where the map's lanes are
  its class layout's (not corrected by correct / correct_oneway): {'forward' | 'backward': [values, each lane left to
  right as seen travelling it]}, only where a line may not be crossed. `section`: the way's lanes [(left, right, 1
  forward / -1 backward)], m right of the link's line seen travelling a -> b. A boundary takes the kind of the white
  line nearest it (nearer it than any other boundary, less than halfway across the lanes beside it: class layouts are
  often a metre or two off the paint) in half the samples or more, else it's taken as dashed. No line there says
  nothing: the files miss thin dashed lane lines the game paints (Vinewood Blvd's, seen from above in the game)."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  if len(files) < MIN_SAMPLES:
    return {}
  out = {}
  for key, heading in (('forward', 1), ('backward', -1)):
    spans = [(a, b) for a, b, h in section if h == heading]
    if len(spans) < 2:
      continue
    bounds = [b for _, b in spans[:-1]]
    votes = [Counter() for _ in bounds]
    for d in files:
      lines = [(sum(m['pair']) / 2 if m.get('pair') else m['offset'], m['type']) for m in d['marks']
               if m['conf'] >= CONF and m['colour'] == 'white' and m['type'] in CROSSING]
      for i, (v, seen) in enumerate(zip(bounds, votes, strict=True)):
        near = [q for q in lines if spans[i][0] + INSIDE < q[0] < spans[i + 1][1] - INSIDE]  # across the lanes beside it
        if not near:
          continue
        o, kind = min(near, key=lambda q: abs(q[0] - v))
        reach = min(spans[i][1] - spans[i][0], spans[i + 1][1] - spans[i + 1][0]) / 2 - LINE_TOL
        mine = min(bounds, key=lambda b: abs(o - b)) == v  # not another boundary's line
        if abs(o - v) <= reach and mine:
          seen[kind] += 1
    kinds = [seen.most_common(1)[0][0] if seen and seen.most_common(1)[0][1] >= len(files) / 2 else 'dashed' for seen in votes]
    crossing = [CROSSING[k] for k in kinds]
    if heading == -1:  # left to right as seen travelling b -> a: the other way round, each line's halves swapped
      crossing = [c[::-1] for c in reversed(crossing)]
    left = [True] + [c[1] for c in crossing]
    right = [c[0] for c in crossing] + [True]
    change = [CHANGE[(a, b)] for a, b in zip(left, right, strict=True)]
    if set(change) != {'yes'}:
      out[key] = change
  return out


def unpainted(samples: list[dict], least: int = UNPAINTED_SAMPLES, edges_seen: bool = True) -> bool:
  """Whether the game files show a road with no centre or lane lines at all: UNPAINTED of `least` or more samples,
  with the asphalt's edges read in half of them (the road surface is in the files, not a gap in them; not asked of a
  road the files draw no edges for, `edges_seen` False). Lines within INSIDE of an edge read are its edge lines (or a
  gutter's, a driveway's), not lines between lanes."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  if len(files) < least:
    return False
  edges = sum(any((d.get('kerbs') or {}).get(s) is not None for s in ('left', 'right')) for d in files)

  def between(d, m):
    kerbs = d.get('kerbs') or {}
    left, right = kerbs.get('left'), kerbs.get('right')
    return (left is None or m['offset'] > left + INSIDE) and (right is None or m['offset'] < right - INSIDE)
  bare = sum(not any(m['conf'] >= CONF and (m['colour'] == 'yellow' or m['type'] in CROSSING) and between(d, m) for m in d['marks'])
             for d in files)
  return (edges * 2 >= len(files) or not edges_seen) and bare >= UNPAINTED * len(files)


OUTER_REACH = 1.2  # m from a one-way link's outer lane edge that a white lane line is that edge's


def outer_lines(samples: list[dict], edges: tuple[float, float]) -> tuple[bool | None, bool | None]:
  """Whether a one-way link's outer lanes may cross the white lane lines at their outer edges (`edges`, m right of its
  line), where another link runs on beside it, as GTA's freeway links side by side: each edge takes the kind of the
  white lane line nearest it within OUTER_REACH in half the samples or more. (left, right); None where no lane line is
  seen there (a kerb or edge line, or a gap in the files)."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  if len(files) < MIN_SAMPLES:
    return None, None
  out = []
  for i, edge in enumerate(edges):
    seen = Counter()
    for d in files:
      near = [(abs(o - edge), kind) for m in d['marks'] if m['conf'] >= CONF and m['colour'] == 'white' and
              (kind := m['type']) in CROSSING and abs((o := sum(m['pair']) / 2 if m.get('pair') else m['offset']) - edge) <= OUTER_REACH]
      if near:
        seen[min(near)[1]] += 1
    kind, n = seen.most_common(1)[0] if seen else (None, 0)
    out.append(CROSSING[kind][1 - i] if kind and n * 2 >= len(files) else None)  # the half facing the link's lanes
  return out[0], out[1]


EDGE_LINE_REACH = 1.5  # m between a one-way link's lane edge and a yellow line painted along it


def edge_lines(samples: list[dict], edges: tuple[float, float]) -> tuple[str | None, str | None]:
  """divider=* for the yellow lines the game files paint along a one-way link's lane edges (`edges`, m right of its
  line), as along a carriageway's left edge beside a median or barrier: each edge takes the kind of the yellow line
  nearest it within EDGE_LINE_REACH in half the samples or more (a solid one for raised markers or an edge line).
  (left, right); None where none is."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  if len(files) < MIN_SAMPLES:
    return None, None
  out = []
  for edge in edges:
    seen = Counter()
    for d in files:
      near = [(abs(o - edge), m['type']) for m in d['marks'] if m['conf'] >= CONF and m['colour'] == 'yellow' and
              abs((o := sum(m['pair']) / 2 if m.get('pair') else m['offset']) - edge) <= EDGE_LINE_REACH]
      if near:
        seen[min(near)[1]] += 1
    kind, n = seen.most_common(1)[0] if seen else (None, 0)
    out.append(DIVIDER.get(kind, 'solid_line') if kind and n * 2 >= len(files) else None)
  return out[0], out[1]


def disagree(a: dict, b: dict, tol: float = AGREE) -> bool:
  """Whether two sources' cross-sections of a link differ by more than tol anywhere."""
  def edges(sec):
    f, b = np.cumsum(sec['forward']), np.cumsum(sec['backward'])
    return np.concatenate([[sec['median']], f, b])
  return len(a['forward']) != len(b['forward']) or len(a['backward']) != len(b['backward']) or \
    bool(np.abs(edges(a) - edges(b)).max() > tol)


TURNS = {'through;left': 'left;through', 'through;right': 'through;right', 'left;through': 'left;through',
         'left': 'left', 'right': 'right', 'through': 'through', 'reverse': 'reverse', 'left;right': 'left;right'}


def arrow_marks(path) -> list[tuple[float, float, float, float, str]]:
  """The game files' painted turn arrows (features.jsonl's "arrow" features): [(x, y, z, the heading they point,
  turn:lanes value)], headings in degrees counterclockwise from north, as paths.heading. A decal's own heading is a
  quarter turn clockwise of where its arrow points."""
  out = []
  for line in open(path):
    f = json.loads(line)
    if f.get('kind') == 'arrow' and f.get('arrow') in TURNS:
      out.append((f['x'], f['y'], f['z'], (f['heading'] - 90.0 + 180.0) % 360.0 - 180.0, TURNS[f['arrow']]))
  return out


CARRIAGEWAY_REACH = 25.0  # m either side of a one-way link that the lines of its carriageway are read
MARKERS_BESIDE = 2.0  # m: raised markers this near another painted line run alongside it, not between lanes of their own
PLACE_TOL = 1.0  # m the painted lines may move to put the link's line where placement=* can say it
CARRIAGEWAY_LANE_MAX = 7.6  # m: GTA's freeway lanes are 5.5-7.5 m wide
LINE_MERGE = 1.6  # m: painted lines nearer each other than this are one boundary


def correct_carriageway(samples: list[dict], n: int):
  """The painted lanes of a one-way link that is one of several side by side making up a carriageway, as GTA draws a
  freeway (its lane changes are links between them), from the game files' samples alone. correct_oneway can't take
  these: the carriageway's edges are far from the link's own kerbs, and its lanes needn't be centred on it. Every lane
  across the carriageway is read (painted lines LANE_MIN to CARRIAGEWAY_LANE_MAX apart, lines nearer than LINE_MERGE
  taken as one, raised markers beside another line left out, the asphalt's edges beyond the outermost lines), and the
  link's are the n side by side, white lines between, whose middle is nearest its line, within half a lane. Where two
  lines are further apart than a lane, a white line between seen in fewer sections splits them (dashes()).
  The link's line is placed among them by place(). Returns ({'lanes': [widths left to right], 'placement': placement=*
  where the line isn't the middle of the lanes, 'change': [change:lanes] where a line can't be crossed}, None), or
  (None, why not). 'change' also says it of the lanes' outer edges where those are lines between lanes of the
  carriageway, where another way runs beside this one."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  if len(files) < MIN_SAMPLES:
    return None, 'carriageway, no game files'
  need = max(MIN_SAMPLES, (len(files) + 1) // 2)
  lines, kinds = [], []
  for k, d in enumerate(files):
    for m in d['marks']:
      if m['conf'] >= CONF and (m['colour'] == 'yellow' or m['type'] in CROSSING or m['type'] == 'edge_line'):
        offset = sum(m['pair']) / 2 if m.get('pair') else m['offset']
        if abs(offset) <= CARRIAGEWAY_REACH:
          lines.append((k, offset))
          kinds.append((offset, m['type'] if m['colour'] == 'white' else 'yellow'))

  def kind_near(v):  # the line's kind most seen at v
    seen = Counter(t for o, t in kinds if abs(o - v) <= AGREE)
    return seen.most_common(1)[0][0] if seen else None
  strong = [(v, c) for v, c in clusters(lines) if c >= need]
  strong = [(v, c) for v, c in strong if kind_near(v) != 'markers' or
            not any(AGREE < abs(v - q) <= MARKERS_BESIDE and kind_near(q) != 'markers' for q, _ in strong)]
  if not strong:
    return None, 'carriageway, no lines'
  groups = []  # lines nearer each other than LINE_MERGE are one: a double line read as two, markers beside markers
  for v, c in sorted(strong):
    if groups and v - groups[-1][-1][0] < LINE_MERGE:
      groups[-1].append((v, c))
    else:
      groups.append([(v, c)])
  known = {}
  for g in groups:
    known[float(np.mean([v for v, _ in g]))] = kind_near(max(g, key=lambda vc: (kind_near(vc[0]) != 'markers', vc[1]))[0])
  seen = list(known)

  def kind(v):
    return known[v] if v in known else kind_near(v)
  bounds = list(seen)
  for side, beyond in (('left', lambda v: v < min(seen)), ('right', lambda v: v > max(seen))):
    asphalt = [d['kerbs'][side] for d in files if (d.get('kerbs') or {}).get(side) is not None]
    if len(asphalt) * 2 >= len(files) and beyond(edge := float(np.median(asphalt))) and \
        all(abs(edge - v) > LANE_MIN * 0.8 for v in seen):
      bounds.append(edge)
  bounds.sort()
  white = [line for line, (_, t) in zip(lines, kinds, strict=True) if t in CROSSING]
  for v in dashes(white, need, bounds, bounds[0], bounds[-1]):
    a, b = max(q for q in bounds if q < v), min(q for q in bounds if q > v)
    if b - a > CARRIAGEWAY_LANE_MAX and v - a >= LANE_MIN and b - v >= LANE_MIN:
      bounds = sorted([*bounds, v])
      known[v] = 'dashed'
  lanes = [(a, b) for a, b in zip(bounds, bounds[1:], strict=False) if LANE_MIN <= b - a <= CARRIAGEWAY_LANE_MAX]
  best = None  # (the run's middle, its first lane)
  for i in range(len(lanes) - n + 1):
    run = lanes[i:i + n]
    if any(run[j][1] != run[j + 1][0] or kind(run[j][1]) not in CROSSING for j in range(n - 1)):
      continue
    middle = (run[0][0] + run[-1][1]) / 2
    if best is None or abs(middle) < abs(best[0]):
      best = (middle, i)
  if best is None:
    return None, f'carriageway, no {n} lanes side by side'
  middle, i = best
  edges = [lanes[i][0]] + [b for _, b in lanes[i:i + n]]
  if abs(middle) > (edges[-1] - edges[0]) / n / 2:
    return None, 'carriageway, off the line'
  weights = [1.0 if e in known else EDGE_WEIGHT for e in edges]  # the asphalt's edges aren't painted lines
  if (found := place(edges, weights)) is None:
    return None, 'carriageway, line not on a lane edge or middle'
  if (found := place(edges, weights, lambda ws: all(LANE_MIN - PLACE_TOL <= round(w, 2) <= LANE_MAX + PLACE_TOL for w in ws))) is None:
    return None, 'lane widths'
  where, drawn = found
  widths = [round(float(b - a), 2) for a, b in zip(drawn, drawn[1:], strict=False)]

  def crossing(v, lane_beyond):  # CROSSING of the line at v; an outer edge's only where another lane runs on beyond it
    return CROSSING[kind(v)] if lane_beyond and kind(v) in CROSSING else (True, True)
  inner = [crossing(v, True) for v in edges[1:-1]]
  outer = (crossing(edges[0], i > 0 and lanes[i - 1][1] == edges[0]),
           crossing(edges[-1], i + n < len(lanes) and lanes[i + n][0] == edges[-1]))
  left = [outer[0][1]] + [c[1] for c in inner]  # each lane may cross the line on its left, on its right
  right = [c[0] for c in inner] + [outer[1][0]]
  out = {'lanes': widths, **({'placement': where} if where else {})}
  if set(change := [CHANGE[(a, b)] for a, b in zip(left, right, strict=True)]) != {'yes'}:
    out['change'] = change
  return out, None
