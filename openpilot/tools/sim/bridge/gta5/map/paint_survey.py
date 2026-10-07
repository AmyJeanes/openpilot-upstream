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

One-way links are corrected from the game files alone (correct_oneway: their painted edges and lane lines). Two-way
links are corrected from either source, only where the samples agree with each other and with GTA's lane counts:
- a yellow centre (one line, or a median's two edges more than MEDIAN_MIN apart) in at least MIN_SAMPLES samples and
  half of them, within CENTRE_REACH of the link's line;
- in each direction exactly its lanes less one white lane lines (dashed, solid or raised markers), seen as often,
  agreeing within AGREE;
- every lane LANE_MIN to LANE_MAX wide, its outer lanes running out to the kerbs: the game files' asphalt edges where
  they are seen on both sides about as far from the line (KERB_TOL), else where the class layout has them.
A way's line can't leave GTA's nodes, so the line stays put and the paint moves round it: with as many lanes each way,
the line is the middle of the road between the kerbs (OSM's default reading), so a centre off the line is said by the
lanes' widths alone; with more lanes one way the line is the centre's (placement), which must then be on the line
(within CENTRE_TOL). Lane counts never change here; disagreements are only counted.
Painted arrows (game files only, a sample's "arrows" or arrow "features") give each direction's lanes their turn arrows,
where there is one per lane, and the game files' kinds of the lines between lanes (solid, solid on one half) their
change:lanes. Other fields (z, a mark's width / cover / line id, kerb_step, hatched spans, stop lines, other
features) are read past.
"""
import json
from collections import Counter, defaultdict

import numpy as np

CONF = 0.7  # a mark's detector confidence
MIN_SAMPLES = 2
AGREE = 0.4  # m between one line's offsets in different samples
MEDIAN_MIN = 1.0  # m between two yellow lines bounding a painted median, rather than one double line
CENTRE_REACH = 3.0  # m from the link's line to its centre
CENTRE_TOL = 0.4  # m, where the line must be the centre's
KERB_TOL = 0.4  # m between the two kerbs' distances from the line, to take them as the road's
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
ARROW_APART = 1.5  # m between two lanes' arrows
GAMEFILES = 'gamefiles'


def node_key(s: str) -> tuple[int, int]:
  a, i = s.split(':')
  return int(a), int(i)


SOLID_PAINTED = 0.7  # of a line's length painted: solid with worn or hidden stretches, not dashed
DASH_PERIOD = 4.0  # m: a "dashed" line repeating faster than this is a solid one laid as tiled decals
MARKER_KERB = 1.0  # m: raised markers this near the asphalt's edge or the kerb are the gutter's edge, not paint


def line_kinds(path) -> dict[int, str]:
  """Whether each of the game files' polylines (polylines.jsonl: id, style, painted, dashes) is solid or dashed, by its
  whole length: a section's reading can call a solid line dashed where it's worn, interrupted or tiled."""
  out = {}
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
    kinds = [lines.get(i) for i in ids]
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


def road_kerbs(samples: list[dict], kerbs: tuple[float, float]) -> tuple[float, float]:
  """The kerbs to run the outer lanes out to: the game files' asphalt edges, where both are seen in half the samples,
  near where the class layout has them and about as far either side of the line; else the layout's."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  lefts = [d['kerbs']['left'] for d in files if (d.get('kerbs') or {}).get('left') is not None]
  rights = [d['kerbs']['right'] for d in files if (d.get('kerbs') or {}).get('right') is not None]
  if min(len(lefts), len(rights)) * 2 < max(len(samples), 1):
    return kerbs
  left, right = float(np.median(lefts)), float(np.median(rights))
  if abs(left - kerbs[0]) > KERB_REACH or abs(right - kerbs[1]) > KERB_REACH or abs(right + left) > KERB_TOL:
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
  if abs(centre) > (CENTRE_REACH if fwd == back else CENTRE_TOL):
    return None, 'centre off the line'
  layout = kerbs
  if fwd == back or counts_from_paint:
    kerbs = road_kerbs(samples, kerbs)
  lines = [v for v, c in white if c >= need]
  right = [v for v in lines if hi + LANE_MIN * 0.8 < v < kerbs[1] - LANE_MIN * 0.8]
  left = sorted(-v for v in lines if kerbs[0] + LANE_MIN * 0.8 < v < lo - LANE_MIN * 0.8)
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
  """divider=* for the one yellow centre line the game files show near `at` m right of the line in half their samples or
  more; None where they show none, or a median."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  seen = Counter()
  # a white centre where no yellow shows at all (Greenwich Pkwy's dashed one)
  colour = 'yellow' if any(m['colour'] == 'yellow' for d in files for m in d['marks']) else 'white'
  reach = CENTRE_REACH + 3.5 if colour == 'yellow' else 2.0  # white lines further out are lane or edge lines
  for d in files:
    yellow = [m for m in d['marks'] if m['colour'] == colour and m['conf'] >= CONF and abs(m['offset']) <= reach]
    near = [m['type'] for m in yellow if abs((sum(m['pair']) / 2 if m.get('pair') else m['offset']) - at) <= tol]
    if len(near) == 1 and len(yellow) == 1:
      seen[near[0]] += 1
  kind, n = seen.most_common(1)[0] if seen else (None, 0)
  return DIVIDER.get(kind) if n >= MIN_SAMPLES and n * 2 >= len(files) else None


EDGE_REACH = 2.5  # m between a one-way road's painted edge and where the class layout has its kerb


def correct_oneway(samples: list[dict], n: int, kerbs: tuple[float, float]):
  """The painted lanes of a one-way link with n lanes and its kerbs where the class layout has them, from the game
  files' samples alone (the camera's lines are too loose for roads without a yellow centre to anchor them): its edges
  are the yellow or white lines nearest those kerbs (within EDGE_REACH), else the asphalt's edges, with n - 1 white lane
  lines between, each lane LANE_MIN to LANE_MAX wide, centred on the link (within CENTRE_TOL; one-way links are centred
  on their lanes, so the line can't say more): ({'lanes': [widths left to right], 'change': [...] where a line can't
  be crossed}, None), or (None, why not)."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  if len(files) < MIN_SAMPLES:
    return None, 'one-way, no game files'
  need = max(MIN_SAMPLES, (len(files) + 1) // 2)
  lines, kinds = [], []
  for k, d in enumerate(files):
    for m in d['marks']:
      if m['conf'] >= CONF and (m['colour'] == 'yellow' or m['type'] in CROSSING or m['type'] == 'edge_line'):
        offset = sum(m['pair']) / 2 if m.get('pair') else m['offset']
        if kerbs[0] - EDGE_REACH <= offset <= kerbs[1] + EDGE_REACH:
          lines.append((k, offset))
          kinds.append((offset, m['type'] if m['colour'] == 'white' else None))
  seen = [v for v, c in clusters(lines) if c >= need]

  def edge(side, i):  # the painted edge nearest the layout's kerb, else the asphalt's
    near = [v for v in seen if abs(v - kerbs[i]) <= EDGE_REACH]
    if near:
      return min(near, key=lambda v: abs(v - kerbs[i]))
    asphalt = [d['kerbs'][side] for d in files if (d.get('kerbs') or {}).get(side) is not None]
    return float(np.median(asphalt)) if len(asphalt) * 2 >= len(files) and abs(np.median(asphalt) - kerbs[i]) <= EDGE_REACH else None
  lo, hi = edge('left', 0), edge('right', 1)
  if lo is None or hi is None:
    return None, 'one-way, no edges'
  between = [v for v in seen if lo + LANE_MIN * 0.8 < v < hi - LANE_MIN * 0.8]
  if len(between) != n - 1:
    return None, f'one-way, {len(between) + 1} lanes painted, GTA has {n}'
  if abs((lo + hi) / 2) > CENTRE_TOL:
    return None, 'one-way, off the line'
  widths = [round(float(x), 2) for x in np.diff([lo, *between, hi])]
  if not all(LANE_MIN <= x <= LANE_MAX for x in widths):
    return None, 'lane widths'
  out = {'lanes': widths}
  if (got := changes(between, kinds, False)) and set(got) != {'yes'}:
    out['change'] = got
  return out, None


def disagree(a: dict, b: dict, tol: float = AGREE) -> bool:
  """Whether two sources' cross-sections of a link differ by more than tol anywhere."""
  def edges(sec):
    f, b = np.cumsum(sec['forward']), np.cumsum(sec['backward'])
    return np.concatenate([[sec['median']], f, b])
  return len(a['forward']) != len(b['forward']) or len(a['backward']) != len(b['backward']) or \
    bool(np.abs(edges(a) - edges(b)).max() > tol)


TURNS = {'through;left': 'left;through', 'through;right': 'through;right', 'left;through': 'left;through',
         'left': 'left', 'right': 'right', 'through': 'through', 'reverse': 'reverse', 'left;right': 'left;right'}


def arrows(samples: list[dict]) -> dict[str, list[str]]:
  """The game files' painted arrows each way along a link: {'forward': [turn:lanes values], 'backward': [...]} (each
  direction's lanes left to right as seen travelling it), an arrow per lane, its kind the one most seen there. Arrows
  come as a sample's "arrows", or "features" whose kind is an arrow's."""
  out = {}
  for key, ahead in (('forward', True), ('backward', False)):
    seen = [(k, a) for k, d in enumerate(samples) if d.get('src') == GAMEFILES for a in (d.get('arrows') or []) + (d.get('features') or [])
            if (a.get('dir') in ('ab', 'ahead')) == ahead and a.get('conf', 1.0) >= CONF and a.get('kind') in TURNS and 'offset' in a]
    if not seen:
      continue
    offsets = sorted(a['offset'] if ahead else -a['offset'] for _, a in seen)
    lanes = [[offsets[0]]]
    for v in offsets[1:]:
      if v - lanes[-1][-1] < ARROW_APART:
        lanes[-1].append(v)
      else:
        lanes.append([v])
    kinds = []
    for lane in lanes:
      votes = Counter(TURNS[a['kind']] for _, a in seen if min(lane) - 1e-6 <= (a['offset'] if ahead else -a['offset']) <= max(lane) + 1e-6)
      kinds.append(votes.most_common(1)[0][0])
    out[key] = kinds
  return out
