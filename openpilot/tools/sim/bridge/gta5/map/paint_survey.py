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

Only two-way links are corrected, and only where the samples agree with each other and with GTA's lane counts:
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
HALVES_SWAPPED = {'solid_dashed': 'dashed_solid', 'dashed_solid': 'solid_dashed'}  # a line seen the other way
CHANGE = {(True, True): 'yes', (False, False): 'no', (False, True): 'not_left', (True, False): 'not_right'}
ARROW_APART = 1.5  # m between two lanes' arrows
GAMEFILES = 'gamefiles'


def node_key(s: str) -> tuple[int, int]:
  a, i = s.split(':')
  return int(a), int(i)


def load(paths) -> dict[tuple, list[dict]]:
  """{(a, b): [sample]} by link, each sample's offsets seen travelling a -> b; junction and bay sections left out."""
  out = defaultdict(list)
  for path in paths:
    with open(path) as f:
      for line in f:
        d = json.loads(line)
        if d.get('junction') or d.get('bay'):
          continue
        a, b = node_key(d['a']), node_key(d['b'])
        out[(b, a) if d.get('dir', 'ab') == 'ba' else (a, b)].append(d)
  return out


def _flip(d: dict) -> dict:
  kerbs = d.get('kerbs') or {}
  return {**d,
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


def sources(samples: list[dict]) -> tuple[list[dict], list[dict]]:
  """(the samples to use, the others to cross-check them): the game files' where there are enough."""
  files = [d for d in samples if d.get('src') == GAMEFILES]
  camera = [d for d in samples if d.get('src') != GAMEFILES]
  return (files, camera) if len(files) >= MIN_SAMPLES else (camera, files)


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


def correct(samples: list[dict], fwd: int, back: int, kerbs: tuple[float, float]):
  """The painted cross-section of a two-way link with fwd and back lanes and its kerbs where the class layout has them
  (m left and right of its line), from its samples seen along it (one source's: sources()): ({'forward': [widths],
  'backward': [widths] (each direction's lanes left to right as seen travelling it), 'median': m, and from the game
  files' line kinds 'change:forward' / 'change:backward': [change:lanes values] where a line can't be crossed}, None),
  or (None, why not)."""
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
  if fwd == back:
    kerbs = road_kerbs(samples, kerbs)
  lines = [v for v, c in white if c >= need]
  right = [v for v in lines if hi + LANE_MIN * 0.8 < v < kerbs[1] - LANE_MIN * 0.8]
  left = sorted(-v for v in lines if kerbs[0] + LANE_MIN * 0.8 < v < lo - LANE_MIN * 0.8)
  if len(right) != fwd - 1 or len(left) != back - 1:
    return None, f'{len(left) + 1}+{len(right) + 1} lanes painted, GTA has {back}+{fwd}'
  f = [round(float(x), 2) for x in np.diff([hi, *right, kerbs[1]])]
  b = [round(float(x), 2) for x in np.diff([-lo, *left, -kerbs[0]])]
  if not all(LANE_MIN <= x <= LANE_MAX for x in f + b):
    return None, 'lane widths'
  out = {'forward': f, 'backward': b, 'median': round(hi - lo, 2)}
  for key, between, flip in (('change:forward', right, False), ('change:backward', left, True)):
    if (got := changes(between, kinds, flip)) and set(got) != {'yes'}:
      out[key] = got
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
