"""Where a tile's paint and kerbs disagree with our map.

Each tile is analysed at about 6 cm a pixel (the grab halved). Issues, each placed in the world with its length in m:
- missing: a run of paint (white or yellow, line-like: 2 m or longer, under 1 m wide) on the map's road with no map line
  of any kind within TOL m, no kerb within KERB_TOL (edge lines, kerb faces), and no arrow or crossing where it is;
- stray: a map line where the image shows the road but no paint within TOL m, judged over windows along it (solid lines
  need a quarter of a window painted, dashed ones a few dashes, as the map has no dash phase);
- colour: a map line on paint of the other colour only;
- kerb: a map kerb where the image shows the same road surface on both sides, or yellow paint along it (a painted median
  drawn as kerbs, kerbs between a freeway's lanes);
- stop: a map stop line with no painted bar under it, and the offset to the nearest bar parallel to it if there is one;
- junction: line-like paint inside a junction's area, where the map draws no lines (an area reaching over lanes).
What a bridge above hides (the map's road surfaces more than ABOVE m over a point) and what isn't visibly road (trees,
roofs) isn't judged."""
import math
from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageDraw

from openpilot.tools.sim.bridge.gta5.map.imgcheck.frame import ABOVE, ARROWS, DASHED, STOPS, WHITE, YELLOW, Camera, MapData
from openpilot.tools.sim.bridge.gta5.map.imgcheck.paint import Paint, box_sum, dilate, downscale

SCALE = 2  # the grab's pixels per analysed pixel
TOL = 0.4  # m: paint this near a map line is that line's
KERB_TOL = 1.0  # m: paint this near a kerb is its edge line or the kerb itself
ROAD_GROW = 1.5  # m: paint is looked for this far out of the map's road surface
BORDER = 0.06  # share of the image along each edge left out of judging (perspective lean of tall things)
CELL_M = 0.25  # m: the grid paint is grouped on
MIN_LINE = 2.0  # m: shorter paint is text, symbols or grit
MAX_WIDTH = 1.0  # m: wider paint is a hatched area or a symbol
FILL = 0.7  # share of its length a line's paint covers at least
LINE_WIDTH = 0.1  # m: paint covers half its length this wide at least
JUNCTION_MIN = 6.0  # m of line-like paint in a junction area before it counts
SAMPLE = 0.25  # m between samples along a map line
WINDOW = 6.0  # m: solid lines are judged over windows this long
DASH_WINDOW = 9.0  # m: dashed ones (3 m dashes, 6 m gaps)
SOLID_COVER = 0.25
DASH_COVER = 0.06
VISIBLE = 0.6  # share of a window that must be visible road to judge it
LINE_DENSITY = 0.04  # share of the 2 TOL square a line's paint fills at least (a 5 cm line across 0.8 m: 0.06)
BG_RATIO = 2.0  # times the paint's density around
BG_M = 3.0  # m: the square that density is taken over
NOISY = 0.12  # paint density around above which the image can't tell a line
KERB_RUN = 4.0  # m of kerb on even road surface before it counts
SIDE = (0.3, 1.0)  # m either side of a kerb its surfaces are compared over
KERB_WINDOW = 3.0  # m along a kerb its profile is averaged over
KERB_DIFF = 10.0  # brightness step across a kerb under which the two sides are one surface
KERB_FLAT = 22.0  # brightness spread across a kerb (10th to 90th percentile of the averaged profile) of an even surface
STOP_SEARCH = 6.0  # m before and after a stop line a painted bar is looked for
STOP_COVER = 0.5
MINOR_CLASS = 7  # osm_to_roads.ROAD_CLASSES: service, track
MINOR_WEIGHT = 0.25
COVERED = 1.5  # m between the game's ground under the camera and the map's road height
WEIGHT = {"missing": 1.0, "stray": 1.0, "colour": 1.0, "kerb": 0.7, "stop": 1.0, "junction": 0.5}


@dataclass
class Issue:
  kind: str
  x: float
  y: float
  length: float  # m (stop lines: their length)
  score: float
  detail: str
  u: float  # in the analysed image
  v: float
  colour: str = ""

  def to_json(self) -> dict:
    return {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(self).items()}


def label(mask: np.ndarray) -> tuple[np.ndarray, int]:
  """8-connected components of a small boolean grid: labels (0 none, 1..n) by repeated min-propagation."""
  big = np.iinfo(np.int32).max
  lab = np.where(mask, np.arange(1, mask.size + 1, dtype=np.int32).reshape(mask.shape), big)
  while True:
    p = np.pad(lab, 1, constant_values=big)
    m = lab.copy()
    for dy in (-1, 0, 1):
      for dx in (-1, 0, 1):
        m = np.minimum(m, p[1 + dy:1 + dy + lab.shape[0], 1 + dx:1 + dx + lab.shape[1]])
    m = np.where(mask, m, big)
    if np.array_equal(m, lab):
      break
    lab = m
  lab = np.where(mask, lab, 0)
  return lab, len(np.unique(lab[mask]))


def block_any(mask: np.ndarray, c: int) -> np.ndarray:
  h, w = mask.shape[0] // c * c, mask.shape[1] // c * c
  return mask[:h, :w].reshape(h // c, c, w // c, c).any(axis=(1, 3))


def resample(points: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
  """Points every `step` m along a polyline [N, 3], and unit tangents there."""
  d = np.hypot(*np.diff(points[:, :2], axis=0).T)
  s = np.concatenate([[0.0], np.cumsum(d)])
  if s[-1] < 1e-6:
    return points[:1], np.array([[1.0, 0.0]])
  q = np.arange(0.0, s[-1] + 1e-9, step)
  out = np.column_stack([np.interp(q, s, points[:, i]) for i in range(3)])
  i = np.clip(np.searchsorted(s, q, side="right") - 1, 0, len(d) - 1)
  t = np.diff(points[:, :2], axis=0)[i] / np.maximum(d[i], 1e-9)[:, None]
  return out, t


def runs(flags: np.ndarray) -> list[tuple[int, int]]:
  e = np.flatnonzero(np.diff(np.concatenate([[0], flags.astype(np.int8), [0]])))
  return list(zip(e[::2].tolist(), e[1::2].tolist(), strict=True))


class TileCheck:
  """One tile's image against the map: masks, samples and issues."""

  def __init__(self, img: np.ndarray, cam: Camera, md: MapData, scale: int = SCALE):
    small = downscale(img, scale)
    self.img = small
    self.cam = cam.scaled(small.shape[1], small.shape[0])
    self.mpp = self.cam.m_per_px
    self.size = (small.shape[1], small.shape[0])
    self.paint = Paint(small, self.mpp)
    self.map = md.tile(self.cam)
    self.issues: list[Issue] = []
    h, w = small.shape[:2]
    b = int(BORDER * min(h, w))
    self.inside = np.zeros((h, w), bool)
    self.inside[b:h - b, b:w - b] = True
    # stacked roads: where one map road lies over another, the lower is hidden (a tile judges the top layer where its
    # road is the top, and leaves its road out where a deck lies over it)
    self.overhead, floor = self.map.stack(self.size)
    lvl = self.map.level
    with np.errstate(invalid="ignore"):
      self.hidden = (self.overhead - floor > ABOVE) & (np.abs(floor - lvl) < np.abs(self.overhead - lvl))
    self.road = self.map.surface(self.size, grow=ROAD_GROW) & ~self.hidden
    self.road_class = self.map.classes(self.size)
    self.sharp = self.paint.sharpness()
    surface = self.map.surface(self.size)
    self.road_share = float(surface.mean())
    # the share of this level's road a deck above hides: judged from a shot under the deck, if there is one
    self.hidden_share = float((surface & self.hidden).sum() / max(surface.sum(), 1))
    k = self.px(1.0) | 1
    self.asphalt_near = box_sum(self.paint.asphalt, k) >= 0.6 * k * k

  def px(self, m: float) -> int:
    return max(1, int(round(m / self.mpp)))

  def world(self, u: float, v: float) -> tuple[float, float]:
    xy = self.cam.unproject(np.array([[u, v]]))[0]
    return float(xy[0]), float(xy[1])

  def add(self, kind: str, u: float, v: float, length: float, detail: str, colour: str = "", score: float | None = None):
    x, y = self.world(u, v)
    score = WEIGHT[kind] * length if score is None else score
    cls = self.road_class[int(np.clip(v, 0, self.size[1] - 1)), int(np.clip(u, 0, self.size[0] - 1))]
    if cls >= MINOR_CLASS:  # car parks, drives, alleys and tracks: the map lays them as a lane down the middle
      score *= MINOR_WEIGHT
      detail += " (service road)"
    self.issues.append(Issue(kind, x, y, length, score, detail, u, v, colour))

  def junction_mask(self, shrink: float = 0.0) -> np.ndarray:
    im = Image.new("L", self.size, 0)
    dr = ImageDraw.Draw(im)
    for _, pts in self.map.lines("j"):
      uv = self.cam.project(pts)
      if len(uv) >= 3:
        dr.polygon([tuple(p) for p in uv], fill=255)
    if shrink > 0:
      for _, pts in self.map.lines("j"):
        uv = self.cam.project(pts)
        dr.line([tuple(p) for p in uv], fill=0, width=self.px(2 * shrink))
    return np.asarray(im) > 0

  def junction_ids(self) -> np.ndarray:
    """Each pixel's junction area (its index among the tile's + 1, 0 for none): int32 [h, w]."""
    im = Image.new("I", self.size, 0)
    dr = ImageDraw.Draw(im)
    for n, (_, pts) in enumerate(self.map.lines("j")):
      uv = self.cam.project(pts)
      if len(uv) >= 3:
        dr.polygon([tuple(p) for p in uv], fill=n + 1)
    return np.asarray(im, np.int32)

  def explained(self) -> np.ndarray:
    """Where paint is accounted for by a map mark (any level: a road below is drawn where it shows)."""
    tm = self.map
    im = tm.draw(self.size, tm.lines(WHITE + YELLOW, level_only=False), 2 * TOL + 0.2)
    tm.draw(self.size, tm.lines("e", level_only=False), 2 * KERB_TOL, base=im)
    tm.draw(self.size, tm.lines(ARROWS, level_only=False), 1.6, base=im)
    tm.draw(self.size, tm.lines("x", level_only=False), 4.0, base=im)
    tm.draw(self.size, tm.lines(STOPS, level_only=False), 2 * TOL + 0.6, base=im)
    return np.asarray(im) > 0

  @property
  def covered(self) -> bool:
    """The game's ground under the camera is well off the map's road there: something over the road (a deck, a plaza,
    a roof) that the camera sees instead of it."""
    return not math.isnan(self.cam.zmap) and abs(self.cam.ground - self.cam.zmap) > COVERED

  def run(self) -> list[Issue]:
    if self.paint.blank or self.covered:
      return self.issues
    self.check_missing()
    self.check_lines()
    self.check_kerbs()
    self.check_stops()
    return self.issues

  # -------------------------------------------------------------------------------------------- paint off the map
  def check_missing(self):
    p = self.paint
    self.junction = self.junction_mask()
    unexplained = p.strong & self.road & ~self.explained()
    self.unexplained = unexplained
    c = self.px(CELL_M)
    grid = block_any(unexplained, c)
    hh, ww = grid.shape
    counts = unexplained[:hh * c, :ww * c].reshape(hh, c, ww, c).sum(axis=(1, 3))
    yel = block_any(unexplained & p.yellow, c)
    inj = block_any(self.junction_mask(shrink=1.5) & unexplained, c)
    kb = self.px(BG_M) | 1
    ins = block_any(self.inside & ~(box_sum(p.paint, kb) / kb ** 2 > NOISY), c)
    lab, n = label(grid)
    if not n:
      return
    ys, xs = np.nonzero(lab)
    ids = lab[ys, xs]
    order = np.argsort(ids)
    ys, xs, ids = ys[order], xs[order], ids[order]
    starts = np.flatnonzero(np.diff(np.concatenate([[0], ids]))) if len(ids) else []
    cell_m = c * self.mpp
    jid = self.junction_ids()
    inside: dict[int, list] = {}
    for k, s in enumerate(starts):
      e = starts[k + 1] if k + 1 < len(starts) else len(ids)
      cy, cx = ys[s:e], xs[s:e]
      if e - s < MIN_LINE / cell_m:
        continue
      pts = np.column_stack([cx, cy]).astype(np.float64) * cell_m
      mean = pts.mean(0)
      if len(pts) >= 3:
        evals, evecs = np.linalg.eigh(np.cov((pts - mean).T))
        major, minor = evecs[:, 1], evecs[:, 0]
      else:
        major, minor = np.array([1.0, 0.0]), np.array([0.0, 1.0])
      along = (pts - mean) @ major
      length = float(np.ptp(along)) + cell_m
      width = float(np.ptp((pts - mean) @ minor)) + cell_m
      if length < MIN_LINE or width > MAX_WIDTH or not ins[cy, cx].mean() > 0.5:
        continue
      # a painted line fills its length; grit strung together by the grid doesn't
      filled = len(np.unique(np.floor((along - along.min()) / cell_m))) / max(1.0, length / cell_m)
      area = float(counts[cy, cx].sum()) * self.mpp ** 2
      if filled < FILL or area < LINE_WIDTH * length * 0.5:
        continue
      colour = "yellow" if yel[cy, cx].mean() > 0.5 else "white"
      u, v = (mean / self.mpp) + c / 2
      if inj[cy, cx].mean() > 0.5:
        ids = jid[np.clip(cy * c + c // 2, 0, jid.shape[0] - 1), np.clip(cx * c + c // 2, 0, jid.shape[1] - 1)]
        ids = ids[ids > 0]
        if len(ids):
          inside.setdefault(int(np.bincount(ids).argmax()), []).append((u, v, length, colour))
      else:
        self.add("missing", u, v, length, f"{colour} paint {length:.1f} m with no map line", colour)
    for pieces in inside.values():  # one issue per junction area: hatching and chevrons come in many pieces
      total = sum(p[2] for p in pieces)
      if total < JUNCTION_MIN:
        continue
      w = np.array([p[2] for p in pieces])
      u, v = (np.array([p[:2] for p in pieces]) * w[:, None]).sum(0) / w.sum()
      yellow = sum(p[2] for p in pieces if p[3] == "yellow")
      self.add("junction", float(u), float(v), total, f"{total:.0f} m of line-like paint in {len(pieces)} pieces inside a junction "
               f"area ({yellow:.0f} m yellow)", "yellow" if yellow > total / 2 else "white")

  # -------------------------------------------------------------------------------------------- map lines off the paint
  def samples(self, pts: np.ndarray, step: float):
    """Samples along a map line: world points, tangents, pixel coordinates, and which are judged (in the image, not
    hidden under a bridge, visibly road)."""
    q, t = resample(pts, step)
    uv = self.cam.project(q)
    h, w = self.paint.v.shape
    iu, iv = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    inside = (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h)
    iu, iv = np.clip(iu, 0, w - 1), np.clip(iv, 0, h - 1)
    judged = inside & self.inside[iv, iu] & ~(np.nan_to_num(self.overhead[iv, iu], nan=-1e9) > q[:, 2] + ABOVE) & self.asphalt_near[iv, iu]
    return q, t, uv, iu, iv, judged

  def check_lines(self):
    p = self.paint
    # a line is painted at a point where paint fills the square of side 2 TOL about it well over the paint's density in
    # the BG_M square around (bright grit, worn concrete and gravel roofs read as scattered paint); where that density
    # is high the image can't tell, and the point isn't judged
    kn, kb = self.px(2 * TOL) | 1, self.px(BG_M) | 1
    masks = (("w", p.white), ("y", p.yellow), ("a", p.paint))
    near = {c: box_sum(m, kn) / kn ** 2 for c, m in masks}
    bg = {c: box_sum(m, kb) / kb ** 2 for c, m in masks}
    white, yellow, painted = ((near[c] >= LINE_DENSITY) & (near[c] >= BG_RATIO * bg[c]) for c in "wya")
    noisy = bg["a"] > NOISY
    for kind, pts in self.map.lines(WHITE.replace("l", "").replace("s", "").replace("k", "") + YELLOW):
      dashed = kind in DASHED
      win = DASH_WINDOW if dashed else WINDOW
      need = DASH_COVER if dashed else SOLID_COVER
      q, t, uv, iu, iv, judged = self.samples(pts, SAMPLE)
      judged &= ~noisy[iv, iu]
      same = (yellow if kind in YELLOW else white)[iv, iu]
      other = (white if kind in YELLOW else yellow)[iv, iu]
      anyp = painted[iv, iu]
      n = max(1, int(round(win / SAMPLE)))
      stray = np.zeros(len(q), bool)
      wrong = np.zeros(len(q), bool)
      for a in range(0, len(q), max(1, n // 2)):
        b = min(len(q), a + n)
        if b - a < n * 0.5:
          continue
        j = judged[a:b]
        if j.mean() < VISIBLE:
          continue
        cov_same, cov_other, cov_any = same[a:b][j].mean(), other[a:b][j].mean(), anyp[a:b][j].mean()
        if cov_any < need:
          stray[a:b] = True
        elif cov_same < 0.5 * need and cov_other >= need:
          wrong[a:b] = True
      colour = "yellow" if kind in YELLOW else "white"
      for flags, issue in ((stray, "stray"), (wrong & ~stray, "colour")):
        for a, b in runs(flags):
          length = (b - a) * SAMPLE
          if length < (win if issue == "stray" else 3.0) * 0.99:
            continue
          m = (a + b) // 2
          detail = (f"map {colour} {'dashed' if dashed else 'solid'} line {length:.0f} m with no paint" if issue == "stray"
                    else f"map {colour} line {length:.0f} m on {'white' if colour == 'yellow' else 'yellow'} paint")
          self.add(issue, uv[m, 0], uv[m, 1], length, detail, colour)

  # -------------------------------------------------------------------------------------------- kerbs
  def across(self, q: np.ndarray, nrm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The brightness profile across a line at each sample, averaged over KERB_WINDOW along it (so tyre tracks and
    patches crossing it average out while a kerb's edge, running along it, stays), paint and what isn't road left
    out: the offsets [K], the profiles [K, N] (NaN where nothing was read)."""
    h, w = self.paint.v.shape
    offs = np.arange(-1.0, 1.0 + 1e-9, 0.1)
    vals = np.full((len(offs), len(q)), np.nan)
    for k, o in enumerate(offs):
      uv = self.cam.project(np.column_stack([q[:, :2] + nrm * o, q[:, 2]]))
      iu, iv = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1), np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
      ok = self.paint.asphalt[iv, iu] & ~self.paint.paint[iv, iu]
      vals[k, ok] = self.paint.v[iv, iu][ok]
    n = max(1, int(round(KERB_WINDOW / 0.5)))
    got = ~np.isnan(vals)

    def moving(a):  # sums over n samples about each, along axis 1, the same length
      c = np.pad(np.cumsum(a, axis=1), ((0, 0), (1, 0)))
      i = np.arange(a.shape[1])
      return c[:, np.minimum(i + n // 2 + 1, a.shape[1])] - c[:, np.maximum(i - n // 2, 0)]
    s, c = moving(np.where(got, vals, 0.0)), moving(got.astype(float))
    with np.errstate(all="ignore"):
      return offs, np.where(c >= 0.5 * min(n, len(q)), s / c, np.nan)

  def check_kerbs(self):
    yellow = dilate(self.paint.yellow, self.px(0.25))
    white_edge = dilate(self.paint.white, self.px(0.3))
    for _, pts in self.map.lines("e"):
      q, t, uv, iu, iv, judged = self.samples(pts, 0.5)
      if len(q) < 2:
        continue
      nrm = np.column_stack([t[:, 1], -t[:, 0]])
      offs, prof = self.across(q, nrm)
      left, right = prof[(offs <= -SIDE[0]) & (offs >= -SIDE[1])], prof[(offs >= SIDE[0]) & (offs <= SIDE[1])]
      with np.errstate(all="ignore"):
        step = np.abs(np.nanmean(left, axis=0) - np.nanmean(right, axis=0))
        spread = np.nanpercentile(prof, 90, axis=0) - np.nanpercentile(prof, 10, axis=0)
      clear = ~np.isnan(step) & ~np.isnan(spread) & (np.isnan(prof).mean(axis=0) < 0.4)  # both sides read
      even = clear & (np.nan_to_num(step, nan=1e9) < KERB_DIFF)
      flat = clear & (np.nan_to_num(spread, nan=1e9) < KERB_FLAT)
      painted = yellow[iv, iu]
      # a white edge line along it: the map's edge on the painted edge, a shoulder beyond (not judged as a kerb)
      edge_line = white_edge[iv, iu]
      bad = judged & clear & even & flat & ~edge_line
      good = judged & ((clear & ~(even & flat)) | edge_line)  # a kerb (or edge line) seen there
      total = float(np.hypot(*np.diff(pts[:, :2], axis=0).T).sum())
      if total < KERB_RUN and len(q) >= 3 and bad.sum() >= 2 and bad.sum() >= 2 * good.sum():  # a kerb stub on the road
        m = len(q) // 2
        self.add("kerb", uv[m, 0], uv[m, 1], total, f"map kerb stub {total:.1f} m on the road surface")
        continue
      # runs with no kerb seen, through samples paint or a gap hides (a kerb drawn across lanes crosses their lines)
      for a, b in runs(judged & ~good):
        length = (b - a) * 0.5
        if length < KERB_RUN or bad[a:b].mean() < 0.6:
          continue
        m = (a + b) // 2
        why = "yellow paint along it" if painted[a:b].mean() > 0.5 else "the same road surface both sides"
        self.add("kerb", uv[m, 0], uv[m, 1], length, f"map kerb {length:.0f} m with {why}")

  # -------------------------------------------------------------------------------------------- stop lines
  def check_stops(self):
    white = dilate(self.paint.white, self.px(0.12))
    h, w = white.shape
    for kind, pts in self.map.lines(STOPS):
      q, t, uv, iu, iv, judged = self.samples(pts, 0.1)
      if len(q) < 10:
        continue
      keep = slice(len(q) // 10, len(q) - len(q) // 10)  # the ends meet kerbs and edge lines
      q, t, judged = q[keep], t[keep], judged[keep]
      if judged.mean() < VISIBLE:
        continue
      nrm = np.column_stack([t[:, 1], -t[:, 0]])
      offs = np.arange(-STOP_SEARCH, STOP_SEARCH + 1e-9, 0.1)
      cover = np.zeros(len(offs))
      for i, o in enumerate(offs):
        uvo = self.cam.project(np.column_stack([q[:, :2] + nrm * o, q[:, 2]]))
        ju, jv = np.round(uvo[:, 0]).astype(int), np.round(uvo[:, 1]).astype(int)
        ok = (ju >= 0) & (ju < w) & (jv >= 0) & (jv < h)
        cover[i] = white[np.clip(jv, 0, h - 1), np.clip(ju, 0, w - 1)][ok & judged].mean() if (ok & judged).any() else 0.0
      here = cover[np.abs(offs) <= TOL + 1e-9].max()
      length = float(np.hypot(*(pts[-1, :2] - pts[0, :2])))
      name = {"l": "lights'", "s": "stop sign's", "k": "give way"}[kind]
      if here >= STOP_COVER:
        continue
      best = int(np.argmax(cover))
      m = len(uv) // 2
      if cover[best] >= STOP_COVER:
        top = cover >= 0.9 * cover[best]  # the bar's middle: the middle of the run of offsets it covers
        a = b = best
        while a > 0 and top[a - 1]:
          a -= 1
        while b < len(top) - 1 and top[b + 1]:
          b += 1
        off = float(offs[(a + b) // 2])
        self.add("stop", uv[m, 0], uv[m, 1], length, f"map {name} stop line {abs(off):.1f} m off the painted bar",
                 score=4.0 + 3.0 * abs(off))
      elif here < 0.15:
        self.add("stop", uv[m, 0], uv[m, 1], length, f"map {name} stop line with no painted bar within {STOP_SEARCH:.0f} m",
                 score=4.0)


def check(img: np.ndarray, cam: Camera, md: MapData, scale: int = SCALE) -> TileCheck:
  tc = TileCheck(img, cam, md, scale)
  tc.run()
  return tc


def unloaded(tc: TileCheck) -> bool:
  """A frame the game hadn't streamed (blurry LOD textures) or drew blank."""
  return tc.paint.blank or tc.sharp < 6.0


def summary(tc: TileCheck) -> dict:
  by = {}
  for i in tc.issues:
    by.setdefault(i.kind, [0, 0.0])
    by[i.kind][0] += 1
    by[i.kind][1] += i.score
  return {"score": round(sum(i.score for i in tc.issues), 1), "by_kind": {k: [n, round(s, 1)] for k, (n, s) in by.items()},
          "sharp": round(tc.sharp, 1), "road_share": round(tc.road_share, 3), "unloaded": unloaded(tc), "covered": tc.covered,
          "m_per_px": round(tc.mpp, 4), "level": round(tc.map.level, 2), "hidden_share": round(tc.hidden_share, 3)}
