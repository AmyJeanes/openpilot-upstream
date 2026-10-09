"""Where a tile's paint and kerbs disagree with our map.

Each tile is analysed at about 6 cm a pixel (the grab halved). Issues, each placed in the world with its length in m:
- missing: a run of paint (white or yellow, line-like: 2 m or longer, under 1 m wide, running along the road) on the
  map's road with no map line of this tile's level within TOL m, no kerb within KERB_TOL (edge lines, kerb faces), and no
  arrow or crossing where it is; a piece shorter than SHORT m counts only in line with another (a dashed line), not
  alone (text, grit, a hatching stroke);
- offset: such paint up to OFFSET m beside a map line of its colour and running with it (the line drawn off the paint);
- stray: a map line where the image shows the road but no paint within TOL m over two windows in a row (solid lines
  need a quarter of a window painted, dashed ones a few dashes in two of the freeway's dash periods, as the map has no
  dash phase); worn paint on pale concrete, which the paint masks miss, counts by how it stands out across the line;
- kind: a map solid line over evenly spaced dashes;
- colour: a map line on paint of the other colour only, that colour told by its tint against the road (faded yellow
  passes for white in the masks);
- kerb: a map kerb where the image shows the same road surface on both sides (a painted edge line along it included: a
  paved shoulder beyond), or yellow paint along it (a painted median drawn as kerbs, kerbs between a freeway's lanes);
  each issue takes the class of the road whose edge the kerb is;
- stop: a map stop line with no painted bar under it, and the offset to the nearest bar parallel to it if there is one
  (stop signs and give way lines with nothing painted are by design, so only lights are called bare; a crossing's
  stripes aren't a bar, and a stop line at a crossing's edge needs none);
- junction: paint inside a junction's area on the run of a map line that ends there (a through road's lines left out);
  the rest of a junction's paint (turn guides, chevrons, hatching) is the game's and isn't an issue.
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
SHORT = 3.0  # m: paint shorter than this is a line only in line with more of it (a dash of a dashed line)
PARTNER = 1.0  # m: the shortest paint that counts as more of a line (a worn dash)
ALIGN_GAP = 15.0  # m along between the middles of two dashes of one line at most (skipped dashes)
ALIGN_OFF = 0.35  # m across
ALIGN_ANGLE = 12.0  # deg
ACROSS = 50.0  # deg off the road's direction beyond which paint is a bar, a crossing's stripe or hatching
MAX_WIDTH = 1.0  # m: wider paint is a hatched area or a symbol
CHUNK = 6.0  # m: a long run of paint too wide for a line (a line on a bend) is judged in pieces this long
FILL = 0.7  # share of its length a line's paint covers at least
LINE_WIDTH = 0.1  # m: paint covers half its length this wide at least
OFFSET = 1.6  # m: missing paint this near a map line of its colour, running with it, is that line off the paint
OFFSET_ANGLE = 15.0  # deg
JUNCTION_MIN = 6.0  # m of paint in a junction area on the run of the map's lines before it counts
THROUGH_OFF = 0.6  # m: paint this near the run of a map line ending at a junction is that line carried through
THROUGH_ANGLE = 15.0  # deg
THROUGH_START = 5.0  # m past the line's end before paint counts (a turn guide starts out along the line it leaves)
THROUGH_REACH = 40.0  # m past the line's end
THROUGH_LONG = 8.0  # m: a run of paint this long inside a junction's area is a through line whatever the map's lines do
STOP_NEAR = 3.0  # m between a line's end and a stop line for the line to end at it
SAMPLE = 0.25  # m between samples along a map line
WINDOW = 6.0  # m: solid lines are judged over windows this long
COLOUR_WINDOW = 9.0  # m: a colour is judged over windows this long (a dash and a gap of a city line's at least)
DASH_WINDOW = 25.0  # m: dashed ones: two dash periods of a freeway's (4 m dashes, 12.2 m apart; a skipped dash leaves 20 m)
SOLID_COVER = 0.25
DASH_COVER = 0.06
VISIBLE = 0.6  # share of a window that must be visible road to judge it
LINE_DENSITY = 0.04  # share of the 2 TOL square a line's paint fills at least (a 5 cm line across 0.8 m: 0.06)
BG_RATIO = 2.0  # times the paint's density around
BG_M = 3.0  # m: the square that density is taken over
NOISY = 0.12  # paint density around above which the image can't tell a line
LIFT_MIN = 24.0  # brightness levels a stripe within TOL of a line stands over the road either side, averaged along it
LIFT_ALONG = 1.0  # m: the average's length (grit and texture average out, a dash doesn't)
DASH_RUN = (1.5, 7.0)  # m: a dash's length under a solid map line
DASH_GAP = 2.5  # m between dashes at least
DASH_SHARE = 0.65  # share of a window dashes cover at most
KIND_VISIBLE = 0.95
TINT_WHITE = 0.3  # paint's red-over-blue lift over the road as a share of its mean lift: white below,
TINT_YELLOW = 1.0  # yellow above (white paint 0 to 0.23, faded yellow 0.5 and up, port yellow 3.6 and up)
KERB_RUN = 4.0  # m of kerb on even road surface before it counts
KERB_STUB = 2.0  # m: a shorter kerb stub isn't worth an issue
SIDE = (0.3, 1.0)  # m either side of a kerb its surfaces are compared over
KERB_WINDOW = 3.0  # m along a kerb its profile is averaged over
KERB_DIFF = 10.0  # brightness step across a kerb under which the two sides are one surface
KERB_FLAT = 22.0  # brightness spread across a kerb (10th to 90th percentile of the averaged profile) of an even surface
KERB_OWNER = 2.0  # m between a kerb and the edge of the road it belongs to at most
STOP_SEARCH = 6.0  # m before and after a stop line a painted bar is looked for
STOP_COVER = 0.5
CROSSING_HALF = 1.8  # m either side of a crossing's middle its stripes take (3 m stripes)
CROSSING_EDGE = 4.0  # m from a crossing's middle a stop line lies at its edge
MINOR_CLASS = 7  # osm_to_roads.ROAD_CLASSES: service, track
MINOR_WEIGHT = 0.25
COVERED = 1.5  # m between the game's ground under the camera and the map's road height
WEIGHT = {"missing": 1.0, "offset": 1.0, "stray": 1.0, "kind": 0.8, "colour": 1.0, "kerb": 0.7, "stop": 1.0, "junction": 0.5}
LINE_KINDS = WHITE.replace("l", "").replace("s", "").replace("k", "") + YELLOW  # the lines along the road


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
  ends: list | None = None  # [[x, y], [x, y]]: where a run along a line starts and ends (world)

  def to_json(self) -> dict:
    out = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in asdict(self).items() if k != "ends"}
    if self.ends is not None:
      out["ends"] = [[round(float(c), 1) for c in p] for p in self.ends]
    return out


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


def angle(a: np.ndarray, b: np.ndarray) -> float:
  """Degrees between two undirected directions (0 to 90)."""
  c = abs(float(a @ b)) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-9)
  return math.degrees(math.acos(min(1.0, c)))


def nearest_on(segs_a: np.ndarray, segs_b: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Distance from a point [2] to each segment [N, 2] -> [N], and each segment's unit direction [N, 2]."""
  d = segs_b - segs_a
  n = np.maximum(np.hypot(*d.T), 1e-9)
  t = np.clip(((p - segs_a) * d).sum(1) / n ** 2, 0.0, 1.0)
  return np.hypot(*(segs_a + d * t[:, None] - p).T), d / n[:, None]


def dashes(flags: np.ndarray) -> bool:
  """Whether painted samples along a window are evenly spaced dashes: two or more runs of a dash's length with gaps
  between them, covering well under the whole."""
  closed = flags.copy()
  for a, b in runs(~flags):
    if a > 0 and b < len(flags) and (b - a) * SAMPLE < 0.5:  # gaps in a worn dash
      closed[a:b] = True
  rs = [(a, b) for a, b in runs(closed) if (b - a) * SAMPLE >= 0.5]
  if len(rs) < 2 or closed.mean() > DASH_SHARE:
    return False
  if any(not DASH_RUN[0] <= (b - a) * SAMPLE <= DASH_RUN[1] for a, b in rs if a > 0 and b < len(flags)):
    return False
  return all((rs[k + 1][0] - rs[k][1]) * SAMPLE >= DASH_GAP for k in range(len(rs) - 1))


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
    tm = self.map
    on = np.array([tm.on_level((a[2] + b[2]) / 2) and st != 2 for a, b, st in zip(tm.road_a, tm.road_b, tm.road_st, strict=True)], bool)
    self.roads = (tm.road_a[on, :2], tm.road_b[on, :2], tm.road_w[on], tm.road_cls[on] if tm.road_cls is not None else np.full(on.sum(), 0))

  def px(self, m: float) -> int:
    return max(1, int(round(m / self.mpp)))

  def world(self, u: float, v: float) -> tuple[float, float]:
    xy = self.cam.unproject(np.array([[u, v]]))[0]
    return float(xy[0]), float(xy[1])

  def add(self, kind: str, u: float, v: float, length: float, detail: str, colour: str = "", score: float | None = None,
          cls: int | None = None, ends=None):
    x, y = self.world(u, v)
    score = WEIGHT[kind] * length if score is None else score
    if cls is None:
      cls = self.road_class[int(np.clip(v, 0, self.size[1] - 1)), int(np.clip(u, 0, self.size[0] - 1))]
    if cls >= MINOR_CLASS:  # car parks, drives, alleys and tracks: the map lays them as a lane down the middle
      score *= MINOR_WEIGHT
      detail += " (service road)"
    self.issues.append(Issue(kind, x, y, length, score, detail, u, v, colour, None if ends is None else np.asarray(ends)[:, :2].tolist()))

  def across_roads(self, pc: dict) -> bool:
    """Whether a piece of paint runs across every map road of this level under it (where roads cross, the nearest
    road needn't be its own)."""
    a, b, w, _ = self.roads
    if not len(a):
      return False
    d, u = nearest_on(a, b, pc["xy"])
    under = d <= w / 2 + 2.0  # roads are cut back short of junctions
    if not under.any():
      under = d <= d.min() + 1e-6
    return all(angle(pc["dir"], u[k]) > ACROSS for k in np.flatnonzero(under))

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
    """Where paint is accounted for by a map mark of this tile's level (paint is only looked for on this level's road,
    and another level's marks over it are a deck's or a road's under it). A kerb accounts for white paint along it
    (an edge line, the kerb's face) but not yellow: a one-way road's left edge is a yellow line the map draws as a line."""
    tm = self.map
    im = tm.draw(self.size, tm.lines(WHITE + YELLOW), 2 * TOL + 0.2)
    tm.draw(self.size, tm.lines(ARROWS), 1.6, base=im)
    tm.draw(self.size, tm.lines("x"), 4.0, base=im)
    tm.draw(self.size, tm.lines(STOPS), 2 * TOL + 0.6, base=im)
    kerbs = np.asarray(tm.draw(self.size, tm.lines("e"), 2 * KERB_TOL)) > 0
    return (np.asarray(im) > 0) | (kerbs & ~self.paint.yellow)

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
  def pieces(self, min_line: float) -> list[dict]:
    """Line-like runs of paint min_line m or longer the map's marks don't account for, on this level's road: each with
    its middle (pixels and world), ends (world), length, direction (world), colour and the junction area it lies in."""
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
      return []
    ys, xs = np.nonzero(lab)
    ids = lab[ys, xs]
    order = np.argsort(ids)
    ys, xs, ids = ys[order], xs[order], ids[order]
    starts = np.flatnonzero(np.diff(np.concatenate([[0], ids]))) if len(ids) else []
    cell_m = c * self.mpp
    jid = self.junction_ids()

    def shape(cy, cx):
      pts = np.column_stack([cx, cy]).astype(np.float64) * cell_m
      mean = pts.mean(0)
      if len(pts) >= 3:
        evals, evecs = np.linalg.eigh(np.cov((pts - mean).T))
        major, minor = evecs[:, 1], evecs[:, 0]
      else:
        major, minor = np.array([1.0, 0.0]), np.array([0.0, 1.0])
      along = (pts - mean) @ major
      return mean, major, along, float(np.ptp(along)) + cell_m, float(np.ptp((pts - mean) @ minor)) + cell_m

    groups = []
    for k, s in enumerate(starts):
      e = starts[k + 1] if k + 1 < len(starts) else len(ids)
      cy, cx = ys[s:e], xs[s:e]
      if e - s < min_line / cell_m:
        continue
      _, _, along, length, width = shape(cy, cx)
      if width > MAX_WIDTH and length > 2 * CHUNK:  # a long line on a bend: judged a piece at a time
        part = np.floor((along - along.min()) / CHUNK).astype(int)
        groups += [(cy[part == q], cx[part == q]) for q in np.unique(part)]
      else:
        groups.append((cy, cx))
    out = []
    for cy, cx in groups:
      if len(cy) < min_line / cell_m:
        continue
      mean, major, along, length, width = shape(cy, cx)
      if length < min_line or width > MAX_WIDTH or not ins[cy, cx].mean() > 0.5:
        continue
      # a painted line fills its length; grit strung together by the grid doesn't
      filled = len(np.unique(np.floor((along - along.min()) / cell_m + 1e-6))) / max(1.0, length / cell_m)  # cells along the grid land on whole steps
      area = float(counts[cy, cx].sum()) * self.mpp ** 2
      if filled < FILL or area < LINE_WIDTH * length * 0.5:
        continue
      u, v = (mean / self.mpp) + c / 2
      h = length / 2 / self.mpp
      ends = self.cam.unproject(np.array([[u, v], [u - major[0] * h, v - major[1] * h], [u + major[0] * h, v + major[1] * h]]))
      junction = 0
      if inj[cy, cx].mean() > 0.5:
        j = jid[np.clip(cy * c + c // 2, 0, jid.shape[0] - 1), np.clip(cx * c + c // 2, 0, jid.shape[1] - 1)]
        j = j[j > 0]
        junction = int(np.bincount(j).argmax()) if len(j) else 0
      out.append({"u": float(u), "v": float(v), "xy": ends[0], "ends": ends[1:], "dir": (ends[2] - ends[1]) / max(np.hypot(*(ends[2] - ends[1])), 1e-9),
                  "length": length, "colour": "yellow" if yel[cy, cx].mean() > 0.5 else "white", "junction": junction})
    return out

  @staticmethod
  def in_line(a: dict, b: dict) -> bool:
    """Two pieces on one line: running the same way, one's middle on the other's run, within a dash gap."""
    if angle(a["dir"], b["dir"]) > ALIGN_ANGLE:
      return False
    d = b["xy"] - a["xy"]
    along = abs(float(d @ a["dir"]))
    return 0.5 < along < ALIGN_GAP and abs(float(d @ np.array([-a["dir"][1], a["dir"][0]]))) < ALIGN_OFF

  def check_missing(self):
    every = self.pieces(PARTNER)
    pieces = [pc for pc in every if pc["length"] >= MIN_LINE]
    self.dropped = []  # pieces left out, why (for the pictures)
    kept = []
    for pc in pieces:
      if not pc["junction"] and self.across_roads(pc):  # a junction's inside has no road direction
        self.dropped.append((pc, "across"))
      elif pc["length"] < SHORT and not any(q is not pc and self.in_line(pc, q) for q in every):
        self.dropped.append((pc, "alone"))
      else:
        kept.append(pc)
    lines = {col: [(pts, k) for k, pts in self.map.lines(LINE_KINDS) if (k in YELLOW) == (col == "yellow")] for col in ("white", "yellow")}
    rays = self.line_ends()
    inside: dict[int, list] = {}
    for pc in kept:
      if pc["junction"]:
        if pc["length"] >= THROUGH_LONG or any(self.on_ray(pc, ray) for ray in rays):
          inside.setdefault(pc["junction"], []).append(pc)
        else:
          self.dropped.append((pc, "junction"))
        continue
      off = self.beside(pc, lines[pc["colour"]])
      col, length = pc["colour"], pc["length"]
      if off is not None:
        self.add("offset", pc["u"], pc["v"], length, f"{col} paint {length:.1f} m {off:.1f} m off the map's {col} line", col, ends=pc["ends"])
      else:
        self.add("missing", pc["u"], pc["v"], length, f"{col} paint {length:.1f} m with no map line", col, ends=pc["ends"])
    for found in inside.values():  # one issue per junction area
      total = sum(p["length"] for p in found)
      if total < JUNCTION_MIN:
        continue
      w = np.array([p["length"] for p in found])
      u, v = (np.array([[p["u"], p["v"]] for p in found]) * w[:, None]).sum(0) / w.sum()
      yellow = sum(p["length"] for p in found if p["colour"] == "yellow")
      detail = f"{total:.0f} m of paint in {len(found)} pieces inside a junction area on the run of a map line ending there ({yellow:.0f} m yellow)"
      self.add("junction", float(u), float(v), total, detail, "yellow" if yellow > total / 2 else "white")

  def line_ends(self) -> list[tuple[np.ndarray, np.ndarray]]:
    """Each end of the map's lines of this level not at a stop line (an arm with lights or a stop has no lines across
    the junction, but GTA's guide paint carries on along them): the end point and the direction on past it (world)."""
    stops = [pts[:, :2] for _, pts in self.map.lines(STOPS) if len(pts) >= 2]
    out = []
    for _, pts in self.map.lines(LINE_KINDS):
      q = pts[:, :2]
      if len(q) < 2 or np.hypot(*(q[-1] - q[0])) < 2.0:
        continue
      for end, prev in ((q[-1], q[:-1][::-1]), (q[0], q[1:])):
        if any(nearest_on(s[:-1], s[1:], end)[0].min() < STOP_NEAR for s in stops):
          continue
        d = np.hypot(*(prev - end).T)
        k = int(np.searchsorted(d, 2.0)) if (d >= 2.0).any() else len(prev) - 1
        t = end - prev[min(k, len(prev) - 1)]
        out.append((end, t / max(np.hypot(*t), 1e-9)))
    return out

  @staticmethod
  def on_ray(pc: dict, ray: tuple[np.ndarray, np.ndarray]) -> bool:
    end, t = ray
    d = pc["xy"] - end
    along = float(d @ t)
    return THROUGH_START < along < THROUGH_REACH and abs(float(d @ np.array([-t[1], t[0]]))) < THROUGH_OFF and \
      angle(pc["dir"], t) < THROUGH_ANGLE

  @staticmethod
  def beside(pc: dict, lines: list) -> float | None:
    """How far a piece's middle is from the nearest map line of its colour that runs with it, if OFFSET or less."""
    best = None
    for pts, _ in lines:
      if len(pts) < 2:
        continue
      d, u = nearest_on(pts[:-1, :2], pts[1:, :2], pc["xy"])
      k = int(np.argmin(d))
      if d[k] <= OFFSET and angle(pc["dir"], u[k]) < OFFSET_ANGLE and (best is None or d[k] < best):
        best = float(d[k])
    return best

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

  def lift(self, q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """How far a narrow stripe within TOL of a line stands over the road either side, at each sample: the brightness
    across the line averaged over LIFT_ALONG m along it, its peak within TOL less the brighter side 0.5-0.7 m out. Worn
    paint on pale concrete, which the paint masks (lift as a share of the road's brightness) miss, shows in it."""
    h, w = self.paint.v.shape
    nrm = np.column_stack([t[:, 1], -t[:, 0]])
    offs = np.arange(-0.7, 0.7 + 1e-9, 0.05)
    vals = np.empty((len(offs), len(q)))
    for k, o in enumerate(offs):
      uv = self.cam.project(np.column_stack([q[:, :2] + nrm * o, q[:, 2]]))
      iu, iv = np.clip(np.round(uv[:, 0]).astype(int), 0, w - 1), np.clip(np.round(uv[:, 1]).astype(int), 0, h - 1)
      vals[k] = self.paint.v[iv, iu]
    n = max(1, int(round(LIFT_ALONG / SAMPLE)))
    c = np.pad(np.cumsum(vals, axis=1), ((0, 0), (1, 0)))
    i = np.arange(len(q))
    hi, lo = np.minimum(i + n // 2 + 1, len(q)), np.maximum(i - n // 2, 0)
    avg = (c[:, hi] - c[:, lo]) / (hi - lo)
    side = np.maximum(np.median(avg[offs <= -0.5], axis=0), np.median(avg[offs >= 0.5], axis=0))
    return avg[np.abs(offs) <= TOL + 1e-9].max(axis=0) - side

  def tint(self, iu: np.ndarray, iv: np.ndarray) -> float | None:
    """The paint within TOL of these samples against the road around them: its red-over-blue lift as a share of its
    mean lift (near 0 for white paint, well over 1 for yellow), or None where too little of either shows."""
    p = self.paint
    r = self.px(1.5)
    h, w = p.v.shape
    u0, u1, v0, v1 = max(0, iu.min() - r), min(w, iu.max() + r + 1), max(0, iv.min() - r), min(h, iv.max() + r + 1)
    at = np.zeros((v1 - v0, u1 - u0), bool)
    at[iv - v0, iu - u0] = True
    near, around = dilate(at, self.px(TOL)), dilate(at, r)
    paint = p.paint[v0:v1, u0:u1]
    pix = near & paint
    road = around & ~dilate(paint, 2) & p.asphalt[v0:v1, u0:u1]
    if pix.sum() < 20 or road.sum() < 50:
      return None
    rgb = p.rgb[v0:v1, u0:u1]
    lift = np.median(rgb[pix], axis=0) - np.median(rgb[road], axis=0)
    return float((lift[0] - lift[2]) / max(float(lift.mean()), 1.0))

  def check_lines(self):
    p = self.paint
    # a line is painted at a point where paint fills the square of side 2 TOL about it well over the paint's density in
    # the BG_M square around (bright grit, worn concrete and gravel roofs read as scattered paint), or a stripe stands
    # out across the line; where that density is high the image can't tell, and the point isn't judged
    kn, kb = self.px(2 * TOL) | 1, self.px(BG_M) | 1
    masks = (("w", p.white), ("y", p.yellow), ("a", p.paint))
    near = {c: box_sum(m, kn) / kn ** 2 for c, m in masks}
    bg = {c: box_sum(m, kb) / kb ** 2 for c, m in masks}
    white, yellow, painted = ((near[c] >= LINE_DENSITY) & (near[c] >= BG_RATIO * bg[c]) for c in "wya")
    noisy = bg["a"] > NOISY
    for kind, pts in self.map.lines(LINE_KINDS):
      dashed = kind in DASHED
      win = DASH_WINDOW if dashed else WINDOW
      need = DASH_COVER if dashed else SOLID_COVER
      q, t, uv, iu, iv, judged = self.samples(pts, SAMPLE)
      judged &= ~noisy[iv, iu]
      same = (yellow if kind in YELLOW else white)[iv, iu]
      other = (white if kind in YELLOW else yellow)[iv, iu]
      anyp = painted[iv, iu] | (self.lift(q, t) > LIFT_MIN)
      colour = "yellow" if kind in YELLOW else "white"
      stray, wrong = self.windows(q, judged, win, need, anyp, same, other, iu, iv, colour)
      kinds = np.zeros(len(q), bool)
      if not dashed:  # a solid line over dashes, judged over a dashed line's windows (seen whole: a hidden stretch
        n = max(1, int(round(DASH_WINDOW / SAMPLE)))  # would cut a solid line into dashes)
        step = max(1, n // 4)
        flagged = np.zeros(len(q), bool)
        for a in range(0, len(q) - n + 1, step):
          if judged[a:a + n].mean() >= KIND_VISIBLE and dashes(anyp[a:a + n]):
            flagged[a:a + n] = True
        for a, b in runs(flagged):
          kinds[a:b] = b - a >= n + step
        stray &= ~kinds
      for flags, issue in ((stray, "stray"), (kinds, "kind"), (wrong & ~stray & ~kinds, "colour")):
        for a, b in runs(flags):
          length = (b - a) * SAMPLE
          if length < (3.0 if issue == "colour" else 0.0):
            continue
          m = (a + b) // 2
          detail = {"stray": f"map {colour} {'dashed' if dashed else 'solid'} line {length:.0f} m with no paint",
                    "kind": f"map {colour} solid line {length:.0f} m over dashes",
                    "colour": f"map {colour} line {length:.0f} m on {'white' if colour == 'yellow' else 'yellow'} paint"}[issue]
          self.add(issue, uv[m, 0], uv[m, 1], length, detail, colour, ends=q[[a, b - 1]])

  def windows(self, q, judged, win, need, anyp, same, other, iu, iv, colour) -> tuple[np.ndarray, np.ndarray]:
    """Samples of a line in runs of failing windows: stray (no paint) over two windows in a row; wrong (the other
    colour's paint only, plainly that colour) over COLOUR_WINDOW m ones, as a colour needs no dash phase. Only whole
    windows are judged: one cut short by the line's end or the frame can't tell a gap between dashes from no paint."""
    stray = np.zeros(len(q), bool)
    wrong = np.zeros(len(q), bool)
    n = max(1, int(round(win / SAMPLE)))
    step = max(1, n // 4) if win > WINDOW else max(1, n // 2)
    for a in range(0, len(q) - n + 1, step):
      j = judged[a:a + n]
      if j.mean() >= VISIBLE and anyp[a:a + n][j].mean() < need:
        stray[a:a + n] = True
    nc = max(1, int(round(min(win, COLOUR_WINDOW) / SAMPLE)))
    for a in range(0, len(q) - nc + 1, max(1, nc // 2)):
      b = a + nc
      j = judged[a:b]
      if j.mean() < VISIBLE or anyp[a:b][j].mean() < need or same[a:b][j].mean() >= 0.5 * need or other[a:b][j].mean() < need:
        continue
      k = np.flatnonzero(j)
      tint = self.tint(iu[a:b][k], iv[a:b][k])
      if tint is not None and (tint < TINT_WHITE if colour == "yellow" else tint > TINT_YELLOW):
        wrong[a:b] = True
    long_enough = np.zeros(len(q), bool)
    for a, b in runs(stray):
      if b - a >= n + step:
        long_enough[a:b] = True
    return long_enough, wrong

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

  def kerb_owner(self, q: np.ndarray, t: np.ndarray) -> int | None:
    """The class of the road whose edge a kerb is: the road of this level running with it whose side lies nearest it
    (the road nearest a kerb is often a drive or an alley meeting it)."""
    a, b, w, cls = self.roads
    if not len(a):
      return None
    m = len(q) // 2
    d, u = nearest_on(a, b, q[m, :2])
    off = np.abs(d - w / 2)
    ok = (off < KERB_OWNER) & (np.abs(u @ t[m]) > 0.8)
    return int(cls[np.flatnonzero(ok)[np.argmin(off[ok])]]) if ok.any() else None

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
      edge_line = white_edge[iv, iu]  # a white edge line along it, which the profile leaves out
      bad = judged & clear & even & flat
      good = judged & clear & ~(even & flat)  # a kerb seen there
      total = float(np.hypot(*np.diff(pts[:, :2], axis=0).T).sum())
      owner = self.kerb_owner(q, t)
      if total < KERB_RUN:  # a kerb stub on the road
        if total >= KERB_STUB and len(q) >= 3 and bad.sum() >= 2 and bad.sum() >= 2 * good.sum():
          m = len(q) // 2
          self.add("kerb", uv[m, 0], uv[m, 1], total, f"map kerb stub {total:.1f} m on the road surface", cls=owner, ends=q[[0, -1]])
        continue
      # runs with no kerb seen, through samples paint or a gap hides (a kerb drawn across lanes crosses their lines)
      for a, b in runs(judged & ~good):
        length = (b - a) * 0.5
        if length < KERB_RUN or bad[a:b].mean() < 0.6:
          continue
        m = (a + b) // 2
        why = "yellow paint along it" if painted[a:b].mean() > 0.5 else \
          "a white edge line along it and the same road surface beyond" if edge_line[a:b].mean() > 0.5 else "the same road surface both sides"
        self.add("kerb", uv[m, 0], uv[m, 1], length, f"map kerb {length:.0f} m with {why}", cls=owner, ends=q[[a, b - 1]])

  # -------------------------------------------------------------------------------------------- stop lines
  def check_stops(self):
    crossings = self.map.lines("x")
    stripes = np.asarray(self.map.draw(self.size, crossings, 2 * CROSSING_HALF)) > 0 if crossings else np.zeros(self.paint.v.shape, bool)
    white = dilate(self.paint.white & ~stripes, self.px(0.12))
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
      elif here < 0.15 and kind == "l" and not self.at_crossing(q[len(q) // 2, :2], crossings):
        self.add("stop", uv[m, 0], uv[m, 1], length, f"map {name} stop line with no painted bar within {STOP_SEARCH:.0f} m",
                 score=4.0)

  @staticmethod
  def at_crossing(xy: np.ndarray, crossings: list) -> bool:
    return any(len(pts) >= 2 and nearest_on(pts[:-1, :2], pts[1:, :2], xy)[0].min() < CROSSING_EDGE for _, pts in crossings)


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
