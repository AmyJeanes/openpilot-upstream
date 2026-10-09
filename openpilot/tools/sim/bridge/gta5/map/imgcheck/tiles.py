"""Tiles to shoot: centres along every road of the map, overlapping, deduplicated and ordered to keep hops short.

A tile looks straight down from HEIGHT m with vertical field FOV, the image's long side along the road (the camera's
heading a quarter turn from the road's), so at 2560x1440 it covers about 75 m along the road and 42 m across it at 2.9 cm
a pixel (analysed at half size). Centres go every STEP m along each road's middle (roads.json's ways joined end to end
where they run on), a tile is dropped where an earlier one's middle part already covers it (within DEDUP m, at the
same level, its road running within 30 degrees of the earlier's), and the rest are ordered by a greedy walk that
follows a road to its end and then takes the nearest road end not yet shot (both its ends are candidates)."""
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

from openpilot.tools.sim.bridge.gta5.map.imgcheck.frame import MapData, heading_of

HEIGHT = 45.0  # m above the game's ground
FOV = 50.0  # deg, vertical
ASPECT = 16 / 9
STEP = 55.0  # m between centres along a road (the tile spans ~75 m)
DEDUP = 14.0  # m
UNDER_STEP = 10.0  # m between tiles under a deck (one covers about 17 m by 10 m at 5 m up)
UNDER_FOV = 90.0
UNDER_MAX = 5.0  # m above the road
UNDER_CLEARANCE = 3.0  # m under the deck's surface (its thickness, and a margin for the map's heights)
# city: Los Santos south of the Vinewood Hills, plus the freeways out of it
CITY = (-3300.0, -3900.0, 1600.0, 1400.0)  # x0, y0, x1, y1
# a few districts for a first look: downtown, Vinewood, Little Seoul, Del Perro, the Olympic interchange, Mirror Park
SAMPLE_AREAS = [(-50, -950, 250, -650), (150, 50, 650, 350), (-850, -1050, -500, -700), (-1650, -650, -1250, -300),
                (-350, -1350, 0, -1050), (1000, -650, 1300, -350)]
CLASSES = {"main": ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified"),
           "roads": ("motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential")}


@dataclass
class Tile:
  name: str
  x: float
  y: float
  z: float  # the map's road height there
  heading: float  # the camera's (GTA deg)
  road: str  # the road's class
  hop: float = 0.0  # m from the tile before
  height: float = HEIGHT  # the camera's, m above the ground (above z itself with under)
  fov: float = FOV
  under: bool = False  # a shot under a bridge deck: the camera at z + height, below the deck (topcam abs=1)


def under_decks(md: MapData, boxes=None, classes: str = "roads", step: float = UNDER_STEP) -> list[list[Tile]]:
  """Tiles under bridge decks: points every `step` m along roads where another road's surface lies 5-30 m above,
  with the camera below the deck (UNDER_CLEARANCE m under the deck's surface, UNDER_MAX m above the road at most), a
  wide field, the image's long side along the road. Where the deck is too low for that, no tile."""
  wanted = {md.classes.index(c) for c in CLASSES[classes]}
  a, b, w, cls = md.road_a, md.road_b, md.road_w, md.road_cls
  mid = (a + b) / 2
  out = []
  for k in np.flatnonzero(np.isin(cls, list(wanted)) & (md.road_st != 2)):
    p, q = a[k], b[k]
    if boxes is not None and not any(in_box(mid[k:k + 1, :2], bx)[0] for bx in boxes):
      continue
    seg = q[:2] - p[:2]
    length = float(np.hypot(*seg))
    if length < 1.0:
      continue
    near = md._near(md.road_cells, mid[k, :2], length / 2 + 30.0)
    row = []
    for s in np.arange(step / 2, length, step):
      pt = p + (q - p) * (s / length)
      d = _seg_dist(pt[:2], a[near, :2], b[near, :2])
      zz = (a[near, 2] + b[near, 2]) / 2
      over = near[(d < w[near] / 2) & (zz > pt[2] + 5.0) & (zz < pt[2] + 30.0)]
      if not len(over):
        continue
      deck = float(((a[over, 2] + b[over, 2]) / 2).min())
      height = min(UNDER_MAX, deck - pt[2] - UNDER_CLEARANCE)
      if height < 1.5:
        continue
      road = heading_of(*seg)
      row.append(Tile("", float(pt[0]), float(pt[1]), float(pt[2]), (road + 90.0) % 360.0, md.classes[cls[k]], 0.0, round(height, 1),
                      UNDER_FOV, True))
    if row:
      out.append(row)
  return out


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
  ab = b - a
  t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-9), 0, 1)
  return np.hypot(*(a + ab * t[:, None] - p).T)


def footprint(height: float = HEIGHT, fov: float = FOV) -> tuple[float, float]:
  """m along (the image's width) and across (its height) a tile covers at the ground."""
  across = 2 * height * math.tan(math.radians(fov) / 2)
  return across * ASPECT, across


def in_box(xy: np.ndarray, box) -> np.ndarray:
  x0, y0, x1, y1 = box
  return (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)


def candidates(md: MapData, classes, boxes=None, step: float = STEP) -> list[list[Tile]]:
  """Centres along each way's middle, a list per way, every `step` m (at least one per way, at its middle)."""
  out = []
  wanted = {md.classes.index(c) for c in classes}
  for k, (cls, _, _, pts, _, structure) in enumerate(md.ways):
    if cls not in wanted or structure == 2 or len(pts) < 2:
      continue
    if boxes is not None and not any(in_box(pts[:, :2], b).any() for b in boxes):
      continue
    d = np.hypot(*np.diff(pts[:, :2], axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(d)])
    if s[-1] < 5.0:
      continue
    n = max(1, int(round(s[-1] / step)))
    at = (np.arange(n) + 0.5) * s[-1] / n
    row = []
    for a in at:
      i = min(int(np.searchsorted(s, a, side="right")) - 1, len(d) - 1)
      f = (a - s[i]) / max(d[i], 1e-6)
      p = pts[i] + (pts[i + 1] - pts[i]) * f
      xs, ys = np.interp([a - 10, a + 10], s, pts[:, 0]), np.interp([a - 10, a + 10], s, pts[:, 1])
      road = heading_of(xs[1] - xs[0], ys[1] - ys[0])
      if boxes is not None and not any(in_box(p[None, :2], b)[0] for b in boxes):
        continue
      row.append(Tile("", float(p[0]), float(p[1]), float(p[2]), (road + 90.0) % 360.0, md.classes[cls]))
    if row:
      out.append(row)
  return out


def dedup(rows: list[list[Tile]], radius: float = DEDUP) -> list[list[Tile]]:
  """Drops tiles another (earlier, bigger roads first) already covers."""
  cell: dict[tuple[int, int], list[Tile]] = {}
  order = sorted(range(len(rows)), key=lambda r: (CLASSES["roads"].index(rows[r][0].road) if rows[r][0].road in CLASSES["roads"] else 9))
  out = []
  for r in order:
    kept = []
    for t in rows[r]:
      cx, cy = int(t.x // radius), int(t.y // radius)
      near = [o for dx in (-1, 0, 1) for dy in (-1, 0, 1) for o in cell.get((cx + dx, cy + dy), ())]
      if any(math.hypot(o.x - t.x, o.y - t.y) < radius and abs(o.z - t.z) < 4.0 and
             abs((o.heading - t.heading + 90) % 180 - 90) < 30 for o in near):
        continue
      cell.setdefault((cx, cy), []).append(t)
      kept.append(t)
    if kept:
      out.append(kept)
  return out


def order(rows: list[list[Tile]], start=(0.0, 0.0)) -> list[Tile]:
  """A greedy walk: along a road, then to the nearest end of a road not yet shot."""
  ends = np.array([[r[0].x, r[0].y, r[-1].x, r[-1].y] for r in rows])
  left = np.ones(len(rows), bool)
  pos = np.asarray(start, np.float64)
  out: list[Tile] = []
  while left.any():
    da = np.hypot(ends[:, 0] - pos[0], ends[:, 1] - pos[1])
    db = np.hypot(ends[:, 2] - pos[0], ends[:, 3] - pos[1])
    da[~left], db[~left] = np.inf, np.inf
    i = int(np.argmin(np.minimum(da, db)))
    row = rows[i] if da[i] <= db[i] else rows[i][::-1]
    left[i] = False
    for t in row:
      t.hop = float(math.hypot(t.x - pos[0], t.y - pos[1]))
      pos = np.array([t.x, t.y])
      out.append(t)
  for n, t in enumerate(out):
    t.name = f"t{n:05d}"
  return out


def plan(md: MapData, area: str = "city", classes: str = "roads", step: float = STEP, limit: int | None = None,
         under: bool = False) -> list[Tile]:
  """The tiles to shoot, in order; for the sample, `limit` is shared out over its districts, each walked in turn.
  under: the second pass, under bridge decks, for the roads the top-down tiles see only the decks of."""
  if under:
    boxes = {"city": [CITY], "all": None, "sample": SAMPLE_AREAS}[area]
    rows = dedup(under_decks(md, boxes, classes, UNDER_STEP), radius=UNDER_STEP * 0.6)
    tiles = order(rows, start=(boxes[0][0], boxes[0][1]) if boxes else (0.0, 0.0))
    return tiles[:limit] if limit else tiles
  if area == "sample":
    out: list[Tile] = []
    each = None if not limit else max(1, limit // len(SAMPLE_AREAS))
    for box in SAMPLE_AREAS:
      rows = dedup(candidates(md, CLASSES[classes], [box], step))
      walk = order(rows, start=(box[0], box[1]))[:each]
      if out and walk:
        walk[0].hop = math.hypot(walk[0].x - out[-1].x, walk[0].y - out[-1].y)
      out += walk
    for n, t in enumerate(out):
      t.name = f"t{n:05d}"
    return out
  boxes = {"city": [CITY], "all": None}[area]
  rows = dedup(candidates(md, CLASSES[classes], boxes, step))
  tiles = order(rows, start=(boxes[0][0], boxes[0][1]) if boxes else (0.0, 0.0))
  return tiles[:limit] if limit else tiles


def wait_for(hop: float) -> float:
  """s to let the game stream the scene after a hop of this many m (it streams around the camera's focus)."""
  if hop < 120:
    return 0.8
  if hop < 600:
    return 2.5
  return 5.0


def estimate(tiles: list[Tile], shot_s: float = 0.35) -> dict:
  """Shots, waits and storage for a plan (shot_s: the grab, its write and the state check per tile; the sample run
  measured 1.14 s a tile in all, with 0.8 s waits)."""
  wait = sum(wait_for(t.hop) for t in tiles)
  return {"tiles": len(tiles), "hours": round((wait + shot_s * len(tiles)) / 3600, 2), "long_hops": sum(t.hop >= 600 for t in tiles),
          "gb_png": round(len(tiles) * 5.5 / 1024, 1), "gb_jpg": round(len(tiles) * 1.55 / 1024, 1)}


def save(tiles: list[Tile], path: str, meta: dict):
  with open(path, "w") as f:
    json.dump({**meta, "tiles": [asdict(t) for t in tiles]}, f)


def load(path: str) -> tuple[list[Tile], dict]:
  with open(path) as f:
    d = json.load(f)
  return [Tile(**t) for t in d.pop("tiles")], d
