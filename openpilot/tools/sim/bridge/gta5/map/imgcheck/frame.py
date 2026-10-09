"""A top-down tile's camera and our map in its frame.

The plugin's topcam (core.cpp UpdateTopCam) puts a scripted camera `height` m above the game's ground under (x, y),
looking straight down, its vertical field of view `fov`, the image's up the GTA heading (deg counterclockwise from
north). A point projects as a pinhole camera does: u = W/2 + f*right/d, v = H/2 - f*up/d, f = (H/2)/tan(fov/2), d its
depth below the camera, so a metre spans 2*height*tan(fov/2)/H pixels at the ground. The map's lines are drawn at their
own heights, moved by the game's ground under the camera less the map's height there (the map's node heights are a
few tens of cm off the game's ground in places; their differences across a tile are what matter). The same convention
as the map audit's topshot.py, which matched the in-game overlay to 1-3 px.

The map is the overlay's own road marks (gta5_overlay.road_marks, from its cache), so a tile shows what the overlay
draws, and roads.json's road surfaces (each way's width about its middle, the junctions' areas) for where the roads are
and which lie above a point (bridges, which hide what is under them)."""
import glob
import hashlib
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

MAP_DIR = os.path.expanduser(os.getenv("GTA5_MAP", "~/gta5map_lanes"))
CELL = 50.0  # m: the spatial index's cells
LEVEL = 6.0  # m: marks further above or below the tile's road height belong to another level
ABOVE = 4.0  # m: a road surface this far above a point hides it
# the marks' kinds (gta5_overlay.py's message format), grouped as the paint they stand for
WHITE = "dwplsk"  # lane lines, parking lane edges, stop and give way lines
YELLOW = "cy"
ARROWS = "LTR"
DASHED = "dypk"  # drawn as one line, painted in dashes (the map has no dash phase)
STOPS = "lsk"


def heading_of(dx: float, dy: float) -> float:
  """GTA heading (deg counterclockwise from north) of a direction."""
  return math.degrees(math.atan2(-dx, dy)) % 360.0


def axes(heading: float) -> tuple[np.ndarray, np.ndarray]:
  """The image's up and right as world directions for a camera at this heading."""
  h = math.radians(heading)
  return np.array([-math.sin(h), math.cos(h)]), np.array([math.cos(h), math.sin(h)])


@dataclass
class Camera:
  x: float
  y: float
  ground: float  # the game's ground under (x, y), which the camera is `height` above
  height: float
  heading: float
  fov: float
  w: int  # image size in pixels
  h: int
  zmap: float = float("nan")  # the map's road height at (x, y): map heights move by ground - zmap

  def __post_init__(self):
    self.c = np.array([self.x, self.y])
    self.zcam = self.ground + self.height
    self.up, self.right = axes(self.heading)
    self.f = (self.h / 2) / math.tan(math.radians(self.fov) / 2)

  @property
  def m_per_px(self) -> float:
    return self.height / self.f

  def scaled(self, w: int, h: int) -> "Camera":
    return Camera(self.x, self.y, self.ground, self.height, self.heading, self.fov, w, h, self.zmap)

  def game_z(self, z: np.ndarray) -> np.ndarray:
    """Map heights as the game's: moved by the game's ground under the camera less the map's height there."""
    return z if math.isnan(self.zmap) else z + (self.ground - self.zmap)

  def project(self, pts: np.ndarray) -> np.ndarray:
    """[N, 3] map points (x, y, map z) -> [N, 2] pixels (u right, v down)."""
    pts = np.asarray(pts, np.float64)
    d = np.maximum(self.zcam - self.game_z(pts[:, 2]), 0.5)
    rel = pts[:, :2] - self.c
    return np.column_stack([self.w / 2 + self.f * (rel @ self.right) / d, self.h / 2 - self.f * (rel @ self.up) / d])

  def unproject(self, uv: np.ndarray, z: float | None = None) -> np.ndarray:
    """[N, 2] pixels -> [N, 2] world xy on the level plane at map height z (default: the road's at the centre)."""
    uv = np.atleast_2d(np.asarray(uv, np.float64))
    zz = self.game_z(np.array(z if z is not None else (self.zmap if not math.isnan(self.zmap) else self.ground)))
    d = self.zcam - zz
    return self.c + np.outer((uv[:, 0] - self.w / 2) * d / self.f, self.right) + np.outer((self.h / 2 - uv[:, 1]) * d / self.f, self.up)

  def footprint_radius(self) -> float:
    """m from the centre to the image's corners at the ground."""
    return math.hypot(self.w, self.h) / 2 * self.m_per_px

  def to_json(self) -> dict:
    return {"x": self.x, "y": self.y, "ground": self.ground, "height": self.height, "heading": self.heading, "fov": self.fov,
            "w": self.w, "h": self.h, "zmap": None if math.isnan(self.zmap) else self.zmap}

  @staticmethod
  def from_json(d: dict) -> "Camera":
    return Camera(d["x"], d["y"], d["ground"], d["height"], d["heading"], d["fov"], d["w"], d["h"],
                  float("nan") if d.get("zmap") is None else d["zmap"])


def file_hash(path: str, n: int = 12) -> str:
  with open(path, "rb") as f:
    return hashlib.file_digest(f, "blake2b").hexdigest()[:n]


def marks_path(map_dir: str = MAP_DIR) -> str:
  """The overlay's cache of the map's road marks (gta5_overlay.marks_key) for the map in map_dir, built into our own
  cache when the overlay's has none: building into the overlay's would push out the bridge's."""
  from types import SimpleNamespace
  from openpilot.tools.sim.bridge.gta5 import gta5_overlay as ov
  paths, osm = os.path.join(map_dir, "paths.jsonl"), os.path.join(map_dir, "gta5.osm.pbf")
  key = ov.marks_key(SimpleNamespace(path=paths), ov.OsmFile(osm, True, True))
  if key is None:
    raise RuntimeError(f"no marks key for {map_dir}")
  if os.path.exists(ov.marks_file(key)):
    return ov.marks_file(key)
  own = os.path.expanduser("~/.cache/gta5_imgcheck")
  path = os.path.join(own, f"marks-{key}.npz")
  if not os.path.exists(path):
    import subprocess
    import sys
    print(f"imgcheck: building the road marks for {map_dir} (about a minute)", flush=True)
    env = {**ov.child_env(), "GTA5_OVERLAY_CACHE": own}
    subprocess.run([sys.executable, "-m", "openpilot.tools.sim.bridge.gta5.gta5_overlay", "cache", paths, osm, "--key", key],
                   env=env, check=True)
  return path


def unpack_heights(h, n: int) -> np.ndarray:
  return np.full(n, float(h)) if not isinstance(h, list) else np.asarray(h, np.float64)


class MapData:
  """The map's marks and road surfaces, indexed by place for a tile's surroundings."""

  def __init__(self, map_dir: str = MAP_DIR, marks_file: str | None = None):
    self.map_dir = map_dir
    self.marks_file = marks_file or marks_path(map_dir)
    self.map_hash = file_hash(os.path.join(map_dir, "gta5.osm.pbf"))
    with np.load(self.marks_file, allow_pickle=False) as z:
      segs, kinds = z["segs"], z["kinds"]
      shape_pts, shape_len, shape_kind = z["shape_pts"], z["shape_len"], z["shape_kind"]
    self.segs = segs.astype(np.float32)
    self.kinds = kinds
    self.shapes = np.split(shape_pts.astype(np.float32), np.cumsum(shape_len)[:-1]) if len(shape_len) else []
    self.shape_kind = shape_kind
    self.seg_cells = self._index(segs.mean(axis=1)[:, :2])
    self.shape_cells = self._index(np.array([s[:, :2].mean(0) for s in self.shapes]) if self.shapes else np.zeros((0, 2)))
    self._load_roads()

  @staticmethod
  def _index(xy: np.ndarray) -> dict:
    cell = np.floor(xy / CELL).astype(np.int64)
    order = np.lexsort((cell[:, 1], cell[:, 0]))
    out = {}
    if not len(order):
      return out
    keys, starts = np.unique(cell[order], axis=0, return_index=True)
    for (cx, cy), s, e in zip(keys, starts, [*starts[1:], len(order)], strict=True):
      out[(int(cx), int(cy))] = order[s:e]
    return out

  @staticmethod
  def _near(cells: dict, c: np.ndarray, r: float) -> np.ndarray:
    (x0, y0), (x1, y1) = np.floor((c - r - CELL) / CELL).astype(int), np.floor((c + r + CELL) / CELL).astype(int)
    found = [cells[(cx, cy)] for cx in range(x0, x1 + 1) for cy in range(y0, y1 + 1) if (cx, cy) in cells]
    return np.concatenate(found) if found else np.zeros(0, np.int64)

  def _load_roads(self):
    """roads.json's road surfaces as segments: each way's middle cut into pieces with its width and height, and the
    junctions' areas as polygons with theirs; and its ways' middles as polylines for planning tiles."""
    with open(os.path.join(self.map_dir, "roads.json")) as f:
      r = json.load(f)
    self.classes = r["classes"]
    a, b, width, cls, structure = [], [], [], [], []
    self.ways = []  # (class, lanes, oneway, [N, 3] points, width, structure)
    for k, way in enumerate(r["ways"]):
      pts = np.asarray(way[3:], np.float64).reshape(-1, 2)
      z = unpack_heights(r["heights"][k], len(pts)) if "heights" in r else np.zeros(len(pts))
      p3 = np.column_stack([pts, z])
      st = int(r["levels"][2 * k + 1])
      self.ways.append((int(way[0]), int(way[1]), int(way[2]), p3, float(r["widths"][k]), st))
      a.append(p3[:-1])
      b.append(p3[1:])
      n = len(pts) - 1
      width += [float(r["widths"][k])] * n
      cls += [int(way[0])] * n
      structure += [st] * n
    self.road_a, self.road_b = np.concatenate(a), np.concatenate(b)
    self.road_w, self.road_cls, self.road_st = np.array(width), np.array(cls), np.array(structure)
    self.road_cells = self._index((self.road_a[:, :2] + self.road_b[:, :2]) / 2)
    self.areas = []  # (polygon [N, 2], z)
    for k, j in enumerate(r["junctions"]):
      poly = np.asarray(j[4:], np.float64).reshape(-1, 2)
      z = float(r["junction_heights"][k]) if "junction_heights" in r else 0.0
      self.areas.append((np.vstack([np.asarray(j[2:4]), poly]), z))  # centre first: fanned from it
    self.area_cells = self._index(np.array([p[0] for p, _ in self.areas]) if self.areas else np.zeros((0, 2)))

  def road_z(self, xy, z_hint: float | None = None, radius: float = 12.0) -> float | None:
    """The map's road height nearest xy (the level nearest z_hint where given): from its marks, which follow GTA's
    node heights."""
    xy = np.asarray(xy, np.float64)
    idx = self._near(self.seg_cells, xy, radius)
    if not len(idx):
      return None
    mid = self.segs[idx].mean(axis=1)
    d = np.hypot(mid[:, 0] - xy[0], mid[:, 1] - xy[1])
    sel = d < radius
    if z_hint is not None:
      sel &= np.abs(mid[:, 2] - z_hint) < LEVEL
    if not sel.any():
      return None
    w = 1.0 / (d[sel] + 1.0) ** 2
    return float((mid[sel, 2] * w).sum() / w.sum())

  def tile(self, cam: Camera, margin: float = 10.0) -> "TileMap":
    r = cam.footprint_radius() + margin
    idx = self._near(self.seg_cells, cam.c, r)
    segs, kinds = self.segs[idx], self.kinds[idx]
    if len(idx):
      mid = segs.mean(axis=1)
      keep = np.hypot(mid[:, 0] - cam.x, mid[:, 1] - cam.y) < r
      segs, kinds = segs[keep], kinds[keep]
    shapes = [(str(self.shape_kind[k]), self.shapes[k]) for k in self._near(self.shape_cells, cam.c, r).tolist()
              if np.hypot(*(self.shapes[k][:, :2].mean(0) - cam.c)) < r]
    ridx = self._near(self.road_cells, cam.c, r)
    areas = [self.areas[k] for k in self._near(self.area_cells, cam.c, r + 40).tolist()]
    return TileMap(cam, segs, kinds, shapes, self.road_a[ridx], self.road_b[ridx], self.road_w[ridx], self.road_st[ridx], areas)


@dataclass
class TileMap:
  """The map around one tile, and what it draws in the tile's image."""
  cam: Camera
  segs: np.ndarray  # [M, 2, 3]
  kinds: np.ndarray  # [M]
  shapes: list  # [(kind, [N, 3])]
  road_a: np.ndarray
  road_b: np.ndarray
  road_w: np.ndarray
  road_st: np.ndarray
  areas: list  # [(polygon [N, 2] with the centre first, z)]
  level: float = field(init=False)

  def __post_init__(self):
    z = self.cam.zmap if not math.isnan(self.cam.zmap) else self.cam.ground
    self.level = z

  def on_level(self, z: np.ndarray) -> np.ndarray:
    return np.abs(z - self.level) < LEVEL

  def lines(self, kinds: str, level_only: bool = True) -> list[np.ndarray]:
    """The polylines [N, 3] of these kinds (segments chained where they meet), on the tile's level."""
    from openpilot.tools.sim.bridge.gta5.gta5_overlay import chain
    sel = np.isin(self.kinds, list(kinds))
    if level_only:
      sel &= self.on_level(self.segs[:, :, 2].mean(axis=1))
    out = []
    for k in sorted(set(self.kinds[sel].tolist())):
      s = self.segs[sel & (self.kinds == k)].astype(np.float64)
      out += [(k, line) for line in chain(s)] if len(s) else []
    for k, pts in self.shapes:
      if k in kinds and (not level_only or self.on_level(pts[:, 2]).all()):
        out.append((k, pts.astype(np.float64)))
    return out

  def draw(self, size: tuple[int, int], items, width_m: float, fill=255, mode: str = "L", base=None) -> Image.Image:
    """Polylines drawn `width_m` wide (rounded joins) into a new image of the tile's size."""
    im = base if base is not None else Image.new(mode, size, 0)
    dr = ImageDraw.Draw(im)
    wpx = max(1, int(round(width_m / self.cam.m_per_px)))
    for _, pts in items:
      uv = self.cam.project(pts)
      if len(uv) < 2:
        continue
      dr.line([tuple(p) for p in uv], fill=fill, width=wpx, joint="curve")
      if wpx > 2:
        r = wpx / 2
        for p in (uv[0], uv[-1]):
          dr.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=fill)
    return im

  def surface(self, size: tuple[int, int], grow: float = 0.0, level_only: bool = True) -> np.ndarray:
    """The map's road surface (ways at their width, junction areas) in the image, grown by `grow` m: bool [h, w]."""
    im = Image.new("L", size, 0)
    dr = ImageDraw.Draw(im)
    mpp = self.cam.m_per_px
    for a, b, w, st in zip(self.road_a, self.road_b, self.road_w, self.road_st, strict=True):
      z = (a[2] + b[2]) / 2
      if (level_only and abs(z - self.level) >= LEVEL) or st == 2:
        continue
      uv = self.cam.project(np.array([a, b]))
      wpx = max(1, int(round((w + 2 * grow) / mpp)))
      dr.line([tuple(uv[0]), tuple(uv[1])], fill=255, width=wpx)
      r = wpx / 2
      for p in uv:
        dr.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=255)
    for poly, z in self.areas:
      if level_only and abs(z - self.level) >= LEVEL:
        continue
      uv = self.cam.project(np.column_stack([poly, np.full(len(poly), z)]))
      for i in range(1, len(uv)):  # fanned from the centre: the outline can fold back on itself
        dr.polygon([tuple(uv[0]), tuple(uv[i]), tuple(uv[i % (len(uv) - 1) + 1])], fill=255)
      if grow > 0:
        dr.line([tuple(p) for p in uv[1:]] + [tuple(uv[1])], fill=255, width=max(1, int(round(2 * grow / mpp))))
    return np.asarray(im) > 0

  def overhead(self, size: tuple[int, int]) -> np.ndarray:
    """The highest map road surface over each pixel, in map metres (-inf where there is none): float32 [h, w]."""
    im = Image.new("F", size, -1e9)
    dr = ImageDraw.Draw(im)
    mpp = self.cam.m_per_px
    z = (self.road_a[:, 2] + self.road_b[:, 2]) / 2
    for k in np.argsort(z):
      if self.road_st[k] == 2 or z[k] < self.level + ABOVE - 0.5:
        continue
      uv = self.cam.project(np.array([self.road_a[k], self.road_b[k]]))
      wpx = max(1, int(round((self.road_w[k] + 1.0) / mpp)))
      dr.line([tuple(uv[0]), tuple(uv[1])], fill=float(z[k]), width=wpx)
    for poly, zz in sorted(self.areas, key=lambda a: a[1]):
      if zz < self.level + ABOVE - 0.5:
        continue
      uv = self.cam.project(np.column_stack([poly, np.full(len(poly), zz)]))
      for i in range(1, len(uv)):
        dr.polygon([tuple(uv[0]), tuple(uv[i]), tuple(uv[i % (len(uv) - 1) + 1])], fill=float(zz))
    return np.asarray(im, np.float32)


def load_image(path: str) -> np.ndarray:
  return np.asarray(Image.open(path).convert("RGB"))


def find_marks_cache() -> list[str]:
  return sorted(glob.glob(os.path.expanduser("~/.cache/gta5_overlay/marks-*.npz")), key=os.path.getmtime)
