#!/usr/bin/env python3
"""Tests of the image check on synthetic tiles (no game, no map files): the projection's convention, paint found by
contrast and colour, and each kind of issue on a drawn road with a made-up map. Run as a script."""
import math

import numpy as np
from PIL import Image, ImageDraw

from openpilot.tools.sim.bridge.gta5.map.imgcheck import compare, report, tiles
from openpilot.tools.sim.bridge.gta5.map.imgcheck.frame import Camera, TileMap, axes, heading_of
from openpilot.tools.sim.bridge.gta5.map.imgcheck.paint import Paint, box_sum

W, H = 1280, 720  # the analysed size: a 2560x1440 grab halved
GROUND = 10.0


def cam(heading=0.0, x=0.0, y=0.0):
  return Camera(x, y, GROUND, 45.0, heading, 50.0, W, H, GROUND)


def test_projection():
  c = cam()
  assert abs(c.m_per_px - 2 * 45 * math.tan(math.radians(25)) / H) < 1e-9
  # heading 0: the image's up is north (+y), right is east (+x)
  uv = c.project(np.array([[0, 10, GROUND], [10, 0, GROUND]]))
  assert abs(uv[0, 0] - W / 2) < 1e-6 and uv[0, 1] < H / 2
  assert uv[1, 0] > W / 2 and abs(uv[1, 1] - H / 2) < 1e-6
  # heading 90 (GTA: counterclockwise, west): up is west
  up, right = axes(90.0)
  assert np.allclose(up, [-1, 0]) and np.allclose(right, [0, 1])
  assert abs(heading_of(-1, 0) - 90) < 1e-9
  # a point above the ground sits further from the middle; unproject inverts project on the level
  c2 = cam(37.0, 100, 200)
  p = np.array([[107.0, 189.0, GROUND]])
  assert np.allclose(c2.unproject(c2.project(p))[0], p[0, :2], atol=1e-6)
  hi = c2.project(np.array([[107.0, 189.0, GROUND + 5]]))
  assert np.hypot(*(hi[0] - [W / 2, H / 2])) > np.hypot(*(c2.project(p)[0] - [W / 2, H / 2]))
  # map heights move by the game's ground less the map's at the camera
  c3 = Camera(0, 0, GROUND + 1.0, 45.0, 0.0, 50.0, W, H, GROUND)
  plain = Camera(0, 0, GROUND + 1.0, 45.0, 0.0, 50.0, W, H)
  assert np.allclose(c3.project(np.array([[5, 5, GROUND]])), plain.project(np.array([[5, 5, GROUND + 1.0]])))
  # a tile's long side runs along its road
  along, across = tiles.footprint()
  assert along > across and abs(across - 2 * 45 * math.tan(math.radians(25))) < 1e-9


def road_image(c: Camera, lines: list[tuple[np.ndarray, tuple, float]], dashed: set[int] = frozenset(), seed=0) -> np.ndarray:
  """Grey asphalt with grain, pavement beyond y = +-8 m, and painted lines (world polylines, colour, width m)."""
  rng = np.random.default_rng(seed)
  img = np.full((H, W, 3), 80.0) + rng.normal(0, 6, (H, W, 1))
  xy = c.unproject(np.stack(np.meshgrid(np.arange(W), np.arange(H)), -1).reshape(-1, 2)).reshape(H, W, 2)
  img[np.abs(xy[..., 1]) > 8.0] = (170, 170, 165)
  im = Image.fromarray(img.clip(0, 255).astype(np.uint8))
  dr = ImageDraw.Draw(im)
  for k, (pts, col, width) in enumerate(lines):
    if k in dashed:
      for x0 in np.arange(-60, 60, 9.0):
        seg = np.array([[x0, pts[0, 1], GROUND], [x0 + 3, pts[0, 1], GROUND]])
        dr.line([tuple(p) for p in c.project(seg)], fill=col, width=max(1, int(width / c.m_per_px)))
    else:
      dr.line([tuple(p) for p in c.project(pts)], fill=col, width=max(1, int(width / c.m_per_px)))
  return np.asarray(im)


def line(y, z=GROUND):
  return np.array([[-60.0, y, z], [60.0, y, z]])


def tile_map(c: Camera, marks: list[tuple[str, np.ndarray]], shapes=()) -> TileMap:
  segs = np.concatenate([np.stack([p[:-1], p[1:]], axis=1) for _, p in marks]) if marks else np.zeros((0, 2, 3))
  kinds = np.concatenate([[k] * (len(p) - 1) for k, p in marks]) if marks else np.zeros(0, "<U1")
  a = np.array([[-60.0, 0.0, GROUND]])
  b = np.array([[60.0, 0.0, GROUND]])
  return TileMap(c, segs.astype(np.float32), kinds, list(shapes), a, b, np.array([16.0]), np.array([0]), [])


class FakeMap:
  def __init__(self, marks, shapes=()):
    self.marks, self.shapes = marks, shapes
    self.map_hash = "test"

  def tile(self, c):
    return tile_map(c, self.marks, self.shapes)


YELLOW, WHITE = (230, 190, 40), (235, 235, 235)


def test_paint():
  c = cam()
  img = road_image(c, [(line(0.15), YELLOW, 0.12), (line(-0.15), YELLOW, 0.12), (line(4.0), WHITE, 0.15)])
  p = Paint(img.astype(np.float32), c.m_per_px)
  u = W // 2
  rows = {round(float(c.unproject(np.array([[u, v]]))[0, 1]), 1): v for v in range(H)}
  col = p.yellow[:, u] | p.white[:, u]
  ys = c.unproject(np.column_stack([np.full(H, u), np.arange(H)]))[:, 1]
  assert p.yellow[:, u][np.abs(ys - 0.15) < 0.04].any() and p.yellow[:, u][np.abs(ys + 0.15) < 0.04].any()
  assert p.white[:, u][np.abs(ys - 4.0) < 0.05].any()
  assert not col[(np.abs(ys) > 0.4) & (np.abs(ys - 4.0) > 0.3) & (np.abs(ys) < 7.5)].any(), "grain or asphalt read as paint"
  assert not p.yellow[:, u][np.abs(ys - 4.0) < 0.1].any()
  assert rows and box_sum(np.ones((5, 5), bool), 3)[2, 2] == 9


def run(img, c, marks, shapes=()):
  tc = compare.TileCheck(img, Camera(c.x, c.y, c.ground, c.height, c.heading, c.fov, img.shape[1], img.shape[0], c.zmap),
                         FakeMap(marks, shapes), scale=1)
  tc.run()
  return tc


def kinds(tc):
  return sorted({i.kind for i in tc.issues})


def test_clean():
  c = cam()
  paint = [(line(0.15), YELLOW, 0.12), (line(-0.15), YELLOW, 0.12), (line(4.0), WHITE, 0.15), (line(-4.0), WHITE, 0.15)]
  img = road_image(c, paint, dashed={2, 3})
  marks = [("c", line(0.15)), ("c", line(-0.15)), ("d", line(4.0)), ("d", line(-4.0)), ("e", line(8.0)), ("e", line(-8.0))]
  tc = run(img, c, marks)
  assert not tc.issues, [i.detail for i in tc.issues]


def test_missing_and_stray():
  c = cam()
  img = road_image(c, [(line(0.15), YELLOW, 0.12), (line(-0.15), YELLOW, 0.12), (line(4.0), WHITE, 0.15)])
  # the map has the white line 1.5 m off the paint, and no other
  marks = [("c", line(0.15)), ("c", line(-0.15)), ("w", line(5.5)), ("e", line(8.0)), ("e", line(-8.0))]
  tc = run(img, c, marks)
  assert "missing" in kinds(tc) and "stray" in kinds(tc), [i.detail for i in tc.issues]
  miss = [i for i in tc.issues if i.kind == "missing"]
  assert all(abs(i.y - 4.0) < 0.5 and i.colour == "white" for i in miss)
  stray = [i for i in tc.issues if i.kind == "stray"]
  assert all(abs(i.y - 5.5) < 0.5 for i in stray)
  # spots: the two together are an offset
  res = [{"name": "a", "camera": tc.cam.to_json(), "summary": {"unloaded": False}, "issues": [i.to_json() for i in tc.issues]}]
  sp = report.spots(res)
  assert sp[0]["type"] == "offset", sp[0]


def test_colour():
  c = cam()
  img = road_image(c, [(line(0.0), WHITE, 0.15)])
  tc = run(img, c, [("c", line(0.0)), ("e", line(8.0)), ("e", line(-8.0))])
  assert kinds(tc) == ["colour"], [i.detail for i in tc.issues]


def test_kerb_on_asphalt():
  c = cam()
  img = road_image(c, [])
  tc = run(img, c, [("e", line(3.0)), ("e", line(8.0)), ("e", line(-8.0))])
  bad = [i for i in tc.issues if i.kind == "kerb"]
  assert bad and all(abs(i.y - 3.0) < 0.3 for i in bad), [i.detail for i in tc.issues]


def test_stop_line():
  c = cam()
  bar = np.array([[10.0, -7.5, GROUND], [10.0, 0.0, GROUND]])
  img = road_image(c, [(bar, WHITE, 0.4)])
  ok = run(img, c, [("e", line(8.0)), ("e", line(-8.0))], [("l", bar)])
  assert "stop" not in kinds(ok), [i.detail for i in ok.issues]
  off = run(img, c, [("e", line(8.0)), ("e", line(-8.0))], [("l", bar + [2.5, 0, 0])])
  stops = [i for i in off.issues if i.kind == "stop"]
  assert stops and abs(float(stops[0].detail.split(" m off")[0].split()[-1]) - 2.5) <= 0.2, [i.detail for i in off.issues]


def test_plan_order():
  rows = [[tiles.Tile("", x, 0.0, 0.0, 90.0, "primary") for x in (0, 55, 110)],
          [tiles.Tile("", x, 500.0, 0.0, 90.0, "primary") for x in (110, 55, 0)],
          [tiles.Tile("", 2, 3, 0.0, 270.0, "residential")]]
  kept = tiles.dedup(rows)
  assert sum(len(r) for r in kept) == 6  # the last duplicates the first road's first tile, reversed
  out = tiles.order(kept, (0, 0))
  assert [t.name for t in out] == [f"t{n:05d}" for n in range(6)]
  assert out[3].hop == 500.0 and out[0].hop == 0.0  # along the first road, then to the second road's near end
  assert tiles.wait_for(50) < tiles.wait_for(300) < tiles.wait_for(2000)


def test_scene_check():
  from openpilot.tools.sim.bridge.gta5.map.imgcheck.capture import scene_ok
  good = {"world": {"hour": 13, "weather": "EXTRASUNNY", "from": "EXTRASUNNY", "to": "EXTRASUNNY", "mix": 0, "frozen": True, "rain": 0},
          "density": {"set": True, "vehicles": 0, "peds": 0, "parked": 0}}
  assert scene_ok(good)[0]
  assert not scene_ok({**good, "world": {**good["world"], "hour": 21}})[0]
  assert not scene_ok({**good, "density": {"set": True, "vehicles": 1, "peds": 0, "parked": 0}})[0]


class FakeGame:
  """The plugin as the capture runner sees it: commands in, state out, grabs written as BMPs."""

  def __init__(self, blank_first: set):
    self.cmds, self.topcam, self.blank_first = [], {"on": False}, set(blank_first)
    self.world = {"hour": 19, "weather": "RAIN", "from": "RAIN", "to": "RAIN", "mix": 0, "frozen": False, "rain": 0.5}
    self.density = {"set": False, "off": False, "vehicles": 1, "peds": 1, "parked": 1}

  def send(self, *cmds):
    for c in cmds:
      self.cmds.append(c)
      if c["type"] == "topcam":
        self.topcam = {"on": bool(c.get("on")), "x": round(c.get("x", 0), 3), "y": round(c.get("y", 0), 3), "ground": 10.0,
                       "height": c.get("height", 45), "heading": c.get("heading", 0), "fov": c.get("fov", 50)}
      elif c["type"] == "world" and "weather" in c:
        self.world = {"hour": c["hour"], "weather": c["weather"], "from": c["weather"], "to": c["weather"], "mix": 0,
                      "frozen": bool(c.get("freeze")), "rain": 0}
      elif c["type"] == "traffic" and "vehicles" in c:
        self.density = {"set": True, "off": not c.get("on", 1), "vehicles": c["vehicles"], "peds": c["peds"], "parked": c["parked"]}
      elif c["type"] == "grab":
        path = c["path"].replace("\\", "/")
        name = path.rsplit("/", 1)[-1][:-4]
        rng = np.random.default_rng(len(self.cmds))
        img = np.full((144, 256, 3), 90, np.uint8) if name in self.blank_first else \
          rng.integers(0, 255, (144, 256, 3)).astype(np.uint8)
        self.blank_first.discard(name)
        Image.fromarray(img).save(path, format="BMP")

  def read_state(self, path=None, wait=2.0):
    return {"topcam": dict(self.topcam), "world": dict(self.world), "density": dict(self.density), "pos": [0, 0, 0]}


def test_capture_runner():
  import json
  import os
  import tempfile
  from openpilot.tools.sim.bridge.gta5.map.imgcheck import capture
  plan = [tiles.Tile(f"t{n:05d}", 10.0 * n, 0.0, 10.0, 90.0, "primary", 10.0) for n in range(4)]
  game = FakeGame({"t00002"})
  saved = (capture.gta5_cmd.send, capture.gta5_cmd.read_state, capture.win_path, capture.RETRY_WAITS, capture.time.sleep)
  capture.gta5_cmd.send, capture.gta5_cmd.read_state, capture.win_path = game.send, game.read_state, lambda p: p
  capture.RETRY_WAITS, capture.time.sleep = (0.0,), lambda s: None
  try:
    with tempfile.TemporaryDirectory() as d:
      out = capture.run(d, plan, log=lambda *a: None)
      assert out["shot"] == 4 and not out["failed"] and not out["still_unloaded"], out
      side = json.load(open(os.path.join(d, "tiles", "t00002.json")))
      assert side["scene"]["world"]["weather"] == "EXTRASUNNY" and side["scene"]["density"]["vehicles"] == 0
      assert side["quality"]["sharp"] > 6 and os.path.exists(os.path.join(d, "tiles", side["image"]))
      grabs = [c for c in game.cmds if c["type"] == "grab"]
      assert len(grabs) == 5  # t00002 shot again after it came out blank
      assert game.cmds[-1]["type"] == "traffic" and game.cmds[-2]["type"] == "world"  # the scene put back
      assert any(c["type"] == "topcam" and not c.get("on") for c in game.cmds[-3:])
      assert capture.run(d, plan, log=lambda *a: None)["shot"] == 0  # resumed: nothing left
  finally:
    capture.gta5_cmd.send, capture.gta5_cmd.read_state, capture.win_path, capture.RETRY_WAITS, capture.time.sleep = saved


if __name__ == "__main__":
  for name, fn in list(globals().items()):
    if name.startswith("test_") and callable(fn):
      fn()
      print(f"{name} ok")
