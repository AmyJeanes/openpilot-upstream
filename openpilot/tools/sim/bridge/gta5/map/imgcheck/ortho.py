#!/usr/bin/env python3
"""Orthographic top-down tiles of the map: the plugin's topcam over each cell of a grid, rendered orthographically
(core.cpp `ortho op=box`, ortho.cpp), each grab saved twice: the G-buffer's albedo (flat, unlit: the map texture) and
the lit frame.

A tile is 2560x1440 px over W x H m (3 cm a pixel: 76.8 x 43.2 m), north up; the grid steps 90% of that, so tiles
overlap by 10%. Its pixel of a world point is exact and linear (the projection has no perspective): the sidecar's
world_to_px. Tiles are named by grid cell (i east, j north of the grid's origin), saved as PNG, and skipped when their
sidecar exists, so a run resumes.

The capture needs the game with the plugin, connected to the bridge or to orthod.py with openpilot's frames off: an
openpilot frame between the player's would render from the car with the box's projection. Before the first tile the
scene is set (noon, sunny, the clock frozen, no traffic, no cast shadows) and the ortho box and the albedo grab source
are found (memory searches of a few seconds); a core reload or a reconnect loses them, which the state shows, and they
are set again.

  python ortho.py plan <out.json> [--area x0 y0 x1 y1] [--roads tiles.json] [--water water_mask.txt]
  python ortho.py run <run_dir> <plan.json> [--minutes N] [--stop-free-gb 30]
"""
import argparse
import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time

import numpy as np
from PIL import Image

from openpilot.tools.sim.bridge.gta5 import gta5_cmd

W_PX, H_PX = 2560, 1440
CM_PER_PX = 3.0
W_M, H_M = W_PX * CM_PER_PX / 100, H_PX * CM_PER_PX / 100  # 76.8 x 43.2 m
STEP_X, STEP_Y = 0.9 * W_M, 0.9 * H_M  # 10% overlap: 2304 and 1296 px
GRID_X0, GRID_Y0 = -4300.0, -4300.0  # the grid's origin, the centre of cell (0, 0)
FOV = 37.3  # the topcam's vertical field: the box replaces the projection of the viewport with this one
HEIGHT = 45.0  # m above the highest surface in the tile (LOD and streaming are the camera's)
NEAR, FAR = -1500.0, 2500.0  # m from the camera: up to 1500 m above it is drawn too, whatever its height
PARK_ABOVE = 60.0  # m: the hidden car is held this far above the ground last found, so the collision there streams
SCENE_WORLD = {"type": "world", "hour": 13, "minute": 0, "weather": "EXTRASUNNY", "freeze": 1, "rain": 0}
SCENE_TRAFFIC = {"type": "traffic", "on": 0, "vehicles": 0, "peds": 0, "parked": 0}
SCENE_SHADOWS = {"type": "shadows", "bounds": 0.0}
SCENE_OFF = [{"type": "debug", "on": 0}, {"type": "gpsroute", "on": 0}, {"type": "pins", "p": ""}]
GRAB_WAIT = 6.0
STABLE = 1.5  # mean abs diff (0-255) of two grabs a moment apart, under which the scene has finished streaming
RECONNECT_WAIT = 300.0


def cell_centre(i: int, j: int) -> tuple[float, float]:
  return GRID_X0 + i * STEP_X, GRID_Y0 + j * STEP_Y


def world_to_px(x: float, y: float) -> list[list[float]]:
  """The tile's affine map from world (x, y, 1) to pixel (column, row), pixels' edges at integers."""
  sx, sy = W_PX / W_M, H_PX / H_M
  return [[sx, 0.0, W_PX / 2 - sx * x], [0.0, -sy, H_PX / 2 + sy * y]]


def win_path(path: str) -> str:
  return subprocess.run(["wslpath", "-w", path], capture_output=True, text=True, check=True).stdout.strip()


# -------------------------------------------------------------------------------------------------------------- plans
def cells_near(points: np.ndarray, radius: float) -> set[tuple[int, int]]:
  out = set()
  ri, rj = int(math.ceil(radius / STEP_X)) + 1, int(math.ceil(radius / STEP_Y)) + 1
  for x, y in points:
    ci, cj = round((x - GRID_X0) / STEP_X), round((y - GRID_Y0) / STEP_Y)
    for i in range(ci - ri, ci + ri + 1):
      for j in range(cj - rj, cj + rj + 1):
        cx, cy = cell_centre(i, j)
        if abs(cx - x) <= W_M / 2 + radius and abs(cy - y) <= H_M / 2 + radius:
          out.add((i, j))
  return out


def snake(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
  """Rows south to north, alternating direction, so each hop is one cell."""
  rows: dict[int, list[int]] = {}
  for i, j in cells:
    rows.setdefault(j, []).append(i)
  out = []
  for k, j in enumerate(sorted(rows)):
    out += [(i, j) for i in sorted(rows[j], reverse=bool(k % 2))]
  return out


def blocks(cells: set[tuple[int, int]], size: int = 12) -> list[tuple[int, int]]:
  """Cells in blocks of size x size cells (about 830 x 470 m), the blocks in a snake, each block's cells in a snake:
  short hops, and the streamed area moves on slowly."""
  by: dict[tuple[int, int], list[tuple[int, int]]] = {}
  for i, j in cells:
    by.setdefault((i // size, j // size), []).append((i, j))
  out = []
  for b in snake(list(by)):
    out += snake(by[b])
  return out


def plan(args) -> None:
  cells: list[tuple[int, int]] = []
  seen: set[tuple[int, int]] = set()
  groups = []
  if args.centres:  # a pilot: n x n cells around each centre
    for k in range(0, len(args.centres), 2):
      x, y = args.centres[k], args.centres[k + 1]
      ci, cj = round((x - GRID_X0) / STEP_X), round((y - GRID_Y0) / STEP_Y)
      h = args.n // 2
      groups.append(snake([(i, j) for i in range(ci - h, ci - h + args.n) for j in range(cj - h, cj - h + args.n)]))
  else:
    water = None
    if args.water:
      water = [row for row in open(args.water).read().split()]
    if args.roads:
      idx = json.load(open(args.roads))
      for part in ("plan_city", "plan_rest"):  # roads first: the perspective capture's tiles, city then the rest
        pts = np.array([[v["x"], v["y"]] for k, v in idx.items() if k.startswith(part + "/")])
        if len(pts):
          groups.append(blocks(cells_near(pts, 10.0) - seen))
          seen |= set(groups[-1])
    if water is not None:  # then the rest of the land: cells not all water
      land = {(i, j) for j, row in enumerate(water) for i, ch in enumerate(row) if ch != "9"}
      groups.append(blocks(land - seen))
  for g in groups:
    for c in g:
      if c not in seen or not args.roads:
        seen.add(c)
      cells.append(c)
  # duplicates (a pilot's blocks overlapping) once, in order
  out, done = [], set()
  for c in cells:
    if c not in done:
      done.add(c)
      out.append(c)
  tiles = []
  for i, j in out:
    x, y = cell_centre(i, j)
    tiles.append({"name": f"o{i:03d}_{j:03d}", "i": i, "j": j, "x": round(x, 3), "y": round(y, 3)})
  with open(args.out, "w") as f:
    json.dump({"cm_per_px": CM_PER_PX, "w_m": W_M, "h_m": H_M, "step": [STEP_X, STEP_Y], "origin": [GRID_X0, GRID_Y0],
               "groups": [len(g) for g in groups], "tiles": tiles}, f)
  print(f"{len(tiles)} tiles in groups {[len(g) for g in groups]} -> {args.out}")


# -------------------------------------------------------------------------------------------------------------- shots
def quality(albedo: np.ndarray) -> dict:
  """Unrendered (black) share of the albedo, and its sharpness (mean absolute Laplacian of the luma, half size)."""
  g = albedo.mean(axis=2)
  black = float((albedo.max(axis=2) < 3).mean())
  s = g[::2, ::2]
  lap = np.abs(4 * s[1:-1, 1:-1] - s[:-2, 1:-1] - s[2:, 1:-1] - s[1:-1, :-2] - s[1:-1, 2:])
  return {"black": round(black, 4), "sharp": round(float(lap.mean()), 2), "mean": round(float(g.mean()), 1)}


class Shooter:
  def __init__(self, run_dir: str, log=print):
    self.run_dir = run_dir
    self.tile_dir = os.path.join(run_dir, "tiles")
    self.raw_dir = os.path.join(run_dir, "raw")
    os.makedirs(self.tile_dir, exist_ok=True)
    os.makedirs(self.raw_dir, exist_ok=True)
    self.raw_win = win_path(self.raw_dir)
    self.log = log
    self.saved: queue.Queue = queue.Queue(maxsize=8)
    self.results: dict[str, dict] = {}
    self.zguess = 50.0  # m: the ground last found
    self.park_z = 0.0
    self.workers = [threading.Thread(target=self._save_loop, daemon=True) for _ in range(3)]
    for w in self.workers:
      w.start()

  def send(self, *cmds: dict) -> None:
    end = time.monotonic() + RECONNECT_WAIT
    while True:
      try:
        return gta5_cmd.send(*cmds)
      except OSError as e:
        if time.monotonic() > end:
          raise RuntimeError(f"ortho: the bridge has been unreachable for {RECONNECT_WAIT:.0f} s: {e}") from e
        time.sleep(3.0)

  def state(self, wait: float = 3.0) -> dict | None:
    try:
      return gta5_cmd.read_state(f"/tmp/ortho_state_{os.getpid()}.json", wait=wait)
    except OSError:
      return None

  def wait_state(self) -> dict:
    end = time.monotonic() + RECONNECT_WAIT
    while (st := self.state()) is None:
      if time.monotonic() > end:
        raise RuntimeError("ortho: no state from the plugin")
      time.sleep(3.0)
    return st

  def scene_ok(self, st: dict) -> tuple[bool, str]:
    w, d, o = st.get("world") or {}, st.get("density") or {}, st.get("ortho") or {}
    bad = []
    if w.get("hour") not in (12, 13, 14) or w.get("weather") != "EXTRASUNNY" or not w.get("frozen") or (w.get("rain") or 0) > 0.01:
      bad.append(f"world {w.get('hour')} {w.get('weather')} frozen {w.get('frozen')}")
    if not d.get("set") or any((d.get(k) or 0) > 0 for k in ("vehicles", "peds", "parked")):
      bad.append("traffic")
    sh = st.get("shadows")
    if sh is not None and not (sh.get("set") and sh.get("bounds") == 0):
      bad.append("shadows")
    if not o.get("on") or not o.get("redirected"):
      bad.append(f"ortho {o}")
    if not o.get("source"):
      bad.append("no albedo source")
    return not bad, ", ".join(bad)

  def setup(self, x: float, y: float) -> None:
    """The scene, the box (and its viewport search) and the albedo source (its search), until the state shows them."""
    for attempt in range(6):
      self.send(SCENE_WORLD, SCENE_TRAFFIC, SCENE_SHADOWS, *SCENE_OFF)
      self.send(self.topcam(x, y), {"type": "ortho", "op": "box", "on": 1, "fov": FOV, "w": W_M, "h": H_M, "near": NEAR, "far": FAR})
      st = self.wait_state()
      if not (st.get("ortho") or {}).get("source"):
        self.send({"type": "ortho", "op": "gbuffer"})
      time.sleep(4.0 + 4.0 * attempt)
      ok, why = self.scene_ok(self.wait_state())
      if ok:
        return
      self.log(f"ortho: setting up: {why}")
    raise RuntimeError(f"ortho: the scene or the box won't set: {why}")

  def topcam(self, x: float, y: float) -> dict:
    # over the highest surface in the tile, the LOD distances stretched to the lowest
    return {"type": "topcam", "on": 1, "x": x, "y": y, "height": HEIGHT, "fov": FOV, "heading": 0, "hud": 0, "dof": 0, "top": 1,
            "tw": W_M, "th": H_M, "lod": -1}

  def grab(self, name: str) -> tuple[str, str] | None:
    bmp, src = os.path.join(self.raw_dir, name + ".bmp"), os.path.join(self.raw_dir, name + "_src.bmp")
    for p in (bmp, src):
      if os.path.exists(p):
        os.remove(p)
    self.send({"type": "grab", "path": self.raw_win + "\\" + name + ".bmp"})
    end = time.monotonic() + GRAB_WAIT
    while time.monotonic() < end:
      if os.path.exists(bmp):  # renamed into place whole, after the source
        return (bmp, src) if os.path.exists(src) else None
      time.sleep(0.02)
    return None

  def park(self, x: float, y: float, z: float) -> None:
    self.send({"type": "park", "on": 1, "x": x, "y": y, "z": z}, self.topcam(x, y))
    self.park_z = z

  def probed(self, st: dict) -> bool:
    """Whether the topcam found the ground: else it stands at about the car's height."""
    return abs(st["topcam"]["ground"] - self.park_z) > 1.0

  def settled(self, name: str, tries: int = 6, gap: float = 1.5) -> tuple[str, str] | None:
    """A grab once the scene stops changing: after a long hop the game is still streaming (props pop in, the terrain's
    low-detail mesh gives way), so grabs are taken gap s apart until two agree (mean abs diff of the albedo at a
    sixteenth of its size under STABLE)."""
    prev, got = None, None
    for k in range(tries):
      got = self.grab(name)
      if got is None:
        return None
      small = np.asarray(Image.open(got[1]).convert("L").reduce(16), dtype=np.float64)  # 48 cm blocks: swaying trees average out
      if prev is not None and float(np.abs(small - prev).mean()) < STABLE:
        return got
      prev = small
      if k + 1 < tries:
        time.sleep(gap)
    self.log(f"ortho: {name}: still changing after {tries} grabs")
    return got

  def shoot(self, t: dict, wait: float) -> bool:
    # the car (hidden) goes with the camera: the game streams collision around the player only, and the camera's
    # height comes from the ground under it
    self.park(t["x"], t["y"], self.zguess + PARK_ABOVE)
    time.sleep(wait)
    st = self.state()
    if st is None or not isinstance(st.get("topcam"), dict):
      self.log(f"ortho: {t['name']}: no state")
      return False
    if not self.probed(st):
      for z in (self.zguess + 250, self.zguess - 150, 350, 650, 100, 900, 20):
        self.park(t["x"], t["y"], z)
        time.sleep(1.5)
        st = self.state() or st
        if self.probed(st):
          break
      else:
        self.log(f"ortho: {t['name']}: no ground found")
    if self.probed(st):
      self.zguess = st["topcam"]["ground"]
    tc = st["topcam"]
    if not tc.get("on") or abs(tc["x"] - t["x"]) > 0.01 or abs(tc["y"] - t["y"]) > 0.01:
      self.log(f"ortho: {t['name']}: topcam not there yet")
      return False
    ok, why = self.scene_ok(st)
    if not ok:
      self.log(f"ortho: {t['name']}: {why}; setting up again")
      self.setup(t["x"], t["y"])
      return False
    got = self.settled(t["name"]) if wait > 1.0 else self.grab(t["name"])
    if got is None:
      self.log(f"ortho: {t['name']}: no grab")
      return False
    side = {"name": t["name"], "cell": [t["i"], t["j"]], "x": t["x"], "y": t["y"], "cm_per_px": CM_PER_PX, "size": [W_PX, H_PX],
            "extent_m": [W_M, H_M], "world_to_px": world_to_px(t["x"], t["y"]), "north_up": True, "time": time.time(),  # noqa: TID251
            "wait": wait, "topcam": tc, "ground_found": self.probed(st), "world": st.get("world")}
    self.saved.put((got, side))
    return True

  def _save_loop(self):
    while True:
      (bmp, src), side = self.saved.get()
      try:
        name = side["name"]
        albedo = np.asarray(Image.open(src).convert("RGB"))
        Image.fromarray(albedo).save(os.path.join(self.tile_dir, name + ".png"), compress_level=2)
        Image.open(bmp).convert("RGB").save(os.path.join(self.tile_dir, name + "_lit.png"), compress_level=2)
        side["quality"] = quality(albedo)
        side["images"] = {"albedo": name + ".png", "lit": name + "_lit.png"}
        with open(os.path.join(self.tile_dir, name + ".json"), "w") as f:
          json.dump(side, f, indent=1)
        os.remove(bmp)
        os.remove(src)
        self.results[name] = side["quality"]
      except Exception as e:
        self.log(f"ortho: {side['name']}: save failed: {type(e).__name__}: {e}")
        self.results[side["name"]] = {"error": str(e)}
      finally:
        self.saved.task_done()

  def done(self, name: str) -> bool:
    return os.path.exists(os.path.join(self.tile_dir, name + ".json"))


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
  """The sub-pixel shift (x, y) between two same-size greyscale images, and the correlation peak (near 1: alike)."""
  a, b = a - a.mean(), b - b.mean()
  win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
  F = np.fft.fft2(a * win) * np.conj(np.fft.fft2(b * win))
  r = np.fft.ifft2(F / (np.abs(F) + 1e-9)).real
  py, px = np.unravel_index(np.argmax(r), r.shape)

  def sub(c, n, get):
    lo, m, hi = get((c - 1) % n), get(c), get((c + 1) % n)
    d = lo - 2 * m + hi
    return c + (0.5 * (lo - hi) / d if d else 0.0)
  sy, sx = sub(py, r.shape[0], lambda i: r[i, px]), sub(px, r.shape[1], lambda i: r[py, i])
  sy -= r.shape[0] if sy > r.shape[0] / 2 else 0
  sx -= r.shape[1] if sx > r.shape[1] / 2 else 0
  return -sx, -sy, float(r.max())


class Reference:
  """A tile shot again every so often, against its first shot: a setting the game dropped (a reload, a scene change)
  shows as a shifted or different image."""
  EVERY = 30 * 60.0  # s

  def __init__(self, sh: "Shooter", tile: dict, log):
    self.sh, self.tile, self.log = sh, dict(tile), log
    self.first: np.ndarray | None = None
    self.next = 0.0
    self.bad = 0

  def check(self) -> bool:
    """False after three bad checks in a row."""
    if time.monotonic() < self.next:
      return True
    self.next = time.monotonic() + self.EVERY
    name = "ref_" + time.strftime("%H%M%S")
    t = {**self.tile, "name": name}
    got = None
    for _ in range(2):
      self.sh.park(t["x"], t["y"], self.sh.zguess + PARK_ABOVE)
      time.sleep(6.0)
      got = self.sh.settled(name, tries=10)
      if got:
        break
    if not got:
      self.bad += 1
      self.log(f"ortho: reference {name}: no grab ({self.bad} bad)")
      return self.bad < 3
    bmp, src = got
    img = np.asarray(Image.open(src).convert("L"), dtype=np.float64)
    ref_dir = os.path.join(self.sh.run_dir, "reference")
    os.makedirs(ref_dir, exist_ok=True)
    Image.open(src).convert("RGB").save(os.path.join(ref_dir, name + ".png"), compress_level=2)
    os.remove(bmp)
    os.remove(src)
    if self.first is None:
      self.first = img
      self.log(f"ortho: reference {name} taken at {t['x']:.1f},{t['y']:.1f}")
      return True
    sx, sy, peak = phase_shift(self.first[200:1240, 300:2260], img[200:1240, 300:2260])
    mad = float(np.abs(self.first - img).mean())
    ok = abs(sx) < 0.5 and abs(sy) < 0.5 and mad < 8.0
    self.bad = 0 if ok else self.bad + 1
    self.log(f"ortho: reference {name}: shift {sx:.2f},{sy:.2f} px, peak {peak:.2f}, mean abs diff {mad:.1f}" + ("" if ok else f" BAD ({self.bad})"))
    if not ok:
      self.sh.setup(t["x"], t["y"])
    return self.bad < 3


def wait_for(hop: float) -> float:
  if hop < 100:
    return 0.7
  if hop < 600:
    return 2.5
  return 6.0


def free_gb(path: str) -> float:
  return shutil.disk_usage(path).free / 1e9


def run(args) -> None:
  p = json.load(open(args.plan))
  tiles = p["tiles"]
  log_path = os.path.join(args.run_dir, "capture.log")
  os.makedirs(args.run_dir, exist_ok=True)

  def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(log_path, "a") as f:
      f.write(line + "\n")

  sh = Shooter(args.run_dir, log)
  todo = [t for t in tiles if not sh.done(t["name"])]
  log(f"ortho: {len(todo)} of {len(tiles)} tiles to shoot")
  if not todo:
    return
  sh.setup(todo[0]["x"], todo[0]["y"])
  time.sleep(5.0)
  ref = Reference(sh, tiles[0], log)
  t0 = time.monotonic()
  shot, failed, prev = 0, [], None
  for t in todo:
    if not ref.check():
      log("ortho: stopping: the reference tile keeps coming out different")
      break
    if args.minutes and time.monotonic() - t0 > args.minutes * 60:
      log(f"ortho: time's up after {shot} tiles")
      break
    if shot % 50 == 0 and free_gb(args.run_dir) < args.stop_free_gb:
      log(f"ortho: stopping: {free_gb(args.run_dir):.1f} GB free")
      break
    hop = math.hypot(t["x"] - prev["x"], t["y"] - prev["y"]) if prev and ref.next - time.monotonic() < Reference.EVERY - 1 else 1e4
    ok = sh.shoot(t, wait_for(hop))
    if not ok:
      ok = sh.shoot(t, max(2.0, wait_for(hop)))
    prev = t
    if not ok:
      failed.append(t["name"])
      if len(failed) > 50 and len(failed) > shot:
        log("ortho: stopping: more tiles failing than shot")
        break
      continue
    shot += 1
    if shot % 50 == 0:
      log(f"ortho: {shot} shot, {(time.monotonic() - t0) / shot:.2f} s a tile, {len(failed)} failed, {free_gb(args.run_dir):.0f} GB free")
  sh.saved.join()
  blank = [n for n, q in sh.results.items() if q.get("error") or q.get("black", 0) > 0.5]
  out = {"shot": shot, "failed": failed, "mostly_black": blank, "s_per_tile": round((time.monotonic() - t0) / max(shot, 1), 2)}
  log("ortho: " + json.dumps(out))


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  a = sub.add_parser("plan")
  a.add_argument("out")
  a.add_argument("--roads", help="the perspective capture's tiles.json: its tiles' cells first")
  a.add_argument("--water", help="the plugin's water mask (ortho op=water over this grid): then the rest of the land")
  a.add_argument("--centres", type=float, nargs="*", help="x y ...: a pilot, n x n cells around each")
  a.add_argument("--n", type=int, default=3)
  b = sub.add_parser("run")
  b.add_argument("run_dir")
  b.add_argument("plan")
  b.add_argument("--minutes", type=float)
  b.add_argument("--stop-free-gb", type=float, default=30.0)
  args = ap.parse_args()
  plan(args) if args.cmd == "plan" else run(args)


if __name__ == "__main__":
  main()
