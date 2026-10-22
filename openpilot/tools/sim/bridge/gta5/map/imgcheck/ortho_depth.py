#!/usr/bin/env python3
"""Height tiles over the ortho grid (ortho.py), and under-deck tiles where roads pass under roads.

depth: per tile of an ortho plan, the camera and box of its albedo tile (the albedo sidecar's camera height and LOD), the
grab source the depth target ("DepthBuffer Pre Alpha 1": reversed-Z float32, valid at the present), saved as 16-bit
heights: h = base_m + v * scale_m (v = NODATA where nothing was drawn), with the albedo tile's world_to_px. Right after
the grab, while the tile is streamed (model lakes and the bed collision answer only then), the game's water over the tile
on a 1 m world grid (plugin `waterprobe`), saved as <name>_w.npz before the sidecar: h (water surface, NaN none), c (0 none,
1 water, 2 blocked, 3 other), bed + bcode (collision under each water point: 1 hit, 2 nothing within WATER_BED m, 3
failed), hn + q (GET_WATER_HEIGHT_NO_WAVES and where it answered), x0, y0, step (row 0 = y0, south). A tile done without
its _w file (the probe failed, or shot before it existed) gets only the probe on a resume. The water is calmed for the
pass (plugin `waves calm=1`: no waves or wind), so the sea in the height tiles is flat at its NO_WAVES height.

underplan: road links (the game's path nodes) crossing in 2D with more than DZ m between them; per grid cell, the lower
roads binned to BAND m levels.

under: per cell-level, the camera between the lower road and the deck over it (abs height), so the deck is behind the
camera and neither drawn nor occluding; near 0.5 m, far 120 m. The camera height comes from shape tests up from the lower
road (the deck's underside) and down (its surface); each shot is checked from its own depth (the lower road reads its node
heights, covers its footprint, nothing under the camera over it, and it agrees with the top-level height tile outside the
decks) and retried higher, lower and nudged along the road when it fails; the best attempt is kept, all metrics recorded.
Albedo (traced: copies from within the frame, no black holes), lit and heights, named <cell>_L<level>.

Both resume (tiles with a sidecar are skipped), check a reference tile every 30 min, stop under --stop-free-gb, and keep
at most a few tiles in memory (the writer's queue is bounded).

  python ortho_depth.py depth <run_dir> <plan.json> --albedo <tiles dir> [--ref x y] [--until HH:MM] [--minutes N]
  python ortho_depth.py underplan <paths.jsonl> <out.json>
  python ortho_depth.py under <run_dir> <plan.json> [--depth <height tiles dir>] [--ref x y] [--until HH:MM]
  python ortho_depth.py report <under run_dir>   (failing list + contact sheet)
"""
import argparse
import collections
import json
import math
import os
import re
import time

import numpy as np
from PIL import Image, ImageDraw

from openpilot.tools.sim.bridge.gta5.map.imgcheck import ortho
from openpilot.tools.sim.bridge.gta5.map.imgcheck.ortho import FAR, FOV, H_M, H_PX, HEIGHT, NEAR, PARK_ABOVE, W_M, W_PX

PLUGIN_LOG = "/mnt/d/SteamLibrary/steamapps/common/Grand Theft Auto V Enhanced/gta5op/gta5op.log"
DEPTH_TARGET = "DepthBuffer Pre Alpha 1"
NODATA = 65535
SCALE = 0.01  # m per step: centimetres
STABLE_M = 0.05  # mean abs height change (m, 48 cm blocks, each capped at 2 m) under which the scene has settled
DZ, BAND = 4.0, 3.0  # m: crossings with more than DZ between the roads; lower roads binned to BAND m levels
CHAIN = 40.0  # m along the lower road either way from a crossing link: the road checked in an under-deck shot
U_NEAR, U_FAR = 0.5, 120.0  # the game draws nothing within ~1% of the near end: (far - near) / 100 under the near plane
U_BLIND = U_NEAR + 0.01 * (U_FAR - U_NEAR) + 0.3  # m under the camera where nothing shows
U_MIN_ABOVE = 2.0  # m: the camera at least this far over the lower road's highest point
WATER_STEP, WATER_BED, WATER_ABOVE = 1.0, 60.0, 50.0  # m: probe grid, bed rays' reach, probes' top over the tile's highest surface
WATER_GIVE_UP = 20  # consecutive failed tiles after which the pass stops probing (heights first)


# ------------------------------------------------------------------------------------------------------------ heights
def load_raw(path: str) -> np.ndarray:
  """The plugin's raw plane: u32 width, height, row pitch, DXGI format, then the rows; the depth plane as float32."""
  b = open(path, "rb").read()
  w, h, pitch, _ = np.frombuffer(b[:16], dtype=np.uint32)
  rows = np.frombuffer(b[16:16 + int(pitch) * int(h)], dtype=np.uint8).reshape(int(h), int(pitch))
  return rows[:, :int(w) * 4].copy().view(np.float32).reshape(int(h), int(w))


def to_height(d: np.ndarray, cam_z: float, near: float, far: float) -> np.ndarray:
  """Ortho, reversed Z: depth 1 at the near plane, 0 at the far one, linear between; 0 = nothing drawn (NaN)."""
  h = (cam_z - (far - d.astype(np.float64) * (far - near))).astype(np.float32)
  h[d <= 0] = np.nan
  return h


def encode(h: np.ndarray) -> tuple[np.ndarray, dict]:
  ok = np.isfinite(h)
  if not ok.any():
    return np.full(h.shape, NODATA, np.uint16), {"base_m": 0.0, "scale_m": SCALE, "nodata": NODATA, "valid": 0.0, "clipped": 0}
  base = math.floor(float(h[ok].min()) / SCALE) * SCALE
  v = np.round((np.where(ok, h, base) - base) / SCALE)
  clipped = int((v > NODATA - 1).sum())
  v = np.clip(v, 0, NODATA - 1).astype(np.uint16)
  v[~ok] = NODATA
  return v, {"base_m": round(base, 2), "scale_m": SCALE, "nodata": NODATA, "valid": round(float(ok.mean()), 4), "clipped": clipped,
             "min_m": round(float(h[ok].min()), 2), "max_m": round(float(h[ok].max()), 2)}


def decode(path: str, meta: dict) -> np.ndarray:
  v = np.asarray(Image.open(path), dtype=np.float32)
  h = meta["base_m"] + v * meta["scale_m"]
  h[v == meta["nodata"]] = np.nan
  return h.astype(np.float32)


def save_height(path: str, v: np.ndarray) -> None:
  Image.fromarray(v.astype("<u2")).save(path, compress_level=3)


def blocks16(h: np.ndarray, fill: float) -> np.ndarray:
  a = np.where(np.isfinite(h), h, fill)
  return a.reshape(H_PX // 16, 16, W_PX // 16, 16).mean(axis=(1, 3))


def changed(a: np.ndarray, b: np.ndarray) -> float:
  return float(np.minimum(np.abs(a - b), 2.0).mean())


def parse_water(text: str) -> dict[str, np.ndarray]:
  """waterprobe rows: per point "-" none, "b" blocked, "?r" other, else the height, with the bed as "h/bed", "h/-", "h/f"."""
  rows = [r.split() for r in text.splitlines() if r.strip()]
  shape = (len(rows), len(rows[0]))
  h, bed = np.full(shape, np.nan, np.float32), np.full(shape, np.nan, np.float32)
  c, bcode = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
  for j, r in enumerate(rows):
    for i, v in enumerate(r):
      if v == "-":
        continue
      if v == "b" or v[0] == "?":
        c[j, i] = 2 if v == "b" else 3
        continue
      v, _, b = v.partition("/")
      c[j, i], h[j, i] = 1, float(v)
      if b:
        bcode[j, i] = {"-": 2, "f": 3}.get(b, 1)
        if bcode[j, i] == 1:
          bed[j, i] = float(b)
  return {"h": h, "c": c, "bed": bed, "bcode": bcode}


# ---------------------------------------------------------------------------------------------------------- the plugin
class PluginLog:
  def __init__(self, path: str = PLUGIN_LOG):
    self.path = path

  def mark(self) -> int:
    return os.path.getsize(self.path)

  def wait(self, pos: int, regex: str, timeout: float):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
      size = os.path.getsize(self.path)
      pos = 0 if size < pos else pos
      if size > pos:
        with open(self.path, "rb") as f:
          f.seek(pos)
          m = re.search(regex, f.read().decode("utf-8", "replace"), re.M)
        if m:
          return m
      time.sleep(0.05)
    return None


class ZShooter(ortho.Shooter):
  """ortho.Shooter with the depth target as a second grab source and shape tests; saving is the pass's own."""

  def __init__(self, run_dir: str, log, saver=None, calm=False):
    self.saver, self.calm = saver, calm
    super().__init__(run_dir, log)
    self.plog = PluginLog()
    self.addr: dict[str, str] = {}
    self.ray_id = 0
    self.source = ""

  def _save_loop(self):  # the pass's saver, run on ortho.Shooter's (bounded) queue
    while True:
      item = self.saved.get()
      try:
        self.saver(item)
      except Exception as e:
        self.log(f"ortho: {item[-1].get('name')}: save failed: {type(e).__name__}: {e}")
      finally:
        self.saved.task_done()

  def find(self, name: str) -> str:
    """A render target object by name (the plugin's memory search), which also becomes the grab source."""
    pos = self.plog.mark()
    self.send({"type": "ortho", "op": "gbuffer", "name": name})
    m = self.plog.wait(pos, rf"ortho gbuffer: {re.escape(name)} (?:at ([0-9a-f]+)|not found)", 120.0)
    if not m or not m.group(1):
      raise RuntimeError(f"ortho: render target {name} not found")
    self.log(f"ortho: {name} at {m.group(1)}")
    return m.group(1)

  def use(self, kind: str) -> None:
    """kind 'albedo' (GBUFFER_0, traced) or 'depth' as the grab source."""
    if kind == self.source:
      return
    if kind == "albedo":
      self.send({"type": "ortho", "op": "gsrc", "addr": self.addr["albedo"], "state": 0xC0}, {"type": "ortho", "op": "gtrace", "on": 1})
    else:
      self.send({"type": "ortho", "op": "gtrace", "on": 0}, {"type": "ortho", "op": "gsrc", "addr": self.addr["depth"], "state": 0xC0})
    self.source = kind
    time.sleep(0.3)

  def setup(self, x: float, y: float) -> None:
    super().setup(x, y)  # scene, box, GBUFFER_0 found (when no source), trace on
    # found again each time: after a game restart they are elsewhere
    self.addr["albedo"] = self.find("GBUFFER_0")
    self.addr["depth"] = self.find(DEPTH_TARGET)
    self.source = ""
    if self.calm:  # the plugin's calm water: no waves, so the sea's surface is its NO_WAVES height
      self.send({"type": "waves", "calm": 1})

  def grab_depth(self, name: str) -> tuple[str, str] | None:
    bmp, raw = os.path.join(self.raw_dir, name + ".bmp"), os.path.join(self.raw_dir, name + "_src.raw")
    for p in (bmp, raw):
      if os.path.exists(p):
        os.remove(p)
    self.send({"type": "grab", "path": self.raw_win + "\\" + name + ".bmp"})
    end = time.monotonic() + ortho.GRAB_WAIT
    while time.monotonic() < end:
      if os.path.exists(bmp) and os.path.exists(raw):
        return bmp, raw
      time.sleep(0.02)
    return None

  def settled_depth(self, name: str, cam_z: float, near: float, far: float, tries: int = 6, gap: float = 1.5):
    """A depth grab once two grabs gap s apart agree (streaming after a long hop)."""
    prev, got, diff = None, None, None
    for k in range(tries):
      got = self.grab_depth(name)
      if got is None:
        return None, diff
      small = blocks16(to_height(load_raw(got[1]), cam_z, near, far), cam_z - far)
      if prev is not None:
        diff = changed(small, prev)
        if diff < STABLE_M:
          return got, diff
      prev = small
      if k + 1 < tries:
        time.sleep(gap)
    self.log(f"ortho: {name}: depth still changing after {tries} grabs ({diff:.3f} m)")
    return got, diff

  def rays(self, segs: list[tuple[float, ...]]) -> list[tuple[float, float, float] | None]:
    """The first surface along each segment (x, y, z0, z1 vertical, or x0, y0, z0, x1, y1, z1)."""
    self.ray_id += 1
    rid = f"{os.getpid()}.{self.ray_id}"
    pos = self.plog.mark()
    self.send({"type": "ray", "id": rid, "p": ";".join(",".join(f"{v:.2f}" for v in s) for s in segs)})
    m = self.plog.wait(pos, rf"ray {re.escape(rid)} (.*)$", 5.0)
    if not m:
      return [None] * len(segs)
    out = []
    for item in m.group(1).strip().split(";"):
      p = item.split(",")
      out.append(tuple(float(v) for v in p) if len(p) == 3 else None)
    return (out + [None] * len(segs))[:len(segs)]

  def waterprobe(self, tag: str, timeout: float, **grid) -> str | None:
    """The plugin's waterprobe over a grid (x0 y0 x1 y1 step z m f bed ms): its rows, None when it doesn't finish in time."""
    self.ray_id += 1
    wid = f"{tag}.{os.getpid()}.{self.ray_id}"
    path = os.path.join(self.raw_dir, wid + ".txt")
    pos = self.plog.mark()
    self.send({"type": "waterprobe", "id": wid, **grid, "path": self.raw_win + "\\" + wid + ".txt"})
    if not self.plog.wait(pos, rf"waterprobe {re.escape(wid)} done: ", timeout) or not os.path.exists(path):
      return None
    text = open(path).read()
    os.remove(path)
    return text

  def ensure(self, st: dict | None, x: float, y: float) -> bool:
    if st is None or not isinstance(st.get("topcam"), dict):
      return False
    ok, why = self.scene_ok(st)
    if not ok:
      self.log(f"ortho: {why}; setting up again")
      self.setup(x, y)
      return False
    return True


def write_json(path: str, obj: dict) -> None:
  """Whole or not at all: a sidecar marks its tile done."""
  with open(path + ".tmp", "w") as f:
    json.dump(obj, f, indent=1)
  os.replace(path + ".tmp", path)


def until_deadline(args) -> float | None:
  """--end (epoch s), --until HH:MM (the next such time) or --minutes, as a time.time() deadline."""
  ends = [args.end] if args.end else []
  if args.minutes:
    ends.append(time.time() + args.minutes * 60)  # noqa: TID251
  if args.until:
    hh, mm = (int(v) for v in args.until.split(":"))
    t = time.localtime()
    end = time.mktime((t.tm_year, t.tm_mon, t.tm_mday, hh, mm, 0, 0, 0, -1))
    ends.append(end + 86400 if end < time.time() else end)  # noqa: TID251
  return min(ends) if ends else None


def make_log(run_dir: str):
  path = os.path.join(run_dir, "capture.log")
  os.makedirs(run_dir, exist_ok=True)

  def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(path, "a") as f:
      f.write(line + "\n")
  return log


# -------------------------------------------------------------------------------------------------------- depth pass
class DepthPass:
  def __init__(self, args, log):
    self.args, self.log = args, log
    self.sh = ZShooter(args.run_dir, log, self.save, calm=True)
    self.results: dict[str, dict] = {}
    self.water_s, self.water_n, self.water_failed, self.water_run = 0.0, 0, 0, 0

  def albedo_side(self, name: str) -> dict | None:
    p = os.path.join(self.args.albedo, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None

  def camera(self, t: dict) -> tuple[dict, float]:
    """The albedo tile's camera: over its highest surface found, its LOD stretch (abs: no probe needed)."""
    side = self.albedo_side(t.get("cell_name", t["name"]))
    tc = (side or {}).get("topcam") or {}
    if tc.get("ground") is not None and (side or {}).get("ground_found", True):
      g, low, hgt = float(tc["ground"]), float(tc.get("low", tc["ground"])), float(tc.get("height", HEIGHT))
      lod = min(max((g - low + hgt) / hgt, 1.0), 10.0)
      return {"type": "topcam", "on": 1, "x": t["x"], "y": t["y"], "z": g, "abs": 1, "height": hgt, "fov": FOV, "heading": 0, "hud": 0,
              "dof": 0, "lod": round(lod, 4)}, g
    return self.sh.topcam(t["x"], t["y"]), self.sh.zguess

  def shot(self, t: dict, wait: float, settle: bool, grab: bool = True):
    cam, g = self.camera(t)
    self.sh.send({"type": "park", "on": 1, "x": t["x"], "y": t["y"], "z": g + PARK_ABOVE}, cam)
    self.sh.park_z = g + PARK_ABOVE
    time.sleep(wait)
    st = self.sh.state()
    if not cam.get("abs") and st and isinstance(st.get("topcam"), dict) and not self.sh.probed(st):
      for z in (g + 250, g - 150, 350, 650, 100, 900, 20):
        self.sh.park(t["x"], t["y"], z)
        time.sleep(1.5)
        st = self.sh.state() or st
        if self.sh.probed(st):
          break
    if not self.sh.ensure(st, t["x"], t["y"]):
      return None
    tc = st["topcam"]
    if not tc.get("on") or abs(tc["x"] - t["x"]) > 0.01 or abs(tc["y"] - t["y"]) > 0.01:
      self.log(f"ortho: {t['name']}: topcam not there yet")
      return None
    if not cam.get("abs") and self.sh.probed(st):
      self.sh.zguess = tc["ground"]
    if not grab:  # there, for the water probe only
      return None, {"topcam": tc}
    self.sh.use("depth")
    cam_z = tc["ground"] + tc["height"]
    diff = None
    if settle:
      got, diff = self.sh.settled_depth(t["name"], cam_z, NEAR, FAR)
    else:
      got = self.sh.grab_depth(t["name"])
    if got is None:
      self.log(f"ortho: {t['name']}: no grab")
      return None
    return got, {"cam_z": cam_z, "near": NEAR, "far": FAR, "topcam": tc, "settle_m": diff, "wait": wait}

  def water(self, t: dict, g: float) -> dict | None:
    """The game's water over the tile's extent (around the albedo sidecar's centre) on the world's 1 m grid, the bed under
    it and the NO_WAVES height; None (logged) when the probe fails. g: the tile's highest surface."""
    if self.water_run >= WATER_GIVE_UP:
      return None
    t0 = time.monotonic()
    side = self.albedo_side(t["name"]) or t
    x, y = side["x"], side["y"]
    grid = {"x0": math.floor(x - W_M / 2), "y0": math.floor(y - H_M / 2), "x1": math.ceil(x + W_M / 2), "y1": math.ceil(y + H_M / 2),
            "step": WATER_STEP, "z": round(g + WATER_ABOVE, 1), "f": 0, "ms": 8}
    try:
      text = self.sh.waterprobe(t["name"], 5.0, m="v", bed=WATER_BED, **grid)
      w = parse_water(text) if text else None
      if w and not (w["c"] == 1).any():  # dry: no NO_WAVES water either
        w["hn"], w["q"] = np.full_like(w["h"], np.nan), np.zeros(w["c"].shape, bool)
      elif w and (nw := self.sh.waterprobe(t["name"] + "_n", 3.0, m="n", **grid)):
        n = parse_water(nw)
        w["hn"], w["q"] = n["h"], n["c"] == 1
    except Exception as e:
      self.log(f"ortho: {t['name']}: water probe: {type(e).__name__}: {e}")
      w = None
    if w is None:
      self.water_failed, self.water_run = self.water_failed + 1, self.water_run + 1
      self.log(f"ortho: {t['name']}: no water probe" + (f"; {WATER_GIVE_UP} in a row: no more probes this pass" if self.water_run >= WATER_GIVE_UP else ""))
      return None
    w.update(x0=grid["x0"], y0=grid["y0"], step=WATER_STEP, z=grid["z"])
    self.water_s, self.water_n, self.water_run = self.water_s + time.monotonic() - t0, self.water_n + 1, 0
    return w

  def water_path(self, name: str) -> str:
    return os.path.join(self.sh.tile_dir, name + "_w.npz")

  def save_water(self, name: str, w: dict) -> dict:
    np.savez_compressed(self.water_path(name)[:-4] + ".tmp.npz", **w)
    os.replace(self.water_path(name)[:-4] + ".tmp.npz", self.water_path(name))
    return {"npz": name + "_w.npz", "water_pts": int((w["c"] == 1).sum()), "no_waves": "q" in w}

  def save(self, item):
    (bmp, raw), w, side = item
    c = side["camera"]
    h = to_height(load_raw(raw), c["cam_z"], c["near"], c["far"])
    v, meta = encode(h)
    name = side["name"]
    meta["png"] = name + "_h.png"
    side["height"] = meta
    save_height(os.path.join(self.sh.tile_dir, meta["png"]), v)
    side["water"] = self.save_water(name, w) if w else None  # before the sidecar: a tile is done with both
    write_json(os.path.join(self.sh.tile_dir, name + ".json"), side)
    os.remove(raw)
    os.remove(bmp)
    self.results[name] = {"valid": meta["valid"], "clipped": meta["clipped"]}

  def reference(self, ref_t: dict, first: list) -> bool:
    """The reference tile again, against its first heights: shift < 0.5 px, median change < 5 cm."""
    got = self.shot({**ref_t, "cell_name": ref_t["name"], "name": "ref_" + time.strftime("%H%M%S")}, 6.0, True)
    if not got:
      return False
    (bmp, raw), c = got
    h = to_height(load_raw(raw), c["cam_z"], c["near"], c["far"])
    os.remove(bmp)
    os.remove(raw)
    h = np.where(np.isfinite(h), h, np.nanmedian(h) if np.isfinite(h).any() else 0.0)
    if not first:
      first.append(h)
      self.log(f"ortho: reference taken at {ref_t['x']:.1f},{ref_t['y']:.1f}")
      return True
    sx, sy, peak = ortho.phase_shift(first[0][200:1240, 300:2260], h[200:1240, 300:2260])
    med = float(np.median(np.abs(first[0] - h)))
    ok = abs(sx) < 0.5 and abs(sy) < 0.5 and med < 0.05
    self.log(f"ortho: reference: shift {sx:.2f},{sy:.2f} px, peak {peak:.2f}, median change {med:.3f} m" + ("" if ok else " BAD"))
    return ok

  def run(self) -> None:
    a, log, sh = self.args, self.log, self.sh
    tiles = json.load(open(a.plan))["tiles"]
    if a.names:
      want = set(open(a.names).read().split())
      tiles = [t for t in tiles if t["name"] in want]
    todo = [t for t in tiles if not sh.done(t["name"]) or not os.path.exists(self.water_path(t["name"]))]
    log(f"ortho depth: {len(todo)} of {len(tiles)} tiles to shoot ({sum(sh.done(t['name']) for t in todo)} for the water probe only)")
    if not todo:
      return
    deadline = until_deadline(a)
    sh.setup(todo[0]["x"], todo[0]["y"])
    time.sleep(3.0)
    ref_t = None
    if a.ref:
      i, j = round((a.ref[0] - ortho.GRID_X0) / ortho.STEP_X), round((a.ref[1] - ortho.GRID_Y0) / ortho.STEP_Y)
      x, y = ortho.cell_centre(i, j)
      ref_t = {"name": f"o{i:03d}_{j:03d}", "i": i, "j": j, "x": x, "y": y}
    first: list = []
    ref_next, ref_bad = 0.0, 0
    t0 = time.monotonic()
    shot, failed, prev, since_ref = 0, [], None, True
    for t in todo:
      if ref_t and time.monotonic() >= ref_next:
        ref_next = time.monotonic() + ortho.Reference.EVERY
        ref_bad = 0 if self.reference(ref_t, first) else ref_bad + 1
        since_ref = True
        if ref_bad:
          sh.setup(t["x"], t["y"])
        if ref_bad >= 3:
          log("ortho: stopping: the reference tile keeps coming out different")
          break
      if deadline and time.time() > deadline:  # noqa: TID251
        log(f"ortho: time's up after {shot} tiles")
        break
      if shot % 50 == 0 and ortho.free_gb(a.run_dir) < a.stop_free_gb:
        log(f"ortho: stopping: {ortho.free_gb(a.run_dir):.1f} GB free")
        break
      water_only = sh.done(t["name"])
      if water_only and self.water_run >= WATER_GIVE_UP:
        continue
      hop = 1e4 if prev is None or since_ref else math.hypot(t["x"] - prev["x"], t["y"] - prev["y"])
      since_ref = False
      wait = ortho.wait_for(hop)
      if water_only:
        got = self.shot(t, max(2.0, wait), False, grab=False)
        w = self.water(t, got[1]["topcam"]["ground"]) if got else None
        if w:
          self.save_water(t["name"], w)
        prev = t
        continue
      got = self.shot(t, wait, wait > 1.0) or self.shot(t, max(2.0, wait), True)
      prev = t
      if not got:
        failed.append(t["name"])
        if len(failed) > 50 and len(failed) > shot:
          log("ortho: stopping: more tiles failing than shot")
          break
        continue
      files, cam = got
      w = self.water(t, cam["topcam"]["ground"])  # the tile still streamed
      side = {"name": t["name"], "cell": [t["i"], t["j"]], "x": t["x"], "y": t["y"], "cm_per_px": ortho.CM_PER_PX, "size": [W_PX, H_PX],
              "extent_m": [W_M, H_M], "world_to_px": ortho.world_to_px(t["x"], t["y"]), "north_up": True, "time": time.time(),  # noqa: TID251
              "camera": cam, "kind": "height", "waves": "calm"}
      sh.saved.put((files, w, side))
      shot += 1
      if shot % 50 == 0:
        log(f"ortho: {shot} shot, {(time.monotonic() - t0) / shot:.2f} s a tile, {len(failed)} failed, {ortho.free_gb(a.run_dir):.0f} GB free, "
            f"water probe {self.water_s / max(self.water_n, 1):.2f} s a tile ({self.water_failed} failed)")
    sh.saved.join()
    sh.send({"type": "waves", "calm": 0})
    bad = [n for n, q in self.results.items() if q.get("valid", 1) < 0.5]
    log("ortho depth: " + json.dumps({"shot": shot, "failed": failed, "mostly_empty": bad, "s_per_tile": round((time.monotonic() - t0) / max(shot, 1), 2),
                                      "water_probes": self.water_n, "water_failed": self.water_failed,
                                      "water_s_per_tile": round(self.water_s / max(self.water_n, 1), 3)}))


# ------------------------------------------------------------------------------------------------------ under: plan
def lanes_width(f2: int) -> float:
  """A link's width from its lane counts (path link flags byte 2: forward lanes in bits 5-7, backward in 2-4)."""
  n = ((f2 >> 5) & 7) + ((f2 >> 2) & 7)
  return float(min(max(3.4 * max(n, 1), 3.5), 25.0))


def underplan(args) -> None:
  nodes, links = {}, []
  for line in open(args.paths):
    r = json.loads(line)
    if r["t"] == "n":
      nodes[(r["a"], r["i"])] = (r["x"], r["y"], r["z"])
    elif r["t"] == "l":
      links.append(((r["a"], r["i"]), (r["ta"], r["ti"]), (r.get("f") or [0, 0, 0])[2]))
  seg, seen, ends = [], set(), []
  for a, b, f2 in links:
    if a not in nodes or b not in nodes or (b, a) in seen or (a, b) in seen:
      continue
    seen.add((a, b))
    seg.append(nodes[a] + nodes[b] + (lanes_width(f2),))
    ends.append((a, b))
  seg = np.array(seg)
  at = collections.defaultdict(list)
  for k, (a, b) in enumerate(ends):
    at[a].append(k)
    at[b].append(k)

  def chain(k0: int) -> set[int]:
    """The lower link and the links joined to it, CHAIN m each way, on its level (within 2 m of its height)."""
    z = 0.5 * (seg[k0][2] + seg[k0][5])
    out, todo = {k0}, [(k0, 0.0)]
    while todo:
      k, run = todo.pop()
      for n in ends[k]:
        for q in at[n]:
          L = math.hypot(seg[q][3] - seg[q][0], seg[q][4] - seg[q][1])
          if q not in out and run + L < CHAIN and abs(0.5 * (seg[q][2] + seg[q][5]) - z) < 2.0:
            out.add(q)
            todo.append((q, run + L))
    return out
  grid = collections.defaultdict(list)
  for k, (x0, y0, _, x1, y1, _, _) in enumerate(seg):
    for cx in range(int(math.floor(min(x0, x1) / 50)), int(math.floor(max(x0, x1) / 50)) + 1):
      for cy in range(int(math.floor(min(y0, y1) / 50)), int(math.floor(max(y0, y1) / 50)) + 1):
        grid[(cx, cy)].append(k)
  cross, done = [], set()
  for ks in grid.values():
    for ii in range(len(ks)):
      for jj in range(ii + 1, len(ks)):
        p, q = ks[ii], ks[jj]
        if (p, q) in done:
          continue
        done.add((p, q))
        x0, y0, z0, x1, y1, z1, _ = seg[p]
        u0, v0, w0, u1, v1, w1, _ = seg[q]
        d = (x1 - x0) * (v1 - v0) - (y1 - y0) * (u1 - u0)
        if abs(d) < 1e-9:
          continue
        t = ((u0 - x0) * (v1 - v0) - (v0 - y0) * (u1 - u0)) / d
        s = ((u0 - x0) * (y1 - y0) - (v0 - y0) * (x1 - x0)) / d
        if not (0.02 < t < 0.98 and 0.02 < s < 0.98):
          continue
        za, zb = z0 + t * (z1 - z0), w0 + s * (w1 - w0)
        if abs(za - zb) > DZ:
          lo = p if za < zb else q
          cross.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0), min(za, zb), max(za, zb), lo))
  by_cell = collections.defaultdict(list)
  for c in cross:
    by_cell[(round((c[0] - ortho.GRID_X0) / ortho.STEP_X), round((c[1] - ortho.GRID_Y0) / ortho.STEP_Y))].append(c)
  tiles = []
  for i, j in ortho.blocks(set(by_cell)):
    cs = sorted(by_cell[(i, j)], key=lambda c: c[2])
    levels, start = [], None
    for c in cs:
      if start is None or c[2] > start + BAND:
        levels.append([])
        start = c[2]
      levels[-1].append(c)
    x, y = ortho.cell_centre(i, j)
    for k, lv in enumerate(levels, 1):
      segs = sorted(set().union(*(chain(c[4]) for c in lv)))
      tiles.append({"name": f"o{i:03d}_{j:03d}_L{k}", "i": i, "j": j, "x": round(x, 3), "y": round(y, 3), "level": k,
                    "z_lo": round(min(c[2] for c in lv), 2), "z_hi": round(max(c[2] for c in lv), 2),
                    "crossings": [[round(v, 2) for v in c[:4]] for c in lv], "segs": [[round(float(v), 2) for v in seg[s]] for s in segs]})
  json.dump({"cm_per_px": ortho.CM_PER_PX, "near": U_NEAR, "far": U_FAR, "dz": DZ, "band": BAND, "crossings": len(cross), "cells": len(by_cell),
             "tiles": tiles}, open(args.out, "w"))
  print(f"{len(cross)} crossings, {len(by_cell)} cells, {len(tiles)} cell-levels -> {args.out}")


# ----------------------------------------------------------------------------------------------------- under: checks
def seg_samples(segs, step=0.5):
  for x0, y0, z0, x1, y1, z1, _ in segs:
    n = max(int(math.hypot(x1 - x0, y1 - y0) / step), 1)
    t = np.linspace(0, 1, n + 1)
    yield x0 + t * (x1 - x0), y0 + t * (y1 - y0), z0 + t * (z1 - z0)


def checks(h: np.ndarray, M: np.ndarray, cam_z: float, segs: list, top: np.ndarray | None, ground: list | None = None) -> dict:
  """(a) the lower road's centreline reads its node heights, or (where the nodes are off) the road's collision surface
  under the shape tests (ground: x, y, z); (b) coverage of its footprint (buffered by its width) at
  its height; (c) nothing between the road and the camera over it, nothing just under the camera; (d) outside the decks
  (where the top-level tile's surface is under the camera) the heights match the top-level tile."""
  out: dict = {}
  errs = []
  for xs, ys, zs in seg_samples(segs):
    px, py = M[0, 0] * xs + M[0, 2], M[1, 1] * ys + M[1, 2]
    k = (px >= 0) & (px < W_PX) & (py >= 0) & (py < H_PX)
    if k.any():
      errs.append(h[py[k].astype(int), px[k].astype(int)] - zs[k])
  e = np.concatenate(errs) if errs else np.zeros(0)
  ae = np.where(np.isfinite(e), np.abs(e), 99.0)
  out["a_samples"] = int(e.size)
  out["a_median_m"] = round(float(np.median(ae)), 3) if e.size else None
  out["a_share_015"] = round(float((ae < 0.15).mean()), 3) if e.size else 0.0
  out["a"] = bool(e.size >= 6 and out["a_median_m"] < 0.1 and out["a_share_015"] >= 0.6)
  if ground:
    g = np.array(ground, dtype=np.float64)
    px, py = (M[0, 0] * g[:, 0] + M[0, 2]).astype(int), (M[1, 1] * g[:, 1] + M[1, 2]).astype(int)
    k = (px >= 0) & (px < W_PX) & (py >= 0) & (py < H_PX)
    ge = np.abs(np.nan_to_num(h[py[k], px[k]] - g[k, 2], nan=99.0))
    out["a_ground_n"] = int(k.sum())
    out["a_ground_share_015"] = round(float((ge < 0.15).mean()), 3) if k.any() else 0.0
    if not out["a"] and k.sum() >= 3 and out["a_ground_share_015"] >= 0.8:
      out["a"], out["a_basis"] = True, "ground"
  # footprint on a 4 px grid
  s = 4
  cols = (np.arange(0, W_PX, s) + s / 2 - M[0, 2]) / M[0, 0]
  rows = (np.arange(0, H_PX, s) + s / 2 - M[1, 2]) / M[1, 1]
  X, Y = np.meshgrid(cols, rows)
  hs = h[s // 2::s, s // 2::s]
  best = np.full(X.shape, np.inf)
  zexp = np.full(X.shape, np.nan)
  for x0, y0, z0, x1, y1, z1, w in segs:
    dx, dy = x1 - x0, y1 - y0
    L2 = max(dx * dx + dy * dy, 1e-6)
    t = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0, 1)
    d = np.hypot(X - (x0 + t * dx), Y - (y0 + t * dy))
    inside = (d < w / 2) & (d < best)
    best = np.where(inside, d, best)
    zexp = np.where(inside, z0 + t * (z1 - z0), zexp)
  foot = np.isfinite(zexp)
  out["b_footprint_px"] = int(foot.sum()) * s * s
  if foot.any():
    dh = hs[foot] - zexp[foot]
    if out.get("a_basis") == "ground":  # the nodes are off here: the footprint against the surface the shape tests found
      off = float(np.clip(np.nanmedian(dh), -3.0, 3.0)) if np.isfinite(dh).any() else 0.0
      out["b_node_offset_m"] = round(off, 2)
      dh = dh - off
    out["b_coverage"] = round(float((np.abs(np.nan_to_num(dh, nan=99.0)) < 0.3).mean()), 3)
    out["c_above_road"] = round(float((np.nan_to_num(dh, nan=-99.0) > 1.0).mean()), 3)
  else:
    out["b_coverage"], out["c_above_road"] = 0.0, 0.0
  out["b"] = bool(foot.any() and out["b_coverage"] >= 0.5)
  fin = np.isfinite(hs)
  out["c_near_cam"] = round(float((np.nan_to_num(hs, nan=-1e9) > cam_z - U_BLIND + 0.3).sum() / max(fin.sum(), 1)), 4)
  out["nodata"] = round(float(1 - fin.mean()), 4)  # above the camera (terrain over a trench): expected, not a failure
  out["c"] = bool(out["c_above_road"] < 0.15)  # c_near_cam is only recorded: terrain beside a trench is that high too
  out["d"] = None
  if top is not None:  # the top-level heights over the shot's own frame
    t2 = top[s // 2::s, s // 2::s]
    open_ = np.isfinite(t2) & fin & (t2 < cam_z - U_BLIND - 0.7)
    out["d_px"] = int(open_.sum()) * s * s
    if open_.sum() > 1000:
      dd = np.abs(hs[open_] - t2[open_])
      out["d_median_m"] = round(float(np.median(dd)), 3)
      out["d_share_01"] = round(float((dd < 0.1).mean()), 3)
      out["d"] = bool(out["d_share_01"] >= 0.75)
  out["pass"] = bool(out["a"] and out["b"] and out["c"] and out["d"] is not False)
  out["score"] = round(sum(bool(out[k]) for k in "abc") + (out["d"] is not False) + out["b_coverage"] + out["a_share_015"], 3)
  return out


# ------------------------------------------------------------------------------------------------------- under: pass
class UnderPass:
  def __init__(self, args, log):
    self.args, self.log = args, log
    self.sh = ZShooter(args.run_dir, log, lambda item: None)
    self.stats = collections.Counter()
    self.tops: dict = {}

  def top_tile(self, i: int, j: int) -> tuple[np.ndarray, list] | None:
    """A height tile's heights and world_to_px (the last few kept)."""
    if (i, j) not in self.tops:
      p = os.path.join(self.args.depth, f"o{i:03d}_{j:03d}.json")
      side = json.load(open(p)) if os.path.exists(p) else None
      self.tops[(i, j)] = (decode(os.path.join(self.args.depth, side["height"]["png"]), side["height"]), side["world_to_px"]) if side else None
      while len(self.tops) > 9:
        self.tops.pop(next(iter(self.tops)))
    return self.tops[(i, j)]

  def top_frame(self, x: float, y: float) -> np.ndarray | None:
    """The top-level heights over the frame centred at x, y, each pixel from its own grid cell's height tile: a shot off
    the grid's centres (a nudge, a gap shot) spans several."""
    if not self.args.depth:
      return None
    M = ortho.world_to_px(x, y)
    ci = np.round(((np.arange(W_PX) + 0.5 - M[0][2]) / M[0][0] - ortho.GRID_X0) / ortho.STEP_X).astype(int)
    cj = np.round(((np.arange(H_PX) + 0.5 - M[1][2]) / M[1][1] - ortho.GRID_Y0) / ortho.STEP_Y).astype(int)
    out, found = np.full((H_PX, W_PX), np.nan, np.float32), False
    for i in np.unique(ci):
      for j in np.unique(cj):
        t = self.top_tile(int(i), int(j))
        if t is None:
          continue
        h, T = t
        cols, rows = np.nonzero(ci == i)[0], np.nonzero(cj == j)[0]
        sc, sr = cols - int(round(M[0][2] - T[0][2])), rows - int(round(M[1][2] - T[1][2]))  # the tile's pixels
        kc, kr = (sc >= 0) & (sc < W_PX), (sr >= 0) & (sr < H_PX)
        out[np.ix_(rows[kr], cols[kc])] = h[np.ix_(sr[kr], sc[kc])]
        found = True
    return out if found else None

  def clearance(self, t: dict) -> dict:
    """Shape tests at the crossings and 4 m either side along the lower road: up from 0.5 m over it (the first hit is
    the underside of what's over it), down from 1.5 m (its surface)."""
    pts = []
    segs = t["segs"]
    for x, y, zl, zh in t["crossings"]:
      sg = min(segs, key=lambda s: abs((s[3] - s[0]) * (s[1] - y) - (s[0] - x) * (s[4] - s[1])) / max(math.hypot(s[3] - s[0], s[4] - s[1]), 1e-6))
      L = max(math.hypot(sg[3] - sg[0], sg[4] - sg[1]), 1e-6)
      ux, uy, gz = (sg[3] - sg[0]) / L, (sg[4] - sg[1]) / L, (sg[5] - sg[2]) / L
      for d in (0.0, -4.0, 4.0):
        pts.append((x + d * ux, y + d * uy, zl + d * gz, zh))
    pts = pts[:30]
    ups = self.sh.rays([(x, y, z + 0.5, z + 60.0) for x, y, z, _ in pts])
    downs = self.sh.rays([(x, y, z + 1.5, z - 4.0) for x, y, z, _ in pts])
    # the road's collision surface along its links in the tile (check a where the nodes' heights are off)
    gp = [p[:3] for p in pts]
    for xs, ys, zs in seg_samples(segs, 4.0):
      k = (np.abs(xs - t["x"]) < W_M / 2) & (np.abs(ys - t["y"]) < H_M / 2)
      gp += list(zip(xs[k], ys[k], zs[k], strict=True))
    gp = gp[:60]
    ground = []
    for (x, y, z), d in zip(gp, self.sh.rays([(x, y, z + 1.5, z - 4.0) for x, y, z in gp]), strict=True):
      if d and abs(d[2] - z) < 3.0 and d[2] < z + 1.45:
        ground.append([round(float(x), 2), round(float(y), 2), round(d[2], 3)])
    ivals, surf, hits = [], [], 0
    for (_, _, z, zh), u, d in zip(pts, ups, downs, strict=True):
      sz = d[2] if d and abs(d[2] - z) < 3.0 and d[2] < z + 1.45 else z
      surf.append(sz)
      up = u[2] if u and u[2] - sz > 1.2 and u[2] < zh + 1.0 else None
      hits += up is not None
      ivals.append((sz + U_MIN_ABOVE, (up if up is not None else zh - 1.5) - 0.3))
    # the camera height inside the most points' gaps (road + U_MIN_ABOVE .. underside - 0.3), the most margin among those
    lo_all, hi_all = min(v[0] for v in ivals), max(v[1] for v in ivals)
    best, cam = (-1, -1e9), lo_all
    for c in np.arange(lo_all, max(hi_all, lo_all) + 0.05, 0.1):
      inside = [min(c - a, b - c) for a, b in ivals if a <= c <= b]
      key = (len(inside), min(inside) if inside else -1e9)
      if key > best:
        best, cam = key, float(c)
    if best[0] <= 0:
      cam = float(np.median([v[0] for v in ivals]))
    return {"rays": len(pts), "hits_up": hits, "surface_max": round(max(surf), 2), "surface_median": round(float(np.median(surf)), 2),
            "gaps": [[round(a, 2), round(b, 2)] for a, b in ivals], "camera": round(cam, 2), "points_inside": max(best[0], 0),
            "measured": hits > 0, "range": [round(lo_all, 2), round(max(hi_all, lo_all), 2)], "ground": ground}

  def attempt(self, t: dict, name: str, cam_z: float, dx: float, dy: float, wait: float):
    x, y = t["x"] + dx, t["y"] + dy
    self.sh.send({"type": "park", "on": 1, "x": x, "y": y, "z": cam_z},
                 {"type": "topcam", "on": 1, "x": x, "y": y, "z": cam_z - 3.0, "abs": 1, "height": 3.0, "fov": FOV, "heading": 0, "hud": 0, "dof": 0,
                  "lod": 0},
                 {"type": "ortho", "op": "box", "on": 1, "fov": FOV, "w": W_M, "h": H_M, "near": U_NEAR, "far": U_FAR})
    time.sleep(wait)
    st = self.sh.state()
    if not self.sh.ensure(st, x, y):
      return None
    tc = st["topcam"]
    if abs(tc["x"] - x) > 0.01 or abs(tc["y"] - y) > 0.01:
      return None
    cz = tc["ground"] + tc["height"]
    self.sh.use("albedo")
    got = self.sh.settled(name, tries=6, gap=1.0)
    if not got:
      return None
    self.sh.use("depth")
    dg = self.sh.grab_depth(name + "_d")
    if not dg:
      return None
    os.remove(dg[0])
    h = to_height(load_raw(dg[1]), cz, U_NEAR, U_FAR)
    return {"albedo": got[1], "lit": got[0], "raw": dg[1], "h": h, "cam_z": cz, "x": x, "y": y, "topcam": tc}

  def shoot(self, t: dict, wait: float) -> dict | None:
    sh = self.sh
    # there first, the camera somewhere under the deck, so collision and the scene stream in
    z0 = t["z_hi"] + U_MIN_ABOVE + 0.5
    sh.send({"type": "park", "on": 1, "x": t["x"], "y": t["y"], "z": z0},
            {"type": "topcam", "on": 1, "x": t["x"], "y": t["y"], "z": t["z_lo"], "abs": 1, "height": max(z0 - t["z_lo"], 1.0), "fov": FOV,
             "heading": 0, "hud": 0, "dof": 0, "lod": 0},
            {"type": "ortho", "op": "box", "on": 1, "fov": FOV, "w": W_M, "h": H_M, "near": U_NEAR, "far": U_FAR})
    time.sleep(wait)
    cl = self.clearance(t)
    base, (zmin, zmax) = cl["camera"], cl["range"]
    # the lower road's direction (the longest lower link) for nudges
    sg = max(t["segs"], key=lambda s: math.hypot(s[3] - s[0], s[4] - s[1]))
    L = max(math.hypot(sg[3] - sg[0], sg[4] - sg[1]), 1e-6)
    ux, uy = (sg[3] - sg[0]) / L, (sg[4] - sg[1]) / L
    tries = [(base, 0.0, 0.0)]
    for dz in (0.8, -0.8, 1.6):
      z = base + dz
      if zmin - 0.2 <= z <= zmax and all(abs(z - c[0]) > 0.3 for c in tries):
        tries.append((z, 0.0, 0.0))
    tries += [(base, 3.0 * ux, 3.0 * uy), (base, -3.0 * ux, -3.0 * uy)]
    attempts, best, best_files = [], None, None
    for k, (cz, dx, dy) in enumerate(tries):
      got = self.attempt(t, f"{t['name']}_a{k}", cz, dx, dy, 1.0 if k else 1.5)
      if got is None:
        attempts.append({"k": k, "camera": round(cz, 2), "nudge_m": [round(dx, 2), round(dy, 2)], "error": "no shot"})
        continue
      M = np.array(ortho.world_to_px(got["x"], got["y"]))
      ch = checks(got["h"], M, got["cam_z"], t["segs"], self.top_frame(got["x"], got["y"]), cl["ground"])
      attempts.append({"k": k, "camera": round(got["cam_z"], 2), "nudge_m": [round(dx, 2), round(dy, 2)], **ch})
      if best is None or ch["score"] > best[1]["score"]:
        if best_files:
          self.drop(best_files)
        best, best_files = (got, ch, k), got
      else:
        self.drop(got)
      if ch["pass"]:
        break
    if best is None:
      return None
    got, ch, k = best
    self.save(t, got, ch, k, cl, attempts)
    return {"pass": ch["pass"], "attempts": len(attempts)}

  @staticmethod
  def drop(got: dict) -> None:
    for key in ("albedo", "lit", "raw"):
      if os.path.exists(got[key]):
        os.remove(got[key])

  def save(self, t, got, ch, k, cl, attempts) -> None:
    d, name = self.sh.tile_dir, t["name"]
    Image.open(got["albedo"]).convert("RGB").save(os.path.join(d, name + ".png"), compress_level=2)
    Image.open(got["lit"]).convert("RGB").save(os.path.join(d, name + "_lit.png"), compress_level=2)
    v, meta = encode(got["h"])
    meta["png"] = name + "_h.png"
    save_height(os.path.join(d, meta["png"]), v)
    self.drop(got)
    albedo = np.asarray(Image.open(os.path.join(d, name + ".png")))
    side = {"name": name, "cell": [t["i"], t["j"]], "level": t["level"], "x": got["x"], "y": got["y"], "grid_xy": [t["x"], t["y"]],
            "cm_per_px": ortho.CM_PER_PX, "size": [W_PX, H_PX], "extent_m": [W_M, H_M], "world_to_px": ortho.world_to_px(got["x"], got["y"]),
            "north_up": True, "time": time.time(), "kind": "under", "camera": {"cam_z": got["cam_z"], "near": U_NEAR, "far": U_FAR, "topcam": got["topcam"]},  # noqa: TID251
            "road": {"z_lo": t["z_lo"], "z_hi": t["z_hi"], "crossings": t["crossings"], "segs": t["segs"]}, "clearance": cl, "checks": ch,
            "best_attempt": k, "attempts": attempts, "pass": ch["pass"], "quality": ortho.quality(albedo), "height": meta,
            "images": {"albedo": name + ".png", "lit": name + "_lit.png", "height": meta["png"]}}
    write_json(os.path.join(d, name + ".json"), side)

  def run(self) -> None:
    a, log, sh = self.args, self.log, self.sh
    tiles = json.load(open(a.plan))["tiles"]
    if a.names:
      want = set(open(a.names).read().split())
      tiles = [t for t in tiles if t["name"] in want]
    todo = [t for t in tiles if not sh.done(t["name"])]
    log(f"ortho under: {len(todo)} of {len(tiles)} cell-levels to shoot")
    if not todo:
      return
    deadline = until_deadline(a)
    sh.setup(todo[0]["x"], todo[0]["y"])
    time.sleep(3.0)
    ref = ortho.Reference(sh, {"name": "ref", "i": 0, "j": 0, "x": a.ref[0], "y": a.ref[1]}, log) if a.ref else None
    t0 = time.monotonic()
    shot, passed, retried, failed, prev = 0, 0, 0, [], None
    for t in todo:
      if ref:
        due = time.monotonic() >= ref.next
        if due:  # the reference is a normal (albedo) tile: the normal box, then back
          sh.use("albedo")
          sh.send({"type": "ortho", "op": "box", "on": 1, "fov": FOV, "w": W_M, "h": H_M, "near": NEAR, "far": FAR}, sh.topcam(a.ref[0], a.ref[1]))
        if not ref.check():
          log("ortho: stopping: the reference tile keeps coming out different")
          break
        if due:
          prev = None
      if deadline and time.time() > deadline:  # noqa: TID251
        log(f"ortho: time's up after {shot} cell-levels")
        break
      if shot % 20 == 0 and ortho.free_gb(a.run_dir) < a.stop_free_gb:
        log(f"ortho: stopping: {ortho.free_gb(a.run_dir):.1f} GB free")
        break
      hop = 1e4 if prev is None else math.hypot(t["x"] - prev["x"], t["y"] - prev["y"])
      wait = 2.0 if hop < 1 else 4.0 if hop < 600 else 7.0
      r = self.shoot(t, wait)
      prev = t
      if r is None:
        failed.append(t["name"])
        log(f"ortho: {t['name']}: no shot")
        continue
      shot += 1
      passed += r["pass"]
      retried += r["attempts"] > 1
      log(f"ortho: {t['name']}: {'pass' if r['pass'] else 'FAIL'} after {r['attempts']} attempt(s)")
      if shot % 20 == 0:
        log(f"ortho: {shot} shot, {passed} pass, {retried} retried, {(time.monotonic() - t0) / shot:.1f} s each, {len(failed)} failed")
    log("ortho under: " + json.dumps({"shot": shot, "pass": passed, "retried": retried, "failed": failed,
                                      "s_per_level": round((time.monotonic() - t0) / max(shot, 1), 1)}))
    report(a.run_dir)


def report(run_dir: str) -> None:
  """failing.txt (names, checks) and failing.jpg (albedo thumbnails, labelled) of the shots that didn't pass."""
  d = os.path.join(run_dir, "tiles")
  rows = [json.load(open(os.path.join(d, f))) for f in sorted(os.listdir(d)) if f.endswith(".json")]
  bad = [s for s in rows if not s.get("pass")]
  with open(os.path.join(run_dir, "failing.txt"), "w") as f:
    f.write(f"{len(bad)} of {len(rows)} under-deck shots failing\n")
    for s in bad:
      c, cl = s["checks"], s["clearance"]
      f.write(f"{s['name']} a={c['a']}({c['a_median_m']} m, {c['a_share_015']}) b={c['b']}({c['b_coverage']}) c={c['c']}({c['c_above_road']}, " +
              f"{c['c_near_cam']}, nodata {c['nodata']}) d={c['d']}({c.get('d_share_01')}) attempts {len(s['attempts'])} " +
              f"camera {s['camera']['cam_z']:.1f} clearance {cl['range']} inside {cl['points_inside']}/{cl['rays']} measured {cl['measured']}\n")
  if bad:
    tw, th, cols = 384, 216, 5
    sheet = Image.new("RGB", (tw * cols, (th + 14) * math.ceil(min(len(bad), 200) / cols)), (40, 40, 40))
    dr = ImageDraw.Draw(sheet)
    for k, s in enumerate(bad[:200]):
      im = Image.open(os.path.join(d, s["images"]["albedo"])).convert("RGB")
      im.thumbnail((tw, th))
      x, y = (k % cols) * tw, (k // cols) * (th + 14)
      sheet.paste(im, (x, y + 14))
      dr.text((x + 2, y + 1), s["name"], fill=(255, 255, 0))
    sheet.save(os.path.join(run_dir, "failing.jpg"), quality=85)
  print(f"{len(bad)} of {len(rows)} failing -> {run_dir}/failing.txt")


def main():
  ap = argparse.ArgumentParser()
  sub = ap.add_subparsers(dest="cmd", required=True)
  for cmd in ("depth", "under"):
    b = sub.add_parser(cmd)
    b.add_argument("run_dir")
    b.add_argument("plan")
    b.add_argument("--albedo", help="depth: the albedo tiles (their sidecars' cameras)")
    b.add_argument("--depth", help="under: the height tiles (check d)")
    b.add_argument("--names", help="a file of tile names: only these")
    b.add_argument("--minutes", type=float)
    b.add_argument("--until", help="HH:MM: stop at the next such time")
    b.add_argument("--end", type=float, help="stop at this time (epoch s)")
    b.add_argument("--stop-free-gb", type=float, default=30.0)
    b.add_argument("--ref", type=float, nargs=2, help="x y of the reference tile (inland)")
  p = sub.add_parser("underplan")
  p.add_argument("paths")
  p.add_argument("out")
  r = sub.add_parser("report")
  r.add_argument("run_dir")
  args = ap.parse_args()
  if args.cmd == "underplan":
    underplan(args)
  elif args.cmd == "report":
    report(args.run_dir)
  else:
    run_dir_log = make_log(args.run_dir)
    (DepthPass if args.cmd == "depth" else UnderPass)(args, run_dir_log).run()


if __name__ == "__main__":
  main()
