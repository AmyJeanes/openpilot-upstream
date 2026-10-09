"""Shoots a plan's tiles in the game: the plugin's topcam over each tile, then its grab of the player's frame.

The grab (core.cpp `grab`, present_hook.cpp RequestGrab) copies the player's next presented frame on the GPU and
writes it as a BMP, so nothing on the desktop can get in the way; it needs the present hook, which runs while the bridge
is connected and the player is in a car (the camera is the plugin's then, and the topcam shows on the player's frames).
`--desktop` falls back to a screen capture, taken only while the game is the foreground window.

Before the first tile, and whenever the plugin's state shows them drifting (a core reload, a bridge restart), the scene
is set: noon (13:00), EXTRASUNNY, the clock frozen, cast shadows off (trial.py: they hide the paint), no traffic,
pedestrians or parked cars, then the camera's focus is
moved away and back so cars already there go; each tile's sidecar records the scene its state showed. The game streams
the world around the camera's focus (SET_FOCUS_POS_AND_VEL away from the player), so after a hop the shot waits
tiles.wait_for(hop); a frame that comes out blurry or blank (unstreamed LOD) is shot again at the end after a longer
wait. Resumable: tiles with a sidecar are skipped. At the end the topcam goes off and the scene settings the game had
are restored."""
import json
import math
import os
import queue
import subprocess
import threading
import time

import numpy as np
from PIL import Image

from openpilot.tools.sim.bridge.gta5 import gta5_cmd
from openpilot.tools.sim.bridge.gta5.map.imgcheck.paint import Paint, downscale
from openpilot.tools.sim.bridge.gta5.map.imgcheck.tiles import FOV, HEIGHT, Tile, wait_for

SCENE_WORLD = {"type": "world", "hour": 13, "minute": 0, "weather": "EXTRASUNNY", "freeze": 1, "rain": 0}
SCENE_TRAFFIC = {"type": "traffic", "on": 0, "vehicles": 0, "peds": 0, "parked": 0}
SCENE_SHADOWS = {"type": "shadows", "bounds": 0.0}  # cascade shadow bounds 0: no cast shadows
SCENE_DEBUG = {"type": "debug", "on": 0}  # a restarted bridge turns the map debug overlay (and its status text) back on
GRAB_WAIT = 6.0  # s for the plugin to write a grab
RETRY_WAITS = (4.0, 8.0)  # s before shooting an unloaded tile again
RECONNECT_WAIT = 300.0  # s a bridge restart may take
SHARP_MIN = 6.0  # compare.unloaded's


def win_path(path: str) -> str:
  return subprocess.run(["wslpath", "-w", path], capture_output=True, text=True, check=True).stdout.strip()


def scene_ok(state: dict) -> tuple[bool, str]:
  w, d = state.get("world") or {}, state.get("density") or {}
  bad = []
  if w.get("hour") not in (12, 13, 14):
    bad.append(f"hour {w.get('hour')}")
  if w.get("weather") != "EXTRASUNNY" or (w.get("mix") not in (None, 0, 0.0, 1, 1.0) and w.get("from") != w.get("to")):
    bad.append(f"weather {w.get('weather')} ({w.get('from')}->{w.get('to')} {w.get('mix')})")
  if not w.get("frozen"):
    bad.append("clock running")
  if (w.get("rain") or 0) > 0.01:
    bad.append(f"rain {w.get('rain')}")
  if not d.get("set") or any((d.get(k) or 0) > 0 for k in ("vehicles", "peds", "parked")):
    bad.append(f"traffic {d}")
  sh = state.get("shadows")
  if sh is not None and not (sh.get("set") and sh.get("bounds") == 0):
    bad.append(f"shadows {sh}")
  if (state.get("debug") or {}).get("on"):
    bad.append("map debug overlay on")
  return not bad, ", ".join(bad)


def scene_of(state: dict) -> dict:
  return {"world": state.get("world"), "density": state.get("density")}


class Shooter:
  def __init__(self, run_dir: str, desktop: bool = False, jpeg: int | None = 94, log=print):
    self.run_dir = run_dir
    self.tile_dir = os.path.join(run_dir, "tiles")
    self.raw_dir = os.path.join(run_dir, "raw")
    os.makedirs(self.tile_dir, exist_ok=True)
    os.makedirs(self.raw_dir, exist_ok=True)
    self.raw_win = win_path(self.raw_dir).replace("\\", "/")  # the plugin reads its JSON without unescaping
    self.desktop = desktop
    self.jpeg = jpeg
    self.log = log
    self.saved: queue.Queue = queue.Queue()
    self.results: dict[str, dict] = {}
    self.worker = threading.Thread(target=self._save_loop, daemon=True)
    self.worker.start()
    self.initial: dict | None = None
    self.last_scene_check = 0.0

  # ------------------------------------------------------------------------------------------------ the game
  def send(self, *cmds: dict) -> None:
    """Commands through the bridge, waiting out a bridge restart (its debug port refuses meanwhile)."""
    end = time.monotonic() + RECONNECT_WAIT
    while True:
      try:
        return gta5_cmd.send(*cmds)
      except OSError as e:
        if time.monotonic() > end:
          raise RuntimeError(f"imgcheck: the bridge has been unreachable for {RECONNECT_WAIT:.0f} s: {e}") from e
        self.log(f"imgcheck: bridge unreachable ({e}); waiting")
        time.sleep(3.0)

  def state(self, wait: float = 3.0) -> dict | None:
    """The plugin's next state; None while the bridge or the game's connection to it is down."""
    try:
      return gta5_cmd.read_state(f"/tmp/imgcheck_state_{os.getpid()}.json", wait=wait)
    except OSError:
      return None

  def wait_state(self) -> dict:
    end = time.monotonic() + RECONNECT_WAIT
    while (st := self.state()) is None:
      if time.monotonic() > end:
        raise RuntimeError("imgcheck: no state from the plugin: is the bridge running and the game connected?")
      self.log("imgcheck: no state from the plugin; waiting for the bridge")
      time.sleep(3.0)
    return st

  def set_scene(self, near: tuple[float, float] | None = None) -> dict:
    self.send(SCENE_WORLD, SCENE_TRAFFIC, SCENE_SHADOWS, SCENE_DEBUG)
    time.sleep(1.0)
    if near is not None:  # the focus away and back, so the cars that were there go
      self.send({"type": "topcam", "on": 1, "x": near[0] + 900.0, "y": near[1] + 900.0, "height": HEIGHT, "fov": FOV, "hud": 0})
      time.sleep(3.0)
      self.send({"type": "topcam", "on": 1, "x": near[0], "y": near[1], "height": HEIGHT, "fov": FOV, "hud": 0})
      time.sleep(3.0)
    for _ in range(5):
      st = self.wait_state()
      ok, why = scene_ok(st)
      if ok:
        return st
      self.log(f"imgcheck: scene not set yet: {why}")
      self.send(SCENE_WORLD, SCENE_TRAFFIC, SCENE_SHADOWS, SCENE_DEBUG)
      time.sleep(1.5)
    raise RuntimeError("imgcheck: the game won't take the scene (noon, sunny, no traffic)")

  def start(self, first: Tile) -> None:
    st = self.wait_state()
    self.initial = {**scene_of(st), "debug": st.get("debug")}
    stamp = int(time.time())  # noqa: TID251  (a wall clock name)
    with open(os.path.join(self.run_dir, f"scene_before_{stamp}.json"), "w") as f:
      json.dump({**self.initial, "topcam": st.get("topcam")}, f, indent=1)
    self.set_scene((first.x, first.y))

  def finish(self) -> None:
    self.send({"type": "topcam", "on": 0}, {"type": "shadows", "reset": 1})
    if not self.initial:
      return
    w, d = self.initial.get("world") or {}, self.initial.get("density") or {}
    dbg = self.initial.get("debug") or {}
    cmds = []
    if dbg:  # the overlay as it was (the bridge keeps what it's sent and sets it again on a reconnect)
      cmds.append({"type": "debug", "on": int(bool(dbg.get("on"))), **{k: dbg[k] for k in ("layers", "grow", "width", "dist") if k in dbg}})
    if w:
      cmds.append({"type": "world", "hour": w.get("hour", 13), "minute": w.get("minute", 0), "freeze": int(bool(w.get("frozen")))})
      if w.get("set"):
        cmds[-1]["weather"] = w["set"]
      else:
        cmds[-1]["clear"] = 1
      if w.get("rainSet") is not None:
        cmds[-1]["rain"] = w["rainSet"]
    if d:
      cmd = {"type": "traffic", "on": int(not d.get("off"))}
      if d.get("set"):
        cmd.update({k: d[k] for k in ("vehicles", "random", "parked", "peds", "scenario") if k in d})
      else:
        cmd["reset"] = 1
      cmds.append(cmd)
    self.send(*cmds)
    self.log(f"imgcheck: topcam off; scene restored: {json.dumps(cmds)}")

  # ------------------------------------------------------------------------------------------------ one tile
  def grab(self, name: str) -> str | None:
    bmp = os.path.join(self.raw_dir, name + ".bmp")
    if os.path.exists(bmp):
      os.remove(bmp)
    if self.desktop:
      return bmp if desktop_shot(win_path(self.raw_dir) + "\\" + name + ".bmp") else None
    self.send({"type": "grab", "path": self.raw_win + "/" + name + ".bmp"})
    end = time.monotonic() + GRAB_WAIT
    while time.monotonic() < end:
      try:
        size = os.path.getsize(bmp)
        with open(bmp, "rb") as f:
          head = f.read(30)
        start, bpp = int.from_bytes(head[10:14], "little"), int.from_bytes(head[28:30], "little")
        w, h = int.from_bytes(head[18:22], "little", signed=True), abs(int.from_bytes(head[22:26], "little", signed=True))
        if w > 0 and h > 0 and size >= start + (bpp * w + 31) // 32 * 4 * h:  # written whole
          return bmp
      except (OSError, ValueError):
        pass
      time.sleep(0.05)
    return None

  def shoot(self, t: Tile, wait: float, scene_every: float = 60.0) -> bool:
    self.send({"type": "topcam", "on": 1, "x": t.x, "y": t.y, "z": t.z, "heading": t.heading, "height": t.height, "fov": t.fov, "hud": 0,
               "abs": int(t.under)})
    time.sleep(wait)
    st = self.state()
    if st is None or not isinstance(st.get("topcam"), dict):
      self.log(f"imgcheck: {t.name}: no state")
      return False
    tc = st["topcam"]
    if not tc.get("on") or abs(tc["x"] - t.x) > 0.05 or abs(tc["y"] - t.y) > 0.05:
      self.log(f"imgcheck: {t.name}: topcam not there yet ({tc})")
      return False
    ok, why = scene_ok(st)
    if not ok:  # a core reload or bridge restart resets the scene
      self.log(f"imgcheck: scene drifted ({why}); setting it again")
      self.set_scene((t.x, t.y))
      return False
    bmp = self.grab(t.name)
    if bmp is None:
      self.log(f"imgcheck: {t.name}: no grab")
      return False
    stamp = time.time()  # noqa: TID251  (sidecars carry the wall clock time)
    side = {"name": t.name, "source": "capture", "time": stamp, "target": t.__dict__, "topcam": tc, "wait": wait,
            "scene": scene_of(st), "car": st.get("pos"), "zmap": t.z}
    self.saved.put((bmp, side))
    return True

  def _save_loop(self):
    while True:
      bmp, side = self.saved.get()
      try:
        im = Image.open(bmp).convert("RGB")
        arr = np.asarray(im)
        small = downscale(arr, 2)
        m_per_px = 2 * side["topcam"]["height"] * math.tan(math.radians(side["topcam"]["fov"]) / 2) / small.shape[0]
        p = Paint(small, m_per_px)
        side["quality"] = {"sharp": round(p.sharpness(), 2), "blank": p.blank, "std": round(float(np.std(p.v)), 1)}
        ext = ".jpg" if self.jpeg else ".png"
        out = os.path.join(self.tile_dir, side["name"] + ext)
        im.save(out, quality=self.jpeg) if self.jpeg else im.save(out)
        side["image"] = os.path.basename(out)
        side["size"] = [im.width, im.height]
        with open(os.path.join(self.tile_dir, side["name"] + ".json"), "w") as f:
          json.dump(side, f, indent=1)
        os.remove(bmp)
        self.results[side["name"]] = side["quality"]
      except Exception as e:  # a torn file: the tile is shot again on the next run
        self.log(f"imgcheck: {side['name']}: save failed: {type(e).__name__}: {e}")
        self.results[side["name"]] = {"error": str(e)}
      finally:
        self.saved.task_done()

  def done(self, name: str) -> bool:
    return os.path.exists(os.path.join(self.tile_dir, name + ".json"))


def unloaded(q: dict) -> bool:
  return bool(q.get("error")) or q.get("blank", False) or q.get("sharp", 99) < SHARP_MIN


def run(run_dir: str, tiles: list[Tile], minutes: float | None = None, desktop: bool = False, jpeg: int | None = 94, log=print) -> dict:
  sh = Shooter(run_dir, desktop, jpeg, log)
  todo = [t for t in tiles if not sh.done(t.name)]
  log(f"imgcheck: {len(todo)} of {len(tiles)} tiles to shoot")
  if not todo:
    return {"shot": 0}
  t0 = time.monotonic()
  sh.start(todo[0])
  shot, failed, retry = 0, [], []
  prev = None
  try:
    for t in todo:
      if minutes and time.monotonic() - t0 > minutes * 60:
        log(f"imgcheck: time's up after {shot} tiles")
        break
      hop = math.hypot(t.x - prev.x, t.y - prev.y) if prev is not None else 1e4
      ok = sh.shoot(t, wait_for(hop))
      if not ok:  # a bridge restart or core reload: wait for the state, put the scene back, and try again
        sh.wait_state()
        sh.set_scene()
        ok = sh.shoot(t, max(2.0, wait_for(hop)))
      prev = t
      if not ok:
        failed.append(t.name)
        continue
      shot += 1
      if shot % 25 == 0:
        rate = (time.monotonic() - t0) / shot
        log(f"imgcheck: {shot} shot, {rate:.2f} s a tile, {len(failed)} failed")
    sh.saved.join()
    retry = [t for t in todo if (sh.results.get(t.name) and unloaded(sh.results[t.name])) or t.name in failed]
    for wait in RETRY_WAITS:
      if not retry:
        break
      log(f"imgcheck: {len(retry)} tiles looked unloaded; shooting them again after {wait:.0f} s")
      for t in retry:
        sh.shoot(t, wait)
      sh.saved.join()
      retry = [t for t in retry if unloaded(sh.results.get(t.name, {"error": "not shot"}))]
    failed = [n for n in failed if not sh.done(n)]
  finally:
    sh.saved.join()
    sh.finish()
  out = {"shot": shot, "failed": failed, "still_unloaded": [t.name for t in retry], "s_per_tile": round((time.monotonic() - t0) / max(shot, 1), 2)}
  log(f"imgcheck: {json.dumps(out)}")
  return out


FOREGROUND_PS1 = r"""
param([string]$Out)
Add-Type @"
using System; using System.Runtime.InteropServices;
public class Fg { [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] public static extern int GetWindowThreadProcessId(IntPtr h, out int pid); }
"@
$pid0 = 0; [void][Fg]::GetWindowThreadProcessId([Fg]::GetForegroundWindow(), [ref]$pid0)
$p = Get-Process -Id $pid0 -ErrorAction SilentlyContinue
if (-not $p -or $p.ProcessName -notlike "GTA5*") { Write-Output "notgame $($p.ProcessName)"; exit 2 }
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
$bmp.Save($Out, [System.Drawing.Imaging.ImageFormat]::Bmp)
$g.Dispose(); $bmp.Dispose()
"""


def desktop_shot(out_win: str) -> bool:
  """A screen capture, only while the game is the foreground window (else nothing: the caller retries)."""
  ps1 = os.path.join("/tmp", "imgcheck_shot.ps1")
  with open(ps1, "w") as f:
    f.write(FOREGROUND_PS1)
  r = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", win_path(ps1), "-Out", out_win],
                     capture_output=True, text=True)
  return r.returncode == 0
