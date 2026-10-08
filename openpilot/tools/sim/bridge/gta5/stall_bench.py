#!/usr/bin/env python3
"""The bridge's stalls, offline: recorded trips (nav_replay.py's inputs) through GTA5World.read_sensors with the debug
overlay on, as nav_replay replays them (the game, openpilot and the router stubbed, routing on the caller's thread).

  stall_bench.py realtime TRIPS --map DIR --cache FILE   the bridge's threads on the real clock: the main loop at 100 Hz,
      a frame thread at 20 Hz (GTA5_LOG's json.dumps) waking a camera thread that copies two NV12 frames, a 100 Hz car
      thread, and the overlay's worker. Prints per trip the main loop's step times and gaps, the camera's and the car
      thread's gaps, the garbage collections and the slow calls on the main thread (ms).
  stall_bench.py ident TRIPS --map DIR --cache FILE --out DIR   fake clock, the overlay kept in step: a hash of each
      overlay message and lane line, to check two trees give the same (zcat and diff the outputs).

PYTHONPATH picks the tree, so an older one can be measured with this script. STALL_WATCHDOG=<s> dumps every thread's
stack when the camera thread waits longer (faulthandler's own thread: it shows whoever holds the GIL)."""
import argparse
import contextlib
import faulthandler
import gc
import gzip
import hashlib
import io
import json
import os
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("GTA5_NOO", "on")
os.environ.setdefault("GTA5_DEBUG", "1")

from openpilot.tools.sim.bridge.gta5 import gta5_overlay, gta5_world, nav_replay
from openpilot.tools.sim.bridge.gta5.map import router as router_mod

NV12 = 1928 * 1208 * 3 // 2
WATCHDOG = float(os.getenv("STALL_WATCHDOG", "0"))


def load(path: str) -> tuple[dict, list]:
  with gzip.open(path, "rt") as f:
    head = json.loads(f.readline())
    return head, [json.loads(line) for line in f]


def make_world(head: dict, mp, rec: dict, wait_maps: bool = False):
  """nav_replay's world with the real overlay and GPS route, and the nav messages built (not sent), as Amy's bridge_env."""
  from openpilot.tools.sim.bridge.gta5.gta5_nav_msgs import NavMessages
  w, _ = nav_replay._world(head, mp, rec)
  del w.__dict__["_overlay"]
  process = getattr(gta5_overlay, "OverlayProcess", None)
  w.overlay = process(wait_maps=wait_maps) if process is not None and gta5_overlay.PROCESS else gta5_overlay.Overlay()
  w.gps = gta5_overlay.GpsRoute()
  w.nav_msgs = NavMessages(pm=SimpleNamespace(send=lambda *a: None))
  return w


def ms(xs, ps=(50, 90, 99, 99.9)) -> dict:
  a = np.asarray(xs) * 1000
  return {f"p{p}": round(float(np.percentile(a, p)), 1) for p in ps} | {"max": round(float(a.max()), 1), "n": len(a)} if len(a) else {}


def realtime(trip: str, mp, layers: str, seconds: float, skip: float) -> dict:
  head, frames = load(trip)
  rec: dict = {"cmd": [], "q": [], "nd": []}
  w = make_world(head, mp, rec)
  debug = {"on": True, "layers": layers}
  stop, ready = threading.Event(), threading.Condition()
  cur: dict = {"state": frames[0]["state"], "seq": 0, "t": 0.0}
  cam_gaps, car_gaps, gcs, gc_t = [], [], [], {}
  src = bytes(2 * NV12)

  def rx():  # the frame reader: a frame each 50 ms, logged as GTA5_LOG does
    nxt = time.perf_counter()
    while not stop.is_set():
      nxt += 0.05
      time.sleep(max(0.0, nxt - time.perf_counter()))
      json.dumps({"mono": time.monotonic(), "state": cur["state"]})
      with ready:
        cur["seq"], cur["t"] = cur["seq"] + 1, time.perf_counter()
        ready.notify_all()

  def camera():
    handed, last = 0, None
    while not stop.is_set():
      with ready:
        if cur["seq"] == handed:
          ready.wait(0.1)
        handed = cur["seq"]
      for k in range(2):
        bytes(memoryview(src)[k * NV12:(k + 1) * NV12])
      now = time.perf_counter()
      if WATCHDOG:
        faulthandler.dump_traceback_later(WATCHDOG, exit=False, file=sys.stderr)
      if last is not None:
        cam_gaps.append((now, now - last))
      last = now

  def car():
    nxt, last = time.perf_counter(), None
    while not stop.is_set():
      now = time.perf_counter()
      if last is not None:
        car_gaps.append((now, now - last))
      last = now
      nxt += 0.01
      d = nxt - time.perf_counter()
      if d > 0:
        time.sleep(d)
      else:
        nxt = time.perf_counter()

  def on_gc(phase, info):
    if phase == "start":
      gc_t["t"] = time.perf_counter()
    elif "t" in gc_t:
      gcs.append((info["generation"], time.perf_counter() - gc_t.pop("t")))

  slow: dict[str, list] = {"lane_line": [], "route": []}

  def timed(name, fn):
    def f(*a, **kw):
      t = time.perf_counter()
      try:
        return fn(*a, **kw)
      finally:
        slow[name].append(time.perf_counter() - t)
    return f

  lane_line = router_mod.Route.lane_line
  router_mod.Route.lane_line = timed("lane_line", lane_line)
  w.navigator._route = timed("route", w.navigator._route)
  gc.callbacks.append(on_gc)
  threads = [threading.Thread(target=f, daemon=True) for f in (rx, camera, car)]
  for t in threads:
    t.start()
  steps, gaps = [], []
  t0 = time.perf_counter()
  nxt, last, k = t0, None, 0
  buf = io.StringIO()
  try:
    with contextlib.redirect_stdout(buf):
      while (now := time.perf_counter()) - t0 < seconds:
        while k + 1 < len(frames) and frames[k + 1]["mono"] - frames[0]["mono"] <= now - t0:
          k += 1
        if k + 1 >= len(frames):
          break
        fr = frames[k]
        cur["state"] = fr["state"]
        w.state, w.last_frame_time = {**fr["state"], "debug": debug}, time.monotonic()
        w.simulator_state.is_engaged = fr["en"]
        w.model.set(fr["meta"])
        for v in rec.values():
          v.clear()
        w.read_sensors(w.simulator_state)
        if now - t0 > skip:
          steps.append(time.perf_counter() - now)
          if last is not None:
            gaps.append(now - last)
        last = now
        nxt += 0.01
        d = nxt - time.perf_counter()
        if d > 0:
          time.sleep(d)
        else:
          nxt = time.perf_counter()
        buf.seek(0)
        buf.truncate()
  finally:
    stop.set()
    for t in threads:
      t.join(1)
    gc.callbacks.remove(on_gc)
    router_mod.Route.lane_line = lane_line
    if WATCHDOG:
      faulthandler.cancel_dump_traceback_later()
    w.overlay.close() if hasattr(w.overlay, "close") else None
  after = t0 + skip
  return {"trip": head["trip"], "s": round(time.perf_counter() - t0, 1), "step": ms(steps), "gap": ms(gaps),
          "camera_gap": ms([g for t, g in cam_gaps if t > after]), "car_gap": ms([g for t, g in car_gaps if t > after]),
          "gc": {"count": [sum(g == n for g, _ in gcs) for n in range(3)], "max": round(max((d for _, d in gcs), default=0) * 1000, 1)},
          "calls": {name: ms(v, (50, 90)) for name, v in slow.items()}, "overlay": dict(w.overlay.stats)}


def digest(x) -> str:
  return hashlib.sha1(json.dumps(x, sort_keys=True).encode()).hexdigest()[:16]


def ident(trip: str, mp, layers: str, out: str) -> dict:
  head, frames = load(trip)
  rec: dict = {"cmd": [], "q": [], "nd": []}
  w = make_world(head, mp, rec, wait_maps=True)
  nav_replay._patch_clocks()
  gta5_overlay.time = nav_replay.CLOCK
  ov = w.overlay
  wait = getattr(ov, "wait", None)
  if wait is None:
    ov.thread = True  # a tree without the overlay's process: made here instead
  debug = {"on": True, "layers": layers}
  lines, last_line = [], None
  buf = io.StringIO()
  with contextlib.redirect_stdout(buf):
    for i, fr in enumerate(frames):
      nxt = frames[i + 1]["mono"] if i + 1 < len(frames) else fr["mono"] + 0.05
      for j in range(5):  # the main loop's 100 Hz over each 20 Hz state
        t = fr["mono"] + (nxt - fr["mono"]) * j / 5
        nav_replay.CLOCK.t = t
        w.state, w.last_frame_time = {**fr["state"], "debug": debug}, t
        w.simulator_state.is_engaged = fr["en"]
        w.model.set(fr["meta"])
        for v in rec.values():
          v.clear()
        w.read_sensors(w.simulator_state)
        if wait is not None:
          wait()
        elif ov.snap is not None:
          snap, ov.snap = ov.snap, None
          ov.outbox = ov.make(snap)
        step = {}
        if msgs := [m for m in rec["cmd"] if m.get("type") in ("debugGeo", "gpsPoints")]:
          step["m"] = [digest(m) for m in msgs]
        if w.lane_line[2] is not last_line:
          last_line = w.lane_line[2]
          step["ll"] = digest(last_line)
        if step:
          lines.append(json.dumps({"t": round(t, 3), **step}))
        buf.seek(0)
        buf.truncate()
  if hasattr(ov, "close"):
    ov.close()
  with gzip.open(out, "wt") as f:
    f.write("\n".join(lines) + "\n")
  return {"trip": head["trip"], "lines": len(lines)}


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("mode", choices=["realtime", "ident"])
  ap.add_argument("trips", nargs="+")
  ap.add_argument("--map", required=True, help="the map's folder: paths.jsonl and gta5.osm.pbf")
  ap.add_argument("--cache", required=True, help="nav_replay's cached router answers")
  ap.add_argument("--layers", default=gta5_overlay.DEFAULT_LAYERS)
  ap.add_argument("--seconds", type=float, default=100.0)
  ap.add_argument("--skip", type=float, default=20.0, help="s at the start left out of the numbers (routing, the maps loading)")
  ap.add_argument("--out", default=".")
  args = ap.parse_args()
  mp = nav_replay.Map(args.map, None, args.cache)
  t = time.perf_counter()  # the overlay's lines, built into its cache first where it has none, so every run has them
  geometry = gta5_overlay.Overlay(background=False)
  geometry._road_geometry(mp.paths, mp.osm)
  print(f"overlay lines: {geometry.stats.get('geometry')} in {time.perf_counter() - t:.1f} s", file=sys.stderr, flush=True)
  if hasattr(gta5_world, "hold_full_collections"):
    gta5_world.hold_full_collections()  # as GTA5World.__init__
  for trip in args.trips:
    if args.mode == "realtime":
      r = realtime(trip, mp, args.layers, args.seconds, args.skip)
    else:
      os.makedirs(args.out, exist_ok=True)
      r = ident(trip, mp, args.layers, os.path.join(args.out, os.path.basename(trip).replace(".jsonl.gz", ".ident.gz")))
    print(json.dumps(r), flush=True)


if __name__ == "__main__":
  main()
