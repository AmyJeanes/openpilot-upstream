#!/usr/bin/env python3
"""Shoots the same tiles under several lighting settings, to choose the one whose shadows hide the least paint.

  trial.py shoot TRIAL PLAN NAME... [--under UNDERPLAN NAME...]   each tile under each of CONDITIONS (needs the game, as
                                                                  capture does), into TRIAL/<condition>/tiles
  trial.py compare TRIAL                                         per tile and condition: the share of the road in
                                                                  shadow, of the map's lines whose paint is seen, and the
                                                                  paint's contrast; TRIAL/compare.jpg side by side

The scene and the shadow settings go back to what they were at the end (capture.Shooter.finish, shadows reset=1)."""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image, ImageDraw

from openpilot.tools.sim.bridge.gta5.map.imgcheck import capture, compare, render, report, tiles
from openpilot.tools.sim.bridge.gta5.map.imgcheck.frame import MapData, load_image

WORLD = {"type": "world", "minute": 0, "freeze": 1, "rain": 0}
CONDITIONS = {
  "a_sunny_13": ({**WORLD, "hour": 13, "weather": "EXTRASUNNY"}, None),
  "b_sunny_12": ({**WORLD, "hour": 12, "weather": "EXTRASUNNY"}, None),
  "c_clouds_13": ({**WORLD, "hour": 13, "weather": "CLOUDS"}, None),
  "c_overcast_13": ({**WORLD, "hour": 13, "weather": "OVERCAST"}, None),
  "d_soft_samples": ({**WORLD, "hour": 13, "weather": "EXTRASUNNY"}, {"sample": "CSM_ST_SOFT16"}),
  "d_bounds_0": ({**WORLD, "hour": 13, "weather": "EXTRASUNNY"}, {"bounds": 0.0}),
  "d_bounds_10": ({**WORLD, "hour": 13, "weather": "EXTRASUNNY"}, {"bounds": 10.0}),
  "d_aircraft": ({**WORLD, "hour": 13, "weather": "EXTRASUNNY"}, {"aircraft": 1}),
}


def shoot(trial: str, plan: list[tiles.Tile], under: list[tiles.Tile], log=print):
  sh = capture.Shooter(os.path.join(trial, "a_sunny_13"), log=log)
  sh.start(plan[0])
  try:
    for cond, (world, shadows) in CONDITIONS.items():
      sh.tile_dir = os.path.join(trial, cond, "tiles")
      sh.raw_dir = os.path.join(trial, cond, "raw")
      os.makedirs(sh.tile_dir, exist_ok=True)
      os.makedirs(sh.raw_dir, exist_ok=True)
      sh.raw_win = capture.win_path(sh.raw_dir).replace("\\", "/")
      sh.send({"type": "shadows", "reset": 1})
      sh.send(world, capture.SCENE_TRAFFIC)
      if shadows:
        sh.send({"type": "shadows", **shadows})
      time.sleep(3.0)
      st = sh.wait_state()
      log(f"imgcheck trial: {cond}: world {st.get('world')}, shadows {st.get('shadows')}")
      for t in plan + (under if cond in ("a_sunny_13", "c_overcast_13") else []):
        sh.send({"type": "topcam", "on": 1, "x": t.x, "y": t.y, "z": t.z, "heading": t.heading, "height": t.height, "fov": t.fov,
                 "hud": 0, "abs": int(t.under)})
        time.sleep(2.5)
        st = sh.wait_state()
        bmp = sh.grab(t.name)
        if bmp is None:
          log(f"imgcheck trial: {cond} {t.name}: no grab")
          continue
        sh.saved.put((bmp, {"name": t.name, "source": "capture", "time": time.time(), "target": t.__dict__, "topcam": st["topcam"],
                            "wait": 2.5, "scene": {**capture.scene_of(st), "shadows": st.get("shadows")}, "zmap": t.z,
                            "condition": cond}))
      sh.saved.join()
  finally:
    sh.send({"type": "shadows", "reset": 1})
    sh.finish()


def metrics(run_dir: str, side: dict, md: MapData) -> dict:
  img = load_image(report.image_path(run_dir, side))
  cam = report.tile_camera({**side, "size": [img.shape[1], img.shape[0]]})
  tc = compare.TileCheck(img, cam, md)
  p = tc.paint
  road = tc.map.surface(tc.size) & ~tc.hidden
  v = p.v[road]
  lit = np.percentile(v, 90) if len(v) else 1.0
  shadow = float((v < 0.55 * lit).mean()) if len(v) else 0.0
  kn, kb = tc.px(2 * compare.TOL) | 1, tc.px(compare.BG_M) | 1
  near = compare.box_sum(p.paint, kn) / kn ** 2
  bg = compare.box_sum(p.paint, kb) / kb ** 2
  painted = (near >= compare.LINE_DENSITY) & (near >= compare.BG_RATIO * bg)
  seen, total = 0, 0
  for kind, pts in tc.map.lines("dwcy"):
    q, t, uv, iu, iv, judged = tc.samples(pts, 0.5)
    if kind in "dy":  # dashed: count a 9 m window as seen when any of it is
      n = 18
      for a in range(0, len(q) - n + 1, n):
        if judged[a:a + n].mean() > 0.6:
          total += 1
          seen += bool(painted[iv[a:a + n], iu[a:a + n]].any())
    else:
      total += int(judged.sum())
      seen += int((painted[iv, iu] & judged).sum())
  lines = np.asarray(tc.map.draw(tc.size, tc.map.lines("dwcy"), 0.15)) > 0
  on = lines & p.paint & road
  contrast = float(np.median(p.v[on]) - np.median(v)) if on.any() and len(v) else 0.0
  return {"shadow": round(shadow, 3), "lines_seen": round(seen / max(total, 1), 3), "contrast": round(contrast, 1),
          "road_v": round(float(np.median(v)) if len(v) else 0.0, 1), "issues": len(tc.run())}


def compare_trial(trial: str, log=print):
  md = MapData()
  report._md = md
  conds = [c for c in CONDITIONS if os.path.isdir(os.path.join(trial, c, "tiles"))]
  names = sorted({f[:-5] for c in conds for f in os.listdir(os.path.join(trial, c, "tiles")) if f.endswith(".json")})
  table = {}
  for c in conds:
    for n in names:
      path = os.path.join(trial, c, "tiles", n + ".json")
      if os.path.exists(path):
        with open(path) as f:
          table[(c, n)] = metrics(os.path.join(trial, c), json.load(f), md)
  with open(os.path.join(trial, "compare.json"), "w") as f:
    json.dump({f"{c}/{n}": m for (c, n), m in table.items()}, f, indent=1)
  for c in conds:
    rows = [table[(c, n)] for n in names if (c, n) in table]
    log(f"{c:16s} " + " ".join(f"{k} {np.mean([r[k] for r in rows]):.3f}" for k in ("shadow", "lines_seen", "contrast", "road_v", "issues")))
  # side by side: a row per tile, a column per condition, the middle of each frame
  W, H = 480, 270
  sheet = Image.new("RGB", (W * len(conds), (H + 20) * len(names) + 24), (16, 16, 16))
  dr = ImageDraw.Draw(sheet)
  f = render.font(14)
  for j, c in enumerate(conds):
    dr.text((j * W + 4, 4), c, fill=(255, 255, 0), font=f)
  for i, n in enumerate(names):
    for j, c in enumerate(conds):
      path = os.path.join(trial, c, "tiles", n + ".jpg")
      if not os.path.exists(path):
        continue
      im = Image.open(path)
      w, h = im.size
      im = im.crop((w // 4, h // 4, 3 * w // 4, 3 * h // 4)).resize((W, H))
      y = 24 + i * (H + 20)
      sheet.paste(im, (j * W, y))
      m = table.get((c, n), {})
      dr.text((j * W + 4, y + H + 2), f"{n} shadow {m.get('shadow', 0):.2f} seen {m.get('lines_seen', 0):.2f} contrast {m.get('contrast', 0):.0f}",
              fill=(220, 220, 220), font=f)
  sheet.save(os.path.join(trial, "compare.jpg"), quality=85)
  log(f"imgcheck trial: {os.path.join(trial, 'compare.jpg')}")


def main(argv=None):
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("command", choices=["shoot", "compare"])
  ap.add_argument("trial")
  ap.add_argument("plan", nargs="?")
  ap.add_argument("names", nargs="*")
  ap.add_argument("--under", nargs="*", default=[], help="UNDERPLAN then tile names in it")
  a = ap.parse_args(argv)
  if a.command == "shoot":
    plan, _ = tiles.load(a.plan)
    pick = [t for t in plan if t.name in set(a.names)]
    under = []
    if a.under:
      up, _ = tiles.load(a.under[0])
      under = [t for t in up if t.name in set(a.under[1:])]
      for t in under:
        t.name = "u" + t.name[1:]
    shoot(a.trial, pick, under)
  else:
    compare_trial(a.trial)


if __name__ == "__main__":
  main()
