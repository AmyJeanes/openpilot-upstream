"""Checks a run's tiles, gathers their issues into spots across tiles, and writes the report.

A spot is the issues within LINK m of each other (any kind, any tile). The same place shows in the tiles that overlap
there, so a spot's score is its issues' score summed over the tiles that flagged it, divided by the tiles that saw the
place (inside their judged frame): paint seen once in three views counts a third. Its type is its main kind's, or
`offset` where map lines without paint and paint without a map line of one colour lie together (a line drawn in the
wrong place). report.html ranks them with crops from the tile that sees each nearest its middle: the game, our map
over it as the overlay colours it, and the paint found with the issues; spots.json has them all and heatmap*.png the
city."""
import html
import json
import math
import multiprocessing
import os
import time
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageDraw

from openpilot.tools.sim.bridge.gta5.map.imgcheck import compare, render
from openpilot.tools.sim.bridge.gta5.map.imgcheck.frame import Camera, MapData, load_image
from openpilot.tools.sim.bridge.gta5.map.imgcheck.tiles import CITY

LINK = 8.0  # m
CROP_M = 32.0  # m: a crop's side
TOP = 150  # spots with crops in the report
TYPE_COLOURS = {k: v for k, v in render.ISSUE_COLOURS.items()} | {"offset": (255, 255, 255)}

_md: MapData | None = None


def tile_camera(side: dict) -> Camera:
  tc = side["topcam"]
  w, h = side.get("size", (2560, 1440))
  zmap = side.get("zmap")
  if side.get("source") == "capture" and _md is not None and zmap is not None:
    # the plan's height is roads.json's, to the metre: the marks' (GTA's node heights) are finer
    zmap = _md.road_z((tc["x"], tc["y"]), z_hint=zmap, radius=15.0) or zmap
  return Camera(tc["x"], tc["y"], tc["ground"], tc["height"], tc["heading"], tc["fov"], w, h, float("nan") if zmap is None else zmap)


def image_path(run_dir: str, side: dict) -> str:
  p = side["image"]
  return p if os.path.isabs(p) else os.path.join(run_dir, "tiles", p)


def sidecars(run_dir: str) -> list[dict]:
  d = os.path.join(run_dir, "tiles")
  out = []
  for f in sorted(os.listdir(d)):
    if f.endswith(".json"):
      with open(os.path.join(d, f)) as fh:
        out.append(json.load(fh))
  return out


def _analyse(args) -> str:
  run_dir, side = args
  out = os.path.join(run_dir, "analysis", side["name"] + ".json")
  try:
    img = load_image(image_path(run_dir, side))
    cam = tile_camera({**side, "size": [img.shape[1], img.shape[0]]})
    tc = compare.check(img, cam, _md)
    res = {"name": side["name"], "map_hash": _md.map_hash, "marks": os.path.basename(_md.marks_file), "camera": tc.cam.to_json(), "summary": compare.summary(tc),
           "issues": [] if compare.unloaded(tc) else [i.to_json() for i in tc.issues]}
  except Exception as e:  # one bad tile shouldn't stop the run
    res = {"name": side["name"], "error": f"{type(e).__name__}: {e}"}
  with open(out, "w") as f:
    json.dump(res, f)
  return side["name"]


def analyse(run_dir: str, procs: int = 8, force: bool = False, log=print) -> None:
  global _md
  os.makedirs(os.path.join(run_dir, "analysis"), exist_ok=True)
  _md = MapData()
  todo = []
  for side in sidecars(run_dir):
    out = os.path.join(run_dir, "analysis", side["name"] + ".json")
    if not force and os.path.exists(out):
      with open(out) as f:
        old = json.load(f)
        if old.get("map_hash") == _md.map_hash and old.get("marks") == os.path.basename(_md.marks_file):
          continue
    todo.append((run_dir, side))
  log(f"imgcheck: analysing {len(todo)} tiles against map {_md.map_hash} ({procs} processes)")
  t0 = time.monotonic()
  with multiprocessing.get_context("fork").Pool(procs) as pool:
    for n, _ in enumerate(pool.imap_unordered(_analyse, todo, chunksize=4)):
      if (n + 1) % 200 == 0:
        log(f"imgcheck: {n + 1} tiles, {(time.monotonic() - t0) / (n + 1) * procs:.2f} s a tile a process")


# ---------------------------------------------------------------------------------------------------- spots
def load_analysis(run_dir: str) -> list[dict]:
  d = os.path.join(run_dir, "analysis")
  out = []
  for f in sorted(os.listdir(d)):
    with open(os.path.join(d, f)) as fh:
      out.append(json.load(fh))
  return out


def sees(cam: Camera, xy: np.ndarray) -> np.ndarray:
  """Which points [N, 2] lie inside a tile's judged frame (at the tile's road height)."""
  z = cam.zmap if not math.isnan(cam.zmap) else cam.ground
  uv = cam.project(np.column_stack([xy, np.full(len(xy), z)]))
  b = compare.BORDER * min(cam.w, cam.h)
  return (uv[:, 0] >= b) & (uv[:, 0] < cam.w - b) & (uv[:, 1] >= b) & (uv[:, 1] < cam.h - b)


def cluster(xy: np.ndarray, link: float = LINK) -> np.ndarray:
  """Single-linkage groups of points within `link` m: a group index per point."""
  parent = np.arange(len(xy))

  def find(i):
    while parent[i] != i:
      parent[i] = parent[parent[i]]
      i = parent[i]
    return i
  cells = defaultdict(list)
  for i, (x, y) in enumerate(xy):
    cells[(int(x // link), int(y // link))].append(i)
  for (cx, cy), members in cells.items():
    near = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in cells.get((cx + dx, cy + dy), ())]
    for i in members:
      for j in near:
        if j > i and math.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]) < link:
          a, b = find(i), find(j)
          if a != b:
            parent[a] = b
  return np.array([find(i) for i in range(len(xy))])


def spots(results: list[dict]) -> list[dict]:
  tiles = [r for r in results if "camera" in r and not r["summary"].get("unloaded") and not r["summary"].get("covered")]
  cams = {r["name"]: Camera.from_json(r["camera"]) for r in tiles}
  centres = np.array([[c.x, c.y] for c in cams.values()]) if cams else np.zeros((0, 2))
  names = list(cams)
  issues = [{**i, "tile": r["name"]} for r in tiles for i in r["issues"]]
  if not issues:
    return []
  xy = np.array([[i["x"], i["y"]] for i in issues])
  groups = cluster(xy)
  out = []
  for g in np.unique(groups):
    members = [issues[k] for k in np.flatnonzero(groups == g)]
    pts = np.array([[m["x"], m["y"]] for m in members])
    w = np.array([m["score"] for m in members]) + 1e-6
    c = (pts * w[:, None]).sum(0) / w.sum()
    near = [names[k] for k in np.flatnonzero(np.hypot(*(centres - c).T) < 80)]
    seen = [n for n in near if sees(cams[n], c[None])[0]]
    flagged = sorted({m["tile"] for m in members})
    n_seen = max(len(set(seen) | set(flagged)), 1)
    by_kind = Counter()
    for m in members:
      by_kind[m["kind"]] += m["score"]
    score = sum(by_kind.values()) / n_seen
    kind = by_kind.most_common(1)[0][0]
    colours = {m["colour"] for m in members if m["kind"] in ("missing", "stray") and m["colour"]}
    if by_kind.get("missing", 0) > 0 and by_kind.get("stray", 0) > 0 and \
       any(any(m["kind"] == "missing" and m["colour"] == col for m in members) and
           any(m["kind"] == "stray" and m["colour"] == col for m in members) for col in colours):
      kind = "offset"
    best = min(flagged, key=lambda n: math.hypot(cams[n].x - c[0], cams[n].y - c[1]))
    extent = float(np.ptp(pts[:, 0]) + np.ptp(pts[:, 1])) if len(pts) > 1 else 0.0
    out.append({"x": round(float(c[0]), 1), "y": round(float(c[1]), 1), "type": kind, "score": round(score, 1),
                "kinds": {k: round(v / n_seen, 1) for k, v in by_kind.most_common()}, "seen": n_seen, "flagged": len(flagged),
                "best_tile": best, "extent_m": round(extent, 1),
                "issues": sorted(members, key=lambda m: -m["score"])[:12]})
  out.sort(key=lambda s: -s["score"])
  for n, s in enumerate(out):
    s["rank"] = n + 1
  return out


# ---------------------------------------------------------------------------------------------------- pictures
def _crop(args) -> str | None:
  run_dir, side, spot, out = args
  img = load_image(image_path(run_dir, side))
  cam = tile_camera({**side, "size": [img.shape[1], img.shape[0]]})
  tc = compare.check(img, cam, _md)
  u, v = tc.cam.project(np.array([[spot["x"], spot["y"], tc.map.level]]))[0]
  half = CROP_M / 2 / tc.mpp
  box = (int(u - half), int(v - half), int(u + half), int(v + half))
  game = Image.fromarray(tc.img.clip(0, 255).astype(np.uint8))
  over = render.draw_map(tc, game)
  marks = render.draw_issues(render.masks_view(tc), tc, numbers=False)
  ImageDraw.Draw(marks).ellipse([u - 10, v - 10, u + 10, v + 10], outline=(255, 255, 255), width=2)
  panels = [p.crop(box).resize((420, 420)) for p in (game, over, marks)]
  sheet = Image.new("RGB", (3 * 420 + 8, 420), (24, 24, 24))
  for k, p in enumerate(panels):
    sheet.paste(p, (k * 424, 0))
  sheet.save(out, quality=88)
  return out


def heatmap(spot_list: list[dict], results: list[dict], md: MapData, path: str, box=None, m_per_px: float = 4.0, top: int = 40):
  """Roads in grey, the tiles checked faintly, each spot a disc sized by its score in its type's colour, the top ranks
  numbered."""
  xs = np.concatenate([w[3][:, 0] for w in md.ways])
  ys = np.concatenate([w[3][:, 1] for w in md.ways])
  x0, y0, x1, y1 = box or (xs.min() - 50, ys.min() - 50, xs.max() + 50, ys.max() + 50)
  W, H = int((x1 - x0) / m_per_px), int((y1 - y0) / m_per_px)
  im = Image.new("RGB", (W, H), (16, 18, 22))
  dr = ImageDraw.Draw(im, "RGBA")

  def px(x, y):
    return (x - x0) / m_per_px, (y1 - y) / m_per_px
  for cls, _, _, pts, width, _ in md.ways:
    q = [px(x, y) for x, y in pts[:, :2]]
    dr.line(q, fill=(70, 74, 82) if cls < 5 else (48, 50, 56), width=max(1, int(width / m_per_px)))
  for r in results:
    if "camera" in r:
      cx, cy = px(r["camera"]["x"], r["camera"]["y"])
      dr.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(40, 120, 200, 70))
  for s in sorted(spot_list, key=lambda s: s["score"]):
    cx, cy = px(s["x"], s["y"])
    r = 2 + 1.6 * math.sqrt(s["score"])
    col = TYPE_COLOURS.get(s["type"], (255, 255, 255))
    dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col + (150,), outline=col + (255,))
  f = render.font(13)
  for s in spot_list[:top]:
    cx, cy = px(s["x"], s["y"])
    dr.text((cx + 6, cy - 6), str(s["rank"]), fill=(255, 255, 255), font=f)
  y = 8
  for k, col in TYPE_COLOURS.items():
    dr.ellipse([8, y, 20, y + 12], fill=col)
    dr.text((26, y - 2), k, fill=(230, 230, 230), font=f)
    y += 18
  im.save(path)


def write(run_dir: str, procs: int = 8, top: int = TOP, log=print) -> str:
  global _md
  _md = _md or MapData()
  results = load_analysis(run_dir)
  spot_list = spots(results)
  rep = os.path.join(run_dir, "report")
  os.makedirs(os.path.join(rep, "crops"), exist_ok=True)
  sides = {s["name"]: s for s in sidecars(run_dir)}
  jobs = [(run_dir, sides[s["best_tile"]], s, os.path.join(rep, "crops", f"spot{s['rank']:04d}.jpg")) for s in spot_list[:top]]
  with multiprocessing.get_context("fork").Pool(procs) as pool:
    pool.map(_crop, jobs, chunksize=2)
  heatmap(spot_list, results, _md, os.path.join(rep, "heatmap_city.png"), box=CITY, m_per_px=3.0)
  heatmap(spot_list, results, _md, os.path.join(rep, "heatmap_all.png"), m_per_px=8.0)
  checked = [r for r in results if "summary" in r]
  stats = {"tiles": len(results), "checked": sum(not r["summary"]["unloaded"] for r in checked),
           "unloaded": sum(r["summary"]["unloaded"] for r in checked),
           "covered": sum(bool(r["summary"].get("covered")) for r in checked), "errors": sum("error" in r for r in results),
           "spots": len(spot_list), "by_type": dict(Counter(s["type"] for s in spot_list)), "map_hash": _md.map_hash,
           "marks": os.path.basename(_md.marks_file), "made": time.strftime("%Y-%m-%d %H:%M")}
  with open(os.path.join(rep, "spots.json"), "w") as f:
    json.dump({"stats": stats, "spots": spot_list}, f, indent=1)
  with open(os.path.join(rep, "index.html"), "w") as f:
    f.write(page(stats, spot_list[:top], spot_list))
  log(f"imgcheck: report {rep}/index.html: {json.dumps(stats)}")
  return os.path.join(rep, "index.html")


def page(stats: dict, shown: list[dict], everything: list[dict]) -> str:
  rows = []
  for s in shown:
    lines = "".join(f"<li><b>{html.escape(i['kind'])}</b> {html.escape(i['detail'])} <span class=d>({i['x']:.1f}, {i['y']:.1f}; "
                    f"{html.escape(i['tile'])})</span></li>" for i in s["issues"][:6])
    kinds = ", ".join(f"{k} {v}" for k, v in s["kinds"].items())
    rows.append(f"""<section id="s{s['rank']}"><h2>#{s['rank']} <span class="t {s['type']}">{s['type']}</span> score {s['score']}
<span class=d>({s['x']}, {s['y']}) &middot; seen in {s['seen']} tiles, flagged in {s['flagged']} &middot; {kinds}</span></h2>
<img loading=lazy src="crops/spot{s['rank']:04d}.jpg" alt="game | our map over the game | paint found and issues"><ul>{lines}</ul></section>""")
  by_type = ", ".join(f"{k}: {v}" for k, v in sorted(stats["by_type"].items(), key=lambda kv: -kv[1]))
  table = "".join(f"<tr><td>{s['rank']}</td><td>{s['type']}</td><td>{s['score']}</td><td>{s['x']}, {s['y']}</td><td>{s['seen']}</td>"
                  f"<td>{html.escape(s['issues'][0]['detail'])}</td></tr>" for s in everything[len(shown):len(shown) + 400])
  return f"""<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Lane map image check</title><style>
:root{{--bg:#121417;--fg:#e8e8e8;--dim:#9aa0a6;--card:#1c1f24}}
body{{background:var(--bg);color:var(--fg);font:14px/1.4 system-ui,sans-serif;margin:0 auto;max-width:1300px;padding:12px 16px}}
img{{max-width:100%;height:auto;display:block;border-radius:4px}} section{{background:var(--card);padding:8px 12px;margin:14px 0;border-radius:6px}}
h2{{font-size:16px;margin:4px 0 8px}} .d{{color:var(--dim);font-weight:normal;font-size:13px}} ul{{margin:6px 0;padding-left:18px}}
.t{{padding:1px 6px;border-radius:4px;color:#111}} .missing{{background:#f0f}} .stray{{background:#0ff}} .colour{{background:#f80}}
.kerb{{background:#f44}} .stop{{background:#ff0}} .junction{{background:#8af}} .offset{{background:#fff}}
table{{border-collapse:collapse;font-size:13px}} td{{padding:2px 8px;border-bottom:1px solid #333}}
</style></head><body><h1>Lane map vs the game's paint, from above</h1>
<p>{stats['checked']} tiles checked ({stats['unloaded']} unloaded, {stats['covered']} looking down on something over the road, {stats['errors']} errors) against map {stats['map_hash']}
({stats['marks']}), {stats['made']}. {stats['spots']} spots: {by_type}.</p>
<p>Each crop is {CROP_M:.0f} m square: the game; our map over it (kerbs green, lane lines white, centre lines yellow, stop
lines red/orange, arrows, junction outlines blue); the paint found (white, yellow), paint no map line accounts for
(magenta), what a bridge hides (blue tint), and the issues circled.</p>
<p><a href="heatmap_city.png"><img src="heatmap_city.png" alt="city heat map" style="max-width:640px"></a>
<a href="heatmap_all.png">whole map</a></p>
{''.join(rows)}<h2>More spots</h2><table><tr><td>#</td><td>type</td><td>score</td><td>x, y</td><td>seen</td><td>main issue</td></tr>
{table}</table></body></html>"""
