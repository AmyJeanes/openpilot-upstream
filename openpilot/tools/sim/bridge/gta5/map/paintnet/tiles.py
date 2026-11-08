"""tiles.py run WORK [--size=512] [--margin=100] [--jobs=4] [--workers=2] [--area=x0,x1,y0,y1] [--steps=a,b,...] [--redo]
tiles.py merge WORK DEST: the chain (build.py) over the whole map as square boxes, and their exports merged into one.

run: the map is cut into SIZE m cells (on multiples of SIZE) holding paint; each cell runs build.py on itself plus
MARGIN m round it (so lanes, links and junctions near its edge see what lies beyond) and exports only the cell (--core).
Boxes run JOBS at a time, each lanes step with WORKERS processes; no box starts while WSL has under MIN_FREE GB free.
Each box's files in WORK/boxes/<name>, its result appended to WORK/runs.jsonl (seconds, peak memory, failed step and the
end of its log); boxes already done are skipped unless --redo.
merge: every done box's export cut to its cell: lanes and strips are already cut by export.py; junctions, turns, link
points, flags and connectors are kept by the box whose cell holds them (a junction by its ring's middle, its paths with
it). Ids get the box's name. Seams: a lane end within SEAM_NEAR m of an edge two done cells share must meet a lane end in
the other cell within SEAM_MATCH m, else it is flagged ("seam break"). Out: DEST/paintnet_full.geojson,
DEST/osm_check.geojson (each box's OSM checker findings in its cell), DEST/summary.json."""
import collections, json, math, os, pathlib, re, subprocess, sys, time

HERE = pathlib.Path(__file__).resolve().parent
PY = sys.executable
MIN_FREE = 10.0  # GB of WSL memory a box needs free to start (a 712 m box peaks at ~4-6 GB)
SEAM_NEAR, SEAM_MATCH = 1.5, 2.5  # m
STEPS = "visible,lanes,direction,groups,connect,junctions,export,osm,osmcheck"


def opts(argv):
  return dict(a[2:].split("=", 1) if "=" in a else (a[2:], "1") for a in argv if a.startswith("--"))


def cells(size, area):
  """the cells (cx, cy) with a paint line's point in them."""
  sys.path.insert(0, str(HERE))
  import src
  have = collections.Counter()
  for r in src.paint():
    for x, y, *_ in r["pts"]:
      if area is None or (area[0] <= x <= area[1] and area[2] <= y <= area[3]):
        have[(math.floor(x / size), math.floor(y / size))] += 1
  return sorted(have)


def warm(boxes):
  """the surface raster tiles every box reads, copied into the cache before boxes run side by side (each copy is
  written once, by one process)."""
  sys.path[:0] = [str(HERE), "/home/amy/git/openpilot-slowroads/openpilot/tools/sim/bridge/gta5/map/roadpaint"]
  import rp, src
  src.fast_tiles(rp)
  d = pathlib.Path("/mnt/e/gta5_scratch/gamefiles_v2/rp_all/tiles")
  names = set(os.listdir(d))
  want = {f"{tx}_{ty}.bin" for x0, x1, y0, y1 in boxes
          for tx in range(math.floor((x0 - 2 * rp.TILE) / rp.TILE), math.floor((x1 + 2 * rp.TILE) / rp.TILE) + 1)
          for ty in range(math.floor((y0 - 2 * rp.TILE) / rp.TILE), math.floor((y1 + 2 * rp.TILE) / rp.TILE) + 1)} & names
  t0 = time.time()
  for n in sorted(want):
    rp.read_tile.__wrapped__(str(d / n))
  print(f"{len(want)} raster tiles cached in {time.time() - t0:.0f} s", flush=True)


def free_gb():
  for l in open("/proc/meminfo"):
    if l.startswith("MemAvailable:"):
      return int(l.split()[1]) / 1e6
  return 0.0


def run(work, o):
  size, margin = float(o.get("size", 512)), float(o.get("margin", 100))
  jobs, workers = int(o.get("jobs", 4)), o.get("workers", "2")
  area = list(map(float, o["area"].split(","))) if "area" in o else None
  work = pathlib.Path(work)
  (work / "boxes").mkdir(parents=True, exist_ok=True)
  log = work / "runs.jsonl"
  done = set()
  if log.exists() and "redo" not in o:
    done = {r["name"] for r in map(json.loads, log.read_text().splitlines()) if r["ok"]}
  todo = []
  for cx, cy in cells(size, area):
    name = f"{cx}_{cy}"
    core = [cx * size, (cx + 1) * size, cy * size, (cy + 1) * size]
    if name not in done:
      todo.append((name, core, [core[0] - margin, core[1] + margin, core[2] - margin, core[3] + margin]))
  print(f"{len(todo)} boxes to run ({len(done)} done)", flush=True)
  warm([b for _, _, b in todo])
  env = {**os.environ, "PAINTNET_WORKERS": workers, "OMP_NUM_THREADS": "1"}
  running, t_all = {}, time.time()
  while todo or running:
    while todo and len(running) < jobs and (not running or free_gb() >= MIN_FREE):
      name, core, box = todo.pop(0)
      out = work / "boxes" / name
      out.mkdir(exist_ok=True)
      f = lambda v: ",".join(f"{x:g}" for x in v)
      cmd = [PY, str(HERE / "build.py"), "--force", f"--box={f(box)}", f"--core={f(core)}", f"--out={out}", f"--only={o.get('steps', STEPS)}"]
      p = subprocess.Popen(cmd, stdout=open(out / "build.log", "w"), stderr=subprocess.STDOUT, env=env)
      running[p.pid] = (name, core, box, time.time())
    if not running:
      time.sleep(5)
      continue
    pid, status, ru = os.wait4(-1, 0)
    if pid not in running:
      continue
    name, core, box, ts = running.pop(pid)
    out = work / "boxes" / name
    text = (out / "build.log").read_text()
    failed = re.findall(r"^(\w+)\s+FAILED", text, re.M)
    why = ""
    if failed:
      lines = (out / "logs" / f"{failed[0]}.log").read_text().strip().splitlines()
      why = lines[-1] if lines else ""
    rec = {"name": name, "core": core, "box": box, "ok": os.waitstatus_to_exitcode(status) == 0, "seconds": round(time.time() - ts, 1),
           "peak_gb": round(ru.ru_maxrss / 1e6, 2), "failed": failed, "why": why[-300:],
           "steps": dict(re.findall(r"^(\w+)\s+([\d.]+) s$", text, re.M))}
    with open(log, "a") as fh:
      fh.write(json.dumps(rec) + "\n")
    print(f"{name:8s} {'ok' if rec['ok'] else 'FAILED ' + ','.join(failed)} {rec['seconds']:6.0f} s {rec['peak_gb']:5.1f} GB"
          f"  ({len(todo)} left, {time.time() - t_all:.0f} s){'  ' + why[-120:] if why else ''}", flush=True)


def merge(work, dest):
  sys.path.insert(0, "/home/amy/git/op-ortho/openpilot/tools/sim/bridge/gta5/map/webmap")
  import numpy as np
  import gframe
  work, dest = pathlib.Path(work), pathlib.Path(dest)
  dest.mkdir(parents=True, exist_ok=True)
  runs = {}
  for r in map(json.loads, (work / "runs.jsonl").read_text().splitlines()):
    runs[r["name"]] = r  # the last run of each box
  ok = {n: r for n, r in runs.items() if r["ok"]}
  size = ok[next(iter(ok))]["core"][1] - ok[next(iter(ok))]["core"][0] if ok else 512

  def xy(c):
    mx, my = gframe.latlon_to_merc(np.asarray(c)[..., 1], np.asarray(c)[..., 0])
    return np.stack([(mx - 0.5) * gframe.K, (gframe.MY0 - my) * gframe.K], -1)

  def inside(p, core):
    return core[0] <= p[0] < core[1] and core[2] <= p[1] < core[3]

  def ring_area(P):
    return abs(float(np.dot(P[:-1, 0], P[1:, 1]) - np.dot(P[1:, 0], P[:-1, 1]))) / 2

  stats = collections.Counter()
  flags = collections.Counter()
  path_flags = collections.Counter()
  lane_m = collections.Counter()
  ends = collections.defaultdict(list)  # cell -> [(x, y)] lane ends near its edges
  fh = open(dest / "paintnet_full.geojson", "w")
  fh.write('{"type": "FeatureCollection", "features": [\n')
  first = True
  for name, r in sorted(ok.items()):
    core = r["core"]
    path = work / "boxes" / name / "pilot.geojson"
    if not path.exists():
      continue
    F = json.load(open(path))["features"]
    kept_j, conn = set(), set()
    for f in F:
      if f["properties"]["kind"] == "junction":
        P = xy(f["geometry"]["coordinates"][0][:-1])
        if inside(P.mean(0), core):
          kept_j.add(f["properties"]["id"].split()[-1])
      elif f["properties"].get("connector"):
        conn.add(f["properties"]["id"])
    for f in F:
      p, g = f["properties"], f["geometry"]
      k = p["kind"]
      if k == "junction":
        keep = p["id"].split()[-1] in kept_j
      elif k == "path":
        keep = str(p["junction"]) in kept_j
      elif g["type"] == "Point":
        keep = inside(xy(g["coordinates"]), core)
      elif p.get("connector") or (k == "edge" and p["id"] in conn):  # (not cut by export.py)
        keep = None
      else:
        keep = True
      if keep is None:  # a connector and its edges: the box holding its middle
        C = g["coordinates"][0] if g["type"] in ("Polygon", "MultiLineString") else g["coordinates"]
        keep = inside(xy(C[len(C) // 2]), core)
      if not keep:
        continue
      if "id" in p and isinstance(p["id"], str):
        p["id"] = f"{name}:{p['id']}"
      p["box"] = name
      stats[k + (" connector" if p.get("connector") else "")] += 1
      if k == "lane" and not p.get("connector"):
        R = xy(g["coordinates"][0])
        lane_m[p["dir"] != "?"] += ring_area(R) / max(p.get("width", 0) or 0, 0.5)
        n = (len(R) - 1) // 2
        for e in ((R[0] + R[2 * n - 1]) / 2, (R[n - 1] + R[n]) / 2):
          d = min(e[0] - core[0], core[1] - e[0], e[1] - core[2], core[3] - e[1])
          if d < SEAM_NEAR:
            ends[name].append((float(e[0]), float(e[1]), round(float(p["z"]), 1)))
      if k == "flag":
        flags[re.sub(r"\b[\w|.>: ]*\d[\w|]*\b", "#", p["why"]).replace("#: ", "").strip()] += 1
      if k == "path" and p.get("flags"):
        for t in p["flags"].split(", "):
          path_flags[t] += 1
      fh.write(("" if first else ",\n") + json.dumps(f))
      first = False
  # seams: lane ends at an edge two done cells share, matched across it
  seam_ok = seam_bad = 0
  for name, E in ends.items():
    cx, cy = map(int, name.split("_"))
    core = ok[name]["core"]
    for x, y, z in E:
      sides = [(nx, ny) for d, nx, ny in ((x - core[0], cx - 1, cy), (core[1] - x, cx + 1, cy), (y - core[2], cx, cy - 1), (core[3] - y, cx, cy + 1))
               if d < SEAM_NEAR]
      for nx, ny in sides:
        other = f"{nx}_{ny}"
        if other not in ok:
          continue
        if any(math.hypot(x - a, y - b) <= SEAM_MATCH for a, b, c in ends.get(other, [])):  # (a piece's z is its median: no use here)
          seam_ok += 1
        else:
          seam_bad += 1
          lat, lon = gframe.lat_lon(x, y)
          f = {"type": "Feature", "properties": {"kind": "flag", "group": None, "why": f"seam break: a lane ends at the edge of box {name}, none carries on in {other}",
                                                 "z": z, "covered": False, "deck": False, "on_deck": False, "box": name},
               "geometry": {"type": "Point", "coordinates": [round(float(lon), 8), round(float(lat), 8)]}}
          fh.write(",\n" + json.dumps(f))
          flags["seam break"] += 1
  fh.write("\n]}\n")
  fh.close()
  # the OSM checker's findings, each in its own box's cell
  osm = collections.Counter()
  feats = []
  for name, r in sorted(ok.items()):
    path = work / "boxes" / name / "osm_check.geojson"
    if not path.exists():
      continue
    for f in json.load(open(path))["features"]:
      if f["geometry"]["type"] == "Point" and inside(xy(f["geometry"]["coordinates"]), r["core"]):
        f["properties"]["box"] = name
        osm[f["properties"].get("check", "?")] += 1
        feats.append(f)
  json.dump({"type": "FeatureCollection", "features": feats}, open(dest / "osm_check.geojson", "w"))
  failed = {n: {"failed": r["failed"], "why": r["why"], "seconds": r["seconds"], "peak_gb": r["peak_gb"]} for n, r in runs.items() if not r["ok"]}
  secs = [r["seconds"] for r in ok.values()]
  step_s = collections.Counter()
  for r in ok.values():
    for s, v in r.get("steps", {}).items():
      step_s[s] += float(v)
  summary = {
    "boxes": {"run": len(runs), "ok": len(ok), "failed": len(failed), "cell_m": size,
              "margin_m": (ok[next(iter(ok))]["core"][0] - ok[next(iter(ok))]["box"][0]) if ok else None,
              "box_seconds": {"total": round(sum(secs)), "median": float(np.median(secs)) if secs else None, "max": max(secs, default=None)},
              "step_seconds_total": {s: round(v) for s, v in step_s.most_common()},
              "peak_gb_max": max((r["peak_gb"] for r in ok.values()), default=None)},
    "lanes_km": {"total": round(sum(lane_m.values()) / 1000, 1), "with_direction": round(lane_m[True] / 1000, 1), "no_direction": round(lane_m[False] / 1000, 1)},
    "features": dict(stats.most_common()),
    "junctions": stats["junction"],
    "flags": dict(flags.most_common()),
    "junction_path_flags": dict(path_flags.most_common()),
    "seams": {"lane_ends_matched": seam_ok, "lane_ends_unmatched": seam_bad},
    "osm_check": dict(osm.most_common()),
    "failed_boxes": failed,
  }
  json.dump(summary, open(dest / "summary.json", "w"), indent=1)
  print(json.dumps({k: v for k, v in summary.items() if k != "failed_boxes"}, indent=1))


if __name__ == "__main__":
  if sys.argv[1] == "run":
    run(sys.argv[2], opts(sys.argv[3:]))
  elif sys.argv[1] == "merge":
    merge(sys.argv[2], sys.argv[3])
