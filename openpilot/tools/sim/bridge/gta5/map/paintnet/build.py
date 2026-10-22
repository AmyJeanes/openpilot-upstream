"""build.py [--force | --from=step] [--only=step,...]: the pilot's whole chain (lanes for the map, the regions checker, the
OSM export), each step rerun only when its code or inputs changed.

One process loads what every step shares (libraries, the game-file sources) and forks a child per step, so the steps
don't each pay for loading them; steps whose inputs are ready run side by side. Logs: $O/logs/<step>.log."""
import hashlib, json, os, pathlib, re, runpy, sys, time, traceback

HERE = pathlib.Path(__file__).resolve().parent
B = ["-296", "-9", "-1333", "-1046"]  # x0 x1 y0 y1: 287 m square round the interchange
O = "/mnt/e/gta5_audit/paintnet"
WEB = "/mnt/e/gta5_webmap"
RP = "/home/amy/git/openpilot-slowroads/openpilot/tools/sim/bridge/gta5/map/roadpaint"
WEBMAP = "/home/amy/git/op-ortho/openpilot/tools/sim/bridge/gta5/map/webmap"
o = lambda n: f"{O}/{n}"
# name, script, args, inputs (other steps' outputs), outputs, other files it reads
STEPS = [
  ("area", "area.py", B + [f"{WEB}/pilot_area.geojson"], [], [f"{WEB}/pilot_area.geojson"], []),
  ("visible", "visible.py", B + [o("ic_visible.json")], [], [o("ic_visible.json")], ["/mnt/e/gta5_webmap/sat.pmtiles", "/mnt/e/gta5_webmap/under.pmtiles"]),
  ("lanes", "lanes.py", B + [o("ic_lanes.json"), "--hatch=0.1", f"--visible={o('ic_visible.json')}"], [o("ic_visible.json")], [o("ic_lanes.json")], []),
  ("direction", "direction.py", B + [o("ic_lanes.json"), o("ic_dir.json")], [o("ic_lanes.json")], [o("ic_dir.json")], []),
  ("groups", "groups.py", B + [o("ic_dir.json"), o("ic_groups.json")], [o("ic_dir.json")], [o("ic_groups.json")], []),
  ("connect", "connect.py", B + [o("ic_groups.json"), o("ic_links.json")], [o("ic_groups.json")], [o("ic_links.json")], []),
  ("export", "export.py", [o("ic_dir.json"), o("ic_groups.json"), f"{WEB}/pilot.geojson", o("ic_links.json")],
   [o("ic_dir.json"), o("ic_groups.json"), o("ic_links.json")], [f"{WEB}/pilot.geojson"], []),
  ("regions", "regions.py", B + [o("ic_regions.npz")], [], [o("ic_regions.npz")], []),
  ("rstats", "regionstats.py", [o("ic_regions.npz"), o("ic_rstats.json")], [o("ic_regions.npz")], [o("ic_rstats.json")], []),
  ("check", "check.py", [o("ic_regions.npz"), o("ic_rstats.json"), o("ic_groups.json"), o("ic_dir.json"), f"{WEB}/check.geojson"],
   [o("ic_regions.npz"), o("ic_rstats.json"), o("ic_groups.json"), o("ic_dir.json")], [f"{WEB}/check.geojson"], [str(HERE / "axes.py")]),
  ("osm", "osm.py", B + [o("ic_groups.json"), o("ic_links.json"), o("ic_dir.json"), f"{WEB}/osm_ic.osm", f"{WEB}/osm.geojson"],
   [o("ic_groups.json"), o("ic_links.json"), o("ic_dir.json")], [f"{WEB}/osm_ic.osm", f"{WEB}/osm.geojson"], []),
]
STATE, LOGS = pathlib.Path(o(".build_state.json")), pathlib.Path(o("logs"))


def code_files(script):
  """the script and the local modules it imports, recursively."""
  out, todo = set(), [HERE / script]
  while todo:
    p = todo.pop()
    if p in out or not p.exists():
      continue
    out.add(p)
    for m in re.findall(r"^\s*import (\w+)", p.read_text(), re.M):
      todo += [d / f"{m}.py" for d in (HERE, pathlib.Path(RP), pathlib.Path(WEBMAP))]
  return sorted(out)


def digest(paths):
  h = hashlib.sha256()
  for p in paths:
    h.update(str(p).encode())
    h.update(pathlib.Path(p).read_bytes() if pathlib.Path(p).exists() else b"missing")
  return h.hexdigest()


def key(step):
  name, script, args, ins, outs, extra = step
  import src
  data = [f"{p}:{os.stat(p).st_mtime_ns}" for p in (src.PAINT, src.FEATS, src.PATHS, *[e for e in extra if e.endswith(".pmtiles")])]
  return digest(code_files(script) + ins + [e for e in extra if not e.endswith(".pmtiles")]) + json.dumps([args, data])


def load():
  """what the steps share, loaded once before the first step runs."""
  t0 = time.time()
  import numpy, scipy.ndimage, cv2, rp, gframe, src
  src.fast_tiles(rp)
  for f in (src.paint, src.features, src.paths):
    f()
  import scan, arrows
  print(f"loaded shared data in {time.time() - t0:.1f} s", flush=True)


def run(name, script, args, prof):
  """in a forked child: the step's script as __main__, its output to its log; exits with its status."""
  code = 0
  try:
    fd = os.open(LOGS / f"{name}.log", os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
    os.dup2(fd, 1); os.dup2(fd, 2)
    sys.argv = [str(HERE / script)] + args
    if prof:
      import cProfile
      cProfile.runctx("runpy.run_path(p, run_name='__main__')", {"runpy": runpy, "p": sys.argv[0]}, {}, str(LOGS / f"{name}.prof"))
    else:
      runpy.run_path(sys.argv[0], run_name="__main__")
  except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
  except BaseException:
    traceback.print_exc(); code = 1
  sys.stdout.flush(); sys.stderr.flush()
  os._exit(code)


def main():
  argv = sys.argv[1:]
  start = next((a[7:] for a in argv if a.startswith("--from=")), None)
  only = next((a[7:].split(",") for a in argv if a.startswith("--only=")), None)
  LOGS.mkdir(parents=True, exist_ok=True)
  state = json.loads(STATE.read_text()) if STATE.exists() and "--force" not in argv else {}
  if start:  # the step and everything downstream of it
    redo = {start}
    for s in STEPS:
      if any(i in t[4] for t in STEPS if t[0] in redo for i in s[3]):
        redo.add(s[0])
    for n in redo:
      state.pop(n, None)
  t0 = time.time()
  sys.path[:0] = [str(HERE), RP, WEBMAP]
  sys.argv = ["build"] + B + ["/dev/null"]  # scan.py reads the box from its arguments
  todo = {s[0]: s for s in STEPS if only is None or s[0] in only}
  produced = {x: s[0] for s in STEPS for x in s[4]}
  failed, running, keys, loaded = set(), {}, {}, False
  while todo or running:
    for name, s in list(todo.items()):
      ins = {produced[i] for i in s[3] if i in produced}
      if ins & failed:
        failed.add(name); del todo[name]; print(f"{name:10s} skipped (an input failed)"); continue
      if ins & (todo.keys() | running.keys()):
        continue
      del todo[name]
      keys[name] = key(s)
      if state.get(name) == keys[name] and all(os.path.exists(x) for x in s[4]):
        print(f"{name:10s} unchanged"); continue
      if not loaded:
        load(); loaded = True
      sys.stdout.flush(); sys.stderr.flush()
      pid = os.fork()
      if pid == 0:
        run(name, s[1], s[2], "--profile" in argv)
      running[name] = (pid, time.time())
    if not running:
      continue
    pid, status = os.wait()
    name = next(n for n, (p, _) in running.items() if p == pid)
    _, ts = running.pop(name)
    if os.waitstatus_to_exitcode(status) == 0:
      state[name] = keys[name]
      print(f"{name:10s} {time.time() - ts:5.1f} s", flush=True)
    else:
      failed.add(name); state.pop(name, None)
      print(f"{name:10s} FAILED ({LOGS / name}.log):", flush=True)
      print("".join((LOGS / f"{name}.log").read_text().splitlines(True)[-15:]), flush=True)
    STATE.write_text(json.dumps(state))
  print(f"total {time.time() - t0:.1f} s" + (f"; FAILED: {', '.join(sorted(failed))}" if failed else ""))
  sys.exit(1 if failed else 0)


if __name__ == "__main__":
  main()
