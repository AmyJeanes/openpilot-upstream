#!/usr/bin/env python3
"""Divergence v0: where GTA's AI, driving our routes in expert recordings, strayed from the lane line nav planned
(routes.json's lane_lines), ranked as candidate map-fix spots. Offline, from existing recordings only:

  python -m openpilot.tools.sim.bridge.gta5.divergence.v0 [--data /mnt/e/gta5rec/data] [--out /mnt/e/gta5_audit/divergence_v0]

Each AI-driven frame (gta5.npz expert, moving, on its route, not in a wrong-way clip) is compared with the lane line
nav had drawn LAG s before, so turns and lane plans count as intent rather than following the car. The signed offset
(m right of the line) is averaged per 5 m driven; a stretch over LAT_M for LAT_RUN m is a `lat` divergence, over LANE_M
for LANE_RUN m a `lane` one (a lane apart), over TURN_M through a turn (heading changing) a `turn` one. Stretches are
clustered across trips (CLUSTER_M, similar heading); each cluster's rate is its trips flagged over the trips that
drove through it. Consistent clusters (rate >= 0.5, 3+ trips) are MAP_suspect (the AI agrees with itself, not with the
map), rare ones AI_aggressive, the rest Ambiguous. Writes spots.json (all clusters with their stretches), top.csv and
an overview plot."""
import argparse
import json
import math
import os
from multiprocessing import Pool

import numpy as np

LAG = 2.0  # s: the lane line drawn this long before the frame
BIN_M = 5.0
LAT_M, LAT_RUN = 1.5, 20.0
LANE_M, LANE_RUN = 2.75, 30.0
TURN_M, TURN_RATE = 2.5, 8.0  # m; deg of heading per 5 m: in a turn
MIN_V = 2.0  # m/s
ROUTE_OFF = 15.0  # m off the route: not driving it
CLUSTER_M = 25.0
CLUSTER_HEADING = 45.0  # deg
PASS_M = 15.0  # m from a cluster's centre that a trip drove through it
WW_DRIVEN = {1, 2, 3, 4, 7}  # gta5_wrongway PHASES driven by the clip's controller (approach..recover, abort)


def project(line: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
  """(m along the line, m right of it, distance) of p's nearest point on a polyline."""
  a, b = line[:-1], line[1:]
  ab = b - a
  L2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-9)
  t = np.clip(np.einsum("ij,ij->i", p - a, ab) / L2, 0.0, 1.0)
  near = a + ab * t[:, None]
  d = np.hypot(*(near - p).T)
  i = int(np.argmin(d))
  seg = math.sqrt(L2[i])
  along = float(np.concatenate(([0.0], np.cumsum(np.sqrt(L2))))[i] + t[i] * seg)
  right = float((p[0] - a[i, 0]) * ab[i, 1] - (p[1] - a[i, 1]) * ab[i, 0]) / seg
  return along, right, float(d[i])


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
  e = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
  return list(zip(e[::2], e[1::2], strict=True))


def segment(path: str) -> dict:
  """One recording segment's divergent stretches and its 5 m bins (for the passes)."""
  name = os.path.basename(path)
  try:
    z = np.load(os.path.join(path, "gta5.npz"))
    rj = json.load(open(os.path.join(path, "routes.json")))
  except (OSError, ValueError):
    return {"seg": name, "stretches": [], "bins": []}
  cols = list(z["frame_columns"])
  f = z["frame"]
  c = {k: f[:, cols.index(k)] for k in ("t", "x", "y", "heading", "v_ego", "expert", "route", "route_off", "collisions")}
  lines = [(int(e["frame"]), np.asarray(e["line"], float)) for e in rj.get("lane_lines") or [] if len(e.get("line") or []) >= 2]
  if not lines or not c["expert"].any():
    return {"seg": name, "stretches": [], "bins": []}
  ok = (c["expert"] > 0) & (c["v_ego"] > MIN_V) & ~(c["route_off"] > ROUTE_OFF)
  if "wrongway" in z:
    ww = z["wrongway"]
    ok &= ~((ww[:, 0] >= 0) & np.isin(ww[:, 1], list(WW_DRIVEN)))
  frames = np.array([fr for fr, _ in lines])
  e_lat = np.full(len(f), np.nan)
  for i in np.flatnonzero(ok):
    # the newest line drawn at least LAG s before this frame
    j = int(np.searchsorted(c["t"], c["t"][i] - LAG, side="right")) - 1
    k = int(np.searchsorted(frames, j, side="right")) - 1
    if j < 0 or k < 0:
      continue
    if c["route"][min(lines[k][0], len(f) - 1)] != c["route"][i]:
      continue  # drawn on another route (a reroute since)
    line = lines[k][1]
    along, right, d = project(line, np.array([c["x"][i], c["y"][i]]))
    if along <= 1.0 or along >= _length(line) - 1.0 or d > 12.0:
      continue  # behind the line's start, past its end, or on another road
    e_lat[i] = right
  # 5 m bins by distance driven
  xy = np.stack([c["x"], c["y"]], axis=1)
  dist = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))))
  b = (dist // BIN_M).astype(int)
  bins = []
  for k in np.unique(b[ok]):
    m = (b == k) & ok
    vals = e_lat[m]
    vals = vals[~np.isnan(vals)]
    h = c["heading"][m]
    bins.append((float(c["x"][m].mean()), float(c["y"][m].mean()), float(np.degrees(np.arctan2(np.sin(np.radians(h)).mean(), np.cos(np.radians(h)).mean()))),
                 float(vals.mean()) if len(vals) else math.nan, int(np.flatnonzero(m)[0]), int(c["collisions"][m].max() - c["collisions"][m].min())))
  out = []
  if bins:
    arr = np.array([bb[3] for bb in bins])
    hd = np.array([bb[2] for bb in bins])
    turning = np.abs(np.diff(np.unwrap(np.radians(hd)), prepend=np.radians(hd[0])))
    turning = np.degrees(np.maximum(turning, np.concatenate((turning[1:], [0.0]))))
    finite = ~np.isnan(arr)
    for kind, mask, run_m in (("lane", finite & (np.abs(np.nan_to_num(arr)) > LANE_M), LANE_RUN),
                              ("lat", finite & (np.abs(np.nan_to_num(arr)) > LAT_M), LAT_RUN),
                              ("turn", finite & (np.abs(np.nan_to_num(arr)) > TURN_M) & (turning > TURN_RATE), BIN_M * 2)):
      for a0, a1 in runs(mask):
        if (a1 - a0) * BIN_M < run_m:
          continue
        part = bins[a0:a1]
        e = arr[a0:a1]
        k_mid = (a0 + a1) // 2
        out.append({"seg": name, "kind": kind, "x": round(bins[k_mid][0], 1), "y": round(bins[k_mid][1], 1), "heading": round(bins[k_mid][2], 1),
                    "length": (a1 - a0) * BIN_M, "max": round(float(e[np.argmax(np.abs(e))]), 2), "mean": round(float(e.mean()), 2),
                    "frame": part[0][4], "contact": int(sum(p[5] for p in part) > 0),
                    "path": [[round(p[0], 1), round(p[1], 1)] for p in part]})
  # a lane stretch is also a lat one: keep the strongest kind per place
  out.sort(key=lambda s: ({"lane": 0, "turn": 1, "lat": 2}[s["kind"]], -s["length"]))
  kept = []
  for s in out:
    if not any(s["frame"] <= k["frame"] + k["length"] and math.hypot(s["x"] - k["x"], s["y"] - k["y"]) < max(k["length"], 20.0) for k in kept):
      kept.append(s)
  return {"seg": name, "stretches": kept, "bins": [(round(x, 1), round(y, 1), round(h, 1)) for x, y, h, *_ in bins]}


def _length(line: np.ndarray) -> float:
  return float(np.hypot(*np.diff(line, axis=0).T).sum())


def heading_close(a: float, b: float, tol: float = CLUSTER_HEADING) -> bool:
  return abs((a - b + 180) % 360 - 180) <= tol


def cluster(stretches: list[dict]) -> list[list[dict]]:
  """Single-linkage clusters of stretches within CLUSTER_M m of each other, headed alike."""
  n = len(stretches)
  parent = list(range(n))

  def find(i):
    while parent[i] != i:
      parent[i] = parent[parent[i]]
      i = parent[i]
    return i
  cells: dict[tuple[int, int], list[int]] = {}
  for i, s in enumerate(stretches):
    cells.setdefault((int(s["x"] // CLUSTER_M), int(s["y"] // CLUSTER_M)), []).append(i)
  for i, s in enumerate(stretches):
    cx, cy = int(s["x"] // CLUSTER_M), int(s["y"] // CLUSTER_M)
    for dx in (-1, 0, 1):
      for dy in (-1, 0, 1):
        for j in cells.get((cx + dx, cy + dy), []):
          if j > i and math.hypot(s["x"] - stretches[j]["x"], s["y"] - stretches[j]["y"]) <= CLUSTER_M and \
             heading_close(s["heading"], stretches[j]["heading"]):
            parent[find(i)] = find(j)
  groups: dict[int, list[dict]] = {}
  for i, s in enumerate(stretches):
    groups.setdefault(find(i), []).append(s)
  return list(groups.values())


def passes(bins_by_seg: dict[str, list], x: float, y: float, heading: float) -> set[str]:
  """The segments that drove within PASS_M m of a place, headed alike."""
  out = set()
  for seg, bins in bins_by_seg.items():
    for bx, by, bh in bins:
      if abs(bx - x) < PASS_M and abs(by - y) < PASS_M and math.hypot(bx - x, by - y) < PASS_M and heading_close(bh, heading):
        out.add(seg)
        break
  return out


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--data", default="/mnt/e/gta5rec/data")
  ap.add_argument("--out", default="/mnt/e/gta5_audit/divergence_v0")
  ap.add_argument("--jobs", type=int, default=16)
  ap.add_argument("--limit", type=int, default=0, help="only the first N segments (testing)")
  args = ap.parse_args()
  segs = sorted(os.path.join(args.data, s) for s in os.listdir(args.data))
  if args.limit:
    segs = segs[:args.limit]
  with Pool(args.jobs) as pool:
    results = pool.map(segment, segs, chunksize=4)
  stretches = [s for r in results for s in r["stretches"]]
  bins_by_seg = {r["seg"]: r["bins"] for r in results if r["bins"]}
  # a "trip" is a recording's run of segments: count one segment name's route as its own pass
  clusters = []
  for g in cluster(stretches):
    x = float(np.mean([s["x"] for s in g]))
    y = float(np.mean([s["y"] for s in g]))
    hd = float(np.degrees(np.arctan2(np.mean([math.sin(math.radians(s["heading"])) for s in g]),
                                     np.mean([math.cos(math.radians(s["heading"])) for s in g]))))
    flagged = {s["seg"] for s in g}
    passed = passes(bins_by_seg, x, y, hd) | flagged
    rate = len(flagged) / max(len(passed), 1)
    excess = float(np.mean([abs(s["max"]) for s in g]))
    kinds = {k: sum(s["kind"] == k for s in g) for k in ("lane", "turn", "lat")}
    signed = float(np.mean([s["mean"] for s in g]))
    cls = "MAP_suspect" if rate >= 0.5 and len(flagged) >= 3 else "AI_aggressive" if rate < 0.25 else "Ambiguous"
    clusters.append({"x": round(x, 1), "y": round(y, 1), "heading": round(hd, 1), "trips": len(flagged), "passes": len(passed),
                     "rate": round(rate, 2), "max_mean": round(excess, 2), "signed_mean": round(signed, 2), "kinds": kinds,
                     "class": cls, "contacts": sum(s["contact"] for s in g),
                     "score": round(len(flagged) * rate * excess, 2), "stretches": g})
  clusters.sort(key=lambda c: -c["score"])
  os.makedirs(args.out, exist_ok=True)
  meta = {"segments": len(segs), "with_ai_frames": len(bins_by_seg), "stretches": len(stretches), "clusters": len(clusters),
          "params": {"lag": LAG, "bin_m": BIN_M, "lat": [LAT_M, LAT_RUN], "lane": [LANE_M, LANE_RUN], "turn": [TURN_M, TURN_RATE],
                     "cluster_m": CLUSTER_M}}
  json.dump({"meta": meta, "clusters": clusters}, open(os.path.join(args.out, "spots.json"), "w"))
  with open(os.path.join(args.out, "top.csv"), "w") as f:
    f.write("rank,x,y,heading,class,trips,passes,rate,max_mean_m,signed_mean_m,lane,turn,lat,contacts,score,example_seg,example_frame\n")
    for n, c in enumerate(clusters, 1):
      ex = max(c["stretches"], key=lambda s: abs(s["max"]))
      f.write(f"{n},{c['x']},{c['y']},{c['heading']},{c['class']},{c['trips']},{c['passes']},{c['rate']},{c['max_mean']},"
              f"{c['signed_mean']},{c['kinds']['lane']},{c['kinds']['turn']},{c['kinds']['lat']},{c['contacts']},{c['score']},{ex['seg']},{ex['frame']}\n")
  _plot(clusters, bins_by_seg, os.path.join(args.out, "overview.png"))
  print(json.dumps(meta))
  for n, c in enumerate(clusters[:20], 1):
    print(f"{n:2d} ({c['x']:8.1f},{c['y']:8.1f}) hdg {c['heading']:6.1f} {c['class']:13s} trips {c['trips']:3d}/{c['passes']:3d} "
          f"rate {c['rate']:.2f} |e| {c['max_mean']:.2f} signed {c['signed_mean']:+.2f} {c['kinds']} score {c['score']}")


def _plot(clusters, bins_by_seg, path):
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    return
  fig, ax = plt.subplots(figsize=(12, 14), dpi=110)
  allb = np.array([(b[0], b[1]) for bins in bins_by_seg.values() for b in bins])
  if len(allb):
    ax.scatter(allb[:, 0], allb[:, 1], s=0.2, c="#bbbbbb", linewidths=0)
  colours = {"MAP_suspect": "#d62728", "Ambiguous": "#ff7f0e", "AI_aggressive": "#1f77b4"}
  for n, c in enumerate(clusters[:60], 1):
    ax.scatter([c["x"]], [c["y"]], s=10 + 4 * c["score"], c=colours[c["class"]], alpha=0.7)
    if n <= 20:
      ax.annotate(str(n), (c["x"], c["y"]), fontsize=8, xytext=(3, 3), textcoords="offset points")
  ax.set_aspect("equal")
  ax.set_title("Divergence v0: AI pose vs nav's lane line (top 60; red MAP_suspect, orange Ambiguous, blue AI_aggressive)")
  fig.tight_layout()
  fig.savefig(path)


if __name__ == "__main__":
  main()
