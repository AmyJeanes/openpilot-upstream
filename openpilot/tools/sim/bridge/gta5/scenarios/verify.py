"""Verify scenario trips on the live router: each line 'NAME spec  # [CAT] text; expect: kind@(x,y) ...; ...'.
kind: left / right (a real maneuver that way within NEAR m), keep_left / keep_right (a fork: stay/slight), ramp (on or off
ramp/exit), merge, straight (the route passes within PASS m with no real maneuver within NEAR m), pass (just passes).
Checks the route takes every expected maneuver in order, the first one has >= --approach m before it (and no real
maneuver before it that isn't expected), and the training distance at each expected point (route geometry of every
training approach, and training junctions). Writes data/verify_<file>.json for the figures."""
import argparse
import json
import math
import os
import re
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from openpilot.tools.sim.bridge.gta5.scenarios import scen_lib as L
from openpilot.tools.sim.bridge.gta5 import e2e, junction_plan as jp

NEAR, PASS = 35.0, 15.0
EXP = re.compile(r"(left|right|keep_left|keep_right|ramp|merge|straight|pass)@\((-?[\d.]+),\s*(-?[\d.]+)\)")
KEEP = {22, 23, 24, 9, 16}
RAMPS = {17, 18, 19, 20, 21}
MERGE = {25, 37, 38}


def match(kind, mm):
  t = mm["type"]
  if kind == "left":
    return mm["real"] and t in e2e.LEFT
  if kind == "right":
    return mm["real"] and t in e2e.RIGHT
  if kind == "keep_left":
    return t in KEEP and t in e2e.LEFT | {22}
  if kind == "keep_right":
    return t in KEEP and t in e2e.RIGHT | {22}
  if kind == "ramp":
    return t in RAMPS
  if kind == "merge":
    return t in MERGE or t in RAMPS
  return False


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("files", nargs="+")
  ap.add_argument("--approach", type=float, default=150.0)
  ap.add_argument("--show", action="store_true", help="print every route's maneuvers")
  a = ap.parse_args()
  router = jp.make_router(L.LIVE, "http://localhost:8002", None)
  train, plan = L.training_points()
  tj = np.array([ap_["junction"] for ap_ in plan["approaches"] if ap_["split"] == "train"])
  ab = L.ab_points()
  for path in a.files:
    out, bad = [], 0
    for line in open(os.path.expanduser(path)):
      body, _, comment = line.rstrip("\n").partition("#")
      p = body.split()
      if len(p) < 2 or ">" not in p[1]:
        continue
      name, spec, opts = p[0], p[1], p[2:]
      cat = re.search(r"\[([A-G]\d)\]", comment)
      exps = [(k, float(x), float(y)) for k, x, y in EXP.findall(comment)]
      rec = {"name": name, "spec": spec, "opts": opts, "cat": cat[1] if cat else None, "comment": comment.strip(), "expect": exps}
      try:
        r = L.route_info(router, spec)
      except Exception as e:
        rec.update(ok=False, why=f"no route: {e}")
        out.append(rec)
        print(f"{name:8s} FAIL no route")
        bad += 1
        continue
      pts = np.array(r["points"])
      along = np.concatenate(([0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))))
      mans = r["maneuvers"]
      why, s_prev, used, firsts, kinds = [], -1.0, set(), [], []
      for kind, x, y in exps:
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        if kind in ("straight", "pass"):
          ks = np.flatnonzero(d <= PASS)
          ks = [k for k in ks if along[k] > s_prev]
          if not ks:
            why.append(f"doesn't pass ({x:.0f},{y:.0f})")
            continue
          s = float(along[ks[0]])
          if kind == "straight" and any(mm["real"] and math.hypot(mm["xy"][0] - x, mm["xy"][1] - y) <= NEAR for mm in mans):
            why.append(f"a maneuver at ({x:.0f},{y:.0f}) (expected straight)")
          firsts.append(s)
          kinds.append(kind)
          s_prev = s
          continue
        hit = [i for i, mm in enumerate(mans) if i not in used and mm["s"] > s_prev - 1 and match(kind, mm)
               and math.hypot(mm["xy"][0] - x, mm["xy"][1] - y) <= NEAR]
        if not hit:
          near = [f"{mm['kind']}@({mm['xy'][0]:.0f},{mm['xy'][1]:.0f})" for mm in mans if math.hypot(mm["xy"][0] - x, mm["xy"][1] - y) <= 60]
          why.append(f"no {kind} at ({x:.0f},{y:.0f}) (near: {near})")
          continue
        used.add(hit[0])
        s_prev = mans[hit[0]]["s"]
        firsts.append(s_prev)
        kinds.append(kind)
      keyed = [s for s, k in zip(firsts, kinds, strict=True) if k not in ("keep_left", "keep_right", "pass")]
      key_s = keyed[0] if keyed else (firsts[0] if firsts else None)
      first_s = firsts[0] if firsts else None
      if key_s is not None and key_s < a.approach and rec["cat"] and rec["cat"][0] in "ABD":
        why.append(f"only {key_s:.0f} m before the first expected maneuver")
      unexpected = [mm for i, mm in enumerate(mans) if mm["real"] and i not in used and first_s is not None and mm["s"] < first_s - 30]
      if unexpected and rec["cat"] and rec["cat"][0] in "ABD":
        why.append("real maneuvers before the first expected one: " + ", ".join(f"{mm['kind']}@{mm['s']:.0f}m" for mm in unexpected))
      dts = [round(L.min_dist(train, x, y)) for _, x, y in exps]
      dtj = [round(float(np.min(np.hypot(tj[:, 0] - x, tj[:, 1] - y)))) for _, x, y in exps]
      dab = [min(((round(math.hypot(ax - x, ay - y)), n) for n, ax, ay in ab)) for _, x, y in exps]
      grade = None
      if r.get("z"):
        z = np.array(r["z"])[:, 1]
        grade = round(100 * max((abs(z[j + 15] - z[j]) / 150 for j in range(len(z) - 15)), default=0), 1)
      rec.update(z=r.get("z"), grade_max=grade)
      rec.update(ok=not why, why=why, route=r, key_s=key_s, dtrain=dts, dtrain_junction=dtj, dab=dab)
      bad += bool(why)
      flag = "ok  " if not why else "FAIL"
      warn = " <300m-train" if dts and min(dts) < 300 else ""
      print(f"{name:8s} {flag} {rec['cat']} route {r['length']} m, key at {key_s if key_s is None else round(key_s)} m, dtrain {dts} (junctions {dtj}){warn}"
            + ("" if not why else "  -- " + "; ".join(why)))
      if a.show or why:
        for mm in mans:
          print(f"      {mm['s']:6.0f} m {mm['kind']:13s} {str(mm['angle']):>5s} ({mm['xy'][0]:.0f},{mm['xy'][1]:.0f}) {mm['street']}")
      out.append(rec)
    base = os.path.splitext(os.path.basename(path))[0]
    json.dump(out, open(f"{L.D}/data/verify_{base}.json", "w"))
    print(f"{path}: {len(out) - bad}/{len(out)} ok -> data/verify_{base}.json")


if __name__ == "__main__":
  main()
