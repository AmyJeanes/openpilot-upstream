"""where.py lat lon [r]: a map spot in game coordinates, and the pilot's lanes, gores and links within r m (default 15)."""
import json, os, sys
import numpy as np
sys.path.insert(0, "/home/amy/git/op-ortho/openpilot/tools/sim/bridge/gta5/map/webmap")
import gframe

O = os.path.expanduser("~/.cache/paintnet/pilot")
lat, lon = float(sys.argv[1]), float(sys.argv[2])
R = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
p = np.array([0.0, 0.0])
for _ in range(30):  # Newton on the map's frame
  f = np.array(gframe.lat_lon(np.array([p[0]]), np.array([p[1]]))).ravel()
  J = np.stack([(np.array(gframe.lat_lon(np.array([p[0] + h[0]]), np.array([p[1] + h[1]]))).ravel() - f) for h in ([1, 0], [0, 1])], 1)
  p -= np.linalg.solve(J, f - [lat, lon])
print(f"game x {p[0]:.1f} y {p[1]:.1f}")
G = json.load(open(f"{O}/ic_groups.json"))
L = json.load(open(f"{O}/ic_links.json"))
near = lambda A: float(np.linalg.norm(np.asarray(A)[:, :2] - p, axis=1).min())
lid = lambda k: f"{G['lanes'][k]['left']}|{G['lanes'][k]['right']}|{k}"
for k, r in enumerate(G["lanes"]):
  A = np.array(L["pts"].get(str(k), r["pts"]))
  if near(A) < R:
    print(f"lane {lid(k)} {len(A)} samples, {near(A):.1f} m away, start {A[0, :2].round(1)} w {A[0, 3]} end {A[-1, :2].round(1)} w {A[-1, 3]} z {A[0, 2]:.1f}")
for k, r in enumerate(json.load(open(f"{O}/ic_dir.json"))):
  if r["kind"] != "lane" and near(r["pts"]) < R:
    A = np.array(r["pts"])
    print(f"{r['kind']} {r['left']}|{r['right']} {len(A)} samples, start {A[0, :2].round(1)} w {A[0, 3]} end {A[-1, :2].round(1)} w {A[-1, 3]}")
for i, j, g, *br in L["links"]:
  if near(G["lanes"][i]["pts"]) < R or near(G["lanes"][j]["pts"]) < R:
    print("link", lid(i), "->", lid(j), g, *br)
