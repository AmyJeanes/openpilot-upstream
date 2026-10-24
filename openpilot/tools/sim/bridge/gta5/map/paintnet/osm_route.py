"""osm_route.py x0 x1 y0 y1 in.osm workdir out.geojson: routes over the OSM export with Valhalla, as a navigation app
would, and checks it drives the junctions the way the export describes them.

- the export becomes Valhalla tiles in workdir (ids made positive: Valhalla's are unsigned)
- at each junction (osm.py's paintnet:junction nodes), every move from an arm into another, as a trip from START m up the
  arm (on along the straightest road behind it) to END m down the other:
  - a move no restriction bans is driven through the junction, or round it by a shorter road (a slip road), not by a
    detour; a banned move isn't driven through it
  - the instruction there turns the move's way (straight on: none, or straight or slight)
  - the lanes Valhalla marks valid for the move are those whose turn:lanes arrows make it ("none" lanes when no arrow does)
- no route turns across traffic (more than ACROSS) at a node that isn't a junction
- every road into the box (a loose end at its edge) reaches every road out of it: the pairs that don't are listed, not
  flagged (a freeway and the streets may join outside it)
Out: points with "why" (and "check", "way") and the flagged moves' routes, also printed; every move printed."""
import sys, os, math, json, collections, shutil, subprocess

VENV = "/home/amy/gta5map/.venv/bin"  # pyvalhalla and pyosmium (map/README.md)
try:
  import valhalla, osmium
except ImportError:
  sys.stdout.flush()
  os.environ["PATH"] = VENV + os.pathsep + os.environ["PATH"]
  os.execv(f"{VENV}/python", [f"{VENV}/python"] + sys.argv)
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, "/home/amy/git/op-ortho/openpilot/tools/sim/bridge/gta5/map/webmap")
import gframe

x0, x1, y0, y1 = map(float, sys.argv[1:5])
src, work, out = sys.argv[5:8]
EDGE = 8.0  # m: osm_check.py's, ends this near the box's edge are cut by it
ARM = 10.0  # m: osm_check.py's, a way's heading at a node is over its first ARM m from it
START = 60.0  # m up an arm a trip starts: before a slip road leaves it (they leave within osm_view.py's JUNCTION_REACH 35)
END = 40.0  # m down an arm a trip ends
STOP_SHORT = 3.0  # m a trip's end stays back from a junction or the box's edge it meets first
SLACK = 10.0  # m a route may run over the trip's own length (snapping) and not be a detour
BEARING = 25.0  # deg: a route's bearings at a junction this near an arm's are that arm
ACROSS = 100.0  # deg: a turn this sharp crosses the other way's traffic (osm_check.py's left band reaches 165, slight 80)
TURNS = {"through": (-40, 40), "slight_left": (10, 80), "slight_right": (-80, -10), "left": (35, 165), "right": (-165, -35),
         "sharp_left": (100, 180), "sharp_right": (-180, -100), "reverse": (150, 180)}  # osm_check.py's
BANS = {"no_left_turn": "left", "no_right_turn": "right", "no_straight_on": "through", "no_u_turn": "reverse"}


def to_xy(lat, lon):
  mx, my = gframe.latlon_to_merc(lat, lon)
  return np.array([(mx - 0.5) * gframe.K, (gframe.MY0 - my) * gframe.K])


def tiles(src, d):
  """Valhalla's tiles and config for the export, in d."""
  os.makedirs(d, exist_ok=True)
  root = ET.parse(src).getroot()
  for el in root.iter():
    for k in ("id", "ref"):
      if el.get(k):
        el.set(k, el.get(k).lstrip("-"))
  ET.ElementTree(root).write(f"{d}/export.osm", encoding="utf-8", xml_declaration=True)

  class Copy(osmium.SimpleHandler):
    def __init__(self, w):
      super().__init__()
      self.node, self.way, self.relation = w.add_node, w.add_way, w.add_relation

  pbf = f"{d}/export.osm.pbf"
  if os.path.exists(pbf):
    os.remove(pbf)
  w = osmium.SimpleWriter(pbf)
  Copy(w).apply_file(f"{d}/export.osm")
  w.close()
  shutil.rmtree(f"{d}/tiles", ignore_errors=True)
  c = json.loads(subprocess.run(["valhalla_build_config", "--mjolnir-tile-dir", f"{d}/tiles", "--mjolnir-tile-extract", "",
                                 "--mjolnir-timezone", "", "--mjolnir-admin", ""], capture_output=True, text=True, check=True).stdout)
  c["mjolnir"]["shortcuts"] = False  # a minute's search for none on a map this small
  cfg = f"{d}/valhalla.json"
  json.dump(c, open(cfg, "w"), indent=1)
  with open(f"{d}/build_tiles.log", "w") as log:
    subprocess.run(["valhalla_build_tiles", "-c", cfg, pbf], stdout=log, stderr=log, check=True)
  return cfg


root = ET.parse(src).getroot()
nodes, ways, rels, node_tags = {}, {}, [], {}
for el in root:
  tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
  if el.tag == "node":
    nodes[el.get("id")] = to_xy(float(el.get("lat")), float(el.get("lon")))
    node_tags[el.get("id")] = tags
  elif el.tag == "way":
    ways[el.get("id")] = ([n.get("ref") for n in el.findall("nd")], tags)
  elif el.tag == "relation":
    rels.append(({m.get("role"): m.get("ref") for m in el.findall("member")}, tags))
roads = {w: v for w, v in ways.items() if v[1].get("highway") and v[1]["highway"] not in ("footway", "path", "cycleway")}
flags = []


def flag(check, p, why, way=None, route=None):
  flags.append((check, p, why, way, route))


def counts(t):
  """forward, backward and both-ways lane counts."""
  n = int(t.get("lanes", "1"))
  both = int(t.get("lanes:both_ways", "0"))
  if t.get("oneway") == "yes":
    return n, 0, both
  f = int(t.get("lanes:forward", (n - both) // 2 + (n - both) % 2))
  return f, int(t.get("lanes:backward", n - both - f)), both


def per(t, key, d):
  """a per-lane tag's values for direction d ("forward"/"backward") of the way, or None."""
  v = t.get(f"{key}:lanes" if t.get("oneway") == "yes" else f"{key}:lanes:{d}")
  return v.split("|") if v is not None else None


def away(w, i, back):
  """unit heading leaving way w's node i along it (towards its start if back), over its first ARM m."""
  nd = ways[w][0]
  seq = nd[i::-1] if back else nd[i:]
  p, s = nodes[seq[0]], 0.0
  q = p
  for m in seq[1:]:
    s += np.linalg.norm(nodes[m] - q)
    q = nodes[m]
    if s >= ARM:
      break
  d = q - p
  return d / max(np.linalg.norm(d), 1e-9)


def turn(h_in, h_out):
  """signed angle (deg, left positive) from arriving heading to leaving heading."""
  return math.degrees(math.atan2(h_in[0] * h_out[1] - h_in[1] * h_out[0], h_in @ h_out))


bearing = lambda h: math.degrees(math.atan2(h[0], h[1])) % 360
bdiff = lambda a, b: abs((a - b + 180) % 360 - 180)
near_edge = lambda p: p[0] < x0 + EDGE or p[0] > x1 - EDGE or p[1] < y0 + EDGE or p[1] > y1 - EDGE
touch = collections.defaultdict(list)
for w, (nd, t) in roads.items():
  for i, n in enumerate(nd):
    touch[n].append((w, i))
junction = {n for n in touch if "paintnet:junction" in node_tags[n]}


def ends_at(n):
  """[(way, index of n in it, at_last)] for road ways ending at n; a way passing through counts as two ends."""
  out_ = []
  for w, i in touch[n]:
    if i > 0:
      out_.append((w, i, True))
    if i < len(ways[w][0]) - 1:
      out_.append((w, i, False))
  return out_


def lanes_in(w, at_last):
  """(lanes travelling into the node at this end, lanes travelling out of it)."""
  f, b, both = counts(roads[w][1])
  return (f, b) if at_last else (b, f)


def walk(w, i, back, dist, upstream):
  """(point, heading of travel there, metres) dist m from way w's node i, walking along it (towards its start if back)
  and on along the straightest road carrying traffic the same way (towards the walker if upstream); short of dist by
  STOP_SHORT at a junction or the box's edge met first."""
  P = [nodes[ways[w][0][i]]]
  for _ in range(50):
    nd = ways[w][0]
    seq = nd[i::-1] if back else nd[i:]
    P += [nodes[m] for m in seq[1:]]
    m, im = seq[-1], 0 if back else len(nd) - 1
    if sum(np.linalg.norm(np.diff(P, axis=0), axis=1)) >= dist or m in junction or near_edge(nodes[m]):
      break
    d = P[-1] - P[-2]
    nxt = [(w2, i2, l2) for w2, i2, l2 in ends_at(m) if (w2, i2, l2) != (w, im, not back) and lanes_in(w2, l2)[0 if upstream else 1]]
    if not nxt:
      break
    w, i, back = max(nxt, key=lambda e: away(*e) @ d)
  P = np.array(P)
  L = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
  s = dist if L[-1] >= dist else max(L[-1] - STOP_SHORT, L[-1] / 2)
  k = min(np.searchsorted(L, s), len(P) - 1)
  seg = P[k] - P[k - 1]
  p = P[k - 1] + seg * (s - L[k - 1]) / max(L[k] - L[k - 1], 1e-9)
  h = seg / max(np.linalg.norm(seg), 1e-9)
  return p, (-h if upstream else h), s


def loc(p, h):
  lat, lon = gframe.lat_lon(np.array([p[0]]), np.array([p[1]]))
  return {"lat": float(lat[0]), "lon": float(lon[0]), "heading": round(bearing(h)), "heading_tolerance": 45}


def decode(s):
  """a polyline6 as [(lon, lat)]."""
  out_, i, v = [], 0, [0, 0]
  while i < len(s):
    for k in range(2):
      r, sh = 0, 0
      while True:
        b = ord(s[i]) - 63
        i += 1
        r |= (b & 0x1F) << sh
        sh += 5
        if b < 0x20:
          break
      v[k] += ~(r >> 1) if r & 1 else r >> 1
    out_.append((v[1] / 1e6, v[0] / 1e6))
  return out_


actor = valhalla.Actor(tiles(src, work))


across = {}  # node -> (in bearing, out bearing, route line): a route turning more than ACROSS where no junction is


def route(a, b):
  """Valhalla's route (OSRM form) between two loc()s, or None; notes where it turns across traffic away from junctions."""
  try:
    r = json.loads(actor.route(json.dumps({"locations": [a, b], "costing": "auto", "format": "osrm", "units": "km"})))["routes"][0]
  except (RuntimeError, KeyError, IndexError):
    return None
  for s in r["legs"][0]["steps"]:
    for x in s["intersections"]:
      if x.get("in") is None or x.get("out") is None:
        continue
      p = to_xy(x["location"][1], x["location"][0])
      n = min(touch, key=lambda m: np.linalg.norm(nodes[m] - p))
      bi, bo = x["bearings"][x["in"]], x["bearings"][x["out"]]
      if n not in junction and not near_edge(nodes[n]) and np.linalg.norm(nodes[n] - p) < 1.0 and 180 - bdiff(bi, bo) > ACROSS:
        across.setdefault(n, (bi, bo, decode(r["geometry"])))
  return r


def compass(h):
  return ["north", "east", "south", "west"][int((bearing(h) + 45) % 360 // 90)]


def lanes_for(tv, a):
  """lanes whose arrows make a turn of a deg ("none" lanes if no arrow does)."""
  vals = [lane.split(";") if lane else ["none"] for lane in tv]
  make = [k for k, v in enumerate(vals) if any(TURNS[x][0] <= a <= TURNS[x][1] for x in v if x in TURNS)]
  return make or [k for k, v in enumerate(vals) if "none" in v]


SIDES = {"straight": {None, "straight", "slight left", "slight right"}, "left": {"slight left", "left", "sharp left"},
         "right": {"slight right", "right", "sharp right"}, "reverse": {"uturn"}}
bans = collections.defaultdict(list)  # (from way, via) -> [(to way, kind)]
for m, t in rels:
  if t.get("type") == "restriction":
    bans[m.get("from"), m.get("via")].append((m.get("to"), t.get("restriction", "")))

moves = 0
for n in sorted(junction, key=int):
  J = nodes[n]
  E = ends_at(n)
  arm = {(w, l): f"{compass(away(w, i, l))} arm ({roads[w][1].get('name', 'way ' + w)})" for w, i, l in E}
  for w, i, l in E:
    if not lanes_in(w, l)[0]:
      continue
    h_in = -away(w, i, l)
    a_pt, a_h, a_len = walk(w, i, l, START, True)
    tv = per(roads[w][1], "turn", "forward" if l else "backward") if i in (0, len(ways[w][0]) - 1) else None
    for w2, i2, l2 in E:
      if not lanes_in(w2, l2)[1]:
        continue
      u = (w2, l2) == (w, l)
      a = 180.0 if u else turn(h_in, away(w2, i2, l2))
      kinds = [k for to, k in bans[w, n] if to == w2 and (w2 != w or u)]
      banned = any(k.startswith("no_") for k in kinds) or (any(k.startswith("only_") for _, k in bans[w, n]) and not
                                                             any(k.startswith("only_") for k in kinds))
      b_pt, b_h, b_len = walk(w2, i2, l2, END, False)
      r = route(loc(a_pt, a_h), loc(b_pt, b_h))
      name = f"node {n} {arm[w, l]} -> {arm[w2, l2]} ({a:+.0f}°{', banned' if banned else ''})"
      moves += 1
      if r is None:
        if not banned:
          flag("route", J, f"{name}: no route", w)
        print(f" {name}: no route")
        continue
      line = decode(r["geometry"])
      steps = r["legs"][0]["steps"]
      at_j = [(s, x) for s in steps for x in s["intersections"] if np.linalg.norm(to_xy(x["location"][1], x["location"][0]) - J) < 1.0]
      made = [(s, x) for s, x in at_j if x.get("in") is not None and x.get("out") is not None
              and bdiff(x["bearings"][x["in"]], bearing(-h_in)) < BEARING and bdiff(x["bearings"][x["out"]], bearing(away(w2, i2, l2))) < BEARING]
      length = r["distance"]  # m, whatever "units" says (the OSRM form)
      by = ": " + " ".join(s["maneuver"]["instruction"] for s in steps if s["maneuver"]["type"] not in ("depart", "arrive"))
      if banned:
        status = "driven through the junction" if made else f"not made here: {length:.0f} m{by}"
        if made:
          flag("banned", J, f"{name}: a banned move, driven through the junction", w, line)
      elif made:
        status = "through the junction"
      elif not at_j and length <= a_len + b_len + SLACK:
        status = f"round the junction: {length:.0f} m{by}"
      else:
        status = f"detour: {length:.0f} m for {a_len + b_len:.0f}{by}"
        flag("detour", J, f"{name}: {status}", w, line)
      extra = ""
      if made:
        s, x = made[0]
        mod = s["maneuver"].get("modifier") if s["maneuver"]["type"] not in ("depart", "arrive") and \
            np.linalg.norm(to_xy(s["maneuver"]["location"][1], s["maneuver"]["location"][0]) - J) < 1.0 else None
        side = "reverse" if u else "straight" if -40 <= a <= 40 else "left" if a > 0 else "right"
        if mod not in SIDES[side]:
          flag("instruction", J, f"{name}: told '{s['maneuver']['type']} {mod}' for a {side} move", w, line)
        extra = f"; told {s['maneuver']['type'] if mod else 'nothing'}{' ' + mod if mod else ''}"
        if tv and not u:
          want = lanes_for(tv, a)
          got = x.get("lanes") or []
          valid = [k for k, ln in enumerate(got) if ln.get("valid")]
          active = [k for k, ln in enumerate(got) if ln.get("active")]
          if len(got) != len(tv) or valid != want:
            flag("lanes", J, f"{name}: Valhalla's lanes {valid} of {len(got)} valid, the arrows' {want} of {len(tv)}", w, line)
          extra += f"; lanes valid {valid} active {active} of {'|'.join(tv)}"
      print(f" {name}: {status}{extra}")

# roads into and out of the box: every pair joined
ins, outs = [], []
for n in touch:
  if not near_edge(nodes[n]) or len(ends_at(n)) != 1:  # a road's loose end at the box's edge
    continue
  for w, i, l in ends_at(n):
    if lanes_in(w, l)[1]:
      ins.append((w, walk(w, i, l, 15.0, False)))
    if lanes_in(w, l)[0]:
      outs.append((w, walk(w, i, l, 15.0, True)))
missing, tried = [], 0
for wa, (pa, ha, _) in ins:
  for wb, (pb, hb, _) in outs:
    tried += wa != wb
    if wa != wb and route(loc(pa, ha), loc(pb, hb)) is None:
      missing.append((wa, wb))

for n, (bi, bo, line) in across.items():
  t = 180 - bdiff(bi, bo)
  flag("across", nodes[n], f"node {n}: a route turns {t:.0f}° here, where no junction is (from bearing {(bi + 180) % 360:.0f} to {bo:.0f})",
       None, line)

feats = []
for check, p, why, w, line in flags:
  lat, lon = gframe.lat_lon(np.array([p[0]]), np.array([p[1]]))
  feats.append({"type": "Feature", "properties": {"kind": "osm_route", "check": check, "why": why, "way": w},
                "geometry": {"type": "Point", "coordinates": [round(float(lon[0]), 8), round(float(lat[0]), 8)]}})
  if line:
    feats.append({"type": "Feature", "properties": {"kind": "osm_route_line", "check": check, "why": why, "way": w},
                  "geometry": {"type": "LineString", "coordinates": [[round(a, 8), round(b, 8)] for a, b in line]}})
json.dump({"type": "FeatureCollection", "features": feats}, open(out, "w"))
kind = lambda w: roads[w][1]["highway"].removesuffix("_link")
print(len(junction), "junctions,", moves, "moves;", len(ins), "roads in,", len(outs), "out,", tried - len(missing), "pairs joined,",
      len(missing), "not", dict(collections.Counter(f"{kind(a)}->{kind(b)}" for a, b in missing)))
for a, b in missing:
  print(f" not joined: {kind(a)} way {a} ({roads[a][1].get('name', '')}) -> {kind(b)} way {b} ({roads[b][1].get('name', '')})")
print(len(flags), "findings", dict(collections.Counter(c for c, *_ in flags)))
for check, p, why, *_ in flags:
  print(f" {check:11s} ({p[0]:.0f}, {p[1]:.0f}) {why}")
