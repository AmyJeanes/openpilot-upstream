"""osm_view.py in.osm out.geojson: the road as an OSM consumer would rebuild it from the export's tags alone, to see what
the map says without reading tags: each road way's lanes laid out across it from lanes (+:forward/:backward/:both_ways)
and width:lanes (else width shared out, else LANE m each; width:lanes:start / :end: the lanes
narrowing or widening evenly between the way's first and last node), the way down the middle of them (OSM's default placement);
backward lanes on its left, forward on its right (right-hand traffic). Lines between lanes from change:lanes (dashed:
both may cross; solid: neither, or one only), the centre line yellow (double where divider=*, a two-way turn lane's
solid+dashed), the kerbs white; shoulders from shoulder(:width); turn:lanes as arrows near each direction's far end;
signals as a stop line across their direction's lanes; footway=crossing as zebra stripes; restriction relations as a
label at the from way's end. Ways meeting at a junction node (three ends or more, two of them crossing) stop short of it
by half the widest other way there, the gap drawn as the junction's area. placement(:forward) puts the way among its
lanes (else it's their middle).
Out: features with "kind": lane (dir forward|backward|both, highway), shoulder, junction, line (style edge|dashed|solid|
centre|centre_dashed|double), zebra, stop, signal, arrow (text, bearing), restriction (text), each with the way's id."""
import sys, math, collections
import xml.etree.ElementTree as ET
import numpy as np
from shapely.geometry import LineString, Point, Polygon, MultiPoint
from shapely.ops import substring

sys.path.insert(0, "/home/amy/git/op-ortho/openpilot/tools/sim/bridge/gta5/map/webmap")
import gframe

src, out = sys.argv[1:3]
LANE = 3.5  # m: a lane's width where the tags give none
LINE = 0.15  # m: a painted line's width
DASH, GAP_ = 3.0, 6.0  # m: a dashed line's dash and gap
ARROW_BACK = 6.0  # m before a way's far end its turn arrows sit
STOP_BACK = 8.0  # m of the way behind a signal its stop line is drawn square to
ZEBRA, ZEBRA_W = 0.5, 3.0  # m: a crossing's stripe and its width across
HEADING = {"north": "#1e88e5", "east": "#43a047", "south": "#8e24aa", "west": "#00897b"}  # export.py's lane colours
GLYPH = {"left": "←", "slight_left": "↖", "sharp_left": "←", "through": "↑", "slight_right": "↗", "right": "→", "sharp_right": "→",
         "merge_to_left": "↖", "merge_to_right": "↗", "reverse": "U", "none": ""}


def to_xy(lat, lon):
  mx, my = gframe.latlon_to_merc(lat, lon)
  return (mx - 0.5) * gframe.K, (gframe.MY0 - my) * gframe.K


root = ET.parse(src).getroot()
nodes, ways, rels = {}, {}, []
for el in root:
  tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
  if el.tag == "node":
    x, y = to_xy(float(el.get("lat")), float(el.get("lon")))
    nodes[el.get("id")] = (float(x), float(y), tags)
  elif el.tag == "way":
    ways[el.get("id")] = ([n.get("ref") for n in el.findall("nd")], tags)
  elif el.tag == "relation":
    rels.append(({m.get("role"): m.get("ref") for m in el.findall("member")}, tags))
xy = lambda n: (nodes[n][0], nodes[n][1])
roads = {w: v for w, v in ways.items() if v[1].get("highway") and v[1]["highway"] not in ("footway", "path", "cycleway")}


def layout(t, at=""):
  """the way's lanes left to right along it: [(direction, width, turn, change)], and its total width; at "start" or
  "end": at the way's first or last node, where width(:lanes):start / :end say they differ."""
  t = {**t, **{k[:-len(at) - 1]: v for k, v in t.items() if at and k.endswith(":" + at) and k.startswith("width")}}
  n = int(t.get("lanes", "1"))
  oneway = t.get("oneway") == "yes"
  both = int(t.get("lanes:both_ways", "0"))
  fwd = n if oneway else int(t.get("lanes:forward", (n - both) // 2 + (n - both) % 2))
  bwd = 0 if oneway else int(t.get("lanes:backward", n - both - fwd))
  sfx = ("", "") if oneway else (":forward", ":backward")

  def per(key, k, default):
    v = t.get(key)
    vals = v.split("|") if v else []
    return [vals[i] if i < len(vals) else default for i in range(k)]

  share = float(t["width"]) / n if "width" in t and "width:lanes" not in t else LANE
  wf = [float(v or share) for v in per("width:lanes" + sfx[0], fwd, share)]
  wb = [float(v or share) for v in per("width:lanes" + sfx[1], bwd, share)]
  tf, tb = per("turn:lanes" + sfx[0], fwd, ""), per("turn:lanes" + sfx[1], bwd, "")
  cf, cb = per("change:lanes" + sfx[0], fwd, "yes"), per("change:lanes" + sfx[1], bwd, "yes")
  mean = float(np.mean(wf + wb)) if wf + wb else LANE
  # backward lanes are listed left to right as their drivers see them: right to left along the way
  L = [("backward", w, tu, c) for w, tu, c in zip(wb, tb, cb)][::-1] + [("both", mean, "", "yes")] * both + \
      [("forward", w, tu, c) for w, tu, c in zip(wf, tf, cf)]
  T = sum(l[1] for l in L)
  # placement: where the way runs among its lanes (left_of / middle_of / right_of lane N, counted from the left; on a
  # two-way road placement:forward counts its forward lanes from the centre); none: the middle of them all
  pl, first = (t.get("placement"), 0) if oneway else (t.get("placement:forward"), bwd + both)
  left = T / 2  # m from the lanes' left edge to the way
  if pl and ":" in pl:
    how, n_ = pl.split(":")
    q = first + int(n_) - 1
    if 0 <= q < len(L):
      e = sum(l[1] for l in L[:q])
      left = {"left_of": e, "middle_of": e + L[q][1] / 2, "right_of": e + L[q][1]}.get(how, T / 2)
  return L, T, left


# junction nodes: three road way ends or more (or a way passing through and one ending), two of them crossing (a fork
# or merge, its ways all along one line, isn't one)
inc, away = collections.Counter(), collections.defaultdict(list)
for nd, t in roads.values():
  for k, n in enumerate(nd):
    inc[n] += 1 if k in (0, len(nd) - 1) else 2
    for q in (k - 1, k + 1):
      if 0 <= q < len(nd):
        away[n].append(math.atan2(nodes[nd[q]][1] - nodes[n][1], nodes[nd[q]][0] - nodes[n][0]))
crossing = lambda A: any(45 < abs((math.degrees(a - b) + 180) % 360 - 180) < 135 for a in A for b in A)
junction = {n for n, c in inc.items() if c >= 3 and crossing(away[n])}
lay = {w: (layout(t, "start"), layout(t, "end")) for w, (nd, t) in roads.items()}


def heading_at(nd, n):
  """unit heading of a way leaving node n (one of its ends) along its first segment."""
  q = nd[1] if nd[0] == n else nd[-2]
  d = np.subtract(xy(q), xy(n))
  return d / max(np.linalg.norm(d), 1e-9)


# each way stops short of a junction node by half the widest way there crossing it (a road carrying straight on through a
# side road's or slip road's node is cut back by the side road's half width, not its own)
trim_at = {}
for n in junction:
  here = [(w, nd) for w, (nd, _) in roads.items() if n in (nd[0], nd[-1])]
  for w, nd in here:
    h = heading_at(nd, n)
    cross = [lay[w2][nd2.index(n) != 0][1] for w2, nd2 in here if w2 != w and abs(h @ heading_at(nd2, n)) < math.cos(math.radians(30))]
    trim_at[n, w] = max(cross) / 2 if cross else 0.0

feats = []


def emit(geom, **p):
  if geom.is_empty:
    return
  feats.append({"type": "Feature", "properties": p, "geom": geom})


def offset(line, d):
  """the line moved d m to its right; d (start, end): moving evenly from one to the other along it."""
  if isinstance(d, tuple):
    if abs(d[1] - d[0]) > 1e-3:
      n_ = max(int(line.length), 2)
      pts = []
      for s_ in np.linspace(0, line.length, n_ + 1):
        c, _, nrm = frame_at(line, s_)
        pts.append(tuple(c + nrm * (d[0] + (d[1] - d[0]) * s_ / max(line.length, 1e-9))))
      return LineString(pts)
    d = d[0]
  if abs(d) < 1e-6:
    return line
  g = line.offset_curve(-d, join_style=2, mitre_limit=3.0)
  return g if g.geom_type == "LineString" else max(g.geoms, key=lambda q: q.length)


def plus(a, b):
  """offsets (numbers, or (start, end) pairs) added."""
  if isinstance(a, tuple) or isinstance(b, tuple):
    a, b = (a if isinstance(a, tuple) else (a, a)), (b if isinstance(b, tuple) else (b, b))
    return a[0] + b[0], a[1] + b[1]
  return a + b


def band(line, a, b):
  """the strip between offsets a and b (m right of the line)."""
  A, B = offset(line, a), offset(line, b)
  P = Polygon(list(A.coords) + list(B.coords)[::-1])
  return P if P.is_valid else P.buffer(0)


def frame_at(line, s):
  """the point s m along the line, the unit way along it there, and the unit normal to its right."""
  a = np.array(line.interpolate(max(s - 0.5, 0)).coords[0])
  b = np.array(line.interpolate(min(s + 0.5, line.length)).coords[0])
  d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
  return np.array(line.interpolate(s).coords[0]), d, np.array([d[1], -d[0]])


def dashed(line):
  out, s = [], 0.0
  while s < line.length:
    out.append(substring(line, s, min(s + DASH, line.length)))
    s += DASH + GAP_
  return out


ends = collections.defaultdict(list)  # junction node -> trimmed end points
way_line = {}
for w, (nd, t) in roads.items():
  line = whole = LineString([xy(n) for n in nd])
  full = line.length
  a, b = 0.0, line.length
  if nd[0] in junction:
    a = min(trim_at.get((nd[0], w), 0.0), line.length / 3)
  if nd[-1] in junction:
    b = max(line.length - trim_at.get((nd[-1], w), 0.0), line.length * 2 / 3)
  line = substring(line, a, b)
  if line.length < 0.5:
    continue
  way_line[w] = line
  (L, T0, left0), (L1, T1, left1) = lay[w]
  at = lambda v0, v1: (v0 + (v1 - v0) * a / full, v0 + (v1 - v0) * b / full)  # (at the trimmed ends)
  T, left = at(T0, T1), at(left0, left1)
  hw = t["highway"]
  for n, p, i in ((nd[0], line.coords[0], 0), (nd[-1], line.coords[-1], 1)):
    if n in junction:
      d, nrm = np.subtract(line.coords[1], line.coords[0]) if p == line.coords[0] else np.subtract(line.coords[-1], line.coords[-2]), None
      d = d / np.linalg.norm(d)
      nrm = np.array([d[1], -d[0]])
      ends[n] += [tuple(np.add(p, nrm * (T[i] - left[i]))), tuple(np.subtract(p, nrm * left[i]))]
  o = (-left[0], -left[1])
  edges = [o]
  run = np.subtract(line.coords[-1], line.coords[0])
  for k, ((dr, w0, tu, ch), (_, w1, _, _)) in enumerate(zip(L, L1)):
    wd = at(w0, w1)
    d = run if dr == "forward" else -run
    hd = ["east", "north", "west", "south"][int(((math.degrees(math.atan2(d[1], d[0])) % 360 + 45) % 360) // 90)]
    emit(band(line, o, plus(o, wd)), kind="lane", dir=dr, highway=hw, way=w, col="#ffb74d" if dr == "both" else HEADING[hd])
    o = plus(o, wd)
    edges.append(o)
  # lines: kerbs, then between each pair of lanes
  for e in (edges[0], edges[-1]):
    emit(offset(line, e).buffer(LINE / 2, cap_style=2), kind="line", style="edge", way=w)
  for k in range(len(L) - 1):
    (d1, _, _, c1), (d2, _, _, c2) = L[k], L[k + 1]
    ln = offset(line, edges[k + 1])
    if d1 != d2:
      if "both" in (d1, d2):
        style = "twltl"  # solid on the through side, dashed on the turn lane's
        sgn = 1 if d1 == "both" else -1
        emit(offset(ln, sgn * 0.2).buffer(LINE / 2, cap_style=2), kind="line", style="centre", way=w)
        for q in dashed(offset(ln, -sgn * 0.2)):
          emit(q.buffer(LINE / 2, cap_style=2), kind="line", style="centre", way=w)
      elif t.get("divider"):
        for s_ in (-0.2, 0.2):
          emit(offset(ln, s_).buffer(LINE / 2, cap_style=2), kind="line", style="centre", way=w)
      else:
        for q in dashed(ln):
          emit(q.buffer(LINE / 2, cap_style=2), kind="line", style="centre", way=w)
      continue
    # same direction: the lane on the drivers' left (k for forward, k+1 for backward) crosses to its right
    left, right = (c1, c2) if d1 == "forward" else (c2, c1)
    may_r = left in ("yes", "not_left")
    may_l = right in ("yes", "not_right")
    if may_r and may_l:
      for q in dashed(ln):
        emit(q.buffer(LINE / 2, cap_style=2), kind="line", style="dashed", way=w)
    else:
      emit(ln.buffer(LINE / 2, cap_style=2), kind="line", style="solid", way=w)
  # shoulders
  sh = t.get("shoulder")
  if sh in ("both", "left", "right"):
    for side, e, sg in (("left", edges[0], -1), ("right", edges[-1], 1)):
      if sh in (side, "both"):
        sw = float(t.get(f"shoulder:{side}:width", t.get("shoulder:width", "1.5")))
        emit(band(line, e, plus(e, sg * sw)), kind="shoulder", way=w)
  # turn arrows near each direction's far end
  for dr, far in (("forward", 1), ("backward", 0)):
    lanes_ = [(k, l) for k, l in enumerate(L) if l[0] == dr]
    if not any(l[2] for _, l in lanes_):
      continue
    s = max(line.length - ARROW_BACK, line.length / 2) if far else min(ARROW_BACK, line.length / 2)
    p, tang, nrm = frame_at(line, s)
    head = tang if dr == "forward" else -tang
    for k, (_, wd, tu, _) in lanes_:
      txt = "".join(GLYPH.get(v, "?") for v in tu.split(";")) if tu else ""
      if txt:
        f = s / line.length
        c = p + nrm * ((1 - f) * (edges[k][0] + edges[k + 1][0]) / 2 + f * (edges[k][1] + edges[k + 1][1]) / 2)
        emit(Point(c), kind="arrow", text=txt, bearing=round(math.degrees(math.atan2(head[0], head[1])), 1), way=w)
  # signals: a stop line across their direction's lanes where the way passes the node
  for k, n in enumerate(nd):
    tg = nodes[n][2]
    if tg.get("highway") != "traffic_signals" and tg.get("highway") != "stop":
      continue
    dr = tg.get("traffic_signals:direction", tg.get("direction", "forward"))
    p = Point(xy(n))  # (on the way before it's cut back at a junction: a stop line may sit at a slip road's node)
    s = whole.project(p)
    f = min(max((s - a) / max(b - a, 1e-9), 0.0), 1.0)
    E = [(1 - f) * e[0] + f * e[1] for e in edges]
    span = [(E[i], E[i + 1]) for i, l in enumerate(L) if l[0] == dr] or [(E[0], E[-1])]
    a_, b_ = span[0][0], span[-1][1]
    # square to the lanes traffic stops in, behind the node (past it the way may bend on to its junction's node)
    c = np.array(whole.interpolate(s).coords[0])
    q = np.array(whole.interpolate(max(s - STOP_BACK, 0) if dr == "forward" else min(s + STOP_BACK, whole.length)).coords[0])
    d = (c - q) * (1 if dr == "forward" else -1)
    d = d / max(np.linalg.norm(d), 1e-9) if np.linalg.norm(d) > 0.5 else frame_at(whole, s)[1]
    nrm = np.array([d[1], -d[0]])
    emit(Polygon([c + nrm * a_ - d * 0.25, c + nrm * a_ + d * 0.25, c + nrm * b_ + d * 0.25, c + nrm * b_ - d * 0.25]), kind="stop", way=w)
    emit(Point(c + nrm * (b_ + 1.0 if dr == "forward" else a_ - 1.0)), kind="signal", control=tg["highway"], way=w)

# junction areas
for n, pts in ends.items():
  if len(pts) >= 4:
    emit(MultiPoint(pts + [xy(n)]).convex_hull, kind="junction", node=n)

# crossings: zebra stripes along the footway, each ZEBRA_W across it
for w, (nd, t) in ways.items():
  if t.get("footway") != "crossing":
    continue
  line = LineString([xy(n) for n in nd])
  s = 0.25
  while s < line.length - 0.25:
    a = np.array(line.interpolate(s).coords[0])
    b = np.array(line.interpolate(min(s + 0.1, line.length)).coords[0])
    d = (b - a) / max(np.linalg.norm(b - a), 1e-9)
    nrm = np.array([d[1], -d[0]])
    c0 = a
    c1 = a + d * ZEBRA
    emit(Polygon([c0 - nrm * ZEBRA_W / 2, c1 - nrm * ZEBRA_W / 2, c1 + nrm * ZEBRA_W / 2, c0 + nrm * ZEBRA_W / 2]), kind="zebra", way=w,
         signals=t.get("crossing") == "traffic_signals")
    s += 2 * ZEBRA

# restrictions: a label at the from way's end by the via node
WORD = {"no_u_turn": "no U", "no_left_turn": "no ←", "no_right_turn": "no →", "no_straight_on": "no ↑",
        "only_left_turn": "only ←", "only_right_turn": "only →", "only_straight_on": "only ↑"}
stack = collections.Counter()
for m, t in rels:
  if t.get("type") != "restriction" or m.get("from") not in way_line:
    continue
  line, via = way_line[m["from"]], m.get("via")
  near_end = line.coords[-1] if np.hypot(*np.subtract(line.coords[-1], xy(via))) < np.hypot(*np.subtract(line.coords[0], xy(via))) else line.coords[0]
  stack[m["from"]] += 1
  p = np.add(near_end, 0)
  label = WORD.get(t.get("restriction"), t.get("restriction", "?"))
  if m.get("to") != m.get("from") and m.get("to") in roads:
    label += " into " + (roads[m["to"]][1].get("name") or roads[m["to"]][1]["highway"])
  emit(Point(p), kind="restriction", text=label, offset=stack[m["from"]] - 1, way=m["from"])

# out, in lat/lon
coll = []
for f in feats:
  g = f.pop("geom")

  def ll(c):
    la, lo = gframe.lat_lon(np.array([c[0]]), np.array([c[1]]))
    return [round(float(lo[0]), 9), round(float(la[0]), 9)]

  if g.geom_type == "Point":
    gj = {"type": "Point", "coordinates": ll(g.coords[0])}
  elif g.geom_type == "Polygon":
    gj = {"type": "Polygon", "coordinates": [[ll(c) for c in g.exterior.coords]]}
  elif g.geom_type == "MultiPolygon":
    gj = {"type": "MultiPolygon", "coordinates": [[[ll(c) for c in p.exterior.coords]] for p in g.geoms]}
  else:
    gj = {"type": "LineString", "coordinates": [ll(c) for c in g.coords]}
  coll.append({"type": "Feature", "properties": f["properties"], "geometry": gj})
import json
json.dump({"type": "FeatureCollection", "features": coll}, open(out, "w"))
print(len(roads), "road ways,", len(junction), "junction nodes ->", len(coll), "features:", dict(collections.Counter(f["properties"]["kind"] for f in coll)), "->", out)
