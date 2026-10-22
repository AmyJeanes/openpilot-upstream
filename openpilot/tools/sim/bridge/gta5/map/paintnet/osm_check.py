"""osm_check.py x0 x1 y0 y1 in.osm out.geojson: whether the OSM export holds together as a road network, read from its
tags alone (as osm_view.py and a router would): what a spot check of the map can't easily see.

- per-lane tags (turn:, width:, change:lanes with their :forward/:backward and :start/:end) list one value per lane
- at a node that isn't a junction, the lanes arriving and leaving match, or a lane is shown ending (merge_to_left/right,
  or tapered to that end) or starting (tapered from it: under TAPER_POINT there, or TAPER_SHARE of its far end's width)
- traffic can leave every node it can reach, and reach every node it can leave, except at the box's edge (EDGE m)
- at a junction (osm.py's paintnet:junction nodes), every turn:lanes arrow has a way leaving that way
  (by the turn's angle, TURNS), no restriction bans a move an arrow points to, and where every lane has an arrow
  every way traffic can legally leave by is covered by one
- a restriction's from way arrives at its via node, its to way leaves it, and the move is the kind it names
Out: points with "why" (and "check", "way"), also printed."""
import sys, math, collections, json
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, "/home/amy/git/op-ortho/openpilot/tools/sim/bridge/gta5/map/webmap")
import gframe

x0, x1, y0, y1 = map(float, sys.argv[1:5])
src, out = sys.argv[5:7]
EDGE = 8.0  # m: ends this near the box's edge are cut by it (osm.py's EDGE_BOX plus its lines' reach past it)
TAPER_POINT = 1.5  # m: osm.py's, a lane this narrow at a way's end is a taper's point there
TAPER_SHARE = 0.5  # or narrower than this share of its width at the way's other end (a taper's first sample: 1.8 m of 4.6)
ARM = 10.0  # m: a way's heading at a node is over its first ARM m from it (osm.py's ARM)
# signed turn (deg, left positive) each turn:lanes value allows: wide bands, since a junction's arms rarely meet square
TURNS = {"through": (-40, 40), "slight_left": (10, 80), "slight_right": (-80, -10), "left": (35, 165), "right": (-165, -35),
         "sharp_left": (100, 180), "sharp_right": (-180, -100), "reverse": (150, 180)}
BANS = {"no_left_turn": "left", "no_right_turn": "right", "no_straight_on": "through", "no_u_turn": "reverse"}


def to_xy(lat, lon):
  mx, my = gframe.latlon_to_merc(lat, lon)
  return (mx - 0.5) * gframe.K, (gframe.MY0 - my) * gframe.K


root = ET.parse(src).getroot()
nodes, ways, rels, node_tags = {}, {}, [], {}
for el in root:
  tags = {t.get("k"): t.get("v") for t in el.findall("tag")}
  if el.tag == "node":
    nodes[el.get("id")] = np.array(to_xy(float(el.get("lat")), float(el.get("lon"))))
    node_tags[el.get("id")] = tags
  elif el.tag == "way":
    ways[el.get("id")] = ([n.get("ref") for n in el.findall("nd")], tags)
  elif el.tag == "relation":
    rels.append(({m.get("role"): m.get("ref") for m in el.findall("member")}, tags))
roads = {w: v for w, v in ways.items() if v[1].get("highway") and v[1]["highway"] not in ("footway", "path", "cycleway")}
flags = []


def flag(check, p, why, way=None):
  flags.append((check, p, why, way))


def counts(t):
  """forward, backward and both-ways lane counts."""
  n = int(t.get("lanes", "1"))
  both = int(t.get("lanes:both_ways", "0"))
  if t.get("oneway") == "yes":
    return n, 0, both
  f = int(t.get("lanes:forward", (n - both) // 2 + (n - both) % 2))
  return f, int(t.get("lanes:backward", n - both - f)), both


def per(t, key, d, at=""):
  """a per-lane tag's values for direction d ("forward"/"backward") of the way, or None."""
  sfx = "" if t.get("oneway") == "yes" else ":" + d
  v = t.get(f"{key}:lanes{sfx}:{at}" if at else f"{key}:lanes{sfx}")
  if v is None and key == "width" and at:
    v = t.get(f"{key}:lanes{sfx}")
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


near_edge = lambda p: p[0] < x0 + EDGE or p[0] > x1 - EDGE or p[1] < y0 + EDGE or p[1] > y1 - EDGE

# per-lane tags list one value per lane
for w, (nd, t) in roads.items():
  f, b, _ = counts(t)
  for d, k in (("forward", f), ("backward", b)):
    if not k:
      continue
    for key in ("turn", "width", "change"):
      for at in ("", "start", "end"):
        v = per(t, key, d, at) if key == "width" or not at else None
        if v is not None and len(v) != k:
          flag("tags", nodes[nd[len(nd) // 2]], f"{key}:lanes{'' if t.get('oneway') == 'yes' else ':' + d}{':' + at if at else ''} has {len(v)} values for {k} lanes", w)

# what touches each node: (way, index); a way's own ends
touch = collections.defaultdict(list)
for w, (nd, t) in roads.items():
  for i, n in enumerate(nd):
    touch[n].append((w, i))


def ends_at(n):
  """[(way, index of n in it, at_last)] for road ways ending at n (at_last: its last node); a way passing through
  counts as two ends, the half before n (at_last) and the half after."""
  out_ = []
  for w, i in touch[n]:
    last = len(ways[w][0]) - 1
    if i > 0:
      out_.append((w, i, True))
    if i < last:
      out_.append((w, i, False))
  return out_


real_end = lambda w, i: i in (0, len(ways[w][0]) - 1)


def lanes_in(w, at_last):
  """(lanes travelling into the node at this end, lanes travelling out of it)."""
  f, b, both = counts(roads[w][1])
  return (f, b) if at_last else (b, f)


def tapered(t, d, at_last):
  """per lane of direction d: narrowed to a point at this end (under TAPER_POINT, or under TAPER_SHARE of its width at
  the way's other end)."""
  here = per(t, "width", d, "end" if at_last else "start") or []
  there = per(t, "width", d, "start" if at_last else "end") or here
  return [float(a) < TAPER_POINT or float(a) < TAPER_SHARE * float(b) for a, b in zip(here, there)]


def shown_ending(w, at_last):
  """lanes travelling into this end shown ending there: merge_to_*, or tapered to it."""
  t = roads[w][1]
  d = "forward" if at_last else "backward"
  tv, tp = per(t, "turn", d) or [], tapered(t, d, at_last)
  return sum(1 for k in range(max(len(tv), len(tp))) if (k < len(tv) and "merge_to" in tv[k]) or (k < len(tp) and tp[k]))


def shown_starting(w, at_last):
  """lanes travelling out of this end shown starting there: tapered from it."""
  return sum(tapered(roads[w][1], "backward" if at_last else "forward", at_last))


# junctions: osm.py's (a slip road leaving square to its road is a fork, not a junction)
junction = {n for n in touch if "paintnet:junction" in node_tags[n]}

# the directed graph: which nodes traffic can arrive at and leave
can_in, can_out = collections.Counter(), collections.Counter()
for w, (nd, t) in roads.items():
  f, b, both = counts(t)
  for a, c in zip(nd, nd[1:]):
    if f or both:
      can_out[a] += 1
      can_in[c] += 1
    if b or both:
      can_out[c] += 1
      can_in[a] += 1
for n in touch:
  p = nodes[n]
  if near_edge(p):
    continue
  E = ends_at(n)
  if len(E) == 1:
    flag("ends", p, "a road ends here, away from the box's edge, joining nothing", E[0][0])
  if can_in[n] and not can_out[n]:
    flag("flow", p, "traffic arrives here and can't leave", E[0][0])
  elif can_out[n] and not can_in[n]:
    flag("flow", p, "traffic leaves here but can't arrive", E[0][0])

# lanes carried through nodes that aren't junctions
for n in touch:
  if n in junction or near_edge(nodes[n]):
    continue
  E = ends_at(n)
  if len(E) < 2:
    continue
  arrive = sum(lanes_in(w, l)[0] for w, i, l in E)
  leave = sum(lanes_in(w, l)[1] for w, i, l in E)
  ending = sum(shown_ending(w, l) for w, i, l in E if real_end(w, i))
  starting = sum(shown_starting(w, l) for w, i, l in E if real_end(w, i))
  if arrive - ending > leave or leave - starting > arrive:
    flag("lanes", nodes[n], f"{arrive} lanes arrive, {leave} leave; shown ending {ending}, starting {starting} "
         f"(ways {', '.join(w for w, *_ in E)})", E[0][0])

# junctions: arrows, exits and restrictions
bans = collections.defaultdict(set)  # (from way, via) -> {to way: kind}
for m, t in rels:
  if t.get("type") == "restriction":
    bans[m.get("from"), m.get("via")].add((m.get("to"), t.get("restriction")))
for n in junction:
  E = ends_at(n)
  for w, i, l in E:
    if not lanes_in(w, l)[0] or not real_end(w, i):  # (a way's turn arrows are for its own end)
      continue
    h_in = -away(w, i, l)
    exits = []
    for w2, i2, l2 in E:
      if lanes_in(w2, l2)[1]:
        a = 180.0 if (w2, l2) == (w, l) else turn(h_in, away(w2, i2, l2))
        exits.append((w2, a))
    banned = {to for to, kind in bans[w, n] if kind.startswith("no_")}
    tv = per(roads[w][1], "turn", "forward" if l else "backward")
    if not tv:
      continue
    vals = [v for lane in tv for v in (lane.split(";") if lane else ["none"])]
    for v in sorted(set(vals) - {"none", "merge_to_left", "merge_to_right"}):
      lo, hi = TURNS.get(v, (-180, 180))
      ok = [w2 for w2, a in exits if lo <= a <= hi]
      if not ok:
        flag("turns", nodes[n], f"way {w}: a lane turns {v} but no road leaves that way ({', '.join(f'{w2} {a:+.0f}°' for w2, a in exits)})", w)
      elif all(w2 in banned for w2 in ok):
        flag("turns", nodes[n], f"way {w}: a lane turns {v} into a banned move ({', '.join(ok)})", w)
    if "none" not in vals:
      for w2, a in exits:
        if w2 in banned or a == 180.0:
          continue
        if not any(TURNS.get(v, (-180, 180))[0] <= a <= TURNS.get(v, (-180, 180))[1] for v in vals):
          flag("turns", nodes[n], f"way {w}: way {w2} leaves at {a:+.0f}° unbanned, but no lane's arrow points that way", w)

# restrictions are the move they name
for m, t in rels:
  if t.get("type") != "restriction":
    continue
  fw, via, tw, kind = m.get("from"), m.get("via"), m.get("to"), t.get("restriction", "")
  p = nodes.get(via)
  if fw not in roads or tw not in roads or p is None:
    flag("restriction", p if p is not None else np.zeros(2), f"{kind}: a member is missing ({fw} -> {via} -> {tw})")
    continue
  E = ends_at(via)
  fe = [(w, i, l) for w, i, l in E if w == fw and lanes_in(w, l)[0]]
  # (a U-turn leaves back along the half it arrived by; any other move by another)
  te = [(w, i, l) for w, i, l in E if w == tw and lanes_in(w, l)[1] and ((w, l) == fe[0][::2]) == (fw == tw)] if fe else []
  if not fe:
    flag("restriction", p, f"{kind}: from way {fw} doesn't arrive at its via node", fw)
    continue
  if not te:
    flag("restriction", p, f"{kind}: to way {tw} can't be driven away from its via node", fw)
    continue
  a = 180.0 if fw == tw else turn(-away(*fe[0]), away(*te[0]))
  lo, hi = TURNS.get(BANS.get(kind, ""), (-180, 180))
  if not lo <= a <= hi:
    flag("restriction", p, f"{kind}: way {fw} -> {tw} turns {a:+.0f}°, not that move", fw)

# out
feats = []
for check, p, why, w in flags:
  lat, lon = gframe.lat_lon(np.array([p[0]]), np.array([p[1]]))
  feats.append({"type": "Feature", "properties": {"kind": "osm_check", "check": check, "why": why, "way": w},
                "geometry": {"type": "Point", "coordinates": [round(float(lon[0]), 8), round(float(lat[0]), 8)]}})
json.dump({"type": "FeatureCollection", "features": feats}, open(out, "w"))
print(len(roads), "road ways,", len(junction), "junction nodes,", sum(t.get("type") == "restriction" for _, t in rels), "restrictions:",
      len(flags), "findings", dict(collections.Counter(c for c, *_ in flags)))
for check, p, why, w in flags:
  print(f" {check:11s} ({p[0]:.0f}, {p[1]:.0f}) {why}")
