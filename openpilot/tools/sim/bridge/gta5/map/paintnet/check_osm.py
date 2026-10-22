"""validity and routability of an osm.py output: ids, refs, repeats, and which way ends reach which by car."""
import sys, collections, xml.etree.ElementTree as ET
r = ET.parse(sys.argv[1]).getroot()
nodes = {int(n.get("id")): n for n in r.iter("node")}
ways = [(int(w.get("id")), [int(x.get("ref")) for x in w.iter("nd")], {t.get("k"): t.get("v") for t in w.iter("tag")}) for w in r.iter("way")]
bad = [(i, nd) for i, nd, _ in ways if any(x not in nodes for x in nd) or len(nd) < 2 or any(a == b for a, b in zip(nd, nd[1:]))]
print(len(nodes), "nodes", len(ways), "ways; bad:", bad, "; ids all negative:", all(i < 0 for i in nodes) and all(i < 0 for i, *_ in ways))
used = collections.Counter(x for _, nd, _ in ways for x in nd)
print("unused nodes:", sum(1 for i in nodes if i not in used))
adj = collections.defaultdict(set)
for i, nd, tg in ways:
  if not tg.get("highway", "").startswith(("motorway", "primary", "residential", "road")):
    continue
  for a, b in zip(nd, nd[1:]):
    adj[a].add(b)
    if tg.get("oneway") != "yes":
      adj[b].add(a)
ends = {}
for i, nd, tg in ways:
  if tg.get("highway") in (None, "footway"):
    continue
  for x in (nd[0], nd[-1]):
    if sum(1 for _, n2, t2 in ways if x in n2 and t2.get("highway") != "footway") == 1:
      ends[x] = i
def reach(s):
  seen, st = {s}, [s]
  while st:
    for b in adj[st.pop()]:
      if b not in seen:
        seen.add(b); st.append(b)
  return seen
for x, i in sorted(ends.items()):
  R = reach(x)
  n = nodes[x]
  print(f" end node {x} of way {i} ({float(n.get('lat')):.6f},{float(n.get('lon')):.6f}) reaches {sum(1 for y in ends if y in R and y != x)} other ends")
