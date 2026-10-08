"""Routing traps: directed road that turn restrictions leave with no way out, or no way in.

A router drives the map's directed ways (way id, 1 along it or -1 against), turning only where no restriction
forbids it, and back along the way it is on only at a dead end. A restriction through ways forbids only that whole
sequence, so a car's state is its way and the restrictions it is part way along. Road the restrictions cut off from the
rest of the map that GTA's own links join to it is a trap: a car there can't be routed out, or to it.
"""
import math
from collections import defaultdict


def _tarjan(n, succ):
  """Strongly connected components of states 0..n-1: [component of each state]."""
  index, low, comp = [-1] * n, [0] * n, [-1] * n
  on, stack, c, i = [False] * n, [], 0, 0
  for s in range(n):
    if index[s] >= 0:
      continue
    work = [(s, 0)]
    index[s] = low[s] = i
    i += 1
    stack.append(s)
    on[s] = True
    while work:
      v, k = work[-1]
      if k < len(succ[v]):
        work[-1] = (v, k + 1)
        w = succ[v][k]
        if index[w] < 0:
          index[w] = low[w] = i
          i += 1
          stack.append(w)
          on[w] = True
          work.append((w, 0))
        elif on[w]:
          low[v] = min(low[v], index[w])
        continue
      work.pop()
      if work:
        low[work[-1][0]] = min(low[work[-1][0]], low[v])
      if low[v] == index[v]:
        while True:
          w = stack.pop()
          on[w] = False
          comp[w] = c
          if w == v:
            break
        c += 1
  return comp


def _reach(seeds, nxt):
  seen = set(seeds)
  stack = list(seeds)
  while stack:
    for t in nxt[stack.pop()]:
      if t not in seen:
        seen.add(t)
        stack.append(t)
  return seen


class TurnGraph:
  """The states a router drives through, from [(way id, a, b, two_way)] and [(restriction, from way, via, to way)]
  with via [('n', node)] or [('w', way id), ...]: `edge[s]`, the directed way of each state; `succ[s]`, the states on;
  `banned[s]`, the moves on that restrictions forbid, as [(directed way, the state's restrictions on it, the indices
  of the restrictions forbidding it)]."""
  def __init__(self, ways, restrictions):
    self.ends, out, near = {}, defaultdict(list), defaultdict(set)
    for wid, a, b, two_way in ways:
      for e, p, q in ((wid, 1), a, b), ((wid, -1), b, a):
        if e[1] == 1 or two_way:
          self.ends[e] = (p, q)
          out[p].append(e)
      near[a].add(b)
      near[b].add(a)
    self.dead_end = {n for n, ms in near.items() if len(ms) == 1}
    at_node, through = defaultdict(list), defaultdict(list)  # (from way, node) -> [(to way, i)]; from way -> [i]
    seqs = {}
    for i, (kind, wf, via, wt) in enumerate(restrictions):
      if not kind.startswith('no_'):
        continue
      if via[0][0] == 'n':
        at_node[(wf, via[0][1])].append((wt, i))
      else:
        seqs[i] = [wf, *(ref for _, ref in via), wt]
        through[wf].append(i)
    self.edge, self.succ, self.banned, index = [], [], [], {}

    def state(e, along):
      key = (e, along)
      if key not in index:
        index[key] = len(self.edge)
        self.edge.append(e)
        self.succ.append([])
        self.banned.append([])
        todo.append(key)
      return index[key]

    todo = []
    for e in sorted(self.ends):
      state(e, ())
    while todo:
      e, along = todo.pop()
      s = index[(e, along)]
      v = self.ends[e][1]
      started = [(i, 0) for i in through[e[0]]] + list(along)
      for e2 in out[v]:
        if e2[0] == e[0] and v not in self.dead_end:
          continue
        why = [i for wt, i in at_node.get((e[0], v), ()) if wt == e2[0] and e2[0] != e[0]]
        on = []
        for i, k in started:
          if seqs[i][k + 1] == e2[0]:
            if k + 2 == len(seqs[i]):
              why.append(i)
            else:
              on.append((i, k + 1))
        on = tuple(sorted(on))
        if why:
          self.banned[s].append((e2, on, why))
        else:
          self.succ[s].append(state(e2, on))
    self.index = index


def cut_off(ways, restrictions):
  """The directed ways (way id, 1 / -1) cut off: (trapped, a car on them can't get back to the rest of the map;
  unreachable, a car can't get on to their way either way from it). Only road GTA's links join to the rest of the map
  counts: the largest part a car can drive round without restrictions."""
  g = TurnGraph(ways, restrictions)
  return _cut(g, _joined(g))[:2]


def _joined(g):
  """The directed ways in the largest part of the map a car can drive round, restrictions aside."""
  nodes = sorted({n for p, q in g.ends.values() for n in (p, q)})
  at = {n: k for k, n in enumerate(nodes)}
  succ = [[] for _ in nodes]
  for p, q in g.ends.values():
    succ[at[p]].append(at[q])
  comp = _tarjan(len(nodes), succ)
  sizes = defaultdict(int)
  for c in comp:
    sizes[c] += 1
  main = max(sizes, key=lambda c: sizes[c])
  return {e for e, (p, q) in g.ends.items() if comp[at[p]] == main and comp[at[q]] == main}


def _cut(g, joined):
  n = len(g.edge)
  comp = _tarjan(n, g.succ)
  sizes = defaultdict(int)
  for s in range(n):
    if g.edge[s] in joined:
      sizes[comp[s]] += 1
  main = max(sizes, key=lambda c: sizes[c])
  core = [s for s in range(n) if comp[s] == main]
  pred = [[] for _ in range(n)]
  for s, ts in enumerate(g.succ):
    for t in ts:
      pred[t].append(s)
  ahead, back = _reach(core, g.succ), _reach(core, pred)
  entered = {g.edge[s] for s in ahead}
  # a car on a way, come whichever way, can't get back: part way along a restriction it may have nowhere to go, but
  # then no route takes it there
  trapped = {e for e in joined & entered if g.index[(e, ())] not in back}
  # a destination on a two-way way is reached either way along it
  reached = {e[0] for e in entered}
  unreachable = {e for e in joined - entered if e[0] not in reached}
  return trapped, unreachable, comp, ahead, back


def _turned(nodes, ends_of, wf, via, wt):
  """deg a restriction's move turns, summed along it."""
  seq = [wf, *(ref for t, ref in via if t == 'w'), wt]
  joints = [via[0][1]] if via[0][0] == 'n' else [next(iter(set(ends_of[x]) & set(ends_of[y])), None) for x, y in zip(seq, seq[1:])]
  total, h = 0.0, None
  for k, n in enumerate(joints):
    if n is None:
      return 180.0
    a, b = ends_of[seq[k]]
    p = a if b == n else b  # the way in, from its far end to the joint
    h_in = math.degrees(math.atan2(nodes[n]['x'] - nodes[p]['x'], nodes[n]['y'] - nodes[p]['y']))
    a, b = ends_of[seq[k + 1]]
    q = b if a == n else a
    h_out = math.degrees(math.atan2(nodes[q]['x'] - nodes[n]['x'], nodes[q]['y'] - nodes[n]['y']))
    if h is not None:
      total += (h_in - h + 180) % 360 - 180
    total += (h_out - h_in + 180) % 360 - 180
    h = h_out
  return abs(total)




def free_traps(nodes, ways, restrictions, max_via=5):
  """Restrictions changed as little as it takes for none to cut road off (see `cut_off`), the move turning least first,
  U-turn bans before GTA's turn flags. Out of a trap, a forbidden move is forbidden from each way on to the trap's way
  instead: a car on it may leave, but no route goes in to make the move (left out where that would take more than
  `max_via` via ways). In to unreachable road, it is left out. Returns (the restrictions, those left out, those
  added)."""
  ends_of = {wid: (a, b) for wid, a, b, _ in ways}
  into = defaultdict(list)  # node -> ways a car arrives at it along
  for wid, a, b, two_way in ways:
    into[b].append(wid)
    if two_way:
      into[a].append(wid)
  turned = {}

  def cost(r):
    kind, wf, via, wt = r
    key = (wf, tuple(via), wt)
    if key not in turned:
      turned[key] = _turned(nodes, ends_of, wf, via, wt)
    return (kind != 'no_u_turn', turned[key])

  def earlier(r):  # the restriction from each way on to its from way instead
    kind, wf, via, wt = r
    if (1 if via[0][0] == 'n' else len(via) + 1) > max_via:
      return None
    joint = via[0][1] if via[0][0] == 'n' else next(iter(set(ends_of[wf]) & set(ends_of[via[0][1]])))
    start = ends_of[wf][0] if ends_of[wf][1] == joint else ends_of[wf][1]
    rest = [] if via[0][0] == 'n' else list(via)
    return [(kind, w, [('w', wf), *rest], wt) for w in sorted(into[start]) if w != wf]

  g = TurnGraph(ways, [])
  joined = _joined(g)
  own = _cut(g, joined)[:2]  # GTA's own: no restriction to free
  current = list(restrictions)
  while True:
    g = TurnGraph(ways, current)
    trapped, unreachable, comp, ahead, back = _cut(g, joined)
    trapped, unreachable = trapped - own[0], unreachable - own[1]
    if not trapped and not unreachable:
      break
    # one forbidden move per area of cut-off road a round, as one often frees it all: out of a trap to road that gets
    # back, or in to unreachable road from road a car gets to, before any other
    area = {e: e for e in sorted(trapped | unreachable)}

    def find(e):
      while area[e] != e:
        area[e] = area[area[e]]
        e = area[e]
      return e
    by_node = defaultdict(list)
    for e in area:
      for n in g.ends[e]:
        by_node[n].append(e)
    for es in by_node.values():
      for e in es[1:]:
        area[find(e)] = find(es[0])
    best = {}

    def offer(e, useful, why, out_of_trap):
      key = (not useful, max(cost(current[i]) for i in why), sorted(why), out_of_trap)
      a = find(e)
      if a not in best or key < best[a]:
        best[a] = key

    starts = defaultdict(list)
    for e in trapped:
      starts[find(e)].append(g.index[(e, ())])
    for a, ss in starts.items():  # the moves forbidden on from its traps
      for s in _reach(ss, g.succ):
        for e2, on, why in g.banned[s]:
          offer(a, g.index.get((e2, on), g.index[(e2, ())]) in back, why, True)
    for s in range(len(g.edge)):
      for e2, on, why in g.banned[s]:
        if e2 in unreachable:
          offer(e2, s in ahead, why, False)
    if not best:
      break  # what's left is GTA's own, no restriction's
    changed, added = set(), []
    for *_, why, out_of_trap in best.values():
      for i in why:
        if i not in changed:
          changed.add(i)
          added += (earlier(current[i]) or []) if out_of_trap else []
    current = [r for i, r in enumerate(current) if i not in changed] + added
  keys = {(k, wf, tuple(via), wt) for k, wf, via, wt in current}
  was = {(k, wf, tuple(via), wt) for k, wf, via, wt in restrictions}
  return (current, [r for r in restrictions if (r[0], r[1], tuple(r[2]), r[3]) not in keys],
          [r for r in current if (r[0], r[1], tuple(r[2]), r[3]) not in was])
