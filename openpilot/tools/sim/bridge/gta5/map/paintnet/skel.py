"""skel.py: thinning a 0/1 image to its 1 px skeleton, and its pixel paths between ends and forks."""
import numpy as np

NB = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]


def thin(img):
  """Zhang-Suen thinning of a 0/1 image."""
  im = np.pad(img.astype(np.uint8), 1)
  changed = True
  while changed:
    changed = False
    for step in (0, 1):
      P = [im[1:-1, 1:-1]]
      n = [im[:-2, 1:-1], im[:-2, 2:], im[1:-1, 2:], im[2:, 2:], im[2:, 1:-1], im[2:, :-2], im[1:-1, :-2], im[:-2, :-2]]
      B = sum(n)
      seq = n + [n[0]]
      A = sum(((seq[k] == 0) & (seq[k + 1] == 1)).astype(np.uint8) for k in range(8))
      p2, p4, p6, p8 = n[0], n[2], n[4], n[6]
      if step == 0:
        c = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
      else:
        c = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
      rm = (P[0] == 1) & (B >= 2) & (B <= 6) & (A == 1) & c
      if rm.any():
        im[1:-1, 1:-1][rm] = 0
        changed = True
  return im[1:-1, 1:-1]


def degree(sk):
  """each skeleton pixel's count of skeleton neighbours."""
  p = np.pad(sk.astype(np.uint8), 1)
  H, W = sk.shape
  return sum(p[1 + dr:1 + dr + H, 1 + dc:1 + dc + W] for dr, dc in NB) * (sk > 0)


def walk(sk, start, n):
  """up to n pixels along the skeleton from an end pixel, as (row, col) from the end inwards."""
  H, W = sk.shape
  path, prev = [start], None
  cur = start
  for _ in range(n):
    nxt = [(cur[0] + dr, cur[1] + dc) for dr, dc in NB
           if 0 <= cur[0] + dr < H and 0 <= cur[1] + dc < W and sk[cur[0] + dr, cur[1] + dc] and (cur[0] + dr, cur[1] + dc) != prev
           and (cur[0] + dr, cur[1] + dc) not in path]
    if len(nxt) != 1:
      break
    prev, cur = cur, nxt[0]
    path.append(cur)
  return path


def paths(sk):
  """the skeleton's pixel paths between ends and forks, as lists of (row, col)."""
  pts = set(zip(*np.nonzero(sk)))
  deg = {p: sum((p[0] + dr, p[1] + dc) in pts for dr, dc in NB) for p in pts}
  nodes = {p for p, d in deg.items() if d != 2}
  seen, out = set(), []
  for s in nodes:
    for dr, dc in NB:
      q = (s[0] + dr, s[1] + dc)
      if q not in pts or (s, q) in seen:
        continue
      path, prev, cur = [s], s, q
      while True:
        seen.add((prev, cur)); seen.add((cur, prev))
        path.append(cur)
        if cur in nodes:
          break
        nxt = [(cur[0] + dr2, cur[1] + dc2) for dr2, dc2 in NB if (cur[0] + dr2, cur[1] + dc2) in pts and (cur[0] + dr2, cur[1] + dc2) != prev]
        nxt = [p for p in nxt if (cur, p) not in seen]
        if not nxt:
          break
        prev, cur = cur, nxt[0]
      out.append(path)
  if not nodes and pts:  # a loop
    out.append(sorted(pts))
  return out, nodes, deg
